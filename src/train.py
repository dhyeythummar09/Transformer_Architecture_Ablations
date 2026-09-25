import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataloaders
from src.models.attention import GroupedQueryAttention, MultiHeadAttention
from src.models.blt import LocalByteDecoder, LocalByteEncoder
from src.models.norm import LayerNorm, RMSNorm
from src.models.positional import RotaryPositionalEmbedding, SinusoidalPositionalEncoding

# Configuration Details
# Only the component named by each ablation changes from the C1 baseline.
CONFIG_PROFILES = {
    "C1": {
        "description": "Sinusoidal + MHA + LayerNorm + BPE",
        "pos_encoding": "sinusoidal",
        "attn_type": "mha",
        "norm_type": "layernorm",
        "num_kv_heads": None,
        "is_token_free": False,
    },
    "C2": {
        "description": "RoPE + MHA + LayerNorm + BPE",
        "pos_encoding": "rope",
        "attn_type": "mha",
        "norm_type": "layernorm",
        "num_kv_heads": None,
        "is_token_free": False,
    },
    "C3": {
        "description": "Sinusoidal + GQA + LayerNorm + BPE",
        "pos_encoding": "sinusoidal",
        "attn_type": "gqa",
        "norm_type": "layernorm",
        "num_kv_heads": 2,
        "is_token_free": False,
    },
    "C4": {
        "description": "Sinusoidal + MHA + RMSNorm + BPE",
        "pos_encoding": "sinusoidal",
        "attn_type": "mha",
        "norm_type": "rmsnorm",
        "num_kv_heads": None,
        "is_token_free": False,
    },
    "C5": {
        "description": "BLT token-free: raw bytes + 4-byte local patches",
        "pos_encoding": "sinusoidal",
        "attn_type": "mha",
        "norm_type": "layernorm",
        "num_kv_heads": None,
        "is_token_free": True,
    },
}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_for_rtx() -> None:
    """Enable CUDA settings that improve throughput on recent NVIDIA GPUs."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device() -> torch.device:
    """Choose CUDA when available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_gpu_memory_mb() -> float:
    """Return the current peak CUDA memory allocation in megabytes."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def get_norm_layer(norm_type: str, d_model: int) -> nn.Module:
    """Create the normalization layer requested by the configuration."""
    if norm_type.lower() == "rmsnorm":
        return RMSNorm(d_model)
    return LayerNorm(d_model)


def get_attention_layer(
    attn_type: str,
    d_model: int,
    num_heads: int,
    num_kv_heads: Optional[int],
    dropout: float,
) -> nn.Module:
    """Create either MHA or GQA with the configuration's head settings."""
    if attn_type.lower() == "gqa":
        kv_heads = num_kv_heads if num_kv_heads is not None else max(1, num_heads // 4)
        return GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_heads,
            num_kv_heads=kv_heads,
            dropout=dropout,
        )
    return MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)


# FFN transforms information inside each token representation
class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward network used in each Transformer block."""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """Build the linear-GELU-dropout feed-forward stack."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   
            nn.GELU(),      # GELU is a smooth nonlinear activation
            nn.Dropout(dropout),    # dropout is a regularization technique that randomly sets some activations to zero during training
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network independently to each token."""
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-normalized Transformer encoder layer."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_type: str,
        norm_type: str,
        num_kv_heads: Optional[int],
        dropout: float,
    ):
        """Create encoder self-attention, feed-forward, normalization, and dropout layers."""
        super().__init__()
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.attn = get_attention_layer(attn_type, d_model, num_heads, num_kv_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, rope=None):
        """Run one pre-normalized encoder block."""
        h = self.norm1(x)
        x = x + self.drop1(self.attn(h, h, h, mask=mask, rope=rope))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """Pre-normalized decoder layer with self-attention and cross-attention."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_type: str,
        norm_type: str,
        num_kv_heads: Optional[int],
        dropout: float,
    ):
        """Create decoder attention, cross-attention, feed-forward, and normalization layers."""
        super().__init__()
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.self_attn = get_attention_layer(attn_type, d_model, num_heads, num_kv_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.cross_attn = get_attention_layer(attn_type, d_model, num_heads, num_kv_heads, dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm3 = get_norm_layer(norm_type, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, enc, tgt_mask=None, src_mask=None, rope_tgt=None):
        """Run masked self-attention, encoder cross-attention, and the feed-forward block."""
        h = self.norm1(tgt)
        tgt = tgt + self.drop1(self.self_attn(h, h, h, mask=tgt_mask, rope=rope_tgt))
        h = self.norm2(tgt)
        tgt = tgt + self.drop2(self.cross_attn(h, enc, enc, mask=src_mask, rope=None))
        tgt = tgt + self.drop3(self.ff(self.norm3(tgt)))
        return tgt


class ConfigurableSeq2SeqTransformer(nn.Module):
    """Encoder-decoder Transformer shared by all five assignment configurations."""
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = 2,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
        pos_encoding: str = "sinusoidal",
        attn_type: str = "mha",
        norm_type: str = "layernorm",
        is_token_free: bool = False,
    ):
        """Build the selected positional, attention, normalization, and tokenization variant."""
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.pos_encoding = pos_encoding.lower()
        self.is_token_free = bool(is_token_free)
        self.patch_size = 4 if self.is_token_free else 1

        # C5 works on local byte patches; C1-C4 use normal token embeddings.
        if self.is_token_free:
            self.src_embed = LocalByteEncoder(      # LocalByteEncoder : custom embedding layer that encodes raw byte sequences into a higher-dimensional representation suitable for the Transformer model.
                vocab_size=src_vocab_size,
                d_model=d_model,
                patch_size=4,
                dropout=dropout,
            )
            self.tgt_embed = LocalByteEncoder(  
                vocab_size=tgt_vocab_size,
                d_model=d_model,
                patch_size=4,
                dropout=dropout,
            )
            self.lm_head = LocalByteDecoder(
                vocab_size=tgt_vocab_size,
                d_model=d_model,
                patch_size=4,
                dropout=dropout,
            )
        else:
            self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
            self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
            self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)

        if self.pos_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
            self.pos_decoder = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
            self.rope = None
        # RoPE is applied inside attention instead of being added to the embeddings.
        elif self.pos_encoding == "rope":
            self.pos_encoder = None
            self.pos_decoder = None
            self.rope = RotaryPositionalEmbedding(d_model // num_heads, max_len=max_len)
        else:
            raise ValueError(f"Unknown positional encoding {pos_encoding}")

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    attn_type,
                    norm_type,
                    num_kv_heads,
                    dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder_norm = get_norm_layer(norm_type, d_model)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    attn_type,
                    norm_type,
                    num_kv_heads,
                    dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = get_norm_layer(norm_type, d_model)
        self._reset_parameters()

        if not self.is_token_free:
            # Weight tying : target embedding matrix and output vocabulary projection share the same weights
            self.lm_head.weight = self.tgt_embed.weight

    def _reset_parameters(self):
        """Initialize matrix parameters with Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def _downsample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert a byte-level mask to the patch-level mask used by BLT."""
        length = mask.size(-1)
        extra = (-length) % self.patch_size
        if extra:
            mask = F.pad(mask, (0, extra), value=False)
        return mask.view(*mask.shape[:-1], -1, self.patch_size).any(dim=-1)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """Encode source tokens or byte patches into contextual states."""
        if self.is_token_free:
            x = self.src_embed(src)
            if src_mask is not None:
                src_mask = self._downsample_mask(src_mask)
        else:
            x = self.src_embed(src) * self.scale

        rope = None
        if self.pos_encoding == "sinusoidal":
            x = self.pos_encoder(x)
        else:
            rope = self.rope(x, x.size(1))

        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask, rope=rope)
        return self.encoder_norm(x)

    def decode(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """Decode target history while attending to the encoded source."""
        if self.is_token_free:
            y = self.tgt_embed(tgt)
            patch_len = y.size(1)
            tgt_mask = torch.tril(
                torch.ones(patch_len, patch_len, dtype=torch.bool, device=y.device)
            ).unsqueeze(0).unsqueeze(1)
            if src_mask is not None:
                src_mask = self._downsample_mask(src_mask)
        else:
            y = self.tgt_embed(tgt) * self.scale

        rope = None
        if self.pos_encoding == "sinusoidal":
            y = self.pos_decoder(y)
        else:
            rope = self.rope(y, y.size(1))

        for layer in self.decoder_layers:
            y = layer(y, enc_output, tgt_mask=tgt_mask, src_mask=src_mask, rope_tgt=rope)
        return self.decoder_norm(y)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Run the complete encoder-decoder path and return output logits."""
        enc = self.encode(src, src_mask)
        dec = self.decode(tgt, enc, tgt_mask=tgt_mask, src_mask=src_mask)
        if self.is_token_free:
            return self.lm_head(dec, target_byte_len=tgt.size(1))
        return self.lm_head(dec)


class WarmupCosineScheduler:
    """Linear warmup followed by cosine learning-rate decay."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ):
        """Store scheduler bounds and initialize the optimizer learning rate."""
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.min_lr = self.peak_lr * float(min_lr_ratio)
        self.last_step = 0
        self.set_for_step(0)

    def lr_for_step(self, step: int) -> float:
        """Compute the learning rate assigned to one optimizer step."""
        step = int(step)
        if step < self.warmup_steps:
            return self.peak_lr * float(step + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return self.min_lr
        # After warmup, decay smoothly to the configured minimum learning rate.
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def set_for_step(self, step: int) -> float:
        """Apply the scheduled learning rate to every optimizer parameter group."""
        lr = self.lr_for_step(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_step = int(step)
        return lr

    def state_dict(self):
        """Return the scheduler settings needed for checkpoint metadata."""
        return {"last_step": self.last_step}



def build_optimizer(model: nn.Module, lr: float, weight_decay: float, device: torch.device):
    """Build AdamW, using the fused CUDA implementation when available."""
    kwargs = dict(
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except Exception:
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def get_amp_settings(device: torch.device, amp_dtype: str):
    """Resolve whether AMP is enabled and which floating-point dtype to use."""
    if device.type != "cuda" or amp_dtype == "none":
        return False, None, None
    if amp_dtype == "bf16":
        return True, torch.bfloat16, None
    return True, torch.float16, "fp16"


def autocast_context(device: torch.device, enabled: bool, dtype):
    """Return the appropriate autocast context for the current device."""
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_grad_scaler(device: torch.device, scaler_kind: Optional[str]):
    """Create a gradient scaler when FP16 CUDA training needs one."""
    if device.type == "cuda" and scaler_kind == "fp16":
        try:
            return torch.amp.GradScaler("cuda")
        except Exception:
            return torch.cuda.amp.GradScaler()
    return None


@torch.inference_mode()
def greedy_decode(
    model: ConfigurableSeq2SeqTransformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype=None,
) -> torch.Tensor:
    """Generate target ids autoregressively using greedy decoding."""
    model.eval()
    with autocast_context(device, amp_enabled, amp_dtype):
        enc = model.encode(src, src_mask)

        # Tokenized models emit one token at a time; BLT uses a separate patch-wise branch below.
        if not model.is_token_free:
            generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)
            for _ in range(max_len):
                length = generated.size(1)
                tgt_mask = torch.tril(
                    torch.ones(length, length, dtype=torch.bool, device=device)
                ).unsqueeze(0).unsqueeze(1)
                dec = model.decode(generated, enc, tgt_mask=tgt_mask, src_mask=src_mask)
                logits = model.lm_head(dec)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
                generated = torch.cat([generated, next_token], dim=1)
                finished |= next_token.squeeze(1).eq(eos_id)
                if finished.all():
                    break
            return generated

        patch = model.patch_size
        decoder_input = torch.full(
            (src.size(0), patch),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        decoder_input[:, 0] = bos_id
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)
        output_patches = []

        for _ in range(math.ceil(max_len / patch)):
            patch_len = decoder_input.size(1) // patch
            tgt_mask = torch.tril(
                torch.ones(patch_len, patch_len, dtype=torch.bool, device=device)
            ).unsqueeze(0).unsqueeze(1)
            dec = model.decode(decoder_input, enc, tgt_mask=tgt_mask, src_mask=src_mask)
            logits = model.lm_head(dec, target_byte_len=decoder_input.size(1))
            next_patch = logits[:, -patch:].argmax(dim=-1)

            for row in range(next_patch.size(0)):
                if finished[row]:
                    next_patch[row].fill_(eos_id)
                    continue
                eos_positions = (next_patch[row] == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first = int(eos_positions[0].item())
                    if first + 1 < patch:
                        next_patch[row, first + 1:] = eos_id
                    finished[row] = True

            output_patches.append(next_patch)
            decoder_input = torch.cat([decoder_input, next_patch], dim=1)
            if finished.all():
                break

        output = (
            torch.cat(output_patches, dim=1)[:, :max_len]
            if output_patches
            else torch.empty((src.size(0), 0), dtype=torch.long, device=device)
        )
        bos = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=device)
        return torch.cat([bos, output], dim=1)


def token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float,
    reduction: str = "mean",
):
    """Compute token-level cross entropy while ignoring padding positions."""
    if logits.size(1) != targets.size(1):
        length = min(logits.size(1), targets.size(1))
        logits = logits[:, :length]
        targets = targets[:, :length]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


@torch.inference_mode()
def validate(
    model,
    loader,
    device,
    tgt_pad_id: int,
    label_smoothing: float,
    amp_enabled: bool,
    amp_dtype,
):
    """Evaluate validation loss and teacher-forced token accuracy."""
    model.eval()
    loss_sum = 0.0
    token_count = 0
    correct = 0

    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        src_mask = batch["src_mask"].to(device, non_blocking=True)
        tgt_input = batch["tgt_input"].to(device, non_blocking=True)
        tgt_output = batch["tgt_output"].to(device, non_blocking=True)
        tgt_mask = batch["tgt_mask"].to(device, non_blocking=True)
        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            if logits.size(1) != tgt_output.size(1):
                length = min(logits.size(1), tgt_output.size(1))
                logits = logits[:, :length]
                tgt_output = tgt_output[:, :length]
            batch_loss = token_cross_entropy(
                logits,
                tgt_output,
                tgt_pad_id,
                label_smoothing,
                reduction="sum",
            )

        valid = tgt_output.ne(tgt_pad_id)
        count = int(valid.sum().item())
        loss_sum += float(batch_loss.item())
        token_count += count
        predictions = logits.argmax(dim=-1)
        correct += int(((predictions == tgt_output) & valid).sum().item())

    return loss_sum / max(1, token_count), correct / max(1, token_count)


@torch.inference_mode()
def source_shuffle_diagnostic(
    model,
    loader,
    device,
    tgt_pad_id,
    label_smoothing,
    amp_enabled,
    amp_dtype,
    max_batches: int = 20,
):
    """Measure how validation loss changes when source examples are shuffled."""
    model.eval()
    normal_sum = 0.0
    shuffled_sum = 0.0
    tokens = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)
        if src.size(0) < 2:
            continue
        # Break source-target alignment while keeping the same batch shapes and masks.
        perm = torch.roll(torch.arange(src.size(0), device=device), 1)
        with autocast_context(device, amp_enabled, amp_dtype):
            normal_logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            shuffled_logits = model(
                src[perm],
                tgt_input,
                src_mask=src_mask[perm],
                tgt_mask=tgt_mask,
            )
            normal = token_cross_entropy(
                normal_logits, tgt_output, tgt_pad_id, label_smoothing, reduction="sum"
            )
            shuffled = token_cross_entropy(
                shuffled_logits, tgt_output, tgt_pad_id, label_smoothing, reduction="sum"
            )
        count = int(tgt_output.ne(tgt_pad_id).sum().item())
        normal_sum += float(normal.item())
        shuffled_sum += float(shuffled.item())
        tokens += count
    if tokens == 0:
        return None
    normal_loss = normal_sum / tokens
    shuffled_loss = shuffled_sum / tokens
    return {
        "normal_val_loss": normal_loss,
        "shuffled_source_val_loss": shuffled_loss,
        "shuffle_penalty": shuffled_loss - normal_loss,
    }


def build_model(config_name: str, src_vocab_size: int, tgt_vocab_size: int, args):
    """Construct the Transformer for a named C1-C5 configuration."""
    cfg = CONFIG_PROFILES[config_name]
    return ConfigurableSeq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.n_heads,
        num_kv_heads=cfg["num_kv_heads"],
        num_encoder_layers=args.n_layers,
        num_decoder_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_seq_len,
        pos_encoding=cfg["pos_encoding"],
        attn_type=cfg["attn_type"],
        norm_type=cfg["norm_type"],
        is_token_free=cfg["is_token_free"],
    )


def maybe_compile(model, enabled: bool):
    """Optionally wrap the model with torch.compile when requested."""
    if not enabled or not hasattr(torch, "compile"):
        return model
    print("torch.compile enabled. First steps may be slower due to compilation.")
    return torch.compile(model)


def save_checkpoint(
    path: str,
    raw_model,
    optimizer,
    scheduler,
    step: int,
    val_loss: float,
    val_acc: float,
    args,
    cfg,
    data_meta,
):
    """Save model weights together with configuration and tokenizer metadata."""
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": int(step),
        "best_val_loss": float(val_loss),
        "best_val_token_accuracy": float(val_acc),
        "config_name": args.config_name,
        "config_profile": cfg,
        "model_config": {
            "src_vocab_size": int(data_meta["src_vocab_size"]),
            "tgt_vocab_size": int(data_meta["tgt_vocab_size"]),
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "max_seq_len": args.max_seq_len,
        },
        "training_config": {
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "betas": [0.9, 0.98],
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "total_steps": args.total_steps,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "clip_grad": args.clip_grad,
            "amp_dtype": args.amp_dtype,
            "seed": args.seed,
        },
        "data_config": {
            "cipher_path": args.cipher_path,
            "plain_path": args.plain_path,
            "chunk_chars": args.chunk_chars,
            "stride_chars": args.stride_chars,
            "cipher_tokenizer_path": args.cipher_tokenizer_path,
            "tokenizer_path": args.tokenizer_path,
            "src_vocab_size": args.src_vocab_size,
            "vocab_size": args.vocab_size,
            "tokenizer_train_chunks": args.tokenizer_train_chunks,
            "generation_max_len": data_meta["generation_max_len"],
            "evaluation_unit": "independent_64_character_chunks",
        },
    }
    torch.save(checkpoint, path)


def train_configuration(args):
    """Train one configuration, evaluate it periodically, and save its best checkpoint."""
    set_seed(args.seed)
    configure_cuda_for_rtx()
    device = get_device()
    cfg = CONFIG_PROFILES[args.config_name]
    print(f"--> Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else device}")
    print(f"--> {args.config_name}: {cfg['description']}")
    print(
        f"Chunking: {args.chunk_chars} ASCII chars -> {args.chunk_chars*8} cipher bits | "
        f"stride={args.stride_chars} chars | chunks are independent samples"
    )

    train_loader, val_loader, _, src_tok, tgt_tok, data_meta = get_dataloaders(
        cipher_path=args.cipher_path,
        plain_path=args.plain_path,
        is_token_free=cfg["is_token_free"],
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        chunk_bytes=args.chunk_chars,
        stride_bytes=args.stride_chars,
        max_seq_len=args.max_seq_len,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.vocab_size,
        tokenizer_train_chunks=args.tokenizer_train_chunks,
        seed=args.seed,
        rebuild_tokenizers=args.rebuild_tokenizers,
        cipher_tokenizer_path=args.cipher_tokenizer_path,
        plain_tokenizer_path=args.tokenizer_path,
    )
    print(
        f"Data: train={data_meta['train_chunks']} chunks | val={data_meta['val_chunks']} chunks | "
        f"src_vocab={data_meta['src_vocab_size']} | tgt_vocab={data_meta['tgt_vocab_size']}"
    )

    raw_model = build_model(
        args.config_name,
        data_meta["src_vocab_size"],
        data_meta["tgt_vocab_size"],
        args,
    ).to(device)
    model = maybe_compile(raw_model, args.compile)
    # Keep the optimizer and schedule identical across configurations for the ablation.
    optimizer = build_optimizer(raw_model, args.lr, args.weight_decay, device)
    scheduler = WarmupCosineScheduler(
        optimizer,
        peak_lr=args.lr,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    amp_enabled, amp_dtype, scaler_kind = get_amp_settings(device, args.amp_dtype)
    scaler = build_grad_scaler(device, scaler_kind)
    print(
        f"Optimizer: AdamW betas=(0.9,0.98), wd={args.weight_decay} | "
        f"lr={args.lr:g} | warmup={args.warmup_steps} | total={args.total_steps} | "
        f"min_lr={args.lr*args.min_lr_ratio:g} | label_smoothing={args.label_smoothing}"
    )
    print(
        f"Batch={args.batch_size} | grad_accum={args.grad_accum_steps} | "
        f"AMP={args.amp_dtype} | clip_grad={args.clip_grad}"
    )

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"Run_{args.config_name}_64char",
            config={**vars(args), **cfg, **data_meta},
        )

    os.makedirs("outputs", exist_ok=True)
    best_path = os.path.join("outputs", f"best_{args.config_name.lower()}_model.pt")
    best_val = float("inf")
    best_val_acc = 0.0
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_updates = 0
    train_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    last_log_time = started

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    while global_step < args.total_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        raw_model.train()
        src = batch["src"].to(device, non_blocking=True)
        src_mask = batch["src_mask"].to(device, non_blocking=True)
        tgt_input = batch["tgt_input"].to(device, non_blocking=True)
        tgt_output = batch["tgt_output"].to(device, non_blocking=True)
        tgt_mask = batch["tgt_mask"].to(device, non_blocking=True)

        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = token_cross_entropy(
                logits,
                tgt_output,
                tgt_tok.pad_token_id,
                args.label_smoothing,
                reduction="mean",
            )
            scaled_loss = loss / args.grad_accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        micro_step += 1

        if micro_step % args.grad_accum_steps != 0:
            continue

        current_lr = scheduler.set_for_step(global_step)
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip_grad)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        running_loss += float(loss.detach().item())
        running_updates += 1

        if global_step % args.log_interval == 0 or global_step == 1:
            now = time.time()
            avg_loss = running_loss / max(1, running_updates)
            steps_per_second = args.log_interval / max(now - last_log_time, 1e-9) if global_step > 1 else 0.0
            print(
                f"step {global_step:05d}/{args.total_steps} | loss {avg_loss:.4f} | "
                f"lr {current_lr:.2e} | grad {float(grad_norm):.3f} | "
                f"{steps_per_second:.1f} step/s | GPU {get_gpu_memory_mb():.0f} MB"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/grad_norm": float(grad_norm),
                        "train/gpu_memory_mb": get_gpu_memory_mb(),
                        "train/steps_per_second": steps_per_second,
                    },
                    step=global_step,
                )
            running_loss = 0.0
            running_updates = 0
            last_log_time = now

        # Evaluate periodically and keep the checkpoint with the lowest validation loss.
        should_eval = global_step % args.eval_interval == 0 or global_step == args.total_steps
        if should_eval:
            val_start = time.time()
            val_loss, val_acc = validate(
                raw_model,
                val_loader,
                device,
                tgt_tok.pad_token_id,
                args.label_smoothing,
                amp_enabled,
                amp_dtype,
            )
            print(
                f"  EVAL step {global_step}: val_loss={val_loss:.4f} | "
                f"teacher-forced token acc={100*val_acc:.2f}% | {time.time()-val_start:.1f}s"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"val/loss": val_loss, "val/token_accuracy": val_acc},
                    step=global_step,
                )

            if val_loss < best_val:
                best_val = val_loss
                best_val_acc = val_acc
                save_checkpoint(
                    best_path,
                    raw_model,
                    optimizer,
                    scheduler,
                    global_step,
                    val_loss,
                    val_acc,
                    args,
                    cfg,
                    data_meta,
                )
                print(f"  ✓ saved best checkpoint: {best_path}")

            sample_batch = next(iter(val_loader))
            sample_src = sample_batch["src"][:1].to(device)
            sample_mask = sample_batch["src_mask"][:1].to(device)
            pred_ids = greedy_decode(
                raw_model,
                sample_src,
                sample_mask,
                data_meta["generation_max_len"],
                tgt_tok.bos_token_id,
                tgt_tok.eos_token_id,
                tgt_tok.pad_token_id,
                device,
                amp_enabled,
                amp_dtype,
            )
            pred_text = tgt_tok.decode(pred_ids[0].tolist(), skip_special_tokens=True)
            print(f"  target: {sample_batch['target_texts'][0]!r}")
            print(f"  pred  : {pred_text!r}")

    elapsed = time.time() - started
    print(
        f"Training finished in {elapsed/60:.1f} min | best val loss={best_val:.4f} | "
        f"best token acc={100*best_val_acc:.2f}%"
    )

    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model_state_dict"])

    if args.source_shuffle_diagnostic:
        diag = source_shuffle_diagnostic(
            raw_model,
            val_loader,
            device,
            tgt_tok.pad_token_id,
            args.label_smoothing,
            amp_enabled,
            amp_dtype,
        )
        if diag:
            print(
                "Source shuffle | "
                f"normal={diag['normal_val_loss']:.4f} | "
                f"shuffled={diag['shuffled_source_val_loss']:.4f} | "
                f"penalty={diag['shuffle_penalty']:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log({f"diagnostic/{k}": v for k, v in diag.items()})

    if wandb_run is not None:
        wandb_run.finish()


def parse_args():
    """Parse command-line options for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", choices=list(CONFIG_PROFILES), default="C1")
    parser.add_argument("--cipher_path", default="data/brown_cipher.txt")
    parser.add_argument("--plain_path", default="data/brown_plain.txt")
    parser.add_argument("--cipher_tokenizer_path", default="outputs/cipher_tokenizer.json")
    parser.add_argument("--tokenizer_path", default="outputs/tokenizer.json")

    parser.add_argument("--src_vocab_size", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--chunk_chars", type=int, default=64)
    parser.add_argument("--stride_chars", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--tokenizer_train_chunks", type=int, default=4000)
    parser.add_argument("--rebuild_tokenizers", action="store_true")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--total_steps", type=int, default=25000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--amp_dtype", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_shuffle_diagnostic", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="advanced-nlp-assignment1")
    return parser.parse_args()


if __name__ == "__main__":
    train_configuration(parse_args())
