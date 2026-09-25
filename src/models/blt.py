import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .norm import LayerNorm

class LocalTransformerLayer(nn.Module):
    """Small pre-normalized Transformer block used inside byte patches."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """Build one local attention and feed-forward block."""
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mix representations within a local byte patch."""
        h = self.norm1(x)
        x = x + self.drop1(self.attn(h, h, h, mask=mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x



# converts group of 4 bytes (i.e. 4 bytes = 1 token) into one embedding vector of size d_model(256)
# group of embeddings of 4 bytes [v1,v2,v3,v4] ==> combine valid positions ==> single patch representation ==> project to 256 dimensions
class LocalByteEncoder(nn.Module):
    """
    BLT-style local encoder.

    Raw bytes are grouped into 4-byte patches. A lightweight custom local
    Transformer mixes the bytes *inside each patch*, then masked pooling turns
    the four byte states into one continuous patch representation for the
    global Transformer.
    """

    def __init__(
        self,
        vocab_size: int = 260,
        byte_dim: int = 128,
        patch_size: int = 4,
        d_model: int = 256,
        local_heads: int = 4,
        local_layers: int = 1,
        dropout: float = 0.1,
    ):
        """Set up byte embeddings, local layers, and the patch projection."""
        super().__init__()
        if byte_dim % local_heads != 0:
            raise ValueError("byte_dim must be divisible by local_heads")
        self.patch_size = int(patch_size)
        self.byte_embed = nn.Embedding(vocab_size, byte_dim, padding_idx=0)
        self.local_pos = nn.Parameter(torch.zeros(1, 1, self.patch_size, byte_dim))
        self.local_layers = nn.ModuleList(
            [
                LocalTransformerLayer(
                    d_model=byte_dim,
                    num_heads=local_heads,
                    d_ff=byte_dim * 2,
                    dropout=dropout,
                )
                for _ in range(local_layers)
            ]
        )
        self.proj = nn.Linear(byte_dim, d_model)
        self.norm = LayerNorm(d_model)
        nn.init.normal_(self.local_pos, std=0.02)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Encode raw byte ids into one continuous vector per local patch."""
        batch, length = byte_ids.shape
        pad_len = (-length) % self.patch_size
        if pad_len:
            byte_ids = F.pad(byte_ids, (0, pad_len), value=0)
        num_patches = byte_ids.size(1) // self.patch_size

        # Keep local attention inside each patch rather than across the full byte sequence.
        patch_ids = byte_ids.view(batch, num_patches, self.patch_size)
        valid = patch_ids.ne(0)
        x = self.byte_embed(patch_ids) + self.local_pos
        x = x.view(batch * num_patches, self.patch_size, -1)
        local_key_mask = valid.view(batch * num_patches, 1, 1, self.patch_size)

        for layer in self.local_layers:
            x = layer(x, mask=local_key_mask)

        x = x.view(batch, num_patches, self.patch_size, -1)
        # Pool only real bytes so padding at the end of a chunk does not affect a patch.
        weights = valid.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return self.norm(self.proj(pooled))


# global decoder patch hidden state (represent 1 target patch) ==> LocalByteDecoder ==> 4 byte predictions(logit scores over byte vocabulary)
# byte vocabulary = 256 possible raw bytes + special tokens(PAD,BOS,EOS,UNK)
class LocalByteDecoder(nn.Module):
    """Decode each global patch latent into four byte logits using local queries."""

    def __init__(
        self,
        vocab_size: int = 260,
        d_model: int = 256,
        patch_size: int = 4,
        byte_dim: int = 128,
        local_heads: int = 4,
        local_layers: int = 1,
        dropout: float = 0.1,
    ):
        """Set up patch-conditioned byte queries and the byte prediction head."""
        super().__init__()
        self.patch_size = int(patch_size)
        self.vocab_size = int(vocab_size)
        self.patch_proj = nn.Linear(d_model, byte_dim)
        self.output_queries = nn.Parameter(torch.zeros(1, 1, self.patch_size, byte_dim))
        self.local_layers = nn.ModuleList(
            [
                LocalTransformerLayer(
                    d_model=byte_dim,
                    num_heads=local_heads,
                    d_ff=byte_dim * 2,
                    dropout=dropout,
                )
                for _ in range(local_layers)
            ]
        )
        self.norm = LayerNorm(byte_dim)
        self.head = nn.Linear(byte_dim, vocab_size)
        nn.init.normal_(self.output_queries, std=0.02)

    def forward(self, patch_latents: torch.Tensor, target_byte_len: int) -> torch.Tensor:
        """Expand patch latents back into byte-level logits."""
        batch, num_patches, _ = patch_latents.shape
        # A learned query for each byte position expands one patch state back to four outputs.
        base = self.patch_proj(patch_latents).unsqueeze(2)
        x = base + self.output_queries
        x = x.view(batch * num_patches, self.patch_size, -1)
        for layer in self.local_layers:
            x = layer(x, mask=None)
        x = self.norm(x)
        logits = self.head(x)
        logits = logits.view(batch, num_patches * self.patch_size, self.vocab_size)
        return logits[:, :target_byte_len]
