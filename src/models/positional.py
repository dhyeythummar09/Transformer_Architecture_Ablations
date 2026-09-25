import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add the standard fixed sinusoidal position encoding to token states."""
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        """Precompute sinusoidal position vectors up to the requested length."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Precompute the fixed table once and reuse it for every batch
        pe = torch.zeros(max_len, d_model)  # positional encoding table
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # position indices
        # Compute the sinusoidal frequencies for each feature dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)    # even indices
        pe[:, 1::2] = torch.cos(position * div_term)    # odd indices
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add cached positional vectors to an input sequence."""
        seq_len = x.size(1)     # length of the input sequence
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds positional max_len {self.pe.size(1)}"
            )
        return self.dropout(x + self.pe[:, :seq_len].to(dtype=x.dtype))     # add positional encoding to the input tensor

# rotates Q & K vectors in the complex plane to encode position information instead of adding anything to embedding
class RotaryPositionalEmbedding(nn.Module):
    """Generate cached cosine and sine tables for rotary position embeddings."""
    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        """Initialize RoPE frequencies and build the first position cache."""
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        self.dim = dim
        # RoPE assigns a different rotation frequency to each feature pair.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        """Build cosine and sine tables for a given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, seq_len: int):
        """Return RoPE cosine and sine values for the requested sequence length."""
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:, :, :seq_len].to(device=x.device, dtype=x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension for the RoPE transform."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to query and key tensors."""
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
