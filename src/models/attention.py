import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention and return both context and weights."""
    
    d_k = q.size(-1)    # head dimension
    # Scaling (divising by sqrt(d_k)) so the dot product values do not become too large before softmax, which helps maintain stable gradients
    # Calculate similarity using (Q*K)/sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Padding mask + Causal mask(used in decoder)
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask != 0
        # forbidden positions are replaced with a huge negative number so that the model cannot attend to the masked position
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    # applying softmax to get the attention weights that sum to 1
    attn_weights = torch.softmax(scores, dim=-1)
    
    # applying dropout to the attention weights for regularization
    if dropout is not None:
        # during training, some attention connections are randomly dropped, reducing over-reliance on specific relationships
        attn_weights = dropout(attn_weights)

    # compute the context vector(Attention) as a weighted sum of the value vectors
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# instead of performing one attention operation using all 256 (d_k) dimensions, we split the representation into multiple heads (8 heads)
# Different heads can learn different relationships (local nearby patterns , long-distance dependencies,etc)
class MultiHeadAttention(nn.Module):
    """Custom multi-head attention used by the encoder and decoder."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """Create the query, key, value, and output projections for MHA."""
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model  # 256
        self.num_heads = num_heads  # 8
        self.head_dim = d_model // num_heads # 32

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply multi-head attention to the supplied query, key, and value states."""
        batch_size, tgt_len, _ = q_in.shape
        src_len = k_in.size(1)

        # matrix multiplication with the layer weights(q_in × Wᵀ) + Split the projected states into independent attention heads (num_heads = 8 & head_dim = 32)
        q = self.q_proj(q_in).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Rotary Positional Embedding
        if rope is not None:
            from .positional import apply_rope
            # extracts pre calculated rotation values (Cosine and Sine matrices)
            cos, sin = rope
            # Physically rotates the Query (q) and Key (k) vectors right before they are multiplied together to calculate attention scores
            q, k = apply_rope(q, k, cos, sin)

        context, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        # Combine the heads
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        return self.out_proj(context)


# Q1,Q2,Q3,Q4 → share K1,V1 & Q5,Q6,Q7,Q8 → share K2,V2
# goal : to reduce K/V parameters and computation/memory while keeping multiple query heads
class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with fewer key/value heads than query heads."""
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
    ):
        """Create grouped-query attention projections and head grouping."""
        super().__init__()
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")

        self.d_model = d_model
        self.num_query_heads = num_query_heads  # 8
        self.num_kv_heads = num_kv_heads        # 2
        self.num_groups = num_query_heads // num_kv_heads   # 8/2 = 4
        self.head_dim = d_model // num_query_heads          # 256/8  = 32

        # Q still produces all(num_query_heads) heads BUT K & V produce fewer heads(num_kv_heads), which are then repeated to match the number of query heads
        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply grouped-query attention, repeating key/value heads by group."""
        batch_size, tgt_len, _ = q_in.shape
        src_len = k_in.size(1)

        q = self.q_proj(q_in).view(batch_size, tgt_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Rotary Positional Embedding
        if rope is not None:
            from .positional import apply_rope
            # extracts pre calculated rotation values (Cosine and Sine matrices)
            cos, sin = rope
            q, k = apply_rope(q, k, cos, sin)

        # Each key/value head is shared by a fixed group of query heads
        # repeats each K/V head
        if self.num_groups > 1:
            k = torch.repeat_interleave(k, repeats=self.num_groups, dim=1)
            v = torch.repeat_interleave(v, repeats=self.num_groups, dim=1)

        context, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        # Combine the heads
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        return self.out_proj(context)
