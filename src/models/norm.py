import torch
import torch.nn as nn

# During training, activations can have very different scales across layers so Normalization helps keep them numerically stable

class LayerNorm(nn.Module):
    """Layer normalization implemented directly from tensor operations."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        """Create learnable scale and bias parameters for LayerNorm."""
        super().__init__()
        self.eps = eps      # small constant to avoid division by zero during normalization
        self.gamma = nn.Parameter(torch.ones(d_model))  # learnable scale parameter for each feature dimension
        self.beta = nn.Parameter(torch.zeros(d_model))  # learnable bias parameter for each feature dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # shape of x : (batch_size, seq_len, d_model)
        """Normalize each token over its feature dimension(256 features)."""
        # Normalize each token independently over its hidden features
        mean = x.mean(dim=-1, keepdim=True)     # calculate mean across the last dimension (features) for each token
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # calculate variance across the last dimension (features) for each token
        # new xi = (xi-mean)/sqrt(var+eps) * gamma + beta
        return self.gamma * (x - mean) * torch.rsqrt(var + self.eps) + self.beta    

# Root Mean Square Normalization
class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned scale."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        """Create the learned RMSNorm scale parameter."""
        super().__init__()
        self.eps = eps  #  small constant to avoid division by zero during normalization
        self.gamma = nn.Parameter(torch.ones(d_model))  # learned scale parameter for each feature dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each token by the inverse root-mean-square of its features."""
        # RMSNorm keeps the scale normalization but does not subtract the mean.
        # rms = sqrt(mean(x^2) + eps) 
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # xi_new = xi * rms_inv * gamma
        return x * rms_inv * self.gamma
