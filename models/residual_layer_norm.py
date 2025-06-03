import torch
import torch.nn as nn

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # Shape: (512,)
        self.beta = nn.Parameter(torch.zeros(d_model))  # Shape: (512,)
        self.eps = eps  # Small constant for numerical stability

    def forward(self, x):
        """Apply Layer Normalization"""
        x = x.float()  # 🔥 Ensure float dtype to avoid mean() dtype errors
        mean = x.mean(dim=-1, keepdim=True)  # Shape: (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # Shape: (batch_size, seq_len, 1)

        # Debug prints to check shapes
        print(f"x shape: {x.shape}")
        print(f"mean shape: {mean.shape}")
        print(f"std shape: {std.shape}")

        # 🔥 Correct Broadcasting
        gamma = self.gamma.view(1, 1, -1)  # Shape: (1, 1, d_model)
        beta = self.beta.view(1, 1, -1)    # Shape: (1, 1, d_model)

        print(f"gamma shape: {gamma.shape}")
        print(f"beta shape: {beta.shape}")

        return gamma * (x - mean) / (std + self.eps) + beta

# Residual Connection with Layer Norm
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection: LayerNorm → Sublayer → Dropout → Residual Connection"""
        return x + self.dropout(sublayer(self.norm(x)))