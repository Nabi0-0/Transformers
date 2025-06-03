#positional encodeing logic

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model) filled with zeros
        pe = torch.zeros(max_len, d_model)

        # Generate position indices (shape: [max_len, 1])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute denominator: 10000^(2i/d_model) (shape: [1, d_model//2])
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra batch dimension (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as a buffer so it's not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input tensor (assume shape [batch, seq_len, d_model])
        return x + self.pe[:, :x.size(1), :]

