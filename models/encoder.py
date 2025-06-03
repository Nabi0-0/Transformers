import torch.nn as nn
from models.multi_head_attention import MultiHeadAttention
from models.feed_forward import FeedForward
from models.residual_layer_norm import ResidualConnection

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = ResidualConnection(d_model, dropout)
        self.norm2 = ResidualConnection(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """Apply self-attention + residual connection, then feedforward + residual connection"""
        x = self.norm1(x, lambda x: self.self_attn(x, x, x, mask))  #  Pass x to self-attention
        x = self.norm2(x, self.ffn)  #  Directly pass ffn
        return x
