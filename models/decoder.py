import torch.nn as nn
from models.multi_head_attention import MultiHeadAttention
from models.feed_forward import FeedForward
from models.residual_layer_norm import ResidualConnection

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = ResidualConnection(d_model, dropout)
        self.norm2 = ResidualConnection(d_model, dropout)
        self.norm3 = ResidualConnection(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, trg_mask):
        x = self.norm1(x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.norm2(x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))
        x = self.norm3(x, self.ffn)
        return x