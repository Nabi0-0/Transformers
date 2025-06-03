#for testing mult head attention mechanism


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.multi_head_attention import MultiHeadAttention

def test_multihead_attention():
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    attention = MultiHeadAttention(d_model, num_heads)

    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)  

    try:
        output = attention(x, x, x, mask)
        assert output.shape == (batch_size, seq_len, d_model), f"Unexpected shape: {output.shape}"
        print("✅ MultiHeadAttention test passed!")
    except Exception as e:
        print(f"❌ Error in MultiHeadAttention: {e}")

if __name__ == "__main__":
    test_multihead_attention()
