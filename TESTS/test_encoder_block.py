#for testing encoder

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.encoder import EncoderLayer

def test_encoder():
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)

    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, 1, 1, seq_len)

    try:
        output = encoder_layer(x, mask)
        assert output.shape == (batch_size, seq_len, d_model), f"Unexpected shape: {output.shape}"
        print("✅ Encoder test passed!")
    except Exception as e:
        print(f"❌ Error in Encoder: {e}")

if __name__ == "__main__":
    test_encoder()
