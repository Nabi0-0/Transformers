#for testing postional encoding

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.positional_encoding import PositionalEncoding

def test_positional_encoding():
    d_model = 512
    max_len = 10
    batch_size = 2

    pos_encoding = PositionalEncoding(d_model, max_len)

    x = torch.rand(batch_size, max_len, d_model)

    try:
        output = pos_encoding(x)
        assert output.shape == (batch_size, max_len, d_model), f"Unexpected shape: {output.shape}"
        print("✅ PositionalEncoding test passed!")
    except Exception as e:
        print(f"❌ Error in PositionalEncoding: {e}")

if __name__ == "__main__":
    test_positional_encoding()
