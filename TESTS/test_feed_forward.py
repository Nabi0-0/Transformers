#test feedforward network 

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.feed_forward import FeedForward

def test_feed_forward():
    d_model = 512
    d_ff = 2048
    batch_size = 2
    seq_len = 10

    ff = FeedForward(d_model, d_ff)

    x = torch.rand(batch_size, seq_len, d_model)

    try:
        output = ff(x)
        assert output.shape == (batch_size, seq_len, d_model), f"Unexpected shape: {output.shape}"
        print("✅ FeedForward test passed!")
    except Exception as e:
        print(f"❌ Error in FeedForward: {e}")

if __name__ == "__main__":
    test_feed_forward()
