#for testing layer normalization and residual connection

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from models.residual_layer_norm import LayerNorm, ResidualConnection

def test_layer_norm():
    batch_size = 2
    seq_length = 5
    d_model = 512

    x = torch.randn(batch_size, seq_length, d_model)
    ln = LayerNorm(d_model)
    output = ln(x)

    assert output.shape == (batch_size, seq_length, d_model), "Output shape is incorrect!"
    print("✅ Layer Normalization Test Passed!")

def test_residual_connection():
    batch_size = 2
    seq_length = 5
    d_model = 512

    x = torch.randn(batch_size, seq_length, d_model)
    residual = ResidualConnection(d_model)

    # Dummy sublayer: Just an identity function
    sublayer = lambda x: x * 0.5  

    output = residual(x, sublayer)

    assert output.shape == (batch_size, seq_length, d_model), "Output shape is incorrect!"
    print("✅ Residual Connection Test Passed!")

if __name__ == "__main__":
    test_layer_norm()
    test_residual_connection()
