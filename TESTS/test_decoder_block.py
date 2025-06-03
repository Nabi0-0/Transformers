#for testing decoder block

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.decoder import DecoderLayer

def test_decoder():
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

    x = torch.rand(batch_size, seq_len, d_model)
    enc_output = torch.rand(batch_size, seq_len, d_model)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    trg_mask = torch.ones(batch_size, 1, seq_len, seq_len)

    try:
        output = decoder_layer(x, enc_output, src_mask, trg_mask)
        assert output.shape == (batch_size, seq_len, d_model), f"Unexpected shape: {output.shape}"
        print("✅ Decoder test passed!")
    except Exception as e:
        print(f"❌ Error in Decoder: {e}")

if __name__ == "__main__":
    test_decoder()
