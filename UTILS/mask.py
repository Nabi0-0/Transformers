# utils/mask.py

import torch

def create_src_mask(src, pad_token_id):
    """
    Creates mask for source sequences.
    Masks out padding tokens.
    Shape: [B, 1, 1, src_len]
    """
    return (src != pad_token_id).unsqueeze(1).unsqueeze(2)

def create_trg_mask(trg, pad_token_id):
    """
    Creates mask for target sequences.
    Masks out padding tokens AND future tokens (look-ahead mask).
    Shape: [B, 1, trg_len, trg_len]
    """
    B, T = trg.shape

    # Padding mask: [B, 1, 1, T]
    padding_mask = (trg != pad_token_id).unsqueeze(1).unsqueeze(2)

    # Look-ahead mask: [T, T]
    look_ahead = torch.tril(torch.ones((T, T), device=trg.device)).bool()

    # Combine both
    combined_mask = padding_mask & look_ahead.unsqueeze(0).unsqueeze(1)
    return combined_mask
