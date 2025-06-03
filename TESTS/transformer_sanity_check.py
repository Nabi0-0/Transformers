import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.Transformer import Transformer
from UTILS.mask import create_src_mask, create_trg_mask
from UTILS.loss_and_optimizer import LabelSmoothingCrossEntropy, get_optimizer
from tokenizers import Tokenizer

# Tokenizers
tokenizer_src = Tokenizer.from_file('Transformer-translation-1/DATA/tokenizer/bpe_tokenizer_en.model')
tokenizer_tgt = Tokenizer.from_file('Transformer-translation-1/DATA/tokenizer/bpe_tokenizer_hi.model')


# --- Example Sentences ---
src_sentence = "Hello, how are you?"
trg_sentence = "नमस्ते, आप कैसे हैं?"

# --- Tokenize ---
src_tokens = tokenizer_src.encode(src_sentence).ids
trg_tokens = tokenizer_tgt.encode(trg_sentence).ids

# --- Convert to tensors ---
src_tensor = torch.tensor([src_tokens], dtype=torch.long)  # (1, src_len)
trg_tensor = torch.tensor([trg_tokens], dtype=torch.long)  # (1, trg_len)

# --- Padding ---
pad_token_id = 0
pad_len = 15

if src_tensor.size(1) < pad_len:
    pad_amt = pad_len - src_tensor.size(1)
    src_tensor = torch.cat([src_tensor, torch.full((1, pad_amt), pad_token_id, dtype=torch.long)], dim=1)

if trg_tensor.size(1) < pad_len:
    pad_amt = pad_len - trg_tensor.size(1)
    trg_tensor = torch.cat([trg_tensor, torch.full((1, pad_amt), pad_token_id, dtype=torch.long)], dim=1)

# --- Create Masks ---
src_mask = create_src_mask(src_tensor, pad_token_id)
trg_mask = create_trg_mask(trg_tensor, pad_token_id)

# --- Model Setup ---
input_vocab_size = 10000
target_vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1

transformer = Transformer(input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)

# --- Loss and Optimizer ---
criterion = LabelSmoothingCrossEntropy(label_smoothing=0.1, vocab_size=target_vocab_size, ignore_index=pad_token_id)
optimizer = get_optimizer(transformer, lr=3e-4, weight_decay=0.01)

# --- Forward Pass + Backprop ---
output = transformer(src_tensor, trg_tensor, src_mask, trg_mask)  # (1, trg_len, vocab_size)
loss = criterion(output, trg_tensor)

loss.backward()
optimizer.step()
optimizer.zero_grad()

print("✅ Sanity Check Passed | Loss:", loss.item())
