import sys
sys.path.append(r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1")
from tokenizers import Tokenizer
import os
import torch

def tokenize_and_save(src_lines, tgt_lines, tokenizer_en, tokenizer_hi, src_filename, tgt_filename, max_len=100):
    # Tokenize the source and target lines
    src_tokenized = [tokenizer_en.encode(line).ids for line in src_lines]
    tgt_tokenized = [tokenizer_hi.encode(line).ids for line in tgt_lines]

    # Padding sequences
    src_tokenized = pad_sequences(src_tokenized, max_len)
    tgt_tokenized = pad_sequences(tgt_tokenized, max_len)

    # Convert tokenized sequences to tensors
    src_tensor = torch.tensor(src_tokenized, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_tokenized, dtype=torch.long)

    # Save the tensors
    torch.save(src_tensor, src_filename)
    torch.save(tgt_tensor, tgt_filename)

def pad_sequences(sequences, max_len, pad_token=0):
    """Pads sequences to a maximum length."""
    return [seq + [pad_token] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

# Load your tokenizers using absolute paths
tokenizer_en = Tokenizer.from_file(r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1/DATA/tokenizer/bpe_tokenizer_en.model")
tokenizer_hi = Tokenizer.from_file(r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1/DATA/tokenizer/bpe_tokenizer_hi.model")

# Define paths for raw and processed data
raw_data_path = r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1/DATA/raw_data/en-hi"
processed_data_path = r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1/DATA/processed_data"

# Ensure the processed data folder exists
os.makedirs(processed_data_path, exist_ok=True)

# Load raw data (train files for example)
with open(os.path.join(raw_data_path, "IITB.en-hi.en"), "r", encoding="utf-8") as f:
    src_lines = f.readlines()

with open(os.path.join(raw_data_path, "IITB.en-hi.hi"), "r", encoding="utf-8") as f:
    tgt_lines = f.readlines()

# Tokenize and save as tensors
tokenize_and_save(src_lines, tgt_lines, tokenizer_en, tokenizer_hi, 
                  os.path.join(processed_data_path, "train_src.pt"), 
                  os.path.join(processed_data_path, "train_tgt.pt"))

print("Data preprocessing completed and saved!")
