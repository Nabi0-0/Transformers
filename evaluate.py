#evaluation script
# This script evaluates a trained Transformer model on a validation dataset using BLEU and METEOR scores.
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm

from UTILS.Data_loader import TranslationDataset
from torch.utils.data import DataLoader
from models.transformer import Transformer  # Adjust if needed
import nltk
nltk.download('wordnet')

# Paths to your data and model
val_src_path = "DATA/processed_data/valid_src.pt"
val_tgt_path = "DATA/processed_data/valid_tgt.pt"
model_path = "transformer_model.pth"

# Hyperparameters (must match training)
input_vocab_size = 20000
target_vocab_size = 20000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# DataLoader
val_dataset = TranslationDataset(val_src_path, val_tgt_path)
val_loader = DataLoader(val_dataset, batch_size=1)  # 1 for sentence-level BLEU/METEOR

# Evaluation loop
bleu_scores = []
meteor_scores = []

for src, tgt in tqdm(val_loader, desc="Evaluating"):
    src, tgt = src.to(device), tgt.to(device)

    # Generate mask if needed
    trg_input = tgt[:, :-1]  # Remove last token for input
    trg_mask = torch.triu(torch.ones((trg_input.size(1), trg_input.size(1))), diagonal=1).bool().to(device)

    # Model prediction
    with torch.no_grad():
        output = model(src, trg_input, None, trg_mask)
    pred_tokens = output.argmax(-1).squeeze().tolist()
    ref_tokens = tgt.squeeze().tolist()[1:]  # remove <sos> token

    # Clean special tokens (e.g., padding)
    pred_clean = [str(tok) for tok in pred_tokens if tok != 0]
    ref_clean = [str(tok) for tok in ref_tokens if tok != 0]

    # Compute scores
    bleu = sentence_bleu([ref_clean], pred_clean, smoothing_function=SmoothingFunction().method4)
    meteor = meteor_score([' '.join(ref_clean)], ' '.join(pred_clean))

    bleu_scores.append(bleu)
    meteor_scores.append(meteor)

# Final results
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_meteor = sum(meteor_scores) / len(meteor_scores)

print(f"Average BLEU score: {avg_bleu:.4f}")
print(f"Average METEOR score: {avg_meteor:.4f}")
