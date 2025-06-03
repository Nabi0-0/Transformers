import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from UTILS.Data_loader import TranslationDataset
from models.Transformer import Transformer
from tqdm import tqdm
import os

input_vocab_size = 20000
target_vocab_size = 20000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1
batch_size = 64
learning_rate = 1e-4
num_epochs = 5
SAVE_EVERY = 100  # Save every N batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = Transformer(input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout).to(device)
print("Model on GPU:", next(model.parameters()).is_cuda)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz), device=device)) == 1
    return mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, float(0.0))

train_loader = DataLoader(
    TranslationDataset(
        '/kaggle/input/nmt-transformer/Transformer-translation/Transformer-translation-1/DATA/processed_data/train_src.pt',
        '/kaggle/input/nmt-transformer/Transformer-translation/Transformer-translation-1/DATA/processed_data/train_tgt.pt'
    ),
    batch_size=batch_size,
    shuffle=True
)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

checkpoint_dir = "/content/drive/MyDrive/transformer_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
        for step, (src_batch, tgt_batch) in enumerate(tepoch):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            trg_mask = generate_square_subsequent_mask(tgt_batch.size(1)).to(device)
            optimizer.zero_grad()
            output = model(src_batch, tgt_batch, None, trg_mask)
            output = output.view(-1, output.size(-1))
            tgt_batch = tgt_batch.view(-1)
            loss = criterion(output, tgt_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            if step % SAVE_EVERY == 0:
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"{checkpoint_dir}/checkpoint_epoch{epoch+1}_step{step}.pt")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"{checkpoint_dir}/checkpoint_epoch{epoch+1}_final.pt")

torch.save(model.state_dict(), f"{checkpoint_dir}/transformer_model_final.pth")
print("Training complete! Final model saved.")
