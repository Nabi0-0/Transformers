import torch
from torch.utils.data import Dataset, DataLoader

# Create a Dataset class for loading the tokenized data
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        self.src_data = torch.load(src_file)
        self.tgt_data = torch.load(tgt_file)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

# Define file paths
train_src_file = r"/kaggle/input/nmt-transformer/Transformer-translation/Transformer-translation-1/DATA/processed_data/train_src.pt"
train_tgt_file = r"/kaggle/input/nmt-transformer/Transformer-translation/Transformer-translation-1/DATA/processed_data/train_tgt.pt"

# Create the dataset
train_dataset = TranslationDataset(train_src_file, train_tgt_file)

# Create DataLoader for batching the data
batch_size = 32  # You can change the batch size according to your available memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("DataLoader created for training.")
