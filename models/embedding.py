#word embeddings

import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)
