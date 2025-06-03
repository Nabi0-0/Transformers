#for testing word_embeddings

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from models.embedding import WordEmbedding


# define parameters
vocab_size = 10000 #example vocab size
d_model = 512  #example vocab size

#initialize embedding layer
embedding_layer = WordEmbedding(vocab_size, d_model)

#create a sample input (batch of tokenized words)
sample_input = torch.tensor([[1,5,20], [7,2, 99]])  # shape : (batch_size = 2, seq_length = 3)

#get embeddings
output = embedding_layer(sample_input)

print("input: ", sample_input)
print("output shape: ", output.shape) #expected: (2, 3, 512)

