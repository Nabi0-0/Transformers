import torch
from models.Transformer import Transformer
from UTILS.tokenizer import load_vocab, tokenize_sentence, detokenize_sentence  # Replace with your actual functions
import torch.nn.functional as F

# Load vocabularies
src_vocab, src_idx2word = load_vocab('DATA/vocab/src_vocab.pkl')
tgt_vocab, tgt_idx2word = load_vocab('DATA/vocab/tgt_vocab.pkl')

# Model parameters (must match training)
input_vocab_size = len(src_vocab)
target_vocab_size = len(tgt_vocab)
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout).to(device)
model.load_state_dict(torch.load('transformer_model.pth', map_location=device))
model.eval()

# Generate target mask (like training)
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# Inference function
def translate_sentence(sentence, max_len=50):
    tokens = tokenize_sentence(sentence, src_vocab)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)  # shape: (1, seq_len)

    src_mask = None
    memory = model.dropout(model.pos_encoding(model.src_embedding(src_tensor)))
    for layer in model.encoder_layers:
        memory = layer(memory, src_mask)

    tgt_indices = [tgt_vocab['<sos>']]  # Start token
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

        out = model.decoder_layers[0](model.dropout(model.pos_encoding(model.trg_embedding(tgt_tensor))), memory, src_mask, tgt_mask)
        for layer in model.decoder_layers[1:]:
            out = layer(out, memory, src_mask, tgt_mask)

        logits = model.fc_out(out)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        if next_token == tgt_vocab['<eos>']:
            break
        tgt_indices.append(next_token)

    return detokenize_sentence(tgt_indices[1:], tgt_idx2word)  # Skip <sos>

# Example
if __name__ == "__main__":
    while True:
        inp = input("Enter a sentence (or 'quit'): ")
        if inp.lower() == 'quit':
            break
        translation = translate_sentence(inp)
        print("Translation:", translation)
