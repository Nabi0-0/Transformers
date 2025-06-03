import os
from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers import pre_tokenizers

# Define the function to train the tokenizer from scratch
def train_bpe_tokenizer(language, input_file, output_model_file, vocab_size=10000, min_frequency=2):
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_model_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the BPE tokenizer
    tokenizer = Tokenizer(BPE())

    # Pre-tokenizer (splits on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Initialize the trainer for BPE
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, show_progress=True)

    # Read input raw data file (e.g., train.en or train.hi)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    # Train the tokenizer on the raw text data
    tokenizer.train_from_iterator(data, trainer=trainer)

    # Save the trained tokenizer to a .model file
    tokenizer.save(output_model_file)
    print(f"Tokenizer saved to {output_model_file}")

# Define paths to your raw training data and output files
train_en_file = r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1/DATA/raw_data/en-hi/IITB.en-hi.en"  # Path to the English training data
train_hi_file = r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1/DATA/raw_data/en-hi/IITB.en-hi.hi"  # Path to the Hindi training data

output_en_model = r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1/DATA/tokenizer/bpe_tokenizer_en.model"
output_hi_model = r"C:/Users/vedan/Projects/Transformer-translation/Transformer-translation-1/DATA/tokenizer/bpe_tokenizer_hi.model"

# Train tokenizer for English and Hindi data
train_bpe_tokenizer("en", train_en_file, output_en_model)
train_bpe_tokenizer("hi", train_hi_file, output_hi_model)
