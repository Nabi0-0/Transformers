### Transformer-Based Neural Machine Translation (NMT)

#### **Project Overview**
Hi! This is my implementation of a **Transformer-based Neural Machine Translation (NMT)** model. I built this project to better understand how Transformers work, inspired by the famous paper *"Attention Is All You Need"* by Vaswani et al. The goal of this project is to translate text from **English to Hindi** using the Transformer architecture.

---

### **What I Learned**
- How the **Transformer architecture** works, including the Encoder and Decoder.
- The importance of **self-attention** and **multi-head attention** in capturing relationships between words.
- How to preprocess data, tokenize text, and train a deep learning model for translation.
- How to evaluate translation quality using metrics like BLEU.

---

### **Technologies I Used**
- **Programming Language**: Python 3.x
- **Framework**: PyTorch
- **Libraries**:
  - NumPy
  - pandas
  - torch
  - scikit-learn
  - nltk

---

### **Dataset**
- **Source**: IITB v2.0 Dataset
- **Link**: [IITB English-Hindi Dataset](https://opus.nlpl.eu/IITB/en&hi/v2.0/IITB)
- This dataset contains parallel sentences in English and Hindi, which I used for training and evaluation.

---

### **How to Set It Up**
Here’s how you can run this project on your own system:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nabi0-0/Transformer-translation.git
   cd Transformer-translation
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**:
   - Download the IITB English-Hindi dataset.
   - Preprocess the data and update the paths in `data/config.py`.

---

### **How to Use It**
1. **Train the Model**:
   Run the following command to train the model:
   ```bash
   python train.py --epochs 10 --batch_size 64 --learning_rate 1e-4
   ```

2. **Evaluate the Model**:
   After training, you can evaluate the model using:
   ```bash
   python evaluate.py --checkpoint model_checkpoint.pth
   ```

3. **Translate a Sentence**:
   Use the trained model to translate a sentence:
   ```bash
   python translate.py --sentence "I am a student"
   ```

---

### **Project Structure**
Here’s how I organized the project:

```
Transformer-translation/
│
├── models/                # Contains the model code (Encoder, Decoder, etc.)
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
|   ......
│
├── DATA/                  # Data preprocessing and tokenization
│   ├── processes_data/
│   ├── raw_data/
│   └── tokenizer/
│
├── models/                # Contains other necessary model code 
│   ├── mask.py
│   ├── loss_and_optimizer.py
│   ├── train_tokenizer.py
│   └── other model components...
│
├── config.py              # Configuration file for model parameters
├── train_loop.py          # Script to train the model
├── evaluate.py            # Script to evaluate the model
├── translate.py           # Script for translating input text
└── README.md              # This file!
```

---

### **Acknowledgments**
- **Paper**: *Attention Is All You Need* by Vaswani et al.  
  [Read the paper](https://arxiv.org/abs/1706.03762)
- **PyTorch Documentation**: [PyTorch Docs](https://pytorch.org/docs/stable/index.html)

---

### **License**

---

### **Final Thoughts**
This project was a great learning experience for me. I got to dive deep into how Transformers work and how they can be used for tasks like translation. If you have any feedback or suggestions, feel free to share! 😊
