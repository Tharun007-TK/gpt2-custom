# 🧠 Mini GPT-2 (TensorFlow/Keras)

This project implements a lightweight version of GPT-2 built from scratch using TensorFlow/Keras.  
You can train it on your own text data, save the trained model as `.h5`, and generate new text step-by-step.

---

## 🚀 Features
- Basic GPT2 architecture with Transformer Blocks.
- Text generation from a custom prompt.
- Save and load models easily (`.h5` format).
- Simple and understandable code structure.

---

## 📦 Files
- `model.py` – Defines the custom GPT2 model, Transformer Blocks, and Attention layers.
- `train.py` – Trains the GPT2 model and saves the `.h5` weights.
- `App.py` – Loads the trained model and generates text from user prompts.
- `gpt2_trained_model.h5` – (Generated) model weights after training.

---

## 🛠 How to Use

1. **Install dependencies**  
```bash
pip install tensorflow transformers numpy
