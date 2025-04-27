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

# 🚀 GPT-2 Mini Project Documentation

## How to Use

### 1. **Train the Model**
First, train the GPT-2 model on your own dataset. This will create the model weights and save them as `.h5`. To train the model, run:

```bash
python train.py
```

This will start the training process and save the trained model as `gpt2_trained_model.h5`.

### 2. **Generate Text**
After training, use `App.py` to generate text. You'll be prompted to enter a text, and the model will generate a continuation of that text.

```bash
python App.py
```

You will be asked for a prompt like:

```
Enter prompt: Once upon a time
```

Example output:

```
Generated Text: Once upon a time, in a land far, far away, there lived a young prince...
```

## 🧑‍💻 Example Usage
* **Prompt**: `"Once upon a time"`
* **Generated Text**: `"Once upon a time, there was a land filled with magic and wonder. The people lived in harmony, and the kingdom was ruled by a wise and just king..."`

## ⚙️ How It Works
* The core model is built around the Transformer architecture with **Multi-Head Self-Attention** and **Feed-Forward Networks**.
* We use **Layer Normalization** and **Dropout** for regularization and to avoid overfitting.
* The GPT-2 model is based on **causal attention** (i.e., it only attends to previous tokens in the sequence, ensuring it's autoregressive).
* **Training**: We use the standard language modeling loss, which is the **categorical cross-entropy** between the predicted token and the actual token in the sequence.
* The model is trained on text data and learns to generate continuations of the input text.

## 🧠 Model Parameters
* `d_model`: Dimensionality of the embedding space (e.g., 256, 512).
* `num_heads`: The number of attention heads in the multi-head self-attention layer.
* `num_layers`: Number of transformer blocks (layers).
* `max_length`: Maximum length of the input sequence (usually between 512-1024 tokens).
* `dff`: The size of the feed-forward layer inside each transformer block.

These parameters can be adjusted for different trade-offs between performance and resource requirements.

## 🔧 Customizing the Model
You can modify the model architecture and hyperparameters to suit your needs:
* **Embedding dimensions**: Change the `embed_dim` parameter in the model for a larger or smaller embedding space.
* **Layers and Attention heads**: Modify `num_layers` and `num_heads` for a deeper or shallower model, depending on your requirements.
* **Training data**: Use your own dataset for fine-tuning the model.

## 📝 Notes
* This is a **mini GPT-2** model, and **not as powerful as large GPT-2 models**. It's a simplified version to learn the architecture and experiment with.
* Due to the reduced model size, the quality of text generation will not be as high as the larger GPT-2 or GPT-3 models.
* The **training** process can be slow, especially if you're using a CPU. Using a **GPU** will speed up the process significantly.
* The **model file** (`gpt2_trained_model.h5`) is saved after training and can be reloaded using Keras to generate text.

## 🧑‍💻 Credits
This project was created by **[Your Name Here]** with TensorFlow and Keras for educational purposes.

Feel free to modify, extend, and improve this code as per your needs. Contributions are welcome!

## 🐞 Issues
If you encounter any bugs or have suggestions for improvement, feel free to open an **issue** on GitHub.
