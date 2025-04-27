from model import GPT2
from transformers import GPT2Tokenizer
import numpy as np

# Create model
model = GPT2(
    vocab_size=50257,
    max_length=1024
)

# Dummy call to build the model
dummy_input = np.zeros((1, 1), dtype=np.int32)
model(dummy_input)

# Now load weights
model.load_weights('gpt2_trained_model.h5')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text
def generate_text(prompt, model, tokenizer, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt, return_tensors='np')

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > 1024:
            input_ids = input_ids[:, -1024:]
        outputs = model.predict(input_ids, verbose=0)
        next_token_logits = outputs[:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1)
        input_ids = np.concatenate([input_ids, next_token_id[:, None]], axis=-1)

    return tokenizer.decode(input_ids[0])

# Example usage
prompt = input("Enter a Prompt: ")
generated_text = generate_text(prompt, model, tokenizer, max_new_tokens=100)
print(generated_text)
