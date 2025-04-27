import tensorflow as tf
import numpy as np
from model import GPT2

# Constants
VOCAB_SIZE = 50257
MAX_LENGTH = 1024

# Dummy data for now
x_train = np.random.randint(0, VOCAB_SIZE, size=(10, MAX_LENGTH))
y_train = np.random.randint(0, VOCAB_SIZE, size=(10, MAX_LENGTH))

# Create model
gpt2 = GPT2(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)
gpt2.build(input_shape=(None, MAX_LENGTH))

# Compile
gpt2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
gpt2.fit(
    x=x_train,
    y=y_train,
    batch_size=2,
    epochs=3
)

# Save
gpt2.save('gpt2_trained_model.h5')
print("✅ Model trained and saved as 'gpt2_trained_model.h5'.")
