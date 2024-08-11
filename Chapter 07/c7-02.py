#!/usr/bin/env python
# Extracted from 'Chapter 7.ipynb'.
# coding: utf-8

# # Training Large Language Models

# ## Building a Character-level Text Generation Model

import numpy as np
import tensorflow as tf


# Set a seed for reproducibility
tf.random.set_seed(42)

# Sample input text to train the model (you can use your own text)
input_text = """
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.
"""

# Preprocess the text data
chars = sorted(set(input_text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
num_chars = len(chars)
input_seq_length = 100
step = 1

input_sequences = []
output_chars = []

for i in range(0, len(input_text) - input_seq_length, step):
    input_seq = input_text[i: i + input_seq_length]
    output_seq = input_text[i + input_seq_length]
    input_sequences.append([char_to_idx[char] for char in input_seq])
    output_chars.append(char_to_idx[output_seq])

X = np.array(input_sequences)
y = tf.keras.utils.to_categorical(output_chars, num_classes=num_chars)

# One-hot encode the input data
X_one_hot = np.zeros((X.shape[0], input_seq_length, num_chars))
for i, seq in enumerate(X):
    for j, char_idx in enumerate(seq):
        X_one_hot[i, j, char_idx] = 1

# Build the LSTM text generator model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(input_seq_length, num_chars)),
    tf.keras.layers.Dense(num_chars, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Training the model
epochs = 100
batch_size = 64

model.fit(X_one_hot, y, epochs=epochs, batch_size=batch_size)

# Generate text using the trained model
def generate_text(model, seed_text, num_chars_to_generate=100):
    generated_text = seed_text
    for _ in range(num_chars_to_generate):
        x_pred = np.zeros((1, input_seq_length, num_chars))
        for t, char in enumerate(generated_text[-input_seq_length:]):
            x_pred[0, t, char_to_idx[char]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_char_idx = np.argmax(preds)
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char
    return generated_text

# Test the text generation
seed_text = "Data science is"
generated_text = generate_text(model, seed_text, num_chars_to_generate=200)
print(generated_text)
