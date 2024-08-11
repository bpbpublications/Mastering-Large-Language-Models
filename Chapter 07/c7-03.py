#!/usr/bin/env python
# Extracted from 'Chapter 7.ipynb'.
# coding: utf-8

# # Training Large Language Models
# Improving model with word tokenization

import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


response = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
shakespeare_data = response.text


# Sample input text to train the model (you can use your own text)
input_text = shakespeare_data

# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([input_text])

# Vocabulary size (number of unique words in the input text)
num_words = len(tokenizer.word_index) + 1

# Convert text to sequences of word indices
input_sequences = tokenizer.texts_to_sequences([input_text])[0]
input_seq_length = 10  # The number of words in each input sequence
step = 1

# Prepare input sequences and output words
X = []
y = []
for i in range(0, len(input_sequences) - input_seq_length, step):
    input_seq = input_sequences[i : i + input_seq_length]
    output_seq = input_sequences[i + input_seq_length]
    X.append(input_seq)
    y.append(output_seq)

X = np.array(X)
y = np.array(y)

# Build the LSTM text generator model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=100, input_length=input_seq_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(num_words, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Training the model
epochs = 10
batch_size = 4096

model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Generate text using the trained model
def generate_text(model, seed_text, num_words_to_generate=50):
    generated_text = seed_text
    for _ in range(num_words_to_generate):
        input_seq = tokenizer.texts_to_sequences([generated_text])[0]
        input_seq = pad_sequences([input_seq], maxlen=input_seq_length, padding='pre')

        preds = model.predict(input_seq)[0]
        next_word_idx = np.argmax(preds)
        next_word = tokenizer.index_word[next_word_idx]
        generated_text += " " + next_word
    return generated_text

# Test the text generation
seed_text = "He had rather see the swords"
generated_text = generate_text(model, seed_text, num_words_to_generate=20)
print(generated_text)
