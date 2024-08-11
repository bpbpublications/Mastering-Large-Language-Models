#!/usr/bin/env python
# Extracted from 'c5-Neural Network Architectures for Language Modeling.ipynb'.
# coding: utf-8

# # Neural network architectures for language modeling
# ## Example of CNN network

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Example usage
vocab_size = 10000
embedding_dim = 100
num_filters = 128
filter_size = 3
hidden_dim = 256
output_dim = 10
dropout = 0.5
max_sequence_length = 100

# Define the model architecture
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(output_dim, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()