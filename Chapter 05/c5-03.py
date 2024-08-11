#!/usr/bin/env python
# Extracted from 'c5-Neural Network Architectures for Language Modeling.ipynb'.
# coding: utf-8

# # Neural network architectures for language modeling
# ## Example of Bidirectional RNN neural network

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb


# Define hyperparameters
max_features = 20000 # Number of words to consider as features
maxlen = 200 # Cut texts after this number of words
batch_size = 32 # Number of samples per batch
embedding_dim = 100 # Dimension of word embeddings
hidden_dim = 128 # Dimension of hidden state
dropout = 0.2 # Dropout probability
learning_rate = 0.001 # Learning rate for optimizer
num_epochs = 10 # Number of training epochs


# Load and preprocess data

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences with zeros
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define Bidirectional RNN model using the Sequential API
# Create a sequential model
model = keras.Sequential()
model.add(layers.Embedding(max_features, embedding_dim))

# Add a Bidirectional RNN layer with dropout
model.add(layers.Bidirectional(layers.SimpleRNN(hidden_dim, dropout=dropout)))
# Add a dense layer with sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

# Compile and train the model
# Compile the model with binary cross entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

# Train the model on training data with validation split
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.2)

# Print model summary
print(model.summary())

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test) # Evaluate the model on test data
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
