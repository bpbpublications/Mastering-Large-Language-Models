#!/usr/bin/env python
# Extracted from 'c4-Neural Networks in Language Modeling.ipynb'.
# coding: utf-8

# ## Example of simple feedforward neural network

import tensorflow as tf
import numpy as np

# Define the training data
train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_labels = np.array([[1], [0], [0], [1]])


# Define the architecture of the feedforward neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=500, verbose=0)

# Test the model
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = model.predict(test_data)

# Print the predictions
for i in range(len(test_data)):
    print(f"Input: {test_data[i]} - Prediction: {np.round(predictions[i])}")
