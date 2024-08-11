#!/usr/bin/env python
# Extracted from 'c3-Data Collection and Pre-Processing for Language Modeling.ipynb'.
# coding: utf-8

# # Data Cleaning Techniques - Advanced

# **Example 2: Entity detection using spaCy library in Python**

import spacy

# Load a pre-trained spaCy model for English
nlp = spacy.load("en_core_web_sm")

# Define a sample sentence
sentence = "Michael Jackson came to India in 1996 for a concert in Mumbai."

# Apply the spaCy model to the sentence
doc = nlp(sentence)

# Print the entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)




