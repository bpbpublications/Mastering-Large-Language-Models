#!/usr/bin/env python
# Extracted from 'c2-Introduction to Language Models.ipynb'.
# coding: utf-8

# # Training 'Statistical Models'

# **Example 5: Example of find ngrams of the sentence**

import nltk
from nltk.util import ngrams

# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

My_text = 'I am interested in machine learning and deep learning.'
nltk.download('punkt')

print("1-gram of the sample text: ", extract_ngrams(My_text, 1), '\n')
print("2-gram of the sample text: ", extract_ngrams(My_text, 2), '\n')
print("3-gram of the sample text: ", extract_ngrams(My_text, 3), '\n')
print("4-gram of the sample text: ", extract_ngrams(My_text, 4), '\n')

