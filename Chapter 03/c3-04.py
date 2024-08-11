#!/usr/bin/env python
# Extracted from 'c3-Data Collection and Pre-Processing for Language Modeling.ipynb'.
# coding: utf-8

# # Text Pre-processing: Preparing Text for Analysis

# **Example 4: Text pre-processing example**

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Sample text
text = "The quick brown foxes jumped over the lazy dogs."

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print("Stemmed Tokens:", stemmed_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmatized Tokens:", lemmatized_tokens)




