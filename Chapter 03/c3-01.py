#!/usr/bin/env python
# Extracted from 'c3-Data Collection and Pre-Processing for Language Modeling.ipynb'.
# coding: utf-8

# # Data Cleaning Techniques - Basic

# **Example 1: Basic Data Cleaning**

import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Lowercasing
def lowercase(text):
    text = text.lower()
    return text

# Punctuation removal
def remove_punctuation(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Removing special characters
def remove_special_chars(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Stop words removal
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Text standardization
def standardize_text(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    standardized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(standardized_tokens)

# Spelling correction
def correct_spelling(text):
    spell = SpellChecker()
    corrected_text = ' '.join([spell.correction(word) for word in text.split()])
    return corrected_text


# Apply data cleaning techniques
def clean_text(text):
    cleaned_text = lowercase(text)
    cleaned_text = remove_punctuation(cleaned_text)
    cleaned_text = remove_special_chars(cleaned_text)
    cleaned_text = remove_stopwords(cleaned_text)
    cleaned_text = standardize_text(cleaned_text)
    cleaned_text = correct_spelling(cleaned_text)
    return cleaned_text


nltk.download('stopwords')
nltk.download('wordnet')
# Try on some of the samples
text = "Hello, World! This is an examplle of text cleaning using Python."
cleaned_text = clean_text(text)
print(cleaned_text)

text = "Hi! I'm Sanket and I'm a Data Scientist. I love working with #data and #NLPs. I a have large experience in this fielld."
cleaned_text = clean_text(text)
print(cleaned_text)
