#!/usr/bin/env python
# Extracted from 'c2-Introduction to Language Models.ipynb'.
# coding: utf-8

# # Training 'Rule-Based Models'

# **Example 3: Matching specific tokens with specific attributes**

import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Define the pattern
pattern = [{'LOWER': 'machine'}, {'LOWER': 'learning'}]

# Add the pattern to the matcher
matcher.add('ML_PATTERN', [pattern])

# Text to be processed
text = "I am interested in machine learning and deep learning."

# Process the text
doc = nlp(text)

# Apply the matcher to the doc
matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
