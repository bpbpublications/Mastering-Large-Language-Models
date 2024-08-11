#!/usr/bin/env python
# Extracted from 'c2-Introduction to Language Models.ipynb'.
# coding: utf-8

# # Training 'Rule-Based Models'

# **Example 2: Matching patterns using part-of-speech tags**

import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Define the pattern using part-of-speech tags
pattern = [{'POS': 'NOUN'}, {'POS': 'VERB'}]

# Add the pattern to the matcher
matcher.add('NOUN_VERB_NOUN_PATTERN', [pattern])

# Text to be processed
text = "I saw a boy playing in the garden."

# Process the text
doc = nlp(text)

# Apply the matcher to the doc
matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
