#!/usr/bin/env python
# Extracted from 'c2-Introduction to Language Models.ipynb'.
# coding: utf-8

# # Training 'Rule-Based Models'

# **Example 1: Using regular expressions**

import spacy
import re

#  Type
#  python -m spacy download en_core_web_sm
#  in the terminal to download the module
nlp = spacy.load('en_core_web_sm')

# Define a pattern using a regular expression
pattern = r"\d{3}-\d{3}-\d{4}"  # Matches phone numbers in the format XXX-XXX-XXXX

# Text to be processed
text = "Please call me at 123-456-7890."

# Process the text
doc = nlp(text)

# Iterate over the matches
for match in re.finditer(pattern, doc.text):
    start, end = match.span()
    span = doc.char_span(start, end)
    # This is a Span object or None if match doesn't map to valid token sequence
    if span is not None:
        print("Found match:", span.text)
