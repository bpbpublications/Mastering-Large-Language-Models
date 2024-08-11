#!/usr/bin/env python
# Extracted from 'c3-Data Collection and Pre-Processing for Language Modeling.ipynb'.
# coding: utf-8

# **Example 3: Anonymization using the spaCy library in Python**

import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the text to be anonymized
text = "Michael Jackson came to India in 1996 for a concert in Mumbai."

# Process the text with spaCy
doc = nlp(text)

# Iterate over the named entities and replace them with placeholders
anonymized_tokens = []
for token in doc:
    if token.ent_type_ in ['PERSON', 'GPE', 'DATE']:
        anonymized_tokens.append(token.ent_type_)
    else:
        anonymized_tokens.append(token.text)

# Join the anonymized tokens back into a single string
anonymized_text = ' '.join(anonymized_tokens)

# Print the anonymized text
print(anonymized_text)




