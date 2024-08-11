#!/usr/bin/env python
# Extracted from 'c3-Data Collection and Pre-Processing for Language Modeling.ipynb'.
# coding: utf-8

# **Example 6: Dependency parsing**

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("John saw a flashy blue hat at the store")
displacy.serve(doc, style="dep", port=5051)
