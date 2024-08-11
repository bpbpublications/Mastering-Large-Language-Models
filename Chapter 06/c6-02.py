#!/usr/bin/env python
# Extracted from 'c6-Transformer-Based Models for Language Modeling.ipynb'.
# coding: utf-8

# # Positional encoding example

# Here is a short Python code to implement positional encoding using NumPy.
# The code is simplified to make the understanding of positional encoding easier.

import numpy as np

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

# n : User-defined scalar
# d: Dimension of the output embedding space
P = getPositionEncoding(seq_len=4, d=4, n=100)
print(P)
