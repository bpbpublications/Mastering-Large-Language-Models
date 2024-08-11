#!/usr/bin/env python
# Extracted from 'c6-Transformer-Based Models for Language Modeling.ipynb'.
# coding: utf-8

# # Self-attention example
# Load model and retrieve attention weights

from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel
import webbrowser

model_version = 'bert-base-uncased'
model = BertModel.from_pretrained(model_version, output_attentions=True)

# Convert inputs and outputs to subwords
tokenizer = BertTokenizer.from_pretrained(model_version)
sentence_a = "Peter loves animals, he likes cats more than dogs"
sentence_b = "He likes apples but hates oranges"
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']

# Get normalized attention weights for each layer
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
sentence_b_start = token_type_ids[0].tolist().index(1)
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)


# ## Display Attention
# The head view visualizes attention in one or more heads from a single Transformer layer.
# Each line shows the attention from one token (left) to another (right).
# Line weight reflects the attention value (ranges from 0 to 1), while line color identifies the attention head.
# When multiple heads are selected (indicated by the colored tiles at the top),
# the corresponding visualizations are overlaid onto one another.
# head view
html = head_view(attention, tokens, sentence_b_start, html_action='return')
with open('attention.html', 'w') as output_file:
    output_file.write(html.data)
    print(f"Attention html written to {output_file.name}")
    webbrowser.open(output_file.name)
