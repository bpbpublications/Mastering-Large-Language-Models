#!/usr/bin/env python
# Extracted from 'Chapter 7.ipynb'.
# coding: utf-8

# # Training Large Language Models
# Transformers

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can choose different sizes such as "t5-base", "t5-large", etc.

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

model = T5ForConditionalGeneration.from_pretrained(model_name)

# Sample input prompt for text generation
input_prompt = "Generate a sentence that describes a beautiful sunset:"

# Tokenize the input prompt and convert to tensor
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# Generate text using the T5 model
output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)

# Convert generated token IDs back to text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print("Generated Text:", generated_text)
