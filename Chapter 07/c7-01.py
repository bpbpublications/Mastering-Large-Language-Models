#!/usr/bin/env python
# Extracted from 'Chapter 7.ipynb'.
# coding: utf-8

# # Training Large Language Models

# ## Building a Tiny Language Model

from collections import defaultdict

# Function to compute bigram probabilities from a given corpus
def compute_bigram_probabilities(corpus):
    # Create a dictionary to store the bigram counts
    bigram_counts = defaultdict(lambda: defaultdict(int))
    # defaultdict creates missing items instead of throwing KeyError like dict

    # Iterate over each sentence in the corpus
    for sentence in corpus:
        words = sentence.split()  # Tokenize the sentence into words

        # Iterate over each word pair and update the bigram counts
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            bigram_counts[current_word][next_word] += 1

    # Create a dictionary to store the bigram probabilities
    bigram_probabilities = defaultdict(lambda: defaultdict(float))

    # Iterate over each word and its following word in the bigram counts
    for current_word, next_words in bigram_counts.items():
        total_count = sum(next_words.values())
        for next_word, count in next_words.items():
            bigram_probabilities[current_word][next_word] = count / total_count

    return bigram_probabilities

# Input corpus
corpus = ["Peter is happy",
          "Anna is happy",
          "Anna is sad",
          "Anna is good"]

# Compute bigram probabilities
bigram_probabilities = compute_bigram_probabilities(corpus)

# Create dictionaries to store the highest and lowest probabilities for each word
highest_probabilities = {}
lowest_probabilities = {}

# Iterate over each word and its following word in the bigram probabilities
for current_word, next_words in bigram_probabilities.items():
    # Find the word with the highest probability for the current word
    highest_probability = max(next_words, key=next_words.get)
    highest_probabilities[current_word] = highest_probability

    # Find the word with the lowest probability for the current word
    lowest_probability = min(next_words, key=next_words.get)
    lowest_probabilities[current_word] = lowest_probability

# Generate a 2-word sentence beginning with the prompt
prompt = "Peter"
hword = prompt  # for high-probability sentence
lword = prompt  # for low-probability sentence
hsentence = prompt + " "
lsentence = prompt + " "

# Generate the highest-probability and lowest-probability sentences
for _ in range(2):
    hword = highest_probabilities[hword]
    hsentence += hword + " "
    lword = lowest_probabilities[lword]
    lsentence += lword + " "

# Print the generated sentences
print("Highest-probability sentence:", hsentence)
print("Lowest-probability sentence:", lsentence)
