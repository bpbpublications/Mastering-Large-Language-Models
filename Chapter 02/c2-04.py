#!/usr/bin/env python
# Extracted from 'c2-Introduction to Language Models.ipynb'.
# coding: utf-8

# # Training 'Statistical Models'

# **Example 4: simple implementation of a bigram language model.**


# Define a function to read the data from a list of sentences
def readData():
    # Initialize an empty list to store the words
    data = ['This is a house','This is a home','I love my house','This is my home', 'Is this your house?']
    dat=[]
    # Loop through each sentence in the data
    for i in range(len(data)):
        # Split the sentence into words and append them to the list
        for word in data[i].split():
            dat.append(word)
    # Print the list of words
    print(dat)
    # Return the list of words
    return dat

# Define a function to create bigrams from the list of words
def createBigram(data):
   # Initialize an empty list to store the bigrams
   listOfBigrams = []
   # Initialize an empty dictionary to store the bigram counts
   bigramCounts = {}
   # Initialize an empty dictionary to store the unigram counts
   unigramCounts = {}
   # Loop through each word in the data except the last one
   for i in range(len(data)-1):
      # Check if the next word is lowercase (to avoid punctuation marks)
      if i < len(data) - 1 and data[i+1].islower():
         # Create a bigram tuple from the current and next word and append it to the list
         listOfBigrams.append((data[i], data[i + 1]))
         # Increment the count of the bigram in the dictionary or set it to 1 if not present
         if (data[i], data[i+1]) in bigramCounts:
            bigramCounts[(data[i], data[i + 1])] += 1
         else:
            bigramCounts[(data[i], data[i + 1])] = 1
      # Increment the count of the current word in the dictionary or set it to 1 if not present
      if data[i] in unigramCounts:
         unigramCounts[data[i]] += 1
      else:
         unigramCounts[data[i]] = 1
   # Return the list of bigrams, the unigram counts and the bigram counts
   return listOfBigrams, unigramCounts, bigramCounts


# Define a function to calculate the bigram probabilities from the counts
def calcBigramProb(listOfBigrams, unigramCounts, bigramCounts):
    # Initialize an empty dictionary to store the bigram probabilities
    listOfProb = {}
    # Loop through each bigram in the list
    for bigram in listOfBigrams:
        # Get the first and second word of the bigram
        word1 = bigram[0]
        word2 = bigram[1]
        # Calculate the probability of the bigram as the ratio of its count and the count of the first word
        listOfProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))
    # Return the dictionary of bigram probabilities
    return listOfProb


# Call the readData function and store the result in data variable
data = readData()
# Call the createBigram function with data as argument and store the results in three variables
listOfBigrams, unigramCounts, bigramCounts = createBigram(data)

# Print some messages and results for debugging purposes
print("\n All the possible Bigrams are ")
print(listOfBigrams)

print("\n Bigrams along with their frequency ")
print(bigramCounts)

print("\n Unigrams along with their frequency ")
print(unigramCounts)

# Call the calcBigramProb function with the counts as arguments and store the result in bigramProb variable
bigramProb = calcBigramProb(listOfBigrams, unigramCounts, bigramCounts)

print("\n Bigrams along with their probability ")
print(bigramProb)