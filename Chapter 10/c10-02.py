#!/usr/bin/env python
# Extracted from 'c10-Building first LLM App - Part 2.ipynb'.
# coding: utf-8

# First LLM app

import openai
import pandas as pd
import numpy as np
import os
import sys
import faiss

# Step 1: Load documents using LangChain
# Load the dataset
data = pd.read_csv('chanakya-quotes.csv')

# Step 2: Split our Documents into Text Chunks
problems = data['problem'].tolist()

openai_api_key = os.environ['OPENAI_API_KEY']
if not  openai_api_key:
    print("OpenAI API key is not defined")
    sys.exit(1)

client = openai.OpenAI(
    api_key=openai_api_key,  # this is also the default, it can be omitted
)

# Step 3: From Text Chunks to Embeddings
# You can skip this step as it's usually handled by the language model API

# Step 4: Define the LLM you want to use (e.g., GPT-3)

# Step 5: Define our Prompt Template
def generate_prompt(problem, quotes):
    prompt = ""
    if quotes:
        prompt = f"""Use any of the relevant quotes from Chanakya below (comma separated) and guide with Chanakya's wisdom for the following problem.
            Do it in below format.

            Chanakya says: <quote> 
            <wisdom elaboration> (max 100 words)
            ---------
            Problem: {problem}
            quotes: {quotes}"""
    else:
        prompt = f"""Use any of the relevant quotes from Chanakya for the following problem. 
            And give his wisdom to help in this problem.
            Do it in below format.

            Chanakya says: <quote>
            
            Wisdom: <wisdom elaboration> (max 100 words)

            ---

            Problem: {problem}"""

    return prompt

# Step 6: Creating a Vector Store

# Function to get an embedding from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Check if embeddings file exists, if not, compute and save
if not os.path.exists('quotes_problem_embeddings.npy'):
    # Precompute embeddings for all problems
    quotes_problem_embeddings = [get_embedding(problem) for problem in data['problem']]

    # Convert the embeddings to numpy array and add to the Faiss index
    quotes_problem_embeddings = np.array(quotes_problem_embeddings).astype('float32')

    # Save the verse embeddings to a file
    np.save('quotes_problem_embeddings.npy', quotes_problem_embeddings)
else:
    # Load the verse embeddings from the file
    quotes_problem_embeddings = np.load('quotes_problem_embeddings.npy')

# Initialize Faiss index
dimension = 1536
index = faiss.IndexFlatL2(dimension)

index.add(quotes_problem_embeddings)

# Function to find the most similar verses to the user's feeling
def find_similar_problems(user_problem, top_k=1):
    user_embedding = np.array(get_embedding(user_problem)).astype('float32')

    # Search for the top k similar verses
    distances, indices = index.search(np.array([user_embedding]), top_k)

    similar_user_problems = [data['problem'][i] for i in indices[0]]

    return similar_user_problems

def get_chanakya_quote(problem):
    # Search for similar problems in your dataset locally
    similar_problems = find_similar_problems(problem)

    quotes = [data[data['problem'] == p]['quote'].values[0] for p in similar_problems]

    prompt = generate_prompt(problem, quotes)
    # print ("prompt: " + prompt)

    response = client.completions.create(
        model="davinci-002",
        prompt=prompt,
        max_tokens=256  # Adjust the length of the generated quote
    )
    quote = response.choices[0].text.strip()
    return quote

for problem in ["I fear failure and it's holding me back.", "I'm feeling lost and unfulfilled in my pursuits."]:
    print('*'*40)
    print(f"Problem: {problem}")
    print(f"Recommendation: {get_chanakya_quote(problem)}")
