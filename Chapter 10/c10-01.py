#!/usr/bin/env python
# Extracted from 'c10-Building first LLM App - Part 1.ipynb'.
# coding: utf-8

# Langchain demo

import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
import os
import sys

csv_file_path = 'AAPL.csv'


def download_csv():
    csv_url = 'https://raw.githubusercontent.com/matplotlib/sample_data/master/aapl.csv'
    response = requests.get(csv_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(csv_file_path, 'wb') as file:
        file.write(response.content)


download_csv()

MODEL_NAME = "gpt-3.5-turbo"
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print('OPENAI API key is not set')
    sys.exit(1)


# Initialize the OpenAI API

agent = create_csv_agent(
    ChatOpenAI(api_key=openai_api_key, model_name=MODEL_NAME),
    csv_file_path,
    verbose=True,
    allow_dangerous_code=True
)

response = agent.run("How many rows of data do you have?")
print(response)

response = agent.run("What is average high for Apple in year 2022?")
print(response)

response = agent.run("What is the highest high?")
print(response)

response = agent.run("What is the lowest low?")
print(response)

response = agent.run("What is the difference between highest high & lowest low?")
print(response)

response = agent.run("Show me line chart for High, Low over dates")
print(response)
