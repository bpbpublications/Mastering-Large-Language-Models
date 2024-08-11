#!/usr/bin/env python
# Extracted from 'c13-Prompt Engineering.ipynb'.
# coding: utf-8

# # Prompt Engieering

# We will use openai APIs for prompt engineering

from openai import OpenAI
import os
import sys

# Your API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print('OpemAI API key is not defined')
    sys.exit(1)


# Initialize OpenAI client
client = OpenAI(api_key=api_key)


# ## Example of Chain-of-Thought (CoT) Prompting


# Chain-of-thought steps
steps = [
    "1. Identify the relevant facts and information from the prompt.",
    "2. Analyze the relationships between these facts and information.",
    "3. Consider different possible conclusions based on the analysis.",
    "4. Evaluate the plausibility of each conclusion.",
    "5. Choose the most likely conclusion and explain your reasoning.",
]

# Information and question
information = """Alice has 3 apples and Bob has 5 oranges. 
    They decide to combine their fruit.
    From those fruits, they made orange juice from 2 fruits."""

question = "How many pieces of fruit now do they have in total?"


# Chain-of-thought prompt with steps
step_str = '\n'.join(steps)
chat_prompt = f"""Follow these steps in your response:

{step_str}

Information: {information}

Question: {question}"""

print (chat_prompt)


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": chat_prompt}
  ]
)

print(completion.choices[0].message)


# ## Prompt Design Examples for Different Tasks

# Example of Text Summarization
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Please summarise below text: Prompt engineering is the art and science of crafting the right input to elicit the desired output from these language models. It is a skill that can transform a powerful language model into a tailored solution for a multitude of NLP tasks. Crafting prompts effectively requires understanding the model's capabilities, the nuances of different NLP tasks, and a knack for linguistic precision. In this chapter, we will delve into the intricate world of prompt engineering, revealing its secrets and teaching you the techniques needed to harness the immense potential of LMs."}
  ]
)
print(completion.choices[0].message)


# Example of Question Answering
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": """With the following information, what is the shape of the earth.
    Info: Earth has never been perfectly round. The planet bulges around the equator by an extra 0.3 percent as a result of the fact that it rotates about its axis. Earth's diameter from North to South Pole is 12,714 kilometers (7,900 miles), while through the equator it is 12,756 kilometers (7,926 miles)."""}
  ]
)
print(completion.choices[0].message)
