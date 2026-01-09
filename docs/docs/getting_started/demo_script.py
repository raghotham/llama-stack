# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Demo script showing Chat Completions, Responses, Conversations, and RAG.

This example demonstrates:
1. Chat Completions API - OpenAI-compatible chat interface
2. Responses API - Single-turn chat with structured output
3. Conversations API - Persistent multi-turn conversations
4. RAG with Vector Stores - Retrieval-augmented generation

Run this script after starting a Llama Stack server:
    llama stack run starter
"""

import io
import os

import requests
from openai import OpenAI

# Initialize OpenAI client pointing to Llama Stack server
client = OpenAI(base_url="http://localhost:8321/v1/", api_key="none")
model = os.getenv("INFERENCE_MODEL", "ollama/llama3.2:3b")

print("=" * 60)
print("Llama Stack Demo")
print("=" * 60)

# --- Part 1: Chat Completions API ---
print("\n1. Chat Completions API (OpenAI-compatible)\n")

completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are three key principles of good software design?"},
    ],
)

print("Question: What are three key principles of good software design?")
print("-" * 40)
print(completion.choices[0].message.content)
print("-" * 40)

# --- Part 2: Responses API ---
print("\n2. Responses API (single-turn chat)\n")

response = client.responses.create(
    model=model,
    input="Give me a one-sentence summary of Python's main advantage.",
)

print("Question: Give me a one-sentence summary of Python's main advantage.")
print("-" * 40)
print(response.output[-1].content[-1].text)
print("-" * 40)

# --- Part 3: Conversations API ---
print("\n3. Conversations API (persistent multi-turn)\n")

# Create a conversation
conversation = client.conversations.create(metadata={"topic": "programming"})
print(f"Created conversation: {conversation.id}")

# First turn - the response is automatically added to the conversation
print("\nUser: My favorite programming language is Python.")
turn1 = client.responses.create(
    model=model,
    input="My favorite programming language is Python. Can you remember that?",
    conversation=conversation.id,
)
print(f"Assistant: {turn1.output[-1].content[-1].text}")

# Second turn - conversation history is automatically loaded
print("\nUser: What's my favorite language?")
turn2 = client.responses.create(
    model=model,
    input="What's my favorite programming language?",
    conversation=conversation.id,
)
print(f"Assistant: {turn2.output[-1].content[-1].text}")

# --- Part 4: RAG with Vector Stores ---
print("\n4. RAG with Vector Stores\n")

url = "https://www.paulgraham.com/greatwork.html"
print(f"Fetching document from: {url}")

# Create vector store and upload document
vs = client.vector_stores.create()

response = requests.get(url)
pseudo_file = io.BytesIO(str(response.content).encode("utf-8"))
uploaded_file = client.files.create(
    file=(url, pseudo_file, "text/html"), purpose="assistants"
)
client.vector_stores.files.create(vector_store_id=vs.id, file_id=uploaded_file.id)
print(f"File uploaded and added to vector store: {uploaded_file.id}")

query = "How do you do great work?"
print(f"\nQuery: {query}")

# Use file_search tool for automatic RAG
print("\nUsing file_search tool for retrieval-augmented generation...")
rag_response = client.responses.create(
    model=model,
    input=query,
    tools=[{"type": "file_search", "vector_store_ids": [vs.id]}],
    include=["file_search_call.results"],
)

print("-" * 40)
print(rag_response.output[-1].content[-1].text)
print("-" * 40)

print("\n" + "=" * 60)
print("Demo complete! See detailed_tutorial.mdx for more examples.")
print("=" * 60)
