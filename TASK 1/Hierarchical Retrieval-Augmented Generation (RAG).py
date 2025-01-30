#https://github.com/mistralai/mistral-inference
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import requests
import numpy as np
import pandas as pd
import faiss
from flask import Flask,request,render_template

from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
from text_generation import Client

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": "api_key"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}  "
    prompt += f"[INST] {message} [/INST]"
    return prompt

generate_kwargs = dict(
    temperature=0.3,
    max_new_tokens=3000,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    seed=42,
)

def generate_text(message, history):
    prompt = format_prompt(message, history)
    payload = {
        "inputs": prompt,
        "parameters": generate_kwargs
    }
    response = query(payload)
    generated_text = response[0]["generated_text"]
    if "[/INST]" in generated_text:
        generated_text = generated_text.split("[/INST]")[-1].strip()
    return generated_text


def get_text_embedding(input_text,history=[]):
    embedding = sbert_model.encode(input_text)
    return embedding.tolist()

def create_the_vector_store_layer1(question,history=[]): # layer 1
    question_embeddings = np.array([get_text_embedding(question)])
    question_embeddings.shape
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information please summaries into give the text format.    
    """
    answer = generate_text(prompt , history=[])
    return answer

def generate_rag(question,history=[]): #layer 2
    question_embeddings = np.array([get_text_embedding(question)])
    question_embeddings.shape
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    summary = create_the_vector_store_layer1(question,history=[])
    answer = generate_text(prompt , history=[])
    return ({"RAG_answer":answer,"summary":summary})

text = extract_text("../pdf_files/dme_deloitte-global-minimum-tax-faq.pdf")

chunk_size = 2500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

input_prompt = input("enter the query:")
answer = generate_rag(input_prompt,history=[])
print(answer)



"""
Why Use text_embeddings.shape[1] Instead of text_embeddings.shape[0]?
In the line:

python
Copy
Edit
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
text_embeddings.shape[1] represents the number of features (dimensions) of each embedding vector.
text_embeddings.shape[0] represents the number of vectors (number of text chunks).
Since FAISS operates on high-dimensional vectors, it needs to know the dimension (d) of each vector, not how many vectors exist.

"""


