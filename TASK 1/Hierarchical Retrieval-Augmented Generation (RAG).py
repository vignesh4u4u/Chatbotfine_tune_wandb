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
from huggingface_hub import InferenceClient
from pdfminer.high_level import extract_text

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

text = extract_text("../pdf_files/administrative-guidance-global-anti-base-erosion-rules-pillar-two-july-2023.pdf")

def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}  "
    prompt += f"[INST] {message} [/INST]"
    return prompt

generate_kwargs = dict(
    temperature=0.7,
    max_new_tokens=6000,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    seed=42,
)

def generate_text(message, history):
    prompt = format_prompt(message, history)
    output = client.text_generation(prompt, **generate_kwargs)
    return output

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


chunk_size = 2500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

input_prompt = input("enter the query:")
answer = generate_rag(input_prompt,history=[])

print(answer)




