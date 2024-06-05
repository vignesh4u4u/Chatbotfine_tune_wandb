import os
import re
from groq import Groq
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


os.environ["GROQ_API_KEY"] ="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"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

text = extract_text("../pdf_files/dme_deloitte-global-minimum-tax-faq.pdf")

def generate_the_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="mixtral-8x7b-32768", seed=42, max_tokens=3500
    )
    answer_text = chat_completion.choices[0].message.content
    return answer_text

def get_text_embedding(input_text,history=[]):
    embedding = sbert_model.encode(input_text)
    return embedding.tolist()

def create_the_vector_store_layer1(question): # layer 1
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
    answer = generate_the_response(prompt)
    return answer

def generate_rag(question): #layer 2
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
    summary = create_the_vector_store_layer1(question)
    answer = generate_the_response(prompt )
    return ({"RAG_answer":answer,"summary":summary})


chunk_size = 2800
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

input_prompt = input("enter the query:")

output_answer = generate_rag(input_prompt)
print(output_answer)