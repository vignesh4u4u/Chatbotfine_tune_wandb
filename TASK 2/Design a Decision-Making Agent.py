import os
import warnings
import os
import tempfile
import docx2txt
import requests
import numpy as np
import pandas as pd
import faiss

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, request, render_template
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": "Bearer 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"}

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
    temperature=0.5,
    max_new_tokens=3000,
    top_p=0.95,
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

def create_the_vector_store_layer1(question,index,chunks,history=[]): # layer 1
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

def generate_rag(question,index,chunks,history=[]): #layer 2
    question_embeddings = np.array([get_text_embedding(question)])
    question_embeddings.shape
    D, I = index.search(question_embeddings, k=6)
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
    summary = create_the_vector_store_layer1(question,index,chunks,history=[])
    answer = generate_text(prompt ,history=[])
    return ({"RAG_answer":answer})

app = Flask(__name__, template_folder="template")

@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    if request.method == "POST":
        prompt = request.form.get("input_prompt", "")
        file = request.files.get("file")
        if not prompt and not file:
            return "Please provide a file or input prompt", 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(tempfile.mkdtemp(), filename)
            file.save(file_path)
            if file.filename.endswith(".pdf"):
                text = extract_text(file_path)
            elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                image_text = ocr.ocr(file_path)
                text = " ".join([word[1][0] for line in image_text for word in line])
            elif file.filename.lower().endswith((".txt", ".docx", ".doc")):
                if file.filename.lower().endswith(".txt"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif file.filename.lower().endswith(('.docx', '.doc')):
                    text = docx2txt.process(file_path)
            else:
                return "Unsupported file type"

            if not prompt:
                return "Please provide the input prompt"
            chunk_size = 3000
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
            d = text_embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(text_embeddings)
            answer = generate_rag(prompt,index,chunks,history=[])

        else:
            text = prompt
            answer = generate_text(prompt,history=[])
        return render_template("sample.html", text=answer)

    return render_template("sample.html", text=" ")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5004)
