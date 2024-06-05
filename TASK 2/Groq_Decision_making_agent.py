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
from groq import Groq

os.environ["GROQ_API_KEY"] ="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"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

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

def create_the_vector_store_layer1(question,index,chunks): # layer 1
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

def generate_rag(question,index,chunks): #layer 2
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
    summary = create_the_vector_store_layer1(question,index,chunks)
    answer = generate_the_response(prompt)
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
            chunk_size = 3500
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
            d = text_embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(text_embeddings)
            answer = generate_rag(prompt,index,chunks)

        else:
            text = prompt
            answer = generate_the_response(prompt)
        return render_template("sample.html", text=answer)

    return render_template("sample.html", text=" ")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5004)