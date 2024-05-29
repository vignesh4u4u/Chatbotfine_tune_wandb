import os
import tempfile
import docx2txt

from flask import Flask, request
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR
"""
app = Flask(__name__)

@app.route("/", methods=["POST"])
def home():
    file = request.files["file"]
    if file.filename.endswith(".pdf"):
        filenames = secure_filename(file.filename)
        file_path = os.path.join(tempfile.mkdtemp(), filenames)
        file.save(file_path)
        with open(file_path, 'rb') as pdf_file:
            text = extract_text(pdf_file)
        return text

    elif file.filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        file_path = file.read()
        image_text = ocr.ocr(file_path)
        text = " "
        for line in image_text:
            for word in line:
                text += word[1][0] + " "
        text += "\n"
        return text

    elif file.filename.endswith(".txt"):
        text = file.read().decode("utf-8")
        return text

    elif file.filename.endswith(('.docx', '.doc')):
        file_path = os.path.join(tempfile.mkdtemp(), secure_filename(file.filename))
        file.save(file_path)
        text = docx2txt.process(file_path)
        return text

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5004)

"""

from pdfminer.high_level import extract_text

text = extract_text(r"C:\Users\VigneshSubramani\Downloads\Archive\agreed-administrative-guidance-for-the-pillar-two-globe-rules.pdf")

print(text)