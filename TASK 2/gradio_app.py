from flask import Flask, request, render_template
import os
import tempfile
import docx2txt
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR

app = Flask(__name__, template_folder=".")
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

        if prompt:
            text += "\n" + prompt

        return render_template("sample.html", text=text)

    return render_template("sample.html", text="")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5004)




"""

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload</title>
    <style type="text/css">
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1 {
            text-align: center;
            font-size: 24px;
            color: blue;
        }
        pre {
            white-space: pre-wrap;
            padding: 10px;
            border: 1px solid #ccc;
            background-color: #fff;
        }
        form {
            text-align: center;
        }
        input[type="file"] {
            margin-top: 10px;
        }
        textarea {
            margin-top: 10px;
            width: 80%;
            height: 100px;
            resize: vertical;
        }
        button[type="submit"] {
            margin-top: 10px;
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>Upload a File</h1>
    <form id="uploadForm" method="POST" enctype="multipart/form-data">
        <label for="file">Choose a file to upload:</label>
        <br>
        <input type="file" id="file" name="file" accept=".pdf, .jpg, .jpeg, .png, .webp, .txt, .doc, .docx">
        <br>
        <label for="input_prompt">Enter your prompt:</label>
        <br>
        <textarea id="input_prompt" name="input_prompt"></textarea>
        <br>
        <button type="submit">Submit</button>
    </form>
    <h1>Result</h1>
    <pre id="result">{{ text }}</pre>
</body>
</html>
"""


from flask import Flask, request, render_template, jsonify
import os
import tempfile
import docx2txt
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR

app = Flask(__name__, template_folder=".")

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

        if prompt:
            text += "\n" + prompt

        return jsonify({"text": text})

    return render_template("sample.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5004)


"""

function submitForm() {
    var fileInput = document.getElementById('file');
    var promptInput = document.getElementById('input_prompt');
    var resultElement = document.getElementById('result');
    if (fileInput.files.length > 0 && promptInput.value.trim()!== '') {
        var formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('input_prompt', promptInput.value);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
       .then(response => response.json())
       .then(data => {
            resultElement.textContent = data.text;
        })
       .catch(error => {
            console.error(error);
        });
    } else {
        // Handle errors
    }
}
"""


"""

code1:

from flask import Flask, request, render_template
import os
import tempfile
import docx2txt
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR

app = Flask(__name__, template_folder=".")
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

        if prompt:
            text += "\n" + prompt
        #return text

        return render_template("sample.html", text=text)

    return render_template("sample.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5004)



code 2:

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload</title>
    <style type="text/css">
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1 {
            text-align: center;
            font-size: 24px;
            color: blue;
        }
        pre {
            white-space: pre-wrap;
            padding: 10px;
            border: 1px solid #ccc;
            background-color: #fff;
        }
        form {
            text-align: center;
        }
        input[type="file"] {
            margin-top: 10px;
        }
        textarea {
            margin-top: 10px;
            width: 80%;
            height: 100px;
            resize: vertical;
        }
        button[type="submit"] {
            margin-top: 10px;
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>Upload a File</h1>
    <form id="uploadForm" enctype="multipart/form-data">
        <label for="file">Choose a file to upload:</label>
        <br>
        <input type="file" id="file" name="file" accept=".pdf, .jpg, .jpeg, .png, .webp, .txt, .doc, .docx">
        <br>
        <label for="input_prompt">Enter your prompt:</label>
        <br>
        <textarea id="input_prompt" name="input_prompt"></textarea>
        <br>
        <button type="button" onclick="submitForm()">Submit</button>
    </form>
    <h1>Result</h1>
    <pre id="result"></pre>
        <script>
        function submitForm() {
            var fileInput = document.getElementById('file');
            var promptInput = document.getElementById('input_prompt');
            var resultElement = document.getElementById('result');
            if (fileInput.files.length > 0 && promptInput.value.trim() !== '') {

                var file = fileInput.files[0];
                var reader = new FileReader();
                reader.onload = function (event) {
                    var content = event.target.result;
                    var prompt = promptInput.value;
                    resultElement.textContent = content + '\n' + prompt;
                };
                reader.readAsText(file);
            } else if (fileInput.files.length === 0 && promptInput.value.trim() !== '') {
                // File not present but prompt is present
                resultElement.textContent = promptInput.value;
            } else if (fileInput.files.length > 0 && promptInput.value.trim() === '') {
                // File present but prompt is not present
                resultElement.textContent = 'Error: Input prompt is missing';
            } else {
                // Neither file nor prompt is present
                resultElement.textContent = 'Error: Please upload a file or enter a prompt';
            }
        }
    </script>


</body>
</html>


above two code why pdf file upload time give the below format output

"""