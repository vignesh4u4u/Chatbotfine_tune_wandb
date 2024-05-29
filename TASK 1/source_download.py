import requests
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text
path = "output_sample.txt"
with open(path,"w") as file:
    file.write(text)