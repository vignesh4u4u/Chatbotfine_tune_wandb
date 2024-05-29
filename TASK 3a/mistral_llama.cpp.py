from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
import os
from flask import Flask,request,render_template
from gevent.pywsgi import WSGIServer
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Mistral-7B-Instruct-v0.2-GGUF",
                                    "mistral-7b-instruct-v0.2.Q2_K.gguf"))
app = Flask(__name__)
@app.route("/cpu-machine-model" , methods=["POST"])
def llama3_test():
    query = request.form["input_prompt"]
    generator = LlamaCppGenerator(
        model=path,
        n_ctx=512,
        model_kwargs={"n_gpu_layers": -1, "seed": 42},
        generation_kwargs={"max_tokens": 1000, "temperature": 0.5},
    )
    generator.warm_up()
    result = generator.run(query)
    output = result
    filtered_output = [{'text': output['replies'][0], 'index': 0, 'logprobs': None, 'finish_reason': 'length'}]
    final_answer = filtered_output[0]['text']
    return ({"query": query, "result": final_answer})

if __name__=="__main__":
    http_server = WSGIServer(("0.0.0.0",8080),app)
    print('Server running on http://localhost:8080/cpu-machine-model')
    http_server.serve_forever()

