import os
import time
import gc
import base64
from multiprocessing import Process
from threading import Thread

import ssl
from http.server import socketserver, SimpleHTTPRequestHandler
from flask import Flask
from flask_socketio import SocketIO, emit

# LLM
from llama_cpp import Llama
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig
# from transformers import TextIteratorStreamer

USE_SSL = True

SERVER_IP = '0.0.0.0'
HTTPS_PORT = 8443 if USE_SSL else 8088
API_PORT = 5000

SSL_CERT_PATH = "/home/bd4sur/bd4sur.crt"
SSL_PRIVATE_KEY_PATH = "/home/bd4sur/key_unencrypted.pem"

CURRENT_LLM_CONFIG_KEY = "DeepSeek-R1-Distill-Qwen-7B-Q4KM-16K"

LLM_CONFIG = {
    "Qwen2.5-7B-Q4KM-16K": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/Qwen25/qwen2.5-7b-instruct-q4_k_m.gguf",
        "context_length": 16384
    },
    "DeepSeek-R1-Distill-Qwen-7B-Q4KM-16K": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/DeepSeek/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "context_length": 16384
    },
}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(12).hex()
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100000000)


LLM = None
IS_LLM_GENERATING = False
CURRENT_IMAGE_PATH = ""


def load_model(model_type, model_path, context_length=16384):
    global LLM
    del LLM
    if model_type == "gguf":
        LLM = load_gguf_model(model_path, context_length)


def load_gguf_model(model_path, context_length=16384):
    print(f"Loading GGUF Model {model_path} ...")
    model = Llama(
        model_path=model_path,
        chat_format="chatml",
        n_ctx=context_length,
        n_threads=36,
        n_gpu_layers=-1,
        verbose=False
    )
    print(f"Loaded GGUF Model {model_path}")
    return ("gguf", model, None, None)



@socketio.on('get_current_llm_key', namespace='/chat')
def get_current_llm_key():
    emit("get_current_llm_key_callback", {"current_llm_key": CURRENT_LLM_CONFIG_KEY})



@socketio.on('interrupt', namespace='/chat')
def interrupt(msg):
    global IS_LLM_GENERATING
    IS_LLM_GENERATING = False
    print("请求：中断生成")





@socketio.on('submit', namespace='/chat')
def predict(msg):
    global IS_LLM_GENERATING
    if IS_LLM_GENERATING == True:
        print("Pass")
        return
    IS_LLM_GENERATING = True

    emit("chat_response", {"timestamp": time.ctime(), "status": "start", "llm_output": None})

    model_type = LLM[0]

    response = ""

    if model_type == "gguf":
        model = LLM[1]
        output = model.create_chat_completion(
            messages=msg["chatml"],
            stream=True,
            temperature=msg["config"]["temperature"],
            top_p=msg["config"]["temperature"],
            top_k=msg["config"]["top_k"]
        )
        for chunk in output:
            if IS_LLM_GENERATING == False:
                print("已中断")
                break
            IS_LLM_GENERATING = True
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                response += delta['content']
                emit("chat_response", {
                    "timestamp": time.ctime(),
                    "status": "generating",
                    "llm_output": {"role": "assistant", "content": response}
                })

    # print(f"LLM Response: {response}")
    emit("chat_response", {
        "timestamp": time.ctime(),
        "status": "end",
        "llm_output": {"role": "assistant", "content": response}
    })
    IS_LLM_GENERATING = False




def start_https_server():
    httpd = socketserver.TCPServer((SERVER_IP, HTTPS_PORT), SimpleHTTPRequestHandler)
    if USE_SSL:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH)
        httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
    print(f"Started HTTPS Server {SERVER_IP}:{HTTPS_PORT}")
    httpd.serve_forever()


if __name__ == '__main__':
    # HTTPS Server
    print("HTTPS")
    https_server_process = Process(target=start_https_server)
    https_server_process.daemon = True
    https_server_process.start()
    # https_server_process.join()

    # LLM Server (flask app)
    llm_config = LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]
    load_model(llm_config["model_type"], llm_config["model_path"], llm_config["context_length"])

    if USE_SSL:
        socketio.run(app, host=SERVER_IP, port=API_PORT, debug=False, log_output=False, ssl_context=(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH))
    else:
        socketio.run(app, host=SERVER_IP, port=API_PORT, debug=False, log_output=False)
