import os
import time
import gc
import base64
from multiprocessing import Process
from threading import Thread
import socket

import ssl
from http.server import socketserver, SimpleHTTPRequestHandler
from flask import Flask
from flask_socketio import SocketIO, emit

# LLM
from llama_cpp import Llama
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import TextIteratorStreamer



USE_API = True
USE_SSL = True



SERVER_IP = '0.0.0.0'
HTTPS_PORT = 8443 if USE_SSL else 8088
API_PORT = 5000

SSL_CERT_PATH = "/home/bd4sur/bd4sur.crt"
SSL_PRIVATE_KEY_PATH = "/home/bd4sur/key_unencrypted.pem"

NUM_GPU_LAYERS = -1 # 设为-1以加载全部层到GPU
CURRENT_LLM_CONFIG_KEY = "QwQ-32B-Q5KM"

LLM_CONFIG = {
    "Qwen2.5-7B-Q4KM": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/Qwen25/qwen2.5-7b-instruct-q4_k_m.gguf",
        "seed": 3407,
        "context_length": 0,
        "temperature": 1,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.0
    },
    "Qwen2.5-14B-1M-Q80": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/Qwen25/Qwen2.5-14B-Instruct-1M-Q8_0.gguf",
        "seed": 3407,
        "context_length": 131072,
        "temperature": 1,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.0
    },
    "Qwen2.5-72B-Q4KM": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/Qwen25/qwen2.5-72b-instruct-q4_k_m.gguf",
        "seed": 3407,
        "context_length": 16384,
        "temperature": 1,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.0
    },
    "DeepSeek-R1-UD-IQ1S": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/DeepSeek-R1/DeepSeek-R1-UD-IQ1_S.gguf",
        "seed": 3407,
        "context_length": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.0
    },
    "QwQ-32B-Q4KM": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/QwQ/qwq-32b-q4_k_m-unsloth.gguf",
        "seed": 3407,
        "context_length": 65536,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.0
    },
    "QwQ-32B-Q5KM": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/QwQ/qwq-32b-q5_k_m-unsloth.gguf",
        "seed": 3407,
        "context_length": 65536,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.0
    },
    "QwQ-32B-Q80": {
        "model_type": "gguf",
        "model_path": "/home/bd4sur/ai/_model/QwQ/qwq-32b-q8_0-unsloth.gguf",
        "seed": 3407,
        "context_length": 65536,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.0,
        "repeat_penalty": 1.0
    },
    "QwQ-32B-AWQ": {
        "model_type": "torch",
        "model_path": "/home/bd4sur/ai/_model/QwQ/QwQ-32B-AWQ",
        "context_length": 65536
    },
}


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(12).hex()
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100000000)


LLM = None
IS_LLM_GENERATING = False
CURRENT_IMAGE_PATH = ""





def load_gguf_model(model_path, context_length=16384):
    print(f"Loading GGUF Model {model_path} ...")
    model = Llama(
        chat_format="chatml",
        model_path=LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]["model_path"],
        seed=LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]["seed"],
        n_ctx=LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]["context_length"],
        n_gpu_layers=NUM_GPU_LAYERS,
        use_mmap=False,
        use_mlock=True,
        flash_attn=True,
        n_threads=8,
        split_mode=1,
            # LLAMA_SPLIT_MODE_NONE  = 0, // single GPU
            # LLAMA_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
            # LLAMA_SPLIT_MODE_ROW   = 2, // split rows across GPUs
        numa=1,
            # https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-cpu.h
            # GGML_NUMA_STRATEGY_DISABLED   = 0,
            # GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
            # GGML_NUMA_STRATEGY_ISOLATE    = 2,
            # GGML_NUMA_STRATEGY_NUMACTL    = 3,
            # GGML_NUMA_STRATEGY_MIRROR     = 4
        type_k=(2 if "DeepSeek" in CURRENT_LLM_CONFIG_KEY else 1),
            # https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
            # GGML_TYPE_F32  = 0,
            # GGML_TYPE_F16  = 1, (default)
            # GGML_TYPE_Q4_0 = 2,
            # GGML_TYPE_Q4_1 = 3,
            # GGML_TYPE_Q5_0 = 6,
            # GGML_TYPE_Q5_1 = 7,
            # GGML_TYPE_Q8_0 = 8,
            # GGML_TYPE_Q8_1 = 9,
        verbose=True
    )
    print(f"Loaded GGUF Model {model_path}")
    return ("gguf", model, None, None, None)

def load_torch_model_and_tokenizer(model_path, context_length=16384):
    print(f"Loading torch(Transformers) Model {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, resume_download=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        resume_download=True,
        torch_dtype=torch.float16
    ).eval()

    config = GenerationConfig.from_pretrained(
        model_path, trust_remote_code=True, resume_download=True,
    )

    print(f"Loaded torch(Transformers) Model {model_path} ...")

    return ("torch", model, tokenizer, config, context_length)


def llm_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_type, model_path, context_length=16384):
    global LLM
    del LLM
    llm_gc()
    if model_type == "gguf":
        LLM = load_gguf_model(model_path, context_length)
    elif model_type == "torch":
        LLM = load_torch_model_and_tokenizer(model_path, context_length)







@socketio.on('get_current_llm_key', namespace='/chat')
def get_current_llm_key():
    emit("get_current_llm_key_callback", {"current_llm_key": CURRENT_LLM_CONFIG_KEY})






@socketio.on('change_llm', namespace='/chat')
def change_llm(msg):
    global CURRENT_LLM_CONFIG_KEY
    if IS_LLM_GENERATING == True:
        emit("change_llm_response", {"is_success": False, "message": "生成中，无法切换LLM。"})
        return
    llm_config_key = msg["llm_config_key"]
    if llm_config_key not in LLM_CONFIG:
        emit("change_llm_response", {"is_success": False, "message": "LLM设置不正确。"})
        return
    if llm_config_key != CURRENT_LLM_CONFIG_KEY:
        llm_config = LLM_CONFIG[llm_config_key]
        load_model(llm_config["model_type"], llm_config["model_path"], llm_config["context_length"])
        CURRENT_LLM_CONFIG_KEY = llm_config_key
        emit("change_llm_response", {"is_success": True, "message": f"LLM已切换为{llm_config_key}。"})
    else:
        emit("change_llm_response", {"is_success": True, "message": f"LLM未切换，仍为{llm_config_key}。"})






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
    is_interrupted = False

    if model_type == "gguf":
        model = LLM[1]
        output = model.create_chat_completion(
            messages=msg["chatml"],
            stream=True,
            temperature=msg["config"]["temperature"],
            top_p=msg["config"]["temperature"],
            top_k=msg["config"]["top_k"],
            min_p=msg["config"]["min_p"],
            repeat_penalty=msg["config"]["repetition_penalty"],
            # temperature=LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]["temperature"],
            # top_p=LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]["top_p"],
            # top_k=LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]["top_k"],
            # min_p=LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]["min_p"],
            # repeat_penalty=LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]["repeat_penalty"],
        )

        for chunk in output:
            if IS_LLM_GENERATING == False:
                print("已中断")
                is_interrupted = True
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

    elif model_type == "torch":
        model = LLM[1]
        tokenizer = LLM[2]
        context_length = LLM[4]
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        text = tokenizer.apply_chat_template(
            msg["chatml"],
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=context_length)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            if IS_LLM_GENERATING == False:
                print("已中断")
                is_interrupted = True
                break
            IS_LLM_GENERATING = True
            response += new_text
            emit("chat_response", {
                "timestamp": time.ctime(),
                "status": "generating",
                "llm_output": {"role": "assistant", "content": response}
            })


    # print(f"LLM Response: {response}")
    emit("chat_response", {
        "timestamp": time.ctime(),
        "status": "interrupted" if is_interrupted else "finished",
        "llm_output": {"role": "assistant", "content": response}
    })
    IS_LLM_GENERATING = False



class IndexRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            try:
                with open('index.html', 'r', encoding='utf-8') as f:
                    content = f.read()

                modified_content = content.replace("{{SERVER_IP}}", self.headers.get("Host").split(':')[0])

                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(modified_content.encode('utf-8'))
                
            except FileNotFoundError:
                self.send_error(404, "File not found")
        else:
            super().do_GET()

    def get_server_ip(self):
        """获取本机真实IP地址"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip


def start_https_server():
    httpd = socketserver.TCPServer((SERVER_IP, HTTPS_PORT), IndexRequestHandler)
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

    if USE_API:
        # LLM Server (flask app)
        llm_config = LLM_CONFIG[CURRENT_LLM_CONFIG_KEY]
        load_model(llm_config["model_type"], llm_config["model_path"], llm_config["context_length"])

        if USE_SSL:
            socketio.run(app, host=SERVER_IP, port=API_PORT, debug=False, log_output=False, ssl_context=(SSL_CERT_PATH, SSL_PRIVATE_KEY_PATH))
        else:
            socketio.run(app, host=SERVER_IP, port=API_PORT, debug=False, log_output=False)
    else:
        https_server_process.join()
