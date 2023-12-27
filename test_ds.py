"""
Forked from https://github.com/karpathy/nanoGPT
BD4SUR 2023.12
"""

import os
import pickle
import json
import torch
from model import GPTConfig, GPT

data_dir = "data"
ckpt_dir = 'ckpt/ds'
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# init from a model saved in a specific directory
ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_dir, 'ckpt_ds.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
with open(os.path.join(os.path.dirname(__file__), ckpt_dir, 'model_args_ds.json')) as f:
    model_args = json.load(f)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint)

model.eval()
model.to(device)

meta_path = os.path.join(os.path.dirname(__file__), data_dir, 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def typewriter(token_tensor):
    print(decode(token_tensor[0].tolist()), end="", flush=True)

with torch.no_grad():
    while True:
        try:
            prompt = input("Prompt: ")
        except EOFError:
            break
        x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
        print(prompt, end="", flush=True)
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, callback=typewriter)
        print("\n")
