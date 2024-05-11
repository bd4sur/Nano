"""
Forked from https://github.com/karpathy/nanoGPT
BD4SUR 2023.12
"""

import os
import pickle
import torch
from model_q import GPTConfig, GPT

data_dir = "data_q"
ckpt_dir = 'ckpt_q'
max_new_tokens = 1
temperature = 1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1
seed = 1337
device = 'cuda'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])

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
    newtokens = decode(token_tensor[0].tolist())
    print(newtokens, end="", flush=True)

NUM_DIGITS = 8
# Q函数：一串数字中有多少个圈儿
def q_function(number: int) -> int:
    """
    Q函数：一串数字中有多少个圈儿。
        例如：q(2024)=1，q(888)=6
        出典：https://www.zhihu.com/question/338618946/answer/831919337、https://www.zhihu.com/question/341026031/answer/841578656
    """
    #         0  1  2  3  4  5  6  7  8  9  10
    qv_map = [1, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0]
    res_map = "0123456789abcdefghijklmnopqrstuvwxyz"
    istr = f"---------------------------{str(number)}"[-NUM_DIGITS:]
    qv = 0
    for i in range(NUM_DIGITS):
        d = 10 if istr[i] == "-" else int(istr[i])
        qv = qv + qv_map[d]
    return res_map[qv]

with torch.no_grad():
    ok_count = 0
    total_count = 0
    label = ""
    for i in range(99900000, 99999999):
        prompt = f"{i + 100000000}-"[1:]
        x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
        y = model.predict_next_token(x, temperature=temperature, top_k=top_k)
        qval = decode(y[0].tolist())
        label = "×"
        total_count += 1
        if qval == q_function(i):
            ok_count += 1
            label = "√"
        print(f"({int(ok_count / total_count * 100)}%) [{label}] {prompt}{qval}")
