import os
import pickle
import torch
from model import ModelConfig, GPT
from qfunc import q_function

data_dir = "data_q"
ckpt_dir = 'ckpt_q'
device = 'cuda'

# 读取模型检查点和训练配置
ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_dir, 'ckpt.pt')
print(f"Loading checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
config = checkpoint['config']

# 设置随机种子与训练设置一致
torch.manual_seed(config.random_seed)
torch.cuda.manual_seed(config.random_seed)

# 加载模型权重
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

# 读取分词器
tokenizer_path = os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.pkl')
print(f"Loading tokenizer from {tokenizer_path}...")
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
stoi, itos = tokenizer['stoi'], tokenizer['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 开始推理
with torch.no_grad():
    ok_count = 0
    total_count = 0
    label = ""
    for i in range(99900000, 99999999):
        prompt = f"{i + 100000000}-"[1:]
        x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
        y = model.predict_next_token(x, temperature=1, top_k=1)
        qval = decode(y[0].tolist())
        label = "×"
        total_count += 1
        if qval == q_function(i, 8):
            ok_count += 1
            label = "√"
        print(f"({int(ok_count / total_count * 100)}%) [{label}] {prompt}{qval}")
