import os
import json
import random
import pickle
import numpy as np
import torch
from tqdm import tqdm
from tokenizer import Tokenizer
from model import GPT
from train import TrainGPT

MAX_ITERS = 3000
Q_DIGITS = 6

config = {
    "block_size": Q_DIGITS + 1,
    "vocab_size": 10000,
    "n_layer": 1,
    "n_head": 2,
    "n_embd": 16,
    "dropout": 0.0,
    "bias": False,
    "is_causal": True,

    "learning_rate": 6e-4,
    "max_iters": MAX_ITERS,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.99,

    "decay_lr": True,
    "warmup_iters": 300,
    "lr_decay_iters": MAX_ITERS,
    "min_lr": 6e-5,

    "init_from": "pretrain",
    "batch_size": 300,
    "random_seed": 114514,
    "eval_only_last_token_loss": True,
    "dataset_path": "dataset/q.pkl",
    "tokenizer_path": "dataset/q.json",
    "checkpoint_path": "checkpoint/q.pt",
    "eval_interval": 100,
    "log_interval": 10,
    "eval_iters": 5,

    "backend": "nccl",
    "device": "cuda:0"
}


# Q函数：一串数字中有多少个圈儿
def q_function(number: int, num_digits: int) -> int:
    """
    Q函数：一串数字中有多少个圈儿。
        例如：q(2024)=1，q(888)=6
        出典：https://www.zhihu.com/question/338618946/answer/831919337、https://www.zhihu.com/question/341026031/answer/841578656
    """
    #         0  1  2  3  4  5  6  7  8  9  10
    qv_map = [1, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0]
    res_map = "0123456789abcdefghijklmnopqrstuvwxyz"
    istr = f"---------------------------{str(number)}"[-num_digits:]
    qv = 0
    for i in range(num_digits):
        d = 10 if istr[i] == "-" else int(istr[i])
        qv = qv + qv_map[d]
    return res_map[qv]


def generate_q_dataset(dataset_path, tokenizer_path):
    os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(dataset_path)), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(tokenizer_path)), exist_ok=True)

    print(f"Generating Q data...")
    text = []
    for i in tqdm(range(10 ** Q_DIGITS)):
        line = f"{i + 10 ** Q_DIGITS}-{q_function(i, Q_DIGITS)}"[1:]
        text.append(line)
    fulltext = "\n".join(text)

    print(f"Building tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build_from_text(fulltext, os.path.join(os.path.dirname(__file__), tokenizer_path))

    print(f"Shuffling and encoding data blocks...")
    train_ids = []
    val_ids = []
    line_indexes = list(range(len(text)))
    random.shuffle(line_indexes)
    for li in tqdm(range(0, int(len(text) * 0.4))):
        train_ids.append(tokenizer.encode(text[line_indexes[li]]))
    for li in tqdm(range(int(len(text) * 0.4), len(text))):
        val_ids.append(tokenizer.encode(text[line_indexes[li]]))

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    dataset = {
        "train_ids": train_ids,
        "val_ids": val_ids
    }
    with open(os.path.join(os.path.dirname(__file__), dataset_path), 'wb') as f:
        pickle.dump(dataset, f)


def inference_q(config):
    device = config["device"]

    # 读取模型检查点和训练配置
    ckpt_path = os.path.join(os.path.dirname(__file__), config["checkpoint_path"])
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    train_config = checkpoint['train_config']
    model_config = checkpoint['model_config']

    # 设置随机种子与训练设置一致
    torch.manual_seed(train_config.random_seed)
    torch.cuda.manual_seed(train_config.random_seed)

    # 加载模型权重
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    # 读取分词器
    tk_path = os.path.join(os.path.dirname(__file__), config["tokenizer_path"])
    print(f"Loading tokenizer from {tk_path}...")
    tokenizer = Tokenizer()
    tokenizer.load_from_config(tk_path)

    with torch.no_grad():
        ok_count = 0
        total_count = 0
        label = ""
        for i in range(99900000, 99999999):
            n = random.randint(0, 10 ** Q_DIGITS)
            prompt = f"{n + 10 ** Q_DIGITS}-"[1:]
            x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, ...]
            y = model.predict_next_token(x, temperature=1, top_k=1)
            qval = tokenizer.decode(y[0].tolist())
            label = "×"
            total_count += 1
            if qval == q_function(n, Q_DIGITS):
                ok_count += 1
                label = "√"
            print(f"({int(ok_count / total_count * 100)}%) [{label}] {prompt}{qval}")

def main():
    print(f"PyTorch version: {torch.__version__}")

    generate_q_dataset(config["dataset_path"], config["tokenizer_path"])

    trainer = TrainGPT(config, is_from_pretrained=False)
    trainer.start()

    inference_q(config)

if __name__ == "__main__":
    main()