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

MAX_ITERS = 5000
INPUT_LENGTH = 8

config = {
    "block_size": 8,
    "vocab_size": 10000,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "dropout": 0.0,
    "bias": False,
    "is_causal": False,

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
    "eval_only_last_token_loss": False,
    "data_dir": "dataset",
    "ckpt_dir": "checkpoint",
    "eval_interval": 100,
    "log_interval": 10,
    "eval_iters": 5,

    "backend": "nccl",
    "device": "cuda:0"
}


def generate_palindromic_dataset(data_dir="dataset"):
    os.makedirs(os.path.join(os.path.dirname(__file__), data_dir), exist_ok=True)
    print(f"Generating palindromic data...")
    text = []
    for i in tqdm(range(10 ** INPUT_LENGTH)):
        origin_str = ("-----------------"+str(i))[-INPUT_LENGTH:]
        reversed_str = origin_str[::-1]
        line = f"{origin_str}{reversed_str}"
        text.append(line)
    fulltext = "\n".join(text)

    print(f"Building tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build_from_text(fulltext, os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.json'))

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
    with open(os.path.join(os.path.dirname(__file__), data_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

def inference_palindrome_gpt(config):
    data_dir = config["data_dir"]
    ckpt_dir = config["ckpt_dir"]
    device   = config["device"]

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
    tokenizer_path = os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.json')
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer()
    tokenizer.load_from_config(tokenizer_path)

    with torch.no_grad():
        ok_count = 0
        total_count = 0
        label = ""
        for i in range(0, 100000):
            n = random.randint(0, 10 ** INPUT_LENGTH)
            input_seq = ("-----------------"+str(n))[-INPUT_LENGTH:]
            target_seq = input_seq[::-1]
            x = torch.tensor(tokenizer.encode(input_seq), dtype=torch.long, device=device)[None, ...]
            y = model.generate_sequence(x, temperature=1, top_k=1)
            output_list = []
            for t in range(len(y)):
                output_list.append(tokenizer.decode(y[t][0].tolist()))
            output_seq = "".join(output_list)
            label = "×"
            total_count += 1
            if target_seq == output_seq:
                ok_count += 1
                label = "√"
            print(f"({int(ok_count / total_count * 100)}%) [{label}] {input_seq} - {output_seq}")

def main():
    print(f"PyTorch version: {torch.__version__}")

    generate_palindromic_dataset(config["data_dir"])

    trainer = TrainGPT(config, is_from_pretraind=False)
    trainer.start()

    inference_palindrome_gpt(config)

if __name__ == "__main__":
    main()