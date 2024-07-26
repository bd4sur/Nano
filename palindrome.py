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
INPUT_LENGTH = 7

config = {
    "block_size": INPUT_LENGTH,
    "vocab_size": 10000,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 24,
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
    "batch_size": 500,
    "random_seed": 1314,
    "eval_only_last_token_loss": False,
    "dataset_path": "dataset/palindrome.pkl",
    "tokenizer_path": "dataset/palindrome.json",
    "checkpoint_path": "checkpoint/palindrome.pt",
    "eval_interval": 100,
    "log_interval": 10,
    "eval_iters": 5,

    "backend": "nccl",
    "device": "cuda:0"
}

CHAR_LIST = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def random_string():
    ranstr = "".join([chr(ord("a")+random.randint(0,25)) for _ in range(0, random.randint(1, INPUT_LENGTH))])
    return ("-----------------"+ranstr)[-INPUT_LENGTH:]

def generate_palindromic_dataset(dataset_path, tokenizer_path):
    os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(dataset_path)), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(tokenizer_path)), exist_ok=True)

    print(f"Generating palindromic data...")
    text = []
    for i in tqdm(range(10 ** INPUT_LENGTH)):
        origin_str = ("-----------------"+str(i))[-INPUT_LENGTH:]
        reversed_str = origin_str[::-1]
        line = f"{origin_str}{reversed_str}"
        text.append(line)
    for _ in tqdm(range(10 ** (INPUT_LENGTH-1))):
        origin_str = random_string()
        reversed_str = origin_str[::-1]
        line = f"{origin_str}{reversed_str}"
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
    for li in tqdm(range(0, int(len(text) * 0.2))):
        train_ids.append(tokenizer.encode(text[line_indexes[li]]))
    for li in tqdm(range(int(len(text) * 0.2), len(text))):
        val_ids.append(tokenizer.encode(text[line_indexes[li]]))

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    dataset = {
        "train_ids": train_ids,
        "val_ids": val_ids
    }
    with open(os.path.join(os.path.dirname(__file__), dataset_path), 'wb') as f:
        pickle.dump(dataset, f)

def inference_palindrome_gpt(config):
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
        for i in range(0, 1000):
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

    generate_palindromic_dataset(config["dataset_path"], config["tokenizer_path"])

    trainer = TrainGPT(config, is_from_pretrained=False)
    trainer.start()

    inference_palindrome_gpt(config)

if __name__ == "__main__":
    main()