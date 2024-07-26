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
SORTING_LENGTH = 7

config = {
    "block_size": SORTING_LENGTH,
    "vocab_size": 10000,
    "n_layer": 1,
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
    "dataset_path": "dataset/sort.pkl",
    "tokenizer_path": "dataset/sort.json",
    "checkpoint_path": "checkpoint/sort.pt",
    "eval_interval": 100,
    "log_interval": 10,
    "eval_iters": 5,

    "backend": "nccl",
    "device": "cuda:0"
}


def generate_sorting_dataset(dataset_path, tokenizer_path):
    os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(dataset_path)), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(tokenizer_path)), exist_ok=True)

    print(f"Generating sorting data...")
    text = []
    for i in tqdm(range(10 ** SORTING_LENGTH)):
        origin_str = f"{i + 10 ** SORTING_LENGTH}"[1:]
        sorted_str = "".join(sorted(list(origin_str)))
        line = f"{origin_str}{sorted_str}"
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

def inference_sorting_gpt(config):
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
        for i in range(0, 100000):
            n = random.randint(0, 10 ** SORTING_LENGTH)
            input_seq = f"{n + 10 ** SORTING_LENGTH}"[1:]
            target_seq = "".join(sorted(list(input_seq)))
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

    generate_sorting_dataset(config["dataset_path"], config["tokenizer_path"])

    trainer = TrainGPT(config, is_from_pretrained=False)
    trainer.start()

    inference_sorting_gpt(config)

if __name__ == "__main__":
    main()