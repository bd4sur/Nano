import os
import time
import base64
import random
import pickle
import torch
from tqdm import tqdm
from tokenizer import Tokenizer
from model import GPT
from train import TrainGPT

MAX_STEPS = 1000
SEQ_LENGTH = 6
CKPT_FILE_NAME = "problem_palindrome.pt"

model_config = {
    "block_size": SEQ_LENGTH,
    "vocab_size": 10000,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "dropout": 0.0,
    "bias": False,
    "use_rope": False,
    "norm_eps": 1e-5,
    "is_causal": False,
}

train_config = {
    "from_checkpoint": "",
    "train_dataset_path": ["dataset/problem_palindrome_train.base64"],
    "val_dataset_path": ["dataset/problem_palindrome_val.base64"],
    "tokenizer_path": "checkpoint/problem_palindrome_tokenizer.json",

    "random_seed": 39,
    "batch_size": 1000,

    "dropout": 0.0,

    "learning_rate": 6e-4,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.99,

    "decay_lr": True,
    "warmup_iters": 300,
    "lr_decay_iters": MAX_STEPS,
    "min_lr": 6e-5,

    "eval_interval": 100,
    "log_interval": 10,
    "eval_iters": 5,

    "backend": "nccl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sdp_kernel": "flash",
    "dtype": "bfloat16",
    "grad_clip": 1.0,
    "gradient_accumulation_steps": 4
}

base_path = os.path.dirname(__file__)

def generate_dataset():
    print(f"Generating palindrome data...")
    all_items = []
    for i in tqdm(range(10 ** SEQ_LENGTH)):
        origin_str = f"{i + 10 ** SEQ_LENGTH}"[1:]
        reversed_str = origin_str[::-1]
        line = f"{origin_str}{reversed_str}"
        all_items.append(line)
    fulltext = "\n".join(all_items)

    print(f"Building tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build_from_text(fulltext, os.path.join(base_path, "checkpoint/problem_palindrome_tokenizer.json"))

    train_path = os.path.join(base_path, "dataset/problem_palindrome_train.base64")
    val_path   = os.path.join(base_path, "dataset/problem_palindrome_val.base64")

    print(f"Shuffling sft blocks and write to file ...")
    line_indexes = list(range(len(all_items)))
    random.shuffle(line_indexes)

    with open(train_path, "w", encoding="utf-8") as f_train:
        for li in tqdm(range(0, int(len(all_items) * 0.4))):
            ids = tokenizer.encode(all_items[line_indexes[li]])
            ids = [ids[i] if i < len(ids) else tokenizer.padding_token for i in range(SEQ_LENGTH * 2)]
            train_data = pickle.dumps([ids, None])
            f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")

    with open(val_path, "w", encoding="utf-8") as f_val:
        for li in tqdm(range(int(len(all_items) * 0.4), len(all_items))):
            ids = tokenizer.encode(all_items[line_indexes[li]])
            ids = [ids[i] if i < len(ids) else tokenizer.padding_token for i in range(SEQ_LENGTH * 2)]
            val_data = pickle.dumps([ids, None])
            f_val.writelines(str(base64.b64encode(val_data), encoding="utf-8") + "\n")

    print(f"Done.")

def inference(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取模型检查点和训练配置
    ckpt_path = os.path.join(base_path, checkpoint_path)
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    train_config = checkpoint['train_config']
    model_config = checkpoint['model_config']

    # 设置随机种子与训练设置一致
    torch.manual_seed(train_config.random_seed)

    # 加载模型权重
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    _ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[train_config.dtype]
    model.to(device, dtype=_ptdtype)

    # 读取分词器
    tokenizer = Tokenizer()
    tokenizer.load_from_config_dict(checkpoint['tokenizer_config'])

    with torch.no_grad():
        ok_count = 0
        total_count = 0
        label = ""
        for i in range(0, 100000):
            n = random.randint(0, 10 ** SEQ_LENGTH)
            input_seq = f"{n + 10 ** SEQ_LENGTH}"[1:]
            target_seq = input_seq[::-1]
            x = torch.tensor(tokenizer.encode(input_seq), dtype=torch.long, device=device)[None, ...]
            y = model.non_auto_regressive_generate(x, temperature=1, top_k=1)
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

    generate_dataset()

    trainer = TrainGPT(model_config, train_config, max_steps=MAX_STEPS, ckpt_filename=CKPT_FILE_NAME)
    trainer.start()

    inference(checkpoint_path=f"checkpoint/{CKPT_FILE_NAME}")

if __name__ == "__main__":
    main()
