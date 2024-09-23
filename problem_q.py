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

MAX_STEPS = 2000
Q_DIGITS = 6
CKPT_FILE_NAME = "problem_q.pt"

model_config = {
    "block_size": Q_DIGITS + 2,
    "vocab_size": 10000,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
    "dropout": 0.0,
    "bias": False,
    "use_rope": True,
    "norm_eps": 1e-5,
    "is_causal": True
}

train_config = {
    "from_checkpoint": "",
    "train_dataset_path": "dataset/problem_q_train.base64",
    "val_dataset_path": "dataset/problem_q_val.base64",
    "tokenizer_path": "checkpoint/problem_q_tokenizer.json",

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


def generate_sft_dataset():

    print(f"Generating Q data...")
    # 9  ...  9           -         5
    # 0       Q_DIGITS-1  Q_DIGITS  Q_DIGITS+1
    q_items = []
    for i in tqdm(range(10 ** Q_DIGITS)):
        istr = f"---------------------------{str(i)}"[-Q_DIGITS:]
        line = f"{istr}-{q_function(i, Q_DIGITS)}"
        q_items.append(line)
    fulltext = "\n".join(q_items)

    print(f"Building tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.build_from_text(fulltext, os.path.join(os.path.dirname(__file__), "checkpoint/problem_q_tokenizer.json"))

    train_path = os.path.join(os.path.dirname(__file__), "dataset/problem_q_train.base64")
    val_path   = os.path.join(os.path.dirname(__file__), "dataset/problem_q_val.base64")

    print(f"Shuffling sft blocks and write to file ...")
    line_indexes = list(range(len(q_items)))
    random.shuffle(line_indexes)

    with open(train_path, "w", encoding="utf-8") as f_train:
        for li in tqdm(range(0, int(len(q_items) * 0.9))):
            ids = tokenizer.encode(q_items[line_indexes[li]])
            ids = [ids[i] if i < len(ids) else tokenizer.padding_token for i in range(Q_DIGITS + 3)]
            mask = [1 if i == Q_DIGITS + 1 else 0 for i in range(Q_DIGITS + 3)]
            train_data = pickle.dumps([ids, mask])
            f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")

    with open(val_path, "w", encoding="utf-8") as f_val:
        for li in tqdm(range(int(len(q_items) * 0.9), len(q_items))):
            ids = tokenizer.encode(q_items[line_indexes[li]])
            ids = [ids[i] if i < len(ids) else tokenizer.padding_token for i in range(Q_DIGITS + 3)]
            mask = [1 if i == Q_DIGITS + 1 else 0 for i in range(Q_DIGITS + 3)]
            val_data = pickle.dumps([ids, mask])
            f_val.writelines(str(base64.b64encode(val_data), encoding="utf-8") + "\n")

    print(f"Done.")

def inference_q(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取模型检查点和训练配置
    ckpt_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
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
        for _ in range(10000):
            i = random.randint(0, 10 ** Q_DIGITS)
            istr = f"---------------------------{str(i)}"[-Q_DIGITS:]
            prompt = f"{istr}-"
            x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, ...]
            y = model.generate_next_token(x, temperature=1, top_k=1)
            qval = tokenizer.decode(y[0].tolist())
            label = "×"
            total_count += 1
            if qval == q_function(i, Q_DIGITS):
                ok_count += 1
                label = "√"
            print(f"({int(ok_count / total_count * 100)}%) [{label}] {prompt}{qval}")

def main():
    print(f"PyTorch version: {torch.__version__}")

    generate_sft_dataset()

    trainer = TrainGPT(model_config, train_config, max_steps=MAX_STEPS, ckpt_filename=CKPT_FILE_NAME)
    trainer.start()

    inference_q(checkpoint_path=f"checkpoint/{CKPT_FILE_NAME}")

if __name__ == "__main__":
    main()
