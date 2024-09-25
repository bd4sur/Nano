import os
import base64
import random
import pickle
import torch
from tqdm import tqdm
from tokenizer import Tokenizer
from model import GPT
from train import TrainGPT

TASK_TAG = "palindrome"

MAX_STEPS = 1000
SEQ_LENGTH = 6

CHECKPOINT_FILE_NAME = f"problem_{TASK_TAG}.pt"
TRAINSET_PATH        = f"dataset_preprocessed/problem_{TASK_TAG}_train.base64"
VALSET_PATH          = f"dataset_preprocessed/problem_{TASK_TAG}_val.base64"
TOKENIZER_PATH       = f"dataset_preprocessed/problem_{TASK_TAG}_tokenizer.json"


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
    "dataset_path": [
        [TRAINSET_PATH, VALSET_PATH]
    ],
    "tokenizer_path": TOKENIZER_PATH,

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

if TASK_TAG == "q":
    model_config["block_size"] = SEQ_LENGTH + 2
    model_config["n_layer"]    = 2
    model_config["n_head"]     = 2
    model_config["n_embd"]     = 64
    model_config["use_rope"]   = True
    model_config["is_causal"]  = True

elif TASK_TAG == "sort":
    model_config["block_size"] = SEQ_LENGTH
    model_config["n_layer"]    = 2
    model_config["n_head"]     = 2
    model_config["n_embd"]     = 32
    model_config["use_rope"]   = False
    model_config["is_causal"]  = False

elif TASK_TAG == "palindrome":
    model_config["block_size"] = SEQ_LENGTH
    model_config["n_layer"]    = 2
    model_config["n_head"]     = 2
    model_config["n_embd"]     = 32
    model_config["use_rope"]   = False
    model_config["is_causal"]  = False

base_path = os.path.dirname(__file__)

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


def generate_dataset():
    if TASK_TAG == "q":
        print(f"Generating Q data...")
        # 9  ...  9           -         5
        # 0       Q_DIGITS-1  Q_DIGITS  Q_DIGITS+1
        all_items = []
        for i in tqdm(range(10 ** SEQ_LENGTH)):
            istr = f"---------------------------{str(i)}"[-SEQ_LENGTH:]
            line = f"{istr}-{q_function(i, SEQ_LENGTH)}"
            all_items.append(line)
        fulltext = "\n".join(all_items)
    elif TASK_TAG == "sort":
        print(f"Generating sort data...")
        all_items = []
        for i in tqdm(range(10 ** SEQ_LENGTH)):
            origin_str = f"{i + 10 ** SEQ_LENGTH}"[1:]
            sorted_str = "".join(sorted(list(origin_str)))
            line = f"{origin_str}{sorted_str}"
            all_items.append(line)
        fulltext = "\n".join(all_items)
    elif TASK_TAG == "palindrome":
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
    tokenizer.build_from_text(fulltext, os.path.join(base_path, TOKENIZER_PATH))

    train_path = os.path.join(base_path, TRAINSET_PATH)
    val_path   = os.path.join(base_path, VALSET_PATH)

    print(f"Shuffling sft blocks and write to file ...")
    line_indexes = list(range(len(all_items)))
    random.shuffle(line_indexes)

    with open(train_path, "w", encoding="utf-8") as f_train:
        for li in tqdm(range(0, int(len(all_items) * 0.4))):
            ids = tokenizer.encode(all_items[line_indexes[li]])
            if TASK_TAG == "q":
                ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(SEQ_LENGTH + 3)]
                mask = [1 if i == SEQ_LENGTH + 1 else 0 for i in range(SEQ_LENGTH + 3)]
            elif TASK_TAG in ["sort", "palindrome"]:
                ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(SEQ_LENGTH * 2)]
                mask = None
            train_data = pickle.dumps([ids, mask])
            f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")

    with open(val_path, "w", encoding="utf-8") as f_val:
        for li in tqdm(range(int(len(all_items) * 0.4), len(all_items))):
            ids = tokenizer.encode(all_items[line_indexes[li]])
            if TASK_TAG == "q":
                ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(SEQ_LENGTH + 3)]
                mask = [1 if i == SEQ_LENGTH + 1 else 0 for i in range(SEQ_LENGTH + 3)]
            elif TASK_TAG in ["sort", "palindrome"]:
                ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(SEQ_LENGTH * 2)]
                mask = None
            val_data = pickle.dumps([ids, mask])
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
        if TASK_TAG == "q":
            for _ in range(10000):
                i = random.randint(0, 10 ** SEQ_LENGTH)
                istr = f"---------------------------{str(i)}"[-SEQ_LENGTH:]
                prompt = f"{istr}-"
                x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, ...]
                y = model.generate_next_token(x, temperature=1, top_k=1)
                qval = tokenizer.decode(y[0].tolist())
                label = "×"
                total_count += 1
                if qval == q_function(i, SEQ_LENGTH):
                    ok_count += 1
                    label = "√"
                print(f"({int(ok_count / total_count * 100)}%) [{label}] {prompt}{qval}")
        elif TASK_TAG == "sort":
            for _ in range(0, 100000):
                n = random.randint(0, 10 ** SEQ_LENGTH)
                input_seq = f"{n + 10 ** SEQ_LENGTH}"[1:]
                target_seq = "".join(sorted(list(input_seq)))
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
        elif TASK_TAG == "palindrome":
            for _ in range(0, 100000):
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

    trainer = TrainGPT(model_config, train_config, max_steps=MAX_STEPS, ckpt_filename=CHECKPOINT_FILE_NAME)
    trainer.start()

    inference(checkpoint_path=f"checkpoint/{CHECKPOINT_FILE_NAME}")

if __name__ == "__main__":
    main()
