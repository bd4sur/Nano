import os
import base64
import random
import pickle
import torch
from tqdm import tqdm
from tokenizer import Tokenizer
from model import GPT
from train import TrainGPT

###################################################
# 注意：先将`model.py`中的 `USE_KV_CACHE` 设为 False

# "q", "sort", "palindrome", "calculator"
TASK_TAG = "calculator"

MAX_STEPS = 500
SEQ_LENGTH = 6

MIN_NUMBER = 0
MAX_NUMBER = 1
EXPR_MAX_DEPTH = 4
EXPR_MAX_LENGTH = 64

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(ROOT_PATH, "dataset_preprocessed"), exist_ok=True)

CHECKPOINT_FILE_NAME = f"problem_{TASK_TAG}.pt"
TRAINSET_PATH        = f"{ROOT_PATH}/dataset_preprocessed/problem_{TASK_TAG}_train.base64"
VALSET_PATH          = f"{ROOT_PATH}/dataset_preprocessed/problem_{TASK_TAG}_val.base64"
TOKENIZER_PATH       = f"{ROOT_PATH}/dataset_preprocessed/problem_{TASK_TAG}_tokenizer.json"


model_config = {
    "block_size": SEQ_LENGTH,
    "vocab_size": 100,
    "n_layer": 2,
    "n_embd": 32,
    "n_head": 2,
    "n_kv_head": 2,
    "n_hidden": 16,
    "dropout": 0.0,
    "use_rope": False,
    "norm_eps": 1e-5,
    "is_causal": False,
}

train_config = {
    "from_checkpoint": "",
    "save_checkpoint_to": f"{ROOT_PATH}/checkpoint",
    "dataset_path": [
        [TRAINSET_PATH, VALSET_PATH]
    ],
    "tokenizer_path": TOKENIZER_PATH,

    "random_seed": 39,
    "batch_size": 100,
    "gradient_accumulation_steps": 1,
    "grad_clip": 1.0,

    "dropout": 0.0,

    "learning_rate": 1e-3,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,

    "decay_lr": True,
    "warmup_iters": int(MAX_STEPS * 0.3),
    "lr_decay_iters": MAX_STEPS,
    "min_lr": 6e-5,

    "eval_interval": 100,
    "log_interval": 10,
    "eval_iters": 5,

    "backend": "nccl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sdp_kernel": "flash",
    "dtype": "bfloat16",
    "use_amp": True
}

if TASK_TAG == "q":
    model_config["block_size"] = SEQ_LENGTH + 2
    model_config["n_layer"]    = 2
    model_config["n_embd"]     = 64
    model_config["n_head"]     = 2
    model_config["n_kv_head"]  = 2
    model_config["n_hidden"]   = 32
    model_config["use_rope"]   = True
    model_config["is_causal"]  = True

elif TASK_TAG == "sort":
    model_config["block_size"] = SEQ_LENGTH
    model_config["n_layer"]    = 2
    model_config["n_embd"]     = 32
    model_config["n_head"]     = 2
    model_config["n_kv_head"]  = 2
    model_config["n_hidden"]   = 16
    model_config["use_rope"]   = False
    model_config["is_causal"]  = False

elif TASK_TAG == "palindrome":
    model_config["block_size"] = SEQ_LENGTH
    model_config["n_layer"]    = 2
    model_config["n_embd"]     = 32
    model_config["n_head"]     = 2
    model_config["n_kv_head"]  = 2
    model_config["n_hidden"]   = 16
    model_config["use_rope"]   = False
    model_config["is_causal"]  = False

elif TASK_TAG == "calculator":
    model_config["block_size"] = EXPR_MAX_LENGTH
    model_config["n_layer"]    = 4
    model_config["n_embd"]     = 2048
    model_config["n_head"]     = 256
    model_config["n_kv_head"]  = 64
    model_config["n_hidden"]   = 4096
    model_config["use_rope"]   = False
    model_config["is_causal"]  = True






def decode_to_expr(tokens, tokenizer):
    expr_str = ""
    following_lb = False
    for ti in tokens:
        tc = tokenizer.itos[ti]
        if tc == ")":
            expr_str += ")"
        elif tc == "(":
            expr_str += (" " + tc)
            following_lb = True
        elif following_lb:
            expr_str += tc
            following_lb = False
        else:
            expr_str += (" " + tc)
    return expr_str.strip()

def get_calculator_tokenizer(min_number, max_number):
    tokenizer_config = {
        "vocab_size": 0,
        "stoi": {"<|padding|>": 0, "<|unknown|>": 1, "<|bos|>": 2, "<|eos|>": 3, "<|instruct_mark|>": 4, "<|response_mark|>": 5, "BD4SUR": 6, },
        "itos": [
            "<|padding|>", "<|unknown|>", "<|bos|>", "<|eos|>", "<|instruct_mark|>", "<|response_mark|>", "BD4SUR",
            "inf", "(", ")", "+", "-", "*", "/", "=", ],
        "special_tokens": {"<|padding|>": 0, "<|unknown|>": 1, "<|bos|>": 2, "<|eos|>": 3, "<|instruct_mark|>": 4, "<|response_mark|>": 5, "BD4SUR": 6}}
    for i in range(min_number, max_number+1):
        tokenizer_config["itos"].append(str(i))
    for index,tk in enumerate(tokenizer_config["itos"]):
        tokenizer_config["stoi"][tk] = index
    tokenizer_config["vocab_size"] = len(tokenizer_config["itos"])
    model_config["vocab_size"] = tokenizer_config["vocab_size"]
    tk = Tokenizer()
    tk.load_from_config_dict(tokenizer_config)
    return tk

def gen_expr(expr_depth, tokenizer):
    p = random.random()
    # number
    if p <= 0.2 or expr_depth >= EXPR_MAX_DEPTH:
        value = random.randint(0, 1)
        # value = random.randint(-1, 1)
        return [[tokenizer.stoi[str(value)]], value]
    # expr
    else:
        op = ["+", "*", "-", "/"][random.randint(0, 1)]
        arg1 = gen_expr(expr_depth + 1, tokenizer)
        arg2 = gen_expr(expr_depth + 1, tokenizer)
        expr = [tokenizer.stoi["("], tokenizer.stoi[op]] + arg1[0] + arg2[0] + [tokenizer.stoi[")"]]
        if op == "+":
            # value = "inf" if arg1[1] == "inf" or arg2[1] == "inf" else arg1[1] + arg2[1]
            value = 1 if (arg1[1] == 1) or (arg2[1] == 1) else 0
        # elif op == "-":
            # value = "inf" if arg1[1] == "inf" or arg2[1] == "inf" else arg1[1] - arg2[1]
        elif op == "*":
            # value = "inf" if arg1[1] == "inf" or arg2[1] == "inf" else arg1[1] * arg2[1]
            value = 1 if (arg1[1] == 1) and (arg2[1] == 1) else 0
        # elif op == "/":
        #     value = "inf" if arg1[1] == "inf" or arg2[1] == "inf" or arg2[1] == 0 else int(arg1[1] / arg2[1])
        # if value != "inf" and (value > MAX_NUMBER or value < MIN_NUMBER):
        #     value = "inf"
        return [expr, value]



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


    if TASK_TAG in ["q", "sort", "palindrome"]:
        print(f"Building tokenizer...")
        tokenizer = Tokenizer()
        tokenizer.build_from_text(fulltext, TOKENIZER_PATH)

        train_path = TRAINSET_PATH
        val_path   = VALSET_PATH

        print(f"Shuffling sft blocks and write to file ...")
        line_indexes = list(range(len(all_items)))
        random.shuffle(line_indexes)

        with open(train_path, "w", encoding="utf-8") as f_train:
            for li in tqdm(range(0, int(len(all_items) * 0.4))):
                ids = tokenizer.encode(all_items[line_indexes[li]])
                if TASK_TAG == "q":
                    ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(SEQ_LENGTH + 2 + 1)]
                    mask = [1 if i == SEQ_LENGTH + 1 else 0 for i in range(SEQ_LENGTH + 2 + 1)]
                elif TASK_TAG in ["sort", "palindrome"]:
                    ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(SEQ_LENGTH * 2)]
                    mask = None
                train_data = pickle.dumps([ids, mask])
                f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")

        with open(val_path, "w", encoding="utf-8") as f_val:
            for li in tqdm(range(int(len(all_items) * 0.4), len(all_items))):
                ids = tokenizer.encode(all_items[line_indexes[li]])
                if TASK_TAG == "q":
                    ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(SEQ_LENGTH + 2 + 1)]
                    mask = [1 if i == SEQ_LENGTH + 1 else 0 for i in range(SEQ_LENGTH + 2 + 1)]
                elif TASK_TAG in ["sort", "palindrome"]:
                    ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(SEQ_LENGTH * 2)]
                    mask = None
                val_data = pickle.dumps([ids, mask])
                f_val.writelines(str(base64.b64encode(val_data), encoding="utf-8") + "\n")

    elif TASK_TAG == "calculator":
        print(f"Generating S-expr ...")

        train_path = TRAINSET_PATH
        val_path   = VALSET_PATH

        tk = get_calculator_tokenizer(MIN_NUMBER, MAX_NUMBER)
        tk.dump_config_file(TOKENIZER_PATH)

        with open(train_path, "w", encoding="utf-8") as f_train:
            for _ in tqdm(range(MAX_STEPS * train_config["batch_size"])):
                expr = gen_expr(0, tk)
                equation_tokens = expr[0] + [tk.stoi["="], tk.stoi[str(expr[1])], tk.special_tokens["<|eos|>"]]
                # print(equation_tokens)
                # print(decode_to_expr(equation_tokens, tk))
                equation_tokens = [equation_tokens[i] if i < len(equation_tokens) else tk.special_tokens["<|padding|>"] for i in range(EXPR_MAX_LENGTH + 1)]
                mask = [1 if i == len(expr[0]) + 1 or i == len(expr[0]) + 2 else 0 for i in range(EXPR_MAX_LENGTH + 1)]
                train_data = pickle.dumps([equation_tokens, mask])
                f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")
        with open(val_path, "w", encoding="utf-8") as f_val:
            for _ in tqdm(range(10000)):
                expr = gen_expr(0, tk)
                equation_tokens = expr[0] + [tk.stoi["="], tk.stoi[str(expr[1])], tk.special_tokens["<|eos|>"]]
                # print(equation_tokens)
                # print(decode_to_expr(equation_tokens, tk))
                equation_tokens = [equation_tokens[i] if i < len(equation_tokens) else tk.special_tokens["<|padding|>"] for i in range(EXPR_MAX_LENGTH + 1)]
                mask = [1 if i == len(expr[0]) + 1 or i == len(expr[0]) + 2 else 0 for i in range(EXPR_MAX_LENGTH + 1)]
                val_data = pickle.dumps([equation_tokens, mask])
                f_val.writelines(str(base64.b64encode(val_data), encoding="utf-8") + "\n")

    print(f"Done.")

def inference(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取模型检查点和训练配置
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
        elif TASK_TAG == "calculator":
            for _ in range(1000):
                expr = gen_expr(0, tokenizer)
                expr_str = decode_to_expr(expr[0], tokenizer)
                x = torch.tensor(expr[0] + [tokenizer.stoi["="]], dtype=torch.long, device=device)[None, ...]
                y = model.generate_next_token(x, temperature=1)
                qval = tokenizer.decode(y[0].tolist())
                label = "×"
                total_count += 1
                if qval == str(expr[1]):
                    ok_count += 1
                    label = "√"
                print(f"({int(ok_count / total_count * 100)}%) [{label}] {expr_str} = {expr[1]} | {qval}")
def main():
    print(f"PyTorch version: {torch.__version__}")

    generate_dataset()

    trainer = TrainGPT(model_config, train_config, max_steps=MAX_STEPS, ckpt_filename=CHECKPOINT_FILE_NAME)
    trainer.start()

    inference(checkpoint_path=f"{ROOT_PATH}/checkpoint/{CHECKPOINT_FILE_NAME}")

if __name__ == "__main__":
    main()
