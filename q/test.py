import os
import random
import pickle
import torch
from model import GPT
from qfunc import q_function, q_digits

data_dir = "data_q"
ckpt_dir = 'ckpt_q'
device = 'cuda'

def test(scene="q"):

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

    def typewriter(token_tensor):
        print(decode(token_tensor[0].tolist()), end="", flush=True)

    # 开始推理
    with torch.no_grad():
        if scene == "q":
            ok_count = 0
            total_count = 0
            label = ""
            qdigits = q_digits()
            for i in range(99900000, 99999999):
                n = random.randint(0, 10 ** qdigits)
                prompt = f"{n + 10 ** qdigits}-"[1:]
                x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
                y = model.predict_next_token(x, temperature=1, top_k=1)
                qval = decode(y[0].tolist())
                label = "×"
                total_count += 1
                if qval == q_function(n, qdigits):
                    ok_count += 1
                    label = "√"
                print(f"({int(ok_count / total_count * 100)}%) [{label}] {prompt}{qval}")
        elif scene == "sorting":
            ok_count = 0
            total_count = 0
            label = ""
            qdigits = q_digits()
            for i in range(0, 100000):
                n = random.randint(0, 10 ** qdigits)
                input_seq = f"{n + 10 ** qdigits}"[1:]
                target_seq = "".join(sorted(list(input_seq)))
                x = torch.tensor(encode(input_seq), dtype=torch.long, device=device)[None, ...]
                y = model.generate_sequence(x, temperature=1, top_k=1)
                output_list = []
                for t in range(len(y)):
                    output_list.append(decode(y[t][0].tolist()))
                output_seq = "".join(output_list)
                label = "×"
                total_count += 1
                if target_seq == output_seq:
                    ok_count += 1
                    label = "√"
                print(f"({int(ok_count / total_count * 100)}%) [{label}] {input_seq} - {output_seq}")
        else:
            while True:
                try:
                    prompt = input("Prompt: ")
                except EOFError:
                    break
                x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
                print(prompt, end="", flush=True)
                y = model.generate(x, 200, temperature=1, top_k=10, callback=typewriter)
                print("\n")

def main():
    test("")

if __name__ == "__main__":
    main()
