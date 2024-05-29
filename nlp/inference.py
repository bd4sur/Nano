import os
import random
import torch
from tokenizer import Tokenizer
from model import GPT
from qfunc import q_function, q_digits

class InferenceGPT:

    def __init__(self, data_dir="dataset", ckpt_dir="checkpoint", device="cuda"):
        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.device = device

        # 读取模型检查点和训练配置
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_dir, 'ckpt.pt')
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint['config']

        # 设置随机种子与训练设置一致
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed(config.random_seed)

        # 加载模型权重
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.to(device)

        # 读取分词器
        tokenizer_path = os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.json')
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer()
        self.tokenizer.load_from_config(tokenizer_path)

    def typewriter(self, token_tensor):
        print(self.tokenizer.decode(token_tensor[0].tolist()), end="", flush=True)


    def inference_q(self):
        with torch.no_grad():
            ok_count = 0
            total_count = 0
            label = ""
            qdigits = q_digits()
            for i in range(99900000, 99999999):
                n = random.randint(0, 10 ** qdigits)
                prompt = f"{n + 10 ** qdigits}-"[1:]
                x = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device)[None, ...]
                y = self.model.predict_next_token(x, temperature=1, top_k=1)
                qval = self.tokenizer.decode(y[0].tolist())
                label = "×"
                total_count += 1
                if qval == q_function(n, qdigits):
                    ok_count += 1
                    label = "√"
                print(f"({int(ok_count / total_count * 100)}%) [{label}] {prompt}{qval}")


    def inference_sorting(self):
        with torch.no_grad():
            ok_count = 0
            total_count = 0
            label = ""
            qdigits = q_digits()
            for i in range(0, 100000):
                n = random.randint(0, 10 ** qdigits)
                input_seq = f"{n + 10 ** qdigits}"[1:]
                target_seq = "".join(sorted(list(input_seq)))
                x = torch.tensor(self.tokenizer.encode(input_seq), dtype=torch.long, device=self.device)[None, ...]
                y = self.model.generate_sequence(x, temperature=1, top_k=1)
                output_list = []
                for t in range(len(y)):
                    output_list.append(self.tokenizer.decode(y[t][0].tolist()))
                output_seq = "".join(output_list)
                label = "×"
                total_count += 1
                if target_seq == output_seq:
                    ok_count += 1
                    label = "√"
                print(f"({int(ok_count / total_count * 100)}%) [{label}] {input_seq} - {output_seq}")


    def inference_nlg(self):
        with torch.no_grad():
            while True:
                try:
                    prompt = input("Prompt: ")
                except EOFError:
                    break
                x = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device)[None, ...]
                print(prompt, end="", flush=True)
                y = self.model.generate(x, 200, temperature=1, top_k=10, callback=self.typewriter)
                print("\n")

def main():
    infer = InferenceGPT("dataset", "checkpoint", "cuda")
    infer.inference_nlg()

if __name__ == "__main__":
    main()
