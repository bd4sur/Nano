import os
import random
import tiktoken
import torch
from tokenizer import Tokenizer
from model import GPT

class InferenceGPT:

    def __init__(self, data_dir="dataset", ckpt_dir="checkpoint", device="cuda"):
        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.device = device

        self.model = None
        self.tokenizer = None
        self.encode = None
        self.decode = None

        if ckpt_dir in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}:
            # 加载GPT模型权重
            self.model = GPT.from_pretrained(ckpt_dir)
            self.model.eval()
            self.model.to(device)
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: self.tokenizer.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: self.tokenizer.decode(l)

            _checkpoint = {
                "model":      self.model.state_dict(),
                "optimizer":  None,
                "iter_count": 0,
                "config":     self.model.config,
            }
            torch.save(_checkpoint, os.path.join(os.path.dirname(__file__), 'gpt2-nano-ckpt.pt'))

        else:
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
            self.encode = lambda s: self.tokenizer.encode(s)
            self.decode = lambda l: self.tokenizer.decode(l)

    def typewriter(self, token_tensor):
        print(self.decode(token_tensor[0].tolist()), end="", flush=True)

    def generate(self):
        with torch.no_grad():
            while True:
                try:
                    prompt = input("Prompt: ")
                except EOFError:
                    break
                x = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)[None, ...]
                print(prompt, end="", flush=True)
                y = self.model.generate(x, 200, temperature=1, top_k=10, callback=self.typewriter)
                print("\n")

def main():
    # infer = InferenceGPT("dataset", "gpt2", "cuda")
    infer = InferenceGPT("dataset", "checkpoint", "cuda")
    infer.generate()

if __name__ == "__main__":
    main()
