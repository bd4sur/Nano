import os
import tiktoken
import torch
from tokenizer import Tokenizer
from model import GPT

class InferenceGPT:

    def __init__(self, tokenizer_path="dataset/tokenizer.json", checkpoint_path="checkpoint/ckpt.pt", device="cuda"):
        self.tokenizer_path = tokenizer_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.model = None
        self.tokenizer = None
        self.encode = None
        self.decode = None

        if self.checkpoint_path in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}:
            # 加载GPT模型权重
            self.model = GPT.from_pretrained(self.checkpoint_path)
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
            ckpt_path = os.path.join(os.path.dirname(__file__), self.checkpoint_path)
            print(f"Loading checkpoint from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            train_config = checkpoint['train_config']
            model_config = checkpoint['model_config']

            # 设置随机种子与训练设置一致
            torch.manual_seed(train_config.random_seed)
            torch.cuda.manual_seed(train_config.random_seed)

            # 加载模型权重
            self.model = GPT(model_config)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()
            self.model.to(device)

            # 读取分词器
            tk_path = os.path.join(os.path.dirname(__file__), self.tokenizer_path)
            print(f"Loading tokenizer from {tk_path}...")
            self.tokenizer = Tokenizer()
            self.tokenizer.load_from_config(tk_path)
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
                prompt = f"\u1337{prompt}\u1338\u1339"
                x = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)[None, ...]
                print(prompt, end="", flush=True)
                y = self.model.auto_regressive_generate(x, 200, temperature=1, top_k=10, callback=self.typewriter)
                print("\n")

def main():
    # infer = InferenceGPT("dataset", "gpt2", "cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer = InferenceGPT("dataset/tokenizer.json", "checkpoint/ckpt.pt", device)
    infer.generate()

if __name__ == "__main__":
    main()
