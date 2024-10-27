import os
import time
import readline
import argparse
import torch
from tokenizer import Tokenizer
from model import GPT

class InferenceGPT:

    def __init__(
            self,
            checkpoint_path=os.path.join(os.path.dirname(__file__), "checkpoint/checkpoint.pt"),
            device="cuda",
            is_instruct=True,
            max_length=None,
            temperature=1.0,
            top_k=5,
            repetition_penalty=1.2,
            profile=False
        ):
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.model = None
        self.tokenizer = None
        self.encode = None
        self.decode = None

        self.is_instruct = is_instruct
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        # 读取模型检查点和训练配置
        print(f"Loading model from `{self.checkpoint_path}`...")
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        train_config = checkpoint['train_config']
        model_config = checkpoint['model_config']
        tokenizer_config = checkpoint['tokenizer_config']

        self.max_length = model_config.block_size if self.max_length is None or self.max_length > model_config.block_size else self.max_length

        # 设置随机种子与训练设置一致
        torch.manual_seed(train_config.random_seed)
        torch.cuda.manual_seed(train_config.random_seed)

        # 加载模型权重
        self.model = GPT(model_config)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()
        self.model.to(device)

        print(f"╭───────────┬───────────┬────────┬──────┬─────────┬──────────┬────────┬──────────────────╮")
        print(f"│ \x1b[1mBlockSize │ VocabSize │ Layers │ Embd │ Q_Heads │ KV_Heads │ Hidden │    Parameters\x1b[0m    │")
        print(f"├───────────┼───────────┼────────┼──────┼─────────┼──────────┼────────┼──────────────────┤")
        print(f"│{'{:^11d}'.format(model_config.block_size, end='')}│{'{:^11d}'.format(model_config.vocab_size, end='')}│{'{:^8d}'.format(model_config.n_layer, end='')}│{'{:^6d}'.format(model_config.n_embd, end='')}│{'{:^9d}'.format(model_config.n_head, end='')}│{'{:^10d}'.format(model_config.n_kv_head, end='')}│{'{:^8d}'.format(model_config.n_hidden, end='')}│{'{:^18,d}'.format(self.model.get_num_params(), end='')}│")
        print(f"╰───────────┴───────────┴────────┴──────┴─────────┴──────────┴────────┴──────────────────╯")

        # 读取分词器
        self.tokenizer = Tokenizer()
        self.tokenizer.load_from_config_dict(tokenizer_config)
        self.encode = lambda s: self.tokenizer.encode(s)
        self.decode = lambda l: self.tokenizer.decode(l)

        # 性能计时器
        self.profile = profile
        self.token_count = 0
        self.times = []
        self.tps_record = []

    def measure(self, interval):
        t = time.time_ns()
        self.times.append(t)
        if len(self.times) > 4:
            self.times = self.times[1:]
            t_avg = (self.times[3] - self.times[0]) / 3
            return interval / (t_avg / 1e9)
        else:
            return False

    def typewriter(self, token_tensor):
        token_list = token_tensor[0].tolist()
        chars = self.decode(token_list)
        self.token_count = self.token_count + 1
        if self.profile and self.token_count % 4 == 0:
            tps = self.measure(4)
            if tps:
                self.tps_record.append(tps)
        if "<|eos|>" in chars:
            print(chars.split("<|eos|>")[0], end="", flush=True)
            return False
        elif "<|padding|>" in chars:
            print(chars.replace("<|padding|>", ""), end="", flush=True)
            return False
        else:
            print(chars, end="", flush=True)
            return True

    def run(self):
        with torch.no_grad():
            while True:
                try:
                    prompt = input("\x1b[32;1mHomo:\x1b[0m ")
                except EOFError:
                    break
                if self.is_instruct:
                    prompt = f"<|instruct_mark|>{prompt}<|response_mark|>"
                x = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)[None, ...]
                print("\x1b[34;1mNano:\x1b[0m ", end="", flush=True)
                y = self.model.auto_regressive_generate(x, self.max_length, self.temperature, self.top_k, self.repetition_penalty, callback=self.typewriter)
                print("\n")
                if self.profile:
                    print(f"TPS = {[round(tps) for tps in self.tps_record]}\n")
                    self.times = []
                    self.tps_record = []
                    self.token_count = 0

def main():
    print(f"\n\x1b[36;1mNano Language Model\x1b[0m - https://github.com/bd4sur/Nano")

    parser = argparse.ArgumentParser(description="Sample (to inference) from Nano model for text generation and question answering.")
    parser.add_argument("-m", "--model", type=str, default="checkpoint/ckpt.pt")
    parser.add_argument("-i", "--instruct", action="store_true")
    parser.add_argument("-l", "--max_length", type=int, default=None)
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("-k", "--top_k", type=int, default=5)
    parser.add_argument("-r", "--repetition_penalty", type=float, default=1.2)
    parser.add_argument("-p", "--profile", action='store_true')
    args = parser.parse_args()

    infer = InferenceGPT(
        args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        is_instruct=args.instruct,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        profile=args.profile
    )
    infer.run()

if __name__ == "__main__":
    main()
