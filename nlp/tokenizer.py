import json

class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    # encoder: take a string, output a list of integers
    def encode(self, text):
        return [self.stoi[c] for c in text]

    # decoder: take a list of integers, output a string
    def decode(self, code_list):
        return ''.join([self.itos[i] for i in code_list])

    def load_from_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            self.vocab_size = config["vocab_size"]
            self.stoi = config["stoi"]
            self.itos = { int(i):config["itos"][i] for i in config["itos"] }

    # 根据已有文本建立编码器，并保存到配置文件
    def build_from_text(self, text, config_path):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(f"  Vocab size: {self.vocab_size:,}")
        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        config = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
