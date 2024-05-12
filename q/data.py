import os
import pickle
import numpy as np
from tqdm import tqdm
from qfunc import q_function

def generate(data_dir="data_q"):
    os.makedirs(os.path.join(os.path.dirname(__file__), data_dir), exist_ok=True)

    text = []
    for i in tqdm(range(100000000)):
        line = f"{i + 100000000}-{q_function(i, 8)}"[1:]
        text.append(line)

    fulltext = "\n".join(text)
    print(f"length of dataset in characters: {len(fulltext):,}")

    chars = sorted(list(set(fulltext)))
    vocab_size = len(chars)
    # print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    idx = int(len(fulltext) * 0.9 / 11) * 11 # 对齐一行的边界位置
    train_ids = encode(fulltext[ : idx])
    val_ids = encode(fulltext[idx : ])
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    dataset = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        "train_ids": train_ids,
        "val_ids": val_ids
    }
    with open(os.path.join(os.path.dirname(__file__), data_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    tokenizer = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

def main():
    generate("data_q")

if __name__ == "__main__":
    main()
