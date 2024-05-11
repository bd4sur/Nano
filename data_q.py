import os
import pickle
import random
import numpy as np
from tqdm import tqdm


NUM_DIGITS = 8

# Q函数：一串数字中有多少个圈儿
def q_function(number: int) -> int:
    """
    Q函数：一串数字中有多少个圈儿。
        例如：q(2024)=1，q(888)=6
        出典：https://www.zhihu.com/question/338618946/answer/831919337、https://www.zhihu.com/question/341026031/answer/841578656
    """
    #         0  1  2  3  4  5  6  7  8  9  10
    qv_map = [1, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0]
    res_map = "0123456789abcdefghijklmnopqrstuvwxyz"
    istr = f"---------------------------{str(number)}"[-NUM_DIGITS:]
    qv = 0
    for i in range(NUM_DIGITS):
        d = 10 if istr[i] == "-" else int(istr[i])
        qv = qv + qv_map[d]
    return res_map[qv]

def generate():
    text = []
    for i in tqdm(range(100000000)):
        line = f"{i + 100000000}-{q_function(i)}"[1:]
        text.append(line)
        # print(line)

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

    idx = int(len(fulltext) * 0.9 / 11) * 11
    train_ids = encode(fulltext[ : idx])
    val_ids = encode(fulltext[idx : ])
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), "data_q", 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), "data_q", 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), "data_q", 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

def main():
    generate()

if __name__ == "__main__":
    main()
