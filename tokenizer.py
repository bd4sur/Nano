"""
Forked from https://github.com/karpathy/nanoGPT
BD4SUR 2023.12
"""

import os
import pickle
import random
import numpy as np

# === configuation begin ==========================
data_dir = "data"
# === configuation end ============================

# Load raw corpus text file
input_file_path = os.path.join(os.path.dirname(__file__), data_dir, "input.txt")
with open(input_file_path, "r", encoding="utf-8") as f:
    fulltext = f.read()
print(f"length of dataset in characters: {len(fulltext):,}")

chars = sorted(list(set(fulltext)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# 按行切成句子，并按句子随机分成测试集和验证集
# 建议对数据集作预处理：一行一句
train_text = ""
val_text = ""
all_sentences = fulltext.split("\n")
line_indexes = list(range(len(all_sentences)))
random.shuffle(line_indexes)
for li in range(0, int(len(all_sentences) * 0.9)):
    train_text = train_text + all_sentences[line_indexes[li]]
for li in range(int(len(all_sentences) * 0.9), len(all_sentences)):
    val_text = val_text + all_sentences[line_indexes[li]]

# encode both to integers
train_ids = encode(train_text)
val_ids = encode(val_text)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), data_dir, 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), data_dir, 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
