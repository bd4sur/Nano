import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from qfunc import q_function, q_digits

def generate_text(input_file, data_dir="data_q", block_size=512, overlap_ratio=0.5):
    input_file_path = os.path.join(os.path.dirname(__file__), data_dir, input_file)
    with open(input_file_path, "r", encoding="utf-8") as f:
        fulltext = f.read()
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

    text_slices = []
    i = 0
    while i < len(fulltext):
        tslice = fulltext[i : i + block_size + 1]
        if len(tslice) == block_size + 1:
            text_slices.append(tslice) # 每一条数据都比block_size多一个，用于预测下一字符的训练
        i += int(block_size * overlap_ratio)

    train_ids = []
    val_ids = []

    line_indexes = list(range(len(text_slices)))
    random.shuffle(line_indexes)
    for li in tqdm(range(0, int(len(text_slices) * 0.8))):
        train_ids.append(encode(text_slices[line_indexes[li]]))
    for li in tqdm(range(int(len(text_slices) * 0.2), len(text_slices))):
        val_ids.append(encode(text_slices[line_indexes[li]]))

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



def generate_problem_q(data_dir="data_q"):
    os.makedirs(os.path.join(os.path.dirname(__file__), data_dir), exist_ok=True)

    qdigits = q_digits()
    text = []
    for i in tqdm(range(10 ** qdigits)):
        line = f"{i + 10 ** qdigits}-{q_function(i, qdigits)}"[1:]
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

    train_ids = []
    val_ids = []

    line_indexes = list(range(len(text)))
    random.shuffle(line_indexes)
    for li in tqdm(range(0, int(len(text) * 0.4))):
        train_ids.append(encode(text[line_indexes[li]]))
    for li in tqdm(range(int(len(text) * 0.6), len(text))):
        val_ids.append(encode(text[line_indexes[li]]))

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



def generate_sorting(data_dir="data_q"):
    os.makedirs(os.path.join(os.path.dirname(__file__), data_dir), exist_ok=True)

    qdigits = q_digits()
    text = []
    for i in tqdm(range(10 ** qdigits)):
        origin_str = f"{i + 10 ** qdigits}"[1:]
        sorted_str = "".join(sorted(list(origin_str)))
        line = f"{origin_str}{sorted_str}"
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

    train_ids = []
    val_ids = []

    line_indexes = list(range(len(text)))
    random.shuffle(line_indexes)
    for li in tqdm(range(0, int(len(text) * 0.4))):
        train_ids.append(encode(text[line_indexes[li]]))
    for li in tqdm(range(int(len(text) * 0.6), len(text))):
        val_ids.append(encode(text[line_indexes[li]]))

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
    # generate_text("psycho.txt", data_dir="data_q", block_size=128, overlap_ratio=0.1)
    generate_problem_q("data_q")
    # generate_sorting("data_q")

if __name__ == "__main__":
    main()
