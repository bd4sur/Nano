import os
import random
import pickle
import base64
import json
import numpy as np
from tqdm import tqdm
from tokenizer import Tokenizer



def build_tokenizer(input_file, data_dir="dataset", is_build_tokenizer=True):
    tokenizer_path = os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.json')
    tokenizer = Tokenizer()
    if is_build_tokenizer:
        input_file_path = os.path.join(os.path.dirname(__file__), data_dir, input_file)
        print(f"Building tokenizer from raw text file...")
        tokenizer.build_from_file(input_file_path, tokenizer_path)
    else:
        print(f"Loading tokenizer...")
        tokenizer.load_from_config_file(tokenizer_path)
    return tokenizer

def generate_pretrain_dataset(input_file, data_dir="dataset", tokenizer=None, block_size=512, overlap_ratio=0.5):
    input_file_path = os.path.join(os.path.dirname(__file__), data_dir, input_file)

    print(f"Reading and encoding raw text file...")
    def read_chunk(filepath, chunk_size=65536):
        with open(filepath, mode="r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                yield chunk

    all_tokens = []
    text_iterator = read_chunk(input_file_path, chunk_size=16777216)
    for chunk in tqdm(text_iterator):
        tokens = tokenizer.encode(chunk)
        all_tokens.extend(tokens)
    print(f"  Length of tokens: {len(all_tokens):,}")

    # print(f"Encoding full text...")
    # all_tokens = tokenizer.encode(fulltext)

    print(f"Slicing all tokens into blocks...")
    blocks = []
    for i in tqdm(range(0, len(all_tokens), int(block_size * overlap_ratio))):
        tslice = all_tokens[i : i + block_size + 1]
        if len(tslice) == block_size + 1:
            blocks.append(tslice) # 每一条数据都比block_size多一个，用于预测下一字符的训练

    del all_tokens

    # 巨大数据集的打乱算法

    print(f"Shuffling text blocks and write to file...")
    line_indexes = list(range(len(blocks)))
    random.shuffle(line_indexes)

    train_path = os.path.join(os.path.dirname(__file__), data_dir, 'pretrain_train.base64')
    val_path   = os.path.join(os.path.dirname(__file__), data_dir, 'pretrain_val.base64')

    with open(train_path, "w", encoding="utf-8") as f_train:
        for li in tqdm(range(0, int(len(blocks) * 0.9))):
            train_block = pickle.dumps([blocks[line_indexes[li]], None])
            f_train.writelines(str(base64.b64encode(train_block), encoding="utf-8") + "\n")

    with open(val_path, "w", encoding="utf-8") as f_val:
        for li in tqdm(range(int(len(blocks) * 0.9), len(blocks))):
            val_block = pickle.dumps([blocks[line_indexes[li]], None])
            f_val.writelines(str(base64.b64encode(val_block), encoding="utf-8") + "\n")

    print(f"Done.")

def generate_sft_dataset(input_file, data_dir="dataset", tokenizer=None, block_size=512):
    input_file_path = os.path.join(os.path.dirname(__file__), data_dir, input_file)
    all_lines = []
    all_masks = []
    current_question = ""
    with open(input_file_path, mode="r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("[Q]"):
                current_question = line[3:]
            elif line.startswith("[A]"):
                answer = line[3:]
                # answer = "人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！人类的本质是复读机！"
                item = f"{tokenizer.instruct_mark_char}{current_question}{tokenizer.response_mark_char}{answer}\u1337\u1337"
                all_lines.append(item[0: block_size + 1])
                mask = [0] * (1 + len(current_question) + 1) + [1] * (len(answer) + 2)
                all_masks.append(mask[0: block_size + 1])
                current_question = ""

    train_path = os.path.join(os.path.dirname(__file__), data_dir, 'sft_train.base64')
    val_path   = os.path.join(os.path.dirname(__file__), data_dir, 'sft_val.base64')

    print(f"Shuffling sft blocks and write to file ...")
    line_indexes = list(range(len(all_lines)))
    random.shuffle(line_indexes)

    with open(train_path, "w", encoding="utf-8") as f_train:
        for li in tqdm(range(0, int(len(all_lines) * 0.9))):
            ids = tokenizer.encode(all_lines[line_indexes[li]])
            ids = [ids[i] if i < len(ids) else tokenizer.padding_token for i in range(block_size + 1)]
            mask = all_masks[line_indexes[li]]
            mask = [mask[i] if i < len(mask) else 0 for i in range(block_size + 1)]
            train_data = pickle.dumps([ids, mask])
            f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")

    with open(val_path, "w", encoding="utf-8") as f_val:
        for li in tqdm(range(int(len(all_lines) * 0.9), len(all_lines))):
            ids = tokenizer.encode(all_lines[line_indexes[li]])
            ids = [ids[i] if i < len(ids) else tokenizer.padding_token for i in range(block_size + 1)]
            mask = all_masks[line_indexes[li]]
            mask = [mask[i] if i < len(mask) else 0 for i in range(block_size + 1)]
            val_data = pickle.dumps([ids, mask])
            f_val.writelines(str(base64.b64encode(val_data), encoding="utf-8") + "\n")

    print(f"Done.")


def main():
    BLOCK_SIZE = 256
    PRETRAIN_DATASET = "pretrain-amateur-radio.txt"
    SFT_DATASET = "sft-amateur-radio.txt"
    tokenizer = build_tokenizer(PRETRAIN_DATASET, data_dir="dataset", is_build_tokenizer=True)
    generate_pretrain_dataset(PRETRAIN_DATASET, data_dir="dataset", tokenizer=tokenizer, block_size=BLOCK_SIZE, overlap_ratio=0.5)
    generate_sft_dataset(SFT_DATASET, data_dir="dataset", tokenizer=tokenizer, block_size=BLOCK_SIZE)

if __name__ == "__main__":
    main()
