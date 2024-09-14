import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from tokenizer import Tokenizer

def generate_nlg_dataset(input_file, data_dir="dataset", is_build_tokenizer=True, block_size=512, overlap_ratio=0.5):
    input_file_path = os.path.join(os.path.dirname(__file__), data_dir, input_file)
    tokenizer_path = os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.json')

    tokenizer = Tokenizer()
    if is_build_tokenizer:
        print(f"Building tokenizer from raw text file...")
        tokenizer.build_from_file(input_file_path, tokenizer_path)
    else:
        print(f"Loading tokenizer...")
        tokenizer.load_from_config(tokenizer_path)

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

    print(f"Shuffling text blocks...")
    train_ids = []
    val_ids = []
    line_indexes = list(range(len(blocks)))
    random.shuffle(line_indexes)
    for li in tqdm(range(0, int(len(blocks) * 0.9))):
        train_ids.append(blocks[line_indexes[li]])
    for li in tqdm(range(int(len(blocks) * 0.9), len(blocks))):
        val_ids.append(blocks[line_indexes[li]])

    print(f"Cast to numpy array...")
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    print(f"Saving pickel file...")
    dataset = {
        "train_ids": train_ids,
        "val_ids": val_ids
    }
    with open(os.path.join(os.path.dirname(__file__), data_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Done.")

def main():
    generate_nlg_dataset("input.txt", data_dir="dataset", block_size=512, overlap_ratio=0.5)

if __name__ == "__main__":
    main()
