import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from tokenizer import Tokenizer

def generate_nlg_dataset(input_file, data_dir="dataset", is_build_tokenizer=True, block_size=512, overlap_ratio=0.5):
    print(f"Reading raw text file...")
    input_file_path = os.path.join(os.path.dirname(__file__), data_dir, input_file)
    with open(input_file_path, "r", encoding="utf-8") as f:
        fulltext = f.read()
    print(f"  Length of dataset in characters: {len(fulltext):,}")

    tokenizer_path = os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.json')
    tokenizer = Tokenizer()
    if is_build_tokenizer:
        print(f"Building tokenizer from raw text...")
        tokenizer.build_from_text(fulltext, tokenizer_path)
    else:
        print(f"Loading tokenizer...")
        tokenizer.load_from_config(tokenizer_path)

    print(f"Slicing raw text into blocks...")
    text_slices = []
    i = 0
    while i < len(fulltext):
        tslice = fulltext[i : i + block_size + 1]
        if len(tslice) == block_size + 1:
            text_slices.append(tslice) # 每一条数据都比block_size多一个，用于预测下一字符的训练
        i += int(block_size * overlap_ratio)

    print(f"Shuffling and encoding text blocks...")
    train_ids = []
    val_ids = []
    line_indexes = list(range(len(text_slices)))
    random.shuffle(line_indexes)
    for li in tqdm(range(0, int(len(text_slices) * 0.9))):
        train_ids.append(tokenizer.encode(text_slices[line_indexes[li]]))
    for li in tqdm(range(int(len(text_slices) * 0.9), len(text_slices))):
        val_ids.append(tokenizer.encode(text_slices[line_indexes[li]]))

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    dataset = {
        "train_ids": train_ids,
        "val_ids": val_ids
    }
    with open(os.path.join(os.path.dirname(__file__), data_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Done.")

def main():
    generate_nlg_dataset("psycho.txt", data_dir="dataset", block_size=512, overlap_ratio=0.1)

if __name__ == "__main__":
    main()
