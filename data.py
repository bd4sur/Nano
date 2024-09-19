import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from tokenizer import Tokenizer



def build_tokenizer(input_file, data_dir="dataset", is_build_tokenizer=True):
    input_file_path = os.path.join(os.path.dirname(__file__), data_dir, input_file)
    tokenizer_path = os.path.join(os.path.dirname(__file__), data_dir, 'tokenizer.json')

    tokenizer = Tokenizer()
    if is_build_tokenizer:
        print(f"Building tokenizer from raw text file...")
        tokenizer.build_from_file(input_file_path, tokenizer_path)
    else:
        print(f"Loading tokenizer...")
        tokenizer.load_from_config(tokenizer_path)
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
    with open(os.path.join(os.path.dirname(__file__), data_dir, 'pretrain.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

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
                item = f"\u1341{current_question}\u1342{answer}\u1337"
                all_lines.append(item[0: block_size + 1])
                mask = [0] * (1 + len(current_question) + 1) + [1] * len(answer)
                all_masks.append(mask[0: block_size + 1])
                current_question = ""

    print(f"Shuffling sft blocks...")
    train_ids = []
    val_ids = []
    train_masks = []
    val_masks = []
    line_indexes = list(range(len(all_lines)))
    random.shuffle(line_indexes)
    for li in tqdm(range(0, int(len(all_lines) * 0.9))):
        ids = tokenizer.encode(all_lines[line_indexes[li]])
        ids = [ids[i] if i < len(ids) else tokenizer.padding_token for i in range(block_size + 1)]
        train_ids.append(ids)
        mask = all_masks[line_indexes[li]]
        mask = [mask[i] if i < len(mask) else 0 for i in range(block_size + 1)]
        train_masks.append(mask)
    for li in tqdm(range(int(len(all_lines) * 0.9), len(all_lines))):
        ids = tokenizer.encode(all_lines[line_indexes[li]])
        ids = [ids[i] if i < len(ids) else tokenizer.padding_token for i in range(block_size + 1)]
        val_ids.append(ids)
        mask = all_masks[line_indexes[li]]
        mask = [mask[i] if i < len(mask) else 0 for i in range(block_size + 1)]
        val_masks.append(mask)

    print(f"Cast to numpy array...")
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_masks = np.array(train_masks, dtype=np.uint16)
    val_masks = np.array(val_masks, dtype=np.uint16)

    print(f"Saving pickel file...")
    dataset = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_masks": train_masks,
        "val_masks": val_masks
    }
    with open(os.path.join(os.path.dirname(__file__), data_dir, 'sft.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Done.")







def main():
    BLOCK_SIZE = 512
    PRETRAIN_DATASET = "psycho.txt"
    SFT_DATASET = "sft.txt"
    tokenizer = build_tokenizer(PRETRAIN_DATASET, data_dir="dataset", is_build_tokenizer=True)
    generate_pretrain_dataset(PRETRAIN_DATASET, data_dir="dataset", tokenizer=tokenizer, block_size=BLOCK_SIZE, overlap_ratio=0.5)
    generate_sft_dataset(SFT_DATASET, data_dir="dataset", tokenizer=tokenizer, block_size=BLOCK_SIZE)

if __name__ == "__main__":
    main()
