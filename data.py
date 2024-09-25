import os
import random
import pickle
import base64
import json
from tqdm import tqdm
from tokenizer import Tokenizer



def build_tokenizer(input_path_list, output_path, is_build_tokenizer=True):
    tokenizer = Tokenizer()
    if is_build_tokenizer:
        abs_paths = []
        for p in input_path_list:
            abs_paths.append(p)
        print(f"Building tokenizer from raw text file...")
        tokenizer.build_from_files(abs_paths, output_path)
    else:
        print(f"Loading tokenizer...")
        tokenizer.load_from_config_file(output_path)
    return tokenizer

def generate_pretrain_dataset(input_path, train_output_path, val_output_path, tokenizer=None, block_size=512, overlap_ratio=0.5):
    print(f"Reading and encoding raw text file...")
    def read_chunk(filepath, chunk_size=65536):
        with open(filepath, mode="r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                yield chunk

    all_tokens = []
    text_iterator = read_chunk(input_path, chunk_size=16777216)
    for chunk in tqdm(text_iterator):
        tokens = tokenizer.encode(chunk)
        all_tokens.extend(tokens)
    print(f"  Length of tokens: {len(all_tokens):,}")

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

    with open(train_output_path, "w", encoding="utf-8") as f_train:
        for li in tqdm(range(0, int(len(blocks) * 0.9))):
            train_block = pickle.dumps([blocks[line_indexes[li]], None])
            f_train.writelines(str(base64.b64encode(train_block), encoding="utf-8") + "\n")

    with open(val_output_path, "w", encoding="utf-8") as f_val:
        for li in tqdm(range(int(len(blocks) * 0.9), len(blocks))):
            val_block = pickle.dumps([blocks[line_indexes[li]], None])
            f_val.writelines(str(base64.b64encode(val_block), encoding="utf-8") + "\n")

    print(f"Done.")

def generate_sft_dataset(input_jsonl_path, train_output_path, val_output_path, tokenizer=None, block_size=512):
    all_lines = []
    all_masks = []
    with open(input_jsonl_path, mode="r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            qa = json.loads(line)
            question = qa["question"]
            answer = qa["answer"]
            item = f"<|instruct_mark|>{question}<|response_mark|>{answer}<|padding|><|padding|>"
            all_lines.append(item[0: block_size + 1])
            mask = [0] * (1 + len(question) + 1) + [1] * (len(answer) + 2)
            all_masks.append(mask[0: block_size + 1])

    print(f"Shuffling sft blocks and write to file ...")
    line_indexes = list(range(len(all_lines)))
    random.shuffle(line_indexes)

    with open(train_output_path, "w", encoding="utf-8") as f_train:
        for li in tqdm(range(0, int(len(all_lines) * 0.9))):
            ids = tokenizer.encode(all_lines[line_indexes[li]])
            ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(block_size + 1)]
            mask = all_masks[line_indexes[li]]
            mask = [mask[i] if i < len(mask) else 0 for i in range(block_size + 1)]
            train_data = pickle.dumps([ids, mask])
            f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")

    with open(val_output_path, "w", encoding="utf-8") as f_val:
        for li in tqdm(range(int(len(all_lines) * 0.9), len(all_lines))):
            ids = tokenizer.encode(all_lines[line_indexes[li]])
            ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(block_size + 1)]
            mask = all_masks[line_indexes[li]]
            mask = [mask[i] if i < len(mask) else 0 for i in range(block_size + 1)]
            val_data = pickle.dumps([ids, mask])
            f_val.writelines(str(base64.b64encode(val_data), encoding="utf-8") + "\n")

    print(f"Done.")


def main():

    BLOCK_SIZE = 256
    PRETRAIN_DATASETS = [
        # "dataset/pretrain-general.txt",
        "dataset/pretrain-chinese-classic.txt",
        "dataset/pretrain-psycho.txt",
        "dataset/pretrain-amateur-radio.txt",
    ]
    SFT_DATASET = "dataset/sft-id.jsonl"

    TOKENIZER_PATH = "dataset_preprocessed/tokenizer.json"

    base_path = os.path.dirname(__file__)

    tokenizer = build_tokenizer(
        PRETRAIN_DATASETS + [SFT_DATASET],
        os.path.join(base_path, TOKENIZER_PATH),
        is_build_tokenizer=True)

    for index, pt in enumerate(PRETRAIN_DATASETS):
        generate_pretrain_dataset(
            os.path.join(base_path, pt),
            os.path.join(base_path, f"dataset_preprocessed/pt_train_{index}.base64"),
            os.path.join(base_path, f"dataset_preprocessed/pt_val_{index}.base64"),
            tokenizer=tokenizer,
            block_size=BLOCK_SIZE,
            overlap_ratio=0.5)

    generate_sft_dataset(
        os.path.join(base_path, SFT_DATASET),
        os.path.join(base_path, f"dataset_preprocessed/sft_train.base64"),
        os.path.join(base_path, f"dataset_preprocessed/sft_val.base64"),
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE)

if __name__ == "__main__":
    main()
