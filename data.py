import os
import random
import pickle
import base64
import json
from multiprocessing import Pool
from tqdm import tqdm
from tokenizer import Tokenizer

BLOCK_SIZE = 512
PRETRAIN_DATASETS = [
    "dataset/pretrain.txt"
]
SFT_DATASET = "dataset/sft-general.jsonl"

STAGE_FLAG = {
    "pretrain": 1,  # 1 - do ; else - pass
    "sft":      0   # 1 - do ; else - pass
}

USE_MP = True # 多进程加速？

#############################################################

base_path = os.path.dirname(__file__)
os.makedirs(os.path.join(base_path, "dataset_preprocessed"), exist_ok=True)

TOKENIZER_PATH = "tokenizer/tokenizer_16384.json"

# tokenizer = build_from_files(PRETRAIN_DATASETS + [SFT_DATASET], os.path.join(base_path, TOKENIZER_PATH))
tokenizer = Tokenizer()
tokenizer.load_from_config_file(os.path.join(base_path, TOKENIZER_PATH))


def get_file_chunk_iterator(filepath, chunk_size=65536):
    with open(filepath, mode="r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            yield chunk

def get_file_line_iterator(filepath):
    with open(filepath, mode="r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return
            yield line

def generate_pretrain_dataset(input_path, train_output_path, val_output_path):
    print(f"Reading and encoding raw text file...")

    with os.popen(f"wc --chars {input_path}") as f:
        res = f.readlines()[0]
        charcount = int(res.split(" ")[0])

    print(f"  Total char = {charcount}")

    blocks = []
    token_buffer = []
    chunk_size = 1024
    text_iterator = get_file_chunk_iterator(input_path, chunk_size=chunk_size)

    if USE_MP:
        pool = Pool(os.cpu_count())
        res = pool.imap(func=tokenizer.encode, iterable=text_iterator, chunksize=64)
        for tokens in tqdm(res, total=int(charcount/chunk_size)):
            token_buffer.extend(tokens)
            while len(token_buffer) >= (BLOCK_SIZE+1):
                # TODO 在<|eos|>处切割chunk
                blocks.append(token_buffer[0 : (BLOCK_SIZE+1)])
                # 每一条数据都比BLOCK_SIZE多一个，用于预测下一字符的训练
                token_buffer = token_buffer[(BLOCK_SIZE+1) : ]
        pool.close()
        pool.join()
    else:
        for chunk in tqdm(text_iterator, total=int(charcount/chunk_size)):
            tokens = tokenizer.encode(chunk)
            token_buffer.extend(tokens)
            while len(token_buffer) >= (BLOCK_SIZE+1):
                blocks.append(token_buffer[0 : (BLOCK_SIZE+1)])
                token_buffer = token_buffer[(BLOCK_SIZE+1) : ]

    print(f"  Length of tokens: {len(token_buffer):,}")

    # 巨大数据集的打乱算法
    print(f"Shuffling text blocks and write to file...")
    line_indexes = list(range(len(blocks)))
    random.shuffle(line_indexes)

    with open(train_output_path, "w", encoding="utf-8") as f_train:
        for li in tqdm(range(len(blocks))):
            train_block = pickle.dumps([blocks[line_indexes[li]], None])
            f_train.writelines(str(base64.b64encode(train_block), encoding="utf-8") + "\n")

    with open(val_output_path, "w", encoding="utf-8") as f_val:
        for li in tqdm(range(int(len(blocks) * 0.95), len(blocks))):
            val_block = pickle.dumps([blocks[line_indexes[li]], None])
            f_val.writelines(str(base64.b64encode(val_block), encoding="utf-8") + "\n")

    print(f"Done.")

def apply_template_and_encode(line):
    line = line.strip()
    try:
        qa = json.loads(line)
        question = qa["question"]
        answer = qa["answer"]
        if len(question) + len(answer) + 3 > BLOCK_SIZE:
            answer = answer[0 : BLOCK_SIZE - 3 - len(question)]
            # print(f"超长QA对，裁剪：{answer}")
            return False
        template = f"<|instruct_mark|>{question}<|response_mark|>{answer}<|eos|>"
        ids = tokenizer.encode(template)
        ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(BLOCK_SIZE + 1)]
        mask = [0] * (1 + len(question) + 1) + [1] * (len(answer) + 1)
        mask = [mask[i] if i < len(mask) else 0 for i in range(BLOCK_SIZE + 1)]
        return (ids, mask)
    except:
        print(line)
        return False

def generate_sft_dataset(input_jsonl_path, train_output_path, val_output_path):
    all_items = []

    with os.popen(f"wc --lines {input_jsonl_path}") as f:
        res = f.readlines()[0]
        linecount = int(res.split(" ")[0])
    print(f"  Total lines = {linecount}")

    line_iterator = get_file_line_iterator(input_jsonl_path)

    if USE_MP:
        pool = Pool(os.cpu_count())
        res = pool.imap(func=apply_template_and_encode, iterable=line_iterator, chunksize=64)
        for item in tqdm(res, total=linecount):
            if not item:
                continue
            all_items.append(item)
        pool.close()
        pool.join()
    else:
        for line in tqdm(line_iterator, total=linecount):
            item = apply_template_and_encode(line)
            if not item:
                continue
            all_items.append(item)

    print(f"Shuffling sft blocks and write to file ...")
    line_indexes = list(range(len(all_items)))
    random.shuffle(line_indexes)

    with open(train_output_path, "w", encoding="utf-8") as f_train:
        for i in tqdm(range(len(all_items))):
            item = all_items[i]
            ids = item[0]
            mask = item[1]
            train_data = pickle.dumps([ids, mask])
            f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")

    with open(val_output_path, "w", encoding="utf-8") as f_val:
        for i in tqdm(range(int(len(all_items) * 0.95), len(all_items))):
            item = all_items[i]
            ids = item[0]
            mask = item[1]
            val_data = pickle.dumps([ids, mask])
            f_val.writelines(str(base64.b64encode(val_data), encoding="utf-8") + "\n")

    print(f"Done.")


def main():
    if STAGE_FLAG["pretrain"] == 1:
        print("Pre-processing pretrain dataset...")
        for index, pt in enumerate(PRETRAIN_DATASETS):
            generate_pretrain_dataset(
                os.path.join(base_path, pt),
                os.path.join(base_path, f"dataset_preprocessed/pt_train_{index}.base64"),
                os.path.join(base_path, f"dataset_preprocessed/pt_val_{index}.base64"))
    if STAGE_FLAG["sft"] == 1:
        print("Pre-processing SFT dataset...")
        generate_sft_dataset(
            os.path.join(base_path, SFT_DATASET),
            os.path.join(base_path, f"dataset_preprocessed/sft_train.base64"),
            os.path.join(base_path, f"dataset_preprocessed/sft_val.base64"))

if __name__ == "__main__":
    main()
