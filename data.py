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
    "dataset/pretrain_psycho.txt"
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

def get_some_chunks(iter, n_chunks=1):
    for _ in range(n_chunks):
        try:
            c = next(iter)
            yield c
        except StopIteration:
            return

def get_file_line_iterator(filepath):
    with open(filepath, mode="r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return
            yield line

def get_some_lines(iter, lines=1):
    for _ in range(lines):
        try:
            c = next(iter)
            yield c
        except StopIteration:
            return

# 将数据集分为若干块，块内打乱，块间乱序拼接，以节约内存
def generate_pretrain_dataset(input_path, train_output_path, val_output_path):

    print(f"Counting character num...")
    cmd = f"powershell -Command (Get-Content {input_path} -Raw).Length" if os.name == "nt" else f"wc --chars {input_path}"
    with os.popen(cmd) as f:
        res = f.readlines()[0]
        charcount = int(res.split(" ")[0])

    print(f"  Total char = {charcount}")

    BLOCKS_PER_PART = 32768

    BLOCKS_PER_CHUNK = 8
    CHUNK_LENGTH = BLOCKS_PER_CHUNK * (BLOCK_SIZE+1)
    CHUNKS_PER_PART = BLOCKS_PER_PART // BLOCKS_PER_CHUNK
    CHARS_PER_PART = CHUNK_LENGTH * CHUNKS_PER_PART
    TOTAL_PARTS = charcount // CHARS_PER_PART + 1

    print(f"Total parts = {TOTAL_PARTS}")

    train_temp_file_paths = []
    val_temp_file_paths = []

    text_iterator = get_file_chunk_iterator(input_path, chunk_size=CHUNK_LENGTH)

    for part_index in range(TOTAL_PARTS):

        print(f"Reading and encoding raw text file of Part {part_index+1}/{TOTAL_PARTS}...")

        train_temp = os.path.join(base_path, f"dataset_preprocessed/train_temp_{part_index}.base64")
        val_temp = os.path.join(base_path, f"dataset_preprocessed/val_temp_{part_index}.base64")

        token_buffer = []
        blocks = []

        chunk_iterator = get_some_chunks(text_iterator, CHUNKS_PER_PART)

        pool = Pool(os.cpu_count())
        res = pool.imap(func=tokenizer.encode, iterable=chunk_iterator, chunksize=64)
        for tokens in tqdm(res, total=CHUNKS_PER_PART):
            token_buffer.extend(tokens)
        pool.close()
        pool.join()

        print(f"  Split tokens into blocks ...")
        for offset in range(0, len(token_buffer), (BLOCK_SIZE+1)):
            # TODO 在<|eos|>处切割chunk
            # 每一条数据都比BLOCK_SIZE多一个，用于预测下一字符的训练
            blk = token_buffer[offset : offset + (BLOCK_SIZE+1)]
            # 如果长度不足(BLOCK_SIZE+1)，则放弃
            if len(blk) < (BLOCK_SIZE+1):
                continue
            blocks.append(blk)

        del token_buffer

        # 在part内部打乱，并写入临时文件
        print(f"  Shuffling text blocks and write to file...")
        line_indexes = list(range(len(blocks)))
        random.shuffle(line_indexes)

        with open(train_temp, "w", encoding="utf-8") as f_train:
            for li in range(len(blocks)):
                train_block = pickle.dumps([blocks[line_indexes[li]], None])
                f_train.writelines(str(base64.b64encode(train_block), encoding="utf-8") + "\n")
                # f_train.writelines(f"Block {line_indexes[li]} : " + tokenizer.decode(blocks[line_indexes[li]]) + "\n")

        with open(val_temp, "w", encoding="utf-8") as f_val:
            for li in range(int(len(blocks) * 0.95), len(blocks)):
                val_block = pickle.dumps([blocks[line_indexes[li]], None])
                f_val.writelines(str(base64.b64encode(val_block), encoding="utf-8") + "\n")

        train_temp_file_paths.append(train_temp)
        val_temp_file_paths.append(val_temp)

    # 在part之间打乱，写入统一文件，并删除临时文件
    print(f"Shuffling all parts and write to file...")
    part_indexes = list(range(len(train_temp_file_paths)))

    random.shuffle(part_indexes)
    with open(train_output_path, "w", encoding="utf-8") as f_train:
        for pindex in part_indexes:
            p = train_temp_file_paths[pindex]
            print(f"  Writing part {p}...")
            with open(p, "r", encoding="utf-8") as tp:
                lines = tp.read()
            f_train.write(lines)
            print(f"  Delete temp file {p}...")
            os.remove(p)

    random.shuffle(part_indexes)
    with open(val_output_path, "w", encoding="utf-8") as f_val:
        for pindex in part_indexes:
            p = val_temp_file_paths[pindex]
            print(f"  Writing part {p}...")
            with open(p, "r", encoding="utf-8") as tp:
                lines = tp.read()
            f_val.write(lines)
            print(f"  Delete temp file {p}...")
            os.remove(p)

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
        if len(ids) > BLOCK_SIZE:
            return False
        ids = [ids[i] if i < len(ids) else tokenizer.special_tokens["<|padding|>"] for i in range(BLOCK_SIZE + 1)]
        mask = [0] * (1 + len(question) + 1) + [1] * (len(answer) + 1)
        mask = [mask[i] if i < len(mask) else 0 for i in range(BLOCK_SIZE + 1)]
        return (ids, mask)
    except:
        print(line)
        return False

def generate_sft_dataset(input_jsonl_path, train_output_path, val_output_path):

    print(f"Counting character num...")
    cmd = f"powershell -Command (Get-Content {input_jsonl_path}).Count" if os.name == "nt" else f"wc -l {input_jsonl_path}"
    with os.popen(cmd) as f:
        res = f.readlines()[0]
        linecount = int(res.split(" ")[0])
    print(f"  Total lines = {linecount}")

    LINES_PER_PART = 131072
    TOTAL_PARTS = linecount // LINES_PER_PART + 1

    print(f"Total parts = {TOTAL_PARTS}")

    train_temp_file_paths = []
    val_temp_file_paths = []

    text_iterator = get_file_line_iterator(input_jsonl_path)

    for part_index in range(TOTAL_PARTS):

        print(f"Reading and encoding raw text file of Part {part_index+1}/{TOTAL_PARTS}...")

        train_temp = os.path.join(base_path, f"dataset_preprocessed/sft_train_temp_{part_index}.base64")
        val_temp = os.path.join(base_path, f"dataset_preprocessed/sft_val_temp_{part_index}.base64")

        all_items = []

        line_iterator = get_some_lines(text_iterator, LINES_PER_PART)

        pool = Pool(os.cpu_count())
        res = pool.imap(func=apply_template_and_encode, iterable=line_iterator, chunksize=64)
        for item in tqdm(res, total=LINES_PER_PART):
            if not item:
                continue
            all_items.append(item)
        pool.close()
        pool.join()

        # 在part内部打乱，并写入临时文件
        print(f"  Shuffling SFT items and write to file...")
        line_indexes = list(range(len(all_items)))
        random.shuffle(line_indexes)

        with open(train_temp, "w", encoding="utf-8") as f_train:
            for i in range(len(all_items)):
                item = all_items[i]
                ids = item[0]
                mask = item[1]
                train_data = pickle.dumps([ids, mask])
                f_train.writelines(str(base64.b64encode(train_data), encoding="utf-8") + "\n")

        with open(val_temp, "w", encoding="utf-8") as f_val:
            for i in range(int(len(all_items) * 0.95), len(all_items)):
                item = all_items[i]
                ids = item[0]
                mask = item[1]
                val_data = pickle.dumps([ids, mask])
                f_val.writelines(str(base64.b64encode(val_data), encoding="utf-8") + "\n")

        train_temp_file_paths.append(train_temp)
        val_temp_file_paths.append(val_temp)

    # 在part之间打乱，写入统一文件，并删除临时文件
    print(f"Shuffling all parts and write to file...")
    part_indexes = list(range(len(train_temp_file_paths)))

    random.shuffle(part_indexes)
    with open(train_output_path, "w", encoding="utf-8") as f_train:
        for pindex in part_indexes:
            p = train_temp_file_paths[pindex]
            print(f"  Writing part {p}...")
            with open(p, "r", encoding="utf-8") as tp:
                lines = tp.read()
            f_train.write(lines)
            print(f"  Delete temp file {p}...")
            os.remove(p)

    random.shuffle(part_indexes)
    with open(val_output_path, "w", encoding="utf-8") as f_val:
        for pindex in part_indexes:
            p = val_temp_file_paths[pindex]
            print(f"  Writing part {p}...")
            with open(p, "r", encoding="utf-8") as tp:
                lines = tp.read()
            f_val.write(lines)
            print(f"  Delete temp file {p}...")
            os.remove(p)

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
