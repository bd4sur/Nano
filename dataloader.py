import os
import base64
import pickle
import numpy as np
import torch

class DataLoader:
    def __init__(self, filepath, buffer_size=10000):
        self.filepath = filepath
        self.line_iterator = self.get_line_iterator(filepath)
        self.line_num = 0
        self.line_pos = 0
        self.buffer_size = buffer_size
        self.buffer = []
        with os.popen(f"wc -l {filepath}") as f:
            res = f.readlines()[0]
            self.line_num = int(res.split(" ")[0])
        # print(f"Lines: {self.line_num}")

    def get_line_iterator(self, filepath):
        with open(filepath, mode="r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    return
                yield line

    def get_batch(self, batch_size, block_size):
        batch = []
        for _ in range(batch_size):
            if self.line_pos == self.line_num:
                # print("Reload dataset (epoch)")
                self.line_iterator = self.get_line_iterator(self.filepath)
                self.line_pos = 0
            try:
                line = next(self.line_iterator)
            except StopIteration:
                print("StopIteration")
                return batch
            obj = pickle.loads(base64.b64decode(line))
            batch.append(obj)
            self.line_pos = self.line_pos + 1
        # 取出一批数据，每条数据只保留前block_size个token，构成tensor，shape=(batch_size, block_size)
        x = torch.stack([torch.tensor(item[0][0 : block_size], dtype=torch.int64) for item in batch])
        # 这批数据每一条都右移一个字符，作为预测目标，shape=(batch_size, block_size)
        y = torch.stack([torch.tensor(item[0][1 : block_size + 1], dtype=torch.int64) for item in batch])
        # Mask
        mask = torch.stack([
            torch.tensor([1 for _ in range(block_size)], dtype=torch.int64) if item[1] is None else # pretrain
            torch.tensor(item[1][1 : block_size + 1], dtype=torch.int64) # SFT
            for item in batch
        ])
        return x, y, mask

# def write_text_file(filepath):
#     with open(filepath, "w", encoding="utf-8") as f:
#         for i in range(10):
#             obj = {
#                 "index": i,
#                 "ids": [j for j in range(10)],
#                 "mask": [j for j in range(10)],
#             }
#             obj = pickle.dumps(obj)
#             f.writelines(str(base64.b64encode(obj), encoding="utf-8") + "\n")