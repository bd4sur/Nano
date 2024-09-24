import os
import base64
import pickle
import torch

class DataLoader:
    def __init__(self, filepath_list):
        self.filepath_list = filepath_list
        self.course_num = len(filepath_list)
        self.current_course_index = 0

        self.line_iterator = [0] * self.course_num
        self.line_num = [0] * self.course_num
        self.current_line_pos = [0] * self.course_num

        self.epoch = 0

        for index,fp in enumerate(filepath_list):
            # 行迭代器
            self.line_iterator[index] = self.get_line_iterator(fp)
            # 当前行号
            self.current_line_pos[index] = 0
            # 行数统计
            with os.popen(f"wc -l {fp}") as f:
                res = f.readlines()[0]
                self.line_num[index] = int(res.split(" ")[0])
            # print(f"Lines: {self.line_num[index]}")

    def get_line_iterator(self, filepath):
        with open(filepath, mode="r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    return
                yield line

    def get_batch(self, batch_size, block_size, is_causal=True):
        batch = []
        for _ in range(batch_size):
            if self.current_line_pos[self.current_course_index] == self.line_num[self.current_course_index]:
                # print("Next course")
                fp = self.filepath_list[self.current_course_index]
                self.line_iterator[self.current_course_index] = self.get_line_iterator(fp)
                self.current_line_pos[self.current_course_index] = 0

                self.current_course_index = self.current_course_index + 1
                if self.current_course_index == self.course_num:
                    # print("Return to first course")
                    self.current_course_index = 0
                    self.epoch = self.epoch + 1

            current_line_iterator = self.line_iterator[self.current_course_index]

            try:
                line = next(current_line_iterator)
            except StopIteration:
                print("StopIteration")
                return batch
            obj = pickle.loads(base64.b64decode(line))
            batch.append(obj)
            self.current_line_pos[self.current_course_index] = self.current_line_pos[self.current_course_index] + 1

        # 如果是因果语言模型：则 x,y = ("12345","2345x")，即构造预测下一个token的xy对，并保留mask信息（用于监督微调）
        if is_causal:
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
        # 如果不是因果注意力模型，则将输入序列seq按照block_size一分为二：seq[0:block_size]和seq[block_size:]，后者截断为block_size长度（不足不填充）
        else:
            x = torch.stack([torch.tensor(item[0][0 : block_size], dtype=torch.int64) for item in batch])
            y = torch.stack([torch.tensor(item[0][block_size : block_size * 2], dtype=torch.int64) for item in batch])
            mask = torch.stack([
                torch.tensor([1 for _ in range(block_size)], dtype=torch.int64) if item[1] is None else # pretrain
                torch.tensor(item[1][0 : block_size], dtype=torch.int64) # SFT
                for item in batch
            ])
        return x, y, mask
