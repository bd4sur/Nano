# Nano Transformer
# BD4SUR 2024
# 
# train.py - 单机单卡和DDP训练

import os
import time
import math
import json
import base64
import pickle
import logging
import argparse
from contextlib import nullcontext
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tokenizer import Tokenizer
from model import TrainConfig, ModelConfig, GPT

logger = logging.getLogger(__name__)

MODEL_VERSION = "2024.10"

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
            cmd = f"powershell -Command (Get-Content {fp}).Count" if os.name == "nt" else f"wc -l {fp}"
            with os.popen(cmd) as f:
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


class TrainGPT():

    def __init__(self, model_config_dict, train_config_dict, max_steps=1e10, ckpt_filename=None, is_continued_pretrain=False):
        self.train_config = TrainConfig(**(train_config_dict))
        self.model_config = ModelConfig(**(model_config_dict))

        # Internal states
        self.model = None
        self.optimizer = None
        self.step_count = 0
        self.max_steps = max_steps
        self.ckpt_filename = ckpt_filename

        self.tokenizer = None
        self.train_data = None
        self.val_data = None

        self.from_checkpoint = self.train_config.from_checkpoint
        self.from_checkpoint = self.from_checkpoint if len(self.from_checkpoint) > 0 else None

        self.is_continued_pretrain = is_continued_pretrain # 用于控制训练开始前是否跳过已经训练过的数据集：如果是预训练，则跳过

        self.torch_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16
        }[self.train_config.dtype]

        # AMP
        self.scaler = None
        self.ctx = None

        # DDP
        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_master_process = True
        self.is_ddp = False
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.gradient_accumulation_steps = self.train_config.gradient_accumulation_steps

    def log(self, logstr):
        if self.is_master_process:
            logger.info(logstr)
            print(logstr)

    def init_training(self):

        self.log(f"Initializing DDP settings...")

        self.is_ddp = int(os.environ.get('RANK', -1)) != -1
        self.log(f"  is_ddp = {self.is_ddp}")

        if self.is_ddp:
            init_process_group(backend=self.train_config.backend)
            _ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.current_device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.current_device)
            self.is_master_process = _ddp_rank == 0 # this process will do logging, checkpointing etc.
            _seed_offset = _ddp_rank # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.gradient_accumulation_steps % self.ddp_world_size == 0
            self.gradient_accumulation_steps //= self.ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.is_master_process = True
            _seed_offset = 0
            self.ddp_world_size = 1

        if self.is_master_process:
            if len(self.train_config.save_checkpoint_to) <= 0 or self.train_config.save_checkpoint_to is None:
                self.train_config.save_checkpoint_to = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")
            os.makedirs(self.train_config.save_checkpoint_to, exist_ok=True)

        torch.manual_seed(self.train_config.random_seed + _seed_offset)

        self.log(f"Initializing PyTorch settings...")

        # 混合精度训练情况下，启用TF32以加速FP32运算（稍微牺牲精度）
        torch.backends.cuda.matmul.allow_tf32 = self.train_config.use_amp
        torch.backends.cudnn.allow_tf32 = self.train_config.use_amp

        if not self.train_config.use_amp:
            torch.set_default_dtype(self.torch_dtype)

        torch.backends.cuda.enable_flash_sdp(self.train_config.sdp_kernel == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(self.train_config.sdp_kernel == "mem_efficient")
        torch.backends.cuda.enable_math_sdp(self.train_config.sdp_kernel == "math")

        self.log(f"Initializing models and optimizers...")

        # 继续训练（分为全参数微调/继续预训练（取决于数据mask）、LoRA微调两种情况）
        if self.from_checkpoint is not None:
            _checkpoint = torch.load(self.from_checkpoint, map_location=self.current_device)
            # 恢复模型配置
            self.model_config = _checkpoint["model_config"]
            self.model_config.dropout = self.model_config.dropout # Overrided by new training configuration
            # 恢复词表
            self.tokenizer = Tokenizer()
            self.tokenizer.load_from_config_dict(_checkpoint["tokenizer_config"])
            # LoRA
            if self.train_config.use_lora:
                self.log(f"  LoRA fine-tuning based on pre-trained model `{self.from_checkpoint}`...")
                # 迭代步数从零开始
                self.step_count = 0
                # 加载预训练模型参数
                self.model = GPT(self.model_config)
                self.model.load_state_dict(_checkpoint["model"], strict=False)
                # 将模型转为LoRA模型（外挂低秩适配层、冻结预训练参数）
                self.model.to_lora(
                    lora_rank=self.train_config.lora_rank,
                    lora_alpha=self.train_config.lora_alpha,
                    lora_dropout=self.train_config.lora_dropout)
                self.model.to(self.current_device)
            # 全参数微调/继续预训练
            else:
                self.log(f"  Resuming training from `{self.from_checkpoint}`...")
                # 恢复迭代步数
                self.step_count = _checkpoint["step_count"]
                # 恢复模型参数
                self.model = GPT(self.model_config)
                self.model.load_state_dict(_checkpoint["model"], strict=False)
                self.model.to(self.current_device)

        # 从零开始训练（忽略任何LoRA设置）
        else:
            self.log("  Initializing a new model from scrach for pre-train...")
            # 从词表文件构建词元编码器
            self.log(f"  Loading tokenizer from {self.train_config.tokenizer_path}...")
            self.tokenizer = Tokenizer()
            self.tokenizer.load_from_config_file(self.train_config.tokenizer_path)
            if self.tokenizer.vocab_size > self.model_config.vocab_size:
                self.log("WARNING: Model's vocab_size is smaller than tokenizer's vocab_size.")
            # 初始化新模型
            self.model = GPT(self.model_config)
            self.model.to(self.current_device)

        # 初始化优化器/恢复优化器状态
        _device_type = 'cuda' if 'cuda' in self.current_device else 'cpu'
        self.optimizer = self.model.configure_optimizers(self.train_config.weight_decay, self.train_config.learning_rate, (self.train_config.beta1, self.train_config.beta2), _device_type)
        # 仅 全参数微调/继续预训练 涉及优化器状态的恢复
        if self.from_checkpoint is not None and not self.train_config.use_lora:
            self.optimizer.load_state_dict(_checkpoint["optimizer"])

        _checkpoint = None # free up memory

        self.log(f"    block_size = {self.model_config.block_size}")
        self.log(f"    vocab_size = {self.model_config.vocab_size}")
        self.log(f"    n_layer    = {self.model_config.n_layer}")
        self.log(f"    n_embd     = {self.model_config.n_embd}")
        self.log(f"    n_head     = {self.model_config.n_head}")
        self.log(f"    n_kv_head  = {self.model_config.n_kv_head}")
        self.log(f"    n_hidden   = {self.model_config.n_hidden}")
        self.log(f"    Parameters")
        self.log(f"         Total = {self.model.get_num_params():,} ({self.model.get_num_params() / 1e9}B)")
        self.log(f"         Train = {self.model.get_num_params_train():,} ({self.model.get_num_params_train() / self.model.get_num_params() * 100}%)")

        if self.train_config.use_amp:
            # initialize a GradScaler. If enabled=False scaler is a no-op
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.train_config.dtype == "bfloat16"))

            _device_type = 'cuda' if 'cuda' in self.current_device else 'cpu'
            # note: float16 data type will automatically use a GradScaler
            self.ctx = nullcontext() if _device_type == 'cpu' else torch.amp.autocast(device_type=_device_type, dtype=self.torch_dtype)

        # wrap model into DDP container
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def load_data(self):
        self.log(f"Loading dataset...")
        train_curriculum = []
        val_curriculum = []
        for ds_path in self.train_config.dataset_path:
            train_path = ds_path[0]
            val_path = ds_path[1]
            train_curriculum.append(train_path)
            val_curriculum.append(val_path)

        self.train_data = DataLoader(train_curriculum)
        self.val_data = DataLoader(val_curriculum)

        for i, line in enumerate(self.train_data.line_num):
            self.log(f"  Train set {i} : {line:,} samples ({line * self.model_config.block_size:,} tokens)")
        for i, line in enumerate(self.val_data.line_num):
            self.log(f"  Valid set {i} : {line:,} samples ({line * self.model_config.block_size:,} tokens)")

        # DDP：每个rank用完全独立的DataLoader，间隔(rank-1)取批次，模拟各个rank交替从训练集中取batch
        #   初始化阶段，让每个worker延迟 self.ddp_local_rank 次取batch。图示如下（等号代表初始延迟，减号代表每一步迭代中间的固定延迟）：
        #   rank0  0---4---8---c
        #   rank1  =1---5---9---
        #   rank2  ==2---6---a--
        #   rank3  ===3---7---b-
        for _ in range(self.ddp_local_rank):
            self.get_batch("train")

    def get_batch(self, phase):
        if phase == "train":
            dataset = self.train_data
        else:
            dataset = self.val_data
        device = self.current_device
        x, y, mask = dataset.get_batch(self.train_config.batch_size, self.model_config.block_size, self.model_config.is_causal)
        x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        return x, y, mask

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        losses = torch.zeros(self.train_config.eval_iters)
        for k in range(self.train_config.eval_iters):
            X, Y, mask = self.get_batch("val")
            if self.train_config.use_amp:
                with self.ctx:
                    _, loss = self.model(X, Y, mask)
            else:
                _, loss = self.model(X, Y, mask)
            losses[k] = loss.item()
        self.model.train()
        return losses.mean()

    # learning rate decay scheduler (cosine with warmup)
    def update_learning_rate(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.train_config.warmup_iters:
            return self.train_config.learning_rate * it / self.train_config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.train_config.lr_decay_iters:
            return self.train_config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.train_config.warmup_iters) / (self.train_config.lr_decay_iters - self.train_config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.train_config.min_lr + coeff * (self.train_config.learning_rate - self.train_config.min_lr)


    def start(self):
        self.init_training()
        self.load_data()

        best_val_loss = math.log(self.model_config.vocab_size) * 2

        start_step = self.step_count
        self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Start training from iteration #{start_step}")

        raw_model = self.model.module if self.is_ddp else self.model # unwrap DDP container if needed

        iter = start_step

        # 继续训练：跳过前面训练过的数据集
        if self.is_continued_pretrain:
            for _ in range(iter - 1):
                self.get_batch("train")

        X, Y, mask = self.get_batch('train') # fetch the very first batch

        t0_total = (time.time_ns(), iter)
        t1_total = None

        while iter <= self.max_steps:
            # determine and set the learning rate for this iteration
            lr = self.update_learning_rate(iter) if self.train_config.decay_lr else self.train_config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 计算验证集损失，保存检查点（分为全量和LoRA两类）
            if iter > 0 and iter % self.train_config.eval_interval == 0 and self.is_master_process:
                val_loss = self.estimate_loss()
                self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Validation | Step: {iter} | Val_loss: {val_loss:.3f} | Best_val_loss: {best_val_loss:.4f}")

                # 保底策略：无论验证集损失是否下降，每1000步保存一个检查点
                if iter > 0 and iter > start_step and (val_loss < best_val_loss or iter % 1000 == 0):
                    # LoRA：保存LoRA模块
                    if self.train_config.use_lora:
                        ckpt_name = f"lora_{time.strftime('%Y%m%d_%H%M%S')}_step_{iter}.pt" if self.ckpt_filename is None else self.ckpt_filename
                        ckpt_path = os.path.join(self.train_config.save_checkpoint_to, ckpt_name)
                        self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Saving LoRA checkpoint to `{ckpt_path}`")
                        _checkpoint = {
                            "version":          MODEL_VERSION,
                            "is_lora":          True,
                            "lora":             raw_model.get_lora_state_dict(),
                            # "model":            raw_model.state_dict(),            # TODO 后续去掉
                            "optimizer":        self.optimizer.state_dict(),
                            "step_count":       iter,
                            "train_config":     self.train_config,
                            "model_config":     self.model_config,
                            # "tokenizer_config": self.tokenizer.config              # TODO 后续去掉
                        }
                    # 预训练/继续与训练/全参数微调：保存模型全部参数和优化器状态
                    else:
                        ckpt_name = f"checkpoint_{time.strftime('%Y%m%d_%H%M%S')}_step_{iter}.pt" if self.ckpt_filename is None else self.ckpt_filename
                        ckpt_path = os.path.join(self.train_config.save_checkpoint_to, ckpt_name)
                        self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Saving full-param checkpoint to `{ckpt_path}`")
                        _checkpoint = {
                            "version":          MODEL_VERSION,
                            "is_lora":          False,
                            "model":            raw_model.state_dict(),
                            "optimizer":        self.optimizer.state_dict(),
                            "step_count":       iter,
                            "train_config":     self.train_config,
                            "model_config":     self.model_config,
                            "tokenizer_config": self.tokenizer.config
                        }
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    torch.save(_checkpoint, ckpt_path)

            t0 = time.time_ns()

            # 使用自动混合精度技术（默认）
            if self.train_config.use_amp:
                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                for micro_step in range(self.gradient_accumulation_steps):
                    if self.is_ddp:
                        # in DDP training we only need to sync gradients at the last micro step.
                        # the official way to do this is with model.no_sync() context manager, but
                        # I really dislike that this bloats the code and forces us to repeat code
                        # looking at the source of that context manager, it just toggles this variable
                        self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                    with self.ctx:
                        _, loss = self.model(X, Y, mask)
                        loss = loss / self.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    X, Y, mask = self.get_batch('train')
                    # backward pass, with gradient scaling if training in fp16
                    self.scaler.scale(loss).backward()
                # clip the gradient
                if self.train_config.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
                # step the optimizer and scaler if training in fp16
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                self.optimizer.zero_grad(set_to_none=True)

            # 不使用自动混合精度技术（兼容较旧的GPU和CPU）
            else:
                for micro_step in range(self.gradient_accumulation_steps):
                    if self.is_ddp:
                        self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                    _, loss = self.model(X, Y, mask)
                    loss = loss / self.gradient_accumulation_steps
                    X, Y, mask = self.get_batch('train')
                    loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time_ns()
            dt = (t1 - t0) / 1e9

            # DDP：每个worker间隔(ddp_world_size-1)取batch，模拟各个rank交替从训练集中取batch
            for _ in range(self.ddp_world_size - 1):
                self.get_batch("train")

            # 记录日志
            if iter > 0 and iter % self.train_config.log_interval == 0:
                t1_total = (time.time_ns(), iter)
                lossf = loss.item() * self.gradient_accumulation_steps # NOTE 计算loss非常耗时
                flops = raw_model.estimate_flops(self.train_config.batch_size * self.gradient_accumulation_steps, dt)
                throughput = self.gradient_accumulation_steps * self.ddp_world_size * self.train_config.batch_size * self.model_config.block_size * (t1_total[1] - t0_total[1]) / ((t1_total[0] - t0_total[0]) / 1e9)
                self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Epoch: {self.train_data.epoch} | Step: {iter} | Dataset: {self.train_data.current_course_index}-{self.train_data.current_line_pos[self.train_data.current_course_index]} | Loss: {lossf:.3f} | {dt*1000:.0f} ms/step , {flops / 1e9:.2f} GFLOP/s , {throughput:.1f} tokens/s")
                t0_total = t1_total

            iter += 1
            self.step_count = iter

        if self.is_ddp:
            destroy_process_group()

def main():
    logging.basicConfig(filename=f"train_{time.strftime('%Y%m%d_%H%M%S')}.log", filemode="w", level=logging.INFO)
    print(f"PyTorch version: {torch.__version__}")

    parser = argparse.ArgumentParser(description="Train Nano model.")
    parser.add_argument("-t", "--train-config", type=str, default=os.path.join(os.path.dirname(__file__), "config_pretrain.json"))
    parser.add_argument("-m", "--model-config", type=str, default=os.path.join(os.path.dirname(__file__), "model_config.json"))
    parser.add_argument("-c", "--continue_pretrain", action="store_true")
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config_dict = json.load(f)
    with open(args.train_config, "r", encoding="utf-8") as f:
        train_config_dict = json.load(f)

    trainer = TrainGPT(model_config_dict, train_config_dict, is_continued_pretrain=args.continue_pretrain)
    trainer.start()

if __name__ == "__main__":
    main()
