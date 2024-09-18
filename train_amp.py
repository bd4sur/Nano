import os
import time
import math
import json
import pickle
import logging
from contextlib import nullcontext

import numpy as np
import torch

from tokenizer import Tokenizer
from model import TrainConfig, ModelConfig, GPT

logger = logging.getLogger(__name__)

class TrainGPT():

    def __init__(self, config_dict, is_from_pretrained=False, task="pretrain") -> None:
        self.train_config = TrainConfig(**(config_dict))
        self.model_config = ModelConfig(**{
            "block_size": config_dict["block_size"],
            "vocab_size": config_dict["vocab_size"],
            "n_layer": config_dict["n_layer"],
            "n_head": config_dict["n_head"],
            "n_embd": config_dict["n_embd"],
            "dropout": config_dict["dropout"],
            "is_causal": config_dict["is_causal"],
        })

        # Internal states
        self.model = None
        self.optimizer = None
        self.iter_count = 0
        self.train_data = None
        self.val_data = None
        self.train_masks = None
        self.val_masks = None
        self.is_from_pretrained = is_from_pretrained
        self.task = task
        assert self.train_config.loss_mask[0] <= self.train_config.loss_mask[1]
        self.global_loss_mask = None
        self.trainset_count = 0

        self.scaler = None
        self.ctx = None

    def log(self, logstr):
        logger.info(logstr)
        print(logstr)

    def load_data(self):
        dataset_path = os.path.join(os.path.dirname(__file__), self.train_config.dataset_path)
        self.log(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        tokenizer_path = os.path.join(os.path.dirname(__file__), self.train_config.tokenizer_path)
        self.log(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = Tokenizer()
        tokenizer.load_from_config(tokenizer_path)

        self.model_config.vocab_size = tokenizer.vocab_size
        self.train_data = np.array(dataset["train_ids"], dtype=np.uint16)
        self.val_data = np.array(dataset["val_ids"], dtype=np.uint16)
        if self.task == "sft":
            self.train_masks = np.array(dataset["train_masks"], dtype=np.uint16)
            self.val_masks = np.array(dataset["val_masks"], dtype=np.uint16)

        self.log(f"  Size of Train set = {len(self.train_data)}")
        self.log(f"  Size of Validation set = {len(self.val_data)}")
        self.log(f"  Size of Vocabulary = {self.model_config.vocab_size}")

    def init(self):
        os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(self.train_config.checkpoint_path)), exist_ok=True)
        torch.manual_seed(self.train_config.random_seed)

        torch.backends.cuda.enable_flash_sdp(self.train_config.sdp_kernel == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(self.train_config.sdp_kernel == "mem_efficient")
        torch.backends.cuda.enable_math_sdp(self.train_config.sdp_kernel == "math")

        self.global_loss_mask = torch.stack(
            [
                torch.from_numpy(np.array(
                    [
                        1 if pos >= self.train_config.loss_mask[0] and pos <= self.train_config.loss_mask[1] else 0
                        for pos in range(self.model_config.block_size)
                    ]).astype(np.int64))
                for _ in range(self.train_config.batch_size)
            ]
        ).to(self.train_config.device, non_blocking=True)

        # Model
        if self.is_from_pretrained:
            _ckpt_path = os.path.join(os.path.dirname(__file__), self.train_config.checkpoint_path)
            self.log(f"Resuming training from {_ckpt_path}")
            # self.log(f"  Model architecture arguments 'block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'is_causal' in training configuration file are ignored. Their values in checkpoint are being used instead.")
            # self.log(f"  Argument 'dropout' in checkpoint is overrided by the value in training configuration file.")
            _checkpoint = torch.load(_ckpt_path, map_location=self.train_config.device)
            # 从Checkpoint中恢复部分模型结构参数
            _model_config = _checkpoint["model_config"]
            _model_config.dropout = self.model_config.dropout # Overrided by new training configuration
            self.model = GPT(_model_config)
            self.model.to(self.train_config.device)
            # 恢复模型参数
            self.model.load_state_dict(_checkpoint["model"])
            self.iter_count = _checkpoint["iter_count"]
        else:
            # init a new model from scratch
            self.log("Initializing a new model for pre-train")
            self.model = GPT(self.model_config)
            self.model.to(self.train_config.device)

        self.log("  Number of Parameters = %.2fM" % (self.model.get_num_params() / 1e6,))

        # Optimizer
        _device_type = 'cuda' if 'cuda' in self.train_config.device else 'cpu'
        self.optimizer = self.model.configure_optimizers(self.train_config.weight_decay, self.train_config.learning_rate, (self.train_config.beta1, self.train_config.beta2), _device_type)
        if self.is_from_pretrained:
            self.optimizer.load_state_dict(_checkpoint["optimizer"]) # 恢复优化器状态

        _checkpoint = None # free up memory

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.train_config.dtype == "float16"))

        _device_type = 'cuda' if 'cuda' in self.train_config.device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        _ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.train_config.dtype]
        self.ctx = nullcontext() if _device_type == 'cpu' else torch.amp.autocast(device_type=_device_type, dtype=_ptdtype)

    def get_batch(self, phase):
        if phase == "train":
            dataset = self.train_data
            if self.task == "sft":
                masks = self.train_masks
            ix = range(self.trainset_count, self.trainset_count + self.train_config.batch_size)
            self.trainset_count += self.train_config.batch_size
            if self.trainset_count >= len(dataset) - self.train_config.batch_size:
                self.trainset_count = 0
        else:
            dataset = self.val_data
            if self.task == "sft":
                masks = self.train_masks
            ix = torch.randint(len(dataset), (self.train_config.batch_size,))

        if self.model_config.is_causal:
            # 取出一批数据，每条数据只保留前block_size个token，构成tensor，shape=(batch_size, block_size)
            x = torch.stack([torch.from_numpy((dataset[i][0 : self.model_config.block_size]).astype(np.int64)) for i in ix])
            # 这批数据每一条都右移一个字符，作为预测目标，shape=(batch_size, block_size)
            y = torch.stack([torch.from_numpy((dataset[i][1 : self.model_config.block_size + 1]).astype(np.int64)) for i in ix])
            if self.task == "pretrain":
                x, y = x.to(self.train_config.device, non_blocking=True), y.to(self.train_config.device, non_blocking=True)
                return x, y, self.global_loss_mask
            elif self.task == "sft":
                mask = torch.stack([torch.from_numpy((masks[i][1 : self.model_config.block_size + 1]).astype(np.int64)) for i in ix])
                x, y, mask = x.to(self.train_config.device, non_blocking=True), y.to(self.train_config.device, non_blocking=True), mask.to(self.train_config.device, non_blocking=True)
                return x, y, mask
        else:
            # 取出一批数据，每条数据只保留前block_size个token，构成tensor，shape=(batch_size, block_size)
            x = torch.stack([torch.from_numpy((dataset[i][0 : self.model_config.block_size]).astype(np.int64)) for i in ix])
            # 取出后面剩余的block_size个token，作为预测目标，shape=(batch_size, block_size)
            y = torch.stack([torch.from_numpy((dataset[i][self.model_config.block_size : self.model_config.block_size * 2]).astype(np.int64)) for i in ix])
            x, y = x.to(self.train_config.device, non_blocking=True), y.to(self.train_config.device, non_blocking=True)
            return x, y, self.global_loss_mask

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        losses = torch.zeros(self.train_config.eval_iters)
        for k in range(self.train_config.eval_iters):
            X, Y, mask = self.get_batch("val")
            with self.ctx:
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

        self.load_data()
        self.init()

        best_val_loss = 1e9
        X, Y, mask = self.get_batch('train') # fetch the very first batch

        self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Start training from iteration #{self.iter_count}")

        iter = self.iter_count

        while iter < self.train_config.max_iters:
            # determine and set the learning rate for this iteration
            lr = self.update_learning_rate(iter) if self.train_config.decay_lr else self.train_config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter > 0 and iter % self.train_config.eval_interval == 0:
                val_loss = self.estimate_loss()
                self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Phase: Validation | Step: {iter} | Val_loss: {val_loss:.3f} | Best_val_loss: {best_val_loss:.4f}")

                if iter > 0 and val_loss < best_val_loss:
                    self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Saving checkpoint to {self.train_config.checkpoint_path}")
                    _checkpoint = {
                        "model":        self.model.state_dict(),
                        "optimizer":    self.optimizer.state_dict(),
                        "iter_count":   iter,
                        "train_config": self.train_config,
                        "model_config": self.model_config
                    }
                    best_val_loss = val_loss
                    torch.save(_checkpoint, os.path.join(os.path.dirname(__file__), self.train_config.checkpoint_path))

            t0 = time.time()

            with self.ctx:
                _, loss = self.model(X, Y, mask)
            X, Y, mask = self.get_batch('train')
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

            t1 = time.time()
            dt = t1 - t0

            if iter % self.train_config.log_interval == 0:
                lossf = loss.item()
                flops = self.model.estimate_flops(self.train_config.batch_size, dt)
                self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Phase: Train | Step: {iter} | TrainDataPos: {self.trainset_count} | Loss: {lossf:.3f} | Time: {dt*1000:.0f} ms | Speed: {flops / 1e9:.2f} GFLOP/s")

            iter += 1
            self.iter_count = iter

def main():
    logging.basicConfig(filename='train.log', filemode="w", level=logging.INFO)
    print(f"PyTorch version: {torch.__version__}")
    # CONFIG_JSON = "train_config.json"
    CONFIG_JSON = "sft_config.json"
    with open(os.path.join(os.path.dirname(__file__), CONFIG_JSON), "r", encoding="utf-8") as f:
        config = json.load(f)
        # trainer = TrainGPT(config, is_from_pretrained=False, task="pretrain")
        trainer = TrainGPT(config, is_from_pretrained=True, task="sft")
        trainer.start()

if __name__ == "__main__":
    main()
