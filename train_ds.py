"""
Forked from https://github.com/karpathy/nanoGPT
BD4SUR 2023.12
"""

import os
import time
import math
import pickle

import numpy as np
import torch
import deepspeed

from model import GPTConfig, GPT

# === Global configuation begin ==========================
config = {
    # GPT Model Args
    "block_size": 256,
    "vocab_size": 10000,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 720,
    "dropout": 0.0, # for pretraining 0 is good, for finetuning try 0.1+
    "bias": False, # do we use bias inside LayerNorm and Linear layers?

    # AdamW Optimizer Args
    "learning_rate": 6e-4, # max learning rate
    "max_iters": 100000, # total number of training iterations
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.99,

    # Learning Rate Scheduler
    "decay_lr": True, # whether to decay the learning rate
    "warmup_iters": 300, # how many steps to warm up for
    "lr_decay_iters": 100000, # should be ~= max_iters per Chinchilla
    "min_lr": 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # Training Task
    "init_from": 'pretrain', # 'pretrain' or 'finetune'
    "batch_size": 50,
    "data_dir": 'data',
    "ckpt_dir": 'ckpt',
    "eval_interval": 100,
    "log_interval": 1,
    "eval_iters": 5,

    # Misc
    "backend": 'nccl', # 'nccl', 'gloo', etc.
    "sdp_kernel": "math", # 选择`scaled_dot_product_attention`所使用的kernel "flash" || "mem_efficient" || "math"
    "device": 'cuda', # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
}
# === Global configuation end ============================

class TrainGPT():

    def __init__(self, config) -> None:

        self.config = config

        # GPT Model Args
        self.block_size = config["block_size"]
        self.vocab_size = config["vocab_size"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.dropout = config["dropout"]
        self.bias = config["bias"]
        # AdamW Optimizer Args
        self.learning_rate = config["learning_rate"]
        self.max_iters = config["max_iters"]
        self.weight_decay = config["weight_decay"]
        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"]
        # Learning Rate Scheduler
        self.decay_lr = config["decay_lr"]
        self.warmup_iters = config["warmup_iters"]
        self.lr_decay_iters = config["lr_decay_iters"]
        self.min_lr = config["min_lr"]
        # Training Task
        self.init_from = config["init_from"]
        self.batch_size = config["batch_size"]
        self.data_dir = config["data_dir"]
        self.ckpt_dir = config["ckpt_dir"]
        self.eval_interval = config["eval_interval"]
        self.log_interval = config["log_interval"]
        self.eval_iters = config["eval_iters"]
        # Misc
        self.backend = config["backend"]
        self.sdp_kernel = config["sdp_kernel"]
        self.device = config["device"]

        # Internal states
        self.model = None
        self.model_args = None
        self.optimizer = None
        self.iter_num = 0
        self.train_data = None
        self.val_data = None

        self.model_engine = None

    def log(self, logstr):
        print(logstr)

    def load_data(self):
        self.train_data = np.memmap(os.path.join(os.path.dirname(__file__), self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(os.path.dirname(__file__), self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        # Metadata of dataset
        meta_path = os.path.join(os.path.dirname(__file__), self.data_dir, 'meta.pkl')
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']

        self.log(f"# Train set = {len(self.train_data)}")
        self.log(f"# Val set = {len(self.val_data)}")
        self.log(f"# Vocab size = {self.vocab_size}")

    def init(self):
        deepspeed.init_distributed(dist_backend=self.backend)

        os.makedirs(os.path.join(os.path.dirname(__file__), self.ckpt_dir), exist_ok=True)
        torch.backends.cuda.enable_flash_sdp(self.sdp_kernel == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(self.sdp_kernel == "mem_efficient")
        torch.backends.cuda.enable_math_sdp(self.sdp_kernel == "math")
        torch.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        # model init
        self.model_args = dict(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias)

        # init a new model for pretrain
        if self.init_from == 'pretrain':
            self.log("Initializing a new model for pretrain")
            _gptconf = GPTConfig(**(self.model_args))
            self.model = GPT(_gptconf)
            self.log("# Parameters = %.2fM" % (self.model.get_num_params()/1e6,))
        # resume training from a checkpoint.
        elif self.init_from == 'finetune':
            self.log(f"Resuming training from {self.ckpt_dir}")
            _ckpt_path = os.path.join(os.path.dirname(__file__), self.ckpt_dir, 'ckpt.pt')
            _checkpoint = torch.load(_ckpt_path, map_location=self.device)
            # 从Checkpoint中恢复模型结构参数（除了dropout）
            for k in ['block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'bias']:
                self.model_args[k] = _checkpoint['model_args'][k]
            _gptconf = GPTConfig(**(self.model_args))
            self.model = GPT(_gptconf)
            self.log("# Parameters = %.2fM" % (self.model.get_num_params()/1e6,))
            # 恢复模型参数
            self.model.load_state_dict(_checkpoint['model'])
            self.iter_num = _checkpoint['iter_num']

        deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(self.model, num_gpus_per_node=2, num_nodes=1, additional_buffer_factor=1.5)

        config_path = os.path.join(os.path.dirname(__file__), 'ds_config.json')
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(model=self.model, config=config_path)

    def get_batch(self, phase):
        dataset = self.train_data if phase == 'train' else self.val_data
        ix = torch.randint(len(dataset) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((dataset[i : i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((dataset[i+1 : i+1+self.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        loss_value = {}
        self.model.eval()
        for phase in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(phase)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            loss_value[phase] = losses.mean()
        self.model.train()
        return loss_value

    # learning rate decay scheduler (cosine with warmup)
    def update_learning_rate(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def start(self):

        self.load_data()
        self.init()

        best_val_loss = 1e9
        X, Y = self.get_batch('train') # fetch the very first batch
        t0 = time.time()
        running_mfu = -1.0
        self.log(f"Start training from iteration #{self.iter_num}!")

        for iter in range(self.max_iters):

            iter = self.iter_num

            # determine and set the learning rate for this iteration
            lr = self.update_learning_rate(iter) if self.decay_lr else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter % self.eval_interval == 0:
                losses = self.estimate_loss()
                self.log(f"Iteration #{iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}, Best val loss {best_val_loss:.4f}")

                if iter > 0 and losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    _checkpoint = {
                        "model":      self.model.state_dict(),
                        "optimizer":  self.optimizer.state_dict(),
                        "model_args": self.model_args,
                        "iter_num":   iter,
                        "config":     self.config,
                    }
                    self.log(f"Saving checkpoint to {self.ckpt_dir}/ckpt.pt")
                    # torch.save(_checkpoint, os.path.join(os.path.dirname(__file__), self.ckpt_dir, 'ckpt.pt'))
                    self.model_engine.save_checkpoint(os.path.join(os.path.dirname(__file__), self.ckpt_dir, 'ckpt_ds.pt'))

            _, loss = self.model(X, Y)
            X, Y = self.get_batch('train')
            self.model_engine.backward(loss)
            self.model_engine.step()

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter % self.log_interval == 0:
                lossf = loss.item()
                mfu = self.model.estimate_mfu(self.batch_size, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Iteration #{iter}: Train loss = {lossf:.4f}, Duration = {dt*1000:.0f} ms, MFU = {running_mfu*100:.2f}% ({312 * running_mfu:.2f} TFLOPS)")
            self.iter_num += 1

def main():
    print(f"PyTorch version: {torch.__version__}")
    trainer = TrainGPT(config)
    trainer.start()

if __name__ == "__main__":
    main()
