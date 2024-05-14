import os
import time
import math
import pickle

import numpy as np
import torch

from qfunc import q_digits
from model import ModelConfig, GPT

# === Global configuation begin ==========================
config = {
    # GPT Model Args
    "block_size": 128, # 如果是Q问题，则为 q_digits() + 1,
    "vocab_size": 10000,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 64,
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
    "batch_size": 100,
    "random_seed": 114514,
    "eval_only_last_token_loss": False, # 如果是Q问题，则为True；如果是NLG问题，则为False
    "data_dir": 'data_q',
    "ckpt_dir": 'ckpt_q',
    "eval_interval": 100,
    "log_interval": 10,
    "eval_iters": 5,

    # Misc
    "backend": 'nccl', # 'nccl', 'gloo', etc.
    "device": 'cuda:0', # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
}
# === Global configuation end ============================

class TrainGPT():

    def __init__(self, config) -> None:

        self.config = ModelConfig(**(config))

        # Internal states
        self.model = None
        self.optimizer = None
        self.iter_num = 0
        self.train_data = None
        self.val_data = None

    def log(self, logstr):
        print(logstr)

    def load_data(self):
        dataset_path = os.path.join(os.path.dirname(__file__), self.config.data_dir, 'dataset.pkl')
        print(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        self.config.vocab_size = dataset['vocab_size']
        self.train_data = np.array(dataset["train_ids"], dtype=np.uint16) # np.memmap(os.path.join(os.path.dirname(__file__), self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.array(dataset["val_ids"], dtype=np.uint16) # np.memmap(os.path.join(os.path.dirname(__file__), self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        self.log(f"# Train set = {len(self.train_data)}")
        self.log(f"# Val set = {len(self.val_data)}")
        self.log(f"# Vocab size = {self.config.vocab_size}")

    def init(self):
        os.makedirs(os.path.join(os.path.dirname(__file__), self.config.ckpt_dir), exist_ok=True)
        torch.manual_seed(self.config.random_seed)
        # init a new model from scratch
        self.log("Initializing a new model for pretrain")
        self.model = GPT(self.config)
        self.model.to(self.config.device)
        self.log("# Parameters = %.2fM" % (self.model.get_num_params()/1e6,))
        # optimizer
        _device_type = 'cuda' if 'cuda' in self.config.device else 'cpu'
        self.optimizer = self.model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, (self.config.beta1, self.config.beta2), _device_type)

    def get_batch(self, phase):
        dataset = self.train_data if phase == 'train' else self.val_data
        # 随机选一批训练数据项
        ix = torch.randint(len(dataset), (self.config.batch_size,))
        # 取出一批数据，每条数据只保留前block_size个token，构成tensor，shape=(batch_size, block_size)
        x = torch.stack([torch.from_numpy((dataset[i][0 : self.config.block_size]).astype(np.int64)) for i in ix])
        # 这批数据每一条都右移一个字符，作为预测目标
        y = torch.stack([torch.from_numpy((dataset[i][1 : self.config.block_size + 1]).astype(np.int64)) for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        loss_value = {}
        self.model.eval()
        for phase in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(phase)
                _, loss = self.model(X, Y, self.config.eval_only_last_token_loss)
                losses[k] = loss.item()
            loss_value[phase] = losses.mean()
        self.model.train()
        return loss_value

    # learning rate decay scheduler (cosine with warmup)
    def update_learning_rate(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def start(self):

        self.load_data()
        self.init()

        best_val_loss = 1e9
        X, Y = self.get_batch('train') # fetch the very first batch
        t0 = time.time()
        running_mfu = -1.0
        self.log(f"Start training from iteration #{self.iter_num}!")

        for iter in range(self.config.max_iters):

            iter = self.iter_num

            # determine and set the learning rate for this iteration
            lr = self.update_learning_rate(iter) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                self.log(f"Iteration #{iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}, Best val loss {best_val_loss:.4f}")

                if iter > 0 and losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    _checkpoint = {
                        "model":      self.model.state_dict(),
                        "optimizer":  self.optimizer.state_dict(),
                        "iter_num":   iter,
                        "config":     self.config,
                    }
                    self.log(f"Saving checkpoint to {self.config.ckpt_dir}/ckpt.pt")
                    torch.save(_checkpoint, os.path.join(os.path.dirname(__file__), self.config.ckpt_dir, 'ckpt.pt'))

            _, loss = self.model(X, Y, self.config.eval_only_last_token_loss)
            X, Y = self.get_batch('train')
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter % self.config.log_interval == 0:
                lossf = loss.item()
                mfu = self.model.estimate_mfu(self.config.batch_size, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Iteration #{iter}: Train loss = {lossf:.4f}, Duration = {dt*1000:.0f} ms, MFU = {running_mfu*100:.2f}% ({312 * running_mfu:.2f} TFLOPS)")
            self.iter_num += 1

def main():
    print(f"PyTorch version: {torch.__version__}")
    trainer = TrainGPT(config)
    trainer.start()

if __name__ == "__main__":
    main()
