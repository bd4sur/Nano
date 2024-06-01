import os
import time
import math
import json
import pickle

import numpy as np
import torch

from tokenizer import Tokenizer
from model import ModelConfig, GPT

CONFIG_JSON = "train_nlg.json"

class TrainGPT():

    def __init__(self, config, is_from_pretraind=False) -> None:

        self.config = ModelConfig(**(config))

        # Internal states
        self.model = None
        self.optimizer = None
        self.iter_count = 0
        self.train_data = None
        self.val_data = None
        self.is_from_pretrained = is_from_pretraind

    def log(self, logstr):
        print(logstr)

    def load_data(self):
        dataset_path = os.path.join(os.path.dirname(__file__), self.config.data_dir, 'dataset.pkl')
        print(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        tokenizer_path = os.path.join(os.path.dirname(__file__), self.config.data_dir, 'tokenizer.json')
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = Tokenizer()
        tokenizer.load_from_config(tokenizer_path)

        self.config.vocab_size = tokenizer.vocab_size
        self.train_data = np.array(dataset["train_ids"], dtype=np.uint16) # np.memmap(os.path.join(os.path.dirname(__file__), self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.array(dataset["val_ids"], dtype=np.uint16) # np.memmap(os.path.join(os.path.dirname(__file__), self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        self.log(f"Size of Train set = {len(self.train_data)}")
        self.log(f"Size of Validation set = {len(self.val_data)}")
        self.log(f"Size of vocabulary = {self.config.vocab_size}")

    def init(self):
        os.makedirs(os.path.join(os.path.dirname(__file__), self.config.ckpt_dir), exist_ok=True)
        torch.manual_seed(self.config.random_seed)
        # Model
        if self.is_from_pretrained:
            _ckpt_path = os.path.join(os.path.dirname(__file__), self.config.ckpt_dir, 'ckpt.pt')
            self.log(f"Resuming training from {_ckpt_path}")
            _checkpoint = torch.load(_ckpt_path, map_location=self.config.device)
            # 从Checkpoint中恢复部分模型结构参数
            _config = _checkpoint["config"]
            _config.block_size = self.config.block_size
            _config.vocab_size = self.config.vocab_size
            _config.n_layer = self.config.n_layer
            _config.n_head = self.config.n_head
            _config.n_embd = self.config.n_embd
            _config.bias = self.config.bias
            self.model = GPT(self.config)
            self.model.to(self.config.device)
            self.log("Number of Parameters = %.2fM" % (self.model.get_num_params()/1e6,))
            # 恢复模型参数
            self.model.load_state_dict(_checkpoint["model"])
            self.iter_count = _checkpoint["iter_count"]
        else:
            # init a new model from scratch
            self.log("Initializing a new model for pretrain")
            self.model = GPT(self.config)
            self.model.to(self.config.device)
            self.log("Number of Parameters = %.2fM" % (self.model.get_num_params()/1e6,))

        # Optimizer
        _device_type = 'cuda' if 'cuda' in self.config.device else 'cpu'
        self.optimizer = self.model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, (self.config.beta1, self.config.beta2), _device_type)
        if self.is_from_pretrained:
            self.optimizer.load_state_dict(_checkpoint["optimizer"]) # 恢复优化器状态

        _checkpoint = None # free up memory


    def get_batch(self, phase):
        dataset = self.train_data if phase == 'train' else self.val_data
        # 随机选一批训练数据项
        ix = torch.randint(len(dataset), (self.config.batch_size,))
        if self.config.is_causal:
            # 取出一批数据，每条数据只保留前block_size个token，构成tensor，shape=(batch_size, block_size)
            x = torch.stack([torch.from_numpy((dataset[i][0 : self.config.block_size]).astype(np.int64)) for i in ix])
            # 这批数据每一条都右移一个字符，作为预测目标，shape=(batch_size, block_size)
            y = torch.stack([torch.from_numpy((dataset[i][1 : self.config.block_size + 1]).astype(np.int64)) for i in ix])
            x, y = x.to(self.config.device), y.to(self.config.device)
            return x, y
        else:
            # 取出一批数据，每条数据只保留前block_size个token，构成tensor，shape=(batch_size, block_size)
            x = torch.stack([torch.from_numpy((dataset[i][0 : self.config.block_size]).astype(np.int64)) for i in ix])
            # 取出后面剩余的block_size个token，作为预测目标，shape=(batch_size, block_size)
            y = torch.stack([torch.from_numpy((dataset[i][self.config.block_size : self.config.block_size * 2]).astype(np.int64)) for i in ix])
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
        self.log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Start training from iteration #{self.iter_count}")

        iter = self.iter_count

        while iter < self.config.max_iters:
            # determine and set the learning rate for this iteration
            lr = self.update_learning_rate(iter) if self.config.decay_lr else self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                self.log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Eval @ iteration #{iter} | Train loss {losses['train']:.4f} | Val loss {losses['val']:.4f} | Best val loss {best_val_loss:.4f}")

                if iter > 0 and losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    _checkpoint = {
                        "model":      self.model.state_dict(),
                        "optimizer":  self.optimizer.state_dict(),
                        "iter_count": iter,
                        "config":     self.config,
                    }
                    self.log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving checkpoint to {self.config.ckpt_dir}/ckpt.pt")
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
                flops = self.model.estimate_flops(self.config.batch_size, dt)
                self.log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Train iteration #{iter} | Loss {lossf:.4f} | {dt*1000:.0f} ms | {flops / 1e9:.2f} GFLOPS")

            iter += 1
            self.iter_count = iter

def main():
    print(f"PyTorch version: {torch.__version__}")
    with open(os.path.join(os.path.dirname(__file__), CONFIG_JSON), "r", encoding="utf-8") as f:
        config = json.load(f)
        trainer = TrainGPT(config, is_from_pretraind=False)
        trainer.start()

if __name__ == "__main__":
    main()
