import os
import time
import math
import json
import pickle
import logging

import numpy as np
import torch

from tokenizer import Tokenizer
from model import TrainConfig, ModelConfig, GPT

logger = logging.getLogger(__name__)

CONFIG_JSON = "train_config.json"

class TrainGPT():

    def __init__(self, config_dict, is_from_pretrained=False) -> None:
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
        self.is_from_pretrained = is_from_pretrained
        assert self.train_config.loss_mask[0] <= self.train_config.loss_mask[1]
        self.loss_mask_array = None
        self.trainset_count = 0

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
        self.train_data = np.array(dataset["train_ids"], dtype=np.uint16) # np.memmap(os.path.join(os.path.dirname(__file__), self.dataset_path, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.array(dataset["val_ids"], dtype=np.uint16) # np.memmap(os.path.join(os.path.dirname(__file__), self.dataset_path, 'val.bin'), dtype=np.uint16, mode='r')

        self.log(f"  Size of Train set = {len(self.train_data)}")
        self.log(f"  Size of Validation set = {len(self.val_data)}")
        self.log(f"  Size of Vocabulary = {self.model_config.vocab_size}")

    def init(self):
        os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(self.train_config.checkpoint_path)), exist_ok=True)
        torch.manual_seed(self.train_config.random_seed)

        self.loss_mask_array = torch.stack(
            [
                torch.from_numpy(np.array(
                    [
                        1 if pos >= self.train_config.loss_mask[0] and pos <= self.train_config.loss_mask[1] else 0
                        for pos in range(self.model_config.block_size)
                    ]).astype(np.int64))
                for _ in range(self.train_config.batch_size)
            ]
        ).to(self.train_config.device)

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


    def get_batch(self, phase):
        if phase == "train":
            dataset = self.train_data
            ix = range(self.trainset_count, self.trainset_count + self.train_config.batch_size)
            self.trainset_count += self.train_config.batch_size
            if self.trainset_count >= len(dataset) - self.train_config.batch_size:
                self.trainset_count = 0
        else:
            dataset = self.val_data
            ix = torch.randint(len(dataset), (self.train_config.batch_size,))

        if self.model_config.is_causal:
            # 取出一批数据，每条数据只保留前block_size个token，构成tensor，shape=(batch_size, block_size)
            x = torch.stack([torch.from_numpy((dataset[i][0 : self.model_config.block_size]).astype(np.int64)) for i in ix])
            # 这批数据每一条都右移一个字符，作为预测目标，shape=(batch_size, block_size)
            y = torch.stack([torch.from_numpy((dataset[i][1 : self.model_config.block_size + 1]).astype(np.int64)) for i in ix])
            x, y = x.to(self.train_config.device), y.to(self.train_config.device)
            return x, y
        else:
            # 取出一批数据，每条数据只保留前block_size个token，构成tensor，shape=(batch_size, block_size)
            x = torch.stack([torch.from_numpy((dataset[i][0 : self.model_config.block_size]).astype(np.int64)) for i in ix])
            # 取出后面剩余的block_size个token，作为预测目标，shape=(batch_size, block_size)
            y = torch.stack([torch.from_numpy((dataset[i][self.model_config.block_size : self.model_config.block_size * 2]).astype(np.int64)) for i in ix])
            x, y = x.to(self.train_config.device), y.to(self.train_config.device)
            return x, y

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        losses = torch.zeros(self.train_config.eval_iters)
        for k in range(self.train_config.eval_iters):
            X, Y = self.get_batch("val")
            _, loss = self.model(X, Y, self.loss_mask_array)
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
        X, Y = self.get_batch('train') # fetch the very first batch

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

            _, loss = self.model(X, Y, self.loss_mask_array)
            X, Y = self.get_batch('train')
            loss.backward()
            self.optimizer.step()
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
    with open(os.path.join(os.path.dirname(__file__), CONFIG_JSON), "r", encoding="utf-8") as f:
        config = json.load(f)
        trainer = TrainGPT(config, is_from_pretrained=False)
        trainer.start()

if __name__ == "__main__":
    main()
