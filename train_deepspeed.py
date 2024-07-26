import os
import time
import math
import json
import pickle
import logging

import numpy as np
import torch
import deepspeed

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
            "eval_only_last_token_loss": config_dict["eval_only_last_token_loss"],
        })

        # Internal states
        self.model = None
        self.optimizer = None
        self.iter_count = 0
        self.train_data = None
        self.val_data = None
        self.is_from_pretrained = is_from_pretrained

        # DeepSpeed
        self.model_engine = None

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

        self.train_config.vocab_size = tokenizer.vocab_size
        self.train_data = np.array(dataset["train_ids"], dtype=np.uint16) # np.memmap(os.path.join(os.path.dirname(__file__), self.dataset_path, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.array(dataset["val_ids"], dtype=np.uint16) # np.memmap(os.path.join(os.path.dirname(__file__), self.dataset_path, 'val.bin'), dtype=np.uint16, mode='r')

        self.log(f"  Size of Train set = {len(self.train_data)}")
        self.log(f"  Size of Validation set = {len(self.val_data)}")
        self.log(f"  Size of Vocabulary = {self.train_config.vocab_size}")

    def init(self):
        deepspeed.init_distributed(dist_backend=self.train_config.backend)

        os.makedirs(os.path.join(os.path.dirname(__file__), os.path.dirname(self.train_config.checkpoint_path)), exist_ok=True)

        torch.backends.cuda.enable_flash_sdp(self.train_config.sdp_kernel == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(self.train_config.sdp_kernel == "mem_efficient")
        torch.backends.cuda.enable_math_sdp(self.train_config.sdp_kernel == "math")
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        torch.manual_seed(self.train_config.random_seed)
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

        deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(self.model, num_gpus_per_node=4, num_nodes=1, additional_buffer_factor=1.5)

        config_path = os.path.join(os.path.dirname(__file__), 'deepspeed_config.json')
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(model=self.model, config=config_path)


    def get_batch(self, phase):
        dataset = self.train_data if phase == 'train' else self.val_data
        # 随机选一批训练数据项
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
        loss_value = {}
        self.model.eval()
        for phase in ['train', 'val']:
            losses = torch.zeros(self.train_config.eval_iters)
            for k in range(self.train_config.eval_iters):
                X, Y = self.get_batch(phase)
                _, loss = self.model(X, Y, self.model_config.eval_only_last_token_loss)
                losses[k] = loss.item()
            loss_value[phase] = losses.mean()
        self.model.train()
        return loss_value

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
                losses = self.estimate_loss()
                self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Phase: Validation | Step: {iter} | Train_loss: {losses['train']:.3f} | Val_loss: {losses['val']:.3f} | Best_val_loss: {best_val_loss:.4f}")

                if iter > 0 and losses['val'] < best_val_loss:
                    self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Saving checkpoint to {self.train_config.checkpoint_path}")
                    self.model_engine.save_checkpoint(os.path.join(os.path.dirname(__file__), os.path.dirname(self.train_config.checkpoint_path), 'deepspeed'))
                    # with open(os.path.join(os.path.dirname(__file__), self.config.ckpt_dir, 'deepspeed/model_config_deepspeed.json'), "w", encoding="utf-8") as f:
                    #     json.dump({
                    #         "iter_count": iter,
                    #         "config":     self.config,
                    #     }, f)
                    best_val_loss = losses['val']
            t0 = time.time()
            _, loss = self.model(X, Y, self.model_config.eval_only_last_token_loss)
            X, Y = self.get_batch('train')
            self.model_engine.backward(loss)
            self.model_engine.step()

            t1 = time.time()
            dt = t1 - t0

            if iter % self.train_config.log_interval == 0:
                lossf = loss.item()
                flops = self.model.estimate_flops(self.train_config.batch_size, dt)
                self.log(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Phase: Train | Step: {iter} | Loss: {lossf:.3f} | Time: {dt*1000:.0f} ms | Speed: {flops / 1e9:.2f} GFLOP/s")

            iter += 1
            self.iter_count = iter

def main():
    logging.basicConfig(filename='train.log', filemode="w", level=logging.INFO)
    # 多机分布式训练情况下，为每个节点设置环境变量
    # ds_env = os.environ.copy()
    # ds_env["PATH"] = "/home/bd4sur/anaconda3/envs/nanogpt/bin:" + ds_env["PATH"]
    # os.environ.update(ds_env)

    print(f"PyTorch version: {torch.__version__}")
    with open(os.path.join(os.path.dirname(__file__), CONFIG_JSON), "r", encoding="utf-8") as f:
        config = json.load(f)
        trainer = TrainGPT(config, is_from_pretrained=False)
        trainer.start()

if __name__ == "__main__":
    main()
