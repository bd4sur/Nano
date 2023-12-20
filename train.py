"""
Forked from https://github.com/karpathy/nanoGPT
BD4SUR 2023.12

To run on a single GPU, example:
$ python train.py

To run with DDP on 2 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=2 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

from model import GPTConfig, GPT

# === Global configuation begin ==========================
config = {
    "ckpt_dir": 'ckpt',
    "eval_interval": 100,
    "log_interval": 1,
    "eval_iters": 5,
    "init_from": 'pretrain', # 'pretrain' or 'finetune'
    # data
    "data_dir": 'data',
    "gradient_accumulation_steps": 2,
    "batch_size": 64,
    "block_size": 512,
    # model
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "dropout": 0.0, # for pretraining 0 is good, for finetuning try 0.1+
    "bias": False, # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    "learning_rate": 6e-4, # max learning rate
    "max_iters": 2000, # total number of training iterations
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.99,
    "grad_clip": 1.0, # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    "decay_lr": True, # whether to decay the learning rate
    "warmup_iters": 300, # how many steps to warm up for
    "lr_decay_iters": 2000, # should be ~= max_iters per Chinchilla
    "min_lr": 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    "backend": 'nccl', # 'nccl', 'gloo', etc.

    # 选择`scaled_dot_product_attention`所使用的kernel
    #   较早的GPU最好选择"math"
    "sdp_kernel": "math", # "flash" || "mem_efficient" || "math"
    "device": 'cuda', # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    "dtype": 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    # 模型编译？
    #   较早的GPU，如P100和P40，不支持，报错如下：
    #   RuntimeError: Found Tesla P40 which is too old to be supported by the triton GPU compiler, which is used as the backend.
    #   Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.1
    "is_compile": False # use PyTorch 2.0 to compile the model to be faster
}
# === Global configuation end ============================

class TrainGPT():

    def __init__(self, config) -> None:

        self.config = config

        # Global configuation parameters
        self.ckpt_dir = config["ckpt_dir"]
        self.eval_interval = config["eval_interval"]
        self.log_interval = config["log_interval"]
        self.eval_iters = config["eval_iters"]
        self.init_from = config["init_from"]
        self.data_dir = config["data_dir"]
        self.gradient_accumulation_steps = config["gradient_accumulation_steps"]
        self.batch_size = config["batch_size"]
        self.block_size = config["block_size"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.dropout = config["dropout"]
        self.bias = config["bias"]
        self.learning_rate = config["learning_rate"]
        self.max_iters = config["max_iters"]
        self.weight_decay = config["weight_decay"]
        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"]
        self.grad_clip = config["grad_clip"]
        self.decay_lr = config["decay_lr"]
        self.warmup_iters = config["warmup_iters"]
        self.lr_decay_iters = config["lr_decay_iters"]
        self.min_lr = config["min_lr"]
        self.sdp_kernel = config["sdp_kernel"]
        self.backend = config["backend"]
        self.device = config["device"]
        self.dtype = config["dtype"]
        self.is_compile = config["is_compile"]

        # Internal states
        self.model = None
        self.model_args = None
        self.scaler = None
        self.optimizer = None
        self.ctx = None
        self.iter_num = 0
        self.vocab_size = 0
        self.train_data = None
        self.val_data = None
        self.is_ddp = False
        self.ddp_local_rank = 0

    def load_data(self):
        self.train_data = np.memmap(os.path.join(os.path.dirname(__file__), self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(os.path.dirname(__file__), self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        # Metadata of dataset
        meta_path = os.path.join(os.path.dirname(__file__), self.data_dir, 'meta.pkl')
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']
        print(f"# Train set = {len(self.train_data)}")
        print(f"# Val set = {len(self.val_data)}")
        print(f"# Vocab size = {self.vocab_size}")

    def init_ddp(self):
        # various inits, derived attributes, I/O setup
        self.is_ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        if self.is_ddp:
            init_process_group(backend=self.backend)
            _ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            _ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.is_master_process = _ddp_rank == 0 # this process will do logging, checkpointing etc.
            _seed_offset = _ddp_rank # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.gradient_accumulation_steps % _ddp_world_size == 0
            self.gradient_accumulation_steps //= _ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.is_master_process = True
            _seed_offset = 0
            _ddp_world_size = 1

        tokens_per_iter = self.gradient_accumulation_steps * _ddp_world_size * self.batch_size * self.block_size
        print(f"Is DDP? {self.is_ddp}")
        print(f"Tokens per iteration = {tokens_per_iter:,}")

        if self.is_master_process:
            os.makedirs(os.path.join(os.path.dirname(__file__), self.ckpt_dir), exist_ok=True)

        torch.backends.cuda.enable_flash_sdp(self.sdp_kernel == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(self.sdp_kernel == "mem_efficient")
        torch.backends.cuda.enable_math_sdp(self.sdp_kernel == "math")

        torch.manual_seed(1337 + _seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        _device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        _ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if _device_type == 'cpu' else torch.amp.autocast(device_type=_device_type, dtype=_ptdtype)

    def init_model(self):

        # model init
        self.model_args = dict(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=self.vocab_size,
            dropout=self.dropout)

        # init a new model for pretrain
        if self.init_from == 'pretrain':
            print("Initializing a new model for pretrain")
            _gptconf = GPTConfig(**(self.model_args))
            self.model = GPT(_gptconf)
            print("# Parameters = %.2fM" % (self.model.get_num_params()/1e6,))
        # resume training from a checkpoint.
        elif self.init_from == 'finetune':
            print(f"Resuming training from {self.ckpt_dir}")
            _ckpt_path = os.path.join(os.path.dirname(__file__), self.ckpt_dir, 'ckpt.pt')
            _checkpoint = torch.load(_ckpt_path, map_location=self.device)
            _checkpoint_model_args = _checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                self.model_args[k] = _checkpoint_model_args[k]
            # create the model
            _gptconf = GPTConfig(**(self.model_args))
            self.model = GPT(_gptconf)
            print("# Parameters = %.2fM" % (self.model.get_num_params()/1e6,))
            _state_dict = _checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            _unwanted_prefix = '_orig_mod.'
            for k,v in list(_state_dict.items()):
                if k.startswith(_unwanted_prefix):
                    _state_dict[k[len(_unwanted_prefix):]] = _state_dict.pop(k)
            self.model.load_state_dict(_state_dict)
            self.iter_num = _checkpoint['iter_num']

        # crop down the model block size if desired, using model surgery
        if self.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.block_size)
            self.model_args['block_size'] = self.block_size # so that the checkpoint will have the right value

        self.model.to(self.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        # optimizer
        _device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self.optimizer = self.model.configure_optimizers(self.weight_decay, self.learning_rate, (self.beta1, self.beta2), _device_type)
        if self.init_from == 'finetune':
            self.optimizer.load_state_dict(_checkpoint['optimizer'])
        _checkpoint = None # free up memory

        # compile the model
        if self.is_compile:
            print("Compiling the model... (takes a ~minute)")
            _unoptimized_model = self.model
            self.model = torch.compile(self.model) # requires PyTorch 2.0

        # wrap model into DDP container
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def get_batch(self, phase):
        dataset = self.train_data if phase == 'train' else self.val_data
        ix = torch.randint(len(dataset) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((dataset[i : i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((dataset[i+1 : i+1+self.block_size]).astype(np.int64)) for i in ix])
        _device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        if _device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
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
                with self.ctx:
                    logits, loss = self.model(X, Y)
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
        self.init_ddp()
        self.init_model()

        tb_writer = SummaryWriter(log_dir="log", comment='train')

        # training loop
        best_val_loss = 1e9
        X, Y = self.get_batch('train') # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = self.model.module if self.is_ddp else self.model # unwrap DDP container if needed
        running_mfu = -1.0

        print(f"Start training from iteration #{self.iter_num}!")

        for iter in range(self.max_iters):

            iter = self.iter_num

            # determine and set the learning rate for this iteration
            lr = self.update_learning_rate(iter) if self.decay_lr else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter % self.eval_interval == 0 and self.is_master_process:
                losses = self.estimate_loss()
                print(f"Iteration #{iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}, Best val loss {best_val_loss:.4f}")

                if iter > 0 and losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    _checkpoint = {
                        "model":      raw_model.state_dict(),
                        "optimizer":  self.optimizer.state_dict(),
                        "model_args": self.model_args,
                        "iter_num":   iter,
                        "config":     self.config,
                    }
                    print(f"Saving checkpoint to {self.ckpt_dir}/ckpt.pt")
                    torch.save(_checkpoint, os.path.join(os.path.dirname(__file__), self.ckpt_dir, 'ckpt.pt'))

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
                    logits, loss = self.model(X, Y)
                    loss = loss / self.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
            # clip the gradient
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter % self.log_interval == 0 and self.is_master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(self.batch_size * self.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"Iteration #{iter}: Train loss = {lossf:.4f}, Duration = {dt*1000:.0f} ms, MFU = {running_mfu*100:.2f}% ({312 * running_mfu:.2f} TFLOPS)")
                tb_writer.add_scalar('loss@trainset', loss.item(), iter)

            self.iter_num += 1
            local_iter_num += 1

        if self.is_ddp:
            destroy_process_group()

def main():
    print(f"PyTorch version: {torch.__version__}")
    trainer = TrainGPT(config)
    trainer.start()

if __name__ == "__main__":
    main()
