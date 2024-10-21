"""
Forked from https://github.com/karpathy/nanoGPT
BD4SUR 2023.12
"""

import math
import inspect
from dataclasses import dataclass, fields
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# NOTE KV-Cache仅为实验性实现，主要影响自回归生成。该标识通过forward逐层传递到Attn层，训练时默认不使用。
#      KV-Cache可缓解长文本生成后期TPS大幅下降的问题，但是对于短文本的生成速度未必有改善（取决于SDPA算子的性能）。
USE_KV_CACHE = True

@dataclass
class ModelConfig:
    block_size: int = 128
    vocab_size: int = 10000
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    use_rope: bool = True
    norm_eps: float = 1e-5
    is_causal: bool = True



@dataclass(init=False)
class TrainConfig:
    # GPT Model Args (Overrided)
    dropout: Optional[float] = 0.0

    # AdamW Optimizer Args
    learning_rate: Optional[float] = 6e-4
    weight_decay: Optional[float] = 1e-1
    beta1: Optional[float] = 0.9
    beta2: Optional[float] = 0.99

    # Learning Rate Scheduler
    decay_lr: Optional[bool] = True
    warmup_iters: Optional[int] = 300
    lr_decay_iters: Optional[int] = 100000
    min_lr: Optional[float] = 6e-5

    # Training Task
    from_checkpoint: Optional[str] = ""
    save_checkpoint_to: Optional[str] = ""
    dataset_path: Optional[list[list[str]]] = None
    tokenizer_path: Optional[str] = ""
    
    batch_size: Optional[int] = 128
    gradient_accumulation_steps: Optional[int] = 4
    grad_clip: Optional[float] = 1.0

    random_seed: Optional[int] = 114514
    eval_interval: Optional[int] = 100
    log_interval: Optional[int] = 1
    eval_iters: Optional[int] = 5

    # Misc & DDP config
    backend: Optional[str] = "nccl"
    device: Optional[str] = "cuda:0"
    sdp_kernel: Optional[str] = "math"
    dtype: Optional[str] = "float16"
    use_amp: Optional[bool] = True


    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MaskedSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        # KV-Cache (experimental)
        self.cache_k = None
        self.cache_v = None
        # is causal self attention?
        self.is_causal = config.is_causal
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.is_causal:
            causal_mask = torch.triu(torch.full((1, 1, self.block_size, self.block_size), float('-inf')), diagonal=1).view(1, 1, config.block_size, config.block_size)
            self.register_buffer("mask", causal_mask)

    def forward(self, x, pos_cis, start_pos, use_kv_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)

        if pos_cis is not None:
            q, k = apply_rotary_emb(q, k, pos_cis)

        if use_kv_cache and (not self.training):
            if start_pos == 0 or (self.cache_k is None and self.cache_v is None):
                self.cache_k = torch.zeros((B, self.block_size, self.n_head, C // self.n_head)).to(q.device)
                self.cache_v = torch.zeros((B, self.block_size, self.n_head, C // self.n_head)).to(q.device)

            self.cache_k[:B, start_pos : start_pos + T] = k
            self.cache_v[:B, start_pos : start_pos + T] = v

            k = self.cache_k[:B, : start_pos + T]
            v = self.cache_v[:B, : start_pos + T]

        k = k.transpose(1, 2) # (B, nh, T, hs)
        q = q.transpose(1, 2) # (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if self.training or (not use_kv_cache):
                # NOTE flash_attn尚不支持非None的attn_mask
                # Ref. https://github.com/pytorch/pytorch/blob/753ba5d30a361be4f610cf7dde4fd63726ed8f86/aten/src/ATen/native/transformers/sdp_utils_cpp.h#L271
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal)
            else:
                if self.is_causal:
                    # NOTE 根据 github.com/pytorch/pytorch/issues/115262 和 108108，当Q长度为1时，需要手动传入注意力掩模，而不能简单设置is_causal
                    #      此处参考了Llama3的代码：https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/model.py#L294
                    mask = self.mask[:,:,:T,:T].contiguous().view(T, T)
                    mask = torch.hstack([torch.zeros((T, min(start_pos, self.block_size)), device=q.device), mask]).type_as(q)
                else:
                    mask = None
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            if self.is_causal:
                mask = self.mask[:,:,:T,:T]
            else:
                mask = torch.zeros(self.mask.shape, device=att.device)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = MaskedSelfAttention(config)
        # self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, pos_cis, start_pos, use_kv_cache=False):
        x = x + self.attn(self.norm_1(x), pos_cis, start_pos, use_kv_cache=use_kv_cache)
        x = x + self.mlp(self.norm_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd) if not self.config.use_rope else None,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm_f = RMSNorm(config.n_embd, eps=config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # RoPE relative positional embeddings
        pos_cis = precompute_pos_cis(config.n_embd // config.n_head, config.block_size) if self.config.use_rope else None
        self.register_buffer("pos_cis", pos_cis, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None, start_pos=0, use_kv_cache=False):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # token embeddings of shape (BatchSize, BlockSize, n_embd)
        tok_emb = self.transformer.wte(idx)

        # RoPE
        if self.config.use_rope:
            self.pos_cis = self.pos_cis.to(tok_emb.device)
            pos_cis = self.pos_cis[start_pos : start_pos + t]
        # Trained position embedding
        else:
            pos_cis = None
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)
            pos_emb = self.transformer.wpe(pos)[start_pos : start_pos + t] # position embeddings of shape (BlockSize, n_embd)
            tok_emb = tok_emb + pos_emb

        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x, pos_cis, start_pos, use_kv_cache=use_kv_cache)
        x = self.transformer.norm_f(x)

        # 计算损失
        if targets is not None: # target.shape=(BatchSize, BlockSize)
            logits = self.lm_head(x) # logits.shape=(BatchSize, BlockSize, VocabSize)
            a = logits.view(-1, logits.size(-1)) # shape=(BatchSize*BlockSize, VocabSize)
            b = targets.view(-1) # shape=(BatchSize*BlockSize)
            loss = F.cross_entropy(a, b)
            if loss_mask is not None:
                lm = loss_mask.view(-1) # shape=(BatchSize*BlockSize)
                loss = torch.sum(loss * lm) / lm.sum()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            if self.config.is_causal:
                logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            else:
                logits = self.lm_head(x)
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size]) if not self.config.use_rope else None
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_flops(self, fwdbwd_per_iter, dt):
        # estimate the number of flops (float ops per second) we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flop_per_token = 6*N + 12*L*H*Q*T
        flop_per_fwdbwd = flop_per_token * T
        flop_per_iter = flop_per_fwdbwd * fwdbwd_per_iter
        flops = flop_per_iter * (1.0/dt) # per second
        return flops


    @torch.no_grad()
    def generate_next_token(self, idx, is_prefill=True, temperature=1.0, top_k=None, repetition_penalty=1.1):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        if (not USE_KV_CACHE) or is_prefill:
            logits, _ = self(idx_cond, start_pos=0, use_kv_cache=USE_KV_CACHE) # shape=(BatchSize, BlockSize, VocabSize)
        else:
            start_pos = min(idx_cond.size(1) - 1, self.config.block_size)
            logits, _ = self(idx_cond[:, -1:], start_pos=start_pos, use_kv_cache=True) # shape=(BatchSize, BlockSize, VocabSize)
        logits = logits[:, -1, :]  # shape=(BatchSize, VocabSize)
        # repetition penalty: ref arxiv:1909.05858
        for token in set(idx_cond.tolist()[0]):
            logits[:, token] /= repetition_penalty
        # pluck the logits at the final step and scale by desired temperature
        logits = logits / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    # 自回归解码（以自回归方式逐个生成token，构成所需序列）
    @torch.no_grad()
    def auto_regressive_generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.1, callback=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        prompt_length = idx.size(1)
        if prompt_length > self.config.block_size:
            print(f"提示语太长了QAQ")
            return idx
        for i in range(max_new_tokens):
            idx_next = self.generate_next_token(
                idx,
                is_prefill=(i == 0),
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if prompt_length + i >= self.config.block_size:
                print("...\n(欲言又止..止言又欲..整理思路..忘了说啥 (●'◡'●))")
                return idx
            if callback is not None:
                res = callback(idx_next)
                if res == False:
                    return idx

        return idx

    # 非自回归解码（一次性生成整个序列）
    @torch.no_grad()
    def non_auto_regressive_generate(self, idx, temperature=1.0, top_k=None):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        output_idx = []
        for i in range(logits.size(1)):
            lg = logits[:, i, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
                lg[lg < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(lg, dim=-1)
            output_idx.append(torch.multinomial(probs, num_samples=1))
        return output_idx
