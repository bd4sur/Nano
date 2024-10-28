"""
Forked from https://github.com/karpathy/nanoGPT
BD4SUR 2023.12
"""

import math
import inspect
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# NOTE KV-Cache仅为实验性实现，主要影响自回归生成。该标识通过forward逐层传递到Attn层，训练时默认不使用。
#      KV-Cache可缓解长文本生成后期TPS大幅下降的问题，但是对于短文本的生成速度未必有改善（取决于SDPA算子的性能）。
USE_KV_CACHE = True

@dataclass
class ModelConfig:
    block_size: int = 512
    vocab_size: int = 16384
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 16
    n_kv_head: Optional[int] = None
    n_hidden: Optional[int] = 1408 # ((n_embd * 8 // 3) + 63) // 64 * 64
    dropout: float = 0.0
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

    # LoRA settings
    use_lora: Optional[bool] = False
    lora_rank :Optional[int] = 16
    lora_alpha :Optional[int] = 32
    lora_dropout :Optional[float] = 0.0

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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_head, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_head, n_rep, head_dim)
        .reshape(bs, slen, n_kv_head * n_rep, head_dim)
    )


class LoRA(torch.nn.Module):
    def __init__(self, target_module: torch.nn.Module, config: ModelConfig, lora_rank=16, lora_alpha=32, lora_dropout=0.0):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.w = target_module
        self.lora_a = nn.Linear(self.w.in_features, self.lora_rank, bias=False)
        self.lora_b = nn.Linear(self.lora_rank, self.w.out_features, bias=False)
        self.lora_input_dropout = nn.Dropout(p=self.lora_dropout)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        out1 = self.w(x)
        out2 = self.lora_b(self.lora_a(self.lora_input_dropout(x))) * (self.lora_alpha / self.lora_rank)
        return out1 + out2


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


class Attention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_kv_head = config.n_head if config.n_kv_head is None else config.n_kv_head
        assert config.n_embd % config.n_head == 0
        assert config.n_head % self.n_kv_head == 0

        model_parallel_size = 1
        self.n_local_heads = config.n_head // model_parallel_size
        self.n_local_kv_heads = self.n_kv_head // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.n_embd // config.n_head

        # query, key, value projections
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # output projection
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # KV-Cache (experimental)
        self.cache_k = None
        self.cache_v = None

        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

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

    def forward(self, x, freqs_cos, freqs_sin, start_pos, use_kv_cache=False):
        B, S, E = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(B, S, self.n_local_heads, self.head_dim)    # (B, S, h, E//h)
        xk = xk.view(B, S, self.n_local_kv_heads, self.head_dim) # (B, S, m, E//h)
        xv = xv.view(B, S, self.n_local_kv_heads, self.head_dim) # (B, S, m, E//h)

        # RoPE
        if freqs_cos is not None and freqs_sin is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (B, S, h, E//h)
        xv = repeat_kv(xv, self.n_rep)  # (B, S, h, E//h)

        if use_kv_cache and (not self.training):
            if start_pos == 0 or (self.cache_k is None and self.cache_v is None):
                self.cache_k = torch.zeros((B, self.block_size, self.n_local_heads, self.head_dim)).to(xq.device)
                self.cache_v = torch.zeros((B, self.block_size, self.n_local_heads, self.head_dim)).to(xq.device)

            self.cache_k[:B, start_pos : start_pos + S] = xk
            self.cache_v[:B, start_pos : start_pos + S] = xv

            xk = self.cache_k[:B, : start_pos + S]
            xv = self.cache_v[:B, : start_pos + S]

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (B, h, S, E//h)
        xk = xk.transpose(1, 2)  # (B, h, S, E//h)
        xv = xv.transpose(1, 2)  # (B, h, S, E//h)

        # causal self-attention; Self-attend: (B, h, S, E//h) x (B, h, E//h, S) -> (B, h, S, S)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if self.training or (not use_kv_cache):
                # NOTE flash_attn尚不支持非None的attn_mask
                # Ref. https://github.com/pytorch/pytorch/blob/753ba5d30a361be4f610cf7dde4fd63726ed8f86/aten/src/ATen/native/transformers/sdp_utils_cpp.h#L271
                output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal)
            else:
                if self.is_causal:
                    # NOTE 根据 github.com/pytorch/pytorch/issues/115262 和 108108，当Q长度为1时，需要手动传入注意力掩模，而不能简单设置is_causal
                    #      此处参考了Llama3的代码：https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/model.py#L294
                    mask = self.mask[:,:,:S,:S].contiguous().view(S, S)
                    mask = torch.hstack([torch.zeros((S, min(start_pos, self.block_size)), device=xq.device), mask]).type_as(xq)
                else:
                    mask = None
                output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            if self.is_causal:
                mask = self.mask[:,:,:S,:S]
            else:
                mask = torch.zeros(self.mask.shape, device=att.device)
            att = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            att = att + mask
            att = F.softmax(att.float(), dim=-1).type_as(xq)
            att = self.attn_dropout(att)
            output = att @ xv # (B, h, S, S) x (B, h, S, E//h) -> (B, h, S, E//h)

        # re-assemble all head outputs side by side
        output = output.transpose(1, 2).contiguous().view(B, S, -1)

        # output projection
        output = self.resid_dropout(self.wo(output))

        return output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        _multiple_of = 256
        if config.n_hidden is None:
            n_hid = int(8 * config.n_embd / 3)
            n_hid = _multiple_of * ((n_hid + _multiple_of - 1) // _multiple_of)
        else:
            n_hid = config.n_hidden
        self.w1 = nn.Linear(config.n_embd, n_hid, bias=False)
        self.w2 = nn.Linear(n_hid, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, n_hid, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x, freqs_cos, freqs_sin, start_pos, use_kv_cache=False):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, start_pos, use_kv_cache=use_kv_cache)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class GPT(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer

        self.is_lora = False

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd) if not self.config.use_rope else None
        self.dropout = nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, layer_id))
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # precompute for the RoPE factors
        if self.config.use_rope:
            freqs_cos, freqs_sin = precompute_freqs_cis(self.config.n_embd // self.config.n_head, self.config.block_size)
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None, loss_mask=None, start_pos=0, use_kv_cache=False):
        B, S = idx.size()
        assert S <= self.config.block_size, f"Cannot forward sequence of length {S}, block size is only {self.config.block_size}"

        h = self.tok_embeddings(idx)

        # RoPE
        if self.config.use_rope:
            freqs_cos = self.freqs_cos[start_pos : start_pos + S].to(h.device)
            freqs_sin = self.freqs_sin[start_pos : start_pos + S].to(h.device)
        else:
            freqs_cos, freqs_sin = None, None
            pos = torch.arange(0, S, dtype=torch.long, device=idx.device)
            pos_emb = self.wpe(pos)[start_pos : start_pos + S] # position embeddings
            h = h + pos_emb

        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, start_pos, use_kv_cache=use_kv_cache)

        h = self.norm(h)

        # 计算损失
        if targets is not None: # target.shape=(B, S)
            logits = self.output(h) # logits.shape=(B, S, V)
            a = logits.view(-1, logits.size(-1)) # shape=(B*S, V)
            b = targets.view(-1) # shape=(B*S)
            loss = F.cross_entropy(a, b)
            if loss_mask is not None:
                lm = loss_mask.view(-1) # shape=(B*S)
                loss = torch.sum(loss * lm) / lm.sum()
            self.last_loss = loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            if self.config.is_causal:
                logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            else:
                logits = self.output(h)
            self.last_loss = None
        return logits, self.last_loss

    # 将模型转为LoRA模型（在wq、wk、wv、wo上附加低秩分解旁路并初始化，同时冻结除LoRA层之外所有其他参数）
    def to_lora(self, lora_rank=16, lora_alpha=32, lora_dropout=0.0):
        self.is_lora = True
        for layer in self.layers:
            layer.attention.wq = LoRA(layer.attention.wq, self.config, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            layer.attention.wk = LoRA(layer.attention.wk, self.config, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            layer.attention.wv = LoRA(layer.attention.wv, self.config, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            layer.attention.wo = LoRA(layer.attention.wo, self.config, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        for pname, p in self.named_parameters():
            if "lora" in pname:
                p.requires_grad = True
            else:
                p.requires_grad = False

    # TODO 待实现：将LoRA模块的参数融合进基座，删除低秩适配分支，模型转回非LoRA模型
    def merge_lora(self):
        pass

    # 获取LoRA模型的低秩适配层的参数
    def get_lora_state_dict(self):
        if not self.is_lora:
            return False
        lora_state_dict = {}
        for k, v in self.state_dict().items():
            if "lora" in k:
                lora_state_dict[k] = v
        return lora_state_dict

    # 将LoRA模型的低秩适配层的参数，载入LoRA模型（基座模型保持不动）
    def load_lora_state_dict(self, lora_state_dict):
        if not self.is_lora:
            return False
        self.load_state_dict(lora_state_dict, strict=False, assign=False)


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

    # 计算总参数量（含冻结参数）
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            n_params -= self.wpe.weight.numel()
        return n_params

    # 计算参与训练的参数量（例如LoRA训练中只有LoRA层参与训练）
    def get_num_params_train(self, non_embedding=True):
        n_params_train = sum(p.numel() if p.requires_grad else 0 for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            n_params_train -= self.wpe.weight.numel()
        return n_params_train

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
