# 
# Nano Language Model
#
#   BD4SUR 2024-10
#
#   Forked from:
#     - https://github.com/karpathy/llama2.c
#
# python export.py qwen3-0b6.bin --hf /path/to/Qwen3-0.6B
# python export.py qwen3-1b7.bin --hf /path/to/Qwen3-1.7B

import json
import struct
import argparse
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


#########################################################
#  模型结构定义
#########################################################

@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: Optional[int] = 8
    head_dim: Optional[int] = 128
    vocab_size: int = 151936
    hidden_dim: Optional[int] = 3072
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-6
    max_seq_len: int = 40960
    dropout: float = 0.0


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


def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0):
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
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = 128 # args.dim // args.n_heads # Qwen3
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.q_norm(self.wq(x)), self.k_norm(self.wk(x)), self.wv(x) # Qwen3
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = 128 # args.dim // args.n_heads # Qwen3
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits


#########################################################
#  数组量化&序列化写入文件
#########################################################

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_base64(file, b64):
    """ writes one base64 bytestring to file that is open in wb mode """
    b = struct.pack(f'{len(b64)}B', *b64)
    file.write(b)

def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

def write_quantized_tensor(out_file, tensor, group_size):
    q, s, err = quantize_q80(tensor, group_size)
    serialize_int8(out_file, q) # save the tensor in int8
    serialize_fp32(out_file, s) # save scale factors
    print(f"Tensor quantized {tuple(tensor.shape)} to Q8_0 with max error {err}")

#########################################################
#  BPE词元编解码器的序列化
#########################################################

# this is a horrible gpt-2 unicode byte encoder hack from https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
# this has poisoned all HF tokenizer configs that use ByteLevel decoder/preprocessor
# as a result we get crazy UTF-8-as-bytes-as-UTF8 in the tokenizer data that we need to convert back
def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def serialize_tokenizer(file):

    token_byte_length = 0

    with open("/home/bd4sur/ai/_model/Qwen3-0.6B/tokenizer.json", "r") as f:
        tokenizer = json.load(f)

    tokens = [""] * 151936
    scores = [0] * 151936

    vocab = tokenizer["model"]["vocab"]

    tokens_gpt2 = not tokenizer["model"].get("byte_fallback", False)

    for t, i in vocab.items():
        tokens[i] = t

    for added in tokenizer["added_tokens"]:
        tokens[added["id"]] = added["content"]

    # compute score as negative merge index so that earlier merges get selected first
    for i, m in enumerate(tokenizer["model"]["merges"]):
        t1, t2 = m[0], m[1] # Qwen3
        ti = vocab[t1 + t2]
        if scores[ti] == 0:
            scores[ti] = -(1 + i)

    # postprocess tokens
    gpt2_decode = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    for i, t in enumerate(tokens):
        if tokens_gpt2:
            b = bytes([gpt2_decode.get(c, 0) for c in t])
        else:
            # t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8')

        b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
        assert b.count(0) == 0 # no null bytes allowed

        tokens[i] = b
        token_byte_length += len(b)

    # record the max token length
    max_token_length = max(len(t) for t in tokens)

    # write to a binary file
    # the tokenizer.bin file is the same as .model file, but .bin

    tokenizer_field_bytes = 4 + 4 + 8 * min(len(tokens), len(scores)) + token_byte_length

    print(f"len(tokens) = {len(tokens)}")
    print(f"len(scores) = {len(scores)}")
    print(f"token_byte_length = {token_byte_length}")

    write_count = 0

    file.write(struct.pack('I', tokenizer_field_bytes))  # 模型文件中词表部分的字节数（不含本字段的4个字节）
    write_count += 4
    file.write(struct.pack("I", max_token_length))
    write_count += 4

    count = 0

    for byte, score in zip(tokens, scores):
        file.write(struct.pack("fI", score, len(byte)))
        write_count += 8
        if count < 100:
            print(f"[{count}] len(byte) = {len(byte)}")
        file.write(byte)
        write_count += len(byte)
        count += 1

    print(f"Write count = {write_count}")


#########################################################
#  模型导出
#########################################################

def export_model(model, filepath, group_size=0):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """

    out_file = open(filepath, 'wb')

    #########################################################
    # 写入文件头（固定长度256B）

    print("Writing header...")

    major_version = 2026
    minor_version = 1

    # 1) write magic, which will be two uint32 of "BD4SURLM" in ASCII
    out_file.write(struct.pack('I', 0x42443453))
    out_file.write(struct.pack('I', 0x55524c4d))
    # --> 8 bytes

    # 2) write version, which will be int
    out_file.write(struct.pack('i', major_version))
    out_file.write(struct.pack('i', minor_version))
    # --> 16 bytes

    # 3) write file type TODO to be defined
    out_file.write(struct.pack('i', 3))  # Model type: Qwen3
    out_file.write(struct.pack('i', 36)) # Config Length: 36 bytes
    # --> 24 bytes

    # 4) write the model config, which will be 9 ints (36 bytes)
    p = model.params

    is_shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    header = struct.pack(
        "iiiiiiiii",
        p.max_seq_len,
        p.vocab_size,
        p.n_layers,
        p.dim,
        p.n_heads,
        p.n_heads if p.n_kv_heads is None else p.n_kv_heads,
        model.layers[0].feed_forward.w1.weight.shape[0],
        int(is_shared_classifier),
        p.head_dim
    )
    out_file.write(header)
    # --> 60 bytes

    # 5) write some other flags (TODO)
    out_file.write(struct.pack('i', 0x00 if group_size == 0 else 0x80))  # 量化类型 见`infer/tensor.h`
    out_file.write(struct.pack('i', group_size)) # 量化参数(分组长度)

    # 6) pad rest with zeros; 'tell' returns current pos
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)


    #########################################################
    # 写入词表

    print("Writing tokenizer...")
    serialize_tokenizer(out_file)


    #########################################################
    # 写入模型参数

    # 如果GS等于0，则为非量化模型
    if group_size == 0:

        print("Writing FP32 model parameters...")

        weights = [
            # No quant
            *[layer.attention_norm.weight for layer in model.layers],
            *[layer.ffn_norm.weight for layer in model.layers],
            model.norm.weight,

            # Quantized
            model.tok_embeddings.weight,
            *[layer.attention.wq.weight for layer in model.layers],
            *[layer.attention.wk.weight for layer in model.layers],
            *[layer.attention.wv.weight for layer in model.layers],
            *[layer.attention.wo.weight for layer in model.layers],
            *[layer.feed_forward.w1.weight for layer in model.layers],
            *[layer.feed_forward.w2.weight for layer in model.layers],
            *[layer.feed_forward.w3.weight for layer in model.layers],

            # Qwen2/Qwen3
            # *[layer.attention.wq.bias for layer in model.layers], # Qwen2
            # *[layer.attention.wk.bias for layer in model.layers], # Qwen2
            # *[layer.attention.wv.bias for layer in model.layers], # Qwen2
            *[layer.attention.q_norm.weight for layer in model.layers], # Qwen3
            *[layer.attention.k_norm.weight for layer in model.layers], # Qwen3

            # Optional
            model.freqs_cos,
            model.freqs_sin,

            # Optional: model.output.weight
        ]
        if not is_shared_classifier:
            weights.append(model.output.weight)

        param_count = 0
        for w in weights:
            param_count += w.detach().cpu().view(-1).numel()

        # 【NOTE 不需要】写入模型参数数（本字段8个字节）
        # out_file.write(struct.pack('Q', param_count)) # unsigned long long - uint64_t

        # 按照上面定义的维度顺序，将模型参数写入文件，没有其他定界符或填充数据
        for w in weights:
            serialize_fp32(out_file, w)

        print(f"Params = {param_count}")
        print(f"Total bin file length = {out_file.tell()}")

    # 如果GS大于0，则为量化模型
    elif group_size > 0:

        print("Writing Q80 quantized model parameters...")

        while p.dim % group_size != 0:
            group_size //= 2
            print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")

        quantized_weights = [
            model.tok_embeddings.weight,
            *[layer.attention.wq.weight for layer in model.layers],
            *[layer.attention.wk.weight for layer in model.layers],
            *[layer.attention.wv.weight for layer in model.layers],
            *[layer.attention.wo.weight for layer in model.layers],
            *[layer.feed_forward.w1.weight for layer in model.layers],
            *[layer.feed_forward.w2.weight for layer in model.layers],
            *[layer.feed_forward.w3.weight for layer in model.layers],
        ]

        for w in quantized_weights:
            assert w.numel() % group_size == 0, f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"


        #########################################################
        # 量化并写入模型参数

        # 首先写入不量化的各层参数

        for layer in model.layers:
            serialize_fp32(out_file, layer.attention_norm.weight)

        for layer in model.layers:
            serialize_fp32(out_file, layer.ffn_norm.weight)

        serialize_fp32(out_file, model.norm.weight)

        # 写入量化的各层参数

        for i, w in enumerate(quantized_weights):
            write_quantized_tensor(out_file, w, group_size)

        # 写入Qwen3的q/k_norm参数

        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.q_norm.weight)
        for layer in model.layers:
            serialize_fp32(out_file, layer.attention.k_norm.weight)

        # 写入RoPE参数

        serialize_fp32(out_file, model.freqs_cos)
        serialize_fp32(out_file, model.freqs_sin)

        # 最后，如果token嵌入层不共享，则按需写入输出解码层

        if not is_shared_classifier:
            w = model.output.weight
            assert w.numel() % group_size == 0, f"output.weight has numel {w.numel()}, not a multiple of group_size {group_size}"
            write_quantized_tensor(out_file, w, group_size)



    #########################################################
    # 写入并关闭文件

    out_file.close()
    print(f"wrote {filepath}")




#########################################################
#  载入HuggingFace模型
#########################################################

def load_hf_model(model_path):

    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    # convert LlamaConfig to ModelArgs
    config = ModelArgs()
    config.dim = hf_model.config.hidden_size
    config.n_layers = hf_model.config.num_hidden_layers
    config.n_heads = hf_model.config.num_attention_heads
    config.n_kv_heads = hf_model.config.num_key_value_heads
    config.head_dim = hf_model.config.head_dim
    config.vocab_size = hf_model.config.vocab_size
    config.hidden_dim = hf_model.config.intermediate_size
    config.norm_eps = hf_model.config.rms_norm_eps
    config.max_seq_len = hf_model.config.max_position_embeddings

    # create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])

    head_dim = 128 # config.dim // config.n_heads # Qwen3

    # huggingface permutes WQ and WK, this function reverses it
    # see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
    def permute_reverse(w, heads, rotary_dim):
        head_dim = 128 # w.shape[0] // heads # Qwen3
        assert rotary_dim <= head_dim
        w = torch.unflatten(w, 0, (-1, head_dim))
        # wr is the rotary part, wk is the part kept unrotated
        wr = w[:, :rotary_dim]
        wk = w[:, rotary_dim:]
        # switch wr from outputting two rotary_dim/2 chunks to outputting values interleaved
        wr = torch.unflatten(wr, 1, (2, -1))
        wr = wr.transpose(1, 2)
        wr = wr.flatten(1, 2)
        # assemble the heads back
        w = torch.cat([wr, wk], dim=1)
        return torch.flatten(w, 0, 1)

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.input_layernorm.weight'])
        print(f"Layer {i} attention_norm.shape = {layer.attention_norm.weight.shape}")

        # layer.attention.wq.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight'], config.n_heads, head_dim))
        # layer.attention.wk.weight = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight'], config.n_kv_heads, head_dim))
        layer.attention.wq.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.q_proj.weight'])
        layer.attention.wk.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.k_proj.weight'])
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
        print(f"Layer {i} wq.shape = {layer.attention.wq.weight.shape}")
        print(f"Layer {i} wk.shape = {layer.attention.wk.weight.shape}")
        print(f"Layer {i} wv.shape = {layer.attention.wv.weight.shape}")
        print(f"Layer {i} wo.shape = {layer.attention.wo.weight.shape}")

        # Qwen2
        # layer.attention.wq.bias = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.q_proj.bias'], config.n_heads, head_dim))
        # layer.attention.wk.bias = nn.Parameter(permute_reverse(hf_dict[f'model.layers.{i}.self_attn.k_proj.bias'], config.n_kv_heads, head_dim))
        # layer.attention.wv.bias = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.bias'])

        # Qwen3
        layer.attention.q_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.q_norm.weight'])
        layer.attention.k_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.k_norm.weight'])
        print(f"Layer {i} q_norm.shape = {layer.attention.q_norm.weight.shape}")
        print(f"Layer {i} k_norm.shape = {layer.attention.k_norm.weight.shape}")

        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'])
        print(f"Layer {i} ffn_norm.shape = {layer.ffn_norm.weight.shape}")
        print(f"Layer {i} w1.shape = {layer.feed_forward.w1.weight.shape}")
        print(f"Layer {i} w2.shape = {layer.feed_forward.w2.weight.shape}")
        print(f"Layer {i} w3.shape = {layer.feed_forward.w3.weight.shape}")

    # final classifier
    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
    print(f"output.shape = {model.output.weight.shape}")
    model.eval()
    return model


#########################################################
#  入口点
#########################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--gs", default=0, type=int, help="Quant group size, default=0. if gs == 0 no quant; else if gs>0 quant")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--hf", type=str, help="HuggingFace model checkpoint")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.hf:
        model = load_hf_model(args.hf)
        export_model(model, args.filepath, args.gs)
