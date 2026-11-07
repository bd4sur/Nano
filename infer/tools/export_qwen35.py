#!/usr/bin/env python3
#
# Nano Language Model - Qwen3.5 weight exporter
#
#   BD4SUR 2026-11
#
# 将 HuggingFace Qwen3.5 模型（文本部分）导出为 Nano 推理引擎使用的 .bin 格式。
#
# 只依赖 numpy，不依赖 torch / transformers / safetensors：
#   - safetensors 文件按官方格式直接用 mmap 读取
#   - bf16 通过 uint16 左移 16 位转换为 fp32
#
# 用法：
#   python3 infer/tools/export_qwen35.py --hf /path/to/Qwen3.5-0.8B -o qwen35.f32.bin --quant f32
#   python3 infer/tools/export_qwen35.py --hf /path/to/Qwen3.5-0.8B -o qwen35.q80.bin --quant q80 --gs 32
#   python3 infer/tools/export_qwen35.py --hf /path/to/Qwen3.5-0.8B -o qwen35.q4k.bin --quant q4k
#
# 兼容旧式参数（与 export_qwen.py 一致）：
#   python3 infer/tools/export_qwen35.py qwen35.q80.bin --hf DIR --gs 32
#   python3 infer/tools/export_qwen35.py qwen35.f32.bin --hf DIR --gs 0
#

import argparse
import glob
import json
import mmap
import os
import struct
import sys

import numpy as np


#########################################################
#  常量
#########################################################

LLM_ARCH_QWEN35 = 4

QUANT_TYPE_F32 = 0x00
QUANT_TYPE_Q80 = 0x80
QUANT_TYPE_Q4K = 0x42

# 模型文件头固定长度
HEADER_BYTE_LENGTH = 256
# 文件头中 quant_type / group_size 的偏移
HEADER_QUANT_TYPE_OFFSET = 60
HEADER_GROUP_SIZE_OFFSET = 64

# Qwen3.5 文本模型的 RoPE 表在推理时由 C 代码按 rope_theta 生成。
# 这里限制 block_size，避免 max_position_embeddings=262144 导致 RoPE 表过大。
MAX_BLOCK_SIZE = 40960


#########################################################
#  safetensors 读取
#########################################################

class SafeTensorFile:
    """极简 safetensors 读取器（只读、mmap、支持 BF16/F16/F32）。"""

    def __init__(self, path):
        self.path = path
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack("<Q", self._mm[:8])[0]
        self.header = json.loads(self._mm[8:8 + header_len].decode("utf-8"))
        self.data_offset = 8 + header_len

    def names(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def has(self, name):
        return name in self.header

    def get_f32(self, name):
        entry = self.header[name]
        dtype = entry["dtype"]
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        count = 1
        for d in shape:
            count *= d
        offset = self.data_offset + start

        if dtype == "BF16":
            u = np.frombuffer(self._mm, dtype="<u2", count=count, offset=offset)
            arr = (u.astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(self._mm, dtype="<f2", count=count, offset=offset).astype(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(self._mm, dtype="<f4", count=count, offset=offset)
        else:
            raise ValueError(f"unsupported dtype {dtype} for tensor {name}")

        return np.ascontiguousarray(arr.reshape(shape), dtype=np.float32)


class SafeTensorSet:
    """支持单文件、HF 原始文件名以及 sharded safetensors。"""

    def __init__(self, hf_dir):
        index_path = os.path.join(hf_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted(set(weight_map.values()))
            paths = [os.path.join(hf_dir, f) for f in files]
        else:
            paths = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
            # 排除可能的 adapter 文件，优先使用 model.safetensors*
            model_paths = [p for p in paths if os.path.basename(p).startswith("model")]
            if model_paths:
                paths = model_paths

        if not paths:
            raise FileNotFoundError(f"no .safetensors file found in {hf_dir}")

        self.files = [SafeTensorFile(p) for p in paths]
        self.tensor_to_file = {}
        for sf in self.files:
            for name in sf.names():
                self.tensor_to_file[name] = sf
        print(f"loaded {len(self.files)} safetensors file(s), {len(self.tensor_to_file)} tensors")

    def has(self, name):
        return name in self.tensor_to_file

    def get_f32(self, name):
        return self.tensor_to_file[name].get_f32(name)


#########################################################
#  模型结构参数
#########################################################

class ModelArgs:
    pass


def load_model_args(hf_dir):
    with open(os.path.join(hf_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tc = cfg.get("text_config", cfg)

    a = ModelArgs()
    a.dim = tc["hidden_size"]
    a.n_layers = tc["num_hidden_layers"]
    a.n_heads = tc["num_attention_heads"]
    a.n_kv_heads = tc["num_key_value_heads"]
    a.head_dim = tc.get("head_dim") or (a.dim // a.n_heads)
    a.vocab_size = tc["vocab_size"]
    a.hidden_dim = tc["intermediate_size"]
    a.norm_eps = tc.get("rms_norm_eps", 1e-6)
    a.max_seq_len = tc.get("max_position_embeddings", 32768)

    # Qwen3.5 混合注意力：linear_attention / full_attention
    a.layer_types = tc["layer_types"]
    a.layer_type_mask = 0
    for i, t in enumerate(a.layer_types):
        if t == "full_attention":
            a.layer_type_mask |= (1 << i)
    a.n_full_layers = sum(1 for t in a.layer_types if t == "full_attention")
    a.n_linear_layers = a.n_layers - a.n_full_layers

    # Gated DeltaNet（线性注意力）
    a.conv_kernel_dim = tc["linear_conv_kernel_dim"]
    a.linear_n_k_head = tc["linear_num_key_heads"]
    a.linear_n_v_head = tc["linear_num_value_heads"]
    a.linear_k_head_dim = tc["linear_key_head_dim"]
    a.linear_v_head_dim = tc["linear_value_head_dim"]

    # RoPE（部分旋转 + mRoPE 配置；纯文本时退化为普通 1D RoPE）
    rope = tc.get("rope_parameters", {})
    a.rope_theta = float(rope.get("rope_theta", 10000000.0))
    a.partial_rotary_factor = float(rope.get("partial_rotary_factor", 1.0))
    a.rope_dim = int(a.head_dim * a.partial_rotary_factor)
    if a.rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {a.rope_dim}")

    a.tie_word_embeddings = bool(tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    a.is_shared_classifier = 1 if a.tie_word_embeddings else 0

    # 推理引擎只保存有限长度的 RoPE 表
    a.block_size = min(int(a.max_seq_len), MAX_BLOCK_SIZE)

    # 派生维度
    a.q_dim = a.n_heads * a.head_dim
    a.kv_dim = a.n_kv_heads * a.head_dim
    a.key_dim = a.linear_n_k_head * a.linear_k_head_dim
    a.value_dim = a.linear_n_v_head * a.linear_v_head_dim
    a.conv_dim = 2 * a.key_dim + a.value_dim

    return a


#########################################################
#  tokenizer 序列化
#########################################################

def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def serialize_tokenizer(out_file, hf_dir, vocab_size):
    with open(os.path.join(hf_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    tokens = [""] * vocab_size
    scores = [0] * vocab_size

    vocab = tokenizer["model"]["vocab"]
    tokens_gpt2 = not tokenizer["model"].get("byte_fallback", False)

    for t, i in vocab.items():
        if i < vocab_size:
            tokens[i] = t

    for added in tokenizer.get("added_tokens", []):
        if added["id"] < vocab_size:
            tokens[added["id"]] = added["content"]

    # 负的 merge 序号作为分数：越早的 merge 分数越高
    for i, m in enumerate(tokenizer["model"].get("merges", [])):
        if isinstance(m, str):
            parts = m.split(" ")
            if len(parts) != 2:
                continue
            t1, t2 = parts[0], parts[1]
        else:
            t1, t2 = m[0], m[1]
        ti = vocab.get(t1 + t2)
        if ti is not None and ti < vocab_size and scores[ti] == 0:
            scores[ti] = -(1 + i)

    gpt2_decode = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    encoded = []
    token_byte_length = 0
    for i, t in enumerate(tokens):
        if tokens_gpt2:
            b = bytes([gpt2_decode.get(c, 0) for c in t])
        else:
            b = t.encode("utf-8")
        b = b.replace(b"\0", b"\7")
        assert b.count(0) == 0
        encoded.append(b)
        token_byte_length += len(b)

    max_token_length = max((len(t) for t in encoded), default=0)

    tokenizer_field_bytes = 4 + 4 + 8 * vocab_size + token_byte_length

    print(f"len(tokens) = {vocab_size}")
    print(f"token_byte_length = {token_byte_length}")
    print(f"max_token_length = {max_token_length}")

    out_file.write(struct.pack("I", tokenizer_field_bytes))
    out_file.write(struct.pack("I", max_token_length))
    for b, score in zip(encoded, scores):
        out_file.write(struct.pack("fI", score, len(b)))
        if b:
            out_file.write(b)

    return tokenizer_field_bytes


#########################################################
#  Q80 量化
#########################################################

def quantize_q80(w, group_size):
    """对称 int8 量化，与 export_qwen.py / tensor.c 的 Q80 布局一致。"""
    w = np.ascontiguousarray(w, dtype=np.float32).reshape(-1, group_size)
    wmax = np.abs(w).max(axis=1)
    scale = (wmax / 127.0).astype(np.float32)

    q = np.zeros_like(w)
    nz = scale > 0
    if np.any(nz):
        q[nz] = np.rint(w[nz] / scale[nz, None])
    q = np.clip(q, -127, 127).astype(np.int8)

    # 计算一下最大误差，便于观察量化质量
    fp32val = q.astype(np.float32) * scale[:, None]
    maxerr = float(np.abs(fp32val - w).max()) if w.size else 0.0
    return q, scale, maxerr


def write_q80_tensor(out_file, arr, group_size, label, verbose=True):
    q, s, err = quantize_q80(arr, group_size)
    q.tofile(out_file)
    s.astype(np.float32).tofile(out_file)
    if verbose:
        print(f"  Q80 {label:24s} shape={tuple(arr.shape)} maxerr={err:.6g}")
    return q.size


def write_q80_group(out_file, arr, ndim, group_size, label):
    """C 端 parse_quantized_tensors 按层读取，因此 3D 张量必须逐层写 q/s。"""
    if ndim == 3:
        total = 0
        for i in range(arr.shape[0]):
            total += write_q80_tensor(out_file, arr[i], group_size, f"{label}[{i}]", verbose=False)
        print(f"  Q80 {label:24s} shape={tuple(arr.shape)} layers={arr.shape[0]}")
        return total
    return write_q80_tensor(out_file, arr, group_size, label)


#########################################################
#  Q4K 量化
#########################################################

Q4K_BLOCK_LEN = 256
Q4K_GROUP_LEN = 32
Q4K_GROUP_NUM = 8
Q4K_BLOCK_BYTES = 160
Q4K_TENSOR_HEADER_BYTES = 44

_Q4K_BLOCK_DTYPE = np.dtype([
    ("header", "<u4"),
    ("length", "<u4"),
    ("meta", "<u4"),
    ("s_scale", "<f4"),
    ("s_bias", "<f4"),
    ("sb", "u1", (12,)),
    ("value", "u1", (128,)),
])
assert _Q4K_BLOCK_DTYPE.itemsize == Q4K_BLOCK_BYTES


def _quantize_q4k_blocks(arr, chunk_blocks=4096):
    """把一维 float32 数组按 256 个元素一块量化，逐块 yield 打包字节。"""
    arr = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1, Q4K_BLOCK_LEN)
    total_blocks = arr.shape[0]
    # C 端使用 FLT_TRUE_MIN（最小的次正规数，约 1.4e-45），不是 FLT_MIN
    tiny = np.float32(1.401298464324817e-45)

    for start in range(0, total_blocks, chunk_blocks):
        blocks = arr[start:start + chunk_blocks]
        n = blocks.shape[0]
        g = blocks.reshape(n, Q4K_GROUP_NUM, Q4K_GROUP_LEN)

        mn = g.min(axis=2)
        mx = np.maximum(g.max(axis=2), tiny)  # 与 tensor.c 的 FLT_TRUE_MIN 初值一致

        s = np.where(mn <= 0.0, (mx - mn) / 15.0, mx / 15.0).astype(np.float32)
        b = np.where(mn <= 0.0, -mn, 0.0).astype(np.float32)

        q = np.zeros_like(g, dtype=np.float32)
        nz = s > 0
        if np.any(nz):
            q[nz] = np.rint((g[nz] + b[nz][:, None]) / s[nz][:, None])
        q = np.clip(q, 0, 15).astype(np.uint8).reshape(n, Q4K_BLOCK_LEN)

        values = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)

        # 与 tensor.c 中 s_max/b_max 以 FLT_TRUE_MIN 为初值保持一致
        s_max = np.maximum(s.max(axis=1), tiny)
        b_max = np.maximum(b.max(axis=1), tiny)
        s_scale = (s_max / 63.0).astype(np.float32)
        b_scale = (b_max / 63.0).astype(np.float32)

        sq = np.zeros_like(s, dtype=np.uint8)
        bq = np.zeros_like(b, dtype=np.uint8)
        nzs = s_scale > 0
        nzb = b_scale > 0
        if np.any(nzs):
            sq[nzs] = (np.rint(s[nzs] / s_scale[nzs, None]).astype(np.uint8) & 0x3F)
        if np.any(nzb):
            bq[nzb] = (np.rint(b[nzb] / b_scale[nzb, None]).astype(np.uint8) & 0x3F)

        sb = np.empty((n, 12), dtype=np.uint8)
        sb[:, 0] = ((sq[:, 4] & 0x30) << 2) | (sq[:, 0] & 0x3F)
        sb[:, 1] = ((sq[:, 5] & 0x30) << 2) | (sq[:, 1] & 0x3F)
        sb[:, 2] = ((sq[:, 6] & 0x30) << 2) | (sq[:, 2] & 0x3F)
        sb[:, 3] = ((sq[:, 7] & 0x30) << 2) | (sq[:, 3] & 0x3F)
        sb[:, 4] = ((bq[:, 4] & 0x30) << 2) | (bq[:, 0] & 0x3F)
        sb[:, 5] = ((bq[:, 5] & 0x30) << 2) | (bq[:, 1] & 0x3F)
        sb[:, 6] = ((bq[:, 6] & 0x30) << 2) | (bq[:, 2] & 0x3F)
        sb[:, 7] = ((bq[:, 7] & 0x30) << 2) | (bq[:, 3] & 0x3F)
        sb[:, 8] = ((bq[:, 4] & 0x0F) << 4) | (sq[:, 4] & 0x0F)
        sb[:, 9] = ((bq[:, 5] & 0x0F) << 4) | (sq[:, 5] & 0x0F)
        sb[:, 10] = ((bq[:, 6] & 0x0F) << 4) | (sq[:, 6] & 0x0F)
        sb[:, 11] = ((bq[:, 7] & 0x0F) << 4) | (sq[:, 7] & 0x0F)

        out = np.empty(n, dtype=_Q4K_BLOCK_DTYPE)
        out["header"] = QUANT_TYPE_Q4K
        out["length"] = Q4K_BLOCK_LEN
        out["meta"] = 0
        out["s_scale"] = s_scale
        out["s_bias"] = b_scale
        out["sb"] = sb
        out["value"] = values

        yield out.tobytes()


def write_q4k_tensor(out_file, arr, ndim, shape, label):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    numel = arr.size
    if numel % Q4K_BLOCK_LEN != 0:
        raise ValueError(f"{label}: numel {numel} is not a multiple of {Q4K_BLOCK_LEN}")
    # Q4K 按行（末维）分块；块不能跨行，否则 C 端反量化/矩阵乘会错位
    if shape[-1] % Q4K_BLOCK_LEN != 0:
        raise ValueError(f"{label}: last dim {shape[-1]} is not a multiple of {Q4K_BLOCK_LEN}")

    num_blocks = numel // Q4K_BLOCK_LEN
    total_bytes = Q4K_TENSOR_HEADER_BYTES + num_blocks * Q4K_BLOCK_BYTES

    padded_shape = list(shape) + [0] * (6 - len(shape))
    out_file.write(struct.pack("<QIII6I", total_bytes, QUANT_TYPE_Q4K, ndim, *padded_shape, num_blocks))

    for chunk in _quantize_q4k_blocks(arr):
        out_file.write(chunk)

    print(f"  Q4K {label:24s} shape={tuple(shape)} blocks={num_blocks}")


#########################################################
#  模型导出
#########################################################

def write_header(out_file, args, quant_type, group_size):
    out_file.write(struct.pack("I", 0x42443453))  # "BD4S"
    out_file.write(struct.pack("I", 0x55524C4D))  # "URLM"
    out_file.write(struct.pack("i", 2026))
    out_file.write(struct.pack("i", 2))

    # Qwen3.5 扩展配置：36 + 8*4 = 68 字节
    out_file.write(struct.pack("i", LLM_ARCH_QWEN35))
    out_file.write(struct.pack("i", 68))

    out_file.write(struct.pack(
        "iiiiiiiii",
        args.block_size,
        args.vocab_size,
        args.n_layers,
        args.dim,
        args.n_heads,
        args.n_kv_heads,
        args.hidden_dim,
        args.is_shared_classifier,
        args.head_dim,
    ))

    out_file.write(struct.pack("i", quant_type))
    out_file.write(struct.pack("i", group_size))

    out_file.write(struct.pack(
        "IIIIIIIf",
        args.layer_type_mask,
        args.conv_kernel_dim,
        args.linear_n_k_head,
        args.linear_n_v_head,
        args.linear_k_head_dim,
        args.linear_v_head_dim,
        args.rope_dim,
        args.rope_theta,
    ))

    pad = HEADER_BYTE_LENGTH - out_file.tell()
    assert pad >= 0
    out_file.write(b"\0" * pad)


def export_model(st, hf_dir, args, out_path, quant_type, group_size):
    n_layer = args.n_layers
    n_full = args.n_full_layers
    n_lin = args.n_linear_layers
    prefix = "model.language_model"

    out_file = open(out_path, "wb")
    print(f"Writing {out_path} ...")
    write_header(out_file, args, quant_type, group_size)

    print("Writing tokenizer...")
    serialize_tokenizer(out_file, hf_dir, args.vocab_size)

    #########################################################
    # 非量化参数（所有量化类型都按 fp32 写入）

    print("Writing fp32 parameters (norms / conv / gating)...")

    def wf(name, extra=0.0):
        a = st.get_f32(name)
        if extra != 0.0:
            a = a + np.float32(extra)
        return np.ascontiguousarray(a, dtype=np.float32)

    def write_f32(a):
        np.ascontiguousarray(a, dtype=np.float32).tofile(out_file)

    # 输入/后注意力 RMSNorm 是 zero-centered：forward 中使用 (1 + weight)
    input_norms = []
    post_norms = []
    for i in range(n_layer):
        input_norms.append(wf(f"{prefix}.layers.{i}.input_layernorm.weight", 1.0))
        post_norms.append(wf(f"{prefix}.layers.{i}.post_attention_layernorm.weight", 1.0))

    for a in input_norms:
        write_f32(a)
    for a in post_norms:
        write_f32(a)
    write_f32(wf(f"{prefix}.norm.weight", 1.0))

    # 线性注意力层（紧凑排列）
    lin_layers = [i for i, t in enumerate(args.layer_types) if t == "linear_attention"]
    full_layers = [i for i, t in enumerate(args.layer_types) if t == "full_attention"]
    assert len(lin_layers) == n_lin and len(full_layers) == n_full

    lin_conv = []
    lin_a_log = []
    lin_dt_bias = []
    lin_norm = []
    lin_proj_b = []
    lin_proj_a = []
    lin_qkv = []
    lin_z = []
    lin_o = []

    for i in lin_layers:
        p = f"{prefix}.layers.{i}.linear_attn"
        lin_conv.append(wf(f"{p}.conv1d.weight").reshape(-1))
        lin_a_log.append(wf(f"{p}.A_log"))
        lin_dt_bias.append(wf(f"{p}.dt_bias"))
        lin_norm.append(wf(f"{p}.norm.weight"))
        lin_proj_b.append(wf(f"{p}.in_proj_b.weight"))
        lin_proj_a.append(wf(f"{p}.in_proj_a.weight"))
        lin_qkv.append(wf(f"{p}.in_proj_qkv.weight"))
        lin_z.append(wf(f"{p}.in_proj_z.weight"))
        lin_o.append(wf(f"{p}.out_proj.weight"))

    for a in lin_conv:
        write_f32(a)
    for a in lin_a_log:
        write_f32(a)
    for a in lin_dt_bias:
        write_f32(a)
    for a in lin_norm:
        write_f32(a)
    for a in lin_proj_b:
        write_f32(a)
    for a in lin_proj_a:
        write_f32(a)

    # 全注意力层的 q_norm / k_norm（zero-centered）
    q_norm = []
    k_norm = []
    wq = []
    wk = []
    wv = []
    wo = []
    for i in full_layers:
        p = f"{prefix}.layers.{i}.self_attn"
        q_norm.append(wf(f"{p}.q_norm.weight", 1.0))
        k_norm.append(wf(f"{p}.k_norm.weight", 1.0))
        wq.append(wf(f"{p}.q_proj.weight"))
        wk.append(wf(f"{p}.k_proj.weight"))
        wv.append(wf(f"{p}.v_proj.weight"))
        wo.append(wf(f"{p}.o_proj.weight"))

    for a in q_norm:
        write_f32(a)
    for a in k_norm:
        write_f32(a)

    #########################################################
    # 量化/浮点主参数

    embed = wf(f"{prefix}.embed_tokens.weight")
    if args.is_shared_classifier:
        lm_head = embed
    else:
        lm_head = wf(f"{prefix}.lm_head.weight") if st.has(f"{prefix}.lm_head.weight") else wf("lm_head.weight")

    w1 = []
    w2 = []
    w3 = []
    for i in range(n_layer):
        p = f"{prefix}.layers.{i}.mlp"
        w1.append(wf(f"{p}.gate_proj.weight"))
        w2.append(wf(f"{p}.down_proj.weight"))
        w3.append(wf(f"{p}.up_proj.weight"))

    # 按层拼接，保持 C 端 (n_tensors, d, n) 的布局
    def stack(xs, label):
        a = np.stack(xs, axis=0)
        print(f"  stack {label:24s} shape={a.shape}")
        return a

    quant_tensors = [
        ("token_embedding", embed, 2, (args.vocab_size, args.dim)),
        ("mlp.w1", stack(w1, "mlp.w1"), 3, (n_layer, args.hidden_dim, args.dim)),
        ("mlp.w2", stack(w2, "mlp.w2"), 3, (n_layer, args.dim, args.hidden_dim)),
        ("mlp.w3", stack(w3, "mlp.w3"), 3, (n_layer, args.hidden_dim, args.dim)),
        ("attn.wq", stack(wq, "attn.wq"), 3, (n_full, 2 * args.q_dim, args.dim)),
        ("attn.wk", stack(wk, "attn.wk"), 3, (n_full, args.kv_dim, args.dim)),
        ("attn.wv", stack(wv, "attn.wv"), 3, (n_full, args.kv_dim, args.dim)),
        ("attn.wo", stack(wo, "attn.wo"), 3, (n_full, args.dim, args.q_dim)),
        ("linear.wqkv", stack(lin_qkv, "linear.wqkv"), 3, (n_lin, args.conv_dim, args.dim)),
        ("linear.wz", stack(lin_z, "linear.wz"), 3, (n_lin, args.value_dim, args.dim)),
        ("linear.wo", stack(lin_o, "linear.wo"), 3, (n_lin, args.dim, args.value_dim)),
    ]
    if not args.is_shared_classifier:
        quant_tensors.append(("lm_head", lm_head, 2, (args.vocab_size, args.dim)))

    # 释放按层拼接前的临时数组，降低峰值内存
    del w1, w2, w3, wq, wk, wv, wo, lin_qkv, lin_z, lin_o, embed

    if quant_type == QUANT_TYPE_F32:
        print("Writing fp32 model parameters...")
        for label, arr, ndim, shape in quant_tensors:
            print(f"  F32 {label:24s} shape={tuple(arr.shape)}")
            write_f32(arr)
        param_count = sum(arr.size for _, arr, _, _ in quant_tensors)
    elif quant_type == QUANT_TYPE_Q80:
        gs = group_size
        while gs > 1 and any(shape[-1] % gs != 0 for _, _, _, shape in quant_tensors):
            gs //= 2
        if gs != group_size:
            print(f"BACKOFF: group size {group_size} -> {gs} to fit tensor dimensions")
        # 头里的 group_size 必须与实际写入的一致
        out_file.seek(HEADER_GROUP_SIZE_OFFSET)
        out_file.write(struct.pack("i", gs))
        out_file.seek(0, os.SEEK_END)

        print(f"Writing Q80 model parameters (group_size={gs})...")
        for label, arr, ndim, shape in quant_tensors:
            write_q80_group(out_file, arr, ndim, gs, label)
        param_count = sum(arr.size for _, arr, _, _ in quant_tensors)
    elif quant_type == QUANT_TYPE_Q4K:
        if group_size != 0:
            raise ValueError("q4k does not use --gs")
        print("Writing Q4K model parameters...")
        for label, arr, ndim, shape in quant_tensors:
            write_q4k_tensor(out_file, arr, ndim, shape, label)
        param_count = sum(arr.size for _, arr, _, _ in quant_tensors)
    else:
        raise ValueError(f"unknown quant type {quant_type}")

    print(f"Params = {param_count}")
    print(f"Total bin file length = {out_file.tell()}")
    out_file.close()
    print(f"wrote {out_path}")


#########################################################
#  入口点
#########################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", nargs="?", type=str, help="output filepath")
    parser.add_argument("-o", "--output", type=str, help="output filepath")
    parser.add_argument("--hf", type=str, required=True, help="HuggingFace model directory")
    parser.add_argument("--quant", type=str, choices=["f32", "q80", "q4k"], help="quantization type")
    parser.add_argument("--gs", default=None, type=int, help="Q80 group size; 0 means fp32; q4k ignores it")
    args = parser.parse_args()

    out_path = args.output or args.filepath
    if out_path is None:
        parser.error("output filepath is required (positional or -o)")

    hf_dir = args.hf
    if not os.path.isdir(hf_dir):
        parser.error(f"not a directory: {hf_dir}")

    # 兼容 export_qwen.py 的参数风格
    if args.quant is None:
        if args.gs is None:
            parser.error("either --quant or --gs is required")
        args.quant = "f32" if args.gs == 0 else ("q80" if args.gs > 0 else "f32")

    quant_type = {"f32": QUANT_TYPE_F32, "q80": QUANT_TYPE_Q80, "q4k": QUANT_TYPE_Q4K}[args.quant]
    group_size = 0
    if args.quant == "q80":
        group_size = 32 if args.gs is None else args.gs
        if group_size <= 0:
            parser.error("q80 requires a positive --gs")
    elif args.quant == "f32":
        group_size = 0

    model_args = load_model_args(hf_dir)
    print("Qwen3.5 model args:")
    for k, v in vars(model_args).items():
        if k != "layer_types":
            print(f"  {k} = {v}")

    st = SafeTensorSet(hf_dir)
    export_model(st, hf_dir, model_args, out_path, quant_type, group_size)


if __name__ == "__main__":
    main()
