# 
# Nano Language Model
#
#   BD4SUR 2024-10
#
#   Forked from:
#     - https://github.com/karpathy/llama2.c
#

import struct
import base64
import json
import argparse

import numpy as np
import torch

from model import GPT

# -----------------------------------------------------------------------------
# common utilities

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
    ori_shape = w.shape
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

# -----------------------------------------------------------------------------
# new version

def lora_export(lora_dict, lora_config, basemodel_config, filepath):
    major_version = 2024
    minor_version = 10

    out_file = open(filepath, 'wb')

    # first write out the header. the header will be 256 bytes

    # 1) write magic, which will be two uint32 of "BD4SURLM" in ASCII
    out_file.write(struct.pack('I', 0x42443453))
    out_file.write(struct.pack('I', 0x55524c4d))
    # --> 8 bytes

    # 2) write version, which will be int
    out_file.write(struct.pack('i', major_version))
    out_file.write(struct.pack('i', minor_version))
    # --> 16 bytes

    # 3) write file type TODO  to be defined
    out_file.write(struct.pack('i', 10))  # Model type: LoRA module
    out_file.write(struct.pack('i', 32))  # Config Length: 32 bytes
    # --> 24 bytes

    # 4) write the LoRA config, which will be 8 ints (32 bytes)
    out_file.write(struct.pack('i', lora_config["lora_rank"]))
    out_file.write(struct.pack('i', lora_config["lora_alpha"]))
    out_file.write(struct.pack('i', basemodel_config.n_layer))   # 用于校验
    out_file.write(struct.pack('i', basemodel_config.n_embd))   # 用于校验
    out_file.write(struct.pack('i', basemodel_config.n_head))    # 用于校验
    out_file.write(struct.pack('i', basemodel_config.n_kv_head)) # 用于校验
    out_file.write(struct.pack('i', basemodel_config.n_hidden))  # 用于校验
    out_file.write(struct.pack('i', 0)) # 预留：用于控制LoRA用到哪些层
    # --> 56 bytes

    # 5) write some other flags (TODO)

    # 6) pad rest with zeros; 'tell' returns current pos
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # now let's write out all the LoRA parameters
    weights = []
    wq_lora_a, wq_lora_b = {}, {}
    wk_lora_a, wk_lora_b = {}, {}
    wv_lora_a, wv_lora_b = {}, {}
    wo_lora_a, wo_lora_b = {}, {}

    for k, v in lora_dict.items():
        if "wq.lora_a" in k:
            wq_lora_a[k] = v
        elif "wq.lora_b" in k:
            wq_lora_b[k] = v
        elif "wk.lora_a" in k:
            wk_lora_a[k] = v
        elif "wk.lora_b" in k:
            wk_lora_b[k] = v
        elif "wv.lora_a" in k:
            wv_lora_a[k] = v
        elif "wv.lora_b" in k:
            wv_lora_b[k] = v
        elif "wo.lora_a" in k:
            wo_lora_a[k] = v
        elif "wo.lora_b" in k:
            wo_lora_b[k] = v

    keycmp = lambda k: int(k.split(".")[1]) # layer index

    for k in sorted(wq_lora_a.keys(), key=keycmp):
        weights.append(wq_lora_a[k])
    for k in sorted(wq_lora_b.keys(), key=keycmp):
        weights.append(wq_lora_b[k])
    for k in sorted(wk_lora_a.keys(), key=keycmp):
        weights.append(wk_lora_a[k])
    for k in sorted(wk_lora_b.keys(), key=keycmp):
        weights.append(wk_lora_b[k])
    for k in sorted(wv_lora_a.keys(), key=keycmp):
        weights.append(wv_lora_a[k])
    for k in sorted(wv_lora_b.keys(), key=keycmp):
        weights.append(wv_lora_b[k])
    for k in sorted(wo_lora_a.keys(), key=keycmp):
        weights.append(wo_lora_a[k])
    for k in sorted(wo_lora_b.keys(), key=keycmp):
        weights.append(wo_lora_b[k])

    param_count = 0
    for w in weights:
        param_count += w.detach().cpu().view(-1).numel() * 4
        serialize_fp32(out_file, w)

    print(f"Params = {param_count}")
    print(f"Total bin file length = {out_file.tell()}")

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")



def version1_export(model, tokenizer_config, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    major_version = 2024
    minor_version = 10

    out_file = open(filepath, 'wb')

    # first write out the header. the header will be 256 bytes

    # 1) write magic, which will be two uint32 of "BD4SURLM" in ASCII
    out_file.write(struct.pack('I', 0x42443453))
    out_file.write(struct.pack('I', 0x55524c4d))
    # --> 8 bytes

    # 2) write version, which will be int
    out_file.write(struct.pack('i', major_version))
    out_file.write(struct.pack('i', minor_version))
    # --> 16 bytes

    # 3) write file type TODO  to be defined
    out_file.write(struct.pack('i', 0))  # Model type: Base model
    out_file.write(struct.pack('i', 32)) # Config Length: 32 bytes
    # --> 24 bytes

    # 4) write the model config, which will be 8 ints (32 bytes)
    cfg = model.config
    is_shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    header = struct.pack(
        "iiiiiiii",
        cfg.block_size,
        cfg.vocab_size,
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_kv_head if cfg.n_kv_head is not None else cfg.n_head,
        cfg.n_hidden if cfg.n_hidden is not None else model.layers[0].feed_forward.w1.weight.shape[0],
        int(is_shared_classifier)
    )
    out_file.write(header)
    # --> 56 bytes

    # 5) write some other flags (TODO)

    # 6) pad rest with zeros; 'tell' returns current pos
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # now let's write out all the model parameters
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
        model.norm.weight,
        model.freqs_cos,
        model.freqs_sin,
    ]
    if not is_shared_classifier:
        weights.append(model.output.weight)
    param_count = 0
    for w in weights:
        param_count += w.detach().cpu().view(-1).numel() * 4
        serialize_fp32(out_file, w)

    # write tokenizer config dict as base64 string
    tk_cfg_json_str = json.dumps(tokenizer_config, ensure_ascii=True)
    b64 = base64.b64encode(bytes(tk_cfg_json_str, encoding="utf-8"))
    out_file.write(struct.pack('I', len(b64)))
    print(f"Tokenizer config length = {len(b64)}")
    serialize_base64(out_file, b64)

    print(f"Params = {param_count}")
    print(f"Total bin file length = {out_file.tell()}")

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

def version2_export(model, tokenizer_config, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 2

    # let's first do some validation for this export type
    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        assert w.numel() % group_size == 0, f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

    # write
    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    out_file.write(struct.pack('B', int(shared_classifier)))
    out_file.write(struct.pack('i', group_size)) # group size used for quantization
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)
    # now that the header is done, let's write out the model

    # first let's write out all the params that we are keeping in fp32: the norms
    for layer in model.layers: # attention norms
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers: # MLP norms
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight) # final pre-classifier norm

    # now let's write out all the params that we are quantizing to Q8_0
    # note we skip classifier weights, which are shared with the embedding
    ew = []
    for i, w in enumerate(weights):
        # quantize this weight
        q, s, err = quantize_q80(w, group_size)
        # save the int8 weights to file
        serialize_int8(out_file, q) # save the tensor in int8
        serialize_fp32(out_file, s) # save scale factors
        # logging
        ew.append((err, w.shape))
        print(f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


# -----------------------------------------------------------------------------
# Load / import functions

def load_checkpoint(checkpoint):
    # load the provided model checkpoint
    checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    model_config = checkpoint_dict['model_config']
    model = GPT(model_config)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    tokenizer_config = checkpoint_dict['tokenizer_config']
    return model, tokenizer_config


def load_lora(lora_path):
    print(f"LoRA module file path: {lora_path}")
    checkpoint_dict = torch.load(lora_path, map_location='cpu')
    if checkpoint_dict["is_lora"]:
        train_config = checkpoint_dict["train_config"]
        model_config = checkpoint_dict["model_config"]
        lora_config = {
            "lora_rank": train_config.lora_rank,
            "lora_alpha": train_config.lora_alpha,
        }
        return checkpoint_dict["lora"], lora_config, model_config
    else:
        return False


# -----------------------------------------------------------------------------
# API entrypoint

def model_export(model, tokenizer_config, filepath, version, dtype=torch.float32):
    """
    v1: float32 export
    v2: int8 quantized Q8_0 export, similar to llama.cpp, in groups
    """
    if version == 1:
        version1_export(model, tokenizer_config, filepath)
    elif version == 2:
        version2_export(model, tokenizer_config, filepath)
    else:
        raise ValueError(f"unknown version {version}")


# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=1, type=int, help="the version to export with")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--lora", type=str, help="lora module, .pt file")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.lora:
        lora_dict, lora_config, basemodel_config = load_lora(args.lora)
        if lora_dict is None:
            parser.error("Can't load input LoRA module!")
        lora_export(lora_dict, lora_config, basemodel_config, args.filepath)

    if args.checkpoint:
        model, tokenizer_config = load_checkpoint(args.checkpoint)
        if model is None or tokenizer_config is None:
            parser.error("Can't load input model!")
        model_export(model, tokenizer_config, args.filepath, args.version, args.dtype)
