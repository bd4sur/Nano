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

# 写入词表
def serialize_tokenizer(out_file, tokenizer_config):
    """
    词表序列化结构如下(BNF)：
           tokenizer_config ::= tokenizer_field_bytes vocab_size tokens
      tokenizer_field_bytes ::= uint32(I)(4B) 其值为整个tokenizer_config的字节长度
                 vocab_size ::= uint32(I)(4B) 其值为tokens中token数目
                     tokens ::= tokens token | token
                      token ::= token_header token_id unicode_chars
               token_header ::= token_length is_special reserved_0 reserved_1
               token_length ::= uint8(B)(1B) 其值为unicode_chars中ucchar数目
                 is_special ::= uint8(B)(1B) 1-True 0-False
                 reserved_0 ::= uint8(B)(1B) 保留不用
                 reserved_1 ::= uint8(B)(1B) 保留不用
                   token_id ::= uint32(I)(4B)
              unicode_chars ::= unicode_chars ucchar | ucchar
                     ucchar ::= uint32(I)(4B)
    """
    vocab = tokenizer_config["itos"]
    vocab_size = tokenizer_config["vocab_size"]
    special_tokens = tokenizer_config["special_tokens"]

    tokenizer_field_bytes = 4 + 4 # 对应tokenizer_field_bytes（本字段）和vocab_size字段
    for i,t in enumerate(vocab):
        tokenizer_field_bytes += (len(t) + 2) * 4  # 计算方法见上文注释

    print(f"  Tokenizer field bytes = {tokenizer_field_bytes}")

    out_file.write(struct.pack('I', tokenizer_field_bytes))  # 模型文件中词表部分的字节数（不含本字段的4个字节）
    out_file.write(struct.pack('I', vocab_size))             # 词表长度
    for i,t in enumerate(vocab):
        token_length = len(t)
        is_special = 1 if t in special_tokens else 0
        # NOTE Little endian 小端序！如果按照uint32解析，顺序是 MSB(reserved_1 reserved_0 is_special token_length)LSB
        out_file.write(struct.pack('B', token_length))
        out_file.write(struct.pack('B', is_special))
        out_file.write(struct.pack('B', 255)) # 预留
        out_file.write(struct.pack('B', 255)) # 预留

        out_file.write(struct.pack('I', i))

        for chr in t:
            out_file.write(struct.pack('I', ord(chr)))





def export_lora(lora_dict, lora_config, basemodel_config, filepath):
    major_version = 2024
    minor_version = 10

    out_file = open(filepath, 'wb')

    #########################################################
    # 写入文件头（固定长度256B）

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

    #########################################################
    # 写入LoRA模型参数

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

    # 【NOTE 不需要】写入模型参数数（本字段8个字节）
    # out_file.write(struct.pack('Q', param_count)) # unsigned long long - uint64_t

    for w in weights:
        serialize_fp32(out_file, w)

    print(f"Params = {param_count}")
    print(f"Total bin file length = {out_file.tell()}")

    #########################################################
    # 写入并关闭文件

    out_file.close()
    print(f"wrote {filepath}")



def export_model(model, tokenizer_config, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """

    out_file = open(filepath, 'wb')

    #########################################################
    # 写入文件头（固定长度256B）

    print("Writing header...")

    major_version = 2025
    minor_version = 12

    # 1) write magic, which will be two uint32 of "BD4SURLM" in ASCII
    out_file.write(struct.pack('I', 0x42443453))
    out_file.write(struct.pack('I', 0x55524c4d))
    # --> 8 bytes

    # 2) write version, which will be int
    out_file.write(struct.pack('i', major_version))
    out_file.write(struct.pack('i', minor_version))
    # --> 16 bytes

    # 3) write file type TODO to be defined
    out_file.write(struct.pack('i', 0))  # Model type: BD4SUR's Nano model
    out_file.write(struct.pack('i', 32)) # Config Length: 32 bytes
    # --> 24 bytes

    # 4) write the model config, which will be 8 ints (32 bytes)
    cfg = model.config
    is_shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    header = struct.pack(
        "iiiiiiiii",
        cfg.block_size,
        cfg.vocab_size,
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_kv_head if cfg.n_kv_head is not None else cfg.n_head,
        cfg.n_hidden if cfg.n_hidden is not None else model.layers[0].feed_forward.w1.weight.shape[0],
        int(is_shared_classifier),
        cfg.n_embd // cfg.n_head # head_dim Nano不用这个参数
    )
    out_file.write(header)
    # --> 60 bytes

    # 5) write some other flags (TODO)
    out_file.write(struct.pack('i', 0))  # 量化类型：QUANT_TYPE_F32=0(无量化)，见`infer/tensor.h`

    # 6) pad rest with zeros; 'tell' returns current pos
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)


    #########################################################
    # 写入词表

    print("Writing tokenizer...")
    serialize_tokenizer(out_file, tokenizer_config)


    #########################################################
    # 写入模型参数

    print("Writing model parameters...")

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

        # Optional
        model.freqs_cos,
        model.freqs_sin,
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

    #########################################################
    # 写入并关闭文件

    out_file.close()
    print(f"wrote {filepath}")



def export_quantized(model, tokenizer_config, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """

    cfg = model.config
    out_file = open(filepath, 'wb')

    #########################################################
    # 写入文件头（固定长度256B）

    print("Writing header...")

    major_version = 2025
    minor_version = 12

    # 1) write magic, which will be two uint32 of "BD4SURLM" in ASCII
    out_file.write(struct.pack('I', 0x42443453))
    out_file.write(struct.pack('I', 0x55524c4d))
    # --> 8 bytes

    # 2) write version, which will be int
    out_file.write(struct.pack('i', major_version))
    out_file.write(struct.pack('i', minor_version))
    # --> 16 bytes

    # 3) write file type TODO to be defined
    out_file.write(struct.pack('i', 0))  # Model type: BD4SUR's Nano model
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
        int(is_shared_classifier),
        cfg.n_embd // cfg.n_head # head_dim Nano不用这个参数
    )
    out_file.write(header)
    # --> 60 bytes

    # 5) write some other flags
    out_file.write(struct.pack('i', 10))         # 量化类型：QUANT_TYPE_Q80=10，见`infer/tensor.h`
    out_file.write(struct.pack('i', group_size)) # 量化参数(分组长度)

    # 6) pad rest with zeros; 'tell' returns current pos
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)


    #########################################################
    # 写入词表

    print("Writing tokenizer...")
    serialize_tokenizer(out_file, tokenizer_config)


    #########################################################
    # 量化并写入模型参数

    print("Writing Q80 quantized model parameters...")

    while cfg.n_embd % group_size != 0:
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


# -----------------------------------------------------------------------------
# Load / import functions

def load_checkpoint(checkpoint):
    # load the provided model checkpoint
    checkpoint_dict = torch.load(checkpoint, weights_only=False, map_location='cpu')
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
    checkpoint_dict = torch.load(lora_path, weights_only=False, map_location='cpu')
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
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=1, type=int, help="the version to export with")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--quant", type=str, help="model checkpoint, .pt file for exporting quantized model bin file")
    group.add_argument("--lora", type=str, help="lora module, .pt file")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.lora:
        lora_dict, lora_config, basemodel_config = load_lora(args.lora)
        if lora_dict is None:
            parser.error("Can't load input LoRA module!")
        export_lora(lora_dict, lora_config, basemodel_config, args.filepath)

    if args.quant:
        model, tokenizer_config = load_checkpoint(args.quant)
        if model is None or tokenizer_config is None:
            parser.error("Can't load input model!")
        export_quantized(model, tokenizer_config, args.filepath, group_size=128)

    if args.checkpoint:
        model, tokenizer_config = load_checkpoint(args.checkpoint)
        if model is None or tokenizer_config is None:
            parser.error("Can't load input model!")
        export_model(model, tokenizer_config, args.filepath)
