#!/usr/bin/env python3
# ============================================================================
# gen_llm_model_b64.py —— 把 Nano 语言模型（F32/Nano 架构）切块转成 Base64 文本
#
# 用途：给「灵机 LLM 测试台」（infer/app/llm.scm + infer/app/lib/llm_pure.scm，纯
# Scheme 实现的 LLM 推理库）准备模型数据。上游 native LLM 库的接口
# (LLM.init modelFileBase64) 要求模型以 **base64 字符串** 传入；本 App 不用本地库，
# 也没有「读二进制」的本地库（(native Fs) 只整读文本、单文件上限 128KB），因此模型
# 以**分块 base64 文本**落在业务资源目录，运行时由 (Fs.read) 逐块读入、在 Scheme 里
# 流式解码（见 app/lib/llm_pure.scm 的 init_begin/init_feed/init_end）。
#
# 用法：
#   python3 gen_llm_model_b64.py <model.bin> <outdir> --name psycho \
#           --label "精神分析黑话模型 230k" [--chunk 98304]
#
# 产物（<outdir> 下）：
#   meta.scm   —— (define nchunk N) (define label "...") (define bytes N) ...（App 读取）
#   m00.b64 …  —— 分块 base64 文本（每块 ≤ chunk 字符，且字符数为 4 的倍数，
#                 这样每块都能独立解码，块间无需跨块拼接 —— 省一次整串拼接）
#
# 模型来源（本仓库不训练模型，两个 F32 nano 模型来自上游 Animac 发行版
# test/llm_model.scm 中的 base64 常量，见 doc/AGENTS.md 2026-11 条）：
#   psycho_230k.bin  https://huggingface.co/bd4sur/Nano-Psycho （精神分析黑话）
#   sort_6.bin       六元素升序序列生成模型（Nano 仓库 6 路归并排序任务）
# ============================================================================

import argparse
import base64
import hashlib
import os
import struct
import sys

MAGIC = None  # 头两个 uint32 为魔数（内容随版本变化，脚本只做非零校验）


def parse_header(raw: bytes) -> dict:
    """解析 256 字节模型头的有效字段（布局见 infer/vendor/Animac/am_native_LLM.c）。"""
    if len(raw) < 256:
        sys.exit("模型文件不足 256 字节")
    f = struct.unpack("<16I", raw[:64])
    names = ["magic0", "magic1", "major", "minor", "model_type", "config_length",
             "block_size", "vocab_size", "n_layer", "n_embd", "n_head", "n_kv_head",
             "n_hidden", "is_shared_classifier", "head_dim", "quant_type"]
    cfg = dict(zip(names, f))
    if cfg["model_type"] != 0:
        sys.exit(f"只支持 Nano 架构（model_type=0），实际 {cfg['model_type']}")
    if cfg["quant_type"] != 0:
        sys.exit(f"只支持 F32 权重（quant_type=0），实际 {cfg['quant_type']}")

    # 词表字段长度（用于报告与自检）
    tok_bytes = struct.unpack("<I", raw[256:260])[0]
    cfg["tokenizer_bytes"] = tok_bytes
    cfg["params_bytes"] = len(raw) - 256 - tok_bytes
    cfg["param_count"] = cfg["params_bytes"] // 4
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("outdir")
    ap.add_argument("--name", required=True, help="模型短名（App 内引用）")
    ap.add_argument("--label", default="", help="菜单/标题里显示的中文名")
    ap.add_argument("--chunk", type=int, default=96 * 1024, help="每块 base64 字符数（4 的倍数）")
    args = ap.parse_args()

    chunk = (args.chunk // 4) * 4
    if chunk <= 0 or chunk > 120 * 1024:
        sys.exit("--chunk 须为 4 的倍数且 ≤ 120KB（(native Fs) 单文件读入上限 128KB）")

    raw = open(args.model, "rb").read()
    cfg = parse_header(raw)
    b64 = base64.b64encode(raw).decode("ascii")

    # 自检：分块解码拼回必须与原文逐字节相同（App 侧按块独立解码，块间不共享字符）
    parts = [b64[i:i + chunk] for i in range(0, len(b64), chunk)]
    rebuilt = base64.b64decode("".join(parts))
    if rebuilt != raw:
        sys.exit("自检失败：分块 base64 往返与原文不一致")

    os.makedirs(args.outdir, exist_ok=True)
    for fn in os.listdir(args.outdir):
        if fn.endswith(".b64") or fn == "meta.scm":
            os.unlink(os.path.join(args.outdir, fn))
    for i, part in enumerate(parts):
        with open(os.path.join(args.outdir, "m%02d.b64" % i), "w") as f:
            f.write(part)

    sha = hashlib.sha256(raw).hexdigest()[:16]
    with open(os.path.join(args.outdir, "meta.scm"), "w") as f:
        f.write(";; 由 infer/tools/gen_llm_model_b64.py 生成，请勿手改\n")
        f.write(";; 源：%s（%d 字节，sha256:%s…）\n" % (os.path.basename(args.model), len(raw), sha))
        f.write('(define name "%s")\n' % args.name)
        f.write('(define label "%s")\n' % args.label)
        f.write("(define nchunk %d)\n" % len(parts))
        f.write("(define nbytes %d)\n" % len(raw))
        f.write("(define b64len %d)\n" % len(b64))
        f.write("(define chunk_chars %d)\n" % chunk)
        f.write("(define block_size %d)\n" % cfg["block_size"])
        f.write("(define vocab_size %d)\n" % cfg["vocab_size"])
        f.write("(define n_layer %d)\n" % cfg["n_layer"])
        f.write("(define n_embd %d)\n" % cfg["n_embd"])
        f.write("(define n_head %d)\n" % cfg["n_head"])
        f.write("(define n_kv_head %d)\n" % cfg["n_kv_head"])
        f.write("(define n_hidden %d)\n" % cfg["n_hidden"])
        f.write("(define head_dim %d)\n" % cfg["head_dim"])

    print("模型 %s：%d 字节 → base64 %d 字符 → %d 块（每块 ≤%d）"
          % (args.name, len(raw), len(b64), len(parts), chunk))
    print("  配置：block_size=%d vocab=%d layer=%d embd=%d head=%d kv_head=%d hidden=%d head_dim=%d"
          % (cfg["block_size"], cfg["vocab_size"], cfg["n_layer"], cfg["n_embd"],
             cfg["n_head"], cfg["n_kv_head"], cfg["n_hidden"], cfg["head_dim"]))
    print("  词表字段 %d 字节，参数 %d 个 float，输出目录 %s"
          % (cfg["tokenizer_bytes"], cfg["param_count"], args.outdir))


if __name__ == "__main__":
    main()