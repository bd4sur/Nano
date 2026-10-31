#!/usr/bin/env python3
# ============================================================================
# gen_mlp_data.py —— 从参考页面生成「多层感知机」App 的数据模块
#
# 参考页面：/home/bd4sur/bd4sur.github.io/html/mlp.html
#   （bd4sur.com 的 MLP 训练可视化：2 维输入、5 类、256 个样本）
# 该页面的数据以 JS 字面量 x_array（256 个 [x0, x1]）与 y_array（256 个
# one-hot 5 维）直接内嵌，本脚本把它们提取出来、转成灵机（Animac）模块
# app/lib/mlp_data.scm，供 app/mlp.scm (import D "app/lib/mlp_data.scm") 使用。
#
# 输出格式：
#   (define mlp_n 256)                       ; 样本数
#   (define mlp_ncls 5)                      ; 类别数
#   (define mlp_x '(...))                    ; 512 个数的平铺列表（样本 i 的输入为 x[2i], x[2i+1]）
#   (define mlp_y '(...))                    ; 256 个类别下标（0..4）
#
# 精度：wasm32 上灵机浮点为单精度且尾数被截 5 位（相对精度 ~2e-6，
#   ），故按 %.6g 导出，写更多位没有意义。
#
# 用法：python3 infer/tools/gen_mlp_data.py [参考页面路径] [输出路径]
# ============================================================================
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # infer/
SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/bd4sur/bd4sur.github.io/html/mlp.html"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "app", "lib", "mlp_data.scm")

html = open(SRC, encoding="utf-8").read()
x_src = re.search(r"const x_array = (\[\[.*?\]\]);", html, re.S)
y_src = re.search(r"const y_array = (\[\[.*?\]\]);", html, re.S)
if not x_src or not y_src:
    sys.exit("参考页面里找不到 x_array / y_array 字面量：%s" % SRC)
xs = json.loads(x_src.group(1))
ys = json.loads(y_src.group(1))
if len(xs) != len(ys):
    sys.exit("样本数与标签数不一致：%d vs %d" % (len(xs), len(ys)))
n = len(xs)
ncls = len(ys[0])

flat = []
for p in xs:
    flat.append(p[0])
    flat.append(p[1])
labels = [row.index(1) for row in ys]

lo = min(min(p) for p in xs)
hi = max(max(p) for p in xs)


def emit_numbers(vals, per_line=8):
    out, line = [], []
    for v in vals:
        line.append("%.6g" % v)
        if len(line) >= per_line:
            out.append("  " + " ".join(line))
            line = []
    if line:
        out.append("  " + " ".join(line))
    return "\n".join(out)


def wrap_ints(vals, per_line=16):
    out, line = [], []
    for v in vals:
        line.append(str(v))
        if len(line) >= per_line:
            out.append("  " + " ".join(line))
            line = []
    if line:
        out.append("  " + " ".join(line))
    return "\n".join(out)


with open(OUT, "w", encoding="utf-8") as f:
    f.write(";; mlp_data.scm —— 多层感知机 App 的样本数据（**由脚本生成，勿手改**）\n")
    f.write(";;\n")
    f.write(";; 生成脚本：infer/tools/gen_mlp_data.py\n")
    f.write(";; 数据来源：%s\n" % SRC)
    f.write(";;           （bd4sur.com 的多层感知机可视化页面，2 维输入 / %d 类 / %d 个样本）\n" % (ncls, n))
    f.write(";; 取值范围：[%.3f, %.3f]（x 与 y 同量级，方图直接按 [-R, R] 映射即可）\n" % (lo, hi))
    f.write(";; 数值精度：6 位有效数字（wasm32 上灵机浮点尾数仅约 19 位，更多位数无意义）\n")
    f.write("\n")
    f.write("(define mlp_n %d)      ; 样本数\n" % n)
    f.write("(define mlp_ncls %d)   ; 类别数\n" % ncls)
    f.write("\n")
    f.write(";; 输入：平铺列表，样本 i 的两个坐标为 (get_item mlp_x (* i 2)) 与 (+ 1 ...)\n")
    f.write("(define mlp_x '(\n")
    f.write(emit_numbers(flat))
    f.write("))\n\n")
    f.write(";; 标签：样本 i 的类别下标（0..%d）\n" % (ncls - 1))
    f.write("(define mlp_y '(\n")
    f.write(wrap_ints(labels))
    f.write("))\n")

print("wrote %s (%d samples, %d classes, x in [%.3f, %.3f])" % (OUT, n, ncls, lo, hi))