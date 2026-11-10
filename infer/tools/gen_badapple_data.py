#!/usr/bin/env python3
# ============================================================================
# gen_badapple_data.py —— 把 infer/badapple.h 的位图帧数据转成「灵机 App 可直接读」的
#                         定长记录文本文件 + 元数据模块
#
# 为什么需要它：
#   Bad Apple 的帧数据（2193 帧 × 64×64 黑白）现在是 C 数组（infer/badapple.h，C 版
#   STATE_BADAPPLE 直接索引它）。灵机 App（Scheme）看不到 C 的静态数组，而 (native Fs)
#   只提供**文本**读取（无二进制、单文件整读上限 128KB，分段读 (Fs.read_range) 上限
#   64KB/次），因此数据必须落成「资源目录里的文本 + Scheme 侧解码」。
#
# 格式（定长记录，便于按帧随机访问；每帧一条记录，字节数固定 = meta.scm 的 rec_bytes）：
#
#   记录 = 2 字符段数头 + 段数 × 3 字符段体 + 『!』填充（填充部分 App 不读）
#     c0 = A[count // 94],  c1 = A[count % 94]                —— 本帧的白色像素段数
#     每段 3 字符：A[y]、A[x]、A[len-1]                        —— 行号、段起始列、段长-1
#   其中 A[v] = chr(33 + v)（v ∈ 0..93），即 ASCII 33..126 的可打印字符。
#
#   为什么是「白色像素段」而不是直接抄 128 个 uint32：32 位字在 wasm32 宿主上装不进
#   解释器的立即数（无符号整数逻辑位宽仅 27 位、浮点尾数仅约 19 位，见
#   infer/am_native_Time.h 的说明），拆成字节再按位取也很贵；而 64×64 画面里平均每帧
#   只有 ~94 段白色像素（最大 218 段），App 侧每帧只需 ~94 次 (Gfx.fill_rect) 就能画完，
#   解释器开销极小（每帧几千条 VM 指令，10fps 下远低于宿主时间片预算 312500）。
#
#   定长记录的用处：偏移 = 帧号 × rec_bytes，(Fs.read_range) 一次只读一帧（≤672 字节），
#   不需要在 Scheme 里加载/索引整份数据，也不必解析行分隔符。
#
# 用法：
#   python3 gen_badapple_data.py                 # 生成到 infer/app/lib/badapple/
#   python3 gen_badapple_data.py --deploy DIR    # 另外复制到业务资源目录（如 ~/linglong）
#
# 产物：
#   infer/app/lib/badapple/frames.txt    —— 定长记录文本（2193 帧）
#   infer/app/lib/badapple/meta.scm      —— (define n_frames ...) 等常量（单一事实来源：
#                                           App 侧 import 它，不手抄常量）
#
# 自检：脚本末尾把生成的数据**逐帧解码回位图**，与 badapple.h 的原位图逐位比对，
#       不一致即报错退出（数据是二进制的，肉眼看不出来，只能机器验）。
# ============================================================================

import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INFER = os.path.dirname(HERE)
SRC_H = os.path.join(INFER, "badapple.h")
OUT_DIR = os.path.join(INFER, "app", "lib", "badapple")
ARRAY_NAME = "bad_apple_10fps_64x64"
FRAME_W = 64
FRAME_H = 64
WORDS_PER_FRAME = FRAME_H * 2      # 每行 2 个 uint32（左 32 像素 + 右 32 像素）
ALPHA = 94                         # A[v] = chr(33+v)，v ∈ 0..93


def parse_badapple_h(path):
    """从 badapple.h 里抠出 uint32 数组的全部元素（帧号 → 128 字）。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    anchor = text.index("const uint32_t %s" % ARRAY_NAME)
    body = text[anchor:]
    vals = [int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]{8})", body)]
    if len(vals) % WORDS_PER_FRAME != 0:
        sys.exit("帧数据长度不是 128 的整数倍：%d" % len(vals))
    frames = []
    for f in range(len(vals) // WORDS_PER_FRAME):
        base = f * WORDS_PER_FRAME
        rows = []
        for y in range(FRAME_H):
            lo = vals[base + y * 2]
            hi = vals[base + y * 2 + 1]
            rows.append([(lo >> (31 - c)) & 1 for c in range(32)]
                        + [(hi >> (31 - c)) & 1 for c in range(32)])
        frames.append(rows)
    return frames


def frame_runs(rows):
    """一帧 → 白色像素段列表 [(y, x, len), ...]（按 y、x 升序）。"""
    runs = []
    for y, row in enumerate(rows):
        x = 0
        while x < FRAME_W:
            if row[x]:
                s = x
                while x < FRAME_W and row[x]:
                    x += 1
                runs.append((y, s, x - s))
            else:
                x += 1
    return runs


def encode_record(runs, rec_bytes):
    if len(runs) >= ALPHA * ALPHA:
        sys.exit("段数过多：%d" % len(runs))
    out = [chr(33 + len(runs) // ALPHA), chr(33 + len(runs) % ALPHA)]
    for (y, x, ln) in runs:
        if not (0 <= y < FRAME_H and 0 <= x < FRAME_W and 1 <= ln <= FRAME_W):
            sys.exit("非法段：%r" % ((y, x, ln),))
        out.append(chr(33 + y))
        out.append(chr(33 + x))
        out.append(chr(33 + ln - 1))
    pad = rec_bytes - len(out)
    if pad < 0:
        sys.exit("记录超长：%d > %d" % (len(out), rec_bytes))
    out.extend(["!"] * pad)
    return "".join(out)


def decode_record(rec):
    """记录 → [(y, x, len), ...]（App 侧解码逻辑的 Python 对照实现，供自检）。"""
    n = (ord(rec[0]) - 33) * ALPHA + (ord(rec[1]) - 33)
    runs = []
    for i in range(n):
        p = 2 + i * 3
        runs.append((ord(rec[p]) - 33, ord(rec[p + 1]) - 33, ord(rec[p + 2]) - 32))
    return runs


def main():
    frames = parse_badapple_h(SRC_H)
    all_runs = [frame_runs(rows) for rows in frames]
    max_runs = max(len(r) for r in all_runs)
    rec_bytes = 2 + 3 * max_runs
    rec_bytes = (rec_bytes + 7) // 8 * 8        # 对齐到 8 字节，便于阅读偏移
    records = [encode_record(r, rec_bytes) for r in all_runs]
    blob = "".join(records)
    assert len(blob) == len(frames) * rec_bytes

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_path = os.path.join(OUT_DIR, "frames.txt")
    meta_path = os.path.join(OUT_DIR, "meta.scm")

    with open(frames_path, "w", encoding="ascii", newline="") as f:
        f.write(blob)

    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(";; 由 infer/tools/gen_badapple_data.py 从 infer/badapple.h 自动生成，请勿手工编辑。\n")
        f.write(";; 帧数据：app/lib/badapple/frames.txt（定长记录，偏移 = 帧号 × rec_bytes）。\n")
        f.write(";; 记录布局常量一并导出，App 侧不手抄数字（改格式只改本脚本）。\n")
        f.write("(define n_frames %d)\n" % len(frames))
        f.write("(define rec_bytes %d)\n" % rec_bytes)
        f.write("(define frame_w %d)\n" % FRAME_W)
        f.write("(define frame_h %d)\n" % FRAME_H)
        f.write("(define fps 10)\n")
        f.write("(define slot_ms 100)\n")
        f.write(";; 记录字符 c → 数值：(- c base_char)\n")
        f.write("(define base_char %d)\n" % 33)
        f.write(";; 段数头：c0*base_radix + c1（两个字符）\n")
        f.write("(define base_radix %d)\n" % ALPHA)
        f.write("(define head_chars %d)\n" % 2)
        f.write(";; 每段 3 字符：(行号, 段起始列, 段长-1)\n")
        f.write("(define chars_per_run %d)\n" % 3)

    # ---- 自检：逐帧解码回位图，与原位图逐位比对 ----
    bad = 0
    for idx, rows in enumerate(frames):
        rec = records[idx]
        got = [[0] * FRAME_W for _ in range(FRAME_H)]
        for (y, x, ln) in decode_record(rec):
            for k in range(ln):
                got[y][x + k] = 1
        if got != rows:
            bad += 1
            if bad <= 3:
                print("自检失败：帧 %d 解码不一致" % idx, file=sys.stderr)
    if bad:
        sys.exit("自检失败：%d/%d 帧不一致" % (bad, len(frames)))

    # 反向检查：填充区没有被误读（段数头必须与真实段数一致）
    for idx, runs in enumerate(all_runs):
        assert decode_record(records[idx]) == runs, idx

    print("frames=%d rec_bytes=%d total=%d bytes (%.2f MB) max_runs=%d mean_runs=%.1f"
          % (len(frames), rec_bytes, len(blob), len(blob) / 1048576.0,
             max_runs, sum(len(r) for r in all_runs) / float(len(frames))))
    print("wrote %s" % frames_path)
    print("wrote %s" % meta_path)

    # ---- 可选：复制进业务资源目录（浏览器/桌面宿主按 PLATFORM_ROOT_DIR 读）----
    if "--deploy" in sys.argv:
        dst = sys.argv[sys.argv.index("--deploy") + 1]
        dst_dir = os.path.join(dst, "app", "lib", "badapple")
        os.makedirs(dst_dir, exist_ok=True)
        for name in ("frames.txt", "meta.scm"):
            shutil.copyfile(os.path.join(OUT_DIR, name), os.path.join(dst_dir, name))
        print("deployed to %s" % dst_dir)


if __name__ == "__main__":
    main()
