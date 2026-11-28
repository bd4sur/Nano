#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_bird_icon.py —— 生成灵机 App「百鸟园」的主菜单图标（36x36 PNG，RGBA）

主菜单对灵机 App 的图标约定是 `icon/<id>.png`（存在则用，否则回退 icon/default.png），
见 infer/ui_main_menu.c 的 s_scheme_app_icon_override 与 ui_main_menu.c 的注释。
本脚本按仓库里现有图标（36x36、圆角方块、单一白色字形）的同一风格程序化生成一张，
避免手工画图与二进制文件入库：

    cd infer && python3 tools/gen_bird_icon.py [输出路径]
    # 默认写 /home/bd4sur/linglong/icon/bird.png（业务资源目录，不入 Git）

依赖：纯标准库（zlib 写 PNG）。4x 超采样做抗锯齿。
"""

import math
import os
import struct
import sys
import zlib

SIZE = 36
SS = 4                     # 超采样倍数
OUT_DEFAULT = "/home/bd4sur/linglong/icon/bird.png"

# 配色：青绿→深青的圆角方块（与「音乐盒」等图标的明度/饱和度同一档）
BG_TOP = (58, 200, 170)
BG_BOT = (20, 122, 138)
FG = (255, 255, 255)


def rounded_rect_cov(x, y, x0, y0, x1, y1, r):
    """点 (x,y) 落在圆角矩形内的判定（用于超采样覆盖率）。"""
    if x < x0 or x > x1 or y < y0 or y > y1:
        return False
    cx = min(max(x, x0 + r), x1 - r)
    cy = min(max(y, y0 + r), y1 - r)
    dx, dy = x - cx, y - cy
    return dx * dx + dy * dy <= r * r


def ellipse_cov(x, y, cx, cy, rx, ry, rot=0.0):
    dx, dy = x - cx, y - cy
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        dx, dy = dx * c + dy * s, -dx * s + dy * c
    return (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0


def tri_cov(x, y, p0, p1, p2):
    def cross(ax, ay, bx, by, px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    d1 = cross(p0[0], p0[1], p1[0], p1[1], x, y)
    d2 = cross(p1[0], p1[1], p2[0], p2[1], x, y)
    d3 = cross(p2[0], p2[1], p0[0], p0[1], x, y)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def bird_at(x, y):
    """白色小鸟字形（朝向右侧鸣叫）：身体 + 头 + 喙 + 尾 + 翅（挖空）。"""
    body = ellipse_cov(x, y, 16.0, 20.5, 8.6, 6.4, rot=-0.18)
    head = ellipse_cov(x, y, 24.0, 14.6, 5.0, 5.0)
    beak = tri_cov(x, y, (28.4, 13.6), (34.0, 15.2), (28.4, 17.2))
    tail = tri_cov(x, y, (8.6, 17.4), (2.0, 12.4), (7.0, 22.6))
    if not (body or head or beak or tail):
        return False
    # 翅膀：在身体上挖一条月牙（负空间，图标更"有鸟样"）
    wing = ellipse_cov(x, y, 14.6, 19.4, 3.4, 1.7, rot=0.42)
    if wing and body and not head:
        return False
    return True


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT
    rows = []
    inv = 1.0 / (SS * SS)
    for py in range(SIZE):
        row = bytearray()
        for px in range(SIZE):
            bg_hit = 0
            fg_hit = 0
            for sy in range(SS):
                for sx in range(SS):
                    x = px + (sx + 0.5) / SS
                    y = py + (sy + 0.5) / SS
                    if not rounded_rect_cov(x, y, 0.6, 0.6, SIZE - 0.6, SIZE - 0.6, 9.0):
                        continue
                    bg_hit += 1
                    if bird_at(x, y):
                        fg_hit += 1
            if bg_hit == 0:
                row += bytes((0, 0, 0, 0))
                continue
            t = py / (SIZE - 1.0)
            base = [int(BG_TOP[i] + (BG_BOT[i] - BG_TOP[i]) * t) for i in range(3)]
            a = int(255 * bg_hit * inv)
            k = fg_hit / float(bg_hit)
            col = [int(base[i] + (FG[i] - base[i]) * k) for i in range(3)]
            row += bytes((col[0], col[1], col[2], a))
        rows.append(bytes(row))

    raw = b"".join(b"\x00" + r for r in rows)

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(png)
    print("已写入 %s（%d 字节，%dx%d）" % (out, len(png), SIZE, SIZE))


if __name__ == "__main__":
    main()
