#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_earth_tex.py —— 生成《地球舷窗》（app/earthport.scm）的地表贴图数据模块
==========================================================================

产物
----
  infer/app/lib/earth_tex.scm   灵机（Animac）模块：四串字符 = 四个通道的贴图数据
  <out>/preview_*.png           参考渲染预览（Python 复刻 Scheme 侧逐像素着色，仅供美术校对）

数据组织（与 app/earthport.scm 的取址约定必须逐条一致）
--------------------------------------------------------
网格是**等距圆柱的纵向重采样版**：列 = 经度（1° 一列，361 列，第 360 列是第 0 列的
环绕复制，这样 Scheme 侧算列号不用做 mod）；行 = sin(纬度)（181 行，第 180 行复制
第 179 行），于是「正交投影下屏幕 y 线性映射到 sin(lat)」→ 行号 = (1+sinlat)*90，
一次乘加即可，不必查 asin。

列 0 的中心在 (120°E - 179.5°)，即贴图以「东经 120°、纬度 0°」为画面中心（视点正
下方就是 120°E / 0°N），与 App 的「定点俯视」一致。

四个通道各存成一串字符串，每个字符的码点是 0x100 + v（v = 0..255）：
  码点 0x100..0x1FF 是 Latin-1 补充区，UTF-8 两字节，且不含引号/反斜杠/换行/空字符，
  因此可以安全地嵌在源码字符串字面量里（0x00 会截断 C 串、0x0A 会断行、0x22/0x5C
  需要转义——统统避开）。Scheme 侧用 (String.charCodeAt idx s) 读一个通道再减 256。

  TXR/TXG/TXB  白天地表颜色（NASA Blue Marble，无云）
  TXN          夜间灯光强度；**水面加 0x100 偏移**（码点 0x200..0x2FF）→ Scheme 侧
               用「码点 > 511」判水面，供太阳反光光斑只在海面/冰面上出现。

为什么用字符串而不是数字列表：灵机解释器为每个数字字面量建 AST 节点，一个 6.5 万元
素的引号列表要吃掉 ~14MB 虚拟机工作区（实测），四张表直接超池；而同样数据的字符串
字面量只占 ~1MB 堆区，且 (String.charCodeAt) 与 (get_item) 同为 O(1) 本地调用。

数据来源与授权
--------------
  · 白天：NASA Visible Earth / Blue Marble（公有领域），经 three.js 示例贴图分发
  · 夜间：NASA Earth Observatory / Black Marble（公有领域），同上
  · 水面掩码：同上（earth_specular，海洋为白）
本脚本把三张原图按面积平均降到 361x181，并做纵向 sin(lat) 重采样。

用法
----
  python3 infer/tools/gen_earth_tex.py            # 源图缺失时自动经代理下载
  HTTP_PROXY=http://host:port python3 ...         # 或直接给环境变量
  --out DIR    预览图输出目录（默认 /tmp/earthtex_preview）
  --src DIR    源图缓存目录（默认 infer/tools/earth_src）
  --no-fetch   只用已缓存的源图
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image

# ---- 网格与取址常量（必须与 app/earthport.scm 的 TW/TH/WRAP 一致）-----------
TW = 361          # 列数（含环绕复制列）
TH = 181          # 行数（含极区复制行）
COLS = 360        # 有效经度列数（1°/列）
CENTER_LON = 120.0

SRC = {
    "day": "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_atmos_2048.jpg",
    "night": "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_lights_2048.png",
    "spec": "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/planets/earth_specular_2048.jpg",
}

HERE = os.path.dirname(os.path.abspath(__file__))
APP_TEX = os.path.join(HERE, "..", "app", "lib", "earth_tex.scm")


# ---------------------------------------------------------------- 源图

def fetch(src_dir, name, url, allow_fetch=True):
    path = os.path.join(src_dir, name + os.path.splitext(url)[1])
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return path
    if not allow_fetch:
        sys.exit("缺少源图：%s（去掉 --no-fetch 让它自动下载）" % path)
    os.makedirs(src_dir, exist_ok=True)
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    cmd = ["curl", "-sS", "-m", "300", "--retry", "3", "-o", path, url]
    if proxy:
        cmd[1:1] = ["-x", proxy]
    print("下载", url)
    subprocess.run(cmd, check=True)
    return path


# ---------------------------------------------------------------- 重采样

def axis_weights(n_src, centers_src, n_dst):
    """返回 (n_dst, n_src) 的面积重叠权重矩阵，用于一维盒式（面积平均）重采样。"""
    w = np.zeros((n_dst, n_src), dtype=np.float64)
    for i in range(n_dst):
        a, b = centers_src[i]
        lo, hi = int(math.floor(a - 0.5)), int(math.ceil(b + 0.5))
        for j in range(lo, hi + 1):
            jj = j % n_src
            ov = min(b, j + 0.5) - max(a, j - 0.5)
            if ov > 0:
                w[i, jj] += ov
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return w / s


def resample(img, W, H):
    """把等距圆柱原图（H x W, lat +90..-90, lon -180..180）重采样到 TW x TH。"""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    hs, ws = arr.shape[0], arr.shape[1]

    # 纵向：目标行 r 覆盖 sin(lat) 区间 → 反解纬度区间
    row_edges = []
    for r in range(TH):
        s1 = 1.0 - (r + 0.5) / (TH - 1) * 2.0
        s1 = max(-1.0, min(1.0, s1))
        d = math.degrees(math.asin(s1))
        # 行中心 → 源像素坐标（行中心 = (r+0.5)/TH*H - 0.5 的等距网格，这里按纬度反解）
        y = (90.0 - d) / 180.0 * hs - 0.5
        row_edges.append(y)
    # 用「相邻行中心的中点」作为行带边界（等距原图的行是等纬度的，故直接插值即可）
    yb = []
    for r in range(TH):
        y0 = (row_edges[r - 1] + row_edges[r]) / 2.0 if r > 0 else row_edges[0] - 0.5
        y1 = (row_edges[r] + row_edges[r + 1]) / 2.0 if r < TH - 1 else row_edges[-1] + 0.5
        yb.append((y0 + 0.5, y1 + 0.5))   # 换成像素边界坐标

    wr = axis_weights(hs, yb, TH)

    # 横向：列 c 中心经度 = 120 + (c - 179.5)
    xb = []
    for c in range(TW):
        lon = CENTER_LON + (c - 179.5)
        xc = (lon + 180.0) / 360.0 * ws
        xb.append((xc - 0.5, xc + 0.5))
    wc = axis_weights(ws, xb, TW)

    out = np.tensordot(wr, arr, axes=([1], [0]))       # (TH, W, C)
    out = np.tensordot(wc, out, axes=([1], [1]))       # (TW, TH, C)
    return np.transpose(out, (1, 0, 2))                # (TH, TW, C)



# ---------------------------------------------------------------- 通道

def build_channels(src_dir, allow_fetch=True):
    day = Image.open(fetch(src_dir, "day", SRC["day"], allow_fetch)).convert("RGB")
    night = Image.open(fetch(src_dir, "night", SRC["night"], allow_fetch)).convert("RGB")
    spec = Image.open(fetch(src_dir, "spec", SRC["spec"], allow_fetch)).convert("L")

    d = resample(day, day.width, day.height)
    n = resample(night, night.width, night.height)
    s = resample(spec, spec.width, spec.height)[:, :, 0]

    # 白天：屏幕只有 180px 的球，Blue Marble 原图偏暗偏灰。海水与陆地各用一条曲线
    # （目视调参，见预览图）：海走低 gamma + 蓝增益（深蓝→中蓝），陆走高 gamma +
    # 轻微暖化（保留雪线不过曝）。两条曲线按 spec 掩码连续混合，海岸线不留缝。
    dv = np.clip(d, 0.0, 255.0) / 255.0
    sea = 255.0 * np.power(dv, DAY_SEA_GAMMA) * np.array([0.48, 0.76, 1.22])
    lnd = 255.0 * np.power(dv, DAY_LAND_GAMMA)
    lum_l = (0.299 * lnd[:, :, 0] + 0.587 * lnd[:, :, 1] + 0.114 * lnd[:, :, 2])[:, :, None]
    lnd = (lum_l + (lnd - lum_l) * DAY_LAND_SAT) * np.array([1.04, 1.00, 0.93])
    t = np.clip((s - 70.0) / 140.0, 0.0, 1.0)[:, :, None]
    day_rgb = np.clip(lnd * (1.0 - t) + sea * t, 0.0, 255.0)

    # 夜间灯光：先扣掉大气辉光底噪，再用 gamma 提亮，保住稀疏的灯点
    nlum = np.maximum(0.0, 0.299 * n[:, :, 0] + 0.587 * n[:, :, 1] + 0.114 * n[:, :, 2] - 13.0)
    nlum = np.clip(nlum / (200.0 - 13.0), 0.0, 1.0) ** 0.62
    # 降到 1° 后，城市周边的"漫射光晕"会连成一片土黄色（陆地中位数 ~43）——扣掉这段
    # 底噪，只留真正的城区亮点，夜面才够黑、灯点才够亮。
    nlum = np.clip((nlum * 255.0 - NIGHT_FLOOR) / (255.0 - NIGHT_FLOOR), 0.0, 1.0) ** NIGHT_CONTRAST
    night_v = nlum * 255.0

    water = (s > 120.0)
    # 夜间通道：陆地 = nv（写成码点 0x100+nv），水面再加 0x100（码点 0x200+nv）。
    # enc() 会统一加 0x100，故这里只加水面偏移。
    night_code = np.rint(night_v).astype(np.int32) + water.astype(np.int32) * 0x100

    # 四舍五入成 8 位整数
    dr = np.rint(day_rgb[:, :, 0]).astype(np.int32)
    dg = np.rint(day_rgb[:, :, 1]).astype(np.int32)
    db = np.rint(day_rgb[:, :, 2]).astype(np.int32)
    return dr, dg, db, night_code, water, day_rgb, night_v


def enc(arr):
    """把 0..255（或 0x100+v）的整数矩阵编码成一串「码点 0x100+v」的 UTF-8 字符。"""
    flat = arr.reshape(-1)
    return "".join(chr(0x100 + int(v)) for v in flat)


# ---------------------------------------------------------------- Scheme 模块

HEADER = ''';; ============================================================================
;; 地球地表贴图数据（**生成物，勿手改**）
;;
;; 由 infer/tools/gen_earth_tex.py 从 NASA Blue Marble / Black Marble（公有领域，
;; 经 three.js 示例贴图分发）重采样而来：
;;   · 白天 RGB 三通道 + 夜间灯光一通道，各一串字符，字符码点 = 0x100 + 分量值；
;;   · 夜间通道的水面像素额外 +0x100（码点 >= 0x200），供太阳反光只落在水上；
;;   · 361 列 × 181 行：列 = 经度（1°/列，第 360 列是第 0 列的环绕复制，
;;     列 0 的中心在 120°E-179.5°），行 = sin(纬度)（第 180 行复制第 179 行）。
;;   取址：row = (1 + sin(lat)) * 90；col = atan2(u, w) 的度数 + 180；
;;         idx = row * 361 + col —— 见 app/earthport.scm。
;; 生成脚本：infer/tools/gen_earth_tex.py（含参考渲染器，可出多时刻预览图）。
;; 用法：python3 infer/tools/gen_earth_tex.py
;; ============================================================================

(define TW {tw})            ; 列数（1°/列，含环绕复制列）
(define TH {th})            ; 行数（sin(lat) 一列行，含极区复制行）
'''


def write_scheme(path, dr, dg, db, night_code, date):
    body = [HEADER.format(date=date, tw=TW, th=TH)]
    for name, arr in (("TXR", dr), ("TXG", dg), ("TXB", db), ("TXN", night_code)):
        body.append('(define %s "%s")\n' % (name, enc(arr)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(body))
    return os.path.getsize(path)


# ---------------------------------------------------------------- 参考渲染（美术校对）

R_GL = 90.0
CX, CY = 160.0, 118.0
BG_R, BG_G, BG_B = 1.0, 2.0, 6.0

# 大气（与 app/earthport.scm 逐条一致；改这里必须同步改那边）
RING_K = 96.0         # 大气"硬"亮环的衰减尺度（1/xr，越大越细）
HAZE_K = 22.0         # 大气"软"雾的衰减尺度
BLOOM_K = 6.0         # 圆外的宽辉光
RING_A = 0.66
RIM_BASE = 0.13       # 大气环的底亮度（夜侧/背光侧）
RIM_LIT = 0.72        # 大气环随太阳天顶角
RIM_SUN = 0.62        # 大气环随 max(0,sUp)²（前向散射：太阳在镜头背后时整圈亮）
HAZE_A = 0.34
BLOOM_A = 0.30
HALO_XR = 0.13        # 圆外辉光的最远距离（归一化半径），窗口函数在此归零
WIN_K = 0.13          # = HALO_XR

# 白天
DAY_AMB = 0.05
DAY_SOFT = 0.06
GLINT_CORE = 205.0    # 太阳反光光斑强度
GLINT_CORE_K = 220.0  # 光斑角半径（1/K 量级为角偏差的平方）
GLINT_WIDE = 26.0     # 海面宽反光强度
GLINT_WIDE_K = 30.0
TWILIGHT_K = 3.0
NIGHT_BASE = 0.020
LIGHT_R, LIGHT_G, LIGHT_B = 1.28, 0.98, 0.62

# 贴图生成参数
NIGHT_FLOOR = 52.0     # 夜间灯光的底噪扣除
NIGHT_CONTRAST = 0.82  # 夜间灯光的对比度
DAY_SEA_GAMMA = 0.66   # 白天贴图：海水提亮 gamma
DAY_LAND_GAMMA = 0.78  # 白天贴图：陆地提亮 gamma
DAY_LAND_SAT = 1.14    # 白天贴图：陆地饱和度


def sun_dir(dec_deg, hour_deg):
    """返回视点局部系（东, 北, 天）里的太阳单位向量；hour = 120°E 的时角。"""
    d = math.radians(dec_deg)
    h = math.radians(hour_deg)
    return (math.cos(d) * math.sin(-h), math.sin(d), math.cos(d) * math.cos(h))


def shade_pixel(dx, dy, se, sn, su, hx, hy, hz, day_rgb, night_v, water):
    """单个像素的最终颜色（None = 不写，保留背景）。公式与 Scheme 侧一一对应。"""
    q = (dx * dx + dy * dy) / (R_GL * R_GL)
    xr = abs(1.0 - q) * 0.5
    u = dx / R_GL
    sn_ = -dy / R_GL
    ring = 1.0 / (1.0 + (xr * RING_K) * (xr * RING_K))
    haze = 1.0 / (1.0 + (xr * HAZE_K) * (xr * HAZE_K))
    if q <= 1.0:
        w = math.sqrt(max(0.0, 1.0 - q))
        dot = u * se + sn_ * sn + w * su
        row = min(TH - 1, max(0, int((1.0 + dy / R_GL) * 90.0)))
        col = min(TW - 1, max(0, int(math.degrees(math.atan2(u, w)) + 180.0)))
        r0 = day_rgb[row, col, 0]
        g0 = day_rgb[row, col, 1]
        b0 = day_rgb[row, col, 2]
        lit = max(0.0, dot)
        if dot > 0.0:
            k = (DAY_AMB + (1.0 - DAY_AMB) * (dot / (dot + DAY_SOFT))) * (0.58 + 0.42 * w)
            r, g, b = r0 * k, g0 * k, b0 * k
            nh = u * hx + sn_ * hy + w * hz
            if nh > 0.0 and water[row, col]:
                dev = 1.0 - nh
                t1 = 1.0 + dev * GLINT_CORE_K
                t2 = 1.0 + dev * GLINT_WIDE_K
                gl = GLINT_CORE / (t1 * t1) + GLINT_WIDE / (t2 * t2)
                r += gl * 1.00
                g += gl * 0.98
                b += gl * 0.94
        else:
            nv = night_v[row, col]
            twi = min(1.0, max(0.0, 1.0 + dot * TWILIGHT_K))
            dk = 1.0 - 0.85 * twi
            r = r0 * NIGHT_BASE * 0.65 + nv * LIGHT_R * dk + twi * 9.0
            g = g0 * NIGHT_BASE * 0.85 + nv * LIGHT_G * dk + twi * 15.0
            b = b0 * NIGHT_BASE * 1.10 + nv * LIGHT_B * dk + twi * 34.0
        aa = min(1.0, RIM_BASE + RIM_LIT * lit + RIM_SUN * max(0.0, su) ** 2)
        a2 = aa * aa
        a3 = a2 * aa
        hh = RING_A * ring + HAZE_A * haze
        r += hh * (8.0 + 200.0 * a3)
        g += hh * (18.0 + 238.0 * a2)
        b += hh * (44.0 + 212.0 * a2)
        return (r, g, b)
    if xr < HALO_XR:
        wn = 1.0 - xr / WIN_K
        win = wn * wn
        bloom = 1.0 / (1.0 + (xr * BLOOM_K) * (xr * BLOOM_K)) * win
        lit = max(0.0, u * se + sn_ * sn)
        aa = min(1.0, RIM_BASE + RIM_LIT * lit + RIM_SUN * max(0.0, su) ** 2)
        a2 = aa * aa
        a3 = a2 * aa
        hh = RING_A * ring + HAZE_A * haze * 0.75 + BLOOM_A * bloom
        return (BG_R + hh * (8.0 + 200.0 * a3),
                BG_G + hh * (18.0 + 238.0 * a2),
                BG_B + hh * (44.0 + 212.0 * a2))
    return None


def preview(day_rgb, night_v, water, se, sn, su, path):
    """按 app/earthport.scm 的公式复刻一遍（数值逐条对齐），输出 320x240 PNG。"""
    W, H = 320, 240
    img = np.zeros((H, W, 3), dtype=np.float64)
    img[:, :, 0] = BG_R
    img[:, :, 1] = BG_G
    img[:, :, 2] = BG_B
    hx, hy, hz = se, sn, su + 1.0
    hn = 1.0 / math.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx * hn, hy * hn, hz * hn
    for y in range(H):
        for x in range(W):
            px = shade_pixel(x - CX, y - CY, se, sn, su, hx, hy, hz, day_rgb, night_v, water)
            if px is not None:
                img[y, x, 0] = min(255.0, px[0])
                img[y, x, 1] = min(255.0, px[1])
                img[y, x, 2] = min(255.0, px[2])
    Image.fromarray(img.astype(np.uint8)).resize((W * 2, H * 2), Image.NEAREST).save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "earth_src"))
    ap.add_argument("--out", default="/tmp/earthtex_preview")
    ap.add_argument("--out-scm", default=APP_TEX)
    ap.add_argument("--date", default="2026-11-25")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--vec", default="", help="额外按给定太阳向量出一张预览图：se,sn,su")
    args = ap.parse_args()

    dr, dg, db, ncode, water, day_rgb, night_v = build_channels(args.src, not args.no_fetch)
    size = write_scheme(args.out_scm, dr, dg, db, ncode, args.date)
    print("写出 %s（%.0f KB，%d x %d x 4 通道）" % (args.out_scm, size / 1024.0, TW, TH))

    if not args.no_preview:
        os.makedirs(args.out, exist_ok=True)
        night_v2 = np.rint(night_v).astype(np.int32)
        cases = [
            ("noon", 0.0, 0.0),        # 太阳正对 120°E：全球白天
            ("terminator", -14.7, 55.0),  # 类似参考图：右半白天、左半夜
            ("night", -14.7, 170.0),   # 太阳在背面：全球夜灯
            ("dawn", -23.0, -80.0),
            ("harness", -18.34, -82.8),  # 与 gcc 回归台的冻结时钟一致（2023-11-14 22:13）
        ]
        for name, dec, hour in cases:
            se, sn, su = sun_dir(dec, hour)
            p = preview(day_rgb, night_v2, water, se, sn, su,
                        os.path.join(args.out, "preview_%s.png" % name))
            print("预览", p)
        if args.vec:
            se, sn, su = [float(x) for x in args.vec.split(",")]
            p = preview(day_rgb, night_v2, water, se, sn, su,
                        os.path.join(args.out, "preview_vec.png"))
            print("预览（指定太阳向量）", p)


if __name__ == "__main__":
    main()
