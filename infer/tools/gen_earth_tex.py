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

列 0 的中心在 (120°E - 179.5°)（= 120°E + (0 − 179.5)）。这只是数据的原点，
**与观察位置无关**：App 现在可以在任意一点的天空、任意滚转角俯视（见其文件头
「相机」一节），取址公式随视点一起旋转。

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

def cloud_channel(water, date, seed=20261125, kind="low"):
    """程序化云量场（1°/格，0..255）。模型见 tools/cloud_field.py 的文件头：

    · 纬向气候带（ITCZ / 副热带高压 / 中纬度风暴路径 / 极区层云）
    · 海陆差异（副热带洋面的海洋层积云、沙漠晴空）
    · 域扭曲分形噪声 + 中纬度气旋涡旋 → 螺旋/丝缕状的真实云系形态
    · ITCZ 位置跟随生成日期的太阳赤纬（南半球夏季时南移）
    """
    import cloud_field as CF
    # 由日期粗算太阳赤纬（±0.3° 足够，只用来摆放 ITCZ）
    try:
        y, m, d = [int(x) for x in date.split("-")]
        doy = sum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30][:m - 1]) + d
        dec = -23.44 * math.cos(2.0 * math.pi * (doy + 10.0) / 365.24)
    except Exception:
        dec = -20.0
    cov = CF.cloud_field(water, dec, seed=seed)
    cov = CF._blur(cov, 1.6)
    img = Image.fromarray((np.clip(cov, 0.0, 1.0) * 255.0).astype(np.uint8))
    v = resample(img, img.width, img.height)[:, :, 0]
    return np.clip(np.rint(v), 0, 255).astype(np.int32)


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
;;   · 白天 RGB 三通道 + 夜间灯光 + **低云云量 TXW** + **高空卷云 TXC**，各一串字符，
;;     字符码点 = 0x100 + 分量值；
;;   · 夜间通道的水面像素额外 +0x100（码点 >= 0x200），供太阳反光只落在水上；
;;   · 361 列 × 181 行：列 = 经度（1°/列，第 360 列是第 0 列的环绕复制，
;;     列 0 的中心在 120°E-179.5°），行 = sin(纬度)（第 180 行复制第 179 行）。
;;     **贴图本身与视点无关**：视点可以在任意一点、任意滚转角，取址公式见下。
;;   · 云量通道 TXW 由 tools/cloud_field.py **程序化生成**（不是卫星数据）：
;;     纬向环流带（ITCZ/副热带高压/中纬度风暴路径/极区）+ 海陆差异 + 域扭曲分形噪声
;;     + 中纬度气旋涡旋，具体模型见该脚本文件头；
;;   取址（相机可任意，见 app/earthport.scm 文件头的「相机」一节）：
;;     sinLat = sinφ0·xU + cosφ0·xN,   Δλ = atan2(xE, cosφ0·xU - sinφ0·xN)
;;     row = 四舍五入((1 - sinLat) * 90),  col = floor(Δλ° + λ0 + 60)
;;     idx = row * 361 + col      （第 r 行的中心在行号 r+0.5 ⇒ 四舍五入到最近 texel）
;;   φ0=0、λ0=120°E、滚转 0 时退化为定点俯视：row = (1 + dy/90)*90、col = atan2(u,w)°+180。
;; 生成脚本：infer/tools/gen_earth_tex.py（含参考渲染器，可出多时刻预览图）。
;; 用法：python3 infer/tools/gen_earth_tex.py
;; ============================================================================

(define TW {tw})            ; 列数（1°/列，含环绕复制列）
(define TH {th})            ; 行数（sin(lat) 一列行，含极区复制行）
'''


def write_scheme(path, dr, dg, db, night_code, cloud, cirrus, date):
    body = [HEADER.format(date=date, tw=TW, th=TH)]
    for name, arr in (("TXR", dr), ("TXG", dg), ("TXB", db), ("TXN", night_code),
                      ("TXW", cloud), ("TXC", cirrus)):
        body.append('(define %s "%s")\n' % (name, enc(arr)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(body))
    return os.path.getsize(path)


# ---------------------------------------------------------------- 参考渲染（美术校对）

R_GL = 90.0
CX, CY = 160.0, 104.0
BG_R, BG_G, BG_B = 1.0, 2.0, 6.0

# 大气（与 app/earthport.scm 逐条一致；改这里必须同步改那边）
HS = 0.033            # 大气标高（归一化半径）
HALO_U = 0.045        # 球外辉光径向厚度（归一化半径，≈4px；真实大气约 1.6% → 1.4px，留 3 倍余量）
AIR0 = 1.0 / math.sqrt(2.0 * HS)
S3L = AIR0 ** 3 - 0.55          # b=1 处的散射量（内外圈在此严格连续）
ATM_R, ATM_G, ATM_B = 0.021, 0.076, 0.260   # 散射通道增益（色调映射前）
ATM_AMB = 0.007       # 夜侧气辉/多次散射下限（夜缘只留一道暗蓝弧）
ATM_FLOOR = 0.55      # 空气量立方的扣除量（天顶方向的散射几乎抵消，只留淡淡的蓝）
ATM_LIFT = 0.10       # 大气被照到的最低太阳天顶角余弦（≈地平线下 7°）
ATM_LIFT_K = 4.5      # 抬升后的照明斜率（dot≈+0.10 时饱和）
EDGE_FADE = 0.071     # 地表边缘羽化（≈ 最外 7px 淡出）

# 白天
SUN_FLOOR = 0.014     # 无直射时的地表残余反照率
SUN_SOFT = 0.10       # 直射份额软化宽度
DUSK_HI, DUSK_SPAN = 0.12, 0.33   # 城市灯点亮区间
SUN_RED = 0.30        # 低太阳红化斜率
# 米氏散射（气溶胶）：近全谱白 + 前向散射 + 柱密度 ∝ 空气量²
MIE_G = 0.65          # HG 各向异性
MIE_ISO = 0.60        # 各向同性底
MIE_A = 0.035         # 柱密度项（随空气量²）
MIE_FLAT = 0.014      # 平坦项（气溶胶垂直光学厚度带来的均匀白纱）
CLOUD_MAX = 0.92      # 云量 255 时的最大遮蔽
CLOUD_K = CLOUD_MAX / 255.0
CIRRUS_K = 0.55 / 255.0   # 卷云通道 0..255 → 不透明度 0..0.55
CIRRUS_ALB = 246.0        # 卷云反照率（薄而亮，略偏冷）
CLOUD_LIFT = 0.035    # 云顶比地面高 ⇒ 晨昏线附近晚一点变暗
CLOUD_BASE = 188.0    # 极薄云的反照率（灰白）
CLOUD_DENSE = 62.0    # 厚度带来的增量（厚云顶 → 纯白）
GLINT_CORE = 150.0    # 太阳反光光斑强度
GLINT_CORE_K = 110.0  # 光斑角半径
GLINT_WIDE = 58.0     # 海面宽反光强度
GLINT_WIDE_K = 20.0

LIGHT_R, LIGHT_G, LIGHT_B = 1.28, 0.98, 0.62

# 贴图生成参数
CLOUD_STRETCH_OFF = 0.30   # 云量对比度拉伸：偏移
CLOUD_STRETCH_GAIN = 0.44  # 云量对比度拉伸：增益
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


def sun_dir_at(dec_deg, slon_deg, lat0, lon0):
    """任意视点 (lat0, lon0) 的局部系（东, 北, 天）太阳单位向量。

    slon_deg = 次太阳点经度 λs（= α - GMST），lat0/lon0 为观察者所在点。
    与 app/earthport.scm 的 sun_local 逐式对应：
        Δ = λs - λ0；sE = cosδ·sinΔ
        sN = -sinφ0·cosδ·cosΔ + cosφ0·sinδ
        sU =  cosφ0·cosδ·cosΔ + sinφ0·sinδ
    （lat0 = 0、lon0 = 120°E 时与 sun_dir(dec, hour=120-λs) 完全一致。）
    """
    d = math.radians(dec_deg)
    dl = math.radians(slon_deg - lon0)
    l0 = math.radians(lat0)
    cdl = math.cos(d) * math.cos(dl)
    return (math.cos(d) * math.sin(dl),
            -math.sin(l0) * cdl + math.cos(l0) * math.sin(d),
            math.cos(l0) * cdl + math.sin(l0) * math.sin(d))


CAM_DEFAULT = (0.0, 120.0, 0.0)      # (lat0, lon0, roll)：观察者所在点 + 画面滚转角


def shade_pixel(dx, dy, se, sn, su, hx, hy, hz, day_rgb, night_v, water, mp=1.0, cw=None, cc=None,
                cam=CAM_DEFAULT):
    """单个像素的最终颜色（None = 不写，保留背景）。公式与 app/earthport.scm 一一对应。

    **昼夜不分枝**：地表亮度只由太阳天顶角余弦 dot 驱动（sunv = max(0,dot) 连续降到 0），
    城市灯随 dusk 连续点亮，大气照明与红化也只由 dot 连续驱动 —— 这样晨昏线上不会出现
    亮度/色相断层。

    cam = (lat0, lon0, roll)：观察者所在点与画面滚转角（度）。屏幕切平面分量先旋到
    视点局部系（东/北），再补天顶分量 w = sqrt(1-q)：
        xE =  p·cosρ + qs·sinρ,   xN = -p·sinρ + qs·cosρ
        sinLat = sinφ0·w + cosφ0·xN,   Δλ = atan2(xE, cosφ0·w - sinφ0·xN)
    着色（dot、切点投影 dl、半角向量）全部在局部系里算，所以只需 (se,sn,su) 随视点变。
    """
    lat0, lon0, roll = cam
    p = dx / R_GL
    qs = -dy / R_GL
    q = p * p + qs * qs
    rho = math.radians(roll)
    xe = p * math.cos(rho) + qs * math.sin(rho)
    xn = -p * math.sin(rho) + qs * math.cos(rho)
    cl0 = math.cos(math.radians(lat0))
    sl0 = math.sin(math.radians(lat0))

    def atmos(air, dl):
        """大气散射：air = 视线方向空气量；dl = **切点法向**的太阳投影（跨视圆盘边缘连续）。

        返回三段色调映射后的加色值。拆成两股：
          · 日照那股（按入射阳光的气柱长度红化 + warm³ 衰减）
          · 环境那股（气辉/多次散射，保持蓝、不红化）
        """
        s3 = air * air * air - ATM_FLOOR
        if s3 < 0.0:
            s3 = 0.0
        lit = (dl + ATM_LIFT) * ATM_LIFT_K
        lit = 0.0 if lit < 0.0 else (1.0 if lit > 1.0 else lit)
        sunv = dl if dl > 0.0 else 0.0
        airs = 1.0 / math.sqrt(sunv * sunv + 2.0 * HS)
        warm = 1.0 / (1.0 + SUN_RED * (airs - 1.0))
        sl = s3 * (1.0 - ATM_AMB) * lit * warm * warm * warm
        sa = s3 * ATM_AMB
        out = []
        # 米氏白雾（三通道等权）：压向灰白、抬起暗部
        mie = (MIE_FLAT + MIE_A * air * air) * lit * mp * warm
        for k, g in ((ATM_R, 2.0 - warm), (ATM_G, 0.4 + 0.6 * warm), (ATM_B, warm)):
            a = sl * k * g + sa * k + mie
            out.append(255.0 * a / (a + 1.0))
        return out


    if q <= 1.0:
        w = math.sqrt(max(0.0, 1.0 - q))
        # 球面点的纬度与经度偏移（行号 ∝ sinLat；列中心经度 = 列 + 60）
        sinlat = sl0 * w + cl0 * xn
        dlon = math.atan2(xe, cl0 * w - sl0 * xn)
        row = min(TH - 1, max(0, int((1.0 - sinlat) * 90.0 + 0.5)))
        col = int((math.degrees(dlon) + lon0 + 60.0) % 360.0)
        col = min(TW - 1, max(0, col))
        dot = xe * se + xn * sn + w * su
        r0 = day_rgb[row, col, 0]
        g0 = day_rgb[row, col, 1]
        b0 = day_rgb[row, col, 2]
        sunv = dot if dot > 0.0 else 0.0
        sfac = SUN_FLOOR + (1.0 - SUN_FLOOR) * (sunv / (sunv + SUN_SOFT))
        k = sfac * (0.44 + 0.56 * w)
        r, g, b = r0 * k, g0 * k, b0 * k
        dusk = min(1.0, max(0.0, (DUSK_HI - dot) / DUSK_SPAN))
        if dusk > 0.0:
            lk = night_v[row, col] * dusk
            r += lk * 1.28
            g += lk * 0.98
            b += lk * 0.62
        nh = xe * hx + xn * hy + w * hz
        if nh > 0.0 and water[row, col]:
            dev = 1.0 - nh
            t1 = 1.0 + dev * GLINT_CORE_K
            t2 = 1.0 + dev * GLINT_WIDE_K
            gl = GLINT_CORE / (t1 * t1) + GLINT_WIDE / (t2 * t2)
            r += gl
            g += gl * 0.98
            b += gl * 0.94
        if cw is not None:
            ca = cw[row, col] * CLOUD_K
            if ca > 0.0:
                csun = sunv + CLOUD_LIFT
                cl = (SUN_FLOOR + (1.0 - SUN_FLOOR) * csun / (csun + SUN_SOFT)) \
                     * (0.62 + 0.38 * w) * (CLOUD_BASE + CLOUD_DENSE * ca)
                r = r * (1.0 - ca) + cl * ca
                g = g * (1.0 - ca) + cl * ca
                b = b * (1.0 - ca) + cl * 1.03 * ca
        if cc is not None:
            ca = cc[row, col] * CIRRUS_K
            if ca > 0.0:
                csun = sunv + CLOUD_LIFT
                cl = (SUN_FLOOR + (1.0 - SUN_FLOOR) * csun / (csun + SUN_SOFT)) \
                     * (0.70 + 0.30 * w) * CIRRUS_ALB
                r = r * (1.0 - ca) + cl * ca
                g = g * (1.0 - ca) + cl * ca
                b = b * (1.0 - ca) + cl * 1.05 * ca
        t = min(1.0, (1.0 - q) * R_GL * EDGE_FADE)
        cov = t * t * (3.0 - 2.0 * t)
        r, g, b = r * cov, g * cov, b * cov
        # 照明/入射气柱用的天顶角余弦：max(dot, dl) = dl + max(0, w·sUp)
        dl = xe * se + xn * sn
        datm = max(dot, dl)
        ar, ag, ab = atmos(1.0 / math.sqrt(1.0 - q + 2.0 * HS), datm)
        return (r + ar, g + ag, b + ab)

    bo = math.sqrt(q) - 1.0
    if bo < HALO_U:
        # 球外只有大气：柱密度从 b=1 处的值起，按三次缓出平滑归零（末段斜率为 0）
        xo = 1.0 - bo / HALO_U
        s3 = S3L * xo * xo * xo
        dl = xe * se + xn * sn
        lit = (dl + ATM_LIFT) * ATM_LIFT_K
        lit = 0.0 if lit < 0.0 else (1.0 if lit > 1.0 else lit)
        sunv = dl if dl > 0.0 else 0.0
        airs = 1.0 / math.sqrt(sunv * sunv + 2.0 * HS)
        warm = 1.0 / (1.0 + SUN_RED * (airs - 1.0))
        sl = s3 * (1.0 - ATM_AMB) * lit * warm * warm * warm
        sa = s3 * ATM_AMB
        air_o = AIR0 * xo
        mie = (MIE_FLAT + MIE_A * air_o * air_o) * lit * mp * warm
        out = []
        for k, g in ((ATM_R, 2.0 - warm), (ATM_G, 0.4 + 0.6 * warm), (ATM_B, warm)):
            a = sl * k * g + sa * k + mie
            out.append(255.0 * a / (a + 1.0))
        return (BG_R + out[0], BG_G + out[1], BG_B + out[2])
    return None


def mie_phase(su):
    """米氏相函数（HG，g=MIE_G）：平行入射+平行投影 ⇒ 盘面处处 cosΘ = sUp。"""
    hg = (1.0 - MIE_G ** 2) / (1.0 + MIE_G ** 2 - 2.0 * MIE_G * su) ** 1.5
    hg1 = (1.0 - MIE_G ** 2) / (1.0 - MIE_G) ** 1.5
    return MIE_ISO + (1.0 - MIE_ISO) * hg / hg1


def preview(day_rgb, night_v, water, se, sn, su, path, cw=None, cc=None, cam=CAM_DEFAULT,
            with_ui=False):
    """按 app/earthport.scm 的公式复刻一遍（数值逐条对齐），输出 320x240 PNG。

    cam = (lat0, lon0, roll)。with_ui=False 时只画地球（不画控制条与 HUD），
    便于与 App 帧缓冲做逐像素比较（比较时应只在圆盘内取均值）。
    """
    mp = mie_phase(su)
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
            px = shade_pixel(x - CX, y - CY, se, sn, su, hx, hy, hz, day_rgb, night_v, water, mp, cw, cc, cam)
            if px is not None:
                img[y, x, 0] = min(255.0, px[0])
                img[y, x, 1] = min(255.0, px[1])
                img[y, x, 2] = min(255.0, px[2])
    Image.fromarray(img.astype(np.uint8)).resize((W * 2, H * 2), Image.NEAREST).save(path)
    return path


def check_continuity(day_rgb, night_v, water, se, sn, su, label=""):
    """着色模型的连续性自检：沿屏幕竖直线扫描，算相邻像素的最大通道跳变。

    **用纯色贴图 + 关掉夜灯**，这样量到的全是着色模型自身的跳变；否则贴图里的
    海岸线/雪线/城市灯会把结果淹没（实测真实贴图能到 300+，纯色 14）。
    球缘（q>0.85）单独统计 —— 那里大气柱密度的梯度本来就陡（边缘亮环是设计）。
    """
    hx, hy, hz = se, sn, su + 1.0
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    hx, hy, hz = hx / hn, hy / hn, hz / hn
    flat = np.full((TH, TW, 3), 140.0)
    nz = np.zeros((TH, TW))
    nw = np.ones((TH, TW), dtype=bool)
    worst_in, where_in = 0.0, ""
    worst_limb, where_limb = 0.0, ""
    for x in range(100, 220, 4):
        prev = None
        for y in range(10, 198):
            px = shade_pixel(x - CX, y - CY, se, sn, su, hx, hy, hz, flat, nz, nw)
            if px is None:
                prev = None
                continue
            px = tuple(min(255.0, v) for v in px)
            if prev is not None:
                d = max(abs(px[i] - prev[i]) for i in range(3))
                q = ((x - CX) ** 2 + (y - CY) ** 2) / (R_GL * R_GL)
                if q < 0.85:
                    if d > worst_in:
                        worst_in, where_in = d, "x=%d,y=%d" % (x, y)
                elif d > worst_limb:
                    worst_limb, where_limb = d, "x=%d,y=%d" % (x, y)
            prev = px
    print("连续性自检%s：盘面内部(q<0.85) 最大相邻跳变 %.1f (%s)；球缘 %.1f (%s)"
          % (label, worst_in, where_in, worst_limb, where_limb))
    return worst_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "earth_src"))
    ap.add_argument("--out", default="/tmp/earthtex_preview")
    ap.add_argument("--out-scm", default=APP_TEX)
    ap.add_argument("--date", default="2026-11-25")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--vec", default="", help="额外按给定太阳向量出一张预览图：se,sn,su")
    ap.add_argument("--cam", default="",
                    help="视角：lat,lon,roll（度）。给出时只出一张该视角的预览图（需 --vec）")
    ap.add_argument("--check", action="store_true", help="只跑着色模型的连续性自检")
    args = ap.parse_args()

    dr, dg, db, ncode, water, day_rgb, night_v = build_channels(args.src, not args.no_fetch)
    cloud = cloud_channel(water, args.date)
    cirrus = cloud_channel(water, args.date, kind="cirrus")
    if args.cam:
        os.makedirs(args.out, exist_ok=True)
        lat0, lon0, roll = [float(x) for x in args.cam.split(",")]
        if not args.vec:
            sys.exit("--cam 需配合 --vec se,sn,su（太阳在**该视点局部系**的分量）")
        se, sn, su = [float(x) for x in args.vec.split(",")]
        night_v2 = np.rint(night_v).astype(np.int32)
        p = preview(day_rgb, night_v2, water, se, sn, su,
                    os.path.join(args.out, "preview_cam.png"), cloud, cirrus,
                    cam=(lat0, lon0, roll))
        print("预览（指定视角）", p)
        return

    size = write_scheme(args.out_scm, dr, dg, db, ncode, cloud, cirrus, args.date)
    print("写出 %s（%.0f KB，%d x %d x 4 通道）" % (args.out_scm, size / 1024.0, TW, TH))

    if args.check:
        night_v2 = np.rint(night_v).astype(np.int32)
        worst = 0.0
        for name, dec, hour in [("正午", 0.0, 0.0), ("晨昏", -14.7, 55.0),
                                ("全夜", -14.7, 170.0), ("黎明", -23.0, -80.0)]:
            se, sn, su = sun_dir(dec, hour)
            worst = max(worst, check_continuity(day_rgb, night_v2, water, se, sn, su, "(%s)" % name))
        print("结论：%s（判据：盘面内部相邻像素跳变 < 20 视为连续，晨昏线上不得出现断层）"
              % ("通过" if worst < 20 else "不通过"))
        return

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
                        os.path.join(args.out, "preview_%s.png" % name), cloud, cirrus)
            print("预览", p)
        if args.vec:
            se, sn, su = [float(x) for x in args.vec.split(",")]
            p = preview(day_rgb, night_v2, water, se, sn, su,
                        os.path.join(args.out, "preview_vec.png"), cloud, cirrus)
            print("预览（指定太阳向量）", p)


if __name__ == "__main__":
    main()
