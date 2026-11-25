#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cloud_field.py —— 程序化生成"符合大气环流"的云量场（供 gen_earth_tex.py 调用）

为什么单独一个文件：这是本次唯一带美术调参的物理模型，独立出来便于快速迭代
（`python3 cloud_field.py` 直接出等距圆柱预览图 + 球面预览）。

模型（由粗到细，全部是"环流 → 云"的定性物理，不做数值天气预报）：

1. **纬向气候带**（三圈环流）：
   · 赤道辐合带 ITCZ：深对流，云量高 —— 位置跟随太阳赤纬（滞后约 1 个月），
     故随生成日期南北摆动（南半球夏季时 ITCZ 移到南半球）。
   · 副热带高压带（±20°~35°）：下沉气流 → 云量极低（世界主要沙漠所在纬度）。
   · 中纬度风暴路径（±40°~65°）：锋面气旋频繁 → 云量中高。
   · 极区：层云/层积云大范围覆盖。
2. **海陆差异**：副热带洋面（冷洋流上）有大片海洋层积云（加利福尼亚/秘鲁/纳米比亚
   外海那种"云毯"），同纬度陆地则晴空；赤道洋面深对流更旺。
3. **天气尺度结构**：用**域扭曲的分形噪声**（domain-warped FBM）+ 若干**中纬度气旋涡旋**
   的旋转位移，得到真实云场那种螺旋/逗号状、被拉伸成丝缕的形态。
4. 最后做轻度高斯模糊（消除格点噪声）并量化到 0..255。

噪声在**单位球上的三维值噪声**里取（天然无经度接缝）；生成分辨率 2048x1024，
再由 gen_earth_tex.py 按面积平均降到 361x181（1° 网格），与地表贴图同一套重采样。
"""

import math
import numpy as np

# ---- 网格（与 gen_earth_tex.py 的源图一致）-------------------------------
SRC_W, SRC_H = 1440, 720   # 4x 目标网格（361x181）：噪声细节到 ~0.5°，重采样后仍保留 1° 尺度的锐利团块

# ---- 气候带参数 ----------------------------------------------------------
ITCZ_LAG = 0.55
ITCZ_MEANDER = 6.0       # ITCZ 纬度沿经度的摆动幅度（度）          # ITCZ 纬度 = ITCZ_LAG × 太阳赤纬（滞后/减弱系数）
ITCZ_SIG = 5.0           # ITCZ 半宽（度）
SUB_LAT = 24.0           # 副热带高压带中心纬度（度，南北对称）
SUB_SIG = 9.5            # 副热带带半宽
MID_LAT = 52.0           # 中纬度风暴路径中心
MID_SIG = 14.0

# ---- 噪声参数 ------------------------------------------------------------
WARP_AMP = 0.34          # 域扭曲强度（越大越"卷"）
WARP_FREQ = 2.8
FBM_FREQ = 4.6
FBM_OCT = 7
CYCLONE_N = 9            # 中纬度气旋个数
CYCLONE_SWIRL = 0.55     # 涡旋旋转位移强度
CYCLONE_SIG = 0.11       # 涡旋半径（球面弧度）


# ---------------------------------------------------------------- 噪声

def _lattice(n, seed, rng_cache):
    key = (n, seed)
    if key not in rng_cache:
        rng = np.random.default_rng(seed * 7919 + n)
        v = rng.random((n, n, n)).astype(np.float32) * 2.0 - 1.0
        rng_cache[key] = v
    return rng_cache[key]


def value_noise3(x, y, z, n, seed, rng_cache):
    """周期性三维值噪声（三线性 + smoothstep 插值），输入可越界（取模环绕）。"""
    L = _lattice(n, seed, rng_cache)
    xi = np.floor(x); yi = np.floor(y); zi = np.floor(z)
    xf = x - xi; yf = y - yi; zf = z - zi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = yf * yf * (3.0 - 2.0 * yf)
    w = zf * zf * (3.0 - 2.0 * zf)
    x0 = (xi.astype(np.int64) % n); x1 = (x0 + 1) % n
    y0 = (yi.astype(np.int64) % n); y1 = (y0 + 1) % n
    z0 = (zi.astype(np.int64) % n); z1 = (z0 + 1) % n

    def g(a, b, c):
        return L[a, b, c]

    c00 = g(x0, y0, z0) * (1 - u) + g(x1, y0, z0) * u
    c10 = g(x0, y1, z0) * (1 - u) + g(x1, y1, z0) * u
    c01 = g(x0, y0, z1) * (1 - u) + g(x1, y0, z1) * u
    c11 = g(x0, y1, z1) * (1 - u) + g(x1, y1, z1) * u
    c0 = c00 * (1 - v) + c10 * v
    c1 = c01 * (1 - v) + c11 * v
    return c0 * (1 - w) + c1 * w


def fbm3(px, py, pz, freq, octaves, seed, rng_cache):
    """分形布朗运动：倍频叠加值噪声，返回约 [-1,1]。"""
    acc = np.zeros_like(px, dtype=np.float32)
    amp = 1.0
    tot = 0.0
    f = freq
    for k in range(octaves):
        n = max(4, int(round(f * 2)))
        acc += amp * value_noise3(px * f, py * f, pz * f, n, seed + k, rng_cache)
        tot += amp
        amp *= 0.5
        f *= 2.0
    return acc / tot


def ridged3(px, py, pz, freq, octaves, seed, rng_cache):
    """脊状分形噪声（1-|noise| 叠加）：产生**丝缕/纤维**状结构 —— 锋面云带、
    卷云羽、对流外流边界的形态都来自它；纯 FBM 只会给出棉絮团。"""
    acc = np.zeros_like(px, dtype=np.float32)
    amp = 1.0
    tot = 0.0
    f = freq
    for k in range(octaves):
        n = max(4, int(round(f * 2)))
        v = np.abs(value_noise3(px * f, py * f, pz * f, n, seed + k, rng_cache))
        acc += amp * (1.0 - 2.0 * v)
        tot += amp
        amp *= 0.55
        f *= 2.1
    return acc / tot


# ---------------------------------------------------------------- 云量场

def cirrus_field(seed=20261125, rng_cache=None):
    """高空卷云（薄、半透明、丝缕状）。

    与低云完全不同的形态来源：卷云是**高空急流切变**拉出来的纤维，所以
      · 用高频**脊状**噪声（丝缕）而不是团块；
      · 主要出现在急流带（±35°~60°）与热带对流外流（ITCZ 上方），
      · 覆盖率低但覆盖面广，透明度低（0.10~0.45），叠加在低云之上会明显增加画面细节。
    """
    if rng_cache is None:
        rng_cache = {}
    jj, ii = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
    lon = (ii + 0.5) / SRC_W * 360.0 - 180.0
    lat = 90.0 - (jj + 0.5) / SRC_H * 180.0
    latr = np.radians(lat)
    lonr = np.radians(lon)
    px = np.cos(latr) * np.cos(lonr)
    py = np.cos(latr) * np.sin(lonr)
    pz = np.sin(latr)
    # 强各向异性的域扭曲：沿经度方向拉长（急流切变把云拉成条带）
    wx = fbm3(px, py, pz, 2.0, 3, seed + 700, rng_cache)
    wy = fbm3(px, py, pz, 2.0, 3, seed + 800, rng_cache)
    wz = fbm3(px, py, pz, 2.0, 3, seed + 900, rng_cache)
    qx = px + 0.55 * wx
    qy = py + 0.55 * wy * 0.35          # 纬向拉伸：y 方向扰动小 → 云系沿纬圈拉长
    qz = pz + 0.55 * wz * 0.35
    fr = ridged3(qx, qy, qz, 9.0, 4, seed + 1000, rng_cache)
    alat = np.abs(lat)
    jet = np.exp(-((alat - 46.0) / 15.0) ** 2)
    trop = np.exp(-((lat - (-0.35 * 20.0)) / 12.0) ** 2)
    cov = 0.34 + 0.55 * jet + 0.30 * trop
    cov = cov + 0.55 * fr
    cov = np.clip((cov - 0.72) * 2.6, 0.0, 1.0) ** 0.9
    return (cov * 0.55).astype(np.float32)     # 卷云最多 ~0.55 的不透明度


def cloud_field(water, sun_dec, seed=20261125, rng_cache=None):
    """生成 (SRC_H, SRC_W) 的云量场（0..1）。

    water   : (SRC_H, SRC_W) 布尔/0-1 数组，True = 海面（用于海陆差异）
    sun_dec : 生成日期的太阳赤纬（度）—— 决定 ITCZ 的南北位置
    """
    if rng_cache is None:
        rng_cache = {}
    # 海面掩码：内部上采样到云场网格（最近邻）
    w = np.asarray(water).astype(np.uint8) * 255
    if w.shape != (SRC_H, SRC_W):
        from PIL import Image
        w = np.asarray(Image.fromarray(w).resize((SRC_W, SRC_H), Image.NEAREST))
    wf = (w > 127).astype(np.float32)

    jj, ii = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
    lon = (ii + 0.5) / SRC_W * 360.0 - 180.0
    lat = 90.0 - (jj + 0.5) / SRC_H * 180.0
    latr = np.radians(lat)
    lonr = np.radians(lon)
    # 单位球坐标（x 指向 0°E 赤道，y 指向 90°E 赤道，z 指北极）
    px = np.cos(latr) * np.cos(lonr)
    py = np.cos(latr) * np.sin(lonr)
    pz = np.sin(latr)

    # ---- 域扭曲：先用低频噪声把采样点推开，得到云带被拉伸/卷曲的形态 ----
    wx = fbm3(px, py, pz, WARP_FREQ, 3, seed + 100, rng_cache)
    wy = fbm3(px, py, pz, WARP_FREQ, 3, seed + 200, rng_cache)
    wz = fbm3(px, py, pz, WARP_FREQ, 3, seed + 300, rng_cache)
    qx = px + WARP_AMP * wx
    qy = py + WARP_AMP * wy
    qz = pz + WARP_AMP * wz

    # ---- 中纬度气旋：局地旋转位移（螺旋/逗号状云系）----
    rng = np.random.default_rng(seed)
    for _ in range(CYCLONE_N):
        clat = math.radians(rng.uniform(34.0, 66.0) * rng.choice([-1.0, 1.0]))
        clon = math.radians(rng.uniform(-180.0, 180.0))
        cx = math.cos(clat) * math.cos(clon)
        cy = math.cos(clat) * math.sin(clon)
        cz = math.sin(clat)
        # 球面距离（近似：用三维弦长）
        dx = px - cx; dy = py - cy; dz = pz - cz
        r2 = dx * dx + dy * dy + dz * dz
        g = np.exp(-r2 / (CYCLONE_SIG * CYCLONE_SIG)).astype(np.float32)
        # 绕"本地天顶"的方向旋转位移：切向 = 天顶 × 径向
        tx = (cz * dy - cy * dz)
        ty = (cx * dz - cz * dx)
        tz = (cy * dx - cx * dy)
        norm = np.sqrt(tx * tx + ty * ty + tz * tz) + 1e-6
        swirl = CYCLONE_SWIRL * g
        qx = qx + swirl * tx / norm
        qy = qy + swirl * ty / norm
        qz = qz + swirl * tz / norm

    # ---- 云量：三尺度叠加（大尺度气旋/云系 + 中尺度团簇 + 脊状丝缕）----
    f = fbm3(qx, qy, qz, FBM_FREQ, FBM_OCT, seed + 400, rng_cache)          # 大尺度
    fm = fbm3(qx, qy, qz, FBM_FREQ * 2.1, 4, seed + 500, rng_cache)         # 中尺度团簇
    fr = ridged3(qx, qy, qz, FBM_FREQ * 2.9, 3, seed + 600, rng_cache)      # 脊状丝缕

    alat = np.abs(lat)
    itcz_lat = ITCZ_LAG * sun_dec
    # ---- 赤道辐合带：**不是一条绕圈的实心环** ----
    #   真实 ITCZ 是"断续的对流团簇链"：纬度沿经度南北摆动几度、强度沿经度起伏、
    #   中间有大量断裂（东太平洋/大西洋常整段消失）。用两组沿经度的一维噪声来实现：
    #     · 摆动：低频（周期 ~60°），振幅 ±ITCZ_MEANDER 度
    #     · 强度：中频（周期 ~25°），0..1，直接调制带宽权重
    ex = np.cos(lonr); ey = np.sin(lonr)
    mn = value_noise3(ex * 3.0, ey * 3.0, 0.37, 8, seed + 1100, rng_cache)
    an = value_noise3(ex * 7.0, ey * 7.0, 0.71, 16, seed + 1200, rng_cache)
    an2 = value_noise3(ex * 13.0, ey * 13.0, 0.19, 32, seed + 1300, rng_cache)
    itcz_here = itcz_lat + ITCZ_MEANDER * mn
    itcz_amp = np.clip(0.50 + 0.65 * an + 0.30 * an2, 0.0, 1.0)
    itcz = np.exp(-((lat - itcz_here) / ITCZ_SIG) ** 2) * itcz_amp
    sub = np.exp(-((alat - SUB_LAT) / SUB_SIG) ** 2)
    mid = np.exp(-((alat - MID_LAT) / MID_SIG) ** 2)
    polar = np.clip((alat - 62.0) / 22.0, 0.0, 1.0) ** 1.5

    sea = wf
    land = 1.0 - sea

    zonal = (0.50
             + 0.32 * itcz
             + 0.22 * mid
             + 0.28 * polar
             - 0.26 * sub)
    # 副热带洋面的海洋层积云（冷洋流上方的云毯）；副热带陆地（沙漠）更晴
    deck = sea * sub * np.clip(0.55 + 1.1 * f, 0.0, 1.0)
    zonal = zonal + 0.30 * deck - 0.12 * land * sub
    # 赤道洋面深对流再加强一点
    zonal = zonal + 0.12 * itcz * sea

    cov = zonal + 0.26 * f + 0.15 * fm + 0.11 * fr
    # 软阈值（保留一层薄的半透明边，云才有"厚薄"而不是贴纸片）
    cov = np.clip((cov - 0.575) * 3.1, 0.0, 1.0) ** 0.92
    return cov.astype(np.float32)


# ---------------------------------------------------------------- 预览

def _blur(a, sigma):
    """可分离高斯模糊（经度方向环绕）。"""
    r = max(1, int(round(sigma * 3)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    out = a
    for axis in (0, 1):
        pad = r
        if axis == 1:
            p = np.concatenate([out[:, -pad:], out, out[:, :pad]], axis=1)
            tmp = np.zeros_like(out)
            for i, w in enumerate(k):
                tmp += w * p[:, i:i + out.shape[1]]
            out = tmp
        else:
            p = np.concatenate([out[:pad][::-1], out, out[-pad:][::-1]], axis=0)
            tmp = np.zeros_like(out)
            for i, w in enumerate(k):
                tmp += w * p[i:i + out.shape[0]]
            out = tmp
    return out


def main():
    from PIL import Image
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gen_earth_tex as G

    day = Image.open(G.fetch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "earth_src"), "day", G.SRC["day"], False)).convert("RGB")
    d = G.resample(day, day.width, day.height)
    # 用海面掩码（从 specular 图）
    spec = Image.open(G.fetch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "earth_src"), "spec", G.SRC["spec"], False)).convert("L")
    s = G.resample(spec, spec.width, spec.height)[:, :, 0]
    water = (s > 120.0)

    cov = cloud_field(water, -20.7)
    cov = _blur(cov, 0.55)
    out = '/tmp/cloud_preview'
    os.makedirs(out, exist_ok=True)
    img = Image.fromarray((np.clip(cov, 0, 1) * 255).astype(np.uint8))
    img.resize((1024, 512), Image.LANCZOS).save(os.path.join(out, 'clouds_equirect.png'))
    print('写出', os.path.join(out, 'clouds_equirect.png'), '平均云量', round(float(cov.mean()), 3))


if __name__ == "__main__":
    main()
