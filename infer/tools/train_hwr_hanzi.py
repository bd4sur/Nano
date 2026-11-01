#!/usr/bin/env python3
"""train_hwr_hanzi.py —— 手写汉字识别器（app/hwr.scm）的训练与权导出脚本

产物（由 app/hwr.scm 在运行时用 (Fs.read) 读入）：
    infer/app/lib/hwr/meta.scm   类数/维数/量化尺度
    infer/app/lib/hwr/w?.txt     int8 量化权重，每文件 per_part 类 × nfeat 维
    infer/app/lib/hwr/params.txt 每类的尺度（16 位定点）+ 偏置 + 汉字表

为什么要导出成「文本数据文件」而不是 Scheme 源码：
    灵机解析器按【源码字符数】预分配 token 数组（am_token_t 在 wasm32 上 40 字节/字符），
    256k 字符的字符串字面量会独占 10MB 解释器内存池。权重里每个字节写成码点 256+q 的
    字符（UTF-8 两字节），读入后即可 String.charCodeAt 随机访问，无需解码成列表。

模型（为什么这么小）：
    灵机的算术是解释执行的（每次乘加约 40 条 VM 指令），wasm 上 25.6 万次乘加约 0.3 秒。
    故用「手工特征 + 线性 softmax」，而不是端到端 CNN：
        输入  墨迹网格（64×64）→ 取墨迹包围盒 → 双线性重采样到 26×26 → 居中放进 32×32
        特征  4 方向梯度（上下/左右/两对角中心差分取绝对值）在 8×8 网格上分块求和
              → 256 维，L2 归一化（对笔画粗细、墨迹浓淡不敏感）
        分类  256 → 1000（最常用汉字，字频降序）线性 softmax
              权重 256k 个，int8 逐类量化（每类一个尺度，精度损失 < 0.5%）

数据（不依赖任何真人笔迹数据集）：
    由 12 种字体风格（黑/宋/楷/行/美术字，含同一家族的多字重）× 1000 字渲染生成，
    每个（字, 字体）做 16 次随机增广：
        · 随机长宽比、笔画粗细（膨胀/腐蚀）、弹性形变、旋转/错切/缩放、墨迹浓淡
        · "手指书写模式"（45%）：细笔画 + 手抖 + 倾斜 + 32×32 笔刷封样（笔画半径 0/1 格）
        · 随机污点（撑大包围盒）+ 随机摆位/大小
    评估用【未参与训练】的 4 种字体（含草书，作难度上界）+ 两字体上的"手指仿真"集。

用法：
    python3 tools/train_hwr_hanzi.py                  # 生成数据 → 训练 → 评估 → 导出
    python3 tools/train_hwr_hanzi.py --classes 1000 --epochs 100 --aug 16
    python3 tools/train_hwr_hanzi.py --export-only    # 复用缓存的 npz 重新导出

字体准备（放到 tools/fonts/，或改 --font-dir）：
    系统自带即可用：arphic ukai/uming(.ttc)、Noto Sans/Serif CJK(.ttc)、Droid Sans Fallback
    Google Fonts 的手写/美术风格（免费，直接下载 ttf）：
        Ma Shan Zheng / Zhi Mang Xing / Long Cang / ZCOOL KuaiLe /
        ZCOOL QingKe HuangYou / ZCOOL XiaoWei / LXGW WenKai TC / Noto Sans SC
        https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng  → 取其中的 .ttf 链接
"""
import argparse, os, sys, time, json

import numpy as np
from PIL import Image, ImageFont, ImageDraw

GRID = 32          # 模型输入边长
BOX = 26           # 归一化后墨迹占用边长
OFF = 3            # (32-26)//2
CANVAS = 64        # 增广画布
RENDER = 96        # 渲染像素
FINGER_P = 0.45    # 训练样本中"手指书写模式"的比例

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # infer/
CHARS_CSV = os.path.join(HERE, 'hanzi.csv')        # 字频表（序号/汉字/字频/….）
OUT_DIR = os.path.join(ROOT, 'app', 'lib', 'hwr')

# 训练字体（12 种风格）。(.ttc 需要给 face index)
TRAIN_FONTS = [
    ('fonts/notosanssc.ttf', 0),
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 2),
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 2),
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Thin.ttc', 2),
    ('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc', 2),
    ('fonts/lxgwwenkai.ttf', 0),
    ('/usr/share/fonts/truetype/arphic/ukai.ttc', 2),
    ('fonts/mashanzheng.ttf', 0),
    ('fonts/zhimangxing.ttf', 0),
    ('fonts/zcoolkuaile.ttf', 0),
    ('fonts/zcoolqkhy.ttf', 0),
    ('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', 0),
]
# 评估用字体（完全不参与训练）
TEST_FONTS = [
    ('/usr/share/fonts/truetype/arphic/uming.ttc', 2),
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc', 2),
    ('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', 0),
]


# ---------------------------------------------------------------------------
# 几何增广
# ---------------------------------------------------------------------------
def _elastic(a, rng, alpha=0.28, smooth=4):
    h, w = a.shape
    gh, gw = max(2, h // smooth + 2), max(2, w // smooth + 2)
    dy = rng.uniform(-1, 1, (gh, gw)).astype(np.float32) * alpha * h / 8.0
    dx = rng.uniform(-1, 1, (gh, gw)).astype(np.float32) * alpha * w / 8.0
    dy = np.asarray(Image.fromarray(dy).resize((w, h), Image.BICUBIC))
    dx = np.asarray(Image.fromarray(dx).resize((w, h), Image.BICUBIC))
    yy, xx = np.mgrid[0:h, 0:w]
    sy = np.clip(yy + dy, 0, h - 1)
    sx = np.clip(xx + dx, 0, w - 1)
    y0 = np.floor(sy).astype(np.int32); x0 = np.floor(sx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1); x1 = np.minimum(x0 + 1, w - 1)
    fy = sy - y0; fx = sx - x0
    return (a[y0, x0] * (1 - fy) * (1 - fx) + a[y0, x1] * (1 - fy) * fx
            + a[y1, x0] * fy * (1 - fx) + a[y1, x1] * fy * fx).astype(np.float32)


def _dilate(a, r):
    """方形结构元膨胀（r>0）/ 腐蚀（r<0），用于改笔画粗细。"""
    if r == 0:
        return a
    k = abs(int(r))
    p = np.pad(a, k, mode='constant')
    h, w = a.shape
    out = p[k:k + h, k:k + w].copy() if r < 0 else a.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            v = p[k + dy:k + dy + h, k + dx:k + dx + w]
            out = np.minimum(out, v) if r < 0 else np.maximum(out, v)
    return out


def _affine(a, ang, shx, s):
    """旋转 ang 度 + 水平错切 shx + 缩放 s（逆向映射双线性）。画布放大 1.6 倍防裁切。"""
    h, w = a.shape
    H, W = int(h * 1.6) + 4, int(w * 1.6) + 4
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = xx - W / 2.0; Y = yy - H / 2.0
    t = np.radians(-ang)
    Xr = X * np.cos(t) - Y * np.sin(t); Yr = X * np.sin(t) + Y * np.cos(t)
    Xs = Xr - shx * Yr; Ys = Yr
    Xi = Xs / s + w / 2.0; Yi = Ys / s + h / 2.0
    x0 = np.clip(np.floor(Xi).astype(np.int32), 0, w - 1); x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(Yi).astype(np.int32), 0, h - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    fx = Xi - np.floor(Xi); fy = Yi - np.floor(Yi)
    out = (a[y0, x0] * (1 - fy) * (1 - fx) + a[y0, x1] * (1 - fy) * fx
           + a[y1, x0] * fy * (1 - fx) + a[y1, x1] * fy * fx)
    mask = (Xi >= -0.5) & (Xi <= w - 0.5) & (Yi >= -0.5) & (Yi <= h - 0.5)
    return np.clip(out * mask, 0, 1).astype(np.float32)


def render_char(path, ch, size=RENDER, index=0):
    """渲染单个字 → 裁紧 bbox 的灰度图（0..1）。缺字返回 None。"""
    try:
        font = ImageFont.truetype(path, size, index=index)
    except Exception:
        return None
    img = Image.new('L', (size * 2, size * 2), 0)
    ImageDraw.Draw(img).text((size // 2, size // 3), ch, fill=255, font=font)
    a = np.asarray(img, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.25)
    if len(ys) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ---------------------------------------------------------------------------
# 归一化（**与 app/hwr.scm 的 normalize 逐位一致**：阈值 0.10、中心对齐双线性、越界取 0）
# ---------------------------------------------------------------------------
def normalize(canvas):
    ys, xs = np.nonzero(canvas > 0.10)
    if len(ys) == 0:
        return np.zeros((GRID, GRID), np.float32)
    t, b, l, r = ys.min(), ys.max(), xs.min(), xs.max()
    h = float(b - t + 1); w = float(r - l + 1)
    out = np.zeros((GRID, GRID), np.float32)
    H, W = canvas.shape
    sy = h / BOX; sx = w / BOX
    for i in range(BOX):
        yy = t + (i + 0.5) * sy - 0.5
        for j in range(BOX):
            xx = l + (j + 0.5) * sx - 0.5
            y0 = int(np.floor(yy)); x0 = int(np.floor(xx))
            fy = yy - y0; fx = xx - x0

            def g(a, c):
                if a < 0 or c < 0 or a >= H or c >= W:
                    return 0.0
                return float(canvas[a, c])
            out[OFF + i, OFF + j] = (g(y0, x0) * (1 - fy) * (1 - fx) + g(y0, x0 + 1) * (1 - fy) * fx
                                     + g(y0 + 1, x0) * fy * (1 - fx) + g(y0 + 1, x0 + 1) * fy * fx)
    return np.clip(out, 0, 1)


def _to_canvas(small, rng):
    """把一张 32×32 的"已书写"图放大放进 64×64 画布（随机大小/位置）。"""
    target = rng.uniform(0.66, 0.95) * CANVAS
    sc = target / small.shape[1]
    n = max(8, int(round(small.shape[1] * sc)))
    up = np.asarray(Image.fromarray(small.astype(np.float32)).resize((n, n), Image.BICUBIC))
    canvas = np.zeros((CANVAS, CANVAS), np.float32)
    ox = int(rng.integers(0, CANVAS - n + 1)); oy = int(rng.integers(0, CANVAS - n + 1))
    canvas[oy:oy + n, ox:ox + n] = np.clip(up, 0, 1)
    return canvas


def finger_sim(glyph, rng):
    """渲染成"手指在 32×32 网格上写字"的样子：细笔画 + 手抖 + 倾斜 + 笔刷封样。"""
    a = _dilate(glyph, -int(rng.integers(0, 3)))
    a = _elastic(a, rng, alpha=rng.uniform(0.08, 0.28))
    a = _affine(a, rng.uniform(-11, 11), rng.uniform(-0.2, 0.2), rng.uniform(0.94, 1.06))
    g = np.asarray(Image.fromarray(a.astype(np.float32)).resize((GRID, GRID), Image.BOX))
    g = np.clip(g * rng.uniform(1.6, 3.2), 0, 1)
    g = _dilate(g, int(rng.integers(0, 2)))          # 笔宽：半径 0/1 格
    k = np.array([[0.45, 0.75, 0.45], [0.75, 1.0, 0.75], [0.45, 0.75, 0.45]], np.float32)
    p = np.pad(g, 1)
    out = np.zeros_like(g)
    for dy in range(3):
        for dx in range(3):
            out = np.maximum(out, p[dy:dy + GRID, dx:dx + GRID] * k[dy, dx])
    return out


def make_sample(glyph, rng, finger=None):
    """字形 → 64×64 画布（含手写风格增广）。
    finger=True 强制走"手指书写模式"，False 强制普通模式，None 按 FINGER_P 概率随机。"""
    if finger is True or (finger is None and rng.random() < FINGER_P):
        return _to_canvas(finger_sim(glyph, rng), rng)
    a = glyph
    ax = rng.uniform(0.82, 1.22); ay = rng.uniform(0.82, 1.22)
    a = np.asarray(Image.fromarray(a).resize((max(4, int(a.shape[1] * ax)),
                                              max(4, int(a.shape[0] * ay))), Image.BICUBIC))
    if rng.random() < 0.75:
        a = _dilate(a, int(rng.integers(-1, 3)))
    if rng.random() < 0.85:
        a = _elastic(a, rng, alpha=rng.uniform(0.10, 0.45))
    a = _affine(a, rng.uniform(-13, 13), rng.uniform(-0.22, 0.22), rng.uniform(0.96, 1.04))
    a = np.clip(a * rng.uniform(0.75, 1.15), 0, 1)
    canvas = np.zeros((CANVAS, CANVAS), np.float32)
    h, w = a.shape[:2]
    target = rng.uniform(0.62, 0.94) * CANVAS
    sc = target / max(h, w)
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    small = np.asarray(Image.fromarray(np.asarray(a, np.float32)).resize((nw, nh), Image.BOX))
    ox = int(rng.integers(0, CANVAS - nw + 1)); oy = int(rng.integers(0, CANVAS - nh + 1))
    canvas[oy:oy + nh, ox:ox + nw] = np.maximum(canvas[oy:oy + nh, ox:ox + nw], small)
    if rng.random() < 0.30:                       # 污点：手指误触，会撑大包围盒
        for _ in range(int(rng.integers(1, 3))):
            px = int(rng.integers(0, CANVAS)); py = int(rng.integers(0, CANVAS))
            r = int(rng.integers(1, 3))
            canvas[max(0, py - r):py + r, max(0, px - r):px + r] = rng.uniform(0.5, 1.0)
    return canvas


# ---------------------------------------------------------------------------
# 特征：4 方向梯度在 8×8 网格上分块求和（**与 app/hwr.scm 的 featurize 逐位一致**）
# ---------------------------------------------------------------------------
def featurize(X, cells=8):
    N = X.shape[0]
    p = np.pad(X, ((0, 0), (1, 1), (1, 1)), mode='constant')
    c = p[:, :-2, 1:-1] - p[:, 2:, 1:-1]        # 上 - 下
    h = p[:, 1:-1, :-2] - p[:, 1:-1, 2:]        # 左 - 右
    d1 = p[:, :-2, :-2] - p[:, 2:, 2:]          # 主对角
    d2 = p[:, :-2, 2:] - p[:, 2:, :-2]          # 副对角
    step = GRID // cells
    out = [np.abs(d).reshape(N, cells, step, cells, step).sum(axis=(2, 4)).reshape(N, -1)
           for d in (c, h, d1, d2)]
    F = np.concatenate(out, axis=1)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# 数据生成（多进程：一个进程一个字体）
# ---------------------------------------------------------------------------
def load_chars(k):
    out = []
    for line in open(CHARS_CSV, encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) > 1 and len(p[1]) == 1:
            out.append(p[1])
    return out[:k]


def _gen_font(args):
    fn, idx, chars, naug, seed, finger = args   # finger: None=按比例混合, True=强制手指模式
    path = fn if os.path.isabs(fn) else os.path.join(HERE, fn)
    if not os.path.exists(path):
        print(f'  [跳过] 缺字体 {fn}', flush=True)
        return np.zeros((0, GRID, GRID), np.uint8), np.zeros(0, np.int32)
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for ci, ch in enumerate(chars):
        g = render_char(path, ch, index=idx)
        if g is None:
            continue
        for _ in range(naug):
            n = normalize(make_sample(g, rng, finger=finger))
            if n.max() <= 0:
                continue
            X.append((n * 255).astype(np.uint8)); Y.append(ci)
    print(f'  {os.path.basename(fn)}[{idx}]: {len(X)}', flush=True)
    return np.array(X, np.uint8), np.array(Y, np.int32)


def build(fonts, chars, naug, seed, finger=None):
    from multiprocessing import Pool
    with Pool(min(len(fonts), os.cpu_count() or 4)) as pool:
        parts = pool.map(_gen_font, [(f, i, chars, naug, seed + k, finger)
                                     for k, (f, i) in enumerate(fonts)])
    X = np.concatenate([a for a, b in parts]); Y = np.concatenate([b for a, b in parts])
    return X, Y


# ---------------------------------------------------------------------------
# 训练：线性 softmax + Adam（注意：Adam + 权重衰减会互相打架，l2 取 0）
# ---------------------------------------------------------------------------
def train(F, Y, K, epochs=100, lr=0.03, bs=256, seed=0):
    N, D = F.shape
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((D, K)) * np.sqrt(1.0 / D)).astype(np.float64)
    b = np.zeros(K)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0
    for ep in range(epochs):
        idx = rng.permutation(N)
        tot = 0.0; cnt = 0
        for s in range(0, N, bs):
            j = idx[s:s + bs]
            x = F[j]; y = Y[j]
            z = x @ W + b
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z); p = e / e.sum(axis=1, keepdims=True)
            tot += -np.log(p[np.arange(len(y)), y] + 1e-9).sum(); cnt += len(y)
            g = p; g[np.arange(len(y)), y] -= 1.0; g /= len(y)
            gW = x.T @ g; gb = g.sum(axis=0)
            t += 1
            for pm, gm, mm, vv in ((W, gW, mW, vW), (b, gb, mb, vb)):
                mm *= 0.9; mm += 0.1 * gm
                vv *= 0.999; vv += 0.001 * gm * gm
                pm -= lr * (mm / (1 - 0.9 ** t)) / (np.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
        if ep % 10 == 9 or ep == epochs - 1:
            print(f'  ep{ep+1:3d} loss={tot/cnt:.3f} train_acc={acc(F, Y, W, b):.4f}', flush=True)
    return W, b


def acc(F, Y, W, b):
    return float((np.argmax(F @ W + b, 1) == Y).mean())


def topk_acc(F, Y, W, b, k):
    idx = np.argsort(-(F @ W + b), 1)[:, :k]
    return float(np.mean([Y[i] in idx[i] for i in range(len(Y))]))


# ---------------------------------------------------------------------------
# 导出：int8 逐类量化 → UTF-8 文本数据文件
# ---------------------------------------------------------------------------
def export(W, b, chars, outdir, per_part=200):
    W = W.T.astype(np.float64)                 # (K, D) 类主序
    K, D = W.shape
    srow = np.maximum(np.abs(W).max(1) / 127.0, 1e-9)
    q = np.clip(np.round(W / srow[:, None]) + 128, 0, 255).astype(np.int32)
    smax = float(srow.max())
    sq = np.clip(np.round(srow / smax * 65535.0), 0, 65535).astype(np.int32)
    sb = float(np.abs(b).max()) / 127.0
    qb = np.clip(np.round(b / sb) + 128, 0, 255).astype(np.int32)
    # 反量化误差（应为 0）
    Wq = (q - 128) * (sq / 65535.0 * smax)[:, None]
    print(f'  量化校验: max|ΔW|={np.abs(W - Wq).max():.4g}  max|Δb|={np.abs(b - (qb-128)*sb).max():.4g}')

    os.makedirs(outdir, exist_ok=True)
    npart = (K + per_part - 1) // per_part
    for p in range(npart):
        lo, hi = p * per_part, min(K, (p + 1) * per_part)
        txt = ''.join(chr(int(v) + 256) for v in q[lo:hi].reshape(-1))
        open(os.path.join(outdir, f'w{p}.txt'), 'w', encoding='utf-8', newline='').write(txt)
    scal = ''.join(chr(int(v >> 8) + 256) + chr(int(v & 255) + 256) for v in sq)
    bias = ''.join(chr(int(v) + 256) for v in qb)
    open(os.path.join(outdir, 'params.txt'), 'w', encoding='utf-8', newline='').write(
        scal + bias + ''.join(chars))
    open(os.path.join(outdir, 'meta.scm'), 'w').write(
        ';; 由 tools/train_hwr_hanzi.py 生成（勿手改）：手写汉字识别器的模型元信息\n'
        f'(define ncls {K})\n(define nfeat {D})\n(define npart {npart})\n'
        f'(define per_part {per_part})\n(define smax {smax:.9g})\n(define sb {sb:.9g})\n')
    print(f'  导出 {npart} 个权重文件 + params.txt + meta.scm → {outdir}')


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--aug', type=int, default=16)
    ap.add_argument('--cache', default='/tmp/hwr_ds.npz')
    ap.add_argument('--export-only', action='store_true')
    args = ap.parse_args()

    chars = load_chars(args.classes)
    K = len(chars)
    print(f'== 类别 {K}（最常用汉字，字频降序） / 特征 256 维 / 增广 ×{args.aug}')

    if args.export_only and os.path.exists(args.cache):
        d = np.load(args.cache)
        X, Y, Xt, Yt, Xf, Yf = d['X'], d['Y'], d['Xt'], d['Yt'], d['Xf'], d['Yf']
    else:
        t0 = time.time()
        print('== 生成训练数据（12 种字体 × 增广 + 手指模式）')
        X, Y = build(TRAIN_FONTS, chars, args.aug, 100)
        print(f'   train {X.shape}  {time.time()-t0:.0f}s', flush=True)
        print('== 生成评估数据（未参与训练的字体 + 手指仿真）')
        Xt, Yt = build(TEST_FONTS, chars, 3, 900)
        Xf, Yf = build(TEST_FONTS[:2], chars, 3, 950, finger=True)
        print(f'   font {Xt.shape}  finger {Xf.shape}')
        np.savez_compressed(args.cache, X=X, Y=Y, Xt=Xt, Yt=Yt, Xf=Xf, Yf=Yf)

    t0 = time.time()
    F = featurize(X.astype(np.float32) / 255.)
    print('== 训练（线性 softmax + Adam）')
    W, b = train(F, Y, K, epochs=args.epochs)
    print(f'   {time.time()-t0:.0f}s')
    Ft = featurize(Xt.astype(np.float32) / 255.)
    Ff = featurize(Xf.astype(np.float32) / 255.)
    print('== 评估')
    print(f'   未见字体 : top1={acc(Ft,Yt,W,b):.4f} top5={topk_acc(Ft,Yt,W,b,5):.4f} top8={topk_acc(Ft,Yt,W,b,8):.4f}')
    print(f'   手指仿真 : top1={acc(Ff,Yf,W,b):.4f} top5={topk_acc(Ff,Yf,W,b,5):.4f} top8={topk_acc(Ff,Yf,W,b,8):.4f}')
    print('== 导出')
    export(W, b, chars, OUT_DIR)
    json.dump({'classes': K, 'feat': 256, 'aug': args.aug, 'epochs': args.epochs,
               'top1_font': acc(Ft, Yt, W, b), 'top1_finger': acc(Ff, Yf, W, b),
               'top8_finger': topk_acc(Ff, Yf, W, b, 8)},
              open(os.path.join(OUT_DIR, 'train_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()