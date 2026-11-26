#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""frcnn_common.py —— Faster R-CNN 教学演示 App（app/frcnn.scm）的**离线端**公共件

本文件是 train_frcnn_digits.py 的「数据合成 + 模型 + 推理参照实现」部分，单独拆出来
是为了让「训练/导出」与「参照推理（numpy，逐位对齐 Scheme 侧实现）」共用同一套约定：

  · 图像：64x64 灰度，1~6 个数字随机摆位/随机大小/随机字体（印刷体 + MNIST 手写体）/
    随机倾斜 ±5°，允许重叠，但每个数字都完整落在画面内；
  · 增广：灰度抖动、高斯噪声、椒盐噪声、高斯模糊、二值化抖动、随机擦除；
  · 网络：conv1(3x3,s2)->conv2(3x3,s2) 骨干 → RPN(1x1 头 + 9 锚框) → RoI 池化 4x4
    → 分类头(11 类) + 回归头(每类 4 个偏移) → 类内 NMS。

**所有几何/数值约定在这里定义一次**，Python 侧与 Scheme 侧（app/lib/frcnn_core.scm）
逐条对应；改任何一条都必须两边一起改（文件头注释里也写明了）。
"""
import math
import os
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# 全局约定（与 Scheme 侧常量一一对应）
# ---------------------------------------------------------------------------
IMG = 64               # 输入边长
STRIDE = 4             # 特征图相对输入的步长（两次 stride 2 卷积）
GRID = IMG // STRIDE   # 16x16 特征图
NCH = 24               # conv2 输出通道数（也是 conv3 的通道数）
NCH1 = 8               # conv1 输出通道数
ANCH_SIZES = (12.0, 17.0, 24.0)     # 锚框 sqrt(面积)
ANCH_RATIOS = (0.5, 0.7, 0.95)      # 锚框 w/h（<1 即偏高，符合数字的形态）
NA = len(ANCH_SIZES) * len(ANCH_RATIOS)   # 9 个锚框/格
NCLS = 11              # 0..9 + 背景（背景只出现在分类头的 11 类里，RPN 用二分类）
POOL = 4               # RoI 池化输出 4x4
FDIM = POOL * POOL * NCH   # 分类头输入维数 384
HID = 64               # 分类头隐层
NROI = 20              # 推理时送入分类头的 RoI 数
PRE_NMS_TOPK = 100     # RPN 提议进入 NMS 前的数量
RPN_NMS_IOU = 0.7     # 提议级 NMS：**数字允许重叠**，阈值太狠会把相邻数字吃掉
FINAL_NMS_IOU = 0.35
CROSS_NMS_IOU = 0.45   # 逐类 NMS 之后再跑一遍「跨类」NMS：同一个数字被几个类各报一次时，
                       # 只留分数最高的那个（纯演示向的后处理，mAP 不变差）
SCORE_TH = 0.30        # 提议/最终框的最低分数
MEAN, STD = 0.80, 0.21  # 输入归一化（训练集 24000 张的均值/标准差）

MNIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mnist')

# 训练用的印刷体（本机 fontconfig 里真实存在；缺字体自动跳过）
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSerif.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf',
    '/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf',
]



# ---------------------------------------------------------------------------
# 字母表（像素 sprite）
# ---------------------------------------------------------------------------
def _bbox_crop(img, pad=1, thr=250):
    """把 L 模式图按非白包围盒裁掉（保留 pad 像素余量），返回 L 图。"""
    a = np.asarray(img)
    m = a < thr
    if not m.any():
        return img
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def load_printed_sprites():
    """印刷体：每个字体 10 个数字，返回 [(font_tag, digit, L图), ...]。"""
    out = []
    for path in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        tag = os.path.basename(path).split('.')[0]
        for d in range(10):
            img = Image.new('L', (128, 128), 255)
            dr = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(path, 84)
            except Exception:
                continue
            s = str(d)
            bb = dr.textbbox((0, 0), s, font=font)
            dr.text((8 - bb[0], 8 - bb[1]), s, font=font, fill=0)
            out.append((tag, d, _bbox_crop(img, pad=1)))
    return out


def _read_idx(path):
    import gzip
    with gzip.open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        ndim = magic & 0xFF
        dims = [int.from_bytes(f.read(4), 'big') for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


def load_mnist_sprites(per_class=None):
    """手写体：MNIST 训练集（每类取 per_class 张，None = 全部），返回 [(tag, digit, L图)]。"""
    ip = os.path.join(MNIST_DIR, 'train-images-idx3-ubyte.gz')
    lp = os.path.join(MNIST_DIR, 'train-labels-idx1-ubyte.gz')
    if not (os.path.exists(ip) and os.path.exists(lp)):
        return []
    imgs = _read_idx(ip)
    labs = _read_idx(lp)
    out = []
    count = {}
    for i in range(imgs.shape[0]):
        d = int(labs[i])
        count[d] = count.get(d, 0) + 1
        if per_class is not None and count[d] > per_class:
            continue
        a = imgs[i]
        # MNIST 前景为亮：转成「白底黑字」的 L 图
        im = Image.fromarray(255 - a, mode='L')
        out.append(('mnist', d, _bbox_crop(im, pad=1, thr=200)))
    return out


def load_mnist_test(n=400):
    """MNIST 测试集里取 n 张，用于「手指画」风格的稳健性检查（可选）。"""
    ip = os.path.join(MNIST_DIR, 't10k-images-idx3-ubyte.gz')
    if not os.path.exists(ip):
        return []
    imgs = _read_idx(ip)
    return [Image.fromarray(255 - imgs[i], mode='L') for i in range(min(n, imgs.shape[0]))]


# ---------------------------------------------------------------------------
# 合成一张图
# ---------------------------------------------------------------------------
def _paste_digit(canvas, sprite, cx, cy, h, angle, ink):
    """把 sprite 缩放/旋转后以「取更暗者」的方式贴到 canvas（L 模式，白底黑字）。

    返回该数字的轴对齐包围盒 (x1,y1,x2,y2)（浮点，未裁剪）。数字完整落在画面内由
    调用方保证（cx,cy 的取值域已收缩到「旋转后外接框不越界」）。
    """
    w = max(3, int(round(h * sprite.width / sprite.height)))
    h = max(3, int(round(h)))
    sp = sprite.resize((w, h), Image.BICUBIC)
    if abs(angle) > 0.01:
        sp = sp.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    # 墨色：把 sprite 的灰度按 ink∈[0,1] 线性映射（1 = 全黑，0.6 = 淡墨）
    a = np.asarray(sp).astype(np.float32) / 255.0
    a = 1.0 - (1.0 - a) * ink
    sp = Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), mode='L')
    w2, h2 = sp.size
    x0 = int(round(cx - w2 / 2.0))
    y0 = int(round(cy - h2 / 2.0))
    x0 = max(0, min(IMG - 1, x0))
    y0 = max(0, min(IMG - 1, y0))
    if x0 + w2 > IMG:
        x0 = IMG - w2
    if y0 + h2 > IMG:
        y0 = IMG - h2
    x0 = max(0, x0)
    y0 = max(0, y0)
    box = canvas.crop((x0, y0, x0 + w2, y0 + h2))
    canvas.paste(ImageChops.darker(box, sp), (x0, y0))
    return (float(x0) + 0.5, float(y0) + 0.5, float(x0 + w2) - 0.5, float(y0 + h2) - 0.5)


def synth_sample(rng, printed, hand, allow_overlap=True):
    """合成一张训练/测试图，返回 (img float32 64x64 ∈[0,1], boxes (N,4) float32, labels (N,))。"""
    bg = rng.uniform(0.80, 0.98)
    canvas = Image.new('L', (IMG, IMG), int(round(bg * 255)))
    n = int(rng.integers(1, 7))
    boxes, labels = [], []
    placed = []
    for _ in range(n):
        use_hand = (len(hand) > 0) and (rng.random() < 0.45)
        pool = hand if use_hand else printed
        tag, d, sp = pool[int(rng.integers(0, len(pool)))]
        h = float(rng.uniform(13.0, 28.0))
        angle = float(rng.uniform(-5.0, 5.0))
        ink = float(rng.uniform(0.72, 1.0))
        # 旋转后外接框（留 1px 余量）→ 收缩中心取值域，保证数字完整落在画面内
        w = h * sp.width / sp.height
        rad = math.radians(abs(angle))
        bw = w * math.cos(rad) + h * math.sin(rad) + 2.0
        bh = w * math.sin(rad) + h * math.cos(rad) + 2.0
        if bw > IMG - 2 or bh > IMG - 2:
            h = min(h, (IMG - 4) * 0.6)
            w = h * sp.width / sp.height
            bw = w * math.cos(rad) + h * math.sin(rad) + 2.0
            bh = w * math.sin(rad) + h * math.cos(rad) + 2.0
        cx = float(rng.uniform(bw / 2.0, IMG - bw / 2.0))
        cy = float(rng.uniform(bh / 2.0, IMG - bh / 2.0))
        if not allow_overlap:
            ok = True
            for b in placed:
                if abs(b[0] - cx) < (b[4] + bw) / 2 and abs(b[1] - cy) < (b[5] + bh) / 2:
                    ok = False
            if not ok:
                continue
        b = _paste_digit(canvas, sp, cx, cy, h, angle, ink)
        boxes.append(b)
        labels.append(d)
        placed.append((cx, cy, 0, 0, bw, bh))
    if not boxes:
        b = _paste_digit(canvas, printed[0][2], 32, 32, 24, 0.0, 1.0)
        boxes.append(b)
        labels.append(printed[0][1])

    # ---- 增广（PIL 阶段：模糊） ----
    blur = float(rng.uniform(0.0, 1.0))
    if blur > 0.15:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur))
    a = np.asarray(canvas).astype(np.float32) / 255.0

    # ---- 增广（numpy 阶段：灰度抖动 / 噪声 / 二值化抖动 / 擦除） ----
    a = a * float(rng.uniform(0.82, 1.10)) + float(rng.uniform(-0.10, 0.08))
    if rng.random() < 0.5:
        sig = float(rng.uniform(0.0, 0.055))
        if sig > 0.002:
            a = a + rng.normal(0.0, sig, a.shape).astype(np.float32)
    if rng.random() < 0.25:
        p = float(rng.uniform(0.0, 0.010))
        m = rng.random(a.shape) < p
        a[m] = 0.0
        m = rng.random(a.shape) < p
        a[m] = 1.0
    if rng.random() < 0.30:
        t = float(rng.uniform(0.42, 0.62))
        a = np.where(a < t, 0.0, 1.0).astype(np.float32) * 0.5 + a * 0.5
    a = np.clip(a, 0.0, 1.0)
    # 随机擦除：填背景色（被遮住的数字仍然保留标签——网络要能忍遮挡）
    if rng.random() < 0.30:
        for _ in range(int(rng.integers(1, 3))):
            ew = int(rng.integers(3, 9))
            eh = int(rng.integers(3, 9))
            ex = int(rng.integers(0, IMG - ew))
            ey = int(rng.integers(0, IMG - eh))
            a[ey:ey + eh, ex:ex + ew] = float(np.clip(rng.uniform(0.85, 1.0), 0, 1))
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    # 包围盒裁剪到画面内，并丢弃退化的框
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, IMG - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, IMG - 1)
    keep = (boxes[:, 2] - boxes[:, 0] >= 4) & (boxes[:, 3] - boxes[:, 1] >= 4)
    return a, boxes[keep], labels[keep]


def make_dataset(seed, n, printed, hand):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, IMG, IMG), dtype=np.float32)
    boxes, labels = [], []
    for i in range(n):
        a, b, l = synth_sample(rng, printed, hand)
        imgs[i] = a
        boxes.append(b)
        labels.append(l)
    return imgs, boxes, labels


# ---------------------------------------------------------------------------
# 锚框与几何工具（Python 侧参照实现，与 Scheme 侧逐条对应）
# ---------------------------------------------------------------------------
def make_anchors():
    """返回 (GRID*GRID*NA, 4) 的锚框 (x1,y1,x2,y2)，行主序：cell(0,0) 的 9 个锚框在前。"""
    out = np.zeros((GRID, GRID, NA, 4), dtype=np.float32)
    k = 0
    for s in ANCH_SIZES:
        for r in ANCH_RATIOS:
            w = s * math.sqrt(r)
            h = s / math.sqrt(r)
            for gy in range(GRID):
                for gx in range(GRID):
                    cx = gx * STRIDE + STRIDE / 2.0
                    cy = gy * STRIDE + STRIDE / 2.0
                    out[gy, gx, k] = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
            k += 1
    return out.reshape(-1, 4)


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) → (N,M) IoU。"""
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1) + 1.0)
    ih = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1) + 1.0)
    inter = iw * ih
    aa = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0)
    bb = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0)
    return inter / np.maximum(1e-6, aa + bb - inter)


def bbox_deltas(boxes, gt):
    """框 → 相对某 GT 的回归目标 (tx,ty,tw,th)。boxes/gt: (N,4)，逐行配对。"""
    bw = boxes[:, 2] - boxes[:, 0] + 1.0
    bh = boxes[:, 3] - boxes[:, 1] + 1.0
    bcx = boxes[:, 0] + bw / 2.0
    bcy = boxes[:, 1] + bh / 2.0
    gw = gt[:, 2] - gt[:, 0] + 1.0
    gh = gt[:, 3] - gt[:, 1] + 1.0
    gcx = gt[:, 0] + gw / 2.0
    gcy = gt[:, 1] + gh / 2.0
    tx = (gcx - bcx) / bw
    ty = (gcy - bcy) / bh
    tw = np.log(np.maximum(1e-6, gw / bw))
    th = np.log(np.maximum(1e-6, gh / bh))
    return np.stack([tx, ty, tw, th], axis=1)


def apply_deltas(boxes, d):
    """回归目标 → 框（d 逐行对应 boxes）。"""
    bw = boxes[:, 2] - boxes[:, 0] + 1.0
    bh = boxes[:, 3] - boxes[:, 1] + 1.0
    bcx = boxes[:, 0] + bw / 2.0
    bcy = boxes[:, 1] + bh / 2.0
    dw = np.clip(d[:, 2], -4.0, 4.0)
    dh = np.clip(d[:, 3], -4.0, 4.0)
    cx = bcx + d[:, 0] * bw
    cy = bcy + d[:, 1] * bh
    w = bw * np.exp(dw)
    h = bh * np.exp(dh)
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)


def nms(boxes, scores, thr):
    """贪心 NMS，返回保留的下标（按分数降序）。"""
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        iou = iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        order = rest[iou < thr]
    return keep


def roi_pool_np(feat, rois, out=POOL, stride=STRIDE):
    """RoI 池化（Caffe 口径，最大池化）——与 Scheme 侧 frcnn_core.scm 的 roi_pool 一致。

    feat: (C,GRID,GRID)；rois: (R,4) 图像坐标 (x1,y1,x2,y2)。
    约定（逐条对应 Scheme）：
        sx = round(x1/stride), sy = round(y1/stride)
        ex = round(x2/stride), ey = round(y2/stride)
        w = max(ex-sx+1, 1), h = max(ey-sy+1, 1)
        每个 bin: x0 = floor(sx + i*w/out), x1 = ceil(sx + (i+1)*w/out)
                  钳到 [0,GRID-1] / 至少 1 格；bin 值 = 该矩形内的最大值
    """
    C = feat.shape[0]
    R = rois.shape[0]
    outp = np.zeros((R, C, out, out), dtype=np.float32)
    for r in range(R):
        sx = int(round(rois[r, 0] / stride))
        sy = int(round(rois[r, 1] / stride))
        ex = int(round(rois[r, 2] / stride))
        ey = int(round(rois[r, 3] / stride))
        sx = min(max(sx, 0), GRID - 1)
        sy = min(max(sy, 0), GRID - 1)
        ex = min(max(ex, 0), GRID - 1)
        ey = min(max(ey, 0), GRID - 1)
        bw = max(ex - sx + 1, 1)
        bh = max(ey - sy + 1, 1)
        for py in range(out):
            y0 = int(math.floor(sy + py * bh / out))
            y1 = int(math.ceil(sy + (py + 1) * bh / out))
            y0 = min(max(y0, 0), GRID - 1)
            y1 = min(max(y1, y0 + 1), GRID)
            for px in range(out):
                x0 = int(math.floor(sx + px * bw / out))
                x1 = int(math.ceil(sx + (px + 1) * bw / out))
                x0 = min(max(x0, 0), GRID - 1)
                x1 = min(max(x1, x0 + 1), GRID)
                outp[r, :, py, px] = feat[:, y0:y1, x0:x1].max(axis=(1, 2))
    return outp


# ---------------------------------------------------------------------------
# numpy 参照推理（与 Scheme 侧 app/lib/frcnn_core.scm 逐条对应）
# ---------------------------------------------------------------------------
def conv3x3_s2(x, w, b):
    """3x3 / 步长 2 / 补边 1 / ReLU。x:(Cin,H,W) → (Cout,H/2,W/2)。

    约定（逐条对应 Scheme 侧）：out[k,r,c] = b[k] + Σ_c' Σ_ky Σ_kx w[k,c',ky,kx]·x[c', 2r+ky-1, 2c+kx-1]
    （越界取 0），随后 ReLU。
    """
    Cin, H, W = x.shape
    Cout = w.shape[0]
    ho, wo = H // 2, W // 2
    xp = np.pad(x.astype(np.float32), ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((Cout, ho, wo), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            patch = xp[:, ky:ky + 2 * ho:2, kx:kx + 2 * wo:2]
            out += np.einsum('oc,chw->ohw', w[:, :, ky, kx].astype(np.float32), patch)
    out += b.astype(np.float32)[:, None, None]
    return np.maximum(out, 0.0)


def conv3x3_s1_dw(x, w, b):
    """深度可分离卷积的 3x3 逐通道部分（步长 1、补边 1）。x:(C,H,W) → (C,H,W)。

    out[c,r,cc] = b[c] + Σ_ky Σ_kx w[c,ky,kx]·x[c, r+ky-1, cc+kx-1]（越界取 0）
    """
    Cc, H, W = x.shape
    xp = np.pad(x.astype(np.float32), ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((Cc, H, W), dtype=np.float32)
    for ky in range(3):
        for kx in range(3):
            out += w[:, ky, kx].reshape(-1, 1, 1) * xp[:, ky:ky + H, kx:kx + W]
    return out + b.reshape(-1, 1, 1)


def conv1x1(x, w, b, relu=True):
    """1x1 卷积。x:(Cin,H*W) → (Cout,H*W)；w:(Cout,Cin)。"""
    out = (w.astype(np.float32) @ x.astype(np.float32)) + b.astype(np.float32)[:, None]
    return np.maximum(out, 0.0) if relu else out


def reference_infer(img, P, detail=False):
    """用 numpy 跑一遍完整两阶段推理。

    img: (64,64) float32 ∈[0,1]（0=黑，1=白）；P: 权重字典（已反量化成 float32，BN 已折叠）。
    网络输入 = ((img/255) - MEAN) / STD —— 训练脚本就是这么喂的（与 app/lib/frcnn_core.scm
    的 u_norm 逐字对应；**这一除不能少**，否则输入对比度大 255 倍，整网输出跑飞）。
    返回 dict：det(list of (cls,score,x1,y1,x2,y2)) 及（detail 时）各阶段中间结果。
    """
    x = (((img.astype(np.float32) / np.float32(255.0)) - np.float32(MEAN))
         / np.float32(STD)).reshape(1, IMG, IMG)

    f1 = conv3x3_s2(x, P['conv1_w'], P['conv1_b'])          # 8x32x32
    f2 = conv3x3_s2(f1, P['conv2_w'], P['conv2_b'])         # 24x16x16
    f3 = conv3x3_s1_dw(f2, P['dw_w'], P['dw_b'])            # 24x16x16（3x3 逐通道）
    f3 = conv1x1(f3.reshape(NCH, GRID * GRID), P['pw_w'], P['pw_b']).reshape(NCH, GRID, GRID)

    # RPN 1x1（把 16x16 格摊成一批列向量）
    flat = f3.reshape(NCH, GRID * GRID)
    h = conv1x1(flat, P['rpn_conv_w'], P['rpn_conv_b'])     # 16x256
    obj_logit = conv1x1(h, P['rpn_cls_w'], P['rpn_cls_b'], relu=False)   # 9x256
    reg = conv1x1(h, P['rpn_reg_w'], P['rpn_reg_b'], relu=False)         # 36x256
    # 通道 c 的 (gy,gx) 值 = 通道 c 的第 gy*GRID+gx 列
    obj_logit = obj_logit.T.reshape(GRID, GRID, NA)
    reg = reg.T.reshape(GRID, GRID, NA * 4)
    h = h.T.reshape(GRID, GRID, NCH)
    obj = 1.0 / (1.0 + np.exp(-obj_logit))

    anchors = make_anchors()
    scores = obj.reshape(-1)
    deltas = reg.reshape(-1, 4)
    boxes = apply_deltas(anchors, deltas)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, IMG - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, IMG - 1)
    ok = (boxes[:, 2] - boxes[:, 0] >= 3) & (boxes[:, 3] - boxes[:, 1] >= 3) & (scores > SCORE_TH)
    idx = np.where(ok)[0]
    if idx.size > PRE_NMS_TOPK:
        idx = idx[np.argsort(-scores[idx])[:PRE_NMS_TOPK]]
    props = boxes[idx]
    pscore = scores[idx]
    keep = nms(props, pscore, RPN_NMS_IOU) if props.size else []
    keep = keep[:NROI]

    rois = np.zeros((NROI, 4), dtype=np.float32)
    rscore = np.zeros(NROI, dtype=np.float32)
    nr = len(keep)
    for i in range(nr):
        rois[i] = props[keep[i]]
        rscore[i] = pscore[keep[i]]

    pooled = np.zeros((max(nr, 1), NCH, POOL, POOL), dtype=np.float32)
    if nr:
        pooled = roi_pool_np(f3, rois[:nr])

    cls_p = np.zeros((max(nr, 1), NCLS), dtype=np.float32)
    reg_h = np.zeros((max(nr, 1), NCLS, 4), dtype=np.float32)
    if nr:
        flat = pooled.reshape(nr, -1)
        fc1 = np.maximum(flat @ P['fc1_w'].T + P['fc1_b'], 0.0)
        logits = fc1 @ P['cls_w'].T + P['cls_b']
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        cls_p[:nr] = e / e.sum(axis=1, keepdims=True)
        reg_h[:nr] = (fc1 @ P['reg_w'].T + P['reg_b']).reshape(nr, NCLS, 4)

    # 逐 RoI × 逐类：任一类概率超阈值就出一个候选（经典 Faster R-CNN 的做法；
    # 只取 argmax 会漏掉「同一 RoI 覆盖两个重叠数字」的情形），再按类别做 NMS
    det = []
    for i in range(nr):
        for c in range(10):
            s = float(cls_p[i, c])
            if s < SCORE_TH:
                continue
            d = reg_h[i, c]
            bb = apply_deltas(rois[i:i + 1], d.reshape(1, 4))[0]
            bb[0::2] = np.clip(bb[0::2], 0, IMG - 1)
            bb[1::2] = np.clip(bb[1::2], 0, IMG - 1)
            det.append((c, s, float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])))
    per_class = []
    for c in range(10):
        sel = [d for d in det if d[0] == c]
        if not sel:
            continue
        bb = np.array([[d[2], d[3], d[4], d[5]] for d in sel], dtype=np.float32)
        sc = np.array([d[1] for d in sel], dtype=np.float32)
        for i in nms(bb, sc, FINAL_NMS_IOU):
            per_class.append(sel[i])
    # 注：**不做**跨类去重——实测会掉 mAP（0.918 → 0.706）：本数据里数字允许重叠，
    # 跨类 NMS 会把相邻数字当重复框吃掉。
    final = sorted(per_class, key=lambda d: -d[1])

    if not detail:
        return final
    return {
        'xn': x, 'f1': f1, 'f2': f2, 'f3': f3, 'rpn_h': h, 'obj': obj, 'reg': reg,
        'anchors': anchors, 'proposals': (props, pscore), 'rois': (rois[:nr], rscore[:nr]),
        'pooled': pooled, 'cls_p': cls_p[:nr], 'reg_h': reg_h[:nr], 'det': final,
    }
