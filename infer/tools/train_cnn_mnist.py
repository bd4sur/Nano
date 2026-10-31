#!/usr/bin/env python3
# ============================================================================
# train_cnn_mnist.py —— 训练「灵机」CNN 手写数字识别 App（app/cnn.scm）所用的
#                       微型卷积网络，并把权重导出为 Scheme 数据模块。
#
# 只依赖 numpy（本机无 torch）。网络刻意做小，因为 Scheme 解释器每帧只有
# 31.25 万条 VM 指令、一次乘法累加约 1 微秒（实测，见 doc/AGENTS.md）：
#
#   输入   1 x 28 x 28  （像素 / 255）
#   conv1  6 个 5x5 卷积核，步长 2，无填充 → 6 x 12 x 12，ReLU
#   pool1  2x2 最大池化，步长 2            → 6 x  6 x 6
#   conv2  8 个 3x3x6 卷积核，步长 1        → 8 x  4 x 4，ReLU
#   fc     128 → 10（logits，softmax 交叉熵）
#
#   一次推理 MAC 数：6*12*12*25 + 8*4*4*54 + 128*10 = 29,792  ≈ 3.6 帧（wasm 上 60~65ms）
#
# 用法：
#   python3 tools/train_cnn_mnist.py --data /tmp/mnist --out app/lib/cnn_weights.scm
#   （MNIST idx 文件可从 https://storage.googleapis.com/cvdf-datasets/mnist/ 取得）
# ============================================================================
import argparse
import gzip
import struct

import numpy as np

# ---------------------------------------------------------------- MNIST 读取


def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(n * h * w)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, 1, h, w)


def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


# ---------------------------------------------------------------- 算子（前向/反向）
# 全部用朴素的 im2col，numpy 足够快（几秒一个 epoch）。


def im2col_stride(x, kh, kw, stride):
    """x: (N, C, H, W) → (N, C*kh*kw, OH*OW)"""
    N, C, H, W = x.shape
    OH = (H - kh) // stride + 1
    OW = (W - kw) // stride + 1
    cols = np.empty((N, C, kh, kw, OH, OW), dtype=x.dtype)
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, ky, kx, :, :] = x[:, :, ky:ky + stride * OH:stride,
                                          kx:kx + stride * OW:stride]
    return cols.reshape(N, C * kh * kw, OH * OW), OH, OW


def conv_forward(x, w, b, stride):
    """w: (F, C, kh, kw) → (N, F, OH, OW)"""
    N, C, H, W = x.shape
    F, _, kh, kw = w.shape
    cols, OH, OW = im2col_stride(x, kh, kw, stride)
    out = np.einsum("fo,nop->nfp", w.reshape(F, -1), cols, optimize=True)
    out += b[None, :, None]
    return out.reshape(N, F, OH, OW), cols, (OH, OW)


def conv_backward(dout, cols, xshape, w, stride, OH, OW):
    N, C, H, W = xshape
    F, _, kh, kw = w.shape
    dflat = dout.reshape(N, F, OH * OW)
    dw = np.einsum("nfp,nop->fo", dflat, cols, optimize=True).reshape(w.shape)
    db = dout.sum(axis=(0, 2, 3))
    dcols = np.einsum("fo,nfp->nop", w.reshape(F, -1), dflat, optimize=True)
    dcols = dcols.reshape(N, C, kh, kw, OH, OW)
    dx = np.zeros((N, C, H, W), dtype=dout.dtype)
    for ky in range(kh):
        for kx in range(kw):
            dx[:, :, ky:ky + stride * OH:stride, kx:kx + stride * OW:stride] += dcols[:, :, ky, kx]
    return dx, dw, db


def maxpool_forward(x):
    N, C, H, W = x.shape
    OH, OW = H // 2, W // 2
    v = x[:, :, :OH * 2, :OW * 2].reshape(N, C, OH, 2, OW, 2)
    v = v.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, OH, OW, 4)
    idx = v.argmax(axis=-1)
    out = np.take_along_axis(v, idx[..., None], axis=-1)[..., 0]
    return out, (idx, x.shape)


def maxpool_backward(dout, cache):
    idx, xshape = cache
    N, C, H, W = xshape
    OH, OW = H // 2, W // 2
    dx = np.zeros(xshape, dtype=dout.dtype)
    dv = np.zeros((N, C, OH, OW, 4), dtype=dout.dtype)
    np.put_along_axis(dv, idx[..., None], dout[..., None], axis=-1)
    dv = dv.reshape(N, C, OH, OW, 2, 2).transpose(0, 1, 2, 4, 3, 5)
    dx[:, :, :OH * 2, :OW * 2] = dv.reshape(N, C, OH * 2, OW * 2)
    return dx


# ---------------------------------------------------------------- 网络


class Net:
    def __init__(self, seed=1234):
        rng = np.random.default_rng(seed)
        # He 初始化
        self.w1 = (rng.standard_normal((6, 1, 5, 5)) * np.sqrt(2.0 / 25)).astype(np.float32)
        self.b1 = np.zeros(6, dtype=np.float32)
        self.w2 = (rng.standard_normal((8, 6, 3, 3)) * np.sqrt(2.0 / 54)).astype(np.float32)
        self.b2 = np.zeros(8, dtype=np.float32)
        self.w3 = (rng.standard_normal((10, 128)) * np.sqrt(2.0 / 128)).astype(np.float32)
        self.b3 = np.zeros(10, dtype=np.float32)

    def params(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def forward(self, x):
        c1, cols1, (oh1, ow1) = conv_forward(x, self.w1, self.b1, 2)
        r1 = np.maximum(c1, 0)
        p1, c1cache = maxpool_forward(r1)
        c2, cols2, (oh2, ow2) = conv_forward(p1, self.w2, self.b2, 1)
        r2 = np.maximum(c2, 0)
        flat = r2.reshape(r2.shape[0], -1)
        logits = flat @ self.w3.T + self.b3
        cache = (cols1, x.shape, (oh1, ow1), c1, c1cache, cols2, p1.shape, (oh2, ow2), c1, c2, flat)
        return logits, cache

    def backward(self, dlogits, cache):
        (cols1, xshape, (oh1, ow1), c1, c1cache, cols2, p1shape, c2shape, _c1,
         c2, flat) = cache
        dw3 = dlogits.T @ flat
        db3 = dlogits.sum(axis=0)
        dflat = dlogits @ self.w3
        dr2 = dflat.reshape(c2.shape)
        dc2 = dr2 * (c2 > 0)
        dp1, dw2, db2 = conv_backward(dc2, cols2, p1shape, self.w2, 1,
                                      c2shape[0], c2shape[1])
        dr1 = maxpool_backward(dp1, c1cache)
        dc1 = dr1 * (c1 > 0)
        dx, dw1, db1 = conv_backward(dc1, cols1, xshape, self.w1, 2, oh1, ow1)
        return [dw1, db1, dw2, db2, dw3, db3]


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n, p


# ---------------------------------------------------------------- 导出


def fmt(v):
    s = "%.6g" % float(v)
    if s == "-0":
        s = "0"
    if "e" in s:
        s = ("%.6f" % float(v)).rstrip("0")
        if s.endswith("."):
            s += "0"
    if "." not in s:
        s += ".0"
    return s


def flat_list(arr, per_line=16, indent="  "):
    vals = [fmt(v) for v in np.asarray(arr).ravel()]
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(indent + " ".join(vals[i:i + per_line]))
    # 注意：灵机没有内建 list 过程，权重一律以 **quote 列表字面量** 导出
    return "'(\n" + "\n".join(lines) + ")"


def export(net, path, info):
    with open(path, "w") as f:
        f.write(";; cnn_weights.scm —— CNN 手写数字识别网络权重（**自动生成，勿手改**）\n")
        f.write(";;\n")
        f.write(";; 生成脚本：infer/tools/train_cnn_mnist.py\n")
        for k, v in info.items():
            f.write(";; %s：%s\n" % (k, v))
        f.write(";;\n")
        f.write(";; 网络结构（NCHW，行主序展平）：\n")
        f.write(";;   conv1 6x1x5x5 步长2  -> ReLU -> 6x12x12\n")
        f.write(";;   pool1 2x2 最大池化     -> 6x6x6\n")
        f.write(";;   conv2 8x6x3x3 步长1    -> ReLU -> 8x4x4\n")
        f.write(";;   fc    128 -> 10\n")
        f.write(";;\n")
        f.write(";; 数值精度：wasm32 上灵机浮点为单精度且尾数被截 5 位（相对精度 ~4e-6），\n")
        f.write(";; 故权重只导出 6 位有效数字。\n\n")
        f.write(";; conv1：6 个 5x5 卷积核（每 25 个一组）\n")
        f.write("(define W1 %s)\n\n" % flat_list(net.w1))
        f.write(";; conv1 偏置\n")
        f.write("(define B1 %s)\n\n" % flat_list(net.b1))
        f.write(";; conv2：8 个 6x3x3 卷积核（每 54 个一组，通道序为主序）\n")
        f.write("(define W2 %s)\n\n" % flat_list(net.w2))
        f.write(";; conv2 偏置\n")
        f.write("(define B2 %s)\n\n" % flat_list(net.b2))
        f.write(";; fc：10 x 128（每 128 个一组）\n")
        f.write("(define W3 %s)\n\n" % flat_list(net.w3, per_line=16))
        f.write(";; fc 偏置\n")
        f.write("(define B3 %s)\n" % flat_list(net.b3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/mnist")
    ap.add_argument("--out", default="app/lib/cnn_weights.scm")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    xtr = load_idx_images(args.data + "/train-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    ytr = load_idx_labels(args.data + "/train-labels-idx1-ubyte.gz")
    xte = load_idx_images(args.data + "/t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    yte = load_idx_labels(args.data + "/t10k-labels-idx1-ubyte.gz")
    print("train", xtr.shape, "test", xte.shape)

    net = Net(args.seed)
    params = net.params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = xtr.shape[0]
    rng = np.random.default_rng(args.seed + 1)
    for ep in range(args.epochs):
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = order[i:i + args.batch]
            xb, yb = xtr[idx], ytr[idx]
            logits, cache = net.forward(xb)
            loss, dlogits, _ = softmax_ce(logits, yb)
            grads = net.backward(dlogits, cache)
            step += 1
            for j, (p, g) in enumerate(zip(params, grads)):
                m[j] = beta1 * m[j] + (1 - beta1) * g
                v[j] = beta2 * v[j] + (1 - beta2) * (g * g)
                mh = m[j] / (1 - beta1 ** step)
                vh = v[j] / (1 - beta2 ** step)
                p -= (args.lr * mh / (np.sqrt(vh) + eps)).astype(p.dtype)
            tot += loss * len(idx)
            if (i // args.batch) % 200 == 0:
                print("  ep%d it%4d loss=%.4f" % (ep, i // args.batch, loss), flush=True)
        # 评估
        correct = 0
        for i in range(0, xte.shape[0], 500):
            logits, _ = net.forward(xte[i:i + 500])
            correct += int((logits.argmax(axis=1) == yte[i:i + 500]).sum())
        acc = correct / xte.shape[0]
        print("epoch %d: train_loss=%.4f test_acc=%.4f" % (ep, tot / n, acc), flush=True)

    info = {
        "训练数据": "MNIST（60000 训练 / 10000 测试）",
        "测试集准确率": "%.2f%%" % (100.0 * acc),
        "训练轮数": str(args.epochs),
    }
    export(net, args.out, info)
    print("wrote", args.out)


if __name__ == "__main__":
    main()