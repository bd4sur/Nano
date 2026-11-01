"""train_hwr_cnn.py —— 手写汉字识别 v3：小 CNN（conv+池化 → 128 维嵌入）+ 最近原型（NCM）

为什么换 CNN：手工特征的 top-1 卡在 58.5%（3755 类），而 CASIA 官方 512 维特征 + 同一个
最近类均值能到 81.5%；两者特征空间近乎线性无关（R²≈0.13），说明差距在"方向算子/聚合方式"。
本脚本改为**端到端学特征**，并把推理规模压在灵机能跑的范围内。

网络（推理规模 ≈ 58 万乘加，WASM 约 0.7 s）：
    输入 32×32（= 64×64 归一化墨迹 2×2 均值池化后的结果）
    conv1  3×3 p1 s2  1→8    → 16×16     18.4k MAC     + BN + ReLU
    maxpool 2×2 s2           → 8×8×8
    conv2  3×3 p1 s1  8→16   → 8×8       55.3k MAC     + BN + ReLU
    avgpool 2×2 s2           → 4×4×16 = 256
    fc     256→128（L2 归一化）            32.8k MAC
    余弦分类器：128 × 3755 原型（= 最近类均值）  480k MAC
BN 折进卷积权重（scale/shift），推理零额外开销。

训练数据：CASIA-OLHWDB1.1 原始笔迹（`.pot`），输入侧做与 App 一致的"仿真"
（64 格量化 + 2.5 格抽稀）→ 与 app/hwr2.scm 完全相同的墨迹渲染 + 伪二维归一化。

用法：
    python3 tools/train_hwr_cnn.py                 # 建缓存 → 训练 → 评测 → 导出
    python3 tools/train_hwr_cnn.py --epochs 24
"""
import argparse, glob, os, struct, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_hwr_online as T          # 复用 .pot 解析 / 渲染 / 伪二维归一化

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), 'app', 'lib', 'hwr3')
CACHE = '/tmp/hwr3_imgs.npy'

IN = 32            # CNN 输入边长
SCALE = 16.0       # 余弦分类器温度


# ---------------------------------------------------------------- 数据
def _img_worker(st):
    """一条笔迹 → 32×32 输入（与 App 侧完全一致的预处理链）

    重采样步长**随机 3~8 格**：真机手指输入每 20ms 采一个点，笔画点距约 5~10 格，
    比 2.5 格密采样明显"折线化"（弦代替弧，墨迹更短）。训练必须覆盖这个范围。"""
    rng = np.random.default_rng(abs(hash(st[0].tobytes())) % (2 ** 31))
    st = T.simulate_app(st, resample=float(rng.uniform(3.0, 8.0)))
    img = T.render_thin(st, ng=64)                # 细线墨迹 64×64
    if img is None or img.sum() <= 0:
        return None
    img = T.normalize_p2d(img * 255.0, alpha=0.7)  # 伪二维归一化
    m = img.max()
    if m > 0:
        img = img / m
    v = img.reshape(32, 2, 32, 2).mean(axis=(1, 3))   # 2×2 均值池化 → 32×32
    return (np.clip(v, 0, 1) * 255).astype(np.uint8)


def build_cache(train_dir, test_dir, writers, stride, procs=14):
    from multiprocessing import Pool
    Str, Ltr, Ste, Lte = T.load_split(train_dir, test_dir, writers or None, stride)
    print(f'  训练样本 {len(Str)}  测试样本 {len(Ste)}', flush=True)
    t0 = time.time()
    out = {}
    with Pool(procs) as p:
        for tag, S, L in (('tr', Str, Ltr), ('te', Ste, Lte)):
            F = p.map(_img_worker, S, chunksize=200)
            keep = np.array([i for i, f in enumerate(F) if f is not None])
            arr = np.stack([F[i] for i in keep]).reshape(len(keep), IN * IN)
            out[tag + '_x'] = arr
            out[tag + '_y'] = L[keep]
            print(f'  {tag}: {arr.shape} ({time.time()-t0:.0f}s)', flush=True)
    np.savez(CACHE + '.npz', **out)
    print('  缓存 →', CACHE + '.npz')


# ---------------------------------------------------------------- 网络（numpy）
def im2col_pad(x, kh, kw, stride):
    """x: (N,C,H,W) 已 pad 过 → (N, C*kh*kw, OH*OW), OH, OW"""
    N, C, H, W = x.shape
    OH = (H - kh) // stride + 1
    OW = (W - kw) // stride + 1
    cols = np.empty((N, C, kh, kw, OH, OW), dtype=x.dtype)
    for ky in range(kh):
        for kx in range(kw):
            cols[:, :, ky, kx] = x[:, :, ky:ky + stride * OH:stride, kx:kx + stride * OW:stride]
    return cols.reshape(N, C * kh * kw, OH * OW), OH, OW


def conv_fwd(x, w, b, stride, pad=1):
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    cols, OH, OW = im2col_pad(xp, w.shape[2], w.shape[3], stride)
    F = w.shape[0]
    out = np.einsum('fo,nop->nfp', w.reshape(F, -1), cols, optimize=True)
    out += b[None, :, None]
    return out.reshape(x.shape[0], F, OH, OW), cols, xp.shape


def conv_bwd(dout, cols, xpshape, w, stride, pad=1):
    N, C, H, W = xpshape
    F, _, kh, kw = w.shape
    dflat = dout.reshape(dout.shape[0], F, -1)
    dw = np.einsum('nfp,nop->fo', dflat, cols, optimize=True).reshape(w.shape)
    db = dout.sum(axis=(0, 2, 3))
    dcols = np.einsum('fo,nfp->nop', w.reshape(F, -1), dflat, optimize=True)
    dcols = dcols.reshape(dout.shape[0], C, kh, kw, dout.shape[2], dout.shape[3])
    dxp = np.zeros(xpshape, dtype=dout.dtype)
    for ky in range(kh):
        for kx in range(kw):
            dxp[:, :, ky:ky + stride * dout.shape[2]:stride,
                kx:kx + stride * dout.shape[3]:stride] += dcols[:, :, ky, kx]
    dx = dxp[:, :, pad:pad + H - 2 * pad, pad:pad + W - 2 * pad] if pad else dxp
    return dx, dw, db


def maxpool_fwd(x):
    N, C, H, W = x.shape
    OH, OW = H // 2, W // 2
    v = x[:, :, :OH * 2, :OW * 2].reshape(N, C, OH, 2, OW, 2).transpose(0, 1, 2, 4, 3, 5)
    v = v.reshape(N, C, OH, OW, 4)
    idx = v.argmax(-1)
    out = np.take_along_axis(v, idx[..., None], -1)[..., 0]
    return out, (idx, x.shape)


def maxpool_bwd(dout, cache):
    idx, xshape = cache
    N, C, H, W = xshape
    OH, OW = H // 2, W // 2
    dv = np.zeros((N, C, OH, OW, 4), dtype=dout.dtype)
    np.put_along_axis(dv, idx[..., None], dout[..., None], -1)
    dv = dv.reshape(N, C, OH, OW, 2, 2).transpose(0, 1, 2, 4, 3, 5)
    dx = np.zeros(xshape, dtype=dout.dtype)
    dx[:, :, :OH * 2, :OW * 2] = dv.reshape(N, C, OH * 2, OW * 2)
    return dx


def avgpool_fwd(x):
    N, C, H, W = x.shape
    OH, OW = H // 2, W // 2
    v = x[:, :, :OH * 2, :OW * 2].reshape(N, C, OH, 2, OW, 2)
    return v.mean(axis=(3, 5)), (x.shape, OH, OW)


def avgpool_bwd(dout, cache):
    xshape, OH, OW = cache
    dx = np.zeros(xshape, dtype=dout.dtype)
    dx[:, :, :OH * 2, :OW * 2] = np.repeat(np.repeat(dout / 4.0, 2, axis=2), 2, axis=3)
    return dx


class Net:
    """conv1(1→C1,s2) - BN - ReLU - maxpool - conv2(C1→C2) - BN - ReLU - avgpool - fc(L2) - 余弦原型"""

    def __init__(self, C1, C2, NRED, K, seed=1234):
        rng = np.random.default_rng(seed)
        self.C1, self.C2, self.NRED, self.K = C1, C2, NRED, K
        self.w1 = (rng.standard_normal((C1, 1, 3, 3)) * np.sqrt(2.0 / 9)).astype(np.float32)
        self.b1 = np.zeros(C1, np.float32)
        self.g1 = np.ones(C1, np.float32); self.be1 = np.zeros(C1, np.float32)
        self.w2 = (rng.standard_normal((C2, C1, 3, 3)) * np.sqrt(2.0 / (9 * C1))).astype(np.float32)
        self.b2 = np.zeros(C2, np.float32)
        self.g2 = np.ones(C2, np.float32); self.be2 = np.zeros(C2, np.float32)
        self.wf = (rng.standard_normal((NRED, 4 * 4 * C2)) * np.sqrt(2.0 / (4 * 4 * C2))).astype(np.float32)
        self.bf = np.zeros(NRED, np.float32)
        self.prot = (rng.standard_normal((K, NRED)) * 0.01).astype(np.float32)
        # 占位：训练期间验证用批次统计量，训练结束后由 bn_stats() 填总体统计量
        self.mu1 = np.zeros(C1, np.float32); self.va1 = np.ones(C1, np.float32)
        self.mu2 = np.zeros(C2, np.float32); self.va2 = np.ones(C2, np.float32)

    def params(self):
        return [self.w1, self.b1, self.g1, self.be1, self.w2, self.b2, self.g2, self.be2,
                self.wf, self.bf, self.prot]

    def bn_stats(self, X, bs=512):
        """在训练集上统计 BN 的总体均值/方差（导出时折进卷积）"""
        n = 0; s1 = np.zeros(self.C1); q1 = np.zeros(self.C1)
        s2 = np.zeros(self.C2); q2 = np.zeros(self.C2)
        for i in range(0, len(X), bs):
            c1, _, _ = conv_fwd(X[i:i+bs], self.w1, self.b1, 2, 1)
            mu = c1.mean(axis=(0, 2, 3)); va = c1.var(axis=(0, 2, 3))
            m = X[i:i+bs].shape[0]
            s1 += mu * m; q1 += (va + mu ** 2) * m
            n += m
        self.mu1 = s1 / n; self.va1 = np.maximum(q1 / n - self.mu1 ** 2, 1e-8)
        n = 0
        for i in range(0, len(X), bs):
            xb = X[i:i+bs]
            c1, _, _ = conv_fwd(xb, self.w1, self.b1, 2, 1)
            xh1 = (c1 - self.mu1[None, :, None, None]) / np.sqrt(self.va1[None, :, None, None] + 1e-5)
            n1 = xh1 * self.g1[None, :, None, None] + self.be1[None, :, None, None]
            r1 = np.maximum(n1, 0)
            p1, _ = maxpool_fwd(r1)
            c2, _, _ = conv_fwd(p1, self.w2, self.b2, 1, 1)
            mu = c2.mean(axis=(0, 2, 3)); va = c2.var(axis=(0, 2, 3))
            m = xb.shape[0]
            s2 += mu * m; q2 += (va + mu ** 2) * m
            n += m
        self.mu2 = s2 / n; self.va2 = np.maximum(q2 / n - self.mu2 ** 2, 1e-8)
        print(f'  BN 总体统计量就绪（mu1[0]={self.mu1[0]:.4f} va1[0]={self.va1[0]:.4f}）', flush=True)

    def forward(self, x, train=True):
        c1, cols1, xs1 = conv_fwd(x, self.w1, self.b1, 2, 1)          # (N,C1,16,16)
        N = x.shape[0]
        if train:
            mu1 = c1.mean(axis=(0, 2, 3))
            va1 = c1.var(axis=(0, 2, 3))
        else:
            mu1, va1 = self.mu1, self.va1
        xh1 = (c1 - mu1[None, :, None, None]) / np.sqrt(va1[None, :, None, None] + 1e-5)
        n1 = xh1 * self.g1[None, :, None, None] + self.be1[None, :, None, None]
        r1 = np.maximum(n1, 0)
        p1, mp1 = maxpool_fwd(r1)                                     # (N,C1,8,8)
        c2, cols2, xs2 = conv_fwd(p1, self.w2, self.b2, 1, 1)         # (N,C2,8,8)
        if train:
            mu2 = c2.mean(axis=(0, 2, 3)); va2 = c2.var(axis=(0, 2, 3))
        else:
            mu2, va2 = self.mu2, self.va2
        xh2 = (c2 - mu2[None, :, None, None]) / np.sqrt(va2[None, :, None, None] + 1e-5)
        n2 = xh2 * self.g2[None, :, None, None] + self.be2[None, :, None, None]
        r2 = np.maximum(n2, 0)
        p2, ap2 = avgpool_fwd(r2)                                     # (N,C2,4,4)
        flat = p2.reshape(N, -1)
        z = flat @ self.wf.T + self.bf
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)
        pn = self.prot / (np.linalg.norm(self.prot, axis=1, keepdims=True) + 1e-6)
        logits = SCALE * (zn @ pn.T)
        cache = dict(x=x, c1=c1, cols1=cols1, xs1=xs1, mu1=mu1, va1=va1, xh1=xh1, n1=n1, r1=r1,
                     mp1=mp1, p1=p1, c2=c2, cols2=cols2, xs2=xs2, mu2=mu2, va2=va2, xh2=xh2,
                     n2=n2, r2=r2, p2=p2, ap2=ap2, flat=flat, z=z, zn=zn, pn=pn)
        return logits, cache

    def backward(self, dlogits, cache):
        N = dlogits.shape[0]
        # 余弦分类器
        dzn = SCALE * (dlogits @ cache['pn'])
        dpn = SCALE * (dlogits.T @ cache['zn'])
        dprot = (dpn - cache['pn'] * (dpn * cache['pn']).sum(1, keepdims=True)) / \
                (np.linalg.norm(self.prot, axis=1, keepdims=True) + 1e-6)
        dz = (dzn - cache['zn'] * (dzn * cache['zn']).sum(1, keepdims=True)) / \
             (np.linalg.norm(cache['z'], axis=1, keepdims=True) + 1e-6)
        dwf = dz.T @ cache['flat']; dbf = dz.sum(0)
        dflat = dz @ self.wf
        dp2 = dflat.reshape(cache['p2'].shape)
        dr2 = avgpool_bwd(dp2, cache['ap2'])
        dn2 = dr2 * (cache['r2'] > 0)
        c = cache
        dg2 = (dn2 * c['xh2']).sum(axis=(0, 2, 3)); dbe2 = dn2.sum(axis=(0, 2, 3))
        dxh2 = dn2 * self.g2[None, :, None, None]
        k = c['c2'].shape[0] * c['c2'].shape[2] * c['c2'].shape[3]
        dmu2 = (-dxh2 / np.sqrt(c['va2'][None, :, None, None] + 1e-5)).sum(axis=(0, 2, 3))
        dva2 = (dxh2 * (c['c2'] - c['mu2'][None, :, None, None]) *
                -(0.5) * (c['va2'][None, :, None, None] + 1e-5) ** -1.5).sum(axis=(0, 2, 3))
        dc2 = (dxh2 / np.sqrt(c['va2'][None, :, None, None] + 1e-5)
               + dmu2[None, :, None, None] / k
               + dva2[None, :, None, None] * 2 * (c['c2'] - c['mu2'][None, :, None, None]) / k)
        dp1, dw2, db2 = conv_bwd(dc2, c['cols2'], c['xs2'], self.w2, 1)
        dr1 = maxpool_bwd(dp1, c['mp1'])
        dn1 = dr1 * (c['r1'] > 0)
        dg1 = (dn1 * c['xh1']).sum(axis=(0, 2, 3)); dbe1 = dn1.sum(axis=(0, 2, 3))
        dxh1 = dn1 * self.g1[None, :, None, None]
        k1 = c['c1'].shape[0] * c['c1'].shape[2] * c['c1'].shape[3]
        dmu1 = (-dxh1 / np.sqrt(c['va1'][None, :, None, None] + 1e-5)).sum(axis=(0, 2, 3))
        dva1 = (dxh1 * (c['c1'] - c['mu1'][None, :, None, None]) *
                -(0.5) * (c['va1'][None, :, None, None] + 1e-5) ** -1.5).sum(axis=(0, 2, 3))
        dc1 = (dxh1 / np.sqrt(c['va1'][None, :, None, None] + 1e-5)
               + dmu1[None, :, None, None] / k1
               + dva1[None, :, None, None] * 2 * (c['c1'] - c['mu1'][None, :, None, None]) / k1)
        _, dw1, db1 = conv_bwd(dc1, c['cols1'], c['xs1'], self.w1, 2)
        return [dw1, db1, dg1, dbe1, dw2, db2, dg2, dbe2, dwf, dbf, dprot]


def evaluate(net, X, Y, bs=512, train=False):
    ok = 0
    for i in range(0, len(X), bs):
        logits, _ = net.forward(X[i:i + bs], train=train)
        ok += int((logits.argmax(1) == Y[i:i + bs]).sum())
    return ok / len(X)


def train(net, X, Y, Xv, Yv, epochs, lr, bs, seed=0):
    rng = np.random.default_rng(seed)
    ps = net.params()
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    t = 0
    best = 0.0
    for ep in range(epochs):
        idx = rng.permutation(len(X)); tot = 0.0; cnt = 0
        t0 = time.time()
        lr_ep = lr * (0.5 ** (ep / max(1, epochs // 3)))
        for s in range(0, len(X), bs):
            j = idx[s:s + bs]
            logits, cache = net.forward(X[j])
            y = Y[j]
            z = logits - logits.max(1, keepdims=True)
            e = np.exp(z); p = e / e.sum(1, keepdims=True)
            tot += -np.log(p[np.arange(len(y)), y] + 1e-12).sum(); cnt += len(y)
            g = p; g[np.arange(len(y)), y] -= 1.0; g /= len(y)
            grads = net.backward(g.astype(np.float32), cache)
            t += 1
            for pm, gm, mm, vv in zip(ps, grads, ms, vs):
                mm *= 0.9; mm += 0.1 * gm
                vv *= 0.999; vv += 0.001 * gm * gm
                pm -= lr_ep * (mm / (1 - 0.9 ** t)) / (np.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
        va = evaluate(net, Xv, Yv, train=True)   # 训练期用批次统计量（导出时才折进总体量）
        best = max(best, va)
        print(f'  ep{ep+1:2d} loss={tot/cnt:.3f} val={va:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return best


# ---------------------------------------------------------------- 导出
def export(net, chars, outdir, PER=400):
    os.makedirs(outdir, exist_ok=True)
    C1, C2, NRED, K = net.C1, net.C2, net.NRED, net.K
    # BN 折进卷积
    s1 = net.g1 / np.sqrt(net.va1 + 1e-5)
    w1 = (net.w1 * s1[:, None, None, None]).astype(np.float64)
    b1 = (net.b1 - net.mu1) * s1 + net.be1
    s2 = net.g2 / np.sqrt(net.va2 + 1e-5)
    w2 = (net.w2 * s2[:, None, None, None]).astype(np.float64)
    b2 = (net.b2 - net.mu2) * s2 + net.be2
    wf = net.wf.astype(np.float64); bf = net.bf.astype(np.float64)
    pn = net.prot / (np.linalg.norm(net.prot, axis=1, keepdims=True) + 1e-6)

    def q(t, per_row=False):
        if per_row:
            sc = np.maximum(np.abs(t).max(1, keepdims=True) / 127.0, 1e-9)
        else:
            sc = np.array([[np.abs(t).max() / 127.0 or 1e-9]])
        return np.clip(np.round(t / sc) + 128, 0, 255).astype(np.int32), sc

    def write(name, arr):
        a = arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)
        k = 0
        for lo in range(0, a.shape[0], PER):
            chunk = a[lo:lo + PER]
            open(os.path.join(outdir, f'{name}{k}.txt'), 'w', encoding='utf-8', newline='').write(
                ''.join(chr(int(v) + 256) for v in chunk.reshape(-1)))
            k += 1
        return k

    qw1, sw1 = q(w1); qb1, sb1 = q(b1)
    qw2, sw2 = q(w2); qb2, sb2 = q(b2)
    qwf, swf = q(wf, per_row=True); qbf, sbf = q(bf)
    qpn, spn = q(pn, per_row=True)
    files = {
        'w1': write('c1', qw1.reshape(C1, 9)),             # C1 行，每行 9（3×3 核）
        'w2': write('c2', qw2.reshape(C2, C1 * 9)),        # C2 行，每行 C1*9
        'wf': write('fc', qwf),                            # NRED×256
        'pn': write('pn', qpn),                            # K×NRED
    }
    # 标量参数 + 每行尺度 + 汉字表：params.txt
    def enc16(v, vmax):
        return np.clip(np.round(np.asarray(v) / (vmax or 1.0) * 65535.0), 0, 65535).astype(np.int32)
    swf_max = float(swf.max()); sbf_max = float(np.abs(sbf).max() or 1.0)
    spn_max = float(spn.max())
    txt = []
    txt.append(''.join(chr(int(v) + 256) for v in qb1.reshape(-1)))          # b1 (C1)
    txt.append(''.join(chr(int(v) + 256) for v in qb2.reshape(-1)))          # b2 (C2)
    txt.append(''.join(chr(int(v) + 256) for v in qbf.reshape(-1)))          # bf (NRED)
    e1 = enc16(swf.reshape(-1), swf_max).reshape(-1)
    txt.append(''.join(chr(int(v >> 8) + 256) + chr(int(v & 255) + 256) for v in e1))
    e2 = enc16(spn, spn_max).reshape(-1)
    txt.append(''.join(chr(int(v >> 8) + 256) + chr(int(v & 255) + 256) for v in e2))
    txt.append(''.join(chars))                                             # 汉字表
    open(os.path.join(outdir, 'params.txt'), 'w', encoding='utf-8', newline='').write(''.join(txt))
    meta = (f'(define ncls {K})\n(define nC1 {C1})\n(define nC2 {C2})\n(define nred {NRED})\n'
            f'(define nc1files {files["w1"]})\n(define nc2files {files["w2"]})\n'
            f'(define nfcfiles {files["wf"]})\n(define npnfiles {files["pn"]})\n'
            f'(define per_rows {PER})\n(define sw1 {sw1.ravel()[0]:.9g})\n'
            f'(define sb1 {sb1.ravel()[0]:.9g})\n(define sw2 {sw2.ravel()[0]:.9g})\n'
            f'(define sb2 {sb2.ravel()[0]:.9g})\n(define swfbase {swf_max:.9g})\n'
            f'(define sbf {sbf_max:.9g})\n(define spnbase {spn_max:.9g})\n'
            f'(define scale {SCALE:.9g})\n')
    open(os.path.join(outdir, 'meta.scm'), 'w').write(
        ';; 由 tools/train_hwr_cnn.py 生成（勿手改）：小 CNN 手写识别器元信息\n' + meta)
    tot = sum(os.path.getsize(os.path.join(outdir, f)) for f in os.listdir(outdir))
    print(f'  导出 → {outdir}（{tot//1024} KB） 文件: {files}')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', default='/tmp/potr')
    ap.add_argument('--test-dir', default='/tmp/pot')
    ap.add_argument('--writers', type=int, default=0)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--c1', type=int, default=8)
    ap.add_argument('--c2', type=int, default=16)
    ap.add_argument('--nred', type=int, default=128)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()

    cache_path = CACHE + '.npz'
    if args.rebuild or not os.path.exists(cache_path):
        print('== 构建输入缓存（App 一致的预处理）')
        build_cache(args.train_dir, args.test_dir, args.writers, args.stride)
    d = np.load(cache_path)
    X = (d['tr_x'].astype(np.float32) / 255.0).reshape(-1, 1, IN, IN)
    Y = d['tr_y']
    Xt = (d['te_x'].astype(np.float32) / 255.0).reshape(-1, 1, IN, IN)
    Yt = d['te_y']
    uc = np.unique(Y); idx = {v: i for i, v in enumerate(uc)}
    y = np.array([idx[v] for v in Y])
    m = np.array([v in idx for v in Yt]); Xt = Xt[m]; Yt = Yt[m]
    yt = np.array([idx[v] for v in Yt])
    K = len(uc)
    g2u = T.gb_to_unicode_table()
    chars = [g2u.get(int(v), '\u25a1') for v in uc]
    print(f'== 训练 CNN：{K} 类, train {len(X)}, test {len(Xt)}, '
          f'C1={args.c1} C2={args.c2} nred={args.nred}', flush=True)
    net = Net(args.c1, args.c2, args.nred, K)
    npar = sum(p.size for p in net.params())
    print(f'   参数量 {npar}（其中原型 {K*args.nred}）', flush=True)
    nv = min(30000, len(X))
    best = train(net, X[nv:], y[nv:], X[:nv], y[:nv], args.epochs, args.lr, args.bs)
    print('== 统计 BN 总体均值/方差（推理态用）', flush=True)
    net.bn_stats(X[:120000])
    np.savez('/tmp/hwr3_weights.npz', w1=net.w1, b1=net.b1, g1=net.g1, be1=net.be1,
             w2=net.w2, b2=net.b2, g2=net.g2, be2=net.be2, wf=net.wf, bf=net.bf,
             prot=net.prot, mu1=net.mu1, va1=net.va1, mu2=net.mu2, va2=net.va2)
    acc = evaluate(net, Xt, yt)
    print(f'== 测试集 top-1 = {acc:.4f}（训练中最好验证 {best:.4f}）', flush=True)
    # top-5
    ok5 = 0
    for i in range(0, len(Xt), 512):
        lg, _ = net.forward(Xt[i:i + 512], train=False)
        top5 = np.argpartition(-lg, 5, axis=1)[:, :5]
        ok5 += int(sum(yt[i + k] in top5[k] for k in range(len(top5))))
    print(f'   测试集 top-5 = {ok5/len(Xt):.4f}', flush=True)
    export(net, chars, OUT_DIR)
    json.dump({'classes': K, 'c1': args.c1, 'c2': args.c2, 'nred': args.nred,
               'params': int(npar), 'top1': float(acc), 'top5': float(ok5 / len(Xt)),
               'mac': int(256 * 9 * args.c1 + 64 * 9 * args.c1 * args.c2
                          + 16 * args.c2 * args.nred + K * args.nred)},
              open(os.path.join(OUT_DIR, 'train_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()