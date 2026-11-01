"""train_hwr_online.py —— 手写汉字识别（在线轨迹特征 + 最近类均值）训练与权导出

数据：CASIA-OLHWDB1.1 原始笔迹（*.pot，240 训练书写人 / 60 测试书写人，3755 类）
      下载：https://nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.1Train.zip
            https://nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.1Test.zip
      许可：学术研究免费（2020 起无需申请）；**禁止商用**，商用需向 CASIA/NLPR 申请授权。
      参考：C.-L. Liu et al., CASIA Online and Offline Chinese Handwriting Databases, ICDAR 2011；
            C.-L. Liu et al., Online and Offline Handwritten Chinese Character Recognition:
            Benchmarking on New Databases, Pattern Recognition 46(1):155-162, 2013.

特征（512 维，全部可在灵机 Scheme 里复刻）：
  1. 轨迹按段插值画进 64×64 细线墨迹图（双线性铺点）
  2. 伪二维归一化：按行/列线密度投影做等化映射（与恒等映射混合 alpha=0.7）后重采样
  3. 8 方向梯度（中心差分 + 相邻方向软分配）
  4. 8×8 分块求和 → 8×8×8 = 512 维；每格 8 方向向量 L2 归一化，最后整体 L2 归一化
分类：PCA-128 降维 + 最近类均值（NCM）。NCM 训练只需类均值（无需梯度下降），
      且能直接 int8 逐类量化，推理开销 = 512×128（投影）+ 3755×128（分类）。
"""
import argparse, glob, os, struct, time, json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), 'app', 'lib', 'hwr2')
NG = 64          # 墨迹图分辨率
GRID = 8         # 分块网格（NG/GRID = 8 像素一格）
ALPHA = 0.7      # 伪二维等化强度
NDIR = 8


# ---------------------------------------------------------------- 数据解析
def gb_to_unicode_table():
    m = {}
    for b1 in range(0xA1, 0xF8):
        for b2 in range(0xA1, 0xFF):
            try:
                m[(b1 << 8) | b2] = bytes([b1, b2]).decode('gb2312')
            except Exception:
                pass
    return m


def parse_pot(path):
    """解析一个书写人的 .pot 文件 → [(gb 码, [stroke(n,2), ...]), ...]
    记录格式：[2B 长度][4B GB 码（位在前、区在后）][2B 笔画数][(x,y) i16 ...]
              笔画结束 (-1,0)，字符结束 (-1,-1)"""
    d = open(path, 'rb').read()
    i, n, out = 0, len(d), []
    while i + 8 <= n:
        size = struct.unpack_from('<H', d, i)[0]
        if size < 8:
            break
        t0, t1 = d[i + 2], d[i + 3]
        gb = (t1 << 8) | t0
        p, end, strokes, cur = i + 8, i + size, [], []
        while p + 4 <= end:
            x, y = struct.unpack_from('<hh', d, p)
            p += 4
            if x == -1 and y == -1:
                break
            if x == -1 and y == 0:
                if cur:
                    strokes.append(np.array(cur, np.float32))
                cur = []
                continue
            cur.append((x, y))
        if cur:
            strokes.append(np.array(cur, np.float32))
        if strokes:
            out.append((gb, strokes))
        i += size
    return out


def load_split(train_dir, test_dir, n_train_writers=None, test_stride=1):
    g2u = gb_to_unicode_table()
    def build(files):
        S, L = [], []
        for f in files:
            for gb, st in parse_pot(f):
                c = g2u.get(gb)
                if c is None or not ('\u4e00' <= c <= '\u9fff'):
                    continue
                S.append(st); L.append(gb)
        return S, np.array(L)
    tr = sorted(glob.glob(os.path.join(train_dir, '*.pot')))
    te = sorted(glob.glob(os.path.join(test_dir, '*.pot')))
    if n_train_writers:
        tr = tr[:n_train_writers]
    Str, Ltr = build(tr)
    Ste, Lte = build(te)
    Ste, Lte = Ste[::test_stride], Lte[::test_stride]
    return Str, Ltr, Ste, Lte


# ---------------------------------------------------------------- App 输入仿真
def simulate_app(strokes, cells=50.0, resample=2.5, rng=None, straight_p=0.0):
    """把高精度笔迹降级为"128px 绘图板（64 格）+ 每 20ms 采样"的形态：
    字的外接方框缩放到 cells 格，坐标落整格，并按弧长 resample 抽稀。

    另外以 straight_p 概率做"书写简化"增广：每笔只保留 2~4 个折点（端点 + 等距采样）
    并轻度平滑——真机上手写常是"几段直线"式的简化笔迹，训练集里必须有这种形态，
    否则模型只认自然笔迹（实测：不加这个增广时，理想折线输入几乎全错）。"""
    allp = np.concatenate(strokes)
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    sc = max(x1 - x0, y1 - y0, 1e-9)
    out = []
    rng = rng if rng is not None else np.random.default_rng()
    straight = rng.random() < straight_p
    for s in strokes:
        p = (s - [x0, y0]) / sc * cells
        if straight:
            # 只保留首尾 + 1~2 个中间折点
            n = len(p)
            k = int(rng.integers(2, 5))
            if n > k:
                idx = np.linspace(0, n - 1, k).astype(int)
                p = p[idx]
        q = [p[0]]
        for k2 in range(1, len(p)):
            seg = p[k2] - q[-1]
            L = float(np.hypot(*seg))
            while L >= resample:
                q.append(q[-1] + seg / L * resample)
                seg = p[k2] - q[-1]
                L = float(np.hypot(*seg))
        q.append(p[-1])
        p = np.round(np.array(q))
        out.append(p)
    return out


# ---------------------------------------------------------------- 特征
def render_thin(strokes, ng=NG):
    strokes = [np.asarray(s, np.float64) for s in strokes if len(s) > 0]
    if not strokes:
        return None
    allp = np.concatenate(strokes)
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    W = max(x1 - x0, 1e-9); H = max(y1 - y0, 1e-9)
    img = np.zeros((ng, ng), np.float64)
    for s in strokes:
        p = np.empty_like(s)
        p[:, 0] = (s[:, 0] - x0) / W * (ng - 1)
        p[:, 1] = (s[:, 1] - y0) / H * (ng - 1)
        if len(p) == 1:
            j = min(ng - 1, int(round(p[0, 0]))); i = min(ng - 1, int(round(p[0, 1])))
            img[i, j] += 1.0
            continue
        for k in range(len(p) - 1):
            dx = p[k + 1, 0] - p[k, 0]; dy = p[k + 1, 1] - p[k, 1]
            n = max(1, int(np.hypot(dx, dy) * 2))
            for t in range(n + 1):
                x = p[k, 0] + dx * t / n; y = p[k, 1] + dy * t / n
                j0 = int(np.floor(x)); i0 = int(np.floor(y))
                fx = x - j0; fy = y - i0
                for jj, ii, w in ((j0, i0, (1 - fx) * (1 - fy)), (j0 + 1, i0, fx * (1 - fy)),
                                  (j0, i0 + 1, (1 - fx) * fy), (j0 + 1, i0 + 1, fx * fy)):
                    if 0 <= ii < ng and 0 <= jj < ng:
                        img[ii, jj] += w / (n + 1)
    return img


def normalize_p2d(img, alpha=ALPHA, smooth=2):
    ng = img.shape[0]
    def warp1d(prof):
        p = prof.astype(np.float64).copy()
        for _ in range(smooth):
            p = np.convolve(np.pad(p, 1, mode='edge'), [0.25, 0.5, 0.25], 'valid')
        if p.sum() <= 0:
            return np.linspace(0, 1, ng)
        c = np.concatenate([[0], np.cumsum(p)]); c /= c[-1]
        idx = np.linspace(0, 1, ng)
        eq = np.maximum.accumulate(np.interp(idx, c, np.linspace(0, 1, len(c))))
        f = alpha * eq + (1 - alpha) * idx
        return f / max(f[-1], 1e-9)
    fx = warp1d(img.sum(0)); fy = warp1d(img.sum(1))
    sx = fx * (ng - 1); sy = fy * (ng - 1)
    x0 = np.clip(np.floor(sx).astype(np.int32), 0, ng - 1); y0 = np.clip(np.floor(sy).astype(np.int32), 0, ng - 1)
    ax = sx - x0; ay = sy - y0
    x1 = np.minimum(x0 + 1, ng - 1); y1 = np.minimum(y0 + 1, ng - 1)
    return (img[np.ix_(y0, x0)] * (1 - ay)[:, None] * (1 - ax)[None, :]
            + img[np.ix_(y0, x1)] * (1 - ay)[:, None] * ax[None, :]
            + img[np.ix_(y1, x0)] * ay[:, None] * (1 - ax)[None, :]
            + img[np.ix_(y1, x1)] * ay[:, None] * ax[None, :])


def grad_dirs(img):
    ng = img.shape[0]
    p = np.pad(img, 1, mode='constant')
    gy = p[2:, 1:-1] - p[:-2, 1:-1]
    gx = p[1:-1, 2:] - p[1:-1, :-2]
    mag = np.hypot(gx, gy)
    bf = np.mod(np.arctan2(gy, gx), 2 * np.pi) / (2 * np.pi) * NDIR
    b0 = np.minimum(bf.astype(np.int32), NDIR - 1)
    b1 = (b0 + 1) % NDIR
    w1 = bf - np.floor(bf); w0 = 1.0 - w1
    return [mag * np.where(b0 == k, w0, 0.0) + mag * np.where(b1 == k, w1, 0.0) for k in range(NDIR)]


def featurize(strokes, ng=NG, grid=GRID, alpha=ALPHA):
    img = render_thin(strokes, ng=ng)
    if img is None or img.sum() <= 0:
        return None
    img = normalize_p2d(img * 255.0, alpha=alpha)
    if img.max() > 0:
        img = img / img.max()
    D = grad_dirs(img)
    c = ng // grid
    P = np.zeros((NDIR, grid, grid), np.float32)
    for k in range(NDIR):
        P[k] = D[k].reshape(grid, c, grid, c).sum(axis=(1, 3))
    v = P.transpose(1, 2, 0).reshape(-1, NDIR)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-3)
    f = v.reshape(-1).astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-6)


_APP_SIM = False
_SEED = 0


def _feat_worker(st):
    if _APP_SIM:
        # 每个样本用确定性的随机种子（同一字符的不同增广互不相同，但可复现）
        rng = np.random.default_rng(abs(hash(st[0].tobytes())) % (2 ** 31))
        st = simulate_app(st, rng=rng)
    return featurize(st)


def build_features(samples, procs=14, app_sim=False):
    from multiprocessing import Pool
    global _APP_SIM
    _APP_SIM = app_sim
    worker = _feat_worker
    with Pool(procs) as p:
        F = p.map(worker, samples, chunksize=100)
    keep = np.array([i for i, f in enumerate(F) if f is not None])
    return np.stack([F[i] for i in keep]), keep


# ---------------------------------------------------------------- 分类器
def ncm_eval(Ftr, ytr, Fte, yte, K, chunk=8192):
    means = np.stack([Ftr[ytr == k].mean(0) for k in range(K)]).astype(np.float32)
    mb = (means ** 2).sum(1)
    h1 = h5 = n = 0
    for i in range(0, len(Fte), chunk):
        x = Fte[i:i + chunk]; y = yte[i:i + chunk]
        sc = x @ (2 * means.T) - mb[None, :]
        t5 = np.argpartition(-sc, 5, axis=1)[:, :5]
        am = np.take_along_axis(sc, t5, axis=1).argmax(1)
        h1 += int((t5[np.arange(len(y)), am] == y).sum())
        h5 += int(sum(y[j] in t5[j] for j in range(len(y))))
        n += len(y)
    return h1 / n, h5 / n, means


# ---------------------------------------------------------------- 导出
def export(pca, means, chars, outdir, scale_row=True):
    """导出：PCA 矩阵 (512×r) + 类均值 (K×r) + 汉字表 + meta.scm（int8 逐行量化）"""
    os.makedirs(outdir, exist_ok=True)
    r = pca.shape[1]
    # PCA 矩阵：整体一个尺度
    sp = float(np.abs(pca).max()) / 127.0
    qp = np.clip(np.round(pca / sp) + 128, 0, 255).astype(np.int32)
    # 类均值：每类一个尺度（与现有实现同款"逐类量化"）
    sm = np.maximum(np.abs(means).max(1) / 127.0, 1e-9)
    qm = np.clip(np.round(means / sm[:, None]) + 128, 0, 255).astype(np.int32)
    smax = float(sm.max())
    sq = np.clip(np.round(sm / smax * 65535.0), 0, 65535).astype(np.int32)
    # 反量化校验
    Mp = (qp - 128) * sp
    Mq = (qm - 128) * (sq / 65535.0 * smax)[:, None]
    print(f'  量化: PCA max|Δ|={np.abs(pca - Mp).max():.4g}  均值 max|Δ|={np.abs(means - Mq).max():.4g}')

    PER_ROWS = 400          # 每个文件 400 行（400×nred 字符 = 51200 字符 = 102400 字节 < 128KB），
                            # 行对齐使 Scheme 侧只需按行切换文件、下标从 0 重算
    def write(name, arr2d):
        n = arr2d.shape[0]
        k = 0
        for lo in range(0, n, PER_ROWS):
            chunk = arr2d[lo:min(n, lo + PER_ROWS)]
            open(os.path.join(outdir, f'{name}{k}.txt'), 'w', encoding='utf-8', newline='').write(
                ''.join(chr(int(v) + 256) for v in chunk.reshape(-1)))
            k += 1
        return k
    np_ = write('p', qp)
    nm_ = write('m', qm)
    # |均值|² 的 16 位定点（最近类均值判别需要，避免启动时重算 480k 次乘加）
    sumsq = (qm.astype(np.float64) - 128.0)
    nrm2 = (sumsq ** 2).sum(1) * (sq / 65535.0 * smax) ** 2      # 真实 |m_k|²
    s2max = float(nrm2.max()) or 1.0
    q2 = np.clip(np.round(nrm2 / s2max * 65535.0), 0, 65535).astype(np.int32)
    # params.txt = [每类尺度 2 字符][每类 |m|² 2 字符][汉字表]
    open(os.path.join(outdir, 'params.txt'), 'w', encoding='utf-8', newline='').write(
        ''.join(chr(int(v >> 8) + 256) + chr(int(v & 255) + 256) for v in sq)
        + ''.join(chr(int(v >> 8) + 256) + chr(int(v & 255) + 256) for v in q2)
        + ''.join(chars))
    open(os.path.join(outdir, 'meta.scm'), 'w').write(
        ';; 由 tools/train_hwr_online.py 生成（勿手改）：在线手写识别器模型元信息\n'
        f'(define ncls {len(chars)})\n(define nfeat {pca.shape[0]})\n(define nred {r})\n'
        f'(define npca {np_})\n(define nmean {nm_})\n(define per_rows {PER_ROWS})\n(define sp {sp:.9g})\n'
        f'(define smax {smax:.9g})\n(define s2max {s2max:.9g})\n')
    print(f'  导出 → {outdir}: PCA {np_} 个文件, 均值 {nm_} 个文件')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', default='/tmp/potr')
    ap.add_argument('--test-dir', default='/tmp/pot')
    ap.add_argument('--writers', type=int, default=0, help='0=全部 240 个训练书写人')
    ap.add_argument('--stride', type=int, default=3, help='测试集抽样步长')
    ap.add_argument('--cache', default='/tmp/hwr2_feats.npz')
    ap.add_argument('--red', type=int, default=128)
    ap.add_argument('--no-app-sim', action='store_true')
    args = ap.parse_args()
    app_sim = not args.no_app_sim

    t0 = time.time()
    print('== 载入 CASIA-OLHWDB1.1 原始笔迹')
    Str, Ltr, Ste, Lte = load_split(args.train_dir, args.test_dir,
                                    args.writers or None, args.stride)
    print(f'   train {len(Str)}  test {len(Ste)}  ({time.time()-t0:.0f}s)')
    if os.path.exists(args.cache):
        d = np.load(args.cache)
        Ftr, Ytr, Fte, Yte = d['Ftr'], d['Ytr'], d['Fte'], d['Yte']
        print('   复用缓存特征', Ftr.shape, Fte.shape)
    else:
        print(f'== 提取特征（512 维；App 输入仿真={app_sim}）')
        Ftr, ki = build_features(Str, app_sim=app_sim); Ytr = Ltr[ki]
        Fte, kj = build_features(Ste, app_sim=False); Yte_raw = Lte[kj]
        Fte2, kj2 = build_features(Ste, app_sim=app_sim); Yte2 = Lte[kj2]
        d = dict(Ftr=Ftr, Ytr=Ytr, Fte=Fte, Yte=Yte_raw, Fte2=Fte2, Yte2=Yte2)
        np.savez_compressed(args.cache, **d)
        Fte, Yte = Fte2, Yte2
        print(f'   完成（{time.time()-t0:.0f}s）')

    uc = np.unique(np.concatenate([Ytr]))
    idx = {v: i for i, v in enumerate(uc)}
    K = len(uc)
    ytr = np.array([idx[v] for v in Ytr])
    def prep(F, Y):
        m = np.array([v in idx for v in Y])
        return F[m], np.array([idx[v] for v in Y[m]])
    F_hi, y_hi = prep(d.get('Fte', Fte), d.get('Yte', Yte))
    F_app, y_app = prep(d.get('Fte2', Fte), d.get('Yte2', Yte))
    print(f'== 分类（{K} 类）：最近类均值')
    t1, t5, means = ncm_eval(Ftr, ytr, F_hi, y_hi, K)
    print(f'   512 维 + NCM（高精度输入）  top1={t1:.4f} top5={t5:.4f}', flush=True)
    t1, t5, _ = ncm_eval(Ftr, ytr, F_app, y_app, K)
    print(f'   512 维 + NCM（App 仿真输入）top1={t1:.4f} top5={t5:.4f}', flush=True)

    # PCA 降维 + 类内散布对角白化（白化可折进投影矩阵，推理零额外开销）
    cov = np.cov(Ftr[:120000], rowvar=False, bias=True)
    w, V = np.linalg.eigh(cov)
    V = V[:, ::-1]
    keep_P = keep_m = None
    best = {}
    for r in (256, args.red, 64):
        P = V[:, :r].astype(np.float32)
        t1, t5, _ = ncm_eval(Ftr @ P, ytr, F_app @ P, y_app, K)
        print(f'   PCA-{r} + NCM（App 仿真输入）  top1={t1:.4f} top5={t5:.4f}', flush=True)
        A = Ftr @ P
        means = np.stack([A[ytr == k].mean(0) for k in range(K)])
        Sw = np.zeros((r, r))
        for k in range(K):
            D = A[ytr == k] - means[k]
            Sw += D.T @ D
        Sw /= len(A)
        Wh = (1.0 / np.sqrt(np.abs(np.diag(Sw)) + 1e-6)).astype(np.float32)
        Pw = (P * Wh[None, :]).astype(np.float32)          # 白化折进投影矩阵
        t1, t5, m_r = ncm_eval(Ftr @ Pw, ytr, F_app @ Pw, y_app, K)
        print(f'   PCA-{r} + 对角白化 + NCM（App 仿真输入） top1={t1:.4f} top5={t5:.4f}', flush=True)
        best[r] = (float(t1), float(t5))
        if r == args.red:
            keep_P, keep_m = Pw, m_r
    print('== 导出')
    g2u = gb_to_unicode_table()
    export(keep_P, keep_m, [g2u.get(int(v), '\u25a1') for v in uc], OUT_DIR)
    json.dump({'classes': K, 'dim': 512, 'reduced': args.red, 'app_sim': app_sim,
               'pca_whiten_top1_top5': {str(r): v for r, v in best.items()},
               'shipped_top1': best[args.red][0], 'shipped_top5': best[args.red][1]},
              open(os.path.join(OUT_DIR, 'train_report.json'), 'w'), indent=1)
    print(f'done {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()