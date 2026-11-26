#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_frcnn_digits.py —— 为灵机 App「Faster R-CNN 两阶段检测」离线训练小模型并导出权重

用法：
    python3 tools/train_frcnn_digits.py gen            # 只合成数据（看统计/写缓存）
    python3 tools/train_frcnn_digits.py train [epochs]  # 训练 + 评估 + 导出
    python3 tools/train_frcnn_digits.py eval            # 用已导出的权重做参照评估

产物（本 App 运行时读到的资源）：
    app/lib/frcnn_weights.scm  量化权重（int16 每通道对称量化，4 字符/3 字节）——**自动生成，勿手改**
    tools/data/frcnn/model.json 参照推理用的 fp32 权重 + 若干测试样本（仅供交叉校验，不入产物）

网络（与 app/lib/frcnn_core.scm 逐条对应，改结构必须两边一起改）：
    输入 1x64x64 灰度，xn = (v/255 - 0.80)/0.21（v∈[0,1]，0=墨 1=纸）
    conv1  3x3/s2/p1  1→8   BN ReLU   → 8x32x32         （感受野 3）
    conv2  3x3/s2/p1  8→24  BN ReLU   → 24x16x16        （步长 4，感受野 7）
    conv3  3x3/s1/p1 逐通道 24→24 + 1x1 融合 24→24 BN ReLU → 24x16x16（感受野 11）
    RPN    1x1 24→24 ReLU → 分类 1x1 24→9（sigmoid 前景）/ 回归 1x1 24→36
           9 个锚框 = 尺寸 {12,17,24} × 宽高比 {0.5,0.7,0.95}，中心在格心
    RoI    4x4 最大池化（Caffe 量化口径）→ 24x4x4 = 384 维
    头     FC 384→64 ReLU → 分类 64→11（0..9 + 背景）/ 回归 64→44（11 类 × 4 偏移）
    后处理 RPN 提议：分数>0.30 → top100 → NMS(0.7) → 20 个 RoI；分类头**逐类**取阈值以上
           的类、按该类偏移细化框，逐类 NMS(0.35) → 最终检测（不做跨类去重：数字允许重叠）
    """
import gzip
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frcnn_common as C   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
APP_LIB = os.path.normpath(os.path.join(HERE, '..', 'app', 'lib'))
DATA_DIR = os.path.join(HERE, 'data', 'frcnn')
CACHE = os.path.join(DATA_DIR, 'dataset.npz')

# ---------------------------------------------------------------------------
# 1) 数据
# ---------------------------------------------------------------------------
def build_sprites():
    printed = C.load_printed_sprites()
    hand = C.load_mnist_sprites(per_class=1200)   # 每类 1200 张 MNIST 手写体
    print('印刷体 sprite %d 个（%d 种字体），MNIST 手写体 sprite %d 个'
          % (len(printed), len(set(t for t, _, _ in printed)), len(hand)))
    return printed, hand


def gen_dataset(n_train=24000, n_test=1200, seed=7):
    printed, hand = build_sprites()
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    tr = C.make_dataset(seed, n_train, printed, hand)
    print('train %d 张，用时 %.1fs' % (n_train, time.time() - t0))
    t0 = time.time()
    te = C.make_dataset(seed + 1, n_test, printed, hand)
    print('test  %d 张，用时 %.1fs' % (n_test, time.time() - t0))
    np.savez_compressed(CACHE,
                        tr_img=tr[0], te_img=te[0],
                        tr_box=np.concatenate(tr[1]), tr_lab=np.concatenate(tr[2]),
                        tr_cnt=np.array([len(x) for x in tr[1]]),
                        te_box=np.concatenate(te[1]), te_lab=np.concatenate(te[2]),
                        te_cnt=np.array([len(x) for x in te[1]]))
    print('cache →', CACHE, '(%.1f MB)' % (os.path.getsize(CACHE) / 1e6))
    m = tr[0].mean()
    s = tr[0].std()
    print('像素均值 %.4f 标准差 %.4f（MEAN/STD 常量请同步为 %.2f/%.2f）' % (m, s, round(m, 2), round(s, 2)))
    return CACHE


def load_dataset():
    if not os.path.exists(CACHE):
        gen_dataset()
    z = np.load(CACHE)
    def unpack(pref):
        cnt = z[pref + '_cnt']
        off = np.concatenate([[0], np.cumsum(cnt)])
        return z[pref + '_img'], z[pref + '_box'], z[pref + '_lab'], off
    return unpack('tr'), unpack('te')


# ---------------------------------------------------------------------------
# 2) torch 模型
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

ANCH = torch.from_numpy(C.make_anchors())          # (2304,4) x1y1x2y2


def roi_pool_torch(feat, rois, out=C.POOL, stride=C.STRIDE):
    """与 C.roi_pool_np 逐条一致的矢量化实现（可反传）。

    feat:(B,C,G,G)；rois:(R,5) = [b, x1,y1,x2,y2] → (R,C,out,out)
    """
    B, Cc, G, _ = feat.shape
    f = rois[:, 1:] / stride
    sx = torch.round(f[:, 0]).long().clamp(0, G - 1)
    sy = torch.round(f[:, 1]).long().clamp(0, G - 1)
    ex = torch.round(f[:, 2]).long().clamp(0, G - 1)
    ey = torch.round(f[:, 3]).long().clamp(0, G - 1)
    bw = (ex - sx + 1).clamp(min=1)
    bh = (ey - sy + 1).clamp(min=1)
    idx = torch.arange(out, dtype=torch.float32)
    x0 = torch.floor(sx[:, None].float() + idx[None, :] * bw[:, None].float() / out).long()
    x1 = torch.ceil(sx[:, None].float() + (idx[None, :] + 1) * bw[:, None].float() / out).long()
    y0 = torch.floor(sy[:, None].float() + idx[None, :] * bh[:, None].float() / out).long()
    y1 = torch.ceil(sy[:, None].float() + (idx[None, :] + 1) * bh[:, None].float() / out).long()
    x0 = x0.clamp(0, G - 1)
    y0 = y0.clamp(0, G - 1)
    x1 = torch.maximum(x1, x0 + 1).clamp(max=G)
    y1 = torch.maximum(y1, y0 + 1).clamp(max=G)

    R = rois.shape[0]
    flat = feat.reshape(B, Cc, G * G)
    bidx = rois[:, 0].long().reshape(R, 1, 1).expand(R, out, out)
    acc = None
    K = int(((bw.max().item() + out - 1) // out) + 1)
    K = min(max(K, 1), G)
    for dy in range(K):
        yy = (y0 + dy).clamp(max=G - 1)                      # (R,out)
        ok_y = (y0 + dy) < y1
        for dx in range(K):
            xx = (x0 + dx).clamp(max=G - 1)
            ok_x = (x0 + dx) < x1
            lin = yy[:, :, None] * G + xx[:, None, :]        # (R,out,out)
            g = flat[bidx, :, lin]                           # (R,out,out,C)
            g = g.permute(0, 3, 1, 2)                        # (R,C,out,out)
            ok = (ok_y[:, None, :, None] & ok_x[:, None, None, :]).float()
            g = g * ok + (-1e4) * (1.0 - ok)
            acc = g if acc is None else torch.maximum(acc, g)
    return acc


class FRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C.NCH1, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(C.NCH1)
        self.conv2 = nn.Conv2d(C.NCH1, C.NCH, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(C.NCH)
        # conv3 = 3x3 深度可分离卷积（逐通道 3x3 + 1x1 融合），感受野 7 -> 11
        self.dw = nn.Conv2d(C.NCH, C.NCH, 3, 1, 1, groups=C.NCH)
        self.pw = nn.Conv2d(C.NCH, C.NCH, 1)
        self.bn3 = nn.BatchNorm2d(C.NCH)
        self.rpn_conv = nn.Conv2d(C.NCH, C.NCH, 1)
        self.rpn_cls = nn.Conv2d(C.NCH, C.NA, 1)
        self.rpn_reg = nn.Conv2d(C.NCH, C.NA * 4, 1)
        self.fc1 = nn.Linear(C.FDIM, C.HID)
        self.cls = nn.Linear(C.HID, C.NCLS)
        self.reg = nn.Linear(C.HID, C.NCLS * 4)

    def backbone(self, x):
        f1 = F.relu(self.bn1(self.conv1(x)))
        f2 = F.relu(self.bn2(self.conv2(f1)))
        f3 = F.relu(self.bn3(self.pw(self.dw(f2))))
        return f1, f2, f3

    def rpn(self, f2):   # 参数名沿用 f2；实际传入 f3（检测特征图）
        h = F.relu(self.rpn_conv(f2))
        obj = self.rpn_cls(h)          # (B,9,16,16) logits
        reg = self.rpn_reg(h)          # (B,36,16,16)
        return h, obj, reg

    def head(self, f2, rois):
        pooled = roi_pool_torch(f2, rois)
        flat = pooled.reshape(rois.shape[0], -1)
        h1 = F.relu(self.fc1(flat))
        return self.cls(h1), self.reg(h1)


# ---------------------------------------------------------------------------
# 3) 目标分配 / 提议
# ---------------------------------------------------------------------------
def assign_anchors(anch, gt, lab):
    """RPN 锚框分配：IoU≥0.5 正样本，<0.3 负样本，每个 GT 的最佳锚框强制为正。"""
    n = anch.shape[0]
    if gt.shape[0] == 0:
        return np.zeros(n, np.int64), np.zeros((n, 4), np.float32)
    iou = C.iou_matrix(anch, gt)                     # (n,G)
    best = iou.argmax(1)
    best_iou = iou[np.arange(n), best]
    pos = best_iou >= 0.5
    neg = best_iou < 0.3
    for g in range(gt.shape[0]):                     # 每个 GT 至少一个正锚框
        a = int(iou[:, g].argmax())
        pos[a] = True
    tgt = np.zeros((n, 4), np.float32)
    tgt[:] = C.bbox_deltas(anch, gt[best])
    return np.where(pos, 1, np.where(neg, 0, -1)).astype(np.int64), tgt


def decode_proposals(obj_prob, reg, topk=C.PRE_NMS_TOPK):
    """锚框 + 回归量 → 提议框（与推理完全同一套后处理）。"""
    scores = obj_prob.reshape(-1)
    boxes = C.apply_deltas(C.make_anchors(), reg.reshape(-1, 4))
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, C.IMG - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, C.IMG - 1)
    ok = ((boxes[:, 2] - boxes[:, 0] >= 3) & (boxes[:, 3] - boxes[:, 1] >= 3) &
          (scores > C.SCORE_TH))
    idx = np.where(ok)[0]
    if idx.size > topk:
        idx = idx[np.argsort(-scores[idx])[:topk]]
    props = boxes[idx]
    pscore = scores[idx]
    keep = C.nms(props, pscore, C.RPN_NMS_IOU) if props.size else []
    keep = keep[:C.NROI]
    return props[keep], pscore[keep]


def sample_head_rois(props, pscore, gt, lab, rng, n_per_img=24):
    """分类头训练用的 RoI：RPN 提议 + GT，按 IoU 分配类别；采样至多 n_per_img 个。"""
    rois = np.concatenate([props, gt], 0) if gt.shape[0] else props
    if rois.shape[0] == 0:
        return None, None, None
    iou = C.iou_matrix(rois, gt)
    best = iou.argmax(1)
    biou = iou[np.arange(rois.shape[0]), best]
    cls = np.where(biou >= 0.5, lab[best], 10).astype(np.int64)   # 10 = 背景
    tgt = np.zeros((rois.shape[0], 4), np.float32)
    tgt[:] = C.bbox_deltas(rois, gt[best])
    pos = np.where(cls < 10)[0]
    neg = np.where(cls == 10)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    npos = min(len(pos), n_per_img // 2)
    nneg = min(len(neg), n_per_img - npos)
    sel = np.concatenate([pos[:npos], neg[:nneg]])
    return rois[sel], cls[sel], tgt[sel]


def smooth_l1(pred, tgt, beta=1.0 / 9.0):
    d = torch.abs(pred - tgt)
    return torch.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta).mean()


# ---------------------------------------------------------------------------
# 4) 训练 / 评估
# ---------------------------------------------------------------------------
def train(epochs=14, bs=16, lr=1.5e-3, n_train=None, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (tr_img, tr_box, tr_lab, tr_off), (te_img, te_box, te_lab, te_off) = load_dataset()
    N = min(n_train, tr_img.shape[0]) if n_train else tr_img.shape[0]
    model = FRCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (N // bs))
    rng = np.random.default_rng(seed + 100)
    step = 0
    for ep in range(epochs):
        perm = np.random.permutation(N)
        t0 = time.time()
        tot = np.zeros(6)
        cnt = 0
        for bi in range(0, N - bs + 1, bs):
            idx = perm[bi:bi + bs]
            x = torch.from_numpy(tr_img[idx].astype(np.float32) / 255.0)
            x = ((x - C.MEAN) / C.STD).unsqueeze(1)
            f1, f2, f3 = model.backbone(x)
            hf, obj, reg = model.rpn(f3)
            # ---- RPN 损失 ----
            obj_list, reg_list = [], []
            roi_list, roi_cls, roi_tgt = [], [], []
            for j in range(bs):
                gi = idx[j]
                g0, g1 = tr_off[gi], tr_off[gi + 1]
                gt = tr_box[g0:g1].astype(np.float32)
                lab = tr_lab[g0:g1].astype(np.int64)
                y, t = assign_anchors(C.make_anchors(), gt, lab)
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                rng.shuffle(pos)
                rng.shuffle(neg)
                npos = min(len(pos), 128)
                nneg = min(len(neg), 256 - npos)
                sel = np.concatenate([pos[:npos], neg[:nneg]])
                obj_list.append((j, sel, y[sel]))
                if npos:
                    reg_list.append((j, pos[:npos], t[pos[:npos]]))
                # ---- 分类头 RoI ----
                ob = 1.0 / (1.0 + np.exp(-obj[j].detach().numpy().transpose(1, 2, 0).reshape(-1)))
                rg = reg[j].detach().numpy().transpose(1, 2, 0).reshape(-1, 4)
                pr, ps = decode_proposals(ob, rg)
                r, c, tt = sample_head_rois(pr, ps, gt, lab, rng)
                if r is not None and r.shape[0]:
                    roi_list.append(r)
                    roi_cls.append(c)
                    roi_tgt.append(tt)
            obj_pred = obj.permute(0, 2, 3, 1).reshape(bs, -1)
            loss_rpn_cls = torch.zeros(())
            nsel = 0
            parts = []
            ys = []
            for j, sel, yv in obj_list:
                parts.append(obj_pred[j, torch.from_numpy(sel)])
                ys.append(torch.from_numpy(yv.astype(np.float32)))
                nsel += len(sel)
            if nsel:
                loss_rpn_cls = F.binary_cross_entropy_with_logits(
                    torch.cat(parts), torch.cat(ys))
            reg_pred = reg.permute(0, 2, 3, 1).reshape(bs, -1, 4)
            if reg_list:
                rp = torch.cat([reg_pred[j, torch.from_numpy(p)] for j, p, _ in reg_list])
                rt = torch.cat([torch.from_numpy(t) for _, _, t in reg_list])
                loss_rpn_reg = smooth_l1(rp, rt)
            else:
                loss_rpn_reg = torch.zeros(())
            # ---- 分类头损失 ----
            if roi_list:
                rois_all = np.concatenate(roi_list, 0)
                b_idx = np.concatenate([[k] * len(r) for k, r in enumerate(roi_list)])
                rois_t = torch.from_numpy(np.concatenate([b_idx[:, None], rois_all], 1).astype(np.float32))
                cls_l, reg_l = model.head(f3, rois_t)
                ycls = torch.from_numpy(np.concatenate(roi_cls))
                yreg = torch.from_numpy(np.concatenate(roi_tgt))
                loss_cls = F.cross_entropy(cls_l, ycls)
                fg = ycls < 10
                if int(fg.sum()) > 0:
                    rr = reg_l.reshape(-1, C.NCLS, 4)
                    rr = rr[torch.arange(rr.shape[0])[fg], ycls[fg]]     # (n_fg,4) 取 GT 类的偏移
                    loss_reg = smooth_l1(rr, yreg[fg])
                else:
                    loss_reg = torch.zeros(())
            else:
                loss_cls = torch.zeros(())
                loss_reg = torch.zeros(())
            loss = loss_rpn_cls + loss_rpn_reg + loss_cls + loss_reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += np.array([loss.item(), loss_rpn_cls.item(), loss_rpn_reg.item(),
                             loss_cls.item(), loss_reg.item(), 1.0])
            cnt += 1
            step += 1
            if step % 50 == 0:
                print('  ep%d it%d loss %.3f (rpn_cls %.3f rpn_reg %.3f cls %.3f reg %.3f) %.1fs'
                      % (ep, step, tot[0] / cnt, tot[1] / cnt, tot[2] / cnt, tot[3] / cnt,
                         tot[4] / cnt, time.time() - t0))
        print('epoch %d: loss %.3f（%.1fs）' % (ep, tot[0] / max(cnt, 1), time.time() - t0))
        model.eval()
        recalibrate_bn(model, tr_img, n=800)
        P = fold_and_float(model)
        m = evaluate(P, te_img, te_box, te_lab, te_off, n=400)
        d = diagnose(P, te_img, te_box, te_lab, te_off, n=200)
        print('  test mAP@0.5 = %.4f  (mean dets/img %.2f)  RPNrecall %.3f  headAcc(on GT) %.3f'
              % (m['map'], m['ndet'], d['recall'], d['cls']))
        model.train()
        torch.save(model.state_dict(), os.path.join(DATA_DIR, 'model.pt'))
    return model


def recalibrate_bn(model, imgs, n=1500, bs=32):
    """重算 BN 的 running_mean/var（"BN recalibration"）。

    为什么必需：BN 让损失对卷积权重的**整体尺度不变**，于是权重尺度在训练中会沿着
    这条零空间漂移；实测 conv2 输出方差被压到 ~1e-3 后，running_var（EMA）与真实的
    批次统计出现可见偏差 —— 训练模式（批次统计）与推理/导出（running 统计）的特征
    随之差一截（实测 f2 差 4.4，头部分类准确率 0.88 → 0.25）。导出前用训练数据把两层
    统计量**精确重算**一遍，eval 模式就与训练时一致。
    """
    model.eval()
    n = min(n, imgs.shape[0])
    with torch.no_grad():
        s1 = torch.zeros(C.NCH1, dtype=torch.float64)
        q1 = torch.zeros(C.NCH1, dtype=torch.float64)
        cnt = 0
        for i in range(0, n, bs):
            x = torch.from_numpy(imgs[i:i + bs].astype(np.float32) / 255.0)
            x = ((x - C.MEAN) / C.STD).unsqueeze(1)
            c1 = model.conv1(x)
            s1 += c1.sum(dim=(0, 2, 3)).double()
            q1 += (c1 * c1).sum(dim=(0, 2, 3)).double()
            cnt += c1.shape[0] * c1.shape[2] * c1.shape[3]
        mean1 = s1 / cnt
        var1 = q1 / cnt - mean1 * mean1
        model.bn1.running_mean.copy_(mean1.float())
        model.bn1.running_var.copy_(var1.clamp(min=1e-6).float())
        s2 = torch.zeros(C.NCH, dtype=torch.float64)
        q2 = torch.zeros(C.NCH, dtype=torch.float64)
        s3 = torch.zeros(C.NCH, dtype=torch.float64)
        q3 = torch.zeros(C.NCH, dtype=torch.float64)
        cnt = 0
        for i in range(0, n, bs):
            x = torch.from_numpy(imgs[i:i + bs].astype(np.float32) / 255.0)
            x = ((x - C.MEAN) / C.STD).unsqueeze(1)
            f1 = torch.relu(model.bn1(model.conv1(x)))
            c2 = model.conv2(f1)
            s2 += c2.sum(dim=(0, 2, 3)).double()
            q2 += (c2 * c2).sum(dim=(0, 2, 3)).double()
            cnt += c2.shape[0] * c2.shape[2] * c2.shape[3]
        mean2 = s2 / cnt
        var2 = q2 / cnt - mean2 * mean2
        model.bn2.running_mean.copy_(mean2.float())
        model.bn2.running_var.copy_(var2.clamp(min=1e-6).float())
        cnt = 0
        for i in range(0, n, bs):
            x = torch.from_numpy(imgs[i:i + bs].astype(np.float32) / 255.0)
            x = ((x - C.MEAN) / C.STD).unsqueeze(1)
            f2 = torch.relu(model.bn2(model.conv2(torch.relu(model.bn1(model.conv1(x))))))
            c3 = model.pw(model.dw(f2))
            s3 += c3.sum(dim=(0, 2, 3)).double()
            q3 += (c3 * c3).sum(dim=(0, 2, 3)).double()
            cnt += c3.shape[0] * c3.shape[2] * c3.shape[3]
        mean3 = s3 / cnt
        var3 = q3 / cnt - mean3 * mean3
        model.bn3.running_mean.copy_(mean3.float())
        model.bn3.running_var.copy_(var3.clamp(min=1e-6).float())
    print('   BN 重标定: conv1 var %.4g  conv2 var %.4g' % (float(var1.mean()), float(var2.mean())))


def fold_and_float(model):
    """BN 折叠进卷积、导出 fp32 权重字典（与 numpy/Scheme 侧的张量布局一致）。"""
    model.eval()
    P = {}
    with torch.no_grad():
        def fold(conv, bn):
            w = conv.weight.clone()
            g, b, mu, var, eps = (bn.weight, bn.bias, bn.running_mean,
                                  bn.running_var, bn.eps)
            # y = gamma*(conv(x)-mu)/sqrt(var+eps) + beta，conv(x)=w*x+b
            #   ⇒ y = (gamma/s)*w*x + (gamma/s)*(b-mu) + beta
            s = g / torch.sqrt(var + eps)
            w = w * s.reshape(-1, 1, 1, 1)
            bb = (conv.bias - mu) * s + bn.bias
            return w.numpy().astype(np.float32), bb.numpy().astype(np.float32)
        P['conv1_w'], P['conv1_b'] = fold(model.conv1, model.bn1)
        P['conv2_w'], P['conv2_b'] = fold(model.conv2, model.bn2)
        P['dw_w'] = model.dw.weight.numpy().reshape(C.NCH, 3, 3).astype(np.float32)
        P['dw_b'] = model.dw.bias.numpy().astype(np.float32)
        pw_w, pw_b = fold(model.pw, model.bn3)
        P['pw_w'] = pw_w.reshape(C.NCH, C.NCH).astype(np.float32)
        P['pw_b'] = pw_b.astype(np.float32)
        P['rpn_conv_w'] = model.rpn_conv.weight.numpy().reshape(C.NCH, C.NCH).astype(np.float32)
        P['rpn_conv_b'] = model.rpn_conv.bias.numpy().astype(np.float32)
        P['rpn_cls_w'] = model.rpn_cls.weight.numpy().reshape(C.NA, C.NCH).astype(np.float32)
        P['rpn_cls_b'] = model.rpn_cls.bias.numpy().astype(np.float32)
        P['rpn_reg_w'] = model.rpn_reg.weight.numpy().reshape(C.NA * 4, C.NCH).astype(np.float32)
        P['rpn_reg_b'] = model.rpn_reg.bias.numpy().astype(np.float32)
        P['fc1_w'] = model.fc1.weight.numpy().astype(np.float32)     # (HID,FDIM)
        P['fc1_b'] = model.fc1.bias.numpy().astype(np.float32)
        P['cls_w'] = model.cls.weight.numpy().astype(np.float32)     # (NCLS,HID)
        P['cls_b'] = model.cls.bias.numpy().astype(np.float32)
        P['reg_w'] = model.reg.weight.numpy().astype(np.float32)     # (NCLS*4,HID)
        P['reg_b'] = model.reg.bias.numpy().astype(np.float32)
    return P


def diagnose(P, imgs, boxes, labs, off, n=200):
    """分阶段诊断：RPN 召回率（12 个 RoI 能否覆盖 GT）+ 分类头在 GT 框上的准确率。"""
    rec = 0
    gt_tot = 0
    cls_ok = 0
    cls_tot = 0
    ndet = 0
    for i in range(min(n, imgs.shape[0])):
        img = imgs[i].astype(np.float32)
        d = C.reference_infer(img, P, detail=True)
        g = boxes[off[i]:off[i + 1]]
        l = labs[off[i]:off[i + 1]]
        gt_tot += len(g)
        ndet += len(d['det'])
        pr = d['proposals'][0]
        for k in range(len(g)):
            if pr.shape[0] and C.iou_matrix(g[k:k + 1], pr)[0].max() >= 0.5:
                rec += 1
        if len(g):
            pooled = C.roi_pool_np(d['f3'], g.astype(np.float32))
            flat = pooled.reshape(len(g), -1)
            fc1 = np.maximum(flat @ P['fc1_w'].T + P['fc1_b'], 0.0)
            pred = (fc1 @ P['cls_w'].T + P['cls_b']).argmax(1)
            cls_ok += int((pred == l).sum())
            cls_tot += len(g)
    return {'recall': rec / max(1, gt_tot), 'cls': cls_ok / max(1, cls_tot),
            'ndet': ndet / max(1, min(n, imgs.shape[0]))}


def evaluate(P, imgs, boxes, labs, off, n=400, verbose=False):
    """mAP@0.5（逐类 AP，再对 10 类取平均）+ 每图的检测数。"""
    dets = []      # (img, cls, score, box)
    gts = []
    ndet = 0
    for i in range(min(n, imgs.shape[0])):
        img = imgs[i].astype(np.float32)
        d = C.reference_infer(img, P)
        ndet += len(d)
        for (c, s, x1, y1, x2, y2) in d:
            dets.append((i, c, s, np.array([x1, y1, x2, y2], np.float32)))
        gts.append((boxes[off[i]:off[i + 1]], labs[off[i]:off[i + 1]]))
    aps = []
    for c in range(10):
        npos = sum(int((l == c).sum()) for _, l in gts)
        if npos == 0:
            continue
        ds = [d for d in dets if d[1] == c]
        ds.sort(key=lambda d: -d[2])
        used = [np.zeros(len(g), bool) for _, g in gts]
        tp = np.zeros(len(ds))
        fp = np.zeros(len(ds))
        for k, (i, _, s, bb) in enumerate(ds):
            g = gts[i][0]
            if len(g) == 0:
                fp[k] = 1
                continue
            iou = C.iou_matrix(bb[None], g)[0]
            j = int(iou.argmax())
            if iou[j] >= 0.5 and not used[i][j]:
                used[i][j] = True
                tp[k] = 1
            else:
                fp[k] = 1
        ctp = np.cumsum(tp)
        cfp = np.cumsum(fp)
        rec = ctp / npos
        prec = ctp / np.maximum(1e-9, ctp + cfp)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = prec[rec >= t]
            ap += (p.max() if p.size else 0.0) / 11.0
        aps.append(ap)
        if verbose:
            print('   class %d AP %.3f (npos %d)' % (c, ap, npos))
    return {'map': float(np.mean(aps)) if aps else 0.0, 'ndet': ndet / max(1, min(n, imgs.shape[0])),
            'aps': aps}


# ---------------------------------------------------------------------------
# 5) 量化 + 导出 Scheme 资源
# ---------------------------------------------------------------------------
ALPHA = ''.join(chr(33 + i) for i in range(64))     # 6 bit/字符，码点 33..96


def _enc_bytes(buf):
    """每 4 个 6bit 字符合成 3 字节（大端位序）；字符串字面量里转义 \" 与 \\。"""
    out = []
    n = len(buf)
    for i in range(0, n, 3):
        b0 = buf[i]
        b1 = buf[i + 1] if i + 1 < n else 0
        b2 = buf[i + 2] if i + 2 < n else 0
        v = (b0 << 16) | (b1 << 8) | b2
        out.append(ALPHA[(v >> 18) & 63])
        out.append(ALPHA[(v >> 12) & 63])
        out.append(ALPHA[(v >> 6) & 63])
        out.append(ALPHA[v & 63])
    s = ''.join(out)
    return s.replace('\\', '\\\\').replace('"', '\\"')


QMAX = 32767          # int16 对称量化（int8 实测会把 mAP 从 0.56 掉到 0.20：RPN 的分数
                     # 正好卡在 0.3 阈値附近，0.4% 的权重误差会把提议整批换掉）


def _q8(w, per_channel):
    """int16 对称量化（函数名沿用 _q8）。per_channel=True 时按输出通道分组。

    返回 (int16 数组, scales[], (组数, 每组元素数))。
    """
    w = np.asarray(w, np.float32)
    if per_channel:
        flat = w.reshape(w.shape[0], -1)
        amax = np.abs(flat).max(1)
    else:
        flat = w.reshape(1, -1)
        amax = np.abs(flat).max(1)
    scale = np.where(amax > 1e-8, amax / QMAX, 1e-8).astype(np.float32)
    if per_channel:
        q = np.clip(np.round(w.reshape(w.shape[0], -1) / scale[:, None]), -QMAX, QMAX).astype(np.int16)
        q = q.reshape(w.shape)
        groups = [w.shape[0], int(np.prod(w.shape[1:]))]
    else:
        q = np.clip(np.round(w / scale[0]), -QMAX, QMAX).astype(np.int16)
        groups = [1, int(w.size)]
    return q, scale, groups


def export_scheme(P, path):
    """把权重写成 app/lib/frcnn_weights.scm：一个 int8 串 + 每通道尺度表。"""
    layers = [
        # (名字, 权重键, 偏置键, 是否按输出通道量化)
        ('conv1', 'conv1_w', 'conv1_b', True),
        ('conv2', 'conv2_w', 'conv2_b', True),
        ('dw', 'dw_w', 'dw_b', True),
        ('pw', 'pw_w', 'pw_b', True),
        ('rpn_conv', 'rpn_conv_w', 'rpn_conv_b', True),
        ('rpn_cls', 'rpn_cls_w', 'rpn_cls_b', True),
        ('rpn_reg', 'rpn_reg_w', 'rpn_reg_b', True),
        ('fc1', 'fc1_w', 'fc1_b', True),
        ('cls', 'cls_w', 'cls_b', True),
        ('reg', 'reg_w', 'reg_b', True),
    ]
    blobs = []
    meta = []
    ntotal = 0
    for name, wk, bk, pc in layers:
        q, scale, (nch, per) = _q8(P[wk], pc)
        blobs.append(q.astype('>i2').tobytes())   # 显式大端（Scheme 侧按大端读）
        meta.append((name, nch, per, scale.astype(np.float32)))
        ntotal += q.size
    qbuf = b''.join(blobs)
    txt = _enc_bytes(qbuf)

    lines = []
    lines.append(';; frcnn_weights.scm —— Faster R-CNN 检测网络权重（**自动生成，勿手改**）')
    lines.append(';;')
    lines.append(';; 生成脚本：infer/tools/train_frcnn_digits.py（export_scheme）')
    lines.append(';; 训练数据：程序化合成的 64x64 灰度图（1~6 个随机位置/大小/字体/±5°倾斜的数字）')
    lines.append(';; 权重：int16 对称量化（每个输出通道一个尺度），尺度用 6 位有效数字的浮点')
    lines.append(';; 字符串编码：每字符 6 bit，码点 = 33+v（v∈[0,63]），4 字符 → 3 字节（大端位序）')
    lines.append(';;')
    lines.append(';; 张量布局（行主序，与 app/lib/frcnn_core.scm 的取址完全一致）：')
    lines.append(';;   conv1    (8,1,3,3)     conv2   (16,8,3,3)')
    lines.append(';;   rpn_conv (16,16,1,1)   rpn_cls (9,16)     rpn_reg (36,16)')
    lines.append(';;   fc1      (48,256)      cls     (11,48)     reg     (40,48)')
    lines.append(';; 偏置直接以 6 位有效数字的浮点列出（不量化）。')
    lines.append('')
    lines.append(';; 权重总字节数 = %d' % len(qbuf))
    lines.append('(define WQ "%s")' % txt)
    lines.append('')
    # 尺度表：(名字 通道数 每通道元素数 (尺度...))
    lines.append(';; 每层的 int8 块元信息：(名字 通道数 每通道元素数 尺度...)')
    lines.append('(define WMETA')
    lines.append("  '(")
    for name, nch, per, scale in meta:
        svals = ' '.join('%.6g' % float(v) for v in scale)
        lines.append("    (%s %d %d (%s))" % (name, nch, per, svals))
    lines.append('  ))')
    lines.append('')
    for name, wk, bk, *_ in layers:
        b = P[bk]
        vals = ' '.join('%.6g' % float(v) for v in b.reshape(-1))
        lines.append("(define %s_B '(%s))" % (name.upper(), vals))
    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('权重 →', path, '(%.1f KB, %d 个权重)' % (os.path.getsize(path) / 1024, ntotal))


def export_reference(P, path, imgs, boxes, labs, off, n=6):
    """导出 fp32 权重 + 若干样本图 + 参照检测结果，供 Scheme 侧交叉校验（不入产物）。"""
    obj = {'weights': {k: v.tolist() for k, v in P.items()}, 'cases': []}
    for i in range(min(n, imgs.shape[0])):
        img = imgs[i].astype(np.float32)
        det = C.reference_infer(img, P)
        obj['cases'].append({
            'gray': np.round(img.reshape(-1) * 255).astype(int).tolist(),
            'gt': [[float(v) for v in b] + [int(l)] for b, l in
                   zip(boxes[off[i]:off[i + 1]], labs[off[i]:off[i + 1]])],
            'det': [[int(c), float(s)] + [float(v) for v in b] for (c, s, x0, y0, x1, y1) in det
                    for b in [(x0, y0, x1, y1)]],
        })
    with open(path, 'w') as f:
        json.dump(obj, f)
    print('参照推理 →', path, '(%.1f KB)' % (os.path.getsize(path) / 1024))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if cmd == 'gen':
        gen_dataset()
        return
    if cmd in ('train', 'eval'):
        if cmd == 'train':
            epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
            model = train(epochs=epochs)
            P = fold_and_float(model)
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'model.pt'))
        else:
            model = FRCNN()
            model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'model.pt')))
            P = fold_and_float(model)
        np.savez(os.path.join(DATA_DIR, 'weights_fp32.npz'), **P)
        export_scheme(P, os.path.join(APP_LIB, 'frcnn_weights.scm'))
        (tr_img, tr_box, tr_lab, tr_off), (te_img, te_box, te_lab, te_off) = load_dataset()
        export_reference(P, os.path.join(DATA_DIR, 'model.json'), te_img, te_box, te_lab, te_off)
        m = evaluate(P, te_img, te_box, te_lab, te_off, n=1000, verbose=True)
        print('\n最终 mAP@0.5 = %.4f（测试集 1000 张，量化后另行校验）' % m['map'])
        return
    print(__doc__)


if __name__ == '__main__':
    main()
