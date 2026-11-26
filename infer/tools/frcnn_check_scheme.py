#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""frcnn_check_scheme.py —— 校验「导出的 Scheme 权重」与「Scheme 侧推理」是否与参照实现一致

做三件事：
  1. 解析 app/lib/frcnn_weights.scm（int8 + 每通道尺度），与 fp32 权重逐张量比对；
  2. 用量化后的权重重跑一遍 numpy 参照推理，报告量化对 mAP 的影响；
  3. 生成 tools/e2e/wasm/frcnn_case.scm：把 3 张测试图的 ASCII 编码喂给
     app/lib/frcnn_core.scm，跑完整 11 步并打印中间量；同目录的
     frcnn_case_expect.txt 是 numpy 侧的同一批数字（供 diff）。

用法：
    python3 tools/frcnn_check_scheme.py            # 校验 + 生成用例
    ./tools/e2e/wasm/run_wasm_probe.sh tools/e2e/wasm/frcnn_case.scm   # 在真 wasm32 上跑
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frcnn_common as C   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
APP_LIB = os.path.normpath(os.path.join(HERE, '..', 'app', 'lib'))
DATA = os.path.join(HERE, 'data', 'frcnn')
CASE_DIR = os.path.join(HERE, 'e2e', 'wasm')

ALPHA = ''.join(chr(33 + i) for i in range(64))


def _dec(s):
    """6 bit/字符 → 字节（4 字符 3 字节，大端位序）。"""
    out = bytearray()
    n = len(s)
    for i in range(0, n, 4):
        v = 0
        for k in range(4):
            ch = s[i + k] if i + k < n else '!'
            v = (v << 6) | (ord(ch) - 33)
        out += bytes([(v >> 16) & 255, (v >> 8) & 255, v & 255])
    return bytes(out)


def load_scheme_weights(path):
    """解析 frcnn_weights.scm → 与 fold_and_float 同布局的 fp32 权重字典。"""
    src = open(path, encoding='utf-8').read()
    m = re.search(r'\(define WQ "((?:[^"\\]|\\.)*)"\)', src, re.S)
    if not m:
        raise SystemExit('WQ 未找到')
    lit = m.group(1)
    out = []
    i = 0
    while i < len(lit):
        if lit[i] == '\\':
            out.append(lit[i + 1]); i += 2
        else:
            out.append(lit[i]); i += 1
    wq = ''.join(out)
    raw = np.frombuffer(_dec(wq), dtype=np.uint8).astype(np.int32)
    raw = raw[:len(raw) // 2 * 2]          # 编码按 3 字节对齐，末尾可能补 1 字节
    # int16 大端：每两个字节一个权重
    q = raw[0::2] * 256 + raw[1::2]
    q = np.where(q > 32767, q - 65536, q)

    meta = {}
    for mm in re.finditer(r"\((\w+) (\d+) (\d+) \(([-0-9.eE ]+)\)\)", src):
        meta[mm.group(1)] = (int(mm.group(2)), int(mm.group(3)),
                             np.array([float(x) for x in mm.group(4).split()], np.float32))

    shapes = {
        'conv1': (C.NCH1, 1, 3, 3), 'conv2': (C.NCH, C.NCH1, 3, 3),
        'dw': (C.NCH, 3, 3), 'pw': (C.NCH, C.NCH),
        'rpn_conv': (C.NCH, C.NCH), 'rpn_cls': (C.NA, C.NCH),
        'rpn_reg': (C.NA * 4, C.NCH), 'fc1': (C.HID, C.FDIM),
        'cls': (C.NCLS, C.HID), 'reg': (C.NCLS * 4, C.HID),
    }
    keys = {'conv1': 'conv1_w', 'conv2': 'conv2_w', 'dw': 'dw_w', 'pw': 'pw_w',
            'rpn_conv': 'rpn_conv_w', 'rpn_cls': 'rpn_cls_w', 'rpn_reg': 'rpn_reg_w',
            'fc1': 'fc1_w', 'cls': 'cls_w', 'reg': 'reg_w'}
    P = {}
    off = 0
    for name in ['conv1', 'conv2', 'dw', 'pw', 'rpn_conv', 'rpn_cls', 'rpn_reg', 'fc1', 'cls', 'reg']:
        nch, per, sc = meta[name]
        cnt = nch * per
        blk = q[off:off + cnt]
        off += cnt
        w = blk.astype(np.float32).reshape(nch, per) * sc[:, None]
        P[keys[name]] = w.reshape(shapes[name])
    # 偏置
    for name, key in [('CONV1', 'conv1_b'), ('CONV2', 'conv2_b'), ('DW', 'dw_b'), ('PW', 'pw_b'),
                      ('RPN_CONV', 'rpn_conv_b'),
                      ('RPN_CLS', 'rpn_cls_b'), ('RPN_REG', 'rpn_reg_b'), ('FC1', 'fc1_b'),
                      ('CLS', 'cls_b'), ('REG', 'reg_b')]:
        mm = re.search(r"\(define %s_B '\(([-0-9.eE ]+)\)\)" % name, src)
        P[key] = np.array([float(x) for x in mm.group(1).split()], np.float32)
    return P, len(q)


def fmt(v, nd=5):
    return ('%.' + str(nd) + 'g') % v


def dump_expect(P, cases, path):
    """numpy 侧的同一批数字（与 frcnn_case.scm 的打印格式逐行对应）。"""
    lines = []
    for ci, img in enumerate(cases):
        d = C.reference_infer(img, P, detail=True)
        lines.append('CASE %d' % ci)
        lines.append('f1sum %.4f' % float(d['f1'].sum()))
        lines.append('f2sum %.4f' % float(d['f2'].sum()))
        obj = d['obj'].reshape(-1)
        lines.append('objmax %.4f idx %d' % (float(obj.max()), int(obj.argmax())))
        props, ps = d['proposals']
        lines.append('proposals %d' % len(props))
        rois, rs = d['rois']
        lines.append('rois %d' % len(rois))
        for i in range(len(rois)):
            lines.append('roi %d %.4f %.2f %.2f %.2f %.2f' % (i, rs[i], rois[i][0], rois[i][1],
                                                             rois[i][2], rois[i][3]))
        for i in range(min(3, len(rois))):
            lines.append('pool %d %.4f' % (i, float(d['pooled'][i].sum())))
            p = d['cls_p'][i]
            lines.append('cls %d argmax %d p %.4f' % (i, int(p.argmax()), float(p.max())))
        lines.append('dets %d' % len(d['det']))
        for (c, s, x1, y1, x2, y2) in d['det']:
            lines.append('det %d %.4f %.2f %.2f %.2f %.2f' % (c, s, x1, y1, x2, y2))
    open(path, 'w').write('\n'.join(lines) + '\n')


def gen_case_scm(cases, path):
    arr = []
    for img in cases:
        s = ''.join(chr(33 + int(round(v * 63))) for v in img.reshape(-1))
        s = s.replace('\\', '\\\\').replace('"', '\\"')
        arr.append('  "%s"' % s)
    body = '''
;; frcnn_case.scm —— **自动生成**（tools/frcnn_check_scheme.py）
;; 把 3 张测试图喂给 app/lib/frcnn_core.scm，在真 wasm32 上跑完整 11 步并打印中间量。
;; 期望值见 frcnn_case_expect.txt（numpy 参照实现，fp32；容差 ~1e-3）。
(native Math)
(native String)
(native System)

(import FR "app/lib/frcnn_core.scm")

(define cases
  '(
%s))

(define f2s (lambda (x) (String.atom_to_string (Math.to_fixed x 4))))

(define run_stage
  (lambda (s) {
    (define f 0.0)
    (FR.begin_stage s)
    (while (< f 1.0)
      (set! f (FR.step_stage s 400000)))
    f
  }))

(define fsum
  (lambda (lst n) {
    (define i 0)
    (define s 0.0)
    (while (< i n) {
      (set! s (+ s (get_item lst i)))
      (set! i (+ i 1))
    })
    s
  }))

(define omax 0.0)
(define oidx 0)
(define i 0)
(define k 0)

(FR.init)
(set! i 0)
(while (< i 3) {
  (FR.set_from_ascii (get_item cases i))
  (FR.reset_results)
  (run_stage 0)
  (run_stage 1)
  (run_stage 2)
  (run_stage 3)
  (run_stage 4)
  (run_stage 5)
  (run_stage 6)
  (run_stage 7)
  (run_stage 8)
  (run_stage 9)
  (run_stage 10)
  (display "CASE ") (display i) (newline)
  (display "f1sum ") (display (f2s (fsum FR.f1 (* 8 (* 32 32))))) (newline)
  (display "f2sum ") (display (f2s (fsum FR.f2 (* 24 256)))) (newline)
  (set! omax 0.0)
  (set! oidx 0)
  (set! k 0)
  (while (< k 2304) {
    (if (> (get_item FR.objp k) omax) {
      (set! omax (get_item FR.objp k))
      (set! oidx k)
    } #f)
    (set! k (+ k 1))
  })
  (display "objmax ") (display (f2s omax)) (display " idx ") (display oidx) (newline)
  (display "proposals ") (display FR.tk_n) (newline)
  (display "rois ") (display FR.roi_n) (newline)
  (set! k 0)
  (while (< k FR.roi_n) {
    (display "roi ") (display k) (display " ") (display (f2s (get_item FR.roi_s k)))
    (display " ") (display (f2s (get_item FR.roi_b (+ (* k 4) 0))))
    (display " ") (display (f2s (get_item FR.roi_b (+ (* k 4) 1))))
    (display " ") (display (f2s (get_item FR.roi_b (+ (* k 4) 2))))
    (display " ") (display (f2s (get_item FR.roi_b (+ (* k 4) 3))))
    (newline)
    (set! k (+ k 1))
  })
  (set! k 0)
  (while (< k (if (> FR.roi_n 3) 3 FR.roi_n)) {
    (display "pool ") (display k) (display " ")
    (display (f2s (FR.sum_range FR.pooled (* k 384) 384))) (newline)
    (display "cls ") (display k) (display " argmax ")
    (display (FR.roi_argmax k))
    (display " p ") (display (f2s (get_item FR.clsp (+ (* k 11) (FR.roi_argmax k))))) (newline)
    (set! k (+ k 1))
  })
  (display "dets ") (display FR.fin_n) (newline)
  (set! k 0)
  (while (< k FR.fin_n) {
    (display "det ") (display (get_item FR.fin_cls k)) (display " ")
    (display (f2s (get_item FR.fin_sc k))) (display " ")
    (display (f2s (get_item FR.fin_b (+ (* k 4) 0)))) (display " ")
    (display (f2s (get_item FR.fin_b (+ (* k 4) 1)))) (display " ")
    (display (f2s (get_item FR.fin_b (+ (* k 4) 2)))) (display " ")
    (display (f2s (get_item FR.fin_b (+ (* k 4) 3)))) (newline)
    (set! k (+ k 1))
  })
  (set! i (+ i 1))
})
(display "DONE") (newline)
''' % (',\n'.join(arr))
    open(path, 'w').write(body)


def main():
    wp = os.path.join(APP_LIB, 'frcnn_weights.scm')
    if not os.path.exists(wp):
        raise SystemExit('先跑训练导出：%s 不存在' % wp)
    Pq, nq = load_scheme_weights(wp)
    print('Scheme 权重：%d 个 int8' % nq)
    ref = np.load(os.path.join(DATA, 'weights_fp32.npz'))
    for k in Pq:
        a, b = Pq[k], ref[k]
        if a.shape != b.shape:
            print('  !! %s 形状不一致 %s vs %s' % (k, a.shape, b.shape))
            continue
        d = float(np.abs(a - b).max())
        s = float(np.abs(b).max()) + 1e-9
        print('  %-9s maxdiff %.4g  (rel %.2e)' % (k, d, d / s))

    z = np.load(os.path.join(HERE, 'data', 'frcnn', 'dataset.npz'))
    imgs = z['te_img']
    print('\n量化模型的 mAP（300 张）：')
    # 复用 train_frcnn_digits.evaluate
    import train_frcnn_digits as T
    cnt = z['te_cnt']
    off = np.concatenate([[0], np.cumsum(cnt)])
    m = T.evaluate(Pq, imgs, z['te_box'], z['te_lab'], off, n=300)
    print('  mAP@0.5 = %.4f  平均检测数 %.2f' % (m['map'], m['ndet']))

    cases = [imgs[i].astype(np.float32) for i in range(3)]   # [0,1]（0=黑，1=白）
    os.makedirs(CASE_DIR, exist_ok=True)
    dump_expect(Pq, cases, os.path.join(CASE_DIR, 'frcnn_case_expect.txt'))
    gen_case_scm(cases, os.path.join(CASE_DIR, 'frcnn_case.scm'))
    print('\n生成：tools/e2e/wasm/frcnn_case.scm + frcnn_case_expect.txt')
    print('运行：./tools/e2e/wasm/run_wasm_probe.sh tools/e2e/wasm/frcnn_case.scm')


if __name__ == '__main__':
    main()
