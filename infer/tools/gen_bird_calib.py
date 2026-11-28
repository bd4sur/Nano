#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_bird_calib.py —— 鸟鸣合成器（(native Bird)）的**离线标定表**生成器

背景
====
鸟鸣合成用的是 Mindlin–Laje 低维鸣管振荡器：一个二阶非线性 ODE，参数
(alpha 气压/阻尼, beta 张力, gamma 时间尺度)。三个量里只有 alpha 与 beta 能
"直接"给，而我们真正想控制的量是**基频 f0**（鸟叫的音高轮廓）与**振幅**
（响度）。二者与 (alpha,beta) 的关系没有解析解，只能离线测出来做表。

本脚本就是把这份表**自己测一遍**：用与 C 侧运行时逐位一致的积分器
（半隐式欧拉，同样的 gamma·dt 步长比）扫一遍 (alpha, beta) 网格，测出

    phi(alpha,beta) = f0 / gamma            （无量纲音高）
    amp(alpha,beta) = peak2peak(y) / gamma  （无量纲振幅，用于响度归一）

生成 infer/bird_calib.h 供 am_native_Bird.c 编译期内联。

为什么自己测而不用现成表
========================
1. 授权：参考实现（esp32-perch / Lyrebird）的标定表是 CC BY-NC-SA 数据文件，
   本仓只**参照算法**，不搬运数据；
2. 自洽：表必须由**与运行时同一个积分器**测得，否则基频与响度都会偏。本仓
   把步长比 gamma·dt 固定为 REF_S（见下）并对每个声部按 gamma 取整步数，
   于是同一张表对所有音高都成立（实测 phi 对 gamma 的依赖 <1%）。

运行
====
    cd infer && python3 tools/gen_bird_calib.py            # 写 bird_calib.h
    cd infer && python3 tools/gen_bird_calib.py --check    # 只打印，不写文件
"""

import math
import sys

# ---- 与 C 侧（am_native_Bird.c）必须一致的常量 ---------------------------
SR = 22050.0            # 采样率（(native Bird) 的固定输出速率）
GAMMA_REF = 24000.0     # 标定用 gamma
REF_S = 0.05            # 目标步长比 gamma·dt（C 侧按此取整每采样步数）
OVERSAMPLE = int(round(GAMMA_REF / (SR * REF_S)))   # → 22 步/采样
DT = 1.0 / (SR * OVERSAMPLE)
PHI_AT_BETA_ONE = 0.17061   # 文献值：alpha=0.1, beta=1 时 f0/gamma（仅做自检）

# 网格：alpha 由压力 pressure 线性映射（alpha = 0.05 + 0.4·pressure，pressure 0..1.4），
# beta 取对数等距（0.12..16，覆盖 0.5kHz~8kHz 的实际鸟叫音域）。
ALPHA_MIN, ALPHA_MAX, ALPHA_N = 0.05, 0.61, 9
BETA_MIN, BETA_MAX, BETA_N = 0.12, 16.0, 24

OUT = "bird_calib.h"


def alpha_grid():
    step = (ALPHA_MAX - ALPHA_MIN) / (ALPHA_N - 1)
    return [ALPHA_MIN + step * i for i in range(ALPHA_N)]


def beta_grid():
    r = (BETA_MAX / BETA_MIN) ** (1.0 / (BETA_N - 1))
    return [BETA_MIN * r ** i for i in range(BETA_N)]


def simulate(alpha, beta, gamma=GAMMA_REF, warm=250, meas=450):
    """半隐式欧拉积分，返回 (phi, amp)。不可振荡（收敛到不动点）时 phi=0。"""
    g = gamma
    g2 = g * g
    a_g2 = alpha * g2
    b_g2 = beta * g2
    dt = DT
    x = 1.0e-4
    y = 0.0
    ys = []
    for i in range((warm + meas) * OVERSAMPLE):
        x2 = x * x
        dy = -a_g2 - b_g2 * x - g2 * x2 * x - g * x2 * y + g2 * x2 - g * x * y
        x += y * dt
        y += dy * dt
        if not (abs(x) < 1e6 and abs(y) < 1e12):
            return 0.0, 0.0
        if i >= warm * OVERSAMPLE:
            ys.append(y)
    if not ys:
        return 0.0, 0.0
    pp = (max(ys) - min(ys)) / g
    cross = []
    for i in range(1, len(ys)):
        if ys[i - 1] <= 0.0 < ys[i]:
            cross.append(i - 1 + ys[i - 1] / (ys[i - 1] - ys[i]))
    if len(cross) < 3:
        return 0.0, pp
    per = (cross[-1] - cross[0]) / (len(cross) - 1)      # 以"过采样步"为单位的周期
    f0 = SR * OVERSAMPLE / per
    return f0 / gamma, pp


def main():
    write = "--check" not in sys.argv
    alphas = alpha_grid()
    betas = beta_grid()

    phi_tab = []
    amp_tab = []
    for a in alphas:
        prow, arow = [], []
        for b in betas:
            phi, amp = simulate(a, b)
            prow.append(phi)
            arow.append(amp if amp > 0.0 else 0.0)
        phi_tab.append(prow)
        amp_tab.append(arow)

    # ---- 自检：alpha=0.1,beta=1 时 phi 应接近文献值 0.17061 -------------
    phi_chk, amp_chk = simulate(0.1, 1.0)
    ok = abs(phi_chk - PHI_AT_BETA_ONE) / PHI_AT_BETA_ONE < 0.02
    print("自检 alpha=0.10 beta=1.00 → phi=%.5f（文献 %.5f，偏差 %.2f%%）%s"
          % (phi_chk, PHI_AT_BETA_ONE, 100.0 * abs(phi_chk - PHI_AT_BETA_ONE) / PHI_AT_BETA_ONE,
             "OK" if ok else "偏差过大！"))
    print("步长：oversample=%d  dt=%.3g  gamma·dt=%.4f" % (OVERSAMPLE, DT, GAMMA_REF * DT))

    # 单调性自检：phi 必须随 beta 单调递增（C 侧靠它反解 beta）
    for i, a in enumerate(alphas):
        vals = [v for v in phi_tab[i] if v > 0.0]
        if any(vals[j + 1] <= vals[j] for j in range(len(vals) - 1)):
            print("警告：alpha=%.2f 行的 phi 非单调，反解会不准" % a)

    print("音域（phi*gamma_ref → Hz）：alpha=%.2f 时 %.0f..%.0f Hz"
          % (alphas[-1], phi_tab[-1][0] * GAMMA_REF, phi_tab[-1][-1] * GAMMA_REF))

    if not write:
        for i, a in enumerate(alphas):
            print("alpha=%.2f phi=%s" % (a, " ".join("%.4f" % v for v in phi_tab[i])))
            print("          amp=%s" % " ".join("%.3f" % v for v in amp_tab[i]))
        return 0

    def arr(name, vals):
        return "static const float %s[%d] = {\n%s\n};\n" % (
            name, len(vals),
            "\n".join("    " + ", ".join("%.6ff" % v for v in vals[i:i + 6]) + ","
                      for i in range(0, len(vals), 6)))

    with open(OUT, "w") as f:
        f.write("// 由 tools/gen_bird_calib.py 生成 —— 请勿手工编辑。\n")
        f.write("// Mindlin–Laje 鸣管振荡器的离线标定表（本仓自测，非搬运数据）：\n")
        f.write("//   phi = f0/gamma（无量纲音高，随 beta 单调递增，C 侧据此反解 beta）\n")
        f.write("//   amp = peak2peak(y)/gamma（无量纲振幅，用于把响度与音高解耦）\n")
        f.write("// 测定条件：sr=%.0f Hz、gamma=%.0f、每采样 %d 步（gamma·dt=%.4f）、\n"
                % (SR, GAMMA_REF, OVERSAMPLE, GAMMA_REF * DT))
        f.write("//   半隐式欧拉（与运行时逐位一致）；alpha = 0.05 + 0.4·pressure。\n")
        f.write("// 自检：alpha=0.10,beta=1.00 时 phi=%.5f（文献 0.17061）。\n"
                % phi_chk)
        f.write("#ifndef __BIRD_CALIB_H__\n#define __BIRD_CALIB_H__\n\n")
        f.write("#define BIRD_CAL_ALPHA_N %d\n" % ALPHA_N)
        f.write("#define BIRD_CAL_BETA_N  %d\n\n" % BETA_N)
        f.write(arr("bird_cal_alpha", alphas))
        f.write("\n")
        f.write(arr("bird_cal_beta", betas))
        f.write("\n// phi[alpha][beta]：f0/gamma\n")
        f.write("static const float bird_cal_phi[BIRD_CAL_ALPHA_N][BIRD_CAL_BETA_N] = {\n")
        for row in phi_tab:
            f.write("    {" + ", ".join("%.6ff" % v for v in row) + "},\n")
        f.write("};\n\n")
        f.write("// amp[alpha][beta]：peak2peak(y)/gamma\n")
        f.write("static const float bird_cal_amp[BIRD_CAL_ALPHA_N][BIRD_CAL_BETA_N] = {\n")
        for row in amp_tab:
            f.write("    {" + ", ".join("%.6ff" % v for v in row) + "},\n")
        f.write("};\n\n")
        f.write("#endif // __BIRD_CALIB_H__\n")
    print("已写入 %s（%d x %d 两张表）" % (OUT, ALPHA_N, BETA_N))
    return 0


if __name__ == "__main__":
    sys.exit(main())
