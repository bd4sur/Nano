#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""贪吃蛇 DQN 离线预训练（生成 app/lib/snake_dqn_weights.scm）

为什么需要这个脚本
==================
灵机 App `app/snake.scm` 的 AI 是**真·DQN**（Q 网络 + 经验回放 + 目标网络 + Adam），
整套前向/反向/更新都写在 Scheme 里（`app/lib/snake_dqn.scm`），在解释器里**在线**
继续学习与游玩。但解释器一个时间片只跑几十万条 VM 指令，从零练出"会玩"的策略需要
十万级环境步数（浏览器里几分钟）。为了让用户点开「AI」立刻看到一条会自己找食物的蛇，
本脚本用 numpy 做一次**离线预训练**，把学到的权重导出成 Scheme 模块
（`app/lib/snake_dqn_weights.scm`），App 启动时 import 它当初始权重。

离线与在线必须"同构"才能衔接：
  · 环境（场地大小/出生姿态/食物/奖励/观测编码）与 `app/lib/snake_core.scm` 逐条一致；
  · 网络结构（obs-NH-3、tanh 隐层、线性输出）与超参（γ/学习率/目标网络周期/ε 衰减）
    与 `app/lib/snake_dqn.scm` 一致；批量在离线端取大一些（32）以加快收敛，App 内在
    线微调用小批量（8）——批量只影响梯度噪声，不影响"同一个函数"这件事。

观测编码两版（`--feat`）：
  v1（16 维，经典 11 维 + 距离/连续量）
      危险(直行/右/左)、朝向 one-hot(4)、食物方向(左右上下)、食物 dx/dy、前方可走距离、
      长度占比、曼哈顿距离占比
  v2（28 维，**以蛇头为中心 5x5 旋转窗口** + 3）
      把局部局面按"前进方向朝上"旋转后取 5x5 邻域（每格：0 空 / 1 身体 / 2 墙 / 3 食物），
      再加食物在蛇体系的前后/左右偏移与长度占比。方向不变性直接由旋转窗口给出，
      比 v1 的"只看一格危险"多得多，策略明显更强（v1 均分 ~20，v2 均分 >40）。

用法
====
    python3 infer/tools/train_snake_dqn.py                      # 默认 v2 训练并导出
    python3 infer/tools/train_snake_dqn.py --episodes 8000 --nh 32
    python3 infer/tools/train_snake_dqn.py --check              # 只跑确定性对拍场景

`--check` 打印"固定食物序列 + 贪心策略"下的逐步动作与状态；Scheme 侧
`tools/e2e/native/snake_parity.scm`（工作区工具）用同一份权重跑同一场景逐行比对，
用来确认 Scheme 的前向/环境与 numpy 版同构。
"""

import argparse
import math
import os
import random
import time

import numpy as np

# ---- 环境常量（与 app/lib/snake_core.scm 一一对应）---------------------------
G = 12                      # 场地 12x12
NC = G * G                  # 144 格
NO = 3                      # 动作数：0=直行 1=右转 2=左转
NH = 32                     # 隐层宽度（导出文件里由 B1 长度决定，App 自适应）
WIN = 2                     # v2 窗口半径 → 5x5
NI_V1 = 16
NI_V2 = (2 * WIN + 1) ** 2 + 3
NI_V3 = NI_V2 + 2           # v3：v2 + 两个"可达空地数"（洪泛）

# 方向：0=上 1=右 2=下 3=左（顺时针）；右转 = (d+1)%4，左转 = (d+3)%4
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]

INIT_LEN = 3
INIT_HX, INIT_HY, INIT_DIR = 5, 6, 1        # 头在 (5,6)、朝右、身体 (4,6)(3,6)

R_FOOD = 10.0
R_DEATH = -10.0
R_STEP = -0.01
SHAPE = 0.1                 # 势函数 Φ = -SHAPE * manhattan(head, food)

# ---- DQN 超参（与 app/lib/snake_dqn.scm 一致的部分已注明）--------------------
GAMMA = 0.9                 # App 同
LR = 1e-3                   # App 同
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8   # App 同
TARGET_EVERY = 500          # App 同（环境步）
EPS_MIN = 0.03              # App 同
REPLAY = 20000              # 离线容量（App 内 4000，见 snake_dqn.scm）
BATCH = 32                  # 离线批量（App 内 8）
UPDATE_EVERY = 2            # App 同


def ni_of(feat):
    if feat == "v1":
        return NI_V1
    if feat == "v2":
        return NI_V2
    return NI_V3


# ============================================================================
# 环境（与 snake_core.scm 逐条对应）
# ============================================================================

class Snake:
    def __init__(self, seed=None, feat="v2"):
        self.rng = random.Random(seed)
        self.feat = feat
        self.occ = [0] * NC
        self.body = []          # 头在前：(x, y)
        self.hx = self.hy = 0
        self.dir = INIT_DIR
        self.food = 0
        self.score = 0
        self.steps = 0
        self.dead = 0
        self.reset()

    def reset(self):
        self.occ = [0] * NC
        self.body = []
        self.hx, self.hy, self.dir = INIT_HX, INIT_HY, INIT_DIR
        for k in range(INIT_LEN):
            x = self.hx - DX[self.dir] * k
            y = self.hy - DY[self.dir] * k
            self.body.append((x, y))
        for (x, y) in self.body:
            self.occ[y * G + x] = 1
        self.score = 0
        self.steps = 0
        self.dead = 0
        self.spawn_food()
        return self.obs()

    def spawn_food(self):
        # 与 Scheme 侧同策略：随机起址 + 线性探测，无空格则 -1
        start = self.rng.randrange(NC)
        for k in range(NC):
            i = (start + k) % NC
            if self.occ[i] == 0:
                self.food = i
                return
        self.food = -1

    def blocked(self, x, y):
        if x < 0 or y < 0 or x >= G or y >= G:
            return 1
        return self.occ[y * G + x]

    def obs(self):
        if self.feat == "v1":
            return self.obs_v1()
        if self.feat == "v2":
            return self.obs_v2()
        return self.obs_v3()

    def flood(self, x0, y0):
        """从 (x0,y0) 出发可达的空格数（含起点，若起点是空格）——蛇类游戏很强的信号：
        它直接告诉智能体"这一步之后还有多少活动空间"，是"别把自己关进笼子"的关键。"""
        if x0 < 0 or y0 < 0 or x0 >= G or y0 >= G:
            return 0
        if self.occ[y0 * G + x0]:
            return 0
        seen = bytearray(NC)
        q = [y0 * G + x0]
        seen[y0 * G + x0] = 1
        n = 0
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            n += 1
            cx = c % G
            cy = c // G
            for k in range(4):
                x = cx + DX[k]
                y = cy + DY[k]
                if x < 0 or y < 0 or x >= G or y >= G:
                    continue
                i = y * G + x
                if seen[i] or self.occ[i]:
                    continue
                seen[i] = 1
                q.append(i)
        return n

    def obs_v3(self):
        o = self.obs_v2()
        h = self.flood(self.hx, self.hy)
        ax = self.hx + DX[self.dir]
        ay = self.hy + DY[self.dir]
        a = self.flood(ax, ay)
        return o + [h / float(NC), a / float(NC)]

    def obs_v1(self):
        fx = self.food % G
        fy = self.food // G
        d = self.dir
        hx, hy = self.hx, self.hy
        ahead = self.blocked(hx + DX[d], hy + DY[d])
        right = self.blocked(hx + DX[(d + 1) % 4], hy + DY[(d + 1) % 4])
        left = self.blocked(hx + DX[(d + 3) % 4], hy + DY[(d + 3) % 4])
        free = 0
        x, y = hx, hy
        for _ in range(G):
            x += DX[d]
            y += DY[d]
            if self.blocked(x, y):
                break
            free += 1
        dx = fx - hx
        dy = fy - hy
        man = abs(dx) + abs(dy)
        o = [0.0] * NI_V1
        o[0] = 1.0 if ahead else 0.0
        o[1] = 1.0 if right else 0.0
        o[2] = 1.0 if left else 0.0
        o[3 + d] = 1.0
        if fx < hx:
            o[7] = 1.0
        if fx > hx:
            o[8] = 1.0
        if fy < hy:
            o[9] = 1.0
        if fy > hy:
            o[10] = 1.0
        o[11] = dx / G
        o[12] = dy / G
        o[13] = free / G
        o[14] = len(self.body) / float(NC)
        o[15] = man / float(2 * G)
        return o

    def obs_v2(self):
        """5x5 旋转窗口（前进方向朝上） + 食物在蛇体系的偏移 + 长度占比"""
        d = self.dir
        ri = (d + 1) % 4
        hx, hy = self.hx, self.hy
        fx = self.food % G
        fy = self.food // G
        o = [0.0] * NI_V2
        idx = 0
        for i in range(-WIN, WIN + 1):
            for j in range(-WIN, WIN + 1):
                x = hx + i * DX[d] + j * DX[ri]
                y = hy + i * DY[d] + j * DY[ri]
                if x < 0 or y < 0 or x >= G or y >= G:
                    v = 2.0
                elif x == fx and y == fy:
                    v = 3.0
                elif self.occ[y * G + x]:
                    v = 1.0
                else:
                    v = 0.0
                o[idx] = v / 3.0
                idx += 1
        dfx = fx - hx
        dfy = fy - hy
        df = dfx * DX[d] + dfy * DY[d]          # 食物在前方多远（蛇系前向分量）
        dr = dfx * DX[ri] + dfy * DY[ri]        # 食物在右手边多远（蛇系右向分量）
        o[idx] = df / G
        o[idx + 1] = dr / G
        o[idx + 2] = len(self.body) / float(NC)
        return o

    def step(self, a):
        """返回 (obs2, reward, done)"""
        if a == 1:
            self.dir = (self.dir + 1) % 4
        elif a == 2:
            self.dir = (self.dir + 3) % 4
        fx = self.food % G
        fy = self.food // G
        d_old = abs(fx - self.hx) + abs(fy - self.hy)
        nx = self.hx + DX[self.dir]
        ny = self.hy + DY[self.dir]
        self.steps += 1
        if nx < 0 or ny < 0 or nx >= G or ny >= G or self.occ[ny * G + nx]:
            self.dead = 1
            return self.obs(), R_DEATH, 1
        self.hx, self.hy = nx, ny
        self.occ[ny * G + nx] = 1
        self.body.insert(0, (nx, ny))
        r = R_STEP
        if ny * G + nx == self.food:
            self.score += 1
            r += R_FOOD
            self.spawn_food()
        else:
            tx, ty = self.body.pop()
            self.occ[ty * G + tx] = 0
        fx = self.food % G if self.food >= 0 else nx
        fy = self.food // G if self.food >= 0 else ny
        d_new = abs(fx - self.hx) + abs(fy - self.hy)
        r += GAMMA * (-SHAPE * d_new) - (-SHAPE * d_old)
        return self.obs(), r, 0


# ============================================================================
# Q 网络（MLP NI-NH-3：tanh 隐层 + 线性输出）
# ============================================================================

class Net:
    def __init__(self, rng, ni, nh=NH):
        self.ni = ni
        self.nh = nh
        s1 = 1.0 / math.sqrt(ni)
        s2 = 1.0 / math.sqrt(nh)
        self.W1 = rng.uniform(-s1, s1, size=(nh, ni))
        self.b1 = np.zeros(nh)
        self.W2 = rng.uniform(-s2, s2, size=(NO, nh))
        self.b2 = np.zeros(NO)

    def copy(self):
        n = Net(np.random.default_rng(0), self.ni, self.nh)
        n.W1 = self.W1.copy()
        n.b1 = self.b1.copy()
        n.W2 = self.W2.copy()
        n.b2 = self.b2.copy()
        return n

    def forward(self, x):
        h = np.tanh(x @ self.W1.T + self.b1)
        q = h @ self.W2.T + self.b2
        return q, h

    def q_one(self, x):
        h = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2

    def grads(self, x, h, q, a, target):
        B = x.shape[0]
        dq = np.zeros_like(q)
        rows = np.arange(B)
        dq[rows, a] = (q[rows, a] - target) * (2.0 / B)
        gW2 = dq.T @ h
        gb2 = dq.sum(axis=0)
        dh = dq @ self.W2
        dpre = dh * (1.0 - h * h)
        gW1 = dpre.T @ x
        gb1 = dpre.sum(axis=0)
        return gW1, gb1, gW2, gb2


class Adam:
    def __init__(self, net):
        self.m = [np.zeros_like(net.W1), np.zeros_like(net.b1),
                  np.zeros_like(net.W2), np.zeros_like(net.b2)]
        self.v = [np.zeros_like(net.W1), np.zeros_like(net.b1),
                  np.zeros_like(net.W2), np.zeros_like(net.b2)]
        self.t = 0

    def step(self, net, grads, lr=LR):
        self.t += 1
        ps = [net.W1, net.b1, net.W2, net.b2]
        bc1 = 1.0 - ADAM_B1 ** self.t
        bc2 = 1.0 - ADAM_B2 ** self.t
        for i in range(4):
            g = grads[i]
            self.m[i] = ADAM_B1 * self.m[i] + (1 - ADAM_B1) * g
            self.v[i] = ADAM_B2 * self.v[i] + (1 - ADAM_B2) * (g * g)
            mh = self.m[i] / bc1
            vh = self.v[i] / bc2
            ps[i] -= lr * mh / (np.sqrt(vh) + ADAM_EPS)


# ============================================================================
# 训练 / 评估
# ============================================================================

def train(args, ni):
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    env = Snake(seed=args.seed, feat=args.feat)
    net = Net(nprng, ni, args.nh)
    tgt = net.copy()
    adam = Adam(net)

    S = np.zeros((args.replay, ni))
    S2 = np.zeros((args.replay, ni))
    A = np.zeros(args.replay, dtype=np.int64)
    R = np.zeros(args.replay)
    D = np.zeros(args.replay)
    ptr, size = 0, 0

    eps = 1.0
    eps_decay = (1.0 - EPS_MIN) / max(1.0, args.eps_episodes)
    total = 0
    ep = 0
    best_eval = -1.0
    best_net = net.copy()
    scores_window = []
    t0 = time.time()

    while ep < args.episodes:
        s = env.reset()
        ep_len = 0
        while True:
            total += 1
            ep_len += 1
            if rng.random() < eps:
                a = rng.randrange(NO)
            else:
                a = int(np.argmax(net.q_one(np.array(s))))
            s2, r, done = env.step(a)
            S[ptr] = s
            A[ptr] = a
            R[ptr] = r
            S2[ptr] = s2
            D[ptr] = done
            ptr = (ptr + 1) % args.replay
            size = min(size + 1, args.replay)
            s = s2

            if total % UPDATE_EVERY == 0 and size >= args.batch:
                idx = nprng.integers(0, size, size=args.batch)
                q, h = net.forward(S[idx])
                q2, _ = tgt.forward(S2[idx])
                if args.double:
                    qn, _ = net.forward(S2[idx])
                    best_a = qn.argmax(axis=1)
                    best_next = q2[np.arange(args.batch), best_a]
                else:
                    best_next = q2.max(axis=1)
                target = R[idx] + GAMMA * (1.0 - D[idx]) * best_next
                grads = net.grads(S[idx], h, q, A[idx], target)
                adam.step(net, grads, args.lr)
            if total % args.target_every == 0:
                tgt = net.copy()

            if done or ep_len >= args.max_steps:
                break

        ep += 1
        eps = max(EPS_MIN, eps - eps_decay)
        scores_window.append(env.score)
        if len(scores_window) > 100:
            scores_window.pop(0)
        if ep % 200 == 0:
            print("[ep %5d] steps=%7d eps=%.3f score=%3d avg100=%.1f best=%3d (%.0fs)"
                  % (ep, total, eps, env.score, sum(scores_window) / len(scores_window),
                     max(scores_window), time.time() - t0), flush=True)
        if ep % args.eval_every == 0:
            ev = evaluate(net, args.eval_games, args.feat, seed=12345 + ep)
            print("   >>> eval %d games: mean=%.2f max=%d" % (args.eval_games, ev[0], ev[1]),
                  flush=True)
            if ev[0] > best_eval:
                best_eval = ev[0]
                best_net = net.copy()
                print("   >>> new best (mean %.2f)" % best_eval, flush=True)

    print("train done: episodes=%d steps=%d best_eval=%.2f" % (ep, total, best_eval))
    return best_net, best_eval, ep, total


def evaluate(net, games, feat, seed=999):
    rng = random.Random(seed)
    tot = 0
    mx = 0
    steps = 0
    for _ in range(games):
        env = Snake(seed=rng.randrange(1 << 30), feat=feat)
        s = env.obs()
        n = 0
        while True:
            s, r, done = env.step(int(np.argmax(net.q_one(np.array(s)))))
            n += 1
            if done or n >= 1500:
                break
        tot += env.score
        mx = max(mx, env.score)
        steps += n
    return tot / games, mx, steps / games


# ============================================================================
# 导出 Scheme 权重模块
# ============================================================================

def fmt(v):
    s = "%.9g" % float(v)
    if s == "-0":
        s = "0"
    return s


def emit_list(name, arr):
    flat = np.asarray(arr).reshape(-1)
    out = ["(define %s '(" % name]
    line = "  "
    for v in flat:
        tok = fmt(v) + " "
        if len(line) + len(tok) > 92:
            out.append(line.rstrip())
            line = "  "
        line += tok
    if line.strip():
        out.append(line.rstrip())
    out.append("))")
    return "\n".join(out)


def write_weights(path, net, meta):
    ni, nh = net.ni, net.nh
    parts = []
    parts.append(";@id=snake_dqn_weights")
    parts.append(";@title=贪吃蛇DQN权重")
    parts.append(";@version=%s" % meta["date"])
    parts.append(";; ============================================================================")
    parts.append(";; 贪吃蛇 DQN 预训练权重 —— **由 tools/train_snake_dqn.py 自动生成，请勿手工编辑**")
    parts.append(";;")
    parts.append(";; 生成：python3 infer/tools/train_snake_dqn.py --feat %s --nh %d ..." % (meta["feat"], nh))
    parts.append(";; 网络：%d-%d-%d（tanh 隐层 + 线性输出），扁平行主序：" % (ni, nh, NO))
    parts.append(";;   W1[j*NI+i] 第 j 个隐单元对第 i 个输入的权重；B1[j]；")
    parts.append(";;   W2[a*NH+j] 第 a 个动作对第 j 个隐单元的权重；B2[a]。")
    parts.append(";; 观测编码与奖励见 app/lib/snake_core.scm，训练/推理见 app/lib/snake_dqn.scm。")
    parts.append(";; 离线成绩：贪心策略 %d 局均分 %.2f、最高 %d（训练 %d 局 / %d 环境步）。"
                 % (meta["eval_games"], meta["eval_mean"], meta["eval_max"],
                    meta["episodes"], meta["steps"]))
    parts.append(";;")
    parts.append(";; App 载入后「训练」模式以它为起点继续在线学习；「从头学」按钮则丢弃它。")
    parts.append(";; ============================================================================")
    parts.append("")
    parts.append(emit_list("W1", net.W1))
    parts.append(emit_list("B1", net.b1))
    parts.append(emit_list("W2", net.W2))
    parts.append(emit_list("B2", net.b2))
    parts.append("")
    parts.append(";; 元数据：(训练局数 环境步数 评估均分 评估最高分)")
    parts.append("(define META '(%d %d %s %d))"
                 % (meta["episodes"], meta["steps"], fmt(meta["eval_mean"]), meta["eval_max"]))
    parts.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))


# ============================================================================
# 确定性对拍场景（供 Scheme 侧逐行比对）
# ============================================================================

FOOD_SEQ = [88, 100, 23, 140, 5, 61, 118, 34, 90, 12, 55, 7]


def check(net, feat):
    """确定性对拍：固定食物序列 + 贪心策略。

    打印格式与 Scheme 侧 tools/e2e/native/snake_parity.scm 逐字一致（q 保留 4 位、
    其余全是整数），便于直接 diff；`food` 一律是**注入后的序列食物**（吃到时不吃
    随机重撒的那个，否则两边无法逐行对齐）。
    """
    env = Snake(seed=0, feat=feat)
    env.food = FOOD_SEQ[0]
    s = env.obs()
    k = 0
    print("start food=%d hx=%d hy=%d len=%d" % (env.food, env.hx, env.hy, len(env.body)))
    for step in range(40):
        q = net.q_one(np.array(s))
        a = int(np.argmax(q))
        s, r, done = env.step(a)
        if not done and env.score > k:
            k += 1
            if k < len(FOOD_SEQ):
                env.food = FOOD_SEQ[k]
                s = env.obs()
        print("step %2d a=%d q=%.2f,%.2f,%.2f score=%d hx=%d hy=%d len=%d food=%d done=%d"
              % (step, a, q[0], q[1], q[2], env.score, env.hx, env.hy, len(env.body),
                 env.food, done))
        if done:
            break


def load_weights(path, ni, nh):
    """读回本脚本导出的 Scheme 权重模块（四个 '( ... ) 数字表 + META）"""
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8").read()

    def grab(name):
        i = txt.find("(define %s '" % name)
        if i < 0:
            return None
        i = txt.find("(", i + len(name) + 9)
        j = txt.find(")", i)
        return np.array([float(t) for t in txt[i + 1:j].split()])

    w1, b1, w2, b2 = grab("W1"), grab("B1"), grab("W2"), grab("B2")
    if w1 is None or b1 is None or w2 is None or b2 is None:
        return None
    net = Net(np.random.default_rng(0), ni, int(b1.size))
    if w1.size != net.nh * ni or w2.size != NO * net.nh:
        return None
    net.W1 = w1.reshape(net.nh, ni).copy()
    net.b1 = b1.copy()
    net.W2 = w2.reshape(NO, net.nh).copy()
    net.b2 = b2.copy()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=["v1", "v2", "v3"], default="v2")
    ap.add_argument("--nh", type=int, default=NH)
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--replay", type=int, default=REPLAY)
    ap.add_argument("--target-every", type=int, default=TARGET_EVERY)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--double", action="store_true")
    ap.add_argument("--eps-episodes", type=float, default=4000.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "app", "lib", "snake_dqn_weights.scm"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ni = ni_of(args.feat)

    if args.check:
        net = load_weights(args.out, ni, args.nh)
        if net is None:
            print("(--check: 权重文件不可用，改用随机初始化网络，仅供跑通流程)")
            net = Net(np.random.default_rng(args.seed), ni, args.nh)
        check(net, args.feat)
        return

    net, best_eval, episodes, steps = train(args, ni)
    mean, mx, st = evaluate(net, args.eval_games, args.feat, seed=4242)
    print("final eval: mean=%.2f max=%d steps=%.1f" % (mean, mx, st))
    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "feat": args.feat,
        "eval_games": args.eval_games,
        "eval_mean": mean,
        "eval_max": mx,
        "episodes": episodes,
        "steps": steps,
    }
    write_weights(os.path.abspath(args.out), net, meta)
    check(net, args.feat)


if __name__ == "__main__":
    main()
