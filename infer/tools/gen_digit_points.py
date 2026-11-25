#!/usr/bin/env python3
"""把 TrueType 数字字形采样成"粒子目标点"表，输出灵机（Animac）Scheme 模块。

用途：`infer/app/chronoglow.scm`（光粒时钟）的字形。粒子要"聚成 Arial 数字"，
就需要字形墨迹范围内的采样点 —— 运行时没有字体栅格化器（Gfx 也没有像素回读），
所以在**离线**把轮廓 flatten、按面密度均匀取点，粘成 Scheme 字面量。

输出：默认**直接内联**进 `infer/app/chronoglow.scm` 的 `BEGIN GDATA` / `END GDATA`
标记之间（App 是自包含的，仓库里没有额外的数据文件）；加 `--out 路径` 则改为写独立片段。
数据内容：
  GD_EM        字号（px，em 方框）
  GD_ADV       数字的 advance（px，等宽数字 → 时钟用固定步进）
  GD_CADV      冒号的 advance（px）
  GD_H         数字高度（px，以数字墨迹的上边缘为 0）
  GD_PTS       扁平点表 (x y ...)：x 以"笔位原点"为 0，y 以数字上边缘为 0
  GD_BASE      每个字符在 GD_PTS 里的起始下标（10 个数字 + ':'）
  GD_CNT       每个字符的点数

采样策略：**按面密度均匀取点**（`--step`，默认 2.9px 网格 + `--jitter 0.35` 抖动）。
  早期版本是"沿轮廓 2.2px 等距 + 内部 5.6px 稀疏网格"，结果轮廓点的面密度比内部高一个量级，
  粒子全挤在笔画边缘、中间发空 —— 观感是"描边"而不是"实心发光笔画"。
  需要更锐的边缘时可用 `--edge-step`（默认 0）额外沿轮廓加密。
点的顺序按"绕质心的极角"排序：字形切换时相邻粒子飞向相邻位置，过渡更顺。

用法：
  python3 infer/tools/gen_digit_points.py            # 用 Liberation Sans Bold（Arial 度量兼容）
  python3 infer/tools/gen_digit_points.py --font /path/to/Arial.ttf --em 64
  python3 infer/tools/gen_digit_points.py --preview /tmp/digits.png   # 同时出采样点预览图
"""
import argparse
import random
import math
import os
import sys

from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

CHARS = "0123456789:"
DEFAULT_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",   # Arial 度量兼容 · 粗体
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


class FlattenPen(BasePen):
    """把字形轮廓 flatten 成折线（三次/二次贝塞尔按弦高细分）。"""

    def __init__(self, glyphSet, tol=0.15):
        super().__init__(glyphSet)
        self.tol = tol
        self.contours = []
        self.cur = None

    def _moveTo(self, pt):
        self.cur = [pt]
        self.contours.append(self.cur)

    def _lineTo(self, pt):
        self.cur.append(pt)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        n = _cubic_steps(p0, p1, p2, p3, self.tol)
        for i in range(1, n + 1):
            t = i / n
            mt = 1.0 - t
            x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
            y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
            self.cur.append((x, y))

    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        n = _quad_steps(p0, p1, p2, self.tol)
        for i in range(1, n + 1):
            t = i / n
            mt = 1.0 - t
            x = (mt ** 2) * p0[0] + 2 * mt * t * p1[0] + (t ** 2) * p2[0]
            y = (mt ** 2) * p0[1] + 2 * mt * t * p1[1] + (t ** 2) * p2[1]
            self.cur.append((x, y))

    def _closePath(self):
        pass

    def _endPath(self):
        pass


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cubic_steps(p0, p1, p2, p3, tol):
    # 控制多边形的"弯折量"作为细分依据（够用且便宜）
    d = _dist(p0, p1) + _dist(p1, p2) + _dist(p2, p3) - _dist(p0, p3)
    return max(2, min(64, int(math.ceil(math.sqrt(max(d, 0.0) / max(tol, 1e-6))))))


def _quad_steps(p0, p1, p2, tol):
    d = _dist(p0, p1) + _dist(p1, p2) - _dist(p0, p2)
    return max(2, min(64, int(math.ceil(math.sqrt(max(d, 0.0) / max(tol, 1e-6))))))


def resample_polyline(pts, step, closed):
    """把折线按弧长等距重采样（间距 ≈ step），避免原控制点疏密不均。"""
    if len(pts) < 2:
        return []
    seq = list(pts) + ([pts[0]] if closed else [])
    segs = []
    total = 0.0
    for i in range(len(seq) - 1):
        L = _dist(seq[i], seq[i + 1])
        segs.append(L)
        total += L
    if total <= 1e-9:
        return [seq[0]]
    n = max(2, int(round(total / step)))
    out = []
    acc = 0.0
    si = 0
    for k in range(n):
        target = total * k / n
        while si < len(segs) - 1 and acc + segs[si] < target:
            acc += segs[si]
            si += 1
        t = (target - acc) / segs[si] if segs[si] > 1e-9 else 0.0
        x = seq[si][0] + (seq[si + 1][0] - seq[si][0]) * t
        y = seq[si][1] + (seq[si + 1][1] - seq[si][1]) * t
        out.append((x, y))
    return out


def point_in_contours(pt, contours):
    """射线法 + 奇偶规则（字形轮廓是 even-odd 语义）。"""
    x, y = pt
    inside = False
    for c in contours:
        n = len(c)
        for i in range(n):
            x1, y1 = c[i]
            x2, y2 = c[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if xin > x:
                    inside = not inside
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default=None)
    ap.add_argument("--em", type=float, default=66.0, help="字号（em，px）；App 内数据区用 66")
    ap.add_argument("--step", type=float, default=2.7,
                    help="面密度采样步长（px）：在字形墨迹内部按等比网格取点，"
                         "再按 --jitter 抖动，使粒子在笔画面积内**均匀分布**")
    ap.add_argument("--jitter", type=float, default=0.35,
                    help="抖动幅度（×step）：打散网格感，0 = 严格网格")
    ap.add_argument("--edge-step", type=float, default=0.0,
                    help="实心风格里额外沿轮廓加密的步长（0 = 关闭，默认）")
    ap.add_argument("--outline-step", type=float, default=2.4, help="描边风格：沿轮廓等距采样步长（px）")
    ap.add_argument("--fill-step", type=float, default=5.6, help="描边风格：内部稀疏网格步长（px）")
    ap.add_argument("--out", default=None,
                    help="写成独立片段文件（不给则内联进 --splice 指定的 App）")
    ap.add_argument("--splice", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", "app", "chronoglow.scm"),
                    help="要内联的目标 App（替换其中 BEGIN GDATA..END GDATA 区段）")
    ap.add_argument("--preview", default=None)
    args = ap.parse_args()

    font_path = args.font
    if font_path is None:
        for f in DEFAULT_FONTS:
            if os.path.exists(f):
                font_path = f
                break
    if not font_path or not os.path.exists(font_path):
        sys.exit("找不到字体文件，请用 --font 指定")
    font = TTFont(font_path, fontNumber=0)
    gs = font.getGlyphSet()
    upem = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]

    scale = args.em / upem
    adv = {}
    for ch in CHARS:
        gname = cmap[ord(ch)]
        adv[ch] = hmtx[gname][0] * scale

    # 数字的统一上边缘（lining figures 通常与 cap height 齐平）：取 0..9 的 yMax
    top = -1e9
    for ch in "0123456789":
        g = gs[cmap[ord(ch)]]
        top = max(top, g.width * 0 + _glyph_y_max(gs, cmap[ord(ch)]))
    top_px_h = top * scale

    def sample(style):
        """返回 (flat, base, cnt, sets)：
        style='solid'   —— 面密度均匀（笔画内等比网格 + 抖动）
        style='outline' —— 轮廓密（沿周长等距）+ 内部稀疏网格（旧风格，"描边"观感）
        """
        flat, bases, cnts, sets = [], [], [], []
        for ch in CHARS:
            gname = cmap[ord(ch)]
            pen = FlattenPen(gs, tol=0.12)
            gs[gname].draw(pen)
            contours = []
            for c in pen.contours:
                pc = [((x) * scale, (top_px_h - y * scale)) for (x, y) in c]
                if len(pc) >= 2:
                    contours.append(pc)
            xs = [q[0] for c in contours for q in c]
            ys = [q[1] for c in contours for q in c]
            pts = []
            if xs:
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                if style == "outline":
                    for c in contours:
                        pts.extend(resample_polyline(c, args.outline_step, closed=True))
                    step = args.fill_step
                else:
                    step = args.step
                rnd = random.Random(1234)
                gy = y0 + step * 0.5
                while gy < y1:
                    gx = x0 + step * 0.5
                    while gx < x1:
                        jx = (rnd.random() - 0.5) * args.jitter * step
                        jy = (rnd.random() - 0.5) * args.jitter * step
                        px, py = gx + jx, gy + jy
                        if point_in_contours((px, py), contours):
                            pts.append((px, py))
                        gx += step
                    gy += step
            # 按绕质心极角排序：字形切换时相邻粒子去相邻位置
            cx = sum(q[0] for q in pts) / max(len(pts), 1)
            cy = sum(q[1] for q in pts) / max(len(pts), 1)
            pts.sort(key=lambda q: (math.atan2(q[1] - cy, q[0] - cx), _dist(q, (cx, cy))))
            bases.append(len(flat) // 2)
            cnts.append(len(pts))
            for (px, py) in pts:
                flat.append(round(px * 2) / 2)
                flat.append(round(py * 2) / 2)
            sets.append(pts)
        return flat, bases, cnts, sets

    solid_flat, solid_base, solid_cnt, solid_sets = sample("solid")
    out_flat, out_base, out_cnt, out_sets = sample("outline")
    for ch, a_, b_, c_, d_ in zip(CHARS, solid_cnt, out_cnt, solid_base, out_base):
        print(f"  '{ch}': 实心={a_:4d} 点  描边={b_:4d} 点")

    def emit(prefix, flat, bases, cnts):
        out = []
        out.append(f"(define {prefix}_BASE '(" + " ".join(str(v) for v in bases) + "))")
        out.append(f"(define {prefix}_CNT '(" + " ".join(str(v) for v in cnts) + "))")
        out.append(f"(define {prefix}_PTS '(")
        line = "  "
        for i in range(0, len(flat), 2):
            tok = f"{_num(flat[i])} {_num(flat[i + 1])} "
            if len(line) + len(tok) > 96:
                out.append(line.rstrip())
                line = "  "
            line += tok
        if line.strip():
            out.append(line.rstrip())
        out.append("))")
        return "\n".join(out) + "\n"

    lines = []
    lines.append(";; 由 infer/tools/gen_digit_points.py 自动生成，请勿手工编辑。")
    _extra = f"　轮廓加密={args.edge_step}px" if args.edge_step > 0 else ""
    lines.append(f";; 字体：{os.path.basename(font_path)}　em={args.em}px")
    lines.append(f";; 实心：面密度步长={args.step}px 抖动={args.jitter}×step" + _extra)
    lines.append(f";; 描边：轮廓步长={args.outline_step}px + 内部网格={args.fill_step}px（稀疏内部）")
    lines.append(";; 数据用途见 app/chronoglow.scm 文件头；改字体/字号直接重跑本脚本即可（自动替换该区段）。")
    lines.append("")
    lines.append(f"(define GD_EM {_num(args.em)})")
    lines.append(f"(define GD_ADV {_num(adv['0'])})")
    lines.append(f"(define GD_CADV {_num(adv[':'])})")
    lines.append(f"(define GD_H {_num(top_px_h)})")
    lines.append("")
    lines.append(";; ---- 实心风格（面密度均匀：粒子铺满笔画内部）----")
    lines.append(emit("GD", solid_flat, solid_base, solid_cnt).rstrip("\n"))
    lines.append("")
    lines.append(";; ---- 描边风格（轮廓密 + 内部稀疏：粒子勾出笔画边缘）----")
    lines.append(emit("GO", out_flat, out_base, out_cnt).rstrip("\n"))
    block = "\n".join(lines) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fp:
            fp.write(block)
        print(f"写出片段 {args.out}")
    else:
        tgt = os.path.abspath(args.splice)
        src = open(tgt).read()
        b0 = src.find(";; ==== BEGIN GDATA ====")
        b1 = src.find(";; ==== END GDATA ====")
        if b0 < 0 or b1 < 0 or b1 < b0:
            sys.exit(f"{tgt} 里找不到 BEGIN/END GDATA 区段标记")
        head = src[:b0] + ";; ==== BEGIN GDATA ====\n"
        tail = ";; ==== END GDATA ====" + src[b1 + len(";; ==== END GDATA ===="):]
        with open(tgt, "w") as fp:
            fp.write(head + block + tail)
        print(f"已内联进 {tgt}（替换 BEGIN/END GDATA 之间的内容）")
    pts_flat = solid_flat
    cnt = solid_cnt

    total = len(pts_flat) // 2
    print(f"共 {total} 个点（{len(block) / 1024:.1f} KB 文本），"
          f"单字最多 {max(cnt)} 点（MAXPT），六字典型 {sum(cnt[d] for d in (0,6,3,9,3,9))} 点")

    if args.preview:
        _preview(preview_sets, args, top_px_h)


def _glyph_y_max(gs, gname):
    """字形 yMax（字体单位）：用 boundsPen 太重，这里扫一遍轮廓控制点即可。"""
    g = gs[gname]
    pen = FlattenPen(gs, tol=0.5)
    g.draw(pen)
    ys = [p[1] for c in pen.contours for p in c]
    return max(ys) if ys else 0


def _num(v):
    """紧凑数字字面量：整数就写整数。"""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def _preview(sets, args, h):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("（未安装 Pillow，跳过预览图）")
        return
    S = 2
    W, H = 340, 80
    img = Image.new("RGB", (W * S, H * S), (8, 10, 16))
    d = ImageDraw.Draw(img)
    x = 6.0
    for ch, pts in zip(CHARS, sets):
        a = args.em * 0.556 if ch != ":" else args.em * 0.333
        for (px, py) in pts:
            d.point((int((x + px) * S), int((14 + py) * S)), fill=(140, 230, 255))
        x += a
    img.save(args.preview)
    print(f"预览图：{args.preview}")


if __name__ == "__main__":
    main()
