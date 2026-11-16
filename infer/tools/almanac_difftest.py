#!/usr/bin/env python3
"""almanac_difftest.py —— 黄历 Scheme 移植版与 C 参考实现的差分回归。

原理：
  · C 侧黄金标准：tools/gen_almanac_tables.c 的 `ref` 模式（直接 include almanac.h，
    与 app/lib/almanac_data.scm 同源）。输出 25 个字段（含神煞名/等第等调试字段）。
  · Scheme 侧：tools/almanac_dump.scm 跑在桌面解释器（/home/bd4sur/Animac/main）或
    wasm32 探针（tools/e2e/wasm/run_wasm_probe.sh）里，输出同格式文本。
  · 两侧逐行 diff；任一字段不同即报错并打印日期与两边行。

用法：
  python3 tools/almanac_difftest.py                       # 全量：1901-01-01..2100-12-31 × 4 个时辰
  python3 tools/almanac_difftest.py --start 0 --count 365 --hours 0,12
  python3 tools/almanac_difftest.py --engine wasm --count 30
  python3 tools/almanac_difftest.py --jobs 8              # 分块并行（默认 os.cpu_count()）
"""
import argparse
import concurrent.futures as futures
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # infer/
ANIMAC = os.environ.get("ANIMAC_MAIN", "/home/bd4sur/Animac/main")
DUMP_TMPL = os.path.join(HERE, "almanac_dump.scm")
WORK = "/tmp/almanac_difftest"
C_BIN = "/tmp/gen_almanac_difftest"
FIELDS = ["date", "bazi", "overall", "week", "zodiac", "star28", "officer", "god",
          "mer", "meryi", "merji", "mernote", "good", "bad", "fly", "fetal", "peng",
          "hourluck", "dirs", "lunar", "nsn", "goodgods", "badgods", "lvl", "tlev"]


def daynum(y, m, d):
    yy = y - (1 if m <= 2 else 0)
    era = (yy if yy >= 0 else yy - 399) // 400
    yoe = yy - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def build_c():
    src = os.path.join(HERE, "gen_almanac_tables.c")
    if not os.path.exists(C_BIN) or os.path.getmtime(C_BIN) < os.path.getmtime(src):
        subprocess.run(["gcc", "-O2", "-I", ROOT, "-o", C_BIN, src], check=True)
    return C_BIN


def prepare_root():
    os.makedirs(WORK, exist_ok=True)
    app = os.path.join(WORK, "app")
    if not os.path.islink(app):
        os.symlink(os.path.join(ROOT, "app"), app)


def scheme_dump(start, count, hours, engine):
    """hours 元素可以是 int（分钟 0）或 (hour, minute) 元组。"""
    with open(DUMP_TMPL) as f:
        src = f.read()
    src = src.replace("START", str(start)).replace("COUNT", str(count))
    hh = [h[0] if isinstance(h, tuple) else h for h in hours]
    mm = [h[1] if isinstance(h, tuple) else 0 for h in hours]
    src = src.replace("'HOURS", "'(" + " ".join(str(h) for h in hh) + ")")
    src = src.replace("'MINS", "'(" + " ".join(str(m) for m in mm) + ")")
    driver = os.path.join(WORK, "dump_%d_%d_%s.scm" % (start, count, engine))
    with open(driver, "w") as f:
        f.write(src)
    if engine == "desk":
        cmd = [ANIMAC, driver]
    else:
        # wasm 探针的帧数必须给"够用但别太大"：程序跑完后事件循环靠帧数上限退出
        # （给 1e8 会让它在进程结束后空转数分钟）。实测每行 ≈ 200~250 帧。
        frames = count * len(hours) * 400 + 2000
        cmd = ["bash", os.path.join(HERE, "e2e", "wasm", "run_wasm_probe.sh"), driver,
               str(frames), "196608"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines()
             if ln and not ln.startswith("RESULT") and not ln.startswith("Module ")
             and not ln.startswith("test_")]
    if not lines or len(lines) != count * len(hours):
        print("Scheme 侧输出行数异常 rc=%d lines=%d expect=%d (start=%d count=%d hours=%s)"
              % (p.returncode, len(lines), count * len(hours), start, count, hours))
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        sys.exit(3)
    return lines


def c_ref(start, count, hour, minute=0):
    p = subprocess.run([C_BIN, "ref", str(start), str(count), str(hour), str(minute)],
                       capture_output=True, text=True, check=True)
    return [ln for ln in p.stdout.splitlines() if ln]


def run_chunk2(start, count, hours, minutes, engine):
    total = 0
    bad = 0
    reports = []
    for hour in hours:
        for minute in minutes:
            ref = c_ref(start, count, hour, minute)
            got = scheme_dump(start, count, [(hour, minute)], engine)
            for a, b in zip(ref, got):
                total += 1
                if a != b:
                    bad += 1
                    if len(reports) < 12:
                        ra = a.split("\t")
                        rb = b.split("\t")
                        diffs = [FIELDS[i] for i in range(min(len(ra), len(rb))) if ra[i] != rb[i]]
                        reports.append("MISMATCH %s fields=%s\n  ref: %s\n  got: %s"
                                       % (ra[0], diffs, a, b))
    return total, bad, reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=None, help="起始 daynum（默认 1901-01-01）")
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--hours", default="0,6,12,18")
    ap.add_argument("--minutes", default="0", help="分钟列表（土王用事按分钟取整，覆盖边界）")
    ap.add_argument("--engine", choices=["desk", "wasm"], default="desk")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=90, help="每块天数")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    build_c()
    prepare_root()
    full_start = daynum(1901, 1, 1)
    full_end = daynum(2101, 1, 1)
    start = full_start if args.start is None else args.start
    count = (full_end - full_start) if args.count is None else args.count
    hours = [int(x) for x in args.hours.split(",") if x != ""]
    minutes = [int(x) for x in args.minutes.split(",") if x != ""]

    chunks = [(s, min(args.chunk, start + count - s)) for s in range(start, start + count, args.chunk)]
    jobs = args.jobs or min(len(chunks), os.cpu_count() or 4)
    total = bad = 0
    done = 0
    with futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        fut = {ex.submit(run_chunk2, s, n, hours, minutes, args.engine): (s, n) for s, n in chunks}
        for f in futures.as_completed(fut):
            s, n = fut[f]
            t, b, reports = f.result()
            total += t
            bad += b
            done += n
            for r in reports:
                print(r)
            if not args.quiet:
                print("chunk %d..%d  lines=%d  bad=%d  progress=%d/%d"
                      % (s, s + n - 1, t, b, done, count), flush=True)
            if bad > 200:
                print("too many mismatches, abort")
                return 1
    print("TOTAL %d lines, %d mismatches" % (total, bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
