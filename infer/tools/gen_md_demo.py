#!/usr/bin/env python3
"""gen_md_demo.py —— 把内置示例文档（app/lib/md_demo.scm）导出成真正的 .md 文件。

为什么需要这个脚本
------------------
Markdown 阅读器的示例文档只有**一处真源**：`infer/app/lib/md_demo.scm`（灵机的字符串
字面量不能跨行，所以那边写成"行数组"）。阅读器在**首次运行时**会把它落盘成
`<资源根>/doc/Markdown 示例.md`，但那有两个现实的坑：

  1. 用户去资源目录里找，很可能什么都看不到（还没运行过 App，或者只跑了别的 App）；
  2. 自动化测试会反复删除/重建 `doc/`，收尾清理后文件就没了。

所以把"导出"做成一个显式的构建期步骤：构建（或手工跑一次）之后，资源目录里
直接就有这份成品 .md，既能用编辑器打开看，也能被其它工具/App 使用。

单一真源原则：**不要手工编辑生成出来的 .md**，改示例请改 `md_demo.scm` 再重新导出。

用法
----
    python3 infer/tools/gen_md_demo.py                 # 写到默认资源根 /home/bd4sur/linglong
    python3 infer/tools/gen_md_demo.py /path/to/root   # 指定资源根（写到 <root>/doc/）
    NANO_LINGLONG=/path/to/root python3 ...            # 同上，环境变量写法
    python3 infer/tools/gen_md_demo.py --stdout        # 只打印到标准输出（不写文件）
    python3 infer/tools/gen_md_demo.py --src A B       # 显式指定真源与资源根（自测用）
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, '..', 'app', 'lib', 'md_demo.scm')
DEFAULT_ROOT = os.environ.get('NANO_LINGLONG', '/home/bd4sur/linglong')
DEMO_NAME = 'Markdown 示例.md'      # 与 app/markdown.scm 的 DEMO_PATH 保持一致

# 灵机字符串字面量的转义集合（manual §2.5）：仅 \" \\ \n \t \r，其它反斜杠序列按原样保留
_UNESCAPE = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\'}


def extract_strings(src_text):
    """从 md_demo.scm 里按出现顺序抽出 (define lines '(...)) 中的字符串字面量。"""
    # 只用 (define lines 之后的部分，避免把文件头的注释当成数据
    start = src_text.find('(define lines')
    if start < 0:
        raise SystemExit('md_demo.scm 里找不到 (define lines ...)')
    s = src_text[start:]

    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == ';':                        # 行注释：跳过（注意字符串里的 ';' 在下面处理）
            while i < n and s[i] != '\n':
                i += 1
            continue
        if c != '"':
            i += 1
            continue
        # 字符串字面量
        i += 1
        buf = []
        while i < n:
            c = s[i]
            if c == '\\':
                nxt = s[i + 1] if i + 1 < n else ''
                if nxt in _UNESCAPE:
                    buf.append(_UNESCAPE[nxt])
                    i += 2
                    continue
                # 未知转义：反斜杠与后续字符都按原样保留（与解释器一致）
                buf.append('\\')
                i += 1
                continue
            if c == '"':
                i += 1
                break
            buf.append(c)
            i += 1
        out.append(''.join(buf))
    return out


def main(argv):
    to_stdout = '--stdout' in argv
    src = DEFAULT_SRC
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--src' and i + 1 < len(argv):
            src = argv[i + 1]
            i += 2
            continue
        if not a.startswith('--'):
            args.append(a)
        i += 1
    root = args[0] if args else DEFAULT_ROOT

    with open(src, encoding='utf-8') as fp:
        lines = extract_strings(fp.read())
    if not lines:
        raise SystemExit('%s 里没抽到任何字符串字面量' % src)

    text = '\n'.join(lines) + '\n'
    if to_stdout:
        sys.stdout.write(text)
        return 0

    doc_dir = os.path.join(root, 'doc')
    os.makedirs(doc_dir, exist_ok=True)
    dst = os.path.join(doc_dir, DEMO_NAME)
    with open(dst, 'w', encoding='utf-8') as fp:
        fp.write(text)
    print('md_demo: %d 行 -> %s（%d 字节）' % (len(lines), dst, len(text.encode())))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))