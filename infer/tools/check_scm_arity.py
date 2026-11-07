#!/usr/bin/env python3
# ============================================================================
# check_scm_arity.py —— 灵机（Animac）Scheme 源码的「调用元数 / cond 子句」静态检查
#
# 为什么需要它：
#   灵机编译器**不检查**调用的参数个数（manual.md §4.3：「参数个数与引数个数不符时，
#   结果是未规定的」）。多传的参数会残留在操作数栈上，于是错误的症状往往不是报错，
#   而是"某处莫名其妙的行为"——本仓库就踩过两次：
#     · `(log (String.concat "a" x) (String.concat "b" y))`（log 只收 1 个参数）→
#       多出的那个值留在栈上，之后 `cond` 分支里的 `gen_step` 再也没被调用；
#     · `(Gfx.draw_text (String.concat …) 10 20 255 255 255)` 括号数错位，参数被塞进
#       `String.concat`，「模型摘要」一行画不出来。
#   另外 `cond` 的每个子句**只接受单个表达式**（manual.md §4.2），多写一个表达式不会
#   报错，但那个表达式不会被执行。
#
# 检查内容：
#   1) 白名单函数/本地库/内建过程的**参数个数**（见 ARITY 表）；
#   2) `cond` 每个子句必须是「测试 + 单个表达式」两元素。
#
# 用法：
#   python3 infer/tools/check_scm_arity.py infer/app/llm.scm infer/app/lib/llm_pure.scm
# 退出码：0 = 无问题；1 = 有问题（逐条打印行号）。
# ============================================================================

import sys

# 名字 → 参数个数（不含过程名本身）。只收录"元数固定且值得盯"的过程；
# 未收录的名字一律跳过（例如用户自定义函数）。
ARITY = {
    # String 本地库
    'String.concat': 2, 'String.atom_to_string': 1, 'String.charCodeAt': 2,
    'String.slice': 3, 'String.fromCharCode': 1, 'String.length': 1,
    'String.equals': 2, 'String.parseNumber': 1, 'String.charAt': 2,
    # Math 本地库
    'Math.floor': 1, 'Math.ceil': 1, 'Math.round': 1, 'Math.sqrt': 1, 'Math.exp': 1,
    'Math.abs': 1, 'Math.pow': 2, 'Math.to_fixed': 2, 'Math.random': 0,
    'Math.bit_and': 2, 'Math.bit_or': 2, 'Math.bit_xor': 2, 'Math.bit_not': 1,
    'Math.bit_shl': 2, 'Math.bit_lshr': 2, 'Math.bit_ashr': 2,
    # Table 本地库
    'Table.make': 0, 'Table.set': 3, 'Table.get': 2, 'Table.contains': 2, 'Table.delete': 2,
    # 内建过程
    'get_item': 2, 'set_item!': 3, 'cons': 2, 'car': 1, 'cdr': 1, 'length': 1,
    'push': 2, 'pop': 1, 'display': 1, 'newline': 0, 'not': 1,
    # Gfx 本地库
    'Gfx.width': 0, 'Gfx.height': 0, 'Gfx.clear': 0, 'Gfx.fill_white': 0,
    'Gfx.refresh': 0, 'Gfx.set_font': 1, 'Gfx.font_info': 1,
    'Gfx.draw_text': 6, 'Gfx.fill_rect': 7, 'Gfx.draw_rect': 7,
    'Gfx.set_pixel': 6, 'Gfx.draw_point': 6, 'Gfx.fill_circle': 6, 'Gfx.draw_circle': 6,
    'Gfx.draw_line': 7,
    # Touch / Imu / Key / App 本地库
    'Touch.is_available': 0, 'Touch.is_pressed': 0, 'Touch.x': 0, 'Touch.y': 0,
    'Touch.take_press': 0, 'Touch.take_release': 0,
    'Imu.is_available': 0, 'Imu.read_angle': 0,
    'App.exit': 0, 'App.exit_now': 0, 'App.hide': 0, 'App.back_requested': 0,
    # System 本地库
    'System.timestamp': 0, 'System.memstat': 0, 'System.gc': 0,
    'System.set_interval': 2, 'System.set_timeout': 2,
    # 本项目自实现的本地库（Time / Fs / Ui / Diag / LLM）
    'Time.utc_days': 0, 'Time.utc_sod': 0,
    'Fs.exists': 1, 'Fs.size': 1, 'Fs.read': 1, 'Fs.write': 2, 'Fs.append': 2,
    'Fs.list': 1, 'Fs.mkdir': 1, 'Fs.remove': 1, 'Fs.rename': 2, 'Fs.is_dir': 1,
    'LLM.init': 1, 'LLM.get_config': 0, 'LLM.get_param': 0, 'LLM.encode': 1,
    'LLM.decode': 1, 'LLM.matmul': 7,
    'Diag.ms': 0, 'Diag.tick': 0, 'Diag.opstack': 0, 'Diag.fstack': 0,
}


def split_args(src, i):
    """从 src[i]（必须是 '(' 或 '{'）开始，返回 (结束位置, 参数区间列表)。"""
    n = len(src)
    closer = ')' if src[i] == '(' else '}'

    def skip_ws(k):
        while k < n:
            if src[k] == ';':
                while k < n and src[k] != '\n':
                    k += 1
            elif src[k] in ' \t\r\n':
                k += 1
            else:
                break
        return k

    def parse(k):
        opener = src[k]
        end_ch = ')' if opener == '(' else '}'
        k += 1
        args = []
        while True:
            k = skip_ws(k)
            if k >= n:
                return k, args
            if src[k] == end_ch:
                return k + 1, args
            if src[k] in "',`":
                k2 = skip_ws(k + 1)
                if k2 < n and src[k2] in '({':
                    k3, sub = parse(k2)
                    args.append((k, k3, sub))
                    k = k3
                elif k2 < n and src[k2] == '"':
                    k3 = k2 + 1
                    while k3 < n:
                        if src[k3] == '\\':
                            k3 += 2
                            continue
                        if src[k3] == '"':
                            k3 += 1
                            break
                        k3 += 1
                    args.append((k, k3, None))
                    k = k3
                else:
                    k3 = k2
                    while k3 < n and src[k3] not in ' \t\r\n(){}':
                        k3 += 1
                    args.append((k, k3, None))
                    k = k3
            elif src[k] in '({':
                k2, sub = parse(k)
                args.append((k, k2, sub))
                k = k2
            elif src[k] == '"':
                st = k
                k += 1
                while k < n:
                    if src[k] == '\\':
                        k += 2
                        continue
                    if src[k] == '"':
                        k += 1
                        break
                    k += 1
                args.append((st, k, None))
            else:
                st = k
                while k < n and src[k] not in ' \t\r\n(){}':
                    k += 1
                args.append((st, k, None))

    i += 1
    args = []
    while True:
        i = skip_ws(i)
        if i >= n:
            return i, args
        if src[i] == closer:
            return i + 1, args
        if src[i] in '({':
            j, sub = parse(i)
            args.append((i, j, sub))
            i = j
        elif src[i] == '"':
            st = i
            i += 1
            while i < n:
                if src[i] == '\\':
                    i += 2
                    continue
                if src[i] == '"':
                    i += 1
                    break
                i += 1
            args.append((st, i, None))
        elif src[i] in "',`":
            i += 1
        else:
            st = i
            while i < n and src[i] not in ' \t\r\n(){}':
                i += 1
            args.append((st, i, None))


def check(path):
    src = open(path, encoding='utf-8').read()
    n = len(src)
    bad = []

    def lineno(pos):
        # 只在**出错时**才算行号：对每个形式都切一次 src[:pos] 会把大文件拖成 O(n²)
        return src.count('\n', 0, pos) + 1

    def walk(i0, i1):
        i = i0
        while True:
            # 跳过空白与注释
            while i < i1:
                if src[i] == ';':
                    while i < i1 and src[i] != '\n':
                        i += 1
                elif src[i] in ' \t\r\n':
                    i += 1
                else:
                    break
            if i >= i1:
                return
            if src[i] in '({':
                j, args = split_args(src, i)
                head = src[args[0][0]:args[0][1]] if args else ''
                if src[i] == '(' and head in ARITY and len(args) - 1 != ARITY[head]:
                    bad.append((lineno(i), '%s 参数 %d 个，应为 %d'
                                % (head, len(args) - 1, ARITY[head])))
                if src[i] == '(' and head == 'cond':
                    for cl in args[1:]:
                        if cl[2] is None:
                            continue
                        if len(cl[2]) != 2:
                            bad.append((lineno(cl[0]),
                                        'cond 子句有 %d 个元素，应为「测试 + 单个表达式」'
                                        % len(cl[2])))
                for a in args:
                    if a[2] is not None:
                        walk(a[0], a[1])
                i = j
            else:
                i += 1

    walk(0, n)
    return bad


def main():
    if len(sys.argv) < 2:
        print(__doc__ or 'usage: check_scm_arity.py <file.scm>...')
        return 2
    total = 0
    for p in sys.argv[1:]:
        bad = check(p)
        total += len(bad)
        print('%s：%d 处问题' % (p, len(bad)))
        for line, msg in bad:
            print('  第 %d 行：%s' % (line, msg))
    return 0 if total == 0 else 1


if __name__ == '__main__':
    sys.exit(main())