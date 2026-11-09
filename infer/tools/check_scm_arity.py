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
#   1) 本地库/内建过程的**参数个数**（ARITY 表：表项均由 am_native_*.h 的签名注释与
#      实现体里的取参次数两头核对过）；
#   2) **本文件 define 的函数**的参数个数（local_arities：这才是 llm.scm 那次
#      `(log x y)` 事故能被准住的地方——`log` 是 App 自己 define 的）；
#   3) `cond` 每个子句必须是「测试 + 单个表达式」两元素；
#   4) 括号**计数**是否平衡（见下方「括号词法」）。
#
# 括号词法（必须与 animac_core.c 一致，否则误报）：
#   灵机的 '(' '[' '{' 都是同一个 LB token，')' ']' '}' 都是同一个 RB token，开闭
#   **可以混用**（`{` 还会额外插入一个 `begin` 关键字——见词法 2.2/2.4）。所以像
#   `(String.concat "a" "b" }`、`{ (f) )` 这种混用写法都是**合法源码**，不能按
#   「括号类型不匹配」报错；只有 RB 多余、LB 未闭合才算错。
#
# 用法：
#   python3 infer/tools/check_scm_arity.py infer/app/llm.scm infer/app/lib/llm_pure.scm
# 退出码：0 = 无问题；1 = 有问题（逐条打印行号）。
#
# 实现上的硬约束（2026-11 修复）：
#   本脚本曾被**合法源码**直接干爆 Python 解释器（glibc 报
#   `malloc(): unaligned tcache chunk detected` 后 SIGABRT + core dump），连带把
#   调用它的 harness 和 LLM 会话一起带走。原因是旧版 split_args 的 else 分支在遇到
#   **另一种闭括号**（`(… }` 里的 `}`、`{… )` 里的 `)`）时 while 条件立即为假、游标
#   原地不动，于是 `args.append((k, k, None))` 无限追加零长度参数，直到内存耗尽
#   （实测单个文件就能吃满 1.9GB RSS）。——而这两种写法按上面的词法都是**合法源码**，
#   即旧版是在合法输入上自杀的。因此现在：
#     · 解析器是**单趟 + 显式栈**，不再递归（旧版深嵌套输入会 RecursionError）；
#     · LB/RB 不按字符配对，只看计数（与词法一致，不误报合法源码）；
#     · 每个分支都保证游标**严格前进**，另有迭代次数上限做兜底；
#     · 行号用 bisect 在预计算换行表上取，避免 src.count() 的 O(n²)。
#
# 实测过的「多传参数」真相（原生与 wasm 同一份 VM）：
#   · 本地库函数：多传的值残留在操作数栈上（永久泄漏）；
#   · 自己 define 的 lambda：被调函数只拿到最后一个实参，多出的值同样泄漏；
#   · 少传：VM 直接报「指令执行异常」。
#   验证方法：在真 WASM 构建（nano_gui.wasm + 浏览器）里注入探针 App，用
#   (Diag.opstack) 读栈深：`(String.concat a b c)` → 结果为 b+c、栈深 +1；
#   `Ui.message` 传 2 个 → 栈深 +1；`cond` 子句写 3 个元素 → 第三个表达式静默不执行。
# ============================================================================

import bisect
import sys

# 名字 → 参数个数（不含过程名本身）。只收录"元数固定且值得盯"的过程；
# 未收录的名字一律跳过（例如用户自定义函数）。
ARITY = {
    # ---- 说明：本表只列「名字带库前缀」的本地库/内建过程。表项均由两头核对：
    #      ① am_native_*.h 的签名注释；② 实现体里的取参次数。
    #      本文件 define 的函数由 local_arities() 按文件推断，**不要**往这里写裸名
    #      （如 log / abs）——它们在 App 里通常被同名 define 遮蔽（llm.scm:141 就是）。
    # String 本地库
    'String.concat': 2, 'String.atom_to_string': 1, 'String.charCodeAt': 2,
    'String.slice': 3, 'String.fromCharCode': 1, 'String.length': 1,
    'String.equals': 2, 'String.parseNumber': 1, 'String.charAt': 2,
    # Math 本地库
    'Math.floor': 1, 'Math.ceil': 1, 'Math.round': 1, 'Math.sqrt': 1, 'Math.exp': 1,
    'Math.abs': 1, 'Math.pow': 2, 'Math.to_fixed': 2, 'Math.random': 0,
    'Math.bit_and': 2, 'Math.bit_or': 2, 'Math.bit_xor': 2, 'Math.bit_not': 1,
    'Math.bit_shl': 2, 'Math.bit_lshr': 2, 'Math.bit_ashr': 2,
    'Math.PI': 0, 'Math.log': 1, 'Math.log10': 1, 'Math.log2': 1,
    'Math.sin': 1, 'Math.cos': 1, 'Math.tan': 1, 'Math.atan': 1, 'Math.atan2': 2,
    # Table 本地库
    'Table.make': 0, 'Table.set': 3, 'Table.get': 2, 'Table.contains': 2, 'Table.delete': 2,
    'Table.keys': 1, 'Table.length': 1,
    # 内建过程
    'get_item': 2, 'set_item!': 3, 'cons': 2, 'car': 1, 'cdr': 1, 'length': 1,
    'push': 2, 'pop': 1, 'display': 1, 'newline': 0, 'not': 1,
    # Gfx 本地库
    'Gfx.width': 0, 'Gfx.height': 0, 'Gfx.clear': 0, 'Gfx.fill_white': 0,
    'Gfx.refresh': 0, 'Gfx.set_font': 1, 'Gfx.font_info': 1,
    'Gfx.draw_text': 6, 'Gfx.fill_rect': 7, 'Gfx.draw_rect': 7,
    # 注：set_pixel / draw_point 收的是 (x y r g b) 共 5 个（见 am_native_Gfx.h），
    # 旧表误写成 6，导致 sky*.scm / mandelbrot.scm 出现假阳性。
    'Gfx.set_pixel': 5, 'Gfx.draw_point': 5, 'Gfx.fill_circle': 6, 'Gfx.draw_circle': 6,
    'Gfx.draw_line': 7, 'Gfx.draw_line_aa': 8,
    'Gfx.image_size': 1, 'Gfx.draw_image': 5, 'Gfx.draw_image_refract': 9,
    'Gfx.sprite_info': 1, 'Gfx.draw_sprite': 3, 'Gfx.draw_sprite_rotated': 4,
    # Touch / Imu / Key / App 本地库
    'Touch.is_available': 0, 'Touch.is_pressed': 0, 'Touch.x': 0, 'Touch.y': 0,
    'Touch.take_press': 0, 'Touch.take_release': 0,
    # 注：Imu 的 Scheme 侧是 roll/pitch/yaw/gravity_x/gravity_y；
    # imu_read_angle 是 HAL 的 C 函数，**不是**本地库过程（旧表里那条是幽灵项）。
    'Imu.is_available': 0, 'Imu.roll': 0, 'Imu.pitch': 0, 'Imu.yaw': 0,
    'Imu.gravity_x': 0, 'Imu.gravity_y': 0,
    'Key.is_available': 0, 'Key.is_down': 1, 'Key.take_press': 0,
    'Key.take_release': 0, 'Key.dropped': 0,
    'App.exit': 0, 'App.exit_now': 0, 'App.hide': 0, 'App.back_requested': 0,
    'App.exit_with': 1, 'App.set_exit_button': 4, 'App.is_active': 0,
    'App.modal_closed': 0, 'App.resumed': 0, 'App.name': 0, 'App.platform': 0,
    # System 本地库
    'System.timestamp': 0, 'System.memstat': 0, 'System.gc': 0,
    'System.set_interval': 2, 'System.set_timeout': 2,
    'System.clear_interval': 1, 'System.clear_timeout': 1,
    'System.exit': 0, 'System.fork': 0, 'System.kill': 1, 'System.exec': 1,
    'System.eval': 1, 'System.make_queue': 1, 'System.write': 3, 'System.read': 2,
    'System.test': 1,
    # 本项目自实现的本地库（Time / Fs / Ui / Diag / LLM）
    'Time.utc_days': 0, 'Time.utc_sod': 0,
    'Fs.exists': 1, 'Fs.size': 1, 'Fs.read': 1, 'Fs.write': 2, 'Fs.append': 2,
    'Fs.list': 1, 'Fs.mkdir': 1, 'Fs.remove': 1, 'Fs.rename': 2, 'Fs.is_dir': 1,
    # Ui：message 只收正文，带标题的是 message_titled（两者容易写错，见 am_native_Ui.h）
    'Ui.input': 1, 'Ui.input_with': 2, 'Ui.message': 1, 'Ui.message_titled': 2,
    'Ui.confirm': 2,
    'LLM.init': 1, 'LLM.get_config': 0, 'LLM.get_param': 0, 'LLM.encode': 1,
    'LLM.decode': 1, 'LLM.matmul': 7,
    'Diag.ms': 0, 'Diag.tick': 0, 'Diag.opstack': 0, 'Diag.fstack': 0,
    'Diag.gcs': 0, 'Diag.timeslice': 0, 'Diag.pids': 0, 'Diag.pc': 0,
    'Diag.ilcode': 0, 'Diag.pgc': 0, 'Diag.heap': 0, 'Diag.frames': 0,
    'Diag.legacy': 0, 'Diag.slots': 0, 'Diag.freeslots': 0,
    'Diag.fillgeneric': 0, 'Diag.fastcl': 0,
}

# 裸算术/比较运算符**固定二元**（manual.md §6.2.4 / §6.2.5）。灵机编译器不检查参数
# 个数，多写一个参数（`(- la 1 i)`）不会报错，只会把操作数栈搞乱——症状是别处莫名其
# 妙地「[Runtime] 指令执行异常」，极难归因（本项目 2026-11 写大整数库时踩过一次）。
# 这类名字在实践中**不会**被 App 遮蔽，故只在「本文件没有 define / 当形参用过它」
# 时才检查（见 check() 的 local_names）。
ARITH_ARITY = {
    '+': 2, '-': 2, '*': 2, '/': 2, 'mod': 2, 'pow': 2,
    '>': 2, '<': 2, '>=': 2, '<=': 2, '==': 2, 'eq?': 2, 'eqv?': 2, 'equal?': 2,
    'not': 1, 'nan?': 1, 'number?': 1, 'list?': 1, 'null?': 1, 'atom?': 1,
    'undefined?': 1, 'typeof': 1, 'String.concat': 2, 'Math.floor': 1,
}

# 空白（含 BOM，否则文件开头的 BOM 会被当成第一个参数而误报）。
WS = ' \t\r\n\ufeff'
# 括号：LB = '(' '[' '{'，RB = ')' ']' '}' —— 与灵机词法一致，可混用
LB = '([{'
RB = ')]}'
# 原子（标识符/数字）的终止字符。`;` 也是终止符：`foo;注释` 里的注释不能被当成代码解析。
ATOM_STOP = set(WS) | set(LB) | set(RB) | {'"', ';'}
# 引用前缀：'x  `x  ,x —— 前缀与后面的数据算**一个**参数。
QUOTES = "'`,"


class Atom:
    """一个原子/字符串字面量。"""
    __slots__ = ('start', 'end', 'text')

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class Form:
    """一个括号形式（LB 开头的都算）。"""
    __slots__ = ('opener', 'start', 'end', 'children')

    def __init__(self, opener, start, end, children):
        self.opener = opener
        self.start = start
        self.end = end
        self.children = children


def make_line_index(src):
    """预计算换行表，返回 pos → 1 起始行号的函数（bisect，O(log n)）。"""
    newlines = [k for k, ch in enumerate(src) if ch == '\n']

    def line_of(pos):
        return bisect.bisect_left(newlines, pos) + 1

    return line_of


def parse_source(src):
    """解析整份源码 → (顶层形式列表, [(位置, 错误信息), ...])。

    单趟扫描 + 显式栈：**不递归**（深嵌套不会再触发 RecursionError），
    且每个分支都保证 i 严格前进（不会再出现旧版的无限追加）。
    """
    n = len(src)
    top = []
    # 栈帧：[开括号位置, 开括号字符, 起始位置(被引用时取引号位置), 子节点]
    stack = []
    errors = []
    pending_quote = None   # 尚未绑定到数据的引用前缀位置
    i = 0
    guard = 0

    def emit(node):
        (stack[-1][3] if stack else top).append(node)

    def start_pos(pos):
        return pending_quote if pending_quote is not None else pos

    def pop_frame(pend):
        """就地结束栈顶形式（pend = 结束位置）。"""
        fr = stack.pop()
        emit(Form(fr[1], fr[2], pend, fr[3]))

    while i < n:
        # 兜底：解析必须前进。真触发说明本函数有回归，宁可报错也不许吃光内存。
        guard += 1
        if guard > 4 * n + 64:
            errors.append((i, '内部错误：解析器未推进，已中止'))
            break

        c = src[i]

        if c == ';':                       # 行注释，跳到行尾
            j = src.find('\n', i)
            i = n if j < 0 else j
            continue

        if c in WS:                        # 空白
            i += 1
            continue

        if c in LB:                        # 开形式（( [ {）
            stack.append([i, c, start_pos(i), []])
            pending_quote = None
            i += 1
            continue

        if c in RB:                        # 闭形式（) ] } 三者等价，不按类型配对）
            pos = i
            i += 1
            if not stack:
                errors.append((pos, "多余的 '%s'（没有与之配对的开括号）" % c))
            else:
                pop_frame(pos)
            pending_quote = None
            continue

        if c in QUOTES:                     # 'x / `x / ,x
            if pending_quote is None:
                pending_quote = i
            i += 1
            continue

        if c == '"':                        # 字符串字面量（含转义）
            st = start_pos(i)
            j = i + 1
            closed = False
            while j < n:
                if src[j] == '\\':
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    closed = True
                    break
                j += 1
            if j > n:
                j = n
            if not closed:
                errors.append((i, '字符串字面量没有闭合的 "'))
            emit(Atom(st, j, src[i:j]))
            pending_quote = None
            i = j
            continue

        # 原子：src[i] 一定不是分隔符，所以 j > i，游标必然前进。
        st = start_pos(i)
        j = i
        while j < n and src[j] not in ATOM_STOP:
            j += 1
        emit(Atom(st, j, src[i:j]))
        pending_quote = None
        i = j
        continue

    # 文件结束时仍未闭合的形式
    while stack:
        fr = stack.pop()
        errors.append((fr[0], "'%s' 没有对应的闭括号" % fr[1]))
        emit(Form(fr[1], fr[2], n, fr[3]))

    return top, errors


def iter_forms(forms):
    """迭代遍历全部形式（不递归）。"""
    stack = list(forms)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, Form):
            for kid in node.children:
                if isinstance(kid, Form):
                    stack.append(kid)


def local_arities(forms):
    """扫出**本文件 define 出来的函数**及其参数个数。

    为什么需要：本工具最初就是为准住 `llm.scm` 里那次 `(log x y)` 事故写的，
    而那个 `log` 是 App **自己 define 的**（app/llm.scm:141，只收 1 个参数），
    任何写死的白名单都盖不到它。实测行为（原生 / wasm 同一份 VM）：多传的参数
    静默丢掉（被调函数只拿到最后一个），多出的值留在操作数栈上。

    只收录能确定的：
      · (define name (lambda (p …) …))  /  (define (name p …) …)
      · 同名多处 define 元数不一致 → 放弃该名
      · 参数表里有 . / … / 非标识符 → 放弃该名
      · 本文件里被 set! 改过 → 放弃该名
      · 本文件里该名字当过形参（可能遮蔽）→ 放弃该名
      · 名字已在 ARITY 表里 → 以 ARITY 为准
    """
    def params_ok(params, names_out):
        for p in params:
            if not isinstance(p, Atom) or p.text in ('.', '...') or p.text.startswith('#'):
                return False
            names_out.add(p.text)
        return True

    found = {}          # 名字 -> set(元数)
    bad_name = set()    # 已判定为不可用
    set_targets = set()
    shadowing = set()   # 当过形参的名字

    for node in iter_forms(forms):
        if not isinstance(node, Form) or node.opener == '{':
            continue
        kids = node.children
        head = kids[0].text if kids and isinstance(kids[0], Atom) else ''

        if head == 'lambda' and len(kids) >= 2 and isinstance(kids[1], Form):
            params_ok(kids[1].children, shadowing)
            continue

        if head == 'set!' and len(kids) >= 2 and isinstance(kids[1], Atom):
            set_targets.add(kids[1].text)
            continue

        if head != 'define' or len(kids) < 3:
            continue

        target = kids[1]
        if isinstance(target, Form):                     # (define (f a b) …)
            if not target.children or not isinstance(target.children[0], Atom):
                continue
            name = target.children[0].text
            params = target.children[1:]
        elif isinstance(target, Atom):                   # (define f (lambda (a b) …))
            name = target.text
            val = kids[2]
            if not (isinstance(val, Form) and val.opener != '{' and val.children
                    and isinstance(val.children[0], Atom) and val.children[0].text == 'lambda'):
                found.setdefault(name, set()).add(None)   # 非 lambda：不是函数
                continue
            if len(val.children) < 2 or not isinstance(val.children[1], Form):
                bad_name.add(name)
                continue
            params = val.children[1].children
        else:
            continue

        seen = set()
        if not params_ok(params, seen):
            bad_name.add(name)
            continue
        found.setdefault(name, set()).add(len(params))

    arity = {}
    for name, counts in found.items():
        if name in bad_name or name in set_targets or name in shadowing or name in ARITY:
            continue
        if None in counts:          # 既有函数值又有非函数值 → 不确定
            continue
        if len(counts) != 1:        # 同名多处 define 元数不一致
            continue
        arity[name] = next(iter(counts))
    return arity


def local_names(forms):
    """本文件里出现过的「自己定义的名字」——用于判断裸运算符是否被遮蔽。"""
    names = set()
    for node in iter_forms(forms):
        if not isinstance(node, Form) or node.opener == '{':
            continue
        kids = node.children
        head = kids[0].text if kids and isinstance(kids[0], Atom) else ''
        if head == 'define' and len(kids) >= 2 and isinstance(kids[1], Atom):
            names.add(kids[1].text)
        elif (head == 'lambda' and len(kids) >= 2 and isinstance(kids[1], Form)):
            for p in kids[1].children:
                if isinstance(p, Atom):
                    names.add(p.text)
    return names


def check(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        src = f.read()

    line_of = make_line_index(src)
    forms, parse_errors = parse_source(src)
    bad = [(line_of(pos), msg) for pos, msg in parse_errors]
    own = local_arities(forms)
    shadowed = local_names(forms)

    # 迭代式遍历（不递归），顺带做元数与 cond 子句形状检查
    stack = list(forms)
    while stack:
        node = stack.pop()
        if not isinstance(node, Form):
            continue
        kids = node.children
        head = kids[0].text if kids and isinstance(kids[0], Atom) else ''
        if node.opener != '{':      # '{' 被词法改写成 (begin …)，不是调用，不查元数
            want = ARITY.get(head)
            if want is None:
                want = own.get(head)
                src_tag = '（本文件 define）'
            else:
                src_tag = ''
            if want is None and head in ARITH_ARITY and head not in shadowed:
                want = ARITH_ARITY[head]
                src_tag = '（灵机算术/比较固定二元）'
            if want is not None and len(kids) - 1 != want:
                bad.append((line_of(node.start),
                            '%s 参数 %d 个，应为 %d%s' % (head, len(kids) - 1, want, src_tag)))
            elif head == 'cond':
                for cl in kids[1:]:
                    if isinstance(cl, Form) and len(cl.children) != 2:
                        bad.append((line_of(cl.start),
                                    'cond 子句有 %d 个元素，应为「测试 + 单个表达式」'
                                    % len(cl.children)))
        for kid in kids:
            if isinstance(kid, Form):
                stack.append(kid)

    bad.sort(key=lambda x: x[0])
    return bad


def main():
    if len(sys.argv) < 2:
        print('usage: check_scm_arity.py <file.scm>...')
        return 2
    total = 0
    for p in sys.argv[1:]:
        try:
            bad = check(p)
        except OSError as e:
            print('%s：读取失败（%s）' % (p, e))
            total += 1
            continue
        except Exception as e:      # 一个坏文件不该让整批检查崩掉
            print('%s：内部错误（%s: %s）' % (p, type(e).__name__, e))
            total += 1
            continue
        total += len(bad)
        print('%s：%d 处问题' % (p, len(bad)))
        for line, msg in bad:
            print('  第 %d 行：%s' % (line, msg))
    return 0 if total == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
