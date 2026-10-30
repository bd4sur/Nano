;; ============================================================================
;; 触屏计算器（Touch Calculator）—— Nano「灵机应用程序」Scheme App
;;
;; 组成：
;;   · app/lib/infix.scm —— 中缀表达式**解析器**（lexer + parser + eval），
;;     逐字取自上游 Animac 回归用例 test/calculator.scm（仅改 import 路径、
;;     加固 eval 的失效路径、新增 validate/calc，见该文件头注释）；
;;   · 本文件 —— 计算器界面：键位布局、计算历史、触屏交互。**不实现任何表达式
;;     语法**，一切求值都走 app/lib/infix.scm（parse 与 eval 的唯一来源）。
;;
;; 功能：
;;   · 中缀输入：数字/小数点/四则/乘方 ^/取余 %/括号，一元负号，自动补右括号；
;;   · 函数：sin cos tan log exp sqrt（√ 键插入 "sqrt("），π 键插入数值常量；
;;   · Ans 键复用上次结果；AC 全清、DEL 退格（可长按连删）；
;;   · **计算历史**：显示区上 5 行为最近 5 次计算（先进先出、仅内存、不落盘、
;;     不持久化），最下一整行是**当前计算**（高亮条 + 亮字），输入过程中实时预览；
;;   · "=" 提交：把自动补齐括号后的表达式定稿、缓存结果供 Ans、置「可续算」态
;;     （随后按数字键用结果开新表达式，按运算符键则在结果上继续算）。
;;
;; 界面（320x240，键为矩形、无圆角；结果与历史均为 12px 文本，不使用数码管）：
;;      0 .. 98   显示区：6 行 x 15px（行 0..4 = 历史，最旧→最新；行 5 = 当前计算）
;;    100 .. 240  键区：6 列 x 5 行，每键 53x28
;;   键区整体位于 y>=100（键盘下半屏更好按）。注：2026-10 起灵机 App 运行态不再有
;;   4x4 宫格软按键（含右上角 esc 格）；退出由 app/lib/exit.scm 的硬 A/ESC 全局确认框
;;   负责（见 infer/am_native_App.h）。
;;
;; 健壮性设计（这是本 App 与上游用例的关键差异）：
;;   上游 calculator.scm 是批处理用例，输入固定且合法；交互式输入则可能残缺
;;   （"1+"、"sqrt"、"5/0"）。灵机没有异常机制，一旦虚拟机报错，**出错进程即被
;;   终止**，宿主还会把错误文字叠加到画面上。因此求值路径做了两层防护：
;;     1) app/lib/infix.scm 的 validate：按文法对词元序列做前置校验，杜绝 parser
;;        因「期待操作数却读到输入末尾」而越界取值（上游 parser 在这类输入上会
;;        直接运行时错误）；
;;     2) 加固后的 eval + calc：只对数字做运算，除数为 0 与非数字操作数都返回
;;        符号 'err 而非崩溃（NaN/inf 由界面层再行拦截）。
;;   本 App 因此可以在任意按键序列下持续运行，不会把进程跑死。
;;
;; 两处浮点相关平台约束（详见 doc/AGENTS.md 同名条目）：
;;   · 上游 lexer 的 read_number 只认数字与小数点、**不认指数形式**，所以「续算」
;;     与 Ans 键都优先用「把原表达式括号化」的方式，而不是把 %g 的
;;     "1.23457e+14" 塞回表达式（那会无法词法分析）；
;;   · wasm32 的解释器浮点载荷位宽小于 double（约 6 位有效数字），显示值末位可能
;;     与 64 位桌面宿主差一两档。
;;
;; 解释器约束（manual.md）：算术固定二参数、无一元取负（负数直接写字面量）、
;; 无 let/list/append/map；cond 每子句只接受单个表达式（多语句用花括号）；
;; / 与 mod 恒返回浮点（下标一律 Math.round/floor 收口）；块内只 set! 不 define
;; （热路径局部量用模块级临时变量，见 doc/AGENTS.md 的布局敏感槽位提示）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)

(import I "app/lib/infix.scm")
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 画布与布局 -----------------------------------------------------------

(define SW (Gfx.width))                  ; 320
(define SH (Gfx.height))                 ; 240
(define DISP_H 98)                       ; 显示区高度（键区从 100 起）
(define LINE_H 14)                       ; 行距（12px 字模 + 2px 间隙）
(define LINE_Y0 1)                       ; 首行文本顶端
(define HIST_MAX 5)                      ; 历史记录条数（显示区行 0..4）
(define CUR_BAR_Y 72)                    ; 当前计算行的高亮条
(define CUR_BAR_H 19)
(define COLS 6)
(define ROWS 5)
(define KEY_W (Math.floor (/ SW COLS)))  ; 53
(define KEY_H 28)
(define KEY_Y0 100)                       ; 5*28 = 140 -> 100+140 = 240
(define TICK_MS 33)                      ; 触屏采样节拍（约 30fps）

;; 文本显示：12px 字模。实测步进——ASCII/希腊/数学符号 7px，CJK 与全角 12px
;; （见 tools/e2e 的字体探针），故对齐/裁剪一律按**像素宽度**算，不能按字符数
(define CHAR_W 7)
(define CJK_W 12)
(define TEXT_MAXW (- SW 16))             ; 一行可用的像素宽度（左右各留 8px）

;; ---- 键盘表（行优先，6 列 x 5 行）----------------------------------------
;; 键面文字同时是动作的依据（见 press_key）：
;;   数字/./括号/%/^/-/+ → 原样追加；÷ → "/"，× → "*"；√ → "sqrt("；
;;   sin/cos/tan/log/exp → 追加同名函数名 + "("；π → 追加数值常量；其余为控制键。

(define KEY_LABELS '(
  "AC" "DEL" "("  ")"   "%"   "÷"
  "7"  "8"   "9"  "Ans" "π"   "×"
  "4"  "5"   "6"  "sin" "cos" "-"
  "1"  "2"   "3"  "tan" "log" "+"
  "0"  "."   "√"  "exp" "^"   "="))

(define KEY_N 30)

;; ---- 状态 -----------------------------------------------------------------

(define expr "")            ; 正在编辑的表达式文本（ASCII，供解析器用）
(define ans "")             ; 上次 "=" 的结果**显示串**（可能含指数形式）
(define ans_expr "")        ; 上次 "=" 的表达式原文（续算/Ans 用它，精度无损且可词法分析）
(define committed 0)        ; 1 = 刚按过 "="，当前行显示已提交的计算

;; 当前行（行 5）两段文本：line_l 亮字（表达式）、line_r 彩色尾段（值/提示）
(define line_l "")
(define line_r "")
(define line_kind 0)        ; 0-无 1-值(绿) 2-错误(红) 3-提示(灰)
(define pend_l "")          ; 已提交、仍占据当前行的计算（表达式部分）
(define pend_r "")          ; 同上（结果部分）

;; 计算历史：h0 最旧 … h4 最新（先进先出，仅内存）
(define h0 "")
(define h1 "")
(define h2 "")
(define h3 "")
(define h4 "")

(define dirty 1)            ; 1 = 需要重绘
(define prev_pressed #f)    ; 上一拍的触屏电平
(define press_k -1)         ; 按下时命中的键下标（-1 = 未命中键区）
(define press_t 0)          ; 按下时刻（ms，用于长按判定）
(define rep_t 0)            ; 上一次连发动作的时刻（ms）
(define repeated 0)         ; 本次按下已进入长按连发（松手时不再重复一次）

;; 模块级临时量（该解释器对块内 define 的槽位布局敏感，热路径一律 define 在此）
(define kr 0)               ; 键底色
(define kg 0)
(define kb 0)
(define cc 0)               ; 字符码点
(define cw "")              ; 临时字符串
(define now_ms 0)           ; 当前时刻（ms）

;; ===========================================================================
;; 文本工具
;; ===========================================================================

;; 单字符是否为 0..9
(define digit_char
  (lambda (ch) {
    (set! cc (String.charCodeAt 0 ch))
    (and (>= cc 48) (<= cc 57))
  }))

;; 单字符显示宽度：CJK/全角 12px，其余 7px
(define char_w
  (lambda (ch)
    ;; 11904 = U+2E80（CJK 部首区起始）：此码点以上按全角宽度计
    (if (>= (String.charCodeAt 0 ch) 11904) CJK_W CHAR_W)))

;; 一行文本的像素宽度
(define text_width
  (lambda (s) {
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (while (< i n) {
      (set! w (+ w (char_w (String.charAt s i))))
      (set! i (+ i 1))
    })
    w
  }))

;; 从左侧裁剪以适配 maxw 像素（保留尾部；一行放不下时用）
(define tail_to_fit
  (lambda (s maxw) {
    (define n (String.length s))
    (define i (- n 1))
    (define w 0)
    (define start n)
    (while (>= i 0) {
      (set! w (+ w (char_w (String.charAt s i))))
      (if (<= w maxw) (set! start i) { (set! i -1) })
      (set! i (- i 1))
    })
    (if (== start n) "" (String.slice s start n))
  }))

;; 展示用美化：* → ×、/ → ÷（只影响显示，表达式本体保持 ASCII 供解析器使用）
(define pretty
  (lambda (s) {
    (define n (String.length s))
    (define i 0)
    (define out "")
    (while (< i n) {
      (set! cw (String.charAt s i))
      (if (String.equals cw "*") (set! cw "×") #f)
      (if (String.equals cw "/") (set! cw "÷") #f)
      (set! out (String.concat out cw))
      (set! i (+ i 1))
    })
    out
  }))

;; 数字串能否直接回填进表达式：上游 lexer 的 read_number 只认数字与小数点，
;; 指数形式（"1.23457e+14"）无法词法分析
(define lexer_safe
  (lambda (s) {
    (define n (String.length s))
    (define i 0)
    (define ok #t)
    (while (and ok (< i n)) {
      (set! cc (String.charCodeAt 0 (String.charAt s i)))
      (if (or (and (>= cc 48) (<= cc 57))
              (== cc 46)
              (and (== cc 45) (== i 0)))
          #f
          { (set! ok #f) })
      (set! i (+ i 1))
    })
    (and ok (> n 0))
  }))

;; 历史槽位按下标取（0 最旧 … 4 最新）
(define hist_get
  (lambda (i)
    (cond ((== i 0) h0)
          ((== i 1) h1)
          ((== i 2) h2)
          ((== i 3) h3)
          (else h4))))

;; ===========================================================================
;; 绘制：历史记录行 + 当前计算行 + 矩形按键
;; ===========================================================================

;; 第 i 行文本的顶端 y
(define line_y (lambda (i) (+ LINE_Y0 (* i LINE_H))))

;; 居中一行键面文字
(define text_center
  (lambda (s x y w r g b) {
    (define len (String.length s))
    (define ox (Math.floor (/ (- w (* CHAR_W len)) 2)))
    (Gfx.draw_text s (+ x ox) y r g b)
  }))

;; 历史记录：行 0..4，最旧在上、最新在下（紧贴当前计算行），灰字
(define draw_history
  (lambda () {
    (define i 0)
    (define w 0)
    (while (< i HIST_MAX) {
      (set! cw (tail_to_fit (hist_get i) TEXT_MAXW))
      (if (> (String.length cw) 0)
          {
            (set! w (text_width cw))
            (Gfx.draw_text cw (- (- SW 8) w) (line_y i) 108 122 140)
          }
          #f)
      (set! i (+ i 1))
    })
  }))

;; 当前计算行：整行高亮条 + 「表达式（亮）+ 值/原因（绿/红/灰）」
(define draw_current
  (lambda () {
    (define wl 0)
    (define wr (text_width line_r))
    (define x0 0)
    (define r 236)
    (define g 244)
    (define b 252)
    (cond ((== line_kind 1) { (set! r 118) (set! g 228) (set! b 172) })   ; 结果
          ((== line_kind 2) { (set! r 255) (set! g 124) (set! b 104) })   ; 错误
          ((== line_kind 3) { (set! r 128) (set! g 142) (set! b 160) })   ; 提示
          (else #f))
    ;; 高亮条与上下描边（当前计算所在行）
    (Gfx.fill_rect 0 (- CUR_BAR_Y 1) SW 1 46 66 92)
    (Gfx.fill_rect 0 CUR_BAR_Y SW CUR_BAR_H 30 45 64)
    (Gfx.fill_rect 0 (+ CUR_BAR_Y CUR_BAR_H) SW 1 46 66 92)
    ;; 表达式过长时只留尾部，确保右侧的值/提示始终可见
    (set! cw (tail_to_fit line_l (- TEXT_MAXW wr)))
    (set! wl (text_width cw))
    (set! x0 (- (- SW 8) (+ wl wr)))
    (if (< x0 8) (set! x0 8) #f)
    (if (> (String.length cw) 0) (Gfx.draw_text cw x0 (+ CUR_BAR_Y 4) 238 245 252) #f)
    (if (> (String.length line_r) 0)
        (Gfx.draw_text line_r (+ x0 wl) (+ CUR_BAR_Y 4) r g b)
        #f)
  }))

;; 键面底色（按类别），结果写入 kr/kg/kb
(define key_color
  (lambda (lab)
    (cond ((String.equals lab "AC") { (set! kr 146) (set! kg 62) (set! kb 54) })
          ((String.equals lab "DEL") { (set! kr 78) (set! kg 70) (set! kb 62) })
          ((String.equals lab "=") { (set! kr 36) (set! kg 138) (set! kb 92) })
          ((String.equals lab "÷") { (set! kr 32) (set! kg 62) (set! kb 112) })
          ((String.equals lab "×") { (set! kr 32) (set! kg 62) (set! kb 112) })
          ((String.equals lab "-") { (set! kr 32) (set! kg 62) (set! kb 112) })
          ((String.equals lab "+") { (set! kr 32) (set! kg 62) (set! kb 112) })
          ((String.equals lab "^") { (set! kr 32) (set! kg 62) (set! kb 112) })
          ((String.equals lab "%") { (set! kr 32) (set! kg 62) (set! kb 112) })
          ((String.equals lab "(") { (set! kr 30) (set! kg 52) (set! kb 90) })
          ((String.equals lab ")") { (set! kr 30) (set! kg 52) (set! kb 90) })
          ((String.equals lab "Ans") { (set! kr 62) (set! kg 74) (set! kb 98) })
          (else { (set! kr 44) (set! kg 50) (set! kb 60) }))))

;; 画一个矩形键（1px 间隙、无圆角）
(define draw_key
  (lambda (k) {
    (define lab (get_item KEY_LABELS k))
    (define x (* (mod k COLS) KEY_W))
    (define y (+ KEY_Y0 (* (Math.floor (/ k COLS)) KEY_H)))
    (define fg_r 226)
    (define fg_g 232)
    (define fg_b 240)
    (key_color lab)
    (cond ((== k press_k) {
             (set! kr (+ kr 52))
             (set! kg (+ kg 52))
             (set! kb (+ kb 52))
           })
          (else #f))
    (if (or (String.equals lab "AC") (String.equals lab "DEL") (String.equals lab "="))
        { (set! fg_r 255) (set! fg_g 255) (set! fg_b 255) }
        #f)
    (Gfx.fill_rect (+ x 1) (+ y 1) (- KEY_W 2) (- KEY_H 2) kr kg kb)
    (text_center lab x (+ y 8) KEY_W fg_r fg_g fg_b)
  }))

;; 整屏重绘（只在状态变化时调用）
(define redraw
  (lambda () {
    (define i 0)
    (Gfx.fill_rect 0 0 SW SH 10 12 16)
    (Gfx.fill_rect 0 0 SW DISP_H 16 20 26)
    (draw_history)
    (draw_current)
    (set! i 0)
    (while (< i KEY_N) {
      (draw_key i)
      (set! i (+ i 1))
    })
      (EX.draw_button_at 2 2)   ; 顶部整块是显示区：按钮放左上角（表达式右对齐，很少顶到左边）
      (Gfx.refresh)
    (set! dirty 0)
  }))

;; ===========================================================================
;; 求值（全部经由 app/lib/infix.scm；本文件只有「结果如何呈现」的逻辑）
;; ===========================================================================

;; 统计未闭合的左括号数（"sqrt(" → 1），供 "=" 时自动补齐右括号
(define unclosed
  (lambda (s) {
    (define n (String.length s))
    (define i 0)
    (define depth 0)
    (while (< i n) {
      (set! cc (String.charCodeAt 0 (String.charAt s i)))
      (if (== cc 40) (set! depth (+ depth 1)) #f)
      (if (== cc 41) (if (> depth 0) (set! depth (- depth 1)) #f) #f)
      (set! i (+ i 1))
    })
    depth
  }))

;; 补右括号
(define autoclose
  (lambda (s) {
    (define n (unclosed s))
    (define i 0)
    (define out s)
    (while (< i n) {
      (set! out (String.concat out ")"))
      (set! i (+ i 1))
    })
    out
  }))

;; 数值 → 显示串；返回 "" 表示无法显示（NaN/溢出）
(define fmt
  (lambda (v)
    (if (nan? v) ""
        (if (> (Math.abs v) 1e300) ""
            (String.atom_to_string v)))))

;; 刷新当前行文本：
;;   已提交        → 「表达式 = 结果」
;;   输入中且有值  → 「表达式 = 预览值」
;;   输入中不完整  → 只显示表达式（不打扰；原因留到按 "=" 时给）
;;   按 "=" 失败   → 表达式 + 原因（红）
(define refresh_line
  (lambda (alert) {
    (define v 0)
    (define s "")
    (cond ((== committed 1)
           { (set! line_l pend_l)
             (set! line_r (String.concat " = " pend_r))
             (set! line_kind 1) })
          ((== (String.length expr) 0)
           { (set! line_l "")
             (set! line_r "输入表达式，按 = 计算")
             (set! line_kind 3) })
          (else
           {
             (set! v (I.calc expr))
             (set! line_l (pretty expr))
             (if (number? v)
                 {
                   (set! s (fmt v))
                   (if (== (String.length s) 0)
                       { (set! line_r (if (== alert 1) " = 非实数" ""))
                         (set! line_kind (if (== alert 1) 2 0)) }
                       { (set! line_r (String.concat " = " s))
                         (set! line_kind 1) })
                 }
                 { (set! line_r (if (== alert 1) (String.concat "  " (I.calc_reason v)) ""))
                   (set! line_kind (if (== alert 1) 2 0)) })
           }))
  }))

;; ===========================================================================
;; 计算历史（先进先出，仅内存）
;; ===========================================================================

(define hist_push
  (lambda (s) {
    (set! h0 h1)
    (set! h1 h2)
    (set! h2 h3)
    (set! h3 h4)
    (set! h4 s)
  }))

;; 结束「当前计算」：把它移入历史，腾出高亮行给下一次计算
(define close_pending
  (lambda () {
    (if (> (String.length pend_r) 0)
        { (hist_push (String.concat pend_l (String.concat " = " pend_r)))
          (set! pend_l "")
          (set! pend_r "") }
        #f)
    (set! committed 0)
  }))

;; ===========================================================================
;; 键动作
;; ===========================================================================

;; 追加文本。kind 1 = 「值起点」键（数字/函数/括号/常数）：若刚按过 "="，
;; 则另起一个新表达式；kind 0 = 运算符键：把上次的表达式括号化后继续算
;; （用原表达式而不是 %g 结果串，既无损又能被上游 lexer 识别）。
(define ins
  (lambda (s kind) {
    (if (== committed 1)
        {
          (close_pending)
          (if (and (== kind 0) (> (String.length ans_expr) 0))
              (set! expr (String.concat "(" (String.concat ans_expr ")")))
              (set! expr ""))
        }
        #f)
    (set! expr (String.concat expr s))
    (refresh_line 0)
    (set! dirty 1)
  }))

;; 小数点是否可以追加：当前数字段里已有 '.' 就不允许（"1.2.3" 会被上游 lexer 静默截断）
(define dot_ok
  (lambda () {
    (define n (String.length expr))
    (define i (- n 1))
    (define seen 0)
    (while (>= i 0) {
      (set! cc (String.charCodeAt 0 (String.charAt expr i)))
      (if (== cc 46) (set! seen 1) #f)
      (if (and (>= cc 48) (<= cc 57)) #f { (set! i -1) })
      (set! i (- i 1))
    })
    (== seen 0)
  }))

;; 小数点键：末尾不是数字时补一个 0（上游 lexer 只认"数字开头的数"，
;; 否则 ".5" / "0+." 会触发 "Unexpected character: ." 而整串解析失败）
(define do_dot
  (lambda () {
    (define n (String.length expr))
    (if (not (dot_ok))
        { (set! line_r "小数点重复") (set! line_kind 2) (set! dirty 1) }
        (if (and (== committed 0) (> n 0) (digit_char (String.charAt expr (- n 1))))
            (ins "." 1)
            (ins "0." 1)))
  }))

(define do_ac
  (lambda () {
    (close_pending)
    (set! expr "")
    (refresh_line 0)
    (set! dirty 1)
  }))

(define do_del
  (lambda () {
    (define n (String.length expr))
    (if (== committed 1) (close_pending) #f)
    (if (> n 0) (set! expr (String.slice expr 0 (- n 1))) #f)
    (refresh_line 0)
    (set! dirty 1)
  }))

;; "=" 提交：定稿表达式（补齐右括号）→ 求值 → 当前行改为「表达式 = 结果」，
;; 结果进 Ans、「可续算」态
(define do_eq
  (lambda () {
    (define e2 (autoclose expr))
    (define v 0)
    (define s "")
    (cond ((== committed 1) #f)          ; 已提交过：避免同一条重复入历史
          ((== (String.length expr) 0)
           { (set! line_l "")
             (set! line_r "请先输入表达式")
             (set! line_kind 2)
             (set! dirty 1) })
          (else
           {
             (set! v (I.calc e2))
             (if (number? v)
                 {
                   (set! s (fmt v))
                   (if (== (String.length s) 0)
                       { (set! line_l (pretty e2))
                         (set! line_r " = 非实数")
                         (set! line_kind 2)
                         (set! dirty 1) }
                       {
                         (set! expr e2)
                         (set! ans s)
                         (set! ans_expr e2)
                         (set! pend_l (pretty e2))
                         (set! pend_r s)
                         (set! committed 1)
                         (refresh_line 1)
                         (set! dirty 1)
                       })
                 }
                 { (set! line_l (pretty e2))
                   (set! line_r (String.concat "  " (I.calc_reason v)))
                   (set! line_kind 2)
                   (set! dirty 1) })
           }))
  }))

;; Ans 键要插入的文本：普通十进制结果直接用结果串本身；指数形式（lexer 无法
;; 词法分析）则回退为「上次表达式括号化」——等值且可解析
(define ans_text
  (lambda ()
    (if (== (String.length ans) 0) "0"
        (if (lexer_safe ans) ans
            (String.concat "(" (String.concat ans_expr ")"))))))

(define press_key
  (lambda (k) {
    (define lab (get_item KEY_LABELS k))
    (cond ((String.equals lab "AC") (do_ac))
          ((String.equals lab "DEL") (do_del))
          ((String.equals lab "=") (do_eq))
          ((String.equals lab "Ans") (ins (ans_text) 1))
          ((String.equals lab "π") (ins "3.14159265358979" 1))
          ((String.equals lab "√") (ins "sqrt(" 1))
          ((String.equals lab "sin") (ins "sin(" 1))
          ((String.equals lab "cos") (ins "cos(" 1))
          ((String.equals lab "tan") (ins "tan(" 1))
          ((String.equals lab "log") (ins "log(" 1))
          ((String.equals lab "exp") (ins "exp(" 1))
          ((String.equals lab "÷") (ins "/" 0))
          ((String.equals lab "×") (ins "*" 0))
          ((String.equals lab ".") (do_dot))
          (else (ins lab 1)))
  }))

;; ===========================================================================
;; 触屏：电平采样 + 自建边沿（按下记录命中的键，松手且仍在同一键才触发）
;; ===========================================================================

(define hit
  (lambda (x y) {
    (define c (Math.floor (/ x KEY_W)))
    (define r (Math.floor (/ (- y KEY_Y0) KEY_H)))
    (if (or (< x 0) (< y KEY_Y0) (>= c COLS) (>= r ROWS))
        -1
        (Math.round (+ (* r COLS) c)))
  }))

;; 长按参数（DEL 按住连续退格；其他键不连发）
(define HOLD_MS 450)        ; 按住多久后开始连发
(define REPEAT_MS 90)       ; 连发间隔

(define tick
  (lambda () {
    (EX.check)
    (define p (Touch.is_pressed))
    ;; 触点坐标在松开后仍保留最后一次按下期间的值（am_native_Touch 的缓存语义），
    ;; 故无条件据此求键位：松开判定要与按下时命中的同一个键比较。
    (define k (hit (Touch.x) (Touch.y)))
    (cond ((and p (not prev_pressed)) {
             (set! prev_pressed #t)
             (set! press_k k)
             (set! press_t (System.timestamp))
             (set! rep_t press_t)
             (set! repeated 0)
             (set! dirty 1)
           })
          ((and (not p) prev_pressed) {
             (set! prev_pressed #f)
             ;; 已经连发过的长按不再补一次动作（否则松手会多退一格）
             (if (and (>= press_k 0) (== k press_k) (== repeated 0))
                 (press_key press_k)
                 #f)
             (set! press_k -1)
             (set! dirty 1)
           })
          (else #f))
    ;; 长按 DEL：连续退格（now_ms 为模块级临时量，避免块内 define）
    (if (and p (== press_k 1))
        {
          (set! now_ms (System.timestamp))
          (if (and (> (- now_ms press_t) HOLD_MS) (> (- now_ms rep_t) REPEAT_MS))
              { (do_del) (set! rep_t now_ms) (set! repeated 1) }
              #f)
        }
        #f)
    (if (== dirty 1) (redraw) #f)
  }))

;; ===========================================================================
;; 启动
;; ===========================================================================

(refresh_line 0)
(redraw)
(System.set_interval TICK_MS tick)

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 redraw 擦掉对话框残影）
(EX.install_with redraw)
