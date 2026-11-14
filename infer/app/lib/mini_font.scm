;; ============================================================================
;; mini_font.scm —— 3×5「超小字体」（Mini Font）绘制库
;;
;; 来源：C 图形层的 gfx_draw_textline_mini（infer/graphics.c）。那是一套 3×5 像素
;; 的位图字模（数字 0-9、大写 A-Z、空格 . - : ( ) %，共 43 个字形），用来在
;; 320×240 这种小屏上给**大量小标注**配字：城市三字码、坐标读数、曲线当前值等。
;; 12px 正文字模在同类场景会糊成一片，且每字 12px 宽放不下 34 个城市标签。
;;
;; 为什么做成 Scheme 库而不是再加一个 native 过程：
;;   · 「灵机应用程序」是插件式的——app/ 下放一个 .scm 就多一个功能，固件零改动。
;;     把字模做成 Scheme 库，任何 App 一行 (import MF "app/lib/mini_font.scm")
;;     即可用，不必为它改 Gfx 本地库、重编固件；
;;   · 每字只需 ~5 次 Gfx.draw_point，一屏一百个字符也就几百次 native 调用
;;     （wasm 实测整屏 34 个城市标签 ≈0.7ms），不需要下沉到 C。
;;
;; 性能设计（2026-11，wasm 实测 109µs/字 → 约 20µs/字）：
;;   逐像素判位（每像素一次 Math.bit_and + 一次 Math.bit_shl）在本解释器里极贵——
;;   **本地库过程调用约 1µs/次**，一个字 15 个像素位就要 30 次调用。故字模在
;;   **模块加载时展开成两张表**（各 43 项）：
;;     advances[i] —— 字形 i 的推进宽度（宽+1）
;;     pixmaps[i]  —— 字形 i 的**点亮像素表**，扁平存 (dx0 dy0 dx1 dy1 …)
;;   绘制时只查表 + 调 Gfx.draw_point（约 0.2µs/次），不再做任何位运算。
;;   展开本身在启动时做一次（43 字形 × 15 位 ≈ 1ms）。
;;
;; 与 C 版的关系（逐条对齐 graphics.c 的 gfx_draw_textline_mini）：
;;   · 字形数据、列优先（低位在上的 5 位列掩码）编码、字间 1px 间隔完全一致；
;;   · 行高 5px，换行（\n）时回到起点 x、y 前进 6px；
;;   · 越界由 Gfx.draw_point 自己裁剪（C 版是遇到右边界 break，视觉等价）。
;; 三处有意的差异：
;;   1) **小写字母折叠为大写**（C 版直接跳过小写）。本库是"给人用的"绘制接口，
;;      调用方不必先转大写；
;;   2) 未收录字符按**空格**前进（C 版前进 3px 不画），保证任何字符串都能画出来；
;;   3) C 版用 mode 参数支持半透明（gfx_draw_textline_mini 的 mode=127），而
;;      Gfx.draw_point 是**置色**语义（无 alpha）。需要"暗一点"的文字时直接给暗色
;;      （黑底上 (127,127,127) 与白色 alpha 127 的观感等价），需要半透明叠色时
;;      由调用方先在底色上取色再传进来。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/list/map/apply；
;;   cond 每个子句只接受一个表达式；while 循环体不是独立词法区域（循环变量一律在
;;   lambda 体顶层 define）；模块内 define 的名字不带点号（`MF.xxx` 是导入方的引用写法）。
;;
;; 用法（导入方）：
;;   (import MF "app/lib/mini_font.scm")
;;   (MF.draw "PEK" 120 64 255 255 255)          ; 左上角 (120,64)，返回推进宽度
;;   (MF.draw_centered "123" 160 100 200 200 200); 以 (160,100) 为**中心**
;;   (MF.draw_right "34" 316 224 255 255 255)    ; 右对齐到 x=316
;;   (MF.width "PEK")  (MF.height)               ; 排版度量（墨迹宽 / 行高）
;; ============================================================================

(native Gfx)
(native Math)
(native String)

;; ---- 字模 ------------------------------------------------------------------
;; 每个字形 = (宽 列0 列1 … 列(宽-1))；列字节自下而上（bit0 = 最上一行）编码，
;; 与 graphics.c 的 mini_glyph 数组逐字节相同。顺序固定为：
;;   0-9 → 下标 0..9；A-Z → 10..35；空格 36；. 37；- 38；: 39；( 40；) 41；% 42。
(define glyphs
  '((3 31 17 31)        ; 0
    (3 18 31 16)        ; 1
    (3 29 21 23)        ; 2
    (3 21 21 31)        ; 3
    (3 15 8 31)         ; 4
    (3 23 21 29)        ; 5
    (3 31 21 29)        ; 6
    (3 1 1 31)          ; 7
    (3 31 21 31)        ; 8
    (3 23 21 31)        ; 9
    (3 30 5 30)         ; A
    (3 31 21 10)        ; B
    (3 14 17 17)        ; C
    (3 31 17 14)        ; D
    (3 31 21 17)        ; E
    (3 31 5 17)         ; F
    (3 14 17 29)        ; G
    (3 31 4 31)         ; H
    (3 17 31 17)        ; I
    (3 8 16 15)         ; J
    (4 31 4 10 17)      ; K
    (3 31 16 16)        ; L
    (5 31 2 4 2 31)     ; M
    (5 31 2 4 8 31)     ; N
    (3 14 17 14)        ; O
    (3 31 5 2)          ; P
    (5 14 17 21 9 22)   ; Q
    (3 31 5 26)         ; R
    (3 18 21 9)         ; S
    (3 1 31 1)          ; T
    (3 31 16 31)        ; U
    (3 15 16 15)        ; V
    (5 15 16 12 16 15)  ; W
    (3 27 4 27)         ; X
    (3 3 12 3)          ; Y
    (3 25 21 19)        ; Z
    (3 0 0 0)           ; 空格（占一个字形宽度，全空）
    (3 16 0 0)          ; .
    (3 4 4 4)           ; -
    (3 0 10 0)          ; :
    (3 0 14 17)         ; (
    (3 17 14 0)         ; )
    (5 19 11 4 26 25))) ; %


;; 行高（像素）。字形数据列高恒为 5。
(define line_h 5)

;; ---- 字模展开：推进宽度表 + 点亮像素表 -------------------------------------
;; 模块加载时执行一次（43 个字形的位运算总共约 1ms），此后绘制路径零位运算。
(define advances (cdr '(0)))
(define pixmaps (cdr '(0)))
(define _gi 0)
(define _gv 0)
(define _w 0)
(define _col 0)
(define _cb 0)
(define _j 0)
(define _px (cdr '(0)))
(while (< _gi 43) {
  (set! _gv (get_item glyphs _gi))
  (set! _w (get_item _gv 0))
  (push advances (+ _w 1))
  (set! _px (cdr '(0)))
  (set! _col 0)
  (while (< _col _w) {
    (set! _cb (get_item _gv (+ _col 1)))
    (set! _j 0)
    (while (< _j line_h) {
      ;; bit0 = 最上一行（与 C 版 (column_data >> j) & 1 一致）
      (if (not (== 0 (Math.bit_and _cb (Math.bit_shl 1 _j))))
          { (push _px _col) (push _px _j) }
          #f)
      (set! _j (+ _j 1))
    })
    (set! _col (+ _col 1))
  })
  (push pixmaps _px)
  (set! _gi (+ _gi 1))
})

;; ---- 字符 → 字形下标 -------------------------------------------------------
;; 返回 -1 表示字模未收录该字符（调用方按空格处理）。
(define glyph_index
  (lambda (code)
    (cond ((and (>= code 48) (<= code 57)) (- code 48))          ; 0-9
          ((and (>= code 65) (<= code 90)) (+ 10 (- code 65)))   ; A-Z
          ((and (>= code 97) (<= code 122)) (+ 10 (- code 97)))  ; a-z → 折叠为大写
          ((== code 32) 36)                                      ; 空格
          ((== code 46) 37)                                      ; .
          ((== code 45) 38)                                      ; -
          ((== code 58) 39)                                      ; :
          ((== code 40) 40)                                      ; (
          ((== code 41) 41)                                      ; )
          ((== code 37) 42)                                      ; %
          (else -1))))

;; ---- 度量 ------------------------------------------------------------------

;; (MF.height) : Number —— 行高（像素）。单行字形的墨迹高度。
(define height
  (lambda () line_h))

;; (MF.glyph code) : List | #null —— 字形原始数据 (宽 列0 列1 …)；未收录返回 #null。
;; 供需要自己排版/自绘的调用方使用（例如把字形放大成方块）；
;; 只要像素表时用下面的 (MF.pixels code)。
(define glyph
  (lambda (code) {
    (define idx (glyph_index code))
    (if (< idx 0) #null (get_item glyphs idx))
  }))

;; (MF.pixels code) : List —— 点亮像素表 (dx0 dy0 dx1 dy1 …)，相对字形左上角。
;; 未收录字符返回空表。
(define pixels
  (lambda (code) {
    (define idx (glyph_index code))
    (get_item pixmaps (if (< idx 0) 36 idx))
  }))

;; (MF.advance code) : Number —— 单个字符的推进宽度（字形宽 + 1px 字间）。
(define advance
  (lambda (code) {
    (define idx (glyph_index code))
    (get_item advances (if (< idx 0) 36 idx))
  }))

;; (MF.width str) : Number —— 单行字符串的**墨迹宽度**（= Σ推进 − 1，即不算
;; 末尾那 1px 字间）。含 \n 时按最长的一行算。空串为 0。
(define width
  (lambda (str) {
    (define n (String.length str))
    (define i 0)
    (define code 0)
    (define cur 0)
    (define best 0)
    (while (< i n) {
      (set! code (String.charCodeAt i str))
      (if (== code 10)
          {
            (if (> cur best) (set! best cur) #f)
            (set! cur 0)
          }
          (set! cur (+ cur (advance code))))
      (set! i (+ i 1))
    })
    (if (> cur best) (set! best cur) #f)
    (if (> best 0) (- best 1) 0)
  }))

;; ---- 绘制 ------------------------------------------------------------------

;; (MF.draw_char code x y r g b) : Number —— 在 (x,y) 画一个字符（左上角对齐），
;; 返回推进宽度。未收录字符按空格前进，不报错。
(define draw_char
  (lambda (code x y r g b) {
    (define idx (glyph_index code))
    (define gi (if (< idx 0) 36 idx))
    (define pi (get_item pixmaps gi))
    (define n (length pi))
    (define k 0)
    (while (< k n) {
      (Gfx.draw_point (+ x (get_item pi k)) (+ y (get_item pi (+ k 1))) r g b)
      (set! k (+ k 2))
    })
    (get_item advances gi)
  }))

;; (MF.draw str x y r g b) : Number —— 从 (x,y) 起画一行文本（左上角对齐），
;; 返回总推进宽度。支持 \n 换行（回到起点 x，y 前进 height+1）。
;; 字符串按码点遍历，越界由 Gfx.draw_point 裁剪。
(define draw
  (lambda (str x y r g b) {
    (define n (String.length str))
    (define i 0)
    (define code 0)
    (define px x)
    (define py y)
    (define adv 0)
    (define total 0)
    (while (< i n) {
      (set! code (String.charCodeAt i str))
      (if (== code 10)
          {
            (set! px x)
            (set! py (+ py (+ line_h 1)))
          }
          {
            (set! adv (draw_char code px py r g b))
            (set! px (+ px adv))
            (set! total (+ total adv))
          })
      (set! i (+ i 1))
    })
    total
  }))

;; (MF.draw_centered str cx y r g b) : Number —— 以 (cx, y) 为**行框中心**画一行
;; （y 仍是行框顶，与 Gfx.draw_text_centered 的参数习惯一致）。返回总推进宽度。
(define draw_centered
  (lambda (str cx y r g b)
    (draw str (- cx (/ (width str) 2)) y r g b)))

;; (MF.draw_right str rx y r g b) : Number —— 右对齐到 x = rx（rx 是墨迹右缘 + 1）。
(define draw_right
  (lambda (str rx y r g b)
    (draw str (- rx (width str)) y r g b)))
