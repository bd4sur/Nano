;; ============================================================================
;; 海龟绘图 + L-system —— 灵机应用程序（"解释器里跑解释器"）
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;;
;; 这个 App 里跑着两台解释器：
;;   1) 外层：灵机，执行本文件（Scheme）；
;;   2) 内层：本文件用 Scheme 写的一个小型 L-system / Logo 解释器——
;;      ① 生成文法：按规则表把公理逐代重写成一条长符号串（基因组）；
;;      ② 海龟状态机：逐字符执行 F G f + - [ ] | 八条指令，驱动画笔。
;;      内层"程序"就是重写出来的符号串本身。
;;
;; 为什么适合灵机：解释器由宿主按时间片协作调度，因此内层指令流的执行位置
;; （cursor）与海龟状态（位置/方向/栈）都能存进模块变量，每个 tick 推进
;; 若干条指令——等价于一台可暂停/可续跑的虚拟机，实现代价为 0。
;;
;; 取景：先"空跑一遍"（不画线、只记录包围盒）算出缩放入参，再从头画，
;;       所以任何预设都能自动铺满画布，无需手调常量。
;;
;; 成本（host x86-64 实测）：整套 App 跑 4000 帧（6 个预设各一轮）墙钟 0.76 s；
;;       单个预设含"重写+完整生长"共 60~120 ms；单 tick（220 符号）≈2.4 ms，
;;       相对 40 ms 定时器周期余量充足（若某代重写超过时间片，宿主会自动切帧，
;;       这正是协作调度的用处）。
;;
;; 本解释器的两个书写约定（本次踩坑，务必遵守）：
;;   1) 算术一律二元嵌套：(+ a b c) 只会发射一条 add，多出来的操作数滞留在
;;      操作数栈上并污染后续表达式 —— 例如 (+ 1 (- 100 20 5)) 会算成 115。
;;   2) cond 子句只能带一个表达式，且子句多了会编译失败（无诊断输出）；
;;      热路径统一用平铺 if / { } 块。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 画布与调度参数 --------------------------------------------------------

(define W (Gfx.width))
(define H (Gfx.height))
(define MARGIN 8)           ; 图形左右留白（像素）
(define HUD_TOP 34)         ; 顶部 HUD 带高度（绘图区从这之下开始；两行 12px 字高 32）
(define HUD_BOT 34)         ; 底部说明带高度（规则 + 一行提示）
(define TICK_MS 40)         ; 定时器周期（25 tick/s）
(define SYM_PER_TICK 220)   ; 每个 tick 推进的符号数（生长速度）
(define HOLD_TICKS 75)      ; 画完后停留的 tick 数（75×40ms = 3s）
(define MAX_LEN 120000)     ; 基因组上限，防止某个预设爆炸
(define DEG2RAD 0.017453292519943295)

;; ---- 字符码（内层指令集）---------------------------------------------------

(define CH_F 70)      ; F、G：前进 1 单位并落笔画线
(define CH_G 71)
(define CH_f 102)     ; f：前进 1 单位但不画线（空移）
(define CH_PLUS 43)   ; +：左转 angle
(define CH_MINUS 45)  ; -：右转 angle
(define CH_LB 91)     ; [ ：压栈（保存位置/方向）
(define CH_RB 93)     ; ] ：出栈（恢复位置/方向）
(define CH_BAR 124)   ; | ：掉头 180°
;; 非终结符（X Y A B）只在重写阶段有意义，海龟阶段直接忽略
(define CH_X 88)
(define CH_Y 89)
(define CH_A 65)
(define CH_B 66)

;; ---- 基因组：双缓冲 --------------------------------------------------------
;; 两个模块级列表轮换使用，set_item! 原地覆写。
;; 注意：'() 字面量是模块级静态槽，push 会跨调用累积，所以只在模块级各取一次，
;; 之后只往这两个缓冲里 push（扩容）或 set_item!（覆写），不再产生新的字面量。

(define gA (quote ()))
(define gB (quote ()))
(define lenA 0)
(define lenB 0)

;; 把 buf[j] 写成 v；容量不够时 push 扩展（push 只作用在模块级缓冲上）
(define store
  (lambda (buf j v)
    (while (< (length buf) (+ j 1)) { (push buf 0) })
    (set_item! buf j v)))

;; ---- 规则表：128 项，按字符码索引；项为 0 表示"无规则，原样保留" ----------

(define rule_tab (quote ()))
(define init_rule_tab
  (lambda ()
    (while (< (length rule_tab) 128) { (push rule_tab 0) })))

(define clear_rules
  (lambda ()
    (define i 0)
    (set! i 0)
    (while (< i 128) { (set_item! rule_tab i 0) (set! i (+ i 1)) })))

(define rule!
  (lambda (c lst) (set_item! rule_tab c lst)))

;; ---- 重写一代 --------------------------------------------------------------

(define rewrite_once
  (lambda ()
    (define i 0) (define j 0) (define c 0) (define rep 0)
    (define k 0) (define n 0) (define tg 0) (define tn 0)
    (set! i 0) (set! j 0)
    (while (< i lenA) {
      (set! c (get_item gA i))
      (set! rep (get_item rule_tab c))
      (if (== rep 0)
          { (store gB j c) (set! j (+ j 1)) }
          { (set! n (length rep)) (set! k 0)
            (while (and (< k n) (< j MAX_LEN))
              { (store gB j (get_item rep k)) (set! j (+ j 1)) (set! k (+ k 1)) }) })
      (set! i (+ i 1))
    })
    (set! lenB j)
    ;; 交换 A/B（gA 变成新基因组，gB 成为下次的写入缓冲）
    (set! tg gA) (set! gA gB) (set! gB tg)
    (set! tn lenA) (set! lenA lenB) (set! lenB tn)))

;; ---- 海龟状态 --------------------------------------------------------------
;; 位置用"单位坐标"（步长恒为 1），方向保存为单位向量 (dx,dy)（屏幕坐标 y 向下），
;; 转向时用预设角度的 (cdd,sdd) 做旋转，因此热路径里没有三角函数。

(define tx 0.0) (define ty 0.0)
(define dx 0.0) (define dy -1.0)      ; 起始朝上
(define depth 0)                      ; 嵌套深度（用于取色）
(define stack (quote ()))             ; 每层压入 x,y,dx,dy 四个数
(define cursor 0)                     ; 内层指令流的位置（可跨帧保存）
(define cdd 1.0) (define sdd 0.0)     ; 当前预设的 cos(angle)、sin(angle)

;; 取景：单位坐标 → 屏幕坐标
(define scale 1.0) (define ox 0.0) (define oy 0.0)
(define minx 0.0) (define maxx 0.0) (define miny 0.0) (define maxy 0.0)

;; 当前画笔颜色（随深度变化）
(define col_r 150) (define col_g 120) (define col_b 80)

(define reset_turtle
  (lambda ()
    (set! tx 0.0) (set! ty 0.0)
    (set! dx 0.0) (set! dy -1.0)
    (set! depth 0) (set! cursor 0)
    (while (> (length stack) 0) { (pop stack) })))

(define pen_color
  (lambda (d)
    (if (== d 0)
        { (set! col_r 150) (set! col_g 116) (set! col_b 74) }        ; 主干：棕
        (if (< d 3)
            { (set! col_r 74) (set! col_g 150) (set! col_b 88) }     ; 分枝：深绿
            { (set! col_r 138) (set! col_g 226) (set! col_b 152) })))) ; 末梢：亮绿

(define rotate
  (lambda (sgn)
    (define ndx 0.0) (define ndy 0.0)
    (if (> sgn 0)
        { (set! ndx (- (* cdd dx) (* sdd dy)))
          (set! ndy (+ (* sdd dx) (* cdd dy))) }
        { (set! ndx (+ (* cdd dx) (* sdd dy)))
          (set! ndy (- (* cdd dy) (* sdd dx))) })
    (set! dx ndx) (set! dy ndy)))

;; 前进一个单位：pen=1 落笔画线（画模式下），pen=0 空移
(define forward
  (lambda (draw pen)
    (define nx 0.0) (define ny 0.0)
    (set! nx (+ tx dx))
    (set! ny (+ ty dy))
    (if (== pen 1)
        (if (== draw 1)
            { (pen_color depth)
              (Gfx.draw_line
                (Math.round (+ ox (* scale tx))) (Math.round (+ oy (* scale ty)))
                (Math.round (+ ox (* scale nx))) (Math.round (+ oy (* scale ny)))
                col_r col_g col_b) }
            { (if (< nx minx) { (set! minx nx) })
              (if (> nx maxx) { (set! maxx nx) })
              (if (< ny miny) { (set! miny ny) })
              (if (> ny maxy) { (set! maxy ny) }) })
        {})
    (set! tx nx) (set! ty ny)))

;; ---- 内层解释器：执行一条指令 ----------------------------------------------

(define step1
  (lambda (draw)
    (define c 0)
    (set! c (get_item gA cursor))
    (if (== c CH_F) { (forward draw 1) } {})
    (if (== c CH_G) { (forward draw 1) } {})
    (if (== c CH_f) { (forward draw 0) } {})
    (if (== c CH_PLUS) { (rotate 1) } {})
    (if (== c CH_MINUS) { (rotate -1) } {})
    (if (== c CH_BAR) { (set! dx (- 0.0 dx)) (set! dy (- 0.0 dy)) } {})
    (if (== c CH_LB)
        { (push stack tx) (push stack ty) (push stack dx) (push stack dy)
          (set! depth (+ depth 1)) }
        {})
    (if (== c CH_RB)
        { (set! dy (pop stack)) (set! dx (pop stack))
          (set! ty (pop stack)) (set! tx (pop stack))
          (if (> depth 0) { (set! depth (- depth 1)) }) }
        {})
    (set! cursor (+ cursor 1))))

;; ---- 自动取景：空跑一遍求包围盒 --------------------------------------------

(define compute_bounds
  (lambda ()
    (define ww 0.0) (define hh 0.0) (define s1 0.0) (define s2 0.0)
    (reset_turtle)
    (set! minx tx) (set! maxx tx) (set! miny ty) (set! maxy ty)
    (while (< cursor lenA) { (step1 0) })
    (set! ww (- maxx minx))
    (set! hh (- maxy miny))
    (if (< ww 0.001) { (set! ww 1.0) })
    (if (< hh 0.001) { (set! hh 1.0) })
    (set! s1 (/ (- W (* MARGIN 2)) ww))
    (set! s2 (/ (- (- (- H HUD_TOP) HUD_BOT) (* MARGIN 2)) hh))
    (set! scale (if (< s1 s2) s1 s2))
    (set! ox (- (/ W 2.0) (* scale (+ minx (/ ww 2.0)))))
    (set! oy (- (+ (/ HUD_TOP 2.0) (/ (- H HUD_BOT) 2.0))
                (* scale (+ miny (/ hh 2.0)))))
    (reset_turtle)))

;; ---- 预设（公理 / 角度 / 代数 / 规则） -------------------------------------
;; 每个预设 = 一条公理 + 若干重写规则 + 转角 + 迭代代数。
;; 规则用字符码字面量书写，行尾注释给出可读形式。

(define preset_count 6)
(define pidx 0)
(define angle_deg 25.0)
(define iters 5)
(define pname "分形树")
(define prule "X → F+[[X]-X]-F[-FX]+X, F → FF")

(define set_axiom
  (lambda (c) (set! lenA 1) (store gA 0 c)))

(define load_preset
  (lambda (id)
    (define it 0)
    (clear_rules)
    (Gfx.clear)
    (set! pidx id)
    (if (== id 0)
        { (set! pname "分形树")
          (set! prule "X → F+[[X]-X]-F[-FX]+X, F → FF")
          (set! angle_deg 25.0) (set! iters 5)
          ;; F + [ [ X ] - X ] - F [ - F X ] + X
          (rule! CH_X (quote (70 43 91 91 88 93 45 88 93 45 70 91 45 70 88 93 43 88)))
          ;; F F
          (rule! CH_F (quote (70 70)))
          (set_axiom CH_X) }
        {})
    (if (== id 1)
        { (set! pname "科赫雪花")
          (set! prule "F → F+F--F+F")
          (set! angle_deg 60.0) (set! iters 4)
          (rule! CH_F (quote (70 43 70 45 45 70 43 70)))
          (set! lenA 5)
          (store gA 0 CH_F) (store gA 1 CH_MINUS) (store gA 2 CH_MINUS)
          (store gA 3 CH_F) (store gA 4 CH_MINUS) (store gA 5 CH_MINUS)
          (store gA 6 CH_F) (set! lenA 7) }
        {})
    (if (== id 2)
        { (set! pname "龙曲线")
          (set! prule "X → X+YF+, Y → -FX-Y")
          (set! angle_deg 90.0) (set! iters 11)
          ;; X + Y F +
          (rule! CH_X (quote (88 43 89 70 43)))
          ;; - F X - Y
          (rule! CH_Y (quote (45 70 88 45 89)))
          (set! lenA 2) (store gA 0 CH_F) (store gA 1 CH_X) }
        {})
    (if (== id 3)
        { (set! pname "希尔伯特曲线")
          (set! prule "A → +BF-AFA-FB+, B → -AF+BFB+FA-")
          (set! angle_deg 90.0) (set! iters 5)
          ;; + B F - A F A - F B +
          (rule! CH_A (quote (43 66 70 45 65 70 65 45 70 66 43)))
          ;; - A F + B F B + F A -
          (rule! CH_B (quote (45 65 70 43 66 70 66 43 70 65 45)))
          (set_axiom CH_A) }
        {})
    (if (== id 4)
        { (set! pname "谢尔宾斯基三角")
          (set! prule "F → F-G+F+G-F, G → GG")
          (set! angle_deg 120.0) (set! iters 5)
          ;; F - G + F + G - F
          (rule! CH_F (quote (70 45 71 43 70 43 71 45 70)))
          ;; G G
          (rule! CH_G (quote (71 71)))
          (set! lenA 5)
          (store gA 0 CH_F) (store gA 1 CH_MINUS) (store gA 2 CH_G)
          (store gA 3 CH_MINUS) (store gA 4 CH_G) }
        {})
    (if (== id 5)
        { (set! pname "海藻")
          (set! prule "F → FF-[-F+F+F]+[+F-F-F]")
          (set! angle_deg 22.5) (set! iters 4)
          ;; F F - [ - F + F + F ] + [ + F - F - F ]
          (rule! CH_F (quote (70 70 45 91 45 70 43 70 43 70 93 43 91 43 70 45 70 45 70 93)))
          (set_axiom CH_F) }
        {})
    ;; 逐代重写
    (set! it 0)
    (while (< it iters) { (rewrite_once) (set! it (+ it 1)) })
    ;; 转角 → 旋转系数（热路径不再需要三角函数）
    (set! cdd (Math.cos (* angle_deg DEG2RAD)))
    (set! sdd (Math.sin (* angle_deg DEG2RAD)))
    (compute_bounds)
    ;; 开始生长
    (set! st 1)))

;; 字符串拼接：String.concat 只接受两个参数，这里做成 3~6 元的便捷包装
(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))
(define cat4 (lambda (a b c d) (String.concat a (cat3 b c d))))
(define cat5 (lambda (a b c d e) (String.concat a (cat4 b c d e))))
(define cat6 (lambda (a b c d e f) (String.concat a (cat5 b c d e f))))

;; ---- 状态机与 HUD ----------------------------------------------------------

(define st 0)         ; 0=待机 1=生长中 2=画完停留
(define hold 0)
(define _timer 0)

(define pct
  (lambda ()
    (define v 0)
    (set! v 100)
    (if (> lenA 0) { (set! v (Math.floor (* 100.0 (/ cursor (+ lenA 0.0))))) })
    v))

(define draw_hud
  (lambda ()
    ;; 两条 HUD 带每 tick 清黑再写，避免数字变化叠加成糊字；绘图区已排除这两块
    (Gfx.fill_rect 0 0 W HUD_TOP 0 0 0)
    (Gfx.fill_rect 0 (- H HUD_BOT) W HUD_BOT 0 0 0)
    (Gfx.draw_text pname 6 5 210 232 255)
    (Gfx.draw_text (cat4 (String.atom_to_string (pct)) "%   "
                         (String.atom_to_string lenA) " 符号")
                   6 20 120 160 190)
    (Gfx.draw_text (cat3 (String.atom_to_string (+ pidx 1)) "/" (String.atom_to_string preset_count))
                   (- W 34) 20 120 160 190)
    (Gfx.draw_text prule 6 (- H 30) 120 158 190)
    (Gfx.draw_text "内层解释器：逐字符执行指令流，可跨帧续跑" 6 (- H 15) 78 104 130)))

;; ---- 主循环（定时器驱动） --------------------------------------------------

(define tick
  (lambda ()
    (define n 0)
    (if (== st 1)
        { (set! n 0)
          (while (and (< n SYM_PER_TICK) (< cursor lenA))
            { (step1 1) (set! n (+ n 1)) })
          (draw_hud)
          (Gfx.refresh)
          (if (>= cursor lenA)
              { (set! st 2) (set! hold 0) }) }
        { (if (== st 2)
              { (set! hold (+ hold 1))
                (if (>= hold HOLD_TICKS)
                    { (define nx 0) (set! nx (+ pidx 1))
                      (if (>= nx preset_count) { (set! nx 0) })
                      (load_preset nx) }) }
              {}) })))

;; ---- 启动 ------------------------------------------------------------------

(init_rule_tab)
(set! st 0)
(load_preset 0)
(set! _timer (System.set_interval TICK_MS tick))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行
(EX.install)
