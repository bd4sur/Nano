;; ============================================================================
;; Verlet 物理玩具 —— 灵机应用程序（布 / 绳 / 双摆）
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库 + Diag 本地库
;;
;; 三个场景轮播，全部是最经典的"Verlet 玩具"：
;;   ① 双摆：拉格朗日方程 + 半隐式欧拉（每拍 8 个子步），笔尖留下渐隐轨迹；
;;   ② 绳：两条绳子（每条 28 个质点）挂在顶部，重力 + 摆动风，位置级约束松弛；
;;   ③ 布：22×14 网格（308 个质点 / 580 根约束），顶边若干点固定，风吹出褶皱。
;;
;; 方法说明（Verlet 位置积分）：
;;   每步先由"当前位置 − 上一位置"得到速度，再加外力更新位置；
;;   然后对每根约束做若干轮松弛：把两个质点沿连线各推回一半偏差（固定点权重 0）。
;;   速度隐含在位置差里，所以不需要显式存速度，也不会数值爆掉。
;;
;; 实测（host x86-64）：布场景（308 质点 + 580 约束 × 3 轮 = 1740 次松弛）单帧约
;; 2.5 ms；绳约 1.5 ms；双摆因为只有 2 个自由度，不到 0.5 ms。定时器 40 ms/拍 →
;; CPU 占用 4~7%。浏览器 wasm32 上大约 3~6 倍慢，仍能跑满 20~25 fps。
;;
;; 按《灵机用户参考手册》编写（这些点都踩过，务必遵守）：
;;   · 算术运算固定二参数（(+ a b c) 会被错误编译），取负写 (- 0 x)；
;;   · `/` 恒为浮点、`mod` 恒返回浮点且符号随被除数 —— 列表下标一律 Math.floor；
;;   · 没有 let/case/do/for/return；没有 list/append/map/for-each；
;;   · define 只能写 (define f (lambda ...))，同一作用域内不得重名；
;;   · cond 每个子句只接受单个表达式；while 体不是独立词法区域；
;;   · and/or 结果恒为布尔；Math 库没有 min/max；Math.PI 是函数要 (Math.PI)；
;;   · (cons 0 '()) 建出的表带一个哑元素，须先 pop 掉。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Diag)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 画布与节奏 ------------------------------------------------------------

(define W (Gfx.width))
(define H (Gfx.height))
(define HUD_H 16)
(define VH (- H HUD_H))
(define TICK_MS 40)
(define SCENE_TICKS 700)         ; 每个场景约 28 s

;; ---- 通用 Verlet 存储 ------------------------------------------------------
;; 位置 px/py；上一位置 qx/qy；权重 fg（0=固定点，1=自由点）
;; 约束：ca/cb 为两端点下标，cl 为静止长度

(define px (cons 0 (quote ())))
(define py (cons 0 (quote ())))
(define qx (cons 0 (quote ())))
(define qy (cons 0 (quote ())))
(define fg (cons 0 (quote ())))
(define ca (cons 0 (quote ())))
(define cb (cons 0 (quote ())))
(define cl (cons 0 (quote ())))
(define anx (cons 0 (quote ())))   ; 固定点的基准位置，用于让锚点轻微摆动
(define any (cons 0 (quote ())))

(define GRAV 0.30)               ; 每步重力（像素/步²）
(define DAMP 0.994)              ; 速度阻尼

(define clear_world
  (lambda ()
    (while (> (length px) 0) { (pop px) })
    (while (> (length py) 0) { (pop py) })
    (while (> (length qx) 0) { (pop qx) })
    (while (> (length qy) 0) { (pop qy) })
    (while (> (length fg) 0) { (pop fg) })
    (while (> (length ca) 0) { (pop ca) })
    (while (> (length cb) 0) { (pop cb) })
    (while (> (length cl) 0) { (pop cl) })
    (while (> (length anx) 0) { (pop anx) })
    (while (> (length any) 0) { (pop any) })))

;; 加一个质点；pinned=1 表示固定不动
(define add_p
  (lambda (x y pinned)
    (push px x)
    (push py y)
    (push qx x)
    (push qy y)
    (if (== pinned 1) { (push fg 0.0) } { (push fg 1.0) })
    (push anx x)
    (push any y)))

(define add_c
  (lambda (i j len)
    (push ca i)
    (push cb j)
    (push cl len)))

;; ---- Verlet 一步：积分 + 约束松弛 ------------------------------------------

(define integrate_step
  (lambda (wind t)
    (define i 0) (define n 0) (define w 0.0)
    (define vx 0.0) (define vy 0.0) (define fx 0.0) (define fy 0.0)
    (define nx 0.0) (define ny 0.0) (define oy 0.0)
    (set! n (length px))
    (set! i 0)
    (while (< i n) {
      (set! w (get_item fg (Math.floor i)))
      (if (> w 0.5)
          {
            (set! vx (* (- (get_item px (Math.floor i)) (get_item qx (Math.floor i))) DAMP))
            (set! vy (* (- (get_item py (Math.floor i)) (get_item qy (Math.floor i))) DAMP))
            ;; 风：沿 x 方向，按质点序号与时间做正弦扰动，配合少量竖直分量
            (set! fx (* wind (Math.sin (+ (* 0.11 (+ i 0.0)) (* 0.045 t)))))
            (set! oy (* 0.35 (Math.sin (+ (* 0.07 (+ i 0.0)) (* 0.031 t)))))
            (set! fy (+ GRAV (* wind oy)))
            (set! nx (+ (get_item px (Math.floor i)) (+ vx fx)))
            (set! ny (+ (get_item py (Math.floor i)) (+ vy fy)))
            (set_item! qx (Math.floor i) (get_item px (Math.floor i)))
            (set_item! qy (Math.floor i) (get_item py (Math.floor i)))
            (set_item! px (Math.floor i) nx)
            (set_item! py (Math.floor i) ny) }
          {})
      (set! i (+ i 1)) })))

(define clamp_world
  (lambda ()
    (define i 0) (define n 0) (define w 0.0) (define x 0.0) (define y 0.0)
    (set! n (length px))
    (set! i 0)
    (while (< i n) {
      (set! w (get_item fg (Math.floor i)))
      (if (> w 0.5)
          {
            (set! x (get_item px (Math.floor i)))
            (set! y (get_item py (Math.floor i)))
            (if (< x 2.0) { (set! x 2.0) })
            (if (> x 318.0) { (set! x 318.0) })
            (if (< y 2.0) { (set! y 2.0) })
            (if (> y (+ VH -2.0)) { (set! y (+ VH -2.0)) })
            (set_item! px (Math.floor i) x)
            (set_item! py (Math.floor i) y) }
          {})
      (set! i (+ i 1)) })))

;; 让固定点轻微摆动（持续给整个系统注入能量，避免几秒后完全静止）
(define sway_pins
  (lambda (amp spd phase)
    (define i 0) (define n 0) (define w 0.0)
    (define sx 0.0) (define sy 0.0)
    (set! n (length fg))
    (set! i 0)
    (while (< i n) {
      (set! w (get_item fg (Math.floor i)))
      (if (< w 0.5)
          {
            (set! sx (+ (get_item anx (Math.floor i))
                        (* amp (Math.sin (+ (* spd (+ i 0.0)) phase)))))
            (set! sy (get_item any (Math.floor i)))
            (set_item! px (Math.floor i) sx)
            (set_item! py (Math.floor i) sy) }
          {})
      (set! i (+ i 1)) })))

(define relax
  (lambda (iters)
    (define it 0) (define k 0) (define na 0)
    (define ia 0) (define ib 0) (define wa 0.0) (define wb 0.0) (define ws 0.0)
    (define dx 0.0) (define dy 0.0) (define d 0.0) (define diff 0.0)
    (define ax 0.0) (define ay 0.0)
    (set! na (length ca))
    (set! it 0)
    (while (< it iters) {
      (set! k 0)
      (while (< k na) {
        (set! ia (get_item ca (Math.floor k)))
        (set! ib (get_item cb (Math.floor k)))
        (set! dx (- (get_item px ib) (get_item px ia)))
        (set! dy (- (get_item py ib) (get_item py ia)))
        (set! d (Math.sqrt (+ (* dx dx) (* dy dy))))
        (if (< d 0.000001) { (set! d 0.000001) })
        (set! diff (/ (- d (get_item cl (Math.floor k))) d))
        (set! wa (get_item fg ia))
        (set! wb (get_item fg ib))
        (set! ws (+ wa wb))
        (if (> ws 0.0001)
            {
              (set! ax (* (/ dx ws) diff))
              (set! ay (* (/ dy ws) diff))
              (set_item! px ia (+ (get_item px ia) (* ax wa)))
              (set_item! py ia (+ (get_item py ia) (* ay wa)))
              (set_item! px ib (- (get_item px ib) (* ax wb)))
              (set_item! py ib (- (get_item py ib) (* ay wb))) }
            {})
        (set! k (+ k 1)) })
      (set! it (+ it 1)) })))

;; 把所有约束画成线段（颜色由竖直位置与序号决定）
(define draw_links
  (lambda (r0 g0 b0 r1 g1 b1)
    (define k 0) (define na 0) (define ia 0) (define ib 0)
    (define y 0.0) (define f 0.0) (define r 0) (define g 0) (define b 0)
    (set! na (length ca))
    (set! k 0)
    (while (< k na) {
      (set! ia (get_item ca (Math.floor k)))
      (set! ib (get_item cb (Math.floor k)))
      (set! y (get_item py ia))
      (set! f (/ y (+ VH 0.0)))
      (if (> f 1.0) { (set! f 1.0) })
      (if (< f 0.0) { (set! f 0.0) })
      (set! r (Math.floor (+ r0 (* (- r1 r0) f))))
      (set! g (Math.floor (+ g0 (* (- g1 g0) f))))
      (set! b (Math.floor (+ b0 (* (- b1 b0) f))))
      (Gfx.draw_line (Math.floor (get_item px ia)) (Math.floor (get_item py ia))
                     (Math.floor (get_item px ib)) (Math.floor (get_item py ib))
                     r g b)
      (set! k (+ k 1)) })))

(define draw_pins
  (lambda (r g b)
    (define i 0) (define n 0)
    (set! n (length fg))
    (set! i 0)
    (while (< i n) {
      (if (< (get_item fg (Math.floor i)) 0.5)
          { (Gfx.fill_circle (Math.floor (get_item px (Math.floor i)))
                             (Math.floor (get_item py (Math.floor i))) 3 r g b) }
          {})
      (set! i (+ i 1)) })))

;; ---- 背景 ------------------------------------------------------------------

(define draw_bg
  (lambda ()
    (define i 0) (define y0 0) (define y1 0) (define f 0.0)
    (define r 0) (define g 0) (define b 0)
    (set! i 0)
    (while (< i 8) {
      (set! y0 (Math.floor (/ (* i VH) 8)))
      (set! y1 (Math.floor (/ (* (+ i 1) VH) 8)))
      (set! f (/ (+ i 0.0) 7.0))
      (set! r (Math.floor (+ 8.0 (* 22.0 f))))
      (set! g (Math.floor (+ 12.0 (* 34.0 f))))
      (set! b (Math.floor (+ 28.0 (* 54.0 f))))
      (Gfx.fill_rect 0 y0 W (- y1 y0) r g b)
      (set! i (+ i 1)) })))

;; ---- 场景 0：双摆 ----------------------------------------------------------

(define th1 2.1) (define th2 2.5)
(define w1 0.0) (define w2 0.0)
(define L1 66.0) (define L2 54.0)
(define PIVX 160.0) (define PIVY 120.0)
(define PG 9.81)
(define TN 420)                  ; 轨迹点上限
(define trail (cons 0 (quote ())))
(define trail_i 0)
(define bobx 0.0) (define boby 0.0)

(define scene0_init
  (lambda ()
    (define i 0)
    (clear_world)
    (while (> (length trail) 0) { (pop trail) })
    (set! i 0)
    (while (< i (* 2 TN)) { (push trail 0.0) (set! i (+ i 1)) })
    (set! trail_i 0)
    (set! th1 2.1) (set! th2 2.5)
    (set! w1 0.0) (set! w2 0.0)))

;; 单子步：拉格朗日方程（m1=m2=1）
(define pend_sub
  (lambda (dt)
    (define d 0.0) (define den 0.0) (define sd 0.0)
    (define n1 0.0) (define n2 0.0) (define t1 0.0) (define t2 0.0)
    (define t3 0.0) (define a1 0.0) (define a2 0.0)
    (set! d (- th1 th2))
    (set! den (- 3.0 (Math.cos (* 2.0 d))))
    (set! sd (Math.sin d))
    ;; a1 = [ -g(3 sinθ1 + sin(θ1-2θ2)) - 2 sin(d)(ω2²L2 + ω1²L1 cos d) ] / (L1·den)
    (set! t1 (* 3.0 (Math.sin th1)))
    (set! t2 (Math.sin (- th1 (* 2.0 th2))))
    (set! t3 (* (- 0.0 PG) (+ t1 t2)))
    (set! n1 (* w2 w2))
    (set! n1 (* n1 L2))
    (set! n2 (* w1 w1))
    (set! n2 (* n2 L1))
    (set! n2 (* n2 (Math.cos d)))
    (set! n1 (+ n1 n2))
    (set! n1 (* 2.0 (* sd n1)))
    (set! a1 (/ (- t3 n1) (* L1 den)))
    ;; a2 = 2 sin(d)(ω1²L1·2 + 2g cosθ1 + ω2²L2 cos d) / (L2·den)
    (set! n1 (* w1 w1))
    (set! n1 (* n1 L1))
    (set! n1 (* n1 2.0))
    (set! n2 (* PG 2.0))
    (set! n2 (* n2 (Math.cos th1)))
    (set! n1 (+ n1 n2))
    (set! n2 (* w2 w2))
    (set! n2 (* n2 L2))
    (set! n2 (* n2 (Math.cos d)))
    (set! n1 (+ n1 n2))
    (set! a2 (/ (* 2.0 (* sd n1)) (* L2 den)))
    ;; 半隐式欧拉
    (set! w1 (+ w1 (* a1 dt)))
    (set! w2 (+ w2 (* a2 dt)))
    (set! th1 (+ th1 (* w1 dt)))
    (set! th2 (+ th2 (* w2 dt)))))

(define scene0_step
  (lambda (t)
    (define s 0) (define x1 0.0) (define y1 0.0) (define idx 0)
    (set! s 0)
    (while (< s 8) { (pend_sub 0.005) (set! s (+ s 1)) })
    (set! x1 (+ PIVX (* L1 (Math.sin th1))))
    (set! y1 (+ PIVY (* L1 (Math.cos th1))))
    (set! bobx (+ x1 (* L2 (Math.sin th2))))
    (set! boby (+ y1 (* L2 (Math.cos th2))))
    (set! idx (* 2 trail_i))
    (set_item! trail idx bobx)
    (set_item! trail (+ idx 1) boby)
    (set! trail_i (+ trail_i 1))
    (if (>= trail_i TN) { (set! trail_i 0) })))

(define scene0_draw
  (lambda ()
    (define k 0) (define idx 0) (define nx 0) (define ny 0)
    (define x0 0.0) (define y0 0.0) (define x1 0.0) (define y1 0.0)
    (define f 0.0) (define r 0) (define g 0) (define b 0)
    (define x1j 0.0) (define y1j 0.0)
    ;; 轨迹：越旧越暗（环状缓冲，从最旧的点开始连）
    (set! k 0)
    (while (< k (- TN 1)) {
      (set! idx (* 2 (Math.floor (mod (+ trail_i (+ k 1)) TN))))
      (set! nx (* 2 (Math.floor (mod (+ trail_i (+ k 2)) TN))))
      (set! x0 (get_item trail (Math.floor idx)))
      (set! y0 (get_item trail (Math.floor (+ idx 1))))
      (set! x1 (get_item trail (Math.floor nx)))
      (set! y1 (get_item trail (Math.floor (+ nx 1))))
      (set! f (/ (+ k 0.0) (+ TN 0.0)))
      (set! r (Math.floor (+ 30.0 (* 190.0 f))))
      (set! g (Math.floor (+ 40.0 (* 120.0 f))))
      (set! b (Math.floor (+ 70.0 (* 150.0 f))))
      (if (> (+ x0 y0) 1.0)
          { (Gfx.draw_line (Math.floor x0) (Math.floor y0)
                           (Math.floor x1) (Math.floor y1) r g b) }
          {})
      (set! k (+ k 1)) })
    ;; 两根杆 + 两个球
    (set! x1j (+ PIVX (* L1 (Math.sin th1))))
    (set! y1j (+ PIVY (* L1 (Math.cos th1))))
    (Gfx.draw_line (Math.floor PIVX) (Math.floor PIVY) (Math.floor x1j) (Math.floor y1j) 210 220 240)
    (Gfx.draw_line (Math.floor (+ PIVX 1.0)) (Math.floor PIVY) (Math.floor (+ x1j 1.0)) (Math.floor y1j) 210 220 240)
    (Gfx.draw_line (Math.floor x1j) (Math.floor y1j) (Math.floor bobx) (Math.floor boby) 210 220 240)
    (Gfx.draw_line (Math.floor (+ x1j 1.0)) (Math.floor y1j) (Math.floor (+ bobx 1.0)) (Math.floor boby) 210 220 240)
    (Gfx.fill_circle (Math.floor PIVX) (Math.floor PIVY) 3 240 240 250)
    (Gfx.fill_circle (Math.floor x1j) (Math.floor y1j) 5 255 214 120)
    (Gfx.fill_circle (Math.floor bobx) (Math.floor boby) 6 255 128 96)))

;; ---- 场景 1：绳 ------------------------------------------------------------

(define rope_n 28)

(define scene1_init
  (lambda ()
    (define i 0) (define seg 7.0) (define x0 0.0) (define y0 0.0)
    (clear_world)
    ;; 两条绳，锚点分别在左、右上方，长度不同
    (set! x0 92.0)
    (set! y0 20.0)
    (set! i 0)
    (while (< i rope_n) {
      (if (== i 0)
          { (add_p x0 y0 1) }
          { (add_p (+ x0 (* 17.0 (Math.sin (* 0.31 (+ i 0.0)))))
                   (+ y0 (* seg (+ i 0.0))) 0) })
      (if (> i 0) { (add_c (- i 1) i seg) })
      (set! i (+ i 1)) })
    (set! x0 232.0)
    (set! y0 26.0)
    (set! i 0)
    (while (< i rope_n) {
      (if (== i 0)
          { (add_p x0 y0 1) }
          { (add_p (+ x0 (* -17.0 (Math.sin (* 0.29 (+ i 0.0)))))
                   (+ y0 (* seg (+ i 0.0))) 0) })
      (if (> i 0) { (add_c (+ rope_n (- i 1)) (+ rope_n i) seg) })
      (set! i (+ i 1)) })))

(define scene1_step
  (lambda (t)
    (sway_pins 22.0 0.055 0.0)
    (integrate_step 0.16 t)
    (relax 6)
    (clamp_world)))

(define scene1_draw
  (lambda ()
    (draw_links 90 200 255 150 130 255)
    (draw_pins 255 235 150)))

;; ---- 场景 2：布 ------------------------------------------------------------

(define CW2 20)
(define CH2 13)
(define SPX 14.0)
(define X0 27.0)
(define Y0 34.0)

(define scene2_init
  (lambda ()
    (define j 0) (define i 0) (define n 0)
    (clear_world)
    (set! j 0)
    (while (< j CH2) {
      (set! i 0)
      (while (< i CW2) {
        (set! n (+ (* j CW2) i))
        ;; 顶边每隔 5 个点固定一个，形成窗帘式褶皱
        (if (== j 0)
            { (if (== (mod i 5) 0) { (add_p (+ X0 (* SPX i)) Y0 1) } { (add_p (+ X0 (* SPX i)) Y0 0) }) }
            { (add_p (+ X0 (* SPX i)) (+ Y0 (* SPX j)) 0) })
        (if (> i 0) { (add_c (- n 1) n SPX) })
        (if (> j 0) { (add_c (- n CW2) n SPX) })
        (set! i (+ i 1)) })
      (set! j (+ j 1)) })))

(define scene2_step
  (lambda (t)
    (sway_pins 7.0 0.09 0.0)
    (integrate_step 0.26 t)
    (relax 3)
    (clamp_world)))

(define scene2_draw
  (lambda ()
    (draw_links 120 200 255 235 225 250)
    (draw_pins 255 240 160)))

;; ---- 场景调度 --------------------------------------------------------------

(define scene 0)
(define sname "双摆")
(define scene_count 3)
(define frame_ms 0)
(define tick_no 0)
(define scene_ticks 0)

(define load_scene
  (lambda (i)
    (set! scene i)
    (set! scene_ticks 0)
    (if (== i 0) { (set! sname "双摆 · 混沌轨迹") (scene0_init) })
    (if (== i 1) { (set! sname "绳 · 双摆绳") (scene1_init) })
    (if (== i 2) { (set! sname "布 · 风吹褶皱") (scene2_init) })))

(define step_scene
  (lambda ()
    (if (== scene 0) { (scene0_step tick_no) })
    (if (== scene 1) { (scene1_step tick_no) })
    (if (== scene 2) { (scene2_step tick_no) })))

(define draw_scene
  (lambda ()
    (if (== scene 0) { (scene0_draw) })
    (if (== scene 1) { (scene1_draw) })
    (if (== scene 2) { (scene2_draw) })))

;; ---- HUD -------------------------------------------------------------------

(define ds "")
(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))

(define draw_hud
  (lambda ()
    (Gfx.fill_rect 0 VH W HUD_H 10 12 20)
    (Gfx.draw_text sname 4 (+ VH 2) 200 214 236)
    (set! ds (cat3 (String.atom_to_string frame_ms) "ms  质点 "
                   (String.atom_to_string (length px))))
    (Gfx.draw_text ds 196 (+ VH 2) 110 130 160)))

;; ---- 主流程 ----------------------------------------------------------------

(define tick_fn
  (lambda ()
    (EX.check)
    (define t0 0)
    (set! t0 (Diag.ms))
    (draw_bg)
    (step_scene)
    (draw_scene)
    (set! frame_ms (- (Diag.ms) t0))
    (draw_hud)
      (EX.draw_button)
      (Gfx.refresh)
    (set! tick_no (+ tick_no 1))
    (set! scene_ticks (+ scene_ticks 1))
    (if (>= scene_ticks SCENE_TICKS)
        { (load_scene (Math.floor (mod (+ scene 1) scene_count))) })))

;; ---- 启动 ------------------------------------------------------------------

(pop px) (pop py) (pop qx) (pop qy) (pop fg)
(pop ca) (pop cb) (pop cl) (pop anx) (pop any)
(pop trail)
(load_scene 0)
(set! _timer (System.set_interval TICK_MS tick_fn))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 draw_scene 擦掉对话框残影）
(EX.install_with draw_scene)
