;; ============================================================================
;; Lenia（连续态元胞自动机 · 人工生命）—— 人工生命合集 emergence.scm 的「Lenia」页
;;
;; 本文件原本是独立 App（app/lenia.scm）。2026-11 并入 app/emergence.scm 后
;; **降格为库**：不装定时器、不推屏、不管退出按钮；对外只有
;;   init（一次性初始化）/ tick（推进一步）/ draw（整屏重绘）/ budget（时间片）。
;; 状态（Acur/Anxt/step_no/preset）留在模块里，切页不重置。
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;;
;; Lenia 是 Bert Chan 提出的"连续态"元胞自动机：状态 A(x) ∈ [0,1] 连续取值，
;; 每一代做一次环形核卷积（"细胞感知邻域"），再过一条增长函数得到变化量：
;;     u(x) = Σ K(‖d‖)·A(x+d)            K 为归一化环形核（环半径 μk、宽度 σk）
;;     G(u) = 2·exp(−(u−μg)²/(2σg²)) − 1  （μg 附近的"宜居带"给 +，其余给 −）
;;     A'(x) = clip(A(x) + dt·G(u), 0, 1)
;; 卷积核非零于环带 → 相邻细胞既不能太挤也不能太疏，于是自发形成会移动、
;; 会分裂、会自愈的"生物"。与 gol 的离散格子不同，这里的状态和邻域都是连续的。
;;
;; 实现要点：
;;   · 网格 20×12，每格 16×16 像素（320×192，铺满页签栏以上的内容区）；状态用
;;     **padding 缓冲** 24+2R × 12+2R 存放，每步把四周环带从对侧内部拷回（环形边界），
;;     这样内层卷积循环里不需要任何 mod/下标表；
;;   · 环形核 R=3（7×7 = 49 taps），权重一次性预建并归一化；μg/σg/Δt 逐预设切换；
;;   · 上屏不铺方块，而是每格画一个半径随 A 变化的实心圆——相邻圆重叠后融成有机
;;     团块，比 16px 方块更接近细胞群落的观感；
;;   · 增长函数 G(u) 预计算成 256 项查表（u∈[0,1]），热路径里没有 exp；
;;   · 双缓冲轮换，无逐帧分配。
;;
;; 节拍：原独立 App 是 50ms 一拍、每拍 2 步（240..300 格，实测 wasm 10~15fps）。
;; 并入合集后每拍 1 步，节拍由宿主按"每 3 个宿主帧一拍"（≈20 拍/秒）给——
;; 推进速度与原 App 的实测值同量级，单帧占用则减半。
;;
;; 书写约定（本解释器的坑，详见 app/turtle.scm 头部）：算术一律二元嵌套
;;（三操作数的 (+ a b c) 会被错误编译并静默算错）；cond 子句只能带一个表达式；
;; (cons 0 '()) 建出的表含哑元素，须先 pop 掉。
;; ============================================================================

(native Gfx)
(native Math)
(native String)

;; ---- 网格与布局 ------------------------------------------------------------

(define W (Gfx.width))            ; 320
(define H (Gfx.height))           ; 240
;; TAB_H：宿主 emergence.scm 底部页签栏高度（18）。内容区 = HUD_TOP..H-TAB_H = 192px，
;; 故 12 行 × 16px = 192px 正好铺满；页签栏底下不再算格子（两边的高度必须一致）。
(define TAB_H 18)
(define GW 20)                    ; 网格列数（20×16 = 320px）
(define GH 12)                    ; 网格行数（12×16 = 192px = 内容区高度）
(define PX 16)                    ; 每格边长（像素）
(define HUD_TOP 30)
(define CONTENT_H (- (- H TAB_H) HUD_TOP))
(define OX (/ (- W (* GW PX)) 2))                      ; 居中偏移（横向）
(define OY (+ HUD_TOP (/ (- CONTENT_H (* GH PX)) 2)))  ; 居中偏移（纵向）
(define CELLS (* GW GH))

(define R 3)                      ; 卷积核半径（7×7）
(define PW (+ GW (* 2 R)))
(define PH (+ GH (* 2 R)))
(define PC (* PW PH))
(define KT 49)                    ; (2R+1)² 个 tap
;; 本页需要的单帧时间片（宿主的 (App.set_timeslice budget) 取这个值）。一步约
;; 240 格 × 49 taps，取宿主默认值 312500：一步会在相邻几帧里跑完（与原独立 App 一致）。
(define budget 312500)
(define PRESET_TICKS 300)         ; 每组参数持续的拍数（@20 拍/秒 ≈ 15 s 一组）

;; ---- 预设（μg, σg, dt）-----------------------------------------------------
;; 环形核固定 μk=0.5R、σk=0.15R（经典取法）；只切换增长函数的宜居带与时间步。
;; 这几组是在本网格上实测存活的组合（σg 必须放大到 0.02~0.035：R=3 的核很窄，
;; 沿用 R=13 那套 σg=0.015 会让整片熄灭）。

(define preset 0)
(define pname "温泉")
(define muG 0.15)
(define sgG 0.025)
(define DT 0.1)
(define pcount 4)

(define apply_preset
  (lambda (i)
    (set! preset i)
    (if (== i 0) { (set! pname "温泉") (set! muG 0.15) (set! sgG 0.025) (set! DT 0.1) })
    (if (== i 1) { (set! pname "浓汤") (set! muG 0.20) (set! sgG 0.035) (set! DT 0.1) })
    (if (== i 2) { (set! pname "黏菌") (set! muG 0.12) (set! sgG 0.022) (set! DT 0.1) })
    (if (== i 3) { (set! pname "慢潮") (set! muG 0.15) (set! sgG 0.025) (set! DT 0.05) })))

;; ---- 环形核（一次预建，权重归一化）-----------------------------------------

(define kw (cons 0 (quote ())))   ; 49 个权重，按 (dy,dx) 行主序
(define koff (cons 0 (quote ()))) ; 对应的 padded 索引偏移

(define build_kernel
  (lambda ()
    (define dy (- 0 R)) (define dx (- 0 R))
    (define rr 0.0) (define r 0.0) (define w 0.0) (define sum 0.0)
    (define mu 0.5) (define sg 0.15) (define i 0)
    (pop kw) (pop koff)
    (set! dy (- 0 R))
    (while (<= dy R) {
      (set! dx (- 0 R))
      (while (<= dx R) {
        (set! rr (+ (* dx dx) (* dy dy)))
        (set! r (Math.sqrt (/ rr (* R R))))
        (set! w (Math.exp (/ (- 0.0 (* (- r mu) (- r mu))) (* 2.0 (* sg sg)))))
        (push kw w)
        (push koff (+ (* dy PW) dx))
        (set! sum (+ sum w))
        (set! dx (+ dx 1)) })
      (set! dy (+ dy 1)) })
    (set! i 0)
    (while (< i KT) { (set_item! kw i (/ (get_item kw i) sum)) (set! i (+ i 1)) })))

;; ---- 增长函数查表 ----------------------------------------------------------

(define G (cons 0 (quote ())))

(define build_G
  (lambda ()
    (define i 0) (define u 0.0) (define e 0.0)
    (while (> (length G) 0) { (pop G) })
    (set! i 0)
    (while (< i 256) {
      (set! u (/ (+ i 0.0) 255.0))
      (set! e (/ (- 0.0 (* (- u muG) (- u muG))) (* 2.0 (* sgG sgG))))
      (push G (- (* 2.0 (Math.exp e)) 1.0))
      (set! i (+ i 1)) })))

;; ---- 状态双缓冲（padded）---------------------------------------------------

(define A1 (cons 0 (quote ())))
(define A2 (cons 0 (quote ())))
(define Acur 0) (define Anxt 0)

(define init_state
  (lambda ()
    (define i 0)
    (pop A1) (pop A2)
    (set! i 0)
    (while (< i PC) { (push A1 0.0) (push A2 0.0) (set! i (+ i 1)) })
    (set! Acur A1) (set! Anxt A2)))

;; 环形边界：把 padding 环带从对侧内部拷回来
(define refresh_border
  (lambda ()
    (define yy 0) (define xx 0) (define sy 0) (define sx 0)
    (set! yy 0)
    (while (< yy PH) {
      (set! xx 0)
      (while (< xx PW) {
        (if (or (< yy R) (>= yy (+ R GH)))
            {
              (set! sy (mod (+ yy (- 0 R)) GH))
              (set! sx (mod (+ xx (- 0 R)) GW))
              (set_item! Acur (+ (* yy PW) xx) (get_item Acur (+ (* (+ sy R) PW) (+ sx R)))) }
            {
              (if (or (< xx R) (>= xx (+ R GW)))
                  {
                    (set! sy (- yy R))
                    (set! sx (mod (+ xx (- 0 R)) GW))
                    (set_item! Acur (+ (* yy PW) xx) (get_item Acur (+ (* (+ sy R) PW) (+ sx R)))) }
                  {}) })
        (set! xx (+ xx 1)) })
      (set! yy (+ yy 1)) })))

;; ---- 播种 ------------------------------------------------------------------

(define reseed
  (lambda ()
    (define i 0) (define x 0) (define y 0) (define base 0)
    (set! i 0)
    (while (< i PC) { (set_item! Acur i 0.0) (set_item! Anxt i 0.0) (set! i (+ i 1)) })
    ;; 稀疏随机斑块：Lenia 从"稀汤"里自组织出生物，纯均匀噪声会糊成一片
    (set! y 0)
    (while (< y GH) {
      (set! x 0)
      (while (< x GW) {
        (if (< (Math.random) 0.10) {
          (set! base (+ (* (+ y R) PW) (+ x R)))
          (set_item! Acur base (+ 0.45 (* 0.55 (Math.random)))) }
          {})
        (set! x (+ x 1)) })
      (set! y (+ y 1)) })
    (refresh_border)))

;; ---- 单步 ------------------------------------------------------------------

(define step
  (lambda ()
    (define x 0) (define y 0) (define base 0) (define t 0)
    (define u 0.0) (define a 0.0) (define gi 0) (define gv 0.0) (define tmp 0)
    (set! y 0)
    (while (< y GH) {
      (set! x 0)
      (while (< x GW) {
        (set! base (+ (* (+ y R) PW) (+ x R)))
        ;; 49 taps 卷积：一次读权重、一次读邻格值
        (set! u 0.0)
        (set! t 0)
        (while (< t KT) {
          (set! u (+ u (* (get_item kw t) (get_item Acur (+ base (get_item koff t))))))
          (set! t (+ t 1)) })
        ;; 查表得增长量
        (set! gi (Math.floor (* u 255.0)))
        (if (< gi 0) { (set! gi 0) })
        (if (> gi 255) { (set! gi 255) })
        (set! gv (get_item G gi))
        (set! a (+ (get_item Acur base) (* DT gv)))
        (if (< a 0.0) { (set! a 0.0) })
        (if (> a 1.0) { (set! a 1.0) })
        (set_item! Anxt base a)
        (set! x (+ x 1)) })
      (set! y (+ y 1)) })
    (set! tmp Acur) (set! Acur Anxt) (set! Anxt tmp)
    (refresh_border)))

;; ---- 上屏 ------------------------------------------------------------------

(define render
  (lambda ()
    (define x 0) (define y 0) (define v 0.0) (define v2 0.0)
    (define r 0) (define g 0) (define b 0) (define rad 0) (define cx 0) (define cy 0)
    (set! y 0)
    (while (< y GH) {
      (set! x 0)
      (while (< x GW) {
        (set! v (get_item Acur (+ (* (+ y R) PW) (+ x R))))
        ;; 每格画一个半径随 A 变化的实心圆：相邻圆重叠后融成有机的团块，
        ;; 比直接铺方块更接近"细胞群落"的观感
        (if (> v 0.12)
            { (set! v2 (* v v))
              (set! r (Math.floor (+ 10.0 (* 200.0 v2))))
              (set! g (Math.floor (+ 20.0 (* 225.0 v))))
              (set! b (Math.floor (+ 40.0 (* 200.0 v))))
              (set! rad (Math.floor (* (+ 0.5 PX) (+ 0.5 (* 0.5 v)))))
              (set! cx (+ (+ OX (* x PX)) (Math.floor (/ PX 2))))
              (set! cy (+ (+ OY (* y PX)) (Math.floor (/ PX 2))))
              (Gfx.fill_circle cx cy rad r g b) }
            {})
        (set! x (+ x 1)) })
      (set! y (+ y 1)) })))

;; ---- HUD -------------------------------------------------------------------

(define ds "")
(define step_no 0)

(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))
(define cat4 (lambda (a b c d) (String.concat a (cat3 b c d))))
(define num (lambda (x) (String.atom_to_string x)))
(define sh4 (lambda (x) (/ (Math.floor (* x 10000.0)) 10000.0)))

(define draw_hud
  (lambda ()
    (Gfx.set_font 1)                       ; 12px 抗锯齿字（字模是跨 App 共享的静态量）
    (Gfx.fill_rect 0 0 W HUD_TOP 0 0 0)
    (Gfx.draw_text (cat3 "Lenia · " pname "") 8 4 225 235 245)
    (set! ds (cat4 "μg=" (num (sh4 muG)) " σg=" (num (sh4 sgG))))
    (Gfx.draw_text ds 8 18 130 165 195)
    (set! ds (cat3 "步" (num step_no) ""))
    (Gfx.draw_text ds 210 18 130 165 195)))

;; ---- 对外入口（宿主 App 只调这几个 + init/budget）--------------------------

;; 整屏重绘（不推进仿真）：内容区铺黑 → 逐格画圆 → 顶栏
(define paint
  (lambda () {
    (Gfx.fill_rect 0 HUD_TOP W (- H HUD_TOP) 0 0 0)
    (render)
    (draw_hud)
  }))

;; 一拍：推进一步，每 PRESET_TICKS 拍换一组 (μg, σg, Δt) 并重新播种，然后重绘
(define tick
  (lambda () {
    (step)
    (set! step_no (+ step_no 1))
    (if (>= step_no (* PRESET_TICKS (+ preset 1)))
        { (apply_preset (mod (+ preset 1) pcount))
          (build_G)
          (reseed) })
    (if (> step_no 1000000) { (set! step_no 0) })
    (paint)
    (if (< (count_live) 3)
        (set! dead_ticks (+ dead_ticks 1))
        (set! dead_ticks 0))
    (if (> dead_ticks 6)
        { (set! dead_ticks 0)
          (display "[lenia] reseed extinct") (newline)
          (reseed)
          (paint) })
  }))

;; 只重绘（切页回来 / 模态框关闭后的补帧）
(define draw
  (lambda () {
    (paint)
  }))

;; ---- 灭绝自愈（合集特有的处理）--------------------------------------------
;; 随机撒种的"稀汤"有时会让整片在几十步内熄灭（μg 附近的宜居带很窄：初值一偏，
;; G(u) 就长期为负）。作为独立 App 时它靠"每组参数只跑 PRESET_TICKS 拍就重播"掩盖
;; 了这一现象；合集里用户可能长时间停在 Lenia 页盯着全黑的画面，故加一条看门狗：
;; 连续 dead_ticks 拍几乎没有活格（A > 0.12）就重新播种。

(define dead_ticks 0)

;; 内容区里"看得见"的格数（与 render 的可见阈值一致）
(define count_live
  (lambda ()
    (define x 0) (define y 0) (define n 0) (define base 0)
    (set! y 0)
    (while (< y GH) {
      (set! x 0)
      (while (< x GW) {
        (set! base (+ (* (+ y R) PW) (+ x R)))
        (if (> (get_item Acur base) 0.12) (set! n (+ n 1)) #f)
        (set! x (+ x 1)) })
      (set! y (+ y 1)) })
    n))

;; 已推进的步数（诊断用，见 alife_gol.scm 的同名条目）
(define steps
  (lambda () step_no))

;; 一次性初始化（宿主启动时调一次，之后状态一直留在本模块里）
(define init
  (lambda () {
    (init_state)
    (build_kernel)
    (apply_preset 0)
    (build_G)
    (reseed)
    (paint)
  }))

;; ---- 启动：本模块作为库被 import 时**不做任何事**（不装定时器、不刷屏）------
;;   旧版独立 App 在这一行的位置是：(init_state) (build_kernel) (apply_preset 0)
;;   (build_G) (reseed) (set! _timer (System.set_interval ...)) (EX.install_with render)。
