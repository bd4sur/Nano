;; ============================================================================
;; 粒子生命（Particle Life）—— Nano「小游戏」Scheme App
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;; 参考实现：ui_particlelife.c（particle.html 移植）：6 种颜色的粒子在环面世界中
;; 按随机种间吸引/排斥参数自组织。差异：81 粒子（C 版 324，帧预算约束）、
;; 全对交互（C 版均匀网格加速）、System.set_interval 100ms/步节流（C 版每帧
;; 全速）——解释器由宿主 GUI 按时间片驱动，定时器回调即演进一步。
;;
;; 实现要点：
;;   - 粒子状态为 5 个平行一维列表（xs/ys/ks/vxs/vys；get_item/set_item! O(1)）；
;;   - 参数矩阵 param-a/param-b 为 36 元素列表（idx = kind_i*NK + kind_j）；
;;   - '() 字面量是模块级静态槽，仅可安全 push 一次（各表独立字面量；vys 由
;;     (cdr (cons 0 vs2)) 复制 vs2 而得——cons 为全表复制，故为独立对象）；
;;   - 语句位置的分支一律写成形如 { (set! x ...) ... } 的净零块（vendored 编译器
;;     不回收语句位置的残留值，两分支若一留值一不留会破坏操作数栈）；
;;   - 算术一律二元嵌套（本解释器编译器对多元乘法只发射一次 mul）。
;;
;; 玩法：进入即随机生成相互作用参数与初始分布并开始演化；A(ESC) 返回列表。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)

;; ---- 物理常数（与 ui_particlelife.c 一致；世界半宽 1.0，x∈[-1,1] 环面）-----
;; 视口几何先行定义（WORLD_HH 依赖 HUD_H/SW/SH）。

(define NK 6)                     ; 粒子种类数
(define RTH1 0.1)                 ; 近距排斥阈值
(define RTH2 0.2)                 ; 中距阈值
(define RMAX 0.4)                 ; 相互作用半径
(define RMAX2 (* RMAX RMAX))
(define RTH2_M_RTH1 (- RTH2 RTH1))
(define RMAX_M_RTH2 (- RMAX RTH2))
(define JITTER 0.002)             ; 随机震颤幅度
(define DAMPING 0.5)              ; 每步速度衰减系数
(define WORLD_HW 1.0)             ; 世界半宽
(define WORLD_W 2.0)              ; 回绕周期（宽）
(define HUD_H 16)                 ; 顶部信息栏高度（像素）
(define SW (Gfx.width))
(define SH (Gfx.height))
(define WORLD_HH (/ (- SH HUD_H) SW))   ; 世界半高（填满 HUD 以下视口）
(define WORLD_HH2 (* 2.0 WORLD_HH))
(define WORLD_HW_NEG (- 0.0 WORLD_HW))
(define WORLD_HW_NEG2 (- 0.0 WORLD_HH))
(define SCALE (/ SW 2.0))         ; 像素/世界单位
(define CX (/ SW 2.0))            ; 世界原点屏幕坐标
(define CY (+ HUD_H (/ (- SH HUD_H) 2.0)))

;; ---- 粒子状态（5 个平行一维列表）与相互作用参数矩阵 -----------------------

(define N (* 9 9))                ; 粒子数（9x9 撒布）
(define SCATTER_N 9)
(define xs '())
(define ys '())
(define ks '())
(define vxs '())
(define vys '())
(define param-a '())              ; 近距（<RTH1）作用强度矩阵（36 元素）
(define param-b '())              ; 中远距作用强度矩阵（36 元素）
(define gen 0)

;; ---- 调色板（与 C 版一致：#f80 #ff0 #4f0 #0cf #04f #f08）-------------------

(define pal-r '(255 255 68 0 0 255))
(define pal-g '(136 255 255 204 68 0))
(define pal-b '(0 0 0 255 255 136))

(define interact
  (lambda () {
    (define i 0)
    (define j 0)
    (define xi 0.0)
    (define yi 0.0)
    (define ki 0)
    (define kj 0)
    (define ax 0.0)
    (define ay 0.0)
    (define cx 0.0)
    (define cy 0.0)
    (define ddx 0.0)
    (define ddy 0.0)
    (define dd 0.0)
    (define d 0.0)
    (define accel 0.0)
    (define aij 0.0)
    (define vx 0.0)
    (while (< i N) {
      (set! xi (get_item xs i))
      (set! yi (get_item ys i))
      (set! ki (get_item ks i))
      (set! ax 0.0)
      (set! ay 0.0)
      (set! j 0)
      (while (< j N) {
        (set! cx 0.0)
        (set! cy 0.0)
        (if (== j i)
        { (set! cx 0.0) (set! cy 0.0) }
        { (set! ddx (- (get_item xs j) xi))
        (if (> ddx WORLD_HW) { (set! ddx (- ddx WORLD_W)) })
        (if (< ddx WORLD_HW_NEG) { (set! ddx (+ ddx WORLD_W)) })
        (set! ddy (- (get_item ys j) yi))
        (if (> ddy WORLD_HH) { (set! ddy (- ddy WORLD_HH2)) })
        (if (< ddy WORLD_HW_NEG2) { (set! ddy (+ ddy WORLD_HH2)) })
        (set! dd (+ (* ddx ddx) (* ddy ddy)))
        (if (> dd RMAX2)
            { (set! cx 0.0) (set! cy 0.0) }
            {
              (set! d (Math.sqrt dd))
              (set! aij (get_item param-a (+ (* ki NK) kj)))
              (if (< d RTH1)
            { (set! accel (* aij (/ (- RTH1 d) RTH1))) }
            { (if (< d RTH2)
                  { (set! accel (* aij (/ (- d RTH1) RTH2_M_RTH1))) }
                  { (set! accel (* aij (/ (- RMAX d) RMAX_M_RTH2))) }) })
              (set! cx (* accel (/ ddx d)))
              (set! cy (* accel (/ ddy d)))
            }) })
        (set! ax (+ ax cx))
        (set! ay (+ ay cy))
        (set! j (+ j 1))
      })
      (set! vx (+ (get_item vxs i) ax))
      (set_item! vxs i vx)
      (set_item! vys i (+ (get_item vys i) ay))
      (set! i (+ i 1))
    })
  }))

(define move
  (lambda () {
    (define i 0)
    (define x 0.0)
    (define y 0.0)
    (while (< i N) {
      (set_item! vxs i (+ (get_item vxs i) (* JITTER (- (Math.random) 0.5))))
      (set_item! vys i (+ (get_item vys i) (* JITTER (- (Math.random) 0.5))))
      (set! x (+ (get_item xs i) (get_item vxs i)))
      (set! y (+ (get_item ys i) (get_item vys i)))
      (set_item! vxs i (* (get_item vxs i) DAMPING))
      (set_item! vys i (* (get_item vys i) DAMPING))
      (if (> x WORLD_HW) { (set! x (- x WORLD_W)) })
      (if (< x WORLD_HW_NEG) { (set! x (+ x WORLD_W)) })
      (if (> y WORLD_HH) { (set! y (- y WORLD_HH2)) })
      (if (< y WORLD_HW_NEG2) { (set! y (+ y WORLD_HH2)) })
      (set_item! xs i x)
      (set_item! ys i y)
      (set! i (+ i 1))
    })
  }))

(define draw
  (lambda () {
    (define i 0)
    (define px 0)
    (define py 0)
    (define k 0)
    (Gfx.clear)
    (Gfx.fill_rect 0 0 SW HUD_H 24 24 28)
    (Gfx.draw_text "粒子生命" 6 2 255 255 255)
    (Gfx.draw_text "A返回" (- SW 60) 2 180 180 180)
    (while (< i N) {
      (set! px (Math.floor (+ CX (* (get_item xs i) SCALE))))
      (set! py (Math.floor (+ CY (* (get_item ys i) SCALE))))
      (set! k (get_item ks i))
      (Gfx.fill_rect px py 2 2 (get_item pal-r k) (get_item pal-g k) (get_item pal-b k))
      (set! i (+ i 1))
    })
    (Gfx.refresh)
  }))

;; ---- 启动：参数与撒布 → 首帧绘制 → 100ms 周期演进 -----------------------

(define init-params
  (lambda () {
    (define pa '())
    (define pb '())
    (define i 0)
    (define j 0)
    (while (< i NK) {
      (set! j 0)
      (while (< j NK) {
        (if (== i j)
            { (push pa (- 0.0 (+ 0.001 (* 0.0005 (Math.random)))))
              (push pb (* 0.001 (- (Math.random) 0.5))) }
            { (push pa (- 0.0 (+ 0.0005 (* 0.0005 (Math.random)))))
              (push pb (* 0.001 (- (Math.random) 0.5))) })
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    (set! param-a pa)
    (set! param-b pb)
  }))

(define scatter
  (lambda () {
    (define xs2 '())
    (define ys2 '())
    (define ks2 '())
    (define vs2 '())
    (define iy 0)
    (define ix 0)
    (define s 0)
    (while (< iy SCATTER_N) {
      (set! ix 0)
      (while (< ix SCATTER_N) {
        (set! s (if (< (Math.random) 0.5) 1 0))
        (push xs2 (+ (* (- (* 2.0 (/ ix (- SCATTER_N 1))) 1.0) WORLD_HW) (* 0.2 (- (Math.random) 0.5))))
        (push ys2 (+ (* (- (* 2.0 (/ iy (- SCATTER_N 1))) 1.0) WORLD_HH) (* 0.2 (- (Math.random) 0.5))))
        (push ks2 (Math.floor (* NK (Math.random))))
        (push vs2 0)
        (set! ix (+ ix 1))
      })
      (set! iy (+ iy 1))
    })
    (set! xs xs2)
    (set! ys ys2)
    (set! ks ks2)
    (set! vxs vs2)
    (set! vys (cdr (cons 0 vs2)))
  }))

(define step
  (lambda () {
    (interact)
    (move)
    (draw)
  }))

(init-params)
(scatter)
(draw)
(step)
(set! _timer (System.set_interval 100 step))
