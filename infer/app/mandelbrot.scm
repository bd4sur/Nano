;; ============================================================================
;; 曼德博集（Mandelbrot Set）—— Nano「小游戏」演示程序
;;
;; 运行环境：灵机（Animac）Scheme 解释器 + Nano Gfx 本地库
;;
;; 说明：解释器由宿主 GUI 每帧按时间片抢占式驱动，因此本程序按普通同步
;; 循环书写即可，无须关心分帧——绘制逐行渐进呈现，ESC 随时退出。
;;
;; 栈纪律：Gfx 绘图函数是"语句型"调用（与 display/newline 同语义，弹参后
;; 不向操作数栈压值），直接写在语句位置即可，操作数栈保持净零增长；
;; Gfx.width/Gfx.height 为值型调用（各压 1 值）。
;;
;; 原理：逃逸时间算法。对屏幕每一点对应的复常数 c，从 z=0 开始迭代
;; z <- z^2 + c；若 |z|^2 超过 4 则该点属于逃逸集，迭代次数映射为调色板
;; 颜色；始终未逃逸的点属于曼德博集，着深色。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)

;; ---- 画布与视野 ------------------------------------------------------------

(define W (Gfx.width))
(define H (Gfx.height))

(define MAX-ITER 32)          ; 每点最大迭代次数
(define VIEW-W 3.4)           ; 复平面实轴方向视野宽度
(define CENTER-RE -0.65)      ; 视野中心（实部）
(define CENTER-IM 0.0)        ; 视野中心（虚部）

;; ---- 逃逸时间计数 ----------------------------------------------------------
;; 对 c = x0 + y0*i 迭代 z <- z^2 + c（x2/y2 缓存 x^2/y^2 避免重复乘法），
;; 返回逃逸前的迭代次数；始终未逃逸则返回 MAX-ITER。
(define iter-count
  (lambda (x0 y0) {
    (define i 0)
    (define x 0.0)
    (define y 0.0)
    (define x2 0.0)
    (define y2 0.0)
    (while (and (<= (+ x2 y2) 4.0) (< i MAX-ITER)) {
      ;; 注意：本解释器编译器对多元乘法只发射一次 mul（左值丢失），
      ;; 故乘法一律写成二元嵌套形式 (* 2.0 (* x y))。
      (set! y (+ (* 2.0 (* x y)) y0))
      (set! x (+ (- x2 y2) x0))
      (set! x2 (* x x))
      (set! y2 (* y y))
      (set! i (+ i 1))
    })
    i
  }))

;; ---- 调色板 ----------------------------------------------------------------
;; 经典 Bernstein 多项式配色：t ∈ [0,1] 映射为 蓝→黄→红 的暖色序列。
(define pal-r (lambda (t) (* 255.0 (* 9.0  (* (- 1.0 t) (Math.pow t 3.0))))))
(define pal-g (lambda (t) (* 255.0 (* 15.0 (* (Math.pow (- 1.0 t) 2.0) (* t t))))))
(define pal-b (lambda (t) (* 255.0 (* 8.5  (* (Math.pow (- 1.0 t) 3.0) t)))))

;; ---- 主绘制 ----------------------------------------------------------------

(define t0 0)   ; 渲染起始时间戳（毫秒）

(define draw
  (lambda () {
    (define py 0)
    (define px 0)
    (define y0 0.0)
    (define x0 0.0)
    (define n 0)
    (define scale (/ (* VIEW-W H) W))   ; 虚轴视野高度 = 宽度×画布纵横比（与实轴同尺度）
    (Gfx.clear)
    (set! t0 (System.timestamp))
    (while (< py H) {
      (set! y0 (+ CENTER-IM (* (- (/ py H) 0.5) scale)))
      (set! px 0)
      (while (< px W) {
        (set! x0 (+ CENTER-RE (* (- (/ px W) 0.5) VIEW-W)))
        (set! n (iter-count x0 y0))
        (if (>= n MAX-ITER)
            (Gfx.set_pixel px py 10 10 16)
            (Gfx.set_pixel px py
                           (pal-r (/ n MAX-ITER))
                           (pal-g (/ n MAX-ITER))
                           (pal-b (/ n MAX-ITER))))
        (set! px (+ px 1))
      })
      ;; 行粒度渐进推屏：宿主不主动推屏，推屏时机由本程序控制
      (Gfx.refresh)
      (set! py (+ py 1))
    })
  }))

;; ---- 结束字幕：左下角标注渲染耗时 ------------------------------------------

(define caption
  (lambda (elapsed_ms) {
    (Gfx.draw_text (String.concat "Mandelbrot  " (String.concat (String.atom_to_string elapsed_ms) " ms"))
                   4 (- H 15) 255 255 255)
    (Gfx.refresh)
  }))

(draw)
(caption (- (System.timestamp) t0))
