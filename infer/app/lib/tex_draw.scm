;; ============================================================================
;; tex_draw.scm —— 把 tex_layout 产出的绘制操作流画到屏上（带视口裁剪）
;;
;; 用法：
;;   (D.set_view x y w h)                    设定视口（公式绘制区域的裁剪矩形）
;;   (D.draw ox oy r g b)                    以 (ox,oy) 为操作流原点整体绘制
;;
;; 裁剪：字形/规则块/圆点由图形层自己的裁剪矩形负责（Gfx.set_clip）；线段
;; （根号、括号、矢量符号）走的是吴小林抗锯齿直线，**不吃裁剪矩形**，因此本模块
;; 在 Scheme 侧做 Cohen–Sutherland 线段裁剪，避免滚动/缩放时画到页眉页脚上。
;;
;; 坐标取整：规则块宽高至少 1px（分数线厚 0.055em 在 12px 下不足 1px，直接截断会消失）。
;; ============================================================================

(native Gfx)
(native Math)

(import B "app/lib/tex_base.scm")

(define vx 0)
(define vy 0)
(define vw 320)
(define vh 240)

(define  set_view (lambda (x y w h)
  (set! vx x)
  (set! vy y)
  (set! vw w)
  (set! vh h)))

;; ---- Cohen–Sutherland 线段裁剪 -------------------------------------------------
(define rx1 0.0)
(define ry1 0.0)
(define rx2 0.0)
(define ry2 0.0)

(define  code_of (lambda (x y)
  (define c 0)
  (if (< x vx) (set! c (+ c 1)) (if (> x (+ vx vw)) (set! c (+ c 2)) #f))
  (if (< y vy) (set! c (+ c 4)) (if (> y (+ vy vh)) (set! c (+ c 8)) #f))
  c))

;; 裁剪到视口；成功返回 #t，结果在 rx1/ry1/rx2/ry2
(define  clip_seg (lambda (x1 y1 x2 y2)
  (define c1 0)
  (define c2 0)
  (define c 0)
  (define x 0.0)
  (define y 0.0)
  (define go #t)
  (define ok #f)
  (set! rx1 x1)
  (set! ry1 y1)
  (set! rx2 x2)
  (set! ry2 y2)
  (set! c1 (code_of rx1 ry1))
  (set! c2 (code_of rx2 ry2))
  (while go
    (cond
      ((and (== c1 0) (== c2 0)) { (set! ok #t) (set! go #f) })
      ((not (== (Math.bit_and c1 c2) 0)) { (set! ok #f) (set! go #f) })
      (else
        {
          (set! c (if (> c1 0) c1 c2))
          (cond
            ((> (Math.bit_and c 1) 0)
              {
                (set! x vx)
                (set! y (+ ry1 (/ (* (- ry2 ry1) (- vx rx1)) (- rx2 rx1))))
              })
            ((> (Math.bit_and c 2) 0)
              {
                (set! x (+ vx vw))
                (set! y (+ ry1 (/ (* (- ry2 ry1) (- (+ vx vw) rx1)) (- rx2 rx1))))
              })
            ((> (Math.bit_and c 4) 0)
              {
                (set! y vy)
                (set! x (+ rx1 (/ (* (- rx2 rx1) (- vy ry1)) (- ry2 ry1))))
              })
            (else
              {
                (set! y (+ vy vh))
                (set! x (+ rx1 (/ (* (- rx2 rx1) (- (+ vy vh) ry1)) (- ry2 ry1))))
              }))
          (if (== c c1)
              {
                (set! rx1 x)
                (set! ry1 y)
                (set! c1 (code_of rx1 ry1))
              }
              {
                (set! rx2 x)
                (set! ry2 y)
                (set! c2 (code_of rx2 ry2))
              })
        })))
  ok))

;; ---- 绘制 -------------------------------------------------------------------
(define  draw_seg (lambda (x1 y1 x2 y2 wd r g b)
  ;; 线段画**两遍**（水平错开 0.45px）：吴小林抗锯齿的 1px 线在深色底上偏细，
  ;; 单遍看着比字形笔画"虚"；两遍的加法混合把覆盖率补到字形笔画的观感。
  ;; 线宽 >1 会切换到图形层的"粗线"路径（同样是加法混合，但接缝处会出亮点），
  ;; 所以这里坚持 <=1 的线宽 + 双遍叠色。
  (if (clip_seg x1 y1 x2 y2)
      {
        (Gfx.draw_line_aa rx1 ry1 rx2 ry2 wd r g b)
        (Gfx.draw_line_aa (+ rx1 0.45) ry1 (+ rx2 0.45) ry2 wd r g b)
      }
      #f)))

(define  draw (lambda (ox oy r g b)
  (define i 0)
  (define n (B.count))
  (define op #f)
  (define kind #f)
  (define x 0.0)
  (define y 0.0)
  (define w 0.0)
  (define h 0.0)
  (while (< i n)
    {
      (set! op (B.at i))
      (set! kind (get_item op 0))
      (cond
        ((equal? kind "g")
          (Gfx.draw_glyph (get_item op 9) (get_item op 6)
                          (+ ox (get_item op 1)) (+ oy (get_item op 2))
                          (get_item op 3) (get_item op 4) (get_item op 5)
                          (get_item op 7) (get_item op 8) r g b))
        ((equal? kind "r")
          {
            (set! x (Math.round (+ ox (get_item op 1))))
            (set! y (Math.round (+ oy (get_item op 2))))
            (set! w (Math.round (get_item op 3)))
            (set! h (Math.round (get_item op 4)))
            (if (< w 1) (set! w 1) #f)
            (if (< h 1) (set! h 1) #f)
            (Gfx.fill_rect x y w h r g b)
          })
        ((equal? kind "l")
          (draw_seg (+ ox (get_item op 1)) (+ oy (get_item op 2))
                    (+ ox (get_item op 3)) (+ oy (get_item op 4))
                    (get_item op 5) r g b))
        ((equal? kind "f")
          (Gfx.fill_circle (Math.round (+ ox (get_item op 1)))
                           (Math.round (+ oy (get_item op 2)))
                           (B.mx 1.0 (Math.round (get_item op 3))) r g b))
        ((equal? kind "o")
          (Gfx.draw_circle (Math.round (+ ox (get_item op 1)))
                           (Math.round (+ oy (get_item op 2)))
                           (B.mx 1.0 (Math.round (get_item op 3))) r g b))
        (else #f))
      (set! i (+ i 1))
    })))
