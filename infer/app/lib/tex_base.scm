;; ============================================================================
;; tex_base.scm —— 数学排版引擎的公共底座：绘制操作缓冲 + 数值/小工具
;;
;; 设计要点（为什么是"扁平操作流"而不是"盒子树"）：
;;   排版结果最终只是一串绘制指令（字形 / 线条 / 规则块 / 圆点）。把它们按顺序
;;   压进一条**扁平列表**，绘制端只需一遍顺序扫描——比"盒子树 + 递归绘制"少一层
;;   间接，重绘一帧的 native 调用数少一半。子结构（分子/分母/上下标）需要整体平移
;;   时，排版器记下它在缓冲里的 [from,to) 区间，再调用 (shift from to dx dy) 原地
;;   平移——这正是 TeX 里 "shift_amount" 的等价物，且不需要复制任何操作。
;;
;; **灵机的坑（务必牢记，见 manual §3.4 与 doc/AGENTS.md）**：
;;   源码里的字面量列表（'(a b)）是**编译期静态共享对象**，对它 push 会永久修改
;;   所有同字面量的使用者（实测 (define mk (lambda () '("a"))) 连调两次，第二次
;;   返回的列表带着上一次 push 的元素）。因此：
;;     · 所有累加器一律用 (empty) 新建（= (cdr (cons #f '()))），绝不对 '() 做 push；
;;     · 所有"构造节点"的地方一律用 quasiquote（每次求值新分配）。
;; ============================================================================

(native Math)

;; ---------------------------------------------------------------------------
;; 绘制操作缓冲
;;   op 形态（第 0 项是种类标签，第 1/2 项恒为锚点坐标，便于整体平移）：
;;     ("g" x y em sx sy cp shear bold fid)   字形（基线对齐；sx/sy 拉伸、shear 斜体）
;;     ("r" x y w h)                          实心规则块（分数线/根号横线/上划线）
;;     ("l" x1 y1 x2 y2 wd)                   抗锯齿线段（根号/矢量符号/括号）
;;     ("f" x y rad)                          实心圆点（\cdots 的点、乘号点）
;;     ("o" x y rad)                          空心圆（\circ \oplus \otimes）
;; ---------------------------------------------------------------------------

;; 新建一个**全新的空列表**。灵机的 '() 是编译期静态共享对象，(push '() v) 会永久
;; 污染所有同字面量的使用者（实测连调两次同一函数，第二次拿到的列表已带上第一次 push
;; 的元素）。(cdr (cons #f '())) 每次都分配新表，是安全的"空表构造器"。
(define  empty (lambda () (cdr (cons #f '()))))

(define ops (empty))
(define n 0)

(define  reset (lambda () { (set! ops (empty)) (set! n 0) }))
(define  count (lambda () n))
(define  at (lambda (i) (get_item ops i)))

;; 把缓冲换回一组**已经排好**的操作（缓存复用）：Markdown 阅读器把每条公式的 op 流
;; 连同操作数一起缓存，绘制时先换回来再 (D.draw ...)，避免每帧重排公式。
;; 只读语义：换回来的流不应再被 shift 之类原地修改（排版器每次布局都会先 reset）。
(define  set_ops (lambda (lst k) { (set! ops lst) (set! n k) }))

(define  g (lambda (x y em sx sy cp shear bold fid)
  { (push ops `("g" ,x ,y ,em ,sx ,sy ,cp ,shear ,bold ,fid))
    (set! n (+ n 1)) }))

(define  r (lambda (x y w h)
  { (push ops `("r" ,x ,y ,w ,h))
    (set! n (+ n 1)) }))

(define  l (lambda (x1 y1 x2 y2 wd)
  { (push ops `("l" ,x1 ,y1 ,x2 ,y2 ,wd))
    (set! n (+ n 1)) }))

(define  fc (lambda (x y rad)
  { (push ops `("f" ,x ,y ,rad))
    (set! n (+ n 1)) }))

(define  oc (lambda (x y rad)
  { (push ops `("o" ,x ,y ,rad))
    (set! n (+ n 1)) }))

;; 把 [from,to) 区间内的操作整体平移 (dx,dy)。线段要平移四个端点。
(define  shift (lambda (from to dx dy)
  (define i from)
  (define op #f)
  (define k 0)
  (while (< i to)
    {
      (set! op (at i))
      (set! k (get_item op 0))
      (set_item! op 1 (+ (get_item op 1) dx))
      (set_item! op 2 (+ (get_item op 2) dy))
      (if (equal? k "l")
          {
            (set_item! op 3 (+ (get_item op 3) dx))
            (set_item! op 4 (+ (get_item op 4) dy))
          }
          #f)
      (set! i (+ i 1))
    })))

;; ---------------------------------------------------------------------------
;; 数值小工具（灵机没有 max/min/abs 内建，也没有一元取负）
;; ---------------------------------------------------------------------------

(define  mx (lambda (a b) (if (> a b) a b)))
(define  mn (lambda (a b) (if (< a b) a b)))
(define  neg (lambda (a) (- 0 a)))
(define  absf (lambda (a) (if (< a 0) (- 0 a) a)))

;; 线性插值
(define  lerp (lambda (a b t) (+ a (* t (- b a)))))

;; 三个/四个数相加（灵机 + 恒二元，这里显式嵌套，避免误写成 (+ a b c d)）
(define  sum3 (lambda (a b c) (+ a (+ b c))))
(define  sum4 (lambda (a b c d) (+ a (+ b (+ c d)))))

;; ---------------------------------------------------------------------------
;; 曲线工具：折线近似（圆弧/椭圆弧/二次与三次贝塞尔），供根号、括号、矢量符号用
;; 角度单位为弧度；y 向下为正（所以椭圆上半部对应 sin 为负）。
;; ---------------------------------------------------------------------------

(define  arc (lambda (cx cy rx ry a0 a1 seg wd)
  (define i 0)
  (define t 0.0)
  (define xa 0.0)
  (define ya 0.0)
  (define xb 0.0)
  (define yb 0.0)
  (while (< i seg)
    {
      (set! t (+ a0 (* (- a1 a0) (/ i seg))))
      (set! xa (+ cx (* rx (Math.cos t))))
      (set! ya (+ cy (* ry (Math.sin t))))
      (set! t (+ a0 (* (- a1 a0) (/ (+ i 1) seg))))
      (set! xb (+ cx (* rx (Math.cos t))))
      (set! yb (+ cy (* ry (Math.sin t))))
      (l xa ya xb yb wd)
      (set! i (+ i 1))
    })))

(define  quad (lambda (x0 y0 cx cy x1 y1 seg wd)
  (define i 0)
  (define t 0.0)
  (define u 0.0)
  (define ca 0.0)
  (define cb 0.0)
  (define cc 0.0)
  (define xa 0.0)
  (define ya 0.0)
  (define xb 0.0)
  (define yb 0.0)
  (while (< i seg)
    {
      ;; P(t) = u²P0 + 2utC + t²P1（算术恒二元，故先算系数）
      (set! t (/ i seg))
      (set! u (- 1.0 t))
      (set! ca (* u u))
      (set! cb (* 2.0 (* u t)))
      (set! cc (* t t))
      (set! xa (+ (* ca x0) (+ (* cb cx) (* cc x1))))
      (set! ya (+ (* ca y0) (+ (* cb cy) (* cc y1))))
      (set! t (/ (+ i 1) seg))
      (set! u (- 1.0 t))
      (set! ca (* u u))
      (set! cb (* 2.0 (* u t)))
      (set! cc (* t t))
      (set! xb (+ (* ca x0) (+ (* cb cx) (* cc x1))))
      (set! yb (+ (* ca y0) (+ (* cb cy) (* cc y1))))
      (l xa ya xb yb wd)
      (set! i (+ i 1))
    })))

(define  cubic (lambda (x0 y0 c1x c1y c2x c2y x1 y1 seg wd)
  (define i 0)
  (define t 0.0)
  (define u 0.0)
  (define ca 0.0)
  (define cb 0.0)
  (define cc 0.0)
  (define cd 0.0)
  (define xa 0.0)
  (define ya 0.0)
  (define xb 0.0)
  (define yb 0.0)
  (while (< i seg)
    {
      ;; P(t) = u³P0 + 3u²tC1 + 3ut²C2 + t³P1
      (set! t (/ i seg))
      (set! u (- 1.0 t))
      (set! ca (* u (* u u)))
      (set! cb (* 3.0 (* (* u u) t)))
      (set! cc (* 3.0 (* u (* t t))))
      (set! cd (* t (* t t)))
      (set! xa (+ (* ca x0) (+ (* cb c1x) (+ (* cc c2x) (* cd x1)))))
      (set! ya (+ (* ca y0) (+ (* cb c1y) (+ (* cc c2y) (* cd y1)))))
      (set! t (/ (+ i 1) seg))
      (set! u (- 1.0 t))
      (set! ca (* u (* u u)))
      (set! cb (* 3.0 (* (* u u) t)))
      (set! cc (* 3.0 (* u (* t t))))
      (set! cd (* t (* t t)))
      (set! xb (+ (* ca x0) (+ (* cb c1x) (+ (* cc c2x) (* cd x1)))))
      (set! yb (+ (* ca y0) (+ (* cb c1y) (+ (* cc c2y) (* cd y1)))))
      (l xa ya xb yb wd)
      (set! i (+ i 1))
    })))
