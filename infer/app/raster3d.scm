;; ============================================================================
;; 3D 软件光栅化 —— 灵机应用程序（旋转的环面 / 二十面体 / 莫比乌斯带）
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库 + Diag 本地库
;;
;; 本 App 自己画 3D：没有 OpenGL，也没有 z-buffer，全部在 Scheme 里做——
;;   ① 顶点：单位空间坐标（对象空间），一次性程序化生成；
;;   ② 旋转：每帧由 (ax, ay) 组装 3×3 旋转矩阵 R = Rx·Ry（正交，故法线也用 R 变换，
;;      不必重新归一化）；
;;   ③ 投影：透视 p = D/(D−z)（相机在 +z，D=3），屏幕坐标 = 中心 + 坐标·p·缩放；
;;   ④ 面序：画家算法。按面深度（顶点 z 均值）分桶（16 桶）后由远及近绘制，
;;      避免 O(n log n) 排序；桶用预建列表 + pop 清空，零分配；
;;   ⑤ 光栅化：逐三角形扫描线——每行用三条边的行交求出 [xl,xr]，再用一次
;;      Gfx.draw_line 填整行（本机实测 draw_line 仅 ~0.25 µs/次，比逐像素
;;      set_pixel 便宜一个数量级）；每轮画 2 行（2px 扫描行）以减半解释器开销；
;;   ⑥ 着色：平面着色 + 兰伯特光照，双面材质（莫比乌斯带）取 |n·L| 并关闭背面剔除。
;;
;; 成本实测（host x86-64）：单帧约 2~3 ms（三角形 100~170 个、扫描行 600~1500 行），
;; 定时器 50 ms/帧 → CPU 占用约 5%。浏览器 wasm32 上因解释器慢约 6 倍，帧率约 10~15 fps。
;;
;; 书写约定（本解释器的坑，详见 app/turtle.scm 头部）：算术一律二元嵌套；
;; cond 子句只能带一个表达式且子句多了会编译失败，故全部用平铺 if。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Diag)

;; ---- 画布与节奏 ------------------------------------------------------------

(define W (Gfx.width))
(define H (Gfx.height))
(define TICK_MS 50)
(define HUD_TOP 30)               ; 顶部 HUD 带（绘图区避开）
(define HUD_BOT 20)               ; 底部说明带
(define YMIN HUD_TOP)
(define YMAX (- (- H HUD_BOT) 1))
(define XMIN 0)
(define XMAX (- W 1))
(define CX (/ W 2.0))
(define CY (+ (/ HUD_TOP 2.0) (/ (- H HUD_BOT) 2.0)))   ; 绘图区中心
(define D 3.0)                    ; 相机距离（透视强度）
(define shape_scale 60.0)         ; 单位空间 → 像素（每个形体各自设置，见 build_*）
(define ROWSTEP 2)                ; 每轮扫描行数（2 = 2px 一行）
(define NB 16)                    ; 深度分桶数
(define Z_FAR -1.8)               ; 深度映射范围
(define Z_SPAN 3.6)
(define AX_STEP 0.020)
(define AY_STEP 0.031)
(define SHAPE_TICKS 100)          ; 每个形体停留的 tick 数（5 s）
(define LIGHT_X -0.35)            ; 视空间光照方向（已归一化）
(define LIGHT_Y 0.55)
(define LIGHT_Z 0.76)

;; ---- 旋转矩阵 --------------------------------------------------------------

(define m00 1.0) (define m01 0.0) (define m02 0.0)
(define m10 0.0) (define m11 1.0) (define m12 0.0)
(define m20 0.0) (define m21 0.0) (define m22 1.0)
(define ax 0.55) (define ay 0.0)

(define update_rot
  (lambda ()
    (define cx 0.0) (define sx 0.0) (define cy 0.0) (define sy 0.0)
    (set! ax (+ ax AX_STEP))
    (set! ay (+ ay AY_STEP))
    (set! cx (Math.cos ax)) (set! sx (Math.sin ax))
    (set! cy (Math.cos ay)) (set! sy (Math.sin ay))
    ;; R = Rx(ax)·Ry(ay)
    (set! m00 cy)          (set! m01 0.0)  (set! m02 sy)
    (set! m10 (* sx sy))   (set! m11 cx)   (set! m12 (- 0.0 (* sx cy)))
    (set! m20 (- 0.0 (* cx sy))) (set! m21 sx) (set! m22 (* cx cy))))

;; ---- 几何缓冲（模块级预建，之后只 push / set_item! / pop，不产生新字面量）----

(define vbuf (cons 0 (quote ())))   ; 顶点：x,y,z 平铺（pop 掉哑元素 → 长度 0）
(define tbuf (cons 0 (quote ())))   ; 三角形：三个顶点索引平铺
(define nbuf (cons 0 (quote ())))   ; 面法线（对象空间，已归一化）
(define sbuf (cons 0 (quote ())))   ; 每帧投影结果：sx,sy,z 平铺

(define num_v 0)
(define num_t 0)

(define buf_clear
  (lambda (b) (while (> (length b) 0) { (pop b) })))

;; (cons 0 '()) 建出来的是长度 1 的列表，那个 0 是哑元素；这里统一 pop 掉，
;; 让所有缓冲从长度 0 开始（否则 get_item 下标会整体偏移一位）
(define init_buffers
  (lambda ()
    (pop vbuf) (pop tbuf) (pop nbuf) (pop sbuf)
    (pop dbuf) (pop bcnt) (pop bpos) (pop order)))

(define put_v
  (lambda (x y z) (push vbuf x) (push vbuf y) (push vbuf z)))

(define put_t
  (lambda (a b c) (push tbuf a) (push tbuf b) (push tbuf c)))

;; 面法线：由三个对象空间顶点求叉积并归一化，追加进 nbuf
(define put_n
  (lambda (ia ib ic)
    (define ax1 0.0) (define ay1 0.0) (define az1 0.0)
    (define bx1 0.0) (define by1 0.0) (define bz1 0.0)
    (define nx 0.0) (define ny 0.0) (define nz 0.0) (define ln 0.0)
    (set! ax1 (- (get_item vbuf (+ (* ib 3) 0)) (get_item vbuf (+ (* ia 3) 0))))
    (set! ay1 (- (get_item vbuf (+ (* ib 3) 1)) (get_item vbuf (+ (* ia 3) 1))))
    (set! az1 (- (get_item vbuf (+ (* ib 3) 2)) (get_item vbuf (+ (* ia 3) 2))))
    (set! bx1 (- (get_item vbuf (+ (* ic 3) 0)) (get_item vbuf (+ (* ia 3) 0))))
    (set! by1 (- (get_item vbuf (+ (* ic 3) 1)) (get_item vbuf (+ (* ia 3) 1))))
    (set! bz1 (- (get_item vbuf (+ (* ic 3) 2)) (get_item vbuf (+ (* ia 3) 2))))
    (set! nx (- (* ay1 bz1) (* az1 by1)))
    (set! ny (- (* az1 bx1) (* ax1 bz1)))
    (set! nz (- (* ax1 by1) (* ay1 bx1)))
    (set! ln (Math.sqrt (+ (* nx nx) (+ (* ny ny) (* nz nz)))))
    (if (< ln 1e-9) { (set! ln 1.0) })
    (push nbuf (/ nx ln)) (push nbuf (/ ny ln)) (push nbuf (/ nz ln))))

;; 四边形 → 两个三角形 + 两条法线
(define put_quad
  (lambda (a b c d)
    (put_t a b c)
    (put_t a c d)
    (put_n a b c)
    (put_n a c d)))

;; ---- 形体生成 --------------------------------------------------------------

(define shape 0)
(define shape_name "环面")
(define shape_hint "环面 torus：12×6 四边形网格，画家算法 + 兰伯特着色")
(define double_sided 0)          ; 1 = 不剔除背面（莫比乌斯带）
(define base_r 230) (define base_g 150) (define base_b 90)

(define two_pi (* 2.0 (Math.PI)))   ; Math.PI 是函数，必须调用

(define build_torus
  (lambda ()
    (define nu 12) (define nv 6)
    (define i 0) (define j 0)
    (define u 0.0) (define v 0.0) (define cu 0.0) (define su 0.0) (define cv 0.0) (define sv 0.0)
    (define rr 0.0)
    (buf_clear vbuf) (buf_clear tbuf) (buf_clear nbuf)
    (set! i 0)
    (while (<= i nu) {
      (set! u (* two_pi (/ (+ i 0.0) (+ nu 0.0))))
      (set! cu (Math.cos u)) (set! su (Math.sin u))
      (set! j 0)
      (while (<= j nv) {
        (set! v (* two_pi (/ (+ j 0.0) (+ nv 0.0))))
        (set! cv (Math.cos v)) (set! sv (Math.sin v))
        (set! rr (+ 1.0 (* 0.42 cv)))
        (put_v (* rr cu) (* 0.42 sv) (* rr su))
        (set! j (+ j 1)) })
      (set! i (+ i 1)) })
    (set! i 0)
    (while (< i nu) {
      (set! j 0)
      (while (< j nv) {
        (put_quad (+ (* i (+ nv 1)) j)
                  (+ (* (+ i 1) (+ nv 1)) j)
                  (+ (* (+ i 1) (+ nv 1)) (+ j 1))
                  (+ (* i (+ nv 1)) (+ j 1)))
        (set! j (+ j 1)) })
      (set! i (+ i 1)) })
    (set! num_v (/ (length vbuf) 3))
    (set! num_t (/ (length tbuf) 3))
    (set! base_r 230) (set! base_g 150) (set! base_b 80)
    (set! shape_scale 58.0)
    (set! double_sided 0)
    (set! shape_name "环面")
    (set! shape_hint "环面 12×6 网格 · 144 三角形 · 画家算法")))

(define build_icosa
  (lambda ()
    (define t 0.0) (define ln 0.0)
    (define vs (cons 0 (quote ())))
    (pop vs)                      ; 去掉哑元素，令下标从 0 开始对应第 0 个坐标
    (define fs (cons 0 (quote ())))
    (define i 0)
    (set! t (/ (+ 1.0 (Math.sqrt 5.0)) 2.0))
    (buf_clear vbuf) (buf_clear tbuf) (buf_clear nbuf)
    ;; 12 个顶点（正二十面体标准坐标）
    (push vs (- 0.0 1.0)) (push vs t) (push vs 0.0)
    (push vs 1.0) (push vs t) (push vs 0.0)
    (push vs (- 0.0 1.0)) (push vs (- 0.0 t)) (push vs 0.0)
    (push vs 1.0) (push vs (- 0.0 t)) (push vs 0.0)
    (push vs 0.0) (push vs (- 0.0 1.0)) (push vs t)
    (push vs 0.0) (push vs 1.0) (push vs t)
    (push vs 0.0) (push vs (- 0.0 1.0)) (push vs (- 0.0 t))
    (push vs 0.0) (push vs 1.0) (push vs (- 0.0 t))
    (push vs t) (push vs 0.0) (push vs (- 0.0 1.0))
    (push vs t) (push vs 0.0) (push vs 1.0)
    (push vs (- 0.0 t)) (push vs 0.0) (push vs (- 0.0 1.0))
    (push vs (- 0.0 t)) (push vs 0.0) (push vs 1.0)
    ;; 归一化到单位球
    (set! i 0)
    (while (< i 12) {
      (set! ln (Math.sqrt (+ (* (get_item vs (+ (* i 3) 0)) (get_item vs (+ (* i 3) 0)))
                            (+ (* (get_item vs (+ (* i 3) 1)) (get_item vs (+ (* i 3) 1)))
                               (* (get_item vs (+ (* i 3) 2)) (get_item vs (+ (* i 3) 2)))))))
      (put_v (/ (get_item vs (+ (* i 3) 0)) ln)
             (/ (get_item vs (+ (* i 3) 1)) ln)
             (/ (get_item vs (+ (* i 3) 2)) ln))
      (set! i (+ i 1)) })
    ;; 20 个面
    (set! fs (quote (0 11 5 0 5 1 0 1 7 0 7 10 0 10 11
                     1 5 9 5 11 4 11 10 2 10 7 6 7 1 8
                     3 9 4 3 4 2 3 2 6 3 6 8 3 8 9
                     4 9 5 2 4 11 6 2 10 8 6 7 9 8 1)))
    (set! i 0)
    (while (< i 60) {
      (put_t (get_item fs i) (get_item fs (+ i 1)) (get_item fs (+ i 2)))
      (put_n (get_item fs i) (get_item fs (+ i 1)) (get_item fs (+ i 2)))
      (set! i (+ i 3)) })
    (set! num_v (/ (length vbuf) 3))
    (set! num_t (/ (length tbuf) 3))
    (set! base_r 120) (set! base_g 210) (set! base_b 235)
    (set! shape_scale 80.0)
    (set! double_sided 0)
    (set! shape_name "二十面体")
    (set! shape_hint "正二十面体 · 12 顶点 20 面 · 平面着色")))

(define build_mobius
  (lambda ()
    (define nu 18) (define nv 3)
    (define i 0) (define j 0)
    (define u 0.0) (define v 0.0) (define cu 0.0) (define su 0.0)
    (define ch 0.0) (define sh 0.0) (define cc 0.0) (define vv 0.0)
    (buf_clear vbuf) (buf_clear tbuf) (buf_clear nbuf)
    (set! i 0)
    (while (<= i nu) {
      (set! u (* two_pi (/ (+ i 0.0) (+ nu 0.0))))
      (set! cu (Math.cos u)) (set! su (Math.sin u))
      (set! ch (Math.cos (* 0.5 u))) (set! sh (Math.sin (* 0.5 u)))
      (set! j 0)
      (while (<= j nv) {
        (set! v (+ -0.42 (* 0.84 (/ (+ j 0.0) (+ nv 0.0)))))
        ;; 接缝处（i = nu）把横向参数取反，形成半扭转闭合
        (set! vv v)
        (if (== i nu) { (set! vv (- 0.0 v)) })
        (set! cc (+ 0.9 (* vv ch)))
        (put_v (* cc cu) (* vv sh) (* cc su))
        (set! j (+ j 1)) })
      (set! i (+ i 1)) })
    (set! i 0)
    (while (< i nu) {
      (set! j 0)
      (while (< j nv) {
        (put_quad (+ (* i (+ nv 1)) j)
                  (+ (* (+ i 1) (+ nv 1)) j)
                  (+ (* (+ i 1) (+ nv 1)) (+ j 1))
                  (+ (* i (+ nv 1)) (+ j 1)))
        (set! j (+ j 1)) })
      (set! i (+ i 1)) })
    (set! num_v (/ (length vbuf) 3))
    (set! num_t (/ (length tbuf) 3))
    (set! base_r 225) (set! base_g 110) (set! base_b 200)
    (set! shape_scale 62.0)
    (set! double_sided 1)
    (set! shape_name "莫比乌斯带")
    (set! shape_hint "莫比乌斯带 · 单侧曲面 · 不剔除背面")))

;; ---- 深度排序：计数排序（画家算法）------------------------------------------
;; 不排序也不做 z-buffer：把三角形按深度装进 NB 个"桶"，由远及近绘制。
;; 实现用计数排序（三趟 O(N) 扫描）+ 一个扁平索引数组，避免嵌套列表与任何
;; 逐帧分配；所有缓冲都是模块级预建列表，只 set_item! 覆写。

(define dbuf (cons 0 (quote ())))    ; 每三角形：桶编号（-1 = 已剔除）
(define bcnt (cons 0 (quote ())))    ; 桶计数
(define bpos (cons 0 (quote ())))    ; 桶写指针
(define order (cons 0 (quote ())))   ; 排序结果：三角形索引序列（远 → 近）

;; 保证 buf 至少有 n 个槽位（只增长，不收缩）
(define ensure_cap
  (lambda (buf n)
    (while (< (length buf) n) { (push buf 0) })))

(define init_sort_buffers
  (lambda ()
    (ensure_cap bcnt NB)
    (ensure_cap bpos NB)))

(define buckets_of
  (lambda (z)
    (define k 0)
    (set! k (Math.floor (/ (* (- z Z_FAR) NB) Z_SPAN)))
    (if (< k 0) { (set! k 0) })
    (if (>= k NB) { (set! k (- NB 1)) })
    k))

;; ---- 变换与投影 ------------------------------------------------------------

(define transform
  (lambda ()
    (define i 0) (define x 0.0) (define y 0.0) (define z 0.0)
    (define rx 0.0) (define ry 0.0) (define rz 0.0) (define p 0.0)
    (while (> (length sbuf) 0) { (pop sbuf) })
    (set! i 0)
    (while (< i num_v) {
      (set! x (get_item vbuf (+ (* i 3) 0)))
      (set! y (get_item vbuf (+ (* i 3) 1)))
      (set! z (get_item vbuf (+ (* i 3) 2)))
      (set! rx (+ (* m00 x) (+ (* m01 y) (* m02 z))))
      (set! ry (+ (* m10 x) (+ (* m11 y) (* m12 z))))
      (set! rz (+ (* m20 x) (+ (* m21 y) (* m22 z))))
      (set! p (/ D (- D rz)))
      (push sbuf (+ CX (* (* rx p) shape_scale)))
      (push sbuf (- CY (* (* ry p) shape_scale)))
      (push sbuf rz)
      (set! i (+ i 1)) })))

;; ---- 绘制：三角形扫描线 ----------------------------------------------------

(define draw_tri
  (lambda (t r g b)
    (define i0 0) (define i1 0) (define i2 0)
    (define x0 0.0) (define y0 0.0) (define x1 0.0) (define y1 0.0) (define x2 0.0) (define y2 0.0)
    (define ytop 0.0) (define ybot 0.0) (define ya 0) (define yb 0) (define y 0) (define yc 0.0)
    (define xl 0.0) (define xr 0.0) (define tmp 0.0) (define cnt 0)
    (set! i0 (get_item tbuf (+ (* t 3) 0)))
    (set! i1 (get_item tbuf (+ (* t 3) 1)))
    (set! i2 (get_item tbuf (+ (* t 3) 2)))
    (set! x0 (get_item sbuf (+ (* i0 3) 0))) (set! y0 (get_item sbuf (+ (* i0 3) 1)))
    (set! x1 (get_item sbuf (+ (* i1 3) 0))) (set! y1 (get_item sbuf (+ (* i1 3) 1)))
    (set! x2 (get_item sbuf (+ (* i2 3) 0))) (set! y2 (get_item sbuf (+ (* i2 3) 1)))
    ;; 本解释器的 Math 库没有 min/max（只有 floor/ceil/round/abs…），自己比较
    (set! ytop y0)
    (if (< y1 ytop) { (set! ytop y1) })
    (if (< y2 ytop) { (set! ytop y2) })
    (set! ybot y0)
    (if (> y1 ybot) { (set! ybot y1) })
    (if (> y2 ybot) { (set! ybot y2) })
    (set! ya (Math.floor (+ ytop 0.5)))
    (set! yb (Math.floor (+ ybot 0.5)))
    (if (< ya YMIN) { (set! ya YMIN) })
    (if (> yb YMAX) { (set! yb YMAX) })
    ;; 起点对齐到偶数行：相邻三角形步长一致，避免奇偶错位留下 1px 空隙
    (set! ya (- ya (mod ya ROWSTEP)))
    (set! y ya)
    (while (<= y yb) {
      (set! yc (+ y 0.5))
      (set! xl 1e9) (set! xr -1e9) (set! cnt 0)
      ;; 三条边：取与该扫描行相交的边的 x
      (if (or (and (<= y0 yc) (> y1 yc)) (and (<= y1 yc) (> y0 yc)))
          { (set! tmp (+ x0 (* (/ (- yc y0) (- y1 y0)) (- x1 x0))))
            (set! cnt (+ cnt 1))
            (if (< tmp xl) { (set! xl tmp) })
            (if (> tmp xr) { (set! xr tmp) }) })
      (if (or (and (<= y1 yc) (> y2 yc)) (and (<= y2 yc) (> y1 yc)))
          { (set! tmp (+ x1 (* (/ (- yc y1) (- y2 y1)) (- x2 x1))))
            (set! cnt (+ cnt 1))
            (if (< tmp xl) { (set! xl tmp) })
            (if (> tmp xr) { (set! xr tmp) }) })
      (if (or (and (<= y2 yc) (> y0 yc)) (and (<= y0 yc) (> y2 yc)))
          { (set! tmp (+ x2 (* (/ (- yc y2) (- y0 y2)) (- x0 x2))))
            (set! cnt (+ cnt 1))
            (if (< tmp xl) { (set! xl tmp) })
            (if (> tmp xr) { (set! xr tmp) }) })
      (if (< xr xl) { (set! tmp xl) (set! xl xr) (set! xr tmp) })
      (set! xl (Math.floor xl))
      (set! xr (Math.floor xr))
      (if (< xl XMIN) { (set! xl XMIN) })
      (if (> xr XMAX) { (set! xr XMAX) })
      (if (and (>= cnt 2) (<= xl xr))
          { (Gfx.draw_line xl y xr y r g b)
            (if (< (+ y 1) (+ yb 1)) { (Gfx.draw_line xl (+ y 1) xr (+ y 1) r g b) }) })
      (set! y (+ y (+ ROWSTEP 0))) })))

;; ---- 每帧：变换 → 分桶 → 由远及近光栅化 ------------------------------------

(define frame_no 0)
(define fps_ms 0)
(define t_prev 0)
(define tri_drawn 0)
(define row_count 0)

(define render3d
  (lambda ()
    (define t 0) (define i 0) (define nb 0) (define off 0) (define tmp 0)
    (define nx 0.0) (define ny 0.0) (define nz 0.0) (define lit 0.0)
    (define zc 0.0) (define vis 0) (define idx 0) (define r 0) (define g 0) (define bl 0)
    (update_rot)
    (transform)
    ;; 趟 1：算每三角形深度 → 桶编号（写进 dbuf 的是桶编号），并累计桶计数
    (set! nb 0)
    (while (< nb NB) { (set_item! bcnt nb 0) (set! nb (+ nb 1)) })
    (ensure_cap dbuf num_t)
    (ensure_cap order num_t)
    (set! t 0)
    (set! vis 0)
    (while (< t num_t) {
      (set! i (* t 3))
      ;; 面法线（对象空间）用 R 旋转到视空间；R 正交，故仍是单位向量
      (set! nx (+ (* m00 (get_item nbuf i)) (+ (* m01 (get_item nbuf (+ i 1)))
                                              (* m02 (get_item nbuf (+ i 2))))))
      (set! ny (+ (* m10 (get_item nbuf i)) (+ (* m11 (get_item nbuf (+ i 1)))
                                              (* m12 (get_item nbuf (+ i 2))))))
      (set! nz (+ (* m20 (get_item nbuf i)) (+ (* m21 (get_item nbuf (+ i 1)))
                                              (* m22 (get_item nbuf (+ i 2))))))
      (set! lit (+ (* nx LIGHT_X) (+ (* ny LIGHT_Y) (* nz LIGHT_Z))))
      (set! idx -1)
      ;; 背面剔除留 0.05 的余量：正好侧对视线的面（轮廓边）也画上，避免轮廓出现 1px 缺口；
      ;; 它们深度更大，画家算法会先画，不会遮住正面
      (if (>= lit -0.05) { (set! idx 1) } { (if (== double_sided 1) { (set! idx 1) }) })
      (if (== idx 1)
          {
            (set! zc (/ (+ (get_item sbuf (+ (* (get_item tbuf i) 3) 2))
                           (+ (get_item sbuf (+ (* (get_item tbuf (+ i 1)) 3) 2))
                              (get_item sbuf (+ (* (get_item tbuf (+ i 2)) 3) 2))))
                        3.0))
            (set! idx (buckets_of zc))
            (set_item! bcnt idx (+ (get_item bcnt idx) 1))
            (set! vis (+ vis 1)) }
          {})
      ;; 不论可见与否都记下桶编号，趟 3 按它落位（不可见者记 -1）
      (set_item! dbuf t idx)
      (set! t (+ t 1)) })
    ;; 趟 2：前缀和 → 每桶起始写指针
    (set! off 0)
    (set! nb 0)
    (while (< nb NB) {
      (set_item! bpos nb off)
      (set! off (+ off (get_item bcnt nb)))
      (set! nb (+ nb 1)) })
    ;; 趟 3：按桶把三角形索引写进 order（远桶在前）
    (set! t 0)
    (while (< t num_t) {
      (set! idx (get_item dbuf t))
      (if (>= idx 0)
          { (set_item! order (get_item bpos idx) t)
            (set_item! bpos idx (+ (get_item bpos idx) 1)) }
          {})
      (set! t (+ t 1)) })
    ;; 绘制：order 已按远→近排好
    (set! tri_drawn vis)
    (set! t 0)
    (while (< t vis) {
      (set! idx (get_item order t))
      (set! i (* idx 3))
      (set! nx (+ (* m00 (get_item nbuf i)) (+ (* m01 (get_item nbuf (+ i 1)))
                                              (* m02 (get_item nbuf (+ i 2))))))
      (set! ny (+ (* m10 (get_item nbuf i)) (+ (* m11 (get_item nbuf (+ i 1)))
                                              (* m12 (get_item nbuf (+ i 2))))))
      (set! nz (+ (* m20 (get_item nbuf i)) (+ (* m21 (get_item nbuf (+ i 1)))
                                              (* m22 (get_item nbuf (+ i 2))))))
      (set! lit (+ (* nx LIGHT_X) (+ (* ny LIGHT_Y) (* nz LIGHT_Z))))
      (if (< lit 0.0) { (set! lit (- 0.0 lit)) })
      (set! lit (+ 0.18 (* 0.82 lit)))
      (set! r (Math.floor (* lit base_r)))
      (set! g (Math.floor (* lit base_g)))
      (set! bl (Math.floor (* lit base_b)))
      (draw_tri idx r g bl)
      (set! t (+ t 1)) })))

;; ---- HUD -------------------------------------------------------------------

(define ds "")

(define draw_hud
  (lambda ()
    (Gfx.fill_rect 0 0 W HUD_TOP 0 0 0)
    (Gfx.fill_rect 0 (- H HUD_BOT) W HUD_BOT 0 0 0)
    (Gfx.draw_text shape_name 8 4 225 235 245)
    (set! ds (String.concat "三角形 " (String.atom_to_string tri_drawn)))
    (Gfx.draw_text ds 8 18 130 165 195)
    (set! ds (String.concat "帧 " (String.atom_to_string frame_no)))
    (Gfx.draw_text ds 110 18 130 165 195)
    (set! ds (String.concat "绘制 " (String.atom_to_string fps_ms)))
    (Gfx.draw_text ds 180 18 130 165 195)
    (Gfx.draw_text shape_hint 8 (- H 15) 110 140 170)))

;; ---- 主流程 ----------------------------------------------------------------

(define tick_fn
  (lambda ()
    (define t 0)
    (set! t (Diag.ms))
    (Gfx.fill_rect 0 HUD_TOP W (- (- H HUD_BOT) HUD_TOP) 0 0 0)
    (render3d)
    (set! frame_no (+ frame_no 1))
    (set! fps_ms (- (Diag.ms) t))
    (draw_hud)
    (Gfx.refresh)
    (if (>= frame_no (* SHAPE_TICKS (+ shape 1)))
        { (set! shape (mod (+ shape 1) 3))
          (if (== shape 0) { (build_torus) })
          (if (== shape 1) { (build_icosa) })
          (if (== shape 2) { (build_mobius) }) })))

;; ---- 启动 ------------------------------------------------------------------

(init_buffers)
(init_sort_buffers)
(build_torus)
(set! _timer (System.set_interval TICK_MS tick_fn))