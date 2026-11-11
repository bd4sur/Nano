;; ============================================================================
;; 水池（快速版 / waterpool_fast）—— `app/waterpool.scm` 的**性能优化副本**
;;
;; 原版（waterpool.scm）是 `infer/ui_water.c` 的逐函数直译：相机 → 网格 →
;; z-buffer 光栅化 → **逐像素光线追迹**（池壁/水面两遍/圆球三个 pass）→ 焦散
;; pass（256×256）→ 76800 次 Gfx.set_pixel 上屏。单帧约 2.2×10⁸ 条 VM 指令，
;; 浏览器（WASM）实测 ~12 秒/帧（~0.08fps）：拖动视角一格要等十几秒。
;;
;; 本文件保留原实现不动，另起一份，把渲染换成**「扫描行大块填充 + 线段 + 逐行混水色」**，
;; 做到**满分辨率 320×240 下 ≥10fps**（浏览器 WASM 实测 27fps，见 doc/AGENTS.md 同日条目）。
;;
;; 为什么不能靠"少算几个像素"解决：灵机解释器在 WASM 上一条 IL 指令约 0.2~0.3µs
;; （x86 宿主约 25ns，差 6~10 倍；且**全局变量取值要沿作用域链回溯**，比局部量再慢
;; 3~6 倍）。10fps 的预算是 100ms ≈ 30 万条指令，而"逐像素"渲染 320×240 即使每像素
;; 只 40 条指令也要 80 万条——**逐像素在这台解释器上根本不可能**。所以这里把着色搬进
;; 宿主 C（图形层）能一次画一大片的地方：
;;
;;   ① 池壁/池底（5 个面）：每面按**扫描行分段**填充（Gfx.fill_rect 一行一次），
;;      瓷砖缝用 Gfx.draw_line 画（面是平面 ⇒ 世界直线投影仍是直线）。整块池子
;;      只要 ~200 次 fill_rect + ~40 次 draw_line，且是**满分辨率**、边缘清晰。
;;   ② 水面：不逐像素算 Fresnel，而是把**水色混进①的逐行填充**里——每行按"水面在
;;      屏幕上的纵向范围"估算这行是掠射还是俯视，混入天光色（权重 0.18~0.82）。
;;      透过水看到的就是①画的瓷砖（"清水池"的真实观感）：近处清透、远处偏蓝，
;;      而且没有半透明叠加层的接缝。
;;   ③ 水波：波场（24×24）只用来驱动**水面高光碎点**（太阳 glitter）与混色明暗，
;;      几何上水面保持水平（波高只有 ±0.02，视觉上无差）。
;;   ④ 圆球：屏幕空间 impostor，只用 fill_circle/draw_circle（5 次 native 调用）。
;;   ⑤ 仿真：波场 128² → 24²；圆球排水只算球影响包围盒；法线/焦散 pass 全删
;;      （不再逐像素着色，就不需要它们）。焦散投影 pass（原版 1.9 秒/帧）删除。
;;
;; 保留不变的部分：场景与相机（同一套 MVP 与 eye 反解）、拖拽交互（拖空白旋转视角、
;; 拖圆球移动圆球）、底部控件栏、圆球浮力/黏滞物理、波场方程（只改网格尺寸）。
;;
;; 灵机解释器的书写约束（详见 manual.md 与 doc/AGENTS.md）：
;;   · 算术/比较恒二参数，多元必须嵌套（`(+ a b c)` 会静默算错并污染操作数栈）；
;;   · 无一元取负（写 `(- 0.0 x)`）；没有 let / list / map / min / max；
;;   · `cond` 每子句只接受单个表达式（本文件一律用平铺的 if）；
;;   · 同一 lambda 内**不能重复 define 同名变量**（会 failed to add defines to scope）；
;;   · **热路径用局部量、不用全局量**（全局取值要回溯作用域链），且循环体内不要
;;     放 define（每次执行都会新建一层闭包环境）；本文件把热循环的局部量都在函数
;;     体首部一次声明，循环里只用 set!。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC / 底部「返回」→ 宿主全局退出确认框

;; ============================================================================
;; 画布 / 场景常量
;; ============================================================================

(define SW (Gfx.width))                  ; 320
(define SH (Gfx.height))                 ; 240

;; 池沿 / 池底高度：C 版把网格 y（-1..1）直接喂给 cube_world_y，于是池沿落在
;; (1-(-1))*7/12-1 = +1/6（略高于水面 y≈0），池底在 -1。
;; 这一条是"看得到池沿、水面嵌在池子里"的关键（照抄 C 版语义）。
(define POOL_TOP (/ 1.0 6.0))            ; +0.16667 池沿
(define POOL_BOT (- 0.0 1.0))            ; -1 池底
(define TILE_N 5.0)                      ; 每单位 5 块瓷砖（同原版 tiles 采样）

(define GRID_N 24)                       ; 波场网格（原版 128）
(define GN2 (* GRID_N GRID_N))
(define GNF (* 1.0 GRID_N))
(define GN1 (- GRID_N 1))
(define BALL_R 0.25)                     ; 球半径
(define GRAVITY_Y -4.0)
(define BUMP 1.7)                        ; 波面法线陡度增益（着色用，1 = 真实斜率）
(define CAU_GAIN 70.0)                   ; 焦散增益（波场曲率 → 亮度）
(define DROP_EVERY 14)                   ; 每 N 拍在水面某处点一滴（让水一直在动）
(define WOB 1.35)                        ; 池底瓷砖被水面折射"扭动"的增益（世界单位）
(define drop_cnt 0)

;; 底部控件栏（几何同 ui_water.c / waterpool.scm；按钮矩形由 wt_button_rect 现算）
(define BAR_H 34)
(define BTN_N 4)
(define BTN_GAP 4)
(define TICK_MS 33)

(define MODE_NONE 0)
(define MODE_ROTATE 1)
(define MODE_SPHERE 2)
(define MODE_BUTTON 3)
(define BTN_BACK 0)
(define BTN_PAUSE 1)
(define BTN_GRAVITY 2)
(define BTN_LIGHT 3)

(define PI 3.141592653589793)
(define DEG (/ PI 180.0))

;; ============================================================================
;; 小工具
;; ============================================================================

;; floor 的正确写法：x - fmod(x,1) 只在 x>=0 或 x<=-1 时等于 floor，(-1,0) 区间会
;; 错给 0。这里统一用「x - frac」并把 frac 归一到 [0,1)（加 64.0 保证非负）。
(define flr (lambda (x) (- x (mod (+ x 64.0) 1.0))))
(define clr (lambda (x) (- (+ x 1.0) (mod (+ (+ x 1.0) 64.0) 1.0))))
(define fmin2 (lambda (a b) (if (< a b) a b)))
(define fmax2 (lambda (a b) (if (> a b) a b)))
(define clampf (lambda (x lo hi) (if (< x lo) lo (if (> x hi) hi x))))

;; ============================================================================
;; 全局状态
;; ============================================================================

;; 相机 / 光照
(define angleX -25.0)
(define angleY -200.5)
(define eye_x 0.0) (define eye_y 0.0) (define eye_z 0.0)
(define ray00x 0.0) (define ray00y 0.0) (define ray00z 0.0)
(define ray10x 0.0) (define ray10y 0.0) (define ray10z 0.0)
(define ray01x 0.0) (define ray01y 0.0) (define ray01z 0.0)
(define ray11x 0.0) (define ray11y 0.0) (define ray11z 0.0)
(define light_dir_x 0.0) (define light_dir_y 1.0) (define light_dir_z 0.0)
;; 世界系视图基（由 modelview 每帧解出）：屏幕点 → 世界方向用
;;   dir = fwd + right*(ndc_x*kx) + up*(ndc_y*ky)
(define vrx 0.0) (define vry 0.0) (define vrz 0.0)
(define vux 0.0) (define vuy 0.0) (define vuz 0.0)
(define vfx 0.0) (define vfy 0.0) (define vfz 0.0)
(define kxf 0.5522847498) (define kyf 0.4142135624)   ; tan(22.5°)*aspect / tan(22.5°)
;; 折射后（水下）的平行光方向：焦散投影沿它把水面的波形挪到池底/池壁
(define rlx 0.0) (define rly -1.0) (define rlz 0.0)

;; MVP 矩阵标量（每帧展开，顶点投影只做算术）
(define m00 1.0) (define m01 0.0) (define m02 0.0) (define m03 0.0)
(define m10 0.0) (define m11 1.0) (define m12 0.0) (define m13 0.0)
(define m20 0.0) (define m21 0.0) (define m22 1.0) (define m23 0.0)
(define m30 0.0) (define m31 0.0) (define m32 0.0) (define m33 1.0)

;; 圆球（物理 + impostor）
(define ball_cx -0.4) (define ball_cy -0.75) (define ball_cz 0.2)
(define ball_ox -0.4) (define ball_oy -0.75) (define ball_oz 0.2)
(define ball_vx 0.0) (define ball_vy 0.0) (define ball_vz 0.0)
(define ball_gravity_y -4.0)
(define ball_radius 0.25)
(define bsx 0.0) (define bsy 0.0) (define brx 1.0) (define bry 1.0)

;; 交互 / 状态
(define use_sphere_physics 1)
(define wt_paused 0)
(define wt_mode MODE_NONE)
(define prev_hit_x 0.0) (define prev_hit_y 0.0) (define prev_hit_z 0.0)
(define plane_n_x 0.0) (define plane_n_y 0.0) (define plane_n_z 0.0)
(define mouse_down 0)
(define prev_px 0) (define prev_py 0)
(define pressed_btn -1)
(define view_dirty 0)
(define tick_timer 0)

;; 缓冲
(define hA (cons 0 (quote ())))          ; 波场高度（页 A）
(define hB (cons 0 (quote ())))          ; 波场高度（页 B）
(define vA (cons 0 (quote ())))          ; 波场速度（页 A）
(define vB (cons 0 (quote ())))          ; 波场速度（页 B）
(define sim_cur 0)                       ; 本拍读的页（0=A 1=B）
(define nxF (cons 0 (quote ())))         ; 逐格水面法线 x（每帧算一遍）
(define nzF (cons 0 (quote ())))         ; 逐格水面法线 z
(define cauF (cons 0 (quote ())))        ; 逐格焦散强度场（波场曲率 → 聚焦）
(define matMV (cons 0 (quote ())))
(define matP (cons 0 (quote ())))
(define matMVP (cons 0 (quote ())))
(define matTmp (cons 0 (quote ())))
(define matInv (cons 0 (quote ())))

;; 池壁几何
(define cube_v (cons 0 (quote ())))      ; 8 角 × 3 世界坐标
(define csx (cons 0 (quote ())))         ; 8 角的屏幕 x（每帧缓存）
(define csy (cons 0 (quote ())))         ; 8 角的屏幕 y
(define csiw (cons 0 (quote ())))        ; 8 角的 1/w
(define fq (cons 0 (quote ())))          ; 5 个面 × 4 角 × 3 坐标（uv 矩形顺序）
(define rq (cons 0 (quote ())))          ; 5 个面的"干池沿"子四边形（同布局）
(define fquad (cons 0 (quote ())))       ; 5 个面 × 4 角点下标（uv 矩形顺序）

;; 三角形/四边形寄存器（投影输出槽）
(define v0sx 0.0) (define v0sy 0.0) (define v0iw 0.0)

;; 水面在屏幕上的纵向范围（每帧在 wt_compose_camera 里算）：池壁/池底逐行按这个
;; 范围估算"透过水面看它的视线有多掠射"，从而把水色混进填充里
(define wty0 0.0) (define wty1 1.0)

;; 当前面的填充参数
(define qr 0.0) (define qg 0.0) (define qb 0.0)
(define qtint 0)                         ; 1 = 这一面在水下，填充时混水色


;; ============================================================================
;; 4×4 矩阵（与 waterpool.scm 同名同构，每帧仅几次调用）
;; ============================================================================

(define m4make
  (lambda () {
    (define l (cons 0 (quote ())))
    (pop l)
    (define i 0)
    (while (< i 16) { (push l 0.0) (set! i (+ i 1)) })
    l
  }))
(define m4id
  (lambda (o) {
    (define i 0)
    (while (< i 16) { (set_item! o i 0.0) (set! i (+ i 1)) })
    (set_item! o 0 1.0) (set_item! o 5 1.0) (set_item! o 10 1.0) (set_item! o 15 1.0)
  }))
(define m4tr
  (lambda (o x y z) {
    (m4id o)
    (set_item! o 3 x) (set_item! o 7 y) (set_item! o 11 z)
  }))
(define m4mul
  (lambda (o a b) {
    (define r (m4make))
    (define i 0)
    (while (< i 4) {
      (define j 0)
      (while (< j 4) {
        (set_item! r (+ (* i 4) j)
          (+ (+ (+ (* (get_item a (* i 4)) (get_item b j))
                    (* (get_item a (+ (* i 4) 1)) (get_item b (+ 4 j))))
                 (* (get_item a (+ (* i 4) 2)) (get_item b (+ 8 j))))
              (* (get_item a (+ (* i 4) 3)) (get_item b (+ 12 j)))))
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    (define k 0)
    (while (< k 16) { (set_item! o k (get_item r k)) (set! k (+ k 1)) })
  }))
(define m4persp
  (lambda (o fov aspect n f) {
    (define yy (* (Math.tan (* fov (/ (Math.PI) 360.0))) n))
    (define xx (* yy aspect))
    (define i 0)
    (while (< i 16) { (set_item! o i 0.0) (set! i (+ i 1)) })
    (set_item! o 0 (/ (* 2.0 n) (* 2.0 xx)))
    (set_item! o 5 (/ (* 2.0 n) (* 2.0 yy)))
    (set_item! o 10 (/ (- 0.0 (+ f n)) (- f n)))
    (set_item! o 11 (/ (* (* (- 0.0 2.0) f) n) (- f n)))
    (set_item! o 14 (- 0.0 1.0))
  }))
(define m4rot
  (lambda (o deg axx ayy azz) {
    (define dd (Math.sqrt (+ (+ (* axx axx) (* ayy ayy)) (* azz azz))))
    (define aa (* deg DEG))
    (define ux (/ axx dd)) (define uy (/ ayy dd)) (define uz (/ azz dd))
    (define c (Math.cos aa)) (define s (Math.sin aa)) (define t (- 1.0 c))
    (set_item! o 0 (+ (* (* ux ux) t) c))
    (set_item! o 1 (- (* (* ux uy) t) (* uz s)))
    (set_item! o 2 (+ (* (* ux uz) t) (* uy s)))
    (set_item! o 3 0.0)
    (set_item! o 4 (+ (* (* uy ux) t) (* uz s)))
    (set_item! o 5 (+ (* (* uy uy) t) c))
    (set_item! o 6 (- (* (* uy uz) t) (* ux s)))
    (set_item! o 7 0.0)
    (set_item! o 8 (- (* (* uz ux) t) (* uy s)))
    (set_item! o 9 (+ (* (* uz uy) t) (* ux s)))
    (set_item! o 10 (+ (* (* uz uz) t) c))
    (set_item! o 11 0.0)
    (set_item! o 12 0.0) (set_item! o 13 0.0) (set_item! o 14 0.0) (set_item! o 15 1.0)
  }))
;; 伴随式求逆（与 C 的 m4inv 逐项一致；仅触摸拾取时求射线用）
(define m4inv
  (lambda (s o) {
    (define a0 (get_item s 0)) (define a1 (get_item s 1)) (define a2 (get_item s 2)) (define a3 (get_item s 3))
    (define a4 (get_item s 4)) (define a5 (get_item s 5)) (define a6 (get_item s 6)) (define a7 (get_item s 7))
    (define a8 (get_item s 8)) (define a9 (get_item s 9)) (define a10 (get_item s 10)) (define a11 (get_item s 11))
    (define a12 (get_item s 12)) (define a13 (get_item s 13)) (define a14 (get_item s 14)) (define a15 (get_item s 15))
    (define r0 (- (+ (- (* (* a5 a10) a15) (* (* a5 a11) a14)) (* (* a9 a7) a14))
                  (+ (- (* (* a9 a6) a15) (* (* a13 a6) a11)) (* (* a13 a7) a10))))
    (define r1 (- (+ (- (* (- 0.0 1.0) (* (* a1 a10) a15)) (* (* a1 a11) a14)) (* (* a9 a3) a14))
                  (+ (- (* (* a9 a2) a15) (* (* a13 a2) a11)) (* (* a13 a3) a10))))
    (define r2 (- (+ (- (* (* a1 a6) a15) (* (* a1 a7) a14)) (* (* a5 a3) a14))
                  (+ (- (* (* a5 a2) a15) (* (* a13 a2) a7)) (* (* a13 a3) a6))))
    (define r3 (- (+ (- (* (- 0.0 1.0) (* (* a1 a6) a11)) (* (* a1 a7) a10)) (* (* a5 a3) a10))
                  (+ (- (* (* a5 a2) a11) (* (* a9 a2) a7)) (* (* a9 a3) a6))))
    (define r4 (- (+ (- (* (- 0.0 1.0) (* (* a4 a10) a15)) (* (* a4 a11) a14)) (* (* a8 a7) a14))
                  (+ (- (* (* a8 a6) a15) (* (* a12 a6) a11)) (* (* a12 a7) a10))))
    (define r5 (- (+ (- (* (* a0 a10) a15) (* (* a0 a11) a14)) (* (* a8 a3) a14))
                  (+ (- (* (* a8 a2) a15) (* (* a12 a2) a11)) (* (* a12 a3) a10))))
    (define r6 (- (+ (- (* (- 0.0 1.0) (* (* a0 a6) a15)) (* (* a0 a7) a14)) (* (* a4 a3) a14))
                  (+ (- (* (* a4 a2) a15) (* (* a12 a2) a7)) (* (* a12 a3) a6))))
    (define r7 (- (+ (- (* (* a0 a6) a11) (* (* a0 a7) a10)) (* (* a4 a3) a10))
                  (+ (- (* (* a4 a2) a11) (* (* a8 a2) a7)) (* (* a8 a3) a6))))
    (define r8 (- (+ (- (* (* a4 a9) a15) (* (* a4 a11) a13)) (* (* a8 a7) a13))
                  (+ (- (* (* a8 a5) a15) (* (* a12 a5) a11)) (* (* a12 a7) a9))))
    (define r9 (- (+ (- (* (- 0.0 1.0) (* (* a0 a9) a15)) (* (* a0 a11) a13)) (* (* a8 a3) a13))
                  (+ (- (* (* a8 a1) a15) (* (* a12 a1) a11)) (* (* a12 a3) a9))))
    (define r10 (- (+ (- (* (* a0 a5) a15) (* (* a0 a7) a13)) (* (* a4 a3) a13))
                   (+ (- (* (* a4 a1) a15) (* (* a12 a1) a7)) (* (* a12 a3) a5))))
    (define r11 (- (+ (- (* (- 0.0 1.0) (* (* a0 a5) a11)) (* (* a0 a7) a9)) (* (* a4 a3) a9))
                   (+ (- (* (* a4 a1) a11) (* (* a8 a1) a7)) (* (* a8 a3) a5))))
    (define r12 (- (+ (- (* (- 0.0 1.0) (* (* a4 a9) a14)) (* (* a4 a10) a13)) (* (* a8 a6) a13))
                   (+ (- (* (* a8 a5) a14) (* (* a12 a5) a10)) (* (* a12 a6) a9))))
    (define r13 (- (+ (- (* (* a0 a9) a14) (* (* a0 a10) a13)) (* (* a8 a2) a13))
                   (+ (- (* (* a8 a1) a14) (* (* a12 a1) a10)) (* (* a12 a2) a9))))
    (define r14 (- (+ (- (* (- 0.0 1.0) (* (* a0 a5) a14)) (* (* a0 a6) a13)) (* (* a4 a2) a13))
                   (+ (- (* (* a4 a1) a14) (* (* a12 a1) a6)) (* (* a12 a2) a5))))
    (define r15 (- (+ (- (* (* a0 a5) a10) (* (* a0 a6) a9)) (* (* a4 a2) a9))
                   (+ (- (* (* a4 a1) a10) (* (* a8 a1) a6)) (* (* a8 a2) a5))))
    (define det (+ (+ (+ (* a0 r0) (* a1 r4)) (* a2 r8)) (* a3 r12)))
    (if (== det 0.0)
        (m4id o)
        {
          (set_item! o 0 (/ r0 det)) (set_item! o 1 (/ r1 det))
          (set_item! o 2 (/ r2 det)) (set_item! o 3 (/ r3 det))
          (set_item! o 4 (/ r4 det)) (set_item! o 5 (/ r5 det))
          (set_item! o 6 (/ r6 det)) (set_item! o 7 (/ r7 det))
          (set_item! o 8 (/ r8 det)) (set_item! o 9 (/ r9 det))
          (set_item! o 10 (/ r10 det)) (set_item! o 11 (/ r11 det))
          (set_item! o 12 (/ r12 det)) (set_item! o 13 (/ r13 det))
          (set_item! o 14 (/ r14 det)) (set_item! o 15 (/ r15 det))
        })
  }))

;; ============================================================================
;; 相机（与原版逐式一致）+ 每帧把 MVP 展开成标量
;; ============================================================================

(define wt_compose_camera
  (lambda ()
    {
      (m4id matMV)
      (m4tr matTmp 0.0 0.0 -4.0)                 (m4mul matMV matMV matTmp)
      (m4rot matTmp (- 0.0 angleX) 1.0 0.0 0.0)  (m4mul matMV matMV matTmp)
      (m4rot matTmp (- 0.0 angleY) 0.0 1.0 0.0)  (m4mul matMV matMV matTmp)
      (m4tr matTmp 0.0 0.5 0.0)                  (m4mul matMV matMV matTmp)
      (m4persp matP 45.0 (/ (* 1.0 SW) (* 1.0 SH)) 0.01 100.0)
      (m4mul matMVP matP matMV)

      ;; MVP → 16 个标量
      (set! m00 (get_item matMVP 0))  (set! m01 (get_item matMVP 1))
      (set! m02 (get_item matMVP 2))  (set! m03 (get_item matMVP 3))
      (set! m10 (get_item matMVP 4))  (set! m11 (get_item matMVP 5))
      (set! m12 (get_item matMVP 6))  (set! m13 (get_item matMVP 7))
      (set! m20 (get_item matMVP 8))  (set! m21 (get_item matMVP 9))
      (set! m22 (get_item matMVP 10)) (set! m23 (get_item matMVP 11))
      (set! m30 (get_item matMVP 12)) (set! m31 (get_item matMVP 13))
      (set! m32 (get_item matMVP 14)) (set! m33 (get_item matMVP 15))

      ;; 世界系相机位置（由 modelview 反解，同原版）
      (set! eye_x (- 0.0 (+ (+ (* (get_item matMV 3) (get_item matMV 0))
                               (* (get_item matMV 7) (get_item matMV 4)))
                            (* (get_item matMV 11) (get_item matMV 8)))))
      (set! eye_y (- 0.0 (+ (+ (* (get_item matMV 3) (get_item matMV 1))
                               (* (get_item matMV 7) (get_item matMV 5)))
                            (* (get_item matMV 11) (get_item matMV 9)))))
      (set! eye_z (- 0.0 (+ (+ (* (get_item matMV 3) (get_item matMV 2))
                               (* (get_item matMV 7) (get_item matMV 6)))
                            (* (get_item matMV 11) (get_item matMV 10)))))

      ;; 四角射线（unProject），用于触摸拾取
      (m4inv matMVP matInv)
      (define cxr 0)
      (while (< cxr 2)
        {
          (define cyr 0)
          (while (< cyr 2)
            {
              (define ndx (- (* 2.0 (* cxr 1.0)) 1.0))
              (define ndy (- (* 2.0 (* cyr 1.0)) 1.0))
              (define ix (+ (+ (+ (* (get_item matInv 0) ndx) (* (get_item matInv 1) ndy))
                                (* (get_item matInv 2) 1.0)) (get_item matInv 3)))
              (define iy (+ (+ (+ (* (get_item matInv 4) ndx) (* (get_item matInv 5) ndy))
                                (* (get_item matInv 6) 1.0)) (get_item matInv 7)))
              (define iz (+ (+ (+ (* (get_item matInv 8) ndx) (* (get_item matInv 9) ndy))
                                (* (get_item matInv 10) 1.0)) (get_item matInv 11)))
              (define iw (+ (+ (+ (* (get_item matInv 12) ndx) (* (get_item matInv 13) ndy))
                                (* (get_item matInv 14) 1.0)) (get_item matInv 15)))
              (define rxr 0.0) (define ryr 0.0) (define rzr 0.0)
              (if (> iw 0.0)
                  {
                    (define iwi (/ 1.0 iw))
                    (set! rxr (- (* ix iwi) eye_x))
                    (set! ryr (- (* iy iwi) eye_y))
                    (set! rzr (- (* iz iwi) eye_z))
                    (define rlen (Math.sqrt (+ (+ (* rxr rxr) (* ryr ryr)) (* rzr rzr))))
                    (set! rxr (/ rxr rlen)) (set! ryr (/ ryr rlen)) (set! rzr (/ rzr rlen))
                  })
              (if (== cxr 0)
                  (if (== cyr 0)
                      { (set! ray00x rxr) (set! ray00y ryr) (set! ray00z rzr) }
                      { (set! ray01x rxr) (set! ray01y ryr) (set! ray01z rzr) })
                  (if (== cyr 0)
                      { (set! ray10x rxr) (set! ray10y ryr) (set! ray10z rzr) }
                      { (set! ray11x rxr) (set! ray11y ryr) (set! ray11z rzr) }))
              (set! cyr (+ cyr 1))
            })
          (set! cxr (+ cxr 1))
        })

      ;; 世界系视图基（modelview 旋转部分的三行；Rᵀ 把视图方向送回世界）
      (set! vrx (get_item matMV 0)) (set! vry (get_item matMV 1)) (set! vrz (get_item matMV 2))
      (set! vux (get_item matMV 4)) (set! vuy (get_item matMV 5)) (set! vuz (get_item matMV 6))
      (set! vfx (- 0.0 (get_item matMV 8)))
      (set! vfy (- 0.0 (get_item matMV 9)))
      (set! vfz (- 0.0 (get_item matMV 10)))

      ;; 折射后的水下平行光（-lightDir 以 (0,1,0) 为法线折射，air→water）
      (define ix0 (- 0.0 light_dir_x))
      (define iy0 (- 0.0 light_dir_y))
      (define iz0 (- 0.0 light_dir_z))
      (define eta (/ 1.0 1.333))
      (define dd (+ (* 0.0 ix0) (* 1.0 iy0)))
      (define kk (- 1.0 (* (* eta eta) (- 1.0 (* dd dd)))))
      (if (< kk 0.0)
          { (set! rlx light_dir_x) (set! rly light_dir_y) (set! rlz light_dir_z) }
          {
            (define sc (+ (* eta dd) (Math.sqrt kk)))
            (set! rlx (- (* ix0 eta) (* 0.0 sc)))
            (set! rly (- (* iy0 eta) (* 1.0 sc)))
            (set! rlz (- (* iz0 eta) (* 0.0 sc)))
          })

      ;; 水面（y=0 的水平面，覆盖池口）在屏幕上的纵向范围
      (set! wty0 10000.0) (set! wty1 -10000.0)
      (define wq 0)
      (while (< wq 4)
        {
          (define wqx (if (== (mod (flr (/ (* 1.0 wq) 2.0)) 2.0) 0.0) -1.0 1.0))
          (define wqz (if (== (mod (* 1.0 wq) 2.0) 1.0) 1.0 -1.0))
          (proj wqx 0.0 wqz)
          (if (> v0iw 0.0)
              {
                (if (< v0sy wty0) (set! wty0 v0sy))
                (if (> v0sy wty1) (set! wty1 v0sy))
              })
          (set! wq (+ wq 1))
        })
    }))

;; 顶点投影：MVP·(x,y,z,1) → 屏幕坐标 + 1/w + ndcZ（放 v0 槽）
(define proj
  (lambda (xx yy zz)
    {
      (define cw (+ (+ (+ (* m30 xx) (* m31 yy)) (* m32 zz)) m33))
      (define cvx (+ (+ (+ (* m00 xx) (* m01 yy)) (* m02 zz)) m03))
      (define cvy (+ (+ (+ (* m10 xx) (* m11 yy)) (* m12 zz)) m13))
      (if (> cw 0.0)
          {
            (define piw (/ 1.0 cw))
            (set! v0iw piw)
            (set! v0sx (* (+ (* (* cvx piw) 0.5) 0.5) (* 1.0 SW)))
            (set! v0sy (* (- 1.0 (+ (* (* cvy piw) 0.5) 0.5)) (* 1.0 SH)))
          }
          {
            (set! v0iw 0.0)
            (set! v0sx -10000.0)
            (set! v0sy -10000.0)
          })
    }))

;; 波面法线双线性采样（世界 x,z ∈ [-1,1]）→ wnx/wnz
(define wnx 0.0) (define wnz 0.0)
(define wave_normal
  (lambda (wqx wqz)
    {
      (define uq (* (+ wqx 1.0) 0.5))
      (define vq (* (+ wqz 1.0) 0.5))
      (define fqx (- (* uq GNF) 0.5))
      (define fqy (- (* vq GNF) 0.5))
      (define iqx (flr fqx))
      (define iqy (flr fqy))
      (if (< iqx 0.0) (set! iqx 0.0))
      (if (> iqx (- GRID_N 2)) (set! iqx (- GRID_N 2)))
      (if (< iqy 0.0) (set! iqy 0.0))
      (if (> iqy (- GRID_N 2)) (set! iqy (- GRID_N 2)))
      (define gqx (- fqx iqx))
      (define gqy (- fqy iqy))
      (define q00 (* (- 1.0 gqx) (- 1.0 gqy)))
      (define q10 (* gqx (- 1.0 gqy)))
      (define q01 (* (- 1.0 gqx) gqy))
      (define q11 (* gqx gqy))
      (define qii (+ (* iqy GNF) iqx))
      (set! wnx (+ (+ (* (get_item nxF qii) q00) (* (get_item nxF (+ qii 1.0)) q10))
                   (+ (* (get_item nxF (+ qii GNF)) q01)
                      (* (get_item nxF (+ (+ qii GNF) 1.0)) q11))))
      (set! wnz (+ (+ (* (get_item nzF qii) q00) (* (get_item nzF (+ qii 1.0)) q10))
                   (+ (* (get_item nzF (+ qii GNF)) q01)
                      (* (get_item nzF (+ (+ qii GNF) 1.0)) q11))))
    }))

;; getRayForPixel(x,y) → 单位方向放 rx/ry/rz（仅触摸拾取用）
(define rx 0.0) (define ry 0.0) (define rz 0.0)
(define wt_get_ray_for_pixel
  (lambda (px py)
    {
      (define fx (/ (* 1.0 px) (* 1.0 SW)))
      (define fy (- 1.0 (/ (* 1.0 py) (* 1.0 SH))))
      (define q0x (+ ray00x (* (- ray10x ray00x) fx)))
      (define q0y (+ ray00y (* (- ray10y ray00y) fx)))
      (define q0z (+ ray00z (* (- ray10z ray00z) fx)))
      (define q1x (+ ray01x (* (- ray11x ray01x) fx)))
      (define q1y (+ ray01y (* (- ray11y ray01y) fx)))
      (define q1z (+ ray01z (* (- ray11z ray01z) fx)))
      (define qx (+ q0x (* (- q1x q0x) fy)))
      (define qy (+ q0y (* (- q1y q0y) fy)))
      (define qz (+ q0z (* (- q1z q0z) fy)))
      (define ql (Math.sqrt (+ (+ (* qx qx) (* qy qy)) (* qz qz))))
      (set! rx (/ qx ql)) (set! ry (/ qy ql)) (set! rz (/ qz ql))
    }))

;; ============================================================================
;; 几何构建（启动一次）
;;   角点编号 i = x_bit*4 + y_bit*2 + z_bit（y_bit=0 → 池底，1 → 池沿）
;; ============================================================================

(define wt_build_meshes
  (lambda ()
    {
      ;; 这些列表都由 push 累积构建：先把初始化时的那个占位 0 弹掉
      (pop cube_v) (pop csx) (pop csy) (pop csiw) (pop fq) (pop rq) (pop fquad)
      (define q 0)
      (while (< q 8)
        {
          (define qx (if (== (mod (flr (/ (* 1.0 q) 4.0)) 2.0) 0.0) -1.0 1.0))
          (define qy (if (== (mod (flr (/ (* 1.0 q) 2.0)) 2.0) 0.0) POOL_BOT POOL_TOP))
          (define qz (if (== (mod (* 1.0 q) 2.0) 1.0) 1.0 -1.0))
          (push cube_v qx) (push cube_v qy) (push cube_v qz)
          (push csx 0.0) (push csy 0.0) (push csiw 0.0)
          (set! q (+ q 1))
        })
      ;; 5 个面：4 个角按 uv 矩形顺序 (u最小v最小) (u最大v最小) (u最大v最大) (u最小v最大)
      ;;   0 底(y=-1)：u←x, v←z       [0 4 5 1]
      ;;   1 x=-1   ：u←y, v←z        [0 2 3 1]
      ;;   2 x=+1   ：u←y, v←z        [4 6 7 5]
      ;;   3 z=-1   ：u←y, v←x        [0 2 6 4]
      ;;   4 z=+1   ：u←y, v←x        [1 3 7 5]
      (push fquad 0) (push fquad 4) (push fquad 5) (push fquad 1)
      (push fquad 0) (push fquad 2) (push fquad 3) (push fquad 1)
      (push fquad 4) (push fquad 6) (push fquad 7) (push fquad 5)
      (push fquad 0) (push fquad 2) (push fquad 6) (push fquad 4)
      (push fquad 1) (push fquad 3) (push fquad 7) (push fquad 5)
      ;; 面的世界坐标 + "干池沿"子四边形
      ;;   侧壁上 u = 0.5*y + 1，池沿（y >= 0）对应 u >= 1.0；u 的范围是
      ;;   y∈[-1, 1/6] → u∈[0.5, 7/6]，故池沿从参数 t = (1-0.5)/(7/6-0.5) = 3/4 起。
      (define fi 0)
      (while (< fi 5)
        {
          (define c0 (get_item fquad (+ (* fi 4) 0)))
          (define c1 (get_item fquad (+ (* fi 4) 1)))
          (define c2 (get_item fquad (+ (* fi 4) 2)))
          (define c3 (get_item fquad (+ (* fi 4) 3)))
          (define k 0)
          (while (< k 4)
            {
              (define ci (get_item fquad (+ (* fi 4) k)))
              (push fq (get_item cube_v (* ci 3)))
              (push fq (get_item cube_v (+ (* ci 3) 1)))
              (push fq (get_item cube_v (+ (* ci 3) 2)))
              (set! k (+ k 1))
            })
          ;; 池沿子四边形：沿 u 方向从 t=0.75 到 1（底面 fi=0 没有池沿，填整面即可）
          (if (== fi 0)
              {
                (push rq 0.0) (push rq 0.0) (push rq 0.0)
                (push rq 0.0) (push rq 0.0) (push rq 0.0)
                (push rq 0.0) (push rq 0.0) (push rq 0.0)
                (push rq 0.0) (push rq 0.0) (push rq 0.0)
              }
              {
                (define ta 0.75)
                ;; 边 c0→c1 上 t 处、边 c3→c2 上 t 处
                (define a0x (+ (get_item cube_v (* c0 3))
                               (* ta (- (get_item cube_v (* c1 3)) (get_item cube_v (* c0 3))))))
                (define a0y (+ (get_item cube_v (+ (* c0 3) 1))
                               (* ta (- (get_item cube_v (+ (* c1 3) 1)) (get_item cube_v (+ (* c0 3) 1))))))
                (define a0z (+ (get_item cube_v (+ (* c0 3) 2))
                               (* ta (- (get_item cube_v (+ (* c1 3) 2)) (get_item cube_v (+ (* c0 3) 2))))))
                (define a1x (+ (get_item cube_v (* c3 3))
                               (* ta (- (get_item cube_v (* c2 3)) (get_item cube_v (* c3 3))))))
                (define a1y (+ (get_item cube_v (+ (* c3 3) 1))
                               (* ta (- (get_item cube_v (+ (* c2 3) 1)) (get_item cube_v (+ (* c3 3) 1))))))
                (define a1z (+ (get_item cube_v (+ (* c3 3) 2))
                               (* ta (- (get_item cube_v (+ (* c2 3) 2)) (get_item cube_v (+ (* c3 3) 2))))))
                (push rq a0x) (push rq a0y) (push rq a0z)
                (push rq (get_item cube_v (* c1 3)))
                (push rq (get_item cube_v (+ (* c1 3) 1)))
                (push rq (get_item cube_v (+ (* c1 3) 2)))
                (push rq (get_item cube_v (* c2 3)))
                (push rq (get_item cube_v (+ (* c2 3) 1)))
                (push rq (get_item cube_v (+ (* c2 3) 2)))
                (push rq a1x) (push rq a1y) (push rq a1z)
              })
          (set! fi (+ fi 1))
        })
    }))

;; ============================================================================
;; 波场仿真（h/v 双页乒乓；方程与原版逐式一致，网格 24×24）
;;   不再需要法线/焦散场：着色不再逐像素，波场只驱动水面高光碎点与明暗。
;; ============================================================================

(define wt_step_sim
  (lambda ()
    {
      (define src (if (== sim_cur 0) hA hB))
      (define svv (if (== sim_cur 0) vA vB))
      (define dst (if (== sim_cur 0) hB hA))
      (define dvv (if (== sim_cur 0) vB vA))
      (define nn GRID_N) (define nm1 GN1)
      (define yy 0) (define xx 0) (define rowoff 0) (define ii 0)
      (define lt 0) (define rt 0) (define up 0) (define dn 0)
      (define hh 0.0) (define vvq 0.0) (define av 0.0)
      (while (< yy nn)
        {
          (set! rowoff (* yy nn))
          (set! xx 0)
          (while (< xx nn)
            {
              (set! ii (+ rowoff xx))
              (set! lt (if (> xx 0) (- ii 1) ii))
              (set! rt (if (< xx nm1) (+ ii 1) ii))
              (set! up (if (> yy 0) (- ii nn) ii))
              (set! dn (if (< yy nm1) (+ ii nn) ii))
              (set! hh (get_item src ii))
              (set! vvq (get_item svv ii))
              (set! av (* (+ (+ (get_item src lt) (get_item src rt))
                             (+ (get_item src up) (get_item src dn))) 0.25))
              (set! vvq (* (+ vvq (* (- av hh) 2.0)) 0.995))
              (set! hh (+ hh vvq))
              (set_item! dst ii hh)
              (set_item! dvv ii vvq)
              (set! xx (+ xx 1))
            })
          (set! yy (+ yy 1))
        })
      (set! sim_cur (- 1 sim_cur))
    }))

;; 水面法线场 + 焦散强度场（一遍扫 24×24；不再逐像素采样，所以只算一次）
;;   · 法线：高度差 × BUMP（ny 在采样处用 1-(nx²+nz²)/2 解出）
;;   · 焦散：波面的拉普拉斯（凹处收光）→ 1 + CAU_GAIN*lap，供池底/池壁的亮纹用
(define wt_update_normals
  (lambda ()
    {
      (define sh (if (== sim_cur 0) hA hB))
      (define lnx nxF) (define lnz nzF) (define lcf cauF)
      (define nn GRID_N) (define nm1 GN1) (define gain (* (* GNF 0.5) BUMP))
      (define yy 0) (define xx 0) (define rowoff 0) (define ii 0)
      (define lt 0) (define rt 0) (define up 0) (define dn 0)
      (define hh 0.0) (define gx 0.0) (define gz 0.0) (define t2 0.0)
      (define kk2 0.0) (define lap 0.0)
      (while (< yy nn)
        {
          (set! rowoff (* yy nn))
          (set! xx 0)
          (while (< xx nn)
            {
              (set! ii (+ rowoff xx))
              (set! lt (if (> xx 0) (- ii 1) ii))
              (set! rt (if (< xx nm1) (+ ii 1) ii))
              (set! up (if (> yy 0) (- ii nn) ii))
              (set! dn (if (< yy nm1) (+ ii nn) ii))
              (set! hh (get_item sh ii))
              (set! gx (* (- hh (get_item sh rt)) gain))
              (set! gz (* (- hh (get_item sh dn)) gain))
              (set! t2 (+ (* gx gx) (* gz gz)))
              (if (> t2 0.60) (set! t2 0.60))
              (set! kk2 (- 1.0 (* 0.5 t2)))
              (set_item! lnx ii (* gx kk2))
              (set_item! lnz ii (* gz kk2))
              (set! lap (- (+ (+ (get_item sh lt) (get_item sh rt))
                              (+ (get_item sh up) (get_item sh dn)))
                           (* 4.0 hh)))
              (set_item! lcf ii (+ 1.0 (* CAU_GAIN lap)))
              (set! xx (+ xx 1))
            })
          (set! yy (+ yy 1))
        })
    }))

;; 圆球排开的水柱高度（体积核，同原版）
(define wt_volume_in_sphere
  (lambda (ccx ccy ccz uu vv)
    {
      (define tx (- (- (* uu 2.0) 1.0) ccx))
      (define tz (- (- (* vv 2.0) 1.0) ccz))
      (define tt (/ (Math.sqrt (+ (* tx tx) (* tz tz))) ball_radius))
      (define q1 (* tt 1.5))
      (define q2 (* (* q1 q1) (* q1 q1)))
      (define dy (Math.exp (- 0.0 (* q2 q2))))
      (define ymin (fmin2 0.0 (- ccy dy)))
      (define ymax (fmin2 (fmax2 0.0 (+ ccy dy)) (+ ymin (* 2.0 dy))))
      (* (- ymax ymin) 0.1)
    }))

;; 圆球移动 → 排开/回填水体（只算球影响包围盒内的格子，原地改当前页）
(define wt_move_sphere
  (lambda (ox oy oz nx2 ny2 nz2)
    {
      (define ddx (- ox nx2))
      (define ddy (- oy ny2))
      (define ddz (- oz nz2))
      (if (< (+ (+ (* ddx ddx) (* ddy ddy)) (* ddz ddz)) 0.000000000001)
          #f
          {
            (define pg (if (== sim_cur 0) hA hB))
            (define ucx (* (+ nx2 1.0) 0.5))
            (define vcz (* (+ nz2 1.0) 0.5))
            (define urd (* (* ball_radius 1.6) 0.5))     ; 影响半径（uv 空间）
            (define bx0 (flr (* (- ucx urd) GNF)))
            (define bx1 (clr (* (+ ucx urd) GNF)))
            (define by0 (flr (* (- vcz urd) GNF)))
            (define by1 (clr (* (+ vcz urd) GNF)))
            (if (< bx0 0) (set! bx0 0))
            (if (> bx1 GN1) (set! bx1 GN1))
            (if (< by0 0) (set! by0 0))
            (if (> by1 GN1) (set! by1 GN1))
            (define yy3 by0)
            (while (<= yy3 by1)
              {
                (define xx3 bx0)
                (while (<= xx3 bx1)
                  {
                    (define ii3 (+ (* yy3 GRID_N) xx3))
                    (define uu3 (/ (* 1.0 xx3) GNF))
                    (define vv3 (/ (* 1.0 yy3) GNF))
                    (set_item! pg ii3
                      (+ (get_item pg ii3)
                         (- (wt_volume_in_sphere ox oy oz uu3 vv3)
                            (wt_volume_in_sphere nx2 ny2 nz2 uu3 vv3))))
                    (set! xx3 (+ xx3 1))
                  })
                (set! yy3 (+ yy3 1))
              })
          })
    }))

;; 圆球物理 + 一次仿真推进（顺序同原版 wt_update）
(define wt_update
  (lambda (seconds0)
    {
      (define seconds (fmin2 seconds0 1.0))
      (define pUW 0.0)
      (define vl 0.0)
      (define sdot 0.0)
      (if (== wt_mode MODE_SPHERE)
          { (set! ball_vx 0.0) (set! ball_vy 0.0) (set! ball_vz 0.0) }
          (if (== use_sphere_physics 1)
              {
                (set! pUW (clampf (/ (- ball_radius ball_cy) (* 2.0 ball_radius)) 0.0 1.0))
                (set! ball_vy (+ ball_vy (* (* ball_gravity_y seconds) (- 1.0 (* 1.1 pUW)))))
                (set! sdot (+ (+ (* ball_vx ball_vx) (* ball_vy ball_vy)) (* ball_vz ball_vz)))
                (set! vl (Math.sqrt sdot))
                (if (> vl 0.000001)
                    {
                      (define kk2 (* (* pUW seconds) sdot))
                      (set! ball_vx (- ball_vx (* (/ ball_vx vl) kk2)))
                      (set! ball_vy (- ball_vy (* (/ ball_vy vl) kk2)))
                      (set! ball_vz (- ball_vz (* (/ ball_vz vl) kk2)))
                    })
                (set! ball_cx (+ ball_cx (* ball_vx seconds)))
                (set! ball_cy (+ ball_cy (* ball_vy seconds)))
                (set! ball_cz (+ ball_cz (* ball_vz seconds)))
                (if (< ball_cy (- (- 0.0 ball_radius) 1.0))
                    {
                      (set! ball_cy (- (- 0.0 ball_radius) 1.0))
                      (set! ball_vy (* (Math.abs ball_vy) 0.7))
                    })
              }))
      (wt_move_sphere ball_ox ball_oy ball_oz ball_cx ball_cy ball_cz)
      (set! ball_ox ball_cx) (set! ball_oy ball_cy) (set! ball_oz ball_cz)
      (wt_step_sim)
      (wt_step_sim)
      ;; 每隔几拍在水面某处点一滴（否则水静下来后既看不到波、也看不到焦散）
      (set! drop_cnt (+ drop_cnt 1))
      (if (>= drop_cnt DROP_EVERY)
          {
            (set! drop_cnt 0)
            (define dpx (flr (* (Math.random) (- GRID_N 4.0))))
            (define dpy (flr (* (Math.random) (- GRID_N 4.0))))
            (define dpi (+ (* (+ dpy 2) GRID_N) (+ dpx 2)))
            (set_item! (if (== sim_cur 0) hA hB) dpi 0.025)
          })
      (wt_update_normals)
    }))

;; ============================================================================
;; 渲染：扫描行填充 + 线段 + 半透明叠加
;; ============================================================================

;; 四边形扫描行填充：4 个屏幕角点（q0..q3 依次）+ 首末行亮度系数（距离渐暗）
;;   qx0/qy0 … qx3/qy3：屏幕坐标；g0/g1：首/末行亮度系数（距离渐暗）
(define span_fill
  (lambda (qx0 qy0 qx1 qy1 qx2 qy2 qx3 qy3 g0 g1)
    {
      (define ymin (flr (fmin2 qy0 (fmin2 qy1 (fmin2 qy2 qy3)))))
      (define ymax (clr (fmax2 qy0 (fmax2 qy1 (fmax2 qy2 qy3)))))
      (if (< ymin 0) (set! ymin 0))
      (if (> ymax (- SH 1)) (set! ymax (- SH 1)))
      (if (> ymin ymax) (set! ymax (- ymin 1)))
      (define span 0.0)
      (define yi 0) (define e 0) (define xl 0.0) (define xr 0.0)
      (define ex0 0.0) (define ey0 0.0) (define ex1 0.0) (define ey1 0.0)
      (define tpar 0.0) (define xx 0.0)
      (define gr 1.0) (define rr 0.0) (define gg 0.0) (define bb 0.0)
      (define wspan 1.0) (define wty 0.0) (define wfr 0.0) (define wal 0.0)
      (set! yi ymin)
      (while (<= yi ymax)
        {
          (set! span (- ymax ymin))
          (set! gr (+ g0 (* (- g1 g0) (if (> span 0.0) (/ (- yi ymin) span) 1.0))))
          (set! xl 100000.0)
          (set! xr -100000.0)
          (set! xx (+ (* 1.0 yi) 0.5))
          ;; 4 条边与该行的交点（取 min/max 即扫描行跨度）
          (set! e 0)
          (while (< e 4)
            {
              (if (== e 0) { (set! ex0 qx0) (set! ey0 qy0) (set! ex1 qx1) (set! ey1 qy1) })
              (if (== e 1) { (set! ex0 qx1) (set! ey0 qy1) (set! ex1 qx2) (set! ey1 qy2) })
              (if (== e 2) { (set! ex0 qx2) (set! ey0 qy2) (set! ex1 qx3) (set! ey1 qy3) })
              (if (== e 3) { (set! ex0 qx3) (set! ey0 qy3) (set! ex1 qx0) (set! ey1 qy0) })
              (if (not (== ey0 ey1))
                  (if (or (and (< ey0 ey1) (and (>= xx ey0) (< xx ey1)))
                          (and (>= ey0 ey1) (and (>= xx ey1) (< xx ey0))))
                      {
                        (set! tpar (/ (- xx ey0) (- ey1 ey0)))
                        (set! tpar (+ ex0 (* tpar (- ex1 ex0))))
                        (if (< tpar xl) (set! xl tpar))
                        (if (> tpar xr) (set! xr tpar))
                      }))
              (set! e (+ e 1))
            })
          (if (< xl 1.0e8)
              {
                (if (< xl 0.0) (set! xl 0.0))
                (if (> xr (- SW 1)) (set! xr (- SW 1)))
                (if (>= xr xl)
                    {
                      (set! rr (* (* qr gr) 255.0))
                      (set! gg (* (* qg gr) 255.0))
                      (set! bb (* (* qb gr) 255.0))
                      (if (== qtint 1)
                          {
                            ;; 透过水面看：远处掠射 → 天光反射占主导（偏蓝、不透明），
                            ;; 近处俯视 → 清水，瓷砖本色。wty 用"水面在屏幕上的纵向范围"
                            (set! wspan (- wty1 wty0))
                            (set! wty (if (> wspan 1.0) (/ (- yi wty0) wspan) 1.0))
                            (if (< wty 0.0) (set! wty 0.0))
                            (if (> wty 1.0) (set! wty 1.0))
                            (set! wfr (* (- 1.0 wty) (- 1.0 wty)))
                            (set! wal (+ 0.10 (* 0.34 wfr)))
                            (set! rr (+ (* rr (- 1.0 wal)) (* 140.0 wal)))
                            (set! gg (+ (* gg (- 1.0 wal)) (* 186.0 wal)))
                            (set! bb (+ (* bb (- 1.0 wal)) (* 226.0 wal)))
                          })
                      (Gfx.fill_rect (flr xl) yi (+ (- (clr xr) (flr xl)) 1) 1 rr gg bb)
                    })
              })
          (set! yi (+ yi 1))
        })
    }))

;; 池壁：8 个角点投影一次 → 逐面（内法线朝向相机才画）扫描行填充 + 瓷砖缝
(define draw_cube
  (lambda ()
    {
      (define ci 0)
      (while (< ci 8)
        {
          (proj (get_item cube_v (* ci 3))
                (get_item cube_v (+ (* ci 3) 1))
                (get_item cube_v (+ (* ci 3) 2)))
          (set_item! csx ci v0sx)
          (set_item! csy ci v0sy)
          (set_item! csiw ci v0iw)
          (set! ci (+ ci 1))
        })
      (define fi 0)
      (while (< fi 5)
        {
          ;; 面（内）法线 → 定向光漫反射 → 基色（水下蓝绿染色）
          (define nix 0.0) (define niy 1.0) (define niz 0.0)
          (if (== fi 1) { (set! nix 1.0) (set! niy 0.0) })
          (if (== fi 2) { (set! nix -1.0) (set! niy 0.0) })
          (if (== fi 3) { (set! niz 1.0) (set! niy 0.0) })
          (if (== fi 4) { (set! niz -1.0) (set! niy 0.0) })
          ;; 面向相机？（用面中心判断，与逐像素光栅化等价）
          (define fwx (/ (+ (+ (get_item cube_v (* (get_item fquad (* fi 4)) 3))
                               (get_item cube_v (* (get_item fquad (+ (* fi 4) 2)) 3))) 1.0) 2.0))
          (define fwy (/ (+ (+ (get_item cube_v (+ (* (get_item fquad (* fi 4)) 3) 1))
                               (get_item cube_v (+ (* (get_item fquad (+ (* fi 4) 2)) 3) 1))) 1.0) 2.0))
          (define fwz (/ (+ (+ (get_item cube_v (+ (* (get_item fquad (* fi 4)) 3) 2))
                               (get_item cube_v (+ (* (get_item fquad (+ (* fi 4) 2)) 3) 2))) 1.0) 2.0))
          (define fvis (+ (+ (* nix (- eye_x fwx)) (* niy (- eye_y fwy))) (* niz (- eye_z fwz))))
          (if (> fvis 0.0)
              {
                (define fdf (+ (+ (* nix light_dir_x) (* niy light_dir_y)) (* niz light_dir_z)))
                (if (< fdf 0.0) (set! fdf 0.0))
                (define fbb (+ 0.20 (* 0.55 fdf)))
                ;; 4 个角点的屏幕坐标
                (define i0 (get_item fquad (* fi 4)))
                (define i1 (get_item fquad (+ (* fi 4) 1)))
                (define i2 (get_item fquad (+ (* fi 4) 2)))
                (define i3 (get_item fquad (+ (* fi 4) 3)))
                (set! qtint 1)
                (set! qr (* fbb 0.72)) (set! qg (* fbb 1.05)) (set! qb (* fbb 1.05))
                (span_fill (get_item csx i0) (get_item csy i0)
                           (get_item csx i1) (get_item csy i1)
                           (get_item csx i2) (get_item csy i2)
                           (get_item csx i3) (get_item csy i3)
                           0.75 1.02)
                ;; 池沿（水面以上那一条）单独用干瓷砖色再填一次
                (if (> fi 0)
                    {
                      (proj (get_item rq (* fi 12)) (get_item rq (+ (* fi 12) 1)) (get_item rq (+ (* fi 12) 2)))
                      (define r0x v0sx) (define r0y v0sy)
                      (proj (get_item rq (+ (* fi 12) 3)) (get_item rq (+ (* fi 12) 4)) (get_item rq (+ (* fi 12) 5)))
                      (define r1x v0sx) (define r1y v0sy)
                      (proj (get_item rq (+ (* fi 12) 6)) (get_item rq (+ (* fi 12) 7)) (get_item rq (+ (* fi 12) 8)))
                      (define r2x v0sx) (define r2y v0sy)
                      (proj (get_item rq (+ (* fi 12) 9)) (get_item rq (+ (* fi 12) 10)) (get_item rq (+ (* fi 12) 11)))
                      (set! qtint 0)
                      (set! qr (* fbb 1.30)) (set! qg (* fbb 1.34)) (set! qb (* fbb 1.40))
                      (span_fill r0x r0y r1x r1y r2x r2y v0sx v0sy 0.95 1.05)
                    })
                ;; 瓷砖缝：面是平面 ⇒ 世界直线投影仍是直线，直接 draw_line
                ;;   u = k/TILE_N 与 v = k/TILE_N 的等值线
                ;;   —— 池底（fi=0）特殊处理：水面下的瓷砖应该被波浪折射"扭动"，
                ;;   所以每条缝采样成折线，顶点按波面法线偏移（世界 xz），再逐段画。
                (if (== fi 0)
                    {
                      (define gk 1)
                      (while (< gk 5)
                        {
                          (define gpos (/ (* 1.0 gk) TILE_N))
                          ;; 两个方向的等值线各画一条
                          (define dir 0)
                          (while (< dir 2)
                            {
                              (define seg 0)
                              (define prevx 0.0) (define prevy 0.0) (define prevw 1.0)
                              (while (<= seg 12)
                                {
                                  (define tt2 (/ (* 1.0 seg) 12.0))
                                  (define gwx 0.0) (define gwz 0.0)
                                  (if (== dir 0)
                                      { (set! gwx (- (* 2.0 gpos) 1.0)) (set! gwz (- (* 2.0 tt2) 1.0)) }
                                      { (set! gwx (- (* 2.0 tt2) 1.0)) (set! gwz (- (* 2.0 gpos) 1.0)) })
                                  (wave_normal gwx gwz)
                                  (set! gwx (+ gwx (* wnx WOB)))
                                  (set! gwz (+ gwz (* wnz WOB)))
                                  (proj gwx -1.0 gwz)
                                  (if (> seg 0)
                                      (if (and (> prevw 0.0) (> v0iw 0.0))
                                          (Gfx.draw_line prevx prevy v0sx v0sy 14 26 30)))
                                  (set! prevx v0sx) (set! prevy v0sy) (set! prevw v0iw)
                                  (set! seg (+ seg 1))
                                })
                              (set! dir (+ dir 1))
                            })
                          (set! gk (+ gk 1))
                        })
                    })
                (define u0 (if (== fi 0) 0.0 0.5))
                (define u1 (if (== fi 0) 1.0 (+ 1.0 (/ 1.0 12.0))))
                (define k 1)
                (while (< k 10)
                  {
                    (define uu (/ (* 1.0 k) TILE_N))
                    (if (and (> fi 0) (and (> uu u0) (< uu u1)))
                        {
                          ;; u 等值线：c0→c1 与 c3→c2 上同参数 t 的两点
                          (define tt (/ (- uu u0) (- u1 u0)))
                          (define p0x (+ (get_item fq (* fi 12))
                                         (* tt (- (get_item fq (+ (* fi 12) 3)) (get_item fq (* fi 12))))))
                          (define p0y (+ (get_item fq (+ (* fi 12) 1))
                                         (* tt (- (get_item fq (+ (* fi 12) 4)) (get_item fq (+ (* fi 12) 1))))))
                          (define p0z (+ (get_item fq (+ (* fi 12) 2))
                                         (* tt (- (get_item fq (+ (* fi 12) 5)) (get_item fq (+ (* fi 12) 2))))))
                          (define p1x (+ (get_item fq (+ (* fi 12) 9))
                                         (* tt (- (get_item fq (+ (* fi 12) 6)) (get_item fq (+ (* fi 12) 9))))))
                          (define p1y (+ (get_item fq (+ (* fi 12) 10))
                                         (* tt (- (get_item fq (+ (* fi 12) 7)) (get_item fq (+ (* fi 12) 10))))))
                          (define p1z (+ (get_item fq (+ (* fi 12) 11))
                                         (* tt (- (get_item fq (+ (* fi 12) 8)) (get_item fq (+ (* fi 12) 11))))))
                          (proj p0x p0y p0z)
                          (define q0sx v0sx) (define q0sy v0sy) (define q0w v0iw)
                          (proj p1x p1y p1z)
                          (if (and (> q0w 0.0) (> v0iw 0.0))
                              (Gfx.draw_line q0sx q0sy v0sx v0sy 14 26 30))
                        })
                    (set! k (+ k 1))
                  })
                (define k2 1)
                (while (< k2 10)
                  {
                    (define vv2 (/ (* 1.0 k2) TILE_N))
                    (if (and (> fi 0) (< vv2 1.0))
                        {
                          ;; v 等值线：c0→c3 与 c1→c2 上同参数 s 的两点
                          (define ss vv2)
                          (define gx0 (+ (get_item fq (* fi 12))
                                         (* ss (- (get_item fq (+ (* fi 12) 9)) (get_item fq (* fi 12))))))
                          (define gy0 (+ (get_item fq (+ (* fi 12) 1))
                                         (* ss (- (get_item fq (+ (* fi 12) 10)) (get_item fq (+ (* fi 12) 1))))))
                          (define gz0 (+ (get_item fq (+ (* fi 12) 2))
                                         (* ss (- (get_item fq (+ (* fi 12) 11)) (get_item fq (+ (* fi 12) 2))))))
                          (define gx1 (+ (get_item fq (+ (* fi 12) 3))
                                         (* ss (- (get_item fq (+ (* fi 12) 6)) (get_item fq (+ (* fi 12) 3))))))
                          (define gy1 (+ (get_item fq (+ (* fi 12) 4))
                                         (* ss (- (get_item fq (+ (* fi 12) 7)) (get_item fq (+ (* fi 12) 4))))))
                          (define gz1 (+ (get_item fq (+ (* fi 12) 5))
                                         (* ss (- (get_item fq (+ (* fi 12) 8)) (get_item fq (+ (* fi 12) 5))))))
                          (proj gx0 gy0 gz0)
                          (define r0sx v0sx) (define r0sy v0sy) (define r0w v0iw)
                          (proj gx1 gy1 gz1)
                          (if (and (> r0w 0.0) (> v0iw 0.0))
                              (Gfx.draw_line r0sx r0sy v0sx v0sy 14 26 30))
                        })
                    (set! k2 (+ k2 1))
                  })
              })
          (set! fi (+ fi 1))
        })
    }))

;; ============================================================================
;; 水面与焦散
;;
;; 这两个效果都是"逐像素"性质的东西，但解释器逐像素不可能达标（见文件头）。做法：
;;   · 水面：在屏幕上按 CS×CS 的**格子**逐格算（波场法线 → Fresnel → 天光色 +
;;     太阳高光），每格一次 alpha 混合的 rounded_rect —— 透过它看到的仍是池底瓷砖。
;;     格子只在水面四边形的屏幕范围 ∩ 池口 [-1,1]² 内画，因此边界是精确的。
;;   · 焦散：光穿过波动的水面在水下表面聚焦出的亮纹。对每个可见的水下表面
;;     （池底/池壁），在其 (u,v) 矩形上取网格点 → 用"齐次双线性 + 透视除法"投到屏幕
;;     → 沿**折射后的平行光**把该点投回水面，取那里的波面曲率（拉普拉斯）当亮度
;;     → 画一个 3~5px 的 alpha 亮点。波场一动，整片光影就跟着流动。
;; ============================================================================

;; 屏幕点 → 世界视线方向（放 dvx/dvy/dvz）
(define dvx 0.0) (define dvy 0.0) (define dvz 0.0)
(define ray_dir
  (lambda (csx0 csy0)
    {
      (define ndx (- (/ (* 2.0 csx0) (* 1.0 SW)) 1.0))
      (define ndy (- 1.0 (/ (* 2.0 csy0) (* 1.0 SH))))
      (set! dvx (+ (+ vfx (* vrx (* ndx kxf))) (* vux (* ndy kyf))))
      (set! dvy (+ (+ vfy (* vry (* ndx kxf))) (* vuy (* ndy kyf))))
      (set! dvz (+ (+ vfz (* vrz (* ndx kxf))) (* vuz (* ndy kyf))))
    }))

;; 一次焦散网格点：屏幕格中心 (gx,gy) 已经由调用方给出 uv 反算
(define draw_water
  (lambda ()
    {
      (define lnx nxF) (define lnz nzF)
      (define ldx light_dir_x) (define ldy light_dir_y) (define ldz light_dir_z)
      ;; 水面四角（y=0）→ 屏幕 bbox
      (proj -1.0 0.0 -1.0) (define w0x v0sx) (define w0y v0sy)
      (proj 1.0 0.0 -1.0)  (define w1x v0sx) (define w1y v0sy)
      (proj 1.0 0.0 1.0)   (define w2x v0sx) (define w2y v0sy)
      (proj -1.0 0.0 1.0)  (define w3x v0sx) (define w3y v0sy)
      (define bx0 (flr (fmin2 w0x (fmin2 w1x (fmin2 w2x w3x)))))
      (define bx1 (clr (fmax2 w0x (fmax2 w1x (fmax2 w2x w3x)))))
      (define by0 (flr (fmin2 w0y (fmin2 w1y (fmin2 w2y w3y)))))
      (define by1 (clr (fmax2 w0y (fmax2 w1y (fmax2 w2y w3y)))))
      (if (< bx0 0) (set! bx0 0))
      (if (> bx1 (- SW 1)) (set! bx1 (- SW 1)))
      (if (< by0 0) (set! by0 0))
      (if (> by1 (- SH 1)) (set! by1 (- SH 1)))
      (define cs 5)
      (define sx 0) (define sy 0)
      (define ccx 0.0) (define ccy 0.0)
      (define tpl 0.0) (define wxp 0.0) (define wzp 0.0)
      (define uu 0.0) (define vv 0.0) (define ffx 0.0) (define ffy 0.0)
      (define ixi 0.0) (define iyi 0.0) (define gxi 0.0) (define gyi 0.0)
      (define w00 0.0) (define w10 0.0) (define w01 0.0) (define w11 0.0) (define sii 0.0)
      (define nxv 0.0) (define nyv 1.0) (define nzv 0.0)
      (define dl 1.0) (define cvx 0.0) (define cvy 0.0) (define cvz 0.0)
      (define cdt 0.0) (define tfv 0.0) (define fres 0.0) (define ryv 0.0)
      (define stp 0.0) (define sdn 0.0) (define sd2 0.0) (define sd4 0.0) (define spe 0.0)
      (define wr 0.0) (define wg 0.0) (define wb 0.0) (define wa 0.0)
      (set! sy by0)
      (while (<= sy by1)
        {
          (set! sx bx0)
          (while (<= sx bx1)
            {
              (set! ccx (+ (* 1.0 sx) (* 0.5 (* 1.0 cs))))
              (set! ccy (+ (* 1.0 sy) (* 0.5 (* 1.0 cs))))
              (ray_dir ccx ccy)
              (if (< dvy -0.02)
                  {
                    (set! tpl (/ (- 0.0 eye_y) dvy))
                    (set! wxp (+ eye_x (* dvx tpl)))
                    (set! wzp (+ eye_z (* dvz tpl)))
                    (if (and (>= wxp -1.0) (and (<= wxp 1.0) (and (>= wzp -1.0) (<= wzp 1.0))))
                        {
                          ;; 波场法线（双线性；ny 用 1-(nx²+nz²)/2 解出）
                          (set! uu (* (+ wxp 1.0) 0.5))
                          (set! vv (* (+ wzp 1.0) 0.5))
                          (set! ffx (- (* uu GNF) 0.5))
                          (set! ffy (- (* vv GNF) 0.5))
                          (set! ixi (flr ffx))
                          (set! iyi (flr ffy))
                          (if (< ixi 0.0) (set! ixi 0.0))
                          (if (> ixi (- GRID_N 2)) (set! ixi (- GRID_N 2)))
                          (if (< iyi 0.0) (set! iyi 0.0))
                          (if (> iyi (- GRID_N 2)) (set! iyi (- GRID_N 2)))
                          (set! gxi (- ffx ixi))
                          (set! gyi (- ffy iyi))
                          (set! w00 (* (- 1.0 gxi) (- 1.0 gyi)))
                          (set! w10 (* gxi (- 1.0 gyi)))
                          (set! w01 (* (- 1.0 gxi) gyi))
                          (set! w11 (* gxi gyi))
                          (set! sii (+ (* iyi GNF) ixi))
                          (set! nxv (+ (+ (* (get_item lnx sii) w00)
                                         (* (get_item lnx (+ sii 1.0)) w10))
                                       (+ (* (get_item lnx (+ sii GNF)) w01)
                                          (* (get_item lnx (+ (+ sii GNF) 1.0)) w11))))
                          (set! nzv (+ (+ (* (get_item lnz sii) w00)
                                         (* (get_item lnz (+ sii 1.0)) w10))
                                       (+ (* (get_item lnz (+ sii GNF)) w01)
                                          (* (get_item lnz (+ (+ sii GNF) 1.0)) w11))))
                          (set! nyv (- 1.0 (* 0.5 (+ (* nxv nxv) (* nzv nzv)))))
                          ;; 视线单位化 → cosθ（-1 是让法线与视线同侧）
                          (set! dl (Math.sqrt (+ (+ (* dvx dvx) (* dvy dvy)) (* dvz dvz))))
                          (set! cvx (/ dvx dl)) (set! cvy (/ dvy dl)) (set! cvz (/ dvz dl))
                          (set! cdt (- 0.0 (+ (+ (* nxv cvx) (* nyv cvy)) (* nzv cvz))))
                          (if (< cdt 0.0)
                              {
                                (set! nxv (- 0.0 nxv)) (set! nyv (- 0.0 nyv))
                                (set! nzv (- 0.0 nzv)) (set! cdt (- 0.0 cdt))
                              })
                          ;; Fresnel：0.22 + 0.78*(1-cos)^3（同原版 waterShader 的量级）
                          (set! tfv (- 1.0 cdt))
                          (set! fres (+ 0.22 (* 0.78 (* (* tfv tfv) tfv))))
                          ;; 反射方向 y → 天光色
                          (set! ryv (+ cvy (* (* 2.0 cdt) nyv)))
                          (set! stp (fmax2 0.0 ryv))
                          ;; 太阳高光：dot(n,L)^4（^4 而不是 ^8：光带更宽，波动更显眼）
                          (set! spe 0.0)
                          (set! sdn (+ (+ (* nxv ldx) (* nyv ldy)) (* nzv ldz)))
                          (if (> sdn 0.0)
                              {
                                (set! sd2 (* sdn sdn))
                                (set! sd4 (* sd2 sd2))
                                (set! spe (* sd4 1.35))
                              })
                          ;; 混色：低 fres = 水体青绿（透出瓷砖）；高 fres = 天光反射
                          (set! wr (+ (* 64.0 (- 1.0 fres))
                                      (* (* (+ 0.34 (* 0.34 stp)) 255.0) fres)))
                          (set! wg (+ (* 146.0 (- 1.0 fres))
                                      (* (* (+ 0.56 (* 0.30 stp)) 255.0) fres)))
                          (set! wb (+ (* 164.0 (- 1.0 fres))
                                      (* (* (+ 0.86 (* 0.14 stp)) 255.0) fres)))
                          (set! wr (+ wr (* spe 235.0)))
                          (set! wg (+ wg (* spe 246.0)))
                          (set! wb (+ wb (* spe 252.0)))
                          (set! wa (+ 46.0 (* 150.0 fres)))
                          (Gfx.rounded_rect sx sy cs cs 0 0 0 0 wr wg wb (flr wa))
                        })
                  })
              (set! sx (+ sx cs))
            })
          (set! sy (+ sy cs))
        })

      ;; 太阳波光碎点（波面斜率大的地方点一颗亮斑）
      (define gp 0)
      (while (< gp 48)
        {
          (define gu (/ (+ (* 1.0 (mod gp 8)) 0.5) 8.0))
          (define gv (/ (+ (* 1.0 (flr (/ (* 1.0 gp) 8.0))) 0.5) 6.0))
          (proj (- (* gu 2.0) 1.0) 0.0 (- (* gv 2.0) 1.0))
          (if (and (> v0iw 0.0) (and (>= v0sy 0.0) (< v0sy (* 1.0 SH))))
              {
                (define gix (if (< gu 0.999) (flr (* gu GNF)) (- GRID_N 2)))
                (define giy (if (< gv 0.999) (flr (* gv GNF)) (- GRID_N 2)))
                (if (and (>= gix 0) (>= giy 0))
                    {
                      (define gi (+ (* giy GRID_N) gix))
                      (define gsl (- (get_item nxF gi) (get_item nxF (+ gi 1))))
                      (define gbr (* (* gsl nxv) 0.0))
                      (set! gbr (- (get_item nxF gi) (get_item nxF (+ gi 1))))
                      (set! gbr (* gbr 6.0))
                      (if (> gbr 0.5)
                          {
                            (if (> gbr 1.0) (set! gbr 1.0))
                            (Gfx.draw_line (- v0sx 4.0) v0sy (+ v0sx 4.0) v0sy
                                           (+ 120.0 (* 135.0 gbr))
                                           (+ 165.0 (* 90.0 gbr))
                                           (+ 185.0 (* 70.0 gbr)))
                          })
                    })
              })
          (set! gp (+ gp 1))
        })
    }))

;; ============================================================================
;; 焦散：光穿过波动的水面，在水下表面（池底/池壁）聚焦出的流动亮纹
;; ============================================================================

;; 一个网格点：uv → 屏幕（齐次双线性 + 透视除法）→ 世界 → 沿折射光回到水面
;;   → 取波场曲率当亮度。参数：面号 fi、(u,v)、该面 4 角的屏幕齐次坐标。
(define caustic_cell
  (lambda (fi uv vv2 h00x h00y h00w h10x h10y h10w h11x h11y h11w h01x h01y h01w)
    {
      ;; 齐次双线性插值
      (define wu (- 1.0 uv)) (define wv (- 1.0 vv2))
      (define px (+ (+ (+ (* (* wu wv) h00x) (* (* uv wv) h10x))
                         (* (* uv vv2) h11x))
                    (* (* wu vv2) h01x)))
      (define py (+ (+ (+ (* (* wu wv) h00y) (* (* uv wv) h10y))
                         (* (* uv vv2) h11y))
                    (* (* wu vv2) h01y)))
      (define pw (+ (+ (+ (* (* wu wv) h00w) (* (* uv wv) h10w))
                         (* (* uv vv2) h11w))
                    (* (* wu vv2) h01w)))
      (if (> pw 0.001)
          {
            (define ipw (/ 1.0 pw))
            (define sxp (* px ipw))
            (define syp (* py ipw))
            ;; uv → 世界（每个面都是 uv 矩形，映射是仿射的）
            (define wx 0.0) (define wy -1.0) (define wz 0.0)
            (if (== fi 0) { (set! wx (- (* 2.0 uv) 1.0)) (set! wz (- (* 2.0 vv2) 1.0)) })
            (if (== fi 1) { (set! wx -1.0) (set! wy (- (* 2.0 uv) 2.0)) (set! wz (- (* 2.0 vv2) 1.0)) })
            (if (== fi 2) { (set! wx 1.0) (set! wy (- (* 2.0 uv) 2.0)) (set! wz (- (* 2.0 vv2) 1.0)) })
            (if (== fi 3) { (set! wx (- (* 2.0 vv2) 1.0)) (set! wy (- (* 2.0 uv) 2.0)) (set! wz -1.0) })
            (if (== fi 4) { (set! wx (- (* 2.0 vv2) 1.0)) (set! wy (- (* 2.0 uv) 2.0)) (set! wz 1.0) })
            ;; 沿折射后的平行光投回水面：P_s = P - rl*(y/rl_y)
            (define tt (/ wy rly))
            (define qx (- wx (* rlx tt)))
            (define qz (- wz (* rlz tt)))
            (if (and (>= qx -1.0) (and (<= qx 1.0) (and (>= qz -1.0) (<= qz 1.0))))
                {
                  ;; 焦散强度场（双线性）
                  (define uq (* (+ qx 1.0) 0.5))
                  (define vq (* (+ qz 1.0) 0.5))
                  (define fqx (- (* uq GNF) 0.5))
                  (define fqy (- (* vq GNF) 0.5))
                  (define iqx (flr fqx))
                  (define iqy (flr fqy))
                  (if (< iqx 0.0) (set! iqx 0.0))
                  (if (> iqx (- GRID_N 2)) (set! iqx (- GRID_N 2)))
                  (if (< iqy 0.0) (set! iqy 0.0))
                  (if (> iqy (- GRID_N 2)) (set! iqy (- GRID_N 2)))
                  (define gqx (- fqx iqx))
                  (define gqy (- fqy iqy))
                  (define q00 (* (- 1.0 gqx) (- 1.0 gqy)))
                  (define q10 (* gqx (- 1.0 gqy)))
                  (define q01 (* (- 1.0 gqx) gqy))
                  (define q11 (* gqx gqy))
                  (define qii (+ (* iqy GNF) iqx))
                  (define bri (+ (+ (* (get_item cauF qii) q00)
                                    (* (get_item cauF (+ qii 1.0)) q10))
                                 (+ (* (get_item cauF (+ qii GNF)) q01)
                                    (* (get_item cauF (+ (+ qii GNF) 1.0)) q11))))
                  ;; 亮度 → 一个 3~5px 的 alpha 亮点（亮处才是光聚焦的地方）
                  (if (> bri 1.02)
                      {
                        (define bb2 (- bri 1.0))
                        (define al2 (* bb2 120.0))
                        (if (> al2 150.0) (set! al2 150.0))
                        (define sz (+ 4.0 (* bb2 7.0)))
                        (if (> sz 9.0) (set! sz 9.0))
                        (define hz (* 0.5 sz))
                        (Gfx.rounded_rect (- sxp hz) (- syp hz) sz sz
                                          hz hz hz hz 224 242 255 (flr al2))
                      })
                })
          })
    }))

;; 逐面撒焦散亮点（网格密度按面的屏幕大小给）
(define draw_caustics
  (lambda ()
    {
      (define fi 0)
      (while (< fi 5)
        {
          (define nix 0.0) (define niy 1.0) (define niz 0.0)
          (if (== fi 1) { (set! nix 1.0) (set! niy 0.0) })
          (if (== fi 2) { (set! nix -1.0) (set! niy 0.0) })
          (if (== fi 3) { (set! niz 1.0) (set! niy 0.0) })
          (if (== fi 4) { (set! niz -1.0) (set! niy 0.0) })
          (define i0 (get_item fquad (* fi 4)))
          (define i2 (get_item fquad (+ (* fi 4) 2)))
          (define fwx (/ (+ (get_item cube_v (* i0 3)) (get_item cube_v (* i2 3))) 2.0))
          (define fwy (/ (+ (get_item cube_v (+ (* i0 3) 1))
                            (get_item cube_v (+ (* i2 3) 1))) 2.0))
          (define fwz (/ (+ (get_item cube_v (+ (* i0 3) 2))
                            (get_item cube_v (+ (* i2 3) 2))) 2.0))
          (define fvis (+ (+ (* nix (- eye_x fwx)) (* niy (- eye_y fwy))) (* niz (- eye_z fwz))))
          (if (> fvis 0.0)
              {
                (define j0 (get_item fquad (* fi 4)))
                (define j1 (get_item fquad (+ (* fi 4) 1)))
                (define j2 (get_item fquad (+ (* fi 4) 2)))
                (define j3 (get_item fquad (+ (* fi 4) 3)))
                (define a00x (* (get_item csx j0) (get_item csiw j0)))
                (define a00y (* (get_item csy j0) (get_item csiw j0)))
                (define a00w (get_item csiw j0))
                (define a10x (* (get_item csx j1) (get_item csiw j1)))
                (define a10y (* (get_item csy j1) (get_item csiw j1)))
                (define a10w (get_item csiw j1))
                (define a11x (* (get_item csx j2) (get_item csiw j2)))
                (define a11y (* (get_item csy j2) (get_item csiw j2)))
                (define a11w (get_item csiw j2))
                (define a01x (* (get_item csx j3) (get_item csiw j3)))
                (define a01y (* (get_item csy j3) (get_item csiw j3)))
                (define a01w (get_item csiw j3))
                (define nu 0) (define nv 0)
                (if (== fi 0) { (set! nu 22) (set! nv 13) } { (set! nu 9) (set! nv 10) })
                (define iu 0) (define iv2 0)
                (set! iv2 0)
                (while (< iv2 nv)
                  {
                    (set! iu 0)
                    (while (< iu nu)
                      {
                        (caustic_cell fi (/ (+ (* 1.0 iu) 0.5) (* 1.0 nu))
                                      (/ (+ (* 1.0 iv2) 0.5) (* 1.0 nv))
                                      a00x a00y a00w a10x a10y a10w a11x a11y a11w a01x a01y a01w)
                        (set! iu (+ iu 1))
                      })
                    (set! iv2 (+ iv2 1))
                  })
              })
          (set! fi (+ fi 1))
        })
    }))

;; 圆球：屏幕空间 impostor（投影圆心 + 屏幕半径），5 次 native 画完（三档明暗 + 高光 + 暗环）
(define draw_ball
  (lambda ()
    {
      (proj ball_cx ball_cy ball_cz)
      (set! bsx v0sx) (set! bsy v0sy)
      (define rwx (get_item matMV 0)) (define rwy (get_item matMV 1)) (define rwz (get_item matMV 2))
      (define uwx (get_item matMV 4)) (define uwy (get_item matMV 5)) (define uwz (get_item matMV 6))
      (proj (+ ball_cx (* rwx ball_radius)) (+ ball_cy (* rwy ball_radius)) (+ ball_cz (* rwz ball_radius)))
      (set! brx (Math.abs (- v0sx bsx)))
      (proj (+ ball_cx (* uwx ball_radius)) (+ ball_cy (* uwy ball_radius)) (+ ball_cz (* uwz ball_radius)))
      (set! bry (Math.abs (- v0sy bsy)))
      (define rad (* 0.5 (+ brx bry)))
      (if (< rad 1.0) (set! rad 1.0))
      ;; 定向光：球体基色 + 光侧高光 + 边缘暗环
      (define bdif (+ (+ (* 0.0 light_dir_x) (* 1.0 light_dir_y)) (* 0.0 light_dir_z)))
      (if (< bdif 0.0) (set! bdif 0.0))
      (define bbas (+ 0.34 (* 0.26 bdif)))
      (define bcr (* bbas 0.62)) (define bcg (* bbas 1.05)) (define bcb (* bbas 1.15))
      ;; 吃水：球心在 y<0 时整体染成水下色（水面叠加层随后还会罩一层）
      (if (< ball_cy 0.0)
          {
            (set! bcr (* bbas 0.42))
            (set! bcg (* bbas 0.85))
            (set! bcb (* bbas 0.95))
          })
      ;; 受光方向在屏幕上的偏移（用于高光位置）
      (proj (+ ball_cx (* light_dir_x ball_radius)) (+ ball_cy (* light_dir_y ball_radius)) (+ ball_cz (* light_dir_z ball_radius)))
      (define lox (- v0sx bsx)) (define loy (- v0sy bsy))
      (define llen (Math.sqrt (+ (* lox lox) (* loy loy))))
      (if (> llen 0.01) { (set! lox (/ lox llen)) (set! loy (/ loy llen)) })
      (Gfx.fill_circle bsx bsy (* rad 0.99) (* bcr 255.0) (* bcg 255.0) (* bcb 255.0))
      (Gfx.fill_circle (+ bsx (* lox (* rad 0.16))) (+ bsy (* loy (* rad 0.16))) (* rad 0.80)
                       (Math.floor (* bcr 300.0)) (Math.floor (* bcg 290.0)) (Math.floor (* bcb 285.0)))
      (Gfx.fill_circle (+ bsx (* lox (* rad 0.34))) (+ bsy (* loy (* rad 0.34))) (* rad 0.46)
                       226 242 250)
      (Gfx.fill_circle (+ bsx (* lox (* rad 0.40))) (+ bsy (* loy (* rad 0.40))) (* rad 0.22)
                       252 255 255)
      (Gfx.draw_circle bsx bsy (* rad 0.99) (* bcr 150.0) (* bcg 150.0) (* bcb 150.0))
    }))

;; 天空背景（水平渐变条 + 太阳光晕）
(define draw_sky
  (lambda ()
    {
      (define nb 24)
      (define bh (clr (/ (* 1.0 SH) (* 1.0 nb))))
      (define i 0)
      (while (< i nb)
        {
          (define ty (/ (* 1.0 i) (* 1.0 nb)))
          (define dp (* ty ty))
          (define sr (+ 0.055 (* 0.30 dp)))
          (define sg (+ 0.13 (* 0.40 dp)))
          (define sb (+ 0.24 (* 0.46 dp)))
          (Gfx.fill_rect 0 (* i bh) SW bh (* sr 255.0) (* sg 255.0) (* sb 255.0))
          (set! i (+ i 1))
        })
      (proj (* light_dir_x 40.0) (* light_dir_y 40.0) (* light_dir_z 40.0))
      (if (> v0iw 0.0)
          {
            (define sunx v0sx)
            (define suny v0sy)
            (Gfx.fill_circle sunx suny 26 190 208 236)
            (Gfx.fill_circle sunx suny 13 242 248 255)
          })
    }))

;; 一帧：天空 → 池壁/池底（填充+池沿+瓷砖缝）→ 焦散 → 水面（逐格 fresnel/高光）→ 圆球
(define wt_scene_render
  (lambda ()
    {
      (draw_sky)
      (draw_cube)
      (draw_caustics)
      (draw_water)
      (draw_ball)
    }))

;; ============================================================================
;; 控件栏（与 waterpool.scm 逐句一致）
;; ============================================================================

(define btx 0) (define bty 0) (define btw 0) (define bth 0)
(define wt_button_rect
  (lambda (idx)
    {
      (define n (if (> BTN_N 0) BTN_N 1))
      (define bw (Math.floor (/ (- SW (* BTN_GAP (+ n 1))) n)))
      (if (< bw 1) (set! bw 1))
      (define bh (- BAR_H BTN_GAP))
      (if (< bh 1) (set! bh 1))
      (set! btx (+ BTN_GAP (* idx (+ bw BTN_GAP))))
      (set! bty (+ (- SH BAR_H) (Math.floor (/ (* 1.0 BTN_GAP) 2))))
      (set! btw bw)
      (set! bth bh)
    }))

(define wt_button_hit
  (lambda (px py)
    {
      (define hitb -1)
      (if (<= SW 0) #f
        (if (<= SH 0) #f
          {
            (define i 0)
            (while (< i BTN_N)
              {
                (wt_button_rect i)
                (if (>= px btx)
                    (if (< px (+ btx btw))
                        (if (>= py bty)
                            (if (< py (+ bty bth)) (set! hitb i))
                            #f)
                        #f)
                    #f)
                (set! i (+ i 1))
              })
          }))
      hitb
    }))

;; 定向光源：与原版 L 键一致（当前视角正对方向作为平行光）
(define wt_light_follow_view
  (lambda ()
    {
      (set! light_dir_x (* (Math.cos (* (/ (- 90.0 angleY) 180.0) (Math.PI)))
                           (Math.cos (* (/ (- 0.0 angleX) 180.0) (Math.PI)))))
      (set! light_dir_y (Math.sin (* (/ (- 0.0 angleX) 180.0) (Math.PI))))
      (set! light_dir_z (* (Math.sin (* (/ (- 90.0 angleY) 180.0) (Math.PI)))
                           (Math.cos (* (/ (- 0.0 angleX) 180.0) (Math.PI)))))
    }))

(define wt_button_action
  (lambda (idx)
    {
      (if (== idx BTN_PAUSE) (set! wt_paused (- 1 wt_paused)))
      (if (== idx BTN_GRAVITY) (set! use_sphere_physics (- 1 use_sphere_physics)))
      (if (== idx BTN_LIGHT) (wt_light_follow_view))
    }))

(define wt_draw_ui
  (lambda ()
    {
      (Gfx.rounded_rect 0 (- SH BAR_H) SW BAR_H 0 0 0 0 0 0 0 110)
      (define i 0)
      (while (< i BTN_N)
        {
          (wt_button_rect i)
          (define label "")
          (define active 0)
          (if (== i BTN_BACK) (set! label "返回"))
          (if (== i BTN_PAUSE)
              {
                (if (== wt_paused 1) (set! label "继续") (set! label "暂停"))
                (set! active wt_paused)
              })
          (if (== i BTN_GRAVITY)
              {
                (if (== use_sphere_physics 1) (set! label "重力开") (set! label "重力关"))
                (set! active use_sphere_physics)
              })
          (if (== i BTN_LIGHT) (set! label "光源"))
          (define pressed (if (== pressed_btn i) 1 0))
          (define cr 34) (define cg 46) (define cb 62)
          (if (== active 1) { (set! cr 28) (set! cg 104) (set! cb 92) })
          (if (== pressed 1) { (set! cr 116) (set! cg 172) (set! cb 226) })
          (Gfx.rounded_rect btx bty btw bth 7 7 7 7 cr cg cb (if (== pressed 1) 240 200))
          (Gfx.draw_text_centered 1 label
                                  (Math.floor (+ btx (Math.floor (/ (* 1.0 btw) 2))))
                                  (Math.floor (+ bty (Math.floor (/ (* 1.0 bth) 2))))
                                  240 246 255 1)
          (set! i (+ i 1))
        })
    }))

;; ============================================================================
;; 交互（与 waterpool.scm 逐句一致）
;; ============================================================================

(define touch_prev 0)
(define down_x 0) (define down_y 0)

(define wt_hit_test_sphere
  (lambda (px py)
    {
      (wt_get_ray_for_pixel px py)
      (define rdx rx) (define rdy ry) (define rdz rz)
      (define tsx (- eye_x ball_cx))
      (define tsy (- eye_y ball_cy))
      (define tsz (- eye_z ball_cz))
      (define aa (+ (+ (* rdx rdx) (* rdy rdy)) (* rdz rdz)))
      (define bb (* 2.0 (+ (+ (* tsx rdx) (* tsy rdy)) (* tsz rdz))))
      (define cc2 (+ (+ (* tsx tsx) (* tsy tsy)) (* tsz tsz)))
      (define cc (- cc2 (* ball_radius ball_radius)))
      (define disc (- (* bb bb) (* 4.0 (* aa cc))))
      (define okh 0)
      (if (> disc 0.0)
          {
            (define tt (/ (- (- 0.0 bb) (Math.sqrt disc)) (* 2.0 aa)))
            (if (> tt 0.0)
                {
                  (set! prev_hit_x (+ eye_x (* rdx tt)))
                  (set! prev_hit_y (+ eye_y (* rdy tt)))
                  (set! prev_hit_z (+ eye_z (* rdz tt)))
                  (set! okh 1)
                })
          })
      okh
    }))

(define wt_start_drag
  (lambda (px py)
    {
      (set! prev_px px) (set! prev_py py)
      (define btn (wt_button_hit px py))
      (if (>= btn 0)
          {
            (set! wt_mode MODE_BUTTON)
            (set! pressed_btn btn)
            (set! mouse_down 1)
          }
          {
            (if (== (wt_hit_test_sphere px py) 1)
                {
                  (set! wt_mode MODE_SPHERE)
                  ;; planeNormal = -ray(屏幕中心)
                  (wt_get_ray_for_pixel (Math.floor (/ (* 1.0 SW) 2)) (Math.floor (/ (* 1.0 SH) 2)))
                  (set! plane_n_x (- 0.0 rx))
                  (set! plane_n_y (- 0.0 ry))
                  (set! plane_n_z (- 0.0 rz))
                  (set! mouse_down 1)
                }
                {
                  (set! wt_mode MODE_ROTATE)
                  (set! mouse_down 1)
                })
          })
    }))

(define wt_during_drag
  (lambda (px py)
    {
      (if (== wt_mode MODE_ROTATE)
          {
            (set! angleY (- angleY (* 1.0 (- px prev_px))))
            (set! angleX (- angleX (* 1.0 (- py prev_py))))
            (if (> angleX 89.999) (set! angleX 89.999))
            (if (< angleX -89.999) (set! angleX -89.999))
            (set! view_dirty 1)
          }
          (if (== wt_mode MODE_SPHERE)
              {
                (wt_get_ray_for_pixel px py)
                (define rdx rx) (define rdy ry) (define rdz rz)
                (define denom (+ (+ (* plane_n_x rdx) (* plane_n_y rdy)) (* plane_n_z rdz)))
                (if (> (Math.abs denom) 0.000000001)
                    {
                      (define num (+ (+ (* plane_n_x (- eye_x prev_hit_x))
                                        (* plane_n_y (- eye_y prev_hit_y)))
                                     (* plane_n_z (- eye_z prev_hit_z))))
                      (define tt (/ (- 0.0 num) denom))
                      (define nhx (+ eye_x (* rdx tt)))
                      (define nhy (+ eye_y (* rdy tt)))
                      (define nhz (+ eye_z (* rdz tt)))
                      (set! ball_cx (+ ball_cx (- nhx prev_hit_x)))
                      (set! ball_cy (+ ball_cy (- nhy prev_hit_y)))
                      (set! ball_cz (+ ball_cz (- nhz prev_hit_z)))
                      (set! ball_cx (clampf ball_cx (- (- 0.0 ball_radius) 1.0) (- 1.0 ball_radius)))
                      (set! ball_cy (clampf ball_cy (- (- 0.0 ball_radius) 1.0) 10.0))
                      (set! ball_cz (clampf ball_cz (- (- 0.0 ball_radius) 1.0) (- 1.0 ball_radius)))
                      (set! prev_hit_x nhx) (set! prev_hit_y nhy) (set! prev_hit_z nhz)
                      (set! view_dirty 1)
                    })
              }))
      (set! prev_px px) (set! prev_py py)
    }))

;; ============================================================================
;; 初始化与每一拍
;; ============================================================================

(define mklist
  (lambda (n v)
    {
      (define l (cons 0 (quote ())))
      (pop l)
      (define i 0)
      (while (< i n) { (push l v) (set! i (+ i 1)) })
      l
    }))

(define wt_alloc
  (lambda ()
    {
      (set! hA (mklist GN2 0.0))
      (set! hB (mklist GN2 0.0))
      (set! vA (mklist GN2 0.0))
      (set! vB (mklist GN2 0.0))
      (set! nxF (mklist GN2 0.0))
      (set! nzF (mklist GN2 0.0))
      (set! cauF (mklist GN2 1.0))
      (set! matMV (m4make)) (set! matP (m4make)) (set! matMVP (m4make))
      (set! matTmp (m4make)) (set! matInv (m4make))
      (wt_build_meshes)

      ;; 原版默认状态
      (set! angleX -25.0)
      (set! angleY -200.5)
      ;; light_dir = normalize(2,2,-1)
      (set! light_dir_x (/ 2.0 3.0))
      (set! light_dir_y (/ 2.0 3.0))
      (set! light_dir_z (/ (- 0.0 1.0) 3.0))
      (set! ball_cx -0.4) (set! ball_cy -0.75) (set! ball_cz 0.2)
      (set! ball_ox -0.4) (set! ball_oy -0.75) (set! ball_oz 0.2)
      (set! ball_vx 0.0) (set! ball_vy 0.0) (set! ball_vz 0.0)
      (set! ball_gravity_y GRAVITY_Y)
      (set! ball_radius BALL_R)
      (set! use_sphere_physics 1)
      (set! wt_paused 0)
      (set! wt_mode MODE_NONE)
      (set! mouse_down 0)
      (set! pressed_btn -1)
      (set! view_dirty 0)
      (set! sim_cur 0)
    }))

(define draw_hud
  (lambda ()
    {
      (wt_draw_ui)
      (Gfx.set_font 1)
      (Gfx.draw_text "水池" 4 2 208 232 255)
      (Gfx.draw_text "拖动旋转视角·拖动圆球" 56 2 190 205 225)
      (Gfx.refresh)
    }))

(define tick
  (lambda ()
    {
      ;; ① 退出出口（硬 A/ESC 与底部「返回」按钮都由 EX 负责）
      (EX.check)

      ;; ② 触屏：按下沿定去向，移动沿执行，松手沿结算控件
      (define pressed (if (Touch.is_pressed) 1 0))
      (define tx (Touch.x))
      (define ty (Touch.y))
      (if (== pressed 1)
          {
            (if (== touch_prev 0)
                {
                  (set! down_x tx) (set! down_y ty)
                  (wt_start_drag tx ty)
                  (wt_during_drag tx ty)
                }
                (if (== mouse_down 1) (wt_during_drag tx ty)))
          }
          (if (== touch_prev 1)
              {
                (if (== mouse_down 1)
                    {
                      (define bdn (wt_button_hit down_x down_y))
                      (set! pressed_btn -1)
                      (if (>= bdn 0)
                          (if (== (wt_button_hit tx ty) bdn)
                              {
                                (if (== bdn BTN_BACK)
                                    (EX.do_exit)
                                    (wt_button_action bdn))
                                (set! view_dirty 1)
                              }))
                    })
                (set! mouse_down 0)
                (set! wt_mode MODE_NONE)
                (set! pressed_btn -1)
              }))
      (set! touch_prev pressed)

      ;; ③ 物理 + 仿真 + 渲染（暂停时冻结水面，但拖视角/圆球仍重画）
      (if (== wt_paused 0)
          {
            (wt_update (/ TICK_MS 1000.0))
            (wt_compose_camera)
            (wt_scene_render)
            (set! view_dirty 0)
          }
          (if (== view_dirty 1)
              {
                (wt_compose_camera)
                (wt_scene_render)
                (set! view_dirty 0)
              }))
      (draw_hud)
    }))

;; ---- 启动 ------------------------------------------------------------------
(EX.install)
(wt_alloc)
(wt_update 0.001)
(wt_compose_camera)
(wt_scene_render)
(draw_hud)
(set! tick_timer (System.set_interval TICK_MS tick))
