;; ============================================================================
;; 水池（Raster 后端版）—— 灵机应用程序：**薄 C 后端 + 重 Scheme 前端**
;;
;; 与同目录另两个水池实现的关系（三者画面一致，分工不同）：
;;   · app/waterpool.scm        —— ui_water.c 的逐函数直译，全部在解释器里跑（~12s/帧，保真基准）
;;   · app/waterpool_native.scm —— (native Water) 整机封装：C 持渲染器+相机+物理+输入+HUD，
;;                                 Scheme 只剩 3 行骨架（60fps，但封装很厚）
;;   · app/waterpool_raster.scm —— **本文件**：(native Raster) 只提供逐像素/逐格点/逐三角形
;;                                 的内层循环（"固定功能 GPU"），渲染器前端全在 Scheme。
;;
;; 边界为什么画在这里（实测归因；A/B/C 三个变体把"着色/光栅化/逐顶点 setup"拆开）：
;;
;;   pass      全量      只光栅化   只 setup   说明
;;   池壁      3060 ms   383        0.5        setup 可以留在 Scheme，着色不行
;;   水面      5005 ms   980        816        **只 setup 就要 816ms** —— 10fps 预算(100ms)的 8 倍
;;   圆球       163 ms   111        86
;;   焦散      2000 ms   —          大头（3872 三角形 × 3 顶点的法线/折射/投影）
;;   波场       307 ms   —          —         16384 格点 × 4 个内核
;;   blit       135 ms   —          —         76800 像素上屏
;;   相机/HUD     1 ms   —          —         ← 完全可以留在解释器
;;
;;   ⇒ 逐三角形循环必须进 C（否则光逐顶点阶段就十几倍超预算）；
;;     而相机、几何、剔除策略、绘制顺序、光照、物理、交互、HUD 可以且**应该**留在 Scheme。
;;
;; Scheme 侧（本文件，约 700 行）负责：
;;   · 建网格并上传：水面平面（2025 顶点 / 3872 三角形）、池壁（含池沿 (1−c)·7/12−1 的
;;     世界坐标映射）、UV 球（740 / 1368，单位球，由后端按 center/radius 缩放）；
;;   · 相机：MV = T(0,0,−4)·Rx(−angleX)·Ry(−angleY)·T(0,0.5,0)、P = perspective(45°,·)、
;;     MVP = P·MV；由 MV 反解 eye；由 MVP⁻¹ 求四角射线（圆球拾取用）；
;;   · 绘制顺序与剔除策略：池壁(只留面积<0) → 水面水上(只留≥0) → 水面水下(只留<0) → 圆球(不剔除)；
;;   · 圆球物理（重力/水中黏滞浮力/池底反弹）、光源跟随视角、触屏交互（旋转/拖球/控件）、
;;     控件栏与标题、上屏与推屏。
;;
;; 后端（infer/am_native_Raster.c）负责：
;;   (Raster.begin) / (Raster.draw mesh kind mvp params) / (Raster.sim_move|sim_step|sim_normals)
;;   / (Raster.caustics mesh light ball) / (Raster.blit) / (Raster.init|mesh|dispose)
;;   —— 每帧 3 次 draw + 4 次仿真/焦散 + begin/blit，共 9 次本地调用；它不知道相机角度、
;;   圆球物理、暂停/重力/光源开关、控件栏和文字。
;;
;; 解释器约束（manual.md）：算术/比较恒二参数；无一元取负；同一 lambda 内不重复 define；
;;   块内 define 提升到 lambda 体首部；没有 let/list/map/apply/min/max。
;; ============================================================================

(native Gfx)
(native Math)
(native System)
(native Touch)
(native Raster)                   ; 3D 软光栅后端（见 am_native_Raster.h）
(import EX "app/lib/exit.scm")    ; 硬 A/ESC / 控件「返回」→ 宿主全局退出确认框

;; ============================================================================
;; 常量（数值与 ui_water.c 一致）
;; ============================================================================

(define POOL_HEIGHT 1.0)          ; WT_POOL_HEIGHT
(define BALL_R 0.25)              ; WT_RADIUS
(define GRAVITY_Y -4.0)           ; WT_GRAVITY

(define WATER_DETAIL 44)          ; WT_WATER_DETAIL（水面细分）
(define WATER_P1 45)
(define SPH_RING 20)              ; WT_SPH_RING
(define SPH_SEG 36)               ; WT_SPH_SEG
(define CUBE_FACES 5)

;; 网格编号（Scheme 决定谁是谁，后端只按 id 取网格）
(define MESH_WATER 0)
(define MESH_CUBE 1)
(define MESH_SPHERE 2)

;; 片元着色器种类（= ui_water.c 的三个 pass，水面分上下两遍）
(define KIND_WALL 0)
(define KIND_WATER_ABOVE 1)
(define KIND_WATER_BELOW 2)
(define KIND_SPHERE 3)

;; 剔除模式（原版 renderCube / renderWater 的 CULL_FACE 策略，这里由 Scheme 逐次指定）
(define CULL_NONE 0)
(define CULL_NEG 1)
(define CULL_POS 2)

;; 画布与节拍（8ms：宿主只在 GUI 帧边界检查定时器，16ms 会被量化成隔帧；见 doc/AGENTS.md）
(define SW (Gfx.width))
(define SH (Gfx.height))
(define TICK_MS 8)

;; 底部控件栏（几何同 ui_water.c 的 wt_button_rect）
(define BAR_H 34)
(define BAR_Y (- SH BAR_H))
(define BTN_N 4)
(define BTN_GAP 4)
(define BTN_W (Math.floor (/ (- SW (* BTN_GAP (+ BTN_N 1))) BTN_N)))
(define BTN_H (- BAR_H BTN_GAP))
(define BTN_Y (+ BAR_Y (/ BTN_GAP 2)))

(define BTN_BACK 0)
(define BTN_PAUSE 1)
(define BTN_GRAVITY 2)
(define BTN_LIGHT 3)

(define MODE_NONE 0)
(define MODE_ROTATE 1)
(define MODE_SPHERE 2)
(define MODE_BUTTON 3)

;; ============================================================================
;; 状态（原版 WT_State 里属于"前端"的那些：相机、圆球、控件、交互）
;; ============================================================================

(define angleX -25.0)
(define angleY -200.5)
(define eye_x 0.0) (define eye_y 0.0) (define eye_z 0.0)
(define ray00x 0.0) (define ray00y 0.0) (define ray00z 0.0)
(define ray10x 0.0) (define ray10y 0.0) (define ray10z 0.0)
(define ray01x 0.0) (define ray01y 0.0) (define ray01z 0.0)
(define ray11x 0.0) (define ray11y 0.0) (define ray11z 0.0)

(define light_dir_x 2.0) (define light_dir_y 2.0) (define light_dir_z -1.0)
(define ball_cx -0.4) (define ball_cy -0.75) (define ball_cz 0.2)
(define ball_ox -0.4) (define ball_oy -0.75) (define ball_oz 0.2)
(define ball_vx 0.0) (define ball_vy 0.0) (define ball_vz 0.0)
(define ball_gravity_y GRAVITY_Y)
(define ball_radius BALL_R)
(define use_sphere_physics 1)
(define wt_paused 0)
(define wt_mode MODE_NONE)
(define prev_hit_x 0.0) (define prev_hit_y 0.0) (define prev_hit_z 0.0)
(define plane_n_x 0.0) (define plane_n_y 0.0) (define plane_n_z 0.0)
(define mouse_down 0)
(define prev_px 0) (define prev_py 0)
(define pressed_btn -1)
(define view_dirty 0)
(define ready 0)

(define tick_timer 0)
(define last_t 0)
(define now_t 0)
(define dt_ms 0)
(define dt_s 0.0)
(define touch_prev 0)
(define down_x 0) (define down_y 0)
(define back 0)

;; 矩阵（16 元列表，行主序）
(define matMV (cons 0 (quote ())))
(define matP (cons 0 (quote ())))
(define matMVP (cons 0 (quote ())))
(define matTmp (cons 0 (quote ())))
(define matInv (cons 0 (quote ())))

;; 网格（上传给后端）与每帧 uniform 列表
(define mesh_v (cons 0 (quote ())))
(define mesh_t (cons 0 (quote ())))
(define p_cull_neg (cons 0 (quote ())))     ; 池壁 / 水面(水下)：只留面积<0
(define p_cull_pos (cons 0 (quote ())))     ; 水面(水上)：只留面积>=0
(define p_cull_none (cons 0 (quote ())))    ; 圆球：不剔除

;; ============================================================================
;; 小工具（与 C 的 flr/clr/fmin/fmax/clamp 等价；floor/ceil 用 mod 自算省指令）
;; ============================================================================

(define flr
  (lambda (x)
    {
      (define fr (mod x 1.0))
      (if (< fr 0.0) (- (- x fr) 1.0) (- x fr))
    }))

(define clr
  (lambda (x)
    {
      (define fr (mod x 1.0))
      (if (> fr 0.0) (+ (- x fr) 1.0) (- x fr))
    }))

(define fmin2
  (lambda (a b) (if (< a b) a b)))

(define fmax2
  (lambda (a b) (if (> a b) a b)))

(define clampf
  (lambda (v lo hi) (if (< v lo) lo (if (> v hi) hi v))))
;; 整数幂替身（Math.pow 是 native 调用；x^3 / x^4 / x^6 用连乘）

;; --- 向量寄存器（结果放 rx/ry/rz；操作数放 ax/ay/az、bx/by/bz）---
(define rx 0.0) (define ry 0.0) (define rz 0.0)
(define ax 0.0) (define ay 0.0) (define az 0.0)
(define bx 0.0) (define by 0.0) (define bz 0.0)
(define rlen 0.0)
(define rdot 0.0)

;; wt_mvp4 的四个结果（矩阵乘向量）
(define mx 0.0) (define my 0.0) (define mz 0.0) (define mw 0.0)

(define v_add (lambda () { (set! rx (+ ax bx)) (set! ry (+ ay by)) (set! rz (+ az bz)) }))

(define v_sub (lambda () { (set! rx (- ax bx)) (set! ry (- ay by)) (set! rz (- az bz)) }))

(define v_scale (lambda (k) { (set! rx (* ax k)) (set! ry (* ay k)) (set! rz (* az k)) }))

(define v_neg (lambda () { (set! rx (- 0.0 ax)) (set! ry (- 0.0 ay)) (set! rz (- 0.0 az)) }))

(define v_dot (lambda () { (set! rdot (+ (+ (* ax bx) (* ay by)) (* az bz))) }))

(define v_cross (lambda ()
  {
    (set! rx (- (* ay bz) (* az by)))
    (set! ry (- (* az bx) (* ax bz)))
    (set! rz (- (* ax by) (* ay bx)))
  }))

(define v_len (lambda () { (set! rlen (Math.sqrt (+ (+ (* ax ax) (* ay ay)) (* az az)))) }))

(define v_norm (lambda ()
  {
    (v_len)
    (set! rx (/ ax rlen))
    (set! ry (/ ay rlen))
    (set! rz (/ az rlen))
  }))
;; r = a + (b-a)*t

(define v_lerp (lambda (t)
  {
    (set! rx (+ ax (* (- bx ax) t)))
    (set! ry (+ ay (* (- by ay) t)))
    (set! rz (+ az (* (- bz az) t)))
  }))

(define v_copy_a (lambda () { (set! ax rx) (set! ay ry) (set! az rz) }))

(define v_set_a (lambda (x y z) { (set! ax x) (set! ay y) (set! az z) }))

(define v_set_b (lambda (x y z) { (set! bx x) (set! by y) (set! bz z) }))

;; --- 4x4 矩阵：行主序，16 元素列表 ---

;; --- 4x4 矩阵（行主序，16 元素列表）---
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
;; o = A·B

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
    (set_item! o 11 (/ (* (* -2.0 f) n) (- f n)))
    (set_item! o 14 -1.0)
  }))

(define m4rot
  (lambda (o deg axx ayy azz) {
    (define dd (Math.sqrt (+ (+ (* axx axx) (* ayy ayy)) (* azz azz))))
    (define aa (* deg (/ (Math.PI) 180.0)))
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
;; MVP·(x,y,z,1) → 结果放 mx/my/mz/mw（对应 wt_mvp4）

;; 伴随式求逆（与 ui_water.c 的 m4inv 逐项一致，左结合二元链展开）
;;   ⚠ 这一版是**重新推导的正确公式**。app/waterpool.scm 里的同名函数把 r1/r2/r3/r5… 的
;;   交替符号整体写反了，逆矩阵是错的；它只被"四角射线"用到，所以画面一切正常，但
;;   圆球拾取（wt_hit_test_sphere → wt_get_ray_for_pixel）永远失败 —— 表现为
;;   **拖圆球变成拖视角**。修好后四角射线与 C 参照台逐位一致：
;;   r00=(0.214963,-0.656717,0.722851)、r11=(-0.737358,-0.038853,0.674384)。
(define m4inv
  (lambda (s o) {
    (define m0 (get_item s 0)) (define m1 (get_item s 1)) (define m2 (get_item s 2)) (define m3 (get_item s 3))
    (define m4 (get_item s 4)) (define m5 (get_item s 5)) (define m6 (get_item s 6)) (define m7 (get_item s 7))
    (define m8 (get_item s 8)) (define m9 (get_item s 9)) (define m10 (get_item s 10)) (define m11 (get_item s 11))
    (define m12 (get_item s 12)) (define m13 (get_item s 13)) (define m14 (get_item s 14)) (define m15 (get_item s 15))
    (define rr0 (- (+ (+ (- (- (* (* m5 m10) m15) (* (* m5 m11) m14)) (* (* m9 m6) m15)) (* (* m9 m7) m14)) (* (* m13 m6) m11)) (* (* m13 m7) m10)))
    (define rr1 (+ (- (- (+ (+ (- 0.0 (* (* m1 m10) m15)) (* (* m1 m11) m14)) (* (* m9 m2) m15)) (* (* m9 m3) m14)) (* (* m13 m2) m11)) (* (* m13 m3) m10)))
    (define rr2 (- (+ (+ (- (- (* (* m1 m6) m15) (* (* m1 m7) m14)) (* (* m5 m2) m15)) (* (* m5 m3) m14)) (* (* m13 m2) m7)) (* (* m13 m3) m6)))
    (define rr3 (+ (- (- (+ (+ (- 0.0 (* (* m1 m6) m11)) (* (* m1 m7) m10)) (* (* m5 m2) m11)) (* (* m5 m3) m10)) (* (* m9 m2) m7)) (* (* m9 m3) m6)))
    (define rr4 (+ (- (- (+ (+ (- 0.0 (* (* m4 m10) m15)) (* (* m4 m11) m14)) (* (* m8 m6) m15)) (* (* m8 m7) m14)) (* (* m12 m6) m11)) (* (* m12 m7) m10)))
    (define rr5 (- (+ (+ (- (- (* (* m0 m10) m15) (* (* m0 m11) m14)) (* (* m8 m2) m15)) (* (* m8 m3) m14)) (* (* m12 m2) m11)) (* (* m12 m3) m10)))
    (define rr6 (+ (- (- (+ (+ (- 0.0 (* (* m0 m6) m15)) (* (* m0 m7) m14)) (* (* m4 m2) m15)) (* (* m4 m3) m14)) (* (* m12 m2) m7)) (* (* m12 m3) m6)))
    (define rr7 (- (+ (+ (- (- (* (* m0 m6) m11) (* (* m0 m7) m10)) (* (* m4 m2) m11)) (* (* m4 m3) m10)) (* (* m8 m2) m7)) (* (* m8 m3) m6)))
    (define rr8 (- (+ (+ (- (- (* (* m4 m9) m15) (* (* m4 m11) m13)) (* (* m8 m5) m15)) (* (* m8 m7) m13)) (* (* m12 m5) m11)) (* (* m12 m7) m9)))
    (define rr9 (+ (- (- (+ (+ (- 0.0 (* (* m0 m9) m15)) (* (* m0 m11) m13)) (* (* m8 m1) m15)) (* (* m8 m3) m13)) (* (* m12 m1) m11)) (* (* m12 m3) m9)))
    (define rr10 (- (+ (+ (- (- (* (* m0 m5) m15) (* (* m0 m7) m13)) (* (* m4 m1) m15)) (* (* m4 m3) m13)) (* (* m12 m1) m7)) (* (* m12 m3) m5)))
    (define rr11 (+ (- (- (+ (+ (- 0.0 (* (* m0 m5) m11)) (* (* m0 m7) m9)) (* (* m4 m1) m11)) (* (* m4 m3) m9)) (* (* m8 m1) m7)) (* (* m8 m3) m5)))
    (define rr12 (+ (- (- (+ (+ (- 0.0 (* (* m4 m9) m14)) (* (* m4 m10) m13)) (* (* m8 m5) m14)) (* (* m8 m6) m13)) (* (* m12 m5) m10)) (* (* m12 m6) m9)))
    (define rr13 (- (+ (+ (- (- (* (* m0 m9) m14) (* (* m0 m10) m13)) (* (* m8 m1) m14)) (* (* m8 m2) m13)) (* (* m12 m1) m10)) (* (* m12 m2) m9)))
    (define rr14 (+ (- (- (+ (+ (- 0.0 (* (* m0 m5) m14)) (* (* m0 m6) m13)) (* (* m4 m1) m14)) (* (* m4 m2) m13)) (* (* m12 m1) m6)) (* (* m12 m2) m5)))
    (define rr15 (- (+ (+ (- (- (* (* m0 m5) m10) (* (* m0 m6) m9)) (* (* m4 m1) m10)) (* (* m4 m2) m9)) (* (* m8 m1) m6)) (* (* m8 m2) m5)))
    (define det (+ (+ (+ (* m0 rr0) (* m1 rr4)) (* m2 rr8)) (* m3 rr12)))
    (if (== det 0.0)
        (m4id o)
        {
          (set_item! o 0 (/ rr0 det)) (set_item! o 1 (/ rr1 det))
          (set_item! o 2 (/ rr2 det)) (set_item! o 3 (/ rr3 det))
          (set_item! o 4 (/ rr4 det)) (set_item! o 5 (/ rr5 det))
          (set_item! o 6 (/ rr6 det)) (set_item! o 7 (/ rr7 det))
          (set_item! o 8 (/ rr8 det)) (set_item! o 9 (/ rr9 det))
          (set_item! o 10 (/ rr10 det)) (set_item! o 11 (/ rr11 det))
          (set_item! o 12 (/ rr12 det)) (set_item! o 13 (/ rr13 det))
          (set_item! o 14 (/ rr14 det)) (set_item! o 15 (/ rr15 det))
        })
  }))

(define wt_mvp4
  (lambda (m x y z) {
    (set! mx (+ (+ (+ (* (get_item m 0) x) (* (get_item m 1) y)) (* (get_item m 2) z)) (get_item m 3)))
    (set! my (+ (+ (+ (* (get_item m 4) x) (* (get_item m 5) y)) (* (get_item m 6) z)) (get_item m 7)))
    (set! mz (+ (+ (+ (* (get_item m 8) x) (* (get_item m 9) y)) (* (get_item m 10) z)) (get_item m 11)))
    (set! mw (+ (+ (+ (* (get_item m 12) x) (* (get_item m 13) y)) (* (get_item m 14) z)) (get_item m 15)))
  }))
;; 伴随式求逆（与 C 的 m4inv 逐项一致）

(define wt_compose_camera
  (lambda ()
    {
      (m4id matMV)
      (m4tr matTmp 0.0 0.0 -4.0)        (m4mul matMV matMV matTmp)
      (m4rot matTmp (- 0.0 angleX) 1.0 0.0 0.0) (m4mul matMV matMV matTmp)
      (m4rot matTmp (- 0.0 angleY) 0.0 1.0 0.0) (m4mul matMV matMV matTmp)
      (m4tr matTmp 0.0 0.5 0.0)         (m4mul matMV matMV matTmp)
      (m4persp matP 45.0 (/ (* 1.0 SW) (* 1.0 SH)) 0.01 100.0)
      (m4mul matMVP matP matMV)

      ;; 由 modelview 反解世界系相机位置：eye = -(axisX·off, axisY·off, axisZ·off)
      (define axX (+ (* (get_item matMV 0) (get_item matMV 0))
                     (+ (* (get_item matMV 4) (get_item matMV 4)) (* (get_item matMV 8) (get_item matMV 8)))))
      (set! eye_x (- 0.0 (+ (+ (* (get_item matMV 3) (get_item matMV 0))
                               (* (get_item matMV 7) (get_item matMV 4)))
                            (* (get_item matMV 11) (get_item matMV 8)))))
      (set! eye_y (- 0.0 (+ (+ (* (get_item matMV 3) (get_item matMV 1))
                               (* (get_item matMV 7) (get_item matMV 5)))
                            (* (get_item matMV 11) (get_item matMV 9)))))
      (set! eye_z (- 0.0 (+ (+ (* (get_item matMV 3) (get_item matMV 2))
                               (* (get_item matMV 7) (get_item matMV 6)))
                            (* (get_item matMV 11) (get_item matMV 10)))))
      (if (== axX 0.0) #f #f)

      ;; 四角射线 = unProject(角, 1) - eye
      (m4inv matMVP matInv)
      (define cx 0)
      (while (< cx 2)
        {
          (define cy 0)
          (while (< cy 2)
            {
              (define nx (- (/ (* 2.0 (* cx SW)) (* 1.0 SW)) 1.0))
              (define ny (- (/ (* 2.0 (* cy SH)) (* 1.0 SH)) 1.0))
              (wt_mvp4 matInv nx ny 1.0)
              (define iw 0.0)
              (define wx 0.0) (define wy 0.0) (define wz 0.0)
              (if (<= mw 0.0)
                  { (set! wx 0.0) (set! wy 0.0) (set! wz 0.0) }
                  {
                    (set! iw (/ 1.0 mw))
                    (set! wx (* mx iw)) (set! wy (* my iw)) (set! wz (* mz iw))
                    (set! wx (- wx eye_x)) (set! wy (- wy eye_y)) (set! wz (- wz eye_z))
                    (v_set_a wx wy wz)
                    (v_norm)
                    (set! wx rx) (set! wy ry) (set! wz rz)
                  })
              (if (== cx 0)
                  (if (== cy 0)
                      { (set! ray00x wx) (set! ray00y wy) (set! ray00z wz) }
                      { (set! ray01x wx) (set! ray01y wy) (set! ray01z wz) })
                  (if (== cy 0)
                      { (set! ray10x wx) (set! ray10y wy) (set! ray10z wz) }
                      { (set! ray11x wx) (set! ray11y wy) (set! ray11z wz) }))
              (set! cy (+ cy 1))
            })
          (set! cx (+ cx 1))
        })
    }))

;; getRayForPixel(x,y) → 单位方向放 rx/ry/rz

(define mklist
  (lambda (n v)
    {
      (define l (cons 0 (quote ())))
      (pop l)
      (define i 0)
      (while (< i n) { (push l v) (set! i (+ i 1)) })
      l
    }))


;; ============================================================================
;; 相机（wt_compose_camera：MV/P/MVP、eye、四角射线）—— 全部在 Scheme 里算
;; ============================================================================


(define wt_get_ray_for_pixel
  (lambda (px py)
    {
      (define fx (/ (* 1.0 px) (* 1.0 SW)))
      (define fy0 (/ (* 1.0 py) (* 1.0 SH)))
      (define fy (- 1.0 fy0))
      ;; r0 = lerp(ray00, ray10, fx)；r1 = lerp(ray01, ray11, fx)；dir = lerp(r0, r1, fy)
      (define r0x (+ ray00x (* (- ray10x ray00x) fx)))
      (define r0y (+ ray00y (* (- ray10y ray00y) fx)))
      (define r0z (+ ray00z (* (- ray10z ray00z) fx)))
      (define r1x (+ ray01x (* (- ray11x ray01x) fx)))
      (define r1y (+ ray01y (* (- ray11y ray01y) fx)))
      (define r1z (+ ray01z (* (- ray11z ray01z) fx)))
      (v_set_a (+ r0x (* (- r1x r0x) fy)) (+ r0y (* (- r1y r0y) fy)) (+ r0z (* (- r1z r0z) fy)))
      (v_norm)
    }))

;; ============================================================================
;; 网格：Scheme 建几何，一次性上传给后端（(Raster.mesh id verts tris)）
;;   verts：每 5 个数一个顶点 (x, y, z, u, v)
;;     · 池壁/圆球 — (x,y,z) 是三维坐标（池壁为世界坐标；圆球为单位球）
;;     · 水面     — 沿用原版 WT_GeoV 布局：(x,y) 是平面 (x,z)，z 未用，y 由波场高度替代
;; ============================================================================

;; 水面平面（同 Mesh.plane(detail)）：顶点 (2s-1, 2t-1) 的平面坐标 + 纹理坐标 (s,t)
(define build_water
  (lambda ()
    {
      (set! mesh_v (cons 0 (quote ()))) (pop mesh_v)
      (set! mesh_t (cons 0 (quote ()))) (pop mesh_t)
      (define gy 0)
      (while (<= gy WATER_DETAIL)
        {
          (define gx 0)
          (define t (/ (* 1.0 gy) (* 1.0 WATER_DETAIL)))
          (while (<= gx WATER_DETAIL)
            {
              (define s (/ (* 1.0 gx) (* 1.0 WATER_DETAIL)))
              (push mesh_v (- (* 2.0 s) 1.0))    ; 平面 x
              (push mesh_v (- (* 2.0 t) 1.0))    ; 平面 z（原版 WT_GeoV.vy）
              (push mesh_v 0.0)                  ; 未用
              (push mesh_v s)                    ; u
              (push mesh_v t)                    ; v
              (set! gx (+ gx 1))
            })
          (set! gy (+ gy 1))
        })
      (define j 0)
      (while (< j WATER_DETAIL)
        {
          (define i 0)
          (while (< i WATER_DETAIL)
            {
              (define v0 (+ (* j WATER_P1) i))
              (push mesh_t v0) (push mesh_t (+ v0 1)) (push mesh_t (+ v0 WATER_P1))
              (push mesh_t (+ v0 WATER_P1)) (push mesh_t (+ v0 1)) (push mesh_t (+ v0 (+ WATER_P1 1)))
              (set! i (+ i 1))
            })
          (set! j (+ j 1))
        })
      (Raster.mesh MESH_WATER mesh_v mesh_t)
    }))

;; 池壁（cubeData 去掉 -y 顶沿面；+y 一组为底面，重映射到 y=-1）
;;   世界 y = ((1 - c) · 7/12 - 1) · POOL_HEIGHT —— 这条"池沿在 +1/6"的映射是几何工作，
;;   在 Scheme 里按原版公式算好后上传（后端只认世界坐标）
(define build_cube
  (lambda ()
    {
      (set! mesh_v (cons 0 (quote ()))) (pop mesh_v)
      (set! mesh_t (cons 0 (quote ()))) (pop mesh_t)
      (define f 0)
      (while (< f CUBE_FACES)
        {
          (define q 0)
          (while (< q 4)
            {
              (define d 0)
              ;; faces[f][q]：-x{0,4,2,6} +x{1,3,5,7} -y底面{2,6,3,7} -z{0,2,1,3} +z{4,5,6,7}
              (if (== f 0) (set! d (get_item (quote (0 4 2 6)) q)) #f)
              (if (== f 1) (set! d (get_item (quote (1 3 5 7)) q)) #f)
              (if (== f 2) (set! d (get_item (quote (2 6 3 7)) q)) #f)
              (if (== f 3) (set! d (get_item (quote (0 2 1 3)) q)) #f)
              (if (== f 4) (set! d (get_item (quote (4 5 6 7)) q)) #f)
              ;; d 的三位分别是 x/y/z 的符号（原版 (d&1)*2-1、(d&2)-1、(d&4)/2-1），
              ;; 这里用整数取位而不是 mod：mod 是取余不是按位与（d=5 时 (mod 5 2)=1）
              (define b0 (mod d 2.0))
              (define b1 (mod (flr (/ (* 1.0 d) 2.0)) 2.0))
              (define b2 (mod (flr (/ (* 1.0 d) 4.0)) 2.0))
              (define cx (- (* b0 2.0) 1.0))
              (define cy (- (* b1 2.0) 1.0))
              (define cz (- (* b2 2.0) 1.0))
              (push mesh_v cx)
              (push mesh_v (* (- (* (- 1.0 cy) (/ 7.0 12.0)) 1.0) POOL_HEIGHT))
              (push mesh_v cz)
              (push mesh_v 0.0) (push mesh_v 0.0)
              (set! q (+ q 1))
            })
          (define b (* f 4))
          (push mesh_t b) (push mesh_t (+ b 1)) (push mesh_t (+ b 2))
          (push mesh_t (+ b 2)) (push mesh_t (+ b 1)) (push mesh_t (+ b 3))
          (set! f (+ f 1))
        })
      (Raster.mesh MESH_CUBE mesh_v mesh_t)
    }))

;; 圆球（UV 球，单位球；后端按 center/radius 缩放）
(define build_sphere
  (lambda ()
    {
      (set! mesh_v (cons 0 (quote ()))) (pop mesh_v)
      (set! mesh_t (cons 0 (quote ()))) (pop mesh_t)
      (define r 0)
      (while (< r SPH_RING)
        {
          (define phi (* (Math.PI) (/ (* 1.0 r) (* 1.0 (- SPH_RING 1)))))
          (define yy (Math.cos phi))
          (define rad (Math.sin phi))
          (define s 0)
          (while (<= s SPH_SEG)
            {
              (define th (* (* 2.0 (Math.PI)) (/ (* 1.0 s) (* 1.0 SPH_SEG))))
              (push mesh_v (* rad (Math.cos th)))
              (push mesh_v yy)
              (push mesh_v (* rad (Math.sin th)))
              (push mesh_v 0.0) (push mesh_v 0.0)
              (set! s (+ s 1))
            })
          (set! r (+ r 1))
        })
      (define rr 0)
      (while (< rr (- SPH_RING 1))
        {
          (define ss 0)
          (while (< ss SPH_SEG)
            {
              (define a (+ (* rr (+ SPH_SEG 1)) ss))
              (define b (+ a 1))
              (define c (+ a (+ SPH_SEG 1)))
              (define dd (+ c 1))
              (push mesh_t a) (push mesh_t c) (push mesh_t b)
              (push mesh_t b) (push mesh_t c) (push mesh_t dd)
              (set! ss (+ ss 1))
            })
          (set! rr (+ rr 1))
        })
      (Raster.mesh MESH_SPHERE mesh_v mesh_t)
    }))

;; ============================================================================
;; 每帧 uniform：params = (eye.xyz light.xyz ball.xyz radius cull)
;;   —— 预分配三个列表（剔除模式固定），每帧只 set_item! 覆写，零分配
;; ============================================================================

(define mkparams
  (lambda (cull)
    {
      (define l (mklist 11 0.0))
      (set_item! l 10 cull)
      l
    }))

(define set_params
  (lambda (l)
    {
      (set_item! l 0 eye_x) (set_item! l 1 eye_y) (set_item! l 2 eye_z)
      (set_item! l 3 light_dir_x) (set_item! l 4 light_dir_y) (set_item! l 5 light_dir_z)
      (set_item! l 6 ball_cx) (set_item! l 7 ball_cy) (set_item! l 8 ball_cz)
      (set_item! l 9 ball_radius)
    }))

;; ============================================================================
;; 圆球物理 + 后端内核（wt_update）
;; ============================================================================
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
                ;; velocity += gravity*(seconds - 1.1*seconds*pUW)
                (set! ball_vy (+ ball_vy (* (* ball_gravity_y seconds) (- 1.0 (* 1.1 pUW)))))
                ;; 水中黏滞：velocity -= unit(velocity) * (pUW*seconds*|v|^2)
                (set! sdot (+ (+ (* ball_vx ball_vx) (* ball_vy ball_vy)) (* ball_vz ball_vz)))
                (set! vl (Math.sqrt sdot))
                (if (> vl 0.000001)
                    {
                      (define kk (* (* pUW seconds) sdot))
                      (set! ball_vx (- ball_vx (* (/ ball_vx vl) kk)))
                      (set! ball_vy (- ball_vy (* (/ ball_vy vl) kk)))
                      (set! ball_vz (- ball_vz (* (/ ball_vz vl) kk)))
                    }
                    #f)
                (set! ball_cx (+ ball_cx (* ball_vx seconds)))
                (set! ball_cy (+ ball_cy (* ball_vy seconds)))
                (set! ball_cz (+ ball_cz (* ball_vz seconds)))
                (if (< ball_cy (- ball_radius 1.0))
                    {
                      (set! ball_cy (- ball_radius 1.0))
                      (set! ball_vy (* (Math.abs ball_vy) 0.7))
                    }
                    #f)
              }
              #f))
      ;; 后端内核：圆球排开/回填水体 → 波场两次推进 → 法线 → 焦散
      ;; （顺序同 main.js update；逐格点/逐像素的部分在本地库里跑）
      (Raster.sim_move ball_ox ball_oy ball_oz ball_cx ball_cy ball_cz ball_radius)
      (set! ball_ox ball_cx) (set! ball_oy ball_cy) (set! ball_oz ball_cz)
      (Raster.sim_step)
      (Raster.sim_step)
      (Raster.sim_normals)
      (Raster.caustics MESH_WATER light_dir_x light_dir_y light_dir_z
                       ball_cx ball_cy ball_cz ball_radius)
    }))

;; ============================================================================
;; 场景绘制：三次 draw（池壁 / 水面×2 / 圆球）+ 上屏
;;   —— 绘制顺序、剔除策略、深度共用都在 Scheme 里决定；后端只按调用光栅化
;; ============================================================================
(define wt_scene_render
  (lambda ()
    {
      (set_params p_cull_neg)
      (set_params p_cull_pos)
      (set_params p_cull_none)
      (Raster.begin)
      (Raster.draw MESH_CUBE KIND_WALL matMVP p_cull_neg)          ; 远壁/底面，剔除近壁
      (Raster.draw MESH_WATER KIND_WATER_ABOVE matMVP p_cull_pos)  ; 水面 pass0：只画背面
      (Raster.draw MESH_WATER KIND_WATER_BELOW matMVP p_cull_neg)  ; 水面 pass1：只画正面
      (Raster.draw MESH_SPHERE KIND_SPHERE matMVP p_cull_none)     ; 圆球
      (Raster.blit)
    }))

;; ============================================================================
;; 控件栏（wt_button_rect / wt_button_hit / wt_button_action / wt_draw_ui）
;; ============================================================================

(define btx 0) (define bty 0) (define btw 0) (define bth 0)
(define wt_button_rect
  (lambda (idx)
    {
      (define n (if (> BTN_N 0) BTN_N 1))
      (define bw (Math.floor (/ (- SW (* BTN_GAP (+ n 1))) n)))
      (if (< bw 1) (set! bw 1) #f)
      (define bh (- BAR_H BTN_GAP))
      (if (< bh 1) (set! bh 1) #f)
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
                            (if (< py (+ bty bth)) (set! hitb i) #f)
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
      (if (== idx BTN_PAUSE) (set! wt_paused (- 1 wt_paused)) #f)
      (if (== idx BTN_GRAVITY) (set! use_sphere_physics (- 1 use_sphere_physics)) #f)
      (if (== idx BTN_LIGHT) (wt_light_follow_view) #f)
    }))

;; 绘制控件栏（先于推屏，直接叠在渲染结果之上）——与 C 的 wt_draw_ui 逐句一致

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
          (if (== i BTN_BACK) (set! label "返回") #f)
          (if (== i BTN_PAUSE)
              {
                (if (== wt_paused 1) (set! label "继续") (set! label "暂停"))
                (set! active wt_paused)
              }
              #f)
          (if (== i BTN_GRAVITY)
              {
                (if (== use_sphere_physics 1) (set! label "重力开") (set! label "重力关"))
                (set! active use_sphere_physics)
              }
              #f)
          (if (== i BTN_LIGHT) (set! label "光源") #f)
          (define pressed (if (== pressed_btn i) 1 0))
          (define cr 34) (define cg 46) (define cb 62)
          (if (== active 1) { (set! cr 28) (set! cg 104) (set! cb 92) } #f)
          (if (== pressed 1) { (set! cr 116) (set! cg 172) (set! cb 226) } #f)
          (Gfx.rounded_rect btx bty btw bth 7 7 7 7 cr cg cb (if (== pressed 1) 240 200))
          (Gfx.draw_text_centered 1 label
                                  (Math.floor (+ btx (Math.floor (/ (* 1.0 btw) 2))))
                                  (Math.floor (+ bty (Math.floor (/ (* 1.0 bth) 2))))
                                  240 246 255 1)
          (set! i (+ i 1))
        })
    }))


;; ============================================================================
;; 交互（wt_hit_test_sphere / wt_start_drag / wt_during_drag）
;; ============================================================================
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
                }
                #f)
          }
          #f)
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
            (if (> angleX 89.999) (set! angleX 89.999) #f)
            (if (< angleX -89.999) (set! angleX -89.999) #f)
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
                      (set! ball_cx (clampf ball_cx (- ball_radius 1.0) (- 1.0 ball_radius)))
                      (set! ball_cy (clampf ball_cy (- ball_radius 1.0) 10.0))
                      (set! ball_cz (clampf ball_cz (- ball_radius 1.0) (- 1.0 ball_radius)))
                      (set! prev_hit_x nhx) (set! prev_hit_y nhy) (set! prev_hit_z nhz)
                      (set! view_dirty 1)
                    }
                    #f)
              }
              #f))
      (set! prev_px px) (set! prev_py py)
    }))


;; ============================================================================
;; 初始化与每一拍
;; ============================================================================

(define wt_alloc
  (lambda ()
    {
      (set! matMV (m4make)) (set! matP (m4make)) (set! matMVP (m4make))
      (set! matTmp (m4make)) (set! matInv (m4make))
      (set! p_cull_neg (mkparams CULL_NEG))
      (set! p_cull_pos (mkparams CULL_POS))
      (set! p_cull_none (mkparams CULL_NONE))

      ;; 后端缓冲（帧/深度/波场/焦散）+ 三份网格上传
      (if (== (Raster.init) 0) (set! ready 0) (set! ready 1))
      (build_water)
      (build_cube)
      (build_sphere)

      ;; 原版默认场景
      (set! angleX -25.0)
      (set! angleY -200.5)
      (set! light_dir_x 2.0) (set! light_dir_y 2.0) (set! light_dir_z -1.0)
      (set! ball_cx -0.4) (set! ball_cy -0.75) (set! ball_cz 0.2)
      (set! ball_ox -0.4) (set! ball_oy -0.75) (set! ball_oz 0.2)
      (set! ball_vx 0.0) (set! ball_vy 0.0) (set! ball_vz 0.0)
      (set! use_sphere_physics 1)
      (set! wt_paused 0)
      (set! wt_mode MODE_NONE)
      (set! mouse_down 0)
      (set! pressed_btn -1)
      (set! view_dirty 0)
    }))

(define tick
  (lambda ()
    {
      ;; ① 退出出口（硬 A/ESC；控件「返回」在 ④ 结算）
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
                (if (== mouse_down 1) (wt_during_drag tx ty) #f))
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
                              }
                              #f)
                          #f)
                    }
                    #f)
                (set! mouse_down 0)
                (set! wt_mode MODE_NONE)
                (set! pressed_btn -1)
              }
              #f))
      (set! touch_prev pressed)

      ;; ③ 时间步（墙钟差分，与 C 版取 global_state->timestamp 同源）
      (set! now_t (System.timestamp))
      (set! dt_ms (- now_t last_t))
      (set! last_t now_t)
      (if (< dt_ms 0) (set! dt_ms 0) #f)
      (if (> dt_ms 1000) (set! dt_ms 1000) #f)
      (set! dt_s (/ (* 1.0 dt_ms) 1000.0))

      ;; ④ 物理 + 仿真（后端内核）+ 相机 + 绘制（暂停时冻结，但拖动仍重画）
      (if (== wt_paused 0)
          {
            (wt_update dt_s)
            (wt_compose_camera)
            (wt_scene_render)
            (set! view_dirty 0)
          }
          (if (== view_dirty 1)
              {
                (wt_compose_camera)
                (wt_scene_render)
                (set! view_dirty 0)
              }
              #f))

      ;; ⑤ 控件栏 + 标题 + 推屏
      (wt_draw_ui)
      (Gfx.set_font 1)
      (Gfx.draw_text "水池" 4 2 208 232 255)
      (Gfx.draw_text "拖动旋转视角·拖动圆球" 56 2 190 205 225)
      (Gfx.refresh)
    }))

;; ---- 启动 ------------------------------------------------------------------
(EX.install)
(EX.on_exit (lambda () (Raster.dispose)))
(wt_alloc)
;; 首帧：dt=0（只建焦散与画面、不推进物理）
(wt_update 0.0)
(wt_compose_camera)
(wt_scene_render)
(wt_draw_ui)
(Gfx.set_font 1)
(Gfx.draw_text "水池" 4 2 208 232 255)
(Gfx.draw_text "拖动旋转视角·拖动圆球" 56 2 190 205 225)
(Gfx.refresh)
(set! last_t (System.timestamp))
(set! tick_timer (System.set_interval TICK_MS tick))
