;; ============================================================================
;; 水池（Water Pool）—— 灵机应用程序（`infer/ui_water.c` 的**逐函数直译**）
;;
;; 本文件的目标是**保真**：把 ui_water.c（WebGL Water 的 C 软渲染移植）的
;; 相机/网格/光栅化/着色/焦散/仿真/物理逐函数翻成灵机 Scheme，使二者在任意
;; 旋转视角下逐像素一致（浮点精度差异导致的末位舍入除外）。**不追性能**：
;; 解释器里逐像素追迹很慢（见文末实测），但几何、次序、常量、公式与 C 版一一对应。
;;
;; 与 C 版的对应关系（C 函数名 → 本文件同名函数）：
;;   m4id/m4mul/m4persp/m4tr/m4rot/m4inv/wt_mvp4  矩阵
;;   wt_compose_camera / wt_get_ray_for_pixel     相机与射线
;;   wt_build_meshes                             水面/池壁/圆球网格
;;   wt_ndc_to_screen / wt_fill                  光栅化（透视校正 + z-buffer）
;;   wt_clear_frame / wt_render_cube / wt_render_water / wt_render_sphere
;;   wt_get_sphere_color / wt_get_wall_color / wt_surface_ray_color
;;   wt_sky_sample / wt_tiles_sample / wt_caustic_sample / wt_caustic_uv
;;   wt_intersect_cube / wt_intersect_sphere / wt_refract / wt_reflect
;;   wt_step_sim / wt_update_normals / wt_move_sphere / wt_volume_in_sphere
;;   wt_update_caustics / wt_caustic_project
;;   wt_update（圆球物理）
;;
;; 交互（与设备侧一致，全部触屏化）：底部控件栏（退出/暂停/重力/光源）、
;;   拖动空白处旋转视角、拖动圆球移动圆球。
;;
;; 保真度（实测，见 doc/AGENTS.md 的同日条目）：
;;   · 与 C 版在同一相机下**逐像素对照**（tools/e2e/host/water_ref.c 参照台）：
;;     76800 像素里约 10% 有差异，且差异集中在**几何边缘的 1 像素**（瓷砖缝、圆球与
;;     水面的轮廓）—— 来源是 C 用 float32、本机解释器用 float64（WASM 上是
;;     19 位尾数的 float32），`floor/ceil` 与 uv 边界判据在临界值上翻面；
;;     整片区域（池壁、水面、焦散、圆球）的颜色与位置一致。焦散纹理校验和
;;     6776.04 / 34855.7，C 版 6775.84 / 34854.65（差 3e-5，纯浮点）。
;;   · 圆球物理逐步对照一致（20 步内 center/velocity 打印值到 4 位小数相同）。
;;
;; 代价（"保真优先、不追性能"的直接后果，实测）：
;;   · 单帧约 **2.2×10⁸ 条 VM 指令**（含 1 次 wt_update：2 步波场 + 法线 + 焦散，
;;     以及池壁/水面×2/圆球三个 pass 的逐像素追迹）。宿主时间片 312500/帧，
;;     故一帧要喂 ~700 个时间片：浏览器里实测 **~12 秒/帧**（WASM，单线程）。
;;   · 已做的**不改结果**的省指令措施：`floor/ceil` 用 `mod` 自算（native 调用
;;     1.65M→0.46M 次）、整数幂用连乘、`wt_fill` 的三角形级前置判断扁平化。
;;   · 交互随之退化：渲染期间事件循环被占满，**一次不到一帧的拖动会被压成一个采样点**
;;     （实测 dx=0、视角不动）；可用形态是"逐帧推进的慢拖动"——手指持续移动时每拍
;;     取到新坐标并累加旋转。按钮同理需按住跨越一拍。
;;   · 想要交互流畅只能降级保真（例如把逐像素追迹换成低模着色）——那是另一条分支，
;;     与"逐像素一致"不可兼得。
;;
;; 灵机解释器的书写约束（详见 manual.md 与 doc/AGENTS.md）：
;;   · 算术/比较恒二参数，多元必须嵌套；
;;   · 无一元取负（写 `(- 0.0 x)`）；
;;   · 没有 let / list / map / apply，也没有 min/max；
;;   · `cond` 每子句只接受单个表达式（本文件一律用平铺的 if）；
;;   · 同一 lambda 内**不能重复 define 同名变量**（会 failed to add defines to scope），
;;     故各函数的局部量都带函数内唯一后缀；
;;   · 块内 define 会被提升到 lambda 体首部；列表用 get_item/set_item!（O(1)）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC / 底部「退出」→ 宿主全局退出确认框

;; ============================================================================
;; 原版常量（helperFunctions / main.js 直译）
;; ============================================================================

(define IOR_AIR 1.0)
(define IOR_WATER 1.333)
(define POOL_HEIGHT 1.0)
(define COL_ABOVEWATER_R 0.25)
(define COL_ABOVEWATER_G 1.0)
(define COL_ABOVEWATER_B 1.25)
(define COL_UNDERWATER_R 0.40)
(define COL_UNDERWATER_G 0.90)
(define COL_UNDERWATER_B 1.00)
(define COL_SKY2_R 0.80)
(define COL_SKY2_G 1.00)
(define COL_SKY2_B 1.10)

;; 网格分辨率：与 C 版一致（水面高度场 128、焦散 256、水面细分 44）
(define GRID_N 128)                     ; WT_GRID_N
(define CAUSTIC_N 256)                  ; WT_CAUSTIC_N
(define WATER_DETAIL 44)                ; WT_WATER_DETAIL
(define WATER_DETAIL_P1 45)
(define WATER_VERTS (* 45 45))          ; 2025
(define WATER_TRIS (* (* 44 44) 2))     ; 3872
(define CELLS (* 128 128))              ; 16384

(define CUBE_FACES 5)
(define CUBE_VERTS 20)
(define CUBE_TRIS 10)

(define SPH_RING 20)
(define SPH_SEG 36)
(define SPH_VN (* 20 37))               ; 740
(define SPH_TN (* (* 19 36) 2))         ; 1368

(define BALL_R 0.25)                    ; WT_RADIUS
(define GRAVITY_Y -4.0)                 ; WT_GRAVITY

;; 画布
(define SW (Gfx.width))
(define SH (Gfx.height))
(define NPIX (* SW SH))
(define TICK_MS 33)

;; 底部控件栏（几何同 ui_water.c）
(define BAR_H 34)
(define BAR_Y (- SH BAR_H))
(define BTN_N 4)
(define BTN_GAP 4)
(define BTN_W (Math.floor (/ (- SW (* BTN_GAP (+ BTN_N 1))) BTN_N)))
(define BTN_H (- BAR_H BTN_GAP))
(define BTN_Y (+ BAR_Y (/ BTN_GAP 2)))

(define MODE_NONE 0)
(define MODE_ROTATE 1)
(define MODE_SPHERE 2)
(define MODE_BUTTON 3)

(define BTN_BACK 0)
(define BTN_PAUSE 1)
(define BTN_GRAVITY 2)
(define BTN_LIGHT 3)

(define tick_timer 0)

;; 逐 pass 开关（调试用；正常全开）
(define PASS_CUBE 1)
(define PASS_WATER 1)
(define PASS_SPHERE 1)
(define PASS_CAUSTICS 1)
(define PASS_UI 1)

;; ============================================================================
;; 状态（对应 C 的 WT_State）
;; ============================================================================

(define angleX -25.0)
(define angleY -200.5)
(define eye_x 0.0) (define eye_y 0.0) (define eye_z 0.0)
;; 四角像素射线
(define ray00x 0.0) (define ray00y 0.0) (define ray00z 0.0)
(define ray10x 0.0) (define ray10y 0.0) (define ray10z 0.0)
(define ray01x 0.0) (define ray01y 0.0) (define ray01z 0.0)
(define ray11x 0.0) (define ray11y 0.0) (define ray11z 0.0)

(define light_dir_x 0.0) (define light_dir_y 0.0) (define light_dir_z 0.0)
(define ball_cx 0.0) (define ball_cy 0.0) (define ball_cz 0.0)
(define ball_ox 0.0) (define ball_oy 0.0) (define ball_oz 0.0)
(define ball_vx 0.0) (define ball_vy 0.0) (define ball_vz 0.0)
(define ball_gravity_x 0.0) (define ball_gravity_y -4.0) (define ball_gravity_z 0.0)
(define ball_radius 0.25)
(define use_sphere_physics 1)
(define wt_paused 0)
(define wt_mode MODE_NONE)
(define prev_hit_x 0.0) (define prev_hit_y 0.0) (define prev_hit_z 0.0)
(define plane_n_x 0.0) (define plane_n_y 0.0) (define plane_n_z 0.0)
(define mouse_down 0)
(define prev_px 0) (define prev_py 0)
(define pressed_btn -1)
(define view_dirty 0)
(define water_cur 0)                    ; 0 / 1：本拍读的页
(define ready 0)
(define light_mode 0)                   ; 0=跟随视角 1=固定斜光 2=顶上
(define light_name "光源")

;; ============================================================================
;; 大块数据（列表）
;; ============================================================================

;; 水面高度场两页：每页 GRID*GRID*4 个 float（h, v, n.x, n.z）
(define water0 (cons 0 (quote ())))
(define water1 (cons 0 (quote ())))
;; 焦散 C*C（R/G 两张）
(define caustR (cons 0 (quote ())))
(define caustG (cons 0 (quote ())))
;; 帧缓冲（RGB565）与深度缓冲
(define frame (cons 0 (quote ())))
(define zbuf (cons 0 (quote ())))
;; 水面网格：每顶点 4 个 float（vx, vy, u, v）；每三角 3 个索引
(define wm (cons 0 (quote ())))
(define wtris (cons 0 (quote ())))
;; 池壁网格：每顶点 3 个 float；每三角 3 个索引；面顶点表
(define cv (cons 0 (quote ())))
(define cube_faces (cons 0 (quote ())))
(define ctri (cons 0 (quote ())))
;; 圆球网格
(define sph (cons 0 (quote ())))
(define stri (cons 0 (quote ())))

;; 矩阵（16 元素列表）
(define matMV (cons 0 (quote ())))
(define matP (cons 0 (quote ())))
(define matMVP (cons 0 (quote ())))
(define matTmp (cons 0 (quote ())))
(define matInv (cons 0 (quote ())))

;; ============================================================================
;; 向量/矩阵工具（对应 C 的 V3/M4）
;; ============================================================================

;; 廉价 floor/ceil：`mod` 是 IL 指令（内建过程），Math.floor 是 native 调用。
;;   floor(x) = x - frac，frac = fmod(x,1) ∈ (-1,1)；frac < 0 时再减 1（= 向 -∞）。
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
(define letcube3 (lambda (x) (* (* x x) x)))
(define letsq (lambda (x) (* x x)))

;; --- 向量寄存器：结果放 rx/ry/rz；操作数放 ax/ay/az、bx/by/bz ---
(define rx 0.0) (define ry 0.0) (define rz 0.0)
(define ax 0.0) (define ay 0.0) (define az 0.0)
(define bx 0.0) (define by 0.0) (define bz 0.0)
(define rlen 0.0)
(define rdot 0.0)

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
(define mx 0.0) (define my 0.0) (define mz 0.0) (define mw 0.0)
(define wt_mvp4
  (lambda (m x y z) {
    (set! mx (+ (+ (+ (* (get_item m 0) x) (* (get_item m 1) y)) (* (get_item m 2) z)) (get_item m 3)))
    (set! my (+ (+ (+ (* (get_item m 4) x) (* (get_item m 5) y)) (* (get_item m 6) z)) (get_item m 7)))
    (set! mz (+ (+ (+ (* (get_item m 8) x) (* (get_item m 9) y)) (* (get_item m 10) z)) (get_item m 11)))
    (set! mw (+ (+ (+ (* (get_item m 12) x) (* (get_item m 13) y)) (* (get_item m 14) z)) (get_item m 15)))
  }))
;; 伴随式求逆（与 C 的 m4inv 逐项一致）
(define m4inv
  (lambda (s o) {
    (define m0 (get_item s 0)) (define m1 (get_item s 1)) (define m2 (get_item s 2)) (define m3 (get_item s 3))
    (define m4 (get_item s 4)) (define m5 (get_item s 5)) (define m6 (get_item s 6)) (define m7 (get_item s 7))
    (define m8 (get_item s 8)) (define m9 (get_item s 9)) (define m10 (get_item s 10)) (define m11 (get_item s 11))
    (define m12 (get_item s 12)) (define m13 (get_item s 13)) (define m14 (get_item s 14)) (define m15 (get_item s 15))
    (define r0 (- (+ (- (* (* m5 m10) m15) (* (* m5 m11) m14)) (* (* m9 m7) m14))
                  (+ (- (* (* m9 m6) m15) (* (* m13 m6) m11)) (* (* m13 m7) m10))))
    (define r1 (- (+ (- (* -1.0 (* (* m1 m10) m15)) (* (* m1 m11) m14)) (* (* m9 m3) m14))
                  (+ (- (* (* m9 m2) m15) (* (* m13 m2) m11)) (* (* m13 m3) m10))))
    (define r2 (- (+ (- (* (* m1 m6) m15) (* (* m1 m7) m14)) (* (* m5 m3) m14))
                  (+ (- (* (* m5 m2) m15) (* (* m13 m2) m7)) (* (* m13 m3) m6))))
    (define r3 (- (+ (- (* -1.0 (* (* m1 m6) m11)) (* (* m1 m7) m10)) (* (* m5 m3) m10))
                  (+ (- (* (* m5 m2) m11) (* (* m9 m2) m7)) (* (* m9 m3) m6))))
    (define r4 (- (+ (- (* -1.0 (* (* m4 m10) m15)) (* (* m4 m11) m14)) (* (* m8 m7) m14))
                  (+ (- (* (* m8 m6) m15) (* (* m12 m6) m11)) (* (* m12 m7) m10))))
    (define r5 (- (+ (- (* (* m0 m10) m15) (* (* m0 m11) m14)) (* (* m8 m3) m14))
                  (+ (- (* (* m8 m2) m15) (* (* m12 m2) m11)) (* (* m12 m3) m10))))
    (define r6 (- (+ (- (* -1.0 (* (* m0 m6) m15)) (* (* m0 m7) m14)) (* (* m4 m3) m14))
                  (+ (- (* (* m4 m2) m15) (* (* m12 m2) m7)) (* (* m12 m3) m6))))
    (define r7 (- (+ (- (* (* m0 m6) m11) (* (* m0 m7) m10)) (* (* m4 m3) m10))
                  (+ (- (* (* m4 m2) m11) (* (* m8 m2) m7)) (* (* m8 m3) m6))))
    (define r8 (- (+ (- (* (* m4 m9) m15) (* (* m4 m11) m13)) (* (* m8 m7) m13))
                  (+ (- (* (* m8 m5) m15) (* (* m12 m5) m11)) (* (* m12 m7) m9))))
    (define r9 (- (+ (- (* -1.0 (* (* m0 m9) m15)) (* (* m0 m11) m13)) (* (* m8 m3) m13))
                  (+ (- (* (* m8 m1) m15) (* (* m12 m1) m11)) (* (* m12 m3) m9))))
    (define r10 (- (+ (- (* (* m0 m5) m15) (* (* m0 m7) m13)) (* (* m4 m3) m13))
                   (+ (- (* (* m4 m1) m15) (* (* m12 m1) m7)) (* (* m12 m3) m5))))
    (define r11 (- (+ (- (* -1.0 (* (* m0 m5) m11)) (* (* m0 m7) m9)) (* (* m4 m3) m9))
                   (+ (- (* (* m4 m1) m11) (* (* m8 m1) m7)) (* (* m8 m3) m5))))
    (define r12 (- (+ (- (* -1.0 (* (* m4 m9) m14)) (* (* m4 m10) m13)) (* (* m8 m6) m13))
                   (+ (- (* (* m8 m5) m14) (* (* m12 m5) m10)) (* (* m12 m6) m9))))
    (define r13 (- (+ (- (* (* m0 m9) m14) (* (* m0 m10) m13)) (* (* m8 m2) m13))
                   (+ (- (* (* m8 m1) m14) (* (* m12 m1) m10)) (* (* m12 m2) m9))))
    (define r14 (- (+ (- (* -1.0 (* (* m0 m5) m14)) (* (* m0 m6) m13)) (* (* m4 m2) m13))
                   (+ (- (* (* m4 m1) m14) (* (* m12 m1) m6)) (* (* m12 m2) m5))))
    (define r15 (- (+ (- (* (* m0 m5) m10) (* (* m0 m6) m9)) (* (* m4 m2) m9))
                   (+ (- (* (* m4 m1) m10) (* (* m8 m1) m6)) (* (* m8 m2) m5))))
    (define det (+ (+ (+ (* m0 r0) (* m1 r4)) (* m2 r8)) (* m3 r12)))
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
;; 相机（wt_compose_camera + wt_get_ray_for_pixel）
;; ============================================================================

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
;; 网格构建（wt_build_meshes）
;; ============================================================================

(define wt_build_meshes
  (lambda ()
    {
      ;; ---- 水面平面（同 Mesh.plane(detail)）----
      (define D WATER_DETAIL)
      (define stride WATER_DETAIL_P1)
      (define yy 0)
      (while (<= yy D)
        {
          (define t (/ (* 1.0 yy) (* 1.0 D)))
          (define xx 0)
          (while (<= xx D)
            {
              (define s (/ (* 1.0 xx) (* 1.0 D)))
              (push wm (- (* 2.0 s) 1.0))
              (push wm (- (* 2.0 t) 1.0))
              (push wm 0.0)
              (push wm s)
              (push wm t)
              (set! xx (+ xx 1))
            })
          (set! yy (+ yy 1))
        })
      (define ty 0)
      (while (< ty D)
        {
          (define tx 0)
          (while (< tx D)
            {
              (define i (+ (* ty stride) tx))
              (push wtris i) (push wtris (+ i 1)) (push wtris (+ i stride))
              (push wtris (+ i stride)) (push wtris (+ i 1)) (push wtris (+ (+ i stride) 1))
              (set! tx (+ tx 1))
            })
          (set! ty (+ ty 1))
        })

      ;; ---- 池壁：cubeData 的 8 个角（x,y,z 为立方体坐标，cy 在渲染时映射到世界 y），
      ;;   以及 5 个面（每面 4 个角 + 2 个三角形），与 C 的 faces 表逐项一致 ----
      (push cv -1.0) (push cv -1.0) (push cv -1.0)   ;; d0
      (push cv 1.0)  (push cv -1.0) (push cv -1.0)   ;; d1
      (push cv -1.0) (push cv 1.0)  (push cv -1.0)   ;; d2
      (push cv 1.0)  (push cv 1.0)  (push cv -1.0)   ;; d3
      (push cv -1.0) (push cv -1.0) (push cv 1.0)    ;; d4
      (push cv 1.0)  (push cv -1.0) (push cv 1.0)    ;; d5
      (push cv -1.0) (push cv 1.0)  (push cv 1.0)    ;; d6
      (push cv 1.0)  (push cv 1.0)  (push cv 1.0)    ;; d7
      (define dlist (cons 0 (quote ())))
      (pop dlist)
      (push dlist 0) (push dlist 4) (push dlist 2) (push dlist 6)   ;; -x
      (push dlist 1) (push dlist 3) (push dlist 5) (push dlist 7)   ;; +x
      (push dlist 2) (push dlist 6) (push dlist 3) (push dlist 7)   ;; 底面(+y)
      (push dlist 0) (push dlist 2) (push dlist 1) (push dlist 3)   ;; -z
      (push dlist 4) (push dlist 5) (push dlist 6) (push dlist 7)   ;; +z
      (define fi 0)
      (while (< fi CUBE_FACES)
        {
          (define b (* fi 4))
          (push ctri b) (push ctri (+ b 1)) (push ctri (+ b 2))
          (push ctri (+ b 2)) (push ctri (+ b 1)) (push ctri (+ b 3))
          (set! fi (+ fi 1))
        })
      ;; 面顶点表（渲染时用 dlist 逐个取角，与 C 的 s_cv[fi*4+j] 一致）
      (set! cube_faces dlist)

      ;; ---- 圆球（UV 球，与 C 的 wt_build_meshes 同序）----
      (define rr 0)
      (while (< rr SPH_RING)
        {
          (define phi (/ (* (Math.PI) (* 1.0 rr)) (* 1.0 (- SPH_RING 1))))
          (define spy (Math.cos phi))
          (define srad (Math.sin phi))
          (define ssx 0)
          (while (<= ssx SPH_SEG)
            {
              (define th (/ (* 2.0 (* (Math.PI) (* 1.0 ssx))) (* 1.0 SPH_SEG)))
              (push sph (* srad (Math.cos th)))
              (push sph spy)
              (push sph (* srad (Math.sin th)))
              (set! ssx (+ ssx 1))
            })
          (set! rr (+ rr 1))
        })
      (set! rr 0)
      (while (< rr (- SPH_RING 1))
        {
          (define tsx 0)
          (while (< tsx SPH_SEG)
            {
              (define a (+ (* rr (+ SPH_SEG 1)) tsx))
              (define b2 (+ a 1))
              (define c2 (+ a (+ SPH_SEG 1)))
              (define d2 (+ c2 1))
              (push stri a) (push stri c2) (push stri b2)
              (push stri b2) (push stri c2) (push stri d2)
              (set! tsx (+ tsx 1))
            })
          (set! rr (+ rr 1))
        })
    }))

;; ============================================================================
;; 光栅化（wt_ndc_to_screen + wt_fill）
;;   顶点数据放全局：v0/v1/v2 的 sx sy ndcZ w wx wy wz u v
;;   frag 用整数 mode 分派（1=cube 2=water0 3=water1 4=sphere）
;; ============================================================================

(define v0sx 0.0) (define v0sy 0.0) (define v0nz 0.0) (define v0w 0.0)
(define v0wx 0.0) (define v0wy 0.0) (define v0wz 0.0) (define v0u 0.0) (define v0v 0.0)
(define v1sx 0.0) (define v1sy 0.0) (define v1nz 0.0) (define v1w 0.0)
(define v1wx 0.0) (define v1wy 0.0) (define v1wz 0.0) (define v1u 0.0) (define v1v 0.0)
(define v2sx 0.0) (define v2sy 0.0) (define v2nz 0.0) (define v2w 0.0)
(define v2wx 0.0) (define v2wy 0.0) (define v2wz 0.0) (define v2u 0.0) (define v2v 0.0)

;; 片段着色输出
(define fc_r 0.0) (define fc_g 0.0) (define fc_b 0.0)
;; 片段入参（插值后的世界坐标与 uv）
(define rw_x 0.0) (define rw_y 0.0) (define rw_z 0.0)
(define ru 0.0) (define rv 0.0)

;; wt_ndc_to_screen(MVP, world, u, v, out*) —— out 由 sel 指定写哪一组（0/1/2）
(define wt_ndc_to_screen
  (lambda (wxp wyp wzp up vp sel)
    {
      (wt_mvp4 matMVP wxp wyp wzp)
      (define cw mw)
      (define inz 0.0) (define isx 0.0) (define isy 0.0) (define iw2 0.0)
      (if (<= cw 0.0)
          { (set! inz 2.0) (set! isx 0.0) (set! isy 0.0) (set! iw2 1.0) }
          {
            (set! iw2 (/ 1.0 cw))
            (set! inz (* mz iw2))
            (set! isx (* (+ (* (* mx iw2) 0.5) 0.5) (* 1.0 SW)))
            (set! isy (* (- 1.0 (+ (* (* my iw2) 0.5) 0.5)) (* 1.0 SH)))
          })
      (if (== sel 0)
          {
            (set! v0sx isx) (set! v0sy isy) (set! v0nz inz) (set! v0w cw)
            (set! v0wx wxp) (set! v0wy wyp) (set! v0wz wzp) (set! v0u up) (set! v0v vp)
          }
          (if (== sel 1)
              {
                (set! v1sx isx) (set! v1sy isy) (set! v1nz inz) (set! v1w cw)
                (set! v1wx wxp) (set! v1wy wyp) (set! v1wz wzp) (set! v1u up) (set! v1v vp)
              }
              {
                (set! v2sx isx) (set! v2sy isy) (set! v2nz inz) (set! v2w cw)
                (set! v2wx wxp) (set! v2wy wyp) (set! v2wz wzp) (set! v2u up) (set! v2v vp)
              }))
    }))

;; 片段分派（对应 C 里传给 wt_fill 的函数指针）
(define wt_frag_dispatch
  (lambda (fmode)
    {
      (if (== fmode 1)
          (wt_cube_frag)
          (if (== fmode 2)
              (wt_water_frag 0)
              (if (== fmode 3)
                  (wt_water_frag 1)
                  (wt_sphere_frag))))
    }))

(define wt_fill
  (lambda (use_depth fmode)
    {
      (define tri_ok 1)
      (if (== v0w 0.0) (set! tri_ok 0) #f)
      (if (== v1w 0.0) (set! tri_ok 0) #f)
      (if (== v2w 0.0) (set! tri_ok 0) #f)
      (define Ax v0sx) (define Ay v0sy)
      (define Bx v1sx) (define By v1sy)
      (define Cx v2sx) (define Cy v2sy)
      (define e0 (+ (* (- By Cy) (- Ax Cx)) (* (- Cx Bx) (- Ay Cy))))
      (define rc 0.0)
      (define minx 0.0) (define maxx 0.0) (define miny 0.0) (define maxy 0.0)
      (define x0 0) (define x1 0) (define y0 0) (define y1 0)
      (if (< (Math.abs e0) 0.000001)
          (set! tri_ok 0)
          (set! rc (/ 1.0 e0)))
      (if (== tri_ok 1)
          {
            (set! minx (fmin2 Ax (fmin2 Bx Cx)))
            (set! maxx (fmax2 Ax (fmax2 Bx Cx)))
            (set! miny (fmin2 Ay (fmin2 By Cy)))
            (set! maxy (fmax2 Ay (fmax2 By Cy)))
            (if (< maxx 0.0) (set! tri_ok 0) #f)
            (if (>= minx (* 1.0 SW)) (set! tri_ok 0) #f)
            (if (< maxy 0.0) (set! tri_ok 0) #f)
            (if (>= miny (* 1.0 SH)) (set! tri_ok 0) #f)
            (if (== tri_ok 1)
                {
                  (set! x0 (flr minx))
                  (if (< x0 0.0) (set! x0 0) #f)
                  (set! x1 (clr maxx))
                  (if (> x1 (- SW 1)) (set! x1 (- SW 1)) #f)
                  (set! y0 (flr miny))
                  (if (< y0 0.0) (set! y0 0) #f)
                  (set! y1 (clr maxy))
                  (if (> y1 (- SH 1)) (set! y1 (- SH 1)) #f)
                }
                #f)
          }
          #f)
      (if (== tri_ok 1)
          {
            (define w0x (- By Cy)) (define w0y (- Cx Bx))
            (define w1x (- Cy Ay)) (define w1y (- Ax Cx))
            (define iwv0 (/ 1.0 v0w))
            (define iwv1 (/ 1.0 v1w))
            (define iwv2 (/ 1.0 v2w))
            (define py y0)
            (while (<= py y1)
              {
                (define pxc x0)
                (while (<= pxc x1)
                  {
                    (define pxd (+ (* 1.0 pxc) 0.5))
                    (define pyd (+ (* 1.0 py) 0.5))
                    (define b0 (* (+ (* w0x (- pxd Cx)) (* w0y (- pyd Cy))) rc))
                    (define b1 (* (+ (* w1x (- pxd Cx)) (* w1y (- pyd Cy))) rc))
                    (define b2 (- 1.0 (+ b0 b1)))
                    (define dodraw 1)
                    (if (< b0 0.0) (set! dodraw 0) #f)
                    (if (< b1 0.0) (set! dodraw 0) #f)
                    (if (< b2 0.0) (set! dodraw 0) #f)
                    (define idx (+ (* py SW) pxc))
                    (define d16 0.0)
                    (if (== dodraw 1)
                        {
                          (define denom (+ (+ (* b0 iwv0) (* b1 iwv1)) (* b2 iwv2)))
                          (define ndcZ (/ (+ (+ (* (* b0 iwv0) v0nz) (* (* b1 iwv1) v1nz))
                                             (* (* b2 iwv2) v2nz)) denom))
                          (set! d16 (flr (+ (* (* ndcZ 0.5) 65535.0) 32767.5)))
                          (if (== use_depth 1)
                              (if (>= d16 (get_item zbuf idx)) (set! dodraw 0) #f)
                              #f)
                          (if (== dodraw 1)
                              {
                                (set! rw_x (/ (+ (+ (* (* b0 iwv0) v0wx) (* (* b1 iwv1) v1wx))
                                                  (* (* b2 iwv2) v2wx)) denom))
                                (set! rw_y (/ (+ (+ (* (* b0 iwv0) v0wy) (* (* b1 iwv1) v1wy))
                                                  (* (* b2 iwv2) v2wy)) denom))
                                (set! rw_z (/ (+ (+ (* (* b0 iwv0) v0wz) (* (* b1 iwv1) v1wz))
                                                  (* (* b2 iwv2) v2wz)) denom))
                                (set! ru (/ (+ (+ (* (* b0 iwv0) v0u) (* (* b1 iwv1) v1u))
                                                (* (* b2 iwv2) v2u)) denom))
                                (set! rv (/ (+ (+ (* (* b0 iwv0) v0v) (* (* b1 iwv1) v1v))
                                                (* (* b2 iwv2) v2v)) denom))
                                (wt_frag_dispatch fmode)
                                (define cr (clampf fc_r 0.0 1.0))
                                (define cg (clampf fc_g 0.0 1.0))
                                (define cb (clampf fc_b 0.0 1.0))
                                (set_item! frame idx
                                  (+ (+ (* (flr (+ (* cr 31.0) 0.5)) 2048)
                                        (* (flr (+ (* cg 63.0) 0.5)) 32))
                                     (flr (+ (* cb 31.0) 0.5))))
                                (if (== use_depth 1) (set_item! zbuf idx d16) #f)
                              }
                              #f)
                        }
                        #f)
                    (set! pxc (+ pxc 1))
                  })
                (set! py (+ py 1))
              })
          }
          #f)
    }))

(define wt_clear_frame
  (lambda ()
    {
      (define i 0)
      (while (< i NPIX)
        {
          (set_item! frame i 0)
          (set_item! zbuf i 65535)
          (set! i (+ i 1))
        })
    }))

;; ============================================================================
;; 采样与几何工具（tiles / sky / 水面高度 / 交点 / 折射反射）
;; ============================================================================

;; 水面页（本拍读的那一页）
(define wt_page
  (lambda () (if (== water_cur 0) water0 water1)))

;; texture2D(water, (x, y)) 的 CLAMP 取值：ch = 0..3
(define wt_wat_at
  (lambda (xx yy ch)
    {
      (define ix xx)
      (define iy yy)
      (if (< ix 0) (set! ix 0) (if (> ix (- GRID_N 1)) (set! ix (- GRID_N 1)) #f))
      (if (< iy 0) (set! iy 0) (if (> iy (- GRID_N 1)) (set! iy (- GRID_N 1)) #f))
      (define pg (wt_page))
      (get_item pg (+ (* (+ (* iy GRID_N) ix) 4) ch))
    }))

;; 双线性采样（GL_LINEAR + CLAMP_TO_EDGE）
(define wt_wat_sample
  (lambda (u v ch)
    {
      (define fx (- (* u (* 1.0 GRID_N)) 0.5))
      (define fy (- (* v (* 1.0 GRID_N)) 0.5))
      (define x0 (flr fx))
      (define y0 (flr fy))
      (define ffx (- fx (* 1.0 x0)))
      (define ffy (- fy (* 1.0 y0)))
      (define a (wt_wat_at x0 y0 ch))
      (define b (wt_wat_at (+ x0 1) y0 ch))
      (define c (wt_wat_at x0 (+ y0 1) ch))
      (define d (wt_wat_at (+ x0 1) (+ y0 1) ch))
      (+ (+ (* a (* (- 1.0 ffx) (- 1.0 ffy))) (* b (* ffx (- 1.0 ffy))))
         (+ (* c (* (- 1.0 ffx) ffy)) (* d (* ffx ffy))))
    }))

;; 焦散纹理采样（双线性，CLAMP）
(define caust_r 0.0) (define caust_g 0.0)
(define wt_caustic_sample
  (lambda (u v)
    {
      (define uu (clampf u 0.0 1.0))
      (define vv (clampf v 0.0 1.0))
      (define fx (- (* uu (* 1.0 CAUSTIC_N)) 0.5))
      (define fy (- (* vv (* 1.0 CAUSTIC_N)) 0.5))
      (define x0 (flr fx))
      (define y0 (flr fy))
      (define x1 (+ x0 1))
      (define y1 (+ y0 1))
      (define xa x0) (define ya y0) (define xb x1) (define yb y1)
      (if (< xa 0) (set! xa 0) (if (> xa (- CAUSTIC_N 1)) (set! xa (- CAUSTIC_N 1)) #f))
      (if (> xb (- CAUSTIC_N 1)) (set! xb (- CAUSTIC_N 1)) #f)
      (if (< xb 0) (set! xb 0) #f)
      (if (< ya 0) (set! ya 0) (if (> ya (- CAUSTIC_N 1)) (set! ya (- CAUSTIC_N 1)) #f))
      (if (> yb (- CAUSTIC_N 1)) (set! yb (- CAUSTIC_N 1)) #f)
      (if (< yb 0) (set! yb 0) #f)
      (define ffx (- fx x0))
      (define ffy (- fy y0))
      (define w0 (* (- 1.0 ffx) (- 1.0 ffy)))
      (define w1 (* ffx (- 1.0 ffy)))
      (define w2 (* (- 1.0 ffx) ffy))
      (define w3 (* ffx ffy))
      (define i00 (+ (* ya CAUSTIC_N) xa))
      (define i01 (+ (* ya CAUSTIC_N) xb))
      (define i10 (+ (* yb CAUSTIC_N) xa))
      (define i11 (+ (* yb CAUSTIC_N) xb))
      (set! caust_r (+ (+ (* (get_item caustR i00) w0) (* (get_item caustR i01) w1))
                       (+ (* (get_item caustR i10) w2) (* (get_item caustR i11) w3))))
      (set! caust_g (+ (+ (* (get_item caustG i00) w0) (* (get_item caustG i01) w1))
                       (+ (* (get_item caustG i10) w2) (* (get_item caustG i11) w3))))
    }))

;; 焦散 uv：0.75*(p - p.y*refr/refr.y)*0.5 + 0.5 → cu/cv
(define caust_u 0.0) (define caust_v 0.0)
(define wt_caustic_uv
  (lambda (px py pz qx qy qz)
    {
      (set! caust_u (+ (* 0.75 (* (- px (/ (* py qx) qy)) 0.5)) 0.5))
      (set! caust_v (+ (* 0.75 (* (- pz (/ (* py qz) qy)) 0.5)) 0.5))
    }))

;; 程序化天空（textureCube(sky, ray)）→ sk_r/sk_g/sk_b
(define sk_r 0.0) (define sk_g 0.0) (define sk_b 0.0)
(define wt_sky_sample
  (lambda (dx dy dz)
    {
      (define up (fmax2 0.0 dy))
      (define tt (Math.pow (fmin2 1.0 (* up 1.7)) 0.85))
      (define rock (* (* 0.04 (Math.sin (* dx 6.0))) (Math.pow (* up 0.6) 3.0)))
      (define rr2 (+ (+ 0.04 (* 0.55 tt)) rock))
      (define gg2 (+ (+ 0.06 (* 0.75 tt)) rock))
      (define bb2 (+ (+ 0.11 (* 0.98 tt)) rock))
      (set! sk_r (clampf rr2 0.0 1.0))
      (set! sk_g (clampf gg2 0.0 1.0))
      (set! sk_b (clampf bb2 0.0 1.0))
    }))

;; 程序化瓷砖（texture2D(tiles, uv)，REPEAT）→ ti_r/ti_g/ti_b
;;   C: fu=floor(u/T), xu=u/T-fu, cu=(int)fmod(fu,2), shade=1-0.06*((cu^cv)&1),
;;      grout = (xu<0.05 || xv<0.05) ? 0.25 : 1
(define ti_r 0.0) (define ti_g 0.0) (define ti_b 0.0)
(define cpar 0.0)
(define wt_tiles_sample
  (lambda (tu tv)
    {
      (define TT (/ 1.0 5.0))
      (define fu (flr (/ tu TT)))
      (define fv (flr (/ tv TT)))
      (define xu (- (/ tu TT) fu))
      (define xv (- (/ tv TT) fv))
      ;; cu = (int)fmod(fu,2)（C 的 (int) 向零截断），parity = |fmod(cu,2)|
      (define mcu (mod fu 2.0))
      (define mcv (mod fv 2.0))
      (define cu2 (flr mcu))
      (define cv2 (flr mcv))
      (set! cpar (Math.abs (mod (+ cu2 cv2) 2.0)))
      (define shade (- 1.0 (* 0.06 cpar)))
      (define grout 1.0)
      (if (< xu 0.05) (set! grout 0.25) #f)
      (if (< xv 0.05) (set! grout 0.25) #f)
      (set! ti_r (* (+ 0.42 (* 0.20 shade)) grout))
      (set! ti_g (* (+ 0.50 (* 0.20 shade)) grout))
      (set! ti_b (* (+ 0.56 (* 0.20 shade)) grout))
    }))

;; 光线与盒（intersectCube）→ tnear=tnx, tfar=tfx
(define tnx 0.0) (define tfx 0.0)
(define wt_intersect_cube
  (lambda (ox oy oz dx dy dz mnx mny mnz mxx mxy mxz)
    {
      (define tmnx (/ (- mnx ox) dx))
      (define tmny (/ (- mny oy) dy))
      (define tmnz (/ (- mnz oz) dz))
      (define tmx (/ (- mxx ox) dx))
      (define tmy (/ (- mxy oy) dy))
      (define tmz (/ (- mxz oz) dz))
      (define t1x (fmin2 tmnx tmx))
      (define t1y (fmin2 tmny tmy))
      (define t1z (fmin2 tmnz tmz))
      (define t2x (fmax2 tmnx tmx))
      (define t2y (fmax2 tmny tmy))
      (define t2z (fmax2 tmnz tmz))
      (set! tnx (fmax2 (fmax2 t1x t1y) t1z))
      (set! tfx (fmin2 (fmin2 t2x t2y) t2z))
    }))

;; 光线与球（intersectSphere）→ 返回值（用 hit_t 传回；>1e6 视为未命中）
(define hit_t 0.0)
(define wt_intersect_sphere
  (lambda (ox oy oz dx dy dz ccx ccy ccz rad)
    {
      (define tsx (- ox ccx))
      (define tsy (- oy ccy))
      (define tsz (- oz ccz))
      (define aa (+ (+ (* dx dx) (* dy dy)) (* dz dz)))
      (define bb (* 2.0 (+ (+ (* tsx dx) (* tsy dy)) (* tsz dz))))
      (define cc2 (+ (+ (* tsx tsx) (* tsy tsy)) (* tsz tsz)))
      (define cc (- cc2 (* rad rad)))
      (define disc (- (* bb bb) (* 4.0 (* aa cc))))
      (if (> disc 0.0)
          {
            (define tt (/ (- (- 0.0 bb) (Math.sqrt disc)) (* 2.0 aa)))
            (if (> tt 0.0) (set! hit_t tt) (set! hit_t 1000000.0))
          }
          (set! hit_t 1000000.0))
    }))

;; 折射（wt_refract）→ rfx/rfy/rfz；ok=0 表示全反射
(define rfx 0.0) (define rfy 0.0) (define rfz 0.0)
(define rf_ok 1)
(define wt_refract
  (lambda (ixx iyy izz nxx nyy nzz eta)
    {
      (define dd (+ (+ (* nxx ixx) (* nyy iyy)) (* nzz izz)))
      (define kk (- 1.0 (* (* eta eta) (- 1.0 (* dd dd)))))
      (if (< kk 0.0)
          { (set! rf_ok 0) (set! rfx 0.0) (set! rfy 0.0) (set! rfz 0.0) }
          {
            (set! rf_ok 1)
            (define sc (+ (* eta dd) (Math.sqrt kk)))
            (set! rfx (- (* ixx eta) (* nxx sc)))
            (set! rfy (- (* iyy eta) (* nyy sc)))
            (set! rfz (- (* izz eta) (* nzz sc)))
          })
    }))

;; 反射（wt_reflect）→ rfx/rfy/rfz
(define wt_reflect
  (lambda (ixx iyy izz nxx nyy nzz)
    {
      (define dd (+ (+ (* nxx ixx) (* nyy iyy)) (* nzz izz)))
      (set! rfx (- ixx (* (* nxx 2.0) dd)))
      (set! rfy (- iyy (* (* nyy 2.0) dd)))
      (set! rfz (- izz (* (* nzz 2.0) dd)))
    }))

;; ============================================================================
;; 着色（getSphereColor / getWallColor / getSurfaceRayColor）
;; ============================================================================

;; 折射后的平行光（-lightDir 以 (0,1,0) 为法线折射）——C 里在多个着色函数里重复计算
(define rl_x 0.0) (define rl_y 0.0) (define rl_z 0.0)
(define wt_refracted_light
  (lambda ()
    {
      (wt_refract (- 0.0 light_dir_x) (- 0.0 light_dir_y) (- 0.0 light_dir_z) 0.0 1.0 0.0
                  (/ IOR_AIR IOR_WATER))
      (set! rl_x rfx) (set! rl_y rfy) (set! rl_z rfz)
    }))

;; getSphereColor(point) → sc_r/sc_g/sc_b
(define sc_r 0.0) (define sc_g 0.0) (define sc_b 0.0)
(define wt_get_sphere_color
  (lambda (px py pz)
    {
      (define rad ball_radius)
      (define col 0.5)
      (set! col (* col (- 1.0 (/ 0.9 (letcube3 (/ (+ (+ 1.0 rad) (- 0.0 (Math.abs px))) rad))))))
      (set! col (* col (- 1.0 (/ 0.9 (letcube3 (/ (+ (+ 1.0 rad) (- 0.0 (Math.abs pz))) rad))))))
      (set! col (* col (- 1.0 (/ 0.9 (letcube3 (/ (+ (+ py 1.0) rad) rad))))))
      ;; sphereNormal
      (define snx (/ (- px ball_cx) rad))
      (define sny (/ (- py ball_cy) rad))
      (define snz (/ (- pz ball_cz) rad))
      (wt_refracted_light)
      ;; diffuse = max(0, dot(-refractedLight, sphereNormal)) * 0.5
      (v_set_a (- 0.0 rl_x) (- 0.0 rl_y) (- 0.0 rl_z))
      (v_set_b snx sny snz)
      (v_dot)
      (define diff (fmax2 0.0 rdot))
      (set! diff (* diff 0.5))
      ;; 水下：乘焦散
      (define hh (wt_wat_sample (+ (* px 0.5) 0.5) (+ (* pz 0.5) 0.5) 0))
      (if (< py hh)
          {
            (wt_caustic_uv px py pz rl_x rl_y rl_z)
            (wt_caustic_sample caust_u caust_v)
            (set! diff (* diff (* caust_r 4.0)))
          }
          #f)
      (set! col (+ col diff))
      (set! sc_r col) (set! sc_g col) (set! sc_b col)
    }))

;; getWallColor(point) → wc_r/wc_g/wc_b
(define wc_r 0.0) (define wc_g 0.0) (define wc_b 0.0)
(define wt_get_wall_color
  (lambda (px py pz)
    {
      (define scale 0.5)
      (define nX 0.0) (define nY 1.0) (define nZ 0.0)
      (if (> (Math.abs px) 0.999)
          {
            (wt_tiles_sample (+ (* py 0.5) 1.0) (+ (* pz 0.5) 0.5))
            (set! nX (- 0.0 px)) (set! nY 0.0) (set! nZ 0.0)
          }
          (if (> (Math.abs pz) 0.999)
              {
                (wt_tiles_sample (+ (* py 0.5) 1.0) (+ (* px 0.5) 0.5))
                (set! nX 0.0) (set! nY 0.0) (set! nZ (- 0.0 pz))
              }
              {
                (wt_tiles_sample (+ (* px 0.5) 0.5) (+ (* pz 0.5) 0.5))
                (set! nX 0.0) (set! nY 1.0) (set! nZ 0.0)
              }))
      (define wallR ti_r) (define wallG ti_g) (define wallB ti_b)
      ;; scale /= |point| ; scale *= 1 - 0.9/pow(|point-center|/radius, 4)
      (v_set_a px py pz)
      (v_len)
      (set! scale (/ scale rlen))
      (v_set_a (- px ball_cx) (- py ball_cy) (- pz ball_cz))
      (v_len)
      (set! scale (* scale (- 1.0 (/ 0.9 (letsq (letsq (/ rlen ball_radius)))))))
      ;; 折射光（取反）
      (wt_refracted_light)
      (define rlx (- 0.0 rl_x)) (define rly (- 0.0 rl_y)) (define rlz (- 0.0 rl_z))
      (v_set_a rlx rly rlz)
      (v_set_b nX nY nZ)
      (v_dot)
      (define diff (fmax2 0.0 rdot))
      (define hh (wt_wat_sample (+ (* px 0.5) 0.5) (+ (* pz 0.5) 0.5) 0))
      (if (< py hh)
          {
            (wt_caustic_uv px py pz rlx rly rlz)
            (wt_caustic_sample caust_u caust_v)
            (set! scale (+ scale (* (* diff (* caust_r 2.0)) caust_g)))
          }
          {
            ;; 池沿阴影（rim）
            (wt_intersect_cube px py pz rlx rly rlz -1.0 (- 0.0 POOL_HEIGHT) -1.0 1.0 2.0 1.0)
            (define t0 tnx) (define t1 tfx)
            (define rimA (/ (- 0.0 200.0) (+ 1.0 (* 10.0 (- t1 t0)))))
            (define rimB (- (+ py (* rly t1)) (/ 2.0 12.0)))
            (define rim (/ 1.0 (+ 1.0 (Math.exp (* rimA rimB)))))
            (set! diff (* diff rim))
            (set! scale (+ scale (* diff 0.5)))
          })
      (set! wc_r (* wallR scale))
      (set! wc_g (* wallG scale))
      (set! wc_b (* wallB scale))
    }))

;; getSurfaceRayColor(origin, ray, waterColor) → sr_r/sr_g/sr_b
(define sr_r 0.0) (define sr_g 0.0) (define sr_b 0.0)
(define wt_surface_ray_color
  (lambda (ox oy oz dx dy dz wr wg wb)
    {
      (define cr 0.0) (define cg 0.0) (define cb 0.0)
      (wt_intersect_sphere ox oy oz dx dy dz ball_cx ball_cy ball_cz ball_radius)
      (define q hit_t)
      (if (< q 1000000.0)
          {
            (wt_get_sphere_color (+ ox (* dx q)) (+ oy (* dy q)) (+ oz (* dz q)))
            (set! cr sc_r) (set! cg sc_g) (set! cb sc_b)
          }
          (if (< dy 0.0)
              {
                (wt_intersect_cube ox oy oz dx dy dz -1.0 (- 0.0 POOL_HEIGHT) -1.0 1.0 2.0 1.0)
                (wt_get_wall_color (+ ox (* dx tfx)) (+ oy (* dy tfx)) (+ oz (* dz tfx)))
                (set! cr wc_r) (set! cg wc_g) (set! cb wc_b)
              }
              {
                (wt_intersect_cube ox oy oz dx dy dz -1.0 (- 0.0 POOL_HEIGHT) -1.0 1.0 2.0 1.0)
                (define hx (+ ox (* dx tfx)))
                (define hy (+ oy (* dy tfx)))
                (define hz (+ oz (* dz tfx)))
                (if (< hy (/ 2.0 12.0))
                    {
                      (wt_get_wall_color hx hy hz)
                      (set! cr wc_r) (set! cg wc_g) (set! cb wc_b)
                    }
                    {
                      (wt_sky_sample dx dy dz)
                      (set! cr sk_r) (set! cg sk_g) (set! cb sk_b)
                      (v_set_a light_dir_x light_dir_y light_dir_z)
                      (v_set_b dx dy dz)
                      (v_dot)
                      (define sun (Math.pow (fmax2 0.0 rdot) 5000.0))
                      (set! cr (+ cr (* sun 10.0)))
                      (set! cg (+ cg (* sun 8.0)))
                      (set! cb (+ cb (* sun 6.0)))
                    })
              }))
      (if (< dy 0.0)
          { (set! cr (* cr wr)) (set! cg (* cg wg)) (set! cb (* cb wb)) }
          #f)
      (set! sr_r cr) (set! sr_g cg) (set! sr_b cb)
    }))

;; ============================================================================
;; 片段着色器（wt_cube_frag / wt_water_frag / wt_sphere_frag）
;; ============================================================================

;; 池壁片段：getWallColor + 水下透染
(define wt_cube_frag
  (lambda ()
    {
      (wt_get_wall_color rw_x rw_y rw_z)
      (define hh (wt_wat_sample (+ (* rw_x 0.5) 0.5) (+ (* rw_z 0.5) 0.5) 0))
      (if (< rw_y hh)
          {
            (set! fc_r (* wc_r (* COL_UNDERWATER_R 1.2)))
            (set! fc_g (* wc_g (* COL_UNDERWATER_G 1.2)))
            (set! fc_b (* wc_b (* COL_UNDERWATER_B 1.2)))
          }
          { (set! fc_r wc_r) (set! fc_g wc_g) (set! fc_b wc_b) })
    }))

;; 水面片段（pass 0 = 水面以上 / 1 = 水面以下）
(define wt_water_frag
  (lambda (pass)
    {
      (define coordU ru)
      (define coordV rv)
      (define infoB 0.0) (define infoA 0.0) (define hgt 0.0)
      (define i5 0)
      ;; “peaked”：5 次沿切线方向偏移重采样
      (while (< i5 5)
        {
          (set! hgt (wt_wat_sample coordU coordV 0))
          (set! infoB (wt_wat_sample coordU coordV 2))
          (set! infoA (wt_wat_sample coordU coordV 3))
          (set! coordU (+ coordU (* infoB 0.005)))
          (set! coordV (+ coordV (* infoA 0.005)))
          (set! i5 (+ i5 1))
        })
      (define nx1 infoB)
      (define ny1 (Math.sqrt (fmax2 0.0 (- 1.0 (+ (* infoB infoB) (* infoA infoA))))))
      (define nz1 infoA)
      ;; incomingRay = normalize(world - eye)
      (v_set_a (- rw_x eye_x) (- rw_y eye_y) (- rw_z eye_z))
      (v_norm)
      (define irx rx) (define iry ry) (define irz rz)
      (if (== pass 0)
          {
            ;; refl = reflect(incoming, normal)
            (wt_reflect irx iry irz nx1 ny1 nz1)
            (define rr2 rfx) (define rg2 rfy) (define rb2 rfz)
            ;; refr = refract(incoming, normal, air/water)
            (wt_refract irx iry irz nx1 ny1 nz1 (/ IOR_AIR IOR_WATER))
            (define erx rfx) (define ery rfy) (define erz rfz)
            ;; fresnel = 0.25 + 0.75*pow(1 - dot(normal, -incoming), 3)
            (v_set_a nx1 ny1 nz1)
            (v_set_b (- 0.0 irx) (- 0.0 iry) (- 0.0 irz))
            (v_dot)
            (define fresnel (+ 0.25 (* 0.75 (letcube3 (- 1.0 rdot)))))
            (wt_surface_ray_color rw_x rw_y rw_z rr2 rg2 rb2
                                  COL_ABOVEWATER_R COL_ABOVEWATER_G COL_ABOVEWATER_B)
            (define ar sr_r) (define ag sr_g) (define ab sr_b)
            (wt_surface_ray_color rw_x rw_y rw_z erx ery erz
                                  COL_ABOVEWATER_R COL_ABOVEWATER_G COL_ABOVEWATER_B)
            (set! fc_r (+ sr_r (* (- ar sr_r) fresnel)))
            (set! fc_g (+ sr_g (* (- ag sr_g) fresnel)))
            (set! fc_b (+ sr_b (* (- ab sr_b) fresnel)))
          }
          {
            ;; 水下：法线取反、eta 取 water/air
            (define nx2 (- 0.0 nx1)) (define ny2 (- 0.0 ny1)) (define nz2 (- 0.0 nz1))
            (wt_reflect irx iry irz nx2 ny2 nz2)
            (define w_rr2 rfx) (define w_rg2 rfy) (define w_rb2 rfz)
            (wt_refract irx iry irz nx2 ny2 nz2 (/ IOR_WATER IOR_AIR))
            (define w_erx rfx) (define w_ery rfy) (define w_erz rfz)
            (v_set_a nx2 ny2 nz2)
            (v_set_b (- 0.0 irx) (- 0.0 iry) (- 0.0 irz))
            (v_dot)
            (define w_fres (+ 0.5 (* 0.5 (letcube3 (- 1.0 rdot)))))
            (wt_surface_ray_color rw_x rw_y rw_z w_rr2 w_rg2 w_rb2
                                  COL_UNDERWATER_R COL_UNDERWATER_G COL_UNDERWATER_B)
            (define w_ar sr_r) (define w_ag sr_g) (define w_ab sr_b)
            (wt_surface_ray_color rw_x rw_y rw_z w_erx w_ery w_erz 1.0 1.0 1.0)
            (define e2r (* sr_r COL_SKY2_R))
            (define e2g (* sr_g COL_SKY2_G))
            (define e2b (* sr_b COL_SKY2_B))
            ;; t = (1-w_fres)*length(refr)
            (v_set_a w_erx w_ery w_erz)
            (v_len)
            (define tt (* (- 1.0 w_fres) rlen))
            (set! fc_r (+ w_ar (* (- e2r w_ar) tt)))
            (set! fc_g (+ w_ag (* (- e2g w_ag) tt)))
            (set! fc_b (+ w_ab (* (- e2b w_ab) tt)))
          })
    }))

;; 圆球片段：getSphereColor + 水下透染
(define wt_sphere_frag
  (lambda ()
    {
      (wt_get_sphere_color rw_x rw_y rw_z)
      (define hh (wt_wat_sample (+ (* rw_x 0.5) 0.5) (+ (* rw_z 0.5) 0.5) 0))
      (if (< rw_y hh)
          {
            (set! fc_r (* sc_r (* COL_UNDERWATER_R 1.2)))
            (set! fc_g (* sc_g (* COL_UNDERWATER_G 1.2)))
            (set! fc_b (* sc_b (* COL_UNDERWATER_B 1.2)))
          }
          { (set! fc_r sc_r) (set! fc_g sc_g) (set! fc_b sc_b) })
    }))

;; ============================================================================
;; 渲染 pass（wt_render_cube / wt_render_water / wt_render_sphere）
;; ============================================================================

;; 屏幕空间带符号面积（判正面用）
(define wt_screen_area
  (lambda ()
    (- (* (- v1sx v0sx) (- v2sy v0sy)) (* (- v1sy v0sy) (- v2sx v0sx)))))

;; 池壁世界 y：((1-cv.y)*7/12-1)*POOL_HEIGHT
(define cube_world_y
  (lambda (cyv) (* (- (* (- 1.0 cyv) (/ 7.0 12.0)) 1.0) POOL_HEIGHT)))

(define wt_render_cube
  (lambda ()
    {
      (define t 0)
      (while (< t CUBE_TRIS)
        {
          (define i0 (get_item ctri (* t 3)))
          (define i1 (get_item ctri (+ (* t 3) 1)))
          (define i2 (get_item ctri (+ (* t 3) 2)))
          ;; d 索引 → 面顶点表（tri 索引本身就是面顶点下标，见 wt_build_meshes）
          (define d0 (get_item cube_faces i0))
          (define d1 (get_item cube_faces i1))
          (define d2 (get_item cube_faces i2))
          (define w0x (get_item cv (* d0 3)))
          (define w0y (cube_world_y (get_item cv (+ (* d0 3) 1))))
          (define w0z (get_item cv (+ (* d0 3) 2)))
          (define w1x (get_item cv (* d1 3)))
          (define w1y (cube_world_y (get_item cv (+ (* d1 3) 1))))
          (define w1z (get_item cv (+ (* d1 3) 2)))
          (define w2x (get_item cv (* d2 3)))
          (define w2y (cube_world_y (get_item cv (+ (* d2 3) 1))))
          (define w2z (get_item cv (+ (* d2 3) 2)))
          (wt_ndc_to_screen w0x w0y w0z 0.0 0.0 0)
          (wt_ndc_to_screen w1x w1y w1z 0.0 0.0 1)
          (wt_ndc_to_screen w2x w2y w2z 0.0 0.0 2)
          (if (<= v0w 0.0) #f
            (if (<= v1w 0.0) #f
              (if (<= v2w 0.0) #f
                {
                  (define area (wt_screen_area))
                  (if (>= area 0.0) #f (wt_fill 1 1))
                })))
          (set! t (+ t 1))
        })
    }))

;; 水面顶点世界坐标（顶点索引 → 世界 x,y,z + u,v）
(define gw_x 0.0) (define gw_y 0.0) (define gw_z 0.0) (define gw_u 0.0) (define gw_v 0.0)
(define wt_water_vert_world
  (lambda (vi)
    {
      (define base (* vi 5))
      (set! gw_x (get_item wm base))
      (set! gw_z (get_item wm (+ base 1)))     ;; 注意：Mesh.plane 的第二个分量是 y，渲染时放到 z
      (set! gw_u (get_item wm (+ base 3)))
      (set! gw_v (get_item wm (+ base 4)))
      (set! gw_y (wt_wat_sample gw_u gw_v 0))
    }))

(define wt_render_water
  (lambda ()
    {
      (define pass 0)
      (while (< pass 2)
        {
          (define t 0)
          (while (< t WATER_TRIS)
            {
              (define i0 (get_item wtris (* t 3)))
              (define i1 (get_item wtris (+ (* t 3) 1)))
              (define i2 (get_item wtris (+ (* t 3) 2)))
              (wt_water_vert_world i0)
              (wt_ndc_to_screen gw_x gw_y gw_z gw_u gw_v 0)
              (wt_water_vert_world i1)
              (wt_ndc_to_screen gw_x gw_y gw_z gw_u gw_v 1)
              (wt_water_vert_world i2)
              (wt_ndc_to_screen gw_x gw_y gw_z gw_u gw_v 2)
              (if (<= v0w 0.0) #f
                (if (<= v1w 0.0) #f
                  (if (<= v2w 0.0) #f
                    {
                      (define area (wt_screen_area))
                      (define drawit 0)
                      (if (== pass 0)
                          (if (< area 0.0) #f (set! drawit 1))
                          (if (>= area 0.0) #f (set! drawit 1)))
                      (if (== drawit 1)
                          (if (== pass 0) (wt_fill 1 2) (wt_fill 1 3))
                          #f)
                    })))
              (set! t (+ t 1))
            })
          (set! pass (+ pass 1))
        })
    }))

(define wt_render_sphere
  (lambda ()
    {
      (define t 0)
      (while (< t SPH_TN)
        {
          (define i0 (get_item stri (* t 3)))
          (define i1 (get_item stri (+ (* t 3) 1)))
          (define i2 (get_item stri (+ (* t 3) 2)))
          (define s0x (+ ball_cx (* (get_item sph (* i0 3)) ball_radius)))
          (define s0y (+ ball_cy (* (get_item sph (+ (* i0 3) 1)) ball_radius)))
          (define s0z (+ ball_cz (* (get_item sph (+ (* i0 3) 2)) ball_radius)))
          (define s1x (+ ball_cx (* (get_item sph (* i1 3)) ball_radius)))
          (define s1y (+ ball_cy (* (get_item sph (+ (* i1 3) 1)) ball_radius)))
          (define s1z (+ ball_cz (* (get_item sph (+ (* i1 3) 2)) ball_radius)))
          (define s2x (+ ball_cx (* (get_item sph (* i2 3)) ball_radius)))
          (define s2y (+ ball_cy (* (get_item sph (+ (* i2 3) 1)) ball_radius)))
          (define s2z (+ ball_cz (* (get_item sph (+ (* i2 3) 2)) ball_radius)))
          (wt_ndc_to_screen s0x s0y s0z 0.0 0.0 0)
          (wt_ndc_to_screen s1x s1y s1z 0.0 0.0 1)
          (wt_ndc_to_screen s2x s2y s2z 0.0 0.0 2)
          (if (<= v0w 0.0) #f
            (if (<= v1w 0.0) #f
              (if (<= v2w 0.0) #f
                (wt_fill 1 4))))
          (set! t (+ t 1))
        })
    }))

;; ============================================================================
;; 焦散 pass（wt_update_caustics / wt_caustic_project）
;; ============================================================================

;; 沿 ray 穿出池盒后，沿折射光投影到 y=-1 平面 → pjx/pjy/pjz
(define pjx 0.0) (define pjy 0.0) (define pjz 0.0)
(define wt_caustic_project
  (lambda (ox oy oz dx dy dz rlx rly rlz)
    {
      (wt_intersect_cube ox oy oz dx dy dz -1.0 (- 0.0 POOL_HEIGHT) -1.0 1.0 2.0 1.0)
      (define hx (+ ox (* dx tfx)))
      (define hy (+ oy (* dy tfx)))
      (define hz (+ oz (* dz tfx)))
      (define tplane (/ (- (- 0.0 hy) 1.0) rly))
      (set! pjx (+ hx (* rlx tplane)))
      (set! pjy (+ hy (* rly tplane)))
      (set! pjz (+ hz (* rlz tplane)))
    }))

;; 每个三角形顶点：(sx, sy), oldP, newP, ratio
(define c0sx 0.0) (define c0sy 0.0) (define c0ox 0.0) (define c0oy 0.0) (define c0oz 0.0)
(define c0nx 0.0) (define c0ny 0.0) (define c0nz 0.0) (define c0rt 1.0)
(define c1sx 0.0) (define c1sy 0.0) (define c1ox 0.0) (define c1oy 0.0) (define c1oz 0.0)
(define c1nx 0.0) (define c1ny 0.0) (define c1nz 0.0) (define c1rt 1.0)
(define c2sx 0.0) (define c2sy 0.0) (define c2ox 0.0) (define c2oy 0.0) (define c2oz 0.0)
(define c2nx 0.0) (define c2ny 0.0) (define c2nz 0.0) (define c2rt 1.0)

(define wt_update_caustics
  (lambda ()
    {
      ;; 清零
      (define z0 0)
      (while (< z0 (* CAUSTIC_N CAUSTIC_N))
        { (set_item! caustR z0 0.0) (set_item! caustG z0 0.0) (set! z0 (+ z0 1)) })
      (wt_refracted_light)
      (define rlx rl_x) (define rly rl_y) (define rlz rl_z)
      (define t 0)
      (while (< t WATER_TRIS)
        {
          (define vi 0)
          (while (< vi 3)
            {
              (define idx (get_item wtris (+ (* t 3) vi)))
              (define base (* idx 5))
              (define vvx (get_item wm base))
              (define vvz (get_item wm (+ base 1)))
              (define vvu (get_item wm (+ base 3)))
              (define vvv (get_item wm (+ base 4)))
              ;; info → normal（b,a 各乘 0.5）
              (define hk (wt_wat_sample vvu vvv 0))
              (define bb (wt_wat_sample vvu vvv 2))
              (define aa (wt_wat_sample vvu vvv 3))
              (set! bb (* bb 0.5))
              (set! aa (* aa 0.5))
              (define nyy (Math.sqrt (fmax2 0.0 (- 1.0 (+ (* bb bb) (* aa aa))))))
              ;; ray = refract(-lightDir, normal, air/water)
              (wt_refract (- 0.0 light_dir_x) (- 0.0 light_dir_y) (- 0.0 light_dir_z) bb nyy aa
                          (/ IOR_AIR IOR_WATER))
              (define ryx rfx) (define ryy rfy) (define ryz rfz)
              ;; oldPos = project((vx,0,vy), refractedLight)
              (wt_caustic_project vvx 0.0 vvz rlx rly rlz rlx rly rlz)
              (define opx pjx) (define opy pjy) (define opz pjz)
              ;; newPos = project((vx,h,vy), ray, refractedLight)
              (wt_caustic_project vvx hk vvz ryx ryy ryz rlx rly rlz)
              (define npx pjx) (define npy pjy) (define npz pjz)
              ;; gl_Position = (0.75*(newPos.xz + refr.xz/refr.y), 0, 1)
              (define ndx (* 0.75 (+ npx (/ rlx rly))))
              (define ndy (* 0.75 (+ npz (/ rlz rly))))
              (define ssx (* (+ (* ndx 0.5) 0.5) (* 1.0 CAUSTIC_N)))
              (define ssy (* (- 1.0 (+ (* ndy 0.5) 0.5)) (* 1.0 CAUSTIC_N)))
              (if (== vi 0)
                  {
                    (set! c0sx ssx) (set! c0sy ssy)
                    (set! c0ox opx) (set! c0oy opy) (set! c0oz opz)
                    (set! c0nx npx) (set! c0ny npy) (set! c0nz npz)
                  }
                  (if (== vi 1)
                      {
                        (set! c1sx ssx) (set! c1sy ssy)
                        (set! c1ox opx) (set! c1oy opy) (set! c1oz opz)
                        (set! c1nx npx) (set! c1ny npy) (set! c1nz npz)
                      }
                      {
                        (set! c2sx ssx) (set! c2sy ssy)
                        (set! c2ox opx) (set! c2oy opy) (set! c2oz opz)
                        (set! c2nx npx) (set! c2ny npy) (set! c2nz npz)
                      }))
              (set! vi (+ vi 1))
            })
          ;; old/new 面积比（光强聚焦度）
          (define o1x (- c1ox c0ox)) (define o1y (- c1oy c0oy)) (define o1z (- c1oz c0oz))
          (define o2x (- c2ox c0ox)) (define o2y (- c2oy c0oy)) (define o2z (- c2oz c0oz))
          (define n1x (- c1nx c0nx)) (define n1y (- c1ny c0ny)) (define n1z (- c1nz c0nz))
          (define n2x (- c2nx c0nx)) (define n2y (- c2ny c0ny)) (define n2z (- c2nz c0nz))
          ;; |cross|
          (define ocx (- (* o1y o2z) (* o1z o2y)))
          (define ocy (- (* o1z o2x) (* o1x o2z)))
          (define ocz (- (* o1x o2y) (* o1y o2x)))
          (define oldArea (Math.sqrt (+ (+ (* ocx ocx) (* ocy ocy)) (* ocz ocz))))
          (define cx2 (- (* n1y n2z) (* n1z n2y)))
          (define cy2 (- (* n1z n2x) (* n1x n2z)))
          (define cz2 (- (* n1x n2y) (* n1y n2x)))
          (define newArea (Math.sqrt (+ (+ (* cx2 cx2) (* cy2 cy2)) (* cz2 cz2))))
          (define ratio (if (< newArea 0.000000001) 0.0 (/ oldArea newArea)))
          (set! c0rt ratio) (set! c1rt ratio) (set! c2rt ratio)

          ;; 包围盒光栅化（无深度、无剔除）
          (define minx (fmin2 c0sx (fmin2 c1sx c2sx)))
          (define maxx (fmax2 c0sx (fmax2 c1sx c2sx)))
          (define miny (fmin2 c0sy (fmin2 c1sy c2sy)))
          (define maxy (fmax2 c0sy (fmax2 c1sy c2sy)))
          (define dopass 1)
          (if (< maxx 0.0) (set! dopass 0) #f)
          (if (>= minx (* 1.0 CAUSTIC_N)) (set! dopass 0) #f)
          (if (< maxy 0.0) (set! dopass 0) #f)
          (if (>= miny (* 1.0 CAUSTIC_N)) (set! dopass 0) #f)
          (define Ax c0sx) (define Ay c0sy)
          (define Bx c1sx) (define By c1sy)
          (define Cx c2sx) (define Cy c2sy)
          (define e0 (+ (* (- By Cy) (- Ax Cx)) (* (- Cx Bx) (- Ay Cy))))
          (define crc 1.0)
          (if (< (Math.abs e0) 0.000001)
              (set! dopass 0)
              (set! crc (/ 1.0 e0)))
          (define cx0 0) (define cx1 0) (define cy0 0) (define cy1 0)
          (if (== dopass 1)
              {
                (set! cx0 (flr minx))
                (if (< cx0 0) (set! cx0 0) #f)
                (set! cx1 (clr maxx))
                (if (> cx1 (- CAUSTIC_N 1)) (set! cx1 (- CAUSTIC_N 1)) #f)
                (set! cy0 (flr miny))
                (if (< cy0 0) (set! cy0 0) #f)
                (set! cy1 (clr maxy))
                (if (> cy1 (- CAUSTIC_N 1)) (set! cy1 (- CAUSTIC_N 1)) #f)
              }
              #f)
          (if (== dopass 1)
              {
                (define w0x (- By Cy)) (define w0y (- Cx Bx))
                (define w1x (- Cy Ay)) (define w1y (- Ax Cx))
                (define py2 cy0)
                (while (<= py2 cy1)
                  {
                    (define px2 cx0)
                    (while (<= px2 cx1)
                      {
                        (define ppx (+ (* 1.0 px2) 0.5))
                        (define ppy (+ (* 1.0 py2) 0.5))
                        (define b0 (* (+ (* w0x (- ppx Cx)) (* w0y (- ppy Cy))) crc))
                        (define b1 (* (+ (* w1x (- ppx Cx)) (* w1y (- ppy Cy))) crc))
                        (define b2 (- 1.0 (+ b0 b1)))
                        (define dok 1)
                        (if (< b0 0.0) (set! dok 0) #f)
                        (if (< b1 0.0) (set! dok 0) #f)
                        (if (< b2 0.0) (set! dok 0) #f)
                        (if (== dok 1)
                            {
                              (define npx3 (+ (+ (* b0 c0nx) (* b1 c1nx)) (* b2 c2nx)))
                              (define npy3 (+ (+ (* b0 c0ny) (* b1 c1ny)) (* b2 c2ny)))
                              (define npz3 (+ (+ (* b0 c0nz) (* b1 c1nz)) (* b2 c2nz)))
                              (define ratio2 (+ (+ (* b0 c0rt) (* b1 c1rt)) (* b2 c2rt)))
                              ;; 圆球挡光（blob shadow）
                              (define ddx (* (- ball_cx npx3) (/ 1.0 ball_radius)))
                              (define ddy (* (- ball_cy npy3) (/ 1.0 ball_radius)))
                              (define ddz (* (- ball_cz npz3) (/ 1.0 ball_radius)))
                              (define axx (- (* ddy rlz) (* ddz rly)))
                              (define ayy (- (* ddz rlx) (* ddx rlz)))
                              (define azz (- (* ddx rly) (* ddy rlx)))
                              (define shv (+ (+ (* axx axx) (* ayy ayy)) (* azz azz)))
                              (define distv (+ (+ (* ddx (- 0.0 rlx)) (* ddy (- 0.0 rly)))
                                                (* ddz (- 0.0 rlz))))
                              (set! shv (+ 1.0 (/ (- shv 1.0) (+ 0.05 (* distv 0.025)))))
                              (set! shv (clampf (/ 1.0 (+ 1.0 (Math.exp (- 0.0 shv)))) 0.0 1.0))
                              (set! shv (+ 1.0 (* (- shv 1.0) (clampf (* distv 2.0) 0.0 1.0))))
                              ;; 池沿阴影（rim）
                              (wt_intersect_cube npx3 npy3 npz3 (- 0.0 rlx) (- 0.0 rly) (- 0.0 rlz)
                                                 -1.0 (- 0.0 POOL_HEIGHT) -1.0 1.0 2.0 1.0)
                              (define t0 tnx) (define t1 tfx)
                              (define rimA (/ (- 0.0 200.0) (+ 1.0 (* 10.0 (- t1 t0)))))
                              (define rimB (- (- npy3 (* rly t1)) (/ 2.0 12.0)))
                              (define rimv (/ 1.0 (+ 1.0 (Math.exp (* rimA rimB)))))
                              (define cidx (+ (* py2 CAUSTIC_N) px2))
                              (set_item! caustR cidx (* (* ratio2 0.2) rimv))
                              (set_item! caustG cidx shv)
                            }
                            #f)
                        (set! px2 (+ px2 1))
                      })
                    (set! py2 (+ py2 1))
                  })
              }
              #f)
          (set! t (+ t 1))
        })
    }))

;; ============================================================================
;; 仿真（wt_step_sim / wt_update_normals / wt_volume_in_sphere / wt_move_sphere）
;; ============================================================================

;; 水面双页乒乓：updateShader（info.g += (avg-h)*2; info.g *= 0.995; info.r += info.g）
(define wt_step_sim
  (lambda ()
    {
      (define src (if (== water_cur 0) water0 water1))
      (define dst (if (== water_cur 0) water1 water0))
      (define nx GRID_N)
      (define y 0)
      (while (< y nx)
        {
          (define x 0)
          (while (< x nx)
            {
              (define i (* (+ (* y nx) x) 4))
              (define lt (if (> x 0) (- i 4) i))
              (define rt (if (< x (- nx 1)) (+ i 4) i))
              (define up (if (> y 0) (- i (* nx 4)) i))
              (define dn (if (< y (- nx 1)) (+ i (* nx 4)) i))
              (define hh (get_item src i))
              (define vv (get_item src (+ i 1)))
              (define avg (* (+ (+ (get_item src lt) (get_item src rt))
                                (+ (get_item src up) (get_item src dn))) 0.25))
              (set! vv (+ vv (* (- avg hh) 2.0)))
              (set! vv (* vv 0.995))
              (set! hh (+ hh vv))
              (set_item! dst i hh)
              (set_item! dst (+ i 1) vv)
              (set_item! dst (+ i 2) (get_item src (+ i 2)))
              (set_item! dst (+ i 3) (get_item src (+ i 3)))
              (set! x (+ x 1))
            })
          (set! y (+ y 1))
        })
      (set! water_cur (- 1 water_cur))
    }))

;; normalShader：normal = normalize(cross(dy,dx)).xz → (info.b, info.a)
(define wt_update_normals
  (lambda ()
    {
      (define pg (wt_page))
      (define nx GRID_N)
      (define delta (/ 1.0 (* 1.0 GRID_N)))
      (define y 0)
      (while (< y nx)
        {
          (define x 0)
          (while (< x nx)
            {
              (define i (* (+ (* y nx) x) 4))
              (define rt (if (< x (- nx 1)) (+ i 4) i))
              (define dn (if (< y (- nx 1)) (+ i (* nx 4)) i))
              (define hh (get_item pg i))
              (define dhdx (- (get_item pg rt) hh))
              (define dhdy (- (get_item pg dn) hh))
              (v_set_a (* (- 0.0 delta) dhdx) (* delta delta) (* (- 0.0 delta) dhdy))
              (v_norm)
              (set_item! pg (+ i 2) rx)
              (set_item! pg (+ i 3) rz)
              (set! x (+ x 1))
            })
          (set! y (+ y 1))
        })
    }))

;; sphereShader 的体积核：圆球在 (u,v) 处排开的水柱高度
(define wt_volume_in_sphere
  (lambda (ccx ccy ccz uu vv)
    {
      (define tx (- (- (* uu 2.0) 1.0) ccx))
      (define tz (- (- (* vv 2.0) 1.0) ccz))
      (define tt (/ (Math.sqrt (+ (* tx tx) (* tz tz))) ball_radius))
      (define dy (Math.exp (- 0.0 (letcube3 (letcube3 (* tt 1.5))))))
      (define ymin (fmin2 0.0 (- ccy dy)))
      (define ymax (fmin2 (fmax2 0.0 (+ ccy dy)) (+ ymin (* 2.0 dy))))
      (* (- ymax ymin) 0.1)
    }))

;; moveSphere：圆球移动 → 排开/回填水体（读一页写另一页后翻页）
(define wt_move_sphere
  (lambda (ox oy oz nx2 ny2 nz2)
    {
      (define ddx (- ox nx2))
      (define ddy (- oy ny2))
      (define ddz (- oz nz2))
      (if (< (+ (+ (* ddx ddx) (* ddy ddy)) (* ddz ddz)) 0.000000000001)
          #f
          {
            (define src (if (== water_cur 0) water0 water1))
            (define dst (if (== water_cur 0) water1 water0))
            (define nxg GRID_N)
            (define i 0)
            (while (< i CELLS)
              {
                (define uidx (mod i nxg))
                (define vidx (flr (/ (* 1.0 i) (* 1.0 nxg))))
                (define uu (/ uidx (* 1.0 nxg)))
                (define vv (/ vidx (* 1.0 nxg)))
                (define i4 (* i 4))
                (set_item! dst i4 (+ (get_item src i4)
                                     (- (wt_volume_in_sphere ox oy oz uu vv)
                                        (wt_volume_in_sphere nx2 ny2 nz2 uu vv))))
                (set_item! dst (+ i4 1) (get_item src (+ i4 1)))
                (set_item! dst (+ i4 2) (get_item src (+ i4 2)))
                (set_item! dst (+ i4 3) (get_item src (+ i4 3)))
                (set! i (+ i 1))
              })
            (set! water_cur (- 1 water_cur))
          })
    }))

;; ============================================================================
;; 圆球物理 & 场景更新（wt_update）
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
      ;; moveSphere + 两次波场推进 + 法线 + 焦散（同 main.js update 的顺序）
      (wt_move_sphere ball_ox ball_oy ball_oz ball_cx ball_cy ball_cz)
      (set! ball_ox ball_cx) (set! ball_oy ball_cy) (set! ball_oz ball_cz)
      (wt_step_sim)
      (wt_step_sim)
      (wt_update_normals)
      (if (== PASS_CAUSTICS 1) (wt_update_caustics) #f)
    }))

;; ============================================================================
;; 一帧（wt_clear_frame + 三个 pass），并推屏
;; ============================================================================

(define wt_scene_render
  (lambda ()
    {
      (wt_clear_frame)
      (if (== PASS_CUBE 1) (wt_render_cube) #f)
      (if (== PASS_WATER 1) (wt_render_water) #f)
      (if (== PASS_SPHERE 1) (wt_render_sphere) #f)
    }))

;; 帧缓冲 RGB565 → 屏幕（等价 C 的 gfx_blit_rgb565 + gfx_refresh）
(define wt_blit
  (lambda ()
    {
      (define i 0)
      (while (< i NPIX)
        {
          (define v (get_item frame i))
          (set! i (+ i 1))
          (Gfx.set_pixel (mod i SW) (flr (/ (* 1.0 i) (* 1.0 SW)))
                         (* 255.0 (/ (flr (/ v 2048.0)) 31.0))
                         (* 255.0 (/ (flr (/ (mod v 2048.0) 32.0)) 63.0))
                         (* 255.0 (/ (mod v 32.0) 31.0)))
        })
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
;; 交互（wt_start_drag / wt_during_drag，与 C 逐句一致）
;; ============================================================================

(define touch_prev 0)                 ; 上一拍触屏电平（电平 → 边沿）
(define down_x 0) (define down_y 0)

;; 圆球命中检测（hitTestSphere）：用 wt_get_ray_for_pixel 的射线
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
      (set! water0 (mklist (* CELLS 4) 0.0))
      (set! water1 (mklist (* CELLS 4) 0.0))
      (set! caustR (mklist (* CAUSTIC_N CAUSTIC_N) 0.0))
      (set! caustG (mklist (* CAUSTIC_N CAUSTIC_N) 0.0))
      (set! frame (mklist NPIX 0))
      (set! zbuf (mklist NPIX 65535))
      (set! wm (cons 0 (quote ())))    (pop wm)
      (set! wtris (cons 0 (quote ()))) (pop wtris)
      (set! cv (cons 0 (quote ())))    (pop cv)
      (set! ctri (cons 0 (quote ())))  (pop ctri)
      (set! sph (cons 0 (quote ())))   (pop sph)
      (set! stri (cons 0 (quote ())))  (pop stri)
      (set! matMV (m4make)) (set! matP (m4make)) (set! matMVP (m4make))
      (set! matTmp (m4make)) (set! matInv (m4make))
      (wt_build_meshes)

      ;; 原版默认状态
      (set! angleX -25.0)
      (set! angleY -200.5)
      (v_set_a 2.0 2.0 -1.0)
      (v_norm)
      (set! light_dir_x rx) (set! light_dir_y ry) (set! light_dir_z rz)
      (set! ball_cx -0.4) (set! ball_cy -0.75) (set! ball_cz 0.2)
      (set! ball_ox -0.4) (set! ball_oy -0.75) (set! ball_oz 0.2)
      (set! ball_vx 0.0) (set! ball_vy 0.0) (set! ball_vz 0.0)
      (set! ball_gravity_x 0.0) (set! ball_gravity_y GRAVITY_Y) (set! ball_gravity_z 0.0)
      (set! ball_radius BALL_R)
      (set! use_sphere_physics 1)
      (set! wt_paused 0)
      (set! wt_mode MODE_NONE)
      (set! mouse_down 0)
      (set! pressed_btn -1)
      (set! view_dirty 0)
      (set! water_cur 0)
      (set! ready 1)
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
                (if (== mouse_down 1) (wt_during_drag tx ty) #f))
          }
          (if (== touch_prev 1)
              {
                ;; 松手沿：控件动作（按下点与松手点须命中同一控件）
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

      ;; ③ 物理 + 仿真 + 渲染（暂停时冻结，但视角/圆球拖动仍重画）
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
              }
              #f))

      ;; ④ 拷到帧缓冲 + 控件 + 标题 + 推屏
      (wt_blit)
      (wt_draw_ui)
      (Gfx.set_font 1)
      (Gfx.draw_text "水池" 4 2 208 232 255)
      (Gfx.draw_text "拖动旋转视角·拖动圆球" 56 2 190 205 225)
      (Gfx.refresh)
    }))

;; ---- 启动 ------------------------------------------------------------------
(EX.install)
(wt_alloc)
;; 设备侧首帧是"先 wt_update(dt) 再渲染"；这里同样先推一拍（dt 取 1ms），
;; 使首帧的波场/法线/焦散与 C 版一致（否则焦散纹理还是零，水面着色会有差异）
(wt_update 0.001)
(wt_compose_camera)
(wt_scene_render)
(wt_blit)
(wt_draw_ui)
(Gfx.set_font 1)
(Gfx.draw_text "水池" 4 2 208 232 255)
(Gfx.draw_text "拖动旋转视角·拖动圆球" 56 2 190 205 225)
(Gfx.refresh)
(set! tick_timer (System.set_interval TICK_MS tick))
