;@id=waterpool
;@title=水池1
;@version=2026-11-17
;; ============================================================================
;; 水池（Water Pool）—— 灵机应用程序（`infer/ui_water.c` 的逐函数直译，**降规模版**）
;;
;; 本文件把 ui_water.c（WebGL Water 的 C 软渲染移植）的
;; 相机/网格/光栅化/着色/焦散/仿真/物理逐函数翻成灵机 Scheme：几何、绘制次序、
;; 着色公式、常量与 C 版一一对应，函数名同 C 版。
;;
;; **本版与原版的区别只有"问题规模"**：原版单帧约 2.2×10⁸ 条 VM 指令，宿主时间片
;; 312500 条/帧 ⇒ 一帧要 ~700 个宿主动作，浏览器 WASM 里 ~12 秒/帧（0.08fps）。
;; 本版把几个"规模维"一起缩小（分辨率、网格、波场、焦散频率），默认档单帧
;; **~1.34×10⁶ 条指令 → 真 WASM 实测 chromium 12.0 / firefox 10.8 fps**；
;; 「画质」按钮还能切 5 档（15 / 12 / 10 / 7.4 / 6 fps）。算法、公式、pass 结构、
;; 交互全部没变；瓷砖缝（亚像素深色线，低分辨率下会走样成黑点）已按用户要求去掉。
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
;;   本版新增的小工具：wt_wat_near（逐像素"是否在水下"判据用最近邻）、
;;                     wt_flip（水面双页翻页 + 页面缓存）、wt_blit（按块放大上屏）、
;;                     wt_build_blit / wt_set_quality（画质档位切换）
;;
;; 交互（与设备侧一致，全部触屏化）：底部控件栏
;;   「返回 / 暂停 / 重力 / 光源 / 画质」、拖动空白处旋转视角、拖动圆球移动圆球。
;;   触屏坐标仍按**屏幕** SW×SH 解算（与内部渲染分辨率无关）。
;;
;; ---------------------------------------------------------------------------
;; 画质档位（运行时按「画质」按钮循环）
;; ---------------------------------------------------------------------------
;; 预算从哪来：宿主（ui_scheme_app.c）每个 rAF 帧只喂
;; MB_TIMESLICE_TICKS = 312500 条 IL 指令，60 个 rAF/秒 ⇒ 上限 18.75M 指令/秒。
;; 所以 **fps = 60 / ceil(单帧指令数 / 312500)**，与设备快慢基本无关（只要 312500
;; 条能在 16.7ms 内跑完——WASM 实测 7.2ms，有 2 倍余量）：
;;   单帧 ≤ 1.25M → 4 个 rAF 帧 → 15fps      ≤ 1.56M → 5 帧 → 12fps
;;   单帧 ≤ 1.875M → 6 帧 → 10fps            ≤ 2.19M → 7 帧 → 8.6fps …
;;
;; 档位（只改内部渲染分辨率 RW×RH，都是 4:3；网格/波场/焦散规模不变）
;;   档   RW×RH   像素    相对档0   指令/帧   chromium  firefox   说明
;;   0    36×27    972    1.00×     0.96M     15.0      14.4
;;   1    44×33   1452    1.49×     1.34M     12.0      10.8   ← 默认
;;   2    52×39   2028    2.09×     1.80M     10.0       8.4   （像素数正好翻倍）
;;   3    60×45   2700    2.78×     2.31M      7.4       6.4
;;   4    68×51   3468    3.57×     2.92M      6.0       5.0   （≈1.9 倍线性分辨率）
;;   实测口径：真 WASM + 无头浏览器、按「画质」逐档测 5 秒（chromium 1248 / firefox 1555）。
;;   注：fps = 浏览器 rAF 速率 / ceil(单帧指令数/312500)；firefox 的 rAF 略慢、有时要多
;;   用一个时间片，所以同一档在 firefox 上低一档 —— 默认取档 1 正是为了三引擎都 ≥10fps。
;;
;; 逐项降级（都是"规模 + 抗走样"，不是换算法）：
;;   项                      原版             本版         说明
;;   内部渲染分辨率          320×240          36×27 起 5 档 逐像素追迹成本 ∝ 面积
;;   控件栏盖住的行          照常渲染         不渲染       RSTOP（省 ~11%）
;;   上屏方式                逐像素 set_pixel 按块 fill_rect 一个内部像素 = 一块 6~9 px
;;   瓷砖缝（grout）         深色缝 ×0.25     **去掉**     缝宽 5% 的瓷砖 ≈ 0.3 个内部像素，
;;                                                        亚像素深色线必然走样成"黑点"；
;;                                                        去掉后只剩每块砖 6% 的明暗差
;;   水面细分 WATER_DETAIL   44 (3872 tri)    3 (18)      波高只有几像素，细分不需要
;;   圆球网格 SPH_RING×SEG   20×36 (1368)     3×4 (16)    球在此分辨率下只有 ~3 内部像素
;;   水面高度场 GRID_N       128 (16384)      8 (64)      波场格数
;;   焦散纹理 CAUSTIC_N      256 (65536)      16 (256)    焦散贴图
;;   焦散 pass 频率          每帧             每 12 拍    CAUSTIC_EVERY（焦散变化远慢于画面）
;;   水面倾斜重采样 PEAKED   5 次             1 次        只少一点折射抖动
;;   逐像素双线性采样        双线性           最近邻      仅"是否在水下"判据与顶点高度
;;   焦散纹理采样            双线性           最近邻      16² 的贴图本来就看不出插值差别
;;
;; 另外几处**不改画面**的省指令改写：
;;   · 折射平行光 rl_* 只依赖 light_dir → 每帧算一次（原版每个着色函数各算一次）；
;;   · wt_water_frag / wt_get_wall_color / wt_get_sphere_color 的向量运算就地展开成局部量
;;     （原来是 v_set_a / v_set_b / v_len / v_dot 这类"全局寄存器"函数逐像素调用）；
;;   · 池壁/圆球片段复用 getWallColor / getSphereColor 内部已算出的水下判据高度；
;;   · wt_fill / wt_blit 的局部量在函数体首部声明一次，循环里只 set!；
;;   · 池壁 pass 是第一个 pass（zbuf 全满）→ `wt_fill` 的 use_depth=2 只写不测。
;;
;; 已知代价（降规模的固有后果，不是 bug）：
;;   · 画面是 36×27~68×51 放大上屏（档 2 是 6~7 倍）→ 块状明显；圆球只有几个像素大。
;;   · 瓷砖只剩低频明暗棋盘格（深色缝已去掉，见上表）。
;;   · 焦散每 12 拍重建一次，圆球快速移动时焦散阴影最多滞后 ~0.4 秒。
;;   · 档位越高越清晰越慢；fps 只按 312500 指令/帧整档跳，所以"再快一点"往往要靠
;;     降一档而不是微调。默认档 2（≈10fps）与档 1（≈12fps）是一对常见取舍。
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

;; 网格分辨率：**降规模**（原版 GRID 128 / 焦散 256 / 水面细分 44 / 球 20×36）。
;;   单帧成本 ≈ 像素数 × 逐像素着色 + 三角形数 × 装配 + GRID² + CAUSTIC²，
;;   所以这几个维要一起缩（只缩一个维度几乎不省）。见文件头「画质档位 / 逐项降级」。
(define GRID_N 8)                       ; WT_GRID_N 128 → 8（水面高度场 64 格）
(define CAUSTIC_N 16)                   ; WT_CAUSTIC_N 256 → 16（焦散 16×16 = 256 像素）
(define WATER_DETAIL 3)                 ; WT_WATER_DETAIL 44 → 3（水面 16 顶点 / 18 三角形）
(define WATER_DETAIL_P1 (+ WATER_DETAIL 1))
(define WATER_VERTS (* WATER_DETAIL_P1 WATER_DETAIL_P1))
(define WATER_TRIS (* (* WATER_DETAIL WATER_DETAIL) 2))
(define CELLS (* GRID_N GRID_N))

(define CUBE_FACES 5)
(define CUBE_VERTS 20)
(define CUBE_TRIS 10)

(define SPH_RING 3)                     ; 原 20（球在此分辨率下只有 ~3 个内部像素）
(define SPH_SEG 4)                      ; 原 36
(define SPH_VN (* SPH_RING (+ SPH_SEG 1)))
(define SPH_TN (* (* (- SPH_RING 1) SPH_SEG) 2))

;; 水面片段的"倾斜重采样"次数（原版 5 次，每次 2 个双线性采样）。
;; 1 次 = 只用该点的波面法线，视觉上只少了极细的折射抖动。
(define PEAKED_N 1)

;; 焦散 pass 的帧步（原版每帧一次）：它只依赖波场，变化比画面滞得多；
;; 每 CAUSTIC_EVERY 拍重算一次，平均成本降为 1/CAUSTIC_EVERY。
(define CAUSTIC_EVERY 12)

(define BALL_R 0.25)                    ; WT_RADIUS
(define GRAVITY_Y -4.0)                 ; WT_GRAVITY

;; 画布：SW×SH 是**屏幕**（控件栏/触屏坐标用）；渲染在 RW×RH 的**内部缓冲**上，
;; 最后按块放大上屏（wt_blit）。这是降规模的第一旋钮：单帧逐像素追迹的成本
;; ∝ RW·RH（原版 320×240 = 76800 像素）。
;; RW:RH 必须与 SW:SH 同比例（都是 4:3），否则投影纵横比会歪。
;;
;; **画质档位（运行时按「画质」按钮循环切换）**：只改内部渲染分辨率 RW×RH，
;; 网格/波场/焦散的规模不变。RW 都取 4:3（RH = 3/4·RW），逐像素成本 ∝ 像素数。
;; 实测（WASM 浏览器；fps = 60/ceil(每帧指令数/312500)，按时间片整档跳）：
;;   档    RW×RH    像素   相对档0   指令/帧   fps
;;   0     36×27     972   1.00×     0.96M     15
;;   1     44×33    1452   1.49×     1.36M     12
;;   2     52×39    2028   2.09×     1.80M     10   ← 默认（像素数正好翻倍）
;;   3     60×45    2700   2.78×     2.31M      7.5
;;   4     68×51    3468   3.57×     2.90M      5.5  （约 2 倍线性分辨率）
(define SW (Gfx.width))
(define SH (Gfx.height))
(define QRW0 36)                        ; 档 0 的 RW
(define QSTEP 8)                        ; 每档 RW 的增量 → 36/44/52/60/68
(define QN 5)                           ; 档数
(define qlevel 1)                       ; 当前档位（默认档 1 = 44×33，三引擎都稳 ≥11fps）
(define RW (+ QRW0 (* QSTEP qlevel)))
(define RH (/ (* 3 RW) 4))            ; 36/48/60/72 的 3/4 都是整数
;; frame/zbuf 一次按最大档分配，切档只改索引步长，不用重新分配
(define MAXRW (+ QRW0 (* QSTEP (- QN 1))))
;; 注意：MAXRW×MAXRH 只是 frame/zbuf 的容量，实际渲染用当前档的 RW×RH
(define MAXRH (/ (* 3 MAXRW) 4))
(define NPIX (* MAXRW MAXRH))
(define TICK_MS 33)


;; 底部控件栏（几何同 ui_water.c）
(define BAR_H 34)
(define BAR_Y (- SH BAR_H))
(define BTN_N 5)
(define BTN_GAP 4)
;; 内部缓冲里**真正需要渲染**的行数：底部 BAR_H 像素被控件栏整块盖住，那几行既不用
;; 着色也不用上屏。RSTOP = ceil(RH·(SH−BAR_H)/SH)（27×206/240 → 24，省掉 ~11%）。
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
(define BTN_QUALITY 4)

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
(define caustic_phase 0)                ; 焦散 pass 的降频计数（见 CAUSTIC_EVERY）

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

;; 内部缓冲里**真正需要渲染**的行数（底部 BAR_H 像素被控件栏整块盖住）——
;;   放在 flr 之后定义（顶层 define 按顺序求值）。
(define RSTOP (flr (+ (/ (* 1.0 (* RH (- SH BAR_H))) (* 1.0 SH)) 0.9999)))

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
;; 伴随式求逆（与 ui_water.c 的 m4inv 逐项一致，左结合二元链展开）
;;   注意：旧版把 r1/r2/r3/r5… 的交替符号写反了（逆矩阵错）→ 四角射线全错 →
;;   (wt_hit_test_sphere) 永远失败 → **拖圆球变成拖视角**（画面本身不受影响，因为
;;   水面着色用的是 eye 而不是角射线）。本版由 ui_water.c 的 m4inv 机械重生成，
;;   修好后 r00=(0.214963,-0.656717,0.722851)、r11=(-0.737358,-0.038853,0.674384)
;;   与 C 参照台一致。
(define m4inv
  (lambda (s o) {
    (define m0 (get_item s 0)) (define m1 (get_item s 1)) (define m2 (get_item s 2)) (define m3 (get_item s 3))
    (define m4 (get_item s 4)) (define m5 (get_item s 5)) (define m6 (get_item s 6)) (define m7 (get_item s 7))
    (define m8 (get_item s 8)) (define m9 (get_item s 9)) (define m10 (get_item s 10)) (define m11 (get_item s 11))
    (define m12 (get_item s 12)) (define m13 (get_item s 13)) (define m14 (get_item s 14)) (define m15 (get_item s 15))
    (define r0 (- (+ (+ (- (- (* (* m5 m10) m15) (* (* m5 m11) m14)) (* (* m9 m6) m15)) (* (* m9 m7) m14)) (* (* m13 m6) m11)) (* (* m13 m7) m10)))
    (define r1 (+ (- (- (+ (+ (- 0.0 (* (* m1 m10) m15)) (* (* m1 m11) m14)) (* (* m9 m2) m15)) (* (* m9 m3) m14)) (* (* m13 m2) m11)) (* (* m13 m3) m10)))
    (define r2 (- (+ (+ (- (- (* (* m1 m6) m15) (* (* m1 m7) m14)) (* (* m5 m2) m15)) (* (* m5 m3) m14)) (* (* m13 m2) m7)) (* (* m13 m3) m6)))
    (define r3 (+ (- (- (+ (+ (- 0.0 (* (* m1 m6) m11)) (* (* m1 m7) m10)) (* (* m5 m2) m11)) (* (* m5 m3) m10)) (* (* m9 m2) m7)) (* (* m9 m3) m6)))
    (define r4 (+ (- (- (+ (+ (- 0.0 (* (* m4 m10) m15)) (* (* m4 m11) m14)) (* (* m8 m6) m15)) (* (* m8 m7) m14)) (* (* m12 m6) m11)) (* (* m12 m7) m10)))
    (define r5 (- (+ (+ (- (- (* (* m0 m10) m15) (* (* m0 m11) m14)) (* (* m8 m2) m15)) (* (* m8 m3) m14)) (* (* m12 m2) m11)) (* (* m12 m3) m10)))
    (define r6 (+ (- (- (+ (+ (- 0.0 (* (* m0 m6) m15)) (* (* m0 m7) m14)) (* (* m4 m2) m15)) (* (* m4 m3) m14)) (* (* m12 m2) m7)) (* (* m12 m3) m6)))
    (define r7 (- (+ (+ (- (- (* (* m0 m6) m11) (* (* m0 m7) m10)) (* (* m4 m2) m11)) (* (* m4 m3) m10)) (* (* m8 m2) m7)) (* (* m8 m3) m6)))
    (define r8 (- (+ (+ (- (- (* (* m4 m9) m15) (* (* m4 m11) m13)) (* (* m8 m5) m15)) (* (* m8 m7) m13)) (* (* m12 m5) m11)) (* (* m12 m7) m9)))
    (define r9 (+ (- (- (+ (+ (- 0.0 (* (* m0 m9) m15)) (* (* m0 m11) m13)) (* (* m8 m1) m15)) (* (* m8 m3) m13)) (* (* m12 m1) m11)) (* (* m12 m3) m9)))
    (define r10 (- (+ (+ (- (- (* (* m0 m5) m15) (* (* m0 m7) m13)) (* (* m4 m1) m15)) (* (* m4 m3) m13)) (* (* m12 m1) m7)) (* (* m12 m3) m5)))
    (define r11 (+ (- (- (+ (+ (- 0.0 (* (* m0 m5) m11)) (* (* m0 m7) m9)) (* (* m4 m1) m11)) (* (* m4 m3) m9)) (* (* m8 m1) m7)) (* (* m8 m3) m5)))
    (define r12 (+ (- (- (+ (+ (- 0.0 (* (* m4 m9) m14)) (* (* m4 m10) m13)) (* (* m8 m5) m14)) (* (* m8 m6) m13)) (* (* m12 m5) m10)) (* (* m12 m6) m9)))
    (define r13 (- (+ (+ (- (- (* (* m0 m9) m14) (* (* m0 m10) m13)) (* (* m8 m1) m14)) (* (* m8 m2) m13)) (* (* m12 m1) m10)) (* (* m12 m2) m9)))
    (define r14 (+ (- (- (+ (+ (- 0.0 (* (* m0 m5) m14)) (* (* m0 m6) m13)) (* (* m4 m1) m14)) (* (* m4 m2) m13)) (* (* m12 m1) m6)) (* (* m12 m2) m5)))
    (define r15 (- (+ (+ (- (- (* (* m0 m5) m10) (* (* m0 m6) m9)) (* (* m4 m1) m10)) (* (* m4 m2) m9)) (* (* m8 m1) m6)) (* (* m8 m2) m5)))
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
      (m4persp matP 45.0 (/ (* 1.0 RW) (* 1.0 RH)) 0.01 100.0)
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
            (set! isx (* (+ (* (* mx iw2) 0.5) 0.5) (* 1.0 RW)))
            (set! isy (* (- 1.0 (+ (* (* my iw2) 0.5) 0.5)) (* 1.0 RH)))
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
      ;; 热路径：**所有局部量在函数体首部声明一次**，循环里只 set!。
      ;; 原来的写法在循环体内 define，每轮都会新建一层闭包环境（逐像素一次），
      ;; 实测是本 App 里最大的一笔开销。数学表达式一字未改。
      (define tri_ok 1)
      (define Ax 0.0) (define Ay 0.0) (define Bx 0.0) (define By 0.0)
      (define Cx 0.0) (define Cy 0.0) (define e0 0.0) (define rc 0.0)
      (define minx 0.0) (define maxx 0.0) (define miny 0.0) (define maxy 0.0)
      (define x0 0.0) (define x1 0.0) (define y0 0.0) (define y1 0.0)
      (define w0x 0.0) (define w0y 0.0) (define w1x 0.0) (define w1y 0.0)
      (define iwv0 0.0) (define iwv1 0.0) (define iwv2 0.0)
      (define py 0.0) (define pxc 0.0) (define pxd 0.0) (define pyd 0.0)
      (define b0 0.0) (define b1 0.0) (define b2 0.0)
      (define dodraw 1) (define idx 0) (define d16 0.0)
      (define denom 0.0) (define ndcZ 0.0) (define rden 1.0)
      (define dtest 1) (define dwrite 1)
      (define cr 0.0) (define cg 0.0) (define cb 0.0)

      (if (== v0w 0.0) (set! tri_ok 0) #f)
      (if (== v1w 0.0) (set! tri_ok 0) #f)
      (if (== v2w 0.0) (set! tri_ok 0) #f)
      (set! Ax v0sx) (set! Ay v0sy)
      (set! Bx v1sx) (set! By v1sy)
      (set! Cx v2sx) (set! Cy v2sy)
      (set! e0 (+ (* (- By Cy) (- Ax Cx)) (* (- Cx Bx) (- Ay Cy))))
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
            (if (>= minx (* 1.0 RW)) (set! tri_ok 0) #f)
            (if (< maxy 0.0) (set! tri_ok 0) #f)
            (if (>= miny (* 1.0 RSTOP)) (set! tri_ok 0) #f)
            (if (== tri_ok 1)
                {
                  (set! x0 (flr minx))
                  (if (< x0 0.0) (set! x0 0.0) #f)
                  (set! x1 (clr maxx))
                  (if (> x1 (- RW 1)) (set! x1 (- RW 1)) #f)
                  (set! y0 (flr miny))
                  (if (< y0 0.0) (set! y0 0.0) #f)
                  (set! y1 (clr maxy))
                  (if (> y1 (- RSTOP 1)) (set! y1 (- RSTOP 1)) #f)
                }
                #f)
          }
          #f)
      ;; use_depth: 0=不测不写  1=测+写  2=只写（池壁 pass：此时 zbuf 全满，不必测）
      (set! dwrite (if (>= use_depth 1) 1 0))
      (set! dtest (if (== use_depth 1) 1 0))
      (if (== tri_ok 1)
          {
            (set! w0x (- By Cy)) (set! w0y (- Cx Bx))
            (set! w1x (- Cy Ay)) (set! w1y (- Ax Cx))
            (set! iwv0 (/ 1.0 v0w))
            (set! iwv1 (/ 1.0 v1w))
            (set! iwv2 (/ 1.0 v2w))
            (set! py y0)
            (while (<= py y1)
              {
                (set! pxc x0)
                (set! pyd (+ py 0.5))
                (while (<= pxc x1)
                  {
                    (set! pxd (+ pxc 0.5))
                    (set! b0 (* (+ (* w0x (- pxd Cx)) (* w0y (- pyd Cy))) rc))
                    (set! b1 (* (+ (* w1x (- pxd Cx)) (* w1y (- pyd Cy))) rc))
                    (set! b2 (- 1.0 (+ b0 b1)))
                    (set! dodraw 1)
                    (if (< b0 0.0) (set! dodraw 0) #f)
                    (if (< b1 0.0) (set! dodraw 0) #f)
                    (if (< b2 0.0) (set! dodraw 0) #f)
                    (set! idx (+ (* py RW) pxc))
                    (if (== dodraw 1)
                        {
                          (set! denom (+ (+ (* b0 iwv0) (* b1 iwv1)) (* b2 iwv2)))
                          (set! ndcZ (/ (+ (+ (* (* b0 iwv0) v0nz) (* (* b1 iwv1) v1nz))
                                          (* (* b2 iwv2) v2nz)) denom))
                          (set! d16 (flr (+ (* ndcZ 32767.5) 32767.5)))
                          (if (== dtest 1)
                              (if (>= d16 (get_item zbuf idx)) (set! dodraw 0) #f)
                              #f)
                          (if (== dodraw 1)
                              {
                                ;; 透视校正插值：5 个属性共用一个 1/denom（原来各除一次，
                                ;; 每次除法都是一条 IL 指令；换成一次倒数 + 5 次乘）
                                (set! rden (/ 1.0 denom))
                                (set! rw_x (* (+ (+ (* (* b0 iwv0) v0wx) (* (* b1 iwv1) v1wx))
                                                 (* (* b2 iwv2) v2wx)) rden))
                                (set! rw_y (* (+ (+ (* (* b0 iwv0) v0wy) (* (* b1 iwv1) v1wy))
                                                 (* (* b2 iwv2) v2wy)) rden))
                                (set! rw_z (* (+ (+ (* (* b0 iwv0) v0wz) (* (* b1 iwv1) v1wz))
                                                 (* (* b2 iwv2) v2wz)) rden))
                                (set! ru (* (+ (+ (* (* b0 iwv0) v0u) (* (* b1 iwv1) v1u))
                                               (* (* b2 iwv2) v2u)) rden))
                                (set! rv (* (+ (+ (* (* b0 iwv0) v0v) (* (* b1 iwv1) v1v))
                                               (* (* b2 iwv2) v2v)) rden))
                                (wt_frag_dispatch fmode)
                                (set! cr (clampf fc_r 0.0 1.0))
                                (set! cg (clampf fc_g 0.0 1.0))
                                (set! cb (clampf fc_b 0.0 1.0))
                                (set_item! frame idx
                                  (+ (+ (* (flr (+ (* cr 31.0) 0.5)) 2048)
                                        (* (flr (+ (* cg 63.0) 0.5)) 32))
                                     (flr (+ (* cb 31.0) 0.5))))
                                (if (== dwrite 1) (set_item! zbuf idx d16) #f)
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
      (while (< i (* RW RSTOP))
        {
          (set_item! frame i 0)
          (set_item! zbuf i 65535)
          (set! i (+ i 1))
        })
    }))

;; ============================================================================
;; 采样与几何工具（tiles / sky / 水面高度 / 交点 / 折射反射）
;; ============================================================================

;; 水面页（本拍读的那一页）。`wpg` 把当前页缓存成一个全局，省掉逐采样一次的
;; `water_cur` 判据 + lambda 调用（wt_wat_at 每像素要调 4 次）。翻页时由 wt_flip 同步。
(define wpg water0)
(define wt_page
  (lambda () wpg))
(define wt_flip
  (lambda ()
    {
      (set! water_cur (- 1 water_cur))
      (if (== water_cur 0) (set! wpg water0) (set! wpg water1))
    }))

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
;; 最近邻取样（只用于逐像素的"这一点是否在水下"判据）。原版用双线性（4 抽头）；
;; 这个判据只是拿高度比大小，最近邻完全够用，而每像素要叫 1~3 次。
(define wt_wat_near
  (lambda (u v)
    {
      (define ax (flr (* u (* 1.0 GRID_N))))
      (define ay (flr (* v (* 1.0 GRID_N))))
      ;; 调用点的 u,v 恒在 [0,1]（世界 x,z ∈ [-1,1] → 0.5p+0.5），只需夹上界
      (if (> ax (- GRID_N 1)) (set! ax (- GRID_N 1)) #f)
      (if (> ay (- GRID_N 1)) (set! ay (- GRID_N 1)) #f)
      (get_item wpg (* (+ (* ay GRID_N) ax) 4))
    }))

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

;; 焦散纹理采样（**最近邻** + CLAMP；原版双线性要 8 个取值、十几个乘加，
;;   而焦散纹理本身只有 CAUSTIC_N² 个格点，最近邻的观感差异看不出来）
(define caust_r 0.0) (define caust_g 0.0)
(define wt_caustic_sample
  (lambda (u v)
    {
      (define uu (clampf u 0.0 1.0))
      (define vv (clampf v 0.0 1.0))
      (define xa (flr (* uu (* 1.0 CAUSTIC_N))))
      (define ya (flr (* vv (* 1.0 CAUSTIC_N))))
      (if (< xa 0) (set! xa 0) (if (> xa (- CAUSTIC_N 1)) (set! xa (- CAUSTIC_N 1)) #f))
      (if (< ya 0) (set! ya 0) (if (> ya (- CAUSTIC_N 1)) (set! ya (- CAUSTIC_N 1)) #f))
      (define ci (+ (* ya CAUSTIC_N) xa))
      (set! caust_r (get_item caustR ci))
      (set! caust_g (get_item caustG ci))
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
;;   C 版：fu=floor(u/T), xu=u/T-fu, cu=(int)fmod(fu,2), shade=1-0.06*((cu^cv)&1),
;;          grout = (xu<0.05 || xv<0.05) ? 0.25 : 1
;;
;;   **本版去掉了 grout（瓷砖缝）**：缝宽只有 1 个瓷砖的 5%，在原版 320×240 上约 3 px，
;;   但在 36×27（放大 8~9 倍上屏）下只有 0.3 个内部像素 —— 亚像素的深色线一定会走样成
;;   "散落的黑点"，整片池壁看起来像脏点。去掉后只剩每块砖 6% 的明暗差（shade），
;;   低分辨率下是干净的低频棋盘格。顺带省掉 xu/xv 两次除法 + 两次比较（逐像素）。
(define ti_r 0.0) (define ti_g 0.0) (define ti_b 0.0)
(define cpar 0.0)
(define wt_tiles_sample
  (lambda (tu tv)
    {
      (define TT (/ 1.0 5.0))
      (define fu (flr (/ tu TT)))
      (define fv (flr (/ tv TT)))
      ;; parity = |fmod(floor(fu) + floor(fv), 2)|（等价原版的 (cu^cv)&1：整数的奇偶）
      (set! cpar (Math.abs (mod (+ fu fv) 2.0)))
      (define shade (- 1.0 (* 0.06 cpar)))
      (set! ti_r (+ 0.42 (* 0.20 shade)))
      (set! ti_g (+ 0.50 (* 0.20 shade)))
      (set! ti_b (+ 0.56 (* 0.20 shade)))
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

;; 折射后的平行光（-lightDir 以 (0,1,0) 为法线折射）。
;;   C 版在多个着色函数里各算一次；这里它只依赖 light_dir，**每帧算一次**即可
;;   （光源只在「光源」按钮与初始化时变，两处都会重算）。
(define rl_x 0.0) (define rl_y 0.0) (define rl_z 0.0)
(define wt_prep_light
  (lambda ()
    {
      (wt_refract (- 0.0 light_dir_x) (- 0.0 light_dir_y) (- 0.0 light_dir_z) 0.0 1.0 0.0
                  (/ IOR_AIR IOR_WATER))
      (set! rl_x rfx) (set! rl_y rfy) (set! rl_z rfz)
    }))

;; getSphereColor(point) → sc_r/sc_g/sc_b
(define sc_r 0.0) (define sc_g 0.0) (define sc_b 0.0)
(define sc_hh 0.0)
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
      ;; diffuse = max(0, dot(-refractedLight, sphereNormal)) * 0.5
      ;;   注：原来是 (v_set_a …)/(v_set_b …)/(v_dot) 三个全局寄存器函数调用；
      ;;   全局量取值/赋值比局部量贵 3~6 倍，逐像素调用非常亏，这里就地展开。
      (define diff (fmax2 0.0 (+ (+ (* (- 0.0 rl_x) snx) (* (- 0.0 rl_y) sny))
                                  (* (- 0.0 rl_z) snz))))
      (set! diff (* diff 0.5))
      ;; 水下：乘焦散
      (define hh (wt_wat_near (+ (* px 0.5) 0.5) (+ (* pz 0.5) 0.5)))
      (set! sc_hh hh)
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
(define wc_hh 0.0)                      ;; getWallColor 内部算出的水面高度（水下判据）
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
      ;;   全部就地展开（原本是 v_set_a/v_len/v_dot 三个全局寄存器函数调用 +
      ;;   letsq 套 letsq；这是逐像素热路径里最贵的一段）。
      (set! scale (/ scale (Math.sqrt (+ (+ (* px px) (* py py)) (* pz pz)))))
      (define wbx (- px ball_cx)) (define wby (- py ball_cy)) (define wbz (- pz ball_cz))
      (define wbl (Math.sqrt (+ (+ (* wbx wbx) (* wby wby)) (* wbz wbz))))
      (define wbr (/ wbl ball_radius))
      (set! scale (* scale (- 1.0 (/ 0.9 (* (* wbr wbr) (* wbr wbr))))))
      ;; 折射光（取反；rl_* 每帧算一次，见 wt_prep_light）
      (define rlx (- 0.0 rl_x)) (define rly (- 0.0 rl_y)) (define rlz (- 0.0 rl_z))
      (define diff (fmax2 0.0 (+ (+ (* rlx nX) (* rly nY)) (* rlz nZ))))
      (define hh (wt_wat_near (+ (* px 0.5) 0.5) (+ (* pz 0.5) 0.5)))
      (set! wc_hh hh)          ;; 片段里要用的同一个水下判据（省一次取样）
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
      ;; 同一个判据（同一个点、同样的取样），直接复用 getWallColor 里的结果
      (if (< rw_y wc_hh)
          {
            (set! fc_r (* wc_r (* COL_UNDERWATER_R 1.2)))
            (set! fc_g (* wc_g (* COL_UNDERWATER_G 1.2)))
            (set! fc_b (* wc_b (* COL_UNDERWATER_B 1.2)))
          }
          { (set! fc_r wc_r) (set! fc_g wc_g) (set! fc_b wc_b) })
    }))

;; 水面片段（pass 0 = 水面以上 / 1 = 水面以下）
;;
;; 热路径改写说明：原来逐像素走 wt_reflect / wt_refract / v_set_a / v_set_b /
;; v_dot / v_norm / v_len（全局寄存器 ax/ay/az/rx/ry/rz/rfx/rfy/rfz/rdot/rlen）。
;; 全局量取值与赋值比局部量贵 3~6 倍，逐像素调这些函数是本 App 第二大开销。
;; 这里把同一批公式就地展开成局部量 —— **算式、运算次序、常量完全不变**，
;; 因此着色结果与改写前逐位相同。
(define wt_water_frag
  (lambda (pass)
    {
      (define coordU ru)
      (define coordV rv)
      (define infoB 0.0) (define infoA 0.0)
      (define i5 0)
      (define nx1 0.0) (define ny1 0.0) (define nz1 0.0)
      ;; incoming = normalize(world - eye)
      (define irx 0.0) (define iry 0.0) (define irz 0.0) (define ilen 0.0)
      ;; reflect / refract 就地展开用的中间量
      (define ddot 0.0) (define kk 0.0) (define sc2 0.0) (define eta 0.0)
      (define rrx 0.0) (define rry 0.0) (define rrz 0.0)
      (define rex 0.0) (define rey 0.0) (define rez 0.0)
      (define fresnel 0.0)
      (define ar 0.0) (define ag 0.0) (define ab 0.0)
      (define e2r 0.0) (define e2g 0.0) (define e2b 0.0)
      (define tt 0.0) (define elen 0.0) (define tir 1)
      (define nX 0.0) (define nY 0.0) (define nZ 0.0)

      ;; “peaked”：沿切线方向偏移重采样 PEAKED_N 次（原版 5 次）
      (while (< i5 PEAKED_N)
        {
          (set! infoB (wt_wat_sample coordU coordV 2))
          (set! infoA (wt_wat_sample coordU coordV 3))
          (set! coordU (+ coordU (* infoB 0.005)))
          (set! coordV (+ coordV (* infoA 0.005)))
          (set! i5 (+ i5 1))
        })
      (set! nx1 infoB)
      (set! ny1 (Math.sqrt (fmax2 0.0 (- 1.0 (+ (* infoB infoB) (* infoA infoA))))))
      (set! nz1 infoA)
      ;; incomingRay = normalize(world - eye)
      (set! irx (- rw_x eye_x)) (set! iry (- rw_y eye_y)) (set! irz (- rw_z eye_z))
      (set! ilen (Math.sqrt (+ (+ (* irx irx) (* iry iry)) (* irz irz))))
      (set! irx (/ irx ilen)) (set! iry (/ iry ilen)) (set! irz (/ irz ilen))
      (if (== pass 0)
          {
            (set! nX nx1) (set! nY ny1) (set! nZ nz1)
            ;; ddot = dot(normal, incoming)
            (set! ddot (+ (+ (* nx1 irx) (* ny1 iry)) (* nz1 irz)))
            ;; refl = incoming - 2*normal*ddot
            (set! rrx (- irx (* (* nx1 2.0) ddot)))
            (set! rry (- iry (* (* ny1 2.0) ddot)))
            (set! rrz (- irz (* (* nz1 2.0) ddot)))
            ;; refr = refract(incoming, normal, air/water)
            (set! eta (/ IOR_AIR IOR_WATER))
            (set! kk (- 1.0 (* (* eta eta) (- 1.0 (* ddot ddot)))))
            (if (< kk 0.0)
                { (set! rex 0.0) (set! rey 0.0) (set! rez 0.0) }
                {
                  (set! sc2 (+ (* eta ddot) (Math.sqrt kk)))
                  (set! rex (- (* irx eta) (* nx1 sc2)))
                  (set! rey (- (* iry eta) (* ny1 sc2)))
                  (set! rez (- (* irz eta) (* nz1 sc2)))
                })
            ;; fresnel = 0.25 + 0.75*pow(1 - dot(normal, -incoming), 3)
            (set! fresnel (+ 0.25 (* 0.75 (* (- 1.0 (- 0.0 ddot))
                                            (* (- 1.0 (- 0.0 ddot)) (- 1.0 (- 0.0 ddot)))))))
            (wt_surface_ray_color rw_x rw_y rw_z rrx rry rrz
                                  COL_ABOVEWATER_R COL_ABOVEWATER_G COL_ABOVEWATER_B)
            (set! ar sr_r) (set! ag sr_g) (set! ab sr_b)
            (wt_surface_ray_color rw_x rw_y rw_z rex rey rez
                                  COL_ABOVEWATER_R COL_ABOVEWATER_G COL_ABOVEWATER_B)
            (set! fc_r (+ sr_r (* (- ar sr_r) fresnel)))
            (set! fc_g (+ sr_g (* (- ag sr_g) fresnel)))
            (set! fc_b (+ sr_b (* (- ab sr_b) fresnel)))
          }
          {
            ;; 水下：法线取反、eta 取 water/air
            (set! nX (- 0.0 nx1)) (set! nY (- 0.0 ny1)) (set! nZ (- 0.0 nz1))
            (set! ddot (+ (+ (* nX irx) (* nY iry)) (* nZ irz)))
            (set! rrx (- irx (* (* nX 2.0) ddot)))
            (set! rry (- iry (* (* nY 2.0) ddot)))
            (set! rrz (- irz (* (* nZ 2.0) ddot)))
            (set! eta (/ IOR_WATER IOR_AIR))
            (set! kk (- 1.0 (* (* eta eta) (- 1.0 (* ddot ddot)))))
            (set! tir 1)
            (if (< kk 0.0)
                { (set! tir 0) (set! rex 0.0) (set! rey 0.0) (set! rez 0.0) }
                {
                  (set! sc2 (+ (* eta ddot) (Math.sqrt kk)))
                  (set! rex (- (* irx eta) (* nX sc2)))
                  (set! rey (- (* iry eta) (* nY sc2)))
                  (set! rez (- (* irz eta) (* nZ sc2)))
                })
            (set! fresnel (+ 0.5 (* 0.5 (* (- 1.0 (- 0.0 ddot))
                                          (* (- 1.0 (- 0.0 ddot)) (- 1.0 (- 0.0 ddot)))))))
            (wt_surface_ray_color rw_x rw_y rw_z rrx rry rrz
                                  COL_UNDERWATER_R COL_UNDERWATER_G COL_UNDERWATER_B)
            (set! ar sr_r) (set! ag sr_g) (set! ab sr_b)
            ;; 全反射（kk<0，从水下看水面超过临界角）：GLSL 的 refract() 返回零向量，
            ;; 折射项权重 t 也是 0 —— 但 0×NaN = NaN，NaN 落到逐像素夹断时 wasm 会夹成
            ;; 1（纯白）、x86 夹成 0（黑）。按 doc/AGENTS.md 2026-11-15 条目（C 版
            ;; ui_water.c / am_native_Raster.c 的同一处缺陷）短路掉整条折射追迹：
            ;; 权重本来就是 0，取反射项即可，结果与"算完再乘 0"在数学上相同。
            (if (== tir 0)
                {
                  (set! e2r ar) (set! e2g ag) (set! e2b ab)
                  (set! tt 0.0)
                }
                {
                  (wt_surface_ray_color rw_x rw_y rw_z rex rey rez 1.0 1.0 1.0)
                  (set! e2r (* sr_r COL_SKY2_R))
                  (set! e2g (* sr_g COL_SKY2_G))
                  (set! e2b (* sr_b COL_SKY2_B))
                  ;; t = (1-w_fres)*length(refr)
                  (set! elen (Math.sqrt (+ (+ (* rex rex) (* rey rey)) (* rez rez))))
                  (set! tt (* (- 1.0 fresnel) elen))
                })
            (set! fc_r (+ ar (* (- e2r ar) tt)))
            (set! fc_g (+ ag (* (- e2g ag) tt)))
            (set! fc_b (+ ab (* (- e2b ab) tt)))
          })
    }))

;; 圆球片段：getSphereColor + 水下透染
(define wt_sphere_frag
  (lambda ()
    {
      (wt_get_sphere_color rw_x rw_y rw_z)
      (if (< rw_y sc_hh)
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
                  (if (>= area 0.0) #f (wt_fill 2 1))
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
      ;; 顶点高度用最近邻（网格只有 DETAIL+1 × DETAIL+1 个顶点，逐顶点省一次双线性）
      (set! gw_y (wt_wat_near gw_u gw_v))
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
      (wt_prep_light)
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
      (wt_flip)
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
            (wt_flip)
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
      ;; 焦散 pass 降频：每 CAUSTIC_EVERY 拍重建一次（caustic_phase 计数）
      (set! caustic_phase (+ caustic_phase 1))
      (if (>= caustic_phase CAUSTIC_EVERY) (set! caustic_phase 0) #f)
      (if (== PASS_CAUSTICS 1)
          (if (== caustic_phase 0) (wt_update_caustics) #f)
          #f)
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

;; 帧缓冲 RGB565 → 屏幕（内部 RW×RH 放大到 SW×SH：一个内部像素 = 一块 fill_rect）
;;   放大块的边界 blkx/blky 在启动时算一次（原来逐像素算，含 4 次除法 + flr）。
(define blkx (cons 0 (quote ())))
(define blky (cons 0 (quote ())))
(define wt_blit
  (lambda ()
    {
      (define iy 0)
      (while (< iy RSTOP)
        {
          (define sy0 (get_item blky iy))
          (define sy1 (get_item blky (+ iy 1)))
          (define bh (- sy1 sy0))
          (define ix 0)
          (while (< ix RW)
            {
              (define bv (get_item frame (+ (* iy RW) ix)))
              (define r5 (flr (/ bv 2048.0)))
              (define g5 (flr (/ (- bv (* r5 2048.0)) 32.0)))
              (define sx0 (get_item blkx ix))
              (define sx1 (get_item blkx (+ ix 1)))
              (Gfx.fill_rect sx0 sy0 (- sx1 sx0) bh
                             (* 255.0 (/ r5 31.0))
                             (* 255.0 (/ g5 63.0))
                             (* 255.0 (/ (- (- bv (* r5 2048.0)) (* g5 32.0)) 31.0)))
              (set! ix (+ ix 1))
            })
          (set! iy (+ iy 1))
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
      (wt_prep_light)
    }))

(define wt_button_action
  (lambda (idx)
    {
      (if (== idx BTN_PAUSE) (set! wt_paused (- 1 wt_paused)) #f)
      (if (== idx BTN_GRAVITY) (set! use_sphere_physics (- 1 use_sphere_physics)) #f)
      (if (== idx BTN_LIGHT) (wt_light_follow_view) #f)
      (if (== idx BTN_QUALITY) (wt_set_quality (mod (+ qlevel 1) QN)) #f)
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
          (if (== i BTN_QUALITY)
              {
                (if (== qlevel 0) (set! label "画质1") #f)
                (if (== qlevel 1) (set! label "画质2") #f)
                (if (== qlevel 2) (set! label "画质3") #f)
                (if (== qlevel 3) (set! label "画质4") #f)
                (if (== qlevel 4) (set! label "画质5") #f)
              }
              #f)
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

;; 上屏放大块边界表（RW+1 / RH+1 个）——切档时要重建
(define wt_build_blit
  (lambda ()
    {
      (set! blkx (cons 0 (quote ())))  (pop blkx)
      (set! blky (cons 0 (quote ())))  (pop blky)
      (define bxi 0)
      (while (<= bxi RW)
        { (push blkx (flr (/ (* 1.0 (* bxi SW)) (* 1.0 RW)))) (set! bxi (+ bxi 1)) })
      (set! bxi 0)
      (while (<= bxi RH)
        { (push blky (flr (/ (* 1.0 (* bxi SH)) (* 1.0 RH)))) (set! bxi (+ bxi 1)) })
    }))

;; 切换画质档位：只改内部渲染分辨率（网格/波场/焦散规模不变），立即重画。
;;   frame/zbuf 按最大档一次性分配过，这里不用重新分配，只是索引步长 RW 变了。
(define wt_set_quality
  (lambda (lvl)
    {
      (set! qlevel lvl)
      (set! RW (+ QRW0 (* QSTEP qlevel)))
      (set! RH (flr (/ (* 3 RW) 4)))
      (set! RSTOP (flr (+ (/ (* 1.0 (* RH (- SH BAR_H))) (* 1.0 SH)) 0.9999)))
      (wt_build_blit)
      (set! view_dirty 1)
    }))

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
      ;; 上屏放大的块边界表（RW+1 / RH+1 个）
      (wt_build_blit)
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
      (set! wpg water0)
      (wt_prep_light)
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
