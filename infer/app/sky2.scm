;; ============================================================================
;; 大气散射天空 II —— Nishita 模型（Atmospheric Sky / Nishita）—— 灵机应用程序
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;; 参考实现：celestial.c 的 scatter_model_2（Nishita 单次散射模型，出处
;; https://www.scratchapixel.com/lessons/procedural_generation_virtual_worlds/
;; simulating_sky/simulating_colors_of_the_sky.html）：视线/太阳路径的
;; 4×4 离散积分、Rayleigh + Mie 相函数、分波长瑞利系数、臭氧吸收、
;; 随太阳高度变化的地球+大气半径 R。只渲染天空，不绘制日月星地物。
;;
;; 与 sky.scm（scatter_model_1 解析模型）的差别：本程序是逐采样点的数值
;; 积分模型——每条视线的光学深度与太阳方向光学深度都要沿路径累加，因此
;; 单点成本更高（13 次 Math native 调用 vs 8 次），分带更细。
;;
; 视角与投影（与 sky.scm/sky3.scm 一致）：
;;   - 视线固定朝天顶：天顶居中，取「人在地面抬头看天」的视角——
;;     **屏幕上方为正北、左侧为正东**（与俯视地图左右相反）；等距方位角投影，
;;     天空为屏幕内切圆，圆外恒黑；视线单位向量 v=(−sinθ·dx/r, −sinθ·dy/r, cosθ)
;;     （x 东、y 北、z 上；dx 向右、dy 向下）。
;;   - 太阳位置取自共享星历模块（app/lib/ephem.scm）：按当前系统时间与项目默认
;;     观测点南京 119.0°E / 32.0°N / UTC+8 求太阳地平坐标；渲染完成后显示太阳
;;     高度/方位与时刻，太阳移动超过 RE_RENDER_DEG 后自动重绘。
;;
;; 积分核的代数化简（与原 C 代码逐位等价，已数值核验 3648 采样点差 0）：
;;   原式内层为 i,j ∈ [0,4) 的双重循环。因 h_R(t)=t·H_R、h_M(t)=t·H_M，
;;   两条路径的归一化密度同为 e^{-t}，于是：
;;     · 视线方向光学深度 depth_R = depth_M = L_ray · PC_i
;;       PC_i = (1/4)·Σ_{k≤i} e^{-t_k}·t_k                （t_k = k/4）
;;     · 太阳方向光学深度 depth_sun_R = depth_sun_M = L_sun · PS_i
;;       PS_i = ((1-t_i)/4)·Σ_j e^{-(t_i + (j/4)(1-t_i))}
;;   PC/PS 只依赖 t_i，可预先算成常量表；且 i=0 时 depth_R≡0，该项贡献恒零。
;;   于是每像素仅剩 i=1..3 的 3×3 次 exp（三通道各 3 次），共 9 次 exp。
;;   透射指数以 (β_c + 1.1·β_Mie) 合并（Mie 消光系数 1.1 已并入 0.055）。
;;
;; 性能设计（宿主每帧一个时间片，约 25 万条 VM 指令）：
;;   - 2×2 降采样；System.set_interval 16ms 状态机；渲染期每步 BAND_ROWS 行
;;     （实测 ≤12 万指令，单步不跨 25 万指令时间片，避免 interval 回调重入），
;;     画完显示字幕，此后按 HOLD_CHECK_MS 轮询太阳是否已明显移动；
;;   - 帧级预计算：太阳向量、大气半径 R、L_sun、臭氧透射；
;;   - 行级预计算：dy、|dy|（圆界精确判定用）。
;;
;; 实现要点（同 gol/particlelife 惯例）：
;;   - 算术一律二元嵌套（编译器对多元乘法只发射一次 mul）；
;;   - 语句位置的 if 分支一律净零块；函数体最后表达式不留值；
;;   - 颜色以浮点直接传给 Gfx（native 端钳位 [0,255] 后截断取整）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Time)

(import Ephem "app/lib/ephem.scm")

;; ---- 画布与投影几何 --------------------------------------------------------

(define PI 3.141592653589793)

(define SW (Gfx.width))
(define SH (Gfx.height))
(define CX (/ SW 2.0))              ; 屏幕中心（天顶投影点）
(define CY (/ SH 2.0))
(define R_SCR (if (< CX CY) CX CY)) ; 天空圆半径（内切于屏幕）
(define R_SCR2 (* R_SCR R_SCR))
(define K_THETA (/ PI (* 2.0 R_SCR)))   ; 每像素天顶角（弧度）：θ = r·K_THETA
(define SS 2)                       ; 降采样步长（采样点绘制 SS×SS 像素块）
(define COLS_S (Math.floor (/ SW SS)))
(define ROWS_S (Math.floor (/ SH SS)))

(define BAND_ROWS 3)                ; 每个 step 渲染的采样行数（本模型较贵）
(define HOLD_MS 700)                ; 每个太阳位置的停留时间（毫秒）

;; ---- 模型常数（与 celestial.c scatter_model_2 一致） -----------------------
;; 瑞利系数 β_c = 0.04·(550/λ_c)⁴，λ = {680,550,450}nm：
;;   β_R ≈ 0.0171189   β_G = 0.04   β_B ≈ 0.0892608
;; 米氏系数 β_Mie = 0.05，相函数 g = 0.76：3(1-g²)/(8π(2+g²)) ≈ 0.0195609
;; 臭氧吸收 {0.005,0.040,0.025} × 0.16；大气标高 H = 8；分层数 S = 4；
;; 全局增益 2.0（×255 → 510）。
;; 积分核常量表 PC_i / PS_i（i = 0..3，见文件头推导；i=0 的 PC 为 0）：
;;   PC = {0, 0.0486750489, 0.1244913814, 0.2130601100}
;;   PS = {0.7144244989, 0.4506483840, 0.2538775826, 0.1077863443}

(define BETA_R 0.0171189)           ; β_R
(define BETA_G 0.04)                ; β_G
(define BETA_B 0.0892608)           ; β_B
(define TAU_R 0.0721189)            ; β_R + 1.1·β_Mie
(define TAU_G 0.095)                ; β_G + 1.1·β_Mie
(define TAU_B 0.1442608)            ; β_B + 1.1·β_Mie

(define PC1 0.0486750489)
(define PC2 0.1244913814)
(define PC3 0.2130601100)
(define PS1 0.4506483840)
(define PS2 0.2538775826)
(define PS3 0.1077863443)

;; ---- 太阳序列与帧级预计算 --------------------------------------------------

;; ---- 太阳位置（取自星历模块的实时值）--------------------------------------
;; 观测点：项目默认南京 119.0°E / 32.0°N / UTC+8（见 app/lib/ephem.scm）。
;; s_sx/s_sy/s_sz 为太阳单位向量（x 东、y 北、z 天顶）；sun_alt_deg 为太阳高度角。
;; 屏幕方位取「抬头看天」视角：上方=北、左侧=东（与俯视地图左右相反）。

(define sun_az 0.0)                 ; 太阳方位角（北=0°，东=90°），显示用
(define sun_alt_deg 90.0)           ; 太阳高度角（度）
(define s_sx 0.0)
(define s_sy 0.0)
(define s_sz 1.0)

;; 行级预计算（render_band 每行更新，shade 读取）
(define row_dy 0.0)                 ; py - CY
(define row_ady1 0.0)               ; (|dy|+1)²：采样块最远角距离平方的行常量部分

;; 帧级预计算（与 celestial.c scatter_model_2 一致）：
;;   R = (3000-300)·e^{-0.1·alt} + 300（alt≥0）；(3000-60)·e^{0.2·alt} + 60（alt<0）
;;   L_sun = √(R² - (R-H)²cos²alt) - (R-H)sin(alt)；臭氧透射 exp(-吸收×L_sun×0.16)
(define r_atm 300.0)
(define r_atm2 90000.0)
(define r_rm 292.0)
(define r_rm2 85264.0)
(define l_sun 0.0)
(define oz_r 1.0)
(define oz_g 1.0)
(define oz_b 1.0)

(define sun_from_ephem
  (lambda () {
    (define ud (Time.utc_days))
    (define us (Time.utc_sod))
    (Ephem.set_time ud us)
    (Ephem.set_gmst)
    (Ephem.sun_equatorial)
    (Ephem.sun_altaz)
    (set! sun_az (Ephem.get_az))
    (set! sun_alt_deg (Ephem.get_alt))
    (define azr (Ephem.to_rad sun_az))
    (define elr (Ephem.to_rad sun_alt_deg))
    (define ca (Math.cos elr))
    (set! s_sx (* ca (Math.sin azr)))
    (set! s_sy (* ca (Math.cos azr)))
    (set! s_sz (Math.sin elr))
    (if (>= sun_alt_deg 0.0)
        { (set! r_atm (+ (* 2700.0 (Math.exp (* -0.1 sun_alt_deg))) 300.0)) }
        { (set! r_atm (+ (* 2940.0 (Math.exp (* 0.2 sun_alt_deg))) 60.0)) })
    (set! r_atm2 (* r_atm r_atm))
    (set! r_rm (- r_atm 8.0))
    (set! r_rm2 (* r_rm r_rm))
    ;; cos²(alt) = s_sx² + s_sy²（一般方位）
    (set! l_sun (- (Math.sqrt (- r_atm2 (* r_rm2 (+ (* s_sx s_sx) (* s_sy s_sy)))))
                   (* r_rm s_sz)))
    (set! oz_r (Math.exp (* -0.0008 l_sun)))
    (set! oz_g (Math.exp (* -0.0064 l_sun)))
    (set! oz_b (Math.exp (* -0.004 l_sun)))
  }))

;; ---- 圆界边缘带的逐像素落点 ------------------------------------------------

(define edge_px
  (lambda (px py dx fr fg fb) {
    (define ddx (- dx 0.5))
    (define ddy (- row_dy 0.5))
    (if (<= (+ (* ddx ddx) (* ddy ddy)) R_SCR2) { (Gfx.set_pixel px py fr fg fb) })
    (set! ddx (+ dx 0.5))
    (if (<= (+ (* ddx ddx) (* ddy ddy)) R_SCR2) { (Gfx.set_pixel (+ px 1) py fr fg fb) })
    (set! ddx (- dx 0.5))
    (set! ddy (+ row_dy 0.5))
    (if (<= (+ (* ddx ddx) (* ddy ddy)) R_SCR2) { (Gfx.set_pixel px (+ py 1) fr fg fb) })
    (set! ddx (+ dx 0.5))
    (if (<= (+ (* ddx ddx) (* ddy ddy)) R_SCR2) { (Gfx.set_pixel (+ px 1) (+ py 1) fr fg fb) })
  }))

;; ---- 单采样点：Nishita 单次散射 → 颜色 → 绘制 SS×SS 块 ---------------------
;; 参数：px/py 采样块左上角像素坐标、dx = px_CX、r2 = dx²+dy²（调用方保证圆内）。

(define shade
  (lambda (px py dx r2) {
    (define r (Math.sqrt r2))
    (define sin_t 0.0)                ; sinθ（= cos(视线仰角)）
    (define vz 1.0)                   ; 视线 z 分量（r≈0 即天顶）
    (define vx 0.0)
    (define vy 0.0)
    (if (> r 0.001)
        { (set! sin_t (Math.sin (* r K_THETA)))
          (set! vz (Math.cos (* r K_THETA)))
          (define inv_r (/ 1.0 r))
          ;; 屏幕上方=北（dy 向下，取负得 +y）、屏幕左侧=东（dx 向右，取负得 +x）
          (set! vx (* (- 0.0 sin_t) (* dx inv_r)))
          (set! vy (* (- 0.0 sin_t) (* row_dy inv_r)))
        })
    ;; 视线路径长：L_ray = √(R² - (R_H)²sin²θ) - (R_H)cosθ
    ;;（cos(视线仰角) = sinθ、sin(视线仰角) = cosθ = vz）
    (define l_ray (- (Math.sqrt (- r_atm2 (* r_rm2 (* sin_t sin_t)))) (* r_rm vz)))
    ;; 视线与太阳夹角余弦、相函数
    (define cos_t (+ (* vx s_sx) (+ (* vy s_sy) (* vz s_sz))))
    (define one_c2 (+ 1.0 (* cos_t cos_t)))
    (define ray_p (* 0.75 one_c2))
    (define mie_p (* 0.0195609 (* one_c2 (Math.pow (- 1.5776 (* 1.52 cos_t)) -1.5))))
    ;; 三层积分的合并光学深度（i=1..3；i=0 项 depth_R≡0 不计）
    (define a1 (+ (* l_ray PC1) (* l_sun PS1)))
    (define a2 (+ (* l_ray PC2) (* l_sun PS2)))
    (define a3 (+ (* l_ray PC3) (* l_sun PS3)))
    ;; 三通道累加：acc_c = Σ_i e^{-(β_c+1.1β_Mie)·A_i}·PC_i，sum_c = L_ray·acc_c
    (define acc_r (+ (* (Math.exp (- 0.0 (* TAU_R a1))) PC1)
                     (+ (* (Math.exp (- 0.0 (* TAU_R a2))) PC2)
                        (* (Math.exp (- 0.0 (* TAU_R a3))) PC3))))
    (define acc_g (+ (* (Math.exp (- 0.0 (* TAU_G a1))) PC1)
                     (+ (* (Math.exp (- 0.0 (* TAU_G a2))) PC2)
                        (* (Math.exp (- 0.0 (* TAU_G a3))) PC3))))
    (define acc_b (+ (* (Math.exp (- 0.0 (* TAU_B a1))) PC1)
                     (+ (* (Math.exp (- 0.0 (* TAU_B a2))) PC2)
                        (* (Math.exp (- 0.0 (* TAU_B a3))) PC3))))
    ;; out_c = min(1, 2·L_ray·acc_c·(β_c·P_Ray + β_Mie·P_Mie)·oz_c) × 255
    (define fr (* 510.0 (* l_ray (* (* acc_r (+ (* BETA_R ray_p) (* 0.05 mie_p))) oz_r))))
    (define fg (* 510.0 (* l_ray (* (* acc_g (+ (* BETA_G ray_p) (* 0.05 mie_p))) oz_g))))
    (define fb (* 510.0 (* l_ray (* (* acc_b (+ (* BETA_B ray_p) (* 0.05 mie_p))) oz_b))))
    ;; 绘制：整块最远角仍在圆内 → SS×SS 块；骑跨圆界 → 逐像素判定
    (define adx (if (< dx 0.0) (- 0.0 dx) dx))
    (if (<= (+ (* (+ adx 1.0) (+ adx 1.0)) row_ady1) R_SCR2)
        { (Gfx.fill_rect px py SS SS fr fg fb) }
        { (edge_px px py dx fr fg fb) })
  }))

;; ---- 渲染状态与分带推进 ----------------------------------------------------

(define render_y 0)
(define hold_until 0)

(define render_band
  (lambda () {
    (define end_y (+ render_y BAND_ROWS))
    (if (> end_y ROWS_S) { (set! end_y ROWS_S) })
    (define sy render_y)
    (while (< sy end_y) {
      (define py (+ (* sy SS) 1))
      (set! row_dy (- py CY))
      (define ady (if (< row_dy 0.0) (- 0.0 row_dy) row_dy))
      (set! row_ady1 (* (+ ady 1.0) (+ ady 1.0)))
      (define sx 0)
      (while (< sx COLS_S) {
        (define px (+ (* sx SS) 1))
        (define dx (- px CX))
        (define r2 (+ (* dx dx) (* row_dy row_dy)))
        (if (<= r2 R_SCR2) { (shade px py dx r2) })
        (set! sx (+ sx 1))
      })
      (set! sy (+ sy 1))
    })
    (set! render_y end_y)
    (if (>= render_y ROWS_S)
        {
          (set! hold_until (+ (System.timestamp) HOLD_MS))
          (draw_caption)
        })
    (Gfx.refresh)
  }))

;; ---- 渲染控制：实时太阳 + 移动判据 + 屏幕字幕 -----------------------------

(define RE_RENDER_DEG 0.4)          ; 太阳移动超过该角度（度）后重绘
(define HOLD_CHECK_MS 700)          ; 停留期间检查太阳位置的间隔

(define rendered_az 0.0)            ; 本帧天空对应的太阳方位/高度
(define rendered_el 0.0)

;; 读一次实时太阳位置（不改变本帧渲染所用的太阳向量），判断是否已明显移动
(define sun_moved_enough
  (lambda () {
    (define ud (Time.utc_days))
    (define us (Time.utc_sod))
    (Ephem.set_time ud us)
    (Ephem.set_gmst)
    (Ephem.sun_equatorial)
    (Ephem.sun_altaz)
    (define az (Ephem.get_az))
    (define el (Ephem.get_alt))
    (define d_az (- az rendered_az))
    (if (> d_az 180.0) { (set! d_az (- d_az 360.0)) })
    (if (< d_az -180.0) { (set! d_az (+ d_az 360.0)) })
    (define d_el (- el rendered_el))
    (define da (Math.abs d_az))
    (define de (Math.abs d_el))
    (if (> (+ da de) RE_RENDER_DEG) 1 0)
  }))

;; ---- HUD：左上太阳高度/方位 · 左下标题 · 右下当前时间 ---------------------
;; 字体为 12px 等宽（ASCII 步进 7px、CJK 步进 12px、行高 15）：
;;   "太阳高度+17.4°  方位114.6°" 约 170px；"YYYY-MM-DD hh:mm:ss" 133px。

(define absv (lambda (v) (if (< v 0.0) (- 0.0 v) v)))

(define pad2
  (lambda (n) {
    (define v (Math.floor n))
    (if (< v 10.0)
        (String.concat "0" (String.atom_to_string v))
        (String.atom_to_string v))
  }))

;; 保留 1 位小数
(define fixed1
  (lambda (v) {
    (define neg (if (< v 0.0) 1 0))
    (define xs (Math.floor (+ (* (absv v) 10.0) 0.5)))
    (define ip (Math.floor (/ xs 10.0)))
    (define fp (- xs (* ip 10.0)))
    (String.concat (if (== neg 1) "-" "")
      (String.concat (String.atom_to_string ip)
        (String.concat "." (String.atom_to_string fp))))
  }))

;; 带符号（太阳高度用）
(define deg_signed1
  (lambda (v) (String.concat (if (>= v 0.0) "+" "-") (String.concat (fixed1 (absv v)) "°"))))

(define hud_clock_x 181)
(define hud_clock_y 224)
(define hud_clock_w 137)
(define hud_clock_h 15)
(define last_sec -1)

;; 当前本地时间 "YYYY-MM-DD hh:mm:ss"（北京时间 = UTC+8）
(define now_string
  (lambda () {
    (define ud (Time.utc_days))
    (define us (Time.utc_sod))
    (define ld ud)
    (define ls (+ us Ephem.TZ_SEC))
    (if (>= ls 86400) { (set! ls (- ls 86400)) (set! ld (+ ld 1)) })
    (Ephem.civil ld)
    (define hh (Math.floor (/ ls 3600.0)))
    (define mi (Math.floor (/ (- ls (* hh 3600.0)) 60.0)))
    (define ss (- ls (+ (* hh 3600.0) (* mi 60.0))))
    (String.concat (String.atom_to_string (Ephem.get_civ_y))
      (String.concat "-" (String.concat (pad2 (Ephem.get_civ_m))
        (String.concat "-" (String.concat (pad2 (Ephem.get_civ_d))
          (String.concat " " (String.concat (pad2 hh)
            (String.concat ":" (String.concat (pad2 mi)
              (String.concat ":" (pad2 ss)))))))))))
  }))

;; 右下角时间：先清掉自己的小块再重画（该块落在天空圆之外的黑角）
(define draw_clock
  (lambda () {
    (Gfx.fill_rect hud_clock_x hud_clock_y hud_clock_w hud_clock_h 0 0 0)
    (Gfx.draw_text (now_string) hud_clock_x (+ hud_clock_y 2) 190 202 214)
    (Gfx.refresh)
  }))

;; 每秒刷新一次时间（仅在天空渲染完成后调用，避免被逐行绘制覆盖）
(define clock_tick
  (lambda () {
    (define us (Time.utc_sod))
    (if (not (== us last_sec)) { (set! last_sec us) (draw_clock) })
  }))

;; 每帧天空完成时：左上一行（太阳高度/方位）、左下标题、右下时间
(define draw_caption
  (lambda () {
    (Gfx.draw_text (String.concat "太阳高度" (String.concat (deg_signed1 sun_alt_deg)
                     (String.concat "  方位" (String.concat (fixed1 sun_az) "°"))))
                   4 3 214 224 238)
    (Gfx.draw_text "玲珑天象仪(基于灵机解释器)" 4 225 170 185 205)
    (set! last_sec -1)
    (draw_clock)
  }))

(define advance
  (lambda () {
    (sun_from_ephem)
    (set! rendered_az sun_az)
    (set! rendered_el sun_alt_deg)
    (set! render_y 0)
    (Gfx.clear)
  }))

(define step
  (lambda () {
    (if (< render_y ROWS_S)
        { (render_band) }
        { (clock_tick)                       ; 右下角时间每秒走一次
          (if (> (System.timestamp) hold_until)
              { (if (== (sun_moved_enough) 1)
                    { (advance) }
                    { (set! hold_until (+ (System.timestamp) HOLD_CHECK_MS)) }) }) })
  }))

;; ---- 启动：取实时太阳 → 清屏 → 步进循环 ------------------------------------

(sun_from_ephem)
(set! rendered_az sun_az)
(set! rendered_el sun_alt_deg)
(Gfx.clear)
(set! update_timer (System.set_interval 16 step))
