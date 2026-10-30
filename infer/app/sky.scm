;; ============================================================================
;; 大气散射天空（Atmospheric Sky）—— Nano「灵机应用程序」Scheme App
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;; 参考实现：celestial.c 的 scatter_model_1（Rayleigh + Mie 双散射解析模型：
;; Kasten_Young 大气质量、分波长瑞利系数、米氏前向散射、臭氧吸收、太阳高度
;; 衰减尺度），在 Scheme 中逐采样点复现；只渲染天空本身，不绘制日月星地物。
;;
;; 视角与投影：
;;   - 视线固定朝天顶（俯仰 90°、横滚 0°、方位 0°）：天顶位于屏幕中央，
;;     取「人在地面抬头看天」的视角：**屏幕上方为正北、左侧为正东**
;;     （与俯视地图左右相反）；等距方位角（鱼眼）投影下天空是屏幕内切圆，
;;     圆外一律为黑色（Gfx.clear 置黑后仅绘制圆内采样点）。
;;   - 屏幕采样点 (px,py) → 天空方向：r=√(dx²+dy²)，天顶角 θ=(r/R)·90°，
;;     视线单位向量 v=(−sinθ·dx/r, −sinθ·dy/r, cosθ)（x 东、y 北、z 上；
;;     dx 向右、dy 向下）。由 −dx/r、−dy/r 直接充当 sin(az)/cos(az)——免去
;;     atan2（vendored Math 库无 atan2/asin/acos），每采样点三角函数只需
;;     sinθ 与 cosθ 两次。
;;
;; 太阳位置：
;;   取自共享星历模块（app/lib/ephem.scm，同一份库也被 ephemeris.scm /
;;   nongli.scm / sky2.scm / sky3.scm 复用）：按当前系统时间与项目默认观测点
;;   南京 119.0°E / 32.0°N / UTC+8 求太阳地平坐标，再换成三分量单位向量送入
;;   散射模型。渲染完成后显示太阳高度/方位与时刻；太阳移动超过
;;   RE_RENDER_DEG 后自动按新位置重绘。
;;
;; 性能设计（宿主每帧只泵一个时间片，约 25 万条 VM 指令）：
;;   - 2×2 降采样：每采样点计算一次散射并绘制 2×2 像素色块（天空平滑，
;;     采样点中心取块中心 px=2sx+1）；
;;   - 事件驱动状态机：System.set_interval 每 16ms 一步，渲染期每步画
;;     BAND_ROWS 个采样行（约 15 万条指令，单步不跨时间片，避免 interval
;;     回调重入）；画完一帧显示字幕，此后每 HOLD_CHECK_MS 检查一次太阳是否
;;     已明显移动（> RE_RENDER_DEG），是则重绘，否则继续停留；
;;   - 行级/帧级预计算：dy²、|dy|、太阳方向向量、太阳大气质量、臭氧透射、
;;     高度衰减尺度均只算一次；
;;   - 每采样点仅 8 次 Math native 调用（sqrt/sin/cos/pow×2/exp×3）：
;;     三通道透射与密度因子合并为单次 exp —— e_c=exp(-(β_c+β_Mie)·q_v_z/8.5)。
;;
;; 实现要点（同 gol/particlelife 惯例）：
;;   - 算术一律二元嵌套（本解释器编译器对多元乘法只发射一次 mul）；
;;   - 语句位置的 if 分支一律净零块（vendored 编译器不回收语句位置的残留
;;     值），shade/edge_px/render_band/step 函数体最后表达式均不留值；
;;   - 颜色以浮点直接传给 Gfx（native 端钳位 [0,255] 后截断取整）；
;;   - 坐标以浮点参与运算、Gfx 端取整。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Time)

(import Ephem "app/lib/ephem.scm")
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 画布与投影几何 --------------------------------------------------------

(define PI 3.141592653589793)

(define SW (Gfx.width))
(define SH (Gfx.height))
(define CX (/ SW 2.0))              ; 屏幕中心（天顶投影点）
(define CY (/ SH 2.0))
(define R (if (< CX CY) CX CY))     ; 天空圆半径（内切于屏幕）
(define R2 (* R R))
(define K_THETA (/ PI (* 2.0 R)))   ; 每像素天顶角（弧度）：θ = r·K_THETA
(define K_DEG (/ 90.0 R))           ; 每像素天顶角（度）：大气质量公式用
(define SS 2)                       ; 降采样步长（采样点绘制 SS×SS 像素块）
(define COLS_S (Math.floor (/ SW SS)))   ; 采样列数
(define ROWS_S (Math.floor (/ SH SS)))   ; 采样行数

(define BAND_ROWS 6)                ; 每个 step 渲染的采样行数（控制单步工作量）
(define HOLD_MS 700)                ; 每个太阳位置的停留时间（毫秒）

;; ---- 太阳序列与帧级预计算 --------------------------------------------------
;; 散射常数（与 celestial.c scatter_model_1 一致，Rayleigh 参考波长 550nm）：
;;   β_R = 0.3×(550/680)⁴ ≈ 0.128392    β_G = 0.3    β_B = 0.3×(550/450)⁴ ≈ 0.669455
;;   Mie 相函数 g=0.9 → (1+g²-2g·cosθ)^-1.5；密度因子 exp(-v_z/8.5)；
;;   Kasten_Young 大气质量 AM=1/(cos z+0.50572·(96.07995-z_°)^-1.6364)；
;;   臭氧吸收 {0.005,0.040,0.025}×0.8；高度衰减 exp(-(alt/20)⁴)+0.01；增益 2。

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

;; 帧级预计算（最终与 celestial.c scatter_model_1 一致）：
;;   sun_am    : 太阳方向大气质量（Kasten-Young；太阳在地平线下时取 40）
;;   attn_sc   : 太阳高度衰减尺度 exp(-(alt/20)⁴)+0.01（alt<0 时取 1）
;;   oz_r/g/b  : 臭氧透射（吸收系数 ×0.8 已并入指数）
;;   night_attr: 太阳低于地平线时的额外衰减 max(0, exp(0.016·alt))
(define sun_am 1.0)
(define attn_sc 1.0)
(define oz_r 1.0)
(define oz_g 1.0)
(define oz_b 1.0)
(define night_attr 1.0)

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
        { ;; Kasten-Young：z_° = 90-alt → 96.07995 - z_° = 6.07995 + alt
          (set! sun_am (/ 1.0 (+ s_sz (* 0.50572 (Math.pow (+ 6.07995 sun_alt_deg) -1.6364)))))
          (set! attn_sc (+ (Math.exp (- 0.0 (Math.pow (/ sun_alt_deg 20.0) 4.0))) 0.01))
          (set! night_attr 1.0) }
        { (set! sun_am 40.0)
          (set! attn_sc 1.0)
          (define na (Math.exp (* 0.016 sun_alt_deg)))
          (if (< na 0.0) { (set! na 0.0) })
          (set! night_attr na) })
    (set! oz_r (Math.exp (* -0.004 sun_am)))
    (set! oz_g (Math.exp (* -0.032 sun_am)))
    (set! oz_b (Math.exp (* -0.020 sun_am)))
  }))

;; ---- 圆界边缘带的逐像素落点 ------------------------------------------------
;; 采样块骑跨天空圆界时，4 个像素各自按精确半径判定（保证圆外像素恒为黑）。

(define edge_px
  (lambda (px py dx fr fg fb) {
    (define ddx (- dx 0.5))
    (define ddy (- row_dy 0.5))
    (if (<= (+ (* ddx ddx) (* ddy ddy)) R2) { (Gfx.set_pixel px py fr fg fb) })
    (set! ddx (+ dx 0.5))
    (if (<= (+ (* ddx ddx) (* ddy ddy)) R2) { (Gfx.set_pixel (+ px 1) py fr fg fb) })
    (set! ddx (- dx 0.5))
    (set! ddy (+ row_dy 0.5))
    (if (<= (+ (* ddx ddx) (* ddy ddy)) R2) { (Gfx.set_pixel px (+ py 1) fr fg fb) })
    (set! ddx (+ dx 0.5))
    (if (<= (+ (* ddx ddx) (* ddy ddy)) R2) { (Gfx.set_pixel (+ px 1) (+ py 1) fr fg fb) })
  }))

;; ---- 单采样点：视线方向散射 → 颜色 → 绘制 SS×SS 块 -------------------------
;; 参数：px/py 采样块左上角像素坐标、dx = px_CX、r2 = dx²+dy²（调用方保证圆内）。

(define shade
  (lambda (px py dx r2) {
    (define r (Math.sqrt r2))
    (define sin_t 0.0)                ; sinθ
    (define vz 1.0)                   ; 视线 z 分量（r≈0 即天顶，直接置 1）
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
    ;; 视线大气质量：alt_view = 90° - r·K_DEG → 指数底 96.07995 - r·K_DEG
    (define vam (/ 1.0 (+ vz (* 0.50572 (Math.pow (- 96.07995 (* r K_DEG)) -1.6364)))))
    ;; 视线与太阳方向夹角余弦 + 相函数
    (define cos_t (+ (* vx s_sx) (+ (* vy s_sy) (* vz s_sz))))
    (define mie_b (* 0.001 vam))      ; Mie 系数随地平线气溶胶增长，上限 0.1
    (if (> mie_b 0.1) { (set! mie_b 0.1) })
    (define mie_p (Math.pow (- 1.81 (* 1.8 cos_t)) -1.5))
    (define ray_p (* 0.75 (+ 1.0 (* cos_t cos_t))))
    ;; 合并透射×密度：e_c = exp(-(β_c+β_Mie)·q - v_z/8.5)，q = AM_view×衰减尺度
    (define q (* vam attn_sc))
    (define vzd (/ vz 8.5))
    (define e_r (Math.exp (- 0.0 (+ (* (+ 0.128392 mie_b) q) vzd))))
    (define e_g (Math.exp (- 0.0 (+ (* (+ 0.3 mie_b) q) vzd))))
    (define e_b (Math.exp (- 0.0 (+ (* (+ 0.669455 mie_b) q) vzd))))
    ;; 散射亮度：out = min(1, 2·(β·P_Ray+β_M·P_Mie)·e_c·oz_c)×255（Gfx 端钳位）
    (define g (* 510.0 night_attr))
    (define fr (* g (* (+ (* 0.128392 ray_p) (* mie_b mie_p)) (* e_r oz_r))))
    (define fg (* g (* (+ (* 0.3 ray_p) (* mie_b mie_p)) (* e_g oz_g))))
    (define fb (* g (* (+ (* 0.669455 ray_p) (* mie_b mie_p)) (* e_b oz_b))))
    ;; 绘制：整块最远角仍在圆内 → SS×SS 块；骑跨圆界 → 逐像素判定
    (define adx (if (< dx 0.0) (- 0.0 dx) dx))
    (if (<= (+ (* (+ adx 1.0) (+ adx 1.0)) row_ady1) R2)
        { (Gfx.fill_rect px py SS SS fr fg fb) }
        { (edge_px px py dx fr fg fb) })
  }))

;; ---- 渲染状态与分带推进 ----------------------------------------------------

(define render_y 0)                 ; 下一个待渲染的采样行
(define hold_until 0)               ; 当前天空渲染完成后停留的截止时刻（毫秒）

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
        (if (<= r2 R2) { (shade px py dx r2) })
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
      (EX.draw_button)
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

;; ---- 右下角时间：透明底 ----------------------------------------------------
;; 时钟块横跨天空圆界（左半落在圆内的天空上、右半落在圆外的黑底上），因此每次
;; 重画前先把这块区域的“真实背景”恢复出来，而不是整块填黑：
;;   · 与时钟块相交的采样块重新着色：块在圆内 → 用当前帧参数重新 shade（含圆界
;;     采样块由 shade 自带的判定逐像素处理）；块在圆外 → 置黑，与逐行渲染时
;;     “圆外一律为黑”一致；
;;   · 随后把时间文字直接画在天空上（alpha 字模按覆盖率与背景混合，边缘自然）。
;; 采样块左上角为 (2sx+1, 2sy+1)、覆盖 px..px+1 / py..py+1，故列/行的相交范围
;; 由 hud_clock_* 折算（两端各容 1 个采样块：落在块外的少量像素本已是黑底或
;; 天空，重绘结果与原画面一致）。

(define clock_bg_sx0 (Math.floor (/ (- hud_clock_x 2) SS)))
(define clock_bg_sx1 (Math.floor (/ (- (+ hud_clock_x hud_clock_w) 2) SS)))
(define clock_bg_sy0 (Math.floor (/ (- hud_clock_y 2) SS)))
(define clock_bg_sy1 (Math.floor (/ (- (+ hud_clock_y hud_clock_h) 2) SS)))

(define repaint_clock_bg
  (lambda () {
    (define sy clock_bg_sy0)
    (while (<= sy clock_bg_sy1) {
      (define py (+ (* sy SS) 1))
      (set! row_dy (- py CY))
      (define ady (if (< row_dy 0.0) (- 0.0 row_dy) row_dy))
      (set! row_ady1 (* (+ ady 1.0) (+ ady 1.0)))
      (define sx clock_bg_sx0)
      (while (<= sx clock_bg_sx1) {
        (define px (+ (* sx SS) 1))
        (define dx (- px CX))
        (define r2 (+ (* dx dx) (* row_dy row_dy)))
        (if (<= r2 R2)
            { (shade px py dx r2) }
            { (Gfx.fill_rect px py SS SS 0 0 0) })
        (set! sx (+ sx 1))
      })
      (set! sy (+ sy 1))
    })
  }))

;; 右下角时间：先恢复该块的真实背景（透明底），再画时间
(define draw_clock
  (lambda () {
    (repaint_clock_bg)
    (Gfx.draw_text (now_string) hud_clock_x (+ hud_clock_y 2) 190 202 214)
      (EX.draw_button)
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
    (EX.check)
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

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 advance 擦掉对话框残影）
(EX.install_with advance)
