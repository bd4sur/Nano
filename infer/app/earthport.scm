;@id=earthport
;@title=地球舷窗
;@version=2026-11-25
;; ============================================================================
;; 地球舷窗（Earth Port）—— 灵机应用程序
;;
;; 一扇"空间站舷窗"：从**东经 120°、纬度 0°**的高空定点俯视地球。画面里的地球就是
;; 此刻真实的那一面 —— 晨昏线的位置与倾角由**当前 UTC 时间**经星历解算得到，夜半球
;; 的城市灯光落在晨昏线的夜侧，大气散射的蓝色边缘与海面太阳反光随太阳方位实时变化。
;; 不渲染云层、不做极光/闪电（按需求范围）。
;;
;; 画面构成（320x240）：
;;   · 地表：NASA Blue Marble（无云）1° 网格 —— 数据在 app/lib/earth_tex.scm（生成物）
;;   · 夜灯：NASA Black Marble 1° 网格 —— 同一份贴图的第四通道（含水面标记）
;;   · 大气：**视线方向的空气柱密度**（Kasten 型空气量）+ 大气自身照明 → 色调映射加色。
;;     天顶方向只有淡淡一层蓝，掠射方向空气量陡增 ⇒ 视圆盘亮区内部是宝石蓝的散射光，
;;     最外圈过曝成青白，圆盘边缘"化开"（大气厚度按视觉需求夸大，HS=0.033R）。
;;   · 反光：海面（贴图蓝通道占优处）按半角向量做镜面高光，光斑位置即真实太阳反光点
;;   · 星空：几十颗暗星（只画在辉光窗之外，重绘地球擦不到）
;;   · HUD：UTC 时刻 / 日期 / 太阳赤纬 / 空间站高度与速度
;;
;; 天文（怎么算出晨昏线）：
;;   视点固定在地固坐标系的 (0°, 120°E)，屏幕上方为正北、右方为正东；正交投影下
;;   屏幕偏移 (dx,dy) 处的球面单位法向 n = (u, sinLat, w)：
;;       u = dx/R,  sinLat = -dy/R,  w = sqrt(1 - u² - sinLat²)
;;   于是"贴图行号 ∝ sinLat"（贴图纵向按 sin(lat) 重采样，省掉 asin），经度偏移
;;   Δλ = atan2(u, w)。太阳方向取自共享星历库 app/lib/ephem.scm（玲珑仪同一份）：
;;     · 太阳赤纬 δ、赤经 α ← (Ephem.sun_equatorial)；GMST ← (Ephem.set_gmst)
;;     · 次太阳点经度 λs = α - GMST ⇒ 视点（120°E）时角 H = 120° - λs
;;     · 视点局部系（东, 北, 天）的太阳单位向量：
;;         sEast = -cosδ·sinH,  sNorth = sinδ,  sUp = cosδ·cosH
;;   逐像素 dot = n·s 就是太阳天顶角余弦：>0 昼、<0 夜；dot=0 的大圆在屏幕上的投影
;;   即晨昏线（一个椭圆），dot∈[-0.2,0] 是暮光过渡带。
;;
;; 渲染节奏（为什么不是每帧整屏重算）：一次整球绘制约 3 万像素、约 600 万条 VM 指令
;;   （wasm 上实测 ≈0.35s 纯解释执行）。而太阳在屏幕上每分钟只移动 0.4 像素，所以：
;;     · 场景（球体/大气）每 SCENE_S=30 秒重算一次，且**分条渐进**绘制
;;       （每拍画 STRIP 行推一次屏），一次重绘跨 ~20 拍完成，看不出"刷"的感觉；
;;     · 只有 UTC 时钟每秒钟重画一行字（fill_rect + draw_text，脏区极小）。
;;   稳态 CPU 占用几个百分点。
;;
;; 输入：触屏点按 / 空格 / 回车 → 显示或隐藏 HUD；硬 A/ESC 与右下角「退出」按钮 →
;;   全局退出确认框（app/lib/exit.scm）。
;;
;; 解释器约束（manual.md + doc/AGENTS.md，逐条遵守）：算术恒二参数；无 let/list/map；
;;   lambda 体内 define 只放在开头；语句位置的 if 两分支都要有值（补 #f）；
;;   `'()` 是模块级静态槽，只 push 一次；块内不 define。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Time)
(native Touch)
(native Key)
(native App)

(import EX "app/lib/exit.scm")      ; 硬 A/ESC + 右下角「退出」按钮 → 全局退出确认框
(import Ephem "app/lib/ephem.scm")  ; 太阳赤纬/赤经、GMST、民用历
(import ET "app/lib/earth_tex.scm") ; 地表四通道贴图（生成物，见 tools/gen_earth_tex.py）

;; ---- 版面 ------------------------------------------------------------------

(define W (Gfx.width))              ; 320
(define H (Gfx.height))             ; 240
(define CX 160.0)                   ; 球心（屏幕上略高于正中，给底部 HUD 让位）
(define CY 118.0)
(define RAD 90.0)                   ; 球半径（像素）
(define RINV (/ 1.0 RAD))           ; 1/R
(define NRINV (- 0.0 RINV))         ; -1/R（sinLat = dy * NRINV）
(define R2I (/ 1.0 (* RAD RAD)))    ; 1/R²
(define DEG 57.29577951308232)      ; 180/π
(define PI 3.141592653589793)

;; 渲染窗口：大气壳 = 半径 1 + HS（归一化）的外圈，外圈按 exp(-(b-1)/HS) 衰减到 3.5HS
(define HS 0.033)                   ; 大气标高（归一化半径；**视觉上把大气厚度夸大**：
                                    ; 真实 100km/6371km ≈ 0.016，这里取 0.02–0.03 的量级，
                                    ; 配合下方的边缘羽化让视圆盘边缘"化开"）
(define HS2 (* 2.0 HS))
(define AIR0 (/ 1.0 (Math.sqrt HS2)))   ; 外圈衰减幅度（与 b=1 处的内圈空气量连续）
(define XMAX 1.15)                  ; 渲染半径 = 1.15R（外圈空气量在此已衰减到 ~1e-3）
(define XMAXR (* RAD XMAX))
(define XMAX2 (* XMAXR XMAXR))
(define ATM_R 0.021)                ; 大气内散射通道增益（色调映射前；R:G:B ≈ 1:3.6:12）
(define ATM_G 0.076)
(define ATM_B 0.260)
(define ATM_FLOOR 0.55)             ; 空气量立方的扣除（天顶方向散射几乎抵消，只留淡淡一层蓝）
(define ATM_AMB 0.015)              ; 夜侧空气辉光的照明下限
;; 大气自身的照明比地表"高"：地平线下 ~7° 以内的高层大气仍被太阳照到，于是晨昏线
;; 两侧会出现一道大气辉光亮边（真实轨道照片里的日出/日落带）。
(define ATM_LIFT 0.10)              ; 被照到的最低太阳天顶角余弦
(define ATM_LIFT_K 4.5)             ; 斜率（抬升后的照明在此饱和）
(define EDGE_FADE 0.071)            ; 地表边缘羽化（最外 ~7px 淡出，交给大气接管）

;; 渲染行范围与每拍行数（球 + 外圈 = y 15..222）
(define ROW_Y0 15)
(define ROW_Y1 222)
(define STRIP 10)

;; 贴图取址（列数/行数取自生成模块，单一事实来源）
(define TCOLS ET.TW)                ; 361：列数（含环绕复制列）
(define ROWH (/ (- ET.TH 1) 2.0))   ; 90：行号 = (1 - sinLat) * ROWH
(define TXR ET.TXR)                 ; 白天红通道（码点 = 0x100 + 值）
(define TXG ET.TXG)
(define TXB ET.TXB)
(define TXN ET.TXN)                 ; 夜间灯光（水面码点再 +0x100）

;; 昼夜常量
(define DAY_AMB 0.05)               ; 昼夜过渡底光
(define DAY_SOFT 0.10)              ; 过渡带软化宽度
(define GLINT_CORE 150.0)           ; 太阳反光亮斑强度（直径较大）
(define GLINT_CORE_K 110.0)          ; 光斑角半径（≈7px 半径）
(define GLINT_WIDE 58.0)            ; 海面宽反光强度（光斑外的大光晕）
(define GLINT_WIDE_K 20.0)
(define TWILIGHT_K 3.0)             ; 暮光带宽度（dot ∈ [-0.33, 0]）
;; 夜面地表残余反照率 = 0.020 × (0.65, 0.85, 1.10)（蓝调，避免夜面发土黄）
(define NIGHT_K_R 0.0130)
(define NIGHT_K_G 0.0170)
(define NIGHT_K_B 0.0220)

;; 配色
(define BG_R 1.0)                   ; 背景（近黑深蓝）
(define BG_G 2.0)
(define BG_B 6.0)
(define HUD_R 168)                  ; HUD 文字
(define HUD_G 194)
(define HUD_B 216)

;; 节拍
(define SCENE_S 30.0)               ; 场景重算周期（秒）
(define TICK_MS 20)                 ; 定时器周期（毫秒）

;; ---- 太阳与世界状态 --------------------------------------------------------

(define SE 0.0)                     ; 太阳方向（视点局部系：东/北/天）
(define SN 0.0)
(define SU -1.0)
(define HX 0.0)                     ; 半角向量（太阳 + 视线 (0,0,1)）
(define HY 0.0)
(define HZ 1.0)
(define SUNUP 0.0)                  ; max(0, SU)
(define DEC_DEG 0.0)                ; 太阳赤纬（度，HUD）
(define alt_km 382.4)               ; 空间站高度（km，HUD）
(define spd_kms 7.68)               ; 轨道速度（km/s，HUD）

;; ---- 运行状态 --------------------------------------------------------------

(define rnd_y ROW_Y0)               ; 下一待绘制行
(define rnd_active 0)               ; 1 = 本场景尚未画完
(define hud_dim 0)                  ; 1 = HUD 已隐藏
(define clk_sec -1)                 ; 上次绘制的 UTC 秒（-1 = 待重画）
(define scen_sod 0.0)               ; 上次场景开始的 UTC 秒
(define cur_days 0)                 ; UTC 日数
(define cur_sod 0)                  ; UTC 当日秒
(define dt_s 0.0)                   ; 场景计时临时量
(define civ_y 2026)                 ; 民用历（HUD）
(define civ_m 1)
(define civ_d 1)
(define tprev 0)                    ; 触屏上一拍是否按住
(define keyv 0)                     ; Key.take_press 临时量
(define _timer 0)                   ; 定时器编号（消费 set_interval 的返回值）

;; ---- 星图（只画一次；重绘地球的区域在辉光窗之内，擦不到它们） --------------

(define star_x '())
(define star_y '())
(define star_b '())

(define mkc_stars
  (lambda ()
    (define n 0)
    (define xx 0.0)
    (define yy 0.0)
    (define dd 0.0)
    (while (< n 200) {
      (set! xx (* (Math.random) 320.0))
      (set! yy (* (Math.random) 240.0))
      (set! dd (Math.sqrt (+ (* (- xx CX) (- xx CX)) (* (- yy CY) (- yy CY)))))
      ;; 只取辉光窗（101.5px）之外、且避开 HUD 两条文字带的点
      (if (and (> dd 108.0) (and (> yy 33.0) (< yy 211.0)))
          {
            (push star_x (Math.floor xx))
            (push star_y (Math.floor yy))
            (push star_b (+ 26.0 (* (Math.random) 90.0)))
          }
          #f)
      (set! n (+ n 1))
    })))

(define draw_stars
  (lambda ()
    (define i 0)
    (define b 0.0)
    (define n (length star_x))
    (while (< i n) {
      (set! b (get_item star_b i))
      (Gfx.set_pixel (get_item star_x i) (get_item star_y i)
                     (+ b 4.0) (+ b 7.0) (+ b 12.0))
      (set! i (+ i 1))
    })))

;; ---- 星历：太阳方向 --------------------------------------------------------

(define sun_update
  (lambda ()
    (define hdeg 0.0)
    (define dr 0.0)
    (define hr 0.0)
    (define cd 0.0)
    (define hl 0.0)
    (define alt 0.0)
    (set! cur_days (Time.utc_days))
    (set! cur_sod (Time.utc_sod))
    (Ephem.set_time cur_days cur_sod)
    (Ephem.set_gmst)
    (Ephem.sun_equatorial)
    (Ephem.civil cur_days)
    (set! civ_y (Ephem.get_civ_y))
    (set! civ_m (Ephem.get_civ_m))
    (set! civ_d (Ephem.get_civ_d))
    (set! DEC_DEG Ephem.s_dec)
    ;; 视点（120°E）的太阳时角 H = 120 - (α - GMST)
    (set! hdeg (- 120.0 (- Ephem.s_ra Ephem.e_gmst)))
    (set! dr (* Ephem.s_dec (/ PI 180.0)))
    (set! hr (* hdeg (/ PI 180.0)))
    (set! cd (Math.cos dr))
    (set! SE (* (- 0.0 cd) (Math.sin hr)))
    (set! SN (Math.sin dr))
    (set! SU (* cd (Math.cos hr)))
    (set! hl (Math.sqrt (+ (* SE SE) (+ (* SN SN) (* (+ SU 1.0) (+ SU 1.0))))))
    ;; 太阳正对镜头背面（SU = -1，δ=0 且 H=180°）时半角向量退化，钳一下防除零
    (if (< hl 0.001) (set! hl 0.001) #f)
    (set! HX (/ SE hl))
    (set! HY (/ SN hl))
    (set! HZ (/ (+ SU 1.0) hl))
    (set! SUNUP (if (> SU 0.0) SU 0.0))
    ;; 空间站高度：75 天周期的缓慢起伏（±2.8km，接近真实的阻尼/抬轨幅度）；
    ;; 速度由活力公式给出 v = sqrt(μ / (Re + h))。两者都只是为了 HUD 有"活气"。
    (set! alt (+ 382.0 (* 2.8 (Math.sin (* (mod cur_days 75.0) 0.08377580409572781)))))
    (set! alt_km alt)
    (set! spd_kms (Math.sqrt (/ 398600.4418 (+ 6378.137 alt))))
  ))

;; ---- 场景绘制（分条渐进） --------------------------------------------------

(define rnd_start
  (lambda ()
    (sun_update)
    (set! rnd_y ROW_Y0)
    (set! rnd_active 1)))

(define rnd_step
  (lambda ()
    (define y rnd_y)
    (define yend (+ rnd_y STRIP))
    (define xa 0.0)
    (define xb 0.0)
    (define x 0.0)
    (define dyf 0.0)
    (define dy2 0.0)
    (define sny 0.0)
    (define rowoff 0.0)
    (define dxf 0.0)
    (define q 0.0)
    (define u 0.0)
    (define w 0.0)
    (define dot 0.0)
    (define col 0.0)
    (define idx 0.0)
    (define cr 0.0)
    (define cg 0.0)
    (define cb 0.0)
    (define r 0.0)
    (define g 0.0)
    (define b 0.0)
    (define k 0.0)
    (define nh 0.0)
    (define dev 0.0)
    (define t1 0.0)
    (define t2 0.0)
    (define gl 0.0)
    (define nc 0.0)
    (define nv 0.0)
    (define twi 0.0)
    (define lk 0.0)
    (define lit 0.0)
    (define air 0.0)
    (define s3 0.0)
    (define ar 0.0)
    (define ag 0.0)
    (define ab 0.0)
    (define cov 0.0)
    (define bo 0.0)
    (define xext 0.0)
    (if (> yend ROW_Y1) (set! yend ROW_Y1) #f)
    (while (< y yend) {
      (set! dyf (- y CY))
      (set! dy2 (* dyf dyf))
      (if (< dy2 XMAX2)
          {
            (set! sny (* dyf NRINV))                        ; sinLat
            (set! rowoff (* (* (- 1.0 sny) ROWH) TCOLS))    ; 行首下标
            (set! xext (Math.sqrt (- XMAX2 dy2)))
            (set! xa (- CX xext))
            (set! xb (+ CX xext))
            (if (< xa 0.0) (set! xa 0.0) #f)
            (if (> xb (- W 1)) (set! xb (- W 1)) #f)
            (set! x xa)
            (while (< x xb) {
              (set! dxf (- x CX))
              (set! q (* (+ (* dxf dxf) dy2) R2I))
              (if (< q 1.0)
                  { ; ==================== 球面 ====================
                    (set! u (* dxf RINV))
                    (set! w (Math.sqrt (- 1.0 q)))
                    (set! dot (+ (* u SE) (+ (* sny SN) (* w SU))))
                    (set! col (+ (* (Math.atan2 u w) DEG) 180.0))
                    (set! idx (+ rowoff col))
                    (if (> dot 0.0)
                        { ; ---------------- 昼半球 ----------------
                          (set! k (* (+ DAY_AMB (* (- 1.0 DAY_AMB) (/ dot (+ dot DAY_SOFT))))
                                     (+ 0.44 (* 0.56 w))))
                          (set! cr (- (String.charCodeAt idx TXR) 256.0))
                          (set! cg (- (String.charCodeAt idx TXG) 256.0))
                          (set! cb (- (String.charCodeAt idx TXB) 256.0))
                          (set! r (* cr k))
                          (set! g (* cg k))
                          (set! b (* cb k))
                          ;; 海面（蓝通道占优）才反光
                          (if (> (- cb cr) 6.0)
                              {
                                (set! nh (+ (* u HX) (+ (* sny HY) (* w HZ))))
                                (if (> nh 0.0)
                                    {
                                      (set! dev (- 1.0 nh))
                                      (set! t1 (+ 1.0 (* dev GLINT_CORE_K)))
                                      (set! t2 (+ 1.0 (* dev GLINT_WIDE_K)))
                                      (set! gl (+ (/ GLINT_CORE (* t1 t1))
                                                  (/ GLINT_WIDE (* t2 t2))))
                                      (set! r (+ r gl))
                                      (set! g (+ g (* gl 0.98)))
                                      (set! b (+ b (* gl 0.94)))
                                    }
                                    #f)
                              }
                              #f)
                        }
                        { ; ---------------- 夜半球 ----------------
                          (set! nc (String.charCodeAt idx TXN))
                          (set! nv (if (> nc 511.0) (- nc 512.0) (- nc 256.0)))
                          (set! twi (if (< dot -0.34) 0.0 (+ 1.0 (* dot TWILIGHT_K))))
                          (set! lk (* nv (- 1.0 (* 0.85 twi))))
                          (set! r (+ (+ (* cr NIGHT_K_R) (* lk 1.28)) (* twi 16.0)))
                          (set! g (+ (+ (* cg NIGHT_K_G) (* lk 0.98)) (* twi 28.0)))
                          (set! b (+ (+ (* cb NIGHT_K_B) (* lk 0.62)) (* twi 52.0)))
                        })
                    ;; 地表边缘羽化：掠射处贴图被压缩成互相矛盾的明暗 texel（锯齿环），
                    ;; 而且大气厚度被夸大后边缘本就该"化开"，所以表面要在最外 ~7px 平滑淡出、
                    ;; 让大气层接管（smoothstep，避免线性斜坡留下环带）。
                    (set! cov (* (- 1.0 q) (* RAD EDGE_FADE)))
                    (if (> cov 1.0) (set! cov 1.0) #f)
                    (set! cov (* cov (* cov (- 3.0 (* 2.0 cov)))))
                    (set! r (* r cov))
                    (set! g (* g cov))
                    (set! b (* b cov))
                    ;; 大气内散射（视线在球内：透过大气壳看到地表）
                    ;;   air = 视线方向上的**空气柱密度**（Kasten 型空气量近似
                    ;;   1/sqrt(1-b²+2H)）：天顶 ≈1，掠射（b→1）≈5；
                    ;;   立方后掠射/天顶的比值 ≈125 —— 这正是"宝石蓝只在天顶附近淡淡一层、
                    ;;   越靠边缘越浓、最后在边缘处过曝成青白"的物理来源。
                    (set! air (/ 1.0 (Math.sqrt (- (+ 1.0 HS2) q))))
                    (set! s3 (- (* air (* air air)) ATM_FLOOR))
                    (if (< s3 0.0) (set! s3 0.0) #f)
                    (set! lit (* (+ dot ATM_LIFT) ATM_LIFT_K))
                    (if (< lit 0.0) (set! lit 0.0) #f)
                    (if (> lit 1.0) (set! lit 1.0) #f)
                    (set! s3 (* s3 (+ ATM_AMB (* (- 1.0 ATM_AMB) lit))))
                    ;; 色调映射 c = 255a/(a+1)：小 a 时线性（蓝），大 a 时各通道一起饱和（青白），
                    ;; 于是边缘自然出现"亮到发白"的散射核，而不是塑料般的均匀提亮。
                    (set! ar (* s3 ATM_R))
                    (set! ag (* s3 ATM_G))
                    (set! ab (* s3 ATM_B))
                    (Gfx.set_pixel x y
                                   (+ r (/ (* 255.0 ar) (+ ar 1.0)))
                                   (+ g (/ (* 255.0 ag) (+ ag 1.0)))
                                   (+ b (/ (* 255.0 ab) (+ ab 1.0))))
                  }
                  { ; ==================== 球外：只有大气（柔和边缘） ====================
                    (set! bo (- (Math.sqrt q) 1.0))
                    (set! air (* AIR0 (Math.exp (- 0.0 (/ bo HS)))))
                    (set! s3 (- (* air (* air air)) ATM_FLOOR))
                    (if (< s3 0.0) (set! s3 0.0) #f)
                    (set! u (* dxf RINV))
                    (set! lit (* (+ (+ (* u SE) (* sny SN)) ATM_LIFT) ATM_LIFT_K))
                    (if (< lit 0.0) (set! lit 0.0) #f)
                    (if (> lit 1.0) (set! lit 1.0) #f)
                    (set! s3 (* s3 (+ ATM_AMB (* (- 1.0 ATM_AMB) lit))))
                    (set! ar (* s3 ATM_R))
                    (set! ag (* s3 ATM_G))
                    (set! ab (* s3 ATM_B))
                    (Gfx.set_pixel x y
                                   (+ BG_R (/ (* 255.0 ar) (+ ar 1.0)))
                                   (+ BG_G (/ (* 255.0 ag) (+ ag 1.0)))
                                   (+ BG_B (/ (* 255.0 ab) (+ ab 1.0))))
                  })
              (set! x (+ x 1))
            })
          }
          #f)
      (set! y (+ y 1))
    })
    (set! rnd_y yend)
    (if (>= rnd_y ROW_Y1) (set! rnd_active 0) #f)
    (Gfx.refresh)))

;; ---- HUD -------------------------------------------------------------------

(define MONTHS '("JAN" "FEB" "MAR" "APR" "MAY" "JUN" "JUL" "AUG" "SEP" "OCT" "NOV" "DEC"))

;; 定长小数：把 v 写成 nd（1 或 2）位小数。**不用 (Math.to_fixed)**——32 位宿主上
;; 它的取整会留下 7.67999 这样的尾巴（wasm 实测）；这里用整数运算自己拼。
(define num_fix
  (lambda (v nd)
    (define neg 0)
    (define a 0.0)
    (define iv 0.0)
    (define fr 0.0)
    (define sc 10.0)
    (define pad "")
    (if (== nd 2) (set! sc 100.0) #f)
    (if (< v 0.0) (set! neg 1) #f)
    (set! a (Math.abs v))
    (set! iv (Math.floor a))
    (set! fr (Math.floor (+ (* (- a iv) sc) 0.5)))
    (if (>= fr sc) { (set! fr 0.0) (set! iv (+ iv 1.0)) } #f)
    (if (and (== nd 2) (< fr 10.0)) (set! pad "0") #f)
    (String.concat (if (== neg 1) "-" "")
      (String.concat (String.atom_to_string iv)
        (String.concat "." (String.concat pad (String.atom_to_string fr)))))))

(define pad2
  (lambda (v)
    (if (< v 10.0)
        (String.concat "0" (String.atom_to_string (Math.floor v)))
        (String.atom_to_string (Math.floor v)))))

;; 一行 HUD：先擦底（背景纯色）再写字，避免残影。宽度按半角 7px/字 + 余量。
(define hud_text
  (lambda (s x y nch)
    (Gfx.fill_rect x y (+ 6 (* nch 7)) 14 BG_R BG_G BG_B)
    (Gfx.draw_text s x y HUD_R HUD_G HUD_B)))

;; 右对齐版本（xr = 右边界）
(define hud_text_r
  (lambda (s xr y nch)
    (hud_text s (- xr (* nch 7)) y nch)))

(define draw_clock
  (lambda ()
    (define sec0 0.0)
    (define min0 0.0)
    (define hr0 0.0)
    (define str "")
    (if (== hud_dim 1)
        #f
        {
          (set! sec0 (mod cur_sod 60.0))
          (set! min0 (mod (/ (- cur_sod sec0) 60.0) 60.0))
          (set! hr0 (/ (- (- cur_sod sec0) (* min0 60.0)) 3600.0))
          (set! str (String.concat "UTC " (String.concat (pad2 hr0)
                          (String.concat ":" (String.concat (pad2 min0)
                          (String.concat ":" (pad2 sec0)))))))
          (hud_text str 6 3 13)
        })))

(define draw_hud_static
  (lambda ()
    (define dstr "")
    (define sdec "")
    (define astr "")
    (define sstr "")
    (if (== hud_dim 1)
        #f
        {
          (set! dstr (String.concat (String.atom_to_string civ_d)
                        (String.concat " " (String.concat (get_item MONTHS (- civ_m 1))
                        (String.concat " " (String.atom_to_string civ_y))))))
          (set! sdec (String.concat "DEC " (String.concat
                        (if (>= DEC_DEG 0.0) "+" "-")
                        (String.concat (num_fix (Math.abs DEC_DEG) 1) "°"))))
          (set! astr (String.concat "CSS " (String.concat (num_fix alt_km 1) " KM")))
          (set! sstr (String.concat (num_fix spd_kms 2) " KM/S"))
          (hud_text dstr 6 18 12)
          (hud_text_r sdec (- W 6) 3 11)
          (hud_text astr 6 222 11)
          (hud_text_r sstr (- W 58) 222 9)
        })))

;; 只擦 HUD 文字所在的小矩形（不碰球/辉光/刻度环）
(define clear_hud
  (lambda ()
    (Gfx.fill_rect 4 1 112 16 BG_R BG_G BG_B)
    (Gfx.fill_rect 4 16 96 16 BG_R BG_G BG_B)
    (Gfx.fill_rect (- W 100) 1 96 16 BG_R BG_G BG_B)
    (Gfx.fill_rect 4 221 96 17 BG_R BG_G BG_B)
    (Gfx.fill_rect (- W 130) 221 74 17 BG_R BG_G BG_B)))

;; ---- 输入 ------------------------------------------------------------------

(define toggle_hud
  (lambda ()
    (set! hud_dim (if (== hud_dim 1) 0 1))
    (set! clk_sec -1)
    (if (== hud_dim 1)
        (clear_hud)
        {
          (draw_stars)
          (draw_clock)
          (draw_hud_static)
        })
    (Gfx.refresh)))

(define read_touch
  (lambda ()
    (if (Touch.is_available)
        {
          (if (Touch.is_pressed)
              {
                (if (== tprev 0)
                    {
                      (Touch.take_press)
                      (toggle_hud)
                    }
                    #f)
                (set! tprev 1)
              }
              (set! tprev 0))
        }
        #f)))

(define read_key
  (lambda ()
    (if (Key.is_available)
        {
          (set! keyv (Key.take_press))
          (while (> keyv 0)
            {
              (if (or (== keyv 32) (== keyv 13)) (toggle_hud) #f)
              (set! keyv (Key.take_press))
            })
        }
        #f)))

;; ---- 背景（一条极淡的深蓝渐变，40 条横带） --------------------------------

(define draw_bg
  (lambda ()
    (define i 0)
    (while (< i 40) {
      (Gfx.fill_rect 0 (* i 6) W 6
                     (+ BG_R (* i 0.02)) (+ BG_G (* i 0.035)) (+ BG_B (* i 0.09)))
      (set! i (+ i 1))
    })))

;; ---- 每拍 ------------------------------------------------------------------

(define redraw_all
  (lambda ()
    (draw_bg)
    (draw_stars)
    (EX.draw_button_at 270 219)
    ;; 先补 HUD（球体由随后的分条绘制补），退出确认框关掉后立刻恢复原样
    (draw_clock)
    (draw_hud_static)
    (set! clk_sec cur_sod)
    (set! rnd_y ROW_Y0)
    (set! rnd_active 1)
    (Gfx.refresh)))

(define on_tick
  (lambda ()
    (EX.check)                          ; 硬 A/ESC / 右下角「退出」
    (read_touch)
    (read_key)
    (set! cur_sod (Time.utc_sod))
    ;; 时钟：每秒一次
    (if (== cur_sod clk_sec)
        #f
        {
          (set! clk_sec cur_sod)
          (draw_clock)
        })
    ;; 场景调度：空闲且距上次场景开始超过 SCENE_S 秒 → 重算
    (if (== rnd_active 0)
        {
          (set! dt_s (- cur_sod scen_sod))
          (if (< dt_s 0.0) (set! dt_s (+ dt_s 86400.0)) #f)
          (if (>= dt_s SCENE_S)
              {
                (set! scen_sod cur_sod)
                (rnd_start)
              }
              #f)
        }
        #f)
    ;; 分条绘制
    (if (== rnd_active 1)
        {
          (rnd_step)
          (if (== rnd_active 0)
              {
                (draw_hud_static)
                ;; 一帧场景画完（e2e 判据：整球分条绘制确实完成，而非只画了上半球；
                ;; heap 用于长跑泄漏观测——逐像素内核零分配，此值应长期恒定）
                (display "[earthport] scene done heap=")
                (display (get_item (System.memstat) 3)) (newline)
              }
              #f)
        }
        #f)
    (Gfx.refresh)))

;; ---- 启动 ------------------------------------------------------------------

(Gfx.set_font 1)            ; 12px 抗锯齿字模（HUD）
(mkc_stars)
(EX.install_with redraw_all)
(draw_bg)
(draw_stars)
(EX.draw_button_at 270 219)
(sun_update)
(set! scen_sod cur_sod)
(set! rnd_y ROW_Y0)
(set! rnd_active 1)
(draw_clock)
(draw_hud_static)
(Gfx.refresh)
(set! _timer (System.set_interval TICK_MS on_tick))
(display "[earthport] sun=(")
(display (Math.to_fixed SE 3)) (display ",")
(display (Math.to_fixed SN 3)) (display ",")
(display (Math.to_fixed SU 3))
(display ") dec=") (display (Math.to_fixed DEC_DEG 2))
(display " tex=") (display (String.length TXR))
(display " mem=") (display (get_item (System.memstat) 2))
(display "/") (display (get_item (System.memstat) 3))
(newline)
