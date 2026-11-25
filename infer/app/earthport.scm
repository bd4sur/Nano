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
;;   · HUD：UTC 时刻 / 日期 / 太阳赤纬 / 时间偏置徽标（暂停或 ±h:mm，琥珀色）
;;   · 底部控制条：[-2分] [+2分] [实时] [暂停/运行] + 右下「退出」
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
;; 时间机器（底部控制条）：
;;   运行态的显示时刻 = 实时 UTC + off_s。[-2分]/[+2分] 每次把偏置挪 ±120 秒；
;;   [暂停] 冻结时刻（sim_d/sim_s 不再跟随实时），再按「运行」时把"冻结期间积累的时间差"
;;   折进 off_s —— 于是画面从你冻结的那一刻继续，而不是突然跳回今天；
;;   [实时] 偏置清零并解除暂停，回到真实时间。
;;   任何时刻变化都置 scene_dirty，让下一拍立刻重算场景（晨昏线跟着跳）。
;;
;; 输入：底部四个按钮（触屏）；其它区域点按 / 空格 / 回车 → 显示或隐藏 HUD 文字；
;;   键盘 ←/→ = ∓2 分钟，↑ = 实时，↓ = 暂停/运行；硬 A/ESC 与右下「退出」按钮 →
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

;; 渲染窗口：球体 + 球外 HALO_U（≈4px）的一层薄辉光
(define HALO_U 0.045)               ; 球外辉光的径向厚度（归一化半径；≈4px）。
                                    ; 真实大气 ~100km/6371km ≈ 1.6%（1.4px），
                                    ; 这里留 3 倍余量让它可见而不显得"裹了层厚壳"
(define HS 0.033)                   ; 大气标高（归一化半径；**视觉上把大气厚度夸大**：
                                    ; 真实 100km/6371km ≈ 0.016，这里取 0.02–0.03 的量级，
                                    ; 配合下方的边缘羽化让视圆盘边缘"化开"）
(define HS2 (* 2.0 HS))
(define AIR0 (/ 1.0 (Math.sqrt HS2)))   ; b=1 处的空气量（内外圈在此严格连续）
(define S3L (- (* AIR0 (* AIR0 AIR0)) 0.55))  ; b=1 处的 s3（= 内圈的 air³-FLOOR），
                                              ; 球外用它做三次缓出的起点
(define XMAX (+ 1.0 HALO_U))        ; 渲染半径 = 1.045R（球外辉光在 HALO_U 处平滑归零）
(define XMAXR (* RAD XMAX))
(define XMAX2 (* XMAXR XMAXR))
(define ATM_R 0.021)                ; 大气内散射通道增益（色调映射前；R:G:B ≈ 1:3.6:12）
(define ATM_G 0.076)
(define ATM_B 0.260)
(define ATM_FLOOR 0.55)             ; 空气量立方的扣除（天顶方向散射几乎抵消，只留淡淡一层蓝）
(define ATM_AMB 0.007)              ; 夜侧气辉/多次散射的下限（夜缘只留一道暗蓝弧）
;; 大气自身的照明比地表"高"：地平线下 ~7° 以内的高层大气仍被太阳照到，于是晨昏线
;; 两侧会出现一道大气辉光亮边（真实轨道照片里的日出/日落带）。
(define ATM_LIFT 0.10)              ; 被照到的最低太阳天顶角余弦
(define ATM_LIFT_K 4.5)             ; 斜率（抬升后的照明在此饱和）
(define EDGE_FADE 0.071)            ; 地表边缘羽化（最外 ~7px 淡出，交给大气接管）

;; 渲染行范围与每拍行数（球 + 外圈，底部让出控制条）
(define ROW_Y0 23)
(define ROW_Y1 213)
(define STRIP 10)

;; 贴图取址（列数/行数取自生成模块，单一事实来源）
(define TCOLS ET.TW)                ; 361：列数（含环绕复制列）
(define ROWH (/ (- ET.TH 1) 2.0))   ; 90：行号 = (1 - sinLat) * ROWH
(define TXR ET.TXR)                 ; 白天红通道（码点 = 0x100 + 值）
(define TXG ET.TXG)
(define TXB ET.TXB)
(define TXN ET.TXN)                 ; 夜间灯光（水面码点再 +0x100）
(define TXW ET.TXW)                 ; 低云云量（程序化生成，见 tools/cloud_field.py）
(define TXC ET.TXC)                 ; 高空卷云（薄、丝缕状，叠加在低云之上）

;; 昼夜（**单一连续式**：地表亮度/色相只由太阳天顶角余弦 dot 驱动，不按昼/夜分支——
;;   否则两侧各用一套系数，必然在 dot=0 处留下一条亮度与色相的断层线）
(define SUN_FLOOR 0.014)            ; 无直射时的地表残余反照率（星光/大气辉光）
(define SUN_SOFT 0.10)              ; 直射份额的软化宽度（dot=0 处恰为 0，连续）
(define DUSK_HI 0.12)               ; 城市灯开始点亮的天顶角余弦
(define DUSK_SPAN 0.33)             ; 完全点亮的跨度（dot: 0.12 → -0.21）
;; 云：一次查表 + alpha 合成。云量 0..255 → alpha 0..CLOUD_MAX
;;   云的几何厚度远小于 1°/像素，所以不做三维体渲染；用"平面云层 + 太阳漫射照明"近似，
;;   掠射处只做更弱的视线暗化（云比地表高、多次散射更强）。
(define CLOUD_MAX 0.92)             ; 云量 255 时的最大遮蔽
(define CLOUD_K 0.0036078)          ; = CLOUD_MAX / 255
(define CIRRUS_K 0.0021569)         ; 卷云通道 0..255 → 不透明度 0..0.55
(define CIRRUS_ALB 246.0)          ; 卷云反照率（薄而亮、略偏冷）
(define CLOUD_LIFT 0.035)           ; 云顶比地面高 ⇒ 晨昏线附近晚一点变暗（可见的细节）
(define CLOUD_BASE 188.0)          ; 极薄云的反照率（灰白）
(define CLOUD_DENSE 62.0)          ; 厚度带来的增量：厚云顶 → 纯白（188+62·a）
;; 海面太阳反光（半角向量；nh≤0 处自然为 0，连续）
(define GLINT_CORE 150.0)           ; 亮斑强度（直径较大）
(define GLINT_CORE_K 110.0)         ; 光斑角半径（≈7px 半径）
(define GLINT_WIDE 58.0)            ; 光斑外的大光晕强度
(define GLINT_WIDE_K 20.0)
;; 大气：低太阳时的红化斜率（入射阳光穿过更长气柱 → 蓝先被散射掉）
(define SUN_RED 0.30)
;; 米氏散射（气溶胶/尘埃/水汽）：与瑞利互补的另一半 —— **近全谱白**（所以能把蓝调
;;   压成灰白、给亮区铺一层朦胧的薄雾，去掉"塑料感/空气过于澄澈"）、**强前向散射**
;;   （Henyey-Greenstein g=0.65，太阳在镜头背后时最亮）、柱密度随空气量²增长（比瑞利的
;;   ³ 缓，所以白雾铺满整个亮区而不只是描边）。
;;   相函数是**帧常量**：平行入射 + 平行投影 ⇒ 盘面处处同一个散射角 cosΘ = sUp。
(define MIE_G 0.65)                 ; 各向异性因子
(define MIE_ISO 0.60)               ; 各向同性底（多次散射的粗近似）
(define MIE_A 0.035)                ; 柱密度项（随空气量²增长，主要压在边缘）
(define MIE_FLAT 0.014)             ; **平坦项**：气溶胶垂直光学厚度带来的均匀白纱
                                    ;   （整片亮区都被抬一点、颜色被压灰 → 朦胧感）
(define MIE_PHASE 1.0)              ; 相函数（每帧由 SU 更新）

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

;; ---- 运行状态 --------------------------------------------------------------

(define rnd_y ROW_Y0)               ; 下一待绘制行
(define rnd_active 0)               ; 1 = 本场景尚未画完
(define hud_dim 0)                  ; 1 = HUD 已隐藏
(define clk_sec -1)                 ; 上次绘制的 UTC 秒（-1 = 待重画）
(define scen_sod 0.0)               ; 上次场景开始的 UTC 秒
(define dt_s 0.0)                   ; 场景计时临时量
(define civ_y 2026)                 ; 民用历（HUD）
(define civ_m 1)
(define civ_d 1)
;; 时间机器：[−2分] [+2分] [实时] [暂停/运行]
;;   运行态的显示时刻 = 实时 UTC + off_s（所以"暂停一段时间再恢复"会自然积累成偏置，
;;   画面里的晨昏线也就一直停留在你离开时的那一刻 —— 直到按『实时』）。
(define off_s 0)                    ; 相对实时的偏置（秒）
(define paused 0)                   ; 1 = 时间冻结
(define sim_d 0)                    ; 模拟 UTC 日数
(define sim_s 0)                    ; 模拟 UTC 当日秒
(define scene_dirty 0)              ; 1 = 立刻重算场景（改时刻后置位）
(define tprev 0)                    ; 触屏上一拍是否按住
(define tpx 0)                      ; 触点（块内不能 define）
(define tpy 0)
(define bi -1)                      ; 命中的按钮下标
(define clouds_on 1)                ; 1 = 显示云（第 5 个按钮可切）
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

;; ---- 时间机器 --------------------------------------------------------------

;; 把 sim_s 归一化到 [0, 86400)，并进位/借位到 sim_d
(define time_norm
  (lambda ()
    (while (>= sim_s 86400) {
      (set! sim_s (- sim_s 86400))
      (set! sim_d (+ sim_d 1))
    })
    (while (< sim_s 0) {
      (set! sim_s (+ sim_s 86400))
      (set! sim_d (- sim_d 1))
    })))

;; 运行态：模拟时刻 ← 实时 + 偏置（暂停时不调用，时刻冻结）
(define time_sync
  (lambda ()
    (set! sim_d (Time.utc_days))
    (set! sim_s (+ (Time.utc_sod) off_s))
    (time_norm)))

;; 步进 d 秒（±120）：暂停时直接挪冻结的时刻，运行态挪偏置
(define time_step
  (lambda (d)
    (if (== paused 1)
        {
          (set! sim_s (+ sim_s d))
          (time_norm)
        }
        {
          (set! off_s (+ off_s d))
          (time_sync)
        })
    (set! scene_dirty 1)))

;; 从暂停恢复：把"冻结期间积累的时间差"折进偏置，于是从冻结那一刻继续走
(define time_resume
  (lambda ()
    (set! off_s (+ (* (- sim_d (Time.utc_days)) 86400)
                   (- sim_s (Time.utc_sod))))
    (set! paused 0)
    (time_sync)
    (set! scene_dirty 1)))

;; 回到实时：偏置清零 + 解除暂停
(define time_live
  (lambda ()
    (set! off_s 0)
    (set! paused 0)
    (time_sync)
    (set! scene_dirty 1)))

;; 诊断行（e2e 断言时刻确实被改过；每次按钮动作打一行）
(define time_log
  (lambda ()
    (display "[earthport] time day=") (display sim_d)
    (display " sod=") (display sim_s)
    (display " off=") (display off_s)
    (display " paused=") (display paused)
    (newline)))

;; ---- 底部控制条（4 个按钮；右下角「退出」由 app/lib/exit.scm 画）-----------

(define BTN_Y 219)
(define BTN_H 18)
(define BTN_W 50)
(define BTN_GAP 3)
(define BTN_N 5)

(define btn_x (lambda (i) (+ 4 (* i (+ BTN_W BTN_GAP)))))

(define btn_label
  (lambda (i)
    (if (== i 0) "-2分"
        (if (== i 1) "+2分"
            (if (== i 2) "实时"
                (if (== i 3) (if (== paused 1) "运行" "暂停")
                    (if (== clouds_on 1) "有云" "无云")))))))

;; 高亮：暂停键在冻结时高亮；『实时』在"正在实时"时高亮；快进快退不高亮
(define btn_active
  (lambda (i)
    (if (== i 2) (if (and (== paused 0) (== off_s 0)) 1 0)
        (if (== i 3) paused
            (if (== i 4) clouds_on 0)))))

(define draw_btns
  (lambda ()
    (define i 0)
    (define x 0)
    (define lab "")
    (define lx 0.0)
    (while (< i BTN_N) {
      (set! x (btn_x i))
      (set! lab (btn_label i))
      (set! lx (+ x (* (- BTN_W (text_w lab)) 0.5)))
      (if (== (btn_active i) 1)
          {
            (Gfx.rounded_rect x BTN_Y BTN_W BTN_H 4 4 4 4 108 168 232 1)
            (Gfx.rounded_rect (+ x 1) (+ BTN_Y 1) (- BTN_W 2) (- BTN_H 2) 3 3 3 3 96 152 216 1)
            (Gfx.draw_text lab lx (+ BTN_Y 3) 8 16 28)
          }
          {
            (Gfx.rounded_rect x BTN_Y BTN_W BTN_H 4 4 4 4 52 74 104 1)
            (Gfx.rounded_rect (+ x 1) (+ BTN_Y 1) (- BTN_W 2) (- BTN_H 2) 3 3 3 3 24 30 46 1)
            (Gfx.draw_text lab lx (+ BTN_Y 3) 186 208 232)
          })
      (set! i (+ i 1))
    })))

;; 触点落在哪个按钮上（不在控制条里返回 -1）
(define btn_hit
  (lambda (px py)
    (define r -1)
    (define i 0)
    (if (and (>= py BTN_Y) (< py (+ BTN_Y BTN_H)))
        {
          (while (< i BTN_N) {
            (if (and (>= px (btn_x i)) (< px (+ (btn_x i) BTN_W)))
                (set! r i)
                #f)
            (set! i (+ i 1))
          })
        }
        #f)
    r))

(define btn_do
  (lambda (i)
    (if (== i 0) (time_step -120) #f)
    (if (== i 1) (time_step 120) #f)
    (if (== i 2) (time_live) #f)
    (if (== i 3)
        (if (== paused 1)
            (time_resume)
            (set! paused 1))
        #f)
    ;; 云层开关：画面内容变了，立刻重算一帧场景
    (if (== i 4)
        {
          (set! clouds_on (if (== clouds_on 1) 0 1))
          (set! scene_dirty 1)
          (display "[earthport] clouds=") (display clouds_on) (newline)
        }
        #f)
    (set! clk_sec -1)
    (draw_btns)
    (draw_clock)
    (draw_hud_static)
    (Gfx.refresh)
    (time_log)))

;; ---- 星历：太阳方向 --------------------------------------------------------

(define sun_update
  (lambda ()
    (define hdeg 0.0)
    (define dr 0.0)
    (define hr 0.0)
    (define cd 0.0)
    (define hl 0.0)
    (define hg 0.0)
    (define hg1 0.0)
    (Ephem.set_time sim_d sim_s)
    (Ephem.set_gmst)
    (Ephem.sun_equatorial)
    (Ephem.civil sim_d)
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
    ;; 米氏相函数：HG(g) 归一化到"太阳在镜头正后方(=1)"，再与各向同性底混合
    (set! hg (/ (- 1.0 (* MIE_G MIE_G))
                (Math.pow (- (+ 1.0 (* MIE_G MIE_G))
                             (* (* 2.0 MIE_G) SU)) 1.5)))
    (set! hg1 (/ (- 1.0 (* MIE_G MIE_G)) (Math.pow (- 1.0 MIE_G) 1.5)))
    (set! MIE_PHASE (+ MIE_ISO (* (- 1.0 MIE_ISO) (/ hg hg1))))
  ))

;; ---- 场景绘制（分条渐进） --------------------------------------------------

(define rnd_start
  (lambda ()
    (display "[earthport] scene start") (newline)
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
    (define lk 0.0)
    (define sunv 0.0)
    (define dl 0.0)
    (define datm 0.0)
    (define sfac 0.0)
    (define dusk 0.0)
    (define airs 0.0)
    (define warm 0.0)
    (define mie 0.0)
    (define ca 0.0)
    (define sh 0.0)
    (define csun 0.0)
    (define cl 0.0)
    (define lit 0.0)
    (define air 0.0)
    (define s3 0.0)
    (define sl 0.0)
    (define sa 0.0)
    (define ar 0.0)
    (define ag 0.0)
    (define ab 0.0)
    (define cov 0.0)
    (define bo 0.0)
    (define xo 0.0)
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
                    (set! cr (- (String.charCodeAt idx TXR) 256.0))
                    (set! cg (- (String.charCodeAt idx TXG) 256.0))
                    (set! cb (- (String.charCodeAt idx TXB) 256.0))
                    ;; ---- 地表：太阳漫射（余弦定律）+ 城市灯（随暮色渐亮），**一式到底** ----
                    ;;   sunv = max(0,dot) 只是"直射"的余弦；dot<0 时它连续地变为 0，
                    ;;   残余反照率 SUN_FLOOR 在两侧取同一个值 ⇒ 晨昏线上亮度连续。
                    ;;   色相也连续：没有任何"昼/夜两套系数"。
                    (set! sunv (if (> dot 0.0) dot 0.0))
                    (set! sfac (+ SUN_FLOOR
                                  (* (- 1.0 SUN_FLOOR) (/ sunv (+ sunv SUN_SOFT)))))
                    (set! k (* sfac (+ 0.44 (* 0.56 w))))      ; 视线角暗化（limb darkening）
                    (set! r (* cr k))
                    (set! g (* cg k))
                    (set! b (* cb k))
                    ;; 城市灯：dot 从 DUSK_HI 降到 DUSK_HI-DUSK_SPAN 之间连续点亮
                    (set! dusk (/ (- DUSK_HI dot) DUSK_SPAN))
                    (if (< dusk 0.0) (set! dusk 0.0) #f)
                    (if (> dusk 1.0) (set! dusk 1.0) #f)
                    (if (> dusk 0.0)
                        {
                          (set! nc (String.charCodeAt idx TXN))
                          (set! nv (if (> nc 511.0) (- nc 512.0) (- nc 256.0)))
                          (set! lk (* nv dusk))
                          (set! r (+ r (* lk 1.28)))
                          (set! g (+ g (* lk 0.98)))
                          (set! b (+ b (* lk 0.62)))
                        }
                        #f)
                    ;; 海面太阳反光：半角向量 nh，nh≤0 处光斑自然为 0（连续）
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
                    ;; ---- 云层合成（在边缘羽化之前，于是云也随地表一起在视圆盘边缘淡出）----
                    ;;   两层：低云（团块，随厚度由灰白到纯白）+ 高空卷云（薄、丝缕、略偏冷）
                    (set! ca 0.0)
                    (if (== clouds_on 1)
                        (set! ca (* (- (String.charCodeAt idx TXW) 256.0) CLOUD_K))
                        #f)
                    (if (> ca 0.0)
                        {
                          ;; 云体亮度随"云量/厚度"变化：薄云/边缘发灰、厚云顶纯白
                          ;; —— 只有单一 alpha 通道时，这一项是唯一的"体积感"来源。
                          ;; 云顶高度带来的"晚变暗"：照明里把太阳高度抬 CLOUD_LIFT（≈2°）
                          (set! csun (+ sunv CLOUD_LIFT))
                          (set! cl (* (* (+ SUN_FLOOR (* (- 1.0 SUN_FLOOR)
                                                 (/ csun (+ csun SUN_SOFT))))
                                         (+ 0.62 (* 0.38 w)))
                                      (+ CLOUD_BASE (* CLOUD_DENSE ca))))
                          (set! r (+ (* r (- 1.0 ca)) (* cl ca)))
                          (set! g (+ (* g (- 1.0 ca)) (* cl ca)))
                          (set! b (+ (* b (- 1.0 ca)) (* (* cl 1.03) ca)))
                        }
                        #f)
                    (set! ca 0.0)
                    (if (== clouds_on 1)
                        (set! ca (* (- (String.charCodeAt idx TXC) 256.0) CIRRUS_K))
                        #f)
                    (if (> ca 0.0)
                        {
                          (set! cl (* (* (+ SUN_FLOOR (* (- 1.0 SUN_FLOOR)
                                                 (/ csun (+ csun SUN_SOFT))))
                                         (+ 0.70 (* 0.30 w)))
                                      CIRRUS_ALB))
                          (set! r (+ (* r (- 1.0 ca)) (* cl ca)))
                          (set! g (+ (* g (- 1.0 ca)) (* cl ca)))
                          (set! b (+ (* b (- 1.0 ca)) (* (* cl 1.05) ca)))
                        }
                        #f)
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
                    ;; 大气照明用**切点法向的太阳投影** dl = u·sEast + sinLat·sNorth
                    ;;（**不含 w 项**）。理由：决定这段大气亮不亮的是"视线掠射路径上
                    ;;  最近点的法向"，而球面在天顶方向每像素只转 0.6°、在视圆盘边缘却
                    ;;  一转就是 6°（球面与视线相切）——用含 w 的地表法向做照明，亮环内侧
                    ;;  必然出现 1px 的亮度断层；dl 跨边缘连续且平滑。
                    (set! dl (+ (* u SE) (* sny SN)))
                    ;; 照明/入射气柱用的天顶角余弦：取 "地表法向" 与 "切点法向" 的较大者
                    ;;   datm = dl + max(0, w·sUp)（= max(dot, dl)）
                    ;;   · 盘面内部（w 大）：= dot —— 散射层就在地表附近，太阳高度取地表值
                    ;;   · 视圆盘边缘（w→0）：= dl —— 与球外连续，且不会因球面相切而 1px 跳变
                    (set! datm dl)
                    (if (> SU 0.0) (set! datm (+ dl (* w SU))) #f)
                    (set! lit (* (+ datm ATM_LIFT) ATM_LIFT_K))
                    (if (< lit 0.0) (set! lit 0.0) #f)
                    (if (> lit 1.0) (set! lit 1.0) #f)
                    ;; 拆成两股：**被阳光照到的那股**（沿视线看到的日照大气）与**环境那股**
                    ;; （气辉/多次散射）。只有前者按"入射阳光穿过的气柱长度"红化并衰减，
                    ;; 后者保持蓝 —— 否则夜半球的边缘会被红化成灰，而真实轨道照片里它仍是蓝的。
                    ;;   airs = 太阳方向空气量（1/sqrt(cos²+2H)：天顶 1、晨昏 ~4.2）
                    ;;   warm = 蓝的保留率（1/(1+0.3(airs-1))：天顶 1 → 晨昏 ~0.51）
                    ;;   透射率取 warm³ ⇒ 晨昏线附近既偏暖又明显变暗，不会与正午亮环同亮。
                    (set! sunv (if (< datm 0.0) 0.0 datm))
                    (set! airs (/ 1.0 (Math.sqrt (+ (* sunv sunv) HS2))))
                    (set! warm (/ 1.0 (+ 1.0 (* SUN_RED (- airs 1.0)))))
                    (set! sl (* (* (* s3 (- 1.0 ATM_AMB)) lit) (* warm (* warm warm))))
                    (set! sa (* s3 ATM_AMB))
                    ;; 米氏白雾：与瑞利同源衰减（入射光谱），但**三通道等权** ⇒
                    ;;   叠在蓝调之上会把亮区压向灰白、并抬起暗部（朦胧感）。
                    (set! mie (* (* (+ MIE_FLAT (* MIE_A (* air air))) (* lit MIE_PHASE))
                                warm))
                    (set! ar (+ (+ (* (* sl ATM_R) (- 2.0 warm)) (* sa ATM_R)) mie))
                    (set! ag (+ (+ (* (* sl ATM_G) (+ 0.4 (* 0.6 warm))) (* sa ATM_G)) mie))
                    (set! ab (+ (+ (* (* sl ATM_B) warm) (* sa ATM_B)) mie))
                    (Gfx.set_pixel x y
                                   (+ r (/ (* 255.0 ar) (+ ar 1.0)))
                                   (+ g (/ (* 255.0 ag) (+ ag 1.0)))
                                   (+ b (/ (* 255.0 ab) (+ ab 1.0))))
                  }
                  { ; ==================== 球外：只有大气（柔和边缘） ====================
                    ;; 球外只有大气：柱密度从 b=1 处的值起，按三次缓出平滑归零
                    ;;   用 exp(-u/H)³ 曾让辉光每 1px 掉一半（1/e 尺度只剩 3px），
                    ;;   肉眼就是"一圈硬边的亮环"；三次窗在整个 HALO_U（≈12px）上
                    ;;   连续衰减且末端斜率为 0 ⇒ 边缘化开、与球内在 b=1 严格连续。
                    (set! bo (- (Math.sqrt q) 1.0))
                    (set! xo (- 1.0 (/ bo HALO_U)))
                    (if (< xo 0.0) (set! xo 0.0) #f)
                    (set! s3 (* S3L (* xo (* xo xo))))
                    (set! u (* dxf RINV))
                    (set! dl (+ (* u SE) (* sny SN)))   ; 缘上法向的太阳投影（可为负）
                    (set! lit (* (+ dl ATM_LIFT) ATM_LIFT_K))
                    (if (< lit 0.0) (set! lit 0.0) #f)
                    (if (> lit 1.0) (set! lit 1.0) #f)
                    (set! sunv (if (< datm 0.0) 0.0 datm))
                    (set! airs (/ 1.0 (Math.sqrt (+ (* sunv sunv) HS2))))
                    (set! warm (/ 1.0 (+ 1.0 (* SUN_RED (- airs 1.0)))))
                    (set! sl (* (* (* s3 (- 1.0 ATM_AMB)) lit) (* warm (* warm warm))))
                    (set! sa (* s3 ATM_AMB))
                    (set! air (* AIR0 xo))          ; 与球内在 b=1 处衔接的等效空气量
                    (set! mie (* (* (+ MIE_FLAT (* MIE_A (* air air))) (* lit MIE_PHASE))
                                warm))
                    (set! ar (+ (+ (* (* sl ATM_R) (- 2.0 warm)) (* sa ATM_R)) mie))
                    (set! ag (+ (+ (* (* sl ATM_G) (+ 0.4 (* 0.6 warm))) (* sa ATM_G)) mie))
                    (set! ab (+ (+ (* (* sl ATM_B) warm) (* sa ATM_B)) mie))
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

;; 单行文本宽度（12px 字模：半角 7px、全角 12px）—— 自算比调 draw_text_centered 可靠
(define text_w
  (lambda (s)
    (define i 0)
    (define n (String.length s))
    (define w 0)
    (while (< i n) {
      (if (> (String.charCodeAt i s) 255.0) (set! w (+ w 12)) (set! w (+ w 7)))
      (set! i (+ i 1))
    })
    w))

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
    (hud_text s (- xr (text_w s)) y nch)))

(define draw_clock
  (lambda ()
    (define sec0 0.0)
    (define min0 0.0)
    (define hr0 0.0)
    (define str "")
    (if (== hud_dim 1)
        #f
        {
          (set! sec0 (mod sim_s 60.0))
          (set! min0 (mod (/ (- sim_s sec0) 60.0) 60.0))
          (set! hr0 (/ (- (- sim_s sec0) (* min0 60.0)) 3600.0))
          (set! str (String.concat "UTC " (String.concat (pad2 hr0)
                          (String.concat ":" (String.concat (pad2 min0)
                          (String.concat ":" (pad2 sec0)))))))
          (hud_text str 6 3 13)
        })))

;; 时间状态徽标（右上第二行）：暂停 → "暂停"；有偏置 → ±h:mm；实时且运行 → 空
(define draw_badge
  (lambda ()
    (define s "")
    (define mins 0.0)
    (define oh 0.0)
    (define om 0.0)
    (define av 0.0)
    (Gfx.fill_rect (- W 60) 17 60 15 BG_R BG_G BG_B)
    (set! av off_s)
    (if (< av 0.0) (set! av (- 0.0 av)) #f)
    (if (== paused 1)
        (set! s "暂停")
        (if (> av 0.0)
            {
              (set! mins (Math.floor (/ av 60.0)))
              (set! oh (Math.floor (/ mins 60.0)))
              (set! om (- mins (* oh 60.0)))
              (set! s (String.concat (if (< off_s 0.0) "-" "+")
                       (String.concat (String.atom_to_string oh)
                       (String.concat ":" (pad2 om)))))
            }
            #f))
    (if (> (String.length s) 0)
        (Gfx.draw_text s (- (- W 9) (text_w s)) 18 226 186 110)
        #f)))

(define draw_hud_static
  (lambda ()
    (define dstr "")
    (define sdec "")
    (if (== hud_dim 1)
        #f
        {
          (set! dstr (String.concat (String.atom_to_string civ_d)
                        (String.concat " " (String.concat (get_item MONTHS (- civ_m 1))
                        (String.concat " " (String.atom_to_string civ_y))))))
          (set! sdec (String.concat "DEC " (String.concat
                        (if (>= DEC_DEG 0.0) "+" "-")
                        (String.concat (num_fix (Math.abs DEC_DEG) 1) "°"))))
          (hud_text dstr 6 18 12)
          (hud_text_r sdec (- W 6) 3 11)
          (draw_badge)
        })))

;; 只擦 HUD 文字所在的小矩形（不碰球/辉光/刻度环）
(define clear_hud
  (lambda ()
    (Gfx.fill_rect 4 1 112 16 BG_R BG_G BG_B)
    (Gfx.fill_rect 4 16 96 16 BG_R BG_G BG_B)
    (Gfx.fill_rect (- W 100) 1 96 16 BG_R BG_G BG_B)
    (Gfx.fill_rect (- W 60) 16 60 16 BG_R BG_G BG_B)))

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
                      (set! tpx (Touch.x))
                      (set! tpy (Touch.y))
                      (set! bi (btn_hit tpx tpy))
                      (if (>= bi 0)
                          (btn_do bi)           ; 控制条：交给按钮（不切换 HUD）
                          (toggle_hud))         ; 其它区域：显示/隐藏 HUD 文字
                    }
                    #f)
                (set! tprev 1)
              }
              (set! tprev 0))
        }
        #f)))

;; 键盘：← → 步进 2 分钟，↑ 回到实时，↓ 暂停/运行，空格/回车 切换 HUD
(define read_key
  (lambda ()
    (if (Key.is_available)
        {
          (set! keyv (Key.take_press))
          (while (> keyv 0)
            {
              (if (== keyv 3) (btn_do 0) #f)
              (if (== keyv 4) (btn_do 1) #f)
              (if (== keyv 1) (btn_do 2) #f)
              (if (== keyv 2) (btn_do 3) #f)
              (if (or (== keyv 99) (== keyv 67)) (btn_do 4) #f)
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
    (EX.draw_button_at 266 220)
    (draw_btns)
    ;; 先补 HUD（球体由随后的分条绘制补），退出确认框关掉后立刻恢复原样
    (draw_clock)
    (draw_hud_static)
    (set! clk_sec -1)
    (set! rnd_y ROW_Y0)
    (set! rnd_active 1)
    (Gfx.refresh)))

(define on_tick
  (lambda ()
    (EX.check)                          ; 硬 A/ESC / 右下角「退出」
    (read_touch)
    (read_key)
    (if (== paused 0) (time_sync) #f)    ; 运行态：模拟时刻 = 实时 + 偏置
    ;; 时钟：每秒一次（暂停时 sim_s 不动，自然不重画）
    (if (== sim_s clk_sec)
        #f
        {
          (set! clk_sec sim_s)
          (draw_clock)
        })
    ;; 场景调度：空闲且（距上次场景超过 SCENE_S 模拟秒 或 时刻被改过）→ 重算
    (if (== rnd_active 0)
        {
          (set! dt_s (- sim_s scen_sod))
          (if (< dt_s 0.0) (set! dt_s (+ dt_s 86400.0)) #f)
          (if (or (>= dt_s SCENE_S) (== scene_dirty 1))
              {
                (set! scen_sod sim_s)
                (set! scene_dirty 0)
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
(time_live)                 ; 初始：显示实时
(draw_bg)
(draw_stars)
(EX.draw_button_at 266 220)
(draw_btns)
(rnd_start)                 ; 第一帧场景（分条渐进，由定时器逐拍推进）
(set! scene_dirty 0)        ; time_live 置过的"待重算"已由上面这一帧满足
(set! scen_sod sim_s)
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
