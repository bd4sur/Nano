;@id=earthport
;@title=地球舷窗
;@version=2026-11-27
;; ============================================================================
;; 地球舷窗（Earth Port）—— 灵机应用程序
;;
;; 一扇"空间站舷窗"：观察者悬在地球**任意一点的正上方**俯视球心，**旋转角任意**
;;（画面"上方"指向哪由 view_roll 决定，0 = 正北朝上）。画面里的地球就是此刻真实的
;; 那一面 —— 晨昏线的位置与倾角由**当前 UTC 时间**经星历解算得到，夜半球的城市灯光
;; 落在晨昏线的夜侧，大气散射的蓝色边缘与海面太阳反光随太阳方位实时变化。
;; 云层（低云 + 高空卷云）可开关；不做极光/闪电（按需求范围）。
;;
;; 画面构成（320x240）：
;;   · 地表：NASA Blue Marble（无云）1° 网格 —— 数据在 app/lib/earth_tex.scm（生成物）
;;   · 夜灯：NASA Black Marble 1° 网格 —— 同一份贴图的第四通道（含水面标记）
;;   · 大气：**视线方向的空气柱密度**（Kasten 型空气量）+ 大气自身照明 → 色调映射加色。
;;     天顶方向只有淡淡一层蓝，掠射方向空气量陡增 ⇒ 视圆盘亮区内部是宝石蓝的散射光，
;;     最外圈过曝成青白，圆盘边缘"化开"（大气厚度按视觉需求夸大，HS=0.033R）。
;;   · 极光：南北**地磁极**周围的极光椭圆（绿色 557.7nm / 红色 630.0nm），用三层同心
;;     发射壳（86/201/277km）近似幕帘的**垂直结构** —— 侧视时是下绿上红、有厚度、
;;     能高出地平线的幕墙，俯视时是略微展开的椭圆环。只在暗侧可见（极昼看不到）。
;;     详见下方「极光」常量块。
;;   · 反光：海面（贴图蓝通道占优处）按半角向量做镜面高光，光斑位置即真实太阳反光点
;;   · 星空：几十颗暗星（只画在辉光窗之外，重绘地球擦不到）
;;   · 空间站：中国空间站（CSS / 天和核心舱，NORAD 48274）的**轨道环 + 此刻的真实位置**。
;;     两行根数（TLE）内嵌在 app/lib/orbit.scm，位置**完全本地推算**（开普勒方程 +
;;     J2 长期摄动，不联网），随时刻推进；时间机器拖动时也跟着走。
;;     轨道按**真实物理比例**绘制：与地球同一把尺（1 px ≈ 70.9 km），所以 388 km 的
;;     轨道高度在画面上只有 5.5 px —— 轨道环就贴着地球边缘外面一小圈（这是真实比例，
;;     不是画小了）。被地球挡住的弧段不画（正交投影：天顶分量 < 0 且在视盘内）。
;;     左下角给出推算值：距地高度 / 轨道速度 / 星下点大地纬经度。
;;   · HUD：左上角**北京日期 + 时刻**（YYYY-MM-DD 换行 hh:mm:ss，左对齐）；
;;     底部信息行：左侧**视角读数**（纬度 N/S、经度 E/W、滚转角 R±），右端时间徽标
;;     （暂停或 ±h:mm，琥珀色；时间机器的键位仍在，只是不再占按钮）
;;   · 底部控制条（5 枚）：[-15°] [+15°] [复位] [有云/无云] [空间站]
;;   · 右上角：「退出」（app/lib/exit.scm 画在 (270,3)）
;;
;; 相机（视点在任意点、任意滚转角）：
;;   观察者在 (φ0, λ0) 正上方沿径向俯视球心（正交投影，视圆盘半径固定 = R 像素）。
;;   设屏幕右向/上向单位向量 = (cosρ·E - sinρ·N, sinρ·E + cosρ·N)（ρ = 滚转角，
;;   E/N = 视点局部系的东/北），则屏幕偏移 (dx,dy) 处：
;;       p  = dx/R,  qs = -dy/R,  q = p² + qs²
;;       xE =  p·cosρ + qs·sinρ      球面点法向的"东"分量
;;       xN = -p·sinρ + qs·cosρ      球面点法向的"北"分量
;;       xU = sqrt(1 - q)            天顶分量（正对观察者；q≤1 时在球面上）
;;   ⇒  sinLat = sinφ0·xU + cosφ0·xN          （贴图行号 ∝ sinLat，省掉 asin）
;;       Δλ     = atan2(xE, cosφ0·xU - sinφ0·xN)（经度偏移，贴图列 = λ + 60）
;;   φ0=0、λ0=120°E、ρ=0 时它逐式退化为原来的定点公式（u→xE、sinLat→xN、w→xU）。
;;   **着色内核全部在视点局部系里做**（dot = n·s、(e,n) 切平面分量、半角向量），
;;   所以只需要把太阳方向投影到当前视点的东/北/天：
;;     · 太阳赤纬 δ、赤经 α ← (Ephem.sun_equatorial)；GMST ← (Ephem.set_gmst)
;;     · 次太阳点经度 λs = α - GMST，与视点的经度差 Δ = λs - λ0
;;     · sEast = cosδ·sinΔ,  sNorth = -sinφ0·cosδ·cosΔ + cosφ0·sinδ,
;;       sUp = cosφ0·cosδ·cosΔ + sinφ0·sinδ
;;   逐像素 dot = n·s 就是太阳天顶角余弦：>0 昼、<0 夜；dot=0 的大圆在屏幕上的投影
;;   即晨昏线（一个椭圆），dot∈[-0.2,0] 是暮光过渡带。
;;
;; 渲染节奏（为什么不是每帧整屏重算）：一次整球绘制约 3 万像素、约 600 万条 VM 指令
;;   （wasm 上实测 ≈0.35s 纯解释执行）。而太阳在屏幕上每分钟只移动 0.4 像素，所以：
;;     · 场景（球体/大气）每 SCENE_S=30 秒重算一次，且**分条渐进**绘制
;;       （每拍画 STRIP 行推一次屏），一次重绘跨 ~20 拍完成，看不出"刷"的感觉；
;;     · 只有左上角时钟每秒钟重画两行字（fill_rect + draw_text，脏区极小）。
;;   稳态 CPU 占用几个百分点（实测 2.6%，含空间站图层）。
;;   空间站图层另有一条轻路径：标记在屏幕上只走 0.09 px/s（92.3 分钟绕一圈 ~600 px），
;;   所以每 ST_UPD_TK 拍（≈10 s）只把**标记前后两个包围盒的并集**那一小片重画一遍
;;   （窗内跑着色内核、窗外填背景渐变），不必等 30 s 的整帧重算 —— 见 st_patch。
;;   轨道环是 1px **抗锯齿**线（亮黄）：抗锯齿不是幂等的，所以它每帧只在"像素刚好是
;;   背景状态"的位置画一次 —— 分条重绘期间不画，收尾时擦掉球外旧环再整条画一次
;;   （见 st_ring / st_erase_ring 的注释）。
;;   拖动时不能等 0.35s/帧：视点一动就先出**粗预览**（每 ST_PREV=4 像素采一个样、
;;   4×4 像素块一次 fill_rect 填同色，像素量 1/16 ⇒ 整帧约 33ms ≈ 30 帧/秒，够跟手）；
;;   松手后再走上面那条整质量的分条渐进路径（cam_settle 倒计时到 0 才开始）。
;;
;; 时间机器（**按钮已去掉，只留键盘**：←/→ = ∓2 分钟、↑ = 实时、↓ = 暂停/运行）：
;;   运行态的显示时刻 = 实时 UTC + off_s；暂停冻结时刻（sim_d/sim_s 不再跟随实时），
;;   再按「运行」时把"冻结期间积累的时间差"折进 off_s —— 于是画面从你冻结的那一刻继续。
;;   状态显示在底部信息行右端的琥珀色徽标里（暂停 / ±h:mm）。
;;   任何时刻变化都置 scene_dirty，让下一拍立刻重算场景（晨昏线跟着跳）。
;;
;; 时区：左上角是**北京时间**（UTC+8，与设备本地时区无关），日期由共享星历的
;;   (Ephem.civil) 按北京日算出，跨日自动进位。
;;
;; 输入：
;;   · 触屏空白处**拖动** → 移动视点（Google Earth 手感：手指往哪拖，地面就往哪走；
;;     水平方向按 1/cos(φ0) 补偿，高纬处经线收敛不打滑；竖直方向拖到 ±90° 处截住）；
;;     **点按**（位移 < 3px）= 显示/隐藏 HUD 文字。按下落在按钮上则不拖动。
;;   · 底部 [-15°]/[+15°] 滚转画面（点按 ±15°，按住每 120ms 连发一档，**在键上横向滑动
;;     = 拨轮连续滚转**，所以旋转角是任意的）、[复位] 回 0°N 120°E 正北朝上、
;;     [空间站] 切换空间站图层（轨道环 + 位置标记 + 左下角推算值）的显隐。
;;   · 键盘 a/d 经度、w/s 纬度、`,`/`.` 滚转、r 复位（各 5°/15° 一档），
;;     ←/→ = ∓2 分钟，↑ = 实时，↓ = 暂停/运行，c = 云层，i = 空间站图层，
;;     空格/回车 = HUD 开关；
;;     硬 A/ESC 与**右上角「退出」**按钮 → 全局退出确认框（app/lib/exit.scm）。
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

(import EX "app/lib/exit.scm")      ; 硬 A/ESC + 右上角「退出」按钮 → 全局退出确认框
(import Ephem "app/lib/ephem.scm")  ; 太阳赤纬/赤经、GMST、民用历
(import Orb "app/lib/orbit.scm")    ; 空间站 TLE + 本地轨道推算（开普勒 + J2）
(import ET "app/lib/earth_tex.scm") ; 地表四通道贴图（生成物，见 tools/gen_earth_tex.py）

;; ---- 版面 ------------------------------------------------------------------

(define W (Gfx.width))              ; 320
(define H (Gfx.height))             ; 240
(define CX 160.0)                   ; 球心（略高于正中：底部让出两行控制条）
(define CY 104.0)
(define RAD 90.0)                   ; 球半径（像素）
(define RINV (/ 1.0 RAD))           ; 1/R
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

;; 渲染行范围与每拍行数（球 + 外圈，底部让出两行控制条）
(define ROW_Y0 10)
(define ROW_Y1 199)
(define STRIP 10)                   ; 整质量：每拍画 10 行
(define ST_PREV 4.0)                ; 粗预览采样步长（像素）：像素量 1/16 ⇒ 实测 ≈33ms/整帧

;; 贴图取址（列数/行数取自生成模块，单一事实来源）
(define TCOLS ET.TW)                ; 361：列数（含环绕复制列）
(define ROWH (/ (- ET.TH 1) 2.0))   ; 90：行号 = (1 - sinLat) * ROWH
(define ROWLAST (- ET.TH 1))        ; 180：最大行号（贴图越界保护）
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

;; 极光（南北**地磁极**处的极光椭圆；三层球壳 = 垂直结构）
;;
;; 物理依据与近似：
;;   · 位置：极光椭圆绕**地磁轴**而不是地理轴（地磁北极 IGRF 偶极轴 2025 历元在
;;     80.7°N / 72.7°W，与地理轴差 ~9.3°）。南北椭圆关于地磁轴共轭对称，所以
;;     用同一个 |cos(与地磁轴的夹角)| 就能同时覆盖两极：椭圆峰值在磁纬 67.5°
;;     （地磁余纬 22.5°，静稳-中等活动量级）。
;;   · **垂直结构**：极光是沿磁力线悬挂的"幕帘"，从 ~90km 一直延伸到 ~300km，
;;     不是贴在地面上的零厚度光影。这里用**三层同心发射壳**近似这条垂直剖面：
;;       L1  1.0165R（105km）  绿 557.7nm  O(¹S) 最亮、最窄，下缘锐（赤道侧收窄）
;;       L2  1.0305R（194km）  绿青（绿线的上段 + 少量蓝紫）  过渡层
;;       L3  1.0425R（271km）  红 630.0nm  O(¹D) 弥散、最宽、上缘柔
;;     视线（正交投影下方向恒为视轴）分别与三层球壳求交，命中点的地磁纬决定是否
;;     落在椭圆上。三层在屏幕上的径向错位 ≈1.3px：**侧视（近地平线）时是一堵有
;;     厚度、下绿上红、上缘化开的幕墙**；俯视时是略微展开的椭圆环。
;;     三层都受渲染窗（1.045R）限制 —— 真实红层可到 400km，这里截到 271km。
;;   · 可见性：极光只在**暗**天空可见 —— 用发射层处的太阳天顶角余弦门控（dot>0 时
;;     完全没有），所以极昼一侧不会出现极光（与真实一致：夏季极盖区看不到）。
;;   · 沿椭圆的疏密与折叠：磁赤道面两根基向量在视点系的投影给出磁经度的前两阶
;;     谐波，既用来让椭圆中心线起伏（折叠/弧带），也用来做亮暗结（亚暴鼓包）。
(define MAG_LAT 80.7)               ; 地磁北极纬度（度）
(define MAG_LON -72.7)              ; 地磁北极经度（度，东经为正）
(define AUR_R1 1.0135)              ; 层 1（86km，绿，下缘锐）
(define AUR_R2 1.0315)              ; 层 2（201km，绿青）
(define AUR_R3 1.0435)              ; 层 3（277km，红，弥散上缘；受渲染窗 1.045R 限制）
(define AUR_R1Q (* AUR_R1 AUR_R1))
(define AUR_R2Q (* AUR_R2 AUR_R2))
(define AUR_R3Q (* AUR_R3 AUR_R3))
(define AUR_I1 (/ 1.0 AUR_R1))
(define AUR_I2 (/ 1.0 AUR_R2))
(define AUR_I3 (/ 1.0 AUR_R3))
(define AUR_C1 (Math.cos (* 22.6 (/ PI 180.0))))   ; 各层椭圆中心（磁纬 67.4/67.9/68.6°）
(define AUR_C2 (Math.cos (* 22.1 (/ PI 180.0))))
(define AUR_C3 (Math.cos (* 21.4 (/ PI 180.0))))
(define AUR_W1 0.029)               ; 各层带宽（cos 空间；≈±4.4°/±6.1°/±11° 磁纬）
(define AUR_W2 0.040)
(define AUR_W3 0.075)               ; 红层最宽（越往高处幕帘越发散）
(define AUR_V1 (/ 1.0 AUR_W1))      ; 预算倒数（逐像素不做除法）
(define AUR_V2 (/ 1.0 AUR_W2))
(define AUR_V3 (/ 1.0 AUR_W3))
(define AUR_ASY 1.25)               ; 层 1 不对称度：赤道侧更陡（幕帘下缘硬边朝赤道）
(define AUR_WAVS 0.011)             ; 折叠波幅（cos 空间；三层同时位移）
(define AUR_D0 0.02)                ; 白天门控：dot > -0.02 开始变暗
(define AUR_DK 8.0)                 ; 门控斜率（dot < -0.145 时全黑）
(define AUR_NB 0.55)                ; 磁午夜侧的亮度下限（极盖内仍随太阳高度变亮）
(define AUR_G1 92.0)                ; 层 1 峰值亮度（0..255 加色）
(define AUR_G2 46.0)                ; 层 2
(define AUR_G3 50.0)                ; 层 3（红）峰值亮度（弥散，所以给得比绿层高）
;; 预筛窗口：外层 ±2.4 带宽（再留折叠波/内层偏移的余量）——用 |cos|² 判，
;; 南北共轭对称所以一个窗口覆盖两极，且省掉 abs 与除法
(define AUR_PLO (- AUR_C3 (* 2.4 AUR_W3)))
(define AUR_PHI (+ AUR_C3 (* 2.4 AUR_W3)))
(define AUR_PL (* AUR_PLO AUR_PLO))
(define AUR_PH (* AUR_PHI AUR_PHI))

;; 空间站图层（中国空间站 CSS / 天和）：绘制参数
;;
;; 比例：与地球同一把尺 —— 地球半径 RAD 像素 = RE_EQ 6378.137 km ⇒ 1 px ≈ 70.9 km。
;;   轨道半长轴 6766 km ⇒ 半径 95.5 px：**只比地球视半径大 5.5 px**。这就是真实
;;   比例的后果（388 km / 6378 km = 6.1%），所以轨道环看起来“贴在地球边缘外面”
;;   而不会飘得很远 —— 不要为了好看把它画大。
;; 刷新：标记的屏幕移速 ≈ 0.09 px/s（92.3 分钟绕一圈 ~600 px），所以整帧场景
;;   （30 s）之间只需重画**标记占用的一小片**（见 st_patch），左下角数值每秒刷一次。
;; 线型：1px 抗锯齿（亮黄）。抗锯齿按覆盖率混合、**不幂等**，所以环在每条绘制路径上
;;   都只画一次（详见 st_ring 上方的说明）。
(define ST_N 180)                   ; 轨道环采样数（每段 ≈3.3px，首尾闭合）
(define ST_UPD_TK 500)              ; 每多少拍刷新一次图层位置（≈10 s；标记只走
                                    ;   0.09 px/s，一步不到 1 px，看着是连续移动）
(define ST_PAD 9)                   ; 标记包围盒半边长（光晕 r=7 + 抗锯齿）
(define ORB_R 255)                    ; 轨道环颜色：**亮黄**。按覆盖率不透明混合（mode 1）：
(define ORB_G 232)                    ;   暗底上是一条鲜黄的细线，压在地球亮面上也不会
(define ORB_B 60)                     ;   像加色混合那样过曝成白光
(define DEGR (/ PI 180.0))          ; 度 → 弧度（着色内核只用 DEG）
(define ST_CARD_X 6)                ; 左下角信息卡：左边界
(define ST_CARD_Y 160)              ; 首行墨迹顶（三行：160 / 174 / 188）

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
(define CAM_SETTLE 10)              ; 视角改动后等多少拍（实际一拍 ≈20~35ms）再画整质量
(define REPEAT_DELAY 400)           ; 滚转键按住多久开始连发（ms）
(define REPEAT_GAP 120)             ; 连发间隔（ms）
(define CAM_ROLL 15.0)              ; 滚转键一档（度）
(define CAM_KEY 5.0)                ; 键盘平移一档（度）

;; ---- 视点（相机）-----------------------------------------------------------
;;
;; 观察者在 (view_lat, view_lon) 正上方沿径向俯视球心（正交投影，视圆盘半径固定）。
;; 屏幕像素 → 球面法向（视点局部系：东 E / 北 N / 天 U）：
;;     p  = dx/R,  qs = -dy/R,  q = p² + qs²
;;     xE =  p·cosρ + qs·sinρ        (ρ = view_roll)
;;     xN = -p·sinρ + qs·cosρ
;;     xU = sqrt(1 - q)              正对观察者；q≤1 表示落在球面上
;;   sinLat = sinφ0·xU + cosφ0·xN,   Δλ = atan2(xE, cosφ0·xU - sinφ0·xN)
;; φ0=0、λ0=120°E、ρ=0 时与原来的定点公式逐式相同（xE→u、xN→sinLat、xU→w），
;; 因此默认视角下的画面与“定点俯视”版本逐位一致。
;;
;; 着色内核全部在这个**随视点旋转的**局部系里做（dot = n·s、切平面分量、半角向量），
;; 所以视点一变只需重算 sun_local 里的三个分量，逐像素代码不必知道纬度/经度。
(define view_lat 0.0)               ; 视点纬度（度，-90..90）
(define view_lon 120.0)             ; 视点经度（度，0..360）
(define view_roll 0.0)              ; 画面滚转角（度，0..360；0 = 正北朝上）
(define vlat_s 0.0)                 ; sin/cos 视点纬度（view_update 里预算）
(define vlat_c 1.0)
(define RC RINV)                    ; RINV·cosρ（屏幕切平面 → 局部系）
(define RS 0.0)                     ; RINV·sinρ
(define RSn 0.0)                    ; -RINV·sinρ
(define lonbase 180.0)              ; 贴图列基准 = view_lon + 60（列中心经度 = 列 + 60）
(define cam_k (/ DEG RAD))          ; 拖动一像素对应的角位移（度）
(define cam_dirty 0)                ; 1 = 视角变了，待重画
(define cam_settle 0)               ; 整质量重绘的倒计时（拍）
(define dragging 0)                 ; 1 = 拖动手势进行中
(define acc_x 0.0)                  ; 拖动累计位移（尚未应用）
(define acc_y 0.0)
(define drag_moved 0)               ; 1 = 本次按下已超过点按阈值（不再算点击）
(define in_btn 0)                   ; 1 = 本次按下落在按钮上（不拖动）
(define last_x 0.0)                 ; 上一拍触点
(define last_y 0.0)
(define press_ms 0)                 ; 按钮按下时刻（ms，System.timestamp）
(define hold_ms 0)                  ; 按住已持续时长（ms）
(define rep_done 0)                 ; 已连发的档数（滚转键）
(define dial0 0.0)                  ; 按下滚转键时的滚转角（拨轮手势的起点）
(define DIAL_K 0.5)                 ; 拨轮灵敏度：滑动一像素转多少度

;; ---- 太阳与世界状态 --------------------------------------------------------

(define SE 0.0)                     ; 太阳方向（视点局部系：东/北/天）
(define SN 0.0)
(define SU -1.0)
(define HX 0.0)                     ; 半角向量（太阳 + 视线 (0,0,1)）
(define HY 0.0)
(define HZ 1.0)
(define SUNUP 0.0)                  ; max(0, SU)
(define DEC_DEG 0.0)                ; 太阳赤纬（度，HUD）
(define SDR 0.0)                    ; 太阳赤纬（弧度）
(define SLON 0.0)                   ; 次太阳点经度 λs = α - GMST（度，不做环绕）
;; 地磁极 + 磁赤道面两根基向量，都投影在视点局部系（东/北/天）里，每帧算一次
(define ME 0.0) (define MN 0.0) (define MU 1.0)      ; 地磁北极方向
(define PE 0.0) (define PN 0.0) (define PU 0.0)      ; 磁赤道基向量 1（磁经度 0°）
(define QE 0.0) (define QN 0.0) (define QU 0.0)      ; 磁赤道基向量 2（磁经度 90°）

;; ---- 运行状态 --------------------------------------------------------------

(define rnd_y ROW_Y0)               ; 下一待绘制行
(define rnd_mode 0)                 ; 0 = 空闲，1 = 整质量分条绘制中
(define hud_dim 0)                  ; 1 = HUD 已隐藏
(define clk_sec -1)                 ; 上次绘制时钟的 sim_s（-1 = 待重画）
(define scen_sod 0.0)               ; 上次场景开始的 UTC 秒
(define dt_s 0.0)                   ; 场景计时临时量
(define bj_y 2026)                  ; 北京日期的年/月/日（HUD，由 bj_sync 算）
(define bj_m 1)
(define bj_dd 1)
(define bj_d 0)                     ; 北京日（1970 起的天数）
(define bj_s 0.0)                   ; 北京当日秒数
(define ui_dirty 0)                 ; 1 = 时间机器/云层状态变了，待补画 HUD
;; 时间机器：[−2分] [+2分] [实时] [暂停/运行]
;;   运行态的显示时刻 = 实时 UTC + off_s（所以"暂停一段时间再恢复"会自然积累成偏置，
;;   画面里的晨昏线也就一直停留在你离开时的那一刻 —— 直到按『实时』）。
(define off_s 0)                    ; 相对实时的偏置（秒）
(define paused 0)                   ; 1 = 时间冻结
(define sim_d 0)                    ; 模拟 UTC 日数
(define sim_s 0)                    ; 模拟 UTC 当日秒
(define scene_dirty 0)              ; 1 = 立刻重算场景（改时刻后置位）
(define tprev 0)                    ; 触屏上一拍是否按住
(define bi -1)                      ; 命中的按钮下标
(define clouds_on 1)                ; 1 = 显示云（底部第 4 枚按钮 / 键位 c）
(define keyv 0)                     ; Key.take_press 临时量
(define _timer 0)                   ; 定时器编号（消费 set_interval 的返回值）

;; 空间站图层的运行状态
(define st_on 1)                    ; 1 = 显示（第 5 枚按钮 / 键位 i）
(define st_tk 0)                    ; 拍计数（到 ST_UPD_TK 刷新一次图层）
(define st_t 0.0)                   ; 图层采样基准时刻（相对 TLE 历元的秒数）
(define st_dt 0.0)                  ; 采样步长（秒）
(define st_cr 1.0)                  ; cos/sin(滚转角)：每帧算一次，逐点复用
(define st_sr 0.0)
(define st_mx 0.0)                  ; 标记屏幕坐标（当前）
(define st_my 0.0)
(define st_mv 0)                    ; 1 = 标记可见（未被地球遮挡）
(define st_bx0 0) (define st_by0 0) ; 已画标记的包围盒（st_patch 据此擦除）
(define st_bx1 0) (define st_by1 0)
(define st_nx0 0) (define st_ny0 0) ; 本次重算出来的新包围盒
(define st_nx1 0) (define st_ny1 0)
(define st_dd 0)                    ; 上一次完整画完图层时的时刻（判"时刻大跳"用）
(define st_ds 0.0)
(define PROJ_X 0.0)                 ; proj_pt 的输出（屏幕坐标 + 天顶分量）
(define PROJ_Y 0.0)
(define PROJ_U 0.0)
(define st_lx '())                  ; 轨道环屏幕坐标缓存（启动时 push 一次，之后 set_item!）
(define st_ly '())
(define st_lv '())                  ; 1 = 该采样点可见

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
      ;; 只取辉光窗（101.5px）之外、且避开 HUD 两条文字带与两行控制条的点；
      ;; 左下角留给空间站信息卡（那块会被每秒重画，星点被擦掉就补不回来）
      (if (and (> dd 108.0)
               (and (> yy 33.0) (< yy 196.0))
               (not (and (< xx 102.0) (> yy 150.0))))
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
    (set! scene_dirty 1)
    (set! ui_dirty 1)))

;; 从暂停恢复：把"冻结期间积累的时间差"折进偏置，于是从冻结那一刻继续走
(define time_resume
  (lambda ()
    (set! off_s (+ (* (- sim_d (Time.utc_days)) 86400)
                   (- sim_s (Time.utc_sod))))
    (set! paused 0)
    (time_sync)
    (set! scene_dirty 1)
    (set! ui_dirty 1)))

;; 回到实时：偏置清零 + 解除暂停
(define time_live
  (lambda ()
    (set! off_s 0)
    (set! paused 0)
    (time_sync)
    (set! scene_dirty 1)
    (set! ui_dirty 1)))

;; 暂停/运行切换（键位 ↓；按钮已去掉）
(define time_toggle_pause
  (lambda ()
    (if (== paused 1)
        (time_resume)
        (set! paused 1))
    (set! ui_dirty 1)))

;; 云层开关（底部第 4 枚按钮 / 键位 c）：画面内容变了，立刻重算一帧场景
(define clouds_toggle
  (lambda ()
    (set! clouds_on (if (== clouds_on 1) 0 1))
    (set! scene_dirty 1)
    (set! ui_dirty 1)
    (display "[earthport] clouds=") (display clouds_on) (newline)))

;; 诊断行（e2e 断言时刻确实被改过；每次按钮动作打一行）
(define time_log
  (lambda ()
    (display "[earthport] time day=") (display sim_d)
    (display " sod=") (display sim_s)
    (display " off=") (display off_s)
    (display " paused=") (display paused)
    (newline)))

;; ---- 底部控制条（5 枚按钮；右上角「退出」由 app/lib/exit.scm 画在 (270,3)）----
;;
;; 索引：0 = [-15°]、1 = [+15°]、2 = [复位]、3 = [有云/无云]、4 = [空间站]。
;; 按下落在任一按钮上就不拖动；滚转键（0/1）按住 400ms 后每 120ms 连发一档，
;; 在其上**横向滑动 = 拨轮**（连续滚转，任意角度）。时间机器的 -2/+2/实时/暂停
;; 按钮已去掉，只留键位（←/→/↑/↓），状态显示在底部信息行右端的徽标里。

(define BTN_Y 219)
(define BTN_H 18)
(define BTN_W 60)                   ; 5×60 + 4×4 = 316（两侧各留 4px）
(define BTN_GAP 4)
(define BTN_N 5)
(define INFO_Y 203)                 ; 底部信息行（视角读数 + 时间徽标）：203..216

(define btn_x (lambda (i) (+ 4 (* i (+ BTN_W BTN_GAP)))))

(define btn_label
  (lambda (i)
    (if (== i 0) "-15°"
        (if (== i 1) "+15°"
            (if (== i 2) "复位"
                (if (== i 3) (if (== clouds_on 1) "有云" "无云")
                    "空间站"))))))

;; 高亮：『复位』在视点已回到默认（0°N 120°E 正北朝上）时高亮；
;; 『有云/无云』高亮 = 当前有云；『空间站』高亮 = 图层可见；滚转键不高亮
(define btn_active
  (lambda (i)
    (if (== i 2)
        (if (and (and (== view_lat 0.0) (== view_lon 120.0)) (== view_roll 0.0)) 1 0)
        (if (== i 3) clouds_on
            (if (== i 4) st_on 0)))))

;; 一枚按钮（高亮版 / 常态）
(define draw_btn
  (lambda (x y w h lab act)
    (define lx (+ x (* (- w (text_w lab)) 0.5)))
    (if (== act 1)
        {
          (Gfx.rounded_rect x y w h 4 4 4 4 108 168 232 1)
          (Gfx.rounded_rect (+ x 1) (+ y 1) (- w 2) (- h 2) 3 3 3 3 96 152 216 1)
          (Gfx.draw_text lab lx (+ y 3) 8 16 28)
        }
        {
          (Gfx.rounded_rect x y w h 4 4 4 4 52 74 104 1)
          (Gfx.rounded_rect (+ x 1) (+ y 1) (- w 2) (- h 2) 3 3 3 3 24 30 46 1)
          (Gfx.draw_text lab lx (+ y 3) 186 208 232)
        })))

(define draw_btns
  (lambda ()
    (define i 0)
    (while (< i BTN_N) {
      (draw_btn (btn_x i) BTN_Y BTN_W BTN_H (btn_label i) (btn_active i))
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
    (if (== i 0) (cam_roll (- 0.0 CAM_ROLL)) #f)
    (if (== i 1) (cam_roll CAM_ROLL) #f)
    (if (== i 2) (cam_reset) #f)
    (if (== i 3) (clouds_toggle) #f)
    (if (== i 4) (st_toggle) #f)
    ;; 高亮/读数可能变了：补画控制条与信息行（画面本身由下一拍处理）
    (draw_btns)
    (draw_info)
    (Gfx.refresh)))

;; ---- 星历：太阳方向 --------------------------------------------------------

;; 只跟“时刻”有关的量：赤纬、次太阳点经度、民用历（每次重建场景算一遍）
(define sun_ephem
  (lambda ()
    (Ephem.set_time sim_d sim_s)
    (Ephem.set_gmst)
    (Ephem.sun_equatorial)
    (set! DEC_DEG Ephem.s_dec)
    (set! SDR (* Ephem.s_dec (/ PI 180.0)))
    ;; 次太阳点经度 λs = α - GMST。**不做环绕**：后面只用 sin/cos(λs - λ0)，
    ;; 环绕只会引入浮点误差（且会让默认视角的画面与旧版差一两个像素）。
    (set! SLON (- Ephem.s_ra Ephem.e_gmst))))

;; 太阳方向投影到**当前视点**的局部系（东/北/天）。视点一变就要重算。
;;   Δ = λs - λ0；  sE = cosδ·sinΔ
;;   sN = -sinφ0·cosδ·cosΔ + cosφ0·sinδ
;;   sU =  cosφ0·cosδ·cosΔ + sinφ0·sinδ
;; φ0=0、λ0=120°E 时退化为旧版的 (-cosδ sinH, sinδ, cosδ cosH)（H = 120°-λs）。
(define sun_local
  (lambda ()
    (define dlt 0.0)
    (define cdl 0.0)
    (define sdl 0.0)
    (define sd 0.0)
    (define hl 0.0)
    (define hg 0.0)
    (define hg1 0.0)
    (set! dlt (* (- SLON view_lon) (/ PI 180.0)))
    (set! sd (Math.sin SDR))
    (set! cdl (* (Math.cos SDR) (Math.cos dlt)))
    (set! sdl (* (Math.cos SDR) (Math.sin dlt)))
    (set! SE sdl)
    (set! SN (+ (* (- 0.0 vlat_s) cdl) (* vlat_c sd)))
    (set! SU (+ (* vlat_c cdl) (* vlat_s sd)))
    (set! hl (Math.sqrt (+ (* SE SE) (+ (* SN SN) (* (+ SU 1.0) (+ SU 1.0))))))
    ;; 太阳正对镜头背面（SU = -1）时半角向量退化，钳一下防除零
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
    (set! MIE_PHASE (+ MIE_ISO (* (- 1.0 MIE_ISO) (/ hg hg1))))))

;; 地磁极方向与磁赤道面基向量 → 视点局部系（东/北/天）。与 sun_local 同一套旋转：
;;   Δ = λm - λ0；p = (cosφm·sinΔ, -sinφ0·cosφm·cosΔ + cosφ0·sinφm,
;;                     cosφ0·cosφm·cosΔ + sinφ0·sinφm)
;; 磁赤道基向量取"极点所在经线的东西向 e"与 e×p（沿经线向北）：
;;   E' = cosΔ, N' = -sinφ0·sinΔ, U' = cosφ0·sinΔ                     （东西向）
;;   E' = -sinφm·sinΔ, N' = sinφ0·sinφm·cosΔ + cosφ0·cosφm,
;;   U' = -cosφ0·sinφm·cosΔ + sinφ0·cosφm                             （沿经线）
(define aurora_update
  (lambda ()
    (define dlt 0.0)
    (define cl 0.0)
    (define sl 0.0)
    (define cx 0.0)
    (define sy 0.0)
    (set! dlt (* (- MAG_LON view_lon) (/ PI 180.0)))
    (set! cl (Math.cos (* MAG_LAT (/ PI 180.0))))
    (set! sl (Math.sin (* MAG_LAT (/ PI 180.0))))
    (set! cx (* cl (Math.cos dlt)))
    (set! sy (Math.sin dlt))
    (set! ME (* cl sy))
    (set! MN (+ (* (- 0.0 vlat_s) cx) (* vlat_c sl)))
    (set! MU (+ (* vlat_c cx) (* vlat_s sl)))
    (set! PE (Math.cos dlt))
    (set! PN (* (- 0.0 vlat_s) sy))
    (set! PU (* vlat_c sy))
    (set! QE (* (- 0.0 sl) sy))
    (set! QN (+ (* (* vlat_s sl) (Math.cos dlt)) (* vlat_c cl)))
    (set! QU (+ (* (* (- 0.0 vlat_c) sl) (Math.cos dlt)) (* vlat_s cl)))))

;; ---- 空间站图层（CSS / 天和）------------------------------------------------
;;
;; 轨道环：卫星在**地固系**里的闭合轨迹（一个周期采样 ST_N 点）。比例与地球完全
;;   一致（RAD 像素 = 6378.137 km），所以它不是“装饰性的一个大圈”——它就在地球
;;   边缘外 5.5 px 处。位置标记：此刻的推算位置（发光点 + 一个细圈）；在地球背面
;;   （天顶分量 < 0 且投影落在视盘内）的弧段不画。
;; 数据/算法：app/lib/orbit.scm（内嵌 TLE、开普勒推进 + J2 长期项，纯本地计算）。
;; 刷新：整帧场景每 30 s 一次；图层本身另按 ST_UPD_TK 拍重画一小片（见 st_patch）——
;;   标记每秒才走 0.09 px，没必要每拍重算。

;; 缓存槽：启动时一次性 push（模块级 '() 是静态槽，只 push 一次）
(define mkc_orb
  (lambda ()
    (define k 0)
    (while (< k ST_N) {
      (push st_lx 0.0)
      (push st_ly 0.0)
      (push st_lv 0)
      (set! k (+ k 1))
    })))

;; 地固纬经度（度）+ 半径因子 kf（1 = 地球表面）→ 屏幕坐标（写 PROJ_X/PROJ_Y/PROJ_U）。
;;   与着色内核同一套旋转：屏幕切平面按滚转角旋到视点局部系（东/北/天），
;;   再取东/北两个分量正交投影（天顶分量 U 只用于遮挡判定）。
(define proj_pt
  (lambda (latd lond kf)
    (define d 0.0)
    (define cl 0.0)
    (define sl 0.0)
    (define cd 0.0)
    (define sd 0.0)
    (define ee 0.0)
    (define nn 0.0)
    (set! d (* (- lond view_lon) DEGR))
    (set! cl (Math.cos (* latd DEGR)))
    (set! sl (Math.sin (* latd DEGR)))
    (set! cd (Math.cos d))
    (set! sd (Math.sin d))
    (set! ee (* kf (* cl sd)))
    (set! nn (* kf (+ (* (- 0.0 vlat_s) (* cl cd)) (* vlat_c sl))))
    (set! PROJ_X (+ CX (* RAD (- (* ee st_cr) (* nn st_sr)))))
    (set! PROJ_Y (- CY (* RAD (+ (* ee st_sr) (* nn st_cr)))))
    (set! PROJ_U (* kf (+ (* vlat_c (* cl cd)) (* vlat_s sl))))))

;; 正交投影下的遮挡：在地球背面（U<0）且投影点落在视盘内（距球心 < RAD）
(define st_hidden
  (lambda (x y u)
    (and (< u 0.0)
         (< (+ (* (- x CX) (- x CX)) (* (- y CY) (- y CY))) (* RAD RAD)))))

;; 重建图层缓存 + 标记位置（每帧场景开始时、每次图层刷新时各一次）
(define st_begin
  (lambda ()
    (define k 0)
    (define kf 0.0)
    (define x 0.0)
    (define y 0.0)
    (Orb.set_time sim_d sim_s)
    (set! st_t Orb.orb_t)
    (set! st_dt (/ Orb.PERIOD ST_N))
    (set! st_cr (Math.cos (* view_roll DEGR)))
    (set! st_sr (Math.sin (* view_roll DEGR)))
    (set! k 0)
    (while (< k ST_N) {
      (Orb.ring (+ st_t (* k st_dt)))   ; 环上每点都用**当前** GMST（见 orbit.scm）
      (set! kf (/ Orb.orb_r Orb.RE_EQ))
      (proj_pt Orb.orb_lat Orb.orb_lon kf)
      (set! x PROJ_X)
      (set! y PROJ_Y)
      (set_item! st_lx k x)
      (set_item! st_ly k y)
      (if (st_hidden x y PROJ_U) (set_item! st_lv k 0) (set_item! st_lv k 1))
      (set! k (+ k 1))
    })
    ;; 标记 = 采样点 0（此刻）；顺带算出它的包围盒（st_patch 据此擦旧、画新）
    (set! st_mx (get_item st_lx 0))
    (set! st_my (get_item st_ly 0))
    (set! st_mv (get_item st_lv 0))
    (set! st_nx0 (Math.floor (- st_mx ST_PAD)))
    (set! st_ny0 (Math.floor (- st_my ST_PAD)))
    (set! st_nx1 (+ (Math.floor (+ st_mx ST_PAD)) 1))
    (set! st_ny1 (+ (Math.floor (+ st_my ST_PAD)) 1))))

;; 整帧路径（分条重绘 / 退出确认框关闭后的整屏重画）画完图层后，把"已画包围盒"
;; 同步为当前标记的包围盒 —— 否则第一次局部重画会拿初值 0 去并集，一擦擦到整屏。
(define st_mark_b
  (lambda ()
    (set! st_bx0 st_nx0) (set! st_bx1 st_nx1)
    (set! st_by0 st_ny0) (set! st_by1 st_ny1)))

;; 一段轨道线，按行裁剪到 [ya, yb) 后**抗锯齿**绘制（1px、不透明混合）。
;; 为什么不用 (Gfx.set_clip) 限制范围：裁剪矩形只对 draw_point / fill_rect / 圆角矩形 /
;; draw_line 生效，而抗锯齿线走的是 blend_pixel（光斑走 add_pixel）——对它们无效。
;; 所以局部重画改用这里的**参数化行裁剪**：解出 y(t) ∈ [ya,yb) 的那一段再画。
(define st_line_clip
  (lambda (x0 y0 x1 y1 ya yb)
    (define dx 0.0)
    (define dy 0.0)
    (define ta 0.0)
    (define tb 0.0)
    (define t0 0.0)
    (define t1 0.0)
    (set! dx (- x1 x0))
    (set! dy (- y1 y0))
    (if (== dy 0.0)
        (if (and (>= y0 (* 1.0 ya)) (< y0 (* 1.0 yb)))
            (Gfx.draw_line_aa_mode x0 y0 x1 y1 1.0 ORB_R ORB_G ORB_B 1)
            #f)
        {
          (set! ta (/ (- ya y0) dy))
          (set! tb (/ (- yb y0) dy))
          (set! t0 ta)
          (set! t1 tb)
          (if (> t0 t1) { (set! t0 tb) (set! t1 ta) } #f)
          (if (< t0 0.0) (set! t0 0.0) #f)
          (if (> t1 1.0) (set! t1 1.0) #f)
          (if (>= t0 t1)
              #f
              (Gfx.draw_line_aa_mode (+ x0 (* t0 dx)) (+ y0 (* t0 dy))
                                     (+ x0 (* t1 dx)) (+ y0 (* t1 dy))
                                     1.0 ORB_R ORB_G ORB_B 1))
        })))

;; 轨道环：只画两端都可见的弧段（被地球挡住的不画）。
;;
;; 抗锯齿线**不是幂等**的（按覆盖率混合，同一处画两遍就更实一分），所以本图层每帧
;; 只在"像素刚好是背景状态"的位置画一次：
;;   · 场景整帧：分条重绘期间不画，`rnd_finish` 里擦一遍球外旧环再整条画一次；
;;   · 粗预览（拖动）：每帧先擦旧环 → 粗预览重画球内 → 整条画一次；
;;   · 局部重画（标记）：只擦标记那一小片，环按行裁剪只补这一片。
;; 1px 的环在 320x240 上是技术线条，抗锯齿让它在亮地球上不再是硬边台阶。
(define st_ring
  (lambda (ya yb)
    (define k 0)
    (define k2 0)
    (if (== st_on 1)
        {
          (set! k 0)
          (while (< k ST_N) {
            (set! k2 (+ k 1))
            (if (>= k2 ST_N) (set! k2 0) #f)
            (if (and (== (get_item st_lv k) 1) (== (get_item st_lv k2) 1))
                (st_line_clip (get_item st_lx k) (get_item st_ly k)
                              (get_item st_lx k2) (get_item st_ly k2) ya yb)
                #f)
            (set! k (+ k 1))
          })
        }
        #f)))

;; 当前位置标记：一层加色光斑 + 一个细准星圈 + 亮核（画在环之后）。
;; 注意它**不是**幂等的（光斑按 alpha 叠加）：每次都要先擦干净再画一次。
;;
;; 卫星在地球背面时（vis=0）**什么都不画** —— 真实遮挡就是看不见；位置本身仍然有
;; 左下角的三行推算值（距地/速度/星下点）可读。（早期版本在这里画过一个很暗的空心圈
;; 当"被遮挡"提示，但它等于把卫星画穿了地球，已去掉。）
(define st_marker
  (lambda ()
    (if (and (== st_on 1) (== st_mv 1))
        {
          (Gfx.fill_glow_aa_add st_mx st_my 7.0 70 170 220 0.55)
          (Gfx.draw_circle (Math.round st_mx) (Math.round st_my) 5 46 120 152)
          (Gfx.fill_dot_aa st_mx st_my 2.0 210 246 255)
        }
        #f)))

;; 只把矩形 [x0,x1)×[y0,y1) 里**落在渲染窗之外**的部分填回背景渐变（窗内不动，
;; 留给着色内核重画）。球外除了背景没有别的内容：星点半径 > 108px，而轨道环
;; 最多到 ~97px（含抗锯齿溢出），所以"背景 = 那条渐变公式"在这里是精确的。
(define st_bg_out
  (lambda (x0 y0 x1 y1)
    (define y y0)
    (define dy 0.0)
    (define dy2 0.0)
    (define xe 0.0)
    (define xa 0.0)
    (define xb 0.0)
    (define la 0.0)
    (define lb 0.0)
    (while (< y y1) {
      (set! dy (- (* 1.0 y) CY))
      (set! dy2 (* dy dy))
      (set! xa 0.0)
      (set! xb 0.0)                       ; 窗宽 0 ⇒ 整行都在窗外
      (if (< dy2 XMAX2)
          {
            (set! xe (Math.sqrt (- XMAX2 dy2)))
            (set! xa (- CX xe))
            (set! xb (+ CX xe))
          }
          #f)
      ;; 左段 [x0, xa)
      (set! la (* 1.0 x0))
      (set! lb xa)
      (if (> lb (* 1.0 x1)) (set! lb (* 1.0 x1)) #f)
      (if (> lb la) (bg_fill la y (- (Math.floor lb) la) 1) #f)
      ;; 右段 [xb, x1]
      (set! la xb)
      (if (< la (* 1.0 x0)) (set! la (* 1.0 x0)) #f)
      (if (> (* 1.0 x1) la)
          (bg_fill (Math.floor la) y (+ 1 (- x1 (Math.floor la))) 1)
          #f)
      (set! y (+ y 1))
    })))

;; 擦掉**上一次画出的**轨道环的球外部分（沿旧路径把那一带填回背景渐变）。
;;
;; 逐段擦：包围盒左右上下各留 1px（抗锯齿的溢出像素落在线的两侧），并且**只填渲染窗
;; 之外**那一侧（窗内留给着色内核重画）。旧版只擦"两端都在窗外"的整段，跨地球边缘的
;; 那几段球外部分永远擦不掉 —— 拖动视点后就在地球边缘留下一圈残影（本次修的就是它）。
;; 必须在 st_begin（它会覆盖缓存）之前调用。
(define st_erase_ring
  (lambda ()
    (define k 0)
    (define k2 0)
    (define x0 0.0)
    (define y0 0.0)
    (define x1 0.0)
    (define y1 0.0)
    (define lx 0.0)
    (define hx 0.0)
    (define ly 0.0)
    (define hy 0.0)
    (while (< k ST_N) {
      (set! k2 (+ k 1))
      (if (>= k2 ST_N) (set! k2 0) #f)
      (if (and (== (get_item st_lv k) 1) (== (get_item st_lv k2) 1))
          {
            (set! x0 (get_item st_lx k))
            (set! y0 (get_item st_ly k))
            (set! x1 (get_item st_lx k2))
            (set! y1 (get_item st_ly k2))
            (set! lx x0)
            (set! hx x1)
            (if (< x1 x0) { (set! lx x1) (set! hx x0) } #f)
            (set! ly y0)
            (set! hy y1)
            (if (< y1 y0) { (set! ly y1) (set! hy y0) } #f)
            (st_bg_out (Math.floor (- lx 1.0))
                       (Math.floor (- ly 1.0))
                       (+ (Math.floor (+ hx 1.0)) 1)
                       (+ (Math.floor (+ hy 1.0)) 1))
          }
          #f)
      (set! k (+ k 1))
    })))

;; 擦掉标记的球外部分（球内那部分由场景重绘/粗预览覆盖）
(define st_erase_marker
  (lambda ()
    (st_bg_out st_bx0 st_by0 st_bx1 st_by1)))

;; 信息卡的擦除矩形（三行，宽度比文字略宽）。**卡片整体落在轨道环内侧的空隙里**
;; （三行右边界 78/82/96；而环在那三行上分别落在 x≈81/93/113 之外），所以擦除
;; 蹭不到环，也就不必为它重跑着色内核。
(define st_card_erase
  (lambda ()
    (bg_fill ST_CARD_X (- ST_CARD_Y 2) 72 16)
    (bg_fill ST_CARD_X (+ ST_CARD_Y 12) 76 16)
    (bg_fill ST_CARD_X (+ ST_CARD_Y 26) 90 18)))

;; 左下角推算值：距地高度 / 轨道速度 / 星下点（大地纬经度）。数值**每秒**重算一次
;; （Orb.set_time 只有一次开普勒迭代 + 十几次三角函数，比画三行字便宜）。
(define st_card
  (lambda ()
    (define ala 0.0)
    (define alo 0.0)
    (define s "")
    (st_card_erase)
    (if (and (== st_on 1) (== hud_dim 0))
        {
          (Orb.set_time sim_d sim_s)
          (set! s (String.concat "距地 " (String.concat
                     (String.atom_to_string (Math.floor (+ Orb.orb_alt 0.5))) "km")))
          (Gfx.draw_text s ST_CARD_X ST_CARD_Y HUD_R HUD_G HUD_B)
          (set! s (String.concat "v " (String.concat (num_fix Orb.orb_v 2) "km/s")))
          (Gfx.draw_text s ST_CARD_X (+ ST_CARD_Y 14) 150 208 236)
          (set! ala (Math.abs Orb.orb_latg))
          (set! alo Orb.orb_lon)
          (if (> alo 180.0) (set! alo (- 360.0 alo)) #f)
          (set! s (num_fix ala 1))
          (set! s (String.concat s (if (>= Orb.orb_latg 0.0) "N " "S ")))
          (set! s (String.concat s (num_fix alo 1)))
          (set! s (String.concat s (if (> Orb.orb_lon 180.0) "W" "E")))
          (Gfx.draw_text s ST_CARD_X (+ ST_CARD_Y 28) HUD_R HUD_G HUD_B)
        }
        #f)))

;; 把 [y0,y1) 行、[x0,x1] 列那一带重画一遍：着色窗之外填背景渐变（与 draw_bg
;; 同一条带公式）、窗内跑同一套着色内核 —— 于是旧图层被真正擦掉，而不是拿一块纯色
;; 去盖。**只填标记那一带的列**：背景重填是不透明的、会顺手擦掉落在里面的星点/按钮
;; 边缘，范围越小越安全。
(define st_repaint
  (lambda (y0 y1 x0 x1)
    (define y y0)
    (define dy 0.0)
    (define dy2 0.0)
    (define xe 0.0)
    (define xa 0.0)
    (define xb 0.0)
    (define la 0.0)
    (define lb 0.0)
    (while (< y y1) {
      (set! dy (- (* 1.0 y) CY))
      (set! dy2 (* dy dy))
      (set! xa 0.0)
      (set! xb (* 1.0 W))
      (if (< dy2 XMAX2)
          {
            (set! xe (Math.sqrt (- XMAX2 dy2)))
            (set! xa (- CX xe))
            (set! xb (+ CX xe))
          }
          #f)
      (if (< xa 0.0) (set! xa 0.0) #f)
      (if (> xb (* 1.0 W)) (set! xb (* 1.0 W)) #f)
      ;; 左段 [x0, xa) 与右段 [xb, x1] 在着色窗之外
      (set! la x0)
      (set! lb xa)
      (if (> lb x1) (set! lb x1) #f)
      (if (> lb la) (bg_fill la y (- (Math.floor lb) la) 1) #f)
      (set! la xb)
      (if (< la x0) (set! la x0) #f)
      (if (< x1 la) (set! la x1) #f)
      (if (> x1 la) (bg_fill (Math.floor la) y (+ 1 (- x1 (Math.floor la))) 1) #f)
      (set! y (+ y 1))
    })
    (rnd_draw y0 y1 1.0)))

;; 矩形相交（用于判定"这一带碰到了哪块 UI"）
(define rect_hit
  (lambda (x0 y0 x1 y1 rx0 ry0 rx1 ry1)
    (and (< x0 rx1) (and (> x1 rx0) (and (< y0 ry1) (> y1 ry0))))))

;; 图层刷新：只重画标记前后两个包围盒的并集那一小片（整帧场景 ~30 s 才重算一次）。
;; 行内先擦（st_repaint）→ 裁到这一带重画图层 → 补被这一带碰到的 UI。
;; 注意：按钮/退出按钮的圆角与抗锯齿边缘**不是**幂等绘制（重复画会越描越深），
;; 所以只有真的碰到才补画；控制条（y≥219）永远不会被碰到，一律不碰。
(define st_patch
  (lambda ()
    (define y0 0)
    (define y1 0)
    (define x0 0)
    (define x1 0)
    (define oy0 st_by0)
    (define oy1 st_by1)
    (define ox0 st_bx0)
    (define ox1 st_bx1)
    (st_begin)
    (set! y0 st_ny0)
    (if (< oy0 y0) (set! y0 oy0) #f)
    (set! y1 st_ny1)
    (if (> oy1 y1) (set! y1 oy1) #f)
    (set! x0 (- st_nx0 2))
    (if (< (- ox0 2) x0) (set! x0 (- ox0 2)) #f)
    (set! x1 (+ st_nx1 2))
    (if (> (+ ox1 2) x1) (set! x1 (+ ox1 2)) #f)
    (if (< y0 0) (set! y0 0) #f)
    (if (> y1 H) (set! y1 H) #f)
    (if (< x0 0) (set! x0 0) #f)
    (if (> x1 W) (set! x1 W) #f)
    (st_repaint y0 y1 x0 x1)        ; 这一带先精确擦回背景状态
    (st_ring y0 y1)                 ; 环只补这一带（AA 线不受裁剪区约束 ⇒ 行裁剪）
    (st_marker)
    (if (rect_hit x0 y0 x1 y1 4 156 100 204) (st_card) #f)
    (if (rect_hit x0 y0 x1 y1 268 0 320 20)
        (if (== hud_dim 0) (EX.draw_button_at 270 3) #f)
        #f)
    (if (rect_hit x0 y0 x1 y1 4 200 320 218)
        (if (== hud_dim 0) (draw_info) #f)
        #f)
    (if (rect_hit x0 y0 x1 y1 4 0 84 34)
        (if (== hud_dim 0) (draw_clock) #f)
        #f)
    (Gfx.refresh)
    (set! st_bx0 st_nx0) (set! st_bx1 st_nx1)
    (set! st_by0 st_ny0) (set! st_by1 st_ny1)
    (st_log)))

;; 图层开关（第 5 枚按钮 / 键位 i）：轨道环的显隐必须重算一帧场景才能擦干净
(define st_toggle
  (lambda ()
    (set! st_on (if (== st_on 1) 0 1))
    (set! scene_dirty 1)
    (set! ui_dirty 1)
    (Orb.set_time sim_d sim_s)
    (st_log)))

;; 诊断行（e2e 断言：星下点/高度/速度/屏幕坐标/TLE 历元与年龄）
(define st_log
  (lambda ()
    (Orb.set_time sim_d sim_s)      ; st_begin 的 ring 采样会把 orb_* 停在第 180 个采样点上
    (display "[earthport] station on=") (display st_on)
    (display " lat=") (display (num_fix Orb.orb_latg 2))
    (display " lon=") (display (num_fix Orb.orb_lon 2))
    (display " alt=") (display (num_fix Orb.orb_alt 1))
    (display " v=") (display (num_fix Orb.orb_v 2))
    (display " r=") (display (num_fix (/ Orb.orb_r Orb.RE_EQ) 2))
    (display " sx=") (display (num_fix st_mx 1))
    (display " sy=") (display (num_fix st_my 1))
    (display " vis=") (display st_mv)
    (display " tle=") (display (Orb.tle_date_str))
    (display " age=") (display (num_fix Orb.orb_age 2))
    (newline)))

;; ---- 视点（相机）操作 ----------------------------------------------------

;; 滚转/经度环绕到 [0,360)，纬度不环绕（直接截到 ±90°）
(define view_norm
  (lambda ()
    (while (>= view_lon 360.0) (set! view_lon (- view_lon 360.0)))
    (while (< view_lon 0.0) (set! view_lon (+ view_lon 360.0)))
    (while (>= view_roll 360.0) (set! view_roll (- view_roll 360.0)))
    (while (< view_roll 0.0) (set! view_roll (+ view_roll 360.0)))
    (view_update)))

;; 视点参数变了：重算三角函数、贴图列基准、屏幕→局部系的旋转系数，再重投太阳
(define view_update
  (lambda ()
    (define a 0.0)
    (set! a (* view_lat (/ PI 180.0)))
    (set! vlat_s (Math.sin a))
    (set! vlat_c (Math.cos a))
    (set! a (* view_roll (/ PI 180.0)))
    (set! RC (* RINV (Math.cos a)))
    (set! RS (* RINV (Math.sin a)))
    (set! RSn (- 0.0 RS))
    (set! lonbase (+ view_lon 60.0))
    (sun_local)
    (aurora_update)))

;; 平移视点（度）：dlat 向北为正，dlon 向东为正。经度方向按 1/cos(φ0) 补偿
;; （拖动时屏幕位移与经度的关系就是 cos(φ0) 倍；高纬下限 0.25 防“打滑”）。
(define cam_nudge
  (lambda (dlat dlon)
    (define cf 0.0)
    (set! view_lat (+ view_lat dlat))
    (if (> view_lat 90.0) (set! view_lat 90.0) #f)
    (if (< view_lat (- 0.0 90.0)) (set! view_lat (- 0.0 90.0)) #f)
    (set! cf (Math.cos (* view_lat (/ PI 180.0))))
    (if (< cf 0.25) (set! cf 0.25) #f)
    (set! view_lon (+ view_lon (/ dlon cf)))
    (view_norm)
    (set! rnd_mode 0)               ; 正在跑的旧视角整质量重绘作废（随后有粗预览盖满）
    (set! cam_dirty 1)
    (set! cam_settle CAM_SETTLE)))

;; 手指拖动 (dx,dy) 像素 → 视点移动（Google Earth 手感：手指往哪拖，地面就往哪走）
;; 拖右 ⇒ 视点往西（经度减），拖下 ⇒ 视点往北（纬度增）。
(define cam_move
  (lambda (dx dy)
    (cam_nudge (* dy cam_k) (* (- 0.0 dx) cam_k))))

;; 滚转到指定角度（度，任意实数）—— 拨轮手势与键盘/按钮都走这里
(define cam_roll_set
  (lambda (r)
    (set! view_roll r)
    (view_norm)
    (set! rnd_mode 0)
    (set! cam_dirty 1)
    (set! cam_settle CAM_SETTLE)))

;; 滚转一档（度）
(define cam_roll
  (lambda (d)
    (cam_roll_set (+ view_roll d))))

;; 复位：回到 0°N 120°E、正北朝上（与启动时的视角一致）
(define cam_reset
  (lambda ()
    (set! view_lat 0.0)
    (set! view_lon 120.0)
    (set! view_roll 0.0)
    (view_norm)
    (set! rnd_mode 0)
    (set! cam_dirty 1)
    (set! cam_settle 1)             ; 下一拍立刻开始整质量重绘
    (display "[earthport] view reset") (newline)))

;; 视角诊断行（e2e / 与 Python 参考渲染器对比用；每次整质量重绘开始时打一行）。
;; bj= 是左上角时钟用的**同一份**北京日期/时刻（同一函数拼出来的字符串），
;; 供 e2e 断言"北京时间 = UTC+8 且格式为 YYYY-MM-DD / hh:mm:ss"。
(define view_log
  (lambda (tag)
    (bj_sync)
    (display "[earthport] ") (display tag)
    (display " lat=") (display (num_fix view_lat 2))
    (display " lon=") (display (num_fix view_lon 2))
    (display " roll=") (display (num_fix view_roll 1))
    (display " sun=(")
    (display (Math.to_fixed SE 4)) (display ",")
    (display (Math.to_fixed SN 4)) (display ",")
    (display (Math.to_fixed SU 4)) (display ")")
    (display " bj=") (display (bj_date_str)) (display "T") (display (bj_time_str))
    (newline)))

;; ---- 场景绘制 ---------------------------------------------------------------
;;
;; 逐像素着色内核。st = 采样步长：1 = 整质量（每拍 STRIP 行、分条推进），
;; st > 1 = 粗预览（每个 st×st 像素块只算一次，fill_rect 一次填满）。
;; 拖动时走粗预览（像素量 1/st²，整帧 ≈33ms），松手后由 cam_settle 倒计时
;; 回到整质量分条重绘。所有着色计算都在**视点局部系**里做：屏幕切平面分量
;; (dxf,dyf)/R 先按滚转角旋到 (xe,xn)，再用 xU = w 补齐球面法向。

(define rnd_start
  (lambda ()
    (define jump 0)
    (sun_ephem)
    (view_update)
    ;; 时刻大跳（时间机器的 ←/→/↑/↓、跨日）时，屏上的旧环已经不在新路径附近：
    ;; 先用**旧缓存**把它擦掉，否则它会在球外一直留到收尾（甚至变成残影）。
    ;; 正常 30 s 周期只漂 0.2 px，收尾时按新路径擦一遍（±1px 余量）就够。
    (if (not (== st_dd sim_d)) (set! jump 1) #f)
    (if (> (Math.abs (- sim_s st_ds)) 240.0) (set! jump 1) #f)
    (if (== jump 1)
        {
          (st_erase_ring)
          (st_erase_marker)
        }
        #f)
    (st_begin)                      ; 空间站图层缓存（与当前视点/时刻一致）
    (set! scen_sod sim_s)
    (set! scene_dirty 0)
    (set! rnd_y ROW_Y0)
    (set! rnd_mode 1)
    (set! st_tk 0)
    (view_log "scene start")
    (st_log)))                      ; 诊断行：每帧场景开始时打一行（e2e 用）

(define rnd_draw
  (lambda (y0 y1 st)
    (define y y0)
    (define yend y1)
    (define hf (* 0.5 (- st 1.0)))      ; 块中心相对块左上角的偏移
    (define bh st)                      ; 本行像素块的高度（末端块会被 yend 截短）
    (define xa 0.0)
    (define xb 0.0)
    (define x 0.0)
    (define dyf 0.0)
    (define dy2 0.0)
    (define rowc 0.0)
    (define rown 0.0)
    (define dxf 0.0)
    (define q 0.0)
    (define xe 0.0)
    (define xn 0.0)
    (define w 0.0)
    (define sny 0.0)
    (define rowoff 0.0)
    (define dlon 0.0)
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
    (define rr 0.0)
    (define gg 0.0)
    (define bb 0.0)
    (define au 0.0)
    (define at 0.0)
    (define av 0.0)
    (define aw 0.0)
    (define bu 0.0)
    (define g1 0.0)
    (define rd 0.0)
    (define bl 0.0)
    (define dk 0.0)
    (define md 0.0)
    (define dn 0.0)
    (define nb 0.0)
    (define aa 0.0)
    (define ab2 0.0)
    (define us 0.0)
    (define us3 0.0)
    (define go 0)
    (define hxy 0.0)
    (define av1 0.0)
    (define av2 0.0)
    (define av3 0.0)
    (define at3 0.0)
    (define g2 0.0)
    (define g3 0.0)
    (define wv 0.0)
    (while (< y yend) {
      (set! dyf (- (+ y hf) CY))
      (set! dy2 (* dyf dyf))
      ;; 粗预览的像素块不越出渲染窗（否则会在窗口下沿留一条被压暗的缝）
      (set! bh st)
      (if (> (+ y st) yend) (set! bh (- yend y)) #f)
      (if (< dy2 XMAX2)
          {
            ;; 屏幕切平面 → 视点局部系（东,北）：按滚转角旋转。dyf 对一行是常量，
            ;; 所以 rowc/rown 提到行外算，逐像素只剩两次乘法。
            (set! rowc (* (- 0.0 dyf) RS))
            (set! rown (* (- 0.0 dyf) RC))
            (set! xext (Math.sqrt (- XMAX2 dy2)))
            (set! xa (- CX xext))
            (set! xb (+ CX xext))
            (if (< xa 0.0) (set! xa 0.0) #f)
            (if (> xb (- W 1)) (set! xb (- W 1)) #f)
            (set! x xa)
            (while (< x xb) {
              (set! dxf (- (+ x hf) CX))
              (set! q (* (+ (* dxf dxf) dy2) R2I))
              (if (< q 1.0)
                  { ; ==================== 球面 ====================
                    (set! w (Math.sqrt (- 1.0 q)))
                    (set! xe (+ (* dxf RC) rowc))
                    (set! xn (+ (* dxf RSn) rown))
                    ;; 球面点的纬度：sinLat = sinφ0·xU + cosφ0·xN
                    (set! sny (+ (* vlat_s w) (* vlat_c xn)))
                    (set! dot (+ (* xe SE) (+ (* xn SN) (* w SU))))
                    ;; 经度偏移：Δλ = atan2(xE, cosφ0·xU - sinφ0·xN)
                    (set! dlon (Math.atan2 xe (- (* vlat_c w) (* vlat_s xn))))
                    (set! col (+ (* dlon DEG) lonbase))
                    ;; 列环绕（lonbase ∈ [60,420)、Δλ ∈ (-180,180] ⇒ 至多溢出一次）
                    (if (< col 0.0) (set! col (+ col 360.0)) #f)
                    (if (>= col 360.0) (set! col (- col 360.0)) #f)
                    ;; 行号必须先取整再乘列数：idx = rowoff + col 、charCodeAt 向下截断，
                    ;; 行号的小数部分会被当成列偏移（旧版 lat=0 时 (1-sinLat)*90 恒为整数，
                    ;; 所以这个隷没暴露；一有纬度/滚转就是满屏环形噪点）。
                    ;; 四舍五入（不是向下取整）：贴图第 r 行的中心在行号 r+0.5 处，
                    ;; 而默认视角下每行恰好落在 .0 —— 向下取整会让 1e-14 的浮点噪声
                    ;; 把整行错到上一行（与旧版逐位不一致），四舍五入稳定。
                    (set! sny (* (- 1.0 sny) ROWH))
                    (if (< sny 0.0) (set! sny 0.0) #f)
                    (if (> sny ROWLAST) (set! sny ROWLAST) #f)
                    (set! rowoff (* (Math.floor (+ sny 0.5)) TCOLS))
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
                          (set! nh (+ (* xe HX) (+ (* xn HY) (* w HZ))))
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
                    ;; 大气照明用**切点法向的太阳投影** dl = xE·sEast + xN·sNorth
                    ;;（**不含 w 项**）。理由：决定这段大气亮不亮的是"视线掠射路径上
                    ;;  最近点的法向"，而球面在天顶方向每像素只转 0.6°、在视圆盘边缘却
                    ;;  一转就是 6°（球面与视线相切）——用含 w 的地表法向做照明，亮环内侧
                    ;;  必然出现 1px 的亮度断层；dl 跨边缘连续且平滑。
                    (set! dl (+ (* xe SE) (* xn SN)))
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
                    (set! rr (+ r (/ (* 255.0 ar) (+ ar 1.0))))
                    (set! gg (+ g (/ (* 255.0 ag) (+ ag 1.0))))
                    (set! bb (+ b (/ (* 255.0 ab) (+ ab 1.0))))}
                  { ; ==================== 球外：只有大气（柔和边缘） ====================
                    ;; 球外只有大气：柱密度从 b=1 处的值起，按三次缓出平滑归零
                    ;;   用 exp(-u/H)³ 曾让辉光每 1px 掉一半（1/e 尺度只剩 3px），
                    ;;   肉眼就是"一圈硬边的亮环"；三次窗在整个 HALO_U（≈12px）上
                    ;;   连续衰减且末端斜率为 0 ⇒ 边缘化开、与球内在 b=1 严格连续。
                    (set! bo (- (Math.sqrt q) 1.0))
                    (set! xo (- 1.0 (/ bo HALO_U)))
                    (if (< xo 0.0) (set! xo 0.0) #f)
                    (set! s3 (* S3L (* xo (* xo xo))))
                    (set! xe (+ (* dxf RC) rowc))
                    (set! xn (+ (* dxf RSn) rown))
                    (set! dl (+ (* xe SE) (* xn SN)))   ; 缘上法向的太阳投影（可为负）
                    (set! lit (* (+ dl ATM_LIFT) ATM_LIFT_K))
                    (if (< lit 0.0) (set! lit 0.0) #f)
                    (if (> lit 1.0) (set! lit 1.0) #f)
                    ;; 球外没有地表法向，入射气柱就看切点方向（w = 0）：sunv = max(0, dl)
                    (set! sunv (if (< dl 0.0) 0.0 dl))
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
                    (set! rr (+ BG_R (/ (* 255.0 ar) (+ ar 1.0))))
                    (set! gg (+ BG_G (/ (* 255.0 ag) (+ ag 1.0))))
                    (set! bb (+ BG_B (/ (* 255.0 ab) (+ ab 1.0))))})
              ;; ==================== 极光（体发射；三层球壳 = 垂直结构）====================
              ;; 视线（正交投影 ⇒ 方向恒为视轴）与三层发射壳求交：命中点地磁纬决定是否
              ;; 落在椭圆上。外层（红）最宽，先算它做窗口预筛 —— 绝大多数像素在这里就退出。
              (set! go 0)
              (if (< q AUR_R3Q)
                  {
                    (set! hxy (+ (* xe ME) (* xn MN)))       ; 三层共用的切平面投影
                    (set! us3 (Math.sqrt (- AUR_R3Q q)))
                    (set! at3 (* (+ hxy (* us3 MU)) AUR_I3))
                    ;; 预筛用 |cos|²（南北共轭对称 ⇒ 一个窗口覆盖两极），省 abs 与除法
                    (set! av3 (* at3 at3))
                    (if (and (> av3 AUR_PL) (< av3 AUR_PH)) (set! go 1) #f)
                  }
                  #f)
              (if (== go 1)
                  {
                    ;; ---- 沿椭圆的折叠与亮暗结（磁经度前两阶谐波；纯乘加）----
                    (set! aa (* (+ (* xe PE) (+ (* xn PN) (* us3 PU))) AUR_I3))
                    (set! ab2 (* (+ (* xe QE) (+ (* xn QN) (* us3 QU))) AUR_I3))
                    (set! wv (* AUR_WAVS (+ (* 0.55 aa) (+ (* 0.35 ab2)
                                            (* 0.40 (- (* aa aa) (* ab2 ab2)))))))
                    (set! md (+ 1.0 (+ (* 0.45 ab2) (+ (* 0.25 aa)
                                          (* 0.35 (- (* ab2 ab2) (* aa aa)))))))
                    (if (< md 0.35) (set! md 0.35) #f)
                    (if (> md 1.7) (set! md 1.7) #f)
                    ;; ---- 夜侧门控 + 磁午夜增亮（用外层壳算**太阳**天顶角；三层差别 <2px）----
                    (set! dn (* (+ (* xe SE) (+ (* xn SN) (* us3 SU))) AUR_I3))
                    (set! dk (* (+ (- 0.0 dn) AUR_D0) AUR_DK))
                    (if (< dk 0.0) (set! dk 0.0) #f)
                    (if (> dk 1.0) (set! dk 1.0) #f)
                    (set! nb (- 0.0 dn))
                    (if (< nb 0.0) (set! nb 0.0) #f)
                    (if (> nb 1.0) (set! nb 1.0) #f)
                    (set! dk (* dk (* md (+ AUR_NB (* (- 1.0 AUR_NB) nb)))))
                    ;; ---- 层 1（86km，绿，下缘锐）----
                    ;; 注意：南北两个椭圆关于地磁轴共轭 ⇒ 每层都要先取 |cos| 再算带宽坐标
                    ;; （只对北侧取绝对值的话，南半球的极光会整块消失）。
                    (set! g1 0.0)
                    (if (< q AUR_R1Q)
                        {
                          (set! us (Math.sqrt (- AUR_R1Q q)))
                          (set! av1 (* (+ hxy (* us MU)) AUR_I1))
                          (if (< av1 0.0) (set! av1 (- 0.0 av1)) #f)
                          (set! av1 (* (- (+ av1 wv) AUR_C1) AUR_V1))
                          (if (< av1 0.0) (set! av1 (* av1 AUR_ASY)) #f)
                          (set! g1 (- 1.0 (* av1 av1)))
                          (if (< g1 0.0) (set! g1 0.0) #f)
                          (set! g1 (* g1 g1))
                        }
                        #f)
                    ;; ---- 层 2 / 层 3 只在整质量时算 ----
                    ;;   粗预览的像素块是 4×4，而三层的屏幕间距只有 ~1px：预览里根本分不出来，
                    ;;   却要多付两次 sqrt + 二十来条指令（拖动时的预览是时间敏感路径）。
                    (set! g2 0.0)
                    (set! g3 0.0)
                    (if (== st 1.0)
                        {
                          ;; 层 2（201km，绿青，过渡）
                          (if (< q AUR_R2Q)
                              {
                                (set! us (Math.sqrt (- AUR_R2Q q)))
                                (set! av2 (* (+ hxy (* us MU)) AUR_I2))
                                (if (< av2 0.0) (set! av2 (- 0.0 av2)) #f)
                                (set! av2 (* (- (+ av2 wv) AUR_C2) AUR_V2))
                                (set! g2 (- 1.0 (* av2 av2)))
                                (if (< g2 0.0) (set! g2 0.0) #f)
                                (set! g2 (* g2 g2))
                              }
                              #f)
                          (if (< at3 0.0) (set! at3 (- 0.0 at3)) #f)   ; 南北共轭
                          ;; 层 3（277km，红，弥散上缘）
                          (set! av3 (* (- (+ at3 wv) AUR_C3) AUR_V3))
                          (set! g3 (- 1.0 (* av3 av3)))
                          (if (< g3 0.0) (set! g3 0.0) #f)
                          (set! g3 (* g3 g3))
                        }
                        #f)
                    ;; ---- 加色合成：下绿（G 主导）→ 上红（R 主导）----
                    (set! g1 (* g1 dk))
                    (set! g2 (* g2 dk))
                    (set! g3 (* g3 dk))
                    (set! rr (+ rr (* g1 (* AUR_G1 0.34))))
                    (set! rr (+ rr (* g2 (* AUR_G2 0.32))))
                    (set! rr (+ rr (* g3 AUR_G3)))
                    (set! gg (+ gg (* g1 AUR_G1)))
                    (set! gg (+ gg (* g2 (* AUR_G2 0.92))))
                    (set! gg (+ gg (* g3 (* AUR_G3 0.28))))
                    (set! bb (+ bb (* g1 (* AUR_G1 0.50))))
                    (set! bb (+ bb (* g2 (* AUR_G2 0.66))))
                    (set! bb (+ bb (* g3 (* AUR_G3 0.34))))
                  }
                  #f)
              (if (== st 1.0)
                  (Gfx.set_pixel x y rr gg bb)
                  (Gfx.fill_rect x y st bh rr gg bb))
              (set! x (+ x st))
            })
          }
          #f)
      (set! y (+ y st))
    })))

;; 整质量：每拍画 STRIP 行
(define rnd_step
  (lambda ()
    (define y0 rnd_y)
    (define yend (+ y0 STRIP))
    (if (> yend ROW_Y1) (set! yend ROW_Y1) #f)
    (rnd_draw y0 yend 1.0)
    (set! rnd_y yend)
    (if (>= rnd_y ROW_Y1) (set! rnd_mode 0) #f)
    ;; 分条重绘期间**不画**图层：抗锯齿线不是幂等的（重画会更实），所以环与标记都
    ;; 留到 rnd_finish 一次画完（此时每一行都已经落在刚算出的背景/地表上）。
    (Gfx.refresh)))

;; 粗预览：本拍把整帧（粗采样）画完 —— 拖动时每拍一张，够跟手。
;; 视点一变，旧图层就"错位"了：球内那部分由粗预览重画盖掉，球外那部分必须自己擦
;; （环 → 背景色折线，标记 → 包围盒填背景），否则拖一会儿就会留下一堆残环。
(define rnd_preview
  (lambda ()
    (st_erase_ring)
    (st_erase_marker)
    (st_begin)
    (rnd_draw ROW_Y0 ROW_Y1 ST_PREV)
    (st_ring 0 H)
    (st_marker)
    (st_mark_b)                     ; 下一帧要按**这一帧**的标记位置去擦（预览帧之间也在动）
    (Gfx.refresh)))

;; 一帧整质量场景画完后的收尾（诊断行；渲染窗只到第 198 行，HUD 不会被它擦到）
(define rnd_finish
  (lambda ()
    ;; 球内那半已被分条重绘覆盖；球外的旧环/旧标记在这里擦掉，然后把环与标记
    ;; **各画一次**（抗锯齿线不能重复叠加）
    (st_erase_ring)
    (st_erase_marker)
    (st_ring 0 H)
    (st_marker)
    (st_mark_b)
    (set! st_dd sim_d)
    (set! st_ds sim_s)
    (st_card)
    (display "[earthport] scene done heap=")
    (display (get_item (System.memstat) 3)) (newline)))

;; ---- HUD -------------------------------------------------------------------

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

;; 北京时间 = UTC + 8h：先按小时偏置拆出"北京日/当日秒"，民用历仍走共享星历
;; 的 (Ephem.civil)（跨日、跨月、跨年都由它处理）
(define bj_sync
  (lambda ()
    (set! bj_d sim_d)
    (set! bj_s (+ sim_s 28800.0))
    (while (>= bj_s 86400.0) {
      (set! bj_s (- bj_s 86400.0))
      (set! bj_d (+ bj_d 1))
    })
    (Ephem.civil bj_d)
    (set! bj_y (Ephem.get_civ_y))
    (set! bj_m (Ephem.get_civ_m))
    (set! bj_dd (Ephem.get_civ_d))))

;; "YYYY-MM-DD"
(define bj_date_str
  (lambda ()
    (String.concat (String.atom_to_string bj_y)
      (String.concat "-" (String.concat (pad2 bj_m)
        (String.concat "-" (pad2 bj_dd)))))))

;; "hh:mm:ss"
(define bj_time_str
  (lambda ()
    (define sec0 0.0)
    (define min0 0.0)
    (define hr0 0.0)
    (set! sec0 (mod bj_s 60.0))
    (set! min0 (mod (/ (- bj_s sec0) 60.0) 60.0))
    (set! hr0 (/ (- (- bj_s sec0) (* min0 60.0)) 3600.0))
    (String.concat (pad2 hr0) (String.concat ":" (String.concat (pad2 min0)
      (String.concat ":" (pad2 sec0)))))))

;; 左上角时钟：两行左对齐（日期 / 时刻），每秒重画一次
(define draw_clock
  (lambda ()
    (if (== hud_dim 1)
        #f
        {
          (bj_sync)
          (hud_text (bj_date_str) 6 3 10)      ; YYYY-MM-DD（10 字符）
          (hud_text (bj_time_str) 6 18 8)      ; hh:mm:ss（8 字符）
        })))

;; 时间状态徽标（底部信息行右端，琥珀色）：暂停 → "暂停"；有偏置 → ±h:mm；实时 → 空
(define badge_str
  (lambda ()
    (define s "")
    (define mins 0.0)
    (define oh 0.0)
    (define om 0.0)
    (define av 0.0)
    (set! av (Math.abs off_s))
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
    s))

;; 底部信息行（控制条正上方）：左侧视角读数（纬度/经度/滚转角，左对齐），
;; 右端时间徽标。整行先擦后画，两者互不遮挡。
(define draw_info
  (lambda ()
    (define ala 0.0)
    (define alo 0.0)
    (define rv 0.0)
    (define s "")
    (define b "")
    (if (== hud_dim 1)
        #f
        {
          (Gfx.fill_rect 4 INFO_Y (- W 8) 14 BG_R BG_G BG_B)
          (set! ala (Math.abs view_lat))
          (set! alo view_lon)
          (if (> alo 180.0) (set! alo (- 360.0 alo)) #f)   ; 240°E → 120°W
          (set! rv view_roll)                              ; 0..360 → -180..180
          (if (> rv 180.0) (set! rv (- rv 360.0)) #f)
          (set! s (num_fix ala 1))
          (set! s (String.concat s (if (>= view_lat 0.0) "N " "S ")))
          (set! s (String.concat s (num_fix alo 1)))
          (set! s (String.concat s (if (> view_lon 180.0) "W " "E ")))
          (set! s (String.concat s "R"))
          (set! s (String.concat s (if (>= rv 0.0) "+" "-")))
          (set! s (String.concat s (String.atom_to_string (Math.floor (Math.abs rv)))))
          (set! s (String.concat s "°"))
          (Gfx.draw_text s 6 (+ INFO_Y 1) HUD_R HUD_G HUD_B)
          (set! b (badge_str))
          (if (> (String.length b) 0)
              (Gfx.draw_text b (- (- W 9) (text_w b)) (+ INFO_Y 1) 226 186 110)
              #f)
        })))

;; 只擦 HUD 文字所在的小矩形（不碰球/辉光/控制条/退出按钮/空间站图层）
(define clear_hud
  (lambda ()
    (Gfx.fill_rect 4 1 112 16 BG_R BG_G BG_B)
    (Gfx.fill_rect 4 16 96 16 BG_R BG_G BG_B)
    (Gfx.fill_rect 4 INFO_Y (- W 8) 14 BG_R BG_G BG_B)
    (st_card_erase)))

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
          (draw_info)
        })
    (st_card)
    (Gfx.refresh)))

;; 触屏：
;;   · 按下落在**普通按钮**上 → 按下沿就执行（[复位] / [有云/无云]）
;;   · 按下落在**滚转键**上 → 不动：松手算点按（±15°），按住 400ms 后每 120ms 连发一档；
;;     **横向滑动**（>3px）= 拨轮：连续滚转（一像素 DIAL_K 度）⇒ 任意旋转角
;;   · 按下落在空白处 → 位移超 3px 就当拖动（每拍取增量、累计后一次性应用），
;;     松手时：整段都没拖动 ⇒ 算点按（切换 HUD），拖过 ⇒ 立刻重画整质量
;;
;;   in_btn：0 = 空白处（拖动改视点），1 = 普通按钮，2 = 滚转键（拨轮）
(define read_touch
  (lambda ()
    (define tx 0.0)
    (define ty 0.0)
    (if (Touch.is_available)
        {
          (if (Touch.is_pressed)
              {
                (set! tx (Touch.x))
                (set! ty (Touch.y))
                (if (== tprev 0)
                    { ; ---- 按下沿 ----
                      (Touch.take_press)
                      (set! last_x tx)
                      (set! last_y ty)
                      (set! acc_x 0.0)
                      (set! acc_y 0.0)
                      (set! drag_moved 0)
                      (set! rep_done 0)
                      (set! press_ms (System.timestamp))
                      (set! bi (btn_hit tx ty))
                      (set! in_btn 0)
                      (if (>= bi 0)
                          (if (and (>= bi 0) (< bi 2))
                              { ; 滚转键：先不动作 —— 松手才算点按，滑动则转成拨轮
                                (set! in_btn 2)
                                (set! dial0 view_roll)
                              }
                              { (set! in_btn 1)
                                (btn_do bi) })
                          #f)
                    }
                    { ; ---- 按住 ----
                      (set! acc_x (+ acc_x (- tx last_x)))
                      (set! acc_y (+ acc_y (- ty last_y)))
                      (set! last_x tx)
                      (set! last_y ty)
                      (if (== in_btn 2)
                          { ; 滚转键：横向滑动 = 拨轮（连续滚转）；不动则长按连发 ±15°
                            (if (== drag_moved 0)
                                (if (> (Math.abs acc_x) 3.0)
                                    (set! drag_moved 1)
                                    #f)
                                #f)
                            (if (== drag_moved 1)
                                { ; 拨轮：右滑 = 画面顺时针转（滚转角减小）
                                  (set! dragging 1)
                                  (cam_roll_set (- dial0 (* DIAL_K acc_x)))
                                  (set! dial0 view_roll)
                                  (set! acc_x 0.0)
                                  (set! acc_y 0.0)
                                }
                                { ; 长按连发：按时钟算而不按拍数（wasm 上一拍 30~50ms）
                                  (set! hold_ms (- (System.timestamp) press_ms))
                                  (if (and (>= hold_ms REPEAT_DELAY)
                                           (>= hold_ms (+ REPEAT_DELAY (* rep_done REPEAT_GAP))))
                                      {
                                        (set! rep_done (+ rep_done 1))
                                        (btn_do bi)
                                      }
                                      #f)
                                })
                          }
                          (if (== in_btn 0)
                              { ; 空白处：超过阈值就成拖动（把已累计的位移一次性补上，不丢头）
                                (if (== drag_moved 0)
                                    (if (> (+ (Math.abs acc_x) (Math.abs acc_y)) 3.0)
                                        (set! drag_moved 1)
                                        #f)
                                    #f)
                                (if (== drag_moved 1)
                                    {
                                      (set! dragging 1)
                                      (cam_move acc_x acc_y)
                                      (set! acc_x 0.0)
                                      (set! acc_y 0.0)
                                    }
                                    #f)
                              }
                              #f))
                    })
                (set! tprev 1)
              }
              { ; ---- 未按下 ----
                (if (== tprev 1)
                    { ; 松开沿
                      (Touch.take_release)
                      (set! dragging 0)
                      ;; 松手时把还没应用的那点位移补上（一拍的亚像素残量不丢）
                      (if (== in_btn 2)
                          { ; 滚转键：没滑动也没连发过 ⇒ 算一次点按（±15°）
                            (if (and (== drag_moved 1) (not (== acc_x 0.0)))
                                (cam_roll_set (- dial0 (* DIAL_K acc_x)))
                                #f)
                            (if (and (== drag_moved 0) (== rep_done 0))
                                (btn_do bi)
                                #f)
                            (set! cam_settle 1)
                          }
                          (if (== drag_moved 1)
                              { (if (or (not (== acc_x 0.0)) (not (== acc_y 0.0)))
                                    (cam_move acc_x acc_y)
                                    #f)
                                (set! acc_x 0.0)
                                (set! acc_y 0.0)
                                (set! cam_settle 1) }   ; 拖完立刻回到整质量重绘
                              (if (== in_btn 0) (toggle_hud) #f)))
                    }
                    #f)
                (set! tprev 0)
              })
        }
        #f)))

;; 键盘：←/→ 步进 2 分钟，↑ 实时，↓ 暂停/运行，空格/回车 HUD 开关，c 云层；
;;   a/d 经度、w/s 纬度、`,`/`.` 滚转、r 复位视点（按钮没有的项都在这里）
(define read_key
  (lambda ()
    (if (Key.is_available)
        {
          (set! keyv (Key.take_press))
          (while (> keyv 0)
            {
              ;; 时间机器（按钮已去掉，只剩键位）
              (if (== keyv 3) (time_step -120) #f)
              (if (== keyv 4) (time_step 120) #f)
              (if (== keyv 1) (time_live) #f)
              (if (== keyv 2) (time_toggle_pause) #f)
              ;; 云层 / 视角
              (if (or (== keyv 99) (== keyv 67)) (clouds_toggle) #f)
              (if (or (== keyv 105) (== keyv 73)) (st_toggle) #f)   ; i = 空间站图层
              (if (or (== keyv 97) (== keyv 65)) (cam_nudge 0.0 (- 0.0 CAM_KEY)) #f)
              (if (or (== keyv 100) (== keyv 68)) (cam_nudge 0.0 CAM_KEY) #f)
              (if (or (== keyv 119) (== keyv 87)) (cam_nudge CAM_KEY 0.0) #f)
              (if (or (== keyv 115) (== keyv 83)) (cam_nudge (- 0.0 CAM_KEY) 0.0) #f)
              (if (or (== keyv 44) (== keyv 60)) (cam_roll (- 0.0 CAM_ROLL)) #f)
              (if (or (== keyv 46) (== keyv 62)) (cam_roll CAM_ROLL) #f)
              (if (or (== keyv 114) (== keyv 82)) (cam_reset) #f)
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

;; 同一套渐变重填一小块（供“只擦一小片”的场景更新用：空间站图层的行列重绘、
;; 左下角信息卡的擦除）——颜色公式与 draw_bg 逐字一致，否则会留下色差方块。
(define bg_fill
  (lambda (x y w h)
    (define yy y)
    (define ye (+ y h))
    (define i 0)
    (define y2 0.0)
    (while (< yy ye) {
      (set! i (Math.floor (/ yy 6.0)))
      (set! y2 (* (+ i 1) 6.0))
      (if (> y2 ye) (set! y2 ye) #f)
      (Gfx.fill_rect x yy w (- y2 yy)
                     (+ BG_R (* i 0.02)) (+ BG_G (* i 0.035)) (+ BG_B (* i 0.09)))
      (set! yy y2)
    })))

;; ---- 每拍 ------------------------------------------------------------------

(define redraw_all
  (lambda ()
    (draw_bg)
    (draw_stars)
    (EX.draw_button_at 270 3)
    (draw_btns)
    ;; 先补 HUD（球体由随后的分条绘制补），退出确认框关掉后立刻恢复原样
    (draw_clock)
    (draw_info)
    (st_card)
    (set! clk_sec -1)
    (set! rnd_y ROW_Y0)
    (set! rnd_mode 1)
    (st_begin)
    (st_mark_b)
    (Gfx.refresh)))

(define on_tick
  (lambda ()
    (EX.check)                          ; 硬 A/ESC / 右上角「退出」
    (read_touch)
    (read_key)
    (if (== paused 0) (time_sync) #f)    ; 运行态：模拟时刻 = 实时 + 偏置
    ;; 时钟：每秒一次（暂停时 sim_s 不动，自然不重画）
    (if (== sim_s clk_sec)
        #f
        {
          (set! clk_sec sim_s)
          (draw_clock)
          (st_card)                     ; 空间站推算值跟着每秒刷新
        })
    ;; ---- 时间机器/云层状态变了（键盘触发）：补画控制条高亮与信息行 ----
    (if (== ui_dirty 1)
        {
          (set! ui_dirty 0)
          (set! clk_sec -1)
          (draw_btns)
          (draw_info)
          (time_log)
        }
        #f)
    ;; ---- 视角变了：先出粗预览（拖动时每拍一张），跟手不卡 ----
    (if (== cam_dirty 1)
        {
          (set! cam_dirty 0)
          (rnd_preview)
          (draw_info)
        }
        #f)
    ;; ---- 相机稳定倒计时：拖动/按键停手 CAM_SETTLE 拍后回到整质量分条重绘 ----
    (if (and (== dragging 0) (== rnd_mode 0) (> cam_settle 0))
        {
          (set! cam_settle (- cam_settle 1))
          (if (== cam_settle 0) (rnd_start) #f)
        }
        #f)
    ;; ---- 整质量分条推进 ----
    (if (== rnd_mode 1)
        {
          (rnd_step)
          (if (== rnd_mode 0)
              (rnd_finish)
              #f)
        }
        #f)
    ;; ---- 空间站图层：每 ST_UPD_TK 拍把标记推到新位置（0.09 px/s，不必每拍）----
    ;;   只重画标记占用的一小片行；正在分条重绘/拖动时跳过（那时整帧在重画）。
    (if (and (and (== st_on 1) (== rnd_mode 0)) (and (== dragging 0) (== cam_settle 0)))
        {
          (set! st_tk (+ st_tk 1))
          (if (>= st_tk ST_UPD_TK)
              {
                (set! st_tk 0)
                (st_patch)
              }
              #f)
        }
        (set! st_tk 0))
    ;; ---- 场景调度（时间变化 / 云层开关）：只在空闲时重算 ----
    (if (and (and (== rnd_mode 0) (== dragging 0)) (== cam_settle 0))
        {
          (set! dt_s (- sim_s scen_sod))
          (if (< dt_s 0.0) (set! dt_s (+ dt_s 86400.0)) #f)
          (if (or (>= dt_s SCENE_S) (== scene_dirty 1))
              (rnd_start)
              #f)
        }
        #f)
    (Gfx.refresh)))

;; ---- 启动 ------------------------------------------------------------------

(Gfx.set_font 1)            ; 12px 抗锯齿字模（HUD）
(mkc_stars)
(mkc_orb)                   ; 空间站图层缓存槽（一次性 push）
(EX.install_with redraw_all)
(view_norm)                 ; 初始视角：0°N 120°E、正北朝上（与定点俯视版逐位一致）
(time_live)                 ; 初始：显示实时
(draw_bg)
(draw_stars)
(EX.draw_button_at 270 3)   ; 右上角「退出」
(draw_btns)
(rnd_start)                 ; 第一帧场景（分条渐进，由定时器逐拍推进）
(draw_clock)
(draw_info)
(st_card)                   ; 左下角空间站推算值
(st_log)                    ; 诊断行：星下点/高度/速度/屏幕坐标/TLE 年龄
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
