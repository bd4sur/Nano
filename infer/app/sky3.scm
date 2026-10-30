;; ============================================================================
;; 大气散射天空 III —— Nishita 一阶 + 二阶多次散射（scatter_model_3）—— 灵机应用程序
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;; 参考实现：celestial.c 的 scatter_model_3：球形几何射线-球面求交、逐采样点
;; 地球遮蔽检测、一阶 Nishita 路径积分、二阶多次散射（对上半球 9 个固定方向
;; 再做一次路径积分的数值积分）、臭氧吸收、夜天光、HDR 色调映射。
;; 全部物理常数与积分分辨率均与原实现一致：
;;   Re=6360km  Ra=6420km  Hr=7994m  Hm=1200m
;;   βR={5.8,13.5,33.1}e_6   β_Mie=16e-6   β_Mie消光=1.6e-6   g=0.76   SUN=20
;;   一阶步进 NV=36/NL=1；二阶视线采样 V2=2（B2 优化，原 3）；二阶步进 NV2=2/NL2=2；9 个环境方向
;;   臭氧 {0.020,0.040,0.000}×90；夜天光 {0.05,0.06,0.12}；HDR ε=0.2
;;
;; 视角与太阳位置：
;;   - 视线固定朝天顶：天顶居中，等距方位角投影，天空为屏幕内切圆，
;;     圆外恒黑（圆界采样块按像素精确判定）。
;;   - 屏幕方位取「人在地面抬头看天」的视角：**屏幕上方为正北、左侧为正东**
;;     （与俯视地图的左右相反）。对应视线单位向量（x 东、y 北、z 上）：
;;        vx = −sinθ·dx/r，vy = −sinθ·dy/r，vz = cosθ
;;     其中 dx 向右、dy 向下（屏幕坐标），θ = r/R·90°。
;;   - **太阳位置取自共享星历模块**（app/lib/ephem.scm，同一份库也被
;;     ephemeris.scm / nongli.scm 复用）：按当前系统时间与项目默认观测点
;;     南京 119.0°E / 32.0°N / UTC+8 求太阳地平坐标（高度、方位），再换成
;;     三分量单位向量送入散射模型。太阳移动超过 RE_RENDER_DEG 后自动重绘。
;;
;; 一般性说明（重要）：本实现**不利用任何镜像/对称性剪枝**，每个采样点都独立
;; 完成完整的散射计算；也不做任何跨采样点平滑（采样块原样以 SS×SS 像素块绘制）。
;;
;; 性能剖析与优化（2026-10-30，WASM 目标实测）---------------------------------
;;
;; 一、量测口径（infer/tools/e2e 之外的一次性探针，结论已固化在本文件）
;;   · 无头 Chromium（playwright）+ 真 wasm 产物，页面内 rAF 钩子逐帧记录
;;     (performance.now, Gfx.refresh 的 epoch, VM 指令数)，一帧天空 = 48 次采样行推屏；
;;   · 解释器吞吐（wasm）实测 ≈ 18.7 M 条 VM 指令/秒，与指令种类无关（紧循环
;;     微基准与 sky3 同值）；因此**一帧墙钟 ≈ 该帧总指令数 / 18.7M 秒**，
;;     优化只剩两条路：少跑指令，或把每帧的时间片用满。
;;   · 单帧成本归因（宿主插桩 + 删块消融，全部 1810 个圆内采样点）：
;;      散射总计 63.2 M 条指令，其中二阶多次散射 41.7 M（66%）
;;      ——其中 9 方向 × V2 的内层 march 占 30.5 M（48%）；一阶 march 16.8 M（32%）；
;;      另有双线性重建/字幕 ≈ 6 M。Math.exp/sqrt/pow 共 150 万次调用但只对应
;;      2.4% 的指令（C 数学本体不是瓶颈，解释器分发开销才是）。
;;
;; 二、热点根因 #1（最大项）：**帧预算没被用满** ——
;;   旧设计每个 step 只算 5 个采样点（约 13 万指令），而解释器每帧给的时间片是
;;   312500 条指令（MB_TIMESLICE_TICKS）；更糟的是 16 ms 的 interval 与 16.7 ms
;;   的 vsync 存在拍频，实测 **约 57% 的帧一条指令都没跑**（逐帧探针）。
;;   修正：把 SAMPLES_PER_STEP 取到「一步吃满一个时间片」的大小，切片责任交回
;;   解释器时间片（进程在步内被抢占、下一帧继续）。因数值与绘制顺序都不变，
;;   画面逐像素相同。实测整帧 7.18 s → 2.87 s（−60%）。
;;
;; 三、热点根因 #2：解释器每条源级算符约 3 条 VM 指令，于是少写操作就是提速：
;;   · rsi（射线-球面求交）原来每个采样点调 92 次，现全部内联，且复用 march
;;     里已算的 |w|²、省掉 b/2、b²−4c 的 4 次乘——每处 ~50 条指令；
;;   · Math.pow(base,1.5) → base·√base（wasm 里 pow 是 libm 调用）；
;;   · 帧级常量上提：dom_g 与 9 个环境方向的相位项及其 SUNF·β 系数（原每采样点
;;     重算 18 次）→ 每帧算一次（frame_setup）；
;;   · 相位项里的 (1+μ²) 等公共子式只算一次。
;;   合计散射指令 63.2 M → 53.0 M（−16%），**实测整帧与优化前逐像素零差异**
;;   （宿主回归台冻结时钟 A/B：76800 像素全等，最终帧 CRC 相同）。
;;
;; 四、结果（无头 Chromium，同一场景，各两次）
;;   基准（原文件）                    7180 / 7130 ms
;;   仅把帧预算用满                    2870 / 2870 ms（2.5×）
;;   再加上述指令削减                  2330 / 2330 ms（3.1×，画面逐像素相同）
;;
;; 五、还想更快？三个显式画质旋钮（均已实测，按需取舍；默认全部关闭）
;;   V2 2→1              散射 35.7 M 指令，画质差 ≤ 9/255（与既有 B2 同一量级）
;;   nV 36→24（一阶）    散射 47.4 M 指令，画质差 ≤ 33/255（地平线附近最明显）
;;   SS 5→8              采样点 1810→708，散射 21.0 M 指令，画质差（圆内）≤ 49/255
;;   （SS 只改采样密度与双线性重建的格子大小，不动物理；改 SS 需能整除 320/240）
;;
;; 六、解释器侧的下一个杠杆（未在本文件内实施，需改 vendor/Animac）
;;   指令分布：load_slot 33%、mul 13%、push 11%、add 9%、set_slot 8%、store_slot 6%；
;;   一条 `(set! x (* a b))` 要 4 条指令（两次取槽、乘、写槽）。若在已有的
;;   frame-rewrite 后置遍历里再加一个 peephole（[load_slot A][load_slot B][mul]
;;   → 一条 mul_slots(A,B)，PC 前进 3 并保证没有跳转目标落在被跳过的位置上），
;;   可去掉约 30%~40% 的指令，即本文件再快 1.3~1.4 倍、所有灵机 App 普适。
;;   验证路径：vendor/Animac/testall.sh + infer/tools/e2e/host 的呈现序列 A/B。
;;
;; 交互：两侧触屏按钮调时刻（2026-10-30 追加）----------------------------------
;;   · 屏左/右边缘各一个按钮（圆外黑底上，不与天空圆、不与左上 80×60 的触屏宫格冲突）：
;;     左 = 时刻 −10 分钟，右 = +10 分钟；点一下即重算该时刻的天象。
;;   · **一旦点过按钮就不再保持实时刷新**：进入「手动」模式后，太阳移动不再触发重绘，
;;     右下时钟也停在被选中的时刻（顶部左列显示「手动 ±N 分」）。
;;   · 回到实时：**长按任一按钮 0.7 s**（偏移归零、恢复实时跟随）。
;;   · 实现：偏移只作用在「取时」处（现在 now_shifted 统一给 cur_days/cur_sod，
;;     sun_from_ephem 与 now_string 共用），不动物理模型与渲染管线；触屏用 Touch 本地库
;;     的电平+边沿锁存（Touch.is_pressed / take_press / x / y），在每个 step 开头与每个
;;     采样行完成时各采一次（长帧渲染途中点按也能在一行内响应），命中即 advance()：
;;     重取时刻 → 重装帧常量 → 清屏重画按钮 → 从头渲染（渐进行推屏保留）。
;;   · 无触屏宿主（Touch.is_available 为假）不画按钮，行为与以前完全一致。
;;
;; 依赖：Gfx（绘图）、Math、System（定时器）、Time（取时）、Touch（按钮）、共享星历模块 Ephem。

;; ---- 平滑滤波（消除 SS×SS 色块的马赛克感）----------------------------------
;; 采样阶段（shade/scatter_v）只把每个采样点的颜色写进一张 ROWS_S×COLS_S 的网格
;; （三张预建列表，set_item! 覆写，重绘帧无需重建），随后按"采样带"对每个像素做
;; 双线性插值逐像素上屏，圆外像素不画。天空辐照度是平滑的低频场，双线性重建在
;; 视觉上等同于提高采样密度，而模型求值次数（几乎全部成本）不变。
;; 实测：整帧 2.44 s → 2.41 s（插值成本 <2%），5×5 色块消失、圆周边由 5px 阶梯
;; 变为 1px 锯齿。圆界另做 1px 线性抗锯齿：cov=(R²−r²)/(2R)，圆内深处 cov>1 取 1，
;; 只影响最外 1px（实测整帧仅圆周带内 617 像素变化、带外 0），圆周由硬锯齿变为平滑过渡。
;; （该抗锯齿一度被撤下：它在旧核心上会触发"闭包帧分析父链失效"缺陷，表现为圆周
;;   附近一串规则亮斑；上游 0291412 已修复，复现件与本文件两处载体均已验证正常。
;;   详见 infer/tools/e2e/REPORT-interp-slot-store.md。）

;; ---- 业务侧优化记录（B1/B2，2026-10-21）------------------------------------
;; B1 内联小函数：phase_r / phase_m / hdr_map 都是 1~2 行数学，原来每个采样点被调用
;;    约 110 次；旧版解释器每次 Scheme 调用要复制整份环境，调用开销远超数学本体。
;;    现已把这些表达式逐字内联进 scatter_v（数值结果与原实现完全一致，实测整帧
;;    0 像素差异）。函数定义保留作参考实现。
;;    ※ 解释器升级到"帧描述符 + 槽位寻址"（vendor 5c244e9）后，调用本身已变便宜，
;;      实测 B1 的收益从早期报告的 −21% 降到 −3% 左右（早先的探针是把数学也一起去掉，
;;      才显得那么大）；保留它是因为零风险且略快，若要可读性可回退。
;; B2 二阶多次散射视线采样 V2：3 → 2（见常量 V2）。二阶项沿视线是低频量，中点法
;;    2 个采样点的整帧最大通道偏差 9/255、平均 ~8（仅影响偏蓝的通道），宿主实测
;;    600 帧 3.34 s → 2.44 s（−27%，含 B1）。这是画质/性能的显式折衷，把 V2 改回 3
;;    即可恢复原画质（时间回到 ~3.2 s）。
;; 宿主内存池：本程序 + 星历模块的编译期 VM 工作区需要较大池（ui_scheme_app.c
;; 现为 12MB），运行期堆占用仍极小。
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
(native Touch)

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
(define K_DEG (/ 90.0 R_SCR))           ; 每像素天顶角（度）：视线仰角用
(define SS 5)                       ; 采样块边长（像素；5 整除 320/240）
(define HALF_S (/ SS 2.0))          ; 采样块中心到块边界的偏移
(define COLS_S (Math.floor (/ SW SS)))
(define ROWS_S (Math.floor (/ SH SS)))

;; 每步推进的采样点数：取一个大到「一步吃满一个时间片」的值。
;; 历史：本项原为 5（约 24 万指令），配合 System.set_interval 16 调度，每个 rAF 帧
;; 实际只跑 1 步就返回、外加定时器与 vsync 的相位差导致不少帧整帧空转——实测
;; 整帧 7.2 s，其中约 55% 是「帧预算没被用掉」的等待。改为一个大数后，切片的责任
;; 交回解释器时间片（MB_TIMESLICE_TICKS=312500 条指令，见 ui_scheme_app.c）：
;; 进程在步内被抢占、下一帧继续，于是每一帧都把时间片跑满，整帧 2.9 s。
;; 数值与绘制顺序完全不变（仍按采样点逐行推进、每画完一行 Gfx.refresh）。
(define SAMPLES_PER_STEP 4096)      ; 每步推进的采样点数（一步吃满解释器时间片）
(define HOLD_MS 300)                ; 渲染完成后的初始停留（毫秒），之后按 HOLD_CHECK_MS 轮询太阳是否移动

;; ---- 物理常数（celestial.c scatter_model_3）-------------------------------

(define Re 6360000.0)               ; 地球半径 (m)
(define Ra 6420000.0)               ; 大气层外边界半径 (m)
(define Hr 7994.0)                  ; Rayleigh 标高 (m)
(define Hm 1200.0)                  ; Mie 标高 (m)
(define bR0 0.0000058)              ; Rayleigh 散射系数（R）
(define bR1 0.0000135)              ; Rayleigh 散射系数（G）
(define bR2 0.0000331)              ; Rayleigh 散射系数（B）
(define bM 0.000016)                ; Mie 散射系数
(define bMe 0.0000016)              ; Mie 消光系数（= 0.1·bM）
(define Re1 6360001.0)              ; 观测点半径（海平面 + 1m）
(define SUNF 20.0)                  ; 太阳强度归一化系数
;; 二阶多次散射的视线采样数（原实现为 3；B2 优化降为 2，见 scatter_v 内注释）
(define V2 2)

;; ---- 相位函数 --------------------------------------------------------------
;; phaseR(μ) = 3/(16π)·(1+μ²)
;; phaseM(μ) = 3/(8π)·(1-g²)(1+μ²) / ((2+g²)·(1+g²-2gμ)^1.5)，g=0.76

;; 注意（B1）：scatter_v 热路径已把它们手工内联（旧版解释器每次 Scheme 调用都要
;; 复制整份环境，sky3 单采样点约 200 次调用，小函数的调用开销远超数学本体）。
;; 下列定义保留作参考实现/文档。
(define phase_r (lambda (mu) (* 0.05968310365946075 (+ 1.0 (* mu mu)))))
(define phase_m
  (lambda (mu) {
    (define base (- 1.5776 (* 1.52 mu)))
    (if (< base 0.0000000001) { (set! base 0.0000000001) })
    (* 0.1193662073189215
       (/ (* 0.4224 (+ 1.0 (* mu mu))) (* 2.5776 (Math.pow base 1.5))))
  }))

;; ---- HDR 色调映射：((√(1+ε))/2)·(√((1+x)²+ε) - √((1-x)²+ε))，ε=0.2 ------

(define hdr_map
  (lambda (x) {
    (define v (* 0.5477225575051661
                 (- (Math.sqrt (+ (* (+ 1.0 x) (+ 1.0 x)) 0.2))
                    (Math.sqrt (+ (* (- 1.0 x) (- 1.0 x)) 0.2)))))
    (if (> v 1.0) { (set! v 1.0) })
    (if (< v 0.0) { (set! v 0.0) })
    v
  }))

;; ---- 太阳方向（取自星历模块的实时值）--------------------------------------
;; 观测点：项目默认南京 119.0°E / 32.0°N / UTC+8（见 app/lib/ephem.scm）。
;; sun_x/sun_y/sun_z 为太阳单位向量（x 东、y 北、z 天顶），供散射模型使用；
;; sun_alt_deg 为太阳高度角（度），供二阶多次散射的 global_gain。

(define sun_x 0.0)
(define sun_y 0.0)
(define sun_z 1.0)
(define sun_alt_deg 90.0)
(define sun_az 0.0)                 ; 太阳方位角（北=0°，东=90°），显示用

;; ---- 时刻偏移（手动查看其他时刻）-------------------------------------------
;; 实时跟随模式下偏移为 0；点按屏幕左右两侧按钮 ±10 分钟后进入「手动」，画面不再
;; 随实时时间自动重绘（冻结在被选中的时刻），长按任一按钮回到实时跟随。
;; 偏移只作用在「取时」这一处：sun_from_ephem（定太阳）与 now_string（右下时刻），
;; 因此不动物理模型与渲染管线。
(define t_shift_min 0)              ; 相对实时的分钟偏移（10 的整数倍）
(define manual 0)                   ; 0=实时跟随，1=手动（画面冻结在该时刻）
(define cur_days 0.0)               ; 最近一次取时的 UTC 天数（已含偏移）
(define cur_sod 0.0)                ; 最近一次取时的 UTC 秒数（已含偏移）

;; 取当前时刻并叠加偏移，写入 cur_days/cur_sod（日/秒进位在这做掉）
(define now_shifted
  (lambda () {
    (define ld (Time.utc_days))
    (define ls (+ (Time.utc_sod) (* 60 t_shift_min)))
    (while (>= ls 86400) { (set! ls (- ls 86400)) (set! ld (+ ld 1)) })
    (while (< ls 0) { (set! ls (+ ls 86400)) (set! ld (- ld 1)) })
    (set! cur_days ld)
    (set! cur_sod ls)
  }))

(define sun_from_ephem
  (lambda () {
    (now_shifted)                   ; 实时 + 手动偏移
    (Ephem.set_time cur_days cur_sod)
    (Ephem.set_gmst)
    (Ephem.sun_equatorial)
    (Ephem.sun_altaz)
    (set! sun_az (Ephem.get_az))
    (set! sun_alt_deg (Ephem.get_alt))
    (define azr (Ephem.to_rad sun_az))
    (define elr (Ephem.to_rad sun_alt_deg))
    (define ca (Math.cos elr))
    (set! sun_x (* ca (Math.sin azr)))
    (set! sun_y (* ca (Math.cos azr)))
    (set! sun_z (Math.sin elr))
  }))

;; ---- 射线-球面求交：|o+t·d|²=r² ⇒ t²+2(o·d)t+(|o|²-r²)=0 ------------------
;; 有交点返回 1 并把两根写入 rt0/rt1，无交点返回 0。
;; 注意：热路径（march 的太阳射线、二阶每方向求交、视线出射根）均已内联求解
;; （复用已有的 |w|² 并省掉闭包调用），本函数与 rt0/rt1 仅作为参考实现保留。

(define rt0 0.0)
(define rt1 0.0)
;; 简化：把 b=2p、c=|o|²-r² 代入 disc=b²-4c=4(p²-c)，两根 t=p∓√(p²-c)——
;; 省掉 4 次乘法（2.0/4.0/两次 0.5），代数等价，浮点末位可能差 1 ulp。
(define rsi
  (lambda (ox oy oz dx dy dz rad) {
    (define p (+ (* ox dx) (+ (* oy dy) (* oz dz))))
    (define c (- (+ (* ox ox) (+ (* oy oy) (* oz oz))) (* rad rad)))
    (define disc (- (* p p) c))
    (if (< disc 0.0)
        { (set! rt0 0.0) (set! rt1 0.0) 0 }
        { (define sq (Math.sqrt disc))
          (set! rt0 (- (- 0.0 p) sq))
          (set! rt1 (- sq p))
          1 })
  }))

;; ---- 路径积分（Nishita 公式 4/5）：沿 o+t·d 求 Rayleigh/Mie 加权光学深度 --
;; nV 个主采样点，每点向太阳方向再取 nL 个采样点做地球遮蔽检测与透射率累积。
;; 结果写入 m_r0..m_r2（Rayleigh 三通道）与 m_m0..m_m2（Mie 三通道）。
;;
;; 性能：每个主采样点的"太阳射线求交"直接内联成 |w+t·sun|²=Ra² 的二次方程求解
;; （复用已算好的 |w|²=r2，不再调 rsi）：一次 rsi 调用约 80 条 VM 指令，其中
;; 闭包调用/传参/全局写回占 ~20，重复的 |w|² 与 Ra² 又占若干——内联后每个
;; 采样点省 40 余条指令，而整帧要跑 36+36+18 次。数值与 rsi 完全一致（同一公式、
;; 同一 |w|² 表达式），实测像素零差异。

(define m_r0 0.0)
(define m_r1 0.0)
(define m_r2 0.0)
(define m_m0 0.0)
(define m_m1 0.0)
(define m_m2 0.0)

(define march
  (lambda (ox oy oz dx dy dz t0 t1 nV nL) {
    (define seg (/ (- t1 t0) nV))
    (define i 0)
    (define j 0)
    (define t 0.0)
    (define wx 0.0) (define wy 0.0) (define wz 0.0) (define h 0.0)
    (define dR 0.0) (define dM 0.0) (define odR 0.0) (define odM 0.0)
    (define lSeg 0.0) (define lR 0.0) (define lM 0.0) (define blocked 0)
    (define lt 0.0) (define lx 0.0) (define ly 0.0) (define lz 0.0) (define lh 0.0)
    (define r2 0.0) (define sp 0.0) (define sdisc 0.0) (define st1 0.0)
    (define oR 0.0) (define oM 0.0)
    (define sr0 0.0) (define sr1 0.0) (define sr2 0.0)
    (define sm0 0.0) (define sm1 0.0) (define sm2 0.0)
    (set! odR 0.0) (set! odM 0.0)
    (set! sr0 0.0) (set! sr1 0.0) (set! sr2 0.0)
    (set! sm0 0.0) (set! sm1 0.0) (set! sm2 0.0)
    (while (< i nV) {
      (set! t (+ t0 (* (+ i 0.5) seg)))
      (set! wx (+ ox (* dx t)))
      (set! wy (+ oy (* dy t)))
      (set! wz (+ oz (* dz t)))
      (set! r2 (+ (* wx wx) (+ (* wy wy) (* wz wz))))
      (set! h (- (Math.sqrt r2) Re))
      (set! dR (* (Math.exp (- 0.0 (/ h Hr))) seg))
      (set! dM (* (Math.exp (- 0.0 (/ h Hm))) seg))
      (set! odR (+ odR dR))
      (set! odM (+ odM dM))
      (set! sp (+ (* wx sun_x) (+ (* wy sun_y) (* wz sun_z))))
      (set! sdisc (- (* sp sp) (- r2 (* Ra Ra))))
      (if (>= sdisc 0.0)
          { (set! st1 (- (Math.sqrt sdisc) sp))
            (if (> st1 0.0)
                {
                  (set! lSeg (/ st1 nL))
                  (set! lR 0.0) (set! lM 0.0) (set! blocked 0) (set! j 0)
                  (while (< j nL) {
                    (set! lt (* (+ j 0.5) lSeg))
                    (set! lx (+ wx (* sun_x lt)))
                    (set! ly (+ wy (* sun_y lt)))
                    (set! lz (+ wz (* sun_z lt)))
                    (set! lh (- (Math.sqrt (+ (* lx lx) (+ (* ly ly) (* lz lz)))) Re))
                    (if (< lh 0.0)
                        { (set! blocked 1) (set! j nL) }
                        { (set! lR (+ lR (* (Math.exp (- 0.0 (/ lh Hr))) lSeg)))
                          (set! lM (+ lM (* (Math.exp (- 0.0 (/ lh Hm))) lSeg)))
                          (set! j (+ j 1)) })
                  })
                  (if (== blocked 0)
                      {
                        (set! oR (+ odR lR))
                        (set! oM (+ odM lM))
                        (define e0 (Math.exp (- 0.0 (+ (* bR0 oR) (* bMe oM)))))
                        (define e1 (Math.exp (- 0.0 (+ (* bR1 oR) (* bMe oM)))))
                        (define e2 (Math.exp (- 0.0 (+ (* bR2 oR) (* bMe oM)))))
                        (set! sr0 (+ sr0 (* e0 dR)))
                        (set! sr1 (+ sr1 (* e1 dR)))
                        (set! sr2 (+ sr2 (* e2 dR)))
                        (set! sm0 (+ sm0 (* e0 dM)))
                        (set! sm1 (+ sm1 (* e1 dM)))
                        (set! sm2 (+ sm2 (* e2 dM)))
                      })
                })
          })
      (set! i (+ i 1))
    })
    (set! m_r0 sr0) (set! m_r1 sr1) (set! m_r2 sr2)
    (set! m_m0 sm0) (set! m_m1 sm1) (set! m_m2 sm2)
  }))

;; ---- 上半球 9 个固定采样方向（归一化；天顶、偏东/西/北/南上、四个斜向）----

(define amb_x '(0.0 0.8944272 -0.8944272 0.0 0.0 0.6666667 -0.6666667 0.6666667 -0.6666667))
(define amb_y '(0.0 0.0 0.0 0.8944272 -0.8944272 0.6666667 0.6666667 -0.6666667 -0.6666667))
(define amb_z '(1.0 0.4472136 0.4472136 0.4472136 0.4472136 0.3333333 0.3333333 0.3333333 0.3333333))

;; ---- 每帧常量：太阳位置一变就要重算，但与视线/采样点无关的东西全部预先算好 ----------
;; 这些量原本在每个采样点里重算（约 1850 个采样点 × V2×9 次），数值完全一样：
;;   dom_g                    二阶多次散射的全球照射度（只依赖太阳高度）
;;   akr0/akr1/akr2/akm[k]   第 k 个环境方向的 "SUNF·β·相位·dom_g" 系数（只依赖方向与太阳）
;; 装订点：启动时一次、每次 advance（太阳移动重绘）时一次。
(define dom_g 0.0)
(define akr0 '())
(define akr1 '())
(define akr2 '())
(define akm '())

(define frame_setup
  (lambda () {
    (define sal (* (/ sun_alt_deg 10.0) (/ sun_alt_deg 10.0)))
    (set! dom_g (* (* 2.0 (/ PI 9.0)) (+ (* 14.0 (Math.exp (- 0.0 sal))) 2.0)))
    ;; 新建列表：模块级 '() 字面量是静态槽，反复 push 会跨帧累积
    (define l0 (cdr (cons 0.0 '())))
    (define l1 (cdr (cons 0.0 '())))
    (define l2 (cdr (cons 0.0 '())))
    (define lm (cdr (cons 0.0 '())))
    (define k 0)
    (while (< k 9) {
      (define adx (get_item amb_x k))
      (define ady (get_item amb_y k))
      (define adz (get_item amb_z k))
      (define mu (+ (* adx sun_x) (+ (* ady sun_y) (* adz sun_z))))
      (define m1 (+ 1.0 (* mu mu)))
      (define p_r (* 0.05968310365946075 m1))
      (define base (- 1.5776 (* 1.52 mu)))
      (if (< base 0.0000000001) { (set! base 0.0000000001) })
      (define p_m (* 0.1193662073189215
                    (/ (* 0.4224 m1) (* 2.5776 (* base (Math.sqrt base))))))
      (push l0 (* (* SUNF bR0) (* p_r dom_g)))
      (push l1 (* (* SUNF bR1) (* p_r dom_g)))
      (push l2 (* (* SUNF bR2) (* p_r dom_g)))
      (push lm (* (* SUNF bM) (* p_m dom_g)))
      (set! k (+ k 1))
    })
    (set! akr0 l0)
    (set! akr1 l1)
    (set! akr2 l2)
    (set! akm lm)
  }))

;; ---- 单视线方向的散射颜色（对应 scatter_model_3 主体）----------------------
;; 输入视线单位向量 (vx,vy,vz) 与视线仰角 elev_deg；结果写入 out_r/out_g/out_b。

(define out_r 0.0)
(define out_g 0.0)
(define out_b 0.0)

(define scatter_v
  (lambda (vx vy vz elev_deg) {
    (define tmin 0.0)
    (define tmax 0.0)
    (define c0 0.0) (define c1 0.0) (define c2 0.0)
    (define seg_v 0.0)
    (define od_r 0.0) (define od_m 0.0)
    (define dR2 0.0) (define dM2 0.0)
    (define i 0) (define k 0)
    (define t 0.0) (define wx 0.0) (define wy 0.0) (define wz 0.0) (define h 0.0)
    (define tv0 0.0) (define tv1 0.0) (define tv2 0.0)
    (define adx 0.0) (define ady 0.0) (define adz 0.0)
    (define mu_s 0.0)
    (define r2 0.0) (define ap 0.0) (define sdisc 0.0) (define st1 0.0)
    (define p_rs 0.0) (define p_ms 0.0)
    (define sg0 0.0) (define sg1 0.0) (define sg2 0.0)
    (define mu1 0.0)
    ;; B1 内联临时量：phase_m 的 (1+g²−2gμ) 底数、HDR 映射中间值
    (define pmb 0.0) (define hq 0.0)
    ;; 观测点在大气内部：入射根 t0<0，故积分起点 tMin=0；终点为大气出射根
    ;; 内联 rsi（o=(0,0,Re1)）：p=Re1·vz、c=Re1²-R² ⇒ t=p∓√(p²+c的相反数)
    (define cz (* Re1 vz))
    (define cz2 (* cz cz))
    (set! tmin 0.0)
    (set! tmax (- (Math.sqrt (+ cz2 (- (* Ra Ra) (* Re1 Re1)))) cz))
    ;; 地球遮挡（朝天顶的视线不会命中，保留与原实现一致的判据）
    (define dE (+ cz2 (- (* Re Re) (* Re1 Re1))))
    (if (>= dE 0.0)
        { (define rt0v (- (- 0.0 cz) (Math.sqrt dE)))
          (if (> rt0v 1.0) { (if (< rt0v tmax) { (set! tmax rt0v) }) }) })
    (if (<= tmax tmin)
        { (set! out_r 0.0) (set! out_g 0.0) (set! out_b 0.0) }
        {
          ;; ---- 一阶单次散射 ----
          (set! mu1 (+ (* vx sun_x) (+ (* vy sun_y) (* vz sun_z))))
          ;; B1：phase_r / phase_m 内联（原每采样点 2 次小函数调用；表达式与函数体
          ;; 逐字一致，数值结果不变）
          (define m1_1 (+ 1.0 (* mu1 mu1)))
          (define p_r1 (* 0.05968310365946075 m1_1))
          (set! pmb (- 1.5776 (* 1.52 mu1)))
          (if (< pmb 0.0000000001) { (set! pmb 0.0000000001) })
          (define p_m1 (* 0.1193662073189215
                         (/ (* 0.4224 m1_1)
                            (* 2.5776 (* pmb (Math.sqrt pmb))))))
          (march 0.0 0.0 Re1 vx vy vz tmin tmax 36 1)
          (set! c0 (* SUNF (+ (* (* m_r0 bR0) p_r1) (* (* m_m0 bM) p_m1))))
          (set! c1 (* SUNF (+ (* (* m_r1 bR1) p_r1) (* (* m_m1 bM) p_m1))))
          (set! c2 (* SUNF (+ (* (* m_r2 bR2) p_r1) (* (* m_m2 bM) p_m1))))
          ;; ---- 二阶多次散射：视线路径 V2 个采样点 × 上半球 9 个方向 ----
          ;; B2：V2 由 3 降为 2。二阶多次散射项沿视线方向是低频量（它是"各方向环境
          ;; 辐照度沿视线的加权积分"），中点法取 2 个采样点的误差远小于 8 位色量化步长
          ;; （实测整帧最大通道偏差见提交说明）。这是一处显式的画质/性能折衷。
          (set! seg_v (/ (- tmax tmin) V2))
          (set! od_r 0.0) (set! od_m 0.0) (set! i 0)
          (while (< i V2) {
            (set! t (+ tmin (* (+ i 0.5) seg_v)))
            (set! wx (* vx t))
            (set! wy (* vy t))
            (set! wz (+ Re1 (* vz t)))
            (set! r2 (+ (* wx wx) (+ (* wy wy) (* wz wz))))
            (set! h (- (Math.sqrt r2) Re))
            (if (>= h 0.0)
                {
                  (set! dR2 (* (Math.exp (- 0.0 (/ h Hr))) seg_v))
                  (set! dM2 (* (Math.exp (- 0.0 (/ h Hm))) seg_v))
                  (set! od_r (+ od_r dR2))
                  (set! od_m (+ od_m dM2))
                  (set! tv0 (Math.exp (- 0.0 (+ (* bR0 od_r) (* bMe od_m)))))
                  (set! tv1 (Math.exp (- 0.0 (+ (* bR1 od_r) (* bMe od_m)))))
                  (set! tv2 (Math.exp (- 0.0 (+ (* bR2 od_r) (* bMe od_m)))))
                  (set! k 0)
                  (while (< k 9) {
                    (set! adx (get_item amb_x k))
                    (set! ady (get_item amb_y k))
                    (set! adz (get_item amb_z k))
                    (set! ap (+ (* wx adx) (+ (* wy ady) (* wz adz))))
                    (set! sdisc (- (* ap ap) (- r2 (* Ra Ra))))
                    (if (>= sdisc 0.0)
                        { (set! st1 (- (Math.sqrt sdisc) ap))
                          (if (> st1 0.0)
                              {
                                (march wx wy wz adx ady adz 0.0 st1 2 2)
                                ;; 方向相位项与其 SUNF·β 系数已由 frame_setup 预算好
                                (define la0 (+ (* m_r0 (get_item akr0 k)) (* m_m0 (get_item akm k))))
                                (define la1 (+ (* m_r1 (get_item akr1 k)) (* m_m1 (get_item akm k))))
                                (define la2 (+ (* m_r2 (get_item akr2 k)) (* m_m2 (get_item akm k))))
                                ;; 相位函数方向约定：入射光子 −ω、出射光子 −d ⇒ μ=dot(ω,d)
                                (set! mu_s (+ (* adx vx) (+ (* ady vy) (* adz vz))))
                                (define ms1 (+ 1.0 (* mu_s mu_s)))
                                (set! p_rs (* 0.05968310365946075 ms1))
                                (set! pmb (- 1.5776 (* 1.52 mu_s)))
                                (if (< pmb 0.0000000001) { (set! pmb 0.0000000001) })
                                (set! p_ms (* 0.1193662073189215
                                             (/ (* 0.4224 ms1)
                                                (* 2.5776 (* pmb (Math.sqrt pmb))))))
                                (define sgm (* (* bM p_ms) dM2))
                                (set! sg0 (+ (* (* bR0 p_rs) dR2) sgm))
                                (set! sg1 (+ (* (* bR1 p_rs) dR2) sgm))
                                (set! sg2 (+ (* (* bR2 p_rs) dR2) sgm))
                                (set! c0 (+ c0 (* (* la0 sg0) tv0)))
                                (set! c1 (+ c1 (* (* la1 sg1) tv1)))
                                (set! c2 (+ c2 (* (* la2 sg2) tv2)))
                              })
                        })
                    (set! k (+ k 1))
                  })
                })
            (set! i (+ i 1))
          })
          ;; ---- 臭氧吸收 ----
          (define oz (* 90.0 (/ (- tmax tmin) Ra)))
          (set! c0 (* c0 (Math.exp (- 0.0 (* 0.020 oz)))))
          (set! c1 (* c1 (Math.exp (- 0.0 (* 0.040 oz)))))
          ;; ---- 夜天光 ----
          (define nl (- 1.0 (/ elev_deg 120.0)))
          (set! c0 (+ c0 (* 0.05 nl)))
          (set! c1 (+ c1 (* 0.06 nl)))
          (set! c2 (+ c2 (* 0.12 nl)))
          ;; ---- HDR 映射 + 钳位（min(1, ·) × 255）----
          ;; B1：hdr_map 内联（原每采样点 3 次小函数调用）
          (set! hq (* 0.5477225575051661
                      (- (Math.sqrt (+ (* (+ 1.0 c0) (+ 1.0 c0)) 0.2))
                         (Math.sqrt (+ (* (- 1.0 c0) (- 1.0 c0)) 0.2)))))
          (if (> hq 1.0) { (set! hq 1.0) })
          (if (< hq 0.0) { (set! hq 0.0) })
          (set! out_r (* 255.0 hq))
          (set! hq (* 0.5477225575051661
                      (- (Math.sqrt (+ (* (+ 1.0 c1) (+ 1.0 c1)) 0.2))
                         (Math.sqrt (+ (* (- 1.0 c1) (- 1.0 c1)) 0.2)))))
          (if (> hq 1.0) { (set! hq 1.0) })
          (if (< hq 0.0) { (set! hq 0.0) })
          (set! out_g (* 255.0 hq))
          (set! hq (* 0.5477225575051661
                      (- (Math.sqrt (+ (* (+ 1.0 c2) (+ 1.0 c2)) 0.2))
                         (Math.sqrt (+ (* (- 1.0 c2) (- 1.0 c2)) 0.2)))))
          (if (> hq 1.0) { (set! hq 1.0) })
          (if (< hq 0.0) { (set! hq 0.0) })
          (set! out_b (* 255.0 hq))
        })
  }))
;; ---- 单采样点：视线方向 → 散射颜色（写入 out_r/out_g/out_b）----------------
;; 参数：px/py 采样块左上角像素、dx/dy 采样块中心相对屏幕中心的偏移。
;; 视线向量按「抬头看天」视角：屏幕上方=北、屏幕左侧=东（见文件头）。

;; 计算一个采样点的三通道颜色，结果写入 out_r/out_g/out_b（不再负责绘制：
;; 绘制统一由双线性重建 blit_band 完成，见下）
(define shade
  (lambda (dx dy) {
    (define r (Math.sqrt (+ (* dx dx) (* dy dy))))
    (define sin_t 0.0)
    (define vz 1.0)
    (define vx 0.0)
    (define vy 0.0)
    (define elev 90.0)
    (if (> r 0.001)
        { (set! sin_t (Math.sin (* r K_THETA)))
          (set! vz (Math.cos (* r K_THETA)))
          (define inv_r (/ 1.0 r))
          ;; 屏幕上方=北（dy 向下，故取负得 +y）、屏幕左侧=东（dx 向右，故取负得 +x）
          (set! vx (* (- 0.0 sin_t) (* dx inv_r)))
          (set! vy (* (- 0.0 sin_t) (* dy inv_r)))
          (set! elev (- 90.0 (* r K_DEG)))
        })
    (scatter_v vx vy vz elev)
  }))

;; ---- 采样网格 + 双线性重建（消除 SS×SS 色块的马赛克感）---------------------
;; 采样阶段只把每个采样点的三通道颜色推进三张列表（行主序，索引 = 采样行*COLS_S+采样列）；
;; 绘制阶段对每个像素取相邻 4 个采样点做双线性插值后逐像素写点（圆外像素不画，保持
;; Gfx.clear 的黑）。天空辐照度是平滑的低频场，双线性重建在视觉上等同于提高采样密度，
;; 而模型求值次数（几乎全部成本）不变——实测整帧只增加约 +3% 的指令。
;; 采样点坐标约定：采样行 sy / 采样列 sx 对应像素 (sx*SS, sy*SS)，故像素 (x,y) 的
;; 双线性参数为 ux=x/SS、uy=y/SS。

(define GRID_N (* ROWS_S COLS_S))
(define pad_grid
  (lambda (acc n) {
    (define i 0)
    (while (< i n) { (push acc 0.0) (set! i (+ i 1)) })
    acc
  }))
;; 每个 '() 字面量只 push 一次（模块级字面量是静态槽，反复 push 会累积）
(define grid_r (pad_grid '() GRID_N))
(define grid_g (pad_grid '() GRID_N))
(define grid_b (pad_grid '() GRID_N))
(define INV_SS (/ 1.0 SS))
(define INV_2R (/ 1.0 (* 2.0 R_SCR)))
(define GRID_ROW_LAST (- ROWS_S 1))
(define GRID_COL_LAST (- COLS_S 1))

;; 画一条采样带：输出行 [b*SS, (b+1)*SS)，使用采样行 b 与 b+1（越界时钳位到末行）。
;; 每条采样带在对应采样行算完后调用一次 → 保持逐带渐进上屏。
(define blit_band
  (lambda (b) {
    ;; 全部局部量在函数级声明 + set!（与本文件其它热路径一致；避免在深层块里 define）
    (define jA 0) (define jB 0)
    (define baseA 0) (define baseB 0)
    (define cur_r 0.0) (define cur_g 0.0) (define cur_b 0.0)
    (define nxt_r 0.0) (define nxt_g 0.0) (define nxt_b 0.0)
    (define k 0) (define y 0) (define ty 0.0) (define w0 0.0)
    (define dyc 0.0) (define dyc2 0.0)
    (define sx 0) (define sc 0)
    (define dr 0.0) (define dg 0.0) (define db 0.0)
    (define kx 0) (define x 0)
    (define ddx 0.0) (define r2p 0.0) (define cov 0.0)
    (set! jA b)
    (set! jB (+ b 1))
    (if (> jB GRID_ROW_LAST) { (set! jB GRID_ROW_LAST) })
    (set! baseA (* jA COLS_S))
    (set! baseB (* jB COLS_S))
    (set! k 0)
    (while (< k SS) {
      (set! y (+ (* b SS) k))
      (set! ty (* k INV_SS))
      (set! w0 (- 1.0 ty))
      (set! dyc (- (+ y 0.5) CY))
      (set! dyc2 (* dyc dyc))
      (set! cur_r (+ (* w0 (get_item grid_r baseA)) (* ty (get_item grid_r baseB))))
      (set! cur_g (+ (* w0 (get_item grid_g baseA)) (* ty (get_item grid_g baseB))))
      (set! cur_b (+ (* w0 (get_item grid_b baseA)) (* ty (get_item grid_b baseB))))
      (set! sx 0)
      (while (< sx COLS_S) {
        (set! sc (+ sx 1))
        (if (> sc GRID_COL_LAST) { (set! sc GRID_COL_LAST) })
        (set! nxt_r (+ (* w0 (get_item grid_r (+ baseA sc))) (* ty (get_item grid_r (+ baseB sc)))))
        (set! nxt_g (+ (* w0 (get_item grid_g (+ baseA sc))) (* ty (get_item grid_g (+ baseB sc)))))
        (set! nxt_b (+ (* w0 (get_item grid_b (+ baseA sc))) (* ty (get_item grid_b (+ baseB sc)))))
        (set! dr (* (- nxt_r cur_r) INV_SS))
        (set! dg (* (- nxt_g cur_g) INV_SS))
        (set! db (* (- nxt_b cur_b) INV_SS))
        (set! kx 0)
        (while (< kx SS) {
          (set! x (+ (* sx SS) kx))
          (set! ddx (- (+ x 0.5) CX))
          (set! r2p (+ (* ddx ddx) dyc2))
          (if (<= r2p R_SCR2)
              {
                (set! cov (* (- R_SCR2 r2p) INV_2R))
                (if (> cov 1.0) { (set! cov 1.0) })
                (Gfx.set_pixel x y
                  (* cov (+ cur_r (* kx dr)))
                  (* cov (+ cur_g (* kx dg)))
                  (* cov (+ cur_b (* kx db))))
              })
          (set! kx (+ kx 1))
        })
        (set! cur_r nxt_r) (set! cur_g nxt_g) (set! cur_b nxt_b)
        (set! sx (+ sx 1))
      })
      (set! k (+ k 1))
    })
  }))


;; ---- 渲染状态：逐采样点推进（每画完一个采样行推屏一次）--------------------

(define cur_x 0)
(define cur_y 0)
(define hold_until 0)

(define render_chunk
  (lambda () {
    (define painted 0)
    (while (and (< painted SAMPLES_PER_STEP) (< cur_y ROWS_S)) {
      (define dx (- (+ (* cur_x SS) HALF_S) CX))
      (define dy (- (+ (* cur_y SS) HALF_S) CY))
      (if (<= (+ (* dx dx) (* dy dy)) R_SCR2)
          { (shade dx dy) (set! painted (+ painted 1)) }
          { ;; 圆外采样点也占网格一格（黑色），保证索引对齐；不入绘制
            (set! out_r 0.0) (set! out_g 0.0) (set! out_b 0.0) })
      ;; push 是"原地追加、返回 #undefined"的语句型内建，不能写成 (set! x (push x v))
      (set_item! grid_r (+ (* cur_y COLS_S) cur_x) out_r)
      (set_item! grid_g (+ (* cur_y COLS_S) cur_x) out_g)
      (set_item! grid_b (+ (* cur_y COLS_S) cur_x) out_b)
      (set! cur_x (+ cur_x 1))
      (if (>= cur_x COLS_S)
          { (set! cur_x 0)
            (set! cur_y (+ cur_y 1))
            ;; 采样行 s 算完后，采样带 s-1（需采样行 s-1 与 s）才可重建并上屏
            (if (>= cur_y 2) { (blit_band (- cur_y 2)) })
            (Gfx.refresh)
            ;; 行边界顺带采一次触屏：长帧渲染途中点按钮也能在一行内（~50ms）响应，
            ;; 响应即把 cur_x/cur_y 复位、清屏并重画按钮（见 touch_poll → advance）。
            (touch_poll) })
    })
    (if (>= cur_y ROWS_S)
        {
          ;; 末尾一条带（最后采样行以下）用末行钳位补画
          (blit_band GRID_ROW_LAST)
          ;; 本帧天空完成：记下太阳、画字幕、推屏，并进入停留/重绘判断
          (set! hold_until (+ (System.timestamp) HOLD_MS))
          (draw_caption)
          (Gfx.refresh)
        })
  }))

;; ---- 渲染控制：实时太阳 + 移动判据 + 屏幕字幕 -----------------------------

(define RE_RENDER_DEG 0.4)          ; 太阳移动超过该角度（度）后重绘
(define HOLD_CHECK_MS 700)          ; 停留期间检查太阳位置的间隔

(define rendered_az 0.0)            ; 本帧天空对应的太阳方位/高度
(define rendered_el 0.0)

;; 读一次实时太阳位置（不改变本帧渲染所用的 sun_*），判断是否已明显移动
(define sun_moved_enough
  (lambda () {
    (now_shifted)
    (Ephem.set_time cur_days cur_sod)
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

;; 当前本地时间 "YYYY-MM-DD hh:mm:ss"（北京时间 = UTC+8；含手动偏移）
(define now_string
  (lambda () {
    (now_shifted)
    (define ld cur_days)
    (define ls (+ cur_sod Ephem.TZ_SEC))
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

;; ---- 右下角时间：透明底（从采样网格双线性重建还原背景）--------------------
;; 时钟块横跨天空圆界（左半落在圆内的天空上、右半落在圆外的黑底上），因此每次
;; 重画前先把这块区域的“真实背景”恢复出来，而不是整块填黑：直接对采样网格做
;; 双线性重建（与 blit_band 同一套插值与圆界判定），恢复出来的就是当前帧画面。
;; 逐像素写点仅 137×15≈2055 点、每秒一次，开销可忽略（远低于旧实现"重算 10~20 个
;; 采样点"所需的数十万条 VM 指令）。

(define repaint_clock_bg
  (lambda () {
    (define y hud_clock_y)
    (while (< y (+ hud_clock_y hud_clock_h)) {
      (define uy (* y INV_SS))
      (define j0 (Math.floor uy))
      (define ty (- uy j0))
      (if (> j0 GRID_ROW_LAST) { (set! j0 GRID_ROW_LAST) (set! ty 0.0) })
      (define j1 (+ j0 1))
      (if (> j1 GRID_ROW_LAST) { (set! j1 GRID_ROW_LAST) })
      (define base0 (* j0 COLS_S))
      (define base1 (* j1 COLS_S))
      (define w0 (- 1.0 ty))
      (define ddy (- (+ y 0.5) CY))
      (define x hud_clock_x)
      (while (< x (+ hud_clock_x hud_clock_w)) {
        (define ddx (- (+ x 0.5) CX))
        (if (<= (+ (* ddx ddx) (* ddy ddy)) R_SCR2)
            {
              (define ux (* x INV_SS))
              (define i0 (Math.floor ux))
              (define tx (- ux i0))
              (if (> i0 GRID_COL_LAST) { (set! i0 GRID_COL_LAST) (set! tx 0.0) })
              (define i1 (+ i0 1))
              (if (> i1 GRID_COL_LAST) { (set! i1 GRID_COL_LAST) })
              (define t0 (- 1.0 tx))
              ;; 与 blit_band 用同一个圆界覆盖率：时钟块压在圆周上（框内最近点距圆心 106.7 < 120），
              ;; 若这里用硬判定，每秒重绘会把那段 1px 抗锯齿带覆盖回"全亮"，出现一条接缝
              (define cov (* (- R_SCR2 (+ (* ddx ddx) (* ddy ddy))) INV_2R))
              (if (> cov 1.0) { (set! cov 1.0) })
              (Gfx.set_pixel x y
                (* cov (+ (* t0 (+ (* w0 (get_item grid_r (+ base0 i0))) (* ty (get_item grid_r (+ base1 i0)))))
                          (* tx (+ (* w0 (get_item grid_r (+ base0 i1))) (* ty (get_item grid_r (+ base1 i1)))))))
                (* cov (+ (* t0 (+ (* w0 (get_item grid_g (+ base0 i0))) (* ty (get_item grid_g (+ base1 i0)))))
                          (* tx (+ (* w0 (get_item grid_g (+ base0 i1))) (* ty (get_item grid_g (+ base1 i1)))))))
                (* cov (+ (* t0 (+ (* w0 (get_item grid_b (+ base0 i0))) (* ty (get_item grid_b (+ base1 i0)))))
                          (* tx (+ (* w0 (get_item grid_b (+ base0 i1))) (* ty (get_item grid_b (+ base1 i1))))))))
            }
            { (Gfx.set_pixel x y 0 0 0) })
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
  }))

;; 右下角时间：先恢复该块的真实背景（透明底），再画时间
(define draw_clock
  (lambda () {
    (repaint_clock_bg)
    (Gfx.draw_text (now_string) hud_clock_x (+ hud_clock_y 2) 190 202 214)
    (Gfx.refresh)
  }))

;; 每秒刷新一次时间（仅在天空渲染完成后调用，避免被逐行绘制覆盖）
;; 手动模式下冻结：画面停在被选中的时刻，时钟不再走。
(define clock_tick
  (lambda () {
    (if (== manual 0) {
      (define us (Time.utc_sod))
      (if (not (== us last_sec)) { (set! last_sec us) (draw_clock) })
    })
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

;; ---- 触屏按钮：屏左边缘 = 时刻 −10 分钟，屏右边缘 = +10 分钟 ------------------
;; 位置选择：天空圆是 240×320 屏的内切圆（半径 120、圆心居中），屏幕两侧约 40px
;; 恒为黑、且逐行双线性重建只画圆内像素，所以按钮画一次不会被覆盖；纵向放在
;; 中部（y=94~146）也避开了左上角 80×60 的系统宫格（触屏 esc 格）。
(define BTN_W 24)
(define BTN_H 52)
(define BTN_M 3)
(define BTN_LX BTN_M)
(define BTN_RX (- SW (+ BTN_M BTN_W)))
(define BTN_Y (Math.floor (- (/ SH 2.0) (/ BTN_H 2.0))))
(define BTN_HOLD_MS 700)            ; 按住超过该时长 = 回到实时跟随

(define has_touch 0)                ; 宿主是否提供触屏（无触屏时不画按钮）
(define btn_dn_at 0)                ; 本次按下的时刻（长按判定用）
(define btn_dn_hold 0)              ; 本次按下是否已触发过长按

;; 偏移量文本：+20 / -10（0 时为 +0，但提示行里 0 会特判成“手动”不带数字）
(define off_str
  (lambda (v)
    (if (>= v 0) (String.concat "+" (String.atom_to_string v)) (String.atom_to_string v))))

;; 手动模式提示文本（偏移为 0 时只显示“手动”，避免“手动+0分”这种碎碎念）
(define manual_txt
  (lambda ()
    (if (== t_shift_min 0)
        "手动"
        (String.concat "手动" (String.concat (off_str t_shift_min) "分")))))

;; 单个侧边按钮：方向 dir=1 向右（快进）、-1 向左（快退）；label 为 "+10" / "-10"
(define draw_button
  (lambda (bx dir label) {
    (define cr 52) (define cg 78) (define cb 110)
    ;; 手动模式下边框转金：一眼能看出「已不跟随实时」
    (if (== manual 1) { (set! cr 200) (set! cg 156) (set! cb 64) })
    (Gfx.fill_rect bx BTN_Y BTN_W BTN_H 20 26 38)
    (Gfx.draw_rect bx BTN_Y BTN_W BTN_H cr cg cb)
    ;; 三角箭头（宽 8、半高 8）
    (define cx (+ bx 12))
    (define cy (+ BTN_Y 15))
    (define i 0)
    (while (< i 8) {
      (define half (- 8 i))
      (define xx (if (== dir 1) (+ (- cx 4) i) (- (+ cx 4) i)))
      (Gfx.fill_rect xx (- cy half) 1 (* 2 half) 150 190 230)
      (set! i (+ i 1))
    })
    (Gfx.draw_text label (+ bx 2) (+ BTN_Y 30) 150 190 230)
  }))

(define draw_buttons
  (lambda () {
    (if (== has_touch 1)
        { (draw_button BTN_LX -1 "-10")
          (draw_button BTN_RX 1 "+10") })
  }))

;; 手动模式的常驻提示：画在圆外左上黑区（x<96 且 y<45 的像素永不参与逐行重建），
;; 所以从起帧到上一帧完成都能看到，交互反馈不靠整帧结束。
(define draw_manual_hint
  (lambda () {
    (if (== manual 1)
        { (Gfx.draw_text (manual_txt) 4 18 224 196 128)
          (Gfx.draw_text "长按回实时" 4 33 150 142 112) })
  }))

;; 点按命中判定（返回 #t/#f；注意本语言只有 #f 为假，不能返回 0/1）
(define in_btn
  (lambda (px py bx)
    (and (>= px bx) (< px (+ bx BTN_W)) (>= py BTN_Y) (< py (+ BTN_Y BTN_H)))))

;; 采一次触屏并处理按钮：
;;   点按左/右按钮 → t_shift_min ∓10 分钟，进入手动模式，重新起帧；
;;   按住任一按钮 ≥BTN_HOLD_MS → 偏移归零，回到实时跟随。
;; 返回值：1 表示本拍已换时刻并重新起帧（调用方据此跳过本拍的停留逻辑）。
(define touch_poll
  (lambda () {
    (define acted 0)
    (if (== has_touch 1)
        {
          (define pressed (Touch.is_pressed))
          (define px (Touch.x))
          (define py (Touch.y))
          (if (Touch.take_press)
              {
                (if (in_btn px py BTN_LX)
                    { (set! btn_dn_at (System.timestamp)) (set! btn_dn_hold 0)
                      (set! t_shift_min (- t_shift_min 10)) (set! manual 1) (set! acted 1) }
                    { (if (in_btn px py BTN_RX)
                          { (set! btn_dn_at (System.timestamp)) (set! btn_dn_hold 0)
                            (set! t_shift_min (+ t_shift_min 10)) (set! manual 1) (set! acted 1) }) })
              })
          (if (and pressed (== btn_dn_hold 0) (> btn_dn_at 0))
              { (if (> (- (System.timestamp) btn_dn_at) BTN_HOLD_MS)
                    { (set! btn_dn_hold 1) (set! t_shift_min 0) (set! manual 0) (set! acted 1) }) })
          (if (not pressed) { (set! btn_dn_at 0) })
        })
    (if (== acted 1) { (advance) })
    acted
  }))

(define advance
  (lambda () {
    (sun_from_ephem)
    (frame_setup)                     ; 太阳变了 → 重装帧级常量
    (set! rendered_az sun_az)
    (set! rendered_el sun_alt_deg)
    (set! cur_x 0)
    (set! cur_y 0)
    (set! last_sec -1)                ; 时刻变了 → 右下时钟重画
    (Gfx.clear)
    (draw_buttons)                    ; 清屏后重画侧边按钮（圆外黑底，后续逐行不会覆盖）
    (draw_manual_hint)
  }))

(define step
  (lambda () {
    (define acted (touch_poll))       ; 触屏按钮在任何阶段都响应
    (if (< cur_y ROWS_S)
        { (render_chunk) }
        { (if (== acted 1)
              { #undefined }          ; 刚换了时刻并已重新起帧，本拍不再做停留逻辑
              { (clock_tick)          ; 右下角时间每秒走一次（手动模式冻结）
                (if (> (System.timestamp) hold_until)
                    { (if (== manual 0)
                          { (if (== (sun_moved_enough) 1)
                                { (advance) }
                                { (set! hold_until (+ (System.timestamp) HOLD_CHECK_MS)) }) }
                          { (set! hold_until (+ (System.timestamp) HOLD_CHECK_MS)) }) }) }) })
  }))

;; ---- 启动：实时太阳 → 清屏（含按钮）→ 步进循环 ------------------------------

(set! has_touch (if (Touch.is_available) 1 0))
(set! t_shift_min 0)
(set! manual 0)
(advance)
(set! update_timer (System.set_interval 16 step))
