;; ============================================================================
;; FLIP 流体（小规模 Scheme 版）—— 灵机应用程序
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库 + Diag 本地库
;;
;; 算法照搬 infer/flip.c（Ten Minute Physics 的 FLIP），按解释器的算力做了裁剪：
;;   ① 积分：粒子速度加"重力"后推进位置，撞四壁则钳位；
;;   ② P2G：把粒子速度按双线性权重散布到 MAC 网格（u 在格心偏下、v 在格心偏右），
;;      同时累加权重分母；权重为 0 的格保持静止；
;;   ③ 归一化：网格速度 = 累加值 / 权重和；
;;   ④ 压力求解：对每个"流体格"做 Gauss-Seidel 松弛，使速度场近似无散
;;      （C 版用 Jacobi 快照，这里改成原地 Gauss-Seidel：省掉每轮两次全表拷贝，
;;       收敛还更快）；
;;   ⑤ G2P：把网格速度双线性插值回粒子，用 FLIP 比例混合
;;      v ← (1-flip)·PIC + flip·(v_old + (网格新 − 网格旧))。
;;
;; 与 C 版的差异（都是为省算力）：
;;   · 不做 push_particles_apart（粒子分离）与 drift 补偿（那还要再跑一遍密度 P2G）；
;;   · 不存压力数组（只用于上色，这里改成按粒子速度上色）；
;;   · 容器就是整个画布：四周边界格固定为固体，没有沙漏/瓶颈节流。
;;
;; 重力方向缓慢随机游走（`gang` 每拍加一个随机小量），于是流体在容器里来回晃、
;; 翻卷、贴壁爬——正好把这个算法最有趣的流动现象演出来。
;;
;; 最终参数与实测（host x86-64）：
;;   网格 26×18（每格 12 px）、粒子 **158**（RAD=0.51，约 1.0 粒/格）、
;;   压力迭代 12 轮 + **粒子分离**（桶排序 + 3×3 邻居，照 flip.c 的 push_particles_apart）；
;;   平均约 **20 ms/拍** → 40 ms 定时器下约 50% CPU；浏览器 wasm32 约 24 ms/拍。
;;
;; 重力源两种，由**左下角按钮**（触屏点按）切换：
;;   ① 自动（缺省）= **洗衣机滚筒**：重力方向连续旋转 360°（不是来回摆），转速按
;;      "洗涤（慢转）→ 加速 → 脱水（快转）→ 减速" 循环（周期 900 拍 ≈ 45 s@20fps）：
;;        drum_ang += ω，ω 在 0.030（约 6 s 一圈）~ 0.170（约 1.1 s 一圈）之间
;;        gx = gm·sin(drum_ang)，gy = gm·cos(drum_ang)，脱水时 gm = 2.6×G_RATE
;;        另有离心项 cent_k·(p − 筒心)：脱水时把水压向筒壁，中间留空 —— 滚筒甩干的样子。
;;   ② 重力感应 = `Imu` 本地库给出的**屏幕系重力分量**（1G = 9.8，x 向右、y 向下，与
;;      gx/gy 同向）实时驱动流体；离心项归零，重力大小恒 1G、方向随设备倾斜实时变化。
;;      无传感器/未授权姿态时 Imu 给出缺省 (0, +9.8)（屏幕向下），按钮文字转"无传感器"。
;;   左上角的重力矢量箭头始终指示**当前生效的重力**（方向 + 大小）；重力感应模式下
;;   面板与按钮改用绿色系，一眼区分重力源。
;;
;; 小黄鸭（浮动障碍物）：
;;   素材来自 openclipart.org/detail/325903/rubber-duck（CC0 / 公有领域，透明底 PNG），
;;   离线重采样成 30×31 的字符位图，加载时展开成调色板索引表；绘制时按当前倾角
;;   逐像素旋转贴图（透明像素跳过），画在粒子之前，于是没入水中的部分会被水盖住。
;;   物理（刚体：牛顿第二定律 + 阿基米德浮力），全部在鸭身局部坐标系里算：
;;     · 水面测量：在鸭身左右各取两条采样带，记"沿当地向上 u 最高的粒子的局部高度"，
;;       四点最小二乘拟合出水面高度与水线斜率 —— 于是"水在哪、往哪边斜"都是从水里读出来的；
;;     · 浮力：把鸭身切成 5 条竖条求排水面积 A 与浮心；A 在静水吃水处解析归一化，
;;       故 A = aeq 时浮力恰与重力平衡；浮力**沿水面法向**作用在浮心上，
;;       所以"向上"随水与重力走（滚筒转到哪、水堆到哪，鸭子就朝那边立），不是世界向上；
;;       吃水不对称时浮心偏移自然产生回正力矩（真实稳性的来源）；
;;     · 惯性：质量 M、转动惯量 I 均为粒子质量单位的数十倍，只受浮力/重力与水的黏性阻力，
;;       所以它是"压在水面上随水漂"的重物（速度与水流同级），而不是被单颗粒子推得乱晃的轻壳子；
;;     · 另有一个弱角弹簧把鸭背拉向当地向上（即使整只被水淹没也有确定的姿态收敛方向）；
;;     · 与水的双向作用：鸭身椭圆内的粒子被推到边缘，冲量按 1/M 折减后回传鸭身。
;;
;; 其它稳定性经验：
;;   · 每步重力不宜超过约 0.015（格/步²），否则压力求解跟不上、流体沸腾；
;;   · 压力迭代 ≥12 轮；
;;   · 贴壁面速度分量必须按"本工程 y 轴向下"归零（flip.c 是 y 向上，照抄 j-1 会漏掉
;;     贴地那一面，压力堵不住流入 → 流体被顶到天花板）；
;;   · 双线性权重必须先把采样点夹到 [1.5, GW-1.5]，否则贴壁粒子得到负权重=注入能量。
;;
;; 按《灵机用户参考手册》编写：算术固定二参数、`/` 与 `mod` 恒浮点（下标一律
;; Math.floor）、不用 let/case/do/for/return 与 list/append/map、define 无简写、
;; cond 未用、and/or 仅用于条件、Math.PI 需调用、(cons 0 '()) 先 pop 哑元素。
;; ============================================================================

(native Gfx)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认
(native Math)
(native String)
(native System)
(native Touch)      ; 触屏（左下角按钮）
(native Imu)        ; 姿态/重力感应
(native Diag)

;; ---- 画布与网格 ------------------------------------------------------------

(define W (Gfx.width))            ; 320
(define H (Gfx.height))           ; 240
(define HUD_H 16)
(define VH (- H HUD_H))           ; 224
(define CELL 12)                  ; 每格像素
(define GW 26)                    ; 网格列数
(define GH 18)                    ; 网格行数
(define NC (* GW GH))
(define TICK_MS 40)
(define PHYS_EVERY 1)             ; 每 N 拍推进一次物理（渲染每拍都做）

;; 物理参数（单位：格 / 步）
(define G_RATE 0.010)             ; 重力大小（格/步²）
(define FLIP_RATIO 0.5)          ; FLIP 比例（1=纯 FLIP，0=纯 PIC）
(define PRESS_ITERS 12)            ; 压力迭代次数（Gauss-Seidel + 超松弛）
(define DRIFT_K 0.0)              ; drift 补偿强度（0=关闭）
(define VEL_DAMP 0.94)           ; 粒子速度阻尼
(define SEP_K 0.05)                ; 按密度梯度分离粒子的强度
(define OVER_RELAX 1.8)           ; 超松弛系数（>1 收敛更快）
(define RAD 0.51)                ; 粒子半径（格）：间距 2R，粒子数 ≈ 面积/dx²
(define PSZ 11)                   ; 粒子在屏幕上的方块边长（px，略小于间距 2R·CELL≈12.2）
(define VMAX 0.60)                ; 速度上限（格/步），保证 CFL
(define DRUM_CX (/ GW 2.0))       ; 滚筒中心（= 画面中心）
(define DRUM_CY (/ GH 2.0))

;; ---- 网格场（扁平表，索引 i*GH + j）---------------------------------------

(define u (cons 0 (quote ())))    ; x 方向速度
(define v (cons 0 (quote ())))    ; y 方向速度
(define pu (cons 0 (quote ())))   ; 上一拍网格速度（FLIP 用）
(define pv (cons 0 (quote ())))
(define du (cons 0 (quote ())))   ; 权重和
(define dv (cons 0 (quote ())))
(define s (cons 0 (quote ())))    ; 1=可流动 0=固体壁
(define ct (cons 0 (quote ())))   ; 0=流体格 1=空气格
(define fl (cons 0 (quote ())))   ; 本拍流体格索引表
(define dens (cons 0 (quote ()))) ; 每格粒子密度（drift 补偿用）
(define pcount (cons 0 (quote ())))   ; 临时：每格粒子数
(define pstart (cons 0 (quote ())))   ; 临时：每格起始下标（前缀和）
(define pids (cons 0 (quote ())))     ; 临时：按格分组的粒子 id
(define rho0 0.0)                 ; 静止密度（首次自动标定）

;; ---- 粒子 ------------------------------------------------------------------

(define px (cons 0 (quote ())))
(define py (cons 0 (quote ())))
(define vx (cons 0 (quote ())))
(define vy (cons 0 (quote ())))
(define np 0)

;; ---- 状态 ------------------------------------------------------------------

(define drum_ang 0.0)              ; 滚筒当前角度（弧度）
(define drum_w 0.02)              ; 当前角速度（rad/步）
(define drum_s 0.0)               ; 0=洗涤 1=脱水
(define drum_t 0.0)               ; 循环相位
(define gm G_RATE)                ; 当前表观重力大小
;; ---- 重力源：自动（洗衣机滚筒）/ 重力感应（IMU），由左下角按钮切换 ------------
(define gmode 0)                  ; 0=自动（滚筒） 1=重力感应（IMU）
(define imu_ok 0)                 ; 重力感应模式下 IMU 是否真的可用（决定按钮配色）
(define imu_gx 0.0)               ; IMU 给出的屏幕系重力（1G = 9.8）
(define imu_gy 9.8)
(define imu_g 9.8)
(define TOG_X 3) (define TOG_Y 202) (define TOG_W 64) (define TOG_H 18)   ; 切换按钮（画布左下角）
(define cent_k 0.0)               ; 当前离心系数（由 turn_gravity 设置）
;; 滚筒循环：0~500 洗涤慢转 → 500~600 加速 → 600~800 脱水快转 → 800~900 减速
(define CYCLE_T 900.0)
(define WASH_T 500.0)
(define RAMP1_T 600.0)
(define SPIN_T 800.0)
(define W_WASH 0.030)             ; 洗涤角速度（rad/步，约 6 s 一圈）
(define W_SPIN 0.170)             ; 脱水角速度（rad/步，约 1.1 s 一圈）
(define CENT_K 0.0016)            ; 离心加速度系数（×距离）
(define gx 0.0)
(define gy 0.0)
(define frame_ms 0)
(define tick_no 0)
(define F (cons 0 (quote ())))    ; 临时：指向 u/v 的别名（当前场）
(define G (cons 0 (quote ())))    ; 临时：指向 pu/pv 的别名（上一拍场）
(define D (cons 0 (quote ())))    ; 临时：指向 du/dv 的别名（权重和）

(define clear_list
  (lambda (b) (while (> (length b) 0) { (pop b) })))

;; ---- 初始化 ----------------------------------------------------------------

(define build_grid
  (lambda ()
    (define i 0) (define j 0) (define c 0)
    (set! k 0)
    (set! i 0)
    (while (< i GW) {
      (set! j 0)
      (while (< j GH) {
        (set! c (+ (* i GH) j))
        (if (or (== i 0) (== j 0) (== i (- GW 1)) (== j (- GH 1)))
            { (push s 0.0) }
            { (push s 1.0) })
        (push u 0.0) (push v 0.0) (push pu 0.0) (push pv 0.0)
        (push du 0.0) (push dv 0.0) (push ct 1) (push dens 0.0) (push pcount 0) (push pstart 0) (push pw 0)
        (set! j (+ j 1)) })
      (set! i (+ i 1)) })))

;; 在容器下部铺一块水体（交错排列，接近六方密堆）
(define fill_block
  (lambda (yw ytop xw xleft)
    (define x 0.0) (define y 0.0) (define dx 0.0) (define dy 0.0) (define row 0)
    (set! dx (* 2.0 RAD))
    (set! dy (* 0.866 (* 2.0 RAD)))
    (set! row 0)
    (set! y (+ ytop dy))
    (while (< y yw) {
      (set! x (+ xleft RAD))
      (if (== (mod row 2) 1) { (set! x (+ x RAD)) })
      (while (< x xw) {
        (push px x) (push py y) (push vx 0.0) (push vy 0.0)
        (set! x (+ x dx)) })
      (set! y (+ y dy))
      (set! row (+ row 1)) })))

(define init_fluid
  (lambda ()
    (define i 0)
    (clear_list px) (clear_list py) (clear_list vx) (clear_list vy)
    ;; 水块：宽度 1..GW-1，高度占下方 45%
    ;; 水块：底部约 1/3 高度（粒子数约为最初的三分之一，换来流畅帧率）
    (fill_block (- GH 1.4) 9.6 (- GW 1.4) 1.4)
    (set! np (length px))
    ;; pids 需要与粒子数等长的容量（分离步骤里按格分组存放粒子 id）
    (while (> (length pids) 0) { (pop pids) })
    (while (< (length pids) np) { (push pids 0) })
    (set! gang 0.0)
    (set! gx 0.0)
    (set! gy G_RATE)
    (set! tick_no 0)))

;; ---- ① 积分与碰壁 ----------------------------------------------------------

(define integrate
  (lambda ()
    (define i 0) (define x 0.0) (define y 0.0) (define ax 0.0) (define ay 0.0)
    (define lo 0.0) (define hi_x 0.0) (define hi_y 0.0)
    (set! lo (+ 1.0 RAD))
    (set! hi_x (- (+ GW 0.0) lo))
    (set! hi_y (- (+ GH 0.0) lo))
    (set! i 0)
    (while (< i np) {
      (set! ax (+ (get_item vx i) gx))
      (set! ay (+ (get_item vy i) gy))
      ;; 滚筒离心项：指向筒外（脱水时 cent_k>0）
      (if (> cent_k 0.0)
          {
            (set! ax (+ ax (* cent_k (- (get_item px i) DRUM_CX))))
            (set! ay (+ ay (* cent_k (- (get_item py i) DRUM_CY)))) }
          {})
      ;; 速度上限，防止 CFL 崩掉
      (if (> ax VMAX) { (set! ax VMAX) })
      (if (< ax (- 0.0 VMAX)) { (set! ax (- 0.0 VMAX)) })
      (if (> ay VMAX) { (set! ay VMAX) })
      (if (< ay (- 0.0 VMAX)) { (set! ay (- 0.0 VMAX)) })
      (set! x (+ (get_item px i) ax))
      (set! y (+ (get_item py i) ay))
      ;; 四壁：钳位 + 速度反向衰减（简单碰撞）
      (if (< x lo) { (set! x lo) (set! ax (* -0.15 ax)) })
      (if (> x hi_x) { (set! x hi_x) (set! ax (* -0.15 ax)) })
      (if (< y lo) { (set! y lo) (set! ay (* -0.15 ay)) })
      (if (> y hi_y) { (set! y hi_y) (set! ay (* -0.15 ay)) })
      (set_item! vx i ax)
      (set_item! vy i ay)
      (set_item! px i x)
      (set_item! py i y)
      (set! i (+ i 1)) })))

;; ---- ② P2G：粒子速度 → 网格 -------------------------------------------------

(define p2g
  (lambda ()
    (define i 0) (define k 0) (define c 0)
    (define x 0.0) (define y 0.0) (define ux 0.0) (define uy 0.0)
    (define x0 0) (define y0 0) (define x1 0) (define y1 0)
    (define tx 0.0) (define ty 0.0) (define wx 0.0) (define wy 0.0)
    (define d0 0.0) (define d1 0.0) (define d2 0.0) (define d3 0.0)
    (define n0 0) (define n1 0) (define n2 0) (define n3 0)
    (define pvx 0.0) (define pvy 0.0) (define el 0.0) (define j 0)
    ;; 清空网格 + 重建流体格表
    (set! k 0)
    (while (< k NC) {
      (set_item! u k 0.0) (set_item! v k 0.0)
      (set_item! du k 0.0) (set_item! dv k 0.0)
      (set_item! ct k 1)
      (set! k (+ k 1)) })
    (clear_list fl)
    (set! i 0)
    (while (< i np) {
      (set! x (get_item px i))
      (set! y (get_item py i))
      (set! pvx (get_item vx i))
      (set! pvy (get_item vy i))
      ;; ---- 标记流体格（用整格坐标）----
      (set! x0 (Math.floor x))
      (set! y0 (Math.floor y))
      (if (< x0 0) { (set! x0 0) })
      (if (> x0 (- GW 1)) { (set! x0 (- GW 1)) })
      (if (< y0 0) { (set! y0 0) })
      (if (> y0 (- GH 1)) { (set! y0 (- GH 1)) })
      ;; 只把**内部格**放进流体格表：粒子的位置始终被夹在 [1+RAD, GW−1−RAD]×[1+RAD, GH−1−RAD]，
      ;; 正常碰不到边界格；这里显式过滤既省掉压力求解里"每格每轮算一次行号"的本地库调用，
      ;; 也顺带兜住万一被挤到边界外的粒子（那里的 s=0，本来也不参与投影）。
      (if (and (and (> x0 0) (< x0 (- GW 1))) (and (> y0 0) (< y0 (- GH 1))))
          {
            (set! c (+ (* x0 GH) y0))
            (if (== (get_item ct c) 1) { (set_item! ct c 0) (push fl c) } {}) }
          {})
      ;; ---- u 分量：采样点偏下半个格 ----
      (set! ux x)
      (set! uy (- y 0.5))
      (if (< ux 1.5) { (set! ux 1.5) })
      (if (> ux (- (+ GW 0.0) 1.5)) { (set! ux (- (+ GW 0.0) 1.5)) })
      (if (< uy 1.5) { (set! uy 1.5) })
      (if (> uy (- (+ GH 0.0) 1.5)) { (set! uy (- (+ GH 0.0) 1.5)) })
      (set! x0 (Math.floor ux))
      (set! y0 (Math.floor uy))
      (set! x1 (+ x0 1))
      (set! y1 (+ y0 1))
      (set! tx (- ux (+ x0 0.0)))
      (set! ty (- uy (+ y0 0.0)))
      (set! wx (- 1.0 tx))
      (set! wy (- 1.0 ty))
      (set! d0 (* wx wy)) (set! d1 (* tx wy)) (set! d2 (* tx ty)) (set! d3 (* wx ty))
      (set! n0 (+ (* x0 GH) y0))
      (set! n1 (+ (* x1 GH) y0))
      (set! n2 (+ (* x1 GH) y1))
      (set! n3 (+ (* x0 GH) y1))
      (set_item! u n0 (+ (get_item u n0) (* pvx d0)))
      (set_item! u n1 (+ (get_item u n1) (* pvx d1)))
      (set_item! u n2 (+ (get_item u n2) (* pvx d2)))
      (set_item! u n3 (+ (get_item u n3) (* pvx d3)))
      (set_item! du n0 (+ (get_item du n0) d0))
      (set_item! du n1 (+ (get_item du n1) d1))
      (set_item! du n2 (+ (get_item du n2) d2))
      (set_item! du n3 (+ (get_item du n3) d3))
      ;; ---- v 分量：采样点偏右半个格 ----
      (set! ux (- x 0.5))
      (set! uy y)
      (if (< ux 1.5) { (set! ux 1.5) })
      (if (> ux (- (+ GW 0.0) 1.5)) { (set! ux (- (+ GW 0.0) 1.5)) })
      (if (< uy 1.5) { (set! uy 1.5) })
      (if (> uy (- (+ GH 0.0) 1.5)) { (set! uy (- (+ GH 0.0) 1.5)) })
      (set! x0 (Math.floor ux))
      (set! y0 (Math.floor uy))
      (set! x1 (+ x0 1))
      (set! y1 (+ y0 1))
      (set! tx (- ux (+ x0 0.0)))
      (set! ty (- uy (+ y0 0.0)))
      (set! wx (- 1.0 tx))
      (set! wy (- 1.0 ty))
      (set! d0 (* wx wy)) (set! d1 (* tx wy)) (set! d2 (* tx ty)) (set! d3 (* wx ty))
      (set! n0 (+ (* x0 GH) y0))
      (set! n1 (+ (* x1 GH) y0))
      (set! n2 (+ (* x1 GH) y1))
      (set! n3 (+ (* x0 GH) y1))
      (set_item! v n0 (+ (get_item v n0) (* pvy d0)))
      (set_item! v n1 (+ (get_item v n1) (* pvy d1)))
      (set_item! v n2 (+ (get_item v n2) (* pvy d2)))
      (set_item! v n3 (+ (get_item v n3) (* pvy d3)))
      (set_item! dv n0 (+ (get_item dv n0) d0))
      (set_item! dv n1 (+ (get_item dv n1) d1))
      (set_item! dv n2 (+ (get_item dv n2) d2))
      (set_item! dv n3 (+ (get_item dv n3) d3))
      (set! i (+ i 1)) })
    ;; ---- 粒子密度（drift 补偿）：把每个粒子的双线性权重累加进 dens ----
    (set! k 0)
    (while (< k NC) { (set_item! dens k 0.0) (set! k (+ k 1)) })
    (set! i 0)
    (while (< i np) {
      (set! ux (get_item px i))
      (set! uy (get_item py i))
      (if (< ux 1.5) { (set! ux 1.5) })
      (if (> ux (- (+ GW 0.0) 1.5)) { (set! ux (- (+ GW 0.0) 1.5)) })
      (if (< uy 1.5) { (set! uy 1.5) })
      (if (> uy (- (+ GH 0.0) 1.5)) { (set! uy (- (+ GH 0.0) 1.5)) })
      (set! x0 (Math.floor ux))
      (set! y0 (Math.floor uy))
      (set! x1 (+ x0 1))
      (set! y1 (+ y0 1))
      (set! tx (- ux (+ x0 0.0)))
      (set! ty (- uy (+ y0 0.0)))
      (set! wx (- 1.0 tx))
      (set! wy (- 1.0 ty))
      (set! n0 (+ (* x0 GH) y0))
      (set! n1 (+ (* x1 GH) y0))
      (set! n2 (+ (* x1 GH) y1))
      (set! n3 (+ (* x0 GH) y1))
      (set_item! dens n0 (+ (get_item dens n0) (* wx wy)))
      (set_item! dens n1 (+ (get_item dens n1) (* tx wy)))
      (set_item! dens n2 (+ (get_item dens n2) (* tx ty)))
      (set_item! dens n3 (+ (get_item dens n3) (* wx ty)))
      (set! i (+ i 1)) })
    ;; 首次运行：用"流体格平均密度"标定静止密度
    (if (== rho0 0.0)
        {
          (set! n0 0) (set! el 0.0) (set! k 0)
          (while (< k (length fl)) {
            (set! c (get_item fl k))
            (set! el (+ el (get_item dens c)))
            (set! n0 (+ n0 1))
            (set! k (+ k 1)) })
          (if (> n0 0) { (set! rho0 (/ el (+ n0 0.0))) }) }
        {})
    ;; 归一化
    (set! k 0)
    (while (< k NC) {
      (if (> (get_item du k) 0.0) { (set_item! u k (/ (get_item u k) (get_item du k))) } {})
      (if (> (get_item dv k) 0.0) { (set_item! v k (/ (get_item v k) (get_item dv k))) } {})
      (set! k (+ k 1)) })
    ;; **贴壁面速度归零**（照 flip.c 的做法）：固体格及其相邻的 u/v 分量必须为零，
    ;; 否则粒子在墙上堆积出来的大速度会当成"流入"，压力求解永远堵不住，
    ;; 流体就会被顶到天花板上（之前"沸腾"的根因）。
    (set! i 0)
    (while (< i GW) {
      (set! j 0)
      (while (< j GH) {
        (set! c (+ (* i GH) j))
        (if (== (get_item s c) 0.0)
            { (set_item! u c 0.0) (set_item! v c 0.0) }
            {
              (if (> i 0) { (if (== (get_item s (- c GH)) 0.0) { (set_item! u c 0.0) }) })
              ;; 注意：本工程 y 轴向下，j+1 才是"下方"。flip.c 是 y 向上，
              ;; 照抄 j-1 会把贴地那一面的速度漏掉，压力求解就会一直把流体往上顶。
              (if (< j (- GH 1)) { (if (== (get_item s (+ c 1)) 0.0) { (set_item! v c 0.0) }) }) })
        (set! j (+ j 1)) })
      (set! i (+ i 1)) })))

;; ---- ③ 压力求解（原地 Gauss-Seidel）----------------------------------------

(define solve_pressure
  (lambda ()
    (define it 0) (define m 0) (define nf 0) (define c 0)
    (define s0 0.0) (define s1 0.0) (define s2 0.0) (define s3 0.0) (define ss 0.0)
    (define dv0 0.0) (define pv0 0.0) (define el 0.0)
    (set! nf (length fl))
    (set! it 0)
    ;; fl 里只有内部格（见 p2g），因此 c±GH、c±1 一律合法，无需再判行号
    (while (< it PRESS_ITERS) {
      (set! m 0)
      (while (< m nf) {
        (set! c (get_item fl m))
        (set! s0 (get_item s (- c GH)))
        (set! s1 (get_item s (+ c GH)))
        (set! s2 (get_item s (- c 1)))
        (set! s3 (get_item s (+ c 1)))
        (set! ss (+ s0 (+ s1 (+ s2 s3))))
        (if (> ss 0.0)
            {
              (set! dv0 (+ (- (get_item u (+ c GH)) (get_item u c))
                           (- (get_item v (+ c 1)) (get_item v c))))
              ;; drift 补偿：某格密度超过静止密度时，额外"鼓"一点散度
              (set! el (- (get_item dens c) rho0))
              (if (> el 0.0) { (set! dv0 (- dv0 el)) })
              (set! pv0 (/ (* (* -1.0 dv0) OVER_RELAX) ss))
              (set_item! u c (- (get_item u c) (* s0 pv0)))
              (set_item! u (+ c GH) (+ (get_item u (+ c GH)) (* s1 pv0)))
              (set_item! v c (- (get_item v c) (* s2 pv0)))
              (set_item! v (+ c 1) (+ (get_item v (+ c 1)) (* s3 pv0))) }
            {})
        (set! m (+ m 1)) })
      (set! it (+ it 1)) })))

;; ---- ④ G2P：网格 → 粒子（FLIP/PIC 混合）------------------------------------

(define save_prev
  (lambda ()
    (define k 0)
    (set! k 0)
    (while (< k NC) {
      (set_item! pu k (get_item u k))
      (set_item! pv k (get_item v k))
      (set! k (+ k 1)) })))

(define g2p
  (lambda ()
    (define i 0)
    (define x 0.0) (define y 0.0) (define ux 0.0) (define uy 0.0) (define el 0.0)
    (define x0 0) (define y0 0) (define x1 0) (define y1 0)
    (define tx 0.0) (define ty 0.0) (define wx 0.0) (define wy 0.0)
    (define d0 0.0) (define d1 0.0) (define d2 0.0) (define d3 0.0)
    (define n0 0) (define n1 0) (define n2 0) (define n3 0)
    (define pic 0.0) (define corr 0.0) (define old 0.0)
    (set! i 0)
    (while (< i np) {
      (set! x (get_item px i))
      (set! y (get_item py i))
      ;; ---------- u 分量 ----------
      (set! ux x)
      (set! uy (- y 0.5))
      (if (< ux 1.5) { (set! ux 1.5) })
      (if (> ux (- (+ GW 0.0) 1.5)) { (set! ux (- (+ GW 0.0) 1.5)) })
      (if (< uy 1.5) { (set! uy 1.5) })
      (if (> uy (- (+ GH 0.0) 1.5)) { (set! uy (- (+ GH 0.0) 1.5)) })
      (set! x0 (Math.floor ux))
      (set! y0 (Math.floor uy))
      (set! x1 (+ x0 1))
      (set! y1 (+ y0 1))
      (set! tx (- ux (+ x0 0.0)))
      (set! ty (- uy (+ y0 0.0)))
      (set! wx (- 1.0 tx))
      (set! wy (- 1.0 ty))
      (set! d0 (* wx wy)) (set! d1 (* tx wy)) (set! d2 (* tx ty)) (set! d3 (* wx ty))
      (set! n0 (+ (* x0 GH) y0))
      (set! n1 (+ (* x1 GH) y0))
      (set! n2 (+ (* x1 GH) y1))
      (set! n3 (+ (* x0 GH) y1))
      (set! pic (+ (* d0 (get_item u n0)) (+ (* d1 (get_item u n1))
                    (+ (* d2 (get_item u n2)) (* d3 (get_item u n3))))))
      (set! corr (+ (* d0 (- (get_item u n0) (get_item pu n0)))
                    (+ (* d1 (- (get_item u n1) (get_item pu n1)))
                       (+ (* d2 (- (get_item u n2) (get_item pu n2)))
                          (* d3 (- (get_item u n3) (get_item pu n3)))))))
      (set! el (+ d0 (+ d1 (+ d2 d3))))
      (if (> el 0.000001) { (set! pic (/ pic el)) (set! corr (/ corr el)) } {})
      (set! old (get_item vx i))
      (set_item! vx i (+ (* (- 1.0 FLIP_RATIO) pic) (* FLIP_RATIO (+ old corr))))
      ;; ---------- v 分量 ----------
      (set! ux (- x 0.5))
      (set! uy y)
      (if (< ux 1.5) { (set! ux 1.5) })
      (if (> ux (- (+ GW 0.0) 1.5)) { (set! ux (- (+ GW 0.0) 1.5)) })
      (if (< uy 1.5) { (set! uy 1.5) })
      (if (> uy (- (+ GH 0.0) 1.5)) { (set! uy (- (+ GH 0.0) 1.5)) })
      (set! x0 (Math.floor ux))
      (set! y0 (Math.floor uy))
      (set! x1 (+ x0 1))
      (set! y1 (+ y0 1))
      (set! tx (- ux (+ x0 0.0)))
      (set! ty (- uy (+ y0 0.0)))
      (set! wx (- 1.0 tx))
      (set! wy (- 1.0 ty))
      (set! d0 (* wx wy)) (set! d1 (* tx wy)) (set! d2 (* tx ty)) (set! d3 (* wx ty))
      (set! n0 (+ (* x0 GH) y0))
      (set! n1 (+ (* x1 GH) y0))
      (set! n2 (+ (* x1 GH) y1))
      (set! n3 (+ (* x0 GH) y1))
      (set! pic (+ (* d0 (get_item v n0)) (+ (* d1 (get_item v n1))
                    (+ (* d2 (get_item v n2)) (* d3 (get_item v n3))))))
      (set! corr (+ (* d0 (- (get_item v n0) (get_item pv n0)))
                    (+ (* d1 (- (get_item v n1) (get_item pv n1)))
                       (+ (* d2 (- (get_item v n2) (get_item pv n2)))
                          (* d3 (- (get_item v n3) (get_item pv n3)))))))
      (set! el (+ d0 (+ d1 (+ d2 d3))))
      (if (> el 0.000001) { (set! pic (/ pic el)) (set! corr (/ corr el)) } {})
      (set! old (get_item vy i))
      (set_item! vy i (+ (* (- 1.0 FLIP_RATIO) pic) (* FLIP_RATIO (+ old corr))))
      ;; 速度阻尼：抑制 FLIP 噪声自激（水看起来仍很活）
      (set_item! vx i (* VEL_DAMP (get_item vx i)))
      (set_item! vy i (* VEL_DAMP (get_item vy i)))
      (set! i (+ i 1)) })))

;; ---- 粒子分离（照 flip.c 的 push_particles_apart：桶排序 + 3×3 邻居）---------
;; 这是"稀疏粒子也能保持成一团水"的关键。没有它，流体在持续晃动下会慢慢扩散成雾气。
(define pw (cons 0 (quote ())))       ; 桶写指针

(define cell_of
  (lambda (i)
    (define cx 0) (define cy 0)
    (set! cx (Math.floor (get_item px i)))
    (set! cy (Math.floor (get_item py i)))
    (if (< cx 0) { (set! cx 0) })
    (if (> cx (- GW 1)) { (set! cx (- GW 1)) })
    (if (< cy 0) { (set! cy 0) })
    (if (> cy (- GH 1)) { (set! cy (- GH 1)) })
    (+ (* cx GH) cy)))

(define sep_pass
  (lambda ()
    (define i 0) (define j 0) (define k 0) (define c 0) (define cc 0)
    (define cx 0) (define cy 0) (define bx -1) (define by -1) (define ox 0) (define oy 0)
    (define acc 0) (define n 0) (define na 0) (define nb2 0)
    (define dx 0.0) (define dy 0.0) (define d2 0.0) (define dist 0.0)
    (define mv 0.0) (define lim 0.0) (define mvx 0.0) (define mvy 0.0)
    (set! lim (* 2.0 RAD))
    ;; 清零计数与写指针
    (set! k 0)
    (while (< k NC) {
      (set_item! pcount k 0)
      (set_item! pw k 0)
      (set! k (+ k 1)) })
    ;; 计数
    (set! i 0)
    (while (< i np) {
      (set! c (cell_of i))
      (set_item! pcount c (+ (get_item pcount c) 1))
      (set! i (+ i 1)) })
    ;; 前缀和
    (set! acc 0)
    (set! k 0)
    (while (< k NC) {
      (set_item! pstart k acc)
      (set! acc (+ acc (get_item pcount k)))
      (set! k (+ k 1)) })
    ;; 填充 pids
    (set! i 0)
    (while (< i np) {
      (set! c (cell_of i))
      (set! n (get_item pw c))
      (set_item! pids (+ (get_item pstart c) n) i)
      (set_item! pw c (+ n 1))
      (set! i (+ i 1)) })
    ;; 3×3 邻域，成对处理（只处理 j > i，避免重复也避免自我）
    (set! i 0)
    (while (< i np) {
      (set! cx (Math.floor (get_item px i)))
      (set! cy (Math.floor (get_item py i)))
      (set! by -1)
      (while (<= by 1) {
        (set! bx -1)
        (while (<= bx 1) {
          (set! ox (+ cx bx))
          (set! oy (+ cy by))
          (if (>= ox 0)
              { (if (< ox GW)
                    { (if (>= oy 0)
                          { (if (< oy GH)
                                {
                                  (set! cc (+ (* ox GH) oy))
                                  (set! na (get_item pstart cc))
                                  (set! nb2 (+ na (get_item pcount cc)))
                                  (set! k na)
                                  (while (< k nb2) {
                                    (set! j (get_item pids k))
                                    (if (> j i)
                                        {
                                          (set! dx (- (get_item px j) (get_item px i)))
                                          (set! dy (- (get_item py j) (get_item py i)))
                                          (set! d2 (+ (* dx dx) (* dy dy)))
                                          (if (< d2 (* lim lim))
                                              {
                                                (set! dist (Math.sqrt d2))
                                                (if (< dist 0.0001) { (set! dist 0.0001) })
                                                (set! mv (/ (- lim dist) dist))
                                                (set! mvx (* 0.5 (* mv dx)))
                                                (set! mvy (* 0.5 (* mv dy)))
                                                (set_item! px i (- (get_item px i) mvx))
                                                (set_item! py i (- (get_item py i) mvy))
                                                (set_item! px j (+ (get_item px j) mvx))
                                                (set_item! py j (+ (get_item py j) mvy)) }
                                              {}) }
                                        {})
                                    (set! k (+ k 1)) }) }
                                {}) }
                          {}) }
                    {}) }
              {})
          (set! bx (+ bx 1)) })
        (set! by (+ by 1)) })
      (set! i (+ i 1)) })))

(define separate
  (lambda ()
    (sep_pass)
    (clamp_all)))

(define clamp_all
  (lambda ()
    (define i 0) (define lo 0.0) (define hx 0.0) (define hy 0.0)
    (set! lo (+ 1.0 RAD))
    (set! hx (- (+ GW 0.0) lo))
    (set! hy (- (+ GH 0.0) lo))
    (set! i 0)
    (while (< i np) {
      (if (< (get_item px i) lo) { (set_item! px i lo) })
      (if (> (get_item px i) hx) { (set_item! px i hx) })
      (if (< (get_item py i) lo) { (set_item! py i lo) })
      (if (> (get_item py i) hy) { (set_item! py i hy) })
      (set! i (+ i 1)) })))

;; ---- 重力方向缓慢随机游走 --------------------------------------------------

(define turn_gravity
  (lambda ()
    (if (== gmode 1)
        ;; ---- 重力感应：直接用 IMU 的屏幕系重力（1G = 9.8），按 G_RATE 换算成本工程的单位。
        ;;      Imu.gravity_* 是**屏幕像素坐标系**（x 右、y 下）下的分量，与 gx/gy 同向，
        ;;      因此取来即可用（右缘下沉 → gx>0 → 水往右堆、左边翘起）。
        {
          (set! imu_gx (Imu.gravity_x))
          (set! imu_gy (Imu.gravity_y))
          (set! imu_g
                 (Math.sqrt (+ (* imu_gx imu_gx) (* imu_gy imu_gy))))
          (if (< imu_g 0.001) { (set! imu_g 0.001) })
          (set! gm (* G_RATE (/ imu_g 9.8)))
          (set! gx (* G_RATE (/ imu_gx 9.8)))
          (set! gy (* G_RATE (/ imu_gy 9.8)))
          (set! cent_k 0.0)          ; 滚筒离心项在重力感应下没有意义
          (set! drum_s 0.0)
          (set! imu_ok (if (Imu.is_available) 1 0)) }
        ;; ---- 自动：洗衣机滚筒（重力方向连续旋转 360°，转速按"洗涤 → 脱水 → 洗涤"循环变化）
        {
          (set! drum_t (+ drum_t 1.0))
          (if (>= drum_t CYCLE_T) { (set! drum_t 0.0) })
          ;; s：0=洗涤（慢转），1=脱水（快转）
          (if (< drum_t WASH_T)
              { (set! drum_s 0.0) }
              {
                (if (< drum_t RAMP1_T)
                    { (set! drum_s (/ (- drum_t WASH_T) (- RAMP1_T WASH_T))) }
                    {
                      (if (< drum_t SPIN_T)
                          { (set! drum_s 1.0) }
                          { (set! drum_s (/ (- CYCLE_T drum_t) (- CYCLE_T SPIN_T))) }) }) })
          ;; 角速度与表观重力随 s 变化
          (set! drum_w (+ W_WASH (* (- W_SPIN W_WASH) drum_s)))
          (set! drum_ang (+ drum_ang drum_w))
          (if (> drum_ang 6.28318) { (set! drum_ang (- drum_ang 6.28318)) })
          (set! gm (* G_RATE (+ 1.0 (* 1.6 drum_s))))
          ;; 方向：滚筒转到哪，重力（在筒内的表观方向）就指向哪
          (set! gx (* gm (Math.sin drum_ang)))
          (set! gy (* gm (Math.cos drum_ang)))
          ;; 离心项：离筒心越远越往外推（脱水时明显），让水贴在筒壁上转
          (set! cent_k (* CENT_K (* drum_s drum_s))) })))
;; 小黄鸭位图（30×31，来源：openclipart.org/detail/325903/rubber-duck，CC0/公有领域）
;; 每行 30 个字符：'.'=透明  'y'=身体黄  'o'=橙  'd'=暗  'k'=眼睛  'w'=高光  'b'=深棕
(define DUCK_W 30)
(define DUCK_H 31)
(define DUCK_ART (quote (
  "..............oyyyyo.........."
  "............oyyyyyyyyo........"
  "...........oyyyyyyyyyyo......."
  "..........oyyyyyyyyyyyyo......"
  "..........yyyywbbyyyyyyy......"
  ".........oyyywbkkoyyooooooooo."
  ".........yyyywbkkoyooooooooooo"
  ".........yyyyyyboyoooooooooooo"
  ".........yyyyyyyyooooooooodoo."
  ".........oyyyyyyyooooooodddo.."
  ".........oyyyyyyyoooodddddo..."
  "..........yyyyyyyoooooodddoo.."
  "....oooo..oyyyyyyyoooooooooo.."
  "...oyyyyooyyyyyyyyyyooooooo..."
  "..oyyyyyyyyyyyyyyyyyyooo......"
  "..oyyyyyyyyyyyyyyyyyyyy......."
  ".oyyyyyyyyyyyyyyyyyyyyyyo....."
  ".oyyyyyyyyyyyyyyyyyyyyyyyo...."
  ".yyyyyyyyyyyyyyyyyyyyyyyyy...."
  "oyyyyyyyyyyyyyyyyyyyyyyyyyo..."
  "oyyyyyyyyyyyyyyyyyyyyyyyyyy..."
  "oyyyyyyyyyyyyyyyyyyyyyyyyyyo.."
  "yyyyyyyyyyyyyyyyyyyyyyyyyyyo.."
  "yyyyyyyyyyyyyyyyyyyyyyyyyyyo.."
  "oyyyyyyyyyyyyyyyyyyyyyyyyyyo.."
  "oyyyyyyyyyyyyyyyyyyyyyyyyyyo.."
  ".oyyyyyyyyyyyyyyyyyyyyyyyyyo.."
  "..yyyyyyyyyyyyyyyyyyyyyyyyy..."
  "...oyyyyyyyyyyyyyyyyyyyyyyo..."
  "....oyyyyyyyyyyyyyyyyyyyyo...."
  "......oyyyyyyyyyyyyyooo......."
)))

;; 调色板（'.' 为透明，占 0 位）
(define DUCK_PAL (quote ((0 0 0) (250 206 46) (214 140 26) (168 98 20) (36 34 38) (246 246 246) (120 84 44))))

;; 把字符位图展开成"调色板索引"扁平表（加载时做一次）
(define DUCK_PIX (cons 0 (quote ())))
(define duck_build
  (lambda ()
    (define r 0) (define cstr "") (define i 0) (define ch "")
    (define nrow 0) (define idx 0)
    (while (> (length DUCK_PIX) 0) { (pop DUCK_PIX) })
    (set! nrow (length DUCK_ART))
    (set! r 0)
    (while (< r nrow) {
      (set! cstr (get_item DUCK_ART (Math.floor r)))
      (set! i 0)
      (while (< i DUCK_W) {
        (set! ch (String.charAt cstr (Math.floor i)))
        (set! idx 0)
        (if (String.equals ch "y") { (set! idx 1) })
        (if (String.equals ch "o") { (set! idx 2) })
        (if (String.equals ch "d") { (set! idx 3) })
        (if (String.equals ch "k") { (set! idx 4) })
        (if (String.equals ch "w") { (set! idx 5) })
        (if (String.equals ch "b") { (set! idx 6) })
        (push DUCK_PIX idx)
        (set! i (+ i 1)) })
      (set! r (+ r 1)) })))

;; ---- 小黄鸭：浮在水面上的刚体 ----------------------------------------------
;; 物理 = 牛顿第二定律 + 阿基米德浮力，状态只有 位置/速度/倾角/角速度 四个量：
;;   ① 水面测量（duck_sweep，随粒子遍历一起做）：在鸭身左右各两条竖带（横向 1.2~4.2 格）
;;      里记下"沿当地向上 u 最高的两颗粒子"（取均值抗飞溅），四点最小二乘拟合出
;;      鸭心处的水面高度 dk_lv 与水线斜率 dk_sk。u = −有效重力（滚筒重力 + 离心项），
;;      水一定被压在 −u 一侧，所以"水面"就是在 u 上最高的那些粒子。
;;      **只在鸭身之外测**：鸭身所在处的水被鸭身挤空了，那里的"水面"是坑底。
;;   ② 鸭身按宽度切成 NS 条竖条；各条水线 = dk_lv 处、法向为水面法向的那条直线，
;;      吃水长度 s_k 由该条两端点的吃水深线性插值得到；累加得排水面积 A 与浮心（形心）。
;;      静水吃水 DUCK_DRAFT 处的排水面积 duck_aeq 在 duck_setup 里解析算出，
;;      于是 A = duck_aeq 时浮力恰好抵消重力（平衡点与增益无关，增益只管软硬）。
;;   ③ 浮力沿**水面法向**（由量到的 dk_sk 修正）作用在浮心上：
;;      水堆到哪边，鸭子就往反方向立 —— 这就是"相对于重力和水的向上"，而不是世界向上；
;;      吃水不再左右对称时，浮心偏移自然产生回正力矩（真实稳性的来源）。
;;   ④ 线性与角运动按 M / I 积分（都是粒子质量单位的数十倍），再叠加水阻：
;;      速度被拉向鸭身周围的水流速度、角速度按 DUCK_KAD 衰减 —— 于是它是"压在水面上
;;      随水漂"的重物，不会像轻壳子那样被每颗粒子的反作用推得乱晃（反作用按 1/M 折减）。
;;   ⑤ 另有一个弱角弹簧（DUCK_KA）把鸭背拉向当地向上：即使整只鸭被水淹没、水线不跨
;;      鸭身，姿态也有确定的收敛方向，不会僵在任意角度。
(define dkx 13.0) (define dky 11.0)
(define dkvx 0.0) (define dkvy 0.0)
(define dka 0.0) (define dkw 0.0)
(define DUCK_HW 1.15)             ; 半宽（格）
(define DUCK_HH 1.05)             ; 半高（格）
(define DUCK_DRAFT -0.55)         ; 平衡吃水（局部 y：中心低于水面多少格；负 = 中心浮在水面之上）
(define DUCK_M 60.0)              ; 质量（粒子质量单位；只影响与水的作用强度）
(define DUCK_I 36.0)              ; 转动惯量（均匀椭圆 M·(HW²+HH²)/4）
(define DUCK_BUOY 2.0)            ; 浮力刚度增益（1=纯阿基米德；>1 更"绷"，平衡点不变）
(define DUCK_TORQ 1.0)            ; 浮心力矩增益（水线不水平的自然回正量）
(define DUCK_KA 0.018)            ; 稳性角弹簧（每拍按水线斜率修正 1.8%）
(define DUCK_KAD 0.15)            ; 角阻尼（每步保留比例）
(define DUCK_DRAG 0.22)           ; 线性水阻（每步把速度拉向水流的速度比例）
(define DUCK_WMAX 0.25)           ; 角速度上限（rad/步）
(define DUCK_NS 5)                ; 鸭身竖条数量（算排水面积用）
(define DUCK_DW 0.46)             ; 每条宽度（duck_setup 里算）
(define DUCK_DLX (cons 0 (quote ())))   ; 每条中心局部 x
(define DUCK_DH (cons 0 (quote ())))    ; 每条半高（椭圆在该 x 处的半高）
(define MP 4)                     ; 水面采样点数量（鸭身左右各两个）
(define DUCK_MPX (cons 0 (quote ())))   ; 采样点的横向偏移（沿 ê，相对鸭心）
(define MP1 (cons 0 (quote ())))        ; 每个采样带里最高的粒子（沿 u 的投影）
(define MP2 (cons 0 (quote ())))        ; 次高的粒子（与 MP1 取均值抗飞溅）
(define DUCK_HWIN 3.0)            ; 量水线时沿 u 的搜索窗（只取鸭心上方 3 格内的粒子）
(define duck_aeq 1.0)             ; 平衡吃水下的排水面积（浮力归一化基准）
(define dk_lv 0.0)                ; 平滑后的"鸭心处水面高度"（沿 u，相对鸭心）
(define dk_sk 0.0)                ; 平滑后的水线斜率（d(水面高度)/d(横向)）
(define wv_n 0) (define wv_u 0.0) (define wv_v 0.0)   ; 附近水流速度临时累加器

(define imin (lambda (a b) (if (< a b) a b)))
(define imax (lambda (a b) (if (> a b) a b)))

;; 预计算竖条几何、静水排水面积、水面采样点位置
(define duck_setup
  (lambda ()
    (define k 0) (define lx 0.0) (define h 0.0) (define ae 0.0) (define s0 0.0)
    (while (> (length DUCK_DLX) 0) { (pop DUCK_DLX) })
    (while (> (length DUCK_DH) 0) { (pop DUCK_DH) })
    (while (> (length DUCK_MPX) 0) { (pop DUCK_MPX) })
    (while (> (length MP1) 0) { (pop MP1) })
    (while (> (length MP2) 0) { (pop MP2) })
    (set! DUCK_DW (/ (* 2.0 DUCK_HW) (+ DUCK_NS 0.0)))
    (set! ae 0.0)
    (set! k 0)
    (while (< k DUCK_NS) {
      (set! lx (* (* 2.0 DUCK_HW) (- (/ (+ k 0.5) (+ DUCK_NS 0.0)) 0.5)))
      (set! h (* DUCK_HH (Math.sqrt (- 1.0 (* (/ lx DUCK_HW) (/ lx DUCK_HW))))))
      (push DUCK_DLX lx)
      (push DUCK_DH h)
      ;; 平衡吃水时该条的吃水长度 = clamp(DRAFT + h, 0, 2h)
      (set! s0 (imax 0.0 (imin (* 2.0 h) (+ DUCK_DRAFT h))))
      (set! ae (+ ae (* DUCK_DW s0)))
      (set! k (+ k 1)) })
    (set! duck_aeq ae)
    ;; 水面采样点：鸭身左右各两条（内 ≈ 2 格、外 ≈ 3.3 格；避开鸭身挤出的水坑）
    (push DUCK_MPX (- 3.3 0.0)) (push DUCK_MPX -1.95)
    (push DUCK_MPX 1.95) (push DUCK_MPX 3.3)
    (set! k 0)
    (while (< k MP) {
      (push MP1 999.0)
      (push MP2 999.0)
      (set! k (+ k 1)) })))

(define duck_init
  (lambda ()
    (define i 0) (define m 0.0)
    (duck_setup)
    (set! dkx (/ GW 2.0))
    (set! dkvx 0.0) (set! dkvy 0.0) (set! dka 0.0) (set! dkw 0.0)
    (set! dk_lv 0.0) (set! dk_sk 0.0)
    ;; 初始位置：取鸭身附近粒子的最高点，让它一开始就正好坐在水面上
    ;; （否则第一秒会从深水里猛窜上来）。
    (set! m (+ GH 0.0))
    (set! i 0)
    (while (< i np) {
      (if (< (Math.abs (- (get_item px i) dkx)) (+ DUCK_HW 1.5))
          { (if (< (get_item py i) m) { (set! m (get_item py i)) }) }
          {})
      (set! i (+ i 1)) })
    (set! dky (+ m DUCK_DRAFT))))

;; 水阻取样：在 (x,y) 处累加流体格的网格速度（固体格与空气格跳过）
(define wv_add
  (lambda (x y)
    (define c 0)
    (set! c (+ (* (imax 0 (imin (- GW 1) (Math.floor x))) GH)
               (imax 0 (imin (- GH 1) (Math.floor y)))))
    (if (== (get_item ct c) 0)
        { (set! wv_u (+ wv_u (get_item u c)))
          (set! wv_v (+ wv_v (get_item v c)))
          (set! wv_n (+ wv_n 1)) }
        {})))

(define dk_wet 0)                 ; 上一拍有几个水面采样点有效
(define dk_ux 0.0) (define dk_uy -1.0)   ; 当地向上单位向量（世界）
(define dk_ex 1.0) (define dk_ey 0.0)    ; 当地横向单位向量（世界）
(define dk_draft 0.0)             ; 平滑后的中心吃水（沿 u，中心在水面下多少格）

;; ---- 坐标系约定 ------------------------------------------------------------
;; 局部 → 世界：(lx,ly) ↦ (dkx + ca·lx + sa·ly, dky − sa·lx + ca·ly)（与 duck_draw 一致）
;; 世界 → 局部：(rx,ry) ↦ (ca·rx − sa·ry, sa·rx + ca·ry)
;; 局部 +y 指向鸭腹（"下"），−y 指向鸭背（"上"）。
;; 另有当地重力系：u = −有效重力（向上单位向量），ê = u 逆时针转 90°（横向单位向量）。
;; 水面一定在 u 方向那一侧（水被重力压向 −u 一侧），因此"水面高度"要在 u 上取最大。

;; 一次粒子遍历：① 量鸭身两侧的水面（u 方向的最高点 + 横向坡度）；② 把落在鸭身椭圆内的
;; 粒子挤到边缘（并对鸭身施反作用）。
(define duck_sweep
  (lambda ()
    (define i 0) (define ca 0.0) (define sa 0.0)
    (define rx 0.0) (define ry 0.0) (define lx 0.0) (define ly 0.0)
    (define q 0.0) (define sc 0.0) (define wx 0.0) (define wy 0.0)
    (define k 0) (define al 0.0) (define w1 0.0) (define w2 0.0) (define rw 8.0)
    (define n 0) (define sx 0.0) (define xx 0.0) (define sw 0.0) (define xw 0.0)
    (define dn 0.0) (define bs 0.0) (define lv 0.0)
    (define ug 0.0) (define ux 0.0) (define uy 0.0) (define ex 0.0) (define ey 0.0)
    (define dp 0.0) (define de 0.0)
    ;; ① 当地有效重力 → u（向上）、ê（横向）
    (set! ug (+ gx (* cent_k (- dkx DRUM_CX))))
    (set! uy (+ gy (* cent_k (- dky DRUM_CY))))
    (set! dp (Math.sqrt (+ (* ug ug) (* uy uy))))
    (if (< dp 0.000001) { (set! dp 0.000001) })
    (set! ux (/ (- 0.0 ug) dp))
    (set! uy (/ (- 0.0 uy) dp))
    (set! ex (- 0.0 uy))
    (set! ey ux)
    (set! dk_ux ux) (set! dk_uy uy) (set! dk_ex ex) (set! dk_ey ey)
    (set! ca (Math.cos dka))
    (set! sa (Math.sin dka))
    (set! k 0)
    (while (< k MP) {
      (set_item! MP1 k -999.0)
      (set_item! MP2 k -999.0)
      (set! k (+ k 1)) })
    (set! rw (+ 4.2 DUCK_HWIN))
    (set! i 0)
    (while (< i np) {
      (set! rx (- (get_item px i) dkx))
      (set! ry (- (get_item py i) dky))
      ;; 粗筛（世界系方窗 ⊇ 局部窗），省掉远处的坐标变换
      (if (and (< (Math.abs rx) rw) (< (Math.abs ry) rw))
          {
            ;; ---- ① 水面采样：鸭身两侧（横向 1.2~4.2 格），沿 u 取最高 ----
            (set! de (+ (* rx ex) (* ry ey)))
            (if (and (> (Math.abs de) 1.2) (< (Math.abs de) 4.2))
                {
                  (set! dp (+ (* rx ux) (* ry uy)))
                  (if (< dp (+ DUCK_HWIN 3.0))
                      {
                        (if (< de 0.0) { (set! k 0) } { (set! k 2) })
                        (if (> (Math.abs de) 2.6) { (set! k (+ k 1)) })
                        (set! w1 (get_item MP1 k))
                        (set! w2 (get_item MP2 k))
                        (if (> dp w1)
                            { (set_item! MP2 k w1) (set_item! MP1 k dp) }
                            { (if (> dp w2) { (set_item! MP2 k dp) } {}) }) }
                      {}) }
                {})
            (set! lx (- (* ca rx) (* sa ry)))
            (set! ly (+ (* sa rx) (* ca ry)))
            ;; ---- ② 鸭身把水挤开（椭圆内 → 推到边界）----
            (set! q (+ (* (/ lx DUCK_HW) (/ lx DUCK_HW)) (* (/ ly DUCK_HH) (/ ly DUCK_HH))))
            (if (< q 1.0)
                {
                  (if (< q 0.0001) { (set! q 0.0001) })
                  (set! sc (/ 1.0 (Math.sqrt q)))
                  (set! wx (+ dkx (+ (* ca (* lx sc)) (* sa (* ly sc)))))
                  (set! wy (+ dky (- (* ca (* ly sc)) (* sa (* lx sc)))))
                  ;; 反作用：粒子质量为 1、鸭子质量为 M，冲量按 1/M 回传到鸭身
                  (set! dkvx (- dkvx (/ (- wx (get_item px i)) DUCK_M)))
                  (set! dkvy (- dkvy (/ (- wy (get_item py i)) DUCK_M)))
                  (set_item! px i wx)
                  (set_item! py i wy) }
                {}) }
          {})
      (set! i (+ i 1)) })
    ;; ② 两点以上有效 → 最小二乘拟合水面直线：level(de) = lv + bs·de
    (set! n 0) (set! sx 0.0) (set! xx 0.0) (set! sw 0.0) (set! xw 0.0)
    (set! k 0)
    (while (< k MP) {
      (set! w1 (get_item MP1 k))
      (set! w2 (get_item MP2 k))
      (set! al (get_item DUCK_MPX k))
      (if (> w1 -900.0)
          {
            (if (> w2 -900.0) { (set! dp (* 0.5 (+ w1 w2))) } { (set! dp w1) })
            (set! n (+ n 1))
            (set! sx (+ sx al))
            (set! xx (+ xx (* al al)))
            (set! sw (+ sw dp))
            (set! xw (+ xw (* al dp))) }
          {})
      (set! k (+ k 1)) })
    (set! dk_wet n)
    (if (>= n 1)
        {
          (set! dn (- (* (* n 1.0) xx) (* sx sx)))
          (if (> dn 0.0001)
              {
                (set! bs (/ (- (* (* n 1.0) xw) (* sx sw)) dn))
                (set! lv (/ (- sw (* bs sx)) (* n 1.0)))
                (if (> bs 1.2) { (set! bs 1.2) })
                (if (< bs -1.2) { (set! bs -1.2) })
                ;; 一阶低通：水面测量本身有飞溅噪声
                (set! dk_lv (+ dk_lv (* 0.35 (- lv dk_lv))))
                (set! dk_sk (+ dk_sk (* 0.35 (- bs dk_sk)))) }
              { (if (== n 1) { (set! dk_lv (+ dk_lv (* 0.35 (- lv dk_lv)))) } {}) }) }
        {})
    ;; dk_lv = 鸭心处水面相对鸭心的 u 分量：水面在鸭心之上（正）＝中心没入多深。
    ;; 这就是"中心吃水"，与 DUCK_DRAFT 同号约定（负 = 中心浮在水面之上）。
    (set! dk_draft dk_lv)))

(define duck_step
  (lambda ()
    (define i 0) (define lx 0.0) (define h 0.0) (define wl 0.0) (define d1 0.0) (define d2 0.0)
    (define base 0.0) (define q 0.0) (define pp 0.0) (define sl 0.0) (define cyk 0.0) (define ak 0.0)
    (define A 0.0) (define Mx 0.0) (define My 0.0)
    (define cbx 0.0) (define cby 0.0) (define ca 0.0) (define sa 0.0)
    (define gn 1.0) (define ww0 0.0) (define tau 0.0) (define gl 0.0)
    (define gex 0.0) (define gey 0.0) (define uw 0.0) (define vw 0.0)
    (define uxl 0.0) (define uyl 0.0) (define nv 0.0) (define nx 0.0) (define ny 0.0)
    (define bs 0.0) (define dd 0.0)
    ;; ① 当地有效重力（滚筒重力 + 离心项）
    (set! gex (+ gx (* cent_k (- dkx DRUM_CX))))
    (set! gey (+ gy (* cent_k (- dky DRUM_CY))))
    (set! gl (Math.sqrt (+ (* gex gex) (* gey gey))))
    (if (< gl 0.000001) { (set! gl 0.000001) })
    ;; ② 水阻取样（同时兜底判断"鸭子是否在水里"）
    (set! wv_n 0) (set! wv_u 0.0) (set! wv_v 0.0)
    (wv_add (- dkx (+ DUCK_HW 0.6)) dky)
    (wv_add (+ dkx (+ DUCK_HW 0.6)) dky)
    (wv_add dkx (+ dky (+ DUCK_HH 0.6)))
    (wv_add (- dkx DUCK_HW) (+ dky (+ DUCK_HH 0.6)))
    (wv_add (+ dkx DUCK_HW) (+ dky (+ DUCK_HH 0.6)))
    (set! uw 0.0) (set! vw 0.0)
    (if (> wv_n 0)
        { (set! uw (/ wv_u (+ wv_n 0.0)))
          (set! vw (/ wv_v (+ wv_n 0.0))) }
        {})
    ;; ③ 水面法向（世界）→ 鸭身局部；水面法向 = 重力向上再叠上量到的横向坡度
    (set! bs (* 0.5 dk_sk))
    (set! nx (- dk_ux (* bs dk_ex)))
    (set! ny (- dk_uy (* bs dk_ey)))
    (set! nv (Math.sqrt (+ (* nx nx) (* ny ny))))
    (if (< nv 0.000001) { (set! nv 0.000001) })
    (set! nx (/ nx nv))
    (set! ny (/ ny nv))
    (set! ca (Math.cos dka))
    (set! sa (Math.sin dka))
    (set! uxl (- (* ca nx) (* sa ny)))
    (set! uyl (+ (* sa nx) (* ca ny)))
    (set! dd dk_draft)
    ;; 采样全失败时兜底：鸭身周围到处都是水 → 当作整只没入（浮力取最大，把它顶上来）；
    ;; 周围是空的 → 当作完全离水（自由落体）。
    (if (< dk_wet 1)
        { (if (>= wv_n 3) { (set! dd 2.0) } { (set! dd 40.0) }) }
        {})
    ;; ④ 逐条：吃水长度 + 形心
    (set! A 0.0) (set! Mx 0.0) (set! My 0.0)
    (set! i 0)
    (while (< i DUCK_NS) {
      (set! lx (get_item DUCK_DLX i))
      (set! h (get_item DUCK_DH i))
      (set! base (- dd (* lx uxl)))
      (set! q (- 0.0 (* h uyl)))
      (set! d1 (- base q))              ; 局部 y = −h 端点的吃水深
      (set! d2 (+ base q))              ; 局部 y = +h 端点的吃水深
      (set! sl 0.0) (set! cyk 0.0)
      (if (>= d1 0.0)
          { (if (>= d2 0.0)
                { (set! sl (* 2.0 h)) (set! cyk 0.0) }
                { (set! pp (/ d1 (- d1 d2)))
                  (set! sl (* (* 2.0 h) pp))
                  (set! cyk (* h (- pp 1.0))) }) }
          { (if (<= d2 0.0)
                { (set! sl 0.0) (set! cyk 0.0) }
                { (set! pp (/ d1 (- d1 d2)))
                  (set! sl (* (* 2.0 h) (- 1.0 pp)))
                  (set! cyk (* h pp)) }) })
      (set! ak (* DUCK_DW sl))
      (set! A (+ A ak))
      (set! Mx (+ Mx (* ak lx)))
      (set! My (+ My (* ak cyk)))
      (set! i (+ i 1)) })
    ;; ⑤ 浮力（作用在浮心、沿水面法向）+ 重力
    (if (> A 0.0001)
        {
          (set! cbx (/ Mx A))
          (set! cby (/ My A))
          ;; 刚度增益：排水量偏离静水值时放大（A = duck_aeq 处 gn = 1，仍严格平衡）
          (set! gn (+ 1.0 (* DUCK_BUOY (- (/ A duck_aeq) 1.0))))
          (if (< gn -0.5) { (set! gn -0.5) })
          (if (> gn 2.5) { (set! gn 2.5) })
          ;; 力矩（局部）：τ = ly·F_x − lx·F_y，F = u_local·(M·|g|·gn)
          (set! ww0 (* (* DUCK_M gl) gn))
          (set! tau (* DUCK_TORQ (* ww0 (- (* cby uxl) (* cbx uyl)))))
          (set! dkw (+ dkw (/ tau DUCK_I)))
          ;; 稳性角弹簧：让鸭背朝当地向上（局部 −y 对齐 u_local）
          (set! dkw (+ dkw (* DUCK_KA (Math.atan2 (- 0.0 uxl) (- 0.0 uyl)))))
          ;; 世界加速度 = 有效重力 + 浮力（浮力 = 局部向上 → 世界）
          (set! dkvx (+ dkvx (+ gex (* (* gl gn) (+ (* ca uxl) (* sa uyl))))))
          (set! dkvy (+ dkvy (+ gey (* (* gl gn) (- (* ca uyl) (* sa uxl)))))) }
        { ;; 完全离水：只受重力（自由落体）
          (set! dkvx (+ dkvx gex))
          (set! dkvy (+ dkvy gey)) })
    ;; ⑥ 水阻：速度拉向鸭身周围的水流，角速度衰减
    (set! dkvx (+ dkvx (* DUCK_DRAG (- uw dkvx))))
    (set! dkvy (+ dkvy (* DUCK_DRAG (- vw dkvy))))
    (set! dkw (* (- 1.0 DUCK_KAD) dkw))
    ;; ⑦ 积分（角度回绕到 ±π：水面朝哪边，鸭子就朝哪边立起来）
    (set! dkx (+ dkx dkvx))
    (set! dky (+ dky dkvy))
    (set! dka (+ dka dkw))
    (if (> dka 3.14159) { (set! dka (- dka 6.28318)) })
    (if (< dka -3.14159) { (set! dka (+ dka 6.28318)) })
    (if (> dkw DUCK_WMAX) { (set! dkw DUCK_WMAX) })
    (if (< dkw (- 0.0 DUCK_WMAX)) { (set! dkw (- 0.0 DUCK_WMAX)) })
    ;; 容器四壁
    (if (< dkx (+ DUCK_HW 1.0)) { (set! dkx (+ DUCK_HW 1.0)) (set! dkvx (* -0.3 dkvx)) })
    (if (> dkx (- GW (+ DUCK_HW 1.0))) { (set! dkx (- GW (+ DUCK_HW 1.0))) (set! dkvx (* -0.3 dkvx)) })
    (if (< dky (+ DUCK_HH 1.0)) { (set! dky (+ DUCK_HH 1.0)) (set! dkvy (* -0.3 dkvy)) })
    (if (> dky (- GH (+ DUCK_HH 1.0))) { (set! dky (- GH (+ DUCK_HH 1.0))) (set! dkvy (* -0.3 dkvy)) })))

;; 绘制：**目的像素驱动**的旋转贴图 —— 遍历旋转后的包围盒，逐像素反向旋转求源坐标，
;; 命中不透明像素才落笔。正映射（源→目的）在斜角度下会有漏点，表现为鸭身内部的黑色
;; 点阵（混叠网格）；反映射对任意角度都是满射，因此不会再出现空洞。
;; 透明像素跳过；画在粒子之前 → 没入水中的部分被水盖住。
(define DUCK_HWD (* DUCK_W 0.5))   ; 源图中心（连续坐标）
(define DUCK_HHD (* DUCK_H 0.5))
(define DUCK_BOX 22.0)             ; 包围盒半边长 = ceil(√(30²+31²)/2)
(define duck_draw
  (lambda ()
    (define ca 0.0) (define sa 0.0)
    (define cx 0.0) (define cy 0.0)
    (define sx 0) (define sy 0) (define x0 0) (define y0 0) (define x1 0) (define y1 0)
    (define dy 0.0) (define lx 0.0) (define ly 0.0)
    (define ii 0.0) (define jj 0.0) (define idx 0.0) (define pal 0)
    ;; 行内游程：同一调色板索引连续命中时合并成一次 fill_rect
    (define rp 0) (define rs 0) (define rn 0)
    (set! ca (Math.cos dka))
    (set! sa (Math.sin dka))
    (set! cx (* dkx (+ CELL 0.0)))
    (set! cy (* dky (+ CELL 0.0)))
    (set! x0 (Math.floor (- cx DUCK_BOX)))
    (set! x1 (Math.floor (+ cx DUCK_BOX)))
    (set! y0 (Math.floor (- cy DUCK_BOX)))
    (set! y1 (Math.floor (+ cy DUCK_BOX)))
    (if (< x0 0) { (set! x0 0) })
    (if (> x1 (- W 1)) { (set! x1 (- W 1)) })
    (if (< y0 0) { (set! y0 0) })
    (if (> y1 (- VH 1)) { (set! y1 (- VH 1)) })
    (set! sy y0)
    (while (<= sy y1) {
      ;; 行首：目的像素中心 (x0+0.5, sy+0.5) 相对鸭心的偏移，反向旋转到源图连续坐标
      (set! dy (- (+ sy 0.5) cy))
      (set! lx (- (* ca (- (+ x0 0.5) cx)) (* sa dy)))
      (set! ly (+ (* sa (- (+ x0 0.5) cx)) (* ca dy)))
      ;; 源像素索引 (jj, ii) 及其展平下标 idx —— 行内用增量维护，内层循环不调用本地库
      (set! jj (Math.floor (+ ly DUCK_HHD)))
      (set! ii (Math.floor (+ lx DUCK_HWD)))
      (set! idx (+ (* jj (+ DUCK_W 0.0)) ii))
      (set! rp 0) (set! rn 0)
      (set! sx x0)
      (while (<= sx x1) {
        (set! pal 0)
        (if (and (and (>= ii 0.0) (< ii (+ DUCK_W 0.0)))
                 (and (>= jj 0.0) (< jj (+ DUCK_H 0.0))))
            { (set! pal (get_item DUCK_PIX idx)) }
            {})
        (if (== pal rp)
            { (set! rn (+ rn 1)) }              ; 同色 → 延长当前游程
            {
              (if (> rp 0)
                  { (Gfx.fill_rect rs sy rn 1
                                   (get_item (get_item DUCK_PAL rp) 0)
                                   (get_item (get_item DUCK_PAL rp) 1)
                                   (get_item (get_item DUCK_PAL rp) 2)) }
                  {})
              (set! rp pal) (set! rs sx) (set! rn 1) })
        ;; 前进一个目的像素 = 源坐标 (lx,ly) 各加 (ca,sa)；|ca|,|sa|<1 ⇒ 索引每步至多进/退 1
        (set! lx (+ lx ca))
        (set! ly (+ ly sa))
        (if (>= lx (- ii (- DUCK_HWD 1.0)))          ; 越过列边界 → ii+1
            { (set! ii (+ ii 1.0)) (set! idx (+ idx 1.0)) }
            {})
        (if (< lx (- ii DUCK_HWD))                   ; 越过列边界 → ii-1
            { (set! ii (- ii 1.0)) (set! idx (- idx 1.0)) }
            {})
        (if (>= ly (- jj (- DUCK_HHD 1.0)))          ; 越过行边界 → jj+1
            { (set! jj (+ jj 1.0)) (set! idx (+ idx (+ DUCK_W 0.0))) }
            {})
        (if (< ly (- jj DUCK_HHD))                   ; 越过行边界 → jj−1
            { (set! jj (- jj 1.0)) (set! idx (- idx (+ DUCK_W 0.0))) }
            {})
        (set! sx (+ sx 1)) })
      (if (> rp 0)
          { (Gfx.fill_rect rs sy rn 1
                           (get_item (get_item DUCK_PAL rp) 0)
                           (get_item (get_item DUCK_PAL rp) 1)
                           (get_item (get_item DUCK_PAL rp) 2)) }
          {})
      (set! sy (+ sy 1)) })))

;; ---- 渲染 ------------------------------------------------------------------
;; 粒子方块的颜色由速度分 12 档查表（CR/CG/CB），内层循环里只留 1 次 Math.sqrt + 1 次
;; Math.floor + 1 次 fill_rect —— 逐个 Math.floor 求 RGB 会把每颗粒子的本地库调用从
;; 3 次抬到 7 次，在解释器里是渲染开销的大头。
(define CBN 12)                   ; 颜色档数
(define CR (cons 0 (quote ())))
(define CG (cons 0 (quote ())))
(define CB (cons 0 (quote ())))
(define PSH (* PSZ 0.5))           ; 方块左上角相对粒子中心的偏移

(define build_colors
  (lambda ()
    (define k 0) (define t 0.0)
    (while (> (length CR) 0) { (pop CR) })
    (while (> (length CG) 0) { (pop CG) })
    (while (> (length CB) 0) { (pop CB) })
    (set! k 0)
    (while (< k CBN) {
      ;; 慢 = 深蓝，快 = 亮青/白（与原来的连续渐变同一条直线，只是量化成 12 档）
      (set! t (/ (+ k 0.0) (- CBN 1.0)))
      (push CR (Math.floor (+ 40.0 (* 190.0 t))))
      (push CG (Math.floor (+ 110.0 (* 130.0 t))))
      (push CB (Math.floor (+ 210.0 (* 40.0 t))))
      (set! k (+ k 1)) })))

(define render
  (lambda ()
    (define i 0) (define cb 0) (define sp 0.0) (define t 0.0)
    (define ax 0.0) (define ay 0.0)
    (Gfx.fill_rect 0 0 W VH 8 12 22)
    (duck_draw)
    (set! i 0)
    (while (< i np) {
      (set! ax (get_item vx i))
      (set! ay (get_item vy i))
      (set! sp (Math.sqrt (+ (* ax ax) (* ay ay))))
      (set! t (* sp 11.0))
      (if (> t 1.0) { (set! t 1.0) })
      (set! cb (Math.floor (* t (- CBN 1.0))))
      ;; 坐标不必先取整：fill_rect 的整型参数在本地库里按 float→int 截断（正值即 floor）
      (Gfx.fill_rect (- (* (get_item px i) CELL) PSH) (- (* (get_item py i) CELL) PSH)
                     PSZ PSZ (get_item CR cb) (get_item CG cb) (get_item CB cb))
      (set! i (+ i 1)) })
    (draw_gravity)))

;; ---- HUD -------------------------------------------------------------------

(define ds "")
(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))
(define cat4 (lambda (a b c d) (String.concat a (cat3 b c d))))

(define draw_hud
  (lambda ()
    (define deg 0)
    (Gfx.fill_rect 0 VH W HUD_H 10 12 20)
    (Gfx.draw_text "FLIP 流体" 4 (+ VH 2) 200 214 236)
    ;; 当前重力方向角（自动模式沿用滚筒角，重力感应模式取实测重力向量）
    (if (== gmode 1)
        { (set! deg (Math.floor (/ (* 180.0 (Math.atan2 gx gy)) (Math.PI)))) }
        { (set! deg (Math.floor (/ (* 180.0 drum_ang) (Math.PI)))) })
    (set! ds (cat4 (String.atom_to_string np) " 粒 " (String.atom_to_string frame_ms) "ms "))
    (Gfx.draw_text ds 170 (+ VH 2) 110 130 160)
    (set! ds (cat3 (if (== gmode 1) { "重力感应 " }
                                { (if (> drum_s 0.5) { "脱水 " } { "洗涤 " }) })
                   (String.atom_to_string deg) "°"))
    (Gfx.draw_text ds 250 (+ VH 2) 140 190 220)
    ;; 左下角：重力源切换按钮
    (draw_toggle)))

;; ---- 左下角按钮：自动（滚筒）/ 重力感应（IMU）切换 --------------------------
;; 画布左下角、HUD 条上方；触屏一次按下切换一次。重力感应模式下若 IMU 不可用
;;（无传感器 / 浏览器未授权姿态），按钮文字转为橙色提示，此时重力退化为"屏幕向下"。
(define draw_toggle
  (lambda ()
    (if (== gmode 0)
        {
          (Gfx.fill_rect TOG_X TOG_Y TOG_W TOG_H 26 36 56)
          (Gfx.draw_rect TOG_X TOG_Y TOG_W TOG_H 78 104 150)
          (Gfx.draw_text "自动" (+ TOG_X 7) (+ TOG_Y 3) 200 214 236) }
        {
          (Gfx.fill_rect TOG_X TOG_Y TOG_W TOG_H 22 56 38)
          (Gfx.draw_rect TOG_X TOG_Y TOG_W TOG_H 96 200 130)
          (if (== imu_ok 1)
              { (Gfx.draw_text "重力感应" (+ TOG_X 6) (+ TOG_Y 3) 210 255 224) }
              { (Gfx.draw_text "无传感器" (+ TOG_X 6) (+ TOG_Y 3) 255 170 120) }) })))

;; 触屏：按钮命中则切换重力源（take_press 为"按下沿"锁存，一次按下只切一次）
(define input_toggle
  (lambda ()
    (if (Touch.take_press)
        {
          (if (and (and (>= (Touch.x) TOG_X) (< (Touch.x) (+ TOG_X TOG_W)))
                   (and (>= (Touch.y) TOG_Y) (< (Touch.y) (+ TOG_Y TOG_H))))
              { (set! gmode (- 1 gmode))
                ;; 切到重力感应时立即采一次，避免第一帧用上次的旧值
                (if (== gmode 1)
                    { (set! imu_gx (Imu.gravity_x))
                      (set! imu_gy (Imu.gravity_y))
                      (set! imu_ok (if (Imu.is_available) 1 0)) }
                    {}) }
              {}) }
        {})))

;; ---- 重力指示器（左上角）---------------------------------------------------
;; 箭头方向 = 当前重力方向（屏幕坐标 y 向下，与 gx/gy 一致）；
;; 箭头长度 ∝ 重力大小（满量程 = 脱水时的 2.6×G_RATE，正好顶到参考圈）；
;; 右侧一行字给出以 G_RATE 为单位的倍数。自动模式下恒为 1G~2.6G；
;; 重力感应模式下恒为 1G、方向随设备倾斜实时变化。整块画在流体之上。
(define GARR_X 18.0)              ; 箭头原点（像素）
(define GARR_Y 17.0)
(define GARR_R 11.0)              ; 满量程箭头长度（像素）= 参考圈半径
(define GARR_GMAX (* 2.6 G_RATE)) ; 满量程重力
(define ds2 "")

(define draw_gravity
  (lambda ()
    (define gm2 0.0) (define m 0.0) (define ux 0.0) (define uy 0.0) (define ln 0.0)
    (define tx 0.0) (define ty 0.0) (define bx 0.0) (define by 0.0)
    (define hl 0.0) (define hw 0.0)
    (set! gm2 (Math.sqrt (+ (* gx gx) (* gy gy))))
    (if (< gm2 0.000001) { (set! gm2 0.000001) })
    (set! ux (/ gx gm2))
    (set! uy (/ gy gm2))
    (set! m (/ gm2 GARR_GMAX))
    (if (> m 1.0) { (set! m 1.0) })
    (set! ln (+ 4.5 (* (- GARR_R 4.5) m)))    ; 4.5 .. 11 px
    (set! hl (+ 2.0 (* 3.0 m)))               ; 箭头长度 2.0 .. 5.0 px
    (set! hw (+ 1.8 (* 1.6 m)))               ; 半宽 1.8 .. 3.4 px
    ;; 底板 + 满量程参考圈 + 原点（重力感应模式改用绿色系，一眼看出重力源）
    (Gfx.fill_rect 3 3 68 30 13 17 27)
    (if (== gmode 1)
        {
          (Gfx.draw_rect 3 3 68 30 52 132 78)
          (Gfx.draw_circle 18 17 (Math.floor GARR_R) 38 104 62) }
        {
          (Gfx.draw_rect 3 3 68 30 62 84 116)
          (Gfx.draw_circle 18 17 (Math.floor GARR_R) 46 62 86) })
    (Gfx.fill_circle 18 17 1 150 165 190)
    ;; 箭杆 + 两根倒刺
    (set! tx (+ GARR_X (* ln ux)))
    (set! ty (+ GARR_Y (* ln uy)))
    (set! bx (- tx (* hl ux)))
    (set! by (- ty (* hl uy)))
    (Gfx.draw_line (Math.floor GARR_X) (Math.floor GARR_Y)
                   (Math.floor tx) (Math.floor ty) 255 196 64)
    (Gfx.draw_line (Math.floor tx) (Math.floor ty)
                   (Math.floor (+ bx (* hw uy))) (Math.floor (- by (* hw ux)))
                   255 196 64)
    (Gfx.draw_line (Math.floor tx) (Math.floor ty)
                   (Math.floor (- bx (* hw uy))) (Math.floor (+ by (* hw ux)))
                   255 196 64)
    ;; 数值（单位 = G_RATE）
    (set! ds2 (cat3 (String.atom_to_string (Math.to_fixed (/ gm2 G_RATE) 1)) "" "G"))
    (Gfx.draw_text ds2 33 11 255 214 120)))

;; ---- 主流程 ----------------------------------------------------------------

;; 一次完整物理步（照 flip.c 的 simulate_step 顺序）
(define simulate
  (lambda ()
    (integrate)
    (p2g)
    (separate)
    (duck_sweep)     ; 量水线 + 把鸭身内的水挤开（先量后算，鸭子当拍就用上最新水面）
    (duck_step)
    (save_prev)
    (solve_pressure)
    (g2p)))

(define tick_fn
  (lambda ()
    (define t0 0)
    (set! t0 (Diag.ms))
    (input_toggle)                   ; 左下角按钮：自动 / 重力感应
    (turn_gravity)
    (if (== (mod tick_no PHYS_EVERY) 0) { (simulate) })
    (render)
    (set! frame_ms (- (Diag.ms) t0))
    (draw_hud)
    (Gfx.refresh)
    (set! tick_no (+ tick_no 1))))

;; ---- 启动 ------------------------------------------------------------------

(pop px) (pop py) (pop vx) (pop vy)
(pop u) (pop v) (pop pu) (pop pv) (pop du) (pop dv) (pop s) (pop ct) (pop fl) (pop dens)
(pop pcount) (pop pstart) (pop pids) (pop pw)
(pop F) (pop G) (pop D)
(build_grid)
(build_colors)
(duck_build)
(init_fluid)
(duck_init)
(set! _timer (System.set_interval TICK_MS tick_fn))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 render 擦掉对话框残影）
(EX.install_with render)
