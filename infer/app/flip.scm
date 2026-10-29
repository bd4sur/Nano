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
;;   网格 26×18（每格 12 px）、粒子 **248**（RAD=0.42，约 1.4 粒/格）、
;;   压力迭代 12 轮 + **粒子分离**（桶排序 + 3×3 邻居，照 flip.c 的 push_particles_apart）；
;;   平均约 **18 ms/拍** → 40 ms 定时器下约 45% CPU；浏览器 wasm32 约 45 ms（≈22 fps）。
;;
;; 重力 = **洗衣机滚筒**：重力方向连续旋转 360°（不是来回摆），转速按
;;   "洗涤（慢转）→ 加速 → 脱水（快转）→ 减速" 循环（周期 900 拍 ≈ 45 s@20fps）：
;;     drum_ang += ω，ω 在 0.030（约 6 s 一圈）~ 0.170（约 1.1 s 一圈）之间
;;     gx = gm·sin(drum_ang)，gy = gm·cos(drum_ang)，脱水时 gm = 2.6×G_RATE
;;     另有离心项 cent_k·(p − 筒心)：脱水时把水压向筒壁，中间留空 —— 就是滚筒甩干的样子。
;;
;; 两个关键点（都是实测出来的）：
;;   1) **粒子分离是"敢大幅摇晃"的前提**。没有它，流体在持续晃动下会慢慢扩散成雾气
;;      （几千帧后彻底散开）；加上它之后，248 粒也能长时间保持成一团水。
;;   2) 摇摆幅度/阻尼是量级问题：G_RATE 0.005 + VEL_DAMP 0.86 时终端速度只有
;;      0.018 格/步，液面几乎不动；0.010 + 0.94 才能看到明显晃动。
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
(native Math)
(native String)
(native System)
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
(define RAD 0.42)                 ; 粒子半径（格）：间距 2R，粒子数 ≈ 面积/dx²
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
      (set! c (+ (* x0 GH) y0))
      (if (== (get_item ct c) 1) { (set_item! ct c 0) (push fl c) } {})
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
    (define it 0) (define m 0) (define nf 0) (define c 0) (define j 0)
    (define s0 0.0) (define s1 0.0) (define s2 0.0) (define s3 0.0) (define ss 0.0)
    (define dv0 0.0) (define pv0 0.0) (define el 0.0)
    (set! nf (length fl))
    (set! it 0)
    (while (< it PRESS_ITERS) {
      (set! m 0)
      (while (< m nf) {
        (set! c (get_item fl m))
        (set! j (Math.floor (mod c GH)))
        (if (> j 0)
            { (if (< j (- GH 1))
                  {
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
                        {}) }
                  {}) }
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
    ;; 洗衣机滚筒：重力方向**连续旋转 360°**（不是来回摆），转速按"洗涤 → 脱水 → 洗涤"
    ;; 循环变化，脱水时表观重力也加大（转速越快，离心力把水压在筒壁上）。
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
    (set! cent_k (* CENT_K (* drum_s drum_s)))))
;; ---- 渲染 ------------------------------------------------------------------

(define render
  (lambda ()
    (define i 0) (define x 0) (define y 0) (define sp 0.0) (define t 0.0)
    (define r 0) (define g 0) (define b 0) (define ax 0.0) (define ay 0.0)
    (Gfx.fill_rect 0 0 W VH 8 12 22)
    (set! i 0)
    (while (< i np) {
      (set! x (Math.floor (* (get_item px i) CELL)))
      (set! y (Math.floor (* (get_item py i) CELL)))
      (set! ax (get_item vx i))
      (set! ay (get_item vy i))
      (set! sp (Math.sqrt (+ (* ax ax) (* ay ay))))
      (set! t (* sp 11.0))
      (if (> t 1.0) { (set! t 1.0) })
      ;; 慢=深蓝，快=亮青/白
      (set! r (Math.floor (+ 40.0 (* 190.0 t))))
      (set! g (Math.floor (+ 110.0 (* 130.0 t))))
      (set! b (Math.floor (+ 210.0 (* 40.0 t))))
      (Gfx.fill_rect x y 8 8 r g b)
      (set! i (+ i 1)) })))

;; ---- HUD -------------------------------------------------------------------

(define ds "")
(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))
(define cat4 (lambda (a b c d) (String.concat a (cat3 b c d))))

(define draw_hud
  (lambda ()
    (define deg 0)
    (Gfx.fill_rect 0 VH W HUD_H 10 12 20)
    (Gfx.draw_text "FLIP 流体 · 洗衣机滚筒" 4 (+ VH 2) 200 214 236)
    (set! deg (Math.floor (/ (* 180.0 drum_ang) (Math.PI))))
    (set! ds (cat4 (String.atom_to_string np) " 粒 " (String.atom_to_string frame_ms) "ms "))
    (Gfx.draw_text ds 170 (+ VH 2) 110 130 160)
    (set! ds (cat3 (if (> drum_s 0.5) { "脱水 " } { "洗涤 " })
                   (String.atom_to_string deg) "°"))
    (Gfx.draw_text ds 262 (+ VH 2) 140 190 220)))

;; ---- 主流程 ----------------------------------------------------------------

;; 一次完整物理步（照 flip.c 的 simulate_step 顺序）
(define simulate
  (lambda ()
    (integrate)
    (p2g)
    (separate)
    (save_prev)
    (solve_pressure)
    (g2p)))

(define tick_fn
  (lambda ()
    (define t0 0)
    (set! t0 (Diag.ms))
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
(init_fluid)
(set! _timer (System.set_interval TICK_MS tick_fn))