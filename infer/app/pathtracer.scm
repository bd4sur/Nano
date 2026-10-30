;; ============================================================================
;; 渐进路径追踪（Progressive Path Tracer）—— 灵机应用程序
;;
;; 运行环境：灵机（Animac）解释器 + Gfx/Diag 本地库（宿主 ui_scheme_app）
;;
;; 做法：40×30 低分辨率路径追踪，**每 tick 为每个像素追加 1 个采样**
;; （1200 采样/帧），累积缓冲 acc[] 取平均后经 色调映射（曝光 → Reinhard →
;; gamma 2.2）以 8×8 方块上屏铺满 320×240 —— 画面从满屏噪声开始，像胶片
;; 显影一样逐渐收敛成照片。与 sdfprog 的"逐行显影"同哲学（重活摊到多帧的
;; 解释器协作调度），但观感不同：这里是整幅画面同步地"从噪声里长出来"。
;;
;; 场景（Cornell box 简版）：单位盒子 [0,1]^3（前口 z=0 敞开，相机在盒外
;; 向内看），左墙黄、右墙蓝、地板/天花板/后墙白，天花板上有一块 0.24×0.24
;; 的矩形面光源（暖色）；盒内三个落在地板上的球（两个中性色 + 一个淡钢蓝）。
;;
;; 收敛快的关键：**NEE（Next-Event Estimation，直接光采样）**——每次命中
;; 非光源表面时，在光源矩形上均匀采一个点并算解析直接光（可见性用阴影射线
;; 只对 3 个球测试：盒子是凸的，墙不可能遮挡盒内点到光源的连线）；然后按
;; 余弦加权半球采样继续弹射（Lambertian：BRDF/pdf 相消，吞吐量只乘反照率）。
;; 弹射深度固定 3（无俄罗斯轮盘）；弹射射线命中光源不再重复计发射项
;; （该连接已由上一次命中点的 NEE 采样过）；射线从前口逸出记 0 贡献。
;;
;; 成本实测（host x86-64，解释器吞吐 ≈108M VM 指令/秒）：每 tick ≈ 1.5M 指令
;; （1200 采样 × ~1200 指令/采样 + 1200 次色调映射/上屏）≈ 14ms/tick，宿主
;; 时间片 312500 指令/帧 → 约 5 帧推进 1 个采样层。浏览器 wasm32（≈18.7M
;; 指令/秒）约 80ms/tick ≈ 12 层/秒：显影过程肉眼可见（1 秒辨认轮廓，10 秒
;; 柔和成形，60 秒接近干净）。
;;
;; 书写约定（本解释器的坑，详见手册与 app/sdfprog.scm 头部；写前已对照）：
;;   算术一律二元嵌套（三操作数会被错误编译且静默算错，可用
;;   infer/tools/e2e/host/check_multi_operand.py 静态检查）；无一元取负，
;;   写 (- 0.0 x)；/ 与 mod 恒返回浮点；Math.PI 是函数（这里直接硬编码 π/τ）；
;;   Math 库没有 min/max；cond 子句只能带一个表达式 → 一律平铺 if；
;;   '() 字面量是模块级静态槽，累积表用 (cdr '(0.0)) 取新对象再 push；
;;   while 循环体内不写 define，局部量全部在 lambda 体顶部 define；
;;   if 的真值只有 #f 是假，数值标志一律 (== f 1) 判断；
;;   语句位置调用产值函数会泄漏操作数栈 → 本文件产值调用全部接 set! 或
;;   位于尾位置（Gfx.* / set_item! / push 等本身是语句型，安全）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Diag)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC / 右上角按钮 → 全局退出确认

;; ---- 渲染/调度常量 ---------------------------------------------------------

(define RW 40)                    ; 渲染分辨率（像素）
(define RH 30)
(define PXW 8)                    ; 上屏方块边长（40×8=320, 30×8=240）
(define RWF 40.0)
(define RHF 30.0)
(define SPP_MAX 4096)             ; 累积上限（浮点有效位与时长双保险）
(define TICK_MS 40)
(define MAXD 3)                   ; 弹射深度（相机命中 + 2 次间接弹射）
(define EPS_T 0.0001)             ; 求交 t 下限（防自交）
(define EPS_O 0.0002)             ; 弹射原点沿法线偏移
(define TAU 6.28318530717959)
(define EXPOS 1.9)                ; 曝光
(define GAMMA 0.4545)             ; 1/2.2

;; ---- 场景常量（球：心/半径/半径平方/半径倒数/反照率）------------------------

(define S0X 0.30) (define S0Y 0.25) (define S0Z 0.38)
(define S0R2 0.0625) (define S0RI 4.0)
(define S0AR 0.78) (define S0AG 0.78) (define S0AB 0.76)

(define S1X 0.68) (define S1Y 0.17) (define S1Z 0.62)
(define S1R2 0.0289) (define S1RI 5.88235294117647)
(define S1AR 0.76) (define S1AG 0.74) (define S1AB 0.70)

(define S2X 0.55) (define S2Y 0.10) (define S2Z 0.20)
(define S2R2 0.01) (define S2RI 10.0)
(define S2AR 0.60) (define S2AG 0.68) (define S2AB 0.82)

;; 面光源：天花板（y=1）上的矩形 [LX0,LX1]×[LZ0,LZ1]，法线 (0,-1,0)，暖色
(define LX0 0.38) (define LZ0 0.38) (define LSZ 0.24)
(define LX1 0.62) (define LZ1 0.62)
(define L_AOPI 0.0183341)          ; 面积/π = 0.24×0.24/π
(define LE_R 24.0) (define LE_G 22.0) (define LE_B 16.5)

;; 墙面反照率：白（地板/天花板/后墙）/ 左黄 / 右蓝
(define WA 0.80)
(define LR_R 0.62) (define LR_G 0.50) (define LR_B 0.12)
(define RR_R 0.10) (define RR_G 0.20) (define RR_B 0.58)

;; ---- 相机（固定机位，从盒外前口向内看）--------------------------------------

(define eye_x 0.5) (define eye_y 0.58) (define eye_z -1.05)
(define look_x 0.5) (define look_y 0.42) (define look_z 0.55)
(define fwd_x 0.0) (define fwd_y 0.0) (define fwd_z 0.0)
(define rgt_x 1.0) (define rgt_y 0.0) (define rgt_z 0.0)
(define up_x 0.0) (define up_y 1.0) (define up_z 0.0)
(define U_MAX 0.60)               ; 半幅视场（水平）；竖直 = ×0.75
(define V_MAX 0.45)

(define setup_camera
  (lambda ()
    (define cl 0.0)
    (set! fwd_x (- look_x eye_x))
    (set! fwd_y (- look_y eye_y))
    (set! fwd_z (- look_z eye_z))
    (set! cl (Math.sqrt (+ (* fwd_x fwd_x) (+ (* fwd_y fwd_y) (* fwd_z fwd_z)))))
    (set! fwd_x (/ fwd_x cl))
    (set! fwd_y (/ fwd_y cl))
    (set! fwd_z (/ fwd_z cl))
    ;; right = normalize(cross((0,1,0), fwd)) = normalize((fwd_z, 0, -fwd_x))
    (set! rgt_x fwd_z)
    (set! rgt_y 0.0)
    (set! rgt_z (- 0.0 fwd_x))
    (set! cl (Math.sqrt (+ (* rgt_x rgt_x) (* rgt_z rgt_z))))
    (set! rgt_x (/ rgt_x cl))
    (set! rgt_z (/ rgt_z cl))
    ;; up = cross(fwd, right)
    (set! up_x (- (* fwd_y rgt_z) (* fwd_z rgt_y)))
    (set! up_y (- (* fwd_z rgt_x) (* fwd_x rgt_z)))
    (set! up_z (- (* fwd_x rgt_y) (* fwd_y rgt_x)))))

;; ---- 路径追踪核心 ----------------------------------------------------------
;; 结果写模块级 res_r/res_g/res_b（解释器无多值返回）。

(define res_r 0.0) (define res_g 0.0) (define res_b 0.0)

(define trace_sample
  (lambda (ox oy oz dx dy dz)
    (define depth 0)
    (define th_r 1.0) (define th_g 1.0) (define th_b 1.0)   ; 路径吞吐量
    (define tb 1.0e9) (define tobj -1)                       ; 最近命中 t / 物体号
    (define ocx 0.0) (define ocy 0.0) (define ocz 0.0)
    (define bq 0.0) (define cq 0.0) (define disc 0.0) (define sq 0.0) (define tt 0.0)
    (define hx 0.0) (define hy 0.0) (define hz 0.0)          ; 命中点
    (define nx 0.0) (define ny 1.0) (define nz 0.0)          ; 法线
    (define ar 0.73) (define ag 0.73) (define ab 0.73)       ; 反照率
    (define is_light 0)
    (define lx 0.0) (define lz 0.0)
    (define Lx 0.0) (define Ly 0.0) (define Lz 0.0)
    (define d2 0.0) (define dist 0.0)
    (define wx 0.0) (define wy 0.0) (define wz 0.0)
    (define ndotl 0.0) (define blk 0) (define gg 0.0)
    (define r1 0.0) (define r2 0.0) (define rr2 0.0) (define phi 0.0)
    (define cu 0.0) (define cv 0.0) (define cw2 0.0)
    (define ux 0.0) (define uy 0.0) (define uz 0.0) (define ul 0.0)
    (define vx 0.0) (define vy 0.0) (define vz 0.0)
    (set! res_r 0.0) (set! res_g 0.0) (set! res_b 0.0)
    (while (< depth MAXD) {
      ;; ================ 最近求交 ================
      (set! tb 1.0e9)
      (set! tobj -1)
      ;; ---- 球 0 ----（d 已单位化：t = -b ± sqrt(b²-c)，b=oc·d，c=|oc|²-r²）
      (set! ocx (- ox S0X))
      (set! ocy (- oy S0Y))
      (set! ocz (- oz S0Z))
      (set! bq (+ (+ (* ocx dx) (* ocy dy)) (* ocz dz)))
      (set! cq (- (+ (+ (* ocx ocx) (* ocy ocy)) (* ocz ocz)) S0R2))
      (set! disc (- (* bq bq) cq))
      (if (> disc 0.0) {
        (set! sq (Math.sqrt disc))
        (set! tt (- (- 0.0 bq) sq))
        (if (<= tt EPS_T) { (set! tt (+ (- 0.0 bq) sq)) })
        (if (and (> tt EPS_T) (< tt tb)) { (set! tb tt) (set! tobj 0) })
      })
      ;; ---- 球 1 ----
      (set! ocx (- ox S1X))
      (set! ocy (- oy S1Y))
      (set! ocz (- oz S1Z))
      (set! bq (+ (+ (* ocx dx) (* ocy dy)) (* ocz dz)))
      (set! cq (- (+ (+ (* ocx ocx) (* ocy ocy)) (* ocz ocz)) S1R2))
      (set! disc (- (* bq bq) cq))
      (if (> disc 0.0) {
        (set! sq (Math.sqrt disc))
        (set! tt (- (- 0.0 bq) sq))
        (if (<= tt EPS_T) { (set! tt (+ (- 0.0 bq) sq)) })
        (if (and (> tt EPS_T) (< tt tb)) { (set! tb tt) (set! tobj 1) })
      })
      ;; ---- 球 2 ----
      (set! ocx (- ox S2X))
      (set! ocy (- oy S2Y))
      (set! ocz (- oz S2Z))
      (set! bq (+ (+ (* ocx dx) (* ocy dy)) (* ocz dz)))
      (set! cq (- (+ (+ (* ocx ocx) (* ocy ocy)) (* ocz ocz)) S2R2))
      (set! disc (- (* bq bq) cq))
      (if (> disc 0.0) {
        (set! sq (Math.sqrt disc))
        (set! tt (- (- 0.0 bq) sq))
        (if (<= tt EPS_T) { (set! tt (+ (- 0.0 bq) sq)) })
        (if (and (> tt EPS_T) (< tt tb)) { (set! tb tt) (set! tobj 2) })
      })
      ;; ---- 地板 y=0（仅向下时相交；墙是有限平面，须做边界检查）----
      (if (< dy 0.0) {
        (set! tt (/ (- 0.0 oy) dy))
        (if (and (> tt EPS_T) (< tt tb)) {
          (set! hx (+ ox (* tt dx)))
          (if (and (>= hx 0.0) (<= hx 1.0)) {
            (set! hz (+ oz (* tt dz)))
            (if (and (>= hz 0.0) (<= hz 1.0)) { (set! tb tt) (set! tobj 3) })
          })
        })
      })
      ;; ---- 天花板 y=1（含光源矩形）----
      (if (> dy 0.0) {
        (set! tt (/ (- 1.0 oy) dy))
        (if (and (> tt EPS_T) (< tt tb)) {
          (set! hx (+ ox (* tt dx)))
          (if (and (>= hx 0.0) (<= hx 1.0)) {
            (set! hz (+ oz (* tt dz)))
            (if (and (>= hz 0.0) (<= hz 1.0)) { (set! tb tt) (set! tobj 4) })
          })
        })
      })
      ;; ---- 后墙 z=1 ----
      (if (> dz 0.0) {
        (set! tt (/ (- 1.0 oz) dz))
        (if (and (> tt EPS_T) (< tt tb)) {
          (set! hx (+ ox (* tt dx)))
          (if (and (>= hx 0.0) (<= hx 1.0)) {
            (set! hy (+ oy (* tt dy)))
            (if (and (>= hy 0.0) (<= hy 1.0)) { (set! tb tt) (set! tobj 5) })
          })
        })
      })
      ;; ---- 左墙 x=0（黄）----
      (if (< dx 0.0) {
        (set! tt (/ (- 0.0 ox) dx))
        (if (and (> tt EPS_T) (< tt tb)) {
          (set! hy (+ oy (* tt dy)))
          (if (and (>= hy 0.0) (<= hy 1.0)) {
            (set! hz (+ oz (* tt dz)))
            (if (and (>= hz 0.0) (<= hz 1.0)) { (set! tb tt) (set! tobj 6) })
          })
        })
      })
      ;; ---- 右墙 x=1（蓝）----
      (if (> dx 0.0) {
        (set! tt (/ (- 1.0 ox) dx))
        (if (and (> tt EPS_T) (< tt tb)) {
          (set! hy (+ oy (* tt dy)))
          (if (and (>= hy 0.0) (<= hy 1.0)) {
            (set! hz (+ oz (* tt dz)))
            (if (and (>= hz 0.0) (<= hz 1.0)) { (set! tb tt) (set! tobj 7) })
          })
        })
      })
      (if (< tobj 0) break)                  ; 从前口逸出：0 贡献
      ;; ================ 命中点 / 法线 / 材质 ================
      (set! hx (+ ox (* tb dx)))
      (set! hy (+ oy (* tb dy)))
      (set! hz (+ oz (* tb dz)))
      (set! is_light 0)
      (if (== tobj 0) {
        (set! nx (* (- hx S0X) S0RI)) (set! ny (* (- hy S0Y) S0RI)) (set! nz (* (- hz S0Z) S0RI))
        (set! ar S0AR) (set! ag S0AG) (set! ab S0AB) })
      (if (== tobj 1) {
        (set! nx (* (- hx S1X) S1RI)) (set! ny (* (- hy S1Y) S1RI)) (set! nz (* (- hz S1Z) S1RI))
        (set! ar S1AR) (set! ag S1AG) (set! ab S1AB) })
      (if (== tobj 2) {
        (set! nx (* (- hx S2X) S2RI)) (set! ny (* (- hy S2Y) S2RI)) (set! nz (* (- hz S2Z) S2RI))
        (set! ar S2AR) (set! ag S2AG) (set! ab S2AB) })
      (if (== tobj 3) { (set! nx 0.0) (set! ny 1.0) (set! nz 0.0)
                        (set! ar WA) (set! ag WA) (set! ab WA) })
      (if (== tobj 5) { (set! nx 0.0) (set! ny 0.0) (set! nz -1.0)
                        (set! ar WA) (set! ag WA) (set! ab WA) })
      (if (== tobj 6) { (set! nx 1.0) (set! ny 0.0) (set! nz 0.0)
                        (set! ar LR_R) (set! ag LR_G) (set! ab LR_B) })
      (if (== tobj 7) { (set! nx -1.0) (set! ny 0.0) (set! nz 0.0)
                        (set! ar RR_R) (set! ag RR_G) (set! ab RR_B) })
      (if (== tobj 4) {
        (if (and (>= hx LX0) (and (<= hx LX1) (and (>= hz LZ0) (<= hz LZ1))))
            { (set! is_light 1) }
            { (set! nx 0.0) (set! ny -1.0) (set! nz 0.0)
              (set! ar WA) (set! ag WA) (set! ab WA) })
      })
      ;; 相机直接看见光源：计发射项；弹射射线命中光源：NEE 已采样过，不加（防双计）
      (if (== is_light 1) {
        (if (== depth 0) {
          (set! res_r (* th_r LE_R))
          (set! res_g (* th_g LE_G))
          (set! res_b (* th_b LE_B)) })
        break
      })
      ;; ================ NEE 直接光 ================
      ;; 弹射原点沿法线移出表面（阴影射线与弹射射线共用）
      (set! ox (+ hx (* nx EPS_O)))
      (set! oy (+ hy (* ny EPS_O)))
      (set! oz (+ hz (* nz EPS_O)))
      (set! lx (+ LX0 (* LSZ (Math.random))))
      (set! lz (+ LZ0 (* LSZ (Math.random))))
      (set! Lx (- lx ox))
      (set! Ly (- 1.0 oy))
      (set! Lz (- lz oz))
      (set! d2 (+ (+ (* Lx Lx) (* Ly Ly)) (* Lz Lz)))
      (set! dist (Math.sqrt d2))
      (set! wx (/ Lx dist))
      (set! wy (/ Ly dist))
      (set! wz (/ Lz dist))
      (set! ndotl (+ (+ (* nx wx) (* ny wy)) (* nz wz)))
      ;; cosl = -w·(0,-1,0) = wy；ndotl>0 且 wy>0 时才有直接光
      (if (and (> ndotl 0.0) (> wy 0.0)) {
        (set! blk 0)
        ;; 阴影测试只测 3 个球（盒子凸，墙不会遮挡到光源的连线）
        (set! ocx (- ox S0X))
        (set! ocy (- oy S0Y))
        (set! ocz (- oz S0Z))
        (set! bq (+ (+ (* ocx wx) (* ocy wy)) (* ocz wz)))
        (set! cq (- (+ (+ (* ocx ocx) (* ocy ocy)) (* ocz ocz)) S0R2))
        (set! disc (- (* bq bq) cq))
        (if (> disc 0.0) {
          (set! sq (Math.sqrt disc))
          (set! tt (- (- 0.0 bq) sq))
          (if (<= tt EPS_T) { (set! tt (+ (- 0.0 bq) sq)) })
          (if (and (> tt EPS_T) (< tt (- dist 0.001))) { (set! blk 1) })
        })
        (if (== blk 0) {
          (set! ocx (- ox S1X))
          (set! ocy (- oy S1Y))
          (set! ocz (- oz S1Z))
          (set! bq (+ (+ (* ocx wx) (* ocy wy)) (* ocz wz)))
          (set! cq (- (+ (+ (* ocx ocx) (* ocy ocy)) (* ocz ocz)) S1R2))
          (set! disc (- (* bq bq) cq))
          (if (> disc 0.0) {
            (set! sq (Math.sqrt disc))
            (set! tt (- (- 0.0 bq) sq))
            (if (<= tt EPS_T) { (set! tt (+ (- 0.0 bq) sq)) })
            (if (and (> tt EPS_T) (< tt (- dist 0.001))) { (set! blk 1) })
          })
        })
        (if (== blk 0) {
          (set! ocx (- ox S2X))
          (set! ocy (- oy S2Y))
          (set! ocz (- oz S2Z))
          (set! bq (+ (+ (* ocx wx) (* ocy wy)) (* ocz wz)))
          (set! cq (- (+ (+ (* ocx ocx) (* ocy ocy)) (* ocz ocz)) S2R2))
          (set! disc (- (* bq bq) cq))
          (if (> disc 0.0) {
            (set! sq (Math.sqrt disc))
            (set! tt (- (- 0.0 bq) sq))
            (if (<= tt EPS_T) { (set! tt (+ (- 0.0 bq) sq)) })
            (if (and (> tt EPS_T) (< tt (- dist 0.001))) { (set! blk 1) })
          })
        })
        (if (== blk 0) {
          ;; g = ndotl · cosl · (A/π) / d²；贡献 = 吞吐 × 反照率 × Le × g
          (set! gg (* ndotl (* wy (/ L_AOPI d2))))
          (set! res_r (+ res_r (* th_r (* ar (* LE_R gg)))))
          (set! res_g (+ res_g (* th_g (* ag (* LE_G gg)))))
          (set! res_b (+ res_b (* th_b (* ab (* LE_B gg)))))
        })
      })
      ;; ================ 余弦加权半球弹射 ================
      ;; Lambertian：BRDF/pdf 相消，吞吐量只乘反照率
      (set! th_r (* th_r ar))
      (set! th_g (* th_g ag))
      (set! th_b (* th_b ab))
      ;; 最后一弹的方向不再使用（循环即结束），跳过采样省 ~5%
      (if (< depth (- MAXD 1)) {
        (set! r1 (Math.random))
        (if (> r1 0.9999) { (set! r1 0.9999) })    ; 兜住 1-r1 <= 0
        (set! r2 (Math.random))
        (set! rr2 (Math.sqrt r1))
        (set! phi (* TAU r2))
        (set! cu (* rr2 (Math.cos phi)))
        (set! cv (* rr2 (Math.sin phi)))
        (set! cw2 (Math.sqrt (- 1.0 r1)))
        ;; 局部标架（smallpt 技巧，|nx|>0.1 时取 (0,1,0)×n 否则 (1,0,0)×n）
        (if (> (Math.abs nx) 0.1)
            { (set! ux nz) (set! uy 0.0) (set! uz (- 0.0 nx)) }
            { (set! ux 0.0) (set! uy (- 0.0 nz)) (set! uz ny) })
        (set! ul (Math.sqrt (+ (+ (* ux ux) (* uy uy)) (* uz uz))))
        (set! ux (/ ux ul))
        (set! uy (/ uy ul))
        (set! uz (/ uz ul))
        ;; v = cross(n, u)
        (set! vx (- (* ny uz) (* nz uy)))
        (set! vy (- (* nz ux) (* nx uz)))
        (set! vz (- (* nx uy) (* ny ux)))
        ;; d = u·cu + v·cv + n·cw（标架正交，结果已单位化）
        (set! dx (+ (+ (* ux cu) (* vx cv)) (* nx cw2)))
        (set! dy (+ (+ (* uy cu) (* vy cv)) (* ny cw2)))
        (set! dz (+ (+ (* uz cu) (* vz cv)) (* nz cw2)))
      })
      (set! depth (+ depth 1))
    })
    res_r))

;; ---- 上屏（色调映射 + 8×8 方块）---------------------------------------------

(define acc (cdr '(0.0)))          ; 3600 项累积缓冲（扁平 rgb 交错）
(define spp 0)                     ; 已累积的每像素采样数
(define last_ms 0)

(define redraw_all
  (lambda ()
    (define i 0) (define px 0) (define py 0)
    (define lr 0.0) (define lg 0.0) (define lb 0.0)
    (define cr 0) (define cg 0) (define cb 0)
    (define inv 1.0)
    (if (> spp 0)
        {
          (set! inv (/ 1.0 spp))
          (set! py 0)
          (while (< py RH) {
            (set! px 0)
            (while (< px RW) {
              ;; 均值 → 曝光 → Reinhard → gamma → 0..255
              (set! lr (* (* (get_item acc i) inv) EXPOS))
              (set! lg (* (* (get_item acc (+ i 1)) inv) EXPOS))
              (set! lb (* (* (get_item acc (+ i 2)) inv) EXPOS))
              (set! lr (/ lr (+ 1.0 lr)))
              (set! lg (/ lg (+ 1.0 lg)))
              (set! lb (/ lb (+ 1.0 lb)))
              (set! cr (Math.floor (* 255.0 (Math.pow lr GAMMA))))
              (set! cg (Math.floor (* 255.0 (Math.pow lg GAMMA))))
              (set! cb (Math.floor (* 255.0 (Math.pow lb GAMMA))))
              (if (< cr 0) { (set! cr 0) })
              (if (> cr 255) { (set! cr 255) })
              (if (< cg 0) { (set! cg 0) })
              (if (> cg 255) { (set! cg 255) })
              (if (< cb 0) { (set! cb 0) })
              (if (> cb 255) { (set! cb 255) })
              (Gfx.fill_rect (* px PXW) (* py PXW) PXW PXW cr cg cb)
              (set! i (+ i 3))
              (set! px (+ px 1))
            })
            (set! py (+ py 1))
          })
        }
        { (Gfx.fill_rect 0 0 (Gfx.width) (Gfx.height) 0 0 0) })
    (draw_hud)))

;; ---- HUD -------------------------------------------------------------------

(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))
(define cat5 (lambda (a b c d e) (String.concat a (String.concat b (cat3 c d e)))))

(define hud_s "")
(define done_note 0)

(define draw_hud
  (lambda ()
    (define pw 0)
    (Gfx.fill_rect 0 0 200 14 8 10 14)
    (if (< spp SPP_MAX)
        (set! hud_s (cat5 "路径追踪 · spp " (String.atom_to_string spp) " · "
                          (String.atom_to_string last_ms) "ms"))
        (set! hud_s (cat3 "路径追踪 · spp " (String.atom_to_string spp) " · 显影完成")))
    (Gfx.draw_text hud_s 4 1 228 224 208)
    ;; 显影进度条
    (set! pw (Math.floor (/ (* 200.0 spp) 4096.0)))
    (if (> pw 200) { (set! pw 200) })
    (Gfx.fill_rect 0 14 pw 2 215 178 92)
    (Gfx.fill_rect pw 14 (- 200 pw) 2 40 42 50)))

;; ---- 主流程 ----------------------------------------------------------------

(define tk_px 0) (define tk_py 0)
(define tk_j1 0.0) (define tk_j2 0.0)
(define tk_u 0.0) (define tk_v 0.0)
(define tk_dx 0.0) (define tk_dy 0.0) (define tk_dz 0.0) (define tk_dl 0.0)
(define tk_ii 0)

(define tick_fn
  (lambda ()
    (EX.check)
    (define t0 0.0)
    (if (< spp SPP_MAX)
        {
          (set! t0 (Diag.ms))
          (set! spp (+ spp 1))
          (set! tk_py 0)
          (while (< tk_py RH) {
            (set! tk_px 0)
            (while (< tk_px RW) {
              ;; 像素内抖动采样（抗锯齿；胶片颗粒的来源）
              (set! tk_j1 (Math.random))
              (set! tk_j2 (Math.random))
              (set! tk_u (* (- (* 2.0 (/ (+ tk_px tk_j1) RWF)) 1.0) U_MAX))
              (set! tk_v (* (- 1.0 (* 2.0 (/ (+ tk_py tk_j2) RHF))) V_MAX))
              (set! tk_dx (+ fwd_x (+ (* rgt_x tk_u) (* up_x tk_v))))
              (set! tk_dy (+ fwd_y (+ (* rgt_y tk_u) (* up_y tk_v))))
              (set! tk_dz (+ fwd_z (+ (* rgt_z tk_u) (* up_z tk_v))))
              (set! tk_dl (Math.sqrt (+ (+ (* tk_dx tk_dx) (* tk_dy tk_dy)) (* tk_dz tk_dz))))
              (set! tk_dx (/ tk_dx tk_dl))
              (set! tk_dy (/ tk_dy tk_dl))
              (set! tk_dz (/ tk_dz tk_dl))
              (trace_sample eye_x eye_y eye_z tk_dx tk_dy tk_dz)
              ;; 累积
              (set! tk_ii (* (+ (* tk_py RW) tk_px) 3))
              (set_item! acc tk_ii (+ (get_item acc tk_ii) res_r))
              (set_item! acc (+ tk_ii 1) (+ (get_item acc (+ tk_ii 1)) res_g))
              (set_item! acc (+ tk_ii 2) (+ (get_item acc (+ tk_ii 2)) res_b))
              (set! tk_px (+ tk_px 1))
            })
            (set! tk_py (+ tk_py 1))
          })
          (set! last_ms (Math.floor (- (Diag.ms) t0)))
          (redraw_all)
          (EX.draw_button)
          (Gfx.refresh)
        }
        { })))

;; ---- 启动 ------------------------------------------------------------------

(define init_i 0)
(while (< init_i 3600) { (push acc 0.0) (set! init_i (+ init_i 1)) })

(setup_camera)
(Gfx.clear)
(redraw_all)
(EX.draw_button)
(Gfx.refresh)

(define _timer 0)
(set! _timer (System.set_interval TICK_MS tick_fn))

;; ---- 退出：硬 A/ESC / 右上角「退出」按钮 → 宿主全局确认框 ------------------
;;  确认 → 宿主本帧末销毁本 App；留下 → 调 redraw_all 擦掉对话框残影（累积
;;  缓冲仍在，重绘的是同一显影进度）。
(EX.install_with redraw_all)
