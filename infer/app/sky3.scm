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
;;   一阶步进 NV=36/NL=1；二阶视线采样 V2=3；二阶步进 NV2=2/NL2=2；9 个环境方向
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
;; 性能设计（宿主每帧一个时间片，约 25 万条 VM 指令）：
;;   该模型单采样点约 4.8 万条 VM 指令（约为 sky.scm 的 240 倍），故：
;;   - 采样块 5×5（64×48=3072 采样点，圆内约 1770 个），远粗于 sky/sky2 的 2×2；
;;   - 每个 step 只推进 SAMPLES_PER_STEP=5 个**圆内**采样点（约 24 万指令，单步
;;     不跨时间片，避免 interval 回调重入）；圆外采样几乎零成本，可批量略过；
;;     每画完一个采样行推屏一次（渐进呈现）；
;;   - 一帧天空约 355 步：原生测试台 ≈6 秒、浏览器 WASM ≈13 秒（实测）；
;;     渲染完成后显示实时太阳高度/方位与时刻，并每隔 HOLD_MS 检查一次太阳是否
;;     已移动超过 RE_RENDER_DEG，是则按新太阳位置重绘。
;;   - 帧级预计算：太阳三向量、太阳高度（供 global_gain）。
;;
;; 依赖：Gfx（绘图）、Math、System（定时器）、Time（取时）、共享星历模块 Ephem。
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

(define SAMPLES_PER_STEP 5)         ; 每步推进的采样点数（本模型极重）
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

;; ---- 相位函数 --------------------------------------------------------------
;; phaseR(μ) = 3/(16π)·(1+μ²)
;; phaseM(μ) = 3/(8π)·(1-g²)(1+μ²) / ((2+g²)·(1+g²-2gμ)^1.5)，g=0.76

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
    (set! sun_x (* ca (Math.sin azr)))
    (set! sun_y (* ca (Math.cos azr)))
    (set! sun_z (Math.sin elr))
  }))

;; ---- 射线-球面求交：|o+t·d|²=r² ⇒ t²+2(o·d)t+(|o|²-r²)=0 ------------------
;; 有交点返回 1 并把两根写入 rt0/rt1，无交点返回 0。

(define rt0 0.0)
(define rt1 0.0)
(define rsi
  (lambda (ox oy oz dx dy dz rad) {
    (define b (* 2.0 (+ (* ox dx) (+ (* oy dy) (* oz dz)))))
    (define c (- (+ (* ox ox) (+ (* oy oy) (* oz oz))) (* rad rad)))
    (define disc (- (* b b) (* 4.0 c)))
    (if (< disc 0.0)
        { (set! rt0 0.0) (set! rt1 0.0) 0 }
        { (define sq (Math.sqrt disc))
          (set! rt0 (* 0.5 (- (- 0.0 b) sq)))
          (set! rt1 (* 0.5 (+ (- 0.0 b) sq)))
          1 })
  }))

;; ---- 路径积分（Nishita 公式 4/5）：沿 o+t·d 求 Rayleigh/Mie 加权光学深度 --
;; nV 个主采样点，每点向太阳方向再取 nL 个采样点做地球遮蔽检测与透射率累积。
;; 结果写入 m_r0..m_r2（Rayleigh 三通道）与 m_m0..m_m2（Mie 三通道）。

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
      (set! h (- (Math.sqrt (+ (* wx wx) (+ (* wy wy) (* wz wz)))) Re))
      (set! dR (* (Math.exp (- 0.0 (/ h Hr))) seg))
      (set! dM (* (Math.exp (- 0.0 (/ h Hm))) seg))
      (set! odR (+ odR dR))
      (set! odM (+ odM dM))
      (if (== (rsi wx wy wz sun_x sun_y sun_z Ra) 1)
          { (if (> rt1 0.0)
                {
                  (set! lSeg (/ rt1 nL))
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
    (define mu_a 0.0) (define mu_s 0.0)
    (define p_ra 0.0) (define p_ma 0.0) (define p_rs 0.0) (define p_ms 0.0)
    (define sg0 0.0) (define sg1 0.0) (define sg2 0.0)
    (define mu1 0.0) (define dom_g 0.0)
    ;; 观测点在大气内部：入射根 t0<0，故积分起点 tMin=0；终点为大气出射根
    (define hit (rsi 0.0 0.0 Re1 vx vy vz Ra))
    (set! tmin 0.0)
    (set! tmax rt1)
    ;; 地球遮挡（朝天顶的视线不会命中，保留与原实现一致的判据）
    (if (== (rsi 0.0 0.0 Re1 vx vy vz Re) 1)
        { (if (> rt0 1.0) { (if (< rt0 tmax) { (set! tmax rt0) }) }) })
    (if (<= tmax tmin)
        { (set! out_r 0.0) (set! out_g 0.0) (set! out_b 0.0) }
        {
          ;; ---- 一阶单次散射 ----
          (set! mu1 (+ (* vx sun_x) (+ (* vy sun_y) (* vz sun_z))))
          (define p_r1 (phase_r mu1))
          (define p_m1 (phase_m mu1))
          (march 0.0 0.0 Re1 vx vy vz tmin tmax 36 1)
          (set! c0 (* SUNF (+ (* (* m_r0 bR0) p_r1) (* (* m_m0 bM) p_m1))))
          (set! c1 (* SUNF (+ (* (* m_r1 bR1) p_r1) (* (* m_m1 bM) p_m1))))
          (set! c2 (* SUNF (+ (* (* m_r2 bR2) p_r1) (* (* m_m2 bM) p_m1))))
          ;; ---- 二阶多次散射：视线路径 3 个采样点 × 上半球 9 个方向 ----
          (set! dom_g (* (* 2.0 (/ PI 9.0))
                         (+ (* 14.0 (Math.exp (- 0.0 (* (/ sun_alt_deg 10.0) (/ sun_alt_deg 10.0))))) 2.0)))
          (set! seg_v (/ (- tmax tmin) 3.0))
          (set! od_r 0.0) (set! od_m 0.0) (set! i 0)
          (while (< i 3) {
            (set! t (+ tmin (* (+ i 0.5) seg_v)))
            (set! wx (* vx t))
            (set! wy (* vy t))
            (set! wz (+ Re1 (* vz t)))
            (set! h (- (Math.sqrt (+ (* wx wx) (+ (* wy wy) (* wz wz)))) Re))
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
                    (if (== (rsi wx wy wz adx ady adz Ra) 1)
                        { (if (> rt1 0.0)
                              {
                                (march wx wy wz adx ady adz 0.0 rt1 2 2)
                                (set! mu_a (+ (* adx sun_x) (+ (* ady sun_y) (* adz sun_z))))
                                (set! p_ra (phase_r mu_a))
                                (set! p_ma (phase_m mu_a))
                                (define la0 (* SUNF (+ (* (* m_r0 bR0) p_ra) (* (* m_m0 bM) p_ma))))
                                (define la1 (* SUNF (+ (* (* m_r1 bR1) p_ra) (* (* m_m1 bM) p_ma))))
                                (define la2 (* SUNF (+ (* (* m_r2 bR2) p_ra) (* (* m_m2 bM) p_ma))))
                                ;; 相位函数方向约定：入射光子 −ω、出射光子 −d ⇒ μ=dot(ω,d)
                                (set! mu_s (+ (* adx vx) (+ (* ady vy) (* adz vz))))
                                (set! p_rs (phase_r mu_s))
                                (set! p_ms (phase_m mu_s))
                                (set! sg0 (+ (* (* bR0 p_rs) dR2) (* (* bM p_ms) dM2)))
                                (set! sg1 (+ (* (* bR1 p_rs) dR2) (* (* bM p_ms) dM2)))
                                (set! sg2 (+ (* (* bR2 p_rs) dR2) (* (* bM p_ms) dM2)))
                                (set! c0 (+ c0 (* (* (* la0 sg0) tv0) dom_g)))
                                (set! c1 (+ c1 (* (* (* la1 sg1) tv1) dom_g)))
                                (set! c2 (+ c2 (* (* (* la2 sg2) tv2) dom_g)))
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
          (set! out_r (* 255.0 (hdr_map c0)))
          (set! out_g (* 255.0 (hdr_map c1)))
          (set! out_b (* 255.0 (hdr_map c2)))
        })
  }))
;; ---- 圆界采样块的逐像素落点（块骑跨天空圆界时按像素精确判定）--------------

(define edge_block
  (lambda (px py fr fg fb) {
    (define kx 0) (define ky 0)
    (define cxp 0.0) (define cyp 0.0) (define eddx 0.0) (define eddy 0.0)
    (set! ky 0)
    (while (< ky SS) {
      (set! cyp (+ py (+ ky 0.5)))
      (set! eddy (- cyp CY))
      (set! kx 0)
      (while (< kx SS) {
        (set! cxp (+ px (+ kx 0.5)))
        (set! eddx (- cxp CX))
        (if (<= (+ (* eddx eddx) (* eddy eddy)) R_SCR2)
            { (Gfx.set_pixel (+ px kx) (+ py ky) fr fg fb) })
        (set! kx (+ kx 1))
      })
      (set! ky (+ ky 1))
    })
  }))

;; ---- 单采样点：视线方向 → 散射颜色 → 绘制 SS×SS 块 ------------------------
;; 参数：px/py 采样块左上角像素、dx/dy 采样块中心相对屏幕中心的偏移。
;; 视线向量按「抬头看天」视角：屏幕上方=北、屏幕左侧=东（见文件头）。

(define shade
  (lambda (px py dx dy) {
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
    ;; 绘制：整块最远像素仍在圆内 → 直接填块；骑跨圆界 → 逐像素判定
    (define adx (if (< dx 0.0) (- 0.0 dx) dx))
    (define ady (if (< dy 0.0) (- 0.0 dy) dy))
    (if (<= (+ (* (+ adx HALF_S) (+ adx HALF_S)) (* (+ ady HALF_S) (+ ady HALF_S))) R_SCR2)
        { (Gfx.fill_rect px py SS SS out_r out_g out_b) }
        { (edge_block px py out_r out_g out_b) })
  }))

;; ---- 渲染状态：逐采样点推进（每画完一个采样行推屏一次）--------------------

(define cur_x 0)
(define cur_y 0)
(define hold_until 0)

(define render_chunk
  (lambda () {
    (define painted 0)
    (while (and (< painted SAMPLES_PER_STEP) (< cur_y ROWS_S)) {
      (define px (* cur_x SS))
      (define py (* cur_y SS))
      (define dx (- (+ px HALF_S) CX))
      (define dy (- (+ py HALF_S) CY))
      (if (<= (+ (* dx dx) (* dy dy)) R_SCR2)
          { (shade px py dx dy) (set! painted (+ painted 1)) })
      (set! cur_x (+ cur_x 1))
      (if (>= cur_x COLS_S)
          { (set! cur_x 0) (set! cur_y (+ cur_y 1)) (Gfx.refresh) })
    })
    (if (>= cur_y ROWS_S)
        {
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
;;   · 与时钟块相交的采样块重新着色：块中心在圆内 → 用当前帧参数重新 shade
;;     （骑跨圆界的块由 shade 自带的判定转 edge_block 逐像素处理）；块中心在
;;     圆外 → 置黑，与逐采样点渲染时“圆外一律为黑”一致；
;;   · 随后把时间文字直接画在天空上（alpha 字模按覆盖率与背景混合，边缘自然）。
;; 采样块左上角为 (sx·SS, sy·SS)、覆盖 SS×SS、块中心偏移为 (+ px HALF_S − CX)，
;; 故列/行的相交范围由 hud_clock_* 折算。

(define clock_bg_sx0 (Math.floor (/ (- hud_clock_x 1) SS)))
(define clock_bg_sx1 (Math.floor (/ (- (+ hud_clock_x hud_clock_w) 1) SS)))
(define clock_bg_sy0 (Math.floor (/ (- hud_clock_y 1) SS)))
(define clock_bg_sy1 (Math.floor (/ (- (+ hud_clock_y hud_clock_h) 1) SS)))

(define repaint_clock_bg
  (lambda () {
    (define sy clock_bg_sy0)
    (while (<= sy clock_bg_sy1) {
      (define py (* sy SS))
      (define dy (- (+ py HALF_S) CY))
      (define sx clock_bg_sx0)
      (while (<= sx clock_bg_sx1) {
        (define px (* sx SS))
        (define dx (- (+ px HALF_S) CX))
        (if (<= (+ (* dx dx) (* dy dy)) R_SCR2)
            { (shade px py dx dy) }
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
    (set! cur_x 0)
    (set! cur_y 0)
    (Gfx.clear)
  }))

(define step
  (lambda () {
    (if (< cur_y ROWS_S)
        { (render_chunk) }
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
