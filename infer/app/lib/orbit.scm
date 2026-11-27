;; ============================================================================
;; 轨道库（Orb）—— 灵机（Animac）模块：由两行根数（TLE）推算卫星的地固位置
;;
;; 定位：**纯计算**模块。不读系统时钟、不联网；时刻一律由调用方以 (dn, sod)
;;   传入（dn：自 1970-01-01 的 UTC 整数日，sod：当日 UTC 秒数，可为小数），
;;   与 app/lib/ephem.scm 的约定一致。因此同一份库既可用于"此刻"，
;;   也可用于时间机器（暂停/回放）下的任意时刻反复求值。
;;
;; 算法：**开普勒根数推进 + J2 长期摄动**（够绘制精度，且不依赖任何外部库）
;;   1. 平均运动 n（圈/日，来自 TLE）→ 半长轴 a = (μ/n²)^(1/3)
;;   2. 平近点角 M = M0 + n·t + (ṅ/2)·t²（TLE 给出的就是一阶导数的 1/2）
;;   3. 牛顿迭代解开普勒方程 E − e·sinE = M → 真近点角 ν、地心距 r
;;   4. J2 长期摄动：升交点西退 RAAṄ（−6.07°/日）、近地点东进 ω̇（+7.32°/日）
;;      —— 这一项**不能省**：省掉它一天后纬度就差 3~4°（南北半球都错）
;;   5. 轨道面内 (r, u = ω+ν) 旋到地心惯性系（TEME ≈ ECI，本精度下等价）
;;   6. 用 GMST 把惯性系旋到地固系（ECEF）→ 地心纬度/经度
;;   不做 SGP4 的阻力项（B*）与短周期摄动：与 SGP4（WGS72 参考实现）逐点对照，
;;   TLE 历元 ±5 天内位置差 < 10 km（本 App 1 px ≈ 71 km，即 < 0.15 px）。
;;   **TLE 会老化**：误差按天累积（同一 TLE +30d 约 180 km、+90d 约 2700 km）。
;;   要更高精度就换新的两行根数（下面的 TLE_* 常量），其余代码不用动。
;;
;; 数值精度（32 位宿主只有约 19 位尾数，见 doc/AGENTS.md）：平近点角按
;;   "整日圈数取小数 + 当日秒数"两段累加，绝不让 10⁴~10⁶ 量级的弧度值进
;;   三角函数——否则 2e-6 的相对误差会被放大成几度的相位误差
;;   （与 ephem.scm 处理月球级数同一思路）。
;;
;; 用法（在 App 源文件中）：
;;   (import Orb "app/lib/orbit.scm")
;;   (Orb.set_time dn sod)        ;; 设定时刻：算 GMST + 该时刻的位置（写模块全局）
;;   (Orb.orb_lat Orb.orb_lon Orb.orb_r ...)   ;; 读结果
;;   (Orb.at tsec)                ;; 求"相对 TLE 历元 tsec 秒"处的位置（用该时刻的 GMST）
;;   (Orb.ring tsec)              ;; 同上但用**当前** GMST —— 画"此刻天空里的轨道环"用它
;;
;; 命名：本库标识符一律下划线分隔（不使用短划线）。
;; ============================================================================

(native Math)
(native String)

(define PI 3.141592653589793)
(define DEGR (/ PI 180.0))          ; 度 → 弧度
(define DEG 57.29577951308232)      ; 弧度 → 度

;; ---- 两行根数（TLE）：中国空间站（CSS / 天和核心舱，NORAD 48274）-----------
;;
;; 来源：https://celestrak.org/NORAD/elements/gp.php?CATNR=48274&FORMAT=TLE
;; 抓取：2026-11-27T03:06Z（celestrak 当时的 GP 数据）
;;
;; 1 48274U 21035A   26330.66136507  .00013885  00000+0  17195-3 0  9994
;; 2 48274  41.4711  54.5817 0009892 299.4712  60.5139 15.59758154318542
;;
;; 第 1 行第 19~32 列 = 历元（YYDDD.DDDDDDDD）：26 + 330.66136507
;;   ⇒ 2026-11-26T15:52:21.9Z ⇒ 自 1970-01-01 的第 20783 天 + 57141.942048 秒
(define SAT_ID 48274)
(define TLE_EPOCH_DN 20783)          ; 历元（UTC 整日，1970 起）
(define TLE_EPOCH_SOD 57141.942048)  ; 历元（当日秒）
(define TLE_EPOCH_Y 2026)            ; 历元的民用日期（只用于对外展示/诊断）
(define TLE_EPOCH_M 11)
(define TLE_EPOCH_D 26)
(define TLE_INC 41.4711)             ; 倾角（度）
(define TLE_RAAN 54.5817)            ; 升交点赤经（度）
(define TLE_ECC 0.0009892)           ; 偏心率
(define TLE_ARGP 299.4712)           ; 近地点幅角（度）
(define TLE_MA 60.5139)              ; 平近点角（度，历元时刻）
(define TLE_NREV 15.59758154)        ; 平均运动（圈/日）⇒ 周期 92.32 分
(define TLE_NDOT2 0.00013885)        ; 平均运动一阶导数的 1/2（圈/日²，TLE 原样）

;; ---- 地球与轨道常量 --------------------------------------------------------
(define RE_EQ 6378.137)              ; 地球赤道半径（km）—— 与 App 里"球半径 = RAD 像素"对应
(define FLAT 0.0033528106647474805)  ; WGS84 扁率 1/298.257223563（算大地高用）
(define E2 0.00669437999014)         ; 第一偏心率平方
(define MU 398600.4418)              ; 地球引力常数（km³/s²）
(define J2 0.00108262668)            ; 二阶带谐系数

;; 由 TLE 推出的常量（模块加载时算一次）
(define NM (* 2.0 (* PI (/ TLE_NREV 86400.0))))          ; 平均角速度（rad/s）
(define SMA (Math.pow (/ MU (* NM NM)) (/ 1.0 3.0)))     ; 半长轴（km）
(define PERIOD (/ 86400.0 TLE_NREV))                     ; 周期（s）
(define ALT0 (- SMA RE_EQ))                              ; 平均高度（km）
(define NDOT_RAD (* (* 2.0 (* PI TLE_NDOT2))          ; (ṅ/2) 的弧度形式（rad/s²）
                    (/ 1.0 (* 86400.0 86400.0))))
;; J2 长期项：fac = 1.5·J2·(Re/p)²·n
(define J2FAC (* 1.5 (* J2 (* (Math.pow (/ RE_EQ (* SMA (- 1.0 (* TLE_ECC TLE_ECC)))) 2.0) NM))))
(define RAAN_DOT (* (- 0.0 J2FAC) (Math.cos (* TLE_INC DEGR))))            ; rad/s
(define ARGP_DOT (* (* 0.5 J2FAC) (- (* 5.0 (* (Math.cos (* TLE_INC DEGR))
                                                       (Math.cos (* TLE_INC DEGR)))) 1.0)))

;; ---- 结果（模块全局；get 语义与 ephem.scm 一致）-----------------------------
(define orb_t 0.0)                   ; 最近一次 set_time 的时刻（相对历元的秒数）
(define orb_gmst 0.0)                ; 该时刻的 GMST（度）
(define orb_lat 0.0)                 ; 地心纬度（度）—— 3D 投影用
(define orb_lon 0.0)                 ; 地固经度（度，0..360）
(define orb_r 0.0)                   ; 地心距（km）
(define orb_latg 0.0)                ; 大地纬度（度）—— 显示用
(define orb_alt 0.0)                 ; 大地高（km）—— 显示用
(define orb_v 0.0)                   ; 轨道速度（km/s，活力公式）
(define orb_age 0.0)                 ; TLE 年龄（天，= 求值时刻 − 历元）
(define orb_x 0.0)                   ; 地固系直角坐标（km）—— 诊断用
(define orb_y 0.0)
(define orb_z 0.0)

;; ---- 角度工具 --------------------------------------------------------------
(define norm360
  (lambda (a)
    (define x a)
    (while (>= x 360.0) (set! x (- x 360.0)))
    (while (< x 0.0) (set! x (+ x 360.0)))
    x))

;; 格林尼治平恒星时（度）。与 app/lib/ephem.scm 的 (Ephem.set_gmst) 同一公式
;; （AA Ch.12 的低精度式），这里独立实现，好让本库不依赖星历库。
;; 精度处理：36000.77·tt = 36000·tt + 0.77·tt，而 36000·tt 的整圈部分对 mod 360
;; 无影响 ⇒ 只取 100·tt 的小数部分乘 360（否则 9684° 量级的中间量在 wasm32 的
;; 19 位尾数下会引入 0.06° 的误差，比整条轨道推算的误差还大）。
(define gmst_of
  (lambda (dn sod)
    (define dd 0.0)
    (define tt 0.0)
    (define t2 0.0)
    (define th0 0.0)
    (set! dd (- dn 10957))           ; J2000.0 起的天数
    (set! tt (/ (- dd 0.5) 36525.0))
    (set! t2 (* tt tt))
    (set! th0 (+ 100.46061837
                 (+ (* 360.0 (mod (* 100.0 tt) 1.0))
                    (+ (* 0.770053608 tt)
                       (+ (* 0.000387933 t2)
                          (- 0.0 (/ (* t2 tt) 38710000.0)))))))
    (set! th0 (+ th0 (/ (* 360.98564724 (/ sod 3600.0)) 24.0)))
    (norm360 th0)))

;; 开普勒方程 E − e·sinE = M（牛顿迭代 4 次；e ~ 1e-3 时 2 次就够，留余量）
(define kepler
  (lambda (m e)
    (define ea m)
    (define k 0)
    (while (< k 4) {
      (set! ea (- ea (/ (- (- ea (* e (Math.sin ea))) m)
                        (- 1.0 (* e (Math.cos ea))))))
      (set! k (+ k 1))
    })
    ea))

;; ---- 核心：相对 TLE 历元 t 秒处的惯性位置 → 地固纬经度 ----------------------
;;
;; gm 是**要把惯性系旋到地固系时用的 GMST（度）**，由两个入口各给一个：
;;   · (Orb.at t)   —— 用 t 时刻的 GMST（求"某时刻卫星在哪里"）
;;   · (Orb.ring t) —— 用**当前**时刻（orb_t）的 GMST（画"此刻天空里那条轨道"）
;; 两者不能混：轨道环是**惯性空间里的一个圆**，画在"此刻"的地固系里 ⇒ 环上每一点
;; 都要用同一个 GMST 去旋。若逐点用各自时刻的 GMST，画出来的就不是轨道，而是卫星
;; 一个周期内的**星下点轨迹**（每圈西移 23°，接缝处会出现一个折角）。
;;
;; 平近点角分两段：整日的圈数先对 1 取余（15.59 圈/日里那 15 整圈不进相位），
;; 再加当日不足一日的部分与 ṅ/2·t² 项 ⇒ 中间量都在 10¹~10³ 量级，19 位尾数够用。
(define at_g
  (lambda (t gm)
    (define dd 0.0)
    (define rem 0.0)
    (define mrev 0.0)
    (define m 0.0)
    (define ea 0.0)
    (define nu 0.0)
    (define rr 0.0)
    (define ra 0.0)
    (define ap 0.0)
    (define u 0.0)
    (define x 0.0)
    (define y 0.0)
    (define z 0.0)
    (define rxy 0.0)
    (define ci 0.0)
    (define si 0.0)
    (define cu 0.0)
    (define su 0.0)
    (define cO 0.0)
    (define sO 0.0)
    (define sinlat 0.0)
    (set! dd (Math.floor (/ t 86400.0)))
    (set! rem (- t (* dd 86400.0)))
    (set! mrev (+ (/ TLE_MA 360.0)
                  (+ (mod (* TLE_NREV dd) 1.0)
                     (+ (* TLE_NREV (/ rem 86400.0))
                        (* TLE_NDOT2 (* dd dd))))))
    (set! m (* 360.0 (mod mrev 1.0)))          ; 圈 → 度（mod 保留被除数符号，norm360 兜底）
    (set! m (* (norm360 m) DEGR))
    (set! ea (kepler m TLE_ECC))
    (set! nu (Math.atan2 (* (Math.sqrt (- 1.0 (* TLE_ECC TLE_ECC))) (Math.sin ea))
                         (- (Math.cos ea) TLE_ECC)))
    (set! rr (* SMA (- 1.0 (* TLE_ECC (Math.cos ea)))))
    (set! ra (+ (* TLE_RAAN DEGR) (* RAAN_DOT t)))
    (set! ap (+ (* TLE_ARGP DEGR) (* ARGP_DOT t)))
    (set! u (+ ap nu))
    (set! ci (Math.cos (* TLE_INC DEGR)))
    (set! si (Math.sin (* TLE_INC DEGR)))
    (set! cu (Math.cos u))
    (set! su (Math.sin u))
    (set! cO (Math.cos ra))
    (set! sO (Math.sin ra))
    (set! x (* rr (- (* cO cu) (* sO (* su ci)))))
    (set! y (* rr (+ (* sO cu) (* cO (* su ci)))))
    (set! z (* rr (* su si)))
    (set! rxy (Math.sqrt (+ (* x x) (* y y))))
    (set! orb_lat (* DEG (Math.atan2 z rxy)))
    (set! orb_lon (norm360 (- (* DEG (Math.atan2 y x)) gm)))
    (set! orb_r rr)
    (set! orb_x x) (set! orb_y y) (set! orb_z z)
    ;; 大地纬度/高度（WGS84）：tanφ_gd = tanφ_c / (1−e²)，高程按当地椭球面半径扣
    (set! orb_latg (* DEG (Math.atan2 z (* rxy (- 1.0 E2)))))
    (set! sinlat (Math.sin (* orb_latg DEGR)))
    (set! orb_alt (- rr (* RE_EQ (- 1.0 (* FLAT (* sinlat sinlat))))))
    ;; 活力公式：v = sqrt(μ(2/r − 1/a))
    (set! orb_v (Math.sqrt (* MU (- (/ 2.0 rr) (/ 1.0 SMA)))))))

;; 某时刻的位置：GMST 也取该时刻的
(define at
  (lambda (t)
    (at_g t (norm360 (+ orb_gmst (* 360.98564724 (/ (- t orb_t) 86400.0)))))))

;; 画轨道环用：惯性位置随时推进，但一律用**当前**时刻的 GMST 旋到地固系
(define ring
  (lambda (t)
    (at_g t orb_gmst)))

;; ---- 对外接口 --------------------------------------------------------------
;; 设定时刻（UTC 整日 dn + 当日秒 sod），求该时刻的位置并写模块全局。
(define set_time
  (lambda (dn sod)
    (set! orb_t (+ (* (- dn TLE_EPOCH_DN) 86400.0) (- sod TLE_EPOCH_SOD)))
    (set! orb_age (/ orb_t 86400.0))
    (set! orb_gmst (gmst_of dn sod))
    (at orb_t)))

;; ---- 展示用的派生量 --------------------------------------------------------
(define get_alt0 (lambda () ALT0))
(define get_period (lambda () PERIOD))
(define get_inc (lambda () TLE_INC))
(define get_tle_age (lambda () orb_age))
;; "YYYY-MM-DD"（TLE 历元的民用日期，由常量拼出，不做历法换算）
(define tle_date_str
  (lambda ()
    (define s "")
    (set! s (String.concat (String.atom_to_string TLE_EPOCH_Y) "-"))
    (set! s (String.concat s (if (< TLE_EPOCH_M 10) "0" "")))
    (set! s (String.concat s (String.atom_to_string TLE_EPOCH_M)))
    (set! s (String.concat s "-"))
    (set! s (String.concat s (if (< TLE_EPOCH_D 10) "0" "")))
    (String.concat s (String.atom_to_string TLE_EPOCH_D))))
