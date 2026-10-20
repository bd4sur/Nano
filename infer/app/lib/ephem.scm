;; ============================================================================
;; 共享星历库（Ephem）—— 灵机（Animac）模块，供各「灵机应用程序」import 复用
;;
;; 定位：**纯计算**模块。不读系统时钟、不依赖任何 native 图形/IO 库，
;;       时间一律由调用方以 (dn, sod) 传入（dn：自 1970-01-01 的 UTC 整数日，
;;       sod：当日 UTC 秒数，可为小数）。因此同一份库既可供「星历」App 显示
;;       当前天象，也可供「农历」App 对任意时刻反复求值（如牛顿迭代求朔）。
;;
;; 算法来源：ephemeris.c（J. Meeus《Astronomical Algorithms》第 2 版：
;;   儒略日 Ch.7、恒星时 Ch.12、太阳位置 Ch.25、月球位置 Ch.47、坐标变换 Ch.13），
;;   以及 vsop87c_milli.c 的地球日心直角坐标级数（太阳黄经的校对值）。
;;
;; 用法（在 App 源文件中）：
;;   (import Ephem "app/lib/ephem.scm")
;;   (Ephem.set_time dn sod)          ;; 设定时间基（后续函数都以此为「当前时刻」）
;;   (Ephem.sun_lon)                  ;; 太阳视黄经（度）
;;   (Ephem.moon_lon)                 ;; 月球黄经（度）
;;   (Ephem.set_gmst)                 ;; 计算 GMST（地平坐标需要；每次 set_time 后调用）
;;   (Ephem.sun_altaz) (Ephem.get_az) ;; 写模块全局，经 get_* 读取（Animac 无多值返回）
;;
;; 数值实现要点（32 位宿主精度陷阱，详见各函数注释）：
;;   vendored VM 以「float 位模式右移 5 位」存放浮点，wasm32 上有效尾数约 19 位；
;;   2.4e6 量级的儒略日粒度可达 8 天。因此本库：
;;     · 时间以 (整数日 dn, 当日秒 sod) 表示，不出现大数浮点；
;;     · 儒略世纪 H = (jd-2451545)/36525 拆成 dn/36525 + f/36525 两项；
;;     · 月球级数辐角以 mulmod360 消去大数中间量。
;;
;; 命名约定：本库及本任务新增代码的标识符一律使用下划线分隔（不使用短划线）。
;; ============================================================================

(native Math)   ;; 本模块只依赖 Math 库（三角函数/取整/幂），不依赖 Gfx/时钟/IO

(define PI 3.141592653589793)
(define TZ 8.0)                     ; 时区（小时，东正；北京时间）
(define TZ_SEC 28800)               ; 时区秒数（整数运算用）
(define LON 119.0)                  ; 观测点经度（东经，度）
(define LAT 32.0)                   ; 观测点纬度（北纬，度）
(define EPS_DEG 23.433333333333334) ; 黄赤交角 23°26′（与 ephemeris.c 一致）
(define J2000_EPOCH_DAY 10957)      ; 1970-01-01 → 2000-01-01 的日数

;; ---- 角度工具 --------------------------------------------------------------

(define to_rad (lambda (d) (* d (/ PI 180.0))))
(define to_deg (lambda (r) (* r (/ 180.0 PI))))

(define norm360
  (lambda (a) {
    (define v (mod a 360.0))
    (if (< v 0.0) { (set! v (+ v 360.0)) })
    v
  }))

;; atan2 / asin / acos（Math 库未提供；用 Math.atan 构造，象限处理与 C 标准库一致）
(define atan2
  (lambda (y x) {
    (define r 0.0)
    (if (> x 0.0)
        { (set! r (Math.atan (/ y x))) }
        { (if (< x 0.0)
              { (if (>= y 0.0)
                    { (set! r (+ (Math.atan (/ y x)) PI)) }
                    { (set! r (- (Math.atan (/ y x)) PI)) }) }
              { (if (> y 0.0)
                    { (set! r (/ PI 2.0)) }
                    { (if (< y 0.0) { (set! r (- 0.0 (/ PI 2.0))) } { (set! r 0.0) }) }) }) })
    r
  }))

(define asin
  (lambda (x) {
    (define v x)
    (if (> v 1.0) { (set! v 1.0) })
    (if (< v -1.0) { (set! v -1.0) })
    (atan2 v (Math.sqrt (- 1.0 (* v v))))
  }))

(define acos (lambda (x) (- (/ PI 2.0) (asin x))))

;; ---- 时间基准（整数）与儒略世纪数 -----------------------------------------
;; jd - 2451545 = D + f，其中 D = utc_days - 10957（整数日）、f = sod/86400 - 0.5。
;; 儒略世纪 H = D/36525 + f/36525（拆两项，避免大数浮点）。

(define e_dn 0)                     ; D（整数）
(define e_sod 0)                    ; UT 当日秒（整数）
(define e_hcent 0.0)                ; H（儒略世纪）
(define e_gmst 0.0)                 ; 格林尼治平恒星时（度）

(define set_time
  (lambda (udays usod) {
    (set! e_dn (- udays J2000_EPOCH_DAY))
    (set! e_sod usod)
    (define dn e_dn)
    (define fr (- (/ usod 86400.0) 0.5))
    (set! e_hcent (+ (/ dn 36525.0) (/ fr 36525.0)))
    (set! e_fday (- (/ usod 86400.0) 0.5))   ;; f = sod/86400 - 0.5（与 H 的定义一致）
  }))

;; 格林尼治平恒星时（ephemeris.c: greenwich_sidereal_time 的等价写法）
;;   T0 = (jd0 - 2451545)/36525 = (D - 0.5)/36525
;;   ut_hours = (jd - jd0)*24 = sod/3600
(define set_gmst
  (lambda () {
    (define tt (/ (- e_dn 0.5) 36525.0))
    (define t2 (* tt tt))
    (define th0 (+ 100.46061837
                   (+ (* 36000.770053608 tt)
                      (+ (* 0.000387933 t2)
                         (- 0.0 (/ (* t2 tt) 38710000.0))))))
    (define ut_h (/ e_sod 3600.0))
    (set! e_gmst (norm360 (+ (norm360 th0) (/ (* 360.98564724 ut_h) 24.0))))
  }))
;; ---- 月球位置级数（AA Ch.47 表 47.A / 47.B；与 ephemeris.c 的十张表逐项一致）----
(define a_md '(
  0 2 2 0 0 0 2 2 2 2
  0 1 0 2 0 0 4 0 4 2
  2 1 1 2 2 4 2 0 2 2
  1 2 0 0 2 2 2 4 0 3
  2 4 0 2 2 2 4 0 4 1
  2 0 1 3 4 2 0 1 2 2))
(define a_mm '(
  0 0 0 0 1 0 0 -1 0 -1
  1 0 1 0 0 0 0 0 0 1
  1 0 1 -1 0 0 0 1 0 -1
  0 -2 1 2 -2 0 0 -1 0 0
  1 -1 2 2 1 -1 0 0 -1 0
  1 0 1 0 0 -1 2 1 0 0))
(define a_mm2 '(
  1 -1 0 2 0 0 -2 -1 1 0
  -1 0 1 0 1 1 -1 3 -2 -1
  0 -1 0 1 2 0 -3 -2 -1 -2
  1 0 2 0 -1 1 0 -1 2 -1
  1 -2 -1 -1 -2 0 1 4 0 -2
  0 2 1 -2 -3 2 1 -1 3 -1))
(define a_mf '(
  0 0 0 0 0 2 0 0 0 0
  0 0 0 -2 2 -2 0 0 0 0
  0 0 0 0 0 0 0 0 2 0
  0 0 0 0 0 -2 2 0 2 0
  0 0 0 0 0 -2 0 0 0 0
  -2 -2 0 0 0 0 0 0 0 -2))
(define a_cs '(
  6288774 1274027 658314 213618 -185116 -114332 58793 57066 53322 45758
  -40923 -34720 -30383 15327 -12528 10980 10675 10034 8548 -7888
  -6766 -5163 4987 4036 3994 3861 3665 -2689 -2602 2390
  -2348 2236 -2120 -2069 2048 -1773 -1595 1215 -1110 -892
  -810 759 -713 -700 691 596 549 537 520 -487
  -399 -381 351 -340 330 327 -323 299 294 0))
(define a_cc '(
  -20905355 -3699111 -2955968 -569925 48888 -3149 246158 -152138 -170733 -204586
  -129620 108743 104755 10321 0 79661 -34782 -23210 -21636 24208
  30824 -8379 -16675 -12831 -10445 -11650 14403 -7003 0 10056
  6322 -9884 5751 0 -4950 4130 0 -3958 0 3258
  2616 -1897 -2117 2354 0 0 -1423 -1117 -1571 -1739
  0 -4421 0 0 0 0 1165 0 0 8752))
(define b_md '(
  0 0 0 2 2 2 2 0 2 0
  2 2 2 2 2 2 2 0 4 0
  0 0 1 0 0 0 1 0 4 4
  0 4 2 2 2 2 0 2 2 2
  2 4 2 2 0 2 1 1 0 2
  1 2 0 4 4 1 4 1 4 2))
(define b_mm '(
  0 0 0 0 0 0 0 0 0 0
  -1 0 0 1 -1 -1 -1 1 0 1
  0 1 0 1 1 1 0 0 0 0
  0 0 0 0 -1 0 0 0 0 1
  1 0 -1 -2 0 1 1 1 1 1
  0 -1 1 0 -1 0 0 0 -1 -2))
(define b_mm2 '(
  0 1 1 0 -1 -1 0 2 1 2
  0 -2 1 0 -1 0 -1 -1 -1 0
  0 -1 0 1 1 0 0 3 0 -1
  1 -2 0 2 1 -2 3 2 -3 -1
  0 0 1 0 1 1 0 0 -2 -1
  1 -2 2 -2 -1 1 1 -1 0 0))
(define b_mf '(
  1 1 -1 -1 1 -1 1 1 -1 -1
  -1 -1 1 -1 1 1 -1 -1 -1 1
  3 1 1 1 -1 -1 -1 1 -1 1
  -3 1 -3 -1 -1 1 -1 1 -1 1
  1 1 1 -1 3 -1 -1 1 -1 -1
  1 -1 1 -1 -1 -1 -1 -1 -1 1))
(define b_cs '(
  5128122 280602 277693 173237 55413 46271 32573 17198 9266 8822
  8216 4324 4200 -3359 2463 2211 2065 -1870 1828 -1794
  -1749 -1565 -1492 -1475 -1410 -1344 -1335 1107 1021 833
  777 671 607 596 491 -451 439 422 421 -366
  -351 331 315 302 -283 -229 223 223 -220 -220
  -185 181 -177 176 166 -164 132 -119 115 107))

;; ---- VSOP87C(milli) 地球日心直角坐标级数（与 vsop87c_milli.c 逐项一致）----
;; 每项 (A B C) 表示 A·cos(B + C·t)，t 为自 J2000 起的儒略千年数；
;; milli 版中地球 z 级数被截断为 0（原文件 earth_z 直接 return 0），故不设 e_z*。
(define e_x0 '(
  (0.99986069925 1.75347045757 6283.31966747490)
  (0.02506324281 4.93819429098 0.24381748350)
  (0.00835274807 1.71033525539 12566.39551746630)
  (0.00010466796 1.66721984219 18849.47136745770)
  (0.00003490608 4.44373803231 6282.83203250789)
  (0.00003110838 0.66875189331 83997.09113559539)
  (0.00002561408 0.58588607490 529.93478257810)
  (0.00002142365 1.09204474884 1577.09972496430)
  (0.00001709161 0.49540863237 6279.79654912590)
  (0.00001707934 6.15314019418 6286.84278582391)
  (0.00001442753 3.47210398336 2353.10997125530)
  (0.00001113027 3.69621650479 5223.45010231870)))
(define e_x1 '(
  (0.00154550744 0.64605836878 0.24381748350)
  (0.00051503383 6.00263199393 12566.39551746630)
  (0.00001290763 5.95941652859 18849.47136745770)))
(define e_x2 '(
  (0.00052911498 3.32403354915 6283.31966747490)
  (0.00006074441 2.05485843872 0.24381748350)
  (0.00002406871 4.23086027149 12566.39551746630)))
(define e_y0 '(
  (0.99986069925 0.18267413078 6283.31966747490)
  (0.02506324281 3.36739796418 0.24381748350)
  (0.00835274807 0.13953892859 12566.39551746630)
  (0.00010466796 0.09642351540 18849.47136745770)
  (0.00003490608 6.01453435910 6282.83203250789)
  (0.00003110838 5.38114087369 83997.09113559539)
  (0.00002561408 5.29827505528 529.93478257810)
  (0.00002142365 2.66284107563 1577.09972496430)
  (0.00001709161 5.20779761275 6279.79654912590)
  (0.00001707934 4.58234386738 6286.84278582391)
  (0.00001442753 1.90130765657 2353.10997125530)
  (0.00001113027 5.26701283158 5223.45010231870)))
(define e_y1 '(
  (0.00154550744 5.35844734917 0.24381748350)
  (0.00051503383 4.43183566713 12566.39551746630)
  (0.00001290763 4.38862020180 18849.47136745770)))
(define e_y2 '(
  (0.00052911498 1.75323722235 6283.31966747490)
  (0.00006074441 0.48406211192 0.24381748350)
  (0.00002406871 2.66006394470 12566.39551746630)))

;; ---- 大辐角三角函数的精度处理 ---------------------------------------------
;; 月球级数的辐角速率高达 445267°/世纪，若先合成 H=(jd_2451545)/36525 再相乘，
;; 32 位宿主 19 位尾数会把 H 的 5e-7（≈26 分钟）误差放大到 0.2° 量级。这里改为
;;   rate·H = q·D + q·f ，  q = rate/36525，D 为整数日、f = sod/86400
;; 且 q·D 用「逐位取模 360」的累加计算，中间量始终 < 4000，精度回到 1e-3 量级。

(define e_fday 0.0)                 ;; 日小数 f = sod/86400 - 0.5（set_time 中更新）

(define mulmod360
  (lambda (q n) {
    ;; 返回 (q·n) mod 360（n 为整数）。32 位宿主 float 仅约 19 位尾数，直接做
    ;; q·n 或逐位 Horner 都会引入 0.2° 量级误差；这里把 q 拆成
    ;;   q = qi（整数部分） + q3/1000（千分位） + res（残差 < 5e-4）
    ;; 前两项用整数乘法（量级 < 2^27，VM 整数精确），残差项再乘 n（量级 ≤ 5）。
    (define qi (Math.floor q))
    (define qf (- q qi))
    (define q3 (Math.floor (+ (* qf 1000.0) 0.5)))
    (define res (- qf (/ q3 1000.0)))
    (define a (mod (* qi n) 360.0))
    (define b (/ (mod (* q3 n) 360000.0) 1000.0))
    (define cc (* res n))
    (norm360 (+ (+ a b) cc))
  }))

;; 辐角「线性部分」：offset + rate·H 的 mod 360 值（q = rate/36525）
(define arg_lin
  (lambda (offset q) (norm360 (+ offset (+ (mulmod360 q e_dn) (* q e_fday))))))

;; ---- 月球位置（ephemeris.c: calculate_lunar_ecliptic_coordinates，AA Ch.47）--

(define m_lam 0.0)                  ; 黄经 λ（度）
(define m_bet 0.0)                  ; 黄纬 β（度）
(define m_dist 0.0)                 ; 地心距离（km）

(define lunar_ecliptic
  (lambda () {
    (define h e_hcent)
    (define h2 (* h h))
    (define h3 (* h2 h))
    (define h4 (* h3 h))
    ;; 月球平黄经、月日平均距角、太阳/月球平近点角、纬度参数、三个修正角
    (define ll (norm360 (+ (arg_lin 218.3164477 13.176396474585)
                           (+ (* -0.0015786 h2)
                              (+ (/ h3 538841.0) (- 0.0 (/ h4 65194000.0)))))))
    (define d (norm360 (+ (arg_lin 297.8501921 12.190749114398)
                          (+ (* -0.0018819 h2)
                             (+ (/ h3 545868.0) (- 0.0 (/ h4 113065000.0)))))))
    (define mm (norm360 (+ (arg_lin 357.5291092 0.985600281749)
                           (+ (* -0.0001536 h2) (/ h3 24490000.0)))))
    (define m2 (norm360 (+ (arg_lin 134.9633964 13.064992950185)
                           (+ (* 0.0087414 h2)
                              (+ (/ h3 69699.0) (- 0.0 (/ h4 14712000.0)))))))
    (define f (norm360 (+ (arg_lin 93.2720950 13.229350240200)
                          (+ (* -0.0036539 h2)
                             (+ (- 0.0 (/ h3 3526000.0)) (/ h4 863310000.0))))))
    (define a1 (arg_lin 119.75 0.003609828884))
    (define a2 (arg_lin 53.09 13.121541136208))
    (define a3 (arg_lin 313.45 13.176358220397))
    (define e (- 1.0 (+ (* 0.002516 h) (* 0.0000074 h2))))
    (define e2 (* e e))
    (define s_l 0.0)
    (define s_r 0.0)
    (define s_b 0.0)
    (define i 0)
    (define dm 0.0)
    (define dmm 0.0)
    (define dm2 0.0)
    (define df 0.0)
    (define ee 1.0)
    (define arg 0.0)
    (define sv 0.0)
    (while (< i 60) {
      ;; 黄经/距离求和（表 47.A）
      (set! dm (get_item a_md i))
      (set! dmm (get_item a_mm i))
      (set! dm2 (get_item a_mm2 i))
      (set! df (get_item a_mf i))
      (set! ee (if (or (== dmm 2) (== dmm -2)) e2 (if (or (== dmm 1) (== dmm -1)) e 1.0)))
      (set! arg (to_rad (+ (* dm d) (+ (* dmm mm) (+ (* dm2 m2) (* df f))))))
      (set! sv (Math.sin arg))
      (set! s_l (+ s_l (* (* (get_item a_cs i) ee) sv)))
      (set! s_r (+ s_r (* (* (get_item a_cc i) ee) sv)))
      ;; 黄纬求和（表 47.B）
      (set! dm (get_item b_md i))
      (set! dmm (get_item b_mm i))
      (set! dm2 (get_item b_mm2 i))
      (set! df (get_item b_mf i))
      (set! ee (if (or (== dmm 2) (== dmm -2)) e2 (if (or (== dmm 1) (== dmm -1)) e 1.0)))
      (set! arg (to_rad (+ (* dm d) (+ (* dmm mm) (+ (* dm2 m2) (* df f))))))
      (set! s_b (+ s_b (* (* (get_item b_cs i) ee) (Math.sin arg))))
      (set! i (+ i 1))
    })
    ;; 金星/木星摄动与地球扁率附加项
    (set! s_l (+ s_l (+ (* 3958.0 (Math.sin (to_rad a1)))
                        (+ (* 1962.0 (Math.sin (to_rad (- ll f))))
                           (* 318.0 (Math.sin (to_rad a2)))))))
    (set! s_b (+ s_b (+ (* -2235.0 (Math.sin (to_rad ll)))
                        (+ (* 382.0 (Math.sin (to_rad a3)))
                           (+ (* 175.0 (Math.sin (to_rad (- a1 f))))
                              (+ (* 175.0 (Math.sin (to_rad (+ a1 f))))
                                 (+ (* 127.0 (Math.sin (to_rad (- ll m2))))
                                    (* -115.0 (Math.sin (to_rad (+ ll m2)))))))))))
    (set! m_lam (+ ll (/ s_l 1000000.0)))
    (set! m_bet (/ s_b 1000000.0))
    (set! m_dist (+ 385000.56 (/ s_r 1000.0)))
  }))

;; ---- 太阳位置（ephemeris.c: calculate_solar_ecliptic_coordinates，AA Ch.25）--

(define s_lam 0.0)                  ; 太阳视黄经（度）

(define solar_ecliptic
  (lambda () {
    (define h e_hcent)
    (define h2 (* h h))
    (define u1 (norm360 (+ (arg_lin 280.46646 0.985647360164) (* 0.0003032 h2))))
    (define mmd (norm360 (+ (arg_lin 357.52911 0.985600281725) (- 0.0 (* 0.0001537 h2)))))
    (define mm (to_rad mmd))
    (define c (+ (* (+ 1.914602 (+ (* -0.004817 h) (* -0.000014 h2))) (Math.sin mm))
                 (+ (* (+ 0.019993 (* -0.000101 h)) (Math.sin (* 2.0 mm)))
                    (* 0.000289 (Math.sin (* 3.0 mm))))))
    (set! s_lam (norm360 (+ u1 c)))
  }))

;; ---- 黄道 → 赤道（ephemeris.c: calculate_lunar/solar_equatorial_coordinates）

(define m_ra 0.0)
(define m_dec 0.0)
(define s_ra 0.0)
(define s_dec 0.0)

(define moon_equatorial
  (lambda () {
    (lunar_ecliptic)
    (define bet (to_rad m_bet))
    (define eps (to_rad EPS_DEG))
    (define lam (to_rad m_lam))
    (define rr m_dist)
    (define cb (* rr (Math.cos bet)))
    (define sb (* rr (Math.sin bet)))
    (define x (+ (- 0.0 (* (* cb (Math.cos eps)) (Math.sin lam))) (* sb (Math.sin eps))))
    (define y (- 0.0 (* cb (Math.cos lam))))
    (define z (+ (* (* cb (Math.sin eps)) (Math.sin lam)) (* sb (Math.cos eps))))
    (define rad (to_deg (atan2 y x)))
    (if (and (> rad -90.0) (<= rad 180.0))
        { (set! m_ra (- 270.0 rad)) }
        { (if (and (<= rad -90.0) (>= rad -180.0)) { (set! m_ra (- (- 0.0 90.0) rad)) }) })
    (set! m_dec (to_deg (atan2 z (Math.sqrt (+ (* x x) (* y y))))))
  }))

(define sun_equatorial
  (lambda () {
    (solar_ecliptic)
    (define eps (to_rad EPS_DEG))
    (define lamr (to_rad s_lam))
    (define mmul (- 0.0 (* (Math.cos eps) (Math.sin lamr))))
    (define nn (- 0.0 (Math.cos lamr)))
    (define oo (* (Math.sin eps) (Math.sin lamr)))
    (define dnm (Math.sqrt (+ (* mmul mmul) (* nn nn))))
    (define asv (to_deg (asin (/ (- 0.0 nn) dnm))))
    (if (and (>= s_lam 0.0) (< s_lam 180.0))
        { (set! s_ra (- 90.0 asv)) }
        { (set! s_ra (+ 270.0 asv)) })
    (set! s_dec (to_deg (atan2 oo dnm)))
  }))

;; ---- 赤道 → 地平（ephemeris.c: equatorial_to_horizontal）---------------------

(define eq_az 0.0)
(define eq_alt 0.0)

(define eq_to_hor
  (lambda (ra dec) {
    (define l (+ e_gmst LON))
    (define ii (norm360 (- l ra)))
    (define tt (if (and (>= ii 0.0) (< ii 180.0)) (+ ii 90.0) (- ii 270.0)))
    (define dr (to_rad dec))
    (define tr (to_rad tt))
    (define phr (to_rad LAT))
    (define x (- (* (* (Math.cos dr) (Math.sin phr)) (Math.sin tr))
                 (* (Math.sin dr) (Math.cos phr))))
    (define y (- 0.0 (* (Math.cos dr) (Math.cos tr))))
    (define z (+ (* (* (Math.cos dr) (Math.cos phr)) (Math.sin tr))
                 (* (Math.sin dr) (Math.sin phr))))
    (set! eq_alt (to_deg (atan2 z (Math.sqrt (+ (* x x) (* y y))))))
    (set! eq_az (norm360 (+ 180.0 (to_deg (atan2 y x)))))
  }))

;; ---- 月相与亮肢方位角（ephemeris.c: moon_phase / moon_bright_limb_pos_angle）--

(define phase_i 0.0)                ; 相位角 i（度）：月心处地球-太阳角距
(define bright_pa 0.0)              ; 亮肢方位角（自天球北向东，度）

(define moon_phase
  (lambda () {
    (define a0 (to_rad s_ra))
    (define d0 (to_rad s_dec))
    (define a (to_rad m_ra))
    (define d (to_rad m_dec))
    (define cosp (+ (* (Math.sin d0) (Math.sin d))
                    (* (* (Math.cos d0) (Math.cos d)) (Math.cos (- a0 a)))))
    (set! phase_i (norm360 (to_deg (acos (- 0.0 cosp)))))
  }))

(define moon_bright_limb
  (lambda () {
    (define a0 (to_rad s_ra))
    (define d0 (to_rad s_dec))
    (define a (to_rad m_ra))
    (define d (to_rad m_dec))
    (define y (* (Math.cos d0) (Math.sin (- a0 a))))
    (define x (- (* (Math.sin d0) (Math.cos d))
                 (* (* (Math.cos d0) (Math.sin d)) (Math.cos (- a0 a)))))
    (set! bright_pa (norm360 (to_deg (atan2 y x))))
  }))

;; ---- 月盘倾斜角（celestial.c render_moon 的算法）---------------------------
;; 屏幕投影：项目默认鱼眼（天顶视角）horizontal_to_screen_xy：
;;   r = (π/2 - alt)/(π/2)，X = r·sin(-az)，Y = -r·cos(-az)。半径 1、中心 0
;;（投影的平移/缩放不影响两点连线方向，故只取角度）。

(define scr_x (lambda (az alt) (* (/ (- (/ PI 2.0) (to_rad alt)) (/ PI 2.0))
                                  (Math.sin (- 0.0 (to_rad az))))))
(define scr_y (lambda (az alt) (- 0.0 (* (/ (- (/ PI 2.0) (to_rad alt)) (/ PI 2.0))
                                         (Math.cos (- 0.0 (to_rad az)))))))

(define tilt_north 0.0)             ; 天球北向在屏幕上的切向角（度）
(define disk_tilt 0.0)              ; 月面亮区方向相对屏幕 +y 的倾角（度）

(define moon_tilt
  (lambda () {
    (eq_to_hor m_ra m_dec)
    (define maz eq_az)
    (define malt eq_alt)
    (define xa (scr_x maz malt))
    (define ya (scr_y maz malt))
    ;; 沿天球赤经圈向北 2° 处的地平方位/高度
    (eq_to_hor m_ra (+ m_dec 2.0))
    (define naz eq_az)
    (define nalt eq_alt)
    (define xb (scr_x naz nalt))
    (define yb (scr_y naz nalt))
    (set! tilt_north (norm360 (to_deg (atan2 (- xb xa) (- 0.0 (- yb ya))))))
    ;; 亮区两端近似为月球地理南北极：亮肢方位角相对天球北向的偏转 → 屏幕倾角
    (set! disk_tilt (norm360 (- tilt_north (- (norm360 bright_pa) 90.0))))
  }))

;; ---- VSOP87C(milli) 地球日心直角坐标 → 太阳地心黄经（校对值）---------------

(define ser
  (lambda (lst t) {
    (define i 0)
    (define n (length lst))
    (define acc 0.0)
    (define e 0)
    (while (< i n) {
      (set! e (get_item lst i))
      (set! acc (+ acc (* (get_item e 0)
                          (Math.cos (+ (get_item e 1) (* (get_item e 2) t))))))
      (set! i (+ i 1))
    })
    acc
  }))

(define vsop_sun_lam 0.0)
(define vsop_sun_r 0.0)

(define vsop_sun
  (lambda () {
    (define t (/ e_hcent 10.0))     ; 儒略千年数 = 儒略世纪数/10
    (define ex (+ (ser e_x0 t) (+ (* (ser e_x1 t) t) (* (ser e_x2 t) (* t t)))))
    (define ey (+ (ser e_y0 t) (+ (* (ser e_y1 t) t) (* (ser e_y2 t) (* t t)))))
    (define sx (- 0.0 ex))
    (define sy (- 0.0 ey))
    (set! vsop_sun_lam (norm360 (to_deg (atan2 sy sx))))
    (set! vsop_sun_r (Math.sqrt (+ (* sx sx) (* sy sy))))
  }))

;; ---- 取值接口（Animac 无多值返回，多结果统一经模块全局 + get_* 读取）------

(define get_az (lambda () eq_az))
(define get_alt (lambda () eq_alt))
(define get_phase (lambda () phase_i))
(define get_bright_pa (lambda () bright_pa))
(define get_tilt_north (lambda () tilt_north))
(define get_disk_tilt (lambda () disk_tilt))
(define get_vsop_lon (lambda () vsop_sun_lam))
(define get_vsop_r (lambda () vsop_sun_r))
(define get_moon_lat (lambda () m_bet))
(define get_moon_dist (lambda () m_dist))
(define get_sun_lon (lambda () s_lam))
(define get_moon_lon (lambda () m_lam))

;; ---- 时间与历法工具 --------------------------------------------------------

;; 太阳视黄经（度）：先算黄道坐标再取值
(define sun_lon (lambda () { (solar_ecliptic) s_lam }))
;; 月球黄经（度）
(define moon_lon (lambda () { (lunar_ecliptic) m_lam }))
;; 太阳/月亮的地平坐标（结果写 eq_az/eq_alt，经 get_az/get_alt 读取）
;; NOTE：需先调用 moon_equatorial / sun_equatorial 更新赤道坐标。
(define sun_altaz (lambda () { (eq_to_hor s_ra s_dec) }))
(define moon_altaz (lambda () { (eq_to_hor m_ra m_dec) }))

;; 月日黄经差 norm360(λm − λs)：朔（=0）与望（=180）的求根函数
(define moon_sun_diff
  (lambda () {
    (solar_ecliptic)
    (define sl s_lam)
    (lunar_ecliptic)
    (norm360 (- m_lam sl))
  }))

;; 整数日历：dn → 年月日（1970 起逐年/逐月推进，全整数运算）
(define civ_y 1970)
(define civ_m 1)
(define civ_d 1)
(define dim_table '(31 28 31 30 31 30 31 31 30 31 30 31))

(define leap_int
  (lambda (yy) (and (== (mod yy 4) 0) (or (not (== (mod yy 100) 0)) (== (mod yy 400) 0)))))
(define year_days (lambda (yy) (if (leap_int yy) 366 365)))
(define month_days
  (lambda (yy mm) {
    (define md (get_item dim_table (- mm 1)))
    (if (and (== mm 2) (leap_int yy)) { (set! md (+ md 1)) })
    md
  }))

(define civil
  (lambda (days) {
    (define yy 1970)
    (define dd days)
    (while (>= dd (year_days yy)) { (set! dd (- dd (year_days yy))) (set! yy (+ yy 1)) })
    (while (< dd 0) { (set! yy (- yy 1)) (set! dd (+ dd (year_days yy))) })
    (define mm 1)
    (while (>= dd (month_days yy mm)) {
      (set! dd (- dd (month_days yy mm)))
      (set! mm (+ mm 1))
    })
    (set! civ_y yy) (set! civ_m mm) (set! civ_d (+ dd 1))
  }))

(define get_civ_y (lambda () civ_y))
(define get_civ_m (lambda () civ_m))
(define get_civ_d (lambda () civ_d))

;; 年月日 → dn（1970 起的天数；与 civil 互逆）
(define days_from_civil
  (lambda (yy mm dd) {
    (define n 0)
    (define y 1970)
    (while (< y yy) { (set! n (+ n (year_days y))) (set! y (+ y 1)) })
    (while (> y yy) { (set! y (- y 1)) (set! n (- n (year_days y))) })
    (define m 1)
    (while (< m mm) { (set! n (+ n (month_days yy m))) (set! m (+ m 1)) })
    (+ n (- dd 1))
  }))

;; 北京时间（UTC+8）民用日编号（等价于 nongli.c 的 cst_day_number）
(define cst_day_number
  (lambda (dn sod) {
    (define s (+ sod TZ_SEC))
    (if (>= s 86400.0) (+ dn 1) dn)
  }))
