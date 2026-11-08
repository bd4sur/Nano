;; ============================================================================
;; 星历 · 农历（Ephemeris / Nongli）—— 灵机应用程序（单页布局）
;;
;; 一屏内并排呈现两类信息（原为 ephemeris.scm / nongli.scm 两个 App，2026-11 合并）：
;;   · 星历：太阳/月亮地平坐标 · 月相（相位角/照亮比/月相名）· 月盘亮肢与倾斜 · VSOP87 校对
;;   · 农历：农历年月日（含闰月）· 干支纪年/月/日 · 生肖 · 本月中气 · 下一节气 · 下朔
;;
;; 本 App 是**薄壳**：全部天文计算来自共享模块 app/lib/ephem.scm（纯计算库），
;; 这里只负责「读系统时间 → 驱动库 → 用 Gfx 绘制文字」。
;;
;; 输入：当前系统时间（本地库 Time：UTC 整数日 + 当日秒）+ 项目默认观测点
;;       南京 119.0°E / 32.0°N / UTC+8（取自 ui_app.c 中 global_state 的初值）。
;;
;; ---- 单页排版（320x240，12px 平滑字模：全角 12px / 半角 7px / 行高 15px）------
;;
;;   y=3   星历 · 农历 · 南京 119.0°E 32.0°N            [退出 36x16 @ x=280]
;;   y=19  2026-10-20 10:00:03 UTC+8
;;   y=36  ───── 分组分隔线 ─────
;;   y=44  太阳   高度 +40.34°   方位 143.82°
;;   y=62  月亮   高度 -49.56°   方位  80.96°
;;   y=79  ─────
;;   y=87  月相   相位角 74.5°  照亮 63.4%  上弦月
;;   y=105 月盘   亮肢 253.3°   倾斜 215.6°  北向 18.9°
;;   y=122 ─────
;;   y=130 农历   丙午马  九月十一
;;   y=148 干支   丙午年 戊戌月 丁卯日
;;   y=166 中气   霜降 10-23 17:36
;;   y=184 节气   霜降 10-23 17:36   下朔 11-09 15:04
;;   y=201 ─────
;;   y=209 VSOP87 黄经 206.703°   日地 0.9959 AU
;;   y=224 年表   农历 2026 年（表 2026）
;;
;; 排版约定：
;;   · 标签列固定在 x=6（暗灰蓝），字段名（高度/方位/…）中灰，数值按主题着色；
;;   · 数值列左边界对齐（x=44 起；月相/月盘/VSOP 组的数值列按字段名宽度顺延），
;;     数值、日期、时刻一律**分片绘制**（符号/整数/小数/单位各一片），见「内存纪律」；
;;   · 分组用 1px 暗分隔线，行距 18px；内容占满 y=3..236，无留白；
;;   · 右上角 280..316 / 2..18 留给「退出」按钮（app/lib/exit.scm），标题宽 219px 不冲突。
;;
;; ---- 与合并前的关系 --------------------------------------------------------
;;   · 字段一个都没丢，只是把两个 App 各自的「小标题 + 若干行」压成「标签 + 一行」：
;;     太阳/月亮各由 3 行（小标题 + 高度 + 方位）压成 1 行，月相/月盘同理；
;;     农历的「年月日 + 干支」两行保留，节气与下朔并到一行，年表与 VSOP87 移到页脚。
;;   · 农历年表在后台按拍推进（每拍 4 步，约 1.3 s 算完），算完前农历组显示进度；
;;     因只有一页，进度与星历数据同帧可见，不需要切页。
;;   · 只保留**一个** System.set_interval：上游每轮事件循环对同一进程只触发一个
;;     定时器回调，多定时器会互相拖延（教训见 app/lib/exit.scm 头部）。
;;   · 星历数据在本帧内先算好并快照到 v_* 变量，再跑农历推算（后者会改写 ephem 的
;;     时间基与黄经中间量），绘制只读快照 —— 两组数据互不污染。
;;
;; ---- 农历算法（移植自 infer/nongli.c，其星历部分来自 ephemeris.c = J. Meeus AA）--
;;   · 牛顿迭代求朔（月日黄经差 = 0）与中气（太阳黄经 = 330°,0°,30°,…,300°）
;;   · 无中气之月为闰月，闰月序号继承前一个月
;;   · 农历年界：含「雨水」的朔望月为正月，其朔日落在本公历年
;;   · 干支纪年（1984 甲子）/纪月（五虎遁）/纪日（1900-01-31 甲辰为基准）、生肖
;;   · 北京时间民用日编号 cst_day_number 统一判定月首/日序（避免时刻偏差）
;;
;; 时间表示：与星历库一致，用 (dn, sod)（UTC 整数日 + 当日秒），避免 32 位宿主
;;   上大数浮点（儒略日）的精度损失。
;;
;; 命名约定：标识符一律下划线分隔（不使用短划线）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Time)

(import Ephem "app/lib/ephem.scm")
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ============================================================================
;; 排版常量
;; ============================================================================

;; 文字行（y_top；行高 15px，组内行距 18px）
(define Y_TITLE 3)
(define Y_CLOCK 19)
(define Y_SUN 44)
(define Y_MOON 62)
(define Y_PHASE 87)
(define Y_DISC 105)
(define Y_LUNAR 130)
(define Y_GANZHI 148)
(define Y_TERM 166)
(define Y_TERM2 184)
(define Y_VSOP 209)
(define Y_TABLE 224)

;; 分组分隔线
(define Y_SEP1 36)
(define Y_SEP2 79)
(define Y_SEP3 122)
(define Y_SEP4 201)

;; 列：标签列 6；字段值列 44（月相/月盘/VSOP 组按字段名宽度顺延，见各行）
(define LX 6)
(define VX 44)

;; ============================================================================
;; 通用格式化（两页共用）
;; ============================================================================

(define absv (lambda (v) (if (< v 0.0) (- 0.0 v) v)))

(define padz
  (lambda (s n) {
    (define r s)
    (while (< (String.length r) n) { (set! r (String.concat "0" r)) })
    r
  }))

(define pad2
  (lambda (n) {
    (define v (Math.floor n))
    (if (< v 10.0)
        (String.concat "0" (String.atom_to_string v))
        (String.atom_to_string v))
  }))

;; 时分：由当日秒（可为小数）给出 "HH:MM"（仅供 fmt_time_bj 拼接中气时刻；
;; 该串每日只变几次，不是每秒生成的新串，不构成驻留表增长源）
(define hhmm
  (lambda (sod) {
    (define h (Math.floor (/ sod 3600.0)))
    (define mi (Math.floor (/ (- sod (* h 3600.0)) 60.0)))
    (String.concat (pad2 h) (String.concat ":" (pad2 mi)))
  }))

;; ============================================================================
;; 星历：显示量（本帧快照）与计算
;; ============================================================================

(define v_y 1970)
(define v_m 1)
(define v_d 1)
(define v_h 0)
(define v_mi 0)
(define v_s 0)
(define v_sun_az 0.0)
(define v_sun_alt 0.0)
(define v_moon_az 0.0)
(define v_moon_alt 0.0)
(define v_phase 0.0)                ; 相位角（度）
(define v_k 0.0)                    ; 月面照亮比例
(define v_elong 0.0)                ; 月日黄经差（度，定月相名用）
(define v_bright_pa 0.0)            ; 亮肢方位角（度）
(define v_tilt 0.0)                 ; 月盘倾斜角（度）
(define v_north 0.0)                ; 天球北向角（度）
(define v_vsop_lon 0.0)             ; VSOP87 校对黄经（度）
(define v_vsop_r 0.0)               ; 日地距离（AU）

(define phase_name
  (lambda (e) (cond ((< e 22.5) "新月（朔）")
                    ((< e 67.5) "娥眉月")
                    ((< e 112.5) "上弦月")
                    ((< e 157.5) "盈凸月")
                    ((< e 202.5) "满月（望）")
                    ((< e 247.5) "亏凸月")
                    ((< e 292.5) "下弦月")
                    ((< e 337.5) "残月")
                    (else "新月（朔）"))))

;; 读时 → 驱动星历库 → 把结果**快照**进 v_*（此后本帧不再依赖 ephem 的模块状态，
;; 农历推算可以随意改写时间基/黄经中间量而不影响本组数值）。
(define astro_compute
  (lambda () {
    (define udays (Time.utc_days))
    (define usod (Time.utc_sod))
    ;; 库的时间基（UT）
    (Ephem.set_time udays usod)
    (Ephem.set_gmst)
    ;; 本地民用时（UTC+8）用于显示
    (define ldays udays)
    (define lsod (+ usod Ephem.TZ_SEC))
    (if (>= lsod 86400) { (set! lsod (- lsod 86400)) (set! ldays (+ ldays 1)) })
    (Ephem.civil ldays)
    (set! v_y (Ephem.get_civ_y))
    (set! v_m (Ephem.get_civ_m))
    (set! v_d (Ephem.get_civ_d))
    (set! v_h (Math.floor (/ lsod 3600.0)))
    (set! v_mi (Math.floor (/ (- lsod (* v_h 3600.0)) 60.0)))
    (set! v_s (- lsod (+ (* v_h 3600.0) (* v_mi 60.0))))
    ;; 日月位置与各种角度
    (Ephem.moon_equatorial)
    (Ephem.sun_equatorial)
    (Ephem.sun_altaz)
    (set! v_sun_az (Ephem.get_az))
    (set! v_sun_alt (Ephem.get_alt))
    (Ephem.moon_altaz)
    (set! v_moon_az (Ephem.get_az))
    (set! v_moon_alt (Ephem.get_alt))
    (Ephem.moon_phase)
    (set! v_phase (Ephem.get_phase))
    (Ephem.moon_bright_limb)
    (set! v_bright_pa (Ephem.get_bright_pa))
    (Ephem.moon_tilt)               ; 内部会改写 eq_az/eq_alt，故上面已先取值
    (set! v_tilt (Ephem.get_disk_tilt))
    (set! v_north (Ephem.get_tilt_north))
    (Ephem.vsop_sun)
    (set! v_vsop_lon (Ephem.get_vsop_lon))
    (set! v_vsop_r (Ephem.get_vsop_r))
    (set! v_k (* 0.5 (+ 1.0 (Math.cos (Ephem.to_rad v_phase)))))
    (set! v_elong (Ephem.norm360 (- (Ephem.get_moon_lon) (Ephem.get_sun_lon))))
  }))

;; ============================================================================
;; 农历：常量表
;; ============================================================================

;; 十二中气对应的太阳黄经（0=雨水…11=大寒）
(define major_term_lon '(330 0 30 60 90 120 150 180 210 240 270 300))
;; 十二中气名
(define term_names '("雨水" "春分" "谷雨" "小满" "夏至" "大暑"
                     "处暑" "秋分" "霜降" "小雪" "冬至" "大寒"))
;; 月名（正月…腊月）
(define month_names '("正" "二" "三" "四" "五" "六" "七" "八" "九" "十" "冬" "腊"))
;; 农历日名
(define lunar_day_names '("初一" "初二" "初三" "初四" "初五" "初六" "初七" "初八" "初九" "初十"
                          "十一" "十二" "十三" "十四" "十五" "十六" "十七" "十八" "十九" "二十"
                          "廿一" "廿二" "廿三" "廿四" "廿五" "廿六" "廿七" "廿八" "廿九" "三十"))
;; 天干 / 地支 / 生肖
(define stems '("甲" "乙" "丙" "丁" "戊" "己" "庚" "辛" "壬" "癸"))
(define branches '("子" "丑" "寅" "卯" "辰" "巳" "午" "未" "申" "酉" "戌" "亥"))
(define zodiacs '("鼠" "牛" "虎" "兔" "龙" "蛇" "马" "羊" "猴" "鸡" "狗" "猪"))
;; 五虎遁年起月法：甲/己→丙寅，乙/庚→戊寅，丙/辛→庚寅，丁/壬→壬寅，戊/癸→甲寅
(define month_gan_start '(2 4 6 8 0 2 4 6 8 0))
;; 二十四节气名（按太阳黄经 315° 起每 15° 一个：315=立春, 330=雨水, 345=惊蛰, 0=春分, …）
(define term24_names '("立春" "雨水" "惊蛰" "春分" "清明" "谷雨" "立夏" "小满"
                       "芒种" "夏至" "小暑" "大暑" "立秋" "处暑" "白露" "秋分"
                       "寒露" "霜降" "立冬" "小雪" "大雪" "冬至" "小寒" "大寒"))

;; ---- 推算结果（模块级；Animac 无多值返回）----------------------------------

(define out_year 0)
(define out_month 0)
(define out_day 0)
(define out_leap 0)
(define out_ok 0)
(define out_ystr "")
(define out_dstr "")
(define out_mname "")
(define out_dname "")
(define out_zodiac "")
(define out_mgz "")
(define out_dgz "")
(define out_termname "")
(define out_termtime "")

;; ---- 小工具 ---------------------------------------------------------------

(define pmod
  (lambda (a n) {
    (define v (mod a n))
    (if (< v 0.0) { (set! v (+ v n)) })
    v
  }))

;; (dn, sod) → 北京时间 "MM-DD HH:MM"
(define fmt_time_bj
  (lambda (dn sod) {
    (define ldays dn)
    (define lsod (+ sod Ephem.TZ_SEC))
    (if (>= lsod 86400.0) { (set! lsod (- lsod 86400.0)) (set! ldays (+ ldays 1)) })
    (Ephem.civil ldays)
    (String.concat (pad2 (Ephem.get_civ_m))
      (String.concat "-"
        (String.concat (pad2 (Ephem.get_civ_d))
          (String.concat " " (hhmm lsod)))))
  }))

;; ---- 星历库驱动：在任意时刻求值（牛顿迭代用）------------------------------

;; 把 (dn, sod) 归一化（sod ∈ [0, 86400)），写回 nrm_dn / nrm_sod
(define nrm_dn 0)
(define nrm_sod 0.0)
(define normalize_time
  (lambda (dn sod) {
    (define s sod)
    (define d dn)
    (while (>= s 86400.0) { (set! s (- s 86400.0)) (set! d (+ d 1)) })
    (while (< s 0.0) { (set! s (+ s 86400.0)) (set! d (- d 1)) })
    (set! nrm_dn d)
    (set! nrm_sod s)
  }))

;; 在 (dn, sod) 之后 off_sec 秒处求「月日黄经差」（归一化到 (−180,180]）
(define diff_at
  (lambda (dn sod off_sec) {
    (normalize_time dn (+ sod off_sec))
    (Ephem.set_time nrm_dn nrm_sod)
    (define f (Ephem.moon_sun_diff))
    (if (> f 180.0) { (set! f (- f 360.0)) })
    f
  }))

;; 在 (dn, sod) 之后 off_sec 秒处求太阳视黄经
(define sunlon_at
  (lambda (dn sod off_sec) {
    (normalize_time dn (+ sod off_sec))
    (Ephem.set_time nrm_dn nrm_sod)
    (Ephem.sun_lon)
  }))

;; 迭代求朔（月日黄经差 = 0）：结果写入 nm_dn / nm_sod
;;
;; NOTE 1：不用 nongli.c 的数值微分牛顿法。32 位宿主浮点有效尾数约 19 位，
;;         黄经差量级 100° 时分辨率约 1e-4°；0.001 天步长只引起 1.2e-5° 的变化，
;;         差分会退化成量化噪声，导数失真会让迭代跳到错误的朔。
;; NOTE 2：也不用较大步长的中心差分——虽然导数可用，但多一倍求值开销。这里
;;         直接采用「月日黄经差的平均速率」作固定斜率（与 find_solar_term 对
;;         太阳用 0.9856 °/天同构）：真实速率在 11.3~13.2 °/天之间，误差因子
;;         |1 - 12.19/rate| < 0.1，三四步即收敛到 1e-5 天以内，且完全无差分噪声。
(define nm_dn 0)
(define nm_sod 0.0)
(define find_new_moon
  (lambda (dn0 sod0) {
    (define dn dn0)
    (define sod sod0)
    (define i 0)
    (define f 0.0)
    (define dj 0.0)
    (while (< i 20) {
      (set! f (diff_at dn sod 0.0))
      (set! dj (/ (- 0.0 f) 12.19))
      (if (> dj 20.0) { (set! dj 20.0) })
      (if (< dj -20.0) { (set! dj -20.0) })
      (normalize_time dn (+ sod (* dj 86400.0)))
      (set! dn nrm_dn)
      (set! sod nrm_sod)
      (set! i (+ i 1))
      (if (< (Math.abs dj) 0.000001) { (set! i 20) })
    })
    (set! nm_dn dn)
    (set! nm_sod sod)
  }))

;; 牛顿迭代求中气（太阳黄经 = target）：结果写入 mt_dn / mt_sod
(define mt_dn 0)
(define mt_sod 0.0)
(define find_solar_term
  (lambda (dnx sdx target) {
    (normalize_time dnx sdx)
    (define dn0 nrm_dn)
    (define sod0 nrm_sod)
    (define dn dn0)
    (define sod sod0)
    (define i 0)
    (define f 0.0)
    (define dj 0.0)
    (while (< i 20) {
      (set! f (Ephem.norm360 (- (sunlon_at dn sod 0.0) target)))
      (if (> f 180.0) { (set! f (- f 360.0)) })
      (set! dj (/ (- 0.0 f) 0.9856))        ; 太阳黄经变化率 °/天
      (normalize_time dn (+ sod (* dj 86400.0)))
      (set! dn nrm_dn)
      (set! sod nrm_sod)
      (set! i (+ i 1))
      (if (< (Math.abs dj) 0.000001) { (set! i 20) })
    })
    ;; 折叠到初值 ± 半年内（与 nongli.c 的 PERIOD/2 折叠等价）
    (define dd (+ (- dn dn0) (/ (- sod sod0) 86400.0)))
    (while (> dd 182.625) {
      (normalize_time dn (- sod (* 365.25 86400.0)))
      (set! dn nrm_dn) (set! sod nrm_sod)
      (set! dd (- dd 365.25))
    })
    (while (< dd -182.625) {
      (normalize_time dn (+ sod (* 365.25 86400.0)))
      (set! dn nrm_dn) (set! sod nrm_sod)
      (set! dd (+ dd 365.25))
    })
    (set! mt_dn dn)
    (set! mt_sod sod)
  }))

;; ---- 年表缓存与分帧推算 ---------------------------------------------------
;; nm_list：升序 ((dn sod) ...)   —— 朔
;; mt_list：升序 ((dn sod idx) ...) —— 中气（idx = 0..11，对应 major_term_lon）

(define synodic 29.5306)
(define cache_year -999999)
(define nm_list '())
(define nm_n 0)
(define mt_list '())
(define mt_n 0)

(define tbl_ready 0)                ; 0 = 推算中，1 = 就绪
(define build_kind 0)               ; 0 = 朔表，1 = 中气表
(define build_i 0)
(define build_total 0)
(define build_year 0)
(define raw_list '())               ; 临时收集（cons 构建，不用 push）
(define cur_dn 0)                   ; 朔表游标
(define cur_sod 0.0)

;; 列表插入（升序，按首字段 dn），函数式构建
(define insert_pair
  (lambda (lst x) {
    (if (null? lst)
        (cons x '())
        (if (< (get_item x 0) (get_item (car lst) 0))
            (cons x lst)
            (cons (car lst) (insert_pair (cdr lst) x))))
  }))
(define sort_pairs
  (lambda (lst) {
    (define out '())
    (define cur lst)
    (while (not (null? cur)) {
      (set! out (insert_pair out (car cur)))
      (set! cur (cdr cur))
    })
    out
  }))

;; 启动一年表推算：确定需要覆盖的时间范围与步数
(define build_begin
  (lambda (ystart) {
    (set! build_year ystart)
    (set! build_kind 0)
    (set! build_i 0)
    (set! raw_list '())
    (set! nm_list '())
    (set! mt_list '())
    (set! tbl_ready 0)
    ;; 朔表游标：公历年 ystart 的 1 月 1 日之前 1 个朔望月
    (set! cur_dn (- (Ephem.days_from_civil ystart 1 1) 40))
    (set! cur_sod 0.0)
    (set! build_total 0)
  }))

;; 推算一步：解一个朔或一个中气
(define build_step
  (lambda () {
    (if (== build_kind 0)
        {
          ;; ---- 朔表：覆盖 ystart 与 ystart+1 两年 ----
          (find_new_moon cur_dn cur_sod)
          (set! raw_list (cons (cons nm_dn (cons nm_sod '())) raw_list))
          (set! build_i (+ build_i 1))
          ;; 游标前移 0.8 个朔望月，继续求下一个
          (normalize_time nm_dn (+ nm_sod (* 0.8 (* synodic 86400.0))))
          (set! cur_dn nrm_dn)
          (set! cur_sod nrm_sod)
          (define end_dn (+ (Ephem.days_from_civil (+ build_year 1) 12 31) 10))
          (if (or (> cur_dn end_dn) (>= build_i 40))
              {
                (set! nm_list (sort_pairs raw_list))
                (set! nm_n (length nm_list))
                (set! raw_list '())
                (set! build_kind 1)
                (set! build_i 0)
              })
        }
        {
          ;; ---- 中气表：ys 年 12 个 + (ys+1) 年 12 个 ----
          (define yr build_year)
          (define k build_i)
          (if (>= k 12) { (set! yr (+ yr 1)) (set! k (- k 12)) })
          (define start_dn (Ephem.days_from_civil yr 1 1))
          ;; 以年初太阳黄经确定初值（nongli.c 的 Bug1 修复）
          (Ephem.set_time start_dn 0)
          (define lon0 (Ephem.sun_lon))
          (define target (get_item major_term_lon k))
          (define diff (Ephem.norm360 (- target lon0)))
          (define guess_sec (* (/ diff 0.9856) 86400.0))
          (find_solar_term start_dn guess_sec target)
          ;; 只保留落在该公历年 ±30 天内的中气
          (define lo (- start_dn 30))
          (define hi (+ (Ephem.days_from_civil (+ yr 1) 1 1) 30))
          (if (and (>= mt_dn lo) (<= mt_dn hi))
              { (set! raw_list (cons (cons mt_dn (cons mt_sod (cons k '()))) raw_list)) })
          (set! build_i (+ build_i 1))
          (if (>= build_i 24)
              {
                (set! mt_list (sort_pairs raw_list))
                (set! mt_n (length mt_list))
                (set! raw_list '())
                (set! cache_year build_year)
                (set! tbl_ready 1)
              })
        })
  }))

;; 每拍推进 BUILD_BATCH 步（每步一解，避免单拍超时）；返回 1 表示本拍推进过
;; （调用方据此刷新进度显示，也在"刚好算完"的那一拍补算农历结果）。
(define BUILD_BATCH 4)
(define build_batch
  (lambda () {
    (define did 0)
    (define k 0)
    (while (and (== tbl_ready 0) (< k BUILD_BATCH)) {
      (build_step)
      (set! did 1)
      (set! k (+ k 1))
    })
    did
  }))

;; 确保覆盖指定农历年（必要时重启推算）
(define ensure_year
  (lambda (ystart) {
    (if (not (== cache_year ystart)) { (build_begin ystart) })
  }))

;; ---- 农历推算（移植 nongli.c: lunar_calculate）-----------------------------

;; 在 mt_list 中取第 i 项的第 j 个字段
(define mt_item (lambda (i j) (get_item (get_item mt_list i) j)))
(define nm_item (lambda (i j) (get_item (get_item nm_list i) j)))

(define lunar_compute
  (lambda (dn sod) {
    (define today_cst (Ephem.cst_day_number dn sod))
    ;; 本地（北京）日期决定「公历年」的取法（与 nongli.c 一致：1 月归上一年）
    (Ephem.civil today_cst)
    (define gy (Ephem.get_civ_y))
    (define gm (Ephem.get_civ_m))
    (define ystart (if (< gm 2) (- gy 1) gy))
    (set! out_ok 0)
    (if (and (== tbl_ready 1) (== cache_year ystart))
        {
          ;; ---- 定位当前朔望月 ----
          (define mi 0)
          (define found 0)
          (define mstart_dn 0)
          (define mstart_sod 0.0)
          (define mnext_dn 0)
          (define mnext_sod 0.0)
          (while (and (< mi (- nm_n 1)) (== found 0)) {
            (define a_cst (Ephem.cst_day_number (nm_item mi 0) (nm_item mi 1)))
            (define b_cst (Ephem.cst_day_number (nm_item (+ mi 1) 0) (nm_item (+ mi 1) 1)))
            (if (and (>= today_cst a_cst) (< today_cst b_cst))
                {
                  (set! found 1)
                  (set! mstart_dn (nm_item mi 0))
                  (set! mstart_sod (nm_item mi 1))
                  (set! mnext_dn (nm_item (+ mi 1) 0))
                  (set! mnext_sod (nm_item (+ mi 1) 1))
                })
            (set! mi (+ mi 1))
          })
          (if (== found 1)
              {
                (define mstart_cst (Ephem.cst_day_number mstart_dn mstart_sod))
                (define mnext_cst (Ephem.cst_day_number mnext_dn mnext_sod))
                (set! out_day (+ (- today_cst mstart_cst) 1))
                ;; ---- 本月中气 ----
                (define ti 0)
                (define tcount 0)
                (define tfirst -1)
                (define tlast_prev -1)
                (while (< ti mt_n) {
                  (define t_cst (Ephem.cst_day_number (mt_item ti 0) (mt_item ti 1)))
                  (if (and (>= t_cst mstart_cst) (< t_cst mnext_cst))
                      {
                        (set! tcount (+ tcount 1))
                        (if (== tfirst -1) { (set! tfirst ti) })
                      })
                  (if (< t_cst mstart_cst) { (set! tlast_prev ti) })
                  (set! ti (+ ti 1))
                })
                ;; ---- 闰月判定 ----
                (define mnum 0)
                (set! out_leap 0)
                (if (== tcount 0)
                    {
                      (set! out_leap 1)
                      (if (>= tlast_prev 0) { (set! mnum (mt_item tlast_prev 2)) })
                    }
                    { (set! mnum (mt_item tfirst 2)) })
                (set! out_month (+ mnum 1))
                (set! out_mname (String.concat
                                  (String.concat (if (== out_leap 1) "闰" "") (get_item month_names mnum))
                                  "月"))
                (define di (- out_day 1))
                (if (< di 0) { (set! di 0) })
                (if (> di 29) { (set! di 29) })
                (set! out_dname (get_item lunar_day_names di))
                ;; 本月中气的名称与时刻
                (if (>= tfirst 0)
                    {
                      (set! out_termname (get_item term_names (mt_item tfirst 2)))
                      (set! out_termtime (fmt_time_bj (mt_item tfirst 0) (mt_item tfirst 1)))
                    }
                    { (set! out_termname "无（闰月）") (set! out_termtime "") })
                ;; ---- 农历年：含「雨水」的朔望月为正月，其朔日在 ystart 年 ----
                (define zhengyu_cst 0)
                (define found_zy 0)
                (define i2 0)
                (while (and (< i2 (- nm_n 1)) (== found_zy 0)) {
                  (define a_cst2 (Ephem.cst_day_number (nm_item i2 0) (nm_item i2 1)))
                  (define b_cst2 (Ephem.cst_day_number (nm_item (+ i2 1) 0) (nm_item (+ i2 1) 1)))
                  (define j2 0)
                  (while (and (< j2 mt_n) (== found_zy 0)) {
                    (define t_cst2 (Ephem.cst_day_number (mt_item j2 0) (mt_item j2 1)))
                    (if (and (>= t_cst2 a_cst2) (< t_cst2 b_cst2) (== (mt_item j2 2) 0))
                        {
                          (Ephem.civil a_cst2)
                          (if (== (Ephem.get_civ_y) ystart)
                              { (set! zhengyu_cst a_cst2) (set! found_zy 1) })
                          (set! j2 mt_n)
                        }
                        { (set! j2 (+ j2 1)) })
                  })
                  (set! i2 (+ i2 1))
                })
                (define lyear ystart)
                (if (and (== found_zy 1) (< today_cst zhengyu_cst)) { (set! lyear (- ystart 1)) })
                (set! out_year lyear)
                ;; ---- 干支 ----
                (define ygan (pmod (- lyear 1984) 10.0))
                (define yzhi (pmod (- lyear 1984) 12.0))
                (set! out_ystr (String.concat (get_item stems ygan) (get_item branches yzhi)))
                (set! out_zodiac (get_item zodiacs yzhi))
                (define mgan (pmod (+ (get_item month_gan_start ygan) mnum) 10.0))
                (define mzhi (pmod (+ mnum 2) 12.0))
                (set! out_mgz (String.concat (get_item stems mgan) (get_item branches mzhi)))
                (define base_dn (Ephem.days_from_civil 1900 1 31))
                (define off (- today_cst base_dn))
                (define dgan (pmod off 10.0))
                (define dzhi (pmod (+ off 4) 12.0))
                (set! out_dgz (String.concat (get_item stems dgan) (get_item branches dzhi)))
                (set! out_ok 1)
              })
        })
  }))

;; ---- 下一个朔 / 下一个节气 ------------------------------------------------

(define next_nm_dn 0)
(define next_nm_sod 0.0)
(define next_nm_ok 0)

(define find_next_new_moon
  (lambda (dn sod) {
    (define i 0)
    (define found 0)
    (set! next_nm_ok 0)
    (while (and (< i nm_n) (== found 0)) {
      (define a_dn (nm_item i 0))
      (define a_sod (nm_item i 1))
      (if (or (> a_dn dn) (and (== a_dn dn) (> a_sod sod)))
          {
            (set! next_nm_dn a_dn)
            (set! next_nm_sod a_sod)
            (set! next_nm_ok 1)
            (set! found 1)
          })
      (set! i (+ i 1))
    })
  }))

(define next_term_dn 0)
(define next_term_sod 0.0)
(define next_term_idx 0)

(define find_next_term
  (lambda (dn sod) {
    (Ephem.set_time dn sod)
    (define lon (Ephem.sun_lon))
    ;; 下一个 15° 整倍数（严格大于当前黄经）
    (define k (Math.floor (/ lon 15.0)))
    (define target (* 15.0 (+ k 1)))
    (define diff (Ephem.norm360 (- target lon)))
    (find_solar_term dn (* diff (/ 86400.0 0.9856)) target)
    (set! next_term_dn mt_dn)
    (set! next_term_sod mt_sod)
    ;; 名称索引：黄经 315° 为立春 ⇒ idx = ((target + 45) / 15) mod 24
    (set! next_term_idx (pmod (/ (+ target 45.0) 15.0) 24.0))
  }))

;; 该时刻应覆盖的公历年起点（1 月归上一年，与 nongli.c 一致）
(define year_start_of
  (lambda (udays usod) {
    (Ephem.civil (Ephem.cst_day_number udays usod))
    (if (< (Ephem.get_civ_m) 2) (- (Ephem.get_civ_y) 1) (Ephem.get_civ_y))
  }))

;; 每秒一次：年表就绪则重算当日农历与下朔/下一节气；跨年则重启年表推算
;; （重启后由 build_batch 继续推进，农历组显示进度）。
(define lunar_update
  (lambda (udays usod) {
    (define ystart (year_start_of udays usod))
    (if (== tbl_ready 1)
        {
          (if (not (== cache_year ystart))
              { (build_begin ystart) }
              {
                (lunar_compute udays usod)
                (find_next_new_moon udays usod)
                (find_next_term udays usod)
              })
        }
        #f)
  }))

;; 读时 + 跑农历推演（可能重启年表推算）
(define lunar_refresh
  (lambda () (lunar_update (Time.utc_days) (Time.utc_sod))))

;; ============================================================================
;; 绘制（单页）
;; ============================================================================

;; 分组分隔线（1px 暗色，只做视觉分块，不抢视线）
(define sep (lambda (y) (Gfx.fill_rect 6 y 308 1 44 48 58)))

;; 标签列（暗灰蓝）与中灰的字段名
(define lbl (lambda (y s) (Gfx.draw_text s LX y 150 158 172)))
(define mfd (lambda (y x s) (Gfx.draw_text s x y 205 210 220)))

;; ---- 分片数值绘制（内存纪律，见下）----------------------------------------
;;
;; **内存纪律（重要）**：VM 会**永久驻留**运行期生成的 ≤32 字符短字符串（manual §3.4），
;; 驻留表不可回收。所以**每秒变化的文本不能整串拼**："2026-10-20 10:00:03 UTC+8" 的每一层
;; concat 中间结果（"2026-"、"2026-10-20 10:00"…）都是新的短串；同理每变一次的角度值
;; （"+40.34°"）也是一个新短串。实测 16MB 池（VM 8MB + 堆 8MB）在 100ms 拍下：整串拼时钟
;; ≈6 min 即 `[Runtime] 指令执行异常`；只修时钟 ≈1 h；两者都改成分片后长期稳定。
;;
;; 分片的依据：12px 平滑字模的**半角步进恒为 7px**（数字、./-/+、°、%、空格一致），
;; 全角 12px，所以位置可精确算出。把会变的文本拆成"符号 / 整数 / . / 小数 / 单位"几片，
;; 每片取值有限（整数几十种、两位小数 100 种、时分秒各 60 种），驻留表很快饱和。

(define HW 7)                   ; 半角步进（px）

;; 分片绘制时的当前颜色（避免把 r/g/b 一路传参）
(define ink_r 255)
(define ink_g 255)
(define ink_b 255)
(define set_ink (lambda (r g b) { (set! ink_r r) (set! ink_g g) (set! ink_b b) }))

;; 非负定点数的"数字部分"：整数右对齐到 iw 个半角位、小数 dgt 位；返回下一个可用 x。
;; 取整方式与旧版 fixed() 一致（×10^dgt 后四舍五入），避免尾差。
(define draw_num
  (lambda (y x v dgt iw) {
    (define sc (Math.pow 10.0 dgt))
    (define xs (Math.floor (+ (* v sc) 0.5)))
    (define ip (Math.floor (/ xs sc)))
    (define fp (- xs (* ip sc)))
    (define istr (String.atom_to_string ip))
    (define fstr (padz (String.atom_to_string fp) dgt))
    (define ni (String.length istr))
    (define nx x)
    (define k ni)
    (while (< k iw) { (set! k (+ k 1)) (set! nx (+ nx HW)) })   ; 左补空白，使整数右对齐
    (Gfx.draw_text istr nx y ink_r ink_g ink_b)
    (set! nx (+ nx (* HW ni)))
    (if (> dgt 0)
        {
          (Gfx.draw_text "." nx y ink_r ink_g ink_b)
          (set! nx (+ nx HW))
          (Gfx.draw_text fstr nx y ink_r ink_g ink_b)
          (set! nx (+ nx (* HW (String.length fstr))))
        }
        #f)
    nx
  }))

;; 无符号定点数 + 单位（unit 为 "°" / "%" / " AU" 等字面量）
(define draw_fix
  (lambda (y x v dgt iw unit) {
    (define nx (draw_num y x v dgt iw))
    (Gfx.draw_text unit nx y ink_r ink_g ink_b)
    (+ nx (* HW (String.length unit)))
  }))

;; 有符号定点数 + 单位
(define draw_fix_signed
  (lambda (y x v dgt iw unit) {
    (Gfx.draw_text (if (>= v 0.0) "+" "-") x y ink_r ink_g ink_b)
    (draw_fix y (+ x HW) (absv v) dgt iw unit)
  }))

;; (dn, sod) → 分片绘制北京时间 "MM-DD HH:MM"（各片：pad2 的两位数字 + 字面量分隔符）
(define draw_time_bj
  (lambda (y x dn sod) {
    (define ldays dn)
    (define lsod (+ sod Ephem.TZ_SEC))
    (if (>= lsod 86400.0) { (set! lsod (- lsod 86400.0)) (set! ldays (+ ldays 1)) })
    (Ephem.civil ldays)
    (define hn (Math.floor (/ lsod 3600.0)))
    (define mn (Math.floor (/ (- lsod (* hn 3600.0)) 60.0)))
    (Gfx.draw_text (pad2 (Ephem.get_civ_m)) x y ink_r ink_g ink_b)
    (Gfx.draw_text "-" (+ x (* HW 2)) y ink_r ink_g ink_b)
    (Gfx.draw_text (pad2 (Ephem.get_civ_d)) (+ x (* HW 3)) y ink_r ink_g ink_b)
    (Gfx.draw_text (pad2 hn) (+ x (* HW 6)) y ink_r ink_g ink_b)
    (Gfx.draw_text ":" (+ x (* HW 8)) y ink_r ink_g ink_b)
    (Gfx.draw_text (pad2 mn) (+ x (* HW 9)) y ink_r ink_g ink_b)
  }))

;; ---- 公历时刻行（分片：日期串每日才变，时分秒用 pad2，分隔符为字面量）--------
(define X_CLK_DATE LX)          ; "YYYY-MM-DD" 10 字符 = 70px
(define X_CLK_H   83)           ; 时（2 位）
(define X_CLK_C1  97)           ; ":"
(define X_CLK_MI  104)          ; 分（2 位）
(define X_CLK_C2  118)          ; ":"
(define X_CLK_S   125)          ; 秒（2 位）
(define X_CLK_TZ  139)          ; " UTC+8"

(define clock_date
  (lambda () (String.concat
               (String.concat (String.atom_to_string v_y) "-")
               (String.concat (pad2 v_m) (String.concat "-" (pad2 v_d))))))

(define draw_clock
  (lambda () {
    (Gfx.draw_text (clock_date) X_CLK_DATE Y_CLOCK 130 200 255)
    (Gfx.draw_text (pad2 v_h) X_CLK_H Y_CLOCK 130 200 255)
    (Gfx.draw_text ":" X_CLK_C1 Y_CLOCK 130 200 255)
    (Gfx.draw_text (pad2 v_mi) X_CLK_MI Y_CLOCK 130 200 255)
    (Gfx.draw_text ":" X_CLK_C2 Y_CLOCK 130 200 255)
    (Gfx.draw_text (pad2 v_s) X_CLK_S Y_CLOCK 130 200 255)
    (Gfx.draw_text " UTC+8" X_CLK_TZ Y_CLOCK 130 200 255)
  }))

;; 农历组：年表未就绪时显示推算进度 / 结果超出年表范围时的提示
(define draw_lunar_rows
  (lambda ()
    (if (== tbl_ready 0)
        {
          (Gfx.draw_text "推算农历年表中…" LX Y_LUNAR 255 220 120)
          (Gfx.draw_text (String.concat
                           (String.concat (if (== build_kind 0) "朔表 " "中气表 ")
                             (String.concat (String.atom_to_string build_i)
                               (if (== build_kind 0) " / ~26" " / 24")))
                           (String.concat "    覆盖公历年 "
                             (String.atom_to_string build_year)))
                         LX Y_GANZHI 170 176 188)
        }
        {
          (if (== out_ok 1)
              {
                (lbl Y_LUNAR "农历")
                (Gfx.draw_text (String.concat out_ystr out_zodiac) VX Y_LUNAR 255 232 140)
                (Gfx.draw_text (String.concat out_mname out_dname) 96 Y_LUNAR 255 232 140)
                (lbl Y_GANZHI "干支")
                (Gfx.draw_text (String.concat out_ystr
                                 (String.concat "年 " (String.concat out_mgz
                                   (String.concat "月 " (String.concat out_dgz "日")))))
                               VX Y_GANZHI 235 235 170)
                (lbl Y_TERM "中气")
                (Gfx.draw_text (String.concat out_termname (String.concat "  " out_termtime))
                               VX Y_TERM 150 210 255)
                (lbl Y_TERM2 "节气")
                (Gfx.draw_text (get_item term24_names next_term_idx) VX Y_TERM2 150 210 255)
                (set_ink 150 210 255)
                (draw_time_bj Y_TERM2 86 next_term_dn next_term_sod)
                (mfd Y_TERM2 170 "下朔")
                (set_ink 150 210 255)
                (if (== next_nm_ok 1)
                    (draw_time_bj Y_TERM2 198 next_nm_dn next_nm_sod)
                    (Gfx.draw_text "—" 198 Y_TERM2 150 210 255))
                (lbl Y_TABLE "年表")
                (Gfx.draw_text (String.concat "农历 "
                                 (String.concat (String.atom_to_string out_year)
                                   (String.concat " 年（表 "
                                     (String.concat (String.atom_to_string cache_year) "）"))))
                               VX Y_TABLE 140 150 160)
              }
              { (Gfx.draw_text "（本日不在年表范围内）" LX Y_LUNAR 255 120 120) })
        })
  ))

(define draw_screen
  (lambda () {
    (Gfx.clear)
    ;; ---- 标题与公历时刻 ----
    (Gfx.draw_text "星历 · 农历 · 南京 119.0°E 32.0°N" LX Y_TITLE 255 255 255)
    (draw_clock)
    (sep Y_SEP1)
    ;; ---- 太阳 ----
    (lbl Y_SUN "太阳")
    (mfd Y_SUN VX "高度")
    (set_ink 255 232 140)
    (draw_fix_signed Y_SUN 72 v_sun_alt 2 2 "°")
    (mfd Y_SUN 140 "方位")
    (set_ink 255 232 140)
    (draw_fix Y_SUN 168 v_sun_az 2 3 "°")
    ;; ---- 月亮 ----
    (lbl Y_MOON "月亮")
    (mfd Y_MOON VX "高度")
    (set_ink 180 220 255)
    (draw_fix_signed Y_MOON 72 v_moon_alt 2 2 "°")
    (mfd Y_MOON 140 "方位")
    (set_ink 180 220 255)
    (draw_fix Y_MOON 168 v_moon_az 2 3 "°")
    (sep Y_SEP2)
    ;; ---- 月相 ----
    (lbl Y_PHASE "月相")
    (mfd Y_PHASE VX "相位角")
    (set_ink 235 235 235)
    (draw_fix Y_PHASE 84 v_phase 1 3 "°")
    (mfd Y_PHASE 140 "照亮")
    (set_ink 235 235 235)
    (draw_fix Y_PHASE 168 (* 100.0 v_k) 1 3 "%")
    (Gfx.draw_text (phase_name v_elong) 220 Y_PHASE 235 235 160)
    ;; ---- 月盘 ----
    (lbl Y_DISC "月盘")
    (mfd Y_DISC VX "亮肢")
    (set_ink 200 220 255)
    (draw_fix Y_DISC 72 (Ephem.norm360 v_bright_pa) 1 3 "°")
    (mfd Y_DISC 128 "倾斜")
    (set_ink 200 220 255)
    (draw_fix Y_DISC 156 v_tilt 1 3 "°")
    (mfd Y_DISC 206 "北向")
    (set_ink 200 220 255)
    (draw_fix Y_DISC 234 v_north 1 3 "°")
    (sep Y_SEP3)
    ;; ---- 农历（年表未就绪时显示进度）----
    (draw_lunar_rows)
    (sep Y_SEP4)
    ;; ---- 页脚：VSOP87 校对与年表覆盖 ----
    (lbl Y_VSOP "VSOP87")
    (mfd Y_VSOP 52 "黄经")
    (set_ink 150 160 170)
    (draw_fix Y_VSOP 80 v_vsop_lon 3 3 "°")
    (mfd Y_VSOP 146 "日地")
    (set_ink 150 160 170)
    (draw_fix Y_VSOP 174 v_vsop_r 4 1 " AU")
  }))

;; ============================================================================
;; 每拍与启动/退出
;; ============================================================================

;; 整屏重绘：内容 +「退出」按钮 + 推屏。
;; 也被 (EX.install_with ...) 在"留下/后台恢复"之后调用，用来擦掉确认框残影。
(define redraw_all
  (lambda () {
    (draw_screen)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; ---- 每拍（100ms）--------------------------------------------------------

;; 上一次绘制时的 UTC 当日秒（"秒数变了才重算+重绘"）与定时器编号
(define last_sec -1)
(define update_timer 0)

;; 每拍：
;;   ① 退出检查（硬 A/ESC + 右上角按钮）；
;;   ② 年表推算推进（每拍 4 步，约 1.3 s 算完）；
;;   ③ 秒数变了 → 重算星历 + 农历，重绘一次（1 Hz）；
;;   ④ 本拍推进过年表推算 → 重绘（进度条/刚算完的结果，推算期约 10 Hz）。
(define tick
  (lambda () {
    (EX.check)
    (define building (build_batch))
    (define s (Time.utc_sod))
    (define need 0)
    (if (not (== s last_sec))
        {
          (set! last_sec s)
          (astro_compute)
          (lunar_refresh)
          (set! need 1)
        }
        #f)
    (if (== building 1)
        {
          ;; 本拍刚好算完年表 → 立刻补算农历结果，本拍就把数字画出来
          (if (== tbl_ready 1) { (lunar_refresh) })
          (set! need 1)
        }
        #f)
    (if (== need 1) (redraw_all) #f)
  }))

;; ---- 启动 ----------------------------------------------------------------

(define ud0 (Time.utc_days))
(define us0 (Time.utc_sod))
(build_begin (year_start_of ud0 us0))   ; 农历年表在后台分拍推进
(astro_compute)
(draw_screen)
(EX.draw_button)
(Gfx.refresh)
(set! last_sec us0)
(set! update_timer (System.set_interval 100 tick))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 redraw_all 擦掉对话框残影）
(EX.install_with redraw_all)
