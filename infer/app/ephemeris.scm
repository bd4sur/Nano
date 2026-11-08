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
;;   y=3   玲珑仪简版 · 南京 119.0°E 32.0°N            [退出 36x16 @ x=280]
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
;;   y=207 VSOP87 黄经 206.703°   日地 0.9959 AU
;;
;; 排版约定：
;;   · 标签列固定在 x=6（暗灰蓝），字段名（高度/方位/…）中灰，数值按主题着色；
;;   · 数值列左边界对齐（x=44 起；月相/月盘/VSOP 组的数值列按字段名宽度顺延），
;;     数值、日期、时刻一律**分片绘制**（符号/整数/小数/单位各一片，降垃圾量），见下；
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
(native Touch)

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
(define Y_VSOP 207)

;; 分组分隔线
(define Y_SEP1 36)
(define Y_SEP2 79)
(define Y_SEP3 122)
(define Y_SEP4 200)

;; 列：标签列 6；字段值列 44（月相/月盘/VSOP 组按字段名宽度顺延，见各行）
(define LX 6)
(define VX 44)

;; 标题（数据页与天空页共用）
(define TITLE "玲珑仪简版 · 南京 119.0°E 32.0°N")

;; 页签（两页共用，右下角）：[数据][天空]
(define page 0)                 ; 0 = 数据页（文字），1 = 天空页（地平坐标图）
(define page_dirty 0)           ; 1 = 换页后本拍需要重绘
(define TAB_Y 221)
(define TAB_W 36)
(define TAB_H 16)
(define TAB_X0 244)             ; 「数据」
(define TAB_X1 280)             ; 「天空」

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
(define v_ldays 0)                  ; 本地民用日编号（天空页按它缓存周日轨迹）

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
    (set! v_ldays ldays)
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

;; ---- 分片数值绘制（降低每秒垃圾量）----------------------------------------
;;
;; **背景（已由上游修复，此处保留为历史与性能考虑）**：旧解释器把运行期生成的 ≤32 字符
;; 短串**永久驻留**且驻留表是 GC 根（manual §3.4 的实现副作用），所以"每秒变化的文本整串
;; 拼"会持续吃内存——实测（旧 vendor 5f82267，16MB 池、100ms 拍）：整串拼时钟 ≈6 min 即
;; `[Runtime] 指令执行异常`，只修时钟 ≈1 h。上游 771b7c7 改为**弱驻留**（不再当根，GC 后
;; 按轮次戳扫尾 + 表压缩），同一写法已能连跑 12 h；本 App **仍保留**分片绘制，因为每秒少
;; 生成上百个中间串意味着更少的 GC 轮次与更小的 WASM 端停顿（属优化，不再是正确性必需）。
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
              }
              { (Gfx.draw_text "（本日不在年表范围内）" LX Y_LUNAR 255 120 120) })
        })
  ))

(define draw_data
  (lambda () {
    (Gfx.clear)
    ;; ---- 标题与公历时刻 ----
    (Gfx.draw_text TITLE LX Y_TITLE 255 255 255)
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
    (draw_tabs)
  }))


;; ============================================================================
;; 页 1：天空 —— 地平坐标可视化（太阳/月亮实时位置 + 周日轨迹 + 日出日落）
;; ============================================================================
;;
;; 圆盘投影（天顶等距）：
;;   圆心 = 天顶；半径 R 对应高度 0°（地平圈）；半径 R/3、2R/3 为高度 60°、30° 的等高圈。
;;   r = R·(90 − alt)/90，x = CX − r·sin(az)，y = CY − r·cos(az)。
;;   方位按**人抬头看天**的视角：北在上、南在下、东在左、西在右
;;   （与"从天上往下看"的星图左右相反；也正好是躺下头朝北看天的样子）。
;;   两条正交轴就是天球子午圈方向（北-天顶-南）与卯酉圈方向（东-天顶-西）。

(define SKY_CX 160)
(define SKY_CY 124)
(define SKY_R  86)
(define SKY_R60 29)               ; R/3（高度 60° 圈）
(define SKY_R30 57)               ; 2R/3（高度 30° 圈）

(define sky_radius (lambda (alt) (/ (* SKY_R (- 90.0 alt)) 90.0)))
(define sky_x (lambda (az alt) (- SKY_CX (* (sky_radius alt) (Math.sin (Ephem.to_rad az))))))
(define sky_y (lambda (az alt) (- SKY_CY (* (sky_radius alt) (Math.cos (Ephem.to_rad az))))))

;; ---- 太阳周日轨迹 / 日出日落（按本地民用日缓存，**分拍**推进）----------------
;; 一天需 ≈290 次太阳位置求值（144 个样本 + 升落二分细化）。**不能在一拍里做完**：
;; 每次求值都会产生一批短命闭包，290 次叠在一起足以把 16MB 池（VM 8MB + 堆 8MB）
;; 的堆峰值顶到 5MB 以上，进而让同拍的农历年表推算分配失败（实测：点开天空页后
;; `[Runtime] 指令执行异常`）。改为每拍推进 TRK_BATCH 个样本（默认 12 个 = 2 小时），
;; 约 1.3 s 建完，期间已算出的那段轨迹逐拍长出来。
;; 日出日落判据：太阳中心高度 = −0.833°（34′ 大气折射 + 16′ 太阳视半径）。

(define SUN_H0 -0.833)
(define TRK_BATCH 12)             ; 每拍推进的采样数（10 分钟/样本）

(define trk_day -999999)          ; 缓存对应的本地日（v_ldays）
(define trk_building 0)           ; 1 = 正在分拍推算
(define trk_k 0)                  ; 下一个待采样的序号（0..144）
(define trk_prev_alt 99.0)
(define trk_prev_off 0.0)
(define trk_above 0)
(define trk_xs '())               ; 轨迹点 x（预分配 145 个，原地 set_item! 写）
(define trk_ys '())               ; 轨迹点 y（同上）
(define trk_n 0)                  ; 已写入的可见点数
(define trk_rise_x 0)             ; 升/落端点（画在地平圈上，接上弧线两端）
(define trk_rise_y 0)
(define trk_set_x 0)
(define trk_set_y 0)
(define trk_rise_ok 0)
(define trk_set_ok 0)
(define trk_rise_az 0.0)
(define trk_set_az 0.0)
(define trk_rise_str "")
(define trk_set_str "")
(define trk_polar 0)              ; 1 = 全天在地平上，2 = 全天在地平下

;; 本地方午夜对应的 UTC 时刻（写 mid_dn / mid_sod）
(define mid_dn 0)
(define mid_sod 0.0)
(define set_local_midnight
  (lambda (ldays) {
    (normalize_time ldays (- 0 Ephem.TZ_SEC))
    (set! mid_dn nrm_dn)
    (set! mid_sod nrm_sod)
  }))

;; 本地方午夜之后 off 秒处的太阳地平坐标（写 sa_alt / sa_az，返回高度）
(define sa_alt 0.0)
(define sa_az 0.0)
(define sun_at_off
  (lambda (off) {
    (normalize_time mid_dn (+ mid_sod off))
    (Ephem.set_time nrm_dn nrm_sod)
    (Ephem.set_gmst)
    (Ephem.sun_equatorial)
    (Ephem.sun_altaz)
    (set! sa_alt (Ephem.get_alt))
    (set! sa_az (Ephem.get_az))
    sa_alt
  }))

;; 在 [lo, hi] 秒区间内二分求高度穿过 SUN_H0 的时刻（写 cr_off / cr_az）
(define cr_off 0.0)
(define cr_az 0.0)
(define refine_cross
  (lambda (lo hi rising) {
    (define a lo)
    (define b hi)
    (define i 0)
    (define al 0.0)
    (while (< i 14) {
      (set! al (sun_at_off (/ (+ a b) 2.0)))
      (if (if (== rising 1) (< al SUN_H0) (>= al SUN_H0))
          { (set! a (/ (+ a b) 2.0)) }
          { (set! b (/ (+ a b) 2.0)) })
      (set! i (+ i 1))
    })
    (set! cr_off (/ (+ a b) 2.0))
    (sun_at_off cr_off)           ; 取该时刻的方位（写 sa_az）
    (set! cr_az sa_az)
  }))

;; 长度 n 的全零列表（一次 cons 构建，之后用 set_item! 原地写，不再复制）
(define zero_list
  (lambda (n) {
    (define out '())
    (define i 0)
    (while (< i n) { (set! out (cons 0 out)) (set! i (+ i 1)) })
    out
  }))

;; 开始（或重启）某一天的分拍推算
(define trk_begin
  (lambda (ldays) {
    (set! trk_day ldays)
    (set! trk_building 1)
    (set! trk_k 0)
    (set! trk_prev_alt 99.0)
    (set! trk_prev_off 0.0)
    (set! trk_above 0)
    (set! trk_xs (zero_list 145))
    (set! trk_ys (zero_list 145))
    (set! trk_n 0)
    (set! trk_rise_ok 0)
    (set! trk_set_ok 0)
    (set! trk_rise_str "")
    (set! trk_set_str "")
    (set! trk_polar 0)
    (set_local_midnight ldays)
  }))

;; 收尾：接上精确的升/落端点、反转成升序、标注极昼/极夜
(define trk_finish
  (lambda () {
    (if (== trk_rise_ok 1)
        { (set! trk_rise_x (Math.round (sky_x trk_rise_az 0.0)))
          (set! trk_rise_y (Math.round (sky_y trk_rise_az 0.0))) }
        #f)
    (if (== trk_set_ok 1)
        { (set! trk_set_x (Math.round (sky_x trk_set_az 0.0)))
          (set! trk_set_y (Math.round (sky_y trk_set_az 0.0))) }
        #f)
    (if (and (== trk_rise_ok 0) (== trk_set_ok 0))
        { (if (> trk_above 0) (set! trk_polar 1) (set! trk_polar 2)) }
        #f)
    (set! trk_building 0)
  }))

;; 每拍推进一批样本（返回 1 表示本拍动过）
(define trk_step
  (lambda () {
    (define moved 0)
    (define n 0)
    (while (and (== trk_building 1) (< n TRK_BATCH)) {
      (define off (* trk_k 600.0))
      (define cur_alt (sun_at_off off))
      (define cur_az sa_az)
      (if (> trk_k 0)
          {
            (if (and (< trk_prev_alt SUN_H0) (>= cur_alt SUN_H0))
                {
                  (refine_cross trk_prev_off off 1)
                  (set! trk_rise_ok 1)
                  (set! trk_rise_str (hhmm cr_off))
                  (set! trk_rise_az cr_az)
                }
                #f)
            (if (and (>= trk_prev_alt SUN_H0) (< cur_alt SUN_H0))
                {
                  (refine_cross trk_prev_off off 0)
                  (set! trk_set_ok 1)
                  (set! trk_set_str (hhmm cr_off))
                  (set! trk_set_az cr_az)
                }
                #f)
          }
          #f)
      (if (>= cur_alt 0.0)
          {
            (set! trk_above (+ trk_above 1))
            {
              (set_item! trk_xs trk_n (Math.round (sky_x cur_az cur_alt)))
              (set_item! trk_ys trk_n (Math.round (sky_y cur_az cur_alt)))
              (set! trk_n (+ trk_n 1))
            }
          }
          #f)
      (set! trk_prev_alt cur_alt)
      (set! trk_prev_off off)
      (set! trk_k (+ trk_k 1))
      (set! moved 1)
      (if (> trk_k 144) { (trk_finish) } #f)
      (set! n (+ n 1))
    })
    moved
  }))

;; ---- 绘制：天空页 ---------------------------------------------------------

;; 等高圈 + 两条正交轴 + 方位/高度标注
(define sky_grid
  (lambda () {
    (Gfx.draw_circle SKY_CX SKY_CY SKY_R60 56 62 76)
    (Gfx.draw_circle SKY_CX SKY_CY SKY_R30 56 62 76)
    (Gfx.draw_circle SKY_CX SKY_CY SKY_R 92 100 120)         ; 地平圈
    (Gfx.draw_line SKY_CX (- SKY_CY SKY_R) SKY_CX (+ SKY_CY SKY_R) 48 54 66)
    (Gfx.draw_line (- SKY_CX SKY_R) SKY_CY (+ SKY_CX SKY_R) SKY_CY 48 54 66)
    ;; 四正方向（圈外；算术恒二参数，故多层嵌套而不写三参）
    (Gfx.draw_text "北" (- SKY_CX 6) (- (- SKY_CY SKY_R) 15) 150 158 172)
    (Gfx.draw_text "南" (- SKY_CX 6) (+ (+ SKY_CY SKY_R) 3) 150 158 172)
    (Gfx.draw_text "东" (- (- SKY_CX SKY_R) 18) (- SKY_CY 6) 150 158 172)
    (Gfx.draw_text "西" (+ (+ SKY_CX SKY_R) 6) (- SKY_CY 6) 150 158 172)
    ;; 高度标注：放在**上半部分**（沿北向轴线、各圈内侧）
    (Gfx.draw_text "60°" (+ SKY_CX 5) (- (- SKY_CY SKY_R60) 12) 112 120 136)
    (Gfx.draw_text "30°" (+ SKY_CX 5) (- (- SKY_CY SKY_R30) 12) 112 120 136)
  }))

;; 太阳周日轨迹（地平以上部分，暖色细线）
(define draw_sun_track
  (lambda () {
    (define px 0)
    (define py 0)
    (define first 1)
    (if (== trk_rise_ok 1)
        { (set! px trk_rise_x) (set! py trk_rise_y) (set! first 0) }
        #f)
    (define i 0)
    (while (< i trk_n) {
      (define pxx (get_item trk_xs i))
      (define pyy (get_item trk_ys i))
      (if (== first 1)
          { (set! px pxx) (set! py pyy) (set! first 0) }
          {
            (Gfx.draw_line px py pxx pyy 138 106 62)
            (set! px pxx)
            (set! py pyy)
          })
      (set! i (+ i 1))
    })
    (if (if (== trk_set_ok 1) (== first 0) #f)
        { (Gfx.draw_line px py trk_set_x trk_set_y 138 106 62) }
        #f)
  }))

;; 日出/日落标记 + 时刻（写在地平圈外侧、标记附近；越界时夹回屏内）
(define draw_rise_set
  (lambda (az txt r g b) {
    (define hx (sky_x az 0.0))
    (define hy (sky_y az 0.0))
    (define ox (* 16.0 (- 0.0 (Math.sin (Ephem.to_rad az)))))
    (define oy (* 16.0 (- 0.0 (Math.cos (Ephem.to_rad az)))))
    (define tx (- (+ hx ox) 17.0))
    (define ty (- (+ hy oy) 6.0))
    (if (< tx 2.0) (set! tx 2.0) #f)
    (if (> tx 283.0) (set! tx 283.0) #f)
    (if (< ty 22.0) (set! ty 22.0) #f)
    (if (> ty 222.0) (set! ty 222.0) #f)
    (Gfx.fill_circle (Math.round hx) (Math.round hy) 2 r g b)
    (Gfx.draw_text txt (Math.round tx) (Math.round ty) r g b)
  }))

;; 太阳：地平以上画红色圆面；以下在对应方位的地平圈上画空心标记
(define draw_sun
  (lambda () {
    (if (>= v_sun_alt 0.0)
        {
          (Gfx.fill_circle (Math.round (sky_x v_sun_az v_sun_alt))
                           (Math.round (sky_y v_sun_az v_sun_alt)) 5 240 76 52)
          (Gfx.draw_circle (Math.round (sky_x v_sun_az v_sun_alt))
                           (Math.round (sky_y v_sun_az v_sun_alt)) 7 150 62 48)
        }
        {
          (Gfx.draw_circle (Math.round (sky_x v_sun_az 0.0))
                           (Math.round (sky_y v_sun_az 0.0)) 4 146 78 62)
        })
  }))

;; 亮瓣朝向：图面上"亮侧"的单位向量（limb_ux, limb_uy，屏幕坐标 x 右 / y 下）
;;
;; 做法：取太阳相对月亮的天球切向分量——沿"高度增大"方向 dalt、沿"方位增大"方向
;; daz·cos(malt)——再乘图面投影的**局部尺度**换成屏幕位移：
;;   · 高度方向 → 径向（朝圆心），每度 R/90 像素；
;;   · 方位方向 → 切向，每度 ρ·π/180 像素（ρ = 该点的图面半径）。
;; 这样得到的是"站在该点看月亮"的真实亮瓣朝向经同一投影映射后的结果（与库的
;; bright_pa/tilt_north 同源；用有限差分核对过，差 < 0.5°）。
(define limb_ux 0.0)
(define limb_uy -1.0)
(define compute_limb_dir
  (lambda () {
    (define dalt (- v_sun_alt v_moon_alt))
    (define daz (Ephem.norm360 (- v_sun_az v_moon_az)))
    (if (> daz 180.0) (set! daz (- daz 360.0)) #f)
    (define we (* daz (Math.cos (Ephem.to_rad v_moon_alt))))
    (define rho (sky_radius v_moon_alt))
    (define kr (/ SKY_R 90.0))
    (define kt (* rho (/ 3.141592653589793 180.0)))
    (define azr (Ephem.to_rad v_moon_az))
    (define gx (- (* (* kr dalt) (Math.sin azr)) (* (* kt we) (Math.cos azr))))
    (define gy (+ (* (* kr dalt) (Math.cos azr)) (* (* kt we) (Math.sin azr))))
    ;; 自屏幕上方顺时针的角度 → 单位向量
    (define ang (Ephem.to_deg (Math.atan2 gx (- 0.0 gy))))
    (set! limb_ux (Math.sin (Ephem.to_rad ang)))
    (set! limb_uy (- 0.0 (Math.cos (Ephem.to_rad ang))))
  }))

;; 月盘：暗部深灰、亮部月白；明暗界线为投圆的半椭圆
;;   局部坐标 u 沿亮侧、v 垂直于它；相位角 i（0=满月，180=新月），r 为盘半径：
;;   盘内（u²+v² ≤ r²）且 u ≥ −cos(i)·√(r²−v²) ⇒ 亮部。
(define draw_moon_disc
  (lambda (mx my r) {
    (compute_limb_dir)
    (define ci (Math.cos (Ephem.to_rad v_phase)))
    (define ux limb_ux)
    (define uy limb_uy)
    (define vx (- 0.0 uy))                ; 局部 v 轴（垂直于 u）
    (define vy ux)
    (define rr (* r r))
    (Gfx.fill_circle mx my r 52 54 62)    ; 暗部：整盘深灰
    ;; 亮部：**按屏幕像素**逐点反算局部坐标（避免按局部栅格正算时旋转造成的漏点）
    (define dy (- 0 r))
    (while (<= dy r) {
      (define py (+ my dy))
      (define dx (- 0 r))
      (while (<= dx r) {
        (define u (+ (* dx ux) (* dy uy)))
        (define v (+ (* dx vx) (* dy vy)))
        (define q (- rr (* v v)))
        (if (>= q 0.0)
            {
              (define e (Math.sqrt q))          ; 该行的盘面半宽
              (if (if (>= u (* (- 0.0 ci) e)) (<= u e) #f)
                  { (Gfx.set_pixel (+ mx dx) py 250 248 214) }
                  #f)
            }
            #f)
        (set! dx (+ dx 1))
      })
      (set! dy (+ dy 1))
    })
    (Gfx.draw_circle mx my r 96 98 110)   ; 盘缘
  }))

;; 月亮：地平以上按月相绘制；以下画空心标记
(define draw_moon
  (lambda () {
    (define mx (Math.round (sky_x v_moon_az v_moon_alt)))
    (define my (Math.round (sky_y v_moon_az v_moon_alt)))
    (if (>= v_moon_alt 0.0)
        (draw_moon_disc mx my 7)
        {
          (Gfx.draw_circle (Math.round (sky_x v_moon_az 0.0))
                           (Math.round (sky_y v_moon_az 0.0)) 4 104 106 118)
        })
  }))

;; 指标栏（要求：太阳高/方在左下、月亮高/方在右下、月亮相位在右上；至少含当前时间）
(define draw_sky_info
  (lambda () {
    ;; 标题（与数据页一致）
    (Gfx.draw_text TITLE LX Y_TITLE 255 255 255)
    ;; 左上：日期 / 时刻 / 时区
    (Gfx.draw_text (clock_date) 3 24 130 200 255)
    (Gfx.draw_text (pad2 v_h) 3 38 205 225 245)
    (Gfx.draw_text ":" 17 38 205 225 245)
    (Gfx.draw_text (pad2 v_mi) 24 38 205 225 245)
    (Gfx.draw_text ":" 38 38 205 225 245)
    (Gfx.draw_text (pad2 v_s) 45 38 205 225 245)
    (Gfx.draw_text "UTC+8" 3 52 118 136 160)
    (if (== trk_polar 1) { (Gfx.draw_text "极昼" 3 66 150 158 172) } #f)
    (if (== trk_polar 2) { (Gfx.draw_text "极夜" 3 66 150 158 172) } #f)
    ;; 右上：月亮相位参数（标题/数值分行，避免"照亮 100.0%"这类宽度溢出 316）
    (Gfx.draw_text "月相" 250 24 150 158 172)
    (Gfx.draw_text "照亮" 250 38 150 158 172)
    (set_ink 235 235 235)
    (draw_fix 52 250 (* 100.0 v_k) 1 3 "%")
    (Gfx.draw_text "相位角" 250 68 150 158 172)
    (set_ink 235 235 235)
    (draw_fix 82 250 v_phase 1 3 "°")
    (Gfx.draw_text (phase_name v_elong) 250 98 235 235 160)
    ;; 左下：太阳高度 / 方位
    (Gfx.draw_text "太阳" 3 180 255 205 70)
    (Gfx.draw_text "高" 3 194 150 158 172)
    (set_ink 255 232 140)
    (draw_fix_signed 194 23 v_sun_alt 1 2 "°")
    (Gfx.draw_text "方" 3 208 150 158 172)
    (set_ink 255 232 140)
    (draw_fix 208 23 v_sun_az 1 3 "°")
    ;; 右下：月亮高度 / 方位
    (Gfx.draw_text "月亮" 250 180 130 200 255)
    (Gfx.draw_text "高" 250 194 150 158 172)
    (set_ink 180 220 255)
    (draw_fix_signed 194 270 v_moon_alt 1 2 "°")
    (Gfx.draw_text "方" 250 208 150 158 172)
    (set_ink 180 220 255)
    (draw_fix 208 270 v_moon_az 1 3 "°")
  }))

(define draw_sky
  (lambda () {
    (if (not (== trk_day v_ldays)) { (trk_begin v_ldays) } #f)
    (Gfx.clear)
    (draw_sky_info)
    (sky_grid)
    (draw_sun_track)
    (if (== trk_rise_ok 1) { (draw_rise_set trk_rise_az trk_rise_str 232 186 96) } #f)
    (if (== trk_set_ok 1) { (draw_rise_set trk_set_az trk_set_str 232 156 96) } #f)
    (draw_sun)
    (draw_moon)
    (draw_tabs)
  }))

;; ============================================================================
;; 页签（两页共用，右下角 [数据][天空]；点按切换）
;; ============================================================================

(define draw_tab
  (lambda (x label on) {
    (if (== on 1)
        { (Gfx.fill_rect x TAB_Y TAB_W TAB_H 46 92 140)
          (Gfx.draw_rect x TAB_Y TAB_W TAB_H 130 180 230)
          (Gfx.draw_text label (+ x 6) (+ TAB_Y 2) 245 250 255) }
        { (Gfx.fill_rect x TAB_Y TAB_W TAB_H 40 42 52)
          (Gfx.draw_rect x TAB_Y TAB_W TAB_H 96 100 116)
          (Gfx.draw_text label (+ x 6) (+ TAB_Y 2) 150 156 172) })
  }))

(define draw_tabs
  (lambda () {
    (if (== page 0)
        { (draw_tab TAB_X0 "数据" 1) (draw_tab TAB_X1 "天空" 0) }
        { (draw_tab TAB_X0 "数据" 0) (draw_tab TAB_X1 "天空" 1) })
  }))

;; 点按落在哪枚页签上：0 / 1，都不在返回 -1
(define tab_at
  (lambda (px py)
    (if (and (>= py TAB_Y) (< py (+ TAB_Y TAB_H)))
        (if (and (>= px TAB_X0) (< px (+ TAB_X0 TAB_W))) 0
            (if (and (>= px TAB_X1) (< px (+ TAB_X1 TAB_W))) 1 -1))
        -1)))

(define select_page
  (lambda (p)
    (if (not (== p page))
        { (set! page p) (set! page_dirty 1) }
        #f)))

;; 每拍一次：只消费「按下」沿（EX.check 已先跑，落在退出按钮上的沿由它吞掉）。
;; tab_at 的越界返回值 -1 必须挡掉，否则会切到不存在的页。
(define tab_poll
  (lambda ()
    (if (Touch.take_press)
        {
          (define t (tab_at (Touch.x) (Touch.y)))
          (if (>= t 0) (select_page t) #f)
        }
        #f)))

;; 页分发
(define draw_page
  (lambda ()
    (if (== page 0) (draw_data) (draw_sky))))

;; ============================================================================
;; 每拍与启动/退出
;; ============================================================================

;; 整屏重绘：内容 +「退出」按钮 + 推屏。
;; 也被 (EX.install_with ...) 在"留下/后台恢复"之后调用，用来擦掉确认框残影。
(define redraw_all
  (lambda () {
    (draw_page)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; ---- 每拍（100ms）--------------------------------------------------------

;; 上一次绘制时的 UTC 当日秒（"秒数变了才重算+重绘"）与定时器编号
(define last_sec -1)
(define update_timer 0)

;; 每拍：
;;   ① 退出检查（硬 A/ESC + 右上角按钮）；
;;   ② 页签点按 → 换页（置 page_dirty）；
;;   ③ 年表推算推进（每拍 4 步，约 1.3 s 算完；进度只在数据页显示）；
;;   ④ 秒数变了 / 换了页 / 数据页在推算中 → 重绘一次（1 Hz，推算期约 10 Hz）。
(define tick
  (lambda () {
    (EX.check)
    (tab_poll)
    (define trk_moved (if (== page 1) (trk_step) 0))
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
    (if (== page_dirty 1)
        {
          (set! page_dirty 0)
          (set! need 1)
        }
        #f)
    (if (== building 1)
        {
          ;; 本拍刚好算完年表 → 立刻补算农历结果，本拍就把数字画出来
          (if (== tbl_ready 1) { (lunar_refresh) })
          (if (== page 0) { (set! need 1) })
        }
        #f)
    ;; 天空页：轨迹推算每拍推进一批 → 逐拍重绘（弧线长出来的过程可见）
    (if (== trk_moved 1) { (set! need 1) } #f)
    (if (== need 1) (redraw_all) #f)
  }))

;; ---- 启动 ----------------------------------------------------------------

(define ud0 (Time.utc_days))
(define us0 (Time.utc_sod))
(build_begin (year_start_of ud0 us0))   ; 农历年表在后台分拍推进
(astro_compute)
(draw_page)
(EX.draw_button)
(Gfx.refresh)
(set! last_sec us0)
(set! update_timer (System.set_interval 100 tick))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 redraw_all 擦掉对话框残影）
(EX.install_with redraw_all)
