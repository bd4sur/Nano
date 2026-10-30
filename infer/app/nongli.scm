;; ============================================================================
;; 农历（Nongli）—— 基于星历模块的农历日期推算 —— 灵机应用程序
;;
;; 复用方式：本 App 是共享星历模块 app/lib/ephem.scm 的**第二个消费者**
;;   （第一个是 ephemeris.scm）。星历算法不在此重复实现，只通过
;;   (import Ephem "app/lib/ephem.scm") 在编译期链接同一份模块，调用其纯计算
;;   接口（Ephem.sun_lon / moon_lon / moon_sun_diff / set_time / civil / ...）。
;;   两个 App 不在运行期互相通信（宿主是单前台 App 模型），详见实现说明。
;;
;; 算法来源：infer/nongli.c（其星历部分来自 ephemeris.c，即 J. Meeus AA）
;;   · 牛顿迭代求朔（月日黄经差 = 0）与中气（太阳黄经 = 330°,0°,30°,…,300°）
;;   · 无中气之月为闰月，闰月序号继承前一个月
;;   · 农历年界：含「雨水」的朔望月为正月，其朔日落在本公历年
;;   · 干支纪年（1984 甲子）/纪月（五虎遁）/纪日（1900-01-31 甲辰为基准）、生肖
;;   · 北京时间民用日编号 cst_day_number 统一判定月首/日序（避免时刻偏差）
;;
;; 时间表示：与星历库一致，用 (dn, sod)（UTC 整数日 + 当日秒），避免 32 位宿主
;;   上大数浮点（儒略日）的精度损失。
;;
;; 工程要点：
;;   · 年表缓存：按 lunar_year_start 缓存「朔表 / 中气表」，跨年重建；
;;   · 分帧推算：首次进入时逐个求解（每步一解），界面显示进度，避免单帧超时；
;;   · 就绪后每秒刷新（只做廉价的查表 + 一次节气求解）。
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

;; ---- 农历常量表（与 nongli.c 一致）----------------------------------------

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
(define out_mstr "")
(define out_dstr "")
(define out_mname "")
(define out_dname "")
(define out_zodiac "")
(define out_mgz "")
(define out_dgz "")
(define out_termname "")
(define out_termtime "")
(define out_nextnm "")
(define out_nextterm "")

;; ---- 小工具 ---------------------------------------------------------------

(define pmod
  (lambda (a n) {
    (define v (mod a n))
    (if (< v 0.0) { (set! v (+ v n)) })
    v
  }))

(define pad2
  (lambda (n) {
    (define v (Math.floor n))
    (if (< v 10.0)
        (String.concat "0" (String.atom_to_string v))
        (String.atom_to_string v))
  }))

;; 时分：由当日秒（可为小数）给出 "HH:MM"
(define hhmm
  (lambda (sod) {
    (define h (Math.floor (/ sod 3600.0)))
    (define mi (Math.floor (/ (- sod (* h 3600.0)) 60.0)))
    (String.concat (pad2 h) (String.concat ":" (pad2 mi)))
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

(define phase 0)                    ; 0 = 推算中，1 = 就绪
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
    (set! phase 0)
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
                (set! phase 1)
              })
        })
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
    (if (and (== phase 1) (== cache_year ystart))
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

;; ---- 绘制 ----------------------------------------------------------------

(define draw_frame
  (lambda () {
    (Gfx.clear)
    (Gfx.draw_text "农历 · 南京 UTC+8" 6 4 255 255 255)
    ;; 公历当前时刻
    (define udays (Time.utc_days))
    (define usod (Time.utc_sod))
    (define ldays udays)
    (define lsod (+ usod Ephem.TZ_SEC))
    (if (>= lsod 86400) { (set! lsod (- lsod 86400)) (set! ldays (+ ldays 1)) })
    (Ephem.civil ldays)
    (Gfx.draw_text (String.concat
                     (String.concat (String.atom_to_string (Ephem.get_civ_y)) "-")
                     (String.concat (pad2 (Ephem.get_civ_m))
                       (String.concat "-"
                         (String.concat (pad2 (Ephem.get_civ_d))
                           (String.concat " " (hhmm lsod))))))
                   6 18 130 200 255)
    (if (== phase 0)
        {
          (Gfx.draw_text "推算年表中…" 6 60 255 220 120)
          (Gfx.draw_text (String.concat (if (== build_kind 0) "朔表 " "中气表 ")
                           (String.concat (String.atom_to_string build_i)
                             (if (== build_kind 0) " / ~26" " / 24")))
                         6 80 200 200 200)
          (Gfx.draw_text (String.concat "覆盖公历年 " (String.atom_to_string build_year))
                         6 96 160 160 160)
        }
        {
          (if (== out_ok 1)
              {
                (Gfx.draw_text (String.concat "农历  " (String.concat out_ystr out_zodiac))
                               6 40 255 232 140)
                (Gfx.draw_text (String.concat "      " (String.concat out_mname out_dname))
                               6 56 255 232 140)
                (Gfx.draw_text (String.concat "干支  " (String.concat out_ystr
                                 (String.concat "年 " (String.concat out_mgz
                                   (String.concat "月 " (String.concat out_dgz "日"))))))
                               6 76 235 235 170)
                (Gfx.draw_text (String.concat "中气  " (String.concat out_termname
                                 (String.concat "  " out_termtime)))
                               6 100 150 210 255)
                (Gfx.draw_text (String.concat "下朔  " (if (== next_nm_ok 1)
                                 (fmt_time_bj next_nm_dn next_nm_sod) "—"))
                               6 118 150 210 255)
                (Gfx.draw_text (String.concat "节气  " (String.concat (get_item term24_names next_term_idx)
                                 (String.concat "  " (fmt_time_bj next_term_dn next_term_sod))))
                               6 136 150 210 255)
                (Gfx.draw_text (String.concat "农历 " (String.concat (String.atom_to_string out_year)
                                 (String.concat " 年 · 年表 " (String.atom_to_string cache_year))))
                               6 160 140 150 160)
              }
              { (Gfx.draw_text "（本日不在年表范围内）" 6 60 255 120 120) })
        })
      (EX.draw_button)
      (Gfx.refresh)
  }))

;; ---- 主循环：分帧推算 → 每秒刷新 -----------------------------------------

(define tick_count 0)

(define display_step
  (lambda () {
    (EX.check)
    (set! tick_count (+ tick_count 1))
    (if (>= tick_count 20)                     ; 约每 20 步（1 秒）刷新一次
        {
          (set! tick_count 0)
          (define udays (Time.utc_days))
          (define usod (Time.utc_sod))
          (define today_cst (Ephem.cst_day_number udays usod))
          (Ephem.civil today_cst)
          (define gy (Ephem.get_civ_y))
          (define gm (Ephem.get_civ_m))
          (define ystart (if (< gm 2) (- gy 1) gy))
          (if (not (== cache_year ystart))
              { (build_begin ystart) }
              {
                (lunar_compute udays usod)
                (find_next_new_moon udays usod)
                (find_next_term udays usod)
              })
          (draw_frame)
        })
  }))

;; ---- 启动 ----------------------------------------------------------------

(define timer_build 0)
(define timer_display 0)

(define build_tick
  (lambda () {
    (if (== phase 1)
        {
          (System.clear_interval timer_build)
          (set! timer_display (System.set_interval 50 display_step))
        }
        { (build_step) (draw_frame) })
  }))

;; 首帧：确定初始农历年并启动分帧推算
(define ud0 (Time.utc_days))
(define us0 (Time.utc_sod))
(define cst0 (Ephem.cst_day_number ud0 us0))
(Ephem.civil cst0)
(define gy0 (Ephem.get_civ_y))
(define gm0 (Ephem.get_civ_m))
(define ystart0 (if (< gm0 2) (- gy0 1) gy0))
(build_begin ystart0)
(draw_frame)
(set! timer_build (System.set_interval 16 build_tick))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 draw_frame 擦掉对话框残影）
(EX.install_with draw_frame)
