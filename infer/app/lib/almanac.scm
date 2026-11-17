;; ============================================================================
;; almanac.scm —— 黄历（农历择日）计算核心：C 版 cnlunar（infer/almanac.h）的灵机移植
;;
;; 目的：app/calendar.scm 的黄历模态框需要与 C 版 ui_almanac.c **逐字节一致**的结果。
;; 本模块只做计算、不碰图形层，因此可以在桌面解释器 / wasm32 探针里跑差分回归：
;;
;;     infer/tools/gen_almanac_tables.c emit > app/lib/almanac_data.scm   ; 数据表
;;     python3 tools/almanac_difftest.py                                  ; 差分回归
;;
;;   (import AL "app/lib/almanac.scm")
;;   (AL.calc 2026 3 9 12 30)   ; => 结果表 | #f（越界）
;;
;; 结果表字段下标见下方 R_* 常量，取值方式见访问器（AL.bazi 等）。
;;
;; 移植原则（**逐函数对照 almanac.h**，不做"等价改写"）：
;;   · 数据表全部由生成器从 C 头文件导出（almanac_data.scm），本文件不硬编码任何
;;     黄历词条；48 位节气表/位段农历表/16 位时辰位图已在生成时解码为小整数。
;;   · 列表语义照抄 C：ladd = 缺席才追加、ldedup = 保序集去重、
;;     lsort = 按 THINGS_SORT 的稳定插入排序；顺序参与回归比对。
;;   · 灵机约束（manual.md）：算术恒二参数、无一元负号、无 let/list/map/append、
;;     cond 每子句单表达式；同一 lambda 区域内不得重名 define（实测解析器会拒绝），
;;     故每个函数把临时量集中在函数体开头声明，循环只 set!。
;;   · 32 位宿主（wasm32）浮点尾数仅约 19 位，故**整数除法/取余一律用 idiv/imod**
;;     （纯整数长除法，27 位整数载荷内精确），不用 `/`、`mod`。
;;   · C 里 options 恒为 CNLUNAR_DEFAULT（ui_almanac_open 的调用），故：
;;     x（立春年柱偏移）=0、isYeargodDuty=1、men=month_earth_num。
;;   · 本地库（String/Math）一律在 lambda 体内以 `库.过程` 调用形式使用：
;;     把 native 过程当值绑定（如 (define slen String.length)）在**被 import 的模块**
;;     里会解析失败（实测），故下面用薄包装。
;; ============================================================================

(native Math)
(native String)

(import A "app/lib/almanac_data.scm")


;; ---- 结果表字段下标 --------------------------------------------------------
(define R_BAZI 0)       ;; 八字四柱连写（年+月+日+时）
(define R_OVERALL 1)    ;; 当日吉凶 大吉/吉/平/凶
(define R_WEEK 2)       ;; 星期一..星期日
(define R_ZODIAC 3)     ;; 星座
(define R_STAR28 4)     ;; 廿八宿
(define R_OFFICER 5)    ;; 建除十二神
(define R_GOD 6)        ;; 十二值神（青龙..勾陈）
(define R_MER 7)        ;; 子午流注当令器官
(define R_MERYI 8)
(define R_MERJI 9)
(define R_MERNOTE 10)
(define R_GOOD 11)      ;; 宜（列表）
(define R_BAD 12)       ;; 忌（列表）
(define R_FLY 13)       ;; 九宫飞星 9 位数字串
(define R_FETAL 14)     ;; 胎神
(define R_PENG 15)      ;; 彭祖百忌（两段逗号连接）
(define R_HOURLUCK 16)  ;; 13 项 吉/凶
(define R_DIRS 17)      ;; 喜神/财神/福神/阳贵/阴贵 方位（5 项）
(define R_LUNAR 18)     ;; (农历年 月 日 是否闰月)
(define R_NEXTSOLAR 19) ;; 下一节气序号 0..23
;; ---- 调试/回归用（GUI 不显示）：神煞名与等第 ----
(define R_GOODGODS 20)
(define R_BADGODS 21)
(define R_TODAYLEVEL 22)
(define R_TLEV 23)

;; ---- 结果表访问器（GUI / 测试用） -----------------------------------------
(define bazi (lambda (R) (get_item R R_BAZI)))
(define overall (lambda (R) (get_item R R_OVERALL)))
(define week (lambda (R) (get_item R R_WEEK)))
(define zodiac (lambda (R) (get_item R R_ZODIAC)))
(define star28 (lambda (R) (get_item R R_STAR28)))
(define officer (lambda (R) (get_item R R_OFFICER)))
(define god (lambda (R) (get_item R R_GOD)))
(define meridian (lambda (R) (get_item R R_MER)))
(define meridian_yi (lambda (R) (get_item R R_MERYI)))
(define meridian_ji (lambda (R) (get_item R R_MERJI)))
(define meridian_note (lambda (R) (get_item R R_MERNOTE)))
(define good (lambda (R) (get_item R R_GOOD)))
(define bad (lambda (R) (get_item R R_BAD)))
(define fly (lambda (R) (get_item R R_FLY)))
(define fetal (lambda (R) (get_item R R_FETAL)))
(define peng (lambda (R) (get_item R R_PENG)))
(define hourluck (lambda (R) (get_item R R_HOURLUCK)))
(define dirs (lambda (R) (get_item R R_DIRS)))
(define lunar (lambda (R) (get_item R R_LUNAR)))
(define next_solar (lambda (R) (get_item R R_NEXTSOLAR)))
;; 神煞名表（GUI 展示用：吉神宜趋 / 凶煞宜忌）
(define goodgods (lambda (R) (get_item R R_GOODGODS)))
(define badgods (lambda (R) (get_item R R_BADGODS)))

;; ctx 字段下标（一次算好的公共中间量，供神煞匹配等函数共享）
(define C_YEAR 0)
(define C_MONTH 1)
(define C_DAY 2)
(define C_HOUR 3)
(define C_MIN 4)
(define C_THNUM 5)      ;; 时辰序号 0..12
(define C_LY 6)
(define C_LM 7)
(define C_LDAY 8)
(define C_LEAP 9)
(define C_MD0 10)
(define C_LEAPM 11)
(define C_LEAPD 12)
(define C_ISLONG 13)
(define C_PHASE 14)
(define C_TERM 15)
(define C_NSN 16)
(define C_NSY 17)       ;; 下一节气所属年
(define C_TERMS 18)     ;; 今年 24 节气 (月 日)
(define C_Y8 19)
(define C_M8 20)
(define C_D8S 21)       ;; 日柱
(define C_H8 22)
(define C_DHE 23)       ;; 日柱六十甲子序号
(define C_YEAR_EARTH 24)
(define C_MONTH_EARTH 25)
(define C_DAY_EARTH 26)
(define C_YEAR_HEAVEN 27)
(define C_MONTH_HEAVEN 28)
(define C_DAY_HEAVEN 29)
(define C_SEASON_TYPE 30)
(define C_SEASON_NUM 31)
(define C_MEN 32)
(define C_OFFC 33)
(define C_DAYGOD 34)
(define C_STAR28 35)
(define C_WEEK 36)
(define C_ZODIAC 37)
(define C_D8 38)        ;; 四柱连写（strstr 用）
(define C_TMD_M 39)     ;; 次日（月）
(define C_TMD_D 40)     ;; 次日（日）
(define C_TUIDI_DAYS 41);; 土王用事天数
(define C_T4J 42)       ;; 四立
(define C_T4L 43)       ;; 二分二至

;; ============================================================================
;; 小工具
;; ============================================================================

(define newl (lambda () (cdr '(0))))   ;; 新建可变空表（'() 是静态对象，不能就地 push）

(define slen (lambda (s) (String.length s)))
(define scat (lambda (a b) (String.concat a b)))
(define seq (lambda (a b) (String.equals a b)))
(define sch (lambda (s i) (String.charAt s i)))

;; 子串判定（cnl_bytes_in 的码点等价物；UTF-8 边界内两者等价）
(define substr? (lambda (hay needle)
    (define hn (slen hay))
    (define nn (slen needle))
    (define i 0)
    (define found 0)
    (if (== nn 0) (set! found 1) #f)
    (while (and (== found 0) (<= (+ i nn) hn))
        (if (seq (String.slice hay i (+ i nn)) needle) (set! found 1) #f)
        (set! i (+ i 1)))
    (== found 1)))

;; 前 n 个码点（cnl_utf8_take）
(define stak (lambda (s n) (if (<= (slen s) n) s (String.slice s 0 n))))

;; 整数除法（非负 a、正 b）：纯整数二进制长除法，避免 32 位宿主浮点丢精度
(define idiv (lambda (a b)
    (define q 0)
    (define ns (newl))
    (define n 1)
    (define p b)
    (define r 0)
    (define i 0)
    (define nn 0)
    (if (< a b)
        (set! q 0)
        (begin
          (while (<= (* 2 p) a)
              (push ns n)
              (set! n (* 2 n))
              (set! p (* 2 p)))
          (push ns n)
          (set! r a)
          (set! i (- (length ns) 1))
          (while (>= i 0)
              (set! nn (get_item ns i))
              (if (>= r (* b nn))
                  (begin (set! r (- r (* b nn))) (set! q (+ q nn)))
                  #f)
              (set! i (- i 1)))))
    q))

;; 取余，结果非负（cnl_mod：C 的 % 截断语义再补正）
(define imod (lambda (a n)
    (define q 0)
    (define r 0)
    (if (>= a 0) (set! q (idiv a n)) (set! q (- 0 (idiv (- 0 a) n))))
    (set! r (- a (* q n)))
    (if (< r 0) (set! r (+ r n)) #f)
    r))

;; 向下取整除法（C 的 tsec/86400 再对负数补 1）
(define idiv_floor (lambda (a b)
    (define q 0)
    (define rem 0)
    (if (>= a 0)
        (set! q (idiv a b))
        (begin
          (set! q (- 0 (idiv (- 0 a) b)))
          (set! rem (imod a b))
          (if (not (== rem 0)) (set! q (- q 1)) #f)))
    q))

;; 列表操作（就地修改调用方自有的可变列表；语义对齐 C 的 cnl_l_*）
(define lhas (lambda (L v)
    (define n (length L))
    (define i 0)
    (define f 0)
    (while (< i n)
        (if (equal? (get_item L i) v) (set! f 1) #f)
        (set! i (+ i 1)))
    (== f 1)))

(define ladd! (lambda (L v) (if (lhas L v) #f (push L v))))
(define lreset! (lambda (L) (while (> (length L) 0) (pop L) #f)))

(define lrm! (lambda (L v)
    (define n (length L))
    (define i 0)
    (define w 0)
    (define x 0)
    (while (< i n)
        (set! x (get_item L i))
        (if (equal? x v)
            #f
            (begin
              (if (< w i) (set_item! L w x) #f)
              (set! w (+ w 1))))
        (set! i (+ i 1)))
    (while (> (length L) w) (pop L) #f)))

(define lunion_arr! (lambda (L arr n)
    (define i 0)
    (while (< i n)
        (ladd! L (get_item arr i))
        (set! i (+ i 1)))))

(define lrm_arr! (lambda (L arr n)
    (define i 0)
    (while (< i n)
        (lrm! L (get_item arr i))
        (set! i (+ i 1)))))

(define lunion_from! (lambda (L src)
    (define n (length src))
    (define i 0)
    (while (< i n)
        (ladd! L (get_item src i))
        (set! i (+ i 1)))))

(define lrm_list! (lambda (L src)
    (define n (length src))
    (define i 0)
    (while (< i n)
        (lrm! L (get_item src i))
        (set! i (+ i 1)))))

;; 保序集去重（保留首个出现）
(define ldedup! (lambda (L)
    (define n (length L))
    (define i 0)
    (define w 0)
    (define x 0)
    (define j 0)
    (define dup 0)
    (while (< i n)
        (set! x (get_item L i))
        (set! j 0)
        (set! dup 0)
        (while (< j w)
            (if (equal? (get_item L j) x) (set! dup 1) #f)
            (set! j (+ j 1)))
        (if (== dup 0)
            (begin
              (if (< w i) (set_item! L w x) #f)
              (set! w (+ w 1)))
            #f)
        (set! i (+ i 1)))
    (while (> (length L) w) (pop L) #f)))

;; 宜忌排序键：命中 THINGS_SORT 取下标，否则 38（C 的兜底键）
(define things_key (lambda (s)
    (define n (length A.THINGS_SORT))
    (define i 0)
    (define k 38)
    (while (< i n)
        (if (seq (get_item A.THINGS_SORT i) s) (begin (set! k i) (set! i n)) (set! i (+ i 1)))
        #f)
    k))

;; 稳定插入排序（cnl_l_sort_stable）
(define lsort! (lambda (L)
    (define n (length L))
    (define i 1)
    (define x 0)
    (define k 0)
    (define j 0)
    (while (< i n)
        (set! x (get_item L i))
        (set! k (things_key x))
        (set! j i)
        (while (and (> j 0) (> (things_key (get_item L (- j 1))) k))
            (set_item! L j (get_item L (- j 1)))
            (set! j (- j 1)))
        (set_item! L j x)
        (set! i (+ i 1)))))

(define mk2 (lambda (a b) (cons a (cons b (newl)))))
(define mk3 (lambda (a b c) (cons a (cons b (cons c (newl))))))
(define mk4 (lambda (a b c d) (cons a (cons b (cons c (cons d (newl)))))))
(define mk5 (lambda (a b c d e)
    (cons a (cons b (cons c (cons d (cons e (newl))))))))
(define mk6 (lambda (a b c d e f)
    (cons a (cons b (cons c (cons d (cons e (cons f (newl)))))))))
(define mk7 (lambda (a b c d e f g)
    (cons a (cons b (cons c (cons d (cons e (cons f (cons g (newl))))))))))

;; ============================================================================
;; 日期 / 干支基础
;; ============================================================================

;; days since 1970-01-01（Hinnant 算法，cnl_daynum）
(define daynum (lambda (y m d)
    (define yy (if (<= m 2) (- y 1) y))
    (define era (idiv (if (>= yy 0) yy (- yy 399)) 400))
    (define yoe (- yy (* era 400)))
    (define m3 (if (> m 2) (- m 3) (+ m 9)))
    (define doy (+ (idiv (+ (* 153 m3) 2) 5) (- d 1)))
    (define doe (+ (+ (* yoe 365) (idiv yoe 4))
                   (+ (- 0 (idiv yoe 100)) doy)))
    (- (+ (* era 146097) doe) 719468)))

;; 分解回 (年 月 日)（cnl_ymd）
(define ymd (lambda (days)
    (define ds (+ days 719468))
    (define era (idiv (if (>= ds 0) ds (- ds 146096)) 146097))
    (define doe (- ds (* era 146097)))
    (define yoe (idiv (+ (+ (- doe (idiv doe 1460)) (idiv doe 36524))
                         (- 0 (idiv doe 146096)))
                      365))
    (define yy (+ yoe (* era 400)))
    (define doy (- doe (+ (+ (* 365 yoe) (idiv yoe 4))
                          (- 0 (idiv yoe 100)))))
    (define mp (idiv (+ (* 5 doy) 2) 153))
    (define dd (+ (- doy (idiv (+ (* 153 mp) 2) 5)) 1))
    (define mm (+ mp (if (< mp 10) 3 (- 0 9))))
    (define yy2 (+ yy (if (<= mm 2) 1 0)))
    (mk3 yy2 mm dd)))

;; 六十甲子（cnl_g60_at）
(define g60_at (lambda (k) (get_item A.G60 (imod k 60))))

;; 八字字符串的干/支序号（cnl_stem_num_of8 / cnl_branch_num_of8）
(define stem_num_of8 (lambda (s8)
    (define i 0)
    (define r 0)
    (define c (sch s8 0))
    (while (< i 10)
        (if (seq (get_item A.STEM10 i) c) (begin (set! r i) (set! i 10)) (set! i (+ i 1)))
        #f)
    r))
(define branch_num_of8 (lambda (s8)
    (define i 0)
    (define r 0)
    (define c (sch s8 1))
    (while (< i 12)
        (if (seq (get_item A.BRANCH12 i) c) (begin (set! r i) (set! i 12)) (set! i (+ i 1)))
        #f)
    r))

;; 某农历年某月的大小（cnl_month_leap_days）：返回 (月天数 闰月号 闰月天数)
;; 数据行 = 12 个月大小 + 闰月号 + 闰月大小。lm 理论上可到 13，此时 C 用 bit12
;; （闰月大小位）当"本月天数"，此处照样复现。
(define month_leap_days (lambda (ly lm)
    (define idx (- ly 1901))
    (define row #f)
    (define md0 29)
    (if (< idx 0) (set! idx 198) #f)
    (if (>= idx 199)
        #f
        (begin
          (set! row (get_item A.LUNAR idx))
          (if (== lm 13)
              (if (== (get_item row 13) 30) (set! md0 30) (set! md0 29))
              (set! md0 (get_item row (- lm 1))))
          (mk3 md0 (get_item row 12) (get_item row 13))))))

;; 农历日期换算（cnl_lunar_date_num）
;; 返回 (农历年 月 日 是否闰月 本月天数 闰月号 闰月天数 距春节天数)
(define lunar_date (lambda (y m d)
    (define ly y)
    (define lm 1)
    (define isleap 0)
    (define nyidx (* (- y 1901) 2))
    (define ref (daynum y (get_item A.NEWYEAR nyidx) (get_item A.NEWYEAR (+ nyidx 1))))
    (define span (- (daynum y m d) ref))
    (define row #f)
    (define md0 0)
    (define leapm 0)
    (define leapd 0)
    (define stop 0)
    (define out #f)
    (if (>= span 0)
        (begin
          (set! row (month_leap_days ly lm))
          (if (== row #f) (set! stop 2) #f)
          (if (== stop 2)
              #f
              (begin
                (set! md0 (get_item row 0))
                (set! leapm (get_item row 1))
                (set! leapd (get_item row 2))))
          (while (and (== stop 0) (>= span md0))
              (set! span (- span md0))
              (if (== lm leapm)
                  (begin
                    (set! md0 leapd)
                    (if (< span md0)
                        (begin (set! isleap 1) (set! stop 1))
                        (set! span (- span md0))))
                  #f)
              (if (== stop 0)
                  (begin
                    (set! lm (+ lm 1))
                    (set! row (month_leap_days ly lm))
                    (if (== row #f)
                        (set! stop 2)
                        (begin
                          (set! md0 (get_item row 0))
                          (set! leapm (get_item row 1))
                          (set! leapd (get_item row 2)))))
                  #f))
          (if (== stop 2) #f (set! out (mk7 ly lm (+ 1 span) isleap md0 leapm leapd))))
        (begin
          (set! lm 12)
          (set! ly (- ly 1))
          (set! row (month_leap_days ly lm))
          (if (== row #f) (set! stop 2) #f)
          (if (== stop 2)
              #f
              (begin
                (set! md0 (get_item row 0))
                (set! leapm (get_item row 1))
                (set! leapd (get_item row 2))))
          (while (and (== stop 0) (> (Math.abs span) md0))
              (set! span (+ span md0))
              (set! lm (- lm 1))
              (if (== lm leapm)
                  (begin
                    (set! md0 leapd)
                    (if (<= (Math.abs span) md0)
                        (begin (set! isleap 1) (set! stop 1))
                        (set! span (+ span md0))))
                  #f)
              (if (== stop 0)
                  (begin
                    (set! row (month_leap_days ly lm))
                    (if (== row #f)
                        (set! stop 2)
                        (begin
                          (set! md0 (get_item row 0))
                          (set! leapm (get_item row 1))
                          (set! leapd (get_item row 2)))))
                  #f))
          (if (== stop 2)
              #f
              (set! out (mk7 ly lm (+ md0 (+ span 1)) isleap md0 leapm leapd)))))
    out))

;; 农历月是否大月 + 月相（cnl_lunar_month_cn 的 long 位 + cnl_phase_of_moon）
;; 返回 (是否大月 月相)
(define month_long_phase (lambda (lday isleap md0 mdleap)
    (define this_days (if (== isleap 1) mdleap md0))
    (define islong (if (>= this_days 30) 1 0))
    (define ph "")
    (if (== (- lday islong) 15) (set! ph "望")
        (if (== lday 1) (set! ph "朔")
            (if (and (>= lday 7) (<= lday 8)) (set! ph "上弦")
                (if (and (>= lday 22) (<= lday 23)) (set! ph "下弦") #f))))
    (mk2 islong ph)))

;; ============================================================================
;; 二十四节气
;; ============================================================================

;; 某公历年 24 节气 (月 日) 列表（cnl_year_solar_terms）
(define year_solar (lambda (year)
    (define base (* (- year 1901) 24))
    (define L (newl))
    (define k 0)
    (while (< k 24)
        (push L (mk2 (+ (idiv k 2) 1)
                     (+ (get_item A.ENC_VECTOR k) (get_item A.SOLAR (+ base k)))))
        (set! k (+ k 1)))
    L))

;; 节气比较（cnlunar_md_le）
(define md_le (lambda (a b)
    (define am (get_item a 0))
    (define bm (get_item b 0))
    (define ad (get_item a 1))
    (define bd (get_item b 1))
    (or (< am bm) (and (== am bm) (<= ad bd)))))

;; 返回 (今日节气名 下一节气序号 下一节气所属年 本年24节气表)
(define today_solar_terms (lambda (y m d)
    (define terms (year_solar y))
    (define find (mk2 m d))
    (define cnt 0)
    (define i 0)
    (define next_num 0)
    (define today_name "无")
    (define next_year y)
    (define t #f)
    (while (< i 24)
        (if (md_le (get_item terms i) find) (set! cnt (+ cnt 1)) #f)
        (set! i (+ i 1)))
    (set! next_num (imod cnt 24))
    (set! i 0)
    (while (< i 24)
        (set! t (get_item terms i))
        (if (and (== (get_item t 0) m) (== (get_item t 1) d))
            (begin (set! today_name (get_item A.SOLAR_TERMS_NAME i)) (set! i 24))
            (set! i (+ i 1))))
    (set! t (get_item terms 23))
    (if (and (== (get_item t 0) m) (>= d (get_item t 1)))
        (set! next_year (+ y 1))
        #f)
    (mk4 today_name next_num next_year terms)))

;; ============================================================================
;; 各字段计算
;; ============================================================================

;; 八字 + 干支序号 + 季节（options=0 → 年柱无立春偏移）
;; ctx 未填字段：C_Y8..C_SEASON_NUM, C_DHE
(define fill_bazi (lambda (ctx)
    (define year (get_item ctx C_YEAR))
    (define month (get_item ctx C_MONTH))
    (define day (get_item ctx C_DAY))
    (define ly (get_item ctx C_LY))
    (define twohour_num (get_item ctx C_THNUM))
    (define nsn (get_item ctx C_NSN))
    (define y8 (g60_at (- (imod (- ly 4) 60) 0)))
    (define apart 0)
    (define m8 "")
    (define apart_days 0)
    (define dbase 0)
    (define dhe 0)
    (define d8v "")
    (define tbegin 0)
    (define thlist (newl))
    (define ti 0)
    (define h8 "")
    (if (and (== nsn 0) (== month 12)) (set! nsn 24) #f)
    (set! apart (idiv (+ nsn 1) 2))
    (set! m8 (g60_at (+ (* (- year 2019) 12) apart)))
    (set! apart_days (- (daynum year month day) (daynum 2019 1 29)))
    (set! dbase (+ 2 (if (== twohour_num 12) 1 0)))
    (set! dhe (imod (+ apart_days dbase) 60))
    (set! d8v (g60_at dhe))
    (set! tbegin (imod (* dhe 12) 60))
    (while (< ti 13)
        (push thlist (get_item A.G60 (imod (+ tbegin ti) 60)))
        (set! ti (+ ti 1)))
    (set! h8 (get_item thlist (imod twohour_num 12)))
    (set_item! ctx C_Y8 y8)
    (set_item! ctx C_M8 m8)
    (set_item! ctx C_D8S d8v)
    (set_item! ctx C_H8 h8)
    (set_item! ctx C_DHE dhe)
    (set_item! ctx C_YEAR_EARTH (branch_num_of8 y8))
    (set_item! ctx C_MONTH_EARTH (branch_num_of8 m8))
    (set_item! ctx C_DAY_EARTH (branch_num_of8 d8v))
    (set_item! ctx C_YEAR_HEAVEN (stem_num_of8 y8))
    (set_item! ctx C_MONTH_HEAVEN (stem_num_of8 m8))
    (set_item! ctx C_DAY_HEAVEN (stem_num_of8 d8v))
    (set_item! ctx C_SEASON_TYPE (imod (get_item ctx C_MONTH_EARTH) 3))
    (set_item! ctx C_SEASON_NUM (idiv (imod (+ (get_item ctx C_MONTH_EARTH) 10) 12) 3))
    (set_item! ctx C_MEN (get_item ctx C_MONTH_EARTH))
    (set_item! ctx C_D8 d8v)
    #undefined))

;; 星期 / 星座
(define week_star (lambda (y m d)
    (define days (daynum y m d))
    (define weekday (get_item A.WEEKDAY (imod (+ days 3) 7)))
    (define find (mk2 m d))
    (define sn 0)
    (define i 0)
    (define bd #f)
    (while (< i 12)
        (set! bd (mk2 (get_item A.STAR_ZODIAC_DATE (* i 2))
                      (get_item A.STAR_ZODIAC_DATE (+ (* i 2) 1))))
        (if (md_le bd find) (set! sn (+ sn 1)) #f)
        (set! i (+ i 1)))
    (mk2 weekday (get_item A.STAR_ZODIAC (imod sn 12)))))

;; 建除十二神 / 值神（cnl_12day_officer；men = month_earth_num）
(define officer_god (lambda (month_earth day_earth)
    (define men month_earth)
    (define oapart (imod (- day_earth (imod men 12)) 12))
    (define offc (get_item A.OFFICER_CHARS oapart))
    (define seeds (cons 8 (cons 10 (cons 0 (cons 2 (cons 4 (cons 6 (cons 8 (cons 10 (cons 0 (cons 2 (cons 4 (cons 6 (newl))))))))))))))
    (define ecl (imod (- day_earth (get_item seeds men)) 12))
    (mk2 offc (get_item A.DAYGOD ecl))))

;; 彭祖百忌（cnl_peng_taboo）
(define peng_of (lambda (day_heaven day_earth)
    (scat (scat (stak (get_item A.PENG_TABOO day_heaven) 9) ",")
          (stak (get_item A.PENG_TABOO (+ day_earth 10)) 9))))

;; 九宫飞星（cnl_9flystar）
(define fly_of (lambda (apart)
    (define flystart (cons 7 (cons 3 (cons 5 (cons 6 (cons 8 (cons 1 (cons 2 (cons 4 (cons 9 (newl)))))))))))
    (define out "")
    (define i 0)
    (define fv 0)
    (while (< i 9)
        (set! fv (+ (imod (- (- (get_item flystart i) 1) apart) 9) 1))
        (set! out (scat out (String.atom_to_string fv)))
        (set! i (+ i 1)))
    out))

;; 吉神方位（cnl_lucky_gods_direction）
(define dirs_of (lambda (day_heaven)
    (define dirnames (cons "喜神" (cons "财神" (cons "福神" (cons "阳贵" (cons "阴贵" (newl)))))))
    (define dirtables (cons A.LUCKY_DIR (cons A.WEALTH_DIR (cons A.MASCOT_DIR (cons A.SUNNOBLE_DIR (cons A.MOONNOBLE_DIR (newl)))))))
    (define out (newl))
    (define i 0)
    (define tri "")
    (define dj 0)
    (define dir -1)
    (while (< i 5)
        (set! tri (get_item (get_item dirtables i) day_heaven))
        (set! dj 0)
        (set! dir -1)
        (while (< dj 8)
            (if (seq (get_item A.TRIGRAM8 dj) tri) (begin (set! dir dj) (set! dj 8)) (set! dj (+ dj 1)))
            #f)
        (if (< dir 0) (set! dir 0) #f)
        (push out (scat (get_item dirnames i) (get_item A.DIRECTION dir)))
        (set! i (+ i 1)))
    out))

;; 时辰吉凶（cnl_twohour_lucky：今日 12 + 明日子时 = 13 项）
(define hourluck_of (lambda (dhe)
    (define hltoday (* dhe 12))
    (define hltom (* (imod (+ dhe 1) 60) 12))
    (define out (newl))
    (define i 0)
    (define idx 0)
    (while (< i 13)
        (if (== i 12) (set! idx hltom) (set! idx (+ hltoday i)))
        (if (== (get_item A.TWOHOUR idx) 1) (push out "凶") (push out "吉"))
        (set! i (+ i 1)))
    out))

;; ============================================================================
;; 神煞（cnl_angel_demon）
;; ============================================================================

;; 单行匹配（cnl_row_match）。rv 字段：
;;  0 名字 1 op 2 base 3 arg 4 list下标 5 list_n 6 pairs下标 7 pairs_n
;;  8 pstr下标 9 good下标 10 good_n 11 bad下标 12 bad_n
(define row_match (lambda (ctx rv)
    (define op 0)
    (define base "")
    (define k 0)
    (define li -1)
    (define ln 0)
    (define pi -1)
    (define pn 0)
    (define psi -1)
    (define ii 0)
    (define res #f)
    (define d1 "")
    (define d2 "")
    (define d8 "")
    (define e "")
    (define tw #f)
    (define tt #f)
    (define gl 0)
    (define grp "")
    (define judge "")
    (define lday (get_item ctx C_LDAY))
    (define lm (get_item ctx C_LM))
    (define dhe (get_item ctx C_DHE))
    (define day_earth (get_item ctx C_DAY_EARTH))
    (define year_earth (get_item ctx C_YEAR_EARTH))
    (define tmd_m (get_item ctx C_TMD_M))
    (define tmd_d (get_item ctx C_TMD_D))
    (define tuidi_days (get_item ctx C_TUIDI_DAYS))
    (define season_type (get_item ctx C_SEASON_TYPE))
    (define star28v (get_item ctx C_STAR28))
    (set! op (get_item rv 1))
    (set! base (get_item rv 2))
    (if (== base #f) (set! base "") #f)
    (set! d8 (get_item ctx C_D8))
    (set! d1 (sch d8 0))
    (set! d2 (sch d8 1))
    (set! k (get_item rv 3))
    (if (== k 1) (set! k (get_item ctx C_SEASON_NUM))
        (if (== k 2) (set! k (get_item ctx C_YEAR_HEAVEN))
            (if (== k 3) (set! k (imod (get_item ctx C_YEAR_EARTH) 4))
                (if (== k 4) (set! k (imod (get_item ctx C_MEN) 6))
                    (if (== k 0) (set! k (get_item ctx C_MEN)) #f)))))
    (set! li (get_item rv 4))
    (set! ln (get_item rv 5))
    (set! pi (get_item rv 6))
    (set! pn (get_item rv 7))
    (set! psi (get_item rv 8))
    (cond
      ((== op 1)
       (if (> (+ k 1) (slen base)) #f
           (set! res (substr? d8 (sch base k)))))
      ((== op 2)
       (if (> (+ k 1) (slen base)) #f (set! res (seq (sch base k) d1))))
      ((== op 3)
       (if (> (+ k 1) (slen base)) #f (set! res (seq (sch base k) d2))))
      ((== op 4)
       (if (> (+ k 1) (slen base)) #f
           (set! res (seq (sch base k) (sch star28v 0)))))
      ((== op 5)
       (set! res (substr? (get_item (get_item A.LL li) k) d1)))
      ((== op 6)
       (set! res (substr? (get_item (get_item A.LL li) k) d2)))
      ((== op 7)
       (set! res (substr? base d2)))
      ((== op 8)
       (begin
         (set! ii 0)
         (while (< ii ln)
             (if (seq (get_item (get_item A.LL li) ii) d8) (set! res #t) #f)
             (set! ii (+ ii 1)))))
      ((== op 9)
       (if (or (< k 0) (>= k ln))
           #f
           (begin
             (set! e (get_item (get_item A.LL li) k))
             (if (== (slen e) 2) (set! res (seq e d8))
                 (if (== (slen e) 4)
                     (set! res (or (seq (String.slice e 0 2) d8) (seq (String.slice e 2 4) d8)))
                     #f)))))
      ((== op 10)
       (begin
         (set! ii 0)
         (while (< ii pn)
             (if (== (get_item (get_item A.PI pi) ii) lday) (set! res #t) #f)
             (set! ii (+ ii 1)))))
      ((== op 11)
       (begin
         (if (seq (get_item rv 0) "四绝") (set! tw (get_item ctx C_T4J)) (set! tw (get_item ctx C_T4L)))
         (set! ii 0)
         (while (< ii 4)
             (set! tt (get_item tw ii))
             (if (and (== (get_item tt 0) tmd_m) (== (get_item tt 1) tmd_d)) (set! res #t) #f)
             (set! ii (+ ii 1)))))
      ((== op 12)
       (begin
         (set! ii 0)
         (while (< ii pn)
             (if (and (== (get_item (get_item A.PI pi) (* ii 2)) lm)
                      (== (get_item (get_item A.PI pi) (+ (* ii 2) 1)) lday))
                 (set! res #t) #f)
             (set! ii (+ ii 1)))))
      ((== op 13)
       (begin
         (set! ii 0)
         (while (< ii pn)
             (if (and (== (get_item (get_item A.PI pi) ii) lm)
                      (seq (get_item (get_item A.LL psi) ii) d8))
                 (set! res #t) #f)
             (set! ii (+ ii 1)))))
      ((== op 14)
       (if (> (+ (* 2 k) 2) (slen base)) #f
           (set! res (seq (String.slice base (* 2 k) (+ (* 2 k) 2)) d8))))
      ((== op 15)
       (if (> (+ (* 3 k) 3) (slen base)) #f
           (set! res (substr? (String.slice base (* 3 k) (+ (* 3 k) 3)) d2))))
      ((== op 16)
       (set! res (and (>= tuidi_days 0) (< tuidi_days 18))))
      ((== op 17)
       (set! res (== (imod (- day_earth (get_item ctx C_MEN)) 4) 0)))
      ((== op 18)
       (set! res (== day_earth (imod (+ year_earth 6) 12))))
      ((== op 19)
       (set! res (and (< (imod dhe 15) 5) (not (== (idiv dhe 15) 2)))))
      ((== op 20)
       (begin
         (if (== season_type 0) (set! judge d2) (set! judge d1))
         (set! res (substr? (get_item (get_item A.LL li) (get_item ctx C_MEN)) judge))))
      ((== op 21)
       (begin
         (set! grp (get_item A.BUJIANG (get_item ctx C_MEN)))
         (set! gl (slen grp))
         (set! ii 0)
         (while (<= (+ (* 2 ii) 2) gl)
             (if (seq (String.slice grp (* 2 ii) (+ (* 2 ii) 2)) d8) (set! res #t) #f)
             (set! ii (+ ii 1)))))
      (else #f))
    (if res #t #f)))

;; 神煞遍历：命中的吉神/凶煞并入 god 名表与宜忌表（就地改写）
(define angel_pass (lambda (ctx rows n goodgods badgods goodT badT)
    (define i 0)
    (define rv #f)
    (while (< i n)
        (set! rv (get_item rows i))
        (if (row_match ctx rv)
            (begin
              (ladd! goodgods (get_item rv 0))
              (lunion_arr! goodT (get_item A.LL (get_item rv 9)) (get_item rv 10))
              (lunion_arr! badT (get_item A.LL (get_item rv 11)) (get_item rv 12)))
            #f)
        (set! i (+ i 1)))))

;; 凶煞遍历（god 名进 badgods；宜忌照 union）
(define demon_pass (lambda (ctx rows n badgods goodT badT)
    (define i 0)
    (define rv #f)
    (while (< i n)
        (set! rv (get_item rows i))
        (if (row_match ctx rv)
            (begin
              (ladd! badgods (get_item rv 0))
              (lunion_arr! goodT (get_item A.LL (get_item rv 9)) (get_item rv 10))
              (lunion_arr! badT (get_item A.LL (get_item rv 11)) (get_item rv 12)))
            #f)
        (set! i (+ i 1)))))

;; 遇德的凶煞清单（cnl_de_is_bad_thing：仅前 6 个吉神行）
(define de_is_bad (lambda (ctx goodgods)
    (define out (newl))
    (define i 0)
    (define rv #f)
    (while (< i 6)
        (set! rv (get_item A.ANGEL i))
        (if (lhas goodgods (get_item rv 0))
            (lunion_arr! out (get_item A.LL (get_item rv 11)) (get_item rv 12))
            #f)
        (set! i (+ i 1)))
    out))

;; 宜忌等第（cnl_angel_demon 的等第表循环）
(define things_level (lambda (allgods mbranch)
    (define lvl -1)
    (define i 0)
    (define gname "")
    (define ei 0)
    (define found 0)
    (define ent #f)
    (define subs #f)
    (define si 0)
    (define sub #f)
    (define pat "")
    (define gods #f)
    (define gn 0)
    (define lv 0)
    (define gi 0)
    (while (< i (length allgods))
        (set! gname (get_item allgods i))
        (set! ei 0)
        (set! found 0)
        (while (< ei 11)
            (set! ent (get_item A.THING_LEVEL ei))
            (if (and (== found 0) (seq (get_item ent 0) gname))
                (begin
                  (set! found 1)
                  (set! subs (get_item ent 1))
                  (set! si 0)
                  (while (< si (length subs))
                      (set! sub (get_item subs si))
                      (set! pat (get_item sub 0))
                      (set! gods (get_item A.LL (get_item sub 1)))
                      (set! gn (get_item sub 2))
                      (set! lv (get_item sub 3))
                      (if (substr? pat mbranch)
                          (begin
                            (set! gi 0)
                            (while (< gi gn)
                                (if (and (lhas allgods (get_item gods gi)) (> lv lvl))
                                    (begin (set! lvl lv) (set! gi gn))
                                    (set! gi (+ gi 1))))
                            #f)
                          #f)
                      (set! si (+ si 1))))
                #f)
            (set! ei (+ ei 1)))
        (set! i (+ i 1)))
    lvl))

;; 宜忌修正规则（cnl_angel_demon 的第二半；就地改写 goodT/badT）
(define apply_rules (lambda (ctx goodgods badgods goodT badT tlev isde today_level dibt)
    (define i 0)
    (define x 0)
    (define d8 (get_item ctx C_D8))
    (define lm (get_item ctx C_LM))
    (define offc (get_item ctx C_OFFC))
    (define isde_she 0)
    (define g "")
    (define rm (newl))
    (define rmv (newl))
    (define rn 0)
    (define firstm "")
    (define mday12 (cons "空" (cons "甲戌" (cons "空" (cons "丙申" (cons "空" (cons "甲子" (cons "戊申" (cons "庚辰" (cons "辛卯" (cons "甲子" (cons "空" (cons "甲子" (newl))))))))))))))
    (define probe "")
    ;; 按等第调整
    (if (== tlev 3)
        (begin
          (lreset! goodT)
          (ladd! goodT "诸事不宜")
          (lreset! badT)
          (ladd! badT "诸事不宜"))
        (if (== tlev 2)
            (begin
              (set! i 0)
              (while (< i (length badT))
                  (set! x (get_item badT i))
                  (if (lhas goodT x) (ladd! rm x) #f)
                  (set! i (+ i 1)))
              (lrm_list! goodT rm))
            (if (== tlev 1)
                (begin
                  (set! i 0)
                  (while (< i (length goodT))
                      (set! x (get_item goodT i))
                      (if (lhas badT x) (ladd! rm x) #f)
                      (set! i (+ i 1)))
                  (lrm_list! goodT rm)
                  (lrm_list! badT rm))
                (begin
                  (set! i 0)
                  (while (< i (length badT))
                      (set! x (get_item badT i))
                      (if (lhas goodT x) (ladd! rm x) #f)
                      (set! i (+ i 1)))
                  (lrm_list! badT rm)))))
    (if (not (== tlev 3))
        (begin
          ;; 宣政事 / 布政事
          (if (and (lhas goodT "宣政事") (lhas goodT "布政事")) (lrm! goodT "布政事") #f)
          ;; 德合与赦愿所汇
          (set! i 0)
          (while (< i (length goodgods))
              (set! g (get_item goodgods i))
              (if (or (seq g "月德合")
                      (or (seq g "天德合")
                      (or (seq g "天赦")
                      (or (seq g "天愿")
                      (or (seq g "月恩")
                      (or (seq g "四相")
                      (or (seq g "时德") (seq g "岁德合"))))))))
                  (set! isde_she 1) #f)
              (set! i (+ i 1)))
          (if (and (== isde_she 1) (not (== tlev 2)))
              (begin
                (lrm_arr! badT (cons "进人口" (cons "安床" (cons "经络" (cons "酝酿" (cons "开市" (cons "立券交易" (cons "纳财" (cons "开仓库" (cons "出货财" (newl)))))))))) 9)
                (lunion_from! badT dibt))
              #f)
          ;; 天狗/寅 → 祭祀
          (if (or (lhas badgods "天狗") (substr? d8 "寅"))
              (begin
                (ladd! badT "祭祀")
                (lrm! goodT "祭祀")
                (lrm! goodT "求福")
                (lrm! goodT "祈嗣"))
              #f)
          ;; 卯日 穿井
          (if (substr? d8 "卯")
              (begin (ladd! badT "穿井") (lrm! goodT "穿井") (lrm! goodT "开渠")) #f)
          ;; 壬日 开渠
          (if (substr? d8 "壬")
              (begin (ladd! badT "开渠") (lrm! goodT "开渠") (lrm! goodT "穿井")) #f)
          ;; 巳日 出行
          (if (substr? d8 "巳")
              (begin (ladd! badT "出行") (lrm! goodT "出行") (lrm! goodT "出师") (lrm! goodT "遣使")) #f)
          ;; 酉日 宴会
          (if (substr? d8 "酉")
              (begin (ladd! badT "宴会") (lrm! goodT "宴会") (lrm! goodT "庆赐") (lrm! goodT "赏贺")) #f)
          ;; 丁日 剃头
          (if (substr? d8 "丁")
              (begin (ladd! badT "剃头") (lrm! goodT "剃头") (lrm! goodT "整容")) #f)
          (if (and (== today_level 0) (== tlev 0)) (lunion_from! badT dibt) #f)
          (if (== today_level 1)
              (begin
                (lunion_from! badT dibt)
                (if (not (lhas badT "祈福")) (lrm! badT "求嗣") #f)
                (if (and (not (lhas badT "结婚姻")) (== isde 0))
                    (begin
                      (lrm_arr! badT (cons "冠带" (cons "纳采问名" (cons "嫁娶" (cons "进人口" (newl))))) 4))
                    #f)
                (if (and (not (lhas badT "嫁娶")) (== isde 0))
                    (begin
                      (if (not (lhas goodgods "不将"))
                          (begin
                            (lrm_arr! badT (cons "冠带" (cons "纳采问名" (cons "结婚姻" (cons "进人口" (cons "搬移" (cons "安床" (newl))))))) 6))
                          #f))
                    #f))
              #f)
          ;; 亥日 嫁娶
          (if (substr? d8 "亥") (ladd! badT "嫁娶") #f)
          (if (and (== today_level 1) (== isde 0))
              (begin
                (if (not (lhas badT "搬移")) (lrm! badT "安床") #f)
                (if (not (lhas badT "安床")) (lrm! badT "搬移") #f)
                (if (not (lhas badT "解除"))
                    (begin
                      (lrm_arr! badT (cons "整容" (cons "剃头" (cons "整手足甲" (newl)))) 3))
                    #f)
                (if (or (not (lhas badT "修造")) (not (lhas badT "竖柱上梁")))
                    (begin
                      (lrm_arr! badT (cons "修宫室" (cons "缮城郭" (cons "整手足甲" (cons "筑提" (cons "修仓库" (cons "鼓铸" (cons "苫盖" (cons "修置产室" (cons "开渠穿井" (cons "安碓硙" (cons "补垣塞穴" (cons "修饰垣墙" (cons "平治道涂" (cons "破屋坏垣" (newl))))))))))))))) 14))
                    #f))
              #f)
          (if (== today_level 1)
              (begin
                (if (not (lhas badT "开市"))
                    (begin (lrm_arr! badT (cons "立券交易" (cons "纳财" (cons "开仓库" (cons "出货财" (newl))))) 4)) #f)
                (if (not (lhas badT "纳财"))
                    (begin (lrm_arr! badT (cons "立券交易" (cons "开市" (newl))) 2)) #f)
                (if (not (lhas badT "立券交易"))
                    (begin (lrm_arr! badT (cons "纳财" (cons "开市" (cons "开仓库" (cons "出货财" (newl))))) 4)) #f)
                (if (not (lhas badT "牧养")) (lrm! badT "纳畜") #f)
                (if (not (lhas badT "纳畜")) (lrm! badT "牧养") #f)
                (if (lhas goodT "安葬") (lrm! badT "启攒") #f)
                (if (lhas goodT "启攒") (lrm! badT "安葬") #f))
              #f)
          (if (or (lhas badT "诏命公卿") (lhas badT "招贤"))
              (begin (lrm! goodT "施恩") (lrm! goodT "举正直")) #f)
          (if (or (lhas badT "施恩") (lhas badT "举正直"))
              (begin (lrm! goodT "诏命公卿") (lrm! goodT "招贤")) #f)
          (if (and (lhas goodT "宣政事") (lhas badgods "往亡"))
              (begin (lrm! goodT "宣政事") (ladd! goodT "布政事")) #f)
          (if (lhas badgods "月厌")
              (begin
                (lrm_arr! goodT (cons "颁诏" (cons "施恩" (cons "招贤" (cons "举正直" (cons "宣政事" (newl)))))) 5)
                (ladd! goodT "布政事")
                (ladd! badT "补垣")
                (if (or (lhas badgods "土府") (or (lhas badgods "土符") (lhas badgods "地囊")))
                    (lrm! goodT "塞穴") #f))
              #f)
          (if (substr? offc "开")
              (begin (lrm! goodT "破土") (lrm! goodT "安葬") (lrm! goodT "启攒")) #f)
          (if (or (lhas badgods "四忌") (lhas badgods "四穷"))
              (begin (ladd! badT "安葬") (lrm! goodT "破土") (lrm! goodT "启攒")) #f)
          (if (or (lhas goodgods "鸣吠") (lhas goodgods "鸣吠对"))
              (begin (lrm! goodT "破土") (lrm! goodT "启攒")) #f)
          ;; 德和与赦愿所汇（mday12 探针）
          (set! probe (get_item mday12 (imod (- lm 1) 12)))
          (if (substr? d8 probe)
              (begin (lreset! badT) (ladd! badT "诸事不忌")) #f)
          (if (and (or (lhas goodgods "岁德合") (or (lhas goodgods "月德合") (lhas goodgods "天德合")))
                   (or (lhas goodgods "天赦") (lhas goodgods "天愿")))
              (begin (lreset! badT) (ladd! badT "诸事不忌")) #f))
        #f)
    ;; rmThing：bad ∩ good（单条"诸事…"时保留）
    (set! i 0)
    (while (< i (length badT))
        (set! x (get_item badT i))
        (if (lhas goodT x)
            (begin
              (if (== rn 0) (set! firstm x) #f)
              (set! rn (+ rn 1)))
            #f)
        (set! i (+ i 1)))
    (if (not (and (== rn 1) (substr? firstm "诸事")))
        (begin
          (set! i 0)
          (while (< i (length badT))
              (set! x (get_item badT i))
              (if (lhas goodT x) (ladd! rmv x) #f)
              (set! i (+ i 1)))
          (lrm_list! goodT rmv))
        #f)
    (if (== (length badT) 0) (push badT "诸事不忌") #f)
    (if (== (length goodT) 0) (push goodT "诸事不宜") #f)
    (lsort! badT)
    (lsort! goodT)
    #undefined))

;; ============================================================================
;; 主入口
;; ============================================================================

(define calc (lambda (year month day hour minute)
    (define out #f)
    (define ctx #f)
    (define twohour_num 0)
    (define LD #f)
    (define lp #f)
    (define ST #f)
    (define WS #f)
    (define OG #f)
    (define apart 0)
    (define dhe 0)
    (define tmd #f)
    (define t4j (newl))
    (define t4l (newl))
    (define t4jidx (cons 2 (cons 8 (cons 14 (cons 20 (newl))))))
    (define t4lidx (cons 5 (cons 11 (cons 17 (cons 23 (newl))))))
    (define ti 0)
    (define tcnt 0)
    (define tj #f)
    (define am 0)
    (define ad 0)
    (define tk 0)
    (define tuidi #f)
    (define tuidi_m 0)
    (define tuidi_d 0)
    (define tsec 0)
    (define tuidi_days 0)
    (define goodgods (newl))
    (define badgods (newl))
    (define goodT (newl))
    (define badT (newl))
    (define allgods (newl))
    (define mbranch "")
    (define lvl -1)
    (define today_level -1)
    (define isde 0)
    (define tlev 1)
    (define dibt (newl))
    (define score 0)
    (define overallv "")
    (define R (newl))
    (define mi 0)
    (define is_zq_r #f)
    (define a1days (cons 1 (cons 6 (cons 15 (cons 19 (cons 21 (cons 23 (newl))))))))
    (define t8 #f)
    (define ot #f)
    (define D8 "")
    (define gname "")
    (if (or (< year 1901) (> year 2100)) #f
    (if (or (< month 1) (> month 12)) #f
    (if (or (< day 1) (> day 31)) #f
    (if (or (< hour 0) (> hour 23)) #f
    (if (or (< minute 0) (> minute 59)) #f
        (begin
          (set! twohour_num (idiv (+ hour 1) 2))
          (set! LD (lunar_date year month day))
          (if (== LD #f)
              #f
              (begin
                ;; ---- 公共中间量 ----
                (set! lp (month_long_phase (get_item LD 2) (get_item LD 3)
                                           (get_item LD 4) (get_item LD 6)))
                (set! ST (today_solar_terms year month day))
                (set! WS (week_star year month day))
                (set! ctx (mk4 year month day hour))
                (push ctx minute)
                (push ctx twohour_num)
                (push ctx (get_item LD 0))
                (push ctx (get_item LD 1))
                (push ctx (get_item LD 2))
                (push ctx (get_item LD 3))
                (push ctx (get_item LD 4))
                (push ctx (get_item LD 5))
                (push ctx (get_item LD 6))
                (push ctx (get_item lp 0))
                (push ctx (get_item lp 1))
                (push ctx (get_item ST 0))
                (push ctx (get_item ST 1))
                (push ctx (get_item ST 2))
                (push ctx (get_item ST 3))
                (while (< (length ctx) 39) (push ctx #f))
                (fill_bazi ctx)
                (set! OG (officer_god (get_item ctx C_MONTH_EARTH) (get_item ctx C_DAY_EARTH)))
                (set! apart (- (daynum year month day) (daynum 2019 1 17)))
                (set_item! ctx C_OFFC (get_item OG 0))
                (set_item! ctx C_DAYGOD (get_item OG 1))
                (set_item! ctx C_STAR28 (get_item A.STAR28 (imod apart 28)))
                (set_item! ctx C_WEEK (get_item WS 0))
                (set_item! ctx C_ZODIAC (get_item WS 1))
                (set! dhe (get_item ctx C_DHE))

                ;; ---- 土王用事 / 四绝四离 上下文 ----
                (set! tmd (ymd (+ (daynum year month day) 1)))
                (set! ti 0)
                (while (< ti 4)
                    (push t4j (get_item (get_item ctx C_TERMS) (get_item t4jidx ti)))
                    (push t4l (get_item (get_item ctx C_TERMS) (get_item t4lidx ti)))
                    (set! ti (+ ti 1)))
                (set! ti 0)
                (while (< ti 4)
                    (set! tj (get_item t4j ti))
                    (set! am (get_item tj 0))
                    (set! ad (get_item tj 1))
                    (if (or (< am (get_item tmd 1))
                            (and (== am (get_item tmd 1)) (< ad (get_item tmd 2))))
                        (set! tcnt (+ tcnt 1)) #f)
                    (set! ti (+ ti 1)))
                (set! tk (imod tcnt 4))
                (set! tuidi (get_item t4j tk))
                (set! tuidi_m (get_item tuidi 0))
                (set! tuidi_d (get_item tuidi 1))
                (set! tsec (- (* (- (daynum (get_item ctx C_NSY) tuidi_m tuidi_d)
                                  (daynum year month day)) 86400)
                              (+ (* hour 3600) (* minute 60))))
                (set! tuidi_days (idiv_floor tsec 86400))
                ;; 追加 ctx 字段（C_TMD_M/C_TMD_D/C_TUIDI_DAYS/C_T4J/C_T4L）
                (push ctx (get_item tmd 1))
                (push ctx (get_item tmd 2))
                (push ctx tuidi_days)
                (push ctx t4j)
                (push ctx t4l)
                (set! D8 (get_item ctx C_D8))

                ;; ---- 建除十二神宜忌 ----
                (set! ti 0)
                (while (< ti 12)
                    (set! ot (get_item A.OFFICER_THINGS ti))
                    (if (seq (get_item ot 0) (get_item ctx C_OFFC))
                        (begin
                          (lunion_arr! goodT (get_item A.LL (get_item ot 1)) (get_item ot 2))
                          (lunion_arr! badT (get_item A.LL (get_item ot 3)) (get_item ot 4)))
                        #f)
                    (set! ti (+ ti 1)))

                ;; ---- 日干宜忌（strstr(d, ch)） ----
                (set! ti 0)
                (while (< ti 16)
                    (set! t8 (get_item A.D8THINGS ti))
                    (if (substr? D8 (get_item t8 0))
                        (begin
                          (lunion_arr! goodT (get_item A.LL (get_item t8 1)) (get_item t8 2))
                          (lunion_arr! badT (get_item A.LL (get_item t8 3)) (get_item t8 4)))
                        #f)
                    (set! ti (+ ti 1)))

                ;; ---- 节气间差类（取鱼/畋猎/伐木） ----
                (set! is_zq_r (or (seq (get_item ctx C_OFFC) "执")
                                  (or (seq (get_item ctx C_OFFC) "危")
                                      (seq (get_item ctx C_OFFC) "收"))))
                (if (and (>= (get_item ctx C_NSN) 4) (<= (get_item ctx C_NSN) 8) is_zq_r)
                    (ladd! goodT "取鱼") #f)
                (if (if (and (>= (get_item ctx C_NSN) 20) (< (get_item ctx C_NSN) 24))
                        is_zq_r
                        (if (and (>= (get_item ctx C_NSN) 0) (< (get_item ctx C_NSN) 3)) is_zq_r #f))
                    (ladd! goodT "畋猎") #f)
                (if (if (and (>= (get_item ctx C_NSN) 21) (< (get_item ctx C_NSN) 24))
                        #t
                        (and (>= (get_item ctx C_NSN) 0) (< (get_item ctx C_NSN) 3)))
                    (if (seq (get_item ctx C_OFFC) "危") (ladd! goodT "伐木") #f)
                    #f)

                ;; ---- 农历日规则 ----
                (set! ti 0)
                (while (< ti 6)
                    (if (== (get_item ctx C_LDAY) (get_item a1days ti)) (ladd! badT "整手足甲") #f)
                    (set! ti (+ ti 1)))
                (if (or (== (get_item ctx C_LDAY) 12) (== (get_item ctx C_LDAY) 15))
                    (begin (ladd! badT "整容") (ladd! badT "剃头")) #f)
                (if (or (== (get_item ctx C_LDAY) 15) (not (seq (get_item ctx C_PHASE) "")))
                    (ladd! badT "求医疗病") #f)

                ;; ---- 神煞 ----
                (angel_pass ctx A.ANGEL 71 goodgods badgods goodT badT)
                (ldedup! goodT)
                (ldedup! badT)
                (demon_pass ctx A.DEMON 100 badgods goodT badT)
                (ldedup! goodT)
                (ldedup! badT)

                ;; ---- 宜忌等第 ----
                (lunion_from! allgods goodgods)
                (lunion_from! allgods badgods)
                (ladd! allgods (scat (get_item ctx C_OFFC) "日"))
                (set! mbranch (sch (get_item ctx C_M8) 1))
                (set! lvl (things_level allgods mbranch))
                (set! today_level lvl)
                (set! ti 0)
                (while (< ti (length goodgods))
                    (set! gname (get_item goodgods ti))
                    (if (or (seq gname "岁德")
                            (or (seq gname "岁德合")
                            (or (seq gname "月德")
                            (or (seq gname "月德合")
                            (or (seq gname "天德") (seq gname "天德合"))))))
                        (set! isde 1) #f)
                    (set! ti (+ ti 1)))
                (if (== lvl 5) (set! tlev 3)
                (if (== lvl 4) (set! tlev (if (== isde 1) 2 3))
                (if (== lvl 3) (set! tlev (if (== isde 1) 1 2))
                (if (== lvl 2) (set! tlev (if (== isde 1) 0 2))
                (if (== lvl 1) (set! tlev (if (== isde 1) 0 1))
                (if (== lvl 0) (set! tlev 0)
                (set! tlev 1)))))))
                (set! dibt (de_is_bad ctx goodgods))
                (apply_rules ctx goodgods badgods goodT badT tlev isde today_level dibt)

                ;; ---- 当日吉凶总评 ----
                (if (substr? "除定执危成开" (get_item ctx C_OFFC))
                    (set! score (+ score 1)) (set! score (- score 1)))
                (if (substr? "青龙明堂金贵天德玉堂司命" (get_item ctx C_DAYGOD))
                    (set! score (+ score 1)) (set! score (- score 1)))
                (if (<= tlev 1) (set! score (+ score 1)) (set! score (- score 1)))
                (set! overallv (if (>= score 2) "大吉" (if (>= score 0) "吉" (if (>= score -1) "平" "凶"))))

                ;; ---- 汇总 ----
                (set! mi (imod twohour_num 12))
                (push R (scat (get_item ctx C_Y8) (scat (get_item ctx C_M8) (scat (get_item ctx C_D8S) (get_item ctx C_H8)))))
                (push R overallv)
                (push R (get_item ctx C_WEEK))
                (push R (get_item ctx C_ZODIAC))
                (push R (get_item ctx C_STAR28))
                (push R (get_item ctx C_OFFC))
                (push R (get_item ctx C_DAYGOD))
                (push R (get_item A.MERIDIANS mi))
                (push R (get_item A.MERIDIAN_YI mi))
                (push R (get_item A.MERIDIAN_JI mi))
                (push R (get_item A.MERIDIAN_NOTE mi))
                (push R goodT)
                (push R badT)
                (push R (fly_of apart))
                (push R (get_item A.FETAL_GOD dhe))
                (push R (peng_of (get_item ctx C_DAY_HEAVEN) (get_item ctx C_DAY_EARTH)))
                (push R (hourluck_of dhe))
                (push R (dirs_of (get_item ctx C_DAY_HEAVEN)))
                (push R (mk4 (get_item ctx C_LY) (get_item ctx C_LM)
                             (get_item ctx C_LDAY) (get_item ctx C_LEAP)))
                (push R (get_item ctx C_NSN))
                (push R goodgods)
                (push R badgods)
                (push R today_level)
                (push R tlev)
                (set! out R)))))))))
    out))
