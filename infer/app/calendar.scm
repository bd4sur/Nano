;@id=calendar
;@title=日历
;@version=2026-11-17
;; ============================================================================
;; calendar.scm —— 日历 + 黄历（灵机应用程序版）
;;
;; 2026-11 GUI 重构：从「与 C 版逐像素对齐的 6 行空白格日历 + 四张固定卡片模态框」
;; 改为现代、易用的两页式界面。
;;
;; 页面一：日历（view 0）
;;   · 顶栏：左「YYYY年M月」（16px 抗锯齿），右「丙午马年」（12px 淡色），
;;     最右是 (EX.draw_button) 的统一「退出」；底边 1px 分隔线。
;;   · 星期条：一…日（周末暖色）。
;;   · 日期网格：**连续排布**——从「含本月 1 日的那一周的周一」起固定 6 周（42 格），
;;     上月尾/下月头一并画出（淡化），因此月份交界处不再出现空白格，视觉上是连着的
;;     一条时间轴。每格两行：上「日」（16px）+ 下「农历」（12px）。
;;       农历行三档优先级：初一 → 农历月名（闰月带「闰」）/ 当日节气 → 节气名 /
;;       其余 → 农历日名（初二…三十）。今天 = 蓝色圆角色块 + 白字；
;;       最近查看的那天 = 蓝色描边环。
;;   · 底栏五枚通用按钮（app/lib/button.scm）：上年 / 上月 / 今天 / 下月 / 下年
;;     —— 按月、按年快速滚动 + 一键回今天；**按住 0.36s 后每 110ms 连发**
;;     （从 2026 年翻到 1950 年不用点 76 下）。
;;   · 点按任意日期格 → 该日黄历（跨月格里点按同样打开该日黄历）。
;;
;; 页面二：黄历（view 1）
;;   · 顶栏：「返回」按钮 + 「2026年3月9日」+ 星期 + 吉凶徽标（大吉/吉/平/凶），
;;     最右仍是统一的「退出」。
;;   · 主体：**整页纵向滚动**（手指拖动 + 惯性滑动 + 右侧滚动条），内容按卡片分组：
;;       ① 农历概览（农历日/干支年/生肖/星座/星宿/节气/下一节气）
;;       ② 干支四柱（年柱/月柱/日柱/时柱 + 建除十二神 + 值神）
;;       ③ 宜忌（彩色标签 + 自动折行，**不再截断成一个「…」**）
;;       ④ 时辰吉凶（12 时辰 6×2 网格，吉绿凶红）
;;       ⑤ 吉神方位（喜神/财神/福神/阳贵/阴贵 五枚方位芯片）
;;       ⑥ 胎神 + 彭祖百忌
;;       ⑦ 九宫飞星（9 枚数字芯片）
;;       ⑧ 吉神宜趋 / ⑨ 凶煞宜忌（完整神煞名，折行显示）
;;       ⑩ 子午流注养生（当令器官 / 宜 / 忌 / 说明）
;;     所有文字 12px 抗锯齿、自动折行；内容只在切日/打开时布局一次
;;     （alm_bg / alm_fg / alm_tx 三张绘制清单），滚动只是重放清单，因此很跟手。
;;   · 底栏三枚按钮：前一日 / 今日 / 后一日 —— 可以逐日翻看黄历（跨月自动跟随，
;;     返回日历时日历也跟到那一天），同样支持按住连发。
;;
;; 输入：
;;   · 触屏：日历页按钮/日期格；黄历页按钮/拖动滚动。
;;   · 实体键盘：日历页 ←→ 切月、↑↓ 切年、0/Home 回今天、Enter 看今天黄历；
;;     黄历页 ←→ 切前/后一日、↑↓/PgUp/PgDn 滚动（可长按连发）、Enter 返回日历。
;;   · 硬 A/ESC 由 app/lib/exit.scm 接管为全局退出确认框（与各 App 一致）。
;;
;; 时间来源：(native Time) 只给 UTC，故按 UTC+8 换算本地民用时（与其它 App 一致）。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数，没有一元取负（写 (- 0 x)）；
;;   没有 let/list/map/append；define 只写在模块顶层或 lambda 体最外层（块内不 define）；
;;   cond 每子句只接受单个表达式（多语句用 {}）；/ 与 mod 恒浮点（索引用 Math.floor 收敛，
;;   大整数除法一律 AL.idiv）；0 是**真值**（布尔一律 #t/#f）；'() 是静态对象，
;;   空表一律 (cdr '(0)) 现造，绝不对 '() 直接 push。
;; ============================================================================

(native Gfx)
(native Key)
(native Touch)
(native Time)
(native String)
(native Math)
(native System)
(native Diag)

(import AL "app/lib/almanac.scm")       ; 黄历/农历计算核心（已通过 C 全量差分回归）
(import AD "app/lib/almanac_data.scm")  ; 节气名表（AL 内部数据模块，同路径只加载一份）
(import BTN "app/lib/button.scm")       ; 通用按钮控件
(import EX "app/lib/exit.scm")          ; 硬 A/ESC 与右上角「退出」

;; ---- 字体 ------------------------------------------------------------------
(define F12 1)
(define F16 2)

;; ---- 配色（深色，与其它灵机 App 同一套观感） --------------------------------
(define BG_R 10)  (define BG_G 14)  (define BG_B 19)
(define HDR_R 14) (define HDR_G 18) (define HDR_B 24)
(define FTR_R 14) (define FTR_G 18) (define FTR_B 24)
(define EDGE_R 33) (define EDGE_G 41) (define EDGE_B 52)
(define TXT_R 232) (define TXT_G 238) (define TXT_B 246)
(define DIM_R 138) (define DIM_G 149) (define DIM_B 164)
(define FNT_R 72) (define FNT_G 80) (define FNT_B 93)        ;; 非本月日（淡化）
(define FNT2_R 56) (define FNT2_G 63) (define FNT2_B 74)     ;; 非本月日的农历行
(define ACC_R 64) (define ACC_G 140) (define ACC_B 246)      ;; 强调蓝
(define ACC2_R 125) (define ACC2_G 211) (define ACC2_B 252)  ;; 农历月名（浅蓝）
(define SUN_R 255) (define SUN_G 122) (define SUN_B 122)     ;; 周末暖红
(define TERM_R 255) (define TERM_G 196) (define TERM_B 107)  ;; 节气（暖黄）
(define GRN_R 110) (define GRN_G 231) (define GRN_B 168)     ;; 宜
(define RED_R 255) (define RED_G 138) (define RED_B 160)     ;; 忌
(define YEL_R 255) (define YEL_G 214) (define YEL_B 102)     ;; 平/警示
(define GOL_R 255) (define GOL_G 200) (define GOL_B 105)     ;; 大吉
(define CARD_R 20) (define CARD_G 26) (define CARD_B 34)
(define CHIP_R 30) (define CHIP_G 38) (define CHIP_B 50)

;; ---- 版面（320x240 基准，按实际画布宽高推导） -------------------------------
(define SW 320)
(define SH 240)
(define HDR_H 26)          ;; 顶栏
(define WEEK_Y 26)         ;; 星期条
(define WEEK_H 14)
(define GRID_TOP 40)       ;; 日期网格（6 行 × ROW_H）
(define ROW_H 28)
(define GRID_BOT 208)
(define FTR_Y 208)         ;; 底栏（按钮行）
(define BTN_H 22)
(define BTN_Y 212)

;; 黄历页
(define ALM_TOP 26)
(define ALM_BOT 206)
(define ALM_VH 180)
(define CARD_X 6)
(define CARD_W 300)
(define CARD_IX 14)        ;; 卡片内左内边距
(define CARD_IW 284)       ;; 卡片内可用宽
(define CARD_RGT 296)      ;; 卡片内右边界
(define LH12 15)

;; ---- 状态 ------------------------------------------------------------------
(define view 0)            ;; 0 = 日历，1 = 黄历

(define cal_y 2026)
(define cal_m 1)
(define cal_dirty 1)
(define cal_title "")
(define cal_last_day -1)

(define now_y 2026)
(define now_m 1)
(define now_d 1)
(define now_h 0)
(define now_min 0)

(define sel_dn 0)          ;; 最近查看的日期（1970 起的天数；网格里画描边环）

(define alm_y 2026)
(define alm_m 1)
(define alm_d 1)
(define alm_R #f)
(define alm_err 0)
(define alm_dirty 1)
(define alm_scroll 0)
(define alm_fpos 0.0)
(define alm_maxs 0)
(define alm_total 0)
(define alm_vel 0.0)

;; 拖动滚动状态
(define drg_on #f)
(define drg_moved #f)
(define drg_y0 0)
(define drg_prev 0)

;; 底栏按钮的「按住连发」状态（-1 = 没有按钮被按住）
(define hold_btn -1)
(define prev_armed -1)
(define hold_t0 0)
(define hold_n 0)
(define hold_last 0)

;; 按键长按重建状态（等价 GUI 的 key_edge：短按松手 -1 / 长按连发 -2）
(define ev_code 0)
(define ev_kind 0)
(define k_prev 0)
(define k_t0 0)
(define k_rep 0)

(define TZ_SEC 28800)
(define timer_id 0)

;; 文本墨迹顶补偿（boot 时算一次）
(define gap12 0)
(define gap16 0)

;; 日期数字字符串缓存（1..31，避免每帧 atom_to_string）
(define NUM_S (cdr '(0)))

;; 农历标签类型（cell_label 的副产物：0=日名 1=月名 2=节气）
(define cl_kind 0)

;; ============================================================================
;; 文本工具
;; ============================================================================

(define slen (lambda (s) (String.length s)))
(define cat (lambda (a b) (String.concat a b)))
(define seq (lambda (a b) (String.equals a b)))

;; 单字形步进（像素）
(define adv (lambda (font cp)
    (define m (Gfx.glyph_metrics font cp))
    (if (null? m) 0 (get_item m 0))))

;; 与 gfx_font_measure_text 等价：逐码点累加字模步进
(define measure (lambda (font s)
    (define n (slen s))
    (define i 0)
    (define w 0)
    (while (< i n)
        (set! w (+ w (adv font (String.charCodeAt i s))))
        (set! i (+ i 1)))
    w))

;; C 的 gfx_font_draw_text 把 y 当"行框顶"；灵机 (Gfx.draw_text) 把 y 当"墨迹顶"，
;; 差值 = 基线 - 字模上高（与 am_native_Gfx.c 的 native_text_top_gap 同式）。
(define calc_gap (lambda (font)
    (define fi (Gfx.font_info font))
    (define base (get_item fi 3))
    (define m (Gfx.glyph_metrics font 20013))   ;; '中'
    (define asc (get_item m 1))
    (define g (- base (Math.round asc)))
    (if (< g 0) 0 g)))

;; ytop = 墨迹顶
(define draw_t (lambda (font s x ytop r g b)
    (Gfx.set_font font)
    (Gfx.draw_text s x (+ ytop (if (== font F16) gap16 gap12)) r g b)))

(define draw_tc (lambda (font s cx cy r g b)
    (Gfx.draw_text_centered font s cx cy r g b 1)))

;; ============================================================================
;; 折行（黄历页用；只在切日/打开时跑一次）
;; ============================================================================

;; 「不能出现在行首」的收尾标点（避头尾规则的简化版）
(define no_start (lambda (cp)
    (or (== cp 65292)                                   ;; ，
    (or (== cp 12290)                                   ;; 。
    (or (== cp 12289)                                   ;; 、
    (or (== cp 65307)                                   ;; ；
    (or (== cp 65306)                                   ;; ：
    (or (== cp 65281)                                   ;; ！
    (or (== cp 65311)                                   ;; ？
    (or (== cp 65289)                                   ;; ）
    (or (== cp 12301)                                   ;; 」
    (or (== cp 12303)                                   ;; 』
    (or (== cp 12305)                                   ;; 】
    (or (== cp 12311)                                   ;; 》
    (== cp 8230)))))))))))))))                          ;; …

;; 按字符折行（CJK 连续文本）。返回行字符串列表。
;; 留 13px 余量：行首不该出现的标点宁可让它挤在上一行行尾（最多溢出一个字宽），
;; 也不要单独掉到下一行行首（`午时一小憩…，/ 有助于养心` 这种断法很难看）。
(define wrap_c (lambda (s maxw)
    (define out (cdr '(0)))
    (define n (slen s))
    (define i 0)
    (define cur "")
    (define w 0)
    (define ch "")
    (define cw 0)
    (define cp 0)
    (define mw (- maxw 13))
    (while (< i n)
        (set! ch (String.charAt s i))
        (set! cp (String.charCodeAt i s))
        (set! cw (adv F12 cp))
        (if (and (and (> (+ w cw) mw) (> (slen cur) 0)) (not (no_start cp)))
            (begin (push out cur) (set! cur ch) (set! w cw))
            (begin (set! cur (cat cur ch)) (set! w (+ w cw))))
        (set! i (+ i 1)))
    (if (> (slen cur) 0) (push out cur) #f)
    (if (== (length out) 0) (push out "") #f)
    out))

;; 按空格切词
(define split_sp (lambda (s)
    (define out (cdr '(0)))
    (define n (slen s))
    (define i 0)
    (define cur "")
    (define ch "")
    (while (< i n)
        (set! ch (String.charAt s i))
        (if (seq ch " ")
            (begin (if (> (slen cur) 0) (push out cur) #f) (set! cur ""))
            (set! cur (cat cur ch)))
        (set! i (+ i 1)))
    (if (> (slen cur) 0) (push out cur) #f)
    out))

;; 按词折行（空白分隔的词条：宜/忌/神煞名/彭祖百忌）
(define wrap_w (lambda (s maxw)
    (define words (split_sp s))
    (define n (length words))
    (define i 0)
    (define out (cdr '(0)))
    (define cur "")
    (define w 0)
    (define wd "")
    (define ww 0)
    (while (< i n)
        (set! wd (get_item words i))
        (set! ww (measure F12 wd))
        (if (== (slen cur) 0)
            (begin (set! cur wd) (set! w ww))
            (if (> (+ (+ w 6) ww) maxw)
                (begin (push out cur) (set! cur wd) (set! w ww))
                (begin (set! cur (cat (cat cur " ") wd)) (set! w (+ (+ w 6) ww)))))
        (set! i (+ i 1)))
    (if (> (slen cur) 0) (push out cur) #f)
    (if (== (length out) 0) (push out "") #f)
    out))

;; 清单拼成一行（空格分隔）
(define join_sp (lambda (L)
    (define n (length L))
    (define i 0)
    (define out "")
    (while (< i n)
        (if (> i 0) (set! out (cat out " ")) #f)
        (set! out (cat out (get_item L i)))
        (set! i (+ i 1)))
    out))

;; 把 s 里的单字符 from 换成 to
(define rep_char (lambda (s from to)
    (define n (slen s))
    (define i 0)
    (define out "")
    (define ch "")
    (while (< i n)
        (set! ch (String.charAt s i))
        (if (seq ch from) (set! out (cat out to)) (set! out (cat out ch)))
        (set! i (+ i 1)))
    out))

;; ============================================================================
;; 时间（UTC+8 本地民用时）
;; ============================================================================

(define update_now (lambda ()
    (define ud (Time.utc_days))
    (define us (+ (Time.utc_sod) TZ_SEC))
    (define d ud)
    (define s us)
    (define z 0) (define era 0) (define doe 0) (define t1 0) (define t2 0)
    (define yoe 0) (define yy 0) (define doy 0) (define mp 0) (define mo 0)
    (if (>= s 86400) (begin (set! s (- s 86400)) (set! d (+ d 1))) #f)
    ;; 整数除法一律用 AL.idiv：32 位宿主浮点尾数只有约 19 位，大整数会丢 1。
    (set! z (+ d 719468))
    (set! era (AL.idiv z 146097))
    (set! doe (- z (* era 146097)))
    (set! t1 (- doe (AL.idiv doe 1460)))
    (set! t2 (+ t1 (AL.idiv doe 36524)))
    (set! t1 (- t2 (AL.idiv doe 146096)))
    (set! yoe (AL.idiv t1 365))
    (set! yy (+ yoe (* era 400)))
    (set! t1 (+ (* 365 yoe) (AL.idiv yoe 4)))
    (set! t2 (- t1 (AL.idiv yoe 100)))
    (set! doy (- doe t2))
    (set! mp (AL.idiv (+ (* 5 doy) 2) 153))
    (set! now_d (+ (- doy (AL.idiv (+ (* 153 mp) 2) 5)) 1))
    (set! mo (if (< mp 10) (+ mp 3) (- mp 9)))
    (if (<= mo 2) (set! yy (+ yy 1)) #f)
    (set! now_y yy)
    (set! now_m mo)
    (set! now_h (AL.idiv s 3600))
    (set! now_min (AL.idiv (- s (* now_h 3600)) 60))
    #undefined))

;; ============================================================================
;; 农历 / 节气小工具
;; ============================================================================

(define MONTH_CN '("正" "二" "三" "四" "五" "六" "七" "八" "九" "十" "冬" "腊"))
(define DAY_P '("初" "十" "廿" "三"))
(define NUM_CN '("一" "二" "三" "四" "五" "六" "七" "八" "九" "十"))
(define ANIMALS '("鼠" "牛" "虎" "兔" "龙" "蛇" "马" "羊" "猴" "鸡" "狗" "猪"))
(define BRANCHES '("子" "丑" "寅" "卯" "辰" "巳" "午" "未" "申" "酉" "戌" "亥"))
(define WEEK_NAMES '("一" "二" "三" "四" "五" "六" "日"))
(define PILLARS '("年柱" "月柱" "日柱" "时柱"))

;; 农历月名（1..12）→ 正月/二月/…/冬月/腊月
(define lunar_month_name (lambda (m)
    (if (or (< m 1) (> m 12))
        ""
        (cat (get_item MONTH_CN (- m 1)) "月"))))

;; 农历日名（1..30）→ 初一…初十 / 十一…二十 / 廿一…廿九 / 三十
;; 末位数字取 (d-1) mod 10（1..9→一..九，10/20/30→十），前缀按区间换 初/十/廿/三。
(define lunar_day_name (lambda (d)
    (define p (get_item DAY_P 0))
    (define nc (get_item NUM_CN (AL.imod (- d 1) 10)))
    (if (>= d 11) (set! p (get_item DAY_P 1)) #f)
    (if (>= d 21) (set! p (get_item DAY_P 2)) #f)
    (if (== d 30) (begin (set! p (get_item DAY_P 3)) (set! nc (get_item NUM_CN 9))) #f)
    (cat p nc)))

;; 节气表缓存（按年；网格一次绘制里年份最多变一次）
(define tcy 0)
(define tlist (cdr '(0)))

(define term_cache (lambda (y)
    (if (not (== y tcy))
        (begin (set! tlist (AL.year_solar y)) (set! tcy y))
        #f)
    tlist))

;; "M月D日" 的紧凑格式（补零：03-05）
(define mmdd (lambda (m d)
    (define s "")
    (set! s (String.atom_to_string m))
    (if (< m 10) (set! s (cat "0" s)) #f)
    (set! s (cat s "-"))
    (if (< d 10) (set! s (cat s "0")) #f)
    (cat s (String.atom_to_string d))))

;; 该日的节气名（无则空串）
(define term_name_of (lambda (y m d)
    (define out "")
    (define i 0)
    (define t #f)
    (if (and (>= y 1901) (<= y 2100))
        (begin
          (set! i 0)
          (while (< i 24)
              (set! t (get_item (term_cache y) i))
              (if (and (== (get_item t 0) m) (== (get_item t 1) d))
                  (begin (set! out (get_item AD.SOLAR_TERMS_NAME i)) (set! i 24))
                  (set! i (+ i 1)))))
        #f)
    out))

;; 本月 1 日是星期几（周一=0..周日=6；Sakamoto 算法）
(define first_weekday (lambda (y m)
    (define t '(0 3 2 5 0 3 5 1 4 6 2 4))
    (define yy (- y (if (< m 3) 1 0)))
    (define q (+ (+ yy (AL.idiv yy 4))
                 (+ (- 0 (AL.idiv yy 100))
                    (+ (AL.idiv yy 400)
                       (+ (get_item t (- m 1)) 1)))))
    (define dow (AL.imod q 7))
    (AL.imod (+ dow 6) 7)))

;; ---- 日期格农历标签 --------------------------------------------------------
;; 返回标签字符串；副产物 cl_kind（0=日名 1=月名 2=节气）
(define cell_label (lambda (y m d)
    (define LD #f)
    (define tn "")
    (define ld 0)
    (define lm 0)
    (define leap 0)
    (define out "")
    (set! cl_kind 0)
    (if (and (>= y 1901) (<= y 2100))
        (set! LD (AL.lunar_date y m d))
        #f)
    (if (not (== LD #f))
        (begin
          (set! lm (get_item LD 1))
          (set! ld (get_item LD 2))
          (set! leap (get_item LD 3)))
        #f)
    (set! tn (term_name_of y m d))
    (if (== LD #f)
        (if (not (seq tn ""))
            (begin (set! out tn) (set! cl_kind 2))
            (set! out ""))
        (if (== ld 1)
            (begin
              (set! out (cat (if (== leap 1) "闰" "") (lunar_month_name lm)))
              (set! cl_kind 1))
            (if (not (seq tn ""))
                (begin (set! out tn) (set! cl_kind 2))
                (set! out (lunar_day_name ld)))))
    out))

;; ============================================================================
;; 日历页
;; ============================================================================

(define rebuild_texts (lambda ()
    (set! cal_title (cat (String.atom_to_string cal_y)
                    (cat "年" (cat (String.atom_to_string cal_m) "月"))))
    #undefined))

;; 顶栏右侧的「干支+生肖年」（用本月 15 日的农历年，跨年月份不至于摇摆）
(define year_sub_text (lambda ()
    (define LD #f)
    (define ly 0)
    (define out "")
    (if (and (>= cal_y 1901) (<= cal_y 2100))
        (begin
          (set! LD (AL.lunar_date cal_y cal_m 15))
          (if (== LD #f) (set! LD (AL.lunar_date cal_y cal_m 1)) #f)
          (if (not (== LD #f))
              (begin
                (set! ly (get_item LD 0))
                (set! out (cat (AL.g60_at (AL.imod (- ly 4) 60))
                          (cat (get_item ANIMALS (AL.imod (- ly 4) 12)) "年"))))
              #f))
        #f)
    out))

(define draw_header (lambda ()
    (define s (year_sub_text))
    (Gfx.fill_rect 0 0 SW HDR_H HDR_R HDR_G HDR_B)
    (draw_t F16 cal_title 10 4 TXT_R TXT_G TXT_B)
    (draw_t F12 s (- (- SW 60) (measure F12 s)) 7 DIM_R DIM_G DIM_B)
    (Gfx.fill_rect 0 (- HDR_H 1) SW 1 EDGE_R EDGE_G EDGE_B)
    #undefined))

(define draw_week (lambda ()
    (define i 0)
    (define x0 0)
    (define cx 0)
    (define s "")
    (while (< i 7)
        (set! x0 (AL.idiv (* i SW) 7))
        (set! cx (AL.idiv (+ x0 (AL.idiv (* (+ i 1) SW) 7)) 2))
        (set! s (get_item WEEK_NAMES i))
        (if (>= i 5)
            (draw_tc F12 s cx (+ WEEK_Y 7) SUN_R SUN_G SUN_B)
            (draw_tc F12 s cx (+ WEEK_Y 7) DIM_R DIM_G DIM_B))
        (set! i (+ i 1)))
    (Gfx.fill_rect 0 (+ WEEK_Y WEEK_H) SW 1 EDGE_R EDGE_G EDGE_B)
    #undefined))

(define draw_footer (lambda ()
    (Gfx.fill_rect 0 FTR_Y SW (- SH FTR_Y) FTR_R FTR_G FTR_B)
    (Gfx.fill_rect 0 FTR_Y SW 1 EDGE_R EDGE_G EDGE_B)
    (BTN.draw_state b_py (BTN.armed b_py))
    (BTN.draw_state b_pm (BTN.armed b_pm))
    (BTN.draw_state b_td (BTN.armed b_td))
    (BTN.draw_state b_nm (BTN.armed b_nm))
    (BTN.draw_state b_ny (BTN.armed b_ny))
    #undefined))

(define draw_grid (lambda ()
    (define fw 0)
    (define start 0)
    (define i 0)
    (define col 0)
    (define row 0)
    (define dn 0)
    (define t #f)
    (define gy 0) (define gm 0) (define gd 0)
    (define x0 0) (define x1 0) (define cx 0) (define ry 0)
    (define in_cur #f)
    (define is_today #f)
    (define label "")
    (define ncol 0)
    (set! fw (first_weekday cal_y cal_m))
    (set! start (- (AL.daynum cal_y cal_m 1) fw))
    (set! i 0)
    (while (< i 42)
        (set! dn (+ start i))
        (set! t (AL.ymd dn))
        (set! gy (get_item t 0))
        (set! gm (get_item t 1))
        (set! gd (get_item t 2))
        (set! col (AL.imod i 7))
        (set! row (AL.idiv i 7))
        (set! x0 (AL.idiv (* col SW) 7))
        (set! x1 (AL.idiv (* (+ col 1) SW) 7))
        (set! cx (AL.idiv (+ x0 x1) 2))
        (set! ry (+ GRID_TOP (* row ROW_H)))
        (set! in_cur (and (== gy cal_y) (== gm cal_m)))
        (set! is_today (and (and (== gy now_y) (== gm now_m)) (== gd now_d)))
        ;; ---- 底板：今天（蓝块） / 最近查看（描边环） ----
        (if is_today
            (Gfx.rounded_rect (- cx 12) (+ ry 1) 25 16 5 5 5 5 ACC_R ACC_G ACC_B 1)
            (if (== dn sel_dn)
                (begin
                  (Gfx.rounded_rect (- cx 12) (+ ry 1) 25 16 5 5 5 5 ACC_R ACC_G ACC_B 1)
                  (Gfx.rounded_rect (- cx 11) (+ ry 2) 23 14 4 4 4 4 BG_R BG_G BG_B 1))
                #f))
        ;; ---- 日 ----
        (set! ncol 0)
        (if (not in_cur) (set! ncol 3) #f)
        (if (and in_cur (>= col 5)) (set! ncol 2) #f)
        (if is_today (set! ncol 1) #f)
        (if (== ncol 0) (draw_tc F16 (get_item NUM_S (- gd 1)) cx (+ ry 8) TXT_R TXT_G TXT_B) #f)
        (if (== ncol 1) (draw_tc F16 (get_item NUM_S (- gd 1)) cx (+ ry 8) 255 255 255) #f)
        (if (== ncol 2) (draw_tc F16 (get_item NUM_S (- gd 1)) cx (+ ry 8) SUN_R SUN_G SUN_B) #f)
        (if (== ncol 3) (draw_tc F16 (get_item NUM_S (- gd 1)) cx (+ ry 8) FNT_R FNT_G FNT_B) #f)
        ;; ---- 农历 ----
        (set! label (cell_label gy gm gd))
        (if is_today
            (draw_tc F12 label cx (+ ry 21) 208 224 245)
            (if in_cur
                (if (== cl_kind 1)
                    (draw_tc F12 label cx (+ ry 21) ACC2_R ACC2_G ACC2_B)
                    (if (== cl_kind 2)
                        (draw_tc F12 label cx (+ ry 21) TERM_R TERM_G TERM_B)
                        (draw_tc F12 label cx (+ ry 21) DIM_R DIM_G DIM_B)))
                (draw_tc F12 label cx (+ ry 21) FNT2_R FNT2_G FNT2_B)))
        (set! i (+ i 1)))
    #undefined))

(define draw_cal (lambda ()
    (Gfx.fill_rect 0 0 SW SH BG_R BG_G BG_B)
    (draw_header)
    (draw_week)
    (draw_grid)
    (draw_footer)
    (EX.draw_button)
    #undefined))

;; ---- 换月 / 换年 / 回今天 ----------------------------------------------
;; 年月夹到黄历可计算区间（1901-2100）。年份、月份分别夹，不互相牵连。
(define clamp_ym (lambda ()
    (if (< cal_y 1901) (set! cal_y 1901) #f)
    (if (> cal_y 2100) (set! cal_y 2100) #f)
    (if (< cal_m 1) (set! cal_m 1) #f)
    (if (> cal_m 12) (set! cal_m 12) #f)
    #undefined))

;; 切月：越界到 1900/2101 时**不动**（保持在边界月，不做"钳回同一年"这种怪动作）
(define step_month (lambda (k)
    (define ny cal_y)
    (define nm (+ cal_m k))
    (if (< nm 1) (begin (set! nm 12) (set! ny (- cal_y 1))) #f)
    (if (> nm 12) (begin (set! nm 1) (set! ny (+ cal_y 1))) #f)
    (if (and (>= ny 1901) (<= ny 2100))
        (begin
          (set! cal_y ny)
          (set! cal_m nm)
          (rebuild_texts)
          (set! cal_dirty 1))
        #f)
    #undefined))

(define step_year (lambda (k)
    (define ny (+ cal_y k))
    (if (and (>= ny 1901) (<= ny 2100))
        (begin (set! cal_y ny) (rebuild_texts) (set! cal_dirty 1))
        #f)
    #undefined))

(define go_today (lambda ()
    (set! cal_y now_y)
    (set! cal_m now_m)
    (clamp_ym)
    (set! sel_dn (AL.daynum now_y now_m now_d))
    (rebuild_texts)
    (set! cal_dirty 1)
    #undefined))

;; ---- 日期格命中：返回该格的天序号（1970 起的天数，**1970 之前是负数**），
;;      未命中返回 #f（不能拿 0/负数当哨兵：1901-1969 的合法天序号就是负数）--------
(define hit_day (lambda (tx ty)
    (define dn #f)
    (define col 0)
    (define row 0)
    (define fw 0)
    (if (or (< ty GRID_TOP) (>= ty GRID_BOT))
        (set! dn #f)
        (begin
          (set! col (AL.idiv (* tx 7) SW))
          (set! row (AL.idiv (- ty GRID_TOP) ROW_H))
          (if (or (< col 0) (> col 6))
              (set! dn #f)
              (if (or (< row 0) (> row 5))
                  (set! dn #f)
                  (begin
                    (set! fw (first_weekday cal_y cal_m))
                    (set! dn (+ (- (AL.daynum cal_y cal_m 1) fw) (+ (* row 7) col))))))))
    dn))

;; ============================================================================
;; 黄历页：绘制清单（bg/fg/tx 三层，滚动时只重放）
;; ============================================================================

(define alm_bg (cdr '(0)))
(define alm_fg (cdr '(0)))
(define alm_tx (cdr '(0)))

;; 绘制项 = 10 元列表：
;;   (种类 y x a b r g b 附加 串)
;;   种类 0 实心圆角矩形：x y w=a h=b 圆角=附加
;;   种类 1 文本（墨迹顶）：x y 字体=a
;;   种类 2 居中文本：cx=x cy=y 字体=a（a 为字模号）
;;   种类 3 实心圆：cx=x cy=y 半径=a
(define IT (lambda (k y x a b r g bb e s)
    (cons k (cons y (cons x (cons a (cons b
      (cons r (cons g (cons bb (cons e (cons s (cdr '(0))))))))))))))

(define lst_clr (lambda (L) (while (> (length L) 0) (pop L) #f)))

(define bg_rect (lambda (x y w h rad r g bb) (push alm_bg (IT 0 y x w h r g bb rad ""))))
(define fg_rect (lambda (x y w h rad r g bb) (push alm_fg (IT 0 y x w h r g bb rad ""))))
(define tx (lambda (x y font s r g bb) (push alm_tx (IT 1 y x font 0 r g bb 0 s))))
(define txc (lambda (cx cy font s r g bb) (push alm_tx (IT 2 cy cx font 0 r g bb 0 s))))

;; 折行推送（返回新的 y）
(define tx_wrap (lambda (x y maxw s r g bb)
    (define lines (wrap_w s maxw))
    (define n (length lines))
    (define i 0)
    (while (< i n)
        (tx x (+ y (* i LH12)) F12 (get_item lines i) r g bb)
        (set! i (+ i 1)))
    (+ y (* n LH12))))

(define tx_wrapc (lambda (x y maxw s r g bb)
    (define lines (wrap_c s maxw))
    (define n (length lines))
    (define i 0)
    (while (< i n)
        (tx x (+ y (* i LH12)) F12 (get_item lines i) r g bb)
        (set! i (+ i 1)))
    (+ y (* n LH12))))

;; 卡片底板（内容排完后才知道高度，故后置；bg 层先画）
(define card_bg (lambda (y0 y1)
    (bg_rect CARD_X y0 CARD_W (- y1 y0) 8 CARD_R CARD_G CARD_B)))

;; 卡片标题行（强调竖条 + 标题），返回正文起始 y
(define sec_head (lambda (y t)
    (fg_rect 12 (+ y 7) 2 11 1 ACC_R ACC_G ACC_B)
    (tx 18 (+ y 6) F12 t ACC_R ACC_G ACC_B)
    (+ y 23)))

;; ============================================================================
;; 黄历页：字段准备
;; ============================================================================

(define a_lunar_big "")
(define a_year_cn "")
(define a_animal "")
(define a_zodiac "")
(define a_star "")
(define a_officer "")
(define a_god "")
(define a_overall "吉")
(define a_fly "")
(define a_fetal "")
(define a_peng "")
(define a_week "")
(define a_good "")
(define a_bad "")
(define a_ggod "")
(define a_bgod "")
(define a_org "")
(define a_org_yi "")
(define a_org_ji "")
(define a_org_note "")
(define a_cur_term "")
(define a_next_term "")
(define a_bazi "")
(define a_hours (cdr '(0)))
(define a_dirs (cdr '(0)))
(define a_date "")
(define a_jishi "")
(define bc_kind 0)

;; 某农历年的干支（六十甲子）名
(define gz_year (lambda (ly) (AL.g60_at (AL.imod (- ly 4) 60))))

;; 当前节气 / 下一节气（形如 "惊蛰 03-05"）
(define term_pair (lambda (y m d)
    (define L (term_cache y))
    (define i 0)
    (define cur 0)
    (define t #f)
    (define nm 0)
    (define nt #f)
    (if (AL.md_le (get_item L 0) (AL.mk2 m d))
        (begin
          (while (< i 24)
              (if (AL.md_le (get_item L i) (AL.mk2 m d)) (set! cur i) #f)
              (set! i (+ i 1)))
          (set! t (get_item L cur))
          (set! a_cur_term (cat (cat (get_item AD.SOLAR_TERMS_NAME cur) " ")
                           (mmdd (get_item t 0) (get_item t 1))))
          (set! nm (AL.imod (+ cur 1) 24))
          (if (== (AL.idiv (+ cur 1) 24) 0)
              (set! nt (get_item L nm))
              (set! nt (get_item (AL.year_solar (+ y 1)) nm)))
          (set! a_next_term (cat (cat (get_item AD.SOLAR_TERMS_NAME nm) " ")
                            (mmdd (get_item nt 0) (get_item nt 1)))))
        (begin
          ;; 早于当年第一个节气（1 月初）：当前仍是去年冬至
          (set! nt (get_item L 0))
          (set! a_cur_term "冬至 12-21")
          (set! a_next_term (cat (cat (get_item AD.SOLAR_TERMS_NAME 0) " ")
                            (mmdd (get_item nt 0) (get_item nt 1))))))
    #undefined))

;; 从 alm_R 取出/拼出全部显示字符串
(define alm_prepare (lambda ()
    (define R alm_R)
    (define LD #f)
    (define ly 0)
    (define lm 0)
    (define ld 0)
    (define leap 0)
    (set! LD (AL.lunar R))
    (set! ly (get_item LD 0))
    (set! lm (get_item LD 1))
    (set! ld (get_item LD 2))
    (set! leap (get_item LD 3))
    (set! a_year_cn (cat (gz_year ly) "年"))
    (set! a_animal (get_item ANIMALS (AL.imod (- ly 4) 12)))
    (set! a_lunar_big (cat (cat (if (== leap 1) "闰" "") (lunar_month_name lm)) (lunar_day_name ld)))
    (set! a_zodiac (AL.zodiac R))
    (set! a_star (AL.star28 R))
    (set! a_officer (AL.officer R))
    (set! a_god (AL.god R))
    (set! a_overall (AL.overall R))
    (set! a_fly (AL.fly R))
    (set! a_fetal (AL.fetal R))
    (set! a_peng (rep_char (AL.peng R) "," " "))
    (set! a_week (AL.week R))
    (set! a_good (join_sp (AL.good R)))
    (set! a_bad (join_sp (AL.bad R)))
    (set! a_ggod (join_sp (AL.goodgods R)))
    (set! a_bgod (join_sp (AL.badgods R)))
    (set! a_org (AL.meridian R))
    (set! a_org_yi (AL.meridian_yi R))
    (set! a_org_ji (AL.meridian_ji R))
    (set! a_org_note (AL.meridian_note R))
    (set! a_bazi (AL.bazi R))
    (set! a_hours (AL.hourluck R))
    (set! a_dirs (AL.dirs R))
    (set! a_cur_term "")
    (set! a_next_term "")
    (term_pair alm_y alm_m alm_d)
    (set! a_jishi (jishi_text))
    #undefined))

;; 吉时清单（"子丑寅" 或 "无"）
(define jishi_text (lambda ()
    (define n (length a_hours))
    (define i 0)
    (define out "")
    (define any 0)
    (if (> n 12) (set! n 12) #f)
    (while (< i n)
        (if (seq (get_item a_hours i) "吉")
            (begin (set! out (cat out (get_item BRANCHES i))) (set! any 1))
            #f)
        (set! i (+ i 1)))
    (if (== any 0) (set! out "无") #f)
    out))

;; ============================================================================
;; 黄历页：分卡片布局
;; ============================================================================

;; ① 农历概览
(define sec_lunar (lambda (y)
    (define y1 (+ y 6))
    (define s "")
    (tx CARD_IX y1 F16 a_lunar_big TXT_R TXT_G TXT_B)
    (set! s (cat a_year_cn (cat " · 属" a_animal)))
    (tx (- CARD_RGT (measure F12 s)) y1 F12 s DIM_R DIM_G DIM_B)
    (set! y1 (+ y1 20))
    (tx CARD_IX y1 F12 (cat "星座 " (cat a_zodiac (cat " · 星宿 " a_star))) DIM_R DIM_G DIM_B)
    (set! y1 (+ y1 LH12))
    (set! s (cat "节气 " (cat a_cur_term " · 下个 ")))
    (tx CARD_IX y1 F12 s DIM_R DIM_G DIM_B)
    (tx (+ CARD_IX (measure F12 s)) y1 F12 a_next_term TERM_R TERM_G TERM_B)
    (set! y1 (+ y1 LH12))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; ② 干支四柱
(define sec_bazi (lambda (y)
    (define y1 (sec_head y "干支四柱"))
    (define i 0)
    (define x 0)
    (define val "")
    (define vr 0) (define vg 0) (define vb 0)
    (while (< i 4)
        (set! x (+ CARD_IX (* i 70)))
        (set! val (String.slice a_bazi (* i 2) (+ (* i 2) 2)))
        (set! vr TXT_R) (set! vg TXT_G) (set! vb TXT_B)
        (if (== i 2) (begin (set! vr TERM_R) (set! vg TERM_G) (set! vb TERM_B)) #f)
        (tx x y1 F12 (get_item PILLARS i) DIM_R DIM_G DIM_B)
        (tx (+ x 26) y1 F12 val vr vg vb)
        (set! i (+ i 1)))
    (set! y1 (+ y1 LH12))
    (tx CARD_IX y1 F12 "建除" DIM_R DIM_G DIM_B)
    (tx (+ CARD_IX 28) y1 F12 (cat a_officer "日") TXT_R TXT_G TXT_B)
    (tx (+ CARD_IX 96) y1 F12 "值神" DIM_R DIM_G DIM_B)
    (tx (+ CARD_IX 124) y1 F12 a_god TXT_R TXT_G TXT_B)
    (set! y1 (+ y1 LH12))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; ③ 宜忌
(define sec_yiji (lambda (y)
    (define y1 (sec_head y "宜忌"))
    (fg_rect CARD_IX (+ y1 1) 18 14 3 28 74 56)
    (txc (+ CARD_IX 9) (+ y1 8) F12 "宜" GRN_R GRN_G GRN_B)
    (set! y1 (tx_wrap (+ CARD_IX 24) y1 (- CARD_IW 24) a_good 176 226 196))
    (set! y1 (+ y1 3))
    (fg_rect CARD_IX (+ y1 1) 18 14 3 78 36 48)
    (txc (+ CARD_IX 9) (+ y1 8) F12 "忌" RED_R RED_G RED_B)
    (set! y1 (tx_wrap (+ CARD_IX 24) y1 (- CARD_IW 24) a_bad 226 176 190))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; ④ 时辰吉凶
(define sec_hours (lambda (y)
    (define y1 (sec_head y "时辰吉凶"))
    (define i 0)
    (define col 0)
    (define row 0)
    (define x 0)
    (define yy 0)
    (define ok #f)
    (while (< i 12)
        (set! col (AL.imod i 6))
        (set! row (AL.idiv i 6))
        (set! x (+ CARD_IX (* col 47)))
        (set! yy (+ y1 (* row 20)))
        (set! ok (seq (get_item a_hours i) "吉"))
        (if ok
            (fg_rect x yy 45 18 4 24 54 44)
            (fg_rect x yy 45 18 4 54 26 34))
        (tx (+ x 8) (+ yy 2) F12 (get_item BRANCHES i) TXT_R TXT_G TXT_B)
        (if ok
            (tx (+ x 25) (+ yy 2) F12 "吉" GRN_R GRN_G GRN_B)
            (tx (+ x 25) (+ yy 2) F12 "凶" RED_R RED_G RED_B))
        (set! i (+ i 1)))
    (set! y1 (+ y1 42))
    (tx CARD_IX y1 F12 (cat "吉时 " a_jishi) DIM_R DIM_G DIM_B)
    (set! y1 (+ y1 LH12))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; ⑤ 吉神方位
(define sec_dirs (lambda (y)
    (define y1 (sec_head y "吉神方位"))
    (define n (length a_dirs))
    (define i 0)
    (define x 0)
    (define yy 0)
    (while (< i n)
        (set! x (+ CARD_IX (* (AL.imod i 3) 94)))
        (set! yy (+ y1 (* (AL.idiv i 3) 20)))
        (fg_rect x yy 90 18 4 CHIP_R CHIP_G CHIP_B)
        (tx (+ x 6) (+ yy 2) F12 (get_item a_dirs i) 176 226 196)
        (set! i (+ i 1)))
    (set! y1 (+ y1 (* (AL.idiv (+ n 2) 3) 20)))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; ⑥ 胎神 · 彭祖百忌
(define sec_fetal (lambda (y)
    (define y1 (sec_head y "胎神 · 彭祖百忌"))
    (tx CARD_IX y1 F12 "胎神" DIM_R DIM_G DIM_B)
    (tx (+ CARD_IX 34) y1 F12 a_fetal TXT_R TXT_G TXT_B)
    (set! y1 (+ y1 LH12))
    (tx CARD_IX y1 F12 "彭祖" DIM_R DIM_G DIM_B)
    (set! y1 (tx_wrap (+ CARD_IX 34) y1 (- CARD_IW 34) a_peng TXT_R TXT_G TXT_B))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; ⑦ 九宫飞星
(define sec_fly (lambda (y)
    (define y1 (sec_head y "九宫飞星"))
    (define i 0)
    (define x 0)
    (define ch "")
    (define cr 0) (define cg 0) (define cb 0)
    (while (< i 9)
        (set! x (+ CARD_IX (* i 31)))
        (fg_rect x y1 28 18 4 CHIP_R CHIP_G CHIP_B)
        (set! ch (String.charAt a_fly i))
        (set! cr TXT_R) (set! cg TXT_G) (set! cb TXT_B)
        (if (seq ch "5") (begin (set! cr TERM_R) (set! cg TERM_G) (set! cb TERM_B)) #f)
        (txc (+ x 14) (+ y1 8) F12 ch cr cg cb)
        (set! i (+ i 1)))
    (set! y1 (+ y1 20))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; ⑧⑨ 吉神宜趋 / 凶煞宜忌
(define sec_gods (lambda (y)
    (define y1 (sec_head y "吉神宜趋"))
    (set! y1 (tx_wrap CARD_IX y1 CARD_IW a_ggod 176 226 196))
    (set! y1 (+ y1 4))
    (fg_rect 12 (+ y1 7) 2 11 1 190 84 104)
    (tx 18 (+ y1 6) F12 "凶煞宜忌" RED_R RED_G RED_B)
    (set! y1 (+ y1 23))
    (set! y1 (tx_wrap CARD_IX y1 CARD_IW a_bgod 226 176 190))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; ⑩ 子午流注养生
(define sec_mer (lambda (y)
    (define y1 (sec_head y "子午流注 · 养生"))
    (tx CARD_IX y1 F12 "当令" DIM_R DIM_G DIM_B)
    (tx (+ CARD_IX 34) y1 F12 (cat a_org "经") TERM_R TERM_G TERM_B)
    (set! y1 (+ y1 LH12))
    (tx CARD_IX y1 F12 "宜" GRN_R GRN_G GRN_B)
    (set! y1 (tx_wrapc (+ CARD_IX 20) y1 (- CARD_IW 20) a_org_yi 176 226 196))
    (tx CARD_IX y1 F12 "忌" RED_R RED_G RED_B)
    (set! y1 (tx_wrapc (+ CARD_IX 20) y1 (- CARD_IW 20) a_org_ji 226 176 190))
    (set! y1 (+ y1 2))
    (set! y1 (tx_wrapc CARD_IX y1 CARD_IW a_org_note 128 140 156))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; 越界错误页
(define sec_err (lambda (y)
    (define y1 (+ y 6))
    (txc (AL.idiv SW 2) (+ y1 24) F12 "该日期超出黄历计算范围" TXT_R TXT_G TXT_B)
    (txc (AL.idiv SW 2) (+ y1 44) F12 "可用下方按钮换到 1901-2100 之间" DIM_R DIM_G DIM_B)
    (set! y1 (+ y1 64))
    (card_bg y (+ y1 6))
    (+ y1 12)))

;; 全量布局（切日/打开时一次）
(define alm_build (lambda ()
    (define y 0)
    (lst_clr alm_bg)
    (lst_clr alm_fg)
    (lst_clr alm_tx)
    (set! y 8)
    (if (== alm_err 1)
        (set! y (sec_err y))
        (begin
          (set! y (sec_lunar y))
          (set! y (sec_bazi y))
          (set! y (sec_yiji y))
          (set! y (sec_hours y))
          (set! y (sec_dirs y))
          (set! y (sec_fetal y))
          (set! y (sec_fly y))
          (set! y (sec_gods y))
          (set! y (sec_mer y))))
    (set! alm_total (+ y 8))
    (set! alm_maxs (- alm_total ALM_VH))
    (if (< alm_maxs 0) (set! alm_maxs 0) #f)
    (if (> alm_fpos alm_maxs) (set! alm_fpos alm_maxs) #f)
    (set! alm_scroll (Math.floor alm_fpos))
    #undefined))

;; ============================================================================
;; 黄历页：绘制
;; ============================================================================

;; 绘制一张清单（按滚动偏移；视口外的项跳过）
(define draw_items (lambda (L)
    (define n (length L))
    (define i 0)
    (define it #f)
    (define k 0)
    (define yy 0)
    (define xx 0)
    (define hh 0)
    (define r 0)
    (define g 0)
    (define bb 0)
    (define a 0)
    (define b 0)
    (define e 0)
    (define s "")
    (while (< i n)
        (set! it (get_item L i))
        (set! k (get_item it 0))
        (set! yy (- (+ (get_item it 1) ALM_TOP) alm_scroll))
        (set! xx (get_item it 2))
        (set! a (get_item it 3))
        (set! b (get_item it 4))
        (set! r (get_item it 5))
        (set! g (get_item it 6))
        (set! bb (get_item it 7))
        (set! e (get_item it 8))
        (set! s (get_item it 9))
        (set! hh 20)
        (if (== k 0) (set! hh b) #f)
        (if (== k 3) (set! hh (* 2 a)) #f)
        (if (< (+ yy hh) (- ALM_TOP 4))
            (set! i (+ i 1))
            (begin
              (if (> yy ALM_BOT)
                  #f
                  (begin
                    (if (== k 0) (Gfx.rounded_rect xx yy a b e e e e r g bb 1) #f)
                    (if (== k 1) (draw_t a s xx yy r g bb) #f)
                    (if (== k 2) (Gfx.draw_text_centered a s xx yy r g bb 1) #f)
                    (if (== k 3) (Gfx.fill_circle_aa xx yy a r g bb 1) #f)))
              (set! i (+ i 1)))))
    #undefined))

(define badge_kind (lambda ()
    (set! bc_kind 1)
    (if (seq a_overall "大吉") (set! bc_kind 0) #f)
    (if (seq a_overall "平") (set! bc_kind 2) #f)
    (if (seq a_overall "凶") (set! bc_kind 3) #f)
    #undefined))

(define draw_badge (lambda (x y s)
    (define w (+ (measure F12 s) 14))
    (if (== bc_kind 0)
        (begin (Gfx.rounded_rect x y w 17 5 5 5 5 66 52 18 1)
               (Gfx.draw_text_centered F12 s (+ x (AL.idiv w 2)) (+ y 9) GOL_R GOL_G GOL_B 1))
        #f)
    (if (== bc_kind 1)
        (begin (Gfx.rounded_rect x y w 17 5 5 5 5 24 54 44 1)
               (Gfx.draw_text_centered F12 s (+ x (AL.idiv w 2)) (+ y 9) GRN_R GRN_G GRN_B 1))
        #f)
    (if (== bc_kind 2)
        (begin (Gfx.rounded_rect x y w 17 5 5 5 5 58 52 22 1)
               (Gfx.draw_text_centered F12 s (+ x (AL.idiv w 2)) (+ y 9) YEL_R YEL_G YEL_B 1))
        #f)
    (if (== bc_kind 3)
        (begin (Gfx.rounded_rect x y w 17 5 5 5 5 70 28 38 1)
               (Gfx.draw_text_centered F12 s (+ x (AL.idiv w 2)) (+ y 9) RED_R RED_G RED_B 1))
        #f)
    #undefined))

(define draw_alm_header (lambda ()
    (define tw 0)
    (define bw 0)
    (Gfx.fill_rect 0 0 SW HDR_H HDR_R HDR_G HDR_B)
    (BTN.draw_state a_back (BTN.armed a_back))
    (draw_t F16 a_date 50 4 TXT_R TXT_G TXT_B)
    (set! tw (+ 50 (measure F16 a_date)))
    (if (== alm_err 1)
        (draw_t F12 "超出范围" (+ tw 8) 7 DIM_R DIM_G DIM_B)
        (begin
          (set! bw (+ (measure F12 a_overall) 14))
          (draw_badge (+ tw 8) 5 a_overall)
          (draw_t F12 a_week (+ tw (+ bw 16)) 7 DIM_R DIM_G DIM_B)))
    (Gfx.fill_rect 0 (- HDR_H 1) SW 1 EDGE_R EDGE_G EDGE_B)
    #undefined))

(define draw_scr (lambda ()
    (define trk_y 0)
    (define trk_h 0)
    (define th 0)
    (define ty 0)
    (if (> alm_total ALM_VH)
        (begin
          (set! trk_y (+ ALM_TOP 3))
          (set! trk_h (- ALM_VH 6))
          (Gfx.rounded_rect 311 trk_y 3 trk_h 1 1 1 1 CHIP_R CHIP_G CHIP_B 1)
          (set! th (Math.floor (/ (* trk_h ALM_VH) alm_total)))
          (if (< th 16) (set! th 16) #f)
          (set! ty (+ trk_y (Math.floor (* (/ alm_scroll alm_total) (- trk_h th)))))
          (Gfx.rounded_rect 311 ty 3 th 1 1 1 1 ACC_R ACC_G ACC_B 1))
        #f)
    #undefined))

(define draw_alm (lambda ()
    (Gfx.fill_rect 0 0 SW SH BG_R BG_G BG_B)
    (Gfx.set_clip 0 ALM_TOP SW ALM_VH)
    (draw_items alm_bg)
    (draw_items alm_fg)
    (draw_items alm_tx)
    (Gfx.reset_clip)
    (draw_alm_header)
    (Gfx.fill_rect 0 ALM_BOT SW 1 EDGE_R EDGE_G EDGE_B)
    (draw_scr)
    (BTN.draw_state a_prev (BTN.armed a_prev))
    (BTN.draw_state a_today (BTN.armed a_today))
    (BTN.draw_state a_next (BTN.armed a_next))
    (EX.draw_button)
    #undefined))

;; ============================================================================
;; 黄历页：动作
;; ============================================================================

(define alm_load (lambda ()
    (set! a_date (cat (cat (String.atom_to_string alm_y) "年")
                   (cat (cat (String.atom_to_string alm_m) "月")
                        (cat (String.atom_to_string alm_d) "日"))))
    (set! alm_R (AL.calc alm_y alm_m alm_d now_h now_min))
    (if (== alm_R #f) (set! alm_err 1) (set! alm_err 0))
    (if (== alm_err 0) (alm_prepare) #f)
    (badge_kind)
    (alm_build)
    (set! alm_vel 0.0)
    (set! alm_dirty 1)
    #undefined))

(define open_alm (lambda (y m d)
    (set! alm_y y)
    (set! alm_m m)
    (set! alm_d d)
    (set! alm_fpos 0.0)
    (set! sel_dn (AL.daynum y m d))
    (alm_load)
    (set! view 1)
    (set! drg_on #f)
    (set! drg_moved #f)
    (set! hold_btn -1)
    (set! prev_armed -1)
    (display "[calendar] alm ") (display y) (display "-") (display m)
    (display "-") (display d) (display " total=") (display alm_total) (newline)
    #undefined))

(define close_alm (lambda ()
    (set! view 0)
    (set! hold_btn -1)
    (set! prev_armed -1)
    (set! cal_y alm_y)
    (set! cal_m alm_m)
    (clamp_ym)
    (rebuild_texts)
    (set! cal_dirty 1)
    (display "[calendar] cal ") (display cal_y) (display "-") (display cal_m) (newline)
    #undefined))

;; 黄历页内换日（k = -1/+1）
(define alm_step (lambda (k)
    (define dn (+ (AL.daynum alm_y alm_m alm_d) k))
    (define t (AL.ymd dn))
    (define ny (get_item t 0))
    (define nm (get_item t 1))
    (define nd (get_item t 2))
    (if (and (>= ny 1901) (<= ny 2100))
        (begin
          (set! alm_y ny)
          (set! alm_m nm)
          (set! alm_d nd)
          (set! sel_dn dn)
          (alm_load))
        #f)
    #undefined))

;; 滚动到目标浮点位置（取整后绘制）
(define move_to (lambda (v)
    (define nv v)
    (if (< nv 0) (set! nv 0.0) #f)
    (if (> nv alm_maxs) (set! nv alm_maxs) #f)
    (set! alm_fpos nv)
    (if (not (== (Math.floor nv) alm_scroll))
        (begin (set! alm_scroll (Math.floor nv)) (set! alm_dirty 1))
        #f)
    #undefined))

;; ============================================================================
;; 交互
;; ============================================================================

;; 按键事件重建（等价 GUI 的 key_edge：短按松手 -1 / 长按连发 -2）
(define poll_key (lambda ()
    (define pr 0)
    (define rl 0)
    (define held 0)
    (set! ev_code 0)
    (set! ev_kind 0)
    (set! pr (Key.take_press))
    (if (not (== pr 0))
        (begin (set! k_prev pr) (set! k_t0 (Diag.ms)) (set! k_rep 0))
        #f)
    (set! rl (Key.take_release))
    (if (not (== rl 0))
        (if (and (not (== k_prev 0)) (== rl k_prev))
            (begin
              (set! held (- (Diag.ms) k_t0))
              (set! ev_code k_prev)
              (if (or (== k_rep 1) (< held 360)) (set! ev_kind 1) (set! ev_kind 2))
              (set! k_prev 0)
              (set! k_rep 0))
            #f)
        #f)
    (if (and (== ev_kind 0) (not (== k_prev 0)))
        (if (Key.is_down k_prev)
            (if (>= (- (Diag.ms) k_t0) 360)
                (begin (set! ev_code k_prev) (set! ev_kind 2) (set! k_rep 1))
                #f)
            #f)
        #f)
    #undefined))

;; 日历页按键
(define key_cal (lambda (code)
    (if (== code 3) (step_month (- 0 1)) #f)          ;; ← 上月
    (if (== code 4) (step_month 1) #f)                ;; → 下月
    (if (== code 1) (step_year (- 0 1)) #f)           ;; ↑ 上一年
    (if (== code 2) (step_year 1) #f)                 ;; ↓ 下一年
    (if (== code 48) (go_today) #f)                   ;; 0 回今天
    (if (== code 28) (go_today) #f)                   ;; Home 回今天
    (if (== code 13) (open_alm now_y now_m now_d) #f) ;; Enter 看今天的黄历
    #undefined))

;; 黄历页按键
(define key_alm (lambda (code)
    (if (== code 3) (alm_step (- 0 1)) #f)
    (if (== code 4) (alm_step 1) #f)
    (if (== code 1) (move_to (+ alm_fpos (- 0 24))) #f)
    (if (== code 2) (move_to (+ alm_fpos 24)) #f)
    (if (== code 30) (move_to (- alm_fpos ALM_VH)) #f)
    (if (== code 31) (move_to (+ alm_fpos ALM_VH)) #f)
    (if (== code 28) (move_to 0.0) #f)
    (if (== code 13) (close_alm) #f)
    (if (== code 48) (alm_jump_today) #f)
    #undefined))

(define alm_jump_today (lambda ()
    (set! alm_y now_y)
    (set! alm_m now_m)
    (set! alm_d now_d)
    (set! sel_dn (AL.daynum now_y now_m now_d))
    (alm_load)
    #undefined))

;; 日历页：底栏按钮动作（1 上年 2 上月 3 今天 4 下月 5 下年）
(define do_act (lambda (a)
    (if (== a 1) (step_year (- 0 1)) #f)
    (if (== a 2) (step_month (- 0 1)) #f)
    (if (== a 3) (go_today) #f)
    (if (== a 4) (step_month 1) #f)
    (if (== a 5) (step_year 1) #f)
    #undefined))

(define open_by_dn (lambda (dn)
    (define t (AL.ymd dn))
    (open_alm (get_item t 0) (get_item t 1) (get_item t 2))
    #undefined))

;; 按钮「按住」状态机：返回 #t 表示这一拍该按钮要触发一次（按下沿立即触发，
;; 按住满 360ms 后每 110ms 连发一次）。id = -1 表示没有按钮被按住。
;; 只对**两个活时间戳做差**：wasm32 的 (Diag.ms) 约 1.2e8，超过有符号整数载荷，
;; 加法会静默溢出（见 app/spectrogram.scm 的同类注释），故状态里存基准时刻而不是截止时刻。
(define armed_step (lambda (id)
    (define nowms (Diag.ms))
    (define fire #f)
    (if (not (== id hold_btn))
        (begin
          (set! hold_btn id)
          (if (>= id 0)
              (begin
                (set! hold_t0 nowms)
                (set! hold_last nowms)
                (set! hold_n 0)
                (set! fire #t))
              #f))
        (if (>= id 0)
            (if (== hold_n 0)
                (if (>= (- nowms hold_t0) 360)
                    (begin (set! hold_last nowms) (set! hold_n 1) (set! fire #t))
                    #f)
                (if (>= (- nowms hold_last) 110)
                    (begin (set! hold_last nowms) (set! hold_n (+ hold_n 1)) (set! fire #t))
                    #f))
            #f))
    fire))

(define tick_cal (lambda ()
    (define hi -1)
    (define dn #f)
    (if (not (== ev_kind 0)) (key_cal ev_code) #f)
    (BTN.track b_py)
    (BTN.track b_pm)
    (BTN.track b_td)
    (BTN.track b_nm)
    (BTN.track b_ny)
    (if (BTN.armed b_py) (set! hi 1) #f)
    (if (BTN.armed b_pm) (set! hi 2) #f)
    (if (BTN.armed b_td) (set! hi 3) #f)
    (if (BTN.armed b_nm) (set! hi 4) #f)
    (if (BTN.armed b_ny) (set! hi 5) #f)
    (if (not (== hi prev_armed))
        (begin (set! prev_armed hi) (set! cal_dirty 1))
        #f)
    (if (armed_step hi) (do_act hi) #f)
    ;; 日期格点按（落在按钮内的按下沿已被控件吞掉）
    (if (Touch.take_press)
        (begin
          (set! dn (hit_day (Touch.x) (Touch.y)))
          (if dn (open_by_dn dn) #f))
        #f)
    (if (== cal_dirty 1)
        (begin (draw_cal) (Gfx.refresh) (set! cal_dirty 0))
        #f)
    ;; 跨天：本月视图里今天变了要重画
    (if (not (== cal_last_day now_d))
        (begin
          (set! cal_last_day now_d)
          (if (and (== cal_y now_y) (== cal_m now_m)) (set! cal_dirty 1) #f))
        #f)
    #undefined))

;; 黄历页：拖动滚动 + 惯性
(define alm_drag (lambda ()
    (define py 0)
    (define dy 0)
    (if (and (not drg_on) (Touch.take_press))
        (begin
          (set! drg_on #t)
          (set! drg_moved #f)
          (set! drg_y0 (Touch.y))
          (set! drg_prev (Touch.y))
          (set! alm_vel 0.0))
        #f)
    (if drg_on
        (if (Touch.is_pressed)
            (begin
              (set! py (Touch.y))
              (set! dy (- py drg_prev))
              (if (not drg_moved)
                  (if (> (Math.abs (- py drg_y0)) 4) (set! drg_moved #t) #f)
                  #f)
              (if (and drg_moved (not (== dy 0)))
                  (begin
                    (move_to (+ alm_fpos (- 0 dy)))
                    (set! alm_vel (- 0 dy)))
                  #f)
              (set! drg_prev py))
            (set! drg_on #f))
        #f)
    (if (not drg_on)
        (if (> (Math.abs alm_vel) 1.0)
            (begin
              (set! alm_vel (* alm_vel 0.86))
              (move_to (+ alm_fpos alm_vel)))
            (set! alm_vel 0.0))
        #f)
    #undefined))

(define tick_alm (lambda ()
    (define a 0)
    (define hi -1)
    (if (not (== ev_kind 0)) (key_alm ev_code) #f)
    (BTN.track a_back)
    (BTN.track a_prev)
    (BTN.track a_today)
    (BTN.track a_next)
    (if (BTN.armed a_back) (set! hi 1) #f)
    (if (BTN.armed a_prev) (set! hi 2) #f)
    (if (BTN.armed a_today) (set! hi 3) #f)
    (if (BTN.armed a_next) (set! hi 4) #f)
    (if (not (== hi prev_armed))
        (begin (set! prev_armed hi) (set! alm_dirty 1))
        #f)
    (if (armed_step hi) (set! a hi) #f)
    (if (== a 1) (close_alm) #f)
    (if (== a 2) (alm_step (- 0 1)) #f)
    (if (== a 3) (alm_jump_today) #f)
    (if (== a 4) (alm_step 1) #f)
    (if (== view 1) (alm_drag) #f)
    (if (and (== view 1) (== alm_dirty 1))
        (begin (draw_alm) (Gfx.refresh) (set! alm_dirty 0))
        #f)
    #undefined))

(define tick (lambda ()
    (EX.check)
    (update_now)
    (poll_key)
    (if (== view 1) (tick_alm) (tick_cal))
    #undefined))

;; 宿主模态框关闭 / 恢复前台后的补绘
(define redraw (lambda ()
    (if (== view 1)
        (begin (draw_alm) (Gfx.refresh) (set! alm_dirty 0))
        (begin (draw_cal) (Gfx.refresh) (set! cal_dirty 0) (set! cal_last_day now_d)))
    #undefined))

;; ============================================================================
;; 按钮（通用控件 app/lib/button.scm）
;; ============================================================================

(define style_btn (lambda (b label x y w h)
    (BTN.set_label b label)
    (BTN.set_rect b x y w h)
    (BTN.set_font b F12)
    (BTN.set_text_color b 226 234 244)
    (BTN.set_bg b 32 40 52)
    (BTN.set_active_bg b 56 118 220)
    (BTN.set_radius b 6)
    (BTN.set_border b 46 56 70)
    #undefined))

(define b_py (BTN.new))
(define b_pm (BTN.new))
(define b_td (BTN.new))
(define b_nm (BTN.new))
(define b_ny (BTN.new))

(define a_back (BTN.new))
(define a_prev (BTN.new))
(define a_today (BTN.new))
(define a_next (BTN.new))

(define build_buttons (lambda ()
    (style_btn b_py "上年" 7 BTN_Y 58 BTN_H)
    (style_btn b_pm "上月" 69 BTN_Y 58 BTN_H)
    (style_btn b_td "今天" 131 BTN_Y 58 BTN_H)
    (style_btn b_nm "下月" 193 BTN_Y 58 BTN_H)
    (style_btn b_ny "下年" 255 BTN_Y 58 BTN_H)
    (style_btn a_back "返回" 4 3 40 20)
    (style_btn a_prev "前一日" 8 BTN_Y 96 BTN_H)
    (style_btn a_today "今日" 112 BTN_Y 96 BTN_H)
    (style_btn a_next "后一日" 216 BTN_Y 96 BTN_H)
    #undefined))

;; ============================================================================
;; 启动
;; ============================================================================

(define boot (lambda ()
    (define i 0)
    (set! SW (Gfx.width))
    (set! SH (Gfx.height))
    (set! gap12 (calc_gap F12))
    (set! gap16 (calc_gap F16))
    (lst_clr NUM_S)
    (set! i 1)
    (while (<= i 31)
        (push NUM_S (String.atom_to_string i))
        (set! i (+ i 1)))
    (build_buttons)
    (update_now)
    (set! cal_y now_y)
    (set! cal_m now_m)
    (clamp_ym)
    (set! sel_dn (AL.daynum now_y now_m now_d))
    (rebuild_texts)
    (set! cal_dirty 1)
    (set! cal_last_day now_d)
    (set! view 0)
    (set! alm_vel 0.0)
    (set! hold_btn -1)
    (EX.install_with redraw)
    (draw_cal)
    (Gfx.refresh)
    (set! cal_dirty 0)
    (set! timer_id (System.set_interval 16 tick))
    (display "[calendar] boot ") (display now_y) (display "-") (display now_m)
    (display "-") (display now_d) (newline)
    #undefined))

(boot)
