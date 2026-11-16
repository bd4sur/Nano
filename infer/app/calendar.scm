;; ============================================================================
;; calendar.scm —— 日历 + 黄历（灵机应用程序版）
;;
;; 与 C 版 infer/ui_calendar.c + infer/ui_almanac.c 的 GUI/交互逐项对齐：
;;   · 日历页（每周从周一开始，固定 6 行，未排满留空）：页眉居中「YYYY年M月」、
;;     页脚四枚软键提示「上月/今天/下月/黄历」、周一~周日表头（周末红）、
;;     今天用蓝色色块 + 白字高亮；静态画面不重复推帧，切月/跨天才重绘；
;;   · 左右键切上月/下月（跨年）、上下键切上一年/下一年、0 键回当前年月、
;;     回车查看今天黄历、A/ESC 返回（灵机 App 里由 app/lib/exit.scm 接管为
;;     「退出确认框」，与其它 App 一致）；
;;   · 点按日期数字 → 打开该日黄历模态框（内容打开时算一次、固定不变）；
;;     模态框内任意触屏按下即关闭；打开 300ms 后到达的按键下降沿亦关闭
;;     （吞掉同一手势的残留键）；关闭后 150ms 内的残余按键直接吞掉；
;;   · 黄历模态框版式：4 张卡片（八字+吉凶徽标 / 星期·星座·宿·建除·十二神 /
;;     子午流注当令器官+宜+忌+说明 / 宜·忌 / 九宫飞星·胎神·彭祖百忌·时辰吉凶+
;;     吉神方位右列），配色与 #141516/#1a1a1a 参考 HTML 一致，12px 抗锯齿字；
;;     超宽文本按 C 版 al_fit_w 规则从尾部截断补「…」。
;;
;; 输入模型的差异（灵机 App 没有 GUI 的 4x4 宫格软按键）：
;;   · C 版页脚四枚提示对应 16 宫格底行的 4 个软键；灵机 App 运行态宿主已抑制
;;     宫格映射（见 am_native_Key.h），因此这里把**页脚四段热区**直接映射到那四个
;;     动作（上月/今天/下月/黄历），触屏-only 设备因此仍可完整操作；
;;   · C 版认 GUI 的 key_edge（-1 短按松手沿 / -2 长按连发）。灵机 (native Key)
;;     给的是按下/松开事件与电平，这里按同一套阈值重建：短按在**松手**时生效，
;;     按住满 360ms 后按拍连发（等价 key_repeat）。
;;
;; 时间来源：C 版用 libc localtime；灵机 (native Time) 只给 UTC（日数+日内秒），
;; 故按 UTC+8 换算本地民用时（与 app/ephemeris.scm 的 TZ 处理一致）。
;;
;; 退出入口：为与 C 版日历逐像素一致，**不绘制**右上角「退出」按钮；退出仍走
;; 硬 A/ESC（app/lib/exit.scm 的全局确认框）以及 EX 在右上角登记的点按热区
;;（不可见，但不影响日历任何可见控件）。
;;
;; 解释器约束（manual.md）：算术恒二参数、无一元负号、无 let/list/map/append；
;; 同一 lambda 区域内不重名 define；块内不 define。
;; ============================================================================

(native Gfx)
(native Key)
(native Touch)
(native Time)
(native String)
(native Math)
(native System)
(native App)
(native Diag)

(import AL "app/lib/almanac.scm")
(import EX "app/lib/exit.scm")

;; ---- 字体与配色（对齐 ui_app.c 的 ui_font = 16px 抗锯齿 + ui_calendar.c 深色）----
(define F12 1)
(define F16 2)

(define C_BG_R 6) (define C_BG_G 6) (define C_BG_B 6)          ;; 日历正文底
(define C_TX_R 255) (define C_TX_G 255) (define C_TX_B 255)    ;; 正文
(define C_WE_R 255) (define C_WE_G 80) (define C_WE_B 80)      ;; 周末红
(define C_TD_R 66) (define C_TD_G 133) (define C_TD_B 244)     ;; 今天色块
(define C_LN_R 66) (define C_LN_G 66) (define C_LN_B 66)       ;; 分隔线
(define C_HD_R 15) (define C_HD_G 16) (define C_HD_B 17)       ;; 页眉/页脚底（深色）
(define C_HT_R 188) (define C_HT_G 188) (define C_HT_B 188)    ;; 页眉/页脚字

;; 黄历模态框配色（ui_almanac.c 的 #141516/#1a1a1a 参考 HTML）
(define A_BG_R 20) (define A_BG_G 21) (define A_BG_B 22)
(define A_CD_R 26) (define A_CD_G 26) (define A_CD_B 26)
(define A_BAZI_R 255) (define A_BAZI_G 170) (define A_BAZI_B 102)
(define A_CY_R 102) (define A_CY_G 204) (define A_CY_B 255)
(define A_YEL_R 255) (define A_YEL_G 242) (define A_YEL_B 0)
(define A_TX_R 226) (define A_TX_G 229) (define A_TX_B 239)
(define A_YI_R 153) (define A_YI_G 235) (define A_YI_B 153)
(define A_JI_R 255) (define A_JI_G 172) (define A_JI_B 199)
(define A_DIR_R 170) (define A_DIR_G 255) (define A_DIR_B 136)
(define A_BAD_R 51) (define A_BAD_G 51) (define A_BAD_B 51)
(define A_GY_R 128) (define A_GY_G 130) (define A_GY_B 137)
(define A_VAL_BUF 96)

;; ---- 状态 ------------------------------------------------------------------
(define cal_y 2026)
(define cal_m 1)
(define cal_dirty 1)
(define cal_last_day -1)
(define alm_active 0)
(define alm_dirty 1)
(define alm_R #f)
(define alm_err 0)
(define alm_open_ms 0)
(define alm_close_ms 0)

;; 当前本地民用时（UTC+8）
(define now_y 2026)
(define now_m 1)
(define now_d 1)
(define now_h 0)
(define now_min 0)

;; 屏幕几何
(define SW 320)
(define SH 240)

;; 按键长按重建状态
(define k_prev 0)
(define k_t0 0)
(define k_rep 0)

;; ============================================================================
;; 文本工具（度量/绘制/截断）
;; ============================================================================

(define slen (lambda (s) (String.length s)))

;; 与 gfx_font_measure_text 等价：逐码点累加字模步进
(define measure (lambda (font s)
    (define n (slen s))
    (define i 0)
    (define w 0)
    (define m #f)
    (while (< i n)
        (set! m (Gfx.glyph_metrics font (String.charCodeAt i s)))
        (set! w (+ w (get_item m 0)))
        (set! i (+ i 1)))
    w))

;; C 的 gfx_font_draw_text 把 y 当"行框顶"；灵机 (Gfx.draw_text) 把 y 当"墨迹顶"，
;; 差值 = 基线 - 字模上高（与 am_native_Gfx.c 的 native_text_top_gap 同式）。
(define text_gap (lambda (font)
    (define fi (Gfx.font_info font))
    (define base (get_item fi 3))
    (define m (Gfx.glyph_metrics font 20013))   ;; '中'
    (define asc (get_item m 1))
    (define g (- base (Math.round asc)))
    (if (< g 0) 0 g)))

(define draw_t (lambda (font s x ytop r g b)
    (Gfx.set_font font)
    (Gfx.draw_text s x (+ ytop (text_gap font)) r g b)))

(define draw_tc (lambda (font s cx cy r g b)
    (Gfx.draw_text_centered font s cx cy r g b 1)))

;; 超宽从尾部截断并补「…」（al_fit_w）
(define fit_w (lambda (font s maxw)
    (define cur s)
    (define dot "…")
    (define wdot 0)
    (if (== (slen s) 0)
        s
        (if (<= (measure font s) maxw)
            s
            (begin
              (set! wdot (measure font dot))
              (while (and (> (slen cur) 1)
                          (> (+ (measure font cur) wdot) maxw))
                  (set! cur (String.slice cur 0 (- (slen cur) 1))))
              (if (<= (+ (measure font cur) wdot) maxw)
                  (String.concat cur dot)
                  cur))))))

(define seq (lambda (a b) (String.equals a b)))

;; 清单拼成一行（空格分隔）并裁剪（al_list_line）
(define list_line (lambda (L maxw)
    (define n (length L))
    (define i 0)
    (define out "")
    (while (< i n)
        (if (> i 0) (set! out (String.concat out " ")) #f)
        (set! out (String.concat out (get_item L i)))
        (set! i (+ i 1)))
    (fit_w F12 out maxw)))

;; 「彩色标签 + 默认色值」（al_pair）
(define al_pair (lambda (x y label lr lg lb val)
    (draw_t F12 label x y lr lg lb)
    (draw_t F12 val (+ x (measure F12 label)) y A_TX_R A_TX_G A_TX_B)))

;; ============================================================================
;; 时间（UTC+8 本地民用时）
;; ============================================================================

(define TZ_SEC 28800)

(define update_now (lambda ()
    (define ud (Time.utc_days))
    (define us (+ (Time.utc_sod) TZ_SEC))
    (define d ud)
    (define s us)
    (define z 0) (define era 0) (define doe 0) (define t1 0) (define t2 0)
    (define yoe 0) (define yy 0) (define doy 0) (define mp 0) (define mo 0)
    (if (>= s 86400) (begin (set! s (- s 86400)) (set! d (+ d 1))) #f)
    ;; 整数除法一律用 AL.idiv（纯整数长除法）：32 位宿主的浮点尾数只有约 19 位，
    ;; 730485 这类大整数经 Math.floor 的浮点结果会丢 1（实测 wasm32 上 5*146097 被
    ;; 舍成 730484，日期整整差一天）。AL.idiv 在 27 位整数载荷内精确。
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
;; 日历页版式与绘制（ui_calendar.c 逐行对齐）
;; ============================================================================

(define LH16 18)
(define HEAD_H 27)
(define FOOT_H 19)
(define TOP 27)
(define BOT 221)
(define WK_CY 38)
(define GRID_TOP 51)
(define ROW_H 28)

(define WEEK_NAMES '("一" "二" "三" "四" "五" "六" "日"))

(define is_leap_year (lambda (y)
    (or (and (== (Math.floor (/ y 4)) (/ y 4)) (not (== (Math.floor (/ y 100)) (/ y 100))))
        (and (== (Math.floor (/ y 400)) (/ y 400)) #t))))

(define days_in_month (lambda (y m)
    (define days '(31 28 31 30 31 30 31 31 30 31 30 31))
    (if (and (== m 2) (is_leap_year y))
        29
        (get_item days (- m 1)))))

;; 某年某月 1 日是星期几（周一=0..周日=6；Sakamoto 算法）
(define first_weekday (lambda (y m)
    (define t '(0 3 2 5 0 3 5 1 4 6 2 4))
    (define yy (- y (if (< m 3) 1 0)))
    (define q (+ (+ yy (AL.idiv yy 4))
                 (+ (- 0 (AL.idiv yy 100))
                    (+ (AL.idiv yy 400)
                       (+ (get_item t (- m 1)) 1)))))
    (define dow (AL.imod q 7))
    (AL.imod (+ dow 6) 7)))

(define draw_header (lambda (title)
    (Gfx.fill_rect 0 0 SW HEAD_H C_HD_R C_HD_G C_HD_B)
    (draw_tc F16 title (Math.floor (/ SW 2)) (Math.floor (/ HEAD_H 2)) C_HT_R C_HT_G C_HT_B)))

(define draw_footer (lambda ()
    (define cell (Math.floor (/ SW 4)))
    (define cy (+ (- SH FOOT_H) (Math.floor (/ FOOT_H 2))))
    (define labels '("上月" "今天" "下月" "黄历"))
    (define i 0)
    (Gfx.fill_rect 0 (- SH FOOT_H) SW FOOT_H C_HD_R C_HD_G C_HD_B)
    (while (< i 4)
        (draw_tc F16 (get_item labels i)
                 (+ (* cell i) (Math.floor (/ cell 2))) cy C_HT_R C_HT_G C_HT_B)
        (set! i (+ i 1)))
    #undefined))

(define draw_cal (lambda ()
    (define title "")
    (define i 0)
    (define j 0)
    (define col 0)
    (define row 0)
    (define idx 0)
    (define cx 0)
    (define cy 0)
    (define fw 0)
    (define dim 0)
    (define is_cur 0)
    (define is_today 0)
    (define day 1)
    (define dstr "")
    (Gfx.set_font F16)
    (set! title (String.concat (String.atom_to_string cal_y)
                  (String.concat "年" (String.concat (String.atom_to_string cal_m) "月"))))
    (draw_header title)
    (draw_footer)
    (Gfx.fill_rect 0 TOP SW (- BOT TOP) C_BG_R C_BG_G C_BG_B)
    ;; 星期表头
    (set! i 0)
    (while (< i 7)
        (set! cx (Math.floor (/ (+ (* i SW) (Math.floor (/ SW 2))) 7)))
        (if (>= i 5)
            (draw_tc F16 (get_item WEEK_NAMES i) cx WK_CY C_WE_R C_WE_G C_WE_B)
            (draw_tc F16 (get_item WEEK_NAMES i) cx WK_CY C_TX_R C_TX_G C_TX_B))
        (set! i (+ i 1)))
    (Gfx.draw_line 0 (- GRID_TOP 3) (- SW 1) (- GRID_TOP 3) C_LN_R C_LN_G C_LN_B)
    ;; 日期网格
    (set! fw (first_weekday cal_y cal_m))
    (set! dim (days_in_month cal_y cal_m))
    (set! is_cur (if (and (== cal_y now_y) (== cal_m now_m)) 1 0))
    (set! day 1)
    (while (<= day dim)
        (set! idx (- (+ fw day) 1))
        (set! col (AL.imod idx 7))
        (set! row (AL.idiv idx 7))
        (if (>= row 6)
            (set! day (+ dim 1))
            (begin
              (set! cx (Math.floor (/ (+ (* col SW) (Math.floor (/ SW 2))) 7)))
              (set! cy (+ (+ GRID_TOP (* row ROW_H)) (Math.floor (/ ROW_H 2))))
              (set! is_today (if (and (== is_cur 1) (== day now_d)) 1 0))
              (set! dstr (String.atom_to_string day))
              (if (== is_today 1)
                  (begin
                    (Gfx.fill_rect (- cx 12) (+ (- cy (Math.floor (/ ROW_H 2))) 1)
                                   25 (- ROW_H 2) C_TD_R C_TD_G C_TD_B)
                    (draw_tc F16 dstr cx (+ cy 1) 255 255 255))
                  (if (>= col 5)
                      (draw_tc F16 dstr cx cy C_WE_R C_WE_G C_WE_B)
                      (draw_tc F16 dstr cx cy C_TX_R C_TX_G C_TX_B)))
              (set! day (+ day 1)))))
    #undefined))

;; ---- 黄历卡1：八字 + 吉凶徽标 / 星期·星座·宿·建除·十二神 ----
(define alm_card1 (lambda (R LH IX IR ty1 ty2)
    (define badge (AL.overall R))
    (define bw (measure F12 badge))
    (define brw (+ bw 16))
    (define brx (- IR brw))
    (define bkind 0)
    (define segs (cons (AL.week R)
                  (cons (AL.zodiac R)
                  (cons (AL.star28 R)
                  (cons (AL.officer R)
                  (cons (AL.god R) (cdr '(0))))))))
    (define sep " | ")
    (define sepw (measure F12 sep))
    (define pen IX)
    (define i 0)
    (define stop 0)
    (define seg "")
    (define sw 0)
    (define tmp "")
    (if (seq badge "凶") (set! bkind 1) (if (seq badge "平") (set! bkind 2) (set! bkind 0)))
    (Gfx.fill_rect brx (- ty1 2) brw (+ LH 4) A_BAD_R A_BAD_G A_BAD_B)
    (if (== bkind 1)
        (draw_t F12 badge (+ brx 8) ty1 A_JI_R A_JI_G A_JI_B)
        (if (== bkind 2)
            (draw_t F12 badge (+ brx 8) ty1 A_YEL_R A_YEL_G A_YEL_B)
            (draw_t F12 badge (+ brx 8) ty1 A_YI_R A_YI_G A_YI_B)))
    (draw_t F12 (AL.bazi R) IX ty1 A_BAZI_R A_BAZI_G A_BAZI_B)
    (while (and (== stop 0) (< i 5))
        (set! seg (get_item segs i))
        (if (> i 0)
            (if (<= (+ pen sepw) IR)
                (begin (draw_t F12 sep pen ty2 A_GY_R A_GY_G A_GY_B)
                       (set! pen (+ pen sepw)))
                (set! stop 1))
            #f)
        (if (== stop 0)
            (begin
              (set! sw (measure F12 seg))
              (if (> (+ pen sw) IR)
                  (begin
                    (set! tmp (fit_w F12 seg (- IR pen)))
                    (draw_t F12 tmp pen ty2 A_CY_R A_CY_G A_CY_B)
                    (set! stop 1))
                  (begin
                    (draw_t F12 seg pen ty2 A_CY_R A_CY_G A_CY_B)
                    (set! pen (+ pen sw))
                    (set! i (+ i 1)))))
            #f))
    #undefined))

;; ---- 黄历卡2：子午流注（当令器官 / 宜 / 忌 + 说明）----
(define alm_card2 (lambda (R LH IX CW y)
    (define inner (- CW 12))
    (define r1y (+ y 4))
    (define r2y (+ (+ r1y LH) 3))
    (define org (AL.meridian R))
    (define myi (AL.meridian_yi R))
    (define mji (AL.meridian_ji R))
    (define note (AL.meridian_note R))
    (define w_lbo (measure F12 "当令器官"))
    (define w_sp (measure F12 " "))
    (define w_yi (measure F12 "宜"))
    (define w_ji (measure F12 "忌"))
    (define w_co (measure F12 "："))
    (define g1 (+ (+ w_lbo w_sp) (measure F12 org)))
    (define budget (- (- inner g1) 8))
    (define g3 0)
    (define g2 0)
    (define gap 0)
    (define x1 IX)
    (define x2 0)
    (define x3 0)
    (define nw 0)
    (set! mji (fit_w F12 mji (- (Math.floor (/ budget 2)) (+ w_ji w_co))))
    (set! g3 (+ (+ w_ji w_co) (measure F12 mji)))
    (set! myi (fit_w F12 myi (- (- budget g3) (+ w_yi w_co))))
    (set! g2 (+ (+ w_yi w_co) (measure F12 myi)))
    (set! gap (Math.floor (/ (- (- (- inner g1) g2) g3) 2)))
    (if (< gap 4) (set! gap 4) #f)
    (set! x2 (+ x1 (+ g1 gap)))
    (set! x3 (+ x2 (+ g2 gap)))
    (draw_t F12 "当令器官" x1 r1y A_TX_R A_TX_G A_TX_B)
    (draw_t F12 org (+ x1 (+ w_lbo w_sp)) r1y A_YEL_R A_YEL_G A_YEL_B)
    (draw_t F12 "宜" x2 r1y A_YI_R A_YI_G A_YI_B)
    (draw_t F12 "：" (+ x2 w_yi) r1y A_TX_R A_TX_G A_TX_B)
    (draw_t F12 myi (+ x2 (+ w_yi w_co)) r1y A_TX_R A_TX_G A_TX_B)
    (draw_t F12 "忌" x3 r1y A_JI_R A_JI_G A_JI_B)
    (draw_t F12 "：" (+ x3 w_ji) r1y A_TX_R A_TX_G A_TX_B)
    (draw_t F12 mji (+ x3 (+ w_ji w_co)) r1y A_TX_R A_TX_G A_TX_B)
    (set! note (fit_w F12 note inner))
    (set! nw (measure F12 note))
    (draw_t F12 note (+ IX (Math.floor (/ (- inner nw) 2))) r2y A_GY_R A_GY_G A_GY_B)
    #undefined))

;; ---- 黄历卡3：宜 / 忌 ----
(define alm_card3 (lambda (R LH CX CW y)
    (define inner (- CW 12))
    (define y1 (+ y 4))
    (define val "")
    (define w_yi (measure F12 "宜："))
    (define w_ji (measure F12 "忌："))
    (set! val (list_line (AL.good R) (- inner w_yi)))
    (al_pair (+ CX 6) y1 "宜：" A_YI_R A_YI_G A_YI_B val)
    (set! val (list_line (AL.bad R) (- inner w_ji)))
    (al_pair (+ CX 6) (+ y1 LH) "忌：" A_JI_R A_JI_G A_JI_B val)
    #undefined))

;; ---- 黄历卡4：九宫飞星/胎神/彭祖百忌/时辰吉凶 + 吉神方位 ----
(define alm_card4 (lambda (R LH CX CW IX IR y)
    (define inner (- CW 12))
    (define ly (+ y 4))
    (define dirs (AL.dirs R))
    (define nd (length dirs))
    (define dline (cons "" (cons "" (cons "" (cdr '(0))))))
    (define dw (cons 0 (cons 0 (cons 0 (cdr '(0))))))
    (define dcnt (cons 0 (cons 0 (cons 0 (cdr '(0))))))
    (define i 0)
    (define row 0)
    (define it "")
    (define cur "")
    (define dmax 0)
    (define left_w 0)
    (define fly "(无)")
    (define fet "")
    (define peng "")
    (define hl "")
    (define any 0)
    (define n 0)
    (define labels '("九宫飞星：" "胎神：" "彭祖百忌：" "时辰吉凶："))
    (define values #f)
    (define branches '("子" "丑" "寅" "卯" "辰" "巳" "午" "未" "申" "酉" "戌" "亥"))
    (define lw 0)
    (define vv "")
    (define hours (AL.hourluck R))
    ;; 右列：2+2+1 三行
    (set! i 0)
    (while (and (< i nd) (< i 5))
        (set! row (if (< i 2) 0 (if (< i 4) 1 2)))
        (set! it (get_item dirs i))
        (set! cur (get_item dline row))
        (if (> (slen cur) 0) (set! cur (String.concat cur " ")) #f)
        (set! cur (String.concat cur it))
        (set_item! dline row cur)
        (set! i (+ i 1)))
    (set! i 0)
    (while (< i 3)
        (set_item! dw i (measure F12 (get_item dline i)))
        (set! i (+ i 1)))
    (set! dmax (get_item dw 0))
    (if (> (get_item dw 1) dmax) (set! dmax (get_item dw 1)) #f)
    (if (> (get_item dw 2) dmax) (set! dmax (get_item dw 2)) #f)
    (set! left_w (- (- inner dmax) 8))
    ;; 左列四行
    (set! fly (AL.fly R))
    (set! fet (AL.fetal R))
    (set! peng (AL.peng R))
    (set! n (length hours))
    (if (> n 12) (set! n 12) #f)
    (set! i 0)
    (set! any 0)
    (set! hl "")
    (while (< i n)
        (if (seq (get_item hours i) "吉")
            (begin
              (set! hl (String.concat hl (get_item branches i)))
              (set! any 1))
            #f)
        (set! i (+ i 1)))
    (if (== any 1)
        (set! hl (String.concat "吉时 " hl))
        (set! hl "今日无吉时"))
    (set! values (cons fly (cons fet (cons peng (cons hl (cdr '(0)))))))
    (set! i 0)
    (while (< i 4)
        (set! lw (measure F12 (get_item labels i)))
        (set! vv (fit_w F12 (get_item values i) (- left_w lw)))
        (al_pair (+ CX 6) (+ ly (* i LH)) (get_item labels i)
                 A_TX_R A_TX_G A_TX_B vv)
        (set! i (+ i 1)))
    ;; 右列（右对齐，3 行垂直居中于 4 行空间）
    (set! i 0)
    (while (< i 3)
        (if (> (slen (get_item dline i)) 0)
            (draw_t F12 (get_item dline i)
                    (- IR (get_item dw i))
                    (+ (+ ly 7) (* i LH))
                    A_DIR_R A_DIR_G A_DIR_B)
            #f)
        (set! i (+ i 1)))
    #undefined))

;; ---- 黄历模态框整体绘制 ----
(define draw_alm (lambda ()
    (define LH (get_item (Gfx.font_info F12) 2))
    (define CX (+ 10 3))
    (define CW (- (- SW 20) 6))
    (define GAP 3)
    (define ch0 (+ 12 (+ (* 2 LH) 3)))
    (define ch1 (+ 8 (+ (* 2 LH) 3)))
    (define ch2 (+ 8 (* 2 LH)))
    (define ch3 (+ 8 (* 4 LH)))
    (define y0 (+ 10 3))
    (define y1 (+ (+ y0 ch0) GAP))
    (define y2 (+ (+ y1 ch1) GAP))
    (define y3 (+ (+ y2 ch2) GAP))
    (define IX (+ CX 6))
    (define IR (- (+ CX CW) 6))
    (define msg "")
    (define mw 0)
    (Gfx.set_font F12)
    (Gfx.fill_rect 0 0 SW SH A_BG_R A_BG_G A_BG_B)
    (if (== alm_err 1)
        (begin
          (Gfx.fill_rect 6 90 (- SW 12) 60 A_CD_R A_CD_G A_CD_B)
          (set! msg "该日期超出黄历计算范围")
          (set! mw (measure F12 msg))
          (draw_tc F12 msg (Math.floor (/ SW 2)) 120 A_TX_R A_TX_G A_TX_B))
        (if (== alm_R #f)
            #f
            (begin
              (Gfx.fill_rect CX y0 CW ch0 A_CD_R A_CD_G A_CD_B)
              (Gfx.fill_rect CX y1 CW ch1 A_CD_R A_CD_G A_CD_B)
              (Gfx.fill_rect CX y2 CW ch2 A_CD_R A_CD_G A_CD_B)
              (Gfx.fill_rect CX y3 CW ch3 A_CD_R A_CD_G A_CD_B)
              (alm_card1 alm_R LH IX IR (+ y0 6) (+ (+ y0 6) (+ LH 3)))
              (alm_card2 alm_R LH IX CW y1)
              (alm_card3 alm_R LH CX CW y2)
              (alm_card4 alm_R LH CX CW IX IR y3))))
    #undefined))

;; ============================================================================
;; 交互
;; ============================================================================

;; 触屏命中日期数字（ui_calendar_touch_hit）
(define touch_day (lambda (tx ty)
    (define col 0)
    (define row 0)
    (define idx 0)
    (define fw 0)
    (define dim 0)
    (define day 0)
    (if (or (< tx 0) (>= tx SW) (< ty GRID_TOP) (>= ty BOT))
        0
        (begin
          (set! col (AL.idiv (* tx 7) SW))
          (set! row (AL.idiv (- ty GRID_TOP) ROW_H))
          (if (or (< row 0) (> row 5))
              0
              (begin
                (set! idx (+ (* row 7) col))
                (set! fw (first_weekday cal_y cal_m))
                (set! dim (days_in_month cal_y cal_m))
                (set! day (+ (- idx fw) 1))
                (if (or (< day 1) (> day dim)) 0 day)))))))

;; 打开黄历模态框（ui_calendar_open_almanac）
(define open_alm (lambda (y m d)
    (set! alm_R (AL.calc y m d now_h now_min))
    (if (== alm_R #f) (set! alm_err 1) (set! alm_err 0))
    (set! alm_active 1)
    (set! alm_dirty 1)
    (set! alm_open_ms (Diag.ms))
    (set! cal_dirty 1)
    #undefined))

(define close_alm (lambda ()
    (set! alm_active 0)
    (set! alm_R #f)
    (set! alm_err 0)
    (set! alm_close_ms (Diag.ms))
    (set! cal_dirty 1)
    (set! alm_dirty 1)
    #undefined))

;; 日历按键动作（ui_calendar_event_handler 的 switch）
(define do_key (lambda (code)
    (cond
      ((== code 3)                              ;; 上月
       { (set! cal_m (- cal_m 1))
         (if (< cal_m 1)
             (begin (set! cal_m 12) (set! cal_y (- cal_y 1)))
             #f)
         (if (< cal_y 1900) (set! cal_y 1900) #f)
         (set! cal_dirty 1) })
      ((== code 4)                              ;; 下月
       { (set! cal_m (+ cal_m 1))
         (if (> cal_m 12)
             (begin (set! cal_m 1) (set! cal_y (+ cal_y 1)))
             #f)
         (if (> cal_y 2100) (set! cal_y 2100) #f)
         (set! cal_dirty 1) })
      ((== code 1)                              ;; 上一年
       (if (> cal_y 1900)
           (begin (set! cal_y (- cal_y 1)) (set! cal_dirty 1))
           #f))
      ((== code 2)                              ;; 下一年
       (if (< cal_y 2100)
           (begin (set! cal_y (+ cal_y 1)) (set! cal_dirty 1))
           #f))
      ((== code 48)                             ;; 回到当前年月
       (if (or (not (== cal_y now_y)) (not (== cal_m now_m)))
           (begin
             (set! cal_y now_y)
             (set! cal_m now_m)
             (if (< cal_y 1900) (set! cal_y 1900) #f)
             (if (> cal_y 2100) (set! cal_y 2100) #f)
             (set! cal_dirty 1))
           #f))
      ((== code 13)                             ;; 回车：查看今天的黄历
       (open_alm now_y now_m now_d))
      (else #f))
    #undefined))

;; 页脚四段热区 → 四个软键动作（灵机 App 无 4x4 宫格，故直接映射）
(define footer_action (lambda (tx)
    (define cell (Math.floor (/ SW 4)))
    (define i (Math.floor (/ tx cell)))
    (if (== i 0) (do_key 3)
        (if (== i 1) (do_key 48)
            (if (== i 2) (do_key 4)
                (do_key 13))))
    #undefined))

;; 触屏按下沿（模态框之外）
(define handle_touch (lambda ()
    (define pressed #f)
    (define tx 0)
    (define ty 0)
    (define day 0)
    (set! pressed (Touch.take_press))
    (if pressed
        (begin
          (set! tx (Touch.x))
          (set! ty (Touch.y))
          (set! day (touch_day tx ty))
          (if (not (== day 0))
              (open_alm cal_y cal_m day)
              (if (>= ty (- SH FOOT_H))
                  (footer_action tx)
                  #f)))
        #f)
    #undefined))

;; 按键事件重建（等价 GUI 的 key_edge：短按松手 -1 / 长按 -2 连发）
(define ev_code 0)
(define ev_kind 0)

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

;; ============================================================================
;; 每拍（等价 C 的 event_handler + render_frame）
;; ============================================================================

(define tick (lambda ()
    (EX.check)
    (update_now)
    (poll_key)
    (if (== alm_active 1)
        (begin
          ;; 打开 300ms 后到达的按键下降沿 → 关闭（吞掉打开手势的残留键）
          (if (not (== ev_kind 0))
              (if (>= (- (Diag.ms) alm_open_ms) 300) (close_alm) #f)
              #f)
          ;; 任意触屏按下沿 → 关闭
          (if (== alm_active 1)
              (if (Touch.take_press) (close_alm) #f)
              #f))
        (begin
          (if (not (== ev_kind 0))
              (if (< (- (Diag.ms) alm_close_ms) 150)
                  #f
                  (do_key ev_code))
              #f)
          (if (== alm_active 0) (handle_touch) #f)))
    ;; 绘制（静态画面不重复推帧）
    (if (== alm_active 1)
        (if (== alm_dirty 1)
            (begin (draw_alm) (Gfx.refresh) (set! alm_dirty 0))
            #f)
        (if (or (== cal_dirty 1)
                (and (== cal_y now_y) (== cal_m now_m) (not (== cal_last_day now_d))))
            (begin (draw_cal) (Gfx.refresh) (set! cal_dirty 0) (set! cal_last_day now_d))
            #f))
    #undefined))

;; 宿主模态框关闭 / 恢复前台后的补绘（退出确认框关闭时不恢复帧缓冲）
(define redraw (lambda ()
    (if (== alm_active 1)
        (begin (draw_alm) (Gfx.refresh) (set! alm_dirty 0))
        (begin (draw_cal) (Gfx.refresh) (set! cal_dirty 0) (set! cal_last_day now_d)))
    #undefined))

;; ============================================================================
;; 启动
;; ============================================================================

(define boot (lambda ()
    (define fi #f)
    (define ts 0)
    (set! SW (Gfx.width))
    (set! SH (Gfx.height))
    (set! fi (Gfx.font_info F16))
    (set! LH16 (get_item fi 2))
    (set! HEAD_H (Math.floor (/ (* LH16 3) 2)))
    (set! FOOT_H (+ LH16 1))
    (set! TOP HEAD_H)
    (set! BOT (- SH FOOT_H))
    (set! WK_CY (+ (+ TOP 2) (Math.floor (/ LH16 2))))
    (set! GRID_TOP (+ (+ TOP 2) (+ LH16 4)))
    (set! ROW_H (Math.floor (/ (- BOT GRID_TOP) 6)))
    (update_now)
    (set! cal_y now_y)
    (set! cal_m now_m)
    (if (< cal_y 1900) (set! cal_y 1900) #f)
    (if (> cal_y 2100) (set! cal_y 2100) #f)
    (if (< cal_m 1) (set! cal_m 1) #f)
    (if (> cal_m 12) (set! cal_m 12) #f)
    (set! cal_dirty 1)
    (set! cal_last_day (- 0 1))
    (set! alm_active 0)
    (set! alm_R #f)
    (set! alm_err 0)
    (set! alm_close_ms (- 0 100000))
    (set! k_prev 0)
    (set! k_rep 0)
    (EX.install_with redraw)
    (draw_cal)
    (Gfx.refresh)
    (set! cal_dirty 0)
    (set! cal_last_day now_d)
    (set! ts (System.set_interval 16 tick))
    #undefined))

(boot)
