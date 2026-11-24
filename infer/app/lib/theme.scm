;; ============================================================================
;; theme.scm —— 通用调色板（灵机 App 共用的一套「键 → (r g b)」主题表）
;;
;; 为什么要有：Markdown 阅读器的配色原来散落在几十处 (fill_rect 30 38 52) 里，
;; 想加一套亮色就得逐处改、还容易漏。现在把**全部**颜色收进本模块：
;;   · 一套主题 = (id 名称 (键 r g b)...)，键是符号（符号 eq? 恒等，查表不必比字符串）；
;;   · 用色方在**切换主题时**一次性把要用的颜色取进自己的模块变量（(TH.get 键)），
;;     绘制路径只读变量——本模块的查表是线性的，放进每帧的绘制循环会拖慢低端设备；
;;   · **加一套新配色 = 在本文件的主题表里追加一段**，用色方一行都不用改；
;;     设置面板按 (TH.count)/(TH.name) 自动列出全部主题（见 app/markdown.scm）。
;;
;; 接口：
;;   (TH.count)       —— 主题数量
;;   (TH.name id)     —— 主题显示名（越界返回 ""）
;;   (TH.select id)   —— 选主题；越界/非数字 → 不生效并返回 #f，成功返回 #t
;;   (TH.current)     —— 当前主题 id
;;   (TH.get key)     —— 当前主题下 key 的 (r g b)（每次新造 3 元列表，可安全保存）
;;   (TH.has key)     —— 当前主题是否配了这个键
;;
;; 约定：一套主题**应当**给全同一组键。漏配不算错误（(TH.get) 兜底品红），
;;   但会画出一眼可见的品红，端到端测试的像素判据很容易抓到。
;;
;; 键的分组（用色方 app/markdown.scm 也按这个顺序取色）：
;;   页面/栏（page_bg header_* footer_*）→ 正文样式（body…imath）→
;;   块背景与装饰（bg_* pill_bg *_ul img_dash）→ 滚动条 → 按钮 →
;;   面板（panel_bg toc_*）→ 菜单组件（menu_* lamp_*）→ 启动提示（busy_*）
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/assoc/append；
;;   while 循环体不是独立词法区域（循环变量一律在函数体顶层 define）；
;;   0 是真值，布尔一律 #t/#f；空表一律 (cdr '(0)) 现造。
;; ============================================================================

;; ---- 主题表 --------------------------------------------------------------
;;
;; 深色（id 0，默认）：Markdown 阅读器 2026-11 之前的原始配色，**逐字保留**——
;;   端到端测试的像素判据就是按这些色值写的
;;   （infer/tools/e2e/browser/regress_markdown.js 的 probe）。改这里的色值
;;   必须同步改那处判据，否则回归全红。
(define dark_colors
  '((page_bg        14  17  22)
    (header_bg      30  38  52)
    (header_text   236 242 250)
    (header_dim    150 168 190)
    (footer_bg      24  28  38)
    (footer_text   130 140 155)
    ;; 正文样式（ST_BODY…ST_IMATH；见 md_layout.scm 的样式常量）
    (body          205 214 228)
    (bold          250 252 255)
    (code          255 200 124)
    (link          112 182 255)
    (h1            255 255 255)
    (h2            228 238 252)
    (h3            176 197 228)
    (codeblock     172 214 180)
    (bullet        130 170 225)
    (table_sep      58  70  92)
    (quote_bar      86  98 120)
    (hr             78  98 132)
    (h2bar         102 204 255)
    (image_text    130 146 172)
    (math          234 238 248)
    (imath         234 238 248)
    ;; 块背景与行内装饰
    (bg_code        26  32  29)
    (bg_table       24  30  38)
    (bg_rule        78  98 132)
    (pill_bg        46  42  34)
    (link_ul        96 160 235)
    (h3_ul          96 128 168)
    (img_dash       92 104 128)
    ;; 滚动条（阅读页）
    (sb_track       34  40  52)
    (sb_thumb       96 116 148)
    ;; 底栏按钮（打开/目录/设置）
    (btn_bg         52  58  76)
    (btn_border    110 118 140)
    (btn_text      225 228 235)
    ;; 右上角按钮（宿主「退出/返回」与面板的「返回」共用一套色）
    (exit_bg        60  62  74)
    (exit_border   120 124 140)
    (exit_text     225 228 235)
    (exit_active   116 172 226)
    ;; 面板（目录/设置）
    (panel_bg       16  19  26)
    (toc_l1        238 244 252)
    (toc_l2        206 218 236)
    (toc_l3        168 182 204)
    (toc_s1        255 255 255)
    (toc_s2        236 244 254)
    (toc_s3        210 224 244)
    ;; 通用菜单组件（app/lib/menu_list.scm）
    (menu_hl        40  50  68)
    (menu_bar       40  47  62)
    (menu_thumb    130 150 184)
    (lamp_on       102 204 255)
    (lamp_off      110 118 140)
    ;; 启动提示
    (busy_slot      40  48  62)
    (busy_bar       96 160 235)))

;; 明亮（id 1）：纸张感的暖白底 + 深色墨迹。取舍：
;;   · 正文不用纯黑（46,50,58）——12px 抗锯齿字模在纯黑上显得过重、也刺眼；
;;   · 标题比正文更深（h1 近黑）仍能靠"更深 + 段前留白 + 分隔线"区分层级；
;;   · 行内代码用暖褐（146,84,18）配浅琥珀底（pill_bg），在暖白底上不像深色主题的
;;     亮橙那样跳，又明显区别于正文；
;;   · 表格列线/引用竖线/分隔线一律降成浅灰（198/186/176 系），否则在亮底上会抢字；
;;   · 强调色（链接/目录灯/H2 装饰线）统一到 20 号左右的蓝（28,130,208），
;;     保证亮底上的对比度而不至于刺眼。
(define light_colors
  '((page_bg       250 249 245)
    (header_bg     229 226 219)
    (header_text    30  34  42)
    (header_dim    104 108 116)
    (footer_bg     236 234 228)
    (footer_text   110 114 122)
    (body           46  50  58)
    (bold           12  16  22)
    (code          146  84  18)
    (link           20 100 190)
    (h1             10  14  20)
    (h2             44  52  66)
    (h3             92 100 116)
    (codeblock      24  90  54)
    (bullet         70 112 176)
    (table_sep     198 202 210)
    (quote_bar     186 192 204)
    (hr            176 182 192)
    (h2bar          28 130 208)
    (image_text    120 126 136)
    (math           30  34  44)
    (imath          30  34  44)
    (bg_code       238 241 234)
    (bg_table      236 239 244)
    (bg_rule       200 206 216)
    (pill_bg       243 235 216)
    (link_ul        70 140 210)
    (h3_ul         168 178 194)
    (img_dash      172 178 190)
    (sb_track      228 226 220)
    (sb_thumb      148 152 162)
    (btn_bg        232 230 225)
    (btn_border    176 180 190)
    (btn_text       34  38  46)
    (exit_bg       226 230 238)
    (exit_border   150 158 172)
    (exit_text      30  36  46)
    (exit_active    60 140 215)
    (panel_bg      246 245 241)
    (toc_l1         24  28  36)
    (toc_l2         66  74  88)
    (toc_l3        112 120 134)
    (toc_s1          8  14  26)
    (toc_s2         24  38  58)
    (toc_s3         48  64  88)
    (menu_hl       208 220 238)
    (menu_bar      224 222 216)
    (menu_thumb    146 150 160)
    (lamp_on        28 130 208)
    (lamp_off      150 154 164)
    (busy_slot     214 212 206)
    (busy_bar       60 140 215)))

;; 主题表：每项 = (id 名称 颜色条目表)。**追加新主题就在这里加一段**——
;; id 必须是连续的小整数（设置面板按 0..count-1 列出并按下标选中）。
(define themes
  (cons (cons 0 (cons "深色" (cons dark_colors (cdr '(0)))))
        (cons (cons 1 (cons "明亮" (cons light_colors (cdr '(0)))))
              (cdr '(0)))))

;; ---- 状态 ---------------------------------------------------------------

(define cur 0)                ;; 当前主题 id
(define entries dark_colors)  ;; 当前主题的颜色条目表

;; ---- 查表 ---------------------------------------------------------------

;; 按 id 找主题项（返回整项，找不到返回 #f）
(define find_theme
  (lambda (id)
    (define n (length themes))
    (define i 0)
    (define e #f)
    (define r #f)
    (while (and (< i n) (not r))
      { (set! e (get_item themes i))
        (if (== (get_item e 0) id) (set! r e) #f)
        (set! i (+ i 1)) })
    r))

;; 在当前主题的条目表里按键找条目（返回整条，找不到返回 #f）
(define find_entry
  (lambda (key)
    (define n (length entries))
    (define i 0)
    (define e #f)
    (define r #f)
    (while (and (< i n) (not r))
      { (set! e (get_item entries i))
        (if (eq? (get_item e 0) key) (set! r e) #f)
        (set! i (+ i 1)) })
    r))

;; ---- 对外接口 -----------------------------------------------------------

(define count (lambda () (length themes)))
(define current (lambda () cur))

(define name
  (lambda (id)
    (define e (find_theme id))
    (if e (get_item e 1) "")))

(define select
  (lambda (id)
    (define e #f)
    (if (number? id) (set! e (find_theme id)) #f)
    (if e
        { (set! cur (get_item e 0))
          (set! entries (get_item e 2))
          #t }
        #f)))

(define has
  (lambda (key)
    (if (find_entry key) #t #f)))

;; 取色。找不到的键返回品红 (255 0 255)：**故意刺眼**——新加的颜色键在哪个主题里
;; 漏配了，屏幕上一眼就能看见（而不是悄悄画成黑色、让人以为是布局问题）。
(define get
  (lambda (key)
    (define e (find_entry key))
    (if e
        (cons (get_item e 1) (cons (get_item e 2) (cons (get_item e 3) (cdr '(0)))))
        (cons 255 (cons 0 (cons 255 (cdr '(0))))))))
