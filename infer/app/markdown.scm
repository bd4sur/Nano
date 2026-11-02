;; ============================================================================
;; Markdown 阅读器 —— 灵机应用程序（解析 / 排版 / 渲染全部用 Scheme 实现）
;;
;; 结构（三个模块，各管一段，可分别单测）：
;;   app/lib/md_parse.scm   文本 → 块树（块级规则表 + 行内规则表，扩展点见其文件头）
;;   app/lib/md_layout.scm  块树 → 显示行（行 = 高度/背景/段列表；段自带绝对 x 与宽度）
;;   app/lib/md_demo.scm    内置示例文档（行数组，覆盖当前支持的整个子集）
;;   本文件                 绘制 + 交互（滚动/翻页/点链接/退出）
;;
;; 支持的子集：# 1~3 级标题、**加粗**、`行内代码`、无序列表（**可嵌套**）、[链接](url)、
;;   围栏代码块、分隔线、**表格**（GFM：列对齐、单元格内行内样式、自动列宽、单元格内
;;   折行）、**引用块**（可嵌套、内部可放任意块）、段落（含 CJK 软换行）。有序列表留有扩展位。
;;
;; 标题的视觉区分（字模只有一套 12px，靠装饰而非字号）：H1 粗体 + 2px 通栏横线；
;; H2 粗体 + 左侧 3px 装饰竖线（#66ccff）；H3 粗体 + 文字下划线。
;;
;; 文档来源（都走 Fs 真文件链路）：
;;   · 业务资源根下的 doc/*.md 与根目录下的 *.md（桌面上即 /home/bd4sur/linglong/）；
;;   · **首次启动会把内置示例（md_demo.scm）写成 doc/Markdown 示例.md** 再按文件打开，
;;     所以"打开文件系统里的 Markdown"这条链路开箱即被走过一遍；文件已存在则不覆盖
;;     （可以像便签一样直接编辑它）；只读文件系统写失败时退回内存里的内置示例。
;;   多于一个文档时先显示列表（↑↓ 选择、Enter/点按打开，列表过长自动滚动），
;;   否则直接打开。
;;
;; 交互：
;;   · **底栏右侧「打开」按钮**（触屏）= 弹出可复用的文件选择器（app/lib/filepicker.scm），
;;     在**整个资源根**里挑任意 .md/.markdown 打开——不再受"doc/ 与根目录"扫描范围的限制；
;;   · 触屏拖动 = 滚动（手指移动多少滚多少）；轻点链接 = 弹窗显示目标地址；
;;   · 实体键盘 ↑↓ 滚动一行，←/→ 翻一页；
;;   · 右上角「退出」按钮 / 硬 A(ESC) —— 由 app/lib/exit.scm 统一接管。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/cond 多表达式；
;;   '() 是模块级静态槽（空表一律 (cdr '(0)) 现造）；push 返回 #undefined 不可赋值；
;;   循环里不用 (set! i 末值) 当 break，一律用 stop 标志。
;; ============================================================================

(native Gfx)
(native Key)
(native Touch)
(native Fs)
(native String)
(native Math)
(native System)
(native Ui)
(native App)
(import EX "app/lib/exit.scm")
(import FP "app/lib/filepicker.scm")
(import FSX "app/lib/fsx.scm")
(import MD "app/lib/md_parse.scm")
(import MDL "app/lib/md_layout.scm")
(import MDE "app/lib/md_demo.scm")

;; ---- 布局常量 ------------------------------------------------------------

(define SW 320)
(define SH 240)
(define TOP_H 18)     ;; 页眉高度
(define BOT_H 16)     ;; 页脚高度
(define PAD_X 6)      ;; 正文左右留白
(define CW 308)       ;; 正文宽 = SW - 2*PAD_X（启动时按实际屏宽重算）
(define ROW_H 20)     ;; 文档列表行高
(define LIST_Y 26)    ;; 文档列表首行 y
(define LIST_ROWS 9)  ;; 文档列表一次显示几项（更多时随选中项滚动）

;; 单篇文档的字节上限。为什么要有：本阅读器是"整篇解析 + 整篇排版"（换来滚动/命中
;; 测试全是算术，见 md_layout.scm 的设计说明），排版出的显示行按 ~1KB/行 占堆；
;; 而灵机 App 的解释器池是 12~24MB（ui_scheme_app.c），编译期已占去大部分。
;; 实测（vendor 核心，24MB 池）：31.8KB / 2260 显示行仍可排版，再大就会分配失败。
;; 宁可提前给出可读提示，也不要让用户看到一个"指令执行异常"的中断。
(define DOC_MAX 30720)   ;; 30KB（实测 31.8KB / 678 块仍可排版，留一点余量）

;; 内置示例首次启动时落盘到这里的文件名（标题即去掉扩展名的文件名）
(define SEED_TITLE "Markdown 示例")
(define DEMO_PATH "doc/Markdown 示例.md")

;; ---- 配色 ----------------------------------------------------------------

;; 正文段（按样式查色；style_weight = 描几次实现"伪粗体"——字模只有一套 12px，
;; 没有粗体字形，因此粗体靠同一行 1px 偏移重复描一次/两次得到视觉加粗）
(define cr 205) (define cg 214) (define cb 228) (define style_weight 1)

(define set_style_color
  (lambda (st)
    (if (== st MDL.ST_BODY)
        { (set! cr 205) (set! cg 214) (set! cb 228) (set! style_weight 1) } #f)
    (if (== st MDL.ST_BOLD)
        { (set! cr 250) (set! cg 252) (set! cb 255) (set! style_weight 2) } #f)
    (if (== st MDL.ST_CODE)
        { (set! cr 255) (set! cg 200) (set! cb 124) (set! style_weight 1) } #f)
    (if (== st MDL.ST_LINK)
        { (set! cr 112) (set! cg 182) (set! cb 255) (set! style_weight 1) } #f)
    ;; 注意：字模只有一套 12px alpha，「伪粗体」= 同一行 1px 偏移重复描。
    ;; 描 3 次对汉字会糊成一团（笔画粘连），故最多描 2 次；标题层级差异靠
    ;; 颜色亮度 + 段前留白 + 分隔线（H1 2px 通栏线、H2 1px 线、H3 无线且更暗）表达。
    (if (== st MDL.ST_H1)
        { (set! cr 255) (set! cg 255) (set! cb 255) (set! style_weight 2) } #f)
    (if (== st MDL.ST_H2)
        { (set! cr 228) (set! cg 238) (set! cb 252) (set! style_weight 2) } #f)
    (if (== st MDL.ST_H3)
        { (set! cr 176) (set! cg 197) (set! cb 228) (set! style_weight 1) } #f)
    (if (== st MDL.ST_CODEBLOCK)
        { (set! cr 172) (set! cg 214) (set! cb 180) (set! style_weight 1) } #f)
    (if (== st MDL.ST_BULLET)
        { (set! cr 130) (set! cg 170) (set! cb 225) (set! style_weight 1) } #f)
    (if (== st MDL.ST_TABLE_SEP)
        { (set! cr 58) (set! cg 70) (set! cb 92) (set! style_weight 1) } #f)
    (if (== st MDL.ST_QUOTE_BAR)
        { (set! cr 86) (set! cg 98) (set! cb 120) (set! style_weight 1) } #f)
    ;; 二级标题左侧装饰竖线的颜色：#66ccff
    (if (== st MDL.ST_H2BAR)
        { (set! cr 102) (set! cg 204) (set! cb 255) (set! style_weight 1) } #f)
    #t))

;; ---- 状态 ----------------------------------------------------------------

(define docs (cdr '(0)))      ;; 文档表：每项 (路径 标题)；路径为空串 = 内置示例
(define sel 0)                ;; 文档列表选中项
(define lines (cdr '(0)))     ;; 当前文档的显示行
(define content_h 0)          ;; 全部显示行的总高（像素）
(define scroll 0)             ;; 纵向滚动偏移（像素）
(define mode 1)               ;; 0 = 阅读，1 = 文档列表
(define cur_title "")         ;; 当前文档标题
(define ts 0)                 ;; 定时器编号

;; ---- 输入状态 ------------------------------------------------------------

(define prev_down 0)          ;; 上一拍触屏电平
(define drag_last 0)          ;; 上一拍触点 y
(define drag_total 0)         ;; 本次按压累计位移（判定"拖动 or 轻点"）
(define press_x 0)
(define press_y 0)
(define key 0)
(define hit_url "")
(define dyy 0)

;; ---- 小工具 --------------------------------------------------------------

(define new_list (lambda () (cdr '(0))))
(define mk2 (lambda (a b) (cons a (cons b (cdr '(0))))))

(define ends_with
  (lambda (s suf)
    (define ls (String.length s))
    (define lf (String.length suf))
    (if (< ls lf) #f
        (String.equals (String.slice s (- ls lf) ls) suf))))

;; ---- 几何：可视高度 / 最大滚动 / 收口 ------------------------------------

(define view_h
  (lambda ()
    (- (- SH TOP_H) BOT_H)))

(define max_scroll
  (lambda ()
    (define m (- content_h (view_h)))
    (if (< m 0) 0 m)))

(define clamp_scroll
  (lambda ()
    (if (< scroll 0) (set! scroll 0) #f)
    (if (> scroll (max_scroll)) (set! scroll (max_scroll)) #f)
    #t))

;; ---- 文档来源 ------------------------------------------------------------

;; 去掉 .md / .markdown 后缀作为标题
(define strip_ext
  (lambda (s)
    (if (ends_with s ".markdown") (String.slice s 0 (- (String.length s) 9))
        (if (ends_with s ".md") (String.slice s 0 (- (String.length s) 3)) s))))

;; 首次启动时把**内置示例**写成真正的 .md 文件（boot 调用），之后一律按文件打开。
;; 为什么这样设计：
;;   · 示例内容只有一处真源（md_demo.scm，在 git 里），不额外维护一份 .md 副本；
;;   · "阅读器打开文件系统中的 Markdown" 这条链路从第一次启动就被真正走过一遍
;;     （Fs.write → Fs.list → Fs.read → 解析 → 排版 → 渲染），而不是只走内存里的字面量；
;;   · 文件已存在时**不覆盖**——用户可以直接编辑它（便签式用法），升级 App 也不会被冲掉。
(define seed_demo_file
  (lambda ()
    (define ok #f)
    (if (Fs.exists DEMO_PATH)
        #f
        { (set! ok (Fs.write DEMO_PATH (String.concat (MD.join_lines MDE.lines) "\n")))
          (display "MDSEED ") (display DEMO_PATH)
          (display (if ok " written" " failed")) (newline) })
    #t))

;; 扫描一个目录下的 .md/.markdown（dir 为 "" 表示资源根；宿主保证按名字升序）
(define scan_dir
  (lambda (dir)
    (define nm (Fs.list dir))
    (define n 0)
    (define i 0)
    (define s 0)
    (define pfx "")
    (if (null? nm)
        #f
        { (set! n (length nm))
          (if (> (String.length dir) 0)
              (set! pfx (String.concat dir "/"))
              #f)
          (set! i 0)
          (while (< i n)
            { (set! s (get_item nm i))
              (if (or (ends_with s ".md") (ends_with s ".markdown"))
                  (push docs (mk2 (String.concat pfx s) (strip_ext s)))
                  #f)
              (set! i (+ i 1)) }) })
    #t))

;; 收集可打开的文档：doc/ 目录优先，其次资源根（"把 .md 丢到根目录也能打开"）。
;; 一个都没有时（只读文件系统、写示例失败）才退回到内存里的内置示例。
(define scan_docs
  (lambda ()
    (set! docs (new_list))
    (scan_dir "doc")
    (scan_dir ".")
    (if (== (length docs) 0)
        (push docs (mk2 "" SEED_TITLE))
        #f)
    #t))

;; 把提示文案做成一个段落块（复用同一条解析路径，免得手工构造 AST）
(define notice_blocks
  (lambda (line1 line2)
    (MD.parse_lines (cons line1 (cons "" (cons line2 (cdr '(0))))))))

;; ---- 打开文档：读文本 → 解析 → 排版 --------------------------------------

;; 按**路径**打开（列表选中与文件选择器共用同一条链路）
(define open_path
  (lambda (path title)
    (define text "")
    (define bs (new_list))
    (define fsize 0)
    (set! cur_title title)
    (set! fsize (Fs.size path))
    (if (null? fsize)
        (set! bs (notice_blocks "读取失败：文件不存在（或超过 Fs 的 128KB 上限）。"
                                "请把文档放到资源目录的 doc/ 下。"))
        (if (> fsize DOC_MAX)
            (set! bs (notice_blocks
                       (String.concat "文档过大（"
                                      (String.concat (String.atom_to_string fsize) " 字节）："))
                       "本阅读器一次排版整篇，请把文档控制在 30KB 以内（当前池装不下）。"))
            { (set! text (Fs.read path))
              (if (null? text)
                  (set! bs (notice_blocks "读取失败。" "文件可能已被删除或不可读。"))
                  ;; 注意：MD.parse 接受**整段文本**（内部自己 split_lines），
                  ;; 别再把行表传进去
                  (set! bs (MD.parse text))) }))
    (set! lines (MDL.layout bs CW))
    (set! content_h (MDL.total_height lines))
    (set! scroll 0)
    (set! mode 0)
    ;; 诊断行（宿主 stdout / 浏览器控制台；端到端测试据此断言解析与排版规模）
    (display "MDSTAT blocks=") (display (length bs))
    (display " lines=") (display (length lines))
    (display " height=") (display content_h)
    (display " title=") (display cur_title) (newline)
    #t))

;; 列表项打开：路径为空串 = 内存里的内置示例（只读文件系统兜底）
(define open_doc
  (lambda (idx)
    (define d (get_item docs idx))
    (define path (get_item d 0))
    (set! sel idx)
    (if (== (String.length path) 0)
        { (set! cur_title (get_item d 1))
          (set! lines (MDL.layout (MD.parse_lines MDE.lines) CW))
          (set! content_h (MDL.total_height lines))
          (set! scroll 0)
          (set! mode 0)
          (display "MDSTAT builtin title=") (display cur_title) (newline)
          #t }
        (open_path path (get_item d 1)))))

;; ---- 文件选择器（可复用组件 app/lib/filepicker.scm）---------------------
;; 为什么要有：列表只扫 doc/ 与资源根，用户把 .md 放在别的目录就打不开。选择器
;; 在整棵资源树里浏览，选中即按上面的同一条链路打开。
;; 活动期间**不调 (EX.check)**——返回请求归选择器消费（= 取消选择）。

(define OPEN_BTN_W 44)
(define OPEN_BTN_H 14)

(define open_btn_x (lambda () (- SW (+ OPEN_BTN_W 6))))
(define open_btn_y (lambda () (- SH 17)))

(define hit_open_btn
  (lambda (px py)
    (if (and (>= px (open_btn_x)) (< px (+ (open_btn_x) OPEN_BTN_W)))
        (if (and (>= py (open_btn_y)) (< py (+ (open_btn_y) OPEN_BTN_H))) 1 0)
        0)))

(define draw_open_btn
  (lambda ()
    (Gfx.fill_rect (open_btn_x) (open_btn_y) OPEN_BTN_W OPEN_BTN_H 52 58 76)
    (Gfx.draw_rect (open_btn_x) (open_btn_y) OPEN_BTN_W OPEN_BTN_H 110 118 140)
    (Gfx.draw_text "打开" (+ (open_btn_x) 7) (+ (open_btn_y) 1) 225 228 235)
    #t))

(define start_picker
  (lambda ()
    { (display "MDOPEN picker") (newline)
      (FP.start "打开 Markdown 文件" "md,markdown") }))

;; 选择器每拍：0 进行中 / 1 已选中 / 2 已取消
(define tick_picker
  (lambda ()
    (define r 0)
    (define picked "")
    (set! r (FP.step))
    (if (== r 1)
        { (set! picked (FP.result))
          (FP.reset)
          (display "MDOPEN picked ") (display picked) (newline)
          (open_path picked (FSX.strip_ext (FSX.leaf picked))) }
        #f)
    (if (== r 2)
        { (FP.reset)
          (draw_all) }
        #f)
    #t))

;; ---- 绘制：单个段 --------------------------------------------------------

;; 段 = (文本 样式 x 宽 url)；行内代码画药丸底、链接画下划线、项目符号画方块。
;; 注意：行内代码底 (46,42,34)、链接下划线 (96,160,235)、项目符号 (130,170,225)、
;; 代码块底 (26,32,29)、分隔线 (78,98,132)、表格竖线 (58,70,92) 这几个色值是与
;; 端到端测试的像素判据共享的
;; 契约（tools/e2e/markdown/test_app.py、tools/e2e/browser/regress_markdown.js），
;; 改色请同步改那两处。
(define draw_seg
  (lambda (sg y h)
    (define txt (get_item sg 0))
    (define st (get_item sg 1))
    (define x (+ PAD_X (get_item sg 2)))
    (define w (get_item sg 3))
    (define k 0)
    (set_style_color st)
    (if (== st MDL.ST_BULLET)
        (Gfx.fill_rect (+ x 4) (+ y 6) 3 3 cr cg cb)
        #f)
    (if (== st MDL.ST_TABLE_SEP)
        (Gfx.fill_rect x y 1 h cr cg cb)
        #f)
    (if (== st MDL.ST_QUOTE_BAR)
        (Gfx.fill_rect x y w h cr cg cb)
        #f)
    (if (== st MDL.ST_H2BAR)
        (Gfx.fill_rect x y w h cr cg cb)
        #f)
    (if (and (> (String.length txt) 0) (== st MDL.ST_CODE))
        (Gfx.fill_rect (- x 1) (+ y 1) (+ w 2) 13 46 42 34)
        #f)
    ;; 链接下划线：基线在 y+13，下划线再往下 3px（y+16），与字形留出呼吸感
    (if (and (> (String.length txt) 0) (== st MDL.ST_LINK))
        (Gfx.fill_rect x (+ y 16) w 1 96 160 235)
        #f)
    ;; 三级标题：文字下划线（比字形亮一档的暗色，避免和链接下划线混淆；同样下移 3px）
    (if (and (> (String.length txt) 0) (== st MDL.ST_H3))
        (Gfx.fill_rect x (+ y 15) w 1 96 128 168)
        #f)
    (if (> (String.length txt) 0)
        { (set! k 0)
          (while (< k style_weight)
            { (Gfx.draw_text txt (+ x k) (+ y 1) cr cg cb)
              (set! k (+ k 1)) }) }
        #f)
    #t))

;; ---- 绘制：单行（背景 + 各段）-------------------------------------------

(define draw_line
  (lambda (ln y)
    (define h (get_item ln 0))
    (define bg (get_item ln 1))
    (define segs (get_item ln 2))
    (define i 0)
    (if (== bg MDL.BG_CODE)
        (Gfx.fill_rect (- PAD_X 4) y (+ CW 8) h 26 32 29)
        #f)
    (if (== bg MDL.BG_TABLE)
        (Gfx.fill_rect (- PAD_X 4) y (+ CW 8) h 24 30 38)
        #f)
    (if (== bg MDL.BG_RULE)
        (Gfx.fill_rect PAD_X y CW h 78 98 132)
        #f)
    (while (< i (length segs))
      { (draw_seg (get_item segs i) y h)
        (set! i (+ i 1)) })
    #t))

;; ---- 绘制：正文 ----------------------------------------------------------
;; 页眉/页脚后画，天然遮住越界的半行（图形层没有裁剪矩形，靠绘制顺序裁剪）。

(define draw_content
  (lambda ()
    (define n (length lines))
    (define i 0)
    (define y 0)
    (define ln #f)
    (define bottom (- SH BOT_H))
    (define stop 0)
    (set! y (- TOP_H scroll))
    (while (and (< i n) (== stop 0))
      { (set! ln (get_item lines i))
        (if (> y bottom)
            (set! stop 1)                       ;; 后面全是屏外，收工
            (if (< (+ y (get_item ln 0)) TOP_H)
                #f                              ;; 整行在页眉之上：跳过
                (draw_line ln y)))
        (set! y (+ y (get_item ln 0)))
        (set! i (+ i 1)) })
    #t))

;; ---- 绘制：页眉 / 页脚 / 滚动条 -----------------------------------------

(define draw_header
  (lambda ()
    (define pct 0)
    (define mx (max_scroll))
    (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
    ;; 标题按可用宽度裁剪（右侧留给百分比与右上角的「退出」按钮，见 tools/e2e 的像素判据）
    (Gfx.draw_text (MDL.clip_text cur_title 232) PAD_X 3 236 242 250)
    (if (> mx 0)
        { (set! pct (Math.floor (/ (* scroll 100) mx)))
          (Gfx.draw_text (String.concat (String.atom_to_string pct) "%") 250 3 150 168 190) }
        (Gfx.draw_text "100%" 250 3 150 168 190))
    #t))

(define draw_footer
  (lambda ()
    (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
    (if (== mode 0)
        (Gfx.draw_text "拖动/↑↓ 滚动 · 点链接看地址" PAD_X (- SH 14) 130 140 155)
        (Gfx.draw_text "↑↓ 选择 · Enter/点按 打开" PAD_X (- SH 14) 130 140 155))
    (draw_open_btn)   ;; 底栏右侧：打开选择器（触屏唯一入口）
    #t))

(define draw_scrollbar
  (lambda ()
    (define vh 0)
    (define mx 0)
    (define bar 0)
    (define by 0)
    (set! mx (max_scroll))
    (if (> mx 0)
        { (set! vh (view_h))
          (set! bar (Math.floor (/ (* vh vh) content_h)))
          (if (< bar 10) (set! bar 10) #f)
          (set! by (+ TOP_H (Math.floor (/ (* scroll (- vh bar)) mx))))
          (Gfx.fill_rect (- SW 3) TOP_H 3 vh 34 40 52)
          (Gfx.fill_rect (- SW 3) by 3 bar 96 116 148) }
        #f)
    #t))

;; ---- 绘制：文档列表 ------------------------------------------------------

;; 文档多于一屏时随选中项滚动：窗口首项 = max(0, sel - LIST_ROWS + 1)
(define list_first
  (lambda ()
    (if (>= sel LIST_ROWS) (- (+ sel 1) LIST_ROWS) 0)))

(define draw_list
  (lambda ()
    (define n (length docs))
    (define first (list_first))
    (define last 0)
    (define i 0)
    (define y 0)
    (define d #f)
    (set! last (+ first LIST_ROWS))
    (if (> last n) (set! last n) #f)
    (set! i first)
    (while (< i last)
      { (set! y (+ LIST_Y (* (- i first) ROW_H)))
        (set! d (get_item docs i))
        (if (== i sel)
            { (Gfx.fill_rect 2 (- y 3) (- SW 4) (- ROW_H 2) 40 50 68)
              (Gfx.fill_rect 2 (- y 3) 3 (- ROW_H 2) 120 180 250) }
            #f)
        (Gfx.draw_text (MDL.clip_text (get_item d 1) 240) (+ PAD_X 8) y 226 232 244)
        (if (== (String.length (get_item d 0)) 0)
            (Gfx.draw_text "内置" (- SW 40) y 150 168 190)
            (Gfx.draw_text ".md" (- SW 34) y 120 140 165))
        (set! i (+ i 1)) })
    #t))

;; ---- 绘制总入口 ----------------------------------------------------------

(define draw_all
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH 14 17 22)
    (if (== mode 0)
        { (draw_content)
          (draw_scrollbar)
          (draw_header) }
        { (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
          (Gfx.draw_text "Markdown 文档" PAD_X 3 236 242 250)
          (Gfx.draw_text (String.concat (String.atom_to_string (length docs)) " 篇")
                         250 3 150 168 190)
          (draw_list) })
    (draw_footer)
    (EX.draw_button)
    (Gfx.refresh)
    #t))

;; ---- 交互：链接命中测试 --------------------------------------------------
;; 返回命中的链接目标（空串表示没点中）。行/段都是排版期定死的绝对坐标，
;; 因此命中测试就是"滚动偏移 + 行高累加"的算术。
(define hit_link
  (lambda (px py)
    (define n (length lines))
    (define i 0)
    (define y 0)
    (define ln #f)
    (define segs #f)
    (define k 0)
    (define sg #f)
    (define sx 0)
    (define r "")
    (define stop 0)
    (set! y (- TOP_H scroll))
    (while (and (< i n) (== stop 0))
      { (set! ln (get_item lines i))
        (if (and (>= py y) (< py (+ y (get_item ln 0))))
            { (set! segs (get_item ln 2))
              (set! k 0)
              (while (< k (length segs))
                { (set! sg (get_item segs k))
                  (if (== (get_item sg 1) MDL.ST_LINK)
                      { (set! sx (+ PAD_X (get_item sg 2)))
                        (if (and (>= px sx) (< px (+ sx (get_item sg 3))))
                            { (set! r (get_item sg 4))
                              (set! k (length segs)) }
                            #f) }
                      #f)
                  (set! k (+ k 1)) })
              (set! stop 1) }
            (set! y (+ y (get_item ln 0))))
        (set! i (+ i 1)) })
    r))

;; 阅读区轻点：命中链接 → 弹窗显示地址（模态期间解释器进程挂起，界面照常刷新）
(define tap_reader
  (lambda (px py)
    (if (< py TOP_H) #f
        (if (>= py (- SH BOT_H)) #f
            { (set! hit_url (hit_link px py))
              (if (> (String.length hit_url) 0)
                  { ;; 诊断行：端到端测试据此断言"点链接确实命中并取了目标地址"
                    (display "MDLINK ") (display hit_url) (newline)
                    (Ui.message_titled "链接目标" hit_url) }
                  #f) }))))

;; 列表轻点：按 y 反算条目下标
(define tap_list
  (lambda (py)
    (define idx 0)
    (if (>= py LIST_Y)
        { (set! idx (+ (list_first) (Math.floor (/ (- py LIST_Y) ROW_H))))
          (if (< idx (length docs))
              (open_doc idx)
              #f) }
        #f)))

;; ---- 交互：触屏（拖动滚动 / 轻点）---------------------------------------

(define handle_touch
  (lambda ()
    (if (Touch.is_pressed)
        { (if (== prev_down 0)
              { (set! prev_down 1)
                (set! drag_total 0)
                (set! drag_last (Touch.y))
                (set! press_x (Touch.x))
                (set! press_y (Touch.y)) }
              #f)
          (set! dyy (- (Touch.y) drag_last))
          (set! drag_last (Touch.y))
          (set! drag_total (+ drag_total dyy))
          (if (== mode 0)
              { (set! scroll (- scroll dyy))
                (clamp_scroll) }
              #f) }
        { (if (== prev_down 1)
              { (set! prev_down 0)
                (if (< (Math.abs drag_total) 8)
                    (if (== (hit_open_btn press_x press_y) 1)
                        (start_picker)
                        (if (== mode 0)
                            (tap_reader press_x press_y)
                            (tap_list press_y)))
                    #f) }
              #f) })
    #t))

;; ---- 交互：键盘 ----------------------------------------------------------

(define apply_key
  (lambda (k)
    (if (== mode 1)
        { (if (== k 1) { (set! sel (- sel 1)) (if (< sel 0) (set! sel 0) #f) } #f)
          (if (== k 2) { (set! sel (+ sel 1)) (if (>= sel (length docs)) (set! sel (- (length docs) 1)) #f) } #f)
          (if (== k 13) (open_doc sel) #f) }
        { (if (== k 1) { (set! scroll (- scroll 15)) (clamp_scroll) } #f)
          (if (== k 2) { (set! scroll (+ scroll 15)) (clamp_scroll) } #f)
          (if (== k 3) { (set! scroll (- scroll (view_h))) (clamp_scroll) } #f)
          (if (== k 4) { (set! scroll (+ scroll (view_h))) (clamp_scroll) } #f) })
    #t))

(define handle_keys
  (lambda ()
    (set! key (Key.take_press))
    (while (not (== key 0))
      { (apply_key key)
        (set! key (Key.take_press)) })
    #t))

;; ---- 主循环 --------------------------------------------------------------

(define on_tick
  (lambda ()
    (if (FP.active)
        (tick_picker)          ;; 选择器活动期间：输入/绘制全归它，EX 不抢返回请求
        { (EX.check)
          (handle_keys)
          (handle_touch)
          (draw_all) })))

;; ---- 启动 ----------------------------------------------------------------

(define boot
  (lambda ()
    { (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (set! CW (- SW (* 2 PAD_X)))
      (seed_demo_file)      ;; 首次启动：把内置示例写成 doc/ 下的真 .md 文件
      (scan_docs)
      ;; 只有一个候选时直接进阅读页，不为一次点按多绕一圈
      (if (<= (length docs) 1)
          (open_doc 0)
          (set! mode 1))
      (draw_all)
      (set! ts (System.set_interval 60 on_tick)) }))

(EX.install_with draw_all)
(boot)