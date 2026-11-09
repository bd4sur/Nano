;; ============================================================================
;; Markdown 阅读器 —— 灵机应用程序（解析 / 排版 / 渲染全部用 Scheme 实现）
;;
;; 结构（四个模块，各管一段，可分别单测）：
;;   app/lib/md_parse.scm   文本 → 块树（块级规则表 + 行内规则表，扩展点见其文件头）
;;   app/lib/md_layout.scm  块树 → 显示行（行 = 高度/背景/段列表；段自带绝对 x 与宽度）
;;   app/lib/md_demo.scm    内置示例文档（行数组，覆盖当前支持的整个子集）
;;   app/lib/md_stream.scm  分段流式读取（大文件底座：字节流 → 段；见其文件头）
;;   app/lib/md_view.scm    分段视图（段缓存 / 锚点滚动 / 预取 / 高度估算 / 目录）
;;   本文件                 绘制 + 交互（滚动/翻页/点链接/退出）
;;
;; 大文件（≤1MB）怎么读：**不再整篇解析+整篇排版**，而是
;;   ① 打开：只切出并排版**第一段**（4KB 源码，首屏 ~90ms），其余段在后台按拍建索引；
;;   ② 滚动：视口附近的若干段被排版并缓存（上限 CACHE_LINES 行），段内滚动零重排，
;;      跨界时才布局相邻段（预取保证提前一两屏就做好了）；
;;   ③ 段高度只有排版过才精确：未访问段按"码点数 × 实测校准系数"估算，而滚动位置由
;;      **锚点 (段号, 段内 dy)** 决定，所以估算被修正时画面不跳；页眉百分比在索引
;;      未覆盖全文时显示 "≈"。
;;   实测（WASM，池 32MB / heap 16MB）：1MB 文档开屏 ~90ms、堆 ~5.8MB、翻页 0.4ms/屏；
;;   而改造前整篇物化在 128KB 就接近 16MB 上限、1MB 需要 >60MB 与 ~15s。
;;
;; 支持的子集：# 1~3 级标题、**加粗**、`行内代码`、无序列表（**可嵌套**）、[链接](url)、
;;   围栏代码块、分隔线、**表格**（GFM：列对齐、单元格内行内样式、自动列宽、单元格内
;;   折行）、**引用块**（可嵌套、内部可放任意块）、**块级数学公式**（$$...$$，见下）、
;;   段落（含 CJK 软换行）。有序列表留有扩展位。
;;
;; 块级数学公式：$$ 单行 $$ 或 $$ + 若干行 + $$。排版/绘制复用数学公式 App 的引擎
;;   （app/lib/tex_*.scm）：解析端只产出 (math 源串)，排版端按注入的度量回调定行高与
;;   居中位置（超宽则按比例缩小字号**重排**到与屏幕等宽），绘制端画缓存的 op 流。
;;   **行内公式 $...$ 不处理**，按普通文本原样显示。
;;
;; 标题的视觉区分（字模只有一套 12px，靠装饰而非字号）：H1 粗体 + 2px 通栏横线；
;; H2 粗体 + 左侧 3px 装饰竖线（#66ccff）；H3 粗体 + 文字下划线。
;;
;; 文档来源（都走 Fs 真文件链路）：
;;   · 业务资源根下的 doc/*.md 与根目录下的 *.md（桌面上即 /home/bd4sur/linglong/）；
;;   · 内置示例（md_demo.scm 是**唯一真源**）：构建期由 `make md-demo`
;;     （或 build_wasm_gui.sh 自动调用）导出成 doc/Markdown 示例.md；
;;     若该文件不存在（例如直接拷 App 到设备上、只读文件系统），App 首次启动会自己
;;     落盘一份再按文件打开——所以"打开文件系统里的 Markdown"这条链路开箱即被走过一遍。
;;     文件已存在时**不覆盖**（可以当便签直接编辑它）；写失败则退回内存里的内置示例。
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
(import MDV "app/lib/md_view.scm")
(import ML "app/lib/menu_list.scm")
;; 数学公式排版引擎（与 app/mathjax.scm 共用同一套：LaTeX → 语法树 → 盒子 → op 流）
(import P "app/lib/tex_parse.scm")
(import L "app/lib/tex_layout.scm")
(import D "app/lib/tex_draw.scm")
(import B "app/lib/tex_base.scm")

;; ---- 布局常量 ------------------------------------------------------------

(define SW 320)
(define SH 240)
(define TOP_H 18)     ;; 页眉高度
(define BOT_H 16)     ;; 页脚高度
(define PAD_X 6)      ;; 正文左右留白
(define CW 308)       ;; 正文宽 = SW - 2*PAD_X（启动时按实际屏宽重算）
(define ROW_H 20)     ;; 文档列表行高（随字模调整）
(define LIST_Y 26)    ;; 文档列表首行 y
(define LIST_ROWS 9)  ;; 文档列表一次显示几项（更多时随选中项滚动）

;; 单篇文档的字节上限：1MB。
;; 为什么能放开到 1MB：本阅读器已改为**分段流式**——文档按 4KB 源码切成段，只有
;; 视口附近的若干段会被解析/排版并缓存在堆上（见 app/lib/md_stream.scm 与
;; app/lib/md_view.scm）。实测（WASM，32MB 池 / 16MB heap）：每 KB 源码要 ~40KB 堆
;; （块树）+ ~26KB（显示行，折合 ~1.1KB/显示行）+ ~15ms；整篇物化在 128KB 时就已
;; 接近 16MB 上限，1MB 需要 >60MB —— 所以必须分段，而不是放大上限。
(define DOC_MAX 1048576)  ;; 1MB

;; 分段参数（实测：布局 4KB 源码 ≈ 60ms ≈ 一个 GUI 节拍，段内滚动零重排）
(define SEG_TARGET 4096)      ;; 目标段大小（字节）；切点落在围栏外的空行处
(define CACHE_LINES 4000)     ;; 段缓存显示行上限（≈4MB 堆，见 md_view.scm）

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
    ;; 表格的上下框线 / 表头下框线（只在表格宽度内画）
    (if (== st MDL.ST_HR)
        { (set! cr 78) (set! cg 98) (set! cb 132) (set! style_weight 1) } #f)
    ;; 二级标题左侧装饰竖线的颜色：#66ccff
    (if (== st MDL.ST_H2BAR)
        { (set! cr 102) (set! cg 204) (set! cb 255) (set! style_weight 1) } #f)
    ;; 行内图片的降级占位文本（一期不支持行内插图）
    (if (== st MDL.ST_IMAGE_TEXT)
        { (set! cr 130) (set! cg 146) (set! cb 172) (set! style_weight 1) } #f)
    ;; 块级数学公式：与 mathjax 同档的亮色（4bpp 字模在小字号上抗锯齿偏细，亮色观感更好）
    (if (== st MDL.ST_MATH)
        { (set! cr 234) (set! cg 238) (set! cb 248) (set! style_weight 1) } #f)
    #t))

;; ---- 状态 ----------------------------------------------------------------

(define docs (cdr '(0)))      ;; 文档表：每项 (路径 标题)；路径为空串 = 内置示例
(define sel 0)                ;; 文档列表选中项
(define MDV_ok 0)              ;; 1 = 流式文档（md_view），0 = 提示文案（notice）
(define notice_lines (cdr '(0)))  ;; 提示文案的显示行（整篇排版，内容极小）
(define notice_h 0)           ;; 提示文案总高
(define toc_scanned 0)        ;; 目录已覆盖的段数（= 索引推进到的段数，诊断/进度）
(define toc_gen_seen -1)      ;; 已反映到 toc_items 的目录世代号（增量刷新用）
;; 面板关闭后的若干帧里把宿主「退出」热区清零（>0 生效）：
;;   面板的「返回」键与阅读页的「退出」键**同位置**，而宿主的退出热区判定发生在触摸**边沿**
;;   （早于 App 的下一拍）。面板关闭的那一帧阅读页会重新登记热区，于是同一次点按会被宿主
;;   再判成"返回请求"，进而在阅读页弹出全局退出确认框（WASM 实测踩到）。静音几帧即可，
;;   阅读页的退出键本身仍由 EX 的电平兜底路径响应，不丢功能。
(define exitbtn_mute 0)
(define content_h 0)          ;; 全文档总高（像素；未索引部分为估算，仅用于滚动条/百分比）
(define scroll 0)             ;; 纵向滚动偏移（像素，**由锚点派生**的只读快照）
(define mode 1)               ;; 0 = 阅读，1 = 文档列表
(define cur_title "")         ;; 当前文档标题
(define cur_path "")          ;; 当前文档路径（"" = 内置示例；换字模后据此重新排版）
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
(define mk3 (lambda (a b c) (cons a (cons b (cons c (cdr '(0)))))))
(define mk7 (lambda (a b c d e f g)
  (cons a (cons b (cons c (cons d (cons e (cons f (cons g (cdr '(0)))))))))))

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

;; 滚动位置由 md_view 的锚点 (段号, 段内 dy) 决定；这里只做"读快照"与"刷新全局量"。
;; 统一的滚动入口：流式文档交给 md_view（跨界自动按需布局），
;; 提示文案按像素直接夹取（内容极小，无分段）。
(define scroll_by_px
  (lambda (px)
    (if (== MDV_ok 1)
        { (MDV.scroll_by px (view_h))
          (sync_scroll) }
        { (set! scroll (+ scroll px))
          (if (< scroll 0) (set! scroll 0) #f)
          (if (> scroll (max_scroll)) (set! scroll (max_scroll)) #f)
          (sync_scroll) })
    #t))

(define sync_scroll
  (lambda ()
    (set! scroll (if (== MDV_ok 1) (MDV.top_y) scroll))
    (set! content_h (if (== MDV_ok 1) (MDV.total_height) notice_h))
    #t))

(define max_scroll
  (lambda ()
    (define m (- (if (== MDV_ok 1) (MDV.total_height) notice_h) (view_h)))
    (if (< m 0) 0 m)))

(define clamp_scroll
  (lambda ()
    (sync_scroll)
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

;; ---- 图片支持（方案 A：宿主解码；方案 D：取不到时画虚线框兜底）----------
;;
;; 分工（详见 doc/Markdown插图渲染可行性调研.md）：
;;   · 解析端只产出 (image 路径 alt)，不碰文件；
;;   · 排版端通过下面的 (doc_image_size) 度量回调拿原始尺寸，按正文宽等比缩放，
;;     产出一行"高度 = 图片高"的显示行（段样式 ST_IMAGE，段里带路径与 alt）；
;;   · 绘制端只调 (Gfx.draw_image ...) 让宿主解码+缩放+落屏——像素永不进解释器堆；
;;     返回 #f（文件缺失/格式不支持）时画虚线框 + alt 文本兜底。
;;
;; 路径规则：Markdown 习惯是"以文档所在目录为基准"。文档路径形如 "doc/x.md" 时
;; 基准是 "doc/"，根目录文档基准是空串；Gfx 的路径规则与 Fs 一致（相对资源根）。
(define doc_dir "")     ;; 当前文档所在目录（含结尾 '/'；根目录为空串）

(define set_doc_dir
  (lambda (path)
    (define n (String.length path))
    (define i 0)
    (define cut -1)
    (while (< i n)
      { (if (== (String.charCodeAt i path) 47) (set! cut i) #f)   ;; '/'
        (set! i (+ i 1)) })
    (if (< cut 0) (set! doc_dir "") (set! doc_dir (String.slice path 0 (+ cut 1))))))

;; Markdown 里的图片引用 → 资源根相对路径。
;; 查找顺序：① 以 '/' 开头 → 宿主绝对路径，原样交给 Gfx；② 相对**文档所在目录**；
;; ③ 回退到**资源根**（`icon/xxx.png` 这类公共资源常放在根下，而文档在 doc/ 里，
;;    又不能写 "../icon/..."——Fs/Gfx 的路径规则明确拒绝 ".."）。
(define image_path
  (lambda (p)
    (define r "")
    (if (== (String.length p) 0) p
        (if (== (String.charCodeAt 0 p) 47) p
            { (set! r (String.concat doc_dir p))
              (if (Fs.exists r) r
                  (if (Fs.exists p) p r)) }))))

;; 排版期的度量回调：(路径) → (宽 高) | #f
;; 注意 (Gfx.image_size ...) 失败时返回 #null，而 #null 在本解释器里是**真值**，
;; 这里统一归一成 #f，排版端只需判真假/类型即可（见 md_layout 的 lay_image）。
(define doc_image_size
  (lambda (p)
    (define r (Gfx.image_size (image_path p)))
    (if (null? r) #f r)))

;; ---- 块级数学公式（复用 app/lib/tex_*：LaTeX → 语法树 → 盒子 → op 流）------
;;
;; 与数学公式 App（app/mathjax.scm）同一套引擎，只换输入源：
;;   · 解析：md_parse 把 $$...$$ 解析成 (math 源串) 块（本文件不碰 Markdown 语法）；
;;   · 度量：md_layout 的 lay_math 调下面的 (doc_math_measure) 拿 (宽 上高 下深)，
;;     据此定行高并水平居中；**超宽就等比缩小 em 重排**（不是位图缩放，字形仍落在
;;     字模的清晰尺度上），保证长公式与正文等宽；
;;   · 绘制：取缓存的 op 流直接画，**不重排**——滚动/翻页时每帧只做"换缓冲 + 一遍
;;     顺序绘制"，否则每帧重排公式会明显掉帧。
;;
;; 缓存：最近 MATH_CACHE_MAX 条公式的 (源串 em 宽 上高 下深 op流 操作数)。
;; tex_base 的操作缓冲是模块级单例，绘制前用 (B.set_ops 缓存流 操作数) 换回去；
;; 缓存的流此后只读（排版器每次 layout 都先 reset，不会原地改旧流）。
(define MATH_CACHE_MAX 12)
(define MATH_MIN_EM 0.20)     ;; 超长公式的最小缩放（再小已不可读，超出部分靠裁剪）
(define math_cache (cdr '(0)))  ;; (源串 em 宽 上高 下深 op流 操作数)，最近加入的在前

(define math_cache_find
  (lambda (src)
    (define n (length math_cache))
    (define i 0)
    (define e #f)
    (define r #f)
    (while (and (< i n) (not r))
      { (set! e (get_item math_cache i))
        (if (String.equals (get_item e 0) src) (set! r e) #f)
        (set! i (+ i 1)) })
    r))

(define math_cache_clear (lambda () (set! math_cache (cdr '(0))) #t))

;; 插入一条（尾部是最久未用的，满了就丢尾部）
(define math_cache_put
  (lambda (e)
    (define out (new_list))
    (define n (length math_cache))
    (define i 0)
    (while (and (< i n) (< i (- MATH_CACHE_MAX 1)))
      { (push out (get_item math_cache i))
        (set! i (+ i 1)) })
    (set! math_cache (cons e out))
    #t))

;; 排一条公式：TeX 排版 → 超宽则缩字号重排 → 连同 op 流一起缓存
(define math_build
  (lambda (src maxw)
    (define ast (P.parse src))
    (define box #f)
    (define w 0.0)
    (define h 0.0)
    (define d 0.0)
    (define em 1.0)
    (define k 0.0)
    (define round 0)
    (define e #f)
    (L.set_emul em)
    (set! box (L.layout ast 0))
    (set! w (get_item box 0))
    (set! h (get_item box 1))
    (set! d (get_item box 2))
    ;; 超宽 → 按比例缩小 em 后**重排**。度量取整会让宽度的缩放不严格线性，
    ;; 因此按"还剩多少宽"再补几轮（最多 3 轮，常规一~两轮就到位）。
    (while (and (> w maxw) (< round 3))
      { (set! k (/ maxw w))
        (set! em (* em k))
        (if (< em MATH_MIN_EM) (set! em MATH_MIN_EM) #f)
        (L.set_emul em)
        (set! box (L.layout ast 0))
        (set! w (get_item box 0))
        (set! h (get_item box 1))
        (set! d (get_item box 2))
        (set! round (+ round 1)) })
    ;; 诊断行（端到端测试据此断言"公式真的排了、长公式被缩放"）。
    ;; 刻意**不打印公式源码**：LaTeX 里的 \r \n \t 在 (display) 时会被解释成控制字符
    ;; （实测 \right 会输出一个 CR，把日志行切碎、带歪端到端断言），只打长度。
    (display "MDMATH em=") (display em)
    (display " w=") (display w)
    (display " maxw=") (display maxw)
    (display " len=") (display (String.length src)) (newline)
    (set! e (mk7 src em w h d B.ops (B.count)))
    (math_cache_put e)
    e))

;; 取（或现排）一条公式的缓存条目
(define math_entry
  (lambda (src maxw)
    (define e (math_cache_find src))
    (if e e (math_build src maxw))))

;; 注入给 md_layout 的度量回调：(源串 最大宽) → (宽 上高 下深) | #f
(define doc_math_measure
  (lambda (src maxw)
    (define e (math_entry src maxw))
    (if e (mk3 (get_item e 2) (get_item e 3) (get_item e 4)) #f)))

;; 画一条公式（绘制端已把 cr/cg/cb 按 ST_MATH 设好）。
;; 裁剪矩形 = 公式行（左右各留 2px 给抗锯齿线段），与 mathjax 的视口裁剪同一机制。
(define draw_math
  (lambda (src x y h)
    (define e (math_entry src CW))
    (define bw 0)
    (define bh 0)
    (if e
        { (set! bw (Math.ceil (get_item e 2)))
          (set! bh (get_item e 3))
          (Gfx.set_clip (- x 2) y (+ bw 4) h)
          (D.set_view (- x 2) y (+ bw 4) h)
          (B.set_ops (get_item e 5) (get_item e 6))
          (D.draw x (+ y (+ MDL.MATH_PAD bh)) cr cg cb)
          (Gfx.reset_clip) }
        #f)
    #t))

;; 虚线框兜底（方案 D）：沿矩形边画 3px 亮 / 3px 暗的短线段 + alt 文本
(define draw_dashed_rect
  (lambda (x y w h r g bl)
    (define k 0)
    (define on 1)
    (while (< k w)
      { (if (== on 1)
            { (Gfx.fill_rect (+ x k) y 3 1 r g bl)
              (Gfx.fill_rect (+ x k) (- (+ y h) 1) 3 1 r g bl) }
            #f)
        (set! on (if (== on 1) 0 1))
        (set! k (+ k 3)) })
    (set! k 0)
    (set! on 1)
    (while (< k h)
      { (if (== on 1)
            { (Gfx.fill_rect x (+ y k) 1 3 r g bl)
              (Gfx.fill_rect (- (+ x w) 1) (+ y k) 1 3 r g bl) }
            #f)
        (set! on (if (== on 1) 0 1))
        (set! k (+ k 3)) })
    #t))

(define draw_image_missing
  (lambda (x y w h alt)
    (draw_dashed_rect x y w h 92 104 128)
    (Gfx.draw_text (MDL.clip_text (if (> (String.length alt) 0)
                                       (String.concat "[图片] " alt)
                                       "[图片缺失]")
                                   (- w 8))
                   (+ x 4) (+ y (Math.floor (/ (- h 14) 2)))
                   150 162 186)
    #t))

;; ---- 打开文档：读文本 → 解析 → 排版 --------------------------------------

;; 按**路径**打开（列表选中与文件选择器共用同一条链路）。
;;
;; 两种显示模式：
;;   stream（正常）：一切交给 md_view —— 打开时只切出并布局**第一段**（首屏 ~60-80ms），
;;                   其余段在后台按拍建索引，滚动时按需布局（见 on_tick / handle_touch）。
;;   notice（兜底）：文件读不出来/超上限时，把两条提示排成一个小文档（整篇排版，
;;                   内容只有两段），仍用同一套绘制代码。
(define open_path
  (lambda (path title)
    (define fsize 0)
    (define ok #f)
    (define msg #f)
    (set! cur_title title)
    (set! cur_path path)
    (set_doc_dir path)
    (set! fsize (Fs.size path))
    (if (null? fsize)
        (set! msg (notice_blocks "读取失败：文件不存在或不可读。"
                                 "请把文档放到资源目录的 doc/ 下。"))
        (if (> fsize DOC_MAX)
            (set! msg (notice_blocks
                        (String.concat "文档过大（"
                                       (String.concat (String.atom_to_string fsize) " 字节）："))
                        "本阅读器上限 1MB。"))
            { (set! ok (MDV.open path CW doc_image_size SEG_TARGET CACHE_LINES))
              (if ok #f (set! msg (notice_blocks "读取失败。" "文件可能已被删除或不可读。"))) }))
    (open_finish ok msg)
    #t))

;; 打开完成后的公共收尾（正常/兜底两条路共用）
(define open_finish
  (lambda (ok msg)
    (set! toc_items (cdr '(0)))
    (set! toc_open 0)
    (set! toc_scanned 0)
    (set! scroll 0)
    (set! mode 0)
    (if ok
        { (set! MDV_ok 1)
          (set! notice_lines (cdr '(0)))
          (set! notice_h 0)
          (index_work IDX_BYTES)         ;; 开屏后立刻推进一点索引（目录/估算随之更新）
          (set! toc_gen_seen -1) }
        { (set! MDV_ok 0)
          (set! notice_lines (MDL.layout msg CW doc_image_size))
          (set! notice_h (MDL.total_height notice_lines)) })
    (sync_scroll)
    ;; 诊断行（宿主 stdout / 浏览器控制台；端到端测试据此断言解析与排版规模）
    (display "MDSTAT blocks=")
    (display (if (== MDV_ok 1) (MDV.blocks_total) 1))
    (display " lines=")
    (display (if (== MDV_ok 1) (MDV.lines_total) (length notice_lines)))
    (display " images=")
    (display (if (== MDV_ok 1) (MDV.images) 0))
    (display " height=")
    (display content_h)
    (display " title=")
    (display cur_title)
    (display " seg=")
    (display (if (== MDV_ok 1) (MDV.count) 0))
    ;; 内存/分段诊断（工具链与回归测试用于守"不涨池也能读大文件"的预算）
    (display " heap=")
    (display (get_item (System.memstat) 3))
    (display " cache=")
    (display (if (== MDV_ok 1) (MDV.cache_lines) 0))
    (newline)
    #t))

;; 内置示例（内存行表；只读文件系统写不出 .md 时的兜底）：同样走 md_view 的单段模式
(define open_builtin
  (lambda ()
    { (set_doc_dir "")
      (set! cur_path "")
      (open_finish (MDV.open_mem MDE.lines CW doc_image_size) #f)
      (display "MDSTAT builtin title=") (display cur_title) (newline)
      #t }))

(define open_doc
  (lambda (idx)
    (define d (get_item docs idx))
    (define path (get_item d 0))
    (set! sel idx)
    ;; 换文档：旧公式缓存的 op 流不再需要。必须在 MDV.open 之前清——
    ;; 打开时会立刻排版首段（公式在那一刻就会排好并进缓存）。
    (math_cache_clear)
    (if (== (String.length path) 0)
        { (set! cur_title (get_item d 1))
          (open_builtin) }
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

;; ---- 目录（TOC）---------------------------------------------------------
;; 策略与主流低资源长文本阅读器一致：**快速打开 → 流式渲染 → 后台异步扫全文建目录**。
;;
;;   · 打开时只切/排第一段（~90ms），用户立刻可读；与此同时 `bg_work` 每拍用
;;     IDX_BYTES 的预算往后切段，**切段时顺手提取该段标题**（md_view.scan 里的
;;     headings_of：只判行首 '#'，跳过围栏与引用块，不解析不排版、不额外读盘）
;;     ⇒ 目录是索引的副产品，1MB 文档约 5s 就长齐（实测 153 → 2221 条）。
;;   · 段真正被排版时，md_view 会用**布局给出的精确 y** 覆盖该段的廉价条目。
;;   · 面板打开期间索引继续跑（IDX_BYTES_TOC 放大预算，面板渲染很轻），并用
;;     目录世代号 `(MDV.toc_gen)` 做**增量刷新**——只有条目真的变了才重建列表，
;;     不每拍 O(条目数) 重算；条目只追加在尾部，选中项下标不会失效。
;;   · 标题栏在未扫完时显示 "n 条 · x%"；跳转 = 设锚点 → 布局该段 → 用标题文本
;;     把估算 dy 精修到精确 y（见 md_view.toc_jump）。
;;
;; 面板是全屏覆盖式：自己消费输入与推屏；活动期间**不调 (EX.check)**，
;; 返回请求（硬 A/ESC）改为"关闭面板"（与文件选择器同一思路）。

(define toc_items (cdr '(0)))   ;; 当前文档目录：(层级 段号 种类 数值 文本) 的列表
(define toc_open 0)             ;; 1 = 面板打开
(define TOC_TOP 26)             ;; 条目区起始 y（页眉 18 + 8）
(define MENU_ROW_K 1.6)         ;; 菜单项高度 = 字号行高 × 1.6（触屏点按更好按）

;; 当前字模下菜单项的统一样式参数（行高 / 墨高 / 顶部内缩）
(define menu_row_h (lambda () (Math.floor (* line_h MENU_ROW_K))))
(define menu_ink_h (lambda () (if (== font_id 2) 16 12)))
(define menu_ink_pad (lambda () text_pad))

(define menu_metrics
  (lambda (info)
    ;; info = (Gfx.font_info id) → (半角 全角 行高 基线)
    (ML.metrics (get_item info 0) (get_item info 1) (menu_ink_h) (menu_ink_pad))
    #t))
(define TOC_BTN_W 40)
(define TOC_BTN_H 14)

(define toc_btn_x (lambda () (- (open_btn_x) (+ TOC_BTN_W 4))))
(define toc_btn_y (lambda () (- SH 17)))

(define hit_toc_btn
  (lambda (px py)
    (if (and (>= px (toc_btn_x)) (< px (+ (toc_btn_x) TOC_BTN_W)))
        (if (and (>= py (toc_btn_y)) (< py (+ (toc_btn_y) TOC_BTN_H))) 1 0)
        0)))

(define draw_toc_btn
  (lambda ()
    (Gfx.fill_rect (toc_btn_x) (toc_btn_y) TOC_BTN_W TOC_BTN_H 52 58 76)
    (Gfx.draw_rect (toc_btn_x) (toc_btn_y) TOC_BTN_W TOC_BTN_H 110 118 140)
    (Gfx.draw_text "目录" (+ (toc_btn_x) 7) (+ (toc_btn_y) 1) 225 228 235)
    #t))

(define toc_refresh
  (lambda ()
    (set! toc_items (MDV.toc_list))
    #t))

;; 目录条目 → 通用菜单条目（样式全在这里配：层级颜色 / 缩进 / 高亮态颜色）
;;   一级最亮、二三级渐暗；缩进按层级；被高亮时统一用更亮一档的同色系。
(define toc_menu_items
  (lambda ()
    (define n (length toc_items))
    (define i 0)
    (define it #f)
    (define lvl 0)
    (define out (new_list))
    (define txt "")
    (while (< i n)
      { (set! it (get_item toc_items i))
        (set! lvl (get_item it 0))
        (set! txt (get_item it 4))
        (if (== lvl 1)
            (push out (ML.item txt 238 244 252 (- lvl 1) 0 255 255 255))
            (if (== lvl 2)
                (push out (ML.item txt 206 218 236 (- lvl 1) 0 236 244 254))
                (push out (ML.item txt 168 182 204 (- lvl 1) 0 210 224 244))))
        (set! i (+ i 1)) })
    out))

;; 打开面板：选中项默认落在"当前阅读位置之前最近的标题"上。
;; 条目形状 = (层级 段号 种类 数值 文本)；种类 0 = 段内精确 y，1 = 段内码点偏移（估算）。
(define toc_open_panel
  (lambda ()
    (define n 0)
    (define i 0)
    (define it #f)
    (define seg 0)
    (define a (MDV.anchor))
    (define aseg (get_item a 0))
    (define ady (get_item a 1))
    (define near 0)
    (toc_refresh)
    (set! toc_gen_seen (MDV.toc_gen))
    (set! n (length toc_items))
    (set! i 0)
    (while (< i n)
      { (set! it (get_item toc_items i))
        (set! seg (get_item it 1))
        (if (< seg aseg)
            (set! near i)
            (if (and (== seg aseg) (== (get_item it 2) 0) (<= (get_item it 3) ady))
                (set! near i)
                #f))
        (set! i (+ i 1)) })
    ;; 交给通用菜单组件：条目样式 + 行高（字号行高 × 1.6）+ 视口（页眉与页脚之间）
    (ML.reset (toc_menu_items) (menu_row_h))
    (ML.geom 0 TOC_TOP SW (- (- SH BOT_H) TOC_TOP))
    (ML.back_top_right SW)          ;; 菜单自己的返回键（右上角，与阅读页「退出」同位）
    (App.set_exit_button 0 0 0 0)   ;; 停用宿主「退出」热区（右上角归本面板，避免误触全局退出）
    (ML.set_sel near)
    (set! toc_open 1)
    (display "MDTOC n=") (display n) (newline)
    #t))

(define toc_close
  (lambda ()
    (set! toc_open 0)
    (set! exitbtn_mute 4)      ;; 见 exitbtn_mute 注释
    #t))

;; 跳到第 i 条（视图层负责：设锚点 → 布局该段 → 用标题文本精修到精确 y）
(define toc_jump
  (lambda (i)
    (define n (length toc_items))
    (if (and (>= i 0) (< i n))
        { (MDV.toc_jump i)
          (sync_scroll)
          (display "MDJUMP i=") (display i) (newline)
          (set! toc_open 0) }
        #f)
    #t))

;; 面板绘制：底色 → **列表（组件绘制，含拖动滚动/高亮/滚动条）** → 页眉页脚盖住边界半行
(define draw_toc_panel
  (lambda ()
    (define n (length toc_items))
    (Gfx.fill_rect 0 0 SW SH 16 19 26)
    (ML.draw)
    (if (== n 0)
        (Gfx.draw_text (if (MDV.toc_done) "（本文没有 1~3 级标题）" "（正在扫描全文标题…）")
                       PAD_X TOC_TOP 150 162 186)
        #f)
    ;; 页眉：标题 + 条目数/扫描进度（左对齐排在标题后，右侧留给返回键）
    (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
    (Gfx.draw_text "目录" PAD_X 3 236 242 250)
    (if (not (MDV.toc_done))
        (Gfx.draw_text (String.concat (String.atom_to_string n)
                                      (String.concat " 条 · "
                                        (String.concat (String.atom_to_string (MDV.index_progress)) "%")))
                       38 3 150 168 190)
        (Gfx.draw_text (String.concat (String.atom_to_string n) " 条") 38 3 150 168 190))
    (ML.draw_back)
    (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
    (Gfx.draw_text "拖动/↑↓ 选择 · 点按跳转 · 右上返回" PAD_X (- SH 14) 130 140 155)
    (Gfx.refresh)
    #t))

(define apply_key_toc
  (lambda (k)
    (if (== k 1) (ML.move (- 0 1)) #f)
    (if (== k 2) (ML.move 1) #f)
    (if (== k 3) (ML.page (- 0 1)) #f)
    (if (== k 4) (ML.page 1) #f)
    (if (== k 13) (toc_jump (ML.sel_of)) #f)
    (if (== k 27) (toc_close) #f)
    #t))

(define tick_toc
  (lambda ()
    (define hit 0)
    { ;; 返回请求归面板消费（= 关闭面板），不交给 EX 去弹退出确认
      (if (App.back_requested) (toc_close) #f)
      (set! key (Key.take_press))
      (while (not (== key 0))
        { (apply_key_toc key)
          (set! key (Key.take_press)) })
      ;; 触屏：拖动 = 滑动滚动（1:1 跟手）；轻点条目 = 选中并跳转；
      ;; 轻点页眉/页脚区 = 关闭（组件只报"视口内的条目"，区外由这里处理）
      (set! hit (ML.step_touch))
      (if (>= hit 0)
          (toc_jump hit)
          (if (== hit -2)
              (if (== (ML.hit_back (ML.tap_x) (ML.tap_y)) 1) (toc_close) #f)  ;; 右上返回键
              #f))
      ;; 面板打开期间**索引继续跑**（预算放大）：全文目录几秒内补齐，
      ;; 条目只会追加在尾部，选中项下标不会失效。
      (index_work IDX_BYTES_TOC)
      ;; 只在"目录世代号变化"时重建条目表（避免每拍 O(条目数) 重建）
      (if (not (== (MDV.toc_gen) toc_gen_seen))
          { (set! toc_gen_seen (MDV.toc_gen))
            (toc_refresh)
            (set! toc_scanned (MDV.toc_scanned_count))
            ;; 增量换表：组件保留滚动位置与选中项（条目只追加在尾部）
            (ML.set_items (toc_menu_items))
            (ML.ensure_sel) }
          #f)
      (draw_toc_panel) }))

;; ---- 设置（模态框）-------------------------------------------------------
;; 第一个设置项：字体。系统有三套字模左右可用——
;;   0 = 12px 二值点阵（锐利、无灰阶）、1 = 12px 抗锯齿（默认）、2 = 16px 抗锯齿（更大更平滑）。
;; 字模决定**行高与字符步进**，所以换字体后必须：
;;   ① (Gfx.set_font id) 让绘制端换字模；
;;   ② 用 (Gfx.font_info id) 拿到 (半角 全角 行高 基线)，喂给 (MDL.set_font_metrics ...)；
;;   ③ 重新排版当前文档（折行位置、行高、表格列宽、目录 y 全都会变）。
;; 选择写入 data/md-font.txt，下次启动沿用。

(define FONT_PATH "data/md-font.txt")
(define font_id 1)       ;; 当前字模
(define line_h 15)       ;; 当前行高（由字模决定）
(define font_base 12)    ;; 基线（行顶到基线）
(define text_pad 1)      ;; 行顶到字模顶的偏移
(define pill_h 13)       ;; 行内代码药丸底高

(define SET_TOP 30)
(define settings_open 0)
(define SET_BTN_W 40)

(define set_btn_x (lambda () (- (toc_btn_x) (+ SET_BTN_W 4))))
(define set_btn_y (lambda () (- SH 17)))

(define hit_set_btn
  (lambda (px py)
    (if (and (>= px (set_btn_x)) (< px (+ (set_btn_x) SET_BTN_W)))
        (if (and (>= py (set_btn_y)) (< py (+ (set_btn_y) 14))) 1 0)
        0)))

(define font_name
  (lambda (id)
    (if (== id 0) "点阵 12px"
        (if (== id 1) "抗锯齿 12px"
            (if (== id 2) "抗锯齿 16px" "?")))))

;; 按字模 id 应用：查度量 → 设绘制字模 → 设排版度量 → 重新排版当前文档
(define apply_font
  (lambda (id)
    (define info (Gfx.font_info id))
    (define keep 0)
    (define old_h line_h)
    (define keep_a #f)
    (define keep_ratio 1)
    (if info
        { (set! font_id id)
          (Gfx.set_font id)
          (set! text_pad (if (== id 0) 0 1))
          (set! font_base (get_item info 3))
          (set! line_h (get_item info 2))
          (set! pill_h (- line_h 2))
          (MDL.set_font_metrics (get_item info 0) (get_item info 1) (get_item info 2))
          ;; 列表/目录/设置的行高跟着字模走（16px 字模下 15px 行距会挤）
          (set! ROW_H (+ line_h 5))
          (menu_metrics info)             ;; 把新字模的步进/墨高/内缩同步给菜单组件
          (Fs.write FONT_PATH (String.atom_to_string id))
          (display "MDFONT id=") (display id)
          (display " line_h=") (display line_h) (newline)
          ;; 换字模后所有段的折行/行高都变了：清空段缓存重来，但**保留阅读位置**——
          ;; 锚点按"段号 + 段内 dy 比例"换算（新行高/旧行高），重开文档后落回原处。
          (if (== mode 0)
              { (set! keep_a (MDV.anchor))
                (set! keep_ratio (/ line_h old_h))
                (if (> (String.length cur_path) 0)
                    (open_path cur_path cur_title)
                    (open_builtin))
                (if (== MDV_ok 1)
                    { (MDV.set_anchor (get_item keep_a 0)
                                      (Math.floor (* (get_item keep_a 1) keep_ratio)))
                      (MDV.ensure (get_item keep_a 0))
                      (sync_scroll) }
                    #f) }
              #f) }
        #f)
    #t))

;; 启动时读回上次的选择
(define load_font
  (lambda ()
    (define s (Fs.read FONT_PATH))
    (define v 1)
    (if (null? s) #f
        { (set! v (String.parseNumber s))
          (if (null? v) #f (apply_font (Math.floor v))) })
    (if (null? s) (apply_font 1) #f)
    #t))

;; 设置面板：同样交给通用菜单组件（三个字体单选项 = 三行"带琴键指示灯"的菜单条目）
;;   视口 = 「字体」小节标签下方到页脚之间；指示灯：当前字模实心、其余空心。
(define settings_menu_items
  (lambda ()
    (define i 0)
    (define out (new_list))
    (while (< i 3)
      { ;; 指示灯 2 = 实心（当前字模）/ 1 = 空心；文字色随主题，高亮态更亮
        (if (== i font_id)
            (push out (ML.item (font_name i) 236 242 250 0 2 255 255 255))
            (push out (ML.item (font_name i) 226 232 244 0 1 250 252 255)))
        (set! i (+ i 1)) })
    out))

(define settings_view_y (lambda () (+ SET_TOP 18)))
(define settings_view_h (lambda () (- (- SH BOT_H) (+ SET_TOP 18))))

(define draw_settings
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH 16 19 26)
    ;; 列表（组件绘制：高亮行只有底色，无左侧竖条；指示灯在左）
    (ML.set_items (settings_menu_items))
    (ML.draw)
    ;; 「字体」小节标签 + 底部说明（页眉页脚最后画，盖住边界半行）
    (Gfx.draw_text "字体" PAD_X SET_TOP 150 168 190)
    (Gfx.draw_text (String.concat "当前：" (String.concat (font_name font_id)
                                                          (String.concat "  行高 "
                                                                         (String.atom_to_string line_h))))
                   PAD_X (- (- SH BOT_H) 16) 150 168 190)
    (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
    (Gfx.draw_text "设置" PAD_X 3 236 242 250)
    (ML.draw_back)
    (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
    (Gfx.draw_text "↑↓ 选择 · 点按切换 · 右上返回" PAD_X (- SH 14) 130 140 155)
    (Gfx.refresh)
    #t))

(define settings_open_panel
  (lambda ()
    (ML.reset (settings_menu_items) (menu_row_h))
    (ML.geom 0 (settings_view_y) SW (settings_view_h))
    (ML.back_top_right SW)
    (App.set_exit_button 0 0 0 0)   ;; 同上：右上角归设置面板自己的返回键
    (ML.set_sel font_id)          ;; 高亮项默认落在当前字模上
    (set! settings_open 1)
    (display "MDSET open") (newline)
    #t))

(define apply_key_settings
  (lambda (k)
    (if (== k 1) (ML.move (- 0 1)) #f)
    (if (== k 2) (ML.move 1) #f)
    (if (== k 3) (ML.page (- 0 1)) #f)
    (if (== k 4) (ML.page 1) #f)
    (if (== k 13) (apply_font (ML.sel_of)) #f)
    (if (== k 27) { (set! settings_open 0) (set! exitbtn_mute 4) } #f)
    #t))

(define tick_settings
  (lambda ()
    (define hit 0)
    { (if (App.back_requested) { (set! settings_open 0) (set! exitbtn_mute 4) } #f)
      (set! key (Key.take_press))
      (while (not (== key 0))
        { (apply_key_settings key)
          (set! key (Key.take_press)) })
      ;; 触屏：拖动 = 滑动滚动（条目只有三条，实际不会滚，但手感一致）；
      ;; 轻点条目 = 选中并立即应用；轻点页眉 = 关闭
      (set! hit (ML.step_touch))
      (if (>= hit 0)
          (apply_font (ML.sel_of))
          (if (== hit -2)
              (if (== (ML.hit_back (ML.tap_x) (ML.tap_y)) 1)
                  { (set! settings_open 0) (set! exitbtn_mute 4) }
                  #f)
              #f))
      (draw_settings) }))

(define draw_set_btn
  (lambda ()
    (Gfx.fill_rect (set_btn_x) (set_btn_y) SET_BTN_W 14 52 58 76)
    (Gfx.draw_rect (set_btn_x) (set_btn_y) SET_BTN_W 14 110 118 140)
    (Gfx.draw_text "设置" (+ (set_btn_x) 7) (+ (set_btn_y) 1) 225 228 235)
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
        (Gfx.fill_rect (+ x 4) (+ y (Math.floor (/ (- line_h 4) 2))) 3 3 cr cg cb)
        #f)
    (if (== st MDL.ST_TABLE_SEP)
        (Gfx.fill_rect x y 1 h cr cg cb)
        #f)
    ;; 块级图片：交给宿主解码（带 LRU 缓存），失败就走虚线框兜底（方案 D）
    (if (== st MDL.ST_IMAGE)
        { (if (Gfx.draw_image (image_path (get_item sg 4)) x y w h)
              #f
              (draw_image_missing x y w h txt)) }
        #f)
    (if (== st MDL.ST_IMAGE_MISSING)
        (draw_image_missing x y w h txt)
        #f)
    (if (== st MDL.ST_HR)
        (Gfx.fill_rect x y w 1 cr cg cb)
        #f)
    (if (== st MDL.ST_QUOTE_BAR)
        (Gfx.fill_rect x y w h cr cg cb)
        #f)
    (if (== st MDL.ST_H2BAR)
        (Gfx.fill_rect x y w h cr cg cb)
        #f)
    ;; 块级数学公式：按缓存的 op 流直接画（不重排），x/行高由 md_layout 定好
    (if (== st MDL.ST_MATH)
        (draw_math txt x y h)
        #f)
    (if (and (> (String.length txt) 0) (== st MDL.ST_CODE))
        (Gfx.fill_rect (- x 1) (+ y text_pad) (+ w 2) pill_h 46 42 34)
        #f)
    ;; 链接下划线：基线在 y+13，下划线再往下 3px（y+16），与字形留出呼吸感
    (if (and (> (String.length txt) 0) (== st MDL.ST_LINK))
        (Gfx.fill_rect x (+ y (+ text_pad (+ font_base 3))) w 1 96 160 235)
        #f)
    ;; 三级标题：文字下划线（比字形亮一档的暗色，避免和链接下划线混淆；同样下移 3px）
    (if (and (> (String.length txt) 0) (== st MDL.ST_H3))
        (Gfx.fill_rect x (+ y (+ text_pad (+ font_base 2))) w 1 96 128 168)
        #f)
    ;; 图片段（ST_IMAGE / ST_IMAGE_MISSING）的 alt 与公式段的源串都不走通用文字绘制
    (if (and (> (String.length txt) 0)
             (not (== st MDL.ST_IMAGE))
             (not (== st MDL.ST_IMAGE_MISSING))
             (not (== st MDL.ST_MATH)))
        {
          (set! k 0)
          (while (< k style_weight)
            { (Gfx.draw_text txt (+ x k) (+ y text_pad) cr cg cb)
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
    (if (== MDV_ok 1)
        (draw_content_stream)
        (draw_content_notice))))

;; 流式文档：只遍历**缓存里的段**（视口 ± 若干屏），且段内一越界就收工。
;; 这与"整篇从第 0 行开始累加 y"的旧实现相比，每帧成本与文档大小无关。
(define draw_content_stream
  (lambda ()
    (define vt (MDV.top_y))
    (define n (MDV.n_cached))
    (define j 0)
    (define m #f)
    (define lines #f)
    (define i 0)
    (define y 0)
    (define bottom (- SH BOT_H))
    (define ln #f)
    (define stop 0)
    (while (< j n)
      { (set! m (MDV.cached j))
        (set! y (+ TOP_H (- (get_item m 1) vt)))     ;; 段首行的屏幕 y
        (set! lines (get_item m 3))
        (set! i 0)
        (set! stop 0)
        (while (and (< i (length lines)) (== stop 0))
          { (set! ln (get_item lines i))
            (if (> y bottom)
                (set! stop 1)                        ;; 屏外：本段收工
                (if (< (+ y (get_item ln 0)) TOP_H)
                    #f                               ;; 整行在页眉之上：跳过
                    (draw_line ln y)))
            (set! y (+ y (get_item ln 0)))
            (set! i (+ i 1)) })
        (set! j (+ j 1)) })
    #t))

;; 提示文案（内容只有两段，整篇排版即可）
(define draw_content_notice
  (lambda ()
    (define n (length notice_lines))
    (define i 0)
    (define y 0)
    (define ln #f)
    (define bottom (- SH BOT_H))
    (define stop 0)
    (set! y (- TOP_H scroll))
    (while (and (< i n) (== stop 0))
      { (set! ln (get_item notice_lines i))
        (if (> y bottom)
            (set! stop 1)
            (if (< (+ y (get_item ln 0)) TOP_H)
                #f
                (draw_line ln y)))
        (set! y (+ y (get_item ln 0)))
        (set! i (+ i 1)) })
    #t))

;; ---- 绘制：页眉 / 页脚 / 滚动条 -----------------------------------------

(define draw_header
  (lambda ()
    (define pct 0)
    (define mx (max_scroll))
    (define txt "100%")
    (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
    ;; 标题按可用宽度裁剪（右侧留给百分比与右上角的「退出」按钮，见 tools/e2e 的像素判据）
    (Gfx.draw_text (MDL.clip_text cur_title 232) PAD_X 3 236 242 250)
    (if (> mx 0)
        { (set! pct (Math.floor (/ (* (MDV.top_y) 100) (MDV.total_height))))
          (if (> pct 100) (set! pct 100) #f)
          ;; 索引还没走到文件尾时百分比是**估算**，用 ≈ 如实标注（宁可不精确也要诚实）
          (set! txt (if (MDV.scan_done)
                        (String.concat (String.atom_to_string pct) "%")
                        (String.concat "≈" (String.concat (String.atom_to_string pct) "%"))))
          (Gfx.draw_text txt 250 3 150 168 190) }
        (Gfx.draw_text "100%" 250 3 150 168 190))
    #t))

(define draw_footer
  (lambda ()
    (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
    (if (== mode 0)
        (Gfx.draw_text "拖动/↑↓ 滚动" PAD_X (- SH 14) 130 140 155)
        (Gfx.draw_text "↑↓ 选择 · Enter 打开" PAD_X (- SH 14) 130 140 155))
    (if (and (== mode 0) (== MDV_ok 1) (> (MDV.toc_count) 0))
        (draw_toc_btn)      ;; 底栏右侧偏左：目录（仅阅读页且有标题时）
        #f)
    (draw_set_btn)    ;; 设置（两种模式都有：字体是 App 级设置）
    (draw_open_btn)   ;; 底栏右侧：打开选择器（触屏唯一入口）
    #t))

(define draw_scrollbar
  (lambda ()
    (define vh 0)
    (define mx 0)
    (define th 0)
    (define bar 0)
    (define by 0)
    (set! mx (max_scroll))
    (if (> mx 0)
        { (set! vh (view_h))
          (set! th (MDV.total_height))
          (set! bar (Math.floor (/ (* vh vh) th)))
          (if (< bar 10) (set! bar 10) #f)
          (set! by (+ TOP_H (Math.floor (/ (* (MDV.top_y) (- vh bar)) mx))))
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
    (if (== settings_open 1)
        (draw_settings)         ;; 设置面板：整屏归它（含推屏）
        (if (== toc_open 1)
            (draw_toc_panel)    ;; 目录面板
            (draw_page)))))

(define draw_page
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
    ;; 面板刚关闭的几帧：把宿主热区清零，避免同一次点按被"面板返回键 + 宿主退出键"重复解释
    (if (> exitbtn_mute 0)
        { (App.set_exit_button 0 0 0 0)
          (set! exitbtn_mute (- exitbtn_mute 1)) }
        #f)
    (Gfx.refresh)
    #t))

;; ---- 交互：链接命中测试 --------------------------------------------------
;; 返回命中的链接目标（空串表示没点中）。行/段都是排版期定死的绝对坐标，
;; 因此命中测试就是"滚动偏移 + 行高累加"的算术。
(define hit_link
  (lambda (px py)
    ;; 屏幕 y → 文档绝对 y 需要减去页眉高度：视图层的 top 参数按"内容坐标原点"给
    (if (== MDV_ok 1)
        (MDV.hit_link px py (- (MDV.top_y) TOP_H))
        "")))

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
              { (scroll_by_px (- 0 dyy))
                (sync_scroll) }
              #f) }
        { (if (== prev_down 1)
              { (set! prev_down 0)
                (if (< (Math.abs drag_total) 8)
                    (if (== (hit_set_btn press_x press_y) 1)
                        { (settings_open_panel) }
                        (if (== (hit_open_btn press_x press_y) 1)
                        (start_picker)
                        (if (and (== mode 0)
                                 (== (hit_toc_btn press_x press_y) 1)
                                 (== MDV_ok 1)
                                 (> (MDV.toc_count) 0))
                            (toc_open_panel)
                            (if (== mode 0)
                                (tap_reader press_x press_y)
                                (tap_list press_y)))))
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
        { (if (== k 1) { (scroll_by_px (- 0 15)) (sync_scroll) } #f)
          (if (== k 2) { (scroll_by_px 15) (sync_scroll) } #f)
          (if (== k 3) { (scroll_by_px (- 0 (view_h))) (sync_scroll) } #f)
          (if (== k 4) { (scroll_by_px (view_h)) (sync_scroll) } #f) })
    #t))

(define handle_keys
  (lambda ()
    (set! key (Key.take_press))
    (while (not (== key 0))
      { (apply_key key)
        (set! key (Key.take_press)) })
    #t))

;; ---- 主循环 --------------------------------------------------------------

;; ---- 后台异步工作（流式加载的第二半：**阅读不被打断，索引/目录在后面追**）------
;; 每拍做两件事，各有预算，保证 GUI 不掉帧：
;;   ① **索引**：从当前位置往后切段（只读块 + 切行 + 顺手提取标题，≈0.2~4ms/段）。
;;      索引是**目录的副产品**——扫到哪，目录就完整到哪，所以不需要"打开目录时再
;;      全篇扫一遍"的二次读盘；页眉百分比也随索引推进从估算变成实测。
;;      预算按字节给（IDX_BYTES），文档多大都不改变单拍开销。
;;   ② **预取**：把锚点相邻的一段排版进缓存（≈60ms，正好一拍），滚动才不卡。
;; 方向跟随阅读位置：段内剩余不足一屏时优先预取下一段。
(define IDX_BYTES 12288)       ;; 每拍索引预算（字节）≈ 3 段
(define IDX_BYTES_TOC 40960)   ;; 目录面板打开时的索引预算（面板渲染轻，可以快点）

(define index_work
  (lambda (budget_bytes)
    (define b0 (MDV.bytes_indexed))
    (define segs 0)
    (define go 1)
    (while (and (== go 1) (< segs 16))
      { (if (and (not (MDV.scan_done))
                 (< (- (MDV.bytes_indexed) b0) budget_bytes))
            { (if (> (MDV.index_step 1) 0)
                  (set! segs (+ segs 1))
                  (set! go 0)) }
            (set! go 0)) })
    segs))

(define bg_work
  (lambda ()
    (define a #f)
    (define seg 0)
    (define dy 0)
    (define dir 1)
    (if (== MDV_ok 1)
        { (index_work IDX_BYTES)
          (set! a (MDV.anchor))
          (set! seg (get_item a 0))
          (set! dy (get_item a 1))
          ;; 段内剩余不足一屏 → 预取下一段；否则预取上一段（回滚更顺）
          (if (< (- (MDV.height seg) dy) (view_h))
              (set! dir 1)
              (set! dir -1))
          (if (== (MDV.prefetch dir) 0)
              (MDV.prefetch (- 0 dir))
              #f) }
        #f)
    #t))

(define on_tick
  (lambda ()
    (if (FP.active)
        (tick_picker)          ;; 选择器活动期间：输入/绘制全归它，EX 不抢返回请求
        (if (== settings_open 1)
            (tick_settings)    ;; 设置面板同理（返回请求 = 关闭面板）
        (if (== toc_open 1)
            (tick_toc)         ;; 目录面板（返回请求 = 关闭面板）
            { (EX.check)
              (handle_keys)
              (handle_touch)
              (bg_work)
              (sync_scroll)
              (draw_all) })))))

;; ---- 启动 ----------------------------------------------------------------

(define boot
  (lambda ()
    { (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (set! CW (- SW (* 2 PAD_X)))
      (MDL.set_math_measure doc_math_measure)  ;; 打开公式排版能力（必须在排版之前）
      (load_font)           ;; 恢复上次选择的字模（字体影响排版度量，必须在打开文档之前）
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