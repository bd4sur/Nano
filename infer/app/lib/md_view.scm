;; ============================================================================
;; md_view.scm —— Markdown 大文件的**分段视图层**（段缓存 / 锚点滚动 / 预取 / 目录）
;;
;; 四层分工：
;;   md_stream.scm  字节流 → 段（行表 + 字节/行/码点记账）
;;   md_parse.scm   行表 → 块树               （不改）
;;   md_layout.scm  块树 → 显示行             （仅加"续接缝"开关与整篇图片预算）
;;   本模块         段 → **窗口**：按需布局、缓存若干段、锚点滚动、预取、高度估算、目录
;;
;; 为什么这样设计（WASM 实测：每 KB 源码 ≈ 40KB 堆（块树）+ 26KB（显示行）+ 15ms）：
;;   · 整篇物化需要 >60MB 堆（1MB 文档），而解释器池只有 16MB heap → 必须分窗口；
;;   · 布局一段 4KB 源码 ≈ 60ms，因此段要小、要预取，且**段内滚动零重排**；
;;   · 段高度只有布局过才知道：未访问段按"码点数 × 实测校准系数"估算，而滚动的视觉
;;     位置由**锚点 (段号, 段内 dy)** 决定，估算被修正时画面不会跳。
;;
;; 缓存策略：已布局段按"总行数上限"淘汰（离锚点最远的先淘汰，锚点段永不淘汰）；
;;   每拍最多布局 1 段（预取预算），避免掉帧。
;;
;; 解释器约束（manual.md）：算术恒二参数、无 let/list/append/map、空表现造、
;;   push 返回 #undefined、while 循环体不是独立词法区域（循环变量都在函数体顶层 define）。
;;   另外：**不要用重复 concat 拼大字符串**（实测会在 ~10KB 处因池内碎片而 OOM）。
;;
;; 单例：模块级状态（App 单前台、单文档）。
;; ============================================================================

(native String)
(native Math)

(import MD "app/lib/md_parse.scm")
(import MDL "app/lib/md_layout.scm")
(import MDST "app/lib/md_stream.scm")

(define new_list (lambda () (cdr '(0))))
(define min2 (lambda (a b) (if (< a b) a b)))
(define max2 (lambda (a b) (if (> a b) a b)))

;; ---- 状态 ---------------------------------------------------------------

(define vw_path "")
(define vw_width 308)
(define vw_measure #f)          ;; 图片度量回调 (path → (宽 高) | #f)
(define vw_segs (new_list))     ;; 段记录（md_stream 的 9 元组），按段号
(define vw_h (new_list))        ;; 每段精确高度（0 = 尚未布局）
(define vw_nlines (new_list))   ;; 每段显示行数（诊断）
(define vw_nblocks (new_list))  ;; 每段块数（诊断）
(define vw_toc (new_list))      ;; 每段 TOC 条目（列表）
(define vw_toc_scanned 0)       ;; 目录廉价扫描已覆盖的段数
(define vw_scan_done 0)         ;; 1 = 段扫描已到文件尾
(define vw_bytes_done 0)        ;; 已切段覆盖的源码字节数
(define vw_chars_done 0)        ;; 已切段覆盖的源码码点数
(define vw_cal 1.30)            ;; 校准：像素/码点（初值，随实测更新）
(define vw_sum_h 0)             ;; 已布局段高度合计（校准用）
(define vw_sum_chars 0)         ;; 已布局段码点合计（校准用）
(define vw_sum_bytes 0)         ;; 已布局段字节合计（校准用）
(define vw_images 0)            ;; 已画出的图片数（整篇预算，见 md_layout IMG_MAX_COUNT）
(define vw_max_lines 4000)      ;; 缓存行数上限（≈4MB 堆）
(define vw_evicted 0)           ;; 淘汰计数（诊断）
;; 缓存（3 个并行列表）
(define vw_c_seg (new_list))
(define vw_c_lines (new_list))
(define vw_c_h (new_list))
;; 锚点
(define vw_seg 0)
(define vw_dy 0)
(define vw_mem #f)              ;; 非 #f = 内存文档（内置示例；不走 Fs）

;; ---- 打开 / 重置 --------------------------------------------------------

(define open
  (lambda (path width measure seg_target max_lines)
    (set! vw_path path)
    (set! vw_width width)
    (set! vw_measure (if (undefined? measure) (lambda (p) #f) measure))
    (set! vw_segs (new_list))
    (set! vw_h (new_list))
    (set! vw_nlines (new_list))
    (set! vw_nblocks (new_list))
    (set! vw_toc (new_list))
    (set! vw_toc_scanned 0)
    (set! vw_scan_done 0)
    (set! vw_bytes_done 0)
    (set! vw_chars_done 0)
    (set! vw_cal 1.30)
    (set! vw_sum_h 0)
    (set! vw_sum_chars 0)
    (set! vw_sum_bytes 0)
    (set! vw_images 0)
    (set! vw_max_lines (if (undefined? max_lines) 4000 max_lines))
    (set! vw_evicted 0)
    (set! vw_c_seg (new_list))
    (set! vw_c_lines (new_list))
    (set! vw_c_h (new_list))
    (set! vw_seg 0)
    (set! vw_dy 0)
    (set! vw_mem #f)
    (if (MDST.open path #undefined seg_target #undefined #undefined)
        { (scan 1)        ;; 切出第一段
          (ensure 0) }    ;; 布局首屏（首段）
        #f)))

;; 内存文档（内置示例；只读文件系统下写不出 .md 时的兜底）：把行表当成**单段**文档，
;; 之后滚动/目录/命中测试全部走同一条路径，App 无需双份绘制逻辑。
(define open_mem
  (lambda (lines width measure)
    (define chars 0)
    (define bytes 0)
    (define k 0)
    (set! vw_path "")
    (set! vw_width width)
    (set! vw_measure (if (undefined? measure) (lambda (p) #f) measure))
    (set! vw_segs (new_list))
    (set! vw_h (new_list))
    (set! vw_nlines (new_list))
    (set! vw_nblocks (new_list))
    (set! vw_toc (new_list))
    (set! vw_toc_scanned 0)
    (set! vw_scan_done 1)
    (set! vw_bytes_done 0)
    (set! vw_chars_done 0)
    (set! vw_cal 1.30)
    (set! vw_sum_h 0)
    (set! vw_sum_chars 0)
    (set! vw_sum_bytes 0)
    (set! vw_images 0)
    (set! vw_evicted 0)
    (set! vw_c_seg (new_list))
    (set! vw_c_lines (new_list))
    (set! vw_c_h (new_list))
    (set! vw_seg 0)
    (set! vw_dy 0)
    (set! vw_mem #f)
    (set! k 0)
    (while (< k (length lines))
      { (set! chars (+ chars (+ (String.length (get_item lines k)) 1)))
        (set! k (+ k 1)) })
    (set! bytes chars)
    (set! vw_mem lines)
    (push vw_segs (cons 0 (cons bytes (cons (length lines)
      (cons chars (cons 0 (cons 0 (cons 96 (cons 0 (cons 0 (cdr '(0))))))))))))
    (push vw_h 0)
    (push vw_nlines 0)
    (push vw_nblocks 0)
    (push vw_toc (new_list))
    (set! vw_bytes_done bytes)
    (set! vw_chars_done chars)
    (ensure 0)
    #t))

(define path (lambda () vw_path))
(define width (lambda () vw_width))
(define count (lambda () (length vw_segs)))
(define scan_done (lambda () (== vw_scan_done 1)))
(define bytes_total (lambda () (MDST.size)))
(define bytes_indexed (lambda () vw_bytes_done))
(define images (lambda () vw_images))
(define cal (lambda () vw_cal))
(define evicted (lambda () vw_evicted))

;; ---- 段扫描（切段，不排版）--------------------------------------------

(define scan
  (lambda (budget)
    (define added 0)
    (define r #f)
    (define go 1)
    (define rec #f)
    (while (and (== go 1) (< added budget) (== vw_scan_done 0))
      { (set! r (MDST.next))
        (if r
            { (set! rec (get_item r 0))
              (push vw_segs rec)
              (push vw_h 0)
              (push vw_nlines 0)
              (push vw_nblocks 0)
              (push vw_toc (new_list))
              (set! vw_bytes_done (+ vw_bytes_done (get_item rec 1)))
              (set! vw_chars_done (+ vw_chars_done (get_item rec 3)))
              (set! added (+ added 1)) }
            { (set! vw_scan_done 1)
              (set! go 0) }) })
    added))

;; 扫描到文件尾（End 键 / 目录全量化时用；调用方应先画"正在计算"提示）
(define scan_all
  (lambda ()
    (while (== vw_scan_done 0)
      (scan 8))
    #t))

;; ---- 高度：精确（已布局）/ 估算（按码点数）-------------------------------

(define est_h
  (lambda (i)
    (define rec (get_item vw_segs i))
    (define e 0)
    (set! e (Math.floor (* (get_item rec 3) vw_cal)))
    (if (< e 15) 15 e)))

(define height
  (lambda (i)
    (define h (get_item vw_h i))
    (if (> h 0) h (est_h i))))

;; 段 0..i-1 的累计高度（已布局段精确、其余估算）
(define height_upto
  (lambda (i)
    (define n (min2 i (length vw_h)))
    (define k 0)
    (define s 0)
    (while (< k n)
      { (set! s (+ s (height k)))
        (set! k (+ k 1)) })
    s))

;; 全文档高度（含"尚未切段"的剩余字节，按已测字节/码点比估算）
(define total_height
  (lambda ()
    (define h (height_upto (length vw_segs)))
    (define left 0)
    (define bpc 1.9)
    (define est 0)
    (set! left (- (MDST.size) vw_bytes_done))
    (if (> left 0)
        { (if (> vw_chars_done 0) (set! bpc (/ vw_bytes_done vw_chars_done)) #f)
          (if (< bpc 0.3) (set! bpc 0.3) #f)
          (set! est (Math.floor (* (/ left bpc) vw_cal)))
          (set! h (+ h est)) }
        #f)
    (if (< h 15) 15 h)))

;; ---- 缓存 ---------------------------------------------------------------

(define cached_idx
  (lambda (i)
    (define n (length vw_c_seg))
    (define k 0)
    (define r -1)
    (while (< k n)
      { (if (== (get_item vw_c_seg k) i) (set! r k) #f)
        (set! k (+ k 1)) })
    r))

(define cache_len
  (lambda ()
    (define n (length vw_c_lines))
    (define k 0)
    (define s 0)
    (while (< k n)
      { (set! s (+ s (length (get_item vw_c_lines k))))
        (set! k (+ k 1)) })
    s))

;; 删掉缓存里第 idx 项（重建三个并行列表）
(define cache_drop
  (lambda (idx)
    (define s2 (new_list))
    (define l2 (new_list))
    (define h2 (new_list))
    (define n (length vw_c_seg))
    (define k 0)
    (while (< k n)
      { (if (== k idx) #f
            { (push s2 (get_item vw_c_seg k))
              (push l2 (get_item vw_c_lines k))
              (push h2 (get_item vw_c_h k)) })
        (set! k (+ k 1)) })
    (set! vw_c_seg s2)
    (set! vw_c_lines l2)
    (set! vw_c_h h2)
    (set! vw_evicted (+ vw_evicted 1))
    #t))

;; 淘汰：超出行数上限时，删掉离锚点最远的一段（锚点段本身不删）
(define cache_evict
  (lambda ()
    (define n (length vw_c_seg))
    (define guard 0)
    (define worst -1)
    (define wd -1)
    (define k 0)
    (define d 0)
    (while (and (> (cache_len) vw_max_lines) (> n 1) (< guard 64))
      { (set! n (length vw_c_seg))
        (set! worst -1)
        (set! wd -1)
        (set! k 0)
        (while (< k n)
          { (if (== (get_item vw_c_seg k) vw_seg) #f
                { (set! d (Math.abs (- (get_item vw_c_seg k) vw_seg)))
                  (if (> d wd) { (set! wd d) (set! worst k) } #f) })
            (set! k (+ k 1)) })
        (if (>= worst 0) (cache_drop worst) #f)
        (set! guard (+ guard 1)) })
    #t))

(define cache_put
  (lambda (i lines h)
    (if (>= (cached_idx i) 0)
        #f
        { (push vw_c_seg i)
          (push vw_c_lines lines)
          (push vw_c_h h) })
    (cache_evict)
    #t))

;; ---- TOC 存储 -----------------------------------------------------------

;; 布局得到的是精确 y（条目 = (层级 y 文本)）；廉价扫描得到的是段内码点偏移。
;; 统一存成 (层级 种类 数值 文本)：种类 0 = 段内 y，1 = 段内码点偏移。
(define toc_store
  (lambda (i entries)
    (define n (length entries))
    (define k 0)
    (define e #f)
    (define out (new_list))
    (define lvl 0)
    (while (< k n)
      { (set! e (get_item entries k))
        (set! lvl (get_item e 0))
        (if (<= lvl 3)
            (push out (cons lvl (cons 0 (cons (get_item e 1)
                      (cons (get_item e 2) (cdr '(0)))))))
            #f)
        (set! k (+ k 1)) })
    (set_item! vw_toc i out)
    #t))

;; ---- 布局一段 -----------------------------------------------------------

(define ensure
  (lambda (i)
    (define rec #f)
    (define lines #f)
    (define bs #f)
    (define out #f)
    (define h 0)
    (define chars 0)
    (define bytes 0)
    (if (or (< i 0) (>= i (length vw_segs))) #f
        (if (> (get_item vw_h i) 0) #t
            { (set! rec (get_item vw_segs i))
              (set! lines (if vw_mem vw_mem
                              (MDST.at (get_item rec 0) (get_item rec 2)
                                       (get_item rec 4) (get_item rec 5)
                                       (get_item rec 6) (get_item rec 7) (get_item rec 8))))
              (if lines
                  { (set! bs (MD.parse_lines lines))
                    ;; 续接：ca=1 丢弃本段尾间隙；olen>0 / clen>0 时首/末块是"接上"的代码块
                    (set! out (MDL.layout_segment bs vw_width vw_measure
                               (get_item rec 5)
                               (if (> (get_item rec 7) 0) 1 0)
                               (if (> (get_item rec 8) 0) 1 0)
                               vw_images))
                    (set! h (MDL.total_height out))
                    (if (< h 1) (set! h 1) #f)
                    (set! chars (max2 1 (get_item rec 3)))
                    (set! bytes (max2 1 (get_item rec 1)))
                    (set! vw_images (MDL.image_count))
                    (set_item! vw_h i h)
                    (set_item! vw_nlines i (length out))
                    (set_item! vw_nblocks i (length bs))
                    (set! vw_sum_h (+ vw_sum_h h))
                    (set! vw_sum_chars (+ vw_sum_chars chars))
                    (set! vw_sum_bytes (+ vw_sum_bytes bytes))
                    (if (> vw_sum_chars 0) (set! vw_cal (/ vw_sum_h vw_sum_chars)) #f)
                    (toc_store i (MDL.toc))
                    (cache_put i out h)
                    #t }
                  #f) }))))

;; ---- 锚点与滚动 ---------------------------------------------------------

(define anchor (lambda () (cons vw_seg (cons vw_dy (cdr '(0))))))
(define set_anchor
  (lambda (seg dy)
    (define s seg)
    (define n (length vw_segs))
    (if (< s 0) (set! s 0) #f)
    (if (>= s n) (set! s (- n 1)) #f)
    (if (< s 0) (set! s 0) #f)
    (set! vw_seg s)
    (set! vw_dy (max2 0 dy))
    #t))

;; 视图顶部在文档中的绝对 y（估算参与，仅供百分比/滚动条）
(define top_y (lambda () (+ (height_upto vw_seg) vw_dy)))

;; 按像素滚动（px > 0 = 内容向上走 = 看后面的内容）。
;; vh = 可视高度（像素）：用于**底部夹取**——文档末尾必须仍被内容填满，
;; 不能停在"最后一行顶到页眉、下方留白"的位置。
;; 跨越段边界时按需布局相邻段（预取到位时无额外开销）。
(define scroll_by
  (lambda (px vh)
    (define go 1)
    (define guard 0)
    (define h 0)
    (define max_top 0)
    (define over 0)
    (while (and (== go 1) (< guard 64))
      { (set! h (height vw_seg))
        (if (>= (+ vw_dy px) h)
            { ;; 越过本段：下一段还没切出来就先切一段（大文件快速滚动时按需推进索引）
              (if (>= (+ vw_seg 1) (length vw_segs))
                  (if (== vw_scan_done 0) (scan 1) #f)
                  #f)
              (if (< (+ vw_seg 1) (length vw_segs))
                  { (set! px (- (+ vw_dy px) h))
                    (set! vw_seg (+ vw_seg 1))
                    (set! vw_dy 0)
                    (ensure vw_seg) }
                  { (set! vw_dy (- h 1))
                    (set! go 0) }) }
            (if (< (+ vw_dy px) 0)
                { (if (> vw_seg 0)
                      { (ensure (- vw_seg 1))
                        (set! vw_seg (- vw_seg 1))
                        (set! vw_dy 0)
                        (set! px (+ px (height vw_seg))) }
                      { (set! vw_dy 0)
                        (set! go 0) }) }
                { (set! vw_dy (+ vw_dy px))
                  (set! go 0) }))
        (set! guard (+ guard 1)) })
    ;; 底部夹取：视图顶部不得超过 (总高 - 可视高)
    (set! max_top (- (total_height) vh))
    (if (< max_top 0) (set! max_top 0) #f)
    (set! guard 0)
    (set! over (- (top_y) max_top))
    (while (and (> over 0) (< guard 4000))
      { (if (>= over vw_dy)
            { (if (> vw_seg 0)
                  { (set! over (- over (+ vw_dy 1)))
                    (ensure (- vw_seg 1))
                    (set! vw_seg (- vw_seg 1))
                    (set! vw_dy (- (height vw_seg) 1)) }
                  { (set! vw_dy 0)
                    (set! over 0) }) }
            { (set! vw_dy (- vw_dy over))
              (set! over 0) })
        (set! guard (+ guard 1)) })
    #t))

;; ---- 预取 ---------------------------------------------------------------

;; dir = 1 向下 / -1 向上：布局一段相邻段；返回 1 = 本次做了工作
(define prefetch
  (lambda (dir)
    (define t 0)
    (define did 0)
    (if (== dir 1)
        { (set! t (+ vw_seg 1))
          (if (>= t (length vw_segs))
              (if (== vw_scan_done 0) (scan 1) #f)
              #f)
          (if (< t (length vw_segs))
              { (if (> (get_item vw_h t) 0) #f
                    { (ensure t)
                      (set! did 1) }) }
              #f) }
        { (set! t (- vw_seg 1))
          (if (>= t 0)
              { (if (> (get_item vw_h t) 0) #f
                    { (ensure t)
                      (set! did 1) }) }
              #f) })
    did))

;; ---- 绘制支持 -----------------------------------------------------------

(define n_cached (lambda () (length vw_c_seg)))
(define cache_lines (lambda () (cache_len)))

;; 第 j 个缓存项 → (段号 段起始绝对y 段高 行表)
(define cached
  (lambda (j)
    (define i (get_item vw_c_seg j))
    (cons i (cons (height_upto i) (cons (get_item vw_c_h j)
                                        (cons (get_item vw_c_lines j) (cdr '(0))))))))

;; 行内节点里"标题样式"的文本（TOC 跳转精确对齐用）
(define line_head_text
  (lambda (ln)
    (define segs (get_item ln 2))
    (define n (length segs))
    (define k 0)
    (define sg #f)
    (define st 0)
    (define out "")
    (while (< k n)
      { (set! sg (get_item segs k))
        (set! st (get_item sg 1))
        (if (and (>= st MDL.ST_H1) (<= st MDL.ST_H3))
            (set! out (String.concat out (get_item sg 0)))
            #f)
        (set! k (+ k 1)) })
    out))

;; 在段 seg 的行表里找"标题文本以 head 开头"的第一行，返回其段内 y（找不到 -1）。
;; 供 TOC 跳转在布局完成后把估算 dy 修正为精确值。
(define find_heading_y
  (lambda (seg head)
    (define idx (cached_idx seg))
    (define y 0)
    (define r -1)
    (define i 0)
    (define lines #f)
    (define ln #f)
    (define t "")
    (define hl 0)
    (if (< idx 0) -1
        { (set! lines (get_item vw_c_lines idx))
          (set! hl (String.length head))
          (while (and (< i (length lines)) (< r 0))
            { (set! ln (get_item lines i))
              (set! t (line_head_text ln))
              (if (and (> (String.length t) 0) (> hl 0))
                  (if (String.equals (String.slice head 0 (min2 hl (String.length t))) t)
                      (set! r y)
                      #f)
                  #f)
              (set! y (+ y (get_item ln 0)))
              (set! i (+ i 1)) })
          r })))

;; ---- 命中测试（链接）---------------------------------------------------

;; 返回命中的链接目标（空串 = 未命中）。只在缓存窗口内做算术。
(define hit_link
  (lambda (px py view_top)
    (define n (length vw_c_seg))
    (define j 0)
    (define top 0)
    (define lines #f)
    (define y 0)
    (define i 0)
    (define ln #f)
    (define segs #f)
    (define k 0)
    (define sg #f)
    (define sx 0)
    (define r "")
    (define stop 0)
    (while (and (< j n) (== stop 0))
      { (set! top (- (height_upto (get_item vw_c_seg j)) view_top))
        (set! lines (get_item vw_c_lines j))
        (set! y top)
        (set! i 0)
        (while (and (< i (length lines)) (== stop 0))
          { (set! ln (get_item lines i))
            (if (and (>= py y) (< py (+ y (get_item ln 0))))
                { (set! segs (get_item ln 2))
                  (set! k 0)
                  (while (< k (length segs))
                    { (set! sg (get_item segs k))
                      (if (== (get_item sg 1) MDL.ST_LINK)
                          { (set! sx (+ 6 (get_item sg 2)))     ;; PAD_X = 6（App 约定）
                            (if (and (>= px sx) (< px (+ sx (get_item sg 3))))
                                { (set! r (get_item sg 4))
                                  (set! k (length segs)) }
                                #f) }
                          #f)
                      (set! k (+ k 1)) })
                  (set! stop 1) }
                (set! y (+ y (get_item ln 0))))
            (set! i (+ i 1)) })
        (set! j (+ j 1)) })
    r))

;; ---- 目录（TOC）--------------------------------------------------------

;; 廉价扫描：只判行首 '#'（跳过围栏与引用块），不解析、不排版。
;; 记录段内码点偏移，跳转时先估算 dy、布局后再精确定位（见 toc_jump）。
(define toc_scan
  (lambda (budget)
    (define total (length vw_segs))
    (define done 0)
    (define rec #f)
    (define lines #f)
    (define i 0)
    (define ln "")
    (define fence #f)
    (define info #f)
    (define lvl 0)
    (define off 0)
    (define out (new_list))
    (define txt "")
    (while (and (< vw_toc_scanned total) (< done budget))
      { (set! rec (get_item vw_segs vw_toc_scanned))
        ;; 已布局的段用布局给出的精确条目，不再重复扫描
        (if (> (get_item vw_h vw_toc_scanned) 0)
            #f
            { (set! lines (MDST.at (get_item rec 0) (get_item rec 2)
                                   (get_item rec 4) (get_item rec 5)
                                   (get_item rec 6) (get_item rec 7) (get_item rec 8)))
              (if lines
                  { (set! out (new_list))
                    (set! fence #f)
                    (set! off 0)
                    (set! i 0)
                    (while (< i (length lines))
                      { (set! ln (get_item lines i))
                        (if fence
                            { (if (MD.fence_close ln fence) (set! fence #f) #f) }
                            { (set! info (MD.fence_open ln))
                              (if info
                                  (set! fence info)
                                  { (if (>= (MD.quote_content_at ln) 0)
                                        #f
                                        { (set! lvl (MD.heading_level ln))
                                          (if (> lvl 0)
                                              { (set! txt (MD.trim (String.slice ln lvl (String.length ln))))
                                                (push out (cons (min2 lvl 3)
                                                            (cons 1 (cons off
                                                              (cons txt (cdr '(0))))))) }
                                              #f) }) }) })
                        (set! off (+ off (+ (String.length ln) 1)))
                        (set! i (+ i 1)) })
                    (set_item! vw_toc vw_toc_scanned out) }
                  #f) })
        (set! vw_toc_scanned (+ vw_toc_scanned 1))
        (set! done (+ done 1)) })
    (== vw_toc_scanned total)))

(define toc_done (lambda () (>= vw_toc_scanned (length vw_segs))))
(define toc_scanned_count (lambda () vw_toc_scanned))

;; 全量目录（按段顺序拼接）：元素 = (层级 段号 种类 数值 文本)
(define toc_list
  (lambda ()
    (define out (new_list))
    (define si 0)
    (define es (new_list))
    (define k 0)
    (define e #f)
    (while (< si (length vw_toc))
      { (set! es (get_item vw_toc si))
        (set! k 0)
        (while (< k (length es))
          { (set! e (get_item es k))
            (push out (cons (get_item e 0)
                        (cons si (cons (get_item e 1)
                          (cons (get_item e 2) (cons (get_item e 3) (cdr '(0))))))))
            (set! k (+ k 1)) })
        (set! si (+ si 1)) })
    out))

(define toc_count
  (lambda ()
    (define out 0)
    (define si 0)
    (while (< si (length vw_toc))
      { (set! out (+ out (length (get_item vw_toc si))))
        (set! si (+ si 1)) })
    out))

;; 跳到第 k 条：先设估算锚点，布局该段后用 find_heading_y 精修
(define toc_jump
  (lambda (k)
    (define es (toc_list))
    (define e #f)
    (define seg 0)
    (define kind 0)
    (define val 0)
    (define txt "")
    (define h 0)
    (define chars 0)
    (define dy 0)
    (define fy 0)
    (if (or (< k 0) (>= k (length es))) #f
        { (set! e (get_item es k))
          (set! seg (get_item e 1))
          (set! kind (get_item e 2))
          (set! val (get_item e 3))
          (set! txt (get_item e 4))
          (if (== kind 0)
              (set_anchor seg val)
              { (set! chars (max2 1 (get_item (get_item vw_segs seg) 3)))
                (set! dy (Math.floor (/ (* (height seg) val) chars)))
                (set_anchor seg dy) })
          (ensure seg)
          (set! fy (find_heading_y seg txt))
          (if (>= fy 0) (set_anchor seg (max2 0 (- fy 6))) #f)
          #t })))

;; ---- 诊断 ---------------------------------------------------------------

;; 已布局段的块数/显示行数合计（MDSTAT 诊断用；单段文档即整篇的真实值）
(define blocks_total
  (lambda ()
    (define s 0)
    (define k 0)
    (while (< k (length vw_nblocks))
      { (set! s (+ s (get_item vw_nblocks k)))
        (set! k (+ k 1)) })
    s))

(define lines_total
  (lambda ()
    (define s 0)
    (define k 0)
    (while (< k (length vw_nlines))
      { (set! s (+ s (get_item vw_nlines k)))
        (set! k (+ k 1)) })
    s))

(define diag
  (lambda ()
    ;; (段数 已布局段数 缓存段数 缓存行数 校准系数 已索引字节 总字节)
    (define laid 0)
    (define k 0)
    (while (< k (length vw_h))
      { (if (> (get_item vw_h k) 0) (set! laid (+ laid 1)) #f)
        (set! k (+ k 1)) })
    (cons (length vw_segs) (cons laid (cons (length vw_c_seg)
      (cons (cache_len) (cons vw_cal (cons vw_bytes_done (cons (MDST.size) (cdr '(0)))))) )))))
