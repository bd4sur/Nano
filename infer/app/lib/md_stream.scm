;; ============================================================================
;; md_stream.scm —— Markdown **分段流式读取**（大文件阅读的底座）
;;
;; 为什么要有本模块（实测数据见 doc/AGENTS.md）：
;;   · 灵机 App 的解释器池固定（WASM 目标 32MB，heap/VM 各 16MB），而本阅读器原先
;;     "整篇解析 + 整篇排版"：实测每 KB 源码要 ~40KB 堆（块树）+ ~26KB 堆（显示行，
;;     折合 ~1.1KB/显示行）+ ~15ms 时间；1MB 文档需要 >60MB 堆与 ~15s 同步时间。
;;   · 因此把文档切成"段"（segment，默认 4KB 源码），每段独立解析/排版/缓存，
;;     堆上任何时刻只保留窗口内的若干段（见 app/lib/md_view.scm）。
;;
;; 分工：
;;   · 本模块只做**字节流 → 段（行表 + 记账）**：(Fs.read_range) 分块读 → 按行切分 →
;;     在"安全切点"处成段，并记录字节偏移/行数/码点数，供上层随机复现同一段。
;;   · **不做** Markdown 语义（那是 md_parse.scm），**不做**排版（md_layout.scm）。
;;
;; 安全切点（正确性的核心）：
;;   只在"围栏代码块之外的**空行之后**"切——本解析器的段落/列表/表格/引用块都在
;;   空行处收尾（md_parse.scm 的 h_para / h_list / h_table / h_quote），因此在该处切开
;;   与整篇解析**等价**（tools/e2e/markdown/test_stream.scm 用"各段拼接后与整篇解析
;;   逐块相等"守这条不变量）。
;;   病态输入（超过硬上限 16KB 仍无空行：巨大代码块/单段长文）在硬上限处强制切，并置续接信息：
;;     · 段尾在围栏内 → 本段尾合成闭合围栏行、下一段首合成同标记开启行，**长度与原开启行
;;       相同**：围栏内容里不可能出现"整行同标记且不短于开启行"（否则原围栏早就闭合了），
;;       因此同长度既安全又与原文档逐行等价（用 +4 之类更长标记会让"真正的结束行"
;;       在后续段里变成内容，实测会多出一行）；
;;     · 段尾在段落/列表/表格内 → 只置续接标志，让排版端**抑制缝上的间隙/内边距**。
;;
;; 段的表示（列表，元素固定 9 个）：
;;   (字节起点 字节长度 源码行数 码点数 首续接 尾续接 围栏字符 合成开启长度 合成闭合长度)
;;   · 首续接/尾续接：1 = 抑制该侧间隙（缝上不额外留白）
;;   · 合成开启长度 > 0 → 段首插一行围栏开启（"接上"上一段被迫切断的代码块）
;;   · 合成闭合长度 > 0 → 段尾插一行围栏闭合
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/list/append/map；
;;   空表一律 (cdr '(0)) 现造；push 返回 #undefined；while 循环体不是独立词法区域，
;;   故所有循环变量都在函数体顶层 define。
;;
;; 单例：模块级状态（同一时刻只流式读一个文档；App 是单前台模型）。
;; ============================================================================

(native Fs)
(native String)
(native Math)

(import MD "app/lib/md_parse.scm")

;; ---- 默认参数（App 可在 open 时覆盖）-------------------------------------

(define DEF_CHUNK      8192)    ;; 单次读块（字节）
(define DEF_SEG_TARGET 4096)    ;; 目标段大小（字节）：达到且落在安全切点即切
(define DEF_SEG_HARD  16384)    ;; 硬上限（字节）：再不安全也要切，保证内存有界
                                    ;; （病态输入才触发；16KB 让单段瞬时占用 ≈ 4MB，
                                    ;;  为 16MB heap 区留足 GC 余量）
(define DEF_LINE_MAX   4096)    ;; 单行码点上限（保留参数：切段按行，不拆行）

;; ---- 工具 ----------------------------------------------------------------

(define new_list (lambda () (cdr '(0))))

;; UTF-8 字节长度（1/2/3/4 字节；与解释器编解码一致，用于字节偏移记账）
(define utf8_len
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define cp 0)
    (define w 0)
    (while (< i n)
      { (set! cp (String.charCodeAt i s))
        (if (< cp 128) (set! w (+ w 1))
            (if (< cp 2048) (set! w (+ w 2))
                (if (< cp 65536) (set! w (+ w 3)) (set! w (+ w 4)))))
        (set! i (+ i 1)) })
    w))

;; 取前 k 个元素（造新表）
(define take_k
  (lambda (lst k)
    (define out (new_list))
    (define i 0)
    (define m (length lst))
    (while (and (< i k) (< i m))
      { (push out (get_item lst i))
        (set! i (+ i 1)) })
    out))

;; 去掉前 k 个元素（造新表）
(define drop_k
  (lambda (lst k)
    (define out (new_list))
    (define i k)
    (define m (length lst))
    (while (< i m)
      { (push out (get_item lst i))
        (set! i (+ i 1)) })
    out))

;; 合计码点数
(define fold_chars
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s 0)
    (while (< i n)
      { (set! s (+ s (String.length (get_item lst i))))
        (set! i (+ i 1)) })
    s))

;; 合计 UTF-8 字节数（含行尾换行）
(define fold_bytes
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s 0)
    (while (< i n)
      { (set! s (+ s (+ (utf8_len (get_item lst i)) 1)))
        (set! i (+ i 1)) })
    s))

;; n 个字符 ch 组成的围栏行（合成行用）
(define mk_fence
  (lambda (ch n)
    (define out "")
    (define i 0)
    (while (< i n)
      { (set! out (String.concat out (String.fromCharCode ch)))
        (set! i (+ i 1)) })
    out))

;; 段记录构造（9 元）
(define mk_seg
  (lambda (start bytes slines chars cb ca fch olen clen)
    (cons start (cons bytes (cons slines (cons chars
      (cons cb (cons ca (cons fch (cons olen (cons clen (cdr '(0)))))))))))))

;; ---- 流状态（模块级单例）-------------------------------------------------

(define sr_path "")
(define sr_size 0)
(define sr_off 0)            ;; 下一次读取的字节偏移
(define sr_line_off 0)       ;; **sr_lines 第一行**的字节偏移（段起点记账用）
(define sr_tail "")          ;; 上一次读块末尾的"不完整行"
(define sr_lines (new_list)) ;; 已切好、尚未成段的行缓冲
(define sr_eof #f)
(define sr_chunk DEF_CHUNK)
(define sr_seg_target DEF_SEG_TARGET)
(define sr_seg_hard DEF_SEG_HARD)
(define sr_line_max DEF_LINE_MAX)
;; 诊断计数
(define sr_blocks 0)
(define sr_reads 0)
(define sr_bytes 0)
;; 续接状态（上一段在围栏内被强制切断时置位，交给下一段）
(define sr_next_cb 0)
(define sr_next_fch 96)
(define sr_next_olen 0)
;; 切段扫描的暂存结果（避免多值返回）
(define plan_cut 0)
(define plan_chars 0)
(define plan_bytes 0)
(define plan_fence #f)
;; at()/tail() 期间保存的前向流状态
(define save_off 0)
(define save_line_off 0)
(define save_tail "")
(define save_lines (new_list))
(define save_eof #f)

;; ---- 打开 / 关闭 ---------------------------------------------------------

(define open
  (lambda (path chunk seg_target seg_hard line_max)
    (set! sr_path path)
    (set! sr_size (Fs.size path))
    (set! sr_off 0)
    (set! sr_line_off 0)
    (set! sr_tail "")
    (set! sr_lines (new_list))
    (set! sr_eof #f)
    (set! sr_chunk (if (undefined? chunk) DEF_CHUNK chunk))
    (set! sr_seg_target (if (undefined? seg_target) DEF_SEG_TARGET seg_target))
    (set! sr_seg_hard (if (undefined? seg_hard) DEF_SEG_HARD seg_hard))
    (set! sr_line_max (if (undefined? line_max) DEF_LINE_MAX line_max))
    (set! sr_blocks 0)
    (set! sr_reads 0)
    (set! sr_bytes 0)
    (set! sr_next_cb 0)
    (set! sr_next_fch 96)
    (set! sr_next_olen 0)
    (set! plan_cut 0)
    (set! plan_fence #f)
    (if (null? sr_size) #f #t)))

(define size (lambda () sr_size))
(define at_eof (lambda () sr_eof))
(define diag (lambda () (cons sr_blocks (cons sr_reads (cons sr_bytes (cdr '(0)))))))

;; ---- 读一块并切行 -------------------------------------------------------

;; 读一块 → 切行：完整行入 sr_lines，末尾不完整行留 sr_tail。
(define fill_once
  (lambda ()
    (define r #f)
    (define text "")
    (define nxt 0)
    (define ls (new_list))
    (define nl 0)
    (define i 0)
    (set! r (Fs.read_range sr_path sr_off sr_chunk))
    (if (null? r)
        (set! sr_eof #t)
        { (set! text (get_item r 0))
          (set! nxt (get_item r 1))
          (set! sr_reads (+ sr_reads 1))
          (set! sr_bytes (+ sr_bytes (String.length text)))
          (set! sr_off nxt)
          (if (== (String.length text) 0)
              { (set! sr_eof #t)
                (if (> (String.length sr_tail) 0)
                    { (push sr_lines sr_tail)
                      (set! sr_tail "") }
                    #f) }
              { (set! ls (MD.split_lines text))
                (set! nl (length ls))
                ;; 上一块残留的半行并到本块第一行（**只拼小串**：避免为每个读块
                ;; 重新拼接 8KB 大串带来的分配抖动与堆碎片）
                (if (> (String.length sr_tail) 0)
                    (if (> nl 0)
                        { (set_item! ls 0 (String.concat sr_tail (get_item ls 0)))
                          (set! sr_tail "") }
                        #f)
                    #f)
                (if (>= nxt sr_size)
                    { ;; 读到文件尾：最后一个元素也是完整行。
                      ;; 但文件以换行结尾时 split_lines 会多出一个空尾行——它不是文件的
                      ;; 一行，丢弃它（否则会派生出"起点 == 文件大小"的幽灵段，
                      ;; 该段无法再用 at() 复现）。
                      (set! i 0)
                      (while (< i nl)
                        { (if (and (>= nxt sr_size) (== i (- nl 1))
                                   (String.equals (get_item ls i) ""))
                              #f
                              (push sr_lines (get_item ls i)))
                          (set! i (+ i 1)) })
                      (set! sr_eof #t) }
                    { ;; 最后一行的换行符还没读到：留作 sr_tail
                      (set! i 0)
                      (while (< i (- nl 1))
                        { (push sr_lines (get_item ls i))
                          (set! i (+ i 1)) })
                      (if (> nl 0) (set! sr_tail (get_item ls (- nl 1))) #f) }) }) })
    #t))

(define fill_upto
  (lambda (k)
    (while (and (< (length sr_lines) k) (not sr_eof))
      (fill_once))
    #t))

;; ---- 扫描切点 -----------------------------------------------------------

;; 扫描 sr_lines 决定本段行数；结果放 plan_*。
;;   plan_cut = 0 表示"还需更多行"；plan_fence = 切点处仍打开的围栏信息（#f = 不在围栏内）
(define cut_scan
  (lambda ()
    (define n (length sr_lines))
    (define i 0)
    (define fence #f)
    (define info #f)
    (define line "")
    (define chars 0)
    (define bytes 0)
    (define stop 0)
    (set! plan_cut 0)
    (set! plan_chars 0)
    (set! plan_bytes 0)
    (set! plan_fence #f)
    (while (and (< i n) (== stop 0))
      { (set! line (get_item sr_lines i))
        ;; 围栏状态推进（与 md_parse 的 h_fence 判定完全一致）
        (if fence
            { (if (MD.fence_close line fence) (set! fence #f) #f) }
            { (set! info (MD.fence_open line))
              (if info (set! fence info) #f) })
        (set! chars (+ chars (String.length line)))
        (set! bytes (+ bytes (+ (utf8_len line) 1)))
        (if (and (>= bytes sr_seg_target) (== fence #f) (MD.is_blank line))
            { (set! plan_cut (+ i 1)) (set! stop 1) }
            (if (>= bytes sr_seg_hard)
                { (set! plan_cut (+ i 1)) (set! stop 1) }
                (set! i (+ i 1))))
        #f })
    (if (== plan_cut 0)
        (if sr_eof
            { (set! plan_cut n)
              (set! plan_chars chars)
              (set! plan_bytes bytes)
              (set! plan_fence fence) }
            #f)
        { (set! plan_chars chars)
          (set! plan_bytes bytes)
          (set! plan_fence fence) })
    #t))

;; ---- 成段 ---------------------------------------------------------------

;; 把已扫出的 plan_cut 行变成一段：返回 (段 行表) 或 #f（没有更多内容）
(define emit_segment
  (lambda ()
    (define real (take_k sr_lines plan_cut))
    (define out (new_list))
    (define n 0)
    (define i 0)
    (define cb 0)
    (define ca 0)
    (define fch sr_next_fch)
    (define olen 0)
    (define clen 0)
    (define lastline "")
    (define start 0)
    (set! start sr_line_off)
    (set! n (length real))
    (set! lastline (if (> n 0) (get_item real (- n 1)) ""))
    ;; 前置：上一段被迫在围栏内切断 → 段首插合成开启行
    (if (> sr_next_olen 0)
        { (set! cb 1)
          (set! olen sr_next_olen)
          (push out (mk_fence fch olen)) }
        (set! cb sr_next_cb))
    (set! i 0)
    (while (< i plan_cut)
      { (push out (get_item real i))
        (set! i (+ i 1)) })
    ;; 后置：本段切点仍在围栏内 → 插合成闭合行；否则若末尾非空行 → 段落续接
    (if plan_fence
        { (set! ca 1)
          (set! clen (get_item plan_fence 1))
          (push out (mk_fence (get_item plan_fence 0) clen))
          (set! sr_next_cb 1)
          (set! sr_next_fch (get_item plan_fence 0))
          (set! sr_next_olen (get_item plan_fence 1)) }
        { (if (MD.is_blank lastline)
              { (set! ca 0) (set! sr_next_cb 0) }
              { (set! ca 1) (set! sr_next_cb 1) })
          (set! sr_next_fch 96)
          (set! sr_next_olen 0) })
    ;; 记账：消费掉这 plan_cut 行
    (set! sr_lines (drop_k sr_lines plan_cut))
    (set! sr_line_off (+ sr_line_off (fold_bytes real)))
    (set! sr_blocks (+ sr_blocks 1))
    (cons (mk_seg start plan_bytes plan_cut plan_chars cb ca fch olen clen)
          (cons out (cdr '(0))))))

;; ---- 前向：取下一段 -----------------------------------------------------

(define next
  (lambda ()
    (define go 1)
    (define res #f)
    (while (== go 1)
      { (fill_upto 1)
        (cut_scan)
        (if (> plan_cut 0)
            { (set! go 0)
              (set! res (emit_segment)) }
            (if sr_eof
                { (set! go 0)
                  (set! res #f) }
                (fill_once))) })
    res))

;; ---- 随机访问：按段记录复现同一段的行表 --------------------------------

;; (at 字节起点 源码行数 首续接 尾续接 围栏字符 合成开启长度 合成闭合长度) → 行表 | #f
;; 段记录来自 emit_segment/tail；这里按"行数"取，与上次切分天然一致。
(define at
  (lambda (start slines cb ca fch olen clen)
    (define real (new_list))
    (define out (new_list))
    (define i 0)
    ;; 保存前向流状态（本函数会临时占用读流）
    (set! save_off sr_off)
    (set! save_line_off sr_line_off)
    (set! save_tail sr_tail)
    (set! save_lines sr_lines)
    (set! save_eof sr_eof)
    (set! sr_off start)
    (set! sr_line_off start)
    (set! sr_tail "")
    (set! sr_lines (new_list))
    (set! sr_eof #f)
    (fill_upto slines)
    (if (< (length sr_lines) slines)
        { (set! out #f) }
        { (set! real (take_k sr_lines slines))
          (if (> olen 0) (push out (mk_fence fch olen)) #f)
          (set! i 0)
          (while (< i slines)
            { (push out (get_item real i))
              (set! i (+ i 1)) })
          (if (> clen 0) (push out (mk_fence fch clen)) #f) })
    ;; 恢复前向流状态
    (set! sr_off save_off)
    (set! sr_line_off save_line_off)
    (set! sr_tail save_tail)
    (set! sr_lines save_lines)
    (set! sr_eof save_eof)
    out))

;; ---- 从文件尾倒推一段（End 键用：无需扫描全文）--------------------------

;; (tail 字节数) → (段 行表) | #f
;; 先探读一块拿到"真实文本起点"（read_range 可能把起点前移到码点边界），再用 at()
;; 按行数读——这样 tail 得到的行表与 at() 复现的结果**天然一致**。
(define tail
  (lambda (maxbytes)
    (define start0 0)
    (define r #f)
    (define text "")
    (define consumed 0)
    (define skip 0)
    (define ls (new_list))
    (define nl 0)
    (define start 0)
    (define nlines 0)
    (define lines #f)
    (set! start0 (if (> sr_size maxbytes) (- sr_size maxbytes) 0))
    (set! r (Fs.read_range sr_path start0 maxbytes))
    (if (null? r)
        #f
        { (set! text (get_item r 0))
          (set! consumed (- (get_item r 1) start0))
          (set! skip (- consumed (utf8_len text)))
          (set! ls (MD.split_lines text))
          (set! nl (length ls))
          (if (< nl 2)
              #f
              { (set! start (+ (+ start0 skip) (+ (utf8_len (get_item ls 0)) 1)))
                (set! nlines (- nl 1))
                ;; 与 at() 同口径：文件以换行结尾时不计那个空尾行
                (if (and (> nlines 0) (String.equals (get_item ls (- nl 1)) ""))
                    (set! nlines (- nlines 1))
                    #f)
                (set! lines (at start nlines 0 0 96 0 0))
                (if lines
                    (cons (mk_seg start (- sr_size start) nlines (fold_chars lines)
                                  0 0 96 0 0)
                          (cons lines (cdr '(0))))
                    #f) }) })))
