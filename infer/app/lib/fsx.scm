;; ============================================================================
;; fsx.scm —— 文件系统小工具库（路径 / 名字排序 / 文本宽度与裁剪 / 尺寸格式化）
;;
;; 为什么单独成库：
;;   「文件管理器」（app/filemgr.scm）与「文件选择器」（app/lib/filepicker.scm，
;;   供 Markdown 阅读器等 App 复用）需要同一套"看起来必须一模一样"的基础动作：
;;   路径拼接与上级目录、**排序（目录在前 + 名字升序）**、按像素宽裁剪文本、可读尺寸。
;;   写两份迟早会漂移，故集中在这里。两边都以 (import FSX "app/lib/fsx.scm") 引入。
;;
;; 为什么自己排序（不用 Fs.list 的顺序）：
;;   (Fs.list dir) 的顺序**各平台不保证**——Linux 侧为了回归稳定按 strcmp 排过，
;;   但 ESP32 的 SD 枚举是 readdir/卷顺序。需求要求"目录在前 + 按字母升序"，因此排序
;;   放在 Scheme 层（名字按 Unicode 码点字典序，ASCII 上与字节序一致），三平台统一。
;;   做法：插入排序（目录条目通常几十个，O(n²) 无感；`cons` 是整表复制，插入排序
;;   只做"push 一次 + 右移"反而比用 cons 建表更省）。
;;
;; 路径约定（与 (native Fs) 一致）：相对业务资源根；根目录写作 "."（Fs 的 fs_resolve
;; 明确拒绝空串，故 "." 是全平台通用的"根"写法，见 doc/AGENTS.md 2026-11-02 条）。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/list/map/append；
;; '() 是模块级静态槽（空表一律 (cdr '(0)) 现造）；循环里不用下标末值当 break。
;; ============================================================================

(native String)
(native Math)

;; ---- 造表小工具 ----------------------------------------------------------

(define new_list (lambda () (cdr '(0))))
(define mk2 (lambda (a b) (cons a (cons b (cdr '(0))))))

;; ---- 路径 ----------------------------------------------------------------

;; (join "." "a") => "a"；(join "doc" "a") => "doc/a"
(define join
  (lambda (dir name)
    (if (or (String.equals dir ".") (== (String.length dir) 0))
        name
        (String.concat dir (String.concat "/" name)))))

;; 最后一个 '/' 的下标；没有则 -1
(define last_slash
  (lambda (s)
    (define n (String.length s))
    (define i (- n 1))
    (define cut -1)
    (while (>= i 0)
      { (if (and (< cut 0) (== (String.charCodeAt i s) 47)) (set! cut i) #f)
        (set! i (- i 1)) })
    cut))

;; "a/b/c.txt" => "c.txt"；"." => "."
(define leaf
  (lambda (s)
    (define cut (last_slash s))
    (if (or (< cut 0) (== (String.length s) 0))
        s
        (String.slice s (+ cut 1) (String.length s)))))

;; "a/b" => "a"；"a" => "."；"." => "."（不允许越出资源根）
(define parent
  (lambda (s)
    (define cut (last_slash s))
    (if (< cut 1) "." (String.slice s 0 cut))))

;; ---- 名字比较与排序 ------------------------------------------------------

;; 按码点字典序比较：-1 = a<b，0 = 相等，1 = a>b
(define cmp_name
  (lambda (a b)
    (define la (String.length a))
    (define lb (String.length b))
    (define n (if (< la lb) la lb))
    (define i 0)
    (define ca 0)
    (define cb 0)
    (define r 0)
    (while (and (< i n) (== r 0))
      { (set! ca (String.charCodeAt i a))
        (set! cb (String.charCodeAt i b))
        (if (< ca cb) (set! r -1) #f)
        (if (> ca cb) (set! r 1) #f)
        (set! i (+ i 1)) })
    (if (== r 0)
        (if (< la lb) -1 (if (> la lb) 1 0))
        r)))

;; 条目比较：**目录永远排在文件前面**，同一类内按名字升序（码点字典序）。
;; 条目形如 (mk2 名字 是否目录) 或 (mk3 名字 是否目录 字节数)——只读前两项，
;; 因此"是否目录"必须放在第二项且取 1/0（见 sort_entries 的约定）。
(define cmp_entry
  (lambda (a b)
    (define da (get_item a 1))
    (define db (get_item b 1))
    (define r 0)
    (if (and (== da 1) (== db 0)) (set! r -1) #f)
    (if (and (== da 0) (== db 1)) (set! r 1) #f)
    (if (== r 0) (set! r (cmp_name (get_item a 0) (get_item b 0))) #f)
    r))

;; 条目表排序（本库唯一的排序入口）：**目录在前、文件在后，各自按名字升序**。
;; 为什么排"条目"而不是排"名字表"：调用方要同时保留「是否目录」「字节数」等附带信息，
;; 只排名字会让附属信息与名字错位；把它们绑在一个小表里一起搬最省事。
;; 约定：元素 0 = 名字，元素 1 = 是否目录（1/0），元素 2（可选）= 字节数。
;; 算法是插入排序：`cons` 是整表复制，而"push 一次 + 右移"在原地完成，几十条目无感。
(define insert_sorted_entry
  (lambda (out item)
    (define n (length out))
    (define j 0)
    (define k 0)
    (while (and (< j n) (< (cmp_entry (get_item out j) item) 0))
      (set! j (+ j 1)))
    (push out item)
    (set! k n)
    (while (> k j)
      { (set_item! out k (get_item out (- k 1)))
        (set! k (- k 1)) })
    (set_item! out j item)
    #t))

(define sort_entries
  (lambda (lst)
    (define out (new_list))
    (define n (length lst))
    (define i 0)
    (while (< i n)
      { (insert_sorted_entry out (get_item lst i))
        (set! i (+ i 1)) })
    out))

;; ---- 字符串小工具 --------------------------------------------------------

(define lower_cp
  (lambda (c)
    (if (and (>= c 65) (<= c 90)) (+ c 32) c)))

;; 大小写不敏感的"以 suf 结尾"（扩展名判断用）；返回 1/0
(define ends_with
  (lambda (s suf)
    (define ls (String.length s))
    (define lf (String.length suf))
    (define i 0)
    (define ok 1)
    (if (< ls lf)
        0
        { (while (< i lf)
            { (if (== (lower_cp (String.charCodeAt (+ (- ls lf) i) s))
                      (lower_cp (String.charCodeAt i suf)))
                  #f
                  (set! ok 0))
              (set! i (+ i 1)) })
          ok })))

;; 扩展名（含点）；没有则空串
(define ext_of
  (lambda (s)
    (define n (String.length s))
    (define i (- n 1))
    (define dot -1)
    (while (>= i 1)
      { (if (and (< dot 0) (== (String.charCodeAt i s) 46)) (set! dot i) #f)
        (set! i (- i 1)) })
    (if (< dot 0) "" (String.slice s dot n))))

;; 去掉扩展名（".md"/".txt" 等）；无点则原样返回
(define strip_ext
  (lambda (s)
    (define e (ext_of s))
    (if (== (String.length e) 0) s (String.slice s 0 (- (String.length s) (String.length e))))))

;; 扩展名过滤：exts 形如 ""（全部）或 "md" / "md,txt"（不含点，大小写不敏感）
(define ext_match
  (lambda (name exts)
    (define n (String.length exts))
    (define i 0)
    (define start 0)
    (define one "")
    (define r 0)
    (if (or (== n 0) (String.equals exts "*"))
        1
        { (while (<= i n)
            { (if (or (== i n) (== (String.charCodeAt i exts) 44))
                  { (set! one (String.slice exts start i))
                    (if (and (> (String.length one) 0)
                             (== (ends_with name (String.concat "." one)) 1))
                        (set! r 1)
                        #f)
                    (set! start (+ i 1)) }
                  #f)
              (set! i (+ i 1)) })
          r })))

;; ---- 文本宽度（12px 字模：半角 7px / 全角 12px，与 md_layout.scm 同口径） ----

(define W_HALF 7)
(define W_FULL 12)
(define CJK_BASE 11904)   ;; 0x2E80

(define char_w
  (lambda (cp)
    (if (>= cp CJK_BASE) W_FULL W_HALF)))

(define text_w
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (while (< i n)
      { (set! w (+ w (char_w (String.charCodeAt i s))))
        (set! i (+ i 1)) })
    w))

;; 按像素宽裁剪，超出部分用 … 收尾
(define clip_text
  (lambda (s maxw)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (define cut -1)
    (define stop 0)
    (while (and (< i n) (== stop 0))
      { (if (> (+ w (char_w (String.charCodeAt i s))) (- maxw W_HALF))
            { (set! cut i) (set! stop 1) }
            { (set! w (+ w (char_w (String.charCodeAt i s))))
              (set! i (+ i 1)) }) })
    (if (< cut 0)
        s
        (if (== cut 0)
            ""
            (String.concat (String.slice s 0 cut) "…")))))

;; ---- 尺寸格式化 ----------------------------------------------------------

(define human_size
  (lambda (b)
    (if (< b 1024)
        (String.concat (String.atom_to_string b) " B")
        (if (< b (* 1024 1024))
            (String.concat (String.atom_to_string (Math.to_fixed (/ b 1024.0) 1)) " KB")
            (String.concat (String.atom_to_string (Math.to_fixed (/ b (* 1024 1024)) 1)) " MB")))))