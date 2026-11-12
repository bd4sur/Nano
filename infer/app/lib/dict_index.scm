;; ============================================================================
;; dict_index.scm —— 词典数据层（稀疏块索引 + 前缀查询 + 记录解析）
;;
;; 为什么单独成模块：词典 App（app/dict.scm）里"界面"与"数据"两件事都很重，
;; 混在一个文件里既难读也无法单独验证——数据层是纯逻辑（不碰 Gfx/Touch），
;; 抽出来就能在无图形宿主上直接跑、与 C 版 ui_dict.c / Python 真值逐条比对
;;（见 tools/e2e/native/dict_truth.scm 与 dict_truth.py）。
;;
;; 索引策略（与 C 版的差异及原因，另见 app/dict.scm 头注释）：
;;   C 版把 8MB 的 CSV 全量扫描成 2.4MB 的二进制索引（ecdict.idx，布局 == RAM 布局，
;;   加载零修复），运行时做 O(log n) 二分。灵机侧做不到同样的事：
;;     · (native Fs) 只给**文本**文件 API（read/read_range 都按 UTF-8 解码，且保证
;;       返回值以"整码点"开头），二进制 .idx 无法逐字节可靠解析；
;;     · 13 万条索引项在 Scheme 堆里是几十 MB 的对象图，任何内存池都装不下。
;;   因此这里用**稀疏块索引**：按 IDX_BLOCK 分块，每块只记"该块边界之后第一条
;;   完整记录"的（词条, 记录文件偏移），8MB 词典约 500 项、约 60KB 堆。
;;   查询 = 先在 500 项里二分定位块，再从该块的记录起点顺读做前缀枚举：
;;   仍是"索引 + 前缀查询"同一套技术，索引体积是 C 版的 1/40，且**不写盘**。
;;
;;   索引是**定向探测**建立的（约 500 次 1KB 小窗口读），不做 8MB 全量扫描，
;;   故冷启动是毫秒级（浏览器 MEMFS）到数百毫秒（SD 卡）——代价是每次进词典
;;   重建一次（不落盘，不需要可写存储；词典文件更新后也无需重建缓存）。
;;
;; 文件格式（ecdict.csv）：word,phonetic,definition
;;   · 按词条升序（大小写不敏感，实测 12.9 万条零逆序）—— 前缀枚举依赖这个前提；
;;   · 字段可加引号，引号字段内允许逗号，"" 转义为 "（引号内的**换行**本实现
;;     不支持：稀疏索引按"\n 分隔一行一条记录"定位记录起点）；
;;   · 首行可能是 CSV 表头（word,phonetic,definition），跳过；可能带 UTF-8 BOM。
;;
;; 接口（全部值型；失败按软失败约定返回 #null / -1）：
;;   (DI.open)        : Number   —— 打开词典并建立稀疏索引；0 成功，-1 缺文件/空文件
;;   (DI.file_size)   : Number   —— CSV 字节数（-1 表示未打开）
;;   (DI.match_max)   : Number   —— 单次查询的候选上限（64，与 C 版一致）
;;   (DI.query pfx)   : List     —— 候选表 ((词条 记录偏移) ...)，最多 match_max 条；
;;                                  空前缀返回空表（调用方自己决定怎么提示）
;;   (DI.entry off)   : List     —— 该偏移处记录的 (词条 音标 释义) 三字段
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数、无一元取负、无 let/map/
;;   append/apply、define 无过程简写、块内不 define、'() 是模块级静态槽（空表一律
;;   用 (cdr '(0)) 现造）、/ 与 mod 恒返回浮点（下标一律 Math.floor）。
;;   易错点：辅助函数必须用**自己 lambda 体内的局部变量**，不能共用模块级临时量
;;   （曾被 rec_word 踩掉 scan 循环的临时量，症状是"前缀查询永远零匹配"）。
;; ============================================================================

(native Fs)
(native String)
(native Math)

(define CSV "dict/ecdict.csv")   ; 词典 CSV（相对业务资源根；含 ".." 的路径被 Fs 拒绝）
(define IDX_BLOCK 16384)         ; 稀疏索引粒度（字节）：8MB / 16KB ≈ 500 项
(define PROBE_W 1024)            ; 探测窗口（最长记录 150 字节，1KB 足够跨一整条记录）
(define SCAN_CHUNK 8192)         ; 前缀枚举的顺读块（8KB ≈ 130 条记录，堆开销有界）
(define MATCH_MAX 64)            ; 候选上限（与 C 版 UI_DICT_MATCH_MAX 一致）

(define file_bytes -1)           ; CSV 字节数（-1 = 未打开/缺文件）
(define idx_words (cdr '(0)))    ; 稀疏索引：块首记录词条（升序）
(define idx_offs (cdr '(0)))     ; 稀疏索引：块首记录的文件偏移（与上表等长）

;; ============================================================================
;; 工具
;; ============================================================================

(define new_list (lambda () (cdr '(0))))

;; ASCII 小写化（词条是英文，不处理多字节大小写，与 C 版 ci_lower 一致）
(define lower_ch
  (lambda (c)
    (if (and (>= c 65) (<= c 90)) (+ c 32) c)))

;; UTF-8 码点字节数（稀疏索引与顺读都要按字节记偏移；灵机字符串按码点索引）
(define cbytes
  (lambda (c)
    (if (< c 128) 1 (if (< c 2048) 2 (if (< c 65536) 3 4)))))

;; 字符串前 i 个码点的 UTF-8 字节数
(define blen
  (lambda (s i)
    (define k 0)
    (define b 0)
    (while (< k i) {
      (set! b (+ b (cbytes (String.charCodeAt k s))))
      (set! k (+ k 1))
    })
    b))

;; 从 from 起找首个码点 code 的下标；没有返回 -1
(define find_ch
  (lambda (s from code)
    (define n (String.length s))
    (define i from)
    (define r -1)
    (while (and (< i n) (< r 0)) {
      (if (== (String.charCodeAt i s) code) (set! r i) #f)
      (set! i (+ i 1))
    })
    r))

;; 大小写不敏感比较：a<b → -1，相等 → 0，a>b → 1（与 C 版 ci_cmp 同语义）
(define ci_cmp
  (lambda (a b)
    (define na (String.length a))
    (define nb (String.length b))
    (define i 0)
    (define r 0)
    (define ca 0)
    (define cb 0)
    (while (and (< i na) (< i nb) (== r 0)) {
      (set! ca (lower_ch (String.charCodeAt i a)))
      (set! cb (lower_ch (String.charCodeAt i b)))
      (if (< ca cb) (set! r -1) (if (> ca cb) (set! r 1) #f))
      (set! i (+ i 1))
    })
    (if (not (== r 0)) r
        (if (< i na) 1 (if (< i nb) -1 0)))))

;; a 是否以 b 为前缀（大小写不敏感）：1/0
(define ci_prefix
  (lambda (a b)
    (define nb (String.length b))
    (define na (String.length a))
    (define i 0)
    (define ok 1)
    (if (> nb na) (set! ok 0)
        (while (and (< i nb) (== ok 1)) {
          (if (== (lower_ch (String.charCodeAt i a))
                  (lower_ch (String.charCodeAt i b)))
              #f
              (set! ok 0))
          (set! i (+ i 1))
        }))
    ok))

;; 记录（CSV 一行）→ 词条字段（字段 0）：引号字段去掉引号并还原 "" 转义
(define rec_word
  (lambda (rec)
    (define n (String.length rec))
    (define i 0)
    (define hit 0)
    (define c 0)
    (define out "")
    (if (and (> n 0) (== (String.charCodeAt 0 rec) 34))
        ;; 引号字段（词条本身含逗号的情形，如 "arbalest,arbalist"）
        {
          (set! i 1)
          (set! hit 0)
          (while (and (< i n) (== hit 0)) {
            (set! c (String.charCodeAt i rec))
            (if (== c 34)
                (if (and (< (+ i 1) n) (== (String.charCodeAt (+ i 1) rec) 34))
                    { (set! out (String.concat out "\""))
                      (set! i (+ i 2)) }
                    (set! hit 1))
                { (set! out (String.concat out (String.charAt rec i)))
                  (set! i (+ i 1)) })
          })
          out
        }
        ;; 普通字段：截到首个逗号
        {
          (set! i 0)
          (set! hit 0)
          (while (and (< i n) (== hit 0)) {
            (if (== (String.charCodeAt i rec) 44) (set! hit 1) #f)
            (set! i (+ i 1))
          })
          (if (== hit 1) (String.slice rec 0 (- i 1)) (String.slice rec 0 n))
        })))

;; CSV 记录 → 三字段表 (词条 音标 释义)。状态机与 C 版 csv_scan_feed 同语义：
;; 引号字段内允许逗号/换行，"" 转义为 "，引号后的杂散字符忽略。
(define parse_rec
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define q 0)
    (define f 0)
    (define cur "")
    (define a "")
    (define b "")
    (define d "")
    (define fin 0)
    (define c 0)
    (while (and (< i n) (== fin 0)) {
      (set! c (String.charCodeAt i s))
      (if (== q 1)
          (if (== c 34)
              (if (and (< (+ i 1) n) (== (String.charCodeAt (+ i 1) s) 34))
                  { (set! cur (String.concat cur "\""))
                    (set! i (+ i 1)) }
                  (set! q 0))
              (set! cur (String.concat cur (String.charAt s i))))
          (if (and (== c 34) (== (String.length cur) 0))
              (set! q 1)
              (if (== c 44)
                  {
                    (if (== f 0) (set! a cur) (if (== f 1) (set! b cur) (set! d cur)))
                    (set! f (+ f 1))
                    (set! cur "")
                  }
                  (if (or (== c 10) (== c 13))
                      (set! fin 1)
                      (set! cur (String.concat cur (String.charAt s i)))))))
      (set! i (+ i 1))
    })
    (if (== f 0) (set! a cur) (if (== f 1) (set! b cur) (set! d cur)))
    (cons a (cons b (cons d (new_list))))))

;; ============================================================================
;; 稀疏块索引：每个表项 = "块边界之后的第一条完整记录"
;;
;; 为什么这样能二分：probe(o) 返回"偏移 o 之后（严格之后）的第一条记录"，
;; 它对 o 单调不减；构建时统一用"跳过窗口内第一条（可能是半条）记录"的规则，
;; 得到的每个偏移都是货真价实的记录起点，枚举时从它开始顺读即可，不会漏记录。
;; ============================================================================

;; 探测 o 之后的第一条完整记录：成功返回 (词条字段 记录偏移)，失败返回 #null
(define probe
  (lambda (o)
    (define r1 (Fs.read_range CSV o 4))   ; 只为量 o 处的"续字节跳过量"
    (define r #null)
    (define t "")
    (define t1 "")
    (define nxt 0)
    (define skip 0)
    (define s 0)
    (define i 0)
    (define i2 0)
    (if (null? r1) #null
        {
          ;; read_range 的返回值以"整码点"开头，可能跳过 o 处 1~3 个续字节；
          ;; next = o + skip + 已返回字节数，故 skip 可由两端差值精确反推。
          ;; 先用 4 字节小读求 skip（只为算几个字节，不扫整窗），再读 1KB 窗口：
          ;; 否则每块都要对整窗做一遍 UTF-8 字节累加，启动期 500 次探测就白花
          ;; 数百毫秒（桌面实测 blen(1KB) ≈ 1.4ms）。
          (set! t1 (get_item r1 0))
          (set! skip (- (- (get_item r1 1) o) (blen t1 (String.length t1))))
          (set! s (+ o skip))
          (set! r (Fs.read_range CSV o PROBE_W))
          (if (null? r) #null
              {
                (set! t (get_item r 0))
                (set! i (find_ch t 0 10))
                (if (< i 0) #null
                    {
                      (set! i2 (find_ch t (+ i 1) 10))
                      (if (< i2 0) #null
                          (cons (rec_word (String.slice t (+ i 1) i2))
                                (cons (+ (+ s (blen t i)) 1) (new_list))))
                    })
              })
        })))

;; 打开词典并建立稀疏索引。返回 0 成功，-1 缺文件/空文件。
(define open
  (lambda ()
    (define r #null)
    (define t "")
    (define s0 0)
    (define i 0)
    (define i2 0)
    (define w0 "")
    (define k 1)
    (define pr #null)
    (set! idx_words (new_list))
    (set! idx_offs (new_list))
    (set! file_bytes -1)
    (set! r (Fs.read_range CSV 0 PROBE_W))
    (if (null? r) -1
        {
          (set! file_bytes (Fs.size CSV))
          (set! t (get_item r 0))
          ;; 首条记录：跳过 UTF-8 BOM（EF BB BF = U+FEFF 一个码点 / 3 字节）；
          ;; CSV 表头（word,phonetic,definition）不是词条，跳过它（与 C 版同规则）。
          (set! s0 0)
          (if (and (> (String.length t) 0) (== (String.charCodeAt 0 t) 65279))
              (set! s0 1))
          (set! i (find_ch t s0 10))
          (if (< i 0) (set! i (String.length t)))
          (set! w0 (rec_word (String.slice t s0 i)))
          (if (== (ci_cmp w0 "word") 0)
              {
                (set! s0 (+ i 1))
                (set! i2 (find_ch t s0 10))
                (if (< i2 0) (set! i2 (String.length t)))
                (set! w0 (rec_word (String.slice t s0 i2)))
              }
              #f)
          (push idx_words w0)
          (push idx_offs (blen t s0))
          ;; 其余块边界（块内记录由"上一块偏移 → 下一块偏移"的区间覆盖）
          (set! k 1)
          (while (< (* k IDX_BLOCK) file_bytes) {
            (set! pr (probe (* k IDX_BLOCK)))
            (if (null? pr) #f {
              (push idx_words (get_item pr 0))
              (push idx_offs (get_item pr 1))
            })
            (set! k (+ k 1))
          })
          0
        })))

;; 在稀疏索引里二分：返回最后一个 "词条 < p" 的下标；没有则 -1
(define locate
  (lambda (p)
    (define lo 0)
    (define hi (length idx_words))
    (define mid 0)
    (while (< lo hi) {
      ;; 这里可以安全地用位右移折半：两个操作数都是**无符号**整数（字面量 0 与
      ;; (length …) 的返回值），位运算不会报错；Math.floor 会返回浮点，而浮点在
      ;; 32 位 wasm 宿主上是 float32，拿它当偏移会在 >2^24 时丢精度。
      ;; 反过来，**字节偏移**（含减法，恒为有符号整数）不能用位运算——见 narrow。
      (set! mid (Math.bit_lshr (+ lo hi) 1))
      (if (< (ci_cmp (get_item idx_words mid) p) 0)
          (set! lo (+ mid 1))
          (set! hi mid))
    })
    (- lo 1)))

;; ============================================================================
;; 前缀枚举：从记录偏移 off 顺读，收集以 pfx 为前缀的词条（最多 MATCH_MAX 条）
;; ============================================================================

(define scan
  (lambda (off pfx out)
    (define off2 off)
    (define r #null)
    (define t "")
    (define nxt 0)
    (define base 0)
    (define n 0)
    (define i 0)
    (define start 0)
    (define bstart 0)
    (define b 0)
    (define done 0)
    (define w "")
    (define c 0)
    (while (== done 0) {
      (set! r (Fs.read_range CSV off2 SCAN_CHUNK))
      (if (null? r)
          (set! done 1)
          {
            (set! t (get_item r 0))
            (set! nxt (get_item r 1))
            (set! n (String.length t))
            ;; 顺读**不需要**算 skip：本函数的每个起点都是"码点边界"——
            ;; 索引里的记录偏移（\n 之后）、上一块的 next、或块内半条记录的起点，
            ;; 三者都是记录/码点的起始字节，故 read_range 不会跳过任何续字节，
            ;; base 就是 off2。省下来的整窗 UTF-8 字节累加是查询耗时的大头。
            (set! base off2)
            (set! i 0)
            (set! start 0)
            (set! bstart 0)
            (set! b 0)
            (while (and (< i n) (== done 0)) {
              (set! c (String.charCodeAt i t))
              (if (== c 10)
                  {
                    (set! w (rec_word (String.slice t start i)))
                    (if (== (ci_prefix w pfx) 1)
                        {
                          (push out (cons w (cons (+ base bstart) (new_list))))
                          (if (>= (length out) MATCH_MAX) (set! done 1) #f)
                        }
                        ;; 词条已越过前缀区间（排序文件里后面不可能再有匹配）
                        (if (>= (ci_cmp w pfx) 0) (set! done 1) #f))
                    (set! start (+ i 1))
                    (set! bstart (+ b 1))
                  }
                  #f)
              ;; 逐字节偏移累加（cbytes 内联：这个循环每字符都走，函数调用开销可观）
              (if (< c 128)
                  (set! b (+ b 1))
                  (set! b (+ b (if (< c 2048) 2 (if (< c 65536) 3 4)))))
              (set! i (+ i 1))
            })
            (if (== done 0)
                (if (< start n)
                    ;; 块尾是半条记录：从这条记录的起点重读（记录远小于块，不会死循环）
                    (set! off2 (+ base bstart))
                    (if (>= nxt file_bytes) (set! done 1) (set! off2 nxt)))
                #f)
            (if (and (== done 0) (<= off2 off)) (set! done 1) #f)
          })
    })
    out))

;; 块内精细定位：把"从块首顺读到命中"的十几 KB 文本扫描换成十几次小窗口探测。
;;
;; 设 f(o) = "偏移 o 之后第一条记录"的词条，它是 o 的单调不减函数。于是
;; "下界记录"（前缀区间里第一条）的起点 L 满足：位置 o ∈ [P, L-1] 上 f(o) >= pfx
;; （P 为前一条记录的起点），而 o < P 上 f(o) < pfx。因此只需在块内找**最大的
;; f(o) < pfx 的位置 q**，则 probe(q+1) 给出的记录偏移就是 L。
;;
;; 找 q 用"二进制提升"（步长 16384→1 逐级折半），只用加法与比较：
;;   · 不用 (Math.floor (/ ...)) —— 那会产生浮点偏移，而 32 位 wasm 宿主的
;;     am_float_t 是 float32，超过 2^24 的文件偏移会丢精度；
;;   · 不用 (Math.bit_lshr ...) —— 位运算要求两侧都是**无符号**整数，
;;     而记录偏移来自 Fs 的返回值与减法（减法恒返回有符号整数），会报错。
;; 单次探测 ~0.08ms（桌面实测），全程 17 次 ≈ 1.4ms，远低于扫 16KB（~16ms）。
;; 返回下界记录的偏移；块内没有 >= pfx 的记录时返回 -1（调用方退回块首顺读）。
(define NARROW_STEPS '(16384 8192 4096 2048 1024 512 256 128 64 32 16 8 4 2 1))

(define narrow
  (lambda (lo hi pfx)
    (define q lo)
    (define pr #null)
    (define i 0)
    (define ns (length NARROW_STEPS))
    (define cand 0)
    (set! pr (probe lo))
    (if (null? pr) -1
        (if (>= (ci_cmp (get_item pr 0) pfx) 0)
            (get_item pr 1)     ;; 块首之后的第一条记录就是下界
            {
              (set! i 0)
              (while (< i ns) {
                (set! cand (+ q (get_item NARROW_STEPS i)))
                (if (< cand hi)
                    {
                      (set! pr (probe cand))
                      (if (null? pr) #f
                          (if (< (ci_cmp (get_item pr 0) pfx) 0)
                              (set! q cand)
                              #f))
                    }
                    #f)
                (set! i (+ i 1))
              })
              (if (>= (+ q 1) hi) -1
                  {
                    (set! pr (probe (+ q 1)))
                    (if (null? pr) -1 (get_item pr 1))
                  })
            }))))

;; 查询：返回候选表 ((词条 记录偏移) ...)，最多 MATCH_MAX 条（空前缀 → 空表）
(define query
  (lambda (pfx)
    (define out (new_list))
    (define k 0)
    (define st 0)
    (define en 0)
    (define nst -1)
    (if (== (String.length pfx) 0) out
        {
          (set! k (locate pfx))
          (if (>= k 0)
              {
                (set! st (get_item idx_offs k))
                (set! en (if (< (+ k 1) (length idx_offs))
                            (get_item idx_offs (+ k 1))
                            file_bytes))
                (set! nst (narrow st en pfx))
                (if (>= nst 0) (set! st nst) #f)
                ;; 块内没有 >= pfx 的记录且这正是最后一块 = 整个文件都没有这样的词条：
                ;; 直接返回空表（省掉"从块首顺读到末尾"的那趟白扫，如 "zzz" 这类查询）
                (if (and (< nst 0) (>= en file_bytes)) (set! st en) #f)
              }
              (set! st (get_item idx_offs 0)))
          (if (< st file_bytes) (scan st pfx out) out)
        })))

;; 取指定偏移处的一条记录并拆字段：返回 (词条 音标 释义)；失败返回 ("" "" "") 之外的 #null
(define entry
  (lambda (off)
    (define r (Fs.read_range CSV off 512))
    (define t "")
    (define i 0)
    (if (null? r) #null
        {
          (set! t (get_item r 0))
          (set! i (find_ch t 0 10))
          (if (>= i 0) (set! t (String.slice t 0 i)) #f)
          (parse_rec t)
        })))

(define file_size (lambda () file_bytes))
(define match_max (lambda () MATCH_MAX))
