;; ============================================================================
;; llm_pure.scm —— 语言模型推理库（**纯 Scheme 实现**，不依赖 (native LLM)）
;;
;; 本模块是上游本地库 am_native_LLM.c（Nano 架构 / F32 权重）的 Scheme 等价物：
;; 同一份模型文件、同一套算法、同一套接口，只是全部用灵机 Scheme 写成语义等价的
;; 代码——模型解析（base64 解码 → 头 → 词表 → 权重张量）、分词与矩阵乘法全部在
;; 解释器里完成，宿主不需要注册任何 LLM 本地库。
;;
;; 对外接口（与 (native LLM) 逐条对应）：
;;   (L.init b64)      —— 从 base64 编码的模型文件字符串加载
;;   (L.get_config)    —— 9 元素列表 (block_size vocab_size n_layer n_embd n_head
;;                        n_kv_head n_hidden is_shared_classifier head_dim)
;;   (L.get_param)     —— 14 个权重张量（浮点列表），顺序同上游
;;   (L.encode text)   —— 文本 → token id 列表
;;   (L.decode token_id) —— token id → 文本
;;   (L.matmul xout x w xout_offset w_offset n d) —— 原地写回 xout 的矩阵乘法
;; 扩展（上游没有；供大模型分块加载，避免为一次性大字符串付出内存峰值）：
;;   (L.init_begin) / (L.init_feed b64) / (L.init_end) —— 流式加载三段式
;;   (L.loaded?) (L.unload) (L.vocab_size) (L.max_token_len) (L.token_count id->str 表长)
;;
;; 与上游的行为差异（均为「Scheme 没有异常」所致，不涉及算法）：
;;   · init 失败不终止进程，返回 #f（成功 #t）；未加载时 get_config/get_param 返回 #f。
;;   · 流式加载：init_feed 可多次调用，4 字符组的中间状态跨调用保留（块边界不必对齐）。
;;   · 阶码 255（inf/NaN）的权重返回 0.0（Nano F32 模型不含非有限权重）。
;;
;; 与上游**保持一致**的若干怪癖（照抄，不修正）：
;;   · token_classifier 恒与 token_embedding 同一把柄（上游 parse_model 无条件别名，
;;     即便 is_shared_classifier=0；其后的文件数据被忽略）；
;;   · 分词为贪心最长匹配；单字符一律取 charset 表的 id（表中没有则 0）；
;;     多字符 token 的最长尝试长度上限 17（AM_LLM_MAX_TOKEN_LEN）；
;;   · base64 解码跳过全部非 base64 字符（含空白），每 4 字符组按 '=' 个数出 1~3 字节；
;;   · 词表同名 token 先到先得（上游建 trie 遇重复返回 -1 忽略）；
;;   · 单字符 charset 映射先到先得（上游 map_set 尾插、get 取首个）；
;;   · matmul 在 d<=0 或 n<=0 时不做任何写入，越界的 xout 位置跳过写入。
;;
;; 解释器约束（manual.md，逐条遵守）：
;;   · 算术/比较恒二参数、无一元取负（写 (- 0 x)）；/ 与 mod 恒返回浮点 → 下标先 Math.floor；
;;   · 无 let / list / map / append / apply；define 无过程简写；define 只写在模块顶层或
;;     lambda 体顶层；循环体不是词法区域，不在其中 define；
;;   · '() 是模块级静态槽（push 会累积），空表一律 (cdr '(0)) 现造；
;;   · wasm32 宿主上无符号整数有效宽度约 27 位，故**不做 32 位整数拼装**：F32 的 4 字节
;;     按 符号/阶码/尾数 分解，再由 2 的幂表还原。该式在 double 下精确（23 位尾数是
;;     整数、2 的幂精确、乘积无舍入），因此与 C 侧 float→double 逐位一致。
;; ============================================================================

(native String)
(native Math)
(native Table)

;; ---------------------------------------------------------------- 基础常量

;; 新建空表（不能用 '() 字面量：那是模块级静态槽，多次 push 会累积）
(define llm_new_list (lambda () (cdr '(0))))

;; 上游硬编码的最长 token（字符数）；词表里的 token 超过该长度时不可达（同上游）
(define LLM_MAX_TOKEN_LEN 17)

;; 2 的幂表偏移：P2[k + LLM_P2_OFF] = 2^k，k ∈ [-149, 127]
(define LLM_P2_OFF 149)

;; 无符号 32 位小端计数字段的低 24 位（头/词表里的计数都远小于 2^24；
;; 高字节是魔数等本实现不校验的字段）
(define llm_rd24
  (lambda (lst off)
    (+ (get_item lst off)
       (+ (* 256 (get_item lst (+ off 1)))
          (* 65536 (get_item lst (+ off 2)))))))

;; 2 的幂表（模块加载时建一次）
(define llm_p2
  (lambda ()
    (define lst (llm_new_list))
    (define k -149)
    (while (<= k 127) {
      (push lst (Math.pow 2.0 k))
      (set! k (+ k 1))
    })
    lst))

(define LLM_P2 (llm_p2))

;; base64 值表：码点 → 6 位值；非法字符（含空白与 '='）为 -1
(define llm_b64_tab
  (lambda ()
    (define tab (llm_new_list))
    (define i 0)
    (while (< i 128) { (push tab -1) (set! i (+ i 1)) })
    (set! i 0)
    (while (< i 26) { (set_item! tab (+ 65 i) i) (set! i (+ i 1)) })          ; A-Z
    (set! i 0)
    (while (< i 26) { (set_item! tab (+ 97 i) (+ i 26)) (set! i (+ i 1)) })   ; a-z
    (set! i 0)
    (while (< i 10) { (set_item! tab (+ 48 i) (+ i 52)) (set! i (+ i 1)) })   ; 0-9
    (set_item! tab 43 62)   ; '+'
    (set_item! tab 47 63)   ; '/'
    tab))

(define LLM_B64 (llm_b64_tab))

;; ------------------------------------------------- F32 字节 → 浮点（精确还原）

(define llm_b_low  (lambda (b) (if (>= b 128) (- b 128) b)))
(define llm_b_high (lambda (b) (if (>= b 128) 1 0)))

;; b0..b3 为小端 4 字节 → 该 F32 的数值（double 精确表示）
(define llm_bits_to_float
  (lambda (b0 b1 b2 b3)
    (define e (+ (* (llm_b_low b3) 2) (llm_b_high b2)))
    (define m (+ (* (llm_b_low b2) 65536) (+ (* b1 256) b0)))
    (define v 0.0)
    (if (== e 255)
        (set! v 0.0)   ;; inf/NaN：Nano F32 权重不含，置 0 以免污染推理
        (if (== e 0)
            (set! v (* m (get_item LLM_P2 0)))                  ;; 非规格化：m·2^-149
            (set! v (* (+ 1.0 (/ m 8388608.0))
                       (get_item LLM_P2 (+ (- e 127) LLM_P2_OFF))))))
    (if (== (llm_b_high b3) 1)
        (if (== v 0.0) (* -1.0 0.0) (set! v (- 0 v)))   ;; -0.0 也保留
        #f)
    v))

;; ---------------------------------------------------------------- 模块状态

(define llm_is_loaded #f)
(define llm_cfg #f)              ;; 9 元素配置列表
(define llm_param #f)            ;; 14 元素参数列表（第 14 项与第 4 项同一把柄）
(define llm_tok_strs #f)         ;; id → token 字符串（按 id 索引的表）
(define llm_uni #f)              ;; Table：字符码点 → token id（单字符 token）
(define llm_multi #f)            ;; Table：token 字符串 → token id（多字符 token）
(define llm_vsize 0)             ;; 词表大小（tokenizer 段声明）
(define llm_maxtok 1)            ;; 词表内最长多字符 token 的字符数（上限 17）

;; 流式解码状态
(define llm_phase 0)             ;; 0=头(256B) 1=词表长度(4B) 2=词表数据 3=参数
(define llm_qbuf #f)             ;; 4 槽 base64 组缓冲
(define llm_qn 0)
(define llm_npad 0)
(define llm_hdr #f)              ;; 256 字节头
(define llm_tlen #f)             ;; 4 字节词表字段长度
(define llm_tdata #f)            ;; 词表字段其余字节
(define llm_tbytes 0)
(define llm_tensors #f)          ;; 13 个张量（列表的列表）
(define llm_sizes #f)            ;; 13 个张量元素个数
(define llm_ti 0)                ;; 当前张量下标
(define llm_tfill 0)             ;; 当前张量已填元素数
(define llm_sbuf #f)             ;; 4 槽 float 字节缓冲
(define llm_sn 0)
(define llm_done #f)             ;; 参数区已全部解出
;; 本 4 字符组内发生过阶段切换（阶段切换可能落在组的中间：词表字段长度不是 3 的倍数），
;; 此时该组的 3 个字节必须逐字节分派，不能整组走参数区快路径
(define llm_gdirty #f)

;; ----------------------------------------------------- 头 / 词表 / 张量表大小

;; 头解析：校验 model_type/quant_type，取 9 项配置
(define llm_parse_header
  (lambda ()
    (define model_type (llm_rd24 llm_hdr 16))
    (define block_size (llm_rd24 llm_hdr 24))
    (define vocab_size (llm_rd24 llm_hdr 28))
    (define n_layer    (llm_rd24 llm_hdr 32))
    (define n_embd     (llm_rd24 llm_hdr 36))
    (define n_head     (llm_rd24 llm_hdr 40))
    (define n_kv_head  (llm_rd24 llm_hdr 44))
    (define n_hidden   (llm_rd24 llm_hdr 48))
    (define is_shared  (llm_rd24 llm_hdr 52))
    (define head_dim   (llm_rd24 llm_hdr 56))
    (define quant_type (llm_rd24 llm_hdr 60))
    (if (not (== model_type 0))
        #f
        (if (not (== quant_type 0))
            #f
            {
              (set! llm_cfg (llm_new_list))
              (push llm_cfg block_size)
              (push llm_cfg vocab_size)
              (push llm_cfg n_layer)
              (push llm_cfg n_embd)
              (push llm_cfg n_head)
              (push llm_cfg n_kv_head)
              (push llm_cfg n_hidden)
              (push llm_cfg is_shared)
              (push llm_cfg head_dim)
              #t
            }))))

;; 13 个张量的元素个数（顺序同上游 parse_model 的指针推进）
(define llm_make_sizes
  (lambda ()
    (define n_layer (get_item llm_cfg 2))
    (define n_embd  (get_item llm_cfg 3))
    (define n_head  (get_item llm_cfg 4))
    (define n_kv_head (get_item llm_cfg 5))
    (define n_hidden (get_item llm_cfg 6))
    (define vocab_size (get_item llm_cfg 1))
    (define block_size (get_item llm_cfg 0))
    (define head_dim (get_item llm_cfg 8))
    (define kv_dim 0)
    (define lst (llm_new_list))

    (if (== head_dim 0) (set! head_dim (Math.floor (/ n_embd n_head))) #f)
    (set! kv_dim (* n_kv_head head_dim))
    (set_item! llm_cfg 8 head_dim)          ;; 上游同样回填 head_dim
    (push lst (* n_layer n_embd))                                       ; rms_norm_attn
    (push lst (* n_layer n_embd))                                       ; rms_norm_ffn
    (push lst n_embd)                                                   ; rms_norm_final
    (push lst (* vocab_size n_embd))                                    ; token_embedding
    (push lst (* (* n_layer (* n_head head_dim)) n_embd))               ; wq
    (push lst (* (* n_layer kv_dim) n_embd))                            ; wk
    (push lst (* (* n_layer kv_dim) n_embd))                            ; wv
    (push lst (* (* n_layer n_embd) (* n_head head_dim)))               ; wo
    (push lst (* (* n_layer n_hidden) n_embd))                          ; w1
    (push lst (* (* n_layer n_embd) n_hidden))                          ; w2
    (push lst (* (* n_layer n_hidden) n_embd))                          ; w3
    (push lst (Math.floor (/ (* block_size head_dim) 2)))               ; freq_cis_real
    (push lst (Math.floor (/ (* block_size head_dim) 2)))               ; freq_cis_imag
    lst))

;; 词表解析
(define llm_parse_vocab
  (lambda ()
    (define strs (llm_new_list))
    (define i 0)
    (define off 4)
    (define bc 0)
    (define header 0)
    (define tid 0)
    (define tlen 0)
    (define s "")
    (define k 0)
    (define cc 0)
    (set! llm_vsize (llm_rd24 llm_tdata 0))
    (while (< i llm_vsize) { (push strs "") (set! i (+ i 1)) })
    (set! llm_tok_strs strs)
    (set! llm_uni (Table.make))
    (set! llm_multi (Table.make))
    (set! llm_maxtok 1)
    (while (< bc (- llm_tbytes 8)) {
      (set! header (llm_rd24 llm_tdata off))
      (set! tid (llm_rd24 llm_tdata (+ off 4)))
      ;; token_length = header & 0xFF（header 其余位上游未使用）
      (set! tlen (- header (* 256 (Math.floor (/ header 256)))))
      (set! s "")
      (set! k 0)
      (while (< k tlen) {
        (set! s (String.concat s
                  (String.fromCharCode (llm_rd24 llm_tdata (+ (+ off 8) (* k 4))))))
        (set! k (+ k 1))
      })
      (if (== tlen 1)
          {
            (set! cc (* 1.0 (llm_rd24 llm_tdata (+ off 8))))
            ;; 注意：Table 的数字键不做 int/float 归一（1 与 1.0 是不同的键），而
            ;; (String.charCodeAt) 返回的是浮点，故 charset 表统一用浮点键
            (if (not (Table.contains llm_uni cc)) (Table.set llm_uni cc tid) #f)
          }
          (if (> tlen 1)
              {
                (if (not (Table.contains llm_multi s)) (Table.set llm_multi s tid) #f)
                (if (> tlen llm_maxtok) (set! llm_maxtok tlen) #f)
              }
              #f))
      (if (< tid llm_vsize) (set_item! strs tid s) #f)
      (set! off (+ off (+ 8 (* tlen 4))))
      (set! bc (+ bc (+ 8 (* tlen 4))))
    })
    (if (> llm_maxtok LLM_MAX_TOKEN_LEN) (set! llm_maxtok LLM_MAX_TOKEN_LEN) #f)
    #t))

;; ------------------------------------------------------- 阶段切换（解码状态机）

(define llm_phase_header_done
  (lambda ()
    (if (llm_parse_header)
        {
          (set! llm_tlen (llm_new_list))
          (set! llm_phase 1)
          (set! llm_gdirty #t)
          #t
        }
        #f)))

(define llm_phase_tlen_done
  (lambda ()
    (set! llm_tbytes (llm_rd24 llm_tlen 0))
    (set! llm_tdata (llm_new_list))
    (set! llm_phase 2)
    (set! llm_gdirty #t)
    #t))

(define llm_phase_tdata_done
  (lambda ()
    (define i 0)
    (define szs (llm_make_sizes))
    (define tens (llm_new_list))
    (if (llm_parse_vocab)
        {
          (while (< i (length szs)) { (push tens (llm_new_list)) (set! i (+ i 1)) })
          (set! llm_sizes szs)
          (set! llm_tensors tens)
          (set! llm_ti 0)
          (set! llm_tfill 0)
          (set! llm_sn 0)
          (set! llm_sbuf (llm_new_list))
          (push llm_sbuf 0) (push llm_sbuf 0) (push llm_sbuf 0) (push llm_sbuf 0)
          (set! llm_phase 3)
          (set! llm_gdirty #t)
          #t
        }
        #f)))

;; 一个 4 字节 float 凑齐：还原成浮点压入当前张量；张量填满则切下一个
(define llm_slot_flush
  (lambda ()
    (if llm_done
        #f
        {
          (set! llm_sn 0)
          (push (get_item llm_tensors llm_ti)
                (llm_bits_to_float (get_item llm_sbuf 0) (get_item llm_sbuf 1)
                                   (get_item llm_sbuf 2) (get_item llm_sbuf 3)))
          (set! llm_tfill (+ llm_tfill 1))
          (if (== llm_tfill (get_item llm_sizes llm_ti))
              {
                (set! llm_ti (+ llm_ti 1))
                (set! llm_tfill 0)
                (if (>= llm_ti (length llm_sizes)) (set! llm_done #t) #f)
              }
              #f)
        })))

;; 慢路径：逐字节按阶段分派。头/词表区恒走这里；参数区只在两种极少数的组里走这里——
;; 「本组内发生了阶段切换」（词表字段长度通常不是 3 的倍数，切换点会落在组中间）与
;; 「流末尾带 '=' 填充」；其余组走 init_feed 里 3 字节一起拼浮点的内联快路径。
(define llm_sink_slow
  (lambda (b)
    (if (== llm_phase 0)
        { (push llm_hdr b) (if (== (length llm_hdr) 256) (llm_phase_header_done) #f) }
        (if (== llm_phase 1)
            { (push llm_tlen b) (if (== (length llm_tlen) 4) (llm_phase_tlen_done) #f) }
            (if (== llm_phase 2)
                {
                  (push llm_tdata b)
                  (if (== (length llm_tdata) (- llm_tbytes 4)) (llm_phase_tdata_done) #f)
                }
                {
                  (set_item! llm_sbuf llm_sn b)
                  (set! llm_sn (+ llm_sn 1))
                  (if (== llm_sn 4) (llm_slot_flush) #f)
                })))))

;; ------------------------------------------------------------ 流式 base64 解码

;; (L.init_feed b64)：喂一段 base64 文本（可多次；4 字符组状态跨调用保留）
(define init_feed
  (lambda (b64)
    (define n (String.length b64))
    (define i 0)
    (define c 0)
    (define v 0)
    (define q0 0)
    (define q1 0)
    (define q2 0)
    (define q3 0)
    (define b0 0)
    (define b1 0)
    (define b2 0)
    (define nb 0)
    (while (< i n) {
      (set! c (String.charCodeAt i b64))
      (set! v (if (< c 128) (get_item LLM_B64 c) -1))
      (if (>= v 0)
          { (set_item! llm_qbuf llm_qn v) (set! llm_qn (+ llm_qn 1)) }
          (if (== c 61)          ;; '='
              {
                (set_item! llm_qbuf llm_qn 0)
                (set! llm_qn (+ llm_qn 1))
                (set! llm_npad (+ llm_npad 1))
              }
              #f))
      (if (== llm_qn 4)
          {
            (set! q0 (get_item llm_qbuf 0))
            (set! q1 (get_item llm_qbuf 1))
            (set! q2 (get_item llm_qbuf 2))
            (set! q3 (get_item llm_qbuf 3))
            (set! b0 (Math.bit_or (Math.bit_shl q0 2) (Math.bit_lshr q1 4)))
            (set! b1 (Math.bit_or (Math.bit_shl (Math.bit_and q1 15) 4)
                                  (Math.bit_lshr q2 2)))
            (set! b2 (Math.bit_or (Math.bit_shl (Math.bit_and q2 3) 6) q3))
            (set! nb (if (== llm_npad 0) 3 (if (== llm_npad 1) 2 1)))
            (set! llm_qn 0)
            (set! llm_npad 0)
            (if (and (== llm_phase 3) (not llm_gdirty))
                ;; 快路径：参数区（占全部字节的绝大多数）内联拼 float，不逐字节调函数
                {
                  (set_item! llm_sbuf llm_sn b0)
                  (set! llm_sn (+ llm_sn 1))
                  (if (== llm_sn 4) (llm_slot_flush) #f)
                  (if (>= nb 2)
                      {
                        (set_item! llm_sbuf llm_sn b1)
                        (set! llm_sn (+ llm_sn 1))
                        (if (== llm_sn 4) (llm_slot_flush) #f)
                      }
                      #f)
                  (if (>= nb 3)
                      {
                        (set_item! llm_sbuf llm_sn b2)
                        (set! llm_sn (+ llm_sn 1))
                        (if (== llm_sn 4) (llm_slot_flush) #f)
                      }
                      #f)
                }
                {
                  (llm_sink_slow b0)
                  (if (>= nb 2) (llm_sink_slow b1) #f)
                  (if (>= nb 3) (llm_sink_slow b2) #f)
                })
            (set! llm_gdirty #f)
          }
          #f)
      (if llm_done break #f)
      (set! i (+ i 1))
    })
    #t))

;; (L.init_begin)：重置全部解码状态
(define init_begin
  (lambda ()
    (set! llm_is_loaded #f)
    (set! llm_cfg #f)
    (set! llm_param #f)
    (set! llm_tok_strs #f)
    (set! llm_uni #f)
    (set! llm_multi #f)
    (set! llm_vsize 0)
    (set! llm_maxtok 1)
    (set! llm_phase 0)
    (set! llm_qbuf (llm_new_list))
    (push llm_qbuf 0) (push llm_qbuf 0) (push llm_qbuf 0) (push llm_qbuf 0)
    (set! llm_qn 0)
    (set! llm_npad 0)
    (set! llm_hdr (llm_new_list))
    (set! llm_tlen #f)
    (set! llm_tdata #f)
    (set! llm_tbytes 0)
    (set! llm_tensors #f)
    (set! llm_sizes #f)
    (set! llm_ti 0)
    (set! llm_tfill 0)
    (set! llm_sbuf #f)
    (set! llm_sn 0)
    (set! llm_done #f)
    (set! llm_gdirty #f)
    #t))

;; (L.init_end)：把 13 个张量拼成 14 项参数表（第 14 项别名第 4 项）并置加载完成
;; 复制一个浮点列表（token_classifier 需要独立把柄，见 init_end）
(define llm_copy_list
  (lambda (src)
    (define n (length src))
    (define dst (llm_new_list))
    (define i 0)
    (while (< i n) { (push dst (get_item src i)) (set! i (+ i 1)) })
    dst))

(define init_end
  (lambda ()
    (define i 0)
    (define lst #f)
    (if llm_done
        {
          (set! lst (llm_new_list))
          (while (< i (length llm_tensors)) {
            (push lst (get_item llm_tensors i))
            (set! i (+ i 1))
          })
          ;; token_classifier：上游 get_param 用**同一份数据另建一个列表**（两条
          ;; 独立把柄，eq? 为假而 equal? 为真），故此处同样复制一份，保持可观测语义
          (push lst (llm_copy_list (get_item llm_tensors 3)))
          (set! llm_param lst)
          ;; 释放中间字节缓冲（大表，尽快交给 GC）
          (set! llm_hdr #f)
          (set! llm_tlen #f)
          (set! llm_tdata #f)
          (set! llm_tensors #f)
          (set! llm_sizes #f)
          (set! llm_sbuf #f)
          (set! llm_is_loaded #t)
          #t
        }
        #f)))

;; (L.init b64)：一次性加载（等价于 init_begin + init_feed + init_end）
(define init
  (lambda (b64)
    (init_begin)
    (init_feed b64)
    (init_end)))

(define unload
  (lambda ()
    (init_begin)
    #t))

;; ------------------------------------------------------------------ 自省接口

(define loaded? (lambda () llm_is_loaded))
(define get_config (lambda () (if llm_is_loaded llm_cfg #f)))
(define get_param (lambda () (if llm_is_loaded llm_param #f)))
(define vocab_size (lambda () llm_vsize))
(define max_token_len (lambda () llm_maxtok))

;; ------------------------------------------------------------ 分词（编解码）

;; 单字符 token 的 id（charset 表没有该字符时为 0，与上游 map_get 缺省一致）。
;; (* 1.0 code) 把码点统一成浮点：Table 的数字键不归一 int/float（见 llm_parse_vocab）
(define llm_char_id
  (lambda (code)
    (define v (Table.get llm_uni (* 1.0 code)))
    (if (undefined? v) 0 v)))

;; (L.encode text)：贪心最长匹配分词（与上游 am_llm_tokenize 等价）
(define encode
  (lambda (text)
    (define n 0)
    (define out (llm_new_list))
    (define i 0)
    (define avail 0)
    (define k 0)
    (define tid #f)
    (define found 0)
    (if (not llm_is_loaded)
        out
        {
          (set! n (String.length text))
          (while (< i n) {
            (set! avail (- n i))
            (if (> avail llm_maxtok) (set! avail llm_maxtok) #f)
            (set! k avail)
            (set! found -1)
            (while (and (>= k 2) (== found -1)) {
              (set! tid (Table.get llm_multi (String.slice text i (+ i k))))
              (if (undefined? tid) (set! k (- k 1)) (set! found tid))
            })
            (if (== found -1)
                {
                  (push out (llm_char_id (String.charCodeAt i text)))
                  (set! i (+ i 1))
                }
                {
                  (push out found)
                  (set! i (+ i k))
                })
          })
          out
        })))

;; (L.decode id)：token id → 文本（越界或空槽返回 ""，与上游 utoken==NULL 时跳过的效果一致）
(define decode
  (lambda (id)
    (define idx (Math.floor id))
    (if (not llm_is_loaded)
        ""
        (if (and (>= idx 0) (< idx (length llm_tok_strs)))
            (get_item llm_tok_strs idx)
            ""))))

;; ------------------------------------------------------------ 矩阵乘法（原地）

;; (L.matmul xout x w xout_offset w_offset n d)
;;   xout[xout_offset+i] = Σⱼ w[w_offset + i·n + j] · x[j]     (i = 0..d-1)
;; 累加顺序与 C 侧逐项一致，故浮点结果逐位相同。
(define matmul
  (lambda (xout x w xout_offset w_offset n d)
    (define xo (Math.floor xout_offset))
    (define wo (Math.floor w_offset))
    (define nn (Math.floor n))
    (define dd (Math.floor d))
    (define lim (length xout))
    (define i 0)
    (define j 0)
    (define val 0)
    (define base 0)
    (if (and (> dd 0) (> nn 0))
        (while (< i dd) {
          (set! val 0)
          (set! base (+ wo (* i nn)))
          (set! j 0)
          (while (< j nn) {
            (set! val (+ val (* (get_item w (+ base j)) (get_item x j))))
            (set! j (+ j 1))
          })
          (if (< (+ xo i) lim) (set_item! xout (+ xo i) val) #f)
          (set! i (+ i 1))
        })
        #f)
    #t))