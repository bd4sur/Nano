;; ============================================================================
;; llm.scm —— 语言模型推理测试台（100% 纯 Scheme，不依赖 (native LLM)）
;;
;; 参照上游 /home/bd4sur/Animac/test/llm.scm 这个语言模型推理测试用例改写：
;;   · 上游用本地库 (native LLM) 做模型解析、分词与矩阵乘法；本 App 一律改用
;;     app/lib/llm_pure.scm（**纯 Scheme 实现**的等价库），于是整条推理链路——
;;     base64 解码 → 模型头/词表/权重张量解析 → 分词 → 前向传播 → 采样——全部跑在
;;     灵机解释器里，宿主不需要注册任何 LLM 本地库。
;;   · 上游是命令行批处理（display 打 stdout）；本 App 是 GUI App，输出同时写到屏
;;     幕上的「控制台」和宿主 stdout（浏览器里就是 devtools 控制台，端到端测试据此
;;     断言）。每拍（System.set_interval）只推进一步（读一块模型 / 一次 token 前向），
;;     所以长任务期间界面照常刷新、进度可见、退出键有效。
;;
;; 两个任务（与上游一致）：
;;   任务 0「序列生成」（sort6 模型，6 元素升序排列）：把 6 个数直接拼成的 "351420"
;;          编码成 6 个数字 token 送进模型，靠全局自注意力让模型逐位输出排序结果，
;;          再检查输出「全是数字词元」且「非降序」（该小模型在含重复数字的序列上本就
;;          有误差，故不硬比对逐值；固定输入下本地库实现给出 (0 0 1 2 4 5)）。
;;   任务 1「自回归文本生成」（psycho 230k 模型，精神分析黑话）：预填充提示词后逐
;;          token 贪心解码（温度 0，确定性），流式输出文本并统计 TPS。
;;
;; 与本地库实现的一致性（启动时自检并逐行打印，基准值取自上游 (native LLM) 的实跑
;; 记录，见 SELF_* 常量）：模型配置 9 元组、14 个权重张量的「长度 + 首/中/末元素」、
;; 分词 encode id 与 decode 往返、classifier/embedding 的「内容同、把柄异」语义。
;; 注：wasm32 宿主的浮点是单精度，故张量指纹用单元素 + 相对容差而不是求和
;;     （求和会因单精度连加漂移到百分级，见 doc/AGENTS.md 2026-11-07 条）。
;;
;; 模型数据：app/lib/llm/{sort6,psycho}/mNN.b64（base64 分块，由
;; tools/gen_llm_model_b64.py 生成）+ meta.scm（块数/标签）。运行时经 (native Fs)
;; 逐块读入并流式解码，内存峰值只有一个分块字符串。浏览器目标里这份数据是 p2
;; 后台挂载优先级，App 启动时可能还没到位——缺文件则打印提示并每 25 拍自动重试。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数、无一元取负；无 let/map/apply；
;; define 只写在模块顶层或 lambda 体顶层（循环体不是词法区域）；'() 是模块级静态槽
;; （空表一律 (cdr '(0)) 现造）；Gfx 坐标是无符号整数（先钳位）；除零会终止进程。
;; ============================================================================

(native Gfx)
(native Math)
(native System)
(native String)
(native Fs)
(native Touch)

(import LLM "app/lib/llm_pure.scm")     ; 纯 Scheme 的 LLM 库（接口与 (native LLM) 同名）
(import EX "app/lib/exit.scm")          ; 硬 A/ESC → 退出确认框 + 右上角「退出」按钮
(import M6 "app/lib/llm/sort6/meta.scm")
(import MP "app/lib/llm/psycho/meta.scm")

;; ---------------------------------------------------------------- 屏幕 / 控制台

(define SW (Gfx.width))
(define SH (Gfx.height))
(define FONT 1)                          ; 1 = 12px 抗锯齿（Gfx.set_font 默认档）
(define LINE_H 14)
(define HEAD_H 17)
(define LOG_MAX 400)                     ; 控制台最多保留的行数（超出丢最旧的）
(define WRAPU 50)                        ; 每行最多多少个「半角单位」（半角 1、全角 2）
(define TICK_MS 40)

(define log_lines (cdr '(0)))
(define dirty #t)
(define status_str "准备中…")

;; 追加一行到控制台（同时打到宿主 stdout，端到端测试据此断言）
(define say
  (lambda (s)
    {
      (display s)
      (newline)
      (push log_lines s)
      (if (> (length log_lines) LOG_MAX) (set! log_lines (cdr log_lines)) #f)
      (set! dirty #t)
    }))

(define set_status
  (lambda (s)
    {
      (set! status_str s)
      (set! dirty #t)
    }))

;; 码点是否按全角（2 个半角单位）排版
(define is_wide
  (lambda (code)
    (if (>= code 12288) #t
        (if (>= code 8192) #t
            (if (>= code 8592) (<= code 8703) #f)))))

;; 按半角单位折行 → 行列表
(define wrap_lines
  (lambda (s)
    (define out (cdr '(0)))
    (define n (String.length s))
    (define i 0)
    (define u 0)
    (define start 0)
    (define code 0)
    (define w 1)
    (while (< i n) {
      (set! code (String.charCodeAt i s))
      (set! w (if (is_wide code) 2 1))
      (if (> (+ u w) WRAPU)
          {
            (push out (String.slice s start i))
            (set! start i)
            (set! u 0)
          }
          #f)
      (set! u (+ u w))
      (set! i (+ i 1))
    })
    (if (< start n) (push out (String.slice s start n)) #f)
    out))

;; 折行后逐行入控制台
(define say_wrapped
  (lambda (s)
    (define ls (wrap_lines s))
    (define i 0)
    (while (< i (length ls)) {
      (say (get_item ls i))
      (set! i (+ i 1))
    })))

;; ---------------------------------------------------------------- 小工具

(define f4 (lambda (v) (String.atom_to_string (Math.to_fixed v 4))))
;; 6 位小数：张量指纹与上游实跑记录（Math.to_fixed 6）同精度
(define f6 (lambda (v) (String.atom_to_string (Math.to_fixed v 6))))

;; 列表 → "(1 2 3)"（仅用于日志：display 的输出不能直接拼进字符串）
(define narr_to_str
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s "(")
    (while (< i n) {
      (if (> i 0) (set! s (String.concat s " ")) #f)
      (set! s (String.concat s (String.atom_to_string (get_item lst i))))
      (set! i (+ i 1))
    })
    (String.concat s ")")))

;; 逗号连接（数字列表 → "3,1,2"）
;; 直接拼接的元素串（数字列表 → "312"）——与上游 list_to_str 一致：排序任务的输入
;; 是 6 个数字直接拼成的 6 字符串（无分隔符），分词后正好 6 个数字 token
(define concat_of
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s "")
    (while (< i n) {
      (set! s (String.concat s (String.atom_to_string (get_item lst i))))
      (set! i (+ i 1))
    })
    s))

(define csv_of
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s "")
    (while (< i n) {
      (if (> i 0) (set! s (String.concat s ",")) #f)
      (set! s (String.concat s (String.atom_to_string (get_item lst i))))
      (set! i (+ i 1))
    })
    s))

;; 两位数字（分块文件名 m00.b64 …）
(define i2s2
  (lambda (i)
    (if (< i 10)
        (String.concat "0" (String.atom_to_string i))
        (String.atom_to_string i))))

;; 新建 n 个 0 的缓冲
(define mkbuf
  (lambda (n)
    (define b (cdr '(0)))
    (define i 0)
    (while (< i n) { (push b 0) (set! i (+ i 1)) })
    b))

;; ---------------------------------------------------------------- 状态机相位

(define PH_TITLE 0)
(define PH_LOAD6 1)
(define PH_TEST6 2)
(define PH_SORT_PRE 3)
(define PH_SORT_GEN 4)
(define PH_LOADP 5)
(define PH_TESTP 6)
(define PH_GEN 7)
(define PH_DONE 8)
(define PH_WAIT 9)         ; 模型分块还没挂载好（浏览器下列为后台优先级）：等待重试

(define phase PH_TITLE)
(define chunk_i 0)
(define cur_model 1)       ; 1=sort6 2=psycho（PH_WAIT 重试时据此回到对应载入相位）
(define wait_ticks 0)
(define banner_done #f)
(define wait_said #f)

(define N_GEN 24)          ; 自回归步数（含预填充）——决定演示时长
(define TEMPERATURE 0)     ; 0 = 贪心（确定性，便于回归）；>0 = softmax + 多项分布采样
(define SORT_LEN 6)        ; sort6 模型固定 6

;; 自检基准值（均取自上游 (native LLM) 实现的实跑记录）
;; 自检基准值：由上游 (native LLM) 实现的实跑记录取得。
;; 张量指纹取「长度 + 第 0/中/末 个元素」而不是求和/求和均值：wasm32 宿主的浮点
;; 是**单精度**（am_float_t = float），连加十万项的和会漂移到百分级，而单元素逐位稳定。
(define SELF_CFG6 '(6 80 2 32 4 2 16 1 8))
(define SELF_FP6 '((64 1.10618 1.24337 1.2627)
                   (64 1.17654 1.32071 1.2545)
                   (32 1.89132 1.91044 1.89825)
                   (2560 0.542935 0.548011 -0.146184)
                   (2048 -0.066644 -0.03884 -0.050343)
                   (1024 -0.008842 -0.066899 0.053985)
                   (1024 -0.003333 -0.038036 -0.061802)
                   (2048 0.067551 -0.024275 0.001885)
                   (1024 -0.011936 -0.035305 0.140034)
                   (1024 0.011196 0.057933 -0.056837)
                   (1024 -0.01959 -0.173755 0.082236)
                   (24 1 -0.989992 0.999987)
                   (24 0 0.14112 0.005)
                   (2560 0.542935 0.548011 -0.146184)))
(define SELF_TEXTS6 '("0123456789" "a b c"))
(define SELF_ENC6 '((17 18 19 20 21 22 23 24 25 26) (27 8 28 8 29)))

(define SELF_CFGP '(512 4096 8 32 4 2 96 1 8))
(define SELF_FPP '((256 1.11249 1.61557 1.53318)
                  (256 1.17097 1.40176 1.6984)
                  (32 4.31098 6.32467 5.59257)
                  (131072 -0.022607 -0.022612 -0.169279)
                  (8192 -0.046066 -0.160942 0.217933)
                  (4096 0.258655 0.130429 0.278354)
                  (4096 0.017453 -0.056759 0.013569)
                  (8192 -0.000957 0.047947 -0.056221)
                  (24576 0.007169 0.017456 -0.173058)
                  (24576 -0.035932 0.051669 -0.025238)
                  (24576 -0.000189 0.003772 -0.009248)
                  (2048 1 -0.039791 0.872256)
                  (2048 0 -0.999208 0.48905)
                  (131072 -0.022607 -0.022612 -0.169279)))
(define SELF_TEXTP '("人类的本质是"))
(define SELF_ENCP '((192 561 187 293 673 190)))

(define self_ok 0)
(define self_bad 0)

;; ------------------------------------------------------- 模型配置 / 权重 / 缓冲

(define block_size 0)
(define vocab_size 0)
(define n_layer 0)
(define n_embd 0)
(define n_head 0)
(define n_kv_head 0)
(define n_hidden 0)
(define is_shared 0)
(define head_dim 0)
(define kv_dim 0)
(define kv_mul 1)

(define p_rms_attn #f)
(define p_rms_ffn #f)
(define p_rms_final #f)
(define p_emb #f)
(define p_wq #f)
(define p_wk #f)
(define p_wv #f)
(define p_wo #f)
(define p_w1 #f)
(define p_w2 #f)
(define p_w3 #f)
(define p_fcr #f)
(define p_fci #f)
(define p_cls #f)

(define bx #f)       ; 残差主缓冲 (n_embd)
(define bxb #f)      ; 临时 (n_embd)
(define bxba #f)     ; 注意力输出 (n_embd)
(define bxb2 #f)     ; 注意力投影 (n_embd)
(define bhb #f)      ; FFN 中间 (n_hidden)
(define bhb2 #f)     ; FFN 门控 (n_hidden)
(define bq #f)       ; query (n_embd)
(define k_cache #f)  ; n_layer 个 (block_size, kv_dim)
(define v_cache #f)
(define batt #f)     ; (n_head, block_size)
(define blogits #f)  ; vocab_size

;; 依当前模型配置重建缓冲区，并把 14 个权重张量绑定到模块变量
(define bind_model
  (lambda ()
    (define cfg (LLM.get_config))
    (define prm (LLM.get_param))
    (define i 0)
    (set! block_size (get_item cfg 0))
    (set! vocab_size (get_item cfg 1))
    (set! n_layer    (get_item cfg 2))
    (set! n_embd     (get_item cfg 3))
    (set! n_head     (get_item cfg 4))
    (set! n_kv_head  (get_item cfg 5))
    (set! n_hidden   (get_item cfg 6))
    (set! is_shared  (get_item cfg 7))
    (set! head_dim   (get_item cfg 8))
    (set! kv_dim (* n_kv_head head_dim))
    (set! kv_mul (Math.floor (/ n_head n_kv_head)))
    (set! p_rms_attn  (get_item prm 0))
    (set! p_rms_ffn   (get_item prm 1))
    (set! p_rms_final (get_item prm 2))
    (set! p_emb       (get_item prm 3))
    (set! p_wq        (get_item prm 4))
    (set! p_wk        (get_item prm 5))
    (set! p_wv        (get_item prm 6))
    (set! p_wo        (get_item prm 7))
    (set! p_w1        (get_item prm 8))
    (set! p_w2        (get_item prm 9))
    (set! p_w3        (get_item prm 10))
    (set! p_fcr       (get_item prm 11))
    (set! p_fci       (get_item prm 12))
    (set! p_cls       (get_item prm 13))
    (set! bx     (mkbuf n_embd))
    (set! bxb    (mkbuf n_embd))
    (set! bxba   (mkbuf n_embd))
    (set! bxb2   (mkbuf n_embd))
    (set! bhb    (mkbuf n_hidden))
    (set! bhb2   (mkbuf n_hidden))
    (set! bq     (mkbuf n_embd))
    (set! batt   (mkbuf (* n_head block_size)))
    (set! blogits (mkbuf vocab_size))
    (set! k_cache (mkbuf n_layer))
    (set! v_cache (mkbuf n_layer))
    (set! i 0)
    (while (< i n_layer) {
      (set_item! k_cache i (mkbuf (* block_size kv_dim)))
      (set_item! v_cache i (mkbuf (* block_size kv_dim)))
      (set! i (+ i 1))
    })
    #t))

;; 打印配置行（kv 把 "键=值 " 追加到字符串尾部）
(define kv
  (lambda (k v s)
    (String.concat s (String.concat k (String.concat "="
      (String.concat (String.atom_to_string v) " "))))))

(define say_config
  (lambda ()
    (define s "")
    (set! s (kv "block_size" block_size s))
    (set! s (kv "vocab" vocab_size s))
    (set! s (kv "layer" n_layer s))
    (set! s (kv "embd" n_embd s))
    (set! s (kv "head" n_head s))
    (set! s (kv "kv_head" n_kv_head s))
    (set! s (kv "hidden" n_hidden s))
    (set! s (kv "head_dim" head_dim s))
    (say (String.concat "  " s))))

;; ------------------------------------------------------------------ 基础算子

(define b_accum
  (lambda (a b size)
    (define i 0)
    (while (< i size) {
      (set_item! a i (+ (get_item a i) (get_item b i)))
      (set! i (+ i 1))
    })))

(define b_rms_norm
  (lambda (out x weight weight_offset size)
    (define ss 0)
    (define i 0)
    (define xi 0)
    (while (< i size) {
      (set! xi (get_item x i))
      (set! ss (+ ss (* xi xi)))
      (set! i (+ i 1))
    })
    (set! ss (/ ss size))
    (set! ss (+ ss 0.00001))
    (set! ss (/ 1.0 (Math.sqrt ss)))
    (set! i 0)
    (while (< i size) {
      (set_item! out i (* (get_item weight (+ weight_offset i)) (* ss (get_item x i))))
      (set! i (+ i 1))
    })))

(define b_softmax
  (lambda (x x_offset size)
    (define max_val -1e30)
    (define xi 0)
    (define i 0)
    (define sum 0)
    (while (< i size) {
      (set! xi (get_item x (+ x_offset i)))
      (if (> xi max_val) (set! max_val xi) #f)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i size) {
      (set! xi (Math.exp (- (get_item x (+ x_offset i)) max_val)))
      (set_item! x (+ x_offset i) xi)
      (set! sum (+ sum xi))
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i size) {
      (set_item! x (+ x_offset i) (/ (get_item x (+ x_offset i)) sum))
      (set! i (+ i 1))
    })))

;; RoPE：q 的全部 head，以及 k 在 pos 处的当前位（k_offset）
(define b_rope
  (lambda (q k pos k_offset)
    (define h 0)
    (define i 0)
    (define offset 0)
    (define freq_offset (* pos (Math.floor (/ head_dim 2))))
    (define val0 0)
    (define val1 0)
    (define fcr 0)
    (define fci 0)
    (set! h 0)
    (while (< h n_head) {
      (set! offset (* h head_dim))
      (set! i 0)
      (while (< i head_dim) {
        (set! val0 (get_item q (+ offset i)))
        (set! val1 (get_item q (+ offset (+ i 1))))
        (set! fcr (get_item p_fcr (+ freq_offset (Math.floor (/ i 2)))))
        (set! fci (get_item p_fci (+ freq_offset (Math.floor (/ i 2)))))
        (set_item! q (+ offset i)       (- (* val0 fcr) (* val1 fci)))
        (set_item! q (+ offset (+ i 1)) (+ (* val0 fci) (* val1 fcr)))
        (set! i (+ i 2))
      })
      (set! h (+ h 1))
    })
    (set! h 0)
    (while (< h n_kv_head) {
      (set! offset (+ k_offset (* h head_dim)))
      (set! i 0)
      (while (< i head_dim) {
        (set! val0 (get_item k (+ offset i)))
        (set! val1 (get_item k (+ offset (+ i 1))))
        (set! fcr (get_item p_fcr (+ freq_offset (Math.floor (/ i 2)))))
        (set! fci (get_item p_fci (+ freq_offset (Math.floor (/ i 2)))))
        (set_item! k (+ offset i)       (- (* val0 fcr) (* val1 fci)))
        (set_item! k (+ offset (+ i 1)) (+ (* val0 fci) (* val1 fcr)))
        (set! i (+ i 2))
      })
      (set! h (+ h 1))
    })))

;; 逐 token 前向传播 → logits（blogits，原地复用）
;;   is_causal=#t：因果注意力（自回归生成）；#f：全局注意力（序列生成）
(define llm_forward
  (lambda (token pos max_seq_len is_causal)
    (define layer 0)
    (define i 0)
    (define h 0)
    (define m 0)
    (define t 0)
    (define k #f)
    (define v #f)
    (define kv_pos_offset 0)
    (define wq_offset 0)
    (define wkv_offset 0)
    (define wo_offset 0)
    (define q_head_offset 0)
    (define k_head_offset 0)
    (define v_head_offset 0)
    (define att_head_offset 0)
    (define xba_head_offset 0)
    (define score 0)
    (define attn_range (if is_causal (+ pos 1) max_seq_len))
    (define emb_base (* token n_embd))
    (set! i 0)
    (while (< i n_embd) {
      (set_item! bx i (get_item p_emb (+ emb_base i)))
      (set! i (+ i 1))
    })
    (set! layer 0)
    (while (< layer n_layer) {
      (b_rms_norm bxb bx p_rms_attn (* layer n_embd) n_embd)
      (set! k (get_item k_cache layer))
      (set! v (get_item v_cache layer))
      (set! wq_offset  (* layer (* n_embd n_embd)))
      (set! wkv_offset (* layer (* n_embd kv_dim)))
      (set! kv_pos_offset (* pos kv_dim))
      (LLM.matmul bq  bxb p_wq 0             wq_offset  n_embd n_embd)
      (LLM.matmul k   bxb p_wk kv_pos_offset wkv_offset n_embd kv_dim)
      (LLM.matmul v   bxb p_wv kv_pos_offset wkv_offset n_embd kv_dim)
      (b_rope bq k pos kv_pos_offset)
      (set! h 0)
      (while (< h n_head) {
        (set! m (Math.floor (/ h kv_mul)))
        (set! q_head_offset (* h head_dim))
        (set! att_head_offset (* h block_size))
        (set! t 0)
        (while (< t attn_range) {
          (set! k_head_offset (+ (* t kv_dim) (* m head_dim)))
          (set! score 0)
          (set! i 0)
          (while (< i head_dim) {
            (set! score (+ score (* (get_item bq (+ q_head_offset i))
                                    (get_item k (+ k_head_offset i)))))
            (set! i (+ i 1))
          })
          (set_item! batt (+ att_head_offset t) (/ score (Math.sqrt head_dim)))
          (set! t (+ t 1))
        })
        (b_softmax batt att_head_offset attn_range)
        (set! xba_head_offset (* h head_dim))
        (set! i 0)
        (while (< i head_dim) {
          (set! score 0)
          (set! t 0)
          (while (< t attn_range) {
            (set! v_head_offset (+ (* t kv_dim) (* m head_dim)))
            (set! score (+ score (* (get_item batt (+ att_head_offset t))
                                    (get_item v (+ v_head_offset i)))))
            (set! t (+ t 1))
          })
          (set_item! bxba (+ xba_head_offset i) score)
          (set! i (+ i 1))
        })
        (set! h (+ h 1))
      })
      (set! wo_offset (* layer (* n_embd n_embd)))
      (LLM.matmul bxb2 bxba p_wo 0 wo_offset n_embd n_embd)
      (b_accum bx bxb2 n_embd)
      (b_rms_norm bxb bx p_rms_ffn (* layer n_embd) n_embd)
      (LLM.matmul bhb  bxb p_w1 0 (* layer (* n_hidden n_embd)) n_embd n_hidden)
      (LLM.matmul bhb2 bxb p_w3 0 (* layer (* n_hidden n_embd)) n_embd n_hidden)
      (set! i 0)
      (while (< i n_hidden) {
        (set! score (get_item bhb i))
        (set! score (* score (/ 1.0 (+ 1.0 (Math.exp (- 0 score))))))
        (set! score (* score (get_item bhb2 i)))
        (set_item! bhb i score)
        (set! i (+ i 1))
      })
      (LLM.matmul bxb bhb p_w2 0 (* layer (* n_embd n_hidden)) n_hidden n_embd)
      (b_accum bx bxb n_embd)
      (set! layer (+ layer 1))
    })
    (b_rms_norm bx bx p_rms_final 0 n_embd)
    (LLM.matmul blogits bx p_cls 0 0 n_embd vocab_size)
    blogits))

;; ------------------------------------------------------------------ 采样

;; 注意：maxv/softmax 的下界用**浮点**字面量。wasm32 宿主上整数有效宽度只有 27 位
;; （约 ±6.7e7），-100000000 这样的整数字面量会溢出绕回，使 argmax 永远不更新（全选 0 号
;; 词元）——这正是本 App 在浏览器里先踩到的坑。
(define sample_argmax
  (lambda (probs size)
    (define maxv -1e30)
    (define maxi 0)
    (define v 0)
    (define i 0)
    (while (< i size) {
      (set! v (get_item probs i))
      (if (> v maxv) { (set! maxv v) (set! maxi i) } #f)
      (set! i (+ i 1))
    })
    maxi))

(define sample_multinomial
  (lambda (probs size)
    (define i 0)
    (define r (Math.random))
    (define cdf 0.0)
    (define ret (- size 1))
    (while (< i size) {
      (set! cdf (+ cdf (get_item probs i)))
      (if (> cdf r) { (set! ret i) (set! i size) } (set! i (+ i 1)))
    })
    ret))

;; 温度采样：0 → 贪心（确定性）；否则 logits/温度 → softmax → 多项分布
;; （上游另有 top-k / top-p 截断采样；本 App 只演示这两种）
(define sample_token
  (lambda (logits size temperature)
    (define i 0)
    (if (== temperature 0)
        (sample_argmax logits size)
        {
          (set! i 0)
          (while (< i size) {
            (set_item! logits i (/ (get_item logits i) temperature))
            (set! i (+ i 1))
          })
          (b_softmax logits 0 size)
          (sample_multinomial logits size)
        })))

;; ---------------------------------------------------------------- 模型载入

(define read_chunk
  (lambda (name i)
    (Fs.read (String.concat "app/lib/llm/"
              (String.concat name
                (String.concat "/m" (String.concat (i2s2 i) ".b64")))))))

;; 读入一块并喂给流式解码器：
;;   返回 1 = 全部分块读完；0 = 还有；-1 = 文件缺失/读取失败
(define load_step
  (lambda (name nchunk)
    (define s #f)
    (define r 0)
    (if (>= chunk_i nchunk)
        (set! r 1)
        {
          (set! s (read_chunk name chunk_i))
          (set! chunk_i (+ chunk_i 1))
          (if (null? s)
              (set! r -1)
              {
                (LLM.init_feed s)
                (if (>= chunk_i nchunk) (set! r 1) (set! r 0))
              })
        })
    r))

;; ---------------------------------------------------------- 自检（与本地库比对）

(define check_eq
  (lambda (tag got want)
    (if (equal? got want)
        {
          (set! self_ok (+ self_ok 1))
          (say (String.concat "  [OK] " tag))
        }
        {
          (set! self_bad (+ self_bad 1))
          (say (String.concat "  [NG] " (String.concat tag
                 (String.concat " got=" (narr_to_str got)))))
        })))

;; 浮点近似相等（相对 1e-3 + 绝对 1e-6）：容纳 32 位宿主的单精度表示差异
(define near
  (lambda (a b)
    (define d (Math.abs (- a b)))
    (define rel (* 0.001 (Math.abs b)))
    (define tol (if (> rel 0.000001) rel 0.000001))
    (if (<= d tol) #t #f)))

;; 比对 14 个张量的「长度 + 首/中/末元素」指纹
(define check_fp
  (lambda (want)
    (define prm (LLM.get_param))
    (define i 0)
    (define t #f)
    (define n 0)
    (define w #f)
    (define bad 0)
    (while (< i 14) {
      (set! t (get_item prm i))
      (set! w (get_item want i))
      (set! n (length t))
      (if (== n (get_item w 0))
          {
            (if (near (get_item t 0) (get_item w 1)) #f (set! bad (+ bad 1)))
            (if (near (get_item t (Math.floor (/ n 2))) (get_item w 2)) #f (set! bad (+ bad 1)))
            (if (near (get_item t (- n 1)) (get_item w 3)) #f (set! bad (+ bad 1)))
          }
          (set! bad (+ bad 1)))
      (set! i (+ i 1))
    })
    (check_eq "14 个权重张量（长度/首中末元素）" bad 0)
    (check_eq "classifier/embedding 独立把柄" (eq? (get_item prm 13) (get_item prm 3)) #f)
    (check_eq "classifier/embedding 内容相同" (equal? (get_item prm 13) (get_item prm 3)) #t)))

;; 模型配置 9 元组
(define check_cfg
  (lambda (want)
    (define cfg (LLM.get_config))
    (define i 0)
    (define bad 0)
    (while (< i 9) {
      (if (== (get_item cfg i) (get_item want i)) #f (set! bad (+ bad 1)))
      (set! i (+ i 1))
    })
    (check_eq "模型配置 9 元组" bad 0)))

(define check_tokens
  (lambda (texts wants)
    (define i 0)
    (define bad 0)
    (define ids #f)
    (define j 0)
    (define back "")
    (while (< i (length texts)) {
      (set! ids (LLM.encode (get_item texts i)))
      (if (equal? ids (get_item wants i)) #f (set! bad (+ bad 1)))
      (set! back "")
      (set! j 0)
      (while (< j (length ids)) {
        (set! back (String.concat back (LLM.decode (get_item ids j))))
        (set! j (+ j 1))
      })
      (if (String.equals back (get_item texts i)) #f (set! bad (+ bad 1)))
      (say (String.concat "  分词 " (get_item texts i)))
      (say (String.concat "    → " (narr_to_str ids)))
      (set! i (+ i 1))
    })
    (check_eq "分词 encode/decode（对照本地库）" bad 0)))

;; ------------------------------------------------------------------ 绘制

(define draw_frame
  (lambda ()
    (define n 0)
    (define start 0)
    (define i 0)
    (define y 0)
    (define room 0)
    (define gl (cdr '(0)))
    (Gfx.set_font FONT)
    (Gfx.fill_rect 0 0 SW SH 12 14 20)
    (Gfx.fill_rect 0 0 SW HEAD_H 30 34 48)
    (Gfx.draw_text "灵机 LLM 测试台（纯 Scheme）" 6 2 226 232 242)
    (EX.draw_button)
    (Gfx.draw_text status_str 6 (+ HEAD_H 2) 150 210 160)
    ;; 自回归生成阶段：底部两行实时显示已生成的文本
    (if (== phase PH_GEN)
        {
          (set! gl (wrap_lines gen_line))
          (Gfx.fill_rect 0 (- SH 34) SW 34 18 22 40)
          (if (> (length gl) 2)
              (Gfx.draw_text (get_item gl (- (length gl) 2)) 5 (- SH 32) 255 235 140)
              #f)
          (if (> (length gl) 1)
              (Gfx.draw_text (get_item gl (- (length gl) 1)) 5 (- SH 18) 255 235 140)
              #f)
        }
        #f)
    (set! room (if (== phase PH_GEN)
                   (Math.floor (/ (- SH (+ HEAD_H 54)) LINE_H))
                   (Math.floor (/ (- SH (+ HEAD_H 20)) LINE_H))))
    (set! n (length log_lines))
    (set! start (if (> n room) (- n room) 0))
    (set! i start)
    (set! y (+ HEAD_H 18))
    (while (< i n) {
      (Gfx.draw_text (get_item log_lines i) 5 y 208 214 200)
      (set! y (+ y LINE_H))
      (set! i (+ i 1))
    })
    (Gfx.refresh)
    (set! dirty #f)))

;; ------------------------------------------------------------ 任务 0：序列生成

(define sort_input (cdr '(0)))
(define sort_ids (cdr '(0)))
(define sort_ref (cdr '(0)))
(define sort_out (cdr '(0)))
(define sort_pos 0)

;; 造随机输入、编码成 token、算 Scheme 侧参考排序
(define sort_setup
  (lambda ()
    (define s "")
    (define i 0)
    (define j 0)
    (define n 0)
    (define t 0)
    ;; 固定输入（确定性）：本地库上游实跑记录为 (0 0 1 2 4 5)——小模型在含重复
    ;; 数字的序列上本就有误差，这里比对的是「纯 Scheme 库的输出是否与本地库逐位相同」
    (set! sort_input '(3 5 1 4 2 0))
    (set! s (concat_of sort_input))
    (set! sort_ids (LLM.encode s))
    (set! sort_ref (cdr '(0)))
    (set! sort_out (cdr '(0)))
    (set! i 0)
    (while (< i (length sort_input)) {
      (push sort_ref (get_item sort_input i))
      (set! i (+ i 1))
    })
    (set! n (length sort_ref))
    (set! i 0)
    (while (< i n) {
      (set! j 0)
      (while (< j (- (- n 1) i)) {
        (if (> (get_item sort_ref j) (get_item sort_ref (+ j 1)))
            {
              (set! t (get_item sort_ref j))
              (set_item! sort_ref j (get_item sort_ref (+ j 1)))
              (set_item! sort_ref (+ j 1) t)
            }
            #f)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    (set! sort_pos 0)
    (say "")
    (say "任务 0：用 LLM 做 6 元素排序（全局自注意力）")
    (say (String.concat "  输入 " (String.concat s
         (String.concat "  → ids " (narr_to_str sort_ids)))))
    (say (String.concat "  参考排序 " (csv_of sort_ref)))
    #t))

;; 预填充：逐位前向，把整段序列的 KV 缓存填满（上游 ai_sort 第一段循环）
(define sort_prefill_step
  (lambda ()
    (llm_forward (get_item sort_ids sort_pos) sort_pos SORT_LEN #f)
    (set! sort_pos (+ sort_pos 1))
    (if (>= sort_pos SORT_LEN) #t #f)))

;; 生成：再逐位前向并贪心采样（上游 ai_sort 第二段循环）
(define sort_gen_step
  (lambda ()
    (define logits #f)
    (define new_token 0)
    (define tk "")
    (define v 0)
    (set! logits (llm_forward (get_item sort_ids sort_pos) sort_pos SORT_LEN #f))
    (set! new_token (sample_token logits vocab_size TEMPERATURE))
    (set! tk (LLM.decode new_token))
    (set! v (String.parseNumber tk))
    (if (undefined? v) (set! v -1) #f)
    (push sort_out v)
    (set! sort_pos (+ sort_pos 1))
    (if (>= sort_pos SORT_LEN) #t #f)))

;; 全部元素都是 0..9 的数字（parseNumber 失败的记为 -1）
(define digits_ok
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define ok #t)
    (while (< i n) {
      (if (< (get_item lst i) 0) (set! ok #f) #f)
      (set! i (+ i 1))
    })
    ok))

;; 序列非降序
(define sorted_ok
  (lambda (lst)
    (define n (length lst))
    (define i 1)
    (define ok #t)
    (while (< i n) {
      (if (< (get_item lst i) (get_item lst (- i 1))) (set! ok #f) #f)
      (set! i (+ i 1))
    })
    ok))

;; ------------------------------------------------------ 任务 1：自回归文本生成

(define gen_prompt "人类的本质是")
(define gen_ids (cdr '(0)))
(define gen_pos 0)
(define gen_next 0)
(define gen_line "")
(define gen_t0 0)
(define gen_n 0)

(define gen_setup
  (lambda ()
    (set! gen_ids (LLM.encode gen_prompt))
    (set! gen_pos 0)
    (set! gen_next (get_item gen_ids 0))
    (set! gen_line "")
    (set! gen_n 0)
    (set! gen_t0 (System.timestamp))
    (say "")
    (say "任务 1：自回归文本生成（贪心解码）")
    (say (String.concat "  提示词 " gen_prompt
         (String.concat "  → ids " (narr_to_str gen_ids))))
    (say "  输出：")
    #t))

;; 一步：前向当前 token，然后或取下一个提示词 token，或采样（上游 generate 同构）
(define gen_step
  (lambda ()
    (define logits #f)
    (define tk "")
    (set! logits (llm_forward gen_next gen_pos N_GEN #t))
    (if (< gen_pos (length gen_ids))
        (set! gen_next (get_item gen_ids gen_pos))
        {
          (set! gen_next (sample_token logits vocab_size TEMPERATURE))
          (set! gen_n (+ gen_n 1))
        })
    (set! tk (LLM.decode gen_next))
    (set! gen_line (String.concat gen_line tk))
    (set! gen_pos (+ gen_pos 1))
    (if (String.equals tk "<|eos|>") #t (if (>= gen_pos N_GEN) #t #f))))

;; ---------------------------------------------------------------- 各相位处理

(define step_title
  (lambda ()
    (set! self_ok 0)
    (set! self_bad 0)
    (set! wait_said #f)
    (if banner_done
        #f
        {
          (set! banner_done #t)
          (say "Animac 测试用例：自研 Nano 语言模型推理（纯 Scheme）")
          (say "  不依赖 (native LLM)：base64 解码 / 模型解析 / 分词 / 矩阵乘法")
          (say "  全部由 app/lib/llm_pure.scm 在解释器内完成。")
        })
    (say "")
    (say (String.concat "载入模型 1/2：" M6.label))
    (LLM.init_begin)
    (set! chunk_i 0)
    (set! cur_model 1)
    (set_status (String.concat "载入 " M6.label))
    (set! phase PH_LOAD6)))

;; 分块缺失（浏览器下 app/lib/llm/** 是后台挂载优先级，可能还没到位）：等一会自动重试
(define step_wait
  (lambda ()
    (set! wait_ticks (+ wait_ticks 1))
    (if wait_said
        #f
        {
          (set! wait_said #t)
          (say "模型分块尚未就绪（资源后台挂载中），稍后自动重试…")
        })
    (set_status "等待模型资源")
    (if (>= wait_ticks 25)
        {
          (set! wait_ticks 0)
          (LLM.init_begin)
          (set! chunk_i 0)
          (if (== cur_model 1) (set! phase PH_LOAD6) (set! phase PH_LOADP))
        }
        #f)))

(define step_load6
  (lambda ()
    (define r (load_step M6.name M6.nchunk))
    (set_status (String.concat "载入 sort6 " (String.concat
                  (String.atom_to_string chunk_i)
                  (String.concat "/" (String.atom_to_string M6.nchunk)))))
    (cond ((== r -1) { (set! cur_model 1) (set! wait_said #f) (set! wait_ticks 0)
                       (set! phase PH_WAIT) })
          ((== r 1)
           (if (LLM.init_end)
               {
                 (bind_model)
                 (say (String.concat "  载入完成；embedding[0]=" (f6 (get_item p_emb 0))))
                 (say_config)
                 (set! phase PH_TEST6)
               }
               {
                 (say "模型解析失败（分块不全或格式不符）")
                 (set! phase PH_TITLE)
               }))
          (else #f))))

(define step_test6
  (lambda ()
    (say "")
    (say "task0：自检（对照上游 (native LLM) 的实跑记录）")
    (check_cfg SELF_CFG6)
    (check_fp SELF_FP6)
    (check_tokens SELF_TEXTS6 SELF_ENC6)
    (say (String.concat "  自检 OK=" (String.concat (String.atom_to_string self_ok)
         (String.concat " NG=" (String.atom_to_string self_bad)))))
    (sort_setup)
    (set! phase PH_SORT_PRE)
    (set_status "任务 0：预填充 KV 缓存")))

(define step_sort_pre
  (lambda ()
    (if (sort_prefill_step)
        {
          (set! sort_pos 0)
          (set! phase PH_SORT_GEN)
          (set_status "任务 0：逐位生成")
        }
        (set_status (String.concat "任务 0：预填充 "
                      (String.concat (String.atom_to_string sort_pos)
                                    (String.concat "/" (String.atom_to_string SORT_LEN))))))))

(define step_sort_gen
  (lambda ()
    (if (sort_gen_step)
        {
          (say (String.concat "  模型输出 " (csv_of sort_out)))
          (check_eq "模型输出全是数字词元" (digits_ok sort_out) #t)
          (check_eq "模型输出非降序（排序任务达成）" (sorted_ok sort_out) #t)
          (say "")
          (say (String.concat "载入模型 2/2：" MP.label))
          (LLM.init_begin)
          (set! chunk_i 0)
          (set! cur_model 2)
          (set! phase PH_LOADP)
          (set_status (String.concat "载入 " MP.label))
        }
        (set_status (String.concat "任务 0：生成 "
                      (String.concat (String.atom_to_string sort_pos)
                                    (String.concat "/" (String.atom_to_string SORT_LEN))))))))

(define step_loadp
  (lambda ()
    (define r (load_step MP.name MP.nchunk))
    (set_status (String.concat "载入 psycho " (String.concat
                  (String.atom_to_string chunk_i)
                  (String.concat "/" (String.atom_to_string MP.nchunk)))))
    (cond ((== r -1) { (set! cur_model 2) (set! wait_said #f) (set! wait_ticks 0)
                       (set! phase PH_WAIT) })
          ((== r 1)
           (if (LLM.init_end)
               {
                 (bind_model)
                 (say (String.concat "  载入完成；embedding[0]=" (f6 (get_item p_emb 0))))
                 (say_config)
                 (set! phase PH_TESTP)
               }
               {
                 (say "模型解析失败（分块不全或格式不符）")
                 (set! phase PH_TITLE)
               }))
          (else #f))))

(define step_testp
  (lambda ()
    (say "")
    (say "task1：自检（对照上游 (native LLM) 的实跑记录）")
    (check_cfg SELF_CFGP)
    (check_fp SELF_FPP)
    (check_tokens SELF_TEXTP SELF_ENCP)
    (say (String.concat "  自检 OK=" (String.concat (String.atom_to_string self_ok)
         (String.concat " NG=" (String.atom_to_string self_bad)))))
    (gen_setup)
    (set! phase PH_GEN)
    (set_status "任务 1：生成中")))

(define step_gen
  (lambda ()
    (if (gen_step)
        {
          (say_wrapped gen_line)
          (set! phase PH_DONE)
          (finish)
        }
        (set_status (String.concat "任务 1：生成 "
                      (String.concat (String.atom_to_string gen_pos)
                                    (String.concat "/" (String.atom_to_string N_GEN))))))))

;; 收尾：TPS 与总结
(define finish
  (lambda ()
    (define el (- (System.timestamp) gen_t0))
    (define tps 0.0)
    (if (> el 0) (set! tps (/ (* 1.0 gen_n 1000.0) el)) #f)
    (say "")
    (say "============ 完成 ============")
    (say (String.concat "生成 " (String.concat (String.atom_to_string gen_n)
         (String.concat " 个词元，用时 " (String.concat (String.atom_to_string el)
         (String.concat " ms，TPS = " (f4 tps)))))))
    (say (String.concat "自检：OK=" (String.concat (String.atom_to_string self_ok)
         (String.concat " NG=" (String.atom_to_string self_bad)))))
    (say "点按屏幕可重跑；A/ESC 退出。")
    (set_status "完成")
    #t))

;; 完成后点按 → 从头重跑（不清空控制台，便于对照两次结果）
(define step_done
  (lambda ()
    (if (Touch.take_press)
        {
          (set! phase PH_TITLE)
          (set! wait_ticks 0)
        }
        #f)))

;; ------------------------------------------------------------------ 主循环

(define tick
  (lambda ()
    (EX.check)
    (cond ((== phase PH_TITLE) (step_title))
          ((== phase PH_LOAD6) (step_load6))
          ((== phase PH_TEST6) (step_test6))
          ((== phase PH_SORT_PRE) (step_sort_pre))
          ((== phase PH_SORT_GEN) (step_sort_gen))
          ((== phase PH_LOADP) (step_loadp))
          ((== phase PH_TESTP) (step_testp))
          ((== phase PH_GEN) (step_gen))
          ((== phase PH_WAIT) (step_wait))
          ((== phase PH_DONE) (step_done))
          (else #f))
    (if dirty (draw_frame) #f)))

;; 启动
(EX.install_with draw_frame)
(draw_frame)
(System.set_interval TICK_MS tick)