;; ============================================================================
;; llm.scm —— 语言模型推理测试台（100% 纯 Scheme，不依赖 (native LLM)）
;;
;; 参照上游 /home/bd4sur/Animac/test/llm.scm 这个语言模型推理测试用例改写：
;; 模型解析、分词、矩阵乘法一律由 app/lib/llm_pure.scm（**纯 Scheme 实现**的等价库）
;; 完成，宿主不注册任何 LLM 本地库。
;;
;; 两个**互斥**的演示场景（顶部两个标签页切换，任一时刻只有一个场景在跑）：
;;
;;   场景 1「LLM 排序」：6 槽位数字输入（像银行密码/短信验证码）+ 数字软键盘
;;     （0-9 / 退格 / 输入）。按「输入」后把 6 个数字直接拼成 6 字符串（如 "351420"）
;;     编码成 6 个 token 送进 sort6 模型，用**全局自注意力**逐位输出排序结果；
;;     该小模型在含重复数字的序列上有误差，故**每轮带温度采样、共跑 5 轮，
;;     逐位投票取多数**作为最终排序结果，并与 Scheme 侧参考排序比对。
;;
;;   场景 2「LLM 可视化生成」：**无需任何输入**。提示词固定为「人类的本质」，随机
;;     种子每次取当前时间戳（并随「重来」轮次递增），逐 token 采样并**流式**输出
;;     文本，实时统计 TPS。点「重来」换种子重新生成（提示词不变）。
;;
;; 交互与渲染的取舍（灵机 App 的通用约束）：
;;   · 每拍（System.set_interval 40ms）只推进一步（读一块模型 / 一次 token 前向），
;;     长任务期间界面照常刷新、标签页与退出键可用；
;;   · 触屏用**按下沿**（Touch.take_press）驱动自绘按钮的命中判定，几何常量集中在
;;     「布局」一节（tools/e2e/browser/regress_llm.js 按同一组常量点按）；
;;   · 输出同时写宿主 stdout（浏览器里是 devtools 控制台），带 [model]/[chk]/[sort]/
;;     [gen] 前缀的稳定标记，端到端测试据此断言。
;;
;; 模型数据：app/lib/llm/{sort6,psycho}/mNN.b64（base64 分块，tools/gen_llm_model_b64.py
;; 生成）+ meta.scm。运行时经 (native Fs) 逐块读入并流式解码，内存峰值只有一个分块
;; 字符串；浏览器目标里这份数据是 p2 后台挂载优先级，缺文件则等待重试。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数、无一元取负；无 let/map/apply；
;; define 只写在模块顶层或 lambda 体顶层（循环体不是词法区域）；'() 是模块级静态槽
;; （空表一律 (cdr '(0)) 现造）；Gfx 坐标是无符号整数（先钳位）；除零会终止进程；
;; 哨兵大数一律写成**浮点**字面量（wasm32 整数只有 27 位有效宽度）。
;; ============================================================================

(native Gfx)
(native Math)
(native System)
(native String)
(native Fs)
(native Touch)
(native Time)

(import LLM "app/lib/llm_pure.scm")     ; 纯 Scheme 的 LLM 库（接口与 (native LLM) 同名）
(import EX "app/lib/exit.scm")          ; 硬 A/ESC → 退出确认框 + 右上角「退出」按钮
(import M6 "app/lib/llm/sort6/meta.scm")
(import MP "app/lib/llm/psycho/meta.scm")

;; ---------------------------------------------------------------- 布局常量

(define SW (Gfx.width))
(define SH (Gfx.height))
(define FONT_S 1)                        ; 12px 抗锯齿
(define FONT_L 2)                        ; 16px 抗锯齿（槽位数字）
(define TICK_MS 40)

;; 顶部标题条与两个标签页
(define HEAD_H 17)
(define TAB_Y 19)
(define TAB_H 18)
(define TAB_W 152)
(define TAB1_X 4)
(define TAB2_X 164)
(define BODY_Y 42)                       ; 正文起点

;; 场景 1：6 个数字槽位（40x30，间距 8，整体居中）
(define SLOT_X0 20)
(define SLOT_STEP 48)
(define SLOT_W 40)
(define SLOT_H 30)
(define SLOT_Y 60)

;; 场景 1：数字软键盘（3 列 × 4 行）
(define KP_X0 10)
(define KP_Y0 96)
(define KP_W 96)
(define KP_H 28)
(define KP_GX 6)
(define KP_GY 6)

;; 场景 1：结果面板
(define RES_Y 95)
(define RES_STEP 12)
(define AGAIN_X 110)                     ; 「重新输入」按钮
(define AGAIN_Y 196)
(define AGAIN_W 100)
(define AGAIN_H 22)

;; 场景 2：流式文本区与底部
(define TXT_Y 58)
(define TXT_STEP 13)
(define TXT_LINES 9)
(define STOP_X 134)                      ; 「停止生成」按钮（只生成中显示）
(define STOP_Y 196)
(define STOP_W 104)
(define STOP_H 22)
(define REDO_X 246)                      ; 「重来」按钮
(define REDO_Y 196)
(define REDO_W 68)
(define REDO_H 22)
(define WRAPU 50)                        ; 每行最多多少个「半角单位」

;; ---------------------------------------------------------------- 日志

;; 打到宿主 stdout（浏览器里是 devtools 控制台）——端到端测试据此断言
(define log
  (lambda (s)
    {
      (display s)
      (newline)
    }))

;; 字符串 → 半角单位数（半角 1、全角 2）
(define is_wide
  (lambda (code)
    (if (>= code 12288) #t
        (if (>= code 8192) #t
            (if (>= code 8592) (<= code 8703) #f)))))

(define str_units
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define u 0)
    (while (< i n) {
      (set! u (+ u (if (is_wide (String.charCodeAt i s)) 2 1)))
      (set! i (+ i 1))
    })
    u))

;; ---------------------------------------------------------------- 小工具

(define f4 (lambda (v) (String.atom_to_string (Math.to_fixed v 4))))

;; 列表 → "(1 2 3)"（display 的输出不能直接拼进字符串）
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

;; 数字列表 → "0 0 1 2 4 5"（屏幕/日志用的紧凑写法）
(define seq_str
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s "")
    (while (< i n) {
      (if (> i 0) (set! s (String.concat s " ")) #f)
      (set! s (String.concat s (String.atom_to_string (get_item lst i))))
      (set! i (+ i 1))
    })
    s))

;; 数字列表 → "351420"（喂给分词器的 6 字符串）
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

;; 两位十进制（分块文件名 m00.b64 …）
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

;; ------------------------------------------------------- 伪随机（LCG，可播种）

;; 为什么不用 Math.random：它的种子在 C 侧一次性来自 time()，App 内无法按需求
;; 「以当前时间戳为随机种子」重播种；这里用状态 < 4093、乘法 < 2^19 的线性同余，
;; 在 wasm32 的单精度下逐位精确（与 app/mlp.scm 同一取舍）。
(define rng 12345)
(define rng_seed
  (lambda (s)
    (set! rng (+ 1 (Math.floor (mod s 4093))))
    rng))
(define rng_next
  (lambda ()
    (set! rng (mod (+ (* 109 rng) 1021) 4093))
    (/ rng 4093.0)))

;; ---------------------------------------------------------------- 模型配置

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
;;   is_causal=#t：因果注意力（自回归生成）；#f：全局注意力（序列生成/排序）
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

;; 多项分布采样（输入须为 softmax 后的概率；随机源是本文件的可播种 LCG）
(define sample_multinomial
  (lambda (probs size)
    (define i 0)
    (define r (rng_next))
    (define cdf 0.0)
    (define ret (- size 1))
    (while (< i size) {
      (set! cdf (+ cdf (get_item probs i)))
      (if (> cdf r) { (set! ret i) (set! i size) } (set! i (+ i 1)))
    })
    ret))

;; 温度采样：0 → 贪心（确定性）；否则 logits/温度 → softmax → 多项分布
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

;; ---------------------------------------------------------- 自检（与本地库比对）

;; 自检基准值（取自上游 (native LLM) 实跑的记录）。张量指纹取「长度 + 第 0/中/末
;; 个元素」而不是求和：wasm32 宿主的浮点是单精度，连加十万项的和会漂到百分级，
;; 单元素则逐位稳定。
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
(define SELF_ENC6 '((17 18 19 20 21 22 23 24 25 26) (27 8 28 8 29)))
(define SELF_TEXTS6 '("0123456789" "a b c"))

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
(define SELF_ENCP '((192 561 187 293 673 190)))
(define SELF_TEXTP '("人类的本质是"))

(define chk_ok 0)
(define chk_bad 0)

;; 浮点近似相等（相对 1e-3 + 绝对 1e-6）：容纳 32 位宿主的单精度表示差异
(define near
  (lambda (a b)
    (define d (Math.abs (- a b)))
    (define rel (* 0.001 (Math.abs b)))
    (define tol (if (> rel 0.000001) rel 0.000001))
    (if (<= d tol) #t #f)))

(define check_eq
  (lambda (tag got want)
    (if (equal? got want)
        (set! chk_ok (+ chk_ok 1))
        {
          (set! chk_bad (+ chk_bad 1))
          (log (String.concat "[chk] NG "
                 (String.concat tag (String.concat " got=" (narr_to_str got)))))
        })))

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
      (log (String.concat "[chk] 分词 "
             (String.concat (get_item texts i) (String.concat " -> " (narr_to_str ids)))))
      (set! i (+ i 1))
    })
    (check_eq "分词 encode/decode（对照本地库）" bad 0)))

;; 载入完成后的自检：返回是否全部通过
(define run_checks
  (lambda (model_id)
    (set! chk_ok 0)
    (set! chk_bad 0)
    (if (== model_id 1)
        {
          (check_cfg SELF_CFG6)
          (check_fp SELF_FP6)
          (check_tokens SELF_TEXTS6 SELF_ENC6)
        }
        {
          (check_cfg SELF_CFGP)
          (check_fp SELF_FPP)
          (check_tokens SELF_TEXTP SELF_ENCP)
        })
    (log (String.concat "[chk] OK=" (String.concat (String.atom_to_string chk_ok)
         (String.concat " NG=" (String.atom_to_string chk_bad)))))
    #t))

;; --------------------------------------------------- 任务：6 元素序列排序（模型）

(define sort_ids (cdr '(0)))
(define sort_out (cdr '(0)))
(define sort_pos 0)
(define sort_prefilled #f)
(define T_SORT 0.8)          ; 每轮的温度（0=贪心；除自检/等价性工具外都>0）

;; 开一轮：把 6 个数字拼成的字符串编码成 token 序列
(define sort_begin
  (lambda (digits)
    (define s (concat_of digits))
    (set! sort_ids (LLM.encode s))
    (set! sort_out (cdr '(0)))
    (set! sort_pos 0)
    s))

;; 预填充：逐位前向，把整段序列的 k/v 填满缓存（全局注意力要用到"未来"的词元）。
;; 注意：k/v 只由 (token, pos) 决定、与被生成的词元无关，故**每轮只需预填充一次**。
(define sort_prefill_step
  (lambda ()
    (llm_forward (get_item sort_ids sort_pos) sort_pos block_size #f)
    (set! sort_pos (+ sort_pos 1))
    (if (>= sort_pos block_size)
        {
          (set! sort_prefilled #t)
          (set! sort_pos 0)
          #t
        }
        #f)))

;; 生成：逐位前向 + 温度采样
(define sort_gen_step
  (lambda ()
    (define logits #f)
    (define new_token 0)
    (define tk "")
    (define v 0)
    (set! logits (llm_forward (get_item sort_ids sort_pos) sort_pos block_size #f))
    (set! new_token (sample_token logits vocab_size T_SORT))
    (set! tk (LLM.decode new_token))
    (set! v (String.parseNumber tk))
    (if (undefined? v) (set! v -1) #f)
    (push sort_out v)
    (set! sort_pos (+ sort_pos 1))
    (if (>= sort_pos block_size) #t #f)))

;; ------------------------------------------- 任务：自回归文本生成（场景 2）

(define N_GEN 64)            ; 生成步数（含预填充的提示词那几步）
(define T_GEN 0.75)          ; 生成温度

(define gen_ids (cdr '(0)))
(define gen_pos 0)
(define gen_next 0)
(define gen_line "")
(define gen_t0 0)
(define gen_n 0)

(define gen_begin
  (lambda (prompt)
    (set! gen_ids (LLM.encode prompt))
    (set! gen_pos 0)
    (set! gen_next (get_item gen_ids 0))
    (set! gen_line "")
    (set! gen_n 0)
    (set! gen_t0 (System.timestamp))
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
          (set! gen_next (sample_token logits vocab_size T_GEN))
          (set! gen_n (+ gen_n 1))
        })
    (set! tk (LLM.decode gen_next))
    (set! gen_line (String.concat gen_line tk))
    (set! gen_pos (+ gen_pos 1))
    (if (String.equals tk "<|eos|>") #t (if (>= gen_pos N_GEN) #t #f))))

;; ---------------------------------------------------------------- 应用状态

(define SC_SORT 1)
(define SC_GEN 2)

(define scenario SC_SORT)
(define model_loaded 0)      ; 0=无 1=sort6 2=psycho
(define loading 0)           ; 1=正在分块载入
(define load_target 0)       ; 正在载入哪个模型
(define chunk_i 0)
(define load_nchunk 0)
(define load_label "")
;; 载入子状态：0=读下一块 1=喂当前块的一小段 2=分块缺失，等资源挂载后重试
;; 为什么不是"一拍喂一整块"：一块 96KB base64 要跑 ~0.4 秒（十几个时间片），
;; 期间只能画一次 → 进度条看起来"不动、然后突然完成"。改成每拍只喂 LOAD_SUB 个字符
;; （12288 是 4 的倍数，解码组不跨段），进度条就能一格一格走，退出键也始终可点。
(define LOAD_SUB 12288)
(define load_stage 0)
(define load_buf "")
(define load_off 0)
(define load_tick 0)
(define wait_said #f)
(define model_str "")        ; 屏幕上显示的模型摘要
(define chk_str "")

;; 触屏（按下沿 → 命中判定）
(define tap_new #f)
(define tap_x 0)
(define tap_y 0)
(define dirty #t)

;; 场景 1 状态
(define s1_stage 0)          ; 0=输入 1=推理 2=结果
(define s1_digits (cdr '(0)))
(define s1_input (cdr '(0)))    ; 6 个数字（提交时的副本）
(define s1_ref (cdr '(0)))      ; 参考排序
(define s1_runs (cdr '(0)))     ; 每轮的 6 个输出
(define s1_round 1)             ; 当前轮号 1..5
(define s1_voted (cdr '(0)))
(define s1_agree 0)
(define s1_diff 0)
(define S1_ROUNDS 5)
(define s1_msg "")

;; 场景 2 状态
(define s2_stage 0)          ; 0=待启动 1=推理中 2=完成
(define s2_prompt "")
(define s2_seed 0)
(define s2_tps 0.0)
(define s2_stopped 0)        ; 1=用户中途点了「停止生成」

;; ---------------------------------------------------------------- 触屏辅助

(define in_rect
  (lambda (px py x y w h)
    (if (>= px x)
        (if (< px (+ x w))
            (if (>= py y) (if (< py (+ y h)) #t #f) #f)
            #f)
        #f)))

;; 键盘第 i 个键的左上角（0..8 = 1..9，9=退格，10=0，11=输入）
(define kp_x
  (lambda (i) (+ KP_X0 (* (- i (* 3 (Math.floor (/ i 3)))) (+ KP_W KP_GX)))))
(define kp_y
  (lambda (i) (+ KP_Y0 (* (Math.floor (/ i 3)) (+ KP_H KP_GY)))))

(define kp_label
  (lambda (i)
    (if (== i 9) "退格"
        (if (== i 10) "0"
            (if (== i 11) "输入" (String.atom_to_string (+ i 1)))))))

;; ---------------------------------------------------------------- 参考排序

(define make_ref
  (lambda (lst)
    (define r (cdr '(0)))
    (define i 0)
    (define j 0)
    (define n 0)
    (define t 0)
    (while (< i (length lst)) { (push r (get_item lst i)) (set! i (+ i 1)) })
    (set! n (length r))
    (set! i 0)
    (while (< i n) {
      (set! j 0)
      (while (< j (- (- n 1) i)) {
        (if (> (get_item r j) (get_item r (+ j 1)))
            {
              (set! t (get_item r j))
              (set_item! r j (get_item r (+ j 1)))
              (set_item! r (+ j 1) t)
            }
            #f)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    r))

;; ---------------------------------------------------- 5 轮结果：逐位投票取多数

;; 第 pos 位的多数值（平票取较小值，保证确定性）
(define vote_at
  (lambda (pos)
    (define vals (cdr '(0)))
    (define i 0)
    (define j 0)
    (define cand 0)
    (define cnt 0)
    (define best 0)
    (define best_n -1)
    (while (< i (length s1_runs)) {
      (push vals (get_item (get_item s1_runs i) pos))
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i (length vals)) {
      (set! cand (get_item vals i))
      (set! cnt 0)
      (set! j 0)
      (while (< j (length vals)) {
        (if (== (get_item vals j) cand) (set! cnt (+ cnt 1)) #f)
        (set! j (+ j 1))
      })
      (if (> cnt best_n)
          { (set! best_n cnt) (set! best cand) }
          (if (== cnt best_n) (if (< cand best) (set! best cand) #f) #f))
      (set! i (+ i 1))
    })
    best))

(define vote_all
  (lambda ()
    (define i 0)
    (set! s1_voted (cdr '(0)))
    (while (< i block_size) {
      (push s1_voted (vote_at i))
      (set! i (+ i 1))
    })
    ;; 与投票结果完全一致（逐位）的轮数
    (set! s1_agree 0)
    (set! i 0)
    (while (< i (length s1_runs)) {
      (if (equal? (get_item s1_runs i) s1_voted) (set! s1_agree (+ s1_agree 1)) #f)
      (set! i (+ i 1))
    })
    ;; 与参考排序之间的位置差异数
    (set! s1_diff 0)
    (set! i 0)
    (while (< i block_size) {
      (if (not (== (get_item s1_voted i) (get_item s1_ref i))) (set! s1_diff (+ s1_diff 1)) #f)
      (set! i (+ i 1))
    })
    ;; 统计行：分段拼接（长 concat 链容易数错括号，这里拆成两条语句）
    (set! s1_msg (String.concat "[sort] voted=" (String.concat (seq_str s1_voted)
                  (String.concat " ref=" (seq_str s1_ref)))))
    (set! s1_msg (String.concat s1_msg (String.concat " agree="
                  (String.concat (String.atom_to_string s1_agree)
                  (String.concat "/" (String.concat (String.atom_to_string S1_ROUNDS)
                  (String.concat " diff=" (String.atom_to_string s1_diff))))))))
    (log s1_msg)
    #t))

;; ---------------------------------------------------------- 载入 / 场景切换

(define begin_load
  (lambda (which)
    {
      (set! load_target which)
      (set! load_nchunk (if (== which 1) M6.nchunk MP.nchunk))
      (set! load_label (if (== which 1) M6.label MP.label))
      (set! chunk_i 0)
      (set! load_stage 0)
      (set! load_buf "")
      (set! load_off 0)
      (set! load_tick 0)
      (set! wait_said #f)
      (set! loading 1)
      ;; 清掉上一个模型的自检摘要，免得载入期间底部还挂着旧结论
      (set! chk_str "")
      (set! dirty #t)
      (LLM.init_begin)
      (log (String.concat "[model] 载入 " load_label))
    }))

;; 载入收尾：init_end → 绑定缓冲区 → 自检
(define finish_load
  (lambda ()
    (if (LLM.init_end)
        {
          (bind_model)
          (set! model_loaded load_target)
          (set! loading 0)
          (set! load_stage 0)
          (set! load_buf "")
          (set! model_str (String.concat (if (== load_target 1) M6.name MP.name)
            (String.concat " · vocab=" (String.concat (String.atom_to_string vocab_size)
            (String.concat " · embd=" (String.atom_to_string n_embd))))))
          (log (String.concat "[model] " (String.concat load_label
               (String.concat " cfg=" (narr_to_str (LLM.get_config))))))
          (run_checks load_target)
          (set! chk_str (String.concat "自检 OK=" (String.concat (String.atom_to_string chk_ok)
                                  (String.concat " NG=" (String.atom_to_string chk_bad)))))
        }
        {
          (set! loading 0)
          (set! load_stage 0)
          (set! load_buf "")
          (set! model_str "模型解析失败")
          (log "[model] 解析失败（分块不全或格式不符）")
        })
    (set! dirty #t)
    #t))

;; 载入进度（0..1）：当前块已喂入的比例按块数折算，进度条因此是连续的
(define load_frac
  (lambda ()
    (define done 0)
    (define blen 0)
    (if (> load_nchunk 0)
        {
          (set! done chunk_i)
          (if (== load_stage 1)
              {
                (set! blen (String.length load_buf))
                (if (> blen 0) (set! done (- chunk_i 1)) #f)
                (if (> blen 0) (set! done (+ done (/ (* 1.0 load_off) blen))) #f)
              }
              #f)
          (/ (* 1.0 done) load_nchunk)
        }
        0.0)))

;; 载入推进（每拍一步）：读块 → 分小段喂流式解码器 → 全部喂完则收尾
(define step_load
  (lambda ()
    (define s #f)
    (define left 0)
    (define take 0)
    (set! dirty #t)
    (cond ((== load_stage 0)
           (if (>= chunk_i load_nchunk)
               (finish_load)
               {
                 (set! s (read_chunk (if (== load_target 1) M6.name MP.name) chunk_i))
                 (if (null? s)
                     {
                       (set! load_stage 2)
                       (set! load_tick 0)
                       (if wait_said #f
                           { (set! wait_said #t)
                             (log "[model] 分块未就绪，稍后自动重试") })
                     }
                     {
                       (set! load_buf s)
                       (set! load_off 0)
                       (set! chunk_i (+ chunk_i 1))
                       (set! load_stage 1)
                     })
               }))
          ((== load_stage 1)
           {
             (set! left (- (String.length load_buf) load_off))
             (set! take (if (> left LOAD_SUB) LOAD_SUB left))
             (LLM.init_feed (String.slice load_buf load_off (+ load_off take)))
             (set! load_off (+ load_off take))
             (if (>= load_off (String.length load_buf))
                 {
                   (set! load_stage 0)
                   (set! load_buf "")
                   (set! load_off 0)
                 }
                 #f)
           })
          ((== load_stage 2)
           {
             (set! load_tick (+ load_tick 1))
             (if (>= load_tick 25)
                 { (set! load_tick 0) (set! load_stage 0) }
                 #f)
           })
          (else #f))))

;; 切换场景（互斥：任一时刻只有一个场景在跑）
(define switch_scenario
  (lambda (n)
    (if (== scenario n)
        #f
        {
          (set! scenario n)
          (set! s1_stage 0)
          (set! s1_digits (cdr '(0)))
          (set! s2_stage 0)
          (set! s2_prompt "")
          (set! s2_tps 0.0)
          (set! dirty #t)
          (log (String.concat "[ui] 切换场景 " (String.atom_to_string n)))
        })))

;; 当前场景需要哪个模型
(define need_model
  (lambda () (if (== scenario SC_SORT) 1 2)))

;; ------------------------------------------------------------ 场景 1 逻辑

;; 提交：6 个数字 → 5 轮推理 → 投票
(define s1_submit
  (lambda ()
    (define i 0)
    (set! s1_input (cdr '(0)))
    (set! i 0)
    (while (< i (length s1_digits)) {
      (push s1_input (get_item s1_digits i))
      (set! i (+ i 1))
    })
    (set! s1_ref (make_ref s1_input))
    (set! s1_runs (cdr '(0)))
    (set! s1_voted (cdr '(0)))
    (set! s1_agree 0)
    (set! s1_diff 0)
    (set! s1_round 1)
    (set! sort_prefilled #f)
    (set! s1_msg "")
    (log (String.concat "[sort] input=" (String.concat (concat_of s1_input)
         (String.concat " ref=" (String.concat (seq_str s1_ref)
         (String.concat " rounds=" (String.atom_to_string S1_ROUNDS)))))))
    (sort_begin s1_input)
    (set! s1_stage 1)
    #t))

;; 键盘点按（tap_x/tap_y 已就绪）
(define s1_keypad_tap
  (lambda ()
    (define i 0)
    (define hitn -1)
    (while (< i 12) {
      (if (in_rect tap_x tap_y (kp_x i) (kp_y i) KP_W KP_H) (set! hitn i) #f)
      (set! i (+ i 1))
    })
    (cond ((== hitn 9)  ; 退格
           (if (> (length s1_digits) 0) (pop s1_digits) #f))
          ((== hitn 11) ; 输入（满 6 位才生效）
           (if (>= (length s1_digits) block_size) (s1_submit) #f))
          ((>= hitn 0)
           (if (== hitn 10)
               (if (< (length s1_digits) block_size) (push s1_digits 0) #f)
               (if (< (length s1_digits) block_size) (push s1_digits (+ hitn 1)) #f)))
          (else #f))
    (set! dirty #t)
    #t))

(define step_s1
  (lambda ()
    (cond ((== s1_stage 0)
           (if tap_new (s1_keypad_tap) #f))
          ((== s1_stage 1)
           (if (not sort_prefilled)
               (sort_prefill_step)
               (if (sort_gen_step)
                   {
                     (push s1_runs sort_out)
                     (log (String.concat "[sort] round "
                           (String.concat (String.atom_to_string s1_round)
                           (String.concat " out=" (seq_str sort_out)))))
                     (set! dirty #t)
                     (if (>= s1_round S1_ROUNDS)
                         {
                           (vote_all)
                           (set! s1_stage 2)
                         }
                         {
                           (set! s1_round (+ s1_round 1))
                           (sort_begin s1_input)
                         })
                   }
                   #f)))
          ((== s1_stage 2)
           (if tap_new
               (if (in_rect tap_x tap_y AGAIN_X AGAIN_Y AGAIN_W AGAIN_H)
                   {
                     (set! s1_stage 0)
                     (set! s1_digits (cdr '(0)))
                     (set! dirty #t)
                   }
                   #f)
               #f))
          (else #f))))

;; ------------------------------------------------------------ 场景 2 逻辑

;; 固定提示词（需求：提示语固定，只有随机种子每次变）
(define GEN_PROMPT "人类的本质")

;; 每次启动/重来都把种子加一个偏移，保证同一秒内连点「重来」也能拿到不同的随机流
(define gen_round 0)

;; (重)开始一次生成：种子取当前时间戳（+ 轮次偏移），提示词固定
(define s2_start
  (lambda ()
    (set! gen_round (+ gen_round 1))
    (set! s2_seed (+ (Time.utc_sod) (* 7919 gen_round)))
    (rng_seed s2_seed)
    (set! s2_prompt GEN_PROMPT)
    (set! s2_tps 0.0)
    (set! s2_stopped 0)
    (gen_begin s2_prompt)
    (set! s2_stage 1)
    (log (String.concat "[gen] seed=" (String.concat (String.atom_to_string s2_seed)
         (String.concat " prompt=" s2_prompt))))
    (set! dirty #t)
    #t))

(define s2_finish
  (lambda (stopped)
    (define el (- (System.timestamp) gen_t0))
    (if (> el 0) (set! s2_tps (/ (* 1.0 gen_n 1000.0) el)) (set! s2_tps 0.0))
    (set! s2_stopped stopped)
    (set! s2_stage 2)
    (log (String.concat "[gen] text=" gen_line))
    (log (String.concat "[gen] tokens=" (String.concat (String.atom_to_string gen_n)
         (String.concat " ms=" (String.concat (String.atom_to_string el)
         (String.concat " tps=" (String.concat (f4 s2_tps)
         (String.concat " stopped=" (String.atom_to_string stopped)))))))))
    (set! dirty #t)
    #t))

(define step_s2
  (lambda ()
    (cond ((== s2_stage 0)
           (if (== model_loaded 2) (s2_start) #f))
          ((== s2_stage 1)
           {
             ;; 生成中可点「停止生成」（保留已生成的部分）或「重来」（换种子重跑）
             (if tap_new
                 {
                   (if (in_rect tap_x tap_y STOP_X STOP_Y STOP_W STOP_H)
                       (s2_finish 1)
                       #f)
                   (if (in_rect tap_x tap_y REDO_X REDO_Y REDO_W REDO_H)
                       (s2_start)
                       #f)
                 }
                 #f)
             (if (== s2_stage 1)
                 (if (gen_step) (s2_finish 0) (set! dirty #t))
                 #f)
           })
          ((== s2_stage 2)
           (if tap_new
               (if (in_rect tap_x tap_y REDO_X REDO_Y REDO_W REDO_H)
                   (s2_start)
                   #f)
               #f))
          (else #f))))

;; ---------------------------------------------------------------- 折行与绘制

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

;; 12px 字一行（在 (x,y,w) 内居中）
(define center_text
  (lambda (s x y w r g b)
    (define u (str_units s))
    (define xx (+ x (Math.floor (/ (- w (* u 6)) 2))))
    (if (< xx (+ x 2)) (set! xx (+ x 2)) #f)
    (Gfx.draw_text s xx y r g b)))

;; 按钮（active=高亮）
(define draw_btn
  (lambda (x y w h label active)
    (if active
        {
          (Gfx.fill_rect x y w h 52 104 196)
          (Gfx.draw_rect x y w h 120 180 255)
        }
        {
          (Gfx.fill_rect x y w h 38 42 56)
          (Gfx.draw_rect x y w h 76 82 98)
        })
    (center_text label x (+ y (Math.floor (/ (- h 12) 2))) w 240 242 250)))

;; 顶部：标题、退出按钮、两个场景标签
(define draw_header
  (lambda ()
    (Gfx.set_font FONT_S)
    (Gfx.fill_rect 0 0 SW HEAD_H 30 34 48)
    (Gfx.draw_text "灵机 LLM 测试台（纯 Scheme）" 6 2 226 232 242)
    (EX.draw_button)
    (draw_btn TAB1_X TAB_Y TAB_W TAB_H "场景1 · LLM 排序" (== scenario SC_SORT))
    (draw_btn TAB2_X TAB_Y TAB_W TAB_H "场景2 · 可视化生成" (== scenario SC_GEN))))

;; 场景 1：6 个数字槽位
(define draw_slots
  (lambda ()
    (define i 0)
    (define x 0)
    (define d #f)
    (while (< i block_size) {
      (set! x (+ SLOT_X0 (* i SLOT_STEP)))
      (Gfx.fill_rect x SLOT_Y SLOT_W SLOT_H 24 28 40)
      (if (== s1_stage 0)
          (Gfx.draw_rect x SLOT_Y SLOT_W SLOT_H 110 150 210)
          (Gfx.draw_rect x SLOT_Y SLOT_W SLOT_H 70 72 88))
      (if (< i (length s1_digits))
          {
            (set! d (get_item s1_digits i))
            (Gfx.set_font FONT_L)
            (Gfx.draw_text (String.atom_to_string d) (+ x 16) (+ SLOT_Y 7) 255 232 120)
            (Gfx.set_font FONT_S)
          }
          #f)
      (set! i (+ i 1))
    })))

;; 场景 1：数字软键盘
(define draw_keypad
  (lambda ()
    (define i 0)
    (while (< i 12) {
      (if (== i 11)
          (draw_btn (kp_x i) (kp_y i) KP_W KP_H (kp_label i)
                    (>= (length s1_digits) block_size))
          (draw_btn (kp_x i) (kp_y i) KP_W KP_H (kp_label i) #f))
      (set! i (+ i 1))
    })))

;; 场景 1：推理进度 / 结果面板
(define draw_result
  (lambda ()
    (define i 0)
    (define y RES_Y)
    (define line "")
    (Gfx.draw_text (String.concat "输入 " (seq_str s1_input)) 10 y 140 200 150)
    (Gfx.draw_text (String.concat "参考 " (seq_str s1_ref)) 170 y 140 200 150)
    (set! y (+ y RES_STEP))
    (set! i 0)
    (while (< i (length s1_runs)) {
      (set! line (String.concat "第 " (String.concat (String.atom_to_string (+ i 1))
                       (String.concat " 轮   " (seq_str (get_item s1_runs i))))))
      (Gfx.draw_text line 10 y 200 206 196)
      (set! y (+ y RES_STEP))
      (set! i (+ i 1))
    })
    (cond ((== s1_stage 1)
           {
             (set! line (String.concat "排序中 " (String.concat (String.atom_to_string s1_round)
                              (String.concat "/" (String.concat (String.atom_to_string S1_ROUNDS)
                              " 轮（温度采样）")))))
             (Gfx.draw_text line 10 y 255 210 120)
           })
          ((== s1_stage 2)
           {
             (set! line (String.concat "投票   " (seq_str s1_voted)))
             (Gfx.draw_text line 10 y 255 232 120)
             (set! y (+ y RES_STEP))
             (set! line (String.concat "一致 " (String.concat (String.atom_to_string s1_agree)
                              (String.concat "/" (String.concat (String.atom_to_string S1_ROUNDS)
                              " 轮，与参考差 ")))))
             (set! line (String.concat line (String.concat (String.atom_to_string s1_diff) " 位")))
             (Gfx.draw_text line 10 y 170 200 210)
             (draw_btn AGAIN_X AGAIN_Y AGAIN_W AGAIN_H "重新输入" #t)
           })
          (else #f))))

(define draw_scenario1
  (lambda ()
    (Gfx.draw_text (String.concat "模型 " model_str) 10 BODY_Y 150 190 160)
    (draw_slots)
    (if (== s1_stage 0) (draw_keypad) (draw_result))))

;; 场景 2：流式文本 + 速度
(define draw_scenario2
  (lambda ()
    (define ls (wrap_lines gen_line))
    (define i 0)
    (define from 0)
    (define y TXT_Y)
    (define line "")
    (Gfx.draw_text (String.concat "种子 " (String.concat (String.atom_to_string s2_seed)
                    (String.concat "   提示 " s2_prompt))) 10 BODY_Y 150 190 160)
    (if (== s2_stage 0)
        (Gfx.draw_text "准备中…" 10 (+ BODY_Y 16) 200 200 200)
        #f)
    ;; 只显示最后 TXT_LINES 行
    (if (> (length ls) TXT_LINES) (set! from (- (length ls) TXT_LINES)) #f)
    (set! i from)
    (while (< i (length ls)) {
      (Gfx.draw_text (get_item ls i) 10 y 232 236 226)
      (set! y (+ y TXT_STEP))
      (set! i (+ i 1))
    })
    ;; 底部状态
    (cond ((== s2_stage 1)
           {
             (set! line (String.concat "推理中 " (String.concat (String.atom_to_string gen_pos)
                             (String.concat "/" (String.concat (String.atom_to_string N_GEN)
                             " 词元（温度采样，逐字流式）")))))
             (Gfx.draw_text line 10 178 255 210 120)
           })
          ((== s2_stage 2)
           {
             (set! line (String.concat (if (== s2_stopped 1) "已停止 · TPS " "TPS ")
                          (String.concat (f4 s2_tps)
                             (String.concat "   " (String.concat (String.atom_to_string gen_n)
                             (String.concat " 词元，用时 " (String.concat (String.atom_to_string
                               (- (System.timestamp) gen_t0)) " ms")))))))
             (Gfx.draw_text line 10 178 255 232 120)
           })
          (else #f))
    (if (== s2_stage 1)
        (draw_btn STOP_X STOP_Y STOP_W STOP_H "停止生成" #f)
        #f)
    (draw_btn REDO_X REDO_Y REDO_W REDO_H "重来" #t)))

;; 载入中的覆盖面板（进度条按「块 + 当前块已喂比例」连续推进）
(define draw_loading
  (lambda ()
    (define w 248)
    (define x 36)
    (define y 96)
    (define frac (load_frac))
    (define bar 0)
    (define txt "")
    (Gfx.fill_rect x y w 70 26 30 44)
    (Gfx.draw_rect x y w 70 90 96 116)
    (Gfx.draw_text "载入模型数据（base64 流式解码）" (+ x 12) (+ y 8) 232 236 246)
    ;; "名字  n/总数 块  46%"
    (set! txt (String.concat load_label (String.concat "   "
                 (String.concat (String.atom_to_string chunk_i)
                 (String.concat "/" (String.concat (String.atom_to_string load_nchunk)
                 (String.concat " 块  " (String.concat
                   (String.atom_to_string (Math.floor (* 100 frac))) "%"))))))))
    (Gfx.draw_text txt (+ x 12) (+ y 26) 170 200 210)
    (Gfx.fill_rect (+ x 12) (+ y 46) (- w 24) 8 50 54 70)
    (set! bar (Math.floor (* frac (- w 24))))
    (if (> bar 0) (Gfx.fill_rect (+ x 12) (+ y 46) bar 8 90 170 255) #f)
    (if (== load_stage 2)
        (Gfx.draw_text "分块尚未就绪，等待资源挂载…" (+ x 12) (+ y 56) 255 190 120)
        #f)))

(define draw_frame
  (lambda ()
    (Gfx.set_font FONT_S)
    (Gfx.fill_rect 0 0 SW SH 14 16 22)
    (draw_header)
    (if (== loading 1)
        (draw_loading)
        (if (== scenario SC_SORT) (draw_scenario1) (draw_scenario2)))
    ;; 底部一行：自检结果（只有载入完成后才有）
    (if (String.equals chk_str "")
        #f
        (Gfx.draw_text chk_str 10 (- SH 14) 110 150 130))
    (Gfx.refresh)
    (set! dirty #f)))

;; ---------------------------------------------------------------- 主循环

(define tick
  (lambda ()
    (define m 0)
    (EX.check)
    ;; 触屏按下沿（EX.check 只在自己的按钮内消费按下沿，其余留给我们）
    (if (Touch.take_press)
        {
          (set! tap_new #t)
          (set! tap_x (Math.floor (Touch.x)))
          (set! tap_y (Math.floor (Touch.y)))
        }
        #f)
    ;; 标签页优先（任何时候都能切）
    (if tap_new
        {
          (if (in_rect tap_x tap_y TAB1_X TAB_Y TAB_W TAB_H) (switch_scenario SC_SORT) #f)
          (if (in_rect tap_x tap_y TAB2_X TAB_Y TAB_W TAB_H) (switch_scenario SC_GEN) #f)
        }
        #f)
    ;; 需要的模型没载入 → 先载入（互斥：模型是全局单例，切场景按需换）
    (set! m (need_model))
    (cond ((== loading 1) (step_load))
          ((not (== model_loaded m)) (begin_load m))
          ((== scenario SC_SORT) (step_s1))
          ((== scenario SC_GEN) (step_s2))
          (else #f))
    (set! tap_new #f)
    (if dirty (draw_frame) #f)))

;; 启动
(rng_seed (+ (Time.utc_sod) 1))
(log "[ui] 灵机 LLM 测试台启动（纯 Scheme 库）")
(EX.install_with draw_frame)
(draw_frame)
(System.set_interval TICK_MS tick)