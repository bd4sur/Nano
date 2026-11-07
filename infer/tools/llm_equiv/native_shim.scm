;; ============================================================================
;; native_shim.scm —— 上游 (native LLM) 本地库的「同名封装」
;;
;; 作用：把本地库的接口包装成与 app/lib/llm_pure.scm 完全相同的名字与语义，使同一份
;; 测试脚本可以分别跑在两条实现上，逐行 diff 输出即可证明两者功能等价（见 run.py）。
;;
;; 与纯 Scheme 库的唯一差别在加载方式：上游 init 只接受**整串** base64，故这里用
;; init_begin/init_feed/init_end 先把分块收齐，init_end 时拼成一串再 (LLM.init …)。
;; ============================================================================

(native LLM)
(native String)

(define init (lambda (b64) (LLM.init b64) #t))
(define get_config (lambda () (LLM.get_config)))
(define get_param (lambda () (LLM.get_param)))
(define encode (lambda (text) (LLM.encode text)))
(define decode (lambda (id) (LLM.decode id)))
(define matmul (lambda (xout x w xo wo n d) (LLM.matmul xout x w xo wo n d)))
(define loaded? (lambda () #t))
(define vocab_size (lambda () 0))
(define max_token_len (lambda () 17))

(define chunks (cdr '(0)))
(define init_begin (lambda () { (while (> (length chunks) 0) (pop chunks)) #t }))
(define init_feed (lambda (s) { (push chunks s) #t }))
(define init_end
  (lambda ()
    (define s "")
    (define i 0)
    (while (< i (length chunks)) {
      (set! s (String.concat s (get_item chunks i)))
      (set! i (+ i 1))
    })
    (init s)))
