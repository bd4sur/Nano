;; ============================================================================
;; 灵机 App 网络辅助库（Net）—— 在 (native Net) 之上补"文本协议"这一层
;;
;; 分工：
;;   · C 侧 am_native_Net.c（(native Net)）只做**字节与句柄**：发起请求、
;;     按行取回（(Net.line …)）、查状态/状态码、关闭。它不认识 JSON，也不认识 SSE；
;;   · 本模块做**应用层协议**：JSON 字符串转义、SSE 数据行提取、JSON 字符串字段
;;     取值（含 \uXXXX 解码）、请求体拼装、头表拼装、配置文件读取。
;;
;; 为什么解析放 Scheme 而不是 C：
;;   · 这些是纯文本变换，跟着 App 一起改最省事（换模型/换 API 只改本文件或 App）；
;;   · 逐字符扫一行（几百字符）的开销远小于一次网络往返；
;;   · C 侧一旦做协议编组，就把"某家 API 的字段名"钉进了所有平台共用的本地库。
;;
;; 解释器约束（manual.md，逐条遵守）：
;;   · 算术恒二参数；无一元取负；/ 与 mod 恒返回浮点；下标一律 Math.floor；
;;   · 无 list/let/cond；无 apply；define 无过程简写；**块（{}）内不 define**；
;;   · '() 字面量是模块级静态槽，空表一律用 (cdr '(0)) 现场造新的；
;;   · 字符串字面量只支持 \" \\ \n \t \r 五种转义，没有 \u——JSON 的 \uXXXX 只能靠
;;     String.fromCharCode + hex4 手工解码；
;;   · **每个函数的循环临时量必须是各自独占的模块级变量**：本文件早期版本让两个函数
;;     共用一个 `nf_j`，hex4 把 json_str 的"字符串结束"标志踩掉，表现为"解码到第一个
;;     \uXXXX 就截断"（实测 2 字符 vs 期望 3 字符）。故下面按函数前缀区分：
;;       find→hf_  hex4→hx_  trim→tm_  escape→je_  json_str→js_  config→cf_
;;
;; 【命名约定】本模块的顶层定义一律用**裸名**（find/trim/chat_body/…）；调用方经
;; (import NET …) 后用 NET.xxx 引用。切勿在本文件里写 (define NET.xxx …)——带点号的
;; 标识符会被链接器当作「模块别名/本地库名 + 后缀」的外部引用来解析（manual §7.3），
;; 而本模块内部并没有名为 NET 的别名，于是链接失败（实测：link_fail）。
;;
;; 用法（见 app/remotechat.scm）：
;;   (import NET "app/lib/net.scm")
;;   (define cfg (NET.config "data/remotechat.txt"))       ; #null = 全部用默认值
;;   (define id  (Net.post (NET.opt cfg "url" "/v1/chat/completions")
;;                         (NET.chat_headers (NET.opt cfg "key" "1"))
;;                         (NET.chat_body (NET.opt cfg "model" "DeepSeek-v4.1-Flash") 提示词)))
;;   每拍：(define ln (Net.line id)) → (NET.sse_payload ln) → (NET.sse_delta 载荷)
;; ============================================================================

(native Net)
(native String)
(native Math)
(native Table)
(native Fs)

;; 造一个全新的空表（'() 是模块级静态槽，反复 push 会跨调用累积）
(define new_list (lambda () (cdr '(0))))

;; ============================================================================
;; 字符串小工具
;; ============================================================================

;; (NET.find hay needle from) : Number —— needle 在 hay 中首次出现的下标（从 from 起），
;;   找不到返回 -1。逐字符比较，O((n-m)·m)；SSE/JSON 单行的长度下开销可忽略。
(define hf_i 0)
(define hf_j 0)
(define hf_ok 0)
(define hf_hit -1)
(define find
  (lambda (hay needle from)
    (define hn (String.length hay))
    (define mn (String.length needle))
    (begin
      (set! hf_hit -1)
      (set! hf_i from)
      (while (and (== hf_hit -1) (<= (+ hf_i mn) hn))
        (begin
          (set! hf_j 0)
          (set! hf_ok 1)
          (while (and (== hf_ok 1) (< hf_j mn))
            (begin
              (if (== (String.charCodeAt (+ hf_i hf_j) hay) (String.charCodeAt hf_j needle))
                  (set! hf_j (+ hf_j 1))
                  (set! hf_ok 0))))
          (if (== hf_ok 1) (set! hf_hit hf_i))
          (set! hf_i (+ hf_i 1))))
      hf_hit)))

;; (NET.starts_with s prefix) : Boolean
(define starts_with
  (lambda (s prefix)
    (if (< (String.length s) (String.length prefix))
        #f
        (String.equals (String.slice s 0 (String.length prefix)) prefix))))

;; (NET.is_space cp) : Boolean
(define is_space
  (lambda (cp)
    (or (== cp 32) (== cp 9) (== cp 13) (== cp 10))))

;; (NET.trim s) : String —— 去两端空格/制表/回车/换行
(define tm_a 0)
(define tm_b 0)
(define trim
  (lambda (s)
    (begin
      (set! tm_a 0)
      (set! tm_b (String.length s))
      (while (and (< tm_a tm_b) (is_space (String.charCodeAt tm_a s)))
        (set! tm_a (+ tm_a 1)))
      (while (and (> tm_b tm_a) (is_space (String.charCodeAt (- tm_b 1) s)))
        (set! tm_b (- tm_b 1)))
      (String.slice s tm_a tm_b))))

;; (NET.hex4 s pos) : Number —— 读 s[pos..pos+4) 的 4 位十六进制（非法字符按 0 计）
(define hx_i 0)
(define hx_d 0)
(define hx_v 0)
(define hx_cp 0)
(define hex4
  (lambda (s pos)
    (begin
      (set! hx_v 0)
      (set! hx_i 0)
      (while (< hx_i 4)
        (begin
          (set! hx_cp (String.charCodeAt (+ pos hx_i) s))
          (set! hx_d 0)
          (if (and (>= hx_cp 48) (<= hx_cp 57))  (set! hx_d (- hx_cp 48)))
          (if (and (>= hx_cp 97) (<= hx_cp 102)) (set! hx_d (+ (- hx_cp 97) 10)))
          (if (and (>= hx_cp 65) (<= hx_cp 70))  (set! hx_d (+ (- hx_cp 65) 10)))
          (set! hx_v (+ (* hx_v 16) hx_d))
          (set! hx_i (+ hx_i 1))))
      hx_v)))

;; ============================================================================
;; JSON：转义 / 取字符串字段
;; ============================================================================

;; (NET.json_escape s) : String —— 变成可安全放进 JSON 字符串字面量的内容。
;;   中文等非 ASCII 字符**不转义**（JSON 允许原样 UTF-8，省一大截体积）。
(define je_i 0)
(define je_n 0)
(define je_cp 0)
(define je_out "")
(define json_escape
  (lambda (s)
    (begin
      (set! je_i 0)
      (set! je_n (String.length s))
      (set! je_out "")
      (while (< je_i je_n)
        (begin
          (set! je_cp (String.charCodeAt je_i s))
          (if (== je_cp 34) (set! je_out (String.concat je_out "\\\"")))
          (if (== je_cp 92) (set! je_out (String.concat je_out "\\\\")))
          (if (== je_cp 10) (set! je_out (String.concat je_out "\\n")))
          (if (== je_cp 13) (set! je_out (String.concat je_out "\\r")))
          (if (== je_cp 9)  (set! je_out (String.concat je_out "\\t")))
          (if (and (< je_cp 32) (not (== je_cp 9)) (not (== je_cp 10)) (not (== je_cp 13)))
              (set! je_out (String.concat je_out " ")))
          (if (> je_cp 31)
              (set! je_out (String.concat je_out (String.charAt s je_i))))
          (set! je_i (+ je_i 1))))
      je_out)))

;; (NET.json_str text key) : String | #null
;;   在 text 里找 "key" 之后第一个**字符串值**并解码（\" \\ \/ \n \r \t \b \f \uXXXX）。
;;   找不到、或该字段不是字符串（true/数字/null/对象/数组）→ #null。
(define js_i 0)
(define js_n 0)
(define js_cp 0)
(define js_end 0)
(define js_ok 0)
(define js_a 0)
(define js_b 0)
(define js_out "")
(define json_str
  (lambda (text key)
    (define pat (String.concat (String.concat "\"" key) "\""))
    (begin
      (set! js_n (String.length text))
      (set! js_out #null)
      (set! js_i (find text pat 0))
      (if (not (== js_i -1))
          (begin
            ;; 从 key 之后找第一个引号（跳过冒号与空白）；先遇到 , } ] 说明该字段非字符串
            (set! js_i (+ js_i (String.length pat)))
            (set! js_ok 0)
            (while (and (== js_ok 0) (< js_i js_n))
              (begin
                (set! js_cp (String.charCodeAt js_i text))
                (if (== js_cp 34) (set! js_ok 1))
                (if (or (== js_cp 44) (== js_cp 125) (== js_cp 93)) (set! js_ok 2))
                (set! js_i (+ js_i 1))))
            (if (== js_ok 1)
                (begin
                  ;; js_i 已在开引号之后；逐字符解码到闭引号
                  (set! js_out "")
                  (set! js_end 0)
                  (while (and (== js_end 0) (< js_i js_n))
                    (begin
                      (set! js_cp (String.charCodeAt js_i text))
                      (if (== js_cp 34)
                          (set! js_end 1)
                          (if (== js_cp 92)
                              (begin
                                (set! js_cp (String.charCodeAt (+ js_i 1) text))
                                (if (== js_cp 117)
                                    (begin
                                      (set! js_a (hex4 text (+ js_i 2)))
                                      ;; UTF-16 代理对（\uD8xx\uDCxx）合成一个码点
                                      (if (and (>= js_a 55296) (<= js_a 56319))
                                          (begin
                                            (set! js_b 0)
                                            (if (and (== (String.charCodeAt (+ js_i 6) text) 92)
                                                     (== (String.charCodeAt (+ js_i 7) text) 117))
                                                (set! js_b (hex4 text (+ js_i 8))))
                                            (if (and (>= js_b 56320) (<= js_b 57343))
                                                (begin
                                                  (set! js_a (+ 65536 (+ (* (- js_a 55296) 1024) (- js_b 56320))))
                                                  (set! js_i (+ js_i 6))))))
                                      (set! js_out (String.concat js_out (String.fromCharCode js_a)))
                                      (set! js_i (+ js_i 6)))
                                    (begin
                                      (if (== js_cp 110) (set! js_out (String.concat js_out "\n")))
                                      (if (== js_cp 116) (set! js_out (String.concat js_out "\t")))
                                      (if (== js_cp 114) (set! js_out (String.concat js_out "\r")))
                                      (if (== js_cp 98)  (set! js_out (String.concat js_out (String.fromCharCode 8))))
                                      (if (== js_cp 102) (set! js_out (String.concat js_out (String.fromCharCode 12))))
                                      (if (== js_cp 34)  (set! js_out (String.concat js_out "\"")))
                                      (if (== js_cp 92)  (set! js_out (String.concat js_out "\\")))
                                      (if (== js_cp 47)  (set! js_out (String.concat js_out "/")))
                                      (set! js_i (+ js_i 2)))))
                              (begin
                                (set! js_out (String.concat js_out (String.charAt text js_i)))
                                (set! js_i (+ js_i 1)))))))
                  #f))
            #f))
      js_out)))

;; ============================================================================
;; SSE（OpenAI 兼容的流式对话协议）
;; ============================================================================

;; (NET.sse_payload line) : String | #null
;;   "data: {...}" → "{...}"（已 trim）；其它行（空行、event:/id:）→ #null
(define sse_payload
  (lambda (line)
    (if (starts_with line "data:")
        (trim (String.slice line 5 (String.length line)))
        #null)))

;; (NET.sse_is_end payload) : Boolean —— 载荷是流结束哨兵 "[DONE]"
(define sse_is_end
  (lambda (payload)
    (String.equals (trim payload) "[DONE]")))

;; (NET.sse_delta payload) : String —— 正文增量（"delta":{"content":"…"}），无则空串
(define sse_delta
  (lambda (payload)
    (define v (json_str payload "content"))
    (if (null? v) "" v)))

;; (NET.sse_reason payload) : String —— 思考增量（"delta":{"reasoning_content":"…"}）
(define sse_reason
  (lambda (payload)
    (define v (json_str payload "reasoning_content"))
    (if (null? v) "" v)))

;; (NET.sse_tool payload) : String —— 工具调用增量（流式拼装）：
;;   OpenAI 的 delta.tool_calls[] 里 name 只在首片出现、arguments 是逐段拼的，
;;   这里把每片能读到的部分拼成 "→ 函数名 参数片段"（无工具调用则空串）。
(define sse_tool
  (lambda (payload)
    (define nm (json_str payload "name"))
    (define ar (json_str payload "arguments"))
    (begin
      (if (== (find payload "tool_calls" 0) -1)
          ""
          (begin
            (if (null? nm) (set! nm "") #f)
            (if (null? ar) (set! ar "") #f)
            (if (> (String.length nm) 0)
                ;; 首片：带函数名 → 起一行 "→ 名字 参数…"
                (String.concat (String.concat "→ " nm) (String.concat " " ar))
                ;; 后续片只有参数：直接接在后面（不再重复箭头，否则会画出 "→  → "）
                ar))))))

;; (NET.error_text payload) : String —— 服务端报错时的 "message" 字段（无则空串）
(define error_text
  (lambda (payload)
    (define v (json_str payload "message"))
    (if (null? v) "" v)))

;; ============================================================================
;; 请求编组
;; ============================================================================

;; (NET.header_pair k v) : List —— 两元素列表（Animac 没有点对，也没有 list 过程）
(define header_pair
  (lambda (k v)
    (cons k (cons v (cdr '(0))))))

;; (NET.chat_headers key) : List —— 请求头。
;;   key 非空：标准写法（application/json + Authorization）——浏览器会先发 CORS 预检
;;     （OPTIONS），所以**服务端必须支持 OPTIONS 并回 CORS 头**；
;;   key 为空：降级为"简单请求"（text/plain、不带 Authorization）——不会触发预检，
;;     对"只加 Access-Control-Allow-Origin 但没实现 OPTIONS"的服务端友好（很多网关如此）。
;;     注意 Content-Type 用 text/plain 时服务端通常仍按 JSON 解析请求体（OpenAI 兼容端点
;;     都不看 Content-Type），实测可用。
(define chat_headers
  (lambda (key)
    (define h (new_list))
    (begin
      (if (> (String.length key) 0)
          (begin
            (push h (header_pair "Content-Type" "application/json"))
            (push h (header_pair "Accept" "text/event-stream"))
            (push h (header_pair "Authorization" (String.concat "Bearer " key))))
          (begin
            (push h (header_pair "Content-Type" "text/plain;charset=UTF-8"))
            (push h (header_pair "Accept" "text/event-stream"))))
      h)))

;; (NET.chat_body model prompt) : String —— OpenAI 兼容的单轮流式对话请求体
(define chat_body
  (lambda (model prompt)
    (String.concat
      "{\"model\":\""
      (String.concat model
        (String.concat
          "\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\""
          (String.concat (json_escape prompt) "\"}]}"))))))

;; ============================================================================
;; LLM 配置：**纯文本三行**（与代码彻底解耦——App 里不再有任何默认 URL/模型/密钥）
;;   第 1 行：接口 URL —— **必须是完整的地址**（http://… 或 https://…），**直连 API**。
;;            WebGUI 是无服务端的静态页（不经任何反代），所以能否连通取决于浏览器 + 网络：
;;            API 必须允许跨源（回 Access-Control-Allow-Origin）；Key 非空时还要实现 OPTIONS
;;            预检（Key 留空则客户端走"简单请求"，免预检）。
;;   第 2 行：模型 ID
;;   第 3 行：API Key（可省/可空：留空则不带 Authorization 头，由服务端反代注入）
;;   · 以 # 开头的行按注释跳过；行首尾空白自动去掉；文件末尾的空行不算数；
;;   · 少于此两行（URL/模型）、或 URL 形态不认识 → 返回 #null（App 据此提示怎么改）。
;; ============================================================================

;; (NET.strip_lines lst) : List —— 去掉 # 注释行（保留空行，空行=该字段为空）
(define lc_i 0)
(define lc_s "")
(define strip_lines
  (lambda (lst)
    (define out (new_list))
    (begin
      (set! lc_i 0)
      (while (< lc_i (length lst))
        (begin
          (set! lc_s (get_item lst lc_i))
          (if (starts_with lc_s "#") #f (push out lc_s))
          (set! lc_i (+ lc_i 1))))
      out)))

;; (NET.trim_tail lst) : List —— 去掉末尾的空行（前面的空行保留）
(define trim_tail
  (lambda (lst)
    (define out (new_list))
    (define k 0)
    (begin
      (set! k (length lst))
      (while (and (> k 0) (String.equals (get_item lst (- k 1)) ""))
        (set! k (- k 1)))
      (set! lc_i 0)
      (while (< lc_i k)
        (begin
          (push out (get_item lst lc_i))
          (set! lc_i (+ lc_i 1))))
      out)))

;; (NET.url_ok s) : Boolean —— **只接受完整的 http(s):// 地址**（不再支持 / 开头的相对路径：
;;   相对路径只有浏览器 + 同源反代能用，配置里看不出来，容易误用；统一要求写全）
(define url_ok
  (lambda (s)
    (or (starts_with s "http://") (starts_with s "https://"))))

;; (NET.llm_config path) : List(url model key) | #null
(define lc_n 0)
(define lc_cp 0)
(define lc_cur "")
(define lc_url "")
(define lc_model "")
(define lc_key "")
(define lc_lines #null)
(define llm_config
  (lambda (path)
    (define txt (Fs.read path))
    (begin
      (if (null? txt)
          #null
          (begin
            ;; ① 按 \n 切行（丢掉 \r；不做其它规范化）
            (set! lc_lines (new_list))
            (set! lc_cur "")
            (set! lc_n 0)
            (while (< lc_n (String.length txt))
              (begin
                (set! lc_cp (String.charCodeAt lc_n txt))
                (if (== lc_cp 10)
                    (begin
                      (push lc_lines (trim lc_cur))
                      (set! lc_cur ""))
                    (if (not (== lc_cp 13))
                        (set! lc_cur (String.concat lc_cur (String.charAt txt lc_n)))
                        #f))
                (set! lc_n (+ lc_n 1))))
            (push lc_lines (trim lc_cur))
            ;; ② 去注释行、去末尾空行
            (set! lc_lines (trim_tail (strip_lines lc_lines)))
            ;; ③ 至少要有 URL 与模型两行
            (if (< (length lc_lines) 2)
                #null
                (begin
                  (set! lc_url (get_item lc_lines 0))
                  (set! lc_model (get_item lc_lines 1))
                  (set! lc_key "")
                  (if (> (length lc_lines) 2) (set! lc_key (get_item lc_lines 2)) #f)
                  (if (and (url_ok lc_url) (> (String.length lc_model) 0))
                      (cons lc_url (cons lc_model (cons lc_key (cdr '(0)))))
                      #null))))))))

;; (NET.conf_url c) / (NET.conf_model c) / (NET.conf_key c) : String（c 为 #null 时返回 ""）
(define conf_url
  (lambda (c)
    (if (null? c) "" (get_item c 0))))
(define conf_model
  (lambda (c)
    (if (null? c) "" (get_item c 1))))
(define conf_key
  (lambda (c)
    (if (null? c) "" (get_item c 2))))
