;@id=remotechat
;@title=远程对话
;@version=2026-11-26
;; ============================================================================
;; 远程对话（RemoteChat）—— 灵机应用程序 · (native Net) 的端到端示例
;;
;; 四个要点：
;;   1) **配置与代码彻底解耦**：App 里没有任何 URL / 模型 / 密钥的默认值，全部来自
;;      纯文本配置文件 data/llm.txt，且**每次发起请求都重新读一遍**（改完文件立刻生效，
;;      不必重启 App，更不必重编固件）：
;;          第 1 行  URL      —— **完整地址**（http:// 或 https:// 开头），直连 API。
;;                              注意：网页是**无服务端的静态页**，网络能力完全取决于浏览器
;;                              与网络环境——API 必须允许跨源（CORS），详见文末说明
;;          第 2 行  模型 ID  —— 如 DeepSeek-v4.1-Flash
;;          第 3 行  API Key  —— 可留空（留空则不带 Authorization，交给同源反代注入）
;;      以 # 开头的行按注释忽略；行首尾空白自动去掉。
;;   2) **原生文本输入控件**：点下方「输入」按钮，借用宿主的模态输入控件
;;      (Ui.input_with …)（全键盘拼音 / 16 键九宫格 / 触屏软键盘，与鹦鹉笼同一套控件），
;;      预填上一次的提示词；取消返回 #null。
;;   3) **全程流式**：思考过程（reasoning_content）、工具调用（tool_calls）、答复正文
;;      （content）三类增量都逐段累积并**即时上屏**——思考区灰色、工具区琥珀色、
;;      正文亮色，正文末尾带「▌」光标，直到流结束。
;;   4) **对话记录不缩进**：续行与首行左对齐（只有首行带「你：」「答：」标签），
;;      右边缘预留标签宽度以免首行被裁。
;;
;; 浏览器直连（本 App 不经任何服务端反代）的两个前提：
;;   · API 必须回 `Access-Control-Allow-Origin`（否则浏览器不让你读响应，连状态码都看不到）；
;;   · 第 3 行 Key 非空 → 会带 Authorization → 浏览器先发 OPTIONS 预检，**服务端必须实现
;;     OPTIONS 并回 CORS 头**；Key 留空 → 客户端改用"简单请求"（text/plain、无自定义头），
;;     不触发预检，对"只加了 ACAO、没实现 OPTIONS"的服务端友好。
;;   页面若是 HTTPS 而 API 是 http://，属混合内容（Chrome 目前只告警仍放行，其它浏览器
;;   可能直接拦）——最稳的做法是静态页也用 http 发布，或给 API 套 TLS。
;;
;; 交互：
;;   · 「输入」按钮   —— 开新话题（进行中则变「停止」，点它中止并关闭连接）
;;   · 配置缺失时按钮变「重试配置」：改好 data/llm.txt 点一下即重读，无需重启
;;   · 右上角「退出」/ 硬 Esc —— app/lib/exit.scm 的全局确认框；确认后先 (Net.close) 再退出
;;
;; 解释器约束（manual.md）：算术恒二参数、无一元取负、/ 与 mod 恒返回浮点（下标
;;   Math.floor）；无 list/let/cond；块（{}）内不 define（循环量一律模块级）；
;;   '() 是静态槽，空表用 (cdr '(0)) 现造；字符串字面量只支持 \" \\ \n \t \r。
;;   计时用**拍数 × 节拍**：32 位宿主的无符号数只有 27 位有效位，毫秒时间戳每 134 秒
;;   回绕一次（见 am_native_Time.h），差值会变成负数。
;; ============================================================================

(native Gfx)
(native Net)
(native String)
(native Math)
(native System)
(native Touch)
(native Ui)       ; 宿主的模态文本输入控件（Ui.input_with）
(native Fs)       ; 记住上次的提示词（data/remotechat_last.txt）
(import EX "app/lib/exit.scm")    ; 硬 A/ESC → 全局退出确认
(import NET "app/lib/net.scm")    ; JSON/SSE 文本协议层 + 三行配置读取
(import BTN "app/lib/button.scm") ; 通用按钮控件（圆角底 + 描边 + 按下高亮）

;; ---- 常量 ----------------------------------------------------------------

(define CONF_PATH "data/llm.txt")
(define LAST_PATH "data/remotechat_last.txt")  ; 上次的提示词（输入框预填）
(define TOP_H 18)
(define CTRL_H 20)      ; 按钮条高度
(define BOT_H 16)       ; 状态条高度
(define BODY_Y 20)
(define LINE_H 14)
(define PAD_X 4)
(define CJK_W 12)
(define ASCII_W 7)
(define TICK_MS 32)
(define MAX_LINES_PER_TICK 16)    ; 每拍最多消费几条 SSE 行（防单拍占满时间片）
(define ANS_MAX 3000)             ; 正文上限（超出只保留最新部分）
(define REASON_MAX 2400)          ; 思考过程上限（同上；思考通常比答复长）
(define TOOL_MAX 800)             ; 工具调用文本上限

;; ---- 状态 ----------------------------------------------------------------

(define SW 320)
(define SH 240)
(define CTRL_Y 204)   ; 按钮条顶边（boot 里按 SH 重算）
(define BODY_BOT 200) ; 正文可用区域底边（CTRL_Y - 4）

(define CONF #null)       ; (url model key)；#null = 配置缺失/写错
(define URL "")
(define MODEL "")
(define KEY "")

(define REQ 0)            ; 请求 id（0 = 无进行中的请求）
(define PHASE 0)          ; 0 空闲 / 1 等待首包 / 2 流式接收 / 3 完成 / 4 失败
(define PROMPT "")        ; 本轮提示词
(define REASON "")        ; 思考过程（流式累积）
(define TOOL "")          ; 工具调用（流式累积）
(define ANS "")           ; 答复正文（流式累积）
(define REASON_N 0)       ; 已收到的思考字符总数（不受上限影响）
(define TOKENS 0)         ; 答复增量块数
(define TICKS 0)          ; 本轮已过的拍数（× TICK_MS = 近似耗时）
(define HTTP 0)           ; HTTP 状态码
(define ERRMSG "")        ; 失败原因
(define STATUS "正在读取配置…")
(define DIRTY 1)

(define TS 0)                 ; 定时器 id
(define PLINES (cdr '(0)))    ; 提示词折行
(define RLINES (cdr '(0)))    ; 思考折行
(define TLINES (cdr '(0)))    ; 工具调用折行
(define ALINES (cdr '(0)))    ; 正文折行
(define LAST_PROMPT "")       ; 上次提交的提示词（输入框预填）
(define BTN_IN #null)         ; 「输入 / 停止 / 重试配置」按钮
(define BTN_LABEL "")         ; 按钮当前标签（变了才 set_label）

;; ---- 折行/绘制用临时量（模块级：块内不 define）---------------------------

(define i 0)
(define n 0)
(define cp 0)
(define cw 0)
(define st 0)
(define curw 0)
(define ln "")
(define payload "")
(define tseg "")
(define delta "")
(define reason "")
(define err "")
(define y 0)
(define avail 0)
(define rmax 0)
(define amax 0)
(define tmax 0)
(define tps 0)
(define hint "")

(define new_list (lambda () (cdr '(0))))
(define cat2 (lambda (a b) (String.concat a b)))

;; 字符步进：CJK/全角 12px，ASCII 7px（与 12px 字模实测步进一致）
(define char_w
  (lambda (code)
    (if (< code 11904) ASCII_W CJK_W)))

;; 避头尾：这些标点不许出现在行首（挂到上一行末尾），中文排版常识
(define no_line_start
  (lambda (code)
    (or (== code 12290) (== code 12289) (== code 65292) (== code 65281)
        (or (== code 65311) (== code 65306) (== code 65307)
            (or (== code 65289) (== code 12301) (== code 12303))))))

;; 追加并裁剪到尾部 maxlen（思考/正文都可能很长，只留最新部分）
(define cap_tail
  (lambda (s x maxlen)
    (define r (cat2 s x))
    (if (> (String.length r) maxlen)
        (String.slice r (- (String.length r) maxlen) (String.length r))
        r)))

;; ---- 折行 ----------------------------------------------------------------

;; 折行：把 s 按宽度 maxw 折成多行，返回新表。
;;   实现用 **String.slice 切行**（每行一次切片），不要逐字符 String.concat——那是
;;   O(n²)，思考过程动辄几千字符、每拍重排，会直接把帧率拖垮。
(define wrap
  (lambda (s maxw)
    (define out (new_list))
    (begin
      (set! i 0)
      (set! st 0)
      (set! curw 0)
      (set! n (String.length s))
      (while (< i n)
        (begin
          (set! cp (String.charCodeAt i s))
          (if (== cp 10)
              (begin
                (push out (String.slice s st i))
                (set! st (+ i 1))
                (set! curw 0))
              (begin
                (set! cw (char_w cp))
                (if (and (> (+ curw cw) maxw) (> i st) (not (no_line_start cp)))
                    (begin
                      (push out (String.slice s st i))
                      (set! st i)
                      (set! curw 0))
                    #f)
                (set! curw (+ curw cw))))
          (set! i (+ i 1))))
      (push out (String.slice s st n))
      out)))

;; 折行宽度：左右各留 PAD_X；首行可能带「你：」「答：」（24px），统一让出，
;; 续行**不缩进**（左对齐），只是右边少用 24px，避免首行贴边被裁。
(define WRAP_W (- (- SW (* 2 PAD_X)) 24))

(define relayout
  (lambda ()
    (begin
      (set! PLINES (wrap PROMPT WRAP_W))
      (set! RLINES (wrap REASON WRAP_W))
      (set! TLINES (wrap TOOL WRAP_W))
      (set! ALINES (wrap ANS WRAP_W)))))

;; ---- 配置 ----------------------------------------------------------------

(define load_conf
  (lambda ()
    (begin
      (set! CONF (NET.llm_config CONF_PATH))
      (set! URL (NET.conf_url CONF))
      (set! MODEL (NET.conf_model CONF))
      (set! KEY (NET.conf_key CONF))
      (set! DIRTY 1))))

;; 上次的提示词：输入框预填，省得每次重新敲
(define load_last
  (lambda ()
    (define t (Fs.read LAST_PATH))
    (if (null? t) #f (set! LAST_PROMPT (NET.trim t)))))

;; ---- 绘制 ----------------------------------------------------------------

;; 配置缺失/写错时的说明（正文区逐行画）
(define draw_conf_help
  (lambda ()
    (begin
      (Gfx.draw_text "配置缺失或格式不对" PAD_X BODY_Y 235 170 120)
      (Gfx.draw_text "请创建 data/llm.txt，三行：" PAD_X (+ BODY_Y LINE_H) 190 200 215)
      (Gfx.draw_text "1  URL（完整地址）例：" PAD_X (+ BODY_Y (* 2 LINE_H)) 170 215 170)
      (Gfx.draw_text "   http://<主机>:11050/v1/chat/completions" PAD_X (+ BODY_Y (* 3 LINE_H)) 140 190 150)
      (Gfx.draw_text "2  模型 ID" PAD_X (+ BODY_Y (* 4 LINE_H)) 170 215 170)
      (Gfx.draw_text "3  API Key（可留空）" PAD_X (+ BODY_Y (* 5 LINE_H)) 170 215 170)
      (Gfx.draw_text "网页直连要求 API 允许跨源（CORS）" PAD_X (+ BODY_Y (* 7 LINE_H)) 150 160 175)
      (Gfx.draw_text "Key 留空可免预检（只发简单请求）" PAD_X (+ BODY_Y (* 8 LINE_H)) 150 160 175)
      (Gfx.draw_text "改好后点下方按钮重读配置" PAD_X (+ BODY_Y (* 10 LINE_H)) 150 160 175))))

;; 从第 nlines 行里取尾部最多 maxn 行，返回起始下标
(define tail_start
  (lambda (nlines maxn)
    (define s 0)
    (begin
      (set! s (- nlines maxn))
      (if (< s 0) (set! s 0) #f)
      s)))

(define draw_all
  (lambda ()
    (begin
      (relayout)
      ;; 背景 + 顶栏
      (Gfx.fill_rect 0 0 SW SH 14 17 22)
      (Gfx.fill_rect 0 0 SW TOP_H 26 34 48)
      (Gfx.draw_text "远程对话" PAD_X 3 235 240 245)
      (if (> (String.length MODEL) 0)
          (Gfx.draw_text MODEL 74 3 130 175 210)
          (Gfx.draw_text "（未配置）" 74 3 190 140 110))
      ;; 正文区
      (if (null? CONF)
          (draw_conf_help)
          (begin
            (set! y BODY_Y)
            ;; ① 提示词（暗色，最多 2 行；首行带「你：」，续行左对齐不缩进）
            (set! i 0)
            (while (and (> (String.length PROMPT) 0) (< i (length PLINES)) (< i 2))
              (begin
                (set! ln (get_item PLINES i))
                (if (== i 0) (set! ln (cat2 "你：" ln)) #f)
                (Gfx.draw_text ln PAD_X y 120 132 150)
                (set! y (+ y LINE_H))
                (set! i (+ i 1))))
            ;; ② 剩余行数：思考/工具区与答复区怎么分
            (set! avail (Math.floor (/ (- BODY_BOT y) LINE_H)))
            (if (< avail 1) (set! avail 1) #f)
            (set! rmax (Math.floor (/ avail 2)))          ; 思考区默认占一半
            (if (== (String.length ANS) 0) (set! rmax avail) #f) ; 还没有正文就全给思考
            (set! tmax (length TLINES))                    ; 工具调用最多 3 行
            (if (> tmax 3) (set! tmax 3) #f)
            (if (> (+ tmax 1) rmax) (set! tmax 0) #f)      ; 空间太挤就只留思考
            ;; ③ 工具调用（琥珀色，尾部对齐）
            (if (> tmax 0)
                (begin
                  (set! i (tail_start (length TLINES) tmax))
                  (while (< i (length TLINES))
                    (begin
                      (Gfx.draw_text (get_item TLINES i) PAD_X y 232 190 110)
                      (set! y (+ y LINE_H))
                      (set! i (+ i 1))))
                  (set! rmax (- rmax tmax)))
                #f)
            ;; ④ 思考过程（灰色，尾部对齐——看最新的思路）
            (if (and (> (String.length REASON) 0) (> rmax 0))
                (begin
                  (set! i (tail_start (length RLINES) rmax))
                  (while (< i (length RLINES))
                    (begin
                      (Gfx.draw_text (get_item RLINES i) PAD_X y 132 138 150)
                      (set! y (+ y LINE_H))
                      (set! i (+ i 1))))
                  (set! DIRTY 1))
                #f)
            ;; ⑤ 答复正文（亮色，尾部对齐；末尾画流式光标）
            (set! amax (Math.floor (/ (- BODY_BOT y) LINE_H)))
            (if (< amax 1) (set! amax 1) #f)
            (if (> (String.length ANS) 0)
                (begin
                  (set! i (tail_start (length ALINES) amax))
                  (while (< i (length ALINES))
                    (begin
                      (set! ln (get_item ALINES i))
                      (if (== i 0) (set! ln (cat2 "答：" ln)) #f)
                      (if (and (> PHASE 0) (< PHASE 3) (== i (- (length ALINES) 1)))
                          (set! ln (cat2 ln "▌"))
                          #f)
                      (Gfx.draw_text ln PAD_X y 226 232 240)
                      (set! y (+ y LINE_H))
                      (set! i (+ i 1))))
                  (set! DIRTY 0))
                (if (and (== (String.length PROMPT) 0) (== (String.length REASON) 0))
                    (begin
                      (Gfx.draw_text "点下方「输入」按钮提问" PAD_X (+ BODY_Y LINE_H) 120 132 150)
                      (Gfx.draw_text "（单轮对话，不保留上下文）" PAD_X (+ BODY_Y (* 2 LINE_H)) 100 112 130))
                    #f))))
      ;; 按钮条
      (Gfx.fill_rect 0 CTRL_Y SW CTRL_H 18 22 30)
      (BTN.draw_state BTN_IN (BTN.armed BTN_IN))
      (set! ln "")
      (if (> (String.length ANS) 0)
          (set! ln (cat2 (String.atom_to_string (String.length ANS)) " 字"))
          #f)
      (if (> TOKENS 0)
          (set! ln (cat2 ln (cat2 " / " (cat2 (String.atom_to_string TOKENS) " 块"))))
          #f)
      (if (> REASON_N 0)
          (set! ln (cat2 ln (cat2 " · 思考 " (String.atom_to_string REASON_N))))
          #f)
      (if (> (String.length ln) 0)
          (Gfx.draw_text ln (+ 74 PAD_X) (+ CTRL_Y 4) 130 150 170)
          #f)
      ;; 状态条
      (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
      (Gfx.draw_text STATUS PAD_X (- SH 14) 150 200 235)
      (EX.draw_button)
      (Gfx.refresh)
      (set! DIRTY 0))))

;; ---- 请求 ----------------------------------------------------------------

;; 把失败原因说人话：0=连不上（地址/同源/反代）；502/503=反代没配或上游不可达
(define fail_text
  (lambda (code)
    (if (== code 0)
        ;; 浏览器直连时 state=0 最常见的原因就是这两条：
        ;;   · API 没开 CORS（跨源被浏览器拦在读响应之前，连状态码都拿不到）
        ;;   · 页面是 HTTPS 而 API 是 http://（混合内容被拦）
        "无法连接（API 未开 CORS？混合内容？）"
        (if (or (== code 502) (== code 503))
            (cat2 "反代未配置 NANO_API_PROXY " (String.atom_to_string code))
            (cat2 "HTTP " (String.atom_to_string code))))))

(define finish
  (lambda ()
    (define elapsed 0)
    (begin
      (set! elapsed (* TICKS TICK_MS))
      (set! tps 0)
      (if (> elapsed 0) (set! tps (Math.floor (/ (* TOKENS 1000) elapsed))) #f)
      (if (== PHASE 4)
          (set! STATUS (cat2 "失败：" ERRMSG))
          (set! STATUS
            (cat2 "完成 "
              (cat2 (String.atom_to_string TOKENS)
                (cat2 " 块 / "
                  (cat2 (String.atom_to_string elapsed)
                    (cat2 " ms / "
                      (cat2 (String.atom_to_string tps) " TPS"))))))))
      (if (not (== REQ 0)) (Net.close REQ) #f)
      (set! REQ 0)
      (set! DIRTY 1))))

;; 处理一条 SSE 数据行（扁平写法：每步一个 if，便于核对括号与阅读）
(define handle_payload
  (lambda (pl)
    (begin
      (set! payload pl)
      ;; ① 流结束哨兵 [DONE]
      (if (NET.sse_is_end payload)
          (begin
            (set! PHASE 3)
            (finish))
          #f)
      ;; ② 服务端错误对象 {"error":{"message":"…"}}
      (if (== REQ 0)
          #f
          (begin
            (set! err (NET.error_text payload))
            (if (> (String.length err) 0)
                (begin
                  (set! ERRMSG err)
                  (set! PHASE 4)
                  (finish))
                #f)))
      ;; ③ 思考过程增量 → 流式显示（灰色区）
      (if (== REQ 0)
          #f
          (begin
            (set! reason (NET.sse_reason payload))
            (if (> (String.length reason) 0)
                (begin
                  (set! REASON_N (+ REASON_N (String.length reason)))
                  (set! REASON (cap_tail REASON reason REASON_MAX))
                  (set! STATUS (cat2 "思考中… " (cat2 (String.atom_to_string REASON_N) " 字")))
                  (set! DIRTY 1))
                #f)))
      ;; ④ 工具调用增量 → 流式显示（琥珀色区）
      (if (== REQ 0)
          #f
          (begin
            (set! tseg (NET.sse_tool payload))
            (if (> (String.length tseg) 0)
                (begin
                  (set! TOOL (cap_tail TOOL tseg TOOL_MAX))
                  (set! STATUS (cat2 "工具调用… " (String.atom_to_string (String.length TOOL))))
                  (set! DIRTY 1))
                #f)))
      ;; ⑤ 答复正文增量 → 流式显示（亮色区）
      (if (== REQ 0)
          #f
          (begin
            (set! delta (NET.sse_delta payload))
            (if (> (String.length delta) 0)
                (begin
                  (set! PHASE 2)
                  (set! TOKENS (+ TOKENS 1))
                  (set! ANS (cap_tail ANS delta ANS_MAX))
                  (set! STATUS (cat2 "接收中… " (cat2 (String.atom_to_string TOKENS) " 块")))
                  (set! DIRTY 1))
                #f)))
      #t)))

;; 每拍：消费若干行（不阻塞；没数据就立刻返回）
;;   【顺序】先把缓冲里的行读完，只有 (Net.line) 返回 #undefined 时才判结束——否则
;;   状态先变 DONE、finish 关连接，缓冲区里剩下的 delta（往往是收尾那几句）就丢了。
;;   【上限】只在还有预算时才再取一行：早期版本先读后判上限，每次撞上限都静默丢掉
;;   一条已取出的行（表现为"回答中间凭空少一段"）。
(define pump
  (lambda ()
    (begin
      (set! TICKS (+ TICKS 1))
      (set! n 0)
      (set! ln (Net.line REQ))
      (while (and (not (undefined? ln)) (< n MAX_LINES_PER_TICK))
        (begin
          (set! payload (NET.sse_payload ln))
          (if (null? payload) #f (handle_payload payload))
          (set! n (+ n 1))
          (if (< n MAX_LINES_PER_TICK) (set! ln (Net.line REQ)) #f)))
      (if (undefined? ln)
          (if (== REQ 0) #f (check_done))
          #f))))

(define check_done
  (lambda ()
    (begin
      (set! HTTP (Net.status REQ))
      (if (Net.error REQ)
          (begin
            (set! ERRMSG (fail_text HTTP))
            (set! PHASE 4)
            (finish))
          (if (Net.done REQ)
              (begin
                (set! PHASE 3)
                (finish))
              #f)))))

;; 发起一轮（每次发起都重读配置：改完 data/llm.txt 立刻生效）
(define start
  (lambda (prompt)
    (begin
      (load_conf)
      (if (null? CONF)
          (begin
            (set! PHASE 4)
            (set! ERRMSG "配置缺失或写错（data/llm.txt）")
            (set! STATUS (cat2 "失败：" ERRMSG))
            (set! DIRTY 1))
          (begin
            (if (not (== REQ 0)) (Net.close REQ) #f)
            (set! REQ 0)
            (set! PROMPT prompt)
            (if (> (String.length prompt) 0) (Fs.write LAST_PATH prompt) #f)
            (set! REASON "")
            (set! TOOL "")
            (set! ANS "")
            (set! REASON_N 0)
            (set! TOKENS 0)
            (set! TICKS 0)
            (set! HTTP 0)
            (set! ERRMSG "")
            (set! PHASE 1)
            (set! STATUS "等待响应…")
            (set! DIRTY 1)
            (set! REQ (Net.post URL (NET.chat_headers KEY) (NET.chat_body MODEL prompt)))
            (if (null? REQ)
                (begin
                  (set! REQ 0)
                  (set! PHASE 4)
                  (set! ERRMSG "请求发起失败（网络不可用，或并发请求已满）")
                  (set! STATUS (cat2 "失败：" ERRMSG))
                  (set! DIRTY 1))
                #f))))))

;; 开新话题：弹**宿主原生**模态输入控件（全键盘拼音 / 九宫格 / 软键盘）
(define ask
  (lambda ()
    (define txt #null)
    (begin
      ;; 弹框前重读"上次提示词"：改过 data/remotechat_last.txt、或换了实例都立刻生效
      (load_last)
      (set! txt (Ui.input_with "输入提示词" LAST_PROMPT))
      (if (null? txt)
          (begin
            (set! STATUS "已取消")
            (set! DIRTY 1))
          (begin
            (set! LAST_PROMPT txt)
            (start txt))))))

;; 按钮动作：进行中 → 停止；配置缺失 → 重读配置；否则 → 输入
(define btn_action
  (lambda ()
    (if (not (== REQ 0))
        (begin
          (Net.close REQ)
          (set! REQ 0)
          (set! PHASE 0)
          (set! STATUS "已停止（可重新输入）")
          (set! DIRTY 1))
        (if (null? CONF)
            (begin
              (load_conf)
              (if (null? CONF)
                  (set! STATUS "仍读不到配置：data/llm.txt")
                  (set! STATUS (cat2 "配置已载入：" MODEL)))
              (set! DIRTY 1))
            (ask)))))

;; 按钮标签：停止 / 重试配置 / 输入（变了才改控件，避免每拍重建描述表）
(define sync_btn_label
  (lambda ()
    (begin
      (set! hint "输入")
      (if (not (== REQ 0)) (set! hint "停止") #f)
      (if (and (== REQ 0) (null? CONF)) (set! hint "重试配置") #f)
      (if (not (String.equals hint BTN_LABEL))
          (begin
            (BTN.set_label BTN_IN hint)
            (set! BTN_LABEL hint)
            (set! DIRTY 1))
          #f))))

;; ---- 主循环 --------------------------------------------------------------

(define on_tick
  (lambda ()
    (begin
      (EX.check)
      (if (not (== REQ 0)) (pump) #f)
      (if (BTN.track BTN_IN) (btn_action) #f)
      (sync_btn_label)
      (if (== DIRTY 1) (draw_all) #f))))

;; ---- 启动 ----------------------------------------------------------------

(define boot
  (lambda ()
    (begin
      (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (set! CTRL_Y (- (- SH BOT_H) CTRL_H))
      (set! BODY_BOT (- CTRL_Y 4))
      (set! BTN_IN (BTN.make "输入" PAD_X (+ CTRL_Y 1) 64 18))
      (load_conf)
      (load_last)
      (if (null? CONF)
          (set! STATUS "配置缺失：data/llm.txt（三行：URL / 模型 / Key）")
          (set! STATUS (cat2 "就绪：点「输入」提问（" (cat2 MODEL "）"))))
      (draw_all)
      (set! TS (System.set_interval TICK_MS on_tick)))))

(boot)

;; ---- 退出：硬 A/ESC → 全局确认框；确认后先 (Net.close) 释放连接再销毁实例 ----
(define on_exit
  (lambda ()
    (begin
      (if (not (== REQ 0)) (Net.close REQ) #f)
      (set! REQ 0))))

(EX.on_exit on_exit)
(EX.install_with draw_all)
