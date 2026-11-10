;; ============================================================================
;; nanochat.scm —— 电子鹦鹉：鹦鹉笼（端侧 LLM）的灵机应用程序
;;
;; 定位：本 App 是「鹦鹉笼」的**灵机（Scheme）实现**——推理全部交给 C 语言写的端侧
;; 推理引擎（infer.c：Q80 量化 前向 / KV-Cache / 采样 / 分词器），经新增的本地库
;; (native Chat) 调用（见 infer/am_native_Chat.h）；本文件只写界面与交互：
;;
;;   · 对话式布局：上面是对话记录（自动滚到最新，可拖拽回看），底部是输入框，
;;     点输入框 → 借用宿主现成的文本控件（全键盘拼音 / 16 键九宫格 / 触屏软键盘，
;;     与 C 版鹦鹉笼同一套），提交即收起键盘 → 文字进入对话记录 → 开始生成；
;;   · 流式输出：一拍一次前向（(Chat.step)），每个词元落地就上屏，
;;     预填充阶段画进度条 + 已读词元数，解码阶段显示词元数 / TPS / 上下文占用；
;;   · 推理原理可视化：右上「可视化」开关。打开后按 **app/llm.scm 场景 2 的布局与
;;     样式**（左列变换块框图 + 层条，右列候选概率柱状图，下方是对话文本）用 Scheme
;;     绘制——本地库只提供状态数据（层/相位、(Chat.top_*) 的 top-6 概率、(Chat.pick_prob)），
;;     不做任何绘制。因为引擎一次前向是原子的（不能中途交还控制权），本地库把这一步
;;     经过的**子步骤序列**记成轨迹（(Chat.trace_len)/(Chat.trace_at)），App 每拍推进
;;     若干项逐帧把"算到哪个部件"点亮，观感与 llm.scm 的「一相位一帧」一致。
;;
;; 为什么要拆成「一拍一步」：一次前向在浏览器 WASM 里要几十~上百毫秒，同步跑完
;; 会让界面整段冻死。宿主每帧只给解释器一个时间片，因此 App 每拍只推进一个词元，
;; 期间照常画界面（与 C 版把 generate_sync 的 while 打开成状态机同构）。
;; 代价与取舍：可视化开启时每步还要花几帧回放相位轨迹（llm.scm 的观感是"一相位一帧"，
;; 更慢），实测 Nano-56M 约 3 词元/秒；关掉可视化则全速（约 25 词元/秒）。
;;
;; 交互一览：
;;   模型选择页：点一行载入模型（载入期间显示进度提示；浏览器目标需先在页面
;;               「资源」面板预载 llm/nano-*.bin，否则列表为空并给出提示）
;;   对话页    ：点底部输入框 → 宿主文本控件（软键盘/拼音）输入 → Enter 提交；
;;               生成中点「停止」中断；「可视化」开关切换推理可视化；对话区上下拖拽回看；
;;               页眉的模型名可点（回模型页并卸载引擎）；右上「退出」
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数、无一元取负；无 let/map/
;; apply；define 只写在模块顶层或 lambda 体顶层（块内不 define）；'() 是模块级静态槽
;; （空表一律用 (cdr '(0)) 现造）；and/or 恒返回布尔；wasm32 上整数只有 27 位有效宽度。
;; ============================================================================

(native Gfx)
(native Chat)
(native Touch)
(native Key)
(native System)
(native String)
(native Math)
(native Ui)
(native Fs)
(native App)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认框 + 右上角「退出」按钮
(import TXT "app/lib/chat_text.scm") ; 文本排版：硬换行(\n) + 软折行 + 控制字符
(import PICK "app/lib/chat_prompts.scm") ; 预置问题（输入为空时随机取一条，同 C 版实现）

;; ---------------------------------------------------------------- 版面常量

(define SW 320)
(define SH 240)
(define FONT_S 1)               ; 12px 抗锯齿
(define LINE_H 12)              ; 正文字行距

(define HEAD_H 16)              ; 页眉
(define CONV_Y 16)              ; 对话区上沿
(define CONV_BOT 200)           ; 对话区下沿（不含）
(define PBAR_Y 201)             ; 预填充进度条（3px）
(define INPUT_Y 205)            ; 输入框
(define INPUT_H 20)
(define STATUS_Y 226)           ; 状态条
(define STATUS_H 14)
(define PAD 4)

;; ---- 可视化（viz=1）：**忠实沿用 app/llm.scm 场景 2 的布局与样式** ----
;; 左列是模型框图（146 宽，纵向 4/5 压缩），右列是候选概率图 + 对话文本。
;; 几何量全部对齐 llm.scm 的同名常量（那边顶栏 17px、正文起点 42，这边页眉 16px，
;; 故整体上移 25px）。
(define VIS_X 0)                ; 框图区（左列）
(define VIS_Y 17)
(define VIS_W 148)
(define VIS_H 183)
(define DG_X 2)                 ; 框图内部坐标原点
(define DG_Y 19)
(define CH_X 152)               ; 候选概率图左界
(define CH_PTR_Y 31)            ; 指针行（选中柱上方的小三角）
(define CH_BASE 71)             ; 柱基线
(define CH_MAXH 30)             ; 最高柱高
(define CH_LBL_Y 73)            ; 词元文本行
(define CH_PCT_Y 85)            ; 概率百分数行
(define TXT_Y_VIS 99)           ; 可视化开时文字列首行（概率图之下）

(define TOG_X 194)              ; 页眉「可视化」开关
(define TOG_Y 1)
(define TOG_W 46)
(define TOG_H 14)

;; 对话区滚动条（右缘）：宽度 4px，轨道高度 = 对话区高度
(define SB_W 4)
(define SB_X (- SW (+ PAD SB_W)))  ; 312..316

(define STOP_W 40)              ; 输入框右侧「停止」按钮
(define STOP_H 16)
(define STOP_X 276)             ; = SW - PAD - STOP_W + 4

;; ---------------------------------------------------------------- 运行参数

(define ST_MODEL 0)             ; 模型选择页
(define ST_CHAT 1)              ; 对话页

;; 生成长度：跑到**模型上下文长度**为止（= max_seq_len − 输入词元数），
;; 或模型自己给出结束词元（EOS）。不设人为的短上限。
(define ctx_limit 512)          ; 上下文上限（载入模型后取模型 max_seq_len）
;; 主循环节拍：可视化开启时**一相位一帧**（20ms ≈ 50 帧/秒，低于 60Hz 合成率，
;; 保证每个相位都真的被画出来、不跳帧——app/llm.scm 的同一取舍）；关闭时用 30ms 尽快推进
;;（生成时每拍跑一整步前向，节拍只影响空闲轮询与 UI 响应）。
(define TICK_VIS 20)
(define TICK_FAST 30)

;; ---------------------------------------------------------------- 模块状态

(define stage ST_MODEL)
(define dirty #t)               ; 需要重绘

;; 模型
(define models (cdr '(0)))      ; ((名称 文件) ...)
(define sel 0)                  ; 选中的模型下标
(define model_name "-")         ; 已载入模型的显示名
(define model_str "")           ; 页眉上的模型摘要
(define n_layer 0)
(define vocab_size 0)
(define ctx_max 0)              ; 模型 max_seq_len（(Chat.config) 第 10 项）
(define loading 0)              ; 1=正在载入（阻塞调用前置位，用于先画出提示）

;; 对话（两条平行表：角色 1=用户 2=助手；文本）
(define c_role (cdr '(0)))
(define c_text (cdr '(0)))

;; 折行缓存（避免每帧重排整段对话）
(define ln_text (cdr '(0)))
(define ln_kind (cdr '(0)))     ; 1=用户标签 2=助手标签 3=用户正文 4=助手正文
(define ln_dirty 1)
(define scroll 0)               ; 距最新一行的上滚行数（0=看最新）

;; 生成
(define gen 0)                  ; 1=正在生成
(define gen_text "")            ; 本轮流式文本（(Chat.output)）
(define gen_tokens 0)
(define gen_tps 0)
(define state_code 0)           ; 最近一次 (Chat.step) 的返回值
(define pfill_pos 0)
(define pfill_tot 0)
(define prefill_ms 0)
(define gen_pick -1)
(define gen_pick_p 0.0)         ; 该词元在本步分布下的概率（(Chat.pick_prob)）
(define gen_msg "")             ; 状态条文案（非生成时）

;; 可视化
(define viz 0)                  ; 1=显示推理原理可视化（默认关闭：先把对话跑快；
                                ; 需要看推理过程时点页眉「可视化」）
(define top_texts (cdr '(0)))
(define top_probs (cdr '(0)))
(define top_ids (cdr '(0)))
(define obs_layer -1)             ; 当前观测层（框图/层条高亮用）
(define obs_phase 0)              ; 当前观测相位
;; 相位回放（引擎一步前向内部会经过上百个子步骤，本地库把这些子步骤记成轨迹，
;; App 每拍推进若干项、逐帧把"算到哪个部件"点出来——与 llm.scm 的「一相位一帧」同观感）
(define obs_ob #f)                 ; (Chat.obs) 的临时取值
(define anim_on 0)                ; 1=正在回放上一拍的轨迹
(define anim_i 0)                 ; 回放游标
(define anim_len 0)               ; 轨迹长度
(define anim_announced 0)         ; 本轮是否已打印过轨迹信息（诊断用）

;; 输入 / 触屏
(define tap_new #f)
(define tap_x 0)
(define tap_y 0)
(define drag_on 0)              ; 1=本次触摸序列处于拖拽滚动
(define drag_y0 0)
(define drag_scroll0 0)

;; 日志
(define LOG_PATH "data/nanochat.log")

;; ---------------------------------------------------------------- 小工具

(define log
  (lambda (s)
    {
      (display s)
      (newline)
    }))

(define logln                    ; 日志前缀统一（便于端到端测试断言）
  (lambda (s)
    (log (String.concat "[nanochat] " s))))

(define max2
  (lambda (a b) (if (> a b) a b)))

;; 整数 → 字符串
(define i2s
  (lambda (v) (String.atom_to_string (Math.floor v))))

;; 一位小数（TPS 显示）
(define f1
  (lambda (v) (String.atom_to_string (Math.to_fixed v 1))))

(define in_rect
  (lambda (px py x y w h)
    (if (and (>= px x) (< px (+ x w)))
        (if (and (>= py y) (< py (+ y h))) 1 0)
        0)))

;; 对话区几何：可视化开时让出左列（框图）与右列上半（概率图），只占右列下半
(define conv_x
  (lambda () (if (== viz 1) CH_X PAD)))

(define conv_y0
  (lambda () (if (== viz 1) TXT_Y_VIS CONV_Y)))

;; 对话区可用宽度（半角单位数；右侧让出 10px 给滚动条）与可显示行数
(define conv_units
  (lambda () (Math.floor (/ (- (- SW (conv_x)) 10) 6))))

(define conv_lines
  (lambda () (Math.floor (/ (- CONV_BOT (conv_y0)) LINE_H))))

;; 折行：把 s 切成若干行追加进 ln_text/ln_kind。
;; 硬换行（'\n'）与软折行都在 app/lib/chat_text.scm 里做——模型输出里的换行符必须
;; 真的换行（12px 字模没有 '\n' 字形，整段直接 draw_text 会"什么也不画"）。
(define wrap_push
  (lambda (s kind maxu)
    (define ls (TXT.wrap_lines s maxu))
    (define n (length ls))
    (define i 0)
    (while (< i n) {
      (push ln_text (get_item ls i))
      (push ln_kind kind)
      (set! i (+ i 1))
    })
    #t))

;; 重建折行缓存（内容/宽度变化时才调用）
(define build_lines
  (lambda ()
    (define i 0)
    (define n (length c_text))
    (define mu (conv_units))
    (set! ln_text (cdr '(0)))
    (set! ln_kind (cdr '(0)))
    (while (< i n) {
      ;; 每条记录之前留一个空行（第一条除外），把相邻的问答隔开
      (if (> i 0)
          {
            (push ln_text "")
            (push ln_kind 0)
          }
          #f)
      (if (== (get_item c_role i) 1)
          {
            (push ln_text "我")
            (push ln_kind 1)
            (wrap_push (get_item c_text i) 3 mu)
          }
          {
            (push ln_text "Nano")
            (push ln_kind 2)
            (wrap_push (get_item c_text i) 4 mu)
          })
      (set! i (+ i 1))
    })
    ;; 生成中的流式文本也排进来（它此刻还没落到对话表里）
    (if (== gen 1)
        {
          ;; 正在生成的这一条同样与上一条隔一个空行
          (if (> (length c_text) 0)
              {
                (push ln_text "")
                (push ln_kind 0)
              }
              #f)
          (push ln_text "Nano")
          (push ln_kind 2)
          (if (String.equals gen_text "")
              {
                (push ln_text "…")
                (push ln_kind 4)
              }
              (wrap_push gen_text 4 mu))
        }
        #f)
    (set! ln_dirty 0)
    #t))

;; 对话行数
(define line_num
  (lambda ()
    (if (== ln_dirty 1) (build_lines) #f)
    (length ln_text)))

;; 折行缓存里"空行（记录分隔）"的条数——诊断用（端到端测试据此断言记录之间有间隔）
(define blank_num
  (lambda ()
    (define i 0)
    (define k 0)
    (while (< i (length ln_kind)) {
      (if (== (get_item ln_kind i) 0) (set! k (+ k 1)) #f)
      (set! i (+ i 1))
    })
    k))

;; ---------------------------------------------------------------- 绘制：公共件

(define draw_bar
  (lambda ()
    {
      (Gfx.set_font FONT_S)
      (Gfx.fill_rect 0 0 SW HEAD_H 28 32 44)
      (Gfx.draw_text "电子鹦鹉" PAD 2 226 234 246)
      (Gfx.draw_text model_str 66 2 128 168 200)
    }))

;; 页眉右侧「可视化」开关
(define draw_toggle
  (lambda ()
    (if (== viz 1)
        {
          (Gfx.fill_rect TOG_X TOG_Y TOG_W TOG_H 36 92 150)
          (Gfx.draw_rect TOG_X TOG_Y TOG_W TOG_H 110 170 240)
          (Gfx.draw_text "可视化" (+ TOG_X 4) (+ TOG_Y 2) 235 242 250)
        }
        {
          (Gfx.fill_rect TOG_X TOG_Y TOG_W TOG_H 44 46 58)
          (Gfx.draw_rect TOG_X TOG_Y TOG_W TOG_H 96 100 116)
          (Gfx.draw_text "可视化" (+ TOG_X 4) (+ TOG_Y 2) 150 156 172)
        })))

;; 底栏输入框（生成中变成「状态 + 停止」）
(define draw_input
  (lambda ()
    (define bw (- SW (* PAD 2)))
    (if (== gen 1)
        {
          (Gfx.fill_rect PAD INPUT_Y bw INPUT_H 40 30 34)
          (Gfx.draw_rect PAD INPUT_Y bw INPUT_H 150 90 90)
          (Gfx.draw_text (String.concat "生成中… " (String.concat (i2s gen_tokens) " 词元"))
                         (+ PAD 6) (+ INPUT_Y 4) 230 170 150)
          (Gfx.fill_rect STOP_X (+ INPUT_Y 2) STOP_W STOP_H 180 60 60)
          (Gfx.draw_rect STOP_X (+ INPUT_Y 2) STOP_W STOP_H 250 130 120)
          (Gfx.draw_text "停止" (+ STOP_X 8) (+ INPUT_Y 5) 250 235 230)
        }
        {
          (Gfx.fill_rect PAD INPUT_Y bw INPUT_H 34 38 50)
          (Gfx.draw_rect PAD INPUT_Y bw INPUT_H 86 96 122)
          (Gfx.draw_text "点这里输入消息…" (+ PAD 6) (+ INPUT_Y 4) 132 140 158)
          (Gfx.draw_text "Enter" (- (- SW PAD) 40) (+ INPUT_Y 4) 96 104 124)
        })))

;; 底栏状态条：预填充进度 / TPS / 提示
(define draw_status
  (lambda ()
    (define txt "")
    (Gfx.fill_rect 0 STATUS_Y SW STATUS_H 16 18 24)
    (cond ((== gen 1)
           (if (< pfill_pos pfill_tot)
               {
                 (set! txt (String.concat "预填充 " (i2s pfill_pos)))
                 (set! txt (String.concat txt "/"))
                 (set! txt (String.concat txt (i2s pfill_tot)))
                 (set! txt (String.concat txt " 词元 · "))
                 (set! txt (String.concat txt (f1 (/ prefill_ms 1000.0))))
                 (set! txt (String.concat txt "s"))
                 ;; 可视化开启时预填充也逐相位回放（慢），这里明示一下，免得以为卡住
                 (if (== viz 1)
                     (set! txt (String.concat txt " · 逐相位回放"))
                     #f)
                 (Gfx.draw_text txt PAD (+ STATUS_Y 1) 130 200 240)
               }
               {
                 (set! txt (String.concat "生成 " (i2s gen_tokens)))
                 (set! txt (String.concat txt " 词元 · "))
                 (set! txt (String.concat txt (f1 gen_tps)))
                 (set! txt (String.concat txt " 词/秒"))
                 ;; 相位名（与 llm.scm 的状态行同义：正在算哪一步）
                 (if (== viz 1)
                     {
                       (set! txt (String.concat txt " · "))
                       (set! txt (String.concat txt (ph_name obs_phase)))
                       (set! txt (String.concat txt " "))
                       (set! txt (String.concat txt (i2s (+ obs_layer 1))))
                     }
                     {
                       (set! txt (String.concat txt " · 上下文 "))
                       (set! txt (String.concat txt (i2s pfill_pos)))
                       (set! txt (String.concat txt "/"))
                       (set! txt (String.concat txt (i2s ctx_limit)))
                     })
                 (Gfx.draw_text txt PAD (+ STATUS_Y 1) 140 220 170)
               }))
          (else
           (Gfx.draw_text gen_msg PAD (+ STATUS_Y 1) 132 140 156)))))

;; 预填充进度条（输入框上方 3px）
(define draw_pbar
  (lambda ()
    (define w 0)
    (define frac 0.0)
    (if (== gen 1)
        (if (> pfill_tot 0)
            (if (< pfill_pos pfill_tot)
                {
                  (Gfx.fill_rect PAD PBAR_Y (- SW (* PAD 2)) 3 40 46 60)
                  (set! frac (/ pfill_pos (* 1.0 pfill_tot)))
                  (set! w (Math.floor (* frac (- SW (* PAD 2)))))
                  (if (> w 1) (Gfx.fill_rect PAD PBAR_Y w 3 90 180 250) #f)
                }
                #f)
            #f)
        #f)))

;; ---------------------------------------------------------------- 绘制：对话区

(define draw_conv_line
  (lambda (s kind x y)
    (cond ((== kind 1) (Gfx.draw_text s x y 110 170 250))
          ((== kind 2) (Gfx.draw_text s x y 110 226 160))
          ((== kind 3) (Gfx.draw_text s x y 230 236 244))
          ((== kind 0) #f)                       ; 空行（记录分隔）
          (else        (Gfx.draw_text s x y 206 244 220)))))

;; 对话区滚动条：滑块长度 ∝ 可视行/总行，位置 ∝ 当前滚动；内容或滚动一变就重画
;; （draw_conv 每帧都按最新的 (line_num) 与 scroll 计算，因此随生成/滚动即时刷新）
(define draw_conv_scrollbar
  (lambda ()
    (define n (line_num))
    (define vis (conv_lines))
    (define y0 (conv_y0))
    (define track (- CONV_BOT y0))
    (define maxs (- n vis))
    (define th 0)
    (define ty 0)
    (define frac 0.0)
    (if (> n vis)
        {
          ;; 轨道
          (Gfx.fill_rect SB_X y0 SB_W track 38 42 54)
          ;; 滑块（太短时保底 8px，保证可点可见）
          (set! th (Math.floor (/ (* track vis) (+ n 0.0))))
          (if (< th 8) (set! th 8) #f)
          (if (> th track) (set! th track) #f)
          ;; scroll=0 看最新 → 滑块贴底；scroll=maxs 看最早 → 滑块贴顶
          (set! frac (/ (- maxs scroll) (* 1.0 maxs)))
          (if (< frac 0.0) (set! frac 0.0) #f)
          (if (> frac 1.0) (set! frac 1.0) #f)
          (set! ty (+ y0 (Math.floor (* (- track th) frac))))
          (Gfx.fill_rect SB_X ty SB_W th 122 134 154)
          (Gfx.draw_rect SB_X ty SB_W th 150 164 186)
        }
        #f)))

(define draw_conv
  (lambda ()
    (define n (line_num))
    (define vis (conv_lines))
    (define from 0)
    (define i 0)
    (define y (conv_y0))
    (define x (conv_x))
    ;; 滚动的合法范围：0（最新）…n-vis（最旧）
    (if (> scroll (- n vis)) (set! scroll (max2 0 (- n vis))) #f)
    (if (< scroll 0) (set! scroll 0) #f)
    (if (> n vis) (set! from (- (- n vis) scroll)) (set! from 0))
    (if (== n 0)
        {
          (Gfx.draw_text "点下面的输入框" x (+ (conv_y0) 8) 140 150 168)
          (Gfx.draw_text "用软键盘拼音输入" x (+ (conv_y0) 24) 140 150 168)
          (Gfx.draw_text "提交后逐词元流式作答" x (+ (conv_y0) 40) 140 150 168)
          (Gfx.draw_text "右上「可视化」开关" x (+ (conv_y0) 56) 120 140 160)
        }
        {
          (set! i from)
          (while (and (< i n) (< y (+ CONV_BOT (- LINE_H 2)))) {
            (draw_conv_line (get_item ln_text i) (get_item ln_kind i) x y)
            (set! y (+ y LINE_H))
            (set! i (+ i 1))
          })
          ;; 还有更多旧消息时给一个提示（列首小三角）
          (if (> from 0) (Gfx.draw_text "^" (+ (conv_x) (- (- SW (conv_x)) 8)) (+ (conv_y0) 1) 120 150 190) #f)
        })
    (draw_conv_scrollbar)))

;; ---------------------------------------------------------------- 绘制：可视化
;;
;; 本段**逐行移植 app/llm.scm 场景 2**（模型框图 dg_* / vis_draw + 候选概率图
;; draw_prob_chart / draw_pointer + 层条 dg_layers + 相位名 ph_name），只把坐标原点
;; 换成本 App 的 VIS_X/VIS_Y/CH_*，以及数据来源从"纯 Scheme 前向"换成本地库
;; (Chat.obs)/(Chat.top_*)/(Chat.info) 给出的状态。样式（配色、方框/圆/连线拓扑、
;; 4/5 纵向压缩、层条位置、柱状图配色与指针三角形）与 llm.scm 完全一致。
;;
;; 上游 ui_llm.c 的 ui_app_llm_model_diagram_draw 在推理过程中反复重画：
;;   ①左侧是 Nano 变换块的结构框图（残差 ×2、Attn Norm、Q/K/V、RoPE、Q*K、
;;     Mask→softmax→A*V、O、FFN Norm、W1/W3/SiLU、⊗、W2），②中间一列层条（当前层青色），
;;   ③右侧 top-6 候选词元。相位（NANO_LLM_PHASE_*）命中哪个部件就把哪个部件点亮成绿色。

;; 相位编号（infer.h 的 NANO_LLM_PHASE_*）
(define PH_EMBED 1)
(define PH_ATTN_NORM 2)
(define PH_QKV 3)
(define PH_QK_ROPE 4)
(define PH_MHA 5)
(define PH_O 6)
(define PH_FFN_NORM 7)
(define PH_W1W3 8)
(define PH_W2 9)
(define PH_FINAL_NORM 10)
(define PH_CLASSIFY 11)
(define PH_SAMPLE 12)

;; 框图内部坐标 → 屏幕坐标（纵向 4/5 压缩）
(define dgx (lambda (dx) (+ DG_X dx)))
(define dgy (lambda (dy) (+ DG_Y (Math.floor (/ (* dy 4) 5)))))
(define dgh (lambda (h) (Math.floor (/ (* h 4) 5))))

(define dg_line
  (lambda (ax ay bx by r g b)
    (Gfx.draw_line (dgx ax) (dgy ay) (dgx bx) (dgy by) r g b)))

;; 30x14 的方框（内含 12px 文字居中）；active=当前相位命中 → 绿色填充 + 白字
(define dg_box
  (lambda (dx dy label phase)
    (define x (dgx dx))
    (define y (dgy dy))
    (define hh (dgh 14))
    (define u (TXT.units label))
    (define active (== obs_phase phase))
    (if active
        {
          (Gfx.fill_rect x y 30 hh 0 150 60)
          (Gfx.draw_rect x y 30 hh 130 255 150)
        }
        {
          (Gfx.fill_rect x y 30 hh 48 50 62)
          (Gfx.draw_rect x y 30 hh 86 90 104)
        })
    (Gfx.draw_text label (+ x (Math.floor (/ (- 30 (* u 6)) 2)))
                   (+ y (Math.floor (/ (- hh 12) 2)))
                   (if active 255 255 255) (if active 255 255 255) 255)))

;; 圆 + 叉（⊗：Hadamard / Q*K / A*V）
(define dg_circle
  (lambda (cx cy phase)
    (define active (== obs_phase phase))
    (define x (dgx cx))
    (define y (dgy cy))
    (if active
        (Gfx.fill_circle x y 6 0 150 60)
        (Gfx.fill_circle x y 6 48 50 62))
    (Gfx.draw_line (- x 4) (- y 4) (+ x 4) (+ y 4) 20 22 30)
    (Gfx.draw_line (+ x 4) (- y 4) (- x 4) (+ y 4) 20 22 30)))

;; 层条（n_layer + 2：顶层=分类器，底层=embedding；当前层青色）
(define dg_layers
  (lambda ()
    (define n (+ n_layer 2))
    (define span (- (+ DG_Y 178) DG_Y))
    (define lh (Math.floor (/ span (* 2 n))))
    (define dy (Math.floor (/ (- span lh) (- n 1))))
    (define y DG_Y)
    (define ll (- n 1))
    (define sx (+ DG_X 125))          ; 小方块的水平中心
    ;; 脊线：一条竖线把各层小方块串起来（方块随后画在线上，缝隙处就露出这段直线）
    (Gfx.draw_line sx DG_Y sx (+ DG_Y (+ (* dy (- n 1)) lh)) 104 108 126)
    (while (> ll -1) {
      (if (or (and (== obs_layer -1) (== ll 0))
              (and (== obs_layer n_layer) (== ll (- n 1)))
              (== ll (+ obs_layer 1)))
          {
            (Gfx.fill_rect (+ DG_X 120) y 10 lh 0 220 220)
          }
          (Gfx.fill_rect (+ DG_X 120) y 10 lh 48 50 62))
      (set! y (+ y dy))
      (set! ll (- ll 1))
    })))

;; 相位名（状态行用）
(define ph_name
  (lambda (p)
    (cond ((== p PH_EMBED) "embed")
          ((== p PH_ATTN_NORM) "attn norm")
          ((== p PH_QKV) "q/k/v")
          ((== p PH_QK_ROPE) "rope")
          ((== p PH_MHA) "attn")
          ((== p PH_O) "wo")
          ((== p PH_FFN_NORM) "ffn norm")
          ((== p PH_W1W3) "w1/w3")
          ((== p PH_W2) "w2")
          ((== p PH_FINAL_NORM) "final norm")
          ((== p PH_CLASSIFY) "classify")
          ((== p PH_SAMPLE) "sample")
          (else ""))))

;; 画整张框图（先清左列）
(define vis_draw
  (lambda ()
    (Gfx.fill_rect VIS_X VIS_Y VIS_W VIS_H 14 16 22)
    ;; ---- 连线（自下而上：Attn Norm → QKV → RoPE → Q*K → A*V → O → Norm → W1/W3 → ⊗ → W2）----
    (dg_line 55 14 55 50 150 150 150)
    (dg_line 15 50 95 50 150 150 150)
    (dg_line 15 79 95 79 150 150 150)
    (dg_line 15 50 15 79 150 150 150)
    (dg_line 95 50 95 79 150 150 150)
    (dg_line 55 79 55 133 150 150 150)
    (dg_line 35 133 95 133 150 150 150)
    (dg_line 35 133 35 160 150 150 150)
    (dg_line 95 133 95 203 150 150 150)
    (dg_line 15 160 55 160 150 150 150)
    (dg_line 15 203 15 160 150 150 150)
    (dg_line 55 160 55 227 150 150 150)
    (dg_line 15 203 95 203 150 150 150)
    ;; 残差支路（X1 / X2）
    (dg_line 36 103 55 103 150 150 150)
    (Gfx.draw_text "X1" (dgx 22) (- (dgy 103) 6) 150 150 150)
    (dg_line 36 20 55 20 150 150 150)
    (Gfx.draw_text "X2" (dgx 22) (- (dgy 20) 6) 150 150 150)
    ;; ---- 部件 ----
    (dg_box 40 25 "W2" PH_W2)
    (dg_circle 55 50 PH_W2)              ; FFN 的 ⊗（与 W2 同相位，同上游）
    (dg_box 0 61 "W1" PH_W1W3)
    (dg_box 80 43 "SiLU" PH_W1W3)
    (dg_box 80 61 "W3" PH_W1W3)
    (dg_box 40 83 "Norm" PH_FFN_NORM)
    (dg_box 40 109 "O" PH_O)
    ;; Mask（12x12 小方块 + 斜线）与 A*V / Q*K
    (if (== obs_phase PH_MHA)
        {
          (Gfx.fill_rect (dgx 29) (dgy 138) 12 (dgh 12) 0 150 60)
        }
        {
          (Gfx.fill_rect (dgx 29) (dgy 138) 12 (dgh 12) 48 50 62)
        })
    (Gfx.draw_line (dgx 29) (dgy 138) (+ (dgx 29) 11) (+ (dgy 138) (dgh 12)) 20 22 30)
    (dg_circle 55 133 PH_MHA)            ; A*V
    (dg_circle 35 160 PH_MHA)            ; Q*K
    (dg_box 0 168 "RoPE" PH_QK_ROPE)
    (dg_box 40 168 "RoPE" PH_QK_ROPE)
    (dg_box 0 185 "Q" PH_QKV)
    (dg_box 40 185 "K" PH_QKV)
    (dg_box 80 185 "V" PH_QKV)
    (dg_box 40 207 "Norm" PH_ATTN_NORM)  ; Attn Norm（数据入口）
    ;; ---- 层条 ----
    (dg_layers)
    #t))

;; 指针（选中柱上方的小三角，逐行加宽的橙色横条）
(define draw_pointer
  (lambda (cx ytop)
    (Gfx.fill_rect (- cx 4) ytop 9 2 255 170 60)
    (Gfx.fill_rect (- cx 3) (+ ytop 2) 7 2 255 170 60)
    (Gfx.fill_rect (- cx 2) (+ ytop 4) 5 2 255 170 60)
    (Gfx.fill_rect (- cx 1) (+ ytop 6) 3 2 255 170 60)
    (Gfx.fill_rect cx (+ ytop 8) 1 2 255 170 60)
    #t))

;; 本次采样的词元是否已在 top-K 里
(define in_obs_ids
  (lambda (id)
    (define i 0)
    (define hit #f)
    (while (< i (length top_ids)) {
      (if (== (get_item top_ids i) id) (set! hit #t) #f)
      (set! i (+ i 1))
    })
    hit))

;; 候选词元采样概率柱状图（样式与 app/llm.scm 的 draw_prob_chart 一致）
(define draw_prob_chart
  (lambda ()
    (define n (length top_probs))
    (define m 0)
    (define colw 0)
    (define bw 0)
    (define pmax 0.000001)
    (define i 0)
    (define p 0.0)
    (define h 0)
    (define x 0)
    (define cx 0)
    (define id 0)
    (define s "")
    (define sel #f)
    (define pct 0)
    (define extra #f)
    ;; 采样选中的词元有可能不在 top-K 里（核采样从截断后的全词表抽），那时追加一列把它
    ;; 画出来，这样"指针"永远指着本次真正选中的那个词元，图与事实一致。
    (set! extra (if (> gen_pick -1) (not (in_obs_ids gen_pick)) #f))
    (Gfx.fill_rect CH_X CH_PTR_Y (- SW CH_X) (- (+ CH_PCT_Y 12) CH_PTR_Y) 14 16 22)
    (cond ((== n 0)
           (Gfx.draw_text "候选概率：等待分类…" CH_X (+ CH_PTR_Y 2) 120 130 140))
          (else
           {
             (set! m (if extra (+ n 1) n))
             (set! colw (Math.floor (/ (- SW (+ CH_X 2)) m)))
             (set! bw (if (> colw 14) (- colw 8) 6))
             ;; 归一化用的最大概率
             (set! i 0)
             (while (< i n) {
               (set! p (get_item top_probs i))
               (if (> p pmax) (set! pmax p) #f)
               (set! i (+ i 1))
             })
             (if extra (if (> gen_pick_p pmax) (set! pmax gen_pick_p) #f) #f)
             (set! i 0)
             (while (< i m) {
               (if (if extra (== i n) #f)
                   {
                     (set! id gen_pick)
                     (set! p gen_pick_p)
                     (set! s (Chat.token_text gen_pick))
                     (set! sel #t)
                   }
                   {
                     (set! id (get_item top_ids i))
                     (set! p (get_item top_probs i))
                     (set! s (get_item top_texts i))
                     (set! sel (== id gen_pick))
                   })
               (if (> (TXT.units s) 2) (set! s (String.slice s 0 1)) #f)
               (set! x (+ CH_X (* i colw)))
               (set! cx (+ x (Math.floor (/ colw 2))))
               (set! h (Math.floor (* (/ p pmax) CH_MAXH)))
               (if (< h 2) (set! h 2) #f)
               (if sel
                   {
                     (Gfx.fill_rect (+ x 4) (- CH_BASE h) bw h 255 170 60)
                     (draw_pointer cx (+ CH_PTR_Y 1))
                   }
                   (Gfx.fill_rect (+ x 4) (- CH_BASE h) bw h 72 128 208))
               (Gfx.draw_text s (- cx (Math.floor (/ (* (TXT.units s) 6) 2))) CH_LBL_Y
                              (if sel 255 240 170) (if sel 220 240 190) (if sel 120 160 230))
               (set! pct (Math.floor (* 100 p)))
               (set! s (String.concat (i2s pct) "%"))
               (Gfx.draw_text s (- cx (Math.floor (/ (* (TXT.units s) 6) 2))) CH_PCT_Y
                              (if sel 255 170 60) (if sel 170 170 60) (if sel 60 60 60))
               (set! i (+ i 1))
             })
           }))))

;; 可视化整体：左列框图（含层条）+ 右列候选概率图
(define draw_viz
  (lambda ()
    (vis_draw)
    (draw_prob_chart)
    #t))

;; ---------------------------------------------------------------- 绘制：整屏

(define draw_chat
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH 14 16 22)
    (draw_bar)
    (if (== viz 1)
        (draw_viz)
        #f)
    (draw_conv)
    (draw_pbar)
    (draw_input)
    (draw_status)
    ;; 页眉控件最后画：右栏底板会盖住页眉右侧，顺序反了开关就"消失"
    (draw_toggle)
    (EX.draw_button)
    (Gfx.refresh)
    (set! dirty #f)))

;; 模型选择页
(define draw_models
  (lambda ()
    (define n (length models))
    (define i 0)
    (define y 30)
    (define m (cdr '(0)))
    (Gfx.fill_rect 0 0 SW SH 14 16 22)
    (Gfx.set_font FONT_S)
    (Gfx.fill_rect 0 0 SW HEAD_H 28 32 44)
    (Gfx.draw_text "电子鹦鹉 · 选择模型" PAD 2 226 234 246)
    (if (== loading 1)
        {
          (Gfx.draw_text "正在载入模型…（解析权重 + 分配 KV-Cache）" PAD 60 240 210 130)
          (Gfx.draw_text "浏览器目标首次载入 56MB 模型约需 1~3 秒" PAD 80 160 170 190)
          (Gfx.draw_text "请稍候，界面会短时冻结" PAD 96 160 170 190)
        }
        (if (== n 0)
            {
              (Gfx.draw_text "未找到本地模型" PAD 40 250 170 130)
              (Gfx.draw_text "浏览器目标：请在页面「资源」面板" PAD 64 150 200 235)
              (Gfx.draw_text "预载 llm/nano-56m-q80.bin" PAD 80 150 200 235)
              (Gfx.draw_text "或 llm/nano-168m-q80.bin" PAD 96 150 200 235)
              (Gfx.draw_text "掌机：把模型放进 llm/ 目录" PAD 120 150 200 235)
            }
            {
              (while (< i n) {
                (set! m (get_item models i))
                (if (== i sel)
                    {
                      (Gfx.fill_rect PAD y (- SW (* PAD 2)) 26 34 84 140)
                      (Gfx.draw_rect PAD y (- SW (* PAD 2)) 26 110 180 250)
                    }
                    {
                      (Gfx.fill_rect PAD y (- SW (* PAD 2)) 26 34 38 50)
                      (Gfx.draw_rect PAD y (- SW (* PAD 2)) 26 80 86 104)
                    })
                (Gfx.draw_text (get_item m 0) (+ PAD 8) (+ y 2) 232 238 248)
                (Gfx.draw_text (get_item m 1) (+ PAD 8) (+ y 13) 130 150 172)
                (set! y (+ y 30))
                (set! i (+ i 1))
              })
              (Gfx.draw_text "点一行载入模型（载入后进入对话）" PAD (+ y 6) 140 150 168)
              (Gfx.draw_text "推理：C 语言端侧引擎 · 界面：灵机 Scheme" PAD (+ y 22) 110 130 150)
              (if (String.equals gen_msg "")
                  #f
                  (Gfx.draw_text gen_msg PAD (+ y 40) 250 170 120))
            }))
    (EX.draw_button)
    (Gfx.refresh)
    (set! dirty #f)))

(define draw_frame
  (lambda ()
    (if (== stage ST_CHAT) (draw_chat) (draw_models))))

;; ---------------------------------------------------------------- 引擎与对话逻辑

;; 记录一条消息（角色 1=用户 2=助手）
(define push_msg
  (lambda (role txt)
    {
      (push c_role role)
      (push c_text txt)
      (set! ln_dirty 1)
      (set! scroll 0)
    }))

;; 载入模型（阻塞）：先画提示帧，再调 (Chat.load)
(define load_model
  (lambda (idx)
    (define r 0)
    (define cfg #f)
    (set! loading 1)
    (draw_models)
    (set! r (Chat.load idx))
    (set! loading 0)
    (if (== r 0)
        {
          (set! model_name (get_item (get_item models idx) 0))
          (set! model_str model_name)
          (set! cfg (Chat.config))
          (set! n_layer (get_item cfg 3))
          (set! vocab_size (get_item cfg 2))
          (set! ctx_max (get_item cfg 9))
          ;; 生成长度 = 上下文长度 − 输入词元数（在 gen_step_once 里按 pos 判）
          (set! ctx_limit (if (> ctx_max 0) ctx_max 512))
          (set! stage ST_CHAT)
          (set! gen_msg "已载入，点下面的输入框开始对话")
          (logln (String.concat "model=" (String.concat model_name
                  (String.concat " layers=" (String.concat (i2s n_layer)
                  (String.concat " vocab=" (i2s vocab_size)))))))
          (set! dirty #t)
          #t
        }
        {
          (set! gen_msg (String.concat "载入失败（" (String.concat (i2s r) "）")))
          (logln (String.concat "load_fail=" (i2s r)))
          (set! dirty #t)
          #f
        })))

;; 采样统计刷新（每步一次）
(define refresh_info
  (lambda ()
    (define inf (Chat.info))
    (if (null? inf)
        #f
        {
          (set! pfill_pos (get_item inf 0))
          (set! pfill_tot (get_item inf 1))
          (set! gen_tokens (get_item inf 2))
          (set! gen_tps (get_item inf 3))
          (set! prefill_ms (get_item inf 4))
          (set! gen_pick (get_item inf 7))
          (set! ln_dirty 1)
          (set! dirty #t)
        })))

;; 可视化数据刷新：候选词元的 top-K 与本次采样概率（观测点由相位回放驱动）
(define refresh_topk
  (lambda ()
    (set! top_ids (Chat.top_ids))
    (set! top_texts (Chat.top_texts))
    (set! top_probs (Chat.top_probs))
    (set! gen_pick_p (Chat.pick_prob))
    #t))

;; 开始一轮对话
(define start_round
  (lambda (txt)
    (define np 0)
    (push_msg 1 txt)
    (set! gen_text "")
    (set! gen_tokens 0)
    (set! gen_tps 0)
    (set! gen_pick -1)
    (set! gen_pick_p 0.0)
    (set! pfill_pos 0)
    (set! pfill_tot 0)
    (set! prefill_ms 0)
    (set! ln_dirty 1)
    (set! np (Chat.begin txt))
    (set! pfill_pos 0)
    (logln (String.concat "user=" txt))
    (if (<= np 0)
        {
          (set! gen 0)
          (set! gen_msg "推理引擎未就绪")
          (logln "begin_fail")
        }
        {
          (set! gen 1)
          (set! pfill_tot np)
          (set! state_code 11)
          (set! gen_msg "")
          (logln (String.concat "begin prompt_tokens=" (i2s np)))
        })
    (set! dirty #t)))

;; 结束一轮。reason：0=模型自然结束（EOS），1=用户中断，2=上下文已满
(define finish_round
  (lambda (reason)
    (define txt "")
    (Chat.end)
    (set! gen 0)
    (set! anim_on 0)
    (set! ln_dirty 1)
    (cond ((not (String.equals gen_text "")) (set! txt gen_text))
          ((== reason 1) (set! txt "(已中断)"))
          ((== reason 2) (set! txt "(已达上下文上限)"))
          (else (set! txt "(无输出)")))
    (push_msg 2 txt)
    (set! gen_msg
          (cond ((== reason 1) "已中断 · 点输入框继续")
                ((== reason 2) "已达上下文上限 · 点输入框继续")
                (else "就绪 · 点输入框继续")))
    (line_num)   ; 强制重建折行缓存，下面的统计才是当前值
    (logln (String.concat "lines=" (String.concat (i2s (length ln_text))
            (String.concat " blanks=" (String.concat (i2s (blank_num)) " ")))))
    (logln (String.concat "done gen=" (String.concat (i2s gen_tokens)
            (String.concat " tps=" (String.concat (f1 gen_tps)
            (String.concat " stopped=" (String.concat (i2s reason)
            (String.concat " limit=" (String.concat (i2s ctx_limit)
            (String.concat " state=" (String.concat (i2s state_code)
            (String.concat " text=" txt))))))))))))
    (set! dirty #t)
    #t))

;; 跑一步前向，并把这一步的相位轨迹挂到回放游标上
(define gen_step_once
  (lambda ()
    (define st 0)
    (set! st (Chat.step))
    (set! state_code st)
    (set! gen_text (Chat.output))
    (refresh_info)
    (set! ln_dirty 1)
    (if (== viz 1)
        {
          (refresh_topk)
          ;; 轨迹回放：**每拍只推进一个相位**（不跳帧）。一步前向的轨迹可达上百项，
          ;; 因此可视化模式下生成明显变慢——这是刻意的取舍（llm.scm 同此）。
          (set! anim_len (Chat.trace_len))
          (set! anim_i 0)
          (set! anim_on (if (> anim_len 1) 1 0))
          (if (== anim_announced 0)
              {
                (set! anim_announced 1)
                (logln (String.concat "anim trace=" (String.concat (i2s anim_len) " per=1")))
              }
              #f)
        }
        #f)
    ;; 打印进度（每 8 个词元一行，便于端到端测试断言）
    (if (and (> gen_tokens 0) (== (mod gen_tokens 8) 0.0))
        (logln (String.concat "step gen=" (String.concat (i2s gen_tokens)
                (String.concat " tps=" (f1 gen_tps)))))
        #f)
    (cond ((< st 0)
           ;; -10 = 模型给出结束词元（自然结束）；其余（-11/-12/-20）算中断
           (if (== st -10)
               (finish_round 0)
               (finish_round 1)))
          ;; 0 = 引擎里已没有会话（例如用户中途点了页眉的模型名，引擎被卸载）
          ((== st 0) (finish_round 1))
          ;; 位置推进到模型上下文长度 → 生成了 (ctx_limit − 输入词元数) 个词元，收场
          ((>= pfill_pos ctx_limit) (finish_round 2))
          (else #f))
    (set! dirty #t)
    st))

;; 相位回放的一拍：只推进观测点（不跑前向）
(define gen_anim_once
  (lambda ()
    (define ob #f)
    (set! anim_i (+ anim_i 1))
    (if (>= anim_i anim_len)
        {
          (set! anim_i (- anim_len 1))
          (set! anim_on 0)
        }
        #f)
    (set! ob (Chat.trace_at anim_i))
    (if (null? ob)
        #f
        {
          (set! obs_layer (get_item ob 0))
          (set! obs_phase (get_item ob 1))
        })
    (set! dirty #t)
    #t))

;; 生成一拍：可视化开启时，「跑一步」与「回放这一步的相位轨迹」交替进行
(define gen_tick
  (lambda ()
    (if (== gen 1)
        (if (and (== viz 1) (== anim_on 1))
            (gen_anim_once)
            (gen_step_once))
        #f)))

;; ---------------------------------------------------------------- 交互

;; 打开宿主输入控件（借用鹦鹉笼同款：全键盘拼音 / 16 键九宫格 / 触屏软键盘）
(define open_input
  (lambda ()
    (define txt #f)
    ;; 借用宿主现成的文本输入控件（鹦鹉笼同款：软键盘 / 全键盘拼音 / 16 键九宫格）；
    ;; 提交即收起键盘、文本进对话记录（模态期间的界面由宿主控件接管）
    (set! txt (Ui.input "说点什么"))
    (if (null? txt)
        {
          (set! gen_msg "已取消输入")
          (set! dirty #t)
        }
        (if (String.equals txt "")
            {
              ;; 空输入 → 随机取一条预置问题（与 C 版 set_random_prompt 同语义：
              ;; 内容相同、时间戳驱动的伪随机选择），并当作本轮用户消息
              (set! txt (PICK.random))
              (logln (String.concat "preset=" txt))
              (start_round txt)
            }
            (start_round txt)))
    #t))

;; 模型选择页的点击
(define models_tap
  (lambda ()
    (define n (length models))
    (define i 0)
    (define hit -1)
    (while (< i n) {
      (if (== (in_rect tap_x tap_y PAD (+ 30 (* i 30)) (- SW (* PAD 2)) 26) 1)
          (set! hit i)
          #f)
      (set! i (+ i 1))
    })
    (if (>= hit 0)
        {
          (set! sel hit)
          (load_model hit)
        }
        #f)
    (set! dirty #t)
    #t))

;; 对话页的点击（按下沿）
(define chat_tap
  (lambda ()
    (cond
      ;; 右上「可视化」开关
      ((== (in_rect tap_x tap_y TOG_X TOG_Y TOG_W TOG_H) 1)
       {
         (set! viz (if (== viz 1) 0 1))
         (set! anim_announced 0)
         (rearm_tick)
         ;; 刚打开：立刻把候选分布与最近的观测点取回来（不必等下一拍）
         (if (== viz 1)
             {
               (refresh_topk)
               (set! obs_ob (Chat.obs))
               (if (null? obs_ob)
                   #f
                   {
                     (set! obs_layer (get_item obs_ob 0))
                     (set! obs_phase (get_item obs_ob 1))
                   })
             }
             #f)
         (set! ln_dirty 1)
         (logln (String.concat "viz=" (i2s viz)))
       })
      ;; 输入框右侧「停止」
      ((if (== gen 1) (== (in_rect tap_x tap_y STOP_X (+ INPUT_Y 2) STOP_W STOP_H) 1) #f)
       {
         (logln "stop")
         (finish_round 1)
       })
      ;; 输入框本体（生成中不响应；先判「停止」再判输入框）
      ((if (== gen 0) (== (in_rect tap_x tap_y PAD INPUT_Y (- SW (* PAD 2)) INPUT_H) 1) #f)
       (open_input))
      ;; 页眉上的模型名：回模型选择页（顺便卸载引擎，释放模型权重与 KV-Cache）
      ((== (in_rect tap_x tap_y 60 0 130 HEAD_H) 1)
       {
         ;; 正在生成就先按"中断"收尾（否则会话视图与引擎一起被释放，界面停在半截）
         (if (== gen 1) (finish_round 1) #f)
         (Chat.unload)
         (set! models (Chat.models))
         (if (null? models) (set! models (cdr '(0))) #f)
         (set! stage ST_MODEL)
         (set! gen_msg "已卸载模型")
         (logln "unload")
       })
      (else #f))
    (set! dirty #t)
    #t))

;; 拖拽滚动（对话区内上下拖动；按下点在对话区内才启动）
(define scroll_drag
  (lambda ()
    (define dy 0)
    (define lines 0)
    (define n (line_num))
    (define vis (Math.floor (/ (- CONV_BOT CONV_Y) LINE_H)))
    (if (== (Touch.is_pressed) #f)
        (set! drag_on 0)
        #f)
    (if (== drag_on 0)
        #f
        {
          ;; 手指向下拖（dy>0）→ 内容跟着往下走 → 看到更早的行 → scroll 增大
          (set! dy (- (Math.floor (Touch.y)) drag_y0))
          (set! lines (Math.floor (/ dy LINE_H)))
          (set! scroll (+ drag_scroll0 lines))
          (if (> scroll (max2 0 (- n vis))) (set! scroll (max2 0 (- n vis))) #f)
          (if (< scroll 0) (set! scroll 0) #f)
          (set! dirty #t)
        })))

;; ---------------------------------------------------------------- 退出落盘

;; 退出前把整段对话写入 data/nanochat.log（一次整写，避免每轮 I/O）
(define save_log
  (lambda ()
    (define i 0)
    (define n (length c_text))
    (define s "")
    (while (< i n) {
      (set! s (String.concat s (if (== (get_item c_role i) 1) "我: " "Nano: ")))
      (set! s (String.concat s (get_item c_text i)))
      (set! s (String.concat s "\n"))
      (set! i (+ i 1))
    })
    (if (> n 0)
        {
          (Fs.write LOG_PATH s)
          (logln (String.concat "saved bytes=" (String.atom_to_string (Fs.size LOG_PATH))))
        }
        #f)
    #t))

;; ---------------------------------------------------------------- 主循环

(define tick
  (lambda ()
    (define k 0)
    (EX.check)
    ;; 键盘：Enter 直接打开输入框（物理键盘/网页键盘均可）
    (set! k (Key.take_press))
    (if (== k 13)
        (if (== stage ST_CHAT)
            (if (== gen 0) (open_input) #f)
            #f)
        #f)
    ;; 触屏按下沿
    (if (Touch.take_press)
        {
          (set! tap_new #t)
          (set! tap_x (Math.floor (Touch.x)))
          (set! tap_y (Math.floor (Touch.y)))
          ;; 对话区内的按下启动拖拽跟踪
          (if (== (in_rect tap_x tap_y (conv_x) (conv_y0) (- SW (conv_x)) (- CONV_BOT (conv_y0))) 1)
              {
                (set! drag_on 1)
                (set! drag_y0 tap_y)
                (set! drag_scroll0 scroll)
              }
              (set! drag_on 0))
        }
        #f)
    ;; 事件分发
    (if tap_new
        {
          (if (== stage ST_CHAT) (chat_tap) (models_tap))
          (set! tap_new #f)
        }
        #f)
    (if (== stage ST_CHAT)
        {
          (scroll_drag)
          (gen_tick)
        }
        #f)
    ;; 后台恢复：引擎是虚拟机全局单例，实例销毁时会释放模型，恢复后按需重载
    (if (App.resumed)
        (if (== stage ST_CHAT)
            (if (Chat.loaded)
                {
                  (set! gen_msg "已恢复")
                  (set! dirty #t)
                }
                (load_model sel))
            #f)
        #f)
    (if dirty (draw_frame) #f)))

;; 节拍重挂（可视化开关切换时调用）：可视化一相位一帧、关闭时尽快推进
(define rearm_tick
  (lambda ()
    (if (> timer 0) (System.clear_interval timer) #f)
    (set! timer (System.set_interval (if (== viz 1) TICK_VIS TICK_FAST) tick))
    #t))

;; ---------------------------------------------------------------- 启动

(define timer 0)

(define boot
  (lambda ()
    (set! SW (Gfx.width))
    (set! SH (Gfx.height))
    (set! models (Chat.models))
    (if (null? models) (set! models (cdr '(0))) #f)
    (logln (String.concat "boot models=" (i2s (length models))))
    (if (> (length models) 0)
        (logln (String.concat "model0=" (get_item (get_item models 0) 1)))
        #f)
    (rearm_tick)
    (draw_frame)
    #t))

(boot)

;; 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）；确认后先落盘再销毁
(EX.install_with draw_frame)
(EX.on_exit save_log)
