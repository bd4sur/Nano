;; ============================================================================
;; filepicker.scm —— 可复用的「文件选择器」UI 组件（灵机应用程序层）
;;
;; 用途：给需要"打开一个文件"的 App 提供统一的浏览/选取界面。首个落地应用是
;; Markdown 阅读器（app/markdown.scm 底栏「打开」按钮）。任何 App 接上只要三步：
;;
;;     (import FP "app/lib/filepicker.scm")
;;     (FP.start "打开文件" "md,markdown")            ;; 开始选取（扩展名过滤，""=全部）
;;     （另有 (FP.start_at 标题 过滤 起始目录)，从任意子目录开始浏览）
;;     ...每拍：
;;         (if (FP.active)
;;             (FP.step)                               ;; 自绘 + 自处理输入，返回状态码
;;             { (EX.check) ... 自己的逻辑与绘制 ... })
;;
;;     状态码：(FP.step) → 0 进行中 / 1 已选中（路径见 (FP.result)）/ 2 已取消。
;;     拿到 1/2 之后调用方**自己重绘**整屏（本组件不恢复画面快照），并按需 (FP.reset)。
;;
;; 为什么是"状态机 + 每拍 step"而不是像 (Ui.input ...) 那样的阻塞函数：
;;   阻塞式模态框是宿主的能力（am_native_Ui.c 把进程置 BLOCKED，靠宿主控件画界面）；
;;   文件选择器完全用 Scheme + Gfx 画，不需要宿主新增任何东西。而灵机 App 的定时器
;;   每拍驱动一次，所以"每拍 step 一次"就是它的自然形态——拖动滚动、选中高亮都能
;;   逐帧响应，与 (native Touch) 的电平模型天然对齐。
;;
;; 本组件**不装定时器**、不抢键盘/触屏事件的所有权以外的东西：
;;   · 输入只读电平/边沿，不吞掉宿主 GUI 的输入管线（与 Touch/Key 两个本地库同一哲学）；
;;   · 活动期间会消费 (App.back_requested)（硬 A/ESC → 取消选择）。因此**调用方在
;;     (FP.active) 为真时不要调 (EX.check)**，否则两边抢同一个返回请求。
;;   · 右上角自己画了一枚「取消」按钮，位置与各 App 的「退出」按钮相同——用户在
;;     网页/掌机上点那个位置时宿主也会记一条返回请求，两条路都通向"取消"。
;;
;; **鼠标滚轮（仅 WASM GUI）**：本组件的列表也接受滚轮——与 App 自己的列表同一套
;; （原生通道 + 每拍缓动落位，见布局段下方「滚轮」）。增量有两个来源，相加不重计：
;;   ① 调用方**已认领**原生通道（(Touch.wheel_enable)）并在自己拍里取走了增量（如
;;      app/markdown.scm）→ 把它转交进来：(FP.set_wheel dy)。
;;   ② 调用方认领了但没取 / 取了没转交 → 增量还在宿主暂存里，组件每拍自己
;;      (Touch.wheel) 取。
;;   组件**不自己认领**通道：(Touch.wheel_enable) 没有对应的"交还"接口，认领会在组件
;;   退出后继续生效，反将害得没认领的调用方在自己的界面上滚不动。因此未认领的原生
;;   通道平台上，滚轮仍走页面"合成为手指"的老路（能滚，但不如原生通道顺）。
;;
;; **节拍要求（拖动要跟手就看这一条）**：组件的滚动是**像素级跟手**的，但 App 每拍
;; 只能采一次触屏电平，所以跟手的上限就是调用方的节拍频率。组件不自己装定时器
;; ——上游运行时**每轮事件循环对同一进程只触发一个定时器回调**（回调期间其余到期
;; 定时器顺延到下一轮），同进程两个 interval 会互相错开、两者都掉到 ~64ms（实测）。
;; 所以活动期间由**调用方**把节拍提上去：
;;     (set_tick (FP.tick_ms))    ;; 打开选择器前：16ms（≈60Hz）
;;     ... (FP.step) 拿到 1/2 后（set_tick 60） 恢复常规节拍
;; 参见 app/markdown.scm 的 set_tick / start_picker / tick_picker。调用方继续按
;; 自己的旧节拍（如 60ms）也能用，只是拖动会成 16fps 的"跳"而不是丝滑。
;;
;; 界面：
;;   ┌ 标题 ────────────────────────────── [取消] ┐
;;   │ /当前/目录                     N 项          │
;;   ├─────────────────────────────────────────────┤
;;   │  ..              （返回上级目录）            │
;;   │  doc/            （目录，点按进入）           │
;;   │  note.md         （文件，点按即选中并返回）   │
;;   （目录永远排在文件前面，同类内按名字升序；这是与文件管理器一致的排序规则）
;;   │  ...（列表超过一屏时可拖动滚动：像素级跟手 + 松手惯性）     │
;;   ├─────────────────────────────────────────────┤
;;   │ [上级]  点按进入/选取 · ↑↓ Enter            │
;;   └─────────────────────────────────────────────┘
;;
;; 路径语义与 (native Fs) 一致：相对业务资源根；根目录写作 "."，选取结果也是
;; 相对路径（"doc/note.md"），可直接喂给 (Fs.read ...)。不允许越出资源根。
;;
;; 扩展点（有意留出，未实现）：
;;   · 选目录而不是文件：加一个"只接目录模式"标志，把 p_activate 的选取分支改成
;;     返回当前目录——文件管理器的"复制/移动到…"需要它；
;;   · 多选：把 p_result 换成列表 + 每行打勾；
;;   · 路径直接输入：接 (Ui.input) 收一个路径字符串再跳转。
;; ============================================================================

(native Gfx)
(native Key)
(native Touch)
(native Fs)
(native String)
(native Math)
(native Diag)
(native App)
(import FSX "app/lib/fsx.scm")

;; ---- 布局 ----------------------------------------------------------------

(define SW 320)
(define SH 240)
(define HEAD_H 18)     ;; 页眉（标题 + 取消按钮）
(define PATH_Y 20)     ;; 当前目录行（12px）
(define LIST_Y 34)     ;; 列表首行 y
(define BOT_H 16)      ;; 页脚

;; 列表行高 = **字体行高 × 1.6**（触屏点按舒适；与 app/filemgr.scm 的列表、
;; app/markdown.scm 的菜单项 MENU_ROW_K 同一比例）。12px 抗锯齿字模行高 15 → 行高 24。
;; 组件**不切换调用方的字模**（那会污染调用方后续绘制：调用方重绘时按自己的字模
;; 排版，而 (Gfx.set_font) 是宿主进程级静态量），只按**进入时的当前字模**排版：
;; 行高与一屏行数由 p_relayout 现算（调用方把字模切到 16px 时行高自动变 29）。
(define ROW_K 1.6)     ;; 行高 / 字体行高（见 p_relayout 里为何用 round 而不是 floor）
(define LINE_H 15)     ;; 字体行高（(Gfx.font_info) 第 3 项）
(define ROW_H 24)      ;; 列表行高 = LINE_H × ROW_K
(define LIST_ROWS 7)   ;; 一屏几行（24px 行高下 7 行 = 168px，34 + 168 = 202）
(define TEXT_GAP 2)    ;; 行顶→墨迹上缘的空隙（12px 抗锯齿字模，见下面 ROW_TXT_DY）
;; 行文字的墨迹顶 = 行框顶 + (行高 − 字体行高)/2 + 墨迹上缘空隙（= 4 + 2 = 6）。
;; 空隙的来历见 am_native_Gfx.c 的 native_text_top_gap：(Gfx.draw_text) 的 y 是墨迹顶，
;; 而这一行要摆的是**行框**在行高里居中。
(define ROW_TXT_DY (+ (Math.floor (/ (- ROW_H LINE_H) 2)) TEXT_GAP))
;; 「页眉下 / 页脚上」的列表可用高度（与页脚留 10px；p_relayout 按它算一屏几行）
;; 注意算术运算符恒为**二参数**（manual §6.2.4），多写一个参数不会做链式减法，
;; 结果未规定——所以这里层层嵌套。
(define P_LIST_AREA_H (- (- (- SH BOT_H) 10) LIST_Y))

(define BTN_CANCEL_W 36)
(define BTN_UP_W 44)

;; ---- 滚动手感参数（与 app/filemgr.scm 的列表完全同一套） ------------------

(define P_VIEW_H (* LIST_ROWS ROW_H))  ;; 列表视口高（可用高度内最多的整行数 × 行高；24px 行高下 7 行 = 168px）
(define P_FLING_FRICTION 0.88)  ;; 惯性每拍衰减系数（16ms/拍）
(define P_FLING_VMAX 45)        ;; 惯性初速上限（px/拍）
(define P_FLING_VMIN 0.6)       ;; 惯性速度低于它就停（px/拍）
;; 松手是否"甩"看手势**平均速度**（px/16ms）：与拍长无关，"停住再抬手"自然不甩。
(define P_FLING_MIN_V 5)        ;; 平均速度 ≥ 5px/16ms（≈310px/s）才算甩
(define P_TAP_SLOP 8)           ;; 位移 < 8px 视为点按而不是拖动
;; 活动期间建议调用方使用的节拍（ms）。见文件头的"节拍要求"。
(define FP_TICK_MS 16)

;; ---- 滚轮（原生通道，仅 WASM GUI：浏览器里的鼠标滚轮） --------------------
;; 与 app/filemgr.scm、app/markdown.scm 同一套：页面在 App 认领原生通道后不再把滚轮
;; 合成为手指（那条路在 16ms 拍下有“滚不动 / 一抖 / 凭空反向位移”三个结构性缺陷，
;; 详见 am_native_Touch.h 的 (Touch.wheel_enable)），而是把像素增量原样交过来，
;; 由组件在自己的拍里缓动落位（一格 ≈ 90ms 走完，与宿主 GUI 的滚轮控件同手感）。
;;
;; 增量从哪来（两个来源相加，不会重复计数）：
;;   ① **调用方转交**： (FP.set_wheel dy) —— 调用方已经认领了通道并在它自己的拍里把
;;      (Touch.wheel) 取走了（如 app/markdown.scm），就把它转交进来；
;;   ② **自己取**： (Touch.wheel) —— 调用方认领了但没取（或者取了没转交），增量还在
;;      宿主的暂存里，本组件自己取。
;; 本组件**不自己去认领**通道（(Touch.wheel_enable) 没有对应的“交还”接口）：认领在
;; 组件退出后仍然生效，会害得未认领的调用方在它自己的界面上滚不动。
(define P_WHEEL_GAIN 0.45)     ;; 一拍吃掉剩余量的比例（≈ 90ms 走完一格）
(define p_wheel_in 0)          ;; 调用方转交的增量（(FP.set_wheel) 累加）
(define p_wheel_rest 0)        ;; 还没落位的剩余量（缓动用；整数值，必然归零）

;; ---- 状态 ----------------------------------------------------------------

(define p_active 0)         ;; 1 = 正在选取
(define p_next 0)           ;; 0 进行中 / 1 已选中 / 2 已取消
(define p_title "")
(define p_exts "")
(define p_dir ".")
(define p_rows (cdr '(0)))  ;; 条目表：(mk2 名字 是否目录)
(define p_sel 0)
;; 列表滚动：**像素级**偏移（0 = 首行贴视口顶）。以前是整行号 p_first，拖动只能
;; 整行跳（每 18px 顿一下）；改成像素偏移后手指移多少内容就移多少（1:1 跟手），
;; 半行也能停在视口上——与文件管理器（app/filemgr.scm）的列表同一套手感。
(define p_scroll 0)
(define p_fling 0)          ;; 松手后的惯性速度（px/拍；正 = 内容继续向上走）
(define p_sum_dy 0)         ;; 本次手势累计纵向位移（px，判“甩”用）
(define p_press_ms 0)       ;; 按下时刻（(Diag.ms)，真实毫秒）
(define p_result "")
(define p_err 0)            ;; 目录读不出来（Fs.list → #null）

;; 触屏
(define p_prev 0)
(define p_drag_last 0)
(define p_drag_max 0)       ;; 本次手势离按下点的最大位移（px，判断点按/拖动）
(define p_press_x 0)
(define p_press_y 0)
(define p_press_in_list 0)  ;; 按下点落在列表视口里（只有它会滚动）
(define p_press_stop 0)     ;; 按下时在刹车（打断惯性）→ 这一下松手不算点按

;; ---- 小工具 --------------------------------------------------------------

;; 按**当前字模**重算行高、一屏行数与视口高（见布局段说明）。在 start_at 里调一次：
;; 组件在调用方的屏幕上画，字模是调用方的（可能是 16px，见 app/markdown.scm 的
;; apply_font），所以不能像 App 那样在模块载入时算死。
(define p_relayout
  (lambda ()
    (define fi (Gfx.font_info (Gfx.get_font)))
    (set! LINE_H (if (list? fi) (get_item fi 2) 15))
    ;; 用 round 而不是 floor：32 位宿主（wasm32/MCU）上浮点是**截断低 5 位尾数**的
    ;; 表示（am_make_value_of_float），1.6 实际是 1.5999999…，15 × 1.6 = 23.99999…，
    ;; floor 会得到 23（行高少 1px）。round 得 24。
    (set! ROW_H (Math.round (* LINE_H ROW_K)))
    (set! ROW_TXT_DY (+ (Math.floor (/ (- ROW_H LINE_H) 2)) TEXT_GAP))
    (set! LIST_ROWS (Math.floor (/ P_LIST_AREA_H ROW_H)))
    (if (< LIST_ROWS 1) (set! LIST_ROWS 1) #f)
    (set! P_VIEW_H (* LIST_ROWS ROW_H))
    (display "FPLAYOUT line_h=") (display LINE_H)
    (display " row_h=") (display ROW_H)
    (display " rows=") (display LIST_ROWS) (newline)
    #t))

(define new_list (lambda () (cdr '(0))))
(define mk2 (lambda (a b) (cons a (cons b (cdr '(0))))))

;; 当前目录的显示形式（根 = "." 显示成 "/"）
(define dir_label
  (lambda ()
    (if (p_at_root) "/" (String.concat "/" p_dir))))

(define p_at_root (lambda () (String.equals p_dir ".")))

(define rows_count
  (lambda () (length p_rows)))

;; 显示行：第 0 行固定是 ".."（返回上级目录），第 d 行（d>=1）= p_rows[d-1]。
;; 与 app/filemgr.scm 的列表模型一致（同一套交互心智）。
(define disp_count
  (lambda () (+ (rows_count) 1)))

;; ---- 载入目录 ------------------------------------------------------------
;; 目录一律列出（可进入）；文件按扩展名过滤。是否目录在**载入时**算一次并缓存，
;; 绘制时不再逐行 stat——ESP32 的 SD stat 有毫秒级开销，60fps 逐帧做承受不起。
(define p_load
  (lambda ()
    (define raw (Fs.list p_dir))
    (define n 0)
    (define i 0)
    (define s "")
    (define keep (new_list))
    (set! p_rows (new_list))
    (set! p_sel 1)     ;; 默认选中第一条真实条目（0 是 ".."）
    (set! p_scroll 0)  ;; 换目录：回到列表顶
    (set! p_fling 0)
    ;; (list? raw)：空目录是 '()（合法），打不开才是 #null（见 filemgr.scm 同名注释）
    (if (list? raw)
        { (set! p_err 0)
          (set! n (length raw))
          (set! i 0)
          (while (< i n)
            { (set! s (get_item raw i))
              (if (Fs.is_dir (FSX.join p_dir s))
                  (push keep (mk2 s 1))
                  (if (== (FSX.ext_match s p_exts) 1)
                      (push keep (mk2 s 0))
                      #f))
              (set! i (+ i 1)) })
          (set! p_rows (FSX.sort_entries keep)) }
        { (set! p_err 1)
          (display "FPERR dir=") (display p_dir) (newline) })
    (display "FP dir=") (display p_dir)
    (display " n=") (display (rows_count))
    (display " ext=") (display p_exts) (newline)
    ;; 诊断行：排序 + 过滤后的条目表（目录带尾 "/"）——端到端测试据此断言
    (display "FP items ../ ")
    (set! i 0)
    (while (< i (rows_count))
      { (display (String.concat (get_item (get_item p_rows i) 0)
                    (if (== (get_item (get_item p_rows i) 1) 1) "/" "")))
        (display " ")
        (set! i (+ i 1)) })
    (newline)
    #t))

;; ---- 导航 ----------------------------------------------------------------

(define p_goto
  (lambda (d)
    { (set! p_dir d)
      (p_load) }))

(define p_up
  (lambda ()
    (p_goto (FSX.parent p_dir))))

;; 激活一个**显示行**：0 = ".."（返回上级）；目录 → 进入；文件 → 选中并返回 1
(define p_activate
  (lambda (idx)
    (define row #f)
    (if (== idx 0)
        (p_up)
        (if (and (> idx 0) (<= idx (rows_count)))
            { (set! row (get_item p_rows (- idx 1)))
              (set! p_sel idx)
              (if (== (get_item row 1) 1)
                  (p_goto (FSX.join p_dir (get_item row 0)))
                  { (set! p_result (FSX.join p_dir (get_item row 0)))
                    (set! p_next 1)
                    (display "FP pick ") (display p_result) (newline) }) }
            #f))
    #t))

;; ---- 滚动与选中移动 ------------------------------------------------------

;; 可滚动的最大偏移（内容高 − 视口高；不足一屏时为 0）
(define p_max_scroll
  (lambda ()
    (define m (- (* (disp_count) ROW_H) P_VIEW_H))
    (if (< m 0) 0 m)))

(define p_clamp_scroll
  (lambda ()
    { (if (< p_scroll 0) (set! p_scroll 0) #f)
      (if (> p_scroll (p_max_scroll)) (set! p_scroll (p_max_scroll)) #f)
      #t }))

;; 把选中行滚进视口（键盘上下、点按、进入目录后用）
(define p_ensure_sel_visible
  (lambda ()
    (define top 0)
    (define bot 0)
    (set! top (* p_sel ROW_H))
    (set! bot (+ top ROW_H))
    (if (< top p_scroll) (set! p_scroll top) #f)
    (if (> bot (+ p_scroll P_VIEW_H)) (set! p_scroll (- bot P_VIEW_H)) #f)
    (p_clamp_scroll)
    #t))

;; 惯性初速限幅
(define p_clamp_fling
  (lambda (v)
    (if (> v P_FLING_VMAX) P_FLING_VMAX
        (if (< v (- 0 P_FLING_VMAX)) (- 0 P_FLING_VMAX) v))))

;; 每拍推进一歩惯性（与 filemgr 同构：位移按速度、速度按摩擦衰减，碰到边界就停）
(define p_step_fling
  (lambda ()
    (if (not (== p_fling 0))
        { (set! p_scroll (- p_scroll p_fling))
          (if (or (< p_scroll 0) (> p_scroll (p_max_scroll)))
              { (p_clamp_scroll) (set! p_fling 0) }
              (set! p_fling (* p_fling P_FLING_FRICTION)))
          (if (< (Math.abs p_fling) P_FLING_VMIN)
              { (set! p_fling 0)
                (display "FPFLING end=")
                (display (Math.floor p_scroll)) (newline) }
              #f) }
        #f)
    #t))

;; ---- 滚轮落位（原生通道，见上面的「滚轮」段） ---------------------------

;; 本拍要走多少像素：一拍吃掉剩余量的 P_WHEEL_GAIN；剩余 ≤2px 时一次走完，
;; 保证必然归零（否则永远差最后一点，停在半行上且每拍重绘）。与 filemgr 同口径。
(define p_wheel_step_px
  (lambda ()
    (define r p_wheel_rest)
    (define a (Math.abs r))
    (define s 0)
    (if (<= a 2)
        (set! s r)
        { (set! s (Math.round (* r P_WHEEL_GAIN)))
          (if (== s 0) (set! s (if (> r 0) 1 -1)) #f) })
    (set! p_wheel_rest (- r s))
    s))

;; 每拍：取增量（转交来的 + 通道里剩的）→ 并入剩余量并刹车 → 向前走一步。
;; 方向：wheel_dy > 0 = 内容上滚 = p_scroll 增大（与宿主 GUI 一致）。
(define p_wheel
  (lambda ()
    (define dy 0)
    (define px 0)
    (set! dy (+ p_wheel_in (Touch.wheel)))
    (set! p_wheel_in 0)
    (if (not (== dy 0))
        { (set! p_fling 0)   ;; 滚轮是主动定位：先刹车，不接着甩
          (set! p_wheel_rest (+ p_wheel_rest dy))
          (display "FPWHEEL dy=") (display dy) (newline) }
        #f)
    (if (not (== p_wheel_rest 0))
        { (set! px (p_wheel_step_px))
          (if (not (== px 0))
              { (set! p_scroll (+ p_scroll px))
                (p_clamp_scroll) }
              #f) }
        #f)
    #t))

(define p_sel_move
  (lambda (d)
    (set! p_sel (+ p_sel d))
    (if (< p_sel 0) (set! p_sel 0) #f)
    (if (>= p_sel (disp_count)) (set! p_sel (- (disp_count) 1)) #f)
    (if (< p_sel 0) (set! p_sel 0) #f)
    (p_ensure_sel_visible)
    #t))

;; ---- 绘制 ----------------------------------------------------------------

;; 顶部的 ".." 行：根目录下变灰（点了只提示）
(define p_draw_parent_row
  (lambda (y)
    (define cr 96)
    (define cg 100)
    (define cb 110)
    (if (== p_sel 0)
        { (Gfx.fill_rect 2 (+ y 1) (- SW 8) (- ROW_H 2) 40 50 68)
          (Gfx.fill_rect 2 (+ y 1) 3 (- ROW_H 2) 120 180 250) }
        #f)
    (if (not (p_at_root)) { (set! cr 138) (set! cg 190) (set! cb 255) } #f)
    (Gfx.draw_text (if (p_at_root) "..   （已在资源根）" "..   上级目录") 10 (+ y ROW_TXT_DY) cr cg cb)
    #t))

(define p_draw_row
  (lambda (idx y)
    (define row (get_item p_rows idx))
    (define nm (get_item row 0))
    (define isd (get_item row 1))
    (define txt "")
    (define cr 214)
    (define cg 222)
    (define cb 234)
    (if (== (+ idx 1) p_sel)
        { (Gfx.fill_rect 2 (+ y 1) (- SW 8) (- ROW_H 2) 40 50 68)
          (Gfx.fill_rect 2 (+ y 1) 3 (- ROW_H 2) 120 180 250) }
        #f)
    (if (== isd 1)
        { (set! txt (String.concat nm "/"))
          (set! cr 138) (set! cg 190) (set! cb 255) }
        (set! txt nm))
    (Gfx.draw_text (FSX.clip_text txt 292) 10 (+ y ROW_TXT_DY) cr cg cb)
    #t))

;; 列表区：像素滚动 + 视口裁剪。
;; 停在半行时首/尾两行只露半截，高亮条与字形墨迹会溢出到页眉/页脚；
;; (Gfx.set_clip) 对 fill_rect 与字形绘制都生效，画完必须 reset_clip。
(define p_draw_list
  (lambda ()
    (define n (disp_count))
    (define i 0)
    (define last 0)
    (define y 0)
    (Gfx.set_clip 0 LIST_Y SW P_VIEW_H)
    (Gfx.fill_rect 0 LIST_Y SW P_VIEW_H 14 17 22)
    ;; 从"视口顶那一行"开始画，多画两行（上下各一行半截，其余交给裁剪）
    (set! i (Math.floor (/ p_scroll ROW_H)))
    (set! last (+ i (+ LIST_ROWS 2)))
    (if (> last n) (set! last n) #f)
    (while (< i last)
      { (set! y (- (+ LIST_Y (* i ROW_H)) p_scroll))
        (if (== i 0) (p_draw_parent_row y) (p_draw_row (- i 1) y))
        (set! i (+ i 1)) })
    ;; 空目录 / 无匹配文件：在 ".." 行下面给一句提示
    (if (== (rows_count) 0)
        (Gfx.draw_text (if (== p_err 1) "（目录无法读取）" "（空 / 无匹配文件）")
                       10 (+ (- (+ LIST_Y ROW_H) p_scroll) ROW_TXT_DY) 110 118 134)
        #f)
    (Gfx.reset_clip)
    #t))

(define p_draw
  (lambda ()
    (define bx 0)
    (define ms (p_max_scroll))
    (define bar 0)
    (define by 0)
    (set! bx (- SW (+ BTN_CANCEL_W 4)))
    (Gfx.fill_rect 0 0 SW SH 14 17 22)
    (Gfx.fill_rect 0 0 SW HEAD_H 30 38 52)
    (Gfx.draw_text (FSX.clip_text p_title 210) 6 3 236 242 250)
    ;; 取消按钮（与各 App 右上角「退出」按钮同位置，语义为"取消选择"）
    (Gfx.fill_rect bx 2 BTN_CANCEL_W 16 60 62 74)
    (Gfx.draw_rect bx 2 BTN_CANCEL_W 16 120 124 140)
    (Gfx.draw_text "取消" (+ bx 6) 4 225 228 235)
    ;; 当前目录 + 条目数
    (Gfx.draw_text (FSX.clip_text (dir_label) 236) 6 (+ PATH_Y 1) 150 172 200)
    (Gfx.draw_text (String.concat (String.atom_to_string (rows_count)) " 项")
                   (- SW 54) (+ PATH_Y 1) 120 136 158)
    ;; 列表（像素滚动 + 裁剪）
    (p_draw_list)
    ;; 右侧细滚动条（条目多于一屏时）：位置按**像素偏移**算，与列表同一口径
    (if (> ms 0)
        { (set! bar (Math.floor (/ (* P_VIEW_H P_VIEW_H) (+ ms P_VIEW_H))))
          (if (< bar 14) (set! bar 14) #f)
          (set! by (+ LIST_Y (Math.floor (/ (* p_scroll (- P_VIEW_H bar)) ms))))
          (Gfx.fill_rect (- SW 4) LIST_Y 3 P_VIEW_H 32 38 50)
          (Gfx.fill_rect (- SW 4) by 3 bar 96 116 148) }
        #f)
    ;; 页脚：上级按钮 + 提示
    (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
    (Gfx.fill_rect 4 (- SH 17) BTN_UP_W 14 52 58 74)
    (Gfx.draw_rect 4 (- SH 17) BTN_UP_W 14 110 118 138)
    (Gfx.draw_text "上级" 10 (- SH 16) 220 226 236)
    (Gfx.draw_text "点按进入/选取 · ↑↓ Enter" (+ BTN_UP_W 14) (- SH 16) 130 140 155)
    (Gfx.refresh)
    #t))

;; ---- 输入 ----------------------------------------------------------------

(define p_keys
  (lambda ()
    (define k (Key.take_press))
    (while (not (== k 0))
      { (if (== k 1) (p_sel_move -1) #f)
        (if (== k 2) (p_sel_move 1) #f)
        (if (== k 13) (p_activate p_sel) #f)
        (if (== k 8) (p_up) #f)
        (if (== k 27) (set! p_next 2) #f)
        (set! k (Key.take_press)) })
    #t))

;; 点按：取消按钮（右上角）/ 上级按钮（页脚）/ 列表行
(define p_tap
  (lambda (px py)
    (define idx 0)
    (if (and (>= px (- SW (+ BTN_CANCEL_W 4))) (<= py HEAD_H))
        (set! p_next 2)
        (if (and (>= py (- SH 17)) (<= py (- SH 3)) (>= px 4) (<= px (+ 4 BTN_UP_W)))
            (p_up)
            (if (and (>= py LIST_Y) (< py (+ LIST_Y P_VIEW_H)))
                ;; 行号由**像素偏移**反算（视口顶对应内容的第 p_scroll 个像素）
                { (set! idx (Math.floor (/ (+ (- py LIST_Y) p_scroll) ROW_H)))
                  (if (and (>= idx 0) (< idx (disp_count)))
                      (p_activate idx)
                      #f) }
                #f)))
    #t))

;; 触屏：拖动 = 像素级跟手滚动 + 松手惯性（细则与 app/filemgr.scm 的列表完全一致）：
;;   · “拖动/点按”看离按下点的最大位移（含横向），不是纵向位移累加；
;;   · 按下即刹车（打断惯性），且这一下松手不算点按；
;;   · 速度用指数滑动平均估，松手时若真拖动过就交给 p_fling；
;;   · 松手那一拍把末段未被采到的位移补上（低帧率下才真的 1:1）。
(define p_touch
  (lambda ()
    (define dy 0)
    (define dx 0)
    (define ady 0)
    (define dur 0)
    (define vavg 0)
    (if (Touch.is_pressed)
        { (if (== p_prev 0)
              { (set! p_prev 1)
                (set! p_press_stop (if (== p_fling 0) 0 1))
                (set! p_fling 0)
                (set! p_sum_dy 0)
                (set! p_press_ms (Diag.ms))
                (set! p_drag_max 0)
                (set! p_drag_last (Touch.y))
                (set! p_press_x (Touch.x))
                (set! p_press_y (Touch.y))
                (set! p_press_in_list
                      (if (and (>= p_press_y LIST_Y) (< p_press_y (+ LIST_Y P_VIEW_H))) 1 0)) }
              #f)
          (set! dy (- (Touch.y) p_drag_last))
          (set! p_drag_last (Touch.y))
          (set! dx (- (Touch.x) p_press_x))
          (if (> (Math.abs dx) p_drag_max) (set! p_drag_max (Math.abs dx)) #f)
          (set! ady (Math.abs (- (Touch.y) p_press_y)))
          (if (> ady p_drag_max) (set! p_drag_max ady) #f)
          (if (and (== p_press_in_list 1) (not (== dy 0)))
              { (set! p_sum_dy (+ p_sum_dy dy))
                (set! p_scroll (- p_scroll dy))
                (p_clamp_scroll) }
              #f) }
        { (if (== p_prev 1)
              { (set! p_prev 0)
                ;; 松手这一拍补末段未被采到的位移（同 filemgr 的列表）
                (if (== p_press_in_list 1)
                    { (set! dy (- (Touch.y) p_drag_last))
                      (set! dx (- (Touch.x) p_press_x))
                      (if (> (Math.abs dx) p_drag_max) (set! p_drag_max (Math.abs dx)) #f)
                      (set! ady (Math.abs (- (Touch.y) p_press_y)))
                      (if (> ady p_drag_max) (set! p_drag_max ady) #f)
                      (if (not (== dy 0))
                          { (set! p_sum_dy (+ p_sum_dy dy))
                            (set! p_scroll (- p_scroll dy))
                            (p_clamp_scroll) }
                          #f) }
                    #f)
                ;; 刹车（按下时打断了惯性）只压制"点按"，不压制"真拖动后的甩"
                (if (< p_drag_max P_TAP_SLOP)
                    (if (== p_press_stop 0) (p_tap p_press_x p_press_y) #f)
                    (if (== p_press_in_list 1)
                            { (set! dur (- (Diag.ms) p_press_ms))
                              (if (< dur 1) (set! dur 1) #f)
                              (set! vavg (/ (* p_sum_dy 16.0) dur))
                              (display "FPDRAG px=")
                              (display (Math.floor p_scroll))
                              (display " v=")
                              (display (Math.floor (* vavg 10)))
                              (newline)
                              (if (> (Math.abs vavg) P_FLING_MIN_V)
                                  { (set! p_fling (p_clamp_fling vavg))
                                    (display "FPFLING start=")
                                    (display (Math.floor p_scroll)) (newline) }
                                  #f) }
                            #f)) }
              #f) })
    #t))

;; ---- 对外接口 ------------------------------------------------------------

;; 活动期间建议调用方使用的节拍（ms）；见文件头的"节拍要求"。
(define tick_ms (lambda () FP_TICK_MS))

;; 调用方转交滚轮增量（帧缓冲像素，正 = 内容上滚）。适用情形：调用方**已经认领**
;; 了原生通道（(Touch.wheel_enable)）并在自己的拍里把 (Touch.wheel) 取走了——那就
;; 把它转交给本组件，否则增量就被调用方丢掉了（如 app/markdown.scm 的选择器分支）。
;; 调用方没取的话增量还在宿主暂存里，本组件自己会取（两者相加不会重复计数）。
(define set_wheel
  (lambda (dy)
    (set! p_wheel_in (+ p_wheel_in dy))
    #t))

(define start_at
  (lambda (title exts dir)
    { (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (p_relayout)     ;; 行高 = 当前字模行高 × 1.6（见布局段）
      (set! p_title title)
      (set! p_exts exts)
      (set! p_dir dir)
      (set! p_result "")
      (set! p_next 0)
      (set! p_prev 0)
      (set! p_scroll 0)
      (set! p_fling 0)
      (set! p_sum_dy 0)
      (set! p_drag_max 0)
      (set! p_press_stop 0)
      ;; 滚轮：进入时清掉上一轮/调用方转交的残留（不能把上一个上下文里的增量
      ;; 攒到这一次，否则一打开就自己滑一段）
      (set! p_wheel_in 0)
      (set! p_wheel_rest 0)
      (set! p_active 1)
      (p_load) }))

;; 注意：入口叫 start 而不是 begin —— `begin` 是**语法关键字**，不能用作变量名
;; （manual §2.6；写成 (define begin ...) 会编译失败，症状是静默的 compile_fail）。
(define start
  (lambda (title exts)
    (start_at title exts ".")))

(define active
  (lambda () (== p_active 1)))

(define result
  (lambda () p_result))

(define dir
  (lambda () p_dir))

(define reset
  (lambda ()
    { (set! p_active 0)
      (set! p_next 0)
      ;; 滚轮：剩余量丢掉，不要带到下一次（或带回调用方的界面）
      (set! p_wheel_in 0)
      (set! p_wheel_rest 0) }))

;; 每拍调用一次：自绘 + 自处理输入；返回 0 进行中 / 1 已选中 / 2 已取消
(define step
  (lambda ()
    (if (== p_active 0)
        0
        { (if (== p_next 0) (if (App.back_requested) (set! p_next 2) #f) #f)
          (if (== p_next 0) (p_keys) #f)
          (if (== p_next 0) (p_touch) #f)
          (if (== p_next 0) (p_wheel) #f)
          (if (== p_next 0) (p_step_fling) #f)
          (if (== p_next 0) (p_draw) #f)
          (if (== p_next 2)
              { (display "FP cancel") (newline) }
              #f)
          p_next })))