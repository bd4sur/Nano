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
;; 界面：
;;   ┌ 标题 ────────────────────────────── [取消] ┐
;;   │ /当前/目录                     N 项          │
;;   ├─────────────────────────────────────────────┤
;;   │  ..              （返回上级目录）            │
;;   │  doc/            （目录，点按进入）           │
;;   │  note.md         （文件，点按即选中并返回）   │
;;   （目录永远排在文件前面，同类内按名字升序；这是与文件管理器一致的排序规则）
;;   │  ...（列表超过一屏时随选中项滚动）           │
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
(native App)
(import FSX "app/lib/fsx.scm")

;; ---- 布局 ----------------------------------------------------------------

(define SW 320)
(define SH 240)
(define HEAD_H 18)     ;; 页眉（标题 + 取消按钮）
(define PATH_Y 20)     ;; 当前目录行（12px）
(define LIST_Y 34)     ;; 列表首行 y
(define ROW_H 18)      ;; 行高
(define LIST_ROWS 10)  ;; 一屏几行（34 + 10*18 = 214）
(define BOT_H 16)      ;; 页脚

(define BTN_CANCEL_W 36)
(define BTN_UP_W 44)

;; ---- 状态 ----------------------------------------------------------------

(define p_active 0)         ;; 1 = 正在选取
(define p_next 0)           ;; 0 进行中 / 1 已选中 / 2 已取消
(define p_title "")
(define p_exts "")
(define p_dir ".")
(define p_rows (cdr '(0)))  ;; 条目表：(mk2 名字 是否目录)
(define p_sel 0)
(define p_first 0)
(define p_result "")
(define p_err 0)            ;; 目录读不出来（Fs.list → #null）

;; 触屏
(define p_prev 0)
(define p_drag_last 0)
(define p_drag_total 0)
(define p_press_x 0)
(define p_press_y 0)

;; ---- 小工具 --------------------------------------------------------------

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
    (set! p_first 0)
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

(define p_clamp_first
  (lambda ()
    (define mx (- (disp_count) LIST_ROWS))
    (if (< mx 0) (set! mx 0) #f)
    (if (< p_first 0) (set! p_first 0) #f)
    (if (> p_first mx) (set! p_first mx) #f)
    #t))

(define p_scroll_by
  (lambda (d)
    { (set! p_first (+ p_first d))
      (p_clamp_first) }))

(define p_sel_move
  (lambda (d)
    (set! p_sel (+ p_sel d))
    (if (< p_sel 0) (set! p_sel 0) #f)
    (if (>= p_sel (disp_count)) (set! p_sel (- (disp_count) 1)) #f)
    (if (< p_sel 0) (set! p_sel 0) #f)
    (if (< p_sel p_first) (set! p_first p_sel) #f)
    (if (>= p_sel (+ p_first LIST_ROWS))
        (set! p_first (- (+ p_sel 1) LIST_ROWS))
        #f)
    (p_clamp_first)
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
    (Gfx.draw_text (if (p_at_root) "..   （已在资源根）" "..   上级目录") 10 (+ y 1) cr cg cb)
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
    (Gfx.draw_text (FSX.clip_text txt 292) 10 (+ y 1) cr cg cb)
    #t))

(define p_draw
  (lambda ()
    (define n (disp_count))
    (define i 0)
    (define last 0)
    (define y 0)
    (define bx 0)
    (define vh 0)
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
    ;; 列表
    (set! last (+ p_first LIST_ROWS))
    (if (> last n) (set! last n) #f)
    { (set! i p_first)
      (while (< i last)
        { (set! y (+ LIST_Y (* (- i p_first) ROW_H)))
          (if (== i 0) (p_draw_parent_row y) (p_draw_row (- i 1) y))
          (set! i (+ i 1)) }) }
    ;; 空目录 / 无匹配文件：在 ".." 行下面给一句提示
    (if (== (rows_count) 0)
        (Gfx.draw_text (if (== p_err 1) "（目录无法读取）" "（空 / 无匹配文件）")
                       10 (+ (+ LIST_Y ROW_H) 2) 110 118 134)
        #f)
    ;; 右侧细滚动条（条目多于一屏时）
    (if (> n LIST_ROWS)
        { (set! vh (* LIST_ROWS ROW_H))
          (set! bar (Math.floor (/ (* vh LIST_ROWS) n)))
          (if (< bar 14) (set! bar 14) #f)
          (set! by (+ LIST_Y (Math.floor (/ (* p_first (- vh bar)) (- n LIST_ROWS)))))
          (Gfx.fill_rect (- SW 4) LIST_Y 3 vh 32 38 50)
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
            (if (and (>= py LIST_Y) (< py (- SH BOT_H)))
                { (set! idx (+ p_first (Math.floor (/ (- py LIST_Y) ROW_H))))
                  (p_activate idx) }
                #f)))
    #t))

(define p_touch
  (lambda ()
    (define dy 0)
    (if (Touch.is_pressed)
        { (if (== p_prev 0)
              { (set! p_prev 1)
                (set! p_drag_total 0)
                (set! p_drag_last (Touch.y))
                (set! p_press_x (Touch.x))
                (set! p_press_y (Touch.y)) }
              #f)
          (set! dy (- (Touch.y) p_drag_last))
          (set! p_drag_last (Touch.y))
          (set! p_drag_total (+ p_drag_total dy))
          (if (and (>= p_press_y LIST_Y) (< p_press_y (- SH BOT_H)))
              (p_scroll_by (- 0 dy))
              #f) }
        { (if (== p_prev 1)
              { (set! p_prev 0)
                (if (< (Math.abs p_drag_total) 8)
                    (p_tap p_press_x p_press_y)
                    #f) }
              #f) })
    #t))

;; ---- 对外接口 ------------------------------------------------------------

(define start_at
  (lambda (title exts dir)
    { (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (set! p_title title)
      (set! p_exts exts)
      (set! p_dir dir)
      (set! p_result "")
      (set! p_next 0)
      (set! p_prev 0)
      (set! p_drag_total 0)
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
      (set! p_next 0) }))

;; 每拍调用一次：自绘 + 自处理输入；返回 0 进行中 / 1 已选中 / 2 已取消
(define step
  (lambda ()
    (if (== p_active 0)
        0
        { (if (== p_next 0) (if (App.back_requested) (set! p_next 2) #f) #f)
          (if (== p_next 0) (p_keys) #f)
          (if (== p_next 0) (p_touch) #f)
          (if (== p_next 0) (p_draw) #f)
          (if (== p_next 2)
              { (display "FP cancel") (newline) }
              #f)
          p_next })))