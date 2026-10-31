;; ============================================================================
;; 续体可视化（call/cc）—— 灵机应用程序
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx / Touch 本地库
;;
;; 三个例子共用同一套"两条时间线"的呈现方式：
;;   · 横向 = 时间（每格 = 一次控制流推进，6px 宽）；
;;   · 两条轨道 = 两个控制上下文（两种颜色 / 两处续体）；
;;   · 竖线 = 一次**续体调用**，控制从一条轨跳到另一条轨；
;;   · 顶部"双色流"把最近 20 个时间单位按轨道的颜色逐字写出来。
;;
;;   例 1  yin-yang puzzle：阴 @ 与阳 * 两处 call/cc 抓下的续体互相调用，
;;         输出 @ * @ ** @ *** …（阴每轮 1 步，阳每轮多 1 步）；
;;   例 2  call/cc 实现生成器：生成器体被切成可续跑的片段，每产出一个值
;;         （素数）就把控制交还调用者；下次调用不是从头跑，而是跳回上次
;;         yield 处的续体；
;;   例 3  协程来回切换：两个协程各存一个续体，靠调用对方的续体轮流跑。
;;
;; 节奏（为什么用"阻塞睡眠"而不是每拍抓续体）：
;;   每个时间单位写完后调用 (System.read 睡眠队列 TICK_MS)，把进程阻塞一拍 ——
;;   零 CPU 开销的定时器，醒来继续跑下一格（实测节拍约 64ms，见 TICK_MS 注释）。
;;   三个例子的代码因此是**普通的同步代码**：yin-yang 就是那句经典表达式原样跑下去，生成器就是"取值—打印"
;;   的循环，协程就是一个小 if 调度器。call/cc 只出现在语义真正需要它的地方
;;   （每轮/每次产出/每次切换各一次），而不是用来做逐帧暂停。
;;
;;   为什么不能"每拍用 call/cc 暂停再恢复"：本解释器**不回收续体**——每次
;;   call/cc 捕获的栈快照永久占用内存（实测约 0.7KB/次，System.gc 无效，
;;   丢弃引用也不行）。逐帧抓续体在几十秒内就会把内存池吃掉；而在
;;   yin-yang 每轮抓 2 次、生成器每次产出抓 1 次、协程每回合抓 1~2 次时，
;;   捕获次数随时间**次线性**增长（yin-yang 每轮比上一轮多走一步），
;;   长时间运行也只占几十 KB。例 1 一个屏周期约 20 轮、跑满 240 格才换例子。
;;
;; 解释器约束（编写本 App 时逐条遵守，见 manual.md 与 doc/AGENTS.md）：
;;   · 算术运算符固定二参数，多元必须嵌套；
;;   · 没有 let/case/do/for/list/append/map；define 无过程简写；
;;   · cond 每个子句只接受单个表达式，多分支一律平铺 if；
;;   · / 与 mod 恒返回浮点 —— 列表下标与绘图坐标一律 Math.round 收口；
;;   · 运行期**不要造字符串**：短字符串会被解释器驻留且永不释放，逐帧
;;     String.concat 会稳定泄漏。本 App 的文案全是字面量常量，数字用
;;     (draw_num) 逐位贴字模（见 draw_num 注释）；
;;   · 顶层 '() 字面量是模块级静态槽，只能 push 一次（本 App 的时间线缓冲
;;     一次性预分配，此后只用 set_item! 原地改写，零分配）。
;;
;; 退出：硬 A/ESC → 宿主全局「是否退出？」确认框（见 app/lib/exit.scm）。
;; 交互：点按屏幕 = 换下一个例子（右上角「退出」按钮除外）。
;; ============================================================================

(native Gfx)
(native Math)
(native System)
(native Touch)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 画布与布局 -----------------------------------------------------------

(define W (Gfx.width))
(define H (Gfx.height))
(define TX0 48)                 ; 时间线左端（留出泳道标签）
(define TX1 (- W 6))            ; 时间线右端（最新一格贴在这里）
(define UNIT 6)                 ; 每个时间单位的像素宽度
(define U_CAP 44)               ; 时间线窗口容纳的单位数
(define Y0 72)                  ; 上轨基线
(define Y1 118)                 ; 下轨基线
(define RIBBON_Y 20)            ; 双色流字符带
(define STATUS_Y 36)            ; 计数行
(define EXPLAIN_Y 138)          ; 说明文字首行
(define HINT_Y (- H 18))        ; 底部提示行
(define TICK_MS 40)             ; 一个时间单位的时长（阻塞睡眠；实测宿主事件循环
                                ;   粒度约 +16ms，故实际节拍约 65ms ≈ 15 格/秒）
(define RUN_LIMIT 240)          ; 一个例子跑多少格后自动换下一个

;; ---- 时间线缓冲（一次性预分配，此后零分配）--------------------------------

(define u_lane '())             ; 每个时间单位属于哪条轨（0/1）
(define u_ch '())               ; 每个时间单位的字符（画双色流用）
(define u_n 0)                  ; 当前窗口内的单位数
(define active -1)              ; 最新一格所在的轨（-1 = 空）
(define switches 0)             ; 续体调用（换轨）次数
(define units 0)                ; 本例子已推进的格数
(define t0 0)                   ; 上轨累计走过的格子（不受窗口滚动影响）
(define t1 0)                   ; 下轨累计走过的格子
(define i 0)

(define alloc_buffers
  (lambda ()
    (while (< i U_CAP)
      { (push u_lane 0)
        (push u_ch "·")
        (set! i (+ i 1)) })))

;; 窗口满时整体左移一格（滚动时间线）
(define shift_left
  (lambda ()
    (define j 0)
    (while (< j (- U_CAP 1))
      { (set_item! u_lane j (get_item u_lane (+ j 1)))
        (set_item! u_ch j (get_item u_ch (+ j 1)))
        (set! j (+ j 1)) })
    (set! u_n (- U_CAP 1))))

(define push_unit
  (lambda (lane ch)
    { (if (>= u_n U_CAP) (shift_left) #f)
      (if (and (> u_n 0) (not (== lane (get_item u_lane (- u_n 1)))))
          (set! switches (+ switches 1))
          #f)
      (set_item! u_lane u_n lane)
      (set_item! u_ch u_n ch)
      (set! u_n (+ u_n 1))
      (set! active lane)
      (if (== lane 0) (set! t0 (+ t0 1)) (set! t1 (+ t1 1))) }))

;; ---- 节奏与总调度 ---------------------------------------------------------

(define sleepq (System.make_queue 1))   ; 只用来做"阻塞睡眠"（无人写入 → 超时返回）
(define esc_k #f)                       ; 本轮流（一个例子的一次运行）的逃逸续体
(define demo 0)
(define skip_req 0)                     ; 用户点按请求换例子

;; 推进一个时间单位：落一格 → 画一帧 → 到点换例子 → 阻塞 TICK_MS
(define step_once
  (lambda (lane ch)
    { (push_unit lane ch)
      (set! units (+ units 1))
      (frame)
      (if (>= units RUN_LIMIT) (esc_k 0) #f)
      (if (== skip_req 1) (esc_k 0) #f)
      (System.read sleepq TICK_MS) }))

;; ---- 例 1：yin-yang puzzle -------------------------------------------------
;;
;; (((lambda (x) (display "@") x) (call/cc (lambda (k) k)))
;;  ((lambda (x) (display "*") x) (call/cc (lambda (k) k))))
;;
;; 第一处 call/cc 的续体 k1 = "阴这一行剩下的计算"，第二处得到 k2。
;; 阳走完后调用 k1 → 控制跳回阴那行（阴再走一步），紧接着阳又走一步并抓一个
;; 新的 k2 …… 于是输出 @ * @ ** @ *** …
;;（本 App 把 display 换成 step_once：落一格 + 画帧 + 阻塞一拍。）

(define d1_puzzle
  (lambda ()
    (((lambda (x) (begin (step_once 0 "@") x)) (call/cc (lambda (k) k)))
     ((lambda (x) (begin (step_once 1 "*") x)) (call/cc (lambda (k) k))))))

;; ---- 例 2：用 call/cc 实现生成器 ------------------------------------------
;;
;; 生成器 = 一个被 call/cc 切成片段的函数体：
;;   g_yield  把"下次从这里继续"的续体存进 g_resume，然后带着值逃回调用者；
;;   g_call   每次调用先抓住调用者的返回续体 g_ret，再恢复 g_resume（续跑）；
;; 于是 (g) 每次只跑到下一个 yield 为止——这正是生成器语义，没有线程参与。

(define g_ret #f)
(define g_resume #f)
(define g_prime 0)
(define g_val 0)
(define g_last 0)

(define g_yield
  (lambda (v)
    (call/cc (lambda (k)
               { (set! g_resume k)
                 (g_ret v) }))))

;; 生成器体（找素数）：每试除一次在生成器轨道上走一格
(define g_trial
  (lambda (n d)
    { (step_once 1 "产")
      (if (> (* d d) n)
          (set! g_prime 1)
          (if (== (mod n d) 0)
              (set! g_prime 0)
              (g_trial n (+ d 1)))) }))

(define g_body
  (lambda (n)
    (if (> n 30)
        (g_ret 'done)
        { (g_trial n 2)
          (if (== g_prime 1)
              { (g_yield n)
                (g_body (+ n 1)) }
              (g_body (+ n 1)) }))))

(define g_call
  (lambda ()
    (call/cc (lambda (ret)
               { (set! g_ret ret)
                 (if g_resume
                     ((lambda (k) { (set! g_resume #f) (k 0) }) g_resume)
                     (g_body 2)) }))))

;; 调用者：每取一个值在调用者轨道上走一格
(define d2_run
  (lambda ()
    { (step_once 0 "取")
      (set! g_val (g_call))
      (if (number? g_val) (set! g_last g_val) #f)
      (d2_run) }))

;; ---- 例 3：协程来回切换 ----------------------------------------------------
;;
;; 两个协程各保存自己的续体（ka / kb），谁想停就 co_yield 一次：把续体存好，
;; 然后逃回调度器本轮的开头。调度器不过是一个 if：轮到谁就恢复谁的续体。
;; 切换成本 = 一次 call/cc（外加调度器自己的那一层）。

(define ka #f)
(define kb #f)
(define who 0)
(define turn_esc #f)

(define co_yield
  (lambda (tag)
    (call/cc (lambda (k)
               { (if (== tag 0) (set! ka k) (set! kb k))
                 (set! who (if (== tag 0) 1 0))
                 (turn_esc 0) }))))

(define co_run
  (lambda (lane n ch)
    (if (> n 0)
        { (step_once lane ch)
          (co_run lane (- n 1) ch) }
        #f)))

;; A、B 每回合的步数逐轮递增，于是两条轨的段越来越长
(define corp_a
  (lambda (n)
    { (co_run 0 n "A")
      (co_yield 0)
      (corp_a (+ n 1)) }))

(define corp_b
  (lambda (n)
    { (co_run 1 n "B")
      (co_yield 1)
      (corp_b (+ n 1)) }))

(define d3_run
  (lambda ()
    { (call/cc (lambda (esc)
                 { (set! turn_esc esc)
                   (if (== who 0)
                       (if ka ((lambda (k) { (set! ka #f) (k 0) }) ka) (corp_a 1))
                       (if kb ((lambda (k) { (set! kb #f) (k 0) }) kb) (corp_b 2))) }))
      (d3_run) }))

;; ---- 样式与例子切换 -------------------------------------------------------

(define idx_str "1/3")
(define title "续体 · yin-yang")
(define name0 "阴 @")
(define name1 "阳 *")
(define c0r 90)
(define c0g 215)
(define c0b 235)
(define c1r 240)
(define c1g 120)
(define c1b 200)
(define d0r 54)                 ; 段长标签用色（泳道色的暗版，见 set_demo 末尾换算）
(define d0g 129)
(define d0b 141)
(define d1r 144)
(define d1g 72)
(define d1b 120)

(define set_demo
  (lambda (d)
    { (set! demo d)
      (set! u_n 0)
      (set! units 0)
      (set! switches 0)
      (set! active -1)
      (set! t0 0)
      (set! t1 0)
      (set! skip_req 0)
      (set! ka #f)
      (set! kb #f)
      (set! who 0)
      (set! g_resume #f)
      (set! g_ret #f)
      (set! g_last 0)
      (if (== d 0)
          { (set! idx_str "1/3")
            (set! title "续体 · yin-yang")
            (set! name0 "阴 @")
            (set! name1 "阳 *")
            (set! c0r 90) (set! c0g 215) (set! c0b 235)
            (set! c1r 240) (set! c1g 120) (set! c1b 200) }
          #f)
      (if (== d 1)
          { (set! idx_str "2/3")
            (set! title "续体 · 生成器")
            (set! name0 "取")
            (set! name1 "产")
            (set! c0r 245) (set! c0g 180) (set! c0b 80)
            (set! c1r 130) (set! c1g 220) (set! c1b 130) }
          #f)
      (if (== d 2)
          { (set! idx_str "3/3")
            (set! title "续体 · 协程")
            (set! name0 "A")
            (set! name1 "B")
            (set! c0r 110) (set! c0g 170) (set! c0b 255)
            (set! c1r 255) (set! c1g 150) (set! c1b 160) }
          #f)
      ;; 段长标签用色 = 泳道色 × 0.6（暗一档，不与格子抢眼）
      (set! d0r (Math.floor (/ (* c0r 3) 5)))
      (set! d0g (Math.floor (/ (* c0g 3) 5)))
      (set! d0b (Math.floor (/ (* c0b 3) 5)))
      (set! d1r (Math.floor (/ (* c1r 3) 5)))
      (set! d1g (Math.floor (/ (* c1g 3) 5)))
      (set! d1b (Math.floor (/ (* c1b 3) 5))) }))

;; ---- 绘制 ---------------------------------------------------------------

;; 数字逐位绘制：解释器的短字符串会**驻留**（永久保留，GC 收不回），
;; 因此运行期一律不构造字符串——文案用字面量常量，数字拆位贴数字字模。
(define digch '())              ; "0".."9"（启动时一次性 push）

(define alloc_digits
  (lambda ()
    { (push digch "0") (push digch "1") (push digch "2") (push digch "3") (push digch "4")
      (push digch "5") (push digch "6") (push digch "7") (push digch "8") (push digch "9") }))

;; 在 (x,y) 处绘制非负整数 v 的十进制表示（最多 3 位），返回结束后的 x
(define draw_num
  (lambda (v x y cr cg cb)
    (define m 1)
    (define d 0)
    (define xx x)
    { (if (< v 10) (set! m 1) (if (< v 100) (set! m 10) (set! m 100)))
      (while (> m 0)
        { (set! d (Math.floor (/ v m)))
          (set! d (- d (* 10 (Math.floor (/ d 10)))))        ; d = d mod 10
          (Gfx.draw_text (get_item digch d) xx y cr cg cb)
          (set! xx (+ xx 7))
          (set! m (Math.floor (/ m 10))) })
      xx }))

;; 同上，但**右对齐**到 xe（数字位数变化时右侧不乱）
(define draw_numr
  (lambda (v xe y cr cg cb)
    (define m 1)
    (define d 0)
    (define xx xe)
    { (if (< v 10) (set! m 1) (if (< v 100) (set! m 10) (set! m 100)))
      (while (> m 0)
        { (set! xx (- xx 7))
          (set! d (Math.floor (/ v m)))
          (set! d (- d (* 10 (Math.floor (/ d 10)))))
          (Gfx.draw_text (get_item digch d) xx y cr cg cb)
          (set! m (Math.floor (/ m 10))) })
      xx }))

;; 双色流：把最近 20 个时间单位按各自轨道的颜色逐字写出来（右对齐）
(define draw_ribbon
  (lambda ()
    (define j 0)
    (define kk 0)
    (define lane 0)
    (define x 0)
    { (Gfx.draw_text "流" 6 RIBBON_Y 110 130 150)
      (set! j (- u_n 1))
      (set! kk 0)
      (while (and (>= j 0) (< kk 20))
        { (set! lane (get_item u_lane j))
          (set! x (+ 74 (* (- 19 kk) 12)))
          (if (== lane 0)
              (Gfx.draw_text (get_item u_ch j) x RIBBON_Y c0r c0g c0b)
              (Gfx.draw_text (get_item u_ch j) x RIBBON_Y c1r c1g c1b))
          (set! j (- j 1))
          (set! kk (+ kk 1)) }) }))

;; 计数行：两条轨各走了多少格、切换了几次（定长列，避免量文字宽度）
(define draw_status
  (lambda ()
    { (Gfx.draw_text name0 6 STATUS_Y c0r c0g c0b)
      (draw_num t0 44 STATUS_Y c0r c0g c0b)
      (Gfx.draw_text "步" 74 STATUS_Y c0r c0g c0b)
      (Gfx.draw_text name1 104 STATUS_Y c1r c1g c1b)
      (draw_num t1 142 STATUS_Y c1r c1g c1b)
      (Gfx.draw_text "步" 172 STATUS_Y c1r c1g c1b)
      (Gfx.draw_text "切换" 208 STATUS_Y 150 165 185)
      (draw_num switches 244 STATUS_Y 150 165 185)
      (Gfx.draw_text "次" 274 STATUS_Y 150 165 185) }))

;; 两条轨道 + 每一格 + 换轨竖线 + 当前推进点
(define draw_timeline
  (lambda ()
    (define j 0)
    (define x 0)
    (define ln 0)
    { (Gfx.draw_text name0 6 (- Y0 6) c0r c0g c0b)
      (Gfx.draw_text name1 6 (- Y1 6) c1r c1g c1b)
      (Gfx.draw_line TX0 Y0 TX1 Y0 44 48 60)
      (Gfx.draw_line TX0 Y1 TX1 Y1 44 48 60)
      (set! j 0)
      (while (< j u_n)
        { (set! x (- TX1 (* (- u_n j) UNIT)))            ; 本格左边缘
          (set! ln (get_item u_lane j))
          (if (== ln 0)
              (Gfx.fill_rect (+ x 1) (- Y0 5) (- UNIT 1) 10 c0r c0g c0b)
              (Gfx.fill_rect (+ x 1) (- Y1 5) (- UNIT 1) 10 c1r c1g c1b))
          (if (and (> j 0) (not (== ln (get_item u_lane (- j 1)))))
              (if (== j (- u_n 1))
                  (Gfx.draw_line x (- Y0 13) x (+ Y1 13) 255 236 130)   ; 最新一次续体调用
                  (Gfx.draw_line x (- Y0 8) x (+ Y1 8) 152 140 74))
              #f)
          (set! j (+ j 1)) })
      ;; 当前推进点：最新一格右侧的小方块
      (if (>= active 0)
          (if (== active 0)
              (Gfx.fill_rect (+ TX1 2) (- Y0 3) 3 7 c0r c0g c0b)
              (Gfx.fill_rect (+ TX1 2) (- Y1 3) 3 7 c1r c1g c1b))
          #f) }))

;; 段长标签：同一轨道连续跑 4 格以上的段，标出它跑了多少格
(define draw_runs
  (lambda ()
    (define j 0)
    (define ln 0)
    (define run 0)
    (define start 0)
    (define xa 0)
    (define xb 0)
    (define cx 0)
    { (set! j 0)
      (while (< j u_n)
        { (set! ln (get_item u_lane j))
          (set! start j)
          (set! run 0)
          (while (and (< j u_n) (== (get_item u_lane j) ln))
            { (set! run (+ run 1))
              (set! j (+ j 1)) })
          (if (>= run 4)
              { (set! xa (- TX1 (* (- u_n (+ start run)) UNIT)))
                (set! xb (- TX1 (* (- u_n start) UNIT)))
                (set! cx (- (Math.round (/ (+ xa xb) 2)) 4))
                (if (== ln 0)
                    (draw_num run cx (- Y0 19) d0r d0g d0b)
                    (draw_num run cx (- Y1 19) d1r d1g d1b)) }
              #f) }) }))

;; 说明文字（每个例子五句）
(define draw_explain
  (lambda ()
    (if (== demo 0)
        { (Gfx.draw_text "阴 @ = 第一个 call/cc 抓下的续体 k1" 6 EXPLAIN_Y 176 190 205)
          (Gfx.draw_text "阳 * = 第二个 call/cc 抓下的续体 k2" 6 (+ EXPLAIN_Y 14) 176 190 205)
          (Gfx.draw_text "调用 k1 → 跳回阴那行：阴再走一步，阳又走一步" 6 (+ EXPLAIN_Y 28) 150 165 182)
          (Gfx.draw_text "输出 @ * @ ** @ *** …（阴 1 步，阳递增）" 6 (+ EXPLAIN_Y 42) 150 165 182)
          (Gfx.draw_text "竖线 = 一次续体调用，控制从一条轨跳到另一条" 6 (+ EXPLAIN_Y 56) 120 138 158) }
        #f)
    (if (== demo 1)
        { (Gfx.draw_text "生成器：函数体被 call/cc 切成可续跑的片段" 6 EXPLAIN_Y 176 190 205)
          (Gfx.draw_text "产（绿）= 生成器试除一步，取（橙）= 调用者取值" 6 (+ EXPLAIN_Y 14) 176 190 205)
          (Gfx.draw_text "每次调用不是从头跑，而是跳回上次 yield 的续体" 6 (+ EXPLAIN_Y 28) 150 165 182)
          { (Gfx.draw_text "最近产出素数 " 6 (+ EXPLAIN_Y 42) 150 165 182)
            (draw_num g_last 96 (+ EXPLAIN_Y 42) 150 165 182) }
          (Gfx.draw_text "没有线程、没有栈复制：只有一个被切片的执行位置" 6 (+ EXPLAIN_Y 56) 120 138 158) }
        #f)
    (if (== demo 2)
        { (Gfx.draw_text "两个协程各存一个续体，靠调用对方的续体切换" 6 EXPLAIN_Y 176 190 205)
          (Gfx.draw_text "A 让出 → B 跑 → B 让出 → A 跑，如此往复" 6 (+ EXPLAIN_Y 14) 176 190 205)
          (Gfx.draw_text "每回合步数递增，于是两条轨的段越来越长" 6 (+ EXPLAIN_Y 28) 150 165 182)
          (Gfx.draw_text "切换的代价 = 一次 call/cc，不需要调度器线程" 6 (+ EXPLAIN_Y 42) 150 165 182)
          (Gfx.draw_text "这里的调度器就是一个 if：轮到谁就恢复谁的续体" 6 (+ EXPLAIN_Y 56) 120 138 158) }
        #f)))

(define draw_all
  (lambda ()
    { (Gfx.clear)
      (Gfx.draw_text title 6 2 215 232 255)
      (Gfx.draw_text idx_str 240 2 130 150 175)
      (draw_ribbon)
      (draw_status)
      (draw_timeline)
      (draw_runs)
      (draw_explain)
      (Gfx.draw_text "点按屏幕换下一个例子 · 右上角退出" 6 HINT_Y 96 112 132) }))

;; ---- 一帧：处理输入 + 整屏重绘 + 推屏 -------------------------------------

;; 点按换例子。注意本 App 每 60ms 才轮询一次触屏：只看"边沿"会整段错过一次
;; 快速点按（电平在两次轮询之间起落），故按"电平 + 一次按下只认一次"判定。
;; 右上角「退出」按钮由宿主在触摸边沿判定（App.set_exit_button），这里按坐标排除，
;; 免得点退出时顺带换题。
(define tap_armed 0)

(define in_exit_button
  (lambda (px py)
    (if (and (>= px (- W 40)) (< px (- W 4)))
        (if (and (>= py 2) (< py 18)) 1 0)
        0)))

(define handle_tap
  (lambda ()
    (if (Touch.is_available)
        { (Touch.take_press)              ; 消费边沿锁存，避免陈旧事件
          (if (Touch.is_pressed)
              (if (== tap_armed 0)
                  { (set! tap_armed 1)
                    (if (== (in_exit_button (Touch.x) (Touch.y)) 0)
                        (set! skip_req 1)
                        #f) }
                  #f)
              (set! tap_armed 0)) }
        #f)))

(define frame
  (lambda ()
    { (EX.check)          ; 硬 A/ESC 与右上角「退出」按钮（可能弹全局确认框）
      (handle_tap)        ; 点按屏幕 → 换例子
      (draw_all)
      (EX.draw_button)
      (Gfx.refresh) }))

;; ---- 一个例子的完整一轮：抓住逃逸续体，跑到底（或用户点按换题）------------

(define run_demo
  (lambda (d)
    { (set_demo d)
      (call/cc (lambda (esc)
                 { (set! esc_k esc)
                   (if (== d 0)
                       (d1_puzzle)
                       (if (== d 1)
                           (d2_run)
                           (d3_run))) })) }))

(define main_loop
  (lambda (d)
    { (run_demo d)
      (main_loop (mod (+ d 1) 3)) }))

;; ---- 启动 ---------------------------------------------------------------

(alloc_buffers)
(alloc_digits)
(EX.install)
(main_loop 0)