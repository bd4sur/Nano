;; ============================================================================
;; 康威生命游戏（Conway's Game of Life）—— Nano「小游戏」Scheme App
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;; 参考实现：ui_app.c 的 STATE_GAMEOFLIFE（C 版）：双缓冲原地写、环绕边界、
;; 50% 随机撒种、每帧演进一代。本版与之一致：每轮事件循环演进一代（= 每帧一代）。
;;
;; ---------------------------------------------------------------------------
;; 性能设计（2026-11 重写；同规模 40×30 下每代 342k VM 指令 → 96k，wasm32 上
;; 8.4 ms/代 → 0.79 ms/代，约 10.6×）
;;
;; 上一版的三个成本来源（解释器插桩按 opcode 归因实测）：
;;   ① **每格 9 次闭包调用**：cell-at 被 neighbours 调 8 次、自身判定再调 1 次。
;;      每次 λ 调用要分配闭包帧并走调用/返回，wasm32 上约 160 ns，9 次/格就占了
;;      全部开销的 ~2/3；
;;   ② **热循环里访问模块级变量**：模块顶层 define 在闭包体内是「自由变量快照」，
;;      每次读取都要沿闭包链 deref（约 30 ns，是本地槽位的 3 倍）。上一版每格
;;      要读 ROWS/COLS 等模块量 ~6 次（全代 6.8 万次 deref）；
;;   ③ **mod 求环绕下标**：每次邻域读取两次 fmod（浮点取余），全代 2.16 万次。
;;
;; 本版的对策：
;;   · **padding 环形缓冲**：世界按 (COLS+2)×(ROWS+2) 存放，四周多一圈镜像副本，
;;     每代结束把逻辑区的末行/首行、末列/首列抄到环带上（见 mirror）。于是热循环里
;;     任何环绕运算都不需要——8 个邻居全是 idx+常量（±1、±(PW-1)、±PW、±(PW+1)），
;;     一次加法即得下标。与 lenia.scm 的 padding 卷积同一手法；
;;   · **单层循环体、零闭包调用**：整个演进内联在 step 里，每格只有 8 次 get_item
;;     + 7 次加法 + 1 次 set_item!，没有一次过程调用；
;;   · **常量搬进本地槽位**：cols/rows/pw/d41… 一律在 step 开头 define 成本地槽位，
;;     热循环只读槽位，不走 deref；
;;   · **规则化简**：alive = (n==3) ? 1 : (n==2 ? 自身 : 0)——邻居数恰为 2 时
;;     “是否存活”就等于“自身是否存活”，于是省掉无条件读取自身与一次比较；
;;     存活计数直接 count += alive（alive ∈ {0,1}）；
;;   · **网格避开 HUD**：细胞从 HUD（12px）下方开始铺，不在状态条底下白算格子。
;;
;; 于是可以负担细网格与快节拍：CELL=3 → 106×76 = 8056 格（ESP32 取 6 px），每帧一代
;; （浏览器 rAF 下 60 代/秒；一代 ~700k 指令 ≈ 4.7 ms）。网格规模只由 CELL 一个数
;; 决定（改大 → 格少而大、更快；改小 → 格多而细、更慢），时间片申请随之缩放。
;;
;; 顶栏与 C 版**逐字一致**：`康威生命游戏 | 迭代:N | 存活:M | 密度:D.DD%`，12px 高
;; (39,39,39) 底 + 12px 二值点阵白字、位置 (0,0)（见 status 注释里的 C 源码对照）。
;; 实测两者 HUD 的白像素逐列剖面在公共前缀上完全相同（tools/e2e/browser/grab_fb.js
;; 抓两份 320x240 帧缓冲比对）。
;;
;; 定时器设 1 ms 而不是「每帧毫秒数」：上游运行时每轮事件循环对同一进程只触发一次
;; 定时器回调，1 ms 的间隔等价于“本轮有空就跑一代”，节奏天然跟随宿主帧率
;; （这与 C 版“每帧一代”的语义一致，且不受 System.timestamp 在 wasm32 上被压成
;; 27 位 uint 的影响——那个时间戳的差值不能用来做节流判断）。
;;
;; 单帧预算：宿主每帧只驱动解释器一个时间片（默认 312500 条指令）。4560 格需要
;; ~400k → 超过默认值就会被抢占拆到下一帧（帧率不变，但推进速度直接减半）。
;; 故用 (App.set_timeslice ...) 按格数申请（90 指令/格 + 30% 余量；宿主夹在默认值
;; 的 1/10 ~ 4 倍之间）。
;;
;; 其它实现要点（沿用上一版已验证的结论）：
;;   · 世界为一维列表（解释器列表是数组存储，get_item/set_item! O(1)）；
;;   · 双缓冲零分配：seed 一次性 push 构建两页（'() 字面量是模块级静态槽，
;;     仅可安全 push 一次），此后每代 set_item! 原地写目标页并翻页；
;;   · 栈纪律：解释器行“单值栈纪律”——任何表达式求值后恰好在操作数栈留 1 个值。
;;     Gfx 绘图函数与 push/set_item! 为语句型（弹参后压 1 个 #undefined，由编译器
;;     在语句位置弹出），仍直接写在语句位置即可；System.set_interval 产值（计时器
;;     编号），以 set! 消费；
;;   · 算术一律二元嵌套（本解释器编译器对多元乘法只发射一次 mul）。
;;
;; 规则（B3/S23）：活细胞邻居数为 2 或 3 时存活；死细胞恰有 3 个邻居时新生。
;; 正确性交叉验证：padding 版与上一版（mod 环绕）在同一种子下逐代比对 60 代、
;; 逐格一致（见 tools/e2e/perf/gol_verify.scm）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native App)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 网格参数 --------------------------------------------------------------
;; HUD_H = 12：与 C 版参考实现（ui_app.c 的 STATE_GAMEOFLIFE）的状态条同高；网格从
;; y=12 起铺满余下画布，不在状态条底下白算格子。
;; CELL（细胞边长像素）按平台取两档 —— 解释器每代成本 ≈「格数 × 87 条 VM 指令」，
;; 而宿主每帧只有一个时间片，格子越细单帧越贵：
;;   · 桌面 / 浏览器（wasm-gui / web / tty / linux）：3 px → 106×76 = **8056 格**
;;     （一代 ~700k 指令；headless chromium 实测 4.7 ms/代，仍是 60 代/秒、60 fps）；
;;   · ESP32（m5esp）：6 px → 53×38 = 2014 格 —— 解释器比桌面慢约一个数量级，
;;     细格会把单帧拖到几十毫秒、界面明显卡顿。
;; 上一版是「40×30 = 1200 格 / 10 代每秒」；本轮是 8056 格 / 60 代每秒。
;; 「退出」按钮改到右下角（C 版没有这个按钮；顶栏文字要与 C 版逐字一致，右上角
;; 40x18 的位置会压住"密度"字段的尾巴）。

(define CELL (if (String.equals (App.platform) "m5esp") 6 3))
(define HUD_H 12)                ; 顶部状态条高度（同 C 版）
(define SW (Gfx.width))
(define COLS (Math.floor (/ (Gfx.width) CELL)))
(define ROWS (Math.floor (/ (- (Gfx.height) HUD_H) CELL)))
(define CELLS (* COLS ROWS))
(define PW (+ COLS 2))           ; padding 宽度（左右各 1 列镜像）
(define PH (+ ROWS 2))           ; padding 高度（上下各 1 行镜像）
(define N (* PW PH))

;; ---- 单帧预算：一代必须在一个时间片内跑完 ---------------------------------
;; 宿主每帧只驱动解释器一个时间片（默认 312500 条 VM 指令，见 ui_scheme_app.c 的
;; MB_TIMESLICE_TICKS）。本 App 一代约「格数 × 87」条指令：桌面档 8056 格要 ~700k、
;; 上一版 1200 格只要 108k —— 超过默认值就会被抢占拆到下一帧，帧率不变但推进速度
;; 直接减半（60→30 代/秒）。故按格数申请：87 指令/格 + 35% 余量（随 CELL 自动缩放）；
;; 宿主夹在默认值的 1/10 ~ 4 倍之间，桌面档实际用到约 4.7 ms/帧。

(App.set_timeslice (* CELLS 117))

;; ---- 世界状态：双缓冲两页 + 当前页码（0=A 1=B）与代数 ---------------------

(define world-a '())
(define world-b '())
(define cur 0)
(define gen 0)

;; ---- 环形边界：把逻辑区的首末行/列镜像到 padding 环带 ----------------------
;; 顺序：先补左右两列（逐逻辑行），再整体补上下两行（逐列，含四角）——后者读的
;; 正是前者刚补好的行，因此四角一次到位。每代 2×(ROWS+COLS+2) 次 set_item!，
;; 相对每代数千格的计算量可忽略，但换来热循环里彻底没有环绕运算。

(define mirror
  (lambda (w) {
    (define pw PW)
    (define cols COLS)
    (define rows ROWS)
    (define ph PH)
    (define y 1)
    (define base 0)
    (define x 0)
    (while (< y (+ rows 1)) {
      (set! base (* y pw))
      (set_item! w base (get_item w (+ base cols)))              ; 左环带 ← 本行最右逻辑格
      (set_item! w (+ base (- pw 1)) (get_item w (+ base 1)))    ; 右环带 ← 本行最左逻辑格
      (set! y (+ y 1))
    })
    (set! x 0)
    (while (< x pw) {
      (set_item! w x (get_item w (+ x (* rows pw))))              ; 上环带 ← 末逻辑行
      (set_item! w (+ x (* (- ph 1) pw)) (get_item w (+ x pw)))   ; 下环带 ← 首逻辑行
      (set! x (+ x 1))
    })
  }))

;; ---- 随机撒种：50% 密度（同 C 版）；push 尾插保序，一次性构建两页 ----------
;; 环带先按随机值填（随后由 mirror 覆盖成镜像，逻辑区才是真正的初始状态）。

(define seed
  (lambda () {
    (define wa '())
    (define wb '())
    (define i 0)
    (define s 0)
    (while (< i N) {
      (set! s (if (< (Math.random) 0.5) 1 0))
      (push wa s)
      (push wb s)
      (set! i (+ i 1))
    })
    (set! world-a wa)
    (set! world-b wb)
    (mirror world-a)
    (mirror world-b)
  }))

;; ---- 顶部状态条：与 C 版逐字一致 ------------------------------------------
;; C 版（ui_app.c 的 STATE_GAMEOFLIFE）：
;;   swprintf(text, 100, L"康威生命游戏 | 迭代:%u | 存活:%u | 密度:%.2f%%",
;;            step_count, total_count, (float)total_count / (float)(w*h) * 100);
;;   gfx_draw_rectangle(gfx, 0, 0, w, 12, 39, 39, 39, 3);    ; mode 3 = **加色**混入
;;   gfx_draw_textline(gfx, text, 0, 0, 255, 255, 255, 1);   ; 12px 二值点阵、覆盖写
;; 本版：网格从 y=12 起铺（HUD 底下不铺格），条底恒为黑 → 加色与覆盖写在黑底上
;; 等价，直接 fill_rect 即可。字体必须显式切到 **0 = 12px 二值点阵**：Gfx.draw_text
;; 的默认字模是 1（12px 抗锯齿），与 C 版的 gfx_draw_textline 点阵字肉眼可辨。

;; 百分比两位小数（等价 printf "%.2f"）。
;; **不能用 atom_to_string 拼**：wasm32 的浮点尾数只有约 19 位，8.27 会打印成
;; "8.26999"（见 doc/AGENTS.md 的踩坑记录）。做法：先把百分比四舍五入成"百分位
;; 整数" q（如 827），再拆成整数段与两位小段分别转字符串。
(define pct2
  (lambda (num den) {
    ;; q = 百分位整数 = round(百分比 × 100)。先做浮点除法再乘 10000，避免整数溢出
    ;; （27 位 uint 上限约 1.34 亿，格数×10000 在大网格上会溢出）。
    (define q (Math.round (* 10000 (/ num den))))
    (define ip (Math.floor (/ q 100)))            ; 整数段
    (define fp (- q (* ip 100)))                  ; 小数两位（0..99）
    (String.concat (String.atom_to_string ip)
      (String.concat "." (if (< fp 10)
                             (String.concat "0" (String.atom_to_string fp))
                             (String.atom_to_string fp))))
  }))

(define status
  (lambda (alive_n) {
    (Gfx.fill_rect 0 0 SW HUD_H 39 39 39)
    (Gfx.set_font 0)
    (Gfx.draw_text (String.concat "康威生命游戏 | 迭代:"
                     (String.concat (String.atom_to_string gen)
                       (String.concat " | 存活:"
                         (String.concat (String.atom_to_string alive_n)
                           (String.concat " | 密度:"
                             (String.concat (pct2 alive_n CELLS) "%"))))))
                   0 0 255 255 255)
  }))

;; ---- 演进一代：读当前页 → 规则计算 → set_item! 原地写目标页 → 翻页 --------
;;（由定时器每轮事件循环调用一次；末尾推屏——宿主不主动推屏，避免代间半成品闪烁）

(define step
  (lambda () {
    (EX.check)
    (define src (if (== cur 0) world-a world-b))
    (define dst (if (== cur 0) world-b world-a))
    ;; 热循环用到的常量一律搬进本地槽位（模块级变量是自由变量快照，读取要走 deref）
    (define cols COLS)
    (define rows ROWS)
    (define pw PW)
    (define ph PH)
    (define cellpx CELL)
    (define hudpx HUD_H)
    (define d43 (+ PW 1))
    (define d42 PW)
    (define d41 (- PW 1))
    (define y 1)
    (define rowstart 0)
    (define rowend 0)
    (define idx 0)
    (define n 0)
    (define alive 0)
    (define count 0)
    (define py 0)
    (Gfx.clear)
    (while (< y (+ rows 1)) {
      (set! rowstart (* y pw))
      (set! idx (+ rowstart 1))
      (set! rowend (+ rowstart (+ cols 1)))
      (set! py (+ hudpx (* (- y 1) cellpx)))
      (while (< idx rowend) {
        ;; 8 邻域和（padding 布局 → 下标全是 idx±常量）
        (set! n (+ (get_item src (- idx d43))
                   (+ (get_item src (- idx d42))
                      (+ (get_item src (- idx d41))
                         (+ (get_item src (- idx 1))
                            (+ (get_item src (+ idx 1))
                               (+ (get_item src (+ idx d41))
                                  (+ (get_item src (+ idx d42))
                                     (get_item src (+ idx d43))))))))))
        ;; B3/S23（n==2 分支里的“自身”只在需要时读一次）
        (set! alive (if (== n 3) 1 (if (== n 2) (get_item src idx) 0)))
        (set_item! dst idx alive)
        (set! count (+ count alive))
        (if (== alive 1)
            (Gfx.fill_rect (* (- idx rowstart) cellpx) py cellpx cellpx 0 255 255)
            #f)
        (set! idx (+ idx 1))
      })
      (set! y (+ y 1))
    })
    (mirror dst)
    (set! cur (if (== cur 0) 1 0))
    (set! gen (+ gen 1))
    (status count)
      ;; C 版没有「退出」按钮（它用硬 A/ESC）；触屏 App 必须自备入口，且顶栏要留给
      ;; 与 C 版一致的文字，故放右下角（热区同由 (App.set_exit_button) 登记）。
      (EX.draw_button_at (- SW 40) (- (Gfx.height) 18))
      (Gfx.refresh)
  }))

;; ---- 启动：撒种 → 演进并绘制第一代 → 每轮事件循环演进一代 ------------------

(seed)
(step)
(set! _timer (System.set_interval 1 step))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行
(EX.install)
