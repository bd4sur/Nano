;; ============================================================================
;; Bad Apple!! —— 灵机应用程序（Scheme 版 BadApple 播放器）
;;
;; 对标的 C 实现：infer/ui_app.c 的 STATE_BADAPPLE + ui_app_badapple_render_frame()
;;   画法完全一致：清屏 → 屏心 64×64 二值位图（白点）→ 底部居中「帧号 / 总帧数」。
;;   帧数据来源也一致：infer/badapple.h 的 bad_apple_10fps_64x64（2193 帧 64×64）。
;;
;; 为什么数据要另存一份文本（app/lib/badapple/frames.txt）：
;;   Scheme 看不到 C 的静态数组，而 (native Fs) 只做文本读取，所以由
;;   infer/tools/gen_badapple_data.py 从 badapple.h 生成「定长记录」文本
;;   （一帧一条记录，偏移 = 帧号 × rec_bytes），每帧只读自己那一条（≤656 字节）。
;;   记录格式与编码见该脚本头部；meta.scm（同目录，脚本生成）提供帧数/记录长度等
;;   常量，App 只 import 它，不手抄数字。
;;
;; 帧率：**严格锁定 10fps**（每帧 100ms 墙钟），用定时器驱动：
;;   · 不用 (System.set_interval 100)：周期定时器是按「触发时刻 + 间隔」重排的，而
;;     宿主事件循环只在 GUI 帧边界（浏览器 rAF ≈16.7ms、掌机 30fps）触发定时器，
;;     于是每周期的实际长度被向上量化到帧边界整数倍（100ms 常变成 ~117ms），
;;     表现为约 8.5fps 且随渲染抖动漂移；
;;   · 改为 (System.set_timeout) 自链式「绝对时刻表」：第 k 帧的到期时刻恒为
;;     t0 + k×100ms（t0 = 起播时刻），每次只睡「到下一个整百毫秒还差多少」。
;;     触发时刻仍受帧边界量化（抖动 ≤ 一帧），但**没有累积漂移**：长时间看严格 10fps。
;;   · 迟到超过一个周期（卡顿/切后台）时按墙钟**跳帧**推进（画面永远对齐墙钟，
;;     而不是放慢播放），与 C 版「按 timestamp 判定该显示第几帧」的语义一致。
;;
;; 定时器链与数值精度（wasm32 宿主！）：
;;   · 32 位宿主的无符号整数逻辑位宽只有 27 位、浮点有效尾数只有约 19 位，且
;;     (System.timestamp) 返回的是「纪元毫秒 mod 2^27」。因此本 App **不做除法**，
;;     帧号 n 与时刻全部用整数加减推进（浮点只用于比较与画坐标），并把时间基每播完
;;     一轮（2193 帧 = 219300ms）整体前移，使参与运算的量始终远小于 2^19；
;;   · 数据偏移 = n × rec_bytes 必须是**整数**（浮点在大数上会丢低位，读错字节）。
;;
;; 数据未就绪（WASM 网页目标把 app/ 资源按优先级后台挂载）时显示「载入中」并每拍
;; 重试；文件被截断/缺失则显示提示而不是崩溃。
;;
;; 退出：装 app/lib/exit.scm（硬 A/ESC 与右上角「退出」按钮 → 宿主全局确认框）。
;; ============================================================================

(native Fs)
(native Gfx)
(native Math)
(native String)
(native System)

(import BA "app/lib/badapple/meta.scm")
(import EX "app/lib/exit.scm")

;; ---- 数据文件与记录布局（全部常量来自 meta.scm，App 不手抄数字）-------------
;; 记录格式（见 infer/tools/gen_badapple_data.py）：
;;   head_chars 个字符 = 段数（base base_radix），随后每段 chars_per_run 个字符
;;   (y, x, len-1)，字符值 = ASCII - base_char。
;;   段 = 一行里连续的一段白色像素（用一条 1px 高的 fill_rect 画完）。
(define PATH "app/lib/badapple/frames.txt")
(define NFRAMES BA.n_frames)
(define REC BA.rec_bytes)
(define FILE_BYTES (* NFRAMES REC))
(define SLOT_MS BA.slot_ms)                ;; 每帧 100ms → 10fps
(define CYCLE_MS (* SLOT_MS NFRAMES))      ;; 一轮播放的时长（219300ms）
(define FW BA.frame_w)
(define FH BA.frame_h)
(define CH0 BA.base_char)                  ;; 字符 → 数值：(- c CH0)
(define RADIX BA.base_radix)               ;; 段数头：c0×RADIX + c1
(define HEAD BA.head_chars)
(define CPR BA.chars_per_run)

;; ---- 屏幕布局（与 C 版一致：屏心 64×64，底部居中帧号）---------------------
(define SW (Gfx.width))
(define SH (Gfx.height))
(define X0 (- (Math.floor (/ SW 2)) (Math.floor (/ FW 2))))
(define Y0 (- (Math.floor (/ SH 2)) (Math.floor (/ FH 2))))
(define LABEL_Y (- SH 16))
(define LABEL_CHARS 11)                     ;; "0000 / 2193"：11 个半角字符
(define FONT (Gfx.font_info 1))             ;; 12px 平滑字模：(半角步进 全角步进 行高 基线)
(define HALF_W (get_item FONT 0))
(define FULL_W (get_item FONT 1))
(define LABEL_X (Math.floor (/ (- SW (* LABEL_CHARS HALF_W)) 2)))

;; ---- 播放状态 --------------------------------------------------------------
(define ready 0)        ;; 0=数据未就绪 1=可播放 2=数据异常
(define waits 0)        ;; 已重试次数（用于区分「还在挂载」与「真的没有这份数据」）
(define t0 0)           ;; 本轮播放的时间基（(System.timestamp) 整数毫秒）
(define n -1)           ;; 当前帧号（整数；-1 = 尚未画过任何一帧）
(define _timer 0)       ;; 当前挂着的定时器编号（只作诊断用）
(define dbg 0)          ;; 1 = 打印诊断行（控制台可见，便于回归脚本断言）

;; ---- 小工具 ----------------------------------------------------------------
;; 字符 → 记录里的数值（编码时 A[v] = chr(base_char + v)）
(define cv (lambda (c) (- c CH0)))

;; 帧号补零到 4 位（帧号 ≤ 2192，最多 4 位）
(define pad4
  (lambda (v) {
    (define s (String.concat "000" (String.atom_to_string v)))
    (String.slice s (- 0 4) (String.length s))
  }))

;; ---- 画一帧位图：每段白色像素一条 1px 高的 fill_rect -----------------------
;; 平均 94 段/帧，最多 218 段 → 每帧约 100 次 native 调用，解释器开销极小。
(define draw_bitmap
  (lambda (fi) {
    (define rec (Fs.read_range PATH (* fi REC) REC))
    (if (null? rec)
        #f
        {
          (define s (get_item rec 0))
          (define cnt (+ (* (cv (String.charCodeAt 0 s)) RADIX) (cv (String.charCodeAt 1 s))))
          (define i 0)
          (define p 0)
          (define y 0)
          (define x 0)
          (define l 0)
          (while (< i cnt) {
            (set! p (+ HEAD (* i CPR)))
            (set! y (cv (String.charCodeAt p s)))
            (set! x (cv (String.charCodeAt (+ p 1) s)))
            (set! l (+ 1 (cv (String.charCodeAt (+ p 2) s))))
            (Gfx.fill_rect (+ X0 x) (+ Y0 y) l 1 255 255 255)
            (set! i (+ i 1))
          })
        })
  }))

;; 底部居中的「帧号 / 总帧数」
(define draw_label
  (lambda (fi) {
    (Gfx.draw_text (String.concat (pad4 fi) (String.concat " / " (pad4 NFRAMES)))
                   LABEL_X LABEL_Y 255 255 255)
  }))

;; 文本宽度（12px 字模：半角/全角各按字模步进累加；用于把提示语居中）
(define text_width
  (lambda (s) {
    (define i 0)
    (define w 0)
    (define sl (String.length s))
    (while (< i sl) {
      (set! w (+ w (if (< (String.charCodeAt i s) 128) HALF_W FULL_W)))
      (set! i (+ i 1))
    })
    w
  }))

(define center_x (lambda (s) (Math.floor (/ (- SW (text_width s)) 2))))

;; 整屏：清底 → 位图 → 帧号 → 退出按钮 → 推屏（推屏时机由 App 自己掌握，
;; 与 C 版一致：一帧只推一次，不出现半成品画面）
(define draw_scene
  (lambda (fi) {
    (Gfx.clear)
    (draw_bitmap fi)
    (draw_label fi)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; 提示画面：第一行中文、第二行路径（ASCII）。分行画而不是拼成一行——12px 平滑字模
;; 里部分汉字字干比字宽（字身 14px / 字步 12px），长串中文挤在一行会互相压边。
(define draw_message2
  (lambda (line1 line2) {
    (Gfx.clear)
    (Gfx.draw_text line1 (center_x line1) (- (Math.floor (/ SH 2)) 22) 220 226 240)
    (if (String.equals line2 "") #f (Gfx.draw_text line2 (center_x line2)
                                                       (- (Math.floor (/ SH 2)) 4) 150 158 175))
    (EX.draw_button)
    (Gfx.refresh)
  }))

(define draw_message (lambda (msg) (draw_message2 msg "")))

;; ---- 数据就绪检测（网页目标按优先级后台挂载，可能晚于 App 启动）------------
;; 每拍 200ms 重试；等超过 4s 仍无此文件，就把提示换成具体路径（掌机/桌面没部署数据时
;; 不会无限停在「载入中」，能直接看出缺什么）。
(define try_load
  (lambda () {
    (define sz (Fs.size PATH))
    (if (null? sz)
        {
          (set! waits (+ waits 1))
          (if (> waits 20)
              (draw_message2 "缺帧数据" PATH)
              (draw_message "Bad Apple!! 载入中…"))
        }
        (if (== sz FILE_BYTES)
            {
              (set! ready 1)
              (set! t0 (System.timestamp))
              (set! n -1)
            }
            {
              (set! ready 2)
              (draw_message2 "帧数据损坏" PATH)
              (if (== dbg 1) { (display "[badapple] bad size=") (display sz) (newline) } #f)
            }))
  }))

;; ---- 定时器链 --------------------------------------------------------------
;; 每拍：
;;   ① 把帧号推进到「墙钟此刻应有的帧」（整数加减；跨轮则时间基整体前移一个周期）；
;;   ② 有推进才重绘（同一槽位内重复触发不重绘，省电）；
;;   ③ 睡到**下一帧的绝对到期时刻** t0 + (n+1)×100ms —— 不是「睡 100ms」，
;;      因此定时器触发的帧边界量化不会累积成漂移。
(define tick
  (lambda () {
    (EX.check)
    (if (== ready 0) (try_load) #f)
    (if (== ready 1)
        {
          (define now (System.timestamp))
          (define elapsed (- now t0))
          (define advanced 0)

          ;; 长时间卡顿/切后台：按整轮回卷，保持「帧号 = 墙钟槽位 mod 2193」的相位
          (while (>= elapsed CYCLE_MS) {
            (set! elapsed (- elapsed CYCLE_MS))
            (set! n 0)
            (set! advanced 1)
            (set! t0 (- now elapsed))
          })

          ;; 推进到此刻应有的帧（迟到多个周期则跳帧，画面始终对齐墙钟）
          (while (>= elapsed (* SLOT_MS (+ n 1))) {
            (set! n (+ n 1))
            (set! advanced 1)
            (if (>= n NFRAMES) {
              (set! n 0)
              (set! elapsed (- elapsed CYCLE_MS))
              (set! t0 (- now elapsed))
            } #f)
          })

          (if (== advanced 1) {
            (draw_scene n)
            (if (== dbg 1) { (display "[badapple] frame=") (display n)
                             (display " t=") (display (- (System.timestamp) t0)) (newline) } #f)
          } #f)

          ;; 以「画完之后」的当前时刻重算到下一槽位的剩余，避免渲染耗时吃进下一拍
          (define el2 (- (System.timestamp) t0))
          (define delay (- (* SLOT_MS (+ n 1)) el2))
          (if (< delay 0) (set! delay 0) #f)
          (set! _timer (System.set_timeout delay tick))
        }
        (set! _timer (System.set_timeout 200 tick)))
  }))

;; ---- 启动 ----------------------------------------------------------------
(EX.install)
(tick)
