;; ============================================================================
;; 水波（Water Ripple）—— 灵机应用程序
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库 + Touch 本地库
;; 参考实现：infer/ui_ripple.c 的 STATE_RIPPLE（C 版，网页版水波纹的定点移植）
;;
;; 算法（与 ui_ripple.c 同一套物理，逐条对应）：
;;   · 波场是二维波动方程的显式差分：
;;         amp' = (上 + 下 + 左 + 右) * 0.5 − amp_上一时刻
;;         amp' = amp' − amp'/32                     （衰减，C 版是算术右移 5 位）
;;     双缓冲：本拍读"当前页"、写"另一页"，写完翻页（对应 C 版的 old_index/new_index）。
;;   · 触点激发：把触点落在的格子及其邻域**累加**震源振幅（C 版 wr_disturb 同款）。
;;   · 显示：位移量 disp = 1024 − amp（1024 = 无位移），以画面中心为原点做径向缩放采样底图：
;;         src = (dst − 中心) * disp/1024 + 中心
;;     disp > 1024 处向外鼓（水面凹陷的折射），< 1024 处向内缩（水面隆起）——即 C 版
;;     wr_render 里那两行 dx/dy 位移公式。
;;
;; 与 C 版的差异（都源于"算力在解释器里"这一硬约束）：
;;   ① **波场分辨率 80×60（每格 4×4 像素）**，C 版是 160×120（半分辨率）。
;;      实测（浏览器 wasm32，本机 Chromium）单步耗时：80×60 → 18.5 ms，
;;      120×90 → 40 ms，160×120 → 72 ms；30fps 只有 33 ms 预算，故取 80×60
;;      （留出底图绘制与 UI 的余量，慢机器上也不至于掉帧）。
;;      格子变大 2 倍 ⇒ 波前每拍跨过的"格子数"不变而格子本身变大，波纹在
;;      屏幕上横向传播**约快 2 倍**（观感仍是标准水波，只是涟漪散得更利索）。
;;      整拍（激发 + 推进 + 折射绘制 + 按钮 + 推屏）在浏览器里实测稳定 29~30 fps。
;;   ② **逐像素纹理采样交给宿主**（(Gfx.draw_image_warp)，见 am_native_Gfx.h）。
;;      解释器不可能每帧读 320×240 个像素；底图解码/缩放/采样全在 C 侧完成
;;      （纹理走图形层的 LRU 缓存，像素不进解释器堆）。为了让 4×4 的粗格子
;;      不露出块状边界，位移场按**格中心双线性插值**放大到像素，纹理也做双线性
;;      过滤。C 版是"2×2 块最近邻直写"，折射纹理会带一层细密的方块感；
;;      本版是平滑折射——同一套位移公式，观感更"水面"一些。
;;   ③ 触屏输入走 (native Touch) 的电平采样（与 C 版从 key_event 取触屏电平同义）；
;;      退出走灵机 App 的统一出口（app/lib/exit.scm：右上角「退出」按钮 + 硬 A/ESC
;;      弹宿主全局确认框），而不是 C 版直接切 STATE_GAME_MENU。
;;
;; 底图路径：**wp.png**（相对业务资源根）。C 版是 PLATFORM_ROOT_DIR "/wp.png"，
;; 两者指向同一个文件（桌面/WASM 上是 /home/bd4sur/linglong/wp.png，掌机上是 SD 卡根），
;; 路径规则见 am_native_Fs.h。读不到就画错误画面（文案与坐标同 C 版），
;; 并切到一个 200ms 的慢拍——只为让硬 A/ESC 仍能弹退出确认框。
;;
;; 书写约定（本解释器的坑，详见 app/turtle.scm 头部）：算术一律二元嵌套；
;; 下标运算走 Math.floor（列表索引虽会四舍五入，显式取整更稳）；不用 let/list/map；
;; while 循环体不是独立作用域（循环内 define 会提升到 lambda 体首部，故循环里复用
;; 同名临时量是安全的）；`(cons 0 '())` 建出的表含一个哑元素，须先 pop 掉。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC / 右上角按钮 → 全局退出确认

;; ---- 底图与画布 -------------------------------------------------------------

(define IMG "wp.png")             ; 底图（与 ui_ripple.c 的 WR_IMAGE_PATH 同一文件）
(define SW (Gfx.width))           ; 画布宽（320）
(define SH (Gfx.height))          ; 画布高（240）

;; ---- 波场几何 ---------------------------------------------------------------
;; 格子边长 4px（C 版 2px，见头部差异 ①）；四周各留一圈**零填充**（上下左右都留），
;; 于是邻格读取不必做边界判定——比 C 版的"只填上下两行 + 左右分支"少两条 if。
;; 页布局：P × PH（P = GW+2，PH = GH+2），两页首尾相接放在同一个列表里。

(define CELL 4)
(define GW (Math.floor (/ SW CELL)))          ; 80 个格子
(define GH (Math.floor (/ SH CELL)))          ; 60 个格子
(define PADW 2)                               ; 每行两侧各 1 格填充
(define P (+ GW PADW))                        ; 行跨度（含填充）
(define PH (+ GH PADW))                       ; 行数（含填充）
(define PAGE (* P PH))                        ; 单页元素数
(define CELLS (* GW GH))                      ; 有效格子数
;; 一行走完后游标要多跳这么多格才落到下一行首格：循环里每处理一格 mi 就 +1（含最后一格），
;; 所以行末 mi 已指向本行右侧填充格，再跳过它与本行末尾、下一行左侧填充格共 2 格。
(define RADV PADW)

;; ---- 物理常数（衰减与位移基准值同 ui_ripple.c；震源两项按粗格子调整，见下）----

(define ATTEN 0.03125)            ; 衰减 = 每拍减去振幅的 1/32（C 版右移 5 位）
(define NEUTRAL 1024.0)           ; disp 的"无位移"基准值（C 版 WR_MAX_AMPLITUDE）
(define SRC_AMP 1024.0)           ; 震源振幅（C 版 WR_SOURCE_AMPLITUDE 是 512：
                                  ; 粗格子上的位移场要被双线性放大，单次点按的视觉强度
                                  ; 会打折，故加倍补偿；按住连发时振幅累积，量级仍与 C 版相当）
(define DROP_R 2)                 ; 波源半径（格）：5×5 格 = 20×20 px，与 C 版 7×7 个
                                  ; 半分辨率格（14×14 px）的落点面积同量级（略大）
(define TICK_MS 33)               ; 30 fps（C 版跟随渲染帧率，约 20~30 fps）

;; ---- 波场状态 ---------------------------------------------------------------
;; map  ：两页振幅场（长度 2*PAGE，零初始 = 水面平静）
;; disp ：逐格位移量（长度 CELLS，初值 NEUTRAL = 无位移），每拍原地刷新，
;;        直接交给宿主做位移采样（对应 C 版的 last_map）
;; pcur ：本拍**读**（= 下一拍被当作 u_n）的页基址；pnxt：本拍**写**的页基址

(define map '())
(define disp '())
(define pcur 0)
(define pnxt PAGE)

(define clampg
  (lambda (v lo hi)
    (if (< v lo) lo (if (> v hi) hi v))))

;; ---- 初始化：两页振幅场 + 位移表（逐项 push，一次建好，此后只 set_item!）-----

(define init_fields
  (lambda () {
    (define m (cons 0 '()))
    (define d (cons 0 '()))
    (pop m) (pop d)                       ; 去掉 (cons 0 '()) 的哑元素
    (define n 0)
    (while (< n (* 2 PAGE)) { (push m 0.0) (set! n (+ n 1)) })
    (set! n 0)
    (while (< n CELLS) { (push d NEUTRAL) (set! n (+ n 1)) })
    (set! map m)
    (set! disp d)
    (set! pcur 0)
    (set! pnxt PAGE)
  }))

;; ---- 触点激发：(px,py) 所在的格子及邻域累加震源振幅（C 版 wr_disturb）------

(define disturb
  (lambda (px py) {
    (define cx (Math.floor (/ px CELL)))
    (define cy (Math.floor (/ py CELL)))
    (define x0 (clampg (- cx DROP_R) 0 (- GW 1)))
    (define x1 (clampg (+ cx DROP_R) 0 (- GW 1)))
    (define y0 (clampg (- cy DROP_R) 0 (- GH 1)))
    (define y1 (clampg (+ cy DROP_R) 0 (- GH 1)))
    (define y y0)
    (while (<= y y1) {
      (define x x0)
      (while (<= x x1) {
        ;; 页内线性下标：页基址 + (y+1)*P + (x+1)（+1 跳过填充行/列）
        (define idx (+ (+ pcur P 1) (+ (* y P) x)))
        (set_item! map idx (+ (get_item map idx) SRC_AMP))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
  }))

;; ---- 波场推进一拍 + 刷新位移表 ---------------------------------------------
;; 逐格：读 4 邻格与"上一时刻的自己"，算新振幅写进另一页，同时把位移量写进 disp。
;; mi/ni 是两页各自的游标（每行末尾跳 2 格填充列），ci 是 disp 的线性下标。

(define step
  (lambda () {
    (define mi (+ pcur P 1))
    (define ni (+ pnxt P 1))
    (define ci 0)
    (define oy 0)
    (while (< oy GH) {
      (define ox 0)
      (while (< ox GW) {
        (define s (+ (+ (get_item map (- mi P)) (get_item map (+ mi P)))
                     (+ (get_item map (- mi 1)) (get_item map (+ mi 1)))))
        (define amp (- (* s 0.5) (get_item map ni)))
        (define na (- amp (* amp ATTEN)))
        (set_item! map ni na)
        (set_item! disp ci (- NEUTRAL na))
        (set! mi (+ mi 1))
        (set! ni (+ ni 1))
        (set! ci (+ ci 1))
        (set! ox (+ ox 1))
      })
      (set! mi (+ mi RADV))
      (set! ni (+ ni RADV))
      (set! oy (+ oy 1))
    })
    ;; 翻页：本拍写的页成为下一拍的"当前页"
    (define tmp pcur)
    (set! pcur pnxt)
    (set! pnxt tmp)
  }))

;; ---- 绘制：底图按位移场折射（逐像素采样在宿主侧完成）----------------------

(define render
  (lambda ()
    (Gfx.draw_image_warp IMG 0 0 SW SH GW GH disp NEUTRAL)))

;; ---- 每一拍：退出检查 → 触点激发 → 推进 → 绘制 → 按钮 → 推屏 --------------

(define tick
  (lambda () {
    (EX.check)
    (if (Touch.is_pressed)
        (disturb (Touch.x) (Touch.y))
        #f)
    (step)
    (render)
    (EX.draw_button)     ; 右上角「退出」（触屏唯一出口，硬 A/ESC 走确认框）
    (Gfx.refresh)
  }))

;; ---- 底图读不到：黑底红字提示（文案与 ui_ripple.c 的错误画面一致），等 A 返回 --
;;   错误态同样跑一个慢拍（200ms）：① 让 (EX.check) 能接管硬 A/ESC（否则要等宿主
;;   的 1500ms 兜底）；② 退出确认框关闭后把画面重画回来（零快照约定）。

(define draw_error
  (lambda () {
    (Gfx.clear)
    (Gfx.set_font 2)                       ; 16px 抗锯齿字模（标题，同 C 版）
    (Gfx.draw_text "水波" 6 2 255 255 255)
    (Gfx.set_font 1)                       ; 恢复 12px 抗锯齿（正文，同 C 版）
    (Gfx.draw_text (String.concat "读取 " (String.concat IMG " 失败")) 6 100 255 80 80)
    (Gfx.draw_text "请将图片放入资源根目录" 6 120 180 180 180)
    (Gfx.draw_text "A返回" 6 220 180 180 180)
    (EX.draw_button)
    (Gfx.refresh)
  }))

(define error_tick
  (lambda () {
    (EX.check)
    (draw_error)
  }))

;; ---- 启动 ------------------------------------------------------------------
;;  ① 先探测底图（Gfx.image_size 只读文件头，代价极小）；
;;  ② 可读 → 建波场 → 画一帧平静水面 → 33ms/拍推进；
;;  ③ 不可读 → 错误画面 + 200ms 慢拍（只跑 (EX.check)，保持硬 A/ESC 可退出）。

(define _timer 0)

(EX.install)

(if (null? (Gfx.image_size IMG))
    {
      (draw_error)
      (set! _timer (System.set_interval 200 error_tick))
    }
    {
      (init_fields)
      (render)
      (EX.draw_button)
      (Gfx.refresh)
      (set! _timer (System.set_interval TICK_MS tick))
    })