;; ============================================================================
;; 康威生命游戏（Conway's Game of Life）—— Nano「小游戏」Scheme App
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;; 参考实现：ui_app.c 的 STATE_GAMEOFLIFE（C 版）：双缓冲原地写、环绕边界、
;; 50% 随机撒种。差异：本版细胞 8px（C 版 2px——解释器每代计算量按格数线性
;; 增长，8px 在帧预算内可稳定 10 代/秒）；经 System.set_interval 以 100ms/代
;; 节流——解释器由宿主 GUI 按时间片驱动，定时器回调即演进一代。
;;
;; 实现要点：
;;   - 世界为一维列表（解释器列表为数组存储，get_item/set_item! O(1)）；
;;   - 双缓冲零分配：seed 一次性 push 构建两页（'() 字面量是模块级静态槽，
;;     仅可安全 push 一次），此后每代 set_item! 原地写目标页并翻转页码；
;;   - 邻居读取用显式环绕坐标（坐标先加网格尺寸再 mod，规避负数取模；
;;     mod 恒为浮点，get_item/set_item! 对浮点索引按 roundf 取整）；
;;   - 栈纪律：Gfx 绘图函数与 push/set_item! 为语句型（弹参不压值），直接
;;     写在语句位置；System.set_interval 产值（计时器编号），以 set! 消费；
;;   - 算术一律二元嵌套（本解释器编译器对多元乘法只发射一次 mul）。
;;
;; 规则（B3/S23）：活细胞邻居数为 2 或 3 时存活；死细胞恰有 3 个邻居时新生。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)

;; ---- 网格参数（细胞 8px；网格随画布尺寸自适应）----------------------------

(define CELL 8)
(define COLS (Math.floor (/ (Gfx.width) CELL)))
(define ROWS (Math.floor (/ (Gfx.height) CELL)))
(define SW (Gfx.width))

;; ---- 世界状态：双缓冲两页 + 当前页码（0=A 1=B）与代数 ---------------------

(define world-a '())
(define world-b '())
(define cur 0)
(define gen 0)

;; ---- 环绕读取一个细胞（idx = y*COLS + x）----------------------------------
;; 坐标先加网格尺寸再取模（结果恒非负）；返回 0/1。

(define cell-at
  (lambda (w x y)
    (get_item w (+ (* (mod (+ y ROWS) ROWS) COLS) (mod (+ x COLS) COLS)))))
(define neighbours
  (lambda (w x y)
    (+ (cell-at w (- x 1) (- y 1)) (+ (cell-at w x (- y 1)) (+ (cell-at w (+ x 1) (- y 1)) (+ (cell-at w (- x 1) y) (+ (cell-at w (+ x 1) y) (+ (cell-at w (- x 1) (+ y 1)) (+ (cell-at w x (+ y 1)) (cell-at w (+ x 1) (+ y 1)))))))))))

;; ---- 随机撒种：50% 密度（同 C 版）；push 尾插保序，一次性构建两页 ----------

(define seed
  (lambda () {
    (define wa '())
    (define wb '())
    (define y 0)
    (while (< y ROWS) {
      (define x 0)
      (while (< x COLS) {
        (define s (if (< (Math.random) 0.5) 1 0))
        (push wa s)
        (push wb s)
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
    (set! world-a wa)
    (set! world-b wb)
  }))

;; ---- 顶部状态条：生命游戏 N 代  存活 M ------------------------------------

(define status
  (lambda (alive_n) {
    (Gfx.fill_rect 0 0 SW 14 24 24 28)
    (Gfx.draw_text (String.concat "生命游戏 " (String.concat (String.atom_to_string gen) (String.concat " 代  存活 " (String.atom_to_string alive_n))))
                   2 1 180 220 255)
  }))

;; ---- 演进一代：读当前页 → 规则计算 → set_item! 原地写目标页 → 翻页 --------
;;（由定时器每 100ms 调用一次；末尾推屏——宿主不主动推屏，避免代间半成品闪烁）

(define step
  (lambda () {
    (define src (if (== cur 0) world-a world-b))
    (define dst (if (== cur 0) world-b world-a))
    (define y 0)
    (define x 0)
    (define n 0)
    (define alive 0)
    (define count 0)
    (Gfx.clear)
    (while (< y ROWS) {
      (set! x 0)
      (while (< x COLS) {
        (set! n (neighbours src x y))
        (if (== (cell-at src x y) 1)
            (if (or (== n 2) (== n 3)) (set! alive 1) (set! alive 0))
            (if (== n 3) (set! alive 1) (set! alive 0)))
        (set_item! dst (+ (* y COLS) x) alive)
        (if (== alive 1) {
          (Gfx.fill_rect (* x CELL) (* y CELL) CELL CELL 0 255 255)
          (set! count (+ count 1))
        })
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
    (set! cur (if (== cur 0) 1 0))
    (set! gen (+ gen 1))
    (status count)
    (Gfx.refresh)
  }))

;; ---- 启动：撒种 → 演进并绘制第一代 → 100ms 周期演进 -----------------------

(seed)
(step)
(set! _timer (System.set_interval 100 step))
