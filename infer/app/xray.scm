;; ============================================================================
;; 解释器 X 光 —— 灵机应用程序（自观测仪表盘）
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库 + Diag 本地库
;;
;; 这个 App 不画风景，画的是"正在运行它的那台虚拟机"：
;;   · 内存：VM 工作区与进程堆的 used/capacity（System.memstat）
;;   · 现场：操作数栈深度、调用栈帧数、PC、指令区长度（Diag.opstack/fstack/pc/ilcode）
;;   · 闭包：按帧描述符创建的闭包数 / 退化为"复制全部绑定"的闭包数 / 槽位总数
;;           / 自由变量槽数 / 走"沿链通用解析"的次数（Diag.frames/legacy/slots/…）
;;   · GC  ：全局与进程级 GC 计数、每次回收量、以及本 App 每 tick 主动制造分配
;;           把堆推到阈值后触发 GC —— 曲线因此出现锯齿，能直接"看见 GC 在工作"
;;   · 吞吐：用 Diag.ms（**单调实时钟**，不受宿主虚拟时钟影响）在固定时间预算里
;;           数 lambda 调用次数，得到实时的"调用/毫秒"
;;
;; 用途：既是演示，也是本仓库排查解释器问题的现场仪器 —— 此前定位"闭包帧/槽位"
;;       类缺陷最费力的就是拿不到现场数据，这里把栈深、帧描述符命中率、退化闭包数
;;       都做成了常驻读数（Diag 本地库见 infer/am_native_Diag.h）。
;;
;; 书写约定（本解释器的坑，详见 app/turtle.scm 头部）：算术一律二元嵌套；
;; cond 子句只能带一个表达式且子句多了会编译失败，故热路径统一用平铺 if。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Diag)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 布局与节奏 ------------------------------------------------------------

(define W (Gfx.width))            ; 320
(define H (Gfx.height))           ; 240
(define TICK_MS 50)               ; 采样周期

(define BAR_X 8)
(define BAR_W 304)
(define VM_BAR_Y 16)
(define HEAP_BAR_Y 32)
(define BAR_H 12)

(define PLOT_X 8)
(define PLOT_Y0 66)               ; 100% 所在行
(define PLOT_Y1 146)              ; 0% 所在行
(define PLOT_H 80)
(define NS 88)                    ; 曲线采样点数
(define STEP 3)                   ; 每点像素宽

;; ---- 环形缓冲（模块级预建一次，之后只 set_item! 覆写）----------------------

(define hist_heap (quote ()))
(define hist_vm (quote ()))
(define hist_gc (quote ()))
(define head 0)

(define init_hist
  (lambda ()
    (while (< (length hist_heap) NS) { (push hist_heap 0) })
    (while (< (length hist_vm) NS) { (push hist_vm 0) })
    (while (< (length hist_gc) NS) { (push hist_gc 0) })))

(define ring_put
  (lambda (buf idx v) (set_item! buf idx v)))

;; ---- 状态 ------------------------------------------------------------------

(define tick 0)
(define vm_used 0) (define vm_cap 1)
(define heap_used 0) (define heap_cap 1)
(define heap_pct 0) (define vm_pct 0)
(define gc_events 0) (define last_gc_kb 0)
(define calls_per_ms 0) (define probe_runs 0)

;; 分配演示：每 tick 往 scratch 里 push 若干项；堆到阈值就释放并 GC
(define scratch (cons 0 (quote ())))
(define ALLOC_PER_TICK 12000)
(define GC_THRESHOLD 45)

;; HUD 字符串暂存（见 draw_readouts 注释）
(define ds "")

;; 吞吐探针
(define probe_budget_ms 3)
(define probe_batch 300)

;; 被调用的目标函数（探针用它产生真实的 call+return 开销）
(define probe_fn
  (lambda (x) (+ x 1)))

;; ---- 采样 ------------------------------------------------------------------

(define sample_mem
  (lambda ()
    (define mem 0)
    (define h 0)
    (define v 0)
    (set! mem (System.memstat))
    (set! vm_cap (get_item mem 0))
    (set! vm_used (get_item mem 1))
    (set! heap_cap (get_item mem 2))
    (set! heap_used (get_item mem 3))
    (set! vm_pct (Math.floor (* 100.0 (/ (+ vm_used 0.0) (+ vm_cap 0.0)))))
    (set! heap_pct (Math.floor (* 100.0 (/ (+ heap_used 0.0) (+ heap_cap 0.0)))))
    (set! h heap_pct)
    (set! v vm_pct)
    (ring_put hist_heap head h)
    (ring_put hist_vm head v)
    (set! head (mod (+ head 1) NS))))

;; 分配 → 到阈值 → 释放并 GC（回收量记入 last_gc_kb，供面板显示）
(define alloc_step
  (lambda ()
    (define i 0)
    (define before 0)
    (define mem 0)
    (ring_put hist_gc head 0)
    (if (< heap_pct GC_THRESHOLD)
        { (set! i 0)
          (while (< i ALLOC_PER_TICK) { (push scratch i) (set! i (+ i 1)) }) }
        { (set! before heap_used)
          (set! scratch (cons 0 (quote ())))     ; 丢弃旧表 → 变成垃圾
          (System.gc)
          (set! mem (System.memstat))
          (set! last_gc_kb (Math.floor (/ (- before (get_item mem 3)) 1024)))
          (set! gc_events (+ gc_events 1))
          (set! heap_used (get_item mem 3))
          (ring_put hist_gc (mod (+ head (- NS 1)) NS) 1) })))

;; 吞吐探针：在 probe_budget_ms 预算内数 probe_fn 的调用次数
(define run_probe
  (lambda ()
    (define t0 0)
    (define n 0)
    (define i 0)
    (define dt 0)
    (set! t0 (Diag.ms))
    (set! n 0)
    (while (< (- (Diag.ms) t0) probe_budget_ms) {
      (set! i 0)
      (while (< i probe_batch) { (probe_fn i) (set! i (+ i 1)) })
      (set! n (+ n probe_batch)) })
    (set! dt (- (Diag.ms) t0))
    (if (> dt 0) { (set! calls_per_ms (Math.floor (/ n dt))) })
    (set! probe_runs (+ probe_runs 1))))

;; ---- 绘制：基础件 ----------------------------------------------------------

(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))
(define cat4 (lambda (a b c d) (String.concat a (cat3 b c d))))
(define cat5 (lambda (a b c d e) (String.concat a (cat4 b c d e))))

(define num (lambda (x) (String.atom_to_string x)))

;; 一条水平进度条：外框 + 填充 + 右侧百分比与数值
(define draw_bar
  (lambda (y label pct used cap)
    (define w 0)
    (Gfx.draw_text label 8 y 120 150 180)
    (Gfx.draw_text (cat3 (num pct) "%  " (num (Math.floor (/ (+ used 0.0) 1024.0)))) 40 y 150 190 220)
    (Gfx.draw_text "K" (- W 24) y 90 112 132)
    (Gfx.draw_rect (+ 8 0) (+ y 13) 304 1 40 60 80)          ; 底线
    (set! w (Math.floor (/ (* pct 304) 100.0)))
    (if (< w 0) { (set! w 0) })
    (if (> w 304) { (set! w 304) })
    (if (> w 0) { (Gfx.fill_rect 8 (+ y 13) w 1 90 170 120) })))

;; ---- 绘制：曲线区 ----------------------------------------------------------

(define plot_y
  (lambda (pct)
    (define v pct)
    (if (> v 100) { (set! v 100) })
    (if (< v 0) { (set! v 0) })
    (- PLOT_Y1 (Math.floor (/ (* v PLOT_H) 100.0)))))

(define draw_plot
  (lambda ()
    (define i 0)
    (define a 0)
    (define b 0)
    (define x0 0)
    (define x1 0)
    ;; 网格与边框
    (Gfx.draw_rect (- PLOT_X 1) (- PLOT_Y0 1) (+ (* NS STEP) 2) (+ PLOT_H 2) 34 52 70)
    (Gfx.draw_line PLOT_X PLOT_Y0 (+ PLOT_X (* NS STEP)) PLOT_Y0 30 46 62)
    (Gfx.draw_line PLOT_X (plot_y 50) (+ PLOT_X (* NS STEP)) (plot_y 50) 30 46 62)
    (Gfx.draw_text "100" (- W 30) (- PLOT_Y0 6) 70 90 110)
    (Gfx.draw_text "50" (- W 24) (- (plot_y 50) 6) 70 90 110)
    (Gfx.draw_text "0" (- W 16) (- PLOT_Y1 6) 70 90 110)
    ;; VM 使用率（青）与 堆使用率（绿）两条滚动曲线
    (set! i 0)
    (while (< i (- NS 1)) {
      (set! a (mod (+ head i) NS))
      (set! b (mod (+ head (+ i 1)) NS))
      (set! x0 (+ PLOT_X (* i STEP)))
      (set! x1 (+ x0 STEP))
      (Gfx.draw_line x0 (plot_y (get_item hist_vm a)) x1 (plot_y (get_item hist_vm b)) 60 130 190)
      (Gfx.draw_line x0 (plot_y (get_item hist_heap a)) x1 (plot_y (get_item hist_heap b)) 110 220 130)
      ;; GC 事件：在底部打一根红色短竖线
      (if (== (get_item hist_gc a) 1)
          { (Gfx.draw_line x0 (- PLOT_Y1 6) x0 PLOT_Y1 220 80 70) }
          {})
      (set! i (+ i 1)) })
    (Gfx.draw_text "VM" (- W 30) (+ PLOT_Y0 10) 60 130 190)
    (Gfx.draw_text "堆" (- W 30) (+ PLOT_Y0 24) 110 220 130)))

;; ---- 绘制：读数区 ----------------------------------------------------------

(define draw_readouts
  (lambda ()
    (define y0 152)
    ;; 拼接结果先落到变量 ds 再传给 draw_text：把多层嵌套调用写在实参位置、且连续
    ;; 堆在同一个 lambda 里，会触发解释器缺陷（call: 错误的调用目标），详见
    ;; doc/AGENTS.md 2026-10-28「解释器 X 光」条目。
    (set! ds (cat3 "操作数栈 " (num (Diag.opstack)) ""))
    (Gfx.draw_text ds 8 y0 150 190 220)
    (set! ds (cat3 "调用栈 " (num (Diag.fstack)) " 帧"))
    (Gfx.draw_text ds 8 (+ y0 12) 150 190 220)
    (set! ds (cat4 "PC " (num (Diag.pc)) " / 指令 " (num (Diag.ilcode))))
    (Gfx.draw_text ds 8 (+ y0 24) 150 190 220)
    (set! ds (cat4 "闭包 " (num (Diag.frames)) " 退化 " (num (Diag.legacy))))
    (Gfx.draw_text ds 168 y0 150 190 220)
    (set! ds (cat4 "槽位 " (num (Diag.slots)) " 自由 " (num (Diag.freeslots))))
    (Gfx.draw_text ds 168 (+ y0 12) 150 190 220)
    (set! ds (cat4 "GC " (num gc_events) " 回收 " (num last_gc_kb)))
    (Gfx.draw_text ds 168 (+ y0 24) 150 190 220)))

(define draw_footer
  (lambda ()
    (set! ds (cat4 "吞吐 " (num calls_per_ms) " 调用/ms   采样 " (num probe_runs)))
    (Gfx.draw_text ds 8 190 210 200 130)
    (set! ds (cat4 "分配 " (num ALLOC_PER_TICK) " 项/tick  堆 " (cat3 (num heap_pct) "%" "")))
    (Gfx.draw_text ds 8 202 150 190 220)
    (set! ds (cat3 "阈值 " (num GC_THRESHOLD) "% 到阈值即释放并 GC（看曲线锯齿）"))
    (Gfx.draw_text ds 8 214 110 140 170)))

(define draw_header
  (lambda ()
    (Gfx.draw_text "解释器 X 光" 8 4 225 235 245)
    (set! ds (cat4 "t=" (num (Math.floor (/ (Diag.ms) 1000.0))) "s  tick " (num (Diag.tick))))
    (Gfx.draw_text ds 150 4 120 160 190)))

(define render
  (lambda ()
    (Gfx.clear)
    (draw_header)
    (draw_bar VM_BAR_Y "VM" vm_pct vm_used vm_cap)
    (draw_bar HEAP_BAR_Y "堆" heap_pct heap_used heap_cap)
    (draw_plot)
    (draw_readouts)
    (draw_footer)
      (EX.draw_button)
      (Gfx.refresh)))

;; ---- 主循环 ----------------------------------------------------------------

(define tick_fn
  (lambda ()
    (EX.check)
    (sample_mem)
    (alloc_step)
    (if (== (mod tick 6) 0) { (run_probe) } {})
    (set! tick (+ tick 1))
    (render)))

;; ---- 启动 ------------------------------------------------------------------

(init_hist)
(set! _timer (System.set_interval TICK_MS tick_fn))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 render 擦掉对话框残影）
(EX.install_with render)
