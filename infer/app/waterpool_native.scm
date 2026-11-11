;; ============================================================================
;; 水池（本地库加速版）—— 灵机应用程序
;;
;; 画面、交互、状态语义与 app/waterpool.scm（ui_water.c 的**逐函数直译**）完全一致，
;; 区别只在**抽象边界**：逐像素/逐格点的数值内核搬进了本地库 (native Water)，
;; 由 C 直接执行；本文件只剩 App 骨架（节拍、触屏电平取样、退出出口）。
;;
;; 为什么这么划边界（profile 见 doc/AGENTS.md 2026-11-11「水池（本地库版）」）：
;;   Scheme 版单帧约 2.2e8 条 IL 指令，逐段归因（x86 宿主，稳态）——
;;     焦散 pass 18% / 池壁 pass 28% / 水面 pass 46% / 圆球 1.5% / 波场+法线 3% /
;;     帧缓冲上屏 1.2% / 相机与 HUD < 0.1%
;;   即 98% 的代价都在「逐像素或逐格点」的内层数值循环里，而这些都是纯数值、
;;   **与解释器语义无关**的代码；交互与 HUD 可忽略。于是把「渲染器内核 + 它的全部
;;   状态」整体封装为一个专用本地库（am_native_Water.c，ui_water.c 的逐行副本），
;;   并把一整拍压成**一次调用** (Water.tick)，Scheme 侧不再逐像素/逐三角/逐格点。
;;
;; 与 ui_water.c 的关系：本地库的渲染内核是 ui_water.c 的逐行副本（相机、网格、
;;   光栅化、z-buffer、着色、焦散、波场、圆球物理、控件栏几何全部保留），因此本
;;   App 与设备侧 C 版水池**逐像素一致**（同一 float32 公式与同一绘制次序）。
;
;; 交互（与 C 版一致）：
;;   · 底部控件栏：返回 / 暂停·继续 / 重力开·关 / 光源（几何与配色同 C 版）；
;;   · 拖动空白处旋转视角（1px ≈ 1°，俯仰 ±89.999°）；拖动圆球移动圆球；
;;   · 「返回」→ 灵机 App 的统一退出出口（EX 的全局确认框，含「后台」）；
;;   · 硬 A/ESC → 同一出口（EX.check 每拍轮询宿主的返回请求）。
;;
;; 本地库接口（见 am_native_Water.h）：
;;   (Water.init)               分配缓冲 + 建网格 + 复位默认场景
;;   (Water.tick down x y dt)   一整拍：输入 → 物理 → 仿真 → 焦散 → 渲染 → 上屏 → HUD
;;                              返回 1 表示「返回」控件刚被松手点按
;;   (Water.probe)              诊断用 9 元数值列表
;;   (Water.dispose)            释放缓冲（确认退出时调用；「后台」冻结时不调用）
;;
;; 解释器约束（manual.md）：算术/比较恒二参数；无一元取负；同一 lambda 内不重复
;;   define 同名变量；块内 define 提升到 lambda 体首部。
;; ============================================================================

(native Gfx)
(native Math)
(native System)
(native Touch)
(native Water)                    ; 水池渲染内核（本地库，见 am_native_Water.h）
(import EX "app/lib/exit.scm")    ; 硬 A/ESC / 控件「返回」→ 宿主全局退出确认框

;; 节拍：8ms（**不是 16ms**）。宿主只在 GUI 帧边界（浏览器 rAF ≈16.7ms）检查定时器，
;; 16ms 周期在 16.7ms 边界上“有时隔两帧才到期”，实测只有 ~57fps；8ms 周期每帧必到期，
;; 实测稳定 60.0fps（与粒子生命的结论一致，见 doc/AGENTS.md）。仿真的步长取自墙钟
;; 差分（不是这个周期），因此调周期不影响水池物理/波场的推进速度。
(define TICK_MS 8)

(define tick_timer 0)
(define last_t 0)
(define now_t 0)
(define dt_ms 0)
(define dt_s 0.0)
(define down 0)
(define tx 0)
(define ty 0)
(define back 0)

;; 确认退出前释放本地库缓冲（「后台」冻结实例时不走这里，恢复后状态仍在）
(define water_release
  (lambda ()
    (Water.dispose)))

(define tick
  (lambda ()
    {
      ;; ① 退出出口（硬 A/ESC → 全局确认框；「返回」控件在 ③ 结算）
      (EX.check)

      ;; ② 触屏电平 → 时间步（dt 由墙钟差分给出，与 C 版取 global_state->timestamp 同源）
      (set! down (if (Touch.is_pressed) 1 0))
      (set! tx (Touch.x))
      (set! ty (Touch.y))
      (set! now_t (System.timestamp))
      (set! dt_ms (- now_t last_t))
      (set! last_t now_t)
      (if (< dt_ms 0) (set! dt_ms 0) #f)
      (if (> dt_ms 1000) (set! dt_ms 1000) #f)
      (set! dt_s (/ (* 1.0 dt_ms) 1000.0))

      ;; ③ 一整拍（输入 → 物理/仿真 → 渲染 → 上屏 → 控件栏）全在本地库里完成
      (set! back (Water.tick down tx ty dt_s))

      ;; ④ 「返回」控件在松手沿被点按 → 走灵机 App 的统一退出出口
      (if (== back 1) (EX.do_exit) #f)
    }))

;; ---- 启动 ------------------------------------------------------------------
(EX.install)
(EX.on_exit water_release)
(Water.init)
;; 首帧：dt=0 —— 只建焦散与画面、不推进物理（避免进 App 先闪一帧没有焦散的画面）
(Water.tick 0 0 0 0.0)
(set! last_t (System.timestamp))
(set! tick_timer (System.set_interval TICK_MS tick))
