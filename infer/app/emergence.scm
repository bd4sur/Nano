;@id=emergence
;@title=人工生命
;@version=2026-11-17
;; ============================================================================
;; 人工生命（Emergence / ALife）—— 灵机应用程序：四个"简单规则 → 复杂行为"的合集
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx / Touch / Key / Diag 本地库
;;
;; 本 App 把原先四个独立 App 合并成一页四签（底部页签栏），Tab / ←→ / 数字键 1~4
;; 或点页签切换：
;;     ① 生命游戏   app/lib/alife_gol.scm       （康威，B3/S23，离散二值 CA）
;;     ② 反应扩散   app/lib/alife_grayscott.scm （Gray-Scott PDE 数值解）
;;     ③ Lenia      app/lib/alife_lenia.scm     （连续态元胞自动机，环形核卷积）
;;     ④ 粒子生命   app/lib/alife_particle.scm  （离格多体：距离核力自组织）
;;
;; 分工（这是本文件唯一的架构决定）：
;;   · 四个模型各自是一个**库**（app/lib/alife_*.scm），内部状态全部是模块级变量；
;;     它们不再自装定时器、不再推屏、不再管退出按钮。对外只有四个入口：
;;         init   —— 一次性初始化（撒种/建核/分配表）
;;         tick   —— 推进一拍（内部可能被抢占拆到多帧，宿主不关心）
;;         draw   —— 只重绘整屏，不改状态（切页回来/模态框关闭后补帧）
;;         budget —— 本页需要的单帧时间片（见下）
;;   · 本 App 只做一个"单定时器 + 页签 + 退出按钮"的外壳：每拍先处理输入，再按当前
;;     页的节拍决定要不要调它的 tick，然后画页签栏、画「退出」按钮、推屏。
;;
;; 为什么切换能"保持状态不变"：状态活在库的模块级变量里，切页只是**不再调它的
;; tick**——不销毁、不重置、不重新初始化。切回来时用 draw 补一帧即可。
;; 反过来说，后台页**不推进**（省 CPU；四个模型都在同一进程里跑会很重）。
;;
;; 节拍（为什么不是每拍都 tick）：一个进程只能有一个定时器（上游运行时每轮事件循环
;; 对同一进程只触发一个定时器回调，见 app/lib/exit.scm 头部），所以四页共用一个 8ms
;; 的定时器（≈每个宿主帧一次回调），每页再按「每几拍走一步」自己的节奏：
;;     生命游戏 每拍 1 代（≈60 代/秒，与原 App 一致）
;;     反应扩散 每 6 拍 1 步（≈10 步/秒，与原 App 的 wasm 实测 8~12fps 同量级）
;;     Lenia    每 3 拍 1 步（≈20 步/秒，原 App 50ms/拍 × 2 步的实测同量级）
;;     粒子生命 每拍 1 步（≈60 步/秒，与原 App 一致）
;; 计时用**回调计数**而不是 (Diag.ms) 的差值：宿主只在本轮有空时才触发回调，错过的
;; 周期会被合并，按墙上时间做累加会让慢机上的页跳步；按帧计数则每页的"一步/几帧"
;; 在任何帧率下都是同一个观感。
;;
;; 单帧预算（App.set_timeslice）：宿主每帧只驱动解释器一个时间片（默认 312500 条
;; 指令）。各页需求差一个数量级，故**切页时**按该页的 budget 申请；超过上限的部分
;; （反应扩散/Lenia 的一步）本来就会被抢占拆到相邻几帧里跑完，与它们作为独立 App
;; 时的行为一致。
;;
;; 版面（320×240）：
;;   y 0..221   当前页的内容区（各页自己带顶栏，自己决定怎么排）
;;   y 222..239 页签栏（18px）：4 个页签（64×16，间隔 2px，从 x=2 起）+ 右下「退出」
;; 页签栏由本 App **在页的内容画完之后**覆盖绘制，所以各页可以放心整屏重绘
;; （包括 Gfx.clear）；各页只需把内容铺到 y<222 为止（TAB_H=18 在四个库里都是常量）。
;;
;; 输入：
;;   · 触屏：点页签切换（只吞掉落在页签栏里的按下沿，页的内容区照旧收得到触摸）；
;;   · 键盘：（native Key）Tab / → 下一签，← 上一签，1~4 直跳；
;;   · 硬 A/ESC：与其它 App 一致——(EX.check) 弹全局退出确认框。
;;
;; 解释器约束（manual.md + doc/AGENTS.md，逐条遵守）：算术恒二参数；无 list/let；
;; lambda 体内 define 只放在开头；`'()` 是模块级静态字面量，只可安全 push 一次；
;; 语句位置的 if 两个分支都要留下值（否则操作数栈失衡）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native App)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC + 底部「退出」按钮 → 全局退出确认框

;; 四个模型（原 app/gol.scm / grayscott.scm / lenia.scm / particlelife.scm，
;; 2026-11 并入本 App 后降格为库；库名不能带点号，别名不能与被导入名冲突）
(import GOL "app/lib/alife_gol.scm")
(import GS  "app/lib/alife_grayscott.scm")
(import LN  "app/lib/alife_lenia.scm")
(import PL  "app/lib/alife_particle.scm")

;; ---- 版面几何 --------------------------------------------------------------

(define W (Gfx.width))            ; 320
(define H (Gfx.height))           ; 240
(define TAB_N 4)                  ; 页签数
(define TB_H 18)                  ; 页签栏高度（与四个库里的 TAB_H 常量必须一致）
(define TB_Y (- H TB_H))          ; 页签栏顶边 = 222
(define TB_X 2)                   ; 第一个页签的左边
(define TB_W 64)                  ; 页签宽度
(define TB_GAP 2)                 ; 页签间距
;; 「退出」按钮：右下角（页签栏右侧的空位）。宽度取 EX.btn_w 而不是写死——
;; 那个按钮加宽过（36 → 46），写死会把按钮推出屏幕右缘。
(define EB_X (- W (+ EX.btn_w 4)))   ; 270
(define EB_Y (+ TB_Y 1))             ; 223
(define TICK_MS 8)                ; 单定时器周期（≈每个宿主帧一次回调）

;; ---- 当前页与节拍计数 ------------------------------------------------------

(define cur 0)                    ; 当前页（0..3）
(define acc 0)                    ; 距上一拍已过的回调数（到 pace 就推进一步）
(define tpx 0)                    ; 触屏坐标临时量（块内不能 define）
(define tpy 0)
(define tprev 0)                  ; 上一拍是否按住（电平→边沿）
(define dirty 1)                  ; 本拍画面是否需要推屏（省掉"什么都没变"的重复 refresh）
(define thit -1)                  ; 页签命中结果临时量（-1 = 没命中）
(define key 0)                    ; Key.take_press 的临时量
(define _timer 0)                 ; 定时器编号（只为让下面那句 set! 有绑定）

;; 每页的节拍：每几个回调走一步（见文件头的「节拍」）
(define pace_of
  (lambda (i)
    (if (== i 0) 1
        (if (== i 1) 6
            (if (== i 2) 3 1)))))

;; 每页的单帧时间片（各库自己声明，这里只做分发）
(define budget_of
  (lambda (i)
    (if (== i 0) GOL.budget
        (if (== i 1) GS.budget
            (if (== i 2) LN.budget PL.budget)))))

;; 页签文案（12px 字：4 个全角字 = 48px，加左内边距 8px = 56px < 64px 可用宽度）
(define label_of
  (lambda (i)
    (if (== i 0) "生命游戏"
        (if (== i 1) "反应扩散"
            (if (== i 2) "Lenia" "粒子生命")))))

;; 页名（诊断行用）
(define name_of
  (lambda (i)
    (if (== i 0) "gol"
        (if (== i 1) "grayscott"
            (if (== i 2) "lenia" "particlelife")))))

;; 该页已推进的步数（诊断用：切页时打一行，e2e 据此断言"切页不重置状态"）
(define steps_of
  (lambda (i)
    (if (== i 0) (GOL.steps)
        (if (== i 1) (GS.steps)
            (if (== i 2) (LN.steps) (PL.steps))))))

;; ---- 分发：推进 / 重绘（用平铺 if，不建闭包表）-----------------------------

(define run_tick
  (lambda ()
    (if (== cur 0) (GOL.tick) #f)
    (if (== cur 1) (GS.tick) #f)
    (if (== cur 2) (LN.tick) #f)
    (if (== cur 3) (PL.tick) #f)))

(define run_draw
  (lambda ()
    (if (== cur 0) (GOL.draw) #f)
    (if (== cur 1) (GS.draw) #f)
    (if (== cur 2) (LN.draw) #f)
    (if (== cur 3) (PL.draw) #f)))

;; ---- 页签栏 ----------------------------------------------------------------

;; 页签 i 的左边（2, 68, 134, 200；右端 264，给 x=270 的退出按钮留 6px）
(define tab_x
  (lambda (i)
    (+ TB_X (* i (+ TB_W TB_GAP)))))

;; 命中的页签下标；不在页签栏里返回 -1（触屏判定用，坐标为像素）
(define tab_hit
  (lambda (px py)
    (if (< py TB_Y)
        -1
        (if (>= py (+ TB_Y TB_H))
            -1
            (if (< px TB_X)
                -1
                (if (< px (+ TB_X (* TAB_N (+ TB_W TB_GAP))))
                    (Math.floor (/ (- px TB_X) (+ TB_W TB_GAP)))
                    -1))))))

;; 画页签栏：深底 + 顶边细线 + 4 个页签（当前页高亮）+ 不画退出按钮（调用方画）
(define draw_tabs
  (lambda ()
    (define i 0)
    (define x 0)
    (Gfx.set_font 1)                       ; 12px 抗锯齿字（字模是跨 App 共享的静态量）
    (Gfx.fill_rect 0 TB_Y W TB_H 16 17 21)
    (Gfx.fill_rect 0 TB_Y W 1 58 60 72)    ; 顶边细线：内容区 / 页签栏分界
    (set! i 0)
    (while (< i TAB_N) {
      (set! x (tab_x i))
      (if (== i cur)
          { (Gfx.fill_rect x (+ TB_Y 2) TB_W (- TB_H 4) 96 152 216)
            (Gfx.draw_text (label_of i) (+ x 8) (+ TB_Y 2) 10 14 22) }
          { (Gfx.fill_rect x (+ TB_Y 2) TB_W (- TB_H 4) 38 40 50)
            (Gfx.draw_text (label_of i) (+ x 8) (+ TB_Y 2) 188 196 212) })
      (set! i (+ i 1)) })))

;; 右下角的「退出」按钮（位置由 EX 记录；热区由 (App.set_exit_button) 登记）
(define draw_exit
  (lambda ()
    (EX.draw_button_at EB_X EB_Y)))

;; ---- 整帧重绘（EX 的"留下"之后 / 从后台恢复时用）-------------------------

(define redraw
  (lambda ()
    {
      (run_draw)
      (draw_tabs)
      (draw_exit)
      (Gfx.refresh)
    }))

;; ---- 切页 ------------------------------------------------------------------

;; 切到第 i 页：换时间片 → 立刻补一帧。状态由各库自己保留，这里不做任何重置。
(define switch_to
  (lambda (i)
    (if (== i cur)
        #f
        (if (< i 0)
            #f
            (if (>= i TAB_N)
                #f
                {
                  (set! cur i)
                  (set! acc 0)
                  (set! dirty 1)
                  (App.set_timeslice (budget_of i))
                  (run_draw)
                  (display "[emergence] tab=") (display i)
                  (display " ") (display (name_of i))
                  (display " steps=") (display (steps_of i))
                  ;; 四页各自已推进的步数（gol/grayscott/lenia/particlelife）：
                  ;; e2e 用它断言"切页保留状态、后台页不推进"（切页那一刻打一行）。
                  (display " all=") (display (GOL.steps)) (display "/") (display (GS.steps))
                  (display "/") (display (LN.steps)) (display "/") (display (PL.steps)) (newline)
                })))))

(define next_tab
  (lambda ()
    (switch_to (mod (+ cur 1) TAB_N))))

(define prev_tab
  (lambda ()
    (switch_to (mod (+ cur (- TAB_N 1)) TAB_N))))

;; ---- 输入：触屏页签 --------------------------------------------------------

;; 与 particlelife 的「重置」按钮同一手法：用**电平 + 自算边沿**判定，
;; 命中才取走按下沿（吞掉这一下，页的内容区不会把它当成自己的点按）。
(define read_touch
  (lambda ()
    (if (Touch.is_available)
        {
          (if (Touch.is_pressed)
              {
                (if (== tprev 0)
                    {
                      (set! dirty 1)
                      (set! tpx (Touch.x))
                      (set! tpy (Touch.y))
                      (if (>= tpy TB_Y)
                          { (set! thit (tab_hit tpx tpy))
                            (if (>= thit 0) (switch_to thit) #f)
                            (Touch.take_press) }
                          #f)
                    }
                    #f)
                (set! tprev 1)
              }
              { (if (== tprev 1) (set! dirty 1) #f) (set! tprev 0) })
        }
        #f)))

;; ---- 输入：键盘 ------------------------------------------------------------

;; 一次取完本拍所有按下沿（不取完会在队列里攒着，切页会"连跳好几页"）。
;; 键码见 manual / hal_key.h：tab=9 left=3 right=4，'1'..'4' = 49..52。
;; 27（ESC/A）不在这里处理——它由宿主记成返回请求，交给 (EX.check)。
(define handle_key
  (lambda (k)
    (if (== k 9) (next_tab) #f)
    (if (== k 3) (prev_tab) #f)
    (if (== k 4) (next_tab) #f)
    (if (>= k 49) { (if (<= k 52) (switch_to (- k 49)) #f) } #f)))

(define read_key
  (lambda ()
    (if (Key.is_available)
        {
          (set! key (Key.take_press))
          (while (> key 0)
            {
              (set! dirty 1)
              (handle_key key)
              (set! key (Key.take_press))
            })
        }
        #f)))

;; ---- 主循环 ----------------------------------------------------------------

(define on_tick
  (lambda ()
    {
      (EX.check)                 ; 返回请求 / 右下角「退出」按钮（宿主在触摸边沿判定）
      (read_touch)               ; 点页签切换
      (read_key)                 ; Tab / ←→ / 1~4
      (set! acc (+ acc 1))
      (if (>= acc (pace_of cur))
          { (set! acc 0) (set! dirty 1) (run_tick) }
          #f)
      (draw_tabs)
      (draw_exit)
      ;; 只有"画的内容真的变了"才推屏：慢节拍的页（反应扩散/ Lenia）不必每拍把同一帧
      ;; 再推给宿主一次。页签栏/退出按钮仍然每拍重画（它们压在页的内容之上，页自己
      ;; 整屏重绘会盖掉），但重画本身不产生推屏。
      (if (== dirty 1)
          { (set! dirty 0) (Gfx.refresh) }
          #f)
    }))

;; ---- 启动 ------------------------------------------------------------------
;;   四个库的 init 都跑一遍（各自撒种/建核/分配表）。之后每页的状态一直留在自己的
;;   模块里，切页只换"谁在被 tick"。

(GOL.init)
(GS.init)
(LN.init)
(PL.init)
(App.set_timeslice (budget_of 0))
(Gfx.clear)
(GOL.draw)
(draw_tabs)
(draw_exit)
(Gfx.refresh)
(display "[emergence] boot tabs=") (display TAB_N) (newline)
(set! _timer (System.set_interval TICK_MS on_tick))

;; ---- 退出：硬 A/ESC / 右下角按钮 → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → redraw 擦掉对话框残影
(EX.install_with redraw)
