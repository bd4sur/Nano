;; ============================================================================
;; 灵机 App 退出辅助库（Exit）—— 让硬 A/ESC 拥有 J2ME 那样的“挂断键”语义
;;
;; 为什么需要它：
;;   2026-10 的接口改造后，宿主不再抢着处理 A/ESC（见 infer/am_native_App.h）：
;;   它只把“用户按了返回”记成一条**返回请求**。谁来消费这条请求？答案是 App 自己。
;;   本模块把“消费 → 弹全局退出确认框 → 收尾 → 退出 / 留下继续”这套动作、
;;   以及触屏上那个可见的「退出」按钮，封成几个可组合的小接口。每个 App 三行：
;;
;;       (import EX "app/lib/exit.scm")   ; 与其它 (native ...)/(import ...) 放一起
;;       (EX.install_with 重绘函数)        ; 启动时一次（只要不重绘就用 (EX.install)）
;;       (EX.check)                       ; **每拍**一次（画完按钮、推屏之前）
;;
;;   装了它的 App 立刻获得：
;;     · 硬 A/ESC（或网页上的 ESC 按钮）→ 宿主弹出全局确认框「是否退出？确认 / 留下」；
;;     · 屏幕右上角统一的「退出」按钮（(EX.draw_button) 画，见下）——触屏不再派生
;;       “返回”软按键，所以每个 App 都要有一个可见入口，这一行就够。
;;
;; 接口：
;;       (EX.install)               —— 记录屏幕宽度（画按钮的几何基准）
;;       (EX.install_with 重绘函数)  —— 另登记“留下”之后的重绘函数，擦掉对话框残影
;;       (EX.check)                 —— 每拍轮询：返回请求 + 右上角按钮
;;       (EX.draw_button)           —— 画右上角「退出」按钮（36x16；无触屏时不画）
;;       (EX.on_exit 收尾函数)       —— 用户确认退出后、实例销毁**前**调用一次：
;;                                     落盘、停外设、发“再见”帧都放这里。
;;
;; **轮询点是 App 自己的每拍**：装好之后再在每帧的绘制/逻辑路径里调一次
;;       (EX.check)
;;   （本目录各 App 都放在「画完右上角按钮、推屏之前」，即 (EX.draw_button) 之后、
;;     (Gfx.refresh) 之前。）这样一个 App 只保留**自己的**定时器，不会被本模块的
;;   定时器抢调度。
;;
;;   为什么按钮的命中判定交给宿主（(App.set_exit_button)）：
;;   本模块的 (EX.check) 跑在 App 自己的每拍上，慢节奏 App（例如星历 1000ms/拍）用
;;   "每拍采样触屏电平"根本抓不住一次 ~150ms 的点按——实测星历里点「退出」完全无反应。
;;   因此 (EX.draw_button) 会把按钮矩形登记给宿主，宿主在**触摸边沿**（事件级）判定命中，
;;   命中即等同硬 A/ESC 置"返回请求"，App 在下一次 (EX.check) 里照常弹全局确认框。
;;   本模块仍保留一份电平式的本地判定，作为没有该宿主服务的实现上的兜底。
;;
;;   为什么不再用本模块自己的 System.set_interval 轮询（2026-10-30 实测教训）：
;;   上游运行时每轮事件循环对**同一进程**只触发一个定时器回调（回调不可重入，
;;   其余的顺延到下一轮）。于是"App 的 tick + 本模块的 poll"两个 interval 会互相
;;   错开：实测 tick 间隔从 33ms 拉到约 63ms；而把 dt 钳到 50ms 防跳变的 App
;;   （如俄罗斯方块）就会每拍丢掉十几毫秒——重力、动画整体变慢。改为每拍
;;   (EX.check) 后，App 的节拍完全不受影响（探针 dt_sum 与墙钟一致）。
;;
;; 触屏退出入口（触屏不再派生“返回”软按键，所以每个 App 都要给一个可见入口）：
;;   —— 点按按钮 = 与硬 A/ESC 完全相同的流程（全局确认框 → 收尾 → 退出）。
;;      落在按钮内的那一次按下/松开沿会被 (EX.check) **吞掉**（不让 App 也响应这一下），
;;      所以 App 的输入代码一行都不用改。布局上只需保证右上角 40x18 不被占用。
;;
;; 后台运行（P4）：
;;   退出确认框是**三按钮**的「确认 / 后台 / 留下」——中间那枚「后台」把实例冻结保留
;;   （解释器与内存池都不释放），回到应用列表后该项带「● 运行中」角标，选中即恢复同一
;;   实例。恢复时宿主会置 (App.resumed)，本模块据此调一次登记的重绘函数。
;;   App 自己也可以直接 (App.hide) 走同一条路（调用前把该落的盘先落完——冻结后不再
;;   有执行机会）。单实例平台（ESP32）没有后台能力，宿主按退出处理。
;;
;;   于是“退出前落盘”在 App 里只值一行：
;;       (EX.install_with draw_all)
;;       (EX.on_exit do_save)                    ; 确认退出 → 先存盘 → 再销毁
;;
;; 两段式收尾（P2，见 am_native_App.h 的协议说明）：
;;   (App.exit) 返回 #t 只代表“用户同意了”，**不代表可以销毁**；宿主会留出
;;   2000ms 的收尾窗口（可跨多拍；期间 (App.is_active) 已为 #f）。本模块在调用
;;   收尾函数之后立刻 (App.exit_now) 宣告收尾完毕，宿主随即在帧末销毁。
;;   收尾函数里可以放心做多拍的事情（含再弹 (Ui.confirm ...) 询问用户）——
;;   模态框存续期间宿主会自动续期，不会把 App 从对话中掐断。
;;
;;   若收尾函数抛出运行时错误（进程停机），宿主按“进程已停”处理，仍会销毁实例，
;;   不会卡在收尾窗口里——但那就没有机会落盘了，所以收尾函数应尽量写简单。
;;
;; 零快照约定（为什么要有“重绘函数”）：
;;   退出确认框是**宿主画的**（ui_exit_confirm_*，与控制台/俄罗斯方块同款），
;;   关闭后画面暂时留着对话框像素，靠 App 重绘覆盖。绝大多数 App 每拍整屏重绘，
;;   (EX.install) 就够；只在对“有状态才重绘”的 App（计算器、二维码等）或长时间
;;   不刷新的 App 才需要 install_with。
;;
;; 语义要点（写错了会踩坑）：
;;   · (App.back_requested) 取走即清零。本模块独占消费它，App 自己别再读——
;;     否则两边抢，谁先读到谁处理，行为不确定。
;;   · (App.exit) 是**阻塞**调用：期间整个虚拟机进程（含本 App 的所有定时器）暂停，
;;     所以确认框存续期间 App 自然定住（与 J2ME 的 pause 同构），不需要额外代码。
;;   · 想用 A 做“应用内返回”（读信页回目录之类）的 App，别装本模块：自己
;;     `(if (App.back_requested) ...)` 处理即可（取走即清零，见上）。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 list/let；块内不 define；
;; System.set_interval 产值，必须用 set! 消费。
;;
;; 注意：模块内的名字**不要带点号**——`EX.xxx` 这种写法是“导入方”的引用形式，
;; 模块内 define 带点号会被当成模块别名成员访问而链接失败。
;; ============================================================================

(native App)
(native System)
(native Gfx)
(native Touch)

;; 排在导入方的名字请勿带点号——`EX.xxx` 是“导入方”的引用形式，模块内 define
;; 带点号会被当成模块别名成员访问而链接失败。

;; 屏幕宽度（画按钮时用；安装时记录，缺省 320）
(define sw 0)

;; “留下”之后要调用的重绘函数；#f 表示不需要（依赖 App 自己的重绘）
(define redraw_fn #f)

;; 用户确认退出后、销毁前的收尾函数；#f 表示不需要
(define cleanup_fn #f)

;; 按钮状态：上一拍的触屏电平 / 本次按下是否落在按钮内 / 本拍的按钮左上角
(define prev_down 0)
(define armed 0)
(define bxv 0)

;; ---- 「退出」按钮的几何 ---------------------------------------------------
;; 默认位置：屏幕**右上角**（btn_x < 0 表示按屏宽自动算）。少数布局把整屏都用满的
;; App（例如计算器：顶部整块是表达式/历史显示区）可用 (EX.draw_button_at x y)
;; 换到别的角落——例如左上角（右对齐的表达式很少顶到左边）。
(define btn_w 36)
(define btn_h 16)
(define btn_x -1)
(define btn_y 2)

;; 按钮左上角（自动位置时按当前屏宽算）
(define bx
  (lambda ()
    (if (== sw 0) (set! sw (Gfx.width)) #f)
    (if (< btn_x 0) (- sw (+ btn_w 4)) btn_x)))

;; 点击是否落在按钮矩形内（帧缓冲像素坐标）
(define hit
  (lambda (px py)
    (set! bxv (bx))
    (if (and (>= px bxv) (< px (+ bxv btn_w)))
        (if (and (>= py btn_y) (< py (+ btn_y btn_h))) 1 0)
        0)))

;; 统一的「退出」按钮（深底 + 细描边 + 12px 字，与各 App 的 HUD 同色系）
(define draw_button
  (lambda ()
    (if (Touch.is_available)
        {
          (set! bxv (bx))
          (Gfx.fill_rect bxv btn_y btn_w btn_h 60 62 74)
          (Gfx.draw_rect bxv btn_y btn_w btn_h 120 124 140)
          (Gfx.draw_text "退出" (+ bxv 6) (+ btn_y 2) 225 228 235)
          ;; 把热区登记给宿主：点按由宿主在**触摸边沿**上判定，慢节奏 App 也不会漏
          (App.set_exit_button bxv btn_y btn_w btn_h)
        }
        (App.set_exit_button 0 0 0 0))))

;; 指定位置画按钮（少数把整屏用满的 App 用；同时更新命中判定用的矩形）
(define draw_button_at
  (lambda (px py)
    {
      (set! btn_x px)
      (set! btn_y py)
      (draw_button)
    }))

;; 退出流程（硬 A/ESC 与按钮共用）：
;;   (App.exit) 确认（#t）→ 收尾函数 → 宣告收尾完毕（宿主帧末销毁）；留下（#f）→ 重绘。
(define do_exit
  (lambda ()
    (if (App.exit)
        {
          (if cleanup_fn (cleanup_fn) #f)
          (App.exit_now)
        }
        (if redraw_fn (redraw_fn) #f))))

;; 每拍调用一次（App 自己的 tick 里，画完按钮、推屏之前）：
;;   ① 硬 A/ESC 的返回请求 → 全局确认框；
;;   ② 右上角按钮：按下落在按钮内 → 武装并**吞掉**该按下沿；在按钮内松手 → 吞掉松开沿 +
;;      走同一条退出流程。用「电平 + 命中」判定而不是无条件 take_press，
;;      是为了不把 App 自己的点按吃掉（落在别处的按下沿原样留给 App）。
(define check
  (lambda ()
    {
      ;; 刚从后台/挂起恢复：画面在挂起期间归别人，补一次重绘（有登记才做）
      (if (App.resumed) (if redraw_fn (redraw_fn) #f) #f)
      (if (App.back_requested) (do_exit) #f)
      (if (Touch.is_pressed)
          {
            (if (== prev_down 0)
                (if (== (hit (Touch.x) (Touch.y)) 1)
                    { (set! armed 1) (Touch.take_press) }
                    (set! armed 0))
                #f)
            (set! prev_down 1)
          }
          {
            (if (and (== prev_down 1) (== armed 1))
                {
                  (Touch.take_release)
                  (set! armed 0)
                  (do_exit)
                }
                #f)
            (set! prev_down 0)
          })
    }))

;; 安装：记录屏幕宽度（画按钮的几何基准）。**不**再安装定时器（轮询点是 (EX.check)）。
(define install
  (lambda ()
    (set! sw (Gfx.width))))

;; 同上，并登记“留下”之后的重绘函数
(define install_with
  (lambda (redraw)
    (set! redraw_fn redraw)
    (install)))

;; 登记退出前的收尾函数（落盘/停外设/最后一帧提示）；在 install 之前或之后皆可
(define on_exit
  (lambda (cleanup)
    (set! cleanup_fn cleanup)))