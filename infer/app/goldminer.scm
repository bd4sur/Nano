;; ============================================================================
;; 黄金矿工（Gold Miner）—— Nano「灵机应用程序」Scheme App
;;
;; 参照实现：infer/ui_goldminer.c（C 版，主菜单「小游戏」里的黄金矿工）。本 App 由
;; 两部分组成：
;;   · app/lib/goldminer_core.scm —— 纯游戏逻辑（摆动/出钩/回收/抓取/关卡生成/计分），
;;     逐条对照 C 版 gm_* 函数，可脱离界面单独回归；
;;   · 本文件 —— 布局、绘制与触屏/键盘交互（对照 C 版渲染段与 ui_goldminer_event_handler）。
;;
;; 贴图（**路径与 C 版完全一致**，C 版为 PLATFORM_ROOT_DIR "/icon/gm_*.png"；
;; Scheme 侧路径相对业务资源根，而资源根就是 PLATFORM_ROOT_DIR，指同一批文件）：
;;   icon/gm_miner.png  icon/gm_hook.png  icon/gm_gold_1..3.png
;;   icon/gm_rock.png   icon/gm_diamond.png  icon/gm_bonus_1..2.png
;; 它们都是**透明背景** PNG，因此必须用 alpha 混合的精灵接口绘制：
;;   (Gfx.sprite_info 路径)          → (宽 高) | #null（解码并常驻缓存，也用于预热）
;;   (Gfx.draw_sprite 路径 cx cy)    → 以 (cx,cy) 为中心按原始尺寸混合绘制
;;   (Gfx.draw_sprite_rotated 路径 px py 角度)
;;                                   → 以 (px,py)（贴图上缘中点）为枢轴旋转后绘制
;;   钩绳用 (Gfx.draw_line_aa ...) 画（抗锯齿，同 C 版 gfx_draw_line_anti_aliasing）
;;   注：Markdown 阅读器用的 (Gfx.draw_image ...) 是"缩放到目标矩形 + RGB 直写"的
;;   插图通道（丢弃 alpha），画透明精灵会得到不透明方块，故本 App 不走那条路。
;;
;; 与 C 版的三处**平台差异**（源于"Scheme App 运行在宿主的应用列表里"）：
;;   1) 退出：C 版右上角自绘「返回」按钮回小游戏菜单；Scheme App 没有那个状态，
;;      退出即回应用列表。故统一走 app/lib/exit.scm —— 右上角「退出」按钮 + 硬
;;      A/ESC，都是"弹全局确认框 → 确认后销毁实例"（与俄罗斯方块/2048 同款），
;;      生命周期问题（平滑关闭、切后台）由宿主负责。
;;   2) 键盘：C 版 D(回车)/数字 2 发射；Scheme 侧经 (native Key) 的按下沿同样支持
;;      回车与 '2'，另有触屏（C 版也有）。
;;   3) 帧节奏：C 版每帧由宿主驱动；这里用 System.set_interval 按 ~30fps 节流，
;;      物理步长用 System.timestamp 的实测间隔并钳到 50ms（与 C 版 GM_DT_MAX 一致）。
;;
;; 解释器约束（manual.md）：算术恒二参数、无一元取负；无 list/let/map；
;; cond 每子句单个表达式；/ 与 mod 恒浮点（取整用 Math.floor）；while 体非独立
;; 词法区域 → 所有局部量都用模块级 define + set!。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native App)

(import GM "app/lib/goldminer_core.scm")
(import EX "app/lib/exit.scm")   ; 右上角「退出」按钮 + 硬 A/ESC → 全局退出确认

;; ---- 节奏 ----------------------------------------------------------------

(define TICK_MS 33)     ;; 定时器周期（ms，约 30fps：出钩 300px/s → 每拍 10px）
(define now_ms 0)
(define last_ms 0)
(define dt 0)

;; ---- 布局（与 C 版常量同源；屏幕尺寸在 boot 时按 Gfx.width/height 落实）----

(define SW 320)
(define SH 240)
;; 场景常量直接取自核心库（单一事实来源；import 的顶层定义在本模块之前求值）
(define GROUND_Y GM.GROUND_Y)   ;; 地表线
(define PIVOT_X GM.PIVOT_X)     ;; 钩绳摆动支点
(define PIVOT_Y GM.PIVOT_Y)
(define MINER_X (+ PIVOT_X 20)) ;; 矿工贴图中心（C 版：PIVOT_X+20, PIVOT_Y-20）
(define MINER_Y (- PIVOT_Y 20))

;; ---- 输入状态 ------------------------------------------------------------

(define tr 0)           ;; 本拍有松开沿
(define key 0)          ;; 本拍取到的按键
(define i 0)            ;; 绘制/拷贝循环量
(define k 0)
(define last_phase -1)  ;; 诊断用：上一拍的阶段/分值/关卡（变化时打一行到宿主 stdout）
(define last_score -1)
(define last_level 0)
(define sp "")          ;; 当前精灵路径
(define it_active 0)
(define it_type 0)

;;  绘制色（与 C 版一致）
(define SKY_R 25) (define SKY_G 45) (define SKY_B 75)
(define GND_R 110) (define GND_G 75) (define GND_B 40)
(define LINE_R 200) (define LINE_G 160) (define LINE_B 110)

;; ============================================================================
;; 贴图尺寸预热：进入游戏时按真实 PNG 尺寸填充核心库的尺寸表
;;   （C 版用 ui_icon_get_size 触发解码并缓存；这里用 Gfx.sprite_info —— 同一语义）。
;;   贴图缺失（#null）时保留核心库里的原型包围盒，游戏仍可玩（不会画出方块）。
;; ============================================================================

(define load_sprite_sizes
  (lambda () {
    (set! i 0)
    (while (< i GM.SPR_NUM) {
      (set! sp (get_item GM.SPR_PATH i))
      (set! k (Gfx.sprite_info sp))          ;; (宽 高) | #null
      (if (null? k)
          #f
          (GM.set_sprite_size i (get_item k 0) (get_item k 1)))
      (set! i (+ i 1))
    })
  }))

;; ============================================================================
;; 绘制
;; ============================================================================

;; 顶栏信息（C 版：「得分 %d  第%d关  剩余 %d」，12px 抗锯齿白字，左上角 6,2）
(define draw_hud
  (lambda ()
    (Gfx.draw_text
      (String.concat "得分 " (String.concat (String.atom_to_string GM.score)
        (String.concat "  第" (String.concat (String.atom_to_string GM.level)
          (String.concat "关  剩余 " (String.atom_to_string GM.items_left))))))
      6 2 255 255 255)
    #t))

;; 物品（贴着精灵中心绘制；C 版用 ui_icon_draw_centered，这里用 Gfx.draw_sprite）
(define draw_items
  (lambda () {
    (set! i 0)
    (while (< i GM.MAX_ITEMS) {
      (set! it_active (GM.item_ref i 0))
      (if (== it_active 1) {
        (set! it_type (GM.item_ref i 1))
        (set! sp (get_item GM.SPR_PATH (+ GM.SPR_GOLD_1 it_type)))
        (Gfx.draw_sprite sp (GM.item_ref i 2) (GM.item_ref i 3))
      })
      (set! i (+ i 1))
    })
  }))

;; 钩绳 + 钩爪：绳用 **1px 抗锯齿直线**（(Gfx.draw_line_aa ...) = C 版
;; gfx_draw_line_anti_aliasing 的吴小林算法，端点保持浮点、不取整——摆动的绳角是连续的，
;; 只有亚像素端点才得到平滑的斜线），颜色/线宽与 C 版一致（230,230,230 / 1px）。
;; 钩爪贴图以绳端锚点（上缘中点）为枢轴随绳角旋转（(Gfx.draw_sprite_rotated)）。
(define draw_hook
  (lambda ()
    (Gfx.draw_line_aa PIVOT_X PIVOT_Y GM.hook_x GM.hook_y 1 230 230 230)
    (Gfx.draw_sprite_rotated (get_item GM.SPR_PATH GM.SPR_HOOK)
                             (Math.floor GM.hook_x) (Math.floor GM.hook_y)
                             GM.hook_angle)
    #t))

(define render
  (lambda () {
    (Gfx.clear)
    ;; 背景：天空 + 矿区台面 + 地下
    (Gfx.fill_rect 0 0 SW GROUND_Y SKY_R SKY_G SKY_B)
    (Gfx.fill_rect 0 GROUND_Y SW (- SH GROUND_Y) GND_R GND_G GND_B)
    (Gfx.draw_line 0 GROUND_Y SW GROUND_Y LINE_R LINE_G LINE_B)

    (draw_hud)
    (draw_items)
    (Gfx.draw_sprite (get_item GM.SPR_PATH GM.SPR_MINER) MINER_X MINER_Y)
    (draw_hook)

    (EX.draw_button)     ;; 右上角「退出」（并把热区登记给宿主，慢节奏也不漏点按）
    (Gfx.refresh)
  }))

;; ============================================================================
;; 输入
;; ============================================================================

;; 键盘按下沿：回车(13) / '2'(50) 发射（C 版 D 键、数字 2）；ESC(27) 由 EX 接管
(define handle_keys
  (lambda () {
    (set! key (Key.take_press))
    (while (not (== key 0)) {
      (if (== key 13) (GM.fire) #f)
      (if (== key 50) (GM.fire) #f)
      (set! key (Key.take_press))
    })
  }))

;; 触屏：松手沿发射（与 C 版 touch_edge UP 一致；落在「退出」按钮上的序列已被
;; (EX.check) 吞掉，不会误发射）
(define handle_touch
  (lambda () {
    (set! tr (if (Touch.take_release) 1 0))
    (if (== tr 1) (GM.fire) #f)
  }))

;; ---- 诊断（宿主 stdout / 浏览器控制台；端到端测试据此断言游戏真的在推进）----
;;   只在**状态变化**时打印：发射/抓取/结算/过关，正常游玩不会刷屏。
(define trace_step
  (lambda ()
    (if (not (== (Math.floor GM.score) last_score))
        { (set! last_score (Math.floor GM.score))
          (display "GMSCORE score=") (display last_score)
          (display " items=") (display GM.items_left)
          (display " level=") (display GM.level) (newline) }
        #f)
    (if (not (== GM.phase last_phase))
        { (set! last_phase GM.phase)
          (if (== GM.phase GM.PHASE_EXTEND)
              { (display "GMFIRE") (newline) }
              #f)
          (if (and (== GM.phase GM.PHASE_RETRACT) (>= GM.grabbed 0))
              { (display "GMGRAB idx=") (display GM.grabbed)
                (display " type=") (display (GM.item_ref GM.grabbed 1)) (newline) }
              #f) }
        #f)
    (if (not (== GM.level last_level))
        { (set! last_level GM.level)
          (display "GMLEVEL level=") (display GM.level)
          (display " items=") (display GM.items_left) (newline) }
        #f)
  ))

;; ============================================================================
;; 主循环：一拍 = 采样输入 → 逻辑推进 → 整帧重绘
;; ============================================================================

(define tick
  (lambda () {
    (EX.check)
    ;; 实测步长（核心库内部钳到 GM.DT_MAX）
    (set! now_ms (System.timestamp))
    (set! dt (- now_ms last_ms))
    (set! last_ms now_ms)
    (handle_keys)
    (handle_touch)
    (GM.step dt)
    (trace_step)
    (render)
  }))

;; ============================================================================
;; 启动
;; ============================================================================

(define boot
  (lambda () {
    (set! SW (Gfx.width))
    (set! SH (Gfx.height))
    (GM.set_screen SW SH)
    (load_sprite_sizes)    ;; 一次性读贴图尺寸（并触发宿主侧解码缓存）
    (GM.reset)
    (set! last_score (Math.floor GM.score))
    (set! last_level GM.level)
    (set! last_phase GM.phase)
    (set! last_ms (System.timestamp))
    (EX.install_with render)
    (tick)
    (System.set_interval TICK_MS tick)
    (display "GMSTAT sprites=") (display GM.SPR_NUM)
    (display " items=") (display GM.items_left)
    (display " score=") (display GM.score)
    (display " level=") (display GM.level) (newline)
  }))

(boot)
