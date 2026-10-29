;; ============================================================================
;; 触屏重力球（Touch + Gravity）—— Nano「灵机应用程序」示例
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx / Touch / Imu 本地库
;;
;; 本 App 是 Touch / Imu 两个输入本地库的**用法示范兼验收程序**：
;;   · Touch —— 手指按住时小球被"抓"在指尖，松手可甩出（速度即手指速度）；
;;               绿色十字准星画在触点上，实时反映触点坐标。
;;   · Imu   —— 重力向量 (Imu.gravity_x, Imu.gravity_y) 直接驱动小球物理；
;;               右上角罗盘黄点指向 yaw（磁航向），右侧洋红点表示 pitch（仰角）。
;;
;; 说明（坐标与符号，见 am_native_Imu.h）：
;;   gravity_x = -9.8·sin(roll)，gravity_y = +9.8·cos(roll)
;;   是**屏幕像素坐标系**（x 向右、y 向下为正）下的重力分量，可直接累加到
;;   Gfx 坐标里的速度：直立时小球向下落，右缘下沉时小球向右侧滚。
;;
;; 解释器约束（编写本 App 时逐条遵守，详见 manual.md 与 doc/AGENTS.md）：
;;   · 算术运算符固定二参数，多元必须嵌套（多传的参数会污染操作数栈）；
;;   · 没有一元取负 —— 负数常量直接写成 -0.72 这样的字面量；
;;   · 没有 let/case/do/for/list/append/map；define 无过程简写；
;;   · cond 每个子句只接受单个表达式，多分支一律平铺 if；
;;   · / 与 mod 恒返回浮点 —— 列表下标与绘图坐标一律 Math.round/floor 收口；
;;   · Math.PI 是函数，须写 (Math.PI)；
;;   · 顶层 '() 字面量是模块级静态槽，本 App 不使用列表，规避该坑；
;;   · 定时器回调即一帧：解释器由宿主 GUI 按时间片驱动，节奏由 set_interval 决定。
;;
;; 交互提示：左上角为系统区（触屏 4x4 宫格的 esc 格），故本 App 的绘制与操作
;; 区域避开右上角 80x60 的 esc 格范围之外无需特判——随手点按即可操作小球。
;; 退出：A（硬件键盘 Esc / 触屏右上角宫格格子）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Imu)

;; ---- 画布与布局 -----------------------------------------------------------

(define SW (Gfx.width))
(define SH (Gfx.height))
(define HUD_H 30)                       ; 顶部状态栏高度

(define CX (- SW 36))                   ; 罗盘圆心
(define CY 52)
(define CR 18)                          ; 罗盘半径
(define PX (- SW 12))                   ; 仰角指示条（竖直）
(define PBASE 52)
(define PSPAN 18)                       ; ±90° 对应的像素跨度

;; ---- 物理参数（每帧步长，33ms/帧 ≈ 30fps） -------------------------------

(define R 6)                            ; 小球半径（像素）
(define GF 0.025)                       ; 重力缩放：9.8 m/s² × GF ≈ 0.245 px/帧²
(define DAMP 0.97)                      ; 每帧速度阻尼
(define MBOUNCE -0.72)                  ; 碰壁反弹系数（取负 = 反向）
(define GRAB 0.60)                      ; 抓取时的速度跟随系数
(define PULL 0.55)                      ; 抓取时的位置跟随系数

;; ---- 状态 -----------------------------------------------------------------

(define bx 160.0)
(define by 70.0)
(define vx 0.0)
(define vy 0.0)
(define gx 0.0)
(define gy 9.8)
(define pressed 0)
(define tx 0)
(define ty 0)
;; 模块级临时量（解释器对"块内 define / 嵌套词法区域"存在布局敏感的槽位问题，
;; 见 doc/AGENTS.md：热路径局部量一律在模块/函数级 define，块内只用 set!）
(define tmp_a 0.0)
(define tmp_b 0.0)
(define roll 0.0)
(define pitch 0.0)
(define yaw 0.0)
(define _timer 0)

;; ---- 小数 → 字符串（HUD 用，保留 1 位小数） ------------------------------
;; 不用 Math.to_fixed + atom_to_string：VM 的浮点载荷位宽有限（见 am_native_Time.h），
;; 9.8 实际存为 9.79999…，%g 会原样打印出来；这里按"×10 取整 → 拼整数/小数两段"生成。

(define s_i 0)
(define s_w 0)
(define s_f 0)

(define f1
  (lambda (x) {
    (set! s_i (Math.round (* x 10.0)))
    (set! s_w (if (< s_i 0) (Math.ceil (/ s_i 10.0)) (Math.floor (/ s_i 10.0))))
    (set! s_f (Math.abs (mod s_i 10.0)))
    (String.concat (String.atom_to_string s_w)
                   (String.concat "." (String.atom_to_string s_f)))
  }))

(define i2s (lambda (x) (String.atom_to_string x)))

;; ---- 状态栏 --------------------------------------------------------------

(define draw_hud
  (lambda (roll pitch yaw) {
    (Gfx.fill_rect 0 0 SW HUD_H 8 10 18)
    (if (== pressed 1)
        (Gfx.draw_text (String.concat "touch " (String.concat (i2s tx) (String.concat "," (i2s ty))))
                       4 2 120 255 140)
        (Gfx.draw_text "touch ---" 4 2 90 110 120))
    (if (number? roll)
        (Gfx.draw_text (String.concat "roll " (String.concat (f1 roll)
                        (String.concat "  pitch " (f1 pitch))))
                       4 16 130 180 235)
        (Gfx.draw_text "roll ----  (无姿态数据)" 4 16 120 120 130))
    (Gfx.draw_text (String.concat "g " (String.concat (f1 gx) (String.concat "," (f1 gy))))
                   (- SW 78) 2 150 150 170)
    (if (number? yaw)
        (Gfx.draw_text (String.concat "yaw " (f1 yaw)) (- SW 78) 16 200 200 140)
        (Gfx.draw_text "yaw ----" (- SW 78) 16 120 120 130))
  }))

;; ---- 罗盘（yaw）：黄点指向磁航向 -----------------------------------------

(define draw_compass
  (lambda (yaw) {
    (Gfx.draw_circle CX CY CR 55 70 95)
    (Gfx.fill_rect (- CX 1) (- CY 1) 2 2 55 70 95)
    (if (number? yaw) {
      (set! tmp_a (* (/ yaw 180.0) (Math.PI)))
      (Gfx.fill_circle (+ CX (Math.round (* (- CR 2) (Math.sin tmp_a))))
                       (- CY (Math.round (* (- CR 2) (Math.cos tmp_a))))
                       3 255 240 0)
    })
  }))

;; ---- 仰角条（pitch）：洋红点表示屏幕法向仰角 -----------------------------

(define draw_pitch_bar
  (lambda (pitch) {
    (Gfx.fill_rect (- PX 1) (- PBASE PSPAN) 3 (+ 1 (* 2 PSPAN)) 45 50 70)
    (if (number? pitch) {
      (set! tmp_b (- PBASE (Math.round (* (/ pitch 90.0) PSPAN))))
      (if (< tmp_b (- PBASE PSPAN)) (set! tmp_b (- PBASE PSPAN)))
      (if (> tmp_b (+ PBASE PSPAN)) (set! tmp_b (+ PBASE PSPAN)))
      (Gfx.fill_circle PX (Math.round tmp_b) 3 235 60 235)
    })
  }))

;; ---- 小球与准星 ----------------------------------------------------------

(define draw_ball
  (lambda () {
    (Gfx.fill_circle (Math.round bx) (Math.round by) R 255 0 0)
    (Gfx.fill_circle (Math.round bx) (Math.round by) 2 255 190 190)
  }))

(define draw_marker
  (lambda () {
    (if (== pressed 1) {
      (Gfx.draw_line (- tx 9) ty (+ tx 9) ty 0 255 0)
      (Gfx.draw_line tx (- ty 9) tx (+ ty 9) 0 255 0)
      (Gfx.draw_circle tx ty 13 0 160 0)
    })
  }))

;; ---- 一帧：采样输入 → 物理 → 绘制 ----------------------------------------

(define tick
  (lambda () {
    ;; 输入采样（每帧一次；Touch 为电平语义，边沿由本 App 自行判定）
    (set! pressed (if (Touch.is_pressed) 1 0))
    (set! tx (Touch.x))
    (set! ty (Touch.y))
    (set! gx (Imu.gravity_x))
    (set! gy (Imu.gravity_y))

    ;; 物理：拖动期随手（位移即甩出速度），松手后受重力 + 阻尼 + 碰壁
    (if (== pressed 1)
        {
          (set! vx (* GRAB (+ (* 0.4 vx) (- tx bx))))
          (set! vy (* GRAB (+ (* 0.4 vy) (- ty by))))
          (set! bx (+ bx (* PULL (- tx bx))))
          (set! by (+ by (* PULL (- ty by))))
        }
        {
          (set! vx (* DAMP (+ vx (* gx GF))))
          (set! vy (* DAMP (+ vy (* gy GF))))
          (set! bx (+ bx vx))
          (set! by (+ by vy))
          (if (< bx R) { (set! bx R) (set! vx (* MBOUNCE vx)) })
          (if (> bx (- SW R)) { (set! bx (- SW R)) (set! vx (* MBOUNCE vx)) })
          (if (< by R) { (set! by R) (set! vy (* MBOUNCE vy)) })
          (if (> by (- SH R)) { (set! by (- SH R)) (set! vy (* MBOUNCE vy)) })
        })

    ;; 绘制（每帧整屏重绘：清屏 → 背景 → 装饰 → 球与准星 → 状态栏）
    (Gfx.clear)
    (Gfx.fill_rect 0 0 SW SH 10 12 22)
    (Gfx.draw_rect 0 HUD_H (- SW 1) (- (- SH HUD_H) 1) 26 30 46)

    (set! roll (Imu.roll))
    (set! pitch (Imu.pitch))
    (set! yaw (Imu.yaw))

    (draw_compass yaw)
    (draw_pitch_bar pitch)
    (draw_ball)
    (draw_marker)
    (draw_hud roll pitch yaw)
    (Gfx.refresh)
  }))

;; ---- 启动：先画一帧，再以 33ms 周期（≈30fps）推进 ------------------------

(tick)
(set! _timer (System.set_interval 33 tick))