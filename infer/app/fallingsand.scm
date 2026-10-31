;; =============================================================================
;; 沙盒颗粒物理（Falling Sand）—— 灵机应用程序（Nano-WASM-GUI）
;;
;; 106×80 格的颗粒物理沙盒：沙 / 水 / 油 / 火 / 植物 / 酸 / 蒸汽 / 石，八种材质，
;; 触屏撒料与擦除，重力方向实时取自 (Imu.gravity_x, Imu.gravity_y) —— 把机器
;; 倒过来，沙就往上流；侧过来，沙就顺着侧壁流。物理内核见 app/lib/sand_core.scm。
;;
;; 本文件只做界面与输入：
;;   · 把内核的“绘制表”（本步变动过的格子）画出来（稳态零绘制成本）；
;;   · 顶部细条 HUD：当前材质按钮（点按打开材质面板）/ 重力方位箭头 / 活跃格数
;;     与单步耗时读数 / 右上角统一的「退出」按钮（app/lib/exit.scm）；
;;   · 底部材质面板（点按顶部材质按钮弹出）：八种材质 + 擦除 + 清屏。
;;
;; 交互：
;;   · 在画面上按住拖动 = 撒当前材质；选「擦」时是擦除；
;;   · 点顶部左侧的材质方块 = 打开材质面板；点面板里的格子 = 选中并收起；
;;   · 右上角「退出」/ 硬 A = 全局退出确认框（宿主提供）。
;;
;; 解释器约束（manual.md + doc/AGENTS.md，逐条遵守）：
;;   算术恒二参数、没有一元取负、没有 let/list/map、`/` 与 `mod` 恒返回浮点
;;   （下标一律 Math.floor）、`'()` 是静态槽（长表只 push 一次）、块内只用 set!
;;   且临时量按“谁写谁读”分名（w_/h_/u_ 前缀）、运行期不造字符串（≤32 字符会被
;;   驻留且永不释放）—— 数字一律用 10 个字模逐位贴。
;; =============================================================================

(native Gfx)
(native Math)
(native System)
(native Touch)
(native Imu)
(native Diag)
(import S "app/lib/sand_core.scm")
(import EX "app/lib/exit.scm")

;; ---- 画布与网格 -------------------------------------------------------------
(define SW (Gfx.width))          ; 320
(define SH (Gfx.height))         ; 240
(define CELL 3)
(define X0 1)                    ; 网格左边界像素（右侧留 1px 边框）
(define TICK_MS 33)              ; ≈30 步/秒
(define HUD_H 20)                ; 顶部 HUD 带高（触屏落在这个带里不当撒料）

;; ---- 顶部 HUD 控件几何 ------------------------------------------------------
(define MB_X 2)                  ; 材质按钮
(define MB_Y 2)
(define MB_W 32)
(define MB_H 16)
(define GA_X 36)                 ; 重力箭头
(define GA_Y 2)
(define GA_W 20)
(define GA_H 16)
(define IF_X 58)                 ; 读数条
(define IF_Y 2)
(define IF_W 110)
(define IF_H 16)

;; ---- 底部材质面板 -----------------------------------------------------------
(define PN_Y 196)
(define PN_H 44)
(define SLOT_W 32)

;; ---- 材质配色（索引 = 材质编号，见 sand_core.scm） -------------------------
(define CR (quote (10 226 44 92 255 58 150 196 122 26)))
(define CG (quote (11 196 96 70 150 176 240 206 126 28)))
(define CB (quote (18 120 200 46 44 72 60 218 138 34)))
;;                    空 沙 水 油 火 植 酸 蒸 石 墙

;; ---- 运行时状态 -------------------------------------------------------------
(define cur_mat 1)               ; 当前材质（默认沙）
(define palette_open 0)          ; 材质面板是否展开
(define prev_down 0)             ; 上一拍触屏电平（做按下沿判定）
(define last_cx -99)             ; 上一拍撒料的格坐标（做拖影插值）
(define last_cy -99)
(define touch_x 0)
(define touch_y 0)
(define step_ms 0)               ; 上一步物理耗时（界面读数）
(define t_begin 0)
(define imu_ok 0)                ; 姿态是否可用（决定重力箭头配色）

;; ---- 绘制用临时量 -----------------------------------------------------------
(define w_v 0)
(define w_x 0)
(define w_y 0)
(define w_i 0)
(define w_k 0)
(define w_x2 0)
(define w_y2 0)

;; ---- HUD 数字字模（运行期不造字符串，见文件头说明） -----------------------
(define DIGS (quote ("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")))
(define h_v 0)
(define h_d 0)
(define h_p 0)
(define h_i 0)
(define h_x 0)

;; ---- 材质面板的标签与顺序 ---------------------------------------------------
(define MAT_ORDER (quote (1 2 3 4 5 6 7 8 0 9)))   ; 前 8 格材质；第 8 格=擦除(0)；第 9 格=清屏(9)
(define T_PEN (quote ("沙" "水" "油" "火" "植" "酸" "汽" "石" "擦" "清")))

;; =============================================================================
;; 绘制基础
;; =============================================================================

;; 画一个格子（k 为带边框网格下标）
(define draw_cell
  (lambda (k)
    (set! w_v (get_item S.g k))
    (set! w_x (+ X0 (* CELL (- (mod k S.PWW) 1))))
    (set! w_y (* CELL (- (Math.floor (/ k S.PWW)) 1)))
    (if (== w_v 4)
        ;; 火：按格子奇偶取两档橙色，看起来才像在抖
        (if (== (mod k 2) 0)
            (Gfx.fill_rect w_x w_y CELL CELL 255 170 40)
            (Gfx.fill_rect w_x w_y CELL CELL 255 120 30))
        (Gfx.fill_rect w_x w_y CELL CELL (get_item CR w_v) (get_item CG w_v) (get_item CB w_v)))))

;; 整屏重绘（启动、清屏、面板收起、退出框关闭后）
(define draw_all
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH 10 11 18)
    (set! w_y2 0)
    (while (< w_y2 S.H) {
      (set! w_x2 0)
      (while (< w_x2 S.W) {
        (set! w_k (S.gidx w_x2 w_y2))
        (if (> (get_item S.g w_k) 0) (draw_cell w_k) #f)
        (set! w_x2 (+ w_x2 1)) })
      (set! w_y2 (+ w_y2 1)) })
    ;; 世界边框
    (Gfx.fill_rect 0 0 1 SH 44 46 56)
    (Gfx.fill_rect (- SW 1) 0 1 SH 44 46 56)
    (draw_hud)))

;; 增量重绘：只画内核标脏的格子
(define draw_dirty
  (lambda ()
    (set! w_i 0)
    (while (< w_i S.st_plen) {
      (draw_cell (get_item S.lp w_i))
      (set! w_i (+ w_i 1)) })
    (S.paint_done)))

;; 重画底部面板覆盖的那一段（面板收起后擦掉残影）
(define repaint_panel_area
  (lambda ()
    (set! w_y2 (Math.floor (/ PN_Y CELL)))
    (while (< w_y2 S.H) {
      (set! w_x2 0)
      (while (< w_x2 S.W) {
        (set! w_k (S.gidx w_x2 w_y2))
        (if (> (get_item S.g w_k) 0)
            (draw_cell w_k)
            (Gfx.fill_rect (+ X0 (* CELL w_x2)) (* CELL w_y2) CELL CELL 10 11 18))
        (set! w_x2 (+ w_x2 1)) })
      (set! w_y2 (+ w_y2 1)) })))

;; =============================================================================
;; HUD
;; =============================================================================

;; 无前导零的整数（逐位贴字模；位数固定 4，左侧留空格）
(define draw_num
  (lambda (val x y r g b)
    (set! h_v val)
    (if (> h_v 9999) (set! h_v 9999) #f)
    (if (< h_v 0) (set! h_v 0) #f)
    (set! h_p (+ x 28))
    (set! h_i 0)
    (while (< h_i 4) {
      (set! h_d (Math.floor (mod h_v 10)))
      ;; 高位补空格（不留前导零）：h_v 已经数完就画空格
      (if (and (== h_v 0) (> h_i 0))
          (Gfx.draw_text " " (- h_p 7) y r g b)
          (Gfx.draw_text (get_item DIGS h_d) (- h_p 7) y r g b))
      (set! h_v (Math.floor (/ h_v 10)))
      (set! h_p (- h_p 7))
      (set! h_i (+ h_i 1)) })))

;; 重力方位箭头
(define draw_gravity
  (lambda ()
    (Gfx.fill_rect GA_X GA_Y GA_W GA_H 16 18 28)
    (Gfx.draw_rect GA_X GA_Y GA_W GA_H 60 64 82)
    ;; 圆心
    (set! w_x (+ GA_X 10))
    (set! w_y (+ GA_Y 8))
    ;; 方向向量（与物理同向：屏幕 x 右、y 下）
    (set! w_x2 (Math.round (* 7 (/ S.d_gx 9.8))))
    (set! w_y2 (Math.round (* 7 (/ S.d_gy 9.8))))
    (Gfx.draw_line w_x w_y (+ w_x w_x2) (+ w_y w_y2) 120 220 140)
    (Gfx.fill_circle (+ w_x w_x2) (+ w_y w_y2) 1 200 255 200)
    ;; 参考圈
    (Gfx.draw_circle w_x w_y 7 50 60 74)))

(define draw_hud
  (lambda ()
    ;; ① 材质按钮：填当前材质色 + 描边
    (Gfx.fill_rect MB_X MB_Y MB_W MB_H 16 18 28)
    (Gfx.fill_rect (+ MB_X 3) (+ MB_Y 4) 19 8 (get_item CR cur_mat) (get_item CG cur_mat) (get_item CB cur_mat))
    (Gfx.draw_rect MB_X MB_Y MB_W MB_H 90 96 116)
    ;; 右侧小三角 = “点这里可以展开材质面板”
    (Gfx.draw_line 25 7 27 9 150 156 172)
    (Gfx.draw_line 27 9 25 11 150 156 172)
    (Gfx.draw_line 25 7 27 7 150 156 172)
    ;; ② 重力箭头
    (draw_gravity)
    ;; ③ 读数条：活跃格数 + 单步毫秒
    (Gfx.fill_rect IF_X IF_Y IF_W IF_H 16 18 28)
    (Gfx.draw_text "活" IF_X IF_Y 150 156 172)
    (draw_num S.st_pending (+ IF_X 14) IF_Y 150 220 180)
    (Gfx.draw_text "步" (+ IF_X 48) IF_Y 150 156 172)
    (draw_num step_ms (+ IF_X 62) IF_Y 220 200 140)
    (Gfx.draw_text "ms" (+ IF_X 94) IF_Y 130 136 150)))

;; 材质面板
(define draw_palette
  (lambda ()
    (Gfx.fill_rect 0 PN_Y SW PN_H 22 24 34)
    (Gfx.fill_rect 0 PN_Y SW 1 60 64 84)
    (set! w_i 0)
    (while (< w_i 10) {
      (set! w_x (* SLOT_W w_i))
      (set! w_v (get_item MAT_ORDER w_i))
      ;; 槽底
      (Gfx.fill_rect w_x (+ PN_Y 2) SLOT_W (- PN_H 4) 30 32 44)
      ;; 色块（擦除/清屏格特殊画）
      (if (== w_i 8)
          {
            (Gfx.fill_rect (+ w_x 6) (+ PN_Y 6) 20 16 12 13 20)
            (Gfx.draw_line (+ w_x 6) (+ PN_Y 6) (+ w_x 25) (+ PN_Y 21) 180 90 90)
            (Gfx.draw_line (+ w_x 6) (+ PN_Y 21) (+ w_x 25) (+ PN_Y 6) 180 90 90)
          }
          (if (== w_i 9)
              (Gfx.fill_rect (+ w_x 6) (+ PN_Y 6) 20 16 90 40 60)
              (Gfx.fill_rect (+ w_x 6) (+ PN_Y 6) 20 16
                             (get_item CR w_v) (get_item CG w_v) (get_item CB w_v))))
      ;; 选中高亮
      (if (and (== w_i 8) (== cur_mat 0))
          (Gfx.draw_rect (+ w_x 1) (+ PN_Y 3) (- SLOT_W 2) (- PN_H 6) 240 240 240)
          #f)
      (if (and (< w_i 8) (== cur_mat w_v))
          (Gfx.draw_rect (+ w_x 1) (+ PN_Y 3) (- SLOT_W 2) (- PN_H 6) 240 240 240)
          #f)
      ;; 标签
      (Gfx.draw_text (get_item T_PEN w_i) (+ w_x 10) (+ PN_Y 26) 210 214 226)
      (set! w_i (+ w_i 1)) })))

;; =============================================================================
;; 输入
;; =============================================================================

;; 触屏像素 → 网格坐标（返回 0 表示落在画布外）
(define pick_cell
  (lambda (px py)
    (set! w_x (Math.floor (/ (- px X0) CELL)))
    (set! w_y (Math.floor (/ py CELL)))
    (if (< w_x 0) (set! w_x 0) #f)
    (if (> w_x (- S.W 1)) (set! w_x (- S.W 1)) #f)
    (if (< w_y 0) (set! w_y 0) #f)
    (if (> w_y (- S.H 1)) (set! w_y (- S.H 1)) #f)
    (if (and (>= px X0) (< px (+ X0 (* CELL S.W))))
        (if (and (>= py 0) (< py (* CELL S.H))) 1 0)
        0)))

;; 点按材质面板：返回 1 表示已处理（不撒料）
(define palette_tap
  (lambda (px py)
    (if (and (>= py PN_Y) (< py SH))
        {
          (set! w_i (Math.floor (/ px SLOT_W)))
          (if (< w_i 0) (set! w_i 0) #f)
          (if (> w_i 9) (set! w_i 9) #f)
          (if (== w_i 9)
              {
                ;; 清屏
                (S.clear_all)
                (draw_all)
              }
              (set! cur_mat (get_item MAT_ORDER w_i)))
          (set! palette_open 0)
          (repaint_panel_area)
          (draw_hud)
          1
        }
        {
          ;; 面板外点按：收起面板（这一次不撒料）
          (set! palette_open 0)
          (repaint_panel_area)
          (draw_hud)
          1
        })))

;; 撒料（含拖动插值：上一拍与这一拍之间最多补一格，避免快速拖动画出虚线）
(define do_pour
  (lambda (px py)
    (if (== (pick_cell px py) 1)
        {
          (if (and (>= last_cx 0) (> (Math.abs (- w_x last_cx)) 1))
              (S.pour (Math.floor (/ (+ w_x last_cx) 2)) w_y cur_mat)
              #f)
          (if (and (>= last_cy 0) (> (Math.abs (- w_y last_cy)) 1))
              (S.pour w_x (Math.floor (/ (+ w_y last_cy) 2)) cur_mat)
              #f)
          (S.pour w_x w_y cur_mat)
          (set! last_cx w_x)
          (set! last_cy w_y)
        }
        #f)))

(define read_touch
  (lambda ()
    (set! touch_x (Touch.x))
    (set! touch_y (Touch.y))
    (if (Touch.is_pressed)
        {
          (if (== prev_down 0)
              {
                ;; ---- 按下沿 ----
                (set! last_cx -99)
                (set! last_cy -99)
                (if (== palette_open 1)
                    (palette_tap touch_x touch_y)
                    (if (and (and (>= touch_x MB_X) (< touch_x (+ MB_X MB_W)))
                             (and (>= touch_y MB_Y) (< touch_y (+ MB_Y MB_H))))
                        {
                          (set! palette_open (- 1 palette_open))
                          (if (== palette_open 1)
                              (draw_palette)
                              {
                                (repaint_panel_area)
                                (draw_hud)
                              })
                        }
                        #f))
              }
              #f)
          ;; ---- 按住拖动：撒料（HUD 带与面板不撒）----
          (if (== palette_open 0)
              (if (>= touch_y HUD_H)
                  (do_pour touch_x touch_y)
                  #f)
              #f)
          (set! prev_down 1)
        }
        (set! prev_down 0))))

;; =============================================================================
;; 每拍
;; =============================================================================

(define tick
  (lambda ()
    (EX.check)                                  ; 返回请求 / 右上角「退出」按钮

    ;; 姿态 → 重力方向（8 方位量化）
    (S.set_gravity (Imu.gravity_x) (Imu.gravity_y))

    ;; 物理一步（带耗时读数）
    (set! t_begin (Diag.ms))
    (S.step)
    (set! step_ms (- (Diag.ms) t_begin))

    ;; 输入（撒料）→ 增量重绘 → HUD
    (read_touch)
    (draw_dirty)
    (draw_hud)
    (if (== palette_open 1) (draw_palette) #f)
    (Gfx.refresh)))

;; =============================================================================
;; 启动
;; =============================================================================

(define _timer 0)
(EX.install_with draw_all)
(set! imu_ok (if (Imu.is_available) 1 0))
(S.scene_init)
(draw_all)
(set! _timer (System.set_interval TICK_MS tick))