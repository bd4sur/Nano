;; ============================================================================
;; 俄罗斯方块（Tetris）—— Nano「灵机应用程序」Scheme App
;;
;; 参照实现：infer/ui_tetris.c（C 版，STATE_TETRIS）。本 App 由两部分组成：
;;   · app/lib/tetris_core.scm —— 纯游戏逻辑（场地/方块/碰撞/旋转/消行/计分/重力），
;;     逐条对照 C 版 trs_* 函数，可脱离界面单独回归（tools/e2e/native/tetris_logic.scm）；
;;   · 本文件 —— 布局、绘制与触屏交互（对照 C 版 trs_layout / trs_draw_* / 事件处理）。
;;
;; 规则与 C 版完全一致：10x20 场地、7 种方块、简单踢墙、消行 100/300/500/800 × 关卡、
;; 每 10 行升一关、下落间隔 max(800-70*(level-1),100) ms、无法入场即结束；左面板
;; （标题/得分/行数/关卡/下一个）与虚拟按键同款。
;;
;; 与 C 版的三处**平台差异**（源于"Scheme App 没有键盘通道，且运行在宿主 GUI 里"）：
;;   1) 输入只有触屏（本地库 Touch）：C 版的 ←/→/1/2/D 硬按键动作全部由虚拟按键提供。
;;   2) 「退出」按钮从右上角移到**场地右侧空白列**：触屏右上角（x≥SW*3/4 且 y<SH/4）
;;      是宿主 4x4 宫格映射的 esc 格，点在那一带会被宿主当作"返回列表"直接退出 App
;;      （C 版把 STATE_TETRIS 列入 ui_app_state_is_menu 抑制表才敢用那个角）；
;;      「速降」留在顶部左侧：场地横向居中（320 宽时从 x=110 起），顶部四分之一
;;      （0..SW/4）落在场地左侧，不会遮住入场中的方块。
;;   3) 退出确认框的「确认」无法真正切回小游戏菜单（Scheme App 不能改宿主 STATE），
;;      故确认后**冻结游戏**并提示"按 A 返回"（A / 触屏右上角即宿主返回）。
;;
;; 虚拟按键交互（同 C 版）：左移/右移/速降 按下沿武装、松手沿结算，按住满 500ms 后
;; 每拍反复触发；旋转/退出 只在松手沿单次触发。确认框打开时游戏暂停。
;;
;; 解释器约束（manual.md）：算术固定二参数、无一元取负、无 list/let/map、
;; cond 每子句单个表达式、/ 与 mod 恒浮点、Math.PI 是函数、while 体非独立词法区域
;; → 所有局部量都用模块级 define + set!。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)

(import T "app/lib/tetris_core.scm")

;; ---- 节奏参数 ------------------------------------------------------------
;; 定时器周期只是"希望多久跑一拍"；真正的物理步长用 System.timestamp 的实测
;; 间隔（同 C 版按帧时间戳累计，并钳到 TRS_DT_MAX = 50ms 防卡顿跳变）——
;; 宿主对过期的定时器按"当前时刻 + 周期"重新排期，实测周期会略大于设定值，
;; 若按固定步长累计，游戏速度就会随宿主的定时器节奏漂移。

(define TICK_MS 33)                     ; 定时器周期（ms）
(define DT_MAX 50)                      ; 单拍最大物理步长（ms，同 C 版）
(define HOLD_MS 500)                    ; 长按反复阈值（ms，同 C 版）

;; ---- 布局（算法同 C 版 trs_layout：底部按键栏 H/8，场地贴屏幕上沿、横向居中，
;;      格子方形，两侧留 ≥96px 侧栏）----------------------------------------

(define SW (Gfx.width))
(define SH (Gfx.height))
(define TOP_H (Math.floor (/ SH 10)))
(define BOT_H (Math.floor (/ SH 8)))
(define PANEL_SPAN (- SH BOT_H))
(define CELL_H (Math.floor (/ PANEL_SPAN T.ROWS)))
(define CELL_W (Math.floor (/ (- SW 192) T.COLS)))
(define CELL_W0 (if (< CELL_H CELL_W) CELL_H CELL_W))
(define CELL (if (< CELL_W0 1) 1 CELL_W0))
(define FIELD_X (Math.floor (/ (- SW (* T.COLS CELL)) 2)))
(define FIELD_Y 0)
(define FIELD_W (* T.COLS CELL))
(define FIELD_H (* T.ROWS CELL))
(define PV_CELL0 (Math.floor (/ (* CELL 2) 3)))
(define PV_CELL (if (< PV_CELL0 5) 5 PV_CELL0))

;; 侧栏各行 y（C 版 S_LINE_FR = {0.02,0.16,0.30,0.44,0.60}）
(define PANEL_Y0 (Math.floor (* PANEL_SPAN 0.02)))
(define PANEL_Y1 (Math.floor (* PANEL_SPAN 0.16)))
(define PANEL_Y2 (Math.floor (* PANEL_SPAN 0.30)))
(define PANEL_Y3 (Math.floor (* PANEL_SPAN 0.44)))
(define PANEL_Y4 (Math.floor (* PANEL_SPAN 0.60)))

;; 退出确认框（按钮整体保持在 x < SW*3/4 之内，避开宿主 esc 宫格）
(define MDLG_W 180)
(define MDLG_H 88)
(define MDLG_X (Math.floor (/ (- SW MDLG_W) 2)))
(define MDLG_Y (Math.floor (/ (- SH MDLG_H) 2)))
(define MBTN_W 70)
(define MBTN_H 30)
(define MBTN_GAP 10)
(define MBTN_Y (+ MDLG_Y 46))
(define MOK_X (+ MDLG_X 15))
(define MSTAY_X (+ MOK_X (+ MBTN_W MBTN_GAP)))

;; ---- 虚拟按键几何（命中判定与绘制共用）-----------------------------------

(define VK_W (Math.floor (/ SW 4)))     ; 顶部「速降」宽（在场地上方左侧）
(define VB_W (Math.floor (/ SW 3)))     ; 底部三等分
(define EX_X (+ FIELD_X (+ FIELD_W 4))) ; 「退出」：场地右侧空白列
(define EX_Y (+ TOP_H 40))
(define EX_W (- SW (+ EX_X 2)))
(define EX_H 34)

;; ---- 界面状态 ------------------------------------------------------------

(define modal 0)        ; 1 = 退出确认框打开（游戏暂停）
(define stopped 0)      ; 1 = 已确认退出（冻结，等宿主 A 返回）
(define stopped_drawn 0)
(define hold_key 0)     ; 按住跟踪中的虚拟键号（0 无 / 1 左移 / 3 右移 / 4 速降）
(define hold_ms 0)      ; 按下时刻（ms）
(define now_ms 0)       ; 本拍时间戳
(define last_ms 0)      ; 上一拍时间戳
(define dt 0)           ; 本拍物理步长（ms）
(define repeating 0)
(define press_x 0)      ; 本触摸序列按下点（松手沿据此判定命中的键）
(define press_y 0)
(define tp 0)           ; 本拍有按下沿
(define tr 0)           ; 本拍有松开沿
(define _timer 0)

;; 模块级临时量（避免块内 define 与嵌套作用域的槽位问题）
(define dc_x 0) (define dc_y 0) (define dc_c 0)
(define dp_i 0) (define dp_x 0) (define dp_y 0)
(define dk_i 0) (define dk_x 0) (define dk_y 0) (define dk_w 0) (define dk_h 0)
(define vh 0) (define mn_hit 0)

;; ============================================================================
;; 虚拟按键：命中判定与动作
;;   顶部左侧四分之一：速降(4)
;;   场地右侧空白列（y 在顶部栏以下，避开宿主 esc 格）：退出(5)
;;   底部三等分：左移(1) / 旋转(2) / 右移(3)
;; ============================================================================

(define vkey_hit
  (lambda (x y) {
    (if (and (>= y EX_Y) (and (< y (+ EX_Y EX_H)) (>= x EX_X)))
        5
        (if (< y TOP_H)
            (if (< x VK_W) 4 0)
            (if (>= y (- SH BOT_H))
                (if (< x VB_W)
                    1
                    (if (< x (* VB_W 2)) 2 3))
                0)))
  }))

;; 左移/右移/速降（短按单次与长按反复共用）
(define vkey_action
  (lambda (hit) {
    (if (== hit 1) (T.move -1))
    (if (== hit 3) (T.move 1))
    (if (== hit 4) {
      (T.step_down)
      (T.reset_fall)
    })
  }))

;; ============================================================================
;; 绘制
;; ============================================================================

;; 12px alpha 字模等宽（advance = 12），居中只需按字符数偏移
(define draw_text_c
  (lambda (s cx cy r g b)
    (Gfx.draw_text s (- cx (* 6 (String.length s))) (- cy 7) r g b)))

(define draw_button
  (lambda (x y w h label r g b) {
    (Gfx.fill_rect x y w h r g b)
    (Gfx.draw_rect x y w h 12 12 16)
    (draw_text_c label (+ x (Math.floor (/ w 2))) (+ y (Math.floor (/ h 2))) 240 240 240)
  }))

(define draw_vkeys
  (lambda () {
    (draw_button 0 0 VK_W TOP_H "速降" 46 46 50)
    (draw_button EX_X EX_Y EX_W EX_H "退出" 90 40 40)
    (draw_button 0 (- SH BOT_H) VB_W BOT_H "左移" 46 46 50)
    (draw_button VB_W (- SH BOT_H) VB_W BOT_H "旋转" 46 46 50)
    (draw_button (* VB_W 2) (- SH BOT_H) (- SW (* VB_W 2)) BOT_H "右移" 46 46 50)
  }))

;; 一个场地格子：填色 + 底部 2px 暗条（同 C 版 trs_draw_cell）
(define draw_cell
  (lambda (fx fy ci) {
    (set! dc_x (+ FIELD_X (* fx CELL)))
    (set! dc_y (+ FIELD_Y (* fy CELL)))
    (set! dc_c (* (- ci 1) 3))
    (Gfx.fill_rect dc_x dc_y (- CELL 1) (- CELL 1)
                   (get_item T.COLORS dc_c)
                   (get_item T.COLORS (+ dc_c 1))
                   (get_item T.COLORS (+ dc_c 2)))
    (Gfx.fill_rect dc_x (+ dc_y (- CELL 3)) (- CELL 1) 2
                   (Math.floor (/ (get_item T.COLORS dc_c) 2))
                   (Math.floor (/ (get_item T.COLORS (+ dc_c 1)) 2))
                   (Math.floor (/ (get_item T.COLORS (+ dc_c 2)) 2)))
  }))

(define draw_field_box
  (lambda () {
    (Gfx.fill_rect (- FIELD_X 2) FIELD_Y (+ FIELD_W 4) (+ FIELD_H 2) 40 40 48)
    (Gfx.fill_rect (- FIELD_X 1) FIELD_Y (+ FIELD_W 2) (+ FIELD_H 1) 18 18 24)
  }))

(define draw_panel
  (lambda () {
    (Gfx.draw_text "俄罗斯方块" 8 PANEL_Y0 255 255 255)
    (Gfx.draw_text (String.concat "得分 " (String.atom_to_string T.score)) 8 PANEL_Y1 220 220 220)
    (Gfx.draw_text (String.concat "行数 " (String.atom_to_string T.lines)) 8 PANEL_Y2 220 220 220)
    (Gfx.draw_text (String.concat "关卡 " (String.atom_to_string T.level)) 8 PANEL_Y3 220 220 220)
    (Gfx.draw_text "下一个:" 8 PANEL_Y4 180 180 180)
    (T.fill_cells T.next_piece 0)
    (set! dk_i 0)
    (while (< dk_i 4) {
      (set! dk_x (+ 12 (* (get_item T.cells_x dk_i) PV_CELL)))
      (set! dk_y (+ (+ PANEL_Y4 20) (* (get_item T.cells_y dk_i) PV_CELL)))
      (set! dk_w (- PV_CELL 1))
      (set! dk_h (- PV_CELL 1))
      (Gfx.fill_rect dk_x dk_y dk_w dk_h
                     (get_item T.COLORS (* T.next_piece 3))
                     (get_item T.COLORS (+ (* T.next_piece 3) 1))
                     (get_item T.COLORS (+ (* T.next_piece 3) 2)))
      (set! dk_i (+ dk_i 1))
    })
  }))

(define draw_board
  (lambda () {
    ;; 已锁定方块
    (set! dp_y 0)
    (while (< dp_y T.ROWS) {
      (set! dp_x 0)
      (while (< dp_x T.COLS) {
        (set! dp_i (T.field_at dp_x dp_y))
        (if (not (== dp_i 0)) (draw_cell dp_x dp_y dp_i))
        (set! dp_x (+ dp_x 1))
      })
      (set! dp_y (+ dp_y 1))
    })
    ;; 当前方块（y<0 的格子在场地之上，不画）
    (if (== T.game_over 0) {
      (T.fill_cells T.piece T.rot)
      (set! dp_i 0)
      (while (< dp_i 4) {
        (set! dp_x (+ T.px (get_item T.cells_x dp_i)))
        (set! dp_y (+ T.py (get_item T.cells_y dp_i)))
        (if (>= dp_y 0) (draw_cell dp_x dp_y (+ T.piece 1)))
        (set! dp_i (+ dp_i 1))
      })
    })
  }))

(define draw_game_over
  (lambda () {
    (set! dk_h (* CELL 6))
    (set! dk_y (+ FIELD_Y (Math.floor (/ (* (- FIELD_H dk_h) 2) 5))))
    (Gfx.fill_rect (+ FIELD_X 2) dk_y (- FIELD_W 4) dk_h 0 0 0)
    (draw_text_c "游戏结束" (+ FIELD_X (Math.floor (/ FIELD_W 2))) (+ dk_y (Math.floor (/ dk_h 2))) 255 80 80)
    (draw_text_c (String.concat "得分 " (String.atom_to_string T.score))
                 (+ FIELD_X (Math.floor (/ FIELD_W 2)))
                 (+ (+ dk_y (Math.floor (/ dk_h 2))) 22) 255 255 255)
  }))

;; 退出确认框（形态同 C 版 ui_exit_confirm_draw）
(define draw_modal
  (lambda () {
    (Gfx.fill_rect (- MDLG_X 2) (- MDLG_Y 2) (+ MDLG_W 4) (+ MDLG_H 4) 70 70 78)
    (Gfx.fill_rect MDLG_X MDLG_Y MDLG_W MDLG_H 24 24 28)
    (draw_text_c "是否退出？" (+ MDLG_X (Math.floor (/ MDLG_W 2))) (+ MDLG_Y 30) 240 240 240)
    (draw_button MOK_X MBTN_Y MBTN_W MBTN_H "确认" 16 72 176)
    (draw_button MSTAY_X MBTN_Y MBTN_W MBTN_H "留下" 46 46 50)
  }))

(define draw_stopped
  (lambda () {
    (Gfx.fill_rect (+ FIELD_X 2) (+ FIELD_Y 60) (- FIELD_W 4) 80 0 0 0)
    (draw_text_c "已退出" (+ FIELD_X (Math.floor (/ FIELD_W 2))) (+ FIELD_Y 84) 200 200 200)
    (draw_text_c "按 A 返回" (+ FIELD_X (Math.floor (/ FIELD_W 2))) (+ FIELD_Y 108) 160 160 170)
  }))

(define render
  (lambda () {
    (Gfx.clear)
    (draw_field_box)
    (draw_board)
    (draw_panel)
    (draw_vkeys)
    (if (== T.game_over 1) (draw_game_over))
    (if (== modal 1) (draw_modal))
    (if (== stopped 1) (draw_stopped))
    (Gfx.refresh)
  }))

;; ============================================================================
;; 输入
;; ============================================================================

;; 确认框打开期间：只处理框内两个按钮，游戏暂停
(define modal_input
  (lambda () {
    (if (== tr 1) {
      (set! mn_hit 0)
      (if (and (>= press_y MBTN_Y) (< press_y (+ MBTN_Y MBTN_H)))
          (if (and (>= press_x MOK_X) (< press_x (+ MOK_X MBTN_W)))
              (set! mn_hit 1)
              (if (and (>= press_x MSTAY_X) (< press_x (+ MSTAY_X MBTN_W)))
                  (set! mn_hit 2))))
      (if (== mn_hit 1) {
        (set! modal 0)
        (set! stopped 1)
      })
      (if (== mn_hit 2) (set! modal 0))
    })
  }))

(define game_input
  (lambda () {
    ;; 按下沿：命中 左移/右移/速降 → 武装按住跟踪（松手才结算，同 C 版）
    (if (== tp 1)
        (if (== tr 0) {
          (set! vh (vkey_hit (Touch.x) (Touch.y)))
          (if (or (== vh 1) (or (== vh 3) (== vh 4))) {
            (set! hold_key vh)
            (set! hold_ms now_ms)
            (set! repeating 0)
          })
        }))

    (if (> hold_key 0)
        ;; 按住跟踪中（独占本触摸序列）
        (if (== tr 1)
            { ; 松手：未进入反复则单次触发
              (if (== repeating 0) (vkey_action hold_key))
              (set! hold_key 0)
              (set! repeating 0)
            }
            { ; 按住：满 HOLD_MS 进入反复，此后每拍一次
              (if (and (== repeating 0) (>= (- now_ms hold_ms) HOLD_MS)) (set! repeating 1))
              (if (== repeating 1) (vkey_action hold_key))
            })
        ;; 无按住跟踪：松手沿单次触发 旋转 / 速降 / 退出
        (if (== tr 1) {
          (set! vh (vkey_hit press_x press_y))
          (if (== vh 2) (T.rotate_piece))
          (if (== vh 4) {
            (T.step_down)
            (T.reset_fall)
          })
          (if (== vh 5) (set! modal 1))
        }))
  }))

;; ============================================================================
;; 主循环：一拍 = 采样输入 → 逻辑推进 → 整帧重绘
;; ============================================================================

(define tick
  (lambda () {
    (if (== stopped 1)
        ;; 冻结态：只补画一次"已退出"提示
        (if (== stopped_drawn 0) {
          (set! stopped_drawn 1)
          (render)
        })
        {
          ;; 实测步长（钳到 [0, DT_MAX]）
          (set! now_ms (System.timestamp))
          (set! dt (- now_ms last_ms))
          (set! last_ms now_ms)
          (if (< dt 0) (set! dt 0))
          (if (> dt DT_MAX) (set! dt DT_MAX))

          (set! tp (if (Touch.take_press) 1 0))
          (set! tr (if (Touch.take_release) 1 0))
          (if (== tp 1) {
            (set! press_x (Touch.x))
            (set! press_y (Touch.y))
          })

          (if (== modal 1) (modal_input) (game_input))

          ;; 重力（确认框打开 / 已结束时暂停；advance 内部再判 game_over）
          (if (== modal 0) (T.advance dt))

          (render)
        })
  }))

;; ============================================================================
;; 启动
;; ============================================================================

(T.reset)
(set! last_ms (System.timestamp))
(tick)
(set! _timer (System.set_interval TICK_MS tick))