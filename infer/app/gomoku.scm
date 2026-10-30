;; ============================================================================
;; 五子棋（Gomoku）—— Nano「灵机应用程序」Scheme App（AI 与人对弈）
;;
;; 玩法：15×15 棋盘，人执黑先行，AI 执白。点交叉点落子；AI 用
;;       「活三 / 冲四」棋形打分 + α-β（深度 2~4，迭代加深 + 700ms 预算）计算应手，
;;       并把**候选点分数热图**画在棋盘上（越红越热）。
;;
;; 分工（同 app/tetris.scm + app/lib/tetris_core.scm 的惯例）：
;;   · app/lib/gomoku_core.scm —— 纯逻辑：棋盘、棋形评分、增量局面评估、α-β 搜索；
;;   · 本文件 —— 布局、绘制、触屏交互、节拍。
;;
;; 交互（全部触屏，右侧面板是按钮区）：
;;   棋盘交叉点   —— 落子（只在你方回合、且该点为空时生效）
;;   新局 / 悔棋 / 热图 / 强度（深度 2~4）
;;   右上角「退出」—— 与硬 A/ESC 同一条通道（app/lib/exit.scm 的全局确认框）
;;
;; 界面口径：
;;   · 热图 = 当前**轮到你**这一方在每个候选空点落子所能形成的棋形分
;;     （活四/冲四/活三/眠三/活二，四方向叠加；取对数归一化后着色）。
;;   · 面板显示：轮到谁、搜索深度、AI 上一手耗时/节点数、局面评估分。
;;
;; 解释器约束（manual.md）：算术恒二参数、无一元取负、无 let/list/map、
;; cond 每子句单个表达式、块内不 define、'() 字面量是静态槽（本 App 的动态表
;; 都用 push/原地改，清空用 pop 到空）、break/continue 必须写裸符号。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Diag)
(native App)

(import G "app/lib/gomoku_core.scm")
(import EX "app/lib/exit.scm")

;; ---- 布局 ----------------------------------------------------------------
(define SW (Gfx.width))
(define SH (Gfx.height))
(define CELL 13)                 ; 棋盘格距（像素）
(define BX 8)                    ; 第 0 列交叉点的 x
(define BY 22)                   ; 第 0 行交叉点的 y
(define SPAN (* 14 CELL))        ; 棋盘线跨度
(define BD_W (+ (+ BX SPAN) 6))  ; 棋盘区域宽（右侧留白给棋子半径）
(define PANEL_X (+ BD_W 4))      ; 右侧面板起点
(define PANEL_W (- SW PANEL_X))

;; 按钮几何（面板内，一列四枚）
(define BTN_X (+ PANEL_X 2))
(define BTN_W (- PANEL_W 6))
(define BTN_H 20)
(define BTN_Y0 92)
(define BTN_DY 24)

;; 配色
(define C_BOARD_R 24) (define C_BOARD_G 30) (define C_BOARD_B 42)
(define C_PANEL_R 14) (define C_PANEL_G 16) (define C_PANEL_B 22)
(define C_GRID_R 62) (define C_GRID_G 76) (define C_GRID_B 100)
(define C_BTN_R 44) (define C_BTN_G 48) (define C_BTN_B 62)
(define C_BTN_ON_R 30) (define C_BTN_ON_G 96) (define C_BTN_ON_B 120)

;; ---- 对局状态 ------------------------------------------------------------
(define human 1)        ; 人执黑
(define ai 2)           ; AI 执白
(define side 1)         ; 轮到谁
(define over 0)         ; 是否终局
(define winner 0)       ; 0 无 / 1 黑胜 / 2 白胜 / 3 和棋
(define depth 4)        ; AI 搜索深度（2~4，可切换）
(define budget 700)     ; AI 每手时间预算（ms）
(define showheat 1)     ; 是否显示候选点热图
(define thinkms 0)      ; AI 上一手耗时
(define thinkn 0)       ; AI 上一手搜索节点数
(define evalv 0)        ; AI 对上一手后局面的评估
(define nmove 0)        ; 已走手数
(define dirty 1)        ; 需要重绘
(define busy 0)         ; 正在思考（绘制用）

;; 历史（原地表：每手 push x, y, pl）
(define hist '())
(define hlen 0)

;; 触屏采样
(define tp 0) (define tr 0)
(define px0 0) (define py0 0)
(define pressed 0)
(define hitx 0) (define hity 0) (define hitc 0) (define hitr 0)
(define tcx 0) (define tcy 0)
(define _timer 0)

;; 绘制临时量
(define dr 0) (define dc 0) (define dx 0) (define dy 0) (define dv 0)
(define hr 0) (define hg 0) (define hb 0)

;; ============================================================================
;; 绘制辅助
;; ============================================================================

(define gx (lambda (c) (+ BX (* c CELL))))
(define gy (lambda (r) (+ BY (* r CELL))))

;; 文本居中（12px 等宽字模，步进 12）
(define draw_ctext
  (lambda (s cx cy r g b)
    (Gfx.draw_text s (- cx (Math.floor (/ (* 12 (String.length s)) 2)))
                   (- cy 7) r g b)))

(define draw_panel_text
  (lambda (s ly r g b) (Gfx.draw_text s (+ PANEL_X 4) ly r g b)))

;; 按钮（on=1 时用高亮配色）
(define draw_btn
  (lambda (by0 label on)
    (if (== on 1)
        (Gfx.fill_rect BTN_X by0 BTN_W BTN_H C_BTN_ON_R C_BTN_ON_G C_BTN_ON_B)
        (Gfx.fill_rect BTN_X by0 BTN_W BTN_H C_BTN_R C_BTN_G C_BTN_B))
    (Gfx.draw_rect BTN_X by0 BTN_W BTN_H 96 102 122)
    (draw_ctext label (+ BTN_X (Math.floor (/ BTN_W 2))) (+ by0 10) 232 236 244)))

;; 热图着色：按**棋形等级**分档（分数 = 四方向棋形分之和）
;;   五连红 / 活四橙红 / 冲四橙 / 活三黄 / 活二青 / 其余极暗（只留"活跃区域"的痕迹）
(define heat_color
  (lambda (v)
    (cond
      ((>= v G.S_FIVE) { (set! hr 255) (set! hg 78) (set! hb 70) })
      ((>= v G.S_OPEN4) { (set! hr 252) (set! hg 118) (set! hb 56) })
      ((>= v G.S_FOUR) { (set! hr 240) (set! hg 170) (set! hb 68) })
      ((>= v G.S_OPEN3) { (set! hr 184) (set! hg 200) (set! hb 84) })
      ((>= v G.S_OPEN2) { (set! hr 52) (set! hg 156) (set! hb 138) })
      ((>= v 1) { (set! hr 28) (set! hg 36) (set! hb 54) })
      (else { (set! hr 0) (set! hg 0) (set! hb 0) }))))

;; ============================================================================
;; 棋盘绘制
;; ============================================================================

(define draw_grid
  (lambda ()
    (set! dc 0)
    (while (< dc 15) {
      (Gfx.fill_rect (gx dc) BY 1 (+ SPAN 1) C_GRID_R C_GRID_G C_GRID_B)
      (Gfx.fill_rect BX (gy dc) (+ SPAN 1) 1 C_GRID_R C_GRID_G C_GRID_B)
      (set! dc (+ dc 1))
    })
    ;; 星位
    (Gfx.fill_rect (- (gx 3) 1) (- (gy 3) 1) 3 3 132 144 168)
    (Gfx.fill_rect (- (gx 11) 1) (- (gy 3) 1) 3 3 132 144 168)
    (Gfx.fill_rect (- (gx 3) 1) (- (gy 11) 1) 3 3 132 144 168)
    (Gfx.fill_rect (- (gx 11) 1) (- (gy 11) 1) 3 3 132 144 168)
    (Gfx.fill_rect (- (gx 7) 1) (- (gy 7) 1) 3 3 152 164 188)))

;; 候选点热图（画在格线之下、棋盘底色之上）
(define draw_heat
  (lambda ()
    (if (== showheat 1)
        {
          (set! dc 0)
          (while (< dc G.hn) {
            (set! dx (get_item G.hmx dc))
            (set! dy (get_item G.hmy dc))
            (set! dv (get_item G.hms dc))
            (if (>= dv 1)
                {
                  (heat_color dv)
                  (Gfx.fill_rect (- (gx dx) 5) (- (gy dy) 5) 11 11 hr hg hb)
                }
                #f)
            (set! dc (+ dc 1))
          })
        }
        #f)))

(define draw_stones
  (lambda ()
    (set! dc 0)
    (while (< dc 15) {
      (set! dr 0)
      (while (< dr 15) {
        (set! dv (G.cell dc dr))
        (if (== dv 1)
            {
              ;; 黑子：亮环 + 深色子身（在暗色棋盘上清晰可辨）
              (Gfx.fill_circle (gx dc) (gy dr) 5 172 178 194)
              (Gfx.fill_circle (gx dc) (gy dr) 4 24 26 34)
            }
            (if (== dv 2)
                {
                  (Gfx.fill_circle (gx dc) (gy dr) 5 120 126 142)
                  (Gfx.fill_circle (gx dc) (gy dr) 4 236 238 244)
                }
                #f))
        (set! dr (+ dr 1))
      })
      (set! dc (+ dc 1))
    })))

;; 最后一手的标记：在棋子正中的小方块（黑子用亮色、白子用深色）
(define draw_last
  (lambda ()
    (if (> nmove 0)
        {
          (set! dc (get_item hist (- hlen 3)))
          (set! dr (get_item hist (- hlen 2)))
          (set! dv (get_item hist (- hlen 1)))
          (if (== dv 1)
              (Gfx.fill_rect (- (gx dc) 2) (- (gy dr) 2) 5 5 250 176 72)
              (Gfx.fill_rect (- (gx dc) 2) (- (gy dr) 2) 5 5 24 24 32))
        }
        #f)))

;; ============================================================================
;; 面板 / 结果 / 忙提示
;; ============================================================================

(define side_label
  (lambda ()
    (cond ((== over 1) (cond ((== winner 1) "你胜") ((== winner 2) "AI胜") (else "和棋")))
          ((== side human) "轮到你")
          (else "AI 思考"))))

(define draw_panel
  (lambda ()
    (Gfx.fill_rect PANEL_X 17 PANEL_W (- SH 17) C_PANEL_R C_PANEL_G C_PANEL_B)
    (Gfx.fill_rect (- PANEL_X 1) 17 1 (- SH 17) 40 44 58)
    (draw_panel_text (side_label) 22 236 240 248)
    (if (== busy 1)
        {
          (Gfx.fill_rect (+ PANEL_X 2) 34 (- PANEL_W 4) 44 30 44 58)
          (draw_panel_text "AI 思考中" 42 240 224 120)
          (draw_panel_text (String.concat "α-β 深度 " (String.atom_to_string depth))
                           58 150 176 200)
        }
        {
          (draw_panel_text (String.concat "用时" (String.concat (String.atom_to_string thinkms) "ms"))
                           38 176 184 200)
          (draw_panel_text (String.concat "节点" (String.atom_to_string thinkn)) 54 176 184 200)
          (draw_panel_text (String.concat "评估" (String.atom_to_string evalv)) 70 176 184 200)
        })
    ;; 按钮：新局 / 悔棋 / 热图 / 强度
    (draw_btn BTN_Y0 "新局" 0)
    (draw_btn (+ BTN_Y0 BTN_DY) "悔棋" 0)
    (draw_btn (+ BTN_Y0 (* 2 BTN_DY))
              (if (== showheat 1) "热图:开" "热图:关")
              (if (== showheat 1) 1 0))
    (draw_btn (+ BTN_Y0 (* 3 BTN_DY))
              (String.concat "强度:" (String.atom_to_string depth)) 0)
    (draw_panel_text (String.concat "手数 " (String.atom_to_string nmove))
                     (+ BTN_Y0 (* 4 BTN_DY)) 150 158 176)
    (draw_panel_text "热图=候选分" (+ BTN_Y0 (+ (* 4 BTN_DY) 18)) 120 128 148)
    (Gfx.draw_text "点交叉点落子" 6 210 150 158 176)
    (Gfx.draw_text "A=退出 · 白=AI" 6 224 110 118 136)))

(define draw_result
  (lambda ()
    (Gfx.fill_rect (+ BX 6) 84 (- SPAN 12) 56 8 10 16)
    (Gfx.draw_rect (+ BX 6) 84 (- SPAN 12) 56 120 130 150)
    (draw_ctext (cond ((== winner 1) "你赢了！")
                      ((== winner 2) "AI 赢了")
                      (else "和棋"))
                (+ BX (Math.floor (/ SPAN 2))) 102 250 250 255)
    (draw_ctext "点「新局」再来一局"
                (+ BX (Math.floor (/ SPAN 2))) 122 170 178 196)))

;; ============================================================================
;; 整帧绘制
;; ============================================================================

(define render
  (lambda ()
    (Gfx.clear)
    (Gfx.fill_rect 0 17 BD_W (- SH 17) C_BOARD_R C_BOARD_G C_BOARD_B)
    (draw_heat)
    (draw_grid)
    (draw_stones)
    (draw_last)
    (draw_panel)
    (if (== over 1) (draw_result) #f)
    (EX.draw_button)
    (Gfx.refresh)
    (set! dirty 0)))

;; ============================================================================
;; 动作
;; ============================================================================

;; 重算热图：当前轮到的一方的候选点棋形分（终局时保留最后一方的热图）
(define do_heat
  (lambda ()
    (G.heat side)))

;; 落一子并记录历史；返回 1 表示成五
(define place
  (lambda (cx cy pl)
    (define f 0)
    (set! f (G.put cx cy pl))
    (push hist cx) (push hist cy) (push hist pl)
    (set! hlen (+ hlen 3))
    (set! nmove (+ nmove 1))
    f))

;; 撤掉最后一手（内部：从历史里取）
(define unplace_last
  (lambda ()
    (define lx 0)
    (define ly 0)
    (if (> hlen 0)
        {
          (set! lx (get_item hist (- hlen 3)))
          (set! ly (get_item hist (- hlen 2)))
          (G.take lx ly)
          (pop hist) (pop hist) (pop hist)
          (set! hlen (- hlen 3))
          (set! nmove (- nmove 1))
        }
        #f)))

(define new_game
  (lambda ()
    (while (> hlen 0) { (pop hist) (set! hlen (- hlen 3)) })
    (G.reset)
    (set! side human)
    (set! over 0)
    (set! winner 0)
    (set! nmove 0)
    (set! thinkms 0)
    (set! thinkn 0)
    (set! evalv 0)
    (do_heat)
    (set! dirty 1)))

;; 悔棋：退回到你方回合（若最后一手是 AI 的，退两手；否则退一手）
(define undo
  (lambda ()
    (if (> hlen 0)
        {
          (set! over 0)
          (set! winner 0)
          (unplace_last)
          ;; 轮到我方时，最后一手是 AI 的：撤掉它之后再把**我的**上一手也撤掉，
          ;; 这样才是"重下我那一手"（撤两手）。
          (if (> hlen 0)
              (if (== (get_item hist (- hlen 1)) human) (unplace_last) #f)
              #f)
          (set! side human)
          (set! thinkms 0)
          (set! thinkn 0)
          (do_heat)
          (set! dirty 1)
        }
        #f)))

;; AI 走一手：先画"思考中"帧 → 搜索 → 落子 → 更新热图
(define ai_move
  (lambda ()
    (define f 0)
    (define mvg 0)
    (set! busy 1)
    (set! dirty 1)
    (render)                       ; 先把"思考中"推给屏幕
    ;; 搜索期间用户可能又点了屏幕：把触屏边沿丢掉，避免思考结束后误落子
    (Touch.take_press) (Touch.take_release)
    (set! pressed 0)
    (G.search ai depth budget)
    (set! thinkms G.ms)
    (set! thinkn G.nodes)
    (set! mvg 0)
    (if (== (G.cell G.bx G.by) 0)
        { (set! f (place G.bx G.by ai)) (set! mvg 1) }
        #f)
    (set! busy 0)
    (if (== mvg 1)
        {
          (set! evalv G.gtotal)
          (if (== f 1)
              { (set! over 1) (set! winner ai) }
              (if (>= nmove 225)
                  { (set! over 1) (set! winner 3) }
                  (set! side human)))
        }
        {
          (set! evalv G.gtotal)
          (set! side human)
        })
    ;; 搜索后触屏电平可能已变：重新同步（丢弃残留边沿）
    (Touch.take_press) (Touch.take_release)
    (set! pressed 0)
    (do_heat)
    (set! dirty 1)))

;; 人走一手
(define human_move
  (lambda (cx cy)
    (define f 0)
    (set! f (place cx cy human))
    (set! evalv G.gtotal)
    (if (== f 1)
        { (set! over 1) (set! winner human) (set! side human) }
        (if (>= nmove 225)
            { (set! over 1) (set! winner 3) (set! side human) }
            (set! side ai)))
    (if (and (== over 0) (== side ai))
        (ai_move)                  ; 人走完立刻轮到 AI（阻塞至搜索结束）
        { (do_heat) (set! dirty 1) })))

;; ============================================================================
;; 触屏处理
;; ============================================================================

;; 触摸点 → 棋盘交叉点（写 hitx/hity；写 hitc/hitr，越界时 hitc=-1）
(define pick_cell
  (lambda (px py)
    (set! hitc -1)
    (set! tcx (/ (- px BX) CELL))
    (set! tcy (/ (- py BY) CELL))
    (set! hitx (Math.round tcx))
    (set! hity (Math.round tcy))
    (if (and (>= hitx 0) (and (<= hitx 14) (and (>= hity 0) (<= hity 14))))
        (if (and (<= (Math.abs (- tcx hitx)) 0.62)
                 (<= (Math.abs (- tcy hity)) 0.62))
            {
              (set! hitc hitx)
              (set! hitr hity)
            }
            #f)
        #f)))

(define in_btn
  (lambda (px py by0)
    (and (>= px BTN_X)
         (and (< px (+ BTN_X BTN_W))
              (and (>= py by0) (< py (+ by0 BTN_H)))))))

(define handle_tap
  (lambda (px py)
    (cond
      ((in_btn px py BTN_Y0) (new_game))
      ((in_btn px py (+ BTN_Y0 BTN_DY)) (undo))
      ((in_btn px py (+ BTN_Y0 (* 2 BTN_DY)))
       { (set! showheat (- 1 showheat)) (set! dirty 1) })
      ((in_btn px py (+ BTN_Y0 (* 3 BTN_DY)))
       {
         (set! depth (if (== depth 4) 2 (+ depth 1)))
         (set! dirty 1)
       })
      (else
        (if (and (== over 0) (== side human))
            {
              (pick_cell px py)
              (if (and (>= hitc 0) (== (G.cell hitc hitr) 0))
                  (human_move hitc hitr)
                  #f)
            }
            #f)))))

;; ============================================================================
;; 主循环
;; ============================================================================

(define tick
  (lambda ()
    (EX.check)
    (set! tp (if (Touch.take_press) 1 0))
    (set! tr (if (Touch.take_release) 1 0))
    (if (== tp 1)
        {
          (set! px0 (Touch.x))
          (set! py0 (Touch.y))
          (set! pressed 1)
        }
        #f)
    (if (== tr 1)
        (if (== pressed 1)
            {
              (set! pressed 0)
              (handle_tap px0 py0)
            }
            #f)
        #f)
    (if (== dirty 1) (render) #f)))

;; ============================================================================
;; 启动
;; ============================================================================

(EX.install_with render)
(new_game)
(render)
(set! _timer (System.set_interval 33 tick))