;; ============================================================================
;; 2048 —— 灵机应用程序（4×4 滑动合并 + 撤销 + 自动存档）
;;
;; 玩法：4×4 棋盘，手指在屏幕上朝任意方向滑动 → 全部方块朝该方向靠拢，相同数字
;;       相撞合并成两倍值；每走一步在空位补一枚新方块（90% 是 2、10% 是 4）。
;;       合出 2048 即为赢（可继续挑战更高分），无处可走即终局。
;;       右侧面板：分数 / 最高分 / 新局 / 撤销 / 步数 / 存档状态。
;;
;; 为什么这个 App 特别适合灵机：**纯逻辑、零物理**。没有粒子、没有流场，一秒里
;; 绝大多数的拍什么都不用做（静止时每拍只采样一次触屏电平 ≈ 4 次本地库调用）；
;; 只有"滑动 / 点按"那一下才真正算一步、重绘一屏。棋盘状态是 16 个整数，撤销是
;; 17 个整数一份快照，存档是 19 个整数的一行文本——没有浮点、没有累积误差。
;;
;; 分工（同 app/tetris.scm + app/lib/tetris_core.scm 的惯例）：
;;   · app/lib/g2048_core.scm —— 纯逻辑：棋盘（指数表示）、四方向合并、随机落子、
;;     终局判定、撤销栈、存档文本的序列化/反序列化；可脱离界面单独回归；
;;   · 本文件 —— 布局、绘制、滑动手势、节拍、文件读写。
;;
;; 触屏手势（本文件的重点）：
;;   按下记起点、每拍刷新最近触点、松手时算位移 (dx,dy)：
;;     · max(|dx|,|dy|) ≥ SWIPE_MIN（16px）→ 判定为主轴方向，走一步；
;;     · 否则算作点按 → 只认面板上的按钮与浮层按钮（**不做"点格子"交互**，
;;       2048 的输入只有滑动，避免落点误判）；
;;     · 右上角「退出」按钮的热区（x ≥ 268 且 y ≤ 20）里的按下**不参与手势**，
;;       否则点退出会被当成一次滑动。
;;   契合点：手指滑动的物理时长 ≈ 100~300ms，跨 3~9 个 tick（33ms/拍），电平采样
;;   必能采到"按住中"的位置；因此不依赖 touch 边沿锁存，只用电平 + 坐标。
;;
;; 自动存档（持久化）：
;;   · 每次有效移动后置脏；静止 SAVE_IDLE 拍（≈0.7s）后落盘一次（不是每步都写）；
;;   · 退出确认后由 (EX.on_exit) 再补一次（刚走完立刻退出也不丢）；
;;   · 启动时读回：文件不存在/损坏就开新局。
;;   · 浏览器目标：写入经 hal_fs_persist_wasm.c 同步进 IndexedDB（刷新页面仍在）；
;;     桌面/掌机：直接写业务资源根下的 data/2048.txt。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数、没有一元取负、无 let/list/map、
;; cond 每子句单个表达式、块内不 define、if 的分支数写全、/ 与 mod 恒返回浮点、
;; 运行期不造多余短字符串（HUD 数字走 10 个字模逐位贴）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native Fs)

(import G "app/lib/g2048_core.scm")
(import EX "app/lib/exit.scm")

;; ---- 布局 ---------------------------------------------------------------

(define SW (Gfx.width))
(define SH (Gfx.height))
(define HDR_H 20)

;; 棋盘：4 格 × 44px + 5 道 4px 缝隙 = 196；左上角 (6,24) → 右下角 (202,220)
(define BX 6)
(define BY 24)
(define BS 196)
(define TS 44)

;; 16 个格子的左上角坐标（常量表：省掉下标除法与取整）
(define XS '(10 58 106 154   10 58 106 154   10 58 106 154   10 58 106 154))
(define YS '(28 28 28 28   76 76 76 76   124 124 124 124   172 172 172 172))

;; 右侧面板（206..314）
(define PX 206)
(define PW 108)
(define BTN_X 208)
(define BTN_W 104)
(define BTN_H 22)
(define BTN_NEW_Y 96)
(define BTN_UND_Y 122)

;; 浮层（胜利 / 终局）
(define OV_X 56)
(define OV_Y 76)
(define OV_W 208)
(define OV_H 96)
(define OV_B1_X 68)
(define OV_B2_X 162)
(define OV_BY 138)
(define OV_BW 90)
(define OV_BH 24)

;; 手势阈值：位移超过它才算滑动（不然就是点按）
(define SWIPE_MIN 16)

;; 动效：一步的滑动用 AN_N 拍走完（33ms/拍 → ≈130ms）
(define AN_N 4)

;; 自动存档
(define SAVE_PATH "data/2048.txt")
(define SAVE_IDLE 20)

;; ---- 配色（经典 2048 的暖色块 + 本机 App 的深色底） ----------------------

(define C_PAGE_R 13) (define C_PAGE_G 15) (define C_PAGE_B 19)
(define C_HDR_R 24) (define C_HDR_G 27) (define C_HDR_B 34)
(define C_PANEL_R 20) (define C_PANEL_G 23) (define C_PANEL_B 29)
(define C_BOARD_R 30) (define C_BOARD_G 33) (define C_BOARD_B 40)
(define C_EMPTY_R 44) (define C_EMPTY_G 48) (define C_EMPTY_B 56)
(define C_BTN_R 48) (define C_BTN_G 52) (define C_BTN_B 64)
(define C_BTN_OFF_R 34) (define C_BTN_OFF_G 37) (define C_BTN_OFF_B 46)

;; ---- 状态 ---------------------------------------------------------------

(define dirty 1)         ; 需要重绘
(define an_t 0)          ; 动效剩余拍数（0 = 没有动效）
(define overlay 0)       ; 0 无 / 1 有浮层
(define ov_kind 1)       ; 1 = 达成 2048 / 2 = 终局
(define save_ok 0)       ; 最近一次落盘是否成功
(define save_dirty 0)    ; 有未落盘的改动
(define save_idle 0)     ; 距上次改动经过的拍数
(define tprev 0)         ; 上一拍的触屏电平
(define gest 0)          ; 1 = 正在跟踪一次可用的手势
(define gsx 0) (define gsy 0)   ; 手势起点
(define glx 0) (define gly 0)   ; 最近触点
(define kc 0)            ; 键盘边沿码
(define _timer 0)

;; 类加载的临时量（避免同名跨调用链串味；每个函数一组前缀）
(define tn 0) (define nd 0)      ; 画数字
(define d_i 0) (define d_ex 0) (define d_s 0)  ; 画方块
(define d_px 0) (define d_py 0) (define d_sx 0) (define d_sy 0)
(define f_dx 0) (define f_dy 0) (define f_adx 0) (define f_ady 0)
(define l_s "")                  ; 读档

;; ---- 数字字模（运行期不造字符串：mlp/callcc 同款做法） -------------------

(define DGS '("0" "1" "2" "3" "4" "5" "6" "7" "8" "9"))
(define DW 7)                    ; 12px 字模的 ASCII 步进（实测 ≈6.9px）

(define ndig
  (lambda (vv)
    {
      (define n 1)
      (define t 10.0)
      (while (>= vv t) { (set! t (* t 10.0)) (set! n (+ n 1)) })
      n
    }))

;; 左对齐画一个非负整数，返回画掉的宽度
(define draw_int
  (lambda (vv x y r g b)
    {
      (define n (ndig vv))
      (define t 1.0)
      (define i 1)
      (define d 0)
      (define rem vv)
      (while (< i n) { (set! t (* t 10.0)) (set! i (+ i 1)) })
      (set! i 0)
      (while (< i n)
        {
          (set! d (Math.floor (/ rem t)))
          (Gfx.draw_text (get_item DGS d) (+ x (* i DW)) y r g b)
          (set! rem (- rem (* d t)))
          (set! t (/ t 10.0))
          (set! i (+ i 1))
        })
      (* n DW)
    }))

;; 右对齐到 x = right 的位置（面板里的数值列）
(define draw_int_r
  (lambda (vv right y r g b)
    {
      (define n (ndig vv))
      (draw_int vv (- right (* n DW)) y r g b)
    }))

;; ---- 方块配色 -----------------------------------------------------------

(define tc_r 0) (define tc_g 0) (define tc_b 0)   ; 底
(define tt_r 0) (define tt_g 0) (define tt_b 0)   ; 字

(define tile_color
  (lambda (e)
    (cond
      ((== e 1)  { (set! tc_r 238) (set! tc_g 228) (set! tc_b 218) (set! tt_r 104) (set! tt_g 94) (set! tt_b 84) })
      ((== e 2)  { (set! tc_r 237) (set! tc_g 224) (set! tc_b 200) (set! tt_r 104) (set! tt_g 94) (set! tt_b 84) })
      ((== e 3)  { (set! tc_r 242) (set! tc_g 177) (set! tc_b 121) (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      ((== e 4)  { (set! tc_r 245) (set! tc_g 149) (set! tc_b 99)  (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      ((== e 5)  { (set! tc_r 246) (set! tc_g 124) (set! tc_b 95)  (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      ((== e 6)  { (set! tc_r 246) (set! tc_g 94)  (set! tc_b 59)  (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      ((== e 7)  { (set! tc_r 237) (set! tc_g 207) (set! tc_b 114) (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      ((== e 8)  { (set! tc_r 237) (set! tc_g 204) (set! tc_b 97)  (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      ((== e 9)  { (set! tc_r 237) (set! tc_g 200) (set! tc_b 80)  (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      ((== e 10) { (set! tc_r 237) (set! tc_g 197) (set! tc_b 63)  (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      ((== e 11) { (set! tc_r 237) (set! tc_g 194) (set! tc_b 46)  (set! tt_r 250) (set! tt_g 248) (set! tt_b 244) })
      (else      { (set! tc_r 94)  (set! tc_g 196) (set! tc_b 204) (set! tt_r 14)  (set! tt_g 40)  (set! tt_b 44) }))))

;; ---- 绘制：方块 ---------------------------------------------------------

(define draw_tile
  (lambda (ex px py)
    {
      (tile_color ex)
      (Gfx.fill_rect px py TS TS tc_r tc_g tc_b)
      (set! tn (get_item G.VALS ex))
      (set! nd (ndig tn))
      (draw_int tn (+ px (Math.floor (/ (- TS (* nd DW)) 2))) (+ py 16) tt_r tt_g tt_b)
    }))

;; 一个方块（ex）画在格子 i 处 / 或动效中的插值位置
(define draw_tiles
  (lambda ()
    {
      (set! d_i 0)
      (while (< d_i 16)
        {
          (set! d_ex (G.cell d_i))
          (if (> d_ex 0)
              {
                (set! d_px (get_item XS d_i))
                (set! d_py (get_item YS d_i))
                (set! d_s (get_item G.an_src d_i))
                (if (> an_t 0)
                    (if (and (>= d_s 0) (not (== d_s d_i)))
                        {
                          (set! d_sx (get_item XS d_s))
                          (set! d_sy (get_item YS d_s))
                          (set! d_px (+ d_sx (Math.round (* (- (get_item XS d_i) d_sx)
                                                           (/ (- AN_N an_t) AN_N)))))
                          (set! d_py (+ d_sy (Math.round (* (- (get_item YS d_i) d_sy)
                                                           (/ (- AN_N an_t) AN_N)))))
                        }
                        #f)
                    #f)
                (draw_tile d_ex d_px d_py)
              }
              #f)
          (set! d_i (+ d_i 1))
        })
    }))

;; ---- 绘制：面板 / 顶栏 / 浮层 -------------------------------------------

(define draw_btn
  (lambda (by0 label on)
    {
      (if (== on 1)
          (Gfx.fill_rect BTN_X by0 BTN_W BTN_H C_BTN_R C_BTN_G C_BTN_B)
          (Gfx.fill_rect BTN_X by0 BTN_W BTN_H C_BTN_OFF_R C_BTN_OFF_G C_BTN_OFF_B))
      (Gfx.draw_rect BTN_X by0 BTN_W BTN_H 92 98 118)
      (Gfx.draw_text label (+ BTN_X 40) (+ by0 5) 226 232 242)
    }))

(define draw_panel
  (lambda ()
    {
      (Gfx.fill_rect PX BY PW BS C_PANEL_R C_PANEL_G C_PANEL_B)
      (Gfx.draw_text "分数" (+ PX 4) 28 120 130 150)
      (draw_int_r G.score (- (+ PX PW) 4) 42 244 214 130)
      (Gfx.draw_text "最高" (+ PX 4) 62 120 130 150)
      (draw_int_r G.best (- (+ PX PW) 4) 76 176 214 200)
      (draw_btn BTN_NEW_Y "新局" 1)
      (draw_btn BTN_UND_Y "撤销" (if (> G.und_n 0) 1 0))
      (Gfx.draw_text "步数" (+ PX 4) 152 120 130 150)
      (draw_int_r G.nmove (- (+ PX PW) 4) 152 170 180 198)
      (Gfx.draw_text "撤销" (+ PX 4) 168 120 130 150)
      (draw_int_r G.und_n (- (+ PX PW) 4) 168 170 180 198)
      (Gfx.draw_text "存档" (+ PX 4) 188 120 130 150)
      (if (== save_dirty 1)
          (Gfx.draw_text "待存" (- (+ PX PW) 28) 188 214 176 96)
          (if (== save_ok 1)
              (Gfx.draw_text "已存" (- (+ PX PW) 28) 188 120 196 140)
              (Gfx.draw_text "未存" (- (+ PX PW) 28) 188 150 120 120)))
    }))

(define draw_overlay
  (lambda ()
    {
      (Gfx.fill_rect OV_X OV_Y OV_W OV_H 18 20 26)
      (Gfx.draw_rect OV_X OV_Y OV_W OV_H 122 132 152)
      (Gfx.draw_rect (+ OV_X 1) (+ OV_Y 1) (- OV_W 2) (- OV_H 2) 40 46 58)
      (if (== ov_kind 1)
          {
            (Gfx.draw_text "达成 2048！" 124 88 244 214 130)
            (Gfx.draw_text "继续可以追求更高分" 100 108 170 180 200)
          }
          {
            (Gfx.draw_text "无路可走" 136 88 240 150 140)
            (Gfx.draw_text "点新局再来一局" 118 108 170 180 200)
          })
      (Gfx.fill_rect OV_B1_X OV_BY OV_BW OV_BH 48 52 64)
      (Gfx.draw_rect OV_B1_X OV_BY OV_BW OV_BH 92 98 118)
      (Gfx.draw_text "新局" 101 144 226 232 242)
      (Gfx.fill_rect OV_B2_X OV_BY OV_BW OV_BH 46 66 54)
      (Gfx.draw_rect OV_B2_X OV_BY OV_BW OV_BH 96 132 104)
      (if (== ov_kind 1)
          (Gfx.draw_text "继续" 195 144 220 240 224)
          (Gfx.draw_text "撤销" 195 144 220 240 224))
    }))

;; ---- 整帧绘制 -----------------------------------------------------------

(define render
  (lambda ()
    {
      (Gfx.fill_rect 0 0 SW SH C_PAGE_R C_PAGE_G C_PAGE_B)
      ;; 顶栏（右上角 40x18 留给统一的「退出」按钮）
      (Gfx.fill_rect 0 0 SW HDR_H C_HDR_R C_HDR_G C_HDR_B)
      (Gfx.draw_text "2048" 6 4 240 196 96)
      (Gfx.draw_text "滑动合并数字" 50 4 120 134 158)
      ;; 棋盘：先铺深色底，再逐格铺浅色空格
      (Gfx.fill_rect BX BY BS BS C_BOARD_R C_BOARD_G C_BOARD_B)
      (set! d_i 0)
      (while (< d_i 16)
        {
          (Gfx.fill_rect (get_item XS d_i) (get_item YS d_i) TS TS
                         C_EMPTY_R C_EMPTY_G C_EMPTY_B)
          (set! d_i (+ d_i 1))
        })
      (draw_tiles)
      (draw_panel)
      ;; 底部提示条
      (Gfx.fill_rect 0 (- SH 18) SW 18 20 23 29)
      (Gfx.draw_text "滑动屏幕移动 · 方向键亦可" 6 (- SH 15) 118 128 148)
      (if (== overlay 1) (draw_overlay) #f)
      (EX.draw_button)
      (Gfx.refresh)
      (set! dirty 0)
    }))

;; ---- 动作 ---------------------------------------------------------------

(define do_save
  (lambda ()
    (if (Fs.write SAVE_PATH (G.save_text))
        (set! save_ok 1)
        (set! save_ok 0))
    (set! save_dirty 0)
    (set! save_idle 0)
    (set! dirty 1)))

(define new_game
  (lambda ()
    {
      (G.reseed)
      (G.reset)
      (set! overlay 0)
      (set! an_t 0)
      (set! save_dirty 1)
      (set! save_idle 0)
      (set! dirty 1)
    }))

(define do_undo
  (lambda ()
    (if (== (G.undo) 1)
        {
          (set! overlay 0)
          (set! an_t 0)
          (set! save_dirty 1)
          (set! save_idle 0)
          (set! dirty 1)
        }
        #f)))

(define dismiss_win
  (lambda ()
    {
      (G.mark_won)
      (set! overlay 0)
      (set! save_dirty 1)
      (set! save_idle 0)
      (set! dirty 1)
    }))

;; 走一步：dir 0=左 1=右 2=上 3=下。返回 1 = 真的动了。
;; 上一步的滑动动画还没播完就来了新输入 → 直接把动画**钉到终点**再走这一步：
;; 动效只是装饰，不能因为它在播就吞掉玩家的手（连划时尤其明显）。
(define play_move
  (lambda (dir)
    (if (> an_t 0) (set! an_t 0) #f)
    (if (== overlay 0)
        {
          (G.snapshot)
          (if (== (G.move dir) 1)
              {
                (G.spawn)
                (G.score_upd)
                (set! an_t AN_N)
                (set! save_dirty 1)
                (set! save_idle 0)
                (if (and (== G.won 0) (== (G.has_win) 1))
                    { (set! overlay 1) (set! ov_kind 1) }
                    (if (== (G.over) 1)
                        { (set! overlay 1) (set! ov_kind 2) }
                        #f))
                (set! dirty 1)
                1
              }
              {
                (G.unsnapshot)
                0
              })
        }
        0)))

;; ---- 触屏：滑动手势 / 点按 ---------------------------------------------

(define in_rect
  (lambda (px py rx ry rw rh)
    (and (and (>= px rx) (< px (+ rx rw)))
         (and (>= py ry) (< py (+ ry rh))))))

(define handle_tap
  (lambda (px py)
    (cond
      ((in_rect px py BTN_X BTN_NEW_Y BTN_W BTN_H) (new_game))
      ((in_rect px py BTN_X BTN_UND_Y BTN_W BTN_H) (do_undo))
      (else #f))))

(define handle_ov_tap
  (lambda (px py)
    (cond
      ((in_rect px py OV_B1_X OV_BY OV_BW OV_BH) (new_game))
      ((in_rect px py OV_B2_X OV_BY OV_BW OV_BH)
       (if (== ov_kind 1) (dismiss_win) (do_undo)))
      (else #f))))

(define do_swipe
  (lambda (dx dy)
    {
      (set! f_adx (Math.abs dx))
      (set! f_ady (Math.abs dy))
      (if (> f_adx f_ady)
          (if (> dx 0) (play_move 1) (play_move 0))
          (if (> dy 0) (play_move 3) (play_move 2)))
    }))

;; 松手：位移够大就是滑动，否则当作点按
(define finish_gesture
  (lambda ()
    {
      (set! f_dx (- glx gsx))
      (set! f_dy (- gly gsy))
      (if (== overlay 1)
          (handle_ov_tap gsx gsy)
          (if (or (> (Math.abs f_dx) SWIPE_MIN) (> (Math.abs f_dy) SWIPE_MIN))
              (do_swipe f_dx f_dy)
              (handle_tap gsx gsy)))
    }))

;; 每拍采样一次触屏电平（边沿由本 App 自己判：按下记起点、松手结算）
(define sample_touch
  (lambda ()
    {
      (if (Touch.is_pressed)
          {
            (set! glx (Touch.x))
            (set! gly (Touch.y))
            (if (== tprev 0)
                {
                  ;; 按下沿：右上角「退出」按钮热区不参与手势
                  (set! gest 0)
                  (if (and (>= glx 268) (<= gly 20))
                      #f
                      {
                        (set! gest 1)
                        (set! gsx glx)
                        (set! gsy gly)
                      })
                }
                #f)
            (set! tprev 1)
          }
          {
            (if (== tprev 1)
                {
                  (if (== gest 1) (finish_gesture) #f)
                  (set! gest 0)
                }
                #f)
            (set! tprev 0)
          })
    }))

;; 方向键（浏览器物理键盘 / 掌机实体键）：1=上 2=下 3=左 4=右
(define drain_keys
  (lambda ()
    {
      (set! kc (Key.take_press))
      (while (not (== kc 0))
        {
          (if (== kc 1)
              (play_move 2)
              (if (== kc 2)
                  (play_move 3)
                  (if (== kc 3)
                      (play_move 0)
                      (if (== kc 4) (play_move 1) #f))))
          (set! kc (Key.take_press))
        })
    }))

;; ---- 读档 ---------------------------------------------------------------

(define load_game
  (lambda ()
    {
      (set! l_s (Fs.read SAVE_PATH))
      (if (== (G.load_text l_s) 1)
          {
            (set! save_ok 1)
            (set! save_dirty 0)
            (if (and (== G.won 0) (== (G.has_win) 1))
                { (set! overlay 1) (set! ov_kind 1) }
                (if (== (G.over) 1)
                    { (set! overlay 1) (set! ov_kind 2) }
                    #f))
          }
          {
            (G.reseed)
            (G.reset)
            (set! save_ok 0)
            (set! save_dirty 1)
          })
    }))

;; ---- 主循环 -------------------------------------------------------------

(define tick
  (lambda ()
    {
      (EX.check)
      (drain_keys)
      (sample_touch)
      ;; 动效推进（每拍减一；到 0 时再画最后一帧，方块正好落在格子上）
      (if (> an_t 0)
          {
            (set! an_t (- an_t 1))
            (set! dirty 1)
          }
          #f)
      ;; 自动存档：改动后静止若干拍再落盘（不是每步都写盘）
      (if (== save_dirty 1)
          {
            (set! save_idle (+ save_idle 1))
            (if (>= save_idle SAVE_IDLE) (do_save) #f)
          }
          (set! save_idle 0))
      (if (== dirty 1) (render) #f)
    }))

;; ---- 启动 ---------------------------------------------------------------

(load_game)
(render)
(EX.on_exit do_save)
(EX.install_with render)
(set! _timer (System.set_interval 33 tick))