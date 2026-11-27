;@id=snake
;@title=贪吃蛇
;@version=2026-11-27
;; ============================================================================
;; 贪吃蛇（Snake）+ DQN 强化学习 AI —— Nano「灵机应用程序」Scheme App
;;
;; 三个模块（沿用 tetris / flappy 的分工惯例）：
;;   · app/lib/snake_core.scm      纯规则：场地/蛇身/食物/碰撞/成长/计分/观测编码/奖励
;;   · app/lib/snake_dqn.scm       智能体：MLP Q 网络 + 目标网络 + 经验回放 + Adam，
;;                                 前向/反向/更新全部用 Scheme 写（不依赖任何 C 侧推理）
;;   · app/lib/snake_dqn_weights.scm   离线预训练权重（tools/train_snake_dqn.py 生成）
;;   · 本文件                       布局/绘制/触屏键盘/节拍/训练调度/模型读写/退出
;;
;; 三种模式（右侧按钮切换）：
;;   手动 —— 右侧方向盘或键盘方向键/WASD 转向。撞墙/撞自己即结束，1.4 s 后自动重开。
;;    AI   —— 用**最佳快照**（训练中"20 局均分"最好那一版参数）贪心游玩：正常速度、
;;           会一路吃到把自己撑得很长。屏幕上同步显示它当下的三个 Q 值。
;;   训练 —— 加速训练：每拍连续推进若干环境步，ε-贪心探索 + DQN 更新（目标网络每
;;           500 步同步），画面同步显示"当前训练局"和最近 40 局的得分曲线。
;;           训练时把本 App 的解释器时间片提到 1250000（默认 312500 的 4 倍），
;;           退出训练自动还原。
;;
;; 出厂即会玩：启动时先载入 app/lib/snake_dqn_weights.scm（离线 DQN 预训练，
;; 贪心均分 ~28/最高 40+），再尝试载入用户在「训练」里练出来的 data/snake_dqn.txt。
;;    · 训练中每 20 局算一次均分，均分创新高 → 更新"最佳快照"（AI 游玩与落盘都用它）；
;;    · 落盘：离开训练态/退出 App 时（有进步才写）；WASM 目标下 (Fs.write) 还会同步进
;;      浏览器 IndexedDB，刷新页面后自动读回；
;;    · 「出厂」按钮删掉 data 文件并回到预训练权重；「从头学」把网络重置为随机初始
;;      化（真正的从零 DQN，界面上能看到均分从 0 一点点涨起来——慢，但很直观）。
;;
;; 为什么 AI 默认用"预训练 + 继续学"而不是纯粹从零学：解释器一个时间片只跑几十万条
;; VM 指令，一个 NI=28/NH=32 的训练步约 9 万条指令（前向+反向+Adam），浏览器里约
;; 数百步/秒——从零练到"会玩"需要十万级步数（几分钟）。预训练权重（离线 numpy 版
;; DQN，环境/网络/超参与这里逐条同构）解决"打开就能看"，在线微调解决"它真的在学"。
;;
;; 解释器约束（manual.md）：算术/比较恒二参数；无一元取负 → (- 0 x)；
;; 无 let/list/map/append；块内不 define（define 只写在 lambda 体顶层）；
;; cond 每子句单个表达式；/ 与 mod 恒浮点（下标一律 Math.floor）；while 体不是独立
;; 词法区域；'() 是模块级静态槽（空表一律 (cdr '(0)) 现造）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native App)
(native Fs)

(import S  "app/lib/snake_core.scm")
(import Q  "app/lib/snake_dqn.scm")
(import WD "app/lib/snake_dqn_weights.scm")
(import EX "app/lib/exit.scm")      ; 硬 A/ESC 与右上角「退出」按钮 → 全局退出确认框

;; ---- 节拍 -----------------------------------------------------------------

(define TICK_MS 33)             ; 定时器周期（约 30 拍/秒）
(define MOVE_BASE_MS 150)       ; 手动/AI 模式下每走一格的基准间隔
(define MOVE_FAST_MS 70)        ; 吃到很多后的最快间隔
(define DEAD_HOLD_MS 1400)      ; 结束后停留多久自动重开
(define TRAIN_STEPS 12)         ; 训练态每拍推进的环境步数（约 9 万指令/步）
(define TRAIN_RENDER_EVERY 3)   ; 训练态每 3 拍渲染一次（省下绘制开销喂训练）
(define SAVE_EVERY_EP 200)      ; 训练中每多少局尝试落盘一次（有进步才真写）
(define HIST 40)                ; 得分曲线保留的局数（与 snake_dqn.scm 的 HIST 一致）
(define MODEL_PATH "data/snake_dqn.txt")

;; ---- 布局（320x240）-------------------------------------------------------
;; 场地 12x12，格 16px → 192x192，放在 (4,30)..(196,222)；右侧 200..320 是面板。

(define SW (Gfx.width))
(define SH (Gfx.height))
(define CELL 16)
(define FX 4)
(define FY 30)
(define FG S.G)                 ; 边长（12）
(define FW (* FG CELL))         ; 192
(define FH (* FG CELL))         ; 192

(define PX 200)                 ; 右侧面板左沿

;; 模式按钮
(define BTN_W 56)
(define BTN_H 22)
(define BTN_Y 78)
(define BTN_Y2 104)

;; 方向盘（手动 / AI 模式）
(define DP_UP_X 236)
(define DP_UP_Y 174)
(define DP_W 36)
(define DP_H 26)
(define DP_L_X 200)
(define DP_L_Y 202)
(define DP_D_X 236)
(define DP_D_Y 202)
(define DP_R_X 272)
(define DP_R_Y 202)

;; 训练态的得分曲线与两个小按钮
(define CH_X 200)
(define CH_Y 172)
(define CH_W 116)
(define CH_H 38)
(define SB_Y 214)
(define SB_H 18)

;; ---- 模式 -----------------------------------------------------------------

(define MODE_MANUAL 0)
(define MODE_AI 1)
(define MODE_TRAIN 2)
(define mode MODE_MANUAL)

;; ---- 运行状态 -------------------------------------------------------------

(define now_ms 0)
(define last_ms 0)
(define acc_ms 0)               ; 走格时间累加器
(define dead_at 0)              ; 本局结束时刻（0 = 未结束）
(define press_x 0)
(define press_y 0)
(define pend_a -1)              ; 待执行的相对动作（-1 = 无）
(define session_best 0)         ; 本会话最高分（手动与 AI 实战共用）
(define ai_sum 0)               ; AI 实战得分累计（只统计 AI 模式跑完的局）
(define ai_cnt 0)
(define ai_avg 0.0)
(define loaded_ok 0)            ; 启动时是否读到了用户训练的模型
(define toast "")
(define toast_at 0)
(define last_save_ep 0)
(define saved_avg -1.0)
(define train_ticks 0)
(define timer_id 0)


;; ---- 模块级临时量 ---------------------------------------------------------

(define ti 0) (define tj 0) (define tk 0) (define tv 0) (define tu 0)
(define tx 0) (define ty 0) (define th 0)
(define tq 0.0)
(define kc 0)
(define want 0)
(define dd 0)
(define sx 0) (define sy 0) (define sslot 0)
(define hud_s "")
(define stat1 "")
(define stat2 "")
(define n_draw 0)
(define mv_ms 0)

;; ============================================================================
;; 小工具
;; ============================================================================

;; 文本像素宽度（12px 字模：ASCII/拉丁 ≈ 7px、CJK 12px）
(define tw_of
  (lambda (s)
    {
      (define i 0)
      (define n (String.length s))
      (define w 0)
      (define cp 0)
      (while (< i n)
        {
          (set! cp (String.charCodeAt i s))
          (set! w (+ w (if (< cp 11904) 7 12)))
          (set! i (+ i 1))
        })
      w
    }))

(define txt
  (lambda (s x y r g b)
    (Gfx.draw_text s x y r g b)))

(define txt_c
  (lambda (s cx y r g b)
    (Gfx.draw_text s (- cx (Math.floor (/ (tw_of s) 2))) y r g b)))

;; 一位小数 / 两位小数（wasm32 的 %g 会把 0.3000000119 打成 "0.3"，
;; 但把 12.4000005 打成 "12.4"；这里仍自己拼，避免依赖浮点打印）
(define fmt1
  (lambda (v)
    {
      (define k (Math.round (* v 10)))
      (define a (Math.floor (/ k 10)))
      (define b (- k (* a 10)))
      (String.concat (String.atom_to_string a)
                     (String.concat "." (String.atom_to_string b)))
    }))

(define fmt2
  (lambda (v)
    {
      (define k (Math.round (* v 100)))
      (define a (Math.floor (/ k 100)))
      (define b (- k (* a 100)))
      (String.concat (String.atom_to_string a)
                     (String.concat "."
                       (if (< b 10)
                           (String.concat "0" (String.atom_to_string b))
                           (String.atom_to_string b))))
    }))

(define in_rect
  (lambda (x y rx ry rw rh)
    (if (and (>= x rx) (< x (+ rx rw))) (if (and (>= y ry) (< y (+ ry rh))) 1 0) 0)))

;; ============================================================================
;; 绘制
;; ============================================================================

(define draw_field_bg
  (lambda () {
    (set! ti 0)
    (while (< ti FG)
      {
        (set! tj 0)
        (while (< tj FG)
          {
            (set! tx (+ FX (* tj CELL)))
            (set! ty (+ FY (* ti CELL)))
            (if (== (mod (+ ti tj) 2) 0)
                (Gfx.fill_rect tx ty CELL CELL 18 25 34)
                (Gfx.fill_rect tx ty CELL CELL 22 30 40))
            (set! tj (+ tj 1))
          })
        (set! ti (+ ti 1))
      })
    ;; 边框（2px）
    (Gfx.draw_rect (- FX 2) (- FY 2) (+ FW 4) (+ FH 4) 44 62 80)
  }))

;; 一个蛇身段：k=0 是头
(define draw_seg
  (lambda (k)
    {
      (set! sslot (S.slot_of k))
      (set! sx (+ FX (* (get_item S.bx sslot) CELL)))
      (set! sy (+ FY (* (get_item S.by sslot) CELL)))
      (if (== k 0)
          { ; 头：亮绿 + 一对白眼睛（朝向当前前进方向）
            (Gfx.rounded_rect (+ sx 1) (+ sy 1) (- CELL 2) (- CELL 2) 6 6 6 6
                              110 244 156 1)
            (set! dd S.dir)
            (if (or (== dd 1) (== dd 3))
                {
                  (set! tx (if (== dd 1) (+ sx 9) (+ sx 3)))
                  (Gfx.fill_rect tx (+ sy 3) 4 4 255 255 255)
                  (Gfx.fill_rect tx (+ sy 9) 4 4 255 255 255)
                }
                {
                  (set! ty (if (== dd 2) (+ sy 9) (+ sy 3)))
                  (Gfx.fill_rect (+ sx 3) ty 4 4 255 255 255)
                  (Gfx.fill_rect (+ sx 9) ty 4 4 255 255 255)
                })
          }
          { ; 身体：沿长度方向的绿色渐变
            (set! tq (/ k (+ S.blen 4)))
            (Gfx.rounded_rect (+ sx 1) (+ sy 1) (- CELL 2) (- CELL 2) 4 4 4 4
                              (Math.floor (- 62 (* 40 tq)))
                              (Math.floor (- 208 (* 96 tq)))
                              (Math.floor (- 128 (* 50 tq)))
                              1)
          })
    }))

(define draw_snake
  (lambda () {
    (set! ti (- S.blen 1))
    (while (>= ti 0)
      {
        (draw_seg ti)     ; 从尾往头画，头压在最上面
        (set! ti (- ti 1))
      })
  }))

(define draw_food
  (lambda () {
    (if (>= S.food 0)
        {
          (set! tx (+ FX (* (mod S.food FG) CELL)))
          (set! ty (+ FY (* (Math.floor (/ S.food FG)) CELL)))
          (Gfx.fill_circle_aa (+ tx 8) (+ ty 9) 5 240 72 72 1)
          (Gfx.fill_dot_aa_mode (+ tx 6) (+ ty 7) 1.6 255 190 180 1)
          (Gfx.fill_rect (+ tx 8) (+ ty 1) 2 4 96 190 110)
          (Gfx.fill_rect (+ tx 9) (+ ty 1) 3 2 120 210 130)
        }
        #f)
  }))

;; 结束/暂停时的中央提示板
(define draw_overlay
  (lambda (title sub) {
    (set! th 46)
    (set! ty (+ FY (Math.floor (/ (- FH th) 2))))
    (Gfx.rounded_rect (+ FX 8) ty (- FW 16) th 6 6 6 6 8 12 18 1)
    (Gfx.draw_rect (+ FX 8) ty (- FW 16) th 120 170 140)
    (txt_c title (+ FX (Math.floor (/ FW 2))) (+ ty 8) 250 250 250)
    (txt_c sub (+ FX (Math.floor (/ FW 2))) (+ ty 26) 170 200 190)
  }))

;; 得分曲线（训练态）：最近 40 局，柱高按本窗口内最大值归一
(define draw_chart
  (lambda () {
    (define n 0) (define k 0) (define slot 0) (define v 0) (define mx 1) (define bw 0)
    (Gfx.rounded_rect CH_X CH_Y CH_W CH_H 4 4 4 4 16 22 30 1)
    (Gfx.draw_rect CH_X CH_Y CH_W CH_H 40 52 68)
    (set! n (length Q.hist))
    (if (> Q.episodes n) (set! n n) (set! n Q.episodes))
    (if (< n 1) (set! n 1) #f)
    (set! k 0)
    (while (< k n)
      {
        (set! slot (mod (+ (- (+ Q.hist_i HIST) n) k) HIST))
        (set! v (get_item Q.hist slot))
        (if (> v mx) (set! mx v) #f)
        (set! k (+ k 1))
      })
    (set! bw (Math.floor (/ (- CH_W 8) n)))
    (set! k 0)
    (while (< k n)
      {
        (set! slot (mod (+ (- (+ Q.hist_i HIST) n) k) HIST))
        (set! v (get_item Q.hist slot))
        (set! th (Math.floor (/ (- CH_H 16) mx)))
        (set! th (Math.floor (* th v)))
        (if (< th 1) (set! th 1) #f)
        ;; 有得分的柱子亮绿，0 分的柱子暗灰
        (if (> v 0) (set! tk 70) (set! tk 44))
        (if (> v 0) (set! tu 190) (set! tu 52))
        (if (> v 0) (set! tv 120) (set! tv 68))
        (Gfx.fill_rect (+ (+ CH_X 4) (* k bw)) (- (+ CH_Y CH_H) th)
                       (if (> bw 2) (- bw 1) 1) th tk tu tv)
        (set! k (+ k 1))
      })
    (txt "最近得分" (+ CH_X 4) (+ CH_Y 2) 120 140 160)
  }))

(define draw_button
  (lambda (x y w h label active r g b)
    {
      (if (== active 1)
          {
            (Gfx.rounded_rect x y w h 4 4 4 4 r g b 1)
            (Gfx.draw_rect x y w h 110 220 170)
          }
          {
            (Gfx.rounded_rect x y w h 4 4 4 4 44 56 72 1)
            (Gfx.draw_rect x y w h 64 82 104)
          })
      (txt_c label (+ x (Math.floor (/ w 2))) (+ y 5) 236 242 250)
    }))

(define draw_buttons
  (lambda () {
    (draw_button PX BTN_Y BTN_W BTN_H "手动" (if (== mode MODE_MANUAL) 1 0) 30 110 80)
    (draw_button (+ (+ PX BTN_W) 4) BTN_Y BTN_W BTN_H "AI" (if (== mode MODE_AI) 1 0) 30 110 80)
    (draw_button PX BTN_Y2 (+ (+ BTN_W BTN_W) 4) BTN_H
                 (if (== mode MODE_TRAIN) "停止训练" "训练")
                 (if (== mode MODE_TRAIN) 1 0)
                 (if (== mode MODE_TRAIN) 120 30)
                 (if (== mode MODE_TRAIN) 84 110)
                 (if (== mode MODE_TRAIN) 60 80))
  }))

(define draw_dpad
  (lambda () {
    (draw_button DP_UP_X DP_UP_Y DP_W DP_H "上" 0 30 110 80)
    (draw_button DP_L_X DP_L_Y DP_W DP_H "左" 0 30 110 80)
    (draw_button DP_D_X DP_D_Y DP_W DP_H "下" 0 30 110 80)
    (draw_button DP_R_X DP_R_Y DP_W DP_H "右" 0 30 110 80)
  }))

(define draw_status
  (lambda () {
    ;; 分数 / 最高 / 长度
    (set! hud_s (String.concat "分数 " (String.atom_to_string S.score)))
    (txt hud_s PX 30 240 246 252)
    (set! hud_s (String.concat "最高 "
                 (String.atom_to_string (if (> S.score session_best)
                                            S.score session_best))))
    (txt hud_s PX 46 190 200 212)
    (set! hud_s (String.concat "长度 " (String.atom_to_string S.blen)))
    (txt hud_s PX 62 190 200 212)

    (draw_buttons)

    ;; 模式与统计
    (if (== mode MODE_TRAIN)
        {
          (txt "训练中" PX 130 250 210 120)
          (set! stat1 (String.concat "局 " (String.atom_to_string Q.episodes)))
          (set! stat1 (String.concat stat1 "  ε "))
          (set! stat1 (String.concat stat1 (fmt2 Q.eps)))
          (txt stat1 PX 146 200 210 220)
          (set! stat2 (String.concat "均分 " (fmt1 Q.ep_avg)))
          (set! stat2 (String.concat stat2 "  最佳 "))
          (set! stat2 (String.concat stat2 (String.atom_to_string Q.best_score)))
          (txt stat2 PX 160 200 210 220)
          (draw_chart)
          (draw_button PX SB_Y BTN_W SB_H "从头学" 0 90 70 60)
          (draw_button (+ (+ PX BTN_W) 4) SB_Y BTN_W SB_H "出厂" 0 90 70 60)
        }
        {
          (if (== mode MODE_AI)
              {
                (txt "AI 自动" PX 130 120 220 255)
                (set! stat1 (String.concat "局 " (String.atom_to_string ai_cnt)))
                (set! stat1 (String.concat stat1 "  均分 "))
                (set! stat1 (String.concat stat1 (fmt1 ai_avg)))
                (txt stat1 PX 146 200 210 220)
                ;; 三个动作的 Q 值（直行 / 右转 / 左转）——取整显示，保证 116px 面板装得下
                (set! stat2 "Q ")
                (set! stat2 (String.concat stat2
                             (String.atom_to_string (Math.round (get_item Q.QV 0)))))
                (set! stat2 (String.concat stat2 "/"))
                (set! stat2 (String.concat stat2
                             (String.atom_to_string (Math.round (get_item Q.QV 1)))))
                (set! stat2 (String.concat stat2 "/"))
                (set! stat2 (String.concat stat2
                             (String.atom_to_string (Math.round (get_item Q.QV 2)))))
                (txt stat2 PX 160 150 180 200)
              }
              {
                (txt "手动操作" PX 130 250 250 250)
                (txt "方向键 / WASD" PX 146 200 210 220)
                (set! stat2 (String.concat "模型 局 "
                             (String.atom_to_string Q.episodes)))
                (txt stat2 PX 160 200 210 220)
              })
          (draw_dpad)
        })
  }))

(define render
  (lambda () {
    (Gfx.clear)
    ;; 顶栏（右侧 54px 留给统一的「退出」按钮）
    (txt "贪吃蛇" 4 6 240 246 252)
    (txt "DQN 强化学习" 52 6 110 150 190)
    (draw_field_bg)
    (draw_food)
    (draw_snake)
    (if (> dead_at 0)
        (if (== S.won 1)
            (draw_overlay "通关！" "场地填满，自动重开")
            (draw_overlay "游戏结束"
                          (String.concat "分数 " (String.atom_to_string S.score))))
        #f)
    (draw_status)
    ;; 提示条（保存反馈）
    (if (> (String.length toast) 0)
        (if (< (- now_ms toast_at) 1600)
            (txt toast 150 6 250 226 130)
            (set! toast ""))
        #f)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; ============================================================================
;; 模型读写
;; ============================================================================

(define save_model
  (lambda () {
    (define txts "")
    (set! txts (Q.serialize))
    (if (Fs.write MODEL_PATH txts)
        {
          (set! saved_avg Q.snap_avg)
          (set! last_save_ep Q.episodes)
          (set! toast "已保存")
          (set! toast_at now_ms)
          (display "[snake] saved bytes=") (display (String.length txts))
          (display " ep=") (display Q.episodes)
          (display " best=") (display Q.best_score)
          (newline)
        }
        {
          (set! toast "保存失败")
          (set! toast_at now_ms)
        })
  }))

(define load_model
  (lambda () {
    (define txts (Fs.read MODEL_PATH))
    (if (null? txts)
        (set! loaded_ok 0)
        (set! loaded_ok (Q.load_text txts)))
    (display "[snake] user model loaded=") (display loaded_ok)
    (display " ep=") (display Q.episodes)
    (display " best=") (display Q.best_score)
    (newline)
  }))

;; ============================================================================
;; 模式切换
;; ============================================================================

(define start_round
  (lambda () {
    (S.reset)
    (Q.sync_obs)
    (set! dead_at 0)
    (set! acc_ms 0)
    (set! pend_a -1)
  }))

(define set_mode
  (lambda (m) {
    ;; 离开训练态：落盘（有进步才写）+ 还原时间片
    (if (and (== mode MODE_TRAIN) (not (== m MODE_TRAIN)))
        {
          (if (> Q.snap_avg saved_avg) (save_model) #f)
          (App.set_timeslice 312500)
        }
        #f)
    ;; 进入训练态：把时间片提到 4 倍（训练吞吐 ≈ ×4，帧率略降）
    (if (and (not (== mode MODE_TRAIN)) (== m MODE_TRAIN))
        {
          (App.set_timeslice 1250000)
          (Q.set_schedule 0.30 0.03 400)   ; 从 0.30 起，400 局内降到 0.03
          (set! last_save_ep Q.episodes)
        }
        #f)
    (set! mode m)
    (start_round)
    (display "[snake] mode=") (display mode)
    (display " ep=") (display Q.episodes)
    (display " eps=") (display (fmt2 Q.eps))
    (newline)
  }))

;; ============================================================================
;; 训练：一拍推进 TRAIN_STEPS 个环境步
;; ============================================================================

(define train_tick
  (lambda () {
    (set! n_draw 0)
    (while (< n_draw TRAIN_STEPS)
      {
        (if (== S.dead 1)
            {
              (Q.end_episode)
              (S.reset)
              (Q.sync_obs)
              ;; 每 SAVE_EVERY_EP 局：均分（或最佳）有进步才落盘
              (if (>= (- Q.episodes last_save_ep) SAVE_EVERY_EP)
                  {
                    (if (> Q.snap_avg saved_avg) (save_model) #f)
                    (set! last_save_ep Q.episodes)
                  }
                  #f)
              (if (== (mod Q.episodes 100) 0)
                  {
                    (display "[snake] ep=") (display Q.episodes)
                    (display " steps=") (display Q.env_steps)
                    (display " eps=") (display (fmt2 Q.eps))
                    (display " avg=") (display (fmt1 Q.ep_avg))
                    (display " best=") (display Q.best_score)
                    (newline)
                  }
                  #f)
            }
            #f)
        (Q.step_train)
        (set! n_draw (+ n_draw 1))
      })
  }))

;; ============================================================================
;; 手动 / AI 模式：按实测时间推进
;; ============================================================================

(define play_dead
  (lambda () {
    (set! dead_at now_ms)
    (if (> S.score session_best) (set! session_best S.score) #f)
    (if (== mode MODE_AI)
        { ; AI 实战统计（与训练态的"含探索均分"分开：这才是用户看到的战斗力）
          (set! ai_sum (+ ai_sum S.score))
          (set! ai_cnt (+ ai_cnt 1))
          (set! ai_avg (/ ai_sum ai_cnt))
        }
        #f)
    (display "[snake] round over score=") (display S.score)
    (display " mode=") (display mode)
    (newline)
  }))

(define play_step
  (lambda () {
    (if (== mode MODE_AI)
        (Q.step_greedy)
        (if (>= pend_a 0)
            {
              (S.step pend_a)
              (set! pend_a -1)
            }
            (S.step 0)))
    (if (== S.dead 1) (play_dead) #f)
  }))

(define play_tick
  (lambda () {
    ;; 走格间隔随得分缩短（吃得多走得快，同经典贪吃蛇的手感）
    (set! mv_ms (- MOVE_BASE_MS (* 4 S.score)))
    (if (< mv_ms MOVE_FAST_MS) (set! mv_ms MOVE_FAST_MS) #f)
    (set! acc_ms (+ acc_ms (- now_ms last_ms)))
    (if (> acc_ms 400) (set! acc_ms 400) #f)
    (set! ti 0)
    (while (and (>= acc_ms mv_ms) (< ti 4))
      {
        (set! acc_ms (- acc_ms mv_ms))
        (if (== dead_at 0) (play_step) #f)
        (set! ti (+ ti 1))
      })
    (if (> dead_at 0)
        (if (>= (- now_ms dead_at) DEAD_HOLD_MS) (start_round) #f)
        #f)
  }))

;; ============================================================================
;; 输入
;; ============================================================================

;; 把一个"绝对方向"变成相对动作排进队列（掉头忽略）
(define want_dir
  (lambda (w) {
    (if (== mode MODE_TRAIN)
        #f
        {
          (if (== mode MODE_AI)
              { ; 按方向键 = 立刻接管为手动
                (set! mode MODE_MANUAL)
                (display "[snake] takeover manual") (newline)
              }
              #f)
          (set! dd (mod (- (+ w 4) S.dir) 4))
          (if (== dd 0) (set! pend_a 0) #f)
          (if (== dd 1) (set! pend_a 1) #f)
          (if (== dd 3) (set! pend_a 2) #f)
          ;; dd == 2：掉头，蛇不能原地转身，忽略
          #f
        })
  }))

(define press_at
  (lambda (x y) {
    ;; 模式按钮
    (if (== (in_rect x y PX BTN_Y BTN_W BTN_H) 1)
        (if (not (== mode MODE_MANUAL)) (set_mode MODE_MANUAL) #f)
        #f)
    (if (== (in_rect x y (+ (+ PX BTN_W) 4) BTN_Y BTN_W BTN_H) 1)
        (set_mode MODE_AI)
        #f)
    (if (== (in_rect x y PX BTN_Y2 (+ (+ BTN_W BTN_W) 4) BTN_H) 1)
        (if (== mode MODE_TRAIN) (set_mode MODE_MANUAL) (set_mode MODE_TRAIN))
        #f)
    ;; 训练态的两个小按钮
    (if (== mode MODE_TRAIN)
        {
          (if (== (in_rect x y PX SB_Y BTN_W SB_H) 1)
              {
                (Q.init_random)
                (Q.reset_stats)
                (Q.set_schedule 1.0 0.03 1200)
                (set! saved_avg -1.0)
                (set! last_save_ep 0)
                (set! toast "从头学：均分从 0 涨起来需要时间")
                (set! toast_at now_ms)
                (start_round)
                (display "[snake] random init") (newline)
              }
              #f)
          (if (== (in_rect x y (+ (+ PX BTN_W) 4) SB_Y BTN_W SB_H) 1)
              {
                (Q.init_from WD.W1 WD.B1 WD.W2 WD.B2)
                (Q.reset_stats)
                (Q.set_schedule 0.30 0.03 400)
                (set! saved_avg -1.0)
                (set! last_save_ep 0)
                (Fs.remove MODEL_PATH)
                (set! toast "已恢复预训练权重")
                (set! toast_at now_ms)
                (start_round)
                (display "[snake] factory weights") (newline)
              }
              #f)
        }
        {
          ;; 方向盘
          (if (== (in_rect x y DP_UP_X DP_UP_Y DP_W DP_H) 1) (want_dir 0) #f)
          (if (== (in_rect x y DP_L_X DP_L_Y DP_W DP_H) 1) (want_dir 3) #f)
          (if (== (in_rect x y DP_D_X DP_D_Y DP_W DP_H) 1) (want_dir 2) #f)
          (if (== (in_rect x y DP_R_X DP_R_Y DP_W DP_H) 1) (want_dir 1) #f)
        })
  }))

(define handle_input
  (lambda () {
    (if (Touch.take_press)
        {
          (set! press_x (Touch.x))
          (set! press_y (Touch.y))
          (press_at press_x press_y)
        }
        #f)
    (if (Touch.take_release) #f #f)
    ;; 键盘：方向键 1..4，或 WASD
    (set! kc (Key.take_press))
    (while (> kc 0)
      {
        (set! want -1)
        (if (or (== kc 1) (or (== kc 119) (== kc 87))) (set! want 0) #f)
        (if (or (== kc 2) (or (== kc 115) (== kc 83))) (set! want 2) #f)
        (if (or (== kc 3) (or (== kc 97) (== kc 65))) (set! want 3) #f)
        (if (or (== kc 4) (or (== kc 100) (== kc 68))) (set! want 1) #f)
        (if (>= want 0) (want_dir want) #f)
        (set! kc (Key.take_press))
      })
  }))

;; ============================================================================
;; 主循环
;; ============================================================================

(define tick
  (lambda () {
    (EX.check)                       ; 硬 A/ESC / 右上角退出按钮（可能阻塞）
    (set! now_ms (System.timestamp))
    (handle_input)
    (if (== mode MODE_TRAIN)
        (train_tick)
        (play_tick))
    (set! last_ms now_ms)
    ;; 训练态降低渲染频率，把时间片留给训练
    (if (== mode MODE_TRAIN)
        {
          (set! train_ticks (+ train_ticks 1))
          (if (>= train_ticks TRAIN_RENDER_EVERY)
              {
                (set! train_ticks 0)
                (render)
              }
              #f)
        }
        (render))
  }))

;; ============================================================================
;; 启动
;; ============================================================================

(Q.init_from WD.W1 WD.B1 WD.W2 WD.B2)     ; ① 出厂预训练权重
(Q.reset_stats)
(Q.set_schedule 0.03 0.03 1)              ; AI/手动模式不需要探索
(load_model)                              ; ② 用户自己练的（有则覆盖）
(set! session_best 0)
(set! saved_avg Q.snap_avg)
(set! last_save_ep Q.episodes)
(S.reset)
(Q.sync_obs)
(set! last_ms (System.timestamp))
(set! now_ms last_ms)
(display "[snake] start ep=") (display Q.episodes)
(display " best=") (display Q.best_score)
(display " loaded=") (display loaded_ok)
(newline)
(render)
(set! timer_id (System.set_interval TICK_MS tick))

;; ---- 退出：确认后先落盘，再销毁（见 app/lib/exit.scm）----
(EX.install_with render)
(EX.on_exit save_model)
