;; ============================================================================
;; Flappy Bird（像素鸟）+ 强化学习 AI —— Nano「灵机应用程序」Scheme App
;;
;; 三个模块（分工同 tetris / 2048 的惯例）：
;;   · app/lib/flappy_core.scm —— 纯游戏逻辑（鸟的竖直物理、三根管道滚动/回收、
;;     碰撞、计分、随机缝隙、给强化学习用的离散观测索引）。不依赖 Gfx/Touch。
;;   · app/lib/flappy_q.scm    —— 表格型 Q 学习智能体（ε-贪心、Q 更新、位势奖励塑形、
;;     模型文本的序列化/反序列化、单步 train_step/play_step）。
;;   · 本文件 —— 布局、绘制、触屏/键盘、节拍、训练调度、模型文件读写、退出。
;;
;; 三种模式（底部按钮切换）：
;;   手动 —— 点屏幕任意位置（或空格/↑/W）扇翅，撞管道/地面/天花板即结束；
;;   AI   —— 用学到的 Q 表贪心飞行（正常速度，能一直飞）；
;;   训练 —— **加速训练**：每拍连续推进若干个决策（每决策 = 3 个物理步），ε-贪心
;;           一边探索一边更新 Q 表，死了立刻开新局；画面同步显示"训练中的那一局"。
;;           训练到几百局就能看到鸟开始稳定穿缝隙；约 1000 局（浏览器里 20 秒）AI 能过
;;           十几根；练到 3000~5000 局（1~2 分钟）基本能一直飞下去。
;;
;; 「模型保存下来」：
;;   · 模型就是 Q 表（112 状态 × 2 动作 = 224 个定点整数，纯文本约 1.6KB），
;;     写到业务资源根的 data/flappy_q.txt；WASM 目标下 (Fs.write) 还会同步进
;;     浏览器 IndexedDB，刷新页面后自动读回（见 am_native_Fs.h 的持久化说明）。
;;   · 训练中每次刷新"历史最高分"都会存一份**最佳快照**（Q 表整体，224 个数），
;;     存盘与 AI 游玩都用这份快照——这样即使训练后期 Q 表在局部最优附近抖动，
;;     落盘的也始终是表现最好的那一版（等价于"保留最优 checkpoint"）。
;;   · 启动时自动 (Fs.read) 载入；载不上就从零开始（界面提示"未训练"）。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数；没有一元取负 → (- 0 x)；
;; 没有 let/list/map/append；块内不 define（define 只写在模块顶层或 lambda 体顶层）；
;; cond 每子句单个表达式；/ 与 mod 恒浮点；while 循环体不是独立词法区域。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native App)
(native Fs)
(native Ui)

(import FL "app/lib/flappy_core.scm")
(import Q  "app/lib/flappy_q.scm")
(import EX "app/lib/exit.scm")     ; 硬 A/ESC 与右上角「退出」按钮 → 全局退出确认框

;; ---- 节奏参数 -------------------------------------------------------------

(define TICK_MS 16)         ; 定时器周期（≈60 拍/s：与浏览器 rAF 同拍，训练吞吐最大）
(define PHYS_MS 17)         ; 一个物理步对应的毫秒数（60Hz；整数化避免累加误差）
(define PLAY_MAX_STEPS 6)   ; 一拍最多补几步物理（防止卡顿后跳变）
(define TRAIN_PER_TICK 150) ; 训练态每拍推进的**决策**数（≈ 一个时间片的量级，见下）
(define DEAD_HOLD_MS 900)   ; 撞死后停留多久再开新局（ms）
(define SAVE_EVERY_EP 20)   ; 训练中每多少局尝试存一次盘（有进步才真写）

;; TRAIN_PER_TICK 的依据：宿主给每个进程的默认时间片是 312500 条 VM 指令/帧，
;; 一个训练决策（3 个物理步 + 一次 Q 更新 + 一次观测）实测约 1.5k 条指令，
;; 故 ~200 决策就把一帧的时间片吃满。取 150 留出渲染与输入的余量，
;; 这样每拍（30 拍/s）推进 4500 个决策，1000 局（≈3~4 万决策）十秒内完成。

;; ---- 布局 ---------------------------------------------------------------
;; 画面 320x240：0..207 是飞行区（天空+管道），208..239 是地面条（放 5 个按钮）。
;; 顶部 0..29 叠一块 HUD 面板（分数/最佳/训练进度）——鸟撞到天花板（y<8）即死，
;; 所以 HUD 只在鸟"临死"那几帧挡住它一点，不影响操作。
(define SW (Gfx.width))
(define SH (Gfx.height))
(define GROUND_Y FL.PLAY_H)          ; 208：地面线（与内核一致）
(define BAR_Y 208)                   ; 按钮条起点
(define BAR_H (- SH BAR_Y))          ; 32
(define BTN_W (Math.floor (/ SW 5))); 64
(define BTN_H 24)
(define BTN_Y (+ BAR_Y 4))           ; 212
(define HUD_W 200)
(define HUD_H 30)

;; ---- 模式 ---------------------------------------------------------------

(define MODE_MANUAL 0)
(define MODE_AI 1)
(define MODE_TRAIN 2)
(define mode MODE_MANUAL)

;; ---- 运行状态 -----------------------------------------------------------

(define now_ms 0)
(define last_ms 0)
(define acc_ms 0)          ; 手动/AI 模式的物理步时间累加器
(define dead_at 0)         ; 本局死亡时刻（0 = 未死）
(define press_x 0)
(define press_y 0)
(define arm_btn -1)        ; 按下时命中的按钮号（-1 = 无）
(define toast "")          ; 一行提示（"已保存"之类），显示 1.2s
(define toast_at 0)
(define best_play 0)       ; 手动/AI 模式的历史最高分
(define last_save_ep 0)    ; 上次存盘时的训练局数
(define saved_best -1)     ; 上次存盘时的最佳分
(define loaded_ok 0)       ; 启动时是否读到了模型
(define timer_id 0)

;; 绘制用临时量
(define d_i 0) (define d_x 0) (define d_y 0) (define d_w 0) (define d_h 0)
(define d_g 0.0) (define d_t 0.0)
(define hud_s "")

(define MODEL_PATH "data/flappy_q.txt")

;; ============================================================================
;; 小工具
;; ============================================================================

;; 文本像素宽度（12px 字模：ASCII/拉丁 ≈ 7px、CJK 12px；同 svm/markdown 的度量）
(define tw
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

;; 带 1px 黑色阴影的文字（压在天空/管道上也看得清）
(define text_sh
  (lambda (s x y r g b)
    {
      (Gfx.draw_text s (+ x 1) (+ y 1) 0 0 0)
      (Gfx.draw_text s x y r g b)
    }))

;; 居中对齐画文字（cx = 水平中心，y = 行框上沿）
(define text_c
  (lambda (s cx y r g b)
    (Gfx.draw_text s (- cx (Math.floor (/ (tw s) 2))) y r g b)))

;; 三位定点格式（wasm32 的 %g 会打出 0.135000005 之类，见 am_native_Time.h 的精度说明）
;; 只用于把 ε 显示成 "0.35" 这种两位小数：整数化后自己拼。
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

;; 点是否在矩形内
(define in_rect
  (lambda (x y rx ry rw rh)
    (if (and (>= x rx) (< x (+ rx rw))) (if (and (>= y ry) (< y (+ ry rh))) 1 0) 0)))

;; ============================================================================
;; 绘制
;; ============================================================================

;; 一根管道（上下两段 + 管口）。x 可能为负（正在移出屏幕）→ 先裁剪再画，
;; 因为 Gfx 的坐标是 uint32，负值会被当成天文数字（见 svm 的同一处踩坑记录）。
(define draw_pipe
  (lambda (pxx pgg)
    {
      (set! d_g (- pgg FL.GAP_HH))          ; 缝隙上沿
      (set! d_t (+ pgg FL.GAP_HH))          ; 缝隙下沿
      ;; 上段
      (if (< pxx SW)
          (if (> (+ pxx FL.PIPE_W) 0)
              {
                (set! d_x (if (< pxx 0) 0 pxx))
                (set! d_w (- (+ pxx FL.PIPE_W) d_x))
                (Gfx.fill_rect d_x 0 d_w d_g 62 150 62)
                (Gfx.fill_rect d_x 0 6 d_g 46 122 48)          ; 左侧暗边
                (Gfx.fill_rect (- (+ d_x d_w) 6) 0 6 d_g 88 176 88)
              }
              #f)
          #f)
      (if (< pxx SW)
          (if (> (+ pxx FL.PIPE_W) 0)
              {
                (set! d_x (if (< pxx 0) 0 pxx))
                (set! d_w (- (+ pxx FL.PIPE_W) d_x))
                (Gfx.fill_rect d_x d_t d_w (- GROUND_Y d_t) 62 150 62)
                (Gfx.fill_rect d_x d_t 6 (- GROUND_Y d_t) 46 122 48)
                (Gfx.fill_rect (- (+ d_x d_w) 6) d_t 6 (- GROUND_Y d_t) 88 176 88)
              }
              #f)
          #f)
      ;; 管口（略宽一点，视觉上像"管子的沿"）
      (if (< (+ (+ pxx FL.PIPE_W) 4) 0)
          #f
          (if (> pxx SW)
              #f
              {
                (set! d_x (- pxx 2))
                (if (< d_x 0) (set! d_x 0) #f)
                (Gfx.fill_rect d_x (- d_g 10) (+ FL.PIPE_W 4) 10 88 176 88)
                (Gfx.fill_rect d_x d_t (+ FL.PIPE_W 4) 10 88 176 88)
              }))
    }))

;; 小鸟（程序化绘制：身体 + 翅膀 + 眼 + 喙）
(define draw_bird
  (lambda ()
    {
      (set! d_t FL.bird_y)
      (if (< d_t 0) (set! d_t 0) #f)
      (if (> d_t 240) (set! d_t 240) #f)
      (Gfx.fill_circle FL.BIRD_X (+ d_t 1) 9 214 168 34)        ; 身体
      (Gfx.fill_circle (- FL.BIRD_X 3) (+ d_t 3) 4 250 214 96)  ; 翅膀
      (Gfx.fill_circle (+ FL.BIRD_X 4) (- d_t 3) 3 255 255 255) ; 眼白
      (Gfx.fill_circle (+ FL.BIRD_X 5) (- d_t 3) 2 24 26 34)    ; 瞳孔
      (Gfx.fill_rect (+ FL.BIRD_X 7) (+ d_t 1) 7 4 240 132 40)  ; 喙
    }))

(define draw_ground
  (lambda ()
    {
      (Gfx.fill_rect 0 GROUND_Y SW BAR_H 96 158 72)
      (Gfx.fill_rect 0 GROUND_Y SW 3 132 196 104)
    }))

;; HUD 面板（在天空之上、管道/鸟之下画会看不清，故放在最后画，压在最上层）
(define draw_hud
  (lambda ()
    {
      (Gfx.fill_rect 0 0 HUD_W HUD_H 16 20 30)
      (Gfx.draw_rect 0 0 HUD_W HUD_H 44 52 70)
      (set! hud_s (String.concat "分数 " (String.atom_to_string FL.score)))
      (set! hud_s (String.concat hud_s "  最佳 "))
      (set! hud_s (String.concat hud_s (String.atom_to_string (if (> Q.best best_play) Q.best best_play))))
      (text_sh hud_s 6 3 235 240 250)
      (if (== mode MODE_TRAIN)
          {
            (set! hud_s "训练 局")
            (set! hud_s (String.concat hud_s (String.atom_to_string Q.episodes)))
            (set! hud_s (String.concat hud_s " ε"))
            (set! hud_s (String.concat hud_s (fmt2 Q.eps)))
            (text_sh hud_s 6 16 150 230 170)
          }
          (if (== mode MODE_AI)
              (if (> Q.episodes 0)
                  (text_sh "AI 自动飞行" 6 16 140 200 255)
                  (text_sh "AI：还没训练（先按「训练」）" 6 16 250 190 120))
              (if (> Q.episodes 0)
                  (text_sh "手动：点屏幕扇翅" 6 16 200 200 210)
                  (text_sh "未训练：先按「训练」" 6 16 250 190 120))))
    }))

;; 底部 5 个按钮：手动 / AI / 训练 / 存盘 / 清零
(define btn_label
  (lambda (i)
    (if (== i 0) "手动"
        (if (== i 1) "AI"
            (if (== i 2) "训练"
                (if (== i 3) "存盘" "清零"))))))

(define btn_active
  (lambda (i)
    (if (== i 0) (== mode MODE_MANUAL)
        (if (== i 1) (== mode MODE_AI)
            (if (== i 2) (== mode MODE_TRAIN) #f)))))

(define draw_buttons
  (lambda ()
    {
      (set! d_i 0)
      (while (< d_i 5)
        {
          (set! d_x (* d_i BTN_W))
          (if (btn_active d_i)
              {
                (Gfx.fill_rect (+ d_x 1) BTN_Y (- BTN_W 3) BTN_H 34 118 78)
                (Gfx.draw_rect (+ d_x 1) BTN_Y (- BTN_W 3) BTN_H 96 210 150)
              }
              {
                (Gfx.fill_rect (+ d_x 1) BTN_Y (- BTN_W 3) BTN_H 52 56 68)
                (Gfx.draw_rect (+ d_x 1) BTN_Y (- BTN_W 3) BTN_H 24 26 34)
              })
          (text_c (btn_label d_i) (+ d_x (Math.floor (/ BTN_W 2))) (+ BTN_Y 6) 232 236 244)
          (set! d_i (+ d_i 1))
        })
    }))

(define render
  (lambda ()
    {
      ;; 天空：四条横向色带（比逐像素渐变便宜得多）
      (Gfx.fill_rect 0 0 SW 60 92 164 226)
      (Gfx.fill_rect 0 60 SW 55 116 186 236)
      (Gfx.fill_rect 0 115 SW 50 148 206 244)
      (Gfx.fill_rect 0 165 SW (- GROUND_Y 165) 176 222 248)
      ;; 云（三团，装饰；位置随时间缓慢右移，视觉上不那么死板）
      (set! d_x (- 300 (Math.floor (* 20 (mod (/ now_ms 40) 12)))))
      (Gfx.fill_circle d_x 40 12 236 244 252)
      (Gfx.fill_circle (+ d_x 14) 40 9 236 244 252)
      (Gfx.fill_circle (- d_x 30) 74 10 236 244 252)
      ;; 管道 + 鸟
      (draw_pipe FL.px0 FL.pg0)
      (draw_pipe FL.px1 FL.pg1)
      (draw_pipe FL.px2 FL.pg2)
      (draw_bird)
      (draw_ground)
      (draw_buttons)
      (draw_hud)
      ;; 提示条（存盘反馈等）
      (if (> (String.length toast) 0)
          (if (< (- now_ms toast_at) 1200)
              (text_sh toast 220 16 250 230 140)
              (set! toast ""))
          #f)
      (EX.draw_button)
      (Gfx.refresh)
    }))

;; ============================================================================
;; 模型读写
;; ============================================================================

(define save_model
  (lambda ()
    {
      (define txt (Q.serialize))
      (if (Fs.write MODEL_PATH txt)
          {
            (set! saved_best Q.best)
            (set! toast "已保存")
            (set! toast_at now_ms)
            (display "[flappy] saved bytes=") (display (String.length txt))
            (display " ep=") (display Q.episodes)
            (display " best=") (display Q.best)
            (newline)
          }
          {
            (set! toast "保存失败")
            (set! toast_at now_ms)
          })
    }))

(define load_model
  (lambda ()
    {
      (set! loaded_ok (Q.load_text (Fs.read MODEL_PATH)))
      (display "[flappy] loaded ok=") (display loaded_ok)
      (display " ep=") (display Q.episodes)
      (display " best=") (display Q.best)
      (newline)
    }))

;; ============================================================================
;; 模式切换
;; ============================================================================

(define start_round
  (lambda ()
    {
      (FL.reset)
      (Q.reset_play)
      (set! dead_at 0)
      (set! acc_ms 0)
    }))

(define set_mode
  (lambda (m)
    {
      ;; 离开训练态：把最佳快照定为游玩策略，并落盘
      (if (and (== mode MODE_TRAIN) (not (== m MODE_TRAIN)))
          {
            (Q.use_best)
            (save_model)
          }
          #f)
      (set! mode m)
      (start_round)
      (display "[flappy] mode=") (display mode)
      (display " ep=") (display Q.episodes)
      (display " best=") (display Q.best)
      (newline)
    }))

;; ============================================================================
;; 一局结束（手动 / AI）
;; ============================================================================

(define on_play_dead
  (lambda ()
    {
      (if (> FL.score best_play) (set! best_play FL.score) #f)
      (set! dead_at now_ms)
      (display "[flappy] score=") (display FL.score)
      (display " best=") (display best_play)
      (display " mode=") (display mode)
      (newline)
    }))

;; 推进一个物理步（手动 / AI 共用）
(define play_step_once
  (lambda ()
    {
      (if (== mode MODE_AI)
          (Q.play_step)
          (FL.step))
      (if (== FL.dead 1)
          (if (== dead_at 0) (on_play_dead) #f)
          #f)
    }))

;; ============================================================================
;; 训练：一拍跑 TRAIN_PER_TICK 个决策
;; ============================================================================

(define train_tick
  (lambda ()
    {
      (define n 0)
      (while (< n TRAIN_PER_TICK)
        {
          (if (== FL.dead 1)
              {
                ;; 一局结束：内核里已经由 Q.train_step 结算（episodes/best/ε），
                ;; 这里只负责开新局（训练态用随机开局高度，见 flappy_core.scm）
                (FL.reset)
                (FL.randomize_start)
                (Q.reset_play)
                ;; 有进步 + 到节流间隔 → 存盘
                (if (>= (- Q.episodes last_save_ep) SAVE_EVERY_EP)
                    {
                      (set! last_save_ep Q.episodes)
                      (if (> Q.best saved_best) (save_model) #f)
                    }
                    #f)
                (if (== (mod Q.episodes 50) 0)
                    {
                      (display "[flappy] ep=") (display Q.episodes)
                      (display " best=") (display Q.best)
                      (display " eps=") (display (fmt2 Q.eps))
                      (display " steps=") (display Q.env_steps)
                      (newline)
                    }
                    #f)
              }
              #f)
          (Q.train_step)
          (set! n (+ n 1))
        })
    }))

;; ============================================================================
;; 输入
;; ============================================================================

;; 按钮动作
(define do_button
  (lambda (i)
    {
      (if (== i 0) (set_mode MODE_MANUAL) #f)
      (if (== i 1) (set_mode MODE_AI) #f)
      (if (== i 2)
          (if (== mode MODE_TRAIN)
              (set_mode MODE_AI)          ; 再点一次 = 停止训练、看它飞
              (set_mode MODE_TRAIN))
          #f)
      (if (== i 3) (save_model) #f)
      (if (== i 4)
          (if (Ui.confirm "清零学习成果？" "将删除已保存的模型并从头学起")
              {
                (Q.reset_model)
                (Q.reset_play)
                (set! best_play 0)
                (set! saved_best -1)
                (set! last_save_ep 0)
                (Fs.remove MODEL_PATH)
                (start_round)
                (display "[flappy] cleared") (newline)
              }
              #f)
          #f)
    }))

;; 命中哪个按钮（-1 = 不在按钮条内）
(define hit_button
  (lambda (x y)
    (if (< y BTN_Y) -1
        (if (>= y (+ BTN_Y BTN_H))
            -1
            (Math.floor (/ x BTN_W))))))

(define handle_input
  (lambda ()
    {
      ;; ---- 触屏：按下沿武装、松开沿结算（可滑出取消，同项目各 App 的范式）----
      (if (Touch.take_press)
          {
            (set! press_x (Touch.x))
            (set! press_y (Touch.y))
            (set! arm_btn (hit_button press_x press_y))
            ;; 飞行区的按下沿：手动模式立即扇翅（即时反馈）
            (if (== arm_btn -1)
                (if (== mode MODE_MANUAL)
                    (if (< press_y GROUND_Y) (FL.flap) #f)
                    #f)
                #f)
          }
          #f)
      (if (Touch.take_release)
          {
            (if (>= arm_btn 0)
                (if (== (in_rect (Touch.x) (Touch.y)
                                 (* arm_btn BTN_W) BTN_Y BTN_W BTN_H) 1)
                    (do_button arm_btn)
                    #f)
                #f)
            (set! arm_btn -1)
          }
          #f)
      ;; ---- 键盘：空格 / ↑ / W / Enter 扇翅（桌面与掌机都能玩）----
      (set! d_i (Key.take_press))
      (while (> d_i 0)
        {
          (if (or (== d_i 32) (or (== d_i 1) (or (== d_i 119) (== d_i 87))))
              (if (== mode MODE_MANUAL) (FL.flap) #f)
              #f)
          (set! d_i (Key.take_press))
        })
    }))

;; ============================================================================
;; 主循环
;; ============================================================================

(define tick
  (lambda ()
    {
      (EX.check)                     ; 硬 A/ESC / 右上角退出按钮（可能阻塞）
      (set! now_ms (System.timestamp))
      (handle_input)
      (if (== mode MODE_TRAIN)
          (train_tick)
          {
            ;; 物理步按**实测流逝时间**推进（定时器周期会有抖动，不能按拍计步）
            (set! acc_ms (+ acc_ms (- now_ms last_ms)))
            (if (> acc_ms 200) (set! acc_ms 200) #f)     ; 卡顿兜底
            (set! d_i 0)
            (while (and (>= acc_ms PHYS_MS) (< d_i PLAY_MAX_STEPS))
              {
                (set! acc_ms (- acc_ms PHYS_MS))
                (if (== dead_at 0) (play_step_once) #f)
                (set! d_i (+ d_i 1))
              })
            ;; 撞死后停留一会儿再开新局
            (if (> dead_at 0)
                (if (>= (- now_ms dead_at) DEAD_HOLD_MS) (start_round) #f)
                #f)
          })
      (set! last_ms now_ms)
      (render)
    }))

;; ============================================================================
;; 启动
;; ============================================================================

(load_model)
(FL.reset)
(Q.reset_play)
(set! last_ms (System.timestamp))
(set! now_ms last_ms)
(set! best_play Q.best)
(set! saved_best Q.best)
(display "[flappy] start ep=") (display Q.episodes)
(display " best=") (display Q.best) (newline)
(render)
(set! timer_id (System.set_interval TICK_MS tick))

;; ---- 退出：确认后先落盘，再销毁（见 app/lib/exit.scm）----
(EX.install_with render)
(EX.on_exit save_model)
