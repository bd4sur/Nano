;; ============================================================================
;; Flappy Bird 逻辑内核（FL）—— 纯逻辑，不依赖 Gfx / Touch / Fs，可脱离界面单独回归
;;
;; 分工（同 app/lib/tetris_core.scm / g2048_core.scm 的惯例）：
;;   · 本文件 —— 鸟的竖直物理、三根管道的滚动/回收、碰撞、计分、随机缝隙、
;;              以及给强化学习用的**离散观测索引**（obs_index）；
;;   · app/lib/flappy_q.scm —— 表格型 Q 学习智能体（动作选择/更新/模型序列化）；
;;   · app/flappy.scm —— 布局、绘制、触屏/键盘、节拍、训练调度、文件读写。
;;
;; 时间模型：**固定步长**。1 步 = 1/60 s，界面层按实际流逝的毫秒数累计出步数
;; （训练态则直接连续跑多步）。固定步长让物理与学习都可复现，也让"观测离散化"
;; 的边界不会因 dt 抖动而漂移。
;;
;; 为什么管道用三个**标量变量**而不是列表：
;;   训练态每拍要跑几百步物理，而 (get_item)/(set_item!) 是本地库调用（每次约
;;   一两百条 VM 指令），列表版每步要多花 15 次以上调用——三个标量的版本把
;;   整步成本压到纯算术+比较的量级（实测差 3~4 倍）。管道固定三根、且回收时
;;   按 +3*SPACING 平移，正好让"始终递增"的不变式得以手工维护：
;;       px0 < px1 < px2   （每步同时左移，回收者跳到最右，次序不变）
;;
;; 解释器约束（manual.md，逐条遵守）：
;;   · 算术与比较恒二参数；没有一元取负 → 写 (- 0 7.0)；
;;   · 没有 let / list / append / vector；模块级 define + set! 携带局部量；
;;   · 块内不 define（define 只写在模块顶层或 lambda 体顶层）；
;;   · cond 每个子句只接受单个表达式；if 的分支数写全；
;;   · / 与 mod 恒返回浮点；Math.* 结果为 NaN 时返回 #null（本项目用不到）。
;; ============================================================================

(native Math)
(native String)

;; ---- 几何与物理常量（单位：像素、步）--------------------------------------

(define BIRD_X 64)          ; 鸟的水平位置（固定；管道向它滚来）
(define BIRD_HW 9)          ; 鸟碰撞盒半宽
(define BIRD_HH 7)          ; 鸟碰撞盒半高
(define PLAY_H 208)         ; 地面线：鸟的下沿触到即死（界面层的地面条从这里开始）
(define CEIL_Y 8)           ; 上边界（鸟中心的最高位置）
(define CEIL_DEAD 1)        ; 1 = 撞顶即死（见 step 里的注释：不这样做时"贴顶狂扇"
                            ; 是 Q 学习的一个强局部最优，实测几百局都过不了第一根管道）

;; 扇翅动力学：一次扇翅给 vy = FLAP_VY，随后按 GRAV 衰减，因此**单次扇翅的
;; 上升高度 = FLAP_VY²/(2·GRAV) = 25px**。这个值必须明显小于缝隙的可用摆动带
;; （GAP_HH − BIRD_HH = 35px），否则鸟只能在两个极端之间跳、根本稳不住高度
;;（实测 GRAV=0.389 / FLAP=-7 时上升 63px ≈ 整个缝隙，手写控制器也过不去）。
(define GRAV 0.35)          ; 每步速度增量（≈1260 px/s²）
(define FLAP_VY (- 0 4.2))  ; 扇翅瞬时竖直速度（≈-252 px/s，向上为负）
(define VY_MAX 8.0)         ; 终端下落速度（≈480 px/s）

(define PIPE_W 46)          ; 管道宽
(define GAP_H 120)           ; 缝隙高（鸟高 14，可用摆动带 ±53px）
(define GAP_HH 60)          ; 缝隙半高
(define SPACING 170)        ; 相邻管道间距（≈1.9 s 一根）
(define SPEED 1.5)          ; 每步左移量（=90 px/s）

(define GAP_MIN 72)         ; 缝隙中心下限
(define GAP_STEP 8)         ; 缝隙中心**吸附到 8px 栅格**：缝隙位置只有 5 档，
(define GAP_SLOTS 5)       ; 于是"缝隙中心"这个观测量是离散的，学习目标不会漂

;; ---- 状态 -----------------------------------------------------------------

(define bird_y 100.0)       ; 鸟中心 y
(define bird_vy 0.0)        ; 鸟竖直速度（px/步）
(define px0 280) (define px1 450) (define px2 620)   ; 三根管道左缘 x（递增）
(define pg0 100.0) (define pg1 100.0) (define pg2 100.0) ; 缝隙中心 y
(define pp0 0) (define pp1 0) (define pp2 0)         ; 本根是否已计过分（0/1）

(define score 0)            ; 当前局得分（穿过一根管道 +1）
(define dead 0)             ; 本局是否已结束
(define steps 0)            ; 本局已推进的步数
(define passed_now 0)       ; **本步**是否刚穿过一根管道（给奖励用，每步清零）

;; 观测离散化的中间量（模块级临时量，避免在块内 define）
(define ox 0) (define og 0) (define od 0) (define oe 0) (define ov 0) (define ob 0)

;; ---- 随机缝隙 -------------------------------------------------------------

;; 缝隙中心：GAP_MIN + 8*k，k ∈ [0, GAP_SLOTS)。Math.floor 返回浮点，故结果是浮点。
(define new_gap
  (lambda ()
    (+ GAP_MIN (* GAP_STEP (Math.floor (* (Math.random) GAP_SLOTS))))))

;; ---- 复位（开新局）--------------------------------------------------------

(define reset
  (lambda ()
    {
      (set! bird_y 100.0)
      (set! bird_vy 0.0)
      (set! px0 280)
      (set! px1 (+ 280 SPACING))
      (set! px2 (+ 280 (* 2 SPACING)))
      (set! pg0 (new_gap))
      (set! pg1 (new_gap))
      (set! pg2 (new_gap))
      (set! pp0 0)
      (set! pp1 0)
      (set! pp2 0)
      (set! score 0)
      (set! dead 0)
      (set! steps 0)
      (set! passed_now 0)
    }))

;; 训练用的**随机开局高度**：只改鸟的起始 y（不动管道/计分）。
;; 为什么训练要这么做：ε-贪心是"逐帧"随机化动作，而一次扇翅会带来 25px 的上升，
;; 于是"随机动作"等价于"高频扇翅"——鸟会一路顶到天花板并贴在那里，中段高度几乎
;; 永远探索不到（实测：贴顶的死循环把 Q 表困在局部最优，几百局都过不了第一根
;; 管道）。把开局高度随机撒在整个可用带内，等于每局都强制从不同高度开始，
;; 探索覆盖立刻打开。**手动/AI 游玩不用它**（固定 100，手感一致）。
(define randomize_start
  (lambda ()
    {
      (set! bird_y (+ 40 (* (Math.random) 120)))
      (set! bird_vy 0.0)
    }))

;; ---- 扇翅 -----------------------------------------------------------------

(define flap
  (lambda ()
    (if (== dead 0) (set! bird_vy FLAP_VY) #f)))

;; ---- 碰撞（鸟碰撞盒 vs 一根管道的上下两段）--------------------------------
;;   水平重叠 且（鸟上沿高于缝隙上沿 或 鸟下沿低于缝隙下沿）→ 撞。
(define hit_pipe
  (lambda (pxx pgg)
    (if (and (> (+ pxx PIPE_W) (- BIRD_X BIRD_HW))
             (< pxx (+ BIRD_X BIRD_HW)))
        (if (or (< (- bird_y BIRD_HH) (- pgg GAP_HH))
                (> (+ bird_y BIRD_HH) (+ pgg GAP_HH)))
            1
            0)
        0)))

;; ---- 推进一 phys 步 -------------------------------------------------------

(define step
  (lambda ()
    (if (== dead 1)
        #f
        {
          (set! passed_now 0)
          ;; 竖直物理
          (set! bird_vy (+ bird_vy GRAV))
          (if (> bird_vy VY_MAX) (set! bird_vy VY_MAX) #f)
          (set! bird_y (+ bird_y bird_vy))
          (if (< bird_y CEIL_Y)
              (if (== CEIL_DEAD 1)
                  {
                    (set! bird_y CEIL_Y)
                    (set! dead 1)
                  }
                  { (set! bird_y CEIL_Y) (set! bird_vy 0.0) })
              #f)
          ;; 管道滚动
          (set! px0 (- px0 SPEED))
          (set! px1 (- px1 SPEED))
          (set! px2 (- px2 SPEED))
          ;; 穿过计分（右缘越过鸟的碰撞盒左沿）
          (if (and (== pp0 0) (< (+ px0 PIPE_W) (- BIRD_X BIRD_HW)))
              { (set! pp0 1) (set! score (+ score 1)) (set! passed_now 1) }
              #f)
          (if (and (== pp1 0) (< (+ px1 PIPE_W) (- BIRD_X BIRD_HW)))
              { (set! pp1 1) (set! score (+ score 1)) (set! passed_now 1) }
              #f)
          (if (and (== pp2 0) (< (+ px2 PIPE_W) (- BIRD_X BIRD_HW)))
              { (set! pp2 1) (set! score (+ score 1)) (set! passed_now 1) }
              #f)
          ;; 回收（整体右移 3 个间距，新缝隙、计分标记清 0）
          (if (< (+ px0 PIPE_W) 0)
              {
                (set! px0 (+ px0 (* 3 SPACING)))
                (set! pg0 (new_gap))
                (set! pp0 0)
              }
              #f)
          (if (< (+ px1 PIPE_W) 0)
              {
                (set! px1 (+ px1 (* 3 SPACING)))
                (set! pg1 (new_gap))
                (set! pp1 0)
              }
              #f)
          (if (< (+ px2 PIPE_W) 0)
              {
                (set! px2 (+ px2 (* 3 SPACING)))
                (set! pg2 (new_gap))
                (set! pp2 0)
              }
              #f)
          ;; 碰撞
          (if (>= (+ bird_y BIRD_HH) PLAY_H) (set! dead 1) #f)
          (if (== (hit_pipe px0 pg0) 1) (set! dead 1) #f)
          (if (== (hit_pipe px1 pg1) 1) (set! dead 1) #f)
          (if (== (hit_pipe px2 pg2) 1) (set! dead 1) #f)
          (set! steps (+ steps 1))
        })))

;; ---- 观测：把连续状态离散成整数索引 -------------------------------
;;
;; 三个观测量（全部是**相对量**，与屏幕分辨率无关）与它们的等宽分档：
;;   dx = 目标管道左缘 − 鸟 x     → floor((dx − DX_LO)/DX_W) 截断到 [0,DX_N)
;;   dy = 鸟中心 − 目标缝隙中心    → 同理（DY_*）
;;   vy = 鸟竖直速度              → 同理（VY_*）
;; 索引 index = (dx*DY_N + dy)*VY_N + vy ∈ [0, DX_N*DY_N*VY_N) = [0,112)
;;
;; 档位用 Math.floor 现算（而不是嵌套 if）：档宽/档数因此是**常量**，调参只改上面
;; 那几个数字；Math.floor 返回浮点下标，但 get_item/set_item! 会四舍五入取整，
;; 且这些下标 < 2^19 在 wasm32 的单精度下逐位精确（见 am_native_Time.h 的精度说明）。
(define DX_LO (- 0 15))    ; dx 档下限（像素）
;; dx 只分"近/远"两档（档宽 90px ≈ 0.6 s）：实测把它切细到 6 档并不能提升上限，
;; 却让状态数 ×3、每格被访问的次数 ÷3，学习明显变慢（见 app/lib/flappy_q.scm 的说明）。
(define DX_W 90)          ; dx 档宽（像素）
(define DX_N 2)           ; dx 档数
(define DY_LO (- 0 48))    ; dy 档下限（像素）
(define DY_W 16)          ; dy 档宽（像素）
(define DY_N 8)           ; dy 档数
(define VY_LO (- 0 5))     ; vy 档下限（px/步）
(define VY_W 2)           ; vy 档宽
(define VY_N 7)           ; vy 档数

;; 通用分档：clamp(floor((v − lo)/w), 0, n−1)
(define bin_of
  (lambda (v lo w n)
    {
      (set! ob (Math.floor (/ (- v lo) w)))
      (if (< ob 0) 0 (if (>= ob n) (- n 1) ob))
    }))

;; 目标管道 = 第一根"右缘仍在鸟的碰撞盒右沿以右（留 6px 余量）"的管道。
(define pick_target
  (lambda ()
    (if (> (+ px0 PIPE_W) (- BIRD_X 6))
        { (set! ox px0) (set! og pg0) }
        (if (> (+ px1 PIPE_W) (- BIRD_X 6))
            { (set! ox px1) (set! og pg1) }
            { (set! ox px2) (set! og pg2) }))))

;; 当前观测索引（整数），并把三个档位留在 od/oe/ov 里（塑形项只用到 oe）
(define obs_index
  (lambda ()
    {
      (pick_target)
      (set! od (bin_of (- ox BIRD_X) DX_LO DX_W DX_N))
      (set! oe (bin_of (- bird_y og) DY_LO DY_W DY_N))
      (set! ov (bin_of bird_vy VY_LO VY_W VY_N))
      (+ (+ (* od DY_N) (* oe VY_N)) ov)
    }))

;; ---- 只读查询（界面层绘制用）--------------------------------------------

(define px_at
  (lambda (i)
    (if (== i 0) px0 (if (== i 1) px1 px2))))

(define pg_at
  (lambda (i)
    (if (== i 0) pg0 (if (== i 1) pg1 pg2))))
