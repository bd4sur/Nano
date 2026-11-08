;; ============================================================================
;; Flappy Bird 强化学习智能体（Q）—— 表格型 Q 学习（tabular Q-learning）
;;
;; 为什么是"表格 + 离散观测"而不是神经网络：
;;   本 App 要在一个虚拟机时间片（每帧 312500 条 VM 指令）里推进尽可能多的环境步数，
;;   才能在一两分钟内完成几百局的训练。一次 get_item/set_item!（本地库调用）就要
;;   一两百条指令，而一个小 MLP 的前向 + 反向动辄上千条指令/步——比表格慢 1~2 个
;;   数量级，浏览器里根本跑不出可见的进步。表格型 Q 学习每个决策只需 5 次列表访问，
;;   状态空间 112×2 = 224 个参数（模型文本约 1.6KB），1000 局左右 AI 就能过十几根
;;   管道，3000~5000 局基本能一直飞。
;;
;; 算法（逐条写清楚，便于核对）：
;;   状态  s = FL.obs_index：112 个离散状态（dx 2 档 × dy 8 档 × vy 7 档）
;;   动作  a ∈ {0 不扇翅, 1 扇翅}
;;   原始奖励  r = R_ALIVE（存活一个**决策**） + R_PASS（一个决策里穿过管道时叠加）
;;                 R_DEATH（撞死那个决策，覆盖式地取代 R_ALIVE）
;;   一个"决策" = 连续 ACTION_REPEAT 个物理步（见下方 ACTION_REPEAT 的说明）
;;   奖励塑形  r' = r + γ·Φ(s') − Φ(s)，Φ(s) = −|dy| / SHAPE_DEN
;;        —— 位势塑形（potential-based shaping, Ng et al. 1999）的一个特例：
;;        dy 是"鸟中心 − 目标缝隙中心"，Φ 在缝隙中心处取最大值 0、偏离越远越负。
;;        为什么必须有它：只给"活着给分 / 撞死扣分"时，智能体在早期探索里学到的最优
;;        局部策略是**贴着天花板不停扇翅**（那样活得更久），而贴着天花板过不了缝隙。
;;        塑形项把"离缝隙中心多远"变成一个**逐步可达的中间目标**，于是从贴顶的
;;        状态也能拿到"往下走更好"的梯度；因为它是位势差，最优策略不被改变。
;;   更新（Q-learning，离线策略、单步自举）：
;;       Q(s,a) ← Q(s,a) + α·[ r' + γ·max_{a'} Q(s',a') − Q(s,a) ]   （未终止）
;;       Q(s,a) ← Q(s,a) + α·[ r' − Q(s,a) ]                         （终止）
;;   行为策略：ε-贪心（ε 从 1.0 按局衰减到 EPS_MIN，先探索后利用）
;;   推理策略：贪心（并列取"扇翅"——悬停比下坠安全）
;;
;; 本模块**持有**单步逻辑（train_step / play_step），app/flappy.scm 与
;; tools/e2e 的回归脚本都调它——"应用里跑的"和"测的"因此是同一段代码。
;;
;; 模型持久化：**纯文本**（风格同 app/lib/g2048_core.scm 的存档）。
;;   文本里全部是整数（Q 值乘 QSCALE 后四舍五入取整），读写两端都不涉及浮点格式化
;;   ——wasm32 宿主的 am_float_t 只有约 19 位有效位（见 am_native_Time.h），"%g"
;;   会打出 0.123000003 之类，而整数在 27 位无符号范围内逐位精确。
;;   格式（空格/换行分隔，首段是元信息，其后是 Q 值）：
;;       1 <NSTATES> <NQ> <eps*1000> <episodes> <best> <环境步数> q0 q1 ... q(NQ-1)
;;   版本号不是 1 / 状态数不符 / 整数个数不足 → 视为损坏，整份模型不采用。
;;
;; 解释器约束（manual.md）：算术与比较恒二参数；无一元取负 → (- 0 x)；
;; 无 let/list/append/map；块内不 define（define 只在 lambda 体顶层）；
;; / 恒浮点（定点换算一律 Math.round 收口）。
;; ============================================================================

(native Math)
(native String)

(import FL "app/lib/flappy_core.scm")

;; ---- 超参数 ---------------------------------------------------------------

;; 状态数由内核的档数常量**算出来**（不写死：改档数只需改 flappy_core.scm 一处，
;; 模型文件头里的 NSTATES 也随之变化，旧的模型文件会被判为"状态数不符"而拒绝）
(define NSTATES (* FL.DX_N (* FL.DY_N FL.VY_N)))   ; 2 × 8 × 7 = 112
(define NA 2)                                      ; 动作数
(define NQ (* NSTATES NA))                         ; 224
;; 动作持续步数（action repeat / frame-skip）：一个决策 = 连续 ACTION_REPEAT 个物理步，
;; 窗口内动作保持不变。**为什么必须有它**（实测教训，别删）：
;;   物理步长 1/60 s、扇翅上升 25px。若逐物理步决策，"随机动作"等价于"每秒 30 次
;;   扇翅"——鸟必定一路顶到天花板并死在第一根管道上（实测：ε=1 的 400 局里没有
;;   一局过了第一根管，且全部恰好 129 步撞在同一位置）。原因是一次扇翅的上升
;;   要 12 个连续"不扇翅"才能抵消，而 ε-贪心根本凑不出这 12 连击：中段高度永远
;;   探索不到，Q 表就锁死在"贴顶狂扇"的局部最优。
;;   改成"一个决策持续 3 步"（出厂值；6 步也能学，但控制粒度更粗）后，一次
;;   "不扇翅"就让鸟下落 10~30px，与一次扇翅的 25px 上升同量级，随机探索才会在
;;   整个高度带上做随机游走。这也正是 DQN 等深度强化学习里的标准做法（frame skip）。
(define ACTION_REPEAT 3)
(define ALPHA 0.10)         ; 学习率
(define GAMMA 0.99)         ; 折扣因子（按**决策**计）
(define EPS_START 1.0)      ; 初始探索率
(define EPS_MIN 0.05)       ; 探索率下限
(define EPS_DECAY 0.998)    ; 每局衰减（≈500 局后探索率降到 0.37）
(define R_ALIVE 0.2)        ; 存活一个决策的奖励
(define R_PASS 10.0)        ; 穿过一根管道的奖励（训练的主信号）
(define R_DEATH (- 0 5.0))  ; 撞死那一个决策的奖励
(define SHAPE_DEN 48.0)     ; 位势塑形尺度：Φ = -|dy| / SHAPE_DEN（|dy| 最大约 64）
(define QSCALE 1000)        ; 模型里的定点比例（Q 值 ×1000 存整数）

;; ---- 模型状态 -------------------------------------------------------------

;; 造一张 n 个 0 的新表（'() 是模块级静态槽，直接 push 会改到字面量本身）
(define mk_list
  (lambda (n)
    {
      (define lst (cdr '(0)))
      (define i 0)
      (while (< i n)
        {
          (push lst 0)
          (set! i (+ i 1))
        })
      lst
    }))

(define q (mk_list NQ))     ; Q 表（扁平：下标 = s*2 + a）
(define eps EPS_START)      ; 当前探索率
(define episodes 0)         ; 已完成的训练局数
(define best 0)             ; 训练中出现过的最高分
(define env_steps 0)        ; 累计环境步数（训练进度的一个量）
(define win_sum 0)          ; 当前统计窗口内的得分和（供"窗口均分"诊断行）
(define win_n 0)            ; 当前统计窗口内的局数
(define loaded 0)           ; 是否成功载入过模型

;; ---- 最佳快照（checkpoint）------------------------------------------------
;; 训练时 Q 表会在局部最优附近抖动：某一阶段可能连续几千局都飞不好（实测有这种
;; 反复）。所以每当**一局训练**打出了历史最高分，就把整张 Q 表复制一份存下来；
;; 存盘与 AI 游玩都用这份快照，落盘的因此永远是表现最好的那一版。
;; 快照只有 NQ 个数（224），复制一次的成本可以忽略。
(define best_q (mk_list NQ))  ; 最佳快照
(define snap_ok 0)            ; 是否已有快照
(define pq q)                 ; 游玩策略读的表（= best_q 或 q 本身，见 use_best）

;; 单步逻辑的中间量（模块级 → 不占堆、不进闭包）
(define ts 0) (define ta 0) (define ts2 0)
(define tr 0.0) (define tp0 0.0) (define tp1 0.0)
(define tk 0) (define tdead 0)
(define pskip 0) (define pa 0)

;; 序列化/反序列化的中间量
(define s_buf "")
(define s_chunk "")
(define s_i 0)
(define p_i 0)
(define p_n 0)
(define p_c 0)
(define p_cur 0)
(define p_neg 0)
(define p_in 0)
(define p_ok 0)
(define p_vals (cdr '(0)))

;; ---- 基本访问 -------------------------------------------------------------

(define qget
  (lambda (s a)
    (get_item q (+ (* s 2) a))))

(define qset
  (lambda (s a v)
    (set_item! q (+ (* s 2) a) v)))

;; 贪心动作：值大者为 1(扇翅)；相等时取 1（悬停比下坠安全）
;; 贪心动作：取 Q 值大的那个动作（并列时取 1=扇翅：悬停比下坠安全）
;; 注意别把比较写反：这里必须比较"扇翅(q1)"与"不扇翅(q0)"，写成 (>= q0 q1)
;; 就变成"故意选更差的动作"——训练能跑但要浪费大量局数才发现。
(define greedy
  (lambda (s)
    (if (>= (qget s 1) (qget s 0)) 1 0)))

;; ε-贪心动作
(define act_eps
  (lambda (s)
    (if (< (Math.random) eps)
        (if (< (Math.random) 0.5) 1 0)
        (greedy s))))

;; ---- 奖励塑形的位势 -------------------------------------------------------
;; 入参是 dy 档位（0..7，档宽 16，中心 -56..56）。用档位中心还原近似的 dy，
;; 于是塑形量的计算不依赖浮点 dy 本身（少一次除法、也少一份状态）。
;;   j → 该档中心 c = DY_LO + DY_W*j + DY_W/2 → Φ = -|c| / SHAPE_DEN
(define phi
  (lambda (j)
    {
      (define c (+ (+ FL.DY_LO (* FL.DY_W j)) (/ FL.DY_W 2)))
      (- 0 (/ (Math.abs c) SHAPE_DEN))
    }))

;; ---- Q 更新 ---------------------------------------------------------------

(define learn
  (lambda (s a r s2 done)
    {
      (define tgt r)
      (define qo 0.0)
      (if (== done 0)
          (if (> (qget s2 0) (qget s2 1))
              (set! tgt (+ r (* GAMMA (qget s2 0))))
              (set! tgt (+ r (* GAMMA (qget s2 1)))))
          #f)
      (set! qo (qget s a))
      (qset s a (+ qo (* ALPHA (- tgt qo))))
      (set! env_steps (+ env_steps 1))
    }))

;; 把当前 Q 表复制进最佳快照
(define snap
  (lambda ()
    {
      (set! s_i 0)
      (while (< s_i NQ)
        {
          (set_item! best_q s_i (get_item q s_i))
          (set! s_i (+ s_i 1))
        })
      (set! snap_ok 1)
      (set! pq best_q)
    }))

;; 让游玩策略用最佳快照（没有快照就用当前 Q 表）
(define use_best
  (lambda ()
    (if (== snap_ok 1) (set! pq best_q) (set! pq q))))

;; ---- 一局结束：统计 + 探索率衰减 -----------------------------------------

(define finish_episode
  (lambda (sc)
    {
      (set! episodes (+ episodes 1))
      (set! win_sum (+ win_sum sc))
      (set! win_n (+ win_n 1))
      (if (> sc best)
          {
            (set! best sc)
            (snap)
          }
          #f)
      (set! eps (* eps EPS_DECAY))
      (if (< eps EPS_MIN) (set! eps EPS_MIN) #f)
    }))

;; 取走当前统计窗口的 (局数, 得分和, 均分)，并把窗口清零（供诊断行使用）
(define take_window
  (lambda ()
    {
      (define n win_n)
      (define sum win_sum)
      (define avg 0.0)
      (if (> n 0) (set! avg (/ sum n)) #f)
      (set! win_n 0)
      (set! win_sum 0)
      (list3 n sum avg)
    }))

;; 三元组（语言里没有 list 过程，且列表是整体容器 → 用 push 拼一张新表）
(define list3
  (lambda (a b c)
    {
      (define l (cdr '(0)))
      (push l a)
      (push l b)
      (push l c)
      l
    }))

;; 清空学习成果（界面上的「清零」按钮）
(define reset_model
  (lambda ()
    {
      (set! q (mk_list NQ))
      (set! best_q (mk_list NQ))
      (set! snap_ok 0)
      (set! pq q)
      (set! eps EPS_START)
      (set! episodes 0)
      (set! best 0)
      (set! env_steps 0)
      (set! win_sum 0)
      (set! win_n 0)
      (set! loaded 0)
    }))

;; ---- 单步：训练（ε-贪心 + 学习）与游玩（贪心，不学习）--------------------

;; 训练一个**决策**：观测 → ε-贪心动作 → 持续 ACTION_REPEAT 个物理步（动作保持）
;; → 奖励（按决策计）→ Q 更新（终止则结算本局）。
(define train_step
  (lambda ()
    {
      (set! ts (FL.obs_index))
      (set! tp0 (phi FL.oe))
      (set! ta (act_eps ts))
      (set! tr 0.0)
      (set! tk 0)
      (set! tdead 0)
      (while (and (< tk ACTION_REPEAT) (== tdead 0))
        {
          (if (== ta 1) (FL.flap) #f)
          (FL.step)
          (if (== FL.passed_now 1) (set! tr (+ tr R_PASS)) #f)
          (if (== FL.dead 1) (set! tdead 1) #f)
          (set! tk (+ tk 1))
        })
      (if (== tdead 1)
          {
            (learn ts ta (+ tr R_DEATH) 0 1)
            (finish_episode FL.score)
          }
          {
            (set! ts2 (FL.obs_index))
            (set! tp1 (phi FL.oe))
            (learn ts ta (+ (+ (+ tr R_ALIVE) (* GAMMA tp1)) (- 0 tp0)) ts2 0)
          })
    }))

;; 游玩策略（读最佳快照；训练中的 greedy 仍读正在学习的 q 表）
(define pget
  (lambda (s a)
    (get_item pq (+ (* s 2) a))))

(define greedy_p
  (lambda (s)
    (if (>= (pget s 1) (pget s 0)) 1 0)))

;; 游玩一个**物理步**（AI 自动 / 也用于贪心评估）：每 ACTION_REPEAT 步重算一次贪心
;; 动作，窗口内保持——与训练时的决策粒度完全一致，否则学到的策略在游玩时会"换脾气"。
(define play_step
  (lambda ()
    {
      (if (== pskip 0)
          {
            (set! pskip ACTION_REPEAT)
            (set! pa (greedy_p (FL.obs_index)))
          }
          #f)
      (if (== pa 1) (FL.flap) #f)
      (FL.step)
      (set! pskip (- pskip 1))
    }))

;; 复位游玩窗口（开新局时调用，避免沿用上一局的"剩余步数"）
(define reset_play
  (lambda ()
    (set! pskip 0)))

;; ---- 序列化 ---------------------------------------------------------------

;; 整数 → 字符串（Q 值与计数都是整数；用 atom_to_string 的十进制表示）
(define itoa
  (lambda (n) (String.atom_to_string n)))

;; 把 Q 值按 QSCALE 定点成整数
(define qfix
  (lambda (v) (Math.round (* v QSCALE))))

;; 生成模型文本（分块拼接：逐个 concat 会让 672 个数产生 O(n²) 的拷贝）
(define serialize
  (lambda ()
    {
      (set! s_buf "")
      (set! s_buf (String.concat s_buf "1 "))
      (set! s_buf (String.concat s_buf (itoa NSTATES)))
      (set! s_buf (String.concat s_buf " "))
      (set! s_buf (String.concat s_buf (itoa NQ)))
      (set! s_buf (String.concat s_buf " "))
      (set! s_buf (String.concat s_buf (itoa (qfix eps))))
      (set! s_buf (String.concat s_buf " "))
      (set! s_buf (String.concat s_buf (itoa episodes)))
      (set! s_buf (String.concat s_buf " "))
      (set! s_buf (String.concat s_buf (itoa best)))
      (set! s_buf (String.concat s_buf " "))
      (set! s_buf (String.concat s_buf (itoa env_steps)))
      (set! s_buf (String.concat s_buf "\n"))
      (set! s_i 0)
      (set! s_chunk "")
      (while (< s_i NQ)
        {
          (set! s_chunk (String.concat s_chunk (itoa (qfix (get_item pq s_i)))))
          (set! s_chunk (String.concat s_chunk " "))
          (if (>= (String.length s_chunk) 96)
              {
                (set! s_buf (String.concat s_buf s_chunk))
                (set! s_chunk "")
              }
              #f)
          (set! s_i (+ s_i 1))
        })
      (set! s_buf (String.concat s_buf s_chunk))
      s_buf
    }))

;; ---- 反序列化 -------------------------------------------------------------

;; 把文本里的全部整数扫进 p_vals（逐码点，支持负号；非数字字符即分隔符）。
;; 语言里没有 split/parse-list，这样一遍扫完不造中间字符串。
(define scan_ints
  (lambda (s)
    {
      (while (> (length p_vals) 0) (pop p_vals))
      (set! p_n (String.length s))
      (set! p_i 0)
      (set! p_cur 0)
      (set! p_neg 0)
      (set! p_in 0)
      (while (< p_i p_n)
        {
          (set! p_c (String.charCodeAt p_i s))
          (if (and (>= p_c 48) (<= p_c 57))
              {
                (set! p_cur (+ (* p_cur 10) (- p_c 48)))
                (set! p_in 1)
              }
              (if (== p_c 45)
                  (set! p_neg 1)
                  (if (== p_in 1)
                      {
                        (push p_vals (if (== p_neg 1) (- 0 p_cur) p_cur))
                        (set! p_cur 0)
                        (set! p_neg 0)
                        (set! p_in 0)
                      }
                      (set! p_neg 0))))
          (set! p_i (+ p_i 1))
        })
      (if (== p_in 1)
          (push p_vals (if (== p_neg 1) (- 0 p_cur) p_cur))
          #f)
      (length p_vals)
    }))

;; 载入模型文本。返回 1 = 成功（q/eps/episodes/best 全部就位），0 = 无文本或损坏。
;; 校验：版本号 = 1、NSTATES 与当前内核一致、整数个数 ≥ 7 + NQ。
(define load_text
  (lambda (s)
    {
      (set! p_ok 0)
      (if (null? s)
          #f
          {
            (scan_ints s)
            (if (>= (length p_vals) (+ 7 NQ))
                (if (== (get_item p_vals 0) 1)
                    (if (== (get_item p_vals 1) NSTATES)
                        {
                          (set! p_ok 1)
                          (set! eps (/ (get_item p_vals 3) QSCALE))
                          (if (< eps EPS_MIN) (set! eps EPS_MIN) #f)
                          (if (> eps 1.0) (set! eps 1.0) #f)
                          (set! episodes (get_item p_vals 4))
                          (set! best (get_item p_vals 5))
                          (set! env_steps (get_item p_vals 6))
                          ;; Q 表整体换成新表（不复用旧表：旧表可能还被闭包引用）
                          (set! q (mk_list NQ))
                          (set! s_i 0)
                          (while (< s_i NQ)
                            {
                              (set_item! q s_i (/ (get_item p_vals (+ 7 s_i)) QSCALE))
                              (set! s_i (+ s_i 1))
                            })
                          ;; 载入的模型同时作为最佳快照（游玩/存盘都用它）
                          (set! best_q (mk_list NQ))
                          (set! s_i 0)
                          (while (< s_i NQ)
                            {
                              (set_item! best_q s_i (get_item q s_i))
                              (set! s_i (+ s_i 1))
                            })
                          (set! snap_ok 1)
                          (set! pq best_q)
                          (set! loaded 1)
                        }
                        #f)
                    #f)
                #f)
          })
      p_ok
    }))
