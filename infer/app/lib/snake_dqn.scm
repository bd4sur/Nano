;; ============================================================================
;; 贪吃蛇 DQN 智能体（Snake DQN）—— app/snake.scm 的「AI / 训练」模式
;;
;; 这里实现的是一个**完整的 DQN**（Mnih et al. 2015 的 Deep Q-Network），全部跑在
;; 灵机解释器里（没有 C 侧推理，也不依赖任何神经网络本地库）：
;;
;;   Q 网络     MLP  NI -> NH -> 3（tanh 隐层、线性输出；权重扁平行主序）
;;   动作       0 = 直行、1 = 右转、2 = 左转（相对动作，网络因此不必学"绝对朝向"）
;;   目标网络   Q_target，每 TARGET_EVERY 个环境步硬同步一次（抑制自举发散）
;;   经验回放   容量 RB 的环形缓冲（s, a, r, s', done 五张平表）
;;   更新       TD 目标 y = r + γ·(1-done)·max_a' Q_target(s')
;;              L = (Q(s,a) − y)²，梯度按批量累加后**一次** Adam 更新
;;   行为策略   ε-贪心（ε 线性衰减到 eps_min）
;;   推理策略   贪心，且优先用"最佳快照"（训练中得分创新高时的参数副本）——
;;              与 flappy 的 checkpoint 思路一致：训练后期在局部最优附近抖动时，
;;              实际游玩与落盘的永远是表现最好的那一版
;;
;; 分工：本模块只做"学习 + 决策"，环境规则全在 app/lib/snake_core.scm（S.*），
;; 界面与节拍在 app/snake.scm。于是同一段 DQN 代码既能被 App 调用，也能被
;; tools/e2e 的无头测试台直接跑（对拍/收敛性断言）。
;;
;; 性能预算（为什么这些超参是这些数）
;;   解释器一个时间片默认 312500 条 VM 指令（App 可经 (App.set_timeslice) 提到
;;   1250000）。实测一次 NI=28/NH=32 的前向约 3.3 万条指令；一次"动作 + 记忆 +
;;   按 1/16 概率的学习"合计约 9 万条指令/环境步 → 一个时间片约 13 步。
;;   故：批量 8、每 16 步一更新（更新太频繁会被 Adam 的 O(参数量) 更新开销吞掉）。
;;   离线预训练（tools/train_snake_dqn.py）用批量 32、每 2 步一更新，快得多。
;;
;; 持久化（(Fs.write) → data/snake_dqn.txt，WASM 下同时进 IndexedDB）
;;   纯文本、全整数（wasm32 的 %g 只有 6 位有效数字，浮点会打出 1.23457e+06 之类的
;;   指数形式；整数在 27 位内逐位精确）：
;;     1 <NI> <NH> <episodes> <env_steps> <best> <eps*1000> w0 w1 ... w(N-1)
;;   权重按 1/WS（WS = 10000）定点存储；版本号不是 1 / 维度不符 / 个数不足 → 整份弃用。
;;
;; 解释器约束（manual.md）：算术/比较恒二参数；无一元取负 → (- 0 x)；
;; 无 let/list/map/append；块内不 define（define 只写在 lambda 体顶层）；
;; / 与 mod 恒浮点（下标一律 Math.floor）；整数有效位 27 位（本模块只放小整数）；
;; Math 库函数结果为 NaN 时返回 #null → 前向里显式夹紧/拦 NaN，避免把 #null 喂进算术。
;; ============================================================================

(native Math)
(native String)

(import S "app/lib/snake_core.scm")

;; ---- 动作与超参 -----------------------------------------------------------

(define NO 3)               ; 直行 / 右转 / 左转
(define NI 0)               ; 观测维数（由载入的权重推出）
(define NH 0)               ; 隐层宽度（由载入的权重推出）

(define GAMMA 0.9)          ; 折扣（与 snake_core 的塑形一致）
(define LR 5e-4)            ; 在线微调学习率（离线预训练用 1e-3；微调更保守）
(define ADAM_B1 0.9)
(define ADAM_B2 0.999)
(define ADAM_EPS 1e-8)
(define RB 4000)            ; 回放容量（环境步）
(define BATCH 8)            ; 每次更新的批量
(define UPD_EVERY 16)       ; 每多少个环境步做一次梯度更新
(define TARGET_EVERY 500)   ; 每多少个环境步同步一次目标网络
(define CLAMP_W 50.0)       ; 权重夹紧（防发散；同时保证持久化不溢出 27 位整数）
(define WS 10000)           ; 持久化定点比例（权重 × 10000 存成整数）
(define HIST 40)            ; 得分曲线保留最近多少局

;; ---- 参数与缓冲 -----------------------------------------------------------

;; 当前网络（CW*/CB*）、目标网络（TW*/TB*）、最佳快照（BW*/BB*）
(define CW1 '()) (define CB1 '()) (define CW2 '()) (define CB2 '())
(define TW1 '()) (define TB1 '()) (define TW2 '()) (define TB2 '())
(define BW1 '()) (define BB1 '()) (define BW2 '()) (define BB2 '())
;; 前向读哪一组（fwd 只认 FW*/FB*，写 HV/QV）
(define FW1 '()) (define FB1 '()) (define FW2 '()) (define FB2 '())
(define HV '()) (define QV '())
;; Adam 一阶/二阶矩
(define MW1 '()) (define MB1 '()) (define MW2 '()) (define MB2 '())
(define VW1 '()) (define VB1 '()) (define VW2 '()) (define VB2 '())
(define adam_t 0)
;; 梯度累加
(define GW1 '()) (define GB1 '()) (define GW2 '()) (define GB2 '())
;; 回放（平表）—— **首次进入训练时才分配**：RS/RS2 各 RB*NI 项，启动期一次性
;; 建 22 万个元素要十几帧（浏览器里约 0.2~0.4 s），AI/手动模式根本用不到它。
(define RS '()) (define RS2 '()) (define RA '()) (define RR '()) (define RD '())
(define rptr 0) (define rsize 0) (define replay_ok 0)
;; 当前观测缓存（与 S.obs_v 分开：S 每次都会覆写自己的缓冲）
(define ob '())

;; ---- 统计与调度 -----------------------------------------------------------

(define eps 0.0) (define eps_min 0.03) (define eps_step 0.01)
(define episodes 0) (define env_steps 0) (define best_score 0)
(define ep_sum 0) (define ep_cnt 0) (define ep_avg 0.0)
(define hist '()) (define hist_i 0)
(define upd_cnt 0) (define tgt_cnt 0)
(define snap_avg 0.0)      ; 最佳快照对应的“20 局均分”
(define snap_ok 0)
(define ready 0)            ; 1 = 已初始化（权重表就位）

;; ---- 模块级临时量（每个函数一族，跨调用链绝不共用）-----------------------

(define mk_i 0) (define mk_n 0) (define mk_lst 0)
(define cp_i 0) (define cp_n 0)
(define z_i 0) (define z_n 0) (define z_lst 0)
(define fw_j 0) (define fw_i 0) (define fw_a 0) (define fw_acc 0.0) (define fw_e 0.0)
(define am_k 0) (define am_bk 0) (define am_bv 0.0) (define am_n 0)
(define bw_j 0) (define bw_i 0) (define bw_d 0.0) (define bw_dh 0.0) (define bw_dp 0.0)
(define bw_h 0.0) (define bw_x 0.0) (define bw_p 0.0)
(define l_i 0) (define l_idx 0) (define l_a 0) (define l_r 0.0) (define l_done 0)
(define l_tgt 0.0) (define l_mx 0.0) (define l_tk 0)
(define ad_k 0) (define ad_gg 0.0) (define ad_mm 0.0) (define ad_vv 0.0) (define ad_pv 0.0)
(define ad_bc1 1.0) (define ad_bc2 1.0)
(define tr_a 0) (define tr_r 0.0)
(define rm_i 0)
(define sn_n 0)
(define sz_out "") (define sz_i 0) (define sz_n 0) (define sz_v 0) (define sz_q 0.0)
(define pt_i 0) (define pt_ch 0) (define pt_acc 0) (define pt_have 0) (define pt_neg 0)
(define pt_n 0) (define pt_ok 0) (define pt_tok 0) (define pt_scale 0.0) (define pt_base 0)
(define rl_i 0) (define rl_v 0.0) (define rl_s 0.0)
(define gr_i 0)
(define in_i 0) (define in_v 0.0)
(define ep_k 0) (define ep_m 0) (define ep_slot 0)
(define ar_k 0) (define ar_bi 0) (define ar_bv 0.0)

;; ---- 小工具：新建表 / 复制 / 清零 ------------------------------------------

(define mk_list
  (lambda (n)
    {
      (define lst (cdr '(0)))
      (set! mk_i 0)
      (while (< mk_i n)
        {
          (push lst 0)
          (set! mk_i (+ mk_i 1))
        })
      lst
    }))

(define copy_list
  (lambda (src)
    {
      (define dst (cdr '(0)))
      (set! cp_n (length src))
      (set! cp_i 0)
      (while (< cp_i cp_n)
        {
          (push dst (get_item src cp_i))
          (set! cp_i (+ cp_i 1))
        })
      dst
    }))

(define zero_list
  (lambda (n)
    {
      (define lst (cdr '(0)))
      (set! z_i 0)
      (while (< z_i n)
        {
          (push lst 0)
          (set! z_i (+ z_i 1))
        })
      lst
    }))

;; ---- 前向：HV = tanh(B1 + W1·x)，QV = B2 + W2·HV --------------------------
;; x 是任意平表，base 是起点偏移（回放里直接按 idx*NI 取，免去拷贝）
;; tanh 用 1 - 2/(1+e^(2z))：两端都能自然饱和，(e^(2z)-1)/(e^(2z)+1) 在 z 很大时
;; 会算成 inf/inf = NaN，而 Math.exp 遇 NaN 返回 #null，喂进算术就报错。

(define fwd
  (lambda (x base) {
    (set! fw_j 0)
    (while (< fw_j NH)
      {
        (set! fw_acc (get_item FB1 fw_j))
        (set! fw_i 0)
        (while (< fw_i NI)
          {
            (set! fw_acc (+ fw_acc (* (get_item x (+ base fw_i))
                                     (get_item FW1 (+ (* fw_j NI) fw_i)))))
            (set! fw_i (+ fw_i 1))
          })
        (if (nan? fw_acc) (set! fw_acc 0.0) #f)
        (if (> fw_acc 40.0) (set! fw_acc 40.0) #f)
        (if (< fw_acc -40.0) (set! fw_acc -40.0) #f)
        (set! fw_e (Math.exp (* 2.0 fw_acc)))
        (set_item! HV fw_j (- 1.0 (/ 2.0 (+ 1.0 fw_e))))
        (set! fw_j (+ fw_j 1))
      })
    (set! fw_a 0)
    (while (< fw_a NO)
      {
        (set! fw_acc (get_item FB2 fw_a))
        (set! fw_j 0)
        (while (< fw_j NH)
          {
            (set! fw_acc (+ fw_acc (* (get_item HV fw_j)
                                     (get_item FW2 (+ (* fw_a NH) fw_j)))))
            (set! fw_j (+ fw_j 1))
          })
        (if (nan? fw_acc) (set! fw_acc 0.0) #f)
        (set_item! QV fw_a fw_acc)
        (set! fw_a (+ fw_a 1))
      })
  }))

(define fwd_current
  (lambda (x base) {
    (set! FW1 CW1) (set! FB1 CB1) (set! FW2 CW2) (set! FB2 CB2)
    (fwd x base)
  }))

(define fwd_target
  (lambda (x base) {
    (set! FW1 TW1) (set! FB1 TB1) (set! FW2 TW2) (set! FB2 TB2)
    (fwd x base)
  }))

(define fwd_best
  (lambda (x base) {
    (set! FW1 BW1) (set! FB1 BB1) (set! FW2 BW2) (set! FB2 BB2)
    (fwd x base)
  }))

;; QV 里最大值的下标（并列取先出现者 → 默认偏好"直行"）
(define argmax
  (lambda (vec n)
    {
      (set! ar_k 1)
      (set! ar_bi 0)
      (set! ar_bv (get_item vec 0))
      (while (< ar_k n)
        {
          (if (> (get_item vec ar_k) ar_bv)
              {
                (set! ar_bv (get_item vec ar_k))
                (set! ar_bi ar_k)
              }
              #f)
          (set! ar_k (+ ar_k 1))
        })
      ar_bi
    }))

;; ---- 观测缓存 -------------------------------------------------------------

(define sync_obs
  (lambda () {
    (S.obs)
    (set! rm_i 0)
    (while (< rm_i NI)
      {
        (set_item! ob rm_i (get_item S.obs_v rm_i))
        (set! rm_i (+ rm_i 1))
      })
  }))

;; ---- 记忆：把 (ob, a, r, S.obs_v, done) 写进环形回放 ---------------------

(define remember
  (lambda (a r)
    {
      (set! rm_i 0)
      (while (< rm_i NI)
        {
          (set_item! RS (+ (* rptr NI) rm_i) (get_item ob rm_i))
          (set_item! RS2 (+ (* rptr NI) rm_i) (get_item S.obs_v rm_i))
          (set! rm_i (+ rm_i 1))
        })
      (set_item! RA rptr a)
      (set_item! RR rptr r)
      (set_item! RD rptr S.dead)
      (set! rptr (mod (+ rptr 1) RB))
      (if (< rsize RB) (set! rsize (+ rsize 1)) #f)
    }))

;; ---- 反向传播（单样本、只对所选动作 a；梯度累加到 G*）--------------------
;; dL/dQ[a] = 2·(Q(s,a) − y)/BATCH（MSE 的平均梯度），其余动作不产生梯度。

(define backward
  (lambda (x base a tgt) {
    (set! bw_d (* 2.0 (/ (- (get_item QV a) tgt) BATCH)))
    (if (nan? bw_d) (set! bw_d 0.0) #f)
    ;; 输出层：W2[a][j] 与 b2[a]
    (set! bw_j 0)
    (while (< bw_j NH)
      {
        (set! bw_h (get_item HV bw_j))
        (set! bw_p (+ (* a NH) bw_j))
        (set_item! GW2 bw_p (+ (get_item GW2 bw_p) (* bw_d bw_h)))
        (set! bw_j (+ bw_j 1))
      })
    (set_item! GB2 a (+ (get_item GB2 a) bw_d))
    ;; 隐层：dh = d·W2[a][j]，dpre = dh·(1−h²)
    (set! bw_j 0)
    (while (< bw_j NH)
      {
        (set! bw_h (get_item HV bw_j))
        (set! bw_dh (* bw_d (get_item CW2 (+ (* a NH) bw_j))))
        (set! bw_dp (* bw_dh (- 1.0 (* bw_h bw_h))))
        (set_item! GB1 bw_j (+ (get_item GB1 bw_j) bw_dp))
        (set! bw_i 0)
        (while (< bw_i NI)
          {
            (set! bw_p (+ (* bw_j NI) bw_i))
            (set! bw_x (get_item x (+ base bw_i)))
            (set_item! GW1 bw_p (+ (get_item GW1 bw_p) (* bw_dp bw_x)))
            (set! bw_i (+ bw_i 1))
          })
        (set! bw_j (+ bw_j 1))
      })
  }))

;; ---- Adam：一次批量更新（p -= lr·m̂/(√v̂+ε)）-----------------------------

(define adam_apply
  (lambda (p m v g n) {
    (set! ad_k 0)
    (while (< ad_k n)
      {
        (set! ad_gg (get_item g ad_k))
        (set! ad_mm (+ (* ADAM_B1 (get_item m ad_k)) (* (- 1.0 ADAM_B1) ad_gg)))
        (set! ad_vv (+ (* ADAM_B2 (get_item v ad_k)) (* (- 1.0 ADAM_B2) (* ad_gg ad_gg))))
        (set_item! m ad_k ad_mm)
        (set_item! v ad_k ad_vv)
        (if (< ad_vv 0.0) (set! ad_vv 0.0) #f)
        (set! ad_pv (- (get_item p ad_k)
                       (* LR (/ (/ ad_mm ad_bc1)
                                (+ (Math.sqrt (/ ad_vv ad_bc2)) ADAM_EPS)))))
        (if (nan? ad_pv) (set! ad_pv 0.0) #f)
        (if (> ad_pv CLAMP_W) (set! ad_pv CLAMP_W) #f)
        (if (< ad_pv (- 0 CLAMP_W)) (set! ad_pv (- 0 CLAMP_W)) #f)
        (set_item! p ad_k ad_pv)
        (set! ad_k (+ ad_k 1))
      })
  }))

(define zero_grads
  (lambda () {
    (set! gr_i 0)
    (while (< gr_i (length GW1)) { (set_item! GW1 gr_i 0.0) (set! gr_i (+ gr_i 1)) })
    (set! gr_i 0)
    (while (< gr_i NH) { (set_item! GB1 gr_i 0.0) (set! gr_i (+ gr_i 1)) })
    (set! gr_i 0)
    (while (< gr_i (length GW2)) { (set_item! GW2 gr_i 0.0) (set! gr_i (+ gr_i 1)) })
    (set! gr_i 0)
    (while (< gr_i NO) { (set_item! GB2 gr_i 0.0) (set! gr_i (+ gr_i 1)) })
  }))

;; 一次学习：抽 BATCH 条样本 → 累加梯度 → 一次 Adam 更新
(define learn
  (lambda () {
    (zero_grads)
    (set! l_i 0)
    (while (< l_i BATCH)
      {
        (set! l_idx (Math.floor (* rsize (Math.random))))
        (set! l_a (get_item RA l_idx))
        (set! l_r (get_item RR l_idx))
        (set! l_done (get_item RD l_idx))
        ;; 目标网络算 s' 的最大 Q
        (fwd_target RS2 (* l_idx NI))
        (set! l_mx (get_item QV 0))
        (set! l_tk 1)
        (while (< l_tk NO)
          {
            (if (> (get_item QV l_tk) l_mx) (set! l_mx (get_item QV l_tk)) #f)
            (set! l_tk (+ l_tk 1))
          })
        (set! l_tgt l_r)
        (if (== l_done 0) (set! l_tgt (+ l_r (* GAMMA l_mx))) #f)
        ;; 当前网络算 s 的 Q（HV/QV 随之就位），再反传
        (fwd_current RS (* l_idx NI))
        (backward RS (* l_idx NI) l_a l_tgt)
        (set! l_i (+ l_i 1))
      })
    (set! adam_t (+ adam_t 1))
    (set! ad_bc1 (- 1.0 (pow ADAM_B1 adam_t)))
    (set! ad_bc2 (- 1.0 (pow ADAM_B2 adam_t)))
    (adam_apply CW1 MW1 VW1 GW1 (length CW1))
    (adam_apply CB1 MB1 VB1 GB1 NH)
    (adam_apply CW2 MW2 VW2 GW2 (length CW2))
    (adam_apply CB2 MB2 VB2 GB2 NO)
  }))

;; ---- 动作选择 -------------------------------------------------------------

;; ε-贪心（训练）：先按当前网络取贪心动作，再按 ε 概率随机
(define pick_eps
  (lambda () {
    (if (< (Math.random) eps)
        (Math.floor (* NO (Math.random)))
        {
          (fwd_current ob 0)
          (argmax QV NO)
        })
  }))

;; 贪心（游玩）：用最佳快照；没有快照就用当前网络
(define pick_best
  (lambda () {
    (if (== snap_ok 1) (fwd_best ob 0) (fwd_current ob 0))
    (argmax QV NO)
  }))

;; ---- 环境交互 -------------------------------------------------------------

;; 回放表按需分配（见上方 RS 的说明）
(define ensure_replay
  (lambda () {
    (if (== replay_ok 0)
        {
          (set! RS (zero_list (* RB NI)))
          (set! RS2 (zero_list (* RB NI)))
          (set! RA (zero_list RB))
          (set! RR (zero_list RB))
          (set! RD (zero_list RB))
          (set! rptr 0) (set! rsize 0)
          (set! replay_ok 1)
        }
        #f)
  }))

;; 训练一步：选动作 → 环境推进 → 记忆 → 观测刷新 → 定期学习/同步
(define step_train
  (lambda () {
    (ensure_replay)
    (set! tr_a (pick_eps))
    (set! tr_r (S.step tr_a))
    (S.obs)                     ; S.obs_v ← 新状态
    (remember tr_a tr_r)        ; 旧状态 ob + 新状态 S.obs_v
    (set! rm_i 0)
    (while (< rm_i NI)
      {
        (set_item! ob rm_i (get_item S.obs_v rm_i))
        (set! rm_i (+ rm_i 1))
      })
    (set! env_steps (+ env_steps 1))
    ;; 用两个小计数器代替 mod(env_steps, N)：env_steps 会跨会话累计（模型文件里带回来），
    ;; 长期跑下去可能顶到 27 位整数上限（溢出后结果未规定），计数器则恒在 [0,N)
    (set! upd_cnt (+ upd_cnt 1))
    (if (>= upd_cnt UPD_EVERY)
        {
          (set! upd_cnt 0)
          (if (>= rsize BATCH) (learn) #f)
        }
        #f)
    (set! tgt_cnt (+ tgt_cnt 1))
    (if (>= tgt_cnt TARGET_EVERY)
        {
          (set! tgt_cnt 0)
          (sync_target)
        }
        #f)
  }))

;; 游玩一步：贪心动作
(define step_greedy
  (lambda () {
    (set! tr_a (pick_best))
    (set! tr_r (S.step tr_a))
    (sync_obs)
  }))

;; ---- 目标网络同步 / 最佳快照 ---------------------------------------------

(define sync_target
  (lambda () {
    (set! in_i 0)
    (while (< in_i (length TW1)) { (set_item! TW1 in_i (get_item CW1 in_i)) (set! in_i (+ in_i 1)) })
    (set! in_i 0)
    (while (< in_i NH) { (set_item! TB1 in_i (get_item CB1 in_i)) (set! in_i (+ in_i 1)) })
    (set! in_i 0)
    (while (< in_i (length TW2)) { (set_item! TW2 in_i (get_item CW2 in_i)) (set! in_i (+ in_i 1)) })
    (set! in_i 0)
    (while (< in_i NO) { (set_item! TB2 in_i (get_item CB2 in_i)) (set! in_i (+ in_i 1)) })
  }))

(define snapshot
  (lambda () {
    (set! in_i 0)
    (while (< in_i (length BW1)) { (set_item! BW1 in_i (get_item CW1 in_i)) (set! in_i (+ in_i 1)) })
    (set! in_i 0)
    (while (< in_i NH) { (set_item! BB1 in_i (get_item CB1 in_i)) (set! in_i (+ in_i 1)) })
    (set! in_i 0)
    (while (< in_i (length BW2)) { (set_item! BW2 in_i (get_item CW2 in_i)) (set! in_i (+ in_i 1)) })
    (set! in_i 0)
    (while (< in_i NO) { (set_item! BB2 in_i (get_item CB2 in_i)) (set! in_i (+ in_i 1)) })
    (set! snap_ok 1)
  }))

;; ---- 一局结束的记账 -------------------------------------------------------

(define end_episode
  (lambda () {
    (set! episodes (+ episodes 1))
    (set! ep_sum (+ ep_sum S.score))
    (set! ep_cnt (+ ep_cnt 1))
    ;; 每 20 局算一次均分；均分创新高才更新“最佳快照”（单局运气好不算数）
    (if (>= ep_cnt 20)
        {
          (set! ep_avg (/ ep_sum ep_cnt))
          (set! ep_sum 0)
          (set! ep_cnt 0)
          (if (> ep_avg snap_avg)
              {
                (snapshot)
                (set! snap_avg ep_avg)
              }
              #f)
        }
        #f)
    ;; 得分曲线（环形，hist_i 恒在 [0,HIST)）
    (set_item! hist hist_i S.score)
    (set! hist_i (mod (+ hist_i 1) HIST))
    (if (> S.score best_score) (set! best_score S.score) #f)
    ;; ε 线性衰减
    (if (> eps eps_min) (set! eps (- eps eps_step)) #f)
  }))

(define set_schedule
  (lambda (e0 emin nsteps)
    {
      (set! eps e0)
      (set! eps_min emin)
      (set! eps_step (/ (- e0 emin) nsteps))
      (if (< eps_step 0.0) (set! eps_step 0.0) #f)
    }))

(define reset_stats
  (lambda () {
    (set! episodes 0)
    (set! env_steps 0)
    (set! ep_sum 0)
    (set! ep_cnt 0)
    (set! ep_avg 0.0)
    (set! hist_i 0)
    (set! ep_k 0)
    (while (< ep_k HIST) { (set_item! hist ep_k 0) (set! ep_k (+ ep_k 1)) })
  }))

;; ---- 初始化：从权重表 -----------------------------------------------------

(define init_from
  (lambda (w1 b1 w2 b2) {
    (set! NH (length b1))
    (set! NI (Math.floor (/ (length w1) NH)))
    (set! CW1 (copy_list w1)) (set! CB1 (copy_list b1))
    (set! CW2 (copy_list w2)) (set! CB2 (copy_list b2))
    (set! TW1 (copy_list w1)) (set! TB1 (copy_list b1))
    (set! TW2 (copy_list w2)) (set! TB2 (copy_list b2))
    (set! BW1 (copy_list w1)) (set! BB1 (copy_list b1))
    (set! BW2 (copy_list w2)) (set! BB2 (copy_list b2))
    (set! MW1 (zero_list (length w1))) (set! MB1 (zero_list (length b1)))
    (set! MW2 (zero_list (length w2))) (set! MB2 (zero_list (length b2)))
    (set! VW1 (zero_list (length w1))) (set! VB1 (zero_list (length b1)))
    (set! VW2 (zero_list (length w2))) (set! VB2 (zero_list (length b2)))
    (set! GW1 (zero_list (length w1))) (set! GB1 (zero_list (length b1)))
    (set! GW2 (zero_list (length w2))) (set! GB2 (zero_list (length b2)))
    (set! HV (zero_list NH)) (set! QV (zero_list NO))
    (set! ob (zero_list NI))
    (set! RS '()) (set! RS2 '()) (set! RA '()) (set! RR '()) (set! RD '())
    (set! replay_ok 0) (set! rptr 0) (set! rsize 0)
    (set! hist (zero_list HIST))
    (set! adam_t 0)
    (set! snap_ok 1) (set! snap_avg 0.0)
    (set! FW1 CW1) (set! FB1 CB1) (set! FW2 CW2) (set! FB2 CB2)
    (set! ready 1)
  }))

;; 随机初始化（「从头学」）
(define init_random
  (lambda () {
    (set! NH (length CB1))
    (set! NI (Math.floor (/ (length CW1) NH)))
    (set! in_i 0)
    (while (< in_i (length CW1))
      {
        (set_item! CW1 in_i (* (/ 1.0 (Math.sqrt NI)) (- (* 2.0 (Math.random)) 1.0)))
        (set_item! MW1 in_i 0.0) (set_item! VW1 in_i 0.0)
        (set! in_i (+ in_i 1))
      })
    (set! in_i 0)
    (while (< in_i NH)
      {
        (set_item! CB1 in_i 0.0) (set_item! MB1 in_i 0.0) (set_item! VB1 in_i 0.0)
        (set! in_i (+ in_i 1))
      })
    (set! in_i 0)
    (while (< in_i (length CW2))
      {
        (set_item! CW2 in_i (* (/ 1.0 (Math.sqrt NH)) (- (* 2.0 (Math.random)) 1.0)))
        (set_item! MW2 in_i 0.0) (set_item! VW2 in_i 0.0)
        (set! in_i (+ in_i 1))
      })
    (set! in_i 0)
    (while (< in_i NO)
      {
        (set_item! CB2 in_i 0.0) (set_item! MB2 in_i 0.0) (set_item! VB2 in_i 0.0)
        (set! in_i (+ in_i 1))
      })
    (set! adam_t 0) (set! rptr 0) (set! rsize 0)
    (sync_target)
    (snapshot)
    (set! snap_avg 0.0)
    (set! best_score 0)
  }))

;; ---- 持久化：整数文本 -----------------------------------------------------
;; 格式：1 NI NH episodes env_steps best eps*1000 然后全部权重（×WS 取整）
;; 顺序：W1 B1 W2 B2

(define push_int
  (lambda (v)
    {
      (set! sz_q (Math.round v))
      (if (nan? sz_q) (set! sz_q 0.0) #f)
      (if (> sz_q 999999.0) (set! sz_q 999999.0) #f)
      (if (< sz_q -999999.0) (set! sz_q -999999.0) #f)
      (set! sz_out (String.concat sz_out (String.atom_to_string sz_q)))
      (set! sz_out (String.concat sz_out " "))
    }))

(define push_list
  (lambda (lst n) {
    (set! sz_i 0)
    (while (< sz_i n)
      {
        (push_int (* WS (get_item lst sz_i)))
        (set! sz_i (+ sz_i 1))
      })
  }))

(define serialize
  (lambda () {
    (set! sz_out "")
    (push_int 1)
    (push_int NI)
    (push_int NH)
    (push_int episodes)
    (push_int env_steps)
    (push_int best_score)
    (push_int (* 1000 eps))
    (push_list CW1 (length CW1))
    (push_list CB1 NH)
    (push_list CW2 (length CW2))
    (push_list CB2 NO)
    sz_out
  }))

;; 把文本解析成数字表（只含整数；空白/换行分隔）
(define parse_nums
  (lambda (txt)
    {
      (set! pt_tok (cdr '(0)))
      (set! pt_n (String.length txt))
      (set! pt_i 0)
      (set! pt_acc 0)
      (set! pt_have 0)
      (set! pt_neg 0)
      (while (< pt_i pt_n)
        {
          (set! pt_ch (String.charCodeAt pt_i txt))
          (if (and (>= pt_ch 48) (<= pt_ch 57))
              {
                (set! pt_acc (+ (* pt_acc 10) (- pt_ch 48)))
                (set! pt_have 1)
              }
              (if (== pt_ch 45)
                  (set! pt_neg 1)
                  (if (== pt_have 1)
                      {
                        (if (== pt_neg 1)
                            (push pt_tok (- 0 pt_acc))
                            (push pt_tok pt_acc))
                        (set! pt_acc 0) (set! pt_have 0) (set! pt_neg 0)
                      }
                      #f)))
          (set! pt_i (+ pt_i 1))
        })
      (if (== pt_have 1)
          (if (== pt_neg 1) (push pt_tok (- 0 pt_acc)) (push pt_tok pt_acc))
          #f)
      pt_tok
    }))

;; 载入文本模型：成功 1 / 失败 0（失败时参数保持原样）
(define load_text
  (lambda (txt)
    {
      (define toks 0)
      (set! pt_ok 0)
      (if (> (String.length txt) 0)
          {
            (set! toks (parse_nums txt))
            (if (>= (length toks) 7)
                (if (== (get_item toks 0) 1)
                    (if (== (get_item toks 1) NI)
                        (if (== (get_item toks 2) NH)
                            (if (>= (length toks) (+ 7 (+ (* NH NI) (+ NH (+ (* NO NH) NO)))))
                                (set! pt_ok 1)
                                #f)
                            #f)
                        #f)
                    #f)
                #f)
            (if (== pt_ok 1)
                {
                  (set! pt_base 7)
                  (set! pt_scale (/ 1.0 WS))
                  (set! in_i 0)
                  (while (< in_i (length CW1))
                    {
                      (set_item! CW1 in_i (* pt_scale (get_item toks (+ pt_base in_i))))
                      (set! in_i (+ in_i 1))
                    })
                  (set! pt_base (+ pt_base (length CW1)))
                  (set! in_i 0)
                  (while (< in_i NH)
                    {
                      (set_item! CB1 in_i (* pt_scale (get_item toks (+ pt_base in_i))))
                      (set! in_i (+ in_i 1))
                    })
                  (set! pt_base (+ pt_base NH))
                  (set! in_i 0)
                  (while (< in_i (length CW2))
                    {
                      (set_item! CW2 in_i (* pt_scale (get_item toks (+ pt_base in_i))))
                      (set! in_i (+ in_i 1))
                    })
                  (set! pt_base (+ pt_base (length CW2)))
                  (set! in_i 0)
                  (while (< in_i NO)
                    {
                      (set_item! CB2 in_i (* pt_scale (get_item toks (+ pt_base in_i))))
                      (set! in_i (+ in_i 1))
                    })
                  (set! episodes (get_item toks 3))
                  (set! env_steps (get_item toks 4))
                  (set! best_score (get_item toks 5))
                  (set! eps (/ (get_item toks 6) 1000))
                  (set! snap_avg 0.0)
                  (set! adam_t 0) (set! rptr 0) (set! rsize 0)
                  (sync_target)
                  (snapshot)
                }
                #f)
          }
          #f)
      pt_ok
    }))
