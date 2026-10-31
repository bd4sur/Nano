;; =============================================================================
;; app/lib/sand_core.scm —— 沙盒颗粒物理（Falling Sand）内核
;;
;; 为什么单独成模块：物理（网格 / 材料 / 规则表 / 活跃格脏表）与显示输入无关，
;; 可以脱离 Gfx/Touch/Imu 单独回归（见 infer/tools/e2e/native/test_fallingsand.py）。
;; 界面层 app/fallingsand.scm 只做三件事：读触屏/姿态、把脏格画出来、画 HUD。
;;
;; 设计要点（对应需求）：
;;   1) 网格 106×80（每格 3px，318×240 铺满 320×240 屏，左右各留 1px 边框）；
;;      网格外再套 1 圈“墙”环（PWW=108 / PHH=82），于是**热路径里没有任何边界判断**
;;      —— 邻居偏移量全是常数（见 OFF），set_item! 永远落在合法下标上。
;;   2) **活跃格脏表**（active / dirty table）：只为“变动过的格子”建队，每步只处理
;;      队列里的格子；稳态（沙堆静止）时队列是空的，单步成本≈0（实测 0.01ms）。
;;      队列是**集合语义**（mk[k]=1 表示“已在队里”，出队清零）—— 队列中绝无重复项，
;;      长度天然 ≤ NP；配合 FIFO 游标 + 单步预算，既不会饿死队尾，单步也有上界。
;;   3) 重力方向取自 Imu（由界面层传入：(set_gravity gx gy)），量化成 8 个方位之一；
;;      所有“下 / 左下 / 右下 / 侧向 / 上”都由方位查表得到常数偏移量，
;;      因此机器倒过来时沙子就往上流，横放时往侧面流，不需要另写一套逻辑。
;;      方位改变时置分片全扫（sweep）标志：分批把所有非空格子重新入表。
;;   4) 两张 10×10 扁平规则表：
;;        MV（可移动表）：MV[我][对方]=1 表示“我可以与该格交换（沉/浮穿过它）”，
;;                        决定材质能否互相穿插（沙沉水、水沉油、汽浮水……）；
;;        RT（交互表）  ：RT[我][对方]=标志位，火＋植物/油=点燃、水＋火=化汽、
;;                        酸＋石/沙/植物=溶解、植物＋水=生长、水＋酸=稀释、汽＋火=窒熄。
;;      规则只在“反应性材质”（水/火/酸/植物/蒸汽）处理时查表。
;;   5) 单步预算可控：每步最多处理 MAXPROC 个活跃格，多出来的**结转到下一步**
;;      （不会丢格），因此再剧烈的场景单步耗时也有上界。
;;   6) **热路径的内层用宏内联**：本解释器里一次 Scheme 闭包调用 ≈ 5 次内建操作
;;      （实测 0.28ms/1000 次 vs 0.06ms/1000 次），而一次移动要往活跃队列里塞 9 个
;;      邻格 —— 故 addf!/addp! 做成 define-syntax 宏（编译期内联，零调用开销）。
;;      注意**别把宏套得太深**：把 try!/mv!/powder! 也做成嵌套宏会把编译期内存打爆
;;      （模块连 16MB 池都 link_fail），所以策略函数保持为普通 lambda。
;;
;; 解释器约束（manual.md，逐条遵守；详见 doc/AGENTS.md）：
;;   · 算术恒二参数（多元会污染操作数栈）；没有一元取负（用 -0 常量）；
;;   · 没有 let/case/do/for/list/append/map；define 无过程简写；
;;   · `/` 与 `mod` 恒返回浮点 —— 列表下标一律 Math.floor 收口；
;;   · `'()` 字面量是模块级静态槽 —— 长表只在加载时 push 一次，此后只 set_item!；
;;   · 模块级临时量按“谁写谁读”分名，跨函数调用链上绝不共用（p_/s_/q_/b_/d_ 前缀）。
;; =============================================================================

(native Math)

;; ---- 材料编号 ---------------------------------------------------------------
(define M_EMPTY 0)
(define M_SAND  1)
(define M_WATER 2)
(define M_OIL   3)
(define M_FIRE  4)
(define M_PLANT 5)
(define M_ACID  6)
(define M_STEAM 7)
(define M_STONE 8)
(define M_WALL  9)          ; 只在边框环上出现，任何规则都不作用于它

;; ---- 网格几何 ---------------------------------------------------------------
(define W 106)              ; 世界列数（可见格）
(define H 80)               ; 世界行数（可见格）
(define PWW 108)            ; 带 1 圈边框环的网格宽 = W+2
(define PHH 82)
(define NP 8856)            ; PWW*PHH

;; ---- 运行参数 ---------------------------------------------------------------
(define MAXPROC 400)        ; 单步处理预算（活跃格数）——“5~8ms 单步”的主旋钮
(define SLICE 1100)         ; 换向后分片全扫的每步格数
(define FIRE_LIFE 7)        ; 火存活步数
(define STEAM_LIFE 70)      ; 蒸汽存活步数（到期凝结为水）
(define ACID_LIFE 14)       ; 酸的“可溶解量”
(define PWW1 109)           ; PWW+1
(define PWW_1 107)          ; PWW-1

;; 8 个方位的邻居偏移（d=0 正下、1 左下、2 左、3 左上、4 上、5 右上、6 右、7 右下）
(define OFF (quote (108 107 -1 -109 -108 -107 1 109)))

;; ---- 规则表（10×10 扁平表，索引 = 我*10 + 对方） ----------------------------
;; MV：1 = 我可以与对方交换（沙沉水、水沉油、汽浮水……）
(define MV (quote (
  0 0 0 0 0 0 0 0 0 0
  1 0 1 1 1 0 1 1 0 0
  1 0 0 1 1 0 1 1 0 0
  1 0 0 0 1 0 0 1 0 0
  1 0 0 0 0 0 0 1 0 0
  0 0 0 0 0 0 0 0 0 0
  1 0 1 1 1 0 0 1 0 0
  1 0 1 1 0 0 1 0 0 0
  0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0
  )))

;; RT：交互标志位
;;   1  = 点燃（对方 → 火）
;;   2  = 溶解（对方 → 空）
;;   4  = 化汽（对方 → 蒸汽）
;;   8  = 生长（对方 → 我，植物吸水）
;;   16 = 稀释（对方/酸 → 水）
;;   32 = 窒熄（对方/火 → 空）
;;   64 = 烧开（对方/火 → 蒸汽，且自己（水）隔次也化汽）
(define R_IGNITE   1)
(define R_DISSOLVE 2)
(define R_STEAMIFY 4)
(define R_GROW     8)
(define R_DILUTE   16)
(define R_SMOTHER  32)
(define R_BOIL     64)
(define RT (quote (
  0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0
  0 0 0 0 64 0 16 0 0 0
  0 0 0 0 0 0 0 0 0 0
  0 0 4 1 0 1 0 0 0 0
  0 0 8 0 0 0 0 0 0 0
  0 2 0 0 0 2 0 0 2 0
  0 0 0 0 32 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0
  )))

;; ---- 状态表（扁平，索引 = 网格下标） ---------------------------------------
(define g  (cons M_WALL (quote ())))   ; 材料
(define lf (cons 0 (quote ())))        ; 生命/计时（火、蒸汽、酸）
(define mk (cons 0 (quote ())))        ; 活跃表代戳
(define mu (cons 0 (quote ())))        ; “本步已更新”代戳（每格每步至多动一次）
(define md (cons 0 (quote ())))        ; “已重新排队”代戳（被跳过时每步只补排一次）
(define pk (cons 0 (quote ())))        ; 绘制表代戳
(define la (cons 0 (quote ())))        ; 活跃表 A（当前）
(define lb (cons 0 (quote ())))        ; 活跃表 B（下一步）
(define lp (cons 0 (quote ())))        ; 绘制表（本步变动过的格子）

;; ---- 当前重力方位及其邻居偏移（每步由 set_gravity 刷新） -------------------
(define dv 0)               ; 方位 0..7
(define d_od 108)           ; 正下
(define d_odl 107)          ; 左下
(define d_odr 109)          ; 右下
(define d_ol1 1)            ; 侧向之一（d+2）
(define d_ol2 -1)           ; 侧向之二（d-2）
(define d_ou -108)          ; 正上（气体浮力方向）
(define d_og1 -109)         ; 左上
(define d_og2 -107)         ; 右上
(define d_gx 0.0)
(define d_gy 9.8)
(define d_ax 0.0)
(define d_ay 9.8)
(define d_new 0)
(define d_ta 0)
(define d_ts 0)

;; ---- 步进计数器与状态 -------------------------------------------------------
(define st_gen 1)           ; 步序号（每步 +1；也用作 mu/md 的代戳）
(define st_pgen 1)          ; 绘制表代戳（每步 +1；同上）
(define QCAP 17712)         ; 活跃队列容量 = 2*NP（队列里每格最多出现一次；游标过半即压缩）
(define st_q (cons 0 (quote ())))      ; 活跃队列（FIFO 环形使用的扁平表）
(define st_qlen 0)          ; 队列长度（尾部下标）
(define st_qpos 0)          ; 处理游标（头部下标）
(define st_w 0)
(define st_def (cons 0 (quote ())))    ; “本步被跳过、留到下一步再审”的延迟队列
(define st_dlen 0)
(define sd 0)
(define st_plen 0)          ; 绘制表长度
(define st_idx 0)
(define st_v 0)
(define st_lv 0)
(define st_budget 0)
(define st_cnt 0)           ; “硬币”：每用一次翻面（0/1），取代随机数的零成本抖动
(define st_sweep 0)         ; 分片全扫游标：0 = 不扫，1..NP = 扫描进度
(define st_fire 0)          ; 上一步的火格数（界面 HUD 用）
(define st_moves 0)         ; 上一步的移动次数（界面 HUD 用）
(define st_processed 0)     ; 上一步实际处理的活跃格数
(define st_pending 0)       ; 活跃表待处理格数
(define st_tk 0)            ; 步序号（供植物慢节拍）
(define st_grow 0)          ; 植物“慢节拍”标志（每 8 步一次）

;; ---- 处理一个格子时用的临时量（p_） ---------------------------------------
(define p_ok 0)
(define p_l 0)
(define p_tv 0)
(define p_t2 0)
(define p_tt 0)
(define p_rc 0)
(define p_eat 0)
(define p_rot 0)            ; 规则抽查方向轮换（4 组，4 步覆盖 8 邻域）
;; ---- 分片扫描用的临时量（s_） ---------------------------------------------
(define s_k 0)
(define s_i 0)
;; ---- 初始化 / 撒料用的临时量（q_） ---------------------------------------
(define q_x 0)
(define q_y 0)
(define q_d 0)
(define q_tv 0)
(define q_coin 0)
(define b_i 0)
(define b_j 0)

;; =============================================================================
;; 热路径宏（编译期内联）
;; =============================================================================

;; 把一个格子加入**下一步**的活跃表（mk 代戳去重）。
;; 集合语义：mk[k]==1 表示“已在队列里”。出队时清零，于是同一格可以在一拍内
;; 被重新入队、也可以在后续拍里再次入队；但队列中**绝不会有重复项**。
(define-syntax addf!
  (syntax-rules ()
    ((addf! k)
     (if (== (get_item mk k) 0)
         ;; 空格与边框墙**不需要处理**，不入队。这一步很关键：一次移动要摸 9 个邻格，
         ;; 若把空格也排进队，队列里绝大多数是无事可做的格子，处理预算被白白摊薄
         ;; （实测：1060 颗沙的堆会让队列膨胀到 3000+，每颗沙七八步才轮到一次）。
         (if (> (get_item g k) 0)
             {
               (set_item! mk k 1)      ; 1 = 已在队列里（出队时清 0）
               (if (< st_qlen QCAP)
                   {
                     (set_item! st_q st_qlen k)
                     (set! st_qlen (+ st_qlen 1))
                   }
                   #f)
             }
             #f)
         #f))))

;; 把一个格子加入绘制表（pk 代戳去重；带容量兜底，界面层忘记消费也不会越界）
(define-syntax addp!
  (syntax-rules ()
    ((addp! k)
     (if (== (get_item pk k) st_pgen)
         #f
         {
           (set_item! pk k st_pgen)
           (if (< st_plen NP)
               {
                 (set_item! lp st_plen k)
                 (set! st_plen (+ st_plen 1))
               }
               #f)
         }))))

;; 把一个格子放进**延迟队列**：本步跳过它，下一步开头再排进活跃表
(define-syntax adddef!
  (syntax-rules ()
    ((adddef! k)
     (if (< st_dlen NP)
         {
           (set_item! st_def st_dlen k)
           (set! st_dlen (+ st_dlen 1))
         }
         #f))))

;; 以 k 为中心的 3×3 邻域全部入活跃表（含 k 自身）。
;; 边框环保证 k±(PWW+1) 恒在 [0,NP) 内，故无需边界判断。
(define tch
  (lambda (k)
    (addf! k)
    (addf! (+ k 1))
    (addf! (- k 1))
    (addf! (+ k PWW))
    (addf! (- k PWW))
    (addf! (+ k PWW1))
    (addf! (- k PWW1))
    (addf! (+ k PWW_1))
    (addf! (- k PWW_1))))

;; 把 st_idx 处的 st_v 与 t 处的 tv 交换（材料 + 生命），标脏并重新激活邻域。
;; 重新激活 5 格：目的格、源格、源格“移动方向后方”那格（沙落下后它上面那格就能
;; 接着掉）、源格两个侧邻（液体摊平要靠它们）。
(define mv
  (lambda (t tv)
    (set! st_lv (get_item lf t))
    (set_item! g t st_v)
    (set_item! g st_idx tv)
    (set_item! lf t (get_item lf st_idx))
    (set_item! lf st_idx st_lv)
    (set_item! mu t st_gen)          ; 目的格本步不再被别的格子占用（防“一步走两格”）
    (addp! st_idx)
    (addp! t)
    ;; 重新激活：**源格的 3×3 邻域**。
    ;; 为什么是源格而不是目的格：颗粒一走，只有“源格空了”这件事会改变别人的可行性
    ;; —— 源格正上方的能接着掉、斜上方的能斜着滑、左右邻（液体）能摊进来，这三类
    ;; 全在源格的 3×3 里（目的格本身也在其中）；而目的格被占住后，它周围的格子
    ;; 只会更动弹不得。早先只激活目的格邻域时，“源格正上方”恰好落在 3 行之外，
    ;; 实测会留下悬空的沙；两处都激活则每次移动多入队一倍，处理吞吐反而被摊薄。
    (tch st_idx)
    (set! st_moves (+ st_moves 1))))

;; 尝试把 st_idx 处材料移到 t（下沉 / 上升 / 斜向）：成功返回 1。
;; 判据：目标格本步未被更新过，且 MV[我][对方]=1。
(define trymv
  (lambda (t)
    (if (== (get_item mu t) st_gen)
        {
          ;; 目标格本步刚更新过（被别的颗粒占用/处理）→ 这一步挪不动。
          ;; 若“目标格材质本来是能进入的”，就必须把自己留在活跃表里，
          ;; 否则会永久冻结在半空（实测会留下悬空的沙）；
          ;; 若材质本身就进不去（例如下面是沙），则不必重排，稳态才能归零。
          (set! p_tv (get_item g t))
          (if (== (get_item MV (+ (* st_v 10) p_tv)) 1) (addf! st_idx) #f)
          0
        }
        {
          (set! p_tv (get_item g t))
          (if (== (get_item MV (+ (* st_v 10) p_tv)) 1)
              { (mv t p_tv) 1 }
              0)
        })))

;; 侧向滑移（液体）：除了目标格可进入，还要求“目标格再往下一格”也可进入。
;; 这样平地上的一层液体不会永远横着漂（否则活跃表永不空、稳态成本失控），
;; 而遇到落差/空腔时仍会流进去（梯级、沟槽、沙堆边缘都正常）。
(define trylat
  (lambda (t)
    (if (== (get_item mu t) st_gen)
        {
          (addf! st_idx)
          0
        }
        {
          (set! p_tv (get_item g t))
          (if (== (get_item MV (+ (* st_v 10) p_tv)) 1)
              {
                (set! p_t2 (+ t d_od))
                (if (== (get_item g p_t2) M_EMPTY)
                    { (mv t p_tv) 1 }
                    (if (== (get_item MV (+ (* st_v 10) (get_item g p_t2))) 1)
                        { (mv t p_tv) 1 }
                        0))
              }
              0)
        })))

;; 侧向扩散（气体）：判据看“目标格再往上一格”
(define trylat_g
  (lambda (t)
    (if (== (get_item mu t) st_gen)
        {
          (addf! st_idx)
          0
        }
        {
          (set! p_tv (get_item g t))
          (if (== (get_item MV (+ (* st_v 10) p_tv)) 1)
              {
                (set! p_t2 (+ t d_ou))
                (if (== (get_item g p_t2) M_EMPTY)
                    { (mv t p_tv) 1 }
                    (if (== (get_item MV (+ (* st_v 10) (get_item g p_t2))) 1)
                        { (mv t p_tv) 1 }
                        0))
              }
              0)
        })))

;; 粉体（沙）：只查正下 / 左下 / 右下三格
(define do_powder
  (lambda ()
    (set! p_tt (+ st_idx d_od))
    (set! p_ok (trymv p_tt))
    (if (== p_ok 0)
        {
          (set! st_cnt (- 1 st_cnt))
          (if (== st_cnt 1)
              {
                (set! p_tt (+ st_idx d_odl))
                (set! p_ok (trymv p_tt))
                (if (== p_ok 0)
                    {
                      (set! p_tt (+ st_idx d_odr))
                      (set! p_ok (trymv p_tt))
                    }
                    #f)
              }
              {
                (set! p_tt (+ st_idx d_odr))
                (set! p_ok (trymv p_tt))
                (if (== p_ok 0)
                    {
                      (set! p_tt (+ st_idx d_odl))
                      (set! p_ok (trymv p_tt))
                    }
                    #f)
              })
        }
        #f)))

;; 液体（水 / 油 / 酸）：正下 → 两个斜下 → 侧向滑移（水加横向滑移）
(define do_liquid
  (lambda ()
    (do_powder)
    (if (== p_ok 0)
        {
          (set! st_cnt (- 1 st_cnt))
          (if (== st_cnt 1)
              {
                (set! p_tt (+ st_idx d_ol1))
                (set! p_ok (trylat p_tt))
                (if (== p_ok 0)
                    {
                      (set! p_tt (+ st_idx d_ol2))
                      (set! p_ok (trylat p_tt))
                    }
                    #f)
              }
              {
                (set! p_tt (+ st_idx d_ol2))
                (set! p_ok (trylat p_tt))
                (if (== p_ok 0)
                    {
                      (set! p_tt (+ st_idx d_ol1))
                      (set! p_ok (trylat p_tt))
                    }
                    #f)
              })
        }
        #f)))

;; 气体（火 / 汽）：逆重力上升（正上 → 两个斜上 → 侧向扩散）
(define do_gas
  (lambda ()
    (set! p_tt (+ st_idx d_ou))
    (set! p_ok (trymv p_tt))
    (if (== p_ok 0)
        {
          (set! st_cnt (- 1 st_cnt))
          (if (== st_cnt 1)
              {
                (set! p_tt (+ st_idx d_og1))
                (set! p_ok (trymv p_tt))
                (if (== p_ok 0)
                    {
                      (set! p_tt (+ st_idx d_og2))
                      (set! p_ok (trymv p_tt))
                    }
                    #f)
              }
              {
                (set! p_tt (+ st_idx d_og2))
                (set! p_ok (trymv p_tt))
                (if (== p_ok 0)
                    {
                      (set! p_tt (+ st_idx d_og1))
                      (set! p_ok (trymv p_tt))
                    }
                    #f)
              })
        }
        #f)
    (if (== p_ok 0)
        {
          (set! st_cnt (- 1 st_cnt))
          (if (== st_cnt 1)
              {
                (set! p_tt (+ st_idx d_ol1))
                (set! p_ok (trylat_g p_tt))
                (if (== p_ok 0)
                    {
                      (set! p_tt (+ st_idx d_ol2))
                      (set! p_ok (trylat_g p_tt))
                    }
                    #f)
              }
              {
                (set! p_tt (+ st_idx d_ol2))
                (set! p_ok (trylat_g p_tt))
                (if (== p_ok 0)
                    {
                      (set! p_tt (+ st_idx d_ol1))
                      (set! p_ok (trylat_g p_tt))
                    }
                    #f)
              })
        }
        #f)))

;; =============================================================================
;; 坐标与建表
;; =============================================================================

;; 世界坐标 (x,y) → 网格下标（0<=x<W, 0<=y<H）
(define gidx (lambda (x y) (+ (* (+ y 1) PWW) (+ x 1))))

(define build
  (lambda ()
    (set! b_i 1)
    (while (< b_i NP) {
      (push g M_WALL)
      (push lf 0)
      (push mk 0)
      (push mu 0)
      (push md 0)
      (push pk 0)
      (push st_def 0)
      (push lp 0)
      (set! b_i (+ b_i 1)) })
    ;; 活跃队列容量 2*NP（集合语义下待处理项 ≤ NP；游标过半即压缩，
    ;; 所以 2*NP 足够，且不必环形取模）
    (set! b_i 0)
    (while (< b_i QCAP)
      {
        (push st_q 0)
        (set! b_i (+ b_i 1))
      })
    ;; 内部格清空
    (set! b_i 0)
    (while (< b_i H) {
      (set! b_j 0)
      (while (< b_j W) {
        (set_item! g (gidx b_j b_i) M_EMPTY)
        (set! b_j (+ b_j 1)) })
      (set! b_i (+ b_i 1)) })
    (set! st_qpos 0)
    (set! st_qlen 0)
    (set! st_dlen 0)))

;; 界面层用完绘制表后调用：清空（代戳会在下一步 +1，无需清 pk 表）
(define paint_done (lambda () (set! st_plen 0)))

;; 入队是“集合语义”（每格最多在队列里出现一次），步内步外都能直接 addf!，
;; 因此不再需要“打开/关闭当前表”的配对动作（保留为空操作，调用点不必改）。
(define open_cur (lambda () 0))
(define close_cur (lambda () 0))

;; =============================================================================
;; 重力方位
;; =============================================================================

(define dadd
  (lambda (x n)
    (set! d_ta (+ x n))
    (if (> d_ta 7) (set! d_ta (- d_ta 8)) #f)
    d_ta))

(define dsub
  (lambda (x n)
    (set! d_ts (- x n))
    (if (< d_ts 0) (set! d_ts (+ d_ts 8)) #f)
    d_ts))

;; 由屏幕坐标系重力分量（x 右、y 下为正，见 am_native_Imu.h）量化出 8 方位。
;; tan(22.5°)=0.41421356 为判别阈值：竖直带 / 水平带 / 对角带。
(define set_gravity
  (lambda (gx gy)
    (set! d_gx gx)
    (set! d_gy gy)
    (set! d_ax (Math.abs gx))
    (set! d_ay (Math.abs gy))
    (if (< d_ay (* 0.41421356 d_ax))
        (set! d_new (if (> gx 0) 6 2))
        (if (< d_ax (* 0.41421356 d_ay))
            (set! d_new (if (> gy 0) 0 4))
            (if (> gy 0)
                (set! d_new (if (> gx 0) 7 1))
                (set! d_new (if (> gx 0) 5 3)))))
    (if (== d_new dv)
        #f
        {
          (set! dv d_new)
          ;; 重力方向变了：所有颗粒都要重新受力 —— 触发一次分片全扫
          (set! st_sweep 1)
        })
    (set! d_od (get_item OFF dv))
    (set! d_odl (get_item OFF (dadd dv 1)))
    (set! d_odr (get_item OFF (dsub dv 1)))
    (set! d_ol1 (get_item OFF (dadd dv 2)))
    (set! d_ol2 (get_item OFF (dsub dv 2)))
    (set! d_ou (get_item OFF (dadd dv 4)))
    (set! d_og1 (get_item OFF (dadd dv 3)))
    (set! d_og2 (get_item OFF (dsub dv 3)))))

;; =============================================================================
;; 规则表作用（频率低于移动，保留为函数）
;; =============================================================================

;; 对邻居 k 应用 RT[st_v][g[k]]
(define rule_at
  (lambda (k)
    (set! p_tv (get_item g k))
    (set! p_rc (get_item RT (+ (* st_v 10) p_tv)))
    (if (> p_rc 0)
        {
          (if (== p_rc R_IGNITE)
              {
                (set_item! g k M_FIRE)
                (set_item! lf k FIRE_LIFE)
                (addp! k) (tch k)
              }
              #f)
          (if (== p_rc R_STEAMIFY)
              {
                (set_item! g k M_STEAM)
                (set_item! lf k STEAM_LIFE)
                (addp! k) (tch k)
              }
              #f)
          (if (== p_rc R_DISSOLVE)
              {
                (set_item! g k M_EMPTY)
                (set_item! lf k 0)
                (addp! k) (tch k)
                (set! p_eat 1)
              }
              #f)
          (if (== p_rc R_GROW)
              {
                (set_item! g k M_PLANT)
                (set_item! lf k 0)
                (addp! k) (tch k)
              }
              #f)
          (if (== p_rc R_DILUTE)
              {
                (set_item! g k M_WATER)
                (set_item! lf k 0)
                (addp! k) (tch k)
              }
              #f)
          (if (== p_rc R_SMOTHER)
              {
                (set_item! g k M_EMPTY)
                (set_item! lf k 0)
                (addp! k) (tch k)
              }
              #f)
          (if (== p_rc R_BOIL)
              {
                (set_item! g k M_STEAM)
                (set_item! lf k STEAM_LIFE)
                (addp! k) (tch k)
                (set! st_cnt (- 1 st_cnt))
                (if (== st_cnt 1)
                    {
                      (set_item! g st_idx M_STEAM)
                      (set_item! lf st_idx STEAM_LIFE)
                      (addp! st_idx)
                      (tch st_idx)
                    }
                    #f)
              }
              #f)
        }
        #f)))

;; 每步抽查 2 个邻格：**方向轮换**（4 组、4 步覆盖全部 8 邻域），
;; 这样斜向接触（例如液体顺着斜角贴住石头）最终也能触发规则，而单步成本不变。
(define rule_pair
  (lambda ()
    (set! p_eat 0)
    (if (== p_rot 3) (set! p_rot 0) (set! p_rot (+ p_rot 1)))
    (if (== p_rot 0)
        {
          (rule_at (+ st_idx d_ol1))
          (rule_at (+ st_idx d_ol2))
        }
        (if (== p_rot 1)
            {
              (rule_at (+ st_idx d_od))
              (rule_at (+ st_idx d_ou))
            }
            (if (== p_rot 2)
                {
                  (rule_at (+ st_idx d_odl))
                  (rule_at (+ st_idx d_og2))
                }
                {
                  (rule_at (+ st_idx d_og1))
                  (rule_at (+ st_idx d_odr))
                })))))

;; =============================================================================
;; 火 / 蒸汽 / 酸 / 植物的每步行为（数量少，保留为函数）
;; =============================================================================

;; 火：生命递减 → 熄灭；未熄灭则八邻全查（接触即燃）+ 隔步上升
(define do_fire
  (lambda ()
    (set! p_l (- (get_item lf st_idx) 1))
    (if (<= p_l 0)
        {
          (set_item! g st_idx M_EMPTY)
          (set_item! lf st_idx 0)
          (addp! st_idx)
          (tch st_idx)
        }
        {
          (set_item! lf st_idx p_l)
          (set! p_eat 0)
          (rule_at (+ st_idx d_od))
          (rule_at (+ st_idx d_ou))
          (rule_at (+ st_idx d_ol1))
          (rule_at (+ st_idx d_ol2))
          (rule_at (+ st_idx d_odl))
          (rule_at (+ st_idx d_odr))
          (rule_at (+ st_idx d_og1))
          (rule_at (+ st_idx d_og2))
          (set! p_ok 0)
          (set! st_v (get_item g st_idx))     ; 规则可能已把本格改成别的材质
          (if (== st_v M_FIRE)
              {
                (set! st_cnt (- 1 st_cnt))
                (if (== st_cnt 1) (do_gas) #f)
                (if (== p_ok 0) (addf! st_idx) #f)   ; 没动 → 留在表里继续燃
              }
              #f)
        })))

;; 蒸汽：上升 + 生命递减（到期凝结成水）
(define do_steam
  (lambda ()
    (set! p_l (- (get_item lf st_idx) 1))
    (if (<= p_l 0)
        {
          (set_item! g st_idx M_WATER)
          (set_item! lf st_idx 0)
          (addp! st_idx)
          (tch st_idx)
        }
        {
          (set_item! lf st_idx p_l)
          (set! p_ok 0)
          (do_gas)
          (if (== p_ok 0) (addf! st_idx) #f)
        })))

;; 酸：生命递减（溶完自己也没了）+ 规则（溶解）+ 液体流动
(define do_acid
  (lambda ()
    (set! p_l (get_item lf st_idx))
    (rule_pair)
    (if (> p_eat 0)
        {
          (set! p_l (- p_l 1))
          (set_item! lf st_idx p_l)
        }
        #f)
    (if (<= p_l 0)
        {
          (set_item! g st_idx M_EMPTY)
          (set_item! lf st_idx 0)
          (addp! st_idx)
          (tch st_idx)
        }
        {
          (addf! st_idx)       ; 静止的酸也要每步查规则，否则贴着石头永远不溶
          (set! st_v (get_item g st_idx))
          (if (== st_v M_ACID) (do_liquid) #f)
        })))

;; 植物：静止，吸水生长（慢节拍自唤醒，保持“稳态近零成本”）
(define do_plant
  (lambda ()
    (rule_pair)
    (if (== st_grow 1) (addf! st_idx) #f)))

;; =============================================================================
;; 一个格子的调度
;; =============================================================================

(define step_cell
  (lambda ()
    (if (== (get_item mu st_idx) st_gen)
        ;; 本步已经更新过（刚被别的颗粒占用 / 本拍里被处理过）：
        ;; 不能丢，要**补排到下一步**再审 —— 否则刚落位的颗粒会永久冻结在半空
        ;; （实测：还给撒料让路时留下的悬空沙就是这个）。
        ;; md 保证同一步只补排一次（否则会在本步内反复“跳过→补排”空转）。
        (if (== (get_item md st_idx) st_gen)
            #f
            {
              (set_item! md st_idx st_gen)
              (adddef! st_idx)
            })
        {
          (set_item! mu st_idx st_gen)
          (set! st_v (get_item g st_idx))
          (if (== st_v M_SAND)
              (do_powder)
              (if (== st_v M_WATER)
                  {
                    (rule_pair)
                    (do_liquid)
                  }
                  (if (== st_v M_OIL)
                      (do_liquid)
                      (if (== st_v M_FIRE)
                          (do_fire)
                          (if (== st_v M_PLANT)
                              (do_plant)
                              (if (== st_v M_ACID)
                                  (do_acid)
                                  (if (== st_v M_STEAM)
                                      (do_steam)
                                      #f)))))))
        })))

;; =============================================================================
;; 分片全扫（换向 / 首次进入时唤醒所有非空格子）
;; =============================================================================

(define sweep_slice
  (lambda ()
    (set! s_i 0)
    (while (< s_i SLICE) {
      (if (< st_sweep NP)
          {
            (set! s_k (get_item g st_sweep))
            (if (> s_k 0)
                (if (< s_k M_WALL) (addf! st_sweep) #f)
                #f)
          }
          #f)
      (set! st_sweep (+ st_sweep 1))
      (set! s_i (+ s_i 1)) })
    (if (>= st_sweep NP) (set! st_sweep 0) #f)))

;; =============================================================================
;; 推进一“步”
;; =============================================================================

(define step
  (lambda ()
    (set! st_gen (+ st_gen 1))
    (set! st_pgen (+ st_pgen 1))
    (set! st_budget MAXPROC)
    (set! st_moves 0)
    (set! st_tk (+ st_tk 1))
    (set! st_grow (if (== (mod st_tk 8) 0) 1 0))
    (set! st_fire 0)
    (set! st_processed 0)
    ;; 上一步“跳过待审”的格子并入队尾（它们保证会被处理，且不会被本步的游标越过）
    (set! sd 0)
    (while (< sd st_dlen) {
      (addf! (get_item st_def sd))
      (set! sd (+ sd 1)) })
    (set! st_dlen 0)
    ;; **FIFO 队列 + 单步预算**：只处理 MAXPROC 个活跃格，剩下的留在队列里
    ;; （游标不清零），因此再剧烈的场景单步耗时也有上界；队列是集合语义
    ;; （每格最多出现一次），长度天然 ≤ NP，不会像“追加式”那样无限增长。
    (while (> st_budget 0)
      (if (>= st_qpos st_qlen)
          (set! st_budget 0)
          {
            (set! st_idx (get_item st_q st_qpos))
            (set_item! mk st_idx 0)          ; 出队
            (step_cell)
            (if (== st_v M_FIRE) (set! st_fire (+ st_fire 1)) #f)
            (set! st_qpos (+ st_qpos 1))
            (set! st_budget (- st_budget 1))
            (set! st_processed (+ st_processed 1))
          }))
    ;; 压缩：游标走过“一个 NP”的距离后，把剩余项搬回表头（摊还成本 ≈ 每格 2 次操作）
    (if (> st_qpos NP)
        {
          (set! st_w 0)
          (while (< st_qpos st_qlen) {
            (set_item! st_q st_w (get_item st_q st_qpos))
            (set! st_w (+ st_w 1))
            (set! st_qpos (+ st_qpos 1)) })
          (set! st_qpos 0)
          (set! st_qlen st_w)
        }
        #f)
    (set! st_pending (+ (- st_qlen st_qpos) st_dlen))
    (if (> st_sweep 0) (sweep_slice) #f)))

;; =============================================================================
;; 撒料 / 擦除 / 清空 / 初始场景
;; =============================================================================

;; 单个格子的写入（k 必须已判过可写）
(define put_cell
  (lambda (k mat)
    (set_item! g k mat)
    (if (== mat M_FIRE)
        (set_item! lf k FIRE_LIFE)
        (if (== mat M_STEAM)
            (set_item! lf k STEAM_LIFE)
            (if (== mat M_ACID)
                (set_item! lf k ACID_LIFE)
                (set_item! lf k 0))))
    (addp! k)
    (addf! k)
    (tch k)))

;; 直接放置一个格子（调试、单测、初始化场景用），并把它唤醒进活跃表
(define place
  (lambda (x y mat)
    (open_cur)
    (put_cell (gidx x y) mat)
    (close_cur)))

;; 唤醒全部非空格子（分片全扫）
(define wake_all (lambda () (set! st_sweep 1)))

;; 在网格坐标 (cx,cy) 处以半径 2 的圆盘撒料（mat=M_EMPTY 即擦除）
(define pour
  (lambda (cx cy mat)
    (open_cur)
    (set! q_y (- cy 2))
    (while (< q_y (+ cy 3)) {
      (set! q_x (- cx 2))
      (while (< q_x (+ cx 3)) {
        (if (and (and (>= q_x 0) (< q_x W)) (and (>= q_y 0) (< q_y H)))
            {
              (set! q_d (+ (* (- q_x cx) (- q_x cx)) (* (- q_y cy) (- q_y cy))))
              (if (<= q_d 5)
                  {
                    (set! q_tv (get_item g (gidx q_x q_y)))
                    (set! q_coin (- 1 q_coin))
                    (if (== mat M_EMPTY)
                        {
                          ;; 擦除：整盘覆盖（除了边框墙）
                          (if (and (> q_tv 0) (< q_tv M_WALL))
                              (put_cell (gidx q_x q_y) M_EMPTY)
                              #f)
                        }
                        {
                          ;; 撒料：空格或自己能沉降进去的格子（沙可以灌进水里），
                          ;; 且隔一格撒一格（硬币），避免一次糊死一大片
                          (if (== q_coin 1)
                              (if (== q_tv M_EMPTY)
                                  (put_cell (gidx q_x q_y) mat)
                                  (if (== (get_item MV (+ (* mat 10) q_tv)) 1)
                                      (put_cell (gidx q_x q_y) mat)
                                      #f))
                              #f)
                        })
                  }
                  #f)
            }
            #f)
        (set! q_x (+ q_x 1)) })
      (set! q_y (+ q_y 1)) })
    (close_cur)))

;; 清空整个世界（内部格置空）
(define clear_all
  (lambda ()
    (set! b_i 0)
    (while (< b_i H) {
      (set! b_j 0)
      (while (< b_j W) {
        (set_item! g (gidx b_j b_i) M_EMPTY)
        (set_item! lf (gidx b_j b_i) 0)
        (set! b_j (+ b_j 1)) })
      (set! b_i (+ b_i 1)) })
    (set! st_qpos 0)
    (set! st_qlen 0)
    (set! st_dlen 0)
    (set! st_sweep 0)))

;; 初始演示场景：石地板 + 左侧沙堆 + 右侧水池/油层 + 几株植物 + 石台阶
(define scene_init
  (lambda ()
    (clear_all)
    ;; 石地板（最下两行）
    (set! b_j 0)
    (while (< b_j W) {
      (set_item! g (gidx b_j (- H 1)) M_STONE)
      (set_item! g (gidx b_j (- H 2)) M_STONE)
      (set! b_j (+ b_j 1)) })
    ;; 沙堆（左）：高度 26 - |x-22| 的三角形
    (set! b_j 6)
    (while (< b_j 39) {
      (set! b_i (- 26 (Math.abs (- b_j 22))))
      (if (> b_i 0)
          {
            (set! b_i (- (- H 3) b_i))
            (while (< b_i (- H 2)) {
              (set_item! g (gidx b_j b_i) M_SAND)
              (set! b_i (+ b_i 1)) })
          }
          #f)
      (set! b_j (+ b_j 1)) })
    ;; 水池（右）
    (set! b_j 52)
    (while (< b_j W) {
      (set! b_i 66)
      (while (< b_i (- H 2)) {
        (set_item! g (gidx b_j b_i) M_WATER)
        (set! b_i (+ b_i 1)) })
      (set! b_j (+ b_j 1)) })
    ;; 油层：浮在水面上
    (set! b_j 70)
    (while (< b_j 96) {
      (set_item! g (gidx b_j 65) M_OIL)
      (set_item! g (gidx b_j 64) M_OIL)
      (set! b_j (+ b_j 1)) })
    ;; 植物：沙堆向阳面每隔 6 格长两节
    (set! b_j 8)
    (while (< b_j 34) {
      (if (== (mod b_j 6) 0)
          {
            (set! b_i (- (- H 3) (- 26 (Math.abs (- b_j 22)))))
            (if (> b_i 2)
                {
                  (set_item! g (gidx b_j (- b_i 1)) M_PLANT)
                  (set_item! g (gidx b_j (- b_i 2)) M_PLANT)
                }
                #f)
          }
          #f)
      (set! b_j (+ b_j 1)) })
    ;; 石台阶（水池里的一段挡板，让水流有戏）
    (set! b_j 60)
    (while (< b_j 66) {
      (set_item! g (gidx b_j 74) M_STONE)
      (set! b_j (+ b_j 1)) })
    ;; 唤醒全部格子（分片全扫），让场景自然落定
    (set! st_qpos 0)
    (set! st_qlen 0)
    (set! st_dlen 0)
    (set! st_sweep 1)))

;; =============================================================================
;; 加载即建表（不建场景 —— 由界面层在启动时决定调用 scene_init 还是 clear_all）
;; =============================================================================
(build)