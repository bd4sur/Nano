;; ============================================================================
;; 贪吃蛇纯逻辑库（Snake Core）—— 供 app/snake.scm、app/lib/snake_dqn.scm 与回归测试共用
;;
;; 职责边界：
;;   · 只管**规则与观测**：场地、蛇身、食物、碰撞、成长、计分、奖励、观测编码；
;;   · 不碰界面（无 Gfx/Touch），也不碰任何学习算法（那是 app/lib/snake_dqn.scm）。
;;   于是"规则"可以在无头宿主/测试台里单独跑（tools/e2e/native/snake_*.scm），
;;   界面与智能体都只通过本模块的公开接口读写状态。
;;
;; 规则（与 tools/train_snake_dqn.py 的 numpy 版逐条一致；对拍脚本
;; tools/e2e/native/snake_parity.scm 用同一份权重跑同一组固定食物，两边逐行比对）：
;;   · 场地 12x12，格索引 = y*12 + x；出生长度 3，头在 (5,6) 朝右，身体在 (4,6)(3,6)
;;   · 每步先按动作转向（0=直行 1=右转 2=左转；方向序 0上 1右 2下 3左，顺时针），再前进一格
;;   · 撞墙或撞到身体（含当前尾巴格）即死亡，蛇不动
;;   · 吃到食物：长度 +1、分数 +1、食物重新随机撒在空格上；否则尾巴格腾空
;;   · 食物无处可撒（场地填满）= 通关，置 dead=1（界面显示"通关"）
;;   · 奖励：吃 +10、死 −10、每步 −0.01，外加**位势塑形**
;;       r += γ·Φ(s') − Φ(s)，Φ(s) = −0.1 · manhattan(头, 食物)
;;     （Ng et al. 1999 的位势塑形：不改变最优策略，但把"靠近食物"变成可逐步获得的
;;       正奖励——没有它，早期探索会退化成"贴墙绕圈苟活"的局部最优）
;;
;; 观测编码（NI = 28，与离线训练脚本的 --feat v2 一致）：
;;   [0..24] 以蛇头为中心的 **5x5 窗口**，按"前进方向朝上、右手方向朝右"旋转：
;;            0 = 空地；1/3 = 身体；2/3 = 墙（越界）；1 = 食物（原值 3 再除 3 归一）
;;            索引 = (i+2)*5 + (j+2)，i = 前向偏移(-2..2)，j = 右向偏移(-2..2)
;;   [25] 食物在**蛇体系前向**的偏移 / 12（可正可负）
;;   [26] 食物在**蛇体系右向**的偏移 / 12
;;   [27] 长度 / 144
;;   旋转窗口让同一局面在任何朝向下编码相同，网络因此不必自己学分方向；实测比经典
;;   11 维编码（只看相邻三格危险 + 四向食物）强不少（离线贪心均分 21 → 26+）。
;;
;; 蛇身用**环形缓冲**存格子坐标（bx/by 各 144 槽，hslot 为头槽位），
;; 每步只写一格、清一格，移动是 O(1)；occ 提供 O(1) 碰撞查询。
;;
;; 解释器约束（manual.md）：算术/比较恒二参数；无一元取负 → (- 0 x)；
;; 无 let/list/map/append；块内不 define（define 只写在 lambda 体顶层）；
;; / 与 mod 恒浮点（下标一律 Math.floor 后再用）；整数有效位只有 27 位（别放大数）。
;; ============================================================================

(native Math)

;; ---- 场地与规则常量 -------------------------------------------------------

(define G 12)                   ; 场地边长（12x12）
(define NC 144)                 ; G*G
(define NI 28)                  ; 观测维数
(define NO 3)                   ; 动作数

;; 方向表：0=上 1=右 2=下 3=左（顺时针）。右转 = (d+1) mod 4，左转 = (d+3) mod 4
(define DX '(0 1 0 -1))
(define DY '(-1 0 1 0))

(define INIT_LEN 3)
(define INIT_HX 5)
(define INIT_HY 6)
(define INIT_DIR 1)

(define R_FOOD 10.0)
(define R_DEATH -10.0)
(define R_STEP -0.01)
(define SHAPE 0.1)              ; 势函数系数：Φ = -SHAPE * manhattan(head, food)
(define GAMMA 0.9)              ; 塑形与 DQN 用同一个折扣

;; ---- 游戏状态（导入方以 S.名字 直接读）------------------------------------

(define occ '())        ; NC 格占用表：0 空 / 1 身体
(define bx '())         ; 蛇身环：每槽的 x
(define by '())         ; 蛇身环：每槽的 y
(define hslot 0)        ; 蛇头所在环槽位
(define hx 0)           ; 蛇头 x（缓存，绘制/观测都用）
(define hy 0)
(define dir INIT_DIR)   ; 当前朝向
(define blen 0)         ; 身体长度
(define score 0)        ; 吃到的食物数
(define steps 0)        ; 本局步数
(define dead 0)         ; 1 = 本局结束（撞死或通关）
(define won 0)          ; 1 = 通关（场地填满）
(define food 0)         ; 食物格索引（-1 = 无处可撒）
(define obs_v '())      ; 观测缓冲（复用，不分配）

;; ---- 模块级临时量（热路径局部量一律提到模块级，见 doc/AGENTS.md）---------

(define mk_i 0)
(define rs_start 0) (define rs_k 0) (define rs_ok 0) (define rs_i 0)
(define bd_i 0) (define bd_slot 0)
(define st_nx 0) (define st_ny 0) (define st_tail 0) (define st_dold 0) (define st_dnew 0)
(define st_ri 0) (define st_r 0.0) (define st_cell 0)
(define ob_ri 0) (define ob_fx 0) (define ob_fy 0) (define ob_i 0) (define ob_j 0)
(define ob_idx 0) (define ob_x 0) (define ob_y 0) (define ob_o 0.0)

;; ---- 一次性建表（'() 是模块级静态槽，故一律用 (cdr '(0)) 现造新表）--------

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

(set! occ (mk_list NC))
(set! bx (mk_list NC))
(set! by (mk_list NC))
(set! obs_v (mk_list NI))

;; ---- 基本查询 -------------------------------------------------------------

;; (x,y) 是否不可进入（出界或已有身体）：1 / 0
(define blocked
  (lambda (x y)
    (if (< x 0)
        1
        (if (< y 0)
            1
            (if (>= x G)
                1
                (if (>= y G)
                    1
                    (get_item occ (+ (* y G) x))))))))

;; 蛇头到食物的曼哈顿距离（无食物时给 0，仅塑形用）
(define manhattan
  (lambda ()
    (if (< food 0)
        0
        (+ (Math.abs (- (mod food G) hx))
           (Math.abs (- (Math.floor (/ food G)) hy))))))

;; 蛇身第 k 段（k=0 是头）所在环槽位
(define slot_of
  (lambda (k)
    (mod (- (+ hslot NC) k) NC)))

;; ---- 撒食物：随机起址 + 线性探测；无处可撒则 food = -1 -------------------

(define spawn_food
  (lambda () {
    (set! rs_start (Math.floor (* NC (Math.random))))
    (set! rs_ok 0)
    (set! rs_k 0)
    (while (< rs_k NC)
      {
        (if (== rs_ok 0)
            {
              (set! rs_i (mod (+ rs_start rs_k) NC))
              (if (== (get_item occ rs_i) 0)
                  {
                    (set! food rs_i)
                    (set! rs_ok 1)
                  }
                  #f)
            }
            #f)
        (set! rs_k (+ rs_k 1))
      })
    (if (== rs_ok 0) (set! food -1))
  }))

;; ---- 开新局 ---------------------------------------------------------------

(define reset
  (lambda () {
    ;; 清场 + 摆身体（从尾巴到头依次填环槽 0..INIT_LEN-1，头占 INIT_LEN-1）
    (set! bd_i 0)
    (while (< bd_i NC)
      {
        (set_item! occ bd_i 0)
        (set_item! bx bd_i 0)
        (set_item! by bd_i 0)
        (set! bd_i (+ bd_i 1))
      })
    (set! hx INIT_HX)
    (set! hy INIT_HY)
    (set! dir INIT_DIR)
    (set! blen INIT_LEN)
    (set! hslot (- INIT_LEN 1))
    (set! bd_i 0)
    (while (< bd_i INIT_LEN)
      {
        (set! bd_slot (- (- INIT_LEN 1) bd_i))
        (set_item! bx bd_slot (- hx (* bd_i (get_item DX dir))))
        (set_item! by bd_slot (- hy (* bd_i (get_item DY dir))))
        (set_item! occ (+ (* (- hy (* bd_i (get_item DY dir))) G)
                          (- hx (* bd_i (get_item DX dir))))
                    1)
        (set! bd_i (+ bd_i 1))
      })
    (set! score 0)
    (set! steps 0)
    (set! dead 0)
    (set! won 0)
    (set! food 0)
    (spawn_food)
  }))

;; ---- 观测编码（写入并返回复用缓冲 obs_v）---------------------------------

(define obs
  (lambda () {
    (set! ob_ri (if (== dir 3) 0 (+ dir 1)))
    (set! ob_fx hx)
    (set! ob_fy hy)
    (if (>= food 0)
        {
          (set! ob_fx (mod food G))
          (set! ob_fy (Math.floor (/ food G)))
        }
        #f)
    (set! ob_idx 0)
    (set! ob_i (- 0 2))
    (while (<= ob_i 2)
      {
        (set! ob_j (- 0 2))
        (while (<= ob_j 2)
          {
            (set! ob_x (+ hx (+ (* ob_i (get_item DX dir))
                                (* ob_j (get_item DX ob_ri)))))
            (set! ob_y (+ hy (+ (* ob_i (get_item DY dir))
                                (* ob_j (get_item DY ob_ri)))))
            ;; 判定顺序必须与离线训练脚本一致：墙 → 食物 → 身体
            (if (or (< ob_x 0)
                    (or (< ob_y 0) (or (>= ob_x G) (>= ob_y G))))
                (set! ob_o (/ 2.0 3.0))            ; 墙
                (if (and (== ob_x ob_fx) (== ob_y ob_fy))
                    (set! ob_o 1.0)                ; 食物
                    (if (== (get_item occ (+ (* ob_y G) ob_x)) 1)
                        (set! ob_o (/ 1.0 3.0))    ; 身体
                        (set! ob_o 0.0))))
            (set_item! obs_v ob_idx ob_o)
            (set! ob_idx (+ ob_idx 1))
            (set! ob_j (+ ob_j 1))
          })
        (set! ob_i (+ ob_i 1))
      })
    ;; 食物在蛇体系的分量 + 长度占比
    (set_item! obs_v 25
               (/ (+ (* (- ob_fx hx) (get_item DX dir))
                     (* (- ob_fy hy) (get_item DY dir)))
                  G))
    (set_item! obs_v 26
               (/ (+ (* (- ob_fx hx) (get_item DX ob_ri))
                     (* (- ob_fy hy) (get_item DY ob_ri)))
                  G))
    (set_item! obs_v 27 (/ blen NC))
    obs_v
  }))

;; ---- 走一步：动作 0/1/2 = 直行/右转/左转；返回本步奖励 -------------------

(define step
  (lambda (a) {
    ;; 转向
    (if (== a 1) (set! dir (if (== dir 3) 0 (+ dir 1))) #f)
    (if (== a 2) (set! dir (if (== dir 0) 3 (- dir 1))) #f)
    (set! st_ri (if (== dir 3) 0 (+ dir 1)))
    (set! st_dold (manhattan))
    (set! st_nx (+ hx (get_item DX dir)))
    (set! st_ny (+ hy (get_item DY dir)))
    (set! steps (+ steps 1))
    (if (== (blocked st_nx st_ny) 1)
        { ; 撞死：蛇不动
          (set! dead 1)
          (set! st_r R_DEATH)
        }
        { ; 前进
          (set! hslot (if (== hslot (- NC 1)) 0 (+ hslot 1)))
          (set_item! bx hslot st_nx)
          (set_item! by hslot st_ny)
          (set! hx st_nx)
          (set! hy st_ny)
          (set! st_cell (+ (* st_ny G) st_nx))
          (set_item! occ st_cell 1)
          (set! st_r R_STEP)
          (if (== st_cell food)
              { ; 吃到：变长 + 撒新食物
                (set! score (+ score 1))
                (set! blen (+ blen 1))
                (set! st_r (+ st_r R_FOOD))
                (if (== blen NC)
                    { ; 场地填满 = 通关
                      (set! food -1)
                      (set! dead 1)
                      (set! won 1)
                    }
                    (spawn_food))
              }
              { ; 没吃到：腾出尾巴格
                (set! st_tail (mod (+ (- NC blen) hslot) NC))
                (set_item! occ (+ (* (get_item by st_tail) G) (get_item bx st_tail)) 0)
              })
        })
    ;; 位势塑形：r += γΦ(s') − Φ(s)，Φ = −SHAPE·manhattan
    (set! st_dnew (manhattan))
    (set! st_r (+ st_r (- (* SHAPE st_dold) (* GAMMA (* SHAPE st_dnew)))))
    st_r
  }))

;; ---- 供对拍/测试用的确定性食物设置 ---------------------------------------

(define set_food
  (lambda (i)
    (set! food i)))
