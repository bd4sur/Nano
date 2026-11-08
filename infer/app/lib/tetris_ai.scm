;; ============================================================================
;; 俄罗斯方块自动游玩（Tetris AI）—— app/tetris.scm 的「AI 自动」模式用
;;
;; 职责划分：
;;   · 本模块只**看** tetris_core 的公开状态并**调**它的动作接口
;;     （T.field_at / T.fill_cells / T.move / T.rotate_piece / T.hard_drop /
;;      T.reset_fall），不复制任何游戏规则——落点、踢墙、消行、计分全由核心模块负责；
;;   · 决策（think）与执行（tick）分离：think 每块只算一次，tick 每拍只做一个动作，
;;     于是"选哪"与"怎么走过去"互不干扰，也不会有"一拍里跑几万条指令卡住画面"。
;;
;; 决策模型（一次全枚举，不做搜索）：
;;   对 4 个旋转态 × 全部合法横位，用"每列表面行号"直接算出**落地行**
;;   （落点 = min over 格子 (surf[列] - 1 - 相对 y)，与 T.collide 的判定等价：
;;    列表面之上的格子一定不冲突，够到表面才冲突），再对落定后的棋盘算 6 项特征：
;;
;;       score = -4.5001588·落地高度 + 3.4181268·消行×消掉的方块格数
;;               -3.2178883·行变换 -9.3486953·列变换 -7.8992654·空洞数
;;               -2.0·井格数
;;
;;   （前 5 项是 El-Tetris / Dellacherie 的经典权重——"能一直玩下去"的公开配方：
;;    落地高度 = 方块最低格离底边的高度，行变换 = 行内相邻格"有/无"变化次数
;;    （含两侧墙），列变换 = 列内上下相邻格变化次数（含地板），空洞 = 被方块压住
;;    的空格。它们共同刻画"表面平整、不留洞、不堆高"，比只看高度和凹凸度的四特征
;;    配方稳得多——实测四特征配方在 84 行左右被自己挖的空洞顶死；第 6 项"井格"
;;    是本模块加的近似项，少了它会留一条 17 层深的空列一直不填、573 行被顶死，
;;    加上后实测连续 5000+ 行不死。）
;;
;;   棋盘用四张小表表示（每块重建一次，200 格）：每列表面行号 surf、每列已填数 nfill、
;;   每行位掩码 rmask、每列位掩码 cmask。候选落定只影响被压到的 ≤4 行与 ≤4 列，
;;   故行/列变换、空洞、井格、消行全走增量；单块决策约 40 个候选、约 7 万条 VM 指令
;;   （约 1.5ms，且一拍只做一个动作——决策只落在换块后的那一拍上）。
;;
;;   位运算用 Math.bit_*（参数必须是无符号整数；wasm32 的逻辑位宽 27 位，
;;   行掩码 10 位、列掩码 20 位都在范围内）。popcount 用一张 4096 项查表（模块加载时建一次）。
;;
;; 与 core 的约定：
;;   · T.serial —— 方块序号（每生成一块 +1）。tick 用它判断"该重算计划了"；
;;   · 计划里的横位 plan_px 就是 T.px（包围盒左上角），执行时逐格走过去，
;;     到位后 T.hard_drop 一次落底（并 T.reset_fall 清掉重力计时）。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数；无一元取负（写 (- 0 x)）；
;; 无 list/let/map；cond 每子句单个表达式；块内不 define；/ 与 mod 恒浮点；
;; 哨兵大数一律写成浮点（整数有效位只有 27 位，-1e9 会绕成正数）；
;; '() 静态槽只在模块加载时 push 一次，此后只用 set_item! 原地改写；
;; 模块级临时量按"谁写谁读"分族命名（bd_/sn_/rf_/cf_/wm_/pr_/ea_/es_/pl_/er_/tk_），
;; 跨函数调用链上绝不共用（见 doc/AGENTS.md 的俄罗斯方块踩坑记录）。
;; ============================================================================

(native Math)

(import T "app/lib/tetris_core.scm")

;; ---- 评估权重（El-Tetris / Dellacherie + 井格项）-------------------------
;; 只调这 6 个数。空洞权重最大（-7.9）：宁可堆高也别挖洞；列变换次之（-9.3，
;; 刻画"竖直方向的坑道"）；落地高度（-4.5）压住整体堆高；消行是唯一的正项；
;; 井格（-2.0）专治"留一条深空列不填"。

(define W_LAND -4.500158825082766)     ; 落地高度（方块最低格离底边的行数）
(define W_ERODE 3.4181268101392694)    ; 消行数 × 该行里被消掉的方块格数
(define W_ROW -3.2178882868487753)     ; 行变换（含左右墙）
(define W_COL -9.348695305445199)      ; 列变换（含地板）
(define W_HOLES -7.899265427351652)    ; 空洞数
(define W_WELL -2.0)                   ; 井格数（左右都被填住的空格；见 well_mask）
;; 井（well）这一项 El-Tetris 按深度加权（1+2+…+d）；本模块改成**井格计数**
;; （线性，见 well_mask）：两者对"深井"都是单调惩罚，而计数可以完全增量算
;; （只重算被压到的 ≤4 行，约 50 条指令/候选），按深度加权则要扫整列。
;; 少了这一项，AI 会留一条 17 层深的空列一直不填，最后把自己顶死（实测 573 行）。

;; ---- 查表 ----------------------------------------------------------------

(define pc '())       ; pc[x] = popcount(x)，x ∈ [0, 4096)
(define bits '())     ; bits[i] = 2^i（i ∈ [0, 20)）——见下方"位运算封装"的说明

;; ---- 棋盘摘要（每次决策重建一次）-----------------------------------------

(define surf '())     ; 每列最高的已填格行号；空列 = ROWS（10 项）
(define nfill '())    ; 每列已填格数（10 项）
(define rmask '())    ; 每行位掩码：bit c = 第 c 列已填（20 项）
(define cmask '())    ; 每列位掩码：bit r = 第 r 行已填（10 项）

;; ---- 候选落定后的增量（每次候选重置）-------------------------------------

(define tcnt '())     ; 候选在本列新增的格数（10 项）
(define tmin '())     ; 候选在本列新增格的最小行号（10 项）
(define tbits '())    ; 候选在本列新增格的位掩码（10 项）

;; ---- 旋转态几何（每次决策按当前方块重建一次）-----------------------------

(define pax '())      ; 4 个旋转态 × 4 格：相对 x（索引 = rt*4 + i，16 项）
(define pay '())      ; 同上，相对 y（0..3）
(define pmay '())     ; 每个旋转态的 max(相对 y)（4 项）——落地高度用

;; ---- 计划（think 写，tick 读）--------------------------------------------

(define plan_ok 0)      ; 是否有有效计划
(define plan_rt 0)      ; 目标旋转态
(define plan_px 0)      ; 目标包围盒左上角 x
(define plan_serial -1) ; 计划对应的方块序号
(define plan_py 0)      ; 计划的目标落地行（执行时据此判断"还剩几格"）
(define plan_score 0)   ; 选中方案的分数（诊断用）

;; ---- 模块级临时量（按函数分族，避免跨调用链串味）-------------------------

(define bd_i 0)                                                    ; build
(define sn_c 0) (define sn_r 0)
(define sn_rt 0) (define sn_ct 0) (define sn_holes 0) (define sn_wc 0)   ; snapshot
(define wm_t 0)                                                     ; well_mask
(define rf_y 0) (define rf_t 0)                                    ; row_trans
(define cf_t 0) (define cf_x19 0)                                  ; col_trans
(define pr_rt 0) (define pr_i 0)                                 ; prep_rot
(define ea_i 0) (define ea_c 0) (define ea_v 0) (define ea_py 0) (define ea_ok 0)
(define es_c 0) (define es_r 0) (define es_i 0) (define es_k 0) (define es_n 0)
(define es_d 0) (define es_rt 0) (define es_ct 0) (define es_holes 0)
(define es_clr 0) (define es_erode 0) (define es_s 0) (define es_found 0) (define es_wc 0)
(define es_nr 0) (define es_land 0)                                ; eval_score
(define pl_best 0) (define pl_brt 0) (define pl_bx 0) (define pl_bpy 0) (define pl_rt 0) ; think
(define er_minx 0) (define er_maxx 0) (define er_x 0) (define er_i 0) (define er_v 0) ; eval_rot
(define tk_old 0) (define tk_d 0) (define tk_acted 0) (define tk_n 0)  ; tick / act_once
(define rw_row '()) (define rw_bits '())                            ; eval_score 的行去重表（各 4 项）

;; ---- 表构建（只做一次；'() 静态槽此后只用 set_item! 改写）----------------

(define build
  (lambda () {
    ;; popcount 表：pc[i] = pc[i/2] + (i&1)，4096 项（行变换用到 12 位、列变换拆成两段）
    (set! pc '())
    (push pc 0)
    (set! bd_i 1)
    (while (< bd_i 4096) {
      (push pc (+ (get_item pc (Math.bit_lshr bd_i 1)) (Math.bit_and bd_i 1)))
      (set! bd_i (+ bd_i 1))
    })

    ;; 位值表：Math.bit_shl 的移位计数要求**无符号整数**，而解释器的 (- 0 x) 与
    ;; 负数常量一律是有符号整数（py=-1 起手、横位 x 从 -minx 起算都是），
    ;; 把这种值当移位计数传进去会静默失败（[Runtime] 指令执行异常，无报错文本）。
    ;; 故移位计数一律走查表，掩码本身则保持无符号。
    (set! bits '())
    (push bits 1)
    (set! bd_i 1)
    (while (< bd_i 20) {
      (push bits (Math.bit_shl (get_item bits (- bd_i 1)) 1))
      (set! bd_i (+ bd_i 1))
    })

    (set! surf '()) (set! nfill '()) (set! tcnt '()) (set! tmin '()) (set! tbits '())
    (set! bd_i 0)
    (while (< bd_i T.COLS) {
      (push surf T.ROWS)
      (push nfill 0)
      (push tcnt 0)
      (push tmin T.ROWS)
      (push tbits 0)
      (set! bd_i (+ bd_i 1))
    })

    (set! rmask '())
    (set! bd_i 0)
    (while (< bd_i T.ROWS) {
      (push rmask 0)
      (set! bd_i (+ bd_i 1))
    })

    (set! cmask '())
    (set! bd_i 0)
    (while (< bd_i T.COLS) {
      (push cmask 0)
      (set! bd_i (+ bd_i 1))
    })

    (set! pax '()) (set! pay '())
    (set! bd_i 0)
    (while (< bd_i 16) {
      (push pax 0)
      (push pay 0)
      (set! bd_i (+ bd_i 1))
    })

    (set! pmay '())
    (set! bd_i 0)
    (while (< bd_i 4) {
      (push pmay 0)
      (set! bd_i (+ bd_i 1))
    })

    (set! rw_row '()) (set! rw_bits '())
    (set! bd_i 0)
    (while (< bd_i 4) {
      (push rw_row 0)
      (push rw_bits 0)
      (set! bd_i (+ bd_i 1))
    })
  }))

(build)

;; ============================================================================
;; 位运算封装
;; ============================================================================

;; 井格掩码：一行的空格中，左右**都被填住**（两侧墙视为已填）的那些格。
;;   空 = m ^ 1023；左邻填 = (m<<1)|1；右邻填 = (m>>1)|512（bit9 = 右墙）
;;   （下面的 1023 / 512 / 2049 / 4095 / 127 / 19 都按核心固定的 10x20 场地写死，
;;    换场地尺寸时这几处要一起改。）
(define well_mask
  (lambda (m) {
    (set! wm_t (Math.bit_and (Math.bit_xor m 1023)
                             (Math.bit_or (Math.bit_shl m 1) 1)))
    (Math.bit_and wm_t (Math.bit_or (Math.bit_lshr m 1) 512))
  }))

;; 行变换：一行内相邻格"有/无"的变化次数（两侧墙视为已填）。
;;   y = 墙 | 格子 | 墙（12 位），变换数 = popcount(y ^ (y>>1))
(define row_trans
  (lambda (m) {
    (set! rf_y (Math.bit_or (Math.bit_shl m 1) 2049))   ; 2049 = 左墙 bit0 | 右墙 bit11
    (set! rf_t (Math.bit_xor rf_y (Math.bit_lshr rf_y 1)))
    (get_item pc rf_t)
  }))

;; 列变换：一列内上下相邻格的变化次数（地板视为已填，顶上空视为空）。
;;   t = x ^ (x>>1)：bit r (r<19) = x[r] ^ x[r+1]，bit19 = x[19]；再补地板那一处
(define col_trans
  (lambda (x) {
    (set! cf_t (Math.bit_xor x (Math.bit_lshr x 1)))
    (set! cf_x19 (Math.bit_and (Math.bit_lshr x 19) 1))
    (+ (+ (get_item pc (Math.bit_and cf_t 4095))
          (get_item pc (Math.bit_and (Math.bit_lshr cf_t 12) 127)))
       (- 1 cf_x19))
  }))

;; ============================================================================
;; 棋盘摘要：从 T 的场地取出"每列表面行号 / 每列已填数 / 每行位掩码 / 每列位掩码"
;;   一次 200 格扫描（每块一次，量级可忽略），后面 40 个候选全靠它做增量计算。
;; ============================================================================

(define snapshot
  (lambda () {
    (set! sn_c 0)
    (while (< sn_c T.COLS) {
      (set_item! surf sn_c T.ROWS)
      (set_item! nfill sn_c 0)
      (set_item! cmask sn_c 0)
      (set_item! tcnt sn_c 0)
      (set_item! tmin sn_c T.ROWS)
      (set_item! tbits sn_c 0)
      (set! sn_c (+ sn_c 1))
    })
    (set! sn_r 0)
    (while (< sn_r T.ROWS) {
      (set_item! rmask sn_r 0)
      (set! sn_r (+ sn_r 1))
    })

    (set! sn_r 0)
    (while (< sn_r T.ROWS) {
      (set! sn_c 0)
      (while (< sn_c T.COLS) {
        (if (not (== (T.field_at sn_c sn_r) 0)) {
          (set_item! nfill sn_c (+ (get_item nfill sn_c) 1))
          ;; 自上而下扫描：第一次命中即该列最高的已填格
          (if (== (get_item surf sn_c) T.ROWS) (set_item! surf sn_c sn_r))
          (set_item! cmask sn_c (Math.bit_or (get_item cmask sn_c) (get_item bits sn_r)))
          (set_item! rmask sn_r (Math.bit_or (get_item rmask sn_r) (get_item bits sn_c)))
        })
        (set! sn_c (+ sn_c 1))
      })
      (set! sn_r (+ sn_r 1))
    })

    ;; 三项基准值（候选落定后只对"被压到的行/列"做增量修正）
    (set! sn_rt 0)
    (set! sn_r 0)
    (while (< sn_r T.ROWS) {
      (set! sn_rt (+ sn_rt (row_trans (get_item rmask sn_r))))
      (set! sn_r (+ sn_r 1))
    })

    (set! sn_wc 0)
    (set! sn_r 0)
    (while (< sn_r T.ROWS) {
      (set! sn_wc (+ sn_wc (get_item pc (well_mask (get_item rmask sn_r)))))
      (set! sn_r (+ sn_r 1))
    })

    (set! sn_ct 0)
    (set! sn_holes 0)
    (set! sn_c 0)
    (while (< sn_c T.COLS) {
      (set! sn_ct (+ sn_ct (col_trans (get_item cmask sn_c))))
      ;; 空洞 = 该列"表面以下的格子数"减去已填数（= 表面以下的空格数）
      (set! sn_holes (+ sn_holes (- (- T.ROWS (get_item surf sn_c)) (get_item nfill sn_c))))
      (set! sn_c (+ sn_c 1))
    })
  }))

;; ============================================================================
;; 旋转态几何：把当前方块四个旋转态的 4 个格子坐标抄进本模块（不碰 T 的临时量）
;; ============================================================================

(define prep_rot
  (lambda (pc_piece) {
    (set! pr_rt 0)
    (while (< pr_rt 4) {
      (T.fill_cells pc_piece pr_rt)     ; 写入 T.cells_x / T.cells_y（界面层每拍都会重算，无残留）
      (set! pr_i 0)
      (while (< pr_i 4) {
        (set_item! pax (+ (* pr_rt 4) pr_i) (get_item T.cells_x pr_i))
        (set_item! pay (+ (* pr_rt 4) pr_i) (get_item T.cells_y pr_i))
        (set! pr_i (+ pr_i 1))
      })
      (set! pr_rt (+ pr_rt 1))
    })
    ;; 每个旋转态的 max(y)：落地高度用（越靠下越差）
    (set! pr_rt 0)
    (while (< pr_rt 4) {
      (set_item! pmay pr_rt 0)
      (set! pr_i 0)
      (while (< pr_i 4) {
        (if (> (get_item pay (+ (* pr_rt 4) pr_i)) (get_item pmay pr_rt))
            (set_item! pmay pr_rt (get_item pay (+ (* pr_rt 4) pr_i))))
        (set! pr_i (+ pr_i 1))
      })
      (set! pr_rt (+ pr_rt 1))
    })
  }))

;; ============================================================================
;; 候选打分
;; ============================================================================

;; 落定后算 6 项特征并更新最优解
(define eval_score
  (lambda (rt x0 py) {
    ;; ① 清空增量表（10 列）与行去重表（≤4 行）
    (set! es_c 0)
    (while (< es_c T.COLS) {
      (set_item! tcnt es_c 0)
      (set_item! tmin es_c T.ROWS)
      (set_item! tbits es_c 0)
      (set! es_c (+ es_c 1))
    })
    (set! es_nr 0)

    ;; ② 逐格登记：列（格数/最上行/位掩码）+ 行（位掩码，按行去重；y<0 的格子不入场）
    (set! es_i 0)
    (while (< es_i 4) {
      (set! es_r (+ py (get_item pay (+ (* rt 4) es_i))))
      (if (>= es_r 0) {
        (set! es_c (+ x0 (get_item pax (+ (* rt 4) es_i))))
        (set_item! tcnt es_c (+ (get_item tcnt es_c) 1))
        (set_item! tbits es_c (Math.bit_or (get_item tbits es_c) (get_item bits es_r)))
        (if (< es_r (get_item tmin es_c)) (set_item! tmin es_c es_r))
        ;; 行去重：同一行只登记一次，把本格并进位掩码
        (set! es_found 0)
        (set! es_k 0)
        (while (< es_k es_nr) {
          (if (== (get_item rw_row es_k) es_r) {
            (set_item! rw_bits es_k (Math.bit_or (get_item rw_bits es_k) (get_item bits es_c)))
            (set! es_found 1)
          })
          (set! es_k (+ es_k 1))
        })
        (if (== es_found 0) {
          (set_item! rw_row es_nr es_r)
          (set_item! rw_bits es_nr (get_item bits es_c))
          (set! es_nr (+ es_nr 1))
        })
      })
      (set! es_i (+ es_i 1))
    })

    ;; ③ 列增量：空洞数 + 列变换
    (set! es_holes sn_holes)
    (set! es_ct sn_ct)
    (set! es_c 0)
    (while (< es_c T.COLS) {
      (if (> (get_item tcnt es_c) 0) {
        ;; 新格都在该列表面之上（落地行按表面算），故新增高度 = 表面行 - 新格最上行；
        ;; 新增空洞 = 新增高度 - 新增格数（格子之间被架空的那些）
        (set! es_d (- (get_item surf es_c) (get_item tmin es_c)))
        (set! es_holes (+ es_holes (- es_d (get_item tcnt es_c))))
        (set! es_ct (+ es_ct (- (col_trans (Math.bit_or (get_item cmask es_c) (get_item tbits es_c)))
                                (col_trans (get_item cmask es_c)))))
      })
      (set! es_c (+ es_c 1))
    })

    ;; ④ 行增量：行变换 + 消行数 + 被消掉的方块格数
    (set! es_rt sn_rt)
    (set! es_wc sn_wc)
    (set! es_clr 0)
    (set! es_erode 0)
    (set! es_k 0)
    (while (< es_k es_nr) {
      (set! es_r (get_item rw_row es_k))
      (set! es_n (Math.bit_or (get_item rmask es_r) (get_item rw_bits es_k)))
      (set! es_rt (+ es_rt (- (row_trans es_n) (row_trans (get_item rmask es_r)))))
      (set! es_wc (+ es_wc (- (get_item pc (well_mask es_n))
                              (get_item pc (well_mask (get_item rmask es_r))))))
      (if (== es_n 1023) {                                  ; 10 格全满 → 这一行会被消掉
        (set! es_clr (+ es_clr 1))
        (set! es_erode (+ es_erode (get_item pc (get_item rw_bits es_k))))
      })
      (set! es_k (+ es_k 1))
    })

    ;; ⑤ 线性评估
    (set! es_land (- T.ROWS (+ py (get_item pmay rt))))      ; 方块最低格离底边的高度
    (set! es_s (* W_LAND es_land))
    (set! es_s (+ es_s (* W_ERODE es_erode)))
    (set! es_s (+ es_s (* W_ROW es_rt)))
    (set! es_s (+ es_s (* W_COL es_ct)))
    (set! es_s (+ es_s (* W_HOLES es_holes)))
    (set! es_s (+ es_s (* W_WELL es_wc)))

    (if (> es_s pl_best) {
      (set! pl_best es_s)
      (set! pl_brt rt)
      (set! pl_bx x0)
      (set! pl_bpy py)
    })
  }))

;; 某旋转态、某横位：先算落地行（与 T.collide 等价），合法才打分
(define eval_at
  (lambda (rt x0) {
    (set! ea_py 99)
    (set! ea_ok 1)
    (set! ea_i 0)
    (while (< ea_i 4) {
      (set! ea_c (+ x0 (get_item pax (+ (* rt 4) ea_i))))
      (if (or (< ea_c 0) (>= ea_c T.COLS))
          (set! ea_ok 0)
          {
            ;; 该格最多能落到 surf[c]-1-相对y 行（落地行 = 所有格子的最小上界）
            (set! ea_v (- (get_item surf ea_c) (+ (get_item pay (+ (* rt 4) ea_i)) 1)))
            (if (< ea_v ea_py) (set! ea_py ea_v))
          })
      (set! ea_i (+ ea_i 1))
    })
    ;; 连入场位置（py=-1）都放不下 → 该候选不可行
    (if (< ea_py -1) (set! ea_ok 0))
    (if (== ea_ok 1) (eval_score rt x0 ea_py))
  }))

;; 某旋转态：枚举全部合法横位
(define eval_rot
  (lambda (rt) {
    (set! er_minx 9)
    (set! er_maxx 0)
    (set! er_i 0)
    (while (< er_i 4) {
      (set! er_v (get_item pax (+ (* rt 4) er_i)))
      (if (< er_v er_minx) (set! er_minx er_v))
      (if (> er_v er_maxx) (set! er_maxx er_v))
      (set! er_i (+ er_i 1))
    })
    ;; 包围盒左上角 x 的取值范围：格子不越出 [0, COLS-1]
    (set! er_x (- 0 er_minx))
    (while (<= er_x (- T.COLS (+ er_maxx 1))) {
      (eval_at rt er_x)
      (set! er_x (+ er_x 1))
    })
  }))

;; ============================================================================
;; 对外接口
;; ============================================================================

;; 清掉当前计划（切换自动/手动、游戏结束时调）
(define reset
  (lambda () {
    (set! plan_ok 0)
    (set! plan_serial -1)
  }))

;; 为当前方块算一次最优落点（写 plan_rt / plan_px；返回是否有解）
(define think
  (lambda () {
    (set! plan_ok 0)
    (snapshot)
    (prep_rot T.piece)
    (set! pl_best -1e9)
    (set! pl_brt T.rot)
    (set! pl_bx T.px)
    (set! pl_rt 0)
    (while (< pl_rt 4) {
      (eval_rot pl_rt)
      (set! pl_rt (+ pl_rt 1))
    })
    (if (> pl_best -1e8) {
      (set! plan_rt pl_brt)
      (set! plan_px pl_bx)
      (set! plan_py pl_bpy)
      (set! plan_score pl_best)
      (set! plan_ok 1)
      (set! plan_serial T.serial)
    })
    plan_ok
  }))

;; 做一个动作：旋转 / 横移一格 / 硬降；返回 1 = 动了，0 = 没动（计划已失效）
;;   走到一半被挡（转不动、挪不动）→ 清计划，下一拍按新地形重算，绝不空转
(define act_once
  (lambda () {
    (set! tk_acted 0)
    (if (not (== T.rot plan_rt))
        { ; ① 先转到位（核心的旋转带踢墙，可能顺带改变 px）
          (T.rotate_piece)
          (set! tk_acted 1)
          (if (not (== T.rot plan_rt)) (set! plan_ok 0))
        }
        (if (not (== T.px plan_px))
            { ; ② 再逐格横移
              (set! tk_old T.px)
              (set! tk_d (if (< T.px plan_px) 1 -1))
              (T.move tk_d)
              (set! tk_acted 1)
              (if (== T.px tk_old) (set! plan_ok 0))
            }
            { ; ③ 到位 → 一次落底并锁定
              (T.hard_drop)
              (T.reset_fall)
              (set! plan_ok 0)
              (set! tk_acted 1)
            }))
    tk_acted
  }))

;; 每拍推进一次自动游玩；返回 1 = 本拍动了，0 = 没动
;;   离落点还远（>2 格）时一拍只做一个动作——用户看得见"方块在走"；
;;   快到落点了（或高关卡下降很快）就把剩下的动作在本拍全做完：
;;   重力每 tick 都在推进，动作慢于重力时方块会被提前锁定在错误的列上
;;   （实测高关卡正是这样死的），所以临近落地必须"果断"。
(define tick
  (lambda () {
    (set! tk_acted 0)
    (if (== T.game_over 0)
        {
          (if (or (== plan_ok 0) (not (== plan_serial T.serial)))
              (think)
              #f)
          (if (== plan_ok 1)
              (if (> (- plan_py T.py) 2)
                  (set! tk_acted (act_once))
                  {
                    (set! tk_n 0)
                    (while (and (== plan_ok 1) (< tk_n 16)) {
                      (act_once)
                      (set! tk_n (+ tk_n 1))
                    })
                    (set! tk_acted 1)
                  })
              #f)
        }
        (set! plan_ok 0))
    tk_acted
  }))
