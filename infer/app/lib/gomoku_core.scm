;; ============================================================================
;; 五子棋核心（Gomoku Core）—— 棋盘 / 棋形评分 / 增量局面评估 / α-β 搜索
;;
;; 与界面无关的纯逻辑：app/gomoku.scm 只负责布局、绘制与触屏交互（同
;; app/lib/tetris_core.scm + app/tetris.scm 的分工）。可脱离界面单独回归。
;;
;; 三层结构：
;;   ① 棋盘表示：15×15 的棋盘放在一张 23×23 的**带 4 圈边**的"填充盘"上
;;      （padded board，边界格恒为 2）。这样所有棋形读取（沿 4 个方向向两侧
;;      各看 4 格）都是纯下标运算、无需任何边界判断；"盘外"与"对方棋子"在
;;      棋形语义上完全等价（都挡住延伸），故可直接同码处理。
;;   ② 棋形评分：把"连续子串 + 两端开闭"与"五元滑窗"两种视角合起来，显式
;;      区分 五连 / 活四 / 冲四 / 活三 / 眠三 / 活二 / 眠二（见下方分值表）。
;;   ③ 局面评估：**增量式差分**——全盘 88 条线（横 15 + 纵 15 + 斜 29 + 反斜 29）
;;      各存一个带符号分值（黑正白负），落子只影响过该点的 4 条线，故只重算
;;      这 4 条线并做差分（gtotal += new - old）。叶子评估因此是 O(1)，不需要
;;      每步重扫全盘。回退更快：mk 把 4 条线的旧分值与总分压栈，unmk 直接弹栈
;;      恢复（完全不重算）。
;;
;; 搜索：α-β + 迭代加深（深度 2→maxd），候选点表在根节点一次生成
;; （"我方棋形分 + 0.8×对方棋形分"，攻守兼顾），各深度复用；形成五连在落子时
;; 由线分值直接判定（O(1)）并立刻取"胜"值，配合 α-β 剪枝。时间预算用
;; (Diag.ms)（单调实时毫秒，不受宿主虚拟时钟影响）——超预算立即回退到上一
;; 深度的结果，保证"1 秒内落子"。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数、无一元取负（写 (- 0 x)）、
;; 无 let/list/map、cond 每子句单个表达式、块内不 define（局部量统一用模块级
;; slot）、while 体不是独立词法区域、mod 与 / 恒浮点（本文件下标运算只用
;; 加法/乘法/减法，避免浮点下标）。数值范围：32 位宿主（wasm32）的浮点有效位
;; 只有约 19 位，故所有分值都压在 50 万以内，避免大整数精度损失。
;;
;; 对外接口（导入方以 G.名字 调用）：
;;   G.reset              清空棋盘
;;   G.cell x y           取 (x,y) 的棋子（0 空 / 1 黑 / 2 白）
;;   G.put x y pl         落子，返回 1 表示成五
;;   G.take x y           撤子（悔棋）
;;   G.makes5 x y pl      (x,y) 落 pl 是否成五
;;   G.total              当前局面分（黑正白负，增量维护）
;;   G.recompute          从零重算局面分（自检用）
;;   G.psc x y pl         单点棋形分（热图/排序用）
;;   G.heat pl            计算全部候选点热图（填 hmx/hmy/hms/hn/hmax）
;;   G.search me maxd ms  搜索并写 bx/by/bv/nodes/ms（用时）
;;   G.bx G.by G.bv       搜索结果（落点与分值）
;;   G.nodes G.ms         上一手的节点数与耗时（ms）
;; ============================================================================

(native Math)
(native Diag)

;; ---- 尺寸 ----------------------------------------------------------------
(define N 15)          ; 棋盘边长
(define PW 23)         ; 填充盘边长 = 15 + 4×2
(define NN 529)        ; PW×PW
(define EMPTY 0)
(define BLACK 1)
(define WHITE 2)

;; ---- 棋形分值（活四 > 冲四 > 活三 > 眠三 > 活二 > 眠二）------------------
(define S_FIVE 50000)     ; 五连
(define S_OPEN4 10000)    ; 活四（两头空，必胜形）
(define S_FOUR 2000)      ; 冲四（一头空，逼迫对方应）
(define S_OPEN3 1500)     ; 活三（可长成活四）
(define S_SLEEP3 150)     ; 眠三
(define S_OPEN2 120)      ; 活二
(define S_SLEEP2 12)      ; 眠二
(define S_ONE 1)          ; 单子（两头空）
;; 五元窗口的补充分（补"跳形"：X_XX / XX_X 这类连续子串看不到的棋形）
(define S_G4 500)
(define S_G3 50)
(define S_G2 5)

;; 搜索用的"胜负"分值：远大于任何真实局面的分值
(define SC_WIN 200000)

;; ---- 棋盘状态 ------------------------------------------------------------
(define pb '())        ; NN 个填充格：0 空 / 1 黑 / 2 白或边界
(define nr '())        ; NN 个"2 格内的棋子数"（候选点判定，增量维护）
(define cand '())      ; 候选点表（填充下标，原地表；可能含已落子/已失效的尾项）
(define candn 0)       ; 候选点表当前长度（自己维护，避免 O(n) 的 length）
(define bis '())       ; NN 个"是否边界格"（1/0；清盘时据此复原边界）
(define xl '())        ; 填充下标 -> 棋盘 x
(define yl '())        ; 填充下标 -> 棋盘 y
(define lsc '())       ; 88 条线的当前带符号分值
(define gtotal 0)      ; 全部线分值之和（黑正白负）—— 局面评估的唯一状态

;; mk/unmk 的现场栈（严格 LIFO，与搜索递归同构）
(define stk_g '())     ; gtotal
(define stk_a '())     ; 线 A 旧分值
(define stk_b '())
(define stk_c '())
(define stk_d '())
(define stk_cn '())    ; candn

;; 24 邻域偏移（±2 格方框去掉中心）
(define ndx '()) (define ndy '()) (define noff '())

;; ---- 模块级临时量（解释器块内不能 define）--------------------------------
(define i 0) (define k 0) (define m 0) (define q 0)
(define px 0) (define py 0) (define pi 0) (define pj 0) (define pxx 0)
(define x 0) (define y 0) (define xx 0) (define yy 0)
(define dd 0) (define ss 0) (define tcn 0)
(define id0 0) (define id1 0) (define id2 0) (define id3 0)
(define va 0) (define vb 0) (define vc 0) (define vd 0) (define vsum 0)
(define nv 0) (define nv2 0)
(define lastx -9) (define lasty -9)
(define dx0 0) (define dy0 0)

;; ============================================================================
;; 一条线的棋形评分（带符号：黑正白负）
;;   start : 线首的填充下标
;;   step  : 填充下标步长（横 1 / 纵 PW / 斜 PW+1 / 反斜 1-PW）
;;   len   : 线上格数（3..15）
;;
;; 两遍：
;;   A. 连续子串 + 两端开闭 —— 判别 五连/活四/冲四/活三/眠三/活二/眠二；
;;   B. 五元滑窗 —— 补"跳形"（X_XX、XX_X 之类），窗口内单色时按子数加分。
;; ============================================================================
(define lsb 0) (define lsw 0)
(define lk 0) (define lcur 0) (define lrun 0) (define lrs 0)
(define lv 0) (define lop 0) (define lval 0)
(define wb 0) (define ww 0)

;; 结算一段连续同色子：[rs, rend) 共 rlen 个，两端空位个数写入 lop
(define runval
  (lambda (c rlen rs rend start step len)
    (set! lop 0)
    (if (> rs 0) (if (== (get_item pb (+ start (* (- rs 1) step))) 0)
                     (set! lop (+ lop 1))))
    (if (< rend len) (if (== (get_item pb (+ start (* rend step))) 0)
                         (set! lop (+ lop 1))))
    (cond
      ((>= rlen 5) (set! lval S_FIVE))
      ((== rlen 4) (set! lval (if (== lop 2) S_OPEN4 (if (== lop 1) S_FOUR 0))))
      ((== rlen 3) (set! lval (if (== lop 2) S_OPEN3 (if (== lop 1) S_SLEEP3 0))))
      ((== rlen 2) (set! lval (if (== lop 2) S_OPEN2 (if (== lop 1) S_SLEEP2 0))))
      ((== rlen 1) (set! lval (if (== lop 2) S_ONE 0)))
      (else (set! lval 0)))
    lval))

(define lcalc
  (lambda (start step len)
    (set! lsb 0) (set! lsw 0)
    ;; ---- A. 连续子串 ----
    (set! lcur 0) (set! lrun 0) (set! lrs 0)
    (set! lk 0)
    (while (< lk len) {
      (set! lv (get_item pb (+ start (* lk step))))
      (if (== lv lcur)
          (set! lrun (+ lrun 1))
          {
            (if (> lcur 0)
                {
                  (set! lop (runval lcur lrun lrs lk start step len))
                  (if (== lcur BLACK)
                      (set! lsb (+ lsb lop))
                      (set! lsw (+ lsw lop)))
                })
            (set! lcur lv) (set! lrun 1) (set! lrs lk)
          })
      (set! lk (+ lk 1))
    })
    (if (> lcur 0)
        {
          (set! lop (runval lcur lrun lrs len start step len))
          (if (== lcur BLACK)
              (set! lsb (+ lsb lop))
              (set! lsw (+ lsw lop)))
        })
    ;; ---- B. 五元滑窗（跳形）----
    (if (>= len 5)
        {
          (set! wb 0) (set! ww 0)
          (set! lk 0)
          (while (< lk 5) {
            (set! lv (get_item pb (+ start (* lk step))))
            (if (== lv BLACK) (set! wb (+ wb 1)) (if (== lv WHITE) (set! ww (+ ww 1))))
            (set! lk (+ lk 1))
          })
          (set! lk 0)
          (while (< lk (- len 4)) {
            (if (== ww 0)
                (if (== wb 4) (set! lsb (+ lsb S_G4))
                    (if (== wb 3) (set! lsb (+ lsb S_G3))
                        (if (== wb 2) (set! lsb (+ lsb S_G2))))))
            (if (== wb 0)
                (if (== ww 4) (set! lsw (+ lsw S_G4))
                    (if (== ww 3) (set! lsw (+ lsw S_G3))
                        (if (== ww 2) (set! lsw (+ lsw S_G2))))))
            ;; 窗口右移一格：移出 lk，移入 lk+5
            (set! lv (get_item pb (+ start (* lk step))))
            (if (== lv BLACK) (set! wb (- wb 1)) (if (== lv WHITE) (set! ww (- ww 1))))
            (if (< (+ lk 5) len) {
              (set! lv (get_item pb (+ start (* (+ lk 5) step))))
              (if (== lv BLACK) (set! wb (+ wb 1)) (if (== lv WHITE) (set! ww (+ ww 1))))
            })
            (set! lk (+ lk 1))
          })
        })
    (- lsb lsw)))
;; ============================================================================
;; 初始化 / 棋盘访问
;; ============================================================================

;; 24 邻域偏移表（只建一次）
(define build_offsets
  (lambda ()
    (set! ndx '()) (set! ndy '()) (set! noff '())
    (set! dy0 0)
    (while (< dy0 5) {
      (set! dx0 0)
      (while (< dx0 5) {
        (if (not (and (== dx0 2) (== dy0 2))) {
          (push ndx (- dx0 2))
          (push ndy (- dy0 2))
          (push noff (+ (* (- dy0 2) PW) (- dx0 2)))
        })
        (set! dx0 (+ dx0 1))
      })
      (set! dy0 (+ dy0 1))
    })))

;; 建表（模块装载时执行一次）：填充盘 / 邻域计数 / 坐标表 / 88 条线分值
;; 注：'() 字面量是**静态槽**，对它 push 会原地改动——因此所有"可反复清空"的表
;; 只在装载时 push 建好一次，之后一律用 set_item! 归零（见 reset）。
(define build_tables
  (lambda ()
    (set! py 0)
    (while (< py PW) {
      (set! pxx 0)
      (while (< pxx PW) {
        (push xl (- pxx 4))
        (push yl (- py 4))
        (push nr 0)
        (if (or (< pxx 4) (or (< py 4) (or (>= pxx 19) (>= py 19))))
            { (push pb 2) (push bis 1) }
            { (push pb 0) (push bis 0) })
        (set! pxx (+ pxx 1))
      })
      (set! py (+ py 1))
    })
    (set! q 0)
    (while (< q 88) { (push lsc 0) (set! q (+ q 1)) })))

;; 清空棋盘（重新开局）：全部表原地归零
(define reset
  (lambda ()
    (define cc 0)
    ;; 填充盘 + 邻域计数
    (set! cc 0)
    (while (< cc NN) {
      ;; 注意：白子也是 2，不能按"格值是否为 2"判断边界，必须查边界表
      (if (== (get_item bis cc) 1)
          (set_item! pb cc 2)
          (set_item! pb cc 0))
      (set_item! nr cc 0)
      (set! cc (+ cc 1))
    })
    ;; 候选点表（pop 到空：pop 是原地操作，列表对象保持同一个）
    (while (> candn 0) { (pop cand) (set! candn (- candn 1)) })
    ;; 88 条线
    (set! cc 0)
    (while (< cc 88) { (set_item! lsc cc 0) (set! cc (+ cc 1)) })
    (set! gtotal 0)
    ;; 现场栈（正常应为空；防御性清空）
    (while (> (length stk_g) 0) {
      (pop stk_g) (pop stk_cn)
      (pop stk_a) (pop stk_b) (pop stk_c) (pop stk_d)
    })))

;; (x,y) -> 填充下标
(define pidx (lambda (ax ay) (+ (* (+ ay 4) PW) (+ ax 4))))

;; (x,y) 处的格子（盘外返回 2）
(define cell (lambda (ax ay) (get_item pb (+ (* (+ ay 4) PW) (+ ax 4)))))

;; ============================================================================
;; 落子 / 撤子（增量差分 + 现场压栈）
;; ============================================================================

;; 过 (x,y) 的四条线的线号，写入 id0..id3
;;   横：y                （0..14）
;;   纵：15+x             （15..29）
;;   斜 ↘：30+(x-y+14)    （30..58）
;;   反斜 ↙：59+(x+y)     （59..87）
(define lineids
  (lambda (ax ay)
    (set! id0 ay)
    (set! id1 (+ 15 ax))
    (set! id2 (+ 30 (+ (- ax ay) 14)))
    (set! id3 (+ 59 (+ ax ay)))))

;; 重算过 (x,y) 的四条线并做增量差分（gtotal += new - old）
;; 结果写入 va..vd（四条线的新分值）；mk 与 take 共用
(define relines
  (lambda (ax ay)
    (lineids ax ay)
    ;; ---- 横 ----
    (set! gtotal (- gtotal (get_item lsc id0)))
    (set! nv (lcalc (+ (* (+ ay 4) PW) 4) 1 15))
    (set_item! lsc id0 nv) (set! gtotal (+ gtotal nv)) (set! va nv)
    ;; ---- 纵 ----
    (set! gtotal (- gtotal (get_item lsc id1)))
    (set! nv (lcalc (+ (* 4 PW) (+ ax 4)) PW 15))
    (set_item! lsc id1 nv) (set! gtotal (+ gtotal nv)) (set! vb nv)
    ;; ---- 斜 ↘（填充步长 PW+1）----
    (set! m (if (< ax ay) ax ay))
    (set! dd (- ax ay)) (if (< dd 0) (set! dd (- 0 dd)))
    (set! gtotal (- gtotal (get_item lsc id2)))
    (set! nv (lcalc (+ (* (+ (- ay m) 4) PW) (+ (- ax m) 4)) (+ PW 1) (- 15 dd)))
    (set_item! lsc id2 nv) (set! gtotal (+ gtotal nv)) (set! vc nv)
    ;; ---- 反斜 ↙（填充步长 1-PW）----
    (set! ss (+ ax ay))
    (set! gtotal (- gtotal (get_item lsc id3)))
    (set! nv (if (< ss N)
                 (lcalc (+ (* (+ ss 4) PW) 4) (- 1 PW) (+ ss 1))
                 (lcalc (+ (* 18 PW) (+ (- ss 14) 4)) (- 1 PW) (- 29 ss))))
    (set_item! lsc id3 nv) (set! gtotal (+ gtotal nv)) (set! vd nv)))

;; 落子：返回 1 表示这一手形成五连
;; （胜负判定 O(1)：只有刚落子的一方可能含五连，看 4 条更新后的线分值即可）
(define mk
  (lambda (ax ay apl)
    (define f 0)
    (set! pi (+ (* (+ ay 4) PW) (+ ax 4)))
    (set_item! pb pi apl)
    (push stk_g gtotal)
    (push stk_cn candn)
    (lineids ax ay)
    (push stk_a (get_item lsc id0))
    (push stk_b (get_item lsc id1))
    (push stk_c (get_item lsc id2))
    (push stk_d (get_item lsc id3))
    (relines ax ay)
    (set! vsum (+ (+ va vb) (+ vc vd)))
    (if (== apl BLACK)
        (if (>= vsum S_FIVE) (set! f 1))
        (if (<= vsum (- 0 S_FIVE)) (set! f 1)))
    ;; ---- 邻域计数 + 候选点表 ----
    (set! k 0)
    (while (< k 24) {
      (set! xx (+ ax (get_item ndx k)))
      (set! yy (+ ay (get_item ndy k)))
      (if (and (>= xx 0) (and (>= yy 0) (and (< xx N) (< yy N)))) {
        (set! pj (+ pi (get_item noff k)))
        (set! nv2 (+ (get_item nr pj) 1))
        (if (== nv2 1) { (push cand pj) (set! candn (+ candn 1)) })
        (set_item! nr pj nv2)
      })
      (set! k (+ k 1))
    })
    f))

;; 撤子：线分值/总分弹栈恢复（不重算）；邻域计数回退；候选点表截断
(define unmk
  (lambda (ax ay)
    (set! pi (+ (* (+ ay 4) PW) (+ ax 4)))
    (set_item! pb pi 0)
    (lineids ax ay)
    (set_item! lsc id0 (pop stk_a))
    (set_item! lsc id1 (pop stk_b))
    (set_item! lsc id2 (pop stk_c))
    (set_item! lsc id3 (pop stk_d))
    (set! gtotal (pop stk_g))
    (set! k 0)
    (while (< k 24) {
      (set! xx (+ ax (get_item ndx k)))
      (set! yy (+ ay (get_item ndy k)))
      (if (and (>= xx 0) (and (>= yy 0) (and (< xx N) (< yy N)))) {
        (set! pj (+ pi (get_item noff k)))
        (set_item! nr pj (- (get_item nr pj) 1))
      })
      (set! k (+ k 1))
    })
    (set! tcn (pop stk_cn))
    (while (> candn tcn) { (pop cand) (set! candn (- candn 1)) })))

;; 丢弃 mk 压入的现场（落子被提交，不回退）
(define trim
  (lambda ()
    (pop stk_g) (pop stk_cn)
    (pop stk_a) (pop stk_b) (pop stk_c) (pop stk_d)))

;; 界面对外：落子（返回 1 = 形成五连）
;; 注：lambda 体内的 define 是本层调用的**局部量**（manual §5.2 内部定义），
;; 搜索里 mk/unmk 会反复改写模块级临时量，故凡需跨调用保存的值一律用局部量。
(define put
  (lambda (ax ay apl)
    (define f 0)
    (set! f (mk ax ay apl))
    (trim)
    (set! lastx ax) (set! lasty ay)
    f))

;; 界面对外：撤子（悔棋）——把棋子拿走并重算过该点的四条线（增量差分）
;; 注意不能用 mk+unmk：mk 存的"旧线分值"是**有这颗子**时的值，unmk 会把它还原回来。
(define take
  (lambda (ax ay)
    (define p 0)
    (set! p (cell ax ay))
    (if (> p 0) {
      (set! lastx -9) (set! lasty -9)
      (set! pi (+ (* (+ ay 4) PW) (+ ax 4)))
      (set_item! pb pi 0)
      (relines ax ay)
      ;; 邻域计数回退（候选点表保留失效项，由扫描时按"空点"过滤）
      (set! k 0)
      (while (< k 24) {
        (set! xx (+ ax (get_item ndx k)))
        (set! yy (+ ay (get_item ndy k)))
        (if (and (>= xx 0) (and (>= yy 0) (and (< xx N) (< yy N)))) {
          (set! pj (+ pi (get_item noff k)))
          (set_item! nr pj (- (get_item nr pj) 1))
        })
        (set! k (+ k 1))
      })
    } #f)))

;; ============================================================================
;; 单点棋形评分（"如果我在 (x,y) 落子"）—— 候选排序与热图
;;   只看过该点的 4 条线、每条线取以该点为中心的 9 格窗口（±4）。
;;   这是**局部**评分：只依赖该点周围 9 格。其值即"这一手做出什么棋形"：
;;   活四/冲四/活三/眠三/活二（含跳形）。
;; ============================================================================
(define pv1 0) (define pv2 0) (define pv3 0) (define pv4 0)
(define pw1 0) (define pw2 0) (define pw3 0) (define pw4 0)
(define lm1 0) (define lm2 0) (define lm3 0) (define lm4 0)
(define lb1 0) (define lb2 0) (define lb3 0) (define lb4 0)
(define rm1 0) (define rm2 0) (define rm3 0) (define rm4 0)
(define rb1 0) (define rb2 0) (define rb3 0) (define rb4 0)
(define prl 0) (define prr 0) (define pol 0) (define por 0)
(define ptot 0) (define pm 0) (define po 0) (define pv0 0)
(define pc5 0) (define pc4 0) (define pc3 0)
(define psc_sum 0)

;; 单方向棋形分：pi0 为该点的填充下标，step 为该方向的填充步长
(define pdir
  (lambda (pi0 step apl)
    ;; ---- 读左右各 4 格，归一化：1=我 / 0=空 / 2=对手或边界 ----
    (set! pv1 (get_item pb (- pi0 step)))
    (set! pv2 (get_item pb (- pi0 (* 2 step))))
    (set! pv3 (get_item pb (- pi0 (* 3 step))))
    (set! pv4 (get_item pb (- pi0 (* 4 step))))
    (set! pw1 (get_item pb (+ pi0 step)))
    (set! pw2 (get_item pb (+ pi0 (* 2 step))))
    (set! pw3 (get_item pb (+ pi0 (* 3 step))))
    (set! pw4 (get_item pb (+ pi0 (* 4 step))))
    (if (== pv1 apl) (set! pv1 1) (if (not (== pv1 0)) (set! pv1 2)))
    (if (== pv2 apl) (set! pv2 1) (if (not (== pv2 0)) (set! pv2 2)))
    (if (== pv3 apl) (set! pv3 1) (if (not (== pv3 0)) (set! pv3 2)))
    (if (== pv4 apl) (set! pv4 1) (if (not (== pv4 0)) (set! pv4 2)))
    (if (== pw1 apl) (set! pw1 1) (if (not (== pw1 0)) (set! pw1 2)))
    (if (== pw2 apl) (set! pw2 1) (if (not (== pw2 0)) (set! pw2 2)))
    (if (== pw3 apl) (set! pw3 1) (if (not (== pw3 0)) (set! pw3 2)))
    (if (== pw4 apl) (set! pw4 1) (if (not (== pw4 0)) (set! pw4 2)))
    ;; ---- 连续子串长度（中心算 1 个）与两端开闭 ----
    (set! prl 0)
    (if (== pv1 1) { (set! prl 1)
      (if (== pv2 1) { (set! prl 2)
        (if (== pv3 1) { (set! prl 3)
          (if (== pv4 1) (set! prl 4)) }) }) })
    (set! prr 0)
    (if (== pw1 1) { (set! prr 1)
      (if (== pw2 1) { (set! prr 2)
        (if (== pw3 1) { (set! prr 3)
          (if (== pw4 1) (set! prr 4)) }) }) })
    (set! pol 0)
    (if (== prl 0) (if (== pv1 0) (set! pol 1)))
    (if (== prl 1) (if (== pv2 0) (set! pol 1)))
    (if (== prl 2) (if (== pv3 0) (set! pol 1)))
    (if (== prl 3) (if (== pv4 0) (set! pol 1)))
    (set! por 0)
    (if (== prr 0) (if (== pw1 0) (set! por 1)))
    (if (== prr 1) (if (== pw2 0) (set! por 1)))
    (if (== prr 2) (if (== pw3 0) (set! por 1)))
    (if (== prr 3) (if (== pw4 0) (set! por 1)))
    (set! ptot (+ (+ prl prr) 1))
    ;; ---- 含中心的 5 个五元窗口：前缀累加"我方子数 / 被挡格数" ----
    (set! lm1 (if (== pv1 1) 1 0)) (set! lb1 (if (== pv1 2) 1 0))
    (set! lm2 (+ lm1 (if (== pv2 1) 1 0))) (set! lb2 (+ lb1 (if (== pv2 2) 1 0)))
    (set! lm3 (+ lm2 (if (== pv3 1) 1 0))) (set! lb3 (+ lb2 (if (== pv3 2) 1 0)))
    (set! lm4 (+ lm3 (if (== pv4 1) 1 0))) (set! lb4 (+ lb3 (if (== pv4 2) 1 0)))
    (set! rm1 (if (== pw1 1) 1 0)) (set! rb1 (if (== pw1 2) 1 0))
    (set! rm2 (+ rm1 (if (== pw2 1) 1 0))) (set! rb2 (+ rb1 (if (== pw2 2) 1 0)))
    (set! rm3 (+ rm2 (if (== pw3 1) 1 0))) (set! rb3 (+ rb2 (if (== pw3 2) 1 0)))
    (set! rm4 (+ rm3 (if (== pw4 1) 1 0))) (set! rb4 (+ rb3 (if (== pw4 2) 1 0)))
    (set! pc5 0) (set! pc4 0) (set! pc3 0)
    ;; 窗口 [-4..0] / [-3..+1] / [-2..+2] / [-1..+3] / [0..+4]
    (set! pm lm4) (set! po lb4)
    (if (== po 0) (if (== pm 4) (set! pc5 (+ pc5 1))
                      (if (== pm 3) (set! pc4 (+ pc4 1))
                          (if (== pm 2) (set! pc3 (+ pc3 1))))))
    (set! pm (+ lm3 rm1)) (set! po (+ lb3 rb1))
    (if (== po 0) (if (== pm 4) (set! pc5 (+ pc5 1))
                      (if (== pm 3) (set! pc4 (+ pc4 1))
                          (if (== pm 2) (set! pc3 (+ pc3 1))))))
    (set! pm (+ lm2 rm2)) (set! po (+ lb2 rb2))
    (if (== po 0) (if (== pm 4) (set! pc5 (+ pc5 1))
                      (if (== pm 3) (set! pc4 (+ pc4 1))
                          (if (== pm 2) (set! pc3 (+ pc3 1))))))
    (set! pm (+ lm1 rm3)) (set! po (+ lb1 rb3))
    (if (== po 0) (if (== pm 4) (set! pc5 (+ pc5 1))
                      (if (== pm 3) (set! pc4 (+ pc4 1))
                          (if (== pm 2) (set! pc3 (+ pc3 1))))))
    (set! pm rm4) (set! po rb4)
    (if (== po 0) (if (== pm 4) (set! pc5 (+ pc5 1))
                      (if (== pm 3) (set! pc4 (+ pc4 1))
                          (if (== pm 2) (set! pc3 (+ pc3 1))))))
    ;; ---- 归类 ----
    (cond
      ((or (>= ptot 5) (> pc5 0)) (set! pv0 S_FIVE))
      ((or (and (== ptot 4) (and (== pol 1) (== por 1))) (>= pc4 2))
       (set! pv0 S_OPEN4))
      ((or (== ptot 4) (== pc4 1)) (set! pv0 S_FOUR))
      ((or (and (== ptot 3) (and (== pol 1) (== por 1))) (>= pc3 2))
       (set! pv0 S_OPEN3))
      ((or (== ptot 3) (== pc3 1)) (set! pv0 S_SLEEP3))
      ((and (== ptot 2) (and (== pol 1) (== por 1))) (set! pv0 S_OPEN2))
      ((== ptot 2) (set! pv0 S_SLEEP2))
      ((and (== ptot 1) (and (== pol 1) (== por 1))) (set! pv0 S_ONE))
      (else (set! pv0 0)))
    pv0))

;; 某点在指定一方的棋形分（4 条线求和；多方向威胁自然叠加，
;; 双活三 ≈ 2×S_OPEN3 已大于一个冲四，符合"双三极强"的棋理）
(define psc
  (lambda (ax ay apl)
    (set! pi (+ (* (+ ay 4) PW) (+ ax 4)))
    (set! psc_sum 0)
    (set! psc_sum (+ psc_sum (pdir pi 1 apl)))
    (set! psc_sum (+ psc_sum (pdir pi PW apl)))
    (set! psc_sum (+ psc_sum (pdir pi (+ PW 1) apl)))
    (set! psc_sum (+ psc_sum (pdir pi (- 1 PW) apl)))
    psc_sum))

;; (x,y) 落 pl 是否直接成五
(define makes5
  (lambda (ax ay apl)
    (define f 0)
    (set! f (mk ax ay apl))
    (unmk ax ay)
    f))

;; 从零重算局面分（自检：应与增量维护的 gtotal 相等）
(define recompute
  (lambda ()
    (set! vsum 0)
    (set! py 0)
    (while (< py 15) {
      (set! vsum (+ vsum (lcalc (+ (* (+ py 4) PW) 4) 1 15)))
      (set! py (+ py 1))
    })
    (set! px 0)
    (while (< px 15) {
      (set! vsum (+ vsum (lcalc (+ (* 4 PW) (+ px 4)) PW 15)))
      (set! px (+ px 1))
    })
    (set! q (- 0 14))
    (while (<= q 14) {
      (set! m (if (< q 0) (- 0 q) q))
      (set! xx (if (< q 0) 0 q))
      (set! yy (if (< q 0) (- 0 q) 0))
      (set! vsum (+ vsum (lcalc (+ (* (+ yy 4) PW) (+ xx 4)) (+ PW 1) (- 15 m))))
      (set! q (+ q 1))
    })
    (set! ss 0)
    (while (< ss 29) {
      (set! vsum (+ vsum (if (< ss N)
                             (lcalc (+ (* (+ ss 4) PW) 4) (- 1 PW) (+ ss 1))
                             (lcalc (+ (* 18 PW) (+ (- ss 14) 4)) (- 1 PW) (- 29 ss)))))
      (set! ss (+ ss 1))
    })
    vsum))

;; ============================================================================
;; 搜索：α-β + 迭代加深
;; ============================================================================

;; 根候选点（三个平行原地表；下标 0..ncand-1，按分值降序）
(define cx '()) (define cy '()) (define cs '())
(define ncand 0)
(define NROOT 20)        ; 根候选上限（同时限定各深度的分支因子）
;; 深层（ply ≥ 2）只展开根候选表的前 NARROW 个：根表已按"我方分 + 0.8×对方分"
;; 降序，前若干个就是主要攻防点；深层再全量展开只会放大节点数而不改变选点。
(define NARROW 10)

(define nodes 0)         ; 本手访问的节点数（诊断/显示）
(define s_abort 0)       ; 超预算标志
(define tstart 0)        ; 搜索起点（单调实时毫秒）
(define tbudget 700)     ; 时间预算（ms）
(define sm 0) (define so 0) (define sv2 0)
(define si 0) (define sj 0) (define scnt 0) (define dd2 0)
(define bv 0) (define bx 0) (define by 0)   ; 最终结果（bx/by 落点，bv 评分）
(define ms 0)                               ; 本手耗时（ms）

;; 生成根候选：对每个"2 格内有子"的空点，
;;   分值 = 我方棋形分 + 0.8×对方棋形分（攻守兼顾），选择排序取前 NROOT
(define build_root
  (lambda (me)
    (while (> ncand 0) { (pop cx) (pop cy) (pop cs) (set! ncand (- ncand 1)) })
    (set! si 0)
    (set! scnt (length cand))
    (while (< si scnt) {
      (set! sj (get_item cand si))
      (if (== (get_item pb sj) 0) {
        (set! x (get_item xl sj))
        (set! y (get_item yl sj))
        (set! sm (psc x y me))
        (set! so (psc x y (- 3 me)))
        (set! sv2 (+ sm (Math.floor (* so 0.8))))
        ;; 位置微调：贴近上一手（局部应对）优先；其次靠天元。
        ;; 棋形分相同时（例如开局第一应手），这条能把"直线邻点"排在"斜邻点"前面。
        (set! so (- x lastx)) (if (< so 0) (set! so (- 0 so)))
        (set! sm (- y lasty)) (if (< sm 0) (set! sm (- 0 sm)))
        (set! dd2 (+ so sm))
        (if (== dd2 1) (set! sv2 (+ sv2 12))
            (if (== dd2 2) (set! sv2 (+ sv2 6))
                (if (== dd2 3) (set! sv2 (+ sv2 2)))))
        (set! so (- x 7)) (if (< so 0) (set! so (- 0 so)))
        (set! sm (- y 7)) (if (< sm 0) (set! sm (- 0 sm)))
        (set! dd2 (+ so sm))
        (if (< dd2 5) (set! sv2 (+ sv2 (- 5 dd2))))
        (push cx x) (push cy y)
        (push cs sv2)
        (set! ncand (+ ncand 1))
      })
      (set! si (+ si 1))
    })
    (set! si 0)
    (while (and (< si ncand) (< si NROOT)) {
      (set! sj (+ si 1))
      (while (< sj ncand) {
        (if (> (get_item cs sj) (get_item cs si)) {
          (set! sv2 (get_item cs si)) (set_item! cs si (get_item cs sj)) (set_item! cs sj sv2)
          (set! sv2 (get_item cx si)) (set_item! cx si (get_item cx sj)) (set_item! cx sj sv2)
          (set! sv2 (get_item cy si)) (set_item! cy si (get_item cy sj)) (set_item! cy sj sv2)
        })
        (set! sj (+ sj 1))
      })
      (set! si (+ si 1))
    })
    (while (> ncand NROOT) {
      (pop cx) (pop cy) (pop cs)
      (set! ncand (- ncand 1))
    })))

;; 时间检查（每 64 个节点一次）：单节点成本约 0.2~0.5ms（wasm），
;; 64 个节点一批可把"超预算"的过冲压在 ~30ms 内。
(define chk_time
  (lambda ()
    (if (> (- (Diag.ms) tstart) tbudget) (set! s_abort 1))))

;; α-β（negamax）：返回"轮到 apl 走"这一局面对 apl 的分值
;;   ply：距根的手数（用于"越快取胜越好"的胜负值）
;; 注意：本函数递归，跨递归需保留的状态（循环下标、本层最优值、落点）一律用
;; lambda 体内的 define（每次调用独立的局部量）——模块级变量会被下一层改写。
(define negamax
  (lambda (d alpha beta ply apl)
    (define nb 0)
    (define kk 0)
    (define mx 0)
    (define my 0)
    (define fv 0)
    (define sv 0)
    (define kend 0)
    (set! nodes (+ nodes 1))
    (if (== (mod nodes 64) 0) (chk_time))
    (if (== s_abort 1)
        alpha
        (if (== d 0)
            (if (== apl BLACK) gtotal (- 0 gtotal))
            {
              (set! nb (- 0 SC_WIN))
              (set! kk 0)
              (set! kend (if (>= ply 2) (if (> ncand NARROW) NARROW ncand) ncand))
              (while (< kk kend) {
                (if (== s_abort 1) break)
                (set! mx (get_item cx kk))
                (set! my (get_item cy kk))
                (if (== (cell mx my) 0)
                    {
                      (set! fv (mk mx my apl))
                      (if (== fv 1)
                          { (unmk mx my) (set! nb (- SC_WIN ply)) break }
                          {
                            (set! sv (- 0 (negamax (- d 1) (- 0 beta) (- 0 alpha)
                                                   (+ ply 1) (- 3 apl))))
                            (unmk mx my)
                            (if (> sv nb) (set! nb sv))
                            (if (> nb alpha) (set! alpha nb))
                            (if (>= alpha beta) break)
                          })
                    })
                (set! kk (+ kk 1))
              })
              nb
            }))))

;; 搜索入口：me 为我方，maxd 为最大深度（2~4），bud 为时间预算（ms）
;; 结果写入 bx / by / bv / nodes / ms。必胜/必败时提前停止加深。
(define search
  (lambda (me maxd bud)
    (define kk 0)
    (define mx 0)
    (define my 0)
    (define fv 0)
    (define sv 0)
    (define al 0)
    (define dvv 0)
    (define dxv 0)
    (define dyv 0)
    (define dd 2)
    (define el 0)
    (set! tbudget bud)
    (set! tstart (Diag.ms))
    (set! nodes 0)
    (set! s_abort 0)
    (set! bx 7) (set! by 7) (set! bv 0)
    (if (> (length cand) 0)
        {
          (build_root me)
          (if (> ncand 0) {
            ;; 兜底：先取根候选第一名（万一超预算也有个合理落点）
            (set! bx (get_item cx 0))
            (set! by (get_item cy 0))
            (set! bv 0)
            (while (<= dd maxd) {
              (set! al (- 0 SC_WIN))
              (set! dvv (- 0 SC_WIN))
              (set! dxv bx) (set! dyv by)
              (set! kk 0)
              (while (< kk ncand) {
                (if (== s_abort 1) break)
                (set! mx (get_item cx kk))
                (set! my (get_item cy kk))
                (if (== (cell mx my) 0)
                    {
                      (set! fv (mk mx my me))
                      (if (== fv 1)
                          { (unmk mx my) (set! sv (- SC_WIN 1)) }
                          {
                            (set! sv (- 0 (negamax (- dd 1) (- 0 SC_WIN) (- 0 al) 1 (- 3 me))))
                            (unmk mx my)
                          })
                      (if (> sv dvv) { (set! dvv sv) (set! dxv mx) (set! dyv my) })
                      (if (> dvv al) (set! al dvv))
                    })
                (set! kk (+ kk 1))
              })
              ;; 只有完整走完的深度才采纳
              (if (== s_abort 1)
                  break
                  {
                    (set! bv dvv) (set! bx dxv) (set! by dyv)
                    ;; 已确定必胜/必败：不必再加深
                    (if (>= bv (- SC_WIN 100)) break)
                    (if (<= bv (- 0 (- SC_WIN 100))) break)
                    ;; 预算估计：下一深度的耗时约为本深度的 5 倍。
                    ;; 剩下的时间不够就先交本深度的结果（避免每次都白跑一次
                    ;; 跑到一半被中断的深搜，用户等待时间显著缩短）。
                    (set! el (- (Diag.ms) tstart))
                    (if (> (+ el (* 5 el)) tbudget) break)
                  })
              (set! dd (+ dd 1))
            })
          })
        })
    (set! ms (- (Diag.ms) tstart))
    #f))

;; ============================================================================
;; 候选点热图（界面用）：对全部"2 格内有子"的空点求单点棋形分
;; ============================================================================
(define hmx '()) (define hmy '()) (define hms '())
(define hn 0) (define hmax 1)
(define hx 0) (define hy 0) (define hv 0)
(define hi 0) (define hj 0) (define hcnt 0)

(define heat
  (lambda (pl)
    (while (> hn 0) { (pop hmx) (pop hmy) (pop hms) (set! hn (- hn 1)) })
    (set! hmax 1)
    (set! hcnt (length cand))
    (set! hi 0)
    (while (< hi hcnt) {
      (set! hj (get_item cand hi))
      (if (== (get_item pb hj) 0) {
        (set! hx (get_item xl hj))
        (set! hy (get_item yl hj))
        (set! hv (psc hx hy pl))
        (push hmx hx) (push hmy hy) (push hms hv)
        (if (> hv hmax) (set! hmax hv))
        (set! hn (+ hn 1))
      })
      (set! hi (+ hi 1))
    })))

;; ============================================================================
;; 模块装载时执行：建偏移表 + 清盘
;; ============================================================================
(build_offsets)
(build_tables)
(reset)
