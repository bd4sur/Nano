;; ============================================================================
;; 俄罗斯方块纯逻辑库（Tetris Core）—— 供 app/tetris.scm 与回归测试共用
;;
;; 逐条对照 infer/ui_tetris.c 的逻辑部分实现：
;;   · 10x20 场地，7 种方块（4x4 包围盒，顺时针旋转 (x,y) -> (3-y,x)）
;;   · 碰撞：越界或压住已锁定格子即冲突；包围盒 y<0 的格子视为合法（方块从上方入场）
;;   · 生成：px=3、py=-1、rot=0；无法入场即游戏结束
;;   · 消行：自下而上扫描，满行整体下移后原地重查同一行（等价 C 版 for 循环里的 y++）
;;   · 计分：100/300/500/800 × 当前关卡；行数每满 10 升一关
;;   · 落地间隔：max(800-70*(level-1), 100) ms
;;
;; 本模块不依赖 Gfx/Touch（只用到 Math 的随机数与取整），因此可以脱离界面单独测试
;; （见 infer/tools/e2e/native/tetris_logic.scm）。
;;
;; 解释器约束（manual.md）：算术固定二参数；无 list/let；cond 子句单个表达式；
;; / 与 mod 恒返回浮点（取整用 Math.floor）；'() 字面量是模块级静态槽（字段表在
;; 模块顶层一次性 push 构建，此后只用 set_item! 原地改写）。
;; ============================================================================

(native Math)

;; ---- 场地与规则常量（与 C 版一致）----------------------------------------

(define COLS 10)
(define ROWS 20)

;; 7 种方块基础形态：扁平表，索引 = piece*8 + i*2 + {0=x, 1=y}（4x4 网格坐标）
(define SHAPES '(
   0 1  1 1  2 1  3 1     ; 0 I
   1 1  2 1  1 2  2 2     ; 1 O
   1 1  0 2  1 2  2 2     ; 2 T
   1 1  2 1  0 2  1 2     ; 3 S
   0 1  1 1  1 2  2 2     ; 4 Z
   0 1  0 2  1 2  2 2     ; 5 J
   2 1  0 2  1 2  2 2))   ; 6 L

;; 7 种方块颜色（索引 = piece*3 + {0=r,1=g,2=b}，扁平 21 项；同 C 版 S_TRS_COLORS）
(define COLORS '(
   0 220 220    ; I 青
   240 220 40   ; O 黄
   170 80 220   ; T 紫
   60 200 80    ; S 绿
   230 60 60    ; Z 红
   70 110 230   ; J 蓝
   240 150 40)) ; L 橙

(define KICKS '(0 -1 1 -2 2))           ; 旋转踢墙偏移（依次尝试）
(define SCORE_TABLE '(100 300 500 800)) ; 消 1/2/3/4 行的基础分

;; ---- 游戏状态（导入方以 T.名字 直接读取）---------------------------------

(define field '())      ; 场地扁平表（COLS*ROWS）：0 空，1..7 = piece+1
(define cells_x '())    ; fill_cells 写出的 4 个格子 x
(define cells_y '())    ; 同上 y

(define piece 0)        ; 当前方块 0..6
(define rot 0)          ; 当前旋转态 0..3
(define px 3)           ; 包围盒左上角场地坐标
(define py -1)
(define next_piece 0)
(define score 0)
(define lines 0)
(define level 1)
(define fall_acc 0)     ; 重力累计（ms）
(define game_over 0)

;; ---- 模块级临时量（热路径局部量一律提到模块级：解释器对块内 define/嵌套区域
;;      存在布局敏感的槽位问题，见 doc/AGENTS.md）----------------------------

(define t_i 0) (define t_x 0) (define t_y 0) (define t_r 0) (define t_t 0)   ; fill_cells 专用
(define t_kick 0)                                                          ; rotate_piece 的踢墙偏移
(define t_fm 0)                                                            ; fall_ms 的中间值
(define t_k 0) (define t_ok 0) (define t_full 0) (define t_cleared 0) (define t_yy 0)
;; collide 的命中标志单独一份：rotate_piece 在循环里用 t_ok 作"已成功"标志，
;; 若与 collide 共用会被 collide 的返回值覆盖，导致踢墙只试第一档。
(define t_hit 0)

;; ---- 场地一次性构建（'() 静态槽只 push 一次；reset 用 set_item! 清零）-----

(define build_tables
  (lambda () {
    (set! field '())
    (set! cells_x '())
    (set! cells_y '())
    (set! t_i 0)
    (while (< t_i (* COLS ROWS)) {
      (push field 0)
      (set! t_i (+ t_i 1))
    })
    (set! t_i 0)
    (while (< t_i 4) {
      (push cells_x 0)
      (push cells_y 0)
      (set! t_i (+ t_i 1))
    })
  }))

(build_tables)

;; ============================================================================
;; 场地读写
;; ============================================================================

(define field_at (lambda (x y) (get_item field (+ (* y COLS) x))))
(define field_set (lambda (x y v) (set_item! field (+ (* y COLS) x) v)))

;; 整场清空（reset 与测试用）
(define clear_field
  (lambda () {
    (set! t_i 0)
    (while (< t_i (* COLS ROWS)) {
      (set_item! field t_i 0)
      (set! t_i (+ t_i 1))
    })
  }))

;; ============================================================================
;; 几何：格子坐标与碰撞
;; ============================================================================

;; 取 (pc, rt) 的 4 个格子坐标写入 cells_x / cells_y（顺时针旋转 rt 次）
(define fill_cells
  (lambda (pc rt) {
    (set! t_i 0)
    (while (< t_i 4) {
      (set! t_x (get_item SHAPES (+ (* pc 8) (* t_i 2))))
      (set! t_y (get_item SHAPES (+ (* pc 8) (+ (* t_i 2) 1))))
      (set! t_r 0)
      (while (< t_r rt) {
        (set! t_t t_x)
        (set! t_x (- 3 t_y))
        (set! t_y t_t)
        (set! t_r (+ t_r 1))
      })
      (set_item! cells_x t_i t_x)
      (set_item! cells_y t_i t_y)
      (set! t_i (+ t_i 1))
    })
  }))

;; 碰撞检测：1 = 冲突，0 = 合法（y<0 的格子在场地上方，视为合法）
(define collide
  (lambda (pc rt x0 y0) {
    (fill_cells pc rt)
    (set! t_hit 0)
    (set! t_i 0)
    (while (and (< t_i 4) (== t_hit 0)) {
      (set! t_x (+ x0 (get_item cells_x t_i)))
      (set! t_y (+ y0 (get_item cells_y t_i)))
      (if (or (< t_x 0) (or (>= t_x COLS) (>= t_y ROWS)))
          (set! t_hit 1)
          (if (and (>= t_y 0) (not (== (field_at t_x t_y) 0)))
              (set! t_hit 1)))
      (set! t_i (+ t_i 1))
    })
    t_hit
  }))

;; ============================================================================
;; 生成 / 锁定 / 消行 / 动作
;; ============================================================================

;; 把当前方块放回入场位置；无法入场即游戏结束
(define place_new
  (lambda () {
    (set! rot 0)
    (set! px 3)
    (set! py -1)
    (if (== (collide piece rot px py) 1) (set! game_over 1))
  }))

;; 以指定方块生成（测试/AI 用；不消耗 next_piece 队列）
(define spawn_piece
  (lambda (pc) {
    (set! piece pc)
    (place_new)
  }))

;; 从队列取下一个方块并抽新的 next_piece
(define spawn_next
  (lambda () {
    (set! piece next_piece)
    (set! next_piece (Math.floor (* 7 (Math.random))))
    (place_new)
  }))

;; 锁定当前方块 → 消行 → 计分 → 生成下一个
(define lock_and_clear
  (lambda () {
    (fill_cells piece rot)
    (set! t_i 0)
    (while (< t_i 4) {
      (set! t_x (+ px (get_item cells_x t_i)))
      (set! t_y (+ py (get_item cells_y t_i)))
      (if (and (>= t_y 0)
               (and (< t_y ROWS) (and (>= t_x 0) (< t_x COLS))))
          (field_set t_x t_y (+ piece 1)))
      (set! t_i (+ t_i 1))
    })

    ;; 消行：自下而上；满行整体下移后原地重查同一行
    (set! t_cleared 0)
    (set! t_y (- ROWS 1))
    (while (>= t_y 0) {
      (set! t_full 1)
      (set! t_x 0)
      (while (< t_x COLS) {
        (if (== (field_at t_x t_y) 0) (set! t_full 0))
        (set! t_x (+ t_x 1))
      })
      (if (== t_full 1)
          { ; 满行：上方各行下移一格，顶行清空，行号不动
            (set! t_yy t_y)
            (while (> t_yy 0) {
              (set! t_x 0)
              (while (< t_x COLS) {
                (field_set t_x t_yy (field_at t_x (- t_yy 1)))
                (set! t_x (+ t_x 1))
              })
              (set! t_yy (- t_yy 1))
            })
            (set! t_x 0)
            (while (< t_x COLS) {
              (field_set t_x 0 0)
              (set! t_x (+ t_x 1))
            })
            (set! t_cleared (+ t_cleared 1))
          }
          (set! t_y (- t_y 1)))
    })

    (if (> t_cleared 0) {
      (set! score (+ score (* (get_item SCORE_TABLE (- t_cleared 1)) level)))
      (set! lines (+ lines t_cleared))
      (set! level (+ (Math.floor (/ lines 10)) 1))
    })

    (spawn_next)
  }))

;; 下落一格；落不下去则锁定
(define step_down
  (lambda () {
    (if (== (collide piece rot px (+ py 1)) 0)
        (set! py (+ py 1))
        (lock_and_clear))
  }))

;; 左右平移（-1 左 / +1 右）
(define move
  (lambda (dx) {
    (if (== game_over 0)
        (if (== (collide piece rot (+ px dx) py) 0) (set! px (+ px dx))))
  }))

;; 顺时针旋转 + 简单踢墙（原位/左1/右1/左2/右2）
(define rotate_piece
  (lambda () {
    (if (== game_over 0) {
      (set! t_r (mod (+ rot 1) 4))
      (set! t_k 0)
      (set! t_ok 0)
      (while (and (< t_k 5) (== t_ok 0)) {
        (set! t_kick (get_item KICKS t_k))
        (if (== (collide piece t_r (+ px t_kick) py) 0) {
          (set! rot t_r)
          (set! px (+ px t_kick))
          (set! t_ok 1)
        })
        (set! t_k (+ t_k 1))
      })
    })
  }))

;; 直接落底并锁定
(define hard_drop
  (lambda () {
    (if (== game_over 0) {
      (while (== (collide piece rot px (+ py 1)) 0) (set! py (+ py 1)))
      (lock_and_clear)
    })
  }))

;; 落地间隔（ms）：随关卡递减，下限 100ms
(define fall_ms
  (lambda (lv) {
    (set! t_fm (- 800 (* 70 (- lv 1))))
    (if (< t_fm 100) 100 t_fm)
  }))

;; 重力累加器清零（手动速降后重置计时，同 C 版 fall_acc = 0）
(define reset_fall
  (lambda () (set! fall_acc 0)))

;; 时间推进：重力累计到间隔即下落一格（调用方在暂停/结束时不要调用）
(define advance
  (lambda (ms) {
    (if (== game_over 0) {
      (set! fall_acc (+ fall_acc ms))
      (if (>= fall_acc (fall_ms level)) {
        (set! fall_acc 0)
        (step_down)
      })
    })
  }))

;; 开局（清场 + 抽号 + 生成首块）
(define reset
  (lambda () {
    (clear_field)
    (set! score 0)
    (set! lines 0)
    (set! level 1)
    (set! fall_acc 0)
    (set! game_over 0)
    (set! next_piece (Math.floor (* 7 (Math.random))))
    (spawn_next)
  }))