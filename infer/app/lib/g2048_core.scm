;; ============================================================================
;; 2048 逻辑内核（G2048）—— 纯逻辑，不依赖 Gfx / Touch / Fs，可脱离界面单独回归
;;
;; 分工（同 app/lib/tetris_core.scm、app/lib/gomoku_core.scm 的惯例）：
;;   · 本文件 —— 棋盘、合并/移动、随机落子、终局判定、撤销历史、存档文本序列化；
;;   · app/2048.scm —— 布局、绘制、滑动手势、节拍、文件读写。
;;
;; 棋盘表示：**指数**而不是数值。
;;   bd[i] = 0 表示空格；k >= 1 表示该格数值为 2^k。
;;   好处：合并就是把相邻的两个 k 变成 k+1（一次加法），比较是否相等就是整数比较
;;   （wasm32 单精度下 2^18 以内的整数逐位精确，永远不会出现 2048 与 2047.999 的
;;   浮点相等性问题）；分数也用一张 2 的幂表取值，全程整数。
;;
;; 行/列的处理顺序表：把四个方向统一成「每条线给 4 个扁平下标、靠拢端在前」的取值
;; 方式，于是四个方向共用同一个合并算法（LINE_IDX 是 4 方向 × 4 条线 × 4 格 = 64 项）。
;;
;; 动效来源表 an_src[i]：每格方块的「从哪里滑过来」（-1 = 该格无方块或方块是新生成的）。
;;   界面层用它做滑动动画；本内核只负责如实记录（合并时记**滑过来那一枚**的来源，
;;   因为合并后原位那一枚看上去是被它吃掉了）。
;;
;; 解释器约束（manual.md，逐条遵守）：
;;   · 算术与比较恒二参数（三参 +/* 会污染操作数栈）；没有一元取负 → 写 (- 0 x)；
;;   · 没有 let / list / append / map / vector；'() 是模块级静态槽 →
;;     所有表都用 (cdr '(0)) 现场造新表（mk_list）；
;;   · 块内不 define（define 只写在 lambda 体顶层）；循环体里只用模块级 set!；
;;   · cond 每个子句只接受单个表达式；if 的分支数必须写全；
;;   · / 与 mod 恒返回浮点（下标一律 Math.floor / Math.round 收口）。
;; ============================================================================

(native Math)
(native String)
(native System)

;; ---- 常量表 ---------------------------------------------------------------

;; 2 的幂表：下标就是指数。用查表而不是 (pow 2 k)：结果恒为整数、逐位精确。
(define VALS '(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144))

;; 最大指数（VALS 的末下标；实际远达不到）
(define EXP_MAX 17)

;; 达成即「赢」的指数：2^11 = 2048
(define EXP_WIN 11)

;; 撤销深度（快照数；每份快照 17 个数 = 16 格 + 分数）
(define UND_MAX 10)

;; 4 方向 × 4 条线 × 4 格的行进顺序（**靠拢端在前**）：
;;   dir 0 = 左（每行从 c=0 读起）    dir 1 = 右（每行从 c=3 读起）
;;   dir 2 = 上（每列从 r=0 读起）    dir 3 = 下（每列从 r=3 读起）
;; 扁平下标 = r*4 + c
(define LINE_IDX '(
     0  1  2  3    4  5  6  7    8  9 10 11   12 13 14 15
     3  2  1  0    7  6  5  4   11 10  9  8   15 14 13 12
     0  4  8 12    1  5  9 13    2  6 10 14    3  7 11 15
    12  8  4  0   13  9  5  1   14 10  6  2   15 11  7  3))

;; ---- 小工具 ---------------------------------------------------------------

;; 造一张 n 个 0 的新表（'() 字面量是静态槽，不能直接往里 push）
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

;; ---- 状态 -----------------------------------------------------------------

(define bd (mk_list 16))       ; 棋盘（指数）
(define an_src (mk_list 16))   ; 每格的动效来源格下标（-1 = 无来源）
(define LN (mk_list 4))        ; 单条线的合并结果（指数）
(define SRCP (mk_list 4))      ; 单条线的合并结果对应的来源下标
(define und (mk_list 0))       ; 撤销栈（扁平：16 格 + 分数）
(define p_tok (mk_list 0))     ; 读档时的整数词表

(define score 0)               ; 本局分数
(define best 0)                ; 历史最高分
(define won 0)                 ; 本局是否已确认过「达成 2048」
(define nmove 0)               ; 本局步数
(define und_n 0)               ; 撤销栈里的快照数

;; 行合并的流式状态
(define ln_p 0)      ; 已产出的格数
(define ln_prev 0)   ; 待配对的前一枚（0 = 无）
(define ln_srcp -1)  ; 待配对那一枚的来源下标
(define ln_gain 0)   ; 本行得分
(define ln_nv 0)     ; 合并后的新指数（夹住表尾用）

;; 移动过程的临时量（m_ 前缀，只在 move 的调用链里用）
(define m_base 0) (define m_li 0) (define m_bi 0) (define m_k 0)
(define m_di 0) (define m_v 0) (define m_s 0) (define m_gain 0) (define m_moved 0)

;; 落子 / 局面 / 撤销的临时量
(define s_n 0) (define s_i 0) (define s_pick -1) (define s_r 0)
(define o_i 0) (define o_r 0) (define o_c 0) (define o_v 0) (define o_free 0) (define o_eq 0)
(define u_i 0) (define u_k 0)

;; 序列化 / 反序列化的临时量
(define t_n 0) (define t_t 1.0) (define t_d 0) (define t_rem 0) (define t_s "")
(define p_s "") (define p_i 0) (define p_n 0) (define p_cp 0) (define p_cur 0)
(define p_in 0) (define p_cnt 0) (define p_ok 1)

;; ---- 伪随机（LCG；状态恒 < 2048，乘法 < 2^19，wasm32 单精度下逐位精确） ------

(define lcg 12345)

(define rnd
  (lambda ()
    {
      (set! lcg (mod (+ (* lcg 33) 7) 2048))
      (/ lcg 2048.0)
    }))

;; 用宿主时间戳的低位播种：mod 会把参数转成浮点、wasm32 尾数只有 19 位，时间戳低位
;; 会被抹平，故用位运算把低位折叠进 [0,2048)（同 app/mlp.scm 的做法）。
(define reseed
  (lambda ()
    {
      (define t 0)
      (define s 0)
      (define i 0)
      (set! t (System.timestamp))
      (set! s (Math.bit_and t 2047))
      (set! s (Math.bit_xor s (Math.bit_and (Math.bit_lshr t 11) 2047)))
      (set! s (Math.bit_xor s (Math.bit_and (Math.bit_lshr t 21) 2047)))
      (if (== s 0) (set! s 1) #f)
      (set! lcg s)
      (set! i 0)
      (while (< i 8)
        {
          (rnd)
          (set! i (+ i 1))
        })
    }))

;; ---- 单条线的合并 ---------------------------------------------------------

;; 把一枚非空格喂进合并器：与待配对的 ln_prev 相同就合并（指数 +1、得分为新值），
;; 否则把待配对的那枚落地、自己成为新的待配对。
(define feed
  (lambda (v fpos)
    (if (== v 0)
        #f
        (if (== ln_prev 0)
            {
              (set! ln_prev v)
              (set! ln_srcp fpos)
            }
            (if (== ln_prev v)
                {
                  (set! ln_nv (+ v 1))
                  (if (> ln_nv EXP_MAX) (set! ln_nv EXP_MAX) #f)   ; 夹住表尾（实际不可达）
                  (set_item! LN ln_p ln_nv)
                  (set_item! SRCP ln_p fpos)      ; 合并：来源记滑过来的那一枚
                  (set! ln_gain (+ ln_gain (get_item VALS ln_nv)))
                  (set! ln_p (+ ln_p 1))
                  (set! ln_prev 0)
                }
                {
                  (set_item! LN ln_p ln_prev)
                  (set_item! SRCP ln_p ln_srcp)
                  (set! ln_p (+ ln_p 1))
                  (set! ln_prev v)
                  (set! ln_srcp fpos)
                })))))

;; 收尾：落地最后一枚待配对的，其余补 0 到满 4 格
(define ln_flush
  (lambda ()
    {
      (if (> ln_prev 0)
          {
            (set_item! LN ln_p ln_prev)
            (set_item! SRCP ln_p ln_srcp)
            (set! ln_p (+ ln_p 1))
          }
          #f)
      (while (< ln_p 4)
        {
          (set_item! LN ln_p 0)
          (set_item! SRCP ln_p -1)
          (set! ln_p (+ ln_p 1))
        })
    }))

;; ---- 移动 -----------------------------------------------------------------

;; 朝 dir 方向推一步。返回 1 = 棋盘发生了变化（调用方据此落子/记历史），0 = 没动。
;; 顺带累加 score / nmove，并写好 an_src。
(define move
  (lambda (dir)
    {
      (set! m_gain 0)
      (set! m_moved 0)
      (set! m_base (* dir 16))
      (set! m_li 0)
      (while (< m_li 4)
        {
          (set! m_bi (+ m_base (* m_li 4)))
          (set! ln_p 0)
          (set! ln_prev 0)
          (set! ln_srcp -1)
          (set! ln_gain 0)
          (set! m_k 0)
          (while (< m_k 4)
            {
              (set! m_di (get_item LINE_IDX (+ m_bi m_k)))
              (feed (get_item bd m_di) m_di)
              (set! m_k (+ m_k 1))
            })
          (ln_flush)
          ;; 写回棋盘 + 记录动效来源
          (set! m_k 0)
          (while (< m_k 4)
            {
              (set! m_di (get_item LINE_IDX (+ m_bi m_k)))
              (set! m_v (get_item LN m_k))
              (set! m_s (get_item SRCP m_k))
              (if (not (== (get_item bd m_di) m_v))
                  {
                    (set_item! bd m_di m_v)
                    (set! m_moved 1)
                  }
                  #f)
              (if (== m_v 0)
                  (set_item! an_src m_di -1)
                  (set_item! an_src m_di m_s))
              (set! m_k (+ m_k 1))
            })
          (set! m_gain (+ m_gain ln_gain))
          (set! m_li (+ m_li 1))
        })
      (if (== m_moved 1)
          {
            (set! score (+ score m_gain))
            (set! nmove (+ nmove 1))
            1
          }
          0)
    }))

;; ---- 落子 -----------------------------------------------------------------

;; 在一个随机空格里放一枚新方块（90% 是 2，10% 是 4）。返回该格下标，无空格返回 -1。
(define spawn
  (lambda ()
    {
      (set! s_n 0)
      (set! s_i 0)
      (while (< s_i 16)
        {
          (if (== (get_item bd s_i) 0) (set! s_n (+ s_n 1)) #f)
          (set! s_i (+ s_i 1))
        })
      (if (== s_n 0)
          -1
          {
            (set! s_r (Math.floor (* (rnd) s_n)))
            (set! s_pick -1)
            (set! s_n 0)
            (set! s_i 0)
            (while (< s_i 16)
              {
                (if (== (get_item bd s_i) 0)
                    {
                      (if (== s_n s_r)
                          (if (== s_pick -1) (set! s_pick s_i) #f)
                          #f)
                      (set! s_n (+ s_n 1))
                    }
                    #f)
                (set! s_i (+ s_i 1))
              })
            (if (< (rnd) 0.9)
                (set_item! bd s_pick 1)
                (set_item! bd s_pick 2))
            (set_item! an_src s_pick -1)
            s_pick
          })
    }))

;; ---- 局面查询 -------------------------------------------------------------

(define cell (lambda (i) (get_item bd i)))

;; 是否还能走：有空格，或存在相邻（右/下）相等的两格。返回 1 = 终局。
(define over
  (lambda ()
    {
      (set! o_free 0)
      (set! o_i 0)
      (while (< o_i 16)
        {
          (if (== (get_item bd o_i) 0) (set! o_free 1) #f)
          (set! o_i (+ o_i 1))
        })
      (if (== o_free 1)
          0
          {
            (set! o_eq 0)
            (set! o_r 0)
            (while (< o_r 4)
              {
                (set! o_c 0)
                (while (< o_c 4)
                  {
                    (set! o_v (get_item bd (+ (* o_r 4) o_c)))
                    (if (< o_c 3)
                        (if (== o_v (get_item bd (+ (* o_r 4) (+ o_c 1))))
                            (set! o_eq 1)
                            #f)
                        #f)
                    (if (< o_r 3)
                        (if (== o_v (get_item bd (+ (* (+ o_r 1) 4) o_c)))
                            (set! o_eq 1)
                            #f)
                        #f)
                    (set! o_c (+ o_c 1))
                  })
                (set! o_r (+ o_r 1))
              })
            (if (== o_eq 1) 0 1)
          })
    }))

;; 棋盘上是否已有 2048（EXP_WIN）。返回 1 / 0。
(define has_win
  (lambda ()
    {
      (set! o_i 0)
      (set! o_eq 0)
      (while (< o_i 16)
        {
          (if (>= (get_item bd o_i) EXP_WIN) (set! o_eq 1) #f)
          (set! o_i (+ o_i 1))
        })
      o_eq
    }))

;; 棋盘的数值总和（回归测试用：一次移动 + 落子前后的守恒检查）
(define sum
  (lambda ()
    {
      (define t 0)
      (set! o_i 0)
      (set! t 0)
      (while (< o_i 16)
        {
          ;; 空格（指数 0）贡献 0：VALS 的下标就是指数，不能直接查表（VALS[0] 是 1）
          (if (> (get_item bd o_i) 0)
              (set! t (+ t (get_item VALS (get_item bd o_i))))
              #f)
          (set! o_i (+ o_i 1))
        })
      t
    }))

;; ---- 撤销 -----------------------------------------------------------------

;; 记一份快照（每次成功移动后调用）
(define snapshot
  (lambda ()
    {
      (set! u_i 0)
      (while (< u_i 16)
        {
          (push und (get_item bd u_i))
          (set! u_i (+ u_i 1))
        })
      (push und score)
      (set! und_n (+ und_n 1))
      ;; 超过深度上限：从头部丢掉最旧的一份（17 个数；cdr 是拷贝，摊还 O(n) 但 n 很小）
      (if (> und_n UND_MAX)
          {
            (set! u_k 0)
            (while (< u_k 17)
              {
                (set! und (cdr und))
                (set! u_k (+ u_k 1))
              })
            (set! und_n (- und_n 1))
          }
          #f)
    }))

;; 丢弃最近一份快照（先记快照才发现这一步走不动时用；不算一次撤销）
(define unsnapshot
  (lambda ()
    (if (> und_n 0)
        {
          (pop und)
          (set! u_i 0)
          (while (< u_i 16)
            {
              (pop und)
              (set! u_i (+ u_i 1))
            })
          (set! und_n (- und_n 1))
          1
        }
        0)))

;; 撤销一步：恢复棋盘与分数。返回 1 = 撤销成功，0 = 没有可撤的。
(define undo
  (lambda ()
    {
      (set! u_i 0)
      (if (== und_n 0)
          0
          {
            (set! score (pop und))
            (set! u_i 15)
            (while (>= u_i 0)
              {
                (set_item! bd u_i (pop und))
                (set_item! an_src u_i -1)
                (set! u_i (- u_i 1))
              })
            (set! und_n (- und_n 1))
            (if (> nmove 0) (set! nmove (- nmove 1)) #f)
            1
          })
    }))

;; 玩家确认了「达成 2048」提示（此后不再弹，但要存档记住）
(define mark_won
  (lambda ()
    {
      (set! won 1)
      1
    }))

;; ---- 新局 -----------------------------------------------------------------

;; 清空一切并落两枚新方块（best 保留）
(define reset
  (lambda ()
    {
      (set! o_i 0)
      (while (< o_i 16)
        {
          (set_item! bd o_i 0)
          (set_item! an_src o_i -1)
          (set! o_i (+ o_i 1))
        })
      (set! score 0)
      (set! won 0)
      (set! nmove 0)
      (while (> (length und) 0) { (pop und) })
      (set! und_n 0)
      (spawn)
      (spawn)
      1
    }))

;; ---- 序列化（存档） -------------------------------------------------------

;; 非负整数 → 字符串（不用 Math.to_fixed + atom_to_string：wasm32 上 %g 会多出小数位）
(define itoa
  (lambda (vv)
    {
      (define n 1)
      (define t 10.0)
      (define i 1)
      (while (>= vv t)
        {
          (set! t (* t 10.0))
          (set! n (+ n 1))
        })
      (set! t (/ t 10.0))
      (set! t_rem vv)
      (set! t_s "")
      (set! i 0)
      (while (< i n)
        {
          (set! t_d (Math.floor (/ t_rem t)))
          (set! t_s (String.concat t_s (String.atom_to_string t_d)))
          (set! t_rem (- t_rem (* t_d t)))
          (set! t (/ t 10.0))
          (set! i (+ i 1))
        })
      t_s
    }))

;; 存档文本：16 个指数 + score + best + won（空格分隔的 19 个非负整数）
(define save_text
  (lambda ()
    {
      (set! p_i 0)
      (set! p_s "")
      (while (< p_i 16)
        {
          (set! p_s (String.concat p_s (itoa (get_item bd p_i))))
          (set! p_s (String.concat p_s " "))
          (set! p_i (+ p_i 1))
        })
      (set! p_s (String.concat p_s (itoa score)))
      (set! p_s (String.concat p_s " "))
      (set! p_s (String.concat p_s (itoa best)))
      (set! p_s (String.concat p_s " "))
      (set! p_s (String.concat p_s (itoa won)))
      (set! p_s (String.concat p_s "\n"))
      p_s
    }))

;; 读回存档文本。返回 1 = 成功（棋盘/分数/最高分/已赢标记都已就位），0 = 无存档或损坏。
;; 解析不造中间字符串：逐码点扫一遍，遇到分隔符就把当前累积值压进 p_tok。
(define load_text
  (lambda (s)
    {
      (set! p_ok 1)
      (if (null? s)
          (set! p_ok 0)
          {
            (while (> (length p_tok) 0) { (pop p_tok) })
            (set! p_n (String.length s))
            (set! p_cnt 0)
            (set! p_cur 0)
            (set! p_in 0)
            (set! p_i 0)
            (while (< p_i p_n)
              {
                (set! p_cp (String.charCodeAt p_i s))
                (if (and (>= p_cp 48) (<= p_cp 57))
                    {
                      (set! p_cur (+ (* p_cur 10) (- p_cp 48)))
                      (set! p_in 1)
                    }
                    (if (== p_in 1)
                        {
                          (push p_tok p_cur)
                          (set! p_cnt (+ p_cnt 1))
                          (set! p_cur 0)
                          (set! p_in 0)
                        }
                        #f))
                (set! p_i (+ p_i 1))
              })
            (if (== p_in 1)
                {
                  (push p_tok p_cur)
                  (set! p_cnt (+ p_cnt 1))
                }
                #f)
            (if (< p_cnt 19) (set! p_ok 0) #f)
            ;; 棋盘取值校验：指数必须在 [0, EXP_MAX]，越界视为存档损坏
            (if (== p_ok 1)
                {
                  (set! p_i 0)
                  (while (< p_i 16)
                    {
                      (if (> (get_item p_tok p_i) EXP_MAX) (set! p_ok 0) #f)
                      (set! p_i (+ p_i 1))
                    })
                }
                #f)
            (if (== p_ok 1)
                {
                  (set! p_i 0)
                  (while (< p_i 16)
                    {
                      (set_item! bd p_i (get_item p_tok p_i))
                      (set_item! an_src p_i -1)
                      (set! p_i (+ p_i 1))
                    })
                  (set! score (get_item p_tok 16))
                  (set! best (get_item p_tok 17))
                  (set! won (get_item p_tok 18))
                  (if (< best score) (set! best score) #f)
                  (set! nmove 0)
                  (while (> (length und) 0) { (pop und) })
                  (set! und_n 0)
                }
                #f)
          })
      p_ok
    }))

;; ---- 分数簿记 -------------------------------------------------------------

;; best 只在变高时更新；返回 1 = 刷新了最高分
(define score_upd
  (lambda ()
    (if (> score best)
        {
          (set! best score)
          1
        }
        0)))