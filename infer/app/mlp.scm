;; ============================================================================
;; mlp.scm —— 多层感知机（前馈神经网络）训练与可视化（Nano「灵机应用程序」）
;;
;; 参考页面：https://bd4sur.com/html/mlp.html
;;   （2 维输入、5 类样本、全连接网络 2→6→6→6→5 + ReLU + softmax + 交叉熵，
;;     小批量 SGD；页面画「决策区域 + 样本点」与「损失曲线」两幅图）
;;
;; 本 App 相对参考页面的取舍（320x240 屏幕 + 解释器算力；实测口径见 doc/AGENTS.md：
;; 一次乘加约 30~40 条 VM 指令，单帧预算 31.25 万条指令 ≈ 10~15ms）：
;;   · 网络缩为 **2→6→6→5**（78 次乘加/样本，比参考页少一层）：决策图每一格都要跑
;;     一遍前向，格数才是预算的主要矛盾；
;;   · 决策图 32x32 格（每格 5px，覆盖输入域 [-2.2, 2.2]）。整图 1024 格一次算完要
;;     200 万条指令（6 帧），故**每拍只算 1 行**、逐拍下滑：32 行 ≈ 两轮 epoch 扫一遍。
;;     训练在同时推进，所以图永远是「最近一轮」的决策面，视觉上像慢慢长出来的
;;     （与 mandelbrot / sky3 的渐进渲染同一哲学，代价是上下相隔几行可能差半轮权重）；
;;   · 每拍 = 1 个批量（16 样本，前向+反向+一次 SGD 更新）+ 1 行决策格（32 格）
;;     + 16 个样本的全量评估 + 局部重绘（只重画变了的行与面板），实测约 26 万条 VM
;;     指令，在单帧预算（31.25 万条）之内；16 拍 = 1 轮 epoch = 256 个样本。
;;     学习率 0.2 / 批 16：约 20 轮（≈5 秒）到 loss≈0.44、准确率≈88%，之后缓慢爬升。
;;
;; 三个可视化（相对参考页面新增了两个）：
;;   · **决策面 + 样本点**（左栏）：32x32 的类别色块 + 256 个白圈彩点，训练中实时变化；
;;   · **权重热图**（右上）：三层权重矩阵并排画成 5px 色块（蓝负 / 暗零 / 红正，全体
;;     权重共用一个 ±max|W| 归一化），每层右侧一列是它的偏置——权重「长出来」的过程
;;     看得见；
;;   · **损失曲线**（底部整幅）：横轴一轮 epoch，纵轴全数据集平均交叉熵（纵轴 [0, 全程
;;     最大值]），同一行文字给出当前 loss / 准确率 / epoch / 学习率。
;;   · 另加**点按或拖动取点**：在决策图上点/拖 → 该点 5 类概率实时显示在右下（深色
;;     四角框标出取点位置），把「决策面」与「softmax 输出」这两层含义连起来。
;;
;; 解释器约束（Animac manual.md + doc/AGENTS.md 的踩坑清单，逐条遵守）：
;;   · 算术/比较恒二参数，多元必须嵌套；没有一元取负（写 (- 0.0 x)）；
;;   · 没有 let / list / append / map；define 无过程简写；define 只写在 lambda 体顶层，
;;     循环变量一律模块级 set!；
;;   · / 与 mod 恒返回浮点 → 下标一律 Math.round/floor 收口；除零与 log(0) 会终止进程，
;;     先判后除（归一化分母、softmax 分母、交叉熵概率都有下限保护）；
;;   · '() 字面量是模块级静态槽（多次 push 会累积）——三个大表（决策格缓存 1024、
;;     损失环 120、洗牌表 256）只在**加载时一次性** push 构建，此后只 set_item!
;;     （「重置」也是原地重写，不新建表）；
;;   · Gfx 的坐标参数是 uint32 —— 一律先钳到合法范围再画（负数会变成天文数字）；
;;   · 模块级临时量不跨函数调用链混用（a1/a2/z3/p5 是 fw/sm 的**输出通道**，
;;     调用方读它们之前不得再调 fw/sm，各调用点均遵守）。
;;
;; 随机性：权重初始化与样本洗牌都用本文件里的 LCG（状态 < 2048、乘法 < 2^19，wasm32
;; 的单精度下逐位精确），**种子取宿主当前时间戳的低位**（见 reseed）——每次启动、
;; 每次「重置」都从不同的初始权重开始（这正是想看的：换一次初始化，决策面长成什么样）。
;; 因此本 App 不再逐像素可复现，回归断言只用与种子无关的性质（决策面出现多种类
;; 色块、损失曲线出线、暂停后画面冻结等）；宿主回归台的时间被冻结成虚拟时钟，
;; 那里仍然是确定的。
;; ============================================================================

(native Gfx)
(native Math)
(native System)
(native Touch)

(import EX "app/lib/exit.scm")          ; 硬 A/ESC → 全局退出确认框 + 右上角「退出」按钮
(import D "app/lib/mlp_data.scm")       ; 256 个样本（2 维、5 类），由脚本生成

;; ---- 数据别名（热循环里直接引用，省一层模块成员解析） ----------------------

(define DX D.mlp_x)
(define DY D.mlp_y)
(define DN D.mlp_n)

;; ---- 布局 ----------------------------------------------------------------

(define SW 320)
(define SH 240)
(define HEAD_H 18)                      ; 与 EX 的「退出」按钮同心对齐（按钮 y 2..17）
(define TICK_MS 16)                        ; 约 1 拍/帧（与 60Hz 上屏节奏对齐）

;; 决策图：左栏 160x160，每格 5px → 32x32 格；覆盖数据域 [-R, R]（样本最远 ±2.07）
(define PX 6)
(define PY 20)
(define PS 5)
(define PG 32)
(define PW (* PG PS))
(define PH PW)
(define R 2.2)
(define STEP (/ (* 2.0 R) PG))          ; 一格对应的数据步长
(define KSCALE (/ PW (* 2.0 R)))        ; 数据 → 像素的比例

;; 右栏（x = RX..316）：上半「权重热图」，下半「该点概率」
(define RX 172)
(define RW 144)

;; 顶部两个按钮（右上角 40x18 留给 EX 的「退出」）
(define B1X 136)
(define B2X 174)
(define BTN_Y 2)
(define BTN_W 34)
(define BTN_H 16)

;; 底部损失曲线
(define LC_X 6)
(define LC_Y 184)
(define LC_W 310)
(define LC_H 53)
(define CH_X 10)                        ; 曲线绘图区
(define CH_Y 202)
(define CH_W 302)
(define CH_H 30)
(define LOSS_N 120)                     ; 曲线保留的 epoch 数

;; ---- 配色 ----------------------------------------------------------------

;; 5 类颜色（取自参考页面：青 / 黄绿 / 紫 / 粉 / 亮黄）
(define CR '(95 207 132 255 245))
(define CG '(199 237 122 126 232))
(define CB '(239 77 218 189 2))

;; ---- 超参数 --------------------------------------------------------------

(define NH 6)                           ; 两个隐层的宽度
(define NCLS 5)                         ; 类别数
(define BATCH 16)                       ; 每拍的样本数（也是 SGD 的批大小）
(define LR 0.2)                         ; 学习率
(define LRB 0.0125)                     ; = LR / BATCH（每拍一次参数更新）
(define WAMP 0.6)                       ; 权重初始化幅度（均匀分布 [-WAMP, WAMP]）
(define ROWS_PER_TICK 1)                ; 每拍推进的决策格行数（32 行约两轮扫完）
(define EVAL_PER_TICK 16)               ; 每拍评估的样本数（256/16 = 16 拍一轮）
(define TICKS_PER_EPOCH 16)             ; 16 拍 = 16 个批量 = 1 轮 epoch（256 样本）

;; ---- 网络参数 ------------------------------------------------------------
;; 权重按行主序平铺：w1[2j+i]（j 是层1 输出单元、i 是输入维）、w2[6j+i]、w3[6j+i]。
;; 这些列表都是模块级静态槽，训练里一律 set_item! 原地改写（重置也是原地重写，
;; 不重新分配）—— 因此没有任何 push/新建表，也就不会踩「'() 静态槽」的坑。

(define w1 '(0 0 0 0 0 0 0 0 0 0 0 0))
(define b1 '(0 0 0 0 0 0))
(define w2 '(0 0 0 0 0 0
             0 0 0 0 0 0
             0 0 0 0 0 0
             0 0 0 0 0 0
             0 0 0 0 0 0
             0 0 0 0 0 0))
(define b2 '(0 0 0 0 0 0))
(define w3 '(0 0 0 0 0 0
             0 0 0 0 0 0
             0 0 0 0 0 0
             0 0 0 0 0 0
             0 0 0 0 0 0))
(define b3 '(0 0 0 0 0))

;; 前向中间量（fw / sm 的输出通道，见文件头「模块级临时量」约定）
(define a1 '(0 0 0 0 0 0))
(define a2 '(0 0 0 0 0 0))
(define z3 '(0 0 0 0 0))
(define p5 '(0 0 0 0 0))
(define pred 0)                         ; sm 给出的 argmax 类别

;; 反向传播的梯度累加器与中间量
(define gw1 '(0 0 0 0 0 0 0 0 0 0 0 0))
(define gb1 '(0 0 0 0 0 0))
(define gw2 '(0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0))
(define gb2 '(0 0 0 0 0 0))
(define gw3 '(0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0
              0 0 0 0 0 0))
(define gb3 '(0 0 0 0 0))
(define d3 '(0 0 0 0 0))                ; dL/dz3
(define d2 '(0 0 0 0 0 0))              ; dL/dz2
(define d1 '(0 0 0 0 0 0))              ; dL/dz1

;; ---- 训练状态 ------------------------------------------------------------

(define lcg 1)                          ; 线性同余伪随机数状态（reset_all 里按时间戳播种）
(define running 1)                      ; 1=训练中 0=暂停
(define epoch 0)                        ; 已完成的 epoch 数
(define tick_in_epoch 0)                ; 本轮内已推进的拍数（0..15）
(define ep_loss 0.0)                    ; 本轮交叉熵累加（全数据集）
(define ep_ok 0)                        ; 本轮预测正确数
(define cur_loss 0.0)                   ; 最近一轮的平均交叉熵
(define cur_acc 0.0)                    ; 最近一轮的准确率
(define perm_pos 0)                     ; 本轮已消费的洗牌位置
(define mrow 0)                         ; 决策图当前扫描行（0..PG-1，逐拍下滑）
(define probe_x 0.0)                    ; 取点坐标（数据域）
(define probe_y 0.0)
(define probe_px 86)                    ; 取点标记的像素位置（拖拽时用于算重画行）
(define probe_py 100)
(define armed 0)                        ; 触屏：「按下时命中的控件」
(define tp 0)                           ; 本拍是否有按下沿
(define tr 0)                           ; 本拍是否有松开沿
(define tx 0)                           ; 触点（像素）
(define ty 0)
(define _timer 0)

;; 决策格类别缓存：32*32 = 1024（加载时一次性 push 构建，此后只 set_item!）
(define map_lab '())
;; 损失曲线的环形缓冲：LOSS_N 项
(define loss_ring '())
(define loss_idx 0)                     ; 下一个写入位置
(define loss_cnt 0)                     ; 已写入的轮数（≤ LOSS_N）
(define loss_max 1.0)                   ; 纵轴上限（全程取的最大值，只增不减，轴不抖）
;; 样本的洗牌表（0..DN-1）
(define perm '())
;; 样本的像素坐标（加载时算一次，绘制时不再做浮点变换；已钳在绘图区内）
(define pt_px '())
(define pt_py '())

;; 建表（**一次性**：'() 是模块级静态槽，push 之后不能再用同一字面量重建）
(define build_i 0)
(while (< build_i 1024) {
  (push map_lab 0)
  (set! build_i (+ build_i 1))
})
(set! build_i 0)
(while (< build_i LOSS_N) {
  (push loss_ring 0.0)
  (set! build_i (+ build_i 1))
})
(set! build_i 0)
(while (< build_i DN) {
  (push perm build_i)
  (push pt_px 0)
  (push pt_py 0)
  (set! build_i (+ build_i 1))
})

;; ---- 小工具 --------------------------------------------------------------

;; ---- 数字绘制（**运行期零字符串分配**） -----------------------------------
;; 解释器把**运行期构造**的短字符串驻留（intern），驻留表是 GC 根、永不释放
;; （doc/AGENTS.md 2026-10-31 条目）。逐帧 `String.concat` / `atom_to_string` 会稳定
;; 涨内存，故本 App 的动态数字一律用 10 个**编译期字面量**字模逐位贴出——运行期
;; 不产生任何字符串（与 callcc.scm 的 draw_num 同一手法）。

(define DGS '("0" "1" "2" "3" "4" "5" "6" "7" "8" "9"))
(define DW 7)                           ; 字模步进（实测 12px 字模 ASCII 步进≈6.9px）

;; 非负整数的十进制位数
(define ndigits
  (lambda (vv)
    {
      (define n 1)
      (define t 10.0)
      (while (>= vv t) {
        (set! t (* t 10.0))
        (set! n (+ n 1))
      })
      n
    }))

;; 画一个非负整数（pad 大于实际位数时左补零），返回占用的像素宽度
(define draw_int
  (lambda (vv pad x y r g b)
    {
      (define n (ndigits vv))
      (define nd (if (> pad n) pad n))
      (define t 1.0)
      (define k 1)
      (define i 0)
      (define d 0)
      (define rem vv)
      (while (< k nd) {
        (set! t (* t 10.0))
        (set! k (+ k 1))
      })
      (while (< i nd) {
        (set! d (Math.floor (/ rem t)))
        (Gfx.draw_text (get_item DGS d) (+ x (* i DW)) y r g b)
        (set! rem (- rem (* d t)))
        (set! t (/ t 10.0))
        (set! i (+ i 1))
      })
      (* nd DW)
    }))

;; 画定点小数（负数带 '-'），返回占用宽度
(define draw_fixed
  (lambda (v dec x y r g b)
    {
      (define neg (if (< v 0.0) 1 0))
      (define a (if (== neg 1) (- 0.0 v) v))
      (define sc 1.0)
      (define k 0)
      (define tt 0.0)
      (define ip 0.0)
      (define fp 0.0)
      (define cx x)
      (while (< k dec) {
        (set! sc (* sc 10.0))
        (set! k (+ k 1))
      })
      (set! tt (Math.round (* a sc)))
      (set! ip (Math.floor (/ tt sc)))
      (set! fp (- tt (* ip sc)))
      (if (== neg 1)
          {
            (Gfx.draw_text "-" cx y r g b)
            (set! cx (+ cx DW))
          }
          #f)
      (set! cx (+ cx (draw_int ip 1 cx y r g b)))
      (Gfx.draw_text "." cx y r g b)
      (set! cx (+ cx DW))
      (set! cx (+ cx (draw_int fp dec cx y r g b)))
      (- cx x)
    }))

;; 伪随机数 [0,1)：状态恒 < 2048，乘法 < 2^19（wasm32 单精度下逐位精确）
(define rnd
  (lambda ()
    {
      (set! lcg (mod (+ (* lcg 33) 7) 2048))
      (/ lcg 2048.0)
    }))

;; 用宿主当前时间戳给 LCG 播种（每次启动、每次「重置」都不同）
;; 为什么不用 (mod ts 2048)：`mod` 会把参数转成浮点，而 wasm32 尾数只有 19 位，
;; 27 位的时间戳转成浮点后低位全被抹平，(mod ts 2048) 只剩少数几个可能值（种子退化）。
;; 位运算全程是整数、逐位精确，故用 bit_and / bit_lshr / bit_xor 把低位折叠进来。
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
      ;; 空转 8 步：让首个输出与种子的低位去相关（这个 LCG 的乘子很小）
      (set! i 0)
      (while (< i 8) {
        (rnd)
        (set! i (+ i 1))
      })
    }))

(define zero_fill
  (lambda (lst n)
    {
      (define i 0)
      (while (< i n) {
        (set_item! lst i 0.0)
        (set! i (+ i 1))
      })
    }))

(define rand_fill
  (lambda (lst n amp)
    {
      (define i 0)
      (while (< i n) {
        (set_item! lst i (* (- (* (rnd) 2.0) 1.0) amp))
        (set! i (+ i 1))
      })
    }))

(define in_rect
  (lambda (x y bx by bw bh)
    (and (and (>= x bx) (< x (+ bx bw))) (and (>= y by) (< y (+ by bh))))))

;; 数值 → 颜色通道（0..255 的整数，写入颜色通道前先钳）
(define clamp255
  (lambda (v)
    {
      (define r (Math.round v))
      (if (< r 0.0) (set! r 0.0) #f)
      (if (> r 255.0) (set! r 255.0) #f)
      r
    }))

;; ---- 前向 ----------------------------------------------------------------
;; fw：两层 ReLU + 一层线性 logits（决策图只需要 argmax，故不做 softmax）
;; 结果写在 a1 / a2 / z3 上。

(define fw
  (lambda (x0 x1)
    {
      (define j 0)
      (define k 0)
      (define s 0.0)
      ;; 层1：a1[j] = relu(w1[2j]*x0 + w1[2j+1]*x1 + b1[j])
      (set! j 0)
      (while (< j NH) {
        (set! s (+ (get_item b1 j)
                   (+ (* (get_item w1 (* j 2)) x0)
                      (* (get_item w1 (+ (* j 2) 1)) x1))))
        (if (> s 0.0) (set_item! a1 j s) (set_item! a1 j 0.0))
        (set! j (+ j 1))
      })
      ;; 层2：a2[j] = relu(Σ_i w2[6j+i]*a1[i] + b2[j])
      (set! j 0)
      (while (< j NH) {
        (set! s (get_item b2 j))
        (set! k 0)
        (while (< k NH) {
          (set! s (+ s (* (get_item w2 (+ (* j NH) k)) (get_item a1 k))))
          (set! k (+ k 1))
        })
        (if (> s 0.0) (set_item! a2 j s) (set_item! a2 j 0.0))
        (set! j (+ j 1))
      })
      ;; 层3：z3[j] = Σ_i w3[6j+i]*a2[i] + b3[j]（logits，未过 softmax）
      (set! j 0)
      (while (< j NCLS) {
        (set! s (get_item b3 j))
        (set! k 0)
        (while (< k NH) {
          (set! s (+ s (* (get_item w3 (+ (* j NH) k)) (get_item a2 k))))
          (set! k (+ k 1))
        })
        (set_item! z3 j s)
        (set! j (+ j 1))
      })
    }))

;; argmax_z3：只取 logits 的最大类（决策面用；softmax 是单调的，argmax 不变）
(define argmax_z3
  (lambda ()
    {
      (define j 1)
      (define bi 0)
      (define bv (get_item z3 0))
      (while (< j NCLS) {
        (if (> (get_item z3 j) bv)
            {
              (set! bv (get_item z3 j))
              (set! bi j)
            }
            #f)
        (set! j (+ j 1))
      })
      (set! pred bi)
    }))

;; sm：把 z3 变成 softmax 概率 p5，并把 argmax 类写进 pred（数值稳定的减法）
(define sm
  (lambda ()
    {
      (define j 1)
      (define m (get_item z3 0))
      (define s 0.0)
      (define e 0.0)
      (define bi 0)
      (define bv 0.0)
      (while (< j NCLS) {
        (if (> (get_item z3 j) m) (set! m (get_item z3 j)) #f)
        (set! j (+ j 1))
      })
      (set! s 0.0)
      (set! j 0)
      (while (< j NCLS) {
        (set! e (Math.exp (- (get_item z3 j) m)))
        (set_item! p5 j e)
        (set! s (+ s e))
        (set! j (+ j 1))
      })
      ;; 最大项恒为 exp(0)=1 ⇒ s ≥ 1；这里仍作下限保护，杜绝除零终止进程
      (if (< s 0.000001) (set! s 1.0) #f)
      (set! j 0)
      (while (< j NCLS) {
        (set_item! p5 j (/ (get_item p5 j) s))
        (set! j (+ j 1))
      })
      ;; argmax
      (set! bv (get_item z3 0))
      (set! bi 0)
      (set! j 1)
      (while (< j NCLS) {
        (if (> (get_item z3 j) bv)
            {
              (set! bv (get_item z3 j))
              (set! bi j)
            }
            #f)
        (set! j (+ j 1))
      })
      (set! pred bi)
    }))

;; ---- 训练一步：前向 + 反向（逐样本累加到 gw*/gb*，一批结束后统一更新一次） --
;; 损失/准确率的统计不在这里做：小批量口径逐批起伏太大，画成曲线只是噪声带，
;; 改由 eval_samples 的全数据集口径给出（见下）。

(define train_one
  (lambda (sidx)
    {
      (define j 0)
      (define k 0)
      (define s 0.0)
      (define cls (get_item DY sidx))
      (define x0 (get_item DX (* sidx 2)))
      (define x1 (get_item DX (+ (* sidx 2) 1)))
      (fw x0 x1)
      (sm)
      ;; d3[j] = p5[j] - 1{j == cls}（softmax + 交叉熵的经典形式）
      (set! j 0)
      (while (< j NCLS) {
        (if (== j cls)
            (set_item! d3 j (- (get_item p5 j) 1.0))
            (set_item! d3 j (get_item p5 j)))
        (set! j (+ j 1))
      })
      ;; 层3：grad_W3[j*6+k] += d3[j]*a2[k]，grad_b3[j] += d3[j]
      (set! j 0)
      (while (< j NCLS) {
        (set! k 0)
        (while (< k NH) {
          (set_item! gw3 (+ (* j NH) k)
                     (+ (get_item gw3 (+ (* j NH) k))
                        (* (get_item d3 j) (get_item a2 k))))
          (set! k (+ k 1))
        })
        (set_item! gb3 j (+ (get_item gb3 j) (get_item d3 j)))
        (set! j (+ j 1))
      })
      ;; d2[k] = (Σ_j w3[j*6+k]*d3[j]) * relu'(a2[k])
      (set! k 0)
      (while (< k NH) {
        (set! s 0.0)
        (set! j 0)
        (while (< j NCLS) {
          (set! s (+ s (* (get_item w3 (+ (* j NH) k)) (get_item d3 j))))
          (set! j (+ j 1))
        })
        (if (> (get_item a2 k) 0.0) (set_item! d2 k s) (set_item! d2 k 0.0))
        (set! k (+ k 1))
      })
      ;; 层2
      (set! j 0)
      (while (< j NH) {
        (set! k 0)
        (while (< k NH) {
          (set_item! gw2 (+ (* j NH) k)
                     (+ (get_item gw2 (+ (* j NH) k))
                        (* (get_item d2 j) (get_item a1 k))))
          (set! k (+ k 1))
        })
        (set_item! gb2 j (+ (get_item gb2 j) (get_item d2 j)))
        (set! j (+ j 1))
      })
      ;; d1[k] = (Σ_j w2[j*6+k]*d2[j]) * relu'(a1[k])
      (set! k 0)
      (while (< k NH) {
        (set! s 0.0)
        (set! j 0)
        (while (< j NH) {
          (set! s (+ s (* (get_item w2 (+ (* j NH) k)) (get_item d2 j))))
          (set! j (+ j 1))
        })
        (if (> (get_item a1 k) 0.0) (set_item! d1 k s) (set_item! d1 k 0.0))
        (set! k (+ k 1))
      })
      ;; 层1
      (set! j 0)
      (while (< j NH) {
        (set_item! gw1 (* j 2) (+ (get_item gw1 (* j 2)) (* (get_item d1 j) x0)))
        (set_item! gw1 (+ (* j 2) 1)
                   (+ (get_item gw1 (+ (* j 2) 1)) (* (get_item d1 j) x1)))
        (set_item! gb1 j (+ (get_item gb1 j) (get_item d1 j)))
        (set! j (+ j 1))
      })
    }))

;; 参数更新：W ← W - (LR/BATCH)·grad（原地 set_item!，不新建列表）
(define apply_grads
  (lambda ()
    {
      (define i 0)
      (set! i 0)
      (while (< i 12) {
        (set_item! w1 i (- (get_item w1 i) (* LRB (get_item gw1 i))))
        (set! i (+ i 1))
      })
      (set! i 0)
      (while (< i NH) {
        (set_item! b1 i (- (get_item b1 i) (* LRB (get_item gb1 i))))
        (set! i (+ i 1))
      })
      (set! i 0)
      (while (< i 36) {
        (set_item! w2 i (- (get_item w2 i) (* LRB (get_item gw2 i))))
        (set! i (+ i 1))
      })
      (set! i 0)
      (while (< i NH) {
        (set_item! b2 i (- (get_item b2 i) (* LRB (get_item gb2 i))))
        (set! i (+ i 1))
      })
      (set! i 0)
      (while (< i 30) {
        (set_item! w3 i (- (get_item w3 i) (* LRB (get_item gw3 i))))
        (set! i (+ i 1))
      })
      (set! i 0)
      (while (< i NCLS) {
        (set_item! b3 i (- (get_item b3 i) (* LRB (get_item gb3 i))))
        (set! i (+ i 1))
      })
    }))

;; 一拍一个批量：清零梯度 → 逐样本累加 → 一次更新
(define train_batch
  (lambda ()
    {
      (define i 0)
      (zero_fill gw1 12)
      (zero_fill gb1 NH)
      (zero_fill gw2 36)
      (zero_fill gb2 NH)
      (zero_fill gw3 30)
      (zero_fill gb3 NCLS)
      (set! i 0)
      (while (< i BATCH) {
        (train_one (get_item perm (+ perm_pos i)))
        (set! i (+ i 1))
      })
      (set! perm_pos (+ perm_pos BATCH))
      (apply_grads)
    }))

;; ---- 全数据集评估（分摊到 16 拍，每拍 16 个样本） ------------------------
;; 损失曲线/准确率取**全数据集**口径（与参考页面一致）：小批量损失逐批起伏太大，
;; 画出来只是一条噪声带；全量评估每轮扫一遍 256 个样本，曲线才是可读的收敛过程。

(define eval_samples
  (lambda (i0 i1)
    {
      (define i i0)
      (define cls 0)
      (define pv 0.0)
      (while (< i i1) {
        (set! cls (get_item DY i))
        (fw (get_item DX (* i 2)) (get_item DX (+ (* i 2) 1)))
        (sm)
        (if (== pred cls) (set! ep_ok (+ ep_ok 1)) #f)
        (set! pv (get_item p5 cls))
        (if (< pv 0.000001) (set! pv 0.000001) #f)
        (set! ep_loss (+ ep_loss (- 0.0 (Math.log pv))))
        (set! i (+ i 1))
      })
    }))

;; ---- 决策面：每拍只算 ROWS_PER_TICK 行 ----------------------------------

(define map_rows
  (lambda (r0 r1)
    {
      (define gy r0)
      (define gx 0)
      (define xv 0.0)
      (define yv 0.0)
      (while (< gy r1) {
        (set! yv (- R (* STEP (+ gy 0.5))))
        (set! gx 0)
        (while (< gx PG) {
          (set! xv (+ (- 0.0 R) (* STEP (+ gx 0.5))))
          (fw xv yv)
          (argmax_z3)
          (set_item! map_lab (+ (* gy PG) gx) pred)
          (set! gx (+ gx 1))
        })
        (set! gy (+ gy 1))
      })
    }))

;; ---- 绘制 ---------------------------------------------------------------

(define draw_header
  (lambda ()
    {
      (Gfx.fill_rect 0 0 SW HEAD_H 22 24 32)
      (Gfx.draw_text "MLP 2-6-6-5" 3 3 150 200 255)
      (Gfx.draw_text "ep " 88 3 170 180 200)
      (draw_int epoch 1 109 3 170 180 200)
      (Gfx.fill_rect B1X BTN_Y BTN_W BTN_H 52 58 76)
      (Gfx.draw_rect B1X BTN_Y BTN_W BTN_H 96 104 128)
      (Gfx.draw_text (if (== running 1) "暂停" "继续") (+ B1X 5) (+ BTN_Y 2) 220 226 236)
      (Gfx.fill_rect B2X BTN_Y BTN_W BTN_H 52 58 76)
      (Gfx.draw_rect B2X BTN_Y BTN_W BTN_H 96 104 128)
      (Gfx.draw_text "重置" (+ B2X 5) (+ BTN_Y 2) 220 226 236)
      ;; **「退出」按钮必须每次重画顶栏之后补画**：它由 EX 画在 y 2..17（在顶栏范围内），
      ;; 而顶栏在「每轮 epoch 结束」「暂停/继续」时都会被整条重涂，不补画就只剩一条边
      ;; （实测：只剩 y=16 那一行可见，按钮等于被 HUD 盖掉）。EX.draw_button 还会顺带
      ;; 重新登记宿主侧的命中矩形，所以每条重画路径都走这里最稳。
      (EX.draw_button)
    }))

;; 决策格：逐行 RLE（同类连续格并成一个矩形），再把落在这些行里的样本点与取点
;; 标记补画回去——局部重绘的关键（整屏 1024 格 → 通常只有几十个矩形）
(define draw_map_rows
  (lambda (r0 r1)
    {
      (define gy r0)
      (define gx 0)
      (define run 1)
      (define c 0)
      (define i 0)
      (define yy 0)
      (define y0 (+ PY (* r0 PS)))
      (define y1 (+ PY (* r1 PS)))
      (define xv 0)
      (define yv 0)
      (while (< gy r1) {
        (set! gx 0)
        (while (< gx PG) {
          (set! c (get_item map_lab (+ (* gy PG) gx)))
          (set! run 1)
          (while (and (< (+ gx run) PG)
                      (== (get_item map_lab (+ (* gy PG) (+ gx run))) c))
            (set! run (+ run 1)))
          (Gfx.fill_rect (+ PX (* gx PS)) (+ PY (* gy PS)) (* run PS) PS
                         (get_item CR c) (get_item CG c) (get_item CB c))
          (set! gx (+ gx run))
        })
        (set! gy (+ gy 1))
      })
      ;; 样本点（白底 + 类色芯，与参考页面的「白圈彩点」同构）
      (set! i 0)
      (while (< i DN) {
        (set! yy (get_item pt_py i))
        (if (and (>= yy y0) (< yy y1))
            {
              (set! xv (get_item pt_px i))
              (set! yv (get_item pt_py i))
              (set! c (get_item DY i))
              (Gfx.fill_rect (- xv 1) (- yv 1) 5 5 250 250 252)
              (Gfx.fill_rect xv yv 3 3 (get_item CR c) (get_item CG c) (get_item CB c))
            }
            #f)
        (set! i (+ i 1))
      })
      ;; 取点标记（画在样本点之上；每行重绘都补一次，位置不变故幂等）
      (draw_probe_mark)
    }))

(define draw_probe_mark
  (lambda ()
    {
      (define cx (Math.round (+ PX (* (+ probe_x R) KSCALE))))
      (define cy (Math.round (+ PY (* (- R probe_y) KSCALE))))
      (if (< cx (+ PX 4)) (set! cx (+ PX 4)) #f)
      (if (> cx (+ PX (- PW 5))) (set! cx (+ PX (- PW 5))) #f)
      (if (< cy (+ PY 4)) (set! cy (+ PY 4)) #f)
      (if (> cy (+ PY (- PH 5))) (set! cy (+ PY (- PH 5))) #f)
      ;; 取点标记：深色四角框 + 深色中心点。用近黑而不是白——采样点是白圈、
      ;; 决策面是浅色，只有深色框在两者之上都抢眼（17x17，不遮挡中心内容）。
      (Gfx.fill_rect (- cx 8) (- cy 8) 7 2 16 18 24)
      (Gfx.fill_rect (- cx 8) (- cy 8) 2 7 16 18 24)
      (Gfx.fill_rect (+ cx 2) (- cy 8) 7 2 16 18 24)
      (Gfx.fill_rect (+ cx 7) (- cy 8) 2 7 16 18 24)
      (Gfx.fill_rect (- cx 8) (+ cy 7) 7 2 16 18 24)
      (Gfx.fill_rect (- cx 8) (+ cy 2) 2 7 16 18 24)
      (Gfx.fill_rect (+ cx 2) (+ cy 7) 7 2 16 18 24)
      (Gfx.fill_rect (+ cx 7) (+ cy 2) 2 7 16 18 24)
      (Gfx.fill_rect (- cx 1) (- cy 1) 2 2 16 18 24)
    }))

;; 权重色标：t ∈ [-1,1] → 蓝负 / 暗零 / 红正（结果写 hc_r/g/b）
(define hc_r 0)
(define hc_g 0)
(define hc_b 0)
(define heat
  (lambda (t)
    (if (> t 0.0)
        {
          (set! hc_r (clamp255 (+ 40.0 (* t 205.0))))
          (set! hc_g (clamp255 (- 44.0 (* t 4.0))))
          (set! hc_b (clamp255 (- 56.0 (* t 16.0))))
        }
        {
          (set! hc_r (clamp255 (+ 40.0 (* t 30.0))))
          (set! hc_g (clamp255 (- 44.0 (* t 106.0))))
          (set! hc_b (clamp255 (- 56.0 (* t 190.0))))
        })))

;; 全体权重的最大绝对值（热图共用一个归一化分母；初值 1e-6 杜绝除零）
(define wmax 0.000001)
(define scan_wmax
  (lambda (lst n)
    {
      (define i 0)
      (define v 0.0)
      (while (< i n) {
        (set! v (Math.abs (get_item lst i)))
        (if (> v wmax) (set! wmax v) #f)
        (set! i (+ i 1))
      })
    }))

;; 一个权重矩阵画成 out×in 的色块阵列
(define draw_mat
  (lambda (lst out in x y cell)
    {
      (define j 0)
      (define i 0)
      (while (< j out) {
        (set! i 0)
        (while (< i in) {
          (heat (/ (get_item lst (+ (* j in) i)) wmax))
          (Gfx.fill_rect (+ x (* i cell)) (+ y (* j cell)) cell cell hc_r hc_g hc_b)
          (set! i (+ i 1))
        })
        (set! j (+ j 1))
      })
    }))

(define draw_heat
  (lambda ()
    {
      (define i 0)
      (define t 0.0)
      (Gfx.fill_rect RX 18 RW 68 20 22 30)
      (Gfx.draw_text "权重热图" RX 20 150 200 255)
      ;; 色标条：蓝(-1) → 暗(0) → 红(+1)
      (set! i 0)
      (while (< i 16) {
        (set! t (- (* (/ (* 1.0 i) 15.0) 2.0) 1.0))
        (heat t)
        (Gfx.fill_rect (+ 250 (* i 4)) 22 4 8 hc_r hc_g hc_b)
        (set! i (+ i 1))
      })
      ;; 三层权重并排（每层右边隔 3px 是它自己的偏置列），共用 ±max|W| 归一化
      (set! wmax 0.000001)
      (scan_wmax w1 12)
      (scan_wmax w2 36)
      (scan_wmax w3 30)
      (draw_mat w1 NH 2 (+ RX 2) 34 5)
      (draw_bias_col b1 NH (+ RX 17) 34 5)
      (draw_mat w2 NH NH (+ RX 36) 34 5)
      (draw_bias_col b2 NH (+ RX 71) 34 5)
      (draw_mat w3 NCLS NH (+ RX 90) 34 5)
      (draw_bias_col b3 NCLS (+ RX 125) 34 5)
      (Gfx.draw_text "W1" (+ RX 2) 66 140 146 162)
      (Gfx.draw_text "b1" (+ RX 18) 66 112 118 134)
      (Gfx.draw_text "W2" (+ RX 36) 66 140 146 162)
      (Gfx.draw_text "b2" (+ RX 72) 66 112 118 134)
      (Gfx.draw_text "W3" (+ RX 90) 66 140 146 162)
      (Gfx.draw_text "b3" (+ RX 126) 66 112 118 134)
    }))

;; 偏置画成一列 5px 色块（与权重同色标、同归一化分母）
(define draw_bias_col
  (lambda (lst n x y cell)
    {
      (define i 0)
      (while (< i n) {
        (heat (/ (get_item lst i) wmax))
        (Gfx.fill_rect x (+ y (* i cell)) cell cell hc_r hc_g hc_b)
        (set! i (+ i 1))
      })
    }))

;; 取点处 5 类概率（顺带把 pred/p5 刷新为「该点」的推理结果）
(define draw_probs
  (lambda ()
    {
      (define j 0)
      (define pv 0.0)
      (define bw 0)
      (define yy 0)
      (define c 0)
      (Gfx.fill_rect RX 88 RW 96 20 22 30)
      (Gfx.draw_text "该点概率" RX 90 150 200 255)
      ;; 坐标单独一行（"x -1.54" + "y -1.37" 共 ~105px，挤在标题行里会顶到右边界）
      (Gfx.draw_text "x" (+ RX 2) 103 150 156 170)
      (draw_fixed probe_x 2 (+ RX 11) 103 150 156 170)
      (Gfx.draw_text "y" (+ RX 53) 103 150 156 170)
      (draw_fixed probe_y 2 (+ RX 60) 103 150 156 170)
      (fw probe_x probe_y)
      (sm)
      (set! j 0)
      (while (< j NCLS) {
        (set! c j)
        (set! yy (+ 118 (* j 12)))
        (set! pv (get_item p5 j))
        (Gfx.fill_rect (+ RX 2) yy 8 8 (get_item CR c) (get_item CG c) (get_item CB c))
        (Gfx.fill_rect (+ RX 12) yy 96 8 34 38 50)
        (set! bw (Math.round (* pv 96.0)))
        (if (< bw 1.0) (set! bw 1.0) #f)
        (Gfx.fill_rect (+ RX 12) yy bw 8 (get_item CR c) (get_item CG c) (get_item CB c))
        (set! c (Math.round (* pv 100.0)))
        (draw_int c 1 (- (+ RX 134) (* (ndigits c) DW)) (- yy 2) 200 206 216)
        (Gfx.draw_text "%" (+ RX 134) (- yy 2) 200 206 216)
        (set! j (+ j 1))
      })
    }))

;; 损失曲线（底部整幅）：x 轴一轮 epoch，y 轴固定 [0, 全程最大值]
(define draw_loss
  (lambda ()
    {
      (define i 0)
      (define n 0)
      (define v 0.0)
      (define vmax 1.0)
      (define span 1.0)
      (define cxp 0)
      (define cyp 0)
      (define ai 0)
      (define px0 0)
      (define py0 0)
      (Gfx.fill_rect LC_X LC_Y LC_W LC_H 16 18 26)
      (Gfx.draw_rect LC_X LC_Y LC_W LC_H 52 56 68)
      (Gfx.draw_text "损失曲线" 10 186 150 200 255)
      (Gfx.draw_text "loss " 64 186 255 150 110)
      (draw_fixed cur_loss 3 99 186 255 150 110)
      (Gfx.draw_text "acc " 140 186 130 230 160)
      (set! ai (Math.round (* cur_acc 100.0)))
      (draw_int ai 1 (- 185 (* (ndigits ai) DW)) 186 130 230 160)   ; 右对齐到 "%"
      (Gfx.draw_text "%" 185 186 130 230 160)
      (Gfx.draw_text "ep " 196 186 160 170 190)
      (draw_int epoch 1 217 186 160 170 190)
      (Gfx.draw_text "lr " 244 186 120 128 148)
      (draw_fixed LR 2 265 186 120 128 148)
      (Gfx.draw_line CH_X (+ CH_Y CH_H) (+ CH_X CH_W) (+ CH_Y CH_H) 44 48 62)
      ;; 纵轴 [0, 全程最大值]（只增不减 → 轴不抖，收敛段贴近底部，与常见 loss 曲线同观感）
      (set! n (if (< loss_cnt LOSS_N) loss_cnt LOSS_N))
      (if (> n 1)
          {
            (set! vmax loss_max)
            (set! span (if (< loss_max 0.000001) 1.0 loss_max))
            (set! i 0)
            (set! px0 0)
            (set! py0 (+ CH_Y CH_H))
            (while (< i n) {
              (set! v (get_item loss_ring (mod (+ (+ (- loss_idx n) i) 240) LOSS_N)))
              (set! cxp (Math.round (+ CH_X (* (/ (* 1.0 i) 119.0) (- CH_W 1)))))
              (set! cyp (Math.round (+ CH_Y (* (/ (- vmax v) span) (- CH_H 1)))))
              (if (< cxp 0) (set! cxp 0) #f)
              (if (< cyp 0) (set! cyp 0) #f)
              (if (> i 0) (Gfx.draw_line px0 py0 cxp cyp 110 170 255) #f)
              (set! px0 cxp)
              (set! py0 cyp)
              (set! i (+ i 1))
            })
          }
          #f)
    }))

(define draw_all
  (lambda ()
    {
      (Gfx.fill_rect 0 0 SW SH 12 14 20)
      (draw_map_rows 0 PG)
      (Gfx.draw_rect PX PY PW PH 60 66 78)
      (draw_header)
      (draw_heat)
      (draw_probs)
      (draw_loss)
    }))

;; ---- 输入 ---------------------------------------------------------------

;; 1=暂停/继续 2=重置 3=图内取点 0=都不是
(define hit_ui
  (lambda (x y)
    (cond ((in_rect x y B1X BTN_Y BTN_W BTN_H) 1)
          ((in_rect x y B2X BTN_Y BTN_W BTN_H) 2)
          ((in_rect x y PX PY PW PH) 3)
          (else 0))))

;; 取点标记移动后，只需重画被它覆盖的那几行（旧位置 ∪ 新位置，各外扩 8px）——
;; 全屏重绘（draw_all）在拖拽时会变成每拍 70k 条指令的额外开销，没必要
(define redraw_probe_band
  (lambda (oy ny)
    {
      (define lo (if (< oy ny) oy ny))
      (define hi (if (> oy ny) oy ny))
      (define r 0)
      (define r1 0)
      (set! r (Math.floor (/ (* 1.0 (- (- lo 8) PY)) PS)))
      (set! r1 (+ 1 (Math.floor (/ (* 1.0 (- (+ hi 8) PY)) PS))))
      (if (< r 0) (set! r 0) #f)
      (if (> r1 PG) (set! r1 PG) #f)
      (while (< r r1) {
        (draw_map_rows r (+ r 1))
        (set! r (+ r 1))
      })
    }))

;; 把触点（绘图区像素）换算成数据坐标并落标记；像素移动不足 2px 时不动
(define set_probe
  (lambda (x y)
    {
      (set! probe_x (- (/ (* 1.0 (- x PX)) KSCALE) R))
      (set! probe_y (- R (/ (* 1.0 (- y PY)) KSCALE)))
      (if (< probe_x (- 0.0 R)) (set! probe_x (- 0.0 R)) #f)
      (if (> probe_x R) (set! probe_x R) #f)
      (if (< probe_y (- 0.0 R)) (set! probe_y (- 0.0 R)) #f)
      (if (> probe_y R) (set! probe_y R) #f)
      (redraw_probe_band probe_py y)
      (set! probe_px x)
      (set! probe_py y)
      (draw_probs)
    }))

(define do_action
  (lambda (a)
    (cond ((== a 1) { (set! running (if (== running 1) 0 1)) (draw_header) })
          ((== a 2) { (reset_all) (draw_all) })
          (else #f))))

;; ---- 复位：重新初始化权重、清空统计与决策图 ------------------------------

(define shuffle_perm
  (lambda ()
    {
      (define i 255)
      (define j 0)
      (define t 0)
      (while (> i 0) {
        (set! j (Math.floor (* (rnd) (+ i 1))))
        (set! t (get_item perm i))
        (set_item! perm i (get_item perm j))
        (set_item! perm j t)
        (set! i (- i 1))
      })
    }))

(define reset_all
  (lambda ()
    {
      (define i 0)
      (reseed)
      (rand_fill w1 12 WAMP)
      (rand_fill w2 36 WAMP)
      (rand_fill w3 30 WAMP)
      (zero_fill b1 NH)
      (zero_fill b2 NH)
      (zero_fill b3 NCLS)
      (zero_fill loss_ring LOSS_N)
      (set! loss_idx 0)
      (set! loss_cnt 0)
      (set! loss_max 1.0)
      (set! cur_loss 0.0)
      (set! cur_acc 0.0)
      (set! epoch 0)
      (set! tick_in_epoch 0)
      (set! ep_loss 0.0)
      (set! ep_ok 0)
      (set! perm_pos 0)
      (set! mrow 0)
      (set! probe_x 0.0)
      (set! probe_y 0.0)
      (set! probe_px (Math.round (+ PX (* R KSCALE))))
      (set! probe_py (Math.round (+ PY (* R KSCALE))))
      (set! armed 0)
      (set! running 1)
      (set! i 0)
      (while (< i 1024) {
        (set_item! map_lab i 0)
        (set! i (+ i 1))
      })
      (set! i 0)
      (while (< i DN) {
        (set_item! perm i i)
        (set! i (+ i 1))
      })
      (shuffle_perm)
    }))

;; ---- 每拍 ---------------------------------------------------------------
;; 顺序（预算约 25 万条 VM 指令，在单帧 31.25 万之内）：
;;   1 个批量（16 样本，前向+反向+一次更新） → 决策图 1 行（32 格）
;;   → 局部重绘（该行的格/样本点/取点标记）→ 全量评估 16 个样本
;;   → 权重热图 → 取点概率
;; 16 拍 = 1 轮 epoch = 256 个样本，正好把决策图扫过两遍（32 行/16 拍）。

(define step_all
  (lambda ()
    {
      (train_batch)
      (map_rows mrow (+ mrow ROWS_PER_TICK))
      (draw_map_rows mrow (+ mrow ROWS_PER_TICK))
      (eval_samples (* tick_in_epoch EVAL_PER_TICK)
                    (+ (* tick_in_epoch EVAL_PER_TICK) EVAL_PER_TICK))
      (set! mrow (+ mrow ROWS_PER_TICK))
      (if (>= mrow PG) (set! mrow 0) #f)
      (draw_heat)
      (draw_probs)
      (set! tick_in_epoch (+ tick_in_epoch 1))
      (if (>= tick_in_epoch TICKS_PER_EPOCH)
          {
            ;; 一轮结束：把本轮累计的 loss/准确率写进曲线，刷新文字
            (set! cur_loss (/ ep_loss (* 1.0 DN)))
            (set! cur_acc (/ (* 1.0 ep_ok) (* 1.0 DN)))
            (set_item! loss_ring loss_idx cur_loss)
            (if (> cur_loss loss_max) (set! loss_max cur_loss) #f)
            (set! loss_idx (+ loss_idx 1))
            (if (>= loss_idx LOSS_N) (set! loss_idx 0) #f)
            (if (< loss_cnt LOSS_N) (set! loss_cnt (+ loss_cnt 1)) #f)
            (set! ep_loss 0.0)
            (set! ep_ok 0)
            (set! tick_in_epoch 0)
            (set! perm_pos 0)
            (set! epoch (+ epoch 1))
            (shuffle_perm)
            (draw_header)
            (draw_loss)
          }
          #f)
    }))

(define tick
  (lambda ()
    {
      (EX.check)
      (set! tp (if (Touch.take_press) 1 0))
      (set! tr (if (Touch.take_release) 1 0))
      (set! tx (Touch.x))
      (set! ty (Touch.y))
      ;; 按下沿：记录命中的控件；图内则直接取点
      (if (== tp 1)
          {
            (set! armed (hit_ui tx ty))
            (if (== armed 3) (set_probe tx ty) #f)
          }
          #f)
      ;; 按住拖动：取点跟着手指走（像素移动 ≥2 才重算，避免每拍都做一次前向 + 重绘）
      (if (== armed 3)
          (if (Touch.is_pressed)
              (if (or (> (Math.abs (- tx probe_px)) 2.0)
                      (> (Math.abs (- ty probe_py)) 2.0))
                  (set_probe tx ty)
                  #f)
              #f)
          #f)
      ;; 松开沿：同一控件内松手才生效（避免误触）
      (if (== tr 1)
          {
            (if (> armed 0)
                (if (== (hit_ui tx ty) armed) (do_action armed) #f)
                #f)
            (set! armed 0)
          }
          #f)
      ;; 训练（暂停时只响应输入，不再重绘）
      (if (== running 1) (step_all) #f)
      (Gfx.refresh)
    }))

;; ---- 启动 ---------------------------------------------------------------

;; 样本像素坐标（钳进绘图区，圆心离边框至少 3px）
(define setup_pts
  (lambda ()
    {
      (define i 0)
      (define v 0.0)
      (while (< i DN) {
        (set! v (Math.round (+ PX (* (+ (get_item DX (* i 2)) R) KSCALE))))
        (if (< v (+ PX 3)) (set! v (+ PX 3)) #f)
        (if (> v (+ PX (- PW 4))) (set! v (+ PX (- PW 4))) #f)
        (set_item! pt_px i v)
        (set! v (Math.round (+ PY (* (- R (get_item DX (+ (* i 2) 1))) KSCALE))))
        (if (< v (+ PY 3)) (set! v (+ PY 3)) #f)
        (if (> v (+ PY (- PH 4))) (set! v (+ PY (- PH 4))) #f)
        (set_item! pt_py i v)
        (set! i (+ i 1))
      })
    }))

(setup_pts)
(reset_all)
(EX.install_with draw_all)
(draw_all)
(Gfx.refresh)
(set! _timer (System.set_interval TICK_MS tick))