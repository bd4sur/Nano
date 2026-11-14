;; ============================================================================
;; 遗传算法求解旅行商问题（TSP）—— Nano「灵机应用程序」Scheme App
;;
;; 参考实现：infer/ui_tsp.h（C 版 STATE_GENETIC_TSP）。本版把同一套算法与同一幅
;; 画面（城市散点 + 最优回路 + 距离下降曲线 + 顶栏读数）用 Scheme 重写，做成
;; **插件式 App**：放进业务资源目录 app/ 即被发现，固件零改动。
;;
;; 画面布局（与 C 版常量逐一对齐，320×240）：
;;   · 顶栏        y 0..13   深灰底 (39,39,39) + 12px 二值点阵白字
;;   · 地图        y 14..209 经纬度线性投影（lon 85..130、lat 18..48，左右各留 6px）
;;   · 曲线        y 216..237 深灰底 (20,20,20) + 绿色折线（最近 300 代最优距离）
;;   城市点 红实心圆 r=2；城市名 3×5 超小字体（app/lib/mini_font.scm）；
;;   回路 抗锯齿蓝线（用浮点屏幕坐标，避免整数截断产生的纯水平/垂直段）。
;;
;; 超小字体为什么要单独成库：C 版用 graphics.c 的 gfx_draw_textline_mini（3×5
;; 位图字模）画 34 个城市三字码与曲线读数。Scheme 侧把同一套字模封装成
;; app/lib/mini_font.scm（(import MF ...) 即可复用，任何 App 都能用它画小标注），
;; 而不是再往 Gfx 本地库加一个 native 过程——见该模块头部的说明。
;;
;; ---------------------------------------------------------------------------
;; 性能设计（2026-11 重构；wasm 目标实测：单代 6ms → 1.1ms，帧率 20 → 60fps）
;;
;; 本解释器里三种操作最贵：**本地库过程调用 ≈1µs**、**列表 get_item/set_item!
;; ≈0.5µs**、普通 VM 指令 ≈0.15µs。上一版每代 6ms、整屏绘制 5.8ms，一拍 12ms
;; 只能每 3 帧跑一次（画面 20 次/秒，看着卡）。对策：
;;
;;   1) **适应度增量维护**（省掉"每代重新评估全部个体"）：
;;      每个个体出生时就带着自己的回路长度（`lenv`），下一代：
;;        · 无交叉的子代 = 父代**引用共享**（`set_item!` 一个槽位即完成"拷贝"，
;;          因为个体列表在出生后永不原地修改）；
;;        · 只有交叉子代要算一遍 34 条边（`path-len`，约 60µs）；
;;        · 逆转变异只换掉两条边界边，长度按 4 次查表求差（`invert-mutate`）。
;;      于是每代只有 ~P-CROSS×POP 个个体需要评估，而不是全部 POP 个。
;;   2) **轮盘赌用二分查找**：累积分布每代算一遍，选择时 7 次比较（原来线性扫描
;;      平均 POP/2 次）。
;;   3) **整表拷贝靠 cdr/cons**：本解释器的 `cdr`/`cons` 是 C 层整表复制（3 条指令），
;;      比逐元素 get_item/set_item!（68 条指令）快一个数量级。变异走 copy-on-write。
;;   4) **超小字体预展开像素表**（见 mini_font.scm）：34 个标签 3.7ms → ≈0.4ms。
;;   5) **曲线抽稀**：历史窗口满时每 `k` 个点取一个（最多画 ~100 段 AA 线），
;;      形状不变，绘制时间砍半。
;;   6) **一拍多代 + 每帧推屏**：定时器间隔 1ms（= 每轮事件循环一次 = 每帧一次），
;;      每拍跑若干代后整屏重绘并推屏。这样"算"与"画"同拍、帧率锁在 60fps；
;;      每拍几代由**实测单拍工作量自适应**（tick 里的 gens-n）：桌面浏览器实测
;;      4 代/拍（≈7ms 工作量）→ 180~240 代/秒（旧版 20 代/秒）；慢机器自动降档，
;;      保证一拍照样在一帧内画完。
;;
;; ---------------------------------------------------------------------------
;; 遗传算法（与 C 版同一套骨架：轮盘赌 + 精英保留 + 顺序交叉；差异见下）：
;;   · 个体 = 34 个城市的排列（0..33）；种群 POP 个个体（C 版 500，Scheme 侧取
;;     100/60，见 POP 的注释——解释器比 C 慢，靠减少种群换帧率）；
;;   · 适应度 = N / 回路长度（越短越大）；轮盘赌选择；
;;   · 精英保留：每代最好的个体原样进下一代 0 号位；
;;   · 顺序交叉（OX，概率 0.2）、**逆转变异**（概率 0.05，见下）；
;;   · 每拍画一帧（一拍若干代，代数自适应）。
;;
;; 与 C 版的**有意差异**：
;;   0) 变异算子换成"逆转变异"（随机选一段、整段反转），而不是 C 版的"交换两个
;;      城市"。理由：交换变异把两条边拆掉再接两条随机边，几乎从不缩短回路长度
;;      （TSP 里它基本是纯噪声）；逆转变异恰好落在 2-opt 邻域里（只改两条边），
;;      是 TSP 遗传算法最常用的变异算子。同样 100 个体的纯 GA，实测同样代数下
;;      最优距离：400 代 183 vs 213、550 代 169 vs 197（2-opt 参考最优 154.6）。
;;      变异概率同时按"每代变异个体数"折算（500×0.01 ≈ 5 → 100×0.05 = 5）。
;;   1) 距离函数：C 版的"偏好约束"在两个优先邻接对（乌鲁木齐-拉萨、乌鲁木齐-西宁）
;;      上**只减 0.1 而不加实际距离**（if/else 分支把距离项整个跳过了，是个笔误），
;;      于是这两条边的代价被抹成负数、显示出来的"距离"不再是真实长度。本版按原意
;;      实现：距离照常累加，偏好对**额外减 0.1**，乌鲁木齐-哈尔滨**额外罚 100000**
;;      （惩罚只影响选择，顶栏显示的仍是真实路径长度）；
;;   2) 绘制的个体：C 版在 evaluate → next_generation 之后用**旧代的 best_id** 去索引
;;      **新一代**的种群（画出来的不是当代最优）；本版把每代最优的索引固定为 0
;;      （精英保留恰好把它放在 0 号位），画的一定是当代最优回路。
;;
;; 交互：
;;   · 演化自动进行（每帧一拍、一拍照帧预算跑若干代）；每拍整屏重绘 + 推屏；
;;   · **点地图**（或实体键盘 Enter）重新撒种、从第 0 代重新演化 —— 对应 C 版的
;;     D/Enter 键；画面左下角有 "TAP=NEW" 提示；
;;   · 右上角「退出」（app/lib/exit.scm）+ 硬 A/ESC → 宿主全局退出确认框。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数（多元乘法会被编译器吃掉左值）；
;;   无 let/list/map/apply；cond 每子句单个表达式；while 循环体不是独立词法区域
;;   （循环变量一律在 lambda 体顶层 define）；System.set_interval 产值须用 set! 消费；
;;   native 绘图过程是语句型（弹参后压 1 个 #undefined，语句位置由编译器弹出）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native App)
(native Diag)
(import EX "app/lib/exit.scm")        ; 硬 A/ESC / 右上角按钮 → 全局退出确认框
(import MF "app/lib/mini_font.scm")   ; 3×5 超小字体（城市三字码、曲线读数）

;; ---- 城市数据：34 个中国主要城市（经度 纬度，与 C 版 tsp_map 逐项一致） -----

(define N 34)

(define city-ll
  '(116.46 39.92   117.20 39.13   121.48 31.22   106.54 29.59
    91.11  29.97   87.68  43.77   106.27 38.47   111.65 40.82
    108.33 22.84   126.63 45.75   125.35 43.88   123.38 41.80
    114.48 38.03   112.53 37.87   101.74 36.56   117.00 36.65
    113.60 34.76   118.78 32.04   117.27 31.86   120.19 30.26
    119.30 26.08   115.89 28.68   113.00 28.21   114.31 30.52
    113.23 23.16   121.50 25.05   110.35 20.02   103.73 36.03
    108.95 34.27   104.06 30.67   106.71 26.57   102.73 25.04
    114.10 22.20   113.33 22.13))

(define city-name
  '("PEK" "TSN" "SHA" "CKG" "LXA" "URC" "INC" "HET" "NNG"
    "HRB" "CGQ" "SHE" "SJW" "TYN" "XNN" "TNA" "CGO" "NKG"
    "HFE" "HGH" "FOC" "KHN" "CSX" "WUH" "CAN" "TPE" "HAK"
    "LHW" "XIY" "TFU" "KWE" "KMG" "HKG" "MFM"))

;; ---- 算法参数 --------------------------------------------------------------
;; 种群规模按平台取两档 —— 解释器比编译代码慢约一个数量级，一"代"的成本 ≈
;; POP ×（基因拷贝 + 交叉子代评估），wasm 实测一代约 1.6ms，故一拍可跑几代。
;; 交叉概率与 C 版一致（0.2）。变异概率则按"每代实际变异的个体数"折算：C 版是
;; 500 × 0.01 = 5 个/代，本版 100 × 0.05 = 5 个/代。
;; 每帧演化的代数**自适应**（gens-n，见 tick）：宿主每轮事件循环只触发一个
;; 定时器回调 = 每帧一拍，故代际吞吐 = 帧率 × 每拍代数。wasm（桌面浏览器）实测
;; 一代约 1.6ms、整屏绘制约 0.9ms：4 代/拍 ≈ 7ms（留一半帧预算），≈240 代/秒
;; （旧版 20 代/秒且每 3 帧才更新一次画面）。慢机器/ESP32 会自动降到 1~2 代。
(define P-CROSS 0.2)
(define P-MUT   0.05)
(define HIST    300)   ; 距离曲线保留的代数（C 版 TSP_HISTORY_SIZE）
(define POP (if (String.equals (App.platform) "m5esp") 60 100))

;; 每拍代数的自适应参数：单拍工作量（算 + 画 + 推屏）超过 HI 就减一代，
;; 低于 LO 就加一代（上限 GENS-MAX）。两个阈值之间留死区，避免来回抖。
(define GENS-MAX 4)
(define GENS-LO 7.0)    ; ms：滑动平均低于它（连续 20 拍）才升一代
(define GENS-KICK 16.0) ; ms：单拍超过它立刻降一代（帧预算 16.7ms）

;; ---- 画面布局（与 C 版常量一致） -------------------------------------------

(define SW (Gfx.width))
(define HDR_H 14)
(define MAP_TOP 14.0)
(define MAP_BOTTOM 210.0)
(define CURVE_TOP 216)
(define CURVE_H 22)

;; ---- 状态 ------------------------------------------------------------------
;; 种群是"个体列表的列表"：个体 = 长度 N 的基因列表。要点：
;;   · 个体列表一旦出生就**不再原地修改**（变异走 copy-on-write），因此种群里的
;;     槽位可以直接存父代个体的**引用**——"复制一个个体"就是 set_item! 一个槽位
;;     （3 条指令），而不是逐元素拷贝（68 条指令）；
;;   · 每个个体配一个已知的回路长度，选择/排名不再需要重新评估（增量适应度）。
;; 空表一律 (cdr '(0)) 现造：'() 字面量是模块级静态槽，push 进去会污染后续复用。

(define pop (cdr '(0)))        ; 当代种群（POP 个个体）
(define nextpop (cdr '(0)))    ; 下一代种群（POP 个个体）
(define lenv (cdr '(0)))       ; 各代个体的回路长度（POP）
(define nextlen (cdr '(0)))
(define fitv (cdr '(0)))       ; 各代个体的适应度 = N/长度（POP）
(define nextfit (cdr '(0)))
(define cum (cdr '(0)))        ; 适应度的累积分布（POP，未归一化）
(define used (cdr '(0)))       ; OX 占位标记（N）：存"本轮编号"，免每轮复位
(define history (cdr '(0)))    ; 历史最优距离（HIST，环形）
(define cost (cdr '(0)))       ; 代价矩阵（N×N 扁平，含偏好加成）
(define tmpl (cdr '(0)))       ; N 个 -1 的模板（交叉子代"待填"的种子，廉价复制）
(define seq (cdr '(0)))        ; (0 1 … N-1) 模板（初始种群的种子）
(define cx (cdr '(0)))         ; 城市屏幕坐标（浮点，N）
(define cy (cdr '(0)))

(define gen 0)                 ; 代数
(define hist-n 0)              ; 已记录的历史点数（可超过 HIST）
(define total-fit 1.0)         ; 当代适应度总和
(define best-id 0)             ; 当代最优个体的槽位（精英保留 → 恒为 0）
(define best-len 0.0)          ; 当代最优回路的长度
(define xid 0)                 ; 交叉轮次编号（OX 的 used 标记用）
(define gens-n 2)              ; 当前每拍演化代数（自适应）
(define work-avg 5.0)          ; 单拍工作量（毫秒）的指数滑动平均
(define adj-cnt 0)             ; 距离下次调节已过的拍数

;; ---- 小工具 ----------------------------------------------------------------

;; 从 [0,n) 取一个随机整数（浮点形式；本解释器的列表索引接受整数型浮点）
(define rnd-int
  (lambda (n) (Math.floor (* (Math.random) n))))

;; 新建长度为 n、元素全为 v 的列表
(define make-fill
  (lambda (n v) {
    (define l (cdr '(0)))
    (define i 0)
    (while (< i n) { (push l v) (set! i (+ i 1)) })
    l
  }))

;; (0 1 … n-1) 序列（初始种群的种子）
(define make-seq
  (lambda (n) {
    (define l (cdr '(0)))
    (define i 0)
    (while (< i n) { (push l i) (set! i (+ i 1)) })
    l
  }))

;; 基因列表整表复制：本解释器的 cdr/cons 是 C 层复制，3 条指令搞定；
;; 逐元素 get_item/set_item! 要 68 条（实测差 ~10 倍）。非空列表专用。
(define copy-list
  (lambda (lst) (cons (get_item lst 0) (cdr lst))))

;; 一位小数，等价 printf "%.1f"。
;; **不能**直接 String.atom_to_string 一个浮点数：wasm32 的浮点尾数只有约 19 位，
;; 267.3 会打成 "267.29999"（见 doc/AGENTS.md 的踩坑记录）。故先放大十倍取整，
;; 再拆成整数段与小数段分别转字符串。
(define fmt1
  (lambda (v) {
    (define q (Math.round (* 10.0 v)))
    (define ip (Math.floor (/ q 10.0)))
    (define fp (- q (* ip 10.0)))
    (String.concat (String.atom_to_string ip)
      (String.concat "." (String.atom_to_string fp)))
  }))

;; ---- 初始化：屏幕投影 / 代价矩阵 -------------------------------------------

;; 经纬度 → 屏幕坐标（与 C 版 tsp_lonlat_to_screen_f 同一投影）
(define init-screen
  (lambda () {
    (define i 0)
    (define lon 0.0)
    (define lat 0.0)
    (define px 0.0)
    (define py 0.0)
    (set! cx (cdr '(0)))
    (set! cy (cdr '(0)))
    (while (< i N) {
      (set! lon (get_item city-ll (* i 2)))
      (set! lat (get_item city-ll (+ (* i 2) 1)))
      (set! px (+ 6.0 (/ (* (- lon 85.0) (- SW 12)) 45.0)))
      (set! py (+ MAP_TOP (/ (* (- 48.0 lat) (- MAP_BOTTOM MAP_TOP)) 30.0)))
      (if (< px 0.0) (set! px 0.0) #f)
      (if (> px (- SW 7.0)) (set! px (- SW 7.0)) #f)
      (if (< py MAP_TOP) (set! py MAP_TOP) #f)
      (if (> py (- MAP_BOTTOM 1.0)) (set! py (- MAP_BOTTOM 1.0)) #f)
      (push cx px)
      (push cy py)
      (set! i (+ i 1))
    })
  }))

;; 代价矩阵：cost[i*N+j] = 城市 i→j 的代价（对称）。
;; 基准是经纬度平面上的欧氏距离（与 C 版 tsp_distance 相同，度数当平面坐标用）；
;; 再加 C 版"偏好约束"的**本意**：乌鲁木齐(5)-拉萨(4)、乌鲁木齐(5)-西宁(14)
;; 各减 0.1；乌鲁木齐(5)-哈尔滨(9) 罚 100000（C 版把这两个偏好对的真实距离整个
;; 跳过了，见文件头的"有意差异"第 1 条）。
(define init-cost
  (lambda () {
    (define i 0)
    (define j 0)
    (define dx 0.0)
    (define dy 0.0)
    (define d 0.0)
    (define m (cdr '(0)))
    (while (< i N) {
      (set! j 0)
      (while (< j N) {
        (set! dx (- (get_item city-ll (* i 2)) (get_item city-ll (* j 2))))
        (set! dy (- (get_item city-ll (+ (* i 2) 1))
                    (get_item city-ll (+ (* j 2) 1))))
        (set! d (Math.sqrt (+ (* dx dx) (* dy dy))))
        (if (and (== i 5) (== j 4)) (set! d (- d 0.1)) #f)
        (if (and (== i 4) (== j 5)) (set! d (- d 0.1)) #f)
        (if (and (== i 5) (== j 14)) (set! d (- d 0.1)) #f)
        (if (and (== i 14) (== j 5)) (set! d (- d 0.1)) #f)
        (if (and (== i 5) (== j 9)) (set! d (+ d 100000.0)) #f)
        (if (and (== i 9) (== j 5)) (set! d (+ d 100000.0)) #f)
        (push m d)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    m
  }))

;; ---- 回路长度：一条 34 边的扫描（含回到起点）。只在交叉子代出生时调用 ------
(define path-len
  (lambda (ind) {
    (define cm cost)
    (define j 1)
    (define a (get_item ind 0))
    (define b 0)
    (define ln 0.0)
    (while (< j N) {
      (set! b (get_item ind j))
      (set! ln (+ ln (get_item cm (+ (* a N) b))))
      (set! a b)
      (set! j (+ j 1))
    })
    (set! b (get_item ind 0))
    (+ ln (get_item cm (+ (* a N) b)))
  }))

;; ---- 初始化：随机种群（同时算出每个个体的长度与适应度） --------------------
(define build-pop
  (lambda () {
    (define p (cdr '(0)))
    (define l (cdr '(0)))
    (define f (cdr '(0)))
    (define i 0)
    (define j 0)
    (define k 0)
    (define t 0)
    (define ind (cdr '(0)))
    (define ln 0.0)
    (while (< i POP) {
      (set! ind (copy-list seq))
      (set! j (- N 1))
      (while (> j 0) {
        (set! k (rnd-int (+ j 1)))
        (set! t (get_item ind j))
        (set_item! ind j (get_item ind k))
        (set_item! ind k t)
        (set! j (- j 1))
      })
      (set! ln (path-len ind))
      (push p ind)
      (push l ln)
      (push f (/ N ln))
      (set! i (+ i 1))
    })
    (set! pop p)
    (set! lenv l)
    (set! fitv f)
  }))

;; ---- 每代一次：找最优个体 + 累积分布（增量适应度，不再重新评估） -----------
(define scan
  (lambda () {
    (define fv fitv)
    (define i 0)
    (define c 0.0)
    (define bestf 0.0)
    (define besti 0)
    (define v 0.0)
    (set! i 0)
    (while (< i POP) {
      (set! v (get_item fv i))
      (set! c (+ c v))
      (set_item! cum i c)
      (if (> v bestf) {
            (set! bestf v)
            (set! besti i)
          }
          #f)
      (set! i (+ i 1))
    })
    (set! total-fit c)
    (set! best-id besti)
    (set! best-len (get_item lenv besti))
  }))

;; ---- 轮盘赌选择（在累积分布上二分，返回被选中的个体下标） ------------------
(define roulette
  (lambda () {
    (define ptr (* (Math.random) total-fit))
    (define lo 0)
    (define hi (- POP 1))
    (define mid 0.0)
    (while (< lo hi) {
      (set! mid (Math.floor (/ (+ lo hi) 2)))
      (if (<= (get_item cum mid) ptr)
          (set! lo (+ mid 1))
          (set! hi mid))
    })
    lo
  }))

;; ---- 顺序交叉（OX）：返回新个体 = a 的一段连续基因 + b 的其余基因按原顺序 ----
;; used 里存"本轮编号"而不是 1/0，于是每轮无需复位（省 N 次 set_item!）。
(define make-crossover
  (lambda (ia ib) {
    (define pa (get_item pop ia))
    (define pb (get_item pop ib))
    (define ch (copy-list tmpl))
    (define q1 (rnd-int N))
    (define q2 (rnd-int N))
    (define lo (if (< q1 q2) q1 q2))
    (define hi (if (< q1 q2) q2 q1))
    (define i 0)
    (define bi 0)
    (define gv 0)
    (set! xid (+ xid 1))
    ;; 1) 保留 a 的 [lo,hi] 段
    (set! i lo)
    (while (<= i hi) {
      (set! gv (get_item pa i))
      (set_item! ch i gv)
      (set_item! used gv xid)
      (set! i (+ i 1))
    })
    ;; 2) 其余位置按 b 的顺序填入未被占用的城市
    (set! bi 0)
    (set! i 0)
    (while (< i N) {
      (if (== -1 (get_item ch i))
          {
            (while (== xid (get_item used (get_item pb bi)))
              (set! bi (+ bi 1)))
            (set! gv (get_item pb bi))
            (set_item! ch i gv)
            (set_item! used gv xid)
            (set! bi (+ bi 1))
          }
          #f)
      (set! i (+ i 1))
    })
    ch
  }))

;; ---- 逆转变异（原地反转一段，返回新的回路长度） ----------------------------
;; 段内边集合反转后不变，只有**两条边界边**被替换：(prev,a)(b,nxt) → (prev,b)(a,nxt)，
;; 故长度只需 4 次查表求差，不必重扫 34 条边。
(define invert-mutate
  (lambda (ch ln) {
    (define cm cost)
    (define q1 (rnd-int N))
    (define q2 (rnd-int N))
    (define lo (if (< q1 q2) q1 q2))
    (define hi (if (< q1 q2) q2 q1))
    (define prev 0)
    (define nxt 0)
    (define a 0)
    (define b 0)
    (define t 0)
    (define d 0.0)
    (cond ((>= lo hi) ln)                       ; 空段 / 单点：不动
          ((and (== lo 0) (== hi (- N 1))) ln)  ; 整条回路反转：边集合不变
          (else {
            (set! a (get_item ch lo))
            (set! b (get_item ch hi))
            (set! prev (if (== lo 0) (get_item ch (- N 1)) (get_item ch (- lo 1))))
            (set! nxt (if (== hi (- N 1)) (get_item ch 0) (get_item ch (+ hi 1))))
            (set! d (- (+ (get_item cm (+ (* prev N) b))
                          (get_item cm (+ (* a N) nxt)))
                       (+ (get_item cm (+ (* prev N) a))
                          (get_item cm (+ (* b N) nxt)))))
            (while (< lo hi) {
              (set! t (get_item ch lo))
              (set_item! ch lo (get_item ch hi))
              (set_item! ch hi t)
              (set! lo (+ lo 1))
              (set! hi (- hi 1))
            })
            (+ ln d)
          }))
  }))

;; ---- 记录当代最优距离（环形缓冲） ------------------------------------------
(define record
  (lambda () {
    (set_item! history (mod hist-n HIST) best-len)
    (set! hist-n (+ hist-n 1))
  }))

;; ---- 演化一代：选代/交叉/变异/精英 → 翻页 -----------------------------------
(define evolve
  (lambda () {
    (define nb 1)
    (define p1 0)
    (define p2 0)
    (define ch (cdr '(0)))
    (define ln 0.0)
    (define nx nextpop)
    (define nl nextlen)
    (define nf nextfit)
    (define t (cdr '(0)))
    (scan)
    (record)
    ;; 精英保留：当代最优原样进下一代 0 号位（存的是**引用**，零拷贝）
    (set_item! nx 0 (get_item pop best-id))
    (set_item! nl 0 (get_item lenv best-id))
    (set_item! nf 0 (get_item fitv best-id))
    (while (< nb POP) {
      (set! p1 (roulette))
      (if (<= (Math.random) P-CROSS)
          {
            (set! p2 (roulette))
            (set! ch (make-crossover p1 p2))
            (set! ln (path-len ch))
          }
          {
            ;; 无交叉 = 父代个体的引用（个体出生后不再原地改，故可安全共享）
            (set! ch (get_item pop p1))
            (set! ln (get_item lenv p1))
          })
      (if (<= (Math.random) P-MUT)
          {
            ;; 变异走 copy-on-write：先整表复制，再原地反转并增量更新长度
            (set! ch (copy-list ch))
            (set! ln (invert-mutate ch ln))
          }
          #f)
      (set_item! nx nb ch)
      (set_item! nl nb ln)
      (set_item! nf nb (/ N ln))
      (set! nb (+ nb 1))
    })
    (set! t pop)
    (set! pop nx)
    (set! nextpop t)
    (set! t lenv)
    (set! lenv nl)
    (set! nextlen t)
    (set! t fitv)
    (set! fitv nf)
    (set! nextfit t)
    (set! best-id 0)
    (set! gen (+ gen 1))
  }))

;; ---- 重新撒种：从第 0 代重开（C 版的 D/Enter 键） ---------------------------
(define reset
  (lambda () {
    (set! gen 0)
    (set! hist-n 0)
    (set! best-len 0.0)
    (set! best-id 0)
    (build-pop)
    (set! nextpop (make-fill POP 0))
    (set! nextlen (make-fill POP 0.0))
    (set! nextfit (make-fill POP 0.0))
  }))

;; ============================================================================
;; 绘制
;; ============================================================================

;; 顶栏：深灰底 + 12px 二值点阵白字（与 C 版 gfx_draw_textline 同一字模；
;; Gfx.draw_text 默认是 1 = 12px 抗锯齿，必须显式切到 0 = 二值点阵）
(define draw-header
  (lambda () {
    (Gfx.fill_rect 0 0 SW HDR_H 39 39 39)
    (Gfx.set_font 0)
    (Gfx.draw_text
      (String.concat "TSP | 距离:"
        (String.concat (fmt1 best-len)
          (String.concat " | 代数:" (String.atom_to_string gen))))
      2 1 255 255 255)
  }))

;; 地图：城市点（红）→ 城市名（超小字体）→ 最优回路（抗锯齿蓝线）
;; 绘制顺序与 C 版一致：路径画在标签之上。
(define draw-map
  (lambda () {
    (define ind (get_item pop best-id))
    (define i 0)
    (define a 0)
    (define c 0)
    (while (< i N) {
      (Gfx.fill_circle (get_item cx i) (get_item cy i) 2 255 0 0)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i N) {
      ;; 半透明白（C 版 mode=127）→ 黑底上等价于中灰；用超小字体库绘制
      (MF.draw (get_item city-name i)
               (+ (get_item cx i) 3) (+ (get_item cy i) 3) 127 127 127)
      (set! i (+ i 1))
    })
    (set! i 1)
    (while (< i N) {
      (set! a (get_item ind (- i 1)))
      (set! c (get_item ind i))
      (Gfx.draw_line_aa (get_item cx a) (get_item cy a)
                        (get_item cx c) (get_item cy c) 1.0 0 0 255)
      (set! i (+ i 1))
    })
    (set! a (get_item ind (- N 1)))
    (set! c (get_item ind 0))
    (Gfx.draw_line_aa (get_item cx a) (get_item cy a)
                      (get_item cx c) (get_item cy c) 1.0 0 0 255)
    ;; 左下角操作提示（地图西南角没有城市，正好空着）
    (MF.draw "TAP=NEW" 6 196 96 96 96)
  }))

;; 距离曲线：最近 HIST 代的最优路径长度（绿折线）+ 当前值（超小字体白字）
;; **抽稀**：窗口满时每隔 kmax 个点取一个（目标 ≤100 段）。AA 线是本地库调用
;; （~5µs/次）加每段十几次 VM 运算（~2µs），300 段就要 2ms+；抽到 100 段后
;; ~0.8ms，而 22px 高的趋势线肉眼无差别。
;; 纵轴自适应时 y 必落在 [CURVE_TOP, CURVE_TOP+CURVE_H-1] 内（hmin/hmax 就取自
;; 同一批数据），故不必逐段钳位——省掉每段 4 次比较。
(define draw-curve
  (lambda () {
    (define valid (if (< hist-n HIST) hist-n HIST))
    (define start (if (<= hist-n HIST) 0 (mod hist-n HIST)))
    (define kmax (if (> valid 100) (+ 1 (Math.floor (/ (- valid 1) 100))) 1))
    (define hs history)
    (define hmin 1.0e30)
    (define hmax 0.0)
    (define i 0)
    (define v 0.0)
    (define x0 0.0)
    (define y0 0.0)
    (define x1 0.0)
    (define y1 0.0)
    (define xs 1.0)
    (define ys 1.0)
    (Gfx.fill_rect 0 CURVE_TOP SW CURVE_H 20 20 20)
    (if (> valid 0)
        {
          ;; 纵轴自适应：取窗口内的最小/最大值（C 版同）
          (set! i 0)
          (while (< i valid) {
            (set! v (get_item hs (mod (+ start i) HIST)))
            (if (< v hmin) (set! hmin v) #f)
            (if (> v hmax) (set! hmax v) #f)
            (set! i (+ i 1))
          })
          (if (<= hmax hmin) (set! hmax (+ hmin 1.0)) #f)
          (if (> valid 1)
              {
                (set! xs (/ SW (- valid 1)))
                (set! ys (/ (- CURVE_H 1) (- hmax hmin)))
                (set! i 0)
                (while (< i valid) {
                  (set! x0 (* i xs))
                  (set! y0 (+ CURVE_TOP (* (- hmax (get_item hs (mod (+ start i) HIST))) ys)))
                  (set! x1 (+ x0 (* kmax xs)))
                  (set! y1 (+ CURVE_TOP (* (- hmax (get_item hs (mod (+ start (+ i kmax)) HIST))) ys)))
                  (Gfx.draw_line_aa x0 y0 x1 y1 1.0 0 255 0)
                  (set! i (+ i kmax))
                })
              }
              #f)
          (MF.draw (fmt1 best-len) 4 (+ CURVE_TOP 2) 255 255 255)
        }
        #f)
  }))

;; 整屏重绘（不含推屏）
(define draw
  (lambda () {
    (Gfx.clear)
    (draw-header)
    (draw-map)
    (draw-curve)
  }))

;; ============================================================================
;; 交互与主循环
;; ============================================================================

;; 是否有"重新演化"请求：实体键盘 Enter，或点在地图区（触屏）。
;; 必须在 (EX.check) 之后调用——落在右上角「退出」按钮上的那一次按下/松开
;; 已被 exit.scm 吞掉，不会误触发重开。
(define restart-requested
  (lambda () {
    (define hit 0)
    (define k 0)
    (set! k (Key.take_press))
    (while (not (== k 0)) {
      (if (== k 13) (set! hit 1) #f)   ; NANO_KEY_enter
      (set! k (Key.take_press))
    })
    (if (Touch.is_available)
        {
          (if (Touch.take_press)
              (if (and (>= (Touch.y) MAP_TOP) (< (Touch.y) MAP_BOTTOM))
                  (set! hit 1)
                  #f)
              #f)
          (Touch.take_release)
        }
        #f)
    hit
  }))

;; 每帧一拍：退出检查 → 重开（如请求）→ 演化 gens-n 代 → 整屏重绘 → 推屏 → 调参。
;; 推屏时机由本程序控制（宿主不主动推屏）：算与画同一拍，画面每帧刷新。
;; **自适应代数**是本 App"保证刷新率"的关键：一拍的工作量必须留在帧预算内，
;; 否则解释器被抢占、这一拍跨两帧才画完（帧率还行但画面每 3 帧才变一次）。
;; 这里用 (Diag.ms) 量单拍工作量（算+画+推屏）：**单拍超预算立刻降一档**（下一拍
;; 就少跑一代），滑动平均连续 20 拍有余量才升一档（两级滞回，不来回抖）。
(define tick
  (lambda () {
    (EX.check)
    (if (== (restart-requested) 1) (reset) #f)
    (define t0 (Diag.ms))
    (define k 0)
    (while (< k gens-n) { (evolve) (set! k (+ k 1)) })
    (draw)
    (EX.draw_button)
    (Gfx.refresh)
    (define work (- (Diag.ms) t0))
    (set! work-avg (+ (* work-avg 0.7) (* 0.3 work)))
    ;; ① 立刻降档：本拍已撑破帧预算（负载突增/慢机器）→ 下一拍就少跑一代，
    ;;    不等滑动平均收敛，避免连续几帧掉帧（掉帧就是"卡一下"）。
    (if (> work GENS-KICK)
        (if (> gens-n 1) (set! gens-n (- gens-n 1)) #f)
        #f)
    ;; ② 缓慢升档：滑动平均确认还有余量、且连续 20 拍都没超预算 → 加一代。
    (set! adj-cnt (+ adj-cnt 1))
    (if (>= adj-cnt 20)
        {
          (set! adj-cnt 0)
          (if (< work-avg GENS-LO)
              (if (< gens-n GENS-MAX) (set! gens-n (+ gens-n 1)) #f)
              #f)
        }
        #f)
  }))

;; ---- 启动 ------------------------------------------------------------------

;; 单拍预算：按 GENS-MAX 代留足时间片。时间片是"一帧内允许解释器跑多少条 VM
;; 指令"的上限——给少了这一拍会被抢占、跨两帧才画完（画面顿），给多了不会浪费
;; （进程没活干就提前让出）。宿主把该值夹在默认值的 1/10 ~ 4 倍之间。
(App.set_timeslice (+ 400000 (* 120000 GENS-MAX)))

(set! cost (init-cost))
(set! tmpl (make-fill N -1))
(set! seq (make-seq N))
(set! cum (make-fill POP 0.0))
(set! used (make-fill N 0))
(set! history (make-fill HIST 0.0))
(init-screen)

(reset)
(evolve)
(draw)
(EX.draw_button)
(Gfx.refresh)

;; 每帧一拍：间隔 1ms = "每轮事件循环跑一次"（与 app/gol.scm 同一手法）。
;; 宿主每轮只对同一进程触发一个定时器回调，故这就是 60fps 的帧节拍。
(set! _timer (System.set_interval 1 tick))

;; ---- 退出：硬 A/ESC / 右上角按钮 → 全局确认框（见 app/lib/exit.scm）-------
;; 每拍已整屏重绘，无需登记重绘函数（(EX.install) 即可擦掉确认框残影）。
(EX.install)
