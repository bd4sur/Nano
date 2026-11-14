;; ============================================================================
;; 粒子生命（Particle Life）—— Nano「小游戏」Scheme App
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库
;; 参考实现：ui_particlelife.c（particle.html 移植）：6 种颜色的粒子在环面世界中
;; 按随机种间吸引/排斥参数自组织。
;;
;; 【性能结构】本文件是**纯 Scheme**实现，性能全部靠"少让 VM 干活"：
;;
;; 一、2026-11 早先一轮：算法从「全对 O(N²)」改为 C 版同构的
;;     「均匀网格 + 半程邻域壳 + 对称化（一次距离、双向加力）」：
;;       · 候选对从全对 4950（无序）降到 ~2200（只查同格 + 7 个前向邻格）；
;;       · 每对只算一次距离，两方向的力同时施加（力矩阵非对称性照样保留）；
;;       · 格子按环面精确等分 → 跨格接缝的整格平移在邻格循环外一次算好，
;;         内层不需要逐对做 4 个回绕判断；
;;       · 网格用 head/next 下标链表表示，每步只覆写元素、不分配对象（零 GC）。
;;
;; 二、本轮（性能专项）：**热循环里的模块级变量全部本地槽位化**。这是最大的一笔。
;;     本解释器把顶层 define 在闭包体内当作"自由变量快照"：每次读取走通用变量解析
;;     （`load` 的 varid 路径：沿闭包链 deref + 绑定表查找），wasm32 实测约 77ns，
;;     是一条普通 VM 指令（~8ns）的 9 倍。原 interact/move 每次迭代要读
;;     xs/ys/ks/vxs/vys/param-*/GRID_*/RMAX2/RTH1/... 几十次，实测每步 **3.37 万次
;;     通用解析**（33,648 × 77ns ≈ 2.6ms，占该步 5.1ms 的一半）。把用到的常量与表在函数开头一次性搬进
;;     本地槽位（`(define lx xs)` 之类）后，每步只剩 **180 次**，同一粒子/同一轨迹下
;;     每步耗时降到 **1/2.5**（指令条数几乎不变，变的是每条指令的代价）。
;;     同一手法见 doc/AGENTS.md「2026-11-14 生命游戏性能专项」的通用结论。
;;
;; 三、配套的三项小改动（合计再省约 10% 指令）：
;;     1) 邻格表预计算（NBC2/NBSX/NBSY）：每格 × 8 个壳的「邻格下标 + 整格周期
;;        位移」在初始化时算好，热循环里只剩 3 次 get_item（原来每格每壳要现场做
;;        环绕加减与两串 if，约 78 条指令 → 约 16 条）；
;;     2) 力核分段改成"远距优先"（远距段覆盖 75% 的对，先测 rth2 再测 rth1，
;;        平均每对少一次比较）；
;;     3) 单帧预算按粒子数申请（App.set_timeslice）：宿主每帧只给解释器一个时间片
;;        （默认 312500 条指令）。一步 ~「22·N² + 500·N」条指令，超了就会被抢占拆到
;;        下一帧（帧率不变、步频减半）。按 N 申请预算后可以把粒子数从 100 提到 169
;;        （13×13），仍是每帧一步（60 步/秒）。
;;
;; 实测（WASM-GUI；headless chromium，真 wasm32，同机同构建脚本，5s 窗口 ×3 取中位）：
;;   版本                粒子  每步 VM 指令  每帧 CPU*  步/秒  CPU 占用*
;;   旧（本文件上一版）  100    23.4 万      6.3 ms     60    37.7%
;;   本轮（同粒子数）    100    22.3 万      3.0 ms     60    17.7%
;;   本轮（默认档）      169    55   万      5.2 ms     60    30.9%
;;   * 每帧 CPU = CDP TaskDuration / 帧长；其中约 1.2ms 是绘制与上屏的固定成本
;;     （与粒子数几乎无关），其余才是物理步。
;; 即：同粒子数下 CPU 降到 1/2.1；默认档用 0.82 倍的 CPU 换 1.69 倍的粒子数。
;; 参照物 C 版（ui_particlelife.c，324 粒子、每帧一步）实测 6.3% CPU。纯 Scheme 在
;; 本解释器里每条指令 ~8ns，324 粒子约需 250 万条指令/步（≈19ms/帧），**在 60 步/秒
;; 的帧预算内做不到**——这是解释器与机器码之间 ~15 倍的固有差距，不是写法问题
;; （本轮已把"写法"这一侧的可省项基本清零，见下）。故默认取 169 粒子：仍能稳定
;; 每帧一步的最大档。
;;
;; 实现要点（都是踩过的坑）：
;;   - 粒子状态为 5 个平行一维列表（xs/ys/ks/vxs/vys；get_item/set_item! O(1)）；
;;   - 参数矩阵 param-a/param-b 为 36 元素列表（idx = kind_i*NK + kind_j）；
;;   - '() 字面量是模块级静态槽，仅可安全 push 一次（各表独立字面量）；
;;   - 语句位置的分支一律写成形如 { (set! x ...) ... } 的净零块（vendored 编译器
;;     不回收语句位置的残留值，两分支若一留值一不留会破坏操作数栈）；
;;   - 算术一律二元嵌套（本解释器编译器对多元乘法只发射一次 mul）；
;;   - while 循环体内不调用 lambda（vendor 版 ARN 在 while 内 call lambda 有作用域
;;     缺陷，见 doc/AGENTS.md 2026-10-18 sky.scm 条目），故物理全部内联；
;;   - 本地槽位化只在 lambda 体开头做一次（interact 53 个、move 16 个、draw 14 个）：
;;     槽位寻址是编译期帧描述符 + O(1) 下标，实测 53 个本地槽位没有性能拐点
;;     （对比：`tools/e2e/host/bench/c_var_*.scm` 的 4~64 个绑定扫描）。
;;
;; 玩法：进入即随机生成相互作用参数与初始分布并开始演化；A(ESC) 返回列表；
;; HUD 右侧「重置」按钮 = 重新抽参数 + 重新撒布（同 C 版按 D 重置），「退出」= 硬 A。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native App)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 物理常数（与 ui_particlelife.c 一致；世界半宽 1.0，x∈[-1,1] 环面）-----
;; 视口几何先行定义（WORLD_HH 依赖 HUD_H/SW/SH）。

(define NK 6)                     ; 粒子种类数
(define RTH1 0.1)                 ; 近距排斥阈值
(define RTH2 0.2)                 ; 中距阈值
(define RMAX 0.4)                 ; 相互作用半径
(define RMAX2 (* RMAX RMAX))
(define RTH2_M_RTH1 (- RTH2 RTH1))
(define RMAX_M_RTH2 (- RMAX RTH2))
;; 倒数常量：把相互作用核里的 3 个除法换成乘法（除法是 VM 里最贵的算术指令，
;; 每步要跑几百次）。语义不变（都是浮点，误差在 1 ulp 量级）。
(define INV_RTH1 (/ 1.0 RTH1))
(define INV_RTH2_M_RTH1 (/ 1.0 RTH2_M_RTH1))
(define INV_RMAX_M_RTH2 (/ 1.0 RMAX_M_RTH2))
(define JITTER 0.002)             ; 随机震颤幅度
(define DAMPING 0.5)              ; 每步速度衰减系数
(define WORLD_HW 1.0)             ; 世界半宽
(define WORLD_W 2.0)              ; 回绕周期（宽）
(define HUD_H 16)                 ; 顶部信息栏高度（像素）
(define SW (Gfx.width))
(define SH (Gfx.height))
(define WORLD_HH (/ (- SH HUD_H) SW))   ; 世界半高（填满 HUD 以下视口）
(define WORLD_HH2 (* 2.0 WORLD_HH))
(define WORLD_HW_NEG (- 0.0 WORLD_HW))
(define WORLD_HW_NEG2 (- 0.0 WORLD_HH))
(define SCALE (/ SW 2.0))         ; 像素/世界单位
(define CX (/ SW 2.0))            ; 世界原点屏幕坐标
(define CY (+ HUD_H (/ (- SH HUD_H) 2.0)))

;; ---- 粒子状态（5 个平行一维列表）与相互作用参数矩阵 -----------------------

;; 粒子撒布网格边长：N = SCATTER_N²。按平台分两档（解释器速度差一个数量级）：
;;   · wasm-gui（浏览器）：13 → 169 粒子。这是"每步仍能在一个时间片内跑完
;;     （60 步/秒）"的最大档：一步约 55 万条指令 / 本机 4 ms（见下方 App.set_timeslice）；
;;   · 其余平台（m5esp / tty / linux / web）：10 → 100 粒子，与上一版一致
;;     （ESP32 上的解释器比 wasm 慢约一个数量级，169 会把单帧拖到几十毫秒）。
;; 想更密/更省就改这里：时间片预算随 N 自动缩放（见 App.set_timeslice 那几行）。
(define SCATTER_N (if (String.equals (App.platform) "wasm-gui") 13 10))
(define N (* SCATTER_N SCATTER_N))
(define xs '())
(define ys '())
(define ks '())
(define vxs '())
(define vys '())
(define param-a '())              ; 近距（<RTH1）作用强度矩阵（36 元素）
(define param-b '())              ; 中远距作用强度矩阵（36 元素）
(define gen 0)

;; ---- 均匀网格（同 ui_particlelife.c 的加速结构，但格子按环面精确等分）-------
;; 「距离 ≤ RMAX」的粒子对必落在相邻格内。横向格宽取 RMAX、纵向取 RMAX/2，
;; 两者都按世界尺寸**等分**（格数 = floor(周期/目标格宽)），使网格恰好是环面的
;; 一个划分——只有这样才能用「邻格周期位移 = ±整周期」代替逐对回绕判断。
;; 每格只与「同格 + 7 个前向邻格」配对（见 shell-x/shell-y），每对恰好访问一次；
;; 一次距离计算同时给双方施加力（对称化）。
;;
;; 表示法：head/next 两个平行一维列表存「下标链表」（-1 表示链尾/空格），
;; 每步只覆写元素、不分配新对象——避免每步重建几十个列表带来的 GC 压力。
;; 格数下限：横向 ≥3（半程壳取模不重复计数）、纵向 ≥5（纵向壳跨 3 行）。
(define CELL_W RMAX)                      ; 横向目标格宽
(define CELL_H (/ RMAX 2.0))              ; 纵向目标格高
;; +0.001：float32 宿主上 1.4/0.2 会算成 6.99998，floor 会白丢一行（格子变高、
;; 候选对变多）；加个容差使 64 位宿主与 wasm32 得到同一套格数（不影响正确性）。
(define GRID_X
  (if (< WORLD_W (* 3.0 CELL_W)) 3.0 (Math.floor (+ (/ WORLD_W CELL_W) 0.001))))
(define GRID_Y
  (if (< WORLD_HH2 (* 5.0 CELL_H)) 5.0 (Math.floor (+ (/ WORLD_HH2 CELL_H) 0.001))))
(define CW (/ WORLD_W GRID_X))            ; 实际格宽（= 周期/格数：环面精确等分）
(define CH (/ WORLD_HH2 GRID_Y))          ; 实际格高
(define INV_CW (/ 1.0 CW))
(define INV_CH (/ 1.0 CH))
(define GRID_N (* GRID_X GRID_Y))
(define ghead '())                ; 每格链头（粒子下标 / -1）
(define gnext '())                ; 每个粒子的后继（下标 / -1）
;; 半程邻域壳：(0,0) 同格 + 7 个前向邻格
(define shell-x '(0 1 -1 0 1 -1 0 1))
(define shell-y '(0 0 1 1 1 2 2 2))

;; 邻格表（初始化时算一次，interact 每步只查表）：下标 = 格号*8 + 壳号
;;   NBC2 = 邻格格号；NBSX/NBSY = 该邻格相对本格的整周期位移（±周期 或 0）
(define NBC2 '())
(define NBSX '())
(define NBSY '())

;; ---- 单帧预算 --------------------------------------------------------------
;; 宿主每帧只驱动解释器一个时间片（默认 312500 条 VM 指令，见 ui_scheme_app.c 的
;; MB_TIMESLICE_TICKS）。本 App 一步实测约「19·N² + 300·N」条指令
;; （N=100 → 22.3 万，N=169 → 55.2 万）：超过预算就被抢占拆到下一帧——帧率不变但
;; **步频直接减半**。故按「22·N² + 500·N」申请（约 25% 余量；宿主把值夹在默认值的
;; 1/10 ~ 4 倍之间）。
;; 注意算术必须二元嵌套：本解释器编译器对多元乘法只发射一次 mul（(* 25 N N) 会算成 25*N）。
;; 下限锁在宿主默认值（312500）：小 N 时不主动缩小别人的预算（没必要冒"个别步被拆帧"的风险）；
;; 上限由宿主自己夹（默认值的 4 倍 = 1250000）。
(define TIMESLICE (if (< (* (* 22 N) N) 262500) 312500 (+ (* (* 22 N) N) (* 500 N))))
(App.set_timeslice TIMESLICE)

;; ---- 调色板（与 C 版一致：#f80 #ff0 #4f0 #0cf #04f #f08）-------------------

(define pal-r '(255 255 68 0 0 255))
(define pal-g '(136 255 255 204 68 0))
(define pal-b '(0 0 0 255 255 136))

;; 网格与邻格表初始化（仅启动时一次；列表内容之后每步被覆写，长度不变）
(define grid-init
  (lambda () {
    (define i 0)
    (define gy 0)
    (define gx 0)
    (define k 0)
    (define wx 0)
    (define wy 0)
    (define shx 0.0)
    (define shy 0.0)
    (while (< i GRID_N) { (push ghead -1) (set! i (+ i 1)) })
    (set! i 0)
    (while (< i N) { (push gnext -1) (set! i (+ i 1)) })
    ;; 邻格表：每格 × 8 个半程壳，预先算好「邻格格号 + 整周期位移」。
    ;; 这样热循环里不必再做「环绕加减 + 两串 if」（原来每格每壳约 78 条指令）。
    (set! gy 0)
    (while (< gy GRID_Y) {
      (set! gx 0)
      (while (< gx GRID_X) {
        (set! k 0)
        (while (< k 8) {
          (set! wx (+ gx (get_item shell-x k)))
          (set! shx 0.0)
          (if (>= wx GRID_X)
              { (set! wx (- wx GRID_X)) (set! shx WORLD_W) }
              { (if (< wx 0.0) { (set! wx (+ wx GRID_X)) (set! shx (- 0.0 WORLD_W)) }) })
          (set! wy (+ gy (get_item shell-y k)))
          (set! shy 0.0)
          (if (>= wy GRID_Y)
              { (set! wy (- wy GRID_Y)) (set! shy WORLD_HH2) }
              { (if (< wy 0.0) { (set! wy (+ wy GRID_Y)) (set! shy (- 0.0 WORLD_HH2)) }) })
          (push NBC2 (+ (* wy GRID_X) wx))
          (push NBSX shx)
          (push NBSY shy)
          (set! k (+ k 1))
        })
        (set! gx (+ gx 1))
      })
      (set! gy (+ gy 1))
    })
  }))

;; 相互作用：网格 + 半程邻域壳 + 对称化（一次距离计算，双向施加力）
;;
;; 全部模块级量在开头搬进本地槽位（`load_slot` 一条指令），热循环里不再碰模块变量
;; （每次 `load` 是通用的闭包链解析，实测 ~77ns ≈ 9 条指令）——这是本文件最大的一笔
;; 性能收入，见文件头"性能结构"第二节。
(define interact
  (lambda () {
    ;; ---- 本地槽位化（33 个本地 define，见文件头关于槽位上限的说明）----
    (define nn N)
    (define nk NK)
    (define hw WORLD_HW)
    (define hh WORLD_HH)
    (define wh2 WORLD_HH2)
    (define ww WORLD_W)
    (define gxn GRID_X)
    (define gyn GRID_Y)
    (define gnn GRID_N)
    (define gh ghead)
    (define gn gnext)
    (define lx xs)
    (define ly ys)
    (define lk ks)
    (define lvx vxs)
    (define lvy vys)
    (define pa param-a)
    (define pb param-b)
    (define tcx NBC2)
    (define tsx NBSX)
    (define tsy NBSY)
    (define rmax2 RMAX2)
    (define rth1 RTH1)
    (define rth2 RTH2)
    (define rmax RMAX)
    (define irth1 INV_RTH1)
    (define iw1 INV_RTH2_M_RTH1)
    (define iw2 INV_RMAX_M_RTH2)
    (define icw INV_CW)
    (define ich INV_CH)
    ;; ---- 循环变量与临时量 ----
    (define gy 0)
    (define gx 0)
    (define k 0)
    (define c 0)
    (define c2 0)
    (define nidx 0)
    (define shx 0.0)
    (define shy 0.0)
    (define i 0)
    (define j 0)
    (define xi 0.0)
    (define yi 0.0)
    (define ki 0)
    (define kbi 0)
    (define kj 0)
    (define kbj 0)
    (define ddx 0.0)
    (define ddy 0.0)
    (define dd 0.0)
    (define d 0.0)
    (define gk 0.0)
    (define s 0.0)
    (define s2 0.0)

    ;; ---- 1) 重建网格：清链头 → 逐粒子头插 ----
    (set! i 0)
    (while (< i gnn) { (set_item! gh i -1) (set! i (+ i 1)) })
    (set! i 0)
    (while (< i nn) {
      (set! gx (Math.floor (* (+ (get_item lx i) hw) icw)))
      (if (< gx 0.0) { (set! gx 0.0) })
      (if (>= gx gxn) { (set! gx (- gxn 1.0)) })
      (set! gy (Math.floor (* (+ (get_item ly i) hh) ich)))
      (if (< gy 0.0) { (set! gy 0.0) })
      (if (>= gy gyn) { (set! gy (- gyn 1.0)) })
      (set! c (+ (* gy gxn) gx))
      (set_item! gn i (get_item gh c))
      (set_item! gh c i)
      (set! i (+ i 1))
    })

    ;; ---- 2) 逐格 × 半程邻域壳 ----
    (set! gy 0)
    (while (< gy gyn) {
      (set! gx 0)
      (while (< gx gxn) {
        (set! c (+ (* gy gxn) gx))
        (set! k 0)
        (while (< k 8) {
          ;; 邻格格号与整周期位移：查表（见 grid-init），热循环里只剩 3 次 get_item。
          ;; 跨接缝时整格平移一个周期，因此内层**不需要逐对做环面回绕判断**。
          (set! nidx (+ (* c 8) k))
          (set! c2 (get_item tcx nidx))
          (set! shx (get_item tsx nidx))
          (set! shy (get_item tsy nidx))
          (set! i (get_item gh c))
          (while (>= i 0) {
            (set! xi (get_item lx i))
            (set! yi (get_item ly i))
            (set! ki (get_item lk i))
            (set! kbi (* ki nk))
            ;; 同格：从 i 的后继开始遍历（无序对，只算一次）；邻格：从头遍历
            (set! j (if (== k 0) (get_item gn i) (get_item gh c2)))
            (while (>= j 0) {
              ;; 最近镜像位移 = 直接差 + 该邻格的周期位移（不再逐对判断回绕）
              (set! ddx (+ (- (get_item lx j) xi) shx))
              (set! ddy (+ (- (get_item ly j) yi) shy))
              (set! dd (+ (* ddx ddx) (* ddy ddy)))
              (if (<= dd rmax2) {
                (set! d (Math.sqrt dd))
                ;; dd==0 只可能出现在两粒子完全重合时（同格自配对已被排除），
                ;; 此时 gk/d 会除零报错——直接跳过这一对（对力无贡献）。
                (if (> d 0.0) {
                  ;; 与 C 版 pl_interact_pair 同构：形状因子 gk(d) 与种类无关（一对只算一次），
                  ;; 乘子分两段取矩阵——近距用 param-a，中/远距用 param-b（后者可正可负，
                  ;; 才是「吸引/排斥混合」与每次运行不同图案的来源）。
                  ;; 力 = 系数 × gk × d⃗，其中 gk 已含 1/d：
                  ;;   近：gk = (RTH1−d)/(RTH1·d)；中：gk = (d−RTH1)/((RTH2−RTH1)·d)；
                  ;;   远：gk = (RMAX−d)/((RMAX−RTH2)·d)
                  ;; 分段顺序「先测远距」：远距段覆盖约 3/4 的对，平均每对少一次比较。
                  (set! kj (get_item lk j))
                  (set! kbj (* kj nk))
                  (if (< d rth2)
                      { (if (< d rth1)
                            { (set! gk (/ (* (- rth1 d) irth1) d))
                              (set! s (* (get_item pa (+ kbi kj)) gk))
                              (set! s2 (* (get_item pa (+ kbj ki)) gk)) }
                            { (set! gk (/ (* (- d rth1) iw1) d))
                              (set! s (* (get_item pb (+ kbi kj)) gk))
                              (set! s2 (* (get_item pb (+ kbj ki)) gk)) }) }
                      { (set! gk (/ (* (- rmax d) iw2) d))
                        (set! s (* (get_item pb (+ kbi kj)) gk))
                        (set! s2 (* (get_item pb (+ kbj ki)) gk)) })
                  (set_item! lvx i (+ (get_item lvx i) (* s ddx)))
                  (set_item! lvy i (+ (get_item lvy i) (* s ddy)))
                  ;; j 受 i 的力（参数矩阵非对称：另取转置下标；方向取反）
                  (set_item! lvx j (- (get_item lvx j) (* s2 ddx)))
                  (set_item! lvy j (- (get_item lvy j) (* s2 ddy)))
                })
              })
              (set! j (get_item gn j))
            })
            (set! i (get_item gn i))
          })
          (set! k (+ k 1))
        })
        (set! gx (+ gx 1))
      })
      (set! gy (+ gy 1))
    })
  }))

;; 运动：震颤 + 阻尼 + 积分 + 环面回绕（同样整体本地槽位化）
(define move
  (lambda () {
    (define nn N)
    (define lx xs)
    (define ly ys)
    (define lvx vxs)
    (define lvy vys)
    (define hw WORLD_HW)
    (define hh WORLD_HH)
    (define ww WORLD_W)
    (define wh2 WORLD_HH2)
    (define hwn WORLD_HW_NEG)
    (define hwn2 WORLD_HW_NEG2)
    (define jit JITTER)
    (define damp DAMPING)
    (define i 0)
    (define x 0.0)
    (define y 0.0)
    (while (< i nn) {
      (set_item! lvx i (+ (get_item lvx i) (* jit (- (Math.random) 0.5))))
      (set_item! lvy i (+ (get_item lvy i) (* jit (- (Math.random) 0.5))))
      (set! x (+ (get_item lx i) (get_item lvx i)))
      (set! y (+ (get_item ly i) (get_item lvy i)))
      (set_item! lvx i (* (get_item lvx i) damp))
      (set_item! lvy i (* (get_item lvy i) damp))
      (if (> x hw) { (set! x (- x ww)) })
      (if (< x hwn) { (set! x (+ x ww)) })
      (if (> y hh) { (set! y (- y wh2)) })
      (if (< y hwn2) { (set! y (+ y wh2)) })
      (set_item! lx i x)
      (set_item! ly i y)
      (set! i (+ i 1))
    })
  }))

;; 绘制（同样本地槽位化：调色板与几何常量在 169 粒子的循环里各读 169 次）
(define draw
  (lambda () {
    (define nn N)
    (define lx xs)
    (define ly ys)
    (define lk ks)
    (define pr pal-r)
    (define pg pal-g)
    (define pb pal-b)
    (define sw SW)
    (define cx CX)
    (define cy CY)
    (define sc SCALE)
    (define hud HUD_H)
    (define i 0)
    (define k 0)
    (Gfx.clear)
    ;; 先画粒子、后画 HUD 条与按钮：HUD 是不透明覆盖层，这样贴到顶部的粒子
    ;; 不会压在「重置/退出」按钮的描边上（按钮占 y=2..18，越过了 HUD 条 0..15）。
    (while (< i nn) {
      (set! k (get_item lk i))
      (Gfx.fill_rect
        (Math.floor (+ cx (* (get_item lx i) sc)))
        (Math.floor (+ cy (* (get_item ly i) sc)))
        2 2 (get_item pr k) (get_item pg k) (get_item pb k))
      (set! i (+ i 1))
    })
    (Gfx.fill_rect 0 0 sw hud 24 24 28)
    (Gfx.draw_text "粒子生命" 6 2 255 255 255)
    (draw_reset_button)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; ---- 启动：参数与撒布 → 首帧绘制 → 每帧一步演进 -------------------------

;; ---- 状态表：一次性分配（长度固定），之后只 set_item! 覆写 ------------------
;; 为什么不在每次重置时重新 push：'() 是模块级静态字面量，重复 push 会让同一个
;; 对象越来越长（本解释器不回收语句位置的残留值，且字面量是共享的）；而 set_item!
;; 覆写既没有这个坑，也不产生任何分配（重置不触发 GC）。

(define state-alloc
  (lambda () {
    (define i 0)
    (while (< i N)
      { (push xs 0.0) (push ys 0.0) (push ks 0) (push vxs 0.0) (push vys 0.0)
        (set! i (+ i 1)) })
    (set! i 0)
    (while (< i (* NK NK)) { (push param-a 0.0) (push param-b 0.0) (set! i (+ i 1)) })
  }))

;; 重置世界：重新抽相互作用参数 + 重新撒布粒子（同 C 版 pl_generate_world 的分布）
(define world-reset
  (lambda () {
    (define i 0)
    (define j 0)
    (define ix 0)
    (define iy 0)
    (define idx 0)
    (define px 0.0)
    (define py 0.0)
    ;; 相互作用参数：同种更近距（param-a 恒为负 = 近距排斥），param-b 可正可负
    (set! i 0)
    (while (< i NK) {
      (set! j 0)
      (while (< j NK) {
        (set! idx (+ (* i NK) j))
        (if (== i j)
            { (set_item! param-a idx (- 0.0 (+ 0.001 (* 0.0005 (Math.random))))) }
            { (set_item! param-a idx (- 0.0 (+ 0.0005 (* 0.0005 (Math.random))))) })
        (set_item! param-b idx (* 0.001 (- (Math.random) 0.5)))
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    ;; 粒子撒布：均匀格 + 抖动，种类随机，速度归零
    (set! iy 0)
    (while (< iy SCATTER_N) {
      (set! ix 0)
      (while (< ix SCATTER_N) {
        (set! idx (+ (* iy SCATTER_N) ix))
        ;; 撒布后立即回绕到世界内：网格划分要求坐标在 [-HW,HW]/[-HH,HH] 内
        ;; （否则会被 clamp 到边格，邻域壳算出的镜像位移会偏），而 move 是下一步
        ;; 才回绕的。环面下回绕只是同一个点的另一种写法，物理不变。
        (set! px (+ (* (- (* 2.0 (/ ix (- SCATTER_N 1))) 1.0) WORLD_HW) (* 0.2 (- (Math.random) 0.5))))
        (if (> px WORLD_HW) { (set! px (- px WORLD_W)) })
        (if (< px WORLD_HW_NEG) { (set! px (+ px WORLD_W)) })
        (set_item! xs idx px)
        (set! py (+ (* (- (* 2.0 (/ iy (- SCATTER_N 1))) 1.0) WORLD_HH) (* 0.2 (- (Math.random) 0.5))))
        (if (> py WORLD_HH) { (set! py (- py WORLD_HH2)) })
        (if (< py WORLD_HW_NEG2) { (set! py (+ py WORLD_HH2)) })
        (set_item! ys idx py)
        (set_item! ks idx (Math.floor (* NK (Math.random))))
        (set_item! vxs idx 0.0)
        (set_item! vys idx 0.0)
        (set! ix (+ ix 1))
      })
      (set! iy (+ iy 1))
    })
  }))

;; ---- 「重置」按钮（HUD 右侧，紧挨「退出」）---------------------------------
;; 命中判定用「电平 + 按下沿」：本 App 每帧一拍（≈16ms），一次点按跨好几拍，
;; 电平采样不会漏；命中后把该按下沿取走，避免同一手势被判两次。
(define RB_W 36)
(define RB_H 16)
(define RB_Y 2)
(define rb_x (- SW 80))           ; 退出按钮在 SW-40，这里留 4px 间距
(define prev_down 0)
(define reset_req 0)

(define rb_hit
  (lambda (px py)
    (if (and (>= px rb_x) (< px (+ rb_x RB_W)))
        (if (and (>= py RB_Y) (< py (+ RB_Y RB_H))) 1 0)
        0)))

(define draw_reset_button
  (lambda ()
    (if (Touch.is_available)
        {
          (Gfx.fill_rect rb_x RB_Y RB_W RB_H 60 62 74)
          (Gfx.draw_rect rb_x RB_Y RB_W RB_H 120 124 140)
          (Gfx.draw_text "重置" (+ rb_x 6) (+ RB_Y 2) 225 228 235)
        }
        #f)))

(define read_touch
  (lambda () {
    (if (Touch.is_available)
        {
          (if (Touch.is_pressed)
              {
                (if (== prev_down 0)
                    (if (== (rb_hit (Touch.x) (Touch.y)) 1)
                        { (set! reset_req 1) (Touch.take_press) }
                        #f)
                    #f)
                (set! prev_down 1)
              }
              { (set! prev_down 0) })
        }
        #f)
  }))

(define step
  (lambda () {
    (EX.check)                       ; 返回请求 / 右上角「退出」按钮
    (read_touch)                     ; 「重置」按钮（命中即置 reset_req）
    (if (== reset_req 1)
        { (set! reset_req 0) (world-reset) }
        { (interact) (move) })
    (draw)
  }))

(state-alloc)
(world-reset)
(grid-init)
(draw)
(step)
;; 节奏：每帧一步（≈60fps）。周期取「半帧」而非整帧——宿主只在 GUI 帧边界触发
;; 定时器（浏览器 rAF ≈16.7ms），16ms 周期会被向上量化成「有时隔两帧」，实测
;; 60fps 下只有 50 步/秒；8ms 则每帧必到期，实测稳定 60 步/秒。一步的指令数由
;; 文件头的 App.set_timeslice 预算保证能在一个时间片内跑完（否则步频减半）。
(set! _timer (System.set_interval 8 step))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 draw 擦掉对话框残影）
(EX.install_with draw)
