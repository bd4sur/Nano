;; ============================================================================
;; forkqueue —— 多进程 / 队列 IPC 演示（灵机 Scheme App）
;;
;; 一句话：把「操作系统」画在屏幕上——1 个父进程 + 4 个 fork 出来的子进程，
;; 每个子进程独占屏幕的 1/4（4 条竖列之一）管自己那批粒子；粒子越过列边界时
;; 经 System.write / System.read 队列**移交给邻列的进程**；顶部 HUD 是这张
;; 进程表：每个 PID 的消息量、入队水位、堆活对象数、VM 栈深，外加池级 memstat。
;; 顺带充当 fork / 队列的压力测试（持续跨界 = 持续出队入队）。
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx/Diag/Touch/App 本地库。
;; 与 App 列表里其它 App 的差别：这是第一个**多进程** App（System.fork）。
;;
;; ---------------------------------------------------------------------------
;; 一、拓扑（画出来的东西）
;;
;;        列0 P?     列1 P?     列2 P?     列3 P?
;;      ┌─────────┬─────────┬─────────┬─────────┐
;;      │  ●      │   ●     │ ●       │      ●  │   ← 每列由一个进程负责，
;;      │      ●  │ ●       │    ●    │  ●      │     粒子颜色 = 归属进程，
;;      │  ●      │      ●  │ ●       │     ●   │     靠近列边框的点画白色
;;      └─────────┴─────────┴─────────┴─────────┘       = 正在被移交
;;       ↑ 右越界 → 右邻列进程    左越界 → 左邻列进程
;;       ↑ 屏幕最右绕过 = 列3 → 列0（环面），最左 = 列0 → 列3
;;
;; 共 16 条队列，全部在 fork **之前**创建——队列编号是虚拟机全局资源，
;; 子进程天然共享，这正是 IPC 的基础（manual §8.4）：
;;   · 8 条粒子的有向边队列（单生产者单消费者）：
;;       列c --右越界--> 列(c+1)%4 用 qr[c]，列c --左越界--> 列(c+3)%4 用 ql[c]
;;   · 4 条统计上报（子 → 父，单生产者）
;;   · 4 条节拍令牌（父 → 子，单生产者）
;;
;; 二、进程角色
;;   · 父进程：唯一**渲染 HUD 与推屏**的人（单一刷新者，避免半帧撕裂）；
;;     40ms 一个 interval 回调：汇总统计 → 画 HUD → 推屏 → 给 4 个子进程
;;     各发一个**节拍令牌**（vsync）。退出流程也归它。
;;   · 4 个子进程：**事件驱动**。平时阻塞在控制队列上（等令牌，不占时间片），
;;     醒来先非阻塞排空两条入边队列（收粒子），再走一步物理并重画自己那一列。
;;
;;   为什么由父进程发令牌，而不是各子进程自己 set_interval：
;;     ① 帧率由唯一刷新者定，5 个进程节拍天然对齐，画面不抖；
;;     ② 退出确认框期间父进程阻塞在 (App.exit) 里、不再发令牌 → 子进程自动
;;        停笔，不必额外设计"暂停/恢复"协议（只多一个"重绘"令牌，见下）；
;;     ③ 子进程的唤醒只取决于一条队列，不受定时器实现细节影响。
;;   顺带一个实测结论（值得知道）：解释器每帧只推进**一个**进程的时间片
;;   （am_runtime_event_handler → 一次 am_runtime_tick），所以 N 个可运行进程
;;   的显示节拍 ≈ 宿主帧率 / N。本 App 5 个进程，浏览器 WASM 下约 12 帧/秒，
;;   单进程 App 则是满帧。这正是"多进程有代价"的现场演示。
;;
;; 三、传输协议（为什么是「数」而不是「表」）
;;   队列可以传任意值，但**堆对象跨不过进程边界**：把柄（handle）是进程私有的
;;   堆内索引（manual §3.4：fork 深拷贝堆，父子进程的"同名"对象身份不同），
;;   把一个列表塞进队列，接收方拿到的是**它自己堆里同号的对象**——纯属巧合。
;;   所以这里只传**立即值**，一颗粒子用两个数编码：
;;     位置 msg（偶数）= 2 * (xq + 160*yq)   xq=x/2∈[0,159]，yq=(y-区顶)/2∈[0,71]
;;     速度 msg（奇数）= 2 * ((vxq+16) + 32*(vyq+16)) + 1     速度按 1/4 像素量化
;;   偶数/奇数即"记录类型"，接收方据此**自同步**：万一位置已写、速度那条写失败
;;   （队列恰好满），接收方看到的是两个偶数——旧的半条被新位置覆盖，丢一颗而已，
;;   不会错配成畸形粒子。数值都压在 2^19 以内：32 位宿主（wasm32）浮点只有
;;   约 19 位有效位（见 infer/am_native_Time.h），再大就做不了 mod/floor 解码。
;;   统计上报同理：每条 = 6*值 + 标签（标签 0..5 在低位），按标签归位，
;;   丢一条也不会错位。
;;
;; 四、HUD 里的数从哪来
;;   · 发/收：该进程写出去 / 读进来的粒子记录条数（累计）。
;;   · 队峰：入队队列的**峰值水位**（条报文 / 容量），由子进程自己实测——
;;         队列没有"查询长度"接口，但排空时数走几条就是它的瞬时长度，
;;         而那个长度恰好在排空前达到最大，取历次最大值即峰值水位。
;;         （父进程按 发/收 差额推算瞬时值也可行，但采样时刻刚好在排空之后，
;;          永远是 0，没有信息量——这是踩过的坑。）
;;   · 堆：Diag.heap —— **该进程自己**堆里的存活对象数。
;;   · 栈：Diag.opstack + Diag.fstack —— 该进程的 VM 操作数栈/调用栈深度。
;;   · 池：System.memstat —— 注意它是**池级**口径（解释器池由全部进程共享，
;;         vm_used/heap_used 是整池数字），fork 之后各进程读到的几乎一样，
;;         所以它只画一次在第二行；per-PID 的堆/栈才用 Diag。
;;   · 丢：队列满导致的丢弃数（非阻塞写失败 + 半条记录被覆盖）。就是"丢包"。
;;
;; 五、退出：确认框期间子进程必须停笔
;;   (App.exit) 是**阻塞**的：父进程停在那里，但 4 个子进程照旧各自重画自己
;;   那一列——它们会把宿主画的确认框啃出一个个 2x2 的小洞。本 App 因此自己
;;   实现退出流程（不用 app/lib/exit.scm）：
;;     Esc / 右上角「退出」按钮 → 宿主置"返回请求" → 父进程 (App.exit) 阻塞
;;     （不再发令牌 = 子进程停笔）→ 用户「留下」→ 广播"重绘令牌" →
;;     子进程整列重绘，确认框残影一并擦掉。
;;   注：右上角按钮的热区由宿主在触摸边沿上判定（App.set_exit_button），
;;   与 App 的节拍快慢无关。
;;
;; 六、解释器约束（manual.md，逐条遵守）
;;   · 算术/比较恒二元：写 (* a (* b c)) 而不是 (* a b c)；
;;   · 没有 let/list/append/map；列表用 (cdr (cons 0 '())) 造独立空表再 push
;;     （直接往 '() 字面量上 push 会改到模块级静态槽）；
;;   · 语句位置的分支一律写成 { ... } 块；
;;   · while 循环体不是独立词法区域；
;;   · 计数值做**饱和**（≤50000）：上报报文用「6*值+标签」，值太大就超出
;;     32 位宿主浮点的精确整数范围，标签解码会错。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Diag)
(native Touch)
(native App)

;; ---------------------------------------------------------------------------
;; 一、布局
;; ---------------------------------------------------------------------------

(define SW (Gfx.width))            ; 320
(define SH (Gfx.height))           ; 240
(define HUD_H 96)                  ; 顶部进程表高度
(define AREA_Y HUD_H)              ; 粒子区顶边
(define AREA_H (- SH HUD_H))       ; 144
(define AREA_END (+ AREA_Y AREA_H))
(define COLW 80)                   ; (/ SW 4)：每列宽度

;; ---------------------------------------------------------------------------
;; 二、物理与节奏
;; ---------------------------------------------------------------------------

(define NP 40)                     ; 每列粒子数（全屏 160 个）
(define TICK_MS 40)                ; 父进程定时器周期 = 全局步进节拍（子进程跟随它）
(define STAT_EVERY 3)              ; 每 3 步上报一次统计（≈150ms）
(define VSYNC_FALLBACK_MS 500)     ; 等令牌的兜底超时（父进程意外停机时不至于永久挂住）
(define QCAP 64)                   ; 粒子入队队列容量（HUD 上按 水位/容量 显示）
(define DRIFT 1.30)                ; 右向漂移：保证跨界有稳定流量
(define SHEAR 0.90)                ; 流速的纵向剪切（把直流水流扭成波浪）
(define WAVE 0.055)                ; 剪切的空间频率（1/像素）
(define DAMP 0.94)                 ; 纵向速度的阻尼（随机游走不会越跑越远）
(define JIT 0.70)                  ; 随机抖动
(define VMAX 4.0)                  ; 限速（像素/步）

;; ---------------------------------------------------------------------------
;; 三、状态
;; ---------------------------------------------------------------------------

;; 角色：0 = 父进程，1 = 子进程（fork 后子进程把它置 1）
(define role 0)
(define my-id 0)                   ; 子进程的列号 0..3
(define lid 0)                     ; 左邻列号（左越界时移交目标）
(define rid 0)                     ; 右邻列号（右越界时移交目标）

;; 列号的左右邻（4 列成环：列 0 的左邻是最右边的列 3）
(define left-of (lambda (c) (if (== c 0) 3 (- c 1))))
(define right-of (lambda (c) (if (== c 3) 0 (+ c 1))))

(define pids '())                  ; 4 个子进程 PID（父进程记录）
(define qr '())                    ; 4 条：列 c → 右邻（单生产者单消费者）
(define ql '())                    ; 4 条：列 c → 左邻（同上）——共 8 条有向边队列
(define qs '())                    ; 4 条：统计上报（子 c → 父）
(define qc '())                    ; 4 条：节拍令牌（父 → 子：1 走一拍 / 2 走一拍并整列重绘）

;; 父进程的进程表（4 槽平行表；子进程也有副本，但不用）
(define st-sr '())                 ; 发向右邻的累计条数
(define st-sl '())                 ; 发向左邻的累计条数
(define st-rc '())                 ; 已收累计条数
(define st-hp '())                 ; 堆活对象数
(define st-st '())                 ; VM 栈深
(define st-pk '())                 ; 入队队列的实测峰值水位（条报文）
(define st-lo '())                 ; 丢弃条数
(define row-strings '())           ; 4 行 HUD 文本缓存（统计不变就不重建）
(define dirty 1)                   ; 有新统计 → 需要重建行文本
(define tot-msg 0)                 ; 全局累计消息量（父进程汇总）
(define tot-lost 0)

;; 子进程私有的粒子状态（4 个平行一维列表；fork 后各进程一份）
(define xs '())
(define ys '())
(define vxs '())
(define vys '())

(define sr 0)                      ; 本进程：发向右邻条数
(define sl 0)                      ; 本进程：发向左邻条数
(define rc 0)                      ; 本进程：已收条数
(define lost 0)                    ; 本进程：丢弃条数
(define pend-a -1)                 ; 左入边队列的半条记录槽
(define pend-b -1)                 ; 右入边队列的半条记录槽
(define arrived 0)                 ; 本拍排空时数到的报文条数
(define peakq 0)                   ; 历次排空的**峰值水位**（条报文）
(define tickn 0)                   ; 距上次上报走了几步
(define v 0)                       ; 临时：读到的队列值

;; 饱和计数：见文件头「解释器约束」最后一条
(define SAT 50000)
(define bump
  (lambda (n)
    (if (< n SAT) (+ n 1) n)))

;; 独立空表（绝不往 '() 字面量上 push——那是模块级静态槽）
(define newlist
  (lambda () (cdr (cons 0 (quote ())))))

;; ---------------------------------------------------------------------------
;; 四、文本小工具
;; ---------------------------------------------------------------------------

;; 右对齐到 w 个字符（12px 等宽字模，1 字符 = 7px，便于对齐成表）
(define padn
  (lambda (n w) {
    (define s (String.atom_to_string n))
    (while (< (String.length s) w) (set! s (String.concat " " s)))
    s
  }))

;; 百分比（避免 (* 100 used) 在 27 位无符号上溢出：先除后乘）
(define pct
  (lambda (used cap)
    (if (<= cap 0) 0 (Math.floor (* 100.0 (/ used cap))))))

;; 水平条形图
(define bar
  (lambda (x y w h p r g b) {
    (Gfx.fill_rect x y w h 40 44 54)
    (define fw (Math.floor (* w (/ p 100.0))))
    (if (> fw 0.0) (Gfx.fill_rect x y fw h r g b) #f)
  }))

;; ---------------------------------------------------------------------------
;; 五、退出流程（父）
;;
;; 与 app/lib/exit.scm 的两点差别：
;;   1) 不用 (EX.check) 的触屏电平判定——宿主已把「退出」按钮热区在**触摸边沿**
;;      上判掉并置"返回请求"，父进程只需轮询 (App.back_requested)；
;;   2) 不需要单独的"暂停/恢复"消息：节拍令牌本来就由父进程发，(App.exit)
;;      阻塞期间没有令牌，子进程自然停笔；"留下"之后补一个"重绘令牌"即可。
;; ---------------------------------------------------------------------------

(define btn_w 36)
(define btn_h 16)
(define btn_x (- SW (+ btn_w 4)))
(define btn_y 2)

(define draw-exit-button
  (lambda () {
    (Gfx.fill_rect btn_x btn_y btn_w btn_h 60 62 74)
    (Gfx.draw_rect btn_x btn_y btn_w btn_h 120 124 140)
    (Gfx.draw_text "退出" (+ btn_x 6) (+ btn_y 2) 225 228 235)
    ;; 热区登记给宿主：点按由宿主在触摸边沿判定（与 App 节拍快慢无关）
    (if (Touch.is_available)
        (App.set_exit_button btn_x btn_y btn_w btn_h)
        (App.set_exit_button 0 0 0 0))
  }))

;; 给 4 个子进程广播一个节拍令牌（1 = 普通节拍，2 = 节拍 + 整列重绘）
(define broadcast-ctrl
  (lambda (code) {
    (define c 0)
    (while (< c 4) {
      (System.write (get_item qc c) code 0)
      (set! c (+ c 1))
    })
  }))

(define do-exit
  (lambda () {
    ;; 不再发 vsync → 子进程停笔；父进程阻塞在 (App.exit) 里等用户答复
    (if (App.exit)
        { (App.exit_now) }        ; 确认：宿主本帧末销毁运行时（全部进程一并结束）
        { (broadcast-ctrl 2) })   ; 留下：恢复节拍并让子进程整列重绘（擦掉确认框残影）
  }))

;; ---------------------------------------------------------------------------
;; 六、父进程：统计汇总 + HUD
;; ---------------------------------------------------------------------------

;; 归位一条统计报文（6*值+标签）
(define take-stat
  (lambda (c msg) {
    (define tag (Math.floor (mod msg 7)))
    (define val (Math.floor (/ msg 7)))
    (if (== tag 0) (set_item! st-sr c val) #f)
    (if (== tag 1) (set_item! st-sl c val) #f)
    (if (== tag 2) (set_item! st-rc c val) #f)
    (if (== tag 3) (set_item! st-hp c val) #f)
    (if (== tag 4) (set_item! st-st c val) #f)
    (if (== tag 5) (set_item! st-lo c val) #f)
    (if (== tag 6) (set_item! st-pk c val) #f)
    (set! dirty 1)
  }))

;; 排空 4 条统计队列（非阻塞：队列空立即返回 #undefined）
(define drain-stats
  (lambda () {
    (define c 0)
    (define m 0)
    (while (< c 4) {
      (set! m (System.read (get_item qs c) 0))
      (while (not (undefined? m)) {
        (take-stat c m)
        (set! m (System.read (get_item qs c) 0))
      })
      (set! c (+ c 1))
    })
  }))

;; 一行进程表：P6 发 1234 收 1198 队 2 堆 842 栈 12
(define build-row
  (lambda (c) {
    (define sr (get_item st-sr c))
    (define sl (get_item st-sl c))
    (define t (String.concat "P" (String.atom_to_string (get_item pids c))))
    (set! t (String.concat t (String.concat " 发" (padn (+ sr sl) 5))))
    (set! t (String.concat t (String.concat " 收" (padn (get_item st-rc c) 5))))
    (set! t (String.concat t (String.concat " 队峰"
                   (String.concat (padn (get_item st-pk c) 3)
                                  (String.concat "/" (String.atom_to_string QCAP))))))
    (set! t (String.concat t (String.concat " 堆" (padn (get_item st-hp c) 5))))
    (set! t (String.concat t (String.concat " 栈" (padn (get_item st-st c) 3))))
    t
  }))

(define rebuild-rows
  (lambda () {
    (define c 0)
    (set! tot-msg 0)
    (set! tot-lost 0)
    (while (< c 4) {
      (set_item! row-strings c (build-row c))
      (set! tot-msg (+ tot-msg (+ (get_item st-sr c) (+ (get_item st-sl c) (get_item st-rc c)))))
      (set! tot-lost (+ tot-lost (get_item st-lo c)))
      (set! c (+ c 1))
    })
  }))

(define draw-hud
  (lambda () {
    (Gfx.fill_rect 0 0 SW HUD_H 14 16 22)
    (Gfx.fill_rect 0 (- HUD_H 2) SW 2 34 38 48)   ; 与粒子区的分隔线
    ;; 标题行
    (Gfx.draw_text "多进程 IPC 演示" 6 2 230 234 240)
    (Gfx.draw_text (String.concat "总 " (String.concat (padn tot-msg 6)
                            (String.concat "  丢 " (padn tot-lost 3))))
                   118 2 140 200 225)
    (draw-exit-button)
    ;; 池级内存（System.memstat：VM 工作区 / 用户堆，全进程共享口径）
    (Gfx.draw_text "池" 6 18 170 180 196)
    (Gfx.draw_text "VM" 22 18 170 180 196)
    (define ms (System.memstat))
    (define vmp (pct (get_item ms 1) (get_item ms 0)))
    (define hmp (pct (get_item ms 3) (get_item ms 2)))
    (bar 40 20 88 8 vmp 90 200 255)
    (Gfx.draw_text (String.concat (padn vmp 3) "%") 132 18 190 220 245)
    (Gfx.draw_text "堆" 174 18 170 180 196)
    (bar 190 20 88 8 hmp 255 190 90)
    (Gfx.draw_text (String.concat (padn hmp 3) "%") 282 18 245 210 160)
    ;; 4 行进程表
    (define c 0)
    (while (< c 4) {
      (Gfx.fill_rect 6 (+ 37 (* 15 c)) 6 10 (get_item col-r c) (get_item col-g c) (get_item col-b c))
      (Gfx.draw_text (get_item row-strings c) 16 (+ 34 (* 15 c)) 205 212 225)
      (set! c (+ c 1))
    })
  }))

(define parent-tick
  (lambda () {
    (if (App.back_requested) (do-exit) #f)
    ;; 从后台/挂起恢复：画面在挂起期间归别人，让子进程整列重绘
    (if (App.resumed) (broadcast-ctrl 2) #f)
    (drain-stats)
    (if (== dirty 1) { (set! dirty 0) (rebuild-rows) } #f)
    (draw-hud)
    (Gfx.refresh)
    (broadcast-ctrl 1)            ; 本拍结束：放行子进程各走一步（唯一的节拍源）
  }))

;; ---------------------------------------------------------------------------
;; 七、子进程：收件 → 物理 → 绘制
;; ---------------------------------------------------------------------------

;; 画一个粒子：正常情况下用本列的颜色（颜色 = 归属进程），
;; 距离本列左右边框不足 4px 时画成白色 —— 那是"正要交出去 / 刚被移交进来"的
;; 位置，一眼就能看出跨界的流量（不需要额外状态：纯几何判定）。
(define paint-at
  (lambda (x y) {
    (define d0 (- x (* COLW my-id)))
    (define d1 (- (+ (* COLW my-id) COLW) x))
    (define d (if (< d0 d1) d0 d1))
    (if (< d 4.0)
        (Gfx.fill_rect (Math.floor x) (Math.floor y) 2 2 245 250 255)
        (Gfx.fill_rect (Math.floor x) (Math.floor y) 2 2
                       (get_item col-r my-id) (get_item col-g my-id) (get_item col-b my-id)))
  }))

;; 整列重绘：清列 + 画列分隔线 + 画出全部粒子
;; （首次进入、确认框关闭、后台恢复时调用——每列的重绘只依赖自己）
(define paint-full
  (lambda () {
    (define x0 (* COLW my-id))
    (define i 0)
    (define n 0)
    (Gfx.fill_rect x0 AREA_Y COLW AREA_H 0 0 0)
    (if (> my-id 0) (Gfx.fill_rect x0 AREA_Y 1 AREA_H 36 40 50) #f)
    (set! n (length xs))
    (while (< i n) {
      (paint-at (get_item xs i) (get_item ys i))
      (set! i (+ i 1))
    })
  }))

;; 撒点：本列内均匀随机
(define seed
  (lambda () {
    (define i 0)
    (define x0 (* COLW my-id))
    (while (< i NP) {
      (push xs (+ x0 (* COLW (Math.random))))
      (push ys (+ AREA_Y (* AREA_H (Math.random))))
      (push vxs (* 2.0 (- (Math.random) 0.5)))
      (push vys (* 2.0 (- (Math.random) 0.5)))
      (set! i (+ i 1))
    })
  }))

;; 接管一个跨界的粒子（位置报文 pm 是偶数，速度报文 vm 是奇数）
(define adopt
  (lambda (pm vm) {
    (define p (Math.floor (/ pm 2.0)))
    (define vv (Math.floor (/ (- vm 1.0) 2.0)))
    (define xq (Math.floor (mod p 160.0)))
    (define yq (Math.floor (/ p 160.0)))
    (define vxq (Math.floor (mod vv 32.0)))
    (define vyq (Math.floor (/ vv 32.0)))
    (push xs (+ (* 2.0 xq) 0.5))
    (push ys (+ AREA_Y (+ (* 2.0 yq) 0.5)))
    (push vxs (* 0.25 (- vxq 16.0)))
    (push vys (* 0.25 (- vyq 16.0)))
    (set! rc (bump rc))
  }))

;; 排空一条入边队列：偶数 = 位置、奇数 = 速度，按奇偶自同步配对。
;; pend 是这条队列的"半条记录"槽——每条有向边一条队列、单一生产者，
;; 所以正常情况下位置与速度紧邻；万一生产者被时间片切开（位置已写、速度还没写），
;; 下一个位置报文会覆盖它（丢一颗，计数进 lost），而不会错配成畸形粒子。
(define drain-edge
  (lambda (q pend)
    (set! v (System.read q 0))
    (while (not (undefined? v)) {
      (set! arrived (+ arrived 1))               ; 队列水位实测：数排空时取走几条
      (if (== (mod v 2.0) 0.0)
          (set! pend v)                          ; 位置：先记住
          (if (>= pend 0)
              { (adopt pend v) (set! pend -1) }
              #f))                               ; 孤立的速度报文：丢弃
      (set! v (System.read q 0))
    })
    pend
  ))

;; 两条入边：左邻发来的（它的"右越界"队列）与右邻发来的（它的"左越界"队列）。
;; 排空时数到的报文条数就是队列的**瞬时水位**——它恰好在排空前达到最大，
;; 所以"每次排空数到的最大值"= 队列的峰值水位（不需要任何"查询队列长度"的接口）。
(define drain-in
  (lambda () {
    (set! arrived 0)
    (set! pend-a (drain-edge (get_item qr lid) pend-a))
    (set! pend-b (drain-edge (get_item ql rid) pend-b))
    (if (> arrived peakq) (set! peakq arrived) #f)
  }))

;; 把粒子打包成两条报文交给 tgt 列
(define send-record
  (lambda (tgt x y vx vy) {
    (define xq (Math.floor (* 0.5 x)))
    (define yq (Math.floor (* 0.5 (- y AREA_Y))))
    (define vxq (Math.round (* 4.0 vx)))
    (define vyq (Math.round (* 4.0 vy)))
    (if (< xq 0.0) (set! xq 0.0) #f)
    (if (> xq 159.0) (set! xq 159.0) #f)
    (if (< yq 0.0) (set! yq 0.0) #f)
    (if (> yq 71.0) (set! yq 71.0) #f)
    (if (< vxq -16.0) (set! vxq -16.0) #f)
    (if (> vxq 15.0) (set! vxq 15.0) #f)
    (if (< vyq -16.0) (set! vyq -16.0) #f)
    (if (> vyq 15.0) (set! vyq 15.0) #f)
    (define pm (* 2.0 (+ xq (* 160.0 yq))))
    (define vm (+ (* 2.0 (+ (+ vxq 16.0) (* 32.0 (+ vyq 16.0)))) 1.0))
    (define q (if (== tgt rid) (get_item qr my-id) (get_item ql my-id)))
    (if (System.write q pm 0)
        { (if (System.write q vm 0)
              (if (== tgt rid) (set! sr (bump sr)) (set! sl (bump sl)))
              (set! lost (bump lost))) }        ; 只写进去半条：丢一个，接收方会自同步
        { (set! lost (bump lost)) })            ; 队列满：丢包（这就是"丢"的来源）
  }))

;; 移交：擦掉旧像素 → 发两条报文 → swap-pop 摘除（O(1)，不打乱未处理的元素）
(define transfer
  (lambda (i tgt x y vx vy) {
    (define last 0)
    (Gfx.fill_rect (Math.floor (get_item xs i)) (Math.floor (get_item ys i)) 2 2 0 0 0)
    (send-record tgt x y vx vy)
    (set! last (- (length xs) 1))
    (set_item! xs i (get_item xs last))
    (set_item! ys i (get_item ys last))
    (set_item! vxs i (get_item vxs last))
    (set_item! vys i (get_item vys last))
    (pop xs)
    (pop ys)
    (pop vxs)
    (pop vys)
  }))

;; 一步物理 + 重画（本列内）
(define child-step
  (lambda () {
    (define i 0)
    (define n 0)
    (define x 0.0)
    (define y 0.0)
    (define vx 0.0)
    (define vy 0.0)
    (define x0 (* COLW my-id))
    (define x1 (+ x0 COLW))
    (while #t {
      (set! n (length xs))
      (if (>= i n) break)
      (set! x (get_item xs i))
      (set! y (get_item ys i))
      (set! vx (get_item vxs i))
      (set! vy (get_item vys i))
      ;; 流向场：右向漂移 + 纵向剪切（sin 波浪，速度只随 y 变）
      ;;   —— 刻意不用"向中心吸引/涡旋"之类的场：那类场会把粒子挤到某几列，
      ;;      密度不均就看不出"每列一个进程"的画面。只随 y 变化的速度场
      ;;      不改变横向密度，粒子因此均匀铺满全屏，持续从左向右穿越边界。
      (set! vx (+ DRIFT (* SHEAR (Math.sin (* WAVE y)))))
      (set! vx (+ vx (* JIT (- (Math.random) 0.5))))
      (set! vy (+ (* vy DAMP) (* JIT (- (Math.random) 0.5))))
      (if (> vx VMAX) (set! vx VMAX) #f)
      (if (< vx (- 0.0 VMAX)) (set! vx (- 0.0 VMAX)) #f)
      (if (> vy VMAX) (set! vy VMAX) #f)
      (if (< vy (- 0.0 VMAX)) (set! vy (- 0.0 VMAX)) #f)
      (set! x (+ x vx))
      (set! y (+ y vy))
      ;; 纵向环绕（列内，不跨进程）
      (if (< y AREA_Y) (set! y (+ y AREA_H)) #f)
      (if (>= y AREA_END) (set! y (- y AREA_H)) #f)
      ;; 横向越界 → 移交给邻列。只有**屏幕侧边**才做环面绕回（±SW），
      ;; 列与列之间的边界上坐标本来就是对的——不然收方会拿到越界的 x，
      ;; 粒子会在两条列之间来回弹（实测会制造 6 倍无效流量并把队列灌满）。
      (if (< x x0)
          { (if (== my-id 0) (set! x (+ x SW)) #f) (transfer i lid x y vx vy) }
          { (if (>= x x1)
                { (if (== my-id 3) (set! x (- x SW)) #f) (transfer i rid x y vx vy) }
                {
                  ;; 本列内：擦旧 → 更新 → 画新（增量重绘：不整列清屏，避免撕裂）
                  (Gfx.fill_rect (Math.floor (get_item xs i)) (Math.floor (get_item ys i)) 2 2 0 0 0)
                  (set_item! xs i x)
                  (set_item! ys i y)
                  (set_item! vxs i vx)
                  (set_item! vys i vy)
                  (paint-at x y)
                  (set! i (+ i 1))
                }) })
    })
  }))

;; 上报本进程的统计（7 条单值报文：7*值+标签，标签在低位，丢一条也不错位）
(define send-stats
  (lambda () {
    (define q (get_item qs my-id))
    (System.write q (+ (* 7 sr) 0) 0)
    (System.write q (+ (* 7 sl) 1) 0)
    (System.write q (+ (* 7 rc) 2) 0)
    (System.write q (+ (* 7 (Diag.heap)) 3) 0)
    (System.write q (+ (* 7 (+ (Diag.opstack) (Diag.fstack))) 4) 0)
    (System.write q (+ (* 7 lost) 5) 0)
    (System.write q (+ (* 7 peakq) 6) 0)
  }))

;; 等父进程的 vsync 令牌（阻塞在这里 = 不占时间片）。
;; 令牌自带两种含义：
;;   1 = 普通节拍：走一步物理并重画；
;;   2 = 重绘节拍：整列重绘（模态框关闭 / 后台恢复之后，先把残影擦干净）。
;; 为什么由父进程发令牌而不是各子进程自己定时：
;;   ① 帧率由**唯一刷新者**决定，5 个进程的节拍天然对齐，画面不抖；
;;   ② 退出确认框期间父进程阻塞在这条路径上、不再发令牌 → 子进程自动停笔，
;;      不需要额外的"暂停/恢复"协议；
;;   ③ 子进程被阻塞唤醒的时机只取决于一条队列，不会被定时器实现的细节左右。
(define slot-tok 0)                ; 本拍：1 = 有节拍令牌，0 = 没有（超时兜底醒来）
(define slot-repaint 0)            ; 本拍：1 = 需要整列重绘
(define wait-vsync
  (lambda () {
    (set! slot-tok 0)
    (set! slot-repaint 0)
    (define m (System.read (get_item qc my-id) VSYNC_FALLBACK_MS))
    (while (not (undefined? m)) {
      (set! slot-tok 1)
      (if (== m 2) (set! slot-repaint 1) #f)
      (set! m (System.read (get_item qc my-id) 0))
    })
  }))

;; 子进程主循环：阻塞等父进程的节拍令牌 → 收件 → 走一步物理并重画自己那一列。
;; 阻塞态不参与调度，所以 4 个子进程平时几乎不占时间片。
(define child-loop
  (lambda () {
    (while #t {
      (wait-vsync)                                       ; 阻塞等父进程的节拍
      ;; 不管有没有节拍，都先把粒子收进自己的列表——父进程在弹确认框时虽然
      ;; 不发令牌，跨界流量照旧，不收就会把队列填满开始丢粒子
      (drain-in)
      (if (== slot-tok 1)
          {
            (if (== slot-repaint 1)
                (paint-full)                             ; 模态框/后台恢复：整列重绘
                {
                  (child-step)
                  (set! tickn (+ tickn 1))
                  (if (>= tickn STAT_EVERY) { (set! tickn 0) (send-stats) } #f)
                })
          }
          #f)
    })
  }))

(define child-entry
  (lambda () {
    (set! lid (left-of my-id))
    (set! rid (right-of my-id))
    (set! xs (newlist))
    (set! ys (newlist))
    (set! vxs (newlist))
    (set! vys (newlist))
    (seed)
    (paint-full)
    (child-loop)
  }))

;; ---------------------------------------------------------------------------
;; 八、父进程入口
;; ---------------------------------------------------------------------------

(define parent-entry
  (lambda () {
    (draw-hud)
    (Gfx.refresh)
    (System.set_interval TICK_MS parent-tick)
  }))

;; ---------------------------------------------------------------------------
;; 九、启动：建队列 → fork ×4 → 各就各位
;; ---------------------------------------------------------------------------

(define boot
  (lambda () {
    (define i 0)
    (define pid 0)

    ;; ① 队列必须在 fork 之前建好：队列编号是虚拟机全局资源，子进程共享
    (set! qr (newlist))
    (set! ql (newlist))
    (set! qs (newlist))
    (set! qc (newlist))
    (set! st-sr (newlist))
    (set! st-sl (newlist))
    (set! st-rc (newlist))
    (set! st-hp (newlist))
    (set! st-st (newlist))
    (set! st-lo (newlist))
    (set! row-strings (newlist))
    (while (< i 4) {
      (push qr (System.make_queue QCAP)) ; c → 右邻
      (push ql (System.make_queue QCAP)) ; c → 左邻
      (push qs (System.make_queue 48))   ; 统计上报
      (push qc (System.make_queue 8))    ; 控制
      (push st-sr 0)
      (push st-sl 0)
      (push st-rc 0)
      (push st-hp 0)
      (push st-st 0)
      (push st-pk 0)
      (push st-lo 0)
      (push row-strings "")
      (set! i (+ i 1))
    })

    ;; ② fork 出 4 个子进程：父进程拿到子 PID，子进程拿到 0
    (set! i 0)
    (while (< i 4) {
      (set! pid (System.fork))
      (if (== pid 0)
          {
            ;; 子进程：认领第 i 列，然后进主循环（永不返回）
            (set! role 1)
            (set! my-id i)
            (child-entry)
            break
          }
          { (push pids pid) })
      (set! i (+ i 1))
    })

    ;; ③ 父进程：起 HUD 定时器
    (if (== role 0) (parent-entry) #f)
  }))

;; 各列配色（粒子颜色 = 归属进程）
(define col-r '(90 255 120 255))
(define col-g '(215 200 250 120))
(define col-b '(255 90 140 200))

(boot)