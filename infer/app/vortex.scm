;@id=vortex
;@title=卡门涡街
;@version=2026-11-26
;; ============================================================================
;; 卡门涡街（vortex）—— 灵机 App：圆柱绕流的**真格子玻尔兹曼（LBM D2Q9）模拟**
;; + 风洞烟线（streakline）可视化。涡街不是画出来的，是流场自己失稳长出来的。
;;
;; 物理（每一步都在 Scheme 里算）：
;;   · D2Q9 单松弛（BGK）：每格 rho=Σf_i、u=Σc_i f_i/rho、
;;     feq_i = w_i·rho·(1 + 3c_i·u + 4.5(c_i·u)² − 1.5|u|²)、f_i ← f_i + (feq_i − f_i)/τ；
;;     用 pull 式平移（读邻居偏移、写本格、双缓冲乒乓），一格一遍、无迭代。
;;   · 障碍物：格点标记固体；贴壁格**半步反弹**（bounce-back）给无滑移边界 ——
;;     必须用**碰撞后**的分布（B 缓冲）回弹，用碰撞前的值会往流场里注入能量、几步就发散。
;;     障碍物形状/大小/位置可**触屏拖动 + 底部按钮调整**，改动时就地重建固体掩码
;;     （状态翻转的格子把分布重置为静止平衡，避免拖出"爆炸格"）。
;;   · 入口定速（可加短暂横向"阵风"给不稳定性播种）；出口零梯度复制；
;;     上下**自由滑移**（边界格写 feq(rho=1, u=(ux_local, 0))）。
;;   · 边界格的速度场每步补全（上/下壁复制相邻内行、入口写 U0、出口零梯度复制）：
;;     **这是"上下两端边缘不对"的根因修复** —— 以前边界格的 UX/UY 永远是 0，涡量场的
;;     中心差分取到它们，于是上下各出现一条假的高涡量色带（烟粒双线性采样在底边也会
;;     读到 0 速度）。
;;   · 参数（底部按钮可调）：入口流速 U0、松弛时间 τ（= 0.5+3ν）、障碍物形状与大小。
;;     Re = U·D·3/(τ−0.5)；默认 U0=0.18、τ=0.52、D=4 格 ⇒ **Re≈108**（卡门涡街的经典区间）。
;;
;; 网格与布局（**铺满整屏**）：32×24 格 × 10px = 320×240，一格像素对应一个格点；
;;   底部 26px 是**浮在流场之上**的半透明控件栏（流场仍在它后面继续算），
;;   左上角一条 HUD 显示当前 U0 / Re / 形状 / 半径。
;;
;; 可视化（两种显示模式，底部「烟丝/矢量」按钮切换）：
;;   · **自由烟丝**（默认）：260 颗烟粒在速度场里双线性插值输运（每帧推进 NSUB 步），
;;     每颗画一条**沿速度反推 6 帧**（≈10px）的短尾迹，亮度随速度（曼哈顿近似）——
;;     数量与形态与改版前完全一致。涡量底图关时是暗底 + 浅蓝白发光小尾迹；
;;     底图开时同一形状改画**墨色**（浅底上加色发光等于看不见）。
;;   · **固定格点速度矢量场**（quiver）：每 2 格一枚箭（16×12=192），方向/长度 = 当地
;;     速度（|u|·58px 钳到 15px，箭头头部随杆长收小）。与烟丝互补：烟粒随流体走、
;;     看得见输运与涡的卷起；矢量场钉在格点上、位置不漂移，看得见瞬时流向与快慢。
;;   · 涡量底图（默认关，点「涡量」按钮开）：**RdBu 发散色标**（ColorBrewer 11 级，
;;     流体/CFD 文献里最常见的涡量场配色：负涡量=蓝、零=近白、正涡量=红），按 65 档
;;     预插值的查找表直接查色（每格一次 fill_rect；`get_item` 的浮点索引四舍五入正好
;;     当查表用），并加 `|t|<0.12` 死区压掉全场淡彩薄雾。
;;
;; 交互：
;;   · 手指按在障碍物上拖动 = 移动障碍物（松手即停在原处；中心夹在流场内部）。
;;     拖动时中心每移动 0.3 格才重建一次固体掩码（重建 ≈3 ms）。
;;   · 底部控件栏 9 枚按钮（通用控件 app/lib/button.scm，32×20）：速−/速+（入口流速）、
;;     粘−/粘+（松弛时间 τ ⇒ 粘性 ⇒ Re）、径−/径+（障碍物大小）、形（圆/椭圆/方/板）、
;;     涡量（底图开关，常亮 = 开）、烟丝/矢量（显示模式轮转，标签即当前模式）。
;;     按钮**吞掉**落在自己身上的点按边沿，所以按下按钮不会被当成一次拖动。
;;   · 右上角「退出」（app/lib/exit.scm）。
;;
;; 成本（WASM 目标，headless Chromium 实测，**整拍口径**）：烟丝模式（底图关）
;;   **19.5 ms/拍、28 fps**；底图开时 +涡量场绘制与整屏推帧，约 25 ms/拍、同样 28 fps。
;;   （定时器 33 ms；相位计时不可信，见下面诊断行的说明。）
;;   为什么每帧只跑 1 步：格子数从 26×18 涨到 32×24（铺满整屏）后 2 步就要 27 ms，
;;   加烟线与底图必超 33 ms 的定时器周期（超过就掉到 15 fps 档）。1 步/帧时把默认
;;   U0 提到 0.18，涡对脱落周期 ≈ D/(St·U) ≈ 140 步，仍是"看得见的卡门涡街"。
;;   解释器里能跑动靠三件事：
;;     ① 每格只做一遍 + 无迭代（LBM 天生如此）；
;;     ② 贴壁反弹只在"贴壁格清单"上做，且清单里**预存 8 个邻居的格号**（省掉每方向
;;        一次 Math.floor）；贴壁格清单只在障碍物变化时重建；
;;     ③ 烟粒速度模长用曼哈顿近似（省 Math.sqrt），涡量色表用 get_item 的
;;        "浮点索引四舍五入"直接查表（省一次 Math.floor）。
;;
;; 诊断行（每 30 拍）：[vx] ms_lbm= ms_dye= ms_frm= re= uy=<探针> c=<障碍物中心> u= tau=
;;   rad= sh= vor= n=（参数与探针都在里面，便于端到端断言）。
;;   **计时口径注意**：ms_lbm / ms_dye 是把一拍切成几段用 (Diag.ms) 夹出来的，而 VM 的
;;   时间片抢占可以落在任意指令边界上（被抢占期间宿主在跑别的），于是某一相位可能
;;   "吃掉"抢占空档 —— 实测同一份代码在 NPD=260 与 NPD=26 下 ms_dye 都是 9~11 ms（不随
;;   粒子数线性变化），就是被抢占污染的。**只有 ms_frm（整拍）是可信的**，相位数据只作
;;   相对参考，不能拿它判断某个循环快慢。
;;
;; 解释器约束：算术**恒二元**；无 list/let/map；局部量在 lambda 体首部声明、循环里 set!；
;; 只有 #f 是假；cond 子句只容一个表达式；(cons 0 '()) 的哑元素要 pop。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native Diag)
(native System)
(native Touch)

(import EX "app/lib/exit.scm")
(import BTN "app/lib/button.scm")

;; ---- 网格与布局：32×24 格 × 10px = **整屏 320×240** -----------------------
(define NX 32)                     ; 内部列
(define NY 24)                     ; 内部行
(define CELL 10)                   ; 每格像素（32*10=320, 24*10=240）
(define W (+ NX 2))                ; 加一圈边界格（免越界判定）
(define H (+ NY 2))
(define NC (* W H))
(define NSUB 1)                    ; 每帧跑几个 LBM 步（见文件头"成本"）
(define TAILF 6.0)                 ; 烟线彗尾 = 沿速度反推几帧（与改版前观感一致）

;; 底部控件栏（浮在流场之上）
(define BAR_Y 214)
(define BAR_H 26)
(define BTN_Y 217)
(define BTN_W 32)
(define BTN_H 20)

;; ---- 物理参数（底部按钮可调）--------------------------------------------
(define U0 0.18)                   ; 入口流速（格子单位；Ma≈0.31，安全）
(define TAU 0.52)                  ; 松弛时间 τ = 0.5 + 3ν
(define OMEGA 1.923076923)         ; = 1/τ（upd_phys 重算）
(define OM1 -0.923076923)          ; = 1−ω
(define RAD 2.0)                   ; 障碍物特征半径（格）→ 圆直径 D=4 格 = 40 px
(define SHAPE 0)                   ; 0=圆 1=椭圆 2=方 3=板
(define CXC 9.0)                   ; 障碍物中心（格子坐标，可为小数）
(define CYC 12.0)
(define PXC 81.0)                  ; 中心像素坐标（upd_shape 缓存）
(define PYC 111.0)
(define ORX 2.0)                   ; 障碍物半宽/半高（格，upd_shape 缓存）
(define ORY 2.0)
(define IRX 0.05)                  ; 1/(ORX*CELL)——烟粒碰撞判定用
(define IRY 0.05)
(define VS 0.135)                  ; 涡量归一化尺度（= 0.65*U0，upd_phys 重算）

;; 九个方向的权、源格偏移（c 的偏移 ×9）、反方向下标
(define WT0 0.4444444)
(define WT 0.1111111)
(define WTD 0.0277778)             ; 1/36
(define SO1 0) (define SO2 0) (define SO3 0) (define SO4 0)
(define SO5 0) (define SO6 0) (define SO7 0) (define SO8 0)

;; ---- 状态 ----------------------------------------------------------------
(define A '())                     ; 分布函数缓冲（NC*9）
(define B '())
(define SOLID '())                 ; 每格 1 = 固体（NC）
(define UX '())                    ; 速度场（NC）
(define UY '())
(define NEAR '())                  ; 贴壁格扁平表：每格 9 项 (格号 8 个邻居格号)

(define NPD 260)                   ; 烟粒数（与改版前一致）
(define PX '())
(define PY '())

(define SHOWVORT 0)                ; 涡量底图：默认关（点亮「涡量」按钮看 RdBu 场）
(define MODE 0)                    ; 显示模式：0=自由烟丝（烟粒） 1=固定格点速度矢量场
(define STEPC 0)                   ; LBM 步数
(define GS 0)                      ; 入口阵风计数器（参数改动后重新播种）
(define GYV 0.0)                   ; 当前入口横向速度（阵风，set_bounds 写、边界补全读）
(define FRAMES 0)
(define ACCL 0.0)
(define ACCD 0.0)
(define ACCF 0.0)
(define MSL 0.0)                   ; 最近一次 LBM 单步耗时
(define MSD 0.0)                   ; 最近一次烟线耗时

;; 拖动
(define PREV_DN 0)                 ; 上一拍触屏电平
(define DRAG 0)                    ; 1 = 正在拖障碍物
(define GRAB_DX 0.0)               ; 抓取点相对中心的偏移（格）
(define GRAB_DY 0.0)

;; HUD
(define HUDD 1)                    ; 1 = HUD 文本需重建
(define HUD_STR "U0.18 Re108 圆 r2")
(define SHNAME '("圆" "椭圆" "方" "板"))
(define MDNAME '("烟丝" "矢量"))   ; 显示模式按钮标签：自由烟丝 / 固定格点速度场

;; ---- 涡量色标：RdBu（ColorBrewer 11 级发散色标）→ 65 档查找表 ------------
(define RC '(5 33 67 146 209 247 253 244 214 178 103))
(define RG '(48 102 147 197 229 247 219 165 96 24 0))
(define RB '(97 172 195 222 240 247 199 130 77 43 31))
(define VN 65)
(define VR '())
(define VG '())
(define VB '())

;; ---- 控件栏的 8 枚按钮 ---------------------------------------------------
(define BTNS '())

;; ---- 启动：建网格 --------------------------------------------------------
(define build_grid
  (lambda ()
    (define i 0)
    (define cn 0)
    ;; 偏移：源格 = c + SOi（off_i = (cy*W + cx)*9）
    (set! SO1 (- 1 (* 9 1)))
    (set! SO2 (- 2 (* 9 W)))
    (set! SO3 (+ 3 (* 9 1)))
    (set! SO4 (+ 4 (* 9 W)))
    (set! SO5 (- (- 5 (* 9 W)) 9))
    (set! SO6 (+ (- 6 (* 9 W)) 9))
    (set! SO7 (+ (+ 7 (* 9 W)) 9))
    (set! SO8 (- (+ 8 (* 9 W)) 9))
    ;; 两个分布缓冲：初始 = 静止平衡（f_i = w_i，rho=1, u=0）
    (set! A (cons 0.0 '())) (pop A)
    (set! B (cons 0.0 '())) (pop B)
    (set! i 0)
    (while (< i NC) {
      (push A WT0) (push A WT) (push A WT) (push A WT) (push A WT)
      (push A WTD) (push A WTD) (push A WTD) (push A WTD)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i NC) {
      (push B WT0) (push B WT) (push B WT) (push B WT) (push B WT)
      (push B WTD) (push B WTD) (push B WTD) (push B WTD)
      (set! i (+ i 1))
    })
    ;; 固体掩码（先全 0，真正的形状由 rebuild_obs 填）+ 速度场
    (set! SOLID (cons 0 '())) (pop SOLID)
    (set! UX (cons 0.0 '())) (pop UX)
    (set! UY (cons 0.0 '())) (pop UY)
    (set! cn 0)
    (while (< cn NC) {
      (push SOLID 0)
      (push UX 0.0)
      (push UY 0.0)
      (set! cn (+ cn 1))
    })))

;; ---- 障碍物：形状判定 ----------------------------------------------------
;; 半宽/半高（格）与像素中心缓存。SHAPE 或 RAD 改动后必须调一次。
(define upd_shape
  (lambda ()
    (if (== SHAPE 0) { (set! ORX RAD) (set! ORY RAD) } #f)
    (if (== SHAPE 1) { (set! ORX (* RAD 0.62)) (set! ORY (* RAD 1.5)) } #f)
    (if (== SHAPE 2) { (set! ORX (* RAD 0.88)) (set! ORY (* RAD 0.88)) } #f)
    (if (== SHAPE 3) { (set! ORX 0.62) (set! ORY (* RAD 1.5)) } #f)
    (set! IRX (/ 1.0 (* ORX CELL)))
    (set! IRY (/ 1.0 (* ORY CELL)))
    (set! PXC (* (- CXC 0.5) CELL))
    (set! PYC (* (- CYC 0.5) CELL))
    (set! HUDD 1)))

;; 格点 (dx,dy)（相对中心，单位：格）是否落在障碍物内
(define solid_at
  (lambda (dx dy)
    (define u 0.0)
    (define v 0.0)
    (if (>= SHAPE 2)
        (if (and (< (Math.abs dx) ORX) (< (Math.abs dy) ORY)) 1 0)
        {
          ;; dx/dy 以**格**为单位，故这里除 ORX/ORY；IRX/IRY 是给像素坐标的烟粒用的
          (set! u (/ dx ORX))
          (set! v (/ dy ORY))
          (if (< (+ (* u u) (* v v)) 1.0) 1 0)
        })))

;; 把某格的两个分布缓冲重置为静止平衡（障碍物扫过时用，防"爆炸格"）
(define reset_cell
  (lambda (cn)
    (define c (* cn 9))
    (set_item! A c WT0) (set_item! A (+ c 1) WT) (set_item! A (+ c 2) WT)
    (set_item! A (+ c 3) WT) (set_item! A (+ c 4) WT)
    (set_item! A (+ c 5) WTD) (set_item! A (+ c 6) WTD)
    (set_item! A (+ c 7) WTD) (set_item! A (+ c 8) WTD)
    (set_item! B c WT0) (set_item! B (+ c 1) WT) (set_item! B (+ c 2) WT)
    (set_item! B (+ c 3) WT) (set_item! B (+ c 4) WT)
    (set_item! B (+ c 5) WTD) (set_item! B (+ c 6) WTD)
    (set_item! B (+ c 7) WTD) (set_item! B (+ c 8) WTD)))

;; 重建固体掩码 + 贴壁格表（障碍物移动/换形状/改大小时调；启动时也调一次）
(define rebuild_obs
  (lambda ()
    (define y 1)
    (define x 1)
    (define cn 0)
    (define dx 0.0)
    (define dy 0.0)
    (define s 0)
    (define old 0)
    ;; ① 掩码：就地改 SOLID；状态翻转的格子重置分布
    (set! y 1)
    (while (< y (+ NY 1)) {
      (set! x 1)
      (set! cn (+ (* y W) 1))
      (while (< x (+ NX 1)) {
        (set! dx (- x CXC))
        (set! dy (- y CYC))
        (set! s (solid_at dx dy))
        (set! old (get_item SOLID cn))
        (if (not (== s old))
            {
              (set_item! SOLID cn s)
              (reset_cell cn)
            }
            #f)
        (set! cn (+ cn 1))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
    ;; ② 贴壁格表：流体格且 4 邻域里有固体。每格存 9 个数：
    ;;    [格号, 8 个方向的邻居格号]（反弹时省掉每方向一次除法/取整）
    (set! NEAR (cons 0 '())) (pop NEAR)
    (set! y 1)
    (while (< y (+ NY 1)) {
      (set! x 1)
      (set! cn (+ (* y W) 1))
      (while (< x (+ NX 1)) {
        (if (== (get_item SOLID cn) 0)
            (if (or (== (get_item SOLID (- cn 1)) 1) (== (get_item SOLID (+ cn 1)) 1)
                    (== (get_item SOLID (- cn W)) 1) (== (get_item SOLID (+ cn W)) 1))
                {
                  (push NEAR cn)
                  (push NEAR (+ cn 1))
                  (push NEAR (- cn W))
                  (push NEAR (- cn 1))
                  (push NEAR (+ cn W))
                  (push NEAR (+ (- cn W) 1))
                  (push NEAR (- (- cn W) 1))
                  (push NEAR (- (+ cn W) 1))
                  (push NEAR (+ (+ cn W) 1))
                }
                #f)
            #f)
        (set! cn (+ cn 1))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })))

;; ---- 烟粒 ----------------------------------------------------------------
;; 烟粒回注：入口左侧窄带（x 2~8）+ 全高随机 y —— 与改版前一致（见 dye_step 内联）。
(define spawn_dye
  (lambda ()
    (set! PX (cons 0.0 '())) (pop PX)
    (set! PY (cons 0.0 '())) (pop PY)
    (define i 0)
    (set! i 0)
    (while (< i NPD) {
      (push PX (+ 2.0 (* (Math.random) 6.0)))
      (push PY (+ 6.0 (* (Math.random) 228.0)))
      (set! i (+ i 1))
    })))

;; ---- 边界条件：写进**目标缓冲** B（下一步它会成为 A 供 pull 读取）--------
(define set_bounds
  (lambda ()
    (define y 0)
    (define x 0)
    (define c 0)
    (define cb 0)
    (define bb 0.0)
    (define cu 0.0)
    (define cv 0.0)
    (define GY 0.0)
    (define q1 0.0)
    (define r1 0.0)
    (define r2 0.0)
    (define uu 0.0)
    (define e0 0.0) (define e1 0.0) (define e2 0.0) (define e3 0.0) (define e4 0.0)
    (define e5 0.0) (define e6 0.0) (define e7 0.0) (define e8 0.0)
    ;; 入口平衡态：rho=1, u=(U0, GY)
    (set! GY 0.0)
    (if (< GS 600) (set! GY (* 0.03 (Math.sin (* 0.15 GS)))) #f)
    (set! GYV GY)
    (set! bb (- 1.0 (* 1.5 (+ (* U0 U0) (* GY GY)))))
    (set! cu (* 3.0 U0))
    (set! cv (* 3.0 GY))
    (set! q1 (* 4.5 (+ (* U0 U0) (* GY GY))))
    (set! e0 (* WT0 bb))
    (set! e1 (* WT (+ (+ bb cu) q1)))
    (set! e2 (* WT (+ (+ bb cv) q1)))
    (set! e3 (* WT (+ (+ bb (- 0.0 cu)) q1)))
    (set! e4 (* WT (+ (+ bb (- 0.0 cv)) q1)))
    (set! e5 (* WTD (+ (+ bb (+ cu cv)) q1)))
    (set! e6 (* WTD (+ (+ bb (+ (- 0.0 cu) cv)) q1)))
    (set! e7 (* WTD (+ (+ bb (- 0.0 (+ cu cv))) q1)))
    (set! e8 (* WTD (+ (+ bb (- cu cv)) q1)))
    (set! y 1)
    (while (< y (+ NY 1)) {
      ;; 入口列 x=0
      (set! c (* 9 (* y W)))
      (set_item! B c e0) (set_item! B (+ c 1) e1) (set_item! B (+ c 2) e2)
      (set_item! B (+ c 3) e3) (set_item! B (+ c 4) e4) (set_item! B (+ c 5) e5)
      (set_item! B (+ c 6) e6) (set_item! B (+ c 7) e7) (set_item! B (+ c 8) e8)
      ;; 出口列 x=NX+1：复制 x=NX（零梯度）
      (set! cb (* 9 (+ (* y W) (+ NX 1))))
      (set! c (* 9 (+ (* y W) NX)))
      (set_item! B cb (get_item B c))
      (set_item! B (+ cb 1) (get_item B (+ c 1)))
      (set_item! B (+ cb 2) (get_item B (+ c 2)))
      (set_item! B (+ cb 3) (get_item B (+ c 3)))
      (set_item! B (+ cb 4) (get_item B (+ c 4)))
      (set_item! B (+ cb 5) (get_item B (+ c 5)))
      (set_item! B (+ cb 6) (get_item B (+ c 6)))
      (set_item! B (+ cb 7) (get_item B (+ c 7)))
      (set_item! B (+ cb 8) (get_item B (+ c 8)))
      (set! y (+ y 1))
    })
    ;; 上下边界：**自由滑移**——取相邻内行的切向速度 ux，边界格写 feq(rho=1, u=(ux,0))。
    ;; 比"写死入口速度"更贴物理（壁面不产生剪切层），也比"复制整行分布"稳（不会反射声波）。
    (set! x 0)
    (while (< x W) {
      ;; 上边界 y=0：读 y=1 求 rho/ux
      (set! c (* 9 (+ W x)))
      (set! r1 (+ (get_item B c) (get_item B (+ c 1))))
      (set! r2 (+ (get_item B (+ c 3)) (get_item B (+ c 5))))
      (set! r1 (+ (+ r1 r2) (get_item B (+ c 2))))
      (set! r1 (+ (+ r1 (get_item B (+ c 4))) (get_item B (+ c 6))))
      (set! r1 (+ (+ r1 (get_item B (+ c 7))) (get_item B (+ c 8))))
      (set! uu (/ (- (get_item B (+ c 1)) (get_item B (+ c 3))) r1))
      (set! bb (- 1.0 (* 1.5 (* uu uu))))
      (set! cu (* 3.0 uu))
      (set! q1 (* 4.5 (* uu uu)))
      (set! e0 (* WT0 bb))
      (set! e1 (* WT (+ (+ bb cu) q1)))
      (set! e2 (* WT bb))
      (set! e3 (* WT (+ (+ bb (- 0.0 cu)) q1)))
      (set! e4 (* WT bb))
      (set! e5 (* WTD (+ (+ bb cu) q1)))
      (set! e6 (* WTD (+ (+ bb (- 0.0 cu)) q1)))
      (set! e7 (* WTD (+ (+ bb (- 0.0 cu)) q1)))
      (set! e8 (* WTD (+ (+ bb cu) q1)))
      (set! cb (* 9 x))
      (set_item! B cb e0) (set_item! B (+ cb 1) e1) (set_item! B (+ cb 2) e2)
      (set_item! B (+ cb 3) e3) (set_item! B (+ cb 4) e4) (set_item! B (+ cb 5) e5)
      (set_item! B (+ cb 6) e6) (set_item! B (+ cb 7) e7) (set_item! B (+ cb 8) e8)
      ;; 下边界 y=NY+1：读 y=NY
      (set! c (* 9 (+ (* W NY) x)))
      (set! r1 (+ (get_item B c) (get_item B (+ c 1))))
      (set! r2 (+ (get_item B (+ c 3)) (get_item B (+ c 5))))
      (set! r1 (+ (+ r1 r2) (get_item B (+ c 2))))
      (set! r1 (+ (+ r1 (get_item B (+ c 4))) (get_item B (+ c 6))))
      (set! r1 (+ (+ r1 (get_item B (+ c 7))) (get_item B (+ c 8))))
      (set! uu (/ (- (get_item B (+ c 1)) (get_item B (+ c 3))) r1))
      (set! bb (- 1.0 (* 1.5 (* uu uu))))
      (set! cu (* 3.0 uu))
      (set! q1 (* 4.5 (* uu uu)))
      (set! e0 (* WT0 bb))
      (set! e1 (* WT (+ (+ bb cu) q1)))
      (set! e2 (* WT bb))
      (set! e3 (* WT (+ (+ bb (- 0.0 cu)) q1)))
      (set! e4 (* WT bb))
      (set! e5 (* WTD (+ (+ bb cu) q1)))
      (set! e6 (* WTD (+ (+ bb (- 0.0 cu)) q1)))
      (set! e7 (* WTD (+ (+ bb (- 0.0 cu)) q1)))
      (set! e8 (* WTD (+ (+ bb cu) q1)))
      (set! cb (* 9 (+ (* W (+ NY 1)) x)))
      (set_item! B cb e0) (set_item! B (+ cb 1) e1) (set_item! B (+ cb 2) e2)
      (set_item! B (+ cb 3) e3) (set_item! B (+ cb 4) e4) (set_item! B (+ cb 5) e5)
      (set_item! B (+ cb 6) e6) (set_item! B (+ cb 7) e7) (set_item! B (+ cb 8) e8)
      (set! x (+ x 1))
    })))

;; ---- 边界格的速度场补全（涡量场与烟粒采样都要用它）---------------------
;; 为什么必须有：内部格在 lbm_step 里写 UX/UY，边界格以前**从来没写过**（恒 0），
;; 于是涡量的中心差分在上下第一行取到 0，画出两条假的高涡量带（"边缘不对"的真因）；
;; 烟粒的双线性采样在 j0=NY 时也会读到 y=NY+1 行的 0 速度。
(define fill_bounds_uv
  (lambda ()
    (define x 0)
    (define y 1)
    (set! x 0)
    (while (< x W) {
      ;; 上/下壁：复制相邻内行的切向速度，法向为 0（自由滑移）
      (set_item! UX x (get_item UX (+ W x)))
      (set_item! UY x 0.0)
      (set_item! UX (+ (* (+ NY 1) W) x) (get_item UX (+ (* NY W) x)))
      (set_item! UY (+ (* (+ NY 1) W) x) 0.0)
      (set! x (+ x 1))
    })
    (set! y 1)
    (while (< y (+ NY 1)) {
      (set_item! UX (* y W) U0)
      (set_item! UY (* y W) GYV)
      (set_item! UX (+ (* y W) (+ NX 1)) (get_item UX (+ (* y W) NX)))
      (set_item! UY (+ (* y W) (+ NX 1)) (get_item UY (+ (* y W) NX)))
      (set! y (+ y 1))
    })))

;; ---- LBM 单步：pull + 碰撞 + 写目标缓冲 ---------------------------------
;; 每格：9 次读（按方向偏移，来自 A）→ 守恒量 → 碰撞 → 9 次写（到 B）。
;; 贴壁格子先按普通格子算，随后由反弹把指向固体的方向覆写成反弹值。
(define lbm_step
  (lambda ()
    (define c 0)
    (define cn 0)
    (define y 1)
    (define x 1)
    (define om 0.0)
    (define om1 0.0)
    (define f0 0.0) (define f1 0.0) (define f2 0.0) (define f3 0.0) (define f4 0.0)
    (define f5 0.0) (define f6 0.0) (define f7 0.0) (define f8 0.0)
    (define rho 0.0) (define ux 0.0) (define uy 0.0)
    (define u2 0.0) (define bb 0.0) (define q1 0.0) (define q2 0.0)
    (define s1 0.0) (define s2 0.0) (define uv 0.0)
    (define p1 0.0) (define p2 0.0) (define m1 0.0) (define m2 0.0)
    (define d1 0.0) (define d2 0.0) (define d3 0.0) (define d4 0.0)
    (define kr 0.0) (define k0 0.0) (define k1 0.0) (define kd 0.0)
    (define nb 0) (define base 0)
    (set! om OMEGA)
    (set! om1 OM1)
    (set! y 1)
    (while (< y (+ NY 1)) {
      (set! c (* 9 (+ (* y W) 1)))
      (set! cn (+ (* y W) 1))
      (set! x 1)
      (while (< x (+ NX 1)) {
        (if (== (get_item SOLID cn) 0)
            {
              ;; ---- 读取九个方向的分布（来自邻居 = pull）----
              (set! f0 (get_item A c))
              (set! f1 (get_item A (+ c SO1)))
              (set! f2 (get_item A (+ c SO2)))
              (set! f3 (get_item A (+ c SO3)))
              (set! f4 (get_item A (+ c SO4)))
              (set! f5 (get_item A (+ c SO5)))
              (set! f6 (get_item A (+ c SO6)))
              (set! f7 (get_item A (+ c SO7)))
              (set! f8 (get_item A (+ c SO8)))
              ;; ---- 守恒量 ----
              (set! rho (+ (+ (+ (+ f0 f1) (+ f2 f3)) (+ (+ f4 f5) (+ f6 f7))) f8))
              (set! s1 (+ (- f1 f3) (- f5 f6)))
              (set! ux (/ (+ s1 (- f8 f7)) rho))
              (set! s2 (+ (- f2 f4) (+ f5 f6)))
              (set! uy (/ (- s2 (+ f7 f8)) rho))
              (set_item! UX cn ux)
              (set_item! UY cn uy)
              ;; ---- 平衡态（把 4.5(c·u)² − 1.5|u|² 按方向展开）----
              (set! u2 (+ (* ux ux) (* uy uy)))
              (set! bb (- 1.0 (* 1.5 u2)))
              (set! q1 (* 4.5 (* ux ux)))
              (set! q2 (* 4.5 (* uy uy)))
              (set! uv (* 9.0 (* ux uy)))
              (set! s1 (* 3.0 ux))
              (set! s2 (* 3.0 uy))
              (set! p1 (+ (+ bb s1) q1))
              (set! m1 (+ (- bb s1) q1))
              (set! p2 (+ (+ bb s2) q2))
              (set! m2 (+ (- bb s2) q2))
              (set! d1 (+ (+ (+ bb (+ s1 s2)) (+ q1 q2)) uv))
              (set! d2 (+ (+ (+ bb (- s2 s1)) (+ q1 q2)) (- 0.0 uv)))
              (set! d3 (+ (+ (- bb (+ s1 s2)) (+ q1 q2)) uv))
              (set! d4 (+ (+ (+ bb (- s1 s2)) (+ q1 q2)) (- 0.0 uv)))
              ;; ---- 碰撞并写出（f_i' = f_i·(1−ω) + ω·w_i·rho·feq项）----
              (set! kr (* om rho))
              (set! k0 (* kr WT0))
              (set! k1 (* kr WT))
              (set! kd (* kr WTD))
              (set_item! B c (+ (* f0 om1) (* k0 bb)))
              (set_item! B (+ c 1) (+ (* f1 om1) (* k1 p1)))
              (set_item! B (+ c 2) (+ (* f2 om1) (* k1 p2)))
              (set_item! B (+ c 3) (+ (* f3 om1) (* k1 m1)))
              (set_item! B (+ c 4) (+ (* f4 om1) (* k1 m2)))
              (set_item! B (+ c 5) (+ (* f5 om1) (* kd d1)))
              (set_item! B (+ c 6) (+ (* f6 om1) (* kd d2)))
              (set_item! B (+ c 7) (+ (* f7 om1) (* kd d3)))
              (set_item! B (+ c 8) (+ (* f8 om1) (* kd d4)))
            }
            { ;; 固体格：速度置零，分布原样搬过去（不参与物理）
              (set_item! UX cn 0.0)
              (set_item! UY cn 0.0)
              (set_item! B c (get_item A c))
              (set_item! B (+ c 1) (get_item A (+ c 1)))
              (set_item! B (+ c 2) (get_item A (+ c 2)))
              (set_item! B (+ c 3) (get_item A (+ c 3)))
              (set_item! B (+ c 4) (get_item A (+ c 4)))
              (set_item! B (+ c 5) (get_item A (+ c 5)))
              (set_item! B (+ c 6) (get_item A (+ c 6)))
              (set_item! B (+ c 7) (get_item A (+ c 7)))
              (set_item! B (+ c 8) (get_item A (+ c 8)))
            })
        (set! c (+ c 9))
        (set! cn (+ cn 1))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
    ;; ---- 贴壁反弹：指向固体邻居的方向，用本格的反方向分布覆写 ----
    (set! x 0)
    (set! nb (length NEAR))
    (while (< x nb) {
      (set! base (* 9 (get_item NEAR x)))
      (if (== (get_item SOLID (get_item NEAR (+ x 1))) 1) (set_item! B (+ base 1) (get_item B (+ base 3))) #f)
      (if (== (get_item SOLID (get_item NEAR (+ x 2))) 1) (set_item! B (+ base 2) (get_item B (+ base 4))) #f)
      (if (== (get_item SOLID (get_item NEAR (+ x 3))) 1) (set_item! B (+ base 3) (get_item B (+ base 1))) #f)
      (if (== (get_item SOLID (get_item NEAR (+ x 4))) 1) (set_item! B (+ base 4) (get_item B (+ base 2))) #f)
      (if (== (get_item SOLID (get_item NEAR (+ x 5))) 1) (set_item! B (+ base 5) (get_item B (+ base 7))) #f)
      (if (== (get_item SOLID (get_item NEAR (+ x 6))) 1) (set_item! B (+ base 6) (get_item B (+ base 8))) #f)
      (if (== (get_item SOLID (get_item NEAR (+ x 7))) 1) (set_item! B (+ base 7) (get_item B (+ base 5))) #f)
      (if (== (get_item SOLID (get_item NEAR (+ x 8))) 1) (set_item! B (+ base 8) (get_item B (+ base 6))) #f)
      (set! x (+ x 9))
    })
    (set! STEPC (+ STEPC 1))
    (set! GS (+ GS 1))
    (set_bounds)
    (fill_bounds_uv)
    ;; 乒乓
    (define tmp A)
    (set! A B)
    (set! B tmp)))

;; ---- 涡量色表（RdBu 11 级 → 65 档）--------------------------------------
(define build_palette
  (lambda ()
    (define k 0)
    (define pos 0.0)
    (define i 0)
    (define f 0.0)
    (set! VR (cons 0.0 '())) (pop VR)
    (set! VG (cons 0.0 '())) (pop VG)
    (set! VB (cons 0.0 '())) (pop VB)
    (set! k 0)
    (while (< k VN) {
      (set! pos (* (/ k (- VN 1)) 10.0))
      (set! i (Math.floor pos))
      (if (> i 9) (set! i 9) #f)
      (set! f (- pos i))
      (push VR (Math.round (+ (* (- 1.0 f) (get_item RC i)) (* f (get_item RC (+ i 1))))))
      (push VG (Math.round (+ (* (- 1.0 f) (get_item RG i)) (* f (get_item RG (+ i 1))))))
      (push VB (Math.round (+ (* (- 1.0 f) (get_item RB i)) (* f (get_item RB (+ i 1))))))
      (set! k (+ k 1))
    })))

;; ---- 涡量底图：铺满整屏（每格一块 CELL×CELL）-----------------------------
;; ω = ∂uy/∂x − ∂ux/∂y（中心差分，格子单位）；归一化到 [-1,1] 后查 RdBu 表：
;; 蓝 = 负涡量、近白 = 零、红 = 正涡量。边界格速度已由 fill_bounds_uv 补全，
;; 所以上下两端不再出现假色带。
(define draw_vort
  (lambda ()
    (define gy 1)
    (define gx 1)
    (define c 0)
    (define w 0.0)
    (define t 0.0)
    (define kk 0.0)
    (define py 0)
    (set! gy 1)
    (while (< gy (+ NY 1)) {
      (set! gx 1)
      (set! c (+ (* gy W) 1))
      (set! py (* (- gy 1) CELL))
      (while (< gx (+ NX 1)) {
        (set! w (* 0.5 (- (- (get_item UY (+ c 1)) (get_item UY (- c 1)))
                          (- (get_item UX (+ c W)) (get_item UX (- c W))))))
        (set! t (/ w VS))
        (if (> t 1.0) (set! t 1.0) #f)
        (if (< t -1.0) (set! t -1.0) #f)
        ;; 死区 + 重映射：|t|<0.12 一律当 0（全场数值噪声只值一点点涡量，不压掉的话
        ;; 满屏都是一层淡蓝淡红的薄雾，反而看不出剪切层）；其余线性拉到 [0,1]。
        (if (> t 0.0)
            {
              (set! t (- t 0.12))
              (if (< t 0.0) (set! t 0.0) #f)
              (set! t (/ t 0.88))
            }
            {
              (set! t (+ t 0.12))
              (if (> t 0.0) (set! t 0.0) #f)
              (set! t (/ t 0.88))
            })
        ;; (t+1)*32 ∈ [0,64]；(get_item) 的浮点索引自动四舍五入 ⇒ 省一次 Math.floor
        (set! kk (* (+ t 1.0) 32.0))
        (Gfx.fill_rect (* (- gx 1) CELL) py CELL CELL
                       (get_item VR kk) (get_item VG kk) (get_item VB kk))
        (set! gx (+ gx 1))
        (set! c (+ c 1))
      })
      (set! gy (+ gy 1))
    })))

;; ---- 烟线：双线性采样速度场 → 推进 → 拖尾 -------------------------------
;; 形态与改版前一致：**短加色发光小尾迹**（沿速度反推 6 帧 ≈10px、快=亮），
;; 只有 260 颗。唯一的分支：涡量底图打开时改画**墨线**（浅底上加色发光等于看不见）。
(define dye_step
  (lambda ()
    (define i 0)
    (define x 0.0)
    (define y 0.0)
    (define gx 0.0)
    (define gy 0.0)
    (define i0 0.0)
    (define j0 0.0)
    (define fx 0.0)
    (define fy 0.0)
    (define c 0)
    (define vx 0.0)
    (define vy 0.0)
    (define sp 0.0)
    (define au 0.0)
    (define av 0.0)
    (define br 0.0)
    (define du 0.0)
    (define dv 0.0)
    (define hit 0)
    (define step_px 0.0)
    (define tail_px 0.0)
    (define w00 0.0)
    (define w10 0.0)
    (define w01 0.0)
    (define w11 0.0)
    (set! step_px (* CELL NSUB))
    (set! tail_px (* step_px TAILF))
    (set! i 0)
    (while (< i NPD) {
      (set! x (get_item PX i))
      (set! y (get_item PY i))
      ;; 屏幕 → 格子连续坐标（格 i 的中心在 (i-0.5)*CELL）
      (set! gx (+ (/ x CELL) 0.5))
      (set! gy (+ (/ y CELL) 0.5))
      ;; 取整用 mod（VM 内建）而不是 Math.floor（native 调用）—— 与改版前的写法一致。
      ;; 注：本 App 上两者实测差异在噪声内（见文件头的"计时"说明），这里只是少一次调用。
      (set! i0 (- gx (mod gx 1.0)))
      (set! j0 (- gy (mod gy 1.0)))
      (set! fx (- gx i0))
      (set! fy (- gy j0))
      (if (< i0 1.0) (set! i0 1.0) #f)
      (if (> i0 NX) (set! i0 NX) #f)
      (if (< j0 1.0) (set! j0 1.0) #f)
      (if (> j0 NY) (set! j0 NY) #f)
      (set! c (+ (* j0 W) i0))
      ;; 双线性权重（先算好，四个格点各查一次）
      (set! w00 (* (- 1.0 fx) (- 1.0 fy)))
      (set! w10 (* fx (- 1.0 fy)))
      (set! w01 (* (- 1.0 fx) fy))
      (set! w11 (* fx fy))
      (set! vx (+ (+ (* w00 (get_item UX c)) (* w10 (get_item UX (+ c 1))))
                  (* w01 (get_item UX (+ c W)))))
      (set! vx (+ vx (* w11 (get_item UX (+ (+ c W) 1)))))
      (set! vy (+ (+ (* w00 (get_item UY c)) (* w10 (get_item UY (+ c 1))))
                  (* w01 (get_item UY (+ c W)))))
      (set! vy (+ vy (* w11 (get_item UY (+ (+ c W) 1)))))
      ;; 推进（格子单位 → 像素）：一帧走 NSUB 步
      (set! x (+ x (* vx step_px)))
      (set! y (+ y (* vy step_px)))
      ;; 出界 / 撞障碍物 → 回入口重新播种
      (set! hit 0)
      (if (or (< x 1.0) (> x 316.0) (< y 2.0) (> y 238.0)) (set! hit 1) #f)
      (set! du (* (- x PXC) IRX))
      (set! dv (* (- y PYC) IRY))
      (if (>= SHAPE 2)
          (if (and (< (if (< du 0.0) (- 0.0 du) du) 1.05) (< (if (< dv 0.0) (- 0.0 dv) dv) 1.05))
              (set! hit 1)
              #f)
          (if (< (+ (* du du) (* dv dv)) 1.1) (set! hit 1) #f))
      (if (== hit 1)
          {
            (set! x (+ 2.0 (* (Math.random) 6.0)))
            (set! y (+ 6.0 (* (Math.random) 228.0)))
          }
          #f)
      (set_item! PX i x)
      (set_item! PY i y)
      ;; 拖尾：亮度按速度（曼哈顿近似，快=亮）
      (set! au (if (< vx 0.0) (- 0.0 vx) vx))
      (set! av (if (< vy 0.0) (- 0.0 vy) vy))
      (set! sp (+ au av))
      (set! br (+ 0.55 (* 14.0 sp)))
      (if (> br 1.0) (set! br 1.0) #f)
      (if (== SHOWVORT 1)
          (Gfx.draw_line_aa_mode (- x (* vx tail_px)) (- y (* vy tail_px))
                                 x y 1.0 30 34 46 150)
          (Gfx.draw_line_aa_mode (- x (* vx tail_px)) (- y (* vy tail_px))
                                 x y 1.0 150 200 240 (* 255.0 br)))
      (set! i (+ i 1))
    })))

;; ---- 障碍物 --------------------------------------------------------------
;; 圆：两层抗锯齿圆盘（浅色外辉光 + 深色本体 + 高光）；
;; 椭圆/方/板：逐行扫描线填充（两次：外圈浅色 + 本体深色），行数 2·ORY·CELL 条。
(define fill_shape_rows
  (lambda (rx ry r g b)
    (define yy 0)
    (define y0 0)
    (define y1 0)
    (define dy 0.0)
    (define hw 0.0)
    (define hwp 0.0)
    (set! y0 (Math.floor (- PYC (* ry CELL))))
    (set! y1 (Math.floor (+ PYC (* ry CELL))))
    (set! yy y0)
    (while (<= yy y1) {
      (set! dy (/ (- yy PYC) (* ry CELL)))
      (if (> dy 1.0) (set! dy 1.0) #f)
      (if (< dy -1.0) (set! dy -1.0) #f)
      (if (>= SHAPE 2)
          (set! hwp (* rx CELL))
          (set! hwp (* (* rx CELL) (Math.sqrt (- 1.0 (* dy dy))))))
      (set! hw (+ hwp hwp))
      (if (> hw 1.0)
          (Gfx.fill_rect (Math.floor (- PXC hwp)) yy (+ hw 1.0) 1 r g b)
          #f)
      (set! yy (+ yy 1))
    })))

(define draw_obs
  (lambda ()
    (define rx 0.0)
    (define ry 0.0)
    (set! rx ORX)
    (set! ry ORY)
    (if (== SHAPE 0)
        {
          (Gfx.fill_circle_aa PXC PYC (+ (* rx CELL) 5.0) 176 198 226 120)
          (Gfx.fill_circle_aa PXC PYC (+ (* rx CELL) 1.5) 92 108 132 190)
          (Gfx.fill_circle_aa PXC PYC (* rx CELL) 26 30 40 255)
          (Gfx.fill_circle_aa (+ PXC (* 0.22 (* rx CELL))) (+ PYC (* 0.16 (* ry CELL)))
                              (* 0.5 (* rx CELL)) 46 52 66 255)
          (Gfx.fill_circle_aa (- PXC (* 0.36 (* rx CELL))) (- PYC (* 0.4 (* ry CELL)))
                              (* 0.26 (* rx CELL)) 206 220 238 150)
        }
        {
          (fill_shape_rows (+ rx 0.16) (+ ry 0.16) 176 198 226)
          (fill_shape_rows rx ry 26 30 40)
          (Gfx.fill_circle_aa (- PXC (* 0.3 (* rx CELL))) (- PYC (* 0.42 (* ry CELL)))
                              (* 0.26 (* rx CELL)) 206 220 238 140)
        })))

;; ---- 固定格点速度矢量场（quiver）：显示模式的第二种 -----------------------
;; 与"自由烟丝"互补：烟粒是**随流体走**的示踪物（看得见输运与涡的卷起），
;; 矢量场是**钉在格点上**的采样（看得见瞬时流向与快慢，位置不随时间漂移）。
;; 每 QSTEP 格一枚（16×12=192 枚）：杆 + 两笔箭头头部，方向/长度 = 当地速度
;; （长度 = |u|·QSCALE，钳到 QMAX 以内，免得大速度箭头互相压住）。
(define QSTEP 2)                   ; 采样间隔（格）
(define QSCALE 58.0)               ; 速度 → 像素长度
(define QMAX 15.0)                 ; 箭头最大长度（px）

(define draw_quiver
  (lambda ()
    (define gy 1)
    (define gx 1)
    (define c 0)
    (define ux 0.0)
    (define uy 0.0)
    (define len 0.0)
    (define scl 0.0)
    (define px 0.0)
    (define py 0.0)
    (define tx 0.0)
    (define ty 0.0)
    (define nx 0.0)
    (define ny 0.0)
    (define bx 0.0)
    (define by 0.0)
    (define hx 0.0)
    (define hy 0.0)
    (define md 0)
    (define cr 0)
    (define cg 0)
    (define cb 0)
    (define hw 2.6)
    (define hl 3.8)
    (define k 0.0)
    ;; 配色：涡量底图开（浅底）→ 墨色不透明；关（暗底）→ 青白加色
    (if (== SHOWVORT 1)
        { (set! md 150) (set! cr 34) (set! cg 38) (set! cb 52) }
        { (set! md 3) (set! cr 120) (set! cg 205) (set! cb 255) })
    (set! gy 1)
    (while (< gy (+ NY 1)) {
      (set! gx 1)
      (set! c (+ (* gy W) 1))
      (while (< gx (+ NX 1)) {
        (if (== (get_item SOLID c) 0)
            {
              (set! ux (get_item UX c))
              (set! uy (get_item UY c))
              (set! len (Math.sqrt (+ (* ux ux) (* uy uy))))
              (if (> len 0.0001)
                  {
                    (set! scl (* QSCALE len))
                    (if (> scl QMAX) (set! scl QMAX) #f)
                    ;; 箭头头部随杆长收小：短箭（低速区）不会被头部"盖过头"变成反向小叉
                    (set! k (* scl 0.45))
                    (set! hl (if (> k 3.8) 3.8 k))
                    (set! k (* scl 0.32))
                    (set! hw (if (> k 2.6) 2.6 k))
                    (set! px (+ (* (- gx 1) CELL) (* 0.5 CELL)))
                    (set! py (+ (* (- gy 1) CELL) (* 0.5 CELL)))
                    (set! nx (/ ux len))
                    (set! ny (/ uy len))
                    (set! tx (+ px (* nx scl)))
                    (set! ty (+ py (* ny scl)))
                    ;; 头部基点在 tip 后退 hl，左右各偏 hw（法向 = (-ny, nx)）
                    (set! bx (- tx (* nx hl)))
                    (set! by (- ty (* ny hl)))
                    (Gfx.draw_line_aa_mode px py tx ty 1.0 cr cg cb md)
                    (set! hx (+ bx (* (- 0.0 ny) hw)))
                    (set! hy (+ by (* nx hw)))
                    (Gfx.draw_line_aa_mode tx ty hx hy 1.0 cr cg cb md)
                    (set! hx (- bx (* (- 0.0 ny) hw)))
                    (set! hy (- by (* nx hw)))
                    (Gfx.draw_line_aa_mode tx ty hx hy 1.0 cr cg cb md)
                  }
                  #f)
            }
            #f)
        (set! gx (+ gx QSTEP))
        (set! c (+ c QSTEP))
      })
      (set! gy (+ gy QSTEP))
    })))

;; ---- 底部控件栏 + HUD ----------------------------------------------------
(define draw_bar
  (lambda ()
    (define i 0)
    (define n 0)
    (define b #f)
    (Gfx.rounded_rect 0 BAR_Y 320 BAR_H 10 10 0 0 18 20 27 198)
    (Gfx.fill_rect 0 BAR_Y 320 1 66 74 92)
    (set! i 0)
    (set! n (length BTNS))
    (while (< i n) {
      (set! b (get_item BTNS i))
      ;; 第 8 枚（涡量）是**开关**：常亮 = 底图开；第 9 枚是模式轮转，靠标签表达
      (if (== i 7)
          (BTN.draw_state b (if (== SHOWVORT 1) #t (BTN.armed b)))
          (BTN.draw_state b (BTN.armed b)))
      (set! i (+ i 1))
    })))

(define draw_hud
  (lambda ()
    (if (== HUDD 1) (upd_hud) #f)
    ;; 底板**不透明**：半透明的话上一拍的旧数字会透过来，参数一改就变成一团重影
    (Gfx.rounded_rect 2 2 158 16 5 5 5 5 74 82 100 1)
    (Gfx.rounded_rect 3 3 156 14 4 4 4 4 18 21 30 1)
    (Gfx.draw_text HUD_STR 7 3 232 238 246)))

;; ---- 参数与 HUD 文本 -----------------------------------------------------
(define reynolds
  (lambda ()
    (/ (* (* U0 (* 2.0 ORY)) 3.0) (- TAU 0.5))))

(define upd_hud
  (lambda ()
    (set! HUD_STR
      (String.concat "U" (String.concat (String.atom_to_string (Math.to_fixed U0 2))
        (String.concat " Re" (String.concat (String.atom_to_string (Math.to_fixed (reynolds) 0))
          (String.concat " " (String.concat (get_item SHNAME SHAPE)
            (String.concat " r" (String.atom_to_string (Math.to_fixed RAD 1))))))))))
    (set! HUDD 0)))

;; 流体参数改动：重算 ω/(1−ω) 与涡量尺度，并重新播种入口阵风
(define upd_phys
  (lambda ()
    (set! OMEGA (/ 1.0 TAU))
    (set! OM1 (- 1.0 OMEGA))
    (set! VS (* 0.65 U0))
    (set! GS 0)
    (set! HUDD 1)))

;; ---- 底部按钮的动作 ------------------------------------------------------
(define btn_action
  (lambda (i)
    (cond
      ((== i 0) { (set! U0 (- U0 0.02)) (if (< U0 0.08) (set! U0 0.08) #f) (upd_phys) })
      ((== i 1) { (set! U0 (+ U0 0.02)) (if (> U0 0.24) (set! U0 0.24) #f) (upd_phys) })
      ((== i 2) { (set! TAU (- TAU 0.005)) (if (< TAU 0.515) (set! TAU 0.515) #f) (upd_phys) })
      ((== i 3) { (set! TAU (+ TAU 0.005)) (if (> TAU 0.575) (set! TAU 0.575) #f) (upd_phys) })
      ((== i 4) { (set! RAD (- RAD 0.4)) (if (< RAD 1.2) (set! RAD 1.2) #f) (upd_shape) (rebuild_obs) })
      ((== i 5) { (set! RAD (+ RAD 0.4)) (if (> RAD 3.2) (set! RAD 3.2) #f) (upd_shape) (rebuild_obs) })
      ((== i 6) { (set! SHAPE (Math.floor (mod (+ SHAPE 1) 4)))
                  (upd_shape) (rebuild_obs)
                  (BTN.set_label (get_item BTNS 6) (get_item SHNAME SHAPE)) })
      ((== i 7) (set! SHOWVORT (Math.floor (mod (+ SHOWVORT 1) 2))))
      ((== i 8) { (set! MODE (Math.floor (mod (+ MODE 1) 2)))
                  (BTN.set_label (get_item BTNS 8) (get_item MDNAME MODE)) })
      (else #t))))

;; 9 枚按钮的建立（标签/位置/样式一次性配好；形状/模式按钮的标签随状态变）
(define build_buttons
  (lambda ()
    (define i 0)
    (define b #f)
    (define lb '("速-" "速+" "粘-" "粘+" "径-" "径+" "圆" "涡量" "烟丝"))
    (set! BTNS (cons 0 '())) (pop BTNS)
    (set! i 0)
    (while (< i 9) {
      (set! b (BTN.make (get_item lb i) (+ 6 (* i 34)) BTN_Y BTN_W BTN_H))
      (BTN.set_font b 1)
      (BTN.set_text_color b 228 235 245)
      (BTN.set_bg b 38 50 66)
      (BTN.set_active_bg b 116 172 226)
      (BTN.set_radius b 5)
      (BTN.set_border b 92 104 124)
      (push BTNS b)
      (set! i (+ i 1))
    })
    ;; 两枚"显示"按钮给一套偏青的底色，与物理参数按钮区分开
    (BTN.set_bg (get_item BTNS 7) 30 62 84)
    (BTN.set_bg (get_item BTNS 8) 30 62 84)))

;; 每拍：9 枚按钮各做一次电平→边沿判定；完成点按时执行动作
(define track_btns
  (lambda ()
    (define i 0)
    (define n 0)
    (set! n (length BTNS))
    (while (< i n) {
      (if (BTN.track (get_item BTNS i)) (btn_action i) #f)
      (set! i (+ i 1))
    })))

(define any_armed
  (lambda ()
    (define i 0)
    (define n 0)
    (define r 0)
    (set! n (length BTNS))
    (while (< i n) {
      (if (BTN.armed (get_item BTNS i)) (set! r 1) #f)
      (set! i (+ i 1))
    })
    r))

;; ---- 触屏拖动障碍物 ------------------------------------------------------
;; 电平 + 边沿判定（与宿主 GUI 同一哲学）：按下落在障碍物上 → 锁定抓取偏移；
;; 拖动期间把中心夹在流场内部；中心移动超过 0.3 格才重建掩码（重建约 3 ms）。
(define drag_input
  (lambda ()
    (define pressed 0)
    (define px 0)
    (define py 0)
    (define nx 0.0)
    (define ny 0.0)
    (define hx 0.0)
    (define hy 0.0)
    (set! pressed (if (Touch.is_pressed) 1 0))
    (if (== pressed 1)
        {
          (set! px (Touch.x))
          (set! py (Touch.y))
          (if (and (== PREV_DN 0) (== DRAG 0) (== (any_armed) 0) (< py BAR_Y))
              (if (< (+ (* (* (- px PXC) IRX) (* (- px PXC) IRX))
                        (* (* (- py PYC) IRY) (* (- py PYC) IRY)))
                     2.4)
                  {
                    (set! DRAG 1)
                    (set! GRAB_DX (- CXC (+ (/ px CELL) 0.5)))
                    (set! GRAB_DY (- CYC (+ (/ py CELL) 0.5)))
                  }
                  #f)
              #f)
          (if (== DRAG 1)
              {
                (set! nx (+ (+ (/ px CELL) 0.5) GRAB_DX))
                (set! ny (+ (+ (/ py CELL) 0.5) GRAB_DY))
                (set! hx (+ ORX 0.6))
                (set! hy (+ ORY 0.6))
                (if (< nx (+ 1.0 hx)) (set! nx (+ 1.0 hx)) #f)
                (if (> nx (- (+ NX 1.0) hx)) (set! nx (- (+ NX 1.0) hx)) #f)
                (if (< ny (+ 1.0 hy)) (set! ny (+ 1.0 hy)) #f)
                (if (> ny (- (+ NY 1.0) hy)) (set! ny (- (+ NY 1.0) hy)) #f)
                (if (or (> (Math.abs (- nx CXC)) 0.3) (> (Math.abs (- ny CYC)) 0.3))
                    {
                      (set! CXC nx)
                      (set! CYC ny)
                      (set! PXC (* (- CXC 0.5) CELL))
                      (set! PYC (* (- CYC 0.5) CELL))
                      (rebuild_obs)
                    }
                    #f)
              }
              #f)
          (set! PREV_DN 1)
        }
        {
          (if (== DRAG 1) (set! DRAG 0) #f)
          (set! PREV_DN 0)
        })))

;; ---- 每拍 ----------------------------------------------------------------
(define tick
  (lambda ()
    (define w0 0)
    (define l0 0)
    (define d0 0)
    (set! w0 (Diag.ms))
    ;; ① 输入：退出按钮 → 控件栏 → 拖动
    (EX.check)
    (track_btns)
    (drag_input)
    ;; ② 流场推进（每拍 NSUB 步）
    (if (== SHOWVORT 0) (Gfx.clear) #f)
    (set! l0 (Diag.ms))
    (lbm_step)
    (set! MSL (- (Diag.ms) l0))
    ;; ③ 涡量底图（铺满整屏）
    (if (== SHOWVORT 1) (draw_vort) #f)
    ;; ④ 可视化：自由烟丝（烟粒）或固定格点速度矢量场
    (set! d0 (Diag.ms))
    (if (== MODE 0) (dye_step) (draw_quiver))
    (set! MSD (- (Diag.ms) d0))
    ;; ⑤ 障碍物 + 控件栏 + HUD + 退出按钮
    (draw_obs)
    (draw_bar)
    (draw_hud)
    (EX.draw_button)
    (Gfx.refresh)
    (set! FRAMES (+ FRAMES 1))
    (set! ACCL (+ ACCL MSL))
    (set! ACCD (+ ACCD MSD))
    (set! ACCF (+ ACCF (- (Diag.ms) w0)))
    (if (>= FRAMES 30)
        {
          (display "[vx] ms_lbm=") (display (/ ACCL 30.0))
          (display " ms_dye=") (display (/ ACCD 30.0))
          (display " ms_frm=") (display (/ ACCF 30.0))
          (display " re=") (display (/ (* (* U0 (* 2.0 ORY)) 3.0) (- TAU 0.5)))
          (display " uy=") (display (Math.to_fixed (get_item UY (+ (* 12 W) 26)) 3))
          (display " c=") (display (Math.to_fixed CXC 1))
          (display ",") (display (Math.to_fixed CYC 1))
          (display " u=") (display (Math.to_fixed U0 2))
          (display " tau=") (display (Math.to_fixed TAU 3))
          (display " rad=") (display (Math.to_fixed RAD 1))
          (display " sh=") (display (Math.to_fixed SHAPE 0))
          (display " vor=") (display SHOWVORT)
          (display " md=") (display MODE)
          (display " n=") (display STEPC)
          (newline)
          (set! FRAMES 0)
          (set! ACCL 0.0)
          (set! ACCD 0.0)
          (set! ACCF 0.0)
        }
        #f)))

;; ---- 启动 ----------------------------------------------------------------
(define _timer 0)

(EX.install)
(Gfx.set_font 1)
(build_palette)
(build_grid)
(upd_phys)
(upd_shape)
(rebuild_obs)
(spawn_dye)
(build_buttons)
(upd_hud)
(Gfx.clear)
(if (== SHOWVORT 1) (draw_vort) #f)
(draw_obs)
(draw_bar)
(draw_hud)
(EX.draw_button)
(Gfx.refresh)
(set! _timer (System.set_interval 33 tick))
