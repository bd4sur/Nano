;@id=vortex
;@title=卡门涡街
;@version=2026-11-25
;; ============================================================================
;; 卡门涡街（vortex）—— 灵机 App：圆柱绕流的**真格子玻尔兹曼（LBM D2Q9）模拟**
;; + 风洞烟线（streakline）可视化。涡街不是画出来的，是流场自己失稳长出来的。
;;
;; 物理（每一步都在 Scheme 里算）：
;;   · D2Q9 单松弛（BGK）：每格 rho=Σf_i、u=Σc_i f_i/rho、
;;     feq_i = w_i·rho·(1 + 3c_i·u + 4.5(c_i·u)² − 1.5|u|²)、f_i ← f_i + (feq_i − f_i)/τ；
;;     用 pull 式平移（读邻居偏移、写本格、双缓冲乒乓），一格一遍、无迭代。
;;   · 圆柱：格点标记固体；贴壁格**半步反弹**（bounce-back）给无滑移边界 ——
;;     必须用**碰撞后**的分布（B 缓冲）回弹，用碰撞前的值会往流场里注入能量、几步就发散。
;;   · 入口定速（可加短暂横向"阵风"给不稳定性播种）；出口零梯度复制；
;;     上下**自由滑移**（边界格写 feq(rho=1, u=(ux_local, 0))）。
;;   · 参数：26×18 格（每格 12px）、D=4 格、U=0.12、τ=0.514 → **Re≈103**，
;;     St≈0.2 ⇒ 涡对脱落周期 ≈ D/(St·U) ≈ 167 步 ≈ 2.8 s（每帧跑 2 步）。
;;
;; 可视化：
;;   · 260 颗烟粒在速度场里双线性插值输运（每帧推进 NSUB 步），画成**沿速度反推 4 帧**
;;     的加色彗尾（10~15px）—— 这才是烟线的观感，且不增加调用次数。
;;   · 点按切换涡量底图（每格一块 fill_rect，红=ω>0、蓝=ω<0，gamma≈2 压暗小涡量）：
;;     开时能直接看到两条剪切层与交替脱落的涡对符号。
;;   · 圆柱是暗金属圆盘 + 外辉光。
;;
;; 成本（WASM 目标实测，headless Chromium）：LBM 单步 8.6~8.9 ms（≈17 µs/格）、
;;   每帧 2 步 ⇒ 23.5 ms/帧 ⇒ 28 fps。解释器里能跑动靠三件事：
;;     ① 格数少（468 格）+ 每格只做一遍 + 无迭代（LBM 天生如此）；
;;     ② 贴壁反弹只在"贴壁格清单"上做（否则每格 9 次固体判定，成本翻倍）；
;;     ③ 烟粒的取整用 `(- x (mod x 1))`、速度模长用曼哈顿近似（省 Math.floor/sqrt），
;;        并且不存"上一帧位置表"（推进前的位置本来就在局部量里）。
;;
;; 交互：点按切换涡量底图；右上角「退出」。
;; 诊断行（每 30 拍）：[vx] ms_lbm=<单步> ms_frm=<每帧> umax=<最大|u|> probe_uy=<探针> re=<Re>
;;
;; 解释器约束：算术**恒二元**；无 list/let/map；局部量在 lambda 体首部声明、循环里 set!；
;; 只有 #f 是假；cond 子句只容一个表达式；(cons 0 '()) 的哑元素要 pop。
;; ============================================================================

(native Gfx)
(native Math)
(native Diag)
(native System)
(native Touch)

(import EX "app/lib/exit.scm")

;; ---- 网格与物理参数 ------------------------------------------------------
(define NX 28)                     ; 内部列（每格 CELL px）
(define NY 18)                     ; 内部行
(define CELL 11)                   ; 每格像素（28*11=308, 18*11=198）
(define GX0 6.0)                   ; 网格在屏幕上的左上角（左右各留 4px）
(define GY0 8.0)                   ; 上留 8px、下方 16px 放 HUD
(define NSUB 2)                    ; 每帧跑几个 LBM 步
(define W (+ NX 2))                ; 加一圈边界格（免越界判定）
(define H (+ NY 2))
(define NC (* W H))
(define N9 (* NC 9))
(define U0 0.12)                   ; 入口速度（格子单位；Ma≈0.2，安全）
(define TAU 0.514)                 ; 松弛时间 τ = 0.5 + 3ν
(define OMEGA (/ 1.0 TAU))
(define OM1 (- 1.0 OMEGA))
(define CXC 6.0)                   ; 圆柱中心（格子坐标）
(define CYC 8.0)
(define RAD 2.0)                   ; 圆柱半径（格）→ D=4 格 = 48 px

;; 九个方向的权、单位速度、源格偏移（c 的偏移 ×9）、反方向下标
(define WT0 0.4444444)
(define WT 0.1111111)
(define WTD 0.0277778)             ; 1/36
(define OPP '(0 3 4 1 2 7 8 5 6))
(define SO0 0)
(define SO1 0) (define SO2 0) (define SO3 0) (define SO4 0)
(define SO5 0) (define SO6 0) (define SO7 0) (define SO8 0)

;; ---- 状态 ----------------------------------------------------------------
(define A '())                     ; 分布函数缓冲（N9）
(define B '())
(define SOLID '())                 ; 每格 1 = 固体（NC）
(define UX '())                    ; 速度场（NC）
(define UY '())
(define NEAR '())                  ; 贴壁格子的基址表（供反弹覆写）
(define DYE '())

(define NPD 260)                   ; 烟粒数（成本随粒子超线性上升，实测 260 是 30fps 的甜点）
(define PX '())
(define PY '())
(define PXV '())                   ; 上一帧位置（画拖尾）
(define PYV '())

(define SHOWVORT 0)   ; 涡量底图：默认关（点按切换；开时能看到交替涡对的符号）
(define STEPC 0)
(define MAXU 0.0)                  ; 诊断：本拍最大 |u|                   ; LBM 步数（前 400 步给入口加横向阵风播种）                ; 是否画涡量底图（点按切换）
(define T 0.0)
(define LASTMS 0)
(define FRAMES 0)
(define ACCL 0.0)
(define ACCF 0.0)
(define MSL 0.0)                   ; 最近一次 LBM 单步耗时

;; ---- 启动：建网格 --------------------------------------------------------
(define build_grid
  (lambda ()
    (define i 0)
    (define cn 0)
    (define y 0)
    (define x 0)
    (define dx 0.0)
    (define dy 0.0)
    ;; 偏移：源格 = c + SOi，其中 off_i = (cy*W + cx)*9
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
    ;; 固体掩码 + 贴壁表 + 速度场
    (set! SOLID (cons 0 '())) (pop SOLID)
    (set! UX (cons 0.0 '())) (pop UX)
    (set! UY (cons 0.0 '())) (pop UY)
    (set! NEAR (cons 0 '())) (pop NEAR)
    (set! cn 0)
    (while (< cn NC) {
      (set! y (Math.floor (/ cn W)))
      (set! x (- cn (* y W)))
      (set! dx (- x CXC))
      (set! dy (- y CYC))
      (if (< (+ (* dx dx) (* dy dy)) (* RAD RAD))
          {
            (push SOLID 1)
            (if (and (> x 0) (< x (+ NX 1)) (> y 0) (< y (+ NY 1))) (push NEAR cn) #f)
          }
          (push SOLID 0))
      (push UX 0.0) (push UY 0.0)
      (set! cn (+ cn 1))
    })
    ;; 把贴壁格的**流体邻居**也登记进来（它们才是需要回弹覆写的格子）
    (set! NEAR (cons 0 '())) (pop NEAR)
    (set! y 1)
    (while (< y (+ NY 1)) {
      (set! x 1)
      (while (< x (+ NX 1)) {
        (set! cn (+ (* y W) x))
        (if (== (get_item SOLID cn) 0)
            (if (or (== (get_item SOLID (- cn 1)) 1) (== (get_item SOLID (+ cn 1)) 1)
                    (== (get_item SOLID (- cn W)) 1) (== (get_item SOLID (+ cn W)) 1))
                (push NEAR (* cn 9))
                #f)
            #f)
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })))

;; ---- 烟粒 ----------------------------------------------------------------
(define spawn_dye
  (lambda ()
    (set! DYE (cons 0.0 '())) (pop DYE)
    (set! PX (cons 0.0 '())) (pop PX)
    (set! PY (cons 0.0 '())) (pop PY)
    (set! PXV (cons 0.0 '())) (pop PXV)
    (set! PYV (cons 0.0 '())) (pop PYV)
    (define i 0)
    (while (< i NPD) {
      (push DYE 0.0)
      (push PX (+ (+ GX0 2.0) (* (Math.random) 6.0)))
      (push PY (+ (+ GY0 6.0) (* (Math.random) (- (* NY CELL) 12.0))))
      (push PXV 2.0)
      (push PYV 0.0)
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
    ;; 入口平衡态：rho=1, u=(U0,0)
    (set! GY 0.0)
    (if (< STEPC 400) (set! GY (* 0.02 (Math.sin (* 0.15 STEPC)))) #f)
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
      ;; 入口列 x=0（基址 = (y*W + 0)*9）
      (set! c (* 9 (* y W)))
      (set_item! B c e0) (set_item! B (+ c 1) e1) (set_item! B (+ c 2) e2)
      (set_item! B (+ c 3) e3) (set_item! B (+ c 4) e4) (set_item! B (+ c 5) e5)
      (set_item! B (+ c 6) e6) (set_item! B (+ c 7) e7) (set_item! B (+ c 8) e8)
      ;; 出口列 x=NX+1：复制 x=NX
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
      ;; 上边界 y=0：读 y=1 的 f1/f3 求 rho/ux
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

;; ---- LBM 单步：pull + 碰撞 + 写目标缓冲 ---------------------------------
;; 每格：9 次读（按方向偏移，来自 A）→ 守恒量 → 碰撞 → 9 次写（到 B）。
;; 贴壁格子先按普通格子算，随后由 fix_wall 把指向固体的方向覆写成反弹值。
(define lbm_step
  (lambda ()
    (define c 0)
    (define cn 0)
    (define y 1)
    (define x 1)
    (define f0 0.0) (define f1 0.0) (define f2 0.0) (define f3 0.0) (define f4 0.0)
    (define f5 0.0) (define f6 0.0) (define f7 0.0) (define f8 0.0)
    (define rho 0.0) (define ux 0.0) (define uy 0.0)
    (define u2 0.0) (define bb 0.0) (define q1 0.0) (define q2 0.0)
    (define s1 0.0) (define s2 0.0) (define uv 0.0)
    (define p1 0.0) (define p2 0.0) (define m1 0.0) (define m2 0.0)
    (define d1 0.0) (define d2 0.0) (define d3 0.0) (define d4 0.0)
    (define kr 0.0) (define k0 0.0) (define k1 0.0) (define kd 0.0)
    (set! y 1)
    (while (< y (+ NY 1)) {
      (set! c (* 9 (+ (* y W) 1)))
      (set! x 1)
      (while (< x (+ NX 1)) {
        (set! cn (+ (* y W) x))
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
              (if (> (Math.abs ux) MAXU) (set! MAXU (Math.abs ux)) #f)
              (if (> (Math.abs uy) MAXU) (set! MAXU (Math.abs uy)) #f)
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
              (set! kr (* OMEGA rho))
              (set! k0 (* kr WT0))
              (set! k1 (* kr WT))
              (set! kd (* kr WTD))
              (set_item! B c (+ (* f0 OM1) (* k0 bb)))
              (set_item! B (+ c 1) (+ (* f1 OM1) (* k1 p1)))
              (set_item! B (+ c 2) (+ (* f2 OM1) (* k1 p2)))
              (set_item! B (+ c 3) (+ (* f3 OM1) (* k1 m1)))
              (set_item! B (+ c 4) (+ (* f4 OM1) (* k1 m2)))
              (set_item! B (+ c 5) (+ (* f5 OM1) (* kd d1)))
              (set_item! B (+ c 6) (+ (* f6 OM1) (* kd d2)))
              (set_item! B (+ c 7) (+ (* f7 OM1) (* kd d3)))
              (set_item! B (+ c 8) (+ (* f8 OM1) (* kd d4)))
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
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
    ;; ---- 贴壁反弹：指向固体邻居的方向，用本格的反方向分布覆写 ----
    (set! x 0)
    (while (< x (length NEAR)) {
      (set! c (get_item NEAR x))
      ;; i=1..8 逐个检查（i=0 静止方向无邻居，不需处理）
      (if (== (get_item SOLID (Math.floor (/ (+ c SO1) 9))) 1) (set_item! B (+ c 1) (get_item B (+ c 3))) #f)
      (if (== (get_item SOLID (Math.floor (/ (+ c SO2) 9))) 1) (set_item! B (+ c 2) (get_item B (+ c 4))) #f)
      (if (== (get_item SOLID (Math.floor (/ (+ c SO3) 9))) 1) (set_item! B (+ c 3) (get_item B (+ c 1))) #f)
      (if (== (get_item SOLID (Math.floor (/ (+ c SO4) 9))) 1) (set_item! B (+ c 4) (get_item B (+ c 2))) #f)
      (if (== (get_item SOLID (Math.floor (/ (+ c SO5) 9))) 1) (set_item! B (+ c 5) (get_item B (+ c 7))) #f)
      (if (== (get_item SOLID (Math.floor (/ (+ c SO6) 9))) 1) (set_item! B (+ c 6) (get_item B (+ c 8))) #f)
      (if (== (get_item SOLID (Math.floor (/ (+ c SO7) 9))) 1) (set_item! B (+ c 7) (get_item B (+ c 5))) #f)
      (if (== (get_item SOLID (Math.floor (/ (+ c SO8) 9))) 1) (set_item! B (+ c 8) (get_item B (+ c 6))) #f)
      (set! x (+ x 1))
    })
    (set! STEPC (+ STEPC 1))
    (set_bounds)
    ;; 乒乓
    (define tmp A)
    (set! A B)
    (set! B tmp)))

;; ---- 涡量底图（每 2 格一块，红=顺时针 / 蓝=逆时针）----------------------
(define draw_vort
  (lambda ()
    (define gy 1)
    (define gx 1)
    (define c 0)
    (define w 0.0)
    (define m 0.0)
    (define r 0.0)
    (define g 0.0)
    (define b 0.0)
    (set! gy 1)
    (while (< gy (+ NY 1)) {
      (set! gx 1)
      (while (< gx (+ NX 1)) {
        (set! c (+ (* gy W) gx))
        ;; ω = ∂uy/∂x − ∂ux/∂y（中心差分，格子单位）
        (set! w (- (/ (- (get_item UY (+ c 1)) (get_item UY (- c 1))) 2.0)
                   (/ (- (get_item UX (+ c W)) (get_item UX (- c W))) 2.0)))
        (set! m (Math.abs w))
        (if (> m 0.075) (set! m 0.075) #f)
        (set! m (/ m 0.075))
        ;; 涡量→冷/暖色，亮度压得很低，作为烟线的底衬
        (if (> w 0.0)
            { (set! r (* m 120.0)) (set! g (* m 34.0)) (set! b (* m 10.0)) }
            { (set! r (* m 6.0)) (set! g (* m 30.0)) (set! b (* m 128.0)) })
        (Gfx.fill_rect (+ GX0 (* (- gx 1) CELL)) (+ GY0 (* (- gy 1) CELL))
                       (+ CELL CELL) (+ CELL CELL) r g b)
        (set! gx (+ gx 1))
      })
      (set! gy (+ gy 1))
    })))

;; ---- 烟线：双线性采样速度场 → 推进 → 加色拖尾 ---------------------------
;; 逐粒子成本很敏感（480 颗粒子 × 每颗几十条指令 + 1 次 native）：这里做了三处抠字：
;;   ① 取整用 `(- x (mod x 1))`（mod 是 VM 内建，比 Math.floor 快一个量级）；
;;   ② 速度模长用曼哈顿近似（|vx|+|vy|），省掉 Math.sqrt；
;;   ③ 不再用"上一帧位置表"（PXV/PYV）：推进前的位置本来就在局部量里，
;;      直接用它画拖尾 —— 一次省 4 次表读写。
(define dye_step
  (lambda ()
    (define i 0)
    (define x 0.0)
    (define y 0.0)
    (define x0 0.0)
    (define y0 0.0)
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
    (define br 0.0)
    (define w00 0.0)
    (define w10 0.0)
    (define w01 0.0)
    (define w11 0.0)
    (set! i 0)
    (while (< i NPD) {
      (set! x (get_item PX i))
      (set! y (get_item PY i))
      (set! x0 x)
      (set! y0 y)
      ;; 屏幕 → 格子连续坐标（格 i 的中心在 (i-0.5)*CELL）
      (set! gx (+ (/ (- x GX0) CELL) 0.5))
      (set! gy (+ (/ (- y GY0) CELL) 0.5))
      (set! i0 (Math.floor gx))
      (set! j0 (Math.floor gy))
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
                  (+ (* w01 (get_item UX (+ (+ c W) 1))) 0.0)))
      (set! vx (+ vx (* w01 (get_item UX (+ c W)))))
      (set! vy (+ (+ (* w00 (get_item UY c)) (* w10 (get_item UY (+ c 1))))
                  (+ (* w11 (get_item UY (+ (+ c W) 1))) (* w01 (get_item UY (+ c W))))))
      ;; 推进（格子单位 → 像素）：一帧走 NSUB 步
      (set! x (+ x (* (* vx CELL) NSUB)))
      (set! y (+ y (* (* vy CELL) NSUB)))
      ;; 出界 / 撞柱 → 回入口重新注入
      (if (or (> x (- (+ GX0 (* NX CELL)) 4.0)) (< y (+ GY0 1.0)) (> y (- (+ GY0 (* NY CELL)) 1.0)))
          { (set! x (+ (+ GX0 2.0) (* (Math.random) 6.0)))
            (set! y (+ (+ GY0 6.0) (* (Math.random) (- (* NY CELL) 12.0))))
            (set! x0 x) (set! y0 y) }
          #f)
      (if (< (+ (* (- x (+ GX0 (* CXC CELL))) (- x (+ GX0 (* CXC CELL))))
                (* (- y (+ GY0 (* CYC CELL))) (- y (+ GY0 (* CYC CELL)))))
             (* (* RAD CELL) (* RAD CELL)))
          { (set! x (+ (+ GX0 2.0) (* (Math.random) 6.0)))
            (set! y (+ (+ GY0 6.0) (* (Math.random) (- (* NY CELL) 12.0))))
            (set! x0 x) (set! y0 y) }
          #f)
      (set_item! PX i x)
      (set_item! PY i y)
      ;; 拖尾：亮度按速度（曼哈顿近似，快=亮）
      ;; 拖尾：不是"这一帧的位移"（只有 2~3px），而是沿速度**反推 4 帧**的彗尾
      ;; （10~15px）—— 这才是风洞烟线（streakline）的观感，且不增加任何调用次数。
      (set! sp (+ (Math.abs vx) (Math.abs vy)))
      (set! br (+ 0.55 (* 14.0 sp)))
      (if (> br 1.0) (set! br 1.0) #f)
      (Gfx.draw_line_aa (- x (* (* vx CELL) (* NSUB 4.0)))
                        (- y (* (* vy CELL) (* NSUB 4.0)))
                        x y 1.0 (* 150.0 br) (* 200.0 br) (* 240.0 br))
      (set! i (+ i 1))
    })))

;; ---- 圆柱 ----------------------------------------------------------------
(define draw_cyl
  (lambda ()
    (define cx 0.0)
    (define cy 0.0)
    (define r 0.0)
    (set! cx (+ GX0 (* CXC CELL)))
    (set! cy (+ GY0 (* CYC CELL)))
    (set! r (* RAD CELL))
    (Gfx.fill_circle_aa cx cy (+ r 5.0) 60 90 120 46)
    (Gfx.fill_circle_aa cx cy (+ r 1.5) 150 180 210 120)
    (Gfx.fill_circle_aa cx cy r 24 28 38 255)
    (Gfx.fill_circle_aa (+ cx (* 0.2 r)) (+ cy (* 0.15 r)) (* 0.55 r) 40 46 60 255)
    (Gfx.fill_circle_aa (- cx (* 0.38 r)) (- cy (* 0.4 r)) (* 0.3 r) 200 215 235 150)))

;; ---- 每拍 ----------------------------------------------------------------
(define tick
  (lambda ()
    (define w0 0)
    (define l0 0)
    (define dt 0)
    (define ns 0)
    (define d0 0)
    (define MSD 0.0)
    (set! w0 (Diag.ms))
    (set! dt (- w0 LASTMS))
    (set! LASTMS w0)
    (if (> dt 200) (set! dt 33) #f)
    (set! T (+ T (/ dt 1000.0)))
    (EX.check)
    (if (Touch.take_press) (set! SHOWVORT (Math.floor (mod (+ SHOWVORT 1) 2))) #f)
    (Gfx.clear)
    (if (== SHOWVORT 1) (draw_vort) #f)
    (set! MAXU 0.0)
    (set! l0 (Diag.ms))
    (set! ns 0)
    (while (< ns NSUB) { (lbm_step) (set! ns (+ ns 1)) })
    (set! MSL (/ (- (Diag.ms) l0) NSUB))
    (set! d0 (Diag.ms))
    (dye_step)
    (set! MSD (- (Diag.ms) d0))
    (draw_cyl)
    (EX.draw_button)
    (Gfx.refresh)
    (set! FRAMES (+ FRAMES 1))
    (set! ACCL (+ ACCL MSL))
    (set! ACCF (+ ACCF (- (Diag.ms) w0)))
    (if (>= FRAMES 30)
        {
          (display "[vx] ms_lbm=") (display (/ ACCL 30.0))
          (display " ms_frm=") (display (/ ACCF 30.0))
          (display " umax=") (display (Math.to_fixed MAXU 3))
          (display " probe_uy=") (display (Math.to_fixed (get_item UY (+ (* 12 W) 24)) 3))
          (display " re=") (display (/ (* (* U0 (+ RAD RAD)) 3.0) (- TAU 0.5)))
          (newline)
          (set! FRAMES 0)
          (set! ACCL 0.0)
          (set! ACCF 0.0)
        }
        #f)))

;; ---- 启动 ----------------------------------------------------------------
(define _timer 0)

(EX.install)
(build_grid)
(spawn_dye)
(set! LASTMS (Diag.ms))
(Gfx.clear)
(draw_cyl)
(EX.draw_button)
(Gfx.refresh)
(set! _timer (System.set_interval 33 tick))
