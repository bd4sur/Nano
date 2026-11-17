;; ============================================================================
;; 反应-扩散（Gray-Scott）—— 人工生命合集 emergence.scm 的「反应扩散」页
;;
;; 本文件原本是独立 App（app/grayscott.scm）。2026-11 并入 app/emergence.scm 后
;; **降格为库**：不装定时器、不推屏、不管退出按钮；对外只有
;;   init（一次性初始化）/ tick（推进一个仿真步）/ draw（整屏重绘）/ budget（时间片）。
;; 状态（Acur/Bcur/step_no/preset）留在模块里，切页不重置。
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库 + Diag 本地库
;;
;; 模型（Gray-Scott 双组分反应-扩散）：
;;     ∂A/∂t = Da·∇²A − A·B² + f·(1−A)
;;     ∂B/∂t = Db·∇²B + A·B² − (k+f)·B
;;   A 是"养料"、B 是"催化剂"；A·B² 是自催化反应项，f 是补给率、k 是消亡率。
;;   同一套方程在 (f,k) 平面上能长出完全不同的花纹。本页用 5 组实测存活的
;;   参数轮播，每组跑 PRESET_TICKS 拍后换参数并重新播种：
;;     (0.0545, 0.062) 珊瑚 · (0.058, 0.065) 蠕虫 · (0.039, 0.058) 空洞
;;     (0.062, 0.0609) 滑翔子 · (0.026, 0.051) 波纹
;;   （(f,k) 必须落在"自催化能点燃"的区域：B 的初值太小会被扩散稀释到熄灭阈值
;;     k+f 以下，所以播种用 8×8 的大种点而不是小点，否则低 f 的参数直接死掉。）
;;
;; 数值细节：
;;   · 网格 53×32（每格 6×6 像素 = 318×192，铺满页签栏以上的内容区），环形边界
;;     （用 mod 取邻格下标）；
;;   · ∇² 用 9 点模板：0.2×(上+下+左+右) + 0.05×(四角) − 1.0×自身；
;;   · 显式欧拉一步 dt=1.0，Da=1.0 / Db=0.5（经典参数），结果裁到 [0,1]；
;;   · 双缓冲：A1/A2、B1/B2 四张扁平表轮换，每步只 set_item! 覆写，无逐帧分配。
;;
;; 规模与节拍：原独立 App 是 64×48 / 40ms 一拍（实测 wasm 8~12fps）。并入合集后
;; 网格收到 53×32（格数 −45%，因为内容区从 240px 高降到 192px），节拍由宿主按
;; “每 6 个宿主帧一步”给（≈10 步/秒，与原 App 的实际推进速度同量级）。
;;
;; 书写约定（本解释器的坑，详见 app/turtle.scm 头部）：算术一律二元嵌套；
;; cond 子句只能带一个表达式且子句多了会编译失败，故全部用平铺 if；
;; (cons 0 '()) 建出的表含一个哑元素，须先 pop 掉。
;; ============================================================================

(native Gfx)
(native Math)
(native String)

;; ---- 网格与参数 ------------------------------------------------------------

(define W (Gfx.width))            ; 320
(define H (Gfx.height))           ; 240
;; TAB_H：宿主 emergence.scm 底部页签栏高度（18）。内容区 = HUD_TOP..H-TAB_H = 192px，
;; 故 32 行 × 6px = 192px 正好铺满，页签栏底下不再算格子（两者的高度必须一致）。
(define TAB_H 18)
(define GW 53)                    ; 网格列数（53×6 = 318px，右侧留 2px）
(define GH 32)                    ; 网格行数（32×6 = 192px = 内容区高度）
(define PX 6)                     ; 每格边长（像素）
(define CELLS (* GW GH))
(define HUD_TOP 30)
(define PRESET_TICKS 600)         ; 每组参数持续的 tick 数（@10 步/秒 ≈ 60 s 一组）

(define Da 1.0)
(define Db 0.5)
(define DT 1.0)
;; 本页需要的单帧时间片（宿主的 (App.set_timeslice budget) 取这个值）。本页一步较重
;; （1696 格 × 18 次邻格读取），取宿主默认值 312500：一步会被拆到相邻几帧里跑完，
;; 与它作为独立 App 时的行为一致（原版也没申请更大的时间片）。
(define budget 312500)

;; ---- 参数组（(f,k) 平面上的人工生命地貌）-----------------------------------

(define preset 0)
(define pname "珊瑚")
(define f 0.055)
(define k 0.062)
(define pcount 5)

(define apply_preset
  (lambda (i)
    (set! preset i)
    (if (== i 0) { (set! pname "珊瑚")   (set! f 0.0545) (set! k 0.062) })
    (if (== i 1) { (set! pname "蠕虫")   (set! f 0.058)  (set! k 0.065) })
    (if (== i 2) { (set! pname "空洞")   (set! f 0.039)  (set! k 0.058) })
    (if (== i 3) { (set! pname "滑翔子") (set! f 0.062)  (set! k 0.0609) })
    (if (== i 4) { (set! pname "波纹")   (set! f 0.026)  (set! k 0.051) })))

;; ---- 双缓冲（模块级扁平表）-------------------------------------------------

(define A1 (cons 0 (quote ())))
(define A2 (cons 0 (quote ())))
(define B1 (cons 0 (quote ())))
(define B2 (cons 0 (quote ())))
(define Acur 0) (define Bcur 0) (define Anxt 0) (define Bnxt 0)

(define init_buffers
  (lambda ()
    ;; (cons 0 '()) 建出的是长度 1 的列表（那个 0 是哑元素），先 pop 掉
    (pop A1) (pop A2) (pop B1) (pop B2)
    (while (< (length A1) CELLS) { (push A1 1.0) })
    (while (< (length A2) CELLS) { (push A2 1.0) })
    (while (< (length B1) CELLS) { (push B1 0.0) })
    (while (< (length B2) CELLS) { (push B2 0.0) })
    (set! Acur A1) (set! Bcur B1) (set! Anxt A2) (set! Bnxt B2)))

;; ---- 播种（A=1 养料，B 随机撒点）-------------------------------------------

(define reseed
  (lambda ()
    (define i 0) (define x 0) (define y 0) (define base 0) (define d 0)
    (set! i 0)
    (while (< i CELLS) {
      (set_item! Acur i 1.0)
      (set_item! Bcur i 0.0)
      (set_item! Anxt i 1.0)
      (set_item! Bnxt i 0.0)
      (set! i (+ i 1)) })
    ;; 4 个 8×8 大种点 + 8% 稀薄噪声：大种点能把低 f 的参数"点燃"，
    ;; 噪声让整片同时开始自组织（只撒小点的话，低 f 参数会在扩散稀释中熄灭）
    (set! i 0)
    (while (< i 4) {
      (set! x (Math.floor (* (Math.random) (+ (- GW 8) 0.0))))
      (set! y (Math.floor (* (Math.random) (+ (- GH 8) 0.0))))
      (set! d 0)
      (while (< d 64) {
        (set! base (+ (* (+ y (Math.floor (/ d 8.0))) GW) (+ x (mod d 8))))
        ;; 手册：mod 恒返回浮点；列表下标一律显式取整（浮点下标会被四舍五入，别依赖它）
        (set_item! Bcur (Math.floor (mod base CELLS)) 1.0)
        (set! d (+ d 1)) })
      (set! i (+ i 1)) })
    (set! i 0)
    (while (< i CELLS) {
      (if (< (Math.random) 0.08) { (set_item! Bcur i 0.3) })
      (set! i (+ i 1)) })))

;; ---- 单步反应-扩散 ---------------------------------------------------------

(define sim_step
  (lambda ()
    (define x 0) (define y 0) (define base 0) (define row 0)
    (define yN 0) (define yS 0) (define xW 0) (define xE 0)
    (define aC 0.0) (define bC 0.0) (define s1 0.0) (define s2 0.0)
    (define la 0.0) (define lb 0.0) (define ab2 0.0) (define d 0.0)
    (define ra 0.0) (define rb 0.0) (define kf 0.0)
    (set! kf (+ k f))
    (set! y 0)
    (while (< y GH) {
      (set! row (* y GW))
      (set! yN (* (mod (+ y (- GH 1)) GH) GW))
      (set! yS (* (mod (+ y 1) GH) GW))
      (set! x 0)
      (while (< x GW) {
        (set! xW (mod (+ x (- GW 1)) GW))
        (set! xE (mod (+ x 1) GW))
        (set! base (+ row x))
        (set! aC (get_item Acur base))
        (set! bC (get_item Bcur base))
        ;; ---- A 的 9 点拉普拉斯：0.2×(正十字) + 0.05×(四角) − 1×(自身) ----
        (set! s1 (+ (get_item Acur (+ row xW)) (get_item Acur (+ row xE))))
        (set! s1 (+ s1 (+ (get_item Acur (+ yN x)) (get_item Acur (+ yS x)))))
        (set! s2 (+ (get_item Acur (+ yN xW)) (get_item Acur (+ yN xE))))
        (set! s2 (+ s2 (+ (get_item Acur (+ yS xW)) (get_item Acur (+ yS xE)))))
        (set! la (- (+ (* 0.2 s1) (* 0.05 s2)) aC))
        ;; ---- B 的 9 点拉普拉斯 ----
        (set! s1 (+ (get_item Bcur (+ row xW)) (get_item Bcur (+ row xE))))
        (set! s1 (+ s1 (+ (get_item Bcur (+ yN x)) (get_item Bcur (+ yS x)))))
        (set! s2 (+ (get_item Bcur (+ yN xW)) (get_item Bcur (+ yN xE))))
        (set! s2 (+ s2 (+ (get_item Bcur (+ yS xW)) (get_item Bcur (+ yS xE)))))
        (set! lb (- (+ (* 0.2 s1) (* 0.05 s2)) bC))
        ;; ---- 反应项 A·B² ----
        (set! ab2 (* aC (* bC bC)))
        ;; ---- dA/dt = Da∇²A − A·B² + f(1−A) ----
        (set! d (- (* Da la) ab2))
        (set! d (+ d (* f (- 1.0 aC))))
        (set! ra (+ aC (* DT d)))
        ;; ---- dB/dt = Db∇²B + A·B² − (k+f)B ----
        (set! d (+ (* Db lb) ab2))
        (set! d (- d (* kf bC)))
        (set! rb (+ bC (* DT d)))
        ;; ---- 裁剪到 [0,1] ----
        (if (< ra 0.0) { (set! ra 0.0) })
        (if (> ra 1.0) { (set! ra 1.0) })
        (if (< rb 0.0) { (set! rb 0.0) })
        (if (> rb 1.0) { (set! rb 1.0) })
        (set_item! Anxt base ra)
        (set_item! Bnxt base rb)
        (set! x (+ x 1)) })
      (set! y (+ y 1)) })
    ;; 交换缓冲
    (set! d Acur) (set! Acur Anxt) (set! Anxt d)
    (set! d Bcur) (set! Bcur Bnxt) (set! Bnxt d)))

;; ---- 上屏 ------------------------------------------------------------------

(define render
  (lambda ()
    (define x 0) (define y 0) (define i 0) (define v 0.0)
    (define r 0) (define g 0) (define b 0)
    (set! i 0)
    (set! y 0)
    (while (< y GH) {
      (set! x 0)
      (while (< x GW) {
        (set! v (* (get_item Bcur i) 2.6))
        (if (> v 1.0) { (set! v 1.0) })
        (set! r (Math.floor (+ 8.0 (* v 180.0))))
        (set! g (Math.floor (+ 16.0 (* v 218.0))))
        (set! b (Math.floor (+ 40.0 (* v 200.0))))
        (Gfx.fill_rect (* x PX) (+ HUD_TOP (* y PX)) PX PX r g b)
        (set! i (+ i 1))
        (set! x (+ x 1)) })
      (set! y (+ y 1)) })))

;; ---- HUD -------------------------------------------------------------------

(define ds "")
(define step_no 0)

(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))
(define cat4 (lambda (a b c d) (String.concat a (cat3 b c d))))
(define num (lambda (x) (String.atom_to_string x)))

(define draw_hud
  (lambda ()
    (Gfx.set_font 1)                       ; 12px 抗锯齿字（字模是跨 App 共享的静态量）
    (Gfx.fill_rect 0 0 W HUD_TOP 0 0 0)
    (Gfx.draw_text (cat3 "Gray-Scott · " pname "") 8 4 225 235 245)
    ;; f/k 只显示 4 位小数（atom_to_string 会打出 0.0389999 这种长串，挤占后面的列）
    (set! ds (cat4 "f=" (num (/ (Math.floor (* f 10000.0)) 10000.0)) " k=" (num (/ (Math.floor (* k 10000.0)) 10000.0))))
    (Gfx.draw_text ds 8 18 130 165 195)
    (set! ds (cat3 "步" (num step_no) ""))
    (Gfx.draw_text ds 210 18 130 165 195)))

;; ---- 对外入口（宿主 App 只调这几个 + init/budget）--------------------------

;; 整屏重绘（不推进仿真）：清屏 → 反应-扩散场 → 顶栏
(define paint
  (lambda () {
    (Gfx.clear)
    (render)
    (draw_hud)
  }))

;; 一拍：推进一个仿真步，每 PRESET_TICKS 拍换一组 (f,k) 并重新播种，然后重绘
(define tick
  (lambda () {
    (sim_step)
    (set! step_no (+ step_no 1))
    (if (>= step_no (* PRESET_TICKS (+ preset 1)))
        { (apply_preset (mod (+ preset 1) pcount))
          (reseed) })
    (if (> step_no 100000) { (set! step_no 0) })
    (paint)
    (if (< (count_live) 8)
        (set! dead_ticks (+ dead_ticks 1))
        (set! dead_ticks 0))
    (if (> dead_ticks 6)
        { (set! dead_ticks 0)
          (display "[grayscott] reseed extinct") (newline)
          (reseed)
          (paint) })
  }))

;; 只重绘（切页回来 / 模态框关闭后的补帧）
(define draw
  (lambda () {
    (paint)
  }))

;; ---- 灭绝自愈（合集特有的处理）--------------------------------------------
;; 5 组 (f,k) 都是实测存活过的组合，但播种是随机的：偶尔会点不燃（B 被扩散稀释到
;; 熄灭阈值 k+f 以下就再也不长了）。作为独立 App 时它靠"每组只跑 PRESET_TICKS 拍"
;; 掩盖；合集里加一条看门狗：连续 dead_ticks 拍几乎没有催化剂（B > 0.05）就重新播种。

(define dead_ticks 0)

;; 有催化剂的格数（与 render 的观感阈值一致）
(define count_live
  (lambda ()
    (define i 0) (define n 0)
    (while (< i CELLS) {
      (if (> (get_item Bcur i) 0.05) (set! n (+ n 1)) #f)
      (set! i (+ i 1)) })
    n))

;; 已推进的步数（诊断用，见 alife_gol.scm 的同名条目）
(define steps
  (lambda () step_no))

;; 一次性初始化（宿主启动时调一次，之后状态一直留在本模块里）
(define init
  (lambda () {
    (init_buffers)
    (apply_preset 0)
    (reseed)
    (paint)
  }))

;; ---- 启动：本模块作为库被 import 时**不做任何事**（不装定时器、不刷屏）------
;;   旧版独立 App 在这一行的位置是：(init_buffers) (apply_preset 0) (reseed)
;;   (set! _timer (System.set_interval ...)) (EX.install_with render)。
