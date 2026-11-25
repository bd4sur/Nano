;@id=aurora
;@title=光影画廊
;@version=2026-11-25
;; ============================================================================
;; 光影画廊（aurora）—— 灵机 App：六种「光」的视觉技法样片（320x240 / 30fps）
;;
;; 目的：证明"好看的视觉艺术光影效果"可以在**纯 Scheme + 现有 Gfx 原语**里做出来，
;; 并给出每种技法在 WASM 目标上的实测每拍耗时（headless chromium，2026-11-25）：
;;
;;   fx  名称        技法                                       实测 ms/拍
;;   0   极光幕      逐列竖直光柱（加色抗锯齿折线）              6.5 ~ 7.2
;;   1   光绘流场    760 粒子沿平滑流场漂移，短光丝               9.0 ~ 11.7
;;   2   散景光柱    fill_circle_aa 定值 alpha 光斑 + 加色射线    2.6 ~ 3.9
;;   3   干涉莫尔    两族同间距同心圆环（牛顿环式干涉）           1.4 ~ 1.9
;;   4   万花筒      花瓣折线 × 10 重旋转镜像                    5.2 ~ 6.6
;;   5   焦散光网    逐格标量场 + marching squares 等值线         14.5 ~ 18.9
;;
;; 共同的手法只有四条：
;;   ① **加法混合**（Gfx.draw_line_aa 的 mode=3）＝"自发光"：黑底上叠线越多越亮，
;;     天然得到辉光/光柱/光丝的观感，不需要任何后期处理；
;;   ② **定值 alpha 的圆**（Gfx.fill_circle_aa 的 mode≥4）＝镜头散景/光斑/光晕；
;;   ③ **低分辨率逐格算 + 块量化落屏**（每格一次 fill_rect 或一条等值线）＝把"逐像素
;;     着色"压到解释器算得动的量级（1200~2100 格/帧）；
;;   ④ **全局颜色寄存器 + 无 define 的热路径**（hsv 写 HR/HG/HB，不返回列表）：
;;     避免每帧上千次闭包环境分配与堆对象。
;;
;; 六个样片每 6 秒自动切换，**点按屏幕**立即切换；右上角是统一「退出」按钮。
;; 诊断行（每 30 拍一条，供端到端脚本断言/测速）：
;;   [fx] fx=<0..5> avg_ms=<每拍实际干活时长> ticks=<拍数>
;;
;; 解释器约束（manual.md + doc/AGENTS.md，逐条遵守）：算术**恒二元**（一元取负、
;; 三元加法都会静默污染操作数栈 → 指令执行异常）；无 list/let/map/append；
;; 局部量在 lambda 体首部一次声明、循环里 set!（lambda 体内的 define 每次调用都会
;; 新建闭包环境）；只有 #f 是假；Math.PI 是函数；cond 子句只容一个表达式（多句用 {}）；
;; (cons 0 '()) 的哑元素要 pop；运行期不造字符串（颜色走全局寄存器，数字不上屏）。
;; ============================================================================

(native Gfx)
(native Math)
(native Diag)
(native System)
(native Touch)

(import EX "app/lib/exit.scm")

(define SW (Gfx.width))            ; 320
(define SH (Gfx.height))           ; 240
(define CX (* SW 0.5))
(define CY (* SH 0.5))
(define TICK_MS 33)                ; 30fps 节拍（宿主时间片会自动分帧，长帧不卡死）
(define DWELL 182)                 ; 每效果停留拍数（≈6s）

(define NP 760)                    ; 光绘粒子数
(define NSB 56)                    ; 散景光斑数
(define NRAY 28)                   ; 体积光射线数
(define NK 9)                      ; 万花筒种子笔画数
(define KFOLD 10)                  ; 万花筒镜像重数
(define CGT 53)                    ; 焦散格列数
(define CGR 40)                    ; 焦散格行数
(define CBW 6)                     ; 焦散块宽（px）
(define CBH 6)                     ; 焦散块高（px）

;; ---- 状态 ----------------------------------------------------------------
(define T 0.0)                     ; 累计秒
(define FX 0)                      ; 当前效果
(define TICKS 0)                   ; 当前效果已停留拍数
(define FRAMES 0)
(define ACC 0.0)                   ; 每拍毫秒累加
(define LASTMS 0)

;; ---- 颜色寄存器（避免每次分配三元列表）-----------------------------------
(define HR 0.0)
(define HG 0.0)
(define HB 0.0)

;; (hsv h s v)：h 为色相环（任意实数，自动取模），写 HR/HG/HB（0..255 浮点）。
;; **不在体内写 define**：热路径函数每帧被调上千次，define 会每次新建闭包环境。
(define _h6 0.0)
(define _hi 0.0)
(define _hf 0.0)
(define _p 0.0)
(define _q 0.0)
(define _w 0.0)

(define hsv
  (lambda (h s v)
    (set! _h6 (* (mod (+ h 64.0) 1.0) 6.0))
    (set! _hi (Math.floor _h6))
    (set! _hf (- _h6 _hi))
    (set! _p (* v (- 1.0 s)))
    (set! _q (* v (- 1.0 (* s _hf))))
    (set! _w (* v (- 1.0 (* s (- 1.0 _hf)))))
    (cond ((< _hi 1.0) { (set! HR (* v 255.0)) (set! HG (* _w 255.0)) (set! HB (* _p 255.0)) })
          ((< _hi 2.0) { (set! HR (* _q 255.0)) (set! HG (* v 255.0)) (set! HB (* _p 255.0)) })
          ((< _hi 3.0) { (set! HR (* _p 255.0)) (set! HG (* v 255.0)) (set! HB (* _w 255.0)) })
          ((< _hi 4.0) { (set! HR (* _p 255.0)) (set! HG (* _q 255.0)) (set! HB (* v 255.0)) })
          ((< _hi 5.0) { (set! HR (* _w 255.0)) (set! HG (* _p 255.0)) (set! HB (* v 255.0)) })
          (else       { (set! HR (* v 255.0)) (set! HG (* _p 255.0)) (set! HB (* _q 255.0)) }))))

;; ---- 加色抗锯齿线：k 直接缩放颜色 = 光的强度（加法混合，重叠处自然饱和）---
(define seg
  (lambda (x1 y1 x2 y2 w k)
    ;; 颜色直接给浮点：Gfx 的整型参数在 C 侧按 (int) 截断（正值即 floor），
    ;; 而 k ≤ 1 保证颜色 ≤ 255，不会回绕 —— 于是每条线省掉 3 次 Math.floor 调用。
    (Gfx.draw_line_aa x1 y1 x2 y2 w (* HR k) (* HG k) (* HB k))))

;; ==== E0 极光幕 ===========================================================
(define SX '())
(define SY '())
(define SB '())

(define build_stars
  (lambda ()
    (define lx (cons 0 '()))
    (define ly (cons 0 '()))
    (define lb (cons 0 '()))
    (pop lx) (pop ly) (pop lb)
    (define i 0)
    (while (< i 90) {
      (push lx (Math.floor (* (Math.random) SW)))
      (push ly (Math.floor (* (Math.random) (* SH 0.8))))
      (push lb (+ 40.0 (* (Math.random) 140.0)))
      (set! i (+ i 1))
    })
    (set! SX lx) (set! SY ly) (set! SB lb)))

(define e0_draw
  (lambda (t)
    (define i 0)
    (define x 0.0)
    (define f 0.0)
    (define yb 0.0)
    (define bri 0.0)
    (define y0 0.0)
    (define dy 0.0)
    (Gfx.clear)
    ;; 星点（set_pixel 直写，最便宜的底衬）
    (set! i 0)
    (while (< i 90) {
      (hsv 0.55 0.22 (/ (get_item SB i) 255.0))
      (Gfx.set_pixel (get_item SX i) (get_item SY i)
                     (* HR 0.55) (* HG 0.55) (* HB 0.55))
      (set! i (+ i 1))
    })
    ;; 帘幕：逐列一根竖直光柱
    (set! i 0)
    (while (< i 107) {
      (set! x (* i 3.0))
      (set! f (+ (+ (Math.sin (+ (* x 0.021) (* t 0.80)))
                    (* 0.62 (Math.sin (+ (* x 0.047) (- 0.0 (* t 0.55))))))
                 (* 0.30 (Math.sin (+ (* x 0.091) (* t 1.50))))))
      (set! yb (- 170.0 (* f 58.0)))
      (if (< yb 56.0) (set! yb 56.0) #f)
      (set! bri (+ 0.28 (* 0.60 (+ (* f 0.5) 0.5))))
      (if (> bri 1.0) (set! bri 1.0) #f)
      (set! dy (/ (- yb 6.0) 3.0))
      (set! y0 6.0)
      (hsv 0.760 0.80 1.0)
      (seg x y0 x (+ y0 dy) 1.0 (* bri 0.40))
      (seg (+ x 1.0) y0 (+ x 1.0) (+ y0 dy) 1.0 (* bri 0.18))
      (seg (- x 1.0) y0 (- x 1.0) (+ y0 dy) 1.0 (* bri 0.18))
      (set! y0 (+ y0 dy))
      (hsv 0.545 0.72 1.0)
      (seg x y0 x (+ y0 dy) 1.0 (* bri 0.60))
      (seg (+ x 1.2) y0 (+ x 1.2) (+ y0 dy) 1.0 (* bri 0.26))
      (seg (- x 1.2) y0 (- x 1.2) (+ y0 dy) 1.0 (* bri 0.26))
      (set! y0 (+ y0 dy))
      (hsv 0.340 0.62 1.0)
      (seg x y0 x yb 1.0 (* bri 0.80))
      (seg (+ x 1.5) y0 (+ x 1.5) yb 1.0 (* bri 0.30))
      (seg (- x 1.5) y0 (- x 1.5) yb 1.0 (* bri 0.30))
      ;; 下沿之外再拖一小段极暗的余光（否则幕尾一刀切）
      (hsv 0.300 0.50 1.0)
      (seg x yb x (+ yb 10.0) 1.0 (* bri 0.10))
      (set! i (+ i 1))
    })))

;; ==== E1 光绘流场 =========================================================
(define PX '())
(define PY '())
(define PL '())   ; 寿命（拍）

(define build_parts
  (lambda ()
    (define lx (cons 0 '()))
    (define ly (cons 0 '()))
    (define ll (cons 0 '()))
    (pop lx) (pop ly) (pop ll)
    (define i 0)
    (while (< i NP) {
      (push lx (* (Math.random) SW))
      (push ly (* (Math.random) SH))
      (push ll (Math.floor (+ 120.0 (* (Math.random) 320.0))))
      (set! i (+ i 1))
    })
    (set! PX lx) (set! PY ly) (set! PL ll)))

(define e1_draw
  (lambda (t)
    (define i 0)
    (define x 0.0)
    (define y 0.0)
    (define a 0.0)
    (define nx 0.0)
    (define ny 0.0)
    (define life 0)
    (define wrap 0)
    (Gfx.clear)
    (set! i 0)
    (while (< i NP) {
      (set! x (get_item PX i))
      (set! y (get_item PY i))
      (set! life (- (get_item PL i) 1))
      ;; 流场角：两个低频正弦叠加（近似 curl noise 的走向），外加随 i 的相位错开
      (set! a (+ (+ (Math.sin (+ (* x 0.013) (* t 0.55)))
                    (Math.cos (+ (* y 0.011) (- 0.0 (* t 0.37)))))
                 (* 0.35 (Math.sin (+ (* i 0.19) (* t 0.15))))))
      (set! nx (+ x (* 5.0 (Math.cos (* a 1.5)))))
      (set! ny (+ y (+ (* 5.0 (Math.sin (* a 1.5))) 0.6)))
      (set! wrap 0)
      (if (< nx 0.0) (set! wrap 1) (if (> nx SW) (set! wrap 1) #f))
      (if (< ny 0.0) (set! wrap 1) (if (> ny SH) (set! wrap 1) #f))
      (if (or (== wrap 1) (< life 1))
          { ;; 重投：换个位置重新开始（不让粒子挤进流场的吸引子）
            (set! nx (* (Math.random) SW))
            (set! ny (* (Math.random) SH))
            (set! life (Math.floor (+ 120.0 (* (Math.random) 320.0))))
          }
          { ;; 正常推进：画一条加色线段（短曝光下的光丝）
            (hsv (+ 0.50 (* 0.30 (Math.sin (+ (* a 0.85) (* t 0.25))))) 0.78 1.0)
            (seg x y nx ny 1.0 0.80)
            (seg (+ x 0.6) (+ y 0.6) (+ nx 0.6) (+ ny 0.6) 1.0 0.30)
          })
      (set_item! PX i nx)
      (set_item! PY i ny)
      (set_item! PL i life)
      (set! i (+ i 1))
    })))

;; ==== E2 散景光柱 =========================================================
(define BX '())
(define BY '())
(define BR '())
(define BH '())
(define BA '())

(define build_bokeh
  (lambda ()
    (define lx (cons 0 '())) (define ly (cons 0 '())) (define lr (cons 0 '()))
    (define lh (cons 0 '())) (define la (cons 0 '()))
    (pop lx) (pop ly) (pop lr) (pop lh) (pop la)
    (define i 0)
    (define near 0)
    (while (< i NSB) {
      (set! near (< i 20))            ; 前 20 枚当"前景"：大而亮
      (push lx (* (Math.random) SW))
      (push ly (* (Math.random) SH))
      (if near
          (push lr (+ 9.0 (* (Math.random) 12.0)))
          (push lr (+ 2.5 (* (Math.random) 6.0))))
      (push lh (Math.random))
      (if near
          (push la (+ 70.0 (* (Math.random) 70.0)))
          (push la (+ 30.0 (* (Math.random) 50.0))))
      (set! i (+ i 1))
    })
    (set! BX lx) (set! BY ly) (set! BR lr)
    (set! BH lh) (set! BA la)))

(define e2_draw
  (lambda (t)
    (define i 0)
    (define x 0.0)
    (define y 0.0)
    (define a 0.0)
    (define r 0.0)
    (define sx 0.0)
    (define sy 0.0)
    (define ex 0.0)
    (define ey 0.0)
    (define w 0.0)
    (Gfx.clear)
    ;; 光源：绕画面中心做慢速李萨如运动
    (set! sx (+ CX (* 74.0 (Math.sin (* t 0.31)))))
    (set! sy (+ 92.0 (* 44.0 (Math.cos (* t 0.23)))))
    ;; 体积光：每束 3 条（中心 + 两侧微偏）→ 光柱边缘的软过渡
    (set! i 0)
    (while (< i NRAY) {
      (set! a (+ (* i (/ (* 2.0 (Math.PI)) NRAY)) (* t 0.09)))
      (set! w (+ 0.60 (* 0.40 (Math.sin (+ (* t 0.7) (* i 0.9))))))
      (hsv (+ 0.085 (* 0.05 (Math.sin (+ (* t 0.2) i)))) 0.48 1.0)
      (set! ex (+ sx (* 760.0 (Math.cos a))))
      (set! ey (+ sy (* 760.0 (Math.sin a))))
      (seg sx sy ex ey 1.0 (* 0.40 w))
      (set! ex (+ sx (* 760.0 (Math.cos (+ a 0.007)))))
      (set! ey (+ sy (* 760.0 (Math.sin (+ a 0.007)))))
      (seg sx sy ex ey 1.0 (* 0.22 w))
      (set! ex (+ sx (* 760.0 (Math.cos (- a 0.007)))))
      (set! ey (+ sy (* 760.0 (Math.sin (- a 0.007)))))
      (seg sx sy ex ey 1.0 (* 0.22 w))
      (set! i (+ i 1))
    })
    ;; 散景光斑：fill_circle_aa 的 mode = 定值 alpha，重叠处自然发亮
    (set! i 0)
    (while (< i NSB) {
      (set! x (+ (get_item BX i) (* 30.0 (Math.sin (+ (* t 0.22) (* i 0.7))))))
      (set! y (+ (get_item BY i) (* 18.0 (Math.cos (+ (* t 0.18) (* i 1.1))))))
      (set! r (get_item BR i))
      (hsv (get_item BH i) 0.45 1.0)
      (Gfx.fill_circle_aa x y r HR HG HB (get_item BA i))
      (set! i (+ i 1))
    })
    ;; 光源核心两层光晕
    (Gfx.fill_circle_aa sx sy 30 110 82 46 34)
    (Gfx.fill_circle_aa sx sy 13 235 190 120 80)))

;; ==== E3 干涉莫尔 =========================================================
;; 两组（三组）**同心圆环**，圆心各自缓慢游走，环距 3.6~4.4px：
;; 两族环相交处形成经典的"牛顿环/双源干涉"大尺度条纹（拍频 = 两心连线方向的疏密），
;; 圆心一动，整张干涉图样就流动起来。纯 draw_circle，两百多次 native 调用。
(define e3_rings
  (lambda (cx cy d k)
    (define n 0)
    (define r 0.0)
    (define cnt 0)
    (set! cnt (Math.floor (/ 430.0 d)))
    (set! n 0)
    (while (< n cnt) {
      (set! r (* n d))
      (Gfx.draw_circle cx cy r (* HR k) (* HG k) (* HB k))
      (set! n (+ n 1))
    })))

(define e3_draw
  (lambda (t)
    (define ax 0.0)
    (define ay 0.0)
    (define bx 0.0)
    (define by 0.0)
    (define qx 0.0)
    (define qy 0.0)
    (Gfx.clear)
    ;; 三个游走的圆心（李萨如轨迹，周期互不相同 → 图样不重复）
    (set! ax (+ 104.0 (* 26.0 (Math.sin (* t 0.21)))))
    (set! ay (+ 104.0 (* 20.0 (Math.cos (* t 0.17)))))
    (set! bx (+ 214.0 (* 26.0 (Math.cos (* t 0.15)))))
    (set! by (+ 138.0 (* 22.0 (Math.sin (* t 0.19)))))
    (set! qx (+ 168.0 (* 18.0 (Math.sin (+ (* t 0.13) 2.0)))))
    (set! qy (+ 52.0 (* 18.0 (Math.cos (+ (* t 0.23) 1.0)))))
    (hsv 0.55 0.55 1.0)
    (e3_rings ax ay 3.4 0.44)
    (hsv 0.62 0.55 1.0)
    (e3_rings bx by 3.4 0.44)
    (hsv 0.48 0.50 1.0)
    (e3_rings qx qy 6.0 0.30)))

;; ==== E4 万花筒 ===========================================================
;; 一条"花瓣"= 3 点折线（内端 → 外端 → 回勾），在极坐标里定义；
;; 每次绘制都按当前折叠角旋转 (rot)，并以 sign=±1 做**镜像**（角度取反），
;; 于是 KFOLD 重旋转 × 2 重镜像 = 2*KFOLD 重对称的曼陀罗。
(define e4_petal
  (lambda (rot c0 s0 i t sign dr)
    (define a0 0.0)
    (define a1 0.0)
    (define r0 0.0)
    (define r1 0.0)
    (define r2 0.0)
    (define xa 0.0)
    (define ya 0.0)
    (define xb 0.0)
    (define yb 0.0)
    (define xc 0.0)
    (define yc 0.0)
    (set! a0 (* sign (+ (* i 0.70) (* t 0.34))))
    (set! r0 (+ (+ 24.0 dr) (* 16.0 (Math.sin (+ (* t 0.6) (* i 1.1))))))
    (set! a1 (+ a0 (* sign 0.46)))
    (set! r1 (+ (+ 78.0 dr) (* 28.0 (Math.sin (+ (* t 0.45) (* i 1.37))))))
    (set! r2 (+ (+ 38.0 dr) (* 20.0 (Math.sin (+ (* t 0.8) (* i 0.9))))))
    (set! xa (* r0 (Math.cos a0)))
    (set! ya (* r0 (Math.sin a0)))
    (set! xb (* r1 (Math.cos a1)))
    (set! yb (* r1 (Math.sin a1)))
    (set! xc (* r2 (Math.cos (+ a1 (* sign 0.70)))))
    (set! yc (* r2 (Math.sin (+ a1 (* sign 0.70)))))
    (seg (+ CX (+ (* xa c0) (- 0.0 (* ya s0)))) (+ CY (+ (* xa s0) (* ya c0)))
         (+ CX (+ (* xb c0) (- 0.0 (* yb s0)))) (+ CY (+ (* xb s0) (* yb c0)))
         1.0 0.85)
    (seg (+ CX (+ (* xb c0) (- 0.0 (* yb s0)))) (+ CY (+ (* xb s0) (* yb c0)))
         (+ CX (+ (* xc c0) (- 0.0 (* yc s0)))) (+ CY (+ (* xc s0) (* yc c0)))
         1.0 0.55)))

(define e4_draw
  (lambda (t)
    (define k 0)
    (define i 0)
    (define rot 0.0)
    (define c0 0.0)
    (define s0 0.0)
    (define rr 0.0)
    (define aa 0.0)
    (Gfx.clear)
    (set! k 0)
    (while (< k KFOLD) {
      (set! rot (+ (* k (/ (* 2.0 (Math.PI)) KFOLD)) (* t 0.11)))
      (set! c0 (Math.cos rot))
      (set! s0 (Math.sin rot))
      (set! i 0)
      (while (< i NK) {
        (hsv (+ 0.46 (* 0.12 (Math.sin (+ (* i 0.6) (* t 0.8))))) 0.72 1.0)
        (e4_petal rot c0 s0 i t 1.0 0.0)
        (e4_petal rot c0 s0 i t 1.0 5.0)
        (e4_petal rot c0 s0 i t 1.0 (- 0.0 5.0))
        (hsv (+ 0.55 (* 0.10 (Math.sin (+ (* i 0.5) (* t 0.6))))) 0.60 1.0)
        (e4_petal rot c0 s0 i t (- 0.0 1.0) 0.0)
        (set! i (+ i 1))
      })
      (set! k (+ k 1))
    })
    ;; 外圈 10 折对称的光点
    (set! k 0)
    (while (< k KFOLD) {
      (set! aa (+ (* k (/ (* 2.0 (Math.PI)) KFOLD)) (* t 0.11)))
      (set! rr (+ 106.0 (* 8.0 (Math.sin (+ (* t 1.1) k)))))
      (hsv 0.52 0.30 1.0)
      (Gfx.fill_circle_aa (+ CX (* rr (Math.cos aa))) (+ CY (* rr (Math.sin aa)))
                          4.0 HR HG HB 110)
      (set! k (+ k 1))
    })
    (Gfx.fill_circle_aa CX CY 32 120 140 255 24)
    (Gfx.fill_circle_aa CX CY 13 225 240 255 70)))

;; ==== E5 焦散光网 =========================================================
;; 两遍画法，同一个标量场 c 只算一次：
;;   ① **等离子底**：逐格 fill_rect（8x8 像素块），RGB 直接由场值算出（不走 hsv，
;;      省掉每格 ~20 条 VM 指令）——低饱和的深蓝海面；
;;   ② **焦散光网**：c 的**零等值线**，用 marching squares（四角线性插值）逐格画。
;;      与"遇到异号就沿块边界画直角段"的土办法相比，插值让等值线是**平滑曲线**
;;      而不是台阶状迷宫；每格只在真的穿过等值线时才发一次 native 调用。
;; 代价：40x30 = 1200 格 × (2 次 sin + 约 30 条指令) + 约 700 次 draw_line_aa。
;; CPR：上一行各列的 c 值（逐列更新，省掉第二个场缓冲区）。
(define CPR '())

(define build_cpr
  (lambda ()
    (define l (cons 0 '()))
    (pop l)
    (define i 0)
    (while (< i CGT) { (push l 1.0) (set! i (+ i 1)) })
    (set! CPR l)))

(define e5_draw
  (lambda (t)
    (define gy 0)
    (define gx 0)
    (define sv 0.0)
    (define c 0.0)
    (define left 0.0)
    (define up 0.0)
    (define upleft 0.0)
    (define b 0.0)
    (define px 0.0)
    (define py 0.0)
    (define xn 0)
    (define x1 0.0)
    (define y1 0.0)
    (define x2 0.0)
    (define y2 0.0)
    (Gfx.clear)
    (set! gy 0)
    (while (< gy CGR) {
      (set! sv (Math.sin (+ (* gy 0.40) (- 0.0 (* t 0.72)))))
      (set! gx 0)
      (set! left 1.0)
      (set! upleft 1.0)
      (while (< gx CGT) {
        (set! c (+ (+ (Math.sin (+ (* gx 0.46) (* t 0.88))) sv)
                   (Math.sin (+ (* (+ gx gy) 0.21) (* t 0.40)))))
        (set! px (* gx CBW))
        (set! py (* gy CBH))
        ;; ① 等离子底（b∈[0,1]）
        (set! b (* (+ c 2.4) 0.21))
        (if (< b 0.0) (set! b 0.0) #f)
        (if (> b 1.0) (set! b 1.0) #f)
        (Gfx.fill_rect px py CBW CBH (+ 8.0 (* b 66.0)) (+ 12.0 (* b 104.0)) (+ 30.0 (* b 170.0)))
        ;; ② marching squares：四角 A(左上) B(右上) C(右下=本格) D(左下)
        (set! up (get_item CPR gx))
        (set! xn 0)
        (set! x1 0.0) (set! y1 0.0) (set! x2 0.0) (set! y2 0.0)
        ;;   上边 A-B：x = px + CBW*A/(A-B)（异号时商恒在 (0,1)，不需要保护）
        (if (< (* upleft up) 0.0)
            {
              (if (== xn 0) { (set! x1 (+ px (* CBW (/ upleft (- upleft up))))) (set! y1 py) }
                            { (set! x2 (+ px (* CBW (/ upleft (- upleft up))))) (set! y2 py) })
              (set! xn (+ xn 1))
            }
            #f)
        ;;   右边 B-C：y = py + CBH*B/(B-C)
        (if (< (* up c) 0.0)
            {
              (if (== xn 0) { (set! x1 (+ px CBW)) (set! y1 (+ py (* CBH (/ up (- up c))))) }
                            { (set! x2 (+ px CBW)) (set! y2 (+ py (* CBH (/ up (- up c))))) })
              (set! xn (+ xn 1))
            }
            #f)
        ;;   下边 D-C：x = px + CBW*D/(D-C)
        (if (< (* left c) 0.0)
            {
              (if (== xn 0) { (set! x1 (+ px (* CBW (/ left (- left c))))) (set! y1 (+ py CBH)) }
                            { (set! x2 (+ px (* CBW (/ left (- left c))))) (set! y2 (+ py CBH)) })
              (set! xn (+ xn 1))
            }
            #f)
        ;;   左边 A-D：y = py + CBH*A/(A-D)
        (if (< (* upleft left) 0.0)
            {
              (if (== xn 0) { (set! x1 px) (set! y1 (+ py (* CBH (/ upleft (- upleft left))))) }
                            { (set! x2 px) (set! y2 (+ py (* CBH (/ upleft (- upleft left))))) })
              (set! xn (+ xn 1))
            }
            #f)
        ;; 两个交点 → 一段等值线（四点全穿是罕见的鞍点，取前两点即可）
        (if (>= xn 2)
            {
              (set! b (+ 0.55 (* 0.40 (Math.sin (+ (+ (* gy 0.31) (* gx 0.19)) (* t 0.9))))))
              (hsv (+ 0.50 (* 0.06 b)) 0.55 1.0)
              (seg x1 y1 x2 y2 1.0 b)
            }
            #f)
        (set_item! CPR gx c)
        (set! upleft up)
        (set! left c)
        (set! gx (+ gx 1))
      })
      (set! gy (+ gy 1))
    })))

;; ==== 效果调度 ============================================================
(define draw_fx
  (lambda ()
    (cond ((== FX 0) (e0_draw T))
          ((== FX 1) (e1_draw T))
          ((== FX 2) (e2_draw T))
          ((== FX 3) (e3_draw T))
          ((== FX 4) (e4_draw T))
          (else      (e5_draw T)))))

(define hud
  (lambda ()
    (cond ((== FX 0) (Gfx.draw_text "极光幕" 6 222 150 255 215))
          ((== FX 1) (Gfx.draw_text "光绘流场" 6 222 150 255 215))
          ((== FX 2) (Gfx.draw_text "散景光柱" 6 222 150 255 215))
          ((== FX 3) (Gfx.draw_text "莫尔干涉" 6 222 150 255 215))
          ((== FX 4) (Gfx.draw_text "万花筒" 6 222 150 255 215))
          (else      (Gfx.draw_text "焦散光网" 6 222 150 255 215)))))

;; 切到下一效果：清计时并立即画一帧（避免切换瞬间停在旧画面上）
(define next_fx
  (lambda ()
    (set! FX (Math.floor (mod (+ FX 1) 6)))
    (set! TICKS 0)
    (draw_fx)
    (hud)))

;; ---- 每拍 ----------------------------------------------------------------
(define tick
  (lambda ()
    (define now 0)
    (define dt 0.0)
    (define w0 0)
    (set! w0 (Diag.ms))
    (set! now w0)
    (set! dt (- now LASTMS))
    (set! LASTMS now)
    (if (> dt 200.0) (set! dt 33.0) #f)
    (set! T (+ T (/ dt 1000.0)))
    (EX.check)
    (if (Touch.take_press) (next_fx) #f)
    (draw_fx)
    (hud)
    (EX.draw_button)
    (Gfx.refresh)
    (set! TICKS (+ TICKS 1))
    ;; 计时口径：**本拍实际干活的时间**（进入 tick 到推屏结束），不含定时器等待
    (set! ACC (+ ACC (- (Diag.ms) w0)))
    (set! FRAMES (+ FRAMES 1))
    (if (>= FRAMES 30)
        {
          (display "[fx] fx=") (display FX)
          (display " avg_ms=") (display (/ ACC 30.0))
          (display " ticks=") (display TICKS)
          (newline)
          (set! FRAMES 0)
          (set! ACC 0.0)
        }
        #f)
    (if (>= TICKS DWELL) (next_fx) #f)))

;; ---- 启动 ----------------------------------------------------------------
(define _timer 0)

(EX.install)
(build_stars)
(build_parts)
(build_bokeh)
(build_cpr)
(set! LASTMS (Diag.ms))
(draw_fx)
(hud)
(EX.draw_button)
(Gfx.refresh)
(set! _timer (System.set_interval TICK_MS tick))
