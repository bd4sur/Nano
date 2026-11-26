;; ============================================================================
;; frcnn_core.scm —— Faster R-CNN 两阶段检测的**模型与推理内核**（无 UI 依赖）
;;
;; 这是 app/frcnn.scm 的计算部分：图像合成、Backbone、RPN、RoI 池化、分类/回归头、
;; NMS，以及「按步推进」的分块计算驱动。UI 只负责画和点。
;;
;; 拆成独立模块的理由：同一份内核可以被
;;   · App（app/frcnn.scm，Gfx 可视化）；
;;   · 离线交叉校验脚本（tools/e2e/wasm 下的 .scm，纯 display，跑真 wasm32）
;; 复用，从而保证「训练/参照实现/Scheme 推理」三者的数值逐位可比。
;;
;; 网络（与 tools/frcnn_common.py 逐条对应，权重见 app/lib/frcnn_weights.scm）：
;;     输入   1x64x64 灰度，xn = (x - 0.80) / 0.21
;;     conv1  3x3/s2/p1  1→8   ReLU  → 8x32x32        （感受野 3x3）
;;     conv2  3x3/s2/p1  8→24  ReLU  → 24x16x16       （步长 4，感受野 7x7）
;;     conv3  3x3/s1/p1 逐通道 24→24，再 1x1 融合 24→24 ReLU → 24x16x16（感受野 11x11）
;;     RPN    1x1 24→24 ReLU → 分类 1x1 16→9（每格 9 个锚框的一个前景 logit）
;;                             → 回归 1x1 16→36（9 锚框 × 4 个偏移）
;;     锚框   9 个/格 = 尺寸{12,17,24} × 宽高比{0.5,0.7,0.95}，中心在格心（16x16 网格）
;;     RoI    4x4 最大池化（Caffe 量化口径）→ 24x4x4 = 384 维
;;     头     FC 384→64 ReLU → 分类 64→11（0..9 + 背景）/ 回归 64→44（11 类 × 4 偏移）
;;     后处理 提议：分数>0.30 → top100 → NMS(0.5) → 12 个 RoI
;;            最终：取 argmax 类别细化框 → 逐类 NMS(0.35)
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数；没有一元取负（写 (- 0 x)）；
;;   没有 let/list/append/map；define 只在 lambda 体顶层（块内只用 set!）；
;;   / 与 mod 恒返回浮点 → 下标一律 Math.floor/round 收口；'() 字面量是静态槽，
;;   新列表一律 (cdr '(0)) 起手；0 是真值，布尔一律 #t/#f。
;; ============================================================================

(native Math)
(native String)
(native System)

(import WT "app/lib/frcnn_weights.scm")

;; ---- 结构常量（改这里必须同步改 tools/frcnn_common.py） --------------------

(define IMG 64)
(define STRIDE 4)
(define GRID 16)
(define NCH1 8)
(define NCH 24)
(define NA 9)
(define NCLS 11)          ; 0..9 + 背景
(define POOL 4)
(define FDIM (* POOL (* POOL NCH)))   ; RoI 池化后的维数 = 4×4×24 = 384
(define HID 64)
(define NROI 20)
(define PRE_TOPK 100)
(define SCORE_TH 0.30)
(define RPN_IOU 0.7)
(define FIN_IOU 0.35)
(define MEAN 0.80)          ; 归一化均值/标准差（训练侧统计，见 tools/frcnn_common.py）
(define STD 0.21)

;; 锚框尺寸（sqrt 面积）与宽高比 w/h
(define ASZ '(12.0 17.0 24.0))
(define ART '(0.5 0.7 0.95))

(define NANCH (* GRID (* GRID NA)))       ; 2304
(define NF1 (* NCH1 (* 32 32)))           ; 8192
(define NF2 (* NCH 256))              ; 4096
(define NIMG (* IMG IMG))             ; 4096

;; ---- 小工具 ---------------------------------------------------------------

(define i2s (lambda (x) (String.atom_to_string x)))
(define maxf (lambda (a b) (if (> a b) a b)))
(define minf (lambda (a b) (if (< a b) a b)))
(define clampf (lambda (v lo hi) (if (< v lo) lo (if (> v hi) hi v))))
(define absf (lambda (x) (if (< x 0.0) (- 0.0 x) x)))

(define zeros
  (lambda (n)
      (define acc (cdr '(0)))
      (define i 0)
    {
    (while (< i n) {
      (push acc 0.0)
      (set! i (+ i 1))
    })
    acc
  }))

;; 把 [0,1) 的随机数变成 [lo,hi] 的浮点
(define rnd (lambda (lo hi) (+ lo (* (- hi lo) (Math.random)))))

;; ---- 权重的解码与反量化 ---------------------------------------------------
;;
;; frcnn_weights.scm 里权重是一个字符串：每字符 6 bit（码点 33+v），4 字符 → 3 字节，
;; 大端位序。这里顺序读出 int8 再乘该输出通道的尺度，得到 float 列表（行主序）。
;; 解码是一次性的（App 启动时 ~2 万次循环），推理热路径上不再付这个代价。

(define wq_pos 0)      ; 当前字符下标
(define wq_v 0)        ; 当前 24 bit 组
(define wq_left 0)     ; 当前组还剩几个字节

(define qb 0)
(define qb0 0)
(define qb1 0)
(define qb2 0)

;; 读一个字节（0..255）。4 个 6bit 字符 = 3 字节，大端位序。
;;
;; **不要**用 (/ wq_v (pow 2 k)) 那种浮点除法取字节：32 位宿主上解释器的浮点只有约
;; 19 位尾数，24 bit 的累加值转换成浮点就会丢低位（实测每组的第 3 个字节全错，权重
;; 反量化出来差 30%）。这里改成**只用 ≤255 的小整数**逐字节拆位（4 个 6bit 字符 →
;; 3 个字节），每一步的中间值都不超过 255，浮点表示精确。
(define next_byte
  (lambda () {
    (define c0 0)
    (define c1 0)
    (define c2 0)
    (define c3 0)
    {
    (if (== wq_left 0)
        {
          (set! c0 (- (String.charCodeAt wq_pos WT.WQ) 33))
          (set! c1 (- (String.charCodeAt (+ wq_pos 1) WT.WQ) 33))
          (set! c2 (- (String.charCodeAt (+ wq_pos 2) WT.WQ) 33))
          (set! c3 (- (String.charCodeAt (+ wq_pos 3) WT.WQ) 33))
          (set! qb0 (+ (* c0 4) (Math.floor (/ c1 16.0))))
          (set! qb1 (+ (* (- c1 (* 16.0 (Math.floor (/ c1 16.0)))) 16.0)
                       (Math.floor (/ c2 4.0))))
          (set! qb2 (+ (* (- c2 (* 4.0 (Math.floor (/ c2 4.0)))) 64.0) c3))
          (set! wq_pos (+ wq_pos 4))
          (set! wq_left 2)
          (set! qb qb0)
        }
        {
          (set! wq_left (- wq_left 1))
          (set! qb (if (== wq_left 1) qb1 qb2))
        })
    qb}
  }))

;; 一个 int16 权重（大端两字节；对称量化，正负各 32767）
(define next_q
  (lambda () {
    (define hi (next_byte))
    (define lo (next_byte))
    (define v (+ (* hi 256) lo))
    (if (> v 32767.0) (- v 65536.0) v)}))

;; 读一层：nch 个输出通道 × 每通道 per 个权重，按通道尺度反量化
(define load_layer
  (lambda (idx)
      (define ent (get_item WT.WMETA idx))
      (define nch (get_item ent 1))
      (define per (get_item ent 2))
      (define sc (get_item ent 3))
      (define acc (cdr '(0)))
      (define c 0)
      (define j 0)
      (define s 1.0)
    {
    (while (< c nch) {
      (set! s (get_item sc c))
      (set! j 0)
      (while (< j per) {
        (push acc (* (next_q) s))
        (set! j (+ j 1))
      })
      (set! c (+ c 1))
    })
    acc
  }))

;; 权重（顺序与 tools/train_frcnn_digits.py 的 export_scheme 一致）
(define W1 (load_layer 0))         ; conv1   (8,1,3,3)
(define W2 (load_layer 1))         ; conv2   (16,8,3,3)
(define DW_W (load_layer 2))        ; conv3 3x3 逐通道 (24,3,3)
(define PW_W (load_layer 3))        ; conv3 1x1 融合  (24,24)
(define WRH (load_layer 4))        ; rpn_conv(24,24,1,1)
(define WRC (load_layer 5))        ; rpn_cls (9,24)
(define WRR (load_layer 6))        ; rpn_reg (36,24)
(define WF1 (load_layer 7))        ; fc1     (64,384)
(define WCL (load_layer 8))        ; cls     (11,64)
(define WRG (load_layer 9))        ; reg     (44,64)

(define B1 WT.CONV1_B)
(define B2 WT.CONV2_B)
(define DW_B WT.DW_B)
(define PW_B WT.PW_B)
(define BRH WT.RPN_CONV_B)
(define BRC WT.RPN_CLS_B)
(define BRR WT.RPN_REG_B)
(define BF1 WT.FC1_B)
(define BCL WT.CLS_B)
(define BRG WT.REG_B)

;; ---- 输入图像 -----------------------------------------------------------
;;
;; img：当前 64x64 灰度输入（**0=墨，1=纸**）。UI 的画笔直接改它；
;; 「示例」按钮用 synth_sample 现场**程序化合成**一张（笔画矢量 + 随机布局），
;; 不依赖任何内嵌图像资源。

(define img (zeros NIMG))
(define xn (zeros NIMG))          ; 归一化后的网络输入（显示用）
(define xpad (zeros (* 66 66)))   ; 补 1px 边（0）的输入，卷积取址不必判边界
(define gt_box (zeros 24))        ; 合成图的真值框（最多 6 个 ×4）
(define gt_lab (zeros 8))
(define gt_n 0)

;; 清空图像与全部中间结果
(define clear_image
  (lambda ()
      (define i 0)
    {
    (while (< i NIMG) {
      (set_item! img i 1.0)
      (set_item! xn i 0.0)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i (* 66 66)) {
      (set_item! xpad i 0.0)
      (set! i (+ i 1))
    })
    (set! gt_n 0)
  }))

;; 往 img 上盖一笔（UI 的画笔；v = 墨强度 0..1），坐标是图像像素
(define stamp
  (lambda (px py r v)
      (define i0 (Math.floor (- py r)))
      (define i1 (Math.ceil (+ py r)))
      (define j0 (Math.floor (- px r)))
      (define j1 (Math.ceil (+ px r)))
      (define i 0)
      (define j 0)
      (define d 0.0)
      (define g 0.0)
    {
    (if (< i0 0) (set! i0 0) #f)
    (if (< j0 0) (set! j0 0) #f)
    (if (> i1 (- IMG 1)) (set! i1 (- IMG 1)) #f)
    (if (> j1 (- IMG 1)) (set! j1 (- IMG 1)) #f)
    (set! i i0)
    (while (<= i i1) {
      (set! j j0)
      (while (<= j j1) {
        (set! d (Math.sqrt (+ (* (- (+ j 0.5) px) (- (+ j 0.5) px))
                              (* (- (+ i 0.5) py) (- (+ i 0.5) py)))))
        (if (< d r)
            {
              (set! g (- 1.0 (/ d r)))                 ; 软边
              (set! g (* (* g g) v))
              (set! g (- 1.0 g))
              (if (< g (get_item img (+ (* i IMG) j)))
                  (set_item! img (+ (* i IMG) j) g)
                  #f)
            }
            #f)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
  }))

;; ============================================================================
;; 「示例」样本的程序化合成（synth_sample）
;;
;; 第一版把字形 sprite 内嵌进 Scheme 现场采样，采样算式写错、画出来不像数字；
;; 第二版改成离线渲染 6 张整图内嵌（frcnn_samples.scm，27KB）；现在改成**现场合成**：
;;   · 每个数字用一组「笔画矢量」描述（单位方框 0..1，y 向下），
;;     直线段与椭圆两种图元，按 ±5° 旋转、缩放、平移到 64x64 上；
;;   · 用 (stamp) 这枚**软边圆点**沿笔画插值盖章 —— 和手指画板同一支笔，
;;     所以边缘自带抗锯齿，和训练集里的手写体同一种"毛边"质感；
;;   · 纸面纹理、字号、笔画粗细、位置、亮度都由一个确定性 LCG 给出（同一 k 出同一张图）。
;; 好处：没有内嵌资源（省 27KB）、随时能调参、生成过程本身就是"数据增广"的演示。
;; ============================================================================

;; 确定性随机：Lehmer LCG（乘 75 取模 65537，取值 < 2^32，32 位宿主也不会丢精度）
(define sy_st 12345)
(define sy_r 0.0)
(define sy_rnd
  (lambda () {
    (set! sy_st (mod (* sy_st 75.0) 65537.0))
    (set! sy_r (/ sy_st 65537.0))
    sy_r
  }))
(define sy_rndf (lambda (lo hi) (+ lo (* (- hi lo) (sy_rnd)))))
(define sy_rndi (lambda (lo hi) (+ lo (Math.floor (* (+ (- hi lo) 1) (sy_rnd))))))

;; 当前数字的变换参数（单位坐标 → 图像像素）
(define sy_s 20.0)
(define sy_cx 32.0)
(define sy_cy 32.0)
(define sy_ca 1.0)
(define sy_sa 0.0)
(define sy_px 0.0)
(define sy_py 0.0)

(define sy_map
  (lambda (u v) {
    (set! sy_px (+ sy_cx (* sy_s (- (* (- u 0.5) sy_ca) (* (- v 0.5) sy_sa)))))
    (set! sy_py (+ sy_cy (* sy_s (+ (* (- u 0.5) sy_sa) (* (- v 0.5) sy_ca)))))
    0
  }))

;; 合成专用的圆点：半径内**满墨**，外面 1px 线性收边（(stamp) 是二次衰减，
;; 用来画手指笔划很自然，但拿来合成 20px 的数字会细成发丝）。
(define sy_di 0)
(define sy_dj 0)
(define sy_dv 0.0)
(define sy_dg 0.0)
(define sy_dot
  (lambda (px py r v)
      (define i0 (Math.floor (- (- py r) 1.0)))
      (define i1 (Math.ceil (+ (+ py r) 1.0)))
      (define j0 (Math.floor (- (- px r) 1.0)))
      (define j1 (Math.ceil (+ (+ px r) 1.0)))
    {
    (if (< i0 0) (set! i0 0) #f)
    (if (< j0 0) (set! j0 0) #f)
    (if (> i1 (- IMG 1)) (set! i1 (- IMG 1)) #f)
    (if (> j1 (- IMG 1)) (set! j1 (- IMG 1)) #f)
    (set! sy_di i0)
    (while (<= sy_di i1) {
      (set! sy_dj j0)
      (while (<= sy_dj j1) {
        (set! sy_dv (Math.sqrt (+ (* (- (+ sy_dj 0.5) px) (- (+ sy_dj 0.5) px))
                                  (* (- (+ sy_di 0.5) py) (- (+ sy_di 0.5) py)))))
        (if (< sy_dv (+ r 1.0))
            {
              (set! sy_dg (if (< sy_dv r) v (* v (- 1.0 (- sy_dv r)))))
              (set! sy_dg (- 1.0 sy_dg))
              (if (< sy_dg (get_item img (+ (* sy_di IMG) sy_dj)))
                  (set_item! img (+ (* sy_di IMG) sy_dj) sy_dg)
                  #f)
            }
            #f)
        (set! sy_dj (+ sy_dj 1))
      })
      (set! sy_di (+ sy_di 1))
    })
  }))

;; 单位坐标里的一段 → 一串圆点（步长 0.45 像素，断笔风险为 0）
(define sy_x1 0.0)
(define sy_y1 0.0)
(define sy_dx 0.0)
(define sy_dy 0.0)
(define sy_n 0)
(define sy_ci 0)
(define sy_ki 0)
(define sy_ei 0)
(define sy_seg
  (lambda (u1 v1 u2 v2 rad) {
    (sy_map u1 v1)
    (set! sy_x1 sy_px)
    (set! sy_y1 sy_py)
    (sy_map u2 v2)
    (set! sy_dx (- sy_px sy_x1))
    (set! sy_dy (- sy_py sy_y1))
    (set! sy_n (Math.ceil (/ (Math.sqrt (+ (* sy_dx sy_dx) (* sy_dy sy_dy))) 0.4)))
    (if (< sy_n 1) (set! sy_n 1) #f)
    (set! sy_ci 0)
    (while (<= sy_ci sy_n) {
      (sy_dot (+ sy_x1 (/ (* sy_dx sy_ci) sy_n))
              (+ sy_y1 (/ (* sy_dy sy_ci) sy_n)) rad 1.0)
      (set! sy_ci (+ sy_ci 1))
    })
  }))

;; 折线：pts 是 (u1 v1 u2 v2 ...) 的扁平静态表
(define sy_np 0)
(define sy_poly
  (lambda (pts rad) {
      (set! sy_np (Math.floor (/ (length pts) 4)))   ; 每段 4 个数（共享顶点重复给出）
      (set! sy_ki 0)
      (while (< sy_ki sy_np) {
        (sy_seg (get_item pts (* sy_ki 4)) (get_item pts (+ (* sy_ki 4) 1))
                (get_item pts (+ (* sy_ki 4) 2)) (get_item pts (+ (* sy_ki 4) 3)) rad)
        (set! sy_ki (+ sy_ki 1))
      })
  }))

;; 椭圆（闭合）：中心 (cu,cv)、半径 (ru,rv)，单位坐标
(define sy_eu 0.0)
(define sy_ev 0.0)
(define sy_a 0.0)
(define sy_da 0.0)
(define sy_ka 0.0)
(define sy_kb 0.0)
(define sy_steps 26)
(define sy_ell
  (lambda (cu cv ru rv rad) {
      (set! sy_da (/ (* 2.0 (Math.PI)) sy_steps))
      (set! sy_a 0.0)
      (set! sy_ei 0)
      (while (< sy_ei sy_steps) {
        (set! sy_ka (+ sy_a sy_da))
        (sy_seg (+ cu (* ru (Math.cos sy_a))) (+ cv (* rv (Math.sin sy_a)))
                (+ cu (* ru (Math.cos sy_ka))) (+ cv (* rv (Math.sin sy_ka))) rad)
        (set! sy_a sy_ka)
        (set! sy_ei (+ sy_ei 1))
      })
  }))

;; ---- 10 个数字的笔画表（单位方框，y 向下；照着常见印刷/手写混合形状拟的）----
(define sy_L0 '(0.0))
(define sy_L1
  '(0.40 0.26 0.54 0.10   0.54 0.10 0.54 0.94   0.30 0.94 0.76 0.94))
(define sy_L2
  '(0.20 0.26 0.29 0.10   0.29 0.10 0.50 0.05   0.50 0.05 0.71 0.13
    0.71 0.13 0.75 0.31   0.75 0.31 0.62 0.51   0.62 0.51 0.24 0.83
    0.24 0.83 0.17 0.93   0.17 0.93 0.81 0.93))
(define sy_L3
  '(0.22 0.14 0.46 0.06   0.46 0.06 0.70 0.14   0.70 0.14 0.72 0.31
    0.72 0.31 0.50 0.44   0.50 0.44 0.75 0.54   0.75 0.54 0.77 0.77
    0.77 0.77 0.56 0.93   0.56 0.93 0.27 0.93   0.27 0.93 0.21 0.86))
(define sy_L4
  '(0.67 0.93 0.67 0.08   0.67 0.08 0.15 0.64   0.15 0.64 0.86 0.64))
(define sy_L5
  '(0.79 0.08 0.28 0.08   0.28 0.08 0.24 0.44   0.24 0.44 0.47 0.38
    0.47 0.38 0.71 0.47   0.71 0.47 0.77 0.71   0.77 0.71 0.60 0.91
    0.60 0.91 0.34 0.94   0.34 0.94 0.22 0.86))
(define sy_L6
  '(0.72 0.18 0.58 0.08   0.58 0.08 0.38 0.10   0.38 0.10 0.25 0.28
    0.25 0.28 0.22 0.56   0.22 0.56 0.29 0.79   0.29 0.79 0.44 0.91
    0.44 0.91 0.64 0.87   0.64 0.87 0.75 0.71   0.75 0.71 0.71 0.55
    0.71 0.55 0.52 0.46   0.52 0.46 0.31 0.54   0.31 0.54 0.24 0.69))
(define sy_L7
  '(0.18 0.08 0.82 0.08   0.82 0.08 0.46 0.94))
(define sy_L8 '(0.0))
(define sy_L9 '(0.0))

;; 单个数字落到 img 上（k = 0..9）
(define sy_digit
  (lambda (k rad) {
    (if (== k 0) (sy_ell 0.50 0.50 0.30 0.43 rad) #f)
    (if (== k 1) (sy_poly sy_L1 rad) #f)
    (if (== k 2) (sy_poly sy_L2 rad) #f)
    (if (== k 3) (sy_poly sy_L3 rad) #f)
    (if (== k 4) (sy_poly sy_L4 rad) #f)
    (if (== k 5) (sy_poly sy_L5 rad) #f)
    (if (== k 6) (sy_poly sy_L6 rad) #f)
    (if (== k 7) (sy_poly sy_L7 rad) #f)
    (if (== k 8)
        {
          (sy_ell 0.50 0.26 0.23 0.20 rad)
          (sy_ell 0.50 0.73 0.27 0.23 rad)
        }
        #f)
    (if (== k 9)
        {
          (sy_ell 0.48 0.29 0.25 0.21 rad)
          (sy_poly '(0.73 0.29 0.73 0.64  0.73 0.64 0.65 0.88  0.65 0.88 0.45 0.94  0.45 0.94 0.32 0.88) rad)
        }
        #f)
  }))

;; 纸面：底 1.0（白），叠 0.03 幅度的纹理
(define sy_v 0.0)
(define sy_pi 0)
(define sy_paper
  (lambda () {
    (set! sy_pi 0)
    (while (< sy_pi NIMG) {
      (set! sy_v (- 1.0 (* 0.030 (sy_rnd))))
      (set_item! img sy_pi sy_v)
      (set! sy_pi (+ sy_pi 1))
    })
  }))

(define sy_cnt 1)
(define sy_j 0)
(define sy_lab 0)
(define sy_rad 0.0)
(define sy_hei 0.0)
(define sy_rot 0.0)

;; 合成第 k 张（k 决定随机种子：同一 k 永远同一张图；不同 k 布局/数字都不同）
(define synth_sample
  (lambda (k) {
    (set! sy_st (mod (+ 12345 (* 7919.0 k)) 65537.0))
    (sy_paper)
    ;; 1~3 个数字（训练集是 1~6，示例取少一点，看得清）
    (set! sy_cnt (+ 1 (sy_rndi 0 2)))
    (set! sy_j 0)
    (while (< sy_j sy_cnt) {
      (set! sy_lab (sy_rndi 0 9))
      ;; 字号 15~25px：与训练集的 13~28px 对齐（3 个数字时小一号，免得挤成一团）
      (if (== sy_cnt 3)
          (set! sy_hei (sy_rndf 14.0 19.0))
          (set! sy_hei (sy_rndf 16.0 25.0)))
      (set! sy_rad (maxf 0.9 (* sy_hei 0.058)))
      (set! sy_rot (* (- (sy_rnd) 0.5) 0.175))   ; ±5°
      (set! sy_ca (Math.cos sy_rot))
      (set! sy_sa (Math.sin sy_rot))
      (set! sy_s sy_hei)
      ;; 横向按个数均分（1 个居中、2 个左右、3 个三列），再抖 ±2.5px
      (set! sy_cx (+ (/ (* 64.0 (+ sy_j 1)) (+ sy_cnt 1)) (sy_rndf (- 0.0 2.5) 2.5)))
      (set! sy_cy (sy_rndf 28.0 38.0))
      (if (< sy_cx (* 0.5 sy_hei)) (set! sy_cx (* 0.5 sy_hei)) #f)
      (if (> sy_cx (- IMG (* 0.5 sy_hei))) (set! sy_cx (- IMG (* 0.5 sy_hei))) #f)
      (sy_digit sy_lab sy_rad)
      (set! sy_j (+ sy_j 1))
    })
    (set! gt_n 0)
  }))

;; 从 ASCII 字符串装入图像（每个字符 = 一像素，码点 33+v，v∈[0,63] 表示 0..1）
;; —— 离线交叉校验用（Python 侧把同一张图编码成同样的串）。
(define set_from_ascii
  (lambda (s)
      (define i 0)
    {
    (while (< i NIMG) {
      (set_item! img i (/ (- (String.charCodeAt i s) 33) 63.0))
      (set! i (+ i 1))
    })
    (set! gt_n 0)
  }))

;; ---- 中间结果缓冲 ---------------------------------------------------------

(define f1 (zeros NF1))            ; conv1 激活 8x32x32
(define p1 (zeros (* NCH1 (* 34 34)))) ; 补边后的 conv1（卷积2 取址不判边界）
(define f2 (zeros NF2))            ; conv2 激活 24x16x16
(define p2 (zeros (* NCH (* 18 18))))  ; 补边后的 conv2（conv3 的 3x3 逐通道取址不判边界）
(define f3dw (zeros NF2))          ; conv3 的 3x3 逐通道卷积结果
(define f3 (zeros NF2))            ; conv3 激活 24x16x16（= 检测特征图）
(define rh (zeros NF2))            ; RPN 中间层（1x1 卷积后）16x16x16
(define objl (zeros (* NA 256)))   ; RPN 前景 logit（每格 9 个）
(define objp (zeros (* NA 256)))   ; 前景概率 sigmoid
(define rreg (zeros (* NA (* 4 256)))) ; RPN 框偏移（每格 9 个锚框 × 4）
(define anch (zeros (* NANCH 4)))  ; 全部锚框坐标（预计算一次）

;; 提议（top-k 表 + NMS 后的 RoI）
(define tk_s (zeros PRE_TOPK))
(define tk_b (zeros (* PRE_TOPK 4)))
(define tk_a (zeros PRE_TOPK))
(define tk_n 0)

(define roi_n 0)
(define roi_b (zeros (* NROI 4)))
(define roi_s (zeros NROI))

;; 头部的中间结果
(define pooled (zeros (* NROI FDIM)))
(define fch (zeros (* NROI HID)))
(define clsl (zeros (* NROI NCLS)))
(define clsp (zeros (* NROI NCLS)))
(define hreg (zeros (* NROI (* NCLS 4))))

;; 最终结果：det_* 是逐 RoI 的候选，fin_* 是逐类 NMS 后的最终检测
(define DET_MAX 64)
(define det_n 0)
(define det_cls (zeros DET_MAX))
(define det_sc (zeros DET_MAX))
(define det_b (zeros (* DET_MAX 4)))
(define fin_n 0)
(define fin_cls (zeros DET_MAX))
(define fin_sc (zeros DET_MAX))
(define fin_b (zeros (* DET_MAX 4)))

;; 进度（0..1）与计时
(define last_ms 0)

;; ---- 锚框预计算 -----------------------------------------------------------
;; 索引约定：a = (gy*16+gx)*9 + k —— 与 Python 的 make_anchors().reshape(-1,4) 一致。

(define anch_init
  (lambda ()
      (define k 0)
      (define gy 0)
      (define gx 0)
      (define s 0.0)
      (define r 0.0)
      (define w 0.0)
      (define h 0.0)
      (define cx 0.0)
      (define cy 0.0)
      (define a 0)
    {
    (while (< k NA) {
      (set! s (get_item ASZ (Math.floor (/ k 3.0))))
      (set! r (get_item ART (mod k 3)))
      (set! w (* s (Math.sqrt r)))
      (set! h (/ s (Math.sqrt r)))
      (set! gy 0)
      (while (< gy GRID) {
        (set! gx 0)
        (while (< gx GRID) {
          (set! cx (+ (* gx STRIDE) (/ STRIDE 2.0)))
          (set! cy (+ (* gy STRIDE) (/ STRIDE 2.0)))
          (set! a (+ (* (+ (* gy GRID) gx) NA) k))
          (set_item! anch (+ (* a 4) 0) (- cx (/ w 2.0)))
          (set_item! anch (+ (* a 4) 1) (- cy (/ h 2.0)))
          (set_item! anch (+ (* a 4) 2) (+ cx (/ w 2.0)))
          (set_item! anch (+ (* a 4) 3) (+ cy (/ h 2.0)))
          (set! gx (+ gx 1))
        })
        (set! gy (+ gy 1))
      })
      (set! k (+ k 1))
    })
  }))

;; ---- 每层输出的「单元」计算 ----------------------------------------------
;; 一个「单元」= 一个 (输出通道, 空间位置) 的输出值。分块推进的粒度就是单元数，
;; UI 每拍给定预算，超了就留到下一拍——这样即使整网要 1~2 秒，画面也不会僵住。

;; 阶段 0：输入归一化 + 补边
(define u_norm
  (lambda (i)
      (define x (mod i IMG))
      (define y (Math.floor (/ i IMG)))
      ;; 训练侧约定：图像按 **uint8 量纲**（0..255，即 v*255）做 (v/255 - 0.80)/0.21。
    ;; 本模块里 img 存的是 [0,1]，所以要先把像素折回 uint8 量纲再归一化——
    ;; 少这一除会让输入对比度大 255 倍，卷积输出整体跑飞（实测 f1 和差 100 倍）。
    (define v (/ (- (/ (get_item img i) 255.0) MEAN) STD))
    {
    (set_item! xn i v)
    (set_item! xpad (+ (* (+ y 1) 66) (+ x 1)) v)
  }))

;; 阶段 1：conv1 —— 8 通道 × 32×32，3x3 步长 2 补边 1 + ReLU
(define u_conv1
  (lambda (i)
      (define k (Math.floor (/ i 1024.0)))
      (define rc (- i (* k 1024)))
      (define r (Math.floor (/ rc 32.0)))
      (define c (- rc (* r 32)))
      (define acc (get_item B1 k))
      (define wi (* k 9))
      (define bx (+ (* (* r 2) 66) (* c 2)))     ; 对应输入行 2r、列 2c（补边后 +0）
      (define ky 0)
      (define kx 0)
      (define jj 0)
    {
    (while (< ky 3) {
      (set! jj bx)
      (set! kx 0)
      (while (< kx 3) {
        (set! acc (+ acc (* (get_item W1 (+ wi kx)) (get_item xpad jj))))
        (set! jj (+ jj 1))
        (set! kx (+ kx 1))
      })
      (set! wi (+ wi 3))
      (set! bx (+ bx 66))
      (set! ky (+ ky 1))
    })
    (set_item! f1 i (if (> acc 0.0) acc 0.0))
  }))

;; 阶段 2：conv2 —— 16 通道 × 16×16，8 通道输入、3x3 步长 2 补边 1 + ReLU
(define u_conv2
  (lambda (i)
      (define k (Math.floor (/ i 256.0)))
      (define rc (- i (* k 256)))
      (define r (Math.floor (/ rc GRID)))
      (define c (- rc (* r GRID)))
      (define acc (get_item B2 k))
      (define wi (* k 72))
      (define ch 0)
      (define ky 0)
      (define kx 0)
      (define bb 0)
      (define jj 0)
    {
    (while (< ch NCH1) {
      (set! bb (+ (* ch 1156) (+ (* (* r 2) 34) (* c 2))))
      (set! ky 0)
      (while (< ky 3) {
        (set! jj bb)
        (set! kx 0)
        (while (< kx 3) {
          (set! acc (+ acc (* (get_item W2 (+ wi kx)) (get_item p1 jj))))
          (set! jj (+ jj 1))
          (set! kx (+ kx 1))
        })
        (set! wi (+ wi 3))
        (set! bb (+ bb 34))
        (set! ky (+ ky 1))
      })
      (set! ch (+ ch 1))
    })
    (set_item! f2 i (if (> acc 0.0) acc 0.0))
  }))

;; 阶段 3：conv3 —— 3x3 逐通道卷积（感受野 7→11），再 1x1 融合 + ReLU
(define u_dw
  (lambda (i) {
    (define ch (Math.floor (/ i 256.0)))
    (define rc (- i (* ch 256)))
    (define r (Math.floor (/ rc GRID)))
    (define c (- rc (* r GRID)))
    (define acc (get_item DW_B ch))
    (define wi (* ch 9))
    (define bb (+ (* ch 324) (+ (* r 18) c)))
    (define ky 0)
    (define kx 0)
    (define jj 0)
    {
    (while (< ky 3) {
      (set! jj bb)
      (set! kx 0)
      (while (< kx 3) {
        (set! acc (+ acc (* (get_item DW_W (+ wi kx)) (get_item p2 jj))))
        (set! jj (+ jj 1))
        (set! kx (+ kx 1))
      })
      (set! wi (+ wi 3))
      (set! bb (+ bb 18))
      (set! ky (+ ky 1))
    })
    (set_item! f3dw i acc)}
  }))

(define u_pw
  (lambda (i) {
    (define k (Math.floor (/ i 256.0)))
    (define rc (- i (* k 256)))
    (define acc (get_item PW_B k))
    (define ci 0)
    (define wi (* k NCH))
    {
    (while (< ci NCH) {
      (set! acc (+ acc (* (get_item PW_W (+ wi ci)) (get_item f3dw (+ (* ci 256) rc)))))
      (set! ci (+ ci 1))
    })
    (set_item! f3 i (if (> acc 0.0) acc 0.0))}
  }))

;; 阶段 4：RPN 中间层（1x1 卷积 24→24 + ReLU）
(define u_rpn_h
  (lambda (i)
      (define k (Math.floor (/ i 256.0)))
      (define rc (- i (* k 256)))
      (define acc (get_item BRH k))
      (define ci 0)
      (define wi (* k NCH))
    {
    (while (< ci NCH) {
      (set! acc (+ acc (* (get_item WRH (+ wi ci)) (get_item f3 (+ (* ci 256) rc)))))
      (set! ci (+ ci 1))
    })
    (set_item! rh i (if (> acc 0.0) acc 0.0))
  }))

;; 阶段 4：RPN 分类分支（每格 9 个锚框的前景 logit → sigmoid 概率）
(define u_rpn_cls
  (lambda (i)
      (define k (Math.floor (/ i 256.0)))
      (define rc (- i (* k 256)))
      (define acc (get_item BRC k))
      (define ci 0)
      (define wi (* k NCH))
    {
    (while (< ci NCH) {
      (set! acc (+ acc (* (get_item WRC (+ wi ci)) (get_item rh (+ (* ci 256) rc)))))
      (set! ci (+ ci 1))
    })
    (set_item! objl i acc)
    (set_item! objp i (/ 1.0 (+ 1.0 (Math.exp (- 0.0 acc)))))
  }))

;; 阶段 5：RPN 回归分支（每格 9 锚框 × 4 个偏移）
(define u_rpn_reg
  (lambda (i)
      (define k (Math.floor (/ i 256.0)))
      (define rc (- i (* k 256)))
      (define acc (get_item BRR k))
      (define ci 0)
      (define wi (* k NCH))
    {
    (while (< ci NCH) {
      (set! acc (+ acc (* (get_item WRR (+ wi ci)) (get_item rh (+ (* ci 256) rc)))))
      (set! ci (+ ci 1))
    })
    (set_item! rreg i acc)
  }))

;; ---- 提议：锚框 + 偏移 → 框，取 top-k --------------------------------------

(define tk_put
  (lambda (sc bx1 by1 bx2 by2)
      (define i 0)
      (define done 0)
    {
    (if (or (< tk_n PRE_TOPK) (> sc (get_item tk_s (- PRE_TOPK 1))))
        {
          (if (< tk_n PRE_TOPK)
              {
                (set! i tk_n)
                (set! tk_n (+ tk_n 1))
              }
              (set! i (- PRE_TOPK 1)))
          ;; 向下挪：把比 sc 小的往后挤（表按分数降序）。
          ;; 注意用 done 标志退出，不能把 i 归零来「跳出」——那样插入位置就丢了
          ;; （踩过：结果表根本没排序，RoI 数目也从 17 掉到 5）。
          (set! done 0)
          (while (and (== done 0) (> i 0))
            (if (> sc (get_item tk_s (- i 1)))
                {
                  (set_item! tk_s i (get_item tk_s (- i 1)))
                  (set_item! tk_b (+ (* i 4) 0) (get_item tk_b (+ (* (- i 1) 4) 0)))
                  (set_item! tk_b (+ (* i 4) 1) (get_item tk_b (+ (* (- i 1) 4) 1)))
                  (set_item! tk_b (+ (* i 4) 2) (get_item tk_b (+ (* (- i 1) 4) 2)))
                  (set_item! tk_b (+ (* i 4) 3) (get_item tk_b (+ (* (- i 1) 4) 3)))
                  (set! i (- i 1))
                }
                (set! done 1)))
          (set_item! tk_s i sc)
          (set_item! tk_b (+ (* i 4) 0) bx1)
          (set_item! tk_b (+ (* i 4) 1) by1)
          (set_item! tk_b (+ (* i 4) 2) bx2)
          (set_item! tk_b (+ (* i 4) 3) by2)
        }
        #f)
  }))

;; 阶段 6：逐个锚框解码（分数够高才进候选表）
(define u_prop
  (lambda (a)
      (define k (mod a NA))
      (define rc (Math.floor (/ a NA)))
      (define sc (get_item objp (+ (* k 256) rc)))
      (define bw 0.0)
      (define bh 0.0)
      (define bcx 0.0)
      (define bcy 0.0)
      (define dx 0.0)
      (define dy 0.0)
      (define dw 0.0)
      (define dh 0.0)
      (define cx 0.0)
      (define cy 0.0)
      (define x1 0.0)
      (define y1 0.0)
      (define x2 0.0)
      (define y2 0.0)
    {
    (if (> sc SCORE_TH)
        {
          (set! bw (+ 1.0 (- (get_item anch (+ (* a 4) 2)) (get_item anch (+ (* a 4) 0)))))
          (set! bh (+ 1.0 (- (get_item anch (+ (* a 4) 3)) (get_item anch (+ (* a 4) 1)))))
          (set! bcx (+ (get_item anch (+ (* a 4) 0)) (* 0.5 bw)))
          (set! bcy (+ (get_item anch (+ (* a 4) 1)) (* 0.5 bh)))
          (set! dx (get_item rreg (+ (* (+ (* k 4) 0) 256) rc)))
          (set! dy (get_item rreg (+ (* (+ (* k 4) 1) 256) rc)))
          (set! dw (clampf (get_item rreg (+ (* (+ (* k 4) 2) 256) rc)) -4.0 4.0))
          (set! dh (clampf (get_item rreg (+ (* (+ (* k 4) 3) 256) rc)) -4.0 4.0))
          (set! cx (+ bcx (* dx bw)))
          (set! cy (+ bcy (* dy bh)))
          (set! x1 (clampf (- cx (* 0.5 (* bw (Math.exp dw)))) 0.0 63.0))
          (set! y1 (clampf (- cy (* 0.5 (* bh (Math.exp dh)))) 0.0 63.0))
          (set! x2 (clampf (+ cx (* 0.5 (* bw (Math.exp dw)))) 0.0 63.0))
          (set! y2 (clampf (+ cy (* 0.5 (* bh (Math.exp dh)))) 0.0 63.0))
          (if (and (>= (- x2 x1) 3.0) (>= (- y2 y1) 3.0))
              (tk_put sc x1 y1 x2 y2)
              #f)
        }
        #f)
  }))

;; 交并比（+1 像素口径，与 Python 的 iou_matrix 一致）
(define iou
  (lambda (ax1 ay1 ax2 ay2 bx1 by1 bx2 by2)
      (define iw (+ 1.0 (- (minf ax2 bx2) (maxf ax1 bx1))))
      (define ih (+ 1.0 (- (minf ay2 by2) (maxf ay1 by1))))
      (define inter 0.0)
      (define aa 0.0)
      (define bb 0.0)
    {
    (if (< iw 0.0) (set! iw 0.0) #f)
    (if (< ih 0.0) (set! ih 0.0) #f)
    (set! inter (* iw ih))
    (set! aa (* (+ 1.0 (- ax2 ax1)) (+ 1.0 (- ay2 ay1))))
    (set! bb (* (+ 1.0 (- bx2 bx1)) (+ 1.0 (- by2 by1))))
    (/ inter (maxf 1e-6 (- (+ aa bb) inter)))
  }))

;; 阶段 6 收尾：对 top-k 候选做 NMS，取前 NROI 个作为 RoI
(define nms_props
  (lambda ()
      (define i 0)
      (define j 0)
      (define kept 0)
      (define drop 0)
      (define ax1 0.0)
      (define ay1 0.0)
      (define ax2 0.0)
      (define ay2 0.0)
    {
    (set! roi_n 0)
    (set! i 0)
    (while (and (< i tk_n) (< roi_n NROI)) {
      (set! drop 0)
      (set! j 0)
      (while (< j roi_n) {
        (set! ax1 (get_item tk_b (+ (* i 4) 0)))
        (set! ay1 (get_item tk_b (+ (* i 4) 1)))
        (set! ax2 (get_item tk_b (+ (* i 4) 2)))
        (set! ay2 (get_item tk_b (+ (* i 4) 3)))
        (if (> (iou ax1 ay1 ax2 ay2
                     (get_item roi_b (+ (* j 4) 0)) (get_item roi_b (+ (* j 4) 1))
                     (get_item roi_b (+ (* j 4) 2)) (get_item roi_b (+ (* j 4) 3)))
               RPN_IOU)
            (set! drop 1)
            #f)
        (set! j (+ j 1))
      })
      (if (== drop 0)
          {
            (set_item! roi_b (+ (* roi_n 4) 0) (get_item tk_b (+ (* i 4) 0)))
            (set_item! roi_b (+ (* roi_n 4) 1) (get_item tk_b (+ (* i 4) 1)))
            (set_item! roi_b (+ (* roi_n 4) 2) (get_item tk_b (+ (* i 4) 2)))
            (set_item! roi_b (+ (* roi_n 4) 3) (get_item tk_b (+ (* i 4) 3)))
            (set_item! roi_s roi_n (get_item tk_s i))
            (set! roi_n (+ roi_n 1))
          }
          #f)
      (set! i (+ i 1))
    })
    (set! kept roi_n)
  }))

;; ---- RoI 池化（Caffe 量化口径，最大池化） --------------------------------
;; 单元 = (RoI, 通道)：把一个 RoI 在特征图上的矩形切成 4x4 个 bin，每 bin 取最大值。

(define u_pool
  (lambda (i)
      (define roi (Math.floor (/ i NCH)))
      (define ch (- i (* roi NCH)))
      (define sx 0)
      (define sy 0)
      (define ex 0)
      (define ey 0)
      (define bw 0)
      (define bh 0)
      (define py 0)
      (define px 0)
      (define y0 0)
      (define y1 0)
      (define x0 0)
      (define x1 0)
      (define yy 0)
      (define xx 0)
      (define m 0.0)
      (define t 0.0)
    {
    (set! sx (Math.round (/ (get_item roi_b (+ (* roi 4) 0)) STRIDE)))
    (set! sy (Math.round (/ (get_item roi_b (+ (* roi 4) 1)) STRIDE)))
    (set! ex (Math.round (/ (get_item roi_b (+ (* roi 4) 2)) STRIDE)))
    (set! ey (Math.round (/ (get_item roi_b (+ (* roi 4) 3)) STRIDE)))
    (set! sx (clampf sx 0.0 (- GRID 1)))
    (set! sy (clampf sy 0.0 (- GRID 1)))
    (set! ex (clampf ex 0.0 (- GRID 1)))
    (set! ey (clampf ey 0.0 (- GRID 1)))
    (set! bw (maxf 1.0 (+ 1.0 (- ex sx))))
    (set! bh (maxf 1.0 (+ 1.0 (- ey sy))))
    (set! py 0)
    (while (< py POOL) {
      (set! y0 (Math.floor (+ sy (/ (* py bh) POOL))))
      (set! y1 (Math.ceil (+ sy (/ (* (+ py 1) bh) POOL))))
      (set! y0 (clampf y0 0.0 (- GRID 1)))
      (set! y1 (clampf y1 (+ y0 1.0) GRID))
      (set! px 0)
      (while (< px POOL) {
        (set! x0 (Math.floor (+ sx (/ (* px bw) POOL))))
        (set! x1 (Math.ceil (+ sx (/ (* (+ px 1) bw) POOL))))
        (set! x0 (clampf x0 0.0 (- GRID 1)))
        (set! x1 (clampf x1 (+ x0 1.0) GRID))
        (set! m (- 0.0 1e9))
        (set! yy y0)
        (while (< yy y1) {
          (set! xx x0)
          (while (< xx x1) {
            (set! t (get_item f3 (+ (* ch 256) (+ (* yy GRID) xx))))
            (if (> t m) (set! m t) #f)
            (set! xx (+ xx 1))
          })
          (set! yy (+ yy 1))
        })
        (set_item! pooled (+ (* roi FDIM) (+ (* ch (* POOL POOL)) (+ (* py POOL) px))) m)
        (set! px (+ px 1))
      })
      (set! py (+ py 1))
    })
  }))

;; ---- 分类头 + 回归头 -----------------------------------------------------
;; 单元前段 = fc1 的一个输出（RoI × 48 个隐单元）；后段 = 分类 11 个 logit 与
;; 回归 40 个偏移量。

(define u_head
  (lambda (i)
      (define nh (* roi_n HID))
      (define roi 0)
      (define u 0)
      (define acc 0.0)
      (define d 0)
      (define wi 0)
      (define base 0)
      (define o 0)
    {
    (if (< i nh)
        {
          (set! roi (Math.floor (/ i HID)))
          (set! u (- i (* roi HID)))
          (set! acc (get_item BF1 u))
          (set! wi (* u FDIM))
          (set! base (* roi FDIM))
          (set! d 0)
          (while (< d FDIM) {
            (set! acc (+ acc (* (get_item WF1 (+ wi d)) (get_item pooled (+ base d)))))
            (set! d (+ d 1))
          })
          (set_item! fch i (if (> acc 0.0) acc 0.0))
        }
        {
          (set! o (- i nh))
          (set! roi (Math.floor (/ o (+ NCLS (* NCLS 4)))))
          (set! u (- o (* roi (+ NCLS (* NCLS 4)))))
          (if (< u NCLS)
              {
                ;; 分类 logit（u = 类别）
                (set! acc (get_item BCL u))
                (set! wi (* u HID))
                (set! base (* roi HID))
                (set! d 0)
                (while (< d HID) {
                  (set! acc (+ acc (* (get_item WCL (+ wi d)) (get_item fch (+ base d)))))
                  (set! d (+ d 1))
                })
                (set_item! clsl (+ (* roi NCLS) u) acc)
              }
              {
                ;; 回归偏移（u - NCLS = 类×4 + 坐标）
                (set! acc (get_item BRG (- u NCLS)))
                (set! wi (* (- u NCLS) HID))
                (set! base (* roi HID))
                (set! d 0)
                (while (< d HID) {
                  (set! acc (+ acc (* (get_item WRG (+ wi d)) (get_item fch (+ base d)))))
                  (set! d (+ d 1))
                })
                (set_item! hreg (+ (* roi (* NCLS 4)) (- u NCLS)) acc)
              })
        })
  }))

;; 归一化分类 logit → softmax 概率（每 RoI 11 类）
(define head_softmax
  (lambda ()
      (define r 0)
      (define c 0)
      (define m 0.0)
      (define s 0.0)
      (define v 0.0)
    {
    (while (< r roi_n) {
      (set! m (get_item clsl (* r NCLS)))
      (set! c 1)
      (while (< c NCLS) {
        (if (> (get_item clsl (+ (* r NCLS) c)) m)
            (set! m (get_item clsl (+ (* r NCLS) c)))
            #f)
        (set! c (+ c 1))
      })
      (set! s 0.0)
      (set! c 0)
      (while (< c NCLS) {
        (set! v (Math.exp (- (get_item clsl (+ (* r NCLS) c)) m)))
        (set_item! clsp (+ (* r NCLS) c) v)
        (set! s (+ s v))
        (set! c (+ c 1))
      })
      (set! c 0)
      (while (< c NCLS) {
        (set_item! clsp (+ (* r NCLS) c) (/ (get_item clsp (+ (* r NCLS) c)) s))
        (set! c (+ c 1))
      })
      (set! r (+ r 1))
    })
  }))

;; ---- 最终检测：逐 RoI 取 argmax 类别 + 细化框 -----------------------------

(define u_det
  (lambda (r) {
    (define bw 0.0)
    (define bh 0.0)
    (define bcx 0.0)
    (define bcy 0.0)
    (define dw 0.0)
    (define dh 0.0)
    (define cx 0.0)
    (define cy 0.0)
    (define x1 0.0)
    (define y1 0.0)
    (define x2 0.0)
    (define y2 0.0)
    (define c 0)
    (define s 0.0)
    {
    (set! bw (+ 1.0 (- (get_item roi_b (+ (* r 4) 2)) (get_item roi_b (+ (* r 4) 0)))))
    (set! bh (+ 1.0 (- (get_item roi_b (+ (* r 4) 3)) (get_item roi_b (+ (* r 4) 1)))))
    (set! bcx (+ (get_item roi_b (+ (* r 4) 0)) (* 0.5 bw)))
    (set! bcy (+ (get_item roi_b (+ (* r 4) 1)) (* 0.5 bh)))
    (set! c 0)
    (while (< c 10) {
      (set! s (get_item clsp (+ (* r NCLS) c)))
      (if (and (>= s SCORE_TH) (< det_n DET_MAX))
          {
            (set! dw (clampf (get_item hreg (+ (* r (* NCLS 4)) (+ (* c 4) 2))) -4.0 4.0))
            (set! dh (clampf (get_item hreg (+ (* r (* NCLS 4)) (+ (* c 4) 3))) -4.0 4.0))
            (set! cx (+ bcx (* (get_item hreg (+ (* r (* NCLS 4)) (+ (* c 4) 0))) bw)))
            (set! cy (+ bcy (* (get_item hreg (+ (* r (* NCLS 4)) (+ (* c 4) 1))) bh)))
            (set! x1 (clampf (- cx (* 0.5 (* bw (Math.exp dw)))) 0.0 63.0))
            (set! y1 (clampf (- cy (* 0.5 (* bh (Math.exp dh)))) 0.0 63.0))
            (set! x2 (clampf (+ cx (* 0.5 (* bw (Math.exp dw)))) 0.0 63.0))
            (set! y2 (clampf (+ cy (* 0.5 (* bh (Math.exp dh)))) 0.0 63.0))
            (set_item! det_cls det_n c)
            (set_item! det_sc det_n s)
            (set_item! det_b (+ (* det_n 4) 0) x1)
            (set_item! det_b (+ (* det_n 4) 1) y1)
            (set_item! det_b (+ (* det_n 4) 2) x2)
            (set_item! det_b (+ (* det_n 4) 3) y2)
            (set! det_n (+ det_n 1))
          }
          #f)
      (set! c (+ c 1))
    })}
  }))

;; 逐类 NMS：把 det_* 里同类的高分框互相抑制，输出 fin_*
;; 把 det_* 按分数降序排一遍（选择排序，n ≤ 64：~2000 次比较，只跑一次）
(define det_sort
  (lambda () {
    (define i 0)
    (define j 0)
    (define b 0)
    (define tv 0.0)
    (define tc 0)
    (define t0 0.0)
    (define t1 0.0)
    (define t2 0.0)
    (define t3 0.0)
    {
    (set! i 0)
    (while (< i det_n) {
      (set! b i)
      (set! j i)
      (while (< j det_n) {
        (if (> (get_item det_sc j) (get_item det_sc b)) (set! b j) #f)
        (set! j (+ j 1))
      })
      (if (> b i)
          {
            (set! tv (get_item det_sc i))
            (set! tc (get_item det_cls i))
            (set! t0 (get_item det_b (* i 4)))
            (set! t1 (get_item det_b (+ (* i 4) 1)))
            (set! t2 (get_item det_b (+ (* i 4) 2)))
            (set! t3 (get_item det_b (+ (* i 4) 3)))
            (set_item! det_sc i (get_item det_sc b))
            (set_item! det_cls i (get_item det_cls b))
            (set_item! det_b (* i 4) (get_item det_b (* b 4)))
            (set_item! det_b (+ (* i 4) 1) (get_item det_b (+ (* b 4) 1)))
            (set_item! det_b (+ (* i 4) 2) (get_item det_b (+ (* b 4) 2)))
            (set_item! det_b (+ (* i 4) 3) (get_item det_b (+ (* b 4) 3)))
            (set_item! det_sc b tv)
            (set_item! det_cls b tc)
            (set_item! det_b (* b 4) t0)
            (set_item! det_b (+ (* b 4) 1) t1)
            (set_item! det_b (+ (* b 4) 2) t2)
            (set_item! det_b (+ (* b 4) 3) t3)
          }
          #f)
      (set! i (+ i 1))
    })}
  }))

(define final_nms
  (lambda ()
      (define c 0)
      (define i 0)
      (define j 0)
      (define drop 0)
      (define k 0)
      (define n 0)
      (define sc 0.0)
      (define bi 0)
    {
    (det_sort)
    (set! fin_n 0)
    (set! c 0)
    (while (< c 10) {
      ;; det_* 已按分数降序：逐类贪心 NMS（候选 ≤ 64 个）
      (set! n 0)
      (set! i 0)
      (while (< i det_n) {
        (if (== (get_item det_cls i) c)
            {
              ;; 与已保留的同类框比 IoU
              (set! drop 0)
              (set! j 0)
              (while (< j fin_n) {
                (if (== (get_item fin_cls j) c)
                    (if (> (iou (get_item det_b (+ (* i 4) 0)) (get_item det_b (+ (* i 4) 1))
                               (get_item det_b (+ (* i 4) 2)) (get_item det_b (+ (* i 4) 3))
                               (get_item fin_b (+ (* j 4) 0)) (get_item fin_b (+ (* j 4) 1))
                               (get_item fin_b (+ (* j 4) 2)) (get_item fin_b (+ (* j 4) 3)))
                           FIN_IOU)
                        (set! drop 1)
                        #f)
                    #f)
                (set! j (+ j 1))
              })
              (if (== drop 0)
                  {
                    (set_item! fin_cls fin_n c)
                    (set_item! fin_sc fin_n (get_item det_sc i))
                    (set_item! fin_b (+ (* fin_n 4) 0) (get_item det_b (+ (* i 4) 0)))
                    (set_item! fin_b (+ (* fin_n 4) 1) (get_item det_b (+ (* i 4) 1)))
                    (set_item! fin_b (+ (* fin_n 4) 2) (get_item det_b (+ (* i 4) 2)))
                    (set_item! fin_b (+ (* fin_n 4) 3) (get_item det_b (+ (* i 4) 3)))
                    (set! fin_n (+ fin_n 1))
                  }
                  #f)
            }
            #f)
        (set! i (+ i 1))
      })
      (set! c (+ c 1))
    })
  }))

;; 求和（离线校验用）与某 RoI 的 argmax 类别
(define sum_range
  (lambda (lst off n)
      (define i 0)
      (define s 0.0)
    {
    (while (< i n) {
      (set! s (+ s (get_item lst (+ off i))))
      (set! i (+ i 1))
    })
    s
  }))

(define ra_i 0)
(define ra_b 0)
(define ra_v 0.0)
(define roi_argmax
  (lambda (r) {
    (set! ra_b 0)
    (set! ra_v (get_item clsp (* r NCLS)))
    (set! ra_i 1)
    (while (< ra_i NCLS) {
      (if (> (get_item clsp (+ (* r NCLS) ra_i)) ra_v)
          {
            (set! ra_v (get_item clsp (+ (* r NCLS) ra_i)))
            (set! ra_b ra_i)
          }
          #f)
      (set! ra_i (+ ra_i 1))
    })
    ra_b
  }))

;; ---- 分块推进驱动 --------------------------------------------------------
;;
;; 阶段表（UI 的 10 步）：
;;   0 输入归一化      1 conv1      2 conv2        3 RPN 中间层
;;   4 RPN 分类分支    5 RPN 回归   6 提议解码+NMS  7 RoI 池化
;;   8 分类/回归头     9 最终检测
;;
;; 每拍调 step_stage(阶段, 预算单元数)，返回值 = 进度 0..1（1 = 该阶段完成）。

(define jp 0)          ; 当前阶段已完成的单元数
(define jp_n 0)        ; 当前阶段单元总数
(define jp_stage -1)

(define stage_units
  (lambda (s)
    (cond ((== s 0) NIMG)
          ((== s 1) NF1)
          ((== s 2) NF2)
          ((== s 3) (* NF2 2))
          ((== s 4) NF2)
          ((== s 5) (* NA 256))
          ((== s 6) (* NA (* 4 256)))
          ((== s 7) NANCH)
          ((== s 8) (* roi_n NCH))
          ((== s 9) (+ (* roi_n HID) (* roi_n (+ NCLS (* NCLS 4)))))
          (else roi_n))))

(define begin_stage
  (lambda (s) {
    (set! jp 0)
    (set! jp_stage s)
    (set! jp_n (stage_units s))
    (set! last_ms (System.timestamp))
  }))

(define run_unit
  (lambda (s i)
    (cond ((== s 0) (u_norm i))
          ((== s 1) (u_conv1 i))
          ((== s 2) (u_conv2 i))
          ((== s 3) (if (< i NF2) (u_dw i) (u_pw (- i NF2))))
          ((== s 4) (u_rpn_h i))
          ((== s 5) (u_rpn_cls i))
          ((== s 6) (u_rpn_reg i))
          ((== s 7) (u_prop i))
          ((== s 8) (u_pool i))
          ((== s 9) (u_head i))
          (else (u_det i)))))

(define step_stage
  (lambda (s budget)
      (define n 0)
    {
    (while (and (< n budget) (< jp jp_n)) {
      (run_unit s jp)
      (set! jp (+ jp 1))
      (set! n (+ n 1))
    })
    (if (< jp jp_n)
        (/ jp jp_n)
        {
          ;; 阶段收尾（把中间结果整理成下一阶段的输入）
          (cond ((== s 1) (pad_f1))
                ((== s 2) (pad_f2))
                ((== s 7) (nms_props))
                ((== s 9) (head_softmax))
                ((== s 10) (final_nms))
                (else #f))
          1.0
        })
  }))

;; conv3 之前把 f2 补 1px 边（0）——同样是为了让热循环不判边界
(define pad_f2
  (lambda () {
    (define ch 0)
    (define r 0)
    (define c 0)
    {
    (while (< ch NCH) {
      (set! r 0)
      (while (< r 18) {
        (set! c 0)
        (while (< c 18) {
          (set_item! p2 (+ (* ch 324) (+ (* r 18) c))
                     (if (and (> r 0) (< r 17) (> c 0) (< c 17))
                         (get_item f2 (+ (* ch 256) (+ (* (- r 1) 16) (- c 1))))
                         0.0))
          (set! c (+ c 1))
        })
        (set! r (+ r 1))
      })
      (set! ch (+ ch 1))
    })}
  }))

;; conv2 之前把 f1 补 1px 边（0）——避免卷积热循环里判边界
(define pad_f1
  (lambda ()
      (define ch 0)
      (define r 0)
      (define c 0)
    {
    (while (< ch NCH1) {
      (set! r 0)
      (while (< r 34) {
        (set! c 0)
        (while (< c 34) {
          (set_item! p1 (+ (* ch 1156) (+ (* r 34) c))
                     (if (and (> r 0) (< r 33) (> c 0) (< c 33))
                         (get_item f1 (+ (* ch 1024) (+ (* (- r 1) 32) (- c 1))))
                         0.0))
          (set! c (+ c 1))
        })
        (set! r (+ r 1))
      })
      (set! ch (+ ch 1))
    })
  }))

;; 清掉一整套推理结果（换图时调；不动 img 本身）
(define reset_results
  (lambda ()
      (define i 0)
    {
    (set! tk_n 0)
    (set! roi_n 0)
    (set! det_n 0)
    (set! fin_n 0)
    (set! jp 0)
    (set! jp_n 0)
    (set! jp_stage -1)
    (set! i 0)
    (while (< i NF2) {
      (set_item! f2 i 0.0)
      (set_item! f3 i 0.0)
      (set_item! f3dw i 0.0)
      (set_item! rh i 0.0)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i NF1) {
      (set_item! f1 i 0.0)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i (* NA 256)) {
      (set_item! objp i 0.0)
      (set_item! objl i 0.0)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i (* NROI FDIM)) {
      (set_item! pooled i 0.0)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i (* NROI NCLS)) {
      (set_item! clsp i 0.0)
      (set_item! clsl i 0.0)
      (set! i (+ i 1))
    })
  }))

;; ---- 初始化 --------------------------------------------------------------

(define ready 0)

(define init
  (lambda () {
    (if (== ready 0)
        {
          (anch_init)
          (clear_image)
          (set! ready 1)
        }
        #f)
    ready
  }))
