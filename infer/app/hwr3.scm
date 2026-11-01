;; ============================================================================
;; hwr3.scm —— 手写汉字识别 v3：小 CNN（卷积 + 池化 → 128 维嵌入）+ 最近原型（NCM）
;;
;; 与 v1（hwr.scm，1000 类、字体合成数据、手工特征）、v2（hwr2.scm，3755 类、真迹、
;; 手工 512 维特征 + PCA）并存，二者均未改动。本版用**端到端学出来的特征**：
;; 手工特征的 top-1 卡在 58.5%，换成小 CNN 后（训练见 tools/train_hwr_cnn.py）实测
;; 见 doc/AGENTS.md。之所以换：CASIA 官方 512 维特征 + 同一个最近类均值能到 81.5%，
;; 而我的手工特征与它近乎线性无关（R²≈0.13），说明差距在"方向算子/聚合方式"。
;;
;; 前处理与 v2 完全相同（同一套墨迹渲染 + 伪二维归一化），网络：
;;   32×32 输入（= 64×64 归一化墨迹 2×2 均值池化）
;;   conv1 3×3 p1 s2  1→8   → 16×16   +BN(已折进权重)+ReLU
;;   maxpool 2×2 s2         → 8×8×8
;;   conv2 3×3 p1 s1  8→16  → 8×8     +BN+ReLU
;;   avgpool 2×2 s2         → 4×4×16 = 256
;;   fc 256→128 + L2 归一化
;;   余弦分类器 = 最近原型（3755 个 128 维单位原型，int8 逐类量化）
;; 推理规模 ≈ 58 万乘加（WASM 实测见 doc/AGENTS.md）。
;;
;; 解释器约束（Animac manual.md，逐条遵守）：算术/比较固定二参数；define 只在 lambda
;; 体顶层；没有 list（用 cons）；热循环一律用 lambda 形参/局部量（比模块级变量快 3 倍）；
;; 自己写的定参函数别多传参数（会静默错）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Diag)
(native Fs)

(import EX "app/lib/exit.scm")
(import M "app/lib/hwr3/meta.scm")

;; 模型数据（数据文件而非源码：灵机按源码字符数预分配 token 数组，见 hwr.scm 注释）
(define c1parts (cdr '(0)))            ; conv1 核：C1 行 × 9
(define c2parts (cdr '(0)))            ; conv2 核：C2 行 × (C1*9)
(define fcparts (cdr '(0)))            ; fc：nred 行 × 256
;; 卷积/fc 的权重在启动时解码成浮点表（int8 串 → list）：这三处权重总量只有 3.4 万，
;; 启动多花 ~0.1s，换来推理内循环里 get_item（比 String.charCodeAt 快且少一次 384 偏移）
(define w1v (cdr '(0)))
(define w2v (cdr '(0)))
(define wfcv (cdr '(0)))
(define pnparts (cdr '(0)))            ; 原型：K 行 × nred
(define params "")
(define b1s "")                        ; conv1 偏置（C1 个数）
(define b2s "")                        ; conv2 偏置
(define bfs "")                        ; fc 偏置（nred）
(define sfcs "")                       ; fc 每行尺度（nred 个 16 位）
(define spns "")                       ; 原型每类尺度（K 个 16 位）
(define cstr "")                       ; 汉字表
(define load_err 0)

;; ---- 布局/常量 ----------------------------------------------------------
(define SW (Gfx.width))
(define TICK_MS 20)
(define PAD_X 4)
(define PAD_Y 36)
(define INK 64)
(define CELLPX 2)
(define PAD_PX (* INK CELLPX))
(define NG 64)
(define ALPHA 0.7)
(define SMOOTH 2)
(define INS 32)                        ; CNN 输入边长
(define O1 16)                         ; conv1 输出边长（stride 2, pad 1）
(define O2 8)                          ; conv2 输出边长（stride 1, pad 1）
(define C1 M.nC1)
(define C2 M.nC2)
(define NRED M.nred)
(define NCLS M.ncls)
(define NFLAT (* (* 4 4) C2))          ; 4×4×C2 = 256
(define NTOP 16)
(define PAGE_N 8)
(define BRUSH 2.0)
(define PEN_STEP 0.5)
(define PEND_MS 420)
(define CX 138)
(define CY 56)
(define CCELL 22)
(define CBOX 20)
(define VIZ_LY 167)
(define VIZ_Y 183)
(define HDR_BTN_X 62)
(define HDR_BTN_W 40)
(define BS_X 252)
(define BS_W 32)
(define CL_X 286)
(define CL_W 30)

;; ---- 小工具 -------------------------------------------------------------
(define i2s (lambda (x) (String.atom_to_string x)))
(define maxf (lambda (a b) (if (> a b) a b)))
(define minf (lambda (a b) (if (< a b) a b)))
(define mkn
  (lambda (n v) {
    (define acc (cdr '(0)))
    (define i 0)
    (while (< i n) {
      (push acc v)
      (set! i (+ i 1))
    })
    acc
  }))
(define tw
  (lambda (s) {
    (define w 0)
    (define i 0)
    (define n (String.length s))
    (while (< i n) {
      (set! w (+ w (if (< (String.charCodeAt i s) 11904) 7 12)))
      (set! i (+ i 1))
    })
    w
  }))
(define fmt_pct
  (lambda (p) {
    (define v (Math.round (* p 1000.0)))
    (String.concat (i2s (Math.floor (/ v 10.0)))
      (String.concat "." (String.concat (i2s (mod v 10.0)) "%")))
  }))
(define cls_at (lambda (k) (String.charAt cstr k)))
(define hit_rect
  (lambda (px py x y w h)
    (if (and (>= px x) (< px (+ x w)))
        (if (and (>= py y) (< py (+ y h))) 1 0)
        0)))

;; ---- 缓冲区 -------------------------------------------------------------
(define ink (mkn (* INK INK) 0.0))          ; 显示用墨迹
(define dens (mkn (* NG NG) 0.0))           ; 特征用细线墨迹
(define nrm (mkn (* NG NG) 0.0))            ; 伪二维归一化后
(define inp (mkn (* INS INS) 0.0))          ; 32×32 网络输入
(define pad1 (mkn (* (+ INS 2) (+ INS 2)) 0.0))   ; conv1 用的 34×34 零填充图
(define f1 (mkn (* C1 (* O1 O1)) 0.0))          ; conv1 输出 8×16×16
(define r1 (mkn (* C1 (* O1 O1)) 0.0))          ; ReLU 后
(define q1 (mkn (* C1 (* O2 O2)) 0.0))      ; maxpool 后 8×8×8
(define pad2 (mkn (* C1 (* (+ O2 2) (+ O2 2))) 0.0))  ; conv2 输入（补零后的 8×8×C1）
(define f2 (mkn (* C2 (* O2 O2)) 0.0))      ; conv2 输出 16×8×8
(define r2 (mkn (* C2 (* O2 O2)) 0.0))      ; ReLU 后
(define flat (mkn NFLAT 0.0))               ; 4×4×C2
(define emb (mkn NRED 0.0))                 ; 128 维嵌入（L2 归一化）
(define b1l (mkn C1 0.0))
(define b2l (mkn C2 0.0))
(define bfl (mkn NRED 0.0))
(define sfcl (mkn NRED 0.0))
(define spnl (mkn NCLS 0.0))
(define prow (mkn (* 4 NG) 0.0))
(define prowb (mkn (* 4 NG) 0.0))
(define ftab (mkn (* 2 NG) 0.0))
(define topk (mkn NTOP 0))
(define topv (mkn NTOP 0.0))

(define strokes (cdr '(0)))
(define cpts (cdr '(0)))
(define has_ink 0)
(define pen_down 0)
(define armed 0)
(define text "")
(define cand_n 0)
(define page 0)
(define infer_ms 0)
(define feat_ms 0)
(define timer_id 0)
(define dirty_x0 0)
(define dirty_y0 0)
(define dirty_x1 -1)
(define dirty_y1 -1)
(define pen_x 0.0)
(define pen_y 0.0)
(define up_ms 0)

;; ---- 绘图板（与 v1/v2 同款） --------------------------------------------
(define stamp
  (lambda (inkv gx gy) {
    (define i0 (Math.floor (- gy (+ BRUSH 0.5))))
    (define i1 (Math.ceil (+ gy (+ BRUSH 0.5))))
    (define j0 (Math.floor (- gx (+ BRUSH 0.5))))
    (define j1 (Math.ceil (+ gx (+ BRUSH 0.5))))
    (define i 0)
    (define j 0)
    (define dx 0.0)
    (define dy 0.0)
    (define d 0.0)
    (define idx 0)
    (if (< i0 0) (set! i0 0) #f)
    (if (< j0 0) (set! j0 0) #f)
    (if (> i1 (- INK 1)) (set! i1 (- INK 1)) #f)
    (if (> j1 (- INK 1)) (set! j1 (- INK 1)) #f)
    (if (< i0 dirty_y0) (set! dirty_y0 i0) #f)
    (if (< j0 dirty_x0) (set! dirty_x0 j0) #f)
    (if (> i1 dirty_y1) (set! dirty_y1 i1) #f)
    (if (> j1 dirty_x1) (set! dirty_x1 j1) #f)
    (set! i i0)
    (while (<= i i1) {
      (set! dy (- (+ i 0.5) gy))
      (set! j j0)
      (while (<= j j1) {
        (set! dx (- (+ j 0.5) gx))
        (set! d (Math.sqrt (+ (* dx dx) (* dy dy))))
        (if (< d BRUSH)
            {
              (set! idx (+ (* i INK) j))
              (set! d (- 1.0 (/ d BRUSH)))
              (if (> d (get_item inkv idx)) (set_item! inkv idx d) #f)
              (set! has_ink 1)
            }
            #f)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
  }))

(define tr_dx 0.0)
(define tr_dy 0.0)
(define tr_n 0)
(define tr_i 0)
(define trace_to
  (lambda (gx gy) {
    (set! tr_dx (- gx pen_x))
    (set! tr_dy (- gy pen_y))
    (set! tr_n (Math.ceil (/ (Math.sqrt (+ (* tr_dx tr_dx) (* tr_dy tr_dy))) PEN_STEP)))
    (if (< tr_n 1) (set! tr_n 1) #f)
    (set! tr_i 1)
    (while (<= tr_i tr_n) {
      (stamp ink (+ pen_x (/ (* tr_dx tr_i) tr_n)) (+ pen_y (/ (* tr_dy tr_i) tr_n)))
      (set! tr_i (+ tr_i 1))
    })
    (push cpts (cons gx (cons gy (cdr '(0)))))
    (set! pen_x gx)
    (set! pen_y gy)
  }))

(define clear_all_ink
  (lambda () {
    (define i 0)
    (while (< i (* INK INK)) {
      (set_item! ink i 0.0)
      (set! i (+ i 1))
    })
    (set! has_ink 0)
    (set! dirty_x0 0)
    (set! dirty_y0 0)
    (set! dirty_x1 -1)
    (set! dirty_y1 -1)
    (while (> (length strokes) 0) (pop strokes))
    (while (> (length cpts) 0) (pop cpts))
  }))

;; ---- 细线墨迹 64×64（与 v2 逐位一致） -----------------------------------
(define rt_minx 0.0)
(define rt_miny 0.0)
(define rt_w 1.0)
(define rt_h 1.0)
(define rt_x 0.0)
(define rt_y 0.0)
(define rt_px 0.0)
(define rt_py 0.0)
(define rt_dx 0.0)
(define rt_dy 0.0)
(define rt_n 0)
(define rt_k 0)
(define rt_t 0)
(define sp_x 0.0)
(define sp_y 0.0)
(define sp_w 0.0)
(define sp_j0 0)
(define sp_i0 0)
(define sp_fx 0.0)
(define sp_fy 0.0)
(define sp_idx 0)
(define splat_cell
  (lambda (dv ii jj w) {
    (if (and (>= ii 0) (< ii NG) (>= jj 0) (< jj NG))
        {
          (set! sp_idx (+ (* ii NG) jj))
          (set_item! dv sp_idx (+ (get_item dv sp_idx) w))
        }
        #f)
  }))
(define splat
  (lambda (dv x y w) {
    (set! sp_j0 (Math.floor x))
    (set! sp_i0 (Math.floor y))
    (set! sp_fx (- x sp_j0))
    (set! sp_fy (- y sp_i0))
    (splat_cell dv sp_i0 sp_j0 (* w (* (- 1.0 sp_fx) (- 1.0 sp_fy))))
    (splat_cell dv sp_i0 (+ sp_j0 1) (* w (* sp_fx (- 1.0 sp_fy))))
    (splat_cell dv (+ sp_i0 1) sp_j0 (* w (* (- 1.0 sp_fx) sp_fy)))
    (splat_cell dv (+ sp_i0 1) (+ sp_j0 1) (* w (* sp_fx sp_fy)))
  }))

(define bb_minx 0.0)
(define bb_miny 0.0)
(define bb_maxx 0.0)
(define bb_maxy 0.0)
(define bb_first 1)
(define bb_add
  (lambda (x y) {
    (if (== bb_first 1)
        {
          (set! bb_minx x) (set! bb_maxx x)
          (set! bb_miny y) (set! bb_maxy y)
          (set! bb_first 0)
        }
        {
          (if (< x bb_minx) (set! bb_minx x) #f)
          (if (> x bb_maxx) (set! bb_maxx x) #f)
          (if (< y bb_miny) (set! bb_miny y) #f)
          (if (> y bb_maxy) (set! bb_maxy y) #f)
        })
  }))

(define bd_si 0)
(define bd_pi 0)
(define bd_ns 0)
(define bd_np 0)
(define bd_x 0.0)
(define bd_y 0.0)
(define bd_nx 0.0)
(define bd_ny 0.0)
(define bd_dx 0.0)
(define bd_dy 0.0)
(define bd_n 0)
(define bd_k 0)
(define bd_t 0)
(define bd_st #f)
(define bd_pt #f)
(define build_dens
  (lambda (stv dv) {
    (set! bd_t 0)
    (while (< bd_t (* NG NG)) {
      (set_item! dv bd_t 0.0)
      (set! bd_t (+ bd_t 1))
    })
    (set! bb_first 1)
    (set! bd_si 0)
    (set! bd_ns (length stv))
    (while (< bd_si bd_ns) {
      (set! bd_st (get_item stv bd_si))
      (set! bd_np (length bd_st))
      (set! bd_pi 0)
      (while (< bd_pi bd_np) {
        (set! bd_pt (get_item bd_st bd_pi))
        (bb_add (get_item bd_pt 0) (get_item bd_pt 1))
        (set! bd_pi (+ bd_pi 1))
      })
      (set! bd_si (+ bd_si 1))
    })
    (if (== bb_first 1) #f
        {
          (set! rt_minx bb_minx)
          (set! rt_miny bb_miny)
          (set! rt_w (- bb_maxx bb_minx))
          (set! rt_h (- bb_maxy bb_miny))
          (if (< rt_w 0.000001) (set! rt_w 0.000001) #f)
          (if (< rt_h 0.000001) (set! rt_h 0.000001) #f)
          (set! bd_si 0)
          (while (< bd_si bd_ns) {
            (set! bd_st (get_item stv bd_si))
            (set! bd_np (length bd_st))
            (set! bd_pi 0)
            (while (< bd_pi bd_np) {
              (set! bd_pt (get_item bd_st bd_pi))
              (set! bd_x (/ (* (- (get_item bd_pt 0) rt_minx) 63.0) rt_w))
              (set! bd_y (/ (* (- (get_item bd_pt 1) rt_miny) 63.0) rt_h))
              (if (== bd_np 1)
                  (splat dv bd_x bd_y 1.0)
                  {
                    (if (< bd_pi (- bd_np 1))
                        {
                          (set! bd_pt (get_item bd_st (+ bd_pi 1)))
                          (set! bd_nx (/ (* (- (get_item bd_pt 0) rt_minx) 63.0) rt_w))
                          (set! bd_ny (/ (* (- (get_item bd_pt 1) rt_miny) 63.0) rt_h))
                          (set! bd_dx (- bd_nx bd_x))
                          (set! bd_dy (- bd_ny bd_y))
                          (set! bd_n (Math.floor (* (Math.sqrt (+ (* bd_dx bd_dx) (* bd_dy bd_dy))) 2.0)))
                          (if (< bd_n 1) (set! bd_n 1) #f)
                          (set! bd_k 0)
                          (while (<= bd_k bd_n) {
                            (splat dv
                                   (+ bd_x (/ (* bd_dx bd_k) bd_n))
                                   (+ bd_y (/ (* bd_dy bd_k) bd_n))
                                   (/ 1.0 (+ bd_n 1.0)))
                            (set! bd_k (+ bd_k 1))
                          })
                        }
                        #f)
                  })
              (set! bd_pi (+ bd_pi 1))
            })
            (set! bd_si (+ bd_si 1))
          })
        })
  }))

;; ---- 伪二维归一化（与 v2 逐位一致） ------------------------------------
(define wm_i 0)
(define wm_k 0)
(define wm_p 0.0)
(define wm_tot 0.0)
(define wm_t 0.0)
(define wm_eq 0.0)
(define wm_mx 0.0)
(define wm_c0 0.0)
(define wm_c1 0.0)
(define wm_ceil 0.0)
(define wm_s 0)
(define build_warp
  (lambda (proj off) {
    (set! wm_s 0)
    (while (< wm_s SMOOTH) {
      (set! wm_i 0)
      (while (< wm_i NG) {
        (set! wm_p (get_item proj wm_i))
        (set! wm_t (* 0.5 wm_p))
        (if (> wm_i 0)
            (set! wm_t (+ wm_t (* 0.25 (get_item proj (- wm_i 1)))))
            (set! wm_t (+ wm_t (* 0.25 wm_p))))
        (if (< wm_i (- NG 1))
            (set! wm_t (+ wm_t (* 0.25 (get_item proj (+ wm_i 1)))))
            (set! wm_t (+ wm_t (* 0.25 wm_p))))
        (set_item! proj (+ NG wm_i) wm_t)
        (set! wm_i (+ wm_i 1))
      })
      (set! wm_i 0)
      (while (< wm_i NG) {
        (set_item! proj wm_i (get_item proj (+ NG wm_i)))
        (set! wm_i (+ wm_i 1))
      })
      (set! wm_s (+ wm_s 1))
    })
    (set! wm_tot 0.0)
    (set! wm_i 0)
    (while (< wm_i NG) {
      (set! wm_tot (+ wm_tot (get_item proj wm_i)))
      (set! wm_i (+ wm_i 1))
    })
    (if (< wm_tot 0.0000001)
        {
          (set! wm_i 0)
          (while (< wm_i NG) {
            (set_item! ftab (+ off wm_i) (/ (+ 0.0 wm_i) 63.0))
            (set! wm_i (+ wm_i 1))
          })
        }
        {
          (set_item! proj (* 2 NG) 0.0)
          (set! wm_i 0)
          (while (< wm_i NG) {
            (set_item! proj (+ (* 2 NG) (+ wm_i 1))
                       (+ (get_item proj (+ (* 2 NG) wm_i)) (/ (get_item proj wm_i) wm_tot)))
            (set! wm_i (+ wm_i 1))
          })
          (set! wm_mx 0.0)
          (set! wm_k 1)
          (set! wm_i 0)
          (while (< wm_i NG) {
            (set! wm_t (/ (+ 0.0 wm_i) 63.0))
            (while (< wm_k NG) {
              (set! wm_c0 (get_item proj (+ (* 2 NG) wm_k)))
              (if (> wm_c0 wm_t) break #f)
              (set! wm_k (+ wm_k 1))
            })
            (set! wm_c0 (get_item proj (+ (* 2 NG) (- wm_k 1))))
            (set! wm_c1 (get_item proj (+ (* 2 NG) wm_k)))
            (set! wm_ceil (- wm_c1 wm_c0))
            (if (< wm_ceil 0.000001)
                (set! wm_eq (/ (+ 0.0 (- wm_k 1)) 64.0))
                (set! wm_eq (/ (+ (+ 0.0 (- wm_k 1)) (/ (- wm_t wm_c0) wm_ceil)) 64.0)))
            (if (> wm_eq wm_mx) (set! wm_mx wm_eq) #f)
            (set_item! ftab (+ off wm_i) (+ (* ALPHA wm_mx) (* (- 1.0 ALPHA) wm_t)))
            (set! wm_i (+ wm_i 1))
          })
          (set! wm_t (get_item ftab (+ off (- NG 1))))
          (if (< wm_t 0.000001) (set! wm_t 1.0) #f)
          (set! wm_i 0)
          (while (< wm_i NG) {
            (set_item! ftab (+ off wm_i) (/ (get_item ftab (+ off wm_i)) wm_t))
            (set! wm_i (+ wm_i 1))
          })
        })
  }))

(define p2d_i 0)
(define p2d_j 0)
(define p2d_x 0.0)
(define p2d_y 0.0)
(define p2d_x0 0)
(define p2d_y0 0)
(define p2d_x1 0)
(define p2d_y1 0)
(define p2d_ax 0.0)
(define p2d_ay 0.0)
(define normalize_p2d
  (lambda (src dst prow2 prow3 ftabv) {
    (set! p2d_i 0)
    (while (< p2d_i NG) {
      (set_item! prow2 p2d_i 0.0)
      (set_item! prow2 (+ NG p2d_i) 0.0)
      (set! p2d_i (+ p2d_i 1))
    })
    (set! p2d_i 0)
    (while (< p2d_i NG) {
      (set! p2d_j 0)
      (while (< p2d_j NG) {
        (set_item! prow2 p2d_j (+ (get_item prow2 p2d_j) (get_item src (+ (* p2d_i NG) p2d_j))))
        (set_item! prow2 (+ NG p2d_i) (+ (get_item prow2 (+ NG p2d_i)) (get_item src (+ (* p2d_i NG) p2d_j))))
        (set! p2d_j (+ p2d_j 1))
      })
      (set! p2d_i (+ p2d_i 1))
    })
    (set! p2d_i 0)
    (while (< p2d_i NG) {
      (set_item! prow3 p2d_i (get_item prow2 (+ NG p2d_i)))
      (set! p2d_i (+ p2d_i 1))
    })
    (build_warp prow2 0)
    (build_warp prow3 NG)
    (set! p2d_i 0)
    (while (< p2d_i NG) {
      (set! p2d_y (* (get_item ftabv (+ NG p2d_i)) 63.0))
      (set! p2d_y0 (Math.floor p2d_y))
      (if (< p2d_y0 0) (set! p2d_y0 0) #f)
      (if (> p2d_y0 62) (set! p2d_y0 62) #f)
      (set! p2d_y1 (+ p2d_y0 1))
      (set! p2d_ay (- p2d_y p2d_y0))
      (set! p2d_j 0)
      (while (< p2d_j NG) {
        (set! p2d_x (* (get_item ftabv p2d_j) 63.0))
        (set! p2d_x0 (Math.floor p2d_x))
        (if (< p2d_x0 0) (set! p2d_x0 0) #f)
        (if (> p2d_x0 62) (set! p2d_x0 62) #f)
        (set! p2d_x1 (+ p2d_x0 1))
        (set! p2d_ax (- p2d_x p2d_x0))
        (set_item! dst (+ (* p2d_i NG) p2d_j)
                   (+ (* (get_item src (+ (* p2d_y0 NG) p2d_x0)) (* (- 1.0 p2d_ay) (- 1.0 p2d_ax)))
                      (+ (* (get_item src (+ (* p2d_y0 NG) p2d_x1)) (* (- 1.0 p2d_ay) p2d_ax))
                         (+ (* (get_item src (+ (* p2d_y1 NG) p2d_x0)) (* p2d_ay (- 1.0 p2d_ax)))
                            (* (get_item src (+ (* p2d_y1 NG) p2d_x1)) (* p2d_ay p2d_ax))))))
        (set! p2d_j (+ p2d_j 1))
      })
      (set! p2d_i (+ p2d_i 1))
    })
  }))

;; ---- 卷积前向：conv1(1→C1, s2, p1) → ReLU → maxpool → conv2(C1→C2, s1, p1) → ReLU → avgpool
;; 权重以 int8（码点 256+q）存在字符串里；反量化用 (q-128)*scale + 偏置
;; 统一技巧：把 (q-256) 的偏移折进"输入和"，故热循环读 charCode 直接乘，最后减 384*Σx
(define cv_sum 0.0)
(define cv_acc 0.0)
(define cv_oy 0)
(define cv_ox 0)
(define cv_f 0)
(define cv_ky 0)
(define cv_kx 0)
(define cv_base 0)
(define cv_w 0.0)
(define cv_idx 0)
(define cv_psum 0.0)
;; conv1：1→C1，3×3，stride 2，pad 1；输入 34×34 零填充图 pad1(34×34)
;; 每个输出位置只读一次 3×3 输入块，再对 C1 个滤波器各算一次点积
(define c1_oy 0)
(define c1_ox 0)
(define c1_f 0)
(define c1_base 0)
(define c1_wb 0)
(define c1_acc 0.0)
(define p00 0.0)
(define p01 0.0)
(define p02 0.0)
(define p10 0.0)
(define p11 0.0)
(define p12 0.0)
(define p20 0.0)
(define p21 0.0)
(define p22 0.0)
(define pdi 0)
(define pdj 0)
(define pad_input
  (lambda (src dst) {
    (set! pdi 0)
    (while (< pdi (+ INS 2)) {
      (set! pdj 0)
      (while (< pdj (+ INS 2)) {
        (if (and (> pdi 0) (< pdi (+ INS 1)) (> pdj 0) (< pdj (+ INS 1)))
            (set_item! dst (+ (* pdi (+ INS 2)) pdj) (get_item src (+ (* (- pdi 1) INS) (- pdj 1))))
            (set_item! dst (+ (* pdi (+ INS 2)) pdj) 0.0))
        (set! pdj (+ pdj 1))
      })
      (set! pdi (+ pdi 1))
    })
  }))

(define conv1_forward
  (lambda (src dst wv bv) {
    (define oy 0)
    (define ox 0)
    (define f 0)
    (define base 0)
    (define wb 0)
    (define acc 0.0)
    (define p00 0.0)
    (define p01 0.0)
    (define p02 0.0)
    (define p10 0.0)
    (define p11 0.0)
    (define p12 0.0)
    (define p20 0.0)
    (define p21 0.0)
    (define p22 0.0)
    (define W (+ INS 2))
    (while (< oy O1) {
      (set! ox 0)
      (while (< ox O1) {
        (set! base (+ (* (* oy 2) W) (* ox 2)))
        (set! p00 (get_item src base))
        (set! p01 (get_item src (+ base 1)))
        (set! p02 (get_item src (+ base 2)))
        (set! p10 (get_item src (+ base W)))
        (set! p11 (get_item src (+ base (+ W 1))))
        (set! p12 (get_item src (+ base (+ W 2))))
        (set! p20 (get_item src (+ base (* W 2))))
        (set! p21 (get_item src (+ base (+ (* W 2) 1))))
        (set! p22 (get_item src (+ base (+ (* W 2) 2))))
        (set! f 0)
        (set! wb 0)
        (while (< f C1) {
          (set! acc (get_item bv f))
          (set! acc (+ acc (* (get_item wv wb) p00)))
          (set! acc (+ acc (* (get_item wv (+ wb 1)) p01)))
          (set! acc (+ acc (* (get_item wv (+ wb 2)) p02)))
          (set! acc (+ acc (* (get_item wv (+ wb 3)) p10)))
          (set! acc (+ acc (* (get_item wv (+ wb 4)) p11)))
          (set! acc (+ acc (* (get_item wv (+ wb 5)) p12)))
          (set! acc (+ acc (* (get_item wv (+ wb 6)) p20)))
          (set! acc (+ acc (* (get_item wv (+ wb 7)) p21)))
          (set! acc (+ acc (* (get_item wv (+ wb 8)) p22)))
          (if (< acc 0.0) (set! acc 0.0) #f)
          (set_item! dst (+ (* f (* O1 O1)) (+ (* oy O1) ox)) acc)
          (set! wb (+ wb 9))
          (set! f (+ f 1))
        })
        (set! ox (+ ox 1))
      })
      (set! oy (+ oy 1))
    })
  }))

(define mp_i 0)
(define mp_j 0)
(define mp_f 0)
(define mp_v 0.0)
(define maxpool1
  (lambda (src dst) {
    (set! mp_f 0)
    (while (< mp_f C1) {
      (set! mp_i 0)
      (while (< mp_i O2) {
        (set! mp_j 0)
        (while (< mp_j O2) {
          (set! cv_base (+ (* mp_f (* O1 O1)) (+ (* (* mp_i 2) O1) (* mp_j 2))))
          (set! mp_v (get_item src cv_base))
          (set! mp_v (maxf mp_v (get_item src (+ cv_base 1))))
          (set! mp_v (maxf mp_v (get_item src (+ cv_base O1))))
          (set! mp_v (maxf mp_v (get_item src (+ cv_base (+ O1 1)))))
          (set_item! dst (+ (* mp_f (* O2 O2)) (+ (* mp_i O2) mp_j)) mp_v)
          (set! mp_j (+ mp_j 1))
        })
        (set! mp_i (+ mp_i 1))
      })
      (set! mp_f (+ mp_f 1))
    })
  }))

(define cv2_oy 0)
(define cv2_ox 0)
(define cv2_f 0)
(define cv2_c 0)
(define cv2_ky 0)
(define cv2_kx 0)
(define cv2_acc 0.0)
(define cv2_base 0)
(define cv2_sidx 0)
(define cv2_psum 0.0)
(define cv2_ws "")
(define cv2_Wn 10)
;; conv2：C1→C2，3×3，stride 1，pad 1；输入 q1（C1×8×8，通道主序）
;; 每个输出位置先把 C1×3×3 = C1*9 个输入值读进 scratch，再算 C2 个点积（避免重复读 16 遍）
(define pat (mkn (* C1 9) 0.0))
(define c2_oy 0)
(define c2_ox 0)
(define c2_f 0)
(define c2_k 0)
(define c2_acc 0.0)
(define c2_sidx 0)
(define c2_r0 0)
(define c2_c 0)
(define c2_ky 0)
(define c2_wb 0)
(define conv2_forward
  (lambda (src dst wv bv) {
    (define oy 0)
    (define ox 0)
    (define f 0)
    (define k 0)
    (define c 0)
    (define ky 0)
    (define acc 0.0)
    (define wb 0)
    (define sidx 0)
    (define r0 0)
    (define nk (* C1 9))
    (define pat (mkn nk 0.0))
    (while (< oy O2) {
      (set! ox 0)
      (while (< ox O2) {
        (set! c 0)
        (set! k 0)
        (while (< c C1) {
          (set! ky 0)
          (while (< ky 3) {
            (set! r0 (+ (* c (* O2 O2)) (* (- (+ oy ky) 1) O2)))
            (set! sidx (+ r0 ox))
            (if (and (> (+ oy ky) 0) (< (+ oy ky) (+ O2 1)))
                {
                  (set_item! pat k (if (> ox 0) (get_item src (- sidx 1)) 0.0))
                  (set_item! pat (+ k 1) (get_item src sidx))
                  (set_item! pat (+ k 2) (if (< ox (- O2 1)) (get_item src (+ sidx 1)) 0.0))
                  (set! k (+ k 3))
                }
                {
                  (set_item! pat k 0.0)
                  (set_item! pat (+ k 1) 0.0)
                  (set_item! pat (+ k 2) 0.0)
                  (set! k (+ k 3))
                })
            (set! ky (+ ky 1))
          })
          (set! c (+ c 1))
        })
        (set! f 0)
        (set! wb 0)
        (while (< f C2) {
          (set! acc (get_item bv f))
          (set! k 0)
          (while (< k nk) {
            (set! acc (+ acc (* (get_item wv (+ wb k)) (get_item pat k))))
            (set! k (+ k 1))
          })
          (if (< acc 0.0) (set! acc 0.0) #f)
          (set_item! dst (+ (* f (* O2 O2)) (+ (* oy O2) ox)) acc)
          (set! wb (+ wb nk))
          (set! f (+ f 1))
        })
        (set! ox (+ ox 1))
      })
      (set! oy (+ oy 1))
    })
  }))

(define ap_f 0)
(define ap_i 0)
(define ap_j 0)
(define ap_s 0.0)
(define ap_base 0)
(define avgpool2
  (lambda (src dst) {
    (set! ap_f 0)
    (while (< ap_f C2) {
      (set! ap_i 0)
      (while (< ap_i 4) {
        (set! ap_j 0)
        (while (< ap_j 4) {
          (set! ap_base (+ (* ap_f (* O2 O2)) (+ (* (* ap_i 2) O2) (* ap_j 2))))
          (set! ap_s (+ (get_item src ap_base) (get_item src (+ ap_base 1))))
          (set! ap_s (+ ap_s (get_item src (+ ap_base O2))))
          (set! ap_s (+ ap_s (get_item src (+ ap_base (+ O2 1)))))
          ;; flat 布局：channel major（与训练侧 p2.reshape(N,-1) 一致）
          (set_item! dst (+ (* ap_f 16) (+ (* ap_i 4) ap_j)) (/ ap_s 4.0))
          (set! ap_j (+ ap_j 1))
        })
        (set! ap_i (+ ap_i 1))
      })
      (set! ap_f (+ ap_f 1))
    })
  }))

;; ---- fc → 128 维嵌入（L2 归一化） ---------------------------------------
(define fc_i 0)
(define fc_j 0)
(define fc_acc 0.0)
(define fc_sum 0.0)
(define fc_off 0)
(define fc_p 0)
(define fc_pend 0)
(define fc_ps "")
(define fc_n 0.0)
;; fc：256 → nred，然后 L2 归一化
(define fc_forward
  (lambda (fv ev wv bv) {
    (define k 0)
    (define j 0)
    (define wb 0)
    (define acc 0.0)
    (define nv 0.0)
    (while (< k NRED) {
      (set! acc (get_item bv k))
      (set! j 0)
      (while (< j NFLAT) {
        (set! acc (+ acc (* (get_item wv (+ wb j)) (get_item fv j))))
        (set! j (+ j 1))
      })
      (set_item! ev k acc)
      (set! nv (+ nv (* acc acc)))
      (set! wb (+ wb NFLAT))
      (set! k (+ k 1))
    })
    (set! nv (Math.sqrt nv))
    (if (< nv 0.000001) (set! nv 1.0) #f)
    (set! k 0)
    (while (< k NRED) {
      (set_item! ev k (/ (get_item ev k) nv))
      (set! k (+ k 1))
    })
  }))

;; ---- 最近原型（余弦）+ top-16 ------------------------------------------
(define ins_p 0)
(define top_insert
  (lambda (topvv topkk ntop k v) {
    (define p 0)
    (if (> v (get_item topvv (- ntop 1)))
        {
          (set! p (- ntop 1))
          (while (> p 0) {
            (if (> v (get_item topvv (- p 1)))
                {
                  (set_item! topvv p (get_item topvv (- p 1)))
                  (set_item! topkk p (get_item topkk (- p 1)))
                  (set! p (- p 1))
                }
                break)
          })
          (set_item! topvv p v)
          (set_item! topkk p k)
        }
        #f)
  }))
(define cl_k 0)
(define cl_j 0)
(define cl_p 0)
(define cl_pend 0)
(define cl_ps "")
(define cl_off 0)
(define cl_acc 0.0)
(define cl_sumz 0.0)
(define cl_sc 0.0)
(define classify
  (lambda (ev topvv topkk) {
    (define nc NCLS)
    (define per M.per_rows)
    (define k 0)
    (define j 0)
    (define p 0)
    (define pend per)
    (define ps "")
    (define off 0)
    (define acc 0.0)
    (define sumz 0.0)
    (define sc 0.0)
    (while (< j NRED) {
      (set! sumz (+ sumz (get_item ev j)))
      (set! j (+ j 1))
    })
    (set! ps (get_item pnparts 0))
    (while (< k nc) {
      (if (>= k pend)
          {
            (set! p (+ p 1))
            (set! pend (+ pend per))
            (set! ps (get_item pnparts p))
            (set! off 0)
          }
          #f)
      (set! acc 0.0)
      (set! j 0)
      (while (< j NRED) {
        (set! acc (+ acc (* (get_item ev j) (String.charCodeAt (+ off j) ps))))
        (set! j (+ j 1))
      })
      (set! sc (* (get_item spnl k) (- acc (* 384.0 sumz))))
      (top_insert topvv topkk NTOP k sc)
      (set! off (+ off NRED))
      (set! k (+ k 1))
    })
  }))

;; ---- 识别 ---------------------------------------------------------------
(define pool_i 0)
(define pool_j 0)
(define pool_v 0.0)
;; 64×64 归一化墨迹 → 32×32（2×2 均值），与训练侧 img.reshape(32,2,32,2).mean 一致
(define pool2x2
  (lambda () {
    (set! pool_i 0)
    (while (< pool_i INS) {
      (set! pool_j 0)
      (while (< pool_j INS) {
        (set! pool_v (+ (get_item nrm (+ (* (* pool_i 2) NG) (* pool_j 2)))
                        (+ (get_item nrm (+ (* (* pool_i 2) NG) (+ (* pool_j 2) 1)))
                           (+ (get_item nrm (+ (* (+ (* pool_i 2) 1) NG) (* pool_j 2)))
                              (get_item nrm (+ (* (+ (* pool_i 2) 1) NG) (+ (* pool_j 2) 1)))))))
        (set_item! inp (+ (* pool_i INS) pool_j) (* 0.25 pool_v))
        (set! pool_j (+ pool_j 1))
      })
      (set! pool_i (+ pool_i 1))
    })
  }))

(define recognize
  (lambda () {
    (define t0 (Diag.ms))
    (define mx 0.0)
    (define s 0.0)
    (define i 0)
    (define v 0.0)
    (define imax 0.0)
    (build_dens strokes dens)
    (normalize_p2d dens nrm prow prowb ftab)
    ;; 归一化到最大值为 1（与训练侧 img/img.max() 一致）
    (set! imax 0.0)
    (set! i 0)
    (while (< i (* NG NG)) {
      (if (> (get_item nrm i) imax) (set! imax (get_item nrm i)) #f)
      (set! i (+ i 1))
    })
    (if (> imax 0.0000001)
        {
          (set! i 0)
          (while (< i (* NG NG)) {
            (set_item! nrm i (/ (get_item nrm i) imax))
            (set! i (+ i 1))
          })
        }
        #f)
    (pool2x2)
    (pad_input inp pad1)
    (conv1_forward pad1 r1 w1v b1l)
    (maxpool1 r1 q1)
    (conv2_forward q1 r2 w2v b2l)
    (avgpool2 r2 flat)
    (fc_forward flat emb wfcv bfl)
    (set! feat_ms (- (Diag.ms) t0))
    (classify emb topv topk)
    (set! page 0)
    (set! cand_n NTOP)
    ;; 候选内 softmax（显示相对置信度）
    (set! mx (get_item topv 0))
    (set! i 0)
    (set! s 0.0)
    (while (< i cand_n) {
      (set! v (Math.exp (- (get_item topv i) mx)))
      (set_item! topv i v)
      (set! s (+ s v))
      (set! i (+ i 1))
    })
    (if (> s 0.0)
        {
          (set! i 0)
          (while (< i cand_n) {
            (set_item! topv i (/ (get_item topv i) s))
            (set! i (+ i 1))
          })
        }
        #f)
    (set! infer_ms (- (Diag.ms) t0))
  }))

;; ---- 绘制 ---------------------------------------------------------------
(define dc_g 0)
(define draw_cell
  (lambda (inkv i j) {
    (define v (get_item inkv (+ (* i INK) j)))
    (set! dc_g (Math.round (- 255.0 (* 255.0 (minf 1.0 (maxf 0.0 v))))))
    (Gfx.fill_rect (+ PAD_X (* j CELLPX)) (+ PAD_Y (* i CELLPX))
                   CELLPX CELLPX dc_g dc_g dc_g)
  }))
(define draw_pad_full
  (lambda (inkv) {
    (define i 0)
    (define j 0)
    (while (< i INK) {
      (set! j 0)
      (while (< j INK) {
        (draw_cell inkv i j)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
  }))
(define dpd_i 0)
(define dpd_j 0)
(define draw_pad_dirty
  (lambda (inkv) {
    (if (>= dirty_x1 dirty_x0)
        {
          (set! dpd_i dirty_y0)
          (while (<= dpd_i dirty_y1) {
            (set! dpd_j dirty_x0)
            (while (<= dpd_j dirty_x1) {
              (draw_cell inkv dpd_i dpd_j)
              (set! dpd_j (+ dpd_j 1))
            })
            (set! dpd_i (+ dpd_i 1))
          })
          (Gfx.refresh)
        }
        #f)
    (set! dirty_x0 0)
    (set! dirty_y0 0)
    (set! dirty_x1 -1)
    (set! dirty_y1 -1)
  }))
(define draw_btn
  (lambda (x y w h label) {
    (Gfx.fill_rect x y w h 226 229 236)
    (Gfx.draw_rect x y w h 170 176 190)
    (Gfx.draw_text label (+ x (Math.round (/ (- w (tw label)) 2.0))) (+ y 2) 45 50 70)
  }))
(define draw_header
  (lambda () {
    (Gfx.fill_rect 0 0 SW 18 248 249 252)
    (Gfx.draw_text "手写识别" 4 3 40 44 62)
    (draw_btn HDR_BTN_X 2 HDR_BTN_W 14 "清屏")
    (if (> infer_ms 0)
        (Gfx.draw_text (String.concat (String.concat "识别 " (i2s infer_ms)) "ms")
                       112 4 150 155 172)
        (Gfx.draw_text "写一个字，松手静置即识别" 146 4 150 155 172))
  }))
(define dcd_i 0)
(define dcd_x 0)
(define dcd_y 0)
(define dcd_ci 0)
(define dcd_hot 0)
(define draw_candidates
  (lambda () {
    (define i 0)
    (Gfx.draw_text (String.concat "候选（点按上屏） " (i2s (+ page 1))) CX 40 55 62 88)
    (Gfx.draw_text "/2" 296 40 150 155 172)
    (set! dcd_i 0)
    (while (< dcd_i PAGE_N) {
      (set! dcd_x (+ CX (* dcd_i CCELL)))
      (set! dcd_y (+ CY (* 24 (Math.floor (/ dcd_i 8.0)))))
      (set! dcd_ci (+ (* page PAGE_N) dcd_i))
      (if (< dcd_ci cand_n)
          {
            (set! dcd_hot (if (and (== page 0) (== dcd_i 0)) 1 0))
            (if (== dcd_hot 1)
                (Gfx.fill_rect dcd_x dcd_y CBOX 22 214 231 250)
                (Gfx.fill_rect dcd_x dcd_y CBOX 22 242 244 248))
            (Gfx.draw_rect dcd_x dcd_y CBOX 22 198 204 216)
            (Gfx.draw_text (cls_at (get_item topk dcd_ci)) (+ dcd_x 4) (+ dcd_y 4)
                           (if (== dcd_hot 1) 20 60 160)
                           (if (== dcd_hot 1) 70 70 100)
                           (if (== dcd_hot 1) 190 90 100))
          }
          #f)
      (set! dcd_i (+ dcd_i 1))
    })
    (Gfx.draw_text "置信度" CX 108 120 126 142)
    (set! dcd_i 0)
    (while (< dcd_i 3) {
      (set! dcd_y (+ 124 (* dcd_i 15)))
      (Gfx.fill_rect CX dcd_y 130 10 235 238 243)
      (Gfx.fill_rect CX dcd_y (Math.round (* 130.0 (get_item topv dcd_i))) 10 90 140 210)
      (Gfx.draw_text (fmt_pct (get_item topv dcd_i)) (+ CX 134) dcd_y 100 106 124)
      (set! dcd_i (+ dcd_i 1))
    })
  }))

;; 底部：归一化墨迹（32×32，1px）+ 网络输入 32×32（1px，放大 2 倍看）+ 8 个 conv1 特征图
(define dv_i 0)
(define dv_j 0)
(define dv_f 0)
(define dv_v 0.0)
(define dv_mx 0.0)
(define dv_l 0)
(define draw_viz
  (lambda () {
    (define i 0)
    (define j 0)
    (Gfx.draw_text "归一化" 4 VIZ_LY 120 126 142)
    (Gfx.draw_text "网络输入 32×32 / conv1 特征图" 40 VIZ_LY 120 126 142)
    (set! dv_i 0)
    (while (< dv_i 32) {
      (set! dv_j 0)
      (while (< dv_j 32) {
        (set! dv_v (minf 1.0 (maxf 0.0 (get_item nrm (+ (* (* dv_i 2) NG) (* dv_j 2))))))
        (set! dv_l (Math.round (- 255.0 (* 255.0 dv_v))))
        (Gfx.fill_rect (+ 4 dv_j) (+ VIZ_Y dv_i) 1 1 dv_l dv_l dv_l)
        (set! dv_j (+ dv_j 1))
      })
      (set! dv_i (+ dv_i 1))
    })
    (Gfx.draw_rect 3 (- VIZ_Y 1) 34 34 205 210 220)
    ;; conv1 的 C1 张特征图（16×16，每格 2px）
    (set! dv_f 0)
    (while (< dv_f C1) {
      (set! dv_mx 0.0001)
      (set! dv_i 0)
      (while (< dv_i (* O1 O1)) {
        (if (> (get_item r1 (+ (* dv_f (* O1 O1)) dv_i)) dv_mx)
            (set! dv_mx (get_item r1 (+ (* dv_f (* O1 O1)) dv_i)))
            #f)
        (set! dv_i (+ dv_i 1))
      })
      (set! dv_i 0)
      (while (< dv_i O1) {
        (set! dv_j 0)
        (while (< dv_j O1) {
          (set! dv_v (minf 1.0 (maxf 0.0
                (/ (get_item r1 (+ (* dv_f (* O1 O1)) (+ (* dv_i O1) dv_j))) dv_mx))))
          (if (< dv_v 0.5)
              (Gfx.fill_rect (+ (+ 40 (* dv_f 34)) (* dv_j 2)) (+ VIZ_Y (* dv_i 2)) 2 2
                             (Math.round (- 252.0 (* 3.0 dv_v)))
                             (Math.round (- 250.0 (* 125.0 dv_v)))
                             (Math.round (- 247.0 (* 382.0 dv_v))))
              (Gfx.fill_rect (+ (+ 40 (* dv_f 34)) (* dv_j 2)) (+ VIZ_Y (* dv_i 2)) 2 2
                             (Math.round (- 255.0 (* 90.0 (- (* 2.0 dv_v) 1.0))))
                             (Math.round (- 190.0 (* 150.0 (- (* 2.0 dv_v) 1.0))))
                             (Math.round (- 60.0 (* 20.0 (- (* 2.0 dv_v) 1.0))))))
          (set! dv_j (+ dv_j 1))
        })
        (set! dv_i (+ dv_i 1))
      })
      (Gfx.draw_rect (- (+ 40 (* dv_f 34)) 1) (- VIZ_Y 1) 34 34 205 210 220)
      (set! dv_f (+ dv_f 1))
    })
  }))

(define draw_textline
  (lambda () {
    (define n (String.length text))
    (Gfx.draw_text "已输入" 4 22 90 96 115)
    (if (> n 0)
        (Gfx.draw_text (String.slice text (if (> n 16) (- n 16) 0) n) 52 22 25 35 70)
        (Gfx.draw_text "（点候选字上屏）" 52 22 175 182 196))
    (draw_btn BS_X 20 BS_W 15 "退格")
    (draw_btn CL_X 20 CL_W 15 "全清")
  }))
(define draw_load_err
  (lambda () {
    (Gfx.fill_white)
    (Gfx.draw_text "手写识别" 4 3 40 44 62)
    (Gfx.draw_text "模型数据缺失" 8 60 200 60 40)
    (Gfx.draw_text "请确认 app/lib/hwr3/ 数据目录已随 App 一并拷贝" 8 84 90 96 115)
    (EX.draw_button)
    (Gfx.refresh)
  }))
(define render
  (lambda () {
    (Gfx.fill_white)
    (draw_header)
    (draw_textline)
    (draw_pad_full ink)
    (Gfx.draw_rect (- PAD_X 2) (- PAD_Y 2) (+ PAD_PX 4) (+ PAD_PX 4) 205 210 220)
    (draw_candidates)
    (draw_viz)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; ---- 命中测试 / 编辑 ----------------------------------------------------
(define hit_at
  (lambda (px py) {
    (define q 0)
    (define cx 0)
    (define cy -1)
    (define i 0)
    (if (== (hit_rect px py HDR_BTN_X 2 HDR_BTN_W 14) 1) (set! q 1) #f)
    (if (== (hit_rect px py BS_X 20 BS_W 15) 1) (set! q 2) #f)
    (if (== (hit_rect px py CL_X 20 CL_W 15) 1) (set! q 3) #f)
    (if (== (hit_rect px py CX 38 178 16) 1) (set! q 4) #f)
    (if (and (>= px CX) (< px 314))
        {
          (set! cx (Math.floor (/ (- px CX) CCELL)))
          (if (and (>= py CY) (< py (+ CY 22))) (set! cy 0) #f)
          (if (and (>= py (+ CY 24)) (< py (+ CY 46))) (set! cy 1) #f)
          (if (>= cy 0)
              {
                (set! i (+ (* cy 8) cx))
                (if (< i PAGE_N)
                    (if (< (+ (* page PAGE_N) i) cand_n)
                        (set! q (+ 10 i))
                        #f)
                    #f)
              }
              #f)
        }
        #f)
    q
  }))
(define pad_hit
  (lambda (px py)
    (if (and (>= px PAD_X) (< px (+ PAD_X PAD_PX)))
        (if (and (>= py PAD_Y) (< py (+ PAD_Y PAD_PX))) 1 0)
        0)))
(define pad_cell_x (lambda (px) (/ (- px PAD_X) CELLPX)))
(define pad_cell_y (lambda (py) (/ (- py PAD_Y) CELLPX)))
(define commit
  (lambda (i) {
    (set! text (String.concat text (cls_at (get_item topk i))))
    (clear_all_ink)
    (set! cand_n 0)
    (render)
  }))
(define backspace
  (lambda () {
    (define n (String.length text))
    (if (> n 0) (set! text (String.slice text 0 (- n 1))) #f)
  }))
(define pen_move
  (lambda (tx ty) {
    (define gx (pad_cell_x tx))
    (define gy (pad_cell_y ty))
    (if (< gx 0.0) (set! gx 0.0) #f)
    (if (< gy 0.0) (set! gy 0.0) #f)
    (if (> gx 63.0) (set! gx 63.0) #f)
    (if (> gy 63.0) (set! gy 63.0) #f)
    (if (== (pad_hit tx ty) 1)
        {
          (trace_to gx gy)
          (draw_pad_dirty ink)
        }
        {
          (set! pen_x gx)
          (set! pen_y gy)
        })
  }))
(define do_release_armed
  (lambda (h tx ty) {
    (if (== (hit_at tx ty) h)
        {
          (if (== h 1)
              {
                (clear_all_ink)
                (set! cand_n 0)
                (render)
              }
              (if (== h 2)
                  {
                    (backspace)
                    (render)
                  }
                  (if (== h 3)
                      {
                        (set! text "")
                        (clear_all_ink)
                        (set! cand_n 0)
                        (render)
                      }
                      (if (== h 4)
                          {
                            (set! page (if (== page 0) 1 0))
                            (render)
                          }
                          (if (>= h 10)
                              (commit (+ (* page PAGE_N) (- h 10)))
                              #f)))))
        }
        #f)
  }))
(define tick
  (lambda () {
    (define tp (if (Touch.take_press) 1 0))
    (define tr (if (Touch.take_release) 1 0))
    (define tx (Touch.x))
    (define ty (Touch.y))
    (EX.check)
    (if (== tp 1)
        {
          (set! armed (hit_at tx ty))
          (if (== armed 0)
              (if (== (pad_hit tx ty) 1)
                  {
                    (if (> up_ms 0) #f { (clear_all_ink) (set! cand_n 0) })
                    (set! up_ms 0)
                    (set! pen_down 1)
                    (set! pen_x (pad_cell_x tx))
                    (set! pen_y (pad_cell_y ty))
                    (set! cpts (cdr '(0)))
                    (push cpts (cons pen_x (cons pen_y (cdr '(0)))))
                    (stamp ink pen_x pen_y)
                    (render)
                  }
                  #f)
              #f)
        }
        #f)
    (if (== pen_down 1)
        (if (Touch.is_pressed) (pen_move tx ty) #f)
        #f)
    (if (== tr 1)
        {
          (if (> armed 0) (do_release_armed armed tx ty) #f)
          (if (== pen_down 1)
              {
                (set! pen_down 0)
                (if (== has_ink 1)
                    {
                      (push strokes cpts)
                      (set! cpts (cdr '(0)))
                      (set! up_ms (+ (System.timestamp) 1))
                    }
                    #f)
              }
              #f)
          (set! armed 0)
        }
        #f)
    (if (> up_ms 0)
        (if (>= (System.timestamp) (+ up_ms PEND_MS))
            {
              (set! up_ms 0)
              (Gfx.draw_text "识别中…" 146 4 200 120 40)
              (EX.draw_button)
              (Gfx.refresh)
              (recognize)
              (render)
            }
            #f)
        #f)
  }))

;; ---- 启动 ---------------------------------------------------------------
(define ld_i 0)
(define ld_p "")
(define ld_load
  (lambda (prefix n lst) {
    (set! ld_i 0)
    (while (< ld_i n) {
      (set! ld_p (Fs.read (String.concat "app/lib/hwr3/"
                     (String.concat prefix (String.concat (i2s ld_i) ".txt")))))
      (if (null? ld_p) (set! load_err 1) (if (> (String.length ld_p) 0) #f (set! load_err 1)))
      (push lst ld_p)
      (set! ld_i (+ ld_i 1))
    })
  }))
(define lm_o 0)
(define lm_n 0)
(define load_model
  (lambda () {
    (ld_load "c1" M.nc1files c1parts)
    (ld_load "c2" M.nc2files c2parts)
    (ld_load "fc" M.nfcfiles fcparts)
    (ld_load "pn" M.npnfiles pnparts)
    (set! params (Fs.read "app/lib/hwr3/params.txt"))
    (if (null? params)
        (set! load_err 1)
        {
          (set! lm_n (+ C1 C2))
          (set! lm_n (+ lm_n NRED))
          (set! lm_n (+ lm_n (* 2 NRED)))
          (set! lm_n (+ lm_n (* 2 NCLS)))
          (set! lm_n (+ lm_n NCLS))
          (if (== (String.length params) lm_n)
              {
                (set! b1s (String.slice params 0 C1))
                (set! lm_o C1)
                (set! b2s (String.slice params lm_o (+ lm_o C2)))
                (set! lm_o (+ lm_o C2))
                (set! bfs (String.slice params lm_o (+ lm_o NRED)))
                (set! lm_o (+ lm_o NRED))
                (set! sfcs (String.slice params lm_o (+ lm_o (* 2 NRED))))
                (set! lm_o (+ lm_o (* 2 NRED)))
                (set! spns (String.slice params lm_o (+ lm_o (* 2 NCLS))))
                (set! lm_o (+ lm_o (* 2 NCLS)))
                (set! cstr (String.slice params lm_o (+ lm_o NCLS)))
              }
              (set! load_err 1))
        })
  }))

(define decode_params
  (lambda () {
    (define i 0)
    (while (< i C1) {
      (set_item! b1l i (* (- (String.charCodeAt i b1s) 384) M.sb1))
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i C2) {
      (set_item! b2l i (* (- (String.charCodeAt i b2s) 384) M.sb2))
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i NRED) {
      (set_item! bfl i (* (- (String.charCodeAt i bfs) 384) M.sbf))
      (set_item! sfcl i (* M.swfbase (/ (+ (* 256 (- (String.charCodeAt (* i 2) sfcs) 256))
                                            (- (String.charCodeAt (+ (* i 2) 1) sfcs) 256))
                                        65535.0)))
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i NCLS) {
      (set_item! spnl i (* M.scale (* M.spnbase (/ (+ (* 256 (- (String.charCodeAt (* i 2) spns) 256))
                                                     (- (String.charCodeAt (+ (* i 2) 1) spns) 256))
                                                 65535.0))))
      (set! i (+ i 1))
    })
  }))

(load_model)
(define dw_i 0)
(define dw_s "")
(define dw_row 0)
(define dw_f 0)
(define dw_r 0)
(define dw_wb 0)
(define dw_sc 0.0)
(define decode_weights
  (lambda () {
    (set! dw_i 0)
    (set! dw_s (get_item c1parts 0))
    (while (< dw_i (* C1 9)) {
      (push w1v (* (- (String.charCodeAt dw_i dw_s) 384) M.sw1))
      (set! dw_i (+ dw_i 1))
    })
    (set! dw_i 0)
    (set! dw_s (get_item c2parts 0))
    (while (< dw_i (* C2 (* C1 9))) {
      (push w2v (* (- (String.charCodeAt dw_i dw_s) 384) M.sw2))
      (set! dw_i (+ dw_i 1))
    })
    ;; fc：每行一个尺度（行号 = 文件号×per_rows + 行内序号）
    (set! dw_f 0)
    (while (< dw_f M.nfcfiles) {
      (set! dw_s (get_item fcparts dw_f))
      (set! dw_r 0)
      (while (< dw_r M.per_rows) {
        (set! dw_row (+ (* dw_f M.per_rows) dw_r))
        (if (< dw_row NRED)
            {
              (set! dw_sc (get_item sfcl dw_row))
              (set! dw_wb 0)
              (while (< dw_wb NFLAT) {
                (push wfcv (* (- (String.charCodeAt (+ (* dw_r NFLAT) dw_wb) dw_s) 384) dw_sc))
                (set! dw_wb (+ dw_wb 1))
              })
            }
            #f)
        (set! dw_r (+ dw_r 1))
      })
      (set! dw_f (+ dw_f 1))
    })
  }))

(if (== load_err 0) (decode_params) #f)
(if (== load_err 0) (decode_weights) #f)

(if (== load_err 1) (draw_load_err) (render))
(set! timer_id (System.set_interval TICK_MS tick))
(EX.install_with render)