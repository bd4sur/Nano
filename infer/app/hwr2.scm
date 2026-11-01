;; ============================================================================
;; hwr2.scm —— 手写汉字识别（在线笔迹特征版，3755 类，纯 Scheme + WASM）
;;
;; 与 hwr.scm（图像特征 + 1000 类，字体合成数据训练）并存：本文件是"第二版"，
;; 换成了**在线（轨迹）特征 + 真实笔迹训练**，类数扩到 GB2312 一级字 3755 个。
;;
;; 识别流程（与 tools/train_hwr_online.py 逐位一致）：
;;   1) 采笔迹轨迹（每 20ms 一个点），绘图板 128×128px = 64×64 格（每格 2px）
;;   2) 细线墨迹图 64×64：按段双线性铺点（笔宽不参与特征，纯轨迹）
;;   3) 伪二维归一化：按行/列线密度投影做等化映射（alpha=0.7 与恒等混合）后重采样
;;   4) 8 方向梯度（中心差分 + 相邻方向软分配）→ 8×8 分块求和 → 512 维
;;      每格 8 方向向量 L2 归一化，最后整体 L2
;;   5) 投影到 128 维（PCA + 类内散布对角白化，白化已折进投影矩阵）
;;   6) 最近类均值（NCM）判 3755 类，取前 16 名做候选（两页各 8 个）
;;
;; 训练数据：CASIA-OLHWDB1.1 原始笔迹（240 书写人训练 / 60 书写人测试，3755 类）。
;;   实测（App 输入仿真：60 格量化 + 2.5 格抽稀）：top-1 58.5% / top-5 79.1%。
;;   权重 int8 量化后精度不变（58.41%/79.05%）。详见 tools/train_hwr_online.py 与
;;   doc/手写识别技术路线调研.md。
;;
;; 开销（WASM 实测见 doc/AGENTS.md）：特征 ≈ 60ms，投影 512×128 ≈ 80ms，
;;   判类 3755×128 ≈ 580ms —— 单次识别约 0.7 秒；拖动笔画期间只有盖章开销。
;;
;; 解释器约束（Animac manual.md，逐条遵守）：算术/比较固定二参数（三元会静默算错）；
;;   define 只在 lambda 体顶层；/ 与 mod 恒返回浮点；'() 字面量是静态槽；
;;   Gfx 坐标是 uint32（负数先钳）；热循环一律用 lambda 形参/局部量（比模块级变量快 3 倍）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Diag)
(native Fs)

(import EX "app/lib/exit.scm")
(import M "app/lib/hwr2/meta.scm")     ; 类数/维数/降维数/量化尺度（自动生成）

;; 模型数据（数据文件而非源码：灵机按源码字符数预分配 token 数组，见 hwr.scm 注释）
(define pparts (cdr '(0)))             ; PCA 投影矩阵 (nfeat × nred)，int8
(define mparts (cdr '(0)))             ; 类均值 (ncls × nred)，int8 逐类量化
(define params "")
(define sstr "")                       ; 每类尺度（2 字符 16 位定点）
(define nstr "")                       ; 每类 |m|²（2 字符 16 位定点）
(define cstr "")                       ; 汉字表
(define load_err 0)

;; ---- 布局常量 -------------------------------------------------------------
(define SW (Gfx.width))
(define TICK_MS 20)

(define PAD_X 4)
(define PAD_Y 36)
(define INK 64)                        ; 墨迹网格边长（格）
(define CELLPX 2)                      ; 每格像素（64*2 = 128）
(define PAD_PX (* INK CELLPX))

(define NG 64)                         ; 特征图分辨率（= 墨迹网格）
(define GRID 8)                        ; 分块网格（每格 NG/GRID = 8 像素）
(define NDIR 8)
(define NFEAT 512)
(define NRED M.nred)
(define NCLS M.ncls)
(define NTOP 16)
(define PAGE_N 8)

(define BRUSH 2.0)                     ; 显示用笔刷半径（只影响观感，不参与特征）
(define PEN_STEP 0.5)
(define PEND_MS 420)                   ; 抬笔静置多久算"一个字写完"
(define ALPHA 0.7)                     ; 伪二维等化强度（与训练一致）
(define SMOOTH 2)                      ; 投影平滑次数（与训练一致）

(define CX 138)
(define CY 56)
(define CCELL 22)
(define CBOX 20)
(define VIZ_LY 167)
(define VIZ_Y 183)
(define HDR_BTN_X 84)
(define HDR_BTN_W 40)
(define BS_X 252)
(define BS_W 32)
(define CL_X 286)
(define CL_W 30)

;; ---- 小工具 --------------------------------------------------------------
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

;; ---- 缓冲区与状态 --------------------------------------------------------
(define ink (mkn (* INK INK) 0.0))     ; 显示用墨迹（带笔宽）
(define dens (mkn (* NG NG) 0.0))      ; 特征用细线墨迹图
(define nrm (mkn (* NG NG) 0.0))       ; 伪二维归一化后的墨迹图
(define feat (mkn NFEAT 0.0))          ; 512 维特征
(define red (mkn NRED 0.0))            ; 128 维降维特征
(define fproj (mkn (* (* NDIR GRID) GRID) 0.0))   ; 分块方向直方图（临时）
(define prow (mkn (* 4 NG) 0.0))        ; 投影/平滑/累积分布（临时）
(define prowb (mkn (* 4 NG) 0.0))       ; y 方向投影的独立临时缓冲
(define ftab (mkn (* 2 NG) 0.0))       ; x/y 两条等化映射表（各 NG 项）
(define scal (mkn NCLS 0.0))           ; 每类尺度（启动解码）
(define nrm2 (mkn NCLS 0.0))           ; 每类 |m|²
(define topk (mkn NTOP 0))
(define topv (mkn NTOP 0.0))

(define strokes (cdr '(0)))            ; 已完成笔画：每项是点表 ((x y) ...)
(define cpts (cdr '(0)))               ; 当前笔画点表
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

;; ---- 绘图板：显示用墨迹（盖章）+ 轨迹记录 --------------------------------

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

;; 记录轨迹点（线性插值补点，避免快速移动时轨迹断裂——与训练侧"按段铺点"等价）
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
    (push cpts `(,gx ,gy))
    (set! pen_x gx)
    (set! pen_y gy)
  }))

(define clear_all_ink
  (lambda () {
    (define i 0)
    (define n (* INK INK))
    (while (< i n) {
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

;; ---- 特征 1：细线墨迹图（按段双线性铺点，与训练 render_thin 一致） ------

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
(define rt_j0 0)
(define rt_i0 0)
(define rt_fx 0.0)
(define rt_fy 0.0)
(define rt_wgt 0.0)
(define rt_idx 0)

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
;; 在 (x,y)（已归一化到 0..63 的浮点）上按双线性权重累加到 dens（越界丢弃，同 numpy）
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

;; 点表的最小/最大值（用闭包累加，避免返回多值）
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
    (define unused 0)
    ;; 清零
    (set! bd_t 0)
    (while (< bd_t (* NG NG)) {
      (set_item! dv bd_t 0.0)
      (set! bd_t (+ bd_t 1))
    })
    ;; 包围盒
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

;; ---- 特征 2：伪二维归一化（线密度投影等化映射，与训练 normalize_p2d 一致） --
;; prow 布局：[0..NG-1] 投影 p；[NG..2NG-1] 平滑暂存；[2NG..3NG] 累积分布 c（NG+1 项）


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

;; 由 proj[0..NG-1] 的投影构造等化映射表（写 ftab 的 off 起 NG 项）
;; proj 需 4*NG 空间：[0..NG-1] 投影、[NG..2NG-1] 平滑暂存、[2NG..3NG] 累积分布 c
(define build_warp
  (lambda (proj off) {
        ;; 平滑 SMOOTH 次
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
        ;; 总密度
    (set! wm_tot 0.0)
    (set! wm_i 0)
    (while (< wm_i NG) {
      (set! wm_tot (+ wm_tot (get_item proj wm_i)))
      (set! wm_i (+ wm_i 1))
    })
        (if (< wm_tot 0.0000001)
        {
          ;; 空投影 → 恒等映射
          (set! wm_i 0)
          (while (< wm_i NG) {
            (set_item! ftab (+ off wm_i) (/ (+ 0.0 wm_i) 63.0))
            (set! wm_i (+ wm_i 1))
          })
        }
        {
          ;; 累积分布 c[0..NG]
          (set_item! proj (* 2 NG) 0.0)
          (set! wm_i 0)
          (while (< wm_i NG) {
            (set_item! proj (+ (* 2 NG) (+ wm_i 1))
                       (+ (get_item proj (+ (* 2 NG) wm_i)) (/ (get_item proj wm_i) wm_tot)))
            (set! wm_i (+ wm_i 1))
          })
          ;; eq(i) = interp(i/63, c, k/64)：t 单调递增 → 游标 k 单调前进，O(NG)
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
                    ;; 末项归一到 1
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
    ;; x 投影（按列求和）→ prow2[0..NG-1]；y 投影（按行求和）→ prow2[NG..2NG-1]
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
        ;; 先把 y 投影拷到 prow3（build_warp 会用 prow2 的 NG..3NG 段做暂存）
    (set! p2d_i 0)
    (while (< p2d_i NG) {
      (set_item! prow3 p2d_i (get_item prow2 (+ NG p2d_i)))
      (set! p2d_i (+ p2d_i 1))
    })
    ;; x 映射（prow2 前段）→ ftab[0..NG-1]
    (build_warp prow2 0)
            (build_warp prow3 NG)
        ;; 重采样
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

;; ---- 特征 3：8 方向梯度 → 8×8 分块 → 512 维 ----------------------------

(define gv
  (lambda (v i j)
    (if (< i 0) 0.0
        (if (< j 0) 0.0
            (if (>= i NG) 0.0
                (if (>= j NG) 0.0
                    (get_item v (+ (* i NG) j))
                    0.0)
                0.0)
            0.0))))

(define gd_z 0)
(define gd_i 1)
(define gd_j 1)
(define gd_gx 0.0)
(define gd_gy 0.0)
(define gd_mag 0.0)
(define gd_ang 0.0)
(define gd_bf 0.0)
(define gd_b0 0)
(define gd_b1 0)
(define gd_w1 0.0)
(define gd_w0 0.0)
(define gd_cell 0)
(define gd_base 0)
(define featurize
  (lambda (src fev) {
    ;; 清零 512 维
    (set! gd_z 0)
    (while (< gd_z (* (* NDIR GRID) GRID)) {
      (set_item! fproj gd_z 0.0)
      (set! gd_z (+ gd_z 1))
    })
    (set! gd_i 0)
    (while (< gd_i NG) {
      (set! gd_j 0)
      (while (< gd_j NG) {
        (set! gd_gy (- (gv src (+ gd_i 1) gd_j) (gv src (- gd_i 1) gd_j)))
        (set! gd_gx (- (gv src gd_i (+ gd_j 1)) (gv src gd_i (- gd_j 1))))
        (set! gd_mag (Math.sqrt (+ (* gd_gx gd_gx) (* gd_gy gd_gy))))
        (if (> gd_mag 0.0000001)
            {
              (set! gd_ang (Math.atan2 gd_gy gd_gx))
              ;; mod 2π 归一到 [0,2π)，再映射到 [0,8) 方向编号（与 numpy 训练侧一致）
              (if (< gd_ang 0.0) (set! gd_ang (+ gd_ang (* 2.0 (Math.PI)))) #f)
              (set! gd_bf (* (/ gd_ang (* 2.0 (Math.PI))) 8.0))
              (set! gd_b0 (Math.floor gd_bf))
              (if (< gd_b0 0) (set! gd_b0 0) #f)
              (if (> gd_b0 7) (set! gd_b0 7) #f)
              (set! gd_b1 (mod (+ gd_b0 1) 8))
              (set! gd_w1 (- gd_bf (Math.floor gd_bf)))
              (set! gd_w0 (- 1.0 gd_w1))
              (set! gd_cell (+ (* (Math.floor (/ gd_i GRID)) GRID) (Math.floor (/ gd_j GRID))))
              (set_item! fproj (+ (* (* gd_b0 GRID) GRID) gd_cell)
                         (+ (get_item fproj (+ (* (* gd_b0 GRID) GRID) gd_cell)) (* gd_mag gd_w0)))
              (set_item! fproj (+ (* (* gd_b1 GRID) GRID) gd_cell)
                         (+ (get_item fproj (+ (* (* gd_b1 GRID) GRID) gd_cell)) (* gd_mag gd_w1)))
            }
            #f)
        (set! gd_j (+ gd_j 1))
      })
      (set! gd_i (+ gd_i 1))
    })
    ;; 每格 8 方向向量 L2 归一化 → 写入 feat
    (set! gd_cell 0)
    (while (< gd_cell (* GRID GRID)) {
      (set! gd_base (* gd_cell 8))
      (set! gd_mag 0.0)
      (set! gd_i 0)
      (while (< gd_i NDIR) {
        (set! gd_w0 (get_item fproj (+ (* (* gd_i GRID) GRID) gd_cell)))
        (set_item! fev (+ gd_base gd_i) gd_w0)
        (set! gd_mag (+ gd_mag (* gd_w0 gd_w0)))
        (set! gd_i (+ gd_i 1))
      })
      (set! gd_mag (Math.sqrt gd_mag))
      (if (< gd_mag 0.001) (set! gd_mag 0.001) #f)
      (set! gd_i 0)
      (while (< gd_i NDIR) {
        (set_item! fev (+ gd_base gd_i) (/ (get_item fev (+ gd_base gd_i)) gd_mag))
        (set! gd_i (+ gd_i 1))
      })
      (set! gd_cell (+ gd_cell 1))
    })
    ;; 整体 L2 归一化
    (set! gd_mag 0.0)
    (set! gd_i 0)
    (while (< gd_i NFEAT) {
      (set! gd_w0 (get_item fev gd_i))
      (set! gd_mag (+ gd_mag (* gd_w0 gd_w0)))
      (set! gd_i (+ gd_i 1))
    })
    (set! gd_mag (Math.sqrt gd_mag))
    (if (< gd_mag 0.000001) (set! gd_mag 1.0) #f)
    (set! gd_i 0)
    (while (< gd_i NFEAT) {
      (set_item! fev gd_i (/ (get_item fev gd_i) gd_mag))
      (set! gd_i (+ gd_i 1))
    })
  }))

;; ---- 分类：投影到 128 维 + 最近类均值 ------------------------------------



;; feat(512) → red(128)：red[j] = sp * Σ_d (q-384) * f[d]
;; 其中 -384 的偏移（矩阵以码点 q+256 存、量化中心 128）用"整体 Σf"一次补掉
(define project
  (lambda (fev rd) {
    (define d 0)
    (define j 0)
    (define f 0.0)
    (define off 0)
    (define p 0)
    (define pend M.per_rows)
    (define ps "")
    (define sumf 0.0)
    (define acc 0.0)
    (while (< d NFEAT) {
      (set! sumf (+ sumf (get_item fev d)))
      (set! d (+ d 1))
    })
    ;; 初值 = -384·Σf（量化中心偏移），之后逐维累加
    (set! acc (* -384.0 sumf))
    (while (< j NRED) {
      (set_item! rd j acc)
      (set! j (+ j 1))
    })
    (set! ps (get_item pparts 0))
    (set! d 0)
    (while (< d NFEAT) {
      (if (>= d pend)
          {
            (set! p (+ p 1))
            (set! pend (+ pend M.per_rows))
            (set! ps (get_item pparts p))
            (set! off 0)
          }
          #f)
      (set! f (get_item fev d))
      (if (> f 0.0000001)
          {
            (set! j 0)
            (while (< j NRED) {
              (set_item! rd j (+ (get_item rd j) (* f (String.charCodeAt (+ off j) ps))))
              (set! j (+ j 1))
            })
          }
          #f)
      (set! off (+ off NRED))
      (set! d (+ d 1))
    })
    (set! j 0)
    (while (< j NRED) {
      (set_item! rd j (* (get_item rd j) M.sp))
      (set! j (+ j 1))
    })
  }))

(define ins_p 0)
(define ins_v 0.0)
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


(define classify
  (lambda (rd_red topvv topkk) {
    (define nc NCLS)
    (define per M.per_rows)
    (define k 0)
    (define j 0)
    (define p 0)
    (define pend per)
    (define ms "")
    (define wi 0)
    (define acc 0.0)
    (define off 0.0)
    (define sumr 0.0)
    (define sc 0.0)
    ;; Σred（均值串存 q+256，故偏移为 384·Σx）
    (while (< j NRED) {
      (set! sumr (+ sumr (get_item rd_red j)))
      (set! j (+ j 1))
    })
    (set! off (* 384.0 sumr))
    (set! ms (get_item mparts 0))
    (while (< k nc) {
      (if (>= k pend)
          {
            (set! p (+ p 1))
            (set! pend (+ pend per))
            (set! ms (get_item mparts p))
            (set! wi 0)
          }
          #f)
      (set! acc 0.0)
      (set! j 0)
      (while (< j NRED) {
        (set! acc (+ acc (* (get_item rd_red j) (String.charCodeAt (+ wi j) ms))))
        (set! j (+ j 1))
      })
      (set! wi (+ wi NRED))
      ;; 2·s_k·(Σq·x − 128Σx) − |m_k|²
      (set! sc (- (* 2.0 (* (get_item scal k) (- acc off))) (get_item nrm2 k)))
      (top_insert topvv topkk NTOP k sc)
      (set! k (+ k 1))
    })
  }))

;; ---- 识别（抬笔静置后一次） ---------------------------------------------

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
        ;; 归一到最大值 1
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
    (featurize nrm feat)
        (set! feat_ms (- (Diag.ms) t0))
    (project feat red)
        (classify red topv topk)
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
    (Gfx.draw_text "手写汉字识别·在线" 4 3 40 44 62)
    (draw_btn HDR_BTN_X 2 HDR_BTN_W 14 "清屏")
    (if (> infer_ms 0)
        (Gfx.draw_text (String.concat (String.concat "识别 " (i2s infer_ms)) "ms")
                       146 4 150 155 172)
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

;; 底部：归一化墨迹（32×32，每格 1px）+ 8 张方向响应图（8×8，每格 6px）
(define dv_i 0)
(define dv_j 0)
(define dv_d 0)
(define dv_v 0.0)
(define dv_mx 0.0)
(define dv_l 0)
(define draw_viz
  (lambda () {
    (define i 0)
    (define j 0)
    (Gfx.draw_text "归一化" 4 VIZ_LY 120 126 142)
    (Gfx.draw_text "8 方向响应（8×8）" 44 VIZ_LY 120 126 142)
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
    (set! dv_d 0)
    (while (< dv_d 8) {
      (set! dv_mx 0.0001)
      (set! dv_i 0)
      (while (< dv_i 64) {
        (if (> (get_item fproj (+ (* dv_d 64) dv_i)) dv_mx)
            (set! dv_mx (get_item fproj (+ (* dv_d 64) dv_i)))
            #f)
        (set! dv_i (+ dv_i 1))
      })
      (set! dv_i 0)
      (while (< dv_i GRID) {
        (set! dv_j 0)
        (while (< dv_j GRID) {
          (set! dv_v (minf 1.0 (maxf 0.0 (/ (get_item fproj (+ (* dv_d 64) (+ (* dv_i GRID) dv_j))) dv_mx))))
          (set! dv_l (Math.round (- 252.0 (* 3.0 dv_v))))
          (if (< dv_v 0.5)
              (Gfx.fill_rect (+ (+ 44 (* dv_d 34)) (* dv_j 4)) (+ VIZ_Y (* dv_i 4))
                             4 4 dv_l
                             (Math.round (- 250.0 (* 125.0 dv_v)))
                             (Math.round (- 247.0 (* 382.0 dv_v))))
              (Gfx.fill_rect (+ (+ 44 (* dv_d 34)) (* dv_j 4)) (+ VIZ_Y (* dv_i 4))
                             4 4
                             (Math.round (- 255.0 (* 90.0 (- (* 2.0 dv_v) 1.0))))
                             (Math.round (- 190.0 (* 150.0 (- (* 2.0 dv_v) 1.0))))
                             (Math.round (- 60.0 (* 20.0 (- (* 2.0 dv_v) 1.0))))))
          (set! dv_j (+ dv_j 1))
        })
        (set! dv_i (+ dv_i 1))
      })
      (Gfx.draw_rect (- (+ 44 (* dv_d 34)) 1) (- VIZ_Y 1) 34 34 205 210 220)
      (set! dv_d (+ dv_d 1))
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
    (Gfx.draw_text "手写汉字识别·在线" 4 3 40 44 62)
    (Gfx.draw_text "模型数据缺失" 8 60 200 60 40)
    (Gfx.draw_text "请确认 app/lib/hwr2/ 数据目录已随 App 一并拷贝" 8 84 90 96 115)
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

;; ---- 命中测试 -----------------------------------------------------------

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

;; ---- 上屏 / 编辑 --------------------------------------------------------

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

;; ---- 每拍 --------------------------------------------------------------

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
                    ;; 抬笔后 PEND_MS 内再落笔 = 同一字的下一笔（不清屏）
                    (if (> up_ms 0)
                        #f
                        {
                          (clear_all_ink)
                          (set! cand_n 0)
                        })
                    (set! up_ms 0)
                    (set! pen_down 1)
                    (set! pen_x (pad_cell_x tx))
                    (set! pen_y (pad_cell_y ty))
                    (set! cpts (cdr '(0)))          ; 新笔画点表
                    (push cpts `(,pen_x ,pen_y))
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

;; ---- 启动 --------------------------------------------------------------

(define load_model
  (lambda () {
    (define i 0)
    (define p "")
    (while (< i M.npca) {
      (set! p (Fs.read (String.concat "app/lib/hwr2/p" (String.concat (i2s i) ".txt"))))
      (if (null? p) (set! load_err 1) (if (> (String.length p) 0) #f (set! load_err 1)))
      (push pparts p)
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i M.nmean) {
      (set! p (Fs.read (String.concat "app/lib/hwr2/m" (String.concat (i2s i) ".txt"))))
      (if (null? p) (set! load_err 1) (if (> (String.length p) 0) #f (set! load_err 1)))
      (push mparts p)
      (set! i (+ i 1))
    })
    (set! params (Fs.read "app/lib/hwr2/params.txt"))
    (if (null? params)
        (set! load_err 1)
        (if (== (String.length params) (* 5 NCLS))
            {
              (set! sstr (String.slice params 0 (* 2 NCLS)))
              (set! nstr (String.slice params (* 2 NCLS) (* 4 NCLS)))
              (set! cstr (String.slice params (* 4 NCLS) (* 5 NCLS)))
            }
            (set! load_err 1)))
  }))

(define decode_params
  (lambda () {
    (define i 0)
    (while (< i NCLS) {
      (set_item! scal i (* M.smax (/ (+ (* 256 (- (String.charCodeAt (* i 2) sstr) 256))
                                        (- (String.charCodeAt (+ (* i 2) 1) sstr) 256))
                                     65535.0)))
      (set_item! nrm2 i (* M.s2max (/ (+ (* 256 (- (String.charCodeAt (* i 2) nstr) 256))
                                         (- (String.charCodeAt (+ (* i 2) 1) nstr) 256))
                                      65535.0)))
      (set! i (+ i 1))
    })
  }))

(load_model)
(if (== load_err 0) (decode_params) #f)
(if (== load_err 1) (draw_load_err) (render))
(set! timer_id (System.set_interval TICK_MS tick))
(EX.install_with render)