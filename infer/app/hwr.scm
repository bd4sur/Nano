;; ============================================================================
;; hwr.scm —— 手写汉字识别（Nano「灵机应用程序」，纯 Scheme + WASM）
;;
;; 用手指在屏幕上写一个汉字，松手即识别，给出候选字；点按候选即把它接到文本行上
;; ——一个"手写输入法"式的小 App。识别算法与推理全部在 Scheme 侧（零 C 改动）。
;;
;; 识别器（训练脚本 tools/train_hwr_hanzi.py，数据文件 app/lib/hwr/）：
;;   1) 绘图板 128×128px = 64×64 墨迹网格（每格 2px，笔刷半径 2 格、软边；
;;      抬笔后静置 PEND_MS 认为一个字写完 —— 一个字的笔画之间抬笔很频繁）
;;   2) 归一化：取墨迹包围盒 → 双线性重采样到 26×26 → 居中放进 32×32（= 模型输入）
;;   3) 特征：4 方向梯度（竖直/水平/主对角/副对角）的中心差分绝对值，在 8×8 网格上
;;      分块求和 → 256 维，L2 归一化（对笔画粗细与墨迹浓淡不敏感）
;;   4) 分类：1000 类（最常用汉字，按字频降序）× 256 维的线性 softmax
;;      权重 int8 逐类量化，存在「字符即权重」的字符串里（码点 256+q），
;;      热循环直接 String.charCodeAt 取权重 —— 不需要启动时解码成列表
;;   5) 前 16 名候选（两页各 8 个），点按上屏
;;
;; 性能（关键：**热循环里一律用 lambda 形参/局部量**）：
;;   灵机的模块级变量访问走 op_load（闭包环境链查找），比帧槽位慢约 3 倍；
;;   把缓冲区作形参、下标作局部 define 后，256k 次乘加从 ~400ms 降到 ~50ms（原生宿主）。
;;   详见 doc/AGENTS.md 的实测记录。
;;
;; 实测（本机 Chromium，WASM 目标）：一次识别约 0.3 秒（含特征与 25.6 万次乘加）；
;;   原生宿主同款代码约 67ms。准确率（tools/train_hwr_hanzi.py 的评估）：
;;   未见过的印刷字体 top-1 ≈ 68% / top-8 ≈ 83%；"手指仿真"测试集 top-1 ≈ 60% / top-8 ≈ 80%。
;;
;; 解释器约束（Animac manual.md 与 doc/AGENTS.md，逐条遵守）：
;;   · 算术/比较固定二参数（三元 + 会静默算错！）；没有一元取负（写 (- 0 x)）；
;;     没有 let/list/append/map；
;;   · define 只在 lambda 体顶层（不进嵌套块）；块内只用 set!；
;;   · / 与 mod 恒返回浮点 → 下标一律 Math.floor/round 收口；除法前先判零；
;;   · '() 字面量是静态槽，push 会跨调用残留 → 新列表一律 (cdr '(0)) 起手；
;;   · Gfx 的坐标是 uint32：负数会变成天文数字，一律先钳再画。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Diag)
(native Fs)

(import EX "app/lib/exit.scm")            ; 硬 A/ESC → 全局退出确认框 + 右上角「退出」按钮
(import H "app/lib/hwr/meta.scm")         ; 模型元信息（类数/维数/量化尺度；自动生成）

;; 权重数据**不放源码**：灵机解析器按「源码字符数」预分配 token 数组（am_token_t 在
;; wasm32 上 40 字节/字符），256k 字符的字符串字面量会吃掉 10MB 解释器内存池。
;; 故权重以 UTF-8 文本文件放在 app/lib/hwr/，运行时经 (Fs.read) 读成字符串：
;; 每个字符是码点 256+q（q∈[0,255] 的 int8 量化权重），读进来即可 String.charCodeAt
;; 随机访问，无需解码成列表。
;;   w0..w4.txt : 每文件 per_part 类 × nfeat 维（类主序；类不跨文件）
;;   params.txt : ncls*2 个字符的尺度（每类 2 字符 16 位定点）+ ncls 个偏置 + ncls 个汉字
(define wparts (cdr '(0)))
(define params "")
(define sstr "")
(define bstr "")
(define cstr "")
(define load_err 0)

;; ---- 布局常量 -------------------------------------------------------------

(define SW (Gfx.width))
(define TICK_MS 20)                       ; 输入采样/重绘周期（≈50Hz）

(define PAD_X 4)                          ; 绘图板左上角（屏幕像素）
(define PAD_Y 36)
(define INK 64)                           ; 墨迹网格边长（格）
(define CELLPX 2)                         ; 每格屏幕像素（64*2 = 128）
(define PAD_PX (* INK CELLPX))

(define NOUT 26)                          ; 归一化目标边长（墨迹包围盒 → 26×26）
(define NOFF 3)                           ; 居中放进 32×32 的偏移
(define GRID 32)
(define NFEAT 256)                        ; 4 方向 × 8×8
(define NCELL 8)

(define BRUSH 2.0)                        ; 笔刷半径（墨迹格，1 格 = 2px）
(define PEN_STEP 0.5)                     ; 采样点之间的插值步长（格）

(define CX 138)                           ; 候选区左边界
(define CY 56)                            ; 第一行候选的顶边
(define CCELL 22)                         ; 候选格间距
(define CBOX 20)                          ; 候选格外框
(define NTOP 16)                          ; 候选总数（两页 × 8）
(define PAGE_N 8)

(define VIZ_LY 167)                       ; 底部说明文字行
(define VIZ_Y 183)                        ; 底部图像顶边
(define NORM_S 1)                         ; 归一化输入每格像素（32*1 = 32）
(define FMAP_X 44)                        ; 特征图左边界
(define FMAP_P 52)                        ; 特征图列间距
(define FMAP_S 6)                         ; 特征图每格像素（8*6 = 48）

(define HDR_BTN_X 84)
(define HDR_BTN_W 40)
(define BS_X 252)                         ; 「退格」
(define BS_W 32)
(define CL_X 286)                         ; 「全清」
(define CL_W 30)

;; ---- 工具 ----------------------------------------------------------------

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

;; 文本宽度（像素）：ASCII 等字符步进 7，CJK（U+2E80 起）步进 12
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

;; 概率 → "12.3%"
(define fmt_pct
  (lambda (p) {
    (define v (Math.round (* p 1000.0)))
    (String.concat (i2s (Math.floor (/ v 10.0)))
      (String.concat "." (String.concat (i2s (mod v 10.0)) "%")))
  }))

;; 类号 → 汉字
(define cls_at (lambda (k) (String.charAt cstr k)))

(define hit_rect
  (lambda (px py x y w h)
    (if (and (>= px x) (< px (+ x w)))
        (if (and (>= py y) (< py (+ y h))) 1 0)
        0)))

;; ---- 缓冲区与状态 --------------------------------------------------------

(define ink (mkn (* INK INK) 0.0))        ; 墨迹网格（0..1）
(define nimg (mkn (* GRID GRID) 0.0))     ; 归一化后的 32×32 输入
(define feat (mkn NFEAT 0.0))             ; 256 维特征（L2 归一化后）
(define scal (mkn H.ncls 1.0))            ; 每类量化尺度（启动时解码）
(define bias (mkn H.ncls 0.0))            ; 每类偏置（同上）
(define topk (mkn NTOP 0))                ; 前 NTOP 名的类号（降序）
(define topv (mkn NTOP 0.0))              ; 前 NTOP 名的分数

(define has_ink 0)
(define pen_down 0)
(define armed 0)                          ; 按下时命中的按钮（松开时结算）
(define text "")                          ; 已上屏的文本
(define cand_n 0)                         ; 候选数（≤ NTOP）
(define page 0)                           ; 候选页码（0/1）
(define infer_ms 0)
(define timer_id 0)
(define dirty_x0 0)                       ; 拖笔脏矩形（墨迹格）
(define dirty_y0 0)
(define dirty_x1 -1)
(define dirty_y1 -1)
(define pen_x 0.0)                        ; 上一采样点（墨迹格坐标）
(define pen_y 0.0)
(define up_ms 0)                          ; 抬笔时刻（宿主时钟毫秒）；0 = 无待识别的墨迹
(define PEND_MS 420)                      ; 抬笔后静置多久算"一个字写完"（多笔画字的笔画间抬笔很频繁）

;; ---- 墨迹：盖章 / 描线 ---------------------------------------------------

;; 在墨迹格坐标 (gx,gy) 盖一枚软边圆章，并把改动范围并入脏矩形。
;; inkv 作形参：热循环里访问形参是帧槽位（比模块级变量快约 3 倍）。
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
              (if (> d (get_item inkv idx))
                  (set_item! inkv idx d)
                  #f)
              (set! has_ink 1)
            }
            #f)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
  }))

;; 从上一采样点到当前点按 PEN_STEP 插值盖章（手指快速移动也不断笔）
(define stroke_to
  (lambda (inkv gx gy) {
    (define dx (- gx pen_x))
    (define dy (- gy pen_y))
    (define n (Math.ceil (/ (Math.sqrt (+ (* dx dx) (* dy dy))) PEN_STEP)))
    (define i 1)
    (if (< n 1) (set! n 1) #f)
    (while (<= i n) {
      (stamp inkv (+ pen_x (/ (* dx i) n)) (+ pen_y (/ (* dy i) n)))
      (set! i (+ i 1))
    })
    (set! pen_x gx)
    (set! pen_y gy)
  }))

(define clear_ink
  (lambda (inkv) {
    (define i 0)
    (define n (* INK INK))
    (while (< i n) {
      (set_item! inkv i 0.0)
      (set! i (+ i 1))
    })
    (set! has_ink 0)
    (set! dirty_x0 0)
    (set! dirty_y0 0)
    (set! dirty_x1 -1)
    (set! dirty_y1 -1)
  }))

;; ---- 归一化：包围盒 → 26×26 → 居中放进 32×32 ----------------------------
;; 与训练侧（tools/train_hwr_hanzi.py 的 normalize）逐位一致：
;; 阈值 0.10 求包围盒；中心对齐的双线性采样；越界取 0。

(define normalize
  (lambda (inkv nim) {
    (define t INK)
    (define b -1)
    (define l INK)
    (define r -1)
    (define i 0)
    (define j 0)
    (define v 0.0)
    (define h 0.0)
    (define w 0.0)
    (define sy 0.0)
    (define sx 0.0)
    (define yy 0.0)
    (define xx 0.0)
    (define y0 0)
    (define x0 0)
    (define y1 0)
    (define x1 0)
    (define fy 0.0)
    (define fx 0.0)
    ;; 局部取值器（越界 0）；闭包捕获 inkv（只读 → 快照槽位）
    (define gv
      (lambda (a c)
        (if (< a 0) 0.0
            (if (> a (- INK 1)) 0.0
                (if (< c 0) 0.0
                    (if (> c (- INK 1)) 0.0
                        (get_item inkv (+ (* a INK) c))
                        0.0)
                    0.0)
                0.0))))
    ;; 包围盒
    (set! i 0)
    (while (< i INK) {
      (set! j 0)
      (while (< j INK) {
        (if (> (get_item inkv (+ (* i INK) j)) 0.10)
            {
              (if (< i t) (set! t i) #f)
              (if (> i b) (set! b i) #f)
              (if (< j l) (set! l j) #f)
              (if (> j r) (set! r j) #f)
            }
            #f)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    ;; 输入清零
    (set! i 0)
    (while (< i (* GRID GRID)) {
      (set_item! nim i 0.0)
      (set! i (+ i 1))
    })
    (if (< b 0)
        #f
        {
          (set! h (+ (- b t) 1.0))
          (set! w (+ (- r l) 1.0))
          (set! sy (/ h NOUT))
          (set! sx (/ w NOUT))
          (set! i 0)
          (while (< i NOUT) {
            (set! yy (- (+ t (* (+ i 0.5) sy)) 0.5))
            (set! y0 (Math.floor yy))
            (set! fy (- yy y0))
            (set! y1 (+ y0 1))
            (set! j 0)
            (while (< j NOUT) {
              (set! xx (- (+ l (* (+ j 0.5) sx)) 0.5))
              (set! x0 (Math.floor xx))
              (set! fx (- xx x0))
              (set! x1 (+ x0 1))
              (set! v (+ (* (gv y0 x0) (* (- 1.0 fy) (- 1.0 fx)))
                         (+ (* (gv y0 x1) (* (- 1.0 fy) fx))
                            (+ (* (gv y1 x0) (* fy (- 1.0 fx)))
                               (* (gv y1 x1) (* fy fx))))))
              (set_item! nim (+ (* (+ NOFF i) GRID) (+ NOFF j)) v)
              (set! j (+ j 1))
            })
            (set! i (+ i 1))
          })
        })
  }))

;; ---- 特征：4 方向梯度在 8×8 网格上分块求和 ------------------------------
;; 与训练侧 numpy 实现逐位对应：上下/左右/两对角中心差分取绝对值，再按 4×4
;; 像素块求和。模型输入中间隙（墨迹只占 3..28），故 0..1 与 30..31 行的贡献
;; 恒为 0，循环只走 2..29 —— 无需越界判断。

(define featurize
  (lambda (nim fev) {
    (define i 0)
    (define y 2)
    (define x 2)
    (define r0 0)
    (define r1 0)
    (define r2 0)
    (define b 0)
    (define v 0.0)
    (define s 0.0)
    (set! i 0)
    (while (< i NFEAT) {
      (set_item! fev i 0.0)
      (set! i (+ i 1))
    })
    (set! y 2)
    (while (< y 30) {
      (set! r0 (* (- y 1) GRID))
      (set! r1 (* y GRID))
      (set! r2 (* (+ y 1) GRID))
      (set! x 2)
      (while (< x 30) {
        (set! b (+ (* (Math.floor (/ y 4.0)) NCELL) (Math.floor (/ x 4.0))))
        ;; 竖直：上 - 下
        (set! v (- (get_item nim (+ r0 x)) (get_item nim (+ r2 x))))
        (set! v (if (< v 0.0) (- 0.0 v) v))
        (set_item! fev b (+ (get_item fev b) v))
        ;; 水平：左 - 右
        (set! v (- (get_item nim (+ r1 (- x 1))) (get_item nim (+ r1 (+ x 1)))))
        (set! v (if (< v 0.0) (- 0.0 v) v))
        (set_item! fev (+ 64 b) (+ (get_item fev (+ 64 b)) v))
        ;; 主对角
        (set! v (- (get_item nim (+ r0 (- x 1))) (get_item nim (+ r2 (+ x 1)))))
        (set! v (if (< v 0.0) (- 0.0 v) v))
        (set_item! fev (+ 128 b) (+ (get_item fev (+ 128 b)) v))
        ;; 副对角
        (set! v (- (get_item nim (+ r0 (+ x 1))) (get_item nim (+ r2 (- x 1)))))
        (set! v (if (< v 0.0) (- 0.0 v) v))
        (set_item! fev (+ 192 b) (+ (get_item fev (+ 192 b)) v))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
    ;; L2 归一化
    (set! s 0.0)
    (set! i 0)
    (while (< i NFEAT) {
      (set! v (get_item fev i))
      (set! s (+ s (* v v)))
      (set! i (+ i 1))
    })
    (set! s (Math.sqrt s))
    (if (< s 0.000001) (set! s 1.0) #f)
    (set! i 0)
    (while (< i NFEAT) {
      (set_item! fev i (/ (get_item fev i) s))
      (set! i (+ i 1))
    })
  }))

;; ---- 分类：1000 类线性 softmax（int8 权重） -----------------------------
;; 权重 (q-128)*scal 与特征 f 的点积：
;;   Σ (q-128)*f*scal = scal*(Σ q*f - 128*Σf)
;; 权重串里存的是码点 q+256（省掉每步减去 256），故
;;   Σ q*f = Σ charCode*f - 256*Σf  →  score = scal*(Σ charCode*f - 384*Σf) + bias
;; 于是热循环只有「非负字节 × 特征」的乘加，偏移项每类只算一次。

;; 把 (类号, 分数) 插入降序前 ntop 表（topvv/topkk 作形参）
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
  (lambda (fev wps scalv biasv topvv topkk) {
    (define nc H.ncls)
    (define per H.per_part)
    (define k 0)
    (define d 0)
    (define p 0)
    (define pend per)
    (define ws "")
    (define wi 0)
    (define acc 0.0)
    (define off 0.0)
    (define sumf 0.0)
    (define sc 0.0)
    ;; 清零候选表
    (set! k 0)
    (while (< k NTOP) {
      (set_item! topvv k -1000000000.0)
      (set_item! topkk k 0)
      (set! k (+ k 1))
    })
    ;; Σf（用于减去量化偏移 384*Σf）
    (set! d 0)
    (while (< d NFEAT) {
      (set! sumf (+ sumf (get_item fev d)))
      (set! d (+ d 1))
    })
    (set! off (* 384.0 sumf))
    (set! ws (get_item wps 0))
    (set! k 0)
    (while (< k nc) {
      (if (>= k pend)
          {
            (set! p (+ p 1))
            (set! pend (+ pend per))
            (set! ws (get_item wps p))
            (set! wi 0)
          }
          #f)
      (set! acc 0.0)
      (set! d 0)
      (while (< d NFEAT) {
        (set! acc (+ acc (* (get_item fev d) (String.charCodeAt (+ wi d) ws))))
        (set! d (+ d 1))
      })
      (set! wi (+ wi NFEAT))
      (set! sc (+ (* (- acc off) (get_item scalv k)) (get_item biasv k)))
      (top_insert topvv topkk NTOP k sc)
      (set! k (+ k 1))
    })
  }))

;; ---- 识别（松手后一次） --------------------------------------------------

(define recognize
  (lambda () {
    (define t0 (Diag.ms))
    (define mx 0.0)
    (define s 0.0)
    (define i 0)
    (define v 0.0)
    (normalize ink nimg)
    (featurize nimg feat)
    (classify feat wparts scal bias topv topk)
    (set! page 0)
    (set! cand_n NTOP)
    ;; 候选集合内的 softmax（显示相对置信度用；先减最大值）
    (set! mx (get_item topv 0))
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

;; 墨迹强度 → 灰度格
(define draw_cell
  (lambda (inkv i j) {
    (define v (get_item inkv (+ (* i INK) j)))
    (define g (Math.round (- 255.0 (* 255.0 (minf 1.0 (maxf 0.0 v))))))
    (Gfx.fill_rect (+ PAD_X (* j CELLPX)) (+ PAD_Y (* i CELLPX))
                   CELLPX CELLPX g g g)
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

;; 只重画脏矩形（拖笔时的高频路径）
(define draw_pad_dirty
  (lambda (inkv) {
    (define i 0)
    (define j 0)
    (if (>= dirty_x1 dirty_x0)
        {
          (set! i dirty_y0)
          (while (<= i dirty_y1) {
            (set! j dirty_x0)
            (while (<= j dirty_x1) {
              (draw_cell inkv i j)
              (set! j (+ j 1))
            })
            (set! i (+ i 1))
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
    (Gfx.draw_text "手写汉字识别" 4 3 40 44 62)
    (draw_btn HDR_BTN_X 2 HDR_BTN_W 14 "清屏")
    (if (> infer_ms 0)
        (Gfx.draw_text (String.concat (String.concat "识别 " (i2s infer_ms)) "ms")
                       130 4 150 155 172)
        (Gfx.draw_text "写一个字，松手即识别" 130 4 150 155 172))
  }))

(define draw_candidates
  (lambda () {
    (define i 0)
    (define x 0)
    (define y 0)
    (define ci 0)
    (define hot 0)
    (Gfx.draw_text (String.concat "候选（点按上屏） " (i2s (+ page 1))) CX 40 55 62 88)
    (Gfx.draw_text "/2" 296 40 150 155 172)
    (while (< i PAGE_N) {
      (set! x (+ CX (* i CCELL)))
      (set! y (+ CY (* 24 (Math.floor (/ i 8.0)))))
      (set! ci (+ (* page PAGE_N) i))
      (if (< ci cand_n)
          {
            (set! hot (if (and (== page 0) (== i 0)) 1 0))
            (if (== hot 1)
                (Gfx.fill_rect x y CBOX 22 214 231 250)
                (Gfx.fill_rect x y CBOX 22 242 244 248))
            (Gfx.draw_rect x y CBOX 22 198 204 216)
            (Gfx.draw_text (cls_at (get_item topk ci)) (+ x 4) (+ y 4)
                           (if (== hot 1) 20 60 160)
                           (if (== hot 1) 70 70 100)
                           (if (== hot 1) 190 90 100))
          }
          #f)
      (set! i (+ i 1))
    })
    ;; 置信度条（前 3 名）
    (Gfx.draw_text "置信度" CX 108 120 126 142)
    (set! i 0)
    (while (< i 3) {
      (set! y (+ 124 (* i 15)))
      (Gfx.fill_rect CX y 130 10 235 238 243)
      (Gfx.fill_rect CX y (Math.round (* 130.0 (get_item topv i))) 10 90 140 210)
      (Gfx.draw_text (fmt_pct (get_item topv i)) (+ CX 134) y 100 106 124)
      (set! i (+ i 1))
    })
  }))

;; 底部：归一化输入 32×32（1px/格）+ 4 张方向梯度特征图（6px/格）
(define draw_viz
  (lambda () {
    (define i 0)
    (define j 0)
    (define d 0)
    (define v 0.0)
    (define mx 0.0)
    (define l 0)
    (Gfx.draw_text "归一化" 4 VIZ_LY 120 126 142)
    (Gfx.draw_text "4 方向梯度特征（8×8）" 44 VIZ_LY 120 126 142)
    (while (< i GRID) {
      (set! j 0)
      (while (< j GRID) {
        (set! v (minf 1.0 (maxf 0.0 (get_item nimg (+ (* i GRID) j)))))
        (set! l (Math.round (- 255.0 (* 255.0 v))))
        (Gfx.fill_rect (+ 4 (* j NORM_S)) (+ VIZ_Y (* i NORM_S)) NORM_S NORM_S l l l)
        (set! j (+ j 1))
      })
      (set! i (+ i 1))
    })
    (Gfx.draw_rect 3 (- VIZ_Y 1) (+ (* GRID NORM_S) 2) (+ (* GRID NORM_S) 2) 205 210 220)
    (set! d 0)
    (while (< d 4) {
      (set! mx 0.0001)
      (set! i 0)
      (while (< i 64) {
        (set! v (get_item feat (+ (* d 64) i)))
        (if (> v mx) (set! mx v) #f)
        (set! i (+ i 1))
      })
      (set! i 0)
      (while (< i NCELL) {
        (set! j 0)
        (while (< j NCELL) {
          (set! v (minf 1.0 (maxf 0.0 (/ (get_item feat (+ (* d 64) (+ (* i NCELL) j))) mx))))
          (if (< v 0.5)
              (Gfx.fill_rect (+ (+ FMAP_X (* d FMAP_P)) (* j FMAP_S)) (+ VIZ_Y (* i FMAP_S))
                             FMAP_S FMAP_S
                             (Math.round (- 252.0 (* 3.0 v)))
                             (Math.round (- 250.0 (* 125.0 v)))
                             (Math.round (- 247.0 (* 382.0 v))))
              (Gfx.fill_rect (+ (+ FMAP_X (* d FMAP_P)) (* j FMAP_S)) (+ VIZ_Y (* i FMAP_S))
                             FMAP_S FMAP_S
                             (Math.round (- 255.0 (* 90.0 (- (* 2.0 v) 1.0))))
                             (Math.round (- 190.0 (* 150.0 (- (* 2.0 v) 1.0))))
                             (Math.round (- 60.0 (* 20.0 (- (* 2.0 v) 1.0))))))
          (set! j (+ j 1))
        })
        (set! i (+ i 1))
      })
      (Gfx.draw_rect (- (+ FMAP_X (* d FMAP_P)) 1) (- VIZ_Y 1)
                     (+ (* NCELL FMAP_S) 2) (+ (* NCELL FMAP_S) 2) 205 210 220)
      (set! d (+ d 1))
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
    (Gfx.draw_text "手写汉字识别" 4 3 40 44 62)
    (Gfx.draw_text "模型数据缺失" 8 60 200 60 40)
    (Gfx.draw_text "请确认 app/lib/hwr/ 数据目录已随 App 一并拷贝" 8 84 90 96 115)
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

;; 1=清屏 2=退格 3=全清 4=候选翻页 10+i=第 i 个候选 0=都不是
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

;; 屏幕像素 → 墨迹格坐标（浮点）
(define pad_cell_x (lambda (px) (/ (- px PAD_X) CELLPX)))
(define pad_cell_y (lambda (py) (/ (- py PAD_Y) CELLPX)))

;; ---- 上屏 / 编辑 --------------------------------------------------------

(define commit
  (lambda (i) {
    (set! text (String.concat text (cls_at (get_item topk i))))
    (clear_ink ink)
    (set! cand_n 0)
    (render)
  }))

(define backspace
  (lambda () {
    (define n (String.length text))
    (if (> n 0) (set! text (String.slice text 0 (- n 1))) #f)
  }))

;; ---- 每拍 --------------------------------------------------------------

;; 手指写入：按住则在上一采样点与当前点之间补线
(define pen_move
  (lambda (tx ty) {
    (define gx (pad_cell_x tx))
    (define gy (pad_cell_y ty))
    (if (< gx 0.0) (set! gx 0.0) #f)
    (if (< gy 0.0) (set! gy 0.0) #f)
    (if (> gx (+ 0.0 INK)) (set! gx (+ 0.0 INK)) #f)
    (if (> gy (+ 0.0 INK)) (set! gy (+ 0.0 INK)) #f)
    (if (== (pad_hit tx ty) 1)
        {
          (stroke_to ink gx gy)
          (draw_pad_dirty ink)
        }
        {
          ;; 滑出板外：不落笔，但仍同步"上一采样点"，回板内不会连出跨屏直线
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
                (clear_ink ink)
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
                        (clear_ink ink)
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
    ;; 按下沿：命中按钮则武装；命中绘图板则起笔（先清屏 = 写下一个字）
    (if (== tp 1)
        {
          (set! armed (hit_at tx ty))
          (if (== armed 0)
              (if (== (pad_hit tx ty) 1)
                  {
                    ;; 抬笔后 PEND_MS 内再落笔 = 同一个字的下一笔（不清屏）；
                    ;; 否则是下一个字（清屏）
                    (if (> up_ms 0) #f (clear_ink ink))
                    (set! cand_n 0)
                    (set! up_ms 0)
                    (set! pen_down 1)
                    (set! pen_x (pad_cell_x tx))
                    (set! pen_y (pad_cell_y ty))
                    (stamp ink pen_x pen_y)
                    (render)
                  }
                  #f)
              #f)
        }
        #f)
    ;; 按住：继续送笔
    (if (== pen_down 1)
        (if (Touch.is_pressed) (pen_move tx ty) #f)
        #f)
    ;; 松开沿：结算按钮；写字则记下抬笔时刻（一个字的笔画之间会抬笔，故不能立刻识别）
    (if (== tr 1)
        {
          (if (> armed 0) (do_release_armed armed tx ty) #f)
          (if (== pen_down 1)
              {
                (set! pen_down 0)
                (if (== has_ink 1) (set! up_ms (+ (System.timestamp) 1)) #f)
              }
              #f)
          (set! armed 0)
        }
        #f)
    ;; 抬笔静置 PEND_MS → 认为一个字写完，跑一次识别
    (if (> up_ms 0)
        (if (>= (System.timestamp) (+ up_ms PEND_MS))
            {
              (set! up_ms 0)
              (Gfx.draw_text "识别中…" 130 4 200 120 40)
              (EX.draw_button)
              (Gfx.refresh)
              (recognize)
              (render)
            }
            #f)
        #f)
  }))

;; ---- 启动 --------------------------------------------------------------

;; 读模型数据文件（缺任一份就显示错误画面，不启动识别）
(define load_model
  (lambda () {
    (define i 0)
    (define p "")
    (while (< i H.npart) {
      (set! p (Fs.read (String.concat "app/lib/hwr/w" (String.concat (i2s i) ".txt"))))
      (if (null? p)
          (set! load_err 1)
          (if (> (String.length p) 0) #f (set! load_err 1)))
      (push wparts p)
      (set! i (+ i 1))
    })
    (set! params (Fs.read "app/lib/hwr/params.txt"))
    (if (null? params)
        (set! load_err 1)
        (if (== (String.length params) (* 4 H.ncls))
            {
              (set! sstr (String.slice params 0 (* 2 H.ncls)))
              (set! bstr (String.slice params (* 2 H.ncls) (* 3 H.ncls)))
              (set! cstr (String.slice params (* 3 H.ncls) (* 4 H.ncls)))
            }
            (set! load_err 1)))
  }))

;; 解码每类的量化尺度与偏置
(define decode_params
  (lambda () {
    (define i 0)
    (define hi 0)
    (define lo 0)
    (while (< i H.ncls) {
      (set! hi (- (String.charCodeAt (* i 2) sstr) 256))
      (set! lo (- (String.charCodeAt (+ (* i 2) 1) sstr) 256))
      (set_item! scal i (* H.smax (/ (+ (* hi 256) lo) 65535.0)))
      (set_item! bias i (* (- (String.charCodeAt i bstr) 384) H.sb))
      (set! i (+ i 1))
    })
  }))

(load_model)
(if (== load_err 0) (decode_params) #f)
(if (== load_err 1) (draw_load_err) (render))
(set! timer_id (System.set_interval TICK_MS tick))

;; 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）
(EX.install_with render)