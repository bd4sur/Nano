;; ============================================================================
;; svm.scm —— 交互式支持向量机（SVM）演示（Nano「灵机应用程序」）
;;
;; 参考实现：https://bd4sur.github.io/html/svm.html
;;   （Karpathy svmjs：简化 SMO 算法 + canvas 分类区域/边界/支持向量绘制）
;;
;; 交互（全部触屏）：
;;   · 点按画面空白处 —— 按当前类别添加一个样本点（正样本橙色 / 负样本蓝色）；
;;   · 左下角三个按钮 ——「正样本/负样本」切换即将添加的类别、「线性核/RBF核」
;;     切换核函数、「重置」清除一切状态回到初始态；每个按钮**只显示当前状态**；
;;   · 样本数 ≥ 2 时立即训练（SMO），随后按参考页面的方式绘制：
;;       分类区域（正侧淡黄 / 负侧淡蓝）、分类边界（红）、
;;       间隔线 w·x+b=±1（灰）、支持向量（圈注；线性核另有到间隔线的垂线）；
;;   · 左上角显示分类边界的参数（线性核 wx/wy/b；RBF 核 b/支持向量数/样本数）。
;;
;; 训练特征空间（关键设计）：
;;   线性核在**归一化特征**上训练：x' = (x - 屏幕中心)/160，使绘图区约等于
;;   [-1,1]x[-0.75,0.75]。原因是线性核的可分性与条件数强烈依赖特征尺度——像素
;;   坐标（~10²）配 C=1 时，"有 1 个线性不可分点"就会把问题推进病态区：
;;   目标函数里唯一带几何信息的项 ½‖w‖² 只占 2/(C·margin²) ≈ 1e-5，训练要 1e4 量级
;;   轮数才收敛（实测 12800~25600 轮），而本 App 的预算只有 2e2 轮，于是边界完全由
;;   随机 j 决定的"半路点"给出 —— 同一组点时而给出完美边界、时而完全找不到。
;;   归一化后 ρ = ½‖w‖²/Σα 回到 O(0.1~0.5)，50~100 轮即收敛且结果可复现（详见
;;   doc/AGENTS.md「线性核弱不可分情形的病态与归一化」）。
;;   RBF 核的核值只依赖归一化距离与 σ 的比值（σ' = σ/160），与本变换等价，故 RBF
;;   路径保持像素尺度计算（数值完全不变，也省掉每格一次换算）。
;;
;; 布局约定（平台相关）：
;;   · 触屏右上角 80x60 曾是宿主 4x4 宫格的 ESC 格；2026-10 起该格已不再映射为
;;     返回键（App 运行态整体退出宫格映射），此处仍不放常驻控件只是沿用旧布局；
;;   · 左下角按钮区（y 212..233）内的点按只作按钮用，不添加样本。
;;
;; 解释器约束（详见 Animac manual.md 与 doc/AGENTS.md，逐条遵守）：
;;   · 算术/比较固定二参数，多元必须嵌套（多传的参数会污染操作数栈）；
;;   · 没有一元取负（负数写字面量，或 (- 0.0 x)）；
;;   · 没有 let/list/append/map；define 无过程简写；块内不用 define，只用模块级 set!；
;;   · / 与 mod 恒返回浮点 → 下标一律 Math.round/floor 收口；除零会终止进程，先判后除；
;;   · 跨函数调用链上的模块级临时量绝不共用（按"谁写谁读"分配前缀名）；
;;   · Gfx.draw_line 的坐标是 uint32：负值会被当成天文数字，故线段一律先裁剪再画。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 画布 -----------------------------------------------------------------

(define SW (Gfx.width))
(define SH (Gfx.height))

;; ---- 训练特征空间（归一化） ----------------------------------------------
;; 中心取屏幕中心，尺度取屏幕半宽：绘图区 ≈ [-1,1] × [-0.75,0.75]
(define TCX (* 0.5 SW))      ; 特征中心 x（像素）
(define TCY (* 0.5 SH))      ; 特征中心 y（像素）
(define TSC (* 0.5 SW))      ; 特征尺度（像素）
(define TISC (/ 1.0 TSC))    ; 1/尺度：像素 → 归一化

;; ---- 超参数（对齐参考页面） ----------------------------------------------

(define MAXN 48)          ; 样本容量上限（内核矩阵 MAXN×MAXN）
(define NUMPASSES 10)     ; 连续多少轮 alpha 无变化即停止（同参考实现的默认值）
(define MAXITER 300)      ; 迭代轮数硬上限
;; 单次训练的工作量预算：轮数 × n² ≤ TRAIN_WORK（≈ 10 帧、约 170ms 的 VM 指令量）。
;; 参考页面在 JS 里 numpasses=200、maxiter=10000 都跑得起；解释器一帧只给 31 万条
;; 指令，故必须限总量：点越多，允许的轮数越少。
;; 归一化后实测收敛轮数：n=12~15 → 15~72 轮；n=30 → 34~44 轮（都在预算内）；
;; n=48 时预算只够 17 轮（未完全收敛，但 |w| 只差 ~10%、角度差 2~3°，视觉一致）。
(define TRAIN_WORK 40000)
(define TOL 0.001)        ; KKT 违背容忍度（判别值的量级，与 α 无关）
(define EPSA 0.000001)   ; α 变化量的相对门槛（× C）：小于它视为"这一步没意义"
                          ; 参考页面写死 1e-4，但那个绝对阈值隐含了"α 的量级"——
                          ; 归一化之前线性核的 α 最优解只有 1e-4 量级，绝对 1e-4 会把
                          ; **所有**更新都当成噪声跳过（实测：训练后 α 全 0、分类区域
                          ; 一片蓝、没有边界也没有支持向量）。归一化之后 α 已是 O(1)
                          ; 量级，仍保留相对形式以确保与数据尺度无关。
(define ALPHA_RATIO 0.01) ; alpha > 该比例 × max(alpha) 视为支持向量
                          ; （参考页面写死 1e-2 的绝对阈值；改为与尺度无关的相对
                          ;  阈值，任何数据尺度下都能正确挑出支持向量）
(define C_LIN 1.0)        ; 线性核的惩罚系数 C —— **作用在归一化特征上**，
                          ; 等价于像素单位下的 C_px = C_LIN/尺度² ≈ 3.9e-5
                          ; （参考页面的像素单位 C=1 相当于此处 C ≈ 25600，
                          ;   属"几何几乎不起作用"的病态配置，见头注释）
(define C_RBF 20.0)       ; RBF 核的惩罚系数 C（核值与归一化前逐位相同，故不变）
(define SIGMA 20.0)       ; RBF 核宽度 σ
(define SIGMA2 800.0)     ; 2σ²（= 800，σ=20）
(define DENS_RBF 4)       ; RBF 核分类区域的采样步长（单点成本高，取 4px；
                          ; 320、240 均为其整数倍，无半格溢出）。支持向量多时
                          ; 自动放粗（见 raster_rbf 开头），把重绘耗时钳在 10 帧量级。
(define TICK_MS 20)       ; 输入采样周期（≈50Hz；计算按需触发，空闲拍成本极低）

;; ---- 配色（对齐参考页面） -------------------------------------------------

;; 正样本橙 #ff9933 / 负样本蓝 #0086bb / 正侧区域淡黄 #fff7c7
;; 负侧区域 = 白底上叠 rgba(0,134,187,0.3) ⇒ (179,219,234)
;; 分类边界红 / 间隔线与支持向量辅助线深灰

;; ---- 模块级状态 -----------------------------------------------------------

(define n 0)              ; 样本数
(define cat 1)            ; 当前类别：1 正样本，-1 负样本
(define kv 0)             ; 当前核：0 线性，1 RBF
(define b 0.0)            ; 偏置（判别函数直接使用；线性核为像素单位，RBF 核与 bn 同）
(define bn 0.0)           ; 训练空间里的偏置（线性核为归一化特征空间；RBF 核等于 b）
(define w0 0.0)           ; 线性权重（w0=x 分量，w1=y 分量）
(define w1 0.0)
(define use_w 0)          ; 1 = 线性核训练完成，用 (w0,w1,b) 判定（与核展开等价）
(define nsv 0)            ; 支持向量个数
(define dirty 0)          ; 1 = 需要重建内核矩阵 + 训练 + 重绘
(define pending 0)        ; 已按下的按钮编号（0 无），松手沿在按钮内才结算
(define press_x 0)
(define press_y 0)
(define tp 0)             ; 本拍采样：按下沿
(define tr 0)             ; 本拍采样：松开沿
(define _timer 0)

;; ---- 样本存储（预分配 + 原地写入，零分配热路径） ---------------------------
;; 解释器无数组；列表支持 O(1) 的 get_item/set_item!，故用"预分配列表 + 计数 n"。
;; 空表用 (cdr '(0)) 现场拷贝出**新**对象——顶层 '() 字面量是模块级静态槽，
;; 对它 push 会原地替换槽指向的对象（跨调用残留膨胀，见 doc/AGENTS.md）。

(define mk_acc 0)
(define mk_i 0)
(define make_zeros
  (lambda (k) {
    (set! mk_acc (cdr '(0)))
    (set! mk_i 0)
    (while (< mk_i k) {
      (push mk_acc 0)
      (set! mk_i (+ mk_i 1))
    })
    mk_acc
  }))

(define px (make_zeros MAXN))            ; 样本 x（像素，绘制用）
(define py (make_zeros MAXN))            ; 样本 y（像素，绘制用）
(define pxn (make_zeros MAXN))           ; 样本 x（归一化，训练用）
(define pyn (make_zeros MAXN))           ; 样本 y（归一化，训练用）
(define plab (make_zeros MAXN))          ; 样本类别 +1 / -1
(define palpha (make_zeros MAXN))        ; 对偶变量 α
(define kern (make_zeros (* MAXN MAXN))) ; 内核矩阵（行主序 i*MAXN+j）
(define svlist (make_zeros MAXN))        ; 支持向量下标
(define svcf (make_zeros MAXN))          ; 支持向量的 α·y（预测时直接乘）
(define svsx (make_zeros MAXN))          ; 支持向量坐标缓存（避免 get_item 链）
(define svsy (make_zeros MAXN))

;; ---- 小工具 ---------------------------------------------------------------

(define i2s (lambda (x) (String.atom_to_string x)))
(define maxf (lambda (a c) (if (> a c) a c)))
(define minf (lambda (a c) (if (< a c) a c)))

;; 文本宽度（像素）：ASCII 等字符步进 7，CJK（U+2E80 起）步进 12
;; （字模实测：glyph_12 的 xAdvance，ASCII=7、汉字=12）
(define tw_s 0)
(define tw_i 0)
(define tw_cp 0)
(define tw
  (lambda (s) {
    (set! tw_s 0)
    (set! tw_i 0)
    (while (< tw_i (String.length s)) {
      (set! tw_cp (String.charCodeAt tw_i s))
      (set! tw_s (+ tw_s (if (< tw_cp 11904) 7 12)))
      (set! tw_i (+ tw_i 1))
    })
    tw_s
  }))

;; 定点格式化（3 位小数）：VM 浮点载荷位宽有限（wasm32 下 am_float_t 为 float 且
;; 尾数又被截 5 位），(Math.to_fixed 3) + %g 会打出 0.123 之类不齐整的串；
;; 这里按"×1000 取整 → 整数段 + 补零小数段"自己拼。
(define f_v 0)
(define f_a 0)
(define f_ip 0)
(define f_fp 0)
(define f_neg 0)
(define f_r "")
(define pad_v 0)
(define pad3
  (lambda (v) {
    (set! pad_v (Math.round v))
    (if (< pad_v 10)
        (String.concat "00" (i2s pad_v))
        (if (< pad_v 100)
            (String.concat "0" (i2s pad_v))
            (i2s pad_v)))
  }))
(define fmt3
  (lambda (x) {
    (set! f_v (Math.round (* x 1000.0)))
    (set! f_neg (if (< f_v 0) 1 0))
    (set! f_a (if (== f_neg 1) (- 0 f_v) f_v))
    (set! f_ip (Math.floor (/ f_a 1000.0)))
    (set! f_fp (- f_a (* f_ip 1000.0)))
    (set! f_r (String.concat (i2s f_ip) (String.concat "." (pad3 f_fp))))
    (if (== f_neg 1) (String.concat "-" f_r) f_r)
  }))

;; ---- 线段裁剪（Liang–Barsky） --------------------------------------------
;; Gfx.draw_line 的坐标类型是 uint32，传负值会被解释成 4e9 量级的天文坐标，
;; 因此所有可能越界的线段（分类边界/间隔线/支持向量垂线）先裁到屏幕矩形内。

(define c_ok 1)
(define c_t0 0.0)
(define c_t1 1.0)
(define c_dx 0.0)
(define c_dy 0.0)
(define c_r 0.0)

(define clip_edge
  (lambda (p q) {
    (if (== p 0.0)
        (if (< q 0.0) (set! c_ok 0))       ; 平行且在界外 → 整段丢弃
        {
          (set! c_r (/ q p))
          (if (< p 0.0)
              (if (> c_r c_t0) (set! c_t0 c_r))
              (if (< c_r c_t1) (set! c_t1 c_r)))
        })
  }))

;; 裁剪 [x1,y1]-[x2,y2]；结果留在 c_ok / c_t0 / c_t1 / c_dx / c_dy
(define clip_seg
  (lambda (x1 y1 x2 y2) {
    (set! c_ok 1)
    (set! c_t0 0.0)
    (set! c_t1 1.0)
    (set! c_dx (- x2 x1))
    (set! c_dy (- y2 y1))
    (clip_edge (- 0.0 c_dx) x1)                     ; 左
    (clip_edge c_dx (- (- SW 1) x1))                ; 右
    (clip_edge (- 0.0 c_dy) y1)                     ; 上
    (clip_edge c_dy (- (- SH 1) y1))                ; 下
    (if (== c_ok 1)
        (if (> c_t0 c_t1) (set! c_ok 0)))
  }))

(define ds_x1 0)
(define ds_y1 0)
(define ds_x2 0)
(define ds_y2 0)
(define draw_seg
  (lambda (x1 y1 x2 y2 r g bl) {
    (clip_seg x1 y1 x2 y2)
    (if (== c_ok 1) {
      (set! ds_x1 (Math.round (+ x1 (* c_t0 c_dx))))
      (set! ds_y1 (Math.round (+ y1 (* c_t0 c_dy))))
      (set! ds_x2 (Math.round (+ x1 (* c_t1 c_dx))))
      (set! ds_y2 (Math.round (+ y1 (* c_t1 c_dy))))
      ;; 浮点四舍五入可能越界 1 像素，保险再钳一次
      (if (< ds_x1 0) (set! ds_x1 0))
      (if (> ds_x1 (- SW 1)) (set! ds_x1 (- SW 1)))
      (if (< ds_y1 0) (set! ds_y1 0))
      (if (> ds_y1 (- SH 1)) (set! ds_y1 (- SH 1)))
      (if (< ds_x2 0) (set! ds_x2 0))
      (if (> ds_x2 (- SW 1)) (set! ds_x2 (- SW 1)))
      (if (< ds_y2 0) (set! ds_y2 0))
      (if (> ds_y2 (- SH 1)) (set! ds_y2 (- SH 1)))
      (Gfx.draw_line ds_x1 ds_y1 ds_x2 ds_y2 r g bl)
    })
  }))

;; ---- 内核矩阵 ------------------------------------------------------------

(define kc_i 0)
(define kc_j 0)
(define kc_dx 0.0)
(define kc_dy 0.0)
(define kc_d2 0.0)
(define kern_build
  (lambda () {
    (set! kc_i 0)
    (while (< kc_i n) {
      (set! kc_j 0)
      (while (< kc_j n) {
        (if (== kv 1)
            {
              (set! kc_dx (- (get_item px kc_i) (get_item px kc_j)))
              (set! kc_dy (- (get_item py kc_i) (get_item py kc_j)))
              (set! kc_d2 (+ (* kc_dx kc_dx) (* kc_dy kc_dy)))
              (set_item! kern (+ (* kc_i MAXN) kc_j)
                          (Math.exp (- 0.0 (/ kc_d2 SIGMA2))))
            }
            {
              ;; 线性核在**归一化特征**上求核：x'·x'。这是本 App 能稳定收敛的关键
              ;; （像素尺度 ⇒ K ~ 1e5 ⇒ 目标函数的几何项只占 1e-5，见头注释）。
              ;; RBF 分支仍用像素距离：K = exp(-‖Δx‖²/2σ²) 只依赖 ‖Δx‖/σ 的比值，
              ;; σ' = σ/尺度 时核值逐位相同，故无需换算。
              (set_item! kern (+ (* kc_i MAXN) kc_j)
                          (+ (* (get_item pxn kc_i) (get_item pxn kc_j))
                             (* (get_item pyn kc_i) (get_item pyn kc_j))))
            })
        (set! kc_j (+ kc_j 1))
      })
      (set! kc_i (+ kc_i 1))
    })
  }))

;; ---- 判别函数 f(x) = b + Σ α_i·y_i·K(x, x_i) ------------------------------

(define mg_k 0)
(define mg_j 0)
(define mg_s 0.0)
(define margin
  (lambda (k) {
    (set! mg_s bn)
    (set! mg_j 0)
    (while (< mg_j n) {
      (set! mg_s (+ mg_s (* (* (get_item palpha mg_j) (get_item plab mg_j))
                           (get_item kern (+ (* k MAXN) mg_j)))))
      (set! mg_j (+ mg_j 1))
    })
    mg_s
  }))

;; ---- SMO 训练（简化 SMO，逐条对齐参考页面 svmjs 的实现） ------------------

(define tr_i 0)
(define tr_ii 0)
(define tr_j 0)
(define tr_iter 0)
(define tr_passes 0)
(define tr_changed 0)
(define tr_delta 0.0)   ; 本轮内 |Δα| 的最大值（收敛判定的辅助量）
(define tr_dmax 0.0)
(define tr_ei 0.0)
(define tr_ej 0.0)
(define tr_ai 0.0)
(define tr_aj 0.0)
(define tr_l 0.0)
(define tr_h 0.0)
(define tr_eta 0.0)
(define tr_naj 0.0)
(define tr_nai 0.0)
(define tr_b1 0.0)
(define tr_b2 0.0)
(define tr_li 0.0)
(define tr_lj 0.0)
(define tr_kii 0.0)
(define tr_kij 0.0)
(define tr_kjj 0.0)
(define tr_c 0.0)
(define tr_imax 0)
(define tr_eps 0.0)   ; 本轮的 α 变化门槛（= EPSA × C）
(define tr_w0 0.0)
(define tr_w1 0.0)

(define sv_i 0)
(define sv_max 0.0)
(define sv_thr 0.0)
(define sv_select
  (lambda () {
    (set! nsv 0)
    (set! sv_i 0)
    (set! sv_max 0.0)
    (while (< sv_i n) {
      (if (> (get_item palpha sv_i) sv_max)
          (set! sv_max (get_item palpha sv_i)))
      (set! sv_i (+ sv_i 1))
    })
    (set! sv_thr (* sv_max ALPHA_RATIO))
    (set! sv_i 0)
    (while (< sv_i n) {
      (if (> (get_item palpha sv_i) sv_thr) {
        (set_item! svlist nsv sv_i)
        (set_item! svcf nsv (* (get_item palpha sv_i) (get_item plab sv_i)))
        (set_item! svsx nsv (get_item px sv_i))
        (set_item! svsy nsv (get_item py sv_i))
        (set! nsv (+ nsv 1))
      })
      (set! sv_i (+ sv_i 1))
    })
  }))

(define train
  (lambda () {
    ;; α 清零、b 清零
    (set! tr_i 0)
    (while (< tr_i n) {
      (set_item! palpha tr_i 0.0)
      (set! tr_i (+ tr_i 1))
    })
    (set! bn 0.0)
    (set! tr_iter 0)
    (set! tr_passes 0)
    (set! tr_c (if (== kv 1) C_RBF C_LIN))
    (set! tr_eps (* tr_c EPSA))
    ;; 轮数上界：按点数摊薄工作量预算（至少 8 轮，至多 MAXITER 轮）
    (set! tr_imax (Math.floor (/ TRAIN_WORK (* n n))))
    (if (> tr_imax MAXITER) (set! tr_imax MAXITER))
    (if (< tr_imax 8) (set! tr_imax 8))

    (while (and (< tr_passes NUMPASSES) (< tr_iter tr_imax)) {
      (set! tr_changed 0)
      (set! tr_dmax 0.0)
      (set! tr_i 0)
      (while (< tr_i n) {
        ;; 先推进下标再做事：下面的 continue 会跳回循环谓词，若把自增写在末尾会死循环
        (set! tr_ii tr_i)
        (set! tr_i (+ tr_i 1))

        (set! tr_li (get_item plab tr_ii))
        (set! tr_ei (- (margin tr_ii) tr_li))

        ;; KKT 条件：违反容差才需要更新 α_ii
        (if (or (and (< (* tr_li tr_ei) (- 0.0 TOL))
                     (< (get_item palpha tr_ii) tr_c))
                (and (> (* tr_li tr_ei) TOL)
                     (> (get_item palpha tr_ii) 0.0)))
            {
              ;; 随机取一个 j ≠ ii（参考页面同样是随机挑选）
              (set! tr_j (Math.floor (* (Math.random) n)))
              (if (== tr_j tr_ii)
                  (set! tr_j (if (< (+ tr_ii 1) n) (+ tr_ii 1) 0)))

              (set! tr_lj (get_item plab tr_j))
              (set! tr_ej (- (margin tr_j) tr_lj))
              (set! tr_ai (get_item palpha tr_ii))
              (set! tr_aj (get_item palpha tr_j))

              ;; α_j 的可行区间 [L, H]（盒约束 + 等式约束的交）
              (if (== tr_li tr_lj)
                  {
                    (set! tr_l (maxf 0.0 (- (+ tr_ai tr_aj) tr_c)))
                    (set! tr_h (minf tr_c (+ tr_ai tr_aj)))
                  }
                  {
                    (set! tr_l (maxf 0.0 (- tr_aj tr_ai)))
                    (set! tr_h (minf tr_c (- (+ tr_c tr_aj) tr_ai)))
                  })

              (if (< (Math.abs (- tr_l tr_h)) tr_eps) continue)

              (set! tr_kii (get_item kern (+ (* tr_ii MAXN) tr_ii)))
              (set! tr_kij (get_item kern (+ (* tr_ii MAXN) tr_j)))
              (set! tr_kjj (get_item kern (+ (* tr_j MAXN) tr_j)))
              (set! tr_eta (- (- (* 2.0 tr_kij) tr_kii) tr_kjj))
              (if (>= tr_eta 0.0) continue)          ; 非负定退化，跳过该对

              (set! tr_naj (- tr_aj (/ (* tr_lj (- tr_ei tr_ej)) tr_eta)))
              (if (> tr_naj tr_h) (set! tr_naj tr_h))
              (if (< tr_naj tr_l) (set! tr_naj tr_l))
              (if (< (Math.abs (- tr_aj tr_naj)) tr_eps) continue)

              (set! tr_nai (+ tr_ai (* (* tr_li tr_lj) (- tr_aj tr_naj))))

              ;; 先算出两个候选偏置再择优（参考页面 b1/b2 的做法）
              (set! tr_b1 (- (- (- bn tr_ei)
                                (* (* tr_li (- tr_nai tr_ai)) tr_kii))
                             (* (* tr_lj (- tr_naj tr_aj)) tr_kij)))
              (set! tr_b2 (- (- (- bn tr_ej)
                                (* (* tr_li (- tr_nai tr_ai)) tr_kij))
                             (* (* tr_lj (- tr_naj tr_aj)) tr_kjj)))
              (set! bn (* 0.5 (+ tr_b1 tr_b2)))
              (if (and (> tr_nai 0.0) (< tr_nai tr_c)) (set! bn tr_b1))
              (if (and (> tr_naj 0.0) (< tr_naj tr_c)) (set! bn tr_b2))

              (set! tr_delta (Math.abs (- tr_naj tr_aj)))
              (if (> tr_delta tr_dmax) (set! tr_dmax tr_delta))
              (set_item! palpha tr_j tr_naj)
              (set_item! palpha tr_ii tr_nai)
              (set! tr_changed (+ tr_changed 1))
            })
      })
      (set! tr_iter (+ tr_iter 1))
      ;; 收敛判定：本轮没有任何 α 更新，或更新量已落入数值噪声（|Δα| < EPSA×C），
      ;; 都算作"安静的一轮"。只判前者时，非完全可分的数据会让 α 在最优解附近
      ;; 来回抖动，永远走满 MAXITER（实测 8~10 帧，而自适应判据通常 2~3 帧）。
      (if (or (== tr_changed 0) (< tr_dmax tr_eps))
          (set! tr_passes (+ tr_passes 1))
          (set! tr_passes 0))
    })

    ;; 线性核：把 α 折算成权重，预测变成一次点积（与核展开完全等价，快得多，
    ;; 而且分类边界 w·x+b=0 可以直接解析地画出来）。
    ;; 先在归一化特征空间里累加 w' = Σ α y x'，再换算回像素单位供显示与绘图：
    ;;   f(x) = w'·x' + b' 且 x' = (x - 中心)/尺度
    ;;   ⇒ w_px = w'/尺度，b_px = b' - w_px·中心
    ;; 两种单位下的判别值是同一个数，因此"w·x+b=±1"的间隔线也完全一致。
    (set! use_w 0)
    (if (== kv 0) {
      (set! tr_w0 0.0)
      (set! tr_w1 0.0)
      (set! tr_i 0)
      (while (< tr_i n) {
        (set! tr_w0 (+ tr_w0 (* (* (get_item palpha tr_i) (get_item plab tr_i))
                                (get_item pxn tr_i))))
        (set! tr_w1 (+ tr_w1 (* (* (get_item palpha tr_i) (get_item plab tr_i))
                                (get_item pyn tr_i))))
        (set! tr_i (+ tr_i 1))
      })
      (set! w0 (* tr_w0 TISC))
      (set! w1 (* tr_w1 TISC))
      (set! b (- bn (+ (* w0 TCX) (* w1 TCY))))
      (set! use_w 1)
    }
    (if (== kv 1) (set! b bn)))

    (sv_select)
  }))

;; ---- 分类区域绘制 --------------------------------------------------------
;; 判别值 f(x) 的符号决定区域颜色（正侧淡黄 / 负侧淡蓝），参考页面以 2px 密度的
;; 栅格逐格填充。两个核分开实现：
;;   · 线性核：f 是 x 的一次函数，每一行只有**一个**颜色分界点，可以解析求出，
;;     于是每行至多两次 fill_rect（240 行 ≈ 480 次调用），成本可以忽略；
;;   · RBF 核：f 是若干高斯的叠加，没有解析等值线，只能采样——采用**行内行程合并
;;     （run-length）**：颜色相同的相邻采样格合成一次 fill_rect（否则 320x240/4px
;;     就是 4800 次 native 调用，每次约 130 条 VM 指令）。
;; 栅格化曾整帧卡 13 帧（≈200ms），改成本节的两种写法后降到 1 帧量级。

(define rl_y 0)
(define rl_dec 0.0)
(define rl_x 0)

(define raster_linear
  (lambda () {
    (set! rl_y 0)
    (while (< rl_y SH) {
      (set! rl_dec (+ b (* w1 rl_y)))          ; f(x,y) 中与 x 无关的部分
      (if (== w0 0.0)
          ;; 法向水平：整行同色
          (if (> rl_dec 0.0)
              (Gfx.fill_rect 0 rl_y SW 1 255 247 199)
              (Gfx.fill_rect 0 rl_y SW 1 179 219 234))
          {
            ;; f = 0 的分界点 x* = -(b + w1·y)/w0；f 随 x 单调
            (set! rl_x (Math.round (/ (- 0.0 rl_dec) w0)))
            (if (< rl_x 0) (set! rl_x 0))
            (if (> rl_x SW) (set! rl_x SW))
            (if (> w0 0.0)
                { ; w0 > 0：x < x* 为负侧，x ≥ x* 为正侧
                  (if (> rl_x 0)  (Gfx.fill_rect 0 rl_y rl_x 1 179 219 234))
                  (if (< rl_x SW) (Gfx.fill_rect rl_x rl_y (- SW rl_x) 1 255 247 199))
                }
                { ; w0 < 0：x < x* 为正侧
                  (if (> rl_x 0)  (Gfx.fill_rect 0 rl_y rl_x 1 255 247 199))
                  (if (< rl_x SW) (Gfx.fill_rect rl_x rl_y (- SW rl_x) 1 179 219 234))
                })
          })
      (set! rl_y (+ rl_y 1))
    })
  }))

(define rr_x 0)
(define rr_y 0)
(define rr_dec 0.0)
(define rr_pdec 0.0)  ; 上一个采样点的判别值（用于亚像素分界插值）
(define rr_px 0)      ; 上一个采样点的 x
(define rr_xn 0)      ; 插值得到的颜色分界 x（新行程起点）
(define rr_k 0)
(define rr_s 0.0)
(define rr_dx 0.0)
(define rr_dy 0.0)
(define rr_d2 0.0)
(define rr_c 0)       ; 本格颜色：0 正侧淡黄，1 负侧淡蓝
(define rr_pc -1)     ; 当前行程的颜色
(define rr_rs 0)      ; 当前行程起点 x
(define rr_rw 0)
(define rr_d 0)       ; 本次重绘用的采样步长

(define raster_flush
  (lambda (x0 x1 c) {
    (set! rr_rw (- x1 x0))
    (if (> rr_rw 0)
        (if (== c 0)
            (Gfx.fill_rect x0 rr_y rr_rw rr_d 255 247 199)
            (Gfx.fill_rect x0 rr_y rr_rw rr_d 179 219 234)))
  }))

(define raster_rbf
  (lambda () {
    ;; 采样步长按支持向量数自适应：每格成本 ∝ 支持向量数，格数 ∝ 1/步长²，
    ;; 放粗一档即可把总成本压回来（点很多时区域略粗、但重绘仍是十几帧内完成）。
    (set! rr_d (if (<= nsv 10) DENS_RBF (if (<= nsv 24) 6 8)))
    (set! rr_y 0)
    (while (< rr_y SH) {
      (set! rr_x 0)
      (set! rr_pc -1)
      (set! rr_rs 0)
      (set! rr_px 0)
      (set! rr_pdec 0.0)
      (while (< rr_x SW) {
        (set! rr_s b)
        (set! rr_k 0)
        (while (< rr_k nsv) {
          (set! rr_dx (- rr_x (get_item svsx rr_k)))
          (set! rr_dy (- rr_y (get_item svsy rr_k)))
          (set! rr_d2 (+ (* rr_dx rr_dx) (* rr_dy rr_dy)))
          ;; d² > 7200 时该支持向量的核值 < 1.2e-4，几十个叠起来也不影响判决
          ;; —— 省掉指数调用（native 调用是这里最贵的一步）
          (if (< rr_d2 7200.0)
              (set! rr_s (+ rr_s (* (get_item svcf rr_k)
                                    (Math.exp (- 0.0 (/ rr_d2 SIGMA2)))))))
          (set! rr_k (+ rr_k 1))
        })
        (set! rr_dec rr_s)
        (set! rr_c (if (> rr_dec 0.0) 0 1))
        (if (not (== rr_c rr_pc)) {
          ;; 颜色切换：在本点与前一个采样点之间按判别值线性插值求亚像素分界，
          ;; 使色块边缘横向精确到像素（否则 4px 采样格会让分界呈明显锯齿；
          ;; 逐格求逆函数代价太高，4px 跨度上 dec 已足够接近线性）。
          (if (>= rr_pc 0)
              (set! rr_xn (if (== rr_dec rr_pdec)
                              rr_px
                              (Math.round
                                (+ rr_px (* rr_d (/ (- 0.0 rr_pdec)
                                                    (- rr_dec rr_pdec)))))))
              (set! rr_xn rr_x))
          (if (< rr_xn rr_px) (set! rr_xn rr_px))
          (if (> rr_xn rr_x) (set! rr_xn rr_x))
          (if (>= rr_pc 0) (raster_flush rr_rs rr_xn rr_pc))
          (set! rr_rs rr_xn)
          (set! rr_pc rr_c)
        })
        (set! rr_px rr_x)
        (set! rr_pdec rr_dec)
        (set! rr_x (+ rr_x rr_d))
      })
      (if (>= rr_pc 0) (raster_flush rr_rs rr_x rr_pc))
      (set! rr_y (+ rr_y rr_d))
    })
  }))

(define raster
  (lambda () {
    (if (== kv 0) (raster_linear) (raster_rbf))
  }))

;; ---- 线性核：边界线 / 间隔线 / 支持向量垂线 ------------------------------
;; 直线 w·x + b = k 过点 p0 = (k-b)/|w|²·w，方向 u = (w1, -w0)/|w|。
;; 这样写对任意方向（含竖直）都成立，不必按 y = f(x) 解一次，也就没有
;; "w1 为 0 时除零"的坑（除零会直接终止 Scheme 进程）。

(define dl_w2 0.0)
(define dl_step 0.0)
(define dl_fx 0.0)
(define dl_fy 0.0)
(define dl_dx 0.0)
(define dl_dy 0.0)

(define draw_line_k
  (lambda (k r g bl) {
    (set! dl_fx (* (/ (- k b) dl_w2) w0))
    (set! dl_fy (* (/ (- k b) dl_w2) w1))
    (set! dl_dx (* w1 dl_step))
    (set! dl_dy (* (- 0.0 w0) dl_step))
    (draw_seg (- dl_fx dl_dx) (- dl_fy dl_dy)
              (+ dl_fx dl_dx) (+ dl_fy dl_dy) r g bl)
  }))

(define sc_i 0)
(define sc_id 0)
(define sc_k 0.0)
(define sc_px 0.0)
(define sc_py 0.0)
(define sc_d 0.0)

;; 支持向量圈注（两种核都画：一眼看出哪些样本决定了边界）
(define draw_sv_rings
  (lambda () {
    (set! sc_i 0)
    (while (< sc_i nsv) {
      (Gfx.draw_circle (get_item svsx sc_i) (get_item svsy sc_i) 6 40 40 50)
      (set! sc_i (+ sc_i 1))
    })
  }))

;; 线性核专属："样本 → 其所在间隔线"的垂线（参考页面同款辅助元素）
(define draw_sv_connectors
  (lambda () {
    (if (== use_w 1)
        (if (> dl_w2 0.000000001)
            {
              (set! sc_i 0)
              (while (< sc_i nsv) {
                (set! sc_id (get_item svlist sc_i))
                (set! sc_px (get_item px sc_id))
                (set! sc_py (get_item py sc_id))
                (set! sc_k (if (> (get_item plab sc_id) 0) 1.0 -1.0))
                ;; 到直线 w·x+b=k 的有符号距离（沿法向）：(w·p+b-k)/|w|²
                (set! sc_d (/ (- (+ (+ (* w0 sc_px) (* w1 sc_py)) b) sc_k) dl_w2))
                (draw_seg sc_px sc_py
                          (- sc_px (* sc_d w0)) (- sc_py (* sc_d w1))
                          95 95 105)
                (set! sc_i (+ sc_i 1))
              })
            }))
  }))

(define draw_linear
  (lambda () {
    (set! dl_w2 (+ (* w0 w0) (* w1 w1)))
    (if (> dl_w2 0.000000001) {
      ;; |w| 归一化后的方向步长：600 ≥ 屏幕对角线（400），足以横穿全屏
      (set! dl_step (/ 600.0 (Math.sqrt dl_w2)))
      (draw_line_k 1.0 95 95 105)      ; w·x+b=+1 间隔线
      (draw_line_k -1.0 95 95 105)     ; w·x+b=-1 间隔线
      (draw_line_k 0.0 225 40 40)      ; 分类边界 w·x+b=0
      (draw_sv_connectors)
    })
  }))

;; ---- 样本点 --------------------------------------------------------------

(define dp_i 0)
(define dp_x 0)
(define dp_y 0)
(define draw_points
  (lambda () {
    (set! dp_i 0)
    (while (< dp_i n) {
      (set! dp_x (get_item px dp_i))
      (set! dp_y (get_item py dp_i))
      (if (> (get_item plab dp_i) 0)
          {
            (Gfx.draw_circle dp_x dp_y 3 255 153 51)
            (Gfx.fill_circle dp_x dp_y 1 220 120 40)
          }
          {
            (Gfx.draw_circle dp_x dp_y 3 0 134 187)
            (Gfx.fill_circle dp_x dp_y 1 0 90 130)
          })
      (set! dp_i (+ dp_i 1))
    })
  }))

;; ---- 左上角参数面板 ------------------------------------------------------

(define hud_s "")
(define draw_hud
  (lambda () {
    (if (== kv 0)
        {
          (Gfx.draw_text (String.concat "线性核  C=" (i2s C_LIN)) 4 1 25 30 55)
          (Gfx.draw_text (String.concat "wx = " (fmt3 w0)) 4 15 25 30 55)
          (Gfx.draw_text (String.concat "wy = " (fmt3 w1)) 4 29 25 30 55)
          (Gfx.draw_text (String.concat "b  = " (fmt3 b)) 4 43 25 30 55)
        }
        {
          (Gfx.draw_text (String.concat "RBF核  σ=" (String.concat (i2s SIGMA)
                            (String.concat "  C=" (i2s C_RBF)))) 4 1 25 30 55)
          (Gfx.draw_text (String.concat "b  = " (fmt3 b)) 4 15 25 30 55)
          (Gfx.draw_text (String.concat "SV = " (i2s nsv)) 4 29 25 30 55)
          (Gfx.draw_text (String.concat "n  = " (i2s n)) 4 43 25 30 55)
        })
    ;; 提示行：与"正样本/负样本"按钮同色，一眼看出当前点击会加哪一类
    (if (> cat 0)
        (Gfx.draw_text "点击添加正样本" 4 57 235 140 30)
        (Gfx.draw_text "点击添加负样本" 4 57 0 120 175))
    ;; 右下角状态：样本数/容量 + 支持向量数
    (set! hud_s (String.concat "n=" (String.concat (i2s n)
                   (String.concat "/" (String.concat (i2s MAXN)
                   (String.concat "  SV=" (i2s nsv)))))))
    (Gfx.draw_text hud_s (- SW (+ 4 (tw hud_s))) (- SH 15) 70 75 95)
    ;; 右上角按旧布局标一个「返回」提示（2026-10 起该格不再是宿主返回键，
    ;; 退出改为硬 A/ESC → 全局确认框；此标记保留以免观众误当绘图区）
    (Gfx.fill_rect (- SW 40) 2 36 16 60 62 72)
    (Gfx.draw_text "返回" (- SW 37) 3 225 228 235)
  }))

;; ---- 左下角三个按钮 ------------------------------------------------------

(define BTN_Y 212)
(define BTN_H 22)
(define BTN_W 64)
(define BTN1_X 6)
(define BTN2_X 74)
(define BTN3_X 142)

(define bd_x 0)
(define bd_y 0)
(define btn_draw
  (lambda (x label r g bl tr tg tb) {
    (Gfx.fill_rect x BTN_Y BTN_W BTN_H r g bl)
    (Gfx.draw_rect x BTN_Y BTN_W BTN_H 25 25 35)
    (set! bd_x (+ x (Math.round (/ (- BTN_W (tw label)) 2))))
    (set! bd_y (+ BTN_Y 4))
    (Gfx.draw_text label bd_x bd_y tr tg tb)
  }))

(define draw_buttons
  (lambda () {
    (if (> cat 0)
        (btn_draw BTN1_X "正样本" 255 153 51 60 35 0)
        (btn_draw BTN1_X "负样本" 0 134 187 245 250 255))
    (if (== kv 0)
        (btn_draw BTN2_X "线性核" 62 118 88 240 245 240)
        (btn_draw BTN2_X "RBF核" 104 76 148 245 240 250))
    (btn_draw BTN3_X "重置" 186 62 62 255 248 245)
  }))

(define btn_hit
  (lambda (x y) {
    (if (and (>= y BTN_Y) (< y (+ BTN_Y BTN_H)))
        (if (and (>= x BTN1_X) (< x (+ BTN1_X BTN_W))) 1
            (if (and (>= x BTN2_X) (< x (+ BTN2_X BTN_W))) 2
                (if (and (>= x BTN3_X) (< x (+ BTN3_X BTN_W))) 3 0)))
        0)
  }))

;; ---- 整帧渲染 ------------------------------------------------------------

(define render
  (lambda () {
    (if (>= n 2)
        (raster)
        (Gfx.fill_rect 0 0 SW SH 255 255 255))
    (if (== kv 0) (draw_linear))
    (draw_sv_rings)
    (draw_points)
    (draw_hud)
    (draw_buttons)
    (Gfx.refresh)
  }))

;; ---- 数据与重算 ----------------------------------------------------------

(define ad_x 0)
(define ad_y 0)
(define add_sample
  (lambda (x y) {
    (if (< n MAXN) {
      (set! ad_x x)
      (set! ad_y y)
      (if (< ad_x 0) (set! ad_x 0))
      (if (> ad_x (- SW 1)) (set! ad_x (- SW 1)))
      (if (< ad_y 0) (set! ad_y 0))
      (if (> ad_y (- SH 1)) (set! ad_y (- SH 1)))
      (set_item! px n ad_x)
      (set_item! py n ad_y)
      (set_item! pxn n (* (- ad_x TCX) TISC))
      (set_item! pyn n (* (- ad_y TCY) TISC))
      (set_item! plab n cat)
      (set_item! palpha n 0.0)
      (set! n (+ n 1))
      (set! dirty 1)
    })
  }))

;; 重算：训练可能跨多帧（解释器按时间片推进），先给一个"计算中"提示并上屏
(define recompute
  (lambda () {
    (Gfx.fill_rect (- SW 46) (- SH 17) 40 15 255 255 255)
    (Gfx.draw_text "计算中" (- SW 44) (- SH 15) 60 60 70)
    (Gfx.refresh)
    (if (> n 0) (kern_build))
    (if (>= n 2)
        (train)
        {
          (set! w0 0.0)
          (set! w1 0.0)
          (set! b 0.0)
          (set! bn 0.0)
          (set! nsv 0)
          (set! use_w 0)
        })
    (render)
  }))

(define do_button
  (lambda (k) {
    (if (== k 1) {
      ;; 切换待添加类别：样本本身不变，只需重绘
      (set! cat (if (> cat 0) -1 1))
      (render)
    })
    (if (== k 2) {
      ;; 切换核函数：内核矩阵与模型都要重算
      (set! kv (if (== kv 0) 1 0))
      (set! dirty 1)
    })
    (if (== k 3) {
      ;; 重置：清空样本、模型、类别与核，回到初始态
      (set! n 0)
      (set! w0 0.0)
      (set! w1 0.0)
      (set! b 0.0)
      (set! bn 0.0)
      (set! nsv 0)
      (set! use_w 0)
      (set! cat 1)
      (set! kv 0)
      (set! pending 0)
      (render)
    })
  }))

(define rl 0)
(define tick
  (lambda () {
    (set! tp (if (Touch.take_press) 1 0))
    (set! tr (if (Touch.take_release) 1 0))

    ;; 按下沿：落在按钮上 → 武装（松手沿结算，同项目触屏交互范式）；
    ;;         落在别处 → 立即添加一个样本点
    (if (== tp 1) {
      (set! press_x (Touch.x))
      (set! press_y (Touch.y))
      (set! pending (btn_hit press_x press_y))
      (if (== pending 0) (add_sample press_x press_y))
    })

    ;; 松开沿：仍在同一按钮内才结算（允许滑出取消）
    (if (== tr 1)
        (if (> pending 0) {
          (set! rl (btn_hit (Touch.x) (Touch.y)))
          (if (== rl pending) (do_button pending))
          (set! pending 0)
        }))

    (if (== dirty 1) {
      (set! dirty 0)
      (recompute)
    })
  }))

;; ---- 启动 ----------------------------------------------------------------

(render)
(set! _timer (System.set_interval TICK_MS tick))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 render 擦掉对话框残影）
(EX.install_with render)
