;; ============================================================================
;; cnn.scm —— 卷积神经网络（CNN）手写数字识别与可视化（Nano「灵机应用程序」）
;;
;; 参考实现：https://adamharley.com/nn_vis/cnn/2d.html
;;   （Adam Harley 的 2D CNN 可视化：画布输入 → 卷积核/特征图 → 逐节点查看计算 → 输出概率）
;;
;; 本 App 的取舍（320x240 装不下参考页面的全部信息，只保留教学上最要紧的）：
;;   · 左栏（常驻）：112x112 绘图板（= 28x28 网格、每格 4px，正是网络看到的输入）、
;;           预测结果及其置信度、10 类 softmax 概率柱状图。特征图/卷积核一律按
;;           **真实数值**着色，不做任何示意。
;;   · 右栏（两个视图，顶部「层2/层1」按钮切换，对应参考页面的图层选择器）：
;;       ‹层1› 第一层 6 个 5x5 卷积核，每个核**正上方**是它自己的 12x12 特征图；
;;              点按任一格 → 下方给出「输入块 × 卷积核 = 逐元素积 → 和 → ReLU」
;;              现场算式，并在绘图板上框出该神经元的感受野（5x5 输入块）。
;;       ‹层2› 卷积2 的 8 张 4x4 特征图；点按任一格 → 下方用 6 根左右向条形图
;;              显示「池化后 6 个通道各自贡献多少」（这正是多通道卷积的本质），
;;              绘图板上框出该神经元的 15x15 感受野。
;;     两层的默认选中格都是**全图最强**的那个神经元（推理后自动更新）。
;;   · 全连接层（128→10）不做空间可视化（128 个权重画不下），其真实数值经 softmax
;;     变成左栏的概率柱状图——识别结果本身就是它的可视化。
;;
;; 网络（权重见 app/lib/cnn_weights.scm，由 tools/train_cnn_mnist.py 训练导出）：
;;   MNIST 测试集准确率 98.11%；一次推理 29,792 次乘加，wasm 上实测 60~65ms
;;   （解释器每帧 31.25 万条 VM 指令，约 38 条指令/乘加 ⇒ 约 3.6 帧；见 doc/AGENTS.md）。
;;     输入   1x28x28（像素/255）
;;     conv1  6 个 5x5，步长 2，无填充  → 6x12x12，ReLU
;;     pool1  2x2 最大池化              → 6x6x6
;;     conv2  8 个 3x3x6，步长 1         → 8x4x4，ReLU
;;     fc     128 → 10 → softmax
;;
;; 设计要点：
;;   · 不做「自动居中/尺寸归一化」：画上去什么就是网络看到的什么（与参考页面一致，
;;     也便于把可视化与输入一一对应）；代价是写得太偏/太大时准确率会下降——这是
;;     本网络的真实行为，不做掩饰。
;;   · 推理只在**松手**后触发（首次进入不推理），拖动笔画期间零推理开销；
;;     推理跨 3~4 帧（时间片），期间预测行显示「计算中…」。
;;   · 特征图按「每图自身的最大值」归一化着色，只为看清图案；右侧检视面板给出的
;;     是未归一化的真实数值。卷积核颜色按全部 150 个权重共用的 ±max|W1| 归一化。
;;   · 顶部「清空」清掉墨迹与**各层激活/概率**（不留上一次的残影）。
;;
;; 解释器约束（详见 Animac manual.md 与 doc/AGENTS.md，逐条遵守）：
;;   · 算术/比较固定二参数，多元必须嵌套；没有一元取负（写 (- 0.0 x)）；
;;   · 没有 let/list/append/map；define 无过程简写；块内不 define，只用模块级 set!；
;;   · / 与 mod 恒返回浮点 → 下标一律 Math.round/floor 收口；除零会终止进程，先判后除；
;;   · 跨函数调用链上的模块级临时量绝不共用（按「谁写谁读」分配前缀名）；
;;   · Gfx.draw_line 的坐标是 uint32：负数会被当成天文数字，故一律先钳再画。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Diag)

(import EX "app/lib/exit.scm")          ; 硬 A/ESC → 全局退出确认框 + 右上角「退出」按钮
(import W "app/lib/cnn_weights.scm")    ; 训练好的权重（自动生成，勿手改）

;; ---- 画布与布局 -----------------------------------------------------------

(define SW (Gfx.width))

(define PAD_X 4)          ; 绘图板左上角（格区）
(define PAD_Y 22)
(define PAD_CELL 4)       ; 每格像素（28*4 = 112）
(define PAD_W (* 28 PAD_CELL))
(define PAD_H (* 28 PAD_CELL))

(define RX 122)           ; 右栏左边界
(define TICK_MS 20)       ; 输入采样/重绘周期（≈50Hz）

;; 右栏纵向布局：卷积核（20px）正上方对齐各自的特征图（48px），两行各 3 组
(define COL_P 52)         ; 列间距（3 列：122 / 174 / 226）
(define KG_Y0 30)         ; 第一组卷积核顶边
(define MG_Y0 52)         ; 第一组特征图顶边
(define KG_Y1 104)        ; 第二组卷积核顶边
(define MG_Y1 126)        ; 第二组特征图顶边
(define MK_S 4)           ; 卷积核每格像素（5*4 = 20）
(define MP_S 4)           ; 特征图每格像素（12*4 = 48）

;; 检视面板：三块 5x5 网格（输入块 / 卷积核 / 逐元素积）
(define RI_Y 190)
(define RI_S 6)           ; 每格像素（5*6 = 30）
(define RI_X 122)
(define RJ_X 156)
(define RK_X 190)
(define RI_TX 224)        ; 结果文字的左边界

;; 第二层视图：卷积2 的 8 张 4x4 特征图（每格 10px、列距 44），下方是通道贡献条形图
(define C2_Y0 30)
(define C2_Y1 74)
(define C2_S 10)
(define C2_P 44)
;; 通道贡献条：以零点为中心左右伸展（0 像素 = 该通道对此神经元的贡献为 0）
(define CB_ZERO 198)
(define CB_HALF 52)

;; 概率柱状图（左栏底部）
(define BAR_Y 196)        ; 基线（底边）
(define BAR_H 38)         ; 柱高上限
(define BAR_W 9)
(define BAR_P 11)         ; 柱间距

;; 绘图笔刷：以格为单位的半径与连线插值步长（MNIST 笔画约 2~3 px 宽）
(define BRUSH_R 1.3)
(define PEN_STEP 0.5)

;; ---- 权重别名（热循环里直接引用，省一层模块成员解析） ----------------------

(define W1 W.W1)
(define B1 W.B1)
(define W2 W.W2)
(define B2 W.B2)
(define W3 W.W3)
(define B3 W.B3)

;; ---- 小工具 ---------------------------------------------------------------

(define i2s (lambda (x) (String.atom_to_string x)))
(define maxf (lambda (a b) (if (> a b) a b)))
(define minf (lambda (a b) (if (< a b) a b)))

(define mz_i 0)
(define make_zeros
  (lambda (k) {
    (define acc (cdr '(0)))     ; 新列表对象（顶层 '() 是静态槽，push 会跨调用残留）
    (set! mz_i 0)
    (while (< mz_i k) {
      (push acc 0.0)
      (set! mz_i (+ mz_i 1))
    })
    acc
  }))

;; 文本宽度（像素）：ASCII 等字符步进 7，CJK（U+2E80 起）步进 12
(define tw_s 0)
(define tw_i 0)
(define tw
  (lambda (s) {
    (set! tw_s 0)
    (set! tw_i 0)
    (while (< tw_i (String.length s)) {
      (set! tw_s (+ tw_s (if (< (String.charCodeAt tw_i s) 11904) 7 12)))
      (set! tw_i (+ tw_i 1))
    })
    tw_s
  }))

;; 定点格式化（两位小数；VM 浮点载荷位宽有限，故自己拼字符串）
(define f_v 0)
(define f_a 0)
(define f_ip 0)
(define f_fp 0)
(define f_neg 0)
(define pad2
  (lambda (v)
    (if (< (Math.round v) 10)
        (String.concat "0" (i2s (Math.round v)))
        (i2s (Math.round v)))))
(define fmt2
  (lambda (x) {
    (set! f_v (Math.round (* x 100.0)))
    (set! f_neg (if (< f_v 0) 1 0))
    (set! f_a (if (== f_neg 1) (- 0 f_v) f_v))
    (set! f_ip (Math.floor (/ f_a 100.0)))
    (set! f_fp (- f_a (* f_ip 100.0)))
    (String.concat (if (== f_neg 1) "-" "")
      (String.concat (i2s f_ip) (String.concat "." (pad2 f_fp))))
  }))

;; 概率 → "98.7%"（一位小数）
(define pc_v 0)
(define fmt_pct
  (lambda (p) {
    (set! pc_v (Math.round (* p 1000.0)))
    (String.concat (i2s (Math.floor (/ pc_v 10.0)))
      (String.concat "." (String.concat (i2s (mod pc_v 10.0)) "%")))
  }))

;; ---- 缓冲区（预分配列表 + 原地写入，热路径零分配） ------------------------
;; 展平索引一律行主序：img[y*28+x]、c1[k*144 + r*12 + c]、p1[k*36 + r*6 + c]、
;; c2[f*16 + r*4 + c]（与 numpy 的 NCHW 展平顺序逐位一致）

(define img (make_zeros 784))    ; 输入 28x28
(define c1 (make_zeros 864))     ; 卷积1 6x12x12（ReLU 后）
(define p1 (make_zeros 216))     ; 池化1 6x6x6
(define c2 (make_zeros 128))     ; 卷积2 8x4x4（ReLU 后）
(define lg (make_zeros 10))      ; logits
(define pb (make_zeros 10))      ; softmax 概率
(define mx1 (make_zeros 6))      ; 每个卷积1 特征图的显示归一化分母
(define mx2 (make_zeros 8))      ; 每个卷积2 特征图的显示归一化分母
(define contrib (make_zeros 6))  ; 选中卷积2 神经元时各输入通道的贡献

;; ---- 交互/显示状态 -------------------------------------------------------

(define has_ink 0)        ; 1 = 画面有墨
(define pred_idx -1)      ; 预测数字（-1 = 尚未推理）
(define pred_conf 0.0)    ; 其概率
(define stale 0)          ; 1 = 需要（重新）推理
(define computing 0)      ; 1 = 正在推理（画「计算中」用）
(define infer_ms 0)       ; 上次推理耗时（毫秒，仅供显示）
(define sel_k 0)          ; 检视中的卷积核/特征图序号（第一层）
(define sel_r 0)          ; 检视中的格（行）
(define sel_c 0)          ; 检视中的格（列）
(define sel_on 0)         ; 1 = 有选中格
(define view 0)           ; 0 = 右栏显示第一层，1 = 显示第二层
(define sel2_f 0)         ; 第二层检视中的特征图 / 格
(define sel2_r 0)
(define sel2_c 0)
(define sel2_on 0)

(define pen_last_x 0.0)   ; 上一采样点的网格坐标（浮点，格中心为 +0.5）
(define pen_last_y 0.0)
(define pen_down 0)       ; 上一拍是否在绘图板上按住
(define armed 0)          ; 按下时落在「清空」按钮上
(define w1max 0.000001)   ; conv1 权重的 ±归一化分母（卷积核着色用）
(define _timer 0)

;; 权重的绝对值上界（卷积核着色用）
(define iw_i 0)
(define iw_m 0.0)
(define wmax_scan
  (lambda (lst n) {
    (set! iw_m 0.0)
    (set! iw_i 0)
    (while (< iw_i n) {
      (set! iw_m (maxf iw_m (Math.abs (get_item lst iw_i))))
      (set! iw_i (+ iw_i 1))
    })
    (if (< iw_m 0.000001) 0.000001 iw_m)
  }))

;; ---- 前向计算：卷积1 ------------------------------------------------------
;; 6 个 5x5 核、步长 2、无填充：c1[k][r][c] = ReLU(b_k + Σ w·img)
;; 内层用「递增下标」而非每次重算索引——每次乘加 ≈ 40 条 VM 指令（实测）。

(define k1_k 0)
(define k1_r 0)
(define k1_c 0)
(define k1_ky 0)
(define k1_kx 0)
(define k1_acc 0.0)
(define k1_wi 0)
(define k1_ii 0)
(define k1_jj 0)

(define eval_conv1
  (lambda () {
    (set! k1_k 0)
    (while (< k1_k 6) {
      (set! k1_r 0)
      (while (< k1_r 12) {
        (set! k1_c 0)
        (while (< k1_c 12) {
          (set! k1_acc (get_item B1 k1_k))
          (set! k1_wi (* k1_k 25))
          (set! k1_ii (+ (* k1_r 56) (* k1_c 2)))   ; img 起始下标（步长 2 ⇒ 行跨 56）
          (set! k1_ky 0)
          (while (< k1_ky 5) {
            (set! k1_jj k1_ii)
            (set! k1_kx 0)
            (while (< k1_kx 5) {
              (set! k1_acc (+ k1_acc (* (get_item W1 k1_wi) (get_item img k1_jj))))
              (set! k1_wi (+ k1_wi 1))
              (set! k1_jj (+ k1_jj 1))
              (set! k1_kx (+ k1_kx 1))
            })
            (set! k1_ii (+ k1_ii 28))
            (set! k1_ky (+ k1_ky 1))
          })
          (set_item! c1 (+ (* k1_k 144) (+ (* k1_r 12) k1_c))
                     (if (> k1_acc 0.0) k1_acc 0.0))
          (set! k1_c (+ k1_c 1))
        })
        (set! k1_r (+ k1_r 1))
      })
      (set! k1_k (+ k1_k 1))
    })
  }))

;; ---- 前向计算：2x2 最大池化 ----------------------------------------------

(define pl_k 0)
(define pl_r 0)
(define pl_c 0)
(define pl_i 0)
(define pl_j 0)
(define pl_m 0.0)
(define pl_v 0.0)
(define pl_b 0)

(define eval_pool1
  (lambda () {
    (set! pl_k 0)
    (while (< pl_k 6) {
      (set! pl_r 0)
      (while (< pl_r 6) {
        (set! pl_c 0)
        (while (< pl_c 6) {
          (set! pl_m 0.0)
          (set! pl_i 0)
          (while (< pl_i 2) {
            (set! pl_b (+ (* pl_k 144) (+ (* (+ (* pl_r 2) pl_i) 12) (* pl_c 2))))
            (set! pl_j 0)
            (while (< pl_j 2) {
              (set! pl_v (get_item c1 (+ pl_b pl_j)))
              (if (> pl_v pl_m) (set! pl_m pl_v) #f)
              (set! pl_j (+ pl_j 1))
            })
            (set! pl_i (+ pl_i 1))
          })
          (set_item! p1 (+ (* pl_k 36) (+ (* pl_r 6) pl_c)) pl_m)
          (set! pl_c (+ pl_c 1))
        })
        (set! pl_r (+ pl_r 1))
      })
      (set! pl_k (+ pl_k 1))
    })
  }))

;; ---- 前向计算：卷积2（8 个 3x3x6 核，步长 1） ----------------------------

(define k2_f 0)
(define k2_r 0)
(define k2_c 0)
(define k2_ch 0)
(define k2_ky 0)
(define k2_kx 0)
(define k2_acc 0.0)
(define k2_wi 0)
(define k2_ii 0)
(define k2_jj 0)

(define eval_conv2
  (lambda () {
    (set! k2_f 0)
    (while (< k2_f 8) {
      (set! k2_r 0)
      (while (< k2_r 4) {
        (set! k2_c 0)
        (while (< k2_c 4) {
          (set! k2_acc (get_item B2 k2_f))
          (set! k2_wi (* k2_f 54))
          (set! k2_ch 0)
          (while (< k2_ch 6) {
            (set! k2_ii (+ (* k2_ch 36) (+ (* k2_r 6) k2_c)))
            (set! k2_ky 0)
            (while (< k2_ky 3) {
              (set! k2_jj k2_ii)
              (set! k2_kx 0)
              (while (< k2_kx 3) {
                (set! k2_acc (+ k2_acc (* (get_item W2 k2_wi) (get_item p1 k2_jj))))
                (set! k2_wi (+ k2_wi 1))
                (set! k2_jj (+ k2_jj 1))
                (set! k2_kx (+ k2_kx 1))
              })
              (set! k2_ii (+ k2_ii 6))
              (set! k2_ky (+ k2_ky 1))
            })
            (set! k2_ch (+ k2_ch 1))
          })
          (set_item! c2 (+ (* k2_f 16) (+ (* k2_r 4) k2_c))
                     (if (> k2_acc 0.0) k2_acc 0.0))
          (set! k2_c (+ k2_c 1))
        })
        (set! k2_r (+ k2_r 1))
      })
      (set! k2_f (+ k2_f 1))
    })
  }))

;; ---- 前向计算：全连接 128 → 10 + softmax --------------------------------

(define fc_d 0)
(define fc_i 0)
(define fc_acc 0.0)
(define fc_wi 0)
(define sm_i 0)
(define sm_m 0.0)
(define sm_s 0.0)
(define sm_v 0.0)

(define eval_fc
  (lambda () {
    (set! fc_d 0)
    (while (< fc_d 10) {
      (set! fc_acc (get_item B3 fc_d))
      (set! fc_wi (* fc_d 128))
      (set! fc_i 0)
      (while (< fc_i 128) {
        (set! fc_acc (+ fc_acc (* (get_item W3 fc_wi) (get_item c2 fc_i))))
        (set! fc_wi (+ fc_wi 1))
        (set! fc_i (+ fc_i 1))
      })
      (set_item! lg fc_d fc_acc)
      (set! fc_d (+ fc_d 1))
    })
    ;; softmax（先减最大值，避免 exp 溢出）
    (set! sm_m (get_item lg 0))
    (set! sm_i 1)
    (while (< sm_i 10) {
      (if (> (get_item lg sm_i) sm_m) (set! sm_m (get_item lg sm_i)) #f)
      (set! sm_i (+ sm_i 1))
    })
    (set! sm_s 0.0)
    (set! sm_i 0)
    (while (< sm_i 10) {
      (set! sm_v (Math.exp (- (get_item lg sm_i) sm_m)))
      (set! sm_s (+ sm_s sm_v))
      (set_item! pb sm_i sm_v)
      (set! sm_i (+ sm_i 1))
    })
    (set! sm_i 0)
    (while (< sm_i 10) {
      (set_item! pb sm_i (/ (get_item pb sm_i) sm_s))
      (set! sm_i (+ sm_i 1))
    })
  }))

;; ---- 一次完整推理 + 显示量统计 ------------------------------------------

(define pp_i 0)
(define pp_best 0)
(define pp_bv 0.0)
(define pp_t 0)
(define pp_argmax
  (lambda () {
    (set! pp_best 0)
    (set! pp_bv (get_item pb 0))
    (set! pp_i 1)
    (while (< pp_i 10) {
      (if (> (get_item pb pp_i) pp_bv) {
        (set! pp_bv (get_item pb pp_i))
        (set! pp_best pp_i)
      })
      (set! pp_i (+ pp_i 1))
    })
  }))

;; 每个特征图的最大激活（着色归一化分母）＋顺带挑选「全图最强神经元」作为检视默认目标
(define sc_k 0)
(define sc_i 0)
(define sc_v 0.0)
(define sc_m 0.0)
(define sc_mk 0)
(define sc_mi 0)
(define sc_best 0.0)
(define scan_maps
  (lambda () {
    (set! sc_best 0.0)
    (set! sc_mk 0)
    (set! sc_mi 0)
    (set! sc_k 0)
    (while (< sc_k 6) {
      (set! sc_m 0.0)
      (set! sc_i 0)
      (while (< sc_i 144) {
        (set! sc_v (get_item c1 (+ (* sc_k 144) sc_i)))
        (if (> sc_v sc_m) (set! sc_m sc_v) #f)
        (if (> sc_v sc_best) {
          (set! sc_best sc_v)
          (set! sc_mk sc_k)
          (set! sc_mi sc_i)
        })
        (set! sc_i (+ sc_i 1))
      })
      (set_item! mx1 sc_k (if (< sc_m 0.15) 0.15 sc_m))
      (set! sc_k (+ sc_k 1))
    })
    (if (> sc_best 0.0) {
      (set! sel_k sc_mk)
      (set! sel_r (Math.floor (/ sc_mi 12.0)))
      (set! sel_c (- sc_mi (* sel_r 12)))
      (set! sel_on 1)
    })
    ;; 第二层同理：显示归一化分母 + 默认选中最强神经元
    (set! sc_best 0.0)
    (set! sc_mk 0)
    (set! sc_mi 0)
    (set! sc_k 0)
    (while (< sc_k 8) {
      (set! sc_m 0.0)
      (set! sc_i 0)
      (while (< sc_i 16) {
        (set! sc_v (get_item c2 (+ (* sc_k 16) sc_i)))
        (if (> sc_v sc_m) (set! sc_m sc_v) #f)
        (if (> sc_v sc_best) {
          (set! sc_best sc_v)
          (set! sc_mk sc_k)
          (set! sc_mi sc_i)
        })
        (set! sc_i (+ sc_i 1))
      })
      (set_item! mx2 sc_k (if (< sc_m 0.15) 0.15 sc_m))
      (set! sc_k (+ sc_k 1))
    })
    (if (> sc_best 0.0) {
      (set! sel2_f sc_mk)
      (set! sel2_r (Math.floor (/ sc_mi 4.0)))
      (set! sel2_c (- sc_mi (* sel2_r 4)))
      (set! sel2_on 1)
    })
  }))

(define run_net
  (lambda () {
    (set! pp_t (Diag.ms))
    (eval_conv1)
    (eval_pool1)
    (eval_conv2)
    (eval_fc)
    (pp_argmax)
    (set! pred_idx pp_best)
    (set! pred_conf pp_bv)
    (set! infer_ms (- (Diag.ms) pp_t))
    (scan_maps)
  }))

;; ---- 三种着色：热色（激活）/ 蓝红（有符号权重）/ 灰阶（墨迹） ------------

(define ac_t 0.0)
(define ac_u 0.0)
;; 激活值 → 热色（0 白 → 0.5 橙 → 1 深红），v 已归一化到 [0,1]
(define draw_act
  (lambda (x y s v) {
    (set! ac_t (minf 1.0 (maxf 0.0 v)))
    (if (< ac_t 0.5)
        (Gfx.fill_rect x y s s
                       (Math.round (- 252.0 (* 3.0 ac_t)))
                       (Math.round (- 250.0 (* 125.0 ac_t)))
                       (Math.round (- 247.0 (* 382.0 ac_t))))
        {
          (set! ac_u (- (* 2.0 ac_t) 1.0))
          (Gfx.fill_rect x y s s
                         (Math.round (- 255.0 (* 90.0 ac_u)))
                         (Math.round (- 190.0 (* 150.0 ac_u)))
                         (Math.round (- 60.0 (* 20.0 ac_u))))
        })
  }))

;; 有符号权重 → 蓝(负) / 白(0) / 红(正)，w 已归一化到 [-1,1]
(define wc_u 0.0)
(define draw_wt
  (lambda (x y s w) {
    (if (< w 0.0)
        {
          (set! wc_u (minf 1.0 (- 0.0 w)))
          (Gfx.fill_rect x y s s
                         (Math.round (- 255.0 (* 195.0 wc_u)))
                         (Math.round (- 255.0 (* 165.0 wc_u)))
                         255)
        }
        {
          (set! wc_u (minf 1.0 w))
          (Gfx.fill_rect x y s s 255
                         (Math.round (- 255.0 (* 210.0 wc_u)))
                         (Math.round (- 255.0 (* 205.0 wc_u))))
        })
  }))

;; 灰度（墨迹强度 v：0 白 → 1 黑）
(define gc_l 0)
(define draw_ink
  (lambda (x y s v) {
    (set! gc_l (Math.round (- 255.0 (* 255.0 (minf 1.0 (maxf 0.0 v))))))
    (Gfx.fill_rect x y s s gc_l gc_l gc_l)
  }))

;; ---- 绘图板 --------------------------------------------------------------

(define rp_i 0)
(define rp_j 0)
(define rp_x 0)
(define rp_y 0)
(define draw_pad
  (lambda () {
    (set! rp_i 0)
    (while (< rp_i 28) {
      (set! rp_j 0)
      (while (< rp_j 28) {
        (set! rp_x (+ PAD_X (* rp_j PAD_CELL)))
        (set! rp_y (+ PAD_Y (* rp_i PAD_CELL)))
        (draw_ink rp_x rp_y PAD_CELL (get_item img (+ (* rp_i 28) rp_j)))
        (set! rp_j (+ rp_j 1))
      })
      (set! rp_i (+ rp_i 1))
    })
  }))

;; 感受野框：把选中神经元在**输入图像**上看到的范围框出来
;;   · 第一层：5x5 输入块（绘图板上的 2r..2r+4 行 / 2c..2c+4 列）
;;   · 第二层：3x3 池化块（每格 2x2 卷积1 格、每格 5x5 输入）⇒ 15x15 输入像素，
;;     起点 4r 列（= 池化格 2r 的 2r*2 个卷积1 格、每格 2 输入像素）
(define rf_x 0)
(define rf_y 0)
(define rf_n 0)
(define draw_receptive
  (lambda () {
    (if (== view 0)
        (if (== sel_on 1) {
          (set! rf_x (+ PAD_X (* (* sel_c 2) PAD_CELL)))
          (set! rf_y (+ PAD_Y (* (* sel_r 2) PAD_CELL)))
          (set! rf_n 5)
          (Gfx.draw_rect (- rf_x 1) (- rf_y 1) (+ (* rf_n PAD_CELL) 1) (+ (* rf_n PAD_CELL) 1) 30 150 90)
        }
        #f)
        (if (== sel2_on 1) {
          (set! rf_x (+ PAD_X (* (* sel2_c 4) PAD_CELL)))
          (set! rf_y (+ PAD_Y (* (* sel2_r 4) PAD_CELL)))
          (set! rf_n 15)
          (Gfx.draw_rect (- rf_x 1) (- rf_y 1) (+ (* rf_n PAD_CELL) 1) (+ (* rf_n PAD_CELL) 1) 30 150 90)
        }
        #f))
  }))

;; ---- 预测结果与概率柱状图 ------------------------------------------------

(define db_i 0)
(define db_x 0)
(define db_h 0)
(define draw_bars
  (lambda () {
    (set! db_i 0)
    (while (< db_i 10) {
      (set! db_x (+ PAD_X (* db_i BAR_P)))
      (Gfx.fill_rect db_x (- BAR_Y BAR_H) BAR_W BAR_H 235 238 243)   ; 轨道
      (set! db_h (Math.round (* BAR_H (get_item pb db_i))))
      (if (> db_h 0)
          (if (== db_i pred_idx)
              (Gfx.fill_rect db_x (- BAR_Y db_h) BAR_W db_h 40 120 215)
              (Gfx.fill_rect db_x (- BAR_Y db_h) BAR_W db_h 150 165 190))
          #f)
      ;; 类别标签（命中的一类加粗——同一字画两遍、错位 1px）
      (if (== db_i pred_idx)
          {
            (Gfx.draw_text (i2s db_i) (+ db_x 1) (+ BAR_Y 2) 20 60 140)
            (Gfx.draw_text (i2s db_i) (+ db_x 2) (+ BAR_Y 2) 20 60 140)
          }
          (Gfx.draw_text (i2s db_i) (+ db_x 1) (+ BAR_Y 2) 115 122 138))
      (set! db_i (+ db_i 1))
    })
  }))

(define dp_s "")
(define draw_pred
  (lambda () {
    (if (== computing 1)
        (Gfx.draw_text "计算中…" PAD_X 138 150 110 20)
        (if (== pred_idx -1)
            (Gfx.draw_text "在框内写 0~9" PAD_X 138 135 140 155)
            {
              (Gfx.draw_text "预测" PAD_X 138 60 66 90)
              (set! dp_s (i2s pred_idx))
              (Gfx.draw_text dp_s (+ PAD_X 26) 138 25 95 200)
              (Gfx.draw_text dp_s (+ PAD_X 27) 138 25 95 200)
              (Gfx.draw_text (fmt_pct pred_conf) (+ PAD_X 42) 138 60 66 90)
            }))
  }))

;; ---- 右栏：卷积核（每个核正上方对齐自己的特征图） ------------------------

(define dk_k 0)
(define dk_i 0)
(define dk_j 0)
(define dk_x 0)
(define dk_y 0)
(define map_x 0)
(define map_y 0)
(define kernel_y
  (lambda (k) (if (< k 3) KG_Y0 KG_Y1)))
(define map_oy
  (lambda (k) (if (< k 3) MG_Y0 MG_Y1)))

(define draw_kernels
  (lambda () {
    (set! dk_k 0)
    (while (< dk_k 6) {
      (set! dk_x (+ RX (* (mod dk_k 3) COL_P)))
      (set! dk_y (kernel_y dk_k))
      (set! dk_i 0)
      (while (< dk_i 5) {
        (set! dk_j 0)
        (while (< dk_j 5) {
          (draw_wt (+ dk_x (* dk_j MK_S)) (+ dk_y (* dk_i MK_S)) MK_S
                   (/ (get_item W1 (+ (* dk_k 25) (+ (* dk_i 5) dk_j))) w1max))
          (set! dk_j (+ dk_j 1))
        })
        (set! dk_i (+ dk_i 1))
      })
      (if (== dk_k sel_k)
          (Gfx.draw_rect (- dk_x 1) (- dk_y 1) (+ (* 5 MK_S) 1) (+ (* 5 MK_S) 1) 25 95 200)
          #f)
      (set! dk_k (+ dk_k 1))
    })
  }))

(define dm_k 0)
(define dm_i 0)
(define dm_j 0)
(define dm_v 0.0)
(define draw_maps
  (lambda () {
    (set! dm_k 0)
    (while (< dm_k 6) {
      (set! map_x (+ RX (* (mod dm_k 3) COL_P)))
      (set! map_y (map_oy dm_k))
      (set! dm_i 0)
      (while (< dm_i 12) {
        (set! dm_j 0)
        (while (< dm_j 12) {
          (set! dm_v (get_item c1 (+ (* dm_k 144) (+ (* dm_i 12) dm_j))))
          (draw_act (+ map_x (* dm_j MP_S)) (+ map_y (* dm_i MP_S)) MP_S
                    (/ dm_v (get_item mx1 dm_k)))
          (set! dm_j (+ dm_j 1))
        })
        (set! dm_i (+ dm_i 1))
      })
      ;; 图框（画在格子之后，否则右/下边会被格子覆盖）
      (Gfx.draw_rect (- map_x 2) (- map_y 2) (+ (* 12 MP_S) 4) (+ (* 12 MP_S) 4) 205 210 220)
      ;; 选中格的框
      (if (== dm_k sel_k)
          (Gfx.draw_rect (- (+ map_x (* sel_c MP_S)) 1) (- (+ map_y (* sel_r MP_S)) 1)
                         (+ MP_S 2) (+ MP_S 2) 25 95 200)
          #f)
      (set! dm_k (+ dm_k 1))
    })
  }))

;; ---- 右栏：检视面板（选中神经元的完整算式） -----------------------------

(define iv_i 0)
(define iv_j 0)
(define iv_v 0.0)
(define iv_w 0.0)
(define iv_p 0.0)
(define iv_sum 0.0)
(define iv_act 0.0)
(define iv_pmax 0.000001)
(define iv_gx 0)
(define iv_gy 0)
(define draw_inspector
  (lambda () {
    (if (== sel_on 0)
        (Gfx.draw_text "点按特征图某格：看它的计算" RI_X (+ RI_Y 4) 135 140 155)
        {
          ;; 先求 |逐元素积| 的最大值（着色归一化），再画三块网格
          (set! iv_pmax 0.000001)
          (set! iv_i 0)
          (while (< iv_i 5) {
            (set! iv_j 0)
            (while (< iv_j 5) {
              (set! iv_p (* (get_item img (+ (* 28 (+ (* sel_r 2) iv_i)) (+ (* sel_c 2) iv_j)))
                            (get_item W1 (+ (* sel_k 25) (+ (* iv_i 5) iv_j)))))
              (set! iv_pmax (maxf iv_pmax (Math.abs iv_p)))
              (set! iv_j (+ iv_j 1))
            })
            (set! iv_i (+ iv_i 1))
          })
          (set! iv_sum (get_item B1 sel_k))
          (set! iv_i 0)
          (while (< iv_i 5) {
            (set! iv_j 0)
            (while (< iv_j 5) {
              (set! iv_gx (+ RI_X (* iv_j RI_S)))
              (set! iv_gy (+ RI_Y (* iv_i RI_S)))
              (set! iv_v (get_item img (+ (* 28 (+ (* sel_r 2) iv_i)) (+ (* sel_c 2) iv_j))))
              (set! iv_w (get_item W1 (+ (* sel_k 25) (+ (* iv_i 5) iv_j))))
              (set! iv_p (* iv_v iv_w))
              (set! iv_sum (+ iv_sum iv_p))
              (draw_ink iv_gx iv_gy RI_S iv_v)
              (draw_wt (+ iv_gx 34) iv_gy RI_S (/ iv_w w1max))
              (draw_wt (+ iv_gx 68) iv_gy RI_S (/ iv_p iv_pmax))
              (set! iv_j (+ iv_j 1))
            })
            (set! iv_i (+ iv_i 1))
          })
          (set! iv_act (if (> iv_sum 0.0) iv_sum 0.0))
          (Gfx.draw_text "入" RI_X (+ RI_Y 32) 120 126 140)
          (Gfx.draw_text "核" RJ_X (+ RI_Y 32) 120 126 140)
          (Gfx.draw_text "积" RK_X (+ RI_Y 32) 120 126 140)
          ;; 结果文字：所选的核与格、加权和（未归一化的真值）、ReLU 后的激活
          (Gfx.draw_text (String.concat "核" (i2s sel_k)) RI_TX (- RI_Y 2) 90 96 115)
          (Gfx.draw_text (String.concat "和 " (fmt2 iv_sum)) RI_TX (+ RI_Y 12) 40 44 60)
          (Gfx.draw_text (String.concat "激活 " (fmt2 iv_act)) RI_TX (+ RI_Y 26) 195 70 35)
        })
  }))

;; ---- 右栏第二层：卷积2 的特征图 + 6 个输入通道的贡献 ------------------

(define dm2_f 0)
(define dm2_i 0)
(define dm2_j 0)
(define dm2_v 0.0)
(define draw_maps2
  (lambda () {
    (set! dm2_f 0)
    (while (< dm2_f 8) {
      (set! map_x (+ RX (* (mod dm2_f 4) C2_P)))
      (set! map_y (+ C2_Y0 (if (< dm2_f 4) 0 (- C2_Y1 C2_Y0))))
      (set! dm2_i 0)
      (while (< dm2_i 4) {
        (set! dm2_j 0)
        (while (< dm2_j 4) {
          (set! dm2_v (get_item c2 (+ (* dm2_f 16) (+ (* dm2_i 4) dm2_j))))
          (draw_act (+ map_x (* dm2_j C2_S)) (+ map_y (* dm2_i C2_S)) C2_S
                    (/ dm2_v (get_item mx2 dm2_f)))
          (set! dm2_j (+ dm2_j 1))
        })
        (set! dm2_i (+ dm2_i 1))
      })
      (Gfx.draw_rect (- map_x 2) (- map_y 2) (+ (* 4 C2_S) 4) (+ (* 4 C2_S) 4) 205 210 220)
      (if (and (== view 1) (== dm2_f sel2_f))
          (Gfx.draw_rect (- (+ map_x (* sel2_c C2_S)) 1) (- (+ map_y (* sel2_r C2_S)) 1)
                         (+ C2_S 2) (+ C2_S 2) 25 95 200)
          #f)
      (set! dm2_f (+ dm2_f 1))
    })
  }))

;; 选中卷积2 神经元时：算 6 个输入通道各自的 Σ(3x3 输入块 × 核) 与总和
(define cb_ch 0)
(define cb_ky 0)
(define cb_kx 0)
(define cb_s 0.0)
(define cb_v 0.0)
(define cb_w 0.0)
(define cb_max 0.000001)
(define cb_sum 0.0)
(define cb_act 0.0)
(define compute_contrib
  (lambda () {
    (set! cb_max 0.000001)
    (set! cb_ch 0)
    (while (< cb_ch 6) {
      (set! cb_s 0.0)
      (set! cb_ky 0)
      (while (< cb_ky 3) {
        (set! cb_kx 0)
        (while (< cb_kx 3) {
          (set! cb_v (get_item p1 (+ (* cb_ch 36) (+ (* (+ sel2_r cb_ky) 6) (+ sel2_c cb_kx)))))
          (set! cb_w (get_item W2 (+ (* sel2_f 54) (+ (* cb_ch 9) (+ (* cb_ky 3) cb_kx)))))
          (set! cb_s (+ cb_s (* cb_v cb_w)))
          (set! cb_kx (+ cb_kx 1))
        })
        (set! cb_ky (+ cb_ky 1))
      })
      (set_item! contrib cb_ch cb_s)
      (set! cb_max (maxf cb_max (Math.abs cb_s)))
      (set! cb_ch (+ cb_ch 1))
    })
    (set! cb_sum (get_item B2 sel2_f))
    (set! cb_ch 0)
    (while (< cb_ch 6) {
      (set! cb_sum (+ cb_sum (get_item contrib cb_ch)))
      (set! cb_ch (+ cb_ch 1))
    })
    (set! cb_act (if (> cb_sum 0.0) cb_sum 0.0))
  }))

(define cb_n 0)
(define cb_w2 0)
(define cb_y 0)
(define draw_contrib
  (lambda () {
    (if (== sel2_on 0)
        (Gfx.draw_text "点按特征图某格：看 6 通道如何相加" RX (+ C2_Y1 56) 135 140 155)
        {
          (compute_contrib)
          (set! cb_n 0)
          (while (< cb_n 6) {
            (set! cb_y (+ 134 (* cb_n 14)))
            (Gfx.draw_text (String.concat "c" (i2s cb_n)) RX cb_y 110 118 135)
            ;; 以零点为中心的左右条（右=正贡献，左=负贡献）
            (set! cb_w2 (Math.round (* CB_HALF (/ (get_item contrib cb_n) cb_max))))
            (if (>= cb_w2 0)
                (if (> cb_w2 0) (Gfx.fill_rect CB_ZERO cb_y cb_w2 10 220 90 60) #f)
                (Gfx.fill_rect (+ CB_ZERO cb_w2) cb_y (- 0 cb_w2) 10 70 110 215))
            (Gfx.draw_line CB_ZERO (- cb_y 1) CB_ZERO (+ cb_y 11) 190 195 205)
            (Gfx.draw_text (fmt2 (get_item contrib cb_n)) 256 cb_y 60 66 90)
            (set! cb_n (+ cb_n 1))
          })
          (Gfx.draw_text (String.concat "和 " (String.concat (fmt2 cb_sum)
                                 (String.concat "  激活 " (fmt2 cb_act))))
                         RX 222 40 44 60)
        })
  }))

(define draw_right
  (lambda () {
    (if (== view 0)
        {
          (Gfx.draw_text "卷积1：6×5×5 步长2" RX 18 55 62 88)
          (draw_kernels)
          (draw_maps)
          (Gfx.draw_text "输入块 × 卷积核 → 和" RX 175 55 62 88)
          (draw_inspector)
        }
        {
          (Gfx.draw_text "卷积2：8×3×3×6 核" RX 18 55 62 88)
          (draw_maps2)
          (Gfx.draw_text "通道贡献：3×3 块 × 核" RX 118 55 62 88)
          (draw_contrib)
        })
  }))

;; ---- 顶部栏 -------------------------------------------------------------

(define BTN_H 16)
(define CLR_X 110)        ; 「清空」按钮
(define CLR_W 44)

(define draw_btn
  (lambda (x w label) {
    (Gfx.fill_rect x 2 w BTN_H 226 229 236)
    (Gfx.draw_rect x 2 w BTN_H 170 176 190)
    (Gfx.draw_text label (+ x (Math.round (/ (- w (tw label)) 2.0))) 4 45 50 70)
  }))

(define LAY_X 158)         ; 「层1/层2」切换按钮
(define LAY_W 40)

(define draw_header
  (lambda () {
    (Gfx.fill_rect 0 0 SW 18 248 249 252)
    (Gfx.draw_text "CNN 手写数字识别" 4 3 40 44 62)
    (draw_btn CLR_X CLR_W "清空")
    (if (== view 0) (draw_btn LAY_X LAY_W "层2") (draw_btn LAY_X LAY_W "层1"))
    (Gfx.draw_text (String.concat (String.concat "推理 " (i2s infer_ms)) "ms")
                   204 4 150 155 172)
  }))

;; ---- 整帧渲染 -----------------------------------------------------------

(define render
  (lambda () {
    (Gfx.fill_white)
    (draw_header)
    (draw_pad)
    (Gfx.draw_rect (- PAD_X 2) (- PAD_Y 2) (+ PAD_W 4) (+ PAD_H 4) 205 210 220)
    (draw_receptive)
    (draw_pred)
    (draw_bars)
    (draw_right)
    (Gfx.draw_text "MNIST 98.1%" PAD_X 224 175 182 196)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; 只重画绘图板 + 感受野（拖笔时的高频路径，约 800 次 fill_rect）
(define render_pad_only
  (lambda () {
    (draw_pad)
    (draw_receptive)
    (Gfx.refresh)
  }))

;; ---- 笔刷：往 28x28 网格上盖章 -----------------------------------------

(define st_i 0)
(define st_j 0)
(define st_d 0.0)
(define st_v 0.0)
(define st_i0 0)
(define st_i1 0)
(define st_j0 0)
(define st_j1 0)
(define stamp
  (lambda (gx gy) {
    (set! st_i0 (Math.floor (- gy (+ BRUSH_R 0.5))))
    (set! st_i1 (Math.ceil (+ gy (+ BRUSH_R 0.5))))
    (set! st_j0 (Math.floor (- gx (+ BRUSH_R 0.5))))
    (set! st_j1 (Math.ceil (+ gx (+ BRUSH_R 0.5))))
    (if (< st_i0 0) (set! st_i0 0))
    (if (< st_j0 0) (set! st_j0 0))
    (if (> st_i1 27) (set! st_i1 27))
    (if (> st_j1 27) (set! st_j1 27))
    (set! st_i st_i0)
    (while (<= st_i st_i1) {
      (set! st_j st_j0)
      (while (<= st_j st_j1) {
        (set! st_d (Math.sqrt (+ (* (- (+ st_j 0.5) gx) (- (+ st_j 0.5) gx))
                                 (* (- (+ st_i 0.5) gy) (- (+ st_i 0.5) gy)))))
        (if (< st_d BRUSH_R) {
          (set! st_v (- 1.0 (/ st_d BRUSH_R)))
          (if (> st_v (get_item img (+ (* st_i 28) st_j)))
              (set_item! img (+ (* st_i 28) st_j) st_v))
          (set! has_ink 1)
        })
        (set! st_j (+ st_j 1))
      })
      (set! st_i (+ st_i 1))
    })
  }))

;; 从上一采样点到当前点的连线盖章（人手快速移动也不断笔）
(define ln_n 0)
(define ln_i 0)
(define ln_dx 0.0)
(define ln_dy 0.0)
(define stroke_to
  (lambda (gx gy) {
    (set! ln_dx (- gx pen_last_x))
    (set! ln_dy (- gy pen_last_y))
    (set! ln_n (Math.ceil (/ (Math.sqrt (+ (* ln_dx ln_dx) (* ln_dy ln_dy))) PEN_STEP)))
    (if (< ln_n 1) (set! ln_n 1))
    (set! ln_i 1)
    (while (<= ln_i ln_n) {
      (stamp (+ pen_last_x (/ (* ln_dx ln_i) ln_n))
             (+ pen_last_y (/ (* ln_dy ln_i) ln_n)))
      (set! ln_i (+ ln_i 1))
    })
    (set! pen_last_x gx)
    (set! pen_last_y gy)
  }))

(define pad_hit
  (lambda (px py)
    (if (>= px PAD_X)
        (if (< px (+ PAD_X PAD_W))
            (if (>= py PAD_Y)
                (if (< py (+ PAD_Y PAD_H)) 1 0)
                0)
            0)
        0)))

;; ---- 右栏点按：选中某张特征图的某一格 -----------------------------------

(define pm_k 0)
(define pm_c 0)
(define pm_r 0)
(define pm_mx 0)
(define pm_my 0)
(define pick_map
  (lambda (tx ty) {
    (set! pm_k -1)
    (if (and (>= tx RX) (< tx (+ RX (+ (* 3 COL_P) (* 12 MP_S)))))
        {
          (if (and (>= ty MG_Y0) (< ty (+ MG_Y0 (* 12 MP_S))))
              (set! pm_k (Math.floor (/ (- tx RX) COL_P)))
              (if (and (>= ty MG_Y1) (< ty (+ MG_Y1 (* 12 MP_S))))
                  (set! pm_k (+ 3 (Math.floor (/ (- tx RX) COL_P))))
                  #f))
          (if (>= pm_k 0)
              {
                (set! pm_mx (+ RX (* (mod pm_k 3) COL_P)))
                (set! pm_my (map_oy pm_k))
                (set! pm_c (Math.floor (/ (- tx pm_mx) MP_S)))
                (set! pm_r (Math.floor (/ (- ty pm_my) MP_S)))
                (if (and (>= pm_c 0) (< pm_c 12))
                    #f
                    (set! pm_k -1))
                (if (and (>= pm_r 0) (< pm_r 12))
                    #f
                    (set! pm_k -1))
                (if (>= pm_k 0)
                    {
                      (set! sel_k pm_k)
                      (set! sel_r pm_r)
                      (set! sel_c pm_c)
                      (set! sel_on 1)
                      (render)
                    })
              })
        })
  }))

;; 第二层视图的点按：命中卷积2 的某张 4x4 特征图的某一格
(define pq_f 0)
(define pq_c 0)
(define pq_r 0)
(define pick_map2
  (lambda (tx ty) {
    (set! pq_f -1)
    (if (and (>= tx RX) (< tx (+ RX (+ (* 3 C2_P) (* 4 C2_S)))))
        {
          (if (and (>= ty C2_Y0) (< ty (+ C2_Y0 (* 4 C2_S))))
              (set! pq_f (Math.floor (/ (- tx RX) C2_P)))
              (if (and (>= ty C2_Y1) (< ty (+ C2_Y1 (* 4 C2_S))))
                  (set! pq_f (+ 4 (Math.floor (/ (- tx RX) C2_P))))
                  #f))
          (if (>= pq_f 0)
              {
                (set! map_x (+ RX (* (mod pq_f 4) C2_P)))
                (set! map_y (+ C2_Y0 (if (< pq_f 4) 0 (- C2_Y1 C2_Y0))))
                (set! pq_c (Math.floor (/ (- tx map_x) C2_S)))
                (set! pq_r (Math.floor (/ (- ty map_y) C2_S)))
                (if (and (>= pq_c 0) (< pq_c 4)) #f (set! pq_f -1))
                (if (and (>= pq_r 0) (< pq_r 4)) #f (set! pq_f -1))
                (if (>= pq_f 0)
                    {
                      (set! sel2_f pq_f)
                      (set! sel2_r pq_r)
                      (set! sel2_c pq_c)
                      (set! sel2_on 1)
                      (render)
                    })
              })
        })
  }))

;; ---- 清空 -------------------------------------------------------------

(define clr_i 0)
(define clear_all
  (lambda () {
    (set! clr_i 0)
    (while (< clr_i 784) {
      (set_item! img clr_i 0.0)
      (set! clr_i (+ clr_i 1))
    })
    (set! has_ink 0)
    (set! pred_idx -1)
    (set! pred_conf 0.0)
    (set! sel_on 0)
    (set! sel2_on 0)
    (set! clr_i 0)
    (while (< clr_i 10) {
      (set_item! pb clr_i 0.0)
      (set_item! lg clr_i 0.0)
      (set! clr_i (+ clr_i 1))
    })
    ;; 各层激活也要清掉：否则清空后右栏还留着上一次的特征图（看着像"识别过了"）
    (set! clr_i 0)
    (while (< clr_i 864) {
      (set_item! c1 clr_i 0.0)
      (set! clr_i (+ clr_i 1))
    })
    (set! clr_i 0)
    (while (< clr_i 216) {
      (set_item! p1 clr_i 0.0)
      (set! clr_i (+ clr_i 1))
    })
    (set! clr_i 0)
    (while (< clr_i 128) {
      (set_item! c2 clr_i 0.0)
      (set! clr_i (+ clr_i 1))
    })
    (set! clr_i 0)
    (while (< clr_i 6) {
      (set_item! mx1 clr_i 1.0)
      (set! clr_i (+ clr_i 1))
    })
    (set! clr_i 0)
    (while (< clr_i 8) {
      (set_item! mx2 clr_i 1.0)
      (set! clr_i (+ clr_i 1))
    })
  }))

;; ---- 每拍 --------------------------------------------------------------

(define tp 0)
(define tr 0)
(define tx 0)
(define ty 0)
(define gx 0.0)
(define gy 0.0)

;; 顶部按钮命中：1 = 清空，2 = 层切换，0 = 都不是
(define clr_hit
  (lambda (x y)
    (if (and (>= y 2) (< y (+ 2 BTN_H)))
        (if (and (>= x CLR_X) (< x (+ CLR_X CLR_W))) 1
            (if (and (>= x LAY_X) (< x (+ LAY_X LAY_W))) 2 0))
        0)))

(define tick
  (lambda () {
    (EX.check)
    (set! tp (if (Touch.take_press) 1 0))
    (set! tr (if (Touch.take_release) 1 0))
    (set! tx (Touch.x))
    (set! ty (Touch.y))

    ;; 按下沿：顶部按钮 / 特征图选中 / 绘图板起笔
    (if (== tp 1) {
      (set! armed (clr_hit tx ty))
      (if (== armed 0)
          (if (== (pad_hit tx ty) 1) {
            (set! pen_down 1)
            (set! pen_last_x (/ (- tx PAD_X) PAD_CELL))
            (set! pen_last_y (/ (- ty PAD_Y) PAD_CELL))
            (stamp pen_last_x pen_last_y)
            (render)
          }
          (if (== view 0) (pick_map tx ty) (pick_map2 tx ty)))
          #f)
    })

    ;; 按住：继续送笔（电平模型，每拍采样一次并补上两点之间的连线）
    ;; 手指滑出绘图板时只在板内落笔，但仍更新「上一采样点」，回到板内不会连出一条跨屏直线
    (if (== pen_down 1)
        (if (Touch.is_pressed)
            {
              (set! gx (/ (- tx PAD_X) PAD_CELL))
              (set! gy (/ (- ty PAD_Y) PAD_CELL))
              (if (< gx 0.0) (set! gx 0.0))
              (if (< gy 0.0) (set! gy 0.0))
              (if (> gx 28.0) (set! gx 28.0))
              (if (> gy 28.0) (set! gy 28.0))
              (if (== (pad_hit tx ty) 1)
                  {
                    (stroke_to gx gy)
                    (render_pad_only)
                  }
                  {
                    (set! pen_last_x gx)
                    (set! pen_last_y gy)
                  })
            }
            #f)
        #f)

    ;; 松开沿：结算顶部按钮；否则若刚画完 → 触发一次推理
    (if (== tr 1) {
      (if (> armed 0)
          (if (== (clr_hit tx ty) armed)
              (if (== armed 1)
                  {
                    (clear_all)
                    (set! stale 1)
                  }
                  {
                    (set! view (if (== view 0) 1 0))
                    (render)
                  })
              #f)
          #f)
      (if (== pen_down 1) (set! stale 1) #f)
      (set! armed 0)
      (set! pen_down 0)
    })

    ;; 推理（只在松手后一次；约 60~90ms，期间画「计算中」并先上屏）
    (if (== stale 1) {
      (set! stale 0)
      (if (== has_ink 1) {
        (set! computing 1)
        (draw_pred)
        (EX.draw_button)
        (Gfx.refresh)
        (run_net)
        (set! computing 0)
        (render)
      }
      (render))
    })
  }))

;; ---- 启动 --------------------------------------------------------------

(set! w1max (wmax_scan W1 150))
;; 特征图着色归一化分母先置 1（避免首次渲染时除零——此时还没有任何激活值）
(set! clr_i 0)
(while (< clr_i 6) {
  (set_item! mx1 clr_i 1.0)
  (set! clr_i (+ clr_i 1))
})
(set! clr_i 0)
(while (< clr_i 8) {
  (set_item! mx2 clr_i 1.0)
  (set! clr_i (+ clr_i 1))
})
(render)
(set! _timer (System.set_interval TICK_MS tick))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
(EX.install_with render)