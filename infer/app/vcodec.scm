;@id=vcodec
;@title=视频编码
;@version=2026-11-17
;; ============================================================================
;; vcodec.scm —— 视频编码教学演示（灵机 Scheme App / Nano-WASM-GUI 目标）
;;
;; 一、这是什么
;;   MPEG-1 / H.261 风格的**迷你视频编码器**原型 + 可视化 + 解码播放器，输入是
;;   app/lib/badapple/frames.txt 里的 64×64 **二值**帧序列（10fps，2193 帧）。
;;   屏幕被分成 6 个「面板」（触屏点顶部标签页 / 键盘 1..6 切换），把一条完整
;;   的编码流水线拆开逐级展示：
;;
;;     ① 总览   编码进度、每帧码率柱状图、帧类型（I/P）、PSNR、块模式着色
;;     ② 帧内   帧内预测（DC / 垂直 / 水平 / 平面）的预测块、残差块、重建块与代价
;;     ③ 帧间   运动估计：搜索窗 SAD 地形、运动矢量场、运动补偿预测块
;;     ④ 变换   8×8 DCT 系数热图、量化电平、之字形扫描顺序、游程（RLE）码字
;;     ⑤ RDO    候选模式的率失真代价（D + λR）对比、全帧 RD 散点、λ/Q 交互
;;     ⑥ 解码   把码流喂给**解码器**逐帧重建并 10fps 播放，与源帧并排对比
;;
;;   交互闭环：编码（可暂停）→ 任意面板观察 → 改 λ/Q 立即重编 → 解码播放。
;;
;; 二、码流语法（简化但可完整解码；比特按 MSB 优先、每帧字节对齐）
;;   帧头： ftype(1) qscale(6)                  ftype 0=I 帧 1=P 帧
;;   I 帧每块： imode(2) [DC/垂直/水平/平面] 残差
;;   P 帧每块： coded(1)；coded=0 → SKIP（MV=(0,0) 直接拷贝参考块，无残差）
;;                      coded=1 → bmode(1) 0=帧内/1=帧间
;;                                 帧内：imode(2) + 残差
;;                                 帧间：mvx(4) mvy(4) [值 = MV+4] + 残差
;;   残差：   nz(6) + nz×(run(6) level(5) sign(1))     —— 之字形扫描后的一维游程
;;
;;   为什么用「定长码 + 游程」而不是 VLC/算术编码：本 App 是教学演示，定长码让
;;   「一个非零系数 = 12 bit」这件事在界面上直接可数，RDO 的 R（比特数）也就
;;   能手算复核。真实编码器会用 Huffman/算术编码把它压到 2~5 bit（这正是
;;   H.261/MPEG-1 的 VLC 表在做的事），那属于「熵编码」一课的续集。
;;
;; 三、编码工具（每一项都可在界面上看到）
;;   · 帧内预测：从**已重建**的上邻行 / 左邻列派生预测块（H.264 式 4 种模式）；
;;   · 帧间预测：8×8 块、±4 像素运动搜索（两阶段：隔点采样粗搜 + 全像素细化），
;;     半像素运动、双向预测、B 帧等省略（保持概念完整而不臃肿）；
;;   · 关键帧（I 帧）识别：第 0 帧强制 I；每 GOP=10 帧强制 I；此外按
;;     「与参考帧差异像素数 > 阈值」触发场景切换 I 帧（自适应关键帧）；
;;   · 变换编码：8×8 **正交归一化 DCT-II**，可分离实现（先行后列）。
;;     为什么二值图像也用 DCT：DCT 的能量集中性对「大片同色 + 边缘」这类
;;     二值线稿同样成立——平坦块的残差全零（一个系数都不用发），含边缘的块
;;     能量集中在 DC 与少数低频 AC 上，量化后大量高频直接归零；而 DCT 是正交
;;     变换，**变换域误差平方和 = 像素域 SSE**（Parseval），于是 RDO 的 D 不必
;;     回头做 IDCT 就能算出来（本 App 正是这么做的，省掉一半变换开销）；
;;   · 量化：均匀量化 + 死区，步长 Q 可调（面板⑤ Q± / 键盘 , .）；
;;   · 游程编码：之字形扫描 → (run, level) 序列 → 定长码；
;;   · RDO（率失真优化）：对每个块枚举候选编码方式，逐一算
;;         J = D + λ·R      D = 重建 SSE，R = 该候选的比特数
;;     取 J 最小者。λ 可调（面板⑤ λ± / 键盘 [ ]）——λ 大 → 省钱（码率低、
;;     失真高），λ 小 → 保真；这就是「率失真曲线的斜率」在交互界面上的直感。
;;
;;   编码器内部**闭环**：重建帧既是最优模式的判定依据，也是下一帧的参考帧；
;;   解码器走完全相同的重建路径，因此「码流回环校验」应当逐块一致（面板⑥显示）。
;;
;; 四、性能设计（wasm32 解释器约束，见 doc/AGENTS.md 的踩坑清单）
;;   · 一次「取两数相乘再累加」在解释器里约 30~40 条 VM 指令（实测口径），
;;     所以 8×8 二维 DCT 的 1024 次乘加 ≈ 4 万条指令：**必须靠稀疏性省**。
;;     两处关键优化：
;;       (a) 残差块全零直接跳过 DCT/IDCT（二值线稿里大片背景的残差恰好全零）；
;;       (b) 先行后列时逐行/逐列检查是否全零，空行整行输出 0。
;;   · 编码是**分块推进**的状态机（每拍 ENC_CHUNK 块）：既让进度可见，也不会
;;     让单拍超过宿主时间片（App.set_timeslice）。
;;   · 列表是数组（get_item/set_item! O(1)），帧缓冲一律用扁平列表；
;;     但 `'()` 字面量是**模块级静态单例**（循环里连续 (set! x '()) 会全部指向
;;     同一个对象！）——空表一律用 (nil) = (cdr '(0)) 新建，这是本文件最重要的
;;     一条纪律（实测见 doc/AGENTS.md）。
;;   · 浮点是「立即数」：wasm32 上有效尾数只有约 19 位（float32 再截掉低 5 位）。
;;     因此：不用浮点累计大数、不拿浮点做位运算、打印小数一律走定点格式化
;;     （fnum），别指望 atom_to_string 给出漂亮的十进制。
;;
;; 五、文件结构
;;   §1 常量   §2 小工具   §3 查找表   §4 缓冲   §5 比特读写   §6 帧数据
;;   §7 编解码核心   §8 编码器   §9 解码器   §10 选中块分析
;;   §11 绘制基元   §12 六个面板   §13 输入   §14 主循环
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Fs)
(native Touch)
(native Key)
(native App)
(native Diag)          ;; (Diag.ms) 单调毫秒：节拍/耗时测量用它（System.timestamp 做区间测量不可靠）

(import BA "app/lib/badapple/meta.scm")
(import EX "app/lib/exit.scm")

;; ============================================================================
;; §1 常量与调参
;; ============================================================================

;; ---- 图像与块 ----
(define W 64)                   ;; 帧宽
(define H 64)                   ;; 帧高
(define BS 8)                   ;; 变换/预测块边长
(define BW 8)                   ;; 横向块数
(define NB 64)                  ;; 每帧块数
(define NPIX 4096)              ;; 每帧像素数

;; ---- 输入数据（定长记录格式见 tools/gen_badapple_data.py）----
(define PATH "app/lib/badapple/frames.txt")
(define REC BA.rec_bytes)
(define CH0 BA.base_char)       ;; 记录字符 → 数值：(- c CH0)
(define RADIX BA.base_radix)    ;; 段数头 c0*RADIX + c1
(define HEAD BA.head_chars)
(define CPR BA.chars_per_run)
(define NF BA.n_frames)

;; ---- 可调参数 ----
(define NFR 20)                 ;; 编码帧数（20 帧 = 2.0 秒：含 2 个 I 帧 + 18 个 P 帧）
(define START_F 600)            ;; 起始帧号（片源内偏移）
(define GOP 10)                 ;; 关键帧周期
(define LAMBDA 30.0)            ;; 拉格朗日乘子 λ（0..255 像素域的 SSE）
(define QSTEP 12)               ;; 量化步长 Q（0..255 像素域，16 ≈ 6% 阶距）
(define QMIN 2)
(define QMAX 48)
(define SR 4)                   ;; 运动搜索半径 ±4
(define SCENE_TH 1200)          ;; 场景切换阈值（相邻源帧差异像素数 > 该值 → 强制 I 帧）
;; 每拍编码块数。一拍 = 「ENC_CHUNK 块编码 + 一次整屏绘制」，取 4 是为了让**每一个中间
;; 状态都真的上屏**：光标每拍前进半个块行，视觉上是连续扫描而不是跳帧。
;; 代价是总时长（宁可慢，也不要丢中间状态）。
(define ENC_CHUNK 4)            ;; 每拍编码块数
(define DEC_CHUNK 16)           ;; 每拍解码块数
;; 界面节拍（毫秒）。取 1 = 「宿主一帧里能干多少就干多少」，一拍内的编码块数由 ENC_CHUNK 定。
;;
;; 定时器为什么用 set_interval、以及为什么还需要 Scheme 侧的重入守卫
;; （三条实测：同一份 12 帧工作量，headless chromium）
;;   set_interval(30)                 → 7.8 s
;;   set_interval(1)                  → 4.5 s   ← 采用
;;   set_timeout(1) 自链（一拍一登记）→ 36.4 s   ← 慢 8 倍，而且越跑越慢
;; 一次性定时器链为什么慢：每拍都新登记一个定时器、回调一结束就到期，宿主被无空隙占满，
;; 帧率塌到个位数，而「一拍墙上耗时」里就含了等下一帧的时间 —— 于是越往后越慢
;; （实测刷新率 5/s → 1/s、编码 5 s → 36 s）。周期定时器不会这样。
;;
;; 但周期定时器在「一拍耗时 > 一个时间片」时会被**重入**：上游运行时的防护靠
;; 「函数栈上每隔一个元素出现 ASYNC_RET」判定，回调里嵌套普通 Scheme 调用时该交替关系
;; 不成立 → 防护漏判 → 同一个 tick 被并发执行两份（实测：enc_frame_start 进两次、第二个
;; 帧头写进码流中间 → 解码失步、mismatch 上万）。所以本 App 在 Scheme 侧加**重入守卫**
;; （见 tick 开头）：重入的那一份立刻返回，外层继续，状态不被踩。两者合起来既快又正确。
(define TICK_MS 1)
(define _in_tick 0)             ;; 重入守卫：>0 表示上一拍还没跑完

;; ---- 候选编码方式号（与码流语法一一对应）----
(define C_SKIP 5)               ;; 跳过：MV=0 拷贝参考块，不编码残差
(define C_INTER 4)              ;; 帧间：带运动矢量 + 残差

;; ---- 码流字段宽度 ----
(define B_IMODE 2)
(define B_MV 4)
(define B_NZ 7)
(define B_RUN 6)               ;; 仅用于上限说明（实际用变长码）
(define B_LEV 8)               ;; 电平幅值 8 bit（DC 可达 ±2040/Q）
(define B_LEV_BITS 8)           ;; 电平幅值上限（8 bit，够 ±255）
(define B_INTER_HDR 10)         ;; 帧间块头：coded(1)+bmode(1)+mvx(4)+mvy(4)

;; ---- 阶段 ----
(define PH_LOAD 0)
(define PH_ENC 1)
(define PH_DEC 2)
(define PH_READY 3)

;; ---- 屏幕布局 ----
(define SW 320)
(define SH 240)
(define TAB_H 15)
(define BY 221)
(define BH 17)

;; ---- 全局临时量（必须在下面所有表构建之前定义好）----
(define _i 0)
(define _j 0)
(define _k 0)
(define _n 0)
(define _x 0)
(define _u 0)
(define _v 0)
(define _cu 0.0)
(define _f 0.0)
(define _imi 0)
(define _ihdr 0)

;; 面板名
(define PANEL_NAMES '("总览" "帧内" "帧间" "变换" "RDO" "解码"))
;; 底部按钮标签（每面板 4 枚）
(define BTN_LABELS '(("编码" "重编" "Q-" "Q+")
                     ("帧-" "帧+" "块-" "块+")
                     ("帧-" "帧+" "块-" "块+")
                     ("帧-" "帧+" "块-" "块+")
                     ("λ-" "λ+" "Q-" "Q+")
                     ("播放" "帧-" "帧+" "重播")))

;; ============================================================================
;; §2 小工具
;; ============================================================================

;; 新建空表。`'()` 是模块级静态单例（循环里连用会指向同一对象），凡是
;; 「每次都要一个新表」的场合一律走 (nil)：(cdr '(0)) 返回拷贝，必新。
(define nil (lambda () (cdr '(0))))

;; 建一个 n 项的填充表
(define mkfilli
  (lambda (n) {
    (define out (nil))
    (define i 0)
    (while (< i n) { (push out 0) (set! i (+ i 1)) })
    out }))
(define mkfillf
  (lambda (n) {
    (define out (nil))
    (define i 0)
    (while (< i n) { (push out 0.0) (set! i (+ i 1)) })
    out }))

(define clampf (lambda (v lo hi) (if (< v lo) lo (if (> v hi) hi v))))
(define clampi (lambda (v lo hi) (if (< v lo) lo (if (> v hi) hi (Math.round v)))))
;; 坐标 → 格子下标必须**向下取整**：用 clampi（四舍五入）会把命中区从
;; 「[i·格, (i+1)·格)」缩成「[(i-0.5)·格, (i+0.5)·格)」，点在格子左半边会命中右邻格
;; ——实测「点按位置与触发位置不一致、整体偏右」即此（标签页/按钮/块/时间轴同病）。
(define celli (lambda (v lo hi) (if (< v lo) lo (if (> v hi) hi (Math.floor v)))))
(define maxf (lambda (a b) (if (> a b) a b)))
(define minf (lambda (a b) (if (< a b) a b)))
(define absf (lambda (a) (if (< a 0) (- 0 a) a)))

;; 整数格式化。注意 %g（String.atom_to_string 的实现）在 ≥1e6 时会输出
;; "1.158e+06" 这种指数形式——界面上的像素域失真很容易上百万，所以这里自己拼：
;; 分「百万以上」与「以下」两段，避免指数记号也避免浮点精度损失。
(define zpadn
  (lambda (v n) {
    (define s (String.atom_to_string (Math.round v)))
    (while (< (String.length s) n) (set! s (String.concat "0" s)))
    s
  }))
(define inum
  (lambda (v) {
    (define m (Math.round v))
    (define sgn "")
    (define hi 0)
    (define lo 0)
    (if (< m 0) { (set! sgn "-") (set! m (- 0 m)) } #f)
    (if (< m 1000000)
        (String.concat sgn (String.atom_to_string m))
        {
          (set! hi (Math.floor (/ m 1000000)))
          (set! lo (- m (* hi 1000000)))
          (String.concat sgn (String.concat (String.atom_to_string hi)
                           (String.concat "." (zpadn lo 6))))
        })
  }))

;; 定点小数格式化：v 保留 d 位（d = 1 或 2）。
;; 不能用 atom_to_string —— wasm32 浮点尾数只有约 19 位，21.3 会印成 21.299999。
(define fnum
  (lambda (v d) {
    (define sc (if (>= d 2) 100 10))
    (define m0 (Math.round (* v sc)))
    (define m (if (== m0 0) 0 m0))
    (define sgn (if (< m 0) "-" ""))
    (define a (if (< m 0) (- 0 m) m))
    (define ip (Math.floor (/ a sc)))
    (define fp (- a (* ip sc)))
    (define fs (inum fp))
    (String.concat sgn
      (String.concat (inum ip)
        (String.concat "." (if (< fp 10) (String.concat (if (>= d 2) "0" "") fs) fs))))
  }))

;; 零填充整数（"012"）
(define znum
  (lambda (v n) (zpadn v n)))

;; 万以上用 k 后缀（面板上的失真/代价动辄上万，7 位数字会挤成一团）
(define knum
  (lambda (v) {
    (define m (Math.round v))
    (if (< m 10000) (inum m) (String.concat (fnum (/ m 1000.0) 1) "k"))
  }))

;; 12px 字模下字符串宽度（半角 6px / 全角 12px）
(define text_w
  (lambda (s) {
    (define i 0)
    (define w 0)
    (define n (String.length s))
    (while (< i n) {
      (set! w (+ w (if (< (String.charCodeAt i s) 128) 6 12)))
      (set! i (+ i 1))
    })
    w
  }))

;; 常用文字颜色（12px 抗锯齿字模）
(define tn (lambda (s x y) (Gfx.draw_text s x y 226 232 242)))   ;; 正常
(define td (lambda (s x y) (Gfx.draw_text s x y 130 138 154)))   ;; 次要
(define ta (lambda (s x y) (Gfx.draw_text s x y 116 172 226)))   ;; 蓝
(define tg (lambda (s x y) (Gfx.draw_text s x y 110 214 150)))   ;; 绿
(define ty (lambda (s x y) (Gfx.draw_text s x y 245 216 96)))    ;; 黄
(define tor (lambda (s x y) (Gfx.draw_text s x y 240 170 100)))  ;; 橙
(define tcy (lambda (s x y) (Gfx.draw_text s x y 120 200 235)))  ;; 青

(define dtc
  (lambda (s cx y r g b) (Gfx.draw_text s (- cx (Math.floor (/ (text_w s) 2))) y r g b)))
(define rc (lambda (x y w h r g b) (Gfx.fill_rect x y w h r g b)))
(define frc (lambda (x y w h r g b) (Gfx.draw_rect x y w h r g b)))

;; 命中判定（返回 #t/#f——注意本语言里 0 是真值，不能返回 0/1）
(define hit
  (lambda (px py rx ry rw rh)
    (if (and (>= px rx) (< px (+ rx rw)) (>= py ry) (< py (+ ry rh))) #t #f)))

;; ============================================================================
;; §3 查找表
;; ============================================================================

;; 8×8 正交归一化 DCT-II 基矩阵，扁平展开：DCTB[u*8+x] = c(u)/2 · cos((2x+1)uπ/16)
(define DCTB (nil))
(set! _u 0)
(while (< _u 8) {
  (set! _cu (if (== _u 0) 0.35355339 0.5))
  (set! _x 0)
  (while (< _x 8) {
    (push DCTB (* _cu (Math.cos (/ (* (* (+ (* 2 _x) 1) _u) 3.14159265) 16))))
    (set! _x (+ _x 1))
  })
  (set! _u (+ _u 1))
})

;; 8 点 DCT 的偶部/奇部基表 —— 利用 B[u][x] = ±B[u][7-x] 的成对对称性
;; （u 偶取 +，u 奇取 -），一个 8 点变换只需 32 次乘加（而不是 64 次）：
;;   u 偶： C[u] = Σ_{x<4} TBE[u/2*4+x] · (r[x] + r[7-x])
;;   u 奇： C[u] = Σ_{x<4} TBO[(u-1)/2*4+x] · (r[x] - r[7-x])
;; 逆变换同理：把输出 y 与 7-y 成对求出（f[y]=P+Q、f[7-y]=P-Q），
;; 于是 TBE/TBO 同一张表双向通用（正变换按输入下标取、逆变换按输出下标取）。
(define TBE (nil))
(define TBO (nil))
(set! _j 0)
(while (< _j 4) {
  (set! _x 0)
  (while (< _x 4) {
    (push TBE (get_item DCTB (+ (* (* 2 _j) 8) _x)))
    (push TBO (get_item DCTB (+ (* (+ (* 2 _j) 1) 8) _x)))
    (set! _x (+ _x 1))
  })
  (set! _j (+ _j 1))
})

;; JPEG 标准之字形扫描顺序（下标 = y*8+x）：低频 → 高频
(define ZIG '(0 1 8 16 9 2 3 10 17 24 32 25 18 11 4 5 12 19 26 33 40 48 41 34
              27 20 13 6 7 14 21 28 35 42 49 56 57 50 43 36 29 22 15 23 30 37 44 51
              58 59 52 45 38 31 39 46 53 60 61 54 47 55 62 63))

;; 反查：之字形下标 → 扫描序号
(define ZIGINV (mkfilli 64))
(set! _i 0)
(while (< _i 64) { (set_item! ZIGINV (get_item ZIG _i) _i) (set! _i (+ _i 1)) })

;; 量化矩阵：每个系数位置的**相对量化步长**。低频细、高频粗——
;; 这正是 JPEG/MPEG 量化表的核心思想（人眼对高频误差不敏感，而高频系数
;; 数量多、每个都要花码字，粗量化它们对码率的收益最大）。
;; 实际步长 = Q × QW[i]，i = u*8+v，u/v 为行列频率下标。
(define QW (nil))
(set! _u 0)
(while (< _u 8) {
  (set! _v 0)
  (while (< _v 8) {
    (push QW (+ 1.0 (* 0.4 (+ _u _v))))
    (set! _v (+ _v 1))
  })
  (set! _u (+ _u 1))
})

;; 2 的幂（写比特用）
(define P2 '(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536))

;; 4 位 → "0101"；再由它拼出 256 项字节表
(define QUAD '("0000" "0001" "0010" "0011" "0100" "0101" "0110" "0111"
               "1000" "1001" "1010" "1011" "1100" "1101" "1110" "1111"))
(define B2S (nil))
(set! _v 0)
(while (< _v 256) {
  (push B2S (String.concat (get_item QUAD (Math.floor (/ _v 16)))
                           (get_item QUAD (mod _v 16))))
  (set! _v (+ _v 1))
})

;; ============================================================================
;; §4 缓冲
;; ============================================================================

;; ---- 帧缓冲（4096）----
(define src (mkfilli NPIX))     ;; 当前源帧（0/1，编码器用）
(define asrc (mkfilli NPIX))    ;; 分析/显示用源帧
(define psrc (mkfilli NPIX))    ;; 上一帧源（关键帧判定用）
(define ref (mkfillf NPIX))     ;; 参考帧
(define rec (mkfillf NPIX))     ;; 当前重建帧（编码器）
(define dfr (mkfillf NPIX))     ;; 解码器输出帧
(define afr (mkfillf NPIX))     ;; 分析用的重建目标（避免污染存档）
(define zf (mkfillf NPIX))      ;; 全零帧
(define lvb (mkfilli NPIX))     ;; 灰度等级缓冲（绘制）

;; ---- 块缓冲（64）----
(define sb (mkfilli 64))        ;; 源块（0/1）
(define pb (mkfillf 64))        ;; 预测块
(define rb (mkfillf 64))        ;; 残差块
(define cb (mkfillf 64))        ;; DCT 系数
(define qb (mkfilli 64))        ;; 量化电平
(define db (mkfillf 64))        ;; 反量化
(define ob (mkfillf 64))        ;; 重建残差
(define tb (mkfillf 64))        ;; 变换中间量
(define qsav (mkfilli 64))      ;; 量化电平备份（RDO 两档量化对比用）

;; ---- 候选表（6 个候选 × 64 系数）----
(define CAND_D (mkfillf 6))
(define CAND_R (mkfilli 6))
(define CAND_J (mkfillf 6))
(define CAND_N (mkfilli 6))
(define CAND_M (mkfilli 6))
(define CAND_C (mkfillf 384))
(define CAND_Q (mkfilli 384))
(define NCAND 0)

;; ---- 分析快照（选中块的详情）----
(define ANC (mkfillf 384))
(define ANQ (mkfilli 384))
(define AN_D (mkfillf 6))
(define AN_R (mkfilli 6))
(define AN_J (mkfillf 6))
(define AN_N (mkfilli 6))
(define AN_M (mkfilli 6))
(define AN_NC 0)
(define AN_BEST 0)
(define an_pred (mkfillf 64))
(define an_res (mkfillf 64))
(define an_recon (mkfillf 64))
(define an_coef (mkfillf 64))
(define an_deq (mkfillf 64))
(define an_q (mkfilli 64))
(define an_sad (mkfillf 81))
(define an_sadbest 0)
(define an_bestmvx0 0)
(define an_bestmvy0 0)
(define an_top (mkfillf 8))
(define an_left (mkfillf 8))
(define an_hastop 0)
(define an_hasleft 0)
(define an_ftype 0)
(define an_bits 0)
(define an_nz 0)
(define an_d 0.0)
(define an_j 0.0)
(define an_mvx 0)
(define an_mvy 0)

;; ---- 每帧统计 ----
(define fr_type (mkfilli NFR))
(define fr_q (mkfilli NFR))
(define fr_bits (mkfilli NFR))
(define fr_sse (mkfillf NFR))
(define fr_ni (mkfilli NFR))
(define fr_np (mkfilli NFR))
(define fr_ns (mkfilli NFR))
(define fr_err (mkfilli NFR))
(define fr_bd (nil))
(define fr_modes (nil))
(define fr_mvx (nil))
(define fr_mvy (nil))
(define fr_bnz (nil))
(define rec_frames (nil))
(define dec_frames (nil))
(define bs (nil))

;; 每帧的块级数组 + 帧缓冲：NFR 个**独立**列表（必须用 mkfill*，不能用 '()）
(set! _i 0)
(while (< _i NFR) {
  (set_item! fr_q _i QSTEP)
  (push fr_modes (mkfilli NB))
  (push fr_mvx (mkfilli NB))
  (push fr_mvy (mkfilli NB))
  (push fr_bnz (mkfilli NB))
  (push fr_bd (mkfillf NB))
  (push rec_frames (mkfillf NPIX))
  (push dec_frames (mkfillf NPIX))
  (push bs "")
  (set! _i (+ _i 1))
})

;; ---- 编解码器全局量 ----
(define g_bx 0)
(define g_by 0)
(define g_boff 0)
(define g_q 6)
(define g_d 0.0)
(define g_nz 0)
(define g_rc 0)         ;; 游程码字总位数（变长码）
(define g_zero 0)
(define g_mvx 0)
(define g_mvy 0)
(define g_best 0)
(define curbuf rec)             ;; 重建目标帧（也是预测的邻域来源）
(define reff zf)                ;; 参考帧
(define recbuf rec)             ;; 重建写入目标

(define rle_run (nil))
(define rle_lvl (nil))
(define bits_buf (nil))
(define frame_bitpos 0)
(define brs "")
(define brp 0)

;; ---- 编码 / 解码 / 播放状态 ----
(define phase PH_LOAD)
(define enc_f 0)
(define enc_b 0)
(define enc_ftype 0)
(define dec_f 0)
(define dec_b 0)
(define dec_ftype 0)
(define rt_mismatch 0)
(define rt_checked 0)
(define t_start 0)
(define enc_ms 0)

;; ---- 界面状态 ----
(define panel 0)
(define view_f 0)
(define sel_b 0)
(define need_analyze 1)
(define dirty 1)
(define playing 0)
(define play_t0 0)
(define play_idx 0)
(define enc_paused 0)
(define msg "")
(define msg_t 0)
(define _timer 0)
(define load_waits 0)
(define data_ok 0)
(define src_f -1)               ;; asrc 缓存的帧号
(define auto_sel 1)             ;; 1=自动选中当前帧失真最大的块（用户一点选就关掉）
(define dbg 0)

;; ============================================================================
;; §5 比特读写
;; ============================================================================

(define wbit (lambda (v) (push bits_buf v)))

;; 写 n bit（MSB 优先）
(define wval
  (lambda (v n) {
    (define k (- n 1))
    (define x (Math.round v))
    (define d 0)
    (while (>= k 0) {
      (set! d (get_item P2 k))
      (if (>= x d)
          { (push bits_buf 1) (set! x (- x d)) }
          (push bits_buf 0))
      (set! k (- k 1))
    })
  }))

;; 把 bits_buf 打包成字节对齐的位串（每帧一次）
(define flush_bits
  (lambda () {
    (define n (length bits_buf))
    (define r (Math.round (mod n 8)))
    (define pad (if (== r 0) 0 (- 8 r)))
    (define s "")
    (define i 0)
    (define b 0)
    (define k 0)
    (set! frame_bitpos n)
    (set! k 0)
    (while (< k pad) { (push bits_buf 0) (set! k (+ k 1)) })
    (set! n (+ n pad))
    (set! i 0)
    (while (< i n) {
      (set! b (+ (* (get_item bits_buf i) 128)
                 (+ (* (get_item bits_buf (+ i 1)) 64)
                    (+ (* (get_item bits_buf (+ i 2)) 32)
                       (+ (* (get_item bits_buf (+ i 3)) 16)
                          (+ (* (get_item bits_buf (+ i 4)) 8)
                             (+ (* (get_item bits_buf (+ i 5)) 4)
                                (+ (* (get_item bits_buf (+ i 6)) 2)
                                   (get_item bits_buf (+ i 7))))))))))
      (set! s (String.concat s (get_item B2S b)))
      (set! i (+ i 8))
    })
    (while (> (length bits_buf) 0) (pop bits_buf))
    s
  }))

;; 指数哥伦布（order-0）变长码：x ≥ 1 → n=floor(log2 x) 个 0 + x 的 n+1 位。
;; 码长 = 2n+1。用它替掉定长电平码：小电平（最常见）只要 1~4 位，
;; 这正是 H.261/MPEG-1 用 VLC 表换来的那点收益的最小实现。
(define gcost
  (lambda (x0) {
    (define x (Math.round x0))
    (if (< x 1) 1 (+ (* 2 (Math.floor (Math.log2 x))) 1))
  }))

(define wgamma
  (lambda (x0) {
    (define x (Math.round x0))
    (define n (Math.floor (Math.log2 x)))
    (define i 0)
    (while (< i n) { (push bits_buf 0) (set! i (+ i 1)) })
    (wval x (+ n 1))
  }))

;; 读 1 bit
(define rbit
  (lambda () {
    (define v (- (String.charCodeAt brp brs) 48))
    (set! brp (+ brp 1))
    v
  }))

;; 读 n bit（MSB 优先）
(define rval
  (lambda (n) {
    (define k 0)
    (define v 0)
    (while (< k n) {
      (set! v (+ (* v 2) (- (String.charCodeAt brp brs) 48)))
      (set! brp (+ brp 1))
      (set! k (+ k 1))
    })
    v
  }))

;; 指数哥伦布码的读取。**前导 0 的个数必须封顶**：合法码值只有 run+1 ≤ 65 与
;; 电平 ≤ 255 两种（n ≤ 7），但一旦码流失步（读到别的位、或读过帧尾），前导 0 可能
;; 连成一长串 —— 旧版这里只判 `(== (rbit) 0)`，n 涨到 18 以上时 `(get_item P2 n)`
;; 就越界返回 #undefined，紧接着 `(+ v rest)` 报「指令执行异常」，**整个虚拟机进程
;; 停机 → 界面冻在最后一帧**（本 App 曾经就这样卡死；见 restart_encode 处注释）。
;; 封顶到 P2 的合法下标上限：码流失步时得到的是错值（回环校验会照实报 mismatch），
;; 而不是让演示程序直接死掉——这与手册「软性失败」的一贯口径一致。
(define GAMMA_NMAX 17)

(define rgamma
  (lambda () {
    (define n 0)
    (define v 1)
    (define i 0)
    (define rest 0)
    (while (and (== (rbit) 0) (< n GAMMA_NMAX)) (set! n (+ n 1)))
    (set! v (get_item P2 n))
    (set! i 0)
    (while (< i n) {
      (set! rest (+ (* rest 2) (rbit)))
      (set! i (+ i 1))
    })
    (+ v rest)
  }))

;; ============================================================================
;; §6 帧数据读取（定长记录 → 4096 个 0/1）
;; ============================================================================

(define load_frame_into
  (lambda (fi buf) {
    (define i 0)
    (define r #null)
    (define s "")
    (define cnt 0)
    (define k 0)
    (define p 0)
    (define yy 0)
    (define xx 0)
    (define ll 0)
    (define base 0)
    (set! i 0)
    (while (< i NPIX) { (set_item! buf i 0) (set! i (+ i 1)) })
    (set! r (Fs.read_range PATH (* (+ START_F fi) REC) REC))
    (if (null? r) 0
      {
        (set! s (get_item r 0))
        (set! cnt (+ (* (- (String.charCodeAt 0 s) CH0) RADIX) (- (String.charCodeAt 1 s) CH0)))
        (set! k 0)
        (while (< k cnt) {
          (set! p (+ HEAD (* k CPR)))
          (set! yy (- (String.charCodeAt p s) CH0))
          (set! xx (- (String.charCodeAt (+ p 1) s) CH0))
          (set! ll (+ 1 (- (String.charCodeAt (+ p 2) s) CH0)))
          (set! base (* yy W))
          (set! i 0)
          (while (< i ll) { (set_item! buf (+ base (+ xx i)) 255) (set! i (+ i 1)) })
          (set! k (+ k 1))
        })
        1
      })
  }))

;; 保证 asrc 里是第 f 帧（缓存，避免每拍读盘）
(define ensure_src
  (lambda (f) {
    (if (not (== src_f f)) { (load_frame_into f asrc) (set! src_f f) } #f)
  }))

;; ============================================================================
;; §7 编解码核心
;; ============================================================================

(define set_bxy
  (lambda (b) {
    (set! g_bx (* (Math.round (mod b BW)) BS))
    (set! g_by (* (Math.floor (/ b BW)) BS))
    (set! g_boff (+ (* g_by W) g_bx))
  }))

;; 把 (pb + ob) 写回 recbuf 的 8×8 块（g_boff 处）——**按二维布局逐行写**
;; （帧缓冲是 W=64 行优先，块占 8 行 × 8 列；早先误用线性 0..63 会把块摊成一行）
(define store_block
  (lambda () {
    (define y 0)
    (define x 0)
    (define p 0)
    (define b2 0)
    (while (< y 8) {
      (set! p (* y 8))
      (set! b2 (+ g_boff (* y W)))
      (set! x 0)
      (while (< x 8) {
        (set_item! recbuf (+ b2 x)
          (clampf (+ (get_item pb (+ p x)) (get_item ob (+ p x))) 0.0 255.0))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
  }))

;; 从 recbuf 的 8×8 块取到线性 64 项（面板显示用）
(define load_block
  (lambda (out) {
    (define y 0)
    (define x 0)
    (define p 0)
    (define b2 0)
    (while (< y 8) {
      (set! p (* y 8))
      (set! b2 (+ g_boff (* y W)))
      (set! x 0)
      (while (< x 8) {
        (set_item! out (+ p x) (get_item recbuf (+ b2 x)))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
  }))

;; 取源块 → sb
(define gather_src
  (lambda (buf) {
    (define y 0)
    (define x 0)
    (define base g_boff)
    (while (< y 8) {
      (set! x 0)
      (while (< x 8) {
        (set_item! sb (+ (* y 8) x) (get_item buf (+ base x)))
        (set! x (+ x 1))
      })
      (set! base (+ base W))
      (set! y (+ y 1))
    })
  }))

;; 帧内预测：从 curbuf 的**已重建**上邻行/左邻列生成预测块 → pb
;; mode: 0=DC 1=垂直 2=水平 3=平面
(define pred_intra
  (lambda (m0) {
    (define y 0)
    (define x 0)
    (define st 0.0)
    (define sl 0.0)
    (define dcv 0.0)
    (define tr 0.0)
    (define bl 0.0)
    (define m m0)
    (define top 0)
    (define left 0)
    (set! top (if (> g_by 0) 1 0))
    (set! left (if (> g_bx 0) 1 0))
    (set! x 0)
    (while (< x 8) {
      (set_item! an_top x (if (== top 1) (get_item curbuf (+ (* (- g_by 1) W) (+ g_bx x))) 0.0))
      (set_item! an_left x (if (== left 1) (get_item curbuf (+ (* (+ g_by x) W) (- g_bx 1))) 0.0))
      (set! x (+ x 1))
    })
    (set! x 0)
    (while (< x 8) {
      (set! st (+ st (get_item an_top x)))
      (set! sl (+ sl (get_item an_left x)))
      (set! x (+ x 1))
    })
    (set! dcv (if (== top 1)
                  (if (== left 1) (/ (+ st sl) 16.0) (/ st 8.0))
                  (if (== left 1) (/ sl 8.0) 0.0)))
    (set! tr (get_item an_top 7))
    (set! bl (get_item an_left 7))
    ;; 模式不可用时退回 DC（解码器走同一函数，语义自动一致）
    (if (== m 1) (if (== top 0) (set! m 0) #f) #f)
    (if (== m 2) (if (== left 0) (set! m 0) #f) #f)
    (if (== m 3) (if (or (== top 0) (== left 0)) (set! m 0) #f) #f)
    (set! y 0)
    (while (< y 8) {
      (set! x 0)
      (while (< x 8) {
        (set_item! pb (+ (* y 8) x)
          (cond ((== m 0) dcv)
                ((== m 1) (get_item an_top x))
                ((== m 2) (get_item an_left y))
                (else (/ (+ (* (- 7 x) (get_item an_left y))
                            (+ (* (+ x 1) tr)
                               (+ (* (- 7 y) (get_item an_top x))
                                  (* (+ y 1) bl)))) 16.0))))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
  }))

;; 帧间预测：从 reff 取 (bx+mvx, by+mvy) 处的块 → pb（边界外按最近像素延拓）
;; 边界钳位在热循环里**内联**成 if（调用 clampi 闭包要 ~50 条指令/次，
;; 逐像素调用会让运动补偿与 SAD 慢 5~10 倍——本文件最大的性能教训）。
(define pred_inter
  (lambda (mvx mvy) {
    (define y 0)
    (define x 0)
    (define sx 0)
    (define sy 0)
    (define ro 0)
    (define p 0)
    (set! y 0)
    (while (< y 8) {
      (set! sy (+ (+ g_by y) mvy))
      (if (< sy 0) (set! sy 0) (if (> sy 63) (set! sy 63) #f))
      (set! ro (* sy 64))
      (set! p (* y 8))
      (set! x 0)
      (while (< x 8) {
        (set! sx (+ (+ g_bx x) mvx))
        (if (< sx 0) (set! sx 0) (if (> sx 63) (set! sx 63) #f))
        (set_item! pb (+ p x) (get_item reff (+ ro sx)))
        (set! x (+ x 1))
      })
      (set! y (+ y 1))
    })
  }))

;; 残差 rb = sb - pb；同时判定是否全零
(define build_res
  (lambda () {
    (define i 0)
    (define v 0.0)
    (set! g_zero 1)
    (while (< i 64) {
      (set! v (- (get_item sb i) (get_item pb i)))
      (set_item! rb i v)
      (if (not (== v 0.0)) (set! g_zero 0) #f)
      (set! i (+ i 1))
    })
  }))

;; ---- 1D 8 点 DCT（正变换）：src[addr+i*st] (i=0..7) → dst[out+u*ost] ----
;; 用偶部/奇部成对对称性：8 个输出只花 32 次乘加；输出顺序 u = 0,1,2,...,7。
;; ---- 1D 8 点 DCT（正变换）：src[addr+i*st] (i=0..7) → dst[out+u*ost] ----
;; 用偶部/奇部成对对称性：8 个输出只花 32 次乘加。
;; 基表与列表把柄都先取到**局部槽位**（模块级变量每次读都要沿闭包链 deref，
;; 一次 8×8 变换要读 64 次基表——提到局部能省掉约 1/4 的指令）。
(define dct1d
  (lambda (srl addr st dst out ost) {
    (define b0 (get_item TBE 0))
    (define b1 (get_item TBE 1))
    (define b2 (get_item TBE 2))
    (define b3 (get_item TBE 3))
    (define b4 (get_item TBE 4))
    (define b5 (get_item TBE 5))
    (define b6 (get_item TBE 6))
    (define b7 (get_item TBE 7))
    (define b8 (get_item TBE 8))
    (define b9 (get_item TBE 9))
    (define b10 (get_item TBE 10))
    (define b11 (get_item TBE 11))
    (define b12 (get_item TBE 12))
    (define b13 (get_item TBE 13))
    (define b14 (get_item TBE 14))
    (define b15 (get_item TBE 15))
    (define o0 (get_item TBO 0))
    (define o1 (get_item TBO 1))
    (define o2 (get_item TBO 2))
    (define o3 (get_item TBO 3))
    (define o4 (get_item TBO 4))
    (define o5 (get_item TBO 5))
    (define o6 (get_item TBO 6))
    (define o7 (get_item TBO 7))
    (define o8 (get_item TBO 8))
    (define o9 (get_item TBO 9))
    (define o10 (get_item TBO 10))
    (define o11 (get_item TBO 11))
    (define o12 (get_item TBO 12))
    (define o13 (get_item TBO 13))
    (define o14 (get_item TBO 14))
    (define o15 (get_item TBO 15))
    (define a 0.0)
    (define bb 0.0)
    (define c 0.0)
    (define dd 0.0)
    (define e 0.0)
    (define f 0.0)
    (define g 0.0)
    (define h 0.0)
    (define p0 0.0)
    (define p1 0.0)
    (define p2 0.0)
    (define p3 0.0)
    (define q0 0.0)
    (define q1 0.0)
    (define q2 0.0)
    (define q3 0.0)
    (define o 0)
    (set! a (get_item srl (+ addr (* 0 st))))
    (set! bb (get_item srl (+ addr (* 1 st))))
    (set! c (get_item srl (+ addr (* 2 st))))
    (set! dd (get_item srl (+ addr (* 3 st))))
    (set! e (get_item srl (+ addr (* 4 st))))
    (set! f (get_item srl (+ addr (* 5 st))))
    (set! g (get_item srl (+ addr (* 6 st))))
    (set! h (get_item srl (+ addr (* 7 st))))
    (set! p0 (+ a h))
    (set! p1 (+ bb g))
    (set! p2 (+ c f))
    (set! p3 (+ dd e))
    (set! q0 (- a h))
    (set! q1 (- bb g))
    (set! q2 (- c f))
    (set! q3 (- dd e))
    (set! o out)
    (set_item! dst o (+ (* b0 p0)
          (+ (* b1 p1) (+ (* b2 p2) (* b3 p3)))))
    (set! o (+ o ost))
    (set_item! dst o (+ (* o0 q0)
          (+ (* o1 q1) (+ (* o2 q2) (* o3 q3)))))
    (set! o (+ o ost))
    (set_item! dst o (+ (* b4 p0)
          (+ (* b5 p1) (+ (* b6 p2) (* b7 p3)))))
    (set! o (+ o ost))
    (set_item! dst o (+ (* o4 q0)
          (+ (* o5 q1) (+ (* o6 q2) (* o7 q3)))))
    (set! o (+ o ost))
    (set_item! dst o (+ (* b8 p0)
          (+ (* b9 p1) (+ (* b10 p2) (* b11 p3)))))
    (set! o (+ o ost))
    (set_item! dst o (+ (* o8 q0)
          (+ (* o9 q1) (+ (* o10 q2) (* o11 q3)))))
    (set! o (+ o ost))
    (set_item! dst o (+ (* b12 p0)
          (+ (* b13 p1) (+ (* b14 p2) (* b15 p3)))))
    (set! o (+ o ost))
    (set_item! dst o (+ (* o12 q0)
          (+ (* o13 q1) (+ (* o14 q2) (* o15 q3)))))
  }))

;; ---- 8×8 二维正 DCT（可分离 + 稀疏感知）：rb → cb ----
;; 先行后列：第一遍每行 1 次 1D（输出 tb[u*8+y]），第二遍每列 1 次 1D（输出 cb[u*8+v]）。
;; 全零的行/列整行跳过（二值线稿里大片背景的残差恰好全零，这一条省掉一半以上开销）。
(define fdct
  (lambda () {
    (define y 0)
    (define u 0)
    (define v 0)
    (define x 0)
    (define nz 0)
    (define p 0)
    (set! y 0)
    (while (< y 8) {
      (set! p (* y 8))
      (set! x 0)
      (set! nz 0)
      (while (< x 8) {
        (if (not (== (get_item rb (+ p x)) 0.0)) (set! nz 1) #f)
        (set! x (+ x 1))
      })
      (if (== nz 0)
          { (set! u 0) (while (< u 8) { (set_item! tb (+ (* u 8) y) 0.0) (set! u (+ u 1)) }) }
          (dct1d rb p 1 tb y 8))
      (set! y (+ y 1))
    })
    (set! u 0)
    (while (< u 8) {
      (set! p (* u 8))
      (set! y 0)
      (set! nz 0)
      (while (< y 8) {
        (if (not (== (get_item tb (+ p y)) 0.0)) (set! nz 1) #f)
        (set! y (+ y 1))
      })
      (if (== nz 0)
          { (set! v 0) (while (< v 8) { (set_item! cb (+ p v) 0.0) (set! v (+ v 1)) }) }
          (dct1d tb p 1 cb p 1))
      (set! u (+ u 1))
    })
  }))

;; 量化：cb → qb（整数电平），并算出变换域失真 g_d 与非零个数 g_nz
(define quantize
  (lambda (q0) {
    (define q (maxf 1 q0))
    (define qe 1.0)
    (define i 0)
    (define v 0.0)
    (define lv 0)
    (define e 0.0)
    (set! g_d 0.0)
    (set! g_nz 0)
    (while (< i 64) {
      (set! v (get_item cb i))
      (set! qe (* q (get_item QW i)))
      (set! lv (clampi (Math.round (/ v qe)) (- 0 255) 255))
      (set_item! qb i lv)
      (set! e (- v (* lv qe)))
      (set! g_d (+ g_d (* e e)))
      (if (not (== lv 0)) (set! g_nz (+ g_nz 1)) #f)
      (set! i (+ i 1))
    })
  }))

;; ---- 1D 8 点逆 DCT：src[addr+u*st] (u=0..7) → dst[out+y*ost] ----
;; 同样利用成对对称性：一次算出 y 与 7-y 两个输出（P+Q / P-Q）。
(define idct1d
  (lambda (srl addr st dst out ost) {
    (define c0 0.0)
    (define c1 0.0)
    (define c2 0.0)
    (define c3 0.0)
    (define c4 0.0)
    (define c5 0.0)
    (define c6 0.0)
    (define c7 0.0)
    (define p 0.0)
    (define q 0.0)
    (define o 0)
    (set! c0 (get_item srl addr))
    (set! c1 (get_item srl (+ addr st)))
    (set! c2 (get_item srl (+ addr (* 2 st))))
    (set! c3 (get_item srl (+ addr (* 3 st))))
    (set! c4 (get_item srl (+ addr (* 4 st))))
    (set! c5 (get_item srl (+ addr (* 5 st))))
    (set! c6 (get_item srl (+ addr (* 6 st))))
    (set! c7 (get_item srl (+ addr (* 7 st))))
    (set! o out)
    (set! p (+ (* (get_item TBE 0) c0) (+ (* (get_item TBE 4) c2)
              (+ (* (get_item TBE 8) c4) (* (get_item TBE 12) c6)))))
    (set! q (+ (* (get_item TBO 0) c1) (+ (* (get_item TBO 4) c3)
              (+ (* (get_item TBO 8) c5) (* (get_item TBO 12) c7)))))
    (set_item! dst o (+ p q))
    (set_item! dst (+ o (* 7 ost)) (- p q))
    (set! p (+ (* (get_item TBE 1) c0) (+ (* (get_item TBE 5) c2)
              (+ (* (get_item TBE 9) c4) (* (get_item TBE 13) c6)))))
    (set! q (+ (* (get_item TBO 1) c1) (+ (* (get_item TBO 5) c3)
              (+ (* (get_item TBO 9) c5) (* (get_item TBO 13) c7)))))
    (set_item! dst (+ o ost) (+ p q))
    (set_item! dst (+ o (* 6 ost)) (- p q))
    (set! p (+ (* (get_item TBE 2) c0) (+ (* (get_item TBE 6) c2)
              (+ (* (get_item TBE 10) c4) (* (get_item TBE 14) c6)))))
    (set! q (+ (* (get_item TBO 2) c1) (+ (* (get_item TBO 6) c3)
              (+ (* (get_item TBO 10) c5) (* (get_item TBO 14) c7)))))
    (set_item! dst (+ o (* 2 ost)) (+ p q))
    (set_item! dst (+ o (* 5 ost)) (- p q))
    (set! p (+ (* (get_item TBE 3) c0) (+ (* (get_item TBE 7) c2)
              (+ (* (get_item TBE 11) c4) (* (get_item TBE 15) c6)))))
    (set! q (+ (* (get_item TBO 3) c1) (+ (* (get_item TBO 7) c3)
              (+ (* (get_item TBO 11) c5) (* (get_item TBO 15) c7)))))
    (set_item! dst (+ o (* 3 ost)) (+ p q))
    (set_item! dst (+ o (* 4 ost)) (- p q))
  }))

;; 量化电平的保存/恢复（RDO 两档量化对比）
(define save_qb
  (lambda () {
    (define i 0)
    (while (< i 64) { (set_item! qsav i (get_item qb i)) (set! i (+ i 1)) })
  }))
(define restore_qb
  (lambda () {
    (define i 0)
    (while (< i 64) { (set_item! qb i (get_item qsav i)) (set! i (+ i 1)) })
  }))

;; ---- 8×8 二维逆 DCT（可分离 + 稀疏感知）：db → ob ----
;; 第一遍按列（对每个 v 沿 u 求和 → tb[y*8+v]），第二遍按行（→ ob[y*8+x]）。
(define idct
  (lambda () {
    (define u 0)
    (define v 0)
    (define y 0)
    (define nz 0)
    ;; 第一遍：对每个水平频率 u，沿垂直频率 v 做 1D（C[u][v] → T[u][y]）
    ;; 注意下标配对：正变换把 u 与 x、v 与 y 配对，逆变换必须反过来
    ;; （早先这里把 u/v 的角色写反了，重建出来是残差的**转置**——
    ;;   对称的测试图案会让自检误报通过，务必用非对称图案验）。
    (set! u 0)
    (while (< u 8) {
      (set! v 0)
      (set! nz 0)
      (while (< v 8) {
        (if (not (== (get_item db (+ (* u 8) v)) 0.0)) (set! nz 1) #f)
        (set! v (+ v 1))
      })
      (if (== nz 0)
          { (set! y 0) (while (< y 8) { (set_item! tb (+ u (* y 8)) 0.0) (set! y (+ y 1)) }) }
          (idct1d db (* u 8) 1 tb u 8))
      (set! u (+ u 1))
    })
    ;; 第二遍：对每行 y，沿水平频率 u 做 1D（T[u][y] → f[y][x]）
    (set! y 0)
    (while (< y 8) {
      (idct1d tb (* y 8) 1 ob (* y 8) 1)
      (set! y (+ y 1))
    })
  }))

;; 之字形游程扫描：tbl[base + ZIG[i]] → rle_run / rle_lvl，个数 → g_nz
(define rle_scan
  (lambda (tbl base) {
    (define i 0)
    (define run 0)
    (define nz 0)
    (define v 0)
    (define cost 0)
    (while (> (length rle_run) 0) (pop rle_run))
    (while (> (length rle_lvl) 0) (pop rle_lvl))
    (set! i 0)
    (while (< i 64) {
      (set! v (get_item tbl (+ base (get_item ZIG i))))
      (if (== v 0)
          (set! run (+ run 1))
          {
            (push rle_run run)
            (push rle_lvl v)
            ;; 码长 = 符号位(1) + 游程变长码 + 电平变长码
            (set! cost (+ cost (+ 1 (+ (gcost (+ run 1)) (gcost (absf v))))))
            (set! run 0)
            (set! nz (+ nz 1))
          })
      (set! i (+ i 1))
    })
    (set! g_nz nz)
    (set! g_rc cost)
  }))

;; 运动补偿 SAD（step = 采样步长：2 = 隔点粗搜，1 = 全像素）
(define sad_mv
  (lambda (mvx mvy step) {
    (define y 0)
    (define x 0)
    (define sx 0)
    (define sy 0)
    (define s 0.0)
    (define d 0.0)
    (define p 0)
    (define ro 0)
    (set! y 0)
    (while (< y 8) {
      (set! sy (+ (+ g_by y) mvy))
      (if (< sy 0) (set! sy 0) (if (> sy 63) (set! sy 63) #f))
      (set! ro (* sy 64))
      (set! p (* y 8))
      (set! x 0)
      (while (< x 8) {
        (set! sx (+ (+ g_bx x) mvx))
        (if (< sx 0) (set! sx 0) (if (> sx 63) (set! sx 63) #f))
        (set! d (- (get_item sb (+ p x)) (get_item reff (+ ro sx))))
        (set! s (+ s (if (< d 0) (- 0 d) d)))
        (set! x (+ x step))
      })
      (set! y (+ y step))
    })
    s
  }))

;; 两阶段运动搜索：±SR 隔点粗搜（步长 2）→ ±1 全像素细化
(define me_search
  (lambda () {
    (define dx 0)
    (define dy 0)
    (define best 0.0)
    (define s 0.0)
    (define bx0 0)
    (define by0 0)
    (set! best 1000000.0)
    (set! bx0 0)
    (set! by0 0)
    (set! dy (- 0 SR))
    (while (<= dy SR) {
      (set! dx (- 0 SR))
      (while (<= dx SR) {
        (set! s (sad_mv dx dy 2))
        (if (< s best) { (set! best s) (set! bx0 dx) (set! by0 dy) } #f)
        (set! dx (+ dx 2))
      })
      (set! dy (+ dy 2))
    })
    (set! g_mvx bx0)
    (set! g_mvy by0)
    (set! dy -1)
    (while (<= dy 1) {
      (set! dx -1)
      (while (<= dx 1) {
        (set! s (sad_mv (+ bx0 dx) (+ by0 dy) 1))
        (if (< s best) { (set! best s) (set! g_mvx (+ bx0 dx)) (set! g_mvy (+ by0 dy)) } #f)
        (set! dx (+ dx 1))
      })
      (set! dy (+ dy 1))
    })
    (set! g_mvx (clampi g_mvx (- 0 SR) SR))
    (set! g_mvy (clampi g_mvy (- 0 SR) SR))
  }))

;; 帧内模式快速预选：4 个预测块里 SAD 最小的那个（真实编码器同样先粗选再精算）。
;; P 帧有 SKIP 与帧间两个强候选，把 4 个帧内模式全做一遍精确 RDO 太贵——
;; 这里先用廉价的 SAD 选一个帧内模式，再与 SKIP/帧间一起做精确 RDO（3 选 1）。
(define preselect_intra
  (lambda () {
    (define m 0)
    (define bm 0)
    (define bs 0.0)
    (define sv 0.0)
    (define d 0.0)
    (define i 0)
    (set! bm 0)
    (set! bs 1.0e30)
    (set! m 0)
    (while (< m 4) {
      (pred_intra m)
      (set! sv 0.0)
      (set! i 0)
      (while (< i 64) {
        (set! d (- (get_item sb i) (get_item pb i)))
        (set! sv (+ sv (if (< d 0) (- 0 d) d)))
        (set! i (+ i 1))
      })
      (if (< sv bs) { (set! bs sv) (set! bm m) } #f)
      (set! m (+ m 1))
    })
    bm
  }))

;; 评估一个候选并写入候选表
;;   k = 槽号, mode = 模式号, hdrbits = 模式/矢量头比特数
(define eval_cand
  (lambda (k0 mode hdrbits) {
    (define k (clampi k0 0 5))
    (define i 0)
    (define d 0.0)
    (define v 0.0)
    (define bits 0)
    (define nz 0)
    (define rc 0)
    (define r1 0)
    (define d1 0.0)
    (if (== mode C_SKIP)
        {
          (pred_inter 0 0)
          (set! d 0.0)
          (set! i 0)
          (while (< i 64) {
            (set! v (- (get_item sb i) (get_item pb i)))
            (set! d (+ d (* v v)))
            (set! i (+ i 1))
          })
          (set! nz 0)
        }
        {
          (if (== mode C_INTER)
              (pred_inter g_mvx g_mvy)
              (pred_intra mode))
          (build_res)
          (if (== g_zero 1)
              { (set! d 0.0) (set! nz 0) (set! rc 0) }
              {
                (fdct)
                ;; 两档量化（Q 与 2Q）各自算 (D,R)，由 λ 决定取哪个：
                ;; 这就是「率失真优化」在量化档位上的最小体现——λ 大时更愿意用
                ;; 粗档（少发系数、失真略增），λ 小时用细档。
                (quantize g_q)
                (save_qb)
                (rle_scan qb 0)
                (set! d1 g_d)
                (set! r1 (+ B_NZ g_rc))
                (quantize (* 2 g_q))
                (rle_scan qb 0)
                (if (< (+ g_d (* LAMBDA (+ B_NZ g_rc))) (+ d1 (* LAMBDA r1)))
                    { (set! d g_d) (set! nz g_nz) (set! rc g_rc) }
                    {
                      (restore_qb)
                      (rle_scan qb 0)
                      (set! d d1) (set! nz g_nz) (set! rc g_rc)
                    })
              })
        })
    (set! bits (if (== mode C_SKIP) hdrbits (+ hdrbits (+ B_NZ rc))))
    (set_item! CAND_M k mode)
    (set_item! CAND_N k nz)
    (set_item! CAND_D k d)
    (set_item! CAND_R k bits)
    (set_item! CAND_J k (+ d (* LAMBDA bits)))
    (set! i 0)
    (while (< i 64) {
      (set_item! CAND_C (+ (* k 64) i)
        (if (== nz 0) 0.0 (if (== mode C_SKIP) 0.0 (get_item cb i))))
      (set_item! CAND_Q (+ (* k 64) i) (if (== nz 0) 0 (get_item qb i)))
      (set! i (+ i 1))
    })
  }))

;; 选最优候选（J 最小）
(define pick_best
  (lambda () {
    (define k 0)
    (define bj 1.0e30)
    (set! g_best 0)
    (set! k 0)
    (while (< k NCAND) {
      (if (< (get_item CAND_J k) bj) { (set! bj (get_item CAND_J k)) (set! g_best k) } #f)
      (set! k (+ k 1))
    })
  }))

;; 按最优候选重建到 recbuf，并快照到 an_*
(define reconstruct
  (lambda () {
    (define mode (get_item CAND_M g_best))
    (define i 0)
    (define q g_q)
    (define nz (get_item CAND_N g_best))
    (if (== mode C_SKIP)
        (pred_inter 0 0)
        (if (== mode C_INTER)
            (pred_inter g_mvx g_mvy)
            (pred_intra mode)))
    (build_res)
    (if (== nz 0)
        { (set! i 0) (while (< i 64) { (set_item! ob i 0.0) (set! i (+ i 1)) }) }
        {
          (set! i 0)
          (while (< i 64) {
            (set_item! db i (* (get_item CAND_Q (+ (* g_best 64) i)) (* q (get_item QW i))))
            (set! i (+ i 1))
          })
          (idct)
        })
    (store_block)
    ;; 快照（面板用；an_* 不受后续编码影响）
    (set! i 0)
    (while (< i 64) {
      (set_item! an_pred i (get_item pb i))
      (set_item! an_res i (get_item rb i))
      (set_item! an_coef i (get_item CAND_C (+ (* g_best 64) i)))
      (set_item! an_q i (get_item CAND_Q (+ (* g_best 64) i)))
      (set_item! an_deq i (* (get_item CAND_Q (+ (* g_best 64) i)) (* q (get_item QW i))))
      (set! i (+ i 1))
    })
    (load_block an_recon)
    (set! an_nz nz)
    (set! an_d (get_item CAND_D g_best))
    (set! an_j (get_item CAND_J g_best))
    (set! an_bits (get_item CAND_R g_best))
  }))

;; 按码流语法写出最优候选
(define write_block_bits
  (lambda () {
    (define mode (get_item CAND_M g_best))
    (define k 0)
    (define lv 0)
    (if (== enc_ftype 1)
        (if (== mode C_SKIP)
            (wbit 0)
            {
              (wbit 1)
              (if (== mode C_INTER)
                  {
                    (wbit 1)
                    (wval (+ g_mvx SR) B_MV)
                    (wval (+ g_mvy SR) B_MV)
                  }
                  { (wbit 0) (wval mode B_IMODE) })
            })
        (wval mode B_IMODE))
    (rle_scan CAND_Q (* g_best 64))
    (if (not (== mode C_SKIP)) (wval g_nz B_NZ) #f)
    (set! k 0)
    (while (< k g_nz) {
      (wgamma (+ (get_item rle_run k) 1))
      (set! lv (get_item rle_lvl k))
      (wgamma (absf lv))
      (wbit (if (< lv 0) 1 0))
      (set! k (+ k 1))
    })
  }))

;; ============================================================================
;; §8 编码器
;; ============================================================================

;; 帧间差异像素数（关键帧判定）
(define diff_count
  (lambda (a b) {
    (define i 0)
    (define c 0)
    (set! i 0)
    (while (< i NPIX) {
      (if (not (== (get_item a i) (get_item b i))) (set! c (+ c 1)) #f)
      (set! i (+ i 1))
    })
    c
  }))

(define enc_frame_start
  (lambda (f) {
    (define i 0)
    (define d 0)
    (set! enc_f f)
    (set! enc_b 0)
    (load_frame_into f src)
    (set! reff (if (== f 0) zf (get_item rec_frames (- f 1))))
    ;; 重建帧**直接写进存档表** rec_frames[f]：它既是最优模式的判定依据，
    ;; 也是下一帧的参考帧（早先另用一个临时缓冲且从不回存 → 参考帧恒为零帧，
    ;; 帧间预测因此永远赢不了，回环校验也对不上）。
    (set! rec (get_item rec_frames f))
    (set! curbuf rec)
    (set! recbuf rec)
    ;; 未编码区域先填参考帧（部分重建的画面看起来才连贯）
    (set! i 0)
    (while (< i NPIX) { (set_item! rec i (get_item reff i)) (set! i (+ i 1)) })
    ;; 关键帧判定用**源帧之间**的差异（用源 vs 重建会把编码损失也算进去，
    ;; 于是每个 P 帧都会被误判成场景切换）
    (set! d (diff_count src psrc))
    (set! enc_ftype (if (== f 0) 0
                        (if (== (Math.round (mod f GOP)) 0) 0
                            (if (> d SCENE_TH) 0 1))))
    ;; 记下本帧源，供下一帧判关键帧
    (set! i 0)
    (while (< i NPIX) { (set_item! psrc i (get_item src i)) (set! i (+ i 1)) })
    (set! g_q QSTEP)
    (set_item! fr_type f enc_ftype)
    (set_item! fr_q f QSTEP)
    (set_item! fr_ni f 0)
    (set_item! fr_np f 0)
    (set_item! fr_ns f 0)
    (set_item! fr_sse f 0.0)
    (wval enc_ftype 1)
    (wval QSTEP 6)
  }))

(define enc_block
  (lambda (b) {
    (define mode 0)
    (set! curbuf rec)
    (set! reff (if (== enc_f 0) zf (get_item rec_frames (- enc_f 1))))
    (set! recbuf rec)
    (set! g_q QSTEP)
    (set_bxy b)
    (gather_src src)
    (set! NCAND 0)
    (if (== enc_ftype 1)
        {
          ;; P 帧：先看零运动矢量下的残差是否全零 —— 全零则 SKIP 必胜
          ;; （D=0、R=1 位，任何别的候选都不可能更便宜）。这就是 RDO 的
          ;; 「提前终止」：真实编码器里绝大多数静态块走的就是这条捷径。
          (pred_inter 0 0)
          (build_res)
          (if (== g_zero 1)
              {}
              {
                ;; 否则：SKIP / 帧间(MV) / 帧内(预选模式) —— 3 选 1
                (me_search)
                (eval_cand 0 C_SKIP 1)
                (set! _ihdr B_INTER_HDR)
                (eval_cand 1 C_INTER _ihdr)
                (set! _imi (preselect_intra))
                (eval_cand 2 _imi B_IMODE)
                (set! NCAND 3)
              })
        }
        {
          ;; I 帧：4 种帧内预测各自做精确 RDO
          (set! mode 0)
          (while (< mode 4) {
            (eval_cand NCAND mode B_IMODE)
            (set! NCAND (+ NCAND 1))
            (set! mode (+ mode 1))
          })
        })
    (if (== NCAND 0)
        { (eval_cand 0 C_SKIP 1) (set! NCAND 1) }
        #f)
    (pick_best)
    (reconstruct)
    (write_block_bits)
    (set_item! (get_item fr_modes enc_f) b (get_item CAND_M g_best))
    (set_item! (get_item fr_mvx enc_f) b (if (== (get_item CAND_M g_best) C_INTER) g_mvx 0))
    (set_item! (get_item fr_mvy enc_f) b (if (== (get_item CAND_M g_best) C_INTER) g_mvy 0))
    (set_item! (get_item fr_bnz enc_f) b (get_item CAND_N g_best))
    (set_item! (get_item fr_bd enc_f) b (get_item CAND_D g_best))
    (set_item! fr_ni enc_f (+ (get_item fr_ni enc_f) (if (< (get_item CAND_M g_best) 4) 1 0)))
    (set_item! fr_np enc_f (+ (get_item fr_np enc_f) (if (== (get_item CAND_M g_best) C_INTER) 1 0)))
    (set_item! fr_ns enc_f (+ (get_item fr_ns enc_f) (if (== (get_item CAND_M g_best) C_SKIP) 1 0)))
    (set_item! fr_sse enc_f (+ (get_item fr_sse enc_f) (get_item CAND_D g_best)))
    (set! enc_b (+ b 1))
  }))

(define enc_frame_end
  (lambda (f) {
    (define i 0)
    (define e 0)
    (define fr (get_item rec_frames f))
    (set_item! bs f (flush_bits))
    (set_item! fr_bits f frame_bitpos)
    (set! i 0)
    (while (< i NPIX) {
      (if (not (== (if (< (get_item fr i) 127.5) 0 255) (get_item src i))) (set! e (+ e 1)) #f)
      (set! i (+ i 1))
    })
    (set_item! fr_err f e)
    (if (== dbg 1)
        { (display "[vcodec] f=") (display f) (display " type=") (display enc_ftype)
          (display " bits=") (display frame_bitpos)
          (display " ni=") (display (get_item fr_ni f))
          (display " np=") (display (get_item fr_np f))
          (display " ns=") (display (get_item fr_ns f))
          (display " t=") (display (System.timestamp))
          (newline) }
        #f)
  }))

;; 编码推进：每拍编码 ENC_CHUNK 个块（可跨帧）
(define enc_step
  (lambda () {
    (define n 0)
    (if (== enc_paused 1) 0
        (while (< n ENC_CHUNK) {
          (if (< enc_f NFR)
              {
                (if (== enc_b 0) (enc_frame_start enc_f) #f)
                (enc_block enc_b)
                (if (>= enc_b NB)
                    {
                      (enc_frame_end enc_f)
                      (set! enc_f (+ enc_f 1))
                      (set! enc_b 0)
                      (if (>= enc_f NFR)
                          {
                            (set! phase PH_DEC)
                            (set! enc_ms (- (Diag.ms) t_start))
                            (display "[vcodec] encoded bits=") (display (total_bits))
                            (display " enc_ms=") (display enc_ms) (newline)
                            (set! dec_f 0)
                            (set! dec_b 0)
                            (set! need_analyze 1)
                          }
                          #f)
                    }
                    #f)
                (set! n (+ n 1))
              }
              (set! n ENC_CHUNK))
        }))
    (set! dirty 1)
  }))

;; ============================================================================
;; §9 解码器
;; ============================================================================

(define dec_frame_start
  (lambda (f) {
    (define i 0)
    (set! dec_f f)
    (set! dec_b 0)
    (set! brs (get_item bs f))
    (set! brp 0)
    (set! dfr (get_item dec_frames f))
    (set! curbuf dfr)
    (set! recbuf dfr)
    (set! reff (if (== f 0) zf (get_item dec_frames (- f 1))))
    (set! i 0)
    (while (< i NPIX) { (set_item! dfr i (get_item reff i)) (set! i (+ i 1)) })
    (set! dec_ftype (rval 1))
    (set! g_q (rval 6))
  }))

(define dec_block
  (lambda (b) {
    (define i 0)
    (define mode 0)
    (define nz 0)
    (define run 0)
    (define lev 0)
    (define sign 0)
    (define k 0)
    (define idx 0)
    (set! curbuf dfr)
    (set! recbuf dfr)
    (set! reff (if (== dec_f 0) zf (get_item dec_frames (- dec_f 1))))
    (set_bxy b)
    (if (== dec_ftype 1)
        (if (== (rbit) 0)
            (set! mode C_SKIP)
            (if (== (rbit) 1)
                {
                  (set! mode C_INTER)
                  (set! g_mvx (- (rval B_MV) SR))
                  (set! g_mvy (- (rval B_MV) SR))
                }
                (set! mode (rval B_IMODE))))
        (set! mode (rval B_IMODE)))
    ;; 残差
    (set! k 0)
    (while (< k 64) { (set_item! qb k 0) (set! k (+ k 1)) })
    (set! nz 0)
    (if (not (== mode C_SKIP))
        {
          (set! nz (rval B_NZ))
          (set! idx 0)
          (set! k 0)
          (while (< k nz) {
            (set! run (- (rgamma) 1))
            (set! lev (rgamma))
            (set! sign (rval 1))
            (set! idx (+ idx run))
            (set_item! qb (get_item ZIG (clampi idx 0 63))
                       (if (== sign 1) (- 0 lev) lev))
            (set! idx (+ idx 1))
            (set! k (+ k 1))
          })
        }
        #f)
    ;; 重建
    (if (== mode C_SKIP)
        (pred_inter 0 0)
        (if (== mode C_INTER)
            (pred_inter g_mvx g_mvy)
            (pred_intra mode)))
    (if (== nz 0)
        { (set! i 0) (while (< i 64) { (set_item! ob i 0.0) (set! i (+ i 1)) }) }
        {
          (set! i 0)
          (while (< i 64) {
            (set_item! db i (* (get_item qb i) (* g_q (get_item QW i))))
            (set! i (+ i 1))
          })
          (idct)
        })
    (store_block)
  }))

;; 与编码器重建帧逐像素比对（码流回环校验）
(define dec_frame_end
  (lambda (f) {
    (define i 0)
    (define a 0.0)
    (define b 0.0)
    (set! i 0)
    (while (< i NPIX) {
      (set! a (get_item dfr i))
      (set! b (get_item (get_item rec_frames f) i))
      (if (not (== a b))
          { (set! rt_mismatch (+ rt_mismatch 1))
            (set! _f (absf (- a b)))
            (if (> _f _mx) { (set! _mx _f) (set! _mi i) } #f)
            }
          #f)
      (set! i (+ i 1))
    })
    (set! rt_checked (+ rt_checked NPIX))
    (if (== dbg 1)
        { (display "[vcodec] dec f=") (display f) (display " mm=")
          (display (+ (- rt_mismatch _mm0) 0))
          (display " maxdiff=") (display _mx) (newline)
          (set! _mx 0.0) }
        #f)
    (set! _mm0 rt_mismatch)
    (set! _mmn 0)
  }))
(define _mm0 0)
(define _mx 0.0)
(define _mi 0)
(define _mmn 0)

(define dec_step
  (lambda () {
    (define n 0)
    (if (>= dec_f NFR)
        {
          (set! phase PH_READY)
          (set! playing 1)
          (set! play_t0 (System.timestamp))
          (set! play_idx 0)
          (set! view_f (minf 1 (- NFR 1)))
          (set! need_analyze 1)
          (display "[vcodec] done bits=") (display (total_bits))
          (display " bytes=") (display (/ (total_bits) 8))
          (display " mismatch=") (display rt_mismatch)
          (display " enc_ms=") (display enc_ms) (newline)
        }
        {
          (while (< n DEC_CHUNK) {
            (if (< dec_f NFR)
                {
                  (if (== dec_b 0) (dec_frame_start dec_f) #f)
                  (dec_block dec_b)
                  (set! dec_b (+ dec_b 1))
                  (if (>= dec_b NB)
                      {
                        (dec_frame_end dec_f)
                        (set! dec_f (+ dec_f 1))
                        (set! dec_b 0)
                      }
                      #f)
                }
                #f)
            (set! n (+ n 1))
          })
          (set! dirty 1)
        })
  }))

;; ============================================================================
;; §10 选中块分析
;; ============================================================================

;; 选一个「有内容」的块：当前帧失真最大的块（全跳过的块看不出东西）
;;   who=1 帧内面板：失真最大的块（预测最难的地方）
;;   who=2 帧间面板：运动矢量最大的块（动得最厉害的地方）
;;   who=3 变换面板：非零系数最多的块（频谱最丰富的地方）
;;   其它        ：失真最大的块
(define pick_interesting
  (lambda (f who) {
    (define b 0)
    (define bb 0)
    (define bd -1.0)
    (define d 0.0)
    (set! b 0)
    (while (< b NB) {
      (set! d (if (== who 2)
                  (+ (absf (get_item (get_item fr_mvx f) b))
                     (absf (get_item (get_item fr_mvy f) b)))
                  (if (== who 3)
                      (get_item (get_item fr_bnz f) b)
                      (get_item (get_item fr_bd f) b))))
      (if (> d bd) { (set! bd d) (set! bb b) } #f)
      (set! b (+ b 1))
    })
    bb
  }))

(define analyze_block
  (lambda () {
    (define f view_f)
    (define b sel_b)
    (define i 0)
    (define mode 0)
    (define dx 0)
    (define dy 0)
    (define s 0.0)
    (define best 0.0)
    (if (== auto_sel 1) (set! sel_b (pick_interesting f panel)) #f)
    (set! b sel_b)
    (load_frame_into f asrc)
    (set! src_f f)
    (set! curbuf (get_item rec_frames f))
    (set! recbuf afr)                    ;; 不要污染存档的重建帧
    (set! reff (if (== f 0) zf (get_item rec_frames (- f 1))))
    (set! g_q (get_item fr_q f))
    (set! an_ftype (get_item fr_type f))
    (set_bxy b)
    (gather_src asrc)
    ;; 全搜索 SAD 地形（9×9，全像素）
    (set! best 1.0e30)
    (set! dy (- 0 SR))
    (set! i 0)
    (while (<= dy SR) {
      (set! dx (- 0 SR))
      (while (<= dx SR) {
        (set! s (sad_mv dx dy 1))
        (set_item! an_sad i s)
        (if (< s best) { (set! best s) (set! an_sadbest i)
                         (set! an_bestmvx0 dx) (set! an_bestmvy0 dy) } #f)
        (set! i (+ i 1))
        (set! dx (+ dx 1))
      })
      (set! dy (+ dy 1))
    })
    ;; 候选枚举（与编码器同一条代码路径，候选集也一致）
    (set! NCAND 0)
    (if (== an_ftype 1)
        {
          (pred_inter 0 0)
          (build_res)
          (if (== g_zero 1)
              (set! NCAND 0)
              {
                (me_search)
                (eval_cand 0 C_SKIP 1)
                (set! _ihdr B_INTER_HDR)
                (eval_cand 1 C_INTER _ihdr)
                (set! _imi (preselect_intra))
                (eval_cand 2 _imi B_IMODE)
                (set! NCAND 3)
              })
        }
        {
          (set! mode 0)
          (while (< mode 4) {
            (eval_cand NCAND mode B_IMODE)
            (set! NCAND (+ NCAND 1))
            (set! mode (+ mode 1))
          })
        })
    (if (== NCAND 0)
        { (eval_cand 0 C_SKIP 1) (set! NCAND 1) }
        #f)
    (pick_best)
    (reconstruct)
    ;; 候选表快照
    (set! AN_NC NCAND)
    (set! AN_BEST g_best)
    (set! i 0)
    (while (< i 6) {
      (set_item! AN_D i (get_item CAND_D i))
      (set_item! AN_R i (get_item CAND_R i))
      (set_item! AN_J i (get_item CAND_J i))
      (set_item! AN_N i (get_item CAND_N i))
      (set_item! AN_M i (get_item CAND_M i))
      (set! i (+ i 1))
    })
    (set! i 0)
    (while (< i 384) {
      (set_item! ANC i (get_item CAND_C i))
      (set_item! ANQ i (get_item CAND_Q i))
      (set! i (+ i 1))
    })
    (set! an_mvx g_mvx)
    (set! an_mvy g_mvy)
  }))

;; ============================================================================
;; §11 绘制基元
;; ============================================================================

;; 帧 → 灰度等级（0..255）写进 lvb
(define to_levels
  (lambda (buf lo hi) {
    (define i 0)
    (define v 0.0)
    (while (< i NPIX) {
      (set! v (get_item buf i))
      (set_item! lvb i (Math.round (* 255.0 (clampf (/ (- v lo) (- hi lo)) 0.0 1.0))))
      (set! i (+ i 1))
    })
  }))

;; 帧 → 灰度等级（lvb，0..255）。**缓存必须同时记住「哪一路信号」和「哪一帧」**——
;; 只记帧号会让「源帧/重建帧/解码帧」三路共用一块 lvb 而互相串味。
(define lvk_who -1)
(define lvk_f -1)
(define ensure_lv
  (lambda (who f) {
    (if (or (not (== lvk_who who)) (not (== lvk_f f)))
        {
          (if (== who 0) (ensure_src f) #f)
          (to_levels (if (== who 0) asrc (if (== who 1) (get_item rec_frames f) (get_item dec_frames f)))
                     0.0 255.0)
          (set! lvk_who who)
          (set! lvk_f f)
        }
        #f)
  }))
(define show_src_lv (lambda (f) (ensure_lv 0 f)))
(define show_rec_lv (lambda (f) (ensure_lv 1 f)))
(define show_dec_lv (lambda (f) (ensure_lv 2 f)))

;; 把 lvb 的 w×h 区域按行游程画到 (x,y)，放大 sc 倍
(define draw_lv
  (lambda (x y sc w h) {
    (define j 0)
    (define i 0)
    (define i0 0)
    (define g0 0)
    (while (< j h) {
      (set! i 0)
      (while (< i w) {
        (set! i0 i)
        (set! g0 (get_item lvb (+ (* j w) i)))
        (set! i (+ i 1))
        (while (and (< i w) (== (get_item lvb (+ (* j w) i)) g0)) (set! i (+ i 1)))
        (rc (+ x (* i0 sc)) (+ y (* j sc)) (* (- i i0) sc) sc g0 g0 g0)
      })
      (set! j (+ j 1))
    })
  }))

;; 8×8 块按灰度画（逐像素）
(define draw_blk
  (lambda (x y sc blk lo hi) {
    (define j 0)
    (define i 0)
    (define g 0)
    (while (< j 8) {
      (set! i 0)
      (while (< i 8) {
        (set! g (Math.round (* 255.0
              (clampf (/ (- (get_item blk (+ (* j 8) i)) lo) (- hi lo)) 0.0 1.0))))
        (rc (+ x (* i sc)) (+ y (* j sc)) sc sc g g g)
        (set! i (+ i 1))
      })
      (set! j (+ j 1))
    })
  }))

;; 双极配色：负→蓝，正→红（残差/系数用）
(define gcr 24) (define gcg 26) (define gcb 34)
(define blk_color
  (lambda (val vmax) {
    (define t (clampf (/ (absf val) vmax) 0.0 1.0))
    (define tt (* t t))
    (if (< val 0)
        { (set! gcr 20) (set! gcg (+ 30 (* 45 tt))) (set! gcb (+ 70 (* 175 tt))) }
        (if (> val 0)
            { (set! gcr (+ 45 (* 210 tt))) (set! gcg (+ 30 (* 110 tt))) (set! gcb (+ 30 (* 60 tt))) }
            { (set! gcr 24) (set! gcg 26) (set! gcb 34) }))
  }))

;; 非负量热图配色（SAD 地形用）：小 → 暗，大 → 亮黄
(define heat_color
  (lambda (val vmax) {
    (define t (clampf (/ val vmax) 0.0 1.0))
    (define tt (* t t))
    (set! gcr (Math.round (+ 20 (* 232 tt))))
    (set! gcg (Math.round (+ 28 (* 196 t))))
    (set! gcb (Math.round (+ 68 (* 40 t))))
  }))

(define draw_blk_bipolar
  (lambda (x y sc blk vmax) {
    (define j 0)
    (define i 0)
    (while (< j 8) {
      (set! i 0)
      (while (< i 8) {
        (blk_color (get_item blk (+ (* j 8) i)) vmax)
        (rc (+ x (* i sc)) (+ y (* j sc)) sc sc gcr gcg gcb)
        (set! i (+ i 1))
      })
      (set! j (+ j 1))
    })
  }))

;; 8×8 单元格热图（系数/电平）
(define draw_cells8
  (lambda (x y cw ch blk vmax) {
    (define j 0)
    (define i 0)
    (while (< j 8) {
      (set! i 0)
      (while (< i 8) {
        (blk_color (get_item blk (+ (* j 8) i)) vmax)
        (rc (+ x (* i cw)) (+ y (* j ch)) (- cw 1) (- ch 1) gcr gcg gcb)
        (set! i (+ i 1))
      })
      (set! j (+ j 1))
    })
    (frc x y (- (* 8 cw) 1) (- (* 8 ch) 1) 70 76 92)
  }))

;; 系数热图（转置读取）：内部 cb[u*8+v] 的 u 是**水平**频率、v 是**垂直**频率，
;; 而屏幕上的惯例是「行=垂直频率、列=水平频率」，所以这里交换下标。
(define draw_cells8t
  (lambda (x y cw ch blk vmax) {
    (define j 0)
    (define i 0)
    (while (< j 8) {
      (set! i 0)
      (while (< i 8) {
        (blk_color (get_item blk (+ (* i 8) j)) vmax)
        (rc (+ x (* i cw)) (+ y (* j ch)) (- cw 1) (- ch 1) gcr gcg gcb)
        (set! i (+ i 1))
      })
      (set! j (+ j 1))
    })
    (frc x y (- (* 8 cw) 1) (- (* 8 ch) 1) 70 76 92)
  }))

;; 水平比例条
(define bar
  (lambda (x y w h frac r g b) {
    (define ww 0)
    (rc x y w h 34 38 48)
    (set! ww (Math.round (* w (clampf frac 0.0 1.0))))
    (if (> ww 0) (rc x y ww h r g b) #f)
  }))

;; 块网格
(define draw_grid
  (lambda (x y sc r g b) {
    (define i 0)
    (while (<= i 8) {
      (rc (+ x (* i BS sc)) y 1 (* H sc) r g b)
      (rc x (+ y (* i (* BS sc))) (* W sc) 1 r g b)
      (set! i (+ i 1))
    })
  }))

;; 从大表取 64 项到 tb（供 draw_cells8 用）
(define slice64
  (lambda (big base) {
    (define i 0)
    (while (< i 64) { (set_item! tb i (get_item big (+ base i))) (set! i (+ i 1)) })
    tb
  }))
;; ============================================================================
;; §12 面板
;; ============================================================================

;; 多段字符串拼接：本语言没有变参过程，深嵌套的 String.concat 极易写错括号，
;; 统一走 (cat `("a" ,x "b")) —— 准引用构造列表，cat 顺序拼接。
(define cat
  (lambda (lst) {
    (define s "")
    (define i 0)
    (define n (length lst))
    (while (< i n) {
      (set! s (String.concat s (get_item lst i)))
      (set! i (+ i 1))
    })
    s
  }))

(define draw_tabs
  (lambda () {
    (define i 0)
    (define tw 53)
    (define x 0)
    (rc 0 0 SW TAB_H 30 34 44)
    (set! i 0)
    (while (< i 6) {
      (set! x (* i tw))
      (if (== i panel)
          {
            (rc x 0 tw TAB_H 46 62 92)
            (rc x (- TAB_H 2) tw 2 116 172 226)
            (dtc (get_item PANEL_NAMES i) (+ x (Math.floor (/ tw 2))) 1 240 246 255)
          }
          (dtc (get_item PANEL_NAMES i) (+ x (Math.floor (/ tw 2))) 1 148 156 174))
      (set! i (+ i 1))
    })
    (rc 0 (- TAB_H 1) SW 1 70 76 92)
  }))

(define draw_buttons
  (lambda () {
    (define labels (get_item BTN_LABELS panel))
    (define i 0)
    (define x 0)
    (rc 0 BY SW 19 24 27 35)
    (rc 0 BY SW 1 44 48 60)
    (set! i 0)
    (while (< i 4) {
      (set! x (+ 4 (* i 62)))
      (Gfx.rounded_rect x (+ BY 2) 60 (- BH 3) 4 4 4 4 46 52 66 1)
      (Gfx.rounded_rect (+ x 1) (+ BY 3) 58 (- BH 5) 3 3 3 3 34 39 51 1)
      (dtc (get_item labels i) (+ x 30) (+ BY 5) 226 232 242)
      (set! i (+ i 1))
    })
    (EX.draw_button_at (- SW 50) (+ BY 2))
  }))

(define draw_progress
  (lambda (x y w h frac r g b) {
    (rc x y w h 40 44 56)
    (bar x y w h frac r g b)
    (frc x y w h 80 86 104)
  }))

;; PSNR（二值域 0..1：MSE = SSE/NPIX）
(define psnr_str
  (lambda (sse) {
    (define mse (/ sse NPIX))
    (define p 0.0)
    (if (<= mse 0.0) "无穷"
      {
        (set! p (* 10.0 (Math.log10 (/ 65025.0 (maxf 0.001 mse)))))
        (if (> p 99.0) ">99" (fnum p 1))
      })
  }))

(define total_bits
  (lambda () {
    (define i 0)
    (define s 0)
    (set! i 0)
    (while (< i NFR) { (set! s (+ s (get_item fr_bits i))) (set! i (+ i 1)) })
    s
  }))

(define cand_name
  (lambda (m) (cond ((== m 0) "DC")
                    ((== m 1) "垂直")
                    ((== m 2) "水平")
                    ((== m 3) "平面")
                    ((== m 4) "帧间")
                    (else "跳过"))))

;; 候选代价表（面板 1）
(define draw_cand_table
  (lambda (x y w rowh) {
    (define k 0)
    (define jmax 1.0)
    (define j 0.0)
    (define _y 0)
    (set! k 0)
    (while (< k AN_NC) {
      (if (> (get_item AN_J k) jmax) (set! jmax (get_item AN_J k)) #f)
      (set! k (+ k 1))
    })
    (set! k 0)
    (while (< k AN_NC) {
      (set! j (get_item AN_J k))
      (set! _y (+ y (* k rowh)))
      (if (== k AN_BEST) (rc x (- _y 1) w rowh 44 56 78) #f)
      (if (== k AN_BEST)
          (ty (cand_name (get_item AN_M k)) x (+ _y 1))
          (tn (cand_name (get_item AN_M k)) x (+ _y 1)))
      (tcy (cat `("D" ,(knum (get_item AN_D k)))) (+ x 32) (+ _y 1))
      (tor (cat `("R" ,(inum (get_item AN_R k)))) (+ x 84) (+ _y 1))
      (tn (cat `("J" ,(knum j))) (+ x 122) (+ _y 1))
      (bar (+ x 172) (+ _y 2) (- w 174) 7 (/ (Math.sqrt j) (Math.sqrt jmax))
           (if (== k AN_BEST) 245 216 96 116 172 226)
           (if (== k AN_BEST) 200 170 60 80 120 170)
           (if (== k AN_BEST) 90 78 30 50 76 110))
      (set! k (+ k 1))
    })
  }))

;; ---- 面板 0：总览 ----
(define panel0
  (lambda () {
    (define sc 3)
    (define vx 4)
    (define vy 19)
    (define i 0)
    (define f view_f)
    (define b 0)
    (define m 0)
    (define mxb 1.0)
    (define isI 0)
    (show_src_lv f)
    (draw_lv vx vy sc W H)
    (if (and (== phase PH_ENC) (== view_f enc_f))
        {
          (set! b 0)
          (while (< b enc_b) {
            (set_bxy b)
            (frc (+ vx (* g_bx sc)) (+ vy (* g_by sc)) (* BS sc) (* BS sc) 38 46 60)
            (set! b (+ b 1))
          })
          (set_bxy (minf enc_b (- NB 1)))
          (frc (+ vx (* g_bx sc)) (+ vy (* g_by sc)) (* BS sc) (* BS sc) 245 216 96)
        }
        {
          (set! b 0)
          (while (< b NB) {
            (set_bxy b)
            (set! m (get_item (get_item fr_modes f) b))
            (if (== m C_SKIP)
                (frc (+ vx (* g_bx sc)) (+ vy (* g_by sc)) (* BS sc) (* BS sc) 44 50 62)
                (if (== m C_INTER)
                    (frc (+ vx (* g_bx sc)) (+ vy (* g_by sc)) (* BS sc) (* BS sc) 52 110 84)
                    (frc (+ vx (* g_bx sc)) (+ vy (* g_by sc)) (* BS sc) (* BS sc) 62 96 142)))
            (set! b (+ b 1))
          })
        })
    (set_bxy sel_b)
    (frc (+ vx (* g_bx sc)) (+ vy (* g_by sc)) (* BS sc) (* BS sc) 245 216 96)
    (frc (- vx 1) (- vy 1) (+ (* W sc) 2) (+ (* H sc) 2) 70 76 92)

    (set! isI (if (== (get_item fr_type f) 0) 1 0))
    (tn (cat `("帧 " ,(inum (+ f 1)) "/" ,(inum NFR))) 202 19)
    (if (== isI 1) (tcy "I 关键帧" 202 31) (tg "P 预测帧" 202 31))
    (td (cat `("Q=" ,(inum (get_item fr_q f)) "  λ=" ,(fnum LAMBDA 1))) 202 43)
    (tn (cat `("比特 " ,(inum (get_item fr_bits f)))) 202 55)
    (tn (cat `("bpp " ,(fnum (/ (get_item fr_bits f) NPIX) 3))) 202 67)
    (tn (cat `("PSNR " ,(psnr_str (get_item fr_sse f)) " dB")) 202 79)
    (tn (cat `("误像素 " ,(inum (get_item fr_err f)))) 202 91)

    (td "每帧码率 (bit)" 202 104)
    (set! i 0)
    (while (< i NFR) {
      (set! mxb (minf 1.0 (/ (get_item fr_bits i) 9000.0)))
      (set! _n (Math.round (* 40 mxb)))
      (if (> _n 0)
          (if (== (get_item fr_type i) 0)
              (rc (+ 202 (* i 5)) (- 154 _n) 4 _n 232 176 84)
              (rc (+ 202 (* i 5)) (- 154 _n) 4 _n 104 156 214))
          #f)
      (set! i (+ i 1))
    })
    (rc 202 155 114 1 70 76 92)
    (frc (+ 201 (* f 5)) 114 6 42 245 216 96)

    (ta (cat `("I " ,(inum (get_item fr_ni f)))) 202 162)
    (tg (cat `("P " ,(inum (get_item fr_np f)))) 234 162)
    (td (cat `("跳 " ,(inum (get_item fr_ns f)))) 266 162)

    (if (== phase PH_ENC)
        {
          (ty (cat `("编码中 " ,(inum (+ 1 enc_f)) "/" ,(inum NFR))) 202 176)
          (draw_progress 202 188 114 6 (/ (+ (* enc_f NB) enc_b) (* NFR NB)) 245 216 96)
          (td (cat `("Q=" ,(inum QSTEP) "  λ=" ,(fnum LAMBDA 0))) 202 200)
        }
        (if (== phase PH_DEC)
            {
              (ta (cat `("解码验证 " ,(inum dec_f) "/" ,(inum NFR))) 202 176)
              (draw_progress 202 188 114 6 (/ (+ (* dec_f NB) dec_b) (* NFR NB)) 116 172 226)
            }
            {
              (tg (cat `("码流 " ,(knum (total_bits)) " bit")) 202 176)
              (tg (cat `("压缩比 " ,(fnum (/ (* NPIX (* 8 NFR)) (maxf 1 (total_bits))) 1)
                         " : 1")) 202 188)
              (td (cat `("耗时 " ,(fnum (/ enc_ms 1000.0) 1) "s"
                         ,(if (== rt_mismatch 0) "  回环✓" ""))) 202 200)
            }))
  }))

;; ---- 面板 1：帧内预测 ----
(define panel1
  (lambda () {
    (define sc 7)
    (define y0 30)
    (define f view_f)
    (ensure_src f)
    (td "原块" 4 18)
    (tcy "帧内预测" 62 18)
    (tor "残差" 120 18)
    (tg "重建" 178 18)
    (draw_blk 4 y0 sc sb 0.0 255.0)
    (frc 4 y0 (* 8 sc) (* 8 sc) 70 76 92)
    (draw_blk 62 y0 sc an_pred 0.0 255.0)
    (frc 62 y0 (* 8 sc) (* 8 sc) 70 76 92)
    (draw_blk_bipolar 120 y0 sc an_res 256.0)
    (frc 120 y0 (* 8 sc) (* 8 sc) 70 76 92)
    (draw_blk 178 y0 sc an_recon 0.0 255.0)
    (frc 178 y0 (* 8 sc) (* 8 sc) 70 76 92)
    ;; 帧缩略图 + 选中块
    (show_rec_lv f)
    (draw_lv 244 19 1 W H)
    (set_bxy sel_b)
    (frc (+ 244 g_bx) (+ 19 g_by) BS BS 245 216 96)
    (frc 243 18 (+ W 2) (+ H 2) 70 76 92)
    ;; 候选表
    (td (cat `("候选（J = D + λR，λ=" ,(fnum LAMBDA 1) "）")) 4 94)
    (draw_cand_table 4 106 312 11)
    ;; 结论
    (set_bxy sel_b)
    (tn (cat `("块#" ,(inum sel_b) " (" ,(inum g_bx) "," ,(inum g_by) ")")) 4 176)
    (if (== (get_item AN_M AN_BEST) C_INTER)
        (tg (cat `("选中：" ,(cand_name (get_item AN_M AN_BEST)))) 82 176)
        (tcy (cat `("选中：" ,(cand_name (get_item AN_M AN_BEST)))) 82 176))
    (tn (cat `("D=" ,(fnum an_d 1) "  R=" ,(inum an_bits) " 位  nz=" ,(inum an_nz))) 4 189)
    (td (cat `("邻域：上 " ,(if (== an_hastop 1) "有" "无")
               "   左 " ,(if (== an_hasleft 1) "有" "无")
               "   （帧内预测只用已重建像素）")) 4 202)
  }))

;; ---- 面板 2：帧间预测 ----
(define panel2
  (lambda () {
    (define f view_f)
    (define isP (if (== (get_item fr_type f) 1) 1 0))
    (define i 0)
    (define b 0)
    (define mvx 0)
    (define mvy 0)
    (define cx 0)
    (define cy 0)
    (show_src_lv f)
    (draw_lv 4 30 2 W H)
    (frc 3 29 (+ (* W 2) 2) (+ (* H 2) 2) 70 76 92)
    (td "当前帧(源)+运动矢量" 4 18)
    (if (== isP 1)
        {
          (set! b 0)
          (while (< b NB) {
            (set! mvx (get_item (get_item fr_mvx f) b))
            (set! mvy (get_item (get_item fr_mvy f) b))
            (set_bxy b)
            (set! cx (+ 4 (* (+ g_bx 4) 2)))
            (set! cy (+ 30 (* (+ g_by 4) 2)))
            (if (not (and (== mvx 0) (== mvy 0)))
                (Gfx.draw_line cx cy (+ cx (* mvx 2)) (+ cy (* mvy 2)) 245 170 80)
                #f)
            (set! b (+ b 1))
          })
        }
        (dtc "I 帧：全部帧内编码，无运动矢量" 68 120 245 216 96))
    ;; 参考帧
    (tcy "参考帧" 136 18)
    (show_rec_lv (if (== f 0) f (- f 1)))
    (draw_lv 136 30 1 W H)
    (frc 135 29 (+ W 2) (+ H 2) 70 76 92)
    (set_bxy sel_b)
    (frc (+ 136 g_bx) (+ 30 g_by) BS BS 245 216 96)
    (if (== isP 1)
        {
          (set! cx (+ 136 (clampi (+ g_bx an_mvx) 0 (- W 1))))
          (set! cy (+ 30 (clampi (+ g_by an_mvy) 0 (- H 1))))
          (frc cx cy BS BS 110 214 150)
        }
        #f)
    ;; MC 预测块
    (tcy "运动补偿预测块" 136 100)
    (draw_blk 136 112 6 an_pred 0.0 255.0)
    (frc 136 112 (* 8 6) (* 8 6) 70 76 92)
    ;; SAD 地形
    (td "全搜索 SAD 地形 9×9" 204 18)
    (set! i 0)
    (while (< i 81) {
      (heat_color (get_item an_sad i) 12000.0)
      (rc (+ 206 (* (Math.round (mod i 9)) 12)) (+ 32 (* (Math.floor (/ i 9)) 12)) 11 11 gcr gcg gcb)
      (set! i (+ i 1))
    })
    (frc 206 32 107 107 70 76 92)
    (rc (+ 206 (* (Math.round (mod an_sadbest 9)) 12)) (+ 32 (* (Math.floor (/ an_sadbest 9)) 12)) 11 11 245 246 255)
    (if (== isP 1)
        {
          (set! cx (+ 206 (* (+ an_mvx SR) 12)))
          (set! cy (+ 32 (* (+ an_mvy SR) 12)))
          (frc cx cy 12 12 245 216 96)
        }
        #f)
    (td "白=全搜索最优 黄=快搜结果" 204 146)
    ;; 结论
    (if (== isP 1)
        {
          (ty (cat `("MV=(" ,(inum an_mvx) "," ,(inum an_mvy) ")   全搜索最优 ("
                     ,(inum an_bestmvx0) "," ,(inum an_bestmvy0) ")")) 4 163)
          (td (cat `("SAD(MV)=" ,(inum (get_item an_sad (+ (* (+ an_mvy SR) 9) (+ an_mvx SR))))
                     "   零矢量 SAD=" ,(inum (get_item an_sad 40)))) 4 175)
        }
        (ty "I 帧：整帧帧内编码，无运动矢量" 4 163))
    (tn (cat `("本块选中 " ,(cand_name (get_item AN_M AN_BEST))
               "   D=" ,(inum an_d) "  R=" ,(inum an_bits) " 位")) 4 189)
    (td "MV 花 8 位（mvx4+mvy4）；SKIP 只花 1 位" 4 202)
    (td "白框=全搜索最优  黄框=两阶段快搜结果" 4 212)
  }))

;; ---- 面板 3：变换编码 ----
(define panel3
  (lambda () {
    (define k 0)
    (define s "")
    (define zz 0)
    (td "残差块" 4 18)
    (draw_blk_bipolar 4 30 9 an_res 256.0)
    (frc 4 30 71 71 70 76 92)
    (tcy "DCT 系数" 84 18)
    (draw_cells8t 84 30 9 9 an_coef 512.0)
    (ty (cat `("量化电平 Q=" ,(inum g_q))) 164 18)
    (draw_cells8t 164 30 9 9 (slice64 ANQ (* AN_BEST 64)) 64.0)
    (tn (cat `("非零 " ,(inum an_nz) "/63")) 246 30)
    (tor (cat `("码字 " ,(inum an_bits) " 位")) 246 42)
    (td "DC 携带" 246 58)
    (td "块均值" 246 70)
    (td "AC 多为 0" 246 82)
    (td "游程吃零串" 246 94)
    ;; 之字形顺序
    (td "之字形序" 4 108)
    (set! k 0)
    (while (< k 64) {
      (set! zz (get_item ZIGINV k))
      (heat_color zz 63.0)
      (rc (+ 4 (* (Math.round (mod k 8)) 8)) (+ 120 (* (Math.floor (/ k 8)) 8)) 7 7 gcr gcg gcb)
      (set! k (+ k 1))
    })
    (frc 4 120 63 63 70 76 92)
    ;; 游程码字
    (td "游程码字 (run,level)" 84 108)
    (rle_scan ANQ (* AN_BEST 64))
    (set! s "")
    (set! k 0)
    (while (< k g_nz) {
      (if (< (String.length s) 40)
          (set! s (String.concat s (cat `("(" ,(inum (get_item rle_run k))
                                          "," ,(inum (get_item rle_lvl k)) ")"))))
          #f)
      (set! k (+ k 1))
    })
    (if (== g_nz 0) (set! s "全零 —— 一个系数都不发") #f)
    (tn s 72 122)
    (td (cat `("共 " ,(inum g_nz) " 个非零系数，整块码字 " ,(inum an_bits) " 位（变长码）")) 72 136)
    (tn (cat `("DC=" ,(fnum (get_item an_coef 0) 2)
               "  电平=" ,(inum (get_item an_q 0))
               "  反量化=" ,(fnum (get_item an_deq 0) 2))) 72 152)
    (tcy (cat `("D=" ,(fnum an_d 2) "  正交变换：D 即像素域 SSE")) 72 166)
    (td (cat `("块内 RMS 误差 " ,(fnum (Math.sqrt (/ an_d 64)) 2) "/255")) 72 180)
    (tor (cat `("本块码字 " ,(inum an_bits) " 位 = " ,(fnum (/ an_bits 64) 2) " bpp")) 72 194)
    (td "之字形：DC→低频→高频，零串靠游程" 72 210)
  }))

;; ---- 面板 4：RDO ----
(define panel4
  (lambda () {
    (define k 0)
    (define jmax 1.0)
    (define j 0.0)
    (define b 0)
    (define nzb 0)
    (define xmax 1)
    (define ymax 1.0)
    (define bx0 180)
    (define by0 34)
    (define bw 132)
    (define bh 92)
    (td "候选代价 J = D + λR" 4 17)
    (set! k 0)
    (while (< k AN_NC) {
      (if (> (get_item AN_J k) jmax) (set! jmax (get_item AN_J k)) #f)
      (set! k (+ k 1))
    })
    (set! k 0)
    (while (< k AN_NC) {
      (set! j (get_item AN_J k))
      (set! _j (+ 30 (* k 22)))
      (if (== k AN_BEST) (rc 2 (- _j 1) 176 20 44 56 78) #f)
      (if (== k AN_BEST)
          (ty (cand_name (get_item AN_M k)) 4 (+ _j 2))
          (tn (cand_name (get_item AN_M k)) 4 (+ _j 2)))
      (tcy (cat `("D" ,(knum (get_item AN_D k)))) 32 (+ _j 2))
      (tor (cat `("λR" ,(knum (* LAMBDA (get_item AN_R k))))) 86 (+ _j 2))
      (tn (cat `("J" ,(knum j))) 132 (+ _j 2))
      (set! _n (Math.round (* 132 (/ (get_item AN_D k) jmax))))
      (set! _x (Math.round (* 132 (/ (* LAMBDA (get_item AN_R k)) jmax))))
      (rc 180 (+ _j 3) 132 9 30 34 44)
      (if (> _n 0) (rc 180 (+ _j 3) _n 9 62 130 190) #f)
      (if (> _x 0) (rc (+ 180 _n) (+ _j 3) _x 9 220 150 70) #f)
      (set! k (+ k 1))
    })
    (td "蓝=D  橙=λR（两条加起来才是 J）" 4 168)
    ;; 全帧块级 RD 散点
    (td "全帧块 RD 分布" 180 20)
    (rc bx0 by0 bw bh 24 27 35)
    (frc bx0 by0 bw bh 70 76 92)
    (set! b 0)
    (while (< b NB) {
      (set! nzb (get_item (get_item fr_bnz view_f) b))
      (if (> nzb xmax) (set! xmax nzb) #f)
      (if (> (get_item (get_item fr_bd view_f) b) ymax)
          (set! ymax (get_item (get_item fr_bd view_f) b))
          #f)
      (set! b (+ b 1))
    })
    (set! b 0)
    (while (< b NB) {
      (set! _x (+ bx0 (Math.round (* (- bw 6) (/ (get_item (get_item fr_bnz view_f) b) xmax)))))
      (set! _n (+ (+ by0 (- bh 6)) (- 0 (Math.round (* (- bh 10) (/ (get_item (get_item fr_bd view_f) b) ymax))))))
      (if (== b sel_b)
          (rc _x _n 4 4 245 216 96)
          (rc _x _n 2 2 110 200 240))
      (set! b (+ b 1))
    })
    (td "→ AC 非零数" (+ bx0 54) (+ by0 (+ bh 2)))
    (td "D" (+ bx0 2) (+ by0 2))
    ;; 底部
    (ty (cat `("λ=" ,(fnum LAMBDA 0) "   Q=" ,(inum QSTEP))) 4 182)
    (td "λ 大 → 偏码率；λ 小 → 偏质量" 4 194)
    (tn (cat `("本帧 " ,(inum (get_item fr_bits view_f)) " bit（"
               ,(fnum (/ (get_item fr_bits view_f) NPIX) 3) " bpp）")) 4 206)
    (tg (cat `("PSNR " ,(psnr_str (get_item fr_sse view_f)) " dB")) 200 206)
  }))

;; ---- 面板 5：解码播放 ----
(define panel5
  (lambda () {
    (define f play_idx)
    (define i 0)
    (if (>= f NFR) (set! f (- NFR 1)) #f)
    (td "源帧（编码输入）" 6 16)
    (tg "解码器重建（码流输出）" 150 16)
    (show_src_lv f)
    (draw_lv 6 28 2 W H)
    (frc 5 27 (+ (* W 2) 2) (+ (* H 2) 2) 70 76 92)
    (show_dec_lv f)
    (draw_lv 150 28 2 W H)
    (frc 149 27 (+ (* W 2) 2) (+ (* H 2) 2) 70 76 92)
    ;; 时间轴
    (td "帧时间轴（黄=I 关键帧）" 6 158)
    (set! i 0)
    (while (< i NFR) {
      (if (== (get_item fr_type i) 0)
          (rc (+ 6 (* i 9)) 170 8 12 150 122 56)
          (rc (+ 6 (* i 9)) 170 8 12 50 56 70))
      (set! i (+ i 1))
    })
    (frc (+ 4 (* f 9)) 168 12 16 245 216 96)
    (tn (cat `("帧 " ,(inum (+ f 1)) "/" ,(inum NFR) "   "
               ,(if (== (get_item fr_type f) 0) "I 帧" "P 帧"))) 6 186)
    (tn (cat `("比特 " ,(inum (get_item fr_bits f))
               "   PSNR " ,(psnr_str (get_item fr_sse f)) " dB")) 6 197)
    ;; 回环校验三态：未解码完 → 不声称「一致」（改 Q/λ 重编时 rt_mismatch 已清零，
    ;; 若只看 mismatch 会误报 ✓，而面板里的重建画面其实还是上一轮码流的；
    ;; 用 rt_checked 的累计量判定「本轮解码是否已经把全部帧走过一遍」。
    (if (< rt_checked (* NFR NPIX))
        (td "回环校验：等待本轮解码…" 6 208)
        (if (== rt_mismatch 0)
            (tg "回环校验：解码器 == 编码器 ✓" 6 208)
            (tcy (cat `("回环不一致 " ,(inum rt_mismatch) " 处")) 6 208)))
    (td (cat `("累计 " ,(knum (total_bits)) " bit")) 176 186)
    (td (cat `("原始 8bpp " ,(knum (* NPIX (* 8 NFR))) " bit")) 176 197)
    (ty (cat `("压缩比 " ,(fnum (/ (* NPIX (* 8 NFR)) (maxf 1 (total_bits))) 1)
               " : 1")) 176 208)
  }))

;; 全屏重绘
(define draw_all
  (lambda () {
    (rc 0 0 SW SH 18 20 26)
    (if (== phase PH_LOAD)
        {
          (draw_tabs)
          (if (> load_waits 100)
              {
                (dtc "缺少帧数据（请部署业务资源）" 160 96 245 170 100)
                (dtc PATH 160 116 150 158 175)
                (dtc "预期 app/lib/badapple/frames.txt" 160 132 130 138 154)
              }
              {
                (dtc "载入 Bad Apple 帧数据…" 160 100 226 232 242)
                (dtc PATH 160 118 130 138 154)
              })
          (EX.draw_button_at (- SW 50) (+ BY 2))
        }
        {
          (cond ((== panel 0) (panel0))
                ((== panel 1) (panel1))
                ((== panel 2) (panel2))
                ((== panel 3) (panel3))
                ((== panel 4) (panel4))
                (else (panel5)))
          (draw_tabs)
          (draw_buttons)
        })
    (if (not (String.equals msg ""))
        {
          (rc 46 176 228 16 40 46 62)
          (frc 46 176 228 16 90 98 120)
          (dtc msg 160 179 245 216 96)
        }
        #f)
    (Gfx.refresh)
    (set! dirty 0)
  }))

;; ============================================================================
;; §13 输入
;; ============================================================================

(define toast
  (lambda (s) {
    (set! msg s)
    (set! msg_t (System.timestamp))
    (set! dirty 1)
  }))

(define restart_encode
  (lambda () {
    (define i 0)
    ;; 清空位缓冲：任何时刻都可能重编（面板⑤的 λ±/Q±、面板①的「重编」、键盘 r 都
    ;; 直接调本函数），**包括一拍编码正走到某一帧中间**的时候。此时 bits_buf 里还留着
    ;; 被打断那一帧已写出的位（帧头 + 前 enc_b 个块），而 flush_bits 是「把整个
    ;; bits_buf 打包成 bs[f]」——残位不清掉，新一轮第 0 帧的码流就会以这些残位开头，
    ;; 解码器从第 0 帧起整条失步（rgamma 一路读越界，取 P2 越界得 #undefined →
    ;; add 报「指令执行异常」→ 虚拟机进程停机 → 整个 App 卡死）。
    ;; 必须与下面 bs 的重置同步，二者是同一件事（码流从头开始）。
    (while (> (length bits_buf) 0) (pop bits_buf))
    (set! frame_bitpos 0)
    ;; 数据未就位时**退回载入态**而不是直接开编：本函数也会被「载入画面上的改参数」
    ;; 调到（面板①②的底部按钮在载入态不画，但键盘 λ±/Q± 与 r 照旧可用）。此时源帧
    ;; 读不到，`load_frame_into` 留下的是全零缓冲——直接进 PH_ENC 就会把 20 帧**黑图**
    ;; 从头编一遍（实测 108628 bit、回环 mismatch=0，看上去一切正常），而且 phase 一旦
    ;; 离开 PH_LOAD 就再也不会去读数据（`try_load` 不再被调）⇒ 画面永远是黑片，只有
    ;; 用户再手动改一次参数才能恢复。退回 PH_LOAD 后轮询继续，数据到达时 `try_load`
    ;; 会再调本函数——于是参数生效的那一刻，仍是「从第 0 帧、带真数据、彻底重编」。
    (set! phase (if (== data_ok 1) PH_ENC PH_LOAD))
    (set! enc_f 0)
    (set! enc_b 0)
    (set! dec_f 0)
    (set! dec_b 0)
    (set! rt_mismatch 0)
    (set! rt_checked 0)
    (set! playing 0)
    (set! view_f 0)
    (set! sel_b 0)
    (set! auto_sel 1)
    (set! src_f -1)
    (set! lvk_who -1)
    (set! lvk_f -1)
    (set! need_analyze 1)
    (set! dirty 1)
    (set! enc_paused 0)
    (set! t_start (Diag.ms))
    (set! enc_ms 0)
    (set! i 0)
    (while (< i NFR) {
      (set_item! fr_bits i 0)
      (set_item! fr_sse i 0.0)
      (set_item! fr_err i 0)
      (set_item! bs i "")
      (set! i (+ i 1))
    })
    (if (== dbg 1)
        { (display "[vcodec] restart lambda=") (display LAMBDA)
          (display " Q=") (display QSTEP) (newline) }
        #f)
  }))

(define sel_frame
  (lambda (df) {
    (define nf (+ view_f df))
    (if (< nf 0) (set! nf 0) #f)
    (if (>= nf NFR) (set! nf (- NFR 1)) #f)
    (set! view_f nf)
    (set! need_analyze 1)
    (set! dirty 1)
  }))

(define sel_block
  (lambda (db) {
    (define nb (+ sel_b db))
    (set! auto_sel 0)
    (if (< nb 0) (set! nb 0) #f)
    (if (>= nb NB) (set! nb (- NB 1)) #f)
    (set! sel_b nb)
    (set! need_analyze 1)
    (set! dirty 1)
  }))

;; 画面点按 → 选块
(define pick_block_at
  (lambda (px py vx vy sc) {
    (if (hit px py vx vy (* W sc) (* H sc))
        {
          (set! auto_sel 0)
          (set! sel_b (+ (* (celli (/ (- py vy) (* BS sc)) 0 7) BW)
                         (celli (/ (- px vx) (* BS sc)) 0 7)))
          (set! need_analyze 1)
          (set! dirty 1)
        }
        #f)
  }))

;; 播放位置微调（面板 5 的 帧-/帧+）
(define sel_play
  (lambda (df) {
    (define np (+ play_idx df))
    (if (< np 0) (set! np 0) #f)
    (if (>= np NFR) (set! np (- NFR 1)) #f)
    (set! play_idx np)
    (set! playing 0)
    (set! play_t0 (- (System.timestamp) (* np 100)))
    (set! dirty 1)
  }))

;; 面板动作（底部按钮与键盘共用）。**注意 cond 的子句顺序**：
;; 通用子句必须带面板条件，否则会抢走面板 4 的 λ/Q 按钮
;;（早期就踩过：RDO 面板按 λ+ 变成了「下一帧」）。
(define do_action
  (lambda (pi ai) {
    (cond
      ;; ---- 面板 0：编码/暂停、重编、Q-/Q+ ----
      ((and (== pi 0) (== ai 0))
       (if (== phase PH_ENC)
           (if (== enc_paused 1)
               { (set! enc_paused 0) (toast "继续编码") }
               { (set! enc_paused 1) (toast "编码已暂停") })
           (if (== playing 1)
               { (set! playing 0) (toast "暂停") }
               { (set! playing 1)
                 (set! play_t0 (- (System.timestamp) (* play_idx 100)))
                 (toast "播放") })))
      ((and (== pi 0) (== ai 1)) (restart_encode) (toast "重新编码"))
      ((and (== pi 0) (== ai 2)) (do_action 4 2))
      ((and (== pi 0) (== ai 3)) (do_action 4 3))
      ;; ---- 面板 4：λ / Q ----
      ((and (== pi 4) (== ai 0))
       { (set! LAMBDA (maxf 10.0 (- LAMBDA 10)))
         (toast (cat `("λ=" ,(fnum LAMBDA 0))))
         (restart_encode) })
      ((and (== pi 4) (== ai 1))
       { (set! LAMBDA (minf 400.0 (+ LAMBDA 10)))
         (toast (cat `("λ=" ,(fnum LAMBDA 0))))
         (restart_encode) })
      ((and (== pi 4) (== ai 2))
       { (set! QSTEP (maxf QMIN (- QSTEP 2)))
         (toast (cat `("Q=" ,(inum QSTEP))))
         (restart_encode) })
      ((and (== pi 4) (== ai 3))
       { (set! QSTEP (minf QMAX (+ QSTEP 2)))
         (toast (cat `("Q=" ,(inum QSTEP))))
         (restart_encode) })
      ;; ---- 面板 1..3：帧 / 块 导航 ----
      ((and (< pi 4) (== ai 0)) (sel_frame (- 0 1)))
      ((and (< pi 4) (== ai 1)) (sel_frame 1))
      ((and (< pi 4) (== ai 2)) (sel_block (- 0 8)))
      ((and (< pi 4) (== ai 3)) (sel_block 8))
      ;; ---- 面板 5：播放 / 逐帧 / 重播 ----
      ((and (== pi 5) (== ai 0))
       (if (== playing 1)
           { (set! playing 0) (toast "已暂停") }
           { (set! playing 1)
             (set! play_t0 (- (System.timestamp) (* play_idx 100)))
             (toast "播放") }))
      ((and (== pi 5) (== ai 1)) (sel_play (- 0 1)))
      ((and (== pi 5) (== ai 2)) (sel_play 1))
      ((and (== pi 5) (== ai 3))
       { (set! play_idx 0) (set! playing 1) (set! play_t0 (System.timestamp))
         (toast "从头播放") })
      (else #f)
    )
  }))

(define on_press
  (lambda (px py) {
    (define ai 0)
    (cond
      ((< py TAB_H)
       { (set! panel (celli (/ px 53) 0 5)) (set! dirty 1) (set! need_analyze 1) })
      ((and (>= py BY) (< px 266))
       {
         (set! ai (celli (/ (- px 4) 62) 0 3))
         (do_action panel ai)
       })
      ((== panel 0) (pick_block_at px py 4 19 3))
      ((== panel 1)
       (if (hit px py 244 19 W H)
           {
             (set! sel_b (+ (* (celli (/ (- py 19) BS) 0 7) BW)
                            (celli (/ (- px 244) BS) 0 7)))
             (set! need_analyze 1) (set! dirty 1)
           }
           (if (hit px py 4 30 (* 8 7) (* 8 7))
               {
                 (set! sel_b (+ (* (celli (/ (- py 30) 7) 0 7) BW)
                                (celli (/ (- px 4) 7) 0 7)))
                 (set! need_analyze 1) (set! dirty 1)
               }
               #f)))
      ((== panel 2)
       (if (hit px py 4 30 (* W 2) (* H 2))
           (pick_block_at px py 4 30 2)
           (if (hit px py 136 30 W H)
               {
                 (set! sel_b (+ (* (celli (/ (- py 30) BS) 0 7) BW)
                                (celli (/ (- px 136) BS) 0 7)))
                 (set! need_analyze 1) (set! dirty 1)
               }
               #f)))
      ((== panel 5)
       (if (hit px py 6 30 (* W 2) (* H 2))
           (pick_block_at px py 6 30 2)
           (if (hit px py 4 174 (* NFR 9) 16)
               {
                 (set! play_idx (celli (/ (- px 6) 9) 0 (- NFR 1)))
                 (set! playing 0)
                 (set! play_t0 (- (System.timestamp) (* play_idx 100)))
                 (set! dirty 1)
               }
               #f)))
      (else #f)
    )
  }))

(define on_key
  (lambda (k) {
    (cond
      ((and (>= k 49) (<= k 54)) { (set! panel (- k 49)) (set! need_analyze 1) (set! dirty 1) })
      ((== k 3) (sel_frame (- 0 1)))
      ((== k 4) (sel_frame 1))
      ((== k 1) (sel_block (- 0 1)))
      ((== k 2) (sel_block 1))
      ((== k 13) (do_action panel 0))
      ((== k 32) (do_action panel 0))
      ((== k 114) (restart_encode))
      ((== k 91) (do_action 4 0))
      ((== k 93) (do_action 4 1))
      ((== k 44) (do_action 4 2))
      ((== k 46) (do_action 4 3))
      (else #f)
    )
  }))

(define poll_input
  (lambda () {
    (define n 0)
    (define k 0)
    (if (Touch.take_press)
        { (if (== dbg 1) { (display "[vcodec] touch ") (display (Touch.x))
                           (display ",") (display (Touch.y)) (newline) } #f)
          (on_press (Touch.x) (Touch.y)) }
        #f)
    (set! n 0)
    (while (< n 8) {
      (set! k (Key.take_press))
      (if (== k 0)
          (set! n 8)
          { (if (== dbg 1) { (display "[vcodec] key ") (display (Math.round k)) (newline) } #f)
            (on_key (Math.round k))
            (set! n (+ n 1)) })
    })
  }))

;; ============================================================================
;; §14 主循环
;; ============================================================================

(define try_load
  (lambda () {
    (define sz (Fs.size PATH))
    (set! load_waits (+ load_waits 1))
    (if (null? sz)
        0
        (if (>= (Math.round sz) (* NF REC))
            { (set! data_ok 1) (set! dirty 1) (restart_encode) }
            0))
  }))

(define _tk 0)
(define _tc 0)
(define tick
  (lambda () {
    ;; 重入守卫（见 TICK_MS 处说明）：被重入的那一份立刻返回，绝不碰任何全局状态。
    (if (> _in_tick 0)
        #f
      {
    (set! _in_tick 1)
    (set! _tc (+ _tc 1))
    (if (== dbg 1)
        (if (== (Math.round (mod _tc 20)) 0)
            { (display "[vcodec] tick ") (display _tc) (display " ph=") (display phase)
              (display " f=") (display enc_f) (display " df=") (display dec_f) (newline) }
            #f)
        #f)
    (EX.check)
    (poll_input)
    (if (== phase PH_LOAD) (try_load) #f)
    (if (== phase PH_ENC) (enc_step) #f)
    (if (== phase PH_DEC) (dec_step) #f)
    (if (and (== phase PH_READY) (== playing 1))
        {
          (set! _n (Math.floor (/ (- (System.timestamp) play_t0) 100)))
          ;; 到片尾回到开头（循环播放），与 10fps 的绝对时刻表一致
          (if (>= _n NFR) { (set! play_t0 (+ play_t0 (* 100 NFR))) (set! _n 0) } #f)
          (if (< _n 0) (set! _n 0) #f)
          (if (not (== _n play_idx)) { (set! play_idx _n) (set! dirty 1) } #f)
        }
        #f)
    (if (and (== phase PH_ENC) (== panel 0) (< enc_f NFR) (not (== view_f enc_f)))
        { (set! view_f enc_f) (set! need_analyze 1) }
        #f)
    (if (and (== need_analyze 1) (not (== phase PH_LOAD)))
        { (if (== dbg 1) { (display "[vcodec] an-begin f=") (display view_f) (newline) } #f)
          (analyze_block)
          (if (== dbg 1) { (display "[vcodec] an-end") (newline) } #f)
          (set! need_analyze 0) (set! dirty 1) }
        #f)
    (if (not (String.equals msg ""))
        (if (> (- (System.timestamp) msg_t) 1400)
            { (set! msg "") (set! dirty 1) }
            #f)
        #f)
    (if (== dbg 1) (if (== dirty 1) { (display "[vcodec] draw-begin ph=") (display phase) (newline) } #f) #f)
    ;; 每拍都重绘（不抽样）：dirty 只表示「状态变了」，变了就画 —— 任何一个中间状态都在
    ;; 下一拍上屏，而不是把一部分中间状态直接丢掉。
    (if (== dirty 1) (draw_all) #f)
    (set! _in_tick 0)
      })
  }))



(define boot
  (lambda () {
    (App.set_timeslice 1200000)
    (EX.install_with draw_all)
    (set! t_start (Diag.ms))
    ;; 周期定时器：由运行时按 TICK_MS 反复触发；tick 自己不重挂。
    (set! _timer (System.set_interval TICK_MS tick))
    (tick)
    #t
  }))

(boot)
