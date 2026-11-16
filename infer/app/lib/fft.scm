;; ============================================================================
;; fft.scm —— 纯 Scheme 的基 2 迭代 FFT（含 Hann 窗与"对数幅度 → 0~255"映射）
;;
;; 为什么要有：`(native Audio)` 的 `(Audio.read)` 在 C 侧做**加窗 + FFT + 对数映射**
;; （"硬 FFT"，1024 点只要几十微秒）。本模块把同一套算法用 Scheme 重写一遍，
;; 供 `app/spectrogram.scm` 的「软 FFT」模式使用——两者的窗函数、蝶形顺序、映射
;; 公式**逐项对齐**，因此两种模式画出来的声谱图肉眼一致（仅浮点末位有别）。
;;
;; 与 C 版对齐的三处细节（改一处就要同步 am_native_Audio.c，否则两种模式不可比）：
;;   · Hann 窗（周期型）：w[n] = 0.5 · (1 − cos(2πn/N))
;;   · 基 2 迭代 FFT（DIT，先做位反转重排），tw[k] = e^(−2πik/N)，k = 0..N/2−1
;;   · 强度映射：v = (log10(|X|+1)·80 − 基准) · 255 / 动态范围，钳到 0~255；
;;     频点从 bin 1 起（跳过直流）
;;
;; 用法（三行一列）：
;;   (import FFT "app/lib/fft.scm")
;;   (define f (FFT.make 1024))          ; 一次性建好（窗/旋转因子/位反转表）
;;   (FFT.capture f)                     ; 从 (Audio.sample k) 取一帧 → 乘窗 → re/im
;;   (FFT.transform f)                   ; 原地 FFT（N/2·log2N 个蝶形）
;;   (FFT.bin f i base range)            ; 第 i 个频点的 0~255 强度
;;
;; 性能（实测，1024 点）：wasm32/浏览器里约 10~20ms/列（= 5120 个蝶形 + 1024 次取样，
;; 每次蝶形含 4 次 get_item / 4 次 set_item）。这就是"软 FFT"与"硬 FFT"的代价差：
;; app/spectrogram.scm 在软模式下每拍只画一列、并丢掉来不及处理的帧（列率会明显低于
;; 43 列/秒，时间轴按音频帧消耗量标定，故标尺仍然诚实）。
;;
;; 数据布局：上下文是 7 元列表（字段下标见下方 F_*），内部用**定长列表**存放复数
;; （灵机列表支持 O(1) 的 get_item/set_item! 原地改写），避免每帧重新分配。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数；无 let/list/map/append；
;;   块内不 define（一律写在 lambda 体顶层）；cond 每子句只接受单个表达式；
;;   / 与 mod 恒浮点（索引处用 Math.floor 收敛）；0 是真值（布尔一律 #t/#f）。
;; ============================================================================

(native Audio)
(native Math)

;; ---- 上下文（7 元列表）字段下标 -------------------------------------------
(define F_N 0)      ;; FFT 点数 N（2 的幂）
(define F_RE 1)     ;; 实部（N 元数值列表，原地改写）
(define F_IM 2)     ;; 虚部（N 元数值列表，原地改写）
(define F_TR 3)     ;; 旋转因子实部（N/2 元）
(define F_TI 4)     ;; 旋转因子虚部（N/2 元）
(define F_WIN 5)    ;; Hann 窗（N 元）
(define F_BR 6)     ;; 位反转表（N 元）

;; ---- 小工具 ---------------------------------------------------------------

;; 空表一律现造（'() 是模块级静态槽，就地 push 不安全，见 manual §3.4）
(define elist (lambda () (cdr '(0))))

;; 造一个 n 元的数值列表（全部填 x）
(define make_zeros
  (lambda (n x)
    (define l (elist))
    (define i 0)
    (while (< i n)
      { (push l x) (set! i (+ i 1)) })
    l))

(define get_item_ (lambda (l i) (get_item l i)))
(define set_item_ (lambda (l i v) (set_item! l i v)))

;; ---- 构造：窗 + 旋转因子 + 位反转表 ---------------------------------------

;; 位反转（log2(n) 位）
(define bitrev
  (lambda (v bits)
    (define r 0)
    (define i 0)
    (while (< i bits)
      {
        (set! r (* r 2))
        (if (== 0 (Math.bit_and v 1)) #f (set! r (+ r 1)))
        (set! v (Math.bit_ashr v 1))
        (set! i (+ i 1))
      })
    r))

;; (FFT.make n) : List —— 建上下文（n 必须是 2 的幂；非法返回空表）
(define make
  (lambda (n)
    (define nn (Math.floor n))
    (define bits 0)
    (define p 1)
    (define re (elist))
    (define im (elist))
    (define tr (elist))
    (define ti (elist))
    (define win (elist))
    (define br (elist))
    (define i 0)
    (define a 0.0)
    ;; 校验：n 为 2 的幂
    (while (< p nn) { (set! p (* p 2)) (set! bits (+ bits 1)) })
    (if (not (== p nn))
        (elist)
        {
          (set! re (make_zeros nn 0.0))
          (set! im (make_zeros nn 0.0))
          (set! tr (make_zeros (Math.floor (/ nn 2)) 0.0))
          (set! ti (make_zeros (Math.floor (/ nn 2)) 0.0))
          (set! win (make_zeros nn 0.0))
          (set! br (make_zeros nn 0))
          (set! i 0)
          (while (< i nn)
            {
              ;; Hann 窗（周期型）与 C 版逐字一致
              (set_item_ win i
                         (* 0.5 (- 1.0 (Math.cos (* 6.283185307179586 (/ i nn))))))
              (set_item_ br i (bitrev i bits))
              (set! i (+ i 1))
            })
          (set! i 0)
          (while (< i (Math.floor (/ nn 2)))
            {
              (set! a (* -6.283185307179586 (/ i nn)))
              (set_item_ tr i (Math.cos a))
              (set_item_ ti i (Math.sin a))
              (set! i (+ i 1))
            })
          (cons nn (cons re (cons im (cons tr (cons ti (cons win (cons br (elist))))))))
        })))

;; ---- 取样 + 加窗 ----------------------------------------------------------

;; (FFT.capture ctx) : Boolean —— 从 (Audio.sample k) 取一帧、乘 Hann 窗填入 re/im。
;; 直接读 native 采样（而不是让调用方传回调）：1024 次闭包调用在本解释器里比
;; 1024 次 native 调用贵得多。换音源时只需改这一个函数。
(define capture
  (lambda (ctx)
    (define n (get_item_ ctx F_N))
    (define re (get_item_ ctx F_RE))
    (define im (get_item_ ctx F_IM))
    (define win (get_item_ ctx F_WIN))
    (define i 0)
    (if (null? ctx)
        #f
        {
          (while (< i n)
            {
              (set_item_ re i (* (Audio.sample i) (get_item_ win i)))
              (set_item_ im i 0.0)
              (set! i (+ i 1))
            })
          #t
        })))

;; ---- 原地基 2 迭代 FFT（DIT） ---------------------------------------------

(define transform
  (lambda (ctx)
    (define n (get_item_ ctx F_N))
    (define re (get_item_ ctx F_RE))
    (define im (get_item_ ctx F_IM))
    (define tr (get_item_ ctx F_TR))
    (define ti (get_item_ ctx F_TI))
    (define br (get_item_ ctx F_BR))
    (define i 0)
    (define j 0)
    (define tmp 0.0)
    (define len 2)
    (define half 1)
    (define step 1)
    (define blk 0)
    (define k 0)
    (define tw 0)
    (define wr 0.0)
    (define wi 0.0)
    (define p 0)
    (define q 0)
    (define ar 0.0) (define ai 0.0)
    (define brr 0.0) (define bii 0.0)
    (define t1 0.0) (define t2 0.0)
    (if (null? ctx)
        #f
        {
          ;; 1) 位反转重排（只在 j > i 时交换，每对只动一次）
          (set! i 0)
          (while (< i n)
            {
              (set! j (get_item_ br i))
              (if (> j i)
                  {
                    (set! tmp (get_item_ re i)) (set_item_ re i (get_item_ re j)) (set_item_ re j tmp)
                    (set! tmp (get_item_ im i)) (set_item_ im i (get_item_ im j)) (set_item_ im j tmp)
                  }
                  #f)
              (set! i (+ i 1))
            })
          ;; 2) 蝶形：长度 2,4,8,…,N
          (set! len 2)
          (while (<= len n)
            {
              (set! half (Math.floor (/ len 2)))
              (set! step (Math.floor (/ n len)))
              (set! blk 0)
              (while (< blk n)
                {
                  (set! k 0)
                  (while (< k half)
                    {
                      (set! tw (* k step))
                      (set! wr (get_item_ tr tw))
                      (set! wi (get_item_ ti tw))
                      (set! p (+ blk k))
                      (set! q (+ (+ blk k) half))
                      (set! brr (get_item_ re q))
                      (set! bii (get_item_ im q))
                      (set! t1 (- (* brr wr) (* bii wi)))
                      (set! t2 (+ (* brr wi) (* bii wr)))
                      (set! ar (get_item_ re p))
                      (set! ai (get_item_ im p))
                      (set_item_ re q (- ar t1))
                      (set_item_ im q (- ai t2))
                      (set_item_ re p (+ ar t1))
                      (set_item_ im p (+ ai t2))
                      (set! k (+ k 1))
                    })
                  (set! blk (+ blk len))
                })
              (set! len (* len 2))
            })
          #t
        })))

;; ---- 强度映射（与 am_native_Audio.c 的 AUDIO_LOG_SCALE 同一式） -----------

;; (FFT.magnitude ctx i) : Number —— 第 i 个频点（0 起，跳过直流）的幅度 |X|
(define magnitude
  (lambda (ctx i)
    (define re (get_item_ ctx F_RE))
    (define im (get_item_ ctx F_IM))
    (define m 0.0)
    (set! m (get_item_ re (+ i 1)))
    (set! m (+ (* m m) (* (get_item_ im (+ i 1)) (get_item_ im (+ i 1)))))
    (Math.sqrt m)))

;; (FFT.bin ctx i base range) : Number —— 第 i 个频点的 0~255 强度
;;   v = (log10(|X|+1)·80 − base) · 255 / range，钳到 0~255
(define bin
  (lambda (ctx i base range)
    (define v 0.0)
    (set! v (* 80.0 (Math.log10 (+ (magnitude ctx i) 1.0))))
    (set! v (/ (* (- v base) 255.0) range))
    (if (< v 0.0) (set! v 0.0) #f)
    (if (> v 255.0) (set! v 255.0) #f)
    (Math.floor v)))
