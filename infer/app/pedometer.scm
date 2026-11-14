;; ============================================================================
;; 计步器（Pedometer）—— 灵机应用程序
;;
;; 参考实现：infer/ui_pedometer.c（C 版，STATE_PEDOMETER）。本 App 是同一算法/同一
;; 界面的 Scheme 重做，只针对 **WASM-GUI 目标**（infer/build_wasm_gui.sh →
;; infer/webgui/，不是 infer/web/ 的 LLM-only 路线），并补齐触屏交互。
;;
;; 算法（逐条对应 C 版，常量也照抄）：
;;   1. 100Hz 内部采样网格：三轴加速度模值 |a| → DC-blocker 高通去重力（R=0.95）→
;;      EMA 低通平滑（α=0.15）。网格用「逻辑时钟追赶」实现：宿主定时器每拍可能跨过
;;      几个 10ms 槽，就把当前加速度值零阶保持补进每个槽（C 版是每 10ms 直接读一次 IMU；
;;      浏览器传感器实际约 20Hz 更新，零阶保持与 C 版语义等价）。
;;   2. 时域候选点：自适应阈值（滑动 2s 窗 均值 + 0.3×标准差，下限 0.015g）且满足
;;      最小间隔 250ms 即记为一步候选（宽条件）。
;;   3. 频域交叉验证：每 1s 对最近 2.56s（256 点 @100Hz）去直流、加 Hann 窗做 FFT，
;;      求 1..10 号 bin（0.39~3.9Hz）的频谱平坦度 SFM=几何均值/算术均值，并取主频；
;;      候选速率与主频在相对容差 0.3（下限 0.3Hz）内一致且 SFM<0.3 时，本窗口的候选点
;;      计入总步数，否则判为无规律抖动全部丢弃。
;;   4. 显示：总步数（16px 大字）、时域步频 / 主频 / SFM / 状态（行走/未确认/静止）、
;;      实时波形（最近 160 点，每点 2px）与峰值标记（橙色竖线）。
;;
;; 与 C 版的差异（都是本宿主环境决定的，写在这里便于回看）：
;;   ① **加速度通道**：C 版 `imu_read_angle(&ax,&ay,&az)` 在 M5Core2 的 HAL 里实际写入的
;;      是加速度计三轴（g），所以它能直接拿模值。而 WASM/Linux/web 的 `imu_read_angle`
;;      是真正的欧拉角（度）——从角度还原不出「步伐引起的模值起伏」。故本轮给 HAL 增加
;;      `imu_read_accel`（g），并在 Imu 本地库暴露 `(Imu.accel_x/y/z/accel_mag)`；
;;      本 App 用 `(Imu.accel_mag)`，与 C 版 `sqrtf(ax²+ay²+az²)` 同义。
;;      无该通道的平台（串口 IMU/无 IMU 桩）返回 #null，App 显示「无加速度数据」。
;;   ② **节拍**：宿主每轮事件循环对同一进程只触发一个定时器回调（见 app/lib/exit.scm
;;      的注释），所以外部节拍取 50ms（≈20Hz）而不是 100Hz；内部仍按 100Hz 网格补采，
;;      于是 FFT 的点数/分辨率与 C 版完全一致（256 点、0.39Hz/bin）。
;;   ③ **时钟**：32 位宿主（wasm32）的 `System.timestamp` 是纪元毫秒 mod 2^27，会回绕。
;;      本 App 只取相邻两拍的差值（负数按 +2^27 校正后钳到 250ms）累加出**自己的逻辑
;;      时钟**，所有节流/窗口判定都用逻辑时钟，跨回绕也正确（C 版直接比较宿主时间戳）。
;;   ④ **交互**：Scheme App 没有宿主页眉/页脚与 D 键语义，本 App 自绘页眉（标题 +
;;      右上角「退出」走 app/lib/exit.scm 的全局确认框）、底部「清零」按钮；键盘
;;      D / Enter 仍可清零（与 C 版 D 键同义）。触屏按钮采用「按下武装、同一按钮内
;;      松手才生效」的既定范式（拖出取消）。
;;   ⑤ **诊断行**：每 2s 向 stdout 打一行 `[pedometer] ...`（步数/候选/步频/主频/SFM/
;;      确认标志/数据源/累计样本数），供宿主回归台与浏览器 e2e 做数值断言（不做 OCR）。
;;
;; 算法上的两处必要修正（保留「时域候选 + 频域校验」的整体结构，只改会直接导致
;; 计数错误的两个局部；理由与实测见 doc/AGENTS.md 的本次条目）：
;;   A. **候选点改为“上穿阈值的上升沿”**（加迟滞：信号回落到阈值以下才重新武装）。
;;      C 版只看 `sig > 阈值`，而阈值只有 0.3×标准差，信号在一个振荡周期里有相当长的
;;      时间都在阈值之上——慢步频时同一周期会反复触发候选点（实测 0.78Hz 注入下
;;      cad_hz 被抬到 2.34Hz），频域校验反而把整窗判成“速率对不上”而丢弃：
;;      10s 只计 1 步。加迟滞后同一周期至多一个候选点（本版同条件 8 步）。
;;   B. **交叉验证的速率改用 cad_hz**（最近若干步间隔的均值）。C 版用「本窗口候选数/1s」
;;      量化到 1Hz：步频 1.78Hz 时每窗口只有 1 个候选，估计值 1.0 与主频 1.95 差 0.95，
;;      超过容差 0.3 → 一半窗口被当成“抖动”丢弃（实测 10s 只计 10 步，而实际约 18 步）。
;;      间隔不足 2 个时退回 C 版的窗口计数。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数、无一元取负（写 (- 0 x)）；
;; `/` 与 `mod` 恒返回浮点（下标一律 Math.floor）；无 let/list/map/append/case/do/for；
;; `cond` 每子句只接受单个表达式；块 `{...}` 内不 define（define 只出现在 lambda 体
;; 顶部或模块顶层）；`'()` 是模块级静态槽（空表一律用 (cdr (cons 0 '())) 现造）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Imu)        ; (Imu.accel_mag)：三轴含重力加速度模值（g）
(native Key)        ; 键盘 D / Enter 清零
(native Touch)      ; 触屏「清零」按钮
(import EX "app/lib/exit.scm")   ; 硬 A/ESC 与右上角按钮 → 全局退出确认

;; ---- 常量（与 C 版 ui_pedometer.c 的同名常量一一对应）---------------------

(define FS              100)     ; 内部采样率 Hz（逻辑网格）
(define SAMPLE_MS       10)      ; 网格步长 ms
(define RING            512)     ; 滤波后信号环形缓冲（5.12s）
(define STATS_WIN       200)     ; 阈值统计窗口（2s）
(define FFT_N           256)     ; 频域校验点数（2.56s）
(define FREQ_BIN_MAX    10)      ; 频谱分析 bin 上限（0.39~3.9Hz）
(define VALID_MS        1000)    ; 频域校验周期（1s）
(define MIN_INTERVAL_MS 250)     ; 候选点最小间隔（上限 240 步/分）
(define DC_R            0.95)    ; DC-blocker 反馈系数
(define EMA_A           0.15)    ; EMA 低通系数
(define STD_K           0.3)     ; 自适应阈值 = 均值 + K×标准差
(define TH_MIN          0.015)   ; 阈值下限（g）
(define SFM_MAX         0.3)     ; 频谱平坦度阈值
(define RATE_TOL_REL    0.3)     ; 候选速率与主频的相对容差

(define TICK_MS         50)      ; 宿主定时器节拍（≈20Hz；内部再补到 100Hz 网格）
(define DRAW_MS         66)      ; 重绘节流（≈15fps，同 C 版）
(define DIAG_MS         2000)    ; 诊断行周期
(define TS_WRAP         134217727) ; System.timestamp 在 32 位宿主上的回绕周期 2^27（取 -1
                                   ; 避免超出无符号整数的 27 位逻辑宽度；差 1ms 无影响）

;; ---- 版面 ------------------------------------------------------------------

(define SW 320)
(define SH 240)
(define TOP_H 20)                 ; 页眉高
(define WAVE_Y0 100)              ; 波形区上沿
(define WAVE_H 80)                ; 波形区高
(define WAVE_N 160)               ; 波形显示点数（每点 2px）
(define WAVE_SCALE (/ 40.0 0.3))  ; 幅值 0.3g → 半高 40px
(define CLR_X 8)                  ; 「清零」按钮
(define CLR_Y 188)
(define CLR_W 56)
(define CLR_H 18)

;; ---- 工具 ------------------------------------------------------------------

;; 造一个全新列表（'() 是模块级静态槽，直接 push 会跨调用累积，见 album/gol 的教训）
(define make_list
  (lambda (n init)
    (define l (cdr (cons 0 (quote ()))))
    (define i 0)
    (while (< i n) { (push l init) (set! i (+ i 1)) })
    l))

(define min2 (lambda (a b) (if (< a b) a b)))

;; 字符步进（12px 抗锯齿字模：半角 7px、全角 12px，与 album/menu_list 同口径）
(define char_w (lambda (cp) (if (>= cp 11904) 12 7)))

(define text_w
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (while (< i n)
      { (set! w (+ w (char_w (String.charCodeAt i s)))) (set! i (+ i 1)) })
    w))

;; 两位小数格式化（Math.to_fixed 返回数字，1.9 会显示成 "1.9"；这里补齐成 "1.90"）
(define fmt2
  (lambda (x)
    (define s (String.atom_to_string (Math.to_fixed x 2)))
    (define n (String.length s))
    (define i 0)
    (define dot -1)
    (while (< i n)
      { (if (== (String.charCodeAt i s) 46) (set! dot i) #f) (set! i (+ i 1)) })
    (if (< dot 0)
        (String.concat s ".00")
        (if (== (- n dot) 2) (String.concat s "0") s))))

;; 环形缓冲取值：k 是**绝对样本序号**（0..sample_total-1；k 可为负，表示启动前的
;; 0 值前史）。环形槽位 = k mod RING；负 k 先 +RING 再取模（mod 的符号跟随被除数）。
(define ring_at
  (lambda (k)
    (get_item ring (Math.floor (mod (+ k RING) RING)))))

;; ---- 状态：时钟/采样 -------------------------------------------------------

(define clk 0)                    ; 逻辑时钟（ms，自启动累计；不随宿主时间戳回绕）
(define prev_now 0)               ; 上一拍的 System.timestamp
(define next_sample_ts 0)         ; 下一个 100Hz 网格槽的到期时刻（逻辑时钟）
(define next_valid_ts 0)          ; 下一次频域校验时刻
(define next_draw_ts 0)           ; 下一次重绘时刻
(define next_diag_ts 0)           ; 下一次诊断输出时刻

(define ring (make_list RING 0.0)) ; 滤波后的信号
(define ring_pos 0)                ; 环形写指针（= sample_total mod RING）
(define sample_total 0)            ; 累计样本数
(define dc_x1 0.0)                 ; DC-blocker 状态
(define dc_y1 0.0)
(define lp_y1 0.0)                 ; EMA 状态
(define primed 0)                  ; 1 = 已经收到过加速度样本（滤波器已初始化）

;; ---- 状态：时域峰值 --------------------------------------------------------

(define peak_thresh TH_MIN)
(define armed 1)                   ; 迟滞标志：信号回落到阈值以下才允许下一个候选点
(define last_step_ms (- 0 1000))   ; 上一个候选点时刻（初值让首个点不受间隔限制）
(define pending 0)                 ; 本窗口内待频域确认的候选步数
(define total_steps 0)             ; 已确认总步数
(define intervals (make_list 6 0)) ; 最近 6 个步间隔（ms）
(define int_num 0)
(define int_pos 0)
(define cad_hz 0.0)                ; 时域步频（Hz）
(define marks (make_list 20 -1))   ; 峰值标记：最近 20 个候选点的绝对样本序号
(define marks_pos 0)

;; ---- 状态：频域校验 --------------------------------------------------------

(define dom_freq 0.0)              ; 主频（Hz）
(define sfm 1.0)                   ; 频谱平坦度（越小越尖锐）
(define confirmed 0)               ; 1 = 本窗口确认为行走

;; ---- 状态：数据源/交互 -----------------------------------------------------

(define has_data 0)                ; 1 = 本拍读到了加速度
(define last_acc 1.0)              ; 最近一次模值（g）
(define timer 0)                   ; 定时器编号
(define t_prev 0)                  ; 上一拍触屏电平
(define t_arm 0)                   ; 「清零」按钮已武装（按下落在按钮内）

;; ---- 状态：文本缓存（避免每帧 String.concat 制造大量短串）--------------------

(define cache_steps -1)
(define cache_cad -1)
(define cache_dom -1)
(define cache_sfm -1)
(define cache_st -1)
(define txt_steps "0")
(define txt_l1 "步频 0 步/分")
(define txt_l2 "主频 0.00 Hz   SFM 1.00")
(define txt_st "无数据")
(define st_r 140)
(define st_g 140)
(define st_b 140)

;; ---- 状态：DSP 常量表 ------------------------------------------------------

(define hann (make_list FFT_N 0.0))       ; Hann 窗
(define tw_re (make_list 128 0.0))        ; e^{-j2πk/N} 实部（N/2 项）
(define tw_im (make_list 128 0.0))        ; 虚部
(define brev (make_list FFT_N 0))         ; 位反转置换表
(define fft_re (make_list FFT_N 0.0))     ; FFT 工作区（原地）
(define fft_im (make_list FFT_N 0.0))

;; 启动时算一次：Hann 窗、旋转因子、位反转表（8 位，N=256）
(define init_tables
  (lambda ()
    (define i 0)
    (define j 0)
    (define t 0)
    (define r 0)
    (define pi2 (* 2.0 (Math.PI)))
    (while (< i FFT_N)
      {
        (set_item! hann i (* 0.5 (- 1.0 (Math.cos (/ (* pi2 i) FFT_N)))))
        (set! t i)
        (set! r 0)
        (set! j 0)
        (while (< j 8)
          {
            (set! r (+ (* r 2) (Math.floor (mod t 2))))
            (set! t (Math.floor (/ t 2)))
            (set! j (+ j 1))
          })
        (set_item! brev i r)
        (set! i (+ i 1))
      })
    (set! i 0)
    (while (< i 128)
      {
        (set_item! tw_re i (Math.cos (- 0.0 (/ (* pi2 i) FFT_N))))
        (set_item! tw_im i (Math.sin (- 0.0 (/ (* pi2 i) FFT_N))))
        (set! i (+ i 1))
      })))

;; ---- 时域：自适应阈值 ------------------------------------------------------

;; 每 10 个样本重算一次：最近 min(sample_total, STATS_WIN) 个样本的 均值 + 0.3×标准差
(define update_stats
  (lambda ()
    (define n (min2 sample_total STATS_WIN))
    (define base (- sample_total n))
    (define i 0)
    (define v 0.0)
    (define mean 0.0)
    (define var 0.0)
    (while (< i n)
      {
        (set! v (ring_at (+ base i)))
        (set! mean (+ mean v))
        (set! i (+ i 1))
      })
    (set! mean (/ mean n))
    (set! i 0)
    (while (< i n)
      {
        (set! v (ring_at (+ base i)))
        (set! var (+ var (* (- v mean) (- v mean))))
        (set! i (+ i 1))
      })
    (set! var (/ var n))
    (set! peak_thresh (+ mean (* STD_K (Math.sqrt var))))
    (if (< peak_thresh TH_MIN) (set! peak_thresh TH_MIN) #f)))

;; ---- 时域：步频 ------------------------------------------------------------

(define update_cadence
  (lambda ()
    (define i 0)
    (define sum 0.0)
    (while (< i int_num)
      {
        (set! sum (+ sum (get_item intervals i)))
        (set! i (+ i 1))
      })
    (set! cad_hz (/ 1000.0 (/ sum int_num)))))

;; ---- 处理一个 100Hz 网格样本 -----------------------------------------------

(define record_step
  (lambda (ts)
    (define iv (- ts last_step_ms))
    ;; 首个候选点之前没有真实间隔（哨兵 -1000），不记入间隔表，否则步频初值会被拉低
    (if (>= last_step_ms 0)
        {
          (if (< int_num 6) (set! int_num (+ int_num 1)) #f)
          (set_item! intervals int_pos iv)
          (set! int_pos (Math.floor (mod (+ int_pos 1) 6)))
        }
        #f)
    (set! last_step_ms ts)
    (set! pending (+ pending 1))
    (set_item! marks marks_pos sample_total)
    (set! marks_pos (Math.floor (mod (+ marks_pos 1) 20)))))

(define process_sample
  (lambda (x ts)
    (define y 0.0)
    (define sig 0.0)
    (if (== primed 0)
        ;; 数据刚到手（首拍或数据源中断后恢复）：只初始化滤波器状态，不产生样本，
        ;; 否则 DC-blocker 会把 0 → 1g 的阶跃当成一个巨大峰值
        { (set! dc_x1 x) (set! dc_y1 0.0) (set! lp_y1 0.0) (set! primed 1) (set! armed 1) }
        {
          (set! y (+ (- x dc_x1) (* DC_R dc_y1)))
          (set! dc_x1 x)
          (set! dc_y1 y)
          (set! lp_y1 (+ lp_y1 (* EMA_A (- y lp_y1))))
          (set! sig lp_y1)
          (set_item! ring ring_pos sig)
          (set! ring_pos (Math.floor (mod (+ ring_pos 1) RING)))
          (set! sample_total (+ sample_total 1))
          (if (== (Math.floor (mod sample_total 10)) 0) (update_stats) #f)
          ;; 候选点 = 一次「上穿阈值」的上升沿（先回落到阈值以下才重新武装）。
          ;; C 版只看 sig>阈值（不要求回落），在慢步频下同一个周期内会反复触发；
          ;; 加迟滞后每个振荡周期至多一个候选点（与"峰值计数"的本意一致）。
          (if (<= sig peak_thresh)
              (set! armed 1)
              (if (and (== armed 1) (>= (- ts last_step_ms) MIN_INTERVAL_MS))
                  { (set! armed 0) (record_step ts) }
                  #f))
          (if (>= int_num 2) (update_cadence) #f)
        })))

;; ---- 频域：原地基 2 迭代 FFT（DIT，位反转输入）-----------------------------

(define fft_exec
  (lambda ()
    (define i 0)
    (define j 0)
    (define len 2)
    (define half 1)
    (define step 128)
    (define a 0)
    (define b 0)
    (define k 0)
    (define wr 0.0)
    (define wi 0.0)
    (define ar 0.0)
    (define ai 0.0)
    (define tr 0.0)
    (define ti 0.0)
    ;; 位反转重排（交换 i 与 brev[i]）
    (set! i 0)
    (while (< i FFT_N)
      {
        (set! j (get_item brev i))
        (if (> j i)
            {
              (set! tr (get_item fft_re i))
              (set! ti (get_item fft_im i))
              (set_item! fft_re i (get_item fft_re j))
              (set_item! fft_im i (get_item fft_im j))
              (set_item! fft_re j tr)
              (set_item! fft_im j ti)
            }
            #f)
        (set! i (+ i 1))
      })
    ;; 蝶形运算：len = 2,4,...,N
    (while (<= len FFT_N)
      {
        (set! half (Math.floor (/ len 2)))
        (set! step (Math.floor (/ FFT_N len)))
        (set! i 0)
        (while (< i FFT_N)
          {
            (set! j 0)
            (while (< j half)
              {
                (set! k (* j step))
                (set! wr (get_item tw_re k))
                (set! wi (get_item tw_im k))
                (set! a (+ i j))
                (set! b (+ a half))
                (set! ar (get_item fft_re a))
                (set! ai (get_item fft_im a))
                (set! tr (- (* wr (get_item fft_re b)) (* wi (get_item fft_im b))))
                (set! ti (+ (* wr (get_item fft_im b)) (* wi (get_item fft_re b))))
                (set_item! fft_re b (- ar tr))
                (set_item! fft_im b (- ai ti))
                (set_item! fft_re a (+ ar tr))
                (set_item! fft_im a (+ ai ti))
                (set! j (+ j 1))
              })
            (set! i (+ i len))
          })
        (set! len (* len 2))
      })))

;; ---- 频域交叉验证（每 1s）--------------------------------------------------

(define freq_validate
  (lambda ()
    (define base (- sample_total FFT_N))
    (define i 0)
    (define mean 0.0)
    (define p 0.0)
    (define log_sum 0.0)
    (define lin_sum 0.0)
    (define peak 0.0)
    (define pbin 0)
    (define b 1)
    (define cand 0.0)
    (define rate 0.0)
    (define tol 0.0)
    ;; 去直流
    (while (< i FFT_N)
      {
        (set! mean (+ mean (ring_at (+ base i))))
        (set! i (+ i 1))
      })
    (set! mean (/ mean FFT_N))
    ;; 加 Hann 窗装入工作区
    (set! i 0)
    (while (< i FFT_N)
      {
        (set_item! fft_re i (* (- (ring_at (+ base i)) mean) (get_item hann i)))
        (set_item! fft_im i 0.0)
        (set! i (+ i 1))
      })
    (fft_exec)
    ;; 1..FREQ_BIN_MAX 号 bin：对数/线性功率和与峰值 bin
    (while (<= b FREQ_BIN_MAX)
      {
        (set! p (+ (+ (* (get_item fft_re b) (get_item fft_re b))
                     (* (get_item fft_im b) (get_item fft_im b)))
                   1e-12))
        (set! log_sum (+ log_sum (Math.log p)))
        (set! lin_sum (+ lin_sum p))
        (if (> p peak) { (set! peak p) (set! pbin b) } #f)
        (set! b (+ b 1))
      })
    (if (< lin_sum 1e-9)
        (set! sfm 1.0)
        (set! sfm (/ (Math.exp (/ log_sum FREQ_BIN_MAX)) (/ lin_sum FREQ_BIN_MAX))))
    (set! dom_freq (* pbin (/ FS FFT_N)))
    ;; 交叉验证：频谱尖锐 且 候选速率与主频大致一致 → 本窗口候选点计入总步数
    ;;
    ;; 速率估计：C 版用「本窗口候选数 / 1s」，量化到 1Hz——步频 60~120 步/分时每窗口
    ;; 只有 1 个候选点，估计值 1.0Hz 与主频 1.8Hz 差 0.8Hz，超过容差，于是**一半窗口
    ;; 被误判为抖动而丢弃**（实测 1.78Hz 走 10s 只记 10 步）。这里改用 C 版自己也算的
    ;; 时域步频 cad_hz（最近若干步间隔的均值，连续值），间隔不足 2 个时退回窗口计数。
    (set! cand (/ pending (/ VALID_MS 1000.0)))
    (set! rate cand)
    (if (>= int_num 2) (set! rate cad_hz) #f)
    (set! tol (* RATE_TOL_REL rate))
    (if (< tol 0.3) (set! tol 0.3) #f)
    (if (and (< sfm SFM_MAX) (> pending 0) (<= (Math.abs (- rate dom_freq)) tol))
        { (set! total_steps (+ total_steps pending)) (set! confirmed 1) }
        (set! confirmed 0))
    (set! pending 0)))

;; ---- 状态词 / 文本缓存 -----------------------------------------------------

;; 0=无加速度数据 1=行走 2=未确认 3=静止
(define status_code
  (lambda ()
    (if (== has_data 0) 0
        (if (== confirmed 1) 1
            (if (> (- clk last_step_ms) 2000) 3 2)))))

(define refresh_text
  (lambda ()
    (define cad_i (Math.floor (* cad_hz 60.0)))
    (define dom_i (Math.floor (* dom_freq 100.0)))
    (define sfm_i (Math.floor (* sfm 100.0)))
    (define st (status_code))
    (if (not (== total_steps cache_steps))
        {
          (set! cache_steps total_steps)
          (set! txt_steps (String.atom_to_string total_steps))
        }
        #f)
    (if (not (== cad_i cache_cad))
        {
          (set! cache_cad cad_i)
          (set! txt_l1 (String.concat "步频 "
                        (String.concat (String.atom_to_string cad_i) " 步/分")))
        }
        #f)
    (if (not (and (== dom_i cache_dom) (== sfm_i cache_sfm)))
        {
          (set! cache_dom dom_i)
          (set! cache_sfm sfm_i)
          (set! txt_l2 (String.concat "主频 "
                        (String.concat (fmt2 dom_freq)
                          (String.concat " Hz   SFM "
                            (fmt2 sfm)))))
        }
        #f)
    (if (not (== st cache_st))
        {
          (set! cache_st st)
          (cond ((== st 0) { (set! txt_st "无数据") (set! st_r 140) (set! st_g 140) (set! st_b 140) })
                ((== st 1) { (set! txt_st "行走")   (set! st_r 0)   (set! st_g 200) (set! st_b 64) })
                ((== st 2) { (set! txt_st "未确认") (set! st_r 255) (set! st_g 160) (set! st_b 0) })
                (else      { (set! txt_st "静止")   (set! st_r 140) (set! st_g 140) (set! st_b 140) }))
        }
        #f)))

;; ---- 绘制 ------------------------------------------------------------------

(define hit_clear
  (lambda (px py)
    (if (and (>= px CLR_X) (< px (+ CLR_X CLR_W)))
        (if (and (>= py CLR_Y) (< py (+ CLR_Y CLR_H))) 1 0)
        0)))

(define draw_clear_button
  (lambda ()
    (define bg (if (== t_arm 1) 74 52))
    (Gfx.fill_rect CLR_X CLR_Y CLR_W CLR_H bg (+ bg 2) (+ bg 12))
    (Gfx.draw_rect CLR_X CLR_Y CLR_W CLR_H 112 118 132)
    (Gfx.draw_text "清零" (+ CLR_X 16) (+ CLR_Y 3) 225 228 235)))

;; 波形区：零线、阈值线、最近 WAVE_N 个采样（每点 2px）、峰值标记
(define draw_wave
  (lambda ()
    (define wy0 WAVE_Y0)
    (define wy1 (+ WAVE_Y0 WAVE_H))
    (define wym (+ WAVE_Y0 (/ WAVE_H 2)))
    (define i 0)
    (define k 0)
    (define v0 0.0)
    (define v1 0.0)
    (define ya 0)
    (define yb 0)
    (define ty 0)
    (define rel 0)
    (define px 0)
    (Gfx.fill_rect 0 wy0 SW WAVE_H 10 12 16)
    (Gfx.draw_line 0 wym (- SW 1) wym 58 60 66)
    ;; 阈值线（超过它才算候选点）
    (set! ty (- wym (Math.floor (* peak_thresh WAVE_SCALE))))
    (if (< ty wy0) (set! ty wy0) #f)
    (if (> ty wy1) (set! ty wy1) #f)
    (Gfx.draw_line 0 ty (- SW 1) ty 92 46 46)
    ;; 波形（尚无任何样本时不画，否则会得到一条假的零值直线）
    (set! k (- sample_total WAVE_N))
    (if (== sample_total 0) (set! i (- WAVE_N 1)) #f)   ; 无样本：i 直接置到循环终点，跳过波形
    (while (< i (- WAVE_N 1))
      {
        (set! v0 (ring_at (+ k i)))
        (set! v1 (ring_at (+ k (+ i 1))))
        (set! ya (- wym (Math.floor (* v0 WAVE_SCALE))))
        (set! yb (- wym (Math.floor (* v1 WAVE_SCALE))))
        (if (< ya wy0) (set! ya wy0) #f)
        (if (> ya wy1) (set! ya wy1) #f)
        (if (< yb wy0) (set! yb wy0) #f)
        (if (> yb wy1) (set! yb wy1) #f)
        (Gfx.draw_line (* i 2) ya (+ (* i 2) 2) yb 0 200 255)
        (set! i (+ i 1))
      })
    ;; 峰值标记（橙色竖线；marks 存的是候选点当时的绝对样本序号）
    (set! i 0)
    (while (< i 20)
      {
        (set! k (get_item marks i))
        (if (>= k 0)
            {
              (set! rel (- sample_total k))
              (if (and (>= rel 0) (< rel WAVE_N))
                  {
                    (set! px (* (- (- WAVE_N 1) rel) 2))
                    (Gfx.draw_line px (+ wy0 4) px (+ wy0 12) 255 96 0)
                  }
                  #f)
            }
            #f)
        (set! i (+ i 1))
      })
    (if (== has_data 0)
        {
          (Gfx.draw_text "无加速度数据" 104 (+ wy0 26) 220 226 236)
          (Gfx.draw_text "网页：点右下角「姿态」开启（需 HTTPS）" 20 (+ wy0 46) 150 158 172)
        }
        #f)))

(define draw
  (lambda ()
    (define nw 0)
    (define x0 0)
    (refresh_text)
    (Gfx.fill_rect 0 0 SW SH 12 14 18)
    ;; 页眉
    (Gfx.fill_rect 0 0 SW TOP_H 24 28 36)
    (Gfx.draw_text "计步器" 6 3 150 210 255)
    (EX.draw_button)
    ;; 总步数：16px 大字与单位「步」作为一组居中（16px 字模半角步进 9、全角 16）
    (set! nw (* 9 (String.length txt_steps)))
    (set! x0 (Math.floor (/ (- SW (+ (+ nw 4) 16)) 2)))
    (Gfx.set_font 2)
    (Gfx.draw_text txt_steps x0 27 120 220 255)
    (Gfx.draw_text "步" (+ x0 (+ nw 4)) 27 190 200 210)
    (Gfx.set_font 1)
    ;; 两行统计 + 右侧状态词
    (Gfx.draw_text txt_l1 6 62 205 210 220)
    (Gfx.draw_text txt_l2 6 80 150 160 175)
    (Gfx.draw_text txt_st (- SW (+ 6 (text_w txt_st))) 62 st_r st_g st_b)
    ;; 波形 + 交互
    (draw_wave)
    (draw_clear_button)
    (Gfx.draw_text "D / ENTER 也可清零" 72 192 110 118 130)
    (Gfx.refresh)))

;; ---- 输入 ------------------------------------------------------------------

(define clear_steps
  (lambda ()
    (set! total_steps 0)
    (set! pending 0)
    (set! cache_steps -1)   ; 强制重建大字文本
    (set! t_arm 0)
    #t))

;; 键盘：D / Enter 清零（与 C 版 D 键同义；浏览器里可打印字符就是 ASCII）
(define handle_keys
  (lambda ()
    (define k (Key.take_press))
    (define hit 0)
    (while (not (== k 0))
      {
        (if (== k 13) (set! hit 1) #f)
        (if (== k 100) (set! hit 1) #f)   ; 'd'
        (if (== k 68) (set! hit 1) #f)    ; 'D'
        (set! k (Key.take_press))
      })
    (if (== hit 1) (clear_steps) #f)))

;; 触屏「清零」按钮：按下落在按钮内即武装，在同一按钮内松手才生效（拖出取消）
(define handle_touch
  (lambda ()
    (define down 0)
    (define px 0)
    (define py 0)
    (if (Touch.is_available)
        {
          (set! down (if (Touch.is_pressed) 1 0))
          (set! px (Touch.x))
          (set! py (Touch.y))
          (if (== down 1)
              {
                (if (== t_prev 0)
                    (set! t_arm (if (== (hit_clear px py) 1) 1 0))
                    #f)
                (set! t_prev 1)
              }
              {
                (if (and (== t_prev 1) (== t_arm 1))
                    (if (== (hit_clear px py) 1) (clear_steps) #f)
                    #f)
                (set! t_prev 0)
              })
        }
        #f)))

;; ---- 每拍 ------------------------------------------------------------------

(define tick
  (lambda ()
    (define now 0)
    (define dt 0)
    (define mag 0.0)
    ;; ① 逻辑时钟：只取差值（32 位宿主的 System.timestamp = 纪元毫秒 mod 2^27，
    ;;    差值为负即发生了回绕），再钳到 [1,250]ms 防止挂起后一次补采过多
    (set! now (System.timestamp))
    (set! dt (- now prev_now))
    (if (< dt 0) (set! dt (+ dt TS_WRAP)) #f)
    (set! prev_now now)
    (if (< dt 1) (set! dt 1) #f)
    (if (> dt 250) (set! dt 250) #f)
    (set! clk (+ clk dt))

    ;; ② 读一次加速度模值（浏览器传感器约 20Hz 更新，本拍值在下面的网格里零阶保持）
    (set! mag (Imu.accel_mag))
    (if (number? mag)
        { (set! has_data 1) (set! last_acc mag) }
        { (set! has_data 0) (set! primed 0) })

    ;; ③ 100Hz 网格采样（宿主定时器可能一次跨过多个 10ms 槽）
    (if (== has_data 1)
        (while (>= clk next_sample_ts)
          {
            (process_sample last_acc next_sample_ts)
            (set! next_sample_ts (+ next_sample_ts SAMPLE_MS))
          })
        (if (< next_sample_ts clk)
            (set! next_sample_ts (+ clk SAMPLE_MS))
            #f))

    ;; ④ 每 1s 频域交叉验证（累计样本够一个 FFT 窗口才做）
    (if (>= clk next_valid_ts)
        {
          (if (>= sample_total FFT_N) (freq_validate) #f)
          (set! next_valid_ts (+ next_valid_ts VALID_MS))
          (if (< next_valid_ts clk) (set! next_valid_ts (+ clk VALID_MS)) #f)
        }
        #f)

    ;; ⑤ 输入 + 退出通道（EX.check 每拍轮询返回请求与右上角按钮）
    (handle_keys)
    (handle_touch)
    (EX.check)

    ;; ⑥ 重绘节流（≈15fps）
    (if (>= clk next_draw_ts)
        {
          (set! next_draw_ts (+ clk DRAW_MS))
          (draw)
        }
        #f)

    ;; ⑦ 诊断行（供宿主回归台/浏览器 e2e 做数值断言）
    (if (>= clk next_diag_ts)
        {
          (set! next_diag_ts (+ clk DIAG_MS))
          (display "[pedometer] steps=")
          (display total_steps)
          (display " pending=")
          (display pending)
          (display " cad=")
          (display (Math.to_fixed cad_hz 2))
          (display " dom=")
          (display (Math.to_fixed dom_freq 2))
          (display " sfm=")
          (display (Math.to_fixed sfm 2))
          (display " conf=")
          (display confirmed)
          (display " data=")
          (display has_data)
          (display " samples=")
          (display sample_total)
          (newline)
        }
        #f)))

;; 收尾：停掉本 App 的定时器（实例即将销毁，这里只做礼节性清理）
(define stop
  (lambda ()
    (if (> timer 0) (System.clear_timeout timer) #f)
    (display "[pedometer] bye")
    (newline)))

;; ---- 启动 ------------------------------------------------------------------

(set! SW (Gfx.width))
(set! SH (Gfx.height))
(init_tables)
(EX.install_with draw)
(EX.on_exit stop)
(set! prev_now (System.timestamp))
(set! clk 0)
(set! next_sample_ts 0)
(set! next_valid_ts VALID_MS)
(set! next_draw_ts 0)
(set! next_diag_ts 0)
(set! last_step_ms (- 0 1000))
(draw)
(set! timer (System.set_interval TICK_MS (lambda () (tick))))
(display "[pedometer] start fs=")
(display FS)
(display " tick=")
(display TICK_MS)
(newline)
