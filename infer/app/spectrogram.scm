;; ============================================================================
;; 声谱图（Spectrogram）—— 灵机应用程序（Scheme App）
;;
;; 与 C 版频谱仪（infer/ui_spectrogram.c）对齐的 Scheme 重写：麦克风每来一帧
;; 1024 采样（≈23ms），加 Hann 窗 + FFT，把前 294 个频点按对数幅度映射成 0~255
;; 的强度，再按调色板取色、以 2px 高的新行画在画面底部；整屏向上滚动，于是历史
;; 声谱自然保留（瀑布图）。频率轴（顶部，kHz）与时间轴（左侧，秒）每帧重绘。
;;
;; 分工（关键）：**采集与 FFT 全在 C 侧**，(native Audio) 提供开麦 / 读一列 /
;;   按频点取强度 / 原始采样 / 电平表，详见 infer/am_native_Audio.h 的设计说明。
;;   1024 点 FFT 若放在 Scheme 里是 ~3×10⁵ VM 指令/帧、每 23ms 一列，解释器根本
;;   跟不上（会掉到个位数 fps）；跨 FFI 边界只传标量，PCM 与频谱列表从不进堆。
;;
;; 交互（**全部触屏控件**，不再依赖 16 宫格软按键）：
;;   底部一排 6 枚按钮（通用控件 app/lib/button.scm）：
;;     基准- / 基准+   基准电平（越大越抑制弱信号），按住连发
;;     范围- / 范围+   动态范围（越小对比越强），按住连发
;;     暂停 / 继续     冻结画面（冻结期间仍消费音频，恢复后立刻回到实时）
;;     热力 / 冰蓝 / 灰度   调色板轮换（按钮标签即当前配色，切换时清屏重画）
;;   右上角「退出」（app/lib/exit.scm）：点按 → 全局确认框；硬 A/ESC 同一条路。
;;   「无麦克风」卡片里的「重试」按钮：重新打开麦克风。
;;   实体键盘兜底（桌面/终端）：←/→ 调基准，4/6 调范围，空格暂停，C 换配色，R 重开麦。
;;
;; 节拍与实时性：
;;   16ms 一拍（浏览器 rAF 下 ≈60Hz），但**列率跟着音频走**：(Audio.read) 一次只取
;;   一帧（1024 采样 = 23ms 音频），每拍最多连取 COL_MAX 帧、每帧画一列——于是
;;   43 列/秒全部上屏（浏览器 AudioWorklet 每 2048 采样投递一次，表现为"一拍 0 列 /
;;   一拍 2 列"交替）。时间轴因此是**精确的音频时间**：一列 = fft/rate ≈ 23.2ms，
;;   1 秒 = 43.1 列 = 86px（不是实测 EMA；宿主再慢也只是滚动变慢，比例关系不变）。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数；没有一元取负 → (- 0 x)；
;;   没有 let/list/map/append；define 只写在模块顶层或 lambda 体顶层（块内不 define）；
;;   cond 每子句只接受单个表达式；/ 与 mod 恒浮点（索引用 Math.floor 收敛）；
;;   while 循环体不是独立词法区域；0 是真值（布尔一律 #t/#f）；'() 是模块级静态槽，
;;   故空表一律 (cdr '(0)) 现造，绝不对 '() 直接 push。
;; ============================================================================

(native Gfx)
(native Touch)
(native Key)
(native Audio)
(native System)
(native String)
(native Math)
(native App)

(import BTN "app/lib/button.scm")   ; 通用按钮控件（圆角底/描边/居中标签/按住高亮/点按）
(import EX "app/lib/exit.scm")      ; 硬 A/ESC 与右上角「退出」按钮 → 全局退出确认框

;; ============================================================================
;; 常量：版面（320x240 基准，按实际画布宽高推导）
;; ============================================================================

(define SW (Gfx.width))
(define SH (Gfx.height))

(define TICK_MS 16)          ; 一拍 16ms（浏览器 rAF ≈60Hz；音频帧 ≈23ms 一列）
(define SAMPLE_RATE 44100)   ; 与 C 版 SP_SAMPLE_RATE 一致（WASM 上 AudioContext 按此建立）

(define TOP_H 22)            ; 顶栏（标题 + 状态 + 右上角「退出」）
(define AXIS_Y TOP_H)        ; 频率轴（14px：TOP_H..TOP_H+13）
(define AXIS_H 14)
(define PLOT_X0 26)          ; 左侧时间轴宽度（px）
(define PLOT_Y0 (+ AXIS_Y AXIS_H))          ; 36
(define BAR_Y (- SH 46))                    ; 194：底部按钮行（高 20）
(define BTN_H 20)
(define STAT_Y (- SH 24))                   ; 216：状态条（高 24）
(define PLOT_BOTTOM (- BAR_Y 2))            ; 192：绘图区下边界（不含）
(define ROW_H 2)                            ; 每列高度（= C 版 SP_ROW_H）
(define PLOT_H (- PLOT_BOTTOM PLOT_Y0))     ; 156
(define ROWS (Math.floor (/ PLOT_H ROW_H))) ; 78 列历史（≈1.8s @43 列/秒）
(define AXIS_W PLOT_X0)                     ; 左侧轴条宽度
(define MAX_BINS (- SW PLOT_X0))            ; 294：每行可绘制的频点数

;; C 版的电平映射策略（本库只做宽松钳位，策略在 App 侧，见 am_native_Audio.h）
(define LEVEL_BASE_DEFAULT 240)
(define LEVEL_BASE_STEP 8)
(define LEVEL_BASE_MIN 0)
(define LEVEL_BASE_MAX 240)
(define LEVEL_RANGE_DEFAULT 300)
(define LEVEL_RANGE_STEP 16)
(define LEVEL_RANGE_MIN 32)
(define LEVEL_RANGE_MAX 512)

;; 每拍最多画几列：AudioWorklet 每 2048 采样投递一次（= 两帧），故典型情况是
;; "一拍 0 列 / 一拍 2 列"交替。上限 4 足以跟上 43 列/秒的音频速率，又给单拍
;; 绘制成本封顶（每列约 294 次 fill_rect）。
(define COL_MAX 4)

;; 按住连发（基准/范围的四个按钮）
(define HOLD_DELAY_MS 360)
(define HOLD_REPEAT_MS 110)
(define ADJUST_FLASH_MS 900)      ; 调节后读数高亮的时长

;; 底部按钮行：6 枚等宽按钮
(define BTN_W 49)
(define BTN_GAP 3)
(define BTN_X0 4)

;; 状态条内的部件
(define WAVE_X 118)               ; 波形缩略图（38 列 × 2px = 76px）
(define WAVE_Y 218)
(define WAVE_W 76)
(define WAVE_H 18)
(define WAVE_COLS 38)
(define RMS_X 226)                ; 电平表
(define RMS_Y (+ STAT_Y 9))
(define RMS_W 86)
(define RMS_H 6)

;; 配色（深色主题，与 C 版各 App 同一套观感）
(define BG_R 12)  (define BG_G 16)  (define BG_B 22)
(define CARD_R 27) (define CARD_G 33) (define CARD_B 44)
(define EDGE_R 48) (define EDGE_G 58) (define EDGE_B 74)
(define SUR_R 38)  (define SUR_G 47)  (define SUR_B 61)
(define ACC_R 56)  (define ACC_G 189) (define ACC_B 248)
(define TXT_R 237) (define TXT_G 242) (define TXT_B 249)
(define DIM_R 140) (define DIM_G 152) (define DIM_B 168)
(define TRK_R 52)  (define TRK_G 63)  (define TRK_B 79)
(define AX_R 150)  (define AX_G 150)  (define AX_B 150)     ; 坐标轴文字
(define TICK_R 96) (define TICK_G 96) (define TICK_B 96)    ; 刻度线
(define WAV_R 90)  (define WAV_G 220) (define WAV_B 200)    ; 波形缩略图
(define WRN_R 255) (define WRN_G 178) (define WRN_B 92)     ; 警示（等数据/无麦克风）
(define DAN_R 248) (define DAN_G 113) (define DAN_B 113)    ; 危险（电平削顶）

;; ============================================================================
;; 状态
;; ============================================================================

(define base LEVEL_BASE_DEFAULT)     ; 基准电平（映射域下限）
(define range LEVEL_RANGE_DEFAULT)   ; 动态范围（映射域宽度）
(define pal_mode 0)                  ; 0=热力 1=冰蓝 2=灰度
(define paused #f)                   ; 画面是否冻结
(define mic_ok #f)                   ; 麦克风是否已打开
(define mic_cols 0)                  ; 已画出的列数（0 = 还没收到数据）
(define mic_err #f)                  ; mic_read 报过硬错误
(define mic_open_ms 0)               ; 最近一次开麦的时刻（"等数据"提示用）

(define fft_size 1024)               ; (Audio.info) 的 FFT 点数
(define max_bins 511)                ; (Audio.info) 的可绘制频点数
(define nbins MAX_BINS)              ; 实际绘制的频点数（= min(画布宽-轴宽, max_bins)）
(define bin_hz 43.066)               ; 每个频点的频率宽度（Hz）

(define rms 0)                       ; 最新一帧的时域电平 0~255
;; 时间轴标定：一列 = 一帧音频（fft_size 个采样）→ 每列代表的音频时长 = fft/rate。
;; 这是**精确值**（不是实测 EMA）：每列就是那 23.2ms 的音频，故 1 秒 = 43.1 列 = 86px。
;; 与 C 版按"实测帧间隔"标定的差异：C 版一行一帧、帧率=实测；这里列率若跟不上音频
;; （极慢宿主），画面只是滚动得比实时慢，而"每列 23ms"的关系始终成立。
(define sec_per_row 0.02322)
(define px_per_sec 86.0)
(define diag_cols 0)                 ; 上一条诊断行时的列数（算实测列率）
(define diag_ms 0)

(define ntick 0)
(define now_ms 0)
(define last_diag 0)
(define timer_id 0)
(define dirty #t)                    ; 需要重画"静态部分"（顶栏/轴/状态条）
(define want_clear #t)               ; 需要清空绘图区（启动/换配色/从覆盖提示恢复）
(define overlay_kind 0)              ; 绘图区当前盖着的提示：0=无 1=等数据 2=无麦克风

;; 按住连发状态（哪一枚按钮正被按住：-1 无；0..5 底部按钮；6 卡片里的「重试」）
(define hold_btn -1)
(define hold_next 0)
(define prev_armed -1)               ; 上一拍的按住序号（高亮变化时立刻重画）

(define adjust_ms -100000)           ; 最近一次调节电平/范围的时刻（读数高亮）
;; 「无麦克风 / 等数据」两态的画面是静止的，但用户需要"还活着"的反馈：顶部状态点做
;; 呼吸闪烁（每 BLINK_MS 重画一次），并顺带让完全静止的场景也有推帧活动。
(define BLINK_MS 400)
(define blink_ms -100000)
(define RETRY_MS 6000)               ; 无麦克风时每 6s 自动重试一次开麦
(define retry_ms 0)

;; 读数/状态文本缓存（**不要每列重新 concat**：短串驻留表的开销见 manual §3.4）
(define lv_text "基准 240")
(define rg_text "范围 300")
(define state_text "启动中")
(define pal_names (cons "热力" (cons "冰蓝" (cons "灰度" (cdr '(0))))))

;; 调色板：三套配色各自三条 256 元列表（R/G/B），启动时建好；切换只换指针
(define cur_r (cdr '(0)))
(define cur_g (cdr '(0)))
(define cur_b (cdr '(0)))
(define pals (cdr '(0)))

;; ============================================================================
;; 通用小工具
;; ============================================================================

(define clamp
  (lambda (v lo hi)
    (if (< v lo) lo (if (> v hi) hi v))))

(define min2 (lambda (a b) (if (< a b) a b)))
(define max2 (lambda (a b) (if (> a b) a b)))

;; 0x2E80（CJK 部首）起算全角，与 app/lib/button.scm / menu_list.scm 同一判据
(define char_w (lambda (cp) (if (>= cp 11904) 12 7)))

(define text_w
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (while (< i n)
      { (set! w (+ w (char_w (String.charCodeAt i s))))
        (set! i (+ i 1)) })
    w))

(define draw_right
  (lambda (s right y r g b)
    (Gfx.draw_text s (- right (text_w s)) y r g b)))

(define draw_center
  (lambda (s cy r g b)
    (Gfx.draw_text_centered 1 s (Math.floor (/ SW 2)) cy r g b 1)))

;; ============================================================================
;; 调色板（0~255 强度 → RGB；热力配色与 C 版 ui_spectrogram.c 逐段一致）
;; ============================================================================

(define pal_heat
  (lambda (i)
    (cond ((< i 64) (cons 0 (cons 0 (cons (* i 4) (cdr '(0))))))
          ((< i 128) (cons 0 (cons (* (- i 64) 4) (cons 255 (cdr '(0))))))
          ((< i 192) (cons (* (- i 128) 4) (cons 255
                       (cons (- 255 (* (- i 128) 4)) (cdr '(0))))))
          (else (cons 255 (cons (- 255 (* (- i 192) 4)) (cons 0 (cdr '(0)))))))))

(define pal_ice
  (lambda (i)
    (cond ((< i 64) (cons 0 (cons (Math.floor (/ (* i 48) 64))
                       (cons (+ 8 (Math.floor (/ (* i 132) 64))) (cdr '(0))))))
          ((< i 128) (cons 0 (cons (+ 48 (Math.floor (/ (* (- i 64) 92) 64)))
                       (cons (+ 140 (Math.floor (/ (* (- i 64) 80) 64))) (cdr '(0))))))
          ((< i 192) (cons (Math.floor (/ (* (- i 128) 120) 64))
                       (cons (+ 140 (Math.floor (/ (* (- i 128) 70) 64)))
                       (cons (+ 220 (Math.floor (/ (* (- i 128) 30) 64))) (cdr '(0))))))
          (else (cons (+ 120 (Math.floor (/ (* (- i 192) 135) 63)))
                (cons (+ 210 (Math.floor (/ (* (- i 192) 45) 63)))
                (cons (+ 250 (Math.floor (/ (* (- i 192) 5) 63))) (cdr '(0)))))))))

(define pal_gray
  (lambda (i)
    (cons i (cons i (cons i (cdr '(0)))))))

;; 建一套配色：(R 列表 G 列表 B 列表)
(define build_pal
  (lambda (mode)
    (define r (cdr '(0)))
    (define g (cdr '(0)))
    (define b (cdr '(0)))
    (define i 0)
    (define c (cdr '(0)))
    (while (< i 256)
      {
        (if (== mode 0) (set! c (pal_heat i)) #f)
        (if (== mode 1) (set! c (pal_ice i)) #f)
        (if (== mode 2) (set! c (pal_gray i)) #f)
        (push r (get_item c 0))
        (push g (get_item c 1))
        (push b (get_item c 2))
        (set! i (+ i 1))
      })
    (cons r (cons g (cons b (cdr '(0)))))))

(define build_all_pals
  (lambda ()
    (set! pals (cdr '(0)))
    (push pals (build_pal 0))
    (push pals (build_pal 1))
    (push pals (build_pal 2))
    (select_pal 0)))

;; 切换到第 mode 套配色（只换指针，不重建）
(define select_pal
  (lambda (mode)
    (define p (cdr '(0)))
    (set! pal_mode mode)
    (set! p (get_item pals mode))
    (set! cur_r (get_item p 0))
    (set! cur_g (get_item p 1))
    (set! cur_b (get_item p 2))
    (BTN.set_label b_pal (get_item pal_names mode))
    (set! want_clear #t)
    (set! dirty #t)))

;; ============================================================================
;; 文本缓存（读数/状态；只在值变化时重建）
;; ============================================================================

(define refresh_readout
  (lambda ()
    (set! lv_text (String.concat "基准 " (String.atom_to_string base)))
    (set! rg_text (String.concat "范围 " (String.atom_to_string range)))))

(define refresh_state
  (lambda ()
    (define s "")
    (if mic_err
        (set! s "麦克风错误")
        (if (not mic_ok)
            (set! s "无麦克风")
            (if paused
                (set! s "已暂停")
                (if (== mic_cols 0) (set! s "等待数据") (set! s "实时")))))
    (set! state_text s)))

;; ============================================================================
;; 音频：开麦 / 关麦 / 电平与范围
;; ============================================================================

(define apply_level
  (lambda (b r)
    (define nb 0)
    (define nr 0)
    (set! nb (Math.floor (clamp b LEVEL_BASE_MIN LEVEL_BASE_MAX)))
    (set! nr (Math.floor (clamp r LEVEL_RANGE_MIN LEVEL_RANGE_MAX)))
    (if (or (not (== nb base)) (not (== nr range)))
        {
          (set! base nb)
          (set! range nr)
          (Audio.set_level base range)
          (refresh_readout)
          (set! adjust_ms now_ms)
          (set! dirty #t)
        }
        #f)))

(define open_mic
  (lambda ()
    (Audio.close)
    (set! mic_cols 0)
    (set! mic_err #f)
    (set! mic_ok (Audio.open SAMPLE_RATE))
    (set! mic_open_ms now_ms)
    (set! retry_ms now_ms)
    (BTN.reset b_retry)
    (refresh_state)
    (set! overlay_kind (if mic_ok 1 2))
    (set! want_clear #t)
    (set! dirty #t)
    (display "[spectrogram] open=") (display mic_ok)
    (display " info=") (display (Audio.info))
    (newline)))

;; 后台恢复：宿主可能已把音频设备交给别的 App（或页面重建了 AudioContext），
;; 关掉再按原采样率重开；画面整体重画（挂起期间画面归别人）。
(define on_resume
  (lambda ()
    (define was_ok mic_ok)
    (Audio.close)
    (set! mic_ok (Audio.open SAMPLE_RATE))
    (set! mic_err #f)
    (set! mic_cols 0)
    (set! mic_open_ms now_ms)
    (refresh_state)
    (set! overlay_kind (if mic_ok 1 2))
    (set! want_clear #t)
    (set! dirty #t)
    (display "[spectrogram] resume was=") (display was_ok)
    (display " now=") (display mic_ok) (newline)))

;; ============================================================================
;; 按钮（通用控件；6 枚底部 + 卡片里的「重试」）
;; ============================================================================

(define b_base_dn (BTN.make "基准-" (+ BTN_X0 (* 0 (+ BTN_W BTN_GAP))) BAR_Y BTN_W BTN_H))
(define b_base_up (BTN.make "基准+" (+ BTN_X0 (* 1 (+ BTN_W BTN_GAP))) BAR_Y BTN_W BTN_H))
(define b_rng_dn  (BTN.make "范围-" (+ BTN_X0 (* 2 (+ BTN_W BTN_GAP))) BAR_Y BTN_W BTN_H))
(define b_rng_up  (BTN.make "范围+" (+ BTN_X0 (* 3 (+ BTN_W BTN_GAP))) BAR_Y BTN_W BTN_H))
(define b_pause   (BTN.make "暂停"  (+ BTN_X0 (* 4 (+ BTN_W BTN_GAP))) BAR_Y BTN_W BTN_H))
(define b_pal     (BTN.make "热力"  (+ BTN_X0 (* 5 (+ BTN_W BTN_GAP))) BAR_Y BTN_W BTN_H))
(define b_retry   (BTN.make "重试"  128 132 64 20))

;; 统一按钮外观（深色底 + 1px 描边 + 12px 抗锯齿字 + 强调色按下底）
(define style_btn
  (lambda (b)
    {
      (BTN.set_font b 1)
      (BTN.set_text_color b 228 233 240)
      (BTN.set_bg b SUR_R SUR_G SUR_B)
      (BTN.set_active_bg b 116 172 226)
      (BTN.set_radius b 6)
      (BTN.set_border b EDGE_R EDGE_G EDGE_B)
    }))

(define style_all_btns
  (lambda ()
    {
      (style_btn b_base_dn) (style_btn b_base_up)
      (style_btn b_rng_dn)  (style_btn b_rng_up)
      (style_btn b_pause)   (style_btn b_pal)
      (style_btn b_retry)
      (BTN.set_active_bg b_retry 116 172 226)
    }))

;; 一枚按钮"生效"时的动作（idx：0 基准- 1 基准+ 2 范围- 3 范围+ 4 暂停 5 配色 6 重试）
(define act
  (lambda (idx)
    (if (== idx 0) (apply_level (- base LEVEL_BASE_STEP) range) #f)
    (if (== idx 1) (apply_level (+ base LEVEL_BASE_STEP) range) #f)
    (if (== idx 2) (apply_level base (- range LEVEL_RANGE_STEP)) #f)
    (if (== idx 3) (apply_level base (+ range LEVEL_RANGE_STEP)) #f)
    (if (== idx 4)
        {
          (set! paused (if paused #f #t))
          (BTN.set_label b_pause (if paused "继续" "暂停"))
          (refresh_state)
          (set! dirty #t)
          (display "[spectrogram] paused=") (display paused) (newline)
        }
        #f)
    (if (== idx 5)
        {
          (select_pal (Math.floor (mod (+ pal_mode 1) 3)))
          (display "[spectrogram] palette=") (display pal_mode) (newline)
        }
        #f)
    (if (== idx 6) (open_mic) #f)))

;; 哪一枚按钮正被按住（-1 = 没有）。只取电平，不改边沿。
(define armed_idx
  (lambda ()
    (define idx -1)
    (if (BTN.armed b_base_dn) (set! idx 0) #f)
    (if (BTN.armed b_base_up) (set! idx 1) #f)
    (if (BTN.armed b_rng_dn)  (set! idx 2) #f)
    (if (BTN.armed b_rng_up)  (set! idx 3) #f)
    (if (BTN.armed b_pause)   (set! idx 4) #f)
    (if (BTN.armed b_pal)     (set! idx 5) #f)
    (if (not mic_ok) (if (BTN.armed b_retry) (set! idx 6) #f) #f)
    idx))

;; 每拍一次：先让控件消费触屏边沿（点按在控件内部结算），再按"按住"驱动连发。
(define handle_buttons
  (lambda ()
    (define idx -1)
    (BTN.track b_base_dn)
    (BTN.track b_base_up)
    (BTN.track b_rng_dn)
    (BTN.track b_rng_up)
    (BTN.track b_pause)
    (BTN.track b_pal)
    (if (not mic_ok) (BTN.track b_retry) #f)
    (set! idx (armed_idx))
    ;; 按住/松手会改变按钮高亮：暂停态下没有新列可画，必须由这里触发一次重画，
    ;; 否则最后画在屏上的是"按下色"，松手后高亮会一直卡住（实测）。
    (if (not (== idx prev_armed)) (set! dirty #t) #f)
    (set! prev_armed idx)
    (if (< idx 0)
        (set! hold_btn -1)
        (if (not (== idx hold_btn))
            {
              ;; 按下沿：立即生效一次；基准/范围类按钮进入连发倒计时
              (set! hold_btn idx)
              (act idx)
              (set! hold_next (+ now_ms HOLD_DELAY_MS))
            }
            (if (and (< idx 4) (>= now_ms hold_next))
                {
                  (act idx)
                  (set! hold_next (+ now_ms HOLD_REPEAT_MS))
                }
                #f)))))

;; ============================================================================
;; 键盘兜底（实体键盘；触屏优先，键码同 hal_key.h 的 NANO_KEY_*）
;; ============================================================================

(define handle_key
  (lambda (k)
    (if (== k 3) (apply_level (- base LEVEL_BASE_STEP) range) #f)      ; ←
    (if (== k 4) (apply_level (+ base LEVEL_BASE_STEP) range) #f)      ; →
    (if (or (== k 52) (== k 2)) (apply_level base (- range LEVEL_RANGE_STEP)) #f)  ; '4'/↓
    (if (or (== k 54) (== k 1)) (apply_level base (+ range LEVEL_RANGE_STEP)) #f)  ; '6'/↑
    (if (or (== k 32) (== k 13)) (act 4) #f)                          ; 空格/回车 = 暂停
    (if (or (== k 99) (== k 67)) (act 5) #f)                          ; 'c'/'C' = 换配色
    (if (or (== k 114) (== k 82)) (open_mic) #f)))                    ; 'r'/'R' = 重开麦

(define handle_keys
  (lambda ()
    (define k 0)
    (set! k (Key.take_press))
    (while (> k 0)
      { (handle_key k) (set! k (Key.take_press)) })))

;; ============================================================================
;; 绘制：静态部件（顶栏 / 频率轴 / 时间轴 / 状态条）
;; ============================================================================

(define draw_top
  (lambda ()
    (define sr 255) (define sg 255) (define sb 255)
    (define sx 0)
    (Gfx.fill_rect 0 0 SW TOP_H BG_R BG_G BG_B)
    (Gfx.fill_rect 0 TOP_H SW 1 EDGE_R EDGE_G EDGE_B)
    (Gfx.rounded_rect 12 7 3 12 1 1 1 1 ACC_R ACC_G ACC_B 1)
    (Gfx.draw_text "声谱图" 20 5 TXT_R TXT_G TXT_B)
    ;; 状态文字右对齐到「退出」按钮左侧；左边一颗状态点
    (if (== state_text "实时")
        { (set! sr ACC_R) (set! sg ACC_G) (set! sb ACC_B) }
        (if (== state_text "已暂停")
            { (set! sr 200) (set! sg 208) (set! sb 220) }
            { (set! sr WRN_R) (set! sg WRN_G) (set! sb WRN_B) }))
    (set! sx (- (- SW (+ EX.btn_w 10)) (text_w state_text)))
    (Gfx.draw_text state_text sx 5 sr sg sb)
    ;; 状态点：实时/暂停常亮；无麦克风/等数据时呼吸闪烁（画面静止，靠它证明"还活着"）
    (if (or (not mic_ok) (== mic_cols 0))
        (if (== (Math.floor (mod (/ now_ms BLINK_MS) 2)) 0)
            (Gfx.fill_circle_aa (- sx 10) 11 3 WRN_R WRN_G WRN_B 1)
            (Gfx.fill_circle_aa (- sx 10) 11 2 WRN_R WRN_G WRN_B 64))
        (Gfx.fill_circle_aa (- sx 10) 11 3 ACC_R ACC_G ACC_B 1))
    ;; 右上角「退出」：**必须每帧调** —— 它除了画按钮，还负责把按钮矩形登记给宿主
    ;; （(App.set_exit_button)），宿主据此在**触摸边沿**上判定命中。只留 (EX.check)
    ;; 的本地电平兜底会出现"看不见按钮、但点那块区域仍能退出"的怪象（漏调一次即复现）。
    (EX.draw_button)))

;; 频率轴：0 / 2k / 4k … 12k（刻度像素位置 = 频点 × bin_hz 反算，与 C 版同一算式）
(define draw_freq_axis
  (lambda ()
    (define fk 2)
    (define px 0)
    (define lb "")
    (Gfx.fill_rect 0 AXIS_Y SW AXIS_H 0 0 0)
    (Gfx.draw_text "0" (- PLOT_X0 9) (+ AXIS_Y 2) AX_R AX_G AX_B)
    (while (<= fk 12)
      {
        (set! px (+ PLOT_X0 (Math.floor (+ (/ (* fk 1000.0) bin_hz) 0.5))))
        (Gfx.fill_rect (- px 1) AXIS_Y 1 2 TICK_R TICK_G TICK_B)
        (set! lb (String.concat (String.atom_to_string fk) "k"))
        (Gfx.draw_text lb (- px (Math.floor (/ (text_w lb) 2))) (+ AXIS_Y 3) AX_R AX_G AX_B)
        (set! fk (+ fk 2))
      })))

;; 时间轴：底部 0s，向上每整秒一条刻度（-1s / -2s…；按实测列间隔换算像素）
(define draw_time_axis
  (lambda ()
    (define k 1)
    (define y 0)
    (define ppx 0.0)
    (Gfx.fill_rect 0 PLOT_Y0 AXIS_W PLOT_H 0 0 0)
    (set! ppx px_per_sec)
    (Gfx.draw_text "0s" 2 (- PLOT_BOTTOM 13) AX_R AX_G AX_B)
    (while (<= k 3)
      {
        (set! y (Math.floor (- (- PLOT_BOTTOM 1) (* k ppx))))
        (if (>= y (+ PLOT_Y0 14))
            {
              (Gfx.fill_rect (- PLOT_X0 4) y 4 1 TICK_R TICK_G TICK_B)
              (Gfx.draw_text (String.concat "-" (String.concat (String.atom_to_string k) "s"))
                             1 (- y 6) AX_R AX_G AX_B)
            }
            #f)
        (set! k (+ k 1))
      })))

;; 波形缩略图：每列取 3 个采样（间隔 9）求 min/max，画成上下对称的包络条。
;; 为什么不"每列取 1 个采样"：1024 采样 / 38 列 = 27，直接抽样对 1kHz 以上的信号是
;; 欠采样（会画出一串毫无意义的点阵）；取窗口极值才是"波形包络"，且能一眼看出静音与削顶。
(define draw_wave
  (lambda ()
    (define i 0)
    (define k 0)
    (define s 0)
    (define lo 0)
    (define hi 0)
    (define y_hi 0)
    (define y_lo 0)
    (define cy (+ WAVE_Y (Math.floor (/ WAVE_H 2))))
    (Gfx.fill_rect WAVE_X WAVE_Y WAVE_W WAVE_H 7 9 13)
    (Gfx.draw_rect WAVE_X WAVE_Y WAVE_W WAVE_H EDGE_R EDGE_G EDGE_B)
    (while (< i WAVE_COLS)
      {
        (set! lo 0)
        (set! hi 0)
        (set! k 0)
        (while (< k 3)
          {
            (set! s (Audio.sample (+ (* i 27) (* k 9))))
            (if (< s lo) (set! lo s) #f)
            (if (> s hi) (set! hi s) #f)
            (set! k (+ k 1))
          })
        (set! y_hi (Math.floor (/ (* hi (- WAVE_H 2)) 65536.0)))
        (set! y_lo (Math.floor (/ (* lo (- WAVE_H 2)) 65536.0)))
        (Gfx.fill_rect (+ WAVE_X (* i 2)) (+ cy (- 0 y_hi)) 2
                       (max2 1 (- y_hi y_lo)) WAV_R WAV_G WAV_B)
        (set! i (+ i 1))
      })))

;; 状态条：基准/范围读数 + 波形缩略图 + 电平表
(define draw_status
  (lambda ()
    (define fw 0)
    (define flash #f)
    (define fr 0) (define fg 0) (define fb 0)
    (Gfx.fill_rect 0 STAT_Y SW (- SH STAT_Y) BG_R BG_G BG_B)
    (Gfx.fill_rect 0 STAT_Y SW 1 EDGE_R EDGE_G EDGE_B)
    ;; 电平/范围读数：刚调节过的 900ms 内用强调色（C 版是限时叠加显示，这里常显 + 高亮）
    (set! flash (and (>= now_ms adjust_ms) (< (- now_ms adjust_ms) ADJUST_FLASH_MS)))
    (if flash { (set! fr ACC_R) (set! fg ACC_G) (set! fb ACC_B) }
              { (set! fr DIM_R) (set! fg DIM_G) (set! fb DIM_B) })
    (Gfx.draw_text lv_text 6 (+ STAT_Y 5) fr fg fb)
    (Gfx.draw_text rg_text 62 (+ STAT_Y 5) fr fg fb)
    (draw_wave)
    ;; 电平表（0 = −60dBFS 以下，255 = 满量程；接近削顶转警示色）
    (Gfx.draw_text "电平" 198 (+ STAT_Y 5) DIM_R DIM_G DIM_B)
    (Gfx.rounded_rect RMS_X RMS_Y RMS_W RMS_H 3 3 3 3 TRK_R TRK_G TRK_B 1)
    (set! fw (Math.floor (/ (* RMS_W rms) 255)))
    (if (> fw 0)
        (if (>= rms 236)
            (Gfx.rounded_rect RMS_X RMS_Y fw RMS_H 3 3 3 3 DAN_R DAN_G DAN_B 1)
            (Gfx.rounded_rect RMS_X RMS_Y fw RMS_H 3 3 3 3 ACC_R ACC_G ACC_B 1))
        #f)))

;; 全部静态部件 + 底栏按钮
(define draw_chrome
  (lambda ()
    {
      (Gfx.set_font 1)                    ; 字模是宿主单例状态：本 App 统一 12px 抗锯齿
      (draw_top)
      (draw_freq_axis)
      (draw_time_axis)
      (Gfx.fill_rect 0 PLOT_BOTTOM SW (- BAR_Y PLOT_BOTTOM)
                     EDGE_R EDGE_G EDGE_B)          ; 绘图区与按钮行之间的分隔带
      (Gfx.fill_rect 0 BAR_Y SW (- SH BAR_Y) BG_R BG_G BG_B)
      (BTN.draw_state b_base_dn (BTN.armed b_base_dn))
      (BTN.draw_state b_base_up (BTN.armed b_base_up))
      (BTN.draw_state b_rng_dn (BTN.armed b_rng_dn))
      (BTN.draw_state b_rng_up (BTN.armed b_rng_up))
      (BTN.draw_state b_pause (BTN.armed b_pause))
      (BTN.draw_state b_pal (BTN.armed b_pal))
      (draw_status)
    }))

;; 绘图区里的提示（等数据 / 无麦克风）；这两种状态不滚动，故可以直接覆盖
(define draw_overlay
  (lambda ()
    (define wait_ms 0)
    (if (== overlay_kind 1)
        {
          ;; 已开麦但还没收到第一帧（浏览器授权是异步的）
          (set! wait_ms (- now_ms mic_open_ms))
          (draw_center "正在等待麦克风数据…" 104 TXT_R TXT_G TXT_B)
          (if (> wait_ms 1200)
              (draw_center "浏览器需 HTTPS 访问并允许麦克风权限" 124 WRN_R WRN_G WRN_B)
              (draw_center "首次打开约需 1 秒" 124 DIM_R DIM_G DIM_B))
        }
        #f)
    (if (== overlay_kind 2)
        {
          (Gfx.rounded_rect 30 60 260 112 12 12 12 12 EDGE_R EDGE_G EDGE_B 1)
          (Gfx.rounded_rect 31 61 258 110 11 11 11 11 CARD_R CARD_G CARD_B 1)
          (draw_center "麦克风不可用" 76 DAN_R DAN_G DAN_B)
          (draw_center "需要 HTTPS 访问并授权麦克风" 98 DIM_R DIM_G DIM_B)
          (draw_center "点下方「重试」重新打开" 114 DIM_R DIM_G DIM_B)
          (BTN.draw_state b_retry (BTN.armed b_retry))
        }
        #f)))

;; ============================================================================
;; 绘制：声谱列（滚动 + 新列 + 静态部件）
;; ============================================================================

;; 清空绘图区像素（不动 overlay_kind：调用方决定"提示"是否继续显示）
(define clear_plot
  (lambda ()
    (Gfx.fill_rect PLOT_X0 PLOT_Y0 (- SW PLOT_X0) PLOT_H 0 0 0)))

;; 画一列：整屏上滚 ROW_H 行 → 底部按强度逐频点着色 → 重画静态部件 → 推屏
(define draw_column
  (lambda ()
    (define x 0)
    (define v 0)
    (Gfx.scroll_up ROW_H)          ; 历史整体上移一行（滚动会整屏标脏）
    (while (< x nbins)
      {
        (set! v (Audio.bin x))
        (Gfx.fill_rect (+ PLOT_X0 x) (- PLOT_BOTTOM ROW_H) 1 ROW_H
                       (get_item cur_r v) (get_item cur_g v) (get_item cur_b v))
        (set! x (+ x 1))
      })
    (draw_chrome)
    (Gfx.refresh)
    (set! dirty #f)))

;; 全屏重绘（启动/恢复/换配色）：清绘图区 + 静态部件 + 提示 + 推屏
(define draw_all
  (lambda ()
    {
      (Gfx.fill_rect 0 0 SW SH BG_R BG_G BG_B)
      (clear_plot)
      (draw_chrome)
      (draw_overlay)
      (Gfx.refresh)
      (set! dirty #f)
      (set! want_clear #f)
    }))

;; 只重画静态部件与提示（不破坏已滚出的声谱历史）
(define redraw
  (lambda ()
    {
      (draw_chrome)
      (draw_overlay)
      (Gfx.refresh)
      (set! dirty #f)
    }))

;; ============================================================================
;; 每拍：采集 → 绘制
;; ============================================================================

(define update_column
  (lambda ()
    (define rc 0)
    (define drawn 0)
    (define stop #f)
    (while (and (not stop) (< drawn COL_MAX))
      {
        (set! rc (Audio.read))
        (if (== rc -1)
            {
              (set! mic_ok #f)
              (set! mic_err #t)
              (refresh_state)
              (set! overlay_kind 2)
              (set! want_clear #t)
              (set! stop #t)
              (display "[spectrogram] mic error") (newline)
            }
            #f)
        (if (== rc 1)
            {
              (set! rms (Audio.rms))
              (set! mic_cols (+ mic_cols 1))
              ;; 从"等数据/无麦克风"提示画面切回瀑布：先把提示擦掉（这两态没有历史可保留）
              (if (not (== overlay_kind 0))
                  { (clear_plot) (set! overlay_kind 0) (refresh_state) (set! dirty #t) }
                  #f)
              (if (== mic_cols 1)
                  { (refresh_state) (set! dirty #t) }
                  #f)
              (if paused #f (draw_column))
              (set! drawn (+ drawn 1))
            }
            (set! stop #t))
      })))

(define diag
  (lambda ()
    (if (>= (- ntick last_diag) 60)
        {
          (set! last_diag ntick)
          (display "[spectrogram] mic=") (display mic_ok)
          (display " cols=") (display mic_cols)
          (display " rms=") (display rms)
          (display " base=") (display base)
          (display " range=") (display range)
          (display " pal=") (display pal_mode)
          (display " paused=") (display paused)
          (display " hist=") (display (Math.floor (/ (* ROWS sec_per_row) 100)))
          (display " rows=")
          (if (> (- now_ms diag_ms) 0)
              (display (Math.floor (/ (* 1000.0 (- mic_cols diag_cols)) (- now_ms diag_ms))))
              (display 0))
          (newline)
          (set! diag_cols mic_cols)
          (set! diag_ms now_ms)
        }
        #f)))

(define tick
  (lambda ()
    {
      ;; 顺序要紧：App.resumed 是"取走即清零"，须早于 (EX.check)（它也读一次）
      (if (App.resumed) (on_resume) #f)
      (EX.check)                     ; 硬 A/ESC 与右上角「退出」（可能阻塞）
      (set! ntick (+ ntick 1))
      (set! now_ms (System.timestamp))
      (handle_buttons)
      (handle_keys)
      (update_column)
      ;; 静止两态（无麦克风 / 等数据）：呼吸指示心跳 + 定期自动重试开麦
      (if (or (not mic_ok) (== mic_cols 0))
          {
            (if (>= (- now_ms blink_ms) BLINK_MS)
                { (set! blink_ms now_ms) (set! dirty #t) }
                #f)
            (if (and (not mic_ok) (>= (- now_ms retry_ms) RETRY_MS))
                { (set! retry_ms now_ms) (open_mic) }
                #f)
          }
          #f)
      (if want_clear
          (draw_all)
          (if dirty (redraw) #f))
      (diag)
    }))

;; ============================================================================
;; 启动 / 退出
;; ============================================================================

(define on_exit
  (lambda ()
    {
      (display "[spectrogram] bye") (newline)
      (Audio.close)                  ; 停采集、释放分析缓冲（幂等）
      (System.clear_interval timer_id)
    }))

(build_all_pals)
(refresh_readout)
(style_all_btns)
(BTN.set_label b_pal (get_item pal_names 0))
(set! now_ms (System.timestamp))
(open_mic)
;; 用实际打开的参数校准频率轴（info 在未打开时也返回默认值，不会失败）
(define audio_info (Audio.info))
(if (list? audio_info)
    {
      (set! fft_size (get_item audio_info 1))
      (set! max_bins (get_item audio_info 2))
      (set! bin_hz (/ (get_item audio_info 0) fft_size))
      (set! nbins (Math.floor (min2 MAX_BINS max_bins)))
      (set! sec_per_row (/ fft_size (get_item audio_info 0)))
      (set! px_per_sec (/ ROW_H sec_per_row))
    }
    #f)
(refresh_state)
(set! diag_ms now_ms)
(set! diag_cols 0)
(draw_all)
(set! timer_id (System.set_interval TICK_MS tick))
(EX.install_with redraw)
(EX.on_exit on_exit)
(display "[spectrogram] start nbins=") (display nbins)
(display " bins_hz=") (display (Math.floor (* 10 bin_hz)))
(newline)
