;@id=bird
;@title=百鸟园
;@version=2026-11-28
;; ============================================================================
;; 百鸟园（Bird）—— 灵机应用程序：**实时鸟鸣合成器**
;;
;; 不是"播放鸟叫录音"，而是按**鸣管物理模型**现场算出来的：每个采样点积分一个
;; 二阶非线性振子（Mindlin–Laje），按音高轮廓逐点反解张力 β，再经"按音高跟踪的
;; 声道"带通、包络、颤音、调幅、失谐第二声门与倍频谐波支路合成一个音节；若干
;; 音节按乐句排布，若干"个体"（音高/时长/响度各差一点点）按模式组织成独唱、
;; 应答与齐鸣。
;;
;; 算法出处：https://github.com/emwstudio/esp32-perch（ESP32 鸟叫合成器，移植自
;;   sha5b/M5Stack-Lyrebird 的 AudioWorklet 合成器）。本仓只**参照算法**：
;;   标定表由 infer/tools/gen_bird_calib.py 自己测（不搬运它的 CC BY-NC-SA 数据），
;;   鸟种/曲目是本仓手工编写的 12 种常见鸟（见 infer/am_native_Bird.c 的数据段）。
;;
;; 分工（关键，与音乐盒/声谱仪同一哲学）：**合成与音频全在 C 侧**
;;   (native Bird) 提供 鸟种表 / 选种 / 模式 / 播放 / 每拍喂音频 / 声部快照 /
;;   电平 / 音高·速度·音量旋钮，详见 infer/am_native_Bird.h。理由：单声部每采样
;;   就要积分 20~45 步 ODE 并查两次标定表，4 声部量级 10⁸ flops/s——比解释器在
;;   wasm32 上的能力高三个数量级（对照 (native Audio) 里 1024 点 FFT 的估算）。
;;   本文件只管界面、交互与可视化，一行 DSP 也没有。
;;
;; 界面（320x240；全部触屏控件，宿主不再为触屏派生按键事件）：
;;   · 顶栏：标题「百鸟园」+ 当前鸟名 + 序号 3/12 + 右上角「退出」（app/lib/exit.scm）；
;;   · 舞台：夜空渐变 + 音高网格；左侧一只栖枝小鸟（随电平张嘴、扇翅、发光）；
;;     右侧是**音高星图**——每个声部一条按个体着色的发光轨迹（横轴 = 时间，
;;     纵轴 = 对数音高），发声时轨迹从左向右生长，静止后淡出；
;;   · 信息条：拟声词 + 拉丁学名 + 六段电平表；
;;   · 控件：‹ 上一只 / 模式（独唱·应答·齐鸣）/ 播放·暂停 / 下一只 › / 音量 − + /
;;     音高 ±（模拟器旋钮）/ 速度 ±；
;;   · 点按舞台空白处 = 让它**立刻叫一声**（试听单个音节，带随机音高与时长微差）。
;;   实体键盘兜底（桌面/终端）：←/→ 换鸟，空格播放/暂停，M 换模式，↑/↓ 音量，
;;   P/O 音高 ±，T/R 速度 ±，C 叫一声。
;;
;; 节拍：33ms 一拍（≈30Hz）——(Bird.pump) 每拍把音频喂给扬声器（HAL 自带背压，
;;   队列满就什么都不做），界面每拍重绘舞台。音频节拍在 C 侧按**采样计数**推进，
;;   所以主循环慢只会让画面掉帧，不会让鸟走音。
;;
;; 降级：没有音频设备（打桩平台 / 浏览器自动播放被挂起 / node 回归台）时引擎转
;;   "静音走带"——照常合成、按墙钟丢弃采样，星图与电平表照样活；诊断行
;;   `[bird] ... rendered=N` 会继续增长，端到端回归据此判定"合成链路在跑"。
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
(native Bird)
(native System)
(native String)
(native Math)
(native App)

(import BTN "app/lib/button.scm")   ; 通用按钮控件（圆角底/描边/居中标签/按住高亮/点按）
(import EX "app/lib/exit.scm")      ; 硬 A/ESC 与右上角「退出」按钮 → 全局退出确认框

;; ============================================================================
;; 版面（320x240 基准，按实际画布宽高推导）
;; ============================================================================

(define SW (Gfx.width))
(define SH (Gfx.height))

(define TICK_MS 33)          ; 一拍 33ms（≈30Hz）：够密的触屏采样，也不至于把时间片吃光
(define DIAG_EVERY 30)       ; 诊断行间隔（≈1s；端到端回归靠它读状态）

(define TOP_H 22)                              ; 顶栏
(define STAGE_TOP TOP_H)                       ; 舞台 22..172
(define STAGE_H 150)
(define STAGE_BOT (+ STAGE_TOP STAGE_H))       ; = 172
(define INFO_Y STAGE_BOT)                      ; 信息条 172..198
(define INFO_H 26)
(define ROW_A_Y (- SH 42))                     ; 控件第一行 198..218
(define ROW_A_H 20)
(define ROW_B_Y (- SH 20))                     ; 控件第二行 220..238
(define ROW_B_H 18)

;; 星图区域（舞台右侧）与小鸟（左侧）
(define TRAIL_X0 (+ STAGE_TOP 74))             ; 轨迹起点 = 96（左侧留给小鸟）
(define TRAIL_X1 (- SW 6))                     ; 轨迹右端 = 314
(define TRAIL_N 18)                            ; 每条轨迹保留的点数
(define TRAIL_STEP 7)                          ; 相邻点的水平间距（px）→ 历史跨度 119px
(define TRAIL_W 2)                             ; 线宽

;; 音高轴（对数）：250Hz..9kHz 覆盖本 App 全部鸟种
(define F_MIN 250.0)
(define F_MAX 9000.0)
(define LOG_F_MIN (Math.log F_MIN))
(define LOG_SPAN (- (Math.log F_MAX) LOG_F_MIN))

(define BIRD_VOICES 8)       ; (native Bird) 声部池大小（vf0/venv/vtag 的下标上限）

;; ============================================================================
;; 调色板
;; ============================================================================

(define C_SKY_TOP '(10 16 30))
(define C_SKY_BOT '(24 44 56))
(define C_TITLE '(232 242 250))
(define C_DIM '(136 156 178))
(define C_DIM2 '(96 116 138))
(define C_ACCENT '(150 235 190))
(define C_WARM '(255 206 130))

;; 四个"个体"的轨迹配色（按 vtag 取）
(define TAG_COLORS (cons '(120 240 180)
                    (cons '(120 200 255)
                     (cons '(255 200 120)
                      (cons '(230 150 255) (cdr '(0)))))))

(define MODE_NAMES (cons "独唱" (cons "应答" (cons "齐鸣" (cdr '(0))))))

;; ============================================================================
;; 小工具
;; ============================================================================

(define min2 (lambda (a b) (if (< a b) a b)))
(define max2 (lambda (a b) (if (> a b) a b)))
(define clampv (lambda (v lo hi) (max2 lo (min2 hi v))))

;; 环形索引：mod 的符号跟随被除数（fmod 语义），负数要先掰回正区间
(define wrapn (lambda (i n) (mod (+ (mod i n) n) n)))

(define color_of
  (lambda (tag)
    (if (or (< tag 0) (> tag 3)) (get_item TAG_COLORS 0) (get_item TAG_COLORS tag))))

;; 音高（Hz）→ 星图纵坐标（对数映射；f<=0 时落在舞台中线）
(define pitch_y
  (lambda (f)
    (define v 0.5)
    (if (> f 0.0) (set! v (/ (- (Math.log f) LOG_F_MIN) LOG_SPAN)) #f)
    (set! v (clampv v 0.0 1.0))
    (Math.floor (+ (+ STAGE_TOP 8) (* (- 1.0 v) (- STAGE_H 18))))))

;; ============================================================================
;; 状态
;; ============================================================================

(define n_species (Bird.count))
(define species 0)            ; 当前鸟种
(define mode 0)               ; 0 独唱 / 1 应答 / 2 齐鸣
(define playing #t)
(define volume 96)
(define pitch_pct 0)
(define tempo_pct 0)
(define name_str "")
(define latin_str "")
(define mnemonic_str "")
(define range_lo 0.0)
(define range_hi 0.0)

(define vol_str "音量96")       ; 只在数值变化时重建（每帧拼串会制造大量短命对象）
(define tune_str "音高+0  速度+0")

(define active_now 0)         ; 本拍在发声的声部数（(Bird.vf0) 顺手数出来）
(define sing_ticks 0)         ; 累计"有鸟在唱"的拍数（诊断行用，回归可判"确实在唱"）

(define ntick 0)
(define last_diag 0)
(define rendered 0)           ; 累计送入设备的采样数（诊断行断言"合成在跑"）
(define dirty #t)             ; 静态件需要重绘

;; 星图：每个声部一条轨迹（环缓冲）——tr_y 是"列表的列表"，其余是数值列表
(define tr_y (cdr '(0)))
(define tr_head (cdr '(0)))
(define tr_n (cdr '(0)))
(define tr_last (cdr '(0)))

;; 按钮
(define b_prev (BTN.new))
(define b_next (BTN.new))
(define b_mode (BTN.new))
(define b_play (BTN.new))
(define b_voldn (BTN.new))
(define b_volup (BTN.new))
(define b_pdn (BTN.new))
(define b_pup (BTN.new))
(define b_tdn (BTN.new))
(define b_tup (BTN.new))

(define timer_id 0)
(define hold_idx -1)
(define hold_next 0)
(define HOLD_DELAY 350)
(define HOLD_REPEAT 90)

;; ============================================================================
;; 星图数据
;; ============================================================================

(define mk_row
  (lambda ()
    (define l (cdr '(0)))
    (define k 0)
    (while (< k TRAIL_N) { (push l 0.0) (set! k (+ k 1)) })
    l))

(define _i 0)
(while (< _i BIRD_VOICES)
  {
    (push tr_y (mk_row))
    (push tr_head 0)
    (push tr_n 0)
    (push tr_last (- 0 999))
    (set! _i (+ _i 1)) })

(define clear_trails
  (lambda ()
    (define i 0)
    (define k 0)
    (define row (cdr '(0)))
    (while (< i BIRD_VOICES)
      {
        (set_item! tr_head i 0)
        (set_item! tr_n i 0)
        (set_item! tr_last i (- 0 999))
        (set! row (get_item tr_y i))
        (set! k 0)
        (while (< k TRAIL_N)
          { (set_item! row k 0.0) (set! k (+ k 1)) })
        (set! i (+ i 1)) })))

;; 把当前在发声的声部推进星图（每拍一次）
(define update_trails
  (lambda ()
    (define i 0)
    (define f 0.0)
    (define y 0)
    (define h 0)
    (define last 0)
    (define row (cdr '(0)))
    (define cnt 0)
    (while (< i BIRD_VOICES)
      {
        (set! f (Bird.vf0 i))
        (if (> f 0.0)
            {
              (set! y (pitch_y f))
              (set! last (get_item tr_last i))
              ;; 音高跳变超过 26px = 换了一声：旧轨迹断掉，从新点重新长
              (if (or (< last (- 0 900)) (> (Math.abs (- y last)) 26))
                  {
                    (set_item! tr_n i 0)
                    (set! h 0)
                  }
                  (set! h (get_item tr_head i)))
              (set! row (get_item tr_y i))
              (set_item! row h y)
              (set_item! tr_head i (wrapn (+ h 1) TRAIL_N))
              (if (< (get_item tr_n i) TRAIL_N)
                  (set_item! tr_n i (+ (get_item tr_n i) 1))
                  #f)
              (set_item! tr_last i y)
              (set! cnt (+ cnt 1))
            }
            #f)
        (set! i (+ i 1)) })
    (set! active_now cnt)))

;; ============================================================================
;; 数据同步（选种 / 模式 / 音量…）
;; ============================================================================

(define refresh_info
  (lambda ()
    (define info (cdr '(0)))
    (set! name_str (Bird.name species))
    (set! latin_str (Bird.latin species))
    (set! mnemonic_str (Bird.mnemonic species))
    (set! info (Bird.voicing species))
    (if (list? info)
        {
          (set! range_lo (get_item info 2))
          (set! range_hi (get_item info 3))
        }
        {
          (set! range_lo 400.0)
          (set! range_hi 4000.0)
        })
    (if (null? name_str) (set! name_str "?") #f)
    (if (null? latin_str) (set! latin_str "") #f)
    (if (null? mnemonic_str) (set! mnemonic_str "") #f)))

(define select_species
  (lambda (idx)
    {
      (set! species (Bird.select idx))
      (refresh_info)
      (clear_trails)
      (set! dirty #t)
    }))

(define step_species
  (lambda (d)
    (select_species (wrapn (+ species d) n_species))))

(define cycle_mode
  (lambda ()
    {
      (set! mode (Bird.set_mode (wrapn (+ mode 1) 3)))
      (clear_trails)
      (refresh_info)
      (set! dirty #t)
    }))

;; 数值 → 人读文本：**只在值变化时重建**（manual §3.4：每帧拼串 = 每帧制造垃圾）
(define sign_of (lambda (v) (if (< v 0) "" "+")))

(define rebuild_text
  (lambda ()
    {
      (set! vol_str (String.concat "音量" (String.atom_to_string volume)))
      (set! tune_str
            (String.concat "音高" (String.concat (sign_of pitch_pct)
              (String.concat (String.atom_to_string pitch_pct)
                (String.concat "  速度" (String.concat (sign_of tempo_pct)
                  (String.concat (String.atom_to_string tempo_pct) "%")))))))
    }))

(define set_volume_to
  (lambda (v)
    {
      (set! volume (Bird.set_volume (clampv v 0 128)))
      (rebuild_text)
      (set! dirty #t)
    }))

(define set_pitch_to
  (lambda (v)
    {
      (set! pitch_pct (Bird.set_pitch (clampv v (- 0 50) 50)))
      (rebuild_text)
      (set! dirty #t)
    }))

(define set_tempo_to
  (lambda (v)
    {
      (set! tempo_pct (Bird.set_tempo (clampv v (- 0 50) 50)))
      (rebuild_text)
      (set! dirty #t)
    }))

(define toggle_play
  (lambda ()
    (if playing
        (Bird.pause)
        (Bird.play))
    (set! playing (Bird.is_playing))
    (clear_trails)
    (set! dirty #t)))

;; ============================================================================
;; 绘制：舞台
;; ============================================================================

(define draw_sky
  (lambda ()
    (define i 0)
    (define bands 12)
    (define t 0.0)
    (define r 0)
    (define g 0)
    (define b 0)
    (while (< i bands)
      {
        (set! t (/ i (- bands 1)))
        (set! r (Math.floor (+ (get_item C_SKY_TOP 0)
                              (* t (- (get_item C_SKY_BOT 0) (get_item C_SKY_TOP 0))))))
        (set! g (Math.floor (+ (get_item C_SKY_TOP 1)
                              (* t (- (get_item C_SKY_BOT 1) (get_item C_SKY_TOP 1))))))
        (set! b (Math.floor (+ (get_item C_SKY_TOP 2)
                              (* t (- (get_item C_SKY_BOT 2) (get_item C_SKY_TOP 2))))))
        (Gfx.fill_rect 0 (+ STAGE_TOP (* i 13)) SW 13 r g b)
        (set! i (+ i 1)) })
    ;; 天边的几颗暗星（固定位置，纯装饰）
    (Gfx.fill_dot_aa 22.0 34.0 1.0 210 230 255)
    (Gfx.fill_dot_aa 268.0 42.0 0.9 200 220 255)
    (Gfx.fill_dot_aa 302.0 30.0 1.1 210 235 255)
    (Gfx.fill_dot_aa 186.0 28.0 0.8 190 215 240)))

;; 音高网格：1k/2k/4k/8k 四条淡线 + 左侧小标签
(define draw_grid
  (lambda ()
    (define i 0)
    (define y 0)
    (define f 1000.0)
    (while (< i 4)
      {
        (set! f (* 1000.0 (Math.pow 2.0 i)))
        (set! y (pitch_y f))
        (Gfx.draw_line_aa_mode TRAIL_X0 y TRAIL_X1 y 1 46 74 88 3)
        (if (== i 0) (Gfx.draw_text "1k" (- TRAIL_X0 22) (- y 6) 70 100 118) #f)
        (if (== i 1) (Gfx.draw_text "2k" (- TRAIL_X0 22) (- y 6) 70 100 118) #f)
        (if (== i 2) (Gfx.draw_text "4k" (- TRAIL_X0 22) (- y 6) 70 100 118) #f)
        (if (== i 3) (Gfx.draw_text "8k" (- TRAIL_X0 22) (- y 6) 70 100 118) #f)
        (set! i (+ i 1)) })))

;; 当前鸟种的音域标尺（舞台左边界的一条淡竖线）
(define draw_range
  (lambda ()
    (define y0 (pitch_y range_hi))
    (define y1 (pitch_y range_lo))
    (if (< y0 y1) (Gfx.draw_line_aa_mode 92 y0 92 y1 2 56 92 80 3) #f)))

;; 一只栖枝的小鸟：随电平张嘴、扇翅、发光
(define draw_bird
  (lambda ()
    (define lv (Math.sqrt (/ (Bird.level) 255.0)))
    (define open 0)
    (define flap 0)
    (define cx 44)
    (define cy 104)
    (set! open (Math.floor (* 6.0 lv)))
    (set! flap (Math.floor (* 9.0 lv)))
    ;; 枝干 + 两片叶子
    (Gfx.draw_line_aa_mode 6 146 92 138 3 74 62 52 1)
    (Gfx.draw_line_aa_mode 66 141 74 128 2 74 62 52 1)
    (Gfx.fill_dot_aa 77.0 125.0 5.0 38 84 70)
    (Gfx.fill_dot_aa 60.0 131.0 4.0 38 84 70)
    ;; 腿（踩在枝上）+ 尾羽
    (Gfx.draw_line_aa_mode 40 118 38 137 2 138 98 58 1)
    (Gfx.draw_line_aa_mode 50 120 52 136 2 138 98 58 1)
    (Gfx.draw_triangle 18 126 34 114 24 100 1 138 98 58 1)
    (Gfx.draw_triangle 22 128 36 118 28 108 1 160 116 70 1)
    ;; 身体 / 头 / 腹部
    (Gfx.fill_circle_aa cx cy 17 186 138 78 1)
    (Gfx.fill_circle_aa 62 92 10 186 138 78 1)
    (Gfx.fill_circle_aa 48 (+ cy 7) 10 226 190 132 1)
    ;; 翅膀：发声时抬起张开
    (Gfx.draw_triangle (- cx 12) (+ cy 4) (+ cx 6) (- (- (- cy 8) flap) 0) (+ cx 11) (+ cy 14) 1 138 98 58 1)
    ;; 眼睛
    (Gfx.fill_dot_aa 66.0 89.0 2.0 24 28 36)
    (Gfx.fill_dot_aa 66.6 88.4 0.8 240 248 255)
    ;; 喙：上喙 / 下喙，张开量跟着电平
    (Gfx.draw_triangle 69 89 86 (- 89 open) 69 92 1 246 196 96 1)
    (Gfx.draw_triangle 69 93 84 (+ 93 open) 69 96 1 220 164 70 1)
    ;; 鸣叫光晕
    (if (> lv 0.03)
        {
          (Gfx.fill_glow_aa_add 74.0 96.0 (+ 14.0 (* 16.0 lv)) 150 235 190 (* 0.30 lv))
          (Gfx.fill_glow_aa cx cy (+ 20.0 (* 18.0 lv)) 255 206 130 (* 0.20 lv))
        }
        #f)))

;; 音高星图：每个声部一条按个体着色的加色折线（越新越亮）
(define draw_trails
  (lambda ()
    (define v 0)
    (define k 0)
    (define n 0)
    (define h 0)
    (define row (cdr '(0)))
    (define col (cdr '(0)))
    (define y0 0)
    (define y1 0)
    (define x0 0)
    (define x1 0)
    (define fade 1.0)
    (define r 0)
    (define g 0)
    (define b 0)
    (while (< v BIRD_VOICES)
      {
        (set! n (get_item tr_n v))
        (if (> n 1)
            {
              (set! h (get_item tr_head v))
              (set! row (get_item tr_y v))
              (set! col (color_of (Bird.vtag v)))
              (set! k 0)
              (while (< k (- n 1))
                {
                  (set! y0 (get_item row (wrapn (- (- h 1) k) TRAIL_N)))
                  (set! y1 (get_item row (wrapn (- (- h 2) k) TRAIL_N)))
                  (set! x0 (- TRAIL_X1 (* k TRAIL_STEP)))
                  (set! x1 (- TRAIL_X1 (* (+ k 1) TRAIL_STEP)))
                  (if (and (> x1 TRAIL_X0) (> (Math.abs y0) 0) (> (Math.abs y1) 0))
                      {
                        (set! fade (/ (- TRAIL_N k) TRAIL_N))
                        (set! r (Math.floor (* (get_item col 0) fade)))
                        (set! g (Math.floor (* (get_item col 1) fade)))
                        (set! b (Math.floor (* (get_item col 2) fade)))
                        (Gfx.draw_line_aa_mode x0 y0 x1 y1 TRAIL_W r g b 3)
                      }
                      #f)
                  (set! k (+ k 1)) })
              ;; 发声位置的亮点
              (if (> (Bird.venv v) 0)
                  {
                    (set! y0 (get_item row (wrapn (- h 1) TRAIL_N)))
                    (Gfx.fill_dot_aa TRAIL_X1 y0 1.8 (get_item col 0) (get_item col 1) (get_item col 2))
                    (Gfx.fill_glow_aa_add TRAIL_X1 y0 7.0 (get_item col 0) (get_item col 1) (get_item col 2) 0.22)
                  }
                  #f)
            }
            #f)
        (set! v (+ v 1)) })))

(define draw_stage
  (lambda ()
    {
      (draw_sky)
      (draw_grid)
      (draw_trails)
      (draw_range)
      (draw_bird)
      (Gfx.draw_text tune_str 6 (- STAGE_BOT 14) 96 116 138)
    }))

;; ============================================================================
;; 绘制：静态件（顶栏 / 信息条 / 控件）
;; ============================================================================

(define draw_top
  (lambda ()
    {
      (Gfx.fill_rect 0 0 SW TOP_H 26 34 50)
      (Gfx.fill_rect 0 (- TOP_H 1) SW 1 44 60 82)
      (Gfx.set_font 2)
      (Gfx.draw_text "百鸟园" 6 3 232 242 250)
      (Gfx.set_font 1)
      (Gfx.draw_text name_str 62 5 150 235 190)
      (Gfx.draw_text (String.concat (String.atom_to_string (+ species 1))
                                    (String.concat "/" (String.atom_to_string n_species)))
                     232 5 136 156 178)
    }))

(define draw_info
  (lambda ()
    (define lv (Bird.level))
    (define seg 0)
    (define i 0)
    (define r 0)
    (define g 0)
    (define b 0)
    (set! seg (Math.floor (+ 0.4 (* (Math.sqrt (/ lv 255.0)) 6.0))))
    (Gfx.fill_rect 0 INFO_Y SW INFO_H 20 28 42)
    (Gfx.draw_text mnemonic_str 6 (+ INFO_Y 1) 255 206 130)
    (Gfx.draw_text latin_str 6 (+ INFO_Y 12) 96 116 138)
    ;; 六段电平表（右对齐）
    (while (< i 6)
      {
        (if (< i seg)
            {
              (set! r 150) (set! g 235) (set! b 190)
              (if (>= i 4) { (set! r 255) (set! g 206) (set! b 130) } #f)
            }
            {
              (set! r 44) (set! g 56) (set! b 72)
            })
        (Gfx.fill_rect (+ 236 (* i 13)) (+ INFO_Y 6) 11 10 r g b)
        (set! i (+ i 1)) })))

(define draw_row_bg
  (lambda ()
    {
      (Gfx.fill_rect 0 ROW_A_Y SW (- SH ROW_A_Y) 22 30 44)
      (Gfx.draw_text vol_str 262 (+ ROW_A_Y 4) 136 156 178)
    }))

(define paint_button
  (lambda (b)
    {
      (BTN.set_bg b 34 46 62)
      (BTN.set_active_bg b 116 172 226)
      (BTN.set_text_color b 226 236 248)
      (BTN.set_border b 66 84 104)
      (BTN.set_radius b 5)
      (BTN.set_font b 1)
    }))

(define style_buttons
  (lambda ()
    {
      (paint_button b_prev) (paint_button b_next) (paint_button b_mode) (paint_button b_play)
      (paint_button b_voldn) (paint_button b_volup)
      (paint_button b_pdn) (paint_button b_pup) (paint_button b_tdn) (paint_button b_tup)
      (BTN.set_rect b_prev 4 ROW_A_Y 24 ROW_A_H)
      (BTN.set_rect b_mode 32 ROW_A_Y 46 ROW_A_H)
      (BTN.set_rect b_play 82 ROW_A_Y 58 ROW_A_H)
      (BTN.set_rect b_next 144 ROW_A_Y 24 ROW_A_H)
      (BTN.set_rect b_voldn 172 ROW_A_Y 40 ROW_A_H)
      (BTN.set_rect b_volup 216 ROW_A_Y 40 ROW_A_H)
      (BTN.set_rect b_pdn 4 ROW_B_Y 42 ROW_B_H)
      (BTN.set_rect b_pup 50 ROW_B_Y 42 ROW_B_H)
      (BTN.set_rect b_tdn 96 ROW_B_Y 42 ROW_B_H)
      (BTN.set_rect b_tup 142 ROW_B_Y 42 ROW_B_H)
    }))

(define sync_labels
  (lambda ()
    {
      (BTN.set_label b_prev "<")
      (BTN.set_label b_next ">")
      (BTN.set_label b_mode (get_item MODE_NAMES mode))
      (BTN.set_label b_play (if playing "暂停" "播放"))
      (BTN.set_label b_voldn "音量-")
      (BTN.set_label b_volup "音量+")
      (BTN.set_label b_pdn "音高-")
      (BTN.set_label b_pup "音高+")
      (BTN.set_label b_tdn "速度-")
      (BTN.set_label b_tup "速度+")
    }))

(define draw_buttons
  (lambda ()
    {
      (BTN.draw_state b_prev (BTN.armed b_prev))
      (BTN.draw_state b_mode (BTN.armed b_mode))
      (BTN.draw_state b_play (BTN.armed b_play))
      (BTN.draw_state b_next (BTN.armed b_next))
      (BTN.draw_state b_voldn (BTN.armed b_voldn))
      (BTN.draw_state b_volup (BTN.armed b_volup))
      (BTN.draw_state b_pdn (BTN.armed b_pdn))
      (BTN.draw_state b_pup (BTN.armed b_pup))
      (BTN.draw_state b_tdn (BTN.armed b_tdn))
      (BTN.draw_state b_tup (BTN.armed b_tup))
      ;; 提示文字每拍重绘：先补一块与控件行同色的底。不补底会**反复叠加抗锯齿覆盖率**，
      ;; 几十拍之后笔画糊成一团（实测：字挤在一起看不出"点按画面叫一声"）——
      ;; 其它文字都在"每拍重画过的底"上，只有这一句没有。
      (Gfx.fill_rect 190 ROW_B_Y 130 ROW_B_H 22 30 44)
      (Gfx.draw_text "点按画面叫一声" 192 (+ ROW_B_Y 4) 96 116 138)
    }))

;; 全屏重绘（切种/切模式/首帧/退出确认框后擦残影）
(define draw_all
  (lambda ()
    {
      (draw_top)
      (draw_stage)
      (draw_info)
      (draw_row_bg)
      (draw_buttons)
      (EX.draw_button)
      (Gfx.refresh)
      (set! dirty #f)
    }))

;; 每拍重绘（星图与电平在动）
(define draw_frame
  (lambda ()
    {
      (draw_stage)
      (draw_info)
      (draw_buttons)
      (Gfx.refresh)
    }))

;; ============================================================================
;; 交互
;; ============================================================================

(define handle_keys
  (lambda ()
    (define k 0)
    (set! k (Key.take_press))
    (while (> k 0)
      {
        (if (== k 3) (step_species -1) #f)                      ; ←
        (if (== k 4) (step_species 1) #f)                       ; →
        (if (or (== k 32) (== k 13)) (toggle_play) #f)           ; 空格 / 回车
        (if (or (== k 109) (== k 77)) (cycle_mode) #f)           ; 'm'/'M'
        (if (== k 1) (set_volume_to (+ volume 8)) #f)            ; ↑
        (if (== k 2) (set_volume_to (- volume 8)) #f)            ; ↓
        (if (or (== k 112) (== k 80)) (set_pitch_to (+ pitch_pct 5)) #f)       ; 'p' 音高 +
        (if (or (== k 111) (== k 79)) (set_pitch_to (- pitch_pct 5)) #f)      ; 'o' 音高 −
        (if (or (== k 116) (== k 84)) (set_tempo_to (+ tempo_pct 5)) #f)       ; 't' 速度 +
        (if (or (== k 114) (== k 82)) (set_tempo_to (- tempo_pct 5)) #f)      ; 'r' 速度 −
        (if (or (== k 99) (== k 67))                              ; 'c' 叫一声（暂停时先恢复）
            { (if (not playing) (toggle_play) #f) (Bird.chirp) }
            #f)
        (set! k (Key.take_press)) })))

;; 点按控件（BTN.track 内部把"落在自己身上"的按下/松开沿吞掉）
(define handle_buttons
  (lambda ()
    {
      (if (BTN.track b_prev) (step_species -1) #f)
      (if (BTN.track b_next) (step_species 1) #f)
      (if (BTN.track b_mode) (cycle_mode) #f)
      (if (BTN.track b_play) (toggle_play) #f)
      ;; 六个"旋钮"键：**只**让 BTN.track 结算触屏边沿与按住状态，动作交给下面的
      ;; 按住连发通路（按下沿即生效一次、350ms 后按 90ms 连发）。两条通路都动会
      ;; 一次点按走两步（实测：音量 96 一下变 108）。
      (BTN.track b_voldn)
      (BTN.track b_volup)
      (BTN.track b_pdn)
      (BTN.track b_pup)
      (BTN.track b_tdn)
      (BTN.track b_tup)
    }))

;; 舞台空白处按下 = 让它叫一声（控件已经把自己的沿吞掉，剩下的才是"空白处"）
(define handle_tap
  (lambda ()
    (define y 0)
    (if (Touch.take_press)
        {
          (set! y (Touch.y))
          (if (and (> y STAGE_TOP) (< y STAGE_BOT))
              {
                (if (not playing) (toggle_play) #f)
                (Bird.chirp)
              }
              #f)
        }
        #f)))

;; 按住"音量/音高/速度 ±"连发（与音乐盒、声谱仪同一手感）
(define armed_idx
  (lambda ()
    (define idx -1)
    (if (BTN.armed b_voldn) (set! idx 4) #f)
    (if (BTN.armed b_volup) (set! idx 5) #f)
    (if (BTN.armed b_pdn) (set! idx 6) #f)
    (if (BTN.armed b_pup) (set! idx 7) #f)
    (if (BTN.armed b_tdn) (set! idx 8) #f)
    (if (BTN.armed b_tup) (set! idx 9) #f)
    idx))

(define any_armed
  (lambda ()
    (if (or (BTN.armed b_play) (BTN.armed b_mode) (BTN.armed b_prev) (BTN.armed b_next)
            (>= (armed_idx) 0)) #t #f)))

;; 按下沿走大步（点一下就明显），按住连发走小步（细调）
(define step_of
  (lambda (small big first)
    (if first big small)))

(define act_knob
  (lambda (idx first)
    (if (== idx 4) (set_volume_to (- volume (step_of 4 8 first))) #f)
    (if (== idx 5) (set_volume_to (+ volume (step_of 4 8 first))) #f)
    (if (== idx 6) (set_pitch_to (- pitch_pct (step_of 1 5 first))) #f)
    (if (== idx 7) (set_pitch_to (+ pitch_pct (step_of 1 5 first))) #f)
    (if (== idx 8) (set_tempo_to (- tempo_pct (step_of 1 5 first))) #f)
    (if (== idx 9) (set_tempo_to (+ tempo_pct (step_of 1 5 first))) #f)))

;; ============================================================================
;; 诊断行（端到端回归的判据来源；人眼也可在终端里看）
;; ============================================================================

(define diag
  (lambda ()
    (if (>= (- ntick last_diag) DIAG_EVERY)
        {
          (set! last_diag ntick)
          (display "[bird] sp=") (display species)
          (display " name=") (display name_str)
          (display " mode=") (display mode)
          (display " playing=") (display playing)
          (display " voices=") (display active_now)
          (display " sing=") (display sing_ticks)
          (display " lvl=") (display (Bird.level))
          (display " pitch=") (display pitch_pct)
          (display " tempo=") (display tempo_pct)
          (display " vol=") (display volume)
          (display " rendered=") (display rendered)
          (newline)
        }
        #f)))

;; ============================================================================
;; 主循环
;; ============================================================================

(define tick
  (lambda ()
    (define idx -1)
    {
      (if (App.resumed) (set! dirty #t) #f)
      (EX.check)                                 ; 硬 A/ESC 与右上角「退出」（可能阻塞）
      (set! ntick (+ ntick 1))
      (set! rendered (+ rendered (Bird.pump)))    ; 喂音频（HAL 自带背压）
      (handle_keys)
      (handle_buttons)
      (handle_tap)
      (update_trails)
      (if (> active_now 0) (set! sing_ticks (+ sing_ticks 1)) #f)
      ;; 按住连发（音量/音高/速度）
      (set! idx (armed_idx))
      (if (< idx 0) (set! hold_idx -1)
          (if (not (== idx hold_idx))
              { (set! hold_idx idx) (act_knob idx #t) (set! hold_next (+ (System.timestamp) HOLD_DELAY)) }
              (if (>= (System.timestamp) hold_next)
                  { (act_knob idx #f) (set! hold_next (+ (System.timestamp) HOLD_REPEAT)) }
                  #f)))
      (if dirty
          {
            (sync_labels)
            (draw_all)
          }
          ;; 暂停且无人按住时没有会变的东西：不重绘（掌机上省电，画面也本来就是静止的）
          (if (or playing (any_armed)) (draw_frame) #f))
      (diag)
    }))

;; ============================================================================
;; 启动 / 退出
;; ============================================================================

(define on_exit
  (lambda ()
    {
      (Bird.stop)                                ; 停合成、释放扬声器设备
      (display "[bird] bye") (newline)
      (System.clear_interval timer_id)
    }))

(style_buttons)
(set! volume (Bird.set_volume 96))
(set! mode (Bird.set_mode 0))
(select_species 0)
(rebuild_text)
(Bird.play)
(set! playing (Bird.is_playing))
(sync_labels)
(EX.install_with draw_all)
(EX.on_exit on_exit)
(draw_all)
(set! timer_id (System.set_interval TICK_MS tick))
(display "[bird] start species=") (display n_species)
(display " rate=") (display (Bird.rate))
(newline)
