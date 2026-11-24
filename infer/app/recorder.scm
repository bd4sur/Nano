;@id=recorder
;@title=录音机
;@version=2026-11-24
;; ============================================================================
;; 录音机（Recorder）—— 灵机应用程序（Scheme App）
;;
;; 功能：把麦克风信号**边录边压**成 MP3（MPEG-1 Layer III · 44100Hz · 单声道 ·
;; 320kbps CBR），以流式方式写入 music/ 目录，随时可在音乐盒里播放。
;;
;; 分工（关键）：
;;   · 采集 / 编码 / 落盘全部在 C 侧（(native Rec)，设计见 infer/am_native_Rec.h；
;;     MP3 内核是 infer/vendor/aqua_mp3/ 的 Aqua 编码器，从 Aqua 仓库原样拷贝，
;;     算法零改动）；
;;   · 本文件只做**节拍驱动 + 界面 + 文件列表**：每拍 (Rec.pump 4) 推进一小块
;;     （读一帧 1152 采样 → 编码 → 立即追加写文件）。一帧 = 26.12ms 音频，
;;     wasm32 上编码一帧约 1~3ms（>10× 实时），所以"边录边压"既不丢样也不卡界面。
;;     为什么要 pump 而不是一次到底：wasm 是单线程，编码必须切成小块夹在渲染帧之间，
;;     否则录音期间界面会僵住（这正是"尽力实时"的具体做法）。
;;
;; 界面（320x240，自上而下）：
;;   · 顶栏：红点 + 「录音机」 + 状态（就绪 / 录音中 / 麦克风不可用）+ 右上「退出」；
;;   · 波形区（64px 高）：最近一帧 1152 个采样的包络（74 列，按 sqrt 幅度映射，
;;     弱信号也看得见），带中心线与上下刻度参考线；
;;   · 电平表 + 统计行（已录时长 · 字节 · 帧数 + 右侧「×实时」/「已保存」）；
;;   · 录音记录：复用 app/lib/menu_list.scm 的通用菜单控件（20px 行高、点行播放、
;;     拖动滚动、滚动条），条目 = music/ 下 rec-*.mp3（倒序，最新在前）；
;;   · 底部**加高的录音按钮**：计时数字**画在按钮里**（不放大字号），
;;     空闲「● 录制」⇄ 录音中「■ 00:12.3」。
;;
;; 落盘与持久化：录音写在 <业务资源根>/music/rec-YYYYMMDD-HHMMSS.mp3。WASM 目标在收尾
;;   时经 am_native_fs_on_write 写进浏览器 IndexedDB（nano-store），刷新页面后由
;;   loader.js 挂载回来 —— 与 (Fs.write) 的持久化通路同一条。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数；没有一元取负 → (- 0 x)；
;;   没有 let/list/map/append；define 只写在模块顶层或 lambda 体顶层（循环体内不 define）；
;;   cond 每子句只接受单个表达式；/ 与 mod 恒浮点（索引用 Math.floor 收敛）；
;;   while 循环体不是独立词法区域；0 是真值（布尔一律 #t/#f）；'() 是模块级静态槽，
;;   故空表一律 (cdr '(0)) 现造，绝不对 '() 直接 push。
;; ============================================================================

(native Gfx)
(native Touch)
(native Key)
(native Rec)
(native Fs)
(native App)
(native System)
(native String)
(native Math)

(import BTN "app/lib/button.scm")     ; 通用按钮控件（圆角底/描边/居中标签/按住高亮/点按）
(import EX "app/lib/exit.scm")        ; 硬 A/ESC 与右上角「退出」按钮 → 全局退出确认框
(import FSX "app/lib/fsx.scm")        ; 文本宽度 / 尺寸格式化 / 后缀判断
(import ML "app/lib/menu_list.scm")   ; 通用滚动菜单列表（高亮 / 拖动滚动 / 滚动条 / 点选）

;; ============================================================================
;; 常量：版面（320x240 基准，按实际画布宽高推导）
;; ============================================================================

(define SW (Gfx.width))
(define SH (Gfx.height))

(define TICK_MS 33)          ; 一拍 33ms（30fps）：足够跟上 38.6 帧/秒的音频速率
(define PUMP_MAX 4)          ; 每拍最多推进的音频帧数（4 帧 ≈ 104ms 音频）
(define SAMPLE_RATE 44100)   ; 与 (native Rec) 的固定格式一致（(Rec.info) 也会回报）
(define FRAME_SAMPLES 1152)  ; 一帧采样数（两个 granule）
(define FRAME_MS 26.122449)  ; 1152/44100·1000
(define KBPS 320)            ; CBR 码率
(define BYTES_PER_SEC 40000) ; 320kbps / 8

(define TOP_H 22)            ; 顶栏（标题 + 状态 + 右上角「退出」）

;; 波形区（= "频谱图"）：最近一帧的时域包络。高度 64px（旧版条带只有 18px）
(define VIZ_X 12)
(define VIZ_Y 26)
(define VIZ_W (- SW 24))     ; 296
(define VIZ_H 64)            ; 26..90
(define WAVE_COLS 74)        ; 每列 4px（3px 柱 + 1px 缝）
(define WAVE_STEP 4)
(define WAVE_BAR 3)
(define WAVE_PER_BAR 15)     ; 每列覆盖 15 个采样（74×15 = 1110 ≤ 1152）

(define METER_X 12)
(define METER_Y 94)
(define METER_W (- SW 24))
(define METER_H 6)

(define STAT_Y 104)          ; 统计行（已录时长 · 字节 · 帧数），12px 行高

(define LIST_HDR_Y 120)      ; 「录音记录」标题行
(define LIST_Y 134)          ; 菜单控件视口（134..194）
(define LIST_H 60)
(define ROW_H 20)            ; 行高（菜单控件的行高参数；比默认 15px 高）

(define BAR_Y 198)           ; 底部按钮条（198..240）
(define REC_BTN_W 220)
(define REC_BTN_H 34)        ; 加高：计时数字直接画在按钮里

;; 配色（深色主题，与仓库内其它灵机 App 同一套观感）
(define BG_R 12)  (define BG_G 16)  (define BG_B 22)
(define EDGE_R 48) (define EDGE_G 58) (define EDGE_B 74)
(define SUR_R 38)  (define SUR_G 47)  (define SUR_B 61)
(define ACC_R 56)  (define ACC_G 189) (define ACC_B 248)
(define TXT_R 237) (define TXT_G 242) (define TXT_B 249)
(define DIM_R 140) (define DIM_G 152) (define DIM_B 168)
(define TRK_R 52)  (define TRK_G 63)  (define TRK_B 79)
(define WAV_R 90)  (define WAV_G 220) (define WAV_B 200)
(define GRID_R 26) (define GRID_G 34) (define GRID_B 43)   ; 波形区网格线
(define REC_R 248) (define REC_G 96)  (define REC_B 96)    ; 录音红
(define WRN_R 255) (define WRN_G 178) (define WRN_B 92)    ; 警示（无麦克风）

;; ============================================================================
;; 状态
;; ============================================================================

(define rec_on #f)            ; 是否正在录音
(define mic_err #f)           ; 是否发生过麦克风硬错误（(Rec.pump) 报 -1）
(define cur_path "")          ; 本次录音的资源根相对路径
(define last_path "")         ; 最近一次**成功保存**的路径（状态提示用）
(define last_ms 0)            ; 最近一次保存的时长（毫秒）
(define ms 0)                 ; 本次已录音频时长（毫秒，来自 (Rec.stats)）
(define frames 0)             ; 本次已编成的帧数
(define bytes 0)              ; 本次已写出的字节数
(define level 0)              ; 最近一帧电平 0~255
(define rec_start_ms 0)       ; 本次开始录音的**墙钟**时刻（算编码速度用）
(define speed_x 0.0)          ; 编码速度（×实时；= 音频毫秒 / 墙钟毫秒）

(define time_text "00:00.0")  ; 计时文本（只在显示分辨率变化时重建；画在按钮里）
(define shown_key -1)         ; 上次重建时的 key = floor(ms/100)
(define stat_text "")
(define btn_text "● 录制")    ; 按钮标签缓存（避免同一内容反复 set_label）

(define files (cdr '(0)))     ; 录音文件名（**倒序**：最新在前）
(define fsizes (cdr '(0)))    ; 与 files 一一对应的字节数
(define menu_ready #f)        ; 菜单控件的字模度量/几何是否已初始化
(define last_sel 0)
(define last_scroll 0)

(define status_text "就绪")

(define ntick 0)
(define now_ms 0)
(define last_diag 0)
(define timer_id 0)
(define dirty #t)             ; 需要重画静态部件
(define want_clear #t)        ; 需要整屏重画
(define blink_ms 0)           ; 无麦克风时状态点的呼吸心跳

;; 时间戳哨兵一律用 0（不能用负数）：wasm32 上 (System.timestamp) 返回纪元毫秒
;; mod 2^27，做 (- now -100000) 这类减法会装不进有符号立即数（见 app/spectrogram.scm
;; 的同名注释）。规矩：只对两个活时间戳做差，哨兵用 0 + 显式判 0。

;; ============================================================================
;; 通用小工具
;; ============================================================================

(define new_list (lambda () (cdr '(0))))
(define max2 (lambda (a b) (if (> a b) a b)))

(define draw_right
  (lambda (s right y r g b)
    (Gfx.draw_text s (- right (FSX.text_w s)) y r g b)))

;; 以 pre 为前缀（返回 1/0，与 FSX.ends_with 同口径）
(define starts_with
  (lambda (s pre)
    (define ls (String.length s))
    (define lp (String.length pre))
    (define i 0)
    (define ok 1)
    (if (< ls lp)
        0
        { (while (< i lp)
            { (if (== (String.charCodeAt i s) (String.charCodeAt i pre))
                  #f
                  (set! ok 0))
              (set! i (+ i 1)) })
          ok })))

(define pad2
  (lambda (n)
    (if (< n 10)
        (String.concat "0" (String.atom_to_string n))
        (String.atom_to_string n))))

;; 毫秒 → "mm:ss.d"（上限 99:59.9）
(define fmt_time
  (lambda (msec)
    (define t (Math.floor msec))
    (define mi 0)
    (define se 0)
    (define de 0)
    (if (< t 0) (set! t 0) #f)
    (if (> t 5999999) (set! t 5999999) #f)
    (set! mi (Math.floor (/ t 60000)))
    (set! se (Math.floor (/ (mod t 60000) 1000)))
    (set! de (Math.floor (/ (mod t 1000) 100)))
    (String.concat (pad2 mi)
      (String.concat ":" (String.concat (pad2 se)
        (String.concat "." (String.atom_to_string de)))))))

;; 毫秒 → "12.3s"（统计行 / 列表行用；恒保留一位小数）
(define fmt_secs
  (lambda (msec)
    (String.concat (FSX.fixed1 (/ msec 1000.0)) "s")))

;; 采样幅度 → 像素高度：v = sqrt(|s| / 32768) 而不是线性。
;; 为什么不用线性：麦克风正常说话/音乐只到满量程的 10%~40%，线性画出来的包络贴着
;; 中线几乎看不见（旧版 18px 高的条带里就是几条断断续续的短横）。sqrt 曲线把弱信号
;; 抬起来（0.4 幅值 → 0.63 满高），视觉上接近常见的电平表压缩。
(define amp_px
  (lambda (s half)
    (define a (Math.abs s))
    (set! a (if (> a 32768) 32768 a))
    (Math.floor (* (/ (Math.sqrt a) 181.019336) half))))

;; ============================================================================
;; 录音：开始 / 推进 / 收尾
;; ============================================================================

(define refresh_status
  (lambda ()
    (if rec_on
        (set! status_text "录音中")
        (if mic_err
            (set! status_text "麦克风不可用")
            (set! status_text "就绪")))))

(define update_stats
  (lambda ()
    (define st (Rec.stats))
    (if (list? st)
        {
          (set! ms (get_item st 1))
          (set! frames (get_item st 2))
          (set! bytes (get_item st 3))
        }
        #f)
    (set! level (Rec.level))))

;; 按钮标签：空闲「● 录制」；录音中「■ 时间」（计时数字在按钮里，不放大字号）
(define set_btn_label
  (lambda (s)
    (if (not (String.equals s btn_text))
        {
          (set! btn_text s)
          (BTN.set_label b_rec s)
          (set! dirty #t)
        }
        #f)))

(define start_rec
  (lambda ()
    (define p (Rec.start))
    (if (null? p)
        {
          (set! rec_on #f)
          (set! mic_err #t)
          (refresh_status)
          (set! dirty #t)
          (set! want_clear #t)
          (display "[recorder] start failed") (newline)
        }
        {
          (set! rec_on #t)
          (set! mic_err #f)
          (set! cur_path p)
          (set! ms 0)
          (set! frames 0)
          (set! bytes 0)
          (set! level 0)
          (set! shown_key -1)
          (set! last_path "")
          (set! rec_start_ms now_ms)
          (set! speed_x 0.0)
          (BTN.set_bg b_rec REC_R REC_G REC_B)
          (BTN.set_active_bg b_rec 200 64 64)
          (set_btn_label "■ 00:00.0")
          (refresh_status)
          (set! dirty #t)
          (display "[recorder] start path=") (display p) (newline)
        })))

;; 收尾：返回是否保存成功（(Rec.stop) 的列表：(毫秒 帧数 字节数 路径)）
(define stop_rec
  (lambda ()
    (define r (Rec.stop))
    (define p #null)
    (define saved #f)
    (set! rec_on #f)
    (if (list? r)
        {
          (set! last_ms (get_item r 0))
          (set! p (get_item r 3))
          (if (null? p)
              (set! saved #f)
              {
                (set! saved #t)
                (set! last_path p)
              })
        }
        #f)
    (if saved
        (display "[recorder] saved path=")
        (display "[recorder] stopped (nothing saved)"))
    (if saved (display last_path) #f)
    (display " ms=") (display last_ms)
    (display " frames=") (display frames)
    (display " bytes=") (display bytes)
    (newline)
    (BTN.set_bg b_rec SUR_R SUR_G SUR_B)
    (BTN.set_active_bg b_rec 116 172 226)
    (set_btn_label "● 录制")
    ;; 保留本次的时长/帧数/字节作为"已保存"读数（下次 (Rec.start) 时清零）
    (set! level 0)
    (set! speed_x 0.0)
    (set! shown_key -1)
    (refresh_texts)
    (refresh_status)
    (refresh_files)
    (set! dirty #t)
    (set! want_clear #t)
    saved))

;; 每拍推进一小块；返回本拍编成的帧数
(define do_pump
  (lambda ()
    (define rc 0)
    (set! rc (Rec.pump PUMP_MAX))
    (if (== rc -1)
        {
          ;; 硬错误（权限被拒 / 采集中断）：库已自行收尾，这里只切回"未录音"画面
          (set! mic_err #t)
          (set! rec_on #f)
          (set! frames 0)
          (set! bytes 0)
          (set! ms 0)
          (set! speed_x 0.0)
          (BTN.set_bg b_rec SUR_R SUR_G SUR_B)
          (BTN.set_active_bg b_rec 116 172 226)
          (set_btn_label "● 录制")
          (refresh_status)
          (refresh_files)
          (set! dirty #t)
          (set! want_clear #t)
          (display "[recorder] mic error") (newline)
          0 }
        rc)))

;; ============================================================================
;; 录音记录（music/rec-*.mp3）：枚举 → 菜单控件条目 / 播放
;; ============================================================================

;; 行文本：名字 + 时长 + 大小（控件按视口宽自动裁剪，超长用 … 收尾）
(define row_label
  (lambda (nm sz)
    (String.concat nm
      (String.concat "  "
        (String.concat (fmt_secs (* 1000.0 (/ sz BYTES_PER_SEC)))
          (String.concat "  " (FSX.human_size sz)))))))

;; 枚举 music/ 下的 rec-*.mp3（倒序：最新在前），重建菜单控件条目
(define refresh_files
  (lambda ()
    (define raw (Fs.list "music"))
    (define rows (new_list))
    (define n 0)
    (define i 0)
    (define nm "")
    (define sz 0)
    (set! files (new_list))
    (set! fsizes (new_list))
    (if (list? raw)
        {
          (set! n (length raw))
          ;; Fs.list 已按名字升序 → 倒序遍历得到"最新在前"（文件名内嵌时间戳）
          (set! i (- n 1))
          (while (>= i 0)
            { (set! nm (get_item raw i))
              (if (and (== (FSX.ends_with nm ".mp3") 1) (== (starts_with nm "rec-") 1))
                  {
                    (set! sz (Fs.size (String.concat "music/" nm)))
                    (if (null? sz) (set! sz 0) #f)
                    (push files nm)
                    (push fsizes sz)
                    (push rows (ML.item (row_label nm sz) 208 216 230 0 0 255 255 255))
                  }
                  #f)
              (set! i (- i 1)) })
        }
        #f)
    (if menu_ready
        {
          (ML.reset rows ROW_H)
          (ML.set_sel 0)
          (set! last_sel 0)
          (set! last_scroll 0)
        }
        #f)
    (display "[recorder] files=") (display (length files)) (newline)))

(define play_at
  (lambda (k)
    (define name #null)
    (define abs "")
    (if (and (>= k 0) (< k (length files))) (set! name (get_item files k)) #f)
    (if (null? name)
        #f
        {
          (set! abs (Fs.abspath (String.concat "music/" name)))
          (if (null? abs)
              (display "[recorder] abspath failed")
              {
                (display "[recorder] play ") (display name)
                (display " path=") (display abs)
                (display " ok=") (display (App.launch_with "musicbox" abs))
                (newline)
              })
        })))

(define play_sel
  (lambda ()
    (play_at (ML.sel_of))))

;; ============================================================================
;; 绘制
;; ============================================================================

(define draw_top
  (lambda ()
    (define sr 255) (define sg 255) (define sb 255)
    (define sx 0)
    (Gfx.fill_rect 0 0 SW TOP_H BG_R BG_G BG_B)
    (Gfx.fill_rect 0 TOP_H SW 1 EDGE_R EDGE_G EDGE_B)
    ;; 标志：录音红点（录音中呼吸；空闲为暗红）
    (if rec_on
        (if (== (Math.floor (mod (/ now_ms 600.0) 2)) 0)
            (Gfx.fill_circle_aa 18 11 4 REC_R REC_G REC_B 1)
            (Gfx.fill_circle_aa 18 11 3 REC_R REC_G REC_B 120))
        (Gfx.fill_circle_aa 18 11 3 150 88 88 1))
    (Gfx.draw_text "录音机" 26 5 TXT_R TXT_G TXT_B)
    ;; 状态文字右对齐到「退出」按钮左侧
    (if rec_on
        { (set! sr REC_R) (set! sg REC_G) (set! sb REC_B) }
        (if mic_err
            { (set! sr WRN_R) (set! sg WRN_G) (set! sb WRN_B) }
            { (set! sr DIM_R) (set! sg DIM_G) (set! sb DIM_B) }))
    (set! sx (- (- SW (+ EX.btn_w 10)) (FSX.text_w status_text)))
    (Gfx.draw_text status_text sx 5 sr sg sb)
    ;; 录音中：状态点常亮；无麦克风时呼吸（画面几乎静止，靠它证明"还活着"）
    (if mic_err
        (if (== (Math.floor (mod (/ now_ms 400.0) 2)) 0)
            (Gfx.fill_circle_aa (- sx 9) 11 3 WRN_R WRN_G WRN_B 1)
            (Gfx.fill_circle_aa (- sx 9) 11 2 WRN_R WRN_G WRN_B 64))
        (if rec_on
            (Gfx.fill_circle_aa (- sx 9) 11 3 REC_R REC_G REC_B 1)
            (Gfx.fill_circle_aa (- sx 9) 11 3 ACC_R ACC_G ACC_B 1)))
    (EX.draw_button)))

;; 波形区：最近一帧 1152 采样的**包络**（74 列，每列 15 个采样取最大幅度，上下对称画）。
;;
;; 为什么用"最大幅度 + 上下对称"而不是"每列 min/max"：一列只覆盖 15 个采样
;; （0.34ms），对 1kHz 这种周期信号而言还不到半个周期，min/max 会随相位在
;; "半波/全波"之间跳变，画出来是一排高低不齐的梳齿（实测截图确认）。取该列
;; 最大 |采样| 画成对称条，得到的才是**幅度包络**——对语音/音乐正是想要的那种
;; 录音机电平视图，而且柱高随幅度单调变化。
(define draw_viz
  (lambda ()
    (define i 0)
    (define k 0)
    (define s 0)
    (define pk 0)
    (define y_h 0)
    (define cy 0)
    (define half 0)
    (define bx 0)
    (set! cy (+ VIZ_Y (Math.floor (/ VIZ_H 2))))
    (set! half (- (Math.floor (/ VIZ_H 2)) 2))
    ;; 底板 + 描边
    (Gfx.rounded_rect VIZ_X VIZ_Y VIZ_W VIZ_H 8 8 8 8 EDGE_R EDGE_G EDGE_B 1)
    (Gfx.rounded_rect (+ VIZ_X 1) (+ VIZ_Y 1) (- VIZ_W 2) (- VIZ_H 2)
                      7 7 7 7 8 11 16 1)
    ;; 参考线：上下 ±half/2 各一条（很暗）、中心一条
    (Gfx.fill_rect (+ VIZ_X 3) (- cy (Math.floor (/ half 2))) (- VIZ_W 6) 1
                   GRID_R GRID_G GRID_B)
    (Gfx.fill_rect (+ VIZ_X 3) (+ cy (Math.floor (/ half 2))) (- VIZ_W 6) 1
                   GRID_R GRID_G GRID_B)
    (Gfx.fill_rect (+ VIZ_X 3) cy (- VIZ_W 6) 1 20 46 46)
    ;; 包络柱
    (while (< i WAVE_COLS)
      {
        (set! pk 0)
        (set! k 0)
        (while (< k WAVE_PER_BAR)
          {
            (set! s (Math.abs (Rec.wave (+ (* i WAVE_PER_BAR) k))))
            (if (> s pk) (set! pk s) #f)
            (set! k (+ k 1))
          })
        (set! y_h (amp_px pk half))
        (set! bx (+ (+ VIZ_X 1) (* i WAVE_STEP)))
        (Gfx.fill_rect bx (- cy y_h) WAVE_BAR (max2 1 (* 2 y_h))
                       WAV_R WAV_G WAV_B)
        (set! i (+ i 1))
      })))

;; 电平表（0 = −60dBFS 以下，255 = 满量程；接近削顶转红色）
(define draw_meter
  (lambda ()
    (define fw 0)
    (Gfx.rounded_rect METER_X METER_Y METER_W METER_H 3 3 3 3 TRK_R TRK_G TRK_B 1)
    (set! fw (Math.floor (/ (* METER_W level) 255)))
    (if (> fw 0)
        (if (>= level 236)
            (Gfx.rounded_rect METER_X METER_Y fw METER_H 3 3 3 3 REC_R REC_G REC_B 1)
            (Gfx.rounded_rect METER_X METER_Y fw METER_H 3 3 3 3 ACC_R ACC_G ACC_B 1))
        #f)))

;; 统计行：已录时长 · 字节 · 帧数（左）+ ×实时 / 已保存（右）；
;; 麦克风打不开时整行换成原因提示（左边那串 0.0s · 0 B 毫无意义，还会与提示挤在一起）
(define draw_stats
  (lambda ()
    (Gfx.fill_rect 0 STAT_Y SW 12 BG_R BG_G BG_B)
    (if (and mic_err (not rec_on))
        (Gfx.draw_text "麦克风不可用：需 HTTPS 且允许权限" 12 STAT_Y WRN_R WRN_G WRN_B)
        {
          (Gfx.draw_text stat_text 12 STAT_Y DIM_R DIM_G DIM_B)
          (if rec_on
              (draw_right (String.concat (FSX.fixed1 speed_x) "× 实时")
                          (- SW 12) STAT_Y ACC_R ACC_G ACC_B)
              (if (== (String.length last_path) 0)
                  #f
                  (draw_right "已保存" (- SW 12) STAT_Y WAV_R WAV_G WAV_B)))
        })))

;; 录音记录：标题行（左：标题 / 右：格式说明）+ 菜单控件视口
(define draw_list
  (lambda ()
    (Gfx.fill_rect 0 LIST_HDR_Y SW 12 BG_R BG_G BG_B)
    (Gfx.draw_text "录音记录" 12 LIST_HDR_Y DIM_R DIM_G DIM_B)
    (draw_right "320k CBR · 44.1k · 单声道" (- SW 12) LIST_HDR_Y DIM_R DIM_G DIM_B)
    (Gfx.fill_rect 0 LIST_Y SW LIST_H BG_R BG_G BG_B)
    (ML.draw)
    ;; 空列表提示：必须画在控件绘制**之后**（控件不涂空行，但绘制顺序保持"内容在上"）
    (if (== (length files) 0)
        (Gfx.draw_text_centered 1 "还没有录音：点下方「● 录制」开始"
                                (Math.floor (/ SW 2)) (+ LIST_Y 24)
                                DIM_R DIM_G DIM_B 1)
        #f)))

(define draw_bar
  (lambda ()
    (Gfx.fill_rect 0 BAR_Y SW (- SH BAR_Y) BG_R BG_G BG_B)
    (Gfx.fill_rect 0 BAR_Y SW 1 EDGE_R EDGE_G EDGE_B)
    (BTN.draw_state b_rec (BTN.armed b_rec))))

;; 静态部件（顶栏 / 波形 / 电平 / 统计 / 记录列表 / 底部按钮）
(define draw_chrome
  (lambda ()
    {
      (Gfx.set_font 1)               ; 字模是宿主单例状态：本 App 统一 12px 抗锯齿
      (draw_top)
      (draw_viz)
      (draw_meter)
      (draw_stats)
      (draw_list)
      (draw_bar)
    }))

(define draw_all
  (lambda ()
    {
      (Gfx.fill_rect 0 0 SW SH BG_R BG_G BG_B)
      (draw_chrome)
      (EX.draw_button)
      (Gfx.refresh)
      (set! dirty #f)
      (set! want_clear #f)
    }))

(define redraw
  (lambda ()
    {
      (draw_chrome)
      (EX.draw_button)
      (Gfx.refresh)
      (set! dirty #f)
    }))

;; 录音中每拍刷新的"活"区域（顶栏状态点 / 波形 / 电平 / 统计 / 按钮里的计时）
;; 顶栏也一起画：录音红点要做呼吸闪烁，且 (EX.draw_button) 每帧都要重新登记热区。
;; 按钮也一起画：计时数字在按钮标签里，标签变了要随拍重画。
(define draw_live
  (lambda ()
    {
      (draw_top)
      (draw_viz)
      (draw_meter)
      (draw_stats)
      (draw_bar)
    }))

;; ============================================================================
;; 输入
;; ============================================================================

(define toggle_rec
  (lambda ()
    (if rec_on (stop_rec) (start_rec))))

(define handle_buttons
  (lambda ()
    (if (BTN.track b_rec) (toggle_rec) #f)))

;; 列表：菜单控件负责"拖动滚动 + 轻点选中"；轻点在条目上 = 播放该录音。
;; 录音中仍然允许滚动，但不接受"点行播放"（正在写的文件不该被播放器抢音频设备）。
;;
;; list_arm：一次手势的"起手点"是否落在列表视口内。菜单控件是按**触屏电平**工作的
;; （它自己记 t_prev），若不设这道闸，按住录音按钮上下滑也会把列表拖着滚。
(define list_arm -1)

(define handle_list
  (lambda ()
    (define hit -1)          ; -1 = 本拍没有"点选"（0 是合法条目下标，不能当默认值）
    (define in_viz 0)
    (set! in_viz (if (and (>= (Touch.y) LIST_Y) (< (Touch.y) (+ LIST_Y LIST_H))) 1 0))
    (if (Touch.is_pressed)
        (if (< list_arm 0) (set! list_arm in_viz) #f)
        #f)
    (if (== list_arm 1) (set! hit (ML.step_touch)) #f)
    (if (>= hit 0)
        {
          (set! dirty #t)
          (if rec_on
              { (display "[recorder] list tap ignored (recording)") (newline) }
              (play_at hit))
        }
        #f)
    (if (not (Touch.is_pressed)) (set! list_arm -1) #f)
    (if (not (== (ML.sel_of) last_sel))
        { (set! last_sel (ML.sel_of)) (set! dirty #t) }
        #f)
    (if (not (== (ML.scroll_of) last_scroll))
        { (set! last_scroll (ML.scroll_of)) (set! dirty #t) }
        #f)))

(define handle_key
  (lambda (k)
    (if (or (== k 32) (== k 114) (== k 82)) (toggle_rec) #f)   ; 空格 / 'r' = 录制·停止
    (if (== k 13) (play_sel) #f)                              ; 回车 = 播放选中
    (if (== k 1) { (ML.move -1) (set! dirty #t) } #f)         ; ↑
    (if (== k 2) { (ML.move 1) (set! dirty #t) } #f)          ; ↓
    (if (or (== k 3) (== k 30)) { (ML.page -1) (set! dirty #t) } #f)  ; ← / PgUp
    (if (or (== k 4) (== k 31)) { (ML.page 1) (set! dirty #t) } #f)   ; → / PgDn
    (if (or (== k 102) (== k 70))                             ; 'f'/'F' = 重新枚举录音目录
        { (refresh_files) (set! dirty #t) }
        #f)))

(define handle_keys
  (lambda ()
    (define k 0)
    (set! k (Key.take_press))
    (while (> k 0)
      { (handle_key k) (set! k (Key.take_press)) })))

;; ============================================================================
;; 每拍
;; ============================================================================

;; 计时/统计文本：只在显示分辨率（100ms）变化时重建。
;; 为什么不每拍重建：stat_text 里有字节数/帧数，每拍都会变；30fps 下每秒要拼
;; 十来个短串（manual §3.4 的短串驻留开销），而人眼在 100ms 粒度上已经看不出来。
(define refresh_texts
  (lambda ()
    (define key 0)
    (set! key (Math.floor (/ ms 100.0)))
    (if (not (== key shown_key))
        {
          (set! shown_key key)
          (set! time_text (fmt_time ms))
          ;; 录音中时长已经画在按钮里（不重复），统计行只报体积与帧数；
          ;; 空闲时（含"已保存"读数）才带上时长
          (set! stat_text
                (if rec_on
                    (String.concat (FSX.human_size bytes)
                      (String.concat " · "
                        (String.concat (String.atom_to_string frames) " 帧")))
                    (String.concat (String.concat (fmt_secs ms) " · ")
                      (String.concat (FSX.human_size bytes)
                        (String.concat " · "
                          (String.concat (String.atom_to_string frames) " 帧"))))))
          ;; 计时数字就在按钮里（不放大），随 100ms 粒度刷新标签
          (if rec_on
              (set_btn_label (String.concat "■ " time_text))
              #f)
        }
        #f)))

(define diag
  (lambda ()
    (if (>= (- ntick last_diag) 30)
        {
          (set! last_diag ntick)
          (display "[recorder] on=") (display rec_on)
          (display " ms=") (display (Math.floor ms))
          (display " frames=") (display frames)
          (display " bytes=") (display bytes)
          (display " level=") (display level)
          (display " files=") (display (length files))
          (display " sel=") (display (ML.sel_of))
          (display " scroll=") (display (ML.scroll_of))
          (display " err=") (display mic_err)
          (display " rt=") (display (FSX.fixed1 speed_x))
          (newline)
        }
        #f)))

(define tick
  (lambda ()
    {
      ;; 顺序要紧：App.resumed 是"取走即清零"，须早于 (EX.check)（它也读一次）
      (if (App.resumed)
          {
            ;; 从后台/音乐盒返回：挂起期间没有执行机会，录音一律收尾（不写坏文件）
            (if rec_on (stop_rec) #f)
            (refresh_files)
            (set! dirty #t)
            (set! want_clear #t)
            (display "[recorder] resume") (newline)
          }
          #f)
      (EX.check)                     ; 硬 A/ESC 与右上角「退出」（可能阻塞）
      (set! ntick (+ ntick 1))
      (set! now_ms (System.timestamp))
      (handle_buttons)
      (handle_keys)
      (handle_list)
      (if rec_on
          {
            (do_pump)
            (update_stats)
            (refresh_texts)
            ;; 编码速度：音频毫秒 / 墙钟毫秒（只对两个活时间戳做差）
            (if (> (- now_ms rec_start_ms) 0)
                (set! speed_x (/ ms (- now_ms rec_start_ms)))
                #f)
          }
          (if mic_err
              {
                ;; 无麦克风：状态点呼吸心跳（画面几乎静止，靠它证明"还活着"）
                (if (or (== blink_ms 0) (>= (- now_ms blink_ms) 400))
                    { (set! blink_ms now_ms) (set! dirty #t) }
                    #f)
              }
              #f))
      ;; 绘制分工：整屏重画 > 静态部件（dirty）> 只刷"活"区域（录音中）
      (if want_clear
          (draw_all)
          (if dirty
              (redraw)
              (if rec_on
                  { (draw_live) (Gfx.refresh) }
                  #f)))
      (diag)
    }))

;; ============================================================================
;; 启动 / 退出
;; ============================================================================

(define on_exit
  (lambda ()
    {
      (display "[recorder] bye") (newline)
      (if rec_on (stop_rec) #f)       ; 收尾：关麦 + 关文件 + 落盘（幂等）
      (System.clear_interval timer_id)
    }))

(define b_rec (BTN.make "● 录制" (Math.floor (/ (- SW REC_BTN_W) 2))
                        (+ BAR_Y 4) REC_BTN_W REC_BTN_H))
(BTN.set_font b_rec 2)               ; 16px 抗锯齿（宿主第二大固定字模；不缩放）
(BTN.set_text_color b_rec 236 241 248)
(BTN.set_bg b_rec SUR_R SUR_G SUR_B)
(BTN.set_active_bg b_rec 116 172 226)
(BTN.set_radius b_rec 8)
(BTN.set_border b_rec EDGE_R EDGE_G EDGE_B)

;; 菜单控件的字模度量（12px 抗锯齿：半角/全角步进取自宿主，墨高 12、框顶内缩 1、行顶空隙 2）
(define fi (Gfx.font_info 1))
(if (null? fi)
    (ML.metrics 7 12 12 1 2)
    (ML.metrics (get_item fi 0) (get_item fi 1) 12 1 2))
(ML.geom 0 LIST_Y SW LIST_H)
(set! menu_ready #t)

;; 启动顺序：先取时间（计时/心跳都依赖 now_ms），再枚举录音记录，最后整屏重绘
(set! now_ms (System.timestamp))
(set! blink_ms now_ms)
(set! last_diag 0)
(refresh_status)
(refresh_files)
(set! last_sel (ML.sel_of))
(set! last_scroll (ML.scroll_of))
(refresh_texts)
(set! last_diag 0)
(draw_all)
(set! timer_id (System.set_interval TICK_MS tick))
(EX.install_with redraw)
(EX.on_exit on_exit)
(display "[recorder] start rec_info=") (display (Rec.info))
(display " files=") (display (length files))
(display " row_h=") (display ROW_H)
(newline)
