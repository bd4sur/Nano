;; ============================================================================
;; 音乐盒（Music Box）—— 灵机应用程序（Scheme App）
;;
;; 与 C 版音乐盒（infer/ui_musicbox.c）功能对齐的 Scheme 重写：
;;   · 文件列表页：music/ 目录下的 WAV/MP3，点按/上下键选曲，滚动手势翻页；
;;   · 播放页：信息卡片（曲目序号/文件名/格式参数/播放状态）、可拖动的进度条、
;;     上一首 / 播放·暂停 / 下一首、音量 −/+（按住连发）、左上「列表」返回；
;;   · 列表循环、读完自动下一首、打不开的曲目自动跳过（全部失败时显示错误）；
;;   · 退出时关闭音频输出（EX 的全局确认框 → 收尾 → 销毁）。
;;
;; 分工（关键）：**解码与音频输出全在 C 侧**，本文件只管界面与交互。
;;   (native Music) 提供文件枚举 / 打开 / 喂数据 / 定位 / 进度 / 音量，
;;   详见 infer/am_native_Music.h 的设计说明。PCM 永不跨 FFI 边界。
;;
;; 与 C 版的差异（有意为之）：
;;   · 返回文件列表**不停止播放**（继续放，列表里用强调色圆点标出正在播放的曲目）；
;;     要停就进播放页按暂停，或直接退出 App（会关闭音频）。
;;   · 列表页支持拖动手势滚动（C 版靠菜单控件）。
;;   · 资源按需加载：WASM 下 music/ 属于 manifest 的 p4 层，首次进入时由
;;     (Music.list) 触发页面挂载，本 App 显示"正在加载音乐资源…"并每 500ms 重试。
;;
;; 节拍与重绘策略：
;;   33ms 一拍（≈30Hz）保证触屏点按采样足够密；画面**不必**每拍重绘——
;;   只有交互（dirty）或播放中每 3 拍（≈100ms）刷新一次进度条。慢节奏重绘让
;;   每帧留给解释器的时间片几乎都花在"用户能感知到的东西"上。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数；没有一元取负 → (- 0 x)；
;;   没有 let/list/map/append；define 只写在模块顶层或 lambda 体顶层（块内不 define）；
;;   cond 每子句只接受单个表达式；/ 与 mod 恒浮点；while 循环体不是独立词法区域；
;;   语句位置的分支写成 { (set! ...) } 净零块。'() 是模块级静态槽，故空表一律
;;   用 (cdr '(0)) 现造，绝不对 '() 直接 push。
;; ============================================================================

(native Gfx)
(native Touch)
(native Key)
(native Music)
(native System)
(native String)
(native Math)
(native App)

(import EX "app/lib/exit.scm")     ; 硬 A/ESC 与右上角「退出」按钮 → 全局退出确认框

;; ============================================================================
;; 常量：布局（320x240 基准，按实际画布宽高推导）
;; ============================================================================

(define SW (Gfx.width))
(define SH (Gfx.height))

(define TICK_MS 33)        ; 一拍 33ms（≈30Hz 输入采样）
(define DRAW_EVERY 3)      ; 播放中每 3 拍（≈100ms）刷新一次进度
(define POLL_EVERY 15)     ; 音乐资源未就绪时每 15 拍（≈500ms）重试一次列目录
(define POLL_MAX 400)      ; 最多重试次数（≈200s；下载 90MB 音频的宽容上限）
(define DIAG_EVERY 30)     ; 诊断行间隔（≈1s）

(define PAGE_LIST 0)
(define PAGE_PLAY 1)

(define TOP_H 26)                                        ; 顶栏高度
(define LIST_Y 30)                                       ; 列表首行 y
(define ROW_H 26)                                        ; 行高
(define LIST_ROWS (Math.floor (/ (- (- SH LIST_Y) 2) ROW_H)))  ; 可见行数（320x240 → 8）

(define CARD_X 8)
(define CARD_Y 28)
(define CARD_W (- SW 16))
(define CARD_H 76)

(define PROG_X 12)
(define PROG_Y (+ (+ CARD_Y CARD_H) 6))  ; 110
(define PROG_W (- SW 24))
(define PROG_H 40)
(define TRK_X 22)
(define TRK_W (- SW 44))
(define TRK_Y (+ PROG_Y 26))            ; 136
(define TRK_H 6)

(define PLAY_R 24)
(define PLAY_CX (Math.floor (/ SW 2)))
(define PLAY_CY 186)
(define SKIP_R 17)
(define SKIP_CX_D 58)
(define PREV_CX (- PLAY_CX SKIP_CX_D))
(define NEXT_CX (+ PLAY_CX SKIP_CX_D))
(define SKIP_CY PLAY_CY)

(define VOL_Y (- SH 26))                 ; 214
(define VOL_BTN_W 34)
(define VOL_BTN_H 24)
(define VOL_DN_X 12)
(define VOL_UP_X (- (- SW 12) VOL_BTN_W))
(define VOL_METER_X (+ (+ VOL_DN_X VOL_BTN_W) 10))
(define VOL_METER_W (- (- VOL_UP_X 10) VOL_METER_X))
(define VOL_METER_Y (+ VOL_Y 17))
(define VOL_STEP 8)
(define VOL_MAX 128)
(define VOL_HOLD_MS 350)                 ; 按住首次延迟
(define VOL_REPEAT_MS 120)               ; 之后连发间隔

(define BACK_X 8)
(define BACK_Y 4)
(define BACK_W 50)
(define BACK_H 18)

;; 配色（深色主题，与 C 版音乐盒同一套观感）
(define BG_R 14)  (define BG_G 18)  (define BG_B 26)
(define CARD_R 27) (define CARD_G 33) (define CARD_B 44)
(define EDGE_R 48) (define EDGE_G 58) (define EDGE_B 74)
(define SUR_R 38)  (define SUR_G 47)  (define SUR_B 61)
(define SURP_R 62) (define SURP_G 78) (define SURP_B 100)
(define ACC_R 56)  (define ACC_G 189) (define ACC_B 248)
(define ACCP_R 14) (define ACCP_G 140) (define ACCP_B 200)
(define TXT_R 237) (define TXT_G 242) (define TXT_B 249)
(define DIM_R 140) (define DIM_G 152) (define DIM_B 168)
(define TRK_R 52)  (define TRK_G 63)  (define TRK_B 79)
(define DAN_R 248) (define DAN_G 113) (define DAN_B 113)
(define HL_R 40)   (define HL_G 50)   (define HL_B 68)

;; 字体度量（12px 抗锯齿字模：半角 7px / 全角 12px）
(define W_HALF 7)
(define W_FULL 12)

;; ============================================================================
;; 状态
;; ============================================================================

(define page PAGE_LIST)        ; 当前页面
(define files (cdr '(0)))      ; music/ 目录下的文件名列表
(define nfiles 0)
(define sel 0)                 ; 列表高亮下标
(define offset 0)              ; 列表滚动首行下标
(define cur 0)                 ; 正在播放的曲目下标
(define fail #f)               ; 当前曲目打不开（显示错误态）
(define playing #f)            ; 引擎是否在播放
(define vol 64)                ; 音量 0~128
(define loading #f)            ; 音乐资源尚未就绪（正在等待挂载/下载）
(define load_polls 0)
(define last_poll 0)

(define ntick 0)
(define last_draw 0)
(define dirty #t)               ; 需要重绘
(define timer_id 0)

;; 触屏状态（与各 App 同一范式：按下记 arm，松手结算，拖出取消）
(define ARM_NONE 0)
(define ARM_ROW 1)
(define ARM_BACK 2)
(define ARM_SEEK 3)
(define ARM_PREV 4)
(define ARM_PLAY 5)
(define ARM_NEXT 6)
(define ARM_VOL_DN 7)
(define ARM_VOL_UP 8)

(define arm ARM_NONE)
(define press_x 0)
(define press_y 0)
(define row_hit 0)
(define moved #f)              ; 本次触摸序列是否已判为拖动
(define seek_prev -1.0)        ; 拖动中的进度预览（<0 = 未在拖动）
(define vol_next 0)            ; 下次音量连发的时刻（System.timestamp）
(define now_ms 0)

;; ============================================================================
;; 通用小工具
;; ============================================================================

(define clamp
  (lambda (v lo hi)
    (if (< v lo) lo (if (> v hi) hi v))))

(define min2 (lambda (a b) (if (< a b) a b)))
(define max2 (lambda (a b) (if (> a b) a b)))

(define char_w
  (lambda (cp) (if (>= cp 11904) W_FULL W_HALF)))   ;; 0x2E80 起算全角

;; 按像素宽估算文本宽度（12px 字模的步进表足够做对齐与裁剪）
(define text_w
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (while (< i n)
      {
        (set! w (+ w (char_w (String.charCodeAt i s))))
        (set! i (+ i 1))
      })
    w))

;; 超宽文本截断为"前缀…"（保留开头比保留扩展名更利于辨认曲名）
(define clip_text
  (lambda (s maxw)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (define cw 0)
    (define stop 0)
    (define cut -1)
    (while (and (< i n) (== stop 0))
      {
        (set! cw (char_w (String.charCodeAt i s)))
        (if (> (+ w cw) (- maxw W_HALF))
            { (set! cut i) (set! stop 1) }
            { (set! w (+ w cw)) (set! i (+ i 1)) })
      })
    (if (< cut 0)
        s
        (if (== cut 0) "" (String.concat (String.slice s 0 cut) "…")))))

;; 左对齐绘制（自动按 maxw 截断）
(define draw_fit
  (lambda (s x y maxw r g b)
    (Gfx.draw_text (clip_text s maxw) x y r g b)))

;; 右对齐绘制
(define draw_right
  (lambda (s right y r g b)
    (Gfx.draw_text s (- right (text_w s)) y r g b)))

;; 毫秒 → "m:ss"
(define fmt_time
  (lambda (ms)
    (define sec (Math.floor (/ ms 1000)))
    (define mm (Math.floor (/ sec 60)))
    (define ss (- sec (* mm 60)))
    (define s0 (String.atom_to_string ss))
    (String.concat (String.concat (String.atom_to_string mm) ":")
                   (if (< ss 10) (String.concat "0" s0) s0))))

;; 采样率 → "44.1 kHz"（/ 恒浮点，%g 打印自然去掉多余的 .0）
(define fmt_rate
  (lambda (hz)
    (String.concat (String.atom_to_string (/ hz 1000)) " kHz")))

(define in_rect
  (lambda (x y rx ry rw rh)
    (and (>= x rx) (< x (+ rx rw)) (and (>= y ry) (< y (+ ry rh))))))

(define in_circle
  (lambda (x y cx cy r)
    (<= (+ (* (- x cx) (- x cx)) (* (- y cy) (- y cy))) (* r r))))

;; 进度条像素 x → 比例（0.0~1.0）
(define ratio_from_x
  (lambda (x)
    (clamp (/ (- x TRK_X) TRK_W) 0.0 1.0)))

;; ============================================================================
;; 播放引擎封装
;; ============================================================================

;; 刷新文件列表（首次/资源就绪后调用）；空目录时置"加载中"并请求页面挂载
(define refresh_list
  (lambda ()
    (define l #null)
    (set! l (Music.list))
    (if (list? l)
        {
          (set! files l)
          (set! nfiles (length files))
          (set! loading (if (> nfiles 0) #f #t))
          (if (>= sel nfiles) (set! sel 0) #f)
          (set! dirty #t)
        }
        #f)))

;; 曲目完整路径（Music 的路径规则与 Fs 一致：相对业务资源根）
(define track_path
  (lambda (idx)
    (String.concat "music/" (get_item files idx))))

;; 打开第 idx 首并从当前位置开始播放；成功返回 #t
(define open_track
  (lambda (idx)
    (define ok #f)
    (define name "")
    (if (or (< idx 0) (>= idx nfiles))
        #f
        {
          (set! cur idx)
          (set! name (get_item files idx))
          (set! ok (Music.open (track_path idx)))
          (set! fail (if ok #f #t))
          (set! playing (Music.is_playing))
          (set! seek_prev -1.0)
          (set! page PAGE_PLAY)
          (set! dirty #t)
          (display "[musicbox] open idx=") (display idx)
          (display " ok=") (display ok)
          (display " ")(display name) (newline)
          ok
        })))

;; 上/下一首（列表循环；打不开的曲目自动跳过，全部失败则置错误态）
(define switch_track
  (lambda (step)
    (define tries 0)
    (define ok #f)
    (define c 0)
    (if (== nfiles 0)
        { (set! fail #t) (set! dirty #t) }
        {
          (set! c cur)
          (while (and (< tries nfiles) (not ok))
            {
              (set! c (Math.floor (mod (+ (+ c step) nfiles) nfiles)))
              (set! ok (Music.open (track_path c)))
              (set! tries (+ tries 1))
            })
          (if ok
              { (set! cur c)
                (set! fail #f)
                (set! playing #t)
                (set! seek_prev -1.0)
                (set! page PAGE_PLAY) }
              { (set! fail #t) (set! playing #f) })
          (set! dirty #t)
          (display "[musicbox] switch cur=") (display cur)
          (display " ok=") (display ok) (newline)
        })))

;; 播放/暂停切换
(define toggle_play
  (lambda ()
    (if fail
        #f
        {
          (set! playing (Music.toggle))
          (set! dirty #t)
          (display "[musicbox] toggle playing=") (display playing) (newline)
        })))

(define change_vol
  (lambda (dir)
    (set! vol (Math.floor (clamp (+ vol (* dir VOL_STEP)) 0 VOL_MAX)))
    (Music.set_volume vol)
    (set! dirty #t)))

;; 切后台 / 锁屏恢复后重建音频会话。
;;
;; 为什么需要：实例被冻结时宿主不再泵本 App（定时器全停），但**音频设备仍被本实例
;; 占着**；而冻结期间别的 App（比如 C 版音乐盒）可能已经 audio_out_close/init 过一轮，
;; 于是恢复后的 s_audio_on 与真实 HAL 状态不一致——继续喂 PCM 会以对方的采样率播放。
;; 因此恢复的第一件事是"关掉并就地重建"：按上次的播放位置 seek 回去（MP3 为近似定位），
;; 并保留原来的播放/暂停状态。代价是位置有几十毫秒误差，换来的是会话状态绝对干净。
;;
;; 轮询点：本函数消费 (App.resumed)（取走即清零，与 EX 的约定一致——EX 只把它当
;; "要不要重绘"的提示，我们这里自己置 dirty 顶上）。
(define reopen_after_resume
  (lambda ()
    (define was_playing playing)
    (define was_fail fail)
    (define idx cur)
    (define ratio 0.0)
    (set! ratio (clamp (Music.progress) 0.0 0.99))
    (Music.close)
    (set! playing #f)
    (if (and (not was_fail) (> nfiles 0) (< idx nfiles))
        {
          (if (Music.open (track_path idx))
              {
                (Music.set_volume vol)
                (Music.seek ratio)
                (set! fail #f)
                (if was_playing #f (Music.pause))
                (set! playing (Music.is_playing))
              }
              { (set! fail #t) (set! playing #f) })
        }
        #f)
    (set! dirty #t)
    (display "[musicbox] resume idx=") (display idx)
    (display " ratio=") (display (Math.floor (* 1000 ratio)))
    (display " playing=") (display playing) (newline)))

;; 每拍喂数据；EOF 自动下一首
(define feed_audio
  (lambda ()
    (define rc 0)
    (set! rc (Music.pump))
    (if (== rc 1) (switch_track 1) #f)
    (if (== rc -1) { (set! fail #t) (set! playing #f) (set! dirty #t) } #f)
    (set! playing (Music.is_playing))))

;; ============================================================================
;; 绘制：公共部件
;; ============================================================================

(define draw_triangle_right
  (lambda (cx cy w hh r g b)
    (define i 0)
    (define x 0)
    (define h 0)
    (define x0 (- cx (Math.floor (/ w 2))))
    (while (< i w)
      {
        (set! x (+ x0 i))
        (set! h (Math.floor (/ (* hh (- w i)) w)))
        (Gfx.draw_line x (- cy h) x (+ cy h) r g b)
        (set! i (+ i 1))
      })))

(define draw_triangle_left
  (lambda (cx cy w hh r g b)
    (define i 0)
    (define x 0)
    (define h 0)
    (define x1 (+ cx (Math.floor (/ w 2))))
    (while (< i w)
      {
        (set! x (- x1 i))
        (set! h (Math.floor (/ (* hh (- w i)) w)))
        (Gfx.draw_line x (- cy h) x (+ cy h) r g b)
        (set! i (+ i 1))
      })))

;; 顶栏（列表页：左侧标题；播放页：左侧「列表」返回 + 居中标题）
(define draw_topbar
  (lambda ()
    (Gfx.fill_rect 0 0 SW TOP_H BG_R BG_G BG_B)
    (Gfx.fill_rect 0 TOP_H SW 1 EDGE_R EDGE_G EDGE_B)
    (if (== page PAGE_LIST)
        {
          (Gfx.fill_rect 12 7 3 12 ACC_R ACC_G ACC_B)
          (Gfx.draw_text "音乐盒" 20 6 TXT_R TXT_G TXT_B)
          (draw_right (String.concat (String.atom_to_string nfiles) " 首")
                      (- SW 48) 7 DIM_R DIM_G DIM_B)
        }
        {
          (Gfx.rounded_rect BACK_X BACK_Y BACK_W BACK_H 6 6 6 6 SUR_R SUR_G SUR_B 1)
          (Gfx.draw_text "返回" (+ BACK_X 13) (+ BACK_Y 3) TXT_R TXT_G TXT_B)
          (Gfx.draw_text_centered 1 "音乐盒" (/ SW 2) 13 TXT_R TXT_G TXT_B 1)
        })
    (EX.draw_button)))    ; 右上角「退出」（点按由宿主在触摸边沿判定）

;; ============================================================================
;; 绘制：列表页
;; ============================================================================

(define draw_list
  (lambda ()
    (define i 0)
    (define n 0)
    (define y 0)
    (define sel_row 0)
    (define nm "")
    (Gfx.fill_rect 0 0 SW SH BG_R BG_G BG_B)
    (draw_topbar)
    (if (== nfiles 0)
        {
          (if loading
              {
                (Gfx.draw_text_centered 1 "正在加载音乐资源…" (/ SW 2) 108 DIM_R DIM_G DIM_B 1)
                (Gfx.draw_text_centered 1 "首次进入需下载，请稍候" (/ SW 2) 126 DIM_R DIM_G DIM_B 1)
              }
              {
                (Gfx.draw_text_centered 1 "未找到音乐文件" (/ SW 2) 104 TXT_R TXT_G TXT_B 1)
                (Gfx.draw_text_centered 1 "请把 .wav / .mp3 放进 music/ 目录" (/ SW 2) 126
                                        DIM_R DIM_G DIM_B 1)
              })
        }
        {
          (set! sel_row (min2 LIST_ROWS (- nfiles offset)))
          (set! i 0)
          (while (< i sel_row)
            {
              (set! n (+ offset i))
              (set! y (+ LIST_Y (* i ROW_H)))
              (if (== n sel)
                  (Gfx.fill_rect 4 y (- SW 8) (- ROW_H 2) HL_R HL_G HL_B)
                  #f)
              (Gfx.draw_text (String.atom_to_string (+ n 1)) 12 (+ y 7) DIM_R DIM_G DIM_B)
              (set! nm (get_item files n))
              (draw_fit nm 34 (+ y 7) (- SW 64) TXT_R TXT_G TXT_B)
              ;; 正在播放的曲目：右侧强调色圆点
              (if (and (== n cur) (not fail))
                  (Gfx.fill_circle (- SW 14) (+ y 12) 3 ACC_R ACC_G ACC_B)
                  #f)
              (set! i (+ i 1))
            })
          ;; 滚动指示（内容多于可见行时）
          (if (> nfiles LIST_ROWS)
              {
                (Gfx.fill_rect (- SW 3) LIST_Y 3 (- (- SH LIST_Y) 4) TRK_R TRK_G TRK_B)
                (Gfx.fill_rect (- SW 3)
                               (+ LIST_Y (Math.floor (/ (* (- (- SH LIST_Y) 4) offset) nfiles)))
                               3
                               (max2 12 (Math.floor (/ (* (- (- SH LIST_Y) 4) LIST_ROWS) nfiles)))
                               DIM_R DIM_G DIM_B)
              }
              #f)
        })))

;; ============================================================================
;; 绘制：播放页
;; ============================================================================

;; 第二行信息："MP3 · 44.1 kHz · 320 kbps · 立体声"
(define format_line
  (lambda ()
    (define inf #null)
    (define fmt #null)
    (define rate 0)
    (define ch 0)
    (define bits 0)
    (define br 0)
    (define s "")
    (set! inf (Music.info))
    (set! fmt (Music.format))
    (if (list? inf)
        {
          (set! rate (get_item inf 0))
          (set! ch (get_item inf 1))
          (set! bits (get_item inf 2))
          (set! br (get_item inf 3))
          (set! s (String.concat (String.atom_to_string fmt) " · "))
          (set! s (String.concat s (String.concat (fmt_rate rate) " · ")))
          (if (== bits 16)
              (set! s (String.concat s "16 bit · "))
              #f)
          (if (> br 0)
              (set! s (String.concat s (String.concat (String.atom_to_string br) " kbps · ")))
              #f)
          (set! s (String.concat s (if (== ch 1) "单声道" "立体声")))
          s
        }
        "—")))

(define draw_card
  (lambda ()
    (define st "")
    (define str "")
    (define nm "")
    (Gfx.rounded_rect CARD_X CARD_Y CARD_W CARD_H 10 10 10 10 CARD_R CARD_G CARD_B 1)
    (Gfx.draw_rect CARD_X CARD_Y CARD_W CARD_H EDGE_R EDGE_G EDGE_B)
    ;; 序号 cur / 总数
    (set! str (String.atom_to_string (+ cur 1)))
    (Gfx.draw_text str (+ CARD_X 12) (+ CARD_Y 7) ACC_R ACC_G ACC_B)
    (Gfx.draw_text (String.concat " / " (String.atom_to_string nfiles))
                   (+ (+ CARD_X 12) (text_w str)) (+ CARD_Y 7) DIM_R DIM_G DIM_B)
    ;; 状态（右对齐）
    (if fail
        (set! st "无法播放")
        (if playing (set! st "播放中") (set! st "已暂停")))
    (draw_right st (- (+ CARD_X CARD_W) 12) (+ CARD_Y 7)
               (if fail DAN_R DIM_R) (if fail DAN_G DIM_G) (if fail DAN_B DIM_B))
    ;; 文件名（居中 + 超宽截断）
    (if fail
        (set! nm "文件格式不支持或已损坏")
        (if (< cur nfiles) (set! nm (get_item files cur)) (set! nm "—")))
    (Gfx.draw_text_centered 1 (clip_text nm (- CARD_W 24)) (+ CARD_X (Math.floor (/ CARD_W 2)))
                            (+ CARD_Y 34) TXT_R TXT_G TXT_B 1)
    ;; 格式参数
    (Gfx.draw_text_centered 1 (clip_text (format_line) (- CARD_W 24))
                            (+ CARD_X (Math.floor (/ CARD_W 2))) (+ CARD_Y 56)
                            DIM_R DIM_G DIM_B 1)))

(define draw_progress
  (lambda ()
    (define ratio 0.0)
    (define dur 0)
    (define el 0)
    (define fw 0)
    (define tx 0)
    (set! ratio (if (>= seek_prev 0.0) seek_prev (Music.progress)))
    (set! ratio (clamp ratio 0.0 1.0))
    (set! dur (Music.duration_ms))
    (set! el (if (>= seek_prev 0.0) (* dur ratio) (Music.elapsed_ms)))
    (if (> el dur) (set! el dur) #f)
    (Gfx.rounded_rect PROG_X PROG_Y PROG_W PROG_H 8 8 8 8 SUR_R SUR_G SUR_B 1)
    (Gfx.draw_text (fmt_time el) (+ PROG_X 8) (+ PROG_Y 4) TXT_R TXT_G TXT_B)
    (draw_right (fmt_time dur) (- (+ PROG_X PROG_W) 8) (+ PROG_Y 4) DIM_R DIM_G DIM_B)
    ;; 轨道 + 已播放部分 + 拇指
    (Gfx.fill_rect TRK_X TRK_Y TRK_W TRK_H TRK_R TRK_G TRK_B)
    (set! fw (Math.floor (* TRK_W ratio)))
    (if (> fw 0) (Gfx.fill_rect TRK_X TRK_Y fw TRK_H ACC_R ACC_G ACC_B) #f)
    (set! tx (+ TRK_X fw))
    (Gfx.fill_circle tx (+ TRK_Y (Math.floor (/ TRK_H 2))) 6 ACC_R ACC_G ACC_B)
    (Gfx.fill_circle tx (+ TRK_Y (Math.floor (/ TRK_H 2))) 4 250 250 252)))

(define draw_transport
  (lambda ()
    ;; 上一首 / 下一首（圆底 + 矢量图标）
    (Gfx.fill_circle PREV_CX SKIP_CY SKIP_R SUR_R SUR_G SUR_B)
    (draw_triangle_left (+ PREV_CX 1) SKIP_CY 11 7 TXT_R TXT_G TXT_B)
    (Gfx.fill_rect (- PREV_CX 8) (- SKIP_CY 7) 2 14 TXT_R TXT_G TXT_B)
    (Gfx.fill_circle NEXT_CX SKIP_CY SKIP_R SUR_R SUR_G SUR_B)
    (draw_triangle_right (+ NEXT_CX 1) SKIP_CY 11 7 TXT_R TXT_G TXT_B)
    (Gfx.fill_rect (+ NEXT_CX 6) (- SKIP_CY 7) 2 14 TXT_R TXT_G TXT_B)
    ;; 播放 / 暂停（主按钮）
    (Gfx.fill_circle PLAY_CX PLAY_CY PLAY_R
                     (if fail SUR_R ACC_R)
                     (if fail SUR_G ACC_G)
                     (if fail SUR_B ACC_B))
    (if playing
        {
          (Gfx.fill_rect (- PLAY_CX 8) (- PLAY_CY 9) 6 18 255 255 255)
          (Gfx.fill_rect (+ PLAY_CX 2) (- PLAY_CY 9) 6 18 255 255 255)
        }
        (draw_triangle_right (+ PLAY_CX 2) PLAY_CY 16 9 255 255 255))))

(define draw_volume
  (lambda ()
    (define fw 0)
    (define pct 0)
    ;; 减 / 加按钮
    (Gfx.rounded_rect VOL_DN_X VOL_Y VOL_BTN_W VOL_BTN_H 6 6 6 6 SUR_R SUR_G SUR_B 1)
    (Gfx.rounded_rect VOL_UP_X VOL_Y VOL_BTN_W VOL_BTN_H 6 6 6 6 SUR_R SUR_G SUR_B 1)
    (Gfx.fill_rect (+ VOL_DN_X 10) (+ VOL_Y 11) 14 2 TXT_R TXT_G TXT_B)
    (Gfx.fill_rect (+ VOL_UP_X 10) (+ VOL_Y 11) 14 2 TXT_R TXT_G TXT_B)
    (Gfx.fill_rect (+ VOL_UP_X 16) (+ VOL_Y 5) 2 14 TXT_R TXT_G TXT_B)
    ;; 音量条 + 百分比
    (Gfx.fill_rect VOL_METER_X VOL_METER_Y VOL_METER_W 4 TRK_R TRK_G TRK_B)
    (set! fw (Math.floor (/ (* VOL_METER_W vol) VOL_MAX)))
    (if (> fw 0) (Gfx.fill_rect VOL_METER_X VOL_METER_Y fw 4 ACC_R ACC_G ACC_B) #f)
    (set! pct (Math.floor (/ (* vol 100) VOL_MAX)))
    (Gfx.draw_text_centered 1 (String.concat (String.atom_to_string pct) "%")
                            (/ SW 2) (+ VOL_Y 4) DIM_R DIM_G DIM_B 1)))

(define draw_play
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH BG_R BG_G BG_B)
    (draw_topbar)
    (draw_card)
    (draw_progress)
    (draw_transport)
    (draw_volume)))

(define draw
  (lambda ()
    (if (== page PAGE_LIST) (draw_list) (draw_play))
    (Gfx.refresh)
    (set! dirty #f)
    (set! last_draw ntick)))

;; ============================================================================
;; 输入：触屏
;; ============================================================================

;; 按下：记下按到的控件（列表页按行、播放页按圆/条/按钮）
(define on_down
  (lambda (x y)
    (set! arm ARM_NONE)
    (set! moved #f)
    (set! press_x x)
    (set! press_y y)
    (if (== page PAGE_LIST)
        (if (in_rect x y 4 LIST_Y (- SW 8) (* LIST_ROWS ROW_H))
            {
              (set! arm ARM_ROW)
              (set! row_hit (+ offset (Math.floor (/ (- y LIST_Y) ROW_H))))
              (if (< row_hit nfiles)
                  { (set! sel row_hit) (set! dirty #t) }
                  #f)
            }
            #f)
        {
          (if (in_rect x y BACK_X BACK_Y BACK_W BACK_H)
              (set! arm ARM_BACK)
              #f)
          (if (in_rect x y (- PROG_X 4) (- PROG_Y 4) (+ PROG_W 8) (+ PROG_H 8))
              {
                (set! arm ARM_SEEK)
                (set! seek_prev (ratio_from_x x))
                (set! dirty #t)
              }
              #f)
          (if (in_circle x y PREV_CX SKIP_CY (+ SKIP_R 6)) (set! arm ARM_PREV) #f)
          (if (in_circle x y PLAY_CX PLAY_CY (+ PLAY_R 5)) (set! arm ARM_PLAY) #f)
          (if (in_circle x y NEXT_CX SKIP_CY (+ SKIP_R 6)) (set! arm ARM_NEXT) #f)
          (if (in_rect x y (- VOL_DN_X 2) (- VOL_Y 2) (+ VOL_BTN_W 4) (+ VOL_BTN_H 4))
              {
                (set! arm ARM_VOL_DN)
                (change_vol -1)
                (set! vol_next (+ now_ms VOL_HOLD_MS))
              }
              #f)
          (if (in_rect x y (- VOL_UP_X 2) (- VOL_Y 2) (+ VOL_BTN_W 4) (+ VOL_BTN_H 4))
              {
                (set! arm ARM_VOL_UP)
                (change_vol 1)
                (set! vol_next (+ now_ms VOL_HOLD_MS))
              }
              #f)
        })))

;; 松手：只有"按下与松开落在同一控件"才结算（拖出取消）
(define on_up
  (lambda (x y)
    (if (== arm ARM_ROW)
        (if (and (not moved) (in_rect x y 4 LIST_Y (- SW 8) (* LIST_ROWS ROW_H)))
            (if (< row_hit nfiles) (open_track row_hit) #f)
            #f)
        #f)
    (if (== arm ARM_BACK)
        (if (in_rect x y BACK_X BACK_Y BACK_W BACK_H)
            { (set! page PAGE_LIST) (set! sel cur) (set! dirty #t) }
            #f)
        #f)
    (if (== arm ARM_SEEK)
        {
          (if (in_rect x y (- PROG_X 4) (- PROG_Y 4) (+ PROG_W 8) (+ PROG_H 8))
              {
                (if (Music.seek (ratio_from_x x))
                    #f
                    { (set! fail #t) (set! playing #f) })
                (display "[musicbox] seek ")(display (Math.floor (* 100 (ratio_from_x x))))
                (newline)
              }
              #f)
          (set! seek_prev -1.0)
          (set! dirty #t)
        }
        #f)
    (if (== arm ARM_PLAY)
        (if (in_circle x y PLAY_CX PLAY_CY (+ PLAY_R 5)) (toggle_play) #f)
        #f)
    (if (== arm ARM_PREV)
        (if (in_circle x y PREV_CX SKIP_CY (+ SKIP_R 6)) (switch_track -1) #f)
        #f)
    (if (== arm ARM_NEXT)
        (if (in_circle x y NEXT_CX SKIP_CY (+ SKIP_R 6)) (switch_track 1) #f)
        #f)
    (set! arm ARM_NONE)
    (set! moved #f)))

;; 每拍：拖动手势（列表滚动 / 进度条预览 / 音量连发）
(define handle_touch
  (lambda ()
    (define dy 0)
    (define r 0.0)
    (define hit 0)
    (if (Touch.take_press) (on_down (Touch.x) (Touch.y)) #f)
    (if (Touch.take_release) (on_up (Touch.x) (Touch.y)) #f)
    (if (Touch.is_pressed)
        {
          ;; 列表：竖向拖动翻行（每 20px 一行，拖过即重定基准）
          (if (and (== arm ARM_ROW) (> nfiles LIST_ROWS) (not moved))
              {
                (set! dy (- (Touch.y) press_y))
                (if (> dy 20)
                    {
                      (if (> offset 0) (set! offset (- offset 1)) #f)
                      (set! press_y (Touch.y))
                      (set! moved #t)
                      (set! dirty #t)
                    }
                    #f)
                (if (< dy -20)
                    {
                      (if (< (+ offset 1) (- nfiles LIST_ROWS))
                          (set! offset (+ offset 1))
                          #f)
                      (set! press_y (Touch.y))
                      (set! moved #t)
                      (set! dirty #t)
                    }
                    #f)
              }
              #f)
          ;; 进度条拖动：只更新预览，松手才真正 seek
          (if (== arm ARM_SEEK)
              {
                (set! r (ratio_from_x (Touch.x)))
                (if (not (== r seek_prev)) { (set! seek_prev r) (set! dirty #t) } #f)
              }
              #f)
          ;; 音量键按住连发（滑出按钮即停）
          (if (or (== arm ARM_VOL_DN) (== arm ARM_VOL_UP))
              {
                (set! hit 0)
                (if (== arm ARM_VOL_DN)
                    (set! hit (if (in_rect (Touch.x) (Touch.y) (- VOL_DN_X 2) (- VOL_Y 2)
                                            (+ VOL_BTN_W 4) (+ VOL_BTN_H 4)) 1 0))
                    #f)
                (if (== arm ARM_VOL_UP)
                    (set! hit (if (in_rect (Touch.x) (Touch.y) (- VOL_UP_X 2) (- VOL_Y 2)
                                            (+ VOL_BTN_W 4) (+ VOL_BTN_H 4)) 1 0))
                    #f)
                (if (== hit 0)
                    (set! arm ARM_NONE)
                    (if (>= now_ms vol_next)
                        {
                          (change_vol (if (== arm ARM_VOL_UP) 1 -1))
                          (set! vol_next (+ now_ms VOL_REPEAT_MS))
                        }
                        #f))
              }
              #f)
        }
        #f)))

;; ============================================================================
;; 输入：键盘（实体键盘兜底；触屏优先）
;; ============================================================================

(define handle_key
  (lambda (k)
    (if (== page PAGE_LIST)
        {
          (if (or (== k 2) (== k 100))
              {
                (if (< sel (- nfiles 1)) (set! sel (+ sel 1)) #f)
                (if (>= sel (+ offset LIST_ROWS)) (set! offset (- sel (- LIST_ROWS 1))) #f)
                (set! dirty #t)
              }
              #f)
          (if (or (== k 1) (== k 104))
              {
                (if (> sel 0) (set! sel (- sel 1)) #f)
                (if (< sel offset) (set! offset sel) #f)
                (set! dirty #t)
              }
              #f)
          (if (or (== k 13) (== k 32))
              (if (> nfiles 0) (open_track sel) #f)
              #f)
        }
        {
          (if (or (== k 13) (== k 32)) (toggle_play) #f)
          (if (== k 3) (change_vol -1) #f)
          (if (== k 4) (change_vol 1) #f)
          (if (or (== k 52) (== k 1)) (switch_track -1) #f)
          (if (or (== k 54) (== k 2)) (switch_track 1) #f)
        })))

(define handle_keys
  (lambda ()
    (define k 0)
    (set! k (Key.take_press))
    (while (> k 0)
      {
        (handle_key k)
        (set! k (Key.take_press))
      })))

;; ============================================================================
;; 诊断行（端到端回归与排障用；≈1s 一条）
;; ============================================================================

(define diag
  (lambda ()
    (if (== (mod ntick DIAG_EVERY) 0.0)
        {
          (display "[musicbox] page=") (display page)
          (display " cur=") (display cur)
          (display "/") (display nfiles)
          (display " playing=") (display playing)
          (display " fail=") (display fail)
          (display " prog=") (display (Math.floor (* 1000 (clamp (Music.progress) 0.0 1.0))))
          (display " el=") (display (Math.floor (Music.elapsed_ms)))
          (display " dur=") (display (Math.floor (Music.duration_ms)))
          (display " vol=") (display vol)
          (newline)
        }
        #f)))

;; ============================================================================
;; 主循环
;; ============================================================================

(define tick
  (lambda ()
    {
      (if (App.resumed) (reopen_after_resume) #f)   ; 恢复：重建音频会话（须早于 EX.check）
      (EX.check)              ; 硬 A/ESC 与右上角「退出」按钮（可能阻塞，返回后继续）
      (set! ntick (+ ntick 1))
      (set! now_ms (System.timestamp))
      (handle_touch)
      (handle_keys)
      (feed_audio)
      ;; 资源未就绪：按 500ms 节奏重试列目录（首次会触发页面挂载 p4 音频层）
      (if (and (== page PAGE_LIST) (== nfiles 0) (< load_polls POLL_MAX))
          (if (>= (- ntick last_poll) POLL_EVERY)
              {
                (set! last_poll ntick)
                (set! load_polls (+ load_polls 1))
                (refresh_list)
              }
              #f)
          #f)
      ;; 重绘：交互后立即；播放中每 3 拍刷一次进度条
      (if (or dirty (and playing (>= (- ntick last_draw) DRAW_EVERY)))
          (draw)
          #f)
      (diag)
    }))

;; ============================================================================
;; 启动 / 退出
;; ============================================================================

(define on_exit
  (lambda ()
    {
      (display "[musicbox] bye") (newline)
      (Music.close)                 ; 停音频、关文件、释放缓冲（幂等）
      (System.clear_interval timer_id)
    }))

(Music.close)                       ; 清掉上一次（可能被后台冻结的实例）留下的状态
(set! vol (Math.floor (Music.volume)))
(set! now_ms (System.timestamp))
(refresh_list)
(draw)
(set! timer_id (System.set_interval TICK_MS tick))
(EX.install_with draw)
(EX.on_exit on_exit)
(display "[musicbox] start files=") (display nfiles) (newline)
