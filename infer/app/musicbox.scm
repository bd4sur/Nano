;@id=musicbox
;@title=音乐盒
;@version=2026-11-17
;; ============================================================================
;; 音乐盒（Music Box）—— 灵机应用程序（Scheme App）
;;
;; 与 C 版音乐盒（infer/ui_musicbox.c）功能对齐的 Scheme 重写：
;;   · 文件列表页：music/ 目录下的 WAV/MP3，点按/上下键选曲，滚动手势翻页；
;;   · 播放页：信息卡片（曲目序号/文件名/格式参数/播放状态）、内嵌可拖动进度条、
;;     上一首 / 播放·暂停 / 下一首、音量 −/+（按住连发）、左上「返回」；
;;     版面与 C 版 `ui_musicbox.c` 逐像素对齐（见下「与 C 版的差异」）；
;;   · 列表循环、读完自动下一首、打不开的曲目自动跳过（全部失败时显示错误）；
;;   · **可被拉起**：文件管理器里点开 .wav/.mp3 时以
;;     (App.launch_with "musicbox" 绝对路径) 把控制权交给本 App，启动参数 = 待播文件的
;;     宿主绝对路径 → 列表换成该文件所在目录、选中它并**立即开播**（见下方「启动参数」）；
;;     从应用列表进入（没有启动参数）时行为与以前完全一致；
;;   · 退出时关闭音频输出（EX 的全局确认框 → 收尾 → 销毁）。
;;
;; 分工（关键）：**解码与音频输出全在 C 侧**，本文件只管界面与交互。
;;   (native Music) 提供文件枚举 / 打开 / 喂数据 / 定位 / 进度 / 音量，
;;   详见 infer/am_native_Music.h 的设计说明。PCM 永不跨 FFI 边界。
;;
;; 与 C 版的差异（有意为之）：
;;   · **播放页的版面与 C 版逐像素一致**（`ui_musicbox.c` 的 musicbox_ui_layout /
;;     musicbox_draw_*）：背景竖向渐变（8px 条带）、卡片 (12,36,296,104) r=14、内嵌
;;     进度面板 (22,98,276,36)、16px 文件名、"MP3  ·  44.1 kHz  ·  320 kbps  ·  立体声"
;;     这样的双空格分隔、播放键圆心 (160,168) r=24、上一首/下一首 ±72 r=17、
;;     底部「音量 + 百分比 + 音量条 + −/+」一排。**唯一的不同是顶栏**：本 App 必须
;;     保留左上「返回」与右上「退出」两个按钮，故 C 版顶栏的标题组（强调色竖条 +
;;     12px 标题）整体右移到「返回」胶囊右侧，其余几何/字号不变。
;;   · 返回文件列表**不停止播放**（继续放，列表里用强调色圆点标出正在播放的曲目）；
;;     要停就进播放页按暂停，或直接退出 App（会关闭音频）。
;;   · 列表页支持拖动手势滚动（C 版靠菜单控件）。
;;   · 资源后台加载：WASM 下 music/ 和别的资源一样由页面在启动后**后台**拉进 MEMFS
;;     （manifest 的 p2 层，无按需层）。首次进入时可能还没拉完，本 App 显示"正在加载
;;     音乐资源…"并每 500ms 重试 (Music.list)。
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
(native Fs)
(native System)
(native String)
(native Math)
(native App)

(import EX "app/lib/exit.scm")     ; 硬 A/ESC 与右上角「退出」按钮 → 全局退出确认框
(import FSX "app/lib/fsx.scm")     ; 路径拆分（leaf/parent/join）与扩展名过滤（被拉起时用）

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

(define TOP_H 26)                                        ; 列表页顶栏高度
(define LIST_Y 30)                                       ; 列表首行 y
(define ROW_H 26)                                        ; 行高
(define LIST_ROWS (Math.floor (/ (- (- SH LIST_Y) 2) ROW_H)))  ; 可见行数（320x240 → 8）

;; ---- 播放页布局：与 C 版 ui_musicbox.c 的 musicbox_ui_layout 逐项一致 ----------
;; C 版只由画布宽高推导（含小屏兜底），这里照抄同一套推导；320x240 下即
;;   卡片 (12,36,296,104) r=14；内嵌进度面板 (22,98,276,36) r=10；时间行 y=100；
;;   轨道 (34,120,252,5) 圆角 2、拇指半径 6；
;;   播放键圆心 (160,168) r=24，上一首/下一首圆心 (88/232,168) r=17；
;;   音量排 y=204 高 30 圆角 10（− 在 x=14、+ 在 x=272），音量条 (58,224,204,4)。
(define CARD_X 12)
(define CARD_W (- SW 24))
(if (< CARD_W 96) { (set! CARD_X 4) (set! CARD_W (- SW 8)) } #f)
(define CARD_Y 36)
(define CARD_H 104)
(if (> (+ CARD_Y CARD_H) (- SH 100)) (set! CARD_H (- (- SH 100) CARD_Y)) #f)
(if (< CARD_H 70) (set! CARD_H 70) #f)
(define CARD_RAD 14)
(define COUNTER_X (+ CARD_X 14))
(define COUNTER_Y (+ CARD_Y 8))
(define FILENAME_CY (+ CARD_Y 35))
(define FORMAT_CY (+ CARD_Y 53))

(define PROG_X (+ CARD_X 10))
(define PROG_W (- CARD_W 20))
(define PROG_H 36)
(define PROG_Y (- (- (+ CARD_Y CARD_H) PROG_H) 6))
(define PROG_RAD 10)
(define TIME_Y (+ PROG_Y 2))

(define TRK_X (+ PROG_X 12))
(define TRK_W (- PROG_W 24))
(if (< TRK_W 20) (set! TRK_W 20) #f)
(define TRK_Y (+ PROG_Y 22))
(define TRK_H 5)
(define THUMB_R 6)

(define PLAY_R 24)
(define PLAY_CX (Math.floor (/ SW 2)))
(define PLAY_CY (+ (+ CARD_Y CARD_H) 28))
(define SKIP_R 17)
(define SKIP_DX 72)
(define PREV_CX (- PLAY_CX SKIP_DX))
(define NEXT_CX (+ PLAY_CX SKIP_DX))

(define VOL_BTN_W 34)
(define VOL_BTN_H 30)
(define VOL_BTN_RAD 10)
(define VOL_Y (- SH 36))
(define VOL_DN_X 14)
(define VOL_UP_X (- (- SW 14) VOL_BTN_W))
(define VOL_TEXT_Y (- VOL_Y 2))
(define VOL_METER_X (+ (+ VOL_DN_X VOL_BTN_W) 10))
(define VOL_METER_W (- (- VOL_UP_X 10) VOL_METER_X))
(if (< VOL_METER_W 40) (set! VOL_METER_W 40) #f)
(define VOL_METER_Y (+ VOL_Y 20))
(define VOL_METER_H 4)
(define VOL_STEP 8)
(define VOL_MAX 128)
(define VOL_HOLD_MS 350)                 ; 按住首次延迟
(define VOL_REPEAT_MS 120)               ; 之后连发间隔

(define BACK_X 8)
(define BACK_Y 4)
(define BACK_W 50)
(define BACK_H 18)

;; 播放页顶栏的标题组：几何取 C 版顶栏（强调色竖条 3x12 @ y=13、标题 12px @ y=10），
;; 但整体右移到「返回」胶囊右侧——C 版顶栏没有返回按钮，本 App 必须保留它。
(define TITLE_BAR_X (+ (+ BACK_X BACK_W) 8))
(define TITLE_X (+ TITLE_BAR_X 8))

;; 配色：逐项取自 C 版 ui_musicbox.c 的 Musicbox_Palette（深色主题）
(define BG_R 14)  (define BG_G 18)  (define BG_B 26)   ; 列表页平底色
;; 播放页背景竖向渐变（C 版 bg_top → bg_bot，按 8px 条带填充）
(define BG_TOP_R 13) (define BG_TOP_G 16) (define BG_TOP_B 22)
(define BG_BOT_R 24) (define BG_BOT_G 30) (define BG_BOT_B 40)
(define CARD_FILL_R 27) (define CARD_FILL_G 33) (define CARD_FILL_B 44)
(define EDGE_R 44) (define EDGE_G 53) (define EDGE_B 68)
(define SUR_R 38)  (define SUR_G 47)  (define SUR_B 61)
(define SURP_R 62) (define SURP_G 78) (define SURP_B 100)
(define ACC_R 56)  (define ACC_G 189) (define ACC_B 248)
(define ACCP_R 14) (define ACCP_G 140) (define ACCP_B 200)
(define TXT_R 237) (define TXT_G 242) (define TXT_B 249)
(define DIM_R 140) (define DIM_G 152) (define DIM_B 168)
(define TRK_R 52)  (define TRK_G 63)  (define TRK_B 79)
(define DAN_R 248) (define DAN_G 113) (define DAN_B 113)
(define HL_R 40)   (define HL_G 50)   (define HL_B 68)

;; 字体度量（12px 抗锯齿字模：半角 7px / 全角 12px；16px 抗锯齿：9px / 16px）
(define W_HALF 7)
(define W_FULL 12)
(define W16_HALF 9)
(define W16_FULL 16)

;; ============================================================================
;; 状态
;; ============================================================================

(define page PAGE_LIST)        ; 当前页面
(define list_dir "music")      ; 列表目录：默认 music/；被拉起时 = 待播文件所在目录
(define files (cdr '(0)))      ; 列表目录下的音频**文件名**（不含路径）
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
(define tick_launched #f)       ; 本拍是否刚被拉起（决定要不要走 reopen_after_resume）

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

;; 16px 抗锯齿字模的步进（半角 9px / 全角 16px，同 gfx_font_char_advance）
(define char_w16
  (lambda (cp) (if (>= cp 11904) W16_FULL W16_HALF)))

(define text_w16
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (while (< i n)
      {
        (set! w (+ w (char_w16 (String.charCodeAt i s))))
        (set! i (+ i 1))
      })
    w))

;; 超宽文本按 16px 字模截断为"前缀…"：判据与 C 版
;; musicbox_draw_text_ellipsis_centered 逐字一致——总宽不超限就整串不动，
;; 否则逐字累加，`w + adv + ell_w > max_w` 即停（ell_w = '…' 在 16px 字模的步进 = 9）。
(define clip_text16
  (lambda (s maxw)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (define cw 0)
    (define stop 0)
    (define cut -1)
    (if (<= (text_w16 s) maxw)
        s
        {
          (while (and (< i n) (== stop 0))
            {
              (set! cw (char_w16 (String.charCodeAt i s)))
              (if (> (+ (+ w cw) W16_HALF) maxw)
                  { (set! cut i) (set! stop 1) }
                  { (set! w (+ w cw)) (set! i (+ i 1)) })
            })
          (String.concat (String.slice s 0 cut) "…")
        })))

;; 左对齐绘制（自动按 maxw 截断）
(define draw_fit
  (lambda (s x y maxw r g b)
    (Gfx.draw_text (clip_text s maxw) x y r g b)))

;; 右对齐绘制
(define draw_right
  (lambda (s right y r g b)
    (Gfx.draw_text s (- right (text_w s)) y r g b)))

;; C 版 gfx_font_draw_text 的 y 是**行框顶**，而 (Gfx.draw_text) 的 y 是**墨迹顶**
;; （宿主已把两种字模的行顶空隙补偿掉：12px 抗锯齿为 2px、16px 为 1px，见
;; am_native_Gfx.c 的 native_text_top_gap）。播放页要与 C 版逐像素对齐，凡按 C 版
;; y_top 排的位置一律 +2，故这两个辅助过程只给播放页用。
(define TEXT_TOP_GAP 2)

(define draw_text_c
  (lambda (s x y r g b)
    (Gfx.draw_text s x (+ y TEXT_TOP_GAP) r g b)))

(define draw_right_c
  (lambda (s right y r g b)
    (Gfx.draw_text s (- right (text_w s)) (+ y TEXT_TOP_GAP) r g b)))

;; 毫秒 → "m:ss"
(define fmt_time
  (lambda (ms)
    (define sec (Math.floor (/ ms 1000)))
    (define mm (Math.floor (/ sec 60)))
    (define ss (- sec (* mm 60)))
    (define s0 (String.atom_to_string ss))
    (String.concat (String.concat (String.atom_to_string mm) ":")
                   (if (< ss 10) (String.concat "0" s0) s0))))

;; 采样率 → "44.1 kHz"（0.1kHz 四舍五入，同 C 版 musicbox_fmt_rate）
(define fmt_rate
  (lambda (hz)
    (define k10 (Math.floor (/ (+ hz 50) 100)))
    (define ip (Math.floor (/ k10 10)))
    (String.concat (String.concat (String.atom_to_string ip) ".")
                   (String.concat (String.atom_to_string (- k10 (* ip 10))) " kHz"))))

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

;; 枚举一个目录里的音频文件名（.wav/.mp3，名字升序 —— (Fs.list) 本身已升序）
;; 为什么不用 (Music.list)：它固定只认 music/；被拉起时的目标文件可能在任意目录。
;; **必须在 (Music.open) 之前调用**：hal_fs 的随机访问是全局单句柄，播放期间再列目录
;; 会把句柄换掉（症状：播放卡住/进度乱跳，见 am_native_Music.h）。
(define scan_dir_audio
  (lambda (dir)
    (define raw (Fs.list dir))
    (define out (FSX.new_list))
    (define n 0)
    (define i 0)
    (define nm "")
    (if (list? raw)
        {
          (set! n (length raw))
          (while (< i n)
            {
              (set! nm (get_item raw i))
              (if (== (FSX.ext_match nm "wav,mp3") 1) (push out nm) #f)
              (set! i (+ i 1))
            })
        }
        #f)
    out))

;; 刷新文件列表（首次/资源就绪后调用）；空目录时置"加载中"并请求页面挂载。
;; music/ 走 (Music.list)（与 C 版音乐盒同一个枚举口径）；被拉起的任意目录用
;; (Fs.list) 自己筛。
(define refresh_list
  (lambda ()
    (define l #null)
    (if (String.equals list_dir "music")
        (set! l (Music.list))
        (set! l (scan_dir_audio list_dir)))
    (if (list? l)
        {
          (set! files l)
          (set! nfiles (length files))
          (set! loading (if (> nfiles 0) #f #t))
          (if (>= sel nfiles) (set! sel 0) #f)
          (set! dirty #t)
        }
        #f)))

;; 曲目完整路径：默认 "music/<名字>"；被拉起时 = "<文件所在目录>/<名字>"
;; （目录可能是宿主绝对路径，直接拼好交给 (native Music) 的路径规则）。
;; (FSX.join) 会把 "."/空串当作"资源根"处理，故根目录下的文件写成绝对路径 "//x"——
;; 单独把 "/" 排除掉，免得出现双斜杠。
(define track_path
  (lambda (idx)
    (if (String.equals list_dir "/")
        (String.concat "/" (get_item files idx))
        (FSX.join list_dir (get_item files idx)))))

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

;; ---- 启动参数（App → App 拉起） ------------------------------------------
;;
;; 拉起方（app/filemgr.scm 里选中 .wav/.mp3 后点「进入」）用
;; (App.launch_with "musicbox" 路径) 把控制权交给本 App，并约定参数 = **待播放音频
;; 文件的宿主绝对路径**（'/' 开头；路径规则与 (native Fs)/(native Music) 一致）。
;;
;; 收到参数后：
;;   ① 列表目录换成该文件所在目录，并重新枚举其中的 .wav/.mp3 —— 于是「上一首/下一首」
;;      与「返回」后的列表页都落在同一个目录里（与 app/album.scm 拉起后换浏览目录同构）；
;;   ② 选中该文件并**立即开播**（直接进播放页，不停在列表页）。
;; 目录读不出来、或目标文件不在枚举结果里时退化为「单文件列表」（nfiles=1、cur=0，
;; 上一首/下一首就只重开这一首），仍然能放。
;;
;; (App.launch_arg) **取走即清零**（同 (App.back_requested)）：每拍只消费一条。
;; 两个消费点：boot（冷启动被拉起）与 tick 开头（后台实例被再次拉起，换一个文件）。
(define launch_open
  (lambda ()
    (define a (App.launch_arg))
    (define dir "")
    (define nm "")
    (define lst (FSX.new_list))
    (define n 0)
    (define i 0)
    (define idx -1)
    ;; 返回值必须是 #t/#f：本解释器里只有 #f 为假（见 am_native_App.h 的坑位记录）
    (if (or (null? a) (== (String.length a) 0))
        #f
        {
          (set! nm (FSX.leaf a))
          (set! dir (FSX.parent a))
          ;; (FSX.parent) 对“根下的文件”会因越界保护给出 "."（它面向资源根相对路径）；
          ;; 绝对路径的父目录就是 "/"，改回来，否则会去枚举资源根。
          (if (and (String.equals dir ".") (String.equals (String.slice a 0 1) "/"))
              (set! dir "/")
              #f)
          (set! lst (scan_dir_audio dir))
          (set! n (length lst))
          ;; 在枚举结果里定位目标文件（大小写敏感：它就是目录项原名）
          (set! i 0)
          (while (< i n)
            { (if (String.equals (get_item lst i) nm) (set! idx i) #f)
              (set! i (+ i 1)) })
          (if (< idx 0)
              { (set! lst (FSX.new_list))
                (push lst nm)
                (set! idx 0) }
              #f)
          (set! list_dir dir)
          (set! files lst)
          (set! nfiles (length files))
          (set! sel idx)
          (set! offset (max2 0 (- (+ idx 1) LIST_ROWS)))
          (set! loading #f)
          ;; 目标不一定在 music/：关掉“等 music/ 就绪”的列表轮询（没意义且会白读目录）
          (set! load_polls POLL_MAX)
          (open_track idx)
          (display "[musicbox] launch ") (display a)
          (display " dir=") (display dir)
          (display " idx=") (display idx)
          (display "/") (display nfiles) (newline)
          #t })))

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

;; ---- 矢量图标（不依赖字库里的 ▶/⏸ 等码点）------------------------------
;; 几何与 C 版 ui_musicbox.c 的 musicbox_draw_play_icon / _skip_icon / _vol_icon
;; **逐项等价**：三角形走 (Gfx.draw_triangle)（= C 版 gfx_draw_triangle），
;; 条形/圆角块走 (Gfx.rounded_rect)（= C 版 gfx_draw_rounded_rectangle）。
;; 早先的"逐列 draw_line 拼三角形"是妥协写法，已删除。

;; 播放 / 暂停图标：播放态两根圆角竖条，暂停态向右的三角形
(define draw_play_icon
  (lambda (cx cy rad is_playing cr cg cb)
    (define bh 0)
    (define bw 0)
    (define gap 0)
    (define hh 0)
    (define ww 0)
    (if is_playing
        {
          (set! bh rad)
          (set! bw (Math.floor (/ rad 4)))
          (if (< bw 3) (set! bw 3) #f)
          (set! gap (Math.floor (/ rad 3)))
          (Gfx.rounded_rect (- (- cx (Math.floor (/ gap 2))) bw) (- cy (Math.floor (/ bh 2)))
                            bw bh 2 2 2 2 cr cg cb 1)
          (Gfx.rounded_rect (+ cx (Math.floor (/ gap 2))) (- cy (Math.floor (/ bh 2)))
                            bw bh 2 2 2 2 cr cg cb 1)
        }
        {
          (set! hh (Math.floor (/ (* rad 4) 5)))
          (set! ww (Math.floor (/ (* rad 3) 4)))
          (Gfx.draw_triangle (- cx (Math.floor (/ ww 3))) (- cy (Math.floor (/ hh 2)))
                             (- cx (Math.floor (/ ww 3))) (+ cy (Math.floor (/ hh 2)))
                             (+ cx (Math.floor (/ (* ww 2) 3))) cy
                             0 cr cg cb 1)
        })))

;; 上一首 / 下一首图标：一根圆角竖条 + 指向该方向（dir<0 左 / dir>0 右）的三角形
(define draw_skip_icon
  (lambda (cx cy rad dir cr cg cb)
    (define hh 0)
    (define bar_w 0)
    (define bar_x 0)
    (set! hh (Math.floor (/ (* rad 4) 5)))
    (if (< hh 6) (set! hh 6) #f)
    (set! bar_w (if (>= rad 20) 3 2))
    (set! bar_x (if (< dir 0)
                    (- cx (Math.floor (/ (* rad 9) 16)))
                    (+ cx (Math.floor (/ (* rad 7) 16)))))
    (Gfx.rounded_rect bar_x (- cy (Math.floor (/ hh 2))) bar_w hh 1 1 1 1 cr cg cb 1)
    (if (< dir 0)
        (Gfx.draw_triangle (+ cx (Math.floor (/ (* rad 7) 16))) (- cy (Math.floor (/ hh 2)))
                           (+ cx (Math.floor (/ (* rad 7) 16))) (+ cy (Math.floor (/ hh 2)))
                           (- cx (Math.floor (/ (* rad 5) 16))) cy
                           0 cr cg cb 1)
        (Gfx.draw_triangle (- cx (Math.floor (/ (* rad 7) 16))) (- cy (Math.floor (/ hh 2)))
                           (- cx (Math.floor (/ (* rad 7) 16))) (+ cy (Math.floor (/ hh 2)))
                           (+ cx (Math.floor (/ (* rad 5) 16))) cy
                           0 cr cg cb 1))))

;; 音量减 / 加图标：12×2 的圆角横条（加号再加一条竖条）
(define draw_vol_icon
  (lambda (cx cy is_plus cr cg cb)
    (Gfx.rounded_rect (- cx 6) (- cy 1) 12 2 1 1 1 1 cr cg cb 1)
    ;; 注意：灵机里只有 #f 为假，0 是真值 —— is_plus 必须是布尔（传 0 会画出加号）
    (if is_plus
        (Gfx.rounded_rect (- cx 1) (- cy 6) 2 12 1 1 1 1 cr cg cb 1)
        #f)))

;; 顶栏
;;   列表页：平底色条 + 分隔线 + 左侧标题（本 App 自己的列表页，与 C 版无关）；
;;   播放页：**不画底色/分隔线**（C 版顶栏直接叠在渐变背景上），只画「返回」胶囊
;;   （灵机 App 必须保留的顶部按钮）与 C 版标题组（强调色竖条 + 12px 标题），标题组
;;   整体右移到「返回」右侧；右上角「退出」仍由 EX 统一绘制。
(define draw_topbar
  (lambda ()
    (if (== page PAGE_LIST)
        {
          (Gfx.fill_rect 0 0 SW TOP_H BG_R BG_G BG_B)
          (Gfx.fill_rect 0 TOP_H SW 1 EDGE_R EDGE_G EDGE_B)
          (Gfx.rounded_rect 12 7 3 12 1 1 1 1 ACC_R ACC_G ACC_B 1)
          (Gfx.draw_text "音乐盒" 20 6 TXT_R TXT_G TXT_B)
          ;; 右对齐到「退出」按钮左侧 6px（按钮矩形 [SW-(EX.btn_w+4), SW-4]，别写死 48）
          (draw_right (String.concat (String.atom_to_string nfiles) " 首")
                      (- SW (+ EX.btn_w 10)) 7 DIM_R DIM_G DIM_B)
        }
        {
          (Gfx.rounded_rect BACK_X BACK_Y BACK_W BACK_H 9 9 9 9
                            (press_bg_r ARM_BACK SUR_R)
                            (press_bg_g ARM_BACK SUR_G)
                            (press_bg_b ARM_BACK SUR_B) 1)
          (Gfx.draw_text "返回" (+ BACK_X 13) (+ BACK_Y 3) TXT_R TXT_G TXT_B)
          (Gfx.rounded_rect TITLE_BAR_X 13 3 12 1 1 1 1 ACC_R ACC_G ACC_B 1)
          (draw_text_c "音乐盒" TITLE_X 10 TXT_R TXT_G TXT_B)
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
                  (Gfx.fill_circle_aa (- SW 14) (+ y 12) 3 ACC_R ACC_G ACC_B 1)
                  #f)
              (set! i (+ i 1))
            })
          ;; 滚动指示（内容多于可见行时）
          (if (> nfiles LIST_ROWS)
              {
                (Gfx.rounded_rect (- SW 3) LIST_Y 3 (- (- SH LIST_Y) 4) 1 1 1 1
                                  TRK_R TRK_G TRK_B 1)
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

;; 第二行信息：文件格式参数（与 C 版 musicbox_format_info 逐字一致：
;;   WAV → "WAV  ·  44.1 kHz  ·  16 bit  ·  立体声"
;;   MP3 → "MP3  ·  44.1 kHz  ·  320 kbps  ·  立体声"（码率未知时省掉该段）
;;   打不开 → "文件格式不支持或已损坏"
;; 注意 "·" 两侧各两个空格——那是 C 版 swprintf 里写死的分隔符。
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
    (if (list? inf)
        {
          (set! fmt (Music.format))
          (set! rate (get_item inf 0))
          (set! ch (get_item inf 1))
          (set! bits (get_item inf 2))
          (set! br (get_item inf 3))
          (set! s (String.concat (String.atom_to_string fmt) "  ·  "))
          (set! s (String.concat s (fmt_rate rate)))
          (if (String.equals fmt "WAV")
              (set! s (String.concat s (String.concat "  ·  "
                            (String.concat (String.atom_to_string bits) " bit"))))
              (if (> br 0)
                  (set! s (String.concat s (String.concat "  ·  "
                                (String.concat (String.atom_to_string br) " kbps"))))
                  #f))
          (set! s (String.concat s (String.concat "  ·  "
                        (if (== ch 1) "单声道" "立体声"))))
          s
        }
        "文件格式不支持或已损坏")))

;; 信息卡片：外层 card_edge 边框 + 内缩 1px 卡片底、序号/总数、状态灯+状态、
;; 16px 文件名（超宽截断）、12px 格式行。几何/配色/字号全部同 C 版 musicbox_draw_card。
(define draw_card
  (lambda ()
    (define st "")
    (define nm "")
    (define idx_txt "")
    (define right 0)
    (define tx 0)
    (define dot_r 0)
    (define dot_g 0)
    (define dot_b 0)
    (Gfx.rounded_rect CARD_X CARD_Y CARD_W CARD_H CARD_RAD CARD_RAD CARD_RAD CARD_RAD
                      EDGE_R EDGE_G EDGE_B 1)
    (Gfx.rounded_rect (+ CARD_X 1) (+ CARD_Y 1) (- CARD_W 2) (- CARD_H 2)
                      (- CARD_RAD 1) (- CARD_RAD 1) (- CARD_RAD 1) (- CARD_RAD 1)
                      CARD_FILL_R CARD_FILL_G CARD_FILL_B 1)
    ;; 序号（强调色）/ 总数（弱化色）
    (set! idx_txt (if (< cur nfiles) (String.atom_to_string (+ cur 1)) "-"))
    (draw_text_c idx_txt COUNTER_X COUNTER_Y ACC_R ACC_G ACC_B)
    (draw_text_c (String.concat " / " (String.atom_to_string nfiles))
                (+ COUNTER_X (text_w idx_txt)) COUNTER_Y DIM_R DIM_G DIM_B)
    ;; 状态（右对齐）+ 左侧状态指示灯：错误=警示色 / 播放中=强调色点 / 已暂停=弱化色点；
    ;; 文字颜色除错误态外一律弱化色（同 C 版 musicbox_draw_status）
    (if fail
        (set! st "错误")
        (if playing (set! st "播放中") (set! st "已暂停")))
    (if fail
        { (set! dot_r DAN_R) (set! dot_g DAN_G) (set! dot_b DAN_B) }
        (if playing
            { (set! dot_r ACC_R) (set! dot_g ACC_G) (set! dot_b ACC_B) }
            { (set! dot_r DIM_R) (set! dot_g DIM_G) (set! dot_b DIM_B) }))
    (set! right (- (+ CARD_X CARD_W) 14))
    (set! tx (- right (text_w st)))
    (draw_text_c st tx COUNTER_Y
                (if fail DAN_R DIM_R) (if fail DAN_G DIM_G) (if fail DAN_B DIM_B))
    (Gfx.fill_circle_aa (- tx 10) (+ COUNTER_Y 7) 3 dot_r dot_g dot_b 1)
    ;; 文件名（16px 抗锯齿字模，超宽截断为"前缀…"后居中）
    (if fail
        (set! nm "无法播放该文件")
        (set! nm (if (< cur nfiles) (get_item files cur) "")))
    (Gfx.draw_text_centered 2 (clip_text16 nm (- CARD_W 28)) (/ SW 2) FILENAME_CY
                            TXT_R TXT_G TXT_B 1)
    ;; 格式参数（12px，居中；C 版不截断）
    (Gfx.draw_text_centered 1 (format_line) (/ SW 2) FORMAT_CY DIM_R DIM_G DIM_B 1)))

;; 内嵌进度面板（在卡片内，不透明底整块重绘）：已播/总时长、圆角轨道、强调色填充、
;; 三层拇指（光晕 alpha 40 / 强调色环 / 白色内芯）。几何同 C 版 musicbox_draw_progress。
(define draw_progress
  (lambda ()
    (define ratio 0.0)
    (define dur 0)
    (define el 0)
    (define fw 0)
    (define tx 0)
    (define rh 2)
    (define tcy 0)
    ;; 先用卡片色清底，再画圆角面板：圆角抗锯齿像素每次都以卡片底色为背景重新混合，
    ;; 避免高频局部重绘使面板四角逐次叠加变深/变糊（C 版同一做法）。
    (Gfx.fill_rect PROG_X PROG_Y PROG_W PROG_H CARD_FILL_R CARD_FILL_G CARD_FILL_B)
    (Gfx.rounded_rect PROG_X PROG_Y PROG_W PROG_H PROG_RAD PROG_RAD PROG_RAD PROG_RAD
                      SUR_R SUR_G SUR_B 1)
    (set! ratio (if (>= seek_prev 0.0) seek_prev (Music.progress)))
    (set! ratio (clamp ratio 0.0 1.0))
    (set! dur (Music.duration_ms))
    (set! el (if (>= seek_prev 0.0) (Math.floor (* dur ratio)) (Music.elapsed_ms)))
    (if (> el dur) (set! el dur) #f)
    (draw_text_c (fmt_time el) (+ PROG_X 8) TIME_Y TXT_R TXT_G TXT_B)
    (draw_right_c (fmt_time dur) (- (+ PROG_X PROG_W) 8) TIME_Y DIM_R DIM_G DIM_B)
    ;; 轨道 + 已播放部分（圆角半径 = 轨道半高 = 2）
    (set! rh (Math.floor (/ TRK_H 2)))
    (if (< rh 1) (set! rh 1) #f)
    (Gfx.rounded_rect TRK_X TRK_Y TRK_W TRK_H rh rh rh rh TRK_R TRK_G TRK_B 1)
    (set! fw (Math.floor (+ (* TRK_W ratio) 0.5)))
    (if (> fw TRK_W) (set! fw TRK_W) #f)
    (if (> fw 0)
        (Gfx.rounded_rect TRK_X TRK_Y fw TRK_H rh rh rh rh ACC_R ACC_G ACC_B 1)
        #f)
    (set! tx (+ TRK_X fw))
    (set! tcy (+ TRK_Y (Math.floor (/ TRK_H 2))))
    (Gfx.fill_circle_aa tx tcy (+ THUMB_R 3) ACC_R ACC_G ACC_B 40)
    (Gfx.fill_circle_aa tx tcy THUMB_R ACC_R ACC_G ACC_B 1)
    (Gfx.fill_circle_aa tx tcy (- THUMB_R 2) 250 250 252 1)))

;; 按住（active）配色的取色器：arm 与控件 id 相同即为"正被按住"，
;; 于是按下沿的下一拍就换色、松手沿（或拖出取消）的下一拍复原——
;; 与 C 版 musicbox 的 s_ui_pressed 高亮同一套语义。
(define press_bg_r (lambda (id normal) (if (== arm id) SURP_R normal)))
(define press_bg_g (lambda (id normal) (if (== arm id) SURP_G normal)))
(define press_bg_b (lambda (id normal) (if (== arm id) SURP_B normal)))

;; 播放控制：上一首/下一首（surface 圆底）+ 播放·暂停主按钮
;; （可播放态 = 投影 + 强调色圆 + 白色图标；错误态 = surface 圆 + 弱化色三角）。
;; 圆心/半径同 C 版 musicbox_draw_transport：主键 (SW/2, card_y+card_h+28) r=24、
;; 上/下一首 ±72 r=17。
(define draw_transport
  (lambda ()
    (define play_ok (not fail))
    (define icon_playing #f)
    (set! icon_playing (and play_ok playing))
    (Gfx.fill_circle_aa PREV_CX PLAY_CY SKIP_R
                        (press_bg_r ARM_PREV SUR_R)
                        (press_bg_g ARM_PREV SUR_G)
                        (press_bg_b ARM_PREV SUR_B)
                        1)
    (draw_skip_icon PREV_CX PLAY_CY SKIP_R -1 TXT_R TXT_G TXT_B)
    (Gfx.fill_circle_aa NEXT_CX PLAY_CY SKIP_R
                        (press_bg_r ARM_NEXT SUR_R)
                        (press_bg_g ARM_NEXT SUR_G)
                        (press_bg_b ARM_NEXT SUR_B)
                        1)
    (draw_skip_icon NEXT_CX PLAY_CY SKIP_R 1 TXT_R TXT_G TXT_B)
    (if play_ok
        {
          (Gfx.fill_circle_aa PLAY_CX (+ PLAY_CY 2) (+ PLAY_R 2) 0 0 0 70)
          (Gfx.fill_circle_aa PLAY_CX PLAY_CY PLAY_R
                              (if (== arm ARM_PLAY) ACCP_R ACC_R)
                              (if (== arm ARM_PLAY) ACCP_G ACC_G)
                              (if (== arm ARM_PLAY) ACCP_B ACC_B)
                              1)
          (draw_play_icon PLAY_CX PLAY_CY PLAY_R icon_playing 255 255 255)
        }
        {
          (Gfx.fill_circle_aa PLAY_CX PLAY_CY PLAY_R SUR_R SUR_G SUR_B 1)
          (draw_play_icon PLAY_CX PLAY_CY PLAY_R #f DIM_R DIM_G DIM_B)
        })))

;; 底部音量排：左侧「音量」名称 + 右对齐百分比 + 圆角音量条 + 减/加按钮，
;; 几何/配色同 C 版 musicbox_draw_volume（按钮 34x30 圆角 10、音量条高 4 圆角 2）。
(define draw_volume
  (lambda ()
    (define fw 0)
    (define pct 0)
    (define rh 2)
    (draw_text_c "音量" VOL_METER_X VOL_TEXT_Y DIM_R DIM_G DIM_B)
    (set! pct (Math.floor (/ (* vol 100) VOL_MAX)))
    (draw_right_c (String.concat (String.atom_to_string pct) "%")
                 (+ VOL_METER_X VOL_METER_W) VOL_TEXT_Y TXT_R TXT_G TXT_B)
    (set! rh (Math.floor (/ VOL_METER_H 2)))
    (if (< rh 1) (set! rh 1) #f)
    (Gfx.rounded_rect VOL_METER_X VOL_METER_Y VOL_METER_W VOL_METER_H rh rh rh rh
                      TRK_R TRK_G TRK_B 1)
    (set! fw (Math.floor (/ (* VOL_METER_W vol) VOL_MAX)))
    (if (> fw 0)
        (Gfx.rounded_rect VOL_METER_X VOL_METER_Y fw VOL_METER_H rh rh rh rh
                          ACC_R ACC_G ACC_B 1)
        #f)
    ;; 减 / 加按钮（按住换 surface_press；到 0 / 满量程时图标转弱化色）
    (Gfx.rounded_rect VOL_DN_X VOL_Y VOL_BTN_W VOL_BTN_H
                      VOL_BTN_RAD VOL_BTN_RAD VOL_BTN_RAD VOL_BTN_RAD
                      (press_bg_r ARM_VOL_DN SUR_R)
                      (press_bg_g ARM_VOL_DN SUR_G)
                      (press_bg_b ARM_VOL_DN SUR_B) 1)
    (Gfx.rounded_rect VOL_UP_X VOL_Y VOL_BTN_W VOL_BTN_H
                      VOL_BTN_RAD VOL_BTN_RAD VOL_BTN_RAD VOL_BTN_RAD
                      (press_bg_r ARM_VOL_UP SUR_R)
                      (press_bg_g ARM_VOL_UP SUR_G)
                      (press_bg_b ARM_VOL_UP SUR_B) 1)
    (draw_vol_icon (+ VOL_DN_X (Math.floor (/ VOL_BTN_W 2)))
                   (+ VOL_Y (Math.floor (/ VOL_BTN_H 2))) #f
                   (if (<= vol 0) DIM_R TXT_R) (if (<= vol 0) DIM_G TXT_G)
                   (if (<= vol 0) DIM_B TXT_B))
    (draw_vol_icon (+ VOL_UP_X (Math.floor (/ VOL_BTN_W 2)))
                   (+ VOL_Y (Math.floor (/ VOL_BTN_H 2))) #t
                   (if (>= vol VOL_MAX) DIM_R TXT_R) (if (>= vol VOL_MAX) DIM_G TXT_G)
                   (if (>= vol VOL_MAX) DIM_B TXT_B))))

;; 背景竖向渐变：按 8px 条带填充，条带按全局 y 对齐，颜色取条带中线 y+4 处做整数插值
;; ——与 C 版 musicbox_fill_bg / musicbox_bg_color 同一算式（无浮点舍入差异）。
(define draw_bg_gradient
  (lambda ()
    (define by 0)
    (define t 0)
    (define h 0)
    (define rr 0)
    (define gg 0)
    (define bb 0)
    (while (< by SH)
      {
        (set! t (Math.floor (/ (* (+ by 4) 255) (- SH 1))))
        (if (< t 0) (set! t 0) #f)
        (if (> t 255) (set! t 255) #f)
        (set! h (min2 8 (- SH by)))
        (set! rr (+ BG_TOP_R (Math.floor (/ (* (- BG_BOT_R BG_TOP_R) t) 255))))
        (set! gg (+ BG_TOP_G (Math.floor (/ (* (- BG_BOT_G BG_TOP_G) t) 255))))
        (set! bb (+ BG_TOP_B (Math.floor (/ (* (- BG_BOT_B BG_TOP_B) t) 255))))
        (Gfx.fill_rect 0 by SW h rr gg bb)
        (set! by (+ by 8))
      })))

(define draw_play
  (lambda ()
    (draw_bg_gradient)
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
          (if (in_rect x y (- PROG_X 6) (- PROG_Y 4) (+ PROG_W 12) (+ PROG_H 8))
              {
                (set! arm ARM_SEEK)
                (set! seek_prev (ratio_from_x x))
                (set! dirty #t)
              }
              #f)
          (if (in_circle x y PREV_CX PLAY_CY (+ SKIP_R 6)) (set! arm ARM_PREV) #f)
          (if (in_circle x y PLAY_CX PLAY_CY (+ PLAY_R 5)) (set! arm ARM_PLAY) #f)
          (if (in_circle x y NEXT_CX PLAY_CY (+ SKIP_R 6)) (set! arm ARM_NEXT) #f)
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
        })
    ;; 按住态是"看得见的状态"：按下沿之后立刻重绘一次（不等 100ms 的进度节拍）
    (set! dirty #t)))

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
          (if (in_rect x y (- PROG_X 6) (- PROG_Y 4) (+ PROG_W 12) (+ PROG_H 8))
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
        (if (in_circle x y PREV_CX PLAY_CY (+ SKIP_R 6)) (switch_track -1) #f)
        #f)
    (if (== arm ARM_NEXT)
        (if (in_circle x y NEXT_CX PLAY_CY (+ SKIP_R 6)) (switch_track 1) #f)
        #f)
    (set! arm ARM_NONE)
    (set! moved #f)
    ;; 松手/拖出取消：无论是否触发了动作，都要把按住配色擦回常态
    (set! dirty #t)))

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
          (display " dir=") (display list_dir)
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
      ;; 后台实例被再次拉起（换一个文件）：先消费启动参数并开播。
      ;; 新曲目已由 (Music.open) 重建过音频会话，故这一拍**不要**再走
      ;; reopen_after_resume（它会把旧曲目按旧位置抢回来）。
      (set! tick_launched (launch_open))
      (if (App.resumed)
          (if tick_launched #f (reopen_after_resume))
          #f)                                         ; 恢复：重建音频会话（须早于 EX.check）
      (EX.check)              ; 硬 A/ESC 与右上角「退出」按钮（可能阻塞，返回后继续）
      (set! ntick (+ ntick 1))
      (set! now_ms (System.timestamp))
      (handle_touch)
      (handle_keys)
      (feed_audio)
      ;; 资源未就绪：按 500ms 节奏重试列目录（WASM 下 music/ 由页面后台挂载，可能还没拉完）
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
;; 冷启动就可能带着启动参数（filemgr 点开音频）：有参数就直接播目标文件并进播放页，
;; 没有参数才是“列 music/”的旧启动路径。
(if (launch_open) #f (refresh_list))
(draw)
(set! timer_id (System.set_interval TICK_MS tick))
(EX.install_with draw)
(EX.on_exit on_exit)
(display "[musicbox] start files=") (display nfiles) (newline)
