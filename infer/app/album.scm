;; ============================================================================
;; 时光集（Album）—— 灵机应用程序：图片浏览 / 相册
;;
;; 参考实现：infer/ui_app.c 的 STATE_ALBUM（C 版「时光集」，键盘浏览 PLATFORM_ROOT_DIR/
;; image 下的图片）。本 App 是同一功能的 Scheme 重做，只针对 **WASM-GUI 目标**
;; （infer/build_wasm_gui.sh → infer/webgui/），并补齐触屏交互。
;;
;; 与 C 版的对应关系：
;;   · 目录         image/（相对业务资源根；(native Fs)/(native Gfx) 的同一套路径规则）
;;   · 排序         枚举顺序（hal_fs 的 list_files 按名字升序；C 版 sort_strings 同序）
;;   · 键位         1=自动放映开关；2/4/5/7/8/←/0=上一张；3/6/Shift/9/Ctrl/→/Enter=下一张
;;                  （C 版的 NANO_KEY_* 逻辑键值，浏览器物理键盘经 (native Key) 忠实送达）
;;   · 自动放映     6000ms 一张，底部 2px 进度条（颜色同 C 版 0,170,255）
;;   · 退出         硬 A/ESC 与右上角「退出」按钮 → 宿主全局确认框（app/lib/exit.scm）
;;
;; 相对 C 版的改进（都写在这里，便于回看为什么不一样）：
;;   ① **等比缩放、居中留黑边**。C 版把图直接解到 SCREEN_WIDTH×SCREEN_HEIGHT
;;      （stbir_resize 拉伸到满屏，宽高比失真）；本版按 min(宽比, 高比) 缩放后居中，
;;      竖拍/超宽图不再被压扁。
;;   ② **触屏**：左右滑动切图；轻点画面左/右 1/4 切图、中间切换「沉浸模式」
;;      （隐藏上下工具条，照片满屏；再点一下恢复）；底部三枚按钮（上一张 / 自动 / 下一张）
;;      按下高亮、在同一按钮上松手才生效（拖出取消），与音乐盒/词典 App 同一范式。
;;   ③ **不睡 3 秒列目录**。C 版进相册先画文件清单再 sleep_in_ms(3000)，是调试期产物；
;;      本版直接显示第一张，文件名/序号/原始分辨率常驻顶栏。
;;   ④ **空目录/坏图不再是死路**：目录为空时自动重扫（前 30 秒每 0.5 秒——WASM 的
;;      image/ 属 manifest 的 p2 后台挂载层，App 可能先于资源就绪启动；之后降到 2 秒）；
;;      单张图读不出来时画占位提示，仍可用上/下一张跳过。非空目录在前 30 秒每 5 秒
;;      静默比对一次条目数，后台挂载陆续到位时列表自增。
;;   ⑤ **像素不进解释器堆**：解码/缩放/缓存全在宿主侧（Gfx.draw_image 的 LRU，
;;      键 = 路径 + 目标尺寸）；Scheme 只拿到 image_size 的两个整数做排版。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二参数；`/` 与 `mod` 恒返回浮点
;; （下标一律 Math.floor）；无 let/list/map/append；cond 每子句只接受单个表达式；
;; 块内不 define；'() 是模块级静态槽（空表一律用 (cdr '(0)) 现造）。
;; ============================================================================

(native Gfx)
(native Fs)
(native Key)
(native Touch)
(native String)
(native Math)
(native System)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC / 右上角按钮 → 全局退出确认

;; ---- 常量 ------------------------------------------------------------------

(define IMG_DIR "image")     ; 图片目录（相对业务资源根）
(define TICK_MS 33)          ; 33ms 一拍（≈30Hz，触屏点按不漏）
(define AUTO_MS 6000)        ; 自动放映间隔（与 C 版一致）
(define TOP_H 18)            ; 顶栏高（12px 字模行高 15 + 上下留白）
(define BAR_H 22)            ; 底栏高（按钮 16 + 进度条 2 + 留白）
(define BTN_H 16)            ; 按钮高
(define SW 320)              ; 画布宽（boot 时按 Gfx.width 覆盖）
(define SH 240)              ; 画布高（boot 时按 Gfx.height 覆盖）

;; ---- 状态 ------------------------------------------------------------------

(define names (cdr '(0)))    ; 过滤后的图片文件名表（升序，来自 Fs.list）
(define n 0)                 ; 图片数
(define dir_n -2)            ; 上次枚举的目录条目数（-2 = 尚未枚举；用于静默重扫比对）
(define dir_ok 0)            ; image/ 目录是否可枚举
(define idx 0)               ; 当前图片下标（0..n-1）
(define cur_name "")         ; 当前图片文件名
(define cur_path "")         ; 当前图片完整相对路径
(define cur_ok 0)            ; 当前图片头是否可读（0 = 坏图/非图片）
(define img_w 0)             ; 当前图片原始宽（Gfx.image_size 只读文件头）
(define img_h 0)             ; 当前图片原始高
(define dx 0) (define dy 0)  ; 当前图片在屏幕上的落点/尺寸（等比缩放后）
(define dw 0) (define dh 0)
(define p_top 0)             ; 照片区上沿
(define p_h SH)              ; 照片区高度
(define chrome 1)            ; 1 = 显示上下工具条；0 = 沉浸模式（照片满屏）
(define auto_on 0)           ; 自动放映（C 版初值也是关）
(define auto_due 0)          ; 下一次自动切图的时刻（ms）
(define dirty 1)             ; 需要整屏重绘
(define last_pw -1)          ; 上一次画出的进度条像素宽（-1 = 未画）
(define t_down 0)            ; 上一拍的触屏电平
(define t_x0 0) (define t_y0 0) ; 本次按下的落点
(define t_start_btn 0)       ; 本次按下是否落在底部按钮上（是则松手不做滑动/轻点）
(define arm_btn -1)          ; 武装中的按钮（-1 无 / 0 上一张 / 1 自动 / 2 下一张）
(define empty_ctr 0)         ; 空目录重扫计数（15 拍 ≈ 0.5s）
(define rescan_ctr 0)        ; 后台挂载期静默重扫计数（150 拍 ≈ 5s）
(define stat_ctr 0)          ; 诊断行计数（60 拍 ≈ 2s）
(define hb 0)                ; 心跳计数（30 拍 ≈ 1s）
(define hb_on 0)             ; 心跳点的明/暗相
(define boot_ms 0)           ; 启动时刻（ms）
(define ts 0)                ; 定时器编号

;; ---- 小工具 ----------------------------------------------------------------

;; 造一个全新的空表（'() 是模块级静态槽，反复 push 会跨调用累积）
(define new_list (lambda () (cdr '(0))))

(define min2 (lambda (a b) (if (< a b) a b)))
(define max2 (lambda (a b) (if (> a b) a b)))

;; 字符步进（12px 抗锯齿字模：半角 7px、全角 12px；与 app/lib/menu_list.scm 同口径）
(define char_w (lambda (cp) (if (>= cp 11904) 12 7)))   ; 11904 = 0x2E80

;; 一行文本的像素宽
(define text_w
  (lambda (s)
    (define nn (String.length s))
    (define k 0)
    (define w 0)
    (while (< k nn)
      { (set! w (+ w (char_w (String.charCodeAt k s))))
        (set! k (+ k 1)) })
    w))

;; 居中落点（x 坐标）
(define center_x
  (lambda (s)
    (define w (text_w s))
    (Math.floor (/ (- SW w) 2))))

;; 按像素宽裁剪（超出用 … 代替）——与 menu_list 的同名逻辑同口径
(define clip_text
  (lambda (s maxw)
    (define nn (String.length s))
    (define k 0)
    (define w 0)
    (define cw 0)
    (define stop 0)
    (define cut -1)
    (while (and (< k nn) (== stop 0))
      { (set! cw (char_w (String.charCodeAt k s)))
        (if (> (+ w cw) (- maxw 7))
            { (set! cut k) (set! stop 1) }
            { (set! w (+ w cw)) (set! k (+ k 1)) }) })
    (if (< cut 0) s
        (if (== cut 0) ""
            (String.concat (String.slice s 0 cut) "…")))))

;; 小写化（只用于扩展名判定；无 String.toLowerCase）
(define lower
  (lambda (s)
    (define nn (String.length s))
    (define k 0)
    (define cp 0)
    (define out "")
    (while (< k nn)
      { (set! cp (String.charCodeAt k s))
        (if (and (>= cp 65) (<= cp 90)) (set! cp (+ cp 32)) #f)
        (set! out (String.concat out (String.fromCharCode cp)))
        (set! k (+ k 1)) })
    out))

;; 是否是本 App 支持的图片扩展名（stb_image 能解码的常见格式）
;; 注意取**后 4 个**与**后 5 个**字符分别比对：.png/.jpg/... 是 4 字符、.jpeg 是 5 字符，
;; 只按固定窗口取 5 个字符会把 4 字符扩展名的前一个字符带进来（实测踩过）。
(define ext_ok
  (lambda (name)
    (define L (String.length name))
    (define s4 "")
    (define s5 "")
    (if (< L 4) #f
        { (set! s4 (lower (String.slice name (- L 4) L)))
          (set! s5 (if (< L 5) s4 (lower (String.slice name (- L 5) L))))
          (or (String.equals s4 ".png")
              (String.equals s4 ".jpg")
              (String.equals s4 ".bmp")
              (String.equals s4 ".gif")
              (String.equals s4 ".tga")
              (String.equals s5 ".jpeg")) })))

(define is_hidden (lambda (name) (String.equals (String.charAt name 0) ".")))

;; 顶栏的分辨率文本（坏图时为空）
(define dims_text
  (lambda ()
    (define a "")
    (define b "")
    (if (== cur_ok 1)
        { (set! a (String.atom_to_string img_w))
          (set! b (String.atom_to_string img_h))
          (String.concat a (String.concat "x" b)) }
        "")))

;; 底栏的序号文本（i/N）
(define idx_text
  (lambda ()
    (define a "")
    (define b "")
    (if (> n 0)
        { (set! a (String.atom_to_string (+ idx 1)))
          (set! b (String.atom_to_string n))
          (String.concat a (String.concat "/" b)) }
        "")))

;; ---- 目录枚举 --------------------------------------------------------------

;; 重新枚举 image/：重建 names/n，并尽量把「当前文件」保持在原处。
;;   · 首次调用（cur_name 为空）→ 选中第 0 张；
;;   · 后台挂载陆续到位（文件数变化）→ 当前文件仍在则原地下标不变，否则回第 0 张。
(define reload
  (lambda ()
    (define raw (Fs.list IMG_DIR))
    (define cnt 0)
    (define k 0)
    (define nm "")
    (define old cur_name)
    (define found -1)
    (set! names (new_list))
    (set! n 0)
    ;; 注意用 list? 而不是 null? 判“枚举失败”：null? 对空表与 #null 都为真，
    ;; 空目录（合法的空表）会被误判成读不到目录（实测踩过）。
    (if (list? raw)
        { (set! dir_ok 1)
          (set! cnt (length raw))
          (set! dir_n cnt)
          (set! k 0)
          (while (< k cnt)
            { (set! nm (get_item raw k))
              (if (and (ext_ok nm) (not (is_hidden nm)))
                  { (push names nm) (set! n (+ n 1)) }
                  #f)
              (set! k (+ k 1)) }) }
        { (set! dir_ok 0) (set! dir_n -1) })
    ;; 定位原来的当前文件（按文件名；列表已升序，找到即用）
    (set! found -1)
    (if (and (> n 0) (not (== (String.length old) 0)))
        { (set! k 0)
          (while (< k n)
            { (if (and (< found 0) (String.equals (get_item names k) old))
                  (set! found k)
                  #f)
              (set! k (+ k 1)) }) }
        #f)
    (if (>= found 0) (set! idx found) (set! idx 0))
    ;; 当前文件变了 / 从未打开 / 上次是坏图 → 重新读文件头
    (if (> n 0)
        (if (or (< found 0) (not (== cur_ok 1)))
            (open_current)
            (set! dirty 1))
        { (set! cur_ok 0)
          (set! cur_name "")
          (set! cur_path "") })
    #t))

;; 重新枚举并把当前图片的落点算一遍（坏图占位页也走这里，保证 chrome 切换正常）
(define open_current
  (lambda ()
    (define dims #null)
    (define nm "")
    (set! nm (get_item names idx))
    (set! cur_name nm)
    (set! cur_path (String.concat (String.concat IMG_DIR "/") nm))
    (set! dims (Gfx.image_size cur_path))
    (if (null? dims)
        { (set! cur_ok 0)
          (set! img_w 0)
          (set! img_h 0)
          (display "[album] unreadable ") (display cur_path) (newline) }
        { (set! img_w (get_item dims 0))
          (set! img_h (get_item dims 1))
          (set! cur_ok 1)
          (display "[album] open ") (display cur_path)
          (display " ") (display img_w) (display "x") (display img_h) (newline) })
    (layout_photo)
    (set! dirty 1)
    (if (== auto_on 1) (set! auto_due (+ (System.timestamp) AUTO_MS)) #f)
    #t))

;; ---- 版式 ------------------------------------------------------------------

;; 按「显示工具条 / 沉浸」计算照片区，再按 min(宽比, 高比) 等比缩放居中。
;; 照片区之外保持纯黑（上下工具条随后覆盖），因此竖拍/超宽图是黑边而非拉伸。
(define layout_photo
  (lambda ()
    (define s 1.0)
    (if (== chrome 1)
        { (set! p_top TOP_H) (set! p_h (- (- SH TOP_H) BAR_H)) }
        { (set! p_top 0) (set! p_h SH) })
    (if (and (== cur_ok 1) (> img_w 0) (> img_h 0))
        { (set! s (/ SW img_w))
          (if (< (/ p_h img_h) s) (set! s (/ p_h img_h)) #f)
          (set! dw (Math.floor (* img_w s)))
          (set! dh (Math.floor (* img_h s)))
          (set! dw (max2 1 (min2 SW dw)))
          (set! dh (max2 1 (min2 p_h dh)))
          (set! dx (Math.floor (/ (- SW dw) 2)))
          (set! dy (+ p_top (Math.floor (/ (- p_h dh) 2)))) }
        { (set! dx 0) (set! dy p_top) (set! dw SW) (set! dh p_h) })
    #t))

;; ---- 图片切换 / 自动放映 ---------------------------------------------------

;; 相对切换（d = ±1；只有上/下一张两处调用，故直接回绕，不用 mod）
(define go
  (lambda (d)
    (if (> n 0)
        { (set! idx (+ idx d))
          (if (< idx 0) (set! idx (- n 1)) #f)
          (if (>= idx n) (set! idx 0) #f)
          (open_current)
          (display "[album] idx=") (display (+ idx 1))
          (display "/") (display n) (newline) }
        #f)))

(define toggle_auto
  (lambda ()
    { (if (== auto_on 1) (set! auto_on 0) (set! auto_on 1))
      (set! auto_due (+ (System.timestamp) AUTO_MS))
      (set! last_pw -1)
      (set! dirty 1)
      (display "[album] auto=") (display auto_on) (newline) }))

;; 沉浸模式开关（隐藏/显示上下工具条）
(define set_chrome
  (lambda (v)
    { (set! chrome v)
      (layout_photo)
      (set! last_pw -1)
      (set! dirty 1) }))

;; 自动放映到点就下一张（go 内部会重置 auto_due）
(define step_auto
  (lambda ()
    (if (>= (System.timestamp) auto_due)
        (go 1)
        #f)))

;; 进度条像素宽：本张已停留 / AUTO_MS × 屏宽
(define progress_w
  (lambda ()
    (define el 0)
    (define w 0)
    (if (== auto_on 0) 0
        { (set! el (- (System.timestamp) (- auto_due AUTO_MS)))
          (if (< el 0) (set! el 0) #f)
          (if (> el AUTO_MS) (set! el AUTO_MS) #f)
          (set! w (Math.floor (/ (* el SW) AUTO_MS)))
          (max2 0 (min2 SW w)) })))

;; ---- 绘制：基础件 ----------------------------------------------------------

;; 「存活心跳」：标题右侧的小圆点（亮/暗两相，约 1s 交替）。
;; 除了视觉上表示 App 在跑，它还是**静态画面下的周期刷新源**——宿主回归台与
;; 浏览器回归都以「帧缓冲刷新次数」判断 App 是否还活着（纯静态页会被误判为卡死）。
;; 沉浸模式下不画（那时光标区归照片），因此也不产生心跳刷新。
(define paint_hb
  (lambda ()
    (if (== hb_on 1)
        (Gfx.fill_circle 47 9 2 0 170 255)
        (Gfx.fill_circle 47 9 2 0 66 100))
    #t))

;; 一枚底部按钮（按下态换亮色；arm_btn 命中同一按钮才高亮）
(define draw_btn
  (lambda (x y w h label id)
    (define pressed 0)
    (if (and (== t_down 1) (== arm_btn id)) (set! pressed 1) #f)
    (if (== pressed 1)
        (Gfx.fill_rect x y w h 62 100 148)
        (Gfx.fill_rect x y w h 46 55 74))
    (Gfx.draw_rect x y w h 92 104 130)
    (Gfx.draw_text label
                   (+ x (Math.floor (/ (- w (text_w label)) 2)))
                   (+ y 1)
                   226 232 242)
    #t))

;; 顶栏：标题 / 心跳点 / 当前文件名 / 原始分辨率（退出按钮由 draw_all 统一画）
(define draw_top
  (lambda ()
    (define dt "")
    (define dt_l 0)
    (define dt_x 0)
    (define name_max 210)
    (Gfx.fill_rect 0 0 SW TOP_H 30 36 46)
    (Gfx.draw_text "时光集" 6 2 150 210 255)
    (paint_hb)
    (if (== cur_ok 1)
        { (set! dt (dims_text))
          (set! dt_l (text_w dt))
          (set! dt_x (- (- (- SW 40) dt_l) 6))
          (Gfx.draw_text dt dt_x 2 118 136 158)
          (set! name_max (- (- dt_x 58) 8)) }
        #f)
    (if (> name_max 30)
        (Gfx.draw_text (clip_text cur_name name_max) 58 2 176 192 212)
        #f)
    #t))

;; 底栏：三枚按钮 + 序号 + 自动放映进度条
(define draw_bottom
  (lambda ()
    (define bt "")
    (Gfx.fill_rect 0 (- SH BAR_H) SW BAR_H 24 28 38)
    (draw_btn 6 (- SH 18) 56 BTN_H "上一张" 0)
    (if (== auto_on 1)
        (draw_btn 70 (- SH 18) 72 BTN_H "自动 开" 1)
        (draw_btn 70 (- SH 18) 72 BTN_H "自动 关" 1))
    (draw_btn 150 (- SH 18) 56 BTN_H "下一张" 2)
    (set! bt (idx_text))
    (Gfx.draw_text bt (- (- 274 (text_w bt)) 6) 222 150 165 185)
    (if (== auto_on 1) (draw_progress_bar) #f)
    #t))

;; 进度条（底栏最下面 2px；track 深、已过部分青色，颜色同 C 版 0,170,255）
(define draw_progress_bar
  (lambda ()
    (define pw (progress_w))
    (Gfx.fill_rect 0 (- SH 2) SW 2 20 24 32)
    (if (> pw 0) (Gfx.fill_rect 0 (- SH 2) pw 2 0 170 255) #f)
    (set! last_pw pw)
    #t))

;; 只重画进度条（自动放映时的逐拍刷新：脏区只有 320x2，代价可忽略）
(define paint_progress
  (lambda ()
    (define pw (progress_w))
    (if (not (== pw last_pw))
        { (Gfx.fill_rect 0 (- SH 2) SW 2 20 24 32)
          (if (> pw 0) (Gfx.fill_rect 0 (- SH 2) pw 2 0 170 255) #f)
          (set! last_pw pw)
          (Gfx.refresh) }
        #f)))

;; 坏图占位（文件头都读不出来时；仍可用上/下一张跳过）
(define draw_broken
  (lambda ()
    (define t1 "无法显示此图片")
    (define t2 "")
    (Gfx.draw_text t1 (center_x t1) 98 255 140 120)
    (set! t2 (clip_text cur_name 280))
    (Gfx.draw_text t2 (center_x t2) 118 190 190 190)
    (set! t2 "用「上一张 / 下一张」跳过")
    (Gfx.draw_text t2 (center_x t2) 148 150 160 175)
    #t))

;; 整屏重绘（照片 + 工具条）；沉浸模式下顺带撤掉宿主的退出按钮热区
(define draw_all
  (lambda ()
    (Gfx.set_font 1)
    ;; 纯黑打底：照片的黑边、工具条、坏图提示都从这块底色上重画
    (Gfx.fill_rect 0 0 SW SH 0 0 0)
    (if (== cur_ok 1)
        (if (not (Gfx.draw_image cur_path dx dy dw dh))
            (draw_broken)
            #f)
        (draw_broken))
    (if (== chrome 1)
        { (draw_top) (draw_bottom) (EX.draw_button) }
        (App.set_exit_button 0 0 0 0))
    (Gfx.refresh)
    (set! dirty 0)
    #t))

;; 空目录 / 目录尚未挂载（WASM 的 image/ 是 p2 后台层，App 可能先启动）
(define draw_empty
  (lambda ()
    (define l1 "")
    (define l2 "把 .png / .jpg 等图片放入 image/ 目录")
    (define l3 "正在自动检测…（会自动重试，按 R 立即重扫）")
    (Gfx.set_font 1)
    (Gfx.fill_rect 0 0 SW SH 14 17 22)
    (Gfx.fill_rect 0 0 SW TOP_H 30 36 46)
    (Gfx.draw_text "时光集" 6 2 150 210 255)
    (EX.draw_button)
    (Gfx.set_font 2)
    (Gfx.draw_text "时光集" (center_x "时光集") 62 235 240 245)
    (Gfx.set_font 1)
    (if (== dir_ok 1)
        (set! l1 "图片目录为空")
        (set! l1 "读不到 image/ 目录（资源可能仍在载入）"))
    (Gfx.draw_text l1 (center_x l1) 110 220 226 236)
    (Gfx.draw_text l2 (center_x l2) 132 170 182 200)
    (Gfx.draw_text l3 (center_x l3) 158 130 144 164)
    (Gfx.refresh)
    #t))

;; ---- 输入：键盘 ------------------------------------------------------------
;; 键码是 hal_key.h 的物理逻辑键值（浏览器里可打印字符 = ASCII）：
;;   '1'=49 自动；'2''4''5''7''8'=50/52/53/55/56、←=3、'0'=48 → 上一张；
;;   '3''6''9'=51/54/57、Shift=7、Ctrl=5、→=4、Enter=13 → 下一张；R=82/114 重扫。

(define is_prev_key
  (lambda (k)
    (or (== k 50) (== k 52) (== k 53) (== k 55) (== k 56) (== k 3) (== k 48))))

(define is_next_key
  (lambda (k)
    (or (== k 51) (== k 54) (== k 7) (== k 57) (== k 5) (== k 4) (== k 13))))

;; 手动重扫（R 键）：目录里换了图 / 外部删了文件时用
(define rescan_now
  (lambda ()
    { (reload)
      (if (> n 0)
          (draw_all)
          (draw_empty))
      (display "[album] rescan n=") (display n) (newline) }))

(define key_action
  (lambda (k)
    (if (> n 0)
        (if (or (== k 82) (== k 114)) (rescan_now)
            (if (== k 49) (toggle_auto)
                (if (is_prev_key k) (go -1)
                    (if (is_next_key k) (go 1) #f))))
        (if (or (== k 82) (== k 114)) (rescan_now) #f))))

;; 取空按下沿队列（一拍内可能积压多个键）
(define drain_keys
  (lambda ()
    (define k 0)
    (set! k (Key.take_press))
    (while (not (== k 0))
      { (key_action k)
        (set! k (Key.take_press)) })))

;; ---- 输入：触屏 ------------------------------------------------------------

;; 底部按钮命中（返回 0/1/2；未命中 -1）
(define hit_btn
  (lambda (px py)
    (if (== chrome 0) -1
        (if (and (>= py (- SH 18)) (< py (- SH 2)))
            (if (and (>= px 6) (< px 62)) 0
                (if (and (>= px 70) (< px 142)) 1
                    (if (and (>= px 150) (< px 206)) 2 -1)))
            -1))))

;; 触点是否落在照片区（顶栏/底栏不算；沉浸模式下就是全屏）
(define in_photo
  (lambda (px py)
    (and (>= py p_top) (< py (+ p_top p_h)))))

;; 触屏状态机：按下（武装按钮 / 记落点）→ 按住（拖出按钮即取消）→ 松手（按钮动作 /
;; 横向滑动 / 轻点）。全部用「电平 + 落点」判定，不消费 take_press/take_release，
;; 因此与 (EX.check) 的按钮判定互不干扰。
(define handle_touch
  (lambda ()
    (define p 0)
    (define x 0)
    (define y 0)
    (define ddx 0)
    (define ddy 0)
    (define b -1)
    (set! p (if (Touch.is_pressed) 1 0))
    (set! x (Touch.x))
    (set! y (Touch.y))
    ;; 按下沿
    (if (and (== p 1) (== t_down 0))
        { (set! t_x0 x)
          (set! t_y0 y)
          (set! arm_btn (hit_btn x y))
          (set! t_start_btn (if (>= arm_btn 0) 1 0))
          (set! dirty 1) }
        #f)
    ;; 按住：拖出按钮即取消武装
    (if (and (== p 1) (== t_down 1))
        { (if (>= arm_btn 0)
              (if (not (== (hit_btn x y) arm_btn))
                  { (set! arm_btn -1) (set! dirty 1) }
                  #f)
              #f) }
        #f)
    ;; 松开沿
    (if (and (== p 0) (== t_down 1))
        { (set! b (hit_btn x y))
          (if (and (>= arm_btn 0) (== b arm_btn))
              { (if (== arm_btn 0) (go -1) #f)
                (if (== arm_btn 1) (toggle_auto) #f)
                (if (== arm_btn 2) (go 1) #f) }
              (if (== t_start_btn 0)
                  { (set! ddx (- x t_x0))
                    (set! ddy (- y t_y0))
                    (if (and (> (Math.abs ddx) 36) (> (Math.abs ddx) (Math.abs ddy)))
                        ;; 横向滑动：左滑=下一张，右滑=上一张
                        (if (< ddx 0) (go 1) (go -1))
                        (if (and (< (Math.abs ddx) 14) (< (Math.abs ddy) 14))
                            ;; 轻点：在照片区内才响应（顶栏/底栏的轻点留给按钮与退出）
                            (if (in_photo t_x0 t_y0)
                                (if (== chrome 0)
                                    (set_chrome 1)
                                    (if (< t_x0 80) (go -1)
                                        (if (> t_x0 240) (go 1)
                                            (set_chrome 0))))
                                #f)
                            #f)) }
                  #f))
          (set! arm_btn -1)
          (set! t_start_btn 0)
          (set! dirty 1) }
        #f)
    (set! t_down p)
    #t))

;; ---- 主循环 ----------------------------------------------------------------

;; 前 30 秒（WASM 后台资源挂载期）每 5 秒静默重扫一次；只在目录条目数变化时重建列表
(define rescan_diff
  (lambda ()
    (define raw (Fs.list IMG_DIR))
    (define c 0)
    (set! c (if (list? raw) (length raw) -1))
    (if (not (== c dir_n))
        { (reload)
          (if (> n 0) (draw_all) (draw_empty)) }
        #f)))

(define tick
  (lambda ()
    { (if (== chrome 1)
          (EX.check)
          ;; 沉浸模式：不跑 EX.check（否则它按自己的矩形把右上角点按判成退出）；
          ;; 硬 A/ESC 在这里只当「唤出工具条」，退出按钮热区已撤掉。
          { (if (App.back_requested) (set_chrome 1) #f)
            (if (App.resumed) (set! dirty 1) #f) })
      (drain_keys)
      (if (> n 0) (handle_touch) #f)
      (if (and (> n 0) (== auto_on 1)) (step_auto) #f)
      (if (> n 0)
          (if (== dirty 1)
              (draw_all)
              (if (and (== auto_on 1) (== chrome 1)) (paint_progress) #f))
          { (set! empty_ctr (+ empty_ctr 1))
            ;; 前 30 秒 0.5s 一次（WASM 的 p2 资源分批挂载），之后降到 2s 一次：
            ;; 空目录页可能长期停留（用户还没放图），不必每半秒读一次目录。
            (if (>= empty_ctr (if (< (- (System.timestamp) boot_ms) 30000) 15 60))
                { (set! empty_ctr 0)
                  (reload)
                  (if (> n 0) (draw_all) (draw_empty)) }
                #f) })
      ;; 后台挂载期静默重扫
      (if (and (< (- (System.timestamp) boot_ms) 30000) (> n 0))
          { (set! rescan_ctr (+ rescan_ctr 1))
            (if (>= rescan_ctr 150)
                { (set! rescan_ctr 0) (rescan_diff) }
                #f) }
          #f)
      ;; 心跳（约 1s 一次）：只在「静态查看器」状态下需要（自动放映/空目录页
      ;; 本来就在周期刷新）；画完呼吸点后立即推帧。
      (set! hb (+ hb 1))
      (if (>= hb 30)
          { (set! hb 0)
            (if (== hb_on 1) (set! hb_on 0) (set! hb_on 1))
            (if (and (== chrome 1) (== auto_on 0) (== dirty 0) (> n 0))
                { (paint_hb) (Gfx.refresh) }
                #f) }
          #f)
      ;; 诊断行（≈2s 一条；e2e 与现场排查用）
      (set! stat_ctr (+ stat_ctr 1))
      (if (>= stat_ctr 60)
          { (set! stat_ctr 0)
            (display "[album] n=") (display n)
            (display " idx=") (display idx)
            (display " ok=") (display cur_ok)
            (display " auto=") (display auto_on)
            (display " chrome=") (display chrome)
            (display " dims=") (display img_w) (display "x") (display img_h)
            (newline) }
          #f) }))

;; ---- 启动 ------------------------------------------------------------------

(define cleanup
  (lambda ()
    { (System.clear_timeout ts)
      (display "[album] bye") (newline) }))

(define boot
  (lambda ()
    { (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (Gfx.set_font 1)
      (set! boot_ms (System.timestamp))
      (EX.install_with draw_all)
      (EX.on_exit cleanup)
      (reload)
      (if (> n 0)
          (draw_all)
          (draw_empty))
      (set! ts (System.set_interval TICK_MS tick))
      (display "[album] boot n=") (display n)
      (display " dir_ok=") (display dir_ok) (newline) }))

(boot)
