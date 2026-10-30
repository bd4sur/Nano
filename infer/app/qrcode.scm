;; ============================================================================
;; 二维码（QR Code）—— 灵机应用程序 · Ui + Gfx + Fs 本地库
;;
;; 运行环境：灵机（Animac）解释器 + Gfx / Key / Touch / Ui / Fs / System 本地库
;;           + 纯逻辑模块 app/lib/qr_core.scm（ISO/IEC 18004 字节模式编码器）
;;
;; 做什么：
;;   1) 借宿主的文本输入控件（(Ui.input ...)，鹦鹉笼同款：全键盘拼音 / 16 键九宫格 /
;;      触屏软键盘）取一段文本，编码成二维码并画在屏幕上；
;;   2) 点「保存」把二维码写成**字符画纯文本**（黑点 = U+25CF「●」，白点 = 全角空格
;;      U+3000），文件落在业务资源根的 data/qrcode.txt：
;;        桌面：/home/bd4sur/linglong/data/qrcode.txt
;;        浏览器：MEMFS 同路径，且经 am_native_fs_on_write 同步进 IndexedDB（刷新不丢）
;;        掌机：SD 卡根 /data/qrcode.txt
;;      字符画保留 4 模块静区（标准要求），因此文件本身就是一张可直接扫描的二维码，
;;      也可以用等宽字体打印/贴进聊天窗口。
;;
;; 交互：
;;   · 「输入」按钮 / Enter —— 弹宿主输入框，提交后即生成（自动选最小版本，默认纠错 M）；
;;   · 「保存」按钮        —— 落盘字符画并回报字节数；
;;   · 「纠错 M/…」按钮    —— 纠错等级 L→M→Q→H→L 循环，按新等级重新生成；若新等级
;;                            放不下当前文本，则回退到原等级（面板显示与屏幕上的码
;;                            不脱节）并在状态条说明；
;;   · 还没有二维码时点一下画面 —— 等同「输入」；
;;   · Esc（或触屏右上角宫格）—— 宿主返回应用列表（宿主行为，App 不拦截）。
;;
;; 节奏：编码在点击后同步进行（8 种掩码 × 罚分择优），期间只上屏一次"正在生成二维码…"
;; 静态提示，进程忙完才回到交互态。wasm 实测纯编码耗时：v1≈40ms、v6≈130ms、
;; v11≈300ms、v17≈570ms；提交→二维码上屏（含两次推帧）约 180~200ms（v1）。
;; 忙等期间**这几次触屏点按会被丢掉**——(native Touch) 是电平差分的，进程没在采样
;; 就看不出按下沿（见 am_native_Touch.h），所以大文本时请等画面出来再操作。
;;
;; 浏览器补充：模态框里若已呼出触屏软键盘，宿主的实体键会被软键盘电平覆盖
;; （既有行为），此时用软键盘右上角的 ENT 键提交即可；软键盘没开时实体 Enter 正常提交。
;;
;; 布局（320x240，见下方常量）：左上 232x208 是二维码（白底，4 模块静区，逐行合并同色
;; 矩形绘制）；右侧 88px 是信息与按钮面板；底部 16px 是状态条（显示保存路径与字节数）。
;;
;; 解释器约束（manual.md，逐条遵守）：
;;   · 算术/比较恒二参数；**没有一元取负**（写 (- 0 x)）；/ 与 mod 恒返回浮点
;;     （下标、像素坐标一律 Math.floor）；没有 min/max；
;;   · 没有 list/append/map；空表用 (cdr '(0)) 现场造；
;;   · define 无过程简写；块内不 define；`set!` 的目标必须已在模块级 define；
;;   · ⚠ 模块级临时量按"谁写谁读"分配（前缀 = 所属函数），跨函数调用链绝不共用
;;     （否则内层函数会悄悄覆盖外层循环变量）。
;; ============================================================================

(native Gfx)
(native Key)
(native Touch)
(native Ui)
(native Fs)
(native String)
(native Math)
(native System)

(import QR "app/lib/qr_core.scm")

;; ---- 布局常量 ------------------------------------------------------------

(define SW 320)
(define SH 240)
(define HDR_H 16)          ; 顶栏高
(define BOT_H 16)          ; 底栏高
(define BOT_Y 224)         ; 底栏起始 y
(define QR_W 232)          ; 二维码区宽（同时也是右侧面板的起点 x）
(define PANEL_X 232)
(define PAD 4)

;; 三个按钮（右侧面板内，竖向排列）
(define BTN_X 236)
(define BTN_W 80)
(define BTN_H 26)
(define BTN1_Y 68)         ; 输入
(define BTN2_Y 100)        ; 保存
(define BTN3_Y 132)        ; 纠错

;; 配色
(define C_BG_R 16)  (define C_BG_G 18)  (define C_BG_B 24)
(define C_HDR_R 28) (define C_HDR_G 34) (define C_HDR_B 46)
(define C_PANEL_R 22) (define C_PANEL_G 26) (define C_PANEL_B 34)
(define C_TXT_R 232) (define C_TXT_G 238) (define C_TXT_B 246)
(define C_MUT_R 132) (define C_MUT_G 142) (define C_MUT_B 160)
(define C_ACC_R 120) (define C_ACC_G 200) (define C_ACC_B 255)
(define C_SAVE_R 40) (define C_SAVE_G 130) (define C_SAVE_B 90)

;; ---- 数据路径 ------------------------------------------------------------

(define QR_PATH "data/qrcode.txt")
(define BLACK "●")         ; U+25CF 黑圆点
(define WHITE "　")        ; U+3000 全角空格（与黑点等宽，字符画才是正方形）

;; ---- 状态 ----------------------------------------------------------------

(define a_text "")         ; 当前文本
(define a_ec 1)            ; 纠错等级索引 0=L 1=M 2=Q 3=H
(define a_has 0)           ; 是否已有二维码
(define a_ver 0)           ; 版本
(define a_n 0)             ; 边长模块数
(define a_mask 0)          ; 掩码
(define a_nbytes 0)        ; 文本 UTF-8 字节数
(define a_mat (cdr '(0)))  ; 扁平模块矩阵
(define a_status "点「输入」或按 Enter 开始")
(define a_ts 0)            ; 定时器编号
(define a_scale 1)         ; 每模块像素
(define a_qx 0)            ; 二维码白底左上角
(define a_qy 0)
(define a_qw 0)

;; ---- 模块级临时量（前缀 = 归属函数）--------------------------------------

;; tw（字符宽度测量）
(define t_i 0)
(define t_n 0)
(define t_cp 0)
(define t_w 0)
;; draw_header
(define h_s "")
;; draw_qr
(define d_m 0)
(define d_avail 0)
(define d_r 0)
(define d_c 0)
(define d_c2 0)
;; draw_panel
(define p_s "")
;; draw_bottom
(define b_s "")
;; qart（字符画）
(define qa_r 0)
(define qa_c 0)
(define qa_line "")
(define qa_out "")
;; generate
(define g_rc 0)
;; do_input
(define i_txt "")
(define e_prev 0)
(define e_bad "")
;; on_tick
(define t_k 0)
(define t_tx 0)
(define t_ty 0)
;; do_save
(define s_txt "")
(define s_ok #f)

;; ---- 小工具 --------------------------------------------------------------

;; 字符串像素宽度：CJK/全角 12px，ASCII 7px（12px 字模的实测步进）
(define tw
  (lambda (s)
    (begin
      (set! t_w 0)
      (set! t_i 0)
      (set! t_n (String.length s))
      (while (< t_i t_n)
        (begin
          (set! t_cp (String.charCodeAt t_i s))
          (if (< t_cp 11904)      ; 0x2E80 起为 CJK
              (set! t_w (+ t_w 7))
              (set! t_w (+ t_w 12)))
          (set! t_i (+ t_i 1))))
      t_w)))

;; 右对齐绘制
(define draw_text_right
  (lambda (s x_right y r g b)
    (Gfx.draw_text s (- x_right (tw s)) y r g b)))

;; "标签 值" 一行：标签用暗色，值用亮色
(define draw_kv
  (lambda (label value y)
    (begin
      (Gfx.draw_text label BTN_X y C_MUT_R C_MUT_G C_MUT_B)
      (Gfx.draw_text value (+ (+ BTN_X (tw label)) 4) y C_TXT_R C_TXT_G C_TXT_B))))

;; 一个按钮（含边框与居中文字）
(define draw_button
  (lambda (label y r g b)
    (begin
      (Gfx.fill_rect BTN_X y BTN_W BTN_H r g b)
      (Gfx.draw_rect BTN_X y BTN_W BTN_H 96 108 132)
      (Gfx.draw_text label (+ BTN_X (Math.floor (/ (- BTN_W (tw label)) 2)))
                     (+ y (Math.floor (/ (- BTN_H 12) 2)))
                     C_TXT_R C_TXT_G C_TXT_B))))

;; ---- 绘制：顶栏 ----------------------------------------------------------

(define draw_header
  (lambda ()
    (begin
      (Gfx.fill_rect 0 0 SW HDR_H C_HDR_R C_HDR_G C_HDR_B)
      (Gfx.draw_text "二维码" PAD 2 C_TXT_R C_TXT_G C_TXT_B)
      (if (== a_has 1)
          (set! h_s (String.concat "v" (String.concat (String.atom_to_string a_ver)
                        (String.concat " · " (QR.ec_name a_ec)))))
          (set! h_s "QR"))
      (draw_text_right h_s (- SW PAD) 2 C_ACC_R C_ACC_G C_ACC_B))))

;; ---- 绘制：二维码区 ------------------------------------------------------

(define draw_qr
  (lambda ()
    (begin
      (if (== a_has 1)
          (begin
            ;; 边长 = 模块数 + 两侧各 4 格静区
            (set! d_m (+ a_n 8))
            ;; 可用区域：宽 QR_W、高 BOT_Y-HDR_H；取二者较小者整除
            (set! d_avail (- BOT_Y HDR_H))
            (if (< QR_W d_avail) (set! d_avail QR_W))
            (set! a_scale (Math.floor (/ d_avail d_m)))
            (if (< a_scale 1) (set! a_scale 1))
            (set! a_qw (* a_scale d_m))
            (set! a_qx (Math.floor (/ (- QR_W a_qw) 2)))
            (set! a_qy (+ HDR_H (Math.floor (/ (- (- BOT_Y HDR_H) a_qw) 2))))
            ;; 白底（含静区）
            (Gfx.fill_rect a_qx a_qy a_qw a_qw 255 255 255)
            ;; 逐行绘制：把连续黑格合并成一次 fill_rect（v20 也只有约千次调用）
            (set! d_r 0)
            (while (< d_r a_n)
              (begin
                (set! d_c 0)
                (while (< d_c a_n)
                  (begin
                    (if (== (get_item a_mat (+ (* d_r a_n) d_c)) 1)
                        (begin
                          (set! d_c2 d_c)
                          (while (and (< d_c2 a_n)
                                      (== (get_item a_mat (+ (* d_r a_n) d_c2)) 1))
                            (set! d_c2 (+ d_c2 1)))
                          (Gfx.fill_rect (+ a_qx (* (+ d_c 4) a_scale))
                                         (+ a_qy (* (+ d_r 4) a_scale))
                                         (* (- d_c2 d_c) a_scale) a_scale 0 0 0)
                          (set! d_c d_c2))
                        (set! d_c (+ d_c 1)))))
                (set! d_r (+ d_r 1)))))
          (begin
            ;; 还没有二维码：给一个居中提示
            (Gfx.draw_text "点画面任意处、或按 Enter" 12 108 C_MUT_R C_MUT_G C_MUT_B)
            (Gfx.draw_text "输入文本即可生成二维码" 12 126 C_MUT_R C_MUT_G C_MUT_B))))))

;; ---- 绘制：右侧面板 ------------------------------------------------------

(define draw_panel
  (lambda ()
    (begin
      (Gfx.fill_rect PANEL_X HDR_H (- SW PANEL_X) (- BOT_Y HDR_H)
                     C_PANEL_R C_PANEL_G C_PANEL_B)
      (Gfx.fill_rect PANEL_X HDR_H 1 (- BOT_Y HDR_H) 44 52 68)
      ;; 规格信息
      (if (== a_has 1)
          (begin
            (draw_kv "字节" (String.atom_to_string a_nbytes) 22)
            (draw_kv "上限" (String.atom_to_string (QR.ec_capacity a_ec)) 36)
            (draw_kv "掩码" (String.atom_to_string a_mask) 50))
          (begin
            (draw_kv "文本" (String.atom_to_string (String.length a_text)) 22)
            (draw_kv "纠错" (QR.ec_name a_ec) 36)))
      ;; 按钮
      (draw_button "输入" BTN1_Y 44 56 76)
      (draw_button "保存" BTN2_Y C_SAVE_R C_SAVE_G C_SAVE_B)
      (set! p_s (String.concat "纠错 " (QR.ec_name a_ec)))
      (draw_button p_s BTN3_Y 44 56 76)
      ;; 底部小提示
      (Gfx.draw_text "Enter=输入" BTN_X 168 C_MUT_R C_MUT_G C_MUT_B)
      (Gfx.draw_text "A=返回列表" BTN_X 182 C_MUT_R C_MUT_G C_MUT_B))))

;; ---- 绘制：底栏（状态） --------------------------------------------------

(define draw_bottom
  (lambda ()
    (begin
      (Gfx.fill_rect 0 BOT_Y SW BOT_H 28 34 46)
      (set! b_s a_status)
      (if (> (tw b_s) (- SW (* 2 PAD)))
          (begin
            (set! b_s (String.concat (String.slice b_s 0 40) "…"))))
      (Gfx.draw_text b_s PAD (+ BOT_Y 2) 168 178 196))))

;; ---- 整屏重绘 ------------------------------------------------------------

(define draw_all
  (lambda ()
    (begin
      (Gfx.fill_rect 0 0 SW SH C_BG_R C_BG_G C_BG_B)
      (draw_header)
      (draw_qr)
      (draw_panel)
      (draw_bottom)
      (Gfx.refresh))))

;; ---- 字符画文本 ----------------------------------------------------------

(define qart
  (lambda ()
    (begin
      (set! qa_out "")
      (set! qa_r 0)
      (while (< qa_r (+ a_n 8))
        (begin
          (set! qa_line "")
          (set! qa_c (- 0 4))
          (while (< qa_c (+ a_n 4))
            (begin
              (if (and (>= qa_r 4) (< qa_r (+ a_n 4)) (>= qa_c 0) (< qa_c a_n))
                  (begin
                    (if (== (get_item a_mat (+ (* (- qa_r 4) a_n) qa_c)) 1)
                        (set! qa_line (String.concat qa_line BLACK))
                        (set! qa_line (String.concat qa_line WHITE))))
                  (set! qa_line (String.concat qa_line WHITE)))
              (set! qa_c (+ qa_c 1))))
          (set! qa_out (String.concat qa_out (String.concat qa_line "\n")))
          (set! qa_r (+ qa_r 1))))
      qa_out)))

;; ---- 生成 ----------------------------------------------------------------

(define generate
  (lambda ()
    (begin
      (if (== (String.length a_text) 0)
          (begin
            (set! a_status "文本为空，未生成")
            (set! g_rc 2)
            (draw_all))
          (begin
            ;; 编码要跨若干帧（wasm 实测：v1 约 40ms、v6 约 130ms、v11 约 300ms、
            ;; v17 约 570ms —— 8 种掩码的最小罚分搜索占大头），先上屏"正在生成"
            (set! a_status "正在生成二维码…")
            (draw_all)
            (set! g_rc (QR.encode a_text a_ec))
            (if (== g_rc 0)
                (begin
                  (set! a_has 1)
                  (set! a_ver (QR.get_version))
                  (set! a_n (QR.get_size))
                  (set! a_mask (QR.get_mask))
                  (set! a_nbytes (QR.get_nbytes))
                  (set! a_mat (QR.get_matrix))
                  (set! a_status (String.concat (String.concat "已生成 v"
                             (String.atom_to_string a_ver))
                             (String.concat " · " (String.concat
                               (String.atom_to_string a_nbytes) " 字节")))))
                (begin
                  ;; 1 = 超过版本 20 的容量；此时保留原二维码不动
                  (set! a_status (String.concat (String.concat "文本过长（上限 "
                             (String.atom_to_string (QR.ec_capacity a_ec))) " 字节）"))))
            (draw_all)
            g_rc)))))

;; ---- 交互动作 ------------------------------------------------------------

;; 弹宿主输入框（阻塞式：进程挂起，宿主的输入法与软键盘全部可用）；
;; 用户提交的文本直接作为新文本；取消（长按 A / 点按页眉返回）时保留原二维码。
(define do_input
  (lambda ()
    (begin
      (set! i_txt (Ui.input "输入要生成二维码的文本"))
      (if (null? i_txt)
          (begin
            (set! a_status "已取消输入")
            (draw_all))
          (begin
            (set! a_text i_txt)
            (generate))))))

(define do_save
  (lambda ()
    (begin
      (if (== a_has 0)
          (begin
            (set! a_status "还没有二维码可保存")
            (draw_all))
          (begin
            (set! a_status "正在保存字符画…")
            (draw_all)
            (set! s_txt (qart))
            (set! s_ok (Fs.write QR_PATH s_txt))
            (if s_ok
                (set! a_status (String.concat (String.concat "已保存 " QR_PATH) 
                               (String.concat " · "
                                 (String.concat (String.atom_to_string (Fs.size QR_PATH)) " 字节"))))
                (set! a_status (String.concat "保存失败：" QR_PATH)))
            (draw_all))))))

;; 循环切换纠错等级 L→M→Q→H→L。已有二维码时立即按新等级重新生成；
;; 若新等级放不下当前文本（rc=1），回退到原等级并说明——否则面板显示 F 而屏幕上是
;; 上一等级的码，容易误读。
(define do_ec
  (lambda ()
    (begin
      (set! e_prev a_ec)
      (set! a_ec (+ a_ec 1))
      (if (> a_ec 3) (set! a_ec 0))
      (if (== a_has 1)
          (begin
            (if (not (== (generate) 0))
                (begin
                  (set! e_bad (QR.ec_name a_ec))
                  (set! a_ec e_prev)
                  (set! a_status (String.concat (String.concat "文本放不下（纠错 "
                             e_bad) "），已保持原等级"))
                  (draw_all))))
          (begin
            (set! a_status (String.concat "纠错等级 " (QR.ec_name a_ec)))
            (draw_all))))))

;; ---- 触屏命中判定 --------------------------------------------------------

(define in_btn
  (lambda (x y by)
    (and (>= x BTN_X) (< x (+ BTN_X BTN_W)) (>= y by) (< y (+ by BTN_H)))))

;; ---- 节拍 ----------------------------------------------------------------

(define on_tick
  (lambda ()
    (begin
      ;; 键盘：Enter = 输入（框内提交用，框外触发新一轮）
      (set! t_k (Key.take_press))
      (if (== t_k 13)
          (do_input)
          (begin
            (if (Touch.take_press)
                (begin
                  (set! t_tx (Touch.x))
                  (set! t_ty (Touch.y))
                  (if (in_btn t_tx t_ty BTN1_Y)
                      (do_input)
                      (if (in_btn t_tx t_ty BTN2_Y)
                          (do_save)
                          (if (in_btn t_tx t_ty BTN3_Y)
                              (do_ec)
                              ;; 还没有二维码时，点画面任意处 = 输入
                              (if (and (== a_has 0) (< t_tx QR_W))
                                  (do_input))))))
                0))))))

;; ---- 启动 ----------------------------------------------------------------

(define boot
  (lambda ()
    (begin
      (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (draw_all)
      (set! a_ts (System.set_interval 60 on_tick)))))

(boot)