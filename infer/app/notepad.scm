;; ============================================================================
;; 便签（Notepad）—— 灵机应用程序 · Key + Fs 两个本地库的用法示范兼验收程序
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx / Key / Fs / Touch 本地库
;;
;; 这个 App 同时是两件事：
;;   1) 一个真能用的便签：打字 → 自动保存到 data/note.txt → 下次进来接着写；
;;   2) 新增两个输入/存储本地库的端到端验收程序：
;;      · (native Key) —— 键盘边沿事件。可打印字符就是 ASCII 码，直接喂给
;;        String.fromCharCode 就能上屏；退格/回车同样按 ASCII 处理（8 / 13）。
;;      · (native Fs)  —— 文本文件整读整写。(Fs.read)/(Fs.write) 走相对业务资源根
;;        的路径 "data/note.txt"：桌面/WASM 落在 /home/bd4sur/linglong/data/ 下，
;;        ESP32 落在 SD 卡根；浏览器里写入还会同步进 IndexedDB（刷新页面仍在）。
;;
;; 交互：
;;   · 实体键盘 / 浏览器物理键盘 —— 打字；Delete（127）删字；Enter 换行；
;;     （浏览器里 Backspace 被宿主映射为"返回键"，故删字统一用 Delete；
;;       终端/掌机键盘上 Backspace 的 ASCII 8 也照样能删）
;;   · 触屏点一下 —— 立即保存（并显示当前字节数）；
;;   · 停止输入约 0.5 秒 —— 自动保存；
;;   · Esc（硬键）—— 由 app/lib/exit.scm 接管：弹全局「是否退出？」确认框；确认后
;;     **先落盘再退出**（(EX.on_exit do_save)：收尾函数补上"自动保存还差半秒"这一手），
;;     留下则继续写；触屏不再派生返回键。
;;
;; 解释器约束（manual.md，逐条遵守）：
;;   · 算术恒二参数；无一元取负；/ 与 mod 恒返回浮点（下标一律 Math.floor）；
;;   · 无 list/let/cond 多表达式；define 无过程简写；块内不 define；
;;   · '() 字面量是模块级静态槽——空表一律用 (cdr '(0)) 现场造新的（见 new_list）。
;; ============================================================================

(native Gfx)
(native Key)
(native Fs)
(native Touch)
(native String)
(native Math)
(native System)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 常量 ----------------------------------------------------------------

(define NOTE_PATH "data/note.txt")
(define TOP_H 18)      ; 顶栏高度
(define BOT_H 16)      ; 底栏高度
(define BODY_Y 20)     ; 正文起始 y
(define LINE_H 14)     ; 行高（12px 字模 + 2px 行距）
(define PAD_X 4)       ; 左右留白
(define CJK_W 12)      ; 全角字符步进
(define ASCII_W 7)     ; 半角字符步进
(define SAVE_IDLE 8)   ; 静止多少拍后自动保存（定时器约 60~80ms/拍）

;; ---- 状态 ----------------------------------------------------------------

(define SW 320)
(define SH 240)

(define text "")           ; 便签全文（字符串不可变，编辑即重建）
(define lines (cdr '(0)))  ; 折行后的各行（layout_dirty 时重建）
(define dirty 0)           ; 有未保存的修改
(define layout_dirty 1)    ; 需要重新折行
(define idle_ticks 0)      ; 距上次编辑经过的拍数
(define status "就绪")
(define kb_ok 0)           ; 键盘事件源是否可用
(define ts 0)              ; 定时器编号
(define TEXT_MAX 2000)     ; 便签长度上限（折行是 O(n) 的，长文请用电子书 App）

;; ---- 热路径临时量（模块级：解释器对块内 define / 嵌套区域有布局敏感问题）----

(define i 0)
(define n 0)
(define cp 0)
(define cw 0)
(define loaded "")
(define tmp_s "")
(define wrap_cur "")
(define wrap_w 0)

;; 造一个全新的空表（'() 是模块级静态槽，反复 push 会跨调用累积）
(define new_list (lambda () (cdr '(0))))

;; 字符步进：CJK/全角 12px，ASCII 7px（与 gfx 12px 字模的实测步进一致）
(define char_w
  (lambda (code)
    (if (< code 11904) ASCII_W CJK_W)))   ; 11904 = 0x2E80（CJK 部首补充的起点）

;; ---- 折行 ----------------------------------------------------------------

(define relayout
  (lambda ()
    (begin
      (set! lines (new_list))
      (set! wrap_cur "")
      (set! wrap_w 0)
      (set! i 0)
      (set! n (String.length text))
      (while (< i n)
        (begin
          (set! cp (String.charCodeAt i text))
          (if (== cp 10)
              (begin  ; 换行符：本行结束
                (push lines wrap_cur)
                (set! wrap_cur "")
                (set! wrap_w 0))
              (begin
                (set! cw (char_w cp))
                (if (> (+ wrap_w cw) (- SW (* 2 PAD_X)))
                    (begin  ; 本行放不下：先落行再起新行
                      (push lines wrap_cur)
                      (set! wrap_cur "")
                      (set! wrap_w 0)))
                (set! wrap_cur (String.concat wrap_cur (String.charAt text i)))
                (set! wrap_w (+ wrap_w cw))))
          (set! i (+ i 1))))
      (push lines wrap_cur)
      (set! layout_dirty 0))))

;; 某一行的像素宽度（画光标用）
(define line_width
  (lambda (s)
    (begin
      (set! i 0)
      (set! n (String.length s))
      (set! wrap_w 0)
      (while (< i n)
        (begin
          (set! wrap_w (+ wrap_w (char_w (String.charCodeAt i s))))
          (set! i (+ i 1))))
      wrap_w)))

;; ---- 编辑 ----------------------------------------------------------------

(define touch_edit
  (lambda ()
    (begin
      (set! dirty 1)
      (set! layout_dirty 1)
      (set! idle_ticks 0))))

(define insert_ch
  (lambda (code)
    (if (< (String.length text) TEXT_MAX)
        (begin
          (set! text (String.concat text (String.fromCharCode code)))
          (touch_edit)))))

(define backspace
  (lambda ()
    (if (> (String.length text) 0)
        (begin
          (set! text (String.slice text 0 (- (String.length text) 1)))
          (touch_edit)))))

;; 一个按下沿 → 一次编辑（8/127=退格、13=回车、9=制表符当空格、其余可打印字符上屏）
;; 注：浏览器页面的 Backspace 在宿主 KEY_MAP 里等于"返回键"（Esc，平台惯例），
;;     会被宿主吃掉，所以浏览器里请用 Delete（127）删字；终端/掌机键盘上 8 直接生效。
(define apply_key
  (lambda (code)
    (if (or (== code 8) (== code 127))
        (backspace)
        (if (== code 13)
            (insert_ch 10)
            (if (== code 9)
                (insert_ch 32)
                (if (and (>= code 32) (<= code 126))
                    (insert_ch code)))))))

;; 取空队列里的全部按下沿（一次 tick 内可能积压多个键）
(define drain_keys
  (lambda ()
    (begin
      (set! cp (Key.take_press))
      (while (not (== cp 0))
        (begin
          (apply_key cp)
          (set! cp (Key.take_press)))))))

;; ---- 存取 ----------------------------------------------------------------

;; 状态行文案："已保存 12 B" / "已载入 12 B"（拼装先落变量，避开深嵌套）
(define status_note
  (lambda (prefix)
    (begin
      (set! tmp_s (String.atom_to_string (Fs.size NOTE_PATH)))
      (set! tmp_s (String.concat prefix tmp_s))
      (set! status (String.concat tmp_s " B")))))

(define do_save
  (lambda ()
    (begin
      (if (Fs.write NOTE_PATH text)
          (status_note "已保存 ")
          (set! status "保存失败：data/ 不可写"))
      (set! dirty 0)
      (set! idle_ticks 0))))

(define do_load
  (lambda ()
    (begin
      (set! loaded (Fs.read NOTE_PATH))
      (if (null? loaded)
          (begin
            (set! text "")
            (set! status "新便签（打字即自动保存）"))
          (begin
            (set! text loaded)
            (status_note "已载入 ")))
      (set! layout_dirty 1))))

;; ---- 绘制 ----------------------------------------------------------------

(define draw_all
  (lambda ()
    (begin
      ;; 背景与顶栏
      (Gfx.fill_rect 0 0 SW SH 14 17 22)
      (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
      (Gfx.draw_text "便签" PAD_X 3 235 240 245)
      (Gfx.draw_text status 150 3 150 200 235)
      ;; 正文（只显示最后若干行，写过一屏自动上滚）
      (set! n (Math.floor (/ (- (- SH BOT_H) BODY_Y) LINE_H)))
      (set! cp (- (length lines) n))
      (if (< cp 0) (set! cp 0))
      (set! i cp)
      (while (< i (length lines))
        (begin
          (Gfx.draw_text (get_item lines i) PAD_X (+ BODY_Y (* (- i cp) LINE_H)) 210 218 232)
          (set! i (+ i 1))))
      ;; 光标：最后一行（cw）末尾的一根竖条。注意 y 要按"可见行序号"（cw-cp）算：
      ;; 最后一行永远是最后一条可见行，无论是否发生过上滚。
      (set! cw (- (length lines) 1))
      (if (< cw 0) (set! cw 0))
      (Gfx.fill_rect (+ PAD_X (line_width (get_item lines cw)))
                     (+ (+ BODY_Y (* (- cw cp) LINE_H)) 2)
                     2 10 120 200 240)
      ;; 底栏
      (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
      (if (== kb_ok 1)
          (Gfx.draw_text "键盘输入 · 触屏=保存 · Esc 返回" PAD_X (- SH 14) 130 140 155)
          (Gfx.draw_text "无键盘事件源（本平台仅触屏）" PAD_X (- SH 14) 200 150 120))
      (Gfx.refresh))))

;; ---- 主循环 --------------------------------------------------------------

(define on_tick
  (lambda ()
    (begin
      (set! idle_ticks (+ idle_ticks 1))
      (drain_keys)
      (if (Touch.take_press) (do_save))
      (if (and (not (== dirty 0)) (>= idle_ticks SAVE_IDLE)) (do_save))
      (if (== layout_dirty 1) (relayout))
      (draw_all))))

;; ---- 启动 ----------------------------------------------------------------

(define boot
  (lambda ()
    (begin
      (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (set! kb_ok ((lambda (b) (if b 1 0)) (Key.is_available)))
      (do_load)
      (relayout)
      (draw_all)
      (set! ts (System.set_interval 60 on_tick)))))

(boot)

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 先落盘（on_exit 登记的收尾函数）→ 再销毁实例；留下 → 继续写（并重绘擦残影）。
;;  自动保存有 ~0.5s 的静止延迟，刚打完字立刻退出就可能还没落盘——收尾函数补上这一手。
(EX.on_exit do_save)
(EX.install_with draw_all)
