;; ============================================================================
;; 输入框演示（Ui.input / Ui.message）—— 灵机应用程序 · 宿主模态文本输入/输出示范兼验收
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx/Key/Touch/Fs/System 本地库 + **Ui 本地库**
;;
;; 这个 App 演示「灵机 App 直接借用宿主的文本输入控件」——就是鹦鹉笼（本地 LLM）用的
;; 那套（全键盘拼音 + 16 键九宫格输入法 + 触屏软键盘 + 候选字翻页 + 光标/滚动）。
;; Scheme 侧一行输入法代码都不用写：
;;
;;     (Ui.input 标题)            -> 弹出输入框，**阻塞**，返回用户提交的文本；取消返回 #null
;;     (Ui.input_with 标题 初值)  -> 同上，输入框预置初值
;;     (Ui.message 正文)          -> 弹出只读文本视图，阻塞到用户关闭
;;
;; 所谓"阻塞"是挂起 Scheme 进程（虚拟机不再调度它），宿主的事件循环与界面照常运转，
;; 因此界面始终可交互；用户提交后进程从同一条指令继续执行。
;;
;; 交互：
;;   ① 说明画面：Enter 或点一下屏幕开始；
;;   ② 输入名字（Ui.input）；
;;   ③ 输入备注（Ui.input_with，初值=刚才的名字，可直接回车接受）；
;;   ④ 结果写进 data/ui_demo.txt，弹一个只读消息框汇报，停在结果画面；
;;   ⑤ 结果画面上再按 Enter/点一下 → 回到说明画面，可以重来。
;;
;; 输入框内的按键（与鹦鹉笼一致）：
;;   · 16 键九宫格：2-9 组字 → Enter 进选字 → 数字键选字；Esc 退格；
;;   · 触屏：页脚左侧「键盘」呼出软键盘（全键盘拼音），右侧「16键」呼出九宫格；
;;   · 提交 = 短按 Enter；取消 = **长按 A/Esc** 或点按页眉右侧「返回」热区。
;;
;; 解释器约束（manual.md）：算术恒二参数、无一元取负、/ 与 mod 恒浮点、无 list/let、
;; define 无过程简写、块内不 define、'() 是模块级静态槽（空表用 (cdr '(0)) 现造）。
;; ============================================================================

(native Gfx)
(native Key)
(native Touch)
(native String)
(native Fs)
(native System)
(native Ui)
(native Math)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 常量与状态 ----------------------------------------------------------

(define DATA_PATH "data/ui_demo.txt")
(define SW 320)
(define SH 240)
(define PAD_X 6)

(define stage 0)        ; 0=说明画面（等待开始） 1=结果画面
(define name_txt "")    ; 名字（或 "#null"）
(define note_txt "")    ; 备注（或 "#null"）
(define status "就绪")
(define saved_bytes 0)
(define tick 0)
(define ts 0)

;; ---- 绘制 ----------------------------------------------------------------

(define draw_bar
  (lambda (text)
    (begin
      (Gfx.fill_rect 0 0 SW 18 30 38 52)
      (Gfx.draw_text text PAD_X 3 235 240 245))))

(define draw_hint
  (lambda (text)
    (begin
      (Gfx.fill_rect 0 (- SH 16) SW 16 24 28 38)
      (Gfx.draw_text text PAD_X (- SH 14) 140 152 168))))

(define draw_idle
  (lambda ()
    (begin
      (Gfx.fill_rect 0 0 SW SH 14 17 22)
      (draw_bar "输入框演示 · Ui")
      (Gfx.draw_text "灵机 App 直接借用宿主的" PAD_X 34 200 210 225)
      (Gfx.draw_text "文本输入控件（与鹦鹉笼同款）" PAD_X 50 200 210 225)
      (Gfx.draw_text "· 全键盘拼音 / 16 键九宫格" PAD_X 74 150 200 235)
      (Gfx.draw_text "· 触屏软键盘 / 候选字翻页" PAD_X 90 150 200 235)
      (Gfx.draw_text "· 提交=Enter  取消=长按 A" PAD_X 106 150 200 235)
      (Gfx.draw_text "① Ui.input      取一行文本" PAD_X 132 210 190 120)
      (Gfx.draw_text "② Ui.input_with 带初值" PAD_X 148 210 190 120)
      (Gfx.draw_text "③ Ui.message    只读文本" PAD_X 164 210 190 120)
      (Gfx.draw_text (String.concat "等待开始 #" (String.atom_to_string tick)) PAD_X 190 90 100 115)
      (draw_hint "按 Enter 或点一下屏幕开始")
      (Gfx.refresh))))

(define draw_result
  (lambda ()
    (begin
      (Gfx.fill_rect 0 0 SW SH 14 17 22)
      (draw_bar "输入框演示 · 结果")
      (Gfx.draw_text "名字：" PAD_X 34 150 200 235)
      (Gfx.draw_text name_txt (+ PAD_X 48) 34 235 235 235)
      (Gfx.draw_text "备注：" PAD_X 52 150 200 235)
      (Gfx.draw_text note_txt (+ PAD_X 48) 52 235 235 235)
      (Gfx.draw_text "状态：" PAD_X 76 150 200 235)
      (Gfx.draw_text status (+ PAD_X 48) 76 210 220 235)
      (Gfx.draw_text "落盘：" PAD_X 94 150 200 235)
      (Gfx.draw_text (String.concat (String.atom_to_string saved_bytes) " B -> data/ui_demo.txt")
                     (+ PAD_X 48) 94 180 200 220)
      (draw_hint "再按 Enter 重来 · A 退出（全局确认框）")
      (Gfx.refresh))))

;; ---- 主流程（含三处阻塞式调用） ------------------------------------------

(define write_log
  (lambda ()
    (begin
      (Fs.write DATA_PATH
                (String.concat "name=" (String.concat name_txt
                  (String.concat "\nnote=" note_txt))))
      (set! saved_bytes (Fs.size DATA_PATH)))))

(define run_flow
  (lambda ()
    (begin
      ;; ① 取名字
      (set! name_txt (Ui.input "请输入你的名字"))
      (if (null? name_txt)
          (begin
            (set! name_txt "#null")
            (set! note_txt "-")
            (set! status "已取消（Ui.input 返回 #null）"))
          (begin
            ;; ② 带初值再问一次（初值 = 刚才的名字，可直接回车接受）
            (set! note_txt (Ui.input_with "备注（可直接回车接受初值）" name_txt))
            (if (null? note_txt)
                (begin
                  (set! note_txt "#null")
                  (set! status "备注已取消"))
                (set! status "两项都已提交"))))

      ;; ③ 落盘 + 只读消息框汇报（消息框也阻塞，关闭后继续）
      (write_log)
      (Ui.message (String.concat "已收到：\nname=" (String.concat name_txt
                    (String.concat "\nnote=" note_txt))))

      (set! stage 1)
      (draw_result))))

(define on_tick
  (lambda ()
    (begin
      (set! tick (+ tick 1))
      (if (== stage 0)
          (begin
            (if (== (Key.take_press) 13)
                (run_flow)
                (if (Touch.take_press)
                    (run_flow)
                    (draw_idle))))
          (begin
            ;; 结果画面：Enter/点按 → 回到说明画面重新来
            (if (== (Key.take_press) 13)
                (begin (set! stage 0) (set! status "就绪") (draw_idle))
                (if (Touch.take_press)
                    (begin (set! stage 0) (set! status "就绪") (draw_idle))
                    0)))))))

;; ---- 启动 ----------------------------------------------------------------

(define boot
  (lambda ()
    (begin
      (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (draw_idle)
      (set! ts (System.set_interval 60 on_tick)))))

(boot)

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行
(EX.install)
