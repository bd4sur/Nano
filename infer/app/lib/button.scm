;; ============================================================================
;; button.scm —— 通用**按钮**控件（Button）：圆角矩形底板 + 居中标签 + 命中/点按
;;
;; 为什么要有：本仓 App 里"圆角底板 + 居中文字"的按钮已经被手写过至少三遍——
;;   · app/waterpool.scm 的控件栏（(Gfx.rounded_rect) + (Gfx.draw_text_centered)，
;;     常态/激活/按下三套底色，圆角 7px，半透明混合）；
;;   · app/dict.scm 的键帽与页脚键（外圈 1px 描边 + 内凹圆角矩形，圆角 2px）；
;;   · app/lib/exit.scm 的右上角「退出」按钮（方角底板 + 1px 描边 + 左对齐文字）。
;; 各自的尺寸、配色、字模、命中判定散落在 App 里，改一处样式要翻三处代码。
;; 本模块把"怎么画"与"怎么算命中/点按"收敛成一个**描述表**（descriptor）式控件：
;; 一枚按钮 = 一个 10 元列表，参数各有默认值，可随时改；同一屏上可并存任意多枚，
;; 互不干扰（每枚按钮自己记着"按下是否落在自己身上"）。
;;
;; 用法（三行）：
;;   (import BTN "app/lib/button.scm")
;;   (define b (BTN.make "确定" 20 100 80 24))    ; 标签 + 矩形，其余参数全取默认值
;;   ...
;;   (BTN.draw b)                                 ; 画（常态）
;;   (if (BTN.track b) (on_click) #f)             ; 每拍一次：完成一次点按 → #t
;;
;; 可配参数（**全部有默认值**，用 BTN.set_* 覆盖；改过的参数随时可再改）：
;;   标签内容  "按钮"                     (BTN.set_label b "暂停")
;;   标签字体  1（12px 抗锯齿）            (BTN.set_font b 2)      ; 0/1/2 同 (Gfx.set_font)
;;   标签颜色  240 246 255                (BTN.set_text_color b r g bb)
;;   常态底色  34 46 62（水池控件栏同色）  (BTN.set_bg b r g bb)
;;   激活底色  116 172 226（水池按下色）   (BTN.set_active_bg b r g bb)
;;   圆角半径  6 6 6 6（四角可分别给）     (BTN.set_radius b 7)    ; 或 (BTN.set_radius4 b tl tr br bl)
;;   矩形      0 0 60 24                  (BTN.set_rect b x y w h)
;;                                        (BTN.set_size b w h)    ; 只改长宽（左上角不动）
;;                                        (BTN.set_pos  b x y)    ; 只改位置（尺寸不动）
;;   描边      #f（不描边）               (BTN.set_border b r g bb) ; 1px 内描边（dict 键帽同款）
;;                                        (BTN.no_border b)
;;   混合模式  1（置色）                  (BTN.set_mode b 110)    ; >=4 = 定值 alpha 半透明
;;                                        ;（waterpool 的控件栏底板就是 alpha 110/200/240）
;;
;; 绘制：
;;   (BTN.draw b)            —— 常态
;;   (BTN.draw_state b st)   —— st 为真（#t 或 1）用激活底色，否则常态
;;   (BTN.armed b)           —— 这枚按钮此刻是否"按住在内"（配合 draw_state 做按下高亮）：
;;                              (BTN.draw_state b (BTN.armed b))
;;   标签按按钮宽度自动裁剪：用 (Gfx.font_info) 的半角/全角步进算宽，**放得下就原样画**，
;;   真放不下才以 … 截断（不要求调用方提供字模度量表）；裁剪留两侧各 4px 内边距。
;;   描边是"外圈 1px 圆角矩形 + 内缩 1px 的底色圆角矩形"，与 app/dict.scm 的键帽
;;   同一手法；圆角半径由图形层按 CSS 规则自动收敛，半径给大了也不会画歪。
;;
;; 交互（触屏**电平 → 边沿**，与 app/lib/exit.scm 的老做法逐条一致）：
;;   (BTN.track b) —— 每拍调用一次（画完按钮、推屏之前）。语义：
;;     · 按下落在按钮内     → 记 armed，并**吞掉按下沿**（(Touch.take_press)）；
;;     · 按住后滑出按钮     → 取消（armed 立即清 0，按钮立刻不再显示按下高亮）；
;;     · 在按钮内松开       → 返回 #t（完成一次点按），并**吞掉松开沿**；
;;     · 按下落在按钮外     → 一概不管（边沿与位置都原样留给 App）。
;;   吞边沿的意义：按钮占住的那一下手势不会再被 App 自己的输入逻辑看到，所以给
;;   App 加一枚按钮不需要动它任何输入代码（退出按钮就是这么做的）。
;;   (BTN.reset b) —— 清掉本控件的交互状态（切页/离开界面时调用，避免残留 armed）。
;;
;;   慢节奏 App（几百 ms 一拍）用 `每拍采样触屏电平` 抓不住 ~150ms 的点按——这类 App
;;   要照 app/lib/exit.scm 的办法，把按钮矩形登记给宿主做**触摸边沿**级命中判定
;;   （(App.set_exit_button) 那一套）；本控件的 (BTN.x_of)/(BTN.y_of)/(BTN.w_of)/(BTN.h_of)
;;   正好用来取登记用的矩形。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/list/append/map；cond 每个
;;   子句只接受单个表达式；空表一律 (cdr '(0)) 现造（`'()` 字面量是静态对象，就地
;;   push/set_item! 不安全，见 manual §3.4）；define 一律放在 lambda 体顶层（块内不
;;   define）；布尔一律 #t/#f——**0 在本解释器里是真值**（manual §4.1）。
;; ============================================================================

(native Gfx)
(native Touch)
(native Math)
(native String)

;; ---- 描述表字段下标 -------------------------------------------------------
;; 描述表 = 10 元列表；其中「颜色」是 3 元列表、「矩形/圆角」是 4 元列表、
;; 「交互状态」是 3 元列表（按住在内 / 已吞按下沿 / 上一拍电平）。
(define F_LABEL 0)
(define F_FONT 1)
(define F_TXT 2)      ;; (r g b)  标签文字颜色
(define F_BG 3)       ;; (r g b)  常态底色
(define F_ON 4)       ;; (r g b)  激活（按下）底色
(define F_RAD 5)      ;; (tl tr br bl) 圆角半径
(define F_RECT 6)     ;; (x y w h)
(define F_BRD 7)      ;; (r g b) 或 #f（无描边）
(define F_ST 8)       ;; (armed consumed prev_down)  交互状态
(define F_MODE 9)     ;; (Gfx.rounded_rect) 的 mode（1=置色，>=4=定值 alpha）

;; ---- 小工具 ---------------------------------------------------------------

;; 空表：一律现造（不用 '() 字面量，理由见文件头）
(define elist (lambda () (cdr '(0))))
(define tri (lambda (a b c) (cons a (cons b (cons c (elist))))))
(define quad (lambda (a b c d) (cons a (cons b (cons c (cons d (elist)))))))

;; 「开」的统一判据：调用方可能传 #t/#f，也可能传 1/0（0 在本解释器里是真值，
;; 但参数上 0 习惯表示"关"，两种写法都要认）
(define is_on (lambda (v) (if (== v #f) #f (if (== v 0) #f #t))))

;; ---- 构造 -----------------------------------------------------------------

;; 按 10 个字段拼一张描述表（字段顺序见上面的 F_* 下标）
(define desc
  (lambda (label font txt bg on rad rect brd st mode)
    (cons label (cons font (cons txt (cons bg (cons on
      (cons rad (cons rect (cons brd (cons st (cons mode
        (elist)))))))))))))

;; 一枚按钮（全部参数取默认值：60x24、圆角 6、12px 抗锯齿字、水池配色、不描边）
(define new
  (lambda ()
    (desc "按钮"                       ; 标签内容
          1                            ; 标签字体（1 = 12px 抗锯齿）
          (tri 240 246 255)            ; 标签颜色
          (tri 34 46 62)               ; 常态底色
          (tri 116 172 226)            ; 激活底色
          (quad 6 6 6 6)               ; 圆角半径
          (quad 0 0 60 24)             ; 矩形
          #f                           ; 描边（#f = 无）
          (tri 0 0 0)                  ; 交互状态（armed / consumed / prev）
          1)))                         ; 混合模式（1 = 置色）

;; 常用构造：标签 + 矩形，其余取默认值
(define make
  (lambda (label x y w h)
    (define b (new))
    (set_item! b F_LABEL label)
    (set_item! b F_RECT (quad x y w h))
    b))

;; ---- 访问器 ---------------------------------------------------------------

(define label_of (lambda (b) (get_item b F_LABEL)))
(define font_of (lambda (b) (get_item b F_FONT)))
(define txt_of (lambda (b) (get_item b F_TXT)))
(define bg_of (lambda (b) (get_item b F_BG)))
(define on_of (lambda (b) (get_item b F_ON)))
(define rad_of (lambda (b) (get_item b F_RAD)))
(define rect_of (lambda (b) (get_item b F_RECT)))
(define brd_of (lambda (b) (get_item b F_BRD)))
(define st_of (lambda (b) (get_item b F_ST)))
(define mode_of (lambda (b) (get_item b F_MODE)))

(define x_of (lambda (b) (get_item (rect_of b) 0)))
(define y_of (lambda (b) (get_item (rect_of b) 1)))
(define w_of (lambda (b) (get_item (rect_of b) 2)))
(define h_of (lambda (b) (get_item (rect_of b) 3)))

;; 是否"按住在内"（绘制按下高亮用）；触屏不可用时恒 #f
(define armed (lambda (b) (if (== (get_item (st_of b) 0) 1) #t #f)))

;; ---- 参数设置 -------------------------------------------------------------

(define set_label (lambda (b s) { (set_item! b F_LABEL s) #t }))
(define set_font (lambda (b id) { (set_item! b F_FONT id) #t }))

(define set_text_color
  (lambda (b r g bb)
    (define t (txt_of b))
    (set_item! t 0 r) (set_item! t 1 g) (set_item! t 2 bb)
    #t))

(define set_bg
  (lambda (b r g bb)
    (define t (bg_of b))
    (set_item! t 0 r) (set_item! t 1 g) (set_item! t 2 bb)
    #t))

(define set_active_bg
  (lambda (b r g bb)
    (define t (on_of b))
    (set_item! t 0 r) (set_item! t 1 g) (set_item! t 2 bb)
    #t))

;; 四角同一半径
(define set_radius
  (lambda (b r)
    (define t (rad_of b))
    (set_item! t 0 r) (set_item! t 1 r) (set_item! t 2 r) (set_item! t 3 r)
    #t))

;; 四角分别给（顺序：左上 右上 右下 左下，与 (Gfx.rounded_rect) 一致）
(define set_radius4
  (lambda (b tl tr br bl)
    (define t (rad_of b))
    (set_item! t 0 tl) (set_item! t 1 tr) (set_item! t 2 br) (set_item! t 3 bl)
    #t))

(define set_rect
  (lambda (b x y w h)
    (set_item! b F_RECT (quad x y w h))
    #t))

(define set_size
  (lambda (b w h)
    (define r (rect_of b))
    (set_item! r 2 w) (set_item! r 3 h)
    #t))

(define set_pos
  (lambda (b x y)
    (define r (rect_of b))
    (set_item! r 0 x) (set_item! r 1 y)
    #t))

;; 1px 内描边（颜色）
(define set_border
  (lambda (b r g bb)
    (set_item! b F_BRD (tri r g bb))
    #t))

(define no_border (lambda (b) { (set_item! b F_BRD #f) #t }))

;; 混合模式：1 = 置色（不透明）；>=4 = 定值 alpha（0~255），如 110 的半透明底板
(define set_mode (lambda (b m) { (set_item! b F_MODE m) #t }))

;; ---- 标签裁剪（按按钮宽度）------------------------------------------------

;; 字模度量缓存：m_fid 记住上次取的字体，避免每次绘制都造一个度量列表
;; （(Gfx.font_info id) → (半角步进 全角步进 行高 基线)）
(define m_fid -1)
(define m_half 7)
(define m_full 12)

(define ensure_metrics
  (lambda (fid)
    (define fi #null)
    (if (== m_fid fid)
        #t
        {
          (set! fi (Gfx.font_info fid))
          (if (null? fi)
              { (set! m_half 7) (set! m_full 12) }
              { (set! m_half (get_item fi 0)) (set! m_full (get_item fi 1)) })
          (set! m_fid fid)
        })))

;; 0x2E80（CJK 部首）起算全角，与 app/lib/menu_list.scm 同一判据
(define char_w (lambda (cp) (if (>= cp 11904) m_full m_half)))

;; 字符串总步进宽度（按字模度量累加）
(define text_w
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (while (< i n)
      { (set! w (+ w (char_w (String.charCodeAt i s))))
        (set! i (+ i 1)) })
    w))

;; 放不下时按像素宽截断并以 … 结尾（调用前已确定放不下，故总要为 … 预留半角位）
(define clip_ellipsis
  (lambda (s maxw)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (define cw 0)
    (define cut -1)
    (define stop 0)
    (while (and (< i n) (== stop 0))
      {
        (set! cw (char_w (String.charCodeAt i s)))
        (if (> (+ w cw) (- maxw m_half))
            { (set! cut i) (set! stop 1) }
            { (set! w (+ w cw)) (set! i (+ i 1)) })
      })
    (if (< cut 0) s
        (if (== cut 0) ""
            (String.concat (String.slice s 0 cut) "…")))))

;; 按像素宽裁剪标签：**放得下就原样返回**（不加省略号），放不下才截断。
;; （早先版本无条件预留…的位置，导致「退出」这种刚好 24px 宽、可用 28px 的标签
;;   被画成「退…」——见 doc/AGENTS.md 2026-11-15（五））
(define clip_text
  (lambda (s maxw)
    (if (<= (text_w s) maxw) s (clip_ellipsis s maxw))))

(define clip_label
  (lambda (b)
    (define maxw (- (w_of b) 8))          ;; 两侧各留 4px 内边距
    (ensure_metrics (font_of b))
    (if (< maxw m_half) "" (clip_text (label_of b) maxw))))

;; ---- 绘制 -----------------------------------------------------------------

(define dec_r (lambda (r) (if (> r 0) (- r 1) 0)))

;; 圆角底色 + 可选 1px 描边（"外圈描边色 + 内缩 1px 的底色"，同 dict 键帽手法）
(define paint
  (lambda (rgb rad x y w h brd mode)
    (if brd
        {
          (Gfx.rounded_rect x y w h
                            (get_item rad 0) (get_item rad 1)
                            (get_item rad 2) (get_item rad 3)
                            (get_item brd 0) (get_item brd 1) (get_item brd 2) mode)
          (if (and (> w 2) (> h 2))
              (Gfx.rounded_rect (+ x 1) (+ y 1) (- w 2) (- h 2)
                                (dec_r (get_item rad 0)) (dec_r (get_item rad 1))
                                (dec_r (get_item rad 2)) (dec_r (get_item rad 3))
                                (get_item rgb 0) (get_item rgb 1) (get_item rgb 2) mode)
              #f)
        }
        (Gfx.rounded_rect x y w h
                          (get_item rad 0) (get_item rad 1)
                          (get_item rad 2) (get_item rad 3)
                          (get_item rgb 0) (get_item rgb 1) (get_item rgb 2) mode))
    #t))

;; 按状态绘制（st 为真 → 激活底色）
(define draw_state
  (lambda (b st)
    (define r (rect_of b))
    (define x (get_item r 0))
    (define y (get_item r 1))
    (define w (get_item r 2))
    (define h (get_item r 3))
    (define rgb (if (is_on st) (on_of b) (bg_of b)))
    (define tc (txt_of b))
    (paint rgb (rad_of b) x y w h (brd_of b) (mode_of b))
    ;; (Gfx.draw_text_centered) 的参数是**行框中心**，字模由第一个参数显式给出
    (Gfx.draw_text_centered (font_of b) (clip_label b)
                            (+ x (Math.floor (/ w 2)))
                            (+ y (Math.floor (/ h 2)))
                            (get_item tc 0) (get_item tc 1) (get_item tc 2) 1)
    #t))

;; 常态绘制
(define draw (lambda (b) (draw_state b #f)))

;; ---- 命中与点按 -----------------------------------------------------------

;; 点是否落在按钮矩形内（帧缓冲像素坐标）
(define hit
  (lambda (b px py)
    (define r (rect_of b))
    (define x (get_item r 0))
    (define y (get_item r 1))
    (and (and (>= px x) (< px (+ x (get_item r 2))))
         (and (>= py y) (< py (+ y (get_item r 3)))))))

;; 每拍调用一次；返回 #t 表示"按下落在按钮内、并在按钮内松开"——完成一次点按。
;; 按下/松开的**边沿**由本控件吞掉，调用方不必再处理这一下手势。
;; "在按钮内松开"与 app/waterpool.scm 的控件栏、各 App 的触屏按钮同一口径：按住后
;; 滑出按钮即取消（armed 立即清 0，按钮也立刻不再显示按下高亮）。
(define track
  (lambda (b)
    (define st (st_of b))
    (define a (get_item st 0))
    (define used (get_item st 1))
    (define prev (get_item st 2))
    (define px (Touch.x))
    (define py (Touch.y))
    (define inside (hit b px py))
    (define r #f)
    (if (Touch.is_pressed)
        {
          ;; 上升沿：落在按钮里才算按中了它，并吞掉按下沿（不让 App 再看到这一下）
          (if (== prev 0)
              (if inside
                  {
                    (set_item! st 0 1)
                    (set_item! st 1 1)
                    (Touch.take_press)
                  }
                  {
                    (set_item! st 0 0)
                    (set_item! st 1 0)
                  })
              #f)
          ;; 按住：手指滑出按钮 → 取消（再滑回来也不算，必须重新按下）
          (if (and (== a 1) (== inside #f)) (set_item! st 0 0) #f)
          (set_item! st 2 1)
        }
        {
          ;; 下降沿：只有"吞过按下沿"的手势才吞松开沿；armed 仍为 1 → 完成一次点按
          (if (== prev 1)
              {
                (if (== used 1)
                    {
                      (if (== a 1) (set! r #t) #f)
                      (Touch.take_release)
                    }
                    #f)
                (set_item! st 0 0)
                (set_item! st 1 0)
                (set_item! st 2 0)
              }
              #f)
        })
    r))

;; 清交互状态（切页/离开界面时调用）
(define reset
  (lambda (b)
    (define st (st_of b))
    (set_item! st 0 0)
    (set_item! st 1 0)
    (set_item! st 2 0)
    #t))
