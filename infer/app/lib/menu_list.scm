;; ============================================================================
;; menu_list.scm —— 通用**滚动菜单列表**组件（菜单性质控件的可复用实现）
;;
;; 为什么要有：目录面板、设置面板这类"一列可选条目"的控件在本 App 里已经出现三处
;; （目录 / 设置 / 文档列表），各自都写过一遍"高亮 + 选中跟随 + 点按反算下标 + 行高"
;; 的逻辑，而且都缺**滑动滚动**（低资源长文本阅读器的目录动辄上千条，靠 ↑↓ 翻不动）。
;; 本模块把它们收敛成一个纯 UI 组件：条目样式可配（文字颜色、缩进、指示灯），
;; 手势（拖动滚动 / 轻点选中）与键盘（↑↓ / 翻页 / 确认）都在组件内统一处理。
;;
;; 条目 = 9 元列表（用 (ML.item …) 构造）：
;;   (文本  r g b  缩进  指示灯  选中r 选中g 选中b)
;;   · 文本     —— 显示字符串（组件负责按视口宽裁剪，CJK 按全角宽度算）
;;   · r g b    —— 常态文字颜色
;;   · 缩进     —— 层级缩进（每级 10px）
;;   · 指示灯   —— 0 无 / 1 空心（未选中）/ 2 实心（选中）
;;                 "琴键开关指示灯"：设置面板的字体单选、目录的分段标记都用它
;;   · 选中r/g/b —— 该行被高亮时的文字颜色（不给就用常态色）
;;
;; 高亮样式：**只有一行底色**（40,50,68），**不画左侧竖直亮条**——左侧亮条会和
;; 指示灯、缩进挤在一起（用户明确要求去掉）。需要更强的指示时靠"指示灯 + 选中色"。
;;
;; 视口与裁剪：调用方给出 (x y w h)；组件只画视口内的行，视口外的行不画，边界上
;; 的半行由调用方**随后**覆盖绘制页眉/页脚（与 app/markdown.scm 里阅读页同一手法：
;; 图形层没有裁剪矩形，靠绘制顺序裁剪）。
;;
;; 触屏语义（与阅读页的拖动手感一致）：手指移动多少滚多少；位移 < 8px 视为轻点，
;; 轻点在条目上 = 选中该条并把它作为"激活"返回给调用方。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/list/append/map/cond 多表达式；
;;   空表一律 (cdr '(0)) 现造；push 返回 #undefined；while 循环体不是独立词法区域，
;;   故所有循环变量都在函数体顶层 define。
;;
;; 单例：模块级状态（同一时刻只有一个菜单在交互；调用方每次打开面板时 ML.begin）。
;; ============================================================================

(native Gfx)
(native Touch)
(native Math)
(native String)

(define new_list (lambda () (cdr '(0))))
(define min2 (lambda (a b) (if (< a b) a b)))
(define max2 (lambda (a b) (if (> a b) a b)))

;; ---- 字模度量（调用方按当前字模喂进来）----------------------------------
;;   half/full —— 半角/全角步进（按像素裁剪文字用）
;;   ink_h     —— 字形墨高（12px 字模 = 12，16px 字模 = 16；行内垂直居中用）
;;   pad       —— 文字框顶相对行内居中位置的内缩（点阵字模 0，抗锯齿字模 1）
;;   top_gap   —— 字模的「行顶 → 墨迹」空隙（点阵 0 / 12px 抗锯齿 2 / 16px 抗锯齿 1）。
;;               (Gfx.draw_text) 现在把 y 当**墨迹顶**绘制（见 am_native_Gfx.c 的
;;               native_text_top_gap）：本组件算出的 ty 是**行框顶**，必须把这段空隙
;;               加回来，墨迹才会落在行框的垂直中线上（否则整列文字比中线高 top_gap 像素）。
;;               调用方从自己的字模表里取（markdown 按 font_id，dict 固定 2）。

(define w_half 7)
(define w_full 12)
(define ink_h 12)
(define ink_pad 1)
(define ink_gap 0)

(define metrics
  (lambda (half full h pad gap)
    (set! w_half half)
    (set! w_full full)
    (set! ink_h (if (undefined? h) 12 h))
    (set! ink_pad (if (undefined? pad) 0 pad))
    (set! ink_gap (if (undefined? gap) 0 gap))
    #t))

(define char_w (lambda (cp) (if (>= cp 11904) w_full w_half)))   ;; 0x2E80 起算全角

;; 按像素宽裁剪（超出用 … 代替）——与 md_layout 的同名逻辑等价，但本组件不依赖它
(define clip_text
  (lambda (s maxw)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (define cw 0)
    (define stop 0)
    (define cut -1)
    (while (and (< i n) (== stop 0))
      { (set! cw (char_w (String.charCodeAt i s)))
        (if (> (+ w cw) (- maxw w_half))
            { (set! cut i) (set! stop 1) }
            { (set! w (+ w cw)) (set! i (+ i 1)) }) })
    (if (< cut 0) s
        (if (== cut 0) ""
            (String.concat (String.slice s 0 cut) "…")))))

;; ---- 状态（单例）--------------------------------------------------------

(define items (new_list))
(define row_h 15)
(define vx 0)          ;; 视口
(define vy 0)
(define vw 320)
(define vh 200)
(define sel 0)
(define scroll 0)
;; 触屏状态
(define t_prev 0)
(define t_last 0)
(define t_total 0)
(define t_px 0)
(define t_py 0)
;; 配色
(define HL_R 40) (define HL_G 50) (define HL_B 68)      ;; 高亮行底色
(define BAR_R 40) (define BAR_G 47) (define BAR_B 62)    ;; 滚动条槽
(define TH_R 130) (define TH_G 150) (define TH_B 184)    ;; 滚动条滑块
(define BAR_W 3)                                        ;; 滚动条宽（贴视口右缘）
(define LAMP_ON_R 102) (define LAMP_ON_G 204) (define LAMP_ON_B 255)  ;; 指示灯亮
(define LAMP_OFF_R 110) (define LAMP_OFF_G 118) (define LAMP_OFF_B 140)

;; ---- 条目构造 -----------------------------------------------------------

(define item
  (lambda (txt r g b ind lamp sr sg sb)
    (cons txt (cons r (cons g (cons b (cons ind (cons lamp
      (cons sr (cons sg (cons sb (cdr '(0)))))))))))))

;; ---- 基本访问器 ---------------------------------------------------------

(define count (lambda () (length items)))
(define sel_of (lambda () sel))
(define scroll_of (lambda () scroll))
(define row_h_of (lambda () row_h))
(define content_h (lambda () (* (length items) row_h)))
(define vis_rows (lambda () (max2 1 (Math.floor (/ vh row_h)))))

(define max_scroll
  (lambda ()
    (define m (- (content_h) vh))
    (if (< m 0) 0 m)))

(define clamp_scroll
  (lambda ()
    (if (< scroll 0) (set! scroll 0) #f)
    (if (> scroll (max_scroll)) (set! scroll (max_scroll)) #f)
    #t))

(define clamp_sel
  (lambda ()
    (define n (length items))
    (if (>= sel n) (set! sel (max2 0 (- n 1))) #f)
    (if (< sel 0) (set! sel 0) #f)
    #t))

;; ---- 生命周期 -----------------------------------------------------------

;; 重置为一个全新菜单（打开面板时调用）
;; 注意：不能叫 begin（那是句法关键字，顶层 define 会编译失败——踩过）
(define reset
  (lambda (rows rh)
    (set! items rows)
    (set! row_h (if (< rh 1) 1 rh))
    (set! sel 0)
    (set! scroll 0)
    (set! t_prev 0)
    (set! t_total 0)
    (set! t_last 0)
    (clamp_sel)
    (clamp_scroll)
    #t))

;; 视口矩形（面板布局后调用）
(define geom
  (lambda (x y w h)
    (set! vx x) (set! vy y) (set! vw w) (set! vh h)
    (clamp_scroll)
    #t))

;; 换条目表（保留 sel / scroll 并夹取；用于增量重建，如目录异步增长）
(define set_items
  (lambda (rows)
    (set! items rows)
    (clamp_sel)
    (clamp_scroll)
    #t))

;; 追加条目（异步增量场景：目录扫描出新标题时只往后追加，选中项下标不会失效）
(define append_items
  (lambda (rows)
    (define n (length rows))
    (define i 0)
    (while (< i n)
      { (push items (get_item rows i))
        (set! i (+ i 1)) })
    (clamp_scroll)
    #t))

(define set_sel
  (lambda (n)
    (set! sel n)
    (clamp_sel)
    (ensure_sel)
    #t))

(define set_scroll
  (lambda (px)
    (set! scroll px)
    (clamp_scroll)
    #t))

;; ---- 选中项与滚动联动 ---------------------------------------------------

;; 让选中项完整落在视口内（键盘/程序改 sel 后调用）
(define ensure_sel
  (lambda ()
    (define top (* sel row_h))
    (define bot (+ top row_h))
    (if (< top scroll) (set! scroll top) #f)
    (if (> bot (+ scroll vh)) (set! scroll (- bot vh)) #f)
    (clamp_scroll)))

;; 键盘：上下移动 d 条
(define move
  (lambda (d)
    (set! sel (+ sel d))
    (clamp_sel)
    (ensure_sel)
    #t))

;; 键盘：翻页（dir = 1 下 / -1 上），保留一行重叠便于连续阅读
(define page
  (lambda (dir)
    (define step (max2 1 (- (vis_rows) 1)))
    (set! sel (+ sel (* dir step)))
    (clamp_sel)
    (ensure_sel)
    #t))

;; ---- 命中测试与触屏 -----------------------------------------------------

;; 屏幕 y → 条目下标；视口外/空表返回 -1
(define tap_at
  (lambda (py)
    (define n (length items))
    (define r -1)
    (define k 0)
    (if (and (>= py vy) (< py (+ vy vh)))
        { (set! k (+ (Math.floor (/ scroll row_h))
                     (Math.floor (/ (- py vy) row_h))))
          (if (and (>= k 0) (< k n)) (set! r k) #f) }
        #f)
    r))

;; 处理触屏的一拍：拖动 = 滚动（1:1 跟手）；轻点 = 选中并返回该条下标。
;; 返回：>= 0 命中的条目下标；-1 本次没有"点选"；-2 轻点落在视口外（调用方
;; 可据此实现"点页眉关闭面板"——用 (ML.tap_y) 取那次轻点的屏幕 y）。
(define step_touch
  (lambda ()
    (define r -1)
    (define dyy 0)
    (define hit 0)
    (if (Touch.is_pressed)
        { (if (== t_prev 0)
              { (set! t_prev 1)
                (set! t_total 0)
                (set! t_last (Touch.y))
                (set! t_px (Touch.x))
                (set! t_py (Touch.y)) }
              #f)
          (set! dyy (- (Touch.y) t_last))
          (set! t_last (Touch.y))
          (set! t_total (+ t_total dyy))
          ;; 手指向上拖（dyy < 0）→ 内容向上走 → scroll 增大
          (set! scroll (- scroll dyy))
          (clamp_scroll) }
        { (if (== t_prev 1)
              { (set! t_prev 0)
                (if (< (Math.abs t_total) 8)
                    { (set! hit (tap_at t_py))
                      (if (>= hit 0)
                          { (set! sel hit)
                            (clamp_sel)
                            (set! r hit) }
                          (set! r -2)) }
                    #f) }
              #f) })
    r))

;; 是否正在拖动（调用方据此抑制"轻点即激活"）
(define dragging (lambda () (if (> (Math.abs t_total) 8) #t #f)))
;; 最近一次轻点的屏幕 y（配合 -2 判定点在了页眉/页脚）
(define tap_y (lambda () t_py))
(define tap_x (lambda () t_px))

;; ---- 菜单自己的返回键（右上角）------------------------------------------
;; 面板是全屏覆盖式的，阅读页右上角那个「退出」按钮在面板里不存在；因此每个菜单都要有
;; 自己的**返回**入口。样式与 app/lib/exit.scm 的「退出」按钮一致（36x16、同配色），
;; 位置也一致（右上角），用户的肌肉记忆不用改。
(define back_x 0)
(define back_y 2)
(define back_w 36)
(define back_h 16)
(define back_label "返回")

(define back_geom
  (lambda (x y w h)
    (set! back_x x) (set! back_y y) (set! back_w w) (set! back_h h)
    #t))

;; 按屏幕宽自动放到右上角（默认 36x16，右边留 4px、上边留 2px）
(define back_top_right
  (lambda (sw)
    (back_geom (- sw (+ back_w 4)) 2 back_w back_h)))

(define hit_back
  (lambda (px py)
    (if (and (>= px back_x) (< px (+ back_x back_w)))
        (if (and (>= py back_y) (< py (+ back_y back_h))) 1 0)
        0)))

(define draw_back
  (lambda ()
    (Gfx.fill_rect back_x back_y back_w back_h 60 62 74)
    (Gfx.draw_rect back_x back_y back_w back_h 120 124 140)
    (Gfx.draw_text back_label (+ back_x 6) (+ back_y 2) 225 228 235)
    #t))

;; ---- 绘制 ---------------------------------------------------------------

(define draw_row
  (lambda (i y)
    (define it (get_item items i))
    (define txt (get_item it 0))
    (define ind (get_item it 4))
    (define lamp (get_item it 5))
    (define tx (+ vx (+ 6 (* ind 10))))
    ;; 行内**垂直居中**：墨高居中（再减掉字模的框顶内缩 pad），最后加回字模的
    ;; 「行顶→墨迹」空隙 gap —— 绘制端以墨迹顶为 y（见 am_native_Gfx.c），不加回就偏高。
    (define ty (+ (+ y (- (Math.floor (/ (- row_h ink_h) 2)) ink_pad)) ink_gap))
    (define ly (+ y (Math.floor (/ (- row_h 5) 2))))
    (if (== i sel)
        { (Gfx.fill_rect vx y vw row_h HL_R HL_G HL_B)
          (Gfx.draw_text (clip_text txt (- (- vw (- tx vx)) 12))
                         tx ty (get_item it 6) (get_item it 7) (get_item it 8)) }
        (Gfx.draw_text (clip_text txt (- (- vw (- tx vx)) 12))
                       tx ty (get_item it 1) (get_item it 2) (get_item it 3)))
    ;; 琴键开关指示灯：实心 = 开，空心 = 关（画在文字左侧，与缩进/高亮底色都不冲突）
    (if (> lamp 0)
        { (set! tx (+ vx (+ 2 (* ind 10))))
          (if (== lamp 2)
              (Gfx.fill_rect tx ly 5 5 LAMP_ON_R LAMP_ON_G LAMP_ON_B)
              (Gfx.draw_rect tx ly 5 5 LAMP_OFF_R LAMP_OFF_G LAMP_OFF_B)) }
        #f)
    #t))

;; 绘制视口内的行（视口外的行不画；边界半行由调用方随后覆盖页眉/页脚）
(define draw
  (lambda ()
    (define n (length items))
    (define i (Math.floor (/ scroll row_h)))
    (define y 0)
    (define ch (content_h))
    (define stop 0)
    (define slot 0)
    (while (and (< i n) (== stop 0))
      { (set! y (+ vy (- (* i row_h) scroll)))
        (if (>= y (+ vy vh))
            (set! stop 1)
            (draw_row i y))
        (set! i (+ i 1)) })
    ;; 滚动条（内容超过视口时才画）：贴视口右缘 3px，槽 + 滑块；
    ;; 滑块长度按"视口/内容"比例，位置按滚动比例——目录上千条时也能一眼看出位置。
    (if (> ch vh)
        { (set! slot (Math.floor (/ (* vh vh) ch)))
          (if (< slot 14) (set! slot 14) #f)
          (if (> slot vh) (set! slot vh) #f)
          (Gfx.fill_rect (- (+ vx vw) BAR_W) vy BAR_W vh BAR_R BAR_G BAR_B)
          (Gfx.fill_rect (- (+ vx vw) BAR_W)
                         (+ vy (Math.floor (/ (* scroll (- vh slot)) (max_scroll))))
                         BAR_W slot TH_R TH_G TH_B) }
        #f)
    #t))
