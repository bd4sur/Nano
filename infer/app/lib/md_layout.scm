;; ============================================================================
;; md_layout.scm —— Markdown 排版器（块树 → 显示行）
;;
;; 运行环境：纯灵机 Scheme，只依赖 String / Math 本地库（不碰 Gfx/Fs），
;;           因此可以在桌面解释器里直接单测（tools/e2e/markdown/test_layout.scm）。
;;
;; 输入：md_parse.scm 产出的块列表；输出：**显示行列表**。
;;
;;   行 = (高度 背景 段列表)
;;        高度：该行占用的像素高（正文 15；间隔行可以是 3~12 的小值）
;;        背景：BG_NONE / BG_CODE（代码块底） / BG_RULE（横线）
;;   段 = (文本 样式 x 宽 url)
;;        x  —— 相对正文左边缘的像素偏移（**排版期就定死**，绘制端不做再排版）
;;        宽 —— 该段的像素宽（绘制端画下划线、行内代码底色时直接用）
;;        url —— 该段所属链接的目标（非链接为空串；点按命中后弹窗用）
;;
;; 为什么排版结果是「行 + 段」而不是「块 + 位置」：
;;   · 绘制端只做一件事——按 y 累加往下画，**不知道任何 Markdown 语义**；
;;   · 滚动/翻页/命中测试都退化成"行号 ↔ 像素"的算术；
;;   · 将来支持表格/引用块/嵌套列表时，只要排版端能产出「一行里的多个绝对定位段」，
;;     绘制端一行都不用改（表格的核心需求正是**段可以按列定位**）。
;;
;; 样式表（绘制端按样式查颜色/字重；这里只给编号，颜色在 app 里）
;;   0 正文        1 加粗        2 行内代码    3 链接
;;   4 标题1       5 标题2       6 标题3       7 代码块正文   8 列表项目符号
;;
;; 字宽：本机 12px alpha 字模是等宽的——半角步进 7px、全角（>= U+2E80）12px
;;   （实测：i/M/W/a 均为 7，中文/全角标点为 12；见 infer/glyph_12.h 头注释）。
;;   因此排版端可以用纯算术得到与绘制端**逐像素一致**的宽度，不需要回问宿主。
;;
;; 解释器约束（manual.md，逐条遵守）
;; ---------------------------------------------------------------------------
;; · 算术固定二参数；无 min/max/abs 内建（用到就自己写）；无 let/cond 多表达式；
;; · 空表一律 (cdr '(0)) 现造（'() 是模块级静态槽，push 会跨调用累积）；
;; · 循环不能用 (set! i 末值) 充当 break（会毁掉结果变量），一律用 go/stop 标志；
;; · 排版状态（cur_lines / atoms）是模块级单例——单线程、非重入（每次 layout 重置）。
;; ============================================================================

(native String)
(native Math)

;; ---- 样式与背景编号 ------------------------------------------------------

(define ST_BODY 0)
(define ST_BOLD 1)
(define ST_CODE 2)
(define ST_LINK 3)
(define ST_H1 4)
(define ST_H2 5)
(define ST_H3 6)
(define ST_CODEBLOCK 7)
(define ST_BULLET 8)

(define BG_NONE 0)
(define BG_CODE 1)
(define BG_RULE 2)
(define BG_TABLE 3)

;; ---- 度量常量 ------------------------------------------------------------

(define LINE_H 15)       ;; 12px 字模的行高（含 3px 行距）
(define W_HALF 7)        ;; 半角步进
(define W_FULL 12)       ;; 全角步进
(define CJK_BASE 11904)  ;; 0x2E80
(define CP_SP 32)
(define CP_TAB 9)

;; ---- 造表小工具 ----------------------------------------------------------

(define new_list (lambda () (cdr '(0))))
(define mk2 (lambda (a b) (cons a (cons b (cdr '(0))))))
(define mk3 (lambda (a b c) (cons a (cons b (cons c (cdr '(0)))))))
(define mk4 (lambda (a b c d) (cons a (cons b (cons c (cons d (cdr '(0))))))))
(define mk5 (lambda (a b c d e) (cons a (cons b (cons c (cons d (cons e (cdr '(0)))))))))

;; ---- 宽度计算 ------------------------------------------------------------

(define char_w
  (lambda (cp)
    (if (>= cp CJK_BASE) W_FULL W_HALF)))

(define text_w
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define w 0)
    (while (< i n)
      { (set! w (+ w (char_w (String.charCodeAt i s))))
        (set! i (+ i 1)) })
    w))

;; 按像素宽裁剪字符串（表格单元格 / 超长不可断词用），超出部分用 … 代替
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
        (if (> (+ w cw) (- maxw W_HALF))
            { (set! cut i) (set! stop 1) }
            { (set! w (+ w cw)) (set! i (+ i 1)) }) })
    (if (< cut 0) s
        (if (== cut 0) ""
            (String.concat (String.slice s 0 cut) "…")))))

;; ============================================================================
;; 一、行内节点 → 原子流（原子 = 排版的最小不可分单位：(文本 样式 宽 url)）
;; ============================================================================

(define atoms (new_list))
(define atom_avail 300)   ;; 单原子最大宽度（超过就逐字符拆，保证一定能折行）
(define atom_url "")      ;; 当前所处的链接目标（非空表示原子属于链接）

(define atoms_last_is_space
  (lambda ()
    (define n (length atoms))
    (if (== n 0) #f
        (String.equals (get_item (get_item atoms (- n 1)) 0) " "))))

;; 词原子入流：宽度超过 atom_avail 的词按字符拆开
(define emit_word
  (lambda (w style)
    (define l (String.length w))
    (define i 0)
    (if (> l 0)
        (if (> (text_w w) atom_avail)
            (while (< i l)
              { (push atoms (mk4 (String.charAt w i) style
                                 (char_w (String.charCodeAt i w)) atom_url))
                (set! i (+ i 1)) })
            (push atoms (mk4 w style (text_w w) atom_url)))
        #f)))

;; 纯文本 → 原子流：空格单独成原子（可在此断行），全角字符逐字成原子（可在任意字间断行），
;; 半角字母数字成词（词内不断行）。
(define atomize_text
  (lambda (s style)
    (define n (String.length s))
    (define i 0)
    (define cp 0)
    (define word "")
    (while (< i n)
      { (set! cp (String.charCodeAt i s))
        (if (== cp CP_SP)
            { (emit_word word style)
              (set! word "")
              (if (atoms_last_is_space) #f
                  (push atoms (mk4 " " style W_HALF atom_url))) }
            (if (== cp CP_TAB)
                { (emit_word word style)
                  (set! word "")
                  (if (atoms_last_is_space) #f
                      (push atoms (mk4 " " style W_HALF atom_url))) }
                (if (>= cp CJK_BASE)
                    { (emit_word word style)
                      (set! word "")
                      (push atoms (mk4 (String.charAt s i) style (char_w cp) atom_url)) }
                    (set! word (String.concat word (String.charAt s i))))))
        (set! i (+ i 1)) })
    (emit_word word style)))

;; 行内节点流 → 原子流（递归：加粗/链接内部再走一遍；嵌套深度即行内语法嵌套深度，
;; 实际 Markdown 里只有一两层，不会碰到帧栈上限）。
;;
;; 形状约定（与 md_parse.scm 一致，写错会立刻炸）：
;;   (text "s")            —— 元素 1 是**字符串**
;;   (bold <节点列表>)      —— 元素 1 是**节点列表**（整表一个元素，不是摊平的节点）
;;   (link <节点列表> "url") —— 元素 1 是节点列表，元素 2 是目标地址
;;   (code "s")            —— 元素 1 是字符串
(define atomize
  (lambda (inls base)
    (define n (length inls))
    (define i 0)
    (define nd #f)
    (define tag 'text)
    (define save "")
    (while (< i n)
      { (set! nd (get_item inls i))
        (set! tag (get_item nd 0))
        (if (== tag 'text)
            (atomize_text (get_item nd 1) base)
            (if (== tag 'bold)
                (atomize (get_item nd 1) (bold_style base))
                (if (== tag 'code)
                    (emit_word (get_item nd 1) ST_CODE)
                    (if (== tag 'link)
                        { (set! save atom_url)
                          (set! atom_url (get_item nd 2))
                          (atomize (get_item nd 1) ST_LINK)
                          (set! atom_url save) }
                        (if (== tag 'bullet)
                            (push atoms (mk4 "" ST_BULLET 14 atom_url))
                            #f)))))
        (set! i (+ i 1)) })
    #t))

;; 标题里的加粗仍是标题样式（标题本身已经是粗体，无需再换色）
(define bold_style
  (lambda (base)
    (if (>= base ST_H1) base ST_BOLD)))

;; ============================================================================
;; 二、原子流 → 显示行（贪心折行）
;; ============================================================================

(define cur_lines (new_list))

(define emit_line
  (lambda (h bg segs)
    (push cur_lines (mk3 h bg segs))))

(define emit_gap
  (lambda (h)
    (emit_line h BG_NONE (new_list))))

;; 把一段文本放进当前行的段表：与上一段样式/链接相同且首尾相接时合并（减少绘制调用）
(define place_atom
  (lambda (segs txt st x aw url)
    (define ns (length segs))
    (define last #f)
    (if (> ns 0)
        { (set! last (get_item segs (- ns 1)))
          (if (== (get_item last 1) st)
              (if (String.equals (get_item last 4) url)
                  (if (== (+ (get_item last 2) (get_item last 3)) x)
                      { (set_item! last 0 (String.concat (get_item last 0) txt))
                        (set_item! last 3 (+ (get_item last 3) aw)) }
                      (push segs (mk5 txt st x aw url)))
                  (push segs (mk5 txt st x aw url)))
              (push segs (mk5 txt st x aw url))) }
        (push segs (mk5 txt st x aw url)))
    (+ x aw)))

;; 中文避头点（禁则）：这些字符不能出现在行首，宁可让它「悬挂」超出正文宽一点点
;; （多出 <=12px，落在右留白里）。否则会出现整行只有一个「。」的孤字行。
(define cjk_no_line_start
  (lambda (cp)
    ;; 。，、；：？！） 」 』 】（十进制码点）
    (define lst '(12290 65292 12289 65307 65306 65311 65281 65289 12301 12303 12305))
    (define n (length lst))
    (define i 0)
    (define r #f)
    (while (< i n)
      { (if (== cp (get_item lst i)) (set! r #t) #f)
        (set! i (+ i 1)) })
    r))

;; 原子文本是否是"不可行首"的单个汉字标点
(define atom_hangs
  (lambda (txt)
    (if (>= (String.length txt) 1)
        (cjk_no_line_start (String.charCodeAt 0 txt))
        #f)))

;; 贪心折行：按原子顺序填行，放不下就换行（行首空格丢弃；续行缩进 cont_ind）
(define pack_atoms
  (lambda (width first_ind cont_ind bg)
    (define na (length atoms))
    (define k 0)
    (define x first_ind)
    (define segs (new_list))
    (define at #f)
    (define txt "")
    (define st 0)
    (define aw 0)
    (define url "")
    (define is_sp #f)
    (while (< k na)
      { (set! at (get_item atoms k))
        (set! txt (get_item at 0))
        (set! st (get_item at 1))
        (set! aw (get_item at 2))
        (set! url (get_item at 3))
        (set! is_sp (String.equals txt " "))
        (if (== (length segs) 0)
            (if is_sp #f
                (set! x (place_atom segs txt st x aw url)))
            (if (> (+ x aw) width)
                (if (atom_hangs txt)
                    (set! x (place_atom segs txt st x aw url))   ;; 避头点：悬挂不换行
                    { (emit_line LINE_H bg segs)
                      (set! segs (new_list))
                      (set! x cont_ind)
                      (if is_sp #f
                          (set! x (place_atom segs txt st x aw url))) })
                (set! x (place_atom segs txt st x aw url))))
        (set! k (+ k 1)) })
    (if (> (length segs) 0) (emit_line LINE_H bg segs) #f)))

;; 行内节点 → 显示行（折行 + 缩进 + 背景）
(define emit_wrapped
  (lambda (inls width first_ind cont_ind bg base)
    (set! atoms (new_list))
    (set! atom_url "")
    (set! atom_avail (- width cont_ind))
    (atomize inls base)
    (pack_atoms width first_ind cont_ind bg)))

;; ============================================================================
;; 三、块 → 显示行（扩展点③：块排版的注册表）
;; ============================================================================
;;
;; 每种块一个排版函数 (块 正文宽) → 若干显示行。
;; 新增块类型（表格/引用块/有序列表…）= (layout_add 种类 函数) 一行。
;; 解析端尚未产出的块类型不影响这里注册（注册了就不会因为"不认识的块"而崩）。

(define block_layouts (new_list))

(define layout_add
  (lambda (kind fn)
    (push block_layouts (mk2 kind fn))))

(define layout_of
  (lambda (kind)
    (define n (length block_layouts))
    (define i 0)
    (define r #f)
    (while (and (< i n) (not r))
      { (if (== (get_item (get_item block_layouts i) 0) kind)
            (set! r (get_item (get_item block_layouts i) 1))
            #f)
        (set! i (+ i 1)) })
    r))

;; ---- 标题：H1 > H2 > H3（4~6 级按 H3 处理）-----------------------------
;; 字号无法切换（Gfx 只有一套 12px alpha 字模），因此层级靠
;; 「粗体强度 + 颜色 + 段前留白 + 分隔线」表达：H1 段前最多且带 2px 通栏线，
;; H2 带 1px 线，H3 只有粗体与配色差异。
(define lay_heading
  (lambda (b width)
    (define lvl (get_item b 1))
    (define shown 1)
    (define base ST_H1)
    (set! shown lvl)
    (if (> shown 3) (set! shown 3) #f)
    (if (== shown 2) (set! base ST_H2) #f)
    (if (== shown 3) (set! base ST_H3) #f)
    (emit_gap (if (> lvl 1) 9 13))
    (emit_wrapped (get_item b 2) width 0 0 BG_NONE base)
    (if (== shown 1)
        { (emit_gap 3) (emit_line 2 BG_RULE (new_list)) }
        #f)
    (if (== shown 2)
        { (emit_gap 3) (emit_line 1 BG_RULE (new_list)) }
        #f)
    (emit_gap 7)))

;; ---- 段落 ----------------------------------------------------------------
;; 块形状：(para inlines) —— 行内节点在下标 1（与 heading 的 lvl 占位不同，注意）
(define lay_para
  (lambda (b width)
    (emit_wrapped (get_item b 1) width 0 0 BG_NONE ST_BODY)
    (emit_gap 7)))

;; ---- 无序列表（无嵌套）：项目符号 14px 悬挂缩进 --------------------------
(define lay_list
  (lambda (b width)
    (define items (get_item b 1))
    (define n (length items))
    (define i 0)
    (define inl #f)
    (while (< i n)
      { (set! inl (cons (mk2 'bullet #f) (get_item items i)))
        (emit_wrapped inl width 0 14 BG_NONE ST_BODY)
        (if (< (+ i 1) n) (emit_gap 3) #f)
        (set! i (+ i 1)) })
    (emit_gap 7)))

;; ---- 围栏代码块 ----------------------------------------------------------
(define lay_code
  (lambda (b width)
    (define lns (get_item b 2))
    (define n (length lns))
    (define i 0)
    (emit_gap 6)
    (emit_line LINE_H BG_CODE (new_list))          ;; 上内边距
    (while (< i n)
      { (emit_code_line (get_item lns i) width)
        (set! i (+ i 1)) })
    (emit_line LINE_H BG_CODE (new_list))          ;; 下内边距
    (emit_gap 8)))

;; 代码块的一行：逐字符排（等宽字模），超出正文宽就续行；制表符按 4 个空格
(define emit_code_line
  (lambda (s width)
    (define n (String.length s))
    (define i 0)
    (define cp 0)
    (define segs (new_list))
    (define x 6)
    (define lim (- width 8))
    (define cw 0)
    (while (< i n)
      { (set! cp (String.charCodeAt i s))
        (set! cw (char_w cp))
        (if (== cp CP_TAB) (set! cw (+ W_HALF (+ W_HALF (+ W_HALF W_HALF)))) #f)
        (if (> (+ x cw) lim)
            { (emit_line LINE_H BG_CODE segs)
              (set! segs (new_list))
              (set! x 6) }
            #f)
        (if (== cp CP_TAB)
            (set! x (place_atom segs "    " ST_CODEBLOCK x cw ""))
            (set! x (place_atom segs (String.charAt s i) ST_CODEBLOCK x cw "")))
        (set! i (+ i 1)) })
    (emit_line LINE_H BG_CODE segs)))

;; ---- 分隔线 --------------------------------------------------------------
(define lay_rule
  (lambda (b width)
    (emit_gap 8)
    (emit_line 1 BG_RULE (new_list))
    (emit_gap 9)))

;; ---- 表格（解析端尚未接入；本函数已可用，见 md_parse.scm 的「表格接入指南」）--
;; 输入 (table 表头行列表 数据行列表)，行 = 单元格字符串列表。
;; 排版方式：等宽分栏（列宽 = 正文宽 / 列数），单元格超宽用 … 截断，表头后画一条横线。
;; 将来若要「按内容自适应列宽」，只需在这里先量一遍各列最大宽再按比例分配——
;; 绘制端无需任何改动（段自带绝对 x）。
(define lay_table
  (lambda (b width)
    (define header (get_item b 1))
    (define rows (get_item b 2))
    (define ncol (table_cols b))
    (define colw 0)
    (define i 0)
    (set! colw (Math.floor (/ width ncol)))
    (emit_gap 6)
    (while (< i (length header))
      { (emit_table_row (get_item header i) colw ST_BOLD)
        (set! i (+ i 1)) })
    (emit_line 1 BG_RULE (new_list))
    (set! i 0)
    (while (< i (length rows))
      { (emit_table_row (get_item rows i) colw ST_BODY)
        (set! i (+ i 1)) })
    (emit_gap 8)))

;; 表格列数 = 所有行里最大的单元格个数
(define table_cols
  (lambda (b)
    (define n 1)
    (define i 0)
    (define r #f)
    (define w 0)
    (while (< i 2)
      { (set! r (get_item b (+ i 1)))
        (set! w (rows_max_cols r))
        (if (> w n) (set! n w) #f)
        (set! i (+ i 1)) })
    n))

(define rows_max_cols
  (lambda (rows)
    (define n (length rows))
    (define i 0)
    (define m 0)
    (while (< i n)
      { (if (> (length (get_item rows i)) m) (set! m (length (get_item rows i))) #f)
        (set! i (+ i 1)) })
    m))

(define emit_table_row
  (lambda (cells colw style)
    (define n (length cells))
    (define i 0)
    (define segs (new_list))
    (define t "")
    (define x 0)
    (define w 0)
    (while (< i n)
      { (set! t (get_item cells i))
        (set! x (* i (+ colw 2)))
        (set! w (text_w t))
        (if (> w (- colw 4))
            { (set! t (clip_text t (- colw 4))) (set! w (text_w t)) }
            #f)
        (push segs (mk5 t style x w ""))
        (set! i (+ i 1)) })
    (emit_line LINE_H BG_TABLE segs)))

;; ---- 注册块排版（顺序无关，按种类查表）----------------------------------

(layout_add 'heading lay_heading)
(layout_add 'para lay_para)
(layout_add 'list lay_list)
(layout_add 'code lay_code)
(layout_add 'rule lay_rule)
(layout_add 'table lay_table)

;; ============================================================================
;; 四、对外 API
;; ============================================================================

;; 块列表 + 正文宽 → 显示行列表
(define layout
  (lambda (blocks width)
    (define n (length blocks))
    (define i 0)
    (define b #f)
    (define fn #f)
    (set! cur_lines (new_list))
    (while (< i n)
      { (set! b (get_item blocks i))
        (set! fn (layout_of (get_item b 0)))
        (if fn (fn b width) #f)          ;; 不认识的块：跳过（容错，不崩）
        (set! i (+ i 1)) })
    cur_lines))

;; 全部显示行的总像素高
(define total_height
  (lambda (lines)
    (define n (length lines))
    (define i 0)
    (define h 0)
    (while (< i n)
      { (set! h (+ h (get_item (get_item lines i) 0)))
        (set! i (+ i 1)) })
    h))

;; 行的纵向起点（第 idx 行的顶部像素 y，相对正文顶端）；越界返回总高
(define line_top
  (lambda (lines idx)
    (define n (length lines))
    (define i 0)
    (define y 0)
    (define stop 0)
    (while (and (< i n) (== stop 0))
      { (if (< i idx) (set! y (+ y (get_item (get_item lines i) 0))) (set! stop 1))
        (set! i (+ i 1)) })
    y))