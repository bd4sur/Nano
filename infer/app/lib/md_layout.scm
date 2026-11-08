;; ============================================================================
;; md_layout.scm —— Markdown 排版器（块树 → 显示行）
;;
;; 运行环境：纯灵机 Scheme，只依赖 String / Math 本地库（不碰 Gfx/Fs），
;;           因此可以在桌面解释器里直接单测（tools/e2e/markdown/test_layout.scm）。
;;
;; 输入：md_parse.scm 产出的块列表；输出：**显示行列表**。
;;
;; 两种入口（同一套块排版器，语义一致）：
;;   (layout 块表 宽 度量)                     —— 整篇排版（小文档 / 单元测试 / 提示文案）
;;   (layout_segment 块表 宽 度量 尾续接 代码上续 代码下续 已画图数)
;;                                            —— **分段排版**（大文件流式阅读，见
;;                                               app/lib/md_view.scm）。后三个开关只在段与段
;;                                               之间"缝"上生效：抑制多出来的间隙与代码块
;;                                               内边距行，使被强制切开的段落/代码块画成一整块。
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
;;   9 表格竖线    10 引用块竖线   11 二级标题装饰竖线
;;   12 块级图片   13 图片占位框   14 行内图片占位文本   15 水平线段
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
(define ST_TABLE_SEP 9)   ;; 表格列间竖线（绘制端画 1px 竖条，文本为空）
(define ST_QUOTE_BAR 10)  ;; 引用块左侧竖线
(define ST_H2BAR 11)      ;; 二级标题左侧装饰竖线（#66ccff，3px）
(define ST_IMAGE 12)      ;; 块级图片（段的 4 格存图片路径，0 格存 alt）
(define ST_IMAGE_MISSING 13) ;; 图片取不到时的占位框（方案 D 兜底）
(define ST_IMAGE_TEXT 14) ;; 行内图片的降级占位文本「[图片: alt]」
(define ST_HR 15)         ;; 水平线段（宽度由段宽决定；表格上/下框线与表头下框线用它，
                          ;; 这样框线只画到表格实际宽度，而不是整幅正文宽）

(define BG_NONE 0)
(define BG_CODE 1)
(define BG_RULE 2)
(define BG_TABLE 3)

;; ---- 度量常量 ------------------------------------------------------------

;; 字模度量：默认是 12px 抗锯齿（半角 7 / 全角 12 / 行高 15）。App 换字模时用
;; (MDL.set_font_metrics 半角 全角 行高) 改这三项，再重新 layout —— 因为**折行宽度与
;; 行高都依赖字模**（16px 抗锯齿是 9/16/18，12px 点阵是 6/12/13）。
(define LINE_H 15)       ;; 行高（字模行高，含行距）
(define W_HALF 7)        ;; 半角步进
(define W_FULL 12)       ;; 全角步进

(define set_font_metrics
  (lambda (half full line_h)
    (set! W_HALF half)
    (set! W_FULL full)
    (set! LINE_H line_h)
    #t))

(define LIST_IND 14)     ;; 列表每级缩进（与项目符号的悬挂缩进一致）
(define QUOTE_IND 10)    ;; 引用块正文右移量
(define QUOTE_BAR_W 2)   ;; 引用块左侧竖线宽度
(define H2_BAR_W 3)      ;; 二级标题装饰竖线宽度
(define H2_BAR_GAP 1)    ;; 竖线与右侧文字之间的间距（1px margin）
(define IMG_MAX_H 160)      ;; 块级图片的最大高度（约 2/3 屏；超出按比例缩）
(define IMG_MISS_H 26)      ;; 图片取不到时占位框的高度
(define IMG_MAX_COUNT 24)   ;; 单篇文档最多渲染多少张图（超出退化为占位文本，防池子被吃光）
(define CJK_BASE 11904)  ;; 0x2E80
(define CP_SP 32)
(define CP_TAB 9)

;; ---- 造表小工具 ----------------------------------------------------------

(define new_list (lambda () (cdr '(0))))
(define mk1 (lambda (a) (cons a (cdr '(0)))))
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
                            (if (== tag 'image)
                                ;; 行内图片（一期）：退化为占位文本，不参与行高计算
                                (atomize_text
                                 (String.concat "[图片] "
                                                (get_item (get_item inls i) 2))
                                 ST_IMAGE_TEXT)
                                #f))))))
        (set! i (+ i 1)) })
    #t))

;; 标题里的加粗仍是标题样式（标题本身已经是粗体，无需再换色）
(define bold_style
  (lambda (base)
    (if (>= base ST_H1) base ST_BOLD)))

;; ============================================================================
;; 二、原子流 → 显示行（贪心折行）
;; ============================================================================

;; 图片度量回调：由 layout 的第三个参数注入（App 传 Gfx.image_size 的包装，
;; 单测传桩函数）。默认返回 #f —— 即"拿不到尺寸"，排版退化为占位框，
;; 因此纯逻辑单测不传也能跑。
(define measure_image (lambda (path) #f))

;; 本次 layout 实际画出的图片张数（诊断用：App 会打到 MDSTAT 里）
(define images_placed 0)

;; 已排显示行的累计高度（emit_line 维护）：目录条目记的就是"标题在文档里的 y 偏移"，
;; 点目录跳转 = 直接把这个值当 scroll 用，不需要反查行号。
(define cur_y 0)

;; 目录：本次 layout 收集到的标题（层级 1~3；4~6 级按 3 级归档），元素 = (层级 y文本 文本)。
;; 引用块里的标题不进目录（quote_depth > 0 时跳过）——那是"引文"而不是本文的结构。
(define toc_entries (new_list))
(define quote_depth 0)

;; 行内节点 → 纯文本（目录条目用；加粗/链接递归取内容，图片取 alt）
(define inlines_text
  (lambda (inls)
    (define n (length inls))
    (define i 0)
    (define out "")
    (define nd #f)
    (define tag 'text)
    (while (< i n)
      { (set! nd (get_item inls i))
        (set! tag (get_item nd 0))
        (if (== tag 'text)
            (set! out (String.concat out (get_item nd 1)))
            (if (== tag 'code)
                (set! out (String.concat out (get_item nd 1)))
                (if (== tag 'bold)
                    (set! out (String.concat out (inlines_text (get_item nd 1))))
                    (if (== tag 'link)
                        (set! out (String.concat out (inlines_text (get_item nd 1))))
                        (if (== tag 'image)
                            (set! out (String.concat out (get_item nd 2)))
                            #f)))))
        (set! i (+ i 1)) })
    out))

(define cur_lines (new_list))

;; ---- 分段渲染的"续接"支持（大文件流式阅读，见 app/lib/md_stream.scm）------
;; 段与段之间的缝本来会多出一份间隙（上一段的尾间隙 + 下一段的首间隙），
;; 对"被强制切开的一段正文/代码块"就是可见的断层。这里用两个开关抹平：
;;   seg_skip_tail : 本段**最后**一个间隙丢弃（挂起到段尾，若无后续内容则丢掉）
;;   seg_code_top  : 本段首块是"接上来"的代码块 → 跳过它的上间隙与上内边距行
;;   seg_code_bot  : 本段末块是"还要接下去"的代码块 → 跳过它的下内边距行
;; （下间隙由 seg_skip_tail 统一丢弃。）整篇排版时三者都是 0，行为与改前一致。
(define seg_skip_tail 0)
(define seg_code_top 0)
(define seg_code_bot 0)

(define emit_line
  (lambda (h bg segs)
    { (push cur_lines (mk3 h bg segs))
      (set! cur_y (+ cur_y h))
      #t }))

(define push_gap
  (lambda (h)
    (emit_line h BG_NONE (new_list))))

(define emit_gap
  (lambda (h)
    (push_gap h)
    #t))

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
;; 返回「段列表的列表」——每个元素是一行（一行里的段已按 x 排好），**不落地成显示行**。
;; 这样表格单元格可以借用同一套折行逻辑，再按列坐标平移；普通块用 emit_wrapped 落地。
(define pack_segs
  (lambda (width first_ind cont_ind)
    (define na (length atoms))
    (define k 0)
    (define x first_ind)
    (define segs (new_list))
    (define out (new_list))
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
                    { (push out segs)
                      (set! segs (new_list))
                      (set! x cont_ind)
                      (if is_sp #f
                          (set! x (place_atom segs txt st x aw url))) })
                (set! x (place_atom segs txt st x aw url))))
        (set! k (+ k 1)) })
    (if (> (length segs) 0) (push out segs) #f)
    out))

;; 折行的显示行版本：把 pack_segs 的每一行按背景落地
(define pack_atoms
  (lambda (width first_ind cont_ind bg)
    (define blocks (pack_segs width first_ind cont_ind))
    (define n (length blocks))
    (define i 0)
    (while (< i n)
      { (emit_line LINE_H bg (get_item blocks i))
        (set! i (+ i 1)) })
    #t))

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
;; 字号无法切换（Gfx 只有一套 12px alpha 字模），因此层级靠装饰表达：
;;   H1 —— 粗体 + 段前留白最多 + 2px 通栏横线；
;;   H2 —— 粗体 + **左侧 3px 装饰竖线**（ST_H2BAR，颜色由绘制端给成 #66ccff）；
;;   H3 —— 粗体 + 更暗的配色 + 文字下划线（下划线在绘制端画，见 app/markdown.scm）。
;; 竖线与文字都是"段"，绘制端不需要知道这是标题装饰。
(define lay_heading
  (lambda (b width)
    (define lvl (get_item b 1))
    (define shown 1)
    (define base ST_H1)
    (define before 0)
    (define i 0)
    (define ln #f)
    (set! shown lvl)
    (if (> shown 3) (set! shown 3) #f)
    (if (== shown 2) (set! base ST_H2) #f)
    (if (== shown 3) (set! base ST_H3) #f)
    (emit_gap (if (> lvl 1) 9 13))
    ;; 目录条目：层级取"显示层级"（4~6 级按 3 级归档），y 就是标题行的顶端偏移
    (if (== quote_depth 0)
        (push toc_entries (mk3 shown cur_y (inlines_text (get_item b 2))))
        #f)
    (set! before (length cur_lines))
    ;; H2 要给左侧竖线让出「竖线宽 + 1px 间距」（折行时同样少这么多）
    (if (== shown 2)
        (emit_wrapped (get_item b 2) (- width (+ H2_BAR_W H2_BAR_GAP)) 0 0 BG_NONE base)
        (emit_wrapped (get_item b 2) width 0 0 BG_NONE base))
    (if (== shown 2)
        { (set! i before)
          (while (< i (length cur_lines))
            { (set! ln (get_item cur_lines i))
              (relocate_segs (get_item ln 2) (+ H2_BAR_W H2_BAR_GAP))
              (set_item! ln 2 (cons (mk5 "" ST_H2BAR 0 H2_BAR_W "") (get_item ln 2)))
              (set! i (+ i 1)) }) }
        #f)
    (if (== shown 1)
        { (emit_gap 3) (emit_line 2 BG_RULE (new_list)) }
        #f)
    (emit_gap 7)))

;; ---- 段落 ----------------------------------------------------------------
;; 块形状：(para inlines) —— 行内节点在下标 1（与 heading 的 lvl 占位不同，注意）
(define lay_para
  (lambda (b width)
    (emit_wrapped (get_item b 1) width 0 0 BG_NONE ST_BODY)
    (emit_gap 7)))

;; ---- 无序列表（支持嵌套）：每层缩进 LIST_IND，项目符号悬挂 ------------------
;; item = (item inlines children)：先排本项的文本（项目符号 14px 悬挂），再递归排 children。
;; 层级只影响缩进与折行右边界；绘制端依旧只认"段"。
(define lay_items
  (lambda (items indent width)
    (define n (length items))
    (define i 0)
    (define it #f)
    (define inl #f)
    (while (< i n)
      { (set! it (get_item items i))
        (set! inl (cons (mk2 'bullet #f) (get_item it 1)))
        (emit_wrapped inl width indent (+ indent LIST_IND) BG_NONE ST_BODY)
        (if (> (length (get_item it 2)) 0)
            (lay_items (get_item it 2) (+ indent LIST_IND) width)
            #f)
        (if (< (+ i 1) n) (emit_gap 3) #f)
        (set! i (+ i 1)) })
    #t))

(define lay_list
  (lambda (b width)
    (lay_items (get_item b 1) 0 width)
    (emit_gap 7)))

;; ---- 引用块 --------------------------------------------------------------
;; 做法：**先把引用树摊平**成 (层号 块) 序列，再逐块排版、逐块右移并插竖线。
;; 为什么不写"递归排版（lay_quote 里再调 lay_quote）"：灵机里同一闭包被重入时共用帧，
;; 嵌套引用会让外层 lay_quote 的 before/i 被内层覆盖（行会重复/错位）。
;; 摊平用的是**普通自递归**（flatten_quote，直接自递归会新建闭包，帧独立），
;; 摊平后每次排版调用的都是别的块排版函数，不再重入 lay_quote。
(define flatten_quote
  (lambda (blocks depth out)
    (define n (length blocks))
    (define i 0)
    (define b #f)
    (while (< i n)
      { (set! b (get_item blocks i))
        (if (== (get_item b 0) 'quote)
            (flatten_quote (get_item b 1) (+ depth 1) out)
            (push out (mk3 depth b 0)))      ;; 第三格留白（保持三元组形状一致）
        (set! i (+ i 1)) })
    out))

;; 排一个已摊平的 (层号 块)：按层号右移 + 插 depth 条竖线
(define lay_quote_items
  (lambda (flat width)
    (define n (length flat))
    (define i 0)
    (define depth 0)
    (define b #f)
    (define before 0)
    (define ln #f)
    (define k 0)
    (while (< i n)
      { (set! depth (get_item (get_item flat i) 0))
        (set! b (get_item (get_item flat i) 1))
        (set! before (length cur_lines))
        (if (layout_of (get_item b 0))
            ((layout_of (get_item b 0)) b (- width (* depth QUOTE_IND)))
            #f)
        (set! k before)
        (while (< k (length cur_lines))
          { (set! ln (get_item cur_lines k))
            (relocate_segs (get_item ln 2) (* depth QUOTE_IND))
            ;; 从最外层到最内层依次插竖线（先插的 x 小）
            (set! depth depth)          ;; 占位：保持 depth 值不变
            (insert_quote_bars ln (get_item (get_item flat i) 0))
            (set! k (+ k 1)) })
        (set! i (+ i 1)) })
    #t))

;; 给一行插 depth 条竖线（x = 0, QUOTE_IND, 2*QUOTE_IND ...）
(define insert_quote_bars
  (lambda (ln depth)
    (define d 0)
    (define segs #f)
    (while (< d depth)
      { (set! segs (get_item ln 2))
        (set_item! ln 2 (cons (mk5 "" ST_QUOTE_BAR (* d QUOTE_IND) QUOTE_BAR_W "")
                              segs))
        (set! d (+ d 1)) })
    #t))

(define lay_quote
  (lambda (b width)
    (define flat (new_list))
    (emit_gap 7)
    (flatten_quote (get_item b 1) 1 flat)
    (if (== (length flat) 0)
        (emit_line LINE_H BG_NONE (cons (mk5 "" ST_QUOTE_BAR 0 QUOTE_BAR_W "")
                                        (new_list)))   ;; 空引用也给一行 + 一条竖线
        { ;; 排引用内容期间置 quote_depth：里面的标题不进目录（见 lay_heading）
          (set! quote_depth (+ quote_depth 1))
          (lay_quote_items flat width)
          (set! quote_depth (- quote_depth 1)) })
    (emit_gap 7)))

;; ---- 围栏代码块 ----------------------------------------------------------
(define lay_code
  (lambda (b width)
    (define lns (get_item b 2))
    (define n (length lns))
    (define i 0)
    ;; 分段续接：首块是"接上来"的代码块时，不再留上间隙与上内边距（缝上无缝拼接）
    (if (== seg_code_top 1)
        (set! seg_code_top 0)
        { (emit_gap 6)
          (emit_line LINE_H BG_CODE (new_list)) })   ;; 上内边距
    (while (< i n)
      { (emit_code_line (get_item lns i) width)
        (set! i (+ i 1)) })
    (if (== seg_code_bot 1)
        (set! seg_code_bot 0)
        (emit_line LINE_H BG_CODE (new_list)))       ;; 下内边距
    (emit_gap 8)))

;; 代码块的一行：等宽字模，超出正文宽就续行；制表符按 4 个空格。
;;
;; 实现要点（性能）：**按"能放下的最长片段"整段落位**，而不是每字符都
;; `String.concat` 一次——后者对长代码行是 O(n²) 的分配（一行 50 字符 ≈ 50 次
;; 增长串拼接），实测在"16KB 巨型代码块"这类病态输入上会白烧掉数 MB 临时对象。
;; 与逐字符版渲染结果**逐段等价**（place_atom 本来就会把相邻同样式同位置的文本合并）。
(define flush_code_chunk
  (lambda (segs s a b x w)
    (if (> b a) (place_atom segs (String.slice s a b) ST_CODEBLOCK x w "") #f)
    #t))

(define emit_code_line
  (lambda (s width)
    (define n (String.length s))
    (define i 0)
    (define cp 0)
    (define segs (new_list))
    (define x 6)
    (define lim (- width 8))
    (define cw 0)
    (define start 0)      ;; 当前片段的起始下标
    (define x0 0)         ;; 当前片段的起始 x
    (define cwsum 0)      ;; 当前片段累计宽度
    (set! x0 x)
    (while (< i n)
      { (set! cp (String.charCodeAt i s))
        (if (== cp CP_TAB)
            { (flush_code_chunk segs s start i x0 cwsum)
              (set! cw (+ W_HALF (+ W_HALF (+ W_HALF W_HALF))))
              (if (> (+ x cw) lim)
                  { (emit_line LINE_H BG_CODE segs)
                    (set! segs (new_list))
                    (set! x 6) }
                  #f)
              (set! x (place_atom segs "    " ST_CODEBLOCK x cw ""))
              (set! i (+ i 1))
              (set! start i)
              (set! x0 x)
              (set! cwsum 0) }
            { (set! cw (char_w cp))
              (if (> (+ x cw) lim)
                  { (flush_code_chunk segs s start i x0 cwsum)
                    (emit_line LINE_H BG_CODE segs)
                    (set! segs (new_list))
                    (set! x 6)
                    (set! start i)
                    (set! x0 6)
                    (set! cwsum 0) }
                  #f)
              (set! x (+ x cw))
              (set! cwsum (+ cwsum cw))
              (set! i (+ i 1)) }) })
    (flush_code_chunk segs s start n x0 cwsum)
    (emit_line LINE_H BG_CODE segs)))

;; ---- 分隔线 --------------------------------------------------------------
(define lay_rule
  (lambda (b width)
    (emit_gap 8)
    (emit_line 1 BG_RULE (new_list))
    (emit_gap 9)))

;; ---- 表格 ---------------------------------------------------------------
;;
;; 输入 (table aligns header cells)：
;;   aligns —— 每列对齐码（0 左 / 1 中 / 2 右，来自分隔行的冒号）
;;   header —— 表头**一行**的单元格列表
;;   cells  —— 数据行列表，每行是单元格列表
;;   单元格 —— 行内节点列表（所以单元格里可以再写加粗/行内代码/链接）
;;
;; 排版策略（"先量后分"）：
;;   ① 每列取所有单元格的**自然宽**（该格内容一行排开需要多少像素），夹在
;;      [TABLE_MINW, TABLE_MAXW] 里——太窄会挤成一条竖排，太宽会独占整行；
;;   ② 自然宽总和放得下就用自然宽（表格只占需要的宽度，不硬拉到通栏）；
;;      放不下就按比例压缩到正文宽，每列不小于下限，再对余数做微调；
;;   ③ 单元格内容用与正文**同一套折行**（pack_segs）排成若干子行，
;;      行高 = 该行各列子行数的最大值（长单元格自然把行撑高）；
;;   ④ 每行按列坐标 + 对齐平移段；列间画 1px 竖线（TABLE_GRID），
;;      表格上/下、表头下、以及**每两行之间**都画 1px 横线（全网格）。
;;      横线宽度 = 表格实际宽度（用 ST_HR 段表达，不是满宽的 BG_RULE 行）。
;;      绘制端只认"段"，不知道这是表格。

(define TABLE_MINW 26)     ;; 单列最小宽（太窄汉字挤成竖排）
(define TABLE_MAXW 120)    ;; 单列最大宽（避免一列吃掉整行）
(define TABLE_GAP 10)      ;; 列间空隙（竖线画在空隙中央）
(define TABLE_PAD 2)       ;; 单元格左右内缩（文字不贴着竖线）
(define TABLE_GRID 1)      ;; 1 = 画列间竖线；0 = 只靠对齐 + 表头横线

;; 行内节点列表的"一行排开"宽度（忽略换行，只加文本宽）
(define inl_width
  (lambda (inls)
    (define n (length inls))
    (define i 0)
    (define w 0)
    (define nd #f)
    (define tag 'text)
    (while (< i n)
      { (set! nd (get_item inls i))
        (set! tag (get_item nd 0))
        (if (== tag 'text)
            (set! w (+ w (text_w (get_item nd 1))))
            (if (== tag 'code)
                (set! w (+ w (text_w (get_item nd 1))))
                (if (== tag 'bold)
                    (set! w (+ w (inl_width (get_item nd 1))))
                    (if (== tag 'link)
                        (set! w (+ w (inl_width (get_item nd 1))))
                        #f))))
        (set! i (+ i 1)) })
    w))

;; 表格列数 = 表头与所有数据行里最大的单元格个数
(define table_cols
  (lambda (b)
    (define header (get_item b 2))
    (define rows (get_item b 3))
    (define n (length header))
    (define i 0)
    (define w 0)
    (while (< i (length rows))
      { (set! w (length (get_item rows i)))
        (if (> w n) (set! n w) #f)
        (set! i (+ i 1)) })
    (if (== n 0) 1 n)))

;; 取某行的第 i 格（越界给空行内列表）
(define cell_at
  (lambda (cells i)
    (if (< i (length cells)) (get_item cells i) (cdr '(0)))))

;; 取第 i 列的对齐码（越界按左对齐）
(define align_at
  (lambda (aligns i)
    (if (< i (length aligns)) (get_item aligns i) 0)))

;; 刷新第 i 列的最大单元格自然宽到 acc（用列表做累加器避免在循环里 define）
(define acc_row_widths
  (lambda (cells acc ncol)
    (define i 0)
    (define w 0)
    (while (< i ncol)
      { (set! w (inl_width (cell_at cells i)))
        (if (> w (get_item acc i)) (set_item! acc i w) #f)
        (set! i (+ i 1)) })
    #t))

;; 各列自然宽（夹到 [MINW, MAXW]）
(define table_natural
  (lambda (b ncol)
    (define acc (new_list))
    (define rows (get_item b 3))
    (define i 0)
    (define w 0)
    (define hw 0)
    (while (< i ncol)
      { (push acc 0) (set! i (+ i 1)) })
    (acc_row_widths (get_item b 2) acc ncol)
    (set! i 0)
    (while (< i (length rows))
      { (acc_row_widths (get_item rows i) acc ncol)
        (set! i (+ i 1)) })
    (set! i 0)
    (while (< i ncol)
      { (set! w (+ (get_item acc i) 2))          ;; 左右各留 1px 余量
        (if (< w TABLE_MINW) (set! w TABLE_MINW) #f)
        (if (> w TABLE_MAXW) (set! w TABLE_MAXW) #f)
        (set_item! acc i w)
        (set! i (+ i 1)) })
    acc))

(define sum_list
  (lambda (lst)
    (define n (length lst))
    (define i 0)
    (define s 0)
    (while (< i n)
      { (set! s (+ s (get_item lst i)))
        (set! i (+ i 1)) })
    s))

;; 超出可用宽时从最宽列往下削（每轮削 1px，削到下限为止）
(define shrink_to_fit
  (lambda (widths avail ncol)
    (define guard 0)
    (define widest 0)
    (define wi 0)
    (define i 0)
    (define w 0)
    (while (and (> (sum_list widths) avail) (< guard 4000))
      { (set! widest 0) (set! wi -1) (set! i 0)
        (while (< i ncol)
          { (set! w (get_item widths i))
            (if (> w widest)
                { (set! widest w) (set! wi i) }
                #f)
            (set! i (+ i 1)) })
        (if (>= wi 0)
            (if (> widest TABLE_MINW)
                (set_item! widths wi (- widest 1))
                (set! guard 4000))                     ;; 全部到下限：放弃（极端窄屏多列）
            (set! guard 4000))
        (set! guard (+ guard 1)) })
    widths))

;; 列宽表
(define table_widths
  (lambda (b ncol width)
    (define avail (- width (* TABLE_GAP (- ncol 1))))
    (define nat (table_natural b ncol))
    (define total (sum_list nat))
    (define out (new_list))
    (define i 0)
    (define w 0)
    (define minw TABLE_MINW)
    (if (<= total avail)
        nat                                        ;; 放得下：用自然宽
        { (if (< (Math.floor (/ avail ncol)) minw)
              (set! minw (Math.floor (/ avail ncol)))
              #f)
          (set! i 0)
          (while (< i ncol)
            { (set! w (Math.floor (/ (* (get_item nat i) avail) total)))
              (if (< w minw) (set! w minw) #f)
              (push out w)
              (set! i (+ i 1)) })
          (shrink_to_fit out avail ncol) })))

;; 各列左边缘 x
(define table_xs
  (lambda (widths ncol)
    (define out (new_list))
    (define i 0)
    (define x 0)
    (while (< i ncol)
      { (push out x)
        (set! x (+ x (+ (get_item widths i) TABLE_GAP)))
        (set! i (+ i 1)) })
    out))

;; 单元格内容 → 若干子行（每子行是一组段，x 从 0 起）
(define cell_blocks
  (lambda (inls w base)
    (set! atoms (new_list))
    (set! atom_url "")
    (set! atom_avail w)
    (atomize inls base)
    (pack_segs w 0 0)))

;; 一行段的右边缘（= 内容宽）
(define blk_width
  (lambda (segs)
    (define n (length segs))
    (define i 0)
    (define e 0)
    (define r 0)
    (while (< i n)
      { (set! e (+ (get_item (get_item segs i) 2) (get_item (get_item segs i) 3)))
        (if (> e r) (set! r e) #f)
        (set! i (+ i 1)) })
    r))

;; 按对齐方式算平移量（slack = 列宽 - 内容宽）
(define align_shift
  (lambda (align slack)
    (if (<= slack 0) 0
        (if (== align 2) slack
            (if (== align 1) (Math.floor (/ slack 2)) 0)))))

;; 把一组段的 x 整体平移 dx
(define relocate_segs
  (lambda (segs dx)
    (define n (length segs))
    (define i 0)
    (define sg #f)
    (while (< i n)
      { (set! sg (get_item segs i))
        (set_item! sg 2 (+ (get_item sg 2) dx))
        (set! i (+ i 1)) })
    #t))

(define append_segs
  (lambda (dst src)
    (define n (length src))
    (define i 0)
    (while (< i n)
      { (push dst (get_item src i))
        (set! i (+ i 1)) })
    #t))

;; 列间竖线：段文本为空、宽 1，绘制端按 ST_TABLE_SEP 画 1px 竖条
(define append_grid
  (lambda (segs xs ncol)
    (define i 1)
    (define x 0)
    (while (< i ncol)
      { (set! x (- (get_item xs i) (Math.floor (/ (+ TABLE_GAP 1) 2))))
        (if (> x 0) (push segs (mk5 "" ST_TABLE_SEP x 1 "")) #f)
        (set! i (+ i 1)) })
    #t))

;; 逻辑行 → 若干显示行（各列先各自折行，再按键号拼）
(define emit_table_row
  (lambda (cells widths xs aligns ncol bg base)
    (define blocks (new_list))
    (define i 0)
    (define maxl 0)
    (define k 0)
    (define segs (new_list))
    (define bl #f)
    (define w 0)
    (define shift 0)
    (set! i 0)
    (while (< i ncol)
      { (push blocks (cell_blocks (cell_at cells i) (get_item widths i) base))
        (set! i (+ i 1)) })
    (set! i 0)
    (while (< i ncol)
      { (if (> (length (get_item blocks i)) maxl)
            (set! maxl (length (get_item blocks i)))
            #f)
        (set! i (+ i 1)) })
    (set! k 0)
    (while (< k maxl)
      { (set! segs (new_list))
        (set! i 0)
        (while (< i ncol)
          { (set! bl (get_item blocks i))
            (if (< k (length bl))
                { (set! w (blk_width (get_item bl k)))
                  (set! shift (align_shift (align_at aligns i)
                                           (- (- (get_item widths i)
                                                 (* 2 TABLE_PAD))
                                              w)))
                  (relocate_segs (get_item bl k)
                                 (+ (+ (get_item xs i) TABLE_PAD) shift))
                  (append_segs segs (get_item bl k)) }
                #f)
            (set! i (+ i 1)) })
        (if (== TABLE_GRID 1) (append_grid segs xs ncol) #f)
        (emit_line LINE_H bg segs)
        (set! k (+ k 1)) })
    #t))

;; 表格实际宽度 = 各列宽之和 + 列间距
(define table_span
  (lambda (widths ncol)
    (+ (sum_list widths) (* TABLE_GAP (- ncol 1)))))

;; 画一条"只到表格宽度"的横线（1px 高的一行，里面只有一个 ST_HR 段）
(define emit_span_hr
  (lambda (w)
    (emit_line 1 BG_NONE (mk1 (mk5 "" ST_HR 0 w "")))))

(define lay_table
  (lambda (b width)
    (define aligns (get_item b 1))
    (define ncol (table_cols b))
    (define widths (table_widths b ncol width))
    (define xs (table_xs widths ncol))
    (define rows (get_item b 3))
    (define span (table_span widths ncol))
    (define i 0)
    (emit_gap 7)
    (emit_span_hr span)                       ;; ── 上框线
    (emit_gap 2)
    (emit_table_row (get_item b 2) widths xs aligns ncol BG_TABLE ST_BOLD)
    (emit_span_hr span)                       ;; ── 表头下框线
    (set! i 0)
    (while (< i (length rows))
      { (emit_table_row (get_item rows i) widths xs aligns ncol BG_NONE ST_BODY)
        ;; ── 行间框线：除最后一行外，每行（含被单元格折行撑高的逻辑行）之后画一条
        (if (< (+ i 1) (length rows))
            (emit_span_hr span)
            #f)
        (set! i (+ i 1)) })
    (emit_span_hr span)                       ;; ── 下框线
    (emit_gap 9)))

;; ---- 图片（块级）---------------------------------------------------------
;; 一期：块级图片独占一行、水平居中、等比缩放进 (正文宽, IMG_MAX_H) 的框里。
;; 行高 = 图片高（渲染端用行高当图片高来画，见 app/markdown.scm 的 ST_IMAGE 分支）。
;; 拿不到尺寸（文件缺失/格式不支持/超过张数上限）时产出占位框（方案 D 兜底）。
;;
;; 段的 5 格用法与普通文本不同：0 格 = alt 文本（占位框里显示），
;; 4 格 = 图片路径（渲染端拿去调 (Gfx.draw_image ...)）。
(define fit_image
  (lambda (sw sh maxw maxh)
    (define w sw)
    (define h sh)
    (if (> sw maxw)
        { (set! h (Math.floor (/ (* sh maxw) sw)))
          (set! w maxw) }
        #f)
    (if (> h maxh)
        { (set! w (Math.floor (/ (* w maxh) h)))
          (set! h maxh) }
        #f)
    (if (< w 1) (set! w 1) #f)
    (if (< h 1) (set! h 1) #f)
    (mk2 w h)))

(define emit_image_placeholder
  (lambda (path alt width)
    (emit_line IMG_MISS_H BG_NONE
               (mk1 (mk5 alt ST_IMAGE_MISSING 0 width path)))))

(define lay_image
  (lambda (b width)
    (define path (get_item b 1))
    (define alt (get_item b 2))
    (define dims #f)
    (define sw 0)
    (define sh 0)
    (define fit #f)
    (define w 0)
    (define h 0)
    (emit_gap 7)
    (if (>= images_placed IMG_MAX_COUNT)
        (emit_image_placeholder path alt width)   ;; 超上限：不再解码绘制
        { (set! dims (measure_image path))
          ;; 注意：#null 在本解释器里是**真值**（只有 #f 为假），量不到尺寸时
          ;; 度量回调可能返回 #null，这里必须用 list? 判定，不能只判真假。
          (if (list? dims)
              { (set! sw (get_item dims 0))
                (set! sh (get_item dims 1))
                (set! fit (fit_image sw sh width IMG_MAX_H))
                (set! w (get_item fit 0))
                (set! h (get_item fit 1))
                (emit_line h BG_NONE
                           (mk1 (mk5 alt ST_IMAGE
                                     (Math.floor (/ (- width w) 2))
                                     w path)))
                (set! images_placed (+ images_placed 1)) }
              (emit_image_placeholder path alt width)) })
    (emit_gap 8)))

;; ---- 注册块排版（顺序无关，按种类查表）----------------------------------

(layout_add 'heading lay_heading)
(layout_add 'para lay_para)
(layout_add 'list lay_list)
(layout_add 'code lay_code)
(layout_add 'rule lay_rule)
(layout_add 'table lay_table)
(layout_add 'quote lay_quote)
(layout_add 'image lay_image)

;; ============================================================================
;; 四、对外 API
;; ============================================================================

;; 块列表 + 正文宽 + 图片度量函数 → 显示行列表
;;   measure: (measure path) → (宽 高) | #f（App 传 Gfx.image_size 的包装；
;;            纯逻辑测试传桩函数或干脆传 #f —— 传 #f 时所有图片都按占位框渲染）
(define layout
  (lambda (blocks width measure)
    (layout_full blocks width measure 0 0 0 0)))

;; 整篇排版（原有语义，等价于 layout_segment 的三个续接开关都取 0）
(define layout_full
  (lambda (blocks width measure skip_tail code_top code_bot img_done)
    (define n (length blocks))
    (define i 0)
    (define b #f)
    (define fn #f)
    (set! cur_lines (new_list))
    (set! images_placed img_done)
    (set! cur_y 0)
    (set! toc_entries (new_list))
    (set! quote_depth 0)
    (set! seg_skip_tail skip_tail)
    (set! seg_code_top code_top)
    (set! seg_code_bot code_bot)
    ;; 容错：调用方漏传第三个参数时 measure 是 #undefined（**不是 #f**，而 #undefined 为真），
    ;; 直接当真会变成"调用 #undefined"报错，因此显式判 undefined? 后退化为占位框。
    (set! measure_image (if (undefined? measure) (lambda (path) #f) measure))
    (while (< i n)
      { (set! b (get_item blocks i))
        (set! fn (layout_of (get_item b 0)))
        (if fn (fn b width) #f)          ;; 不认识的块：跳过（容错，不崩）
        (set! i (+ i 1)) })
    ;; 段尾续接：弹掉本段最后一个"纯间隙行"（= 最后一个块的后间隙），使缝上无缝。
    ;; 用"弹"而不是"挂起"：挂起会在 emit_line 内部插入行，破坏 lay_heading 这类
    ;; 「先记 before、再回头改这段行」的块排版器的行下标（踩过）。
    (if (and (== skip_tail 1) (> (length cur_lines) 0))
        { (set! b (get_item cur_lines (- (length cur_lines) 1)))
          (if (and (== (get_item b 1) BG_NONE) (== (length (get_item b 2)) 0))
              { (set! cur_y (- cur_y (get_item b 0)))
                (pop cur_lines) }
              #f) }
        #f)
    (set! seg_skip_tail 0)
    cur_lines))

;; 分段排版入口（大文件流式阅读）：
;;   skip_tail —— 丢弃本段最后一个间隙（正文/列表/表格/引用被强制切开时抹平缝）
;;   code_top  —— 本段首块是"接上来"的代码块（跳过上间隙与上内边距）
;;   code_bot  —— 本段末块是"还要接下去"的代码块（跳过下内边距）
;;   img_done  —— 已画出的图片数（IMG_MAX_COUNT 按整篇而不是按段计数）
(define layout_segment
  (lambda (blocks width measure skip_tail code_top code_bot img_done)
    (layout_full blocks width measure skip_tail code_top code_bot
                 (if (undefined? img_done) 0 img_done))))

;; 本次 layout 实际画进文档的图片张数（诊断/测试用）
(define image_count
  (lambda ()
    images_placed))

;; 本次 layout 收集到的目录（元素 = (层级 y偏移 文本)；层级 1~3）
(define toc
  (lambda ()
    toc_entries))

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