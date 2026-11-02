;; ============================================================================
;; md_parse.scm —— Markdown 解析器（纯灵机 Scheme 实现）
;;
;; 运行环境：灵机（Animac）解释器，只依赖 String / Math 两个本地库。
;;   · 不碰 Gfx / Fs / Touch —— 因此**可以在桌面解释器（./main）里直接单测**，
;;     不需要 Nano GUI、不需要浏览器（见 tools/e2e/markdown/test_parse.scm）。
;;   · 解析产物是「块列表」，每个块的首元素是种类符号：
;;
;;        (heading 层级 inlines)      标题，层级 1..6
;;        (para inlines)              段落
;;        (list items)                无序列表（可嵌套）；每个 item = (item inlines children)，
;;                                    children 是更深一层的 item 列表
;;        (quote blocks)              引用块（> 开头），内层是任意块（可以再嵌套引用/列表/代码块）
;;                                    ——(quote-raw 行数组) 只是解析中间产物，对外 API 里不会出现
;;        (code 语言 lines)           围栏代码块，lines 为原始行字符串（不做行内解析）
;;        (rule)                      分隔线（--- / *** / ___）
;;        (table aligns header cells) 表格：aligns = 每列对齐方式，header = 表头单元格，
;;                                    cells = 数据行（行 = 单元格列表），
;;                                    每个单元格是**行内节点列表**（单元格里可再写加粗/代码/链接）
;;
;;     行内节点（inlines 的元素）：
;;        (text "字符串")             纯文本
;;        (bold inlines)              **加粗** / __加粗__
;;        (code "字符串")             `行内代码`
;;        (link inlines "url")        [文本](url)
;;
;; 设计要点（为「完整 Markdown」预留的三个扩展点）
;; ---------------------------------------------------------------------------
;; ① 块级规则表 `block_rules`：每行 (种类 谓词 处理函数)。
;;    · 谓词      (lines 下标) → #t/#f     —— 从这一行**开始**是否属于该种块
;;    · 处理函数  (lines 起始下标) → (块 下一行下标)
;;    · 未命中任何规则的行走 "段落" 兜底（段落自己按谓词判断在哪里收尾）。
;;    新增块类型 = (rule_add 种类 谓词 处理函数) 一行，**不必改解析主循环**；
;;    顺序即优先级（谓词可以重叠，先注册者先匹配）。
;;    谓词带行号而不是只带单行内容：表格这类块必须**前瞻**才能判定（要看下一行是不是
;;    |---|---| 分隔行），只给单行就只能靠"先猜再回退"，会把普通段落里的一个 '|' 切断。
;;
;; ② 行内规则表 `inline_rules`：每行 (触发字符 匹配函数)。
;;    · 匹配函数  (s 下标) → #f 或 (节点 下一下标)   —— 不匹配时必须返回 #f
;;    新增行内语法（斜体、图片、脚注……）= (inline_rule_add 字符 匹配函数) 一行。
;;    未命中任何规则时该字符进纯文本缓冲，因此**任何字符都可以安全作触发符**。
;;
;; ③ 解析器与排版/渲染完全解耦：本模块只产出上面的块树，不做折行、不产像素。
;;    排版在 md_layout.scm，绘制在 app/markdown.scm。因此新增块类型时，
;;    「解析」「排版」可以分别推进、分别单测。
;;
;; 解释器约束（manual.md，逐条遵守）
;; ---------------------------------------------------------------------------
;; · 算术固定二参数；取负写 (- 0 x)；/ 与 mod 恒返回浮点（本文件不用 / 与 mod 取下标）；
;; · 无 list/append/map/let/cond 多表达式；define 无过程简写；
;; · '() 字面量是**模块级静态槽**（反复 push 会跨调用累积），空表一律用 (cdr '(0)) 现造；
;; · push 返回 #undefined，不能写成 (set! x (push x v))；
;; · while 循环体不是独立词法区域：循环变量都在函数体顶层 define（每次调用独立）。
;; ============================================================================

(native String)
(native Math)

;; ---- 码点常量 ------------------------------------------------------------

(define CP_TAB 9)
(define CP_NL 10)
(define CP_CR 13)
(define CP_SP 32)
(define CP_HASH 35)
(define CP_LPAR 40)
(define CP_RPAR 41)
(define CP_STAR 42)
(define CP_PLUS 43)
(define CP_DASH 45)
(define CP_COLON 58)
(define CP_GT 62)
(define CP_LBRK 91)
(define CP_BSLASH 92)
(define CP_RBRK 93)
(define CP_UNDER 95)
(define CP_TICK 96)
(define CP_PIPE 124)
(define CP_TILDE 126)

;; 全角/宽字符起点（0x2E80）。与 12px 字模一致：>= 此值者宽 12px，否则 7px。
(define CJK_BASE 11904)

;; ---- 小工具：造表 --------------------------------------------------------
;; 注意：'() 是模块级静态槽，凡是要 push 的表都必须现场造一个新的空表。

(define new_list (lambda () (cdr '(0))))
(define mk1 (lambda (a) (cons a (cdr '(0)))))
(define mk2 (lambda (a b) (cons a (cons b (cdr '(0))))))
(define mk3 (lambda (a b c) (cons a (cons b (cons c (cdr '(0)))))))
(define mk4 (lambda (a b c d) (cons a (cons b (cons c (cons d (cdr '(0))))))))

;; ---- 字符级判定 ----------------------------------------------------------

;; 空白（空格 / 制表符 / 回车）：行首缩进与掐头去尾都用它
(define is_ws
  (lambda (cp)
    (or (== cp CP_SP) (or (== cp CP_TAB) (== cp CP_CR)))))

;; s[i] 是否等于码点 cp（越界返回 #f）
(define char_is
  (lambda (s i cp)
    (if (< i (String.length s))
        (== (String.charCodeAt i s) cp)
        #f)))

;; 前导空白个数
;; 注意：循环**不能**用 (set! i n) 来"跳出"——那会把结果 i 一起毁掉（踩过一次），
;;       统一用 go 标志控制循环条件。
(define lead_ws
  (lambda (s)
    (define i 0)
    (define n (String.length s))
    (define go 1)
    (while (and (< i n) (== go 1))
      { (if (is_ws (String.charCodeAt i s)) (set! i (+ i 1)) (set! go 0)) })
    i))

;; 掐头去尾空白
(define trim
  (lambda (s)
    (define n (String.length s))
    (define a 0)
    (define b 0)
    (set! b n)
    (while (and (< a b) (is_ws (String.charCodeAt a s))) (set! a (+ a 1)))
    (while (and (> b a) (is_ws (String.charCodeAt (- b 1) s))) (set! b (- b 1)))
    (if (>= a b) "" (String.slice s a b))))

;; 去尾部 \r（CRLF 文本）
(define chomp_cr
  (lambda (s)
    (define n (String.length s))
    (if (> n 0)
        (if (== (String.charCodeAt (- n 1) s) CP_CR) (String.slice s 0 (- n 1)) s)
        s)))

;; 从 from 起查找字符 cp，找不到返回 -1
(define find_char
  (lambda (s from cp)
    (define i from)
    (define n (String.length s))
    (define r -1)
    (while (and (< i n) (== r -1))
      { (if (== (String.charCodeAt i s) cp) (set! r i) #f)
        (set! i (+ i 1)) })
    r))

;; 从 from 起查找相邻两字符 c1 c2（如 "**"），找不到返回 -1
(define find_pair
  (lambda (s from c1 c2)
    (define i from)
    (define n (String.length s))
    (define r -1)
    (while (and (< i (- n 1)) (== r -1))
      { (if (== (String.charCodeAt i s) c1)
            (if (== (String.charCodeAt (+ i 1) s) c2) (set! r i) #f)
            #f)
        (set! i (+ i 1)) })
    r))

;; 两个字符串相拼接：两侧都是全角（CJK）时直接相连，否则补一个空格（软换行语义）
(define soft_join
  (lambda (a b)
    (define la (String.length a))
    (if (== la 0) b
        (if (== (String.length b) 0) a
            (if (>= (String.charCodeAt (- la 1) a) CJK_BASE)
                (String.concat a b)
                (String.concat a (String.concat " " b)))))))

;; 去掉行首最多 k 个空格（围栏代码块的内容缩进对齐）
(define strip_lead
  (lambda (s k)
    (define n (String.length s))
    (define i 0)
    (define go 1)
    (while (and (< i k) (and (< i n) (== go 1)))
      { (if (== (String.charCodeAt i s) CP_SP) (set! i (+ i 1)) (set! go 0)) })
    (String.slice s i n)))

;; ---- 行切分 --------------------------------------------------------------

;; 整段文本 → 行列表（UTF-8 已由解释器解码为字符；去掉行尾 \r）
(define split_lines
  (lambda (text)
    (define n (String.length text))
    (define i 0)
    (define start 0)
    (define out (new_list))
    (while (< i n)
      { (if (== (String.charCodeAt i text) CP_NL)
            { (push out (chomp_cr (String.slice text start i)))
              (set! start (+ i 1)) }
            #f)
        (set! i (+ i 1)) })
    (push out (chomp_cr (String.slice text start n)))
    out))

;; 行列表 → 整段文本（供单测与「文档预览/复制」用）
(define join_lines
  (lambda (lines)
    (define n (length lines))
    (define i 0)
    (define out "")
    (while (< i n)
      { (if (> i 0) (set! out (String.concat out "\n")) #f)
        (set! out (String.concat out (get_item lines i)))
        (set! i (+ i 1)) })
    out))

;; ============================================================================
;; 一、行内解析（扩展点②：行内规则表）
;; ============================================================================

;; 行内规则表：每行 (触发字符 匹配函数)。顺序即优先级。
(define inline_rules (new_list))

(define inline_rule_add
  (lambda (ch matcher)
    (push inline_rules (mk2 ch matcher))))

;; 取触发字符 ch 的匹配函数；无则 #f
(define inline_matcher_of
  (lambda (ch)
    (define n (length inline_rules))
    (define i 0)
    (define r #f)
    (while (and (< i n) (not r))
      { (if (== (get_item (get_item inline_rules i) 0) ch)
            (set! r (get_item (get_item inline_rules i) 1))
            #f)
        (set! i (+ i 1)) })
    r))

;; ---- 行内匹配函数 --------------------------------------------------------

;; `代码`：不闭合则当普通字符
(define m_code
  (lambda (s i)
    (define j -1)
    (define c "")
    (if (char_is s i CP_TICK)
        { (set! j (find_char s (+ i 1) CP_TICK))
          (if (< j 0) #f
              { (set! c (String.slice s (+ i 1) j))
                (mk2 (mk2 'code c) (+ j 1)) }) }
        #f)))

;; **加粗** / __加粗__：开闭标记相同；内容递归做行内解析
(define bold_at
  (lambda (s i ch)
    (define j -1)
    (define content "")
    (if (char_is s i ch)
        (if (char_is s (+ i 1) ch)
            { (set! j (find_pair s (+ i 2) ch ch))
              (if (< j 0) #f
                  { (set! content (String.slice s (+ i 2) j))
                    ;; 注意：行内节点列表必须**整表**作为一个元素（(bold <节点列表>)），
                    ;; 不能写成 (cons 'bold 节点列表)——那会把节点摊平，下游按"元素 1 是
                    ;; 节点列表"取值时拿到的就是单个节点，后续 get_item 立刻越界报错。
                    (mk2 (mk2 'bold (parse_inlines content)) (+ j 2)) }) }
            #f)
        #f)))

(define m_bold_star (lambda (s i) (bold_at s i CP_STAR)))
(define m_bold_under (lambda (s i) (bold_at s i CP_UNDER)))

;; [文本](url)
(define m_link
  (lambda (s i)
    (define j -1)
    (define k -1)
    (define t "")
    (define u "")
    (if (char_is s i CP_LBRK)
        { (set! j (find_char s (+ i 1) CP_RBRK))
          (if (< j 0) #f
              (if (char_is s (+ j 1) CP_LPAR)
                  { (set! k (find_char s (+ j 2) CP_RPAR))
                    (if (< k 0) #f
                        { (set! t (String.slice s (+ i 1) j))
                          (set! u (String.slice s (+ j 2) k))
                          (mk2 (mk3 'link (parse_inlines t) u) (+ k 1)) }) }
                  #f)) }
        #f)))

;; \* 之类转义：反斜杠后的字符按字面收进文本
(define m_escape
  (lambda (s i)
    (define n (String.length s))
    (if (char_is s i CP_BSLASH)
        (if (< (+ i 1) n)
            (mk2 (mk2 'text (String.charAt s (+ i 1))) (+ i 2))
            #f)
        #f)))

;; 注册行内规则（顺序即优先级）
(inline_rule_add CP_TICK m_code)
(inline_rule_add CP_STAR m_bold_star)
(inline_rule_add CP_UNDER m_bold_under)
(inline_rule_add CP_LBRK m_link)
(inline_rule_add CP_BSLASH m_escape)

;; ---- 行内解析主循环 ------------------------------------------------------

;; 字符串 → 行内节点列表。未命中任何规则的字符进文本缓冲（相邻字符合并成一个节点）。
(define parse_inlines
  (lambda (s)
    (define n (String.length s))
    (define i 0)
    (define out (new_list))
    (define buf "")
    (define mt #f)
    (define m #f)
    (while (< i n)
      { (set! mt (inline_matcher_of (String.charCodeAt i s)))
        (set! m (if mt (mt s i) #f))
        (if m
            { (if (> (String.length buf) 0)
                  { (push out (mk2 'text buf))
                    (set! buf "") }
                  #f)
              (push out (get_item m 0))
              (set! i (get_item m 1)) }
            { (set! buf (String.concat buf (String.charAt s i)))
              (set! i (+ i 1)) }) })
    (if (> (String.length buf) 0) (push out (mk2 'text buf)) #f)
    out))

;; ============================================================================
;; 二、块级解析（扩展点①：块级规则表）
;; ============================================================================

(define block_rules (new_list))

;; (种类 谓词 处理函数)
(define rule_add
  (lambda (kind pred handler)
    (push block_rules (mk3 kind pred handler))))

;; 取某种类的处理函数；无则 #f
(define handler_of
  (lambda (kind)
    (define n (length block_rules))
    (define i 0)
    (define r #f)
    (while (and (< i n) (not r))
      { (if (== (get_item (get_item block_rules i) 0) kind)
            (set! r (get_item (get_item block_rules i) 2))
            #f)
        (set! i (+ i 1)) })
    r))

;; ---- 各类块的行判定 ------------------------------------------------------

(define is_blank (lambda (line) (== (String.length (trim line)) 0)))

;; ATX 标题层级：'#' 后必须接空白或行尾；1..6；否则 0
(define heading_level
  (lambda (line)
    (define n (String.length line))
    (define i 0)
    (define lvl 0)
    (while (and (< i n) (char_is line i CP_HASH))
      (set! i (+ i 1)))
    (set! lvl i)
    (if (== lvl 0) 0
        (if (> lvl 6) 0
            (if (== i n) lvl
                (if (is_ws (String.charCodeAt i line)) lvl 0))))))

;; 分隔线：整行只由 >=3 个 - / * / _ 组成（标记之间允许空格，如 "- - -"，同 CommonMark）
(define is_rule
  (lambda (line)
    (define t (trim line))
    (define n (String.length t))
    (define i 1)
    (define c0 0)
    (define cnt 0)
    (define ok #t)
    (if (== n 0) #f
        { (set! c0 (String.charCodeAt 0 t))
          (if (or (== c0 CP_DASH) (or (== c0 CP_STAR) (== c0 CP_UNDER)))
              { (while (< i n)
                  { (if (== (String.charCodeAt i t) c0)
                        (set! cnt (+ cnt 1))
                        (if (is_ws (String.charCodeAt i t)) #f (set! ok #f)))
                    (set! i (+ i 1)) })
                (if (>= cnt 2) ok #f) }
              #f) })))

;; 围栏代码块起始行：(标记字符 个数 缩进) 或 #f
(define fence_open
  (lambda (line)
    (define ind (lead_ws line))
    (define n (String.length line))
    (define c0 0)
    (define i 0)
    (if (>= ind n) #f
        { (set! c0 (String.charCodeAt ind line))
          (if (or (== c0 CP_TICK) (== c0 CP_TILDE))
              { (set! i ind)
                (while (and (< i n) (== (String.charCodeAt i line) c0)) (set! i (+ i 1)))
                (if (>= (- i ind) 3) (mk3 c0 (- i ind) ind) #f) }
              #f) })))

;; 围栏代码块结束行：同标记字符、个数不少于起始个数
(define fence_close
  (lambda (line info)
    (define ch (get_item info 0))
    (define need (get_item info 1))
    (define t (trim line))
    (define n (String.length t))
    (define i 0)
    (while (and (< i n) (== (String.charCodeAt i t) ch)) (set! i (+ i 1)))
    (if (>= i need) (if (== i n) #t #f) #f)))

;; 无序列表项识别：'*'/'-'/'+' 后必须跟空白（或行尾）。
;; 返回 (缩进 内容起点) 或 #f。**不再限制缩进 0..3**——嵌套就是靠缩进表达的：
;; 缩进比上一层更深 → 成为上一层的子项（层号在 h_list 里用"祖先栈"算，见下）。
(define list_item_at
  (lambda (line)
    (define n (String.length line))
    (define lead (lead_ws line))
    (define c0 0)
    (define r #f)
    (if (>= lead n) #f
        { (set! c0 (String.charCodeAt lead line))
          (if (or (== c0 CP_DASH) (or (== c0 CP_STAR) (== c0 CP_PLUS)))
              (if (>= (+ lead 1) n)
                  (set! r (mk2 lead (+ lead 1)))
                  (if (is_ws (String.charCodeAt (+ lead 1) line))
                      (set! r (mk2 lead (+ lead 2)))
                      #f))
              #f) })
    r))

;; 兼容旧接口：只要内容起点（不是列表项返回 -1）
(define list_content_at
  (lambda (line)
    (define it (list_item_at line))
    (if it (get_item it 1) -1)))

;; 取列表前 k 个元素（造新表；用于"层级路径"截断）
(define list_prefix
  (lambda (lst k)
    (define out (new_list))
    (define i 0)
    (define m (length lst))
    (while (and (< i k) (< i m))
      { (push out (get_item lst i))
        (set! i (+ i 1)) })
    out))

;; ---- 各类块的处理函数（输入 lines/下标，输出 (块 下一行下标)）-------------

;; 段落：连续的非空、非其它块起始行；行间以「软换行」拼接（CJK 之间不加空格）
(define h_para
  (lambda (lines idx)
    (define n (length lines))
    (define i idx)
    (define txt "")
    (define k 'para)
    (define stop 0)
    (while (and (< i n) (== stop 0))
      { (set! k (classify lines i))
        (if (== k 'para)
            { (set! txt (soft_join txt (trim (get_item lines i))))
              (set! i (+ i 1)) }
            (set! stop 1)) })
    (mk2 (mk2 'para (parse_inlines txt)) i)))

;; 标题：## 文本（去掉可选的收尾 #）
(define h_heading
  (lambda (lines idx)
    (define line (get_item lines idx))
    (define lvl (heading_level line))
    (define t (trim (String.slice line lvl (String.length line))))
    (define n (String.length t))
    (define e n)
    (while (and (> e 0) (== (String.charCodeAt (- e 1) t) CP_HASH)) (set! e (- e 1)))
    (set! t (trim (String.slice t 0 e)))
    (mk2 (mk3 'heading lvl (parse_inlines t)) (+ idx 1))))

;; 围栏代码块：```lang ... ```（内容行原样保留；公共缩进按起始缩进剥离）
(define h_fence
  (lambda (lines idx)
    (define n (length lines))
    (define info (fence_open (get_item lines idx)))
    (define ind (get_item info 2))
    (define lang (trim (String.slice (get_item lines idx)
                                     (+ ind (get_item info 1))
                                     (String.length (get_item lines idx)))))
    (define i (+ idx 1))
    (define body (new_list))
    (define stop 0)
    (while (and (< i n) (== stop 0))
      { (if (fence_close (get_item lines i) info)
            { (set! i (+ i 1))
              (set! stop 1) }
            { (push body (strip_lead (get_item lines i) ind))
              (set! i (+ i 1)) }) })
    (mk2 (mk3 'code lang body) i)))

;; 无序列表（**支持嵌套**）：item 行 + 其后续行（软换行续行）。
;;
;; 嵌套层级用"祖先栈"判定（与 CommonMark 的精神一致）：栈里存每一层的**内容起点缩进**，
;; 遇到新行先把比它浅的祖先弹出，剩下的栈深就是它的层号：
;;     - a            ind=0  → 栈 []      → 层 0，栈 [2]
;;       - b          ind=2  → 栈 [2]     → 层 1，栈 [2,4]
;;         - c        ind=4  → 栈 [2,4]   → 层 2，栈 [2,4,6]
;;     - d            ind=0  → 弹到空     → 层 0
;; 建树时 path[k] = 第 k 层当前 item，新 item 挂到 path[层-1] 的 children 上。
;; item 形状 (item inlines children)：inlines 在"该 item 的文本收够"时才解析（续行会追加）。
(define h_list
  (lambda (lines idx)
    (define n (length lines))
    (define i idx)
    (define root (new_list))
    (define path (new_list))
    (define stack (new_list))
    (define cur_item #f)
    (define cur_text "")
    (define has 0)
    (define stop 0)
    (define line "")
    (define k 'para)
    (define pos #f)
    (define ind 0)
    (define con 0)
    (define lvl 0)
    (define it #f)
    (define parent #f)
    (while (and (< i n) (== stop 0))
      { (set! line (get_item lines i))
        (set! k (classify lines i))
        (set! pos (list_item_at line))
        (if (== k 'blank)
            (set! stop 1)
            (if pos
                { ;; 先把上一个 item 的文本定稿（续行可能已经追加过）
                  (if (== has 1)
                      (set_item! cur_item 1 (parse_inlines (trim cur_text)))
                      #f)
                  (set! ind (get_item pos 0))
                  (set! con (get_item pos 1))
                  (while (and (> (length stack) 0)
                              (< ind (get_item stack (- (length stack) 1))))
                    (pop stack))
                  (set! lvl (length stack))
                  (push stack con)
                  (set! it (mk3 'item (new_list) (new_list)))
                  (if (== lvl 0)
                      (push root it)
                      { (set! parent (get_item path (- lvl 1)))
                        (push (get_item parent 2) it) })
                  (set! path (list_prefix path lvl))
                  (push path it)
                  (set! cur_item it)
                  (set! cur_text (String.slice line con (String.length line)))
                  (set! has 1)
                  (set! i (+ i 1)) }
                (if (== k 'para)
                    { (set! cur_text (soft_join cur_text (trim line)))
                      (set! i (+ i 1)) }
                    (set! stop 1)))) })
    (if (== has 1)
        (set_item! cur_item 1 (parse_inlines (trim cur_text)))
        #f)
    (mk2 (mk2 'list root) i)))

;; ---- 引用块 --------------------------------------------------------------
;; 行首（允许 <=3 空格缩进）是 '>' 时返回其后的内容起点（跳过一个空格），否则 -1。
(define quote_content_at
  (lambda (line)
    (define n (String.length line))
    (define lead (lead_ws line))
    (define r -1)
    (if (> lead 3) -1
        (if (>= lead n) -1
            (if (== (String.charCodeAt lead line) CP_GT)
                { (set! r (+ lead 1))
                  (if (< r n)
                      (if (is_ws (String.charCodeAt r line)) (set! r (+ r 1)) #f)
                      #f)
                  r }
                -1)))))

;; 引用块：只剥掉**一层** '>' 前缀，产出中间块 (quote-raw 内层行数组)。
;;
;; **为什么不在这里直接递归解析**（这是本轮踩到的一个解释器坑）：
;; 灵机里"同一个闭包被重入调用"会共用帧——parse_lines → h_quote → parse_lines
;; 这种两级回调会让外层 h_quote 的局部变量（记录"下一行下标"的 i）在递归返回后
;; 变成内层的值，表现为引用块被反复重解析（一块变三块）。因此这里不做递归，
;; 由 fixup_quotes 在同一次 parse_lines **返回之后**逐层展开（见下）——
;; 那时外层调用已经结束，不存在重入。
;; 简单子集取舍：不做 CommonMark 的"懒惰续行"（非 '>' 行一律结束引用块）。
(define h_quote
  (lambda (lines idx)
    (define n (length lines))
    (define i idx)
    (define inner (new_list))
    (define stop 0)
    (define p -1)
    (while (and (< i n) (== stop 0))
      { (set! p (quote_content_at (get_item lines i)))
        (if (>= p 0)
            { (push inner (String.slice (get_item lines i) p
                                        (String.length (get_item lines i))))
              (set! i (+ i 1)) }
            (set! stop 1)) })
    (mk2 (mk2 'quote-raw inner) i)))

;; 分隔线
(define h_rule
  (lambda (lines idx)
    (mk2 (mk1 'rule) (+ idx 1))))

;; ---- 表格 --------------------------------------------------------------
;;
;; GFM/CommonMark 扩展的表格语法：表头行 + 分隔行（`|---|:--:|`）之后是若干数据行。
;; 判定要点（全部集中在这里，排版端只管画）：
;;   · 分隔行**必须**存在且每格只由 - : 空格 组成、且至少有一个 '-'；
;;   · 表头行与数据行按 '|' 切格（行首/行尾的 '|' 造成的空单元丢弃）；
;;   · 单元格里的 \| 是字面竖线（反斜杠原样保留，交给行内转义规则处理）；
;;   · 数据行在遇到空行、或不含 '|' 的行、或围栏代码块时结束；
;;   · 列数取表头与所有数据行的最大值，缺的格由排版端补空。
;; 对齐来自分隔行的冒号：---: 右对齐、:---: 居中、:--- 与 --- 左对齐。

;; 本行含有未转义的 '|'（即可能是表格行）
(define has_pipe
  (lambda (line)
    (>= (find_char line 0 CP_PIPE) 0)))

;; 一行按 '|' 切格：保留 \| 的字面竖线（反斜杠不丢，交给行内转义），去首尾空单元
(define split_cells
  (lambda (line)
    (define n (String.length line))
    (define i 0)
    (define cp 0)
    (define esc 0)
    (define cur "")
    (define raw (new_list))
    (define out (new_list))
    (define a 0)
    (define b 0)
    (while (< i n)
      { (set! cp (String.charCodeAt i line))
        (if (== esc 1)
            { (set! cur (String.concat cur (String.charAt line i)))
              (set! esc 0) }
            (if (== cp CP_BSLASH)
                { (set! cur (String.concat cur "\\"))
                  (set! esc 1) }
                (if (== cp CP_PIPE)
                    { (push raw (trim cur))
                      (set! cur "") }
                    (set! cur (String.concat cur (String.charAt line i))))))
        (set! i (+ i 1)) })
    (push raw (trim cur))
    ;; 去掉首/尾的空单元（只可能由行首、行尾的 '|' 造成）
    (set! a 0)
    (set! b (length raw))
    (if (> b 0)
        (if (String.equals (get_item raw 0) "") (set! a 1) #f)
        #f)
    (if (> b a)
        (if (String.equals (get_item raw (- b 1)) "") (set! b (- b 1)) #f)
        #f)
    (while (< a b)
      { (push out (get_item raw a))
        (set! a (+ a 1)) })
    out))

;; 单个分隔单元格 → 对齐码（0 左 / 1 中 / 2 右）；不合法返回 -1
(define delim_cell_align
  (lambda (c)
    (define m (String.length c))
    (define k 0)
    (define cp 0)
    (define nd 0)
    (define left 0)
    (define right 0)
    (define ok 1)
    (if (== m 0) -1
        { (if (== (String.charCodeAt 0 c) CP_COLON)
              { (set! left 1) (set! k 1) }
              (set! k 0))
          (while (< k m)
            { (set! cp (String.charCodeAt k c))
              (if (== cp CP_DASH)
                  (set! nd (+ nd 1))
                  (if (== cp CP_COLON)
                      (if (== k (- m 1)) (set! right 1) (set! ok 0))
                      (set! ok 0)))
              (set! k (+ k 1)) })
          (if (== nd 0) (set! ok 0) #f)
          (if (== ok 0) -1
              (if (== left 1)
                  (if (== right 1) 1 0)
                  (if (== right 1) 2 0))) })))

;; 分隔行 → 每列对齐码列表；不是分隔行返回 #f
(define delim_aligns
  (lambda (line)
    (define cells (split_cells line))
    (define n (length cells))
    (define i 0)
    (define a 0)
    (define bad 0)
    (define out (new_list))
    (while (< i n)
      { (set! a (delim_cell_align (get_item cells i)))
        (if (< a 0) (set! bad 1) #f)
        (push out a)
        (set! i (+ i 1)) })
    (if (== n 0) #f
        (if (== bad 1) #f out))))

;; 一组单元格字符串 → 单元格行内节点列表（每格再走一遍行内解析）
(define cells_inlines
  (lambda (cells)
    (define n (length cells))
    (define i 0)
    (define out (new_list))
    (while (< i n)
      { (push out (parse_inlines (get_item cells i)))
        (set! i (+ i 1)) })
    out))

;; 从 idx 起是否是表格：表头行含 '|'，下一行是合法分隔行，**且两者格数一致**
;; （格数一致这条很关键：否则「一行带竖线的普通文字 + 下面的 --- 」会被误判成单列表格，
;;   而它在 Markdown 里应当是「段落 + 分隔线」。GFM 同样要求列数匹配。）
(define table_start
  (lambda (lines idx)
    (define n (length lines))
    (define head (new_list))
    (define dl (new_list))
    (if (>= (+ idx 1) n) #f
        (if (has_pipe (get_item lines idx))
            { (set! head (split_cells (get_item lines idx)))
              (set! dl (delim_aligns (get_item lines (+ idx 1))))
              (if dl
                  (if (== (length head) (length dl)) #t #f)
                  #f) }
            #f))))

(define h_table
  (lambda (lines idx)
    (define n (length lines))
    (define aligns (delim_aligns (get_item lines (+ idx 1))))
    (define header (cells_inlines (split_cells (get_item lines idx))))
    (define i (+ idx 2))
    (define rows (new_list))
    (define stop 0)
    (define line "")
    (while (and (< i n) (== stop 0))
      { (set! line (get_item lines i))
        (if (has_pipe line)
            (if (fence_open line)
                (set! stop 1)                       ;; 围栏开头：表格结束
                (if (is_blank line)
                    (set! stop 1)
                    { (push rows (cells_inlines (split_cells line)))
                      (set! i (+ i 1)) }))
            (set! stop 1)) })
    (mk2 (mk4 'table aligns header rows) i)))

;; ---- 注册块级规则（顺序即优先级；段落为兜底，不注册）---------------------

(rule_add 'fence (lambda (lines idx) (if (fence_open (get_item lines idx)) #t #f)) h_fence)
(rule_add 'heading (lambda (lines idx) (> (heading_level (get_item lines idx)) 0)) h_heading)
(rule_add 'quote (lambda (lines idx) (>= (quote_content_at (get_item lines idx)) 0)) h_quote)
;; 表格排在 hr/list 之前：分隔行的 `---` 不是分隔线，而是表格的一部分
(rule_add 'table (lambda (lines idx) (table_start lines idx)) h_table)
(rule_add 'hr (lambda (lines idx) (is_rule (get_item lines idx))) h_rule)
(rule_add 'list (lambda (lines idx) (if (list_item_at (get_item lines idx)) #t #f)) h_list)
;; 段落：兜底规则，谓词即「非空行」——必须**最后**注册（先注册会抢走所有行）
(rule_add 'para (lambda (lines idx) (not (is_blank (get_item lines idx)))) h_para)

;; ============================================================================
;; 四、解析主循环与对外 API
;; ============================================================================

;; 行分类：第一个谓词命中的规则种类；都不命中 → 'blank / 'para
;; 谓词需要行号（见文件头扩展点①：表格要前瞻一行）
(define classify
  (lambda (lines idx)
    (define n (length block_rules))
    (define i 0)
    (define r 'none)
    (define row #f)
    (define pred #f)
    (while (and (< i n) (== r 'none))
      { (set! row (get_item block_rules i))
        (set! pred (get_item row 1))
        (if (pred lines idx) (set! r (get_item row 0)) #f)
        (set! i (+ i 1)) })
    (if (== r 'none)
        (if (is_blank (get_item lines idx)) 'blank 'para)
        r)))

;; 行列表 → 块列表（解析主循环：查表分发，不认识的行只占 1 行并跳过，绝不卡住）。
;; 注意：**不处理引用块的嵌套**——引用先落成 (quote-raw 行数组)，由 fixup_quotes 摊平。
(define parse_lines_once
  (lambda (lines)
    (define n (length lines))
    (define i 0)
    (define out (new_list))
    (define kind 'para)
    (define h #f)
    (define r #f)
    (define nx 0)
    (while (< i n)
      { (set! kind (classify lines i))
        (if (== kind 'blank)
            (set! i (+ i 1))
            { (set! h (handler_of kind))
              (if h
                  { (set! r (h lines i))
                    (push out (get_item r 0))
                    (set! nx (get_item r 1))
                    (if (> nx i) (set! i nx) (set! i (+ i 1))) }
                  (set! i (+ i 1))) }) })
    out))

;; 把块列表里的 (quote-raw 行数组) 逐层展开成 (quote 块列表)。
;; 关键点：这里对 parse_lines_once 的调用**互不嵌套**（每次都等上一次返回），
;; 因此不会触发"同一闭包重入共用帧"的问题；递归只发生在 fixup_quotes 自己身上
;; （直接自递归会按 ISSUES #8 新建闭包，帧是独立的）。
(define fixup_quotes
  (lambda (blocks)
    (define n (length blocks))
    (define i 0)
    (define b #f)
    (while (< i n)
      { (set! b (get_item blocks i))
        (if (== (get_item b 0) 'quote-raw)
            (set_item! blocks i (mk2 'quote (fixup_quotes (parse_lines_once (get_item b 1)))))
            #f)
        (set! i (+ i 1)) })
    blocks))

;; 对外：行列表 → 块列表（解析 + 引用展开）
(define parse_lines
  (lambda (lines)
    (fixup_quotes (parse_lines_once lines))))

;; 整段文本 → 块列表
(define parse
  (lambda (text)
    (parse_lines (split_lines text))))

;; ---- 统计（供 App 显示「块数/行数」与单测断言）---------------------------

(define count_kind
  (lambda (blocks kind)
    (define n (length blocks))
    (define i 0)
    (define c 0)
    (while (< i n)
      { (if (== (get_item (get_item blocks i) 0) kind) (set! c (+ c 1)) #f)
        (set! i (+ i 1)) })
    c))