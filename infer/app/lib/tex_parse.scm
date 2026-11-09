;; ============================================================================
;; tex_parse.scm —— LaTeX 数学公式的词法分析 + 语法分析（MathJax 兼容子集）
;;
;; 输入：MathJax 风格的 LaTeX 数学串（不含 $ / $$ 定界符，"公式体"本身）
;; 输出：语法树（纯数据列表，标签在第 0 项）：
;;
;;   ("c" cp)              单字符原子（cp 为 Unicode 码点数值）
;;   ("g" nodes)           分组（花括号组 / 整个公式）
;;   ("f" num den rule)    分式；rule=1 有分数线，0 无（\binom）
;;   ("r" idx body)        根号；idx = 指数节点或 #f
;;   ("s" base sub sup)    上下标；sub/sup = 节点或 #f
;;   ("o" kind val lim)    大运算符：kind=0 符号（val=码点）/ 1 名字（val=字符串）；
;;                         lim: 0 自动、1 强制上下限（\limits）、2 强制同侧
;;   ("d" open close body) \left...\right；open/close = 码点或 #f（\left. 不可见）
;;   ("a" name body)       重音/上划线：overline underline bar vec hat tilde dot ddot
;;   ("t" str)             正文（\text{...}：空格保留、正体）
;;   ("F" name nodes)      字体切换：rm（正体）bf（粗）it（斜）bb（粗黑）cal（斜）
;;   ("k" em)              显式空白（单位 em，可为负）
;;   ("m" kind rows)       矩阵/方程组：kind 0 matrix、1 pmatrix、2 bmatrix、3 Bmatrix、
;;                         4 vmatrix、5 Vmatrix、6 cases、7 aligned
;;   ("x" name)            无法识别的控制词（按正体文本原样显示，便于自查）
;;
;; 与 TeX 的差异（有意简化，见 app/mathjax.scm 头注释）：\over 一类中缀形式不支持；
;; 自定义宏/环境不支持；\left..\right 只支持一层（可嵌套）。
;;
;; 灵机约束：算术恒二参数、无 let/list/append、块内不 define、cond 每子句单表达式、
;; 源码字面量列表是静态共享对象（本文件不修改任何节点，节点一律 quasiquote 新建）。
;; ============================================================================

(native String)
(native Math)

;; 全新的空列表（'() 是编译期静态共享对象，push 会污染全局，见 tex_base.scm 说明）
(define  empty (lambda () (cdr (cons #f '()))))

;; ---------------------------------------------------------------------------
;; 词法
;; ---------------------------------------------------------------------------

(define src "")
(define slen 0)
(define ti 0)
(define toks (empty))
(define ntoks 0)

(define  tpush (lambda (t) { (push toks t) (set! ntoks (+ ntoks 1)) }))

(define  is_alpha_code (lambda (c)
  (or (and (>= c 65) (<= c 90)) (and (>= c 97) (<= c 122)))))

(define  is_space_ch (lambda (c)
  (or (equal? c " ") (equal? c "\t") (equal? c "\n") (equal? c "\r"))))

(define  skip_spaces (lambda ()
  (while (and (< ti slen) (is_space_ch (String.charAt src ti)))
    (set! ti (+ ti 1)))))

(define  skip_comment (lambda ()
  (while (and (< ti slen) (not (equal? (String.charAt src ti) "\n")))
    (set! ti (+ ti 1)))))

;; 反斜杠引导的控制序列：字母序列 → 控制词（"\\frac"），否则单字符 → 控制符号（"\\{"）
(define ct "")
(define  scan_control (lambda ()
  (set! ti (+ ti 1))
  (if (< ti slen)
      {
        (set! ct (String.charAt src ti))
        (if (is_alpha_code (String.charCodeAt 0 ct))
            {
              (set! ct "\\")
              (while (and (< ti slen) (is_alpha_code (String.charCodeAt 0 (String.charAt src ti))))
                {
                  (set! ct (String.concat ct (String.charAt src ti)))
                  (set! ti (+ ti 1))
                })
              (tpush ct)
              ;; 控制词之后的空白被吞掉（TeX 规则：\alpha x 与 \alphax 不同）
              (skip_spaces)
            }
            {
              (tpush (String.concat "\\" ct))
              (set! ti (+ ti 1))
            })
      }
      #f)))

(define tc "")
(define  scan_one (lambda ()
  (set! tc (String.charAt src ti))
  (cond
    ((equal? tc "\\") (scan_control))
    ((equal? tc "%") (skip_comment))
    ((is_space_ch tc) { (tpush " ") (skip_spaces) })
    (else { (tpush tc) (set! ti (+ ti 1)) }))))

(define  tokenize (lambda (s)
  (set! src s)
  (set! slen (String.length s))
  (set! ti 0)
  (set! toks (empty))
  (set! ntoks 0)
  (while (< ti slen) (scan_one))))

;; ---------------------------------------------------------------------------
;; 词法辅助
;; ---------------------------------------------------------------------------

(define  tk (lambda () (get_item toks tp_)))
(define  adv (lambda () (set! tp_ (+ tp_ 1))))
(define  at_end (lambda () (>= tp_ ntoks)))
(define  is_space_tok (lambda (t) (equal? t " ")))
(define  is_ctrl (lambda (t)
  (if (undefined? t) #f (equal? (String.charAt t 0) "\\"))))
(define  ctrl_name (lambda (t) (String.slice t 1 (String.length t))))

(define tp_ 0)

(define  skip_space_toks (lambda ()
  (while (and (not (at_end)) (is_space_tok (tk))) (adv))))

(define  mem? (lambda (t lst)
  (define i 0)
  (define n (length lst))
  (define r #f)
  (while (< i n)
    {
      (if (equal? (get_item lst i) t) { (set! r #t) (set! i n) } (set! i (+ i 1)))
    })
  r))

;; ---------------------------------------------------------------------------
;; 符号表：控制词 → Unicode 码点（正体/斜体由排版层按字符类别决定）
;; ---------------------------------------------------------------------------

(define syms
  '(
    ;; 希腊小写
    ("alpha" 945) ("beta" 946) ("gamma" 947) ("delta" 948) ("epsilon" 949)
    ("zeta" 950) ("eta" 951) ("theta" 952) ("iota" 953) ("kappa" 954)
    ("lambda" 955) ("mu" 956) ("nu" 957) ("xi" 958) ("omicron" 959)
    ("pi" 960) ("rho" 961) ("varsigma" 963) ("sigma" 963) ("tau" 964)
    ("upsilon" 965) ("phi" 966) ("chi" 967) ("psi" 968) ("omega" 969)
    ("varepsilon" 949) ("vartheta" 952) ("varphi" 966) ("varrho" 961)
    ("varkappa" 954) ("digamma" 949)
    ;; 希腊大写
    ("Gamma" 915) ("Delta" 916) ("Theta" 920) ("Lambda" 923) ("Xi" 926)
    ("Pi" 928) ("Sigma" 931) ("Upsilon" 933) ("Phi" 934) ("Psi" 936)
    ("Omega" 937)
    ;; 二元运算
    ("cdot" 183) ("times" 215) ("div" 247) ("pm" 177) ("mp" 8723)
    ("ast" 42) ("star" 42) ("circ" 8728) ("bullet" 8226)
    ("oplus" 8853) ("ominus" 8854) ("otimes" 8855) ("odot" 8857)
    ("cup" 8746) ("cap" 8745) ("setminus" 8726) ("smallsetminus" 8726)
    ("wedge" 8743) ("land" 8743) ("vee" 8744) ("lor" 8744)
    ("uplus" 8746) ("sqcup" 8746) ("sqcap" 8745) ("dagger" 8224) ("ddagger" 8225)
    ;; 关系
    ("le" 8804) ("leq" 8804) ("ge" 8805) ("geq" 8805) ("ne" 8800) ("neq" 8800)
    ("approx" 8776) ("sim" 8764) ("simeq" 8771) ("cong" 8773) ("equiv" 8801)
    ("propto" 8733) ("ll" 8810) ("gg" 8811) ("perp" 8869) ("parallel" 8741)
    ("mid" 124) ("nmid" 8740) ("subset" 8834) ("supset" 8835)
    ("subseteq" 8838) ("supseteq" 8839) ("subsetneq" 8838) ("supsetneq" 8839)
    ("in" 8712) ("notin" 8713) ("ni" 8715) ("owns" 8715)
    ("prec" 60) ("succ" 62) ("preceq" 8804) ("succeq" 8805)
    ("doteq" 61) ("asymp" 8776) ("bowtie" 8904) ("models" 8871)
    ("vdash" 8866) ("dashv" 8867) ("smile" 8995) ("frown" 8994)
    ("lesssim" 8804) ("gtrsim" 8805) ("lessgtr" 8776) ("gtrless" 8776)
    ;; 箭头
    ("to" 8594) ("rightarrow" 8594) ("leftarrow" 8592) ("gets" 8592)
    ("leftrightarrow" 8596) ("Rightarrow" 8658) ("Leftarrow" 8656)
    ("Leftrightarrow" 8660) ("implies" 8658) ("iff" 8660)
    ("mapsto" 8614) ("longmapsto" 8614) ("longrightarrow" 8594)
    ("longleftarrow" 8592) ("Longrightarrow" 8658) ("Longleftrightarrow" 8660)
    ("uparrow" 8593) ("downarrow" 8595) ("updownarrow" 8597)
    ("nearrow" 8599) ("searrow" 8600) ("swarrow" 8601) ("nwarrow" 8598)
    ("hookrightarrow" 8594) ("rightleftharpoons" 8596) ("rightsquigarrow" 8594)
    ;; 杂类
    ("forall" 8704) ("exists" 8705) ("nexists" 8708) ("emptyset" 8709)
    ("varnothing" 8709) ("infty" 8734) ("partial" 8706) ("nabla" 8711)
    ("ell" 8467) ("hbar" 8463) ("hslash" 8463) ("Re" 8476) ("Im" 8465)
    ("aleph" 8501) ("wp" 8472) ("angle" 8736) ("triangle" 9651)
    ("square" 9633) ("blacksquare" 9632) ("therefore" 8756) ("because" 8755)
    ("dots" 8230) ("ldots" 8230) ("cdots" 8943) ("vdots" 8942) ("ddots" 8945)
    ("prime" 8242) ("backslash" 92) ("neg" 172) ("lnot" 172)
    ("imath" 105) ("jmath" 106) ("surd" 8730) ("checkmark" 10003)
    ("S" 167) ("P" 182) ("copyright" 169) ("pounds" 163) ("euro" 8364)
    ;; 定界符（字符形式）
    ("langle" 10216) ("rangle" 10217) ("lfloor" 8970) ("rfloor" 8971)
    ("lceil" 8968) ("rceil" 8969) ("lbrace" 123) ("rbrace" 125)
    ("vert" 124) ("lvert" 124) ("rvert" 124) ("Vert" 8214) ("lVert" 8214)
    ("rVert" 8214) ("lgroup" 10216) ("rgroup" 10217)
    ;; 转义字符
    ("{" 123) ("}" 125) ("%" 37) ("&" 38) ("#" 35) ("_" 95) ("$" 36)
    ))

(define  sym_lookup (lambda (name)
  (define i 0)
  (define n (length syms))
  (define r #f)
  (define e #f)
  (while (< i n)
    {
      (set! e (get_item syms i))
      (if (equal? (get_item e 0) name)
          { (set! r (get_item e 1)) (set! i n) }
          (set! i (+ i 1)))
    })
  r))

;; ---------------------------------------------------------------------------
;; 语法分析
;; ---------------------------------------------------------------------------

;; 分组：{ ... }（不接上下标；由调用方决定是否 attach）
(define  parse_group_raw (lambda ()
  (define rows (empty))
  (adv)
  (set! rows (parse_row '("}")))
  (if (equal? (tk) "}") (adv) #f)
  `("g" ,rows)))

;; 参数：单个原子（分组 / 单字符 / 控制序列），不接上下标
(define ptmp #f)
(define  parse_primary (lambda ()
  (skip_space_toks)
  (set! ptmp (tk))
  (cond
    ((at_end) #f)
    ((equal? ptmp "{") (parse_group_raw))
    ((equal? ptmp "}") #f)
    ((equal? ptmp "&") #f)
    ((equal? ptmp "^") #f)
    ((equal? ptmp "_") #f)
    ((is_ctrl ptmp) (parse_control ptmp))
    (else { (adv) `("c" ,(String.charCodeAt 0 ptmp)) }))))

;; 上下标（可连续出现，也可夹 \limits / \nolimits）
(define  attach_scripts (lambda (node)
  (define base node)
  (define sub #f)
  (define sup #f)
  (define lim 0)
  (define t #f)
  (while #t
    {
      (set! t (tk))
      (cond
        ((equal? t "^") { (adv) (set! sup (parse_primary)) })
        ((equal? t "_") { (adv) (set! sub (parse_primary)) })
        ((equal? t "\\limits") { (adv) (set! lim 1) })
        ((equal? t "\\nolimits") { (adv) (set! lim 2) })
        (else break))
    })
  (if (and (equal? (get_item base 0) "o") (> lim 0))
      (set! base `("o" ,(get_item base 1) ,(get_item base 2) ,lim))
      #f)
  (if (or sub sup) `("s" ,base ,sub ,sup) base)))

;; 原子（含其后的上下标）
(define atmp #f)
(define  parse_atom (lambda ()
  (set! atmp (tk))
  (cond
    ((at_end) #f)
    ((equal? atmp "}") #f)
    ((equal? atmp "&") #f)
    ((equal? atmp "\\\\") #f)
    ((equal? atmp "\\right") #f)
    ((equal? atmp "\\end") #f)
    ((equal? atmp "{") (attach_scripts (parse_group_raw)))
    ;; 孤立的 ^ _ ：丢掉标记，继续解析后面的东西（TeX 会报错，这里容错）
    ((equal? atmp "^") { (adv) (parse_atom) })
    ((equal? atmp "_") { (adv) (parse_atom) })
    ((is_ctrl atmp) (attach_scripts (parse_control atmp)))
    (else { (adv) (attach_scripts `("c" ,(String.charCodeAt 0 atmp))) }))))

;; 节点序列（遇到 stop 中的记号 / 输入结束即停）
(define  parse_row (lambda (stop)
  (define nodes (empty))
  (define node #f)
  (define go #t)
  (while go
    (cond
      ((at_end) (set! go #f))
      ((mem? (tk) stop) (set! go #f))
      ((is_space_tok (tk)) (adv))
      (else
        {
          (set! node (parse_atom))
          (if node (push nodes node) (set! go #f))
        })))
  nodes))

;; 定界符（\left 之后）：返回码点，"." → #f
(define  parse_delim (lambda ()
  (define t #f)
  (define d #f)
  (skip_space_toks)
  (set! t (tk))
  (adv)
  (cond
    ((undefined? t) #f)
    ((equal? t ".") #f)
    ((is_ctrl t)
      {
        (set! d (sym_lookup (ctrl_name t)))
        (if d d #f)
      })
    (else (String.charCodeAt 0 t)))))

;; 环境名：{ name }
(define  read_env_name (lambda ()
  (define out "")
  (if (equal? (tk) "{") (adv) #f)
  (while (and (not (at_end)) (not (equal? (tk) "}")))
    {
      (set! out (String.concat out (tk)))
      (adv)
    })
  (if (equal? (tk) "}") (adv) #f)
  out))

;; 正文分组：{ ... }，空格保留（\text{...} 用）
(define  read_text_group (lambda ()
  (define depth 0)
  (define out "")
  (define t #f)
  (if (equal? (tk) "{") (adv) #f)
  (set! depth 1)
  (while (and (> depth 0) (not (at_end)))
    {
      (set! t (tk))
      (adv)
      (cond
        ((equal? t "{") { (set! depth (+ depth 1)) (set! out (String.concat out "{")) })
        ((equal? t "}") { (set! depth (- depth 1))
                          (if (> depth 0) (set! out (String.concat out "}")) #f) })
        ((is_ctrl t) (set! out (String.concat out (text_of_ctrl (ctrl_name t)))))
        ((equal? t " ")
          (if (equal? (String.charAt out (- (String.length out) 1)) " ")
              #f
              (set! out (String.concat out " "))))
        (else (set! out (String.concat out t)))
      )
    })
  out))

(define  text_of_ctrl (lambda (name)
  (cond
    ((equal? name " ") " ")
    ((equal? name ",") "")
    ((equal? name ";") " ")
    ((equal? name ":") " ")
    ((equal? name "quad") " ")
    ((equal? name "qquad") "  ")
    ((equal? name "{") "{")
    ((equal? name "}") "}")
    ((equal? name "%") "%")
    ((equal? name "_") "_")
    ((equal? name "&") "&")
    ((equal? name "#") "#")
    ((equal? name "$") "$")
    ((equal? name "\\") " ")
    (else name))))

;; 矩阵 / 方程组 / aligned
(define  env_kind (lambda (name)
  (cond
    ((equal? name "pmatrix") 1)
    ((equal? name "bmatrix") 2)
    ((equal? name "Bmatrix") 3)
    ((equal? name "vmatrix") 4)
    ((equal? name "Vmatrix") 5)
    ((equal? name "cases") 6)
    ((equal? name "aligned") 7)
    ((equal? name "align") 7)
    ((equal? name "align*") 7)
    ((equal? name "split") 7)
    ((equal? name "gathered") 0)
    ((equal? name "smallmatrix") 0)
    (else 0))))

(define mrows (empty))
(define mcells (empty))
(define  parse_matrix (lambda ()
  (define env (read_env_name))
  (define go #t)
  (define mcell #f)
  (if (equal? env "array") (read_env_name) #f)   ;; 列格式 {cc} 丢弃
  (set! mrows (empty))
  (set! mcells (empty))
  (set! go #t)
  (while go
    {
      (set! mcell (parse_row '("&" "\\\\" "\\end")))
      (push mcells mcell)
      (cond
        ((equal? (tk) "&") (adv))
        ((equal? (tk) "\\\\")
          {
            (adv)
            ;; 可选的行距参数 \\[2pt] 丢弃
            (if (equal? (tk) "[")
                {
                  (while (and (not (at_end)) (not (equal? (tk) "]"))) (adv))
                  (if (equal? (tk) "]") (adv) #f)
                }
                #f)
            (push mrows mcells)
            (set! mcells (empty))
          })
        (else { (push mrows mcells) (set! go #f) })
      )
    })
  ;; 吃掉 \end{env}
  (if (equal? (tk) "\\end") { (adv) (read_env_name) } #f)
  `("m" ,(env_kind env) ,mrows)))

;; 控制序列分发。**局部量必须声明在 lambda 内**：parse_control 会递归（分组、参数里
;; 都可能有控制序列），模块级的临时槽会被内层调用冲掉（曾因此把 \left( 的定界符读成
;; 了分子的分组节点）。
(define  parse_control (lambda (t)
  (define cn "")
  (define a1 #f)
  (define a2 #f)
  (define a3 #f)
  (set! cn (ctrl_name t))
  (adv)

  (cond
    ;; ---- 结构 ----
    ((or (equal? cn "frac") (equal? cn "dfrac") (equal? cn "tfrac") (equal? cn "cfrac"))
      {
        (set! a1 (parse_primary))
        (set! a2 (parse_primary))
        `("f" ,a1 ,a2 1)
      })
    ((or (equal? cn "binom") (equal? cn "dbinom") (equal? cn "tbinom"))
      {
        (set! a1 (parse_primary))
        (set! a2 (parse_primary))
        `("f" ,a1 ,a2 0)
      })
    ((equal? cn "sqrt")
      {
        (set! a1 #f)
        (skip_space_toks)
        (if (equal? (tk) "[")
            {
              (adv)
              (set! a1 `("g" ,(parse_row '("]"))))
              (if (equal? (tk) "]") (adv) #f)
            }
            #f)
        (set! a2 (parse_primary))
        `("r" ,a1 ,a2)
      })
    ((equal? cn "left")
      {
        (set! a1 (parse_delim))
        (set! a2 `("g" ,(parse_row '("\\right"))))
        (if (equal? (tk) "\\right") (adv) #f)
        (set! a3 (parse_delim))
        `("d" ,a1 ,a3 ,a2)
      })
    ((equal? cn "begin") (parse_matrix))
    ((equal? cn "end") #f)
    ((equal? cn "middle") { (parse_delim) #f })
    ;; ---- 字体 ----
    ((equal? cn "mathrm") `("F" "rm" ,(cons (parse_primary) '())))
    ((equal? cn "mathbf") `("F" "bf" ,(cons (parse_primary) '())))
    ((equal? cn "mathit") `("F" "it" ,(cons (parse_primary) '())))
    ((or (equal? cn "mathsf") (equal? cn "mathtt") (equal? cn "mathnormal"))
      `("F" "rm" ,(cons (parse_primary) '())))
    ((equal? cn "mathcal") `("F" "cal" ,(cons (parse_primary) '())))
    ((equal? cn "mathbb") `("F" "bb" ,(cons (parse_primary) '())))
    ((or (equal? cn "boldsymbol") (equal? cn "bm") (equal? cn "mathbfit"))
      `("F" "bfit" ,(cons (parse_primary) '())))
    ((equal? cn "operatorname") `("o" 1 ,(read_text_group) 2))
    ;; ---- 正文 ----
    ((or (equal? cn "text") (equal? cn "mbox") (equal? cn "textrm")
         (equal? cn "textnormal") (equal? cn "textup") (equal? cn "hbox"))
      `("t" ,(read_text_group)))
    ;; ---- 重音 / 线 ----
    ((or (equal? cn "overline") (equal? cn "bar")) `("a" "overline" ,(parse_primary)))
    ((equal? cn "underline") `("a" "underline" ,(parse_primary)))
    ((or (equal? cn "vec") (equal? cn "overrightarrow")) `("a" "vec" ,(parse_primary)))
    ((equal? cn "overleftarrow") `("a" "vecleft" ,(parse_primary)))
    ((or (equal? cn "hat") (equal? cn "widehat")) `("a" "hat" ,(parse_primary)))
    ((or (equal? cn "tilde") (equal? cn "widetilde")) `("a" "tilde" ,(parse_primary)))
    ((equal? cn "dot") `("a" "dot" ,(parse_primary)))
    ((equal? cn "ddot") `("a" "ddot" ,(parse_primary)))
    ((or (equal? cn "check") (equal? cn "breve")) `("a" "hat" ,(parse_primary)))
    ((or (equal? cn "acute") (equal? cn "grave")) `("a" "acute" ,(parse_primary)))
    ((equal? cn "mathring") `("a" "dot" ,(parse_primary)))
    ((or (equal? cn "overbrace") (equal? cn "underbrace")) `("a" "overline" ,(parse_primary)))
    ;; ---- 空白 ----
    ((or (equal? cn " ") (equal? cn "quad")) `("k" 1.0))
    ((equal? cn "qquad") `("k" 2.0))
    ((or (equal? cn "enspace") (equal? cn "enskip")) `("k" 0.5))
    ((or (equal? cn ",") (equal? cn "thinspace")) `("k" 0.17))
    ((or (equal? cn ":") (equal? cn ">") (equal? cn "medspace")) `("k" 0.22))
    ((or (equal? cn ";") (equal? cn "thickspace")) `("k" 0.28))
    ((equal? cn "!") `("k" -0.17))
    ((equal? cn "negthinspace") `("k" -0.17))
    ;; ---- 忽略项 ----
    ((or (equal? cn "displaystyle") (equal? cn "textstyle") (equal? cn "scriptstyle")
         (equal? cn "scriptscriptstyle") (equal? cn "not") (equal? cn "hline")
         (equal? cn "limits") (equal? cn "nolimits") (equal? cn "!"))
      #f)
    ;; ---- 大运算符 ----
    ((equal? cn "sum") `("o" 0 8721 0))
    ((equal? cn "prod") `("o" 0 8719 0))
    ((equal? cn "coprod") `("o" 0 8720 0))
    ((equal? cn "int") `("o" 0 8747 0))
    ((equal? cn "iint") `("o" 0 8748 0))
    ((equal? cn "iiint") `("o" 0 8749 0))
    ((equal? cn "oint") `("o" 0 8750 0))
    ((equal? cn "bigcup") `("o" 0 8899 0))
    ((equal? cn "bigcap") `("o" 0 8898 0))
    ((equal? cn "bigvee") `("o" 0 8897 0))
    ((equal? cn "bigwedge") `("o" 0 8896 0))
    ((equal? cn "bigoplus") `("o" 0 10753 0))
    ;; ---- 带上下限的函数名 ----
    ((or (equal? cn "lim") (equal? cn "limsup") (equal? cn "liminf")
         (equal? cn "max") (equal? cn "min") (equal? cn "sup") (equal? cn "inf")
         (equal? cn "det") (equal? cn "gcd") (equal? cn "Pr") (equal? cn "arg"))
      `("o" 1 ,cn 0))
    ;; ---- 正体函数名（无上下限）----
    ((or (equal? cn "sin") (equal? cn "cos") (equal? cn "tan") (equal? cn "cot")
         (equal? cn "sec") (equal? cn "csc") (equal? cn "arcsin") (equal? cn "arccos")
         (equal? cn "arctan") (equal? cn "sinh") (equal? cn "cosh") (equal? cn "tanh")
         (equal? cn "coth") (equal? cn "log") (equal? cn "ln") (equal? cn "lg")
         (equal? cn "exp") (equal? cn "dim") (equal? cn "ker") (equal? cn "hom")
         (equal? cn "deg") (equal? cn "sgn") (equal? cn "mod") (equal? cn "bmod"))
      `("o" 1 ,cn 2))
    ;; ---- 大号定界符（\big( 之类，尺寸差异忽略）----
    ((or (equal? cn "big") (equal? cn "Big") (equal? cn "bigg") (equal? cn "Bigg")
         (equal? cn "bigl") (equal? cn "bigr") (equal? cn "Bigl") (equal? cn "Bigr")
         (equal? cn "biggl") (equal? cn "biggr") (equal? cn "Biggl") (equal? cn "Biggr"))
      {
        (set! a1 (parse_delim))
        (if a1 `("c" ,a1) #f)
      })
    ;; ---- 符号表 ----
    (else
      {
        (set! a1 (sym_lookup cn))
        (if a1 `("c" ,a1) `("x" ,cn))
      }))))

;; ---------------------------------------------------------------------------
;; 对外入口
;; ---------------------------------------------------------------------------

;; 解析一个公式串，返回根节点（"g" ...）
(define  parse (lambda (s)
  (tokenize s)
  (set! tp_ 0)
  `("g" ,(parse_row '()))))
