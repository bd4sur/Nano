;; ============================================================================
;; tex_layout.scm —— LaTeX 数学公式的盒子排版（TeX 风格的 atom 间距 + 四级字号）
;;
;; 输入：tex_parse.scm 产出的语法树
;; 输出：把绘制操作压进 tex_base 的操作缓冲，并返回整式尺寸 (宽 基线以上高 基线以下深)
;;
;; 排版模型（与 TeX 的对应关系）：
;;   · 字号四级：D（display，16px）/ T（text，12px）/ S（script，8.6px）/ SS（6.2px）；
;;     上下标降一级（D/T → S，S → SS），分式的分子分母降一级。
;;   · atom 间距：8 个原子类（Ord Op Bin Rel Open Close Punct Inner）× 18 分制薄/中/厚
;;     空隙表，行首与 Bin 在 Bin/Op/Rel/Open/Punct 之后按 TeX 规则降为 Ord（一元负号）。
;;   · 分数轴（axis）取 0.26em：分式横线、大运算符的垂直中心都对齐到它。
;;   · 上下标：上标基线抬高 max(0.42em, 底高的 0.94 倍)，下标下移 0.2em，
;;     二者重叠时把下标再压低（TeX 规则 18 的简化）。
;;   · 定界符按内容高度现场绘制（小尺寸直接用字形，大尺寸用矢量曲线——
;;     位图括号纵向拉伸会退化成一根竖线）。
;;
;; 所有坐标单位为像素，原点在式子左端、基线 y=0（y 向下为正）。
;; 灵机约束：算术恒二参数、无 let/max/min/abs、块内不 define、cond 单表达式。
;; ============================================================================

(native Gfx)
(native Math)
(native String)

(import B "app/lib/tex_base.scm")
(import S "app/lib/tex_sym.scm")

;; ---------------------------------------------------------------------------
;; 字号与字模
;; ---------------------------------------------------------------------------

;; 四级字号（px）。D 与 T **同号**（与 TeX 一致：\displaystyle 不改变基准字号，只改变
;; 分式的分子分母风格与大运算符尺寸/上下限位置）；S = 0.7×基准，SS = 0.45×基准。
(define EMS '(16 16 11.2 7.2))
(define emul 1.0)
(define  set_emul (lambda (k) (set! emul k)))
(define  get_emul (lambda () emul))
(define  em_of (lambda (st) (* (get_item EMS st) emul)))
(define  font_of (lambda (em) (if (>= em 13.5) 2 1)))
(define  nat_of (lambda (fid) (if (== fid 2) 16 12)))

;; 当前字体状态（0 = 数学默认：字母斜、数字/运算符正；1 = 强制正体；2 = 强制斜体）
(define cur_it 0)
(define cur_bold 0)

;; ---------------------------------------------------------------------------
;; 字形度量缓存（同一次排版内同字符会反复出现；一次 native 调用约百条 VM 指令）
;; 记录形态：(fid cp 步进 上高 下深 位图宽)，单位=字模单位
;; ---------------------------------------------------------------------------

(define mcache (B.empty))
(define mmtmp #f)

(define  mc_get (lambda (fid cp)
  (define i 0)
  (define n (length mcache))
  (define e #f)
  (define r #f)
  (while (< i n)
    {
      (set! e (get_item mcache i))
      (if (and (== (get_item e 0) fid) (== (get_item e 1) cp))
          { (set! r e) (set! i n) }
          (set! i (+ i 1)))
    })
  (if r
      r
      {
        (set! mmtmp (Gfx.glyph_metrics fid cp))
        (if (null? mmtmp)
            (set! r `(,fid ,cp 0 0 0 0 0))
            (set! r `(,fid ,cp ,(get_item mmtmp 0) ,(get_item mmtmp 1)
                           ,(get_item mmtmp 2) ,(get_item mmtmp 3) ,(get_item mmtmp 4))))
        (if (< (length mcache) 128) (push mcache r) #f)
        r
      })))

;; ---------------------------------------------------------------------------
;; 原子类与间距
;;   0=Ord 1=Op 2=Bin 3=Rel 4=Open 5=Close 6=Punct 7=Inner，-1=不成原子（空白节点）
;; ---------------------------------------------------------------------------

(define BINOPS '(43 45 177 8723 215 183 247 8727 8902 8746 8745 8853 8855 8857
                 8854 8728 8846 8852 8851 8726 8743 8744 8901))
(define RELOPS '(61 60 62 8804 8805 8800 8776 8764 8771 8773 8801 8733 8810 8811
                 8869 8741 8712 8713 8715 8834 8835 8838 8839 8594 8592 8596
                 8658 8656 8660 8614 8593 8595 8597 8599 8600 8756 8755 8866
                 8867 8784 8781 8826 8827 8828 8829 8842 8843 8739 8740 8904
                 8871 8598 8601 8618 8652 8663 8776 8784))
(define OPENOPS '(40 91 123 10216 8970 8968))
(define CLOSEOPS '(41 93 125 10217 8971 8969 33 63))
(define PUNCTOPS '(44 59))

(define  mem_int (lambda (cp lst)
  (define i 0)
  (define n (length lst))
  (define r #f)
  (while (< i n)
    {
      (if (== (get_item lst i) cp) { (set! r #t) (set! i n) } (set! i (+ i 1)))
    })
  r))

(define  char_cls (lambda (cp)
  (cond
    ((mem_int cp BINOPS) 2)
    ((mem_int cp RELOPS) 3)
    ((mem_int cp OPENOPS) 4)
    ((mem_int cp CLOSEOPS) 5)
    ((mem_int cp PUNCTOPS) 6)
    (else 0))))

;; 行主序 8×8：thin=1 med=2 thick=3（单位 1/18 em）；Bin 行的 "*" 位置填 med（不可达）
(define SPTAB
  '(0 1 2 3 0 0 0 1
    1 1 2 3 0 0 0 1
    2 2 2 2 2 2 2 2
    3 3 2 0 3 0 0 3
    0 0 2 0 0 0 0 0
    0 1 2 3 0 0 0 1
    1 1 2 1 1 1 1 1
    1 1 2 3 1 0 1 1))

(define  space_of (lambda (prev cur st)
  (define v (get_item SPTAB (+ (* prev 8) cur)))
  (if (== v 0) 0.0 (* v (/ (em_of st) 18)))))

;; 数学斜体的字符集：拉丁字母、希腊小写、少数变体字母
(define  italic_ch (lambda (cp)
  (or (and (>= cp 65) (<= cp 90))
      (and (>= cp 97) (<= cp 122))
      (and (>= cp 945) (<= cp 969))
      (== cp 1013) (== cp 977) (== cp 981) (== cp 1009) (== cp 1008)
      (== cp 305) (== cp 567))))

;; ---------------------------------------------------------------------------
;; 节点类
;; ---------------------------------------------------------------------------

(define  cls_of (lambda (node)
  (define tag #f)
  (define r 0)
  (if (not node) 0
    {
  (set! tag (get_item node 0))
  (cond
    ((equal? tag "c") (char_cls (get_item node 1)))
    ((equal? tag "o") 1)
    ((equal? tag "f") 7)
    ((equal? tag "s") (cls_of (get_item node 1)))
    ((equal? tag "r") 0)
    ((equal? tag "d") 7)
    ((equal? tag "a") (cls_of (get_item node 2)))
    ((equal? tag "k") -1)
    ((equal? tag "m") 0)
    ((equal? tag "F")
      {
        (set! r (get_item node 2))
        (if (null? r) 0 (cls_of (get_item r 0)))
      })
    ((equal? tag "g")
      {
        (set! r (get_item node 1))
        (if (null? r) 0 (cls_of (get_item r 0)))
      })
    (else 0))})))

;; ---------------------------------------------------------------------------
;; 单字符排版（emul2: 相对当前样式的额外字号倍数，大运算符用）
;; ---------------------------------------------------------------------------

(define  lay_char_em (lambda (cp st x y emul2)
  (define em (* (em_of st) emul2))
  (define fid (font_of em))
  (define nat (nat_of fid))
  (define k (/ em nat))
  (define sh 0.0)
  (define e #f)
  (define adv 0.0)
  (define asc 0.0)
  (define dsc 0.0)
  (define ic 0.0)
  (define inkr 0.0)
  (define r #f)
  ;; 斜体：数学变量（拉丁/希腊小写）默认斜；\mathit 强制斜；\mathrm 强制正
  (if (== cur_it 2)
      (set! sh 0.2)
      (if (and (== cur_it 0) (italic_ch cp)) (set! sh 0.2) #f))
  (cond
    ((== cp 32) (set! r `(,(* 0.33 em) 0.0 0.0)))
    (else
      {
        (set! e (mc_get fid cp))
        (if (== (get_item e 5) 0)
            ;; 字模里没有：交给矢量符号表（\partial \nabla \Rightarrow …）
            (set! r (S.draw cp x y em fid cur_bold))
            (set! r #f))
        (if r
            #f
            {
              (set! adv (* (get_item e 2) k))
              (set! asc (* (get_item e 3) k))
              (set! dsc (* (get_item e 4) k))
              (if (> sh 0) (set! ic (* 0.05 em)) #f)
              (B.g x y em 1 1 cp sh cur_bold fid)
              ;; 墨迹右边界：步进往往小于位图（如大写 A / 斜体 f 的过冲），
              ;; 斜体切变还会把顶部再向右推 sh*asc —— 盒子须覆盖，否则右侧被裁。
              (set! inkr (* k (+ (get_item e 6) (get_item e 5))))
              (if (> sh 0) (set! inkr (B.mx inkr (+ inkr (* (* sh asc) 0.55)))) #f)
              (set! r `(,(B.mx (+ adv ic) inkr) ,(B.mx asc 0.0) ,(B.mx dsc 0.0)))
            })
      }))
  r))

(define  lay_char (lambda (cp st x y) (lay_char_em cp st x y 1.0)))

;; ---------------------------------------------------------------------------
;; 正文（\text{}）：正体、空格 0.32em、缺字留空
;; ---------------------------------------------------------------------------

(define  lay_text (lambda (s st x y)
  (define em (em_of st))
  (define fid (font_of em))
  (define nat (nat_of fid))
  (define k (/ em nat))
  (define i 0)
  (define n (String.length s))
  (define pen x)
  (define cp 0)
  (define e #f)
  (define h 0.0)
  (define d 0.0)
  (define asc 0.0)
  (define dsc 0.0)
  (while (< i n)
    {
      (set! cp (String.charCodeAt i s))
      (cond
        ((== cp 32) (set! pen (+ pen (* 0.32 em))))
        ((== cp 126) (set! pen (+ pen (* 0.4 em))))   ;; ~ 作为空格
        (else
          {
            (set! e (mc_get fid cp))
            (if (== (get_item e 5) 0)
                (set! pen (+ pen (* 0.55 em)))
                {
                  (B.g pen y em 1 1 cp 0 cur_bold fid)
                  (set! asc (* (get_item e 3) k))
                  (set! dsc (* (get_item e 4) k))
                  (set! h (B.mx h asc))
                  (set! d (B.mx d dsc))
                  (set! pen (+ pen (* (get_item e 2) k)))
                })
          }))
      (set! i (+ i 1))
    })
  `(,(- pen x) ,h ,d)))

;; ---------------------------------------------------------------------------
;; 水平序列（含 atom 间距）
;; ---------------------------------------------------------------------------

(define  lay_row (lambda (nodes st x y)
  (define n (length nodes))
  (define i 0)
  (define prev -1)
  (define pen x)
  (define h 0.0)
  (define d 0.0)
  (define node #f)
  (define cls 0)
  (define sp 0.0)
  (define bx #f)
  (while (< i n)
    {
      (set! node (get_item nodes i))
      (set! cls (cls_of node))
      (if (and (== cls 2)
               (or (== prev -1) (== prev 1) (== prev 2) (== prev 3) (== prev 4) (== prev 6)))
          (set! cls 0)
          #f)
      (set! sp (if (or (== prev -1) (== cls -1)) 0.0 (space_of prev cls st)))
      (set! pen (+ pen sp))
      (set! bx (lay node st pen y))
      (set! pen (+ pen (get_item bx 0)))
      (set! h (B.mx h (get_item bx 1)))
      (set! d (B.mx d (get_item bx 2)))
      (if (== cls -1) #f (set! prev cls))
      (set! i (+ i 1))
    })
  `(,(- pen x) ,h ,d)))

;; ---------------------------------------------------------------------------
;; 上下标（普通原子）
;; ---------------------------------------------------------------------------

(define  italic_corr (lambda (node st)
  (define tag (get_item node 0))
  (if (equal? tag "c")
      (if (and (== cur_it 0) (italic_ch (get_item node 1))) (* 0.05 (em_of st)) 0.0)
      0.0)))

(define  lay_ord_scripts (lambda (base sub sup st x y)
  (define em (em_of st))
  (define sst (if (<= st 1) 2 3))
  (define ic 0.0)
  (define bx #f)
  (define bw 0.0)
  (define bh 0.0)
  (define bd 0.0)
  (define sx 0.0)
  (define supsh 0.0)
  (define subdr 0.0)
  (define u0 0)
  (define ubx #f)
  (define uh 0.0)
  (define ud 0.0)
  (define uw 0.0)
  (define d0 0)
  (define dbx #f)
  (define dh 0.0)
  (define dd 0.0)
  (define dw 0.0)
  (define rw 0.0)
  (define need 0.0)
  (set! bx (lay base st x y))
  (set! bw (get_item bx 0))
  (set! bh (get_item bx 1))
  (set! bd (get_item bx 2))
  (set! ic (italic_corr base st))
  (set! sx (+ x (+ bw ic)))
  (set! supsh (B.mx (* 0.42 em) (- bh (* 0.06 em))))
  (set! subdr (B.mx (* 0.2 em) 0.0))
  ;; 上标
  (if sup
      {
        (set! u0 (B.count))
        (set! ubx (lay sup sst sx (- y supsh)))
        (set! uw (get_item ubx 0))
        (set! uh (get_item ubx 1))
        (set! ud (get_item ubx 2))
      }
      #f)
  ;; 下标（先按名义位置排，再按与上标的重叠量压低）
  (if sub
      {
        (set! d0 (B.count))
        (set! dbx (lay sub sst sx (+ y subdr)))
        (set! dw (get_item dbx 0))
        (set! dh (get_item dbx 1))
        (set! dd (get_item dbx 2))
        (if sup
            {
              ;; 下标顶端不得高于上标底端（留 0.08em 间隙）
              (set! need (+ dh (+ (- 0 supsh) (+ ud (* 0.08 em)))))
              (if (> need subdr)
                  {
                    (B.shift d0 (B.count) 0.0 (- need subdr))
                    (set! subdr need)
                  }
                  #f)
            }
            #f)
      }
      #f)
  (set! rw (B.mx uw dw))
  `(,(B.sum3 (+ bw ic) rw (* 0.03 em))
    ,(B.mx bh (if sup (+ supsh uh) 0.0))
    ,(B.mx bd (if sub (+ subdr dd) 0.0)))))

;; ---------------------------------------------------------------------------
;; 大运算符本体（符号放大到轴线居中；名字用正体文本）
;; ---------------------------------------------------------------------------

(define LIMIT_OPS '(8721 8719 8720 8899 8898 8897 8896))

(define  lay_bigop_body (lambda (op st x y)
  (define kind (get_item op 1))
  (define val (get_item op 2))
  (define em (em_of st))
  (define big (if (<= st 1) 1.45 1.0))
  (define r #f)
  (define h 0.0)
  (define d 0.0)
  (define dy 0.0)
  (define s0 0)
  (if (== kind 1)
      {
        (set! s0 (B.count))
        (set! r (lay_text val st x y))
        ;; 名字型大运算符：垂直不居中（\sin \lim 坐在基线上）
        r
      }
      {
        (set! s0 (B.count))
        (set! r (lay_char_em val st x y big))
        (set! h (get_item r 1))
        (set! d (get_item r 2))
        ;; 让符号中心对齐数学轴线（0.26em 上方）
        (set! dy (- (- 0 (* 0.26 em)) (/ (- d h) 2)))
        (if (> (B.absf dy) 0.25)
            {
              (B.shift s0 (B.count) 0.0 dy)
              `(,(get_item r 0) ,(- h dy) ,(+ d dy))
            }
            r)
      })))

;; 上下限版（\sum_{i=1}^{n}、\lim_{x\to 0}）
;; 三段（运算符、上限、下限）各自居中到最宽者；区间平移互不重叠（按发射顺序记界）。
(define  lay_limits (lambda (op sub sup st x y)
  (define em (em_of st))
  (define sst (if (<= st 1) 2 3))
  (define o0 0)
  (define o1 0)
  (define obx #f)
  (define ow 0.0)
  (define oh 0.0)
  (define od 0.0)
  (define p0 0)
  (define p1 0)
  (define pbx #f)
  (define pw 0.0)
  (define ph 0.0)
  (define pd 0.0)
  (define q0 0)
  (define q1 0)
  (define qbx #f)
  (define qw 0.0)
  (define qh 0.0)
  (define qd 0.0)
  (define tot 0.0)
  (set! o0 (B.count))
  (set! obx (lay_bigop_body op st x y))
  (set! o1 (B.count))
  (set! ow (get_item obx 0))
  (set! oh (get_item obx 1))
  (set! od (get_item obx 2))
  (set! tot ow)
  (if sup
      {
        (set! p0 (B.count))
        (set! pbx (lay sup sst x (- y oh)))
        (set! p1 (B.count))
        (set! pw (get_item pbx 0))
        (set! ph (get_item pbx 1))
        (set! pd (get_item pbx 2))
        (B.shift p0 p1 0.0 (- 0 (+ (* 0.14 em) pd)))
        (set! tot (B.mx tot pw))
      }
      #f)
  (if sub
      {
        (set! q0 (B.count))
        (set! qbx (lay sub sst x (+ y od)))
        (set! q1 (B.count))
        (set! qw (get_item qbx 0))
        (set! qh (get_item qbx 1))
        (set! qd (get_item qbx 2))
        (B.shift q0 q1 0.0 (+ (* 0.14 em) qh))
        (set! tot (B.mx tot qw))
      }
      #f)
  ;; 水平居中
  (if (> tot ow) (B.shift o0 o1 (/ (- tot ow) 2) 0.0) #f)
  (if (and sup (> tot pw)) (B.shift p0 p1 (/ (- tot pw) 2) 0.0) #f)
  (if (and sub (> tot qw)) (B.shift q0 q1 (/ (- tot qw) 2) 0.0) #f)
  `(,tot
    ,(if sup (B.sum3 (+ oh (* 0.14 em)) pd ph) oh)
    ,(if sub (B.sum3 (+ od (* 0.14 em)) qh qd) od))))

;; 同侧版（\int_0^1、\sin^2）：符号在左，上下标在右侧
(define  lay_side_scripts (lambda (op sub sup st x y)
  (define em (em_of st))
  (define sst (if (<= st 1) 2 3))
  (define obx #f)
  (define bw 0.0)
  (define bh 0.0)
  (define bd 0.0)
  (define sx 0.0)
  (define supsh 0.0)
  (define subdr 0.0)
  (define uw 0.0)
  (define uh 0.0)
  (define ud 0.0)
  (define dw 0.0)
  (define dh 0.0)
  (define dd 0.0)
  (define ubx #f)
  (define dbx #f)
  (define d0 0)
  (define need 0.0)
  (set! obx (lay_bigop_body op st x y))
  (set! bw (get_item obx 0))
  (set! bh (get_item obx 1))
  (set! bd (get_item obx 2))
  (set! sx (+ x (+ bw (* 0.16 em))))
  (set! supsh (B.mx (* 0.42 em) (- bh (* 0.12 em))))
  (set! subdr (B.mx (* 0.2 em) (* 0.35 bd)))
  (if sup
      {
        (set! ubx (lay sup sst sx (- y supsh)))
        (set! uw (get_item ubx 0))
        (set! uh (get_item ubx 1))
        (set! ud (get_item ubx 2))
      }
      #f)
  (if sub
      {
        (set! d0 (B.count))
        (set! dbx (lay sub sst sx (+ y subdr)))
        (set! dw (get_item dbx 0))
        (set! dh (get_item dbx 1))
        (set! dd (get_item dbx 2))
        (if sup
            {
              (set! need (+ dh (+ (- 0 supsh) (+ ud (* 0.08 em)))))
              (if (> need subdr)
                  {
                    (B.shift d0 (B.count) 0.0 (- need subdr))
                    (set! subdr need)
                  }
                  #f)
            }
            #f)
      }
      #f)
  `(,(B.sum4 bw (* 0.06 em) (B.mx uw dw) (* 0.03 em))
    ,(B.mx bh (if sup (+ supsh uh) 0.0))
    ,(B.mx bd (if sub (+ subdr dd) 0.0)))))

(define  lay_bigop_scripts (lambda (op sub sup st x y)
  (define lim (get_item op 3))
  (define kind (get_item op 1))
  (define val (get_item op 2))
  (define above #f)
  (set! above
    (cond
      ((== lim 1) #t)
      ((== lim 2) #f)
      ((== kind 1) (<= st 1))            ;; \lim \max 等：显示/正文样式都置下
      (else (and (<= st 1) (mem_int val LIMIT_OPS)))))
  (if above
      (lay_limits op sub sup st x y)
      (lay_side_scripts op sub sup st x y))))

;; ---------------------------------------------------------------------------
;; 分式 / 二项式系数
;; ---------------------------------------------------------------------------

(define  lay_frac (lambda (node st x y)
  (define num (get_item node 1))
  (define den (get_item node 2))
  (define ruleon (get_item node 3))
  (define em (em_of st))
  (define cst (B.mn (+ st 1) 3))
  (define axis (* 0.26 em))
  (define rth (B.mx 1.0 (* 0.055 em)))
  (define clr (* 0.16 em))
  (define pad (* 0.1 em))
  (define n0 0)
  (define nbx #f)
  (define n1 0)
  (define dbx #f)
  (define d1 0)
  (define nw 0.0)
  (define nh 0.0)
  (define nd 0.0)
  (define dw 0.0)
  (define dh 0.0)
  (define dd 0.0)
  (define w 0.0)
  (define ny 0.0)
  (define dy 0.0)
  (define r #f)
  (set! n0 (B.count))
  (set! nbx (lay num cst 0.0 0.0))
  (set! n1 (B.count))
  (set! dbx (lay den cst 0.0 0.0))
  (set! d1 (B.count))
  (set! nw (get_item nbx 0))
  (set! nh (get_item nbx 1))
  (set! nd (get_item nbx 2))
  (set! dw (get_item dbx 0))
  (set! dh (get_item dbx 1))
  (set! dd (get_item dbx 2))
  (set! w (+ (B.mx nw dw) (* 2 pad)))
  (set! ny (- 0 (+ axis (+ (/ rth 2) (+ clr nd)))))
  (set! dy (+ (- dh (- axis (/ rth 2))) clr))
  (B.shift n0 n1 (+ x (/ (- w nw) 2)) (+ y ny))
  (B.shift n1 d1 (+ x (/ (- w dw) 2)) (+ y dy))
  (if (== ruleon 1) (B.r x (- y (+ axis (/ rth 2))) w rth) #f)
  `(,w
    ,(B.sum4 axis (/ rth 2) clr (+ nd nh))
    ,(+ (- axis (/ rth 2)) (+ clr (+ dh dd))))))

;; ---------------------------------------------------------------------------
;; 根号：矢量绘制（钩+斜杆+横线），指数放在左上
;; ---------------------------------------------------------------------------

(define  lay_sqrt (lambda (node st x y)
  (define idx (get_item node 1))
  (define body (get_item node 2))
  (define em (em_of st))
  (define b0 0)
  (define bbx #f)
  (define bw 0.0)
  (define bh 0.0)
  (define bd 0.0)
  (define rth 0.0)
  (define top 0.0)
  (define radw 0.0)
  (define lw 0.0)
  (define i0 0)
  (define ibx #f)
  (define iw 0.0)
  (define ih 0.0)
  (define idx_shift 0.0)
  (set! rth (B.mx 0.9 (* 0.05 em)))
  (set! lw 0.9)
  (set! b0 (B.count))
  (set! bbx (lay body st 0.0 y))
  (set! bw (get_item bbx 0))
  (set! bh (get_item bbx 1))
  (set! bd (get_item bbx 2))
  (set! top (- y (+ bh (* 0.17 em))))
  (set! radw (+ (* 0.34 em) (* 0.10 (B.mx bh 0.0))))
  (B.shift b0 (B.count) (+ x radw) 0.0)
  ;; 钩 + 斜杆 + 横线
  (B.l (+ x (* 0.02 em)) (- y (* 0.16 em)) (+ x (* 0.12 em)) (- y (* 0.24 em)) lw)
  (B.l (+ x (* 0.12 em)) (- y (* 0.24 em)) (+ x (* 0.24 em)) top lw)
  (B.r (+ x (* 0.22 em)) (- top (/ rth 2)) (+ (- (+ radw bw) (* 0.22 em)) (* 0.08 em)) rth)
  ;; 指数
  (if idx
      {
        (set! i0 (B.count))
        (set! ibx (lay idx 3 0.0 top))
        (set! iw (get_item ibx 0))
        (set! ih (get_item ibx 1))
        (set! idx_shift (- (* 0.30 em) iw))
        (if (< idx_shift 0.0) (set! idx_shift 0.0) #f)
        (B.shift i0 (B.count) (+ x idx_shift) (* 0.08 em))
      }
      #f)
  `(,(B.sum3 radw bw (* 0.06 em))
    ,(B.sum3 (- y top) (/ rth 2) (if idx (+ ih (* 0.1 em)) 0.0))
    ,bd)))

;; ---------------------------------------------------------------------------
;; 定界符：小尺寸用字形，大尺寸用矢量曲线
;; ---------------------------------------------------------------------------

;; 返回 (宽 上高 下深)
(define  draw_delim_vec (lambda (cp x y hr dr em)
  (define lw 0.9)
  (define H (+ hr dr))
  (define mid (- y (/ (- hr dr) 2)))
  (define top (- y hr))
  (define bot (+ y dr))
  (define w 0.0)
  (define PI (Math.PI))
  (cond
    ;; 圆括号：半椭圆
    ((== cp 40)
      {
        (set! w (+ (* 0.28 em) (* 0.12 H)))
        (B.arc (+ x w) mid w (/ H 2) (- 0 (/ PI 2)) (/ PI 2) 14 lw)
        `(,w ,hr ,dr)
      })
    ((== cp 41)
      {
        (set! w (+ (* 0.28 em) (* 0.12 H)))
        (B.arc x mid w (/ H 2) (/ PI 2) (- 0 (/ PI 2)) 14 lw)
        `(,w ,hr ,dr)
      })
    ;; 方括号：竖线 + 上下横挑
    ((== cp 91)
      {
        (set! w (* 0.3 em))
        (B.l (+ x (* 0.1 em)) top (+ x (* 0.1 em)) bot lw)
        (B.l (+ x (* 0.1 em)) top (+ x w) top lw)
        (B.l (+ x (* 0.1 em)) bot (+ x w) bot lw)
        `(,w ,hr ,dr)
      })
    ((== cp 93)
      {
        (set! w (* 0.3 em))
        (B.l (+ x (- w (* 0.1 em))) top (+ x (- w (* 0.1 em))) bot lw)
        (B.l x top (+ x (- w (* 0.1 em))) top lw)
        (B.l x bot (+ x (- w (* 0.1 em))) bot lw)
        `(,w ,hr ,dr)
      })
    ;; 花括号：两段四分之一椭圆 + 中央尖角
    ((== cp 123)
      {
        (set! w (* 0.42 em))
        (B.arc (+ x w) mid w (- 0 (/ H 2)) (/ PI 2) (* 0.94 PI) 10 lw)
        (B.arc (+ x w) mid w (/ H 2) (* 1.06 PI) (* 1.5 PI) 10 lw)
        (B.l (+ x (* 0.04 w)) (- mid (* 0.16 H)) x mid lw)
        (B.l x mid (+ x (* 0.04 w)) (+ mid (* 0.16 H)) lw)
        `(,w ,hr ,dr)
      })
    ((== cp 125)
      {
        (set! w (* 0.42 em))
        (B.arc x mid w (- 0 (/ H 2)) (* 0.5 PI) (* 0.94 PI) 10 lw)
        (B.arc x mid w (/ H 2) (* 1.06 PI) (* 1.5 PI) 10 lw)
        (B.l (+ x (* 0.96 w)) (- mid (* 0.16 H)) (+ x w) mid lw)
        (B.l (+ x w) mid (+ x (* 0.96 w)) (+ mid (* 0.16 H)) lw)
        `(,w ,hr ,dr)
      })
    ;; 尖括号
    ((== cp 10216)
      {
        (set! w (* 0.32 em))
        (B.l (+ x w) top x mid lw)
        (B.l x mid (+ x w) bot lw)
        `(,w ,hr ,dr)
      })
    ((== cp 10217)
      {
        (set! w (* 0.32 em))
        (B.l x top (+ x w) mid lw)
        (B.l (+ x w) mid x bot lw)
        `(,w ,hr ,dr)
      })
    ;; 竖线
    ((== cp 124)
      {
        (set! w (* 0.14 em))
        (B.l (+ x (* 0.07 em)) top (+ x (* 0.07 em)) bot lw)
        `(,w ,hr ,dr)
      })
    ((== cp 8214)
      {
        (set! w (* 0.3 em))
        (B.l (+ x (* 0.08 em)) top (+ x (* 0.08 em)) bot lw)
        (B.l (+ x (* 0.22 em)) top (+ x (* 0.22 em)) bot lw)
        `(,w ,hr ,dr)
      })
    ;; 地板/天花板
    ((== cp 8970)
      {
        (set! w (* 0.3 em))
        (B.l (+ x (* 0.1 em)) top (+ x (* 0.1 em)) (- bot (/ lw 2)) lw)
        (B.l (+ x (* 0.1 em)) (- bot (/ lw 2)) (+ x w) (- bot (/ lw 2)) lw)
        `(,w ,hr ,dr)
      })
    ((== cp 8971)
      {
        (set! w (* 0.3 em))
        (B.l (+ x (- w (* 0.1 em))) top (+ x (- w (* 0.1 em))) (- bot (/ lw 2)) lw)
        (B.l x (- bot (/ lw 2)) (+ x (- w (* 0.1 em))) (- bot (/ lw 2)) lw)
        `(,w ,hr ,dr)
      })
    ((== cp 8968)
      {
        (set! w (* 0.3 em))
        (B.l (+ x (* 0.1 em)) (+ top (/ lw 2)) (+ x (* 0.1 em)) bot lw)
        (B.l (+ x (* 0.1 em)) (+ top (/ lw 2)) (+ x w) (+ top (/ lw 2)) lw)
        `(,w ,hr ,dr)
      })
    ((== cp 8969)
      {
        (set! w (* 0.3 em))
        (B.l (+ x (- w (* 0.1 em))) (+ top (/ lw 2)) (+ x (- w (* 0.1 em))) bot lw)
        (B.l x (+ top (/ lw 2)) (+ x (- w (* 0.1 em))) (+ top (/ lw 2)) lw)
        `(,w ,hr ,dr)
      })
    ;; 其他：把字形纵向拉伸兜底
    (else
      {
        (set! w (* 0.45 em))
        (B.g x y em 1 1.35 0 0 cur_bold (font_of em))
        `(,w ,hr ,dr)
      }))))

(define  draw_delim (lambda (cp x y hr dr em)
  (define fid (font_of em))
  (define nat (nat_of fid))
  (define k (/ em nat))
  (define e (mc_get fid cp))
  (define gw (get_item e 5))
  (define adv 0.0)
  (define asc 0.0)
  (define dsc 0.0)
  (define sy 1.0)
  (define gy 0.0)
  (if (== gw 0)
      (draw_delim_vec cp x y hr dr em)
      {
        (set! adv (* (get_item e 2) k))
        (set! asc (* (get_item e 3) k))
        (set! dsc (* (get_item e 4) k))
        ;; 需要的纵向拉伸比：字形的天然墨迹高为 asc+dsc
        (if (> (+ asc dsc) 0.0) (set! sy (/ (+ hr dr) (+ asc dsc))) (set! sy 1.0))
        (if (< sy 1.0) (set! sy 1.0) #f)
        (cond
          ;; ① 天然尺寸就够：直接用字形
          ((and (>= asc hr) (>= dsc dr))
            {
              (B.g x y em 1 1 cp 0 cur_bold fid)
              `(,adv ,(B.mx asc hr) ,(B.mx dsc dr))
            })
          ;; ② 中等拉伸（≤2.4 倍）：字形纵向拉伸——形状（尤其花括号的尖角）比矢量近似更准
          ((<= sy 2.4)
            {
              ;; 基线要下移让墨迹的上下边界正好覆盖 hr/dr
              (set! gy (- (+ y hr) (* sy asc)))
              (B.g x gy em 1 sy cp 0 cur_bold fid)
              `(,adv ,hr ,dr)
            })
          ;; ③ 很长的定界符：矢量曲线（位图拉长了会变成一条竖线/糊掉）
          (else (draw_delim_vec cp x y hr dr em)))
      })))

(define  lay_delim (lambda (node st x y)
  (define open (get_item node 1))
  (define close (get_item node 2))
  (define body (get_item node 3))
  (define em (em_of st))
  (define b0 0)
  (define bbx #f)
  (define bw 0.0)
  (define bh 0.0)
  (define bd 0.0)
  (define hr 0.0)
  (define dr 0.0)
  (define obx #f)
  (define cbx #f)
  (define ow 0.0)
  (define cw 0.0)
  (define oh 0.0)
  (define od 0.0)
  (define ch 0.0)
  (define cd 0.0)
  (define b1 0)
  (set! b0 (B.count))
  (set! bbx (lay body st 0.0 y))
  (set! b1 (B.count))
  (set! bw (get_item bbx 0))
  (set! bh (get_item bbx 1))
  (set! bd (get_item bbx 2))
  (set! hr (+ bh (* 0.1 em)))
  (set! dr (+ bd (* 0.1 em)))
  ;; 先按内容高度决定开括号尺寸（只算宽度，不发射操作）
  (if open
      {
        (set! obx (draw_delim open x y hr dr em))
        (set! ow (get_item obx 0))
        (set! oh (get_item obx 1))
        (set! od (get_item obx 2))
      }
      #f)
  (B.shift b0 b1 (+ x ow) 0.0)
  (if close
      {
        (set! cbx (draw_delim close (+ x (+ ow bw)) y hr dr em))
        (set! cw (get_item cbx 0))
        (set! ch (get_item cbx 1))
        (set! cd (get_item cbx 2))
      }
      #f)
  `(,(B.sum3 ow bw cw) ,(B.mx bh (B.mx oh ch)) ,(B.mx bd (B.mx od cd)))))

;; ---------------------------------------------------------------------------
;; 重音 / 上划线 / 下划线
;; ---------------------------------------------------------------------------

;; 把字形重音（^ ~ 等）画在本体上方：按**墨迹底边**定位（这些字形的墨迹全在基线上方，
;; desc 为负，直接按基线摆会浮空一大截）。
;; 返回：重音相对本体顶端的高度贡献。
(define  lay_accent_glyph (lambda (cp bw bh x y em2)
  (define em (* (em_of 1) em2))
  (define fid (font_of em))
  (define nat (nat_of fid))
  (define k (/ em nat))
  (define e (mc_get fid cp))
  (define adv (* (get_item e 2) k))
  (define asc (* (get_item e 3) k))
  (define dsc (* (get_item e 4) k))
  (define gap (* 0.05 em))
  (define gx (+ x (/ (- bw adv) 2)))
  (define gy (- (- (- y bh) gap) dsc))
  (B.g gx gy em 1 1 cp 0 cur_bold fid)
  (+ gap (+ asc dsc))))

(define  lay_accent (lambda (node st x y)
  (define name (get_item node 1))
  (define body (get_item node 2))
  (define em (em_of st))
  (define b0 0)
  (define bbx #f)
  (define bw 0.0)
  (define bh 0.0)
  (define bd 0.0)
  (define rth 0.0)
  (define lw 0.0)
  (define ly 0.0)
  (define ah 0.0)
  (define cy 0.0)
  (define hw 0.0)
  (define cx 0.0)
  (set! b0 (B.count))
  (set! bbx (lay body st x y))
  (set! bw (get_item bbx 0))
  (set! bh (get_item bbx 1))
  (set! bd (get_item bbx 2))
  (set! rth (B.mx 1.0 (* 0.055 em)))
  (set! lw 0.9)
  (set! cx (+ x (/ bw 2)))          ;; 重音一律以**本体中心**为基准
  (cond
    ((or (equal? name "overline") (equal? name "bar"))
      {
        (set! ly (- y (+ bh (* 0.14 em))))
        (B.r (- x (* 0.02 em)) (- ly (/ rth 2)) (+ bw (* 0.04 em)) rth)
        `(,bw ,(+ bh (+ (* 0.14 em) (/ rth 2))) ,bd)
      })
    ((equal? name "underline")
      {
        (set! ly (+ y (+ bd (* 0.12 em))))
        (B.r (- x (* 0.02 em)) ly (+ bw (* 0.04 em)) rth)
        `(,bw ,bh ,(+ bd (+ (* 0.12 em) rth)))
      })
    ((or (equal? name "vec") (equal? name "vecleft"))
      {
        (set! cy (- y (+ bh (* 0.1 em))))
        (set! hw (* 0.16 em))
        (set! ah (* 0.14 em))
        (if (equal? name "vec")
            {
              (B.l (+ x (* 0.04 em)) cy (- (+ x bw) (* 0.04 em)) cy lw)
              (B.l (- (+ x bw) (* 0.04 em)) cy (- (+ x bw) (+ (* 0.04 em) ah)) (- cy ah) lw)
              (B.l (- (+ x bw) (* 0.04 em)) cy (- (+ x bw) (+ (* 0.04 em) ah)) (+ cy ah) lw)
            }
            {
              (B.l (+ x (* 0.04 em)) cy (- (+ x bw) (* 0.04 em)) cy lw)
              (B.l (+ x (* 0.04 em)) cy (+ (+ x (* 0.04 em)) ah) (- cy ah) lw)
              (B.l (+ x (* 0.04 em)) cy (+ (+ x (* 0.04 em)) ah) (+ cy ah) lw)
            })
        `(,bw ,(+ bh (+ (* 0.1 em) ah)) ,bd)
      })
    ((equal? name "hat")
      {
        (set! ah (lay_accent_glyph 94 bw bh x y 0.95))
        `(,bw ,(+ bh ah) ,bd)
      })
    ((equal? name "tilde")
      {
        (set! cy (- y (+ bh (* 0.1 em))))
        (set! ah (* 0.16 em))
        (B.quad (- cx (/ bw 3)) cy cx (- cy ah) (+ cx (/ bw 3)) cy 8 lw)
        `(,bw ,(+ bh (+ (* 0.1 em) ah)) ,bd)
      })
    ((equal? name "dot")
      {
        (B.fc cx (- y (+ bh (* 0.09 em))) (B.mx 0.8 (* 0.055 em)))
        `(,bw ,(+ bh (* 0.15 em)) ,bd)
      })
    ((equal? name "ddot")
      {
        (B.fc (- cx (* 0.12 em)) (- y (+ bh (* 0.09 em))) (B.mx 0.8 (* 0.055 em)))
        (B.fc (+ cx (* 0.12 em)) (- y (+ bh (* 0.09 em))) (B.mx 0.8 (* 0.055 em)))
        `(,bw ,(+ bh (* 0.15 em)) ,bd)
      })
    ((equal? name "acute")
      {
        (set! cy (- y (+ bh (* 0.06 em))))
        (set! ah (* 0.16 em))
        (B.l (- cx (* 0.12 em)) cy (+ cx (* 0.12 em)) (- cy ah) lw)
        `(,bw ,(+ bh (+ (* 0.06 em) ah)) ,bd)
      })
    (else `(,bw ,bh ,bd)))))

;; ---------------------------------------------------------------------------
;; 字体切换
;; ---------------------------------------------------------------------------

(define  lay_font (lambda (node st x y)
  (define name (get_item node 1))
  (define nodes (get_item node 2))
  (define sb cur_bold)
  (define si cur_it)
  (define r #f)
  (cond
    ((equal? name "bf") (set! cur_bold 1))
    ((equal? name "bb") (set! cur_bold 1))
    ((equal? name "rm") (set! cur_it 1))
    ((equal? name "it") (set! cur_it 2))
    ((equal? name "cal") (set! cur_it 2))
    ((equal? name "bfit")
      {
        (set! cur_bold 1)
        (set! cur_it 2)
      })
    (else #f))
  (set! r (lay_row nodes st x y))
  (set! cur_bold sb)
  (set! cur_it si)
  r))

;; ---------------------------------------------------------------------------
;; 矩阵 / 方程组 / aligned
;; ---------------------------------------------------------------------------

(define  lay_matrix (lambda (node st x y)
  (define kind (get_item node 1))
  (define rows (get_item node 2))
  (define em (em_of st))
  (define colgap (* 0.8 em))
  (define rowgap (* 0.45 em))
  (define r 0)
  (define c 0)
  (define ncell 0)
  (define cell #f)
  (define cw 0.0)
  (define rh 0.0)
  (define rd 0.0)
  (define totw 0.0)
  (define cbase 0.0)
  (define ytop 0.0)
  (define base 0.0)
  (define dx 0.0)
  (define ow 0.0)
  (define obx #f)
  (define cbx #f)
  (define cw2 0.0)
  (define hr 0.0)
  (define dr 0.0)
  (define align 0)
  (define off 0.0)
  (define rec #f)
  (define cells #f)
  ;; 矩阵的状态全部是**局部量**：单元格里可以再嵌一个矩阵（递归调用不会互相踩）
  (define mrowbuf (B.empty))   ;; 每行：cell 记录列表
  (define mrowrec (B.empty))   ;; 每行记录：(宽 上高 下深)
  (define mcolw (B.empty))     ;; 每列宽
  (define mncol 0)
  (define mnrow (length rows))
  (define mtotal 0.0)
  (define pad 0.0)
  (define mcellrec #f)
  (define mr #f)
  (define mst0 0)
  (define mbx #f)
  ;; ① 逐个单元格排版到原点，记录区间与尺寸
  (set! r 0)
  (while (< r mnrow)
    {
      (set! cells (get_item rows r))
      (set! ncell (length cells))
      (set! mr (B.empty))
      (set! rh 0.0)
      (set! rd 0.0)
      (set! c 0)
      (while (< c ncell)
        {
          (set! mst0 (B.count))
          (set! mbx (lay `("g" ,(get_item cells c)) st 0.0 0.0))
          (set! mcellrec `(,mst0 ,(- (B.count) mst0)
                           ,(get_item mbx 0) ,(get_item mbx 1) ,(get_item mbx 2)))
          (push mr mcellrec)
          (set! rh (B.mx rh (get_item mbx 1)))
          (set! rd (B.mx rd (get_item mbx 2)))
          (set! c (+ c 1))
        })
      (push mrowbuf mr)
      (push mrowrec `(,rh ,rd))
      (if (> ncell mncol) (set! mncol ncell) #f)
      (set! r (+ r 1))
    })
  ;; ② 列宽 = 该列最宽单元格
  (set! c 0)
  (while (< c mncol)
    {
      (set! cw 0.0)
      (set! r 0)
      (while (< r mnrow)
        {
          (set! mr (get_item mrowbuf r))
          (if (< c (length mr))
              (set! cw (B.mx cw (get_item (get_item mr c) 2)))
              #f)
          (set! r (+ r 1))
        })
      (push mcolw cw)
      (set! c (+ c 1))
    })
  ;; ③ 总宽与行基线
  (set! totw 0.0)
  (set! c 0)
  (while (< c mncol)
    {
      (set! totw (+ totw (get_item mcolw c)))
      (if (> c 0) (set! totw (+ totw colgap)) #f)
      (set! c (+ c 1))
    })
  (set! mtotal 0.0)
  (set! r 0)
  (while (< r mnrow)
    {
      (set! rec (get_item mrowrec r))
      (set! mtotal (+ mtotal (get_item rec 0)))
      (set! mtotal (+ mtotal (get_item rec 1)))
      (if (> r 0) (set! mtotal (+ mtotal rowgap)) #f)
      (set! r (+ r 1))
    })
  ;; ④ 定界符宽度（先算，单元格整体右移让开）
  (set! hr (/ mtotal 2))
  (set! dr (- mtotal hr))
  (set! ow 0.0)
  (set! cw2 0.0)
  (set! pad (* 0.1 em))
  (if (<= kind 6)
      {
        (set! obx (draw_delim (delim_open kind) x y hr dr em))
        (set! ow (+ (get_item obx 0) pad))
      }
      #f)
  ;; ⑤ 逐行逐列平移单元格
  (set! ytop (- y hr))
  (set! base ytop)
  (set! r 0)
  (while (< r mnrow)
    {
      (set! rec (get_item mrowrec r))
      (set! base (+ ytop (get_item rec 0)))
      (set! cbase x)
      (set! cells (get_item mrowbuf r))
      (set! c 0)
      (while (< c (length cells))
        {
          (set! mcellrec (get_item cells c))
          (set! align 0)
          (cond
            ((== kind 6) (set! align 0))                    ;; cases：左对齐
            ((== kind 7) (if (== (mod c 2) 0.0) (set! align 1) (set! align 0)))
            (else (set! align 2)))                          ;; 其余居中
          (set! off 0.0)
          (cond
            ((== align 0) (set! off 0.0))
            ((== align 1) (set! off (- (get_item mcolw c) (get_item mcellrec 2))))
            (else (set! off (/ (- (get_item mcolw c) (get_item mcellrec 2)) 2))))
          (set! dx (+ ow (+ cbase off)))
          (B.shift (get_item mcellrec 0)
                   (+ (get_item mcellrec 0) (get_item mcellrec 1))
                   dx (- base 0.0))
          (set! cbase (+ cbase (+ (get_item mcolw c) colgap)))
          (set! c (+ c 1))
        })
      (set! ytop (+ ytop (B.sum3 (get_item rec 0) (get_item rec 1) rowgap)))
      (set! r (+ r 1))
    })
  ;; ⑥ 闭合定界符
  (if (<= kind 5)
      {
        (set! cbx (draw_delim (delim_close kind) (+ x (+ ow totw)) y hr dr em))
        (set! cw2 (+ (get_item cbx 0) pad))
      }
      #f)
  `(,(B.sum3 ow totw cw2) ,hr ,dr)))

(define  delim_open (lambda (kind)
  (cond ((== kind 1) 40) ((== kind 2) 91) ((== kind 3) 123)
        ((== kind 4) 124) ((== kind 5) 8214) ((== kind 6) 123) (else 40))))

(define  delim_close (lambda (kind)
  (cond ((== kind 1) 41) ((== kind 2) 93) ((== kind 3) 125)
        ((== kind 4) 124) ((== kind 5) 8214) (else 41))))

;; ---------------------------------------------------------------------------
;; 分发
;; ---------------------------------------------------------------------------

(define  lay (lambda (node st x y)
  (define tag #f)
  (if (not node) `(0.0 0.0 0.0)
    {
  (set! tag (get_item node 0))
  (cond
    ((equal? tag "c") (lay_char (get_item node 1) st x y))
    ((equal? tag "g") (lay_row (get_item node 1) st x y))
    ((equal? tag "f") (lay_frac node st x y))
    ((equal? tag "r") (lay_sqrt node st x y))
    ((equal? tag "s") (lay_scripts node st x y))
    ((equal? tag "o") (lay_bigop_body node st x y))
    ((equal? tag "d") (lay_delim node st x y))
    ((equal? tag "a") (lay_accent node st x y))
    ((equal? tag "t") (lay_text (get_item node 1) st x y))
    ((equal? tag "F") (lay_font node st x y))
    ((equal? tag "k") `(,(* (get_item node 1) (em_of st)) 0.0 0.0))
    ((equal? tag "m") (lay_matrix node st x y))
    ((equal? tag "x") (lay_text (get_item node 1) st x y))
    (else `(0.0 0.0 0.0)))})))

(define  lay_scripts (lambda (node st x y)
  (define base (get_item node 1))
  (define sub (get_item node 2))
  (define sup (get_item node 3))
  (if (equal? (get_item base 0) "o")
      (lay_bigop_scripts base sub sup st x y)
      (lay_ord_scripts base sub sup st x y))))

;; ---------------------------------------------------------------------------
;; 对外入口：排版一个公式树
;;   返回 (宽 上高 下深)；绘制操作在 tex_base 的缓冲里（下标 0..count-1）
;; ---------------------------------------------------------------------------

(define  layout (lambda (ast st)
  (B.reset)
  (set! mcache (B.empty))
  (set! cur_it 0)
  (set! cur_bold 0)
  (lay ast st 0.0 0.0)))
