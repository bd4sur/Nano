;; ============================================================================
;; mathjax.scm —— 灵机「LaTeX 公式排版」（类 MathJax 的数学公式渲染器）
;;
;; 输入：MathJax 兼容的 LaTeX 数学串（不含 $ 定界符）
;; 输出：按 TeX 规则的排版结果（D/T/S/SS 四级字号、atom 间距、分式轴线、
;;       按内容高度伸缩的定界符、大运算符的上下限……），直接画到帧缓冲上。
;;
;; 组成（本文件只做界面与交互，排版能力全部来自 lib/）：
;;   app/lib/tex_parse.scm   词法 + 语法分析（LaTeX 子集 → 语法树）
;;   app/lib/tex_layout.scm  盒子排版（TeX 风格的尺寸/间距/上下标/分式/根号/矩阵）
;;   app/lib/tex_sym.scm     字模里没有的数学符号（矢量绘制 / 字形翻转镜像合成）
;;   app/lib/tex_draw.scm    操作流绘制（视口裁剪 + 线段 Cohen–Sutherland 裁剪）
;;   app/lib/tex_base.scm    操作缓冲 + 曲线工具
;;
;; 支持的 LaTeX 子集（覆盖高等数学/普通物理常用写法）：
;;   分式 \frac \dfrac \tfrac \cfrac、二项式 \binom、根号 \sqrt[n]{}
;;   上下标 ^ _（含 \limits \nolimits）、大运算符 \sum \prod \int \iint \iiint \oint
;;   \bigcup \bigcap，带限函数名 \lim \max \min \sup \inf \det \gcd，正体函数名
;;   \sin \cos \tan \log \ln \exp …；\left( \right) 及 \big… 系列（含 \left.）
;;   重音 \vec \hat \tilde \bar \overline \underline \dot \ddot \acute
;;   字体 \mathrm \mathbf \mathit \mathcal \mathbb \boldsymbol
;;   正文 \text \mbox、空白 \, \; \: \! \quad \qquad
;;   矩阵/方程组/对齐 \begin{matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|cases|aligned}
;;   希腊字母（大小写）、常用关系/运算/集合/箭头/杂项符号（缺字用矢量或字形变形补齐）
;;   不支持的：自定义宏/宏包、\over 一类中缀形式、多行公式编号 \tag、颜色、图片
;;
;; 交互：
;;   ◀ ▶ 切换内置示例公式；「编辑」弹出输入框（触屏软键盘含 \ { } ^ _）；
;;   − + 缩放（重新排版，不是位图放大），「1:1」回到自适应；公式区按住拖动可平移；
;;   右上角「退出」与硬 ESC 走全局确认框（app/lib/exit.scm）。
;;
;; 排版性能：一次排版的成本 ≈ 每个字形 1 次 native 度量调用 + 1 个绘制操作。
;; 所以只在「换公式 / 缩放 / 换字号」时重排，拖动平移只重画不重排；每拍只在脏时推屏。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Ui)
(native App)

(import EX "app/lib/exit.scm")
(import P "app/lib/tex_parse.scm")
(import L "app/lib/tex_layout.scm")
(import D "app/lib/tex_draw.scm")
(import B "app/lib/tex_base.scm")

;; ---------------------------------------------------------------------------
;; 画布与配色
;; ---------------------------------------------------------------------------

(define SW 320)
(define SH 240)
(define TITLE_H 18)
(define SRC_Y 203)          ;; 源文本行
(define BTN_Y 220)
(define BTN_H 18)
(define VIEW_Y 19)
(define VIEW_H 182)         ;; 公式视口 19..201

(define COL_BG_R 18)
(define COL_BG_G 20)
(define COL_BG_B 26)
(define COL_VIEW_R 26)
(define COL_VIEW_G 28)
(define COL_VIEW_B 36)
(define COL_FG_R 234)
(define COL_FG_G 236)
(define COL_FG_B 244)
(define COL_DIM_R 122)
(define COL_DIM_G 128)
(define COL_DIM_B 146)
(define COL_BTN_R 52)
(define COL_BTN_G 56)
(define COL_BTN_B 68)
(define COL_EDGE_R 96)
(define COL_EDGE_G 102)
(define COL_EDGE_B 120)
(define COL_ON_R 96)
(define COL_ON_G 168)
(define COL_ON_B 255)

;; ---------------------------------------------------------------------------
;; 内置示例公式（覆盖：分式/根号/上下标/大运算符/矩阵/方程组/对齐/重音/字体/正文）
;; ---------------------------------------------------------------------------

(define DEMOS '(
  "x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}"
  "e^{i\\pi}+1=0"
  "e^x=\\sum_{n=0}^{\\infty}\\frac{x^n}{n!}"
  "\\int_{-\\infty}^{\\infty}e^{-x^2}\\,dx=\\sqrt{\\pi}"
  "\\lim_{x\\to 0}\\frac{\\sin x}{x}=1"
  "\\frac{d}{dx}\\left(\\sin x\\right)=\\cos x"
  "\\vec{F}=m\\vec{a},\\quad \\vec{F}=-\\frac{Gm_1m_2}{r^2}\\hat{r}"
  "\\nabla\\times\\vec{E}=-\\frac{\\partial\\vec{B}}{\\partial t},\\quad \\nabla\\cdot\\vec{B}=0"
  "i\\hbar\\frac{\\partial\\psi}{\\partial t}=\\hat{H}\\psi"
  "\\oint_{\\partial S}\\vec{B}\\cdot d\\vec{l}=\\mu_0 I+\\mu_0\\varepsilon_0\\frac{d\\Phi_E}{dt}"
  "\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}\\begin{pmatrix}x\\\\y\\end{pmatrix}=\\begin{pmatrix}ax+by\\\\cx+dy\\end{pmatrix}"
  "|x|=\\begin{cases}x,&x\\ge 0\\\\-x,&x<0\\end{cases}"
  "(a+b)^n=\\sum_{k=0}^{n}\\binom{n}{k}a^{n-k}b^k"
  "\\forall\\varepsilon>0,\\ \\exists\\delta>0:\\ |x-a|<\\delta\\Rightarrow|f(x)-f(a)|<\\varepsilon"
  "\\begin{aligned}a^2-b^2&=(a-b)(a+b)\\\\a^3\\pm b^3&=(a\\pm b)(a^2\\mp ab+b^2)\\end{aligned}"
  "\\text{欧拉公式：}e^{i\\theta}=\\cos\\theta+i\\sin\\theta"
  "m_1\\ddot{x}_1=-k(x_1-x_2),\\quad m_2\\ddot{x}_2=k(x_1-x_2)"
  "\\mathbb{R}^n\\to\\mathbb{R}^m,\\quad \\mathbf{v}=\\frac{d\\mathbf{r}}{dt}"
))

;; ---------------------------------------------------------------------------
;; 状态
;; ---------------------------------------------------------------------------

(define cur 0)
(define src "")
(define ast #f)
(define box '(0.0 0.0 0.0))
(define zoom 1.0)        ;; 用户缩放（相对自适应）
(define fit 1.0)         ;; 自动适配系数
(define panx 0.0)
(define pany 0.0)
(define dirty #t)
(define drag 0)
(define dragx 0)
(define dragy 0)
(define tick #f)
(define msg "")

(define BTNX '(4 57 110 163 216 269))

;; 从后台/挂起恢复时补一次重绘（本 App 只在脏时推屏，否则画面会停在挂起前）
(define mark_dirty (lambda () (set! dirty #t)))
(define BTNS '("←" "→" "编辑" "-" "+" "1:1"))

;; ---------------------------------------------------------------------------
;; 排版
;; ---------------------------------------------------------------------------

;; 自动适配：先按当前缩放排一遍，超出视口就等比缩小并**重排**（不是位图缩放，
;; 这样字形始终落在字模的清晰尺度上）。
(define do_layout (lambda ()
  (define w 0.0)
  (define h 0.0)
  (define d 0.0)
  (define kx 0.0)
  (define ky 0.0)
  (define k 0.0)
  (set! ast (P.parse src))
  (L.set_emul zoom)
  (set! box (L.layout ast 0))
  (set! w (get_item box 0))
  (set! h (get_item box 1))
  (set! d (get_item box 2))
  (set! kx (/ (- SW 8.0) (B.mx w 1.0)))
  (set! ky (/ (- VIEW_H 8.0) (B.mx (+ h d) 1.0)))
  (set! k (B.mn kx ky))
  (cond
    ((> k 1.0) (set! k 1.0))
    ((< k 0.28) (set! k 0.28))
    (else #f))
  (set! fit k)
  (if (< k 0.999)
      {
        (L.set_emul (* zoom k))
        (set! box (L.layout ast 0))
      }
      #f)
  (set! panx 0.0)
  (set! pany 0.0)
  (set! dirty #t)))

(define set_source (lambda (s)
  (set! src s)
  (set! msg "")
  (do_layout)))

;; ---------------------------------------------------------------------------
;; 绘制
;; ---------------------------------------------------------------------------

(define draw_button (lambda (i on)
  (define x (get_item BTNX i))
  (define bg (if on COL_ON_R COL_BTN_R))
  (define bg2 (if on COL_ON_G COL_BTN_G))
  (define bg3 (if on COL_ON_B COL_BTN_B))
  (Gfx.fill_rect x BTN_Y 50 BTN_H bg bg2 bg3)
  (Gfx.draw_rect x BTN_Y 50 BTN_H COL_EDGE_R COL_EDGE_G COL_EDGE_B)
  (Gfx.draw_text (get_item BTNS i) (+ x 6) (+ BTN_Y 3) COL_FG_R COL_FG_G COL_FG_B)))

;; 源文本（单行，超宽截断；CJK 宽度差异忽略不计——源串以 ASCII 为主）
(define draw_src (lambda ()
  (define n (String.length src))
  (define s src)
  (if (> n 43)
      (set! s (String.concat (String.slice src 0 42) "…"))
      #f)
  (Gfx.draw_text s 4 SRC_Y COL_DIM_R COL_DIM_G COL_DIM_B)))

(define draw_all (lambda ()
  (define w 0.0)
  (define h 0.0)
  (define d 0.0)
  (define ox 0.0)
  (define oy 0.0)
  (define lab "")
  (define num "")
  ;; 背景与页眉
  (Gfx.fill_rect 0 0 SW SH COL_BG_R COL_BG_G COL_BG_B)
  (Gfx.draw_text "LaTeX 公式排版" 6 3 COL_FG_R COL_FG_G COL_FG_B)
  (set! num (String.atom_to_string (+ cur 1)))
  (set! lab (String.concat num (String.concat "/" (String.atom_to_string (length DEMOS)))))
  (Gfx.draw_text lab (- 236 (String.length lab)) 3 COL_DIM_R COL_DIM_G COL_DIM_B)
  ;; 公式视口
  (Gfx.fill_rect 0 VIEW_Y SW VIEW_H COL_VIEW_R COL_VIEW_G COL_VIEW_B)
  (Gfx.fill_rect 0 VIEW_Y SW 1 44 48 60)
  (Gfx.set_clip 0 (+ VIEW_Y 1) SW (- VIEW_H 1))
  (D.set_view 0 (+ VIEW_Y 1) SW (- VIEW_H 1))
  (set! w (get_item box 0))
  (set! h (get_item box 1))
  (set! d (get_item box 2))
  (set! ox (+ (/ (- SW w) 2) panx))
  (set! oy (+ (+ VIEW_Y (/ (- VIEW_H (+ h d)) 2)) h))
  (set! oy (+ oy pany))
  (D.draw ox oy COL_FG_R COL_FG_G COL_FG_B)
  (Gfx.reset_clip)
  ;; 源文本 + 按钮 + 退出
  (draw_src)
  (Gfx.draw_text msg 4 (+ SRC_Y 0) COL_ON_R COL_ON_G COL_ON_B)
  (draw_button 0 #f)
  (draw_button 1 #f)
  (draw_button 2 #f)
  (draw_button 3 #f)
  (draw_button 4 #f)
  (draw_button 5 (if (< (B.absf (- zoom 1.0)) 0.001) #f #t))
  (EX.draw_button)
  (set! dirty #f)
  (Gfx.refresh)))

;; ---------------------------------------------------------------------------
;; 交互
;; ---------------------------------------------------------------------------

(define go_formula (lambda (i)
  (define n (length DEMOS))
  (set! cur (mod (+ i n) n))
  (set! zoom 1.0)
  (set_source (get_item DEMOS cur))))

(define zoom_by (lambda (k)
  (set! zoom (* zoom k))
  (if (< zoom 0.45) (set! zoom 0.45) #f)
  (if (> zoom 3.0) (set! zoom 3.0) #f)
  (do_layout)))

(define edit_source (lambda ()
  (define r #f)
  (set! r (Ui.input_with "输入 LaTeX 公式" src))
  (if (null? r)
      (set! dirty #t)
      {
        (set! src r)
        (set! msg "")
        (do_layout)
      })))

(define in_view (lambda (px py)
  (and (>= py VIEW_Y) (< py (+ VIEW_Y VIEW_H)))))

(define do_button (lambda (px py)
  (define i 0)
  (define hit -1)
  (while (< i 6)
    {
      (if (and (>= px (get_item BTNX i)) (< px (+ (get_item BTNX i) 50)))
          (if (and (>= py BTN_Y) (< py (+ BTN_Y BTN_H)))
              (set! hit i)
              #f)
          #f)
      (set! i (+ i 1))
    })
  (cond
    ((== hit 0) (go_formula (- cur 1)))
    ((== hit 1) (go_formula (+ cur 1)))
    ((== hit 2) (edit_source))
    ((== hit 3) (zoom_by 0.8))
    ((== hit 4) (zoom_by 1.25))
    ((== hit 5)
      {
        (set! zoom 1.0)
        (do_layout)
      })
    (else #f))))

(define on_tick (lambda ()
  (define pressed #f)
  (define px 0)
  (define py 0)
  (set! pressed (Touch.is_pressed))
  (if (Touch.take_press)
      {
        (set! px (Touch.x))
        (set! py (Touch.y))
        (if (in_view px py)
            {
              (set! drag 1)
              (set! dragx px)
              (set! dragy py)
            }
            (do_button px py))
      }
      #f)
  (if (and (== drag 1) pressed)
      {
        (set! panx (+ panx (- (Touch.x) dragx)))
        (set! pany (+ pany (- (Touch.y) dragy)))
        (set! dragx (Touch.x))
        (set! dragy (Touch.y))
        (set! dirty #t)
      }
      #f)
  (if (Touch.take_release) (set! drag 0) #f)
  (EX.check)
  (if dirty (draw_all) #f)))

;; ---------------------------------------------------------------------------
;; 启动
;; ---------------------------------------------------------------------------

(EX.install_with mark_dirty)
(set! cur 0)
(set! src (get_item DEMOS 0))
(do_layout)
(draw_all)
(set! tick (System.set_interval 33 on_tick))
