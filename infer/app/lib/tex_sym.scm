;; ============================================================================
;; tex_sym.scm —— 字模里没有的数学符号（矢量绘制 / 字形变形合成）
;;
;; 12px 内嵌字模覆盖了 ∫ ∮ ∑ ∏ √ ∈ ∪ ∩ ⊥ ∠ ∝ ≈ ≤ ≥ → ← ↑ ↓ ∴ ∵ ∞ ′ ″ …
;; 但缺 ∂ ∇ ∅ ∀ ∃ ⊂ ⊃ ⊆ ⊇ ⇒ ⇐ ⇔ ↔ ↦ ∼ ≃ ≪ ≫ ⋯ ∘ ⊕ ⊗ ∓ ℏ ℓ ℝ ⟨⟩ ∥ ¬ △ 等
;; 常用数学/物理符号。本模块用两种手段补齐：
;;   · **字形变形合成**：翻转（∀ = 倒 A、∇ = 倒 Δ）、镜像（∃ = 反 E、∋ = 反 ∈）、
;;     重复（∬ = 两个 ∫）、叠加（∅ = 0 + 斜线、≃ = = 上方加波浪、⊕ = ○ + +）；
;;   · **矢量绘制**：⊂ ⊇ ⟨⟩ ⇒ ⇔ ↦ ∼ ≪ ≫ ⋯ ⋮ ⋱ ∘ ¬ △ 等直接用线段/曲线画，
;;     与字号无关地保持清晰（这正是位图字形做不到的）。
;;
;; 对外只暴露一个过程：(draw cp x y em fid bold) → (宽 上高 下深) 或 #f
;; 返回 #f 表示"不是本模块负责的符号"，由排版层继续走字形路径。
;; 未识别的缺字最终由 fallback 画一个 '?'（缺字可见，便于自查源码）。
;; ============================================================================

(native Gfx)
(native Math)

(import B "app/lib/tex_base.scm")

(define PI2 6.283185307179586)
(define PI (Math.PI))

(define  nat (lambda (fid) (if (== fid 2) 16 12)))
(define  lw_of (lambda (em) (B.mx 0.9 (* 0.055 em))))

;; 字形度量（像素），返回 (adv asc desc gw)
(define mm1 #f)
(define  met (lambda (cp em fid)
  (define k (/ em (nat fid)))
  (set! mm1 (Gfx.glyph_metrics fid cp))
  (if (null? mm1)
      `(,em 0.0 0.0 0.0)
      `(,(* (get_item mm1 0) k) ,(* (get_item mm1 1) k)
        ,(* (get_item mm1 2) k) ,(* (get_item mm1 3) k)))))

;; 画一个字形（可翻转/镜像），返回 (w h d)
(define  emit_glyph (lambda (cp x y em fid sx sy shear bold)
  (define m (met cp em fid))
  (define adv (get_item m 0))
  (define asc (get_item m 1))
  (define dsc (get_item m 2))
  (define gw (get_item m 3))
  (define px x)
  (define py y)
  (define w adv)
  (define h asc)
  (define d (B.mx dsc 0.0))
  (if (> sy 0.0)
      #f
      {
        ;; 纵向翻转：墨迹全部落到基线以下 → 上移 asc，占据 [y-asc, y]
        (set! py (- y asc))
        (set! h gw)
        (set! d 0.0)
      })
  (if (< sx 0.0)
      {
        ;; 横向镜像：墨迹向 pen 左侧展开 → pen 右移一个字形宽
        (set! px (+ x gw))
        (set! w gw)
      }
      #f)
  (B.g px py em sx sy cp shear bold fid)
  `(,w ,h ,d)))

;; ---------------------------------------------------------------------------
;; 箭头（单/双线、单/双头）
;; ---------------------------------------------------------------------------

(define  sym_arrow (lambda (x y em left right dbl)
  (define w (* 1.0 em))
  (define cy (- y (* 0.3 em)))
  (define lw (lw_of em))
  (define hd (* 0.22 em))
  (define hw (* 0.3 em))
  (define gap (* 0.17 em))
  (define tipy cy)
  (if dbl
      {
        (B.l x (- cy (/ gap 2)) (+ x w) (- cy (/ gap 2)) lw)
        (B.l x (+ cy (/ gap 2)) (+ x w) (+ cy (/ gap 2)) lw)
      }
      (B.l x cy (+ x w) cy lw))
  (if right
      {
        (B.l (+ x w) tipy (- (+ x w) hw) (- tipy hd) lw)
        (B.l (+ x w) tipy (- (+ x w) hw) (+ tipy hd) lw)
      }
      #f)
  (if left
      {
        (B.l x tipy (+ x hw) (- tipy hd) lw)
        (B.l x tipy (+ x hw) (+ tipy hd) lw)
      }
      #f)
  `(,w ,(+ (* 0.3 em) hd) ,(B.mx 0.0 (- 0 hd)))))

;; ↦：右箭头 + 左端竖杠
(define  sym_mapsto (lambda (x y em)
  (define w (* 0.95 em))
  (define cy (- y (* 0.3 em)))
  (define lw (lw_of em))
  (define hd (* 0.2 em))
  (define hw (* 0.26 em))
  (B.l x (- cy (* 0.16 em)) x (+ cy (* 0.16 em)) lw)
  (B.l x cy (+ x w) cy lw)
  (B.l (+ x w) cy (- (+ x w) hw) (- cy hd) lw)
  (B.l (+ x w) cy (- (+ x w) hw) (+ cy hd) lw)
  `(,w ,(+ (* 0.3 em) hd) 0.0)))

;; ---------------------------------------------------------------------------
;; 集合/关系符号
;; ---------------------------------------------------------------------------

(define  sym_subset (lambda (x y em mir under)
  (define h (* 0.6 em))
  (define w (* 0.34 em))
  (define cy (- y (/ h 2)))
  (define lw (lw_of em))
  (define rx (* 2.0 w))
  (if mir
      (B.arc (- (+ x w) rx) cy rx (/ h 2) (- 0 (/ PI 3)) (/ PI 3) 10 lw)
      (B.arc (+ x rx) cy rx (/ h 2) (- PI (/ PI 3)) (+ PI (/ PI 3)) 10 lw))
  (if under
      (B.r x (+ y (* 0.1 em)) w (* 0.055 em))
      #f)
  `(,w ,h ,(if under (* 0.1 em) 0.0))))

(define  sym_angle_bracket (lambda (x y em close)
  (define w (* 0.32 em))
  (define h (* 0.62 em))
  (define lw (lw_of em))
  (if close
      {
        (B.l x (- y h) (+ x w) (- y (/ h 2)) lw)
        (B.l (+ x w) (- y (/ h 2)) x y lw)
      }
      {
        (B.l (+ x w) (- y h) x (- y (/ h 2)) lw)
        (B.l x (- y (/ h 2)) (+ x w) y lw)
      })
  `(,w ,h 0.0)))

(define  sym_ll (lambda (x y em)
  (define w (* 0.62 em))
  (define h (* 0.5 em))
  (define lw (lw_of em))
  (define dy (* 0.12 em))
  (B.l (+ x (* 0.24 em)) (- y h) x (- y (/ h 2)) lw)
  (B.l x (- y (/ h 2)) (+ x (* 0.24 em)) y lw)
  (B.l (+ x w) (- y h) (+ x (* 0.4 em)) (- y (/ h 2)) lw)
  (B.l (+ x (* 0.4 em)) (- y (/ h 2)) (+ x w) y lw)
  `(,w ,h 0.0)))

(define  sym_gg (lambda (x y em)
  (define w (* 0.62 em))
  (define h (* 0.5 em))
  (define lw (lw_of em))
  (B.l x (- y h) (+ x (* 0.24 em)) (- y (/ h 2)) lw)
  (B.l (+ x (* 0.24 em)) (- y (/ h 2)) x y lw)
  (B.l (+ x (* 0.4 em)) (- y h) (+ x w) (- y (/ h 2)) lw)
  (B.l (+ x w) (- y (/ h 2)) (+ x (* 0.4 em)) y lw)
  `(,w ,h 0.0)))

;; ∼：轴线上方的波浪线
(define  sym_sim (lambda (x y em)
  (define w (* 0.6 em))
  (define cy (- y (* 0.3 em)))
  (define lw (lw_of em))
  (B.quad x cy (+ x (* 0.15 em)) (- cy (* 0.2 em)) (+ x (* 0.3 em)) cy 6 lw)
  (B.quad (+ x (* 0.3 em)) cy (+ x (* 0.45 em)) (+ cy (* 0.2 em)) (+ x w) cy 6 lw)
  `(,w ,(* 0.42 em) 0.0)))

;; ≃：= 上方加波浪
(define  sym_simeq (lambda (x y em fid bold)
  (define m (emit_glyph 61 x y em fid 1.0 1.0 0.0 bold))
  (define w (get_item m 0))
  (define lw (lw_of em))
  (define cy (- y (* 0.62 em)))
  (B.quad x cy (+ x (* 0.13 em)) (- cy (* 0.16 em)) (+ x (* 0.26 em)) cy 6 lw)
  (B.quad (+ x (* 0.26 em)) cy (+ x (* 0.39 em)) (+ cy (* 0.16 em)) (+ x w) cy 6 lw)
  `(,w ,(+ (* 0.62 em) (* 0.16 em)) 0.0)))

;; ---------------------------------------------------------------------------
;; 圆点类
;; ---------------------------------------------------------------------------

(define  sym_dots3 (lambda (x y em)
  (define w (* 0.66 em))
  (define cy (- y (* 0.26 em)))
  (define r (B.mx 0.7 (* 0.055 em)))
  (define gap (* 0.24 em))
  (B.fc x cy r)
  (B.fc (+ x gap) cy r)
  (B.fc (+ x (* 2.0 gap)) cy r)
  `(,w ,(* 0.32 em) ,(* 0.2 em))))

(define  sym_vdots (lambda (x y em)
  (define w (* 0.12 em))
  (define r (B.mx 0.7 (* 0.055 em)))
  (define gap (* 0.24 em))
  (B.fc (+ x (/ w 2)) (- y (* 0.72 em)) r)
  (B.fc (+ x (/ w 2)) (- y (* 0.38 em)) r)
  (B.fc (+ x (/ w 2)) (- y (* 0.04 em)) r)
  `(,w ,(* 0.78 em) 0.0)))

(define  sym_ddots (lambda (x y em)
  (define w (* 0.6 em))
  (define r (B.mx 0.7 (* 0.055 em)))
  (B.fc (+ x (* 0.06 em)) (- y (* 0.7 em)) r)
  (B.fc (+ x (* 0.3 em)) (- y (* 0.38 em)) r)
  (B.fc (+ x (* 0.54 em)) (- y (* 0.06 em)) r)
  `(,w ,(* 0.76 em) 0.0)))

(define  sym_bullet (lambda (x y em)
  (define w (* 0.42 em))
  (define r (B.mx 0.9 (* 0.08 em)))
  (B.fc (+ x (/ w 2)) (- y (* 0.3 em)) r)
  `(,w ,(* 0.42 em) 0.0)))

(define  sym_circ (lambda (x y em)
  (define w (* 0.4 em))
  (define r (* 0.13 em))
  (B.oc (+ x (/ w 2)) (- y (* 0.32 em)) r)
  (B.oc (+ x (/ w 2)) (- y (* 0.32 em)) r)
  `(,w ,(* 0.45 em) 0.0)))

;; ⊕ / ⊗：圆圈 + 十字/叉
(define  sym_oplus (lambda (x y em mode)
  (define w (* 0.66 em))
  (define r (* 0.3 em))
  (define cx (+ x (/ w 2)))
  (define cy (- y (* 0.3 em)))
  (define lw (lw_of em))
  (B.oc cx cy r)
  (cond
    ((== mode 1)
      {
        (B.l (- cx (* 0.7 r)) (- cy (* 0.7 r)) (+ cx (* 0.7 r)) (+ cy (* 0.7 r)) lw)
        (B.l (- cx (* 0.7 r)) (+ cy (* 0.7 r)) (+ cx (* 0.7 r)) (- cy (* 0.7 r)) lw)
      })
    ((== mode 2) (B.l (- cx r) cy (+ cx r) cy lw))
    (else
      {
        (B.l (- cx r) cy (+ cx r) cy lw)
        (B.l cx (- cy r) cx (+ cy r) lw)
      }))
  `(,w ,(+ (* 0.3 em) r) ,(- r (* 0.3 em)))))

;; ∓：− 在 + 上方
(define  sym_mpm (lambda (x y em fid bold)
  (define m (emit_glyph 43 x y em fid 1.0 1.0 0.0 bold))
  (define w (get_item m 0))
  (define lw (lw_of em))
  (B.l x (- y (* 0.72 em)) (+ x w) (- y (* 0.72 em)) lw)
  `(,w ,(- (* 0.72 em) (* 0.03 em)) ,(B.mx (get_item m 2) 0.0))))

;; ---------------------------------------------------------------------------
;; 其他
;; ---------------------------------------------------------------------------

(define  sym_emptyset (lambda (x y em fid bold)
  (define m (emit_glyph 48 x y em fid 1.0 1.0 0.0 bold))
  (define w (get_item m 0))
  (define lw (lw_of em))
  (B.l (+ x (* 0.12 em)) (+ y (* 0.02 em)) (- (+ x w) (* 0.08 em)) (- y (* 0.66 em)) lw)
  `(,w ,(get_item m 1) ,(get_item m 2))))

;; ∂ = **数字 6 的水平镜像**（字形变形合成，不画矢量）。
;; 这条是逐一比对出来的：把参考字体（DejaVu Serif）的 ∂ 镜像过来，形状就是 6；把 6
;; 镜像过去，形状就是 ∂（环在下、柄向上外挑，二者手性相反）。本字模有 6，于是
;;   ∂ = ('6', sx = −1, shear ≈ −0.12)
;; ——shear 取负是因为 sx<0 时切变项随 kx 一起变号，负值才让字形的上部像数学斜体那样
;; 向右倾。此前用矢量画的四轮尝试（圆环 + 挑尾）都因笔画过细、接头不自然而读成 "ó"。
(define  sym_partial (lambda (x y em fid bold)
  (emit_glyph 54 x y em fid -1.0 1.0 -0.12 bold)))

(define  sym_nabla (lambda (x y em fid bold)
  (define m (emit_glyph 916 x y em fid 1.0 -1.0 0.0 bold))
  m))

(define  sym_neg (lambda (x y em)
  (define w (* 0.6 em))
  (define lw (lw_of em))
  (define cy (- y (* 0.42 em)))
  (B.l x cy (+ x w) cy lw)
  (B.l (+ x w) cy (+ x w) (+ y (* 0.06 em)) lw)
  `(,w ,(* 0.48 em) 0.0)))

(define  sym_parallel (lambda (x y em)
  (define w (* 0.34 em))
  (define lw (lw_of em))
  (B.l (+ x (* 0.08 em)) (- y (* 0.6 em)) (+ x (* 0.08 em)) y lw)
  (B.l (+ x (* 0.24 em)) (- y (* 0.6 em)) (+ x (* 0.24 em)) y lw)
  `(,w ,(* 0.6 em) 0.0)))

(define  sym_backslash (lambda (x y em)
  (define w (* 0.5 em))
  (define lw (lw_of em))
  (B.l x (- y (* 0.6 em)) (+ x w) y lw)
  `(,w ,(* 0.6 em) 0.0)))

(define  sym_triangle (lambda (x y em)
  (define w (* 0.72 em))
  (define h (* 0.6 em))
  (define lw (lw_of em))
  (B.l (+ x (/ w 2)) (- y h) x y lw)
  (B.l x y (+ x w) y lw)
  (B.l (+ x w) y (+ x (/ w 2)) (- y h) lw)
  `(,w ,h 0.0)))

(define  sym_double_int (lambda (x y em fid bold n)
  (define m (emit_glyph 8747 x y em fid 1.0 1.0 0.0 bold))
  (define adv (get_item m 0))
  (define i 1)
  (while (< i n)
    {
      (B.g (+ x (* i (* 0.42 em))) y em 1.0 1.0 8747 0.0 bold fid)
      (set! i (+ i 1))
    })
  `(,(+ adv (* (- n 1) (* 0.42 em))) ,(get_item m 1) ,(get_item m 2))))

;; ℏ / ℓ
(define  sym_hbar (lambda (x y em fid bold)
  (define m (emit_glyph 104 x y em fid 1.0 1.0 0.2 bold))
  (define w (get_item m 0))
  (define lw (lw_of em))
  (B.l (+ x (* 0.1 em)) (- y (* 0.78 em)) (+ x (* 0.62 em)) (- y (* 0.78 em)) lw)
  `(,w ,(* 0.82 em) ,(get_item m 2))))

(define  sym_ell (lambda (x y em fid bold)
  (emit_glyph 108 x y em fid 1.0 1.0 0.2 bold)))

;; 双线体（ℝ ℂ ℕ ℤ ℚ）：同字形右移 0.6px 再描一次（小字号下与黑板粗体近似）
(define  sym_bb (lambda (cp x y em fid)
  (define m (emit_glyph cp x y em fid 1.0 1.0 0.0 1))
  (B.g (+ x (* 0.045 em)) y em 1.0 1.0 cp 0.0 1 fid)
  m))

;; † / ‡：竖线 + 横杠（1 或 2 条）
(define  sym_dagger (lambda (x y em fid bold n)
  (define lw (lw_of em))
  (define cx (+ x (* 0.2 em)))
  (define h (* 0.66 em))
  (B.l cx (- y h) cx y lw)
  (B.l (- cx (* 0.16 em)) (- y (* 0.52 em)) (+ cx (* 0.16 em)) (- y (* 0.52 em)) lw)
  (if (> n 1)
      (B.l (- cx (* 0.16 em)) (- y (* 0.24 em)) (+ cx (* 0.16 em)) (- y (* 0.24 em)) lw)
      #f)
  `(,(* 0.42 em) ,h 0.0)))

;; ∉ / ∤ / ∦：在关系符号上叠一条斜杠
(define  sym_not (lambda (cp x y em fid bold)
  (define lw (lw_of em))
  (define r #f)
  (cond
    ((== cp 8713) (set! r (emit_glyph 8712 x y em fid 1.0 1.0 0.0 bold)))   ;; ∉
    ((== cp 8740) (set! r (emit_glyph 124 x y em fid 1.0 1.0 0.0 bold)))    ;; ∤
    (else (set! r (emit_glyph 124 x y em fid 1.0 1.0 0.0 bold))))
  (B.l (+ x (* 0.1 em)) (- y (* 0.55 em)) (+ x (* 0.62 em)) y lw)
  r))

;; 斜向箭头（↗ ↘ ↖ ↙）：单箭头 + 斜杆
(define  sym_darrow (lambda (x y em dx dy)
  (define lw (lw_of em))
  (define len (* 0.55 em))
  (define ex (+ x (* len dx)))
  (define ey (+ (- y (* 0.3 em)) (* len dy)))
  (define hd (* 0.18 em))
  (B.l x (- y (* 0.3 em)) ex ey lw)
  ;; 箭头两撇：沿反方向 ± 垂直方向偏 hd
  (B.l ex ey (- ex (* hd (* 0.8 dx))) (- ey (* hd (* 0.8 dy))) lw)
  (B.l ex ey (- ex (* hd dy)) (+ ey (* hd dx)) lw)
  (B.l ex ey (+ ex (* hd dy)) (- ey (* hd dx)) lw)
  `(,(+ (* 0.6 em) hd) ,(+ (* 0.3 em) hd) 0.0)))

;; ---------------------------------------------------------------------------
;; 分发
;; ---------------------------------------------------------------------------

(define  fallback (lambda (x y em fid bold)
  (define m (emit_glyph 63 x y em fid 1.0 1.0 0.0 bold))
  `(,(get_item m 0) ,(get_item m 1) ,(get_item m 2))))

(define  draw (lambda (cp x y em fid bold)
  (cond
    ;; 字形变形合成
    ((== cp 8711) (sym_nabla x y em fid bold))                  ;; ∇
    ((== cp 8706) (sym_partial x y em fid bold))                ;; ∂
    ((== cp 8704) (emit_glyph 65 x y em fid 1.0 -1.0 0.0 bold)) ;; ∀ = 倒 A
    ((== cp 8705) (emit_glyph 69 x y em fid -1.0 1.0 0.0 bold)) ;; ∃ = 反 E
    ((== cp 8715) (emit_glyph 8712 x y em fid -1.0 1.0 0.0 bold)) ;; ∋ = 反 ∈
    ((== cp 8467) (sym_ell x y em fid bold))                    ;; ℓ
    ((== cp 8463) (sym_hbar x y em fid bold))                   ;; ℏ
    ((== cp 8476) (sym_bb 82 x y em fid))                       ;; ℝ
    ((== cp 8450) (sym_bb 67 x y em fid))                       ;; ℂ
    ((== cp 8469) (sym_bb 78 x y em fid))                       ;; ℕ
    ((== cp 8484) (sym_bb 90 x y em fid))                       ;; ℤ
    ((== cp 8474) (sym_bb 81 x y em fid))                       ;; ℚ
    ((== cp 8748) (sym_double_int x y em fid bold 2))           ;; ∬
    ((== cp 8749) (sym_double_int x y em fid bold 3))           ;; ∭
    ((== cp 8709) (sym_emptyset x y em fid bold))               ;; ∅
    ((== cp 8773) (emit_glyph 8776 x y em fid 1.0 1.0 0.0 bold)) ;; ≅ → ≈
    ((== cp 8771) (sym_simeq x y em fid bold))                  ;; ≃
    ;; 箭头
    ((== cp 8596) (sym_arrow x y em 1 1 0))                     ;; ↔
    ((== cp 8658) (sym_arrow x y em 0 1 1))                     ;; ⇒
    ((== cp 8656) (sym_arrow x y em 1 0 1))                     ;; ⇐
    ((== cp 8660) (sym_arrow x y em 1 1 1))                     ;; ⇔
    ((== cp 8614) (sym_mapsto x y em))                          ;; ↦
    ;; 集合关系
    ((== cp 8834) (sym_subset x y em 0 0))                      ;; ⊂
    ((== cp 8835) (sym_subset x y em 1 0))                      ;; ⊃
    ((== cp 8838) (sym_subset x y em 0 1))                      ;; ⊆
    ((== cp 8839) (sym_subset x y em 1 1))                      ;; ⊇
    ((== cp 10216) (sym_angle_bracket x y em 0))                ;; ⟨
    ((== cp 10217) (sym_angle_bracket x y em 1))                ;; ⟩
    ((== cp 8810) (sym_ll x y em))                              ;; ≪
    ((== cp 8811) (sym_gg x y em))                              ;; ≫
    ((== cp 8764) (sym_sim x y em))                             ;; ∼
    ((== cp 8943) (sym_dots3 x y em))                           ;; ⋯
    ((== cp 8942) (sym_vdots x y em))                           ;; ⋮
    ((== cp 8945) (sym_ddots x y em))                           ;; ⋱
    ((== cp 8226) (sym_bullet x y em))                          ;; •
    ((== cp 8728) (sym_circ x y em))                            ;; ∘
    ((== cp 8853) (sym_oplus x y em 0))                         ;; ⊕
    ((== cp 8855) (sym_oplus x y em 1))                         ;; ⊗
    ((== cp 8854) (sym_oplus x y em 2))                         ;; ⊖
    ((== cp 8713) (sym_not 8713 x y em fid bold))               ;; ∉
    ((== cp 8740) (sym_not 8740 x y em fid bold))               ;; ∤
    ((== cp 8224) (sym_dagger x y em fid bold 1))               ;; †
    ((== cp 8225) (sym_dagger x y em fid bold 2))               ;; ‡
    ((== cp 8465) (sym_bb 73 x y em fid))                       ;; ℑ（用双线 I 近似）
    ((== cp 8599) (sym_darrow x y em 1 -1))                     ;; ↗
    ((== cp 8600) (sym_darrow x y em 1 1))                      ;; ↘
    ((== cp 8598) (sym_darrow x y em -1 -1))                    ;; ↖
    ((== cp 8601) (sym_darrow x y em -1 1))                     ;; ↙
    ((== cp 8723) (sym_mpm x y em fid bold))                    ;; ∓
    ((== cp 172) (sym_neg x y em))                              ;; ¬
    ((== cp 8741) (sym_parallel x y em))                        ;; ∥
    ((== cp 8726) (sym_backslash x y em))                       ;; ∖
    ((== cp 9651) (sym_triangle x y em))                        ;; △
    ;; 矢量定界符（作为原子出现时：⌈⌉⌊⌋）
    ((== cp 8968) (draw_delim_atom 8968 x y em fid))            ;; ⌈
    ((== cp 8969) (draw_delim_atom 8969 x y em fid))            ;; ⌉
    ((== cp 8970) (draw_delim_atom 8970 x y em fid))            ;; ⌊
    ((== cp 8971) (draw_delim_atom 8971 x y em fid))            ;; ⌋
    (else #f))))

;; 地板/天花板当普通原子用：画小一号的矢量角括号
(define  draw_delim_atom (lambda (cp x y em fid)
  (define w (* 0.3 em))
  (define lw (lw_of em))
  (define h (* 0.62 em))
  (cond
    ((== cp 8968)
      {
        (B.l (+ x (* 0.1 em)) (- y h) (+ x (* 0.1 em)) (- y (* 0.05 em)) lw)
        (B.l (+ x (* 0.1 em)) (- y h) (+ x w) (- y h) lw)
        `(,w ,h 0.0)
      })
    ((== cp 8969)
      {
        (B.l (+ x (- w (* 0.1 em))) (- y h) (+ x (- w (* 0.1 em))) (- y (* 0.05 em)) lw)
        (B.l x (- y h) (+ x (- w (* 0.1 em))) (- y h) lw)
        `(,w ,h 0.0)
      })
    ((== cp 8970)
      {
        (B.l (+ x (* 0.1 em)) (- y h) (+ x (* 0.1 em)) y lw)
        (B.l (+ x (* 0.1 em)) y (+ x w) y lw)
        `(,w ,h 0.0)
      })
    (else
      {
        (B.l (+ x (- w (* 0.1 em))) (- y h) (+ x (- w (* 0.1 em))) y lw)
        (B.l x y (+ x (- w (* 0.1 em))) y lw)
        `(,w ,h 0.0)
      }))))
