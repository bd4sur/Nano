;; ============================================================================
;; 渐进式 SDF 光线步进（Shader 风格）—— 灵机应用程序
;;
;; 运行环境：灵机（Animac）解释器 + Nano Gfx 本地库 + Diag 本地库
;;
;; 做法：每个像素从相机发一条射线，沿射线用"符号距离场（SDF）"做球体步进
;; （sphere tracing）求交；命中后用 4 次差分估法线，做兰伯特光照 + 距离雾。
;;
;; 因为整幅图的计算量远超解释器单帧预算，这里用**渐进式渲染**：
;;   每 tick 只算若干扫描行，直接画进帧缓冲、不清屏，跨帧累积成一整幅；
;;   画完停留几拍 → 换场景、换机位、清屏重来。观感上像逐行显影的示波器，
;;   也正好演示"没有 GPU 时 shadertoy 该怎么做"：时间片协作调度把重活摊到多帧。
;;
;; 渲染分辨率 80×60（每像素画 4×4 方块铺满 320×240）：解释器里一次列表访问约
;; 0.09 µs、一次算术约 0.01 µs，全分辨率 320×240 逐像素步进要几十秒，降采样是
;; 唯一现实选择；4×4 方块反而带来明显的老显示器质感。
;;
;; 三个场景自动循环（机位每幅偏转一点）：
;;   ① 融合球：3 个球用 smooth-min 融合 + 地面
;;   ② 置换球：球面被 sin 乘积起伏，像熔岩球 + 地面
;;   ③ 无限柱阵：柱体按 mod 重复成阵列 + 地面
;;
;; 成本实测（host x86-64）：每行（80 像素 × 平均 ~20 步 + 4 次差分求法线）
;; 约 0.7 ms；每 tick 推进 10 行 ≈ 7 ms；一整幅 47 行用 5 个 tick（≈0.35 s 的
;; CPU 时间）。浏览器 wasm32 上单行约 4 ms，整幅约 2 s 显影完成。
;;
;; 书写约定（本解释器的坑，详见 app/turtle.scm 头部）：算术一律二元嵌套
;;（可用 infer/tools/e2e/host/check_multi_operand.py 静态检查——三操作数的
;;  (+ a b c) 会被错误编译并静默算错）；cond 子句只能带一个表达式且多子句
;;  编译失败；Math 库没有 min/max，自己比较；(cons 0 '()) 含哑元素，须先 pop。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Diag)

;; ---- 渲染分辨率与调度 ------------------------------------------------------

(define W (Gfx.width))            ; 320
(define H (Gfx.height))           ; 240
(define PX 4)                     ; 每像素方块边长
(define RW (/ W PX))              ; 80
(define RH (/ H PX))              ; 60
(define HUD_TOP 32)
(define HUD_BOT 20)
(define ROW_FIRST (/ HUD_TOP PX))                 ; 8：跳过顶部 HUD 带
(define ROW_LAST (- (- RH (/ HUD_BOT PX)) 1))      ; 54：留出底部说明带
(define ROWS_TOTAL (- (+ ROW_LAST 1) ROW_FIRST))  ; 47
(define ROWS_PER_TICK 8)
(define TICK_MS 40)
(define HOLD_TICKS 12)            ; 画完后停留的 tick 数
(define MAXSTEPS 20)
(define EPS 0.004)
(define TMAX 12.0)
(define FLOOR_Y -1.15)

;; ---- 相机（每幅固定一次，画完再转一点）-------------------------------------

(define cam_ang 0.6)
(define eye_x 0.0) (define eye_y 0.0) (define eye_z 0.0)
(define fwd_x 0.0) (define fwd_y 0.0) (define fwd_z 0.0)
(define rgt_x 0.0) (define rgt_y 0.0) (define rgt_z 0.0)
(define upx 0.0) (define upy 0.0) (define upz 0.0)
(define FOV 1.0)

(define setup_camera
  (lambda ()
    (define cl 0.0)
    (set! eye_x (* 3.6 (Math.sin cam_ang)))
    (set! eye_y 0.95)
    (set! eye_z (* 3.6 (Math.cos cam_ang)))
    ;; fwd = normalize(-eye)（看向原点）
    (set! cl (Math.sqrt (+ (* eye_x eye_x) (+ (* eye_y eye_y) (* eye_z eye_z)))))
    (set! fwd_x (/ (- 0.0 eye_x) cl))
    (set! fwd_y (/ (- 0.0 eye_y) cl))
    (set! fwd_z (/ (- 0.0 eye_z) cl))
    ;; right = normalize(cross(fwd, (0,1,0))) = normalize((fwd_z, 0, -fwd_x))
    (set! cl (Math.sqrt (+ (* fwd_z fwd_z) (* fwd_x fwd_x))))
    (set! rgt_x (/ fwd_z cl))
    (set! rgt_y 0.0)
    (set! rgt_z (/ (- 0.0 fwd_x) cl))
    ;; up = cross(right, fwd)
    (set! upx (- (* rgt_y fwd_z) (* rgt_z fwd_y)))
    (set! upy (- (* rgt_z fwd_x) (* rgt_x fwd_z)))
    (set! upz (- (* rgt_x fwd_y) (* rgt_y fwd_x)))))

;; ---- 场景 ------------------------------------------------------------------

(define scene 0)
(define sname "融合球")
(define sdesc "两球 smooth-min 融合 + 地面")
(define scene_count 3)
(define stime 0.0)
(define base_r 220) (define base_g 170) (define base_b 120)

(define apply_scene
  (lambda (i)
    (set! scene i)
    (if (== i 0)
        { (set! sname "融合球") (set! sdesc "两球 smooth-min 融合 + 地面")
          (set! base_r 225) (set! base_g 165) (set! base_b 115) })
    (if (== i 1)
        { (set! sname "置换球") (set! sdesc "球面 sin 乘积位移，像熔岩球 + 地面")
          (set! base_r 120) (set! base_g 210) (set! base_b 235) })
    (if (== i 2)
        { (set! sname "无限柱阵") (set! sdesc "柱体 mod 重复成阵列 + 地面")
          (set! base_r 210) (set! base_g 120) (set! base_b 200) })))

;; ---- SDF -------------------------------------------------------------------
;; 只在"差分求法线"时调用（每像素 4 次）。光线步进里为了省函数调用开销，
;; 把同一套数学内联在 march_row 里——两处必须保持一致，改动时一起改。

(define sdf
  (lambda (px py pz)
    (define sd 0.0) (define d2 0.0) (define h 0.0)
    (define ax 0.0) (define ay 0.0) (define az 0.0) (define r 0.0)
    (define qx 0.0) (define qz 0.0) (define df 0.0)
    (if (== scene 0)
        {
          ;; 两个球 + smooth-min（k=0.5）
          (set! ax (- px (* 0.62 (Math.sin stime))))
          (set! ay (- py 0.12))
          (set! az pz)
          (set! sd (- (Math.sqrt (+ (* ax ax) (+ (* ay ay) (* az az)))) 0.66))
          (set! ax (- px (* 0.55 (Math.sin (+ 2.1 stime)))))
          (set! ay py)
          (set! az (- pz (* 0.55 (Math.cos (+ 1.2 stime)))))
          (set! d2 (- (Math.sqrt (+ (* ax ax) (+ (* ay ay) (* az az)))) 0.58))
          (set! h (+ 0.5 (* 0.5 (/ (- d2 sd) 0.5))))
          (if (< h 0.0) { (set! h 0.0) })
          (if (> h 1.0) { (set! h 1.0) })
          (set! sd (- (+ d2 (* (- sd d2) h)) (* 0.5 (* h (- 1.0 h)))))
          ;; 地面
          (set! df (+ py (- 0.0 FLOOR_Y)))
          (if (< df sd) { (set! sd df) }) }
        {})
    (if (== scene 1)
        {
          (set! r (Math.sqrt (+ (* px px) (+ (* py py) (* pz pz)))))
          (set! sd (- r 1.0))
          (set! sd (- sd (* 0.22 (* (Math.sin (* 4.5 px))
                                   (* (Math.sin (* 4.5 py)) (Math.sin (* 4.5 pz)))))))
          (set! df (+ py (- 0.0 FLOOR_Y)))
          (if (< df sd) { (set! sd df) }) }
        {})
    (if (== scene 2)
        {
          (set! qx (- (mod (+ px 100.0) 2.0) 1.0))
          (set! qz (- (mod (+ pz 100.0) 2.0) 1.0))
          (set! sd (- (Math.sqrt (+ (* qx qx) (* qz qz))) 0.40))
          ;; 柱高限制（max）：py 高于 0.55 时改成横向平面距离
          (set! d2 (- py 0.55))
          (if (> d2 sd) { (set! sd d2) })
          (set! df (+ py (- 0.0 FLOOR_Y)))
          (if (< df sd) { (set! sd df) }) }
        {})
    sd))

;; ---- 一行像素的光线步进 ----------------------------------------------------

(define ds "")

(define march_row
  (lambda (ry)
    (define px 0)
    (define u 0.0) (define v 0.0)
    (define rdx 0.0) (define rdy 0.0) (define rdz 0.0) (define rl 0.0)
    (define t 0.0) (define cxp 0.0) (define cyp 0.0) (define czp 0.0)
    (define sd 0.0) (define d2 0.0) (define h 0.0)
    (define ax 0.0) (define ay 0.0) (define az 0.0) (define r 0.0)
    (define qx 0.0) (define qz 0.0) (define df 0.0)
    (define st 0) (define hit 0)
    (define nx 0.0) (define ny 0.0) (define nz 0.0) (define e 0.0016)
    (define nl 0.0) (define dif 0.0) (define fog 0.0) (define chk 0.0)
    (define rr 0) (define gg 0) (define bb 0) (define sky 0.0)
    (set! v (* (/ (- (/ (+ ry 0.5) RH) 0.5) 0.5) 0.75))
    (set! px 0)
    (while (< px RW) {
      (set! u (* (/ (- (/ (+ px 0.5) RW) 0.5) 0.5) FOV))
      ;; rd = normalize(fwd + right*u + up*v)
      (set! rdx (+ fwd_x (+ (* rgt_x u) (* upx v))))
      (set! rdy (+ fwd_y (+ (* rgt_y u) (* upy v))))
      (set! rdz (+ fwd_z (+ (* rgt_z u) (* upz v))))
      (set! rl (Math.sqrt (+ (* rdx rdx) (+ (* rdy rdy) (* rdz rdz)))))
      (set! rdx (/ rdx rl)) (set! rdy (/ rdy rl)) (set! rdz (/ rdz rl))
      (set! t 0.0) (set! hit 0) (set! st 0)
      (while (and (< st MAXSTEPS) (< t TMAX))
        {
          (set! cxp (+ eye_x (* t rdx)))
          (set! cyp (+ eye_y (* t rdy)))
          (set! czp (+ eye_z (* t rdz)))
          ;; ---- 内联 SDF（与 sdf 保持一致）----
          (if (== scene 0)
              {
                (set! ax (- cxp (* 0.62 (Math.sin stime))))
                (set! ay (- cyp 0.12))
                (set! az czp)
                (set! sd (- (Math.sqrt (+ (* ax ax) (+ (* ay ay) (* az az)))) 0.66))
                (set! ax (- cxp (* 0.55 (Math.sin (+ 2.1 stime)))))
                (set! ay cyp)
                (set! az (- czp (* 0.55 (Math.cos (+ 1.2 stime)))))
                (set! d2 (- (Math.sqrt (+ (* ax ax) (+ (* ay ay) (* az az)))) 0.58))
                (set! h (+ 0.5 (* 0.5 (/ (- d2 sd) 0.5))))
                (if (< h 0.0) { (set! h 0.0) })
                (if (> h 1.0) { (set! h 1.0) })
                (set! sd (- (+ d2 (* (- sd d2) h)) (* 0.5 (* h (- 1.0 h)))))
                (set! df (+ cyp (- 0.0 FLOOR_Y)))
                (if (< df sd) { (set! sd df) }) }
              {})
          (if (== scene 1)
              {
                (set! r (Math.sqrt (+ (* cxp cxp) (+ (* cyp cyp) (* czp czp)))))
                (set! sd (- r 1.0))
                (set! sd (- sd (* 0.22 (* (Math.sin (* 4.5 cxp))
                                         (* (Math.sin (* 4.5 cyp)) (Math.sin (* 4.5 czp)))))))
                (set! df (+ cyp (- 0.0 FLOOR_Y)))
                (if (< df sd) { (set! sd df) }) }
              {})
          (if (== scene 2)
              {
                (set! qx (- (mod (+ cxp 100.0) 2.0) 1.0))
                (set! qz (- (mod (+ czp 100.0) 2.0) 1.0))
                (set! sd (- (Math.sqrt (+ (* qx qx) (* qz qz))) 0.40))
                (set! d2 (- cyp 0.55))
                (if (> d2 sd) { (set! sd d2) })
                (set! df (+ cyp (- 0.0 FLOOR_Y)))
                (if (< df sd) { (set! sd df) }) }
              {})
          (if (< sd EPS) { (set! hit 1) (set! st MAXSTEPS) }
              { (set! t (+ t sd)) (set! st (+ st 1)) })
        })
      (if (== hit 1)
          {
            ;; ---- 着色：4 次差分法线 + 兰伯特 + 棋盘地面 + 雾 ----
            (set! nx (+ (sdf (+ cxp e) cyp czp) (sdf cxp (+ cyp e) czp)))
            (set! nx (- (+ nx (sdf cxp cyp (+ czp e))) (sdf cxp cyp czp)))
            (set! ny (+ (sdf (+ cxp e) cyp czp) (sdf cxp cyp (+ czp e))))
            (set! ny (- (+ ny (sdf cxp (+ cyp e) czp)) (sdf cxp cyp czp)))
            (set! nz (+ (sdf (+ cxp e) cyp czp) (sdf cxp (+ cyp e) czp)))
            (set! nz (- (+ nz (sdf cxp cyp (+ czp e))) (sdf cxp cyp czp)))
            (set! nl (Math.sqrt (+ (* nx nx) (+ (* ny ny) (* nz nz)))))
            (if (< nl 1e-6) { (set! nl 1.0) })
            (set! nx (/ nx nl)) (set! ny (/ ny nl)) (set! nz (/ nz nl))
            (set! dif (+ (* nx 0.55) (+ (* ny 0.75) (* nz 0.35))))
            (if (< dif 0.0) { (set! dif 0.0) })
            (set! dif (+ 0.30 (* 0.70 dif)))
            (set! chk 1.0)
            (if (< ny 0.6)
                { (if (< (mod (+ (Math.floor (+ cxp 0.5)) (Math.floor (+ czp 0.5))) 2.0) 1.0)
                      { (set! chk 0.55) }
                      { (set! chk 1.0) }) }
                { (set! chk 1.0) })
            (set! fog (/ 1.0 (+ 1.0 (* 0.05 (* t t)))))
            (set! rr (Math.floor (* (* base_r dif) (* chk fog))))
            (set! gg (Math.floor (* (* base_g dif) (* chk fog))))
            (set! bb (Math.floor (* (* base_b dif) (* chk fog))))
            (if (> rr 255) { (set! rr 255) })
            (if (> gg 255) { (set! gg 255) })
            (if (> bb 255) { (set! bb 255) })
            ;; 重复空间里 SDF 不连续，差分法线偶尔退化 —— 兜一个下限，别出现纯黑麻点
            (if (< rr 10) { (set! rr 10) })
            (if (< gg 10) { (set! gg 10) })
            (if (< bb 10) { (set! bb 10) }) }
          {
            ;; ---- 未命中：天光渐变 ----
            (set! sky (+ 0.5 (* 0.5 rdy)))
            (if (< sky 0.0) { (set! sky 0.0) })
            (set! rr (Math.floor (+ 16.0 (* 90.0 sky))))
            (set! gg (Math.floor (+ 24.0 (* 110.0 sky))))
            (set! bb (Math.floor (+ 46.0 (* 150.0 sky)))) })
      (Gfx.fill_rect (* px PX) (* ry PX) PX PX rr gg bb)
      (set! px (+ px 1)) })))

;; ---- HUD -------------------------------------------------------------------

(define rows_done 0)
(define row_ms 0)
(define frame_no 0)
(define hold 0)
(define state 0)                   ; 0=正在显影 1=画完停留

(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))
(define cat4 (lambda (a b c d) (String.concat a (cat3 b c d))))

(define cat5 (lambda (a b c d e) (String.concat a (String.concat b (String.concat c (String.concat d e))))))
(define cat6 (lambda (a b c d e f) (String.concat a (cat5 b c d e f))))

(define draw_hud
  (lambda ()
    (define pct 0)
    (Gfx.fill_rect 0 0 W HUD_TOP 0 0 0)
    (Gfx.fill_rect 0 (- H HUD_BOT) W HUD_BOT 0 0 0)
    (Gfx.draw_text (cat3 "SDF 步进 · " sname "") 8 4 225 235 245)
    (set! pct (Math.floor (/ (* 100.0 (+ rows_done 0.0)) (+ ROWS_TOTAL 0.0))))
    (if (> pct 100) { (set! pct 100) })
    (set! ds (cat6 (String.atom_to_string pct) "%  行 " (String.atom_to_string rows_done)
                   "/" (String.atom_to_string ROWS_TOTAL)
                   (cat3 "   " (String.atom_to_string row_ms) "ms/行")))
    (Gfx.draw_text ds 8 18 130 165 195)
    (Gfx.draw_text sdesc 8 (- H 15) 110 140 170)))

;; ---- 主流程 ----------------------------------------------------------------

(define start_image
  (lambda ()
    (Gfx.fill_rect 0 HUD_TOP W (- (- H HUD_BOT) HUD_TOP) 0 0 0)
    (setup_camera)
    (set! rows_done 0)
    (set! state 0)
    (set! hold 0)))

(define tick_fn
  (lambda ()
    (define t0 0) (define k 0)
    (set! t0 (Diag.ms))
    (if (== state 0)
        {
          (set! k 0)
          (while (and (< k ROWS_PER_TICK) (< rows_done ROWS_TOTAL))
            { (march_row (+ ROW_FIRST rows_done))
              (set! rows_done (+ rows_done 1))
              (set! k (+ k 1)) })
          (set! row_ms (Math.floor (/ (- (Diag.ms) t0) (+ k 0.0))))
          (if (>= rows_done ROWS_TOTAL) { (set! state 1) (set! hold 0) }) }
        {
          (set! hold (+ hold 1))
          (if (>= hold HOLD_TICKS)
              { (apply_scene (mod (+ scene 1) scene_count))
                (set! stime (+ stime 1.7))
                (set! cam_ang (+ cam_ang 0.55))
                (set! frame_no (+ frame_no 1))
                (start_image) }
              {}) })
    (draw_hud)
    (Gfx.refresh)))

;; ---- 启动 ------------------------------------------------------------------

(apply_scene 0)
(start_image)
(set! _timer (System.set_interval TICK_MS tick_fn))