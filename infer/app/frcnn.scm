;@id=frcnn
;@title=两阶段检测
;@version=2026-11-26
;; ============================================================================
;; frcnn.scm —— Faster R-CNN 两阶段目标检测的**可视化教学演示**（Nano「灵机应用程序」）
;;
;; 交互（三段）：
;;   ① 绘制：启动就是一张空白 64x64 画板（放大 3 倍显示），手指直接写 1~6 个数字
;;      （允许重叠）。写完点「开始识别」。
;;   ② 流水线：把两阶段检测拆成 11 步，自动跑一遍（进度条可见），跑完停在「最终检测」。
;;      之后可以用 « < » « > » 单步回看（中间结果全部保留、不重算），点左图的框或右栏
;;      的列表项都能在左下角看到它的类别/置信度/坐标。
;;   ③ 卷积：单独一页讲 CNN 每层怎么算出来的——3×3 窗口在输入上滑动、9 个数加权求和、
;;      8/24 个通道的数据如何汇合到下一层的一格。窗口会自己慢慢滑，也能点格子跳过去。
;;
;; 网络（权重由 tools/train_frcnn_digits.py 离线训练导出，见 app/lib/frcnn_weights.scm）：
;;   【第一阶段】conv1(3x3s2, 8ch) → conv2(3x3s2, 24ch) → conv3(3x3 逐通道 + 1x1, 24ch)
;;              → RPN(1x1 头 + 9 锚框/格) → 提议解码 + NMS
;;   【第二阶段】RoI 池化 4x4 → 分类头(11 类) + 回归头(每类 4 偏移) → 逐类 NMS
;;
;; 解释器约束（manual.md / doc/AGENTS.md，逐条严格遵守）：
;;   · 算术/比较恒二参数；没有一元取负（写 (- 0 x)）；没有 let/list/append/float；
;;   · define 只在 lambda 体顶层（块内只用 set!），且局部 define 的初始化器不引用
;;     同体的其它 define（会崩）；
;;   · `(...)` 是过程调用，`{...}` 才是 begin；
;;   · if 只有 2~3 支——**不能**写 (if c r1 g1 b1 r2 g2 b2) 来选颜色（多出的值留在
;;     操作数栈上，后面某个 call 才炸）；颜色一律先算进 cr/cg/cb；
;;   · / 与 mod 恒返回浮点 → 下标一律 Math.floor/round 收口；mod 符号跟随被除数；
;;   · get_item 越界返回 #undefined，喂给 String.concat 会在 native 里炸；
;;   · 12px 字模里只有 ASCII 与 GB2312 汉字，**箭头只能用 < >**（◀ ▶ ‹ › 会画成 ?）；
;;   · Gfx 坐标是 uint32，负数先钳再画。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Diag)

(import EX "app/lib/exit.scm")
(import BTN "app/lib/button.scm")   ; 通用按钮控件（圆角底/描边/居中标签），与 waterpool_raster 同一套
(import FR "app/lib/frcnn_core.scm")

;; ---- 版面 ---------------------------------------------------------------

(define SW (Gfx.width))

(define RX 136)            ; 结果页右栏左边界
(define IMGX 4)            ; 结果页左图（64x64 按 2px）
(define IMGY 36)

;; 绘制页
(define CVX 4)
(define CVY 26)
(define CVS 3)             ; 64*3 = 192
(define CVW 192)
(define DCOL 200)          ; 绘制页右栏左边界

(define TICK_MS 20)
(define BUDGET '(4096 8000 6000 12000 6000 2400 9000 2400 400 2400 20))

;; ---- 小工具 -------------------------------------------------------------

(define i2s (lambda (x) (String.atom_to_string x)))
(define cat_t 0)
(define cat (lambda (a b) {
  (set! cat_t (String.concat a b))
  cat_t }))
(define maxf (lambda (a b) (if (> a b) a b)))
(define minf (lambda (a b) (if (< a b) a b)))
(define maxi (lambda (a b) (if (> a b) a b)))
(define mini (lambda (a b) (if (< a b) a b)))
(define clx (lambda (v) (maxi 0 (mini 316 (Math.round v)))))
(define cly (lambda (v) (maxi 0 (mini 236 (Math.round v)))))

(define f_v 0)
(define f_neg 0)
(define f_a 0)
(define pad2
  (lambda (v)
    (if (< (Math.round v) 10.0)
        (cat "0" (i2s (Math.round v)))
        (i2s (Math.round v)))))
(define fmt2
  (lambda (x) {
    (set! f_v (Math.round (* x 100.0)))
    (set! f_neg (if (< f_v 0) 1 0))
    (set! f_a (if (== f_neg 1) (- 0 f_v) f_v))
    (cat (if (== f_neg 1) "-" "")
      (cat (i2s (Math.floor (/ f_a 100.0)))
        (cat "." (pad2 (- f_a (* 100.0 (Math.floor (/ f_a 100.0))))))))
  }))
(define fmt1
  (lambda (x) {
    (set! f_v (Math.round (* x 10.0)))
    (set! f_neg (if (< f_v 0) 1 0))
    (set! f_a (if (== f_neg 1) (- 0 f_v) f_v))
    (cat (if (== f_neg 1) "-" "")
      (cat (i2s (Math.floor (/ f_a 10.0)))
        (cat "." (i2s (mod f_a 10.0)))))
  }))
(define fmt_pct
  (lambda (p) {
    (set! f_v (Math.round (* p 1000.0)))
    (cat (i2s (Math.floor (/ f_v 10.0)))
      (cat "." (cat (i2s (mod f_v 10.0)) "%")))
  }))

;; "r,c"
(define gstr (lambda (r c) (cat (i2s r) (cat "," (i2s c)))))

;; 平铺框表第 i 个框 → "x1,y1-x2,y2"（整数，右栏一行放得下）
(define box_str0
  (lambda (lst i)
    (cat (i2s (Math.round (get_item lst (* i 4))))
      (cat "," (cat (i2s (Math.round (get_item lst (+ (* i 4) 1))))
        (cat "-" (cat (i2s (Math.round (get_item lst (+ (* i 4) 2))))
          (cat "," (i2s (Math.round (get_item lst (+ (* i 4) 3))))))))))))

;; 平铺框表第 i 个框 → "x1,y1-x2,y2"（一位小数，左下角信息面板用）
(define box_str
  (lambda (lst i)
    (cat (fmt1 (get_item lst (* i 4)))
      (cat "," (cat (fmt1 (get_item lst (+ (* i 4) 1)))
        (cat "-" (cat (fmt1 (get_item lst (+ (* i 4) 2)))
          (cat "," (fmt1 (get_item lst (+ (* i 4) 3)))))))))))

;; ---- 颜色 ---------------------------------------------------------------

(define CT_R 40)
(define CT_G 44)
(define CT_B 62)
(define CD_R 132)
(define CD_G 140)
(define CD_B 156)
(define CA_R 25)
(define CA_G 95)
(define CA_B 200)
(define CL_R 205)
(define CL_G 210)
(define CL_B 220)
(define CG_R 24)
(define CG_G 150)
(define CG_B 96)

;; ---- 基元 ---------------------------------------------------------------

(define ac_t 0.0)
(define ac_u 0.0)
(define draw_act
  (lambda (x y s v) {
    (set! ac_t (minf 1.0 (maxf 0.0 v)))
    (if (< ac_t 0.5)
        (Gfx.fill_rect x y s s
                       (Math.round (- 252.0 (* 3.0 ac_t)))
                       (Math.round (- 250.0 (* 125.0 ac_t)))
                       (Math.round (- 247.0 (* 382.0 ac_t))))
        {
          (set! ac_u (- (* 2.0 ac_t) 1.0))
          (Gfx.fill_rect x y s s
                         (Math.round (- 255.0 (* 90.0 ac_u)))
                         (Math.round (- 190.0 (* 150.0 ac_u)))
                         (Math.round (- 60.0 (* 20.0 ac_u))))
        })
  }))

(define wc_u 0.0)
(define draw_wt
  (lambda (x y s w) {
    (if (< w 0.0)
        {
          (set! wc_u (minf 1.0 (- 0.0 w)))
          (Gfx.fill_rect x y s s
                         (Math.round (- 255.0 (* 195.0 wc_u)))
                         (Math.round (- 255.0 (* 172.0 wc_u)))
                         255)
        }
        {
          (set! wc_u (minf 1.0 w))
          (Gfx.fill_rect x y s s 255
                         (Math.round (- 255.0 (* 214.0 wc_u)))
                         (Math.round (- 255.0 (* 206.0 wc_u))))
        })
  }))

;; 灰度：v 是亮度（1 = 白纸，0 = 黑墨），与 frcnn_core 里 img 的约定一致
(define gc_l 0)
(define draw_ink
  (lambda (x y s v) {
    (set! gc_l (Math.round (* 255.0 (minf 1.0 (maxf 0.0 v)))))
    (Gfx.fill_rect x y s s gc_l gc_l gc_l)
  }))

;; 归一化值（xn = (v/255 - 0.80)/0.21）→ 灰度
(define draw_norm
  (lambda (x y s xn) {
    (draw_ink x y s (+ (* 0.21 xn) 0.80))
  }))

;; ---- 按钮：统一用 app/lib/button.scm 的控件 ----------------------------------
;; 与 waterpool_raster.scm / exit.scm 用的是同一枚控件（BTN 描述表：圆角底板 +
;; 1px 描边 + **居中**标签 + 按下高亮），不再自己拼 fill_rect/draw_rect/draw_text。
;;   · 矩形在启动时一次性设好：各模式**复用同一组槽位**，位置恒定 → 每拍只改标签与
;;     激活态（(BTN.set_label) 只写字段，不分配），命中与绘制永远用同一份矩形；
;;   · 标签由 (Gfx.draw_text_centered) 画在按钮**正中**（水平、垂直都居中）；
;;   · 命中判定用 (BTN.hit)（纯几何），边沿仍由本 App 自己的 _p 判定（画板笔划也要用）。

(define NB 15)
(define BL 0)            ; 按钮池（NB 枚 BTN 描述表）
(define B_ON 0)          ; 每枚按钮的激活标记

(define S_H0 0)          ; 头部：清空 / 重画 / 层1
(define S_H1 1)          ; 头部：示例 / 层2
(define S_H2 2)          ; 头部：卷积 / 检测
(define S_PREV 3)        ; 结果页 <
(define S_NEXT 4)        ; 结果页 >
(define S_AUTO 5)        ; 结果页 自动/暂停
(define S_CHL 6)         ; 右栏 通道 <
(define S_CHR 7)         ; 右栏 通道 >
(define S_FINE 8)        ; 画板页 细
(define S_COAR 9)        ; 画板页 粗
(define S_SAMP 10)       ; 画板页 换个示例
(define S_RUN 11)        ; 画板页 开始识别
(define S_CVL 12)        ; 卷积页 通道 <
(define S_CVR 13)        ; 卷积页 通道 >
(define S_CVA 14)        ; 卷积页 停窗/滑窗

(define bslot (lambda (i) (get_item BL i)))
(define bhit (lambda (i px py) (BTN.hit (bslot i) px py)))
(define bset
  (lambda (i x y w h) {
    (BTN.set_pos (bslot i) x y)
    (BTN.set_size (bslot i) w h)
  }))
;; 每拍：标签 + 激活态（激活 = 蓝底白字）
(define bput
  (lambda (i label on) {
    (BTN.set_label (bslot i) label)
    (set_item! B_ON i on)
    (if (== on 1)
        (BTN.set_text_color (bslot i) 255 255 255)
        (BTN.set_text_color (bslot i) 38 46 68))
  }))
(define bdraw
  (lambda (i) (BTN.draw_state (bslot i) (== (get_item B_ON i) 1))))

(define b_i 0)
(define btn_init
  (lambda () {
    (set! BL (cdr '(0)))
    (set! B_ON (cdr '(0)))
    (set! b_i 0)
    (while (< b_i NB) {
      (set! BL (cons (BTN.new) BL))
      (set! B_ON (cons 0 B_ON))
      (set! b_i (+ b_i 1))
    })
    ;; 统一外观：浅底深字，激活蓝底白字（本 App 是浅色界面）
    (set! b_i 0)
    (while (< b_i NB) {
      (BTN.set_font (get_item BL b_i) 1)
      (BTN.set_bg (get_item BL b_i) 233 237 246)
      (BTN.set_active_bg (get_item BL b_i) 52 116 204)
      (BTN.set_text_color (get_item BL b_i) 38 46 68)
      (BTN.set_border (get_item BL b_i) 170 178 196)
      (BTN.set_radius (get_item BL b_i) 4)
      (set! b_i (+ b_i 1))
    })
    ;; ---- 矩形：一次设好，各模式共用（头部三枚在三种模式下位置相同）----
    (bset S_H0 96 2 46 16)
    (bset S_H1 146 2 46 16)
    (bset S_H2 196 2 46 16)
    ;; 结果页第二行（步骤条）
    (bset S_PREV 4 19 20 15)
    (bset S_NEXT 26 19 20 15)
    (bset S_AUTO 48 19 44 15)
    ;; 右栏通道选择（结果页第 1~4 步与第 8 步共用同一处）
    (bset S_CHL 136 50 18 14)
    (bset S_CHR 222 50 18 14)
    ;; 画板页右栏
    (bset S_SAMP 200 28 112 22)
    (bset S_FINE 200 56 52 22)
    (bset S_COAR 260 56 52 22)
    (bset S_RUN 200 104 112 30)
    ;; 卷积页第二行
    (bset S_CVL 78 19 18 15)
    (bset S_CVR 164 19 18 15)
    (bset S_CVA 186 19 46 15)
    #t
  }))

;; 每拍同步标签/激活态（切换模式后立刻生效，不等某次重绘）
(define ui_sync
  (lambda () {
    (if (== mode 0)
        {
          (bput S_H0 "清空" 0)
          (bput S_H1 "示例" 0)
          (bput S_H2 "卷积" 0)
          (bput S_FINE "细" 0)
          (bput S_COAR "粗" 0)
          (bput S_SAMP "换个示例" 0)
          (bput S_RUN "开始识别" (if (and (== drawn 1) (== busy 0)) 1 0))
        }
        (if (== mode 1)
            {
              (bput S_H0 "重画" 0)
              (bput S_H1 "示例" 0)
              (bput S_H2 "卷积" 0)
              (bput S_PREV "<" 0)
              (bput S_NEXT ">" 0)
              (bput S_AUTO (if (== autorun 1) "暂停" "自动") (if (== autorun 1) 1 0))
              (bput S_CHL "<" 0)
              (bput S_CHR ">" 0)
            }
            {
              (bput S_H0 "层1" (if (== cv_layer 0) 1 0))
              (bput S_H1 "层2" (if (== cv_layer 1) 1 0))
              (bput S_H2 "检测" 0)
              (bput S_CVL "<" 0)
              (bput S_CVR ">" 0)
              (bput S_CVA (if (== cv_anim 1) "停窗" "滑窗") (if (== cv_anim 1) 1 0))
            }))
  }))

;; 命中测试必须返回布尔：本解释器里 **0 也是真值**，用 0/1 当真假会让每个按钮
;; 在任何位置都被"命中"（踩过：点画板同时触发了「开始识别」）。
(define in_rect
  (lambda (px py x y w h)
    (and (>= px x) (< px (+ x w)) (>= py y) (< py (+ y h)))))

;; 结果页图像坐标 → 屏幕（2px/像素）
(define sx_of (lambda (v) (+ IMGX (* v 2))))
(define sy_of (lambda (v) (+ IMGY (* v 2))))
(define draw_box
  (lambda (x1 y1 x2 y2 r g b) {
    (Gfx.draw_rect (clx (sx_of x1)) (cly (sy_of y1))
                   (maxi 2 (Math.round (* (- x2 x1) 2))) (maxi 2 (Math.round (* (- y2 y1) 2)))
                   r g b)
  }))
(define draw_box2
  (lambda (x1 y1 x2 y2 r g b) {
    (Gfx.draw_rect (clx (sx_of x1)) (cly (sy_of y1))
                   (maxi 3 (Math.round (* (- x2 x1) 2))) (maxi 3 (Math.round (* (- y2 y1) 2)))
                   r g b)
    (Gfx.draw_rect (clx (+ (sx_of x1) 1)) (cly (+ (sy_of y1) 1))
                   (maxi 1 (- (Math.round (* (- x2 x1) 2)) 2))
                   (maxi 1 (- (Math.round (* (- y2 y1) 2)) 2)) r g b)
  }))

;; ---- 状态 ---------------------------------------------------------------

(define mode 0)          ; 0 = 绘制（空白画板）/ 1 = 流水线结果 / 2 = 卷积细节
(define stage 0)         ; 已完成的最后一步（0..10）
(define todo 0)
(define busy 0)
(define prog 0.0)
(define autorun 1)
(define auto_wait 0)
(define infer_ms 0)
(define stage_t0 0)
(define drawn 0)         ; 1 = 画板上有墨（决定「开始识别」是否可用）

(define sel_ch 0)
(define sel_cell 0)
(define sel_on 0)
(define sel_roi 0)
(define sel_fin 0)
(define sel_bin 0)
(define pick 0)          ; 左下角信息：0 概览 / 2 RoI / 3 检测

(define pen_down 0)
(define pen_last_x 0.0)
(define pen_last_y 0.0)
(define brush_r 1.8)

(define cv_layer 0)
(define cv_cell 0)
(define cv_ch 0)
(define cv_anim 1)
(define cv_t 0)
(define CV_MS 130)

(define _p 0)
(define _pp 0)           ; 上一拍的电平：自己算「按下沿」，不依赖 Touch.take_press
(define _t 0)
(define armed 0)         ; 0 = 还没确认「手指已离开屏幕」；拉起本 App 的那一下不能算输入
(define _tx 0)
(define _ty 0)
(define _i 0)

;; ---- 顶部栏 -------------------------------------------------------------


(define stage_names
  '("输入" "conv1" "conv2" "conv3" "RPN 头" "RPN 分类" "RPN 回归"
    "提议+NMS" "RoI 池化" "检测头" "最终检测"))

(define draw_header
  (lambda () {
    (Gfx.fill_rect 0 0 SW 18 248 249 252)
    ;; 标题收窄到 88px 以内，右边三枚按钮从 96 起 —— 不再被盖住
    (if (== mode 0)
        (Gfx.draw_text "两阶段检测" 4 3 CT_R CT_G CT_B)
        (if (== mode 1)
            (Gfx.draw_text "Faster R-CNN" 4 3 CT_R CT_G CT_B)
            (Gfx.draw_text "CNN 逐层计算" 4 3 CT_R CT_G CT_B)))
    (bdraw S_H0)
    (bdraw S_H1)
    (bdraw S_H2)
    (Gfx.fill_rect 0 17 SW 1 CL_R CL_G CL_B)
  }))

;; ---- 步骤条（结果页/卷积页共用第二行） ---------------------------------

(define busy_names '("计算中" "计算中." "计算中.." "计算中..."))
(define bd_i 0)
(define draw_stagebar
  (lambda () {
    (Gfx.fill_rect 0 18 SW 16 252 253 255)
    (bdraw S_PREV)
    (bdraw S_NEXT)
    (bdraw S_AUTO)
    ;; 进度：10 步的粗粒度进度（"3/10 conv3"），细进度条已去掉（与「计算中」重叠）
    (Gfx.draw_text (cat (i2s stage) (cat "/10 " (get_item stage_names stage)))
                   96 21 CT_R CT_G CT_B)
    ;; 计算中提示：居中在右侧空白处，末尾的点做跳动（只靠文字表示"还在算"）
    (if (== busy 1)
        {
          (set! bd_i (Math.floor (mod (Math.floor (/ (System.timestamp) 300)) 4)))
          (Gfx.draw_text_centered 1 (get_item busy_names bd_i) 289 26 CD_R CD_G CD_B 1)
        }
        #f)
    (Gfx.fill_rect 0 34 SW 1 CL_R CL_G CL_B)
  }))

(define phase_of '(0 1 1 1 2 2 2 2 3 4 4))
(define phase_names '("输入" "骨干" "RPN" "RoI池化" "检测头"))

(define ph_r 0)
(define ph_g 0)
(define ph_b 0)
(define draw_phases
  (lambda () {
    (define cur (get_item phase_of stage))
    (Gfx.fill_rect 2 218 SW 1 CL_R CL_G CL_B)
    (set! _i 0)
    (while (< _i 5) {
      (set! ph_r (if (== _i cur) 214 240))
      (set! ph_g (if (== _i cur) 232 242))
      (set! ph_b (if (== _i cur) 250 246))
      (Gfx.fill_rect (+ 3 (* _i 63)) 220 60 18 ph_r ph_g ph_b)
      (if (== _i cur)
          (Gfx.fill_rect (+ 3 (* _i 63)) 220 60 2 70 150 230)
          #f)
      (Gfx.draw_text_centered 1 (get_item phase_names _i) (+ 33 (* _i 63)) 229
                              (if (== _i cur) CA_R 120) (if (== _i cur) CA_G 128)
                              (if (== _i cur) CA_B 145) 1)
      (set! _i (+ _i 1))
    })
  }))

;; ---- 图像与特征图 -------------------------------------------------------

(define ri 0)
(define rj 0)
(define draw_image_at
  (lambda (x y s) {
    (set! ri 0)
    (while (< ri 64) {
      (set! rj 0)
      (while (< rj 64) {
        (draw_ink (+ x (* rj s)) (+ y (* ri s)) s (get_item FR.img (+ (* ri 64) rj)))
        (set! rj (+ rj 1))
      })
      (set! ri (+ ri 1))
    })
  }))

(define fm_i 0)
(define fm_m 1.0)
(define sg_v 0)
(define dbg_bad 0)
;; 取特征图元素（越界返回 0 并报一次，避免把 #undefined 丢进比较运算里直接把进程打死）
(define fget
  (lambda (buf i) {
    (define n (length buf))
    (if (and (>= i 0) (< i n))
        (get_item buf i)
        {
          (if (== dbg_bad 0)
              {
                (display "[frcnn] fget out of range i=") (display i)
                (display " len=") (display n) (newline)
                (set! dbg_bad 1)
              }
              #f)
          0.0
        })
  }))
(define chmax
  (lambda (buf n ch) {
    (set! fm_i 0)
    (set! fm_m 0.0)
    (while (< fm_i (* n n)) {
      (set! sg_v (fget buf (+ (* ch (* n n)) fm_i)))
      (if (> sg_v fm_m) (set! fm_m sg_v) #f)
      (set! fm_i (+ fm_i 1))
    })
    (if (< fm_m 0.05) 0.05 fm_m)
  }))

(define fm_r 0)
(define fm_c 0)
(define fm_b 0)
(define draw_fmap
  (lambda (buf n ch x y s mark) {
    (set! fm_m (chmax buf n ch))
    (set! fm_b (* ch (* n n)))
    (set! fm_r 0)
    (while (< fm_r n) {
      (set! fm_c 0)
      (while (< fm_c n) {
        (draw_act (+ x (* fm_c s)) (+ y (* fm_r s)) s
                  (/ (fget buf (+ fm_b (+ (* fm_r n) fm_c))) fm_m))
        (set! fm_c (+ fm_c 1))
      })
      (set! fm_r (+ fm_r 1))
    })
    (if (>= s 6)
        {
          (set! fm_r 0)
          (while (<= fm_r n) {
            (Gfx.fill_rect (+ x (* fm_r s)) y 1 (* n s) 236 239 245)
            (Gfx.fill_rect x (+ y (* fm_r s)) (* n s) 1 236 239 245)
            (set! fm_r (+ fm_r 1))
          })
        }
        #f)
    (Gfx.draw_rect (- x 1) (- y 1) (+ (* n s) 2) (+ (* n s) 2) 200 206 218)
    (if (>= mark 0)
        (Gfx.draw_rect (- (+ x (* (mod mark n) s)) 2)
                       (- (+ y (* (Math.floor (/ mark n)) s)) 2)
                       (+ s 4) (+ s 4) 20 90 200)
        #f)
  }))

;; 某个缓冲区里「最有内容」的通道（用于卷积页默认选中，避免一片白）
(define bc_i 0)
(define bc_b 0)
(define bc_m 0.0)
(define bc_v 0.0)
(define best_ch
  (lambda (buf n nch) {
    (set! bc_b 0)
    (set! bc_m -1.0)
    (set! bc_i 0)
    (while (< bc_i nch) {
      (set! bc_v (chmax buf n bc_i))
      (if (> bc_v bc_m)
          {
            (set! bc_m bc_v)
            (set! bc_b bc_i)
          }
          #f)
      (set! bc_i (+ bc_i 1))
    })
    bc_b
  }))

;; RoI 的 argmax 类别（10 = 背景）
(define ra_i 0)
(define ra_b 0)
(define ra_v 0.0)
(define roi_argmax
  (lambda (r) {
    (set! ra_b 0)
    (set! ra_v (get_item FR.clsp (* r 11)))
    (set! ra_i 1)
    (while (< ra_i 11) {
      (if (> (get_item FR.clsp (+ (* r 11) ra_i)) ra_v)
          {
            (set! ra_v (get_item FR.clsp (+ (* r 11) ra_i)))
            (set! ra_b ra_i)
          }
          #f)
      (set! ra_i (+ ra_i 1))
    })
    ra_b
  }))

;; ---- 绘制页 -------------------------------------------------------------

(define draw_canvas
  (lambda () {
    (draw_image_at CVX CVY CVS)
    (Gfx.draw_rect (- CVX 1) (- CVY 1) (+ CVW 2) (+ CVW 2) 150 158 172)
    ;; 8x8 参考格
    (set! _i 1)
    (while (< _i 8) {
      (Gfx.fill_rect (+ CVX (* _i 24)) CVY 1 CVW 226 230 236)
      (Gfx.fill_rect CVX (+ CVY (* _i 24)) CVW 1 226 230 236)
      (set! _i (+ _i 1))
    })
  }))

(define render_draw
  (lambda () {
    (Gfx.fill_white)
    (draw_header)
    (draw_canvas)
    (bdraw S_SAMP)
    (bdraw S_FINE)
    (bdraw S_COAR)
    (Gfx.draw_text_centered 1 (cat "笔粗 " (fmt1 brush_r)) 256 86 CD_R CD_G CD_B 1)
    (bdraw S_RUN)
    (Gfx.draw_text_centered 1 "写 1~6 个数字" 256 148 CT_R CT_G CT_B 1)
    (Gfx.draw_text_centered 1 "每个约 2~3 格高" 256 162 CT_R CT_G CT_B 1)
    (Gfx.draw_text_centered 1 "（格=8 像素）" 256 176 CD_R CD_G CD_B 1)
    ;; 底部提示（水平居中）
    (Gfx.fill_rect 0 222 SW 1 CL_R CL_G CL_B)
    (Gfx.draw_text_centered 1
      (if (== drawn 1) "点「开始识别」跑整套检测" "用一根手指在左边画板上写数字")
      160 232 CD_R CD_G CD_B 1)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; ---- 结果页：左图叠加 ---------------------------------------------------

(define ov_i 0)
(define draw_rois
  (lambda () {
    (set! ov_i 0)
    (while (< ov_i FR.roi_n) {
      (if (== ov_i sel_roi)
          (draw_box2 (get_item FR.roi_b (* ov_i 4)) (get_item FR.roi_b (+ (* ov_i 4) 1))
                     (get_item FR.roi_b (+ (* ov_i 4) 2)) (get_item FR.roi_b (+ (* ov_i 4) 3))
                     220 60 40)
          (draw_box (get_item FR.roi_b (* ov_i 4)) (get_item FR.roi_b (+ (* ov_i 4) 1))
                    (get_item FR.roi_b (+ (* ov_i 4) 2)) (get_item FR.roi_b (+ (* ov_i 4) 3))
                    70 130 210))
      (set! ov_i (+ ov_i 1))
    })
  }))

(define draw_fins
  (lambda () {
    (set! ov_i 0)
    (while (< ov_i FR.fin_n) {
      (if (== ov_i sel_fin)
          (draw_box2 (get_item FR.fin_b (* ov_i 4)) (get_item FR.fin_b (+ (* ov_i 4) 1))
                     (get_item FR.fin_b (+ (* ov_i 4) 2)) (get_item FR.fin_b (+ (* ov_i 4) 3))
                     200 40 30)
          (draw_box2 (get_item FR.fin_b (* ov_i 4)) (get_item FR.fin_b (+ (* ov_i 4) 1))
                     (get_item FR.fin_b (+ (* ov_i 4) 2)) (get_item FR.fin_b (+ (* ov_i 4) 3))
                     CG_R CG_G CG_B))
      (Gfx.draw_text (i2s (get_item FR.fin_cls ov_i))
                     (clx (+ (sx_of (get_item FR.fin_b (* ov_i 4))) 1))
                     (cly (- (sy_of (get_item FR.fin_b (+ (* ov_i 4) 1))) 12))
                     20 120 80)
      (set! ov_i (+ ov_i 1))
    })
  }))

(define draw_props
  (lambda () {
    (set! ov_i 0)
    (while (and (< ov_i FR.tk_n) (< ov_i 24)) {
      (draw_box (get_item FR.tk_b (* ov_i 4)) (get_item FR.tk_b (+ (* ov_i 4) 1))
                (get_item FR.tk_b (+ (* ov_i 4) 2)) (get_item FR.tk_b (+ (* ov_i 4) 3))
                (+ 150 (Math.round (* 90.0 (/ ov_i 24.0)))) 150 225)
      (set! ov_i (+ ov_i 1))
    })
  }))

(define an_k 0)
(define an_a 0)
(define an_sc 0.0)
(define draw_anchors_cell
  (lambda (gy gx) {
    (set! an_k 0)
    (while (< an_k 9) {
      (set! an_a (+ (* (+ (* gy 16) gx) 9) an_k))
      (set! an_sc (get_item FR.objp (+ (* an_k 256) (+ (* gy 16) gx))))
      (if (> an_sc 0.5)
          (draw_box (get_item FR.anch (* an_a 4)) (get_item FR.anch (+ (* an_a 4) 1))
                    (get_item FR.anch (+ (* an_a 4) 2)) (get_item FR.anch (+ (* an_a 4) 3))
                    214 128 50)
          (draw_box (get_item FR.anch (* an_a 4)) (get_item FR.anch (+ (* an_a 4) 1))
                    (get_item FR.anch (+ (* an_a 4) 2)) (get_item FR.anch (+ (* an_a 4) 3))
                    168 190 222))
      (set! an_k (+ an_k 1))
    })
  }))

;; ---- 结果页：右栏各步 ---------------------------------------------------

(define rtitle (lambda (s y) (Gfx.draw_text s RX y 55 62 88)))
(define rhint (lambda (s y) (Gfx.draw_text s RX y CD_R CD_G CD_B)))

;; 第 0 步：输入
(define draw_step0
  (lambda () {
    (rtitle "输入 1x64x64 灰度" 38)
    (rhint "像素 0(黑) ~ 1(白)" 54)
    (rhint "归一化后送进网络：" 70)
    (Gfx.draw_text "x' = (x - 0.80) / 0.21" RX 82 CT_R CT_G CT_B)
    (rhint "这张图来自手写板" 102)
    (rhint "（或「示例」的内嵌样本）" 116)
    (rhint "按 > 走一步，" 140)
    (rhint "「自动」会一路跑完 11 步" 154)
    (rhint "点左图的框看细节" 176)
    (rhint "「卷积」页讲逐层计算" 190)
  }))

;; 第 1~4 步：特征图（带通道选择）
(define sc_buf 0)
(define sc_n 0)
(define sc_nch 0)
(define sc_s 0)
(define draw_step_fmap
  (lambda (step) {
    (set! sc_buf FR.f1)
    (set! sc_n 32)
    (set! sc_nch 8)
    (set! sc_s 4)
    (if (>= step 2)
        {
          (set! sc_n 16)
          (set! sc_nch 24)
          (set! sc_s 8)
        }
        #f)
    (if (== step 2) (set! sc_buf FR.f2) #f)
    (if (== step 3) (set! sc_buf FR.f3) #f)
    (if (== step 4) (set! sc_buf FR.rh) #f)
    (if (== step 1)
        (rtitle "conv1 3x3/s2 -> 32x32x8" 38)
        (if (== step 2)
            (rtitle "conv2 3x3/s2 -> 16x16x24" 38)
            (if (== step 3)
                (rtitle "conv3 3x3逐通道 + 1x1" 38)
                (rtitle "RPN 中间层 1x1 24->24" 38))))
    (bdraw S_CHL)
    (Gfx.draw_text_centered 1 (cat "通道" (cat (i2s sel_ch) (cat "/" (i2s sc_nch))))
                            (+ RX 52) 57 CT_R CT_G CT_B 1)
    (bdraw S_CHR)
    (draw_fmap sc_buf sc_n sel_ch RX 70 sc_s (if (== sel_on 1) sel_cell -1))
    (if (== step 1)
        (rhint "点格子: 输入上的 3x3 感受野" 204)
        (rhint "点格子: 它由哪些数据算出" 204))
  }))

;; 第 5 步：RPN 分类（前景概率图 + 选中格的 9 个锚框分数）
(define d5_r 0)
(define d5_c 0)
(define d5_m 0.0)
(define d5_k 0)
(define draw_step4
  (lambda () {
    (rtitle "RPN 分类: 每格 9 个锚框" 38)
    (set! d5_r 0)
    (while (< d5_r 16) {
      (set! d5_c 0)
      (while (< d5_c 16) {
        (set! d5_m 0.0)
        (set! d5_k 0)
        (while (< d5_k 9) {
          (set! d5_m (maxf d5_m (get_item FR.objp (+ (* d5_k 256) (+ (* d5_r 16) d5_c)))))
          (set! d5_k (+ d5_k 1))
        })
        (draw_act (+ RX (* d5_c 6)) (+ 52 (* d5_r 6)) 6 d5_m)
        (set! d5_c (+ d5_c 1))
      })
      (set! d5_r (+ d5_r 1))
    })
    (Gfx.draw_rect (- RX 1) 51 (+ (* 16 6) 2) (+ (* 16 6) 2) 200 206 218)
    (if (== sel_on 1)
        (Gfx.draw_rect (- (+ RX (* (mod sel_cell 16) 6)) 2)
                       (- (+ 52 (* (Math.floor (/ sel_cell 16)) 6)) 2)
                       10 10 20 90 200)
        #f)
    (rhint "亮 = 该格像有数字(前景概率高)" 152)
    (if (== sel_on 1)
        {
          (Gfx.draw_text (cat "格(" (cat (gstr (Math.floor (/ sel_cell 16)) (mod sel_cell 16))
                          ") 的 9 个锚框:")) RX 164 CT_R CT_G CT_B)
          (set! d5_k 0)
          (while (< d5_k 9) {
            (set! d5_m (get_item FR.objp (+ (* d5_k 256) sel_cell)))
            (Gfx.fill_rect (+ RX (* d5_k 12)) 178 10 30 236 239 245)
            (if (> d5_m 0.0)
                (Gfx.fill_rect (+ RX (* d5_k 12)) (- 208 (Math.round (* 30.0 d5_m))) 10
                               (maxi 1 (Math.round (* 30.0 d5_m)))
                               (if (> d5_m 0.5) 214 150) (if (> d5_m 0.5) 128 178)
                               (if (> d5_m 0.5) 50 214))
                #f)
            (Gfx.draw_text (i2s d5_k) (+ (+ RX 2) (* d5_k 12)) 210 110 118 135)
            (set! d5_k (+ d5_k 1))
          })
        }
        (rhint "先点一个格子看它的 9 个锚框" 164))
  }))

;; 第 6 步：RPN 回归（选中格的 9 组偏移）
(define draw_step5
  (lambda () {
    (rtitle "RPN 回归: 9 锚框 x 4 偏移" 38)
    (rhint "偏移是相对锚框的修正量" 54)
    (Gfx.draw_text "dx,dy 平移 dw,dh 缩放" RX 66 110 118 135)
    (if (== sel_on 1)
        {
          (Gfx.draw_text (cat "格(" (cat (gstr (Math.floor (/ sel_cell 16)) (mod sel_cell 16))
                          ") 的 9 组:")) RX 84 CT_R CT_G CT_B)
          (Gfx.draw_text "k    dx    dy    dw    dh" RX 98 150 155 172)
          (set! _i 0)
          (while (< _i 9) {
            (Gfx.draw_text (cat (i2s _i) ":") RX (+ 112 (* _i 11)) CT_R CT_G CT_B)
            (Gfx.draw_text (fmt2 (get_item FR.rreg (+ (* (+ (* _i 4) 0) 256) sel_cell)))
                           (+ RX 16) (+ 112 (* _i 11)) 70 78 100)
            (Gfx.draw_text (fmt2 (get_item FR.rreg (+ (* (+ (* _i 4) 1) 256) sel_cell)))
                           (+ RX 56) (+ 112 (* _i 11)) 70 78 100)
            (Gfx.draw_text (fmt2 (get_item FR.rreg (+ (* (+ (* _i 4) 2) 256) sel_cell)))
                           (+ RX 96) (+ 112 (* _i 11)) 70 78 100)
            (Gfx.draw_text (fmt2 (get_item FR.rreg (+ (* (+ (* _i 4) 3) 256) sel_cell)))
                           (+ RX 136) (+ 112 (* _i 11)) 70 78 100)
            (set! _i (+ _i 1))
          })
        }
        (rhint "先在上一步点一个格子" 104))
  }))

;; 第 7 步：提议解码 + NMS → RoI 列表
(define d7_y 0)
(define draw_step6
  (lambda () {
    (rtitle "提议解码 -> NMS -> RoI" 38)
    (Gfx.draw_text (cat "候选 " (cat (i2s FR.tk_n)
                    (cat " -> NMS -> " (cat (i2s FR.roi_n) " 个")))) RX 54 CT_R CT_G CT_B)
    (set! _i 0)
    (while (and (< _i FR.roi_n) (< _i 11)) {
      (set! d7_y (+ 66 (* _i 12)))
      (if (== _i sel_roi)
          (Gfx.fill_rect (- RX 2) (- d7_y 1) 172 12 214 232 250)
          #f)
      (Gfx.draw_text (cat "#" (i2s _i)) RX d7_y CT_R CT_G CT_B)
      (Gfx.draw_text (fmt2 (get_item FR.roi_s _i)) (+ RX 26) d7_y
                     (if (> (get_item FR.roi_s _i) 0.7) CA_R CD_R)
                     (if (> (get_item FR.roi_s _i) 0.7) CA_G CD_G)
                     (if (> (get_item FR.roi_s _i) 0.7) CA_B CD_B))
      (Gfx.draw_text (box_str0 FR.roi_b _i) (+ RX 58) d7_y 110 118 135)
      (set! _i (+ _i 1))
    })
    (if (> FR.roi_n 11)
        (Gfx.draw_text (cat "… 共 " (cat (i2s FR.roi_n) " 个，列出前 11")) RX 204
                       CD_R CD_G CD_B)
        (rhint "点左图或这里选一个 RoI" 204))
  }))

;; 第 8 步：RoI 池化（选中 RoI、选中通道的 4x4）
(define d8_i 0)
(define d8_j 0)
(define d8_v 0.0)
(define d8_m 0.001)
(define draw_step7
  (lambda () {
    (rtitle (cat "RoI 池化: RoI #" (i2s sel_roi)) 38)
    (set! d8_m 0.001)
    (set! d8_i 0)
    (while (< d8_i 16) {
      (set! d8_v (get_item FR.pooled (+ (* sel_roi FR.FDIM) (+ (* sel_ch 16) d8_i))))
      (if (> d8_v d8_m) (set! d8_m d8_v) #f)
      (set! d8_i (+ d8_i 1))
    })
    (set! d8_i 0)
    (while (< d8_i 4) {
      (set! d8_j 0)
      (while (< d8_j 4) {
        (set! d8_v (get_item FR.pooled (+ (* sel_roi FR.FDIM) (+ (* sel_ch 16) (+ (* d8_i 4) d8_j)))))
        (draw_act (+ RX (* d8_j 26)) (+ 70 (* d8_i 26)) 24 (/ d8_v d8_m))
        (if (== (+ (* d8_i 4) d8_j) sel_bin)
            (Gfx.draw_rect (- (+ RX (* d8_j 26)) 2) (- (+ 70 (* d8_i 26)) 2) 28 28 20 90 200)
            #f)
        (set! d8_j (+ d8_j 1))
      })
      (set! d8_i (+ d8_i 1))
    })
    (Gfx.draw_rect (- RX 1) 69 (+ (* 4 26) 2) (+ (* 4 26) 2) 200 206 218)
    (bdraw S_CHL)
    (Gfx.draw_text_centered 1 (cat "通道" (cat (i2s sel_ch) "/23")) (+ RX 52) 57
                            CT_R CT_G CT_B 1)
    (bdraw S_CHR)
    (rhint "每个 bin = 该块里的最大值" 176)
    (rhint "点左图的 RoI 换 RoI" 190)
    (Gfx.draw_text "384 = 24 通道 x 4x4" RX 204 CT_R CT_G CT_B)
  }))

;; 第 9 步：分类头 + 回归头
(define d9_c 0)
(define d9_i 0)
(define d9_x 0)
(define d9_h 0)
(define draw_step8
  (lambda () {
    (define r sel_roi)
    (rtitle "分类头 + 回归头" 38)
    (Gfx.draw_text (cat "RoI #" (i2s r)) RX 52 CT_R CT_G CT_B)
    (set! d9_i 0)
    (while (< d9_i 11) {
      (set! d9_x (+ (+ RX 2) (* d9_i 15)))
      (Gfx.fill_rect d9_x 68 13 82 236 239 245)
      (set! d9_h (Math.round (* 82.0 (get_item FR.clsp (+ (* r 11) d9_i)))))
      (if (> d9_h 0)
          (if (== d9_i (roi_argmax r))
              (Gfx.fill_rect d9_x (- 150 d9_h) 13 d9_h 30 110 210)
              (if (== d9_i 10)
                  (Gfx.fill_rect d9_x (- 150 d9_h) 13 d9_h 165 172 186)
                  (Gfx.fill_rect d9_x (- 150 d9_h) 13 d9_h 130 160 205)))
          #f)
      (Gfx.draw_text (if (== d9_i 10) "背" (i2s d9_i)) (+ d9_x 3) 152
                     (if (== d9_i (roi_argmax r)) CA_R 120)
                     (if (== d9_i (roi_argmax r)) CA_G 128)
                     (if (== d9_i (roi_argmax r)) CA_B 145))
      (set! d9_i (+ d9_i 1))
    })
    (set! d9_c (roi_argmax r))
    (Gfx.draw_text (cat "预测 " (if (== d9_c 10) "背景" (i2s d9_c))) RX 170 CA_R CA_G CA_B)
    (Gfx.draw_text (cat "置信 " (fmt_pct (get_item FR.clsp (+ (* r 11) d9_c))))
                   (+ RX 78) 170 CA_R CA_G CA_B)
    (rhint "预测类的 4 个细化偏移:" 186)
    (Gfx.draw_text (cat (fmt2 (get_item FR.hreg (+ (* r 44) (* d9_c 4))))
                     (cat "," (cat (fmt2 (get_item FR.hreg (+ (* r 44) (+ (* d9_c 4) 1))))
                     (cat "," (cat (fmt2 (get_item FR.hreg (+ (* r 44) (+ (* d9_c 4) 2))))
                     (cat "," (fmt2 (get_item FR.hreg (+ (* r 44) (+ (* d9_c 4) 3))))))))))
                   RX 200 70 78 100)
  }))

;; 第 10 步：最终检测列表
(define draw_step9
  (lambda () {
    (rtitle "逐类 NMS -> 最终检测" 38)
    (Gfx.draw_text (cat "共 " (cat (i2s FR.fin_n) " 个目标")) RX 54 CT_R CT_G CT_B)
    (set! _i 0)
    (while (and (< _i FR.fin_n) (< _i 11)) {
      (if (== _i sel_fin)
          (Gfx.fill_rect (- RX 2) (- (+ 70 (* _i 12)) 1) 172 12 216 240 224)
          #f)
      (Gfx.draw_text (cat "类" (i2s (get_item FR.fin_cls _i))) RX (+ 70 (* _i 12))
                     CT_R CT_G CT_B)
      (Gfx.draw_text (fmt_pct (get_item FR.fin_sc _i)) (+ RX 34) (+ 70 (* _i 12))
                     CA_R CA_G CA_B)
      (Gfx.draw_text (box_str0 FR.fin_b _i) (+ RX 88) (+ 70 (* _i 12)) 110 118 135)
      (set! _i (+ _i 1))
    })
    (if (== FR.fin_n 0)
        (Gfx.draw_text "没有检测到目标" RX 72 CD_R CD_G CD_B)
        #f)
    (Gfx.draw_text (cat "整网 " (cat (i2s infer_ms) " ms")) RX 190 CD_R CD_G CD_B)
  }))

(define draw_right
  (lambda () {
    (cond ((== stage 0) (draw_step0))
          ((== stage 1) (draw_step_fmap 1))
          ((== stage 2) (draw_step_fmap 2))
          ((== stage 3) (draw_step_fmap 3))
          ((== stage 4) (draw_step_fmap 4))
          ((== stage 5) (draw_step4))
          ((== stage 6) (draw_step5))
          ((== stage 7) (draw_step6))
          ((== stage 8) (draw_step7))
          ((== stage 9) (draw_step8))
          (else (draw_step9)))
  }))

;; ---- 结果页：左下角信息（三行，每行不超过 128px） -----------------------

(define ip1 "")
(define ip2 "")
(define ip3 "")
(define info_draw
  (lambda (r1 g1 b1 r2 g2 b2) {
    (Gfx.fill_rect 0 166 132 48 250 251 254)
    (Gfx.fill_rect 0 166 132 1 CL_R CL_G CL_B)
    (Gfx.draw_text ip1 4 170 r1 g1 b1)
    (Gfx.draw_text ip2 4 183 r2 g2 b2)
    (Gfx.draw_text ip3 4 196 110 118 135)
  }))

(define draw_info
  (lambda () {
    (cond
      ((== pick 2) {
        (set! ip1 (cat "RoI #" (cat (i2s sel_roi)
                          (cat " 分 " (fmt2 (get_item FR.roi_s sel_roi))))))
        (set! ip2 (cat "分类: " (if (== (roi_argmax sel_roi) 10) "背景"
                          (cat "数字" (i2s (roi_argmax sel_roi))))))
        (set! ip3 (cat "置信 " (fmt_pct (get_item FR.clsp
                          (+ (* sel_roi 11) (roi_argmax sel_roi))))))
        (info_draw CT_R CT_G CT_B CA_R CA_G CA_B)
      })
      ((== pick 3) {
        (set! ip1 (cat "检测: 数字" (i2s (get_item FR.fin_cls sel_fin))))
        (set! ip2 (cat "置信 " (fmt_pct (get_item FR.fin_sc sel_fin))))
        (set! ip3 (box_str FR.fin_b sel_fin))
        (info_draw CT_R CT_G CT_B CG_R CG_G CG_B)
      })
      (else {
        (set! ip1 (cat "第" (cat (i2s stage) (cat "步 " (get_item stage_names stage)))))
        (set! ip2 (cat "提议" (cat (i2s FR.tk_n) (cat " RoI " (i2s FR.roi_n)))))
        (set! ip3 (cat "检测" (cat (i2s FR.fin_n) " 个 点框看细节")))
        (info_draw CT_R CT_G CT_B CD_R CD_G CD_B)
      })
    )
  }))

;; ---- 结果页渲染 ---------------------------------------------------------

(define render_result
  (lambda () {
    (Gfx.fill_white)
    (draw_header)
    (draw_stagebar)
    (draw_image_at IMGX IMGY 2)
    (Gfx.draw_rect (- IMGX 1) (- IMGY 1) (+ 128 2) (+ 128 2) 150 158 172)
    (if (and (or (== stage 5) (== stage 6)) (== sel_on 1))
        (draw_anchors_cell (Math.floor (/ sel_cell 16)) (mod sel_cell 16))
        #f)
    (if (== stage 7) (draw_props) #f)
    (if (or (== stage 8) (== stage 9)) (draw_rois) #f)
    (if (== stage 10) (draw_fins) #f)
    (draw_right)
    (draw_info)
    (draw_phases)
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; ---- 卷积页 -------------------------------------------------------------

(define cv_title (lambda (s x y) (Gfx.draw_text s x y 55 62 88)))

(define cv_src 0)         ; 层2 的"源"面板显示 conv1 的哪个通道（与 conv2 的序号解耦）
(define cvu_n 8)          ; 当前层的通道数（通道按钮循环用）
(define cvu_n2 32)        ; 目标特征图的边长（格数）
(define cvu_s2 4)         ; 目标特征图每格的像素

(define cw_px 0)
(define cw_py 0)
(define cw_s 2)
(define draw_cv_window
  (lambda () {
    (define n (if (== cv_layer 0) 32 16))
    (define gx (mod cv_cell n))
    (define gy (Math.floor (/ cv_cell n)))
    (set! cw_s (if (== cv_layer 0) 2 4))
    (set! cw_px (+ 4 (* (- (* gx 2) 1) cw_s)))
    (set! cw_py (+ 52 (* (- (* gy 2) 1) cw_s)))
    (if (< cw_px 4) (set! cw_px 4) #f)
    (if (< cw_py 52) (set! cw_py 52) #f)
    (Gfx.draw_rect cw_px cw_py (+ 1 (* 3 cw_s)) (+ 1 (* 3 cw_s)) 222 70 45)
  }))

(define d1_i 0)
(define d1_j 0)
(define d1_sum 0.0)
(define d1_p 0.0)
(define d1_pm 0.0001)
(define d1_gx 0)
(define d1_gy 0)
(define d1_xp 0)
(define d1_wp 0)
(define draw_cv_detail1
  (lambda () {
    (set! d1_gx (mod cv_cell 32))
    (set! d1_gy (Math.floor (/ cv_cell 32)))
    (set! d1_pm 0.0001)
    (set! d1_j 0)
    (while (< d1_j 3) {
      (set! d1_i 0)
      (while (< d1_i 3) {
        (set! d1_p (* (get_item FR.xpad (+ (* (+ (* d1_gy 2) d1_j) 66) (+ (* d1_gx 2) d1_i)))
                      (get_item FR.W1 (+ (* cv_ch 9) (+ (* d1_j 3) d1_i)))))
        (set! d1_pm (maxf d1_pm (Math.abs d1_p)))
        (set! d1_i (+ d1_i 1))
      })
      (set! d1_j (+ d1_j 1))
    })
    (set! d1_sum (get_item FR.B1 cv_ch))
    (set! d1_j 0)
    (while (< d1_j 3) {
      (set! d1_i 0)
      (while (< d1_i 3) {
        (set! d1_xp (get_item FR.xpad (+ (* (+ (* d1_gy 2) d1_j) 66) (+ (* d1_gx 2) d1_i))))
        (set! d1_wp (get_item FR.W1 (+ (* cv_ch 9) (+ (* d1_j 3) d1_i))))
        (set! d1_p (* d1_xp d1_wp))
        (set! d1_sum (+ d1_sum d1_p))
        (draw_norm (+ 6 (* d1_i 8)) (+ 194 (* d1_j 8)) 7 d1_xp)
        (draw_wt (+ 40 (* d1_i 8)) (+ 194 (* d1_j 8)) 7 (/ d1_wp 2.5))
        (draw_wt (+ 74 (* d1_i 8)) (+ 194 (* d1_j 8)) 7 (/ d1_p d1_pm))
        (set! d1_i (+ d1_i 1))
      })
      (set! d1_j (+ d1_j 1))
    })
    (Gfx.draw_text_centered 1 "入" 18 226 120 126 140 1)
    (Gfx.draw_text_centered 1 "核" 52 226 120 126 140 1)
    (Gfx.draw_text_centered 1 "积" 86 226 120 126 140 1)
    (Gfx.draw_text (cat "核" (i2s cv_ch)) 4 184 CT_R CT_G CT_B)
    (Gfx.draw_text "3x3 窗口在输入上滑动" 104 184 CD_R CD_G CD_B)
    (Gfx.draw_text "步长2 每落一次得右图一格" 104 198 CD_R CD_G CD_B)
    (Gfx.draw_text (cat "和 = " (fmt2 d1_sum)) 104 214 CT_R CT_G CT_B)
    (Gfx.draw_text (cat "ReLU -> " (fmt2 (maxf d1_sum 0.0))) 104 228 195 70 35)
    (Gfx.draw_text (cat "格" (gstr d1_gy d1_gx)) 250 184 CA_R CA_G CA_B)
    (Gfx.draw_text "9 个乘积求和" 238 228 CD_R CD_G CD_B)
  }))

(define d2_c 0)
(define d2_ky 0)
(define d2_kx 0)
(define d2_s 0.0)
(define d2_v 0.0)
(define d2_m 0.0001)
(define d2_tot 0.0)
(define d2_gx 0)
(define d2_gy 0)
(define d2_bar 0)
(define d2_x 0)
(define d2_contrib
  (lambda (ch) {
    (set! d2_s 0.0)
    (set! d2_ky 0)
    (while (< d2_ky 3) {
      (set! d2_kx 0)
      (while (< d2_kx 3) {
        (set! d2_v (get_item FR.p1 (+ (* ch 1156)
                       (+ (* (+ (* d2_gy 2) d2_ky) 34) (+ (* d2_gx 2) d2_kx)))))
        (set! d2_s (+ d2_s (* d2_v (get_item FR.W2 (+ (* cv_ch 72)
                        (+ (* ch 9) (+ (* d2_ky 3) d2_kx)))))))
        (set! d2_kx (+ d2_kx 1))
      })
      (set! d2_ky (+ d2_ky 1))
    })
    d2_s
  }))

(define draw_cv_detail2
  (lambda () {
    (set! d2_gx (mod cv_cell 16))
    (set! d2_gy (Math.floor (/ cv_cell 16)))
    (set! d2_m 0.0001)
    (set! d2_c 0)
    (while (< d2_c 8) {
      (set! d2_v (d2_contrib d2_c))
      (set! d2_m (maxf d2_m (Math.abs d2_v)))
      (set! d2_c (+ d2_c 1))
    })
    (set! d2_tot (get_item FR.B2 cv_ch))
    (set! d2_c 0)
    (while (< d2_c 8) {
      (set! d2_v (d2_contrib d2_c))
      (set! d2_tot (+ d2_tot d2_v))
      (set! d2_x (+ 6 (* d2_c 12)))
      (Gfx.draw_text_centered 1 (i2s d2_c) (+ d2_x 5) 226 110 118 135 1)
      (Gfx.fill_rect d2_x 188 10 36 236 239 245)
      (set! d2_bar (Math.round (* 17.0 (/ (Math.abs d2_v) d2_m))))
      (if (>= d2_v 0.0)
          (Gfx.fill_rect d2_x (- 206 d2_bar) 10 d2_bar 210 110 60)
          (Gfx.fill_rect d2_x 206 10 d2_bar 60 110 210))
      (set! d2_c (+ d2_c 1))
    })
    (Gfx.fill_rect 4 205 98 1 168 176 194)
    (Gfx.draw_text (cat "核" (i2s cv_ch)) 4 184 CT_R CT_G CT_B)
    (Gfx.draw_text (cat "和 = " (fmt2 d2_tot)) 108 186 CT_R CT_G CT_B)
    (Gfx.draw_text (cat "ReLU -> " (fmt2 (maxf d2_tot 0.0))) 108 202 195 70 35)
    (Gfx.draw_text "8 个通道各贡献一份" 108 218 CD_R CD_G CD_B)
    (Gfx.draw_text (cat "格" (gstr d2_gy d2_gx)) 250 184 CA_R CA_G CA_B)
  }))

(define draw_cvbar
  (lambda () {
    (Gfx.fill_rect 0 18 SW 16 252 253 255)
    (bdraw S_CVL)
    (Gfx.draw_text_centered 1 (cat "通道" (cat (i2s cv_ch)
                    (cat "/" (i2s (if (== cv_layer 0) 7 23))))) 129 26 CT_R CT_G CT_B 1)
    (bdraw S_CVR)
    (bdraw S_CVA)
    (Gfx.draw_text_centered 1 (if (== cv_layer 0) "3x3 步长2" "8ch x 3x3") 264 26
                            CD_R CD_G CD_B 1)
    (Gfx.fill_rect 0 34 SW 1 CL_R CL_G CL_B)
  }))

(define render_conv
  (lambda () {
    (Gfx.fill_white)
    (draw_header)
    (draw_cvbar)
    (if (== cv_layer 0)
        {
          (cv_title "源: 输入 64x64" 4 38)
          (draw_image_at 4 52 2)
          (Gfx.draw_rect 3 51 130 130 CL_R CL_G CL_B)
          (draw_cv_window)
          (cv_title (cat "目标: conv1 通道" (i2s cv_ch)) 150 38)
          (draw_fmap FR.f1 32 cv_ch 150 52 4 cv_cell)
        }
        {
          (cv_title (cat "源: conv1 通道" (i2s cv_src)) 4 38)
          (draw_fmap FR.f1 32 cv_src 4 52 4 -1)
          (draw_cv_window)
          (cv_title (cat "目标: conv2 通道" (i2s cv_ch)) 150 38)
          (draw_fmap FR.f2 16 cv_ch 150 52 8 cv_cell)
        })
    (Gfx.fill_rect 0 181 SW 1 CL_R CL_G CL_B)
    (if (== stage 0)
        {
          (Gfx.draw_text "还没有跑过推理：先在画板写数字、点「开始识别」，" 4 186 CT_R CT_G CT_B)
          (Gfx.draw_text "再来看这里的逐层计算细节。" 4 202 CD_R CD_G CD_B)
        }
        (if (== cv_layer 0) (draw_cv_detail1) (draw_cv_detail2)))
    (EX.draw_button)
    (Gfx.refresh)
  }))

;; ---- 计算推进 -----------------------------------------------------------

(define start_stage
  (lambda (s) {
    (if (>= s 11)
        #f
        (if (== busy 1)
            #f
            {
              (set! todo s)
              (set! busy 1)
              (set! prog 0.0)
              (set! stage_t0 (System.timestamp))
              (FR.begin_stage s)
            }))
  }))

(define fx_nch 24)
(define fx_cells 256)
(define fix_sel
  (lambda () {
    (set! fx_nch 24)
    (set! fx_cells 256)
    (if (== stage 1)
        {
          (set! fx_nch 8)
          (set! fx_cells 1024)
        }
        #f)
    (if (== stage 8)
        {
          (set! fx_nch 24)
          (set! fx_cells 16)
        }
        #f)
    (if (== stage 1) (set! sel_ch (best_ch FR.f1 32 8)) #f)
    (if (== stage 2) (set! sel_ch (best_ch FR.f2 16 24)) #f)
    (if (== stage 3) (set! sel_ch (best_ch FR.f3 16 24)) #f)
    (if (== stage 4) (set! sel_ch (best_ch FR.rh 16 24)) #f)
    (if (>= sel_ch fx_nch) (set! sel_ch 0) #f)
    (if (>= sel_cell fx_cells) (set! sel_cell 0) #f)
    (if (>= sel_roi (maxi 1 FR.roi_n)) (set! sel_roi 0) #f)
    (if (>= sel_fin (maxi 1 FR.fin_n)) (set! sel_fin 0) #f)
  }))

;; ---- 输入：结果页右栏/左图 ---------------------------------------------

(define hit_nch 24)
(define hit_right
  (lambda (px py) {
    (define r 0)
    (cond
      ((and (>= stage 1) (<= stage 4)) {
        (set! hit_nch (if (== stage 1) 8 24))
        (if (in_rect px py RX 70 (* sc_n sc_s) (* sc_n sc_s))
            {
              (set! sel_cell (+ (* (Math.floor (/ (- py 70) sc_s)) sc_n)
                                (Math.floor (/ (- px RX) sc_s))))
              (set! sel_on 1)
              (set! pick 0)
              1
            }
            (if (bhit S_CHL px py)
                {
                  (if (> sel_ch 0) (set! sel_ch (- sel_ch 1)) (set! sel_ch (- hit_nch 1)))
                  1
                }
                (if (bhit S_CHR px py)
                    {
                      (if (< sel_ch (- hit_nch 1)) (set! sel_ch (+ sel_ch 1)) (set! sel_ch 0))
                      1
                    }
                    0)))
      })
      ((== stage 5)
        (if (in_rect px py RX 52 96 96)
            {
              (set! sel_cell (+ (* (Math.floor (/ (- py 52) 6.0)) 16)
                                (Math.floor (/ (- px RX) 6.0))))
              (set! sel_on 1)
              (set! pick 0)
              1
            }
            0))
      ((== stage 7)
        (if (in_rect px py RX 66 172 148)
            {
              (set! r (Math.floor (/ (- py 70) 11.0)))
              (if (and (>= r 0) (< r FR.roi_n))
                  {
                    (set! sel_roi r)
                    (set! pick 2)
                    1
                  }
                  0)
            }
            0))
      ((== stage 8) {
        (if (in_rect px py RX 70 104 104)
            {
              (set! sel_bin (+ (* (Math.floor (/ (- py 70) 26.0)) 4)
                               (Math.floor (/ (- px RX) 26.0))))
              1
            }
            (if (bhit S_CHL px py)
                {
                  (if (> sel_ch 0) (set! sel_ch (- sel_ch 1)) (set! sel_ch 23))
                  1
                }
                (if (bhit S_CHR px py)
                    {
                      (if (< sel_ch 23) (set! sel_ch (+ sel_ch 1)) (set! sel_ch 0))
                      1
                    }
                    0)))
      })
      ((== stage 10)
        (if (in_rect px py RX 64 172 150)
            {
              (set! r (Math.floor (/ (- py 70) 12.0)))
              (if (and (>= r 0) (< r FR.fin_n))
                  {
                    (set! sel_fin r)
                    (set! pick 3)
                    1
                  }
                  0)
            }
            0))
      (else 0))}))

(define hi_best 0)
(define hi_area 0.0)
(define hi_besta 1e9)
(define hi_x1 0.0)
(define hi_y1 0.0)
(define hi_x2 0.0)
(define hi_y2 0.0)
(define hi_any 0)

;; 点到了哪个 RoI（选最小的那个框），返回 1 表示命中，hi_best 给下标
(define hit_roi_at
  (lambda (px py) {
    (set! hi_any 0)
    (set! hi_best 0)
    (set! hi_besta 1e9)
    (set! _i 0)
    (while (< _i FR.roi_n) {
      (set! hi_x1 (sx_of (get_item FR.roi_b (* _i 4))))
      (set! hi_y1 (sy_of (get_item FR.roi_b (+ (* _i 4) 1))))
      (set! hi_x2 (sx_of (get_item FR.roi_b (+ (* _i 4) 2))))
      (set! hi_y2 (sy_of (get_item FR.roi_b (+ (* _i 4) 3))))
      (if (and (>= px hi_x1) (<= px hi_x2) (>= py hi_y1) (<= py hi_y2))
          {
            (set! hi_area (* (- hi_x2 hi_x1) (- hi_y2 hi_y1)))
            (if (< hi_area hi_besta)
                {
                  (set! hi_besta hi_area)
                  (set! hi_best _i)
                  (set! hi_any 1)
                }
                #f)
          }
          #f)
      (set! _i (+ _i 1))
    })
    hi_any
  }))

;; 点到了哪个最终检测
(define hit_fin_at
  (lambda (px py) {
    (set! hi_any 0)
    (set! hi_best 0)
    (set! hi_besta 1e9)
    (set! _i 0)
    (while (< _i FR.fin_n) {
      (set! hi_x1 (sx_of (get_item FR.fin_b (* _i 4))))
      (set! hi_y1 (sy_of (get_item FR.fin_b (+ (* _i 4) 1))))
      (set! hi_x2 (sx_of (get_item FR.fin_b (+ (* _i 4) 2))))
      (set! hi_y2 (sy_of (get_item FR.fin_b (+ (* _i 4) 3))))
      (if (and (>= px hi_x1) (<= px hi_x2) (>= py hi_y1) (<= py hi_y2))
          {
            (set! hi_area (* (- hi_x2 hi_x1) (- hi_y2 hi_y1)))
            (if (< hi_area hi_besta)
                {
                  (set! hi_besta hi_area)
                  (set! hi_best _i)
                  (set! hi_any 1)
                }
                #f)
          }
          #f)
      (set! _i (+ _i 1))
    })
    hi_any
  }))

;; 结果页左图点击：最终检测优先（第 10 步），否则选 RoI（第 7~9 步）
(define dh_hit 0)
(define draw_img_hit
  (lambda (px py) {
    (set! dh_hit 1)
    (if (< px IMGX) (set! dh_hit 0) #f)
    (if (>= px (+ IMGX 128)) (set! dh_hit 0) #f)
    (if (< py IMGY) (set! dh_hit 0) #f)
    (if (>= py (+ IMGY 128)) (set! dh_hit 0) #f)
    (if (== dh_hit 1)
        {
          (if (== stage 10)
              (if (== (hit_fin_at px py) 1)
                  {
                    (set! sel_fin hi_best)
                    (set! pick 3)
                  }
                  #f)
              #f)
          (if (or (== stage 7) (== stage 8) (== stage 9))
              (if (== (hit_roi_at px py) 1)
                  {
                    (set! sel_roi hi_best)
                    (set! pick 2)
                  }
                  #f)
              #f)
        }
        #f)
    dh_hit
  }))

;; 开始识别：清掉上一轮结果，从第 0 步自动跑到底
(define run_all
  (lambda () {
    (set! busy 0)
    (set! sel_on 0)
    (set! sel_roi 0)
    (set! sel_fin 0)
    (set! pick 0)
    (set! infer_ms 0)
    (set! mode 1)
    (set! stage 0)
    (set! autorun 1)
    (FR.reset_results)
    (start_stage 0)
  }))

;; 换一张内嵌样本（自动跑）
(define load_next_sample
  (lambda () {
    (FR.synth_random)
    (set! drawn 1)
    (if (== mode 0) #f (run_all))
  }))

(define clear_board
  (lambda () {
    (FR.clear_image)
    (FR.reset_results)
    (set! drawn 0)
    (set! busy 0)
    (set! stage 0)
    (set! infer_ms 0)
  }))

;; ---- 笔划 ---------------------------------------------------------------

(define ln_n 0)
(define ln_i 1)
(define ln_dx 0.0)
(define ln_dy 0.0)
(define stroke_to
  (lambda (gx gy) {
    (set! ln_dx (- gx pen_last_x))
    (set! ln_dy (- gy pen_last_y))
    (set! ln_n (Math.ceil (/ (Math.sqrt (+ (* ln_dx ln_dx) (* ln_dy ln_dy))) 0.6)))
    (if (< ln_n 1) (set! ln_n 1) #f)
    (set! ln_i 1)
    (while (<= ln_i ln_n) {
      (FR.stamp (+ pen_last_x (/ (* ln_dx ln_i) ln_n))
                (+ pen_last_y (/ (* ln_dy ln_i) ln_n))
                brush_r 1.0)
      (set! ln_i (+ ln_i 1))
    })
    (set! pen_last_x gx)
    (set! pen_last_y gy)
    (set! drawn 1)
  }))

;; ---- 每拍 ---------------------------------------------------------------

(define tick
  (lambda () {
    ;; 按下沿用宿主**锁存**的 take_press（读后清零）：快速点按不会漏。代价是它与
    ;; (EX.check) 里的按钮控件争同一个沿——所以 EX.check 挪到本拍所有输入处理**之后**
    ;; 再调（EX 自己的按钮在右上角，和本 App 的按钮不重叠）。
    (set! _p (if (Touch.take_press) 1 0))
    (set! _tx (Touch.x))
    (set! _ty (Touch.y))
    ;; 拉起本 App 的那一次点按（应用列表里的图标）会残留一个按下电平与一次"松开"沿，
    ;; 直接当输入用就会在画板上凭空多一笔、甚至误触「开始识别」。这里等到**第一次看到
    ;; 手指离开屏幕**之后才武装输入（与宿主 am_native_Touch_drop_inflight 同一语义）。
    (if (== armed 0)
        (if (Touch.is_pressed)
            {
              (set! _p 0)
              (set! pen_down 0)
            }
            (set! armed 1))
        #f)
    (if (== armed 0) (set! _p 0) #f)
    (if (and (== armed 0) (== pen_down 1)) (set! pen_down 0) #f)

    ;; 本拍按钮的文案与激活态（切换模式后立刻生效；只写字段，不分配）
    (ui_sync)

    ;; ---- 顶部按钮（三种模式复用同一组槽位：位置相同，只是文案与动作不同） ----
    (if (== _p 1)
        {
          (if (bhit S_H0 _tx _ty)
              (if (== mode 2)
                  {
                    (set! cv_layer 0)
                    (set! cv_ch (best_ch FR.f1 32 8))
                    (set! cv_src cv_ch)
                    (set! cv_cell 0)
                    (set! cv_anim 1)
                  }
                  {
                    (clear_board)
                    (set! mode 0)
                  })
              #f)
          (if (bhit S_H1 _tx _ty)
              (if (== mode 2)
                  {
                    (set! cv_layer 1)
                    (set! cv_ch (best_ch FR.f2 16 24))
                    (set! cv_src (best_ch FR.f1 32 8))
                    (set! cv_cell 0)
                    (set! cv_anim 1)
                  }
                  (load_next_sample))
              #f)
          (if (bhit S_H2 _tx _ty)
              (if (== mode 2)
                  (set! mode 1)
                  {
                    (set! mode 2)
                    (set! cv_ch (if (== cv_layer 0) (best_ch FR.f1 32 8)
                                   (best_ch FR.f2 16 24)))
                    (set! cv_src (if (== cv_layer 0) cv_ch (best_ch FR.f1 32 8)))
                    (set! cv_anim 1)
                  })
              #f)
        }
        #f)

    ;; ---- 绘制页：画板 / 工具 ----
    (if (== mode 0)
        {
          (if (== _p 1)
              {
                (if (bhit S_FINE _tx _ty)
                    (set! brush_r (maxf 0.8 (- brush_r 0.3)))
                    #f)
                (if (bhit S_COAR _tx _ty)
                    (set! brush_r (minf 3.6 (+ brush_r 0.3)))
                    #f)
                (if (bhit S_SAMP _tx _ty) (load_next_sample) #f)
                (if (bhit S_RUN _tx _ty)
                    (if (== drawn 1) (run_all) #f)
                    #f)
                (if (in_rect _tx _ty CVX CVY CVW CVW)
                    {
                      (set! pen_down 1)
                      (set! pen_last_x (/ (- _tx CVX) 3.0))
                      (set! pen_last_y (/ (- _ty CVY) 3.0))
                      (FR.stamp pen_last_x pen_last_y brush_r 1.0)
                      (set! drawn 1)
                    }
                    #f)
              }
              #f)
          (if (== pen_down 1)
              (if (Touch.is_pressed)
                  (stroke_to (maxf 0.0 (minf 63.0 (/ (- _tx CVX) 3.0)))
                             (maxf 0.0 (minf 63.0 (/ (- _ty CVY) 3.0))))
                  (set! pen_down 0))
              #f)
        }
        #f)

    ;; ---- 结果页：步骤条 / 右栏 / 左图 ----
    (if (== mode 1)
        {
          (if (== _p 1)
              {
                (if (bhit S_PREV _tx _ty)
                    (if (== busy 0)
                        {
                          (set! autorun 0)
                          (if (> stage 0) (set! stage (- stage 1)) #f)
                        }
                        #f)
                    #f)
                (if (bhit S_NEXT _tx _ty)
                    (if (== busy 0)
                        {
                          (set! autorun 0)
                          (if (< stage 10) (start_stage (+ stage 1)) #f)
                        }
                        #f)
                    #f)
                (if (bhit S_AUTO _tx _ty)
                    {
                      (set! autorun (if (== autorun 1) 0 1))
                      (if (== autorun 1) (set! auto_wait 0) #f)
                    }
                    #f)
              }
              #f)
          (if (and (== _p 1) (== busy 0))
              (if (== (hit_right _tx _ty) 0) (draw_img_hit _tx _ty) #f)
              #f)
        }
        #f)

    ;; ---- 卷积页 ----
    (if (== mode 2)
        {
          (if (== _p 1)
              {
                (set! cvu_n (if (== cv_layer 0) 8 24))
                (set! cvu_n2 (if (== cv_layer 0) 32 16))
                (set! cvu_s2 (if (== cv_layer 0) 4 8))
                (if (bhit S_CVL _tx _ty)
                    (if (> cv_ch 0) (set! cv_ch (- cv_ch 1)) (set! cv_ch (- cvu_n 1)))
                    #f)
                (if (bhit S_CVR _tx _ty)
                    (if (< cv_ch (- cvu_n 1)) (set! cv_ch (+ cv_ch 1)) (set! cv_ch 0))
                    #f)
                (if (bhit S_CVA _tx _ty)
                    (set! cv_anim (if (== cv_anim 1) 0 1))
                    #f)
                ;; 点目标特征图的任一格 → 窗口跳过去
                (if (in_rect _tx _ty 150 52 (* cvu_n2 cvu_s2) (* cvu_n2 cvu_s2))
                    {
                      (set! cv_cell (+ (* (Math.floor (/ (- _ty 52) cvu_s2)) cvu_n2)
                                       (Math.floor (/ (- _tx 150) cvu_s2))))
                      (set! cv_anim 0)
                    }
                    #f)
              }
              #f)
        }
        #f)

    ;; ---- 推进计算 ----
    (if (== busy 1)
        {
          (set! prog (FR.step_stage todo (get_item BUDGET todo)))
          (if (>= prog 1.0)
              {
                (set! stage todo)
                (set! busy 0)
                (set! prog 0.0)
                (set! infer_ms (+ infer_ms (- (System.timestamp) stage_t0)))
                (set! auto_wait (+ (System.timestamp) 420))
                (fix_sel)
                (display "[frcnn] stage=") (display stage)
                (display " rois=") (display FR.roi_n)
                (display " dets=") (display FR.fin_n)
                (display " ms=") (display infer_ms) (newline)
              }
              #f)
        }
        #f)

    ;; ---- 自动模式（只在结果页推进） ----
    (if (and (== autorun 1) (== busy 0) (== mode 1) (< stage 10))
        (if (>= (System.timestamp) auto_wait)
            (start_stage (+ stage 1))
            #f)
        #f)

    ;; ---- 卷积页：窗口自动滑动 ----
    (if (and (== mode 2) (== cv_anim 1) (> (System.timestamp) cv_t))
        {
          (set! cv_t (+ (System.timestamp) CV_MS))
          (set! cv_cell (+ cv_cell 1))
          (if (>= cv_cell (* (if (== cv_layer 0) 32 16) (if (== cv_layer 0) 32 16)))
              (set! cv_cell 0)
              #f)
        }
        #f)

    ;; ---- 退出按钮/返回请求（放在输入处理之后，免得抢走本拍的按下沿）----
    (EX.check)

    ;; ---- 渲染 ----
    (if (== mode 0) (render_draw) (if (== mode 1) (render_result) (render_conv)))
  }))

;; 统一重绘（EX 退出确认框关掉后擦残影）
(define render_any
  (lambda () {
    (if (== mode 0) (render_draw) (if (== mode 1) (render_result) (render_conv)))
  }))

;; ---- 启动：空白画板，等用户写 ----

(btn_init)
(FR.init)
(FR.clear_image)
(set! mode 0)
(set! drawn 0)
(render_draw)
(set! _t (System.set_interval TICK_MS tick))

(EX.install_with render_any)
