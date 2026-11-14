;; ============================================================================
;; 三维弹球 / SPACE CADET —— Nano「灵机应用程序」Scheme App（单文件）
;;
;; 复刻对象：Windows XP 自带的「三维弹球 - 太空军校生」（3D Pinball for Windows
;; – Space Cadet，Cinematronics / Maxis 1995）。原版是全 3D 台面 + 预渲染位图，
;; 这里做的是**二维台面上的等价物**：保留原版的台面构成（三个弹射柱、左下三连
;; 翻板落靶、右侧立靶、顶部三条滚道灯、左上旋转器、右上踢球洞、右侧发射通道）、
;; 计分与军衔晋升（Cadet → Ensign → Lieutenant → Captain → Admiral）、三球一局、
;; 倍数递增与"任务目标"机制，视觉上是深蓝紫星域 + 霓虹描边的矢量台面。
;;
;; 物理（自研，参照 SpaceCadetPinball 反编译源码的模型简化，见
;; k4zmu2a/SpaceCadetPinball 的 TBall/TEdgeSegment/TFlipperEdge）：
;;   · 球是半径 5 的圆，重力 900 px/s²，速度上限 800 px/s；
;;   · 静态碰撞体：线段（台面外框 / 穹顶 / 挡板 / 落靶 / 单向门），球心到线段的
;;     最近点做穿透推出 + 法向反弹（各线段有自己的恢复系数与"踢击"冲量）；
;;   · 圆碰撞体：三个弹射柱（反弹 + 沿法向 190 px/s 的踢出 + 切向随机）与三个立靶；
;;   · 挡板（flipper）是绕枢轴旋转的胶囊；碰撞时把**接触点的表面速度**
;;     v = ω × r 计入相对速度，于是"挥拍"自然把球打飞（ω 取自本帧角度变化）；
;;   · 每帧 6 个子步（dt ≤ 40ms → 子步 ≤ 6.7ms，球每步位移 < 5px），防穿透。
;;
;; 玩法（按键与触屏）：
;;   · 左挡板：Z / A / ← ；右挡板：X / / / →（触屏：屏幕左半 / 右半，**任何状态都响应**）；
;;   · 发射：球停在发射道里时，按住空格 / 回车 / 任意触点蓄力（1s 满），松手弹出；
;;     闲置 8s 自动满力发射（触屏设备不会卡在待发状态）；
;;   · 掉球后有 ball save（发球后 7s 内不扣球，每球一次）；三球打完 → 结算面板 →
;;     点一下重开；
;;   · 右上角「退出」按钮 = app/lib/exit.scm 的全局退出确认框（触屏按钮与硬 A/ESC
;;     同一路径）。
;;
;; 计分表（原版数值的量级压缩到本作尺度，"×倍数"表示乘当前倍数）：
;;   弹射柱 500、弹弓（sling）100、落靶 1000（清空 10000 且倍数 +1）、
;;   立靶 750（三靶齐亮 5000）、旋转器 250/次、顶部滚道 500/条
;;   （三条齐亮 10000 + 奖励球）、踢球洞 15000（连进加成，第三次起封顶）。
;;
;; 台面几何集中在常量区与 build_table():  机箱 (68,2)-(316,240)；台面左墙 x=76、
;; 右墙/发射道隔墙 x=286、穹顶为圆心 (181,130.6) 半径 106.5 的圆弧（170°→56°）；
;; 发射道 x 290..316、单向门 y=90、顶部导流斜板 (316,86)-(284,56)。
;;
;; 解释器约束（manual.md，逐条遵守）：算术/比较恒二元；无 let/list/map/apply；
;; cond 子句只接受单个表达式；`define` 只允许出现在 lambda 体顶层（块内禁止）；
;; / 与 mod 恒返回浮点（下标/取整用 Math.floor）；wasm32 上 System.timestamp 是
;; 27 位回绕值，因此时间差要做 +2^27 校正并钳位。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native App)

(import EX "app/lib/exit.scm")

;; ============================================================================
;; 常量
;; ============================================================================

(define PI2 3.14159265358979)
(define RAD 0.0174532925199433)     ;; 度 → 弧度
(define DEG 57.2957795130823)       ;; 弧度 → 度

(define BALL_R 5.0)
(define GRAV 900.0)
(define MAXV 800.0)
(define NSUB 6)
(define TICK_MS 16)
(define DT_MAX_MS 40)

;; 台面几何（屏幕坐标，y 向下）
(define CAB_L 68)
(define CAB_R 316)
(define CAB_T 2)
(define CAB_B 240)
(define F_L 76)                     ;; 台面左墙
(define F_R 286)                    ;; 台面右墙（发射道隔墙）
(define DOME_CX 181.0)
(define DOME_CY 130.6)
(define DOME_R 106.5)
(define DOME_Y 112.0)               ;; 穹顶与左右墙的接点高度
(define LANE_L 290)                 ;; 发射道内壁
(define LANE_R 316)                 ;; 发射道外壁（= 机箱右边）
(define LANE_TOP 84)                ;; 隔墙上端（此高度以上与台面连通）
(define GATE_Y 90)                  ;; 单向门
(define DRAIN_Y 236)

;; 挡板
(define FLIP_LEN 47.0)
(define FLIP_RAD 3.5)
(define PIV_LX 126.0)
(define PIV_LY 202.0)
(define PIV_RX 236.0)
(define PIV_RY 202.0)
(define REST_L (* 18.0 RAD))        ;; 左挡板静止角（向下 18°）：两挡板尖端中缝 20.6px
                                    ;; （胶囊表面间 13.6px）——球直径 10px，只有几乎正中
                                    ;; 的球才漏得下去（D 项：减少中缝直落）
(define UP_L (* -40.0 RAD))         ;; 左挡板抬起角（向上 40°）
(define REST_R (- (* 180.0 RAD) REST_L))
(define UP_R (+ (* 180.0 RAD) (* 40.0 RAD)))
(define ZONE_L 106)                   ;; 触屏分区边界：x<106 左挡板，106..214 双拍，>214 右挡板
(define ZONE_R 214)
(define FLIP_GRACE 0.06)            ;; 连拍宽限（秒）：只够极快连拍衔接；
                                    ;; 再长就有明显粘手感（回位延迟 = 宽限 + 12ms）

(define FLIP_UP_SPD 38.0)           ;; rad/s：58° 约 27ms 挥满（≈1.7 帧）
(define FLIP_DN_SPD 26.0)           ;; rad/s：回位 18° 约 12ms，松手即落

;; 弹射柱 / 立靶
(define NBUMP 3)
(define NSU 3)

;; 计分
(define SC_BUMPER 500)
(define SC_SLING 100)
(define SC_TARGET 1000)
(define SC_BANK 10000)
(define SC_STANDUP 750)
(define SC_MISSION 5000)
(define SC_SPIN 250)
(define SC_LANE 500)
(define SC_LANES 10000)
(define SC_HOLE 15000)
(define RANK_PTS '(40000 100000 200000 350000 600000))
(define RANK_NAME '("CADET" "ENSIGN" "LT" "CAPTAIN" "ADMIRAL"))

;; ============================================================================
;; 状态
;; ============================================================================

;; 球：bst 0 = 在发射道待发，1 = 在台面，2 = 被踢球洞捕获，3 = 掉球等待重发，
;;     4 = 游戏结束
(define bx 303.0)
(define by 228.0)
(define bvx 0.0)
(define bvy 0.0)
(define bst 0)
(define btimer 0.0)
(define btrail_x (cdr '(0)))
(define btrail_y (cdr '(0)))
(define left_lane 0)
(define stuck_t 0.0)
(define play_t 0.0)
(define idle_t 0.0)          ;; 本次投球已在台面上的时间（秒）
(define save_used 0)         ;; 本球是否已用过 ball save
(define btrail_head 0)

(define score 0)
(define balls 3)
(define mult 1)
(define rankidx 0)
(define charge 0.0)
(define charging 0)
(define game_over 0)

(define msg_text "")
(define msg_timer 0.0)
(define msg_color 0)

(define fl_ang REST_L)
(define fl_omega 0.0)
(define fl_on 0)
(define fr_ang REST_R)
(define fr_omega 0.0)
(define fr_on 0)
(define last_fl -1)
(define last_fr -1)
(define l_latch 0.0)         ;; 连拍宽限计时：>0 表示"刚松手不久"，若另一侧此刻按下则继续保持抬起
(define r_latch 0.0)

(define last_ts 0)
(define frame_no 0)
(define score_dirty 1)
(define score_str "00000000")
(define ball_str "3")
(define mult_str "X1")
(define last_stat 0)

;; 台面元素状态
(define tg_down (cdr '(0)))         ;; 落靶：1 = 已倒下
(define tg_timer 0.0)
(define su_lit (cdr '(0)))          ;; 立靶：1 = 已点亮
(define lane_lit (cdr '(0)))        ;; 滚道灯
(define lane_in (cdr '(0)))         ;; 上一帧是否在滚道区内（边沿检测）
(define spin_phase 0.0)
(define spin_cd 0.0)
(define spin_count 0)
(define hole_cd 0.0)
(define hole_hits 0)
(define bumper_flash (cdr '(0)))
(define bumper_cd (cdr '(0)))
(define sling_flash 0.0)
(define extra_given 0)

;; 粒子
(define NPART 24)
(define part_x (cdr '(0)))
(define part_y (cdr '(0)))
(define part_vx (cdr '(0)))
(define part_vy (cdr '(0)))
(define part_life (cdr '(0)))
(define part_col (cdr '(0)))
(define part_next 0)

;; 星野（启动时生成）
(define star_x (cdr '(0)))
(define star_y (cdr '(0)))
(define star_b (cdr '(0)))
(define NSTAR 54)

;; 台面碰撞体（并行数组）
(define nseg 0)
(define sx1 (cdr '(0)))
(define sy1 (cdr '(0)))
(define sx2 (cdr '(0)))
(define sy2 (cdr '(0)))
(define srad (cdr '(0)))
(define srest (cdr '(0)))
(define skick (cdr '(0)))
(define styp (cdr '(0)))            ;; 0 墙 / 1 弹弓 / 2 落靶 / 3 单向门 / 4 隔墙
(define sact (cdr '(0)))
(define sinv (cdr '(0)))
(define sid (cdr '(0)))
(define seg_gate 0)
(define tg_seg (cdr '(0)))

;; 圆碰撞体（弹射柱 + 立靶）
(define ncir 0)
(define cx_ (cdr '(0)))
(define cy_ (cdr '(0)))
(define cr_ (cdr '(0)))
(define cre_ (cdr '(0)))
(define cki_ (cdr '(0)))
(define cty_ (cdr '(0)))
(define cid_ (cdr '(0)))

;; 滚道区 / 旋转器 / 踢球洞
(define lane_px '((136.0 52.0) (181.0 45.0) (226.0 52.0)))
(define spin_x 100.0)
(define spin_y 80.0)
(define spin_r 14.0)
(define hole_x 248.0)
(define hole_y 120.0)
(define hole_r 9.0)

;; ============================================================================
;; 小工具
;; ============================================================================

(define mklist
  (lambda (n v)
    (define r (cdr '(0)))
    (define i 0)
    (while (< i n) { (push r v) (set! i (+ i 1)) })
    r))

(define clamp
  (lambda (v lo hi)
    (define r v)
    (if (< r lo) (set! r lo))
    (if (> r hi) (set! r hi))
    r))

(define fmt_num
  (lambda (n width)
    (define s "")
    (define v (Math.floor n))
    (define k 0)
    (while (< k width)
      { (set! s (String.concat (String.charAt "0123456789" (Math.floor (mod v 10))) s))
        (set! v (Math.floor (/ v 10)))
        (set! k (+ k 1)) })
    s))

;; 十进制整数串（不用 atom_to_string：它对浮点用 %g，1234567 会变成 1.23457e+06）
(define fmt_int
  (lambda (n)
    (define v (Math.floor n))
    (define s "")
    (if (< v 0.5)
        (set! s "0")
        (while (> v 0.5)
          { (set! s (String.concat (String.charAt "0123456789" (Math.floor (mod v 10))) s))
            (set! v (Math.floor (/ v 10))) }))
    s))

(define fmt_g
  (lambda (n)
    (define s (fmt_int n))
    (define L (String.length s))
    (define out "")
    (define i 0)
    (while (< i L)
      { (if (and (> i 0) (== (mod (- L i) 3) 0))
            (set! out (String.concat out ",")))
        (set! out (String.concat out (String.charAt s i)))
        (set! i (+ i 1)) })
    out))

;; ============================================================================
;; 台面构建
;; ============================================================================

(define add_seg
  (lambda (x1 y1 x2 y2 rad rest kick typ ident)
    (define dx (- x2 x1))
    (define dy (- y2 y1))
    (define len2 (+ (* dx dx) (* dy dy)))
    (if (< len2 0.0001) (set! len2 0.0001))
    (push sx1 x1) (push sy1 y1) (push sx2 x2) (push sy2 y2)
    (push srad rad) (push srest rest) (push skick kick) (push styp typ)
    (push sinv (/ 1.0 len2))
    (push sact 1)
    (push sid ident)
    (set! nseg (+ nseg 1))
    (- nseg 1)))

(define add_circle
  (lambda (cx cy cr rest kick typ ident)
    (push cx_ cx) (push cy_ cy) (push cr_ cr)
    (push cre_ rest) (push cki_ kick) (push cty_ typ) (push cid_ ident)
    (set! ncir (+ ncir 1))
    (- ncir 1)))

(define build_table
  (lambda ()
    (define i 0)
    (define ang 0.0)
    (define px 0.0)
    (define py 0.0)
    (define ox 0.0)
    (define oy 0.0)

    ;; --- 机箱外框（左上/右上圆角用折线近似）
    (add_seg 68 240 68 44 2.0 0.35 0 0 0)
    (add_seg 68 44 74 26 2.0 0.35 0 0 0)
    (add_seg 74 26 88 14 2.0 0.35 0 0 0)
    (add_seg 88 14 108 6 2.0 0.35 0 0 0)
    (add_seg 108 6 132 3 2.0 0.35 0 0 0)
    (add_seg 132 3 252 3 2.0 0.35 0 0 0)
    (add_seg 252 3 278 6 2.0 0.35 0 0 0)
    (add_seg 278 6 298 14 2.0 0.35 0 0 0)
    (add_seg 298 14 310 26 2.0 0.35 0 0 0)
    (add_seg 310 26 316 44 2.0 0.35 0 0 0)
    (add_seg 316 44 316 240 2.0 0.35 0 0 0)

    ;; --- 台面左墙
    (add_seg F_L 240 F_L 112 2.0 0.45 0 0 0)

    ;; --- 穹顶（圆心 (181,130.6) 半径 106.5，从 170° 到 56°）
    (set! i 0)
    (while (<= i 10)
      (set! ang (* (- 170.0 (* i 11.4)) RAD))
      (set! px (+ DOME_CX (* DOME_R (Math.cos ang))))
      (set! py (- DOME_CY (* DOME_R (Math.sin ang))))
      (if (> i 0) (add_seg ox oy px py 2.0 0.45 0 0 0))
      (set! ox px)
      (set! oy py)
      (set! i (+ i 1)))

    ;; --- 台面右墙 / 发射道隔墙
    (add_seg F_R 240 F_R LANE_TOP 2.0 0.45 0 0 0)

    ;; --- 单向门（只在球向下运动时生效：防止球从台面掉回发射道）
    (set! seg_gate (add_seg LANE_L GATE_Y LANE_R GATE_Y 1.5 0.2 0 3 0))

    ;; --- 发射道顶部的导流斜板：把上行的球从"台面上方"折回台面右上角
    (add_seg 316 86 284 56 2.0 0.2 0 5 0)

    ;; --- 弹弓（三角形：竖边 + 底边是普通墙，斜面是踢击面）
    (add_seg 96 150 96 196 2.0 0.35 0 0 0)
    (add_seg 96 196 136 196 2.0 0.35 0 0 0)
    (add_seg 96 150 136 196 2.0 0.55 320 1 0)
    (add_seg 266 150 266 196 2.0 0.35 0 0 0)
    (add_seg 266 196 226 196 2.0 0.35 0 0 0)
    (add_seg 266 150 226 196 2.0 0.55 320 1 1)

    ;; --- 落靶（三块横板，球从上方压下去）
    (set! tg_seg (cdr '(0)))
    (set! i 0)
    (while (< i 3)
      { (push tg_seg (add_seg 84 (+ 106 (* i 14)) 106 (+ 106 (* i 14)) 2.0 0.35 0 2 i))
        (push tg_down 0)
        (set! i (+ i 1)) })

    ;; --- 弹射柱（三个圆）
    (add_circle 146.0 92.0 13.0 0.45 190 0 0)
    (add_circle 216.0 92.0 13.0 0.45 190 0 1)
    (add_circle 181.0 128.0 13.0 0.45 190 0 2)

    0))

;; 上面的立靶构造写得含糊，这里重写成明确的版本（构建时调用）
(define build_standups
  (lambda ()
    (add_circle 268.0 104.0 6.0 0.5 40 1 0)
    (add_circle 266.0 122.0 6.0 0.5 40 1 1)
    (add_circle 268.0 140.0 6.0 0.5 40 1 2)
    0))

(define build_stars
  (lambda ()
    (define i 0)
    (define x 0.0)
    (define y 0.0)
    (define dy 0.0)
    (define half 0.0)
    (define h2 0.0)
    (while (< i NSTAR)
      { ;; y 取 [28,232]：穹顶顶点在 y=24.1，取 28 以下可保证 dy² < R²（sqrt 参数恒正）
        (set! y (+ 28.0 (* 204.0 (Math.random))))
        (set! x 0.0)
        (if (< y DOME_Y)
            { (set! dy (- y DOME_CY))
              (set! h2 (- (* DOME_R DOME_R) (* dy dy)))
              (if (< h2 1.0) (set! h2 1.0))
              (set! half (Math.sqrt h2))
              (set! x (+ DOME_CX (* half (- (* 2.0 (Math.random)) 1.0))))
              (if (< x (+ F_L 4.0)) (set! x (+ F_L 4.0)))
              (if (> x (- F_R 4.0)) (set! x (- F_R 4.0))) }
            { (set! x (+ (+ F_L 4.0) (* (- (- F_R F_L) 8.0) (Math.random)))) })
        (push star_x x)
        (push star_y y)
        (push star_b (+ 40.0 (* 150.0 (Math.random))))
        (set! i (+ i 1)) })
    (set! i 0)
    (while (< i 8)
      { (push btrail_x -100.0) (push btrail_y -100.0) (set! i (+ i 1)) })
    (set! i 0)
    (while (< i NPART)
      { (push part_x -100.0) (push part_y -100.0) (push part_vx 0.0)
        (push part_vy 0.0) (push part_life 0.0) (push part_col 0)
        (set! i (+ i 1)) })
    (set! i 0)
    (while (< i NBUMP) { (push bumper_flash 0.0) (push bumper_cd 0.0) (set! i (+ i 1)) })
    (set! i 0)
    (while (< i 3) { (push su_lit 0) (push lane_lit 0) (push lane_in 0) (set! i (+ i 1)) })
    0))

;; ============================================================================
;; 粒子
;; ============================================================================

(define spawn_particles
  (lambda (x y n col)
    (define i 0)
    (define k part_next)
    (while (< i n)
      { (set_item! part_x k x)
        (set_item! part_y k y)
        (set_item! part_vx k (* 150.0 (- (* 2.0 (Math.random)) 1.0)))
        (set_item! part_vy k (* 150.0 (- (* 2.0 (Math.random)) 1.0)))
        (set_item! part_life k (+ 0.25 (* 0.35 (Math.random))))
        (set_item! part_col k col)
        (set! k (mod (+ k 1) NPART))
        (set! i (+ i 1)) })
    (set! part_next k)
    0))

(define update_particles
  (lambda (dt)
    (define i 0)
    (define l 0.0)
    (while (< i NPART)
      { (set! l (get_item part_life i))
        (if (> l 0.0)
            { (set! l (- l dt))
              (set_item! part_life i l)
              (set_item! part_x i (+ (get_item part_x i) (* (get_item part_vx i) dt)))
              (set_item! part_y i (+ (get_item part_y i) (* (get_item part_vy i) dt)))
              (set_item! part_vy i (+ (get_item part_vy i) (* 400.0 dt))) })
        (set! i (+ i 1)) })
    0))

;; ============================================================================
;; 计分 / 规则
;; ============================================================================

(define set_msg
  (lambda (s col dur)
    (set! msg_text s)
    (set! msg_color col)
    (set! msg_timer dur)
    0))

(define rank_threshold
  (lambda (i)
    (if (>= i 5) 999999999 (get_item RANK_PTS i))))

(define add_score
  (lambda (base)
    (set! score (+ score (* base mult)))
    (set! score_dirty 1)
    (while (and (< rankidx 4) (>= score (rank_threshold rankidx)))
      { (set! rankidx (+ rankidx 1))
        (set! score_dirty 1)
        (set_msg (String.concat "PROMOTED: " (get_item RANK_NAME rankidx)) 1 2.0)
        (display "[pinball] promo rank=")
        (display (get_item RANK_NAME rankidx))
        (newline) })
    0))

(define bumper_hit
  (lambda (i)
    (if (<= (get_item bumper_cd i) 0.0)
        { (set_item! bumper_cd i 0.10)
          (set_item! bumper_flash i 1.0)
          (add_score SC_BUMPER)
          (spawn_particles (get_item cx_ i) (get_item cy_ i) 5 1)
          })
    0))

(define sling_hit
  (lambda (side)
    ;; side 仅用于将来区分左右弹弓的视觉反馈，当前两侧共用同一个闪光计时
    (if (< side 0) (set! side 0))
    (add_score SC_SLING)
    (set! sling_flash 0.16)
    0))

(define target_hit
  (lambda (i)
    (define k (get_item sid (get_item tg_seg i)))
    (if (== (get_item tg_down k) 0)
        { (set_item! tg_down k 1)
          (set_item! sact (get_item tg_seg i) 0)
          (add_score SC_TARGET)
          (spawn_particles 95.0 (+ 106.0 (* k 14.0)) 5 2)
          (display "[pinball] target=") (display k) (newline)
          (if (and (== (get_item tg_down 0) 1) (== (get_item tg_down 1) 1)
                   (== (get_item tg_down 2) 1))
              { (add_score SC_BANK)
                (if (< mult 5) (set! mult (+ mult 1)))
                (set! mult_str (String.concat "X" (fmt_int mult)))
                (set! tg_timer 1.4)
                (set_msg "BANK CLEARED" 2 1.4)
                (display "[pinball] bank mult=") (display mult) (newline) }) })
    0))

(define standup_hit
  (lambda (i)
    (if (== (get_item su_lit i) 0)
        { (set_item! su_lit i 1)
          (add_score SC_STANDUP)
          (spawn_particles (get_item cx_ (+ 3 i)) (get_item cy_ (+ 3 i)) 4 3)
          (if (and (== (get_item su_lit 0) 1) (== (get_item su_lit 1) 1)
                   (== (get_item su_lit 2) 1))
              { (add_score SC_MISSION)
                (set_msg "MISSION TARGETS" 2 1.4)
                (set! su_lit (mklist 3 0))
                (display "[pinball] mission")(newline) }) })
    0))

(define do_drain
  (lambda ()
    ;; ball save：发球后 7 秒内掉球不扣球（每球一次）
    (if (and (< play_t 7.0) (== save_used 0))
        { (set! save_used 1)
          (serve_ball)
          (set_msg "BALL SAVED" 2 1.4)
          (display "[pinball] save")(newline) }
        { (do_drain_real) })
    0))

(define do_drain_real
  (lambda ()
    (set! bst 3)
    (set! btimer 1.1)
    (set! bvx 0.0)
    (set! bvy 0.0)
    (set! charging 0)
    (set! charge 0.0)
    (set! fl_on 0)
    (set! fr_on 0)
    (clear_trail)
    (set! balls (- balls 1))
    (set! save_used 0)
    (set! score_dirty 1)
    (display "[pinball] drain ball=") (display balls)
    (display " score=") (display score)
    (display " bx=") (display (Math.floor bx)) (newline)
    (if (<= balls 0)
        { (set! bst 4)
          (set! game_over 1)
          (set! msg_timer 0.0)
          (display "[pinball] over score=") (display score) (newline) }
        { (set_msg "BALL LOST" 4 1.2) })
    0))

(define clear_trail
  (lambda ()
    (define i 0)
    (while (< i 8)
      { (set_item! btrail_x i -100.0) (set_item! btrail_y i -100.0) (set! i (+ i 1)) })
    0))

(define serve_ball
  (lambda ()
    (clear_trail)
    (set! bx 303.0)
    (set! by 228.0)
    (set! bvx 0.0)
    (set! bvy 0.0)
    (set! bst 0)
    (set! left_lane 0)
    (set! charge 0.0)
    (set! charging 0)
    (set! fl_on 0)
    (set! fr_on 0)
    0))

(define new_ball
  (lambda ()
    (set! mult 1)
    (set! mult_str "X1")
    (set! save_used 0)
    (set! su_lit (mklist 3 0))
    (set! lane_lit (mklist 3 0))
    (set_targets_up)
    (serve_ball)
    0))

(define set_targets_up
  (lambda ()
    (define i 0)
    (while (< i 3)
      { (set_item! tg_down i 0)
        (set_item! sact (get_item tg_seg i) 1)
        (set! i (+ i 1)) })
    0))

(define reset_game
  (lambda ()
    (set! score 0)
    (set! balls 3)
    (set! mult 1)
    (set! mult_str "X1")
    (set! rankidx 0)
    (set! game_over 0)
    (set! hole_hits 0)
    (set! extra_given 0)
    (set! score_dirty 1)
    (set_targets_up)
    (set! su_lit (mklist 3 0))
    (set! lane_lit (mklist 3 0))
    (new_ball)
    (set_msg "LAUNCH BALL" 1 1.6)
    0))

;; ============================================================================
;; 物理
;; ============================================================================

(define clamp_ball
  (lambda ()
    (define sp (Math.sqrt (+ (* bvx bvx) (* bvy bvy))))
    (if (> sp MAXV)
        { (set! bvx (* bvx (/ MAXV sp)))
          (set! bvy (* bvy (/ MAXV sp))) })
    0))

(define flip_hit
  (lambda (pvx pvy ang omega)
    (define tx 0.0)
    (define ty 0.0)
    (define dx 0.0)
    (define dy 0.0)
    (define inv 1.0)
    (define t 0.0)
    (define cx 0.0)
    (define cy 0.0)
    (define nx 0.0)
    (define ny 0.0)
    (define rr 0.0)
    (define d2 0.0)
    (define dist 1.0)
    (define vsx 0.0)
    (define vsy 0.0)
    (define vn 0.0)
    (set! tx (+ pvx (* FLIP_LEN (Math.cos ang))))
    (set! ty (+ pvy (* FLIP_LEN (Math.sin ang))))
    (set! dx (- tx pvx))
    (set! dy (- ty pvy))
    (set! inv (/ 1.0 (+ (* dx dx) (* dy dy))))
    (set! t (* (+ (* (- bx pvx) dx) (* (- by pvy) dy)) inv))
    (if (< t 0.0) (set! t 0.0))
    (if (> t 1.0) (set! t 1.0))
    (set! cx (+ pvx (* t dx)))
    (set! cy (+ pvy (* t dy)))
    (set! nx (- bx cx))
    (set! ny (- by cy))
    (set! rr (+ BALL_R FLIP_RAD))
    (set! d2 (+ (* nx nx) (* ny ny)))
    (if (< d2 (* rr rr))
        { (set! dist (Math.sqrt d2))
          (if (< dist 0.001)
              { (set! nx 0.0) (set! ny -1.0) (set! dist 1.0) }
              { (set! nx (/ nx dist)) (set! ny (/ ny dist)) })
          (set! bx (+ cx (* nx rr)))
          (set! by (+ cy (* ny rr)))
          ;; 接触点表面速度 v = ω × r（屏幕坐标 y 向下，ω 顺正）
          (set! vsx (* omega (- pvy cy)))
          (set! vsy (* omega (- cx pvx)))
          (set! vn (+ (* (- bvx vsx) nx) (* (- bvy vsy) ny)))
          (if (< vn 0.0)
              { (set! vn (* 1.55 vn))
                (set! bvx (- bvx (* vn nx)))
                (set! bvy (- bvy (* vn ny))) })
          (clamp_ball) })
    0))

(define phys_step
  (lambda (h)
    ;; 局部化（热循环只读槽位，不读模块级自由变量）
    (define px 0.0)
    (define py 0.0)
    (define vx 0.0)
    (define vy 0.0)
    (define i 0)
    (define n nseg)
    (define x1 0.0) (define y1 0.0) (define dx 0.0) (define dy 0.0)
    (define inv 1.0) (define t 0.0) (define ccx 0.0) (define ccy 0.0)
    (define nx 0.0) (define ny 0.0) (define d2 0.0) (define dist 1.0)
    (define rr 0.0) (define vn 0.0) (define e 0.0) (define kick 0.0)
    (define a_x1 sx1) (define a_y1 sy1) (define a_x2 sx2) (define a_y2 sy2)
    (define a_r srad) (define a_re srest) (define a_k skick)
    (define a_ty styp) (define a_ac sact) (define a_iv sinv)
    (define nc ncir)
    (define a_cx cx_) (define a_cy cy_) (define a_cr cr_)
    (define a_cre cre_) (define a_cki cki_) (define a_cty cty_)

    (set! px bx)
    (set! py by)
    (set! vx (* bvx (- 1.0 (* 0.12 h))))
    (set! vy (+ (* bvy (- 1.0 (* 0.12 h))) (* GRAV h)))
    (set! px (+ px (* vx h)))
    (set! py (+ py (* vy h)))

    ;; ---- 单向门：只在球向下运动时激活
    (if (> vy 0.0)
        (set_item! a_ac seg_gate 1)
        (set_item! a_ac seg_gate 0))

    ;; ---- 线段
    (set! i 0)
    (while (< i n)
      (if (== (get_item a_ac i) 1)
          { (set! x1 (get_item a_x1 i))
            (set! y1 (get_item a_y1 i))
            (set! dx (- (get_item a_x2 i) x1))
            (set! dy (- (get_item a_y2 i) y1))
            (set! inv (get_item a_iv i))
            (set! t (* (+ (* (- px x1) dx) (* (- py y1) dy)) inv))
            (if (< t 0.0) (set! t 0.0))
            (if (> t 1.0) (set! t 1.0))
            (set! ccx (+ x1 (* t dx)))
            (set! ccy (+ y1 (* t dy)))
            (set! nx (- px ccx))
            (set! ny (- py ccy))
            (set! rr (+ BALL_R (get_item a_r i)))
            (set! d2 (+ (* nx nx) (* ny ny)))
            (if (< d2 (* rr rr))
                { (set! dist (Math.sqrt d2))
                  (if (< dist 0.001)
                      { (set! nx 0.0) (set! ny -1.0) (set! dist 1.0) }
                      { (set! nx (/ nx dist)) (set! ny (/ ny dist)) })
                  (set! px (+ ccx (* nx rr)))
                  (set! py (+ ccy (* ny rr)))
                  (set! vn (+ (* vx nx) (* vy ny)))
                  (if (< vn 0.0)
                      { (set! e (get_item a_re i))
                        (set! vn (* (* (+ 1.0 e) vn) 1.0))
                        (set! vx (- vx (* vn nx)))
                        (set! vy (- vy (* vn ny))) })
                  (set! kick (get_item a_k i))
                  (if (> kick 0.0)
                      { (set! vx (+ vx (* nx kick)))
                        (set! vy (+ vy (* ny kick)))
                        (if (== (get_item a_ty i) 1) (sling_hit (get_item sid i))) })
                  (if (== (get_item a_ty i) 2)
                      (target_hit (get_item sid i))) })
          })
      (set! i (+ i 1)))

    ;; ---- 圆
    (set! i 0)
    (while (< i nc)
      { (set! nx (- px (get_item a_cx i)))
        (set! ny (- py (get_item a_cy i)))
        (set! rr (+ BALL_R (get_item a_cr i)))
        (set! d2 (+ (* nx nx) (* ny ny)))
        (if (< d2 (* rr rr))
            { (set! dist (Math.sqrt d2))
              (if (< dist 0.001)
                  { (set! nx 0.0) (set! ny -1.0) (set! dist 1.0) }
                  { (set! nx (/ nx dist)) (set! ny (/ ny dist)) })
              (set! px (+ (get_item a_cx i) (* nx rr)))
              (set! py (+ (get_item a_cy i) (* ny rr)))
              (set! vn (+ (* vx nx) (* vy ny)))
              (if (< vn 0.0)
                  { (set! vn (* (* (+ 1.0 (get_item a_cre i)) vn) 1.0))
                    (set! vx (- vx (* vn nx)))
                    (set! vy (- vy (* vn ny))) })
              (set! kick (get_item a_cki i))
              (set! vx (+ (+ vx (* nx kick)) (* (* (* -1.0 ny) 70.0) (- (* 2.0 (Math.random)) 1.0))))
              (set! vy (+ (+ vy (* ny kick)) (* (* nx 70.0) (- (* 2.0 (Math.random)) 1.0))))
              (if (== (get_item a_cty i) 0)
                  (bumper_hit (get_item cid_ i))
                  (standup_hit (get_item cid_ i))) })
        (set! i (+ i 1)) })

    ;; ---- 挡板（flip_hit 直接读写模块级 bx/by/bvx/bvy，故先回写再取回）
    (set! bx px) (set! by py) (set! bvx vx) (set! bvy vy)
    (flip_hit PIV_LX PIV_LY fl_ang fl_omega)
    (flip_hit PIV_RX PIV_RY fr_ang fr_omega)
    (set! px bx) (set! py by) (set! vx bvx) (set! vy bvy)

    ;; ---- 速度上限
    (set! d2 (+ (* vx vx) (* vy vy)))
    (if (> d2 (* MAXV MAXV))
        { (set! dist (Math.sqrt d2))
          (set! vx (* vx (/ MAXV dist)))
          (set! vy (* vy (/ MAXV dist))) })

    (set! bx px)
    (set! by py)
    (set! bvx vx)
    (set! bvy vy)
    0))

(define update_flippers
  (lambda (dt)
    (define na fl_ang)
    (if (== fl_on 1)
        { (set! na (- fl_ang (* FLIP_UP_SPD dt)))
          (if (< na UP_L) (set! na UP_L)) }
        { (set! na (+ fl_ang (* FLIP_DN_SPD dt)))
          (if (> na REST_L) (set! na REST_L)) })
    (set! fl_omega (/ (- na fl_ang) dt))
    (set! fl_ang na)
    (set! na fr_ang)
    (if (== fr_on 1)
        { (set! na (+ fr_ang (* FLIP_UP_SPD dt)))
          (if (> na UP_R) (set! na UP_R)) }
        { (set! na (- fr_ang (* FLIP_DN_SPD dt)))
          (if (< na REST_R) (set! na REST_R)) })
    (set! fr_omega (/ (- na fr_ang) dt))
    (set! fr_ang na)
    0))

;; ============================================================================
;; 触发区
;; ============================================================================

(define zone_dist
  (lambda (zx zy)
    (define dx (- bx zx))
    (define dy (- by zy))
    (Math.sqrt (+ (* dx dx) (* dy dy)))))

(define check_triggers
  (lambda (dt)
    (define i 0)
    (define d 0.0)
    (define p (cdr '(0)))
    (define alllit 1)

    ;; 旋转器
    (set! spin_phase (+ spin_phase (* dt (+ 6.0 spin_count))))
    (if (> spin_count 0) (set! spin_count (- spin_count (* dt 3.0))))
    (if (< spin_count 0.0) (set! spin_count 0.0))
    (if (< spin_cd 0.0) (set! spin_cd 0.0))
    (set! spin_cd (- spin_cd dt))
    (if (<= (zone_dist spin_x spin_y) spin_r)
        (if (<= spin_cd 0.0)
            { (set! spin_cd 0.16)
              (set! spin_count (+ spin_count 3.0))
              (add_score SC_SPIN)
              (spawn_particles spin_x spin_y 2 4) }))
    (if (< spin_phase (* 2.0 PI2)) (set! spin_phase 0.0))

    ;; 顶部滚道
    (set! i 0)
    (while (< i 3)
      { (set! p (get_item lane_px i))
        (set! d (zone_dist (get_item p 0) (get_item p 1)))
        (if (and (< d 12.0) (== (get_item lane_in i) 0))
            { (set_item! lane_in i 1)
              (if (== (get_item lane_lit i) 0)
                  { (set_item! lane_lit i 1)
                    (add_score SC_LANE) }) })
        (if (>= d 13.0) (set_item! lane_in i 0))
        (set! i (+ i 1)) })
    (set! i 0)
    (set! alllit 1)
    (while (< i 3)
      { (if (== (get_item lane_lit i) 0) (set! alllit 0))
        (set! i (+ i 1)) })
    (if (== alllit 1)
        { (add_score SC_LANES)
          (set_msg "LANES COMPLETE" 2 1.6)
          (set! lane_lit (mklist 3 0))
          (if (== extra_given 0)
              { (set! extra_given 1)
                (set! balls (+ balls 1))
                (set_msg "EXTRA BALL" 6 2.0)
                (display "[pinball] extraball")(newline) }) })

    ;; 踢球洞
    (if (< hole_cd 0.0) (set! hole_cd 0.0))
    (set! hole_cd (- hole_cd dt))
    (if (and (== bst 1) (<= (zone_dist hole_x hole_y) hole_r) (<= hole_cd 0.0))
        { (set! bst 2)
          (set! btimer 0.55)
          (set! bvx 0.0) (set! bvy 0.0)
          (set! hole_hits (+ hole_hits 1))
          (add_score (+ SC_HOLE (* 3000 (- (if (> hole_hits 3) 3 hole_hits) 1))))
          (set_msg "JACKPOT" 6 1.2)
          (spawn_particles hole_x hole_y 8 5)
          (display "[pinball] hole hits=") (display hole_hits) (newline) })

    ;; 掉球
    (if (and (== bst 1) (> by DRAIN_Y)) (do_drain))
    0))

;; ============================================================================
;; 输入
;; ============================================================================

(define input_update
  (lambda (dt)
    (define touched (Touch.is_pressed))
    (define tx (Touch.x))
    (define t_l #f)
    (define t_r #f)
    (define kl #f)
    (define kr #f)
    (define kspace #f)

    (if (or (Key.is_down 90) (Key.is_down 122) (Key.is_down 65) (Key.is_down 97)
            (Key.is_down 3))
        (set! kl #t))
    (if (or (Key.is_down 88) (Key.is_down 120) (Key.is_down 47) (Key.is_down 4))
        (set! kr #t))
    (if (or (Key.is_down 32) (Key.is_down 13))
        (set! kspace #t))

    ;; 触屏三分区：左 1/3 = 左挡板，中间 1/3 = 两枚同时，右 1/3 = 右挡板。
    ;; 本平台只有单触点（不能两指同按），中间的"双拍区"是一根手指也能同时抬两枚的入口。
    (if touched
        (if (< tx ZONE_L)
            (set! t_l #t)
            (if (> tx ZONE_R)
                (set! t_r #t)
                { (set! t_l #t) (set! t_r #t) })))
    (if kl (set! t_l #t))
    (if kr (set! t_r #t))

    ;; 连拍宽限：松手后挡板**继续抬起 FLIP_GRACE 秒**再落。单指快速"左—右"两下时，
    ;; 左手还抬着右手就接上了——效果等于同时抬两枚（真实弹球里双板同抬也就几十毫秒）。
    ;; 代价：普通单拍的回位晚 ~0.12s（回位 13rad/s，观感是"落得稍缓"，不影响接球）。
    (if t_l (set! l_latch FLIP_GRACE) (set! l_latch (- l_latch dt)))
    (if t_r (set! r_latch FLIP_GRACE) (set! r_latch (- r_latch dt)))
    (if (< l_latch 0.0) (set! l_latch 0.0))
    (if (< r_latch 0.0) (set! r_latch 0.0))
    (if (or t_l (> l_latch 0.0)) (set! fl_on 1) (set! fl_on 0))
    (if (or t_r (> r_latch 0.0)) (set! fr_on 1) (set! fr_on 0))

    ;; 发射：球停在发射道里时，任意触点/空格蓄力，松手弹出；闲置 8s 自动满力发射
    (if (== bst 0)
        { (if (or touched kspace)
              { (set! charging 1)
                (set! charge (+ charge (* 1.5 dt)))
                (if (> charge 1.0) (set! charge 1.0)) }
              { (if (== charging 1) (launch_ball charge)) })
          (set! idle_t (+ idle_t dt))
          (if (and (> idle_t 8.0) (== charging 0))
              { (set_msg "AUTO LAUNCH" 1 1.2)
                (launch_ball 0.9) }) }
        (set! idle_t 0.0))

    ;; 挡板电平变化时打一行（端到端回归据此断言"三分区/连拍"的语义）
    (if (or (not (== fl_on last_fl)) (not (== fr_on last_fr)))
        { (set! last_fl fl_on)
          (set! last_fr fr_on)
          (display "[pinball] flip ") (display fl_on) (display fr_on) (newline) })

    (if (== game_over 1)
        (if (or touched kspace kl kr)
            { (display "[pinball] restart")(newline) (reset_game) }))
    0))

(define launch_ball
  (lambda (ch)
    (set! charging 0)
    (set! bvy (- 0.0 (+ 470.0 (* 320.0 ch))))
    (set! bvx -8.0)
    (set! bst 1)
    (set! left_lane 0)
    (set! play_t 0.0)
    (set! idle_t 0.0)
    (set! msg_timer 0.0)
    (set! charge 0.0)
    (display "[pinball] launch charge=")
    (display (Math.to_fixed ch 2)) (newline)
    0))

;; ============================================================================
;; 绘制：调色板与小组件
;; ============================================================================

(define txt
  (lambda (s x y r g b)
    (Gfx.set_font 1)
    (Gfx.draw_text s x y r g b)
    0))

;; 12px 抗锯齿字模（Gfx.set_font 1，步进 7px）。原先的 12px 二值点阵（font 0，
;; 步进 6px）在深色霓虹底上笔画发硬，故全台面统一用平滑字模；步进 6→7px 后
;; 面板 8 位数刚好 56px（x=4..60），提示语一律 ≤8 字符。
(define txtb
  (lambda (s x y r g b)
    (Gfx.set_font 1)
    (Gfx.draw_text s x y r g b)
    0))

(define txtc
  (lambda (s cx y r g b)
    (define w (* (String.length s) 7))
    (txt s (- cx (Math.floor (/ w 2))) y r g b)))

(define txt16
  (lambda (s cx y r g b)
    (define w (* (String.length s) 9))
    (Gfx.set_font 2)
    (Gfx.draw_text s (- cx (Math.floor (/ w 2))) y r g b)
    0))

(define line
  (lambda (x1 y1 x2 y2 w r g b)
    (Gfx.draw_line_aa x1 y1 x2 y2 w r g b)
    0))

;; ============================================================================
;; 绘制：背景
;; ============================================================================

(define draw_panel
  (lambda ()
    (define i 0)
    (define t 0.0)
    (define r 0) (define g 0) (define b 0)
    ;; 面板底：自上而下由深紫到近黑
    (set! i 0)
    (while (< i 24)
      { (set! t (/ i 23.0))
        (set! r (Math.floor (+ 22.0 (* -12.0 t))))
        (set! g (Math.floor (+ 12.0 (* -8.0 t))))
        (set! b (Math.floor (+ 52.0 (* -34.0 t))))
        (Gfx.fill_rect 0 (* i 10) 68 10 r g b)
        (set! i (+ i 1)) })
    ;; 分隔线
    (Gfx.fill_rect 66 0 2 240 40 70 130)
    (Gfx.fill_rect 0 0 66 1 40 70 130)
    ;; 装饰：三颗小星
    (Gfx.set_pixel 12 6 255 255 255)
    (Gfx.set_pixel 54 10 200 220 255)
    (Gfx.set_pixel 30 4 180 200 255)
    0))

(define draw_logo
  (lambda ()
    (define yy 6)
    (txt16 "SPACE" 34 (+ yy 1) 120 40 160)
    (txt16 "SPACE" 33 yy 150 240 255)
    (txt16 "CADET" 34 (+ yy 20) 120 40 160)
    (txt16 "CADET" 33 (+ yy 19) 255 120 210)
    (txt "PINBALL" 4 (+ yy 42) 110 160 210)
    0))

(define draw_field
  (lambda ()
    (define y 0)
    (define yc 0.0)
    (define xl 0.0)
    (define xr 0.0)
    (define h2 0.0)
    (define half 0.0)
    (define t 0.0)
    (define r 0) (define g 0) (define b 0)
    (define i 0)
    ;; 机箱底
    (Gfx.fill_rect 68 0 248 240 8 6 20)
    ;; 台面：按 8px 横条填充，上缘按穹顶圆裁形，同时做垂直渐变
    (set! y 4)
    (while (< y 240)
      { (set! yc (+ y 2))
        (if (>= yc DOME_Y)
            { (set! xl F_L) (set! xr F_R) }
            { (set! h2 (- (* DOME_R DOME_R) (* (- yc DOME_CY) (- yc DOME_CY))))
              (if (< h2 0.0) (set! h2 0.0))
              (set! half (Math.sqrt h2))
              (set! xl (- DOME_CX half))
              (set! xr (+ DOME_CX half)) })
        (if (> (- xr xl) 1.0)
            { (set! t (/ yc 240.0))
              (set! r (Math.floor (+ 27.0 (* -15.0 t))))
              (set! g (Math.floor (+ 19.0 (* -11.0 t))))
              (set! b (Math.floor (+ 66.0 (* -34.0 t))))
              (Gfx.fill_rect (Math.floor xl) y (Math.floor (- xr xl)) 4 r g b) })
        (set! y (+ y 4)) })
    ;; 发射道
    (Gfx.fill_rect LANE_L LANE_TOP (- LANE_R LANE_L) (- 240 LANE_TOP) 12 9 30)
    ;; 星野
    (set! i 0)
    (while (< i NSTAR)
      { (Gfx.set_pixel (Math.floor (get_item star_x i)) (Math.floor (get_item star_y i))
                       (Math.floor (get_item star_b i))
                       (Math.floor (get_item star_b i))
                       (Math.floor (+ 30.0 (* 0.9 (get_item star_b i)))))
        (set! i (+ i 1)) })
    ;; 台面装饰：大圆环（"轨道"）与底部星球
    (Gfx.draw_circle 181 196 58 34 52 118)
    (Gfx.draw_circle 181 196 50 24 38 86)
    (Gfx.fill_circle 181 210 32 16 12 44)
    (Gfx.fill_circle 181 210 28 22 16 58)
    (Gfx.fill_circle 181 208 20 28 20 72)
    (Gfx.draw_circle 181 210 32 72 116 210)
    (Gfx.draw_circle 181 210 26 40 70 140)
    (Gfx.draw_circle 181 210 18 30 52 104)
    ;; 穹顶内的"雷达"同心弧（跟随穹顶曲率，给上半场一点纵深）
    (Gfx.draw_circle 181 130 92 26 36 78)
    (Gfx.draw_circle 181 130 78 24 33 72)
    (Gfx.draw_circle 181 130 64 22 30 66)
    ;; 中央霓虹环（弹射柱群外圈）
    (Gfx.draw_circle 181 106 46 30 44 92)
    ;; 落靶区底板
    (Gfx.fill_rect 80 98 30 48 14 10 34)
    (Gfx.draw_rect 80 98 30 48 40 60 110)
    ;; 台面文字与箭头装饰（12px 抗锯齿字模）
    (Gfx.set_font 1)
    (Gfx.draw_text "JACKPOT" 200 144 120 90 60)
    (Gfx.draw_text "BONUS" 86 150 90 120 170)
    (Gfx.draw_text "RANK" 160 60 90 120 170)
    (Gfx.set_font 1)
    0))

(define draw_walls
  (lambda ()
    (define i 0)
    (define ox 0.0) (define oy 0.0)
    (define px 0.0) (define py 0.0)
    ;; 机箱描边
    (line 68 44 68 240 2 50 80 150)
    (line 68 44 74 26 2 50 80 150)
    (line 74 26 88 14 2 50 80 150)
    (line 88 14 108 6 2 50 80 150)
    (line 108 6 132 3 2 50 80 150)
    (line 132 3 252 3 2 50 80 150)
    (line 252 3 278 6 2 50 80 150)
    (line 278 6 298 14 2 50 80 150)
    (line 298 14 310 26 2 50 80 150)
    (line 310 26 316 44 2 50 80 150)
    (line 316 44 316 240 2 50 80 150)
    ;; 台面左墙
    (line F_L 240 F_L 112 2 110 200 255)
    ;; 穹顶
    (set! i 0)
    (set! ox 0.0) (set! oy 0.0)
    (while (<= i 10)
      { (set! px (+ DOME_CX (* DOME_R (Math.cos (* (- 170.0 (* i 11.4)) RAD)))))
        (set! py (- DOME_CY (* DOME_R (Math.sin (* (- 170.0 (* i 11.4)) RAD)))))
        (if (> i 0) (line ox oy px py 2 110 200 255))
        (set! ox px) (set! oy py)
        (set! i (+ i 1)) })
    ;; 台面右墙 / 发射道隔墙
    (line F_R 240 F_R LANE_TOP 2 110 200 255)
    (line LANE_L GATE_Y LANE_L 240 1 40 70 120)
    ;; 单向门（画成虚线）
    (set! i 0)
    (while (< i 4)
      { (line (+ LANE_L (* i 7)) GATE_Y (+ (+ LANE_L 4) (* i 7)) GATE_Y 1 60 120 160)
        (set! i (+ i 1)) })
    ;; 发射道顶部的导流斜板
    (line LANE_R 86 284 56 3 110 200 255)
    (line LANE_R 86 284 56 1 220 250 255)
    ;; 挡板下的排水口
    (Gfx.fill_rect 150 232 62 8 4 3 10)
    0))

;; ============================================================================
;; 绘制：台面元素
;; ============================================================================

(define draw_targets
  (lambda ()
    (define i 0)
    (define y 0)
    (while (< i 3)
      { (set! y (+ 106 (* i 14)))
        (if (== (get_item tg_down i) 1)
            { (Gfx.fill_rect 86 (+ y 1) 18 3 30 30 44)
              (Gfx.draw_rect 86 (+ y 1) 18 3 60 60 90) }
            { (Gfx.fill_rect 85 (- y 2) 20 5 40 190 120)
              (Gfx.fill_rect 86 (- y 1) 18 3 120 255 170)
              (Gfx.draw_rect 85 (- y 2) 20 5 200 255 220) })
        (set! i (+ i 1)) })
    0))

(define draw_sling
  (lambda ()
    (define a 0) (define b 0) (define c 0)
    (set! a (if (> sling_flash 0.0) 1 0))
    ;; 左
    (Gfx.draw_triangle 96 150 96 196 136 196 0 90 40 130 1)
    (line 96 150 136 196 3 (if (== a 1) 255 200) (if (== a 1) 240 80) 120)
    (line 96 150 96 196 2 120 60 170)
    (line 96 196 136 196 2 120 60 170)
    ;; 右
    (Gfx.draw_triangle 266 150 266 196 226 196 0 90 40 130 1)
    (line 266 150 226 196 3 (if (== a 1) 255 200) (if (== a 1) 240 80) 120)
    (line 266 150 266 196 2 120 60 170)
    (line 266 196 226 196 2 120 60 170)
    0))

(define draw_spinner
  (lambda ()
    (define dx (* 13.0 (Math.cos spin_phase)))
    (define dy (* 4.0 (Math.sin spin_phase)))
    (Gfx.draw_circle spin_x spin_y 14 50 60 120)
    (line (- spin_x dx) (- spin_y dy) (+ spin_x dx) (+ spin_y dy) 3 255 170 60)
    (line (- spin_x (* dx 0.5)) (- spin_y (* dy 0.5)) (+ spin_x (* dx 0.5)) (+ spin_y (* dy 0.5)) 1 255 240 180)
    0))

(define draw_hole
  (lambda ()
    (Gfx.fill_circle_aa hole_x hole_y 10 8 6 20 1)
    (Gfx.draw_circle hole_x hole_y 10 120 200 255)
    (Gfx.draw_circle hole_x hole_y 7 60 110 180)
    (if (> hole_cd 0.0)
        (Gfx.fill_circle hole_x hole_y 5 80 140 200))
    0))

(define draw_lanes
  (lambda ()
    (define i 0)
    (define p (cdr '(0)))
    (while (< i 3)
      { (set! p (get_item lane_px i))
        (if (== (get_item lane_lit i) 1)
            { (Gfx.fill_circle (get_item p 0) (get_item p 1) 6 90 220 255)
              (Gfx.draw_circle (get_item p 0) (get_item p 1) 8 150 240 255) }
            { (Gfx.fill_circle (get_item p 0) (get_item p 1) 6 30 34 70)
              (Gfx.draw_circle (get_item p 0) (get_item p 1) 8 60 90 150) })
        (set! i (+ i 1)) })
    0))

(define draw_bumpers
  (lambda ()
    (define i 0)
    (define f 0.0)
    (define cx 0.0)
    (define cy 0.0)
    (define k 0)
    (while (< i NBUMP)
      { (set! cx (get_item cx_ i))
        (set! cy (get_item cy_ i))
        (set! f (get_item bumper_flash i))
        ;; 光晕
        (Gfx.draw_circle cx cy (+ 17 (* 3 f)) (Math.floor (+ 40.0 (* 120.0 f))) 20 60)
        ;; 外圈
        (Gfx.draw_circle cx cy 14 (Math.floor (+ 150.0 (* 105.0 f))) 70 170)
        ;; 本体
        (Gfx.fill_circle_aa cx cy 12 (Math.floor (+ 96.0 (* 150.0 f)))
                            (Math.floor (+ 36.0 (* 90.0 f)))
                            (Math.floor (+ 118.0 (* 90.0 f))) 1)
        ;; 内环与中心
        (Gfx.draw_circle cx cy 8 (Math.floor (+ 200.0 (* 55.0 f))) 120 220)
        (Gfx.fill_circle cx cy 4 (Math.floor (+ 190.0 (* 65.0 f)))
                         (Math.floor (+ 130.0 (* 120.0 f))) 80)
        (Gfx.fill_circle (- cx 1) (- cy 1) 2 255 250 225)
        (set! i (+ i 1)) })
    ;; 立靶
    (set! i 0)
    (while (< i NSU)
      { (set! cx (get_item cx_ (+ NBUMP i)))
        (set! cy (get_item cy_ (+ NBUMP i)))
        (if (== (get_item su_lit i) 1)
            { (Gfx.draw_circle cx cy 9 120 90 40)
              (Gfx.fill_circle_aa cx cy 6 255 210 80 1)
              (Gfx.fill_circle cx cy 3 255 250 200) }
            { (Gfx.draw_circle cx cy 9 50 60 110)
              (Gfx.fill_circle_aa cx cy 6 56 58 92 1)
              (Gfx.draw_circle cx cy 4 110 120 170) })
        (set! i (+ i 1)) })
    0))

(define flip_draw
  (lambda (px py ang)
    (define tx (+ px (* FLIP_LEN (Math.cos ang))))
    (define ty (+ py (* FLIP_LEN (Math.sin ang))))
    (line px py tx ty 11 60 16 30)
    (line px py tx ty 8 200 70 30)
    (line px py tx ty 5 255 140 60)
    (line px py tx ty 2 255 225 180)
    (Gfx.fill_circle_aa px py 5 255 190 120 1)
    (Gfx.fill_circle_aa tx ty 3 255 210 150 1)
    0))

(define draw_flippers
  (lambda ()
    (flip_draw PIV_LX PIV_LY fl_ang)
    (flip_draw PIV_RX PIV_RY fr_ang)
    0))

(define draw_ball
  (lambda ()
    (define i 0)
    (define k 0)
    ;; 拖尾
    (set! i 1)
    (while (< i 5)
      { (set! k (mod (+ btrail_head (- 5 i)) 8))
        (if (> (get_item btrail_x k) -50.0)
            (Gfx.fill_circle (get_item btrail_x k) (get_item btrail_y k)
                             (- 4 i) (+ 30 (* i 12)) (+ 40 (* i 14)) (+ 60 (* i 18))))
        (set! i (+ i 1)) })
    (if (< bst 2)
        { (Gfx.fill_circle_aa (+ bx 1) (+ by 2) BALL_R 0 0 0 1)
          (Gfx.fill_circle_aa bx by BALL_R 220 225 240 1)
          (Gfx.fill_circle_aa bx by 3 245 250 255 1)
          (Gfx.fill_circle (- bx 1.5) (- by 1.5) 1.6 255 255 255) })
    0))

(define draw_particles
  (lambda ()
    (define i 0)
    (define l 0.0)
    (define c 0)
    (while (< i NPART)
      { (set! l (get_item part_life i))
        (if (> l 0.0)
            { (set! c (get_item part_col i))
              (if (== c 1) (Gfx.fill_circle (get_item part_x i) (get_item part_y i) 2 255 200 120)
                  (if (== c 2) (Gfx.fill_circle (get_item part_x i) (get_item part_y i) 2 120 255 170)
                      (if (== c 3) (Gfx.fill_circle (get_item part_x i) (get_item part_y i) 2 255 220 120)
                          (if (== c 4) (Gfx.fill_circle (get_item part_x i) (get_item part_y i) 2 255 160 80)
                              (Gfx.fill_circle (get_item part_x i) (get_item part_y i) 2 150 220 255))))) })
        (set! i (+ i 1)) })
    0))

(define draw_plunger
  (lambda ()
    (define h 0.0)
    (define i 0)
    (set! h (* 14.0 charge))
    (if (== bst 0)
        { (line 296 (- 232 h) 310 (- 232 h) 4 200 200 220)
          (set! i 0)
          (while (< i 4)
            { (line 297 (- (- 234 (* i 3)) h) 309 (- (- 234 (* i 3)) h) 1 90 90 130)
              (set! i (+ i 1)) }) }
        { (line 296 232 310 232 3 90 90 130) })
    0))

;; ============================================================================
;; 绘制：HUD
;; ============================================================================

(define draw_hud
  (lambda ()
    ;; 分数
    (txt "SCORE" 4 62 110 150 200)
    (if (== score_dirty 1)
        { (set! score_str (fmt_num score 8))
          (set! ball_str (String.atom_to_string (Math.floor balls)))
          (set! score_dirty 0) })
    (txtb score_str 4 74 255 230 90)
    ;; 球数
    (txt "BALL" 4 92 110 150 200)
    (txtb ball_str 40 92 255 255 255)
    ;; 倍数
    (txt "MULT" 4 106 110 150 200)
    (txtb mult_str 40 106 120 250 160)
    ;; 军衔阶梯
    (define i 0)
    (define yy 0)
    (set! i 0)
    (while (< i 5)
      { (set! yy (+ 118 (* i 12)))
        (if (<= i rankidx)
            { (Gfx.fill_circle_aa 8 (+ yy 4) 3 120 250 160 1)
              (txtb (get_item RANK_NAME i) 15 yy 180 255 200) }
            { (Gfx.fill_circle_aa 8 (+ yy 4) 3 46 46 60 1)
              (txtb (get_item RANK_NAME i) 15 yy 90 100 120) })
        (set! i (+ i 1)) })
    ;; 操作提示（三分区 + 键盘）
    (txtb "Z/< =L" 4 184 100 150 200)
    (txtb "MID=LR" 4 196 150 220 255)
    (txtb "X/> =R" 4 208 100 150 200)
    (txtb "SPACE:GO" 4 220 150 130 90)
    0))

;; 底部三分区指示条：一根手指按在哪一段就抬哪一枚（中间 = 两枚一起）
(define draw_zones
  (lambda ()
    (define mid 26)
    (if (and (== fl_on 1) (== fr_on 1)) (set! mid 62))
    (Gfx.fill_rect 0 235 ZONE_L 5 40 12 18)
    (Gfx.fill_rect ZONE_L 235 (- ZONE_R ZONE_L) 5 46 mid 12)
    (Gfx.fill_rect ZONE_R 235 (- 320 ZONE_R) 5 40 12 18)
    (Gfx.fill_rect 0 235 ZONE_L 1 110 40 50)
    (Gfx.fill_rect ZONE_L 235 (- ZONE_R ZONE_L) 1 (+ 110 mid) (+ 60 mid) (+ 30 mid))
    (Gfx.fill_rect ZONE_R 235 (- 320 ZONE_R) 1 110 40 50)
    0))

(define draw_msg
  (lambda ()
    (if (> msg_timer 0.0)
        { (Gfx.rounded_rect 76 168 210 18 4 4 4 4 6 6 18 150)
          (Gfx.fill_rect 78 169 206 1 60 90 150)
          (if (== msg_color 1) (txtc msg_text 181 171 150 230 255)
              (if (== msg_color 2) (txtc msg_text 181 171 120 250 160)
                  (if (== msg_color 6) (txtc msg_text 181 171 255 200 80)
                      (txtc msg_text 181 171 255 120 140)))) })
    0))

(define draw_lane_hint
  (lambda ()
    (define pul 0)
    (set! pul (Math.floor (+ 70.0 (* 90.0 (Math.sin (* 0.09 frame_no))))))
    (if (and (== bst 0) (== game_over 0))
        { ;; 待发：向上的呼吸箭头（静止画面也有生气，兼作存活心跳）
          (Gfx.draw_triangle 298 126 308 126 303 117 0 24 (Math.floor (* 0.55 pul)) 110 1)
          (Gfx.draw_triangle 298 150 308 150 303 141 0 24 (Math.floor (* 0.55 pul)) 110 1) })
    (if (> charge 0.0)
        { ;; 蓄力条
          (Gfx.fill_rect 292 100 16 100 20 20 40)
          (Gfx.fill_rect 294 (+ 198 (Math.floor (* -96.0 charge))) 12
                         (Math.floor (* 96.0 charge)) 255 200 60)
          (Gfx.draw_rect 292 100 16 100 90 120 180) })
    0))

(define draw_over
  (lambda ()
    (if (== game_over 1)
        { (Gfx.rounded_rect 84 86 194 92 6 6 6 6 4 4 14 175)
          (Gfx.rounded_rect 86 88 190 88 6 6 6 6 14 10 40 95)
          (Gfx.fill_rect 88 88 186 1 70 100 160)
          (txtc "GAME OVER" 180 94 255 110 130)
          (txtc "FINAL SCORE" 180 112 130 170 220)
          (Gfx.set_font 1)
          (Gfx.draw_text (fmt_g score) (Math.floor (- 180 (/ (* (String.length (fmt_g score)) 7) 2))) 126 255 230 90)
          (txtc (String.concat "RANK: " (get_item RANK_NAME rankidx)) 180 142 120 250 160)
          (txtc "TAP TO PLAY AGAIN" 180 160 150 220 255) })
    0))

(define draw_all
  (lambda ()
    (Gfx.clear)
    (draw_panel)
    (draw_logo)
    (draw_field)
    (draw_walls)
    (draw_targets)
    (draw_lanes)
    (draw_spinner)
    (draw_hole)
    (draw_sling)
    (draw_bumpers)
    (draw_flippers)
    (draw_ball)
    (draw_particles)
    (draw_plunger)
    (draw_lane_hint)
    (draw_hud)
    (draw_msg)
    (draw_over)
    (draw_zones)
    (EX.draw_button)      ;; 右上角「退出」按钮（宿主在触屏边沿判定命中）
    0))

;; ============================================================================
;; 主循环
;; ============================================================================

(define step_game
  (lambda (dt)
    (define i 0)
    (define h 0.0)
    (define k 0)

    ;; 计时器
    (if (> msg_timer 0.0) (set! msg_timer (- msg_timer dt)))
    (if (> sling_flash 0.0) (set! sling_flash (- sling_flash dt)))
    (set! i 0)
    (while (< i NBUMP)
      { (if (> (get_item bumper_flash i) 0.0)
            (set_item! bumper_flash i (- (get_item bumper_flash i) (* dt 4.0))))
        (if (> (get_item bumper_cd i) 0.0)
            (set_item! bumper_cd i (- (get_item bumper_cd i) dt)))
        (set! i (+ i 1)) })
    (if (> tg_timer 0.0)
        { (set! tg_timer (- tg_timer dt))
          (if (<= tg_timer 0.0) (set_targets_up)) })

    (update_particles dt)

    (if (== bst 1)
        ;; 拖尾
        { (set_item! btrail_x btrail_head bx)
          (set_item! btrail_y btrail_head by)
          (set! btrail_head (mod (+ btrail_head 1) 8)) })

    ;; 挡板
    (update_flippers dt)

    ;; 球
    (if (== bst 1)
        { (set! i 0)
          (set! h (/ dt NSUB))
          (while (< i NSUB)
            { (phys_step h)
              (set! i (+ i 1)) })
          ;; 离开过发射道后才允许"回到发射道"重挂
          (if (and (== left_lane 0)
                   (or (< bx (- LANE_L 4.0)) (< by 110.0)))
              (set! left_lane 1))
          (if (and (== left_lane 1) (> bx (- LANE_L 4.0)) (> by 216.0))
              { (set! bst 0)
                (set! bx 303.0)
                (set! by 228.0)
                (set! bvx 0.0)
                (set! bvy 0.0)
                (set! charge 0.0) }) }
        #f)
    (if (== bst 2)
        { (set! btimer (- btimer dt))
          (if (<= btimer 0.0)
              { (set! bst 1)
                (set! bx hole_x)
                (set! by (- hole_y 6.0))
                (set! bvx (- -300.0 (* 120.0 (Math.random))))
                (set! bvy (- -260.0 (* 160.0 (Math.random))))
                (set! hole_cd 2.4) }) })
    (if (== bst 3)
        { (set! btimer (- btimer dt))
          (if (<= btimer 0.0)
              { (if (> balls 0) (new_ball)) }) })

    (if (== bst 1) { (check_triggers dt) (set! play_t (+ play_t dt)) })

    ;; 防卡死：球在台面上几乎不动超过 1.2s 就给一次随机轻推（水平台面/夹角处会停住）
    (if (== bst 1)
        { (set! stuck_t (+ stuck_t dt))
          (if (> stuck_t 1.2)
              { (set! stuck_t 0.0)
                (if (< (Math.sqrt (+ (* bvx bvx) (* bvy bvy))) 30.0)
                    { (set! bvx (+ (* 160.0 (- (* 2.0 (Math.random)) 1.0)) 60.0))
                      (set! bvy -220.0)
                      (display "[pinball] nudge")(newline) }) }) }
        (set! stuck_t 0.0))


    ;; 诊断行（每 2 秒）
    (if (>= frame_no last_stat)
        { (set! last_stat (+ frame_no 120))
          (display "[pinball] st=") (display bst)
          (display " score=") (display score)
          (display " balls=") (display balls)
          (display " mult=") (display mult)
          (display " rank=") (display rankidx)
          (display " bx=") (display (Math.floor bx))
          (display " by=") (display (Math.floor by))
          (display " fl=") (display fl_on)
          (display " fr=") (display fr_on)
          (newline) })
    0))

(define tick
  (lambda ()
    (define now 0)
    (define d 0)
    (define dt 0.0)
    (set! now (System.timestamp))
    (set! d (- now last_ts))
    (if (< d 0) (set! d (+ d 134217728)))
    (if (< d 1) (set! d 1))
    (if (> d DT_MAX_MS) (set! d DT_MAX_MS))
    (set! last_ts now)
    (set! dt (/ d 1000.0))
    (set! frame_no (+ frame_no 1))
    (input_update dt)
    (step_game dt)
    (draw_all)
    (EX.check)
    (Gfx.refresh)
    0))

;; ============================================================================
;; 启动
;; ============================================================================

(define main
  (lambda ()
    (build_table)
    (build_standups)
    (build_stars)
    (reset_game)
    (set! last_ts (System.timestamp))
    (EX.install_with draw_all)
    (display "[pinball] start segs=") (display nseg)
    (display " circles=") (display ncir) (newline)
    (set! last_stat 120)
    (draw_all)
    (Gfx.refresh)
    (System.set_interval TICK_MS tick)
    0))

(main)
