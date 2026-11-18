;@id=fly
;@title=精灵战机
;@version=2026-11-18
;; ============================================================================
;; 精灵堆叠战机（fly）—— 灵机应用程序
;;
;; 一台贴着海面低飞的像素风战机：高度场岛屿 + 渐变海面 + 积云 + 体素叠层战机。
;;
;;   · 世界：128×128 高度场（8 世界单位/格，环绕），20 个岛由"1 大 + 2 小"椭圆团
;;     叠加生成（半径带角向扰动，另加正弦哈希的地表起伏）；
;;   · 海面：平面（h = 0），屏幕每一行都对应唯一距离，由投影公式**反解**后逐行
;;     画渐变带 —— 光滑、几乎不花算力；
;;   · 陆地：107 列 × 3 像素的列式 ray-march（voxel-space 推进 + 逐列 ybottom
;;     遮挡），只对 h > 0 的采样点投影成带，带只画到该距离处的水面 y；
;;   · 战机：13 层 34×46 体素切片按后上方斜视堆叠烘焙成 8 个偏航角桶的透明贴图
;;     （app/lib/fly/jet_0..7.png + shadow.png），运行期每帧 1 次
;;     draw_sprite_rotated；压坡度 = 贴图平面旋转（正坡度 = 右转、右翼下沉）；
;;   · 其余：9 团程序化积云（随偏航视差横移）、30 条海面波光短划、距离雾、
;;     翼尖拖尾、机影、自动驾驶（10 s 无输入后缓慢盘旋）。
;;
;; 操作：
;;   · 触屏：按住拖动 = 转向/俯仰（触点在战机左侧就左转），按住 = 加力；
;;   · 键盘（WASM 目标由浏览器注入）：←/→ 转向，↑/↓ 俯仰，Shift 加力，空格换视角；
;;   · 右上「退出」走宿主全局确认框（app/lib/exit.scm）。
;;
;; 书写约定（本解释器的坑，见手册与 app/turtle.scm 头部）：
;;   · 算术一律二元嵌套；取负写 (- 0 x)；/ 与 mod 恒返回浮点；
;;   · 没有 let/case/for/return；cond 子句只能带一个表达式 → 平铺 if；
;;   · while 循环体内不写 define（局部量全部在 lambda 体顶部 define）；
;;   · 语句位置调用产值函数会泄漏操作数栈 → 产值调用一律接 set!；
;;   · Math 库没有 min/max；Math.PI 是函数；位运算只接受无符号整数 → 不用位运算，
;;     索引一律 mod + get_item 的浮点索引（get_item 索引会四舍五入）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Diag)
(native Touch)
(native Key)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认（各 App 统一）

;; ---- 画布与节奏 ------------------------------------------------------------

(define W (Gfx.width))
(define H (Gfx.height))
(define TICK_MS 33)              ; 30 Hz

;; ---- 世界参数 --------------------------------------------------------------
;; 高度图 128×128 格，每格 8 世界单位 → 世界 1024 单位，环绕。
;; 视距 780 单位 ≈ 97 格，够看出"岛屿层叠到地平线"的纵深。

(define HN 128)
(define WORLD 1024.0)
(define CELL 8.0)
(define CELL_INV 0.125)

;; ---- 相机 / 投影 -----------------------------------------------------------

(define FOCAL 250.0)             ; 焦距（像素）：FOV ≈ 2·atan(160/250) ≈ 65°
(define CAM_BACK 122.0)          ; 相机在机尾后方的距离（世界单位）
(define CAM_UP 26.0)             ; 相机比战机高多少
(define CAM_PITCH 0.155)         ; 相机俯角（弧度，正 = 向下看）
(define V_NEAR 13.0)             ; 最近采样距离
(define V_FAR 660.0)             ; 最远采样距离
(define V_GROW 0.30)            ; 采样步长增长率（几何推进）
(define V_STEP_MAX 42.0)        ; 单步上限（世界单位）：太大就会跨过小岛

(define COLS 107)                ; 屏幕列数
(define COLW 3)                  ; 每列宽（像素）：107×3 = 321 ≈ 屏宽
(define TAU 6.2831853)

;; ---- 色彩（分量变量，热循环里省 get_item） ---------------------------------

(define C_SKY_T_R 74) (define C_SKY_T_G 132) (define C_SKY_T_B 208)
(define C_SKY_H_R 176) (define C_SKY_H_G 214) (define C_SKY_H_B 238)
(define C_SEA_D_R 10) (define C_SEA_D_G 44) (define C_SEA_D_B 92)
(define C_SEA_R 22) (define C_SEA_G 86) (define C_SEA_B 148)
(define C_SHOAL_R 58) (define C_SHOAL_G 146) (define C_SHOAL_B 186)
(define C_FOAM_R 208) (define C_FOAM_G 232) (define C_FOAM_B 240)
(define C_SAND_R 216) (define C_SAND_G 198) (define C_SAND_B 142)
(define C_GRASS_R 54) (define C_GRASS_G 126) (define C_GRASS_B 72)
(define C_GRASS2_R 88) (define C_GRASS2_G 158) (define C_GRASS2_B 86)
(define C_ROCK_R 126) (define C_ROCK_G 118) (define C_ROCK_B 108)
(define C_SNOW_R 232) (define C_SNOW_G 238) (define C_SNOW_B 244)

;; ---- 战机飞行状态 ----------------------------------------------------------

(define jet_x 300.0)
(define jet_y 300.0)
(define jet_alt 70.0)
(define jet_yaw 0.8)
(define jet_pitch 0.0)
(define jet_roll 0.0)
(define jet_speed 96.0)
(define jet_thrust 0.0)
(define crash_flash 0)
(define eject_msg 0)
(define view_mode 0)             ; 0 = 追尾，1 = 俯瞰
(define view_mix 0.0)

;; ---- 相机状态 --------------------------------------------------------------

(define cam_yaw 0.8)
(define cam_x 0.0)
(define cam_y 0.0)
(define cam_alt 0.0)
(define horizon 100.0)
(define cam_pitch 0.10)   ; 相机俯角（弧度）：越大越往下看，地平线上移

;; ---- 输入 ------------------------------------------------------------------

(define in_turn 0.0)
(define in_climb 0.0)
(define in_boost 0)
(define idle_ticks 400)          ; 起始即自动驾驶（先让人看到画面）
(define auto_phase 0.0)
(define view_key_down 0)

;; ---- 帧计测 ----------------------------------------------------------------

(define frame_ms 0)

;; ---- 战机贴图 --------------------------------------------------------------

(define SPR_BASE "app/lib/fly/jet_")
(define SPR_SHADOW "app/lib/fly/shadow.png")
(define SPR (cdr '(0)))
(define spr_w (cdr '(0)))
(define spr_h (cdr '(0)))
(define shadow_w 0)
(define shadow_h 0)

;; ============================================================================
;; 世界生成：深海底铺满，再涂椭圆岛屿
;; ============================================================================

(define hmap (cdr '(0)))

(define clampf
  (lambda (v a b)
    (if (< v a) a (if (> v b) b v))))

;; 世界坐标 → 高度图行首下标（格）
(define row_of
  (lambda (wy)
    (* (Math.floor (mod (+ (* wy CELL_INV) 8192.0) 128.0)) 128.0)))

;; 世界坐标 → 高度图列下标（格）
(define col_of
  (lambda (wx)
    (Math.floor (mod (+ (* wx CELL_INV) 8192.0) 128.0))))

(define height_index
  (lambda (wx wy)
    (+ (row_of wy) (col_of wx))))

(define ground_at
  (lambda (wx wy)
    (get_item hmap (height_index wx wy))))

;; 一个椭圆岛团：中心 (cx,cy) 世界单位、半径 rx,ry 世界单位、峰值高度、相位
(define paint_blob
  (lambda (cx cy rx ry peak ph)
    (define ex 0.0)
    (define ey 0.0)
    (define dx 0.0)
    (define dy 0.0)
    (define wob 0.0)
    (define d2 0.0)
    (define f 0.0)
    (define h 0.0)
    (define idx 0.0)
    (define prev 0.0)
    (set! ex (+ 1.0 (Math.floor (* rx CELL_INV))))
    (set! ey (+ 1.0 (Math.floor (* ry CELL_INV))))
    (set! dy (- 0.0 ey))
    (while (<= dy ey) {
      (set! dx (- 0.0 ex))
      (while (<= dx ex) {
        ;; 归一化椭圆距离；半径乘一个角向扰动，海岸线才不像标准椭圆
        (set! wob (+ 0.80 (* 0.20 (Math.sin (+ (+ (* (+ cx (* dx CELL)) 0.031)
                                                  (* (+ cy (* dy CELL)) 0.017))
                                               ph)))))
        (set! d2 (+ (/ (* dx dx) (* ex ex)) (/ (* dy dy) (* ey ey))))
        (if (< d2 (* wob wob))
            {
              (set! f (- 1.0 (/ d2 (* wob wob))))
              (set! h (* peak (* f f)))
              (set! idx (height_index (+ cx (* dx CELL)) (+ cy (* dy CELL))))
              (set! prev (get_item hmap idx))
              (if (> h prev) (set_item! hmap idx h))
            })
        (set! dx (+ dx 1.0))
      })
      (set! dy (+ dy 1.0))
    })
    #undefined))

(define build_world
  (lambda ()
    (define i 0)
    (define k 0)
    (define cx 0.0)
    (define cy 0.0)
    (define r 0.0)
    (define peak 0.0)
    (define ang 0.0)
    (set! i 0)
    (while (< i (* HN HN)) {
      (push hmap (- 0.0 (+ 2.2 (* 1.8 (Math.random)))))
      (set! i (+ i 1))
    })
    (set! k 0)
    (while (< k 20) {
      (set! cx (* WORLD (Math.random)))
      (set! cy (* WORLD (Math.random)))
      (set! r (+ 44.0 (* 70.0 (Math.random))))
      (set! peak (+ 7.0 (* 17.0 (Math.random))))
      (set! ang (* TAU (Math.random)))
      (paint_blob cx cy r (* r (+ 0.62 (* 0.5 (Math.random)))) peak ang)
      (paint_blob (+ cx (* 0.95 (* r (Math.cos ang)))) (+ cy (* 0.95 (* r (Math.sin ang))))
                  (* 0.46 r) (* 0.40 r) (* 0.60 peak) (+ ang 1.7))
      (paint_blob (- cx (* 0.72 (* r (Math.sin ang)))) (+ cy (* 0.72 (* r (Math.cos ang))))
                  (* 0.34 r) (* 0.30 r) (* 0.42 peak) (+ ang 3.1))
      (set! k (+ k 1))
    })
    #undefined))

;; ============================================================================
;; 战机贴图预热与度量（缓存 12 槽，8 张 + 机影刚好放得下）
;; ============================================================================

(define load_sprites
  (lambda ()
    (define i 0)
    (define p "")
    (define sz #null)
    (while (< i 8) {
      (set! p (String.concat SPR_BASE (String.concat (String.atom_to_string i) ".png")))
      (push SPR p)
      (set! sz (Gfx.sprite_info p))
      (if (list? sz)
          {
            (push spr_w (get_item sz 0))
            (push spr_h (get_item sz 1))
          }
          {
            (push spr_w 0)
            (push spr_h 0)
          })
      (set! i (+ i 1))
    })
    (set! sz (Gfx.sprite_info SPR_SHADOW))
    (if (list? sz)
        {
          (set! shadow_w (get_item sz 0))
          (set! shadow_h (get_item sz 1))
        })
    #undefined))

;; ============================================================================
;; 输入
;; ============================================================================

(define read_input
  (lambda ()
    (define t 0.0)
    (define c 0.0)
    (define b 0)
    (define pressed #f)
    (if (Key.is_down 3) (set! t (- 0.0 1.0)))       ; ←
    (if (Key.is_down 4) (set! t 1.0))               ; →
    (if (Key.is_down 1) (set! c 1.0))               ; ↑
    (if (Key.is_down 2) (set! c (- 0.0 1.0)))       ; ↓
    (if (Key.is_down 7) (set! b 1))                 ; Shift
    (set! pressed (Touch.is_pressed))
    (if pressed
        {
          (set! t (clampf (/ (- (Touch.x) (* 0.5 W)) 110.0) (- 0.0 1.0) 1.0))
          (set! c (clampf (/ (- (* 0.60 H) (Touch.y)) 90.0) (- 0.0 1.0) 1.0))
          (set! b 1)
        })
    (set! in_turn t)
    (set! in_climb c)
    (set! in_boost b)
    (if (or (not (== t 0.0)) (not (== c 0.0)))
        (set! idle_ticks 0)
        (set! idle_ticks (+ idle_ticks 1)))
    #undefined))

(define poll_view
  (lambda ()
    (define c 0)
    (set! c (Key.take_press))
    (while (not (== c 0)) {
      (if (== c 32) (set! view_mode (- 1 view_mode)))
      (set! c (Key.take_press))
    })
    #undefined))

;; ============================================================================
;; 飞行模型
;; ============================================================================

(define step_flight
  (lambda (dt)
    (define t 0.0)
    (define c 0.0)
    (define g 0.0)
    (define tgt_roll 0.0)
    (define tgt_pitch 0.0)
    (define spd 0.0)
    (define climb 0.0)
    (define rate 0.0)
    (set! t in_turn)
    (set! c in_climb)
    ;; 自动驾驶：长时间无输入 → 缓慢盘旋 + 轻微起伏
    (if (> idle_ticks 300)
        {
          (set! auto_phase (+ auto_phase (* dt 0.42)))
          (set! t (* 0.62 (Math.sin auto_phase)))
          (set! c (* 0.16 (Math.sin (* 0.83 auto_phase))))
        })
    ;; 加力 → 速度
    (if (== in_boost 1)
        (set! jet_thrust (+ jet_thrust (* dt 2.2)))
        (set! jet_thrust (- jet_thrust (* dt 1.6))))
    (set! jet_thrust (clampf jet_thrust 0.0 1.0))
    (set! spd (+ 84.0 (* 60.0 jet_thrust)))
    (set! jet_speed (+ jet_speed (* (- spd jet_speed) (* dt 1.5))))
    ;; 姿态：坡度跟转向，俯仰跟操纵杆
    ;; 坡度 = 压向转弯内侧：向右转（t>0）机身右倾（右翼下沉），正号即物理坡度
    (set! tgt_roll (* t 0.60))
    (set! jet_roll (+ jet_roll (* (- tgt_roll jet_roll) (* dt 3.2))))
    (set! tgt_pitch (* c 0.32))
    (set! g (ground_at jet_x jet_y))
    (if (< jet_alt (+ g 26.0))
        (set! tgt_pitch (clampf (+ tgt_pitch 0.30) (- 0.0 0.4) 0.5)))
    (set! jet_pitch (+ jet_pitch (* (- tgt_pitch jet_pitch) (* dt 2.4))))
    ;; 转向速率：低速时更灵活一点（与速度的平方根成反比）
    (set! rate (/ 0.85 (Math.sqrt (/ (+ jet_speed 40.0) 100.0))))
    (set! jet_yaw (+ jet_yaw (* t (* rate dt))))
    ;; 位移
    (set! climb (* (Math.sin jet_pitch) jet_speed))
    (set! jet_x (+ jet_x (* (Math.sin jet_yaw) (* jet_speed dt))))
    (set! jet_y (+ jet_y (* (Math.cos jet_yaw) (* jet_speed dt))))
    (set! jet_alt (+ jet_alt (* climb dt)))
    (set! jet_x (mod (+ jet_x WORLD) WORLD))
    (set! jet_y (mod (+ jet_y WORLD) WORLD))
    ;; 撞海/撞地：闪白 + 「弹射」+ 拉起复位
    (set! g (ground_at jet_x jet_y))
    (if (<= jet_alt (+ g 4.0))
        {
          (set! jet_alt (+ g 52.0))
          (set! jet_pitch 0.40)
          (set! crash_flash 3)
          (set! eject_msg 36)
        })
    #undefined))

;; ============================================================================
;; 相机跟随
;; ============================================================================

(define wrap_pi
  (lambda (a)
    (- a (* TAU (Math.round (/ a TAU))))))

(define cam_back
  (lambda ()
    (+ CAM_BACK (* 210.0 view_mix))))

(define cam_up
  (lambda ()
    (+ CAM_UP (* 155.0 view_mix))))

(define step_camera
  (lambda (dt)
    (define d 0.0)
    (define back 0.0)
    (set! view_mix (+ view_mix (* (- (* 1.0 view_mode) view_mix) (* dt 3.0))))
    (set! d (wrap_pi (- jet_yaw cam_yaw)))
    (set! cam_yaw (+ cam_yaw (* d (* dt 3.6))))
    (set! cam_yaw (mod (+ cam_yaw TAU) TAU))
    (set! back (cam_back))
    (set! cam_x (- jet_x (* (Math.sin cam_yaw) back)))
    (set! cam_y (- jet_y (* (Math.cos cam_yaw) back)))
    (set! cam_alt (+ jet_alt (cam_up)))
    (set! cam_pitch (+ CAM_PITCH (* 0.30 view_mix)))
    (set! horizon (- (* 0.5 H) (* FOCAL (Math.tan cam_pitch))))
    #undefined))

;; ============================================================================
;; 渲染：天空（横带渐变 + 太阳）
;; ============================================================================

(define draw_sky
  (lambda (hz0)
    (define hz 0)
    (define y 0)
    (define f 0.0)
    (define bh 0)
    (define sx 0)
    ;; 俯瞰视角会把地平线推到屏幕上方之外（甚至恰好 0）→ 夹一个下限，避免除零
    (set! hz (if (< hz0 3) 3 hz0))
    (set! bh (+ 1 (Math.floor (/ hz 12.0))))
    (set! sx (+ (* 0.72 W) (* 52.0 (Math.sin cam_yaw))))
    (set! y (- 0 bh))
    (while (< y hz) {
      (set! f (/ (+ y bh) hz))
      (set! f (* f f))
      (Gfx.fill_rect 0 (+ y bh) W bh
                     (+ C_SKY_T_R (Math.round (* f (- C_SKY_H_R C_SKY_T_R))))
                     (+ C_SKY_T_G (Math.round (* f (- C_SKY_H_G C_SKY_T_G))))
                     (+ C_SKY_T_B (Math.round (* f (- C_SKY_H_B C_SKY_T_B)))))
      (set! y (+ y bh))
    })
    (Gfx.fill_circle sx (- hz 46) 30 255 243 206)
    (Gfx.fill_circle sx (- hz 46) 22 255 252 234)
    (Gfx.fill_rect 0 (- hz 1) W 2 C_SKY_H_R C_SKY_H_G C_SKY_H_B)
    #undefined))

;; ============================================================================
;; 渲染：高度场列式 ray-march
;; ============================================================================
;; 每列沿水平方向推进，采样高度场，把采样点投影成屏幕 y，维护 ybottom（已画到
;; 的最低行）：新采样点只在更靠上时成带绘制。这就是 voxel-space（Comanche）
;; 渲染法 —— 遮挡天然正确，且每列只画真正可见的那些带。

(define V_TAB (cdr '(0)))
(define V_N 0)

;; 采样距离表：只由 V_NEAR/V_GROW/上限决定，与列无关 → 启动时算一遍
(define build_vtab
  (lambda ()
    (define v V_NEAR)
    (define dv 0.0)
    (while (< v V_FAR) {
      (push V_TAB v)
      (set! dv (* v V_GROW))
      (if (< dv 2.0) (set! dv 2.0))
      (if (> dv V_STEP_MAX) (set! dv V_STEP_MAX))
      (set! v (+ v dv))
    })
    (set! V_N (length V_TAB))
    #undefined))

;; 海面：屏幕行 → 反解视线与海平面(h=0)的交点距离 → 横带渐变
(define draw_sea
  (lambda (top0)
    (define top 0)
    (define y 0)
    (define s 0.0)
    (define q 0.0)
    (define v 0.0)
    (define f 0.0)
    (define d 0.0)
    (define tp 0.0)
    (define cr 0) (define cg 0) (define cb 0)
    (set! tp (Math.tan cam_pitch))
    (set! top (if (< top0 2) 2 top0))
    (set! y top)
    (while (< y H) {
      (set! s (/ (- (+ y 1.0) (* 0.5 H)) FOCAL))
      (set! q (/ (+ s tp) (- 1.0 (* s tp))))
      (if (< q 0.0008) (set! q 0.0008))
      (set! v (/ cam_alt q))
      (set! d (/ v 520.0))
      (if (> d 1.0) (set! d 1.0))
      (set! f (* d d))
      (set! cr (+ C_SHOAL_R (* f (- C_SEA_D_R C_SHOAL_R))))
      (set! cg (+ C_SHOAL_G (* f (- C_SEA_D_G C_SHOAL_G))))
      (set! cb (+ C_SHOAL_B (* f (- C_SEA_D_B C_SHOAL_B))))
      (set! f (/ v 820.0))
      (if (> f 0.92) (set! f 0.92))
      (set! f (* f f))
      (Gfx.fill_rect 0 y W 2
                     (+ cr (* f (- C_SKY_H_R cr)))
                     (+ cg (* f (- C_SKY_H_G cg)))
                     (+ cb (* f (- C_SKY_H_B cb))))
      (set! y (+ y 2))
    })
    #undefined))

;; 陆地：列式 ray-march（只对 h > 0 的采样点投影成带）
(define draw_land
  (lambda (cols colw)
    (define col 0) (define px 0) (define k 0)
    (define ang 0.0) (define dxc 0.0) (define dyc 0.0)
    (define v 0.0) (define vn 0.0) (define dv 0.0)
    (define cwx 0.0) (define cwy 0.0)
    (define h 0.0) (define ys 0.0) (define yb 0.0)
    (define q 0.0) (define f 0.0)
    (define cr 0) (define cg 0) (define cb 0)
    (define camx 0.0) (define camy 0.0) (define camz 0.0)
    (define half 0.0) (define tp 0.0)
    (define ywl 0.0) (define ytop 0.0) (define hw CELL_INV)
    (set! camx (* cam_x CELL_INV))
    (set! camy (* cam_y CELL_INV))
    (set! camz cam_alt)
    (set! half (* 0.5 H))
    (set! tp (Math.tan cam_pitch))
    (while (< col cols) {
      (set! px (* col colw))
      (set! ang (+ cam_yaw (Math.atan (/ (- (+ px (* 0.5 colw)) (* 0.5 W)) FOCAL))))
      (set! dxc (* (Math.sin ang) hw))
      (set! dyc (* (Math.cos ang) hw))
      (set! v V_NEAR)
      (set! cwx (+ camx (* dxc v)))
      (set! cwy (+ camy (* dyc v)))
      (set! yb (+ H 4.0))
      (set! k 0)
      (while (< k V_N) {
        (set! vn (get_item V_TAB k))
        (set! dv (- vn v))
        (set! v vn)
        (set! cwx (+ cwx (* dxc dv)))
        (set! cwy (+ cwy (* dyc dv)))
        (if (>= cwx 128.0) (set! cwx (- cwx 128.0)))
        (if (< cwx 0.0) (set! cwx (+ cwx 128.0)))
        (if (>= cwy 128.0) (set! cwy (- cwy 128.0)))
        (if (< cwy 0.0) (set! cwy (+ cwy 128.0)))
        (set! h (get_item hmap (+ (* (Math.floor cwy) 128.0) (Math.floor cwx))))
        (if (> h 0.30)
            {
              (set! q (/ (- camz h) v))
              (set! ys (+ half (* FOCAL (/ (- q tp) (+ 1.0 (* q tp))))))
              (set! q (/ camz v))
              (set! ywl (+ half (* FOCAL (/ (- q tp) (+ 1.0 (* q tp))))))
              (set! ytop (if (< yb ywl) yb ywl))
              (if (< ys ytop)
                  {
                    (set! f (* v 0.00115))
                    (if (> f 0.86) (set! f 0.86))
                    (set! f (* f f))
                    (if (< h 0.9)
                        { (set! cr C_FOAM_R) (set! cg C_FOAM_G) (set! cb C_FOAM_B) }
                        (if (< h 2.0)
                            { (set! cr C_SAND_R) (set! cg C_SAND_G) (set! cb C_SAND_B) }
                            (if (< h 9.0)
                                { (set! cr C_GRASS_R) (set! cg C_GRASS_G) (set! cb C_GRASS_B) }
                                (if (< h 15.0)
                                    { (set! cr C_GRASS2_R) (set! cg C_GRASS2_G) (set! cb C_GRASS2_B) }
                                    (if (< h 22.0)
                                        { (set! cr C_ROCK_R) (set! cg C_ROCK_G) (set! cb C_ROCK_B) }
                                        { (set! cr C_SNOW_R) (set! cg C_SNOW_G) (set! cb C_SNOW_B) })))))
                    (Gfx.fill_rect px ys colw (- ytop ys)
                                   (+ cr (* f (- C_SKY_H_R cr)))
                                   (+ cg (* f (- C_SKY_H_G cg)))
                                   (+ cb (* f (- C_SKY_H_B cb))))
                    (set! yb ys)
                  })
            })
        (set! k (+ k 1))
      })
      (set! col (+ col 1))
    })
    #undefined))

;; ---- 云层（程序化积云，块状 = 像素风）
(define CLD_N 9)
(define cld (cdr '(0)))

(define cloud_init
  (lambda ()
    (define i 0)
    (while (< i CLD_N) {
      (push cld (* TAU (Math.random)))
      (push cld (+ 10.0 (* 46.0 (Math.random))))
      (push cld (+ 26.0 (* 46.0 (Math.random))))
      (push cld (+ 6.0 (* 8.0 (Math.random))))
      (push cld (* (* 0.6 (- (Math.random) 0.5)) 0.02))
      (set! i (+ i 1))
    })
    #undefined))

(define draw_clouds
  (lambda (dt)
    (define i 0) (define k 0)
    (define a 0.0) (define ho 0.0) (define cw 0.0) (define ch 0.0) (define dr 0.0)
    (define sx 0.0) (define sy 0.0)
    (set! i 0)
    (while (< i CLD_N) {
      (set! k (* i 5))
      (set! a (get_item cld k))
      (set! ho (get_item cld (+ k 1)))
      (set! cw (get_item cld (+ k 2)))
      (set! ch (get_item cld (+ k 3)))
      (set! dr (get_item cld (+ k 4)))
      (set! a (mod (+ (+ a (* dr dt)) TAU) TAU))
      (set_item! cld k a)
      (set! sx (+ (* 0.5 W) (* (wrap_pi (- a cam_yaw)) (* FOCAL 0.42))))
      (set! sx (- sx (* cw 0.5)))
      (while (> sx (+ W 40.0)) { (set! sx (- sx W)) })
      (while (< sx (- 0.0 (+ 40.0 cw))) { (set! sx (+ sx W)) })
      (set! sy (- (- horizon 12.0) ho))
      (if (< sy 2.0) (set! sy 2.0))
      (Gfx.fill_rect (+ sx 2.0) (+ sy (- ch 3.0)) (- cw 4.0) 3.0 198 214 230)
      (Gfx.fill_rect (+ sx (* cw 0.14)) (+ sy (* ch 0.66)) (* cw 0.72) (* ch 0.34) 232 240 250)
      (Gfx.fill_rect (+ sx (* cw 0.03)) (+ sy (* ch 0.40)) (* cw 0.94) (* ch 0.30) 244 248 253)
      (Gfx.fill_rect (+ sx (* cw 0.16)) (+ sy (* ch 0.18)) (* cw 0.68) (* ch 0.24) 251 253 255)
      (Gfx.fill_rect (+ sx (* cw 0.30)) sy (* cw 0.40) (* ch 0.20) 255 255 255)
      (set! i (+ i 1))
    })
    #undefined))

;; ---- 波光（海面闪光短划）
(define SPK_N 30)
(define spk (cdr '(0)))

(define sparkle_init
  (lambda ()
    (define i 0)
    (while (< i SPK_N) {
      (push spk (* W (Math.random)))
      (push spk (+ 100 (* 140 (Math.random))))
      (push spk (+ 18.0 (* 46.0 (Math.random))))
      (set! i (+ i 1))
    })
    #undefined))

(define draw_sparkle
  (lambda (dt)
    (define i 0) (define k 0)
    (define sx 0.0) (define sy 0.0) (define sv 0.0)
    (set! i 0)
    (while (< i SPK_N) {
      (set! k (* i 3))
      (set! sx (get_item spk k))
      (set! sy (get_item spk (+ k 1)))
      (set! sv (get_item spk (+ k 2)))
      (set! sy (+ sy (* sv dt)))
      (if (> sy (+ H 4.0))
          {
            (set! sy (+ horizon (* 26.0 (Math.random))))
            (set! sx (* W (Math.random)))
            (set! sv (+ 18.0 (* 46.0 (Math.random))))
            (set_item! spk k sx)
            (set_item! spk (+ k 1) sy)
            (set_item! spk (+ k 2) sv)
          })
      (if (> sy (+ horizon 12.0))
          (Gfx.fill_rect sx sy 3.0 1.0 214 236 244)
          #undefined)
      (set! i (+ i 1))
    })
    #undefined))

;; ============================================================================
;; 渲染：战机（贴图 + 机影 + 翼尖拖尾）
;; ============================================================================

(define jet_scr_x
  (lambda ()
    (+ (* 0.5 W) (* (Math.sin (wrap_pi (- jet_yaw cam_yaw))) FOCAL))))

(define jet_scr_y
  (lambda ()
    (+ horizon (/ (* (- cam_alt jet_alt) FOCAL) (cam_back)))))

(define draw_jet
  (lambda ()
    (define rel 0.0)
    (define bucket 0)
    (define idx 0)
    (define cx 0.0)
    (define cy 0.0)
    (define sw 0)
    (define shh 0)
    (define ang 0.0)
    (define pvx 0.0)
    (define pvy 0.0)
    (define gr 0.0)
    (define sy 0.0)
    (set! rel (wrap_pi (- jet_yaw cam_yaw)))
    (set! bucket (mod (Math.round (/ rel 0.78539816)) 8.0))
    (set! idx bucket)
    (set! sw (get_item spr_w idx))
    (set! shh (get_item spr_h idx))
    (set! cx (jet_scr_x))
    (set! cy (jet_scr_y))
    ;; 机影（先画，压在机体下面）
    (if (== shadow_w 0)
        #undefined
        {
          (set! gr (ground_at jet_x jet_y))
          (set! sy (+ horizon (/ (* (- cam_alt gr) FOCAL) (cam_back))))
          (if (> sy (+ cy 4.0))
              (Gfx.draw_sprite SPR_SHADOW cx sy)
              #undefined)
        })
    (if (== sw 0)
        #undefined
        {
          ;; 贴图以上缘中点为枢轴旋转；换算成「绕贴图中心旋转」：
          ;;   pivot = center − R(θ)·(0, h/2)
          ;; 库的正角为逆时针（局部 (lx,ly) → pivot + (lx·c + ly·s, −lx·s + ly·c)）：
          ;; 右倾（jet_roll > 0，右翼下沉）必须用负角；枢轴（上缘中点）补偿同理：
          ;;   pvx = cx − (h/2)·sin(ang)，pvy = cy − (h/2)·cos(ang)
          (set! ang (- 0.0 jet_roll))
          (set! pvx (- cx (* (* 0.5 shh) (Math.sin ang))))
          (set! pvy (- cy (* (* 0.5 shh) (Math.cos ang))))
          (Gfx.draw_sprite_rotated (get_item SPR idx) pvx pvy ang)
        })
    #undefined))

(define draw_trails
  (lambda ()
    (define cx 0.0)
    (define cy 0.0)
    (define d 0.0)
    (set! cx (jet_scr_x))
    (set! cy (jet_scr_y))
    (if (< jet_thrust 0.05)
        #undefined
        {
          (set! d (* 0.09 (+ 30.0 (* 50.0 jet_thrust))))
          (Gfx.draw_line (+ cx 20.0) (+ cy 6.0) (+ cx (+ 20.0 d)) (+ cy (+ 26.0 d)) 226 236 246)
          (Gfx.draw_line (- cx 20.0) (+ cy 6.0) (- cx (+ 20.0 d)) (+ cy (+ 26.0 d)) 226 236 246)
        })
    #undefined))

;; ============================================================================
;; HUD
;; ============================================================================

(define draw_hud
  (lambda ()
    (define spd "")
    (define alt "")
    (set! spd (String.atom_to_string (Math.round jet_speed)))
    (set! alt (String.atom_to_string (Math.round jet_alt)))
    (Gfx.draw_text "速度" 7 7 20 30 44) (Gfx.draw_text "速度" 6 6 250 250 250)
    (Gfx.draw_text " 节" 46 7 20 30 44) (Gfx.draw_text " 节" 45 6 250 250 250)
    (Gfx.draw_text spd 34 7 20 30 44) (Gfx.draw_text spd 33 6 255 246 210)
    (Gfx.draw_text "高度" 7 21 20 30 44) (Gfx.draw_text "高度" 6 20 250 250 250)
    (Gfx.draw_text alt 34 21 20 30 44) (Gfx.draw_text alt 33 20 255 246 210)
    (if (> jet_thrust 0.35)
        {
          (Gfx.draw_text "加力" 7 35 20 30 44)
          (Gfx.draw_text "加力" 6 34 255 206 84)
        })
    (if (> eject_msg 0)
        {
          (Gfx.draw_text "弹射！" 129 93 20 30 44)
          (Gfx.draw_text "弹射！" 128 92 255 232 128)
        })
    (if (> view_mix 0.5)
        {
          (Gfx.draw_text "俯瞰" 245 7 20 30 44)
          (Gfx.draw_text "俯瞰" 244 6 250 250 250)
        })
    (Gfx.draw_text (String.concat (String.atom_to_string frame_ms) "ms") 241 223 20 30 44)
    (Gfx.draw_text (String.concat (String.atom_to_string frame_ms) "ms") 240 222 236 242 250)
    #undefined))

;; ============================================================================
;; 每拍主流程
;; ============================================================================

(define tick_fn
  (lambda ()
    (define t0 0)
    (define dt 0.0333)
    (set! t0 (Diag.ms))
    (EX.check)
    (read_input)
    (poll_view)
    (step_flight dt)
    (step_camera dt)
    (draw_sky (Math.floor horizon))
    (draw_clouds dt)
    (draw_sea (Math.floor horizon))
    (draw_land COLS COLW)
    (draw_sparkle dt)
    (draw_jet)
    (draw_trails)
    (draw_hud)
    (if (> crash_flash 0)
        {
          (Gfx.fill_rect 0 0 W H 250 250 255)
          (set! crash_flash (- crash_flash 1))
        })
    (if (> eject_msg 0) (set! eject_msg (- eject_msg 1)))
    (EX.draw_button)
    (Gfx.refresh)
    (set! frame_ms (- (Diag.ms) t0))
    #undefined))

;; ============================================================================
;; 启动
;; ============================================================================

(Gfx.clear)
(Gfx.draw_text "生成岛屿…" 108 108 40 60 90)
(Gfx.refresh)
(build_world)
(build_vtab)
(load_sprites)
(sparkle_init)
(cloud_init)

;; 起始位置：沿对角线搜索一块低海拔陆地，停在它上方
(define find_start
  (lambda ()
    (define k 0)
    (define x 0.0)
    (define y 0.0)
    (define g 0.0)
    (define found 0)
    (set! k 0)
    (while (and (< k 1400) (== found 0)) {
      (set! x (mod (+ (* k 37.0) 120.0) WORLD))
      (set! y (mod (+ (* k 53.0) 240.0) WORLD))
      (set! g (ground_at x y))
      (if (and (> g 3.0) (< g 13.0))
          {
            (set! jet_x x)
            (set! jet_y y)
            (set! jet_alt (+ g 64.0))
            (set! found 1)
          })
      (set! k (+ k 1))
    })
    (set! cam_yaw jet_yaw)
    (set! cam_x (- jet_x (* (Math.sin cam_yaw) CAM_BACK)))
    (set! cam_y (- jet_y (* (Math.cos cam_yaw) CAM_BACK)))
    (set! cam_alt (+ jet_alt CAM_UP))
    (set! cam_pitch CAM_PITCH)
    (set! horizon (- (* 0.5 H) (* FOCAL (Math.tan cam_pitch))))
    #undefined))

(find_start)
(EX.install_with tick_fn)
(set! _timer (System.set_interval TICK_MS tick_fn))
