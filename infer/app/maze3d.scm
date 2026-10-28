;; ============================================================================
;; 三维迷宫（第一人称漫游）—— 灵机应用程序
;;
;; 灵感来自 Windows 95 的 "3D Maze" 屏保：以第一人称视角在迷宫里自动穿行，
;; 转过拐角时会看到新的走廊。这里同样不需要任何 3D 硬件——
;;
;; 渲染用经典的 **raycasting**（Wolfenstein 3D 那套）：
;;   视线平面 → 每列一条射线 → 用 DDA 步进网格找第一个墙格 → 墙高 = H / 垂直距离
;;   → 画一根竖条。距离越远越暗（雾），东西向与南北向墙面亮度不同（假光照），
;;   墙格颜色由格子坐标哈希决定（相邻墙块颜色不同，拐角一眼可见）。
;; 顶棚与地板用几条渐变横带打底，墙条覆盖其上。
;;
;; 自动漫游（本平台 Scheme App 暂无按键通道）：
;;   迷宫由"递归回溯"算法随机生成（保证连通、无环）；
;;   **路径由 A\* 预先算好**（曼哈顿启发式 + 线性扫描取最小 f），拿到格心序列后再
;;   逐个航点走过去——不是盲目游走，所以镜头沿着有目的性的路线推进。
;;   到达目标格后随机挑一个远端格重新规划，形成"一直在赶路"的观感。
;; 转向是**平缓**的：角速度 0.10 rad/拍（90° 要约 16 拍 ≈0.6 s），且转向未完成时
;;   按角度差压低前进速度；遇到需要掉头（角度差 > 2.4 rad）时更慢，像一点一点把
;;   车头掰过来，而不是瞬间甩头。
;; 左上角常驻**迷宫全图**（2 像素/格）：墙、通路、A\* 规划路线、以及摄像机位置与朝向。
;;
;; 成本实测（host x86-64）：160 根竖条（每列 2 像素宽）× 平均 ~8 次 DDA 步进，
;; 单帧约 3~4 ms；定时器 40 ms/帧 → 约 10% CPU。浏览器 wasm32 约 15~20 fps。
;;
;; 书写约定（本解释器的坑，详见 app/turtle.scm 头部）：算术一律二元嵌套
;;（用 infer/tools/e2e/host/check_multi_operand.py 静态检查）；cond 子句只能带
;; 一个表达式且多子句编译失败；Math 库没有 min/max；括号与花括号都要数。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Diag)

;; ---- 画布与节奏 ------------------------------------------------------------

(define W (Gfx.width))            ; 320
(define H (Gfx.height))           ; 240
(define COLPX 2)                  ; 每根竖条的像素宽
(define NCOL (/ W COLPX))         ; 160 根
(define HUD_H 16)                 ; 底部信息带
(define VIEW_H (- H HUD_H))       ; 画面高度
(define TICK_MS 40)
(define MAZE_TICKS 1600)          ; 走这么多拍后换一座迷宫

;; ---- 迷宫尺寸（格子数）-----------------------------------------------------

(define CW 12)                    ; 横向格子
(define CH 9)                     ; 纵向格子
(define MW (+ (* 2 CW) 1))        ; 地图宽（含墙）
(define MH (+ (* 2 CH) 1))        ; 地图高
(define MSIZE (* MW MH))

(define map (cons 0 (quote ())))  ; 1=墙 0=通路
(define visited (cons 0 (quote ())))

;; ---- 玩家状态 --------------------------------------------------------------

(define posx 1.5) (define posy 1.5)
(define ang 0.0)                  ; 当前朝向（弧度）
(define tang 0.0)                 ; 目标朝向（平滑转向用）
(define dirx 1.0) (define diry 0.0)
(define planex 0.0) (define planey 0.66)   ; 相机平面（FOV ≈ 66°）
(define SPEED 0.075)              ; 每拍前进的格数
(define TURN_RATE 0.10)           ; 每拍转向弧度（平缓：90° 约 16 拍）

(define cellx 0) (define celly 0) ; 当前所在格（格子坐标，不是地图坐标）
(define tgtx 0) (define tgty 0)   ; 目标格
(define lastdx 0) (define lastdy 0) ; 上一步的格间方向（避免立刻回头）

(define maze_seed 0)
(define tick_no 0)
(define frame_ms 0)

;; ---- 随机数 ----------------------------------------------------------------

;; 用 Math.random（vendored 实现是 srand(time)+rand()，各宿主都有）。
;; 不用自建 LCG：乘法会超 27 位整数载荷，在 wasm32 / armv7 这类 32 位宿主上会溢出。
(define rnd (lambda () (Math.random)))

(define rnd_int
  (lambda (n)
    (define v 0)
    (set! v (Math.floor (* (rnd) (+ n 0.0))))
    (if (>= v n) { (set! v 0) })
    (if (< v 0) { (set! v 0) })
    v))

;; ---- 迷宫生成（递归回溯 + 显式栈）------------------------------------------

(define map_at
  (lambda (mx my) (get_item map (+ (* my MW) mx))))

(define map_set
  (lambda (mx my v) (set_item! map (+ (* my MW) mx) v)))

(define gen_maze
  (lambda ()
    (define i 0) (define x 0) (define y 0) (define stack (cons 0 (quote ())))
    (define cx 0) (define cy 0) (define n 0) (define tried 0) (define ok 0)
    (define nx 0) (define ny 0) (define dirs 0) (define d 0) (define pick 0)
    ;; 全填墙
    (set! i 0)
    (while (< i MSIZE) { (set_item! map i 1) (set_item! visited i 0) (set! i (+ i 1)) })
    (pop stack)
    ;; 从 (0,0) 开始
    (set! cx 0) (set! cy 0)
    (set_item! visited (+ (* cy CW) cx) 1)
    (map_set (+ (* 2 cx) 1) (+ (* 2 cy) 1) 0)
    (push stack cx) (push stack cy)
    (while (> (length stack) 0) {
      (set! cy (pop stack)) (set! cx (pop stack))
      ;; 收集未访问邻居（0=东 1=南 2=西 3=北）
      (set! dirs (cons 0 (quote ())))
      (pop dirs)
      (set! d 0)
      (while (< d 4) {
        (set! nx cx) (set! ny cy)
        (if (== d 0) { (set! nx (+ cx 1)) })
        (if (== d 1) { (set! ny (+ cy 1)) })
        (if (== d 2) { (set! nx (- cx 1)) })
        (if (== d 3) { (set! ny (- cy 1)) })
        (set! ok 0)
        (if (>= nx 0) { (if (< nx CW) { (if (>= ny 0) { (if (< ny CH) {
          (if (== (get_item visited (+ (* ny CW) nx)) 0) { (set! ok 1) }) }) }) }) })
        (if (== ok 1) { (push dirs d) })
        (set! d (+ d 1)) })
      (set! n (length dirs))
      (if (> n 0)
          {
            ;; 本格还有未访问邻居：把自己压回栈，再选一个邻居打通并前进
            (push stack cx) (push stack cy)
            (set! pick (rnd_int n))
            (set! d (get_item dirs pick))
            (set! nx cx) (set! ny cy)
            (if (== d 0) { (set! nx (+ cx 1)) })
            (if (== d 1) { (set! ny (+ cy 1)) })
            (if (== d 2) { (set! nx (- cx 1)) })
            (if (== d 3) { (set! ny (- cy 1)) })
            ;; 打通两格之间的墙
            (map_set (+ (+ (* 2 cx) 1) (- nx cx)) (+ (+ (* 2 cy) 1) (- ny cy)) 0)
            (map_set (+ (* 2 nx) 1) (+ (* 2 ny) 1) 0)
            (set_item! visited (+ (* ny CW) nx) 1)
            (push stack nx) (push stack ny) }
          {})
      (set! tried (+ tried 1)) })
    (pop stack)))

;; ---- 起始位置 --------------------------------------------------------------

(define reset_player
  (lambda ()
    (set! posx 1.5) (set! posy 1.5)
    (set! cellx 0) (set! celly 0)
    (set! tgtx 0) (set! tgty 0)
    (set! lastdx 0) (set! lastdy 0)
    (set! ang 0.0) (set! tang 0.0)
    (set! dirx 1.0) (set! diry 0.0)
    (set! planex 0.0) (set! planey 0.66)
    ;; 若东侧是墙就朝南
    (if (== (map_at 2 1) 1) { (set! ang 1.5708) (set! tang 1.5708)
                              (set! dirx 0.0) (set! diry 1.0)
                              (set! planex -0.66) (set! planey 0.0) })
    (set! tgtx cellx) (set! tgty celly)
    (plan_leg)
    (take_waypoint)))

;; ---- 朝向平滑过渡 ----------------------------------------------------------

(define apply_angle
  (lambda (a)
    (set! ang a)
    (set! dirx (Math.cos ang))
    (set! diry (Math.sin ang))
    (set! planex (* (- 0.0 (Math.sin ang)) 0.66))
    (set! planey (* (Math.cos ang) 0.66))))

(define turn_toward
  (lambda ()
    (define d 0.0) (define dd 0.0)
    (set! d (- tang ang))
    ;; 归一到 (-π, π]
    (while (> d 3.14159) { (set! d (- d 6.28318)) })
    (while (< d -3.14159) { (set! d (+ d 6.28318)) })
    (set! dd d)
    (if (> dd TURN_RATE) { (set! dd TURN_RATE) })
    (if (< dd (- 0.0 TURN_RATE)) { (set! dd (- 0.0 TURN_RATE)) })
    (if (> (Math.abs d) 0.001) { (apply_angle (+ ang dd)) })))

;; ---- 走到下一格 ------------------------------------------------------------

(define open_at
  (lambda (mx my)
    (define v 1)
    (if (>= mx 0) { (if (< mx MW) { (if (>= my 0) { (if (< my MH) {
      (set! v (map_at mx my)) }) }) }) })
    (if (== v 0) { 1 } { 0 })))

;; ---- A* 路径规划 -----------------------------------------------------------
;; 节点 = 格子 (cx,cy)，邻居 = 相邻且中间的墙已打通；代价 1（4 邻域），
;; 启发式 = 曼哈顿距离。规模只有 12×9=108 个节点，所以开表用普通列表 + 线性取最小，
;; 不必上二叉堆。

(define gs (cons 0 (quote ())))     ; g 值
(define fs (cons 0 (quote ())))     ; f 值
(define came (cons 0 (quote ())))   ; 父节点
(define inopen (cons 0 (quote ()))) ; 是否在开表中

(define astar_init
  (lambda ()
    (pop gs) (pop fs) (pop came) (pop inopen)
    (define k 0)
    (while (< k (* CW CH)) {
      (push gs 1e9) (push fs 1e9) (push came -1) (push inopen 0)
      (set! k (+ k 1)) })))

;; 两格之间是否连通（中间那道墙必须已被打通）
(define linked
  (lambda (ax ay bx by)
    (define wx 0) (define wy 0) (define v 1)
    (set! wx (+ (+ (* 2 ax) 1) (- bx ax)))
    (set! wy (+ (+ (* 2 ay) 1) (- by ay)))
    (if (>= wx 0) { (if (< wx MW) { (if (>= wy 0) { (if (< wy MH) {
      (set! v (map_at wx wy)) }) }) }) })
    (if (== v 0) { 1 } { 0 })))

(define hcost
  (lambda (a b)
    (define ax 0) (define ay 0) (define bx 0) (define by 0) (define d 0)
    (set! ax (mod a CW)) (set! ay (Math.floor (/ (+ a 0.0) (+ CW 0.0))))
    (set! bx (mod b CW)) (set! by (Math.floor (/ (+ b 0.0) (+ CW 0.0))))
    (set! d (+ (Math.abs (- ax bx)) (Math.abs (- ay by))))
    d))

;; 返回格心序列（列表，从当前格之后的第一格开始）
(define plan_path
  (lambda (sx sy gx gy)
    (define start 0) (define goal 0) (define open (cons 0 (quote ())))
    (define cur 0) (define best 0) (define bestf 0) (define found 0)
    (define n 0) (define i 0) (define c 0) (define cx 0) (define cy 0)
    (define nb 0) (define nx 0) (define ny 0) (define tg 0) (define tgv 0)
    (define path (cons 0 (quote ()))) (define rev (cons 0 (quote ())))
    (pop open) (pop path) (pop rev)
    (set! start (Math.floor (+ (* sy CW) sx)))
    (set! goal (Math.floor (+ (* gy CW) gx)))
    (set_item! gs start 0)
    (set_item! fs start (hcost start goal))
    (push open start)
    (set_item! inopen start 1)
    (set! found 0)
    (set! i 0)
    (while (and (== found 0) (> (length open) 0) (< i 4000))
      {
        ;; 取 f 最小的节点
        (set! n (length open))
        (set! best 0) (set! bestf 1e9) (set! c 0)
        (while (< c n) {
          (set! cur (get_item open c))
          (if (< (get_item fs cur) bestf) { (set! bestf (get_item fs cur)) (set! best c) })
          (set! c (+ c 1)) })
        (set! cur (get_item open best))
        (set_item! inopen cur 0)
        ;; 从开表移除（末尾补位 + pop）
        (set! n (- n 1))
        (set_item! open best (get_item open n))
        (pop open)
        (if (== cur goal) { (set! found 1) }
            {
              (set! cx (Math.floor (mod cur CW)))
              (set! cy (Math.floor (/ (+ cur 0.0) (+ CW 0.0))))
              ;; 四个邻居
              (set! nb 0)
              (while (< nb 4) {
                (set! nx cx) (set! ny cy)
                (if (== nb 0) { (set! nx (+ cx 1)) })
                (if (== nb 1) { (set! ny (+ cy 1)) })
                (if (== nb 2) { (set! nx (- cx 1)) })
                (if (== nb 3) { (set! ny (- cy 1)) })
                (if (>= nx 0)
                    { (if (< nx CW)
                          { (if (>= ny 0)
                                { (if (< ny CH)
                                      {
                                        (if (== (linked cx cy nx ny) 1)
                                            {
                                              (set! tg (Math.floor (+ (* ny CW) nx)))
                                              (set! tgv (+ (get_item gs cur) 1))
                                              (if (< tgv (get_item gs tg))
                                                  {
                                                    (set_item! came tg cur)
                                                    (set_item! gs tg tgv)
                                                    (set_item! fs tg (+ tgv (hcost tg goal)))
                                                    (if (== (get_item inopen tg) 0)
                                                        { (push open tg) (set_item! inopen tg 1) }
                                                        {}) }
                                                  {}) }
                                            {}) }
                                      {}) }
                                {}) }
                          {}) }
                    {})
                (set! nb (+ nb 1)) }) })
        (set! i (+ i 1)) })
    ;; 回溯：从 goal 走回 start（不含 start），再反转成前进顺序
    (if (== found 1)
        {
          (set! cur goal)
          (while (> cur -1) { (push rev cur) (set! cur (get_item came cur)) })
          (pop rev)                      ; 去掉起点自身
          (while (> (length rev) 0) { (push path (pop rev)) }) }
        {})
    path))

(define path (cons 0 (quote ())))
(define path_i 0)
(define goalx 0) (define goaly 0)

(define plan_leg
  (lambda ()
    (define tries 0) (define gx 0) (define gy 0)
    (pop path)
    (set! path_i 0)
    ;; 目标：远离当前位置的一个格子（A* 保证连通，任意格都能到）
    (set! tries 0)
    (while (< tries 30) {
      (set! gx (rnd_int CW))
      (set! gy (rnd_int CH))
      (if (>= (+ (Math.abs (- gx cellx)) (Math.abs (- gy celly))) 10) { (set! tries 100) }
          { (set! tries (+ tries 1)) })
      (if (== tries 100) {} {}) })
    (set! goalx gx) (set! goaly gy)
    (set! path (plan_path cellx celly gx gy))
    (set! path_i 0)))

;; 取下一个航点（格心 + 目标朝向）
(define take_waypoint
  (lambda ()
    (define id 0)
    (if (>= path_i (length path))
        { (plan_leg) }
        {})
    (if (> (length path) 0)
        {
          (set! id (get_item path (Math.floor path_i)))
          (set! tgtx (Math.floor (mod id CW)))
          (set! tgty (Math.floor (/ (+ id 0.0) (+ CW 0.0))))
          (set! path_i (+ path_i 1))
          ;; 目标朝向
          (if (> tgtx cellx) { (set! tang 0.0) })
          (if (< tgtx cellx) { (set! tang 3.14159) })
          (if (> tgty celly) { (set! tang 1.5708) })
          (if (< tgty celly) { (set! tang -1.5708) }) }
        { (set! tgtx cellx) (set! tgty celly) })))

(define step_walk
  (lambda ()
    (define cx 0.0) (define cy 0.0) (define dvx 0.0) (define dvy 0.0) (define dl 0.0)
    (define sp 0.0) (define ad 0.0)
    (turn_toward)
    ;; 目标格心（地图连续坐标：格 (i,j) 的格心 = (2i+1.5, 2j+1.5)）
    (set! cx (+ (* (+ tgtx 0.0) 2.0) 1.5))
    (set! cy (+ (* (+ tgty 0.0) 2.0) 1.5))
    (set! dvx (- cx posx))
    (set! dvy (- cy posy))
    (set! dl (Math.sqrt (+ (* dvx dvx) (* dvy dvy))))
    ;; 转向没完成时压低速度；掉头（|角度差| 大）时压得更狠，像一点点掰车头
    (set! ad (Math.abs (- tang ang)))
    (set! sp (* SPEED 1.0))
    (if (> ad 1.2) { (set! sp (* SPEED 0.45)) })
    (if (> ad 2.4) { (set! sp (* SPEED 0.22)) })
    (if (> dl 1e-6)
        { (if (> dl sp)
              { (set! posx (+ posx (* (/ dvx dl) sp)))
                (set! posy (+ posy (* (/ dvy dl) sp))) }
              { (set! posx cx) (set! posy cy)
                (set! cellx tgtx) (set! celly tgty)
                (take_waypoint) }) }
        { (set! cellx tgtx) (set! celly tgty) (take_waypoint) })))

;; ---- 渲染 ------------------------------------------------------------------

(define draw_backdrop
  (lambda ()
    (define i 0) (define y0 0) (define y1 0) (define c 0) (define half 0)
    (set! half (/ VIEW_H 2))          ; 112
    ;; 顶棚 6 条（由上到下渐亮）、地板 6 条（由近到远渐暗）；用整数分段保证无缝
    (set! i 0)
    (while (< i 6) {
      (set! y0 (/ (* i half) 6))
      (set! y1 (/ (* (+ i 1) half) 6))
      (set! c (+ 16 (* i 7)))
      (Gfx.fill_rect 0 y0 W (- y1 y0) c (+ c 5) (+ c 16))
      (set! i (+ i 1)) })
    (set! i 0)
    (while (< i 6) {
      (set! y0 (+ half (/ (* i half) 6)))
      (set! y1 (+ half (/ (* (+ i 1) half) 6)))
      (set! c (- 52 (* i 7)))
      (if (< c 8) { (set! c 8) })
      (Gfx.fill_rect 0 y0 W (- y1 y0) (+ c 12) c (+ c 5))
      (set! i (+ i 1)) })))

(define render
  (lambda ()
    (define col 0) (define camx 0.0) (define rdx 0.0) (define rdy 0.0)
    (define mx 0) (define my 0) (define ddx 0.0) (define ddy 0.0)
    (define sx 0.0) (define sy 0.0) (define stx 0) (define sty 0) (define side 0)
    (define hit 0) (define guard 0) (define pdist 0.0) (define lh 0.0)
    (define y0 0) (define y1 0) (define r 0) (define g 0) (define b 0)
    (define shade 0.0) (define base 0) (define hh 0)
    (define wallx 0.0) (define stripe 0) (define hm 0)
    (define CX 0.0) (define CY 0.0)
    (draw_backdrop)
    (set! CX (+ posx 0.0))
    (set! CY (+ posy 0.0))
    (set! col 0)
    (while (< col NCOL) {
      (set! camx (- (/ (+ (* col 2.0) 1.0) (+ NCOL 0.0)) 1.0))
      (set! rdx (+ dirx (* planex camx)))
      (set! rdy (+ diry (* planey camx)))
      (set! mx (Math.floor CX))
      (set! my (Math.floor CY))
      (if (== rdx 0.0) { (set! ddx 1e30) } { (set! ddx (Math.abs (/ 1.0 rdx))) })
      (if (== rdy 0.0) { (set! ddy 1e30) } { (set! ddy (Math.abs (/ 1.0 rdy))) })
      (if (< rdx 0.0)
          { (set! stx -1) (set! sx (* (- CX mx) ddx)) }
          { (set! stx 1) (set! sx (* (- (+ mx 1.0) CX) ddx)) })
      (if (< rdy 0.0)
          { (set! sty -1) (set! sy (* (- CY my) ddy)) }
          { (set! sty 1) (set! sy (* (- (+ my 1.0) CY) ddy)) })
      (set! hit 0) (set! side 0) (set! guard 0)
      (while (and (== hit 0) (< guard 40))
        {
          (if (< sx sy)
              { (set! sx (+ sx ddx)) (set! mx (+ mx stx)) (set! side 0) }
              { (set! sy (+ sy ddy)) (set! my (+ my sty)) (set! side 1) })
          (set! guard (+ guard 1))
          (if (== (open_at mx my) 0) { (set! hit 1) })
        })
      (if (== side 0)
          { (set! pdist (- sx ddx)) }
          { (set! pdist (- sy ddy)) })
      (if (< pdist 0.05) { (set! pdist 0.05) })
      (set! lh (/ VIEW_H pdist))
      (set! y0 (Math.floor (- (/ VIEW_H 2.0) (/ lh 2.0))))
      (set! y1 (Math.floor (+ (/ VIEW_H 2.0) (/ lh 2.0))))
      (if (< y0 0) { (set! y0 0) })
      (if (> y1 VIEW_H) { (set! y1 VIEW_H) })
      ;; 墙色：格子哈希选基色，南北/东西面亮度不同，再加距离雾
      (set! base (mod (+ (* mx 5) (* my 3)) 4))
      (if (== base 0) { (set! r 190) (set! g 170) (set! b 140) })
      (if (== base 1) { (set! r 165) (set! g 150) (set! b 175) })
      (if (== base 2) { (set! r 150) (set! g 175) (set! b 160) })
      (if (== base 3) { (set! r 185) (set! g 155) (set! b 130) })
      (set! shade (/ 1.0 (+ 1.0 (* (* 0.30 pdist) pdist))))
      (if (== side 1) { (set! shade (* shade 0.72)) })
      ;; 墙面纹理：命中点在格内的横向位置 → 竖条纹；再叠一条"踢脚线"暗带
      (if (== side 0)
          { (set! wallx (+ CY (* pdist rdy))) }
          { (set! wallx (+ CX (* pdist rdx))) })
      (set! wallx (- wallx (Math.floor wallx)))
      (set! stripe (Math.floor (* wallx 6.0)))
      (if (== (mod stripe 2) 1) { (set! shade (* shade 0.82)) })
      (set! r (Math.floor (* r shade)))
      (set! g (Math.floor (* g shade)))
      (set! b (Math.floor (* b shade)))
      (set! hh (- y1 y0))
      (if (> hh 0)
          {
            (set! hm (+ y0 (Math.floor (* hh 0.62))))
            (Gfx.fill_rect (* col COLPX) y0 COLPX (- hm y0) r g b)
            (Gfx.fill_rect (* col COLPX) hm COLPX (- y1 hm)
                           (Math.floor (* r 0.72)) (Math.floor (* g 0.72)) (Math.floor (* b 0.72))) })
      (set! col (+ col 1)) })
    (draw_minimap)))

;; ---- 左上角迷宫全图 --------------------------------------------------------

(define MMX 4) (define MMY 4) (define MMS 3)   ; 位置与每格像素

(define draw_minimap
  (lambda ()
    (define mx 0) (define my 0) (define i 0) (define id 0) (define cx 0) (define cy 0)
    (define px 0) (define py 0) (define hx 0) (define hy 0)
    ;; 底板
    (Gfx.fill_rect MMX MMY (* MW MMS) (* MH MMS) 8 10 18)
    ;; 墙
    (set! my 0)
    (while (< my MH) {
      (set! mx 0)
      (while (< mx MW) {
        (if (== (map_at mx my) 1)
            { (Gfx.fill_rect (+ MMX (* mx MMS)) (+ MMY (* my MMS)) MMS MMS 118 138 168) }
            {})
        (set! mx (+ mx 1)) })
      (set! my (+ my 1)) })
    ;; A* 规划路线（还没走到的航点，用暗绿点标出）
    (set! i path_i)
    (while (< i (length path)) {
      (set! id (get_item path (Math.floor i)))
      (set! cx (Math.floor (mod id CW)))
      (set! cy (Math.floor (/ (+ id 0.0) (+ CW 0.0))))
      (Gfx.fill_rect (+ MMX (* (+ (* cx 2) 1) MMS)) (+ MMY (* (+ (* cy 2) 1) MMS)) MMS MMS 60 150 80)
      (set! i (+ i 1)) })
    ;; 目标格
    (Gfx.fill_rect (+ MMX (* (+ (* goalx 2) 1) MMS)) (+ MMY (* (+ (* goaly 2) 1) MMS)) MMS MMS 220 90 90)
    ;; 摄像机：位置 + 朝向短线
    (set! px (+ MMX (Math.floor (* posx MMS))))
    (set! py (+ MMY (Math.floor (* posy MMS))))
    (Gfx.fill_rect (- px 1) (- py 1) 3 3 255 235 120)
    (set! hx (+ px (Math.floor (* dirx 7.0))))
    (set! hy (+ py (Math.floor (* diry 7.0))))
    (Gfx.draw_line px py hx hy 255 215 90)
    ;; 边框
    (Gfx.draw_rect (- MMX 1) (- MMY 1) (+ (* MW MMS) 2) (+ (* MH MMS) 2) 70 90 120)))

;; ---- HUD -------------------------------------------------------------------

(define ds "")
(define cat3 (lambda (a b c) (String.concat a (String.concat b c))))

(define draw_hud
  (lambda ()
    (Gfx.fill_rect 0 VIEW_H W HUD_H 0 0 0)
    (Gfx.draw_text "三维迷宫 · 自动漫游" 4 (+ VIEW_H 2) 200 215 235)
    (set! ds (cat3 "A* " (String.atom_to_string path_i)
                   (cat3 "/" (String.atom_to_string (length path))
                         (cat3 "  " (String.atom_to_string frame_ms) "ms"))))
    (Gfx.draw_text ds 220 (+ VIEW_H 2) 110 140 170)))

;; ---- 主流程 ----------------------------------------------------------------

(define new_maze
  (lambda ()
    (set! maze_seed (+ maze_seed 1))
    (gen_maze)
    (reset_player)
    (set! tick_no 0)))

(define tick_fn
  (lambda ()
    (define t0 0) (define k 0)
    (set! t0 (Diag.ms))
    (set! k 0)
    (while (< k 2) { (step_walk) (set! k (+ k 1)) })
    (render)
    (set! frame_ms (- (Diag.ms) t0))
    (draw_hud)
    (Gfx.refresh)
    (set! tick_no (+ tick_no 1))
    (if (>= tick_no MAZE_TICKS) { (new_maze) })))

;; ---- 启动 ------------------------------------------------------------------

(pop map) (pop visited)
(define q 0)
(while (< q MSIZE) { (push map 1) (push visited 0) (set! q (+ q 1)) })
(astar_init)
(new_maze)
(set! _timer (System.set_interval TICK_MS tick_fn))