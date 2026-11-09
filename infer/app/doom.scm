;; ============================================================================
;; 灵机 DOOM —— 第一人称射击（raycasting 渲染 + 单点触屏适配）
;;
;; 参考开源实现：github.com/farrelke/console-doom（主机版 Doom 的渲染/游戏循环分层），
;; 但本 App 是**从零手写的独立实现**：没有 WAD 资源、没有 BSP、没有逐像素贴图，
;; 只用灵机的 (native Gfx) 画矩形/圆/文字，把 Doom 的观感用「列式 raycasting +
;; 程序化砖墙纹理 + 广告牌精灵 + HUD」重建出来。全部逻辑与素材都在本文件里。
;;
;; 渲染管线（每拍重画整屏，脏区由图形层记录，实测 320x240 / 80 列 在 x86 浏览器里
;; 有 ~16ms 余量：把节拍设成 1ms 时能跑到 rAF 上限 60fps；发布节拍取 33ms=30fps）：
;;   1) 天花板/地板：各 5 条渐变横带（越接近地平线越亮 → 天然的景深雾）；
;;   2) 墙体：80 列 × 4px 的 DDA raycasting（Wolfenstein 3D 同款几何）。
;;      每列画「底色块 + 3 条砖缝 + 顶部高光」共 5 个矩形 —— 用列内亮度差伪造出砖墙
;;      纹理，比逐像素贴图便宜两个数量级（本解释器里一条 fill_rect 是一次 native 调用）；
;;      东西向/南北向墙面亮度不同（假光照），距离衰减为雾，门有升降动画；
;;   3) 精灵：敌人/道具/出口/火球按相机空间变换成屏幕包围盒，再以**列级 z-buffer
;;      遮挡**逐块绘制（墙后的部分自然被裁掉，绕柱子能看到敌人的一半）；
;;   4) 武器视图：三把枪的程序化像素造型 + 走动摆动 + 后坐力 + 枪口焰；
;;   5) HUD：Doom 风格四面板（生命/弹药/护甲/击杀）+ 关卡 + 帧率；
;;   6) 触屏层：左侧虚拟摇杆、右侧拖动转向、开火/开门/换枪按钮、自动开火开关。
;;
;; 游戏内容：
;;   · 3 张手工关卡（房间 + 走廊 + 门 + 柱子，砖/钢/石/大理石/木五种材质）；
;;     全部格子连通性/实体可达性在离线脚本里做过 BFS 校验；
;;   · 3 种敌人：僵尸兵（近战）、火球兵（远程火球）、重甲兵（高血近战）；
;;     各有待机→追击→攻击蓄力→硬直→死亡状态机、视线检测（LOS）、贴墙滑动碰撞；
;;   · 3 把武器：手枪（无限弹、15 伤害）、霰弹枪（8 弹丸扇形 ×9）、链枪（8 伤害、高射速）；
;;     命中判定是 hitscan（沿射线做 DDA 求墙距，只有比墙更近的敌人才算命中）；
;;   · 拾取：医疗包 +25、弹药 +20 弹/+4 霰弹、护甲 +25；受击有 0.24s 无敌帧防围殴瞬秒；
;;   · 踩上绿色发光柱＝出口 → 过关；死亡/过关/通关三套结算画面，点击或按键继续。
;;   · **AI 自动游玩**（左上「AI 手动 / AI 自动」按钮或键盘 G）：BFS 距离场寻路 + 交战
;;     目标获取 + 自动开门/换枪/拾取 + 卡住自愈，能自己把三关打完并自动续关（演示模式）；
;;     AI 的输出走与人类输入**同一条通路**，物理/命中判定完全一致。详见文件内 AI 段落。
;;
;; 触屏适配（本平台只有**单点**触控：页面 Pointer Events 只跟踪一个 activePointerId，
;; Touch 本地库也只给电平 + 一个坐标，因此不能「左手走 + 右手转 + 同时开火」）：
;;   · 左半屏按下并拖动 = 虚拟摇杆（x 平移、y 前后），松手即停；
;;   · 右半屏按下拖动 = 转头（横向 1px ≈ 0.42°）；原地轻点 = 开火一枪；
;;   · 右下「开火」按住连发；其上方「开门」「换枪」单次；左上「自动开火」开关：
;;     打开后若 30° 锥内有看得见的敌人，视角会被**限速**拉向它（95°/s，不抢手动转向）
;;     并在对准时自动扣扳机 —— 这是单指也能走位+输出、且不会从背后被白打的关键；
;;   · 物理键盘同样可用（WASD 走、←→ 转、空格开火、E 开门、1/2/3 换枪、T 自动开火、
;;     G 切换 AI）；AI 运行中在屏幕别处按下手指立刻交还手动（与俄罗斯方块同约定）。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数（多操作数会污染操作数栈）；
;; 无一元取负、无 let/list/map/apply；cond 每子句只能一个表达式（本文件一律用 if）；
;; 块（{} / begin）内不 define（局部量用模块级槽或 lambda 形参）；/ 与 mod 恒浮点；
;; Math.PI 是函数；while 体不是独立词法区域。
;;
;; 两个与平台相关的坑（都已在本文件里规避，详见各处的注释）：
;;   ① **定时器回调必须包一层 lambda**：wasm32 上把自由变量很多的模块级闭包直接注册成
;;      (System.set_interval TICK_MS tick) 时回调一次都不会被调用（桌面 64 位宿主正常）；
;;      改成 (lambda () (tick)) 立刻正常。见文件末尾的注释。
;;   ② **fill_rect 的负坐标**：graphics.c 的 RGB888 慢路径原先把入参当 uint32 处理，
;;      x<0 的矩形会整体被丢弃（RGB565 快路径则正确裁剪）。精灵部分探出左边界时必然踩到，
;;      已在 spr_flush 里把坐标钳到屏内，并顺手修好了 graphics.c 的慢路径裁剪。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native Diag)
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认框 + 右上角「退出」按钮

;; ============================================================================
;; 画布与视口
;; ============================================================================

(define W (Gfx.width))
(define H (Gfx.height))
(define HUD_H 34)                 ; 底部状态栏高度
(define VH (- H HUD_H))           ; 3D 视口高度（206）
(define COLW 4)                   ; 每根竖条的像素宽
(define NCOL 80)                  ; 竖条数（80 × 4 = 320）
(define PLANE 0.66)               ; 相机平面长度 → FOV ≈ 66°
(define WK 1.45)                  ; 墙高系数（1 格高的墙在距离 d 处画 WK*VH/d 像素）
(define D2R 0.017453292519943295)

;; ============================================================================
;; 地图与格子类型
;; ============================================================================

(define MW 32)
(define MH 22)
(define MSIZE (* MW MH))

;; 0 空地 / 1 砖 / 2 钢 / 3 石 / 4 大理石 / 5 木 / 6 门 / 9 出口
;; 墙基色（下标 (t-1)*3），t=6 时为门色
(define PAL (quote (170 86 54  118 130 148  140 136 122  168 154 178  136 98 62  196 150 74)))

;; 关卡（手工设计：房间/走廊/门/柱子；字符见 parse_level 的注释）
(define LEVELS (quote (
  ( ; L1
   "################################"
   "#........####MMMMMMMMMMMMMSSSSSS"
   "#....z...####M......i....M.....S"
   "#........####M..z........M..a..S"
   "#.p......D...D........M..D.....S"
   "#........####M...MMM..M..M.....S"
   "#........####M...MMM..M..M...h.S"
   "#.....h..####M...MMM.....M.....S"
   "#......a.####M.........i.M.....S"
   "####D########M...MMM.....MSSSSSS"
   "####.########M...MMM.....M######"
   "####D########M...MMM..z..M######"
   "##.....######M...........M######"
   "##..a..######MMMMMMDMMMMMM######"
   "##.....############.BBBBBBBBBBBB"
   "###################.B..........B"
   "##W....D............D........h.B"
   "##Wh........W#######B..z.......B"
   "##W......i..W#######B.a.....E..B"
   "##W..z......W#######B......i...B"
   "##W.b.......W#######B..........B"
   "##WWWWWWWWWWW#######BBBBBBBBBBBB"
  )
  ( ; L2
   "#########MMMMMMMMMMMMMMM########"
   "#.......#M...........z.MSSSSSSSS"
   "#.p.....#M.z...........M....a..S"
   "#.......#M..SSS...SSS..M.......S"
   "#.a...h.#M..SSS...SSS..D.......S"
   "#.......D.......i......M..h....S"
   "#...z...#M..SSS...SSS..M.....i.S"
   "#.......#M..SSS...SSS..M.......S"
   "####.####M...........a.MSSSSSSSS"
   "####.####M.............M########"
   "WWWWDWWWWMMMMMMM.MMMMMMMMMMMMMMM"
   "W........W######.######M.....h.M"
   "W......a.WSSSSSSDSSSSSSS.z.....M"
   "W........W.............S.......M"
   "W..WWW...W.z........i..S..MMM..M"
   "W..WWW...W.....MMM.....D..MMM..M"
   "W........W.....MMM.....S.......M"
   "W.z......W.............S....E..M"
   "W........D........z....S..MMM..M"
   "W..h...z.W.....MMM.....S..MMMi.M"
   "W........W..a........b.S..a....M"
   "WWWWWWWWWWSSSSSSSSSSSSSSMMMMMMMM"
  )
  ( ; L3
   "SSSSSSSSBBBBBBBBBBBBBBBBBBBBBBBB"
   "S......SB......................B"
   "S....z.SB...BBi......BB...BB.h.B"
   "S......SB.z.BB.....z.BB.i.BB...B"
   "S.p....D.......................B"
   "S......SBBBBBBB....BBBBBBBBBBBBB"
   "S.a..h.S#######....#############"
   "S......S#######..b.#############"
   "SSSSSSSSBBBBBBB....BBBBBBB######"
   "########B....a...........B######"
   "########B..BB........BBh.B######"
   "########B.iBB...z....BB..B######"
   "########B................B######"
   "########BBBBBBBBDBBBBBBBBB######"
   "WWWWWWWWWWWWWWWW.WWWWWWWWWWWWWWW"
   "W.z.............D....i.........W"
   "WWWWWDWWWWWWWWWW.WWWWWWWWWDWWWWW"
   "#W..W.....W########W....W....h.W"
   "#W..W...a.W########W.a..W...E..W"
   "#W....h...W########W......z....W"
   "#W........W########W...........W"
   "#WWWWWWWWWW########WWWWWWWWWWWWW"
  )
)))

;; 关卡运行态
(define lv 0)                     ; 当前关卡下标（0..2）
(define map (cons 0 (quote ())))  ; 平铺格子表（MSIZE 项）
(define doorf (cons 0 (quote ()))) ; 每格的门开启进度 0..1（非门恒 0）
(define doors (cons 0 (quote ()))) ; 门格下标表
(define sp_x 0.0)                 ; 玩家起点
(define sp_y 0.0)
(define exit_x 0.0)               ; 出口格中心
(define exit_y 0.0)

;; ============================================================================
;; 实体数组（并行列表：比「列表套列表」少一层 get_item，热路径更快）
;; 敌人槽：x y hp 状态 计时 类型 受击闪烁 动画相位
;; ============================================================================

(define EMAX 24)
(define en_n 0)
(define en_x (quote ()))
(define en_y (quote ()))
(define en_hp (quote ()))
(define en_st (quote ()))    ; 0 待机 1 追击 2 攻击蓄力 3 硬直 4 死亡
(define en_t (quote ()))     ; 状态计时（秒）
(define en_k (quote ()))     ; 0 僵尸兵 1 火球兵 2 重甲兵
(define en_fl (quote ()))    ; 受击闪白剩余时间
(define en_ph (quote ()))    ; 走路动画相位

;; 道具槽：x y 类型 存活
(define IMAX 24)
(define it_n 0)
(define it_x (quote ()))
(define it_y (quote ()))
(define it_k (quote ()))     ; 10 医疗包 11 弹药 12 护甲
(define it_al (quote ()))

;; 火球槽：x y dx dy 存活
(define PMAX 24)
(define pr_n 0)
(define pr_x (quote ()))
(define pr_y (quote ()))
(define pr_dx (quote ()))
(define pr_dy (quote ()))
(define pr_al (quote ()))

;; 新建一个 n 项的 0 列表（必须从 (cons 0 '()) + pop 起步：字面量 '() 是静态对象，
;; 直接 push 会跨关卡累积）
(define mklist
  (lambda (n)
    (define l (cons 0 (quote ())))
    (define i 0)
    (pop l)
    (while (< i n) {
      (push l 0)
      (set! i (+ i 1))
    })
    l))

;; ============================================================================
;; 玩家状态
;; ============================================================================

(define px 2.5) (define py 2.5)   ; 位置（格单位）
(define pang 0.0)                 ; 朝向（度，0=东，顺时针为正）
(define pdirx 1.0) (define pdiry 0.0)
(define pplanex 0.0) (define pplaney PLANE)
(define hp 100)
(define armor 0)
(define bullets 0)                ; 链枪弹药
(define shells 0)                 ; 霰弹
(define wsel 0)                   ; 0 手枪 1 霰弹枪 2 链枪
(define fire_cd 0.0)              ; 射击冷却（秒）
(define fire_anim 0.0)            ; 开枪动画（秒，用于后坐力/枪口焰）
(define hurt_flash 0.0)           ; 受伤红屏（秒）
(define pain_cd 0.0)              ; 受击后的短暂无敌（秒，防多敌同帧叠加暴毙）
(define kills 0)
(define total_kills 0)
(define level_enemies 0)
(define msg_text "")
(define msg_ms 0)
(define st_state 0)               ; 0 标题 1 游戏中 2 死亡 3 过关 4 通关
(define st_timer 0.0)             ; 状态计时（秒）
(define auto_fire 1)              ; 自动开火（单指可玩的关键）
(define fps_v 0)
(define fps_acc 0.0)
(define fps_cnt 0)
(define game_ms 0)                ; 本局累计毫秒
(define shots_n 0)                ; 本会话累计开火次数（诊断/回归断言用，不随关卡重置）
(define walk_ph 0.0)              ; 走动相位（武器摆动/敌人走路动画）
(define dbg_ms 0)                 ; 诊断输出节拍（500ms 一行，供 e2e 断言）
(define dbg_nd 0.0)               ; 最近活敌距离
(define dbg_na 0.0)               ; 最近活敌方位角差
(define dbg_ns 0)                 ; 本帧精灵数

;; ============================================================================
;; 数学 / 小工具
;; ============================================================================

(define absf (lambda (x) (Math.abs x)))
(define minf (lambda (a b) (if (< a b) a b)))
(define maxf (lambda (a b) (if (> a b) a b)))
(define clampf (lambda (v lo hi) (if (< v lo) lo (if (> v hi) hi v))))

(define wrap360
  (lambda (a)
    (define r (mod a 360.0))
    (if (< r 0.0) (+ r 360.0) r)))

;; 有符号最短角差（-180, 180]
(define ang_diff
  (lambda (a b)
    (define d (wrap360 (- a b)))
    (if (> d 180.0) (- d 360.0) d)))

(define dist2d
  (lambda (ax ay bx by)
    (define dx (- ax bx))
    (define dy (- ay by))
    (Math.sqrt (+ (* dx dx) (* dy dy)))))

(define update_cam
  (lambda ()
    (define r (* pang D2R))
    (set! pdirx (Math.cos r))
    (set! pdiry (Math.sin r))
    (set! pplanex (* (- 0.0 pdiry) PLANE))
    (set! pplaney (* pdirx PLANE))))

;; ============================================================================
;; 地图查询
;; ============================================================================

(define tile_xy
  (lambda (x y)
    (define ix (Math.floor x))
    (define iy (Math.floor y))
    (if (< ix 0) 1
        (if (>= ix MW) 1
            (if (< iy 0) 1
                (if (>= iy MH) 1
                    (get_item map (+ (* iy MW) ix))))))))

;; 该格是否阻挡（门看开启进度）
(define solid_xy
  (lambda (x y)
    (define t (tile_xy x y))
    (if (== t 0) 0
        (if (== t 9) 0
            (if (== t 6)
                (if (>= (get_item doorf (+ (* (Math.floor y) MW) (Math.floor x))) 0.999) 0 1)
                1)))))

;; 半径 r 的圆形体积能否站在 (x,y)（取四角采样，够用且便宜）
(define can_stand
  (lambda (x y r)
    (define ok 1)
    (if (== (solid_xy (- x r) (- y r)) 1) (set! ok 0))
    (if (== (solid_xy (+ x r) (- y r)) 1) (set! ok 0))
    (if (== (solid_xy (- x r) (+ y r)) 1) (set! ok 0))
    (if (== (solid_xy (+ x r) (+ y r)) 1) (set! ok 0))
    ok))

;; 从 (x,y) 沿角度 ang（弧度）走的射线撞墙距离（用于射击命中判定）
(define ray_wall_dist
  (lambda (x y rx ry)
    (define mx (Math.floor x))
    (define my (Math.floor y))
    (define ddx 0.0) (define ddy 0.0)
    (define sdx 0.0) (define sdy 0.0)
    (define stx 0) (define sty 0)
    (define side 0) (define guard 0) (define hit 0) (define t 0)
    (if (== rx 0.0) (set! ddx 1000000.0) (set! ddx (Math.abs (/ 1.0 rx))))
    (if (== ry 0.0) (set! ddy 1000000.0) (set! ddy (Math.abs (/ 1.0 ry))))
    (if (< rx 0.0) { (set! stx -1) (set! sdx (* (- x mx) ddx)) }
                   { (set! stx 1) (set! sdx (* (- (+ mx 1.0) x) ddx)) })
    (if (< ry 0.0) { (set! sty -1) (set! sdy (* (- y my) ddy)) }
                   { (set! sty 1) (set! sdy (* (- (+ my 1.0) y) ddy)) })
    (set! guard 0)
    (while (== hit 0) {
      (if (< sdx sdy) { (set! sdx (+ sdx ddx)) (set! mx (+ mx stx)) (set! side 0) }
                      { (set! sdy (+ sdy ddy)) (set! my (+ my sty)) (set! side 1) })
      (set! guard (+ guard 1))
      (set! t (if (or (< mx 0) (or (>= mx MW) (or (< my 0) (>= my MH))))
                  1
                  (get_item map (+ (* my MW) mx))))
      (if (== t 0) #f (if (== t 9) #f
        (if (== t 6)
            (if (>= (get_item doorf (+ (* my MW) mx)) 0.999) #f (set! hit 1))
            (set! hit 1))))
      (if (> guard 200) (set! hit 1))
    })
    (if (== side 0) (maxf 0.05 (- sdx ddx)) (maxf 0.05 (- sdy ddy)))))

;; 视线可见（敌人 AI 用）：沿线每 0.35 格采样
(define los_clear
  (lambda (ax ay bx by)
    (define d (dist2d ax ay bx by))
    (define n (Math.floor (/ d 0.35)))
    (define i 1)
    (define ok 1)
    (define t 0.0)
    (if (> n 24) (set! n 24))
    (if (< n 2) (set! n 2))
    (while (< i n) {
      (set! t (/ i n))
      (if (== (solid_xy (+ ax (* (- bx ax) t)) (+ ay (* (- by ay) t))) 1) (set! ok 0))
      (set! i (+ i 1))
    })
    ok))

;; ============================================================================
;; 关卡解析
;; ============================================================================

;; 行内字符： # 砖  M 钢  S 石  B 大理石  W 木  D 门  E 出口  . 空地
;;           p 玩家起点  z 僵尸兵  i 火球兵  b 重甲兵  h 医疗包  a 弹药  r 护甲
(define load_level
  (lambda (li)
    (define rows (get_item LEVELS li))
    (define r 0) (define c 0) (define s "") (define ch 0) (define t 0) (define idx 0)
    (define ne 0) (define ni 0)
    (define bx 0.0) (define by 0.0) (define bk 0)
    (define sx (cons 0 (quote ()))) (define sy (cons 0 (quote ())))
    (define sk (cons 0 (quote ())))
    (set! lv li)
    (set! map (mklist MSIZE))
    (set! doorf (mklist MSIZE))
    (set! doors (cons 0 (quote ())))
    (pop doors)
    (pop sx) (pop sy) (pop sk)
    (set! r 0)
    (while (< r MH) {
      (set! s (get_item rows r))
      (set! c 0)
      (while (< c MW) {
        (set! ch (String.charCodeAt c s))
        (set! t 0)
        (set! bk 0)
        (if (== ch 35) (set! t 1))      ; #
        (if (== ch 77) (set! t 2))      ; M
        (if (== ch 83) (set! t 3))      ; S
        (if (== ch 66) (set! t 4))      ; B
        (if (== ch 87) (set! t 5))      ; W
        (if (== ch 68) (set! t 6))      ; D
        (if (== ch 69) (set! t 9))      ; E
        (if (== ch 112) { (set! sp_x (+ c 0.5)) (set! sp_y (+ r 0.5)) })   ; p
        (if (== ch 122) (set! bk 1))    ; z
        (if (== ch 105) (set! bk 2))    ; i
        (if (== ch 98) (set! bk 3))     ; b
        (if (== ch 104) (set! bk 10))   ; h
        (if (== ch 97) (set! bk 11))    ; a
        (if (== ch 114) (set! bk 12))   ; r
        (set! idx (+ (* r MW) c))
        (set_item! map idx t)
        (if (== t 6) (push doors idx))
        (if (== t 9) { (set! exit_x (+ c 0.5)) (set! exit_y (+ r 0.5)) })
        (if (> bk 0) {
          (push sx (+ c 0.5))
          (push sy (+ r 0.5))
          (push sk bk)
        })
        (set! c (+ c 1))
      })
      (set! r (+ r 1))
    })
    ;; 实体落位
    (set! en_n 0) (set! it_n 0) (set! pr_n 0)
    (set! en_x (mklist EMAX)) (set! en_y (mklist EMAX)) (set! en_hp (mklist EMAX))
    (set! en_st (mklist EMAX)) (set! en_t (mklist EMAX)) (set! en_k (mklist EMAX))
    (set! en_fl (mklist EMAX)) (set! en_ph (mklist EMAX))
    (set! it_x (mklist IMAX)) (set! it_y (mklist IMAX))
    (set! it_k (mklist IMAX)) (set! it_al (mklist IMAX))
    (set! pr_x (mklist PMAX)) (set! pr_y (mklist PMAX)) (set! pr_dx (mklist PMAX))
    (set! pr_dy (mklist PMAX)) (set! pr_al (mklist PMAX))
    (set! ne 0) (set! ni 0)
    (while (< ne (length sx)) {
      (set! bx (get_item sx ne))
      (set! by (get_item sy ne))
      (set! bk (get_item sk ne))
      (if (== bk 1) (set! bk 0))
      (if (== bk 2) (set! bk 1))
      (if (== bk 3) (set! bk 2))
      (if (< bk 3) {
        (if (< en_n EMAX) {
          (set_item! en_x en_n bx)
          (set_item! en_y en_n by)
          (set_item! en_hp en_n (get_item (quote (34 26 90)) bk))
          (set_item! en_st en_n 0)
          (set_item! en_t en_n 0.0)
          (set_item! en_k en_n bk)
          (set_item! en_fl en_n 0.0)
          (set_item! en_ph en_n 0.0)
          (set! en_n (+ en_n 1))
        })
      })
      (if (and (>= bk 10) (<= bk 12)) {
        (if (< it_n IMAX) {
          (set_item! it_x it_n bx)
          (set_item! it_y it_n by)
          (set_item! it_k it_n bk)
          (set_item! it_al it_n 1)
          (set! it_n (+ it_n 1))
        })
      })
      (set! ne (+ ne 1))
    })
    ;; 玩家与全局状态
    (set! px sp_x)
    (set! py sp_y)
    (set! pang 0.0)
    (set! hp 100)
    (set! armor 0)
    (set! bullets 60)
    (set! shells 10)
    (set! wsel 0)
    (set! fire_cd 0.0)
    (set! fire_anim 0.0)
    (set! hurt_flash 0.0)
    (set! pain_cd 0.0)
    (set! kills 0)
    (set! level_enemies en_n)
    (set! st_state 1)
    (set! st_timer 0.0)
    (set! msg_text (String.concat "第 " (String.concat (String.atom_to_string (+ li 1)) " 关 · 找到绿色出口")))
    (set! msg_ms 2600)
    (ai_build_walk)              ; AI 可通行表（门/出口算路）
    (set! ai_replan 0.0)
    (set! ai_stuck 0.0)
    (set! ai_detour 0.0)
    (set! ai_ti -1)
    (set! ai_px_prev px)
    (set! ai_py_prev py)
    (update_cam)))

;; ============================================================================
;; 墙体渲染（DDA raycasting）
;; ============================================================================

(define zbuf (mklist NCOL))

(define render_walls
  (lambda ()
    (define col 0)
    (define camx 0.0) (define rdx 0.0) (define rdy 0.0)
    (define ix 0) (define iy 0)
    (define ddx 0.0) (define ddy 0.0) (define sdx 0.0) (define sdy 0.0)
    (define stx 0) (define sty 0)
    (define side 0) (define guard 0) (define hit 0)
    (define t 0) (define idx 0) (define open 0.0)
    (define perp 0.0) (define lh 0.0) (define ytop 0.0) (define ybot 0.0)
    (define shade 0.0) (define u 0.0) (define r 0) (define g 0) (define b 0)
    (define xcol 0) (define hpx 0) (define k 0) (define yl 0) (define bh 0)
    (set! col 0)
    (while (< col NCOL) {
      (set! camx (- (/ (* 2.0 col) NCOL) 1.0))
      (set! rdx (+ pdirx (* pplanex camx)))
      (set! rdy (+ pdiry (* pplaney camx)))
      (set! ix (Math.floor px))
      (set! iy (Math.floor py))
      (if (== rdx 0.0) (set! ddx 1000000.0) (set! ddx (Math.abs (/ 1.0 rdx))))
      (if (== rdy 0.0) (set! ddy 1000000.0) (set! ddy (Math.abs (/ 1.0 rdy))))
      (if (< rdx 0.0) { (set! stx -1) (set! sdx (* (- px ix) ddx)) }
                     { (set! stx 1) (set! sdx (* (- (+ ix 1.0) px) ddx)) })
      (if (< rdy 0.0) { (set! sty -1) (set! sdy (* (- py iy) ddy)) }
                     { (set! sty 1) (set! sdy (* (- (+ iy 1.0) py) ddy)) })
      (set! side 0) (set! hit 0) (set! guard 0) (set! idx 0)
      (while (== hit 0) {
        (if (< sdx sdy) { (set! sdx (+ sdx ddx)) (set! ix (+ ix stx)) (set! side 0) }
                        { (set! sdy (+ sdy ddy)) (set! iy (+ iy sty)) (set! side 1) })
        (set! guard (+ guard 1))
        (if (or (< ix 0) (or (>= ix MW) (or (< iy 0) (>= iy MH))))
            (set! t 1)
            (set! t (get_item map (+ (* iy MW) ix))))
        (if (== t 0) #f
            (if (== t 9) #f
                (if (== t 6)
                    (if (>= (get_item doorf (+ (* iy MW) ix)) 0.999) #f
                        { (set! hit 1) (set! idx (+ (* iy MW) ix)) })
                    { (set! hit 1) (set! idx (+ (* iy MW) ix)) })))
        (if (> guard 96) (set! hit 1))
      })
      (if (== side 0) (set! perp (- sdx ddx)) (set! perp (- sdy ddy)))
      (if (< perp 0.08) (set! perp 0.08))
      (set_item! zbuf col perp)
      ;; 屏幕上的墙高（1 格高 = VH/perp 像素）
      (set! lh (/ (* WK VH) perp))
      (set! ytop (- (/ VH 2.0) (/ lh 2.0)))
      (set! ybot (+ (/ VH 2.0) (/ lh 2.0)))
      ;; 门：随开启进度下沉（可见部分为上半段）
      (set! open 0.0)
      (if (== t 6) (set! open (get_item doorf idx)))
      (if (> open 0.001) { (set! ybot (- ytop (* lh (- 1.0 open)))) })
      ;; 亮度：距离雾 × 南北墙变暗 × 砖缝
      (set! shade (/ 1.0 (+ 1.0 (* perp 0.13))))
      (if (< shade 0.34) (set! shade 0.34))
      (if (== side 1) (set! shade (* shade 0.78)))
      (if (== side 0) (set! u (- (+ px (* perp rdx)) (Math.floor (+ px (* perp rdx)))))
                      (set! u (- (+ py (* perp rdy)) (Math.floor (+ py (* perp rdy))))))
      (if (< (mod (* u 3.0) 1.0) 0.17) (set! shade (* shade 0.74)))
      ;; 基色
      (set! k (* (- t 1) 3))
      (set! r (Math.floor (* (get_item PAL k) shade)))
      (set! g (Math.floor (* (get_item PAL (+ k 1)) shade)))
      (set! b (Math.floor (* (get_item PAL (+ k 2)) shade)))
      ;; 裁剪到视口
      (set! xcol (* col COLW))
      (if (< ytop 0.0) (set! ytop 0.0))
      (if (> ybot VH) (set! ybot VH))
      (set! hpx (- ybot ytop))
      (if (> hpx 0.5) {
        (Gfx.fill_rect xcol ytop COLW hpx r g b)
        ;; 砖缝：3 条横线（墙足够高时才画，远处省掉）
        (if (> hpx 22.0) {
          (set! bh (/ hpx 4.0))
          (set! k 1)
          (while (< k 4) {
            (set! yl (+ ytop (* bh k)))
            (Gfx.fill_rect xcol yl COLW 1 (Math.floor (* r 0.46)) (Math.floor (* g 0.46)) (Math.floor (* b 0.46)))
            (Gfx.fill_rect xcol (+ yl 2) COLW 1 (Math.floor (* (+ r 22) 1.0)) (Math.floor (* (+ g 22) 1.0)) (Math.floor (* (+ b 22) 1.0)))
            (set! k (+ k 1))
          })
          ;; 顶部高光（一条 2px 亮边 → 墙面有厚度感）
          (Gfx.fill_rect xcol ytop COLW 2 (minf 255 (Math.floor (* (+ r 46) 1.0))) (minf 255 (Math.floor (* (+ g 46) 1.0))) (minf 255 (Math.floor (* (+ b 46) 1.0))))
        })
      })
      (set! col (+ col 1))
    })))

;; ============================================================================
;; 天花板 / 地板渐变
;; ============================================================================

(define draw_bg
  (lambda ()
    (define k 0) (define y0 0) (define y1 0) (define c 0) (define hh 0)
    (set! hh (/ VH 2.0))
    (set! k 0)
    (while (< k 5) {
      (set! y0 (Math.floor (/ (* hh k) 5.0)))
      (set! y1 (Math.floor (/ (* hh (+ k 1)) 5.0)))
      (set! c (+ 18 (* k 5)))
      (Gfx.fill_rect 0 y0 W (- y1 y0) c (+ c 4) (+ c 12))
      (set! k (+ k 1))
    })
    (set! k 0)
    (while (< k 5) {
      (set! y0 (+ hh (Math.floor (/ (* hh k) 5.0))))
      (set! y1 (+ hh (Math.floor (/ (* hh (+ k 1)) 5.0))))
      (set! c (- 74 (* k 9)))
      (Gfx.fill_rect 0 y0 W (- y1 y0) c (Math.floor (* c 0.80)) (Math.floor (* c 0.58)))
      (set! k (+ k 1))
    })))

;; ============================================================================
;; 精灵（广告牌）绘制：形状用「单位方框内的矩形表」描述，7 个数一块：
;;   x0 x1 y0 y1 r g b
;; ============================================================================

;; 僵尸兵（近战）：头盔/脸/眼/躯干/双臂/双腿/腰带
(define SPR_Z (quote (
  0.31 0.69 0.01 0.12  92 100 76
  0.28 0.72 0.12 0.16  54 60 46
  0.36 0.64 0.16 0.29  128 134 106
  0.385 0.45 0.19 0.235  234 72 56
  0.55 0.615 0.19 0.235  234 72 56
  0.27 0.73 0.29 0.575  68 78 60
  0.455 0.545 0.29 0.575  46 54 42
  0.135 0.27 0.29 0.42  58 68 52
  0.73 0.865 0.29 0.42  58 68 52
  0.14 0.265 0.42 0.60  80 90 70
  0.735 0.86 0.42 0.60  80 90 70
  0.13 0.275 0.575 0.635  124 104 82
  0.725 0.87 0.575 0.635  124 104 82
  0.27 0.73 0.575 0.62  42 46 38
  0.30 0.455 0.62 0.985  54 60 48
  0.545 0.70 0.62 0.985  54 60 48
  0.29 0.465 0.93 0.99  36 40 32
  0.535 0.71 0.93 0.99  36 40 32
)))

;; 火球兵（远程）：红色恶魔
(define SPR_I (quote (
  0.235 0.33 0.00 0.09  104 48 32
  0.67 0.765 0.00 0.09  104 48 32
  0.32 0.68 0.03 0.13  176 78 50
  0.32 0.68 0.13 0.27  154 64 42
  0.375 0.45 0.155 0.20  255 216 64
  0.55 0.625 0.155 0.20  255 216 64
  0.42 0.58 0.225 0.265  255 150 60
  0.25 0.75 0.27 0.60  136 52 38
  0.115 0.25 0.28 0.46  120 44 32
  0.75 0.885 0.28 0.46  120 44 32
  0.10 0.245 0.46 0.615  108 40 30
  0.755 0.90 0.46 0.615  108 40 30
  0.09 0.255 0.60 0.66  150 130 60
  0.745 0.91 0.60 0.66  150 130 60
  0.25 0.75 0.545 0.605  70 28 24
  0.30 0.47 0.60 0.99  106 40 32
  0.53 0.70 0.60 0.99  106 40 32
  0.29 0.48 0.94 0.99  74 30 26
  0.52 0.71 0.94 0.99  74 30 26
)))

;; 重甲兵（高血近战）：更宽更暗，带肩甲
(define SPR_B (quote (
  0.30 0.70 0.00 0.11  104 82 58
  0.27 0.73 0.11 0.15  62 50 36
  0.34 0.66 0.15 0.27  146 126 90
  0.375 0.45 0.18 0.225  255 126 62
  0.55 0.625 0.18 0.225  255 126 62
  0.42 0.58 0.225 0.255  70 56 40
  0.20 0.80 0.27 0.56  88 68 48
  0.42 0.58 0.27 0.56  62 48 34
  0.06 0.20 0.25 0.44  78 60 42
  0.80 0.94 0.25 0.44  78 60 42
  0.09 0.21 0.44 0.60  66 50 36
  0.79 0.91 0.44 0.60  66 50 36
  0.06 0.22 0.575 0.655  110 96 78
  0.78 0.94 0.575 0.655  110 96 78
  0.20 0.80 0.545 0.60  46 36 26
  0.245 0.47 0.60 0.985  68 54 40
  0.53 0.755 0.60 0.985  68 54 40
  0.235 0.48 0.94 0.99  42 34 26
  0.52 0.765 0.94 0.99  42 34 26
)))

;; 尸体（被击杀后留在地上的一摊）
(define SPR_D (quote (
  0.10 0.90 0.74 0.86  74 58 46
  0.18 0.82 0.86 0.94  52 40 32
  0.24 0.44 0.70 0.78  96 62 50
)))

;; 医疗包（红十字符）+ 弹药盒 + 护甲
(define SPR_H (quote (
  0.18 0.82 0.40 0.80  196 196 200
  0.44 0.56 0.46 0.74  214 60 52
  0.26 0.74 0.55 0.66  214 60 52
  0.18 0.82 0.78 0.84  150 150 156
)))
(define SPR_A (quote (
  0.14 0.86 0.44 0.80  96 92 44
  0.20 0.80 0.48 0.56  176 168 60
  0.22 0.78 0.66 0.70  70 66 32
  0.14 0.86 0.78 0.84  70 66 32
)))
(define SPR_R (quote (
  0.24 0.76 0.36 0.44  56 118 122
  0.18 0.82 0.44 0.74  44 96 104
  0.38 0.62 0.48 0.66  96 196 186
  0.18 0.82 0.74 0.82  32 74 78
)))

;; 出口标记：绿色发光柱（踩上去过关）
(define SPR_E (quote (
  0.30 0.70 0.00 0.09  90 240 170
  0.22 0.78 0.09 0.88  28 140 96
  0.34 0.66 0.18 0.80  120 255 190
  0.22 0.78 0.84 0.94  18 74 52
)))

(define SPR_SHAPES (quote (0 0 0)))
(define init_shapes
  (lambda ()
    (define l (cons SPR_E (quote ())))
    (set! l (cons SPR_R l))
    (set! l (cons SPR_A l))
    (set! l (cons SPR_H l))
    (set! l (cons SPR_D l))
    (set! l (cons SPR_B l))
    (set! l (cons SPR_I l))
    (set! l (cons SPR_Z l))
    (set! SPR_SHAPES l)))

;; 精灵表（每帧重建；并行数组 + 深度排序索引）
(define SMAX 40)
(define sl_n 0)
(define sl_d (quote ()))   ; 深度（相机空间 y）
(define sl_t (quote ()))   ; 形状下标（0..6 见 SPR_SHAPES）；7 = 火球；8 = 空
(define sl_x (quote ()))   ; 屏幕左
(define sl_y (quote ()))   ; 屏幕上
(define sl_w (quote ()))
(define sl_h (quote ()))
(define sl_s (quote ()))   ; 额外（闪烁 / 相位）
(define sl_o (mklist SMAX)) ; 排序索引
(define init_sl
  (lambda ()
    (set! sl_d (mklist SMAX)) (set! sl_t (mklist SMAX)) (set! sl_x (mklist SMAX))
    (set! sl_y (mklist SMAX)) (set! sl_w (mklist SMAX)) (set! sl_h (mklist SMAX))
    (set! sl_s (mklist SMAX)) (set! sl_o (mklist SMAX))))

;; 把矩形 (x0..x1, y0..y1) 画到屏幕，按列 z-buffer 裁剪（被墙挡住的部分不画）
(define spr_flush
  (lambda (ca cb x0 x1 y0 y1 r g b)
    (define sx0 (* ca COLW)) (define sx1 (* cb COLW))
    (if (< sx0 x0) (set! sx0 x0))
    (if (> sx1 x1) (set! sx1 x1))
    ;; 坐标钳到屏内：gfx 的 RGB888 慢路径按 uint32 处理入参，负坐标会被当成巨大
    ;; 正数而整块丢弃（RGB565 快路径则会正确裁剪）——两端行为一致只能靠调用方钳位。
    (if (< sx0 0.0) (set! sx0 0.0))
    (if (< y0 0.0) (set! y0 0.0))
    (if (> sx1 W) (set! sx1 W))
    (if (> y1 H) (set! y1 H))
    (if (and (> sx1 sx0) (> y1 y0)) (Gfx.fill_rect sx0 y0 (- sx1 sx0) (- y1 y0) r g b))))

(define spr_rect
  (lambda (x0 y0 x1 y1 r g b depth)
    (define c0 0) (define c1 0) (define c 0) (define run -1)
    (if (< y1 0.0) #f
        (if (> y0 VH) #f
            {
              (if (< y0 0.0) (set! y0 0.0))
              (if (> y1 VH) (set! y1 VH))
              (set! c0 (Math.floor (/ x0 COLW)))
              (set! c1 (Math.floor (/ (- x1 0.02) COLW)))
              (if (< c0 0) (set! c0 0))
              (if (>= c1 NCOL) (set! c1 (- NCOL 1)))
              (set! c c0)
              (while (<= c c1) {
                (if (> (get_item zbuf c) depth)
                    (if (< run 0) (set! run c))
                    (if (>= run 0) { (spr_flush run c x0 x1 y0 y1 r g b) (set! run -1) }))
                (set! c (+ c 1))
              })
              (if (>= run 0) (spr_flush run (+ c1 1) x0 x1 y0 y1 r g b))
            }))))

;; 相机空间变换：世界点 → (深度, 屏幕 x)
(define cam_depth 0.0)
(define cam_sx 0.0)
(define to_camera
  (lambda (wx wy)
    (define rx (- wx px))
    (define ry (- wy py))
    (define det (- (* pplanex pdiry) (* pdirx pplaney)))
    (define inv 0.0)
    (define tx 0.0)
    (if (== det 0.0) (set! det 0.0001))
    (set! inv (/ 1.0 det))
    ;; 标准广告牌变换（与 raycasting 同一套相机基）：
    ;;   transformY(depth) = invDet·(−planeY·rx + planeX·ry)
    ;;   transformX(横向)  = invDet·( dirY·rx  −  dirX·ry )
    ;; 之前把 dir/plane 写反了：depth 实际算成了「侧向偏移」，精灵因此全部错位。
    (set! cam_depth (* inv (- (* pplanex ry) (* pplaney rx))))
    (set! tx (* inv (- (* pdiry rx) (* pdirx ry))))
    ;; 深度接近 0 时（正侧向，理论上到不了）钳一下，避免除零
    (if (< (Math.abs cam_depth) 0.0001)
        (set! cam_depth (if (< cam_depth 0.0) (- 0.0 0.0001) 0.0001)))
    (set! cam_sx (* (/ W 2.0) (+ 1.0 (/ tx cam_depth))))))

;; 收集本帧可见精灵
(define gather_sprites
  (lambda ()
    (define i 0) (define t 0) (define d 0.0) (define bw 0.0) (define bh 0.0)
    (define cx 0.0) (define yfloor 0.0) (define kk 0)
    (set! sl_n 0)
    ;; 敌人
    (set! i 0)
    (while (< i en_n) {
      (set! d (dist2d (get_item en_x i) (get_item en_y i) px py))
      (if (< d 22.0) {
        (if (>= d 0.05) {
          (to_camera (get_item en_x i) (get_item en_y i))
          (if (> cam_depth 0.15) {
            (if (and (> cam_sx -80.0) (< cam_sx (+ W 80.0))) {
              (set! kk (get_item en_k i))
              (set! t (if (== kk 0) 0 (if (== kk 1) 1 (if (== kk 2) 2 0))))
              (set! bh (/ (* (* WK VH) (if (== kk 2) 1.06 (if (== kk 1) 0.94 0.90))) cam_depth))
              (set! bw (* bh (if (== kk 2) 0.62 (if (== kk 1) 0.54 0.48))))
              (set! cx (/ (* WK VH) cam_depth))
              (set! yfloor (+ (/ VH 2.0) (* 0.5 cx)))
              (if (== (get_item en_st i) 2) {
                (set! bh (* bh 1.06))
                (set! bw (* bw 1.06))
              })
              (if (== (get_item en_st i) 4) {
                (set! t 3)
                (set! bh (* bh 0.42))
                (set! bw (* bh 1.7))
              })
              ;; 走路上下轻微起伏（死亡不动）
              (if (< (get_item en_st i) 4) {
                (set! yfloor (+ yfloor (* bh (* 0.02 (Math.sin (* (get_item en_ph i) 7.0))))))
              })
              (if (< sl_n SMAX) {
                (set_item! sl_d sl_n cam_depth)
                (set_item! sl_t sl_n t)
                (set_item! sl_x sl_n (- cam_sx (/ bw 2.0)))
                (set_item! sl_y sl_n (- yfloor bh))
                (set_item! sl_w sl_n bw)
                (set_item! sl_h sl_n bh)
                (set_item! sl_s sl_n (get_item en_fl i))
                (set! sl_n (+ sl_n 1))
              })
            })
          })
        })
      })
      (set! i (+ i 1))
    })
    ;; 道具
    (set! i 0)
    (while (< i it_n) {
      (if (== (get_item it_al i) 1) {
        (to_camera (get_item it_x i) (get_item it_y i))
        (if (> cam_depth 0.25) {
          (if (and (> cam_sx -40.0) (< cam_sx (+ W 40.0))) {
            (set! kk (get_item it_k i))
            (set! t (if (== kk 10) 4 (if (== kk 11) 5 6)))
            (set! bh (/ (* (* WK VH) 0.26) cam_depth))
            (set! bw (* bh 1.15))
            (set! cx (/ (* WK VH) cam_depth))
            (set! yfloor (+ (/ VH 2.0) (* 0.40 cx)))
            ;; 缓慢上下浮动
            (set! yfloor (+ yfloor (* bh (* 0.06 (Math.sin (* 0.004 (Diag.ms)))))))
            (if (< sl_n SMAX) {
              (set_item! sl_d sl_n cam_depth)
              (set_item! sl_t sl_n t)
              (set_item! sl_x sl_n (- cam_sx (/ bw 2.0)))
              (set_item! sl_y sl_n (- yfloor bh))
              (set_item! sl_w sl_n bw)
              (set_item! sl_h sl_n bh)
              (set_item! sl_s sl_n 0.0)
              (set! sl_n (+ sl_n 1))
            })
          })
        })
      })
      (set! i (+ i 1))
    })
    ;; 出口标记
    (to_camera exit_x exit_y)
    (if (> cam_depth 0.2) {
      (if (< sl_n SMAX) {
        (set! bh (/ (* (* WK VH) 0.95) cam_depth))
        (set! bw (* bh 0.56))
        (set! cx (/ (* WK VH) cam_depth))
        (set! kk (+ (/ VH 2.0) (* 0.5 cx)))
        (set_item! sl_d sl_n cam_depth)
        (set_item! sl_t sl_n 8)
        (set_item! sl_x sl_n (- cam_sx (/ bw 2.0)))
        (set_item! sl_y sl_n (- kk bh))
        (set_item! sl_w sl_n bw)
        (set_item! sl_h sl_n bh)
        (set_item! sl_s sl_n (* 0.5 (+ 0.5 (Math.sin (* 0.003 (Diag.ms))))))
        (set! sl_n (+ sl_n 1))
      })
    })
    ;; 火球
    (set! i 0)
    (while (< i pr_n) {
      (if (== (get_item pr_al i) 1) {
        (to_camera (get_item pr_x i) (get_item pr_y i))
        (if (> cam_depth 0.2) {
          (if (< sl_n SMAX) {
            (set_item! sl_d sl_n cam_depth)
            (set_item! sl_t sl_n 7)
            (set_item! sl_x sl_n cam_sx)
            (set_item! sl_y sl_n (+ (/ VH 2.0) (* 0.06 (/ (* WK VH) cam_depth))))
            (set_item! sl_w sl_n (* (/ VH 0.30) cam_depth))
            (set_item! sl_h sl_n 0.0)
            (set_item! sl_s sl_n 0.0)
            (set! sl_n (+ sl_n 1))
          })
        })
      })
      (set! i (+ i 1))
    })))

;; 深度排序（远 → 近；插入排序，最多 40 项）
(define sort_sprites
  (lambda ()
    (define i 0) (define j 0) (define v 0) (define d 0.0)
    (set! i 0)
    (while (< i sl_n) {
      (set_item! sl_o i i)
      (set! i (+ i 1))
    })
    (set! i 1)
    (while (< i sl_n) {
      (set! v (get_item sl_o i))
      (set! d (get_item sl_d v))
      (set! j (- i 1))
      (while (and (>= j 0) (< (get_item sl_d (get_item sl_o j)) d)) {
        (set_item! sl_o (+ j 1) (get_item sl_o j))
        (set! j (- j 1))
      })
      (set_item! sl_o (+ j 1) v)
      (set! i (+ i 1))
    })))

(define draw_sprites
  (lambda ()
    (define i 0) (define k 0) (define t 0) (define shp 0) (define n 0) (define j 0)
    (define bx 0.0) (define by 0.0) (define bw 0.0) (define bh 0.0)
    (define d 0.0) (define sh 0.0) (define fl 0.0) (define r 0) (define g 0) (define b 0)
    (define cx 0.0) (define cy 0.0) (define rad 0.0) (define col 0)
    (sort_sprites)
    (set! i 0)
    (while (< i sl_n) {
      (set! k (get_item sl_o i))
      (set! t (get_item sl_t k))
      (set! d (get_item sl_d k))
      (set! bx (get_item sl_x k))
      (set! by (get_item sl_y k))
      (set! bw (get_item sl_w k))
      (set! bh (get_item sl_h k))
      (if (== t 7) {
        ;; 火球：双层圆（外焰 + 白热核心），按列 z-buffer 判遮挡
        (set! col (Math.floor (/ bx COLW)))
        (if (and (>= col 0) (< col NCOL))
            (if (> (get_item zbuf col) d) {
              (set! rad (/ (* (* WK VH) 0.13) d))
              (if (< rad 1.5) (set! rad 1.5))
              (Gfx.fill_circle bx by rad 236 120 40)
              (Gfx.fill_circle bx by (* rad 0.55) 255 224 120)
            }))
      })
      (if (or (< t 7) (== t 8)) {
        (if (== t 8) (set! t 7))
        (set! shp (get_item SPR_SHAPES t))
        (set! n (Math.floor (/ (length shp) 7.0)))
        (set! sh (/ 1.0 (+ 1.0 (* d 0.20))))
        (if (< sh 0.30) (set! sh 0.30))
        (set! fl (get_item sl_s k))
        (set! j 0)
        (while (< j n) {
          (set! cx (* 7 j))
          (set! r (Math.floor (* (get_item shp (+ cx 4)) sh)))
          (set! g (Math.floor (* (get_item shp (+ cx 5)) sh)))
          (set! b (Math.floor (* (get_item shp (+ cx 6)) sh)))
          (if (> fl 0.0) {
            (set! r (minf 255 (Math.floor (* (+ r 130) 1.0))))
            (set! g (minf 255 (Math.floor (* (+ g 90) 1.0))))
            (set! b (minf 255 (Math.floor (* (+ b 80) 1.0))))
          })
          (spr_rect
            (+ bx (* bw (get_item shp cx)))
            (+ by (* bh (get_item shp (+ cx 2))))
            (+ bx (* bw (get_item shp (+ cx 1))))
            (+ by (* bh (get_item shp (+ cx 3))))
            r g b d)
          (set! j (+ j 1))
        })
      })
      (set! i (+ i 1))
    })))

;; ============================================================================
;; 武器视图
;; ============================================================================

(define gun_x 0)   ; 枪身中心 x
(define gun_y 0)   ; 枪口底部 y
(define rec 0)     ; 后坐力像素

(define draw_weapon
  (lambda ()
    (define cx (/ W 2.0))
    ;; 走动时轻微上下摆动（Doom 的武器晃动），开枪时后坐
    (set! rec (Math.floor (* 16.0 fire_anim)))
    (set! gun_x (Math.floor (+ cx (* 2.0 (Math.sin (* walk_ph 4.0))))))
    ;; 基准线放在视口底沿上方一点：握把/双手都要露在 HUD 之上（HUD 后画，会盖住下缘）
    (set! gun_y (Math.floor (+ (+ VH -4) (+ rec (* 3.0 (Math.cos (* walk_ph 8.0)))))))
    (if (== wsel 1) (draw_shotgun 1))
    (if (== wsel 2) (draw_chaingun 1))
    (if (== wsel 0) (draw_pistol 1))))

;; 手枪：套筒 + 枪管 + 机匣 + 木握把 + 双手（右手为主）
(define draw_pistol
  (lambda (dummy)
    (define cx gun_x) (define by gun_y)
    (Gfx.fill_rect (- cx 11) (- by 78) 22 46 76 78 88)        ; 套筒
    (Gfx.fill_rect (- cx 11) (- by 78) 22 5 124 128 142)      ; 套筒顶面高光
    (Gfx.fill_rect (- cx 11) (- by 38) 22 5 52 54 62)         ; 套筒下沿阴影
    (Gfx.fill_rect (- cx 5) (- by 84) 10 8 40 42 50)          ; 枪口
    (Gfx.fill_rect (- cx 9) (- by 33) 18 16 62 64 74)         ; 机匣
    (Gfx.fill_rect (- cx 7) (- by 19) 14 8 92 86 70)          ; 扳机护圈
    (Gfx.fill_rect (- cx 8) (- by 16) 16 24 116 86 58)        ; 木握把
    (Gfx.fill_rect (- cx 6) (- by 12) 6 16 138 104 72)        ; 握把高光
    (Gfx.fill_rect (- cx 18) (- by 12) 18 26 158 118 92)      ; 右手掌
    (Gfx.fill_rect (- cx 14) (- by 2) 14 16 140 102 78)       ; 右手指
    (Gfx.fill_rect (- cx 30) (- by 44) 16 30 150 112 86)      ; 左手（扶握把）
    (Gfx.fill_rect (- cx 28) (- by 40) 12 22 132 96 74)
    (if (> fire_anim 0.02) (draw_flash cx (- by 88) 1))))

;; 霰弹枪：双管 + 木质护木 + 枪托（更宽更重）
(define draw_shotgun
  (lambda (dummy)
    (define cx gun_x) (define by gun_y)
    (Gfx.fill_rect (- cx 19) (- by 100) 38 20 96 100 112)     ; 上管
    (Gfx.fill_rect (- cx 19) (- by 78) 38 20 88 92 104)       ; 下管
    (Gfx.fill_rect (- cx 19) (- by 100) 38 5 140 146 158)     ; 高光
    (Gfx.fill_rect (- cx 15) (- by 80) 30 5 52 54 62)         ; 管间阴影
    (Gfx.fill_rect (- cx 21) (- by 68) 42 12 74 78 90)        ; 管箍
    (Gfx.fill_rect (- cx 14) (- by 56) 28 26 126 88 50)       ; 木护木
    (Gfx.fill_rect (- cx 12) (- by 52) 6 18 150 108 66)       ; 木纹高光
    (Gfx.fill_rect (- cx 10) (- by 30) 20 24 96 66 38)        ; 枪身
    (Gfx.fill_rect (- cx 17) (- by 8) 34 14 152 112 74)       ; 枪托
    (Gfx.fill_rect (- cx 16) (- by 6) 32 5 176 134 92)
    (Gfx.fill_rect (- cx 30) (- by 46) 16 30 158 118 92)      ; 左手
    (Gfx.fill_rect (- cx 28) (- by 42) 12 22 136 100 78)
    (Gfx.fill_rect (+ cx 10) (- by 16) 18 26 142 104 80)      ; 右手
    (if (> fire_anim 0.02) (draw_flash cx (- by 104) 2))))

;; 链枪：三根转管 + 机匣 + 弹链
(define draw_chaingun
  (lambda (dummy)
    (define cx gun_x) (define by gun_y)
    (define i -16)
    (while (<= i 16) {
      (Gfx.fill_rect (+ cx i) (- by 92) 10 46 84 88 100)
      (Gfx.fill_rect (+ cx i) (- by 92) 10 4 134 140 152)
      (Gfx.fill_rect (+ cx i) (- by 50) 10 4 56 58 68)
      (set! i (+ i 16))
    })
    (Gfx.fill_rect (- cx 25) (- by 44) 50 18 72 76 88)        ; 机匣
    (Gfx.fill_rect (- cx 22) (- by 26) 44 20 60 62 72)
    (Gfx.fill_rect (- cx 25) (- by 44) 50 4 118 124 136)
    (Gfx.fill_rect (- cx 18) (- by 12) 36 12 96 92 60)        ; 弹链盒
    (Gfx.fill_rect (- cx 34) (- by 40) 16 30 158 118 92)      ; 左手
    (Gfx.fill_rect (+ cx 16) (- by 20) 18 28 142 104 80)      ; 右手
    (if (> fire_anim 0.02) (draw_flash cx (- by 96) 3))))

;; 枪口焰（size 越大越夸张）
(define draw_flash
  (lambda (cx cy size)
    (define s (* 6 size))
    (Gfx.fill_circle cx cy s 255 226 120)
    (Gfx.fill_circle cx cy (* s 0.55) 255 255 220)
    (Gfx.fill_rect (- cx (* s 1.6)) (- cy 2) (* s 3.2) 4 255 200 90)
    (Gfx.fill_rect (- cx 2) (- cy (* s 1.6)) 4 (* s 3.2) 255 200 90)))

;; ============================================================================
;; HUD
;; ============================================================================

;; 文本像素宽估算：12px 字模实测 CJK/全角步进 12px、ASCII 6px。
;; （按"字符数 × 6"居中在混合中英文时会把整行推偏，标题页实测会被裁掉左边。）
(define text_w
  (lambda (s units)
    (define n 0) (define i 0) (define w 0) (define c 0)
    (set! n (String.length s))
    (while (< i n) {
      (set! c (String.charCodeAt i s))
      (if (>= c 8192) (set! w (+ w units)) (set! w (+ w (/ units 2))))
      (set! i (+ i 1))
    })
    w))

(define draw_text_c
  (lambda (s cx y r g b)
    (Gfx.draw_text s (- cx (/ (text_w s 12) 2)) y r g b)))

;; 16px 字模（CJK 16 / ASCII 8）的居中文本
(define draw_text_c16
  (lambda (s cx y r g b)
    (Gfx.draw_text s (- cx (/ (text_w s 16) 2)) y r g b)))

(define draw_panel
  (lambda (x w)
    (Gfx.fill_rect x (+ VH 2) (- w 2) (- HUD_H 4) 40 38 36)
    (Gfx.draw_rect x (+ VH 2) (- w 2) (- HUD_H 4) 74 68 58)))

(define ammo_now
  (lambda ()
    (if (== wsel 1) shells (if (== wsel 2) bullets 999))))

(define draw_hud
  (lambda ()
    (define y0 VH)
    (define yl (+ VH 4)) (define yv (+ VH 17))
    (Gfx.fill_rect 0 y0 W HUD_H 22 20 19)
    (Gfx.fill_rect 0 y0 W 2 96 84 66)
    (draw_panel 2 77)
    (draw_panel 81 77)
    (draw_panel 160 77)
    (draw_panel 239 79)
    ;; 生命（AI 自动游玩时面板上打「AI」角标）
    (Gfx.draw_text "生命" 6 yl 196 186 168)
    (if (== ai_on 1) {
      (Gfx.fill_rect 46 yl 18 12 40 110 74)
      (Gfx.draw_text "AI" 49 (+ yl 1) 235 244 236)
    })
    (Gfx.draw_text (String.atom_to_string hp) 6 yv (if (< hp 31) 236 96 92) 226 214)
    ;; 弹药
    (Gfx.draw_text (if (== wsel 0) "弹药·手枪" (if (== wsel 1) "弹药·霰弹" "弹药·链枪")) 85 yl 196 186 168)
    (Gfx.draw_text (if (== wsel 0) "∞" (String.atom_to_string (ammo_now))) 85 yv 240 216 120)
    ;; 护甲
    (Gfx.draw_text "护甲" 164 yl 196 186 168)
    (Gfx.draw_text (String.atom_to_string armor) 164 yv 120 196 210)
    ;; 击杀 / 关卡 / FPS
    (Gfx.draw_text "击杀" 243 yl 196 186 168)
    (Gfx.draw_text (String.concat (String.atom_to_string kills) "/") 243 yv 236 214 160)
    (Gfx.draw_text (String.atom_to_string level_enemies) 267 yv 200 190 170)
    (Gfx.draw_text (String.concat "L" (String.atom_to_string (+ lv 1))) 288 yl 196 186 168)
    (Gfx.draw_text (String.atom_to_string fps_v) 288 yv 150 170 180)))

;; ============================================================================
;; 触屏层（虚拟摇杆 / 按钮 / 十字准星 / 提示）
;; ============================================================================

(define FIREx 286) (define FIREy (- VH 32)) (define FIREr 24)
(define USEx 240) (define USEy (- VH 26)) (define USEr 17)
(define SWX 240) (define SWY (- VH 70)) (define SWr 15)
(define AFx 6) (define AFy 26) (define AFw 60) (define AFh 16)     ; 自动开火
(define AIBx 70) (define AIBy 26) (define AIBw 56) (define AIBh 16) ; AI 手动/自动

(define draw_crosshair
  (lambda ()
    (define cx (/ W 2.0)) (define cy (/ VH 2.0))
    (Gfx.fill_rect (- cx 7) (- cy 1) 5 2 210 226 190)
    (Gfx.fill_rect (+ cx 2) (- cy 1) 5 2 210 226 190)
    (Gfx.fill_rect (- cx 1) (- cy 7) 2 5 210 226 190)
    (Gfx.fill_rect (- cx 1) (+ cy 2) 2 5 210 226 190)))

(define draw_btn
  (lambda (cx cy r label r0 g0 b0)
    (Gfx.fill_circle cx cy r r0 g0 b0)
    (Gfx.fill_circle cx cy (- r 2) (Math.floor (* r0 0.62)) (Math.floor (* g0 0.62)) (Math.floor (* b0 0.62)))
    (draw_text_c label cx (- cy 6) 246 240 232)))

(define draw_touch_ui
  (lambda ()
    ;; 自动开火 / AI 自动游玩 两枚开关
    (if (== auto_fire 1) {
      (Gfx.fill_rect AFx AFy AFw AFh 40 110 74)
      (Gfx.draw_rect AFx AFy AFw AFh 96 190 140)
      (Gfx.draw_text "自动开火" (+ AFx 3) (+ AFy 2) 235 244 236)
    }
    {
      (Gfx.fill_rect AFx AFy AFw AFh 46 46 50)
      (Gfx.draw_rect AFx AFy AFw AFh 12 12 16)
      (Gfx.draw_text "自动开火" (+ AFx 3) (+ AFy 2) 150 152 158)
    })
    (if (== ai_on 1) {
      (Gfx.fill_rect AIBx AIBy AIBw AIBh 40 110 74)
      (Gfx.draw_rect AIBx AIBy AIBw AIBh 96 190 140)
      (Gfx.draw_text "AI 自动" (+ AIBx 3) (+ AIBy 2) 235 244 236)
    }
    {
      (Gfx.fill_rect AIBx AIBy AIBw AIBh 46 46 50)
      (Gfx.draw_rect AIBx AIBy AIBw AIBh 12 12 16)
      (Gfx.draw_text "AI 手动" (+ AIBx 3) (+ AIBy 2) 150 152 158)
    })
    (draw_btn FIREx FIREy FIREr "开火" 168 54 48)
    (draw_btn USEx USEy USEr "开门" 62 86 130)
    (draw_btn SWX SWY SWr "换枪" 96 84 52)
    ;; 摇杆
    (if (== t_mode 1) {
      (Gfx.draw_circle t_ox t_oy 30 120 190 220)
      (Gfx.fill_circle (+ t_ox (Math.floor (* joy_dx 26.0))) (+ t_oy (Math.floor (* joy_dy 26.0))) 11 200 220 230)
    })
    ;; 右半屏拖动转向时的辅助刻度
    (if (== t_mode 2) {
      (Gfx.fill_rect 0 (- (/ VH 2.0) 1) 10 2 90 140 170)
      (Gfx.fill_rect (- W 10) (- (/ VH 2.0) 1) 10 2 90 140 170)
    })))

;; 受伤红屏（边角红晕，比整屏覆盖便宜且不挡视线）
(define draw_hurt
  (lambda ()
    (define a 0)
    (if (> hurt_flash 0.0) {
      (set! a (Math.floor (* 160.0 (minf 1.0 hurt_flash))))
      (Gfx.fill_rect 0 0 W 6 a 20 18)
      (Gfx.fill_rect 0 (- VH 6) W 6 a 20 18)
      (Gfx.fill_rect 0 0 6 VH a 20 18)
      (Gfx.fill_rect (- W 6) 0 6 VH a 20 18)
    })))

;; 提示条
(define draw_msg
  (lambda ()
    (define wpx 0)
    (if (> msg_ms 0)
        (if (not (== msg_text "")) {
          (set! wpx (+ (* 12 (String.length msg_text)) 12))
          (Gfx.fill_rect (- (/ W 2.0) (/ wpx 2)) 4 wpx 16 20 24 30)
          (draw_text_c msg_text (/ W 2.0) 6 236 226 196)
        }))))

;; ============================================================================
;; 场景渲染主入口
;; ============================================================================

(define draw_world
  (lambda ()
    (draw_bg)
    (render_walls)
    (gather_sprites)
    (draw_sprites)
    (set! dbg_ns sl_n)
    (draw_weapon)
    (draw_crosshair)
    (draw_hurt)
    (draw_touch_ui)
    (draw_msg)))

;; ============================================================================
;; 标题 / 结算画面
;; ============================================================================

(define draw_title
  (lambda ()
    ;; 背景直接渲染玩家出生点看到的那一眼（引擎自展示：天花板 + 走廊 + 远墙），
    ;; 再叠两块文字面板。标题页不画 HUD，因此可用满 240 的高度。
    (draw_bg)
    (render_walls)
    (Gfx.fill_rect 0 92 W 66 8 6 10)          ; 标题面板
    (Gfx.fill_rect 0 92 W 2 190 60 40)
    (Gfx.fill_rect 0 158 W 2 190 60 40)
    (Gfx.set_font 2)
    (draw_text_c16 "灵机 DOOM" (/ W 2.0) 96 226 90 60)
    (Gfx.set_font 1)
    (draw_text_c "RAYCASTING 第一人称射击" (/ W 2.0) 118 178 168 158)
    (draw_text_c "3 关 · 3 种怪物 · 3 把枪 · 单指可玩" (/ W 2.0) 136 150 150 160)
    (Gfx.fill_rect 0 160 W 80 12 10 14)       ; 操作提示面板
    (draw_text_c "触屏：左半屏拖动=移动 · 右半屏拖动=转向" (/ W 2.0) 164 196 196 206)
    (draw_text_c "轻点右半屏 / 右下角按钮=开火 · T 自动开火" (/ W 2.0) 180 196 196 206)
    (draw_text_c "键盘：WASD 走 · ←→ 转 · 空格开火 · E 开门" (/ W 2.0) 196 158 160 172)
    (draw_text_c "G = AI 自动游玩（自己寻路/交战/开门/拾取）" (/ W 2.0) 212 176 200 176)
    (Gfx.fill_rect (- (/ W 2.0) 72) 214 144 20 150 52 40)
    (Gfx.fill_rect (- (/ W 2.0) 72) 214 144 2 226 128 80)
    (draw_text_c "点  击  开  始" (/ W 2.0) 217 250 240 232)))

(define draw_overlay
  (lambda ()
    (define title "")
    (define sub "")
    (Gfx.fill_rect 40 (- (/ VH 2.0) 44) (- W 80) 88 18 14 16)
    (Gfx.draw_rect 40 (- (/ VH 2.0) 44) (- W 80) 88 190 70 50)
    (if (== st_state 2) { (set! title "你  死  了") (set! sub "点击重来本关") })
    (if (== st_state 3) { (set! title "关卡完成") (set! sub "点击进入下一关") })
    (if (== st_state 4) { (set! title "全  部  通  关") (set! sub "点击重新开始") })
    (draw_text_c title (/ W 2.0) (- (/ VH 2.0) 24) 240 200 90)
    (draw_text_c sub (/ W 2.0) (- (/ VH 2.0) 4) 200 200 200)
    (draw_text_c (String.concat "击杀 " (String.atom_to_string total_kills)) (/ W 2.0) (+ (/ VH 2.0) 14) 170 180 190)
    (draw_text_c (String.concat "用时 " (String.concat (String.atom_to_string (Math.floor (/ game_ms 1000))) " 秒"))
                 (/ W 2.0) (+ (/ VH 2.0) 30) 150 160 172)))

;; ============================================================================
;; 主渲染
;; ============================================================================

(define render
  (lambda ()
    (if (== st_state 0) (draw_title))
    (if (> st_state 0) {
      (draw_world)
      (draw_hud)
      (if (> st_state 1) (draw_overlay))
    })
    (EX.draw_button)
    (Gfx.refresh)))

;; ============================================================================
;; 输入：触屏（单点）+ 物理键盘
;; ============================================================================

(define t_down 0)        ; 触屏是否在按下中（本 App 的状态机）
(define t_mode 0)        ; 0 无 1 摇杆 2 转向 3 按钮
(define t_ox 0) (define t_oy 0)     ; 摇杆原点

(define t_lx 0) (define t_ly 0)     ; 转向上一位置
(define t_move 0.0)      ; 本次触摸累计位移（判定轻点开火）
(define t_start 0)       ; 按下时刻
(define joy_dx 0.0) (define joy_dy 0.0)
(define btn 0)           ; 按下的按钮：1 开火 2 开门 3 换枪 4 自动开火
(define want_fire 0)
(define want_use 0)
(define want_swap 0)
(define kb_fwd 0) (define kb_back 0) (define kb_left 0) (define kb_right 0)
(define kb_turnl 0) (define kb_turnr 0)
(define turn_drag 0.0)   ; 本拍触屏转向角（度）
(define turn_in 0)       ; 本拍是否有人为转向输入（辅助瞄准据此让路）

(define hit_rect
  (lambda (x y rx ry rw rh)
    (if (>= x rx) (if (< x (+ rx rw)) (if (>= y ry) (if (< y (+ ry rh)) 1 0) 0) 0) 0)))

(define hit_circle
  (lambda (x y cx cy r)
    (define dx (- x cx)) (define dy (- y cy))
    (if (<= (+ (* dx dx) (* dy dy)) (* r r)) 1 0)))

(define touch_begin
  (lambda (x y)
    (set! tap_now 1)
    ;; AI 自动运行中：除「AI 开关」外任意位置按下 → 立刻交还手动（这一下照常生效）
    (if (== ai_on 1)
        (if (== (hit_rect x y AIBx AIBy AIBw AIBh) 0) (ai_set 0)))
    (set! t_down 1)
    (set! t_move 0.0)
    (set! t_start now_ms)
    (set! btn 0)
    (set! t_mode 0)
    (if (== (hit_circle x y FIREx FIREy (+ FIREr 6)) 1) { (set! btn 1) (set! t_mode 3) })
    (if (and (== t_mode 0) (== (hit_circle x y USEx USEy (+ USEr 6)) 1)) { (set! btn 2) (set! t_mode 3) })
    (if (and (== t_mode 0) (== (hit_circle x y SWX SWY (+ SWr 6)) 1)) { (set! btn 3) (set! t_mode 3) })
    (if (and (== t_mode 0) (== (hit_rect x y AFx AFy AFw AFh) 1)) { (set! btn 4) (set! t_mode 3) })
    (if (and (== t_mode 0) (== (hit_rect x y AIBx AIBy AIBw AIBh) 1)) { (set! btn 5) (set! t_mode 3) })
    (if (and (== t_mode 0) (< x 150)) {
      (set! t_mode 1)
      (set! t_ox x) (set! t_oy y)

      (set! joy_dx 0.0) (set! joy_dy 0.0)
    })
    (if (== t_mode 0) {
      (set! t_mode 2)
      (set! t_lx x) (set! t_ly y)
    })
    ;; 按钮的即时动作
    (if (== btn 2) (set! want_use 1))
    (if (== btn 3) (set! want_swap 1))
    (if (== btn 4) (set! auto_fire (if (== auto_fire 1) 0 1)))
    (if (== btn 5) (ai_set (if (== ai_on 1) 0 1)))
    (if (== btn 1) (set! want_fire 1))))

(define touch_move
  (lambda (x y)
    (set! t_move (+ t_move (absf (- x t_lx))))
    (if (== t_mode 1) {
      (set! joy_dx (clampf (/ (- x t_ox) 34.0) (- 0.0 1.0) 1.0))
      (set! joy_dy (clampf (/ (- y t_oy) 34.0) (- 0.0 1.0) 1.0))
    })
    (if (== t_mode 2) {
      (set! turn_drag (+ turn_drag (* (- x t_lx) 0.42)))
      (set! t_lx x) (set! t_ly y)
    })))

(define touch_end
  (lambda (x y)
    (if (== t_mode 2)
        (if (< t_move 9.0)
            (if (< (- now_ms t_start) 320)
                (if (== btn 0) (set! want_fire 1)))))
    (set! t_down 0)
    (set! t_mode 0)
    (set! btn 0)
    (set! joy_dx 0.0)
    (set! joy_dy 0.0)))

(define read_keys
  (lambda ()
    (define d 0) (define guard 0)
    (set! guard 0)
    (while (< guard 8) {
      (set! guard (+ guard 1))
      (set! d (Key.take_press))
      (if (== d 0) (set! guard 8)
          {
            (set! tap_now 1)
            (if (== d 32) (set! want_fire 1))                 ; 空格
            (if (== d 101) (set! want_use 1))                 ; e
            (if (== d 102) (set! want_use 1))                 ; f
            (if (== d 113) (set! want_swap 1))                ; q
            (if (== d 49) (set! want_swap 101))               ; 1 手枪
            (if (== d 50) (set! want_swap 102))               ; 2 霰弹枪
            (if (== d 51) (set! want_swap 103))               ; 3 链枪
            (if (== d 116) (set! auto_fire (if (== auto_fire 1) 0 1)))  ; t 自动开火
            (if (== d 103) (ai_set (if (== ai_on 1) 0 1)))              ; g AI 自动
          })
    })
    (set! kb_fwd (if (or (Key.is_down 119) (Key.is_down 1)) 1 0))
    (set! kb_back (if (or (Key.is_down 115) (Key.is_down 2)) 1 0))
    (set! kb_left (if (Key.is_down 97) 1 0))
    (set! kb_right (if (Key.is_down 100) 1 0))
    (set! kb_turnl (if (Key.is_down 3) 1 0))
    (set! kb_turnr (if (Key.is_down 4) 1 0))))

(define read_input
  (lambda ()
    (define tx 0) (define ty 0)
    (set! want_fire 0)
    (set! want_use 0)
    (set! want_swap 0)
    (set! turn_drag 0.0)
    (set! tx (Touch.x))
    (set! ty (Touch.y))
    (if (== (if (Touch.take_press) 1 0) 1) (touch_begin tx ty))
    (if (== t_down 1) (if (Touch.is_pressed) (touch_move tx ty)))
    (if (== (if (Touch.take_release) 1 0) 1) (touch_end tx ty))
    ;; 按住「开火」按钮 = 持续开火
    (if (and (== t_down 1) (== btn 1)) (set! want_fire 1))
    (read_keys)
    ;; want_swap 101/102/103 = 直接选枪；113(q) = 循环换枪
    (if (== want_swap 101) (set! wsel 0))
    (if (== want_swap 102) (set! wsel 1))
    (if (== want_swap 103) (set! wsel 2))
    (if (== want_swap 113) (set! wsel (if (== wsel 2) 0 (+ wsel 1))))))

;; ============================================================================
;; 游戏逻辑
;; ============================================================================

(define msg_show
  (lambda (s ms)
    (set! msg_text s)
    (set! msg_ms ms)))

;; 开门：看玩家正前方 1.2 格
;; 开门：沿视线在 0.5 / 0.85 / 1.2 格三处采样（贴着门站时单点采样会越过门格）
(define try_use
  (lambda ()
    (define d 0.5)
    (define fx 0.0) (define fy 0.0) (define t 0) (define idx 0) (define ok 0)
    (set! ok 0)
    (while (<= d 1.2001) {
      (set! fx (+ px (* pdirx d)))
      (set! fy (+ py (* pdiry d)))
      (set! t (tile_xy fx fy))
      (if (and (== t 6) (== ok 0)) {
        (set! idx (+ (* (Math.floor fy) MW) (Math.floor fx)))
        (if (< (get_item doorf idx) 0.02) {
          (set_item! doorf idx 0.02)
          (set! ok 1)
        })
      })
      (set! d (+ d 0.35))
    })
    (if (== ok 1) (msg_show "舱门开启" 1200) (msg_show "附近没有门" 800))))

;; 子弹命中判定：返回敌人下标，或 -1
(define hitscan
  (lambda (ax ay rad)
    (define best -1)
    (define bd 999.0)
    (define wd 0.0)
    (define i 0)
    (define relx 0.0) (define rely 0.0)
    (define pr 0.0) (define lat 0.0)
    (define d 0.0)
    (set! wd (ray_wall_dist px py ax ay))
    (set! i 0)
    (while (< i en_n) {
      (if (> (get_item en_hp i) 0) {
        (set! relx (- (get_item en_x i) px))
        (set! rely (- (get_item en_y i) py))
        (set! pr (+ (* relx ax) (* rely ay)))
        (if (and (> pr 0.1) (< pr (+ wd 0.3))) {
          (set! lat (absf (- (* relx ay) (* rely ax))))
          (set! d (dist2d (get_item en_x i) (get_item en_y i) px py))
          (if (< lat (+ rad (* d 0.06)))
              (if (< d bd) {
                (set! bd d)
                (set! best i)
              }))
        })
      })
      (set! i (+ i 1))
    })
    best))

;; 对敌人造成伤害
(define damage_enemy
  (lambda (i dmg)
    (define h 0)
    (if (>= i 0) {
      (set! h (- (get_item en_hp i) dmg))
      (set_item! en_hp i h)
      (set_item! en_fl i 0.10)
      (if (<= h 0) {
        (set_item! en_st i 4)
        (set! kills (+ kills 1))
        (set! total_kills (+ total_kills 1))
        (if (== kills level_enemies) (msg_show "区域肃清 · 前往绿色出口" 2000))
      })
      (if (and (> h 0) (< (get_item en_st i) 2)) (set_item! en_st i 1)))
    }))

;; 开火（hitscan + 弹丸；霰弹枪打 8 发扇形）
(define do_fire
  (lambda ()
    (define cost 0)
    (define dmg 0)
    (define cd 0.0)
    (define pellets 1)
    (define spread 0.0)
    (define i 0)
    (define a 0.0)
    (define ax 0.0) (define ay 0.0)
    (if (== wsel 1) { (set! pellets 8) (set! spread 5.5) (set! dmg 9) (set! cd 0.78) })
    (if (== wsel 2) { (set! pellets 1) (set! spread 2.6) (set! dmg 8) (set! cd 0.10) })
    (if (== wsel 0) { (set! pellets 1) (set! spread 0.8) (set! dmg 15) (set! cd 0.32) })
    ;; 弹药检查
    (set! cost 1)
    (if (== wsel 0) (set! cost 0))
    (if (and (== cost 1) (== (if (== wsel 1) shells bullets) 0))
        {
          (set! fire_cd 0.35)
          (msg_show "弹药耗尽 · 换手枪" 900)
          (set! wsel 0)
        }
        {
          (if (== wsel 1) (set! shells (- shells 1)))
          (if (== wsel 2) (set! bullets (- bullets 1)))
          (set! fire_cd cd)
          (set! fire_anim 0.10)
          (set! shots_n (+ shots_n 1))
          (set! i 0)
          (while (< i pellets) {
            (set! a (+ pang (* spread (- (/ (* 2.0 i) pellets) 1.0))))
            (set! ax (Math.cos (* a D2R)))
            (set! ay (Math.sin (* a D2R)))
            (damage_enemy (hitscan ax ay 0.30) dmg)
            (set! i (+ i 1))
          })
        })))

;; 玩家受伤（护甲吸收 1/3）
(define hurt_player
  (lambda (dmg)
    (define a 0) (define rest 0)
    (if (> pain_cd 0.0) (set! dmg 0))
    (if (> dmg 0) (set! pain_cd 0.24))
    (set! rest dmg)
    (if (> armor 0) {
      (set! a (Math.floor (/ dmg 3.0)))
      (if (> a armor) (set! a armor))
      (set! armor (- armor a))
      (set! rest (- dmg a))
    })
    (set! hp (- hp rest))
    (set! hurt_flash 0.45)
    (if (<= hp 0) {
      (set! hp 0)
      (set! st_state 2)
      (set! st_timer 0.0)
    })))

;; 玩家移动（先 x 后 y 分别尝试，撞墙时沿墙滑）
(define move_player
  (lambda (vx vy)
    (define m 0.0) (define nx 0.0) (define ny 0.0) (define PR 0.26)
    (set! m (Math.sqrt (+ (* vx vx) (* vy vy))))
    (if (> m 1.0) {
      (set! vx (/ vx m))
      (set! vy (/ vy m))
    })
    (if (> m 0.05) (set! walk_ph (+ walk_ph (* dt 2.2))))
    (set! nx (+ px (* (* vx 2.9) dt)))
    (if (== (can_stand nx py PR) 1) (set! px nx))
    (set! ny (+ py (* (* vy 2.9) dt)))
    (if (== (can_stand px ny PR) 1) (set! py ny))))

;; 敌人 AI
(define update_enemies
  (lambda ()
    (define i 0)
    (define k 0) (define st 0) (define d 0.0)
    (define ex 0.0) (define ey 0.0)
    (define dx 0.0) (define dy 0.0) (define n 0.0)
    (define spd 0.0) (define range 0.0)
    (define t 0.0)
    (define nx 0.0) (define ny 0.0)
    (set! i 0)
    (while (< i en_n) {
      (set! st (get_item en_st i))
      (if (< st 4) {
        (set! ex (get_item en_x i))
        (set! ey (get_item en_y i))
        (set! k (get_item en_k i))
        (set! d (dist2d ex ey px py))
        (set! spd (if (== k 2) 1.35 (if (== k 1) 1.55 1.85)))
        (set! range (if (== k 1) 6.5 1.25))
        ;; 受击闪白衰减
        (set_item! en_fl i (maxf 0.0 (- (get_item en_fl i) dt)))
        ;; 待机：看到玩家就转入追击
        (if (== st 0)
            (if (< d 10.5)
                (if (== (los_clear ex ey px py) 1)
                    { (set_item! en_st i 1) (set_item! en_t i 0.0) })))
        (if (== st 1) {
          ;; 追击：远程兵在冷却好了且看得见时先手；近战兵贴近就出手
          (if (and (== k 1) (<= (get_item en_t i) 0.0) (< d range)
                   (== (los_clear ex ey px py) 1))
              { (set_item! en_st i 2) (set_item! en_t i 0.45) }
              {
                (if (and (< d range) (not (== k 1)))
                    { (set_item! en_st i 2) (set_item! en_t i 0.32) }
                    {
                      (set! dx (- px ex)) (set! dy (- py ey))
                      (set! n (Math.sqrt (+ (* dx dx) (* dy dy))))
                      (if (> n 0.001) {
                        (set! dx (/ dx n)) (set! dy (/ dy n))
                        (set! nx (+ ex (* dx (* spd dt))))
                        (set! ny (+ ey (* dy (* spd dt))))
                        (if (== (can_stand nx ey 0.24) 1) (set_item! en_x i nx))
                        (if (== (can_stand (get_item en_x i) ny 0.24) 1) (set_item! en_y i ny))
                        (set_item! en_ph i (+ (get_item en_ph i) (* (* dt spd) 3.0)))
                      })
                      ;; 远程兵在追击途中累积冷却
                      (if (== k 1)
                          (if (> (get_item en_t i) 0.0)
                              (set_item! en_t i (- (get_item en_t i) dt))))
                    })
              })
        })
        (if (== st 2) {
          (set! t (- (get_item en_t i) dt))
          (set_item! en_t i t)
          (if (<= t 0.0) {
            (if (== k 1) {
              ;; 火球兵：吐火球
              (if (< d 12.0) {
                (set! dx (- px ex)) (set! dy (- py ey))
                (set! n (Math.sqrt (+ (* dx dx) (* dy dy))))
                (if (> n 0.001) {
                  (set! dx (/ dx n)) (set! dy (/ dy n))
                  (spawn_projectile (+ ex (* dx 0.45)) (+ ey (* dy 0.45)) dx dy)
                })
              })
            }
            {
              ;; 近战：够近就砍
              (if (< d 1.7) (hurt_player (if (== k 2) 13 7)))
            })
            (set_item! en_st i 1)
            (set_item! en_t i (if (== k 1) 1.60 1.15))
          })
        })
        (if (== st 3) {
          (set! t (- (get_item en_t i) dt))
          (set_item! en_t i t)
          (if (<= t 0.0) (set_item! en_st i 1))
        })
      })
      (set! i (+ i 1))
    })))

;; 火球生成 / 推进
(define spawn_projectile
  (lambda (x y dx dy)
    (define i 0) (define ok -1)
    (set! i 0)
    (while (< i pr_n) {
      (if (== (get_item pr_al i) 0) (if (< ok 0) (set! ok i)))
      (set! i (+ i 1))
    })
    (if (< ok 0) (if (< pr_n PMAX) { (set! ok pr_n) (set! pr_n (+ pr_n 1)) }))
    (if (>= ok 0) {
      (set_item! pr_x ok x)
      (set_item! pr_y ok y)
      (set_item! pr_dx ok dx)
      (set_item! pr_dy ok dy)
      (set_item! pr_al ok 1)
    })))

(define update_projectiles
  (lambda ()
    (define i 0) (define x 0.0) (define y 0.0) (define d 0.0)
    (set! i 0)
    (while (< i pr_n) {
      (if (== (get_item pr_al i) 1) {
        (set! x (+ (get_item pr_x i) (* (get_item pr_dx i) (* 4.2 dt))))
        (set! y (+ (get_item pr_y i) (* (get_item pr_dy i) (* 4.2 dt))))
        (set_item! pr_x i x)
        (set_item! pr_y i y)
        (set! d (dist2d x y px py))
        (if (< d 0.42) {
          (hurt_player 10)
          (set_item! pr_al i 0)
        }
        (if (== (solid_xy x y) 1) (set_item! pr_al i 0)))
      })
      (set! i (+ i 1))
    })))

;; 拾取
(define update_items
  (lambda ()
    (define i 0) (define d 0.0) (define k 0)
    (set! i 0)
    (while (< i it_n) {
      (if (== (get_item it_al i) 1) {
        (set! d (dist2d (get_item it_x i) (get_item it_y i) px py))
        (if (< d 0.62) {
          (set! k (get_item it_k i))
          (if (== k 10) { (set! hp (minf 100 (+ hp 25))) (msg_show "拾取医疗包 +25" 900) })
          (if (== k 11) { (set! bullets (+ bullets 20)) (set! shells (+ shells 4)) (msg_show "拾取弹药 +20/+4" 900) })
          (if (== k 12) { (set! armor (minf 100 (+ armor 25))) (msg_show "拾取护甲 +25" 900) })
          (set_item! it_al i 0)
        })
      })
      (set! i (+ i 1))
    })))

;; 门动画 + 出口判定
(define update_world_extras
  (lambda ()
    (define i 0) (define idx 0) (define f 0.0)
    (set! i 0)
    (while (< i (length doors)) {
      (set! idx (get_item doors i))
      (set! f (get_item doorf idx))
      (if (and (> f 0.0) (< f 1.0)) {
        (set! f (+ f (* dt 1.7)))
        (if (> f 1.0) (set! f 1.0))
        (set_item! doorf idx f)
      })
      (set! i (+ i 1))
    })
    ;; 出口
    (if (< (dist2d px py exit_x exit_y) 0.55) {
      (set! st_state 3)
      (set! st_timer 0.0)
      (if (>= lv 2) (set! st_state 4))
    })))

;; 自动开火 + 瞄准辅助（单指可玩的关键）：
;;   打开「自动开火」后，若 38° 锥内存在看得见的敌人，视角会被**限速**拉向它
;;   （130°/s，不会突然甩头），角度小于 6° 且冷却就绪时自动扣扳机。
;;   这是移动端 FPS 的常规「辅助瞄准」做法：玩家专心走位，开火交给准星。
(define combat_assist
  (lambda ()
    (define i 0) (define best -1) (define bd 999.0)
    (define d 0.0) (define a 0.0) (define ad 0.0) (define sp 0.0)
    (if (and (== auto_fire 1) (== turn_in 0))
        (if (== st_state 1) {
          (set! i 0)
          (while (< i en_n) {
            (if (> (get_item en_hp i) 0) {
              (set! d (dist2d (get_item en_x i) (get_item en_y i) px py))
              (if (< d 5.5) {
                (set! a (Math.atan2 (- (get_item en_y i) py) (- (get_item en_x i) px)))
                (set! a (/ (* a 180.0) (Math.PI)))
                (set! ad (ang_diff a pang))
                (if (and (< (Math.abs ad) 30.0) (< d bd))
                    (if (== (los_clear px py (get_item en_x i) (get_item en_y i)) 1)
                        { (set! bd d) (set! best i) }))
              })
            })
            (set! i (+ i 1))
          })
          (if (>= best 0) {
            (set! a (Math.atan2 (- (get_item en_y best) py) (- (get_item en_x best) px)))
            (set! a (/ (* a 180.0) (Math.PI)))
            (set! ad (ang_diff a pang))
            ;; 限速转向
            (set! sp (* 95.0 dt))
            (set! ad (clampf ad (- 0.0 sp) sp))
            (if (> (Math.abs ad) 0.05) {
              (set! pang (wrap360 (+ pang ad)))
              (update_cam)
            })
            (set! a (Math.atan2 (- (get_item en_y best) py) (- (get_item en_x best) px)))
            (set! a (/ (* a 180.0) (Math.PI)))
            (if (<= fire_cd 0.0)
                (if (< (Math.abs (ang_diff a pang)) 6.5) (set! want_fire 1)))
          })
        }))))

(define dbg_print
  (lambda ()
        (if (>= dbg_ms 500) {
        (set! dbg_ms 0)
        (display "[doom] p=") (display (Math.to_fixed px 1)) (display ",") (display (Math.to_fixed py 1))
        (display " a=") (display (Math.floor pang))
        (display " hp=") (display hp)
        (display " ar=") (display armor)
        (display " k=") (display kills) (display "/") (display level_enemies)
        (display " w=") (display wsel) (display " b=") (display bullets) (display " s=") (display shells)
        (display " sh=") (display shots_n)
        (display " st=") (display st_state)
        (display " nd=") (display (Math.to_fixed dbg_nd 1))
        (display " na=") (display (Math.floor dbg_na))
        (display " ns=") (display dbg_ns)
        (display " ai=") (display ai_on)
        (display " fps=") (display fps_v)
        (newline)
      })))

;; 诊断：最近的活敌距离 / 方位角差（供 e2e 取景与断言）
(define dbg_nearest
  (lambda ()
    (define i 0) (define ed 0.0) (define ea 0.0)
    (set! dbg_nd 999.0)
    (set! dbg_na 0.0)
    (set! i 0)
    (while (< i en_n) {
      (if (> (get_item en_hp i) 0) {
        (set! ed (dist2d (get_item en_x i) (get_item en_y i) px py))
        (if (< ed dbg_nd) {
          (set! dbg_nd ed)
          (set! ea (Math.atan2 (- (get_item en_y i) py) (- (get_item en_x i) px)))
          (set! dbg_na (ang_diff (* (/ ea (Math.PI)) 180.0) pang))
        })
      })
      (set! i (+ i 1))
    })))

;; ============================================================================
;; AI 自动游玩（自动寻路 + 交战 + 开门 + 拾取）
;;
;; 目标：一个「会自己把一关打完」的演示模式（对齐 app/tetris.scm、app/flappy.scm 里
;; 已有的「AI 手动 / AI 自动」开关惯例）。每拍做一次决策，输出与人类输入**同一条通路**
;; （ai_aim 直接加到朝向上、ai_fwd/ai_strafe 走 move_player、ai_fire 走 do_fire），
;; 因此 AI 与玩家的物理/碰撞/命中判定完全一致，没有"作弊穿墙"的旁路。
;;
;; 三层决策（每拍）：
;;   ① **交战**：9 格内有视线（los_clear）的最近敌人 → 限速转向对准它，对准即开火
;;      （目标每 0.25 秒重取一次，且只对由近到远前 3 个候选做视线检测 —— 每拍全量检测
;;      在 L2 大厅会同时命中七八只怪，帧率掉到 14fps）；
;;      按距离换枪（<3 格霰弹枪、<7 格链枪、否则手枪，弹药不足自动退回手枪）；
;;      远程怪（火球兵）在 2.5 格外时左右横移躲火球。
;;   ② **寻路**：没有可见敌人时，沿「目标距离场」的梯度走进下一格。
;;      距离场只有一张 MSIZE 长的表 ai_dist：以**候选目标格**为源做 BFS，既用于梯度下降
;;      （每拍取四邻里距离最小且小于自身的那一格，朝它的格心走），也用于可达性判定
;;      （玩家格在该场里的距离 ≥ 0 才算可达；否则换下一个候选）。
;;      目标优先级：最近的活敌 → 血量低于 45 时的医疗包 → 出口。
;;      BFS 只在「重规划时刻（1.5 秒）/目标失效/卡住」时跑，一次约 700 格 × 4 邻；
;;      寻路期间实测仍是 28~30fps（见 regress_doom.js 的帧率判据）。
;;   ③ **脱困**：连续 0.4 秒想走却走不动 → 强制重规划 + 侧移 0.35 秒（左右交替），
;;      避免卡在门框/墙角上原地磨。
;;
;; 门：距离场把门当成可通行格（否则一整片房间会被判成不可达）；真正走到门前时
;;     AI 面向门并调 (try_use) 开门，门没开满就不往前顶。
;;
;; 人机切换：左上「AI 手动 / AI 自动」按钮、键盘 G；AI 运行中在屏幕别处按下手指
;; 立刻交还手动（与俄罗斯方块/弹幕陨石的约定一致，按下的那一下照常生效）。
;; ============================================================================

(define ai_on 0)                  ; 0 手动 1 AI 自动
(define ai_walk (quote ()))       ; 可通行查表（0/1，随关卡预计算；门与出口算可通行）
(define ai_skip (quote ()))       ; 候选跳过表（选目标时避免重复试同一只怪）
(define ai_dist (quote ()))       ; BFS：以目标格为源的距离场（梯度用）
(define ai_gx 0) (define ai_gy 0) ; 当前目标格（只用于调试观察；寻路走 ai_dist）
(define ai_gk 0)                  ; 目标类型：1 敌人 2 医疗包 3 出口
(define ai_replan 0.0)            ; 距下次重规划的秒数
(define ai_stuck 0.0)             ; 「想走走不动」累计秒数
(define ai_px_prev 0.0)           ; 上次记录的位置（判进度）
(define ai_py_prev 0.0)
(define ai_detour 0.0)            ; 侧移脱困剩余秒数
(define ai_detour_dir 1)          ; 侧移方向 ±1
(define ai_aim 0.0)               ; 本拍 AI 转向量（度，已含限速）
(define ai_fwd 0.0)               ; 本拍 AI 前进量（-1..1）
(define ai_strafe 0.0)            ; 本拍 AI 平移量（-1..1）
(define ai_fire 0)                ; 本拍 AI 是否要开火
(define ai_ti -1)                 ; 当前交战目标下标（-1 无）
(define ai_td 0.0)                ; 交战目标距离
(define ai_acq 0.0)               ; 交战目标重取倒计时（秒）
(define ai_ndoor 0)               ; 本拍是否要开门

(define init_ai
  (lambda ()
    (set! ai_walk (mklist MSIZE))
    (set! ai_skip (mklist EMAX))
    (set! ai_dist (mklist MSIZE))))

;; 重算可通行表：空地/门/出口算可通行（门要走到跟前才开，寻路阶段先当路）
(define ai_build_walk
  (lambda ()
    (define i 0) (define t 0)
    (set! i 0)
    (while (< i MSIZE) {
      (set! t (get_item map i))
      (set_item! ai_walk i (if (== t 0) 1 (if (== t 6) 1 (if (== t 9) 1 0))))
      (set! i (+ i 1))
    })))

;; 以 (tx,ty) 为源做 BFS，把距离写进 field（-1 = 不可达）。
;; 队列是「push 到尾部 + 头指针前移」的 FIFO（本解释器的 push 是尾插、pop 是尾取，
;; 直接用 pop 会变成 DFS，得不到最短路）。
(define ai_bfs
  (lambda (field tx ty)
    (define i 0) (define head 0) (define cur 0) (define cx 0) (define cy 0)
    (define d0 0) (define q (cons 0 (quote ())))
    (define nxt 0)
    (pop q)
    (set! i 0)
    (while (< i MSIZE) {
      (set_item! field i -1)
      (set! i (+ i 1))
    })
    (if (== (get_item ai_walk (+ (* ty MW) tx)) 1) {
      (set_item! field (+ (* ty MW) tx) 0)
      (push q (+ (* ty MW) tx))
    })
    (set! head 0)
    (while (< head (length q)) {
      (set! cur (get_item q head))
      (set! d0 (+ (get_item field cur) 1))
      (set! cx (Math.floor (mod cur MW)))
      (set! cy (Math.floor (/ cur MW)))
      (if (> cx 0) {
        (set! nxt (- cur 1))
        (if (and (== (get_item field nxt) -1) (== (get_item ai_walk nxt) 1))
            { (set_item! field nxt d0) (push q nxt) })
      })
      (if (< cx (- MW 1)) {
        (set! nxt (+ cur 1))
        (if (and (== (get_item field nxt) -1) (== (get_item ai_walk nxt) 1))
            { (set_item! field nxt d0) (push q nxt) })
      })
      (if (> cy 0) {
        (set! nxt (- cur MW))
        (if (and (== (get_item field nxt) -1) (== (get_item ai_walk nxt) 1))
            { (set_item! field nxt d0) (push q nxt) })
      })
      (if (< cy (- MH 1)) {
        (set! nxt (+ cur MW))
        (if (and (== (get_item field nxt) -1) (== (get_item ai_walk nxt) 1))
            { (set_item! field nxt d0) (push q nxt) })
      })
      (set! head (+ head 1))
    })))

;; 选目标：最近的「可达」活敌 → （血量 < 45 时）最近的医疗包 → 出口。
;;
;; 只跑**一次** BFS：以候选目标格为源算距离场，顺带用「玩家格在该场里的距离是否 ≥ 0」
;; 回答可达性；不可达就换下一个候选（最多各试 4 / 3 个）。因为门在可通行表里算路，
;; 全图基本连通，所以绝大多数重规划只跑一次 BFS（这是 AI 模式不拖慢节拍的关键：
;; 早先版本每次重规划跑两次 BFS，浏览器里帧率一度掉到 ~14fps）。
(define ai_choose_goal
  (lambda ()
    (define i 0) (define pass 0) (define bi -1) (define bd 999.0) (define d 0.0)
    (define gk 0) (define gx 0) (define gy 0) (define pc 0)
    (set! pc (+ (* (Math.floor py) MW) (Math.floor px)))
    (set! gk 0)
    (set! i 0)
    (while (< i EMAX) {
      (set_item! ai_skip i 0)
      (set! i (+ i 1))
    })
    ;; ① 敌人：按欧氏距离由近到远试，最多 4 个
    (set! pass 0)
    (while (and (< pass 4) (== gk 0)) {
      (set! bi -1)
      (set! bd 999.0)
      (set! i 0)
      (while (< i en_n) {
        (if (and (> (get_item en_hp i) 0) (== (get_item ai_skip i) 0)) {
          (set! d (dist2d (get_item en_x i) (get_item en_y i) px py))
          (if (< d bd) {
            (set! bd d)
            (set! bi i)
          })
        })
        (set! i (+ i 1))
      })
      (if (>= bi 0) {
        (set_item! ai_skip bi 1)
        (set! gx (Math.floor (get_item en_x bi)))
        (set! gy (Math.floor (get_item en_y bi)))
        (ai_bfs ai_dist gx gy)
        (if (>= (get_item ai_dist pc) 0) (set! gk 1))
      })
      (set! pass (+ pass 1))
    })
    ;; ② 血量偏低时找医疗包（同样按距离由近到远试，最多 3 个）
    (set! pass 0)
    (while (and (== gk 0) (< hp 45) (< pass 3)) {
      (set! bi -1)
      (set! bd 999.0)
      (set! i 0)
      (while (< i it_n) {
        (if (and (== (get_item it_al i) 1) (== (get_item it_k i) 10)
                 (== (get_item ai_skip i) 0)) {
          (set! d (dist2d (get_item it_x i) (get_item it_y i) px py))
          (if (< d bd) {
            (set! bd d)
            (set! bi i)
          })
        })
        (set! i (+ i 1))
      })
      (if (>= bi 0) {
        (set_item! ai_skip bi 1)
        (set! gx (Math.floor (get_item it_x bi)))
        (set! gy (Math.floor (get_item it_y bi)))
        (ai_bfs ai_dist gx gy)
        (if (>= (get_item ai_dist pc) 0) (set! gk 2))
      })
      (set! pass (+ pass 1))
    })
    ;; ③ 出口
    (if (== gk 0) {
      (set! gx (Math.floor exit_x))
      (set! gy (Math.floor exit_y))
      (ai_bfs ai_dist gx gy)
      (if (>= (get_item ai_dist pc) 0) (set! gk 3))
    })
    (set! ai_gk gk)
    (set! ai_gx gx)
    (set! ai_gy gy)))

;; 限速转向：把朝向朝 target 角转最多 maxdeg 度，写进 ai_aim
(define ai_face
  (lambda (target maxdeg)
    (define ad 0.0)
    (set! ad (ang_diff target pang))
    (set! ad (clampf ad (- 0.0 maxdeg) maxdeg))
    (set! ai_aim ad)))

;; 交战目标获取（每 0.25 秒一次，或目标死亡/跑远时立刻重取）。
;; 只对「由近到远的前 3 个候选」做视线检测：L2 的大厅里同时可见的怪可能有七八只，
;; 每拍全量 los_clear 会把帧率拖到 14fps（实测），而目标并不会每帧都变。
(define ai_acquire
  (lambda ()
    (define i 0) (define pass 0) (define bi -1) (define bd 999.0) (define d 0.0)
    (set! ai_ti -1)
    (set! ai_td 999.0)
    (set! i 0)
    (while (< i en_n) {
      (set_item! ai_skip i 0)
      (set! i (+ i 1))
    })
    (set! pass 0)
    (while (and (< pass 3) (< ai_ti 0)) {
      (set! bi -1)
      (set! bd 999.0)
      (set! i 0)
      (while (< i en_n) {
        (if (and (> (get_item en_hp i) 0) (== (get_item ai_skip i) 0)) {
          (set! d (dist2d (get_item en_x i) (get_item en_y i) px py))
          (if (< d bd) {
            (set! bd d)
            (set! bi i)
          })
        })
        (set! i (+ i 1))
      })
      (if (>= bi 0) {
        (set_item! ai_skip bi 1)
        (if (< bd 9.0)
            (if (== (los_clear px py (get_item en_x bi) (get_item en_y bi)) 1) {
              (set! ai_ti bi)
              (set! ai_td bd)
            }))
      })
      (set! pass (+ pass 1))
    })))

;; 交战：返回 1 表示本拍进入交战（已写好 ai_aim / ai_fwd / ai_fire）
(define ai_combat
  (lambda ()
    (define ex 0.0) (define ey 0.0) (define a 0.0) (define ad 0.0) (define k 0)
    ;; 目标维护：没有/已死/跑远/定时到期 → 重新获取
    (if (< ai_ti 0) (set! ai_acq 0.0))
    (if (>= ai_ti 0)
        (if (<= (get_item en_hp ai_ti) 0) (set! ai_acq 0.0)))
    (if (>= ai_ti 0)
        (if (> (dist2d (get_item en_x ai_ti) (get_item en_y ai_ti) px py) 10.0)
            (set! ai_acq 0.0)))
    (set! ai_acq (- ai_acq dt))
    (if (<= ai_acq 0.0) {
      (set! ai_acq 0.25)
      (ai_acquire)
    })
    (if (>= ai_ti 0) {
      (set! ex (get_item en_x ai_ti))
      (set! ey (get_item en_y ai_ti))
      (set! k (get_item en_k ai_ti))
      (set! ai_td (dist2d ex ey px py))
      (set! a (/ (* (Math.atan2 (- ey py) (- ex px)) 180.0) (Math.PI)))
      ;; 转向（300°/s 上限）
      (ai_face a (* 300.0 dt))
      (set! ad (Math.abs (ang_diff a pang)))
      ;; 换枪：贴身霰弹、中距离链枪、其余手枪；对应弹药为 0 就退回手枪
      (set! wsel (if (and (< ai_td 3.0) (> shells 0)) 1
                     (if (and (< ai_td 7.0) (> bullets 0)) 2 0)))
      ;; 开火：对准（±7°）且冷却就绪
      (if (and (< ad 7.0) (<= fire_cd 0.0)) (set! ai_fire 1))
      ;; 走位：>2.6 格就往前压；对火球兵再加左右横移躲弹
      (set! ai_fwd (if (> ai_td 2.6) 0.85 0.0))
      (if (and (== k 1) (> ai_td 2.5))
          (set! ai_strafe (* 0.7 (Math.sin (* 0.006 now_ms)))))
      (set! ai_ndoor 0)
      1
    }
    (if (< ai_ti 0) 0))))

;; 寻路：沿 ai_dist 梯度走（返回 1 表示已在走）
(define ai_nav
  (lambda ()
    (define cx 0) (define cy 0) (define cur 0) (define d 0) (define bd 9999)
    (define ni -1) (define t 0) (define nx 0.0) (define ny 0.0) (define a 0.0)
    (define ad 0.0)
    (set! cx (Math.floor px))
    (set! cy (Math.floor py))
    (set! cur (+ (* cy MW) cx))
    (set! d (get_item ai_dist cur))
    (set! ni -1)
    (set! bd 9999)
    (if (> cx 0) {
      (set! t (get_item ai_dist (- cur 1)))
      (if (and (>= t 0) (< t d) (< t bd)) { (set! bd t) (set! ni (- cur 1)) })
    })
    (if (< cx (- MW 1)) {
      (set! t (get_item ai_dist (+ cur 1)))
      (if (and (>= t 0) (< t d) (< t bd)) { (set! bd t) (set! ni (+ cur 1)) })
    })
    (if (> cy 0) {
      (set! t (get_item ai_dist (- cur MW)))
      (if (and (>= t 0) (< t d) (< t bd)) { (set! bd t) (set! ni (- cur MW)) })
    })
    (if (< cy (- MH 1)) {
      (set! t (get_item ai_dist (+ cur MW)))
      (if (and (>= t 0) (< t d) (< t bd)) { (set! bd t) (set! ni (+ cur MW)) })
    })
    (if (>= ni 0) {
      ;; 下一格若是没开满的门 → 面向它、开门，不往前顶
      (set! t (get_item map ni))
      (if (and (== t 6) (< (get_item doorf ni) 0.9)) {
        (set! nx (+ (Math.floor (mod ni MW)) 0.5))
        (set! ny (+ (Math.floor (/ ni MW)) 0.5))
        (set! a (/ (* (Math.atan2 (- ny py) (- nx px)) 180.0) (Math.PI)))
        (ai_face a (* 260.0 dt))
        (set! ad (Math.abs (ang_diff a pang)))
        (if (< ad 22.0) (set! ai_ndoor 1))
        (set! ai_fwd 0.0)
      }
      {
        (set! nx (+ (Math.floor (mod ni MW)) 0.5))
        (set! ny (+ (Math.floor (/ ni MW)) 0.5))
        (set! a (/ (* (Math.atan2 (- ny py) (- nx px)) 180.0) (Math.PI)))
        (ai_face a (* 260.0 dt))
        (set! ad (Math.abs (ang_diff a pang)))
        ;; 大致对准才前进（否则原地转），避免贴墙磨
        (set! ai_fwd (if (< ad 45.0) 1.0 0.0))
      })
    })
    (if (>= ni 0) 1 0)))

;; 每拍 AI 决策入口（在 update_world 里、应用移动之前调用）
(define ai_tick
  (lambda ()
    (define moving 0) (define prog 0.0)
    (set! ai_aim 0.0)
    (set! ai_fwd 0.0)
    (set! ai_strafe 0.0)
    (set! ai_fire 0)
    (set! ai_ndoor 0)
    ;; 重规划倒计时 / 目标失效时重规划
    (set! ai_replan (- ai_replan dt))
    (if (<= ai_replan 0.0) {
      (set! ai_replan 1.5)
      (ai_choose_goal)
    })
    ;; 目标敌人已死 → 立刻重规划（不必等 1.2 秒）
    (if (and (== ai_gk 1) (>= ai_ti 0) (<= (get_item en_hp ai_ti) 0)) (set! ai_replan 0.0))
    ;; ① 交战优先
    (set! moving (ai_combat))
    ;; ② 无可见敌人 → 寻路
    (if (== moving 0) (ai_nav))
    ;; ③ 脱困：想走却几乎没动 → 侧移 + 重规划（显式嵌套 if，避免把 else 写成
    ;;    "无条件执行的额外语句"）
    (if (> ai_fwd 0.5)
        (if (< ai_detour 0.0)
            (if (< (dist2d px py ai_px_prev ai_py_prev) 0.02)
                {
                  (set! prog 0.0)
                  (set! ai_stuck (+ ai_stuck dt))
                  (if (> ai_stuck 0.4) {
                    (set! ai_stuck 0.0)
                    (set! ai_detour 0.35)
                    (set! ai_detour_dir (- 0.0 ai_detour_dir))
                    (set! ai_replan 0.0)
                  })
                }
                {
                  (set! ai_stuck 0.0)
                  (set! ai_px_prev px)
                  (set! ai_py_prev py)
                })
            #f)
        #f)
    (if (> ai_detour 0.0) {
      (set! ai_detour (- ai_detour dt))
      (set! ai_strafe ai_detour_dir)
      (set! ai_fwd 0.55)
    })
    ;; 开门：寻路请求或贴身时兜底
    (if (== ai_ndoor 1) (set! want_use 1))
    (if (== ai_fire 1) (set! want_fire 1))))

;; 开关 AI（按钮 / 键盘 G 共用）；切换时重置内部状态
(define ai_set
  (lambda (v)
    (set! ai_on v)
    (set! ai_replan 0.0)
    (set! ai_stuck 0.0)
    (set! ai_detour 0.0)
    (set! ai_px_prev px)
    (set! ai_py_prev py)
    (if (== v 1) (msg_show "AI 自动游玩：寻路 · 交战 · 开门 · 拾取" 2200)
                 (msg_show "已交还手动控制" 1400))
    (display "[doom] ai=") (display ai_on) (newline)))

;; 一拍的世界推进
(define update_world
  (lambda ()
    (define fwd 0.0) (define str 0.0) (define trn 0.0)
    (define vx 0.0) (define vy 0.0)
    (set! game_ms (+ game_ms (Math.floor (* dt 1000.0))))
    ;; 转向
    (set! turn_in 0)
    (if (> fire_cd 0.0) (set! fire_cd (- fire_cd dt)))
    (if (> fire_anim 0.0) (set! fire_anim (- fire_anim dt)))
    (if (> hurt_flash 0.0) (set! hurt_flash (- hurt_flash dt)))
    (if (> pain_cd 0.0) (set! pain_cd (- pain_cd dt)))
    (if (> msg_ms 0) (set! msg_ms (- msg_ms (Math.floor (* dt 1000.0)))))
    (if (or (== kb_turnl 1) (== kb_turnr 1)) (set! turn_in 1))
    (if (> (Math.abs turn_drag) 0.01) (set! turn_in 1))
    ;; AI 自动游玩：决策先跑，随后把它的输出并进与人类输入同一条通路
    (if (== ai_on 1) {
      (ai_tick)
      (set! turn_in 1)              ; 让辅助瞄准让路（AI 自己负责瞄准）
    })
    ;; 键盘转向是角速度（°/s），触屏拖动是"本拍的角度增量"（°，已按像素折算）——
    ;; 两者量纲不同，必须分别处理：早先把 turn_drag 也乘了 dt，拖动转向几乎不动。
    (set! trn 0.0)
    (if (== kb_turnl 1) (set! trn (- 0.0 150.0)))
    (if (== kb_turnr 1) (set! trn 150.0))
    (set! pang (wrap360 (+ pang (+ (* trn dt) (+ turn_drag ai_aim)))))
    (update_cam)
    ;; 前后/平移
    (set! fwd (- (+ kb_fwd 0.0) (+ kb_back 0.0)))
    (set! fwd (+ fwd (- 0.0 joy_dy)))
    (set! str (- (+ kb_right 0.0) (+ kb_left 0.0)))
    (set! str (+ str joy_dx))
    (if (== ai_on 1) {
      (set! fwd ai_fwd)
      (set! str ai_strafe)
    })
    (set! fwd (clampf fwd (- 0.0 1.0) 1.0))
    (set! str (clampf str (- 0.0 1.0) 1.0))
    (set! vx (+ (* fwd pdirx) (* str (- 0.0 pdiry))))
    (set! vy (+ (* fwd pdiry) (* str pdirx)))
    (move_player vx vy)
    ;; 动作
    (if (== want_use 1) (try_use))
    (if (== ai_on 0) (combat_assist))
    (if (== want_fire 1) (if (<= fire_cd 0.0) (do_fire)))
    ;; 世界
    (update_enemies)
    (update_projectiles)
    (update_items)
    (update_world_extras)
    (dbg_nearest)))

;; ============================================================================
;; 节拍
;; ============================================================================

(define TICK_MS 33)
(define _timer 0)
(define now_ms 0)
(define last_ms 0)
(define dt 0.0)

(define tick
  (lambda ()
    (define d 0)
    (EX.check)
    (set! tap_now 0)
    ;; 用宿主的 System.timestamp（毫秒，虚拟时钟在回归台里可加速）；
    ;; 32 位宿主的载荷是 27 位 → 差值仍正确，跨越回绕时按 0 处理（下面钳位）
    (set! now_ms (System.timestamp))
    (set! d (- now_ms last_ms))
    (set! last_ms now_ms)
    (if (< d 0) (set! d 0))
    (if (> d 60) (set! d 60))
    (set! dt (/ (* d 1.0) 1000.0))
    ;; FPS 统计
    (set! fps_acc (+ fps_acc dt))
    (set! fps_cnt (+ fps_cnt 1))
    (if (>= fps_acc 0.5) {
      (set! fps_v (Math.floor (/ fps_cnt fps_acc)))
      (set! fps_acc 0.0)
      (set! fps_cnt 0)
    })
    (read_input)
    ;; 状态机
    (if (== st_state 0) {
      (if (== tap_now 1) {
        (set! total_kills 0)
        (load_level 0)
      })
    })
    (if (== st_state 1) (update_world))
    (if (> st_state 1) {
      (set! st_timer (+ st_timer dt))
      (if (> st_timer 0.6)
          (if (== tap_now 1) (advance_state)))
      ;; AI 自动游玩时自行续关/重来（演示模式可以一直跑下去）
      (if (and (== ai_on 1) (> st_timer 1.4)) (advance_state))
    })
    (render)
    (set! dbg_ms (+ dbg_ms (Math.floor (* dt 1000.0))))
    (dbg_print)))

;; 状态推进（死亡重来 / 下一关 / 重开）
(define tap_now 0)        ; 本拍是否发生了「点按」（触屏按下或任何按键）
(define advance_state
  (lambda ()
    (if (== st_state 2) (load_level lv))
    (if (== st_state 3) {
      (if (< lv 2) (load_level (+ lv 1)) (load_level 0))
    })
    (if (== st_state 4) {
      (set! total_kills 0)
      (load_level 0)
    })))

;; ============================================================================
;; 启动
;; ============================================================================

;; ---- 启动：形状表 / 数组 / 第 1 关 → 标题页 ----
(init_shapes)
(init_sl)
(init_ai)
(set! last_ms (System.timestamp))
(set! now_ms last_ms)
(load_level 0)
(set! st_state 0)          ; 先停在标题页（load_level 会把它置成 1）
(set! total_kills 0)
(render)
;; 注意：这里**必须**用一层 lambda 包住 tick，不能直接 (System.set_interval TICK_MS tick)。
;; wasm32 上把「自由变量很多的模块级闭包」直接注册成定时器回调时，回调永远不会被调用
;; （桌面 64 位宿主正常；实测探针：直接注册 tick → 一次都不跑，包一层 lambda 立刻正常）。
;; tick 引用了近百个模块级变量、还引用了文本上位于其后的 advance_state，属退化闭包路径；
;; 包一层小闭包后走普通帧描述符路径。回归见 tools/e2e/browser/regress_doom.js。
(set! _timer (System.set_interval TICK_MS (lambda () (tick))))
(EX.install_with render)
