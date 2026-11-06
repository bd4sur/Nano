;; ============================================================================
;; 黄金矿工（Gold Miner）—— Nano「灵机应用程序」Scheme App（**单文件**）
;;
;; 参照实现：infer/ui_goldminer.c（C 版，主菜单「小游戏」里的黄金矿工）。
;; 本文件按 C 版的组织顺序排列两段：
;;   ① 规则与状态（对照 gm_generate_level 与 ui_goldminer_render_frame 的**逻辑段**：
;;      摆动 / 出钩 / 回收 / 抓取 / 结算 / 过关 / 关卡生成）；
;;   ② 布局、绘制、输入与主循环（对照渲染段与 ui_goldminer_event_handler）。
;;
;; 为什么是单文件：一度拆为 app/lib/goldminer_core.scm + app/goldminer.scm，用一处
;; 人为接缝（(set_screen w h)、(set_sprite_size id w h) 两个注入接口 + 物品槽位表示
;; 泄漏给绘制端）换取"无 UI 依赖的纯逻辑单测"。但黄金矿工的规则量（约 120 行）不足以
;; 摊平这处接缝，故合并；回归改为**经 HAL 驱动真实 App + 控制台诊断行**断言
;; （tools/e2e/native/test_goldminer.py），仍然覆盖关卡构成、落点约束、抓取分值、
;; 带物回收按重量减速、过关等要点。
;;
;; 规则（逐条同 C 版）：320x240，地表线 y=70，摆动支点 (160,66)，绳长下限 16px；
;; 摆动 hook_angle = 1.30*sin(phase)（速度 2.2 + 0.1*(level-1) rad/s）；出钩 300px/s，
;; 撞到边界（x<=3 / x>=W-3 / y>=H-4）或抓到物体即回收；回收 240px/s，抓到物体时除以
;; 重量；回到绳长下限处结算分值；命中判定 = 钩尖进入"以物体中心为圆心、直径 = 精灵
;; 宽高较小者"的内切圆；关卡构成 大/中金块各 1、小金块 3+min(level-1,3)、岩石
;; 3+level/2、钻石 level>=2、福袋 1、宝箱 level>=3；随机放置并拒绝重叠（外扩 8px
;; 椭圆判定）；收完该关全部物品即进入下一关；单拍步长钳 50ms（= C 版 GM_DT_MAX）。
;;
;; 贴图（**路径与 C 版完全一致**，C 版为 PLATFORM_ROOT_DIR "/icon/gm_*.png"；
;; Scheme 侧路径相对业务资源根，而资源根就是 PLATFORM_ROOT_DIR，指同一批文件）：
;;   icon/gm_miner.png  icon/gm_hook.png  icon/gm_gold_1..3.png
;;   icon/gm_rock.png   icon/gm_diamond.png  icon/gm_bonus_1..2.png
;; 它们都是**透明背景** PNG，因此必须用 alpha 混合的精灵接口绘制：
;;   (Gfx.sprite_info 路径)          → (宽 高) | #null（解码并常驻缓存，也用于预热）
;;   (Gfx.draw_sprite 路径 cx cy)    → 以 (cx,cy) 为中心按原始尺寸混合绘制
;;   (Gfx.draw_sprite_rotated 路径 px py 角度)
;;                                   → 以 (px,py)（贴图上缘中点）为枢轴旋转后绘制
;; 钩绳用 (Gfx.draw_line_aa ...) 画（抗锯齿，同 C 版 gfx_draw_line_anti_aliasing）。
;;   注：Markdown 阅读器用的 (Gfx.draw_image ...) 是"缩放到目标矩形 + RGB 直写"的
;;   插图通道（丢弃 alpha），画透明精灵会得到不透明方块，故本 App 不走那条路。
;;
;; 与 C 版的三处**平台差异**（源于"Scheme App 运行在宿主的应用列表里"）：
;;   1) 退出：C 版右上角自绘「返回」按钮回小游戏菜单；Scheme App 没有那个状态，
;;      退出即回应用列表。故统一走 app/lib/exit.scm —— 右上角「退出」按钮 + 硬
;;      A/ESC，都是"弹全局确认框 → 确认后销毁实例"（与俄罗斯方块/2048 同款）。
;;   2) 键盘：C 版 D(回车)/数字 2 发射；Scheme 侧经 (native Key) 的按下沿同样支持
;;      回车与 '2'，另有触屏（C 版也有）。
;;   3) 帧节奏：C 版每帧由宿主驱动；这里用 System.set_interval 按 ~30fps 节流，
;;      物理步长用 System.timestamp 的实测间隔并钳到 50ms（与 C 版 GM_DT_MAX 一致）。
;;
;; 诊断行（只在状态变化时打印，端到端测试据此断言）：
;;   GMSTAT  —— 启动：精灵数、本关物品数、类型构成、物品包围盒、分值、关卡
;;   GMFIRE  —— 发射（进入出钩）
;;   GMGRAB  —— 抓到物品（下标 + 类型）
;;   GMSHOT  —— 一次投掷的结算明细（类型/分值/重量/绳长/出钩与回收耗时）
;;   GMSCORE —— 分值变化（分值/剩余/关卡）
;;   GMLEVEL —— 过关（新关卡物品数 + 新的类型构成 + 新的包围盒）
;;
;; 解释器约束（manual.md）：算术恒二参数、无一元取负；无 list/let/map/filter；
;; cond 每子句只接受单个表达式；/ 与 mod 恒返回浮点（取整用 Math.floor）；
;; '() 是模块级静态槽（空表一律 (cdr '(0)) 现造）；while 体不是独立词法区域
;; → 所有局部量都用模块级 define + set!。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Touch)
(native Key)
(native App)

(import EX "app/lib/exit.scm")   ; 右上角「退出」按钮 + 硬 A/ESC → 全局退出确认

;; ============================================================================
;; 场景与物理常量（与 C 版一致）
;; ============================================================================

(define SW 320)             ;; 屏幕宽（boot 时按 Gfx.width 落实）
(define SH 240)             ;; 屏幕高
(define GROUND_Y 70)        ;; 地表线 y（以上为天空/矿区台面）
(define PIVOT_X 160)        ;; 钩绳摆动支点
(define PIVOT_Y 66)
(define ROPE_MIN 16)        ;; 绳长下限（收回完成位置）
(define SWING_MAX 1.30)     ;; 最大摆角（约 75°）
(define SWING_SPEED 2.2)    ;; 摆动角速度（rad/s，随关卡略增）
(define SWING_STEP 0.1)     ;; 每关增量
(define EXTEND_SPEED 300)   ;; 出钩速度（px/s）
(define RETRACT_SPEED 240)  ;; 空钩回收速度（px/s）；抓到物体后除以重量
(define DT_MAX 50)          ;; 单拍最大物理步长（ms），防卡顿跳变
(define MAX_ITEMS 24)

;; 阶段（与 C 版 GM_Phase 一一对应）
(define PHASE_SWING 0)      ;; 摆动待发射
(define PHASE_EXTEND 1)     ;; 出钩
(define PHASE_RETRACT 2)    ;; 回收

(define TICK_MS 33)         ;; 定时器周期（ms，约 30fps：出钩 300px/s → 每拍 10px）

;; ============================================================================
;; 精灵表（顺序与 C 版 GM_Sprite_Id 完全一致）
;; ============================================================================

(define SPR_MINER 0)
(define SPR_HOOK 1)
(define SPR_GOLD_1 2)       ;; 物品精灵的起点：物品类型 t 对应 SPR_GOLD_1 + t
(define SPR_NUM 9)

(define SPR_PATH '(
  "icon/gm_miner.png"
  "icon/gm_hook.png"
  "icon/gm_gold_1.png"
  "icon/gm_gold_2.png"
  "icon/gm_gold_3.png"
  "icon/gm_rock.png"
  "icon/gm_diamond.png"
  "icon/gm_bonus_1.png"
  "icon/gm_bonus_2.png"))

;; 精灵尺寸缓存（像素）：进入游戏时按真实 PNG 尺寸一次性填充（C 版 s_gm_sprite_w/h
;; 的同一用途）。初值是 C 版"无贴图时"的原型包围盒，贴图缺失时退化为同样的布局。
(define spr_w (cdr '(0)))
(define spr_h (cdr '(0)))

;; ============================================================================
;; 物品原型（7 种，顺序与 C 版 GM_Item_Type 一一对应）
;;   扁平表，索引 = type*6 + {0=value 1=weight 2=radius 3=r 4=g 5=b}
;;   颜色只在贴图缺失时用于画原型圆点（与 C 版一致的降级路径）
;; ============================================================================

(define TYPE_NUM 7)

(define PROTO '(
   500 2.5  13  255 200  30    ; 大金块
   300 1.8  10  255 208  45    ; 中金块
   200 1.2   8  255 215  60    ; 小金块
    80 3.5  12  150 150 160    ; 岩石
   600 0.8   6  120 220 255    ; 钻石
   350 0.6   7  120 230 120    ; 福袋
  1000 0.5   7  230  90 220))  ; 宝箱

;; ============================================================================
;; 游戏状态
;; ============================================================================

(define phase PHASE_SWING)
(define angle_phase 0)      ;; 摆动相位
(define hook_angle 0)       ;; 当前钩绳与竖直方向夹角（发射后固定）
(define rope_len ROPE_MIN)
(define hook_x PIVOT_X)
(define hook_y (+ PIVOT_Y ROPE_MIN))
(define grabbed -1)         ;; 抓到的物品索引（-1 无）
(define score 0)
(define level 1)
(define items_left 0)       ;; 本关剩余物品数
(define items '())          ;; MAX_ITEMS 个 (active type x y w h) 子表

;; ---- 模块级临时量（热路径局部量一律提到模块级：解释器对块内 define/嵌套区域的
;;      槽位存在布局敏感问题，见 doc/AGENTS.md）----------------------------

(define t_i 0) (define t_k 0) (define t_t 0) (define t_a 0)
(define t_w 0) (define t_h 0) (define t_x 0) (define t_y 0)
(define t_xr 0) (define t_yr 0) (define t_ok 0) (define t_ov 0)
(define t_dx 0) (define t_dy 0) (define t_nx 0) (define t_ny 0)
(define t_ox 0) (define t_oy 0) (define t_rr 0) (define t_done 0)
(define t_dt 0) (define t_speed 0) (define t_ms 0)

;; 绘制侧临时量
(define i 0)
(define k 0)
(define sp "")
(define it_active 0)
(define it_type 0)

;; ---- 表构建（模块加载时一次性 push；此后只用 set_item! 原地改写）----------

(define mk_item
  (lambda () (cons 0 (cons 0 (cons 0 (cons 0 (cons 0 (cons 0 (cdr '(0))))))))))

(define build_tables
  (lambda () {
    (set! items (cdr '(0)))
    (set! spr_w (cdr '(0)))
    (set! spr_h (cdr '(0)))
    (set! t_i 0)
    (while (< t_i MAX_ITEMS)
      { (push items (mk_item)) (set! t_i (+ t_i 1)) })
    ;; 精灵尺寸初值 = C 版原型包围盒：矿工 16x26 / 钩爪 12x8 / 物品 = 原型直径
    (set! t_i 0)
    (while (< t_i SPR_NUM)
      { (push spr_w 16) (push spr_h 26) (set! t_i (+ t_i 1)) })
    (set_item! spr_w SPR_MINER 16)  (set_item! spr_h SPR_MINER 26)
    (set_item! spr_w SPR_HOOK 12)   (set_item! spr_h SPR_HOOK 8)
    (set! t_i 0)
    (while (< t_i TYPE_NUM)
      { (set_item! spr_w (+ SPR_GOLD_1 t_i) (* 2 (get_item PROTO (+ (* t_i 6) 2))))
        (set_item! spr_h (+ SPR_GOLD_1 t_i) (* 2 (get_item PROTO (+ (* t_i 6) 2))))
        (set! t_i (+ t_i 1)) })
  }))

(build_tables)

;; ---- 表访问小工具 --------------------------------------------------------

(define item_ref (lambda (i k) (get_item (get_item items i) k)))
(define item_set (lambda (i k v) (set_item! (get_item items i) k v)))
(define proto_ref (lambda (ty k) (get_item PROTO (+ (* ty 6) k))))

;; ============================================================================
;; 关卡生成（对照 C 版 gm_generate_level）
;; ============================================================================

(define counts (cdr '(0)))

(define gen_counts
  (lambda () {
    (set! counts (cdr '(0)))
    (push counts 1)                                                  ; 大金块：每关 1
    (push counts 1)                                                  ; 中金块：每关 1
    (push counts (+ 3 (if (> (- level 1) 3) 3 (- level 1))))          ; 小金块
    (push counts (+ 3 (Math.floor (/ level 2))))                      ; 岩石
    (push counts (if (>= level 2) 1 0))                               ; 钻石
    (push counts 1)                                                   ; 福袋
    (push counts (if (>= level 3) 1 0))                               ; 宝箱
  }))

(define clear_items
  (lambda () {
    (set! t_i 0)
    (while (< t_i MAX_ITEMS)
      { (item_set t_i 0 0) (set! t_i (+ t_i 1)) })
  }))

(define generate_level
  (lambda () {
    (clear_items)
    (gen_counts)
    (set! items_left 0)
    (set! t_t 0)
    (while (< t_t TYPE_NUM) {
      ;; 该类型精灵的实际宽高（查尺寸缓存表）
      (set! t_w (get_item spr_w (+ SPR_GOLD_1 t_t)))
      (set! t_h (get_item spr_h (+ SPR_GOLD_1 t_t)))
      ;; 随机放置范围：保证整个精灵不出屏幕、整体位于地下
      (set! t_xr (Math.floor (- (- SW t_w) 12)))
      (if (< t_xr 1) (set! t_xr 1))
      (set! t_yr (Math.floor (- (- (- SH GROUND_Y) t_h) 24)))
      (if (< t_yr 1) (set! t_yr 1))
      (set! t_k 0)
      (while (and (< t_k (get_item counts t_t)) (< items_left MAX_ITEMS)) {
        (set! t_a 0)
        (set! t_ok 0)
        ;; 随机放置，拒绝与已放置物体重叠（最多尝试 50 次）
        (while (and (< t_a 50) (== t_ok 0)) {
          (set! t_x (+ (+ 6 (/ t_w 2)) (Math.floor (* (Math.random) t_xr))))
          (set! t_y (+ (+ (+ GROUND_Y 12) (/ t_h 2)) (Math.floor (* (Math.random) t_yr))))
          (set! t_ov 0)
          (set! t_i 0)
          (while (< t_i items_left) {
            (if (== (item_ref t_i 0) 1) {
              ;; 按双方精灵实际宽高做椭圆重叠判定（外扩 8px 间距）
              (set! t_ox (+ (/ (+ (item_ref t_i 4) t_w) 2) 8))
              (set! t_oy (+ (/ (+ (item_ref t_i 5) t_h) 2) 8))
              (set! t_nx (/ (- (item_ref t_i 2) t_x) t_ox))
              (set! t_ny (/ (- (item_ref t_i 3) t_y) t_oy))
              (if (< (+ (* t_nx t_nx) (* t_ny t_ny)) 1) (set! t_ov 1))
            })
            (set! t_i (+ t_i 1))
          })
          (if (== t_ov 0) {
            (item_set items_left 0 1)
            (item_set items_left 1 t_t)
            (item_set items_left 2 t_x)
            (item_set items_left 3 t_y)
            (item_set items_left 4 t_w)
            (item_set items_left 5 t_h)
            (set! items_left (+ items_left 1))
            (set! t_ok 1)
          })
          (set! t_a (+ t_a 1))
        })
        (set! t_k (+ t_k 1))
      })
      (set! t_t (+ t_t 1))
    })
  }))

;; ---- 诊断用的统计（关卡构成 / 物品包围盒）-------------------------------

(define stats_txt "")
(define bx0 0) (define by0 0) (define bx1 0) (define by1 0)

;; 按类型统计仍存活的物品数（**独占**临时量 c_i/c_k：本函数会被 build_stats_txt 的
;; 循环体调用，若与循环计数器共用槽位会把外层循环的计数冲掉）
(define c_i 0)
(define c_k 0)
(define count_type
  (lambda (want) {
    (set! c_i 0)
    (set! c_k 0)
    (while (< c_i MAX_ITEMS) {
      (if (== (item_ref c_i 0) 1)
          (if (== (item_ref c_i 1) want) (set! c_k (+ c_k 1)) #f)
          #f)
      (set! c_i (+ c_i 1))
    })
    c_k
  }))

;; 七类物品数量的 "a/b/c/d/e/f/g" 串（供端到端断言关卡构成）
(define build_stats_txt
  (lambda () {
    (set! stats_txt "")
    (set! t_i 0)
    (while (< t_i TYPE_NUM) {
      (if (> t_i 0) (set! stats_txt (String.concat stats_txt "/")) #f)
      (set! stats_txt (String.concat stats_txt (String.atom_to_string (count_type t_i))))
      (set! t_i (+ t_i 1))
    })
  }))

;; 全部存活物品的像素包围盒（整数），写入 bx0/by0/bx1/by1
(define calc_bounds
  (lambda () {
    (set! bx0 0) (set! by0 0) (set! bx1 0) (set! by1 0)
    (set! t_i 0)
    (set! t_k 0)
    (while (< t_i MAX_ITEMS) {
      (if (== (item_ref t_i 0) 1) {
        (set! t_x (Math.floor (- (item_ref t_i 2) (/ (item_ref t_i 4) 2))))
        (set! t_y (Math.floor (- (item_ref t_i 3) (/ (item_ref t_i 5) 2))))
        (if (== t_k 0)
            { (set! bx0 t_x) (set! by0 t_y)
              (set! bx1 (Math.floor (+ (item_ref t_i 2) (/ (item_ref t_i 4) 2))))
              (set! by1 (Math.floor (+ (item_ref t_i 3) (/ (item_ref t_i 5) 2)))) }
            { (if (< t_x bx0) (set! bx0 t_x) #f)
              (if (< t_y by0) (set! by0 t_y) #f)
              (if (> (+ (item_ref t_i 2) (/ (item_ref t_i 4) 2)) bx1)
                  (set! bx1 (Math.floor (+ (item_ref t_i 2) (/ (item_ref t_i 4) 2)))) #f)
              (if (> (+ (item_ref t_i 3) (/ (item_ref t_i 5) 2)) by1)
                  (set! by1 (Math.floor (+ (item_ref t_i 3) (/ (item_ref t_i 5) 2)))) #f) })
        (set! t_k (+ t_k 1))
      })
      (set! t_i (+ t_i 1))
    })
  }))

;; 打印一行关卡摘要（boot 用 GMSTAT、过关用 GMLEVEL）
(define print_level_line
  (lambda (tag) {
    (build_stats_txt)
    (calc_bounds)
    (display tag)
    (display " items=") (display items_left)
    (display " counts=") (display stats_txt)
    (display " rect=") (display bx0) (display ",") (display by0)
    (display ",") (display bx1) (display ",") (display by1)
    (display " score=") (display score)
    (display " level=") (display level)
    (newline)
  }))

;; ============================================================================
;; 投掷动作
;; ============================================================================

;; 一次投掷的计时（出钩/回收耗时，供 GMSHOT 诊断行）
(define fire_ms 0)
(define grab_ms 0)
(define grab_rope 0)

;; 发射（仅摆动状态有效；落点/命中在 step 里判定）
(define fire
  (lambda ()
    (if (== phase PHASE_SWING)
        { (set! phase PHASE_EXTEND)
          (set! fire_ms (System.timestamp)) }
        #f)
    #t))

;; 重置一局（App 启动/重开时调用）
(define reset
  (lambda () {
    (set! phase PHASE_SWING)
    (set! angle_phase 0)
    (set! hook_angle 0)
    (set! rope_len ROPE_MIN)
    (set! hook_x PIVOT_X)
    (set! hook_y (+ PIVOT_Y ROPE_MIN))
    (set! grabbed -1)
    (set! score 0)
    (set! level 1)
    (generate_level)
  }))

;; 出钩状态的抓取判定（钩尖进入精灵中心内切圆即命中）
(define try_grab
  (lambda () {
    (set! t_i 0)
    (set! t_done 0)
    (while (and (< t_i MAX_ITEMS) (== t_done 0)) {
      (if (== (item_ref t_i 0) 1) {
        (set! t_dx (- (item_ref t_i 2) hook_x))
        (set! t_dy (- (item_ref t_i 3) hook_y))
        (set! t_w (item_ref t_i 4))
        (set! t_h (item_ref t_i 5))
        (set! t_rr (/ (if (< t_w t_h) t_w t_h) 2))
        (if (<= (+ (* t_dx t_dx) (* t_dy t_dy)) (* t_rr t_rr)) {
          (set! grabbed t_i)
          (set! phase PHASE_RETRACT)
          (set! grab_rope rope_len)
          (set! grab_ms (System.timestamp))
          (set! t_done 1)
        })
      })
      (set! t_i (+ t_i 1))
    })
  }))

;; 收回完成：结算抓到的物品，清空则进入下一关
(define collect_grabbed
  (lambda () {
    (set! t_k (item_ref grabbed 1))                    ;; 类型（供 GMSHOT）
    (set! score (+ score (proto_ref t_k 0)))
    (item_set grabbed 0 0)
    (set! grabbed -1)
    (set! items_left (- items_left 1))
    (display "GMSHOT type=") (display t_k)
    (display " value=") (display (proto_ref t_k 0))
    (display " weight=") (display (proto_ref t_k 1))
    (display " rope=") (display (Math.to_fixed grab_rope 1))
    (display " extend=") (display (- grab_ms fire_ms))
    (display " retract=") (display (- (System.timestamp) grab_ms))
    (newline)
    (if (<= items_left 0) {
      (set! level (+ level 1))
      (generate_level)
      (print_level_line "GMLEVEL")
    })
  }))

;; 一拍逻辑推进（dt_ms 为实测毫秒间隔，内部钳到 [0, DT_MAX]，同 C 版 GM_DT_MAX）
(define step
  (lambda (dt_ms) {
    (set! t_ms dt_ms)
    (if (< t_ms 0) (set! t_ms 0))
    (if (> t_ms DT_MAX) (set! t_ms DT_MAX))
    (set! t_dt (/ t_ms 1000))

    (if (== phase PHASE_SWING) {
      (set! angle_phase (+ angle_phase (* (+ SWING_SPEED (* SWING_STEP (- level 1))) t_dt)))
      (set! hook_angle (* SWING_MAX (Math.sin angle_phase)))
      (set! rope_len ROPE_MIN)
    })
    (if (== phase PHASE_EXTEND)
        (set! rope_len (+ rope_len (* EXTEND_SPEED t_dt))))
    (if (== phase PHASE_RETRACT) {
      (set! t_speed RETRACT_SPEED)
      (if (>= grabbed 0)
          (set! t_speed (/ t_speed (proto_ref (item_ref grabbed 1) 1))))
      (set! rope_len (- rope_len (* t_speed t_dt)))
      (if (<= rope_len ROPE_MIN) {
        (set! rope_len ROPE_MIN)
        (if (>= grabbed 0) (collect_grabbed))
        (set! phase PHASE_SWING)
      })
    })

    ;; 钩尖坐标
    (set! hook_x (+ PIVOT_X (* rope_len (Math.sin hook_angle))))
    (set! hook_y (+ PIVOT_Y (* rope_len (Math.cos hook_angle))))

    ;; 出钩：边界与抓取检测
    (if (== phase PHASE_EXTEND) {
      (if (or (<= hook_x 3) (>= hook_x (- SW 3)))
          (set! phase PHASE_RETRACT)
          (if (>= hook_y (- SH 4))
              (set! phase PHASE_RETRACT)
              (try_grab)))
    })

    ;; 抓到的物品跟随钩尖
    (if (>= grabbed 0) {
      (item_set grabbed 2 hook_x)
      (item_set grabbed 3 (+ hook_y 6))
    })
  }))

;; ============================================================================
;; 布局常量（绘制侧）
;; ============================================================================

(define MINER_X (+ PIVOT_X 20))    ;; 矿工贴图中心（C 版：PIVOT_X+20, PIVOT_Y-20）
(define MINER_Y (- PIVOT_Y 20))

;; 绘制色（与 C 版一致）
(define SKY_R 25) (define SKY_G 45) (define SKY_B 75)
(define GND_R 110) (define GND_G 75) (define GND_B 40)
(define LINE_R 200) (define LINE_G 160) (define LINE_B 110)

;; ============================================================================
;; 贴图尺寸预热：进入游戏时按真实 PNG 尺寸填充尺寸表
;;   （C 版用 ui_icon_get_size 触发解码并缓存；这里用 Gfx.sprite_info —— 同一语义）。
;;   贴图缺失（#null）时保留原型包围盒，游戏仍可玩。
;; ============================================================================

(define load_sprite_sizes
  (lambda () {
    (set! i 0)
    (while (< i SPR_NUM) {
      (set! sp (get_item SPR_PATH i))
      (set! k (Gfx.sprite_info sp))          ;; (宽 高) | #null
      (if (null? k)
          #f
          { (set_item! spr_w i (get_item k 0))
            (set_item! spr_h i (get_item k 1)) })
      (set! i (+ i 1))
    })
  }))

;; ============================================================================
;; 绘制
;; ============================================================================

;; 顶栏信息（C 版：「得分 %d  第%d关  剩余 %d」，12px 抗锯齿白字，左上角 6,2）
(define draw_hud
  (lambda ()
    (Gfx.draw_text
      (String.concat "得分 " (String.concat (String.atom_to_string score)
        (String.concat "  第" (String.concat (String.atom_to_string level)
          (String.concat "关  剩余 " (String.atom_to_string items_left))))))
      6 2 255 255 255)
    #t))

;; 物品（贴着精灵中心绘制；C 版用 ui_icon_draw_centered，这里用 Gfx.draw_sprite）
(define draw_items
  (lambda () {
    (set! i 0)
    (while (< i MAX_ITEMS) {
      (set! it_active (item_ref i 0))
      (if (== it_active 1) {
        (set! it_type (item_ref i 1))
        (set! sp (get_item SPR_PATH (+ SPR_GOLD_1 it_type)))
        (Gfx.draw_sprite sp (item_ref i 2) (item_ref i 3))
      })
      (set! i (+ i 1))
    })
  }))

;; 钩绳 + 钩爪：绳用 **1px 抗锯齿直线**（(Gfx.draw_line_aa ...) = C 版
;; gfx_draw_line_anti_aliasing 的吴小林算法，端点保持浮点、不取整——摆动的绳角是连续的，
;; 只有亚像素端点才得到平滑的斜线），颜色/线宽与 C 版一致（230,230,230 / 1px）。
;; 钩爪贴图以绳端锚点（上缘中点）为枢轴随绳角旋转（(Gfx.draw_sprite_rotated)）。
(define draw_hook
  (lambda ()
    (Gfx.draw_line_aa PIVOT_X PIVOT_Y hook_x hook_y 1 230 230 230)
    (Gfx.draw_sprite_rotated (get_item SPR_PATH SPR_HOOK)
                             (Math.floor hook_x) (Math.floor hook_y)
                             hook_angle)
    #t))

(define render
  (lambda () {
    (Gfx.clear)
    ;; 背景：天空 + 矿区台面 + 地下
    (Gfx.fill_rect 0 0 SW GROUND_Y SKY_R SKY_G SKY_B)
    (Gfx.fill_rect 0 GROUND_Y SW (- SH GROUND_Y) GND_R GND_G GND_B)
    (Gfx.draw_line 0 GROUND_Y SW GROUND_Y LINE_R LINE_G LINE_B)

    (draw_hud)
    (draw_items)
    (Gfx.draw_sprite (get_item SPR_PATH SPR_MINER) MINER_X MINER_Y)
    (draw_hook)

    (EX.draw_button)     ;; 右上角「退出」（并把热区登记给宿主，慢节奏也不漏点按）
    (Gfx.refresh)
  }))

;; ============================================================================
;; 输入
;; ============================================================================

;; 键盘按下沿：回车(13) / '2'(50) 发射（C 版 D 键、数字 2）；ESC(27) 由 EX 接管
(define handle_keys
  (lambda () {
    (set! k (Key.take_press))
    (while (not (== k 0)) {
      (if (== k 13) (fire) #f)
      (if (== k 50) (fire) #f)
      (set! k (Key.take_press))
    })
  }))

;; 触屏：松手沿发射（与 C 版 touch_edge UP 一致；落在「退出」按钮上的序列已被
;; (EX.check) 吞掉，不会误发射）
(define tr 0)           ;; 本拍有松开沿
(define handle_touch
  (lambda () {
    (set! tr (if (Touch.take_release) 1 0))
    (if (== tr 1) (fire) #f)
  }))

;; ---- 诊断（宿主 stdout / 浏览器控制台；端到端测试据此断言游戏真的在推进）----
;;   只在**状态变化**时打印：发射/抓取/结算/过关，正常游玩不会刷屏。
(define last_phase -1)
(define last_score -1)
(define last_level 0)

(define trace_step
  (lambda ()
    (if (not (== (Math.floor score) last_score))
        { (set! last_score (Math.floor score))
          (display "GMSCORE score=") (display last_score)
          (display " items=") (display items_left)
          (display " level=") (display level) (newline) }
        #f)
    (if (not (== phase last_phase))
        { (set! last_phase phase)
          (if (== phase PHASE_EXTEND)
              { (display "GMFIRE") (newline) }
              #f)
          (if (and (== phase PHASE_RETRACT) (>= grabbed 0))
              { (display "GMGRAB idx=") (display grabbed)
                (display " type=") (display (item_ref grabbed 1)) (newline) }
              #f) }
        #f)
  ))

;; ============================================================================
;; 主循环：一拍 = 采样输入 → 逻辑推进 → 整帧重绘
;; ============================================================================

(define now_ms 0)
(define last_ms 0)
(define dt 0)

(define tick
  (lambda () {
    (EX.check)
    ;; 实测步长（step 内部钳到 DT_MAX）
    (set! now_ms (System.timestamp))
    (set! dt (- now_ms last_ms))
    (set! last_ms now_ms)
    (handle_keys)
    (handle_touch)
    (step dt)
    (trace_step)
    (render)
  }))

;; ============================================================================
;; 启动
;; ============================================================================

(define boot
  (lambda () {
    (set! SW (Gfx.width))
    (set! SH (Gfx.height))
    (load_sprite_sizes)    ;; 一次性读贴图尺寸（并触发宿主侧解码缓存）
    (reset)
    (set! last_score (Math.floor score))
    (set! last_phase phase)
    (set! last_ms (System.timestamp))
    (EX.install_with render)
    (tick)
    (System.set_interval TICK_MS tick)
    (display "GMSTAT sprites=") (display SPR_NUM) (newline)
    (print_level_line "GMSTAT")
  }))

(boot)
