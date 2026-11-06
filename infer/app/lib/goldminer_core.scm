;; ============================================================================
;; 黄金矿工纯逻辑库（Gold Miner Core）—— 供 app/goldminer.scm 与回归测试共用
;;
;; 逐条对照 infer/ui_goldminer.c（C 版，STATE_GAME_MENU 里的「黄金矿工」）实现：
;;   · 场景：320x240，地表线 y=70，摆动支点 (160,66)，绳长下限 16px；
;;   · 摆动：hook_angle = 1.30 * sin(phase)，速度 2.2 + 0.1*(level-1) rad/s；
;;   · 出钩 300px/s；撞到边界（x<=3 / x>=W-3 / y>=H-4）或抓到物体即回收；
;;   · 回收 240px/s，抓到物体时除以重量；回到绳长下限处结算分值；
;;   · 命中判定：钩尖进入以物体中心为圆心、直径 = 精灵宽高较小者 的内切圆；
;;   · 关卡：大/中金块各 1，小金块 3+min(level-1,3)，岩石 3+level/2，
;;     钻石 level>=2，福袋 1，宝箱 level>=3；随机放置并拒绝重叠（外扩 8px 椭圆判定）；
;;     收完该关全部物品即进入下一关。
;;
;; 本模块不依赖 Gfx/Touch（只用 Math 的三角函数/随机/取整），因此可以脱离界面
;; 单独回归。精灵尺寸由调用方通过 (set_sprite_size id w h) 注入——C 版是从
;; ui_icon_get_size 读贴图实际宽高，Scheme 侧由 App 用 (Gfx.sprite_info ...) 读，
;; 这样"关卡生成与命中判定用真实贴图尺寸"这条语义与 C 版一致，而核心库保持无 UI 依赖。
;;
;; 解释器约束（manual.md）：算术恒二参数、无一元取负；无 list/let/map/filter；
;; cond 每子句只接受单个表达式；/ 与 mod 恒返回浮点（取整用 Math.floor）；
;; '() 是模块级静态槽（空表一律 (cdr '(0)) 现造）；while 体不是独立词法区域
;; → 所有局部量都用模块级 define + set!。
;; ============================================================================

(native Math)

;; ---- 场景常量（与 C 版一致）----------------------------------------------

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

;; 屏幕尺寸（由 (init w h) 注入；默认 320x240，与 C 版一致）
(define sw 320)
(define sh 240)

;; ---- 阶段（与 C 版 GM_Phase 一一对应）------------------------------------

(define PHASE_SWING 0)      ;; 摆动待发射
(define PHASE_EXTEND 1)     ;; 出钩
(define PHASE_RETRACT 2)    ;; 回收

;; ---- 精灵表 --------------------------------------------------------------
;; 顺序与 C 版 GM_Sprite_Id 完全一致；路径亦与 C 版 S_GM_SPRITE_PATH 相同
;; （C 版为 PLATFORM_ROOT_DIR "/icon/gm_*.png"，Scheme 侧路径相对**业务资源根**，
;;  而资源根就是 PLATFORM_ROOT_DIR，故指同一批文件）。
;; 全部是**透明背景** PNG —— 绘制必须走 alpha 混合的 (Gfx.draw_sprite ...)。
(define SPR_MINER 0)
(define SPR_HOOK 1)
(define SPR_GOLD_1 2)
(define SPR_ROCK 5)
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

;; 精灵尺寸缓存（像素）：进入游戏时由 App 用 (set_sprite_size ...) 一次性填充，
;; 之后关卡生成与渲染全程查表（同 C 版 s_gm_sprite_w/h 的用途）。
;; 初值是 C 版"无贴图时"的原型包围盒，贴图缺失时退化为同样的布局。
(define spr_w (cdr '(0)))
(define spr_h (cdr '(0)))

;; ---- 物品原型（7 种，顺序与 C 版 GM_Item_Type / 精灵 GOLD_1 起一一对应）-----
;; 扁平表，索引 = type*6 + {0=value 1=weight 2=radius 3=r 4=g 5=b}
(define T_GOLD_1 0)
(define T_GOLD_2 1)
(define T_GOLD_3 2)
(define T_ROCK 3)
(define T_DIAMOND 4)
(define T_BONUS_1 5)
(define T_BONUS_2 6)
(define TYPE_NUM 7)

(define PROTO '(
   500 2.5  13  255 200  30    ; 大金块
   300 1.8  10  255 208  45    ; 中金块
   200 1.2   8  255 215  60    ; 小金块
    80 3.5  12  150 150 160    ; 岩石
   600 0.8   6  120 220 255    ; 钻石
   350 0.6   7  120 230 120    ; 福袋
  1000 0.5   7  230  90 220))  ; 宝箱

;; ---- 游戏状态（导入方以 GM.名字 直接读取）--------------------------------

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

;; ---- 模块级临时量（热路径局部量一律提到模块级：见 tetris_core.scm 同一说明）----

(define t_i 0) (define t_k 0) (define t_t 0) (define t_a 0)
(define t_w 0) (define t_h 0) (define t_x 0) (define t_y 0)
(define t_xr 0) (define t_yr 0) (define t_ok 0) (define t_ov 0)
(define t_dx 0) (define t_dy 0) (define t_nx 0) (define t_ny 0)
(define t_ox 0) (define t_oy 0) (define t_rr 0) (define t_done 0)
(define t_dt 0) (define t_speed 0) (define t_ms 0)

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
    ;; 精灵尺寸初值 = C 版原型包围盒
    ;; 0 矿工 16x26 / 1 钩爪 12x8 / 其余 = 原型直径
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

;; 由 App 注入真实贴图尺寸（(Gfx.sprite_info 路径) 的宽高；缺失时传原型值即可）
(define set_sprite_size
  (lambda (id w h) {
    (set_item! spr_w id w)
    (set_item! spr_h id h)
  }))

;; 屏幕尺寸（App 启动时按 Gfx.width/height 注入）
(define set_screen
  (lambda (w h) {
    (set! sw w)
    (set! sh h)
  }))

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
      (set! t_xr (Math.floor (- (- sw t_w) 12)))
      (if (< t_xr 1) (set! t_xr 1))
      (set! t_yr (Math.floor (- (- (- sh GROUND_Y) t_h) 24)))
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

;; ============================================================================
;; 游戏动作
;; ============================================================================

;; 发射（仅摆动状态有效；落点/命中在 step 里判定）
(define fire
  (lambda ()
    (if (== phase PHASE_SWING) (set! phase PHASE_EXTEND))
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

;; 出钩状态的抓取判定
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
          (set! t_done 1)
        })
      })
      (set! t_i (+ t_i 1))
    })
  }))

;; 收回完成：结算抓到的物品，清空则进入下一关
(define collect_grabbed
  (lambda () {
    (set! score (+ score (proto_ref (item_ref grabbed 1) 0)))
    (item_set grabbed 0 0)
    (set! grabbed -1)
    (set! items_left (- items_left 1))
    (if (<= items_left 0) {
      (set! level (+ level 1))
      (generate_level)
    })
  }))

;; 一拍逻辑推进（dt_ms 为实测毫秒间隔，内部钳到 [0, DT_MAX]）
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
      (if (or (<= hook_x 3) (>= hook_x (- sw 3)))
          (set! phase PHASE_RETRACT)
          (if (>= hook_y (- sh 4))
              (set! phase PHASE_RETRACT)
              (try_grab)))
    })

    ;; 抓到的物品跟随钩尖
    (if (>= grabbed 0) {
      (item_set grabbed 2 hook_x)
      (item_set grabbed 3 (+ hook_y 6))
    })
  }))
