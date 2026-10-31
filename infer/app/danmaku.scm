;; ============================================================================
;; 弹幕陨石（danmaku）—— 程序化 sprite 的竖版陨石射击 App
;;
;; **没有任何位图资源**：飞船、陨石、爆炸、子弹、星空全部用 Gfx 的
;; fill_rect / fill_circle / draw_line 现拼出来。这是本 App 的全部视觉技法：
;;
;;   飞船   —— 15 行逐行 fill_rect 拼出三角机身（每行的半宽查 HW 表），
;;             行色沿 i 做线性渐变（机头亮青 → 机尾深蓝），座舱是 fill_circle，
;;             尾焰是两枚 fill_rect（长度每帧随机抖动 = 火焰闪烁）。
;;   陨石   —— 8 边不规则多边形 + 旋转：顶点单位坐标 (UX,UY) 在**加载期**算好
;;             （3 种形状 × 8 顶点；方向角 45° 步进 + 抖动 JA，半径系数 0.76~1.12），
;;             运行时每颗只算一次 (ca,sa) 旋转矩阵，再把 8 个顶点变换到屏幕坐标，
;;             然后**扫描线**逐行求交 → 插入排序 → 按奇偶规则成对 fill_rect
;;             （凹缺口不会被填平）；每行按"距中心高度"取三档明暗（顶光），
;;             最后 draw_line 描 8 条亮边 + 2 个 fill_circle 陨坑。
;;             旋转 = 每颗陨石自己的角速度 aspd，形状/相位各不相同。
;;   爆炸   —— 粒子：火星（颜色按剩余寿命比走 白黄 → 橙 → 暗红，2px → 1px）+
;;             中心闪光（半径随寿命收缩的三层同心圆）。
;;   星空   —— 46 颗 fill_rect 星点，分 3 档亮度/速度（视差）。
;;
;; 玩法：触屏拖动飞船（飞船跟在指尖上方 26px，不被手指挡住），**自动开火**；
;;       陨石按波次落下，难度随波次递增（下落速度 / 刷怪频率 / 射速 / 第 4 波起三连发）；
;;       大陨石被击碎分裂成两颗小的；第 2 波起掺入**横掠型**陨石——它在飞船的高度带
;;       横穿屏幕，垂直弹幕打不到，必须挪位躲开（否则"蹲着不动"几乎无敌）；
;;       撞到飞船扣一条命（3 条命，1.8s 无敌闪烁），命尽 → 游戏结束 → 点按重开。
;;
;; 算力：单拍 ≈ 2.5~3 万条 VM 指令（312500 的时间片余量约 10 倍），TICK_MS=33。
;;
;; 解释器约束（manual.md + doc/AGENTS.md，逐条遵守）：
;;   · 算术恒二参数 —— **多元会静默算错**（`(+ SW r 6.0)` 实测得 14，应为 334）；
;;     没有一元取负，写 (- 0 x)；
;;   · 没有 list/append/map/let，空表只能写 '()；
;;   · **'() 字面量是模块级静态槽**：在会被多次调用的 lambda 里 `(define l '())`
;;     再 push 会跨调用累积。故所有定长表都在**顶层**建一次（一张零模板 ZT +
;;     cons/cdr 整表复制得到独立副本），此后只 set_item!；
;;   · 运行期不造字符串（<=32 字符会被驻留且永不释放）：数字用 10 个编译期字模
;;     逐位贴（DGS + draw_int，与 mlp.scm / callcc.scm 同一手法）；
;;   · / 与 mod 恒返回浮点；列表下标与绘图坐标一律 Math.floor/round 收口；
;;   · 块内只有 set!，局部量一律写在 lambda 体顶部（每次调用独立）。
;;
;; **draw_line 的两个坑**（第一个是宿主真 bug，已修 graphics.c）：
;;   1) gfx_draw_line 的慢路径（非 RGB565 帧缓冲，WASM GUI 就是 RGB888）历史上用
;;      int32 与 uint32 混比控制循环：反向水平/垂直线（x1>x2 / y1>y2）递减到 -1 后
;;      被提升为 4294967295 → 循环永不结束（整页卡死）。本 App 的陨石描边必然会
;;      产生反向的竖直边，于是踩中。修法是两端先转 int32 再比较；本文件另加
;;      `line_xy` 把垂直/水平线段统一按升序下发，做双保险。
;;   2) 坐标是 uint32：负数会变成天文数字。故本 App 的描边坐标**先判界再画**；
;;      fill_rect / fill_circle 本身对负坐标安全（C 侧按 int32 裁剪）。
;;
;; 退出：右上角「退出」按钮 / 硬 A/ESC → 宿主全局确认框（app/lib/exit.scm）。
;; ============================================================================

(native Gfx)
(native Math)
(native System)
(native Touch)
(native App)

(import EX "app/lib/exit.scm")

;; ---- 帧节奏 --------------------------------------------------------------
;; 定时器周期只是"希望多久跑一拍"；物理步长用 System.timestamp 的实测间隔
;; （钳到 DT_MAX 防卡顿跳变，与 tetris.scm 同一手法）。
(define TICK_MS 33)
(define DT_MAX 50)

;; ---- 画面几何 ------------------------------------------------------------
(define SW (Gfx.width))
(define SH (Gfx.height))
(define HUD_H 19)                       ; 顶部状态栏（EX 的退出按钮画在 y 2..17 内）

;; ---- 容量（定长表的上限） ------------------------------------------------
(define NBUL 12)                        ; 子弹
(define NAST 20)                        ; 陨石
(define NPART 96)                       ; 爆炸粒子
(define NSTAR 46)                       ; 星点
(define CAP 96)                         ; 零模板长度（>= 上面最大者）

;; ---- 定长表：顶层建一次，此后只 set_item! --------------------------------
;; 一张零模板 ZT，其余表用 (cp ZT) 整表复制（cons/cdr 都是整表复制）得到
;; **独立**的列表对象。cp 体内没有 '() 字面量，因此反复调用不会累积。
(define ZT (cons 0 '()))
(define zti 1)
(while (< zti CAP) {
  (push ZT 0)
  (set! zti (+ zti 1))
})
(define cp (lambda (t) (cdr (cons 0 t))))

;; 子弹：x / y / 竖直速度（负 = 向上）
(define bxx (cp ZT))
(define byy (cp ZT))
(define bvv (cp ZT))
;; 陨石：x / y / vx / vy / 半径 / 旋转角 / 角速度 / 形状号
(define axx (cp ZT))
(define ayy (cp ZT))
(define avx (cp ZT))
(define avy (cp ZT))
(define arr (cp ZT))
(define aang (cp ZT))
(define aspd (cp ZT))
(define ashp (cp ZT))
(define aknd (cp ZT))                   ; 0=下落型（撞墙反弹） 1=横掠型（穿屏而过）
;; 粒子：x / y / vx / vy / 剩余寿命 / 初始寿命
(define pxx (cp ZT))
(define pyy (cp ZT))
(define pvx (cp ZT))
(define pvy (cp ZT))
(define plf (cp ZT))
(define pmx (cp ZT))
(define pkd (cp ZT))                    ; 0=火星  1=中心闪光
;; 星点：x / y / 下落速度 / 亮度档
(define stx (cp ZT))
(define sty (cp ZT))
(define stv (cp ZT))
(define stc (cp ZT))
;; 当前陨石的 8 个顶点（屏幕坐标，逐个陨石用完即弃）
(define vx8 (cp ZT))
(define vy8 (cp ZT))
;; 扫描线与 8 条边的交点（奇偶填充用，最多 8 个）
(define XC (cp ZT))

;; ---- 伪随机（LCG） --------------------------------------------------------
;; 状态恒 < 2048，乘法 < 2^19 —— wasm32 单精度下逐位精确（见 doc/AGENTS.md）。
(define lcg 1)
(define rnd
  (lambda ()
    {
      (set! lcg (mod (+ (* lcg 33) 7) 2048))
      (/ lcg 2048.0)
    }))

;; 用宿主时间戳折叠出种子（mod 会把参数转浮点、低位被抹平，故用位运算）
(define reseed
  (lambda ()
    {
      (define t 0)
      (define s 0)
      (define i 0)
      (set! t (System.timestamp))
      (set! s (Math.bit_and t 2047))
      (set! s (Math.bit_xor s (Math.bit_and (Math.bit_lshr t 11) 2047)))
      (set! s (Math.bit_xor s (Math.bit_and (Math.bit_lshr t 21) 2047)))
      (if (== s 0) (set! s 1) #f)
      (set! lcg s)
      (set! i 0)
      (while (< i 8) {
        (set! lcg (mod (+ (* lcg 33) 7) 2048))
        (set! i (+ i 1))
      })
    }))

;; ---- 陨石多边形的单位顶点（加载期算一次） --------------------------------
;; 3 种形状 × 8 顶点 = 24 项；顶点 k 的方向角固定为 k×45°，半径系数见 UF。
;; 运行时只需 (ca,sa) 旋转 + 乘半径，不需要任何三角函数（省 VM 指令）。
(define UF '(1.00 0.78 1.06 0.86 1.00 0.80 1.04 0.88
             0.86 1.12 0.76 1.04 0.90 1.10 0.78 1.02
             1.08 0.82 0.96 0.76 1.10 0.84 1.00 0.80))
;; 顶点方向的抖动（度）：固定 45° 步进 + 抖动，多边形才不会像"正八边形"
(define JA '(0 5 -4 3 -5 4 -3 6
             -6 3 5 -4 2 -5 4 -2
             4 -5 2 6 -3 5 -6 3))
(define UX (cp ZT))
(define UY (cp ZT))
(define ufi 0)
(define ufa 0.0)
(while (< ufi 24) {
  (set! ufa (* (+ (* (mod ufi 8) 45) (get_item JA ufi)) 0.0174532925))
  (set_item! UX ufi (* (get_item UF ufi) (Math.cos ufa)))
  (set_item! UY ufi (* (get_item UF ufi) (Math.sin ufa)))
  (set! ufi (+ ufi 1))
})

;; ---- 飞船半宽表（15 行，机头在上） --------------------------------------
(define HW '(0 1 1 2 2 3 4 5 6 7 9 11 12 13 13))

;; ---- 数字字模（运行期零字符串分配） -------------------------------------
(define DGS '("0" "1" "2" "3" "4" "5" "6" "7" "8" "9"))
(define DW 7)

(define ndigits
  (lambda (v)
    {
      (define n 1)
      (define t 10.0)
      (while (>= v t) {
        (set! t (* t 10.0))
        (set! n (+ n 1))
      })
      n
    }))

;; 画一个非负整数（pad 大于实际位数时左补零），返回占用像素宽度
(define draw_int
  (lambda (v pad x y r g b)
    {
      (define n (ndigits v))
      (define nd (if (> pad n) pad n))
      (define t 1.0)
      (define k 1)
      (define i 0)
      (define d 0)
      (define rem v)
      (while (< k nd) {
        (set! t (* t 10.0))
        (set! k (+ k 1))
      })
      (while (< i nd) {
        (set! d (Math.floor (/ rem t)))
        (if (> d 9) (set! d 9) #f)
        (Gfx.draw_text (get_item DGS d) (+ x (* i DW)) y r g b)
        (set! rem (- rem (* d t)))
        (set! t (/ t 10.0))
        (set! i (+ i 1))
      })
      (* nd DW)
    }))

;; ============================================================================
;; 游戏状态
;; ============================================================================

(define gt 0)                           ; 本 App 自己的游戏时钟（ms，累加 dt）
(define tnow 0)                         ; 本拍 System.timestamp
(define tlast 0)
(define dms 0)                          ; 本拍毫秒步长
(define dts 0.0)                        ; 本拍秒步长

(define playing 1)                      ; 1 出怪/开火，0 停止（结束态）
(define over 0)                         ; 1 = 游戏结束浮层
(define score 0)
(define best 0)
(define wave 1)
(define lives 3)
(define fire_t 0)                       ; 开火计时（ms）
(define spawn_left 0)                   ; 本波还没刷出的陨石数
(define spawn_t 0)                      ; 刷怪计时（ms）
(define wave_t 0)                       ; 波次间隔计时（ms）
(define banner_t 0.0)                   ; 「第 N 波」横幅剩余秒
(define hint_t 6.0)                     ; 操作提示剩余秒
(define invuln 0.0)                     ; 无敌剩余秒
(define flash 0.0)                      ; 受击红框剩余秒

(define ship_x 160.0)
(define ship_y 200.0)
(define tgo 0.0)                        ; 触点原始坐标
(define tgt 0.0)
(define tgx 0.0)                        ; 飞船目标位置
(define tgy 0.0)
(define down 0)                         ; 本拍触屏电平
(define prev_down 0)

(define nbul 0)
(define nast 0)
(define npart 0)

;; 跨函数临时量（按"谁写谁读"分前缀，绝不共用）
(define c_i 0) (define c_j 0) (define c_k 0) (define c_k2 0)
(define c_dx 0.0) (define c_dy 0.0) (define c_rr 0.0) (define c_hit 0)
(define g_row 0.0) (define g_rowmax 0.0)
(define g_xa 0.0) (define g_xb 0.0) (define g_xmn 0.0) (define g_xmx 0.0)
(define g_y0 0.0) (define g_y1 0.0) (define g_ux 0.0) (define g_uy 0.0)
(define g_ca 0.0) (define g_sa 0.0) (define g_cx 0.0) (define g_cy 0.0)
(define g_rr 0.0) (define g_sh 0) (define g_itr 0.0)
(define s_r 0.0) (define s_x 0.0) (define s_y 0.0) (define s_vy 0.0)
(define d_y 0.0) (define d_x 0.0)
(define _timer 0)

;; ============================================================================
;; 生成 / 回收（定长表的增删；删除用"末项覆盖"保持 O(1) 且不移动其它项）
;; ============================================================================

(define addbul
  (lambda (x y vy)
    (if (< nbul NBUL)
        {
          (set_item! bxx nbul x)
          (set_item! byy nbul y)
          (set_item! bvv nbul vy)
          (set! nbul (+ nbul 1))
        }
        #f)))

(define rmbul
  (lambda (i)
    {
      (set! nbul (- nbul 1))
      (set_item! bxx i (get_item bxx nbul))
      (set_item! byy i (get_item byy nbul))
      (set_item! bvv i (get_item bvv nbul))
    }))

(define addast
  (lambda (x y r vx vy ang spd shp knd)
    (if (< nast NAST)
        {
          (set_item! axx nast x)
          (set_item! ayy nast y)
          (set_item! arr nast r)
          (set_item! avx nast vx)
          (set_item! avy nast vy)
          (set_item! aang nast ang)
          (set_item! aspd nast spd)
          (set_item! ashp nast shp)
          (set_item! aknd nast knd)
          (set! nast (+ nast 1))
        }
        #f)))

(define rmast
  (lambda (i)
    {
      (set! nast (- nast 1))
      (set_item! axx i (get_item axx nast))
      (set_item! ayy i (get_item ayy nast))
      (set_item! arr i (get_item arr nast))
      (set_item! avx i (get_item avx nast))
      (set_item! avy i (get_item avy nast))
      (set_item! aang i (get_item aang nast))
      (set_item! aspd i (get_item aspd nast))
      (set_item! ashp i (get_item ashp nast))
      (set_item! aknd i (get_item aknd nast))
    }))

(define addpart
  (lambda (x y vx vy life)
    (if (< npart NPART)
        {
          (set_item! pxx npart x)
          (set_item! pyy npart y)
          (set_item! pvx npart vx)
          (set_item! pvy npart vy)
          (set_item! plf npart life)
          (set_item! pmx npart life)
          (set_item! pkd npart 0)
          (set! npart (+ npart 1))
        }
        #f)))

(define rmpart
  (lambda (i)
    {
      (set! npart (- npart 1))
      (set_item! pxx i (get_item pxx npart))
      (set_item! pyy i (get_item pyy npart))
      (set_item! pvx i (get_item pvx npart))
      (set_item! pvy i (get_item pvy npart))
      (set_item! plf i (get_item plf npart))
      (set_item! pmx i (get_item pmx npart))
      (set_item! pkd i (get_item pkd npart))
    }))

;; 爆炸：n 个粒子，随机方向、随机速度、随机寿命
(define burst
  (lambda (cx cy n spd frad)
    {
      (define i 0)
      (define a 0.0)
      (define s 0.0)
      ;; 中心闪光：寿命极短，半径随寿命收缩（存 pvx）
      (if (< npart NPART)
          {
            (addpart cx cy frad 0.0 0.20)
            ;; 注意下标：addpart 已把 npart 加一，闪光本身在 npart-1
            (set_item! pkd (- npart 1) 1)
          }
          #f)
      (while (< i n) {
        (set! a (* (rnd) 6.2831853))
        (set! s (* spd (+ 0.25 (rnd))))
        (addpart cx cy (* (Math.cos a) s) (* (Math.sin a) s)
                 (+ 0.42 (* 0.55 (rnd))))
        (set! i (+ i 1))
      })
    }))

;; 大陨石被击碎 → 分裂成两颗小的（撞毁飞船时也用它，视觉一致）
(define split_ast
  (lambda (i)
    {
      (define cx (get_item axx i))
      (define cy (get_item ayy i))
      (define rr (get_item arr i))
      (if (>= rr 12.0)
          (if (< nast (- NAST 2))
              {
                (addast (- cx 3.0) cy (* rr 0.52) (- 0.0 (+ 45.0 (* (rnd) 35.0)))
                        (+ (* (get_item avy i) 0.6) (* (rnd) 26.0))
                        (* (rnd) 6.2831853) 1.6 (Math.floor (* (rnd) 3.0)) 0)
                (addast (+ cx 3.0) cy (* rr 0.52) (+ 45.0 (* (rnd) 35.0))
                        (+ (* (get_item avy i) 0.6) (* (rnd) 26.0))
                        (* (rnd) 6.2831853) -1.6 (Math.floor (* (rnd) 3.0)) 0)
              }
              #f)
          #f)
    }))

;; ============================================================================
;; 局面
;; ============================================================================

(define new_game
  (lambda ()
    {
      (define i 0)
      (set! playing 1)
      (set! over 0)
      (set! score 0)
      (set! wave 1)
      (set! lives 3)
      (set! fire_t 0)
      (set! spawn_left 6)
      (set! spawn_t 800)                ; 首只陨石立刻刷出
      (set! wave_t 0)
      (set! banner_t 2.4)               ; 开局横幅停久一点（首屏就能读到"第 1 波"）
      (set! hint_t 6.0)
      (set! invuln 1.2)
      (set! flash 0.0)
      (set! ship_x (/ SW 2.0))
      (set! ship_y 200.0)
      (set! nbul 0)
      (set! nast 0)
      (set! npart 0)
      ;; 星点重撒
      (set! i 0)
      (while (< i NSTAR) {
        (set_item! stx i (+ 2.0 (* (rnd) (- SW 4))))
        (set_item! sty i (+ HUD_H (* (rnd) (- SH HUD_H))))
        (set_item! stv i (+ 9.0 (* (rnd) 34.0)))
        (set_item! stc i (Math.floor (* (rnd) 3.0)))
        (set! i (+ i 1))
      })
    }))

;; 刷一只陨石：半径 9~16，横向随机，速度随波次递增
(define spawn_one
  (lambda ()
    {
      (define r 0.0)
      (define x 0.0)
      (define y 0.0)
      (define vx 0.0)
      (define vy 0.0)
      (define kd 0)
      (if (and (>= wave 2) (< (rnd) (+ 0.12 (* wave 0.05))))
          ;; 横掠型：从左右边缘**横着**飞过（第 2 波起出现，概率随波次上升）。
          ;; 它飞在飞船的高度带上，垂直弹幕打不到它 —— 玩家必须移动飞船，
          ;; 让弹幕柱与它的航路相交，或者干脆躲开。这条机制是"不能一直蹲着"的关键。
          {
            (set! kd 1)
            (set! r (+ 7.0 (* (rnd) 5.0)))
            (set! y (+ 118.0 (* (rnd) 92.0)))
            (set! vy (* (rnd) 14.0))
            (if (> (rnd) 0.5)
                {
                  (set! x (- 0.0 (+ r 6.0)))
                  (set! vx (+ 100.0 (* (rnd) 60.0)))
                }
                {
                  (set! x (+ (+ SW r) 6.0))
                  (set! vx (- 0.0 (+ 100.0 (* (rnd) 60.0))))
                })
          }
          ;; 下落型
          {
            (set! kd 0)
            (set! r (+ 9.0 (* (rnd) 8.0)))
            (set! x (+ 15.0 (* (rnd) (- SW 30))))
            (set! y (- 0.0 (+ r 4.0)))
            (set! vx (- (* (rnd) 90.0) 45.0))
            (set! vy (+ (+ 52.0 (* wave 7.0)) (* (rnd) 38.0)))
          })
      (addast x y r vx vy (* (rnd) 6.2831853)
              (* (+ 0.5 (* (rnd) 1.2)) (if (> (rnd) 0.5) 1.0 -1.0))
              (Math.floor (* (rnd) 3.0)) kd)
    }))

;; ============================================================================
;; 每拍推进
;; ============================================================================

(define upd_stars
  (lambda ()
    {
      (define i 0)
      (define y 0.0)
      (while (< i NSTAR) {
        (set! y (+ (get_item sty i) (* (get_item stv i) dts)))
        (if (> y SH) (set! y (+ y (- HUD_H SH))) #f)
        (set_item! sty i y)
        (set! i (+ i 1))
      })
    }))

(define upd_parts
  (lambda ()
    {
      (define i 0)
      (define l 0.0)
      (define dr 0.0)
      (set! dr (- 1.0 (* 1.3 dts)))     ; 每秒约衰减到 0.27
      (while (< i npart) {
        (set! l (- (get_item plf i) dts))
        (if (<= l 0.0)
            (rmpart i)
            {
              (set_item! plf i l)
              (set_item! pxx i (+ (get_item pxx i) (* (get_item pvx i) dts)))
              (set_item! pyy i (+ (get_item pyy i) (* (get_item pvy i) dts)))
              (set_item! pvx i (* (get_item pvx i) dr))
              (set_item! pvy i (* (get_item pvy i) dr))
              (set! i (+ i 1))
            })
      })
    }))

(define upd_bullets
  (lambda ()
    {
      (define i 0)
      (define y 0.0)
      (while (< i nbul) {
        (set! y (+ (get_item byy i) (* (get_item bvv i) dts)))
        ;; 出了游玩区（HUD 下沿）即回收：否则子弹会在**屏幕上方**命中陨石，
        ;; 爆炸与得分都发生在看不见的地方（实测第一版就是这样）
        (if (< y HUD_H)
            (rmbul i)
            {
              (set_item! byy i y)
              (set! i (+ i 1))
            })
      })
    }))

(define upd_asts
  (lambda ()
    {
      (define i 0)
      (define x 0.0)
      (define y 0.0)
      (define r 0.0)
      (define px0 0.0)
      (while (< i nast) {
        (set! r (get_item arr i))
        (set! px0 (get_item axx i))
        (set! x (+ px0 (* (get_item avx i) dts)))
        (set! y (+ (get_item ayy i) (* (get_item avy i) dts)))
        (if (== (get_item aknd i) 1)
            ;; 横掠型：不反弹，飞出左右边缘即回收
            (if (or (< x (- 0.0 (+ r 14.0))) (> x (+ (+ SW r) 14.0)))
                (rmast i)
                {
                  (set_item! axx i x)
                  (set_item! ayy i y)
                  (set_item! aang i (+ (get_item aang i) (* (get_item aspd i) dts)))
                  (set! i (+ i 1))
                })
            ;; 下落型：掉出下缘即回收；撞左右壁反弹（只在"从里往外穿"时翻向，
            ;; 否则刚生成在屏幕外的陨石会被立刻弹回去）
            (if (> y (+ SH r))
                (rmast i)
                {
                  (if (and (< x r) (>= px0 r))
                      {
                        (set! x r)
                        (set_item! avx i (- 0.0 (get_item avx i)))
                      }
                      #f)
                  (if (and (> x (- SW r)) (<= px0 (- SW r)))
                      {
                        (set! x (- SW r))
                        (set_item! avx i (- 0.0 (get_item avx i)))
                      }
                      #f)
                  (set_item! axx i x)
                  (set_item! ayy i y)
                  (set_item! aang i (+ (get_item aang i) (* (get_item aspd i) dts)))
                  (set! i (+ i 1))
                })
      )
      })
    }))

;; 开火：射速随波次提高（下限 92ms）；第 4 波起三连发
(define upd_fire
  (lambda ()
    {
      (define fi 0)
      (set! fire_t (+ fire_t dms))
      (set! fi (- 122 (* wave 5)))
      (if (< fi 78) (set! fi 78) #f)
      (if (>= fire_t fi)
          {
            (set! fire_t (- fire_t fi))
            (addbul ship_x (- ship_y 14.0) -390.0)
            (if (>= wave 4)
                {
                  (addbul (- ship_x 9.0) (- ship_y 6.0) -390.0)
                  (addbul (+ ship_x 9.0) (- ship_y 6.0) -390.0)
                }
                #f)
          }
          #f)
    }))

;; 刷怪 / 波次推进：本波清场后隔 900ms 进入下一波
(define upd_waves
  (lambda ()
    {
      (define iv 0)
      (define n 0)
      (if (> spawn_left 0)
          {
            (set! iv (- 780 (* wave 70)))
            (if (< iv 300) (set! iv 300) #f)
            (set! spawn_t (+ spawn_t dms))
            (if (>= spawn_t iv)
                {
                  (set! spawn_t 0)
                  (set! spawn_left (- spawn_left 1))
                  (spawn_one)
                }
                #f)
          }
          (if (== nast 0)
              {
                (set! wave_t (+ wave_t dms))
                (if (>= wave_t 900)
                    {
                      (set! wave (+ wave 1))
                      (set! n (+ 4 (* 2 wave)))
                      (set! spawn_left n)
                      (set! spawn_t 0)
                      (set! wave_t 0)
                      (set! banner_t 1.3)
                      ;; 清场奖励：每波 +5×波次，避免"只靠击碎"单调
                      (set! score (+ score (* 5 wave)))
                    }
                    #f)
              }
              #f)
      )
    }))

;; 子弹 × 陨石
(define collide_bullets
  (lambda ()
    {
      (define i 0)
      (define j 0)
      (define hit 0)
      (while (< i nbul) {
        (set! hit -1)
        (set! j 0)
        (while (< j nast) {
          (if (== hit -1)
              {
                (set! c_dx (- (get_item bxx i) (get_item axx j)))
                (set! c_dy (- (get_item byy i) (get_item ayy j)))
                (set! c_rr (get_item arr j))
                (set! c_rr (+ c_rr 3.0))
                (if (< (+ (* c_dx c_dx) (* c_dy c_dy)) (* c_rr c_rr))
                    (set! hit j)
                    #f)
              }
              #f)
          (set! j (+ j 1))
        })
        (if (== hit -1)
            (set! i (+ i 1))
            {
              (set! score (+ score (if (>= (get_item arr hit) 12.0) 20 10)))
              (burst (get_item axx hit) (get_item ayy hit)
                     (+ 6 (Math.floor (get_item arr hit))) 90.0
                     (* (get_item arr hit) 1.3))
              (split_ast hit)
              (rmast hit)
              (rmbul i)                  ; 不推进 i：末项已搬到这里
            })
      })
    }))

;; 陨石 × 飞船
(define collide_ship
  (lambda ()
    {
      (define j 0)
      (define r 0.0)
      (while (< j nast) {
        (set! c_dx (- ship_x (get_item axx j)))
        (set! c_dy (- ship_y (get_item ayy j)))
        (set! r (+ (get_item arr j) 6.0))
        (if (and (<= invuln 0.0) (< (+ (* c_dx c_dx) (* c_dy c_dy)) (* r r)))
            {
              (burst ship_x ship_y 16 130.0 18.0)
              (burst (get_item axx j) (get_item ayy j) 8 90.0 (get_item arr j))
              (rmast j)
              (set! lives (- lives 1))
              (set! invuln 1.8)
              (set! flash 0.2)
              (if (<= lives 0)
                  {
                    (set! lives 0)
                    (set! playing 0)
                    (set! over 1)
                    (burst ship_x ship_y 30 190.0 28.0)
                  }
                  #f)
            }
            #f)
        (set! j (+ j 1))
      })
    }))

;; ============================================================================
;; 绘制
;; ============================================================================

;; 画一条线：垂直/水平线段一律按坐标升序传下去。
;; 为什么需要这层归一（实测踩到的宿主缺陷）：gfx_draw_line 的**慢路径**（非 RGB565
;; 帧缓冲，WASM GUI 就是 RGB888）历史上用 int32 与 uint32 混比来控制循环，
;; 反向竖线（y1 > y2）递减到 -1 后会被提升成 4294967295，比较恒真 → 死循环。
;; 已经修了 graphics.c（见 doc/AGENTS.md），但这里仍做归一：一是双保险，
;; 二是让本 App 不依赖宿主某条路径的实现细节。
(define line_xy
  (lambda (x1 y1 x2 y2 r g b)
    {
      (define l1x x1)
      (define l1y y1)
      (define l2x x2)
      (define l2y y2)
      (if (and (== x1 x2) (> y1 y2))
          {
            (set! l1y y2)
            (set! l2y y1)
          }
          #f)
      (if (and (== y1 y2) (> x1 x2))
          {
            (set! l1x x2)
            (set! l2x x1)
          }
          #f)
      (Gfx.draw_line l1x l1y l2x l2y r g b)
    }))

(define draw_stars
  (lambda ()
    {
      (define i 0)
      (define c 0)
      (while (< i NSTAR) {
        (set! c (get_item stc i))
        (if (== c 0)
            (Gfx.fill_rect (Math.floor (get_item stx i)) (Math.floor (get_item sty i)) 1 1 90 100 130)
            (if (== c 1)
                (Gfx.fill_rect (Math.floor (get_item stx i)) (Math.floor (get_item sty i)) 1 1 150 165 195)
                (Gfx.fill_rect (Math.floor (get_item stx i)) (Math.floor (get_item sty i)) 2 1 225 232 245)))
        (set! i (+ i 1))
      })
    }))

;; 一颗陨石：旋转 → 扫描线填充 → 描边 → 陨坑
(define draw_ast
  (lambda (j)
    {
      (define cx 0.0)
      (define cy 0.0)
      (define rr 0.0)
      (define sh 0)
      (define ca 0.0)
      (define sa 0.0)
      (define k 0)
      (define k2 0)
      (define row 0.0)
      (define ymin 0.0)
      (define ymax 0.0)
      (define nc 0)
      (define si 0)
      (define sj 0)
      (define sv 0.0)
      (define y0 0.0)
      (define y1 0.0)
      (define xa 0.0)
      (define xb 0.0)
      (define ux 0.0)
      (define uy 0.0)
      (define cr 0)
      (define cg 0)
      (define cb 0)
      (define qr 0)
      (define qg 0)
      (define qb 0)
      (set! cx (get_item axx j))
      (set! cy (get_item ayy j))
      (set! rr (get_item arr j))
      (set! sh (get_item ashp j))
      (if (== sh 0) { (set! cr 118) (set! cg 110) (set! cb 98) } #f)
      (if (== sh 1) { (set! cr 104) (set! cg 100) (set! cb 114) } #f)
      (if (== sh 2) { (set! cr 124) (set! cg 104) (set! cb 92) } #f)
      ;; 顶点：单位坐标绕 (0,0) 旋转 → 乘半径 → 平移。每颗陨石只算一次 (ca,sa)。
      (set! ca (Math.cos (get_item aang j)))
      (set! sa (Math.sin (get_item aang j)))
      (set! k 0)
      (while (< k 8) {
        (set! ux (get_item UX (+ (* sh 8) k)))
        (set! uy (get_item UY (+ (* sh 8) k)))
        (set_item! vx8 k (+ cx (* rr (- (* ux ca) (* uy sa)))))
        (set_item! vy8 k (+ cy (* rr (+ (* ux sa) (* uy ca)))))
        (set! k (+ k 1))
      })
      ;; 扫描线填充：每 2 行求一次水平跨度（星形多边形 → 取最小/最大交点即内区间）
      (set! ymin (get_item vy8 0))
      (set! ymax (get_item vy8 0))
      (set! k 1)
      (while (< k 8) {
        (if (< (get_item vy8 k) ymin) (set! ymin (get_item vy8 k)) #f)
        (if (> (get_item vy8 k) ymax) (set! ymax (get_item vy8 k)) #f)
        (set! k (+ k 1))
      })
      (set! row (Math.floor ymin))
      (if (< row 0.0) (set! row 0.0) #f)
      (while (<= row ymax) {
        ;; 顶光：行在中心上方最亮、两侧居中、下方最暗（每行一次比较，几乎免费）
        (if (< row (- cy (* 0.35 rr))) {
          (set! qr (+ cr 16)) (set! qg (+ cg 16)) (set! qb (+ cb 14))
        } #f)
        (if (and (>= row (- cy (* 0.35 rr))) (< row (+ cy (* 0.45 rr)))) {
          (set! qr cr) (set! qg cg) (set! qb cb)
        } #f)
        (if (>= row (+ cy (* 0.45 rr))) {
          (set! qr (- cr 20)) (set! qg (- cg 20)) (set! qb (- cb 18))
        } #f)
        ;; ① 收集本扫描线与 8 条边的交点（半开区间 [y0,y1)，顶点不会被算两次）
        (set! nc 0)
        (set! k 0)
        (while (< k 8) {
          (set! k2 (if (== k 7) 0 (+ k 1)))
          (set! y0 (get_item vy8 k))
          (set! y1 (get_item vy8 k2))
          (if (or (and (<= y0 row) (> y1 row)) (and (<= y1 row) (> y0 row)))
              {
                (set! xa (get_item vx8 k))
                (set! xb (+ xa (/ (* (- row y0) (- (get_item vx8 k2) xa)) (- y1 y0))))
                (set_item! XC nc xb)
                (set! nc (+ nc 1))
              }
              #f)
          (set! k (+ k 1))
        })
        ;; ② 交点排序（插入排序；一个扫描线只有 2~4 个交点）
        (set! si 1)
        (while (< si nc) {
          (set! sv (get_item XC si))
          (set! sj si)
          (while (and (> sj 0) (> (get_item XC (- sj 1)) sv)) {
            (set_item! XC sj (get_item XC (- sj 1)))
            (set! sj (- sj 1))
          })
          (set_item! XC sj sv)
          (set! si (+ si 1))
        })
        ;; ③ 奇偶规则成对填充（凹多边形也正确，不会被"最大跨度"填掉缺口）
        (set! si 0)
        (while (< (+ si 1) nc) {
          (set! xa (get_item XC si))
          (set! xb (get_item XC (+ si 1)))
          (if (> xb xa)
              {
                (if (< xa 0.0) (set! xa 0.0) #f)
                (if (> xb (- SW 1.0)) (set! xb (- SW 1.0)) #f)
                (if (< row (- SH 1.0))
                    (Gfx.fill_rect (Math.floor xa) row (- (Math.floor xb) (Math.floor xa)) 1 qr qg qb)
                    #f)
              }
              #f)
          (set! si (+ si 2))
        })
        (set! row (+ row 1.0))
      })
      ;; 描边：8 条边（只画完全落在屏内的，避免 draw_line 的 uint32 负坐标坑）
      (set! k 0)
      (while (< k 8) {
        (set! k2 (if (== k 7) 0 (+ k 1)))
        (if (and (and (>= (get_item vx8 k) 0.0) (>= (get_item vy8 k) 0.0))
                 (and (and (<= (get_item vx8 k) (- SW 1.0)) (<= (get_item vy8 k) (- SH 1.0)))
                      (and (and (>= (get_item vx8 k2) 0.0) (>= (get_item vy8 k2) 0.0))
                           (and (<= (get_item vx8 k2) (- SW 1.0)) (<= (get_item vy8 k2) (- SH 1.0))))))
            (line_xy (Math.floor (get_item vx8 k)) (Math.floor (get_item vy8 k))
                     (Math.floor (get_item vx8 k2)) (Math.floor (get_item vy8 k2))
                     (+ cr 34) (+ cg 32) (+ cb 30))
            #f)
        (set! k (+ k 1))
      })
      ;; 陨坑（两个随多边形一起转的暗点）
      (if (and (> cy 0.0) (< cy SH))
          {
            (set! ux (- (* 0.28 ca) (* -0.18 sa)))
            (set! uy (+ (* 0.28 sa) (* -0.18 ca)))
            (Gfx.fill_circle (Math.floor (+ cx (* rr ux))) (Math.floor (+ cy (* rr uy)))
                             (Math.floor (* rr 0.26)) (- cr 34) (- cg 32) (- cb 30))
            (set! ux (- (* -0.22 ca) (* 0.24 sa)))
            (set! uy (+ (* -0.22 sa) (* 0.24 ca)))
            (Gfx.fill_circle (Math.floor (+ cx (* rr ux))) (Math.floor (+ cy (* rr uy)))
                             (Math.floor (* rr 0.18)) (- cr 30) (- cg 28) (- cb 26))
          }
          #f)
    }))

(define draw_asts
  (lambda ()
    {
      (define i 0)
      (while (< i nast) {
        (draw_ast i)
        (set! i (+ i 1))
      })
    }))

(define draw_bullets
  (lambda ()
    {
      (define i 0)
      (define x 0.0)
      (define y 0.0)
      (while (< i nbul) {
        (set! x (get_item bxx i))
        (set! y (get_item byy i))
        (Gfx.fill_rect (- x 1.0) y 3 7 255 168 40)
        (Gfx.fill_rect x (+ y 1.0) 1 5 255 250 215)
        (set! i (+ i 1))
      })
    }))

(define draw_parts
  (lambda ()
    {
      (define i 0)
      (define t 0.0)
      (define fr 0)
      (while (< i npart) {
        (set! t (/ (get_item plf i) (get_item pmx i)))
        (if (== (get_item pkd i) 1)
            {
              (set! fr (Math.floor (* (get_item pvx i) t)))
              (if (> fr 0)
                  {
                    (Gfx.fill_circle (Math.floor (get_item pxx i)) (Math.floor (get_item pyy i)) fr 255 168 66)
                    (if (> fr 2) (Gfx.fill_circle (Math.floor (get_item pxx i)) (Math.floor (get_item pyy i)) (- fr 2) 255 220 130) #f)
                    (if (> fr 4) (Gfx.fill_circle (Math.floor (get_item pxx i)) (Math.floor (get_item pyy i)) (- fr 4) 255 250 224) #f)
                  }
                  #f)
            }
            (if (> t 0.66)
            (Gfx.fill_rect (- (get_item pxx i) 1.0) (- (get_item pyy i) 1.0) 2 2 255 240 170)
                (if (> t 0.33)
                    (Gfx.fill_rect (- (get_item pxx i) 1.0) (- (get_item pyy i) 1.0) 2 2 255 150 55)
                    (Gfx.fill_rect (get_item pxx i) (get_item pyy i) 1 1 168 70 44))))
        (set! i (+ i 1))
      })
    }))

;; 飞船：逐行渐变三角机身 + 座舱 + 尾焰
(define draw_ship
  (lambda ()
    {
      (define i 0)
      (define hw 0)
      (define rr 0)
      (define gg 0)
      (define bb 0)
      (define fh 0)
      (define sy0 (- ship_y 12.0))
      (set! i 0)
      (while (< i 15) {
        (set! hw (get_item HW i))
        (set! rr (- 150 (* i 7)))
        (set! gg (- 240 (* i 9)))
        (set! bb (- 255 (* i 5)))
        (Gfx.fill_rect (- ship_x hw) (+ sy0 i) (+ (* hw 2) 1) 1 rr gg bb)
        (set! i (+ i 1))
      })
      ;; 机翼高光
      (line_xy (Math.floor (- ship_x 12.0)) (Math.floor (+ ship_y 1.0))
               (Math.floor ship_x) (Math.floor (- ship_y 6.0)) 60 130 210)
      (line_xy (Math.floor (+ ship_x 12.0)) (Math.floor (+ ship_y 1.0))
               (Math.floor ship_x) (Math.floor (- ship_y 6.0)) 60 130 210)
      ;; 座舱
      (Gfx.fill_circle (Math.floor ship_x) (Math.floor (- ship_y 4.0)) 2 245 252 255)
      ;; 尾焰（长度每帧抖动 = 闪烁）
      (set! fh (+ 4 (Math.floor (* 5.0 (rnd)))))
      (Gfx.fill_rect (- ship_x 3.0) (+ ship_y 3.0) 6 fh 255 140 40)
      (Gfx.fill_rect (- ship_x 1.0) (+ ship_y 3.0) 2 (+ fh 3) 255 226 130)
    }))

;; 敌我识别用的小飞船图标（生命数）
(define draw_life
  (lambda (x y)
    {
      (Gfx.fill_rect (+ x 3) y 3 3 140 235 255)
      (Gfx.fill_rect (+ x 1) (+ y 3) 7 3 70 170 230)
      (Gfx.fill_rect x (+ y 6) 9 3 40 105 195)
    }))

(define draw_hud
  (lambda ()
    {
      (Gfx.fill_rect 0 0 SW HUD_H 14 16 30)
      (Gfx.fill_rect 0 (- HUD_H 1) SW 1 36 42 70)
      (Gfx.draw_text "波" 4 3 140 190 245)
      (draw_int wave 1 17 3 225 240 255)
      (Gfx.draw_text "分" 34 3 140 190 245)
      (draw_int score 5 47 3 240 244 250)
      ;; 生命图标（最多 3 枚；更多时只画到 3，另用数字补）
      (if (>= lives 1) (draw_life 206 5) #f)
      (if (>= lives 2) (draw_life 220 5) #f)
      (if (>= lives 3) (draw_life 234 5) #f)
      (if (> lives 3) (draw_int lives 1 250 3 190 220 245) #f)
    }))

;; 波次横幅（居中）
(define draw_banner
  (lambda ()
    {
      (define nd 0)
      (define x0 0.0)
      (set! nd (ndigits wave))
      (set! x0 (Math.floor (- (/ SW 2.0) (/ (+ 24 (* 7 nd)) 2.0))))
      (Gfx.draw_text "第" (Math.floor x0) 104 255 232 150)
      (draw_int wave 1 (Math.floor (+ x0 12.0)) 104 255 232 150)
      (Gfx.draw_text "波" (Math.floor (+ (+ x0 12.0) (* 7 nd))) 104 255 232 150)
    }))

;; 游戏结束浮层（点按屏幕重开）
(define draw_game_over
  (lambda ()
    {
      (Gfx.fill_rect 44 82 232 78 12 14 30)
      (Gfx.draw_rect 44 82 232 78 190 82 82)
      (Gfx.draw_text "游戏结束" 136 90 255 120 110)
      (Gfx.draw_text "得分" 112 112 150 200 245)
      (draw_int score 1 140 112 240 244 250)
      (Gfx.draw_text "最佳" 112 126 150 200 245)
      (draw_int best 1 140 126 210 220 240)
      (Gfx.draw_text "点按屏幕重开" 124 144 200 214 235)
    }))

(define render
  (lambda ()
    {
      (Gfx.fill_rect 0 HUD_H SW (- SH HUD_H) 6 8 20)
      (draw_stars)
      (draw_asts)
      (draw_bullets)
      (draw_parts)
      (if (== over 1)
          #f
          (if (> invuln 0.0)
              (if (== (mod (Math.floor (/ invuln 0.08)) 2) 1)
                  (draw_ship)
                  #f)
              (draw_ship)))
      (draw_hud)
      (if (and (> banner_t 0.0) (== over 0)) (draw_banner) #f)
      (if (and (> hint_t 0.0) (== over 0))
          (Gfx.draw_text "拖动飞船 自动开火" 112 226 96 118 156)
          #f)
      (if (> flash 0.0)
          {
            (Gfx.draw_rect 0 HUD_H SW (- SH HUD_H) 220 70 70)
            (Gfx.draw_rect 1 (+ HUD_H 1) (- SW 2) (- (- SH HUD_H) 2) 150 40 40)
          }
          #f)
      (if (== over 1) (draw_game_over) #f)
      (EX.draw_button)
      (Gfx.refresh)
    }))

;; ============================================================================
;; 输入：触屏拖动（电平采样，慢节奏 App 比边沿锁存稳）
;;   · 手指在哪儿，飞船就跟到哪儿——但**上抬 26px**，否则整艘船被指尖盖住；
;;   · 平滑跟随系数 0.42/拍（30fps 下约 4 拍到位）：既跟手又不抖；
;;   · 结局态下「松手再按」才重开（避免撞击那一下立刻重来）。
;; ============================================================================

(define input
  (lambda ()
    {
      (set! down (if (Touch.is_pressed) 1 0))
      (if (== down 1)
          {
            (set! tgo (Touch.x))
            (set! tgt (Touch.y))
            (set! tgx tgo)
            (set! tgy (- tgt 26.0))
            (if (< tgy 128.0) (set! tgy 128.0) #f)
            (if (> tgy 212.0) (set! tgy 212.0) #f)
            (set! ship_x (+ ship_x (* (- tgx ship_x) 0.42)))
            (set! ship_y (+ ship_y (* (- tgy ship_y) 0.42)))
          }
          #f)
      (if (< ship_x 12.0) (set! ship_x 12.0) #f)
      (if (> ship_x (- SW 12.0)) (set! ship_x (- SW 12.0)) #f)
      (if (< ship_y 128.0) (set! ship_y 128.0) #f)
      (if (> ship_y 212.0) (set! ship_y 212.0) #f)
    }))

;; ============================================================================
;; 主循环：一拍 = 采样输入 → 推进 → 整帧重绘
;; ============================================================================

(define tick
  (lambda ()
    {
      (EX.check)
      (set! tnow (System.timestamp))
      (set! dms (- tnow tlast))
      (set! tlast tnow)
      (if (< dms 0) (set! dms 0) #f)
      (if (> dms DT_MAX) (set! dms DT_MAX) #f)
      (set! dts (/ dms 1000.0))
      (set! gt (+ gt dms))

      (input)

      (if (== over 1)
          {
            ;; 结局：松手后的一次按下 = 重开
            (if (and (== down 1) (== prev_down 0))
                (new_game)
                #f)
          }
          {
            (upd_stars)
            (upd_parts)
            (upd_bullets)
            (upd_asts)
            (if (== playing 1)
                {
                  (upd_fire)
                  (upd_waves)
                  (collide_bullets)
                }
                #f)
            (collide_ship)
            ;; 计时器（秒）
            (if (> invuln 0.0) (set! invuln (- invuln dts)) #f)
            (if (> flash 0.0) (set! flash (- flash dts)) #f)
            (if (> banner_t 0.0) (set! banner_t (- banner_t dts)) #f)
            (if (> hint_t 0.0) (set! hint_t (- hint_t dts)) #f)
          })

      (set! prev_down down)
      (if (> score best) (set! best score) #f)
      (render)
    }))

;; ============================================================================
;; 启动
;; ============================================================================

(reseed)
(new_game)
(set! tlast (System.timestamp))
(tick)
(set! _timer (System.set_interval TICK_MS tick))

;; 退出：硬 A/ESC / 右上角按钮 → 宿主全局确认框；「留下」后重绘以擦掉对话框残影
(EX.install_with render)