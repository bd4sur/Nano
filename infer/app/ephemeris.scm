;; ============================================================================
;; 星历（Ephemeris）—— 太阳/月亮地平坐标 · 月相 · 月盘倾斜角 —— 灵机应用程序
;;
;; 本 App 是**薄壳**：全部天文计算来自共享模块 app/lib/ephem.scm（纯计算库），
;; 这里只负责「读系统时间 → 驱动库 → 用 Gfx 绘制文字」。
;; 同一份库也被农历 App（nongli.scm）复用，详见库文件头部的接口说明。
;;
;; 输入：当前系统时间（本地库 Time：UTC 整数日 + 当日秒）+ 项目默认观测点
;;       南京 119.0°E / 32.0°N / UTC+8（取自 ui_app.c 中 global_state 的初值）。
;; 输出：太阳高度角/方位角、月亮高度角/方位角、月相（相位角、照亮比、月相名）、
;;       月亮圆盘倾斜角（亮肢方位角、倾斜角、天球北向角）、VSOP87 校对黄经/日地距离。
;;
;; 命名约定：标识符一律下划线分隔（不使用短划线）。
;; ============================================================================

(native Gfx)
(native Math)
(native String)
(native System)
(native Time)

(import Ephem "app/lib/ephem.scm")
(import EX "app/lib/exit.scm")   ; 硬 A/ESC → 全局退出确认

;; ---- 显示量 ---------------------------------------------------------------

(define v_y 1970)
(define v_m 1)
(define v_d 1)
(define v_h 0)
(define v_mi 0)
(define v_s 0)
(define v_sun_az 0.0)
(define v_sun_alt 0.0)
(define v_moon_az 0.0)
(define v_moon_alt 0.0)
(define v_k 0.0)                    ; 月面照亮比例
(define v_elong 0.0)                ; 月日黄经差（度）

;; ---- 格式化 ---------------------------------------------------------------

(define absv (lambda (v) (if (< v 0.0) (- 0.0 v) v)))

(define padz
  (lambda (s n) {
    (define r s)
    (while (< (String.length r) n) { (set! r (String.concat "0" r)) })
    r
  }))

(define pad2
  (lambda (n) {
    (define v (Math.floor n))
    (if (< v 10.0)
        (String.concat "0" (String.atom_to_string v))
        (String.atom_to_string v))
  }))

;; 定点格式化（不依赖 Math.to_fixed：32 位宿主 am_float_t 为单精度，
;; to_fixed 的乘积会留下 9.92999 这类尾差）
(define fixed
  (lambda (v dgt) {
    (define sc (Math.pow 10.0 dgt))
    (define neg (if (< v 0.0) 1 0))
    (define xs (Math.floor (+ (* (absv v) sc) 0.5)))
    (define ip (Math.floor (/ xs sc)))
    (define fp (- xs (* ip sc)))
    (String.concat (if (== neg 1) "-" "")
      (String.concat (String.atom_to_string ip)
        (String.concat "." (padz (String.atom_to_string fp) dgt))))
  }))

(define deg_signed
  (lambda (v dgt) (String.concat (if (>= v 0.0) "+" "-")
                                 (String.concat (fixed (absv v) dgt) "°"))))
(define deg_plain
  (lambda (v dgt) (String.concat (fixed (Ephem.norm360 v) dgt) "°")))

(define phase_name
  (lambda (e) (cond ((< e 22.5) "新月（朔）")
                    ((< e 67.5) "娥眉月")
                    ((< e 112.5) "上弦月")
                    ((< e 157.5) "盈凸月")
                    ((< e 202.5) "满月（望）")
                    ((< e 247.5) "亏凸月")
                    ((< e 292.5) "下弦月")
                    ((< e 337.5) "残月")
                    (else "新月（朔）"))))

;; ---- 每秒：读时 → 驱动星历库 → 绘制 ---------------------------------------

(define update
  (lambda () {
    (EX.check)
    (define udays (Time.utc_days))
    (define usod (Time.utc_sod))
    ;; 库的时间基（UT）
    (Ephem.set_time udays usod)
    (Ephem.set_gmst)
    ;; 本地民用时（UTC+8）用于显示
    (define ldays udays)
    (define lsod (+ usod Ephem.TZ_SEC))
    (if (>= lsod 86400) { (set! lsod (- lsod 86400)) (set! ldays (+ ldays 1)) })
    (Ephem.civil ldays)
    (set! v_y (Ephem.get_civ_y))
    (set! v_m (Ephem.get_civ_m))
    (set! v_d (Ephem.get_civ_d))
    (set! v_h (Math.floor (/ lsod 3600.0)))
    (set! v_mi (Math.floor (/ (- lsod (* v_h 3600.0)) 60.0)))
    (set! v_s (- lsod (+ (* v_h 3600.0) (* v_mi 60.0))))
    ;; 日月位置与各种角度
    (Ephem.moon_equatorial)
    (Ephem.sun_equatorial)
    (Ephem.sun_altaz)
    (set! v_sun_az (Ephem.get_az))
    (set! v_sun_alt (Ephem.get_alt))
    (Ephem.moon_altaz)
    (set! v_moon_az (Ephem.get_az))
    (set! v_moon_alt (Ephem.get_alt))
    (Ephem.moon_phase)
    (Ephem.moon_bright_limb)
    (Ephem.moon_tilt)               ; 内部会改写 eq_az/eq_alt，故上面已先取值
    (Ephem.vsop_sun)
    (set! v_k (* 0.5 (+ 1.0 (Math.cos (Ephem.to_rad (Ephem.get_phase))))))
    (set! v_elong (Ephem.norm360 (- (Ephem.get_moon_lon) (Ephem.get_sun_lon))))
    ;; ---- 绘制 ----
    (Gfx.clear)
    (Gfx.draw_text "星历 · 南京 119.0°E 32.0°N" 6 4 255 255 255)
    (Gfx.draw_text (String.concat
                      (String.concat (String.atom_to_string v_y) "-")
                      (String.concat (pad2 v_m)
                        (String.concat "-"
                          (String.concat (pad2 v_d)
                            (String.concat " "
                              (String.concat (pad2 v_h)
                                (String.concat ":"
                                  (String.concat (pad2 v_mi)
                                    (String.concat ":" (String.concat (pad2 v_s) " UTC+8"))))))))))
                    6 18 130 200 255)
    (Gfx.draw_text "太阳" 6 38 255 205 70)
    (Gfx.draw_text (String.concat "  高度 " (deg_signed v_sun_alt 2)) 6 52 255 232 140)
    (Gfx.draw_text (String.concat "  方位 " (deg_plain v_sun_az 2)) 6 66 255 232 140)
    (Gfx.draw_text "月亮" 6 86 130 200 255)
    (Gfx.draw_text (String.concat "  高度 " (deg_signed v_moon_alt 2)) 6 100 180 220 255)
    (Gfx.draw_text (String.concat "  方位 " (deg_plain v_moon_az 2)) 6 114 180 220 255)
    (Gfx.draw_text "月相" 6 134 255 255 255)
    (Gfx.draw_text (String.concat "  相位角 " (String.concat (fixed (Ephem.get_phase) 1)
                     (String.concat "°  照亮 " (String.concat (fixed (* 100.0 v_k) 1) "%"))))
                   6 148 255 255 255)
    (Gfx.draw_text (String.concat "  " (phase_name v_elong)) 6 162 235 235 160)
    (Gfx.draw_text "月亮圆盘" 6 182 255 255 255)
    (Gfx.draw_text (String.concat "  亮肢方位角 " (deg_plain (Ephem.get_bright_pa) 1)) 6 196 200 220 255)
    (Gfx.draw_text (String.concat "  倾斜角 " (String.concat (fixed (Ephem.get_disk_tilt) 1)
                     (String.concat "°  北向 " (String.concat (fixed (Ephem.get_tilt_north) 1) "°"))))
                   6 210 200 220 255)
    (Gfx.draw_text (String.concat "VSOP87 黄经 " (String.concat (fixed (Ephem.get_vsop_lon) 3)
                     (String.concat "° " (String.concat (fixed (Ephem.get_vsop_r) 4) " AU"))))
                   6 226 150 160 170)
      (EX.draw_button)
      (Gfx.refresh)
  }))

;; ---- 启动：立即算一次并显示，此后每拍判"该不该重算" -----------------------
;;
;; 轮询周期取 100ms 而不是 1000ms：退出按钮/硬 A 的响应跟着**拍子**走（(EX.check) 在每拍
;; 里跑），1 秒一拍的 App 点按钮会明显发木；而重算（星历级数 + 整屏重绘）只在"秒数变了"
;; 时做一次，CPU 开销与原来相同（实测刷新频率仍是 1 Hz）。

(define last_sec -1)

(define tick
  (lambda () {
    (EX.check)
    (define s (Time.utc_sod))
    (if (not (== s last_sec))
        { (set! last_sec s) (update) }
        #f)
  }))

(update)
(set! update_timer (System.set_interval 100 tick))

;; ---- 退出：硬 A/ESC → 宿主全局退出确认框（见 app/lib/exit.scm）----
;;  确认 → 宿主本帧末销毁本 App；留下 → 继续运行（并调 update 擦掉对话框残影）
(EX.install_with update)
