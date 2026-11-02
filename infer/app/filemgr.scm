;; ============================================================================
;; filemgr.scm —— 文件管理器（灵机应用程序 · 完全权限版）
;;
;; 能力（全部经 (native Fs) 真文件链路，桌面 / 浏览器 MEMFS / 掌机 SD 行为一致）：
;;   · 列表查看：当前目录下的**文件与目录**——**目录永远排在文件前面，同类内按名字升序**
;;     （排序在 Scheme 层做，不依赖各平台 readdir 顺序）；目录以「名字/」+ 蓝色区分；
;;   · 列表顶部固定一条 **".."（返回上级目录）** 条目（根目录下变灰，点了只提示）；
;;   · 进入目录 / 返回上级；文件选中后「进入」= 显示详情（路径 / 大小 / 类型）；
;;   · 新建文件：默认 **.txt**，默认文件名为**时间戳**（如 20261020-020000.txt）；
;;   · 新建目录：默认名同样是时间戳；
;;   · 重命名 / 移动：改名时可带 "/" 指定其它目录（同一资源根内移动）；
;;   · 删除：先弹确认框；目录会**递归清空**（先删子树再删自身，因为 Fs.remove
;;     只删空目录——这是 (native Fs) 的既定能力边界）；
;;   · 刷新：外部（其它 App）改了文件时手动重扫。
;;
;; 「完全权限」的范围：业务资源根（PLATFORM_ROOT_DIR）以下的**全部**文件与目录
;; 都可读写/改名/删除——包括 icon/、app/、doc/、data/ 这些系统自己在用的目录。
;; 边界就是资源根本身：根目录不被删除/改名（am_native_Fs.c 的 fs_is_root_path
;; 在原生层再兜一道），也没有 ".." 越界（(native Fs) 的路径规则）。
;;
;; 为复制 / 移动留出的扩展空间（明确的两处，实现时不需要动别的代码）：
;;   ① **操作表**：下面的 build_ops 是唯一的"按钮清单"——每项 =（名称 处理函数
;;      是否可用）。bar 的绘制、命中测试、键盘映射全部按表驱动，因此新增一项
;;      操作 = 表里加一行 + 写一个处理函数；「复制」「移动」两枚按钮已经占位
;;      （enabled=0 时画成灰色、点按提示"预留"），落地时把标志改成 1 并实现
;;      op_copy / op_move 即可。
;;   ② **数据通路**：跨目录复制需要"读字节 → 写字节"或原生层的 copy 原语。
;;      当前 (native Fs) 只有文本整读整写（128KB 上限）：小文件可直接
;;      (Fs.read)+(Fs.write) 组合，大文件/目录树建议在 am_native_Fs.c 里加一个
;;      Fs.copy（hal_fs 已有 platform_read_file_to_buffer/write 可作为基础）。
;;      移动已经可用：它在同一文件系统内就是 (Fs.rename ...)，见 op_rename。
;;
;; 交互：
;;   · 点按列表行 = 选中；再点已选中的行 = 进入/详情/返回上级（小屏没有双击，靠"再点一次"）；
;;   · 底栏两排按钮：进入 / 新建文件 / 新建目录 / 重命名 / 删除（第一排），
;;     上级 / 复制(预留) / 移动(预留) / 刷新（第二排）；
;;   · 实体键盘：↑↓ 选择，Enter 进入，退格(8) 上级；
;;   · 硬 A/ESC 与右上角「退出」按钮 —— 由 app/lib/exit.scm 统一接管。
;;
;; 命名/文案：新建与重命名走宿主的模态输入框（(Ui.input_with)），因此中文文件名
;; 可以直接用拼音输入法打出来；删除用宿主的两按钮确认框（(Ui.confirm)）。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/list/map/append；
;; '() 是模块级静态槽（空表一律 (cdr '(0)) 现造）；push 返回 #undefined 不可赋值；
;; 循环里不用下标末值当 break，一律用 stop 标志；块内不 define。
;; ============================================================================

(native Gfx)
(native Key)
(native Touch)
(native Fs)
(native String)
(native Math)
(native System)
(native Time)
(native Ui)
(native App)
(import EX "app/lib/exit.scm")
(import FSX "app/lib/fsx.scm")

;; ---- 布局 ----------------------------------------------------------------

(define SW 320)
(define SH 240)
(define HEAD_H 18)      ;; 页眉（标题 + 右上角退出按钮）
(define PATH_Y 20)      ;; 当前路径行（12px）
(define LIST_Y 34)      ;; 列表首行 y
(define ROW_H 18)       ;; 行高
(define LIST_ROWS 8)    ;; 一屏 8 行（34 + 8*18 = 178）
(define INFO_Y 182)     ;; 选中项信息 / 操作反馈，一行 12px
(define BAR1_Y 198)     ;; 第一排按钮
(define BAR2_Y 218)     ;; 第二排按钮
(define BTN_H 16)
(define BTN_W 60)
(define BTN_GAP 4)
(define BOT_H 8)        ;; 底边留白（EX 的退出按钮在**顶部**，这里不冲突）

;; ---- 状态 ----------------------------------------------------------------

(define cur ".")             ;; 当前目录（相对业务资源根；"." = 根）
(define rows (cdr '(0)))     ;; 条目表：(mk3 名字 是否目录 字节数)，已按名字升序
                             ;; （尺寸也在载入时缓存：绘制不再逐帧 stat）
(define sel 0)               ;; 选中下标（0..n-1）
(define first 0)             ;; 列表窗口首项
(define err 0)               ;; 目录读不出来（Fs.list → #null）
(define flash_text "")       ;; 操作反馈（几拍后自动消失）
(define flash_ttl 0)
(define flash_r 190) (define flash_g 200) (define flash_b 150)

;; 触屏
(define prev_down 0)
(define drag_last 0)
(define drag_total 0)
(define press_x 0)
(define press_y 0)

(define ts 0)                ;; 定时器编号
(define stopped 0)

;; ---- 小工具 --------------------------------------------------------------

(define new_list (lambda () (cdr '(0))))
(define mk2 (lambda (a b) (cons a (cons b (cdr '(0))))))
(define mk3 (lambda (a b c) (cons a (cons b (cons c (cdr '(0)))))))

(define n_rows (lambda () (length rows)))

;; 列表的"显示行" = 1 行固定在最顶上的 ".."（返回上级目录）+ 各条目。
;; 于是下标约定是：显示行 0 = ".."，显示行 d（d>=1）= rows[d-1]。
;; （用户要的就是"列表顶部有一个 .. 条目"；「上级」按钮也还在，两条路都通）
(define n_disp (lambda () (+ (n_rows) 1)))
(define at_root (lambda () (String.equals cur ".")))

(define dir_label
  (lambda ()
    (if (at_root) "/" (String.concat "/" cur))))

;; 当前选中的条目（#f = 选中的是 ".." 行，或列表里没有条目）
(define sel_row
  (lambda ()
    (if (and (> sel 0) (<= sel (n_rows)))
        (get_item rows (- sel 1))
        #f)))

;; 反馈文案（下一拍起显示 FLASH_TICKS 拍）
(define FLASH_TICKS 25)
(define say
  (lambda (msg)
    { (set! flash_r 190) (set! flash_g 200) (set! flash_b 150)
      (set! flash_text msg)
      (set! flash_ttl FLASH_TICKS)
      (display "FM ") (display msg) (newline) }))

(define say_warn
  (lambda (msg)
    { (set! flash_r 240) (set! flash_g 150) (set! flash_b 120)
      (set! flash_text msg)
      (set! flash_ttl FLASH_TICKS)
      (display "FMW ") (display msg) (newline) }))

;; ---- 时间戳（默认文件名）------------------------------------------------
;; 为什么用 Time.utc_days/utc_sod 而不是 System.timestamp：32 位宿主（wasm32）上
;; 毫秒时间戳会被截断成「mod 2^27」（见 am_native_Time.h），无法还原日期；Time 库
;; 把时间拆成"日数 + 日内秒数"两个小整数，float32 下也精确。
;; 结果是 **UTC** 时间（宿主不提供时区信息）——作为文件名的唯一性标识足够。

(define pad2
  (lambda (v)
    (if (< v 10)
        (String.concat "0" (String.atom_to_string v))
        (String.atom_to_string v))))

(define stamp
  (lambda ()
    (define z 0) (define era 0) (define doe 0) (define t1 0) (define t2 0)
    (define yoe 0) (define y 0) (define doy 0) (define mp 0)
    (define dd 0) (define mo 0) (define sod 0)
    (define hh 0) (define mi 0) (define ss 0)
    ;; 民用历反算（Howard Hinnant 的 days → y/m/d 算法，整数运算）
    (set! z (+ (Time.utc_days) 719468))
    (set! era (Math.floor (/ z 146097)))
    (set! doe (- z (* era 146097)))
    (set! t1 (- doe (Math.floor (/ doe 1460))))
    (set! t2 (+ t1 (Math.floor (/ doe 36524))))
    (set! t1 (- t2 (Math.floor (/ doe 146096))))
    (set! yoe (Math.floor (/ t1 365)))
    (set! y (+ yoe (* era 400)))
    (set! t1 (+ (* 365 yoe) (Math.floor (/ yoe 4))))
    (set! t2 (- t1 (Math.floor (/ yoe 100))))
    (set! doy (- doe t2))
    (set! mp (Math.floor (/ (+ (* 5 doy) 2) 153)))
    (set! dd (+ (- doy (Math.floor (/ (+ (* 153 mp) 2) 5))) 1))
    (set! mo (if (< mp 10) (+ mp 3) (- mp 9)))
    (if (<= mo 2) (set! y (+ y 1)) #f)
    ;; 日内时间
    (set! sod (Time.utc_sod))
    (set! hh (Math.floor (/ sod 3600)))
    (set! mi (Math.floor (/ (mod sod 3600) 60)))
    (set! ss (mod sod 60))
    (String.concat (String.atom_to_string y)
      (String.concat (pad2 mo)
        (String.concat (pad2 dd)
          (String.concat "-"
            (String.concat (pad2 hh)
              (String.concat (pad2 mi) (pad2 ss)))))))))

;; ---- 载入当前目录 --------------------------------------------------------

(define clamp_sel
  (lambda ()
    (if (>= sel (n_disp)) (set! sel (- (n_disp) 1)) #f)
    (if (< sel 0) (set! sel 0) #f)
    (if (< sel first) (set! first sel) #f)
    (if (>= sel (+ first LIST_ROWS))
        (set! first (- (+ sel 1) LIST_ROWS))
        #f)
    (if (< first 0) (set! first 0) #f)
    #t))

(define load_dir
  (lambda ()
    (define raw (Fs.list cur))
    (define n 0)
    (define i 0)
    (define s "")
    (define isd 0)
    (define sz #f)
    (define keep (new_list))
    (set! rows (new_list))
    ;; 注意用 (list? raw) 而不是 (null?) —— null? 对 #null **和空表** 都为真，
    ;; 空目录（'()）会被误判成"读不出来"。（#null = 目录打不开；'() = 空目录）
    (if (list? raw)
        { (set! err 0)
          (set! n (length raw))
          (set! i 0)
          (while (< i n)
            { (set! s (get_item raw i))
              ;; 是否目录与文件大小在**载入时**算一次并缓存：ESP32 的 SD stat 有
              ;; 毫秒级开销，60fps 逐帧重算承受不起（绘制只读缓存）
              (set! isd (if (Fs.is_dir (FSX.join cur s)) 1 0))
              (if (== isd 1)
                  (push keep (mk3 s 1 0))
                  { (set! sz (Fs.size (FSX.join cur s)))
                    (push keep (mk3 s 0 (if (null? sz) 0 sz))) })
              (set! i (+ i 1)) })
          (set! rows (FSX.sort_entries keep)) })
    (clamp_sel)
    ;; 诊断行：目录名 + 条目数 + **排序后的条目表**（目录带尾 "/"，首项固定是 ".."）。
    ;; 端到端测试据此断言"目录在前 + 名字升序"，也方便排查"文件没出现/顺序不对"。
    (display "FM dir=") (display cur)
    (display " n=") (display (n_rows)) (newline)
    (display "FM items ../ ")
    (set! i 0)
    (while (< i (n_rows))
      { (display (String.concat (get_item (get_item rows i) 0)
                    (if (== (get_item (get_item rows i) 1) 1) "/" "")))
        (display " ")
        (set! i (+ i 1)) })
    (newline)
    #t))

(define goto
  (lambda (d)
    { (set! cur d)
      (set! sel 1)     ;; 选中第一条真实条目（0 是 ".."；别让"顺手回车"退回上级）
      (set! first 0)
      (load_dir) }))

;; 载入后把选中项定位到某个名字（新建/改名后用）
(define select_name
  (lambda (nm)
    (define n (n_rows))
    (define i 0)
    (define r -1)
    (while (and (< i n) (< r 0))
      { (if (String.equals (get_item (get_item rows i) 0) nm) (set! r i) #f)
        (set! i (+ i 1)) })
    (if (>= r 0) (set! sel (+ r 1)) #f)
    (clamp_sel)
    #t))

;; 递归清空并删除目录树（文件返回 1/0；目录返回 1/0）
(define remove_tree
  (lambda (path)
    (define nm (Fs.list path))
    (define n 0)
    (define i 0)
    (define child "")
    (define ok 1)
    ;; 同上：空目录是 '()（可删），读不出来是 #null（也直接试删，删不掉就是失败）
    (if (list? nm)
        { (set! n (length nm))
          (set! i 0)
          (while (< i n)
            { (set! child (FSX.join path (get_item nm i)))
              (if (Fs.is_dir child)
                  (if (== (remove_tree child) 0) (set! ok 0) #f)
                  (if (Fs.remove child) #f (set! ok 0)))
              (set! i (+ i 1)) })
          (if (== ok 1) (if (Fs.remove path) 1 0) 0) }
        (if (Fs.remove path) 1 0))))

;; ---- 操作实现 ------------------------------------------------------------

(define show_info
  (lambda (name)
    (define path (FSX.join cur name))
    (define isd (if (Fs.is_dir path) 1 0))
    (define sz (Fs.size path))
    (define kind "目录")
    (define body "")
    (if (== isd 1)
        #f
        (set! kind (if (== (String.length (FSX.ext_of name)) 0) "文件" (FSX.ext_of name))))
    (set! body (String.concat "路径：" (String.concat path "\n")))
    (if (== isd 0)
        (set! body (String.concat body
                    (String.concat "大小："
                      (String.concat (if (null? sz) "未知" (FSX.human_size sz)) "\n"))))
        #f)
    (Ui.message_titled name
                       (String.concat body (String.concat "类型：" kind)))))

(define op_enter
  (lambda ()
    (define row (sel_row))
    (if row
        (if (== (get_item row 1) 1)
            (goto (FSX.join cur (get_item row 0)))
            (show_info (get_item row 0)))
        (op_up))))     ;; 选中 ".." 行（或空列表）= 返回上级

(define op_new_file
  (lambda ()
    (define def (String.concat (stamp) ".txt"))
    (define nm #f)
    (define name "")
    (define path "")
    (display "FMDEF file=") (display def) (newline)
    (set! nm (Ui.input_with "新建文件（默认 .txt）" def))
    ;; ① 整理名字：取消 → 空串；没写扩展名 → 补 ".txt"
    (if (null? nm) (set! name "") (set! name nm))
    (if (and (> (String.length name) 0) (== (String.length (FSX.ext_of name)) 0))
        (set! name (String.concat name ".txt"))
        #f)
    ;; ② 执行（平铺书写而不是层层嵌套 if：手工数嵌套括号太容易错，见 2026-11 教训）
    (if (== (String.length name) 0)
        (say "已取消（文件名为空）")
        { (set! path (FSX.join cur name))
          (if (Fs.exists path)
              (say_warn (String.concat "已存在：" name))
              { (if (Fs.write path "")
                    (say (String.concat "已新建文件 " name))
                    (say_warn "新建失败（不可写）"))
                (load_dir)
                (select_name name) }) })))

(define op_new_dir
  (lambda ()
    (define def (stamp))
    (define nm #f)
    (define name "")
    (define path "")
    (display "FMDEF dir=") (display def) (newline)
    (set! nm (Ui.input_with "新建目录" def))
    (if (null? nm) (set! name "") (set! name nm))
    (if (== (String.length name) 0)
        (say "已取消（目录名为空）")
        { (set! path (FSX.join cur name))
          (if (Fs.exists path)
              (say_warn (String.concat "已存在：" name))
              { (if (Fs.mkdir path)
                    (say (String.concat "已新建目录 " name))
                    (say_warn "新建目录失败（不可写）"))
                (load_dir)
                (select_name name) }) })))

(define op_rename
  (lambda ()
    (define row (sel_row))
    (define old "")
    (define nm "")
    (if row (set! old (get_item row 0)) #f)
    (if (== (String.length old) 0)
        (say "请先选择一个条目")
        { (set! nm (Ui.input_with "重命名 / 移动（可带 / 指定目录）" old))
          (if (null? nm) (set! nm "") #f)
          (if (or (== (String.length nm) 0) (String.equals nm old))
              (say "已取消（名字未变）")
              { (if (Fs.rename (FSX.join cur old) (FSX.join cur nm))
                    (say (String.concat "已重命名 "
                           (String.concat old (String.concat " → " nm))))
                    (say_warn "重命名失败（目标已存在或不可写）"))
                (load_dir)
                (select_name (FSX.leaf nm)) }) })))

(define op_delete
  (lambda ()
    (define row (sel_row))
    (define nm "")
    (define isd 0)
    (define path "")
    (define ok 0)
    (if row { (set! nm (get_item row 0)) (set! isd (get_item row 1)) } #f)
    (if (== (String.length nm) 0)
        (say "请先选择一个条目")
        { (set! path (FSX.join cur nm))
          (if (Ui.confirm "删除确认"
                          (if (== isd 1)
                              (String.concat "目录 " (String.concat nm
                                " 及其下全部内容将被删除，且不可恢复。"))
                              (String.concat "文件 " (String.concat nm " 将被删除，且不可恢复。"))))
              { (if (== isd 1)
                    (set! ok (remove_tree path))
                    (set! ok (if (Fs.remove path) 1 0)))
                (load_dir)
                (if (== ok 1)
                    (say (String.concat "已删除 " nm))
                    (say_warn "删除失败（不可写）")) }
              (say "已取消删除")) })))

(define op_up
  (lambda ()
    (if (String.equals cur ".")
        (say "已在资源根")
        { (goto (FSX.parent cur))
          (say (String.concat "上级 → " (dir_label))) })))

(define op_refresh
  (lambda ()
    { (load_dir)
      (say "已刷新") }))

;; ---- 复制 / 移动（预留位，见文件头的"扩展空间"说明）--------------------

(define op_copy
  (lambda ()
    (Ui.message "复制（预留）"
                (String.concat "复制尚未实现。落地方式：\n"
                  (String.concat "① build_ops 里把「复制」的可用标志置 1；\n"
                    (String.concat "② 实现本函数：小文件用 (Fs.read)+(Fs.write)，\n"
                      "   大文件/目录树在 am_native_Fs.c 加 Fs.copy 原语。"))))))

(define op_move
  (lambda ()
    (Ui.message "移动（预留）"
                (String.concat "跨目录移动可用「重命名」完成：\n"
                  "在重命名框里输入 目标目录/新名字。\n"
                  "同名移动/覆盖策略待定，故单列此按钮占位。"))))

;; ---- 操作表（唯一按钮清单：绘制 / 命中 / 键盘都按表驱动）---------------
;; 每项 = (mk3 名称 处理函数 是否可用)；前 5 项画在第一排，其余画在第二排。

(define ops (cdr '(0)))

(define build_ops
  (lambda ()
    (set! ops (new_list))
    (push ops (mk3 "进入" op_enter 1))
    (push ops (mk3 "新建文件" op_new_file 1))
    (push ops (mk3 "新建目录" op_new_dir 1))
    (push ops (mk3 "重命名" op_rename 1))
    (push ops (mk3 "删除" op_delete 1))
    (push ops (mk3 "上级" op_up 1))
    (push ops (mk3 "复制" op_copy 0))   ;; 预留：可用标志 0 = 灰显
    (push ops (mk3 "移动" op_move 0))   ;; 预留
    (push ops (mk3 "刷新" op_refresh 1))
    #t))

;; 第 idx 个操作按钮的左上角；越界返回 #f
(define op_pos
  (lambda (idx)
    (define n (length ops))
    (if (or (< idx 0) (>= idx n))
        #f
        (if (< idx 5)
            (mk2 (+ 2 (* idx (+ BTN_W BTN_GAP))) BAR1_Y)
            (mk2 (+ 2 (* (- idx 5) (+ BTN_W BTN_GAP))) BAR2_Y)))))

;; 像素坐标 → 操作下标（#f = 不在任何按钮上）
(define op_hit
  (lambda (px py)
    (define row 0)
    (define base 0)
    (define i 0)
    (define p #f)
    (define r #f)
    (define stop 0)
    (if (and (>= py BAR1_Y) (< py (+ BAR1_Y BTN_H)))
        { (set! row 0) (set! base 0) }
        (if (and (>= py BAR2_Y) (< py (+ BAR2_Y BTN_H)))
            { (set! row 1) (set! base 5) }
            (set! row -1)))
    (if (>= row 0)
        { (set! i base)
          (while (and (< i (length ops)) (== stop 0))
            { (set! p (op_pos i))
              (if p
                  (if (and (>= px (get_item p 0)) (< px (+ (get_item p 0) BTN_W)))
                      { (set! r i) (set! stop 1) }
                      #f)
                  (set! stop 1))
              (set! i (+ i 1)) }) })
    r))

;; ---- 绘制 ----------------------------------------------------------------

(define draw_button
  (lambda (label x y enabled)
    (if (== enabled 1)
        { (Gfx.fill_rect x y BTN_W BTN_H 48 56 74)
          (Gfx.draw_rect x y BTN_W BTN_H 92 104 130)
          (Gfx.draw_text label (+ x (Math.floor (/ (- BTN_W (FSX.text_w label)) 2))) (+ y 2) 226 232 244) }
        { (Gfx.fill_rect x y BTN_W BTN_H 32 34 40)
          (Gfx.draw_rect x y BTN_W BTN_H 60 64 74)
          (Gfx.draw_text label (+ x (Math.floor (/ (- BTN_W (FSX.text_w label)) 2))) (+ y 2) 96 100 110) })
    #t))

(define draw_ops
  (lambda ()
    (define n (length ops))
    (define i 0)
    (define p #f)
    (define it #f)
    (while (< i n)
      { (set! p (op_pos i))
        (if p
            { (set! it (get_item ops i))
              (draw_button (get_item it 0) (get_item p 0) (get_item p 1) (get_item it 2)) }
            #f)
        (set! i (+ i 1)) })
    ;; 第二排右侧的提示（说明两枚灰按钮的来历）
    (Gfx.draw_text "灰=预留" (- SW 52) (+ BAR2_Y 2) 70 76 90)
    #t))

;; 顶部的 ".." 行（返回上级目录）：根目录下变灰（按下只提示，不跳转）
(define draw_parent_row
  (lambda (y)
    (define cr 96)
    (define cg 100)
    (define cb 110)
    (if (== sel 0)
        { (Gfx.fill_rect 2 (+ y 1) (- SW 8) (- ROW_H 2) 40 50 68)
          (Gfx.fill_rect 2 (+ y 1) 3 (- ROW_H 2) 120 180 250) }
        #f)
    (if (not (at_root)) { (set! cr 138) (set! cg 190) (set! cb 255) } #f)
    (Gfx.draw_text (if (at_root) "..   （已在资源根）" "..   上级目录") 10 (+ y 1) cr cg cb)
    #t))

(define draw_row
  (lambda (idx y)
    (define row (get_item rows idx))
    (define nm (get_item row 0))
    (define isd (get_item row 1))
    (define txt nm)
    (define cr 214) (define cg 222) (define cb 234)
    (define sz 0)
    ;; idx 是**条目**下标，显示行 = idx+1（0 是 ".."），高亮判定要用显示行
    (if (== (+ idx 1) sel)
        { (Gfx.fill_rect 2 (+ y 1) (- SW 8) (- ROW_H 2) 40 50 68)
          (Gfx.fill_rect 2 (+ y 1) 3 (- ROW_H 2) 120 180 250) }
        #f)
    (if (== isd 1)
        { (set! txt (String.concat nm "/"))
          (set! cr 138) (set! cg 190) (set! cb 255) }
        (set! txt nm))
    (Gfx.draw_text (FSX.clip_text txt 232) 10 (+ y 1) cr cg cb)
    (if (== isd 0)
        { (set! sz (get_item row 2))
          (if (> sz 0)
              (Gfx.draw_text (FSX.human_size sz) (- SW 62) (+ y 1) 104 116 138)
              #f) }
        #f)
    #t))

;; 右侧细滚动条（只在条目多于一屏时画）：位置 = first 在可滚动范围里的比例
(define draw_scrollbar
  (lambda (n)
    (define vh (* LIST_ROWS ROW_H))
    (define bar 0)
    (define by 0)
    (if (> n LIST_ROWS)
        { (set! bar (Math.floor (/ (* vh LIST_ROWS) n)))
          (if (< bar 14) (set! bar 14) #f)
          (set! by (+ LIST_Y (Math.floor (/ (* first (- vh bar)) (- n LIST_ROWS)))))
          (Gfx.fill_rect (- SW 4) LIST_Y 3 vh 32 38 50)
          (Gfx.fill_rect (- SW 4) by 3 bar 96 116 148) }
        #f)
    #t))

(define draw_info
  (lambda ()
    (define row (sel_row))
    (Gfx.fill_rect 0 (- INFO_Y 3) SW 16 26 30 38)
    (if (> flash_ttl 0)
        (Gfx.draw_text (FSX.clip_text flash_text 292) 10 INFO_Y flash_r flash_g flash_b)
        (if row
            { (Gfx.draw_text (FSX.clip_text
                               (String.concat "/" (FSX.join cur (get_item row 0))) 180)
                             10 INFO_Y 150 172 200)
              (Gfx.draw_text (if (== (get_item row 1) 1) "目录" "文件")
                             (- SW 54) INFO_Y 120 136 158) }
            ;; 选中 ".." 行时描述它（空目录/读不出来也一样——列表区另有专门提示）
            (Gfx.draw_text (if (at_root) "..   资源根（没有更上级目录）"
                                          "..   点按返回上级目录")
                           10 INFO_Y 120 140 165)))
    #t))

(define draw_all
  (lambda ()
    (define n (n_disp))
    (define i 0)
    (define last 0)
    (define y 0)
    (Gfx.fill_rect 0 0 SW SH 14 17 22)
    (Gfx.fill_rect 0 0 SW HEAD_H 30 38 52)
    (Gfx.draw_text "文件管理" 6 3 236 242 250)
    ;; 路径行 + 条目数（右上角留给 EX 的「退出」按钮）
    (Gfx.draw_text (FSX.clip_text (dir_label) 236) 6 (+ PATH_Y 1) 150 172 200)
    (Gfx.draw_text (String.concat (String.atom_to_string (n_rows)) " 项")
                   (- SW 54) (+ PATH_Y 1) 120 136 158)
    ;; 列表
    (set! last (+ first LIST_ROWS))
    (if (> last n) (set! last n) #f)
    { (set! i first)
      (while (< i last)
        { (set! y (+ LIST_Y (* (- i first) ROW_H)))
          (if (== i 0) (draw_parent_row y) (draw_row (- i 1) y))
          (set! i (+ i 1)) }) }
    ;; 空目录/读不出来时，在 ".." 行下面给一句提示
    (if (== (n_rows) 0)
        (Gfx.draw_text (if (== err 1) "（目录无法读取）" "（空目录 · 用「新建文件/新建目录」开始）")
                       10 (+ (+ LIST_Y ROW_H) 2) 110 118 134)
        #f)
    (draw_scrollbar n)
    (draw_info)
    (draw_ops)
    (EX.draw_button)
    (Gfx.refresh)
    #t))

;; ---- 输入 ----------------------------------------------------------------

(define handle_tap
  (lambda (px py)
    (define idx #f)
    (define op_idx #f)
    (define row #f)
    (define old 0)
    (set! op_idx (op_hit px py))
    (if op_idx
        (if (== (get_item (get_item ops op_idx) 2) 1)
            ((get_item (get_item ops op_idx) 1))
            (say "该操作尚未实现（预留位）"))
        (if (and (>= py LIST_Y) (< py (+ LIST_Y (* LIST_ROWS ROW_H))))
            { (set! idx (+ first (Math.floor (/ (- py LIST_Y) ROW_H))))
              (if (< idx (n_disp))
                  { (set! old sel)
                    (set! sel idx)
                    (clamp_sel)
                    ;; 再点已选中的行 = 进入 / 详情 / 上级（小屏没有双击）
                    (if (== old idx) (op_enter) #f) }
                  #f) }
            #f))
    #t))

(define handle_touch
  (lambda ()
    (define dy 0)
    (if (Touch.is_pressed)
        { (if (== prev_down 0)
              { (set! prev_down 1)
                (set! drag_total 0)
                (set! drag_last (Touch.y))
                (set! press_x (Touch.x))
                (set! press_y (Touch.y)) }
              #f)
          (set! dy (- (Touch.y) drag_last))
          (set! drag_last (Touch.y))
          (set! drag_total (+ drag_total dy))
          ;; 在列表区拖动 = 滚动
          (if (and (>= press_y LIST_Y) (< press_y (+ LIST_Y (* LIST_ROWS ROW_H))))
              { (set! first (- first dy))
                (if (< first 0) (set! first 0) #f)
                (if (> (+ first LIST_ROWS) (n_disp))
                    (set! first (- (n_disp) LIST_ROWS))
                    #f)
                (if (< first 0) (set! first 0) #f) }
              #f) }
        { (if (== prev_down 1)
              { (set! prev_down 0)
                (if (< (Math.abs drag_total) 8)
                    (handle_tap press_x press_y)
                    #f) }
              #f) })
    #t))

(define handle_keys
  (lambda ()
    (define k 0)
    (set! k (Key.take_press))
    (while (not (== k 0))
      { (if (== k 1) { (set! sel (- sel 1)) (clamp_sel) } #f)
        (if (== k 2) { (set! sel (+ sel 1)) (clamp_sel) } #f)
        (if (== k 13) (op_enter) #f)
        (if (== k 8) (op_up) #f)
        (if (== k 127) (op_delete) #f)
        (set! k (Key.take_press)) })
    #t))

;; ---- 主循环 --------------------------------------------------------------

(define tick
  (lambda ()
    { (EX.check)
      (if (> flash_ttl 0)
          { (set! flash_ttl (- flash_ttl 1))
            (if (== flash_ttl 0) (set! flash_text "") #f) }
          #f)
      (handle_keys)
      (handle_touch)
      (draw_all) }))

;; ---- 启动 ----------------------------------------------------------------

(define boot
  (lambda ()
    { (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (build_ops)
      (load_dir)
      (draw_all)
      (set! ts (System.set_interval 60 tick)) }))

(EX.install_with draw_all)
(boot)