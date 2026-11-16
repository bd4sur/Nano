;; ============================================================================
;; filemgr.scm —— 文件管理器（灵机应用程序 · 完全权限版）
;;
;; 能力（全部经 (native Fs) 真文件链路，桌面 / 浏览器 MEMFS / 掌机 SD 行为一致）：
;;   · 列表查看：当前目录下的**文件与目录**——**目录永远排在文件前面，同类内按名字升序**
;;     （排序在 Scheme 层做，不依赖各平台 readdir 顺序）；目录以「名字/」+ 蓝色区分；
;;   · 列表顶部固定一条 **".."（返回上级目录）** 条目（根目录下变灰，点了只提示）；
;;   · 进入目录 / 返回上级；文件选中后「进入」= **用对应类型的 App 打开**
;;     （映射表见下方 openers；txt / md / markdown → Markdown 阅读器 App，
;;      png / jpg / jpeg / bmp / gif / tga → 时光集（图片查看器）App，
;;      wav / mp3 → 音乐盒 App，传绝对路径直接开播）；
;;     没有映射的类型仍弹出详情框（路径 / 大小 / 类型）；
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
;;   · 点按列表行 = 选中；再点已选中的行 = 进入目录 / 打开文件 / 返回上级（小屏没有双击，
;;     靠“再点一次”）；
;;   · **拖动列表 = 像素级跟手滚动**（手指移多少内容移多少，不是整行跳），**松手带惯性**
;;     （速度快的划动继续滑一段再停），按住不动即刹车（打断惯性）；位移 < 8px 才算点按；
;;   · 实体键盘 ↑↓ 选择（选中行自动滚进视口），Enter 进入，退格(8) 上级；
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
(native Diag)
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
;; 列表行文字的垂直补偿：本地库现在把 (Gfx.draw_text) 的 y 当「墨迹顶」绘制
;; （见 am_native_Gfx.c 的 native_text_top_gap，12px 抗锯齿字模的行顶空隙 2px），
;; 而本 App 的行文字是按旧口径（行顶 = 行框顶 + 1，墨迹再落 2px）摆的；不补回这 2px，
;; 整列文件名（含右侧的文件大小）都会比行框中线高出一截。
(define ROW_TXT_DY 2)
(define INFO_Y 182)     ;; 选中项信息 / 操作反馈，一行 12px
(define BAR1_Y 198)     ;; 第一排按钮
(define BAR2_Y 218)     ;; 第二排按钮
(define BTN_H 16)
(define BTN_W 60)
(define BTN_GAP 4)
(define BOT_H 8)        ;; 底边留白（EX 的退出按钮在**顶部**，这里不冲突）

;; ---- 滚动手感参数（列表的触屏拖动） --------------------------------------

(define VIEW_H (* LIST_ROWS ROW_H))  ;; 列表视口高（8 行 = 144px）
;; 节拍 16ms = 60Hz。拖动跟手的**上限就是节拍频率**：App 每拍只能采一次触屏电平，
;; 原来 60ms（16fps）的拍子再准也只能一格一格跳；改成 60Hz 后与屏幕刷新同频。
;; 空闲帧不再重绘（见 dirty 标记），所以 60Hz 空转只是几次 native 取值，代价可忽。
(define TICK_MS 16)
(define FLING_FRICTION 0.88)  ;; 惯性每拍衰减系数（16ms/拍）
(define FLING_VMAX 45)        ;; 惯性初速上限（px/拍，≈2800px/s）
(define FLING_VMIN 0.6)       ;; 惯性速度低于它就停（px/拍）
;; 松手是否"甩"：看这次手势的**平均速度**（px/16ms）。取平均而不是"最后两拍瞬时速度"：
;; 后者与时基强耦合（60Hz 与掉帧后的拍长不同，同一个手势算出的速度差几倍），
;; 而且"停住再抬手"必须不甩——平均速度天然把这段停顿算进去。
(define FLING_MIN_V 5)        ;; 平均速度 ≥ 5px/16ms（≈310px/s）才算"甩" 
(define TAP_SLOP 8)           ;; 位移 < 8px 视为点按而不是拖动
(define FLASH_TICKS 90)       ;; 反馈文案显示时长（≈1.4s @60Hz）

;; ---- 状态 ----------------------------------------------------------------

(define cur ".")             ;; 当前目录（相对业务资源根；"." = 根）
(define rows (cdr '(0)))     ;; 条目表：(mk3 名字 是否目录 字节数)，已按名字升序
                             ;; （尺寸也在载入时缓存：绘制不再逐帧 stat）
(define sel 0)               ;; 选中下标（0..n-1）
;; 列表滚动：**像素级**偏移（0 = 首行贴住视口顶）。
;; 以前是“窗口首项 first（整行号）”：拖动只能整行跳、每 18px 顿一下，观感是一格一格地顿；
;; 改成像素偏移后手指移多少内容就移多少（1:1 跟手），半行、1/4 行都能停在视口上，
;; 配合 60Hz 节拍 + 松手惯性，连续且顺滑。
(define scroll 0)
(define fling 0)             ;; 松手后的惯性速度（px/拍；正 = 内容继续向上走）
(define sum_dy 0)            ;; 本次手势的累计纵向位移（px，判"甩"用）
(define press_ms 0)          ;; 按下时刻（(Diag.ms)，真实毫秒；手势计时用，与拍长无关）
(define dirty 1)             ;; 需要重绘（tick 里按需 draw：空闲帧零绘制）
(define err 0)               ;; 目录读不出来（Fs.list → #null）
(define flash_text "")       ;; 操作反馈（几拍后自动消失）
(define flash_ttl 0)
(define flash_r 190) (define flash_g 200) (define flash_b 150)

;; 触屏
(define prev_down 0)
(define drag_last 0)
(define drag_max 0)          ;; 本次手势离按下点的最大位移（px，判断点按/拖动）
(define press_x 0)
(define press_y 0)
(define press_in_list 0)     ;; 按下点落在列表视口里（只有它会滚动）
(define press_stop 0)        ;; 按下时在刹车（打断惯性）→ 这一下松手不算点按

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
(define say
  (lambda (msg)
    { (set! flash_r 190) (set! flash_g 200) (set! flash_b 150)
      (set! flash_text msg)
      (set! flash_ttl FLASH_TICKS)
      (set! dirty 1)
      (display "FM ") (display msg) (newline) }))

(define say_warn
  (lambda (msg)
    { (set! flash_r 240) (set! flash_g 150) (set! flash_b 120)
      (set! flash_text msg)
      (set! flash_ttl FLASH_TICKS)
      (set! dirty 1)
      (display "FMW ") (display msg) (newline) }))

;; ---- 滚动：限幅 / 选中项进视口 / 惯性积分 --------------------------------

;; 可滚动的最大偏移（内容高 − 视口高；内容不足一屏时为 0）
(define max_scroll
  (lambda ()
    (define m (- (* (n_disp) ROW_H) VIEW_H))
    (if (< m 0) 0 m)))

(define clamp_scroll
  (lambda ()
    { (if (< scroll 0) (set! scroll 0) #f)
      (if (> scroll (max_scroll)) (set! scroll (max_scroll)) #f)
      #t }))

;; 把选中行滚进视口（键盘上下、点按、新建/改名定位后用）
(define ensure_sel_visible
  (lambda ()
    (define top 0)
    (define bot 0)
    (set! top (* sel ROW_H))
    (set! bot (+ top ROW_H))
    (if (< top scroll) (set! scroll top) #f)
    (if (> bot (+ scroll VIEW_H)) (set! scroll (- bot VIEW_H)) #f)
    (clamp_scroll)
    #t))

;; 惯性初速限幅
(define clamp_fling
  (lambda (v)
    (if (> v FLING_VMAX) FLING_VMAX
        (if (< v (- 0 FLING_VMAX)) (- 0 FLING_VMAX) v))))

;; 每拍推进一歩惯性：位移按速度、速度按摩擦衰减，碰到边界就停
;;（不做回弹：小屏文件列表上“硬边界”比橡皮筋更好控）
(define step_fling
  (lambda ()
    (if (not (== fling 0))
        { (set! scroll (- scroll fling))
          (set! dirty 1)
          (if (or (< scroll 0) (> scroll (max_scroll)))
              { (clamp_scroll) (set! fling 0) }
              (set! fling (* fling FLING_FRICTION)))
          (if (< (Math.abs fling) FLING_VMIN)
              { (set! fling 0)
                (display "FMFLING end=")
                (display (Math.floor scroll)) (newline) }
              #f) }
        #f)
    #t))

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
    { (if (>= sel (n_disp)) (set! sel (- (n_disp) 1)) #f)
      (if (< sel 0) (set! sel 0) #f)
      (ensure_sel_visible)
      (set! dirty 1)
      #t }))

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
      (set! scroll 0)  ;; 换目录：回到列表顶
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

;; ---- 打开即拉起：文件类型 → 灵机 App 映射表 ----------------------------
;;
;; "打开文件"在这里不是本 App 自己实现的事，而是**把控制权交给能看懂它的 App**：
;;   · 表项 = (mk2 扩展名列表 应用名)；扩展名列表不含点、逗号分隔、大小写不敏感
;;     （直接用 FSX.ext_match 的既有语义）；应用名 = app/ 下源码文件名去 .scm
;;     （即「灵机应用程序」列表里的显示名）；
;;   · 拉起时传一个字符串参数 = **待打开文件的宿主绝对路径**（用 (Fs.abspath) 从
;;     资源根相对路径换算；约定见 am_native_App.h 的 (App.launch_with ...)）；
;;   · 宿主把本 App **冻结保留**（列表里带「● 运行中」角标），目标 App 退出后**回到本 App**
;;     （实例级返回链，见 am_native_App.h 的“返回链”；本 App 的目录与选中行原样保留）；
;;   · 新增一个"查看器"只需在这张表里加一行（并保证那个 .scm 在 app/ 下）。
;; 表项顺序即匹配优先级；没有命中的类型照旧弹详情框（行为与以前一致）。
(define openers (cdr '(0)))

(define build_openers
  (lambda ()
    (set! openers (new_list))
    ;; 图片 → 时光集（app/album.scm）：扩展名表与 album 的 ext_ok 保持一致
    ;; （stb_image 能解码的常见格式；改一处要同步改另一处）
    (push openers (mk2 "png,jpg,jpeg,bmp,gif,tga" "album"))
    (push openers (mk2 "md,markdown" "markdown"))
    (push openers (mk2 "txt" "markdown"))
    ;; 音频 → 音乐盒（app/musicbox.scm）：只列 (native Music) **实际能解码**的两种格式
    ;; （WAV / MP3，见 am_native_Music.h）；musicbox 收到绝对路径就直接打开播放该文件，
    ;; 并把列表换成它所在目录（上一首/下一首仍在同一目录里）。
    (push openers (mk2 "wav,mp3" "musicbox"))
    #t))

;; 文件名 → 应用名（#f = 没有可拉起的 App，调用方退化为显示详情）
(define opener_for
  (lambda (name)
    (define n (length openers))
    (define i 0)
    (define r #f)
    (while (< i n)
      { (if r
            #f
            (if (== (FSX.ext_match name (get_item (get_item openers i) 0)) 1)
                (set! r (get_item (get_item openers i) 1))
                #f))
        (set! i (+ i 1)) })
    r))

;; 打开一个文件：有映射 → 拉起对应 App（传绝对路径）；无映射 → 显示详情
(define op_open
  (lambda (name)
    (define app (opener_for name))
    (define ap #f)
    (if app
        { (set! ap (Fs.abspath (FSX.join cur name)))
          (if (null? ap)
              (say_warn "路径过长，无法作为启动参数传递")
              (if (App.launch_with app ap)
                  { (display "FMOPEN ") (display name) (display " -> ") (display app) (newline)
                    (say (String.concat "打开 " (String.concat name
                           (String.concat " → " app)))) }
                  (say_warn (String.concat "拉起失败：" app)))) }
        (show_info name))))

(define op_enter
  (lambda ()
    (define row (sel_row))
    (if row
        (if (== (get_item row 1) 1)
            (goto (FSX.join cur (get_item row 0)))
            (op_open (get_item row 0)))
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
    (Gfx.draw_text (if (at_root) "..   （已在资源根）" "..   上级目录") 10 (+ (+ y 1) ROW_TXT_DY) cr cg cb)
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
    (Gfx.draw_text (FSX.clip_text txt 232) 10 (+ (+ y 1) ROW_TXT_DY) cr cg cb)
    (if (== isd 0)
        { (set! sz (get_item row 2))
          (if (> sz 0)
              (Gfx.draw_text (FSX.human_size sz) (- SW 62) (+ (+ y 1) ROW_TXT_DY) 104 116 138)
              #f) }
        #f)
    #t))

;; 右侧细滚动条（只在条目多于一屏时画）：滑块位置与大小都由**像素偏移**
;; scroll / max_scroll 推出（与列表的像素滚动同一口径，停在半行时滑块也是连续位置）
(define draw_scrollbar
  (lambda ()
    (define ms (max_scroll))
    (define bar 0)
    (define by 0)
    (if (> ms 0)
        { (set! bar (Math.floor (/ (* VIEW_H VIEW_H) (+ ms VIEW_H))))
          (if (< bar 14) (set! bar 14) #f)
          (set! by (+ LIST_Y (Math.floor (/ (* scroll (- VIEW_H bar)) ms))))
          (Gfx.fill_rect (- SW 4) LIST_Y 3 VIEW_H 32 38 50)
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

;; 列表区：像素滚动 + 视口裁剪。
;; 裁剪为何必要：停在半行时首/尾两行只露半截，高亮条与字形的墨迹会溢出到页眉、
;; 信息行上；(Gfx.set_clip) 对 fill_rect / 字形绘制都生效（见 graphics.h 的裁剪矩形），
;; 所以这里先夹后画、画完立即 reset_clip（否则底栏按钮与滚动条也会被裁掉）。
(define draw_list
  (lambda ()
    (define n (n_disp))
    (define i 0)
    (define last 0)
    (define y 0)
    (Gfx.set_clip 0 LIST_Y SW VIEW_H)
    (Gfx.fill_rect 0 LIST_Y SW VIEW_H 14 17 22)
    ;; 从“视口顶那一行”开始画，多画两行（上下各一行半截，其余交给裁剪）
    (set! i (Math.floor (/ scroll ROW_H)))
    (set! last (+ i (+ LIST_ROWS 2)))
    (if (> last n) (set! last n) #f)
    (while (< i last)
      { (set! y (- (+ LIST_Y (* i ROW_H)) scroll))
        (if (== i 0) (draw_parent_row y) (draw_row (- i 1) y))
        (set! i (+ i 1)) })
    ;; 空目录/读不出来时，在 ".." 行下面给一句提示
    (if (== (n_rows) 0)
        (Gfx.draw_text (if (== err 1) "（目录无法读取）" "（空目录 · 用「新建文件/新建目录」开始）")
                       10 (+ (- (+ LIST_Y ROW_H) scroll) 2) 110 118 134)
        #f)
    (Gfx.reset_clip)
    #t))

(define draw_all
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH 14 17 22)
    (Gfx.fill_rect 0 0 SW HEAD_H 30 38 52)
    (Gfx.draw_text "文件管理" 6 3 236 242 250)
    ;; 路径行 + 条目数（右上角留给 EX 的「退出」按钮）
    (Gfx.draw_text (FSX.clip_text (dir_label) 236) 6 (+ PATH_Y 1) 150 172 200)
    (Gfx.draw_text (String.concat (String.atom_to_string (n_rows)) " 项")
                   (- SW 54) (+ PATH_Y 1) 120 136 158)
    (draw_list)
    (draw_scrollbar)
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
        (if (and (>= py LIST_Y) (< py (+ LIST_Y VIEW_H)))
            ;; 行号由**像素偏移**反算：视口顶对应内容的第 scroll 个像素
            { (set! idx (Math.floor (/ (+ (- py LIST_Y) scroll) ROW_H)))
              (if (and (>= idx 0) (< idx (n_disp)))
                  { (set! old sel)
                    (set! sel idx)
                    (clamp_sel)
                    ;; 再点已选中的行 = 进入 / 详情 / 上级（小屏没有双击）
                    (if (== old idx) (op_enter) #f) }
                  #f) }
            #f))
    #t))

;; 触屏：拖动 = **像素级跟手**滚动（指移多少内容移多少）+ 松手惯性。
;; 判据与手感细节：
;;   · 是否“拖动”看离按下点的**最大位移**（含横向），而不是纵向位移的累加：
;;     来回揉手不会把一次长拖动误判成点按；
;;   · 按下即断惯性（刹车）：这一下的松手不算点按（否则“想停住”会变成“打开了文件”）；
;;   · 速度用指数滑动平均（α=0.5）估，松手时若真的拖动过就交给 fling；
;;     手指停住不动时 dy=0，均值自然衰减，不会“停了很久一松手又飞出去”；
;;   · 只有**按下点落在列表视口内**才滚动（从底栏按钮上开始的拖动不误触列表）。
(define handle_touch
  (lambda ()
    (define dy 0)
    (define dx 0)
    (define ady 0)
    (define dur 0)
    (define vavg 0)
    (if (Touch.is_pressed)
        { (if (== prev_down 0)
              { (set! prev_down 1)
                (set! press_stop (if (== fling 0) 0 1))
                (set! fling 0)
                (set! sum_dy 0)
                (set! press_ms (Diag.ms))
                (set! drag_max 0)
                (set! drag_last (Touch.y))
                (set! press_x (Touch.x))
                (set! press_y (Touch.y))
                (set! press_in_list
                      (if (and (>= press_y LIST_Y) (< press_y (+ LIST_Y VIEW_H))) 1 0)) }
              #f)
          (set! dy (- (Touch.y) drag_last))
          (set! drag_last (Touch.y))
          (set! dx (- (Touch.x) press_x))
          (if (> (Math.abs dx) drag_max) (set! drag_max (Math.abs dx)) #f)
          (set! ady (Math.abs (- (Touch.y) press_y)))
          (if (> ady drag_max) (set! drag_max ady) #f)
          (if (and (== press_in_list 1) (not (== dy 0)))
              { (set! sum_dy (+ sum_dy dy))
                (set! scroll (- scroll dy))
                (set! dirty 1)
                (clamp_scroll) }
              #f) }
        { (if (== prev_down 1)
              { (set! prev_down 0)
                ;; 松手这一拍先把“最后一次没被采到的位移”补上：App 每拍只能采一次触屏
                ;; 电平（60Hz），一次快速划动的末段 10~20px 很容易落在两次采样之间；
                ;; 不在松手时补，手指移了 30px 内容却只走 20px——观感就是“没跟上”。
                (if (== press_in_list 1)
                    { (set! dy (- (Touch.y) drag_last))
                      (set! dx (- (Touch.x) press_x))
                      (if (> (Math.abs dx) drag_max) (set! drag_max (Math.abs dx)) #f)
                      (set! ady (Math.abs (- (Touch.y) press_y)))
                      (if (> ady drag_max) (set! drag_max ady) #f)
                      (if (not (== dy 0))
                          { (set! sum_dy (+ sum_dy dy))
                            (set! scroll (- scroll dy))
                            (set! dirty 1)
                            (clamp_scroll) }
                          #f) }
                    #f)
                ;; 刹车（按下时打断了惯性）只压制"点按"，不压制"真拖动后的甩"
                (if (< drag_max TAP_SLOP)
                    (if (== press_stop 0) (handle_tap press_x press_y) #f)
                    (if (== press_in_list 1)
                            { ;; 手势平均速度 → 惯性初速（与拍长无关；停住再抬手自然不甩）
                              (set! dur (- (Diag.ms) press_ms))
                              (if (< dur 1) (set! dur 1) #f)
                              (set! vavg (/ (* sum_dy 16.0) dur))
                              (display "FMDRAG px=")
                              (display (Math.floor scroll))
                              (display " v=")
                              (display (Math.floor (* vavg 10)))
                              (newline)
                              (if (> (Math.abs vavg) FLING_MIN_V)
                                  { (set! fling (clamp_fling vavg))
                                    (set! dirty 1)
                                    (display "FMFLING start=")
                                    (display (Math.floor scroll)) (newline) }
                                  #f) }
                            #f)) }
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

;; 节拍 60Hz；**只在脏的时候重绘**（空闲帧只做几次 native 取值，几乎不花 CPU）。
;; 脏的来源：目录重载 / 选中变化 / 滚动与惯性 / 反馈文案的出现与消失 /
;; 宿主模态框关闭或实例恢复（画面归自己，必须补一帧）。
(define tick
  (lambda ()
    { (EX.check)
      (if (App.modal_closed) (set! dirty 1) #f)
      (if (> flash_ttl 0)
          { (set! flash_ttl (- flash_ttl 1))
            (if (== flash_ttl 0) { (set! flash_text "") (set! dirty 1) } #f) }
          #f)
      (handle_keys)
      (handle_touch)
      (step_fling)
      (if (== dirty 1) { (set! dirty 0) (draw_all) } #f) }))

;; ---- 恢复钩子（从别的 App 返回 / 从后台恢复）----------------------------

;; 从别的 App 返回 / 从后台恢复：**重新扫一遍目录**。
;; 为什么不用 op_refresh：返回是常态（读文档、看图之后都会回），不该每次都闪一条
;; “已刷新”；而且这里必须自己把 dirty 置上（EX 的恢复钩子**代替**了重绘回调）。
;; 别的 App 可能刚保存/改名/删除了文件，不重扫就会显示过期的列表。
(define on_resume
  (lambda ()
    { (load_dir)
      (set! dirty 1)
      (display "FMRESUME ") (display cur) (newline) }))

;; ---- 启动 ----------------------------------------------------------------

(define boot
  (lambda ()
    { (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (build_ops)
      (build_openers)
      (load_dir)
      (draw_all)
      (set! dirty 0)
      (set! ts (System.set_interval TICK_MS tick)) }))

(EX.install_with draw_all)
(EX.on_resume on_resume)   ;; 返回/恢复时重扫当前目录（见 on_resume 的说明）
(boot)