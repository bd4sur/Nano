;@id=markdown
;@title=MD阅读器
;@version=2026-11-20
;; ============================================================================
;; Markdown 阅读器 —— 灵机应用程序（解析 / 排版 / 渲染全部用 Scheme 实现）
;;
;; 结构（四个模块，各管一段，可分别单测）：
;;   app/lib/md_parse.scm   文本 → 块树（块级规则表 + 行内规则表，扩展点见其文件头）
;;   app/lib/md_layout.scm  块树 → 显示行（行 = 高度/背景/段列表；段自带绝对 x 与宽度）
;;   app/lib/md_stream.scm  分段流式读取（大文件底座：字节流 → 段；见其文件头）
;;   app/lib/md_view.scm    分段视图（段缓存 / 锚点滚动 / 预取 / 高度估算 / 目录）
;;   本文件                 绘制 + 交互（滚动/翻页/点链接/退出）
;;
;; 大文件（≤1MB）怎么读：**不再整篇解析+整篇排版**，而是
;;   ① 打开：只切出并排版**第一段**（4KB 源码，首屏 ~90ms），其余段在后台按拍建索引；
;;   ② 滚动：视口附近的若干段被排版并缓存（上限 CACHE_LINES 行），段内滚动零重排，
;;      跨界时才布局相邻段（预取保证提前一两屏就做好了）；
;;   ③ 段高度只有排版过才精确：未访问段按"码点数 × 实测校准系数"估算，而滚动位置由
;;      **锚点 (段号, 段内 dy)** 决定，所以估算被修正时画面不跳；页眉百分比在索引
;;      未覆盖全文时显示 "≈"。
;;   实测（WASM，池 32MB / heap 16MB）：1MB 文档开屏 ~90ms、堆 ~5.8MB、翻页 0.4ms/屏；
;;   而改造前整篇物化在 128KB 就接近 16MB 上限、1MB 需要 >60MB 与 ~15s。
;;
;; 启动两拍（先安抚，再干活）：见 boot / boot_work / draw_splash。
;;   ① boot 只设几何、画一屏「启动提示」就 return（并装好常规拍）；
;;   ② 读字模记录、续读上次文档、切首段、**解析 + 排版首段（~60-136ms）** 全放到
;;      第一个常规拍（boot_work）里，做完再交回常规循环（同一条定时器，相位不变）。
;;   为什么要分拍而不是「在 boot 里画一帧就接着干」：宿主导入 App 时先把屏幕 soft_clear
;;   成纯黑，而 WASM 的 (Gfx.refresh) 只是同步 putImageData —— 浏览器要等 JS 栈退空才
;;   合成，所以留在 boot 里画的那一帧用户根本看不到（掌机/桌面的 SPI/直接上屏看得见）。
;;   跨一拍让出 = 栈退空 = 提示真的上屏。代价只有一拍（TICK_MS = 60ms）。
;;
;; 节拍两档（60ms / 16ms）：静止阅读 60ms 省电；**手指拖动、滚轮缓动期间**提到 16ms
;;   （跟手的上限就是节拍频率）。实测拖动 15fps→52~57fps（900KB 文档因字型栅格化
;;   占满一拍，只到 ~40fps）；滚轮一格 28px 缓动 5 拍 ≈77ms。见 pace_tick。
;;
;; 甩滑惯性（触屏松手后继续滑，见 fling_start / handle_fling）：
;;   速度取**近几拍的 EMA**（不是整段手势平均）→ 慢慢停下再松手不会甩出去；按下即
;;   刹车；惯性期间节拍保持 16ms；撞到上/下边界立即停（不回弹）。参数对齐宿主列表
;;   （app/lib/filepicker.scm 的 P_FLING_*）：阈值 320px/s、每 16ms 保留 0.88、
;;   初速上限 2.8px/ms、衰减到 0.05px/ms 停。一次快甩多滑 ~60~150px。
;;
;; 鼠标滚轮/触控板（只 WASM GUI 有）：走**原生滚轮通道**而不是合成手指——
;;   合成手指在长文档上有三个结构性缺陷（走不出画布 / 重按被合进上一段手势 → 凭空
;;   一段反向位移 / 首格位移被整段丢掉），实测 5 格只前进 56px（应 140px）。
;;   boot 里 (Touch.wheel_enable) 认领后，页面把滚轮的像素增量**原样**交过来
;;   （(Touch.wheel)，正 = 内容上滚），本文件每拍取走、按 WHEEL_GAIN 缓动落位
;;   （handle_wheel），并在有滚动期间把节拍提到 16ms（整屏重绘的 60ms 拍会把
;;   缓动压成 16fps 的跳动）。实测一格恰好 28px、正反严格 1:1，且不再有反向位移。
;;   掌机/桌面返回 wheel_available = #f，WHEEL_OK 恒 0，以下逻辑全部走空。
;;
;; 支持的子集：# 1~3 级标题、**加粗**、`行内代码`、无序列表（**可嵌套**）、[链接](url)、
;;   围栏代码块、分隔线、**表格**（GFM：列对齐、单元格内行内样式、自动列宽、单元格内
;;   折行）、**引用块**（可嵌套、内部可放任意块）、**块级数学公式**（$$...$$）、
;;   **行内数学公式**（$...$，见下）、段落（含 CJK 软换行）。有序列表留有扩展位。
;;
;; 数学公式（块级与行内共用一套引擎，见 app/lib/tex_*.scm）：
;;   · 块级：$$ 单行 $$ 或 $$ + 若干行 + $$；解析端只产出 (math 源串)，排版端按注入的
;;     度量回调定行高与居中位置（超宽则按比例缩小字号**重排**到与屏幕等宽）；
;;   · 行内：$...$ / 段落中间的 $$...$$，解析端产出 (imath 源串)；排版端以同一行文字的
;;     **基线**为对齐基准，公式比文字高时用「上方空行 + 抬高本行行高」真实让位
;;     （见 md_layout 的 IMATH_ASC/DSC），超高/超宽同样按比例缩小字号重排；
;;   · 绘制端都画缓存的 op 流（行内段带 em，按 (源串 em) 命中同一份缓存），不重排。
;;   引擎侧的 Markdown 来源容错（在 tex_parse 里）：行分隔符写成 `\\\\`（转义后两个记号）
;;   按一个处理、末尾空行丢弃——所以从网页源码里拷下来的 align/cases 能直接排对。
;;
;; 标题的视觉区分（字模只有一套 12px，靠装饰而非字号）：H1 粗体 + 2px 通栏横线；
;; H2 粗体 + 左侧 3px 装饰竖线（#66ccff）；H3 粗体 + 文字下划线。
;;
;; 文档来源（都走 Fs 真文件链路）——**没有内置示例**，能读的文档一律来自文件系统：
;;   · 启动时默认打开**上次打开的文档**：路径记在 data/md-last.txt（用户数据目录），
;;     每次成功打开一篇就覆盖写一次；启动时读回并用 (Fs.exists) 校验，存在就直接续读；
;;   · 读不到记录 / 文件已不在 → 显示空状态提示（没有文档可看），由用户自己挑一篇；
;;   · 挑文档的唯一入口是**底栏「打开」按钮**（触屏）→ 可复用的文件选择器
;;     （app/lib/filepicker.scm），在**整个资源根**里挑任意 .md/.markdown；
;;   · **被别的 App 拉起**（App.launch_with，例如 app/filemgr.scm 选中 .md/.txt）：
;;     启动参数 = 待打开文件的宿主绝对路径，boot / 每拍开头由 (App.launch_arg)
;;     取走（一次即清零），直接打开该文档（成功后同样写回 data/md-last.txt）——
;;     见下方 launch_open。记录的是**原样路径**（相对资源根或宿主绝对路径皆可），
;;     读回时只做一次 (Fs.exists) 判断，不做路径换算。
;;
;; 交互：
;;   · **底栏右侧「打开」按钮**（触屏）= 弹出文件选择器；没有打开文档时，点按正文
;;     或按 Enter 也直接弹出选择器（空状态下整屏只有这一件事可做）；
;;   · 触屏拖动 = 滚动（手指移动多少滚多少）；轻点链接 = 弹窗显示目标地址；
;;   · 实体键盘 ↑↓ 滚动一行，←/→ 翻一页；
;;   · 右上角「退出」按钮 / 硬 A(ESC) —— 由 app/lib/exit.scm 统一接管。
;;
;; 解释器约束（manual.md，逐条遵守）：算术恒二参数；无 let/cond 多表达式；
;;   '() 是模块级静态槽（空表一律 (cdr '(0)) 现造）；push 返回 #undefined 不可赋值；
;;   循环里不用 (set! i 末值) 当 break，一律用 stop 标志。
;; ============================================================================

(native Gfx)
(native Key)
(native Touch)
(native Fs)
(native String)
(native Math)
(native System)
(native Ui)
(native App)
(import EX "app/lib/exit.scm")
(import FP "app/lib/filepicker.scm")
(import FSX "app/lib/fsx.scm")
(import MD "app/lib/md_parse.scm")
(import MDL "app/lib/md_layout.scm")
(import MDV "app/lib/md_view.scm")
(import ML "app/lib/menu_list.scm")
;; 数学公式排版引擎（与 app/mathjax.scm 共用同一套：LaTeX → 语法树 → 盒子 → op 流）
(import P "app/lib/tex_parse.scm")
(import L "app/lib/tex_layout.scm")
(import D "app/lib/tex_draw.scm")
(import B "app/lib/tex_base.scm")

;; ---- 布局常量 ------------------------------------------------------------

(define SW 320)
(define SH 240)
(define TOP_H 18)     ;; 页眉高度
(define BOT_H 16)     ;; 页脚高度
(define PAD_X 6)      ;; 正文左右留白
(define CW 308)       ;; 正文宽 = SW - 2*PAD_X（启动时按实际屏宽重算）

;; 单篇文档的字节上限：1MB。
;; 为什么能放开到 1MB：本阅读器已改为**分段流式**——文档按 4KB 源码切成段，只有
;; 视口附近的若干段会被解析/排版并缓存在堆上（见 app/lib/md_stream.scm 与
;; app/lib/md_view.scm）。实测（WASM，32MB 池 / 16MB heap）：每 KB 源码要 ~40KB 堆
;; （块树）+ ~26KB（显示行，折合 ~1.1KB/显示行）+ ~15ms；整篇物化在 128KB 时就已
;; 接近 16MB 上限，1MB 需要 >60MB —— 所以必须分段，而不是放大上限。
(define DOC_MAX 1048576)  ;; 1MB

;; 分段参数（实测：布局 4KB 源码 ≈ 60ms ≈ 一个 GUI 节拍，段内滚动零重排）
(define SEG_TARGET 4096)      ;; 目标段大小（字节）；切点落在围栏外的空行处
(define CACHE_LINES 4000)     ;; 段缓存显示行上限（≈4MB 堆，见 md_view.scm）

;; 「上次打开的文档」记录文件（用户数据目录；相对业务资源根，与其它 App 的
;; data/xxx.txt 同一约定：桌面/WASM 在 /home/bd4sur/linglong/data/ 下，掌机在 SD 卡根）。
;; 内容就是一行路径（不写换行），可能是资源根相对路径，也可能是拉起方给的宿主绝对路径。
(define LAST_PATH "data/md-last.txt")

;; ---- 配色 ----------------------------------------------------------------

;; 正文段（按样式查色；style_weight = 描几次实现"伪粗体"——字模只有一套 12px，
;; 没有粗体字形，因此粗体靠同一行 1px 偏移重复描一次/两次得到视觉加粗）
(define cr 205) (define cg 214) (define cb 228) (define style_weight 1)

(define set_style_color
  (lambda (st)
    (if (== st MDL.ST_BODY)
        { (set! cr 205) (set! cg 214) (set! cb 228) (set! style_weight 1) } #f)
    (if (== st MDL.ST_BOLD)
        { (set! cr 250) (set! cg 252) (set! cb 255) (set! style_weight 2) } #f)
    (if (== st MDL.ST_CODE)
        { (set! cr 255) (set! cg 200) (set! cb 124) (set! style_weight 1) } #f)
    (if (== st MDL.ST_LINK)
        { (set! cr 112) (set! cg 182) (set! cb 255) (set! style_weight 1) } #f)
    ;; 注意：字模只有一套 12px alpha，「伪粗体」= 同一行 1px 偏移重复描。
    ;; 描 3 次对汉字会糊成一团（笔画粘连），故最多描 2 次；标题层级差异靠
    ;; 颜色亮度 + 段前留白 + 分隔线（H1 2px 通栏线、H2 1px 线、H3 无线且更暗）表达。
    (if (== st MDL.ST_H1)
        { (set! cr 255) (set! cg 255) (set! cb 255) (set! style_weight 2) } #f)
    (if (== st MDL.ST_H2)
        { (set! cr 228) (set! cg 238) (set! cb 252) (set! style_weight 2) } #f)
    (if (== st MDL.ST_H3)
        { (set! cr 176) (set! cg 197) (set! cb 228) (set! style_weight 1) } #f)
    (if (== st MDL.ST_CODEBLOCK)
        { (set! cr 172) (set! cg 214) (set! cb 180) (set! style_weight 1) } #f)
    (if (== st MDL.ST_BULLET)
        { (set! cr 130) (set! cg 170) (set! cb 225) (set! style_weight 1) } #f)
    (if (== st MDL.ST_TABLE_SEP)
        { (set! cr 58) (set! cg 70) (set! cb 92) (set! style_weight 1) } #f)
    (if (== st MDL.ST_QUOTE_BAR)
        { (set! cr 86) (set! cg 98) (set! cb 120) (set! style_weight 1) } #f)
    ;; 表格的上下框线 / 表头下框线（只在表格宽度内画）
    (if (== st MDL.ST_HR)
        { (set! cr 78) (set! cg 98) (set! cb 132) (set! style_weight 1) } #f)
    ;; 二级标题左侧装饰竖线的颜色：#66ccff
    (if (== st MDL.ST_H2BAR)
        { (set! cr 102) (set! cg 204) (set! cb 255) (set! style_weight 1) } #f)
    ;; 行内图片的降级占位文本（一期不支持行内插图）
    (if (== st MDL.ST_IMAGE_TEXT)
        { (set! cr 130) (set! cg 146) (set! cb 172) (set! style_weight 1) } #f)
    ;; 块级/行内数学公式：与 mathjax 同档的亮色（4bpp 字模在小字号上抗锯齿偏细，
    ;; 亮色观感更好）
    (if (== st MDL.ST_MATH)
        { (set! cr 234) (set! cg 238) (set! cb 248) (set! style_weight 1) } #f)
    (if (== st MDL.ST_IMATH)
        { (set! cr 234) (set! cg 238) (set! cb 248) (set! style_weight 1) } #f)
    #t))

;; ---- 状态 ----------------------------------------------------------------

(define MDV_ok 0)              ;; 1 = 流式文档（md_view），0 = 提示文案（notice）
(define notice_lines (cdr '(0)))  ;; 提示文案的显示行（整篇排版，内容极小）
(define notice_h 0)           ;; 提示文案总高
(define toc_scanned 0)        ;; 目录已覆盖的段数（= 索引推进到的段数，诊断/进度）
(define toc_gen_seen -1)      ;; 已反映到 toc_items 的目录世代号（增量刷新用）
;; 面板关闭后的若干帧里把宿主「退出」热区清零（>0 生效）：
;;   面板的「返回」键与阅读页的「退出」键**同位置**，而宿主的退出热区判定发生在触摸**边沿**
;;   （早于 App 的下一拍）。面板关闭的那一帧阅读页会重新登记热区，于是同一次点按会被宿主
;;   再判成"返回请求"，进而在阅读页弹出全局退出确认框（WASM 实测踩到）。静音几帧即可，
;;   阅读页的退出键本身仍由 EX 的电平兜底路径响应，不丢功能。
(define exitbtn_mute 0)
(define content_h 0)          ;; 全文档总高（像素；未索引部分为估算，仅用于滚动条/百分比）
(define scroll 0)             ;; 纵向滚动偏移（像素，**由锚点派生**的只读快照）
(define cur_title "")         ;; 当前文档标题（没有文档时为 App 名）
(define cur_path "")          ;; 当前文档路径（"" = 没有打开文档；换字模后据此重新排版）
(define ts 0)                 ;; 定时器编号

;; ---- 启动首帧（先安抚，再干活）----------------------------------------------
;; boot 只画提示并立刻 return，重活交给**第一个常规拍**（boot_work）——完整理由见文件头
;; 「启动两拍」。这里只留两个状态量：
;;   · boot_stage：0 = 还没有内容（只能画启动提示）；1 = 阅读页 / 空状态提示可用。
;;     on_tick 的头一拍、EX 的重绘回调 draw_screen、open_path 的具名提示都看这一位。
;;   · splash_msg：启动提示的第二行。boot 里是通用的「正在启动…」，open_path 拿到具体
;;     文档后改写成「正在排版 xxx.md…」。
;; 注意：**不额外装一个“短启动拍”**。曾经试过 boot 装 16ms、boot_work 做完再换回 60ms，
;; 结果常规拍的相位整体漂移了几帧（宿主的周期定时器按“触发时刻 + interval”重排，见
;; app/verlet.scm 文件头），端到端用例里按帧注入的轻点就落在了两次采样之间。
;; 现在 boot 装的仍是 TICK_MS 的常规拍（与改造前逐字一致），只在回调头上多一个阶段
;; 判断：相位、节奏、set_tick 的用法全部不变。
(define boot_stage 0)
(define splash_msg "正在启动…")
;; 「正在排版 xxx.md…」这一帧只在文档够大时才画（见 open_path）：小于这个值，排版本身
;; 只几十 ms，多推一帧全屏是净亏损（掌机 SPI 推一屏约 30ms）。
(define SPLASH_MIN_BYTES 32768)

;; ---- 节拍（本 App 的每拍周期）----------------------------------------------
;; 阅读页每拍重绘整屏：**常规 60ms（≈16fps）够省电，但跟手场景不够用**——
;; 拖动与滚轮缓动的跟手上限就是节拍频率，60ms 的拍子再准也只能 16fps 地跳
;; （实测拖动每拍 6.5~19.4px 的大步）。所以两档：
;;   · 60ms（TICK_MS）——静止阅读/后台索引；
;;   · 16ms（FAST_TICK_MS，≈60Hz）——手指按住有位移、滚轮缓动未完成、
;;     以及文件选择器活动期间（它自己要求 16ms）。
;; 节拍的提升是**统一决策**的（见 pace_tick）：早期把 set_tick 散布在多处，两处
;; 各自 set_tick 会互相改回去（2026-11-20 那次选择器就因此掉到 16fps）。
;; 另外组件本身不装定时器（上游每轮事件循环对同一进程只触发一个定时器回调，
;; 同进程两个 interval 会互相错开、两拍都掉到 ~64ms，实测）。
(define TICK_MS 60)
(define tick_ms TICK_MS)    ;; 当前节拍周期（set_tick 的幂等判据，见下）
(define set_tick
  (lambda (ms)
    ;; 幂等：值没变就什么都不做。滚轮每拍都要“确认节拍”（见 handle_wheel），
    ;; 不判等会每拍 clear+set 一次定时器（无谓的堆分配，且节拍相位被反复重置）。
    (if (== ms tick_ms)
        #f
        { (System.clear_interval ts)                 ;; 安全：回调体在 fire_expired_timers
          (set! ts (System.set_interval ms on_tick)) ;; 返回之后才执行，不会 UAF
          (set! tick_ms ms) })))

;; ---- 输入状态 ------------------------------------------------------------

(define prev_down 0)          ;; 上一拍触屏电平
(define drag_last 0)          ;; 上一拍触点 y
(define drag_total 0)         ;; 本次按压累计位移（判定"拖动 or 轻点"）
(define press_x 0)
(define press_y 0)
(define key 0)
(define hit_url "")
(define dyy 0)

;; ---- 滚轮（原生通道，WASM GUI 专有）--------------------------------------
;; 背景：宿主的 GUI 滚动控件直接吃 key_event->wheel_dy（按像素落位），而灵机 App
;; 只能看触屏电平。页面过去对 App 运行态把滚轮**合成为一根手指**，对长文档有三个
;; 结构性缺陷（详见 am_native_Touch.h 的 (Touch.wheel_enable)）：合成手指走不出画布
;; （每 4 格重按一次）、重按被合进上一段手势（触点从画布顶“瞬移”回中心 = 一段凭空
;; 的反向位移）、起手点（中心 → 首格）的位移与 DOWN 同一拍而被整段丢掉。实测在
;; 320x240 / 60ms 拍下：滚 5 格只前进 56px（应 140px），继续滚还会倒回去。
;;
;; 现在：boot 里认领原生通道（wheel_enable），页面不再合成手指，而是把像素增量
;; **原样**交过来（一格 100 deltaY = 28 像素；页面不做平滑，平滑由下面 handle_wheel
;; 在本 App 自己的拍子里做）。
;; 节拍：见下方「跟手节拍」——有滚轮活动期间同样提到 FAST_TICK_MS。
;; 掌机/桌面没有滚轮源（wheel_available=#f），WHEEL_OK 恒 0，下面全部走空，
;; 行为与改造前完全一致。
(define WHEEL_OK 0)            ;; 1 = 本平台有原生滚轮通道且已认领
(define WHEEL_IDLE_TICKS 9)    ;; 连续多少拍没滚轮增量就回常规节拍（9×16ms ≈ 144ms）
(define WHEEL_GAIN 0.45)       ;; 一拍吃掉剩余量的比例（≈ GUI 滚轮通道的手感：一格 ~90ms 走完）
(define wheel_total 0)         ;; 本拍取到的滚轮增量（帧缓冲像素，正 = 内容上滚）
(define wheel_rest 0)          ;; 还没落位的剩余量（缓动用；整数值，必然归零）
(define wheel_idle 0)          ;; 连续多少拍没收到滚轮增量

;; ---- 跟手节拍（触屏拖动 / 滚轮缓动共用）------------------------------------
;; 常规拍 60ms 是为了「整屏重绘」省电（~16fps）；但**拖动跟手的上限就是拍子**：
;; 60ms 拍只能 16fps，手指移动被量化成 6~20px 的大步（实测：慢拖每拍 6.5px、
;; 快拖 19.4px，拍间隔 66.6ms）——这就是“拖动不够丝滑”的全部原因。
;; 所以：手指按住且有位移、或滚轮缓动没走完时，把节拍提到 FAST_TICK_MS（≈60Hz）；
;; 安静 DRAG_IDLE_TICKS 拍后回常规。与文件选择器的做法一致（它自己也提到 16ms），
;; 但**统一由 pace_tick 一个地方决定**——两个地方各自 set_tick 会互相改回去。
(define FAST_TICK_MS 16)       ;; 跟手期间的节拍（≈60Hz）
(define DRAG_IDLE_TICKS 15)    ;; 按住不动多少拍后放手（15×16ms ≈ 240ms，盖住拖顿一下）
(define drag_moved 0)          ;; 本拍触屏拖动有没有真的产生位移（handle_touch 置位）
(define drag_idle 0)           ;; 连续多少拍没有拖动位移

;; ---- 甩滑惯性（松手后继续滑）----------------------------------------------
;; 手感参数对齐宿主列表（app/lib/filepicker.scm 的 P_FLING_*、app/filemgr.scm）：
;;   · 速度用**近几拍的指数滑动平均**估，不是整段手势的平均速度——缓慢停下来的拖动
;;     不该再甩出去；松手前 120ms 内没动过就干脆不甩；
;;   · 按下即刹车（打断惯性），且这一下松手不算点按（drag_total 已保证）；
;;   · 惯性期间节拍保持 16ms（见 pace_tick），撞到上/下边界立即停（不做回弹）。
;; 速度单位是 **px/ms**（与节拍无关）：每拍推进量 = v × 实测 dt，浏览器掉帧不会让
;; 惯性变慢或变远（这就是 verlet 那条"别把物理绑在拍子上"的同一条教训）。
(define FLING_MIN_V 0.20)      ;; 松手速度低于它不甩（px/ms ≈ 320px/s ≈ filepicker 的 5px/16ms）
(define FLING_VMAX 2.8)        ;; 初速上限（px/ms ≈ 45px/16ms，同 filepicker 的 VMAX）
(define FLING_VMIN 0.05)       ;; 衰减到它以下就停（px/ms ≈ 0.8px/16ms）
(define FLING_DECAY 0.88)      ;; 每 16ms 的速度保留系数（同 filepicker 的 FRICTION）
(define FLING_STALE_MS 120)    ;; 松手前这么久没动过 → 不甩
(define FLING_EMA 0.4)         ;; 速度 EMA 的新样本权重（≈ 4 拍记忆，~64ms）
(define fling_v 0)             ;; 当前惯性速度（px/ms，正 = 内容上滚）
(define fling_acc 0)           ;; 未落位的亚像素（让慢速长距离推进不丢余量）
(define fling_t 0)             ;; 上一拍推进的时刻（System.timestamp）
(define drag_v 0)              ;; 拖动速度 EMA（px/ms，正 = 内容上滚）
(define drag_vt 0)             ;; 上一次**真实位移**的时刻（速度 EMA 的 dt 基准 + 甩滑新鲜度）

;; ---- 小工具 --------------------------------------------------------------

(define new_list (lambda () (cdr '(0))))
(define mk4 (lambda (a b c d) (cons a (cons b (cons c (cons d (cdr '(0))))))))
(define mk7 (lambda (a b c d e f g)
  (cons a (cons b (cons c (cons d (cons e (cons f (cons g (cdr '(0)))))))))))

;; ---- 几何：可视高度 / 最大滚动 / 收口 ------------------------------------

(define view_h
  (lambda ()
    (- (- SH TOP_H) BOT_H)))

;; 滚动位置由 md_view 的锚点 (段号, 段内 dy) 决定；这里只做"读快照"与"刷新全局量"。
;; 统一的滚动入口：流式文档交给 md_view（跨界自动按需布局），
;; 提示文案按像素直接夹取（内容极小，无分段）。
(define scroll_by_px
  (lambda (px)
    (if (== MDV_ok 1)
        { (MDV.scroll_by px (view_h))
          (sync_scroll) }
        { (set! scroll (+ scroll px))
          (if (< scroll 0) (set! scroll 0) #f)
          (if (> scroll (max_scroll)) (set! scroll (max_scroll)) #f)
          (sync_scroll) })
    #t))

(define sync_scroll
  (lambda ()
    (set! scroll (if (== MDV_ok 1) (MDV.top_y) scroll))
    (set! content_h (if (== MDV_ok 1) (MDV.total_height) notice_h))
    #t))

(define max_scroll
  (lambda ()
    (define m (- (if (== MDV_ok 1) (MDV.total_height) notice_h) (view_h)))
    (if (< m 0) 0 m)))

(define clamp_scroll
  (lambda ()
    (sync_scroll)
    #t))

;; ---- 上次打开的文档（data/md-last.txt）----------------------------------
;;
;; 记录「上次成功打开的文档路径」，供下次从主菜单冷启动时直接续读。要点：
;;   · 只记**成功的打开**（读失败/超上限不覆盖旧记录，免得一次误触把记忆抹掉）；
;;   · 路径**原样存取**：可能来自文件选择器（资源根相对路径 "doc/a.md"），也可能
;;     来自别的 App 的启动参数（宿主绝对路径 "/home/bd4sur/linglong/doc/a.md"）；
;;     (Fs.exists) 两种写法都认，所以不需要换算/拼接；
;;   · 写失败（只读文件系统）只影响「下次能不能续读」，不影响本次阅读；
;;   · 读回时去掉首尾空白——用户可能用编辑器打开/改过这个文件。

(define is_space
  (lambda (c) (or (== c 32) (== c 9) (== c 10) (== c 13))))

(define trim_ws
  (lambda (s)
    (define n (String.length s))
    (define b 0)
    (define e n)
    (define stop 0)
    (while (and (< b e) (== stop 0))
      { (if (is_space (String.charCodeAt b s)) (set! b (+ b 1)) (set! stop 1)) })
    (set! stop 0)
    (while (and (> e b) (== stop 0))
      { (if (is_space (String.charCodeAt (- e 1) s)) (set! e (- e 1)) (set! stop 1)) })
    (String.slice s b e)))

;; 读回上次的路径；没有记录/读失败 → 空串
(define last_path_load
  (lambda ()
    (define s (Fs.read LAST_PATH))
    (if (null? s) "" (trim_ws s))))

;; 覆盖写上次的路径（空串不写：没打开过任何文档时不该制造一条空记录）
(define last_path_save
  (lambda (p)
    (if (> (String.length p) 0)
        { (Fs.write LAST_PATH p)
          (display "MDLAST save ") (display p) (newline) }
        #f)
    #t))

;; 把提示文案做成一个段落块（复用同一条解析路径，免得手工构造 AST）
(define notice_blocks
  (lambda (line1 line2)
    (MD.parse_lines (cons line1 (cons "" (cons line2 (cdr '(0))))))))

;; ---- 图片支持（方案 A：宿主解码；方案 D：取不到时画虚线框兜底）----------
;;
;; 分工（详见 doc/Markdown插图渲染可行性调研.md）：
;;   · 解析端只产出 (image 路径 alt)，不碰文件；
;;   · 排版端通过下面的 (doc_image_size) 度量回调拿原始尺寸，按正文宽等比缩放，
;;     产出一行"高度 = 图片高"的显示行（段样式 ST_IMAGE，段里带路径与 alt）；
;;   · 绘制端只调 (Gfx.draw_image ...) 让宿主解码+缩放+落屏——像素永不进解释器堆；
;;     返回 #f（文件缺失/格式不支持）时画虚线框 + alt 文本兜底。
;;
;; 路径规则：Markdown 习惯是"以文档所在目录为基准"。文档路径形如 "doc/x.md" 时
;; 基准是 "doc/"，根目录文档基准是空串；Gfx 的路径规则与 Fs 一致（相对资源根）。
(define doc_dir "")     ;; 当前文档所在目录（含结尾 '/'；根目录为空串）

(define set_doc_dir
  (lambda (path)
    (define n (String.length path))
    (define i 0)
    (define cut -1)
    (while (< i n)
      { (if (== (String.charCodeAt i path) 47) (set! cut i) #f)   ;; '/'
        (set! i (+ i 1)) })
    (if (< cut 0) (set! doc_dir "") (set! doc_dir (String.slice path 0 (+ cut 1))))))

;; Markdown 里的图片引用 → 资源根相对路径。
;; 查找顺序：① 以 '/' 开头 → 宿主绝对路径，原样交给 Gfx；② 相对**文档所在目录**；
;; ③ 回退到**资源根**（`icon/xxx.png` 这类公共资源常放在根下，而文档在 doc/ 里，
;;    又不能写 "../icon/..."——Fs/Gfx 的路径规则明确拒绝 ".."）。
(define image_path
  (lambda (p)
    (define r "")
    (if (== (String.length p) 0) p
        (if (== (String.charCodeAt 0 p) 47) p
            { (set! r (String.concat doc_dir p))
              (if (Fs.exists r) r
                  (if (Fs.exists p) p r)) }))))

;; 排版期的度量回调：(路径) → (宽 高) | #f
;; 注意 (Gfx.image_size ...) 失败时返回 #null，而 #null 在本解释器里是**真值**，
;; 这里统一归一成 #f，排版端只需判真假/类型即可（见 md_layout 的 lay_image）。
(define doc_image_size
  (lambda (p)
    (define r (Gfx.image_size (image_path p)))
    (if (null? r) #f r)))

;; ---- 数学公式（复用 app/lib/tex_*：LaTeX → 语法树 → 盒子 → op 流）----------
;;
;; 与数学公式 App（app/mathjax.scm）同一套引擎，只换输入源：
;;   · 解析：md_parse 把 `$$...$$` 解析成块级 (math 源串)、把 `$...$` 解析成行内
;;     (imath 源串)（本文件不碰 Markdown 语法）；
;;   · 度量：md_layout 调下面的 (doc_math_measure 源串 最大宽 最大高) 拿
;;     (宽 上高 下深 em)：块级公式按正文宽居中、**超宽就等比缩小 em 重排**（不是位图
;;     缩放，字形仍落在字模的清晰尺度上）；行内公式额外受行高约束（maxh，见
;;     md_layout 的 IMATH_ASC/DSC），保证它落在一行文字里不撑破段落；
;;   · 绘制：取缓存的 op 流直接画，**不重排**——滚动/翻页时每帧只做"换缓冲 + 一遍
;;     顺序绘制"，否则每帧重排公式会明显掉帧。
;;
;; 缓存键 = (源串 em)：同一段 LaTeX 在不同宽度/行高约束下会有不同的 em（块级按页宽、
;; 行内按行高），键里带 em 才不会张冠李戴；绘制端拿到段里的 em 直接命中同一份 op 流。
;; 行内公式一条屏幕可能同时出现十几条，缓存上限因此调到 48（每条 ~1-2KB op）。
;;
;; 缓存条目 = (源串 em 宽 上高 下深 op流 操作数)。tex_base 的操作缓冲是模块级单例，
;; 绘制前用 (B.set_ops 缓存流 操作数) 换回去；缓存的流此后只读（排版器每次 layout
;; 都先 reset，不会原地改旧流）。
(define MATH_CACHE_MAX 48)
(define MATH_MIN_EM 0.20)     ;; 超长公式的最小缩放（再小已不可读，超出部分靠裁剪）
(define math_cache (cdr '(0)))  ;; (源串 em 宽 上高 下深 op流 操作数)，最近加入的在前

;; 按 (源串 em) 找缓存条目；em < 0 表示"只要源串匹配的那一条"（诊断/兼容用）
(define math_cache_find
  (lambda (src em)
    (define n (length math_cache))
    (define i 0)
    (define e #f)
    (define r #f)
    (while (and (< i n) (not r))
      { (set! e (get_item math_cache i))
        (if (String.equals (get_item e 0) src)
            (if (< em 0) (set! r e)
                (if (< (B.absf (- (get_item e 1) em)) 0.0005) (set! r e) #f))
            #f)
        (set! i (+ i 1)) })
    r))

(define math_cache_clear (lambda () (set! math_cache (cdr '(0))) #t))

;; 插入一条（尾部是最久未用的，满了就丢尾部）
(define math_cache_put
  (lambda (e)
    (define out (new_list))
    (define n (length math_cache))
    (define i 0)
    (while (and (< i n) (< i (- MATH_CACHE_MAX 1)))
      { (push out (get_item math_cache i))
        (set! i (+ i 1)) })
    (set! math_cache (cons e out))
    #t))

;; 以指定 em 排一条公式并缓存
(define math_build_em
  (lambda (src em)
    (define box #f)
    (define w 0.0)
    (define h 0.0)
    (define d 0.0)
    (define e #f)
    (L.set_emul em)
    (set! box (L.layout (P.parse src) 0))
    (set! w (get_item box 0))
    (set! h (get_item box 1))
    (set! d (get_item box 2))
    (set! e (mk7 src em w h d B.ops (B.count)))
    (math_cache_put e)
    e))

;; 排一条公式并满足约束：先按 em=1 排，再按 maxw/maxh 等比缩小 em **重排**。
;;   · maxw 是宽度上限（块级 = 正文宽；行内 = 本块正文宽）；
;;   · maxh > 0 是高度上限（行内 = 一行文字能容的上下空间；块级传 0 = 不限）。
;; 度量取整会让缩放不严格线性，因此最多补 3 轮迭代（常规一~两轮就到位）。
(define math_build
  (lambda (src maxw maxh)
    (define e #f)
    (define w 0.0)
    (define h 0.0)
    (define d 0.0)
    (define em 1.0)
    (define k 0.0)
    (define round 0)
    (set! e (math_build_em src em))
    (set! w (get_item e 2))
    (set! h (get_item e 3))
    (set! d (get_item e 4))
    (while (and (or (> w maxw) (and (> maxh 0.0) (> (+ h d) maxh))) (< round 3))
      { (set! k 1.0)
        (if (> w maxw) (set! k (B.mn k (/ maxw w))) #f)
        (if (and (> maxh 0.0) (> (+ h d) maxh))
            (set! k (B.mn k (/ maxh (+ h d))))
            #f)
        (set! em (* em k))
        (if (< em MATH_MIN_EM) (set! em MATH_MIN_EM) #f)
        (set! e (math_build_em src em))
        (set! w (get_item e 2))
        (set! h (get_item e 3))
        (set! d (get_item e 4))
        (set! round (+ round 1)) })
    ;; 诊断行（端到端测试据此断言"公式真的排了、长公式被缩放"）。
    ;; 刻意**不打印公式源码**：LaTeX 里的 \r \n \t 在 (display) 时会被解释成控制
    ;; 字符（实测 \right 会输出一个 CR，把日志行切碎、带歪端到端断言），只打长度。
    (display "MDMATH em=") (display em)
    (display " w=") (display w)
    (display " h=") (display h)
    (display " maxw=") (display maxw)
    (display " len=") (display (String.length src)) (newline)
    e))

;; 缓存里找一条**已满足约束**的条目（源串相同、宽不超、高不超）；没有返回 #f
(define math_find_fit
  (lambda (src maxw maxh)
    (define n (length math_cache))
    (define i 0)
    (define e #f)
    (define r #f)
    (define ok #f)
    (while (and (< i n) (not r))
      { (set! e (get_item math_cache i))
        (if (String.equals (get_item e 0) src)
            { (set! ok (<= (get_item e 2) maxw))
              (if (and ok (> maxh 0.0))
                  (if (> (+ (get_item e 3) (get_item e 4)) maxh) (set! ok #f) #f)
                  #f)
              (if ok (set! r e) #f) }
            #f)
        (set! i (+ i 1)) })
    r))

;; 取（或现排）一条满足约束的公式缓存条目
(define math_entry
  (lambda (src maxw maxh)
    (define e (math_find_fit src maxw maxh))
    (if e e (math_build src maxw maxh))))

;; 注入给 md_layout 的度量回调：(源串 最大宽 最大高) → (宽 上高 下深 em) | #f
(define doc_math_measure
  (lambda (src maxw maxh)
    (define hh (if (undefined? maxh) 0 maxh))
    (define e (math_entry src maxw hh))
    (if e (mk4 (get_item e 2) (get_item e 3) (get_item e 4) (get_item e 1)) #f)))

;; 画一条行内公式（绘制端已把 cr/cg/cb 按 ST_IMATH 设好）。
;; 基线跟**同一行的文字基线**对齐（y 是行顶）；裁剪矩形就是公式盒本身。
;; 段里带的 em 就是排版时用的那个 em，因此这里是纯缓存命中（缓存被挤掉才重排）。
(define draw_imath
  (lambda (src x y em)
    (define e (math_cache_find src em))
    (define w 0)
    (define h 0)
    (define d 0)
    (define base 0)
    (if e #f (set! e (math_build_em src em)))
    (if e
        { (set! w (Math.ceil (get_item e 2)))
          (set! h (get_item e 3))
          (set! d (get_item e 4))
          (set! base (+ y (+ text_pad font_base)))
          ;; 盒子顶边 = 基线 - 上高（y 向下为正）；裁剪矩形略放宽 2px 容抗锯齿笔画
          (Gfx.set_clip (- x 2) (- (- base h) 2) (+ w 4) (+ (+ h d) 4))
          (D.set_view (- x 2) (- (- base h) 2) (+ w 4) (+ (+ h d) 4))
          (B.set_ops (get_item e 5) (get_item e 6))
          (D.draw x base cr cg cb)
          (Gfx.reset_clip) }
        #f)
    #t))

;; 画一条块级公式。
;; 裁剪矩形 = 公式行（左右各留 2px 给抗锯齿线段），与 mathjax 的视口裁剪同一机制。
(define draw_math
  (lambda (src x y h)
    (define e (math_entry src CW 0))
    (define bw 0)
    (define bh 0)
    (if e
        { (set! bw (Math.ceil (get_item e 2)))
          (set! bh (get_item e 3))
          (Gfx.set_clip (- x 2) y (+ bw 4) h)
          (D.set_view (- x 2) y (+ bw 4) h)
          (B.set_ops (get_item e 5) (get_item e 6))
          (D.draw x (+ y (+ MDL.MATH_PAD bh)) cr cg cb)
          (Gfx.reset_clip) }
        #f)
    #t))

;; 虚线框兜底（方案 D）：沿矩形边画 3px 亮 / 3px 暗的短线段 + alt 文本
(define draw_dashed_rect
  (lambda (x y w h r g bl)
    (define k 0)
    (define on 1)
    (while (< k w)
      { (if (== on 1)
            { (Gfx.fill_rect (+ x k) y 3 1 r g bl)
              (Gfx.fill_rect (+ x k) (- (+ y h) 1) 3 1 r g bl) }
            #f)
        (set! on (if (== on 1) 0 1))
        (set! k (+ k 3)) })
    (set! k 0)
    (set! on 1)
    (while (< k h)
      { (if (== on 1)
            { (Gfx.fill_rect x (+ y k) 1 3 r g bl)
              (Gfx.fill_rect (- (+ x w) 1) (+ y k) 1 3 r g bl) }
            #f)
        (set! on (if (== on 1) 0 1))
        (set! k (+ k 3)) })
    #t))

(define draw_image_missing
  (lambda (x y w h alt)
    (draw_dashed_rect x y w h 92 104 128)
    (Gfx.draw_text (MDL.clip_text (if (> (String.length alt) 0)
                                       (String.concat "[图片] " alt)
                                       "[图片缺失]")
                                   (- w 8))
                   (+ x 4) (+ y (Math.floor (/ (- h 14) 2)))
                   150 162 186)
    #t))

;; ---- 打开文档：读文本 → 解析 → 排版 --------------------------------------

;; 按**路径**打开（冷启动续读、文件选择器、被别的 App 拉起，三条路共用这一条链路）。
;;
;; 两种显示模式：
;;   stream（正常）：一切交给 md_view —— 打开时只切出并布局**第一段**（首屏 ~60-80ms），
;;                   其余段在后台按拍建索引，滚动时按需布局（见 on_tick / handle_touch）。
;;   notice（兜底）：文件读不出来/超上限时，把两条提示排成一个小文档（整篇排版，
;;                   内容只有两段），仍用同一套绘制代码。
(define open_path
  (lambda (path title)
    (define fsize 0)
    (define ok #f)
    (define msg #f)
    ;; 换文档：旧公式缓存的 op 流不再需要。必须在 MDV.open 之前清——
    ;; 打开时会立刻排版首段（公式在那一刻就会排好并进缓存）。
    (math_cache_clear)
    (set! cur_title title)
    (set! cur_path path)
    (set_doc_dir path)
    (set! fsize (Fs.size path))
    (if (null? fsize)
        (set! msg (notice_blocks "读取失败：文件不存在或不可读。"
                                 "点底栏「打开」重新选一篇文档。"))
        (if (> fsize DOC_MAX)
            (set! msg (notice_blocks
                        (String.concat "文档过大（"
                                       (String.concat (String.atom_to_string fsize) " 字节）："))
                        "本阅读器上限 1MB。"))
            { ;; 首帧安抚：告诉用户正在排的是哪一篇。只有**大文档**才画：
              ;;   · 掌机上这一帧落在 MDV.open（首段解析 + 排版）之前，正好把等待时间显示满；
              ;;   · 小文档排版本身只几十 ms，多推一帧全屏反而白花一次 SPI（320×240 ≈ 30ms）；
              ;;   · WASM 上这帧看不见（栈不退空不合成），但只有 5 个 Gfx 调用，可忽略。
              (if (and (== boot_stage 0) (> fsize SPLASH_MIN_BYTES))
                  { (set! splash_msg (String.concat "正在排版 "
                                                    (String.concat (FSX.leaf path) "…")))
                    (draw_splash) }
                  #f)
              (set! ok (MDV.open path CW doc_image_size SEG_TARGET CACHE_LINES))
              (if ok #f (set! msg (notice_blocks "读取失败。" "文件可能已被删除或不可读。"))) }))
    (open_finish ok msg)
    #t))

;; 空状态：没有任何可续读的文档时的提示页（由用户自己挑一篇）。
;; 复用 open_finish 的 notice 路径（整篇排版），所以点按/键盘的落点与错误提示一致：
;; 底栏「打开」、点按正文、Enter 三条路都通向文件选择器。
(define open_prompt
  (lambda ()
    (set! cur_title "Markdown 阅读器")
    (set! cur_path "")
    (set_doc_dir "")
    (open_finish #f (notice_blocks "还没有打开文档。"
                                   "点底栏「打开」选一篇 .md / .markdown；也可点按本页或按 Enter。"))
    (display "MDEMPTY") (newline)
    #t))

;; 冷启动（无启动参数）：续读上次的文档；记录缺失/文件已不在 → 空状态提示。
(define boot_open
  (lambda ()
    (define p (last_path_load))
    (if (and (> (String.length p) 0) (Fs.exists p))
        { (display "MDLAST resume ") (display p) (newline)
          (open_path p (FSX.strip_ext (FSX.leaf p))) }
        { (display "MDLAST none ") (display p) (newline)
          (open_prompt) })
    #t))

;; 打开完成后的公共收尾（正常/兜底两条路共用）
(define open_finish
  (lambda (ok msg)
    (set! img_seen -1)
    (set! toc_items (cdr '(0)))
    (set! toc_open 0)
    (set! toc_scanned 0)
    (set! scroll 0)
    (if ok
        { (set! MDV_ok 1)
          (set! notice_lines (cdr '(0)))
          (set! notice_h 0)
          (index_work IDX_BYTES)         ;; 开屏后立刻推进一点索引（目录/估算随之更新）
          (set! toc_gen_seen -1)
          ;; 成功打开才记住路径：失败（缺文件/超上限）不应抹掉上次的好记录
          (last_path_save cur_path) }
        { (set! MDV_ok 0)
          (set! notice_lines (MDL.layout msg CW doc_image_size))
          (set! notice_h (MDL.total_height notice_lines)) })
    (sync_scroll)
    ;; 诊断行（宿主 stdout / 浏览器控制台；端到端测试据此断言解析与排版规模）
    (display "MDSTAT blocks=")
    (display (if (== MDV_ok 1) (MDV.blocks_total) 1))
    (display " lines=")
    (display (if (== MDV_ok 1) (MDV.lines_total) (length notice_lines)))
    (display " images=")
    (display (if (== MDV_ok 1) (MDV.images) 0))
    (display " height=")
    (display content_h)
    (display " title=")
    (display cur_title)
    (display " seg=")
    (display (if (== MDV_ok 1) (MDV.count) 0))
    ;; 内存/分段诊断（工具链与回归测试用于守"不涨池也能读大文件"的预算）
    (display " heap=")
    (display (get_item (System.memstat) 3))
    (display " cache=")
    (display (if (== MDV_ok 1) (MDV.cache_lines) 0))
    (newline)
    #t))

;; ---- 文件选择器（可复用组件 app/lib/filepicker.scm）---------------------
;; 用户挑文档的唯一入口：在整棵资源树里浏览，选中即按上面的同一条链路打开。
;; 活动期间**不调 (EX.check)**——返回请求归选择器消费（= 取消选择）。

(define OPEN_BTN_W 44)
(define OPEN_BTN_H 14)

(define open_btn_x (lambda () (- SW (+ OPEN_BTN_W 6))))
(define open_btn_y (lambda () (- SH 17)))

(define hit_open_btn
  (lambda (px py)
    (if (and (>= px (open_btn_x)) (< px (+ (open_btn_x) OPEN_BTN_W)))
        (if (and (>= py (open_btn_y)) (< py (+ (open_btn_y) OPEN_BTN_H))) 1 0)
        0)))

(define draw_open_btn
  (lambda ()
    (Gfx.fill_rect (open_btn_x) (open_btn_y) OPEN_BTN_W OPEN_BTN_H 52 58 76)
    (Gfx.draw_rect (open_btn_x) (open_btn_y) OPEN_BTN_W OPEN_BTN_H 110 118 140)
    (Gfx.draw_text "打开" (+ (open_btn_x) 7) (+ (open_btn_y) 1) 225 228 235)
    #t))

(define start_picker
  (lambda ()
    { (display "MDOPEN picker") (newline)
      (FP.start "打开 Markdown 文件" "md,markdown")
      ;; 选择器活动期间提速到 60Hz（拖动跟手，见 TICK_MS 处注释）
      (set_tick (FP.tick_ms)) }))

;; 选择器每拍：0 进行中 / 1 已选中 / 2 已取消
(define tick_picker
  (lambda ()
    (define r 0)
    (define picked "")
    (set! r (FP.step))
    (if (== r 1)
        { (set! picked (FP.result))
          (FP.reset)
          (set_tick TICK_MS)          ;; 选择器结束：回到常规节拍
          (display "MDOPEN picked ") (display picked) (newline)
          (open_path picked (FSX.strip_ext (FSX.leaf picked))) }
        #f)
    (if (== r 2)
        { (FP.reset)
          (set_tick TICK_MS)
          (draw_all) }
        #f)
    #t))

;; ---- 目录（TOC）---------------------------------------------------------
;; 策略与主流低资源长文本阅读器一致：**快速打开 → 流式渲染 → 后台异步扫全文建目录**。
;;
;;   · 打开时只切/排第一段（~90ms），用户立刻可读；与此同时 `bg_work` 每拍用
;;     IDX_BYTES 的预算往后切段，**切段时顺手提取该段标题**（md_view.scan 里的
;;     headings_of：只判行首 '#'，跳过围栏与引用块，不解析不排版、不额外读盘）
;;     ⇒ 目录是索引的副产品，1MB 文档约 5s 就长齐（实测 153 → 2221 条）。
;;   · 段真正被排版时，md_view 会用**布局给出的精确 y** 覆盖该段的廉价条目。
;;   · 面板打开期间索引继续跑（IDX_BYTES_TOC 放大预算，面板渲染很轻），并用
;;     目录世代号 `(MDV.toc_gen)` 做**增量刷新**——只有条目真的变了才重建列表，
;;     不每拍 O(条目数) 重算；条目只追加在尾部，选中项下标不会失效。
;;   · 标题栏在未扫完时显示 "n 条 · x%"；跳转 = 设锚点 → 布局该段 → 用标题文本
;;     把估算 dy 精修到精确 y（见 md_view.toc_jump）。
;;
;; 面板是全屏覆盖式：自己消费输入与推屏；活动期间**不调 (EX.check)**，
;; 返回请求（硬 A/ESC）改为"关闭面板"（与文件选择器同一思路）。

(define toc_items (cdr '(0)))   ;; 当前文档目录：(层级 段号 种类 数值 文本) 的列表
(define toc_open 0)             ;; 1 = 面板打开
(define TOC_TOP 26)             ;; 条目区起始 y（页眉 18 + 8）
(define MENU_ROW_K 1.6)         ;; 菜单项高度 = 字号行高 × 1.6（触屏点按更好按）

;; 当前字模下菜单项的统一样式参数（行高 / 墨高 / 顶部内缩）
(define menu_row_h (lambda () (Math.floor (* line_h MENU_ROW_K))))
(define menu_ink_h (lambda () (if (== font_id 2) 16 12)))
(define menu_ink_pad (lambda () text_pad))
(define menu_ink_gap (lambda () text_gap))

(define menu_metrics
  (lambda (info)
    ;; info = (Gfx.font_info id) → (半角 全角 行高 基线)
    (ML.metrics (get_item info 0) (get_item info 1) (menu_ink_h) (menu_ink_pad) (menu_ink_gap))
    #t))
(define TOC_BTN_W 40)
(define TOC_BTN_H 14)

(define toc_btn_x (lambda () (- (open_btn_x) (+ TOC_BTN_W 4))))
(define toc_btn_y (lambda () (- SH 17)))

(define hit_toc_btn
  (lambda (px py)
    (if (and (>= px (toc_btn_x)) (< px (+ (toc_btn_x) TOC_BTN_W)))
        (if (and (>= py (toc_btn_y)) (< py (+ (toc_btn_y) TOC_BTN_H))) 1 0)
        0)))

(define draw_toc_btn
  (lambda ()
    (Gfx.fill_rect (toc_btn_x) (toc_btn_y) TOC_BTN_W TOC_BTN_H 52 58 76)
    (Gfx.draw_rect (toc_btn_x) (toc_btn_y) TOC_BTN_W TOC_BTN_H 110 118 140)
    (Gfx.draw_text "目录" (+ (toc_btn_x) 7) (+ (toc_btn_y) 1) 225 228 235)
    #t))

(define toc_refresh
  (lambda ()
    (set! toc_items (MDV.toc_list))
    #t))

;; 目录条目 → 通用菜单条目（样式全在这里配：层级颜色 / 缩进 / 高亮态颜色）
;;   一级最亮、二三级渐暗；缩进按层级；被高亮时统一用更亮一档的同色系。
(define toc_menu_items
  (lambda ()
    (define n (length toc_items))
    (define i 0)
    (define it #f)
    (define lvl 0)
    (define out (new_list))
    (define txt "")
    (while (< i n)
      { (set! it (get_item toc_items i))
        (set! lvl (get_item it 0))
        (set! txt (get_item it 4))
        (if (== lvl 1)
            (push out (ML.item txt 238 244 252 (- lvl 1) 0 255 255 255))
            (if (== lvl 2)
                (push out (ML.item txt 206 218 236 (- lvl 1) 0 236 244 254))
                (push out (ML.item txt 168 182 204 (- lvl 1) 0 210 224 244))))
        (set! i (+ i 1)) })
    out))

;; 打开面板：选中项默认落在"当前阅读位置之前最近的标题"上。
;; 条目形状 = (层级 段号 种类 数值 文本)；种类 0 = 段内精确 y，1 = 段内码点偏移（估算）。
(define toc_open_panel
  (lambda ()
    (define n 0)
    (define i 0)
    (define it #f)
    (define seg 0)
    (define a (MDV.anchor))
    (define aseg (get_item a 0))
    (define ady (get_item a 1))
    (define near 0)
    (toc_refresh)
    (set! toc_gen_seen (MDV.toc_gen))
    (set! n (length toc_items))
    (set! i 0)
    (while (< i n)
      { (set! it (get_item toc_items i))
        (set! seg (get_item it 1))
        (if (< seg aseg)
            (set! near i)
            (if (and (== seg aseg) (== (get_item it 2) 0) (<= (get_item it 3) ady))
                (set! near i)
                #f))
        (set! i (+ i 1)) })
    ;; 交给通用菜单组件：条目样式 + 行高（字号行高 × 1.6）+ 视口（页眉与页脚之间）
    (ML.reset (toc_menu_items) (menu_row_h))
    (ML.geom 0 TOC_TOP SW (- (- SH BOT_H) TOC_TOP))
    (ML.back_top_right SW)          ;; 菜单自己的返回键（右上角，与阅读页「退出」同位）
    (App.set_exit_button 0 0 0 0)   ;; 停用宿主「退出」热区（右上角归本面板，避免误触全局退出）
    (ML.set_sel near)
    (set! toc_open 1)
    (display "MDTOC n=") (display n) (newline)
    #t))

(define toc_close
  (lambda ()
    (set! toc_open 0)
    (set! exitbtn_mute 4)      ;; 见 exitbtn_mute 注释
    #t))

;; 跳到第 i 条（视图层负责：设锚点 → 布局该段 → 用标题文本精修到精确 y）
(define toc_jump
  (lambda (i)
    (define n (length toc_items))
    (if (and (>= i 0) (< i n))
        { (MDV.toc_jump i)
          (sync_scroll)
          (display "MDJUMP i=") (display i) (newline)
          (set! toc_open 0) }
        #f)
    #t))

;; 面板绘制：底色 → **列表（组件绘制，含拖动滚动/高亮/滚动条）** → 页眉页脚盖住边界半行
(define draw_toc_panel
  (lambda ()
    (define n (length toc_items))
    (Gfx.fill_rect 0 0 SW SH 16 19 26)
    (ML.draw)
    (if (== n 0)
        (Gfx.draw_text (if (MDV.toc_done) "（本文没有 1~3 级标题）" "（正在扫描全文标题…）")
                       PAD_X TOC_TOP 150 162 186)
        #f)
    ;; 页眉：标题 + 条目数/扫描进度（左对齐排在标题后，右侧留给返回键）
    (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
    (Gfx.draw_text "目录" PAD_X 3 236 242 250)
    (if (not (MDV.toc_done))
        (Gfx.draw_text (String.concat (String.atom_to_string n)
                                      (String.concat " 条 · "
                                        (String.concat (String.atom_to_string (MDV.index_progress)) "%")))
                       38 3 150 168 190)
        (Gfx.draw_text (String.concat (String.atom_to_string n) " 条") 38 3 150 168 190))
    (ML.draw_back)
    (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
    (Gfx.draw_text "拖动/↑↓ 选择 · 点按跳转 · 右上返回" PAD_X (- SH 14) 130 140 155)
    (Gfx.refresh)
    #t))

(define apply_key_toc
  (lambda (k)
    (if (== k 1) (ML.move (- 0 1)) #f)
    (if (== k 2) (ML.move 1) #f)
    (if (== k 3) (ML.page (- 0 1)) #f)
    (if (== k 4) (ML.page 1) #f)
    (if (== k 13) (toc_jump (ML.sel_of)) #f)
    (if (== k 27) (toc_close) #f)
    #t))

(define tick_toc
  (lambda ()
    (define hit 0)
    { ;; 返回请求归面板消费（= 关闭面板），不交给 EX 去弹退出确认
      (if (App.back_requested) (toc_close) #f)
      (set! key (Key.take_press))
      (while (not (== key 0))
        { (apply_key_toc key)
          (set! key (Key.take_press)) })
      ;; 触屏：拖动 = 滑动滚动（1:1 跟手）；轻点条目 = 选中并跳转；
      ;; 轻点页眉/页脚区 = 关闭（组件只报"视口内的条目"，区外由这里处理）
      (set! hit (ML.step_touch))
      (if (>= hit 0)
          (toc_jump hit)
          (if (== hit -2)
              (if (== (ML.hit_back (ML.tap_x) (ML.tap_y)) 1) (toc_close) #f)  ;; 右上返回键
              #f))
      ;; 面板打开期间**索引继续跑**（预算放大）：全文目录几秒内补齐，
      ;; 条目只会追加在尾部，选中项下标不会失效。
      (index_work IDX_BYTES_TOC)
      ;; 只在"目录世代号变化"时重建条目表（避免每拍 O(条目数) 重建）
      (if (not (== (MDV.toc_gen) toc_gen_seen))
          { (set! toc_gen_seen (MDV.toc_gen))
            (toc_refresh)
            (set! toc_scanned (MDV.toc_scanned_count))
            ;; 增量换表：组件保留滚动位置与选中项（条目只追加在尾部）
            (ML.set_items (toc_menu_items))
            (ML.ensure_sel) }
          #f)
      (draw_toc_panel) }))

;; ---- 设置（模态框）-------------------------------------------------------
;; 第一个设置项：字体。系统有三套字模左右可用——
;;   0 = 12px 二值点阵（锐利、无灰阶）、1 = 12px 抗锯齿（默认）、2 = 16px 抗锯齿（更大更平滑）。
;; 字模决定**行高与字符步进**，所以换字体后必须：
;;   ① (Gfx.set_font id) 让绘制端换字模；
;;   ② 用 (Gfx.font_info id) 拿到 (半角 全角 行高 基线)，喂给 (MDL.set_font_metrics ...)；
;;   ③ 重新排版当前文档（折行位置、行高、表格列宽、目录 y 全都会变）。
;; 选择写入 data/md-font.txt，下次启动沿用。

(define FONT_PATH "data/md-font.txt")
(define font_id 1)       ;; 当前字模
(define font_saved -1)   ;; 已经落到 data/md-font.txt 里的 id（-1 = 还不知道）；见 apply_font
(define line_h 15)       ;; 当前行高（由字模决定）
(define font_base 12)    ;; 基线（行顶到基线）
(define text_pad 1)      ;; 文字框顶相对行框顶的偏移（点阵 0 / 抗锯齿 1）
;; 字模的「行顶 → 墨迹」空隙（点阵 0 / 12px 抗锯齿 2 / 16px 抗锯齿 1）。
;; 本地库现在把 (Gfx.draw_text) 的 y 当**墨迹顶**（见 am_native_Gfx.c 的 native_text_top_gap），
;; 所以：① 行内装饰（链接/三级标题的下划线）要按墨迹的位置**上移**这段空隙，
;;       ② 菜单组件（目录/设置）的行框顶要**加回**这段空隙（否则整列文字偏高）。
(define text_gap 2)
(define pill_h 13)       ;; 行内代码药丸底高
(define IMATH_PAD 6)     ;; 行内公式允许比文字盒上下各多占的像素（分式/根号靠它不变形）

(define SET_TOP 30)
(define settings_open 0)
(define SET_BTN_W 40)

(define set_btn_x (lambda () (- (toc_btn_x) (+ SET_BTN_W 4))))
(define set_btn_y (lambda () (- SH 17)))

(define hit_set_btn
  (lambda (px py)
    (if (and (>= px (set_btn_x)) (< px (+ (set_btn_x) SET_BTN_W)))
        (if (and (>= py (set_btn_y)) (< py (+ (set_btn_y) 14))) 1 0)
        0)))

(define font_name
  (lambda (id)
    (if (== id 0) "点阵 12px"
        (if (== id 1) "抗锯齿 12px"
            (if (== id 2) "抗锯齿 16px" "?")))))

;; 按字模 id 应用：查度量 → 设绘制字模 → 设排版度量 → 重新排版当前文档
(define apply_font
  (lambda (id)
    (define info (Gfx.font_info id))
    (define keep 0)
    (define old_h line_h)
    (define keep_a #f)
    (define keep_ratio 1)
    (if info
        { (set! font_id id)
          (Gfx.set_font id)
          (set! text_pad (if (== id 0) 0 1))
          (set! text_gap (if (== id 0) 0 (if (== id 1) 2 1)))
          (set! font_base (get_item info 3))
          (set! line_h (get_item info 2))
          (set! pill_h (- line_h 2))
          (MDL.set_font_metrics (get_item info 0) (get_item info 1) (get_item info 2))
          ;; 行内公式的纵向预算：文字基线以上/以下各留 6px 余量（分式/根号能落在
          ;; 一行文字里而不被压成看不清），多出来的位由 md_layout 用空行/行高让开
          (MDL.set_imath_metrics (+ (+ font_base text_pad) IMATH_PAD)
                                 (+ (- line_h (+ font_base text_pad)) IMATH_PAD)
                                 (+ font_base text_pad))
          ;; 目录/设置菜单的行高跟着字模走（16px 字模下 15px 行距会挤）
          (menu_metrics info)             ;; 把新字模的步进/墨高/内缩同步给菜单组件
          ;; 只在「文件里的值与本次要用的不同」时才写盘：
          ;;   · 启动时 load_font 已经把读到的值记进 font_saved，这里就变成空操作——
          ;;     否则每次冷启动都要在首帧前白白做一次写盘（掌机是 SD 卡写）；
          ;;   · 用户在设置面板里换字模时 id 一定变了，照旧落盘。
          (if (== id font_saved)
              #f
              { (Fs.write FONT_PATH (String.atom_to_string id))
                (set! font_saved id) })
          (display "MDFONT id=") (display id)
          (display " line_h=") (display line_h) (newline)
          ;; 换字模后所有段的折行/行高都变了：清空段缓存重来，但**保留阅读位置**——
          ;; 锚点按"段号 + 段内 dy 比例"换算（新行高/旧行高），重开文档后落回原处。
          ;; 没有打开文档时：有错误提示就重排提示文案（行高也变了），否则什么都不做
          ;;（boot 阶段的 load_font 就跑在这里，那时还没决定打开哪篇文档）。
          (if (== MDV_ok 1)
              { (set! keep_a (MDV.anchor))
                (set! keep_ratio (/ line_h old_h))
                (open_path cur_path cur_title)
                (if (== MDV_ok 1)
                    { (MDV.set_anchor (get_item keep_a 0)
                                      (Math.floor (* (get_item keep_a 1) keep_ratio)))
                      (MDV.ensure (get_item keep_a 0))
                      (sync_scroll) }
                    #f) }
              (if (> (String.length cur_path) 0)
                  (open_path cur_path cur_title)
                  (if (> (length notice_lines) 0) (open_prompt) #f))) }
        #f)
    #t))

;; 启动时读回上次的选择。注意两个坑：
;;   ① 记录被改坏（内容不是 0/1/2，或只写了半截）时 (String.parseNumber) 返回 #null。
;;      此时必须兜底 (apply_font 1)：否则 (Gfx.set_font) 与 (MDL.set_font_metrics) 一次都
;;      不调，整套排版会用宿主默认度量（行高/折行/表格列宽全错），而且没有报错——静默错排。
;;   ② 读到合法值就先记进 font_saved，让 apply_font 跳过重复写盘（见上）。
(define load_font
  (lambda ()
    (define s (Fs.read FONT_PATH))
    (define v #f)
    (define applied 0)
    (if (null? s) #f
        { (set! v (String.parseNumber s))
          (if (null? v)
              #f
              { (set! v (Math.floor v))
                (set! font_saved v)
                (apply_font v)
                (set! applied 1) }) })
    (if (== applied 0) (apply_font 1) #f)
    #t))

;; 设置面板：同样交给通用菜单组件（三个字体单选项 = 三行"带琴键指示灯"的菜单条目）
;;   视口 = 「字体」小节标签下方到页脚之间；指示灯：当前字模实心、其余空心。
(define settings_menu_items
  (lambda ()
    (define i 0)
    (define out (new_list))
    (while (< i 3)
      { ;; 指示灯 2 = 实心（当前字模）/ 1 = 空心；文字色随主题，高亮态更亮
        (if (== i font_id)
            (push out (ML.item (font_name i) 236 242 250 0 2 255 255 255))
            (push out (ML.item (font_name i) 226 232 244 0 1 250 252 255)))
        (set! i (+ i 1)) })
    out))

(define settings_view_y (lambda () (+ SET_TOP 18)))
(define settings_view_h (lambda () (- (- SH BOT_H) (+ SET_TOP 18))))

(define draw_settings
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH 16 19 26)
    ;; 列表（组件绘制：高亮行只有底色，无左侧竖条；指示灯在左）
    (ML.set_items (settings_menu_items))
    (ML.draw)
    ;; 「字体」小节标签 + 底部说明（页眉页脚最后画，盖住边界半行）
    (Gfx.draw_text "字体" PAD_X SET_TOP 150 168 190)
    (Gfx.draw_text (String.concat "当前：" (String.concat (font_name font_id)
                                                          (String.concat "  行高 "
                                                                         (String.atom_to_string line_h))))
                   PAD_X (- (- SH BOT_H) 16) 150 168 190)
    (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
    (Gfx.draw_text "设置" PAD_X 3 236 242 250)
    (ML.draw_back)
    (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
    (Gfx.draw_text "↑↓ 选择 · 点按切换 · 右上返回" PAD_X (- SH 14) 130 140 155)
    (Gfx.refresh)
    #t))

(define settings_open_panel
  (lambda ()
    (ML.reset (settings_menu_items) (menu_row_h))
    (ML.geom 0 (settings_view_y) SW (settings_view_h))
    (ML.back_top_right SW)
    (App.set_exit_button 0 0 0 0)   ;; 同上：右上角归设置面板自己的返回键
    (ML.set_sel font_id)          ;; 高亮项默认落在当前字模上
    (set! settings_open 1)
    (display "MDSET open") (newline)
    #t))

(define apply_key_settings
  (lambda (k)
    (if (== k 1) (ML.move (- 0 1)) #f)
    (if (== k 2) (ML.move 1) #f)
    (if (== k 3) (ML.page (- 0 1)) #f)
    (if (== k 4) (ML.page 1) #f)
    (if (== k 13) (apply_font (ML.sel_of)) #f)
    (if (== k 27) { (set! settings_open 0) (set! exitbtn_mute 4) } #f)
    #t))

(define tick_settings
  (lambda ()
    (define hit 0)
    { (if (App.back_requested) { (set! settings_open 0) (set! exitbtn_mute 4) } #f)
      (set! key (Key.take_press))
      (while (not (== key 0))
        { (apply_key_settings key)
          (set! key (Key.take_press)) })
      ;; 触屏：拖动 = 滑动滚动（条目只有三条，实际不会滚，但手感一致）；
      ;; 轻点条目 = 选中并立即应用；轻点页眉 = 关闭
      (set! hit (ML.step_touch))
      (if (>= hit 0)
          (apply_font (ML.sel_of))
          (if (== hit -2)
              (if (== (ML.hit_back (ML.tap_x) (ML.tap_y)) 1)
                  { (set! settings_open 0) (set! exitbtn_mute 4) }
                  #f)
              #f))
      (draw_settings) }))

(define draw_set_btn
  (lambda ()
    (Gfx.fill_rect (set_btn_x) (set_btn_y) SET_BTN_W 14 52 58 76)
    (Gfx.draw_rect (set_btn_x) (set_btn_y) SET_BTN_W 14 110 118 140)
    (Gfx.draw_text "设置" (+ (set_btn_x) 7) (+ (set_btn_y) 1) 225 228 235)
    #t))

;; ---- 绘制：单个段 --------------------------------------------------------

;; 段 = (文本 样式 x 宽 url)；行内代码画药丸底、链接画下划线、项目符号画方块。
;; 注意：行内代码底 (46,42,34)、链接下划线 (96,160,235)、项目符号 (130,170,225)、
;; 代码块底 (26,32,29)、分隔线 (78,98,132)、表格竖线 (58,70,92) 这几个色值是与
;; 端到端测试的像素判据共享的
;; 契约（tools/e2e/markdown/test_app.py、tools/e2e/browser/regress_markdown.js），
;; 改色请同步改那两处。
(define draw_seg
  (lambda (sg y h)
    (define txt (get_item sg 0))
    (define st (get_item sg 1))
    (define x (+ PAD_X (get_item sg 2)))
    (define w (get_item sg 3))
    (define k 0)
    (set_style_color st)
    (if (== st MDL.ST_BULLET)
        (Gfx.fill_rect (+ x 4) (+ y (Math.floor (/ (- line_h 4) 2))) 3 3 cr cg cb)
        #f)
    (if (== st MDL.ST_TABLE_SEP)
        (Gfx.fill_rect x y 1 h cr cg cb)
        #f)
    ;; 块级图片：交给宿主解码（带 LRU 缓存），失败就走虚线框兜底（方案 D）
    (if (== st MDL.ST_IMAGE)
        { (if (Gfx.draw_image (image_path (get_item sg 4)) x y w h)
              #f
              (draw_image_missing x y w h txt)) }
        #f)
    (if (== st MDL.ST_IMAGE_MISSING)
        (draw_image_missing x y w h txt)
        #f)
    (if (== st MDL.ST_HR)
        (Gfx.fill_rect x y w 1 cr cg cb)
        #f)
    (if (== st MDL.ST_QUOTE_BAR)
        (Gfx.fill_rect x y w h cr cg cb)
        #f)
    (if (== st MDL.ST_H2BAR)
        (Gfx.fill_rect x y w h cr cg cb)
        #f)
    ;; 数学公式：按缓存的 op 流直接画（不重排）
    ;;   块级：x/行高由 md_layout 定好；行内：4 格是 em，基线对齐本行文字基线
    (if (== st MDL.ST_MATH)
        (draw_math txt x y h)
        #f)
    (if (== st MDL.ST_IMATH)
        (draw_imath txt x y (get_item sg 4))
        #f)
    (if (and (> (String.length txt) 0) (== st MDL.ST_CODE))
        (Gfx.fill_rect (- x 1) (+ y text_pad) (+ w 2) pill_h 46 42 34)
        #f)
    ;; 链接下划线：墨迹底在 y+text_pad+12，下划线再往下 3px，与字形留出呼吸感。
    ;; 绘制端把 y 当墨迹顶（文字上移了 text_gap），下划线必须跟着上移同样的量。
    (if (and (> (String.length txt) 0) (== st MDL.ST_LINK))
        (Gfx.fill_rect x (- (+ y (+ text_pad (+ font_base 3))) text_gap) w 1 96 160 235)
        #f)
    ;; 三级标题：文字下划线（比字形亮一档的暗色，避免和链接下划线混淆；同样下移 3px）
    (if (and (> (String.length txt) 0) (== st MDL.ST_H3))
        (Gfx.fill_rect x (- (+ y (+ text_pad (+ font_base 2))) text_gap) w 1 96 128 168)
        #f)
    ;; 图片段（ST_IMAGE / ST_IMAGE_MISSING）的 alt 与公式段的源串都不走通用文字绘制
    (if (and (> (String.length txt) 0)
             (not (== st MDL.ST_IMAGE))
             (not (== st MDL.ST_IMAGE_MISSING))
             (not (== st MDL.ST_MATH))
             (not (== st MDL.ST_IMATH)))
        {
          (set! k 0)
          (while (< k style_weight)
            { (Gfx.draw_text txt (+ x k) (+ y text_pad) cr cg cb)
              (set! k (+ k 1)) }) }
        #f)
    #t))

;; ---- 绘制：单行（背景 + 各段）-------------------------------------------

(define draw_line
  (lambda (ln y)
    (define h (get_item ln 0))
    (define bg (get_item ln 1))
    (define segs (get_item ln 2))
    (define i 0)
    (if (== bg MDL.BG_CODE)
        (Gfx.fill_rect (- PAD_X 4) y (+ CW 8) h 26 32 29)
        #f)
    (if (== bg MDL.BG_TABLE)
        (Gfx.fill_rect (- PAD_X 4) y (+ CW 8) h 24 30 38)
        #f)
    (if (== bg MDL.BG_RULE)
        (Gfx.fill_rect PAD_X y CW h 78 98 132)
        #f)
    (while (< i (length segs))
      { (draw_seg (get_item segs i) y h)
        (set! i (+ i 1)) })
    #t))

;; ---- 绘制：正文 ----------------------------------------------------------
;; 页眉/页脚后画，天然遮住越界的半行（图形层没有裁剪矩形，靠绘制顺序裁剪）。

(define draw_content
  (lambda ()
    (if (== MDV_ok 1)
        (draw_content_stream)
        (draw_content_notice))))

;; 流式文档：只遍历**缓存里的段**（视口 ± 若干屏），且段内一越界就收工。
;; 这与"整篇从第 0 行开始累加 y"的旧实现相比，每帧成本与文档大小无关。
(define draw_content_stream
  (lambda ()
    (define vt (MDV.top_y))
    (define n (MDV.n_cached))
    (define j 0)
    (define m #f)
    (define lines #f)
    (define i 0)
    (define y 0)
    (define bottom (- SH BOT_H))
    (define ln #f)
    (define stop 0)
    (while (< j n)
      { (set! m (MDV.cached j))
        (set! y (+ TOP_H (- (get_item m 1) vt)))     ;; 段首行的屏幕 y
        (set! lines (get_item m 3))
        (set! i 0)
        (set! stop 0)
        (while (and (< i (length lines)) (== stop 0))
          { (set! ln (get_item lines i))
            (if (> y bottom)
                (set! stop 1)                        ;; 屏外：本段收工
                (if (< (+ y (get_item ln 0)) TOP_H)
                    #f                               ;; 整行在页眉之上：跳过
                    (draw_line ln y)))
            (set! y (+ y (get_item ln 0)))
            (set! i (+ i 1)) })
        (set! j (+ j 1)) })
    #t))

;; 提示文案（内容只有两段，整篇排版即可）
(define draw_content_notice
  (lambda ()
    (define n (length notice_lines))
    (define i 0)
    (define y 0)
    (define ln #f)
    (define bottom (- SH BOT_H))
    (define stop 0)
    (set! y (- TOP_H scroll))
    (while (and (< i n) (== stop 0))
      { (set! ln (get_item notice_lines i))
        (if (> y bottom)
            (set! stop 1)
            (if (< (+ y (get_item ln 0)) TOP_H)
                #f
                (draw_line ln y)))
        (set! y (+ y (get_item ln 0)))
        (set! i (+ i 1)) })
    #t))

;; ---- 绘制：页眉 / 页脚 / 滚动条 -----------------------------------------

(define draw_header
  (lambda ()
    (define pct 0)
    (define mx (max_scroll))
    (define txt "100%")
    (Gfx.fill_rect 0 0 SW TOP_H 30 38 52)
    ;; 标题按可用宽度裁剪（右侧留给百分比与右上角的「退出」按钮，见 tools/e2e 的像素判据）
    (Gfx.draw_text (MDL.clip_text cur_title 232) PAD_X 3 236 242 250)
    (if (> mx 0)
        { (set! pct (Math.floor (/ (* (MDV.top_y) 100) (MDV.total_height))))
          (if (> pct 100) (set! pct 100) #f)
          ;; 索引还没走到文件尾时百分比是**估算**，用 ≈ 如实标注（宁可不精确也要诚实）
          (set! txt (if (MDV.scan_done)
                        (String.concat (String.atom_to_string pct) "%")
                        (String.concat "≈" (String.concat (String.atom_to_string pct) "%"))))
          (Gfx.draw_text txt 250 3 150 168 190) }
        (Gfx.draw_text "100%" 250 3 150 168 190))
    #t))

(define draw_footer
  (lambda ()
    (Gfx.fill_rect 0 (- SH BOT_H) SW BOT_H 24 28 38)
    (if (== MDV_ok 1)
        (Gfx.draw_text "拖动/↑↓ 滚动" PAD_X (- SH 14) 130 140 155)
        (Gfx.draw_text "点按/Enter 选文档" PAD_X (- SH 14) 130 140 155))
    (if (and (== MDV_ok 1) (> (MDV.toc_count) 0))
        (draw_toc_btn)      ;; 底栏右侧偏左：目录（仅阅读页且有标题时）
        #f)
    (draw_set_btn)    ;; 设置（App 级：字体）
    (draw_open_btn)   ;; 底栏右侧：打开选择器（挑文档的入口）
    #t))

(define draw_scrollbar
  (lambda ()
    (define vh 0)
    (define mx 0)
    (define th 0)
    (define bar 0)
    (define by 0)
    (set! mx (max_scroll))
    (if (> mx 0)
        { (set! vh (view_h))
          (set! th (MDV.total_height))
          (set! bar (Math.floor (/ (* vh vh) th)))
          (if (< bar 10) (set! bar 10) #f)
          (set! by (+ TOP_H (Math.floor (/ (* (MDV.top_y) (- vh bar)) mx))))
          (Gfx.fill_rect (- SW 3) TOP_H 3 vh 34 40 52)
          (Gfx.fill_rect (- SW 3) by 3 bar 96 116 148) }
        #f)
    #t))

;; ---- 绘制总入口 ----------------------------------------------------------

(define draw_all
  (lambda ()
    (if (== settings_open 1)
        (draw_settings)         ;; 设置面板：整屏归它（含推屏）
        (if (== toc_open 1)
            (draw_toc_panel)    ;; 目录面板
            (draw_page)))))

(define draw_page
  (lambda ()
    (Gfx.fill_rect 0 0 SW SH 14 17 22)
    (draw_content)          ;; 正文（没打开文档时就是提示文案）
    (draw_scrollbar)
    (draw_header)
    (draw_footer)
    (EX.draw_button)
    ;; 面板刚关闭的几帧：把宿主热区清零，避免同一次点按被"面板返回键 + 宿主退出键"重复解释
    (if (> exitbtn_mute 0)
        { (App.set_exit_button 0 0 0 0)
          (set! exitbtn_mute (- exitbtn_mute 1)) }
        #f)
    (Gfx.refresh)
    #t))

;; ---- 绘制：启动提示（第一帧，不依赖任何已打开的文档）------------------------
;;
;; 只做一件事：用 5 个 Gfx 调用把「机器在忙、在看哪一篇」画出来并推屏。
;; 版式/配色照抄宿主 ui_scheme_app.c 的 app_draw_start_screen（同一块底 14/17/22、
;; 同一个忙条色 96/160/235），这样「宿主启动画面 → 本 App 启动提示 → 阅读页」三段
;; 过渡不闪。字模 id 显式给出（2 = 16px 标题、1 = 12px 说明），不依赖当前
;; (Gfx.set_font) 的值——启动时它还没被 apply_font 设过。
(define draw_splash
  (lambda ()
    (define cx 0)
    (define cy 0)
    (set! cx (Math.floor (/ SW 2)))
    (set! cy (Math.floor (/ SH 2)))
    (Gfx.fill_rect 0 0 SW SH 14 17 22)
    (Gfx.draw_text_centered 2 "MD阅读器" cx (- cy 22) 236 242 250 1)
    (Gfx.draw_text_centered 1 splash_msg cx (+ cy 2) 150 168 190 1)
    (Gfx.fill_rect (- cx 64) (+ cy 24) 128 3 40 48 62)    ;; 忙条槽
    (Gfx.fill_rect (- cx 64) (+ cy 24) 42 3 96 160 235)  ;; 忙条亮段（静态）
    (Gfx.refresh)
    #t))

;; EX 的重绘回调必须是这个分发器，而不是 draw_all：
;;   退出确认框选「留下」、从后台/挂起恢复、从别的 App 返回——都会调一次重绘，
;;   而那时可能还没进入阅读页（boot_stage = 0），draw_all 会画出一张“空阅读页”。
(define draw_screen
  (lambda ()
    { (if (== boot_stage 1) (draw_all) (draw_splash))
      #t }))

;; ---- 交互：链接命中测试 --------------------------------------------------
;; 返回命中的链接目标（空串表示没点中）。行/段都是排版期定死的绝对坐标，
;; 因此命中测试就是"滚动偏移 + 行高累加"的算术。
(define hit_link
  (lambda (px py)
    ;; 屏幕 y → 文档绝对 y 需要减去页眉高度：视图层的 top 参数按"内容坐标原点"给
    (if (== MDV_ok 1)
        (MDV.hit_link px py (- (MDV.top_y) TOP_H))
        "")))

;; 阅读区轻点：命中链接 → 弹窗显示地址（模态期间解释器进程挂起，界面照常刷新）
(define tap_reader
  (lambda (px py)
    (if (< py TOP_H) #f
        (if (>= py (- SH BOT_H)) #f
            { (set! hit_url (hit_link px py))
              (if (> (String.length hit_url) 0)
                  { ;; 诊断行：端到端测试据此断言"点链接确实命中并取了目标地址"
                    (display "MDLINK ") (display hit_url) (newline)
                    (Ui.message_titled "链接目标" hit_url) }
                  #f) }))))

;; ---- 交互：触屏（拖动滚动 / 轻点）---------------------------------------

;; 没有打开文档（空状态/错误提示）时，正文区轻点 = 直接弹文件选择器：
;; 此时整屏只有“挑一篇文档”这一件事可做，不必非去点底栏那个小按钮。
(define handle_touch
  (lambda ()
    (define vnow 0)
    (define vdt 16)
    (set! drag_moved 0)
    (if (Touch.is_pressed)
        { (if (== prev_down 0)
              { (set! prev_down 1)
                (set! drag_total 0)
                (set! drag_last (Touch.y))
                (set! press_x (Touch.x))
                (set! press_y (Touch.y))
                ;; 按下即刹车：打断正在跑的惯性（手指接管，立刻跟手）
                (fling_stop)
                (set! drag_v 0)
                (set! drag_vt (System.timestamp))
                ;; 按下沿就把节拍提到 16ms（见 pace_tick）：否则第一段位移仍按 60ms
                ;; 采样，一次拖动的**开头**会先跳一大步（实测快拖首拍 66ms/22px），
                ;; 中段才变顺——手感上"起步发滞"。
                (set! drag_idle 0) }
              #f)
          (set! dyy (- (Touch.y) drag_last))
          (set! drag_last (Touch.y))
          (set! drag_total (+ drag_total dyy))
          (if (not (== dyy 0)) (set! drag_moved 1) #f)
          ;; 速度 EMA（px/ms，正 = 内容上滚）：**只在真有位移的拍**更新，dt 也取
          ;; 「距上一次位移」——因为 WASM 触屏松手后会做亚帧点按锁存（hal_touch_wasm.c
          ;; 的 WASM_TOUCH_LATCH_FRAMES：松手后仍有几拍报"按下"、坐标不动），
          ;; 若把这几拍的 dy=0 也喂进 EMA，刚甩出去的速度会被自己磨掉一半。
          (if (not (== dyy 0))
              { (set! vnow (System.timestamp))
                (set! vdt (ms_dt drag_vt vnow))
                (set! drag_v (+ (* drag_v (- 1 FLING_EMA))
                                (* FLING_EMA (/ (- 0 dyy) vdt))))
                (set! drag_vt vnow) }
              #f)
          (if (== MDV_ok 1)
              { (scroll_by_px (- 0 dyy))
                (sync_scroll) }
              #f) }
        { (if (== prev_down 1)
              { (set! prev_down 0)
                (if (< (Math.abs drag_total) 8)
                    (if (== (hit_set_btn press_x press_y) 1)
                        { (settings_open_panel) }
                        (if (== (hit_open_btn press_x press_y) 1)
                        (start_picker)
                        (if (and (== (hit_toc_btn press_x press_y) 1)
                                 (== MDV_ok 1)
                                 (> (MDV.toc_count) 0))
                            (toc_open_panel)
                            (if (== MDV_ok 1)
                                (tap_reader press_x press_y)
                                (start_picker)))))
                    ;; 真拖动过：够快就起甩（慢停下来的不算，见 fling_start）
                    (fling_start)) }
              #f) })
    #t))

;; ---- 交互：滚轮（原生通道）-------------------------------------------
;; 平滑在这里做（页面只把像素增量原样交过来）——本 App 的拍子会在 16/60ms 之间切，
;; 页面替它定平滑节奏只会两层缓动叠在一起。一拍走「剩余量的 WHEEL_GAIN」，
;; 剩余 ≤2px 时一次走完（**保证必然归零**：否则永远差最后一点点，节拍就回不到 60ms）。
;; 注意与滚动位置的关系：轮滚一格 = 28px ≈ 两行，一格 ~90ms 缓动到位（与宿主 GUI
;; 的滚动控件同手感）。方向：wheel_dy > 0 = 内容上滚 = scroll_by_px 的正方向。
(define wheel_step_px
  (lambda ()
    (define r wheel_rest)
    (define a (Math.abs r))
    (define s 0)
    (if (<= a 2)
        (set! s r)
        { (set! s (Math.round (* r WHEEL_GAIN)))
          (if (== s 0) (set! s (if (> r 0) 1 -1)) #f) })
    (set! wheel_rest (- r s))
    s))

(define handle_wheel
  (lambda ()
    (define px 0)
    ;; 面板/选择器是**本拍**由 handle_touch 刚打开的（on_tick 分支选择发生在它之前）：
    ;; 此时输入归面板，滚轮既不落位也不动节拍（节拍见 pace_tick，判据同源）；
    ;; 剩余量丢掉，免得退出面板时凭空多滚一段。
    (if (or (FP.active) (== settings_open 1) (== toc_open 1))
        { (set! wheel_rest 0)
          (set! wheel_idle WHEEL_IDLE_TICKS) }
        { (if (not (== wheel_total 0))
              { (fling_stop)   ;; 滚轮是主动定位：先刹车，再按像素缓动
                (set! wheel_rest (+ wheel_rest wheel_total))
                (set! wheel_idle 0) }
              (set! wheel_idle (+ wheel_idle 1)))
          (if (not (== wheel_rest 0))
              { (set! px (wheel_step_px))
                (if (not (== px 0))
                    (scroll_by_px px)   ;; 它内部已调 sync_scroll
                    #f) }
              #f) })
    #t))

;; ---- 甩滑惯性：实现 ------------------------------------------------------
(define cur_scroll
  (lambda () (if (== MDV_ok 1) (MDV.top_y) scroll)))

;; 两次取时刻之间的毫秒差：钳到 [4,120]（0/负值=同一毫秒或 27 位时间戳回绕 → 按一拍算）
(define ms_dt
  (lambda (t0 t1)
    (define d (- t1 t0))
    (if (< d 0) 16 (if (< d 4) 4 (if (> d 120) 120 d)))))

;; t1 是否在 t0 之后 limit 毫秒内（回绕当作"刚刚"，不会误判成"很久没动"而丢掉甩滑）
(define ms_recent
  (lambda (t0 t1 limit)
    (define d (- t1 t0))
    (if (< d 0) #t (if (> d limit) #f #t))))

;; 停掉惯性（并给 e2e 一行可断言的收尾诊断）
(define fling_stop
  (lambda ()
    (if (not (== fling_v 0))
        { (set! fling_v 0)
          (set! fling_acc 0)
          (display "MDFLING end=") (display (Math.floor (cur_scroll))) (newline) }
        #f)
    #t))

;; 松手时若够快就起甩（drag_v 是松手前的速度 EMA，正 = 内容上滚）
(define fling_start
  (lambda ()
    (define v drag_v)
    (if (and (>= (Math.abs drag_total) 8)
             (>= (Math.abs v) FLING_MIN_V)
             (ms_recent drag_vt (System.timestamp) FLING_STALE_MS))
        { (if (> v FLING_VMAX) (set! v FLING_VMAX) #f)
          (if (< v (- 0 FLING_VMAX)) (set! v (- 0 FLING_VMAX)) #f)
          (set! fling_v v)
          (set! fling_acc 0)
          (set! fling_t (System.timestamp))
          (display "MDFLING start=") (display (Math.floor (cur_scroll))) (newline) }
        #f)
    #t))

;; 每拍推进惯性：撞边界 / 速度衰减到阈值以下就停。放在 handle_touch 之后调——新按下
;; 会先把 fling_v 清零（刹车），所以这里不会和拖动抢滚动位置。
;; dt 用**实测墙钟差**（不是"每拍固定 16ms"）：掉帧/长帧不会让惯性变慢变远。
;; 但 dt 太离谱（负数=27 位时间戳回绕、>150ms=App 刚从后台恢复或某拍卡死）时不补步，
;; 直接停——否则一恢复就会瞬移好几屏。
(define handle_fling
  (lambda ()
    (define now 0)
    (define raw 0)
    (define dt 16)
    (define px 0)
    (define before 0)
    (if (not (== fling_v 0))
        { (set! now (System.timestamp))
          (set! raw (- now fling_t))
          (if (or (< raw 0) (> raw 150))
              (fling_stop)
              { (set! dt (ms_dt fling_t now))
                (set! fling_t now)
                (set! fling_acc (+ fling_acc (* fling_v dt)))
                (set! px (Math.round fling_acc))
                (set! fling_acc (- fling_acc px))
                (if (not (== px 0))
                    { (set! before (cur_scroll))
                      (scroll_by_px px)
                      (sync_scroll)
                      ;; 位置没动 = 撞到顶部/底部：立即停（不做橡皮筋回弹）
                      (if (== (cur_scroll) before) (fling_stop) #f) }
                    #f)
                (if (not (== fling_v 0))
                    { (set! fling_v (* fling_v (Math.pow FLING_DECAY (/ dt 16.0))))
                      (if (< (Math.abs fling_v) FLING_VMIN) (fling_stop) #f) }
                    #f) }) }
        #f)
    #t))

;; ---- 节拍：跟手/缓动时提速（**唯一**的 set_tick 决策点）----------------------
;; 必须在 handle_touch / handle_wheel 之后调——它们会更新 drag_moved / fling_v /
;; wheel_rest / wheel_idle。放在两处各自 set_tick 会互相改回去（选择器那次就是这样掉到 16fps 的）。
(define pace_tick
  (lambda ()
    ;; 面板/选择器自己管节拍（选择器要 16ms 跟手）：本拍刚被 handle_touch 打开时
    ;; 绝不能动，否则会把 start_picker 刚设的 16ms 改回 60ms——下一拍就走面板分支，
    ;; 再没人纠正，选择器整段只能跑 16fps。
    (if (or (FP.active) (== settings_open 1) (== toc_open 1))
        #f
        { (if (== drag_moved 1)
              (set! drag_idle 0)
              (set! drag_idle (+ drag_idle 1)))
          (if (or (and (== prev_down 1) (< drag_idle DRAG_IDLE_TICKS))
                  (not (== fling_v 0))
                  (not (== wheel_rest 0))
                  (< wheel_idle WHEEL_IDLE_TICKS))
              (set_tick FAST_TICK_MS)
              (set_tick TICK_MS)) })
    #t))

;; ---- 交互：键盘 ----------------------------------------------------------

;; 阅读页：↑↓ 滚一行、←→ 翻页；没有文档时 Enter 弹选择器（鼠标/触屏之外的键盘路径）。
;; Esc 不在这里处理（由 app/lib/exit.scm 的 (EX.check) 接管）。
(define apply_key
  (lambda (k)
    (if (== MDV_ok 1)
        { (if (== k 1) { (scroll_by_px (- 0 15)) (sync_scroll) } #f)
          (if (== k 2) { (scroll_by_px 15) (sync_scroll) } #f)
          (if (== k 3) { (scroll_by_px (- 0 (view_h))) (sync_scroll) } #f)
          (if (== k 4) { (scroll_by_px (view_h)) (sync_scroll) } #f) }
        (if (== k 13) (start_picker) #f))
    #t))

(define handle_keys
  (lambda ()
    (set! key (Key.take_press))
    (while (not (== key 0))
      { (apply_key key)
        (set! key (Key.take_press)) })
    #t))

;; ---- 主循环 --------------------------------------------------------------

;; ---- 后台异步工作（流式加载的第二半：**阅读不被打断，索引/目录在后面追**）------
;; 每拍做两件事，各有预算，保证 GUI 不掉帧：
;;   ① **索引**：从当前位置往后切段（只读块 + 切行 + 顺手提取标题，≈0.2~4ms/段）。
;;      索引是**目录的副产品**——扫到哪，目录就完整到哪，所以不需要"打开目录时再
;;      全篇扫一遍"的二次读盘；页眉百分比也随索引推进从估算变成实测。
;;      预算按字节给（IDX_BYTES），文档多大都不改变单拍开销。
;;   ② **预取**：把锚点相邻的一段排版进缓存（≈60ms，正好一拍），滚动才不卡。
;; 方向跟随阅读位置：段内剩余不足一屏时优先预取下一段。
(define IDX_BYTES 12288)       ;; 每拍索引预算（字节）≈ 3 段
(define IDX_BYTES_TOC 40960)   ;; 目录面板打开时的索引预算（面板渲染轻，可以快点）

(define index_work
  (lambda (budget_bytes)
    (define b0 (MDV.bytes_indexed))
    (define segs 0)
    (define go 1)
    (while (and (== go 1) (< segs 16))
      { (if (and (not (MDV.scan_done))
                 (< (- (MDV.bytes_indexed) b0) budget_bytes))
            { (if (> (MDV.index_step 1) 0)
                  (set! segs (+ segs 1))
                  (set! go 0)) }
            (set! go 0)) })
    segs))

;; 后台/滚动时新排出来的段里画了几张图（诊断行 MDIMG，e2e 据它断言「资源根回退」
;; 真的把插图取到了；MDSTAT 只在打开时打一次，那时插图段往往还没排到）
(define img_seen -1)

(define bg_work
  (lambda ()
    (define a #f)
    (define seg 0)
    (define dy 0)
    (define dir 1)
    (define ni 0)
    (if (== MDV_ok 1)
        { (index_work IDX_BYTES)
          (set! a (MDV.anchor))
          (set! seg (get_item a 0))
          (set! dy (get_item a 1))
          ;; 段内剩余不足一屏 → 预取下一段；否则预取上一段（回滚更顺）
          (if (< (- (MDV.height seg) dy) (view_h))
              (set! dir 1)
              (set! dir -1))
          (if (== (MDV.prefetch dir) 0)
              (MDV.prefetch (- 0 dir))
              #f)
          (set! ni (MDV.images))
          (if (not (== ni img_seen))
              { (set! img_seen ni)
                (display "MDIMG images=") (display ni) (newline) }
              #f) }
        #f)
    #t))

;; ---- 启动参数（App → App 拉起） ------------------------------------------
;;
;; 拉起方（例如 app/filemgr.scm 选中 .md/.txt 后点「进入」）用 (App.launch_with
;; "markdown" 路径) 把控制权交给本 App，并约定参数 = **待打开文件的宿主绝对路径**
;; （(native Fs)/Gfx 的路径规则：'/' 开头 = 宿主绝对路径，原样使用）。
;;
;; (App.launch_arg) 是**取走即清零**的（同 (App.back_requested)）：每一条参数只
;; 消费一次，没参数时返回 #null。两处消费点：
;;   · boot_work 里（启动第二拍）——被冷启动拉起：直接打开参数指定的文档（不读
;;     data/md-last.txt）；
;;   · on_tick 开头——后台实例被**再次拉起**（换一篇文档）：重新走同一条打开链路。
;; 为什么用绝对路径而不是资源根相对路径：参数要跨 App 传递，接收方的「当前目录」
;; 概念与发起方无关；绝对路径在桌面 / WASM MEMFS / 掌机 SD 上语义一致，且图片
;; 相对路径的基准（见 set_doc_dir）能从绝对路径直接推出来。
(define launch_open
  (lambda ()
    (define a (App.launch_arg))
    ;; 注意返回值必须是 **#t/#f**：本解释器里只有 #f 为假，0 是**真值**
    ;; （见 manual 附录 C 与 am_vm 的 `condition != AM_VALUE_FALSE`），
    ;; 写 "返回 1/0" 会让 (if (launch_open) ...) 永远走真分支。
    (if (null? a)
        #f
        (if (== (String.length a) 0)
            #f
            { ;; 面板/选择器一律收起：新文档要立刻接管画面与输入
              (set! settings_open 0)
              (set! toc_open 0)
              (FP.reset)
              ;; 公式缓存与上一次的文档路径都归 open_path 管（它会清缓存、成功后落盘）
              (open_path a (FSX.strip_ext (FSX.leaf a)))
              (display "MDLAUNCH ") (display a) (newline)
              #t }))))

(define on_tick
  (lambda ()
    ;; 滚轮（见文件头的滚轮说明）：每拍先把本拍到达的增量取走，并幂等重新声明认领
    ;; （宿主在新实例冷启动时会清掉认领，重复声明无副作用）。**只在这里取走**：
    ;; 面板/选择器/启动拍期间增量被直接丢弃，不会攒到回阅读页时一下子跳一大段。
    (if (== WHEEL_OK 1)
        { (Touch.wheel_enable)
          (set! wheel_total (Touch.wheel)) }
        (set! wheel_total 0))
    ;; 启动第二拍：画完启动提示的那一帧已经上屏（栈退空 → 浏览器合成），现在才真正
    ;; 读盘 + 排版首段。做完 boot_stage = 1，下一拍开始就是常规阅读循环。
    ;; 这一拍**不**处理键/触屏（提示页上没有可点的东西），也不装新定时器：
    ;; 用的就是 boot 装的这条 TICK_MS 定时器，相位不变。
    (if (== boot_stage 0)
        (boot_work)
        (if (launch_open)
            { (draw_all) }        ;; 被再次拉起：已经换了文档，本拍直接画新画面
                                  ;;（只有“换文档”这一拍跳过 EX.check；挂起的返回请求会在下一拍照常被消费）
            (if (FP.active)
                (tick_picker)          ;; 选择器活动期间：输入/绘制全归它，EX 不抢返回请求
                (if (== settings_open 1)
                    (tick_settings)    ;; 设置面板同理（返回请求 = 关闭面板）
                    (if (== toc_open 1)
                        (tick_toc)     ;; 目录面板（返回请求 = 关闭面板）
                        { (EX.check)
                          (handle_keys)
                          (handle_touch)
                          (handle_wheel)
                          (handle_fling)
                          (pace_tick)
                          (bg_work)
                          (sync_scroll)
                          (draw_all) })))))))

;; ---- 启动 ----------------------------------------------------------------
;;
;; 两拍：本函数只把几何算好、画一屏启动提示、装好常规拍就 return；一切会阻塞的工作
;; （读字模记录 / 读上次文档记录 / 切首段 / **解析 + 排版首段**）都在 boot_work 里，
;; 即这条定时器的第一下。理由见文件头「启动两拍」：WASM 的 (Gfx.refresh) 只写
;; canvas 的 backing store，浏览器要等 JS 栈退空才合成——只有 return 出去，这一帧才真的
;; 上屏。代价一拍（TICK_MS = 60ms），换来「点开就有反馈」。
(define boot
  (lambda ()
    { (set! SW (Gfx.width))
      (set! SH (Gfx.height))
      (set! CW (- SW (* 2 PAD_X)))
      ;; 滚轮原生通道（见文件头「滚轮」）：WASM GUI 有，掌机/桌面/串口键盘没有
      ;; （返回 #f → WHEEL_OK 保持 0，滚轮/拖动全部走原路）。
      ;; 必须在装定时器之前认领：页面收到第一个 wheel 事件时就要知道该走哪条通路。
      (if (Touch.wheel_available)
          (if (Touch.wheel_enable) (set! WHEEL_OK 1) #f)
          #f)
      ;; 诊断行（e2e 据此断言“滚轮通道认领成功/未启用”，与 MDBOOT 同类）：
      ;; 掌机/桌面恒为 0，浏览器里应为 1。
      (display "MDWHEEL ok=") (display WHEEL_OK) (newline)
      (MDL.set_math_measure doc_math_measure)  ;; 打开公式排版能力（必须在排版之前）
      (draw_splash)                           ;; ① 启动提示：5 个 Gfx 调用，<1ms
      (display "MDBOOT splash t=") (display (System.timestamp)) (newline)
      ;; ② 装常规拍就 return（回调包一层 lambda 注册：wasm32 上“自由变量很多的
      ;;    模块级闭包”直接当定时器回调会一次都不被调用，见 app/doom.scm）。
      ;;    这一拍的第一下就是 boot_work（重活），之后是常规阅读循环。
      (set! ts (System.set_interval TICK_MS (lambda () (on_tick)))) }))

;; 启动第二拍（重活）。注意这里仍然走**完全同一条**打开链路：
;; (launch_open) 消费启动参数（拉起方给的文档），否则 (boot_open) 续读上次的文档；
;; 两条路都落到 open_path → open_finish，所以 MDSTAT/MDLAUNCH/MDLAST/MDEMPTY
;; 这些诊断行的内容与顺序与改造前完全一致（只多了 MDBOOT 两行）。
(define boot_work
  (lambda ()
    { (load_font)           ;; 恢复上次选择的字模（字体影响排版度量，必须在打开文档之前）
      ;; 启动分两条路（没有内置示例，也没有文档列表）：
      ;;   · 被拉起（启动参数 = 待读文件）→ 直接打开它；
      ;;   · 从主菜单冷启动 → 续读 data/md-last.txt 里的上次文档；
      ;;     记录缺失或文件已不在 → 空状态提示（用户自己点「打开」挑一篇）。
      (if (launch_open)
          #f
          (boot_open))
      (set! boot_stage 1)      ;; 内容就绪：on_tick / draw_screen / 具名提示从此都走阅读页
      (draw_screen)           ;; 首帧真正的阅读页（或空状态/错误提示页）
      (display "MDBOOT ready t=") (display (System.timestamp)) (newline) }))

(EX.install_with draw_screen)
(boot)