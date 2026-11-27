#ifndef __AM_NATIVE_APP_H__
#define __AM_NATIVE_APP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : App —— 灵机（Animac）应用外壳/生命周期本地库
//
//  为什么需要本库：
//    「灵机应用程序」此前**无法自己决定何时离开**——退出只有一条路：宿主在
//    ui_scheme_app_event_handler 里无条件拦截 A/ESC（当年触屏上是 4x4 宫格右上角），
//    直接把 STATE 切回列表并销毁解释器（见 ui_scheme_app.c）。于是：
//      · 俄罗斯方块只能在 Scheme 里手抄一份退出确认框，且「确认」后无法真正退出，
//        只能冻结游戏并提示"按 A 返回"（app/tetris.scm 的平台差异第 3 条）；
//      · 计算器/SVM 的布局必须**避开**右上角那一格，否则点一下就没了；
//      · 便签/游戏想"退出前落盘"没有任何时机；
//      · 将来要做多任务/后台运行，连"离开"这件事都没有词汇。
//    本库把「应用生命周期」从 GUI 状态机里拆出来，做成 Scheme 可调用的显式接口。
//
//  语义分层（关键设计，对应 manual §3.8 的进程模型）：
//    hide   切后台：**保留实例**（保留解释器池与全部进程/定时器），前台让出。
//                   本期（P1）**打桩**：宿主按 exit 处理（实现见 am_native_App_hide），
//                   接口先立住，多任务落地时只改宿主侧。
//    exit   关闭应用：请求退出 → 宿主弹**全局退出确认框**（J2ME 挂断键那套语义），
//                   用户「确认」后销毁实例回列表；「留下」则继续运行。
//    back   A/ESC 返回请求：宿主不再抢着退出，只把"用户按了返回"记下来，
//                   由 App 自己处理（见下"硬 A/ESC 的归属"）。
//
//  阻塞（(App.exit) 为什么是一个"卡住"的函数）：
//    与 (Ui.input ...) 同一套机制（上游本地库契约 N7 的阻塞型函数形态）：native 函数
//    里把**进程**置 BLOCKED 后直接 return 0——不压值、不 step。此后虚拟机不再调度该
//    进程，宿主的事件循环/渲染照常运转（App 的全部定时器随之暂停，动画自然定住）；
//    用户答复时 am_native_ui_resolve_confirm() 压 #t/#f + step 一次 + 置 READY + 重新
//    入队，进程就在被挂起的那条指令处继续——Scheme 侧看到的就是一个返回布尔的函数。
//
//    确认（#t）之后发生什么——**两段式收尾（P2）**：
//      ① 用户点「确认」→ 本库置「退出已确认」，但**不**立刻要求销毁；
//      ② App 拿到 #t 后做收尾：落盘/停外设/画最后一帧…（可以跨多拍，最多 2000ms，
//         期间 is_active 已为 #f）；
//      ③ App 调 (App.exit_now) 宣告收尾完毕 → 宿主在**本帧末尾**销毁实例；
//         若 2000ms 内始终没调，宿主也强制销毁（收尾超时，控制台会打印一行提示）。
//    也就是说 (App.exit) 返回 #t 之后**必须**以 (App.exit_now) 收尾（或干脆结束进程）：
//        (if (App.exit)
//            { (do_save) (App.exit_now) }     ; 确认：存盘后退出
//            #f)                               ; 留下：什么都不做
//    app/lib/exit.scm 已经把这套协议包好了：登记 (EX.on_exit 收尾函数) 即可，
//    调用顺序是「(App.exit) 确认 → 收尾函数 → (App.exit_now)」。
//
//    为什么不是"确认即销毁"：销毁必须发生在**帧末**，而收尾可能不止一拍（画个
//    "正在保存…"、等落盘、松手外设）；显式两段式把这段窗口写进契约，App 不必靠
//    "一帧时间片"的巧合来抢时间。
//
//  硬 A/ESC 的归属（本库的核心变化）：
//    · **触屏不再产生 A**：触屏不派生任何键码（全屏宫格软键已整体退场，含原先的
//      右上角 esc 格）。App 要退出就自己画一个可点按钮，调用 (App.exit)
//      —— 这是"各 App 按需"的一半。
//    · **硬 A/ESC 仍然有意义**：实体键盘的 ESC 由宿主拦下，但不再直接退出，而是置一个
//      "返回请求"位：(App.back_requested) 为真。谁来消费它？答案是 App 自己的每拍代码——
//      推荐一行 `(import EX "app/lib/exit.scm")` + 每拍 `(EX.check)`，就得到 J2ME 那种
//      "按挂断键 → 全局提示是否退出"的完整体验（该辅助模块见 app/lib/exit.scm）。
//    · **兜底**：宿主在请求置位后 700ms 内没人消费（App 尚未迁移、或 App 已经停机/报错），
//      就按旧行为直接退出——保证永远不会出现"按了 A 却出不去"的死角。
//
//  Unicode/文案：标题与正文按码点拷进本模块静态缓冲（Scheme 的字符串对象随进程堆走，
//  不能长期持有裸指针），与 (native Ui) 的文案交接方式完全一致。
//
//  Scheme 侧用法：
//      (native App)
//      (if (App.exit) (App.exit_now))        ; 弹全局确认框；确认则退出
//      (App.exit_with "结束本局？")           ; 自定义确认框标题
//      (App.exit_now)                        ; 不问，直接退（App 自己已确认过）
//      (App.back_to)                         ; 退出后回到哪个 App（#null = 顶层）
//      (App.back_requested)                  ; 用户按了硬 A/ESC（取走即清零）
//      (App.modal_closed)                    ; 最近一次宿主模态框已关闭（取走即清零）
//
//  拉起另一个 App（App → App，见下）：
//      (App.launch "markdown")               ; 请求宿主拉起 app/markdown.scm
//      (App.launch_with "markdown" 路径)     ; 同时传 1 个字符串作为启动参数
//      (App.launch_arg)                      ; 本实例收到的启动参数（取走即清零）
//
//  本地库契约：全部值型过程——每条 return 0 路径恰好压 1 值 + step 1 次（N1/N6）；
//  仅 (App.exit)/(App.exit_with) 走 N7 阻塞路径（压 0 不 step）。
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_App_lib;

// ---------------------------------------------------------------------------
// Scheme 可见过程
// ---------------------------------------------------------------------------

// (App.exit) : Boolean —— 弹全局退出确认框（阻塞）；
//   #t = 用户确认（宿主将于本帧末销毁本 App），#f = 留下（或已有其它模态框在等待时软失败）
int32_t am_native_App_exit(am_runtime_t *rt, am_process_t *proc);

// (App.exit_with 标题:String) : Boolean —— 同上，自定义确认框标题
int32_t am_native_App_exit_with(am_runtime_t *rt, am_process_t *proc);

// (App.exit_now) : Boolean —— 不询问，立即退出（本帧末销毁）。恒 #t
int32_t am_native_App_exit_now(am_runtime_t *rt, am_process_t *proc);

// (App.hide) : Boolean —— 请求切后台：宿主在本帧末把实例**冻结**（保留解释器与内存池）、
//   前台让给应用列表；列表里该 App 带「● 运行中」角标，选中即恢复同一实例。
//   恒 #t。单实例平台（ESP32）无后台能力，宿主按退出处理（见 ui_scheme_app.c）。
int32_t am_native_App_hide(am_runtime_t *rt, am_process_t *proc);

// (App.back_requested) : Boolean —— 硬 A/ESC 待处理（取走即清零）
int32_t am_native_App_back_requested(am_runtime_t *rt, am_process_t *proc);

// (App.modal_closed) : Boolean —— 宿主模态框刚关闭（取走即清零）；
//   供"只在状态变化时才重绘"的 App 在模态框关闭后补一次重绘
//  （画面叠加式模态框不做帧缓冲快照，关闭后由 App 自己重绘——见 ui_scheme_app.c）
int32_t am_native_App_modal_closed(am_runtime_t *rt, am_process_t *proc);

// (App.is_active) : Boolean —— 是否前台活动（切后台/锁屏/退出流程中为 #f）
int32_t am_native_App_is_active(am_runtime_t *rt, am_process_t *proc);

// (App.resumed) : Boolean —— 刚从后台/挂起恢复（取走即清零）。
//   App 据此补一次重绘：挂起期间画面归别人，恢复后它自己的下一拍可能还没画。
int32_t am_native_App_resumed(am_runtime_t *rt, am_process_t *proc);

// (App.set_exit_button x y w h) : #undefined —— 把右上角「退出」按钮的矩形告诉宿主。
//   **为什么需要**：按钮的点按如果只靠 App 每拍采样触屏电平来判定，慢节奏的 App
//   （例如星历：1000ms 一拍）会漏掉几乎每一次点按（人手点按 ~100~150ms，两次采样之间
//   就按完了）。登记热区后，宿主在**事件级**（触摸边沿）判定命中，命中即等同硬 A/ESC
//   置"返回请求"，App 在下一次轮询里照常弹全局确认框——与 App 的节拍快慢无关。
//   无触屏硬件时 app/lib/exit.scm 会登记 (0,0,0,0) 表示"没有按钮"。
int32_t am_native_App_set_exit_button(am_runtime_t *rt, am_process_t *proc);

// (App.set_timeslice ticks) : #undefined —— 调整**本 App 解释器每帧时间片**的指令条数上限。
//   背景：宿主每帧只驱动解释器一个时间片（ui_scheme_app.c 的 MB_TIMESLICE_TICKS，默认
//   312500 条 VM 指令）。时间片不够时，一件"必须一口气做完"的工作（例如生命游戏铺满
//   网格逐格演进一代）会被抢占拆到下一帧——帧率不变但推进速度**直接减半**（60→30 代/秒）。
//   计算密集型 App 用它换更大的单帧预算（代价是本帧的解释器耗时增加，帧率仍由渲染决定）。
//   夹在 [31250, 1250000]（默认值的 1/10 ~ 4 倍；4 倍约 13 ms，超过就会明显掉帧）。
//   每个 App 实例有独立解释器运行时，因此本设置只影响调用它的 App。
//   示例：(App.set_timeslice 550000)  ; 一代约 410k 指令，留约 30% 余量
int32_t am_native_App_set_timeslice(am_runtime_t *rt, am_process_t *proc);

// (App.name) : String —— 当前 App 名（源码文件名去 .scm）
int32_t am_native_App_name(am_runtime_t *rt, am_process_t *proc);

// (App.platform) : String —— 平台标识（"wasm-gui"/"web"/"tty"/"m5esp"/"linux"…）
int32_t am_native_App_platform(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 拉起另一个灵机 App（App → App）
//
//  为什么需要：App 之间此前完全隔离——文件管理器只能「显示详情」，打不开
//  阅读器/播放器；每个 App 都得自己抄一份别人的功能（或者干脆不做）。本组接口
//  把「把控制权交给另一个 App」变成一条与 (App.exit)/(App.hide) 同构的**请求**。
//
//  三段式（与 exit/hide 逐字同构，原因也一样）：
//    ① (App.launch*) 在**进程内**只做「校验 + 置请求位」，压 #t 后继续跑完这一拍
//       （不能在 native 函数里销毁/切换运行时：栈上还压着当前进程的上下文）；
//    ② 宿主在**本帧泵完之后**（ui_scheme_app_perform_launch）执行切换：
//       定槽 → 调用者让出前台 → 目标上前台 → 交付启动参数；
//    ③ 目标 App 在**启动（boot）或下一拍**里 (App.launch_arg) 取走参数。
//
//  调用者去往何处：
//    · 多实例平台（WASM/桌面，实例槽 3）——调用者被**挂起冻结**（与 (App.hide)
//      同一条通道）：解释器与内存池原地保留，列表里带「● 运行中」角标，选中即恢复
//      原状态。**目标 App 退出后回到调用者**（实例级返回链，见下“返回链”）。
//    · 单实例平台（ESP32，实例槽 1）——没有后台能力，调用者被**销毁**
//      （等价于「拉起即替换」，与该平台 (App.hide) 的退化语义一致）；没有“上一 App”
//      可回，目标退出后仍回应用列表。
//
//  目标已在后台（曾切后台/曾被拉起）：走「恢复」路径，实例内的状态全部保留；
//  新的一次拉起只把新的启动参数交给它（(App.launch_arg) 会再取到一条）——
//  这就是「同一个阅读器被再次拉起、打开另一篇文档」的自然实现。
//
//  失败一律软失败（返回 #f）：名字为空/超长、参数超长、已有未处理的拉起请求、
//  目标 App 不存在（宿主注册了存在性钩子时）、目标是调用者自己。宿主侧另会
//  打印一行 [scheme-app] 拉起失败：… 便于事后排查；调用者不受影响（继续活着）。
//
//  名字的写法：app/ 下源码文件名去 .scm（= 应用列表里显示的名字），也接受
//  "xxx.scm" 与 "app/xxx.scm"。注意列表显示名在宿主侧按 SCHEME_APP_NAME_MAX 截断
//  （当前 39 个字符），超长文件名请用截断后的名字。
//
//  返回链（本机制的第四段，2026-11 新增：App → App → …）
//    每个**实例**记住“我是被谁拉起的”（宿主侧 UI_Scheme_App_State.back_rel，按源码
//    路径标识），于是：
//      · 被拉起方点「退出」→ 确认 → 宿主销毁它，并把**调用者**恢复为前台
//        （与列表里选中「● 运行中」是同一条恢复路径，调用者的状态/滚动位置全部保留）；
//      · 调用者自己再退出 → 继续向它的调用者回，形成任意级的回退链；
//      · 链上某个实例已被 LRU 驱逐（实例槽 3：最多 2 级返回）、或本来就是从应用列表
//        进入的（顶层）→ 退出回应用列表。
//    两条不变量：每个“回退”都先销毁当前实例再前台化父节点 —— 因此链**必然终止**
//    （即使 A→B→A 绕圈）；父实例不存在时只降级为回列表，不需要任何回滚逻辑。
//    「后台」按钮语义不变：冻结本 App → 回应用列表（≈ home；「退出」才是 back）。
//    ESP32 单槽平台不建返回链（调用者已被销毁），行为与旧版一致。
//    文案：宿主通过 am_native_app_set_back_hint() 告知名义上的上一 App，于是
//      · (App.exit) 的确认框标题变成「退出并返回 文件管理器？」；
//      · (App.back_to) 读同一个值 —— App 可把退出入口直接画成「返回」
//        （app/lib/exit.scm 的右上角按钮已经这么做）。
//    时序保证：拉起**冷启动**时，boot（含 (EX.install)）里 (App.back_to) 已可用；
//    从后台恢复 / 返回上一 App 时它会变，App 在 (App.resumed) 那拍重读即可。
// ---------------------------------------------------------------------------

// 启动参数的长度上限（码点）。取 256：宿主绝对路径的上限是 FS_PATH_MAX=256 **字节**
// （UTF-8），再长的参数在 Fs/Gfx 侧也会被拒，故更长的参数一律拒绝而不是静默截断。
#define AM_NATIVE_APP_LAUNCH_ARG_MAX (256)

// (App.launch name:String) : Boolean —— 请求宿主拉起另一个 App（不带参数）。
//   #t = 请求已排队（将在本帧末执行）；#f = 未受理（见上方失败清单）
int32_t am_native_App_launch(am_runtime_t *rt, am_process_t *proc);

// (App.launch_with name:String arg:String) : Boolean —— 同上，并传递 1 个字符串
int32_t am_native_App_launch_with(am_runtime_t *rt, am_process_t *proc);

// (App.launch_arg) : String | #null —— 本实例收到的启动参数；**取走即清零**
//   （语义同 (App.back_requested)；没有参数时返回 #null，空串从不会被交付）
int32_t am_native_App_launch_arg(am_runtime_t *rt, am_process_t *proc);

// (App.back_to) : String | #null —— “退出后回到哪个 App”（显示名）；顶层实例返回 #null。
//   **只读、不取走**（与 launch_arg 不同：整个生命周期内都可用）。
//   用途：App 把自己的退出入口写成“返回”（而不是“退出”）——例如
//   app/lib/exit.scm 的右上角按钮：被拉起时画「返回」，顶层时画「退出」。
//   与 (App.exit) 的确认框标题同源（都是宿主经 am_native_app_set_back_hint 告知的）。
//   注意：拉起**冷启动**时它在 boot 里已经可用（宿主在目标上前台之前就写好了），
//   从后台恢复/返回上一 App 时会变（宿主会重新告知）——恢复后请重读。
int32_t am_native_App_back_to(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 宿主服务接口（供 ui_scheme_app.c 使用；Scheme 侧不可见）
// ---------------------------------------------------------------------------

// 每次启动一个 App 时调用：清空全部状态并登记 App 名（宽字符，可为 NULL）
void am_native_app_reset(const wchar_t *name);

// 退出实例时调用：与 am_native_ui_abort() 同一时机，丢弃未处理的请求
void am_native_app_abort(void);

// 前台/后台切换（宿主调用）：active=0 → (App.is_active) 为 #f（后台冻结 / 锁屏挂起）
void am_native_app_set_active(int32_t active);

// 恢复通知：置 (App.resumed) 标志（前台化时由宿主调用）
void am_native_app_notify_resumed(void);

// 「退出」按钮热区（宿主用）：由 (App.set_exit_button x y w h) 登记；
// 宿主在触摸**边沿**上问一次即可（与 App 的轮询节拍无关）。
//   hit_exit_button 返回 1 表示该点落在按钮矩形内；未登记（宽或高为 0）恒 0。
int32_t am_native_app_hit_exit_button(int32_t x, int32_t y);

// 切后台请求：(App.hide) 或退出确认框的「后台」按钮置位，宿主取走后执行冻结
void am_native_app_request_hide(void);
int32_t am_native_app_hide_requested(void);   // 1=有未处理的切后台请求

// 宿主要求退出（硬 A/ESC）：置"返回请求"位并记录时刻（am_runtime_now_ms 时间线）
void am_native_app_set_back_request(am_runtime_t *rt);

// 返回请求已挂起的毫秒数；无请求时返回 -1（宿主据此做兜底退出）
int32_t am_native_app_back_pending_ms(am_runtime_t *rt);

// 清掉返回请求（宿主决定兜底退出、或模态框接管时调用）
void am_native_app_clear_back_request(void);

// 退出收尾窗口（毫秒）：用户确认退出后，App 有这么久做收尾（可跨多拍）；
// 超时由宿主强制销毁。模态框存续期间由宿主调 am_native_app_extend_grace() 续期
// （App 正在问用户话，显然还没收尾完）。
#define AM_NATIVE_APP_EXIT_GRACE_MS (2000)

// 是否应该销毁本 App 实例（1-是）。三个来源：
//   · (App.exit_now)/(App.hide) —— 立即；
//   · (App.exit) 已获确认且**收尾窗口已超时**；
//   · 宿主自己的兜底退出（返回请求无人接管等）。
// 注意：仅仅"确认了退出"**不**等于可以销毁——那只是收尾窗口的起点（见上方协议）。
int32_t am_native_app_exit_wanted(am_runtime_t *rt);

// 退出是否已获用户确认（1-是）；供宿主区分"用户在收尾"与"没人管"
int32_t am_native_app_exit_confirmed(void);

// 收尾窗口已过去的毫秒数；未进入收尾窗口时返回 -1
int32_t am_native_app_exit_grace_ms(am_runtime_t *rt);

// 续期收尾窗口（宿主在模态框存续期间每帧调用：App 正在与用户交互，不是卡死）
void am_native_app_extend_grace(am_runtime_t *rt);

// 宿主发现"应该退出"后调用（或直接由 am_native_app_abort 清掉）
void am_native_app_clear_exit_wanted(void);

// 由 (native Ui) 在"退出型确认框"得到肯定答复时调用（启动收尾窗口，不立刻销毁）
void am_native_app_mark_exit_confirmed(am_runtime_t *rt);

// 宿主模态框关闭时调用（置 (App.modal_closed) 标志）
void am_native_app_notify_modal_closed(void);

// ---------------------------------------------------------------------------
// 拉起：宿主侧服务（Scheme 不可见）
// ---------------------------------------------------------------------------

// 取走未处理的拉起请求（**取走即清零**）。
// 返回 1 = 取到请求（name/arg 已拷入调用方缓冲，无参数时 arg 为空串）；0 = 没有请求。
// name 容量建议 >= 64 码点，arg 容量建议 >= AM_NATIVE_APP_LAUNCH_ARG_MAX。
int32_t am_native_app_launch_take(wchar_t *name, size_t ncap, wchar_t *arg, size_t acap);

// 交付启动参数给**刚上前台的实例**：宿主必须在 am_native_app_reset() 之后调用
// （reset 会清空 inbox；冷启动/恢复两条路径都以 reset 收尾）。arg 为 NULL/空串 = 无参数。
void am_native_app_set_launch_arg(const wchar_t *arg);

// 注册「App 名字是否存在」的查询钩子（宿主 ui_scheme_app.c 提供）；传 NULL 注销。
// 未注册时 (App.launch) 无法预先校验名字，一律按"受理"处理（由宿主在帧末打日志）。
void am_native_app_bind_lookup(int32_t (*fn)(const wchar_t *name));

// 登记“退出后会回到哪个 App”（显示名；NULL/空串 = 顶层，退出回应用列表）。
//   · **必须**在前台切换的 am_native_app_reset() **之后**调用（reset 会清空它）；
//   · 两个用途：(App.exit) 的确认框标题变成「退出并返回 X？」（告诉用户“确认后不是
//     回桌面，而是回上一个 App”，否则用户会以为退不出去）；(App.back_to) 读同一个值，
//     App 可据此把退出按钮画成「返回」。
//   · (App.exit_with 标题) 仍然优先（App 自定义标题覆盖本提示）。
void am_native_app_set_back_hint(const wchar_t *title);

#ifdef __cplusplus
}
#endif

#endif