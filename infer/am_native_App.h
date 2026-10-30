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
//    ui_scheme_app_event_handler 里无条件拦截 A/ESC（触屏上是 4x4 宫格右上角），
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
//    · **触屏不再产生 A**：STATE_SCHEME_APP_RUN 已加入 ui_app_state_is_menu 抑制表，
//      4x4 宫格不再为该状态生成任何软按键（含右上角 esc）。App 要退出就自己画一个
//      可点按钮，调用 (App.exit) —— 这是"各 App 按需"的一半。
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
//      (App.back_requested)                  ; 用户按了硬 A/ESC（取走即清零）
//      (App.modal_closed)                    ; 最近一次宿主模态框已关闭（取走即清零）
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

// (App.hide) : Boolean —— 请求切后台（**P1 打桩**：宿主按退出处理）。恒 #t
int32_t am_native_App_hide(am_runtime_t *rt, am_process_t *proc);

// (App.back_requested) : Boolean —— 硬 A/ESC 待处理（取走即清零）
int32_t am_native_App_back_requested(am_runtime_t *rt, am_process_t *proc);

// (App.modal_closed) : Boolean —— 宿主模态框刚关闭（取走即清零）；
//   供"只在状态变化时才重绘"的 App 在模态框关闭后补一次重绘
//  （画面叠加式模态框不做帧缓冲快照，关闭后由 App 自己重绘——见 ui_scheme_app.c）
int32_t am_native_App_modal_closed(am_runtime_t *rt, am_process_t *proc);

// (App.is_active) : Boolean —— 是否前台活动（退出已获确认、或已宣告退出后为 #f）
int32_t am_native_App_is_active(am_runtime_t *rt, am_process_t *proc);

// (App.name) : String —— 当前 App 名（源码文件名去 .scm）
int32_t am_native_App_name(am_runtime_t *rt, am_process_t *proc);

// (App.platform) : String —— 平台标识（"wasm-gui"/"web"/"tty"/"m5esp"/"linux"…）
int32_t am_native_App_platform(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 宿主服务接口（供 ui_scheme_app.c 使用；Scheme 侧不可见）
// ---------------------------------------------------------------------------

// 每次启动一个 App 时调用：清空全部状态并登记 App 名（宽字符，可为 NULL）
void am_native_app_reset(const wchar_t *name);

// 退出实例时调用：与 am_native_ui_abort() 同一时机，丢弃未处理的请求
void am_native_app_abort(void);

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

#ifdef __cplusplus
}
#endif

#endif