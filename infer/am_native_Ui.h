#ifndef __AM_NATIVE_UI_H__
#define __AM_NATIVE_UI_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

////////////////////////////////////////////////////////////////////////////
//  Native Library : Ui —— 灵机（Animac）宿主文本输入/输出本地库
//
//  为什么需要本库：
//    「灵机应用程序」此前只能通过 (native Key)/(native Touch) 自己绘界面、自己攒
//    文本——要用中文，就得在 Scheme 里重写一遍九键拼音输入法，不现实。而宿主早就有
//    一整套**成熟的文本输入控件**（ui_widget_input_*：全键盘拼音 + 16 键九宫格输入法 +
//    英文/数字模式 + 候选字分页 + 触屏软键盘 + 光标/滚动/粘滞修饰键），鹦鹉笼（本地
//    LLM）的输入框就是它。本库把这套控件以「模态对话框」的形式借给 Scheme 用。
//
//  语义（关键设计，与 J2ME 的 TextBox / 浏览器 prompt() 同构）：
//    (Ui.input 标题)              : String | #null   —— 弹出输入框，**阻塞**直到用户提交
//    (Ui.input_with 标题 初值)    : String | #null   —— 同上，输入框预置初值
//    (Ui.message 正文)            : #undefined       —— 弹出只读文本视图，阻塞到用户关闭
//    (Ui.message_titled 标题 正文) : #undefined       —— 同上，带页眉标题
//
//    · 提交（短按 Enter）= 返回输入框里的文本（String，UTF-8/宽字符原样）；
//    · 取消（长按 A/Esc，或触屏点按页眉右侧「返回」热区）= 返回 #null；
//      注意：**短按 Esc 是控件自身的退格键**（平台惯例，见 ui_widget_input_event_handler），
//      故取消取长按——这样打字时不会误退出。
//    · (Ui.message ...) 关闭后返回 #undefined（只作语句用）。
//
//  「阻塞」是怎么实现的（宿主协作式，不是死等）：
//    本库在 native 函数里把**Scheme 进程**置为 BLOCKED（AM_PROCESS_STATE_BLOCKED）
//    并直接返回，既不入操作数栈也不推进 PC——这正是上游本地库契约 N7 的「阻塞型函数」
//    形态（System.read 等队列等待同理）。此后虚拟机不再调度该进程，宿主的事件循环、
//    渲染、其它进程/定时器照常运转；用户提交时宿主调用 am_native_ui_resolve()，
//    本库把结果压栈 + step 一次 + 置 READY 并重新入队，进程在被挂起的那条指令处继续。
//    因此 Scheme 侧看到的就是「一个返回文本的函数」，而界面始终可交互（这对浏览器/
//    安卓单线程目标尤其重要——真死等会把整个 UI 冻住）。
//
//    附带影响（有意为之）：进程处于 BLOCKED 时其 set_interval 定时器**暂停**（上游
//    runtime 的既定语义），即 App 的动画在输入模态框期间自然停住，无需 App 自己判断。
//
//  Scheme 侧用法：
//      (native Ui)
//      (define name (Ui.input "请输入名字"))
//      (if (null? name) (Ui.message "已取消") (Ui.message (String.concat "你好，" name)))
//
//  同时只允许一个请求（一个 App 内的多进程/多窗口输入会排队失败）：请求在等待期间
//  第二个 (Ui.input ...) 立即返回 #null，避免出现两个模态框争用同一套控件状态。
////////////////////////////////////////////////////////////////////////////

// 由本文件定义
extern const am_native_lib_entry_t am_native_Ui_lib;

// 请求种类（宿主据此选择控件与关闭方式）
#define AM_NATIVE_UI_KIND_NONE    (0)
#define AM_NATIVE_UI_KIND_INPUT   (1) // 文本输入框（ui_widget_input_*）
#define AM_NATIVE_UI_KIND_MESSAGE (2) // 只读文本视图（ui_widget_textarea_*）

// 文案缓冲区上限（码点数；超出截断）
#define AM_NATIVE_UI_TITLE_MAX (128)
#define AM_NATIVE_UI_BODY_MAX  (4096)

// ---------------------------------------------------------------------------
// Scheme 可见过程
// ---------------------------------------------------------------------------

// (Ui.input 标题:String) : String | #null —— 阻塞式文本输入
int32_t am_native_Ui_input(am_runtime_t *rt, am_process_t *proc);

// (Ui.input_with 标题:String 初值:String) : String | #null —— 带初值的阻塞式文本输入
int32_t am_native_Ui_input_with(am_runtime_t *rt, am_process_t *proc);

// (Ui.message 正文:String) : #undefined —— 阻塞式只读文本视图
int32_t am_native_Ui_message(am_runtime_t *rt, am_process_t *proc);

// (Ui.message_titled 标题:String 正文:String) : #undefined
int32_t am_native_Ui_message_titled(am_runtime_t *rt, am_process_t *proc);

// ---------------------------------------------------------------------------
// 宿主服务接口（供 ui_scheme_app.c 的模态框宿主使用；Scheme 侧不可见）
// ---------------------------------------------------------------------------

// 是否有请求在等待：1-有（宿主每帧须驱动模态框），0-无
int32_t am_native_ui_pending(void);

// 当前请求种类（AM_NATIVE_UI_KIND_*）；无请求时返回 AM_NATIVE_UI_KIND_NONE
int32_t am_native_ui_kind(void);

// 页眉标题（不会为 NULL；无标题时为空串）
const wchar_t *am_native_ui_title(void);

// (Ui.input_with) 的初值（不会为 NULL；无初值时为空串）
const wchar_t *am_native_ui_initial(void);

// (Ui.message*) 的正文（不会为 NULL）
const wchar_t *am_native_ui_body(void);

// 结束当前请求并唤醒被阻塞的进程：
//   result != NULL → 以该字符串为返回值（INPUT 类）；result == NULL → 返回 #null（取消）。
//   MESSAGE 类忽略 result，返回值恒为 #undefined。
// 无请求时为空操作。调用后 am_native_ui_pending() 立即变为 0。
void am_native_ui_resolve(am_runtime_t *rt, const wchar_t *result);

// 丢弃当前请求而不唤醒（宿主销毁解释器/退出 App 时调用，避免悬空 PID）
void am_native_ui_abort(void);

#ifdef __cplusplus
}
#endif

#endif