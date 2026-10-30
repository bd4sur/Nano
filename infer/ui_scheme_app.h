#ifndef __NANO_UI_SCHEME_APP_H__
#define __NANO_UI_SCHEME_APP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 灵机应用程序（Scheme App）——插件式应用框架
//
// 业务资源目录 app/ 下的 .scm 即可插拔的"Scheme App"：进入"灵机应用程序"菜单
// 时**自动扫描**该目录（文件名即菜单项），选中后由通用宿主运行——创建解释器
// 运行时、注册本地库（System/Math/String/Table/Gfx/Time/Diag/Touch/Imu/Key/Fs/Ui）、**运行时读取源码**
// （非编译期内嵌）编译加载为虚拟机进程；此后每帧以固定时间片驱动解释器
// （解释器内部自动抢占分帧），推屏时机由源码内 (Gfx.refresh) 控制。
//
// 状态机：
//   STATE_SCHEME_APP_LIST —— 动态菜单（每次进入重新扫描，支持热插拔）
//   STATE_SCHEME_APP_RUN  —— 解释执行选中的 App
//
// 退出交互（2026-10 改造，见 am_native_App.h 的设计说明）：
//   · **触屏不再派生“返回”**：STATE_SCHEME_APP_RUN 已加入 ui_app_state_is_menu 抑制表，
//     4x4 宫格软按键（含右上角 esc 格）在该状态整体退场。App 要退出就自己画按钮、
//     调 (App.exit) / (App.exit_now)。
//   · **硬 A/ESC 仍归宿主拦下**，但不再直接销毁实例：只置一个“返回请求”位，
//     由 App 自己消费（推荐 (import EX "app/lib/exit.scm") + 每拍 (EX.check)，
//     即 J2ME 那种“按挂断键 → 全局提示是否退出”）。700ms 内无人接管则按旧行为兜底退出。
//   · 退出确认框就是 (native Ui) 的确认框（AM_NATIVE_UI_KIND_CONFIRM），由 App 侧
//     (App.exit) 发起、宿主用 ui_exit_confirm_* 绘制，**不做帧缓冲快照**：
//     关闭后由 App 自己在同一拍里重绘（事件驱动型 App 用 (App.modal_closed) 得知）。
//
// 模态文本输入/输出：App 调 (Ui.input ...) 时 Scheme 进程被挂起，本模块借用宿主的
// 文本控件（鹦鹉笼同款：全键盘拼音 + 16 键九宫格）当模态框，用户提交后唤醒进程
// （见 ui_scheme_app.c 的“模态文本输入/输出”段落与 am_native_Ui.h）。
//
// 新增 App：把 xxx.scm 放入业务资源目录 app/ 即可，固件无需改动。
// ===============================================================================

// 状态编号（原 STATE_MANDELBROT/STATE_SCHEME_GOL 已合并为此二态）
#define STATE_SCHEME_APP_LIST (79)
#define STATE_SCHEME_APP_RUN  (80)

// ---------------- 列表菜单 ----------------
int32_t ui_scheme_app_list_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_scheme_app_list_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_scheme_app_list_item_action(Key_Event *key_event, Global_State *global_state,
                                       Widget_Menu_State *menu_state);

// ---------------- 运行宿主 ----------------
int32_t ui_scheme_app_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_scheme_app_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_scheme_app_render_frame(Key_Event *key_event, Global_State *global_state);
void    ui_scheme_app_on_exit(void);

// 是否应该销毁本 App 实例（1-是）。由 ui_app.c 的状态机在**泵完本帧之后**调用：
//   · (App.exit_now) / (App.hide) → 立即退出；
//   · (App.exit) 已获用户确认 → 先给 App **收尾窗口**（AM_NATIVE_APP_EXIT_GRACE_MS，
//     默认 2000ms；期间宿主照常泵，App 可落盘/停外设/画最后一帧），App 调
//     (App.exit_now) 或窗口超时才销毁（两段式收尾，见 am_native_App.h）；
//     模态框存续期间自动续期（App 正在问用户话，不是卡死）；
//   · 硬 A/ESC 的“返回请求”700ms 内无人消费 → 兜底退出（App 未迁移／已停机报错）。
int32_t ui_scheme_app_should_exit(Global_State *global_state);

// 模态文本输入/输出（本地库 (native Ui)）是否正在显示。
// 事件层（ui_app.c get_input_event）据此抑制触屏 4x4 宫格映射：模态框显示期间
// 触屏归宿主文本控件所有（宫格会把点按变成软按键、把右上角变成 Esc）。
int32_t ui_scheme_app_hosts_input_widget(void);

#ifdef __cplusplus
}
#endif

#endif
