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
// 返回链（2026-11）：被拉起的 App 退出后回**拉起它的 App**，而不是应用列表。
//   · 每个实例记住自己是被谁拉起的（UI_Scheme_App_State.back_rel，按源码路径标识）；
//   · 所有退出都汇到 ui_scheme_app_perform_exit：有活着的父实例就恢复它（留在
//     STATE_SCHEME_APP_RUN），否则回**主菜单**（STATE_MAIN_MENU，由 ui_app.c 切：
//     2026-11 起灵机 App 的图标就摆在主菜单里，退出即回主菜单）；
//   · 「后台」也回主菜单（「退出」才是 back），实例冻结保留、主菜单图标带运行角标。
//
// 模态文本输入/输出：App 调 (Ui.input ...) 时 Scheme 进程被挂起，本模块借用宿主的
// 文本控件（鹦鹉笼同款：全键盘拼音 + 16 键九宫格）当模态框，用户提交后唤醒进程
// （见 ui_scheme_app.c 的“模态文本输入/输出”段落与 am_native_Ui.h）。
//
// 新增 App：把 xxx.scm 放入业务资源目录 app/ 即可，固件无需改动。
//
// 元数据（2026-11）：App 源码头几行可带注释形式的元数据，扫描时读取，用于主菜单图标：
//   ;@id=album
//   ;@title=时光集
//   ;@version=2026-11-17
//   规则：id = 文件名去 .scm（须匹配 [_a-zA-Z]+[_a-zA-Z0-9]*）；title = 不含空格的短名；
//   version = 日期。缺 metadata 的旧 App 仍可运行，title 回退为 id。
// ===============================================================================

// 状态编号（原 STATE_MANDELBROT/STATE_SCHEME_GOL 已合并为此二态）
#define STATE_SCHEME_APP_LIST (79)
#define STATE_SCHEME_APP_RUN  (80)

// app/ 目录下 Scheme App 的数量上限（扫描缓冲容量）。主菜单按此预留条目表容量，
// 改动时与本文件的目录接口一起核对。
#define UI_SCHEME_APP_MAX (128)

// ---------------- 主菜单表驱动用的目录视图 ----------------
// 主菜单把 app/ 下的每个 Scheme App 也当作一个图标（标题取源码元数据 ;@title=）。
// 用法：进入主菜单时 ui_scheme_app_catalog_refresh() 重新扫描（支持热插拔），
// 然后按下标逐项查询标题/图标 id/后台运行状态；点按某图标时
// ui_scheme_app_catalog_enter(i) 记录选中项并返回 STATE_SCHEME_APP_RUN。
//   注意：返回的标题指针指向扫描缓冲，下一次 refresh 后失效——调用方须在
//   重建自己的条目表时同步重新取用。
int32_t        ui_scheme_app_catalog_refresh(void);              // 重新扫描，返回 App 数
int32_t        ui_scheme_app_catalog_count(void);
const wchar_t *ui_scheme_app_catalog_title(int32_t index);        // 显示名（元数据 ;@title=）
const char    *ui_scheme_app_catalog_id(int32_t index);           // id（文件名去 .scm）
int32_t        ui_scheme_app_catalog_running(int32_t index);      // 1 = 有存活实例（前台/后台）
int32_t        ui_scheme_app_catalog_enter(int32_t index);        // 选中并转入运行态（返回 STATE_SCHEME_APP_RUN）

// 退出/切后台瞬间的在途触摸序列余波抑制（见 ui_scheme_app.c 的 s_list_touch_guard）。
// 返回 1 = 本帧应丢弃触摸（函数已就地把 key_event 的触摸字段清零，按键字段保留），
// 调用方按剥离后的事件继续处理即可；返回 0 = 无抑制。
// 主菜单与列表菜单共用（App 退出后落回的是主菜单）。
int32_t ui_scheme_app_touch_aftermath_filter(Key_Event *key_event);

// ---------------- 列表菜单 ----------------
int32_t ui_scheme_app_list_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_scheme_app_list_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_scheme_app_list_item_action(Key_Event *key_event, Global_State *global_state,
                                       Widget_Menu_State *menu_state);

// ---------------- 运行宿主 ----------------
// 进入运行态时调用：**已有同源实例（后台冻结中）则恢复**，否则冷启动一个新实例。
int32_t ui_scheme_app_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_scheme_app_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_scheme_app_render_frame(Key_Event *key_event, Global_State *global_state);
void    ui_scheme_app_on_exit(void);

// 是否应该**切后台**（保留实例，1-是）。置位来源：(App.hide)、退出确认框的「后台」按钮。
// 同样在泵完本帧之后判定；宿主随即调 ui_scheme_app_suspend_to_list()。
// 单实例平台（ESP32）恒返回 0（hide 退化为退出）。
int32_t ui_scheme_app_should_hide(Global_State *global_state);

// 切后台：冻结当前实例（保留解释器与内存池）、前台让给应用列表
void ui_scheme_app_suspend_to_list(void);

// 锁屏等"临时挂起"用：只翻转 active 状态，不切状态机
void ui_scheme_app_suspend_current(void);
void ui_scheme_app_resume_current(void);

// 某个 App 是否还有活着的实例（列表「● 运行中」角标用）
int32_t ui_scheme_app_has_live_instance(const char *source_rel);

// ---------------- 拉起另一个 App（App → App） -------------
//
// 帧末执行 (App.launch)/(App.launch_with) 排下的请求（三段式的第二段，见
// am_native_App.h “拉起另一个灵机 App”）：
//   ① 名字 → s_apps[] 下标（接受 "markdown" / "markdown.scm" / "app/markdown.scm"）；
//   ② 定目标实例槽位——**必须在调用者仍是前台的时刻完成**（app_instance_alloc 只会
//      驱逐后台实例，否则会出现“拉起别人把自己驱逐掉”）；
//   ③ 调用者让出前台（多实例平台 = 冻结保留，与 (App.hide) 同一条通道；单实例平台
//      = 销毁，等同“拉起即替换”）；
//   ④ 目标上前台（已有后台实例 → 恢复；否则冷启动）；
//   ⑤ 交付启动参数（必须在 am_native_app_reset 之后，否则会被清掉）。
//
// 返回值：1 = 已切换到目标 App（本帧到此为止，ui_app.c 直接 break）；
//         0 = 无请求或拉起失败（调用者继续前台，已打日志）。
//
// 优先级：由 ui_app.c 保证「退出 > 拉起 > 切后台」。
int32_t ui_scheme_app_perform_launch(Global_State *global_state);

// 是否应该销毁本 App 实例（1-是）。由 ui_app.c 的状态机在**泵完本帧之后**调用：
//   · (App.exit_now) / (App.hide) → 立即退出；
//   · (App.exit) 已获用户确认 → 先给 App **收尾窗口**（AM_NATIVE_APP_EXIT_GRACE_MS，
//     默认 2000ms；期间宿主照常泵，App 可落盘/停外设/画最后一帧），App 调
//     (App.exit_now) 或窗口超时才销毁（两段式收尾，见 am_native_App.h）；
//     模态框存续期间自动续期（App 正在问用户话，不是卡死）；
//   · 硬 A/ESC 的“返回请求”700ms 内无人消费 → 兜底退出（App 未迁移／已停机报错）。
int32_t ui_scheme_app_should_exit(Global_State *global_state);

// 结束当前前台实例（**所有**退出路径的唯一出口，“退出 > 拉起 > 切后台”里的第一位）：
//   1 = 有活着的返回链 → 已恢复到上一 App（状态机仍是 STATE_SCHEME_APP_RUN，
//       下一帧由被恢复的 App 自己绘制，ui_app.c 什么都不要做）；
//   0 = 实例已销毁 → 调用方把 STATE 切到 STATE_MAIN_MENU（主菜单里就摆着该 App 的
//       图标，退出即回主菜单；列表菜单入口仍在“小游戏”里，但退出统一回主菜单）。
// 会内部完成实例收尾（模态框/退出请求/Chat 引擎/解释器池/Gfx 解绑），调用方**不要**
// 再调 ui_scheme_app_on_exit（那是“直接回列表”的旧入口，保留给“加载失败时按 A”以外的
// 特殊路径；正常退出都走本函数）。
int32_t ui_scheme_app_perform_exit(Global_State *global_state);

// 模态文本输入/输出（本地库 (native Ui)）是否正在显示。
// 事件层（ui_app.c get_input_event）据此抑制触屏 4x4 宫格映射：模态框显示期间
// 触屏归宿主文本控件所有（宫格会把点按变成软按键、把右上角变成 Esc）。
int32_t ui_scheme_app_hosts_input_widget(void);

#ifdef __cplusplus
}
#endif

#endif
