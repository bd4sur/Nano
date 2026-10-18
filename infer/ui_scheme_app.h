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
// 运行时、注册 System/Math/String/Table/Gfx 五个本地库、**运行时读取源码**
// （非编译期内嵌）编译加载为虚拟机进程；此后每帧以固定时间片驱动解释器
// （解释器内部自动抢占分帧），推屏时机由源码内 (Gfx.refresh) 控制。
//
// 状态机：
//   STATE_SCHEME_APP_LIST —— 动态菜单（每次进入重新扫描，支持热插拔）
//   STATE_SCHEME_APP_RUN  —— 解释执行选中的 App（A 返回列表）
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

#ifdef __cplusplus
}
#endif

#endif
