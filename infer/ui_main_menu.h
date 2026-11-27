#ifndef __NANO_UI_MAIN_MENU_H__
#define __NANO_UI_MAIN_MENU_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 主菜单（STATE_MAIN_MENU）：表驱动的应用宫格
//
// 条目表 = 内置应用静态表 + app/ 目录扫描出的灵机 App（标题取源码元数据 ;@title=），
// 后台运行中的灵机 App 带绿色运行角标，点即恢复同一实例。
// 新增内置入口 = 往 ui_main_menu.c 的 s_app_menu_builtin_items[] 加一行；
// 新增灵机 App = 往 app/ 丢一个带元数据的 .scm（固件无需改动）。
//
// 布局：左侧 3 列滚动区（一屏 4 行，按像素跟手滚动 + 松手惯性）、右侧 1 列固定区
//（返回 / 设置 / 自述 / 关机）。全部为自绘触屏控件，实体键盘仅作兜底。
// ===============================================================================

// 进入主菜单时调用：重建条目表（含灵机 App 目录扫描与运行角标刷新）、复位触摸跟踪，
// 并夹住滚动位置（保留上次停留的屏位）
void ui_main_menu_entry(Global_State *global_state);

// 整屏重绘（is_full_redraw=1；0=仅重绘左侧滚动区），内含 gfx_refresh 推屏
void ui_main_menu_draw(Key_Event *key_event, Global_State *global_state, int32_t is_full_redraw);

// 触屏拖动滚动 + 点按执行 + 实体键盘映射
void ui_main_menu_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
