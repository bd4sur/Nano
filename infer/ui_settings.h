#ifndef __NANO_UI_SETTINGS_H__
#define __NANO_UI_SETTINGS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"

// ===============================================================================
// 系统设置（设置菜单 + 日期/时间/经纬度编辑器）
//
// 视觉与交互范式（与音乐盒/番茄表等已完成触屏化改造的模块一致）：
//   - 320x240 设计基准，尺寸按实际画布推导；
//   - 亮色/暗色双主题，强调色 #66CCFF；
//   - 开关沿用「开=绿、关=红」的颜色隐喻；
//   - 设置菜单为可拖动的分组卡片列表（触屏拖动滚动、滚轮经页面合成手势亦可滚动）；
//   - 日期/时间/经度/纬度使用独立编辑器：大字预览 + 三列步进器 + 快捷操作，
//     支持按住连续调整；实体键盘（方向键/Enter/Esc）照常可用。
//
// 状态机约定：
//   - 菜单态 STATE_SETTING_MENU；编辑器态 STATE_SETTING_INPUT；
//   - 两个状态都直接消费触屏流（见 ui_app.c 的 ui_app_state_is_menu），
//     不再依赖 4x4 宫格软按键；
//   - 本模块自行设置 global_state->STATE 完成状态切换。
// ===============================================================================

// 编辑器类型（与历史 value_type 编号一致，便于兼容）
#define UI_SETTINGS_EDIT_DATE      (0) // 日期
#define UI_SETTINGS_EDIT_TIME      (1) // 时间与时区
#define UI_SETTINGS_EDIT_LONGITUDE (2) // 经度
#define UI_SETTINGS_EDIT_LATITUDE  (3) // 纬度

// 进入设置菜单。reset=1 时回到列表顶部（从主菜单进入）；从编辑器返回时传 0 保留滚动位置。
void ui_settings_menu_enter(Global_State *global_state, int32_t reset);

// 绘制设置菜单并推屏（内部调用 gfx_refresh）
void ui_settings_menu_draw(Global_State *global_state);

// 设置菜单事件：实体按键下降沿 + 触屏边沿/电平。可能切换到 STATE_SETTING_INPUT /
// STATE_SPLASH_SCREEN；发生需要重绘的交互时本函数自行重绘并推屏。
void ui_settings_menu_event(Key_Event *key_event, Global_State *global_state);

// 进入编辑器（从菜单的日期/时间/经度/纬度行调用），并设置 STATE_SETTING_INPUT
void ui_settings_editor_enter(Global_State *global_state, int32_t edit_type);

// 绘制编辑器并推屏
void ui_settings_editor_draw(Global_State *global_state);

// 编辑器事件：取消/保存会切回 STATE_SETTING_MENU。
void ui_settings_editor_event(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
