#ifndef __NANO_UI_CALENDAR_H__
#define __NANO_UI_CALENDAR_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 日历
//
// 每周从周一开始，每月一页。进入时定位到系统当前年月，当前日期以高亮方块标出。
//   - 左右键（触屏底行 ←/→）：切换上月/下月；
//   - 上下键：切换上一年/下一年；
//   - 0 键：回到系统当前年月；
//   - A(ESC) 或 D(回车)：返回小游戏菜单。
//   - 触屏点击日期数字：打开该日的黄历模态框（ui_almanac，数据来自 almanac.h 的
//     cnlunar）；点击任意处或按任意键关闭并返回日历。
// ===============================================================================

void ui_calendar_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_calendar_render_frame(Key_Event *key_event, Global_State *global_state);
int32_t ui_calendar_event_handler(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
