#ifndef __NANO_UI_APP_H__
#define __NANO_UI_APP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"

// ===============================================================================
// 获取按键事件
// ===============================================================================

void get_key_event(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// 开机欢迎画面
// ===============================================================================

void ui_app_splash_render_frame(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// Bad Apple
// ===============================================================================

void ui_app_badapple_render_frame(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// Game of Life
// ===============================================================================

void ui_app_gol_init(Key_Event *key_event, Global_State *global_state);
void ui_app_gol_render_frame(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// 玲珑天象仪
// ===============================================================================

void ui_app_linglong_render_frame(Key_Event *key_event, Global_State *global_state);
void ui_app_linglong_toggle_timemachine(Key_Event *key_event, Global_State *global_state);
void ui_app_linglong_set_timemachine_speed(Key_Event *key_event, Global_State *global_state, int32_t speed);
void ui_app_linglong_set_realtime(Key_Event *key_event, Global_State *global_state);



#ifdef __cplusplus
}
#endif

#endif
