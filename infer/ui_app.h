#ifndef __NANO_UI_APP_H__
#define __NANO_UI_APP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"


// UI应用层全局状态定义
#define STATE_DEFAULT           (-100)
#define STATE_SPLASH_SCREEN     (-1)
#define STATE_MAIN_MENU         (-2)
#define STATE_EBOOK             (-3)
#define STATE_LLM_INPUT         (0)
#define STATE_MODEL_MENU        (4)
#define STATE_SETTING_MENU      (5)
#define STATE_LLM_ON_INFER      (8)
#define STATE_LLM_AFTER_INFER   (10)
#define STATE_ASR_RUNNING       (21)
#define STATE_README            (25)
#define STATE_BADAPPLE          (26)
#define STATE_GAMEOFLIFE        (27)
#define STATE_LINGLONG          (28)
#define STATE_FLASHMEMO         (29)
#define STATE_SHUTDOWN          (31)
#define STATE_TTS_SETTING       (32)
#define STATE_ASR_SETTING       (33)
#define STATE_LINGLONG_SETTING  (34)
#define STATE_LINGLONG_TIMELOC  (35)
#define STATE_FLIP              (36)


#define PREFILL_LED_ON  system("echo \"1\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define PREFILL_LED_OFF system("echo \"0\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define DECODE_LED_ON   system("echo \"1\" > /sys/devices/platform/leds/leds/blue:status/brightness");
#define DECODE_LED_OFF  system("echo \"0\" > /sys/devices/platform/leds/leds/blue:status/brightness");



// ===============================================================================
// 获取按键事件
// ===============================================================================

void get_key_event(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// 全局GUI+gfx初始化
// ===============================================================================

void ui_init(Key_Event *key_event, Global_State *global_state);

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
// FLIP流体模拟
// ===============================================================================

void ui_app_flip_init(Key_Event *key_event, Global_State *global_state);
void ui_app_flip_render_frame(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// 玲珑天象仪
// ===============================================================================

int32_t linglong_setting_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);
void ui_app_linglong_draw_lite(
    Key_Event *key_event, Global_State *global_state,
    int32_t x, int32_t y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double longitude, double latitude, double timezone
);
void ui_app_linglong_render_frame(Key_Event *key_event, Global_State *global_state);
void ui_app_linglong_toggle_timemachine(Key_Event *key_event, Global_State *global_state);
void ui_app_linglong_set_timemachine_speed(Key_Event *key_event, Global_State *global_state, int32_t speed);
void ui_app_linglong_set_realtime(Key_Event *key_event, Global_State *global_state);
void ui_app_linglong_event_handler(Key_Event *key_event, Global_State *global_state);


#ifdef __cplusplus
}
#endif

#endif
