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
#define STATE_GENETIC           (23)
#define STATE_GENETIC_TSP       (24)
#define STATE_README            (25)
#define STATE_BADAPPLE          (26)
#define STATE_GAMEOFLIFE        (27)
#define STATE_LINGLONG          (28)
#define STATE_FLASHMEMO         (29)
#define STATE_TORCH             (30)
#define STATE_ALBUM             (32)
#define STATE_ASR_SETTING       (33)
#define STATE_SETTING_INPUT     (35)
#define STATE_FLIP              (36)
#define STATE_PEDOMETER         (37)
#define STATE_ANIMAC_INIT       (50)
#define STATE_ANIMAC_CONSOLE    (51)
#define STATE_ANIMAC_RUNNING    (52)
#define STATE_SPECTROGRAM       (53)
#define STATE_GAME_MENU         (54)
#define STATE_GOLDMINER         (55)
#define STATE_TETRIS            (56)
#define STATE_EBOOK_READING     (57)
#define STATE_OFDM_MENU         (58)
#define STATE_OFDM_TX           (59)
#define STATE_OFDM_RX           (60)
#define STATE_OFDM_TXING        (61)
#define STATE_OFDM_LOOP         (62)
#define STATE_OFDM_LOOPING      (63)
#define STATE_MUSICBOX_MENU     (64)
#define STATE_MUSICBOX_PLAYING  (65)
#define STATE_PARTICLELIFE      (66)
#define STATE_RIPPLE            (67)
#define STATE_DICT_QUERY        (68)
#define STATE_DICT_DETAIL       (69)
#define STATE_CALENDAR          (70)
#define STATE_WATER             (71)
#define STATE_CLOUD             (72)
// 编号 73~76 已退役：原“小鹦鹉笼”独立的模型菜单/输入/推理/结果状态已并入鹦鹉笼统一状态
// （STATE_MODEL_MENU / STATE_LLM_INPUT / STATE_LLM_ON_INFER / STATE_LLM_AFTER_INFER），
// 仅推理引擎不同（nano_min，见 ui_llm.c 引擎适配层）；保留编号空缺避免冲突
#define STATE_SHUTDOWN          (99)


#define PREFILL_LED_ON  system("echo \"1\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define PREFILL_LED_OFF system("echo \"0\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define DECODE_LED_ON   system("echo \"1\" > /sys/devices/platform/leds/leds/blue:status/brightness");
#define DECODE_LED_OFF  system("echo \"0\" > /sys/devices/platform/leds/leds/blue:status/brightness");



#define FLIP_RESOLUTION (24)


// ===============================================================================
// UI框架：获取按键事件
// ===============================================================================

void get_key_event(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// UI框架：全局GUI+gfx初始化
// ===============================================================================

void ui_init(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// 核心业务：电子鹦鹉（LLM 推理及其可视化）——已提取至独立模块 ui_llm（ui_llm.h/ui_llm.c），
// 使用方（ui_app.c 等）直接包含 ui_llm.h
// ===============================================================================

// ===============================================================================
// 小游戏菜单
// ===============================================================================

void init_game_menu(Key_Event *key_event, Global_State *global_state);
int32_t game_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms);

// ===============================================================================
// 主菜单
// ===============================================================================

void ui_widget_grid16_draw(Key_Event *key_event, Global_State *global_state);
void ui_widget_grid16_event_handler(Key_Event *key_event, Global_State *global_state);

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

void ui_app_gol_init(Key_Event *key_event, Global_State *global_state, int32_t gol_width, int32_t gol_height);
void ui_app_gol_render_frame(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// FLIP流体模拟
// ===============================================================================

void ui_app_flip_init(Key_Event *key_event, Global_State *global_state);
void ui_app_flip_render_frame(Key_Event *key_event, Global_State *global_state);
void ui_app_flip_event_handler(Key_Event *key_event, Global_State *global_state);


// ===============================================================================
// 玲珑天象仪
// ===============================================================================

void ui_app_linglong_init(Key_Event *key_event, Global_State *global_state);
void ui_app_linglong_setting_draw(Key_Event *key_event, Global_State *global_state);
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


// ===============================================================================
// 设置菜单
// ===============================================================================

void ui_app_setting_value_input_draw(Key_Event *key_event, Global_State *global_state, int32_t value_type, wchar_t *value_text, int32_t cursor_pos);




// ===============================================================================
// UI主体框架
// ===============================================================================

int32_t main_init(Key_Event *key_event, Global_State *global_state);
int32_t main_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t main_periodic_task(Key_Event *key_event, Global_State *global_state);
int32_t main_deinit(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
