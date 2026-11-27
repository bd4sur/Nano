#ifndef __NANO_UI_LINGLONG_H__
#define __NANO_UI_LINGLONG_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 玲珑天象仪（STATE_LINGLONG 的状态页）
//
// 状态页主体：配置读写、渲染帧（星图 / 海洋天空，走专用画布 llgfx）、事件处理
// （触屏控件编排 + 实体键盘兜底）、时光机（时间机器）与视角漫游状态机、
// 5x4 设置界面的动作语义。
//
// 渲染与天文计算在别处：
//   celestial.c / celestial_ocean.c   render_sky / linglong_ocean_*（星图、大气散射、海洋）
//   ephemeris.c                       日月位置、月相、日出日落
//   ui_linglong_ctrl.c                叠加在主界面上的显式触屏控件（只绘制与命中判定）
// ===============================================================================

// 全局资源初始化（main_init 调用一次）：Linglong_Config + 专用画布 llgfx
void ui_linglong_global_init(Global_State *global_state);

// 进入 STATE_LINGLONG 首帧初始化（子状态复位 + 控件复位）
void ui_linglong_init(Key_Event *key_event, Global_State *global_state);
void ui_linglong_render_frame(Key_Event *key_event, Global_State *global_state);
void ui_linglong_event_handler(Key_Event *key_event, Global_State *global_state);

// 设置界面（5x4 显式触屏按钮区）整页绘制
void ui_linglong_setting_draw(Key_Event *key_event, Global_State *global_state);

// 启动画面的“玲珑仪青春版”缩略星图（当前调用点已注释保留，接口留作备用）
void ui_linglong_draw_lite(
    Key_Event *key_event, Global_State *global_state,
    int32_t x, int32_t y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double longitude, double latitude, double timezone
);

// 时光机（时间机器）：启停切换 / 设定速度（正=未来，负=过去，单位秒）/ 回到实时
void ui_linglong_toggle_timemachine(Key_Event *key_event, Global_State *global_state);
void ui_linglong_set_timemachine_speed(Key_Event *key_event, Global_State *global_state, int32_t speed);
void ui_linglong_set_realtime(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
