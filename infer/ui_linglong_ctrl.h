#ifndef __NANO_UI_LINGLONG_CTRL_H__
#define __NANO_UI_LINGLONG_CTRL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "graphics.h"

// ===============================================================================
// 玲珑天象仪主界面显式触屏控件（硬件无关）
//
// 替代主天空界面原有的全屏 4x4 宫格隐式软按键（“软 key”）：
//   - 顶部靠右一排按钮：控制 / 设置 / 退出；
//   - “控制”同时切换两组控件的显隐（同显同隐）：
//       · 底部四分之一区域的四个时间控制按钮：
//           <<（快退） 实时（恢复当前时间） 运行/暂停（时间流逝开关） >>（快进）；
//       · 画面中央的 3x3 半透明九宫格，功能与十六宫格软键 1~9 一一对应；
//   - “设置”等价于十六宫格 C 键（Ctrl），打开现有玲珑仪设置界面；
//   - “退出”等价于十六宫格 A 键（Esc），退出玲珑仪。
//
// 设计约束（与项目触屏化改造的既有范式一致）：
//   - 只绘制到调用方传入的帧缓冲，不调用 gfx_refresh；
//   - 不直接访问触屏 HAL：触屏样本由事件层（ui_app.c）统一采样后作为参数传入；
//   - 命中判定与绘制共用同一套几何常量，保证严格一致；
//   - 控件只在玲珑仪主天空界面使用；设置界面继续沿用十六宫格软键（遗留兜底），
//     实体硬键在两种界面下均照常工作。
// ===============================================================================

// 控件动作（ui_linglong_ctrl_handle_touch 的返回值，NONE 表示无动作）
#define UI_LL_CTRL_NONE       (0)
#define UI_LL_CTRL_CONTROLS   (1)  // 同时切换时间控制按钮与视角九宫格的显隐
#define UI_LL_CTRL_SETTINGS   (3)  // 等价于十六宫格 C 键：打开设置界面
#define UI_LL_CTRL_EXIT       (4)  // 等价于十六宫格 A 键：退出玲珑仪
#define UI_LL_CTRL_REWIND     (5)  // 时间快退 <<
#define UI_LL_CTRL_REALTIME   (6)  // 恢复当前时间
#define UI_LL_CTRL_RUNPAUSE   (7)  // 运行/暂停（时间流逝开关）
#define UI_LL_CTRL_FORWARD    (8)  // 时间快进 >>
// 视角九宫格：UI_LL_CTRL_VIEW_1 + (n-1)，n=1..9，对应软键 NANO_KEY_1..9
#define UI_LL_CTRL_VIEW_1     (10)
#define UI_LL_CTRL_VIEW_9     (18)

// 初始化/复位：隐藏控制面板（时间按钮 + 视角九宫格）并清除按住高亮
void    ui_linglong_ctrl_init(void);
void    ui_linglong_ctrl_reset(void);

// 触屏事件处理（由玲珑仪事件处理器在主天空界面调用）：
//   touch_edge：TOUCH_EDGE_DOWN / TOUCH_EDGE_UP 位掩码（同帧可叠加）；
//   x,y：当前触点（松开沿为最后有效触点）；down_x,down_y：本次触摸序列按下点。
// 按下沿只锁存高亮，松开沿在按下点与松开点命中同一按钮时返回对应动作。
// 返回 UI_LL_CTRL_*；无动作返回 UI_LL_CTRL_NONE。
int32_t ui_linglong_ctrl_handle_touch(int32_t touch_edge, int32_t x, int32_t y,
                                      int32_t down_x, int32_t down_y);

// 主界面拖动视角（仅在控制面板隐藏、且按下点不在顶部按钮上时启用）：
// 输出自上次查询以来触点的像素增量（帧缓冲坐标）；返回1表示有拖动，0表示无。
// 本模块只负责“是否允许拖动/增量多少”，角度换算由调用方按实际画布尺寸完成，
// 不在此写死屏幕宽高。
int32_t ui_linglong_ctrl_poll_drag(int32_t touch_edge, int32_t is_touching,
                                   int32_t x, int32_t y,
                                   int32_t down_x, int32_t down_y,
                                   int32_t *out_dx, int32_t *out_dy);

// 控件显示状态自上次查询后是否变化（面板显隐或按住高亮变化）：变化过返回1并清除标记。
// 调用方据此决定是否需要重绘（避免点击空白天空也强制重绘整幅星图）。
uint8_t ui_linglong_ctrl_take_dirty(void);

// 绘制控件（不刷新屏幕，由调用方统一 gfx_refresh）。
// running_state/speed 用于时间按钮的状态高亮（语义同 ui_app.c 的时光机状态）。
void    ui_linglong_ctrl_draw(Nano_GFX *gfx, int32_t running_state, int32_t speed);

#ifdef __cplusplus
}
#endif

#endif
