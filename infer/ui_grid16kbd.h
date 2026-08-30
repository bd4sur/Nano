#ifndef __NANO_UI_GRID16KBD_H__
#define __NANO_UI_GRID16KBD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"
#include "graphics.h"
#include "hal_key.h"
#include "hal_touch.h"

// ===============================================================================
// 16键虚拟键盘（硬件无关）
//
// 文本输入控件"16键"模式的显式虚拟键盘：屏幕下半部分绘制 4x4 圆角矩形按钮，
// 替代旧的全屏 4x4 宫格隐式映射（get_input_event 中的 ui_app_map_touch_to_grid16_key
// 兼容适配层在文本输入场景下的继任者）。键位布局与实体十六键/旧宫格映射一致：
//   1   2   3   退格(ESC)
//   4   5   6   输入法(SFT)
//   7   8   9   Ctrl
//   ←   0   →   确认(ENT)
// 按钮内容同已移除的"按键提示遮罩"，随全局 Ctrl 状态（is_ctrl_enabled）切换两套文案。
//
// 实现范式与触屏软键盘（ui_softkbd）一致：
//   - 绘制只写帧缓冲区（ui_grid16kbd_draw），不调用 gfx_refresh，由调用方统一刷新；
//   - 触屏输入由事件层（get_input_event）统一采样后作为参数传入，不直接访问触屏HAL；
//   - 命中按钮产生 NANO_KEY_* 键码，走 get_input_event 统一的边沿/长按/连发机制，
//     对上层与消费者而言与实体十六键、旧宫格软按键完全等价（is_soft_key=1）。
//
// 使用方式：
//   - 事件侧（轮询任务）：可见时每次主循环由 get_input_event 调用
//     ui_grid16kbd_poll(x, y, is_pressed)；ui_grid16kbd_touch_claimed() 用于判断
//     当前触摸是否落在键盘区域内（键盘区域外的触屏不再产生任何宫格软按键）。
//   - 绘制侧（渲染任务）：ui_grid16kbd_draw() 把键盘画入帧缓冲；
//     ui_grid16kbd_take_dirty() 用于查询按下高亮是否变化。
//   - 显隐由文本输入控件页脚 [16键] 热点控制（ui_widget_input_toggle_grid16），
//     与全键盘软键盘互斥；显隐联动文本区/页脚/输入法候选区布局（ui_grid16kbd_height）。
// ===============================================================================

// 布局：4行x4列，与实体十六键布局一致；键盘区域靠屏幕下沿
#define UI_GRID16KBD_ROWS    (4)
#define UI_GRID16KBD_COLS    (4)
#define UI_GRID16KBD_HEIGHT  (150) //(SCREEN_HEIGHT / 2) // 键盘区域总高度（px），默认半屏

void    ui_grid16kbd_init();

uint8_t ui_grid16kbd_is_visible();
void    ui_grid16kbd_show();
void    ui_grid16kbd_hide();

// 键盘当前占用的屏幕高度：隐藏时为0，显示时为 UI_GRID16KBD_HEIGHT。
// UI布局（页脚、文本区高度等）统一调用本函数为键盘让出空间。
int32_t ui_grid16kbd_height();

// 触屏轮询（可见时每次主循环由 get_input_event 调用一次；触屏样本由事件层统一采样传入）：
// 返回当前键码（NANO_KEY_*），无键返回 NANO_KEY_IDLE。按下沿命中按钮时解析一次并锁存，
// 按住期间持续上报锁存键码（长按/连发交给 get_input_event 的既有机制，与旧宫格软按键一致）。
uint8_t ui_grid16kbd_poll(int32_t x, int32_t y, int32_t is_pressed);

// 当前触摸是否落在键盘区域内（含按钮间隙），供上层吞掉其他触屏映射
uint8_t ui_grid16kbd_touch_claimed();

// 按下高亮自上次查询后是否变化：变化过返回1并清除标记
uint8_t ui_grid16kbd_take_dirty();

// 把16键虚拟键盘绘制到帧缓冲区（不刷新屏幕，由调用方统一 gfx_refresh）
// is_ctrl_active：全局Ctrl激活状态（Global_State.is_ctrl_enabled），
// 据此切换按钮文案（普通/Ctrl两套）并高亮 Ctrl 键
void ui_grid16kbd_draw(Nano_GFX *gfx, uint8_t is_ctrl_active);

#ifdef __cplusplus
}
#endif

#endif
