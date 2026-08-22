#ifndef __NANO_UI_SOFTKBD_H__
#define __NANO_UI_SOFTKBD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"
#include "graphics.h"
#include "hal_key.h"
#include "hal_touch.h"

// ===============================================================================
// 触屏软键盘（硬件无关）
//
// 参照 main.cpp-ref 的软键盘实现，但完全基于本项目自制GFX框架（graphics.c）：
//   - 绘制只写帧缓冲区（ui_softkbd_draw），不调用 gfx_refresh，由调用方统一全量刷新；
//   - 触屏输入通过触屏HAL（hal_touch.h）轮询获得，不依赖具体硬件库；
//   - 键码映射到 hal_key.h 的 NANO_KEY_* 定义（可打印字符即ASCII码）。
//
// 使用方式：
//   - 事件侧（轮询任务）：每次主循环调用 ui_softkbd_poll()，返回按下沿键码
//     （NANO_KEY_IDLE 表示无新键）；ui_softkbd_touch_claimed() 用于判断当前
//     触摸是否落在键盘区域内（供上层吞掉其他触屏映射，如4x4网格键）。
//   - 绘制侧（渲染任务）：ui_softkbd_draw() 把键盘画入帧缓冲；
//     ui_softkbd_take_dirty() 用于查询键盘自身状态（粘滞键/按下高亮）是否变化。
// ===============================================================================

// 布局：4行x12列，与 hal_key.h 头部注释的48键物理键盘布局一致
#define UI_SOFTKBD_ROWS    (4)
#define UI_SOFTKBD_COLS    (12)
#define UI_SOFTKBD_HEIGHT  (128) // 键盘总高度（px），靠屏幕下沿

void    ui_softkbd_init();

uint8_t ui_softkbd_is_visible();
void    ui_softkbd_show();
void    ui_softkbd_hide();

// 请求切换键盘显隐（供上滑手势识别等轮询侧逻辑调用，由控制台状态机消费）
void    ui_softkbd_request_toggle();
// 有未消费的切换请求时返回1并清除
uint8_t ui_softkbd_take_toggle_request();

// 键盘当前占用的屏幕高度：隐藏时为0，显示时为 UI_SOFTKBD_HEIGHT。
// UI布局（页脚、文本区高度等）统一调用本函数为键盘让出空间。
int32_t ui_softkbd_height();

// 轮询触屏（每次主循环调用一次）：返回当前键码（NANO_KEY_*），无键返回 NANO_KEY_IDLE。修饰键行为：
//   - SFT：切换大写粘滞态，同时产生一次 NANO_KEY_shift 传递给UI框架（两者耦合）；
//     框架对软键盘来源的Shift不切换输入模式（软键盘模式下用 Ctrl+空格 切换），仅在Ctrl激活时显示帮助；
//   - SYM：点按切换符号层粘滞态，不产生键码；
//   - CTRL/ALT：点按产生 NANO_KEY_ctrl / NANO_KEY_alt，复用UI框架原生Ctrl机制。
// 按住重复触发：除 Ctrl/SFT/Alt/SYM/Esc 外，其余键在按住期间持续上报锁存键码
// （按下沿解析一次，保证按住期间键码不变），交给框架原生的长按/连发机制
// （get_key_event 的 360ms 长按判定与连续触发，与4x4网格键行为一致）。
uint8_t ui_softkbd_poll();

// 当前触摸是否落在软键盘区域内（含键盘上沿之外的容忍带），供上层吞掉其他触屏映射
uint8_t ui_softkbd_touch_claimed();

// 键盘自身状态（粘滞修饰键、按下高亮）自上次查询后是否变化：变化过返回1并清除标记
uint8_t ui_softkbd_take_dirty();

// 把软键盘绘制到帧缓冲区（不刷新屏幕，由调用方统一 gfx_refresh）
// is_ctrl_active：全局Ctrl激活状态（Global_State.is_ctrl_enabled），CTRL键据此显示高亮底色
void ui_softkbd_draw(Nano_GFX *gfx, uint8_t is_ctrl_active);

#ifdef __cplusplus
}
#endif

#endif
