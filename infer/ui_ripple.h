#ifndef __NANO_UI_RIPPLE_H__
#define __NANO_UI_RIPPLE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "ui.h"
#include "ui_app.h"

// ===============================================================================
// 水波（Water Ripple）
//
// 移植自 ripple.html（youyouzh/water_ripple）：在二维振幅场上模拟波的传播与衰减，
// 按振幅对背景图做位移采样，产生水面折射般的涟漪效果。
//
// 每次进入本功能时从 SD 卡读取固定文件 /wp.png，解码并缩放至屏幕分辨率
//（320x240）存入内存缓冲区作为背景纹理；退出时释放全部内存。
//
// 实现包含定点/浮点双版本，由 ui_ripple.c 顶部 WR_USE_FIXED_POINT 宏切换
//（物理常数统一以浮点形式给出，定点版换算为移位运算）；定点版针对 ESP32 无 FPU
// 优化（int16 振幅场，PSRAM 带宽减半 + 全部移位运算 + 行指针直写帧缓冲），
// 浮点版保留作算法可读性参考（JS 原版的直译）。
//
// 玩法：
//   - 触摸屏幕，即在触摸点激发水波纹；按住拖动可持续激发
//    （等价于原网页的 click + mousemove 两个监听器）；
//   - 按 A(ESC) 返回小游戏菜单（注意触屏 16 宫格映射使触摸右上角等效于 A 键）。
// ===============================================================================

int32_t ui_ripple_init(Key_Event *key_event, Global_State *global_state);
int32_t ui_ripple_event_handler(Key_Event *key_event, Global_State *global_state);
int32_t ui_ripple_render_frame(Key_Event *key_event, Global_State *global_state);
void    ui_ripple_on_exit(void);

#ifdef __cplusplus
}
#endif

#endif
