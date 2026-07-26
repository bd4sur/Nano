#ifndef __NANO_TOUCH_H__
#define __NANO_TOUCH_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

// 触屏HAL：硬件无关的触屏状态读取接口。
// 各平台在对应的 touch_<platform>.cpp 中实现本接口（如 touch_m5core2.cpp）。
// 使用方式：每次轮询调用 touch_read 读取当前触屏状态（坐标+是否按下），
// 边沿检测等逻辑由上层（如 ui_softkbd）自行实现，与 input_device 的轮询机制一致。

int32_t touch_init();

// 读取当前触屏状态。
//   x / y       ：触点坐标（像素），可为NULL表示不关心；未按下时不保证有效
//   is_pressed  ：1-正在触摸，0-未触摸
// 返回值：0-正常；负值-无触屏硬件或读取失败
int32_t touch_read(int32_t *x, int32_t *y, int32_t *is_pressed);

#ifdef __cplusplus
}
#endif

#endif
