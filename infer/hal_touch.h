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

// 取走并清零【累计滚轮/触控板垂直增量】（帧缓冲像素，正 = 内容上滚）。
// 只有带回滚轮输入的平台实现（当前为 WASM GUI：页面 wheel 事件 →
// wasm_gui_on_wheel）；无滚轮的平台不定义本函数，调用点由
// NANO_WASM_GUI 宏包住，不会产生未定义引用。
// 为何不合成成触屏事件：滚轮是“直接定位”语义，而合成手指拖动受
// “手指走不出画布”限制（可用行程只有一屏），长文档必须反复松手重按，
// 重按与松开若落在同一帧还会被后端当成一次点按（菜单直接执行中心条）。
int32_t hal_touch_wheel_take(void);

#ifdef __cplusplus
}
#endif

#endif
