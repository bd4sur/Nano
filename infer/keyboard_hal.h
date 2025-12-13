#ifndef __NANO_KEYBOARD_H__
#define __NANO_KEYBOARD_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

// 逻辑键值定义
// ITU E.161 12键电话键盘+4个额外案件=16按键
#define KEYCODE_NUM_0 (0)
#define KEYCODE_NUM_1 (1)
#define KEYCODE_NUM_2 (2)
#define KEYCODE_NUM_3 (3)
#define KEYCODE_NUM_4 (4)
#define KEYCODE_NUM_5 (5)
#define KEYCODE_NUM_6 (6)
#define KEYCODE_NUM_7 (7)
#define KEYCODE_NUM_8 (8)
#define KEYCODE_NUM_9 (9)
#define KEYCODE_NUM_A (10)
#define KEYCODE_NUM_B (11)
#define KEYCODE_NUM_C (12)
#define KEYCODE_NUM_D (13)
#define KEYCODE_NUM_STAR (14)
#define KEYCODE_NUM_HASH (15)
#define KEYCODE_NUM_IDLE (16)

int32_t keyboard_hal_init();
uint8_t keyboard_hal_read_key();

#ifdef __cplusplus
}
#endif

#endif