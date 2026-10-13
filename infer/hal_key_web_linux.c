// hal_key_web_linux.c —— Nano-Web 键盘 HAL 打桩
//
// Nano-Web 平台无实体键盘（platform.h：NANO_HAS_HW_KEYBOARD=0），全部交互经
// 网页客户端触屏事件（行为对齐 M5Core2 触屏-only 平台：文本输入走触屏软键盘
// ui_softkbd / 16键虚拟键盘 ui_grid16kbd，主菜单等走 4x4 宫格触屏映射）。

#include "hal_key.h"

#include "platform.h"

int32_t input_device_init() {
    return 0;
}

uint8_t input_device_read_key() {
    return NANO_KEY_IDLE;
}
