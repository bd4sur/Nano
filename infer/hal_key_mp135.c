#include "platform.h"
#include "hal_key.h"

// MP135 按键HAL：本机无实体键盘，触屏由 hal_touch_evdev_linux.c
// 经 /dev/input/event0 上报。触屏派生键码的映射曾置于此处（已随全屏宫格软键
// 整体退场一并消失）；本HAL只负责实体按键，本机型无实体键，故恒返回无键。

int32_t input_device_init() {
    return 0;
}

uint8_t input_device_read_key() {
    return NANO_KEY_IDLE;
}
