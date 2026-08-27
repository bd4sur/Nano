#include "platform.h"
#include "hal_key.h"

// MP135 按键HAL：本机无实体键盘，触屏由 hal_touch_evdev_linux.c
// 经 /dev/input/event0 上报。触屏 → 4x4 宫格虚拟按键的兼容映射曾置于此处，
// 因旁路干净的触屏路径造成架构混乱，已上移至输入事件层（ui_app.c
// ui_app_map_touch_to_grid16_key）；本HAL只负责实体按键，故恒返回无键。

int32_t input_device_init() {
    return 0;
}

uint8_t input_device_read_key() {
    return NANO_KEY_IDLE;
}
