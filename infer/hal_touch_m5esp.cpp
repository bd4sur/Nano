#include "platform.h"
#include "hal_touch.h"

#include <Arduino.h>
#include "M5Unified.h"

// M5Core2 触屏HAL实现：基于 M5Unified 的 M5.Touch。
// 注意：M5.update() 已在主循环（Core1）中周期性调用，getDetail() 读取的是
// M5Unified 缓存的最新触屏状态，因此此处与 input_device_m5core2.cpp 一样直接轮询即可。

int32_t touch_init() {
    // M5.begin() 已完成触屏初始化，无需额外操作
    return 0;
}

int32_t touch_read(int32_t *x, int32_t *y, int32_t *is_pressed) {
    m5::touch_detail_t touch_detail = M5.Touch.getDetail();
    if (touch_detail.isPressed()) {
        if (x) *x = touch_detail.x;
        if (y) *y = touch_detail.y;
        if (is_pressed) *is_pressed = 1;
    }
    else {
        if (is_pressed) *is_pressed = 0;
    }
    return 0;
}
