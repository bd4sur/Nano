#include "platform.h"
#include "mic.h"

// Linux 麦克风HAL打桩实现：本平台无麦克风硬件，仅用于满足链接。
// 所有接口返回失败/空操作，上层应将其视为无麦克风可用。

int32_t mic_init() {
    return -1;  // 无麦克风硬件
}

int32_t mic_read(int16_t *buffer, uint32_t samples) {
    (void)buffer;
    (void)samples;
    return -1;
}

int32_t mic_close() {
    return 0;
}
