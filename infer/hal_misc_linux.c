// hal_misc Linux 实现（PC / 树莓派等普通 Linux 系统）。
// 普通 Linux 平台无指示灯 / 振动马达 / 蜂鸣器这类外设，全部为空操作，
// 仅用于满足链接，上层应将其视为无对应硬件可用。

#include "platform.h"
#include "hal_misc.h"

// ---------------- 指示灯 ----------------

void misc_led_init(void) {
}

void misc_led_set(int32_t on, int32_t color) {
    (void)on;
    (void)color;
}

void misc_led_blink(int32_t color, uint32_t duration_ms) {
    (void)color;
    (void)duration_ms;
}

// ---------------- 振动马达 ----------------

void set_vibration(uint32_t level) {
    (void)level;
}

// ---------------- 蜂鸣（扬声器 tone 提示音） ----------------

void misc_tone(uint32_t freq_hz, uint32_t duration_ms) {
    (void)freq_hz;
    (void)duration_ms;
}
