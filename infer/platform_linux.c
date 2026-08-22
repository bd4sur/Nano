// Linux 平台抽象层实现（PC / 树莓派等普通 Linux 系统）。
//
// ram / fs / os / misc 的 Linux 实现已分别拆分至
// hal_ram_linux.c / hal_fs_linux.c / hal_os_linux.c / hal_misc_linux.c，
// 本文件仅保留尚未归入各 HAL 模块的全局主音量状态。

#include "platform.h"

// 主音量（0~255）：与 ui_app.c 中 global_state->volume 初值一致的默认值。
// 进程内静态保存；实际出声增益由 hal_audio_out_linux.c 在 init/set_volume 时应用。
static uint8_t s_master_volume = 16;

void platform_set_master_volume(uint8_t volume) {
    s_master_volume = volume;
}

uint8_t platform_get_master_volume(void) {
    return s_master_volume;
}
