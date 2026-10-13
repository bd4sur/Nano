// hal_audio_out_wasm.c —— Nano-WASM-GUI 扬声器 HAL 打桩
//
// WASM 平台无音频输出通路（Web Audio 接入属「做不到就不做」项），全部打桩。
// 各调用点（ui_musicbox/ui_ofdm）均有失败处理（无声但 UI 正常）。
// 主音量状态照常缓存（系统设置读写不空转）。

#include "hal_audio_out.h"

#include "platform.h"

static uint8_t s_master_volume = 16; // 与 ui_init 的 volume 初值一致

int32_t audio_out_init(uint32_t sample_rate, uint8_t volume) {
    (void)sample_rate;
    (void)volume;
    return -1;
}

int32_t audio_out_queue_free(void) {
    return 0;
}

int32_t audio_out_enqueue(const int16_t *pcm, uint32_t samples) {
    (void)pcm;
    (void)samples;
    return -1;
}

void audio_out_stop(void) {
    return;
}

void audio_out_set_volume(uint8_t volume) {
    (void)volume;
}

void audio_out_set_master_volume(uint8_t volume) {
    s_master_volume = volume;
}

uint8_t audio_out_get_master_volume(void) {
    return s_master_volume;
}

// platform.h 声明的全局主音量接口（其余平台由各自 audio_out HAL 实现，对齐之）
void platform_set_master_volume(uint8_t volume) {
    audio_out_set_master_volume(volume);
}

uint8_t platform_get_master_volume(void) {
    return audio_out_get_master_volume();
}

void audio_out_close(void) {
    return;
}
