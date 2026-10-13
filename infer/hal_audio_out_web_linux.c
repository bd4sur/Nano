// hal_audio_out_web_linux.c —— Nano-Web 扬声器 HAL 打桩
//
// Nano-Web 为 C/S 分离的远程显示目标，服务端（本机）不一定有声卡，且网页薄客户端
// 不传输音频，故本 HAL 全部打桩：init/enqueue 恒失败，queue_free 恒无空槽。
// 各调用点均有失败处理（ui_musicbox 播放停在 0:00 无声、ui_ofdm 发射静默），
// UI 行为不受影响。主音量状态照常缓存（系统设置读写不空转）。

#include "hal_audio_out.h"

#include "platform.h"

static uint8_t s_master_volume = 16; // 与 ui_init 的 volume 初值一致

int32_t audio_out_init(uint32_t sample_rate, uint8_t volume) {
    (void)sample_rate;
    (void)volume;
    return -1; // 无音频输出设备
}

int32_t audio_out_queue_free(void) {
    return 0; // 恒无空槽（init 失败后调用方不会入队）
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
