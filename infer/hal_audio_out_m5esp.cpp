#include "platform.h"
#include "hal_audio_out.h"

#include <Arduino.h>
#include "M5Unified.h"

// 扬声器原始PCM流式播放HAL——M5Core2/CoreS3 实现（M5Unified Speaker，板型差异已封装）。
// Core2 为 NS4168（I2S_NUM_0），CoreS3 为 AW88298（I2S_NUM_1，与麦克风共用），
// M5.Speaker API 两平台一致，无需平台宏。
//
// 要点（排障记录，2026-07-30）：
// - M5Unified 的 8 个"通道"是并行混音关系：多通道同时播放会被混音器叠加，
//   绝不能用作顺序播放队列（曾误用 3 通道轮换播放 OFDM 帧，导致 3 帧信号
//   被叠加发射，接收端训练符号相关度仅 ~1/3，完全无法解调）。
// - 顺序无缝播放的正确方式：同一通道（ch0）内置 wavinfo 双槽队列——
//   一块在播、一块待播，段尾由混音器任务在采样边界无缝切换；
//   isPlaying(ch) 返回占用槽数（0/1/2），<2 即可继续投入。
// - playRaw 引用而非拷贝PCM数据（见 Speaker_Class.hpp 注释），
//   调用方须乒乓双缓冲轮换，覆写前确认对应块已播完（由队列空槽保证）。
// - 混音器任务优先级 2（高于 core0 渲染任务的 1），不会被 UI 饿死；
//   M5Core2 扬声器 I2S 输出 48kHz（M5Unified 默认），与 OFDM 采样率一致，
//   输入输出同速率时线性插值重采样逐采样精确（1:1）。

#define AUDIO_OUT_CHANNEL (0) // 固定使用通道0

static uint32_t s_sample_rate = 48000;
static uint8_t s_prev_volume = 12; // 与 setup() 中一致

int32_t audio_out_init(uint32_t sample_rate, uint8_t volume) {
    s_sample_rate = sample_rate;
    s_prev_volume = M5.Speaker.getVolume();
    if (!M5.Speaker.isEnabled()) {
        M5.Speaker.begin();
    }
    M5.Speaker.setVolume(volume);
    return 0;
}

int32_t audio_out_queue_free(void) {
    return (M5.Speaker.isPlaying(AUDIO_OUT_CHANNEL) < AUDIO_OUT_QUEUE_DEPTH) ? 1 : 0;
}

int32_t audio_out_enqueue(const int16_t *pcm, uint32_t samples) {
    if (!pcm || samples == 0) return -1;
    bool ok = M5.Speaker.playRaw(pcm, samples, s_sample_rate, false, 1, AUDIO_OUT_CHANNEL, false);
    return ok ? 0 : -2;
}

void audio_out_stop(void) {
    M5.Speaker.stop(AUDIO_OUT_CHANNEL);
}

void audio_out_set_volume(uint8_t volume) {
    M5.Speaker.setVolume(volume);
}

// 系统级扬声器主音量（全局缓存 + 应用到硬件；按键音/OFDM 发射/音乐盒/mic 恢复共用）
static uint8_t s_master_volume = 16; // 与 ui_init 的 volume 初值一致

void audio_out_set_master_volume(uint8_t volume) {
    s_master_volume = volume;
    if (M5.Speaker.isEnabled()) {
        M5.Speaker.setVolume(volume);
    }
}

uint8_t audio_out_get_master_volume(void) {
    return s_master_volume;
}

void audio_out_close(void) {
    M5.Speaker.stop(AUDIO_OUT_CHANNEL);
    M5.Speaker.setVolume(s_prev_volume);
}
