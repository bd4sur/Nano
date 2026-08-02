// Linux 麦克风HAL：基于 ALSA（libasound）实现。
// 依赖：alsa-lib（Debian/Ubuntu: libasound2-dev；链接选项 -lasound）。
//
// 与 ESP32（mic_m5core2.cpp，ESP-IDF I2S 驱动）的语义对应关系：
// - mic_read 阻塞至数据就绪或超时（ESP32 为 i2s_channel_read 超时 100ms，
//   此处用 snd_pcm_wait 实现相同的 100ms 超时上限），允许返回部分采样；
// - ESP32 上 mic_init/mic_close 负责在麦克风与扬声器之间切换 I2S 外设；
//   Linux 上采集与播放是相互独立的 PCM 设备（dmix/dsnoop 自动混音），
//   无需切换，故 mic_close 仅需关闭采集句柄（保持幂等）；
// - 调用方可能在独立任务（线程）中调用 mic_read（见 ui_ofdm.c 采集任务），
//   ALSA 句柄本身可由单一线程安全读写，本实现满足该用法。

#include "platform.h"
#include "mic.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>

// 采集设备名：默认 "default"，可用环境变量 NANO_ALSA_DEVICE 覆盖
static const char *mic_device(void) {
    const char *dev = getenv("NANO_ALSA_DEVICE");
    return (dev && dev[0]) ? dev : "default";
}

// mic_read 单次等待数据的超时上限（ms），与 ESP32 i2s_channel_read 的 100ms 对齐
#define MIC_READ_TIMEOUT_MS (100)

static snd_pcm_t *s_cap = NULL;

int32_t mic_init(uint32_t sample_rate) {
    // 幂等：重复 init 先关闭旧句柄
    if (s_cap) {
        snd_pcm_close(s_cap);
        s_cap = NULL;
    }

    // 非阻塞模式打开，配合 snd_pcm_wait 实现“阻塞至就绪或超时”语义
    int err = snd_pcm_open(&s_cap, mic_device(), SND_PCM_STREAM_CAPTURE, SND_PCM_NONBLOCK);
    if (err < 0) {
        fprintf(stderr, "mic_init: snd_pcm_open(%s) failed: %s\n",
                mic_device(), snd_strerror(err));
        s_cap = NULL;
        return -1; // 无可用采集设备
    }

    err = snd_pcm_set_params(s_cap,
                             SND_PCM_FORMAT_S16_LE,
                             SND_PCM_ACCESS_RW_INTERLEAVED,
                             1,            // 单声道
                             sample_rate,
                             1,            // 允许软重采样
                             100000);      // 采集缓冲约 100ms，覆盖 UI 消费间隙
    if (err < 0) {
        fprintf(stderr, "mic_init: snd_pcm_set_params failed: %s\n", snd_strerror(err));
        snd_pcm_close(s_cap);
        s_cap = NULL;
        return -2;
    }

    snd_pcm_prepare(s_cap);
    return 0;
}

int32_t mic_read(int16_t *buffer, uint32_t samples) {
    if (!s_cap || !buffer || samples == 0) return -1;

    uint32_t got = 0;
    while (got < samples) {
        // 等待数据就绪，超时上限 MIC_READ_TIMEOUT_MS
        int w = snd_pcm_wait(s_cap, MIC_READ_TIMEOUT_MS);
        if (w == 0) break;    // 超时：返回已读到的部分（可能为 0）
        if (w < 0) {
            if (w == -EPIPE) {
                snd_pcm_prepare(s_cap); // 过载（overrun）：丢弃并重新开始采集
                continue;
            }
            return -2;
        }

        snd_pcm_sframes_t r = snd_pcm_readi(s_cap, buffer + got, samples - got);
        if (r == -EAGAIN) {
            continue;
        }
        if (r == -EPIPE) {
            snd_pcm_prepare(s_cap); // 过载恢复
            continue;
        }
        if (r == -ESTRPIPE) {
            snd_pcm_resume(s_cap);
            continue;
        }
        if (r < 0) return -2;
        got += (uint32_t)r;
    }

    return (int32_t)got;
}

int32_t mic_close() {
    // 幂等；Linux 上采集与播放互不占用，无需恢复扬声器
    if (s_cap) {
        snd_pcm_close(s_cap);
        s_cap = NULL;
    }
    return 0;
}
