// CoreMP135 麦克风HAL：基于 tinyalsa 实现（参照 M5Stack_Linux_Libs BSP
// tinyalsa_example 中 tinycap 的方式，直接操作声卡 hw 节点，无 ALSA 插件层）。
// 依赖：tinyalsa（头文件 tinyalsa/pcm.h；链接选项 -ltinyalsa）。
//
// 与 hal_audio_in_alsa_linux.c 语义一致，对应关系：
// - mic_read 阻塞至数据就绪或超时（ESP32 为 i2s_channel_read 超时 100ms，
//   此处用 pcm_wait 实现相同的 100ms 超时上限），允许返回部分采样；
// - 过载（overrun/XRUN）由 tinyalsa 内部自动 prepare 恢复；pcm_wait 报告
//   XRUN 时此处同样 prepare 后重试；
// - MP135 上采集与播放是相互独立的 PCM 设备（同一声卡的不同流），
//   无需在麦克风与扬声器间切换外设，故 mic_close 仅需关闭采集句柄（保持幂等）；
// - 调用方可能在独立任务（线程）中调用 mic_read（见 ui_ofdm.c 采集任务），
//   tinyalsa 句柄本身可由单一线程安全读写，本实现满足该用法。

#include "platform.h"
#include "hal_audio_in.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <tinyalsa/pcm.h>

// 声卡号/设备号：默认 0/0（BSP tinycap 的默认值），
// 可用环境变量 NANO_PCM_CARD / NANO_PCM_DEVICE 覆盖
static unsigned int mic_card(void) {
    const char *s = getenv("NANO_PCM_CARD");
    return (s && s[0]) ? (unsigned int)atoi(s) : 0;
}

static unsigned int mic_device(void) {
    const char *s = getenv("NANO_PCM_DEVICE");
    return (s && s[0]) ? (unsigned int)atoi(s) : 0;
}

// mic_read 单次等待数据的超时上限（ms），与 ESP32 i2s_channel_read 的 100ms 对齐
#define MIC_READ_TIMEOUT_MS (100)

// 采集缓冲：4 个 period × 1024 帧 ≈ 85ms @48kHz，覆盖 UI 消费间隙
#define MIC_PERIOD_FRAMES (1024)
#define MIC_PERIOD_COUNT  (4)

static struct pcm *s_cap = NULL;

int32_t mic_init(uint32_t sample_rate, uint8_t restore_volume) {
    (void)restore_volume; // MP135 上采集与播放互不占用，mic_close 无需恢复扬声器音量

    // 幂等：重复 init 先关闭旧句柄
    if (s_cap) {
        pcm_close(s_cap);
        s_cap = NULL;
    }

    // 单声道 S16_LE；tinyalsa 直接操作 hw 节点，不支持软重采样，
    // 采样率须为声卡硬件支持的值（44100/48000 等）
    struct pcm_config config;
    memset(&config, 0, sizeof(config));
    config.channels     = 1;
    config.rate         = sample_rate;
    config.period_size  = MIC_PERIOD_FRAMES;
    config.period_count = MIC_PERIOD_COUNT;
    config.format       = PCM_FORMAT_S16_LE;

    // 非阻塞模式打开，配合 pcm_wait 实现“阻塞至就绪或超时”语义
    s_cap = pcm_open(mic_card(), mic_device(), PCM_IN | PCM_NONBLOCK, &config);
    if (!s_cap || !pcm_is_ready(s_cap)) {
        fprintf(stderr, "mic_init: pcm_open(card=%u,dev=%u,rate=%u) failed: %s\n",
                mic_card(), mic_device(), sample_rate,
                s_cap ? pcm_get_error(s_cap) : "out of memory");
        if (s_cap) {
            pcm_close(s_cap);
            s_cap = NULL;
        }
        return -1; // 无可用采集设备或参数不受支持
    }

    pcm_prepare(s_cap);
    return 0;
}

int32_t mic_read(int16_t *buffer, uint32_t samples) {
    if (!s_cap || !buffer || samples == 0) return -1;

    uint32_t got = 0;
    while (got < samples) {
        // 等待数据就绪，超时上限 MIC_READ_TIMEOUT_MS
        int w = pcm_wait(s_cap, MIC_READ_TIMEOUT_MS);
        if (w == 0) break; // 超时：返回已读到的部分（可能为 0）
        if (w < 0) {
            if (w == -EPIPE) {
                pcm_prepare(s_cap); // 过载（overrun）：丢弃并重新开始采集
                continue;
            }
            return -2;
        }

        int r = pcm_readi(s_cap, buffer + got, samples - got);
        if (r < 0) {
            if (errno == EAGAIN) continue;
            // EPIPE/ESTRPIPE 已由 tinyalsa 内部自动 prepare 恢复
            return -2;
        }
        got += (uint32_t)r;
    }

    return (int32_t)got;
}

int32_t mic_close() {
    // 幂等；MP135 上采集与播放互不占用，无需恢复扬声器
    if (s_cap) {
        pcm_close(s_cap);
        s_cap = NULL;
    }
    return 0;
}
