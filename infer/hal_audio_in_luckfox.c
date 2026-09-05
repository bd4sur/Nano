// Luckfox-Pico-86-Panel 麦克风HAL：基于 tinyalsa 实现（直接操作声卡 hw 节点，
// 无 ALSA 插件层）。依赖：tinyalsa（头文件 tinyalsa/pcm.h、mixer.h；链接 -ltinyalsa）。
//
// 硬件链路（RV1106G3 内置 ACodec，详见原理图与 Rockchip RV1106 ACodec 开发指南）：
// - 贴片模拟麦克风接 MIC0（差分对 MIC0_P/MIC0_N，由 CODEC_MICBIAS 偏置供电），
//   对应 ADC 左声道；MX1.25-2P 麦克风扩展口为 MIC1（ADC 右声道），默认不使用；
// - 声卡 rv1106-acodec（hw:0,0），capture 为 /dev/snd/pcmC0D0c；
// - 采样率支持 8k/12k/16k/24k/32k/44.1k/48k/96k，单声道 S16_LE。
//
// 与 hal_audio_in_mp135.c 语义一致（pcm_wait 100ms 超时上限、过载自动恢复、
// 采集与播放为独立 PCM 流无需切换外设），差异仅在：
// - init 时经 tinyalsa mixer API 初始化 ACodec 采集通路 mixer
//   （打开 MICBIAS 偏置、左 MIC 取消静音并设模拟增益 20dB、数字增益 0dB、
//   ADC Mode 固定 DiffadcL——贴片麦克风所在通道），避免外部程序改动 mixer
//   后采集无声；mixer 设置失败不影响主流程（仅告警）。
//
// 已实测：arecord 采集扬声器播放的 440Hz 正弦（声学回环），左声道信噪比约
// 5 个数量级，硬件通路（MICBIAS 偏置 + 差分 ADC + I2S 回读）完好。

#include "platform.h"
#include "hal_audio_in.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tinyalsa/pcm.h>
#include <tinyalsa/mixer.h>

// 声卡号/设备号：默认 0/0（rv1106-acodec 为卡0设备0），
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

// 立体声采集暂存（I2S0 不支持单声道，须以 2 声道打开，mic_read 取左声道还原单声道）
static int16_t s_stereo_buf[MIC_PERIOD_FRAMES * 2];

// 按名字设置整型/枚举 mixer 控件（控件不存在或设置失败仅告警，不致命）
static void luckfox_mixer_set_int(struct mixer *mx, const char *name, int value) {
    struct mixer_ctl *ctl = mixer_get_ctl_by_name(mx, name);
    if (!ctl) {
        fprintf(stderr, "mic_init: mixer ctl '%s' not found\n", name);
        return;
    }
    if (mixer_ctl_set_value(ctl, 0, value) < 0) {
        fprintf(stderr, "mic_init: set mixer ctl '%s'=%d failed\n", name, value);
    }
}

static void luckfox_mixer_set_enum(struct mixer *mx, const char *name, const char *value) {
    struct mixer_ctl *ctl = mixer_get_ctl_by_name(mx, name);
    if (!ctl) {
        fprintf(stderr, "mic_init: mixer ctl '%s' not found\n", name);
        return;
    }
    if (mixer_ctl_set_enum_by_string(ctl, value) < 0) {
        fprintf(stderr, "mic_init: set mixer ctl '%s'='%s' failed\n", name, value);
    }
}

// 初始化 RV1106 ACodec 采集通路（贴片麦克风在 MIC0 = ADC 左声道，差分模式）
static void luckfox_acodec_capture_setup(void) {
    struct mixer *mx = mixer_open(mic_card());
    if (!mx) {
        fprintf(stderr, "mic_init: mixer_open(card=%u) failed\n", mic_card());
        return;
    }
    luckfox_mixer_set_enum(mx, "ADC Main MICBIAS", "On");     // 驻极体麦克风偏置供电
    luckfox_mixer_set_enum(mx, "ADC MIC Left Switch", "Work"); // 左 MIC 取消静音
    luckfox_mixer_set_int(mx,  "ADC MIC Left Gain", 2);        // 模拟 Boost +20dB
    luckfox_mixer_set_int(mx,  "ADC ALC Left Volume", 6);      // ALC PGA 0dB
    luckfox_mixer_set_int(mx,  "ADC Digital Left Volume", 195); // 数字增益 0dB
    luckfox_mixer_set_enum(mx, "ADC Mode", "DiffadcL");        // 差分左通道（MIC0）
    mixer_close(mx);
}

int32_t mic_init(uint32_t sample_rate, uint8_t restore_volume) {
    (void)restore_volume; // 本平台上采集与播放为独立 PCM 流，mic_close 无需恢复扬声器音量

    // 幂等：重复 init 先关闭旧句柄
    if (s_cap) {
        pcm_close(s_cap);
        s_cap = NULL;
    }

    // RV1106 I2S0 不支持单声道（channels=1 报 EINVAL），以 2 声道打开；
    // 贴片麦克风在 MIC0 = 左声道，mic_read 取左声道还原单声道。
    // tinyalsa 直接操作 hw 节点，不支持软重采样，
    // 采样率须为 ACodec 硬件支持的值（44100/48000 等）
    struct pcm_config config;
    memset(&config, 0, sizeof(config));
    config.channels     = 2;
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
    // RV1106 采集路径不会在 prepare/readi 时自动启动（实测 prepare 后 poll 永远
    // 无数据，与 MP135 不同），必须显式 START 进入 XFER 状态
    if (pcm_start(s_cap) < 0) {
        fprintf(stderr, "mic_init: pcm_start failed: %s\n", pcm_get_error(s_cap));
        pcm_close(s_cap);
        s_cap = NULL;
        return -1;
    }
    luckfox_acodec_capture_setup();
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
                pcm_start(s_cap);   // 本平台采集须显式 START（见 mic_init 注释）
                continue;
            }
            return -2;
        }

        // 按帧读取立体声数据，取左声道（MIC0 贴片麦克风）组成单声道输出
        uint32_t want = samples - got;
        if (want > MIC_PERIOD_FRAMES) want = MIC_PERIOD_FRAMES;
        int r = pcm_readi(s_cap, s_stereo_buf, want);
        if (r < 0) {
            if (errno == EAGAIN) continue;
            // EPIPE/ESTRPIPE 已由 tinyalsa 内部自动 prepare 恢复
            return -2;
        }
        for (int i = 0; i < r; i++) {
            buffer[got + (uint32_t)i] = s_stereo_buf[i * 2];
        }
        got += (uint32_t)r;
    }

    return (int32_t)got;
}

int32_t mic_close() {
    // 幂等；本平台上采集与播放为独立 PCM 流，无需恢复扬声器
    if (s_cap) {
        pcm_close(s_cap);
        s_cap = NULL;
    }
    return 0;
}
