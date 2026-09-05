// Luckfox-Pico-86-Panel 扬声器输出HAL：基于 tinyalsa 实现（直接操作声卡 hw 节点，
// 无 ALSA 插件层）。依赖：tinyalsa（头文件 tinyalsa/pcm.h；链接 -ltinyalsa）。
//
// 硬件链路（RV1106G3 内置 ACodec，详见原理图与 Rockchip RV1106 ACodec 开发指南）：
// - ACodec DAC LINEOUT → LM4890S 功放 → MX1.25-2P 喇叭座子；
// - 声卡 rv1106-acodec（hw:0,0），playback 为 /dev/snd/pcmC0D0p；
// - 采样率支持 8k/12k/16k/24k/32k/44.1k/48k/96k，单声道 S16_LE。
//
// 与 hal_audio_out_mp135.c 语义一致（同一套契约：环形缓冲当播放队列、
// pending buffer 保证整块接受/拒绝、软件增益音量），差异仅在：
// - init 时经 tinyalsa mixer API 初始化 ACodec 播放通路 mixer
//   （DAC LINEOUT/HPMIX 音量设为 0dB 参考位），避免外部程序（如出厂 demo）
//   改动 mixer 后播放无声；mixer 设置失败不影响主流程（仅告警）；
// - RV1106 I2S0 控制器不支持单声道（channels=1 会被驱动拒绝），
//   故设备以 2 声道打开，enqueue 时将单声道采样复制到 L/R 两声道
//   （对外仍是契约规定的单声道接口；喇叭为单 PA，L/R 内容相同）；
// - 该驱动环形缓冲上限仅 4096 帧（实测 period 2048×2；MP135 的 32768 帧
//   配置在本机报 EINVAL），容不下 OFDM 整块（31680 采样），故 queue_free
//   的空槽判据改为“pending 冲刷完 + 声卡至少释放一个 period”，
//   由 pending buffer 承接块内剩余数据，随播放实时流入声卡。
//
// 已实测：aplay 播放 440Hz 正弦，贴片麦克风端可清晰捕获（声学回环验证通过）。

#include "platform.h"
#include "hal_audio_out.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tinyalsa/pcm.h>
#include <tinyalsa/mixer.h>

// 声卡号/设备号：默认 0/0（rv1106-acodec 为卡0设备0），
// 可用环境变量 NANO_PCM_CARD / NANO_PCM_DEVICE 覆盖
static unsigned int audio_out_card(void) {
    const char *s = getenv("NANO_PCM_CARD");
    return (s && s[0]) ? (unsigned int)atoi(s) : 0;
}

static unsigned int audio_out_device(void) {
    const char *s = getenv("NANO_PCM_DEVICE");
    return (s && s[0]) ? (unsigned int)atoi(s) : 0;
}

// RV1106 acodec 驱动环形缓冲上限 4096 帧（实测），取 period 2048 × 2。
// 大于 4096 帧的播放块（如 OFDM 一帧 31680 采样）由 pending buffer 承接，
// 随播放实时流入声卡（见文件头注释）。
#define AUDIO_OUT_PERIOD_FRAMES (2048) // 每个 period 的帧数
#define AUDIO_OUT_PERIOD_COUNT  (2)

static struct pcm *s_pcm           = NULL;
static uint32_t    s_buffer_frames = 0;    // 环形缓冲总帧数（period_size * period_count）
static uint32_t    s_block_samples = 4096; // 最近一次 enqueue 的块长（queue_free 判据）
static uint8_t     s_volume        = 16;
static uint8_t     s_prev_volume   = 16;   // init 时保存，close 时恢复
static int16_t    *s_gain_buf      = NULL; // 软件增益暂存缓冲
static uint32_t    s_gain_buf_cap  = 0;    // s_gain_buf 容量（采样点）
static int16_t    *s_pending       = NULL; // 未能及时写入声卡的剩余采样
static uint32_t    s_pending_len   = 0;
static uint32_t    s_pending_cap   = 0;

// 按名字设置整型 mixer 控件（控件不存在或设置失败仅告警，不致命）
static void luckfox_mixer_set_int(struct mixer *mx, const char *name, int value) {
    struct mixer_ctl *ctl = mixer_get_ctl_by_name(mx, name);
    if (!ctl) {
        fprintf(stderr, "audio_out_init: mixer ctl '%s' not found\n", name);
        return;
    }
    if (mixer_ctl_set_value(ctl, 0, value) < 0) {
        fprintf(stderr, "audio_out_init: set mixer ctl '%s'=%d failed\n", name, value);
    }
}

// 初始化 RV1106 ACodec 播放通路（音量参考位 0dB；LINEOUT 最小值即静音，故必须显式设置）
static void luckfox_acodec_playback_setup(void) {
    struct mixer *mx = mixer_open(audio_out_card());
    if (!mx) {
        fprintf(stderr, "audio_out_init: mixer_open(card=%u) failed\n", audio_out_card());
        return;
    }
    luckfox_mixer_set_int(mx, "DAC LINEOUT Volume", 26); // 0dB（0..30，-39dB..+6dB，26=0dB）
    luckfox_mixer_set_int(mx, "DAC HPMIX Volume", 1);    // 0dB（前级，有效取值 1/2）
    mixer_close(mx);
}

// 确保增益暂存缓冲至少能容纳 samples 个采样点；成功返回 0，失败返回 -1
static int32_t audio_out_gain_buf_reserve(uint32_t samples) {
    if (samples <= s_gain_buf_cap) return 0;
    int16_t *new_buf = (int16_t *)platform_realloc(s_gain_buf, samples * sizeof(int16_t));
    if (!new_buf) return -1;
    s_gain_buf = new_buf;
    s_gain_buf_cap = samples;
    return 0;
}

int32_t audio_out_init(uint32_t sample_rate, uint8_t volume) {
    // 允许重复 init（音乐盒切歌时采样率可能变化）：先关闭旧设备
    if (s_pcm) {
        pcm_close(s_pcm);
        s_pcm = NULL;
    }

    s_prev_volume = s_volume;
    s_volume = volume;

    // 环形缓冲 = period × count = 4096 帧（该驱动的上限；见文件头注释）
    uint32_t buf_frames = AUDIO_OUT_PERIOD_FRAMES * AUDIO_OUT_PERIOD_COUNT;

    // RV1106 I2S0 不支持单声道（channels=1 报 EINVAL），以 2 声道打开；
    // 单声道采样在 enqueue 内复制到 L/R。tinyalsa 直接操作 hw 节点，
    // 不支持软重采样，采样率须为 ACodec 硬件支持的值（44100/48000 等）。
    // 注意：pcm_config 的 period/buffer 以“帧”为单位，与声道数无关。
    struct pcm_config config;
    memset(&config, 0, sizeof(config));
    config.channels          = 2;
    config.rate              = sample_rate;
    config.period_size       = AUDIO_OUT_PERIOD_FRAMES;
    config.period_count      = AUDIO_OUT_PERIOD_COUNT;
    config.format            = PCM_FORMAT_S16_LE;
    config.start_threshold   = AUDIO_OUT_PERIOD_FRAMES;   // 攒够一个 period 即起播
    config.stop_threshold    = buf_frames;                // 欠载即停（XRUN）
    config.silence_threshold = 0;

    s_pcm = pcm_open(audio_out_card(), audio_out_device(),
                     PCM_OUT | PCM_NONBLOCK, &config);
    if (!s_pcm || !pcm_is_ready(s_pcm)) {
        fprintf(stderr, "audio_out_init: pcm_open(card=%u,dev=%u,rate=%u) failed: %s\n",
                audio_out_card(), audio_out_device(), sample_rate,
                s_pcm ? pcm_get_error(s_pcm) : "out of memory");
        if (s_pcm) {
            pcm_close(s_pcm);
            s_pcm = NULL;
        }
        return -1; // 无可用声卡或参数不受支持
    }

    pcm_prepare(s_pcm);
    s_buffer_frames = config.period_size * config.period_count;
    s_block_samples = 4096;
    s_pending_len = 0;

    luckfox_acodec_playback_setup();
    return 0;
}

// 向声卡尽量写入 frames 帧立体声数据（非阻塞；XRUN 由 tinyalsa 自动恢复并重发手中数据）。
// 返回实际写入的帧数；-1 表示不可恢复的错误。
static int32_t audio_out_try_write(const int16_t *pcm, uint32_t frames) {
    uint32_t written = 0;
    while (written < frames) {
        int r = pcm_writei(s_pcm, pcm + written * 2, frames - written);
        if (r < 0) {
            if (errno == EAGAIN) break; // 缓冲满：已尽力，返回已写入数
            return -1;
        }
        written += (uint32_t)r;
    }
    return (int32_t)written;
}

// 将待写缓存冲刷进声卡。返回 0 表示已全部写入，1 表示仍有积压，-1 表示错误。
static int32_t audio_out_flush_pending(void) {
    if (s_pending_len == 0) return 0;
    int32_t w = audio_out_try_write(s_pending, s_pending_len);
    if (w < 0) return -1;
    if (w > 0) {
        memmove(s_pending, s_pending + w * 2, (s_pending_len - (uint32_t)w) * 2 * sizeof(int16_t));
        s_pending_len -= (uint32_t)w;
    }
    return (s_pending_len == 0) ? 0 : 1;
}

int32_t audio_out_queue_free(void) {
    if (!s_pcm) return 0;

    // 先冲刷待写缓存；仍有积压则视为无空槽（等价于 ESP32 槽位未播完）
    if (audio_out_flush_pending() != 0) return 0;

    // 查询环形缓冲可用空间（XRUN 状态下内核报告全空，下次写入时自动恢复）。
    // 空槽判据：至少释放一个 period（本驱动缓冲上限 4096 帧，容不下 OFDM 整块，
    // 故不能用“容得下整块”作判据，否则 31680 采样的块永远等不到空槽而死锁；
    // 块内剩余数据由 pending buffer 承接，enqueue 后随播放实时流入声卡）。
    uint32_t need = (s_block_samples < AUDIO_OUT_PERIOD_FRAMES)
                  ? s_block_samples : AUDIO_OUT_PERIOD_FRAMES;
    unsigned int avail = 0;
    struct timespec tstamp;
    if (pcm_get_htimestamp(s_pcm, &avail, &tstamp) < 0) return 0;

    return (avail >= need) ? 1 : 0;
}

int32_t audio_out_enqueue(const int16_t *pcm, uint32_t samples) {
    if (!pcm || samples == 0) return -1;
    if (!s_pcm) return -1;
    if (s_pending_len > 0) return -2; // 上一块尚未完全入队：队列满（调用方应先查 queue_free）

    // 软件增益（0~255 → 0.0~1.0 线性增益）+ 单声道复制到 L/R 两声道
    int16_t *out = s_gain_buf;
    if (audio_out_gain_buf_reserve(samples * 2) == 0) {
        int32_t gain = s_volume;
        for (uint32_t i = 0; i < samples; i++) {
            int16_t v = (int16_t)((int32_t)pcm[i] * gain / 255);
            s_gain_buf[i * 2]     = v;
            s_gain_buf[i * 2 + 1] = v;
        }
    } else {
        return -2; // 内存不足
    }

    // 尽量写入声卡（samples 个单声道采样 = samples 帧立体声）；写不下的剩余
    // 部分转入待写缓存（由 queue_free 冲刷）。整块必然被接受，与环形缓冲尺寸无关。
    int32_t w = audio_out_try_write(out, samples);
    if (w < 0) return -2;

    uint32_t rem = samples - (uint32_t)w; // 剩余帧数
    if (rem > 0) {
        if (rem * 2 > s_pending_cap) {
            int16_t *new_buf = (int16_t *)platform_realloc(s_pending, rem * 2 * sizeof(int16_t));
            if (!new_buf) return -2; // 极端情况：内存不足，部分数据已入声卡，按失败处理
            s_pending = new_buf;
            s_pending_cap = rem * 2;
        }
        memcpy(s_pending, out + w * 2, rem * 2 * sizeof(int16_t));
        s_pending_len = rem;
    }

    s_block_samples = samples; // 块长以帧计（queue_free 的 avail 判据同单位）
    return 0;
}

void audio_out_stop(void) {
    if (!s_pcm) return;
    pcm_stop(s_pcm);    // SNDRV_PCM_IOCTL_DROP：立即停止并丢弃缓冲中未播放的数据
    pcm_prepare(s_pcm); // 复位，供后续重新入队
    s_pending_len = 0;  // 清空待写缓存（对齐 ESP32“停止并清空队列”语义）
}

void audio_out_set_volume(uint8_t volume) {
    s_volume = volume;
}

// 系统级扬声器主音量（全局缓存 + 应用到软件增益；按键音/OFDM 发射/音乐盒/mic 恢复共用）
static uint8_t s_master_volume = 16; // 与 ui_init 的 volume 初值一致

void audio_out_set_master_volume(uint8_t volume) {
    s_master_volume = volume;
    s_volume = volume; // 本 HAL 音量为 enqueue 时的软件增益，立即生效
}

uint8_t audio_out_get_master_volume(void) {
    return s_master_volume;
}

// platform.h 声明的全局主音量接口：
// 由本 HAL 统一实现，确保与 audio_out 当前音量状态一致。
void platform_set_master_volume(uint8_t volume) {
    audio_out_set_master_volume(volume);
}

uint8_t platform_get_master_volume(void) {
    return audio_out_get_master_volume();
}

void audio_out_close(void) {
    audio_out_stop();
    if (s_pcm) {
        pcm_close(s_pcm);
        s_pcm = NULL;
    }
    s_volume = s_prev_volume; // 恢复进入前的音量（对齐 ESP32 语义）
    if (s_gain_buf) {
        free(s_gain_buf);
        s_gain_buf = NULL;
        s_gain_buf_cap = 0;
    }
    if (s_pending) {
        free(s_pending);
        s_pending = NULL;
        s_pending_cap = 0;
        s_pending_len = 0;
    }
}
