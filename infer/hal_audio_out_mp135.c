// CoreMP135 扬声器输出HAL：基于 tinyalsa 实现（参照 M5Stack_Linux_Libs BSP
// tinyalsa_example 中 tinyplay 的方式，直接操作声卡 hw 节点，无 ALSA 插件层）。
// 依赖：tinyalsa（头文件 tinyalsa/pcm.h；链接选项 -ltinyalsa）。
//
// 与 hal_audio_out_alsa_linux.c 语义一致，对应关系：
// - 声卡内部环形缓冲扮演“播放队列”的角色，enqueue 时数据被拷贝进该缓冲，
//   因此调用方的乒乓双缓冲纪律天然安全（拷贝语义是引用语义的安全超集）；
// - audio_out_queue_free 依据可用空间（pcm_get_htimestamp 查询 avail）是否
//   容得下“最近一个块”来返回空槽，等价于 ESP32 上 isPlaying(0) <
//   AUDIO_OUT_QUEUE_DEPTH 的槽位语义；
// - 为保证“整块接受/拒绝”的双槽契约与环形缓冲尺寸无关，enqueue 内部设有
//   “待写缓存”（pending buffer）：写不进声卡的剩余采样暂存起来，由
//   queue_free 冲刷；XRUN 由 tinyalsa 内部自动 prepare 恢复并重发手中数据；
// - 音量用软件增益实现（enqueue 时缩放采样），不依赖具体声卡的 Mixer 元素，
//   与 Linux/ESP32 各平台行为一致。

#include "platform.h"
#include "hal_audio_out.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <tinyalsa/pcm.h>

// 声卡号/设备号：默认 0/0（BSP tinyplay/tinycap 的默认值），
// 可用环境变量 NANO_PCM_CARD / NANO_PCM_DEVICE 覆盖
static unsigned int audio_out_card(void) {
    const char *s = getenv("NANO_PCM_CARD");
    return (s && s[0]) ? (unsigned int)atoi(s) : 0;
}

static unsigned int audio_out_device(void) {
    const char *s = getenv("NANO_PCM_DEVICE");
    return (s && s[0]) ? (unsigned int)atoi(s) : 0;
}

// 播放环形缓冲目标帧数下限。须大于调用方单个块的最大时长：
// OFDM 寻呼机一帧 31680 采样 @48kHz ≈ 0.66s，故至少覆盖 1s 且不小于 32768 帧。
#define AUDIO_OUT_PERIOD_FRAMES (4096) // 每个 period 的帧数
#define AUDIO_OUT_MIN_BUFFER_FRAMES (32768)

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

    // 环形缓冲覆盖约 1s 音频（下限 AUDIO_OUT_MIN_BUFFER_FRAMES 帧），
    // 保证容得下最大块（OFDM 一帧 31680 采样），queue_free 判据才不会死锁
    uint32_t buf_frames = (sample_rate > AUDIO_OUT_MIN_BUFFER_FRAMES)
                        ? sample_rate : AUDIO_OUT_MIN_BUFFER_FRAMES;
    uint32_t period_count = (buf_frames + AUDIO_OUT_PERIOD_FRAMES - 1) / AUDIO_OUT_PERIOD_FRAMES;
    buf_frames = period_count * AUDIO_OUT_PERIOD_FRAMES; // 实际环形缓冲总帧数

    // 单声道 S16_LE；tinyalsa 直接操作 hw 节点，不支持软重采样，
    // 采样率须为声卡硬件支持的值（44100/48000 等）
    struct pcm_config config;
    memset(&config, 0, sizeof(config));
    config.channels          = 1;
    config.rate              = sample_rate;
    config.period_size       = AUDIO_OUT_PERIOD_FRAMES;
    config.period_count      = period_count;
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

    return 0;
}

// 向声卡尽量写入 samples 个采样（非阻塞；XRUN 由 tinyalsa 自动恢复并重发手中数据）。
// 返回实际写入的采样数；-1 表示不可恢复的错误。
static int32_t audio_out_try_write(const int16_t *pcm, uint32_t samples) {
    uint32_t written = 0;
    while (written < samples) {
        int r = pcm_writei(s_pcm, pcm + written, samples - written);
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
        memmove(s_pending, s_pending + w, (s_pending_len - (uint32_t)w) * sizeof(int16_t));
        s_pending_len -= (uint32_t)w;
    }
    return (s_pending_len == 0) ? 0 : 1;
}

int32_t audio_out_queue_free(void) {
    if (!s_pcm) return 0;

    // 先冲刷待写缓存；仍有积压则视为无空槽（等价于 ESP32 槽位未播完）
    if (audio_out_flush_pending() != 0) return 0;

    // 查询环形缓冲可用空间（XRUN 状态下内核报告全空，下次写入时自动恢复）
    unsigned int avail = 0;
    struct timespec tstamp;
    if (pcm_get_htimestamp(s_pcm, &avail, &tstamp) < 0) return 0;

    return (avail >= s_block_samples) ? 1 : 0;
}

int32_t audio_out_enqueue(const int16_t *pcm, uint32_t samples) {
    if (!pcm || samples == 0) return -1;
    if (!s_pcm) return -1;
    if (s_pending_len > 0) return -2; // 上一块尚未完全入队：队列满（调用方应先查 queue_free）

    // 软件增益（0~255 → 0.0~1.0 线性增益）
    const int16_t *out = pcm;
    if (audio_out_gain_buf_reserve(samples) == 0) {
        int32_t gain = s_volume;
        for (uint32_t i = 0; i < samples; i++) {
            s_gain_buf[i] = (int16_t)((int32_t)pcm[i] * gain / 255);
        }
        out = s_gain_buf;
    }

    // 尽量写入声卡；写不下的剩余部分转入待写缓存（由 queue_free 冲刷）。
    // 整块必然被接受（除非不可恢复错误），与环形缓冲尺寸无关。
    int32_t w = audio_out_try_write(out, samples);
    if (w < 0) return -2;

    uint32_t rem = samples - (uint32_t)w;
    if (rem > 0) {
        if (rem > s_pending_cap) {
            int16_t *new_buf = (int16_t *)platform_realloc(s_pending, rem * sizeof(int16_t));
            if (!new_buf) return -2; // 极端情况：内存不足，部分数据已入声卡，按失败处理
            s_pending = new_buf;
            s_pending_cap = rem;
        }
        memcpy(s_pending, out + w, rem * sizeof(int16_t));
        s_pending_len = rem;
    }

    s_block_samples = samples;
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
