// Linux 扬声器输出HAL：基于 ALSA（libasound）实现。
// 依赖：alsa-lib（Debian/Ubuntu: libasound2-dev；链接选项 -lasound）。
//
// 与 ESP32（audio_out_m5core2.cpp，M5Unified 双槽队列）的语义对应关系：
// - ALSA 内部环形缓冲扮演“播放队列”的角色，enqueue 时数据被拷贝进该缓冲，
//   因此调用方的乒乓双缓冲纪律天然安全（拷贝语义是引用语义的安全超集）；
// - audio_out_queue_free 依据 ALSA 可用空间是否容得下“最近一个块”来返回空槽，
//   等价于 ESP32 上 isPlaying(0) < AUDIO_OUT_QUEUE_DEPTH 的槽位语义；
// - 由于 ALSA 插件（pulse/dmix 等）协商出的缓冲可能装不下两个块（例如 WSLg 的
//   PulseAudio 插件），enqueue 内部设有“待写缓存”（pending buffer）：写不进 ALSA
//   的剩余采样暂存起来，由 queue_free 冲刷，从而与 ALSA 缓冲尺寸无关地严格保证
//   ESP32 的“整块接受/拒绝”双槽契约；XRUN 恢复后也会自动重发手中数据；
// - 音量用软件增益实现（enqueue 时缩放采样），避免依赖具体声卡的 Mixer 元素，
//   在 PC、树莓派（含 USB 声卡/HDMI/耳机口）上行为一致。

#include "platform.h"
#include "audio_out.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>

// 播放设备名：默认 "default"，可用环境变量 NANO_ALSA_DEVICE 覆盖
//（例如 default 配置损坏时指定 "plughw:0,0"）
static const char *audio_out_device(void) {
    const char *dev = getenv("NANO_ALSA_DEVICE");
    return (dev && dev[0]) ? dev : "default";
}

// ALSA 播放缓冲目标时长（us）。须大于调用方单个块的最大时长：
// OFDM 寻呼机一帧 31680 采样 @48kHz ≈ 0.66s，故取 1s。
#define AUDIO_OUT_BUFFER_TIME_US (1000000)

static snd_pcm_t        *s_pcm           = NULL;
static snd_pcm_uframes_t s_buffer_frames = 0;    // ALSA 环形缓冲总帧数
static uint32_t          s_block_samples = 4096; // 最近一次 enqueue 的块长（queue_free 判据）
static uint8_t           s_volume        = 16;
static uint8_t           s_prev_volume   = 16;   // init 时保存，close 时恢复
static int16_t          *s_gain_buf      = NULL; // 软件增益暂存缓冲
static uint32_t          s_gain_buf_cap  = 0;    // s_gain_buf 容量（采样点）
static int16_t          *s_pending       = NULL; // 未能及时写入 ALSA 的剩余采样
static uint32_t          s_pending_len   = 0;
static uint32_t          s_pending_cap   = 0;

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
        snd_pcm_close(s_pcm);
        s_pcm = NULL;
    }

    s_prev_volume = s_volume;
    s_volume = volume;

    int err = snd_pcm_open(&s_pcm, audio_out_device(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);
    if (err < 0) {
        fprintf(stderr, "audio_out_init: snd_pcm_open(%s) failed: %s\n",
                audio_out_device(), snd_strerror(err));
        s_pcm = NULL;
        return -1; // 无可用声卡
    }

    // 单声道 S16_LE；允许 ALSA 软重采样以兼容音乐盒中任意文件采样率
    err = snd_pcm_set_params(s_pcm,
                             SND_PCM_FORMAT_S16_LE,
                             SND_PCM_ACCESS_RW_INTERLEAVED,
                             1,               // 通道数：单声道
                             sample_rate,
                             1,               // soft_resample：允许重采样
                             AUDIO_OUT_BUFFER_TIME_US);
    if (err < 0) {
        fprintf(stderr, "audio_out_init: snd_pcm_set_params failed: %s\n", snd_strerror(err));
        snd_pcm_close(s_pcm);
        s_pcm = NULL;
        return -2;
    }

    snd_pcm_prepare(s_pcm);
    // 刚 prepare 完时可用空间即缓冲总帧数
    snd_pcm_sframes_t avail = snd_pcm_avail_update(s_pcm);
    s_buffer_frames = (avail > 0) ? (snd_pcm_uframes_t)avail : 0;
    s_block_samples = 4096;
    s_pending_len = 0;

    return 0;
}

// 向 ALSA 尽量写入 samples 个采样（非阻塞，XRUN/挂起自动恢复后重发手中数据）。
// 返回实际写入的采样数；-1 表示不可恢复的错误。
static int32_t audio_out_try_write(const int16_t *pcm, uint32_t samples) {
    uint32_t written = 0;
    while (written < samples) {
        snd_pcm_sframes_t r = snd_pcm_writei(s_pcm, pcm + written, samples - written);
        if (r == -EAGAIN) {
            break; // 缓冲满：已尽力，返回已写入数
        }
        if (r == -EPIPE || r == -ESTRPIPE) {
            if (snd_pcm_recover(s_pcm, (int)r, 1) < 0) return -1;
            continue; // 恢复后重发未写入的数据
        }
        if (r < 0) return -1;
        written += (uint32_t)r;
    }
    return (int32_t)written;
}

// 将待写缓存冲刷进 ALSA。返回 0 表示已全部写入，1 表示仍有积压，-1 表示错误。
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

    snd_pcm_sframes_t avail = snd_pcm_avail_update(s_pcm);
    if (avail == -EPIPE) {
        // 欠载（underrun）：恢复后缓冲为空
        snd_pcm_prepare(s_pcm);
        avail = snd_pcm_avail_update(s_pcm);
    }
    if (avail < 0) return 0;

    return ((snd_pcm_uframes_t)avail >= s_block_samples) ? 1 : 0;
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

    // 尽量写入 ALSA；写不下的剩余部分转入待写缓存（由 queue_free 冲刷）。
    // 整块必然被接受（除非不可恢复错误），与 ALSA 插件协商出的缓冲尺寸无关。
    int32_t w = audio_out_try_write(out, samples);
    if (w < 0) return -2;

    uint32_t rem = samples - (uint32_t)w;
    if (rem > 0) {
        if (rem > s_pending_cap) {
            int16_t *new_buf = (int16_t *)platform_realloc(s_pending, rem * sizeof(int16_t));
            if (!new_buf) return -2; // 极端情况：内存不足，部分数据已入 ALSA，按失败处理
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
    snd_pcm_drop(s_pcm);    // 立即停止并丢弃缓冲中未播放的数据
    snd_pcm_prepare(s_pcm); // 复位，供后续重新入队
    s_pending_len = 0;      // 清空待写缓存（对齐 ESP32“停止并清空队列”语义）
}

void audio_out_set_volume(uint8_t volume) {
    s_volume = volume;
}

void audio_out_close(void) {
    audio_out_stop();
    if (s_pcm) {
        snd_pcm_close(s_pcm);
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
