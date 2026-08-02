#ifndef __NANO_AUDIO_OUT_H__
#define __NANO_AUDIO_OUT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

// 扬声器原始PCM流式播放HAL：硬件无关接口。
// 各平台在对应的 audio_out_<platform>.cpp 中实现本接口（如 audio_out_m5core2.cpp）。
//
// 设计背景（OFDM 寻呼机发射）：
// - M5Unified 的各"通道"是并行混音关系（多通道同时播放会被叠加混合），
//   不能用作顺序播放队列；顺序无缝播放必须用同一通道内置的双槽队列；
// - playRaw 引用而非拷贝PCM数据，因此调用方须以乒乓双缓冲轮换：
//   每轮查询 audio_out_queue_free，有空槽即填入下一块并 audio_out_enqueue。

#define AUDIO_OUT_QUEUE_DEPTH (2) // 单通道队列深度（M5Unified 每通道 wavinfo 双槽）

// 初始化扬声器输出（begin + 设定主音量；会保存原音量供 audio_out_close 恢复）
//   sample_rate ：PCM 采样率（Hz），如 48000
//   volume      ：主音量（0~255）
int32_t audio_out_init(uint32_t sample_rate, uint8_t volume);

// 查询队列是否有空槽（可投入下一块PCM）
int32_t audio_out_queue_free(void);

// 投入一块 int16 单声道 PCM（数据被引用而非拷贝，须保持有效至该块播放完成）
int32_t audio_out_enqueue(const int16_t *pcm, uint32_t samples);

// 停止播放并清空队列
void audio_out_stop(void);

// 运行时调节主音量（0~255；供音乐盒等场景的音量调节，不影响 close 时恢复原音量）
void audio_out_set_volume(uint8_t volume);

// 关闭输出并恢复原主音量
void audio_out_close(void);

#ifdef __cplusplus
}
#endif

#endif
