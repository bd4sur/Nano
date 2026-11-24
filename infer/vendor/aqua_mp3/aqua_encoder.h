/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_encoder.h
//
//    编码器主体流程。
//
/////////////////////////////////////////////////////////////////

#ifndef AQUA_ENCODER_H
#define AQUA_ENCODER_H

#include "aqua_common.h"
#include "aqua_bits.h"

// 编码器上下文（保存跨帧状态：比特储备、上一 granule 子带）
typedef struct {
    int channels;        // 1 或 2
    int sampleRateIdx;   // SAMPLE_RATE_*
    int bitRateIdx;      // BIT_RATE_*
    long reservoirMax;   // 比特储备最大容量(bit)
    long reservoirSize;  // 比特储备当前大小(bit)
    double prevSubbands[MAX_CHANNELS][SUBBANDS][SUBBAND_LEN];
    int prevBlockType[MAX_CHANNELS];
    double attackAvg[MAX_CHANNELS][SUBBANDS]; // attack 检测：各子带能量滑动平均
    int attackInit[MAX_CHANNELS];             // attack 平均是否已初始化
} AquaEncoder;

// 初始化（含全局数表的一次性初始化）并重置状态
int aqua_encoder_init(AquaEncoder *enc, int channels, int sampleRateIdx, int bitRateIdx);

// 仅重置帧间状态（channel 数、采样率、比特率也可一同重设）
int aqua_encoder_reset(AquaEncoder *enc, int channels, int sampleRateIdx, int bitRateIdx);

// 编码一帧（两个 granule）。
// pcm[ch] 为浮点 PCM（[-1,1]），numSamples 为数组长度，offset 为 granule 起点。
// 输出字节追加写入 out。返回 1 成功，0 失败。
// 注意：与 JS 一样，调用方须提供包含“上一帧”的连续 PCM（offset 取第二帧开头）。
int aqua_encoder_encode_frame(AquaEncoder *enc, const double *pcm[MAX_CHANNELS],
                              long numSamples, long offset, AquaBitsWriter *out);

#endif /* AQUA_ENCODER_H */
