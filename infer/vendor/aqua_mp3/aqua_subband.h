/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_subband.h
//
//    分析子带滤波器（编码端）。
//
/////////////////////////////////////////////////////////////////

#ifndef AQUA_SUBBAND_H
#define AQUA_SUBBAND_H

#include "aqua_common.h"

// 分析子带滤波：对一 granule（576 点，起点为 offset）做滤波，
// 输出 32 个频带的时域序列，每频带 18 点。
// PCMData 为浮点 PCM（[-1,1] 标度），len 为数组长度。
void aqua_analysis_subband_filter(const double *PCMData, long len, long offset,
                                  double subbands[SUBBANDS][SUBBAND_LEN]);

#endif /* AQUA_SUBBAND_H */
