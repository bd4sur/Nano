/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_mdct.h
//
//    改进的离散余弦变换（编码端）。
//
/////////////////////////////////////////////////////////////////

#ifndef AQUA_MDCT_H
#define AQUA_MDCT_H

#include "aqua_common.h"

// 初始化 MDCT 系数表（每个编码上下文只需一次）
void aqua_mdct_init(void);

// MDCT：length=36 输出 18 点（长窗）；length=12 输出 6 点（短窗）
// output[i] = sum_k input[k] * factor[i*length+k] / length * 4
void aqua_mdct(const double *input, int length, double *output);

// 去混叠蝶形结运算（仅长块，就地修改）
void aqua_reduce_aliasing(double *longBlockSpectrum);

// 计算一个 Granule 的频谱。
// current: 当前 granule 的 32 子带 x 18 点；prev: 上一 granule 的同样结构
// windowType: WINDOW_NORMAL/START/STOP -> 输出 576 点长块；WINDOW_SHORT -> 3x192
// outLong: 长块输出（576）；outShort: 短块输出（3x192）
void aqua_granule_mdct(const double current[SUBBANDS][SUBBAND_LEN],
                       const double prev[SUBBANDS][SUBBAND_LEN], int windowType,
                       double *outLong, double outShort[3][192]);

#endif /* AQUA_MDCT_H */
