/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_common.h
//
//    编码/解码共用常量与宏定义。
//
/////////////////////////////////////////////////////////////////

#ifndef AQUA_COMMON_H
#define AQUA_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#include "aqua_tables.h"

// 圆周率常量（C99 不保证提供 M_PI）
#define AQUA_PI 3.14159265358979323846

// 采样率（枚举，对应表下标）
#define SAMPLE_RATE_32000 0
#define SAMPLE_RATE_44100 1
#define SAMPLE_RATE_48000 2

// 比特率（枚举，对应表下标）
#define BIT_RATE_64K  0
#define BIT_RATE_128K 1
#define BIT_RATE_224K 2
#define BIT_RATE_320K 3

// 块/窗类型枚举
#define LONG_BLOCK  0
#define SHORT_BLOCK 1

#define WINDOW_NORMAL 0
#define WINDOW_START  1
#define WINDOW_SHORT  2
#define WINDOW_STOP   3

// 采样/分块尺寸常量
#define GRANULE_SAMPLES 576   // 一个 granule 的采样数
#define FRAME_SAMPLES   1152  // 一帧的采样数（两个 granule）
#define SUBBANDS        32    // 子带数
#define SUBBAND_LEN     18    // 每个子带每 granule 点数
#define MAX_CHANNELS    2

// 数学辅助：与 JS Math.round 行为一致（floor(x + 0.5)）
static inline double js_round(double x) {
    return floor(x + 0.5);
}

#endif /* AQUA_COMMON_H */
