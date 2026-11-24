/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_qloop.h
//
//    量化和码率控制循环（量化、外层循环和内层循环）。
//
/////////////////////////////////////////////////////////////////

#ifndef AQUA_QLOOP_H
#define AQUA_QLOOP_H

#include "aqua_common.h"
#include "aqua_bits.h"
#include "aqua_huff.h"

// 内层循环输出（对应 qloop.js 中 InnerLoop 的返回对象）
typedef struct {
    AquaHuffmanResult huffman;
    AquaBitsWriter codeBits;   // part3 哈夫曼码流
    int codeBitLen;            // 哈夫曼码流比特数
    int globalGain;
    int qquant;
    int quantized[GRANULE_SAMPLES];
    int success;               // 1 成功，0 量化超时（对应 JS 的 null）
} InnerLoopResult;

// 外层循环输出（对应 qloop.js 中 OuterLoop 的返回对象）
typedef struct {
    InnerLoopResult inner;
    int scalefactors[3][21];   // 长块用 [0][0..20]；短块为 [0..2][0..11]
    int scalefactorCompress;
    int part2Length;
    int part23Length;
    int globalGain;
    int subblockGain[3];
    int success;
} OuterLoopResult;

// 初始化量化数表（POWER/INV_OF_ROOT_2_4），使用一次即可
void aqua_qloop_init(void);

// 对 spectrum 执行外层循环（长块/起始块/结束块走长块分支，短块走短块分支）。
// bitRateLimit 为 part3 预算。
int aqua_outer_loop(double Spectrum[3][192], int windowType, int bitRateLimit,
                    int sampleRateIdx, const double xmin[42], OuterLoopResult *out);

#endif /* AQUA_QLOOP_H */
