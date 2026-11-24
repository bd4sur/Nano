/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_qloop.c
//
//    量化和码率控制循环实现（对齐 src/qloop.js）。
//
/////////////////////////////////////////////////////////////////

#include <math.h>
#include <string.h>
#include "aqua_qloop.h"

// 量化计算用数表（对应全局 POWER_OF_ROOT_2_4 / INV_POWER_OF_ROOT_2_4）
static double POWER_OF_ROOT_2_4[512];
static double INV_POWER_OF_ROOT_2_4[512];

void aqua_qloop_init(void) {
    for (int i = 0; i < 512; i++) {
        POWER_OF_ROOT_2_4[i] = pow(1.189207115002721, i - 256);
        INV_POWER_OF_ROOT_2_4[i] = 1.0 / POWER_OF_ROOT_2_4[i];
    }
}

// 快速计算 Math.pow(x, 0.75)，分段线性逼近（与 common.js fastPower0p75 相同）
static double fastPower0p75(double x) {
    if (x < 0)            return 0;
    else if (x < 50)      return (x          ) * 0.376060309 + 0           ;
    else if (x < 100)     return (x - 50     ) * 0.256395223 + 18.80301547 ;
    else if (x < 200)     return (x - 100    ) * 0.215601824 + 31.6227766  ;
    else if (x < 400)     return (x - 200    ) * 0.181298801 + 53.18295897 ;
    else if (x < 800)     return (x - 400    ) * 0.152453512 + 89.4427191  ;
    else if (x < 1500)    return (x - 800    ) * 0.12943486  + 150.4241237 ;
    else if (x < 3000)    return (x - 1500   ) * 0.109554347 + 241.0285257 ;
    else if (x < 4500)    return (x - 3000   ) * 0.096044119 + 405.3600464 ;
    else if (x < 6000)    return (x - 4500   ) * 0.088203596 + 549.4262252 ;
    else if (x < 7500)    return (x - 6000   ) * 0.082797219 + 681.7316199 ;
    else if (x < 10000)   return (x - 7500   ) * 0.07762902  + 805.9274489 ;
    else if (x < 25000)   return (x - 10000  ) * 0.065878455 + 1000        ;
    else if (x < 50000)   return (x - 25000  ) * 0.054220988 + 1988.176822 ;
    else if (x < 80000)   return (x - 50000  ) * 0.047104231 + 3343.701525 ;
    else if (x < 100000)  return (x - 80000  ) * 0.04332924  + 4756.82846  ;
    else if (x < 120000)  return (x - 100000) * 0.041200317 + 5623.413252 ;
    else if (x < 140000)  return (x - 120000) * 0.039510228 + 6447.419591 ;
    else if (x < 170000)  return (x - 140000) * 0.037817329 + 7237.624155 ;
    else                  return 8372.14402859;
}

// 频谱平坦度（对应 SFM）
static double SFM(const double *xr576) {
    double temp1 = 0, temp2 = 0;
    int xrlen = 0;
    for (int i = 0; i < GRANULE_SAMPLES; i++) {
        if (xr576[i] == 0) continue;
        double sqr = xr576[i] * xr576[i];
        temp1 += log(sqr);
        temp2 += sqr;
        xrlen++;
    }
    if (temp2 == 0) return 1.0;
    temp1 /= xrlen;
    temp1 = exp(temp1);
    temp2 /= xrlen;
    return temp1 / temp2;
}

// 计算各个尺度因子频带的量化误差（对应 QuantNoise）
static void QuantNoise(const double *xr, const int *ix, int quantStep,
                       int blockType, int sampleRateIdx, double *xfsf) {
    int nBands = (blockType == WINDOW_SHORT) ? 12 : 21;
    const int(*SFB)[2] = (blockType == WINDOW_SHORT)
                             ? SFB_SR[sampleRateIdx]
                             : SFB_LR[sampleRateIdx];
    double power_of_root_2_4 = POWER_OF_ROOT_2_4[quantStep + 256];
    for (int sbindex = 0; sbindex < nBands; sbindex++) {
        double sum = 0.0;
        int start = SFB[sbindex][0], end = SFB[sbindex][1];
        for (int i = start; i <= end; i++) {
            double temp1;
            if (ix[i] == 0) {
                temp1 = (xr[i] >= 0) ? xr[i] : -xr[i];
            } else {
                temp1 = ((xr[i] >= 0) ? xr[i] : -xr[i])
                        - pow((ix[i] >= 0 ? ix[i] : -ix[i]), 1.3333333333333)
                          * power_of_root_2_4;
            }
            sum += (temp1 * temp1);
        }
        xfsf[sbindex] = sum / (end - start + 1);
    }
}

// 对 576 点序列执行量化（对应 Quantize）
static void Quantize(const double *xr576, int quantStep, int ix576[576]) {
    double inv_power = INV_POWER_OF_ROOT_2_4[quantStep + 256];
    for (int i = 0; i < 576; i++) {
        double xr = xr576[i];
        if (xr > 0) {
            double a = fastPower0p75(xr * inv_power);
            ix576[i] = (int)(a + 0.4054);
        } else if (xr < 0) {
            double a = fastPower0p75((-xr) * inv_power);
            ix576[i] = -(int)(a + 0.4054);
        } else {
            ix576[i] = 0;
        }
    }
}

// 短块频谱重排（对应 MuxShortBlockSpectrum）：按 sfb 优先、窗口次之展开为 576 点
static void muxShortBlockSpectrum(const double spect[3][192], int sampleRateIdx,
                                  double out576[576]) {
    int offset = 0;
    const int(*SFBands)[2] = SFB_SR[sampleRateIdx];
    for (int sfb = 0; sfb < 12; sfb++) {
        int a = SFBands[sfb][0], b = SFBands[sfb][1];
        for (int w = 0; w < 3; w++) {
            for (int i = a; i <= b; i++) {
                out576[offset++] = spect[w][i];
            }
        }
    }
    // 短块只打包标准 12 个 SFB；最后一带(136..191 等)之后置零，与解码端 13 边界布局对齐
    for (; offset < 576; offset++) {
        out576[offset] = 0.0;
    }
}

// 短块频谱逆重排（对应 DemuxShortBlockSpectrum）
static void demuxShortBlockSpectrum(const int in576[576], int sampleRateIdx,
                                    int out[3][192]) {
    int offset = 0;
    const int(*SFBands)[2] = SFB_SR[sampleRateIdx];
    for (int sfb = 0; sfb < 12; sfb++) {
        int a = SFBands[sfb][0], b = SFBands[sfb][1];
        for (int w = 0; w < 3; w++) {
            for (int i = a; i <= b; i++) {
                out[w][i] = in576[offset++];
            }
        }
    }
    // 与 mux 对称：最后一带之后置零
    for (int w = 0; w < 3; w++) {
        for (int i = SFBands[11][1] + 1; i < 192; i++) {
            out[w][i] = 0;
        }
    }
}

// 内层循环（码率控制循环），对应 InnerLoop
static InnerLoopResult InnerLoop(const double Spectrum[3][192], int blockType,
                                 int bitRateLimit, int sampleRateIdx) {
    InnerLoopResult out;
    memset(&out, 0, sizeof(out));
    out.codeBits.data = NULL;
    out.success = 0;

    double muxed[576];
    const double *spectrum576;
    if (blockType == WINDOW_SHORT) {
        muxShortBlockSpectrum(Spectrum, sampleRateIdx, muxed);
        spectrum576 = muxed;
    } else {
        spectrum576 = &Spectrum[0][0];
    }

    int initLength = -1;
    double quantanf = js_round(8.0 * log(SFM(spectrum576))) - 50.0;

    for (int qquant = 0; qquant < 256; qquant++) {
        int quantStep = (int)(quantanf + qquant);
        int quantized[576];
        Quantize(spectrum576, quantStep, quantized);
        int globalGain = quantStep + 210;

        AquaBitsWriter localBits;
        aqua_bw_init(&localBits);
        AquaHuffmanResult huff;
        int huffOk = aqua_huff_encode(quantized, blockType, sampleRateIdx,
                                      &localBits, &huff);

        if (huffOk) {
            if (initLength < 0) initLength = localBits.len;
            int gone = initLength - localBits.len;
            int togo = localBits.len - bitRateLimit;
            if (gone < 2 * togo) {
                qquant += 4;
            } else if (gone < 4 * togo) {
                qquant += 2;
            }
        }

        if (huffOk && localBits.len < bitRateLimit) {
            if (localBits.len == 0) globalGain = 0;
            if (out.codeBits.data == NULL) {
                aqua_bw_init(&out.codeBits);
            }
            out.codeBits.len = 0;
            for (int i = 0; i < localBits.len; i++) {
                aqua_bw_put(&out.codeBits,
                            (uint32_t)((localBits.data[i >> 3] >> (7 - (i & 7))) & 1u), 1);
            }
            out.codeBitLen = localBits.len;
            out.huffman = huff;
            out.globalGain = (globalGain > 255) ? 255 : globalGain;
            out.qquant = qquant;
            out.success = 1;
            memcpy(out.quantized, quantized, sizeof(quantized));
            aqua_bw_free(&localBits);
            return out;
        }
        aqua_bw_free(&localBits);
    }
    return out;
}

// 计算尺度因子的 scalefac_compress（对应 ScalefactorCompress）
// scalefactors[3][21]：长块用 [0][0..20]；短块用 [0..2][0..11]
static int ScalefactorCompress(const int scalefactors[3][21], int blockType) {
    int maxValue1 = -1;
    int maxValue2 = -1;
    if (blockType != WINDOW_SHORT) {
        for (int i = 0; i <= 10; i++) {
            if (scalefactors[0][i] > maxValue1) maxValue1 = scalefactors[0][i];
        }
        for (int i = 11; i <= 20; i++) {
            if (scalefactors[0][i] > maxValue2) maxValue2 = scalefactors[0][i];
        }
    } else {
        for (int i = 0; i <= 5; i++) {
            int sf = scalefactors[0][i];
            if (scalefactors[1][i] > sf) sf = scalefactors[1][i];
            if (scalefactors[2][i] > sf) sf = scalefactors[2][i];
            if (sf > maxValue1) maxValue1 = sf;
        }
        for (int i = 6; i <= 11; i++) {
            int sf = scalefactors[0][i];
            if (scalefactors[1][i] > sf) sf = scalefactors[1][i];
            if (scalefactors[2][i] > sf) sf = scalefactors[2][i];
            if (sf > maxValue2) maxValue2 = sf;
        }
    }
    static const int SF_COMPRESS[5][4] = {
        { 0,  1,  2,  3},
        {-1,  5,  6,  7},
        {-1,  8,  9, 10},
        { 4, 11, 12, 13},
        {-1, -1, 14, 15}
    };
    // 与 JS 一致：slen = floor(log2(max))+1
    int slen1, slen2;
    slen1 = (maxValue1 == 0) ? 0 : (int)(floor(log2((double)maxValue1)) + 1.0);
    slen2 = (maxValue2 == 0) ? 0 : (int)(floor(log2((double)maxValue2)) + 1.0);
    int sfcompress = (slen1 < 5 && slen2 < 4) ? SF_COMPRESS[slen1][slen2] : -1;
    return (sfcompress < 0) ? 15 : sfcompress;
}

// 计算 part2（尺度因子）长度（对应 Part2Length）
static int Part2Length(int scalefactorCompress, int blockType) {
    static const int SF_COMPRESS_INDEX[16][2] = {
        {0,0},{0,1},{0,2},{0,3},{3,0},{1,1},{1,2},{1,3},
        {2,1},{2,2},{2,3},{3,1},{3,2},{3,3},{4,2},{4,3}
    };
    int slen1 = SF_COMPRESS_INDEX[scalefactorCompress][0];
    int slen2 = SF_COMPRESS_INDEX[scalefactorCompress][1];
    int part2Length;
    if (blockType != WINDOW_SHORT) {
        part2Length = 11 * slen1 + 10 * slen2;
    } else {
        part2Length = (6 * slen1 + 6 * slen2) * 3;
    }
    return part2Length;
}

// 长块外层循环（对应 OuterLoop 的长块分支；WINDOW_START/STOP 亦走此分支）
static int OuterLoopLong(double Spectrum[3][192], int windowType, int bitRateLimit,
                         int sampleRateIdx, const double xmin[42], OuterLoopResult *out) {
    int scalefactors[21];
    for (int i = 0; i < 21; i++) scalefactors[i] = 0;

    double xminForLong[21];
    for (int i = 0; i < 21; i++) xminForLong[i] = xmin[i];

    int outerLoopCount = 0;
    double ifqstep = 1.4142135623730951;

    while (outerLoopCount < 100) {
        InnerLoopResult inner = InnerLoop(Spectrum, windowType, bitRateLimit, sampleRateIdx);
        if (!inner.success) {
            if (inner.codeBits.data != NULL) aqua_bw_free(&inner.codeBits);
            out->success = 0;
            return 0;
        }

        // 计算量化噪声
        double xfsf[21];
        QuantNoise(Spectrum[0], inner.quantized, inner.globalGain - 210,
                   LONG_BLOCK, sampleRateIdx, xfsf);

        // 对所有超限的 SFB 放大一步
        int nOver = 0;
        for (int sbindex = 0; sbindex < 21; sbindex++) {
            if (xfsf[sbindex] > xminForLong[sbindex]) {
                nOver++;
                xminForLong[sbindex] *= (ifqstep * ifqstep);
                scalefactors[sbindex]++;
                int start = SFB_LR[sampleRateIdx][sbindex][0];
                int end = SFB_LR[sampleRateIdx][sbindex][1];
                for (int i = start; i <= end; i++) {
                    Spectrum[0][i] *= ifqstep;
                }
            }
        }

        // 保存尺度因子
        for (int i = 0; i < 21; i++) {
            out->scalefactors[0][i] = scalefactors[i];
        }
        if (out->inner.codeBits.data != NULL) {
            aqua_bw_free(&out->inner.codeBits);
        }
        out->inner = inner; // codeBits 所有权转移
        out->globalGain = inner.globalGain;
        out->subblockGain[0] = out->subblockGain[1] = out->subblockGain[2] = 0;

        // 检查退出条件
        int isExit = false;
        int isAllSfbAmplified = true;
        for (int sb = 0; sb < 21; sb++) {
            if (scalefactors[sb] == 0) { isAllSfbAmplified = false; break; }
        }
        if (isAllSfbAmplified == false) {
            int isScalefactorExceeded = false;
            for (int sb = 0; sb <= 10; sb++) {
                if (scalefactors[sb] >= 15) { isScalefactorExceeded = true; break; }
            }
            if (!isScalefactorExceeded) {
                for (int sb = 11; sb <= 20; sb++) {
                    if (scalefactors[sb] >= 7) { isScalefactorExceeded = true; break; }
                }
            }
            if (isScalefactorExceeded == false) {
                if (nOver <= 0) {
                    isExit = true;
                }
            } else {
                isExit = true;
            }
        } else {
            isExit = true;
        }

        if (isExit) {
            int scalefactorCompress = ScalefactorCompress(out->scalefactors, windowType);
            int part2Length = Part2Length(scalefactorCompress, windowType);
            out->scalefactorCompress = scalefactorCompress;
            out->part2Length = part2Length;
            out->part23Length = part2Length + out->inner.codeBitLen;
            out->success = 1;
            return 1;
        }

        outerLoopCount++;
    }
    if (out->inner.codeBits.data != NULL) {
        aqua_bw_free(&out->inner.codeBits);
    }
    out->success = 0;
    return 0;
}

// 短块外层循环（对应 OuterLoop 的短块分支）
static int OuterLoopShort(double Spectrum[3][192], int bitRateLimit,
                          int sampleRateIdx, const double xmin[42], OuterLoopResult *out) {
    int scalefactors[3][12];
    memset(scalefactors, 0, sizeof(scalefactors));

    double xminForShort[3][12];
    for (int w = 0; w < 3; w++) {
        for (int i = 0; i < 12; i++) {
            xminForShort[w][i] = xmin[i];
        }
    }

    int isFinished[3] = {0, 0, 0};
    int outerLoopCount = 0;
    double ifqstep = 1.4142135623730951;

    while (outerLoopCount < 100) {
        InnerLoopResult inner = InnerLoop(Spectrum, WINDOW_SHORT, bitRateLimit, sampleRateIdx);
        if (!inner.success) {
            if (inner.codeBits.data != NULL) aqua_bw_free(&inner.codeBits);
            out->success = 0;
            return 0;
        }

        // 将量化后的 576 点频谱分解为 3 个 192 点频谱
        int shortSpectrums[3][192];
        demuxShortBlockSpectrum(inner.quantized, sampleRateIdx, shortSpectrums);

        for (int window = 0; window < 3; window++) {
            if (isFinished[window]) continue;

            double xfsf[12];
            QuantNoise(Spectrum[window], shortSpectrums[window], inner.globalGain - 210,
                       WINDOW_SHORT, sampleRateIdx, xfsf);

            int nOver = 0;
            for (int sbindex = 0; sbindex < 12; sbindex++) {
                if (xfsf[sbindex] > xminForShort[window][sbindex]) {
                    nOver++;
                    xminForShort[window][sbindex] *= (ifqstep * ifqstep);
                    scalefactors[window][sbindex]++;
                    int start = SFB_SR[sampleRateIdx][sbindex][0];
                    int end = SFB_SR[sampleRateIdx][sbindex][1];
                    for (int i = start; i <= end; i++) {
                        Spectrum[window][i] *= ifqstep;
                    }
                }
            }

            // 保存尺度因子
            for (int i = 0; i < 12; i++) {
                out->scalefactors[window][i] = scalefactors[window][i];
            }

            // 检查该子块的退出条件
            int isAllSfbAmplified = true;
            for (int sb = 0; sb < 12; sb++) {
                if (scalefactors[window][sb] == 0) { isAllSfbAmplified = false; break; }
            }
            if (isAllSfbAmplified) { isFinished[window] = 1; continue; }

            int isScalefactorExceeded = false;
            for (int sb = 0; sb <= 5; sb++) {
                if (scalefactors[window][sb] >= 15) { isScalefactorExceeded = true; break; }
            }
            if (!isScalefactorExceeded) {
                for (int sb = 6; sb <= 11; sb++) {
                    if (scalefactors[window][sb] >= 7) { isScalefactorExceeded = true; break; }
                }
            }
            if (isScalefactorExceeded) { isFinished[window] = 1; continue; }
            if (nOver <= 0) { isFinished[window] = 1; continue; }
        }

        // 保存内层循环结果
        if (out->inner.codeBits.data != NULL) {
            aqua_bw_free(&out->inner.codeBits);
        }
        out->inner = inner; // codeBits 所有权转移
        out->globalGain = inner.globalGain;
        out->subblockGain[0] = out->subblockGain[1] = out->subblockGain[2] = 0;

        if (isFinished[0] && isFinished[1] && isFinished[2]) {
            int scalefactorCompress = ScalefactorCompress(out->scalefactors, WINDOW_SHORT);
            int part2Length = Part2Length(scalefactorCompress, WINDOW_SHORT);
            out->scalefactorCompress = scalefactorCompress;
            out->part2Length = part2Length;
            out->part23Length = part2Length + out->inner.codeBitLen;
            out->success = 1;
            return 1;
        }

        outerLoopCount++;
    }
    if (out->inner.codeBits.data != NULL) {
        aqua_bw_free(&out->inner.codeBits);
    }
    out->success = 0;
    return 0;
}

// 外层循环（噪声控制循环），对应 OuterLoop
int aqua_outer_loop(double Spectrum[3][192], int windowType, int bitRateLimit,
                    int sampleRateIdx, const double xmin[42], OuterLoopResult *out) {
    if (windowType == WINDOW_SHORT) {
        return OuterLoopShort(Spectrum, bitRateLimit, sampleRateIdx, xmin, out);
    }
    return OuterLoopLong(Spectrum, windowType, bitRateLimit, sampleRateIdx, xmin, out);
}
