/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_encoder.c
//
//    编码器主体流程实现。对齐 src/encoder.js + src/reservoir.js + src/stream.js。
//
/////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include "aqua_encoder.h"
#include "aqua_mdct.h"
#include "aqua_subband.h"
#include "aqua_qloop.h"

// ----------------------------------------------------------------
//  全 局 一 次 性 初 始 化
// ----------------------------------------------------------------
static void aqua_encoder_globals_init(void) {
    static int done = 0;
    if (!done) {
        aqua_mdct_init();
        aqua_qloop_init();
        done = 1;
    }
}

// AQUA_DEBUG=1 时输出逐帧编码详情（默认关闭）
static int encDbgEnabled(void) {
    static const char *env = NULL;
    if (env == NULL) {
        env = getenv("AQUA_DEBUG");
    }
    return (env != NULL && env[0] != '\0' && env[0] != '0');
}

// ----------------------------------------------------------------
//  一 个 channel 的编码结果（对应 encoder.js 的 channel 对象）
// ----------------------------------------------------------------
typedef struct {
    int part23Length;
    int bigvalues;
    int globalGain;
    int scalefactorCompress;
    int windowSwitchingFlag;
    int blockType;
    int tableSelect[3];
    int subblockGain[3];
    int region0Count;
    int region1Count;
    int count1TableSelect;
    int scalefactors[3][21];   // 长块用 [0][0..20]；短块为 3x12（编码端不使用）
    AquaBitsWriter huffBits;   // part3 哈夫曼码流
    int huffBitLen;
} AquaChannel;

static void dumpChannel(int frameNo, int gr, const AquaChannel *c) {
    if (!encDbgEnabled()) return;
    fprintf(stderr, "CFRAME %d GR %d gg=%d p23=%d bv=%d t0=%d t1=%d t2=%d r0=%d r1=%d sc=%d c1t=%d hcl=%d sf=",
            frameNo, gr, c->globalGain, c->part23Length, c->bigvalues,
            c->tableSelect[0], c->tableSelect[1], c->tableSelect[2],
            c->region0Count, c->region1Count, c->scalefactorCompress,
            c->count1TableSelect, c->huffBitLen);
    for (int i = 0; i < 21; i++) fprintf(stderr, "%d,", c->scalefactors[0][i]);
    fprintf(stderr, "\n");
}

// 每 channel 的跨 granule 状态（对应 BUFFER[ch]）
typedef struct {
    double prevSubbands[SUBBANDS][SUBBAND_LEN];
    int prevBlockType;
    double attackAvg[SUBBANDS]; // attack 检测：各子带能量滑动平均
    int attackInit;
} AquaChannelState;

// AQUA_LONG_ONLY=1 时强制长块（复现旧行为/对比用）
static int aquaLongOnlyEnabled(void) {
    static const char *env = NULL;
    if (env == NULL) {
        env = getenv("AQUA_LONG_ONLY");
    }
    return (env != NULL && env[0] != '\0' && env[0] != '0');
}

// 窗口切换状态机（对应 psy.js 的 SwitchWindowType）
static int SwitchWindowType(int prevWindowType, int isAttack) {
    if (prevWindowType == WINDOW_NORMAL) {
        return isAttack ? WINDOW_START : WINDOW_NORMAL;
    } else if (prevWindowType == WINDOW_START) {
        return WINDOW_SHORT;
    } else if (prevWindowType == WINDOW_SHORT) {
        return isAttack ? WINDOW_SHORT : WINDOW_STOP;
    } else { // WINDOW_STOP
        return isAttack ? WINDOW_START : WINDOW_NORMAL;
    }
}

// 确定性 attack 检测（替代 JS 中随机的 Math.random 占位）:
// 高子带能量相对其滑动平均值骤然上升视为瞬态（attack）。
static int DetectAttack(const double subbands[SUBBANDS][SUBBAND_LEN],
                        AquaChannelState *state) {
    double curE[SUBBANDS];
    for (int sb = 0; sb < SUBBANDS; sb++) {
        double e = 0.0;
        for (int t = 0; t < SUBBAND_LEN; t++) {
            double v = subbands[sb][t];
            e += v * v;
        }
        curE[sb] = e;
    }

    int attack = 0;
    if (state->attackInit) {
        /* 从第 8 子带起（约 4 kHz 以上）的能量骤增视为瞬态 */
        for (int sb = 8; sb < SUBBANDS; sb++) {
            if (curE[sb] > state->attackAvg[sb] * 8.0) {
                attack = 1;
                break;
            }
        }
    }

    for (int sb = 0; sb < SUBBANDS; sb++) {
        state->attackAvg[sb] = state->attackInit
            ? 0.5 * state->attackAvg[sb] + 0.5 * curE[sb]
            : curE[sb];
    }
    state->attackInit = 1;
    return attack;
}

// ----------------------------------------------------------------
//  比 特 储 备
// ----------------------------------------------------------------

static void SetReservoirMax(AquaEncoder *enc) {
    if (enc->reservoirMax > 4088) {
        enc->reservoirMax = 4088;
    }
}

static int AllocateBudget(AquaEncoder *enc, int perceptualEntropy, int meanBitsPerChannel) {
    int budget = meanBitsPerChannel;
    if (enc->reservoirSize == 0) {
        return (budget > 4095) ? 4095 : budget;
    }
    double moreBits = perceptualEntropy * 3.1 - meanBitsPerChannel;
    int addBits = 0;
    if (moreBits > 100) {
        double m = (moreBits < 0.6 * enc->reservoirSize) ? moreBits : 0.6 * enc->reservoirSize;
        addBits = (int)m;
    }
    budget += addBits;
    double overBits = enc->reservoirSize - 0.8 * enc->reservoirMax - addBits;
    budget += (overBits > 0) ? (int)overBits : 0;
    int r = (budget > 4095) ? 4095 : budget;
    return (int)js_round((double)r);
}

static void ReturnUnusedBits(AquaEncoder *enc, int part23Length, int meanBitsPerChannel) {
    enc->reservoirSize += (meanBitsPerChannel - part23Length);
}

static void RegulateAndStuff(AquaEncoder *enc, AquaChannel granules[2][MAX_CHANNELS]) {
    int stuffingBits = 0;
    if (enc->reservoirSize > enc->reservoirMax) {
        stuffingBits += (int)(enc->reservoirSize - enc->reservoirMax);
        enc->reservoirSize = enc->reservoirMax;
    }
    int remainder = (int)(enc->reservoirSize & 7);
    enc->reservoirSize -= remainder;
    stuffingBits += remainder;

    int isFinished = 0;
    for (int gr = 0; gr < 2; gr++) {
        for (int ch = 0; ch < enc->channels; ch++) {
            if (granules[gr][ch].part23Length + stuffingBits > 4095) {
                stuffingBits -= (4095 - granules[gr][ch].part23Length);
                granules[gr][ch].part23Length = 4095;
            } else {
                granules[gr][ch].part23Length += stuffingBits;
                isFinished = 1;
                break;
            }
        }
        if (isFinished) break;
    }
}

// ----------------------------------------------------------------
//  编 码 一 个 channel（对应 EncodeChannel，心理声学模型未启用）
// ----------------------------------------------------------------
static void EncodeChannel(const double *PCM, long numSamples, long offset,
                          int meanBitsPerChannel, int sampleRateIdx, int bitRateIdx,
                          AquaChannelState *state, AquaEncoder *enc, AquaChannel *channel) {
    (void)bitRateIdx;
    // 分析子带滤波
    double subbands[SUBBANDS][SUBBAND_LEN];
    aqua_analysis_subband_filter(PCM, numSamples, offset, subbands);
    if (encDbgEnabled()) {
        static int sdc = 0;
        if (sdc++ == 0) {
            fprintf(stderr, "C_SUB ");
            for (int b = 0; b < SUBBANDS; b++) {
                for (int t = 0; t < SUBBAND_LEN; t++) {
                    fprintf(stderr, "%.9f ", subbands[b][t]);
                }
            }
            fprintf(stderr, "\n");
        }
    }

    // 窗切换：确定性 attack 检测 + SwitchWindowType 状态机（对应 psy.js）
    int blockType;
    if (aquaLongOnlyEnabled()) {
        blockType = WINDOW_NORMAL;
    } else {
        int isAttack = DetectAttack(subbands, state);
        blockType = SwitchWindowType(state->prevBlockType, isAttack);
    }

    // 心理声学模型未启用：xmin 全为最大值
    double xmin[42];
    for (int i = 0; i < 42; i++) {
        xmin[i] = DBL_MAX;
    }

    // 时频变换（长块或短块）
    double outLong[GRANULE_SAMPLES];
    double outShort[3][192];
    aqua_granule_mdct(subbands, state->prevSubbands, blockType, outLong, outShort);
    if (encDbgEnabled()) {
        static int mdc = 0;
        if (mdc++ == 0) {
            fprintf(stderr, "C_MDCT ");
            for (int i = 0; i < GRANULE_SAMPLES; i++) {
                fprintf(stderr, "%.9f ", outLong[i]);
            }
            fprintf(stderr, "\n");
        }
    }
    double Spectrum[3][192];
    if (blockType == WINDOW_SHORT) {
        for (int w = 0; w < 3; w++) {
            memcpy(Spectrum[w], outShort[w], sizeof(Spectrum[w]));
        }
    } else {
        memcpy(Spectrum, outLong, sizeof(outLong));
    }

    // 分配比特预算
    int budget = AllocateBudget(enc, 0, meanBitsPerChannel);
    int huffmanBudget = budget - ((blockType != WINDOW_SHORT) ? 74 : 126);

    // 量化循环（外层）
    OuterLoopResult outer;
    memset(&outer, 0, sizeof(outer));
    outer.inner.codeBits.data = NULL;
    if (!aqua_outer_loop(Spectrum, blockType, huffmanBudget, sampleRateIdx, xmin, &outer)) {
        if (encDbgEnabled()) {
            fprintf(stderr, "OUTER_FAIL\n");
        }
        channel->part23Length = 0;
        channel->huffBits.data = NULL;
        return;
    }

    // 构造编码结果
    if (encDbgEnabled()) {
        long sumAbs = 0;
        for (int i = 0; i < 576; i++) {
            int v = outer.inner.quantized[i];
            sumAbs += (v < 0) ? -v : v;
        }
        fprintf(stderr, "EQ gg=%d p23=%d bv=%d sl=%d sum=%ld\n",
                outer.globalGain, outer.part23Length,
                outer.inner.huffman.bigvalues, outer.scalefactorCompress, sumAbs);
    }

    channel->part23Length = outer.part23Length;
    channel->bigvalues = outer.inner.huffman.bigvalues;
    channel->globalGain = outer.globalGain;
    channel->scalefactorCompress = outer.scalefactorCompress;
    channel->windowSwitchingFlag = (blockType == WINDOW_NORMAL) ? 0 : 1;
    channel->blockType = blockType;
    channel->tableSelect[0] = outer.inner.huffman.tableSelect[0];
    channel->tableSelect[1] = outer.inner.huffman.tableSelect[1];
    channel->tableSelect[2] = outer.inner.huffman.tableSelect[2];
    channel->subblockGain[0] = outer.subblockGain[0];
    channel->subblockGain[1] = outer.subblockGain[1];
    channel->subblockGain[2] = outer.subblockGain[2];
    channel->region0Count = outer.inner.huffman.region0Count;
    channel->region1Count = outer.inner.huffman.region1Count;
    channel->count1TableSelect = outer.inner.huffman.smallvalueTableSelect;
    for (int w = 0; w < 3; w++) {
        for (int i = 0; i < 21; i++) {
            channel->scalefactors[w][i] = outer.scalefactors[w][i];
        }
    }
    channel->huffBitLen = outer.inner.codeBitLen;
    channel->huffBits = outer.inner.codeBits; // 转移 codeBits 所有权，避免二次释放

    // 返还剩余比特
    ReturnUnusedBits(enc, channel->part23Length, meanBitsPerChannel);

    // 保存前一个 granule 的结果
    memcpy(state->prevSubbands, subbands, sizeof(subbands));
    state->prevBlockType = blockType;
}

// ----------------------------------------------------------------
//  编 码 一 个 granule（对应 EncodeGranule，含一个或两个声道）
// ----------------------------------------------------------------
static void EncodeGranule(const double *pcm[MAX_CHANNELS], long numSamples, long offset,
                          int meanBitsPerGranule, int sampleRateIdx, int bitRateIdx,
                          AquaEncoder *enc, AquaChannelState *states,
                          AquaChannel *channels) {
    for (int ch = 0; ch < enc->channels; ch++) {
        EncodeChannel(pcm[ch], numSamples, offset, meanBitsPerGranule / enc->channels,
                      sampleRateIdx, bitRateIdx, &states[ch], enc, &channels[ch]);
    }
}

// ----------------------------------------------------------------
//  装 配 帧 比 特 流（对应 stream.js 的 FormatFrameBitStream）
// ----------------------------------------------------------------
static const int SF_COMPRESS_INDEX_ENC[16][2] = {
    {0,0},{0,1},{0,2},{0,3},{3,0},{1,1},{1,2},{1,3},
    {2,1},{2,2},{2,3},{3,1},{3,2},{3,3},{4,2},{4,3}
};

static void FormatFrameBitStream(AquaEncoder *enc, AquaChannel granules[2][MAX_CHANNELS],
                                 int isPadding, int mainDataBegin, AquaBitsWriter *out) {
    AquaBitsWriter fixedData;
    AquaBitsWriter mainData;
    aqua_bw_init(&fixedData);
    aqua_bw_init(&mainData);

    // ---- Header (32 bits) ----
    aqua_bw_put(&fixedData, 0xfffu, 12);                          // syncword
    aqua_bw_put(&fixedData, 1u, 1);                               // ID = 1 (MPEG1)
    aqua_bw_put(&fixedData, 1u, 2);                               // Layer 3 = 01
    aqua_bw_put(&fixedData, 1u, 1);                               // no CRC
    aqua_bw_put(&fixedData, AQUA_BR_CODE[enc->bitRateIdx], 4);    // bitrate index
    aqua_bw_put(&fixedData, AQUA_SR_CODE[enc->sampleRateIdx], 2); // sample rate index
    aqua_bw_put(&fixedData, isPadding ? 1u : 0u, 1);              // padding bit
    aqua_bw_put(&fixedData, 0u, 1);                               // private bit
    aqua_bw_put(&fixedData, (enc->channels >= 2) ? 0u : 3u, 2);   // channel mode
    aqua_bw_put(&fixedData, 0u, 2);                               // mode extension
    aqua_bw_put(&fixedData, 1u, 2);                               // "01": copyright=0, original=1
    aqua_bw_put(&fixedData, 0u, 2);                               // emphasis

    // ---- SideInfo ----
    aqua_bw_put(&fixedData, (uint32_t)mainDataBegin, 9);          // main_data_begin
    aqua_bw_put(&fixedData, 0u, (enc->channels >= 2) ? 3 : 5);    // private bits
    for (int ch = 0; ch < enc->channels; ch++) {
        aqua_bw_put(&fixedData, 0u, 4);                           // scfsi
    }

    for (int gr = 0; gr < 2; gr++) {
        for (int ch = 0; ch < enc->channels; ch++) {
            AquaChannel *c = &granules[gr][ch];
            aqua_bw_put(&fixedData, (uint32_t)c->part23Length, 12);
            aqua_bw_put(&fixedData, (uint32_t)c->bigvalues, 9);
            aqua_bw_put(&fixedData, (uint32_t)c->globalGain, 8);
            aqua_bw_put(&fixedData, (uint32_t)c->scalefactorCompress, 4);
            aqua_bw_put(&fixedData, (uint32_t)c->windowSwitchingFlag, 1);
            if (c->windowSwitchingFlag == 1) {
                aqua_bw_put(&fixedData, (uint32_t)c->blockType, 2);
                aqua_bw_put(&fixedData, 0u, 1);                   // mixed_block_flag
                aqua_bw_put(&fixedData, (uint32_t)c->tableSelect[0], 5);
                aqua_bw_put(&fixedData, (uint32_t)c->tableSelect[1], 5);
                for (int w = 0; w < 3; w++) {
                    aqua_bw_put(&fixedData, 0u, 3);               // subblock_gain
                }
            } else {
                aqua_bw_put(&fixedData, (uint32_t)c->tableSelect[0], 5);
                aqua_bw_put(&fixedData, (uint32_t)c->tableSelect[1], 5);
                aqua_bw_put(&fixedData, (uint32_t)c->tableSelect[2], 5);
                aqua_bw_put(&fixedData, (uint32_t)c->region0Count, 4);
                aqua_bw_put(&fixedData, (uint32_t)c->region1Count, 3);
            }
            aqua_bw_put(&fixedData, 0u, 1);                       // preflag
            aqua_bw_put(&fixedData, 0u, 1);                       // scalefactor_scale
            aqua_bw_put(&fixedData, (c->count1TableSelect == 0) ? 0u : 1u, 1);
        }
    }

    // ---- 主数据（Part2 尺度因子 + Part3 哈夫曼 + 按 part23Length 填充） ----
    for (int gr = 0; gr < 2; gr++) {
        for (int ch = 0; ch < enc->channels; ch++) {
            AquaChannel *c = &granules[gr][ch];
            int startBits = aqua_bw_len(&mainData);
            int slen1 = SF_COMPRESS_INDEX_ENC[c->scalefactorCompress][0];
            int slen2 = SF_COMPRESS_INDEX_ENC[c->scalefactorCompress][1];
            if ((c->windowSwitchingFlag == 1) && (c->blockType == WINDOW_SHORT)) {
                for (int sfb = 0; sfb < 6; sfb++) {
                    for (int w = 0; w < 3; w++) {
                        aqua_bw_put(&mainData, (uint32_t)c->scalefactors[w][sfb], slen1);
                    }
                }
                for (int sfb = 6; sfb < 12; sfb++) {
                    for (int w = 0; w < 3; w++) {
                        aqua_bw_put(&mainData, (uint32_t)c->scalefactors[w][sfb], slen2);
                    }
                }
            } else {
                for (int sfb = 0; sfb < 11; sfb++) {
                    aqua_bw_put(&mainData, (uint32_t)c->scalefactors[0][sfb], slen1);
                }
                for (int sfb = 11; sfb < 21; sfb++) {
                    aqua_bw_put(&mainData, (uint32_t)c->scalefactors[0][sfb], slen2);
                }
            }
            // Part3：哈夫曼码流（逐比特写入）
            {
                const uint8_t *hd = aqua_bw_packed(&c->huffBits);
                for (int i = 0; i < c->huffBitLen; i++) {
                    aqua_bw_put(&mainData,
                                (uint32_t)((hd[i >> 3] >> (7 - (i & 7))) & 1u), 1);
                }
            }
            // 将该声道主数据补零至 part23Length
            int usedBits = aqua_bw_len(&mainData) - startBits;
            int padBits = c->part23Length - usedBits;
            if (padBits > 0) {
                aqua_bw_zero(&mainData, padBits);
            }
        }
    }

    // ---- 装配字节流：main_data_begin 之前的 mainData 先写，再写帧头与边信息 ----
    {
        const uint8_t *m = aqua_bw_packed(&mainData);
        const uint8_t *f = aqua_bw_packed(&fixedData);
        int mainNBytes = aqua_bw_nbytes(&mainData);
        int fixedNBytes = aqua_bw_nbytes(&fixedData);
        for (int i = 0; i < mainDataBegin && i < mainNBytes; i++) {
            aqua_bw_put(out, m[i], 8);
        }
        for (int i = 0; i < fixedNBytes; i++) {
            aqua_bw_put(out, f[i], 8);
        }
        for (int i = mainDataBegin; i < mainNBytes; i++) {
            aqua_bw_put(out, m[i], 8);
        }
        if (isPadding) {
            aqua_bw_put(out, 0x80u, 8);
        }
    }

    aqua_bw_free(&fixedData);
    aqua_bw_free(&mainData);
}

// ----------------------------------------------------------------
//  编 码 一 帧（对应 Aqua_EncodeFrame，含两个 granule）
// ----------------------------------------------------------------
int aqua_encoder_encode_frame(AquaEncoder *enc, const double *pcm[MAX_CHANNELS],
                              long numSamples, long offset, AquaBitsWriter *out) {
    int mainDataBegin = (int)enc->reservoirSize;

    // 帧间距（bits，8 的倍数）
    int frameLength = (int)((long)AQUA_BR_VALUE[enc->bitRateIdx] * FRAME_SAMPLES /
                            AQUA_SR_VALUE[enc->sampleRateIdx] / 8) * 8;
    // 每个 Granule 的平均长度（含所有声道，bits）
    int sideLength = (enc->channels >= 2) ? 256 : 136;
    int meanBitsPerGranule = (frameLength - sideLength - 32) / 2;

    // 设置比特储备的最大容量
    SetReservoirMax(enc);

    // 从编码器上下文恢复各声道跨 granule 状态
    AquaChannelState states[MAX_CHANNELS];
    for (int ch = 0; ch < enc->channels; ch++) {
        memcpy(states[ch].prevSubbands, enc->prevSubbands[ch],
               sizeof(states[ch].prevSubbands));
        states[ch].prevBlockType = enc->prevBlockType[ch];
        memcpy(states[ch].attackAvg, enc->attackAvg[ch],
               sizeof(states[ch].attackAvg));
        states[ch].attackInit = enc->attackInit[ch];
    }

    AquaChannel channels[2][MAX_CHANNELS];
    memset(channels, 0, sizeof(channels));

    // 分别编码两个 granules
    EncodeGranule(pcm, numSamples, offset, meanBitsPerGranule,
                  enc->sampleRateIdx, enc->bitRateIdx, enc, states, channels[0]);
    EncodeGranule(pcm, numSamples, offset + 576, meanBitsPerGranule,
                  enc->sampleRateIdx, enc->bitRateIdx, enc, states, channels[1]);

    {
        static int dbgFrame = 0;
        if (encDbgEnabled()) {
            for (int gr = 0; gr < 2; gr++) {
                for (int ch = 0; ch < enc->channels; ch++) {
                    dumpChannel(dbgFrame, gr, &channels[gr][ch]);
                }
            }
        }
        dbgFrame++;
    }

    // 调整比特储备池，将多余比特填充进 main_data
    RegulateAndStuff(enc, channels);

    // p22 与旧版一致：每帧都填充 padding 字节（槽）
    int isPadding = 1;

    // 组装比特流并追加到 out
    FormatFrameBitStream(enc, channels, isPadding, mainDataBegin, out);

    // 保存各声道跨 granule 状态
    for (int ch = 0; ch < enc->channels; ch++) {
        memcpy(enc->prevSubbands[ch], states[ch].prevSubbands,
               sizeof(states[ch].prevSubbands));
        enc->prevBlockType[ch] = states[ch].prevBlockType;
        memcpy(enc->attackAvg[ch], states[ch].attackAvg,
               sizeof(states[ch].attackAvg));
        enc->attackInit[ch] = states[ch].attackInit;
    }

    return 1;
}

// ----------------------------------------------------------------
//  初 始 化 / 重 置
// ----------------------------------------------------------------
int aqua_encoder_init(AquaEncoder *enc, int channels, int sampleRateIdx, int bitRateIdx) {
    aqua_encoder_globals_init();
    return aqua_encoder_reset(enc, channels, sampleRateIdx, bitRateIdx);
}

int aqua_encoder_reset(AquaEncoder *enc, int channels, int sampleRateIdx, int bitRateIdx) {
    if (channels < 1 || channels > MAX_CHANNELS ||
        sampleRateIdx < SAMPLE_RATE_32000 || sampleRateIdx > SAMPLE_RATE_48000 ||
        bitRateIdx < BIT_RATE_64K || bitRateIdx > BIT_RATE_320K) {
        return 0;
    }
    memset(enc, 0, sizeof(*enc));
    enc->channels = channels;
    enc->sampleRateIdx = sampleRateIdx;
    enc->bitRateIdx = bitRateIdx;
    enc->reservoirMax = 0;
    enc->reservoirSize = 0;
    for (int ch = 0; ch < MAX_CHANNELS; ch++) {
        enc->prevBlockType[ch] = WINDOW_NORMAL;
    }
    return 1;
}
