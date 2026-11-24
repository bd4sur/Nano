/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_huff.c
//
//    哈夫曼编码实现。逐位对齐 src/huffman.js。
//
/////////////////////////////////////////////////////////////////


#include <stdlib.h>
#include <string.h>
#include "aqua_huff.h"

// ------------------------------------------------------------------
//  哈 夫 曼 解 码（树形，与编码共用同一码本）
// ------------------------------------------------------------------

typedef struct {
    int next[2];   // 子节点下标，-1 表示无
    int isLeaf;    // 是否叶子
    int x, y, v, w;// 叶子代表的值
} HuffNode;

static HuffNode htNodes[34][1024];
static int htNodeCount[34];
static int htBuilt = 0;

static void htBuild(void) {
    if (htBuilt) return;
    for (int t = 0; t < 34; t++) {
        HuffNode *nodes = htNodes[t];
        int *count = &htNodeCount[t];
        nodes[0].next[0] = -1;
        nodes[0].next[1] = -1;
        nodes[0].isLeaf = 0;
        nodes[0].x = nodes[0].y = nodes[0].v = nodes[0].w = 0;
        *count = 1;

        if (t <= 31) {
            // 大值表：逐码插入
            for (int x = 0; x < 16; x++) {
                for (int y = 0; y < 16; y++) {
                    int len = HT_LEN[t][x * 16 + y];
                    if (len <= 0) continue;
                    uint32_t code = HT_CODE[t][x * 16 + y];
                    int node = 0;
                    for (int b = 0; b < len; b++) {
                        int bit = (int)((code >> (31 - b)) & 1u);
                        if (b == len - 1) {
                            // 叶子
                            nodes[node].next[bit] = *count;
                            HuffNode *leaf = &nodes[*count];
                            leaf->next[0] = leaf->next[1] = -1;
                            leaf->isLeaf = 1;
                            leaf->x = x;
                            leaf->y = y;
                            leaf->v = leaf->w = 0;
                            (*count)++;
                        } else {
                            if (nodes[node].next[bit] < 0) {
                                nodes[node].next[bit] = *count;
                                HuffNode *n = &nodes[*count];
                                n->next[0] = n->next[1] = -1;
                                n->isLeaf = 0;
                                (*count)++;
                            }
                            node = nodes[node].next[bit];
                        }
                    }
                }
            }
            if (*count == 1) {
                // 空表（如 4、14）：根直接作为叶子，返回 (0,0) 不消耗比特
                nodes[0].isLeaf = 1;
                nodes[0].x = 0;
                nodes[0].y = 0;
            }
        } else {
            // 四元组表 32/33
            int qt = t - 32;
            for (int key = 0; key < 16; key++) {
                int len = HT_QUAD_LEN[qt][key];
                if (len <= 0) continue;
                uint32_t code = HT_QUAD_CODE[qt][key];
                int node = 0;
                for (int b = 0; b < len; b++) {
                    int bit = (int)((code >> (7 - b)) & 1u);
                    if (b == len - 1) {
                        nodes[node].next[bit] = *count;
                        HuffNode *leaf = &nodes[*count];
                        leaf->next[0] = leaf->next[1] = -1;
                        leaf->isLeaf = 1;
                        leaf->v = (key >> 3) & 1;
                        leaf->w = (key >> 2) & 1;
                        leaf->x = (key >> 1) & 1;
                        leaf->y = key & 1;
                        (*count)++;
                    } else {
                        if (nodes[node].next[bit] < 0) {
                            nodes[node].next[bit] = *count;
                            HuffNode *n = &nodes[*count];
                            n->next[0] = n->next[1] = -1;
                            n->isLeaf = 0;
                            (*count)++;
                        }
                        node = nodes[node].next[bit];
                    }
                }
            }
        }
    }
    htBuilt = 1;
}

int aqua_huff_decode_word(AquaBitsReader *br, int tableNum,
                          int *x, int *y, int *v, int *w) {
    htBuild();
    if (tableNum < 0 || tableNum > 33) {
        *x = *y = *v = *w = 0;
        return 0;
    }
    HuffNode *nodes = htNodes[tableNum];

    if (tableNum <= 31) {
        if (nodes[0].isLeaf) {
            *x = nodes[0].x;
            *y = nodes[0].y;
            *v = *w = 0;
        } else {
            int node = 0;
            int guard = 0;
            while (!nodes[node].isLeaf) {
                int bit = (int)aqua_br_get(br, 1);
                node = nodes[node].next[bit];
                if (node < 0 || ++guard > 64) {
                    *x = *y = *v = *w = 0;
                    return 0;
                }
            }
            *x = nodes[node].x;
            *y = nodes[node].y;
            *v = *w = 0;
        }
        // linbits 与符号位（与 js-mp3 decode 一致）
        int linbits = HT_LINBITS[tableNum];
        if (linbits != 0 && *x == 15) {
            *x += (int)aqua_br_get(br, linbits);
        }
        if (*x != 0 && aqua_br_get(br, 1) == 1) {
            *x = -*x;
        }
        if (linbits != 0 && *y == 15) {
            *y += (int)aqua_br_get(br, linbits);
        }
        if (*y != 0 && aqua_br_get(br, 1) == 1) {
            *y = -*y;
        }
        return 1;
    } else {
        // 四元组
        if (nodes[0].isLeaf) {
            *v = nodes[0].v;
            *w = nodes[0].w;
            *x = nodes[0].x;
            *y = nodes[0].y;
        } else {
            int node = 0;
            int guard = 0;
            while (!nodes[node].isLeaf) {
                int bit = (int)aqua_br_get(br, 1);
                node = nodes[node].next[bit];
                if (node < 0 || ++guard > 64) {
                    *x = *y = *v = *w = 0;
                    return 0;
                }
            }
            *v = nodes[node].v;
            *w = nodes[node].w;
            *x = nodes[node].x;
            *y = nodes[node].y;
        }
        // 符号位
        if (*v != 0 && aqua_br_get(br, 1) == 1) *v = -*v;
        if (*w != 0 && aqua_br_get(br, 1) == 1) *w = -*w;
        if (*x != 0 && aqua_br_get(br, 1) == 1) *x = -*x;
        if (*y != 0 && aqua_br_get(br, 1) == 1) *y = -*y;
        return 1;
    }
}

// 返回某正整数的指定长度的二进制串（JS BinaryString，含负数时的行为）
// 这里直接在比特写入器上按序写出（MSB 优先）。
static void bw_binstr(AquaBitsWriter *bw, int intNumber, int length) {
    int shift = (length + 3) >> 2;
    int remain = 4 - (length % 4);
    char seq[64];
    int seqlen = 0;
    static const char nibble[64] =
        "0000000100100011010001010110011110001001101010111100110111101111";
    for (int i = 0; i < shift; i++) {
        int rightBits = ((unsigned)intNumber) & 15u;
        // 前插 4 位
        for (int k = seqlen - 1; k >= 0; k--) {
            seq[k + 4] = seq[k];
        }
        for (int k = 0; k < 4; k++) {
            seq[k] = nibble[rightBits * 4 + k];
        }
        seqlen += 4;
        intNumber >>= 4; // 算术右移（与 JS 一致）
    }
    int start = (remain == 4) ? 0 : remain;
    for (int i = start; i < seqlen; i++) {
        aqua_bw_put(bw, (seq[i] == '1') ? 1u : 0u, 1);
    }
}

// 量化后频谱分区（对应 PartitionQuantizedSpectrum）
static void partitionQuantizedSpectrum(const int *spect, int blockType,
                                       int partBig[2], int partSmall[2]) {
    if (blockType == WINDOW_SHORT) {
        partBig[0] = 0;
        partBig[1] = 576;
        partSmall[0] = 576;
        partSmall[1] = 576;
        return;
    }
    int rzero = 0;
    int i;
    for (i = 576; i > 1; i -= 2) {
        if (spect[i - 1] == 0 && spect[i - 2] == 0) {
            rzero++;
        } else {
            break;
        }
    }
    int count1 = 0;
    for (; i > 3; i -= 4) {
        if (abs(spect[i - 1]) <= 1 && abs(spect[i - 2]) <= 1 &&
            abs(spect[i - 3]) <= 1 && abs(spect[i - 4]) <= 1) {
            count1++;
        } else {
            break;
        }
    }
    int bigvalues = i / 2;
    partBig[0] = 0;
    partBig[1] = bigvalues * 2;
    partSmall[0] = bigvalues * 2;
    partSmall[1] = count1 * 4 + bigvalues * 2;
}

// 大值二元组编码（对应 EncodeDuple）；返回码长与码字、linbits 信息
static int encodeDuple(int x, int y, int tableSelect,
                       uint32_t *code, int *codeLen, int *linX, int *linY) {
    int ax = abs(x);
    int ay = abs(y);
    int linbits = HT_LINBITS[tableSelect];
    int hasX = 0, hasY = 0;
    *linX = 0;
    *linY = 0;
    if (ax >= 15) {
        *linX = ax - 15;
        hasX = 1;
        ax = 15;
    }
    if (ay >= 15) {
        *linY = ay - 15;
        hasY = 1;
        ay = 15;
    }
    int idx = 16 * ax + ay;
    *code = HT_CODE[tableSelect][idx];
    *codeLen = HT_LEN[tableSelect][idx];
    return (hasX << 1) | hasY;
}

// 小值四元组编码（对应 EncodeQuadruple）
static void encodeQuadruple(int v, int w, int x, int y, int tableSelect,
                            uint32_t *code, int *codeLen) {
    int key = (abs(v) << 3) + (abs(w) << 2) + (abs(x) << 1) + abs(y);
    *code = HT_QUAD_CODE[tableSelect][key];
    *codeLen = HT_QUAD_LEN[tableSelect][key];
}

int aqua_huff_encode(const int *qs576, int blockType, int sampleRateIdx,
                     AquaBitsWriter *outBits, AquaHuffmanResult *res) {
    int Bigvalues = -1;
    int tableSelect[3] = { -1, -1, -1 };
    int region0Count = -1;
    int region1Count = -1;
    int smallvalueTableSelect = 0;

    // 最大值超过 8206 则置失败标志（返回 false 由调用方处理）
    for (int i = 0; i < 576; i++) {
        if (abs(qs576[i]) > 8206) {
            res->bigvalues = 0;
            return 0;
        }
    }

    int partBig[2], partSmall[2];
    partitionQuantizedSpectrum(qs576, blockType, partBig, partSmall);

    Bigvalues = (partBig[1] - partBig[0]) / 2;

    // 处理大值区
    if (partBig[1] > 0) {
        int region01 = 0, region12 = 0;
        int lastSFBIndexOfBigvalues = -1;
        int bigvaluesEndIndex = partBig[1] - 1;

        if (blockType == WINDOW_NORMAL) {
            const int(*SFBands)[2] = SFB_LR[sampleRateIdx];
            for (int sfb = 0; sfb < 21; sfb++) {
                int a = SFBands[sfb][0], b = SFBands[sfb][1];
                if (sfb == 20) {
                    b = 575;
                }
                if (bigvaluesEndIndex > 0 && bigvaluesEndIndex >= a && bigvaluesEndIndex <= b) {
                    lastSFBIndexOfBigvalues = sfb;
                    break;
                }
            }

            int sfbNumberInBigvalues = lastSFBIndexOfBigvalues + 1;
            int region0SFBNum = (int)js_round((double)sfbNumberInBigvalues / 3.0);
            int region1SFBNum =
                sfbNumberInBigvalues - (int)js_round((double)sfbNumberInBigvalues / 4.0) - region0SFBNum;
            int region2SFBNum = sfbNumberInBigvalues - region0SFBNum - region1SFBNum;

            if (region1SFBNum <= 0) {
                region1SFBNum = region2SFBNum;
                region2SFBNum = 0;
            }

            region0SFBNum = (region0SFBNum > 16) ? 16 : region0SFBNum;
            region1SFBNum = (region1SFBNum > 8) ? 8 : region1SFBNum;

            region0Count = region0SFBNum - 1;
            region1Count = region1SFBNum - 1;

            region01 = SFBands[region0SFBNum][0];
            region12 = SFBands[region0SFBNum + region1SFBNum][0];
        } else {
            // 编码端只用长块；其余块类型保留结构（不会进入）
            if (blockType == WINDOW_SHORT) {
                region0Count = 8;
            } else {
                region0Count = 7;
            }
            region1Count = 36;
            const int(*SFBands)[2] = SFB_SR[sampleRateIdx];
            for (int sfb = 0; sfb < 3; sfb++) {
                region01 += (SFBands[sfb][1] - SFBands[sfb][0] + 1) * 3;
            }
            region12 = partBig[1];
        }

        // 计算每个 region 的最大值
        int maxValue0 = -1, maxValue1 = -1, maxValue2 = -1;
        for (int i = 0; i < region01; i++) {
            int qa = abs(qs576[i]);
            if (qa > maxValue0) maxValue0 = qa;
        }
        for (int i = region01; i < region12; i++) {
            int qa = abs(qs576[i]);
            if (qa > maxValue1) maxValue1 = qa;
        }
        for (int i = region12; i < partBig[1]; i++) {
            int qa = abs(qs576[i]);
            if (qa > maxValue2) maxValue2 = qa;
        }

        int sel0 = -1, sel1 = -1, sel2 = -1;
        for (int i = 0; i < 32; i++) {
            if (i == 4 || i == 14) continue; // 与 JS HuffmanTableDuple 中的 null 表一致
            int mv = HT_MAXVALUE[i];
            if (sel0 < 0 && maxValue0 <= mv) sel0 = i;
            if (sel1 < 0 && maxValue1 <= mv) sel1 = i;
            if (sel2 < 0 && maxValue2 <= mv) sel2 = i;
            if (sel0 >= 0 && sel1 >= 0 && sel2 >= 0) break;
        }
        tableSelect[0] = sel0;
        tableSelect[1] = sel1;
        tableSelect[2] = sel2;

        uint32_t code;
        int codeLen;
        int linX, linY;
        int linbits;
        // region 0
        for (int i = 0; i < region01; i += 2) {
            int x = qs576[i], y = qs576[i + 1];
            int mask = encodeDuple(x, y, sel0, &code, &codeLen, &linX, &linY);
            linbits = HT_LINBITS[sel0];
            aqua_bw_put(outBits, code >> (32 - codeLen), codeLen);
            if (mask & 2) aqua_bw_put(outBits, (uint32_t)linX, linbits);
            if (x != 0) aqua_bw_put(outBits, (x < 0) ? 1u : 0u, 1);
            if (mask & 1) aqua_bw_put(outBits, (uint32_t)linY, linbits);
            if (y != 0) aqua_bw_put(outBits, (y < 0) ? 1u : 0u, 1);
        }
        // region 1
        for (int i = region01; i < region12; i += 2) {
            int x = qs576[i], y = qs576[i + 1];
            int mask = encodeDuple(x, y, sel1, &code, &codeLen, &linX, &linY);
            linbits = HT_LINBITS[sel1];
            aqua_bw_put(outBits, code >> (32 - codeLen), codeLen);
            if (mask & 2) aqua_bw_put(outBits, (uint32_t)linX, linbits);
            if (x != 0) aqua_bw_put(outBits, (x < 0) ? 1u : 0u, 1);
            if (mask & 1) aqua_bw_put(outBits, (uint32_t)linY, linbits);
            if (y != 0) aqua_bw_put(outBits, (y < 0) ? 1u : 0u, 1);
        }
        // region 2
        for (int i = region12; i < partBig[1]; i += 2) {
            int x = qs576[i], y = qs576[i + 1];
            int mask = encodeDuple(x, y, sel2, &code, &codeLen, &linX, &linY);
            linbits = HT_LINBITS[sel2];
            aqua_bw_put(outBits, code >> (32 - codeLen), codeLen);
            if (mask & 2) aqua_bw_put(outBits, (uint32_t)linX, linbits);
            if (x != 0) aqua_bw_put(outBits, (x < 0) ? 1u : 0u, 1);
            if (mask & 1) aqua_bw_put(outBits, (uint32_t)linY, linbits);
            if (y != 0) aqua_bw_put(outBits, (y < 0) ? 1u : 0u, 1);
        }
    }

    // 处理小值区（分别用两个四元组表编码，取短者）
    if (partSmall[1] > partSmall[0]) {
        int start = outBits->len;
        // 表 A
        uint32_t code;
        int codeLen;
        for (int i = partSmall[0]; i < partSmall[1]; i += 4) {
            int v = qs576[i], w = qs576[i + 1], x = qs576[i + 2], y = qs576[i + 3];
            encodeQuadruple(v, w, x, y, 0, &code, &codeLen);
            aqua_bw_put(outBits, code >> (8 - codeLen), codeLen);
            if (v != 0) aqua_bw_put(outBits, (v < 0) ? 1u : 0u, 1);
            if (w != 0) aqua_bw_put(outBits, (w < 0) ? 1u : 0u, 1);
            if (x != 0) aqua_bw_put(outBits, (x < 0) ? 1u : 0u, 1);
            if (y != 0) aqua_bw_put(outBits, (y < 0) ? 1u : 0u, 1);
        }
        int lenA = outBits->len - start;
        // 表 B
        AquaBitsWriter temp;
        aqua_bw_init(&temp);
        for (int i = partSmall[0]; i < partSmall[1]; i += 4) {
            int v = qs576[i], w = qs576[i + 1], x = qs576[i + 2], y = qs576[i + 3];
            encodeQuadruple(v, w, x, y, 1, &code, &codeLen);
            aqua_bw_put(&temp, code >> (8 - codeLen), codeLen);
            if (v != 0) aqua_bw_put(&temp, (v < 0) ? 1u : 0u, 1);
            if (w != 0) aqua_bw_put(&temp, (w < 0) ? 1u : 0u, 1);
            if (x != 0) aqua_bw_put(&temp, (x < 0) ? 1u : 0u, 1);
            if (y != 0) aqua_bw_put(&temp, (y < 0) ? 1u : 0u, 1);
        }
        int lenB = temp.len;
        if (lenA <= lenB) {
            smallvalueTableSelect = 0;
            // 保留 outBits 中已写好的表A内容即可
        } else {
            // 回退表A，改写作表B
            outBits->len = start;
            for (int i = 0; i < temp.len; i++) {
                int byte = i >> 3;
                int bit = 7 - (i & 7);
                if ((temp.data[byte] >> bit) & 1u) {
                    aqua_bw_put(outBits, 1u, 1);
                } else {
                    aqua_bw_put(outBits, 0u, 1);
                }
            }
            smallvalueTableSelect = 1;
        }
        aqua_bw_free(&temp);
    }

    if (region0Count < 0) region0Count = 0;
    if (region1Count < 0) region1Count = 0;

    res->bigvalues = Bigvalues;
    res->tableSelect[0] = tableSelect[0];
    res->tableSelect[1] = tableSelect[1];
    res->tableSelect[2] = tableSelect[2];
    res->region0Count = region0Count;
    res->region1Count = region1Count;
    res->smallvalueTableSelect = smallvalueTableSelect;
    res->partitionBig[0] = partBig[0];
    res->partitionBig[1] = partBig[1];
    res->partitionSmall[0] = partSmall[0];
    res->partitionSmall[1] = partSmall[1];
    return 1;
}
