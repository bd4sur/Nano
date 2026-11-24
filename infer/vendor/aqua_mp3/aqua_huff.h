/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_huff.h
//
//    哈夫曼编码（编码端）。
//
/////////////////////////////////////////////////////////////////

#ifndef AQUA_HUFF_H
#define AQUA_HUFF_H

#include "aqua_common.h"
#include "aqua_bits.h"

// 哈夫曼编码结果（对应 huffman.js 中 HuffmanEncode 的返回值）
typedef struct {
    int bigvalues;             // 大值对的个数
    int tableSelect[3];        // 各大值区子分区使用的哈夫曼表号
    int region0Count;
    int region1Count;
    int smallvalueTableSelect; // 小值区表号（0/1）
    int partitionBig[2];       // 大值区范围 [起,止)
    int partitionSmall[2];     // 小值区范围 [起,止)
} AquaHuffmanResult;

// 对 576 点量化频谱做哈夫曼编码，写入 out_bits，并填充结果结构。
// blockType 目前编码端恒为 WINDOW_NORMAL（长块）。
// 返回 1 表示成功；0 表示量化值超出 8206（对应 JS 的返回 null）。
int aqua_huff_encode(const int *qs576, int blockType, int sampleRateIdx,
                     AquaBitsWriter *outBits, AquaHuffmanResult *res);

// 从比特流读取一个哈夫曼码字。
// tableNum: 0..31 大值二元组表；32/33 小值四元组表。
// 大值表在函数内部处理 linbits 与符号位。返回 1 成功，0 失败（值为 0）。
int aqua_huff_decode_word(AquaBitsReader *br, int tableNum,
                          int *x, int *y, int *v, int *w);

#endif /* AQUA_HUFF_H */
