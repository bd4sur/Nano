/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_bits.h
//
//    比特流写入器（编码端）与比特流读取器（解码端）。
//
/////////////////////////////////////////////////////////////////

#ifndef AQUA_BITS_H
#define AQUA_BITS_H

#include "aqua_common.h"

// ---------- 写入器（MSB 优先） ----------

typedef struct {
    uint8_t *data;  // 动态增长的字节缓存
    int      len;   // 已写入比特数
    int      cap;   // 缓存容量（字节）
    int      failed;
} AquaBitsWriter;

void  aqua_bw_init(AquaBitsWriter *bw);      // 初始化（分配初始缓存）
void  aqua_bw_free(AquaBitsWriter *bw);      // 释放缓存
void  aqua_bw_put(AquaBitsWriter *bw, uint32_t value, int nbits); // 写 nbits（MSB 在前）
void  aqua_bw_zero(AquaBitsWriter *bw, int nbits);               // 写 nbits 个 0
int   aqua_bw_len(AquaBitsWriter *bw);       // 返回比特数
int   aqua_bw_nbytes(AquaBitsWriter *bw);    // 返回按字节对齐后的字节数
void  aqua_bw_pad_byte(AquaBitsWriter *bw);  // 补零至字节边界
const uint8_t *aqua_bw_packed(AquaBitsWriter *bw); // 返回按字节打包结果

// ---------- 读取器（MSB 优先，可越界读 0） ----------
typedef struct {
    const uint8_t *data;
    size_t         len;    // 字节数
    size_t         bitpos; // 已读比特数
} AquaBitsReader;

void aqua_br_init(AquaBitsReader *br, const uint8_t *data, size_t len);
uint32_t aqua_br_get(AquaBitsReader *br, int nbits);  // 读 nbits（越界返回 0）
size_t aqua_br_pos(AquaBitsReader *br);
void aqua_br_set_pos(AquaBitsReader *br, size_t bitpos);

#endif /* AQUA_BITS_H */
