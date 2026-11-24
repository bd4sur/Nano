/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_bits.c
//
//    比特流读写实现。
//
/////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <string.h>
#include "aqua_bits.h"

// ------------------------------------------------------------------
//  写 入 器
// ------------------------------------------------------------------

void aqua_bw_init(AquaBitsWriter *bw) {
    bw->cap = 1024;
    bw->data = (uint8_t *)malloc(bw->cap);
    bw->len = 0;
    bw->failed = (bw->data == NULL) ? 1 : 0;
}

void aqua_bw_free(AquaBitsWriter *bw) {
    if (bw->data != NULL) {
        free(bw->data);
    }
    bw->data = NULL;
    bw->len = 0;
    bw->cap = 0;
}

static void aqua_bw_reserve(AquaBitsWriter *bw, int nbytes) {
    if (bw->failed) {
        return;
    }
    if (bw->len / 8 + nbytes <= bw->cap) {
        return;
    }
    int ncap = bw->cap * 2;
    while (ncap < bw->len / 8 + nbytes) {
        ncap *= 2;
    }
    uint8_t *nd = (uint8_t *)realloc(bw->data, (size_t)ncap);
    if (nd == NULL) {
        bw->failed = 1;
        return;
    }
    bw->data = nd;
    bw->cap = ncap;
}

void aqua_bw_put(AquaBitsWriter *bw, uint32_t value, int nbits) {
    if (nbits <= 0 || bw->failed) {
        return;
    }
    aqua_bw_reserve(bw, nbits / 8 + 2);
    for (int i = nbits - 1; i >= 0; i--) {
        int byte = bw->len >> 3;
        int bit  = 7 - (bw->len & 7);
        if ((bw->len & 7) == 0 && byte < bw->cap) {
            bw->data[byte] = 0;
        }
        // 同时设置/清除当前位：先写 1 后写 0 的重写（如哈夫曼小值区 A/B 表回退）
        // 也必须正确清零，否则残留旧位会导致码流损坏。
        if ((value >> i) & 1u) {
            bw->data[byte] |= (uint8_t)(1u << bit);
        } else {
            bw->data[byte] &= (uint8_t)~(1u << bit);
        }
        bw->len++;
    }
}

void aqua_bw_zero(AquaBitsWriter *bw, int nbits) {
    for (int i = 0; i < nbits; i++) {
        aqua_bw_put(bw, 0, 1);
    }
}

int aqua_bw_len(AquaBitsWriter *bw) {
    return bw->len;
}

int aqua_bw_nbytes(AquaBitsWriter *bw) {
    return (bw->len + 7) >> 3;
}

void aqua_bw_pad_byte(AquaBitsWriter *bw) {
    aqua_bw_zero(bw, (8 - (bw->len & 7)) & 7);
}

const uint8_t *aqua_bw_packed(AquaBitsWriter *bw) {
    return bw->data;
}

// ------------------------------------------------------------------
//  读 取 器
// ------------------------------------------------------------------

void aqua_br_init(AquaBitsReader *br, const uint8_t *data, size_t len) {
    br->data = data;
    br->len = len;
    br->bitpos = 0;
}

uint32_t aqua_br_get(AquaBitsReader *br, int nbits) {
    uint32_t value = 0;
    for (int i = 0; i < nbits; i++) {
        value <<= 1;
        size_t byte = br->bitpos >> 3;
        int bit = 7 - (int)(br->bitpos & 7);
        if (byte < br->len) {
            value |= (uint32_t)((br->data[byte] >> bit) & 1u);
        }
        br->bitpos++;
    }
    return value;
}

size_t aqua_br_pos(AquaBitsReader *br) {
    return br->bitpos;
}

void aqua_br_set_pos(AquaBitsReader *br, size_t bitpos) {
    br->bitpos = bitpos;
}
