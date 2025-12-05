#ifndef __NANO_INFER_QUANT_H__
#define __NANO_INFER_QUANT_H__

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// 量化相关
#define GS (128)        // 分组量化的组长度
#define QTYPE int8_t    // 量化类型
#define Q_MAX (127.0f)  // 量化类型对应的最大正值（对称量化）

typedef struct {
    QTYPE* q;     // quantized values
    float* s;     // scaling factors
} QuantizedTensor;

void dequantize(QuantizedTensor *qx, float* x, int n);
void quantize(QuantizedTensor *qx, float* x, int n);
QuantizedTensor *init_quantized_tensors(float *w, int n, int size_each);
QuantizedTensor *parse_quantized_tensors(void **ptr, int n, int size_each);

#endif
