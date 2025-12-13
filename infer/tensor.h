#ifndef __NANO_INFER_QUANT_H__
#define __NANO_INFER_QUANT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

// 量化类型
#define QUANT_TYPE_F32  (0)
#define QUANT_TYPE_F16  (1)
#define QUANT_TYPE_BF16 (2)
#define QUANT_TYPE_Q80  (10)

// 量化相关
#define QTYPE int8_t    // 量化类型
#define Q_MAX (127.0f)  // 量化类型对应的最大正值（对称量化）

typedef struct {
    QTYPE* q;     // quantized values
    float* s;     // scaling factors
} Q80_Tensor;

typedef union {
    Q80_Tensor tensor_q80;  // type = QUANT_TYPE_Q80
    float *tensor_f32;      // type = QUANT_TYPE_F32
} Typed_Tensor;

void dequantize(Q80_Tensor *qx, float* x, int n, uint32_t group_size);
void quantize(Q80_Tensor *qx, float* x, int n, uint32_t group_size);
Typed_Tensor *parse_quantized_tensors(void **ptr, int n, int size_each, uint32_t group_size);

#ifdef __cplusplus
}
#endif

#endif
