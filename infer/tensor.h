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
#define QUANT_TYPE_Q4K  (0x42)


/////////////////////////////////////////////////////////////
// Q80量化
/////////////////////////////////////////////////////////////

#define QTYPE int8_t    // 量化类型
#define Q_MAX (127.0f)  // 量化类型对应的最大正值（对称量化）

typedef struct {
    QTYPE* q;     // quantized values
    float* s;     // scaling factors
} Q80_Tensor;

/////////////////////////////////////////////////////////////
// Q4k量化
/////////////////////////////////////////////////////////////

typedef struct {
    uint32_t header;     // 报头（包含量化类型等信息，待定）
    uint32_t length;     // 块内数组的实际长度（不大于256）
    uint32_t meta;       // 保留字段
    float    s_scale;    // 8个缩放因子进行6bit量化的缩放因子
    float    s_bias;     // 8个偏置进行6bit量化的缩放因子
    uint8_t  sb[12];     // 打包存储8个6bit缩放因子+8个6bit偏置
    uint8_t  value[128]; // 打包存储256个4bit权重
} Q4k_Block;

typedef struct {
    uint32_t header;     // 报头（包含量化类型等信息，待定）
    uint32_t ndim;       // 张量维度数
    uint32_t shape[6];   // 张量各维度大小，最多支持6维
    uint32_t num_blocks; // 量化块数
    Q4k_Block **blocks;  // 指向量化块指针数组
} Q4k_Tensor;


/////////////////////////////////////////////////////////////
// 一般量化数据结构
/////////////////////////////////////////////////////////////

typedef union {
    Q4k_Tensor tensor_q4k;  // type = QUANT_TYPE_Q4K
    Q80_Tensor tensor_q80;  // type = QUANT_TYPE_Q80
    float *tensor_f32;      // type = QUANT_TYPE_F32
} Typed_Tensor;


/////////////////////////////////////////////////////////////
// 量化相关函数
/////////////////////////////////////////////////////////////

void dequantize(Q80_Tensor *qx, float* x, int n, uint32_t group_size);
void quantize(Q80_Tensor *qx, float* x, int n, uint32_t group_size);
Typed_Tensor *parse_quantized_tensors(void **ptr, int n, int size_each, uint32_t group_size);


Q4k_Tensor *quantize_tensor_q4k(float *t, uint32_t ndim, uint32_t shape[]);
void dequantize_tensor_q4k(Q4k_Tensor *Q, float *t_out, uint32_t *ndim, uint32_t *shape);
uint8_t *pack_q4k_tensor(Q4k_Tensor *Q);
Q4k_Tensor *unpack_q4k_tensor(uint8_t *buffer, uint64_t *p_total_bytes);


void matmul_q4k(float *xout, Q4k_Tensor *x, Q4k_Tensor *w);

#ifdef __cplusplus
}
#endif

#endif
