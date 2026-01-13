#include "tensor.h"


static inline int nearest_int(float fval) {
    // assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

/////////////////////////////////////////////////////////////
// Q80量化
/////////////////////////////////////////////////////////////

void dequantize(Q80_Tensor *qx, float* x, int n, uint32_t group_size) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / group_size];
    }
}

void quantize(Q80_Tensor *qx, float* x, int n, uint32_t group_size) {
    int num_groups = n / group_size;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < group_size; i++) {
            float val = fabs(x[group * group_size + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < group_size; i++) {
            float quant_value = x[group * group_size + i] / scale; // scale
            QTYPE quantized = (QTYPE) round(quant_value); // round and clamp
            qx->q[group * group_size + i] = quantized;
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
Typed_Tensor *parse_quantized_tensors(void **ptr, int n, int size_each, uint32_t group_size) {
    void *p = *ptr;
    Typed_Tensor *res = malloc(n * sizeof(Typed_Tensor));
    for(int i = 0; i < n; i++) {
        /* map quantized int8 values*/
        res[i].tensor_q80.q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].tensor_q80.s = (float*)p;
        p = (float*)p + size_each / group_size;
    }
    *ptr = p; // advance ptr to current position
    return res;
}



/////////////////////////////////////////////////////////////
// Q4k量化
/////////////////////////////////////////////////////////////


Q4k_Tensor *make_q4k_tensor(uint32_t ndim, uint32_t shape[]) {
    const uint32_t block_len = 256;

    Q4k_Tensor *T = (Q4k_Tensor*)calloc(1, sizeof(Q4k_Tensor));
    T->header = QUANT_TYPE_Q4K; // 代表Q4k量化，其余字段待定
    T->ndim = ndim;

    // 每个末维向量称为一“行”。首先计算末维每行对应的量化块数，这决定了总的块数
    uint32_t line_dim = shape[ndim-1];
    for (uint32_t i = 0; i < ndim; i++) {
        T->shape[i] = shape[i];
    }
    uint32_t n_blocks_per_line = (uint32_t)ceilf((float)(line_dim) / (float)(block_len));

    // 将非末维全部展平，展平得到的维数称为“行数”
    uint32_t n_lines = 1;
    for (uint32_t i = 0; i < ndim-1; i++) {
        n_lines *= shape[i];
    }

    // 总的块数
    uint32_t n_blocks = n_lines * n_blocks_per_line;
    T->num_blocks = n_blocks;

    // 分配块
    T->blocks = (Q4k_Block*)calloc(n_blocks, sizeof(Q4k_Block));
    for (uint32_t i = 0; i < n_blocks; i++) {
        Q4k_Block *qb = T->blocks + i;
        qb->header = QUANT_TYPE_Q4K;
    }
    return T;
}


static inline void get_group_scale_and_bias(const Q4k_Block *Q, float *s_f32, float *b_f32) {
    const uint8_t *sb = Q->sb;

    uint8_t s_i6[8];
    uint8_t b_i6[8];

    s_i6[0] = sb[0] & 0x3f;
    s_i6[1] = sb[1] & 0x3f;
    s_i6[2] = sb[2] & 0x3f;
    s_i6[3] = sb[3] & 0x3f;
    s_i6[4] = (((sb[0] >> 6) << 4) | (sb[8 ] & 0x0f)) & 0x3f;
    s_i6[5] = (((sb[1] >> 6) << 4) | (sb[9 ] & 0x0f)) & 0x3f;
    s_i6[6] = (((sb[2] >> 6) << 4) | (sb[10] & 0x0f)) & 0x3f;
    s_i6[7] = (((sb[3] >> 6) << 4) | (sb[11] & 0x0f)) & 0x3f;

    b_i6[0] = sb[4] & 0x3f;
    b_i6[1] = sb[5] & 0x3f;
    b_i6[2] = sb[6] & 0x3f;
    b_i6[3] = sb[7] & 0x3f;
    b_i6[4] = (((sb[4] >> 6) << 4) | ((sb[8 ] & 0xf0) >> 4)) & 0x3f;
    b_i6[5] = (((sb[5] >> 6) << 4) | ((sb[9 ] & 0xf0) >> 4)) & 0x3f;
    b_i6[6] = (((sb[6] >> 6) << 4) | ((sb[10] & 0xf0) >> 4)) & 0x3f;
    b_i6[7] = (((sb[7] >> 6) << 4) | ((sb[11] & 0xf0) >> 4)) & 0x3f;

    for (int32_t i = 0; i < 8; i++) {
        s_f32[i] = (float)(s_i6[i]) * Q->s_scale;
        b_f32[i] = (float)(b_i6[i]) * Q->s_bias;
    }
}


void quantize_one_block_q4k_in_situ(float *vec, uint32_t d, Q4k_Block *Q) {
    const uint32_t block_len = 256;
    const uint32_t group_len = 32;
    const uint32_t group_num = 8; // 256/32

    Q->header = QUANT_TYPE_Q4K;  // 代表Q4k量化，其余字段待定
    Q->length = d;     // 块内向量实际长度（必不大于256）

    // 统计各组的s和b

    float group_s[group_num];  // 每组的缩放因子
    float group_b[group_num];  // 每组的偏置

    for (uint32_t g = 0; g < group_num; g++) {
        float min_in_group = FLT_MAX;
        float max_in_group = FLT_TRUE_MIN;
        for (uint32_t i = g * group_len; i < (g+1) * group_len; i++) {
            if (i >= d) break; // 不统计有效长度之外的填充部分
            float vv = vec[i];
            if (vv > max_in_group) max_in_group = vv;
            if (vv < min_in_group) min_in_group = vv;
        }
        group_s[g] = (min_in_group <= 0.0f) ?
                     ((max_in_group - min_in_group) / 15.0f) :
                     (max_in_group / 15.0f);
        group_b[g] = (min_in_group <= 0.0f) ? (-min_in_group) : 0.0f; // 偏置非负
    }

    // 对FP32数值 分组进行4bit非对称量化

    uint8_t v[block_len];

    for (uint32_t g = 0; g < group_num; g++) {
        float s = group_s[g];
        float b = group_b[g];
        uint32_t end_index = 0;
        for (uint32_t i = g * group_len; i < (g+1) * group_len; i++) {
            if (i >= d) { // 不处理有效长度之外的部分
                end_index = i;
                break;
            }
            v[i] = (!s) ? 0 : (uint8_t)(nearest_int((vec[i] + b) / s) & 0x0f);
        }
        // 有效长度之外填0
        if (end_index > 0 && end_index < block_len) {
            for (uint32_t i = end_index; i < block_len; i++) {
                v[i] = 0;
            }
            break;
        }
    }

    // 将4bit量化值打包塞进value字段

    for (uint32_t i = 0; i < block_len; i += 2) { // 两个一组处理
        uint8_t v0 = v[ i ]; // 前一个在低4位
        uint8_t v1 = v[i+1]; // 后一个在高4位
        Q->value[i>>1] = ((v0 & 0x0f) | (v1 << 4));
    }

    // 对各组的s和b进行6bit对称量化

    uint8_t group_s_quantized[group_num]; // 量化到6bit的各组的缩放因子
    uint8_t group_b_quantized[group_num]; // 量化到6bit的各组的偏置

    float s_max = FLT_TRUE_MIN;
    float b_max = FLT_TRUE_MIN;
    for (uint32_t g = 0; g < group_num; g++) {
        float s = group_s[g];
        float b = group_b[g];
        if (s > s_max) s_max = s;
        if (b > b_max) b_max = b;
    }

    Q->s_scale = s_max / 63.0f;
    Q->s_bias  = b_max / 63.0f;

    for (uint32_t g = 0; g < group_num; g++) {
        group_s_quantized[g] = (!(Q->s_scale)) ? 0 : (uint8_t)(nearest_int(group_s[g] / Q->s_scale) & 0x3f);
        group_b_quantized[g] = (!(Q->s_bias))  ? 0 : (uint8_t)(nearest_int(group_b[g] / Q->s_bias ) & 0x3f);
    }

    // 将6bit量化的s和b打包进12个字节的Q->sb中

    Q->sb[0] = ((group_s_quantized[4] & 0x30) << 2) | (group_s_quantized[0] & 0x3f);
    Q->sb[1] = ((group_s_quantized[5] & 0x30) << 2) | (group_s_quantized[1] & 0x3f);
    Q->sb[2] = ((group_s_quantized[6] & 0x30) << 2) | (group_s_quantized[2] & 0x3f);
    Q->sb[3] = ((group_s_quantized[7] & 0x30) << 2) | (group_s_quantized[3] & 0x3f);

    Q->sb[4] = ((group_b_quantized[4] & 0x30) << 2) | (group_b_quantized[0] & 0x3f);
    Q->sb[5] = ((group_b_quantized[5] & 0x30) << 2) | (group_b_quantized[1] & 0x3f);
    Q->sb[6] = ((group_b_quantized[6] & 0x30) << 2) | (group_b_quantized[2] & 0x3f);
    Q->sb[7] = ((group_b_quantized[7] & 0x30) << 2) | (group_b_quantized[3] & 0x3f);

    Q->sb[8] = ((group_b_quantized[4] & 0x0f) << 4) | (group_s_quantized[4] & 0x0f);
    Q->sb[9] = ((group_b_quantized[5] & 0x0f) << 4) | (group_s_quantized[5] & 0x0f);
    Q->sb[10]= ((group_b_quantized[6] & 0x0f) << 4) | (group_s_quantized[6] & 0x0f);
    Q->sb[11]= ((group_b_quantized[7] & 0x0f) << 4) | (group_s_quantized[7] & 0x0f);
}

Q4k_Block *quantize_one_block_q4k(float *vec, uint32_t d) {
    Q4k_Block *Q = (Q4k_Block*)calloc(1, sizeof(Q4k_Block));
    quantize_one_block_q4k_in_situ(vec, d, Q);
    return Q;
}


// 反量化一个块，返回值是块的实际长度
uint32_t dequantize_one_block_q4k(Q4k_Block *Q, float *out) {
    // const uint32_t block_len = 256;
    const uint32_t group_len = 32;
    const uint32_t group_num = 8; // 256/32

    uint32_t len = Q->length;

    // 反量化各组的s和b
    float s_f32[group_num];
    float b_f32[group_num];
    get_group_scale_and_bias(Q, s_f32, b_f32);

    // 逐组处理
    for (uint32_t g = 0; g < group_num; g++) {
        float s = s_f32[g];
        float b = b_f32[g];
        int32_t actual_group_len = (len >= ((g+1) * group_len)) ? group_len : (len - group_len * g);
        if (actual_group_len <= 0) break;
        for (int32_t i = 0; i < actual_group_len; i++) {
            uint32_t vindex = group_len * g + i;
            uint8_t v = ((vindex & 1) == 0) ? ((Q->value[vindex>>1]) & 0x0f) : (((Q->value[vindex>>1]) >> 4) & 0x0f);
            out[vindex] = (float)(v) * s - b;
        }
    }
    return len;
}


void quantize_tensor_q4k_in_situ(float *t, uint32_t ndim, uint32_t shape[], Q4k_Tensor *T) {
    const uint32_t block_len = 256;

    assert(T->ndim == ndim);

    // 每个末维向量称为一“行”。首先计算末维每行对应的量化块数，这决定了总的块数
    uint32_t line_dim = shape[ndim-1];
    for (uint32_t i = 0; i < ndim; i++) {
        T->shape[i] = shape[i];
    }
    uint32_t n_blocks_per_line = (uint32_t)ceilf((float)(line_dim) / (float)(block_len));

    // 将非末维全部展平，展平得到的维数称为“行数”
    uint32_t n_lines = 1;
    for (uint32_t i = 0; i < ndim-1; i++) {
        n_lines *= shape[i];
    }

    // 逐行逐块进行量化
    uint32_t i = 0;
    #pragma omp parallel for private(i)
    for (i = 0; i < n_lines; i++) {
        // 每行有若干块
        for (uint32_t j = 0; j < n_blocks_per_line; j++) {
            // 当前块的实际长度
            uint32_t d = (line_dim >= (j+1) * block_len) ? block_len : (line_dim - j * block_len);
            quantize_one_block_q4k_in_situ(t + i * line_dim + j * d, d, T->blocks + i * n_blocks_per_line + j);
        }
    }
}

Q4k_Tensor *quantize_tensor_q4k(float *t, uint32_t ndim, uint32_t shape[]) {
    Q4k_Tensor *T = make_q4k_tensor(ndim, shape);
    quantize_tensor_q4k_in_situ(t, ndim, shape, T);
    return T;
}

void dequantize_tensor_q4k(Q4k_Tensor *Q, float *t_out, uint32_t *ndim, uint32_t *shape) {
    const uint32_t block_len = 256;

    *ndim = Q->ndim;
    for (uint32_t i = 0; i < Q->ndim; i++) {
        shape[i] = Q->shape[i];
    }

    uint32_t line_dim = shape[Q->ndim-1];
    uint32_t n_blocks_per_line = (uint32_t)ceilf((float)(line_dim) / (float)(block_len));
    uint32_t n_lines = 1;
    for (uint32_t i = 0; i < Q->ndim-1; i++) {
        n_lines *= Q->shape[i];
    }

    uint32_t bcount = 0;
    for (uint32_t i = 0; i < n_lines; i++) {
        // 每行有若干块
        for (uint32_t j = 0; j < n_blocks_per_line; j++) {
            // 当前块的实际长度
            uint32_t d = (line_dim >= (j+1) * block_len) ? block_len : (line_dim - j * block_len);
            uint32_t qd = dequantize_one_block_q4k(Q->blocks + bcount, t_out + i * line_dim + j * d);
            assert(d == qd);
            bcount++;
        }
    }
}


// 计算一个Q4k_Tensor打包后的字节数
uint64_t bytes_num_of_q4k_tensor(Q4k_Tensor *Q) {
    const uint64_t bytes_per_block = 160;
    uint64_t bytes = 0;
    bytes += sizeof(uint64_t);     // 帧字节数（含本字段）
    bytes += sizeof(uint32_t);     // header
    bytes += sizeof(uint32_t);     // ndim
    bytes += sizeof(uint32_t) * 6; // shape[6]
    bytes += sizeof(uint32_t);     // num_blocks
    bytes += (Q->num_blocks) * bytes_per_block;
    return bytes;
}

void pack_q4k_block(Q4k_Block *qb, uint8_t *buffer) {
    uint64_t offset = 0;
    memcpy(buffer + offset, &(qb->header), sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(buffer + offset, &(qb->length), sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(buffer + offset, &(qb->meta),   sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(buffer + offset, &(qb->s_scale),sizeof(float));    offset += sizeof(float);
    memcpy(buffer + offset, &(qb->s_bias), sizeof(float));    offset += sizeof(float);
    memcpy(buffer + offset, qb->sb,   12 * sizeof(uint8_t));  offset += sizeof(uint8_t) * 12;
    memcpy(buffer + offset, qb->value,128* sizeof(uint8_t));  offset += sizeof(uint8_t) * 128;
}

uint8_t *pack_q4k_tensor(Q4k_Tensor *Q) {
    const uint64_t bytes_per_block = 160;
    uint64_t total_bytes = bytes_num_of_q4k_tensor(Q);
    uint8_t *buffer = (uint8_t*)calloc(total_bytes, sizeof(uint8_t));
    uint64_t offset = 0;
    memcpy(buffer + offset, &(total_bytes),    sizeof(uint64_t)); offset += sizeof(uint64_t);
    memcpy(buffer + offset, &(Q->header),      sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(buffer + offset, &(Q->ndim),        sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(buffer + offset, Q->shape,      6 * sizeof(uint32_t)); offset += sizeof(uint32_t) * 6;
    memcpy(buffer + offset, &(Q->num_blocks),  sizeof(uint32_t)); offset += sizeof(uint32_t);
    for (uint32_t i = 0; i < Q->num_blocks; i++) {
        pack_q4k_block(Q->blocks + i, buffer + offset);
        offset += bytes_per_block * sizeof(uint8_t);
    }
    return buffer;
}

void unpack_q4k_block(uint8_t *buffer, Q4k_Block *qb) {
    uint64_t offset = 0;
    memcpy(&(qb->header) , buffer + offset, sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(&(qb->length) , buffer + offset, sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(&(qb->meta)   , buffer + offset, sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(&(qb->s_scale), buffer + offset, sizeof(float));    offset += sizeof(float);
    memcpy(&(qb->s_bias) , buffer + offset, sizeof(float));    offset += sizeof(float);
    memcpy(qb->sb        , buffer + offset, 12 * sizeof(uint8_t));  offset += sizeof(uint8_t) * 12;
    memcpy(qb->value     , buffer + offset, 128* sizeof(uint8_t));  offset += sizeof(uint8_t) * 128;
}

Q4k_Tensor *unpack_q4k_tensor(uint8_t *buffer, uint64_t *p_total_bytes) {
    const uint64_t bytes_per_block = 160;
    
    uint32_t header = 0;
    uint32_t ndim = 0;
    uint32_t shape[6];
    uint32_t num_blocks = 0;

    uint64_t offset = 0;
    memcpy(p_total_bytes, buffer + offset, sizeof(uint64_t)); offset += sizeof(uint64_t);
    memcpy(&header,       buffer + offset, sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(&ndim,         buffer + offset, sizeof(uint32_t)); offset += sizeof(uint32_t);
    memcpy(shape,         buffer + offset, 6 * sizeof(uint32_t)); offset += sizeof(uint32_t) * 6;
    memcpy(&num_blocks,   buffer + offset, sizeof(uint32_t)); offset += sizeof(uint32_t);

    Q4k_Tensor *Q = make_q4k_tensor(ndim, shape);

    for (uint32_t i = 0; i < Q->num_blocks; i++) {
        unpack_q4k_block(buffer + offset, Q->blocks + i);
        offset += bytes_per_block * sizeof(uint8_t);
    }
    return Q;
}


// 对两个等长的量化block作点积
static inline float dot_two_blocks_q4k(const Q4k_Block * __restrict__ P, const Q4k_Block * __restrict__ Q) {
    // const uint32_t block_len = 256;
    const uint32_t group_len = 32;
    const uint32_t group_num = 8; // 256/32

    assert(P->length == Q->length);
    uint32_t len = P->length;

    // 反量化P块和Q块的各组的s和b
    float ps_f32[8]; float pb_f32[8];
    float qs_f32[8]; float qb_f32[8];
    get_group_scale_and_bias(P, ps_f32, pb_f32);
    get_group_scale_and_bias(Q, qs_f32, qb_f32);

    // 逐组作点积
    float dot_sum = 0.0f;
    for (uint32_t g = 0; g < group_num; g++) {
        float sp = ps_f32[g];
        float sq = qs_f32[g];
        float bp = pb_f32[g];
        float bq = qb_f32[g];

        int32_t sum_pq = 0;
        int32_t sum_p = 0;
        int32_t sum_q = 0;

        int32_t actual_group_len = (len >= ((g+1) * group_len)) ? group_len : (len - group_len * g);

        if (actual_group_len <= 0) break;

        // 预解包当前 group 的 32 个 nibble 到本地数组
        uint8_t p_nibbles[32];
        uint8_t q_nibbles[32];

        const uint8_t *p_bytes = &P->value[g * 16]; // 16 bytes = 32 nibbles
        const uint8_t *q_bytes = &Q->value[g * 16];

        #pragma GCC unroll 8
        for (int i = 0; i < 16; i++) {
            uint8_t pb = p_bytes[i];
            uint8_t qb = q_bytes[i];
            p_nibbles[i * 2 + 0] = pb & 0x0F;
            p_nibbles[i * 2 + 1] = pb >> 4;
            q_nibbles[i * 2 + 0] = qb & 0x0F;
            q_nibbles[i * 2 + 1] = qb >> 4;
        }

        // 将超出 actual_group_len 的位置清零
        // actual_group_len ∈ [1, 32]，若为 0 则全清零
        if (actual_group_len < 32) {
            // 可展开为 switch 或 memset，但为可读性用循环（编译器会优化）
            for (int i = actual_group_len; i < 32; i++) {
                p_nibbles[i] = 0;
                q_nibbles[i] = 0;
            }
        }

        #pragma GCC unroll 8
        for (int32_t i = 0; i < group_len; i++) {
            uint8_t pv = p_nibbles[i];
            uint8_t qv = q_nibbles[i];
            sum_pq += (int32_t)(pv) * (int32_t)(qv);
            sum_p  += (int32_t)(pv);
            sum_q  += (int32_t)(qv);
        }

        float dot_group_sum = sp * sq * (float)sum_pq
                            - sp * bq * (float)sum_p
                            - sq * bp * (float)sum_q
                            + actual_group_len * bp * bq;

        dot_sum += dot_group_sum;
    }

    return dot_sum;
}

// 情形1：w.ndim == 2：W(d,n) @ x(n) -> xout(d)
// 情形2：w.ndim == 3：W(l,d,n)选取第0维的layer切片进行计算：W[layer](d,n) @ x(n) -> xout(d)
void matmul_q4k(float *xout, Q4k_Tensor *x, Q4k_Tensor *w, uint32_t layer) {
    const uint32_t block_len = 256;

    if (w->ndim != 2 && w->ndim != 3) {
        assert(0);
        return;
    }

    uint32_t wdim = w->ndim;
    uint32_t l = (wdim == 3) ? w->shape[wdim-3] : 0;
    uint32_t d = w->shape[wdim-2];
    uint32_t n = w->shape[wdim-1];

    assert(x->shape[0] == n);
    if (wdim == 3) assert(layer < l);
    if (wdim == 2) layer = 0;

    // uint32_t n_blocks_per_line = (uint32_t)ceilf((float)(n) / (float)(block_len));
    uint32_t n_blocks_per_line = (n + block_len - 1) / block_len;

    uint32_t k = 0;
    // #pragma omp parallel for schedule(static,1)
    #pragma omp parallel for private(k)
    for (k = layer * d; k < (layer+1) * d; k++) {
        uint32_t w_block_index = k * n_blocks_per_line;
        float line_sum = 0.0f;
        for (uint32_t i = 0; i < n_blocks_per_line; i++) {
            Q4k_Block *w_block = w->blocks + w_block_index + i;
            Q4k_Block *x_block = x->blocks + i;
            line_sum += dot_two_blocks_q4k(w_block, x_block);
        }
        xout[k - layer * d] = line_sum;
    }
}

