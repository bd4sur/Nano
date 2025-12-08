#include "tensor.h"

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
