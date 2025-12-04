#include "quant.h"

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            QTYPE quantized = (QTYPE) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

QuantizedTensor *init_quantized_tensors(float *w, int n, int size_each) {
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i = 0; i < n; i++) {
        QTYPE *qv = (QTYPE *)calloc(size_each, sizeof(QTYPE));
        float *sf = (float *)calloc(size_each / GS, sizeof(float));
        res[i] = (QuantizedTensor){ .q = qv, .s = sf };
        quantize(res + i, w + i * size_each, size_each);
    }
    return res;
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *parse_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}
