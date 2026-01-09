#include "infer.h"

// cc -O3 -march=native -ffast-math -Wall -fopenmp qtest.c platform_linux.c utils.c prompt.c tokenizer.c tensor.c infer.c -o qtest -lm -fopenmp

int main(void) {
    uint64_t seed = 39;

    const uint32_t d = 8;
    const uint32_t n = 768;

    float *w_f32 = (float*)calloc(d*n, sizeof(float));
    float *x_f32 = (float*)calloc(n, sizeof(float));
    float *y_f32 = (float*)calloc(d, sizeof(float));

    for (uint32_t i = 0; i < d*n; i++) w_f32[i] = random_f32(&seed) * ((random_f32(&seed) > 0.5) ? (+1) : (-1));
    for (uint32_t i = 0; i <  n ; i++) x_f32[i] = random_f32(&seed) * ((random_f32(&seed) > 0.5) ? (+1) : (-1));
    for (uint32_t i = 0; i <  d ; i++) y_f32[i] = 0.0f;

    // FP32原值乘
    matmul(y_f32, x_f32, w_f32, n, d);
    printf("原值乘 = ");
    for (uint32_t i = 0; i < d; i++) {
        printf("%.3f  ", y_f32[i]);
    }
    printf("\n");

    // 量化后直接乘
    uint32_t w_shape[2] = {d, n};
    Q4k_Tensor *w = quantize_tensor_q4k(w_f32, 2, w_shape);
    uint32_t x_shape[1] = {n};
    Q4k_Tensor *x = quantize_tensor_q4k(x_f32, 1, x_shape);

    uint8_t *w_stream = pack_q4k_tensor(w);
    uint8_t *x_stream = pack_q4k_tensor(x);

    uint64_t frame_length = 0;
    Q4k_Tensor *w_up = unpack_q4k_tensor(w_stream, &frame_length);
    Q4k_Tensor *x_up = unpack_q4k_tensor(x_stream, &frame_length);

    matmul_q4k(y_f32, x_up, w_up);
    printf("量化乘 = ");
    for (uint32_t i = 0; i < d; i++) {
        printf("%.3f  ", y_f32[i]);
    }
    printf("\n");

    // 量化后再反量化回FP32再乘
    uint32_t ndim = 2;
    uint32_t shape_tmp[2];
    float *ww_f32 = (float*)calloc(d*n, sizeof(float));
    float *xx_f32 = (float*)calloc(n,   sizeof(float));
    dequantize_tensor_q4k(w_up, ww_f32, &ndim, shape_tmp);
    dequantize_tensor_q4k(x_up, xx_f32, &ndim, shape_tmp);
    matmul(y_f32, xx_f32, ww_f32, n, d);
    printf("反量乘 = ");
    for (uint32_t i = 0; i < d; i++) {
        printf("%.3f  ", y_f32[i]);
    }
    printf("\n");

    // free(all);

    return 0;
}