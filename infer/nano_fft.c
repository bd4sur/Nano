#include <math.h>

#include "nano_fft.h"

#define NANO_FFT_PI (3.14159265358979f)

void nano_fft_twiddle(float *tw_re, float *tw_im, int32_t n) {
    for (int32_t k = 0; k < n / 2; k++) {
        tw_re[k] = cosf(-2.0f * NANO_FFT_PI * k / n);
        tw_im[k] = sinf(-2.0f * NANO_FFT_PI * k / n);
    }
}

void nano_fft_execute(float *re, float *im, const float *tw_re, const float *tw_im, int32_t n) {
    // 位反转重排
    for (int32_t i = 1, j = 0; i < n; i++) {
        int32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    // 蝶形运算
    for (int32_t len = 2; len <= n; len <<= 1) {
        int32_t half = len >> 1;
        int32_t step = n / len;
        for (int32_t i = 0; i < n; i += len) {
            for (int32_t j = 0, k = 0; j < half; j++, k += step) {
                float wr = tw_re[k];
                float wi = tw_im[k];
                float xr = re[i + j + half] * wr - im[i + j + half] * wi;
                float xi = re[i + j + half] * wi + im[i + j + half] * wr;
                re[i + j + half] = re[i + j] - xr;
                im[i + j + half] = im[i + j] - xi;
                re[i + j] += xr;
                im[i + j] += xi;
            }
        }
    }
}
