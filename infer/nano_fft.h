#ifndef __NANO_FFT_H__
#define __NANO_FFT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

// 基2迭代复数FFT（DIT）公共模块：供频谱仪、计步器等需要频域分析的功能复用。
// 用法：先用 nano_fft_twiddle 构建旋转因子表（可长期保留），
//       再反复调用 nano_fft_execute 进行变换。

// 构建旋转因子全表：tw[k] = e^{-j2πk/n}, k = 0..n/2-1
//   tw_re / tw_im：输出表，长度 n/2
void nano_fft_twiddle(float *tw_re, float *tw_im, int32_t n);

// 基2迭代复数FFT（输入位反转重排；twiddle按 k*step 从全表取用）
//   re / im ：输入输出数组（长度 n，n 为 2 的幂）
//   tw_re / tw_im：nano_fft_twiddle 构建的表（长度 n/2）
void nano_fft_execute(float *re, float *im, const float *tw_re, const float *tw_im, int32_t n);

#ifdef __cplusplus
}
#endif

#endif
