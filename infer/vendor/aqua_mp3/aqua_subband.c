/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_subband.c
//
//    分析子带滤波器实现（对齐 src/subband_filter.js）。
//
/////////////////////////////////////////////////////////////////

#include <math.h>
#include "aqua_subband.h"

// 512 点加窗 + 部分求和 + 矩阵运算（M_Mat）
static void basicAnalysisSubbandFilter(const double *inputBuffer, double s[32]) {
    double z[512];
    for (int i = 0; i < 512; i++) {
        z[i] = inputBuffer[i] * ANALYSIS_WINDOW[i];
    }
    double y[64];
    for (int i = 0; i < 64; i++) {
        double sum = 0.0;
        for (int j = 0; j < 8; j++) {
            sum += z[i + 64 * j];
        }
        y[i] = sum;
    }
    for (int i = 0; i < 32; i++) {
        double sum = 0.0;
        for (int k = 0; k < 64; k++) {
            sum += SUB_ANALYSIS_M[i][k] * y[k];
        }
        s[i] = sum;
    }
}

void aqua_analysis_subband_filter(const double *PCMData, long len, long offset,
                                  double subbands[SUBBANDS][SUBBAND_LEN]) {
    for (int step = 1; step <= 18; step++) {
        long o = offset + 32 * step;
        double inputBuffer[512];
        // 倒序读取 512 个采样，越界处以 0 填充
        for (int i = 0; i < 512; i++) {
            long dataIndex = o - i;
            inputBuffer[i] =
                (dataIndex >= 0 && dataIndex < len) ? PCMData[dataIndex] : 0.0;
        }
        double s[32];
        basicAnalysisSubbandFilter(inputBuffer, s);
        for (int band = 0; band < 32; band++) {
            double v = s[band];
            // 子带滤波器的输入缓冲区反向，需对输出做纠（与 JS 相同：奇数频带、奇数时刻取反）
            // JS 中 t = step - 1，条件为 (band%2==1) && (t%2==1)
            if ((band % 2 == 1) && ((step - 1) % 2 == 1)) {
                v *= -1.0;
            }
            subbands[band][step - 1] = v;
        }
    }
}
