/////////////////////////////////////////////////////////////////
//
//  Project Aqua - MP3 Audio Encoder / MP3音频编码器
//
//  aqua_mdct.c
//
//    改进的离散余弦变换实现（对齐 src/mdct.js）。
//
/////////////////////////////////////////////////////////////////

#include <string.h>
#include "aqua_mdct.h"

// MDCT 系数表（与 JS 的 MDCT_FACTOR_36/12 相同）
static double MDCT_FACTOR_36[18 * 36];
static double MDCT_FACTOR_12[6 * 12];

void aqua_mdct_init(void) {
    for (int i = 0; i < 18; i++) {
        for (int k = 0; k < 36; k++) {
            MDCT_FACTOR_36[i * 36 + k] =
                cos(AQUA_PI * (2 * k + 1 + 18) * (2 * i + 1) / (2 * 36));
        }
    }
    for (int i = 0; i < 6; i++) {
        for (int k = 0; k < 12; k++) {
            MDCT_FACTOR_12[i * 12 + k] =
                cos(AQUA_PI * (2 * k + 1 + 6) * (2 * i + 1) / (2 * 12));
        }
    }
}

void aqua_mdct(const double *input, int length, double *output) {
    const double *factor = (length == 12) ? MDCT_FACTOR_12 : MDCT_FACTOR_36;
    for (int i = 0; i < (length / 2); i++) {
        double sum = 0.0;
        for (int k = 0; k < length; k++) {
            sum += input[k] * factor[i * length + k];
        }
        output[i] = sum / length * 4.0;
    }
}

void aqua_reduce_aliasing(double *spectrum) {
    double input[576];
    memcpy(input, spectrum, sizeof(input));
    double output[576];
    memcpy(output, input, sizeof(output));
    for (int i = 1; i < 32; i++) {
        int indexA = 18 * i - 1;
        int indexB = 18 * i;
        for (int butterfly = 0; butterfly < 8; butterfly++) {
            double inputA = input[indexA - butterfly];
            double inputB = input[indexB + butterfly];
            output[indexA - butterfly] =  inputA * ALIAS_CS[butterfly] + inputB * ALIAS_CA[butterfly];
            output[indexB + butterfly] = -inputA * ALIAS_CA[butterfly] + inputB * ALIAS_CS[butterfly];
        }
    }
    memcpy(spectrum, output, sizeof(output));
}

void aqua_granule_mdct(const double current[SUBBANDS][SUBBAND_LEN],
                       const double prev[SUBBANDS][SUBBAND_LEN], int windowType,
                       double *outLong, double outShort[3][192]) {
    if (windowType == WINDOW_NORMAL || windowType == WINDOW_START || windowType == WINDOW_STOP) {
        const double *windowFunction =
            (windowType == WINDOW_NORMAL) ? MDCT_WIN_NORMAL :
            (windowType == WINDOW_START)  ? MDCT_WIN_START :
                                            MDCT_WIN_STOP;
        double LongBlockSpectrum[576];
        for (int sbindex = 0; sbindex < 32; sbindex++) {
            double windowedMdctInput[36];
            for (int i = 0; i < 36; i++) {
                windowedMdctInput[i] = (i < 18)
                    ? (prev[sbindex][i] * windowFunction[i])
                    : (current[sbindex][i - 18] * windowFunction[i]);
            }
            double *dst = &LongBlockSpectrum[sbindex * 18];
            aqua_mdct(windowedMdctInput, 36, dst);
        }
        aqua_reduce_aliasing(LongBlockSpectrum);
        memcpy(outLong, LongBlockSpectrum, sizeof(LongBlockSpectrum));
    } else {
        // 短块：3 个 12 点短窗口
        double spectrums[3][192];
        memset(spectrums, 0, sizeof(spectrums));
        for (int sbindex = 0; sbindex < 32; sbindex++) {
            double frame[36];
            for (int i = 0; i < 18; i++) {
                frame[i] = prev[sbindex][i];
            }
            for (int i = 0; i < 18; i++) {
                frame[18 + i] = current[sbindex][i];
            }
            for (int shortBlockCount = 0; shortBlockCount < 3; shortBlockCount++) {
                double shortWindowedMdctInput[12];
                for (int i = 0; i < 12; i++) {
                    shortWindowedMdctInput[i] =
                        frame[(shortBlockCount + 1) * 6 + i] * MDCT_WIN_SHORT[i];
                }
                double mdctOut[6];
                aqua_mdct(shortWindowedMdctInput, 12, mdctOut);
                for (int i = 0; i < 6; i++) {
                    spectrums[shortBlockCount][sbindex * 6 + i] = mdctOut[i];
                }
            }
        }
        for (int w = 0; w < 3; w++) {
            memcpy(outShort[w], spectrums[w], sizeof(spectrums[w]));
        }
    }
}
