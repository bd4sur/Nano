// ============================================================================
// OFDM Modem Core（C 语言移植）/ BD4SUR 2026-07
// 移植自 ofdm/ofdm-modem-worklet.js，物理层/帧协议参数与 JS 版完全一致。
// 设计目标：自包含、可复用、硬件无关（纯 C + libm，浮点采样进/出）。
// 实时性：接收侧定容环形缓冲（满时覆盖最旧采样），O(1) 写入/丢弃，稳态零堆分配。
// ============================================================================

#include "ofdm_modem.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

// ---------------- 私有常量 ----------------

#define BANDWIDTH           (OFDM_BASE_FREQ * OFDM_CARRIER_NUMBER) // 3000 Hz
#define MIX_FREQ            (OFDM_CARRIER_FREQ + OFDM_BASE_FREQ / 2.0) // 2023.4375 Hz
#define TWO_PI              (6.283185307179586)
#define MIX_STEP            (TWO_PI * MIX_FREQ / OFDM_SAMPLE_RATE)  // 混频相位步进(rad/采样)
#define T_SLOT              ((double)OFDM_GROSS_SYMBOL_LENGTH / OFDM_SAMPLE_RATE) // 槽长(秒)

#define DDS_LUT_LEN         (1024)
#define PHASE_TO_IDX        (DDS_LUT_LEN / TWO_PI)

#define LPF_TAPS            (97)
#define LPF_GD              ((LPF_TAPS - 1) / 2) // 48 = 6×DECIM

#define IQ_AMP              (0.02f)
#define A1                  (2.0f * IQ_AMP)

#define PILOT_SPACING       (8)
#define PILOT_SHIFT         (3)

#define SC_HALF_LEN         (OFDM_GROSS_SYMBOL_LENGTH / 2) // 720
#define SC_HALF_BB          (OFDM_SLOT_BB / 2)             // 90
#define SC_DETECT_THRESHOLD (0.2f)
// SC 模板验证阈值：强多径信道中接收前导是信道冲激响应与干净模板的卷积，
// 归一化模板相关会被压到 0.2 量级，故此处仅作弱旁证（排除纯噪声候选）；
// 真正的帧确认迁移到训练A/B信道一致性检查（LTI 信道 H1≈H2，噪声不成立）。
#define SC_VALIDATE_THRESHOLD (0.15f)
#define FINE_SEARCH_LEN     (16)

#define SYM_BLK_LEN         (OFDM_SYMBOL_LENGTH + 2 * LPF_GD) // 1120

#define DET_RING_CAP        (16384)  // next_pow2(4×1440+4096)
#define BUF_RING_CAP        (65536)  // next_pow2(31680+1440)

#define TRAINING_LOCK_THRESHOLD (0.5f)
#define STALL_WATCHDOG          (500)

// SFO 斜率项门控：跨帧 EMA 平滑（α=0.25）+ 高门限(500ppm)连续2帧确认的迟滞启用，
// 仅病态大频偏才启用显式斜率补偿；≤300ppm 时被动跟踪（细同步+信道重估+导频）更稳健
#define SFO_EMA_ALPHA         (0.25f)
// 显式 SFO 补偿启用门限：训练细同步的整数采样量化残差（±0.3 采样 ≈ ±200ppm 拟合偏置）
// 会污染 SFO 拟合，而被动跟踪（散布导频+帧中训练）在 ±1000ppm 内已足够稳健；
// 门限取值应显著大于拟合偏置，钳制值不应小于门限（否则大 SFO 时施加错误的钳制值反而有害）。
#define SFO_ENABLE_PPM        (1000.0f)
#define SFO_ENABLE_STREAK     (2)
#define SFO_CLAMP_PPM         (1500.0f)

#define RS_NROOTS (16)
#define RS_K      (16)
#define RS_N      (32)
#define RS_FCR    (1)

#define PILOT_TRACK_ALPHA   (0.25f)

// packet 头 "BD4SUR"
static const uint8_t PKT_MAGIC[6] = {0x42, 0x44, 0x34, 0x53, 0x55, 0x52};

// QAM4 映射表（M）与解映射表（L），与 JS 一致
static const float QAM4_M[4][2] = {{A1, A1}, {A1, -A1}, {-A1, A1}, {-A1, -A1}};
static const uint8_t QAM4_L[2][2] = {{3, 2}, {1, 0}};

// 帧内槽调度（SC前导之后）：T,T, D×5,T, D×5,T, D×5,T, D（长度21）
#define FRAME_SCHEDULE_LEN  (OFDM_FRAME_SLOTS - 1) // 21
static char FRAME_SCHEDULE[FRAME_SCHEDULE_LEN];

// ---------------- 共享码表 ----------------
// 码表合计约 24KB，运行时生成（算法与 JS 版完全一致），内存由调用方在
// ofdm_tables_init 时注入（ESP32 侧为 PSRAM），ofdm_tables_free 释放。
// 弃用说明：早期版本使用离线生成的 const 表（ofdm_modem_tables.h，驻留 Flash），
// 已改为运行时初始化以支持参数化与按需释放。

static int32_t s_tables_ready = 0;
static void *s_tables_block = NULL;

float *DDS_COS = NULL, *DDS_SIN = NULL;
float *LPF = NULL;
float *CARRIER_FREQS_F = NULL; // 各子载波频率(Hz)，CFO/SFO 拟合用

float *TRAINING_I = NULL, *TRAINING_Q = NULL;
float *PILOT_I = NULL, *PILOT_Q = NULL;
float *TRAINING_BB_RE = NULL, *TRAINING_BB_IM = NULL;
float *TRAINING_SYMBOL_TIME = NULL;      // 训练符号通带时域模板
float TRAINING_TPL_ENERGY = 0.0f;
float *SC_BB_RE = NULL, *SC_BB_IM = NULL;
float *SC_PREAMBLE = NULL;               // SC 前导通带模板
float SC_ENERGY = 0.0f;

uint8_t *GF_EXP = NULL, *GF_LOG = NULL;
uint8_t *RS_GEN = NULL;

// 码表生成用暂存区（位于码表块内，避免占用 core0 任务栈——栈仅 12KB）
static float *s_gen_pb = NULL;    // 2×OFDM_GROSS_SYMBOL_LENGTH
static float *s_gen_bb_re = NULL; // 2×OFDM_SLOT_BB
static float *s_gen_bb_im = NULL; // 2×OFDM_SLOT_BB

// ============================================================================
// 信号处理基础
// ============================================================================

// 基2复数 FFT（原地；正变换 e^{-j2πkn/N}，逆变换 e^{+j} 并乘 1/N）
// 蝶形旋转因子递推生成（与 JS 同构）
static void fft_radix2(float *re, float *im, int32_t inverse) {
    int32_t n = OFDM_BB_FFT_LEN;
    for (int32_t i = 1, j = 0; i < n; i++) {
        int32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int32_t len = 2; len <= n; len <<= 1) {
        float ang = (float)(2.0 * M_PI / len * (inverse ? 1 : -1));
        float wr = cosf(ang), wi = sinf(ang);
        for (int32_t i = 0; i < n; i += len) {
            float cwr = 1.0f, cwi = 0.0f;
            for (int32_t j = 0; j < len / 2; j++) {
                float ur = re[i + j], ui = im[i + j];
                float vr = re[i + j + len / 2] * cwr - im[i + j + len / 2] * cwi;
                float vi = re[i + j + len / 2] * cwi + im[i + j + len / 2] * cwr;
                re[i + j] = ur + vr; im[i + j] = ui + vi;
                re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi;
                float nwr = cwr * wr - cwi * wi; cwi = cwr * wi + cwi * wr; cwr = nwr;
            }
        }
    }
    if (inverse) {
        for (int32_t i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
}

// 逐槽升余弦窗（rolloff=0.01，槽长 1440 → A=14, B=1426）
static void raised_cosine_window(float *wave, int32_t len, float rolloff) {
    int32_t A = (int32_t)(len * rolloff + 0.5f);
    int32_t B = (int32_t)(len * (1.0f - rolloff) + 0.5f);
    for (int32_t i = 0; i < A; i++) wave[i] *= (0.5f - 0.5f * cosf((float)M_PI * i / A));
    for (int32_t i = B; i < len; i++) wave[i] *= (0.5f + 0.5f * cosf((float)M_PI * (i - B) / A));
}

// 子载波 c → 基带 bin（整数栅格：c − N_c/2，DC 不使用）
static int32_t bb_bin(int32_t c) {
    return (c - (OFDM_CARRIER_NUMBER >> 1) + OFDM_BB_FFT_LEN) & (OFDM_BB_FFT_LEN - 1);
}

// 一个基带复符号：载波 IQ 填入基带频域 → 复 IFFT → re/im[BB_FFT_LEN]
static void baseband_symbol(const float *si, const float *sq, float *re, float *im) {
    memset(re, 0, OFDM_BB_FFT_LEN * sizeof(float));
    memset(im, 0, OFDM_BB_FFT_LEN * sizeof(float));
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        int32_t k = bb_bin(c);
        re[k] = si[c]; im[k] = sq[c];
    }
    fft_radix2(re, im, 1);
}

// 基带槽流（复） → 通带实信号：×DECIM 多相内插 → IQ混频取实部 → 逐槽升余弦窗
// 混频相位逐槽复位（各槽载波相位对齐槽起点，训练模板才能逐槽复用）。
// 内插不物化 tmp 数组，直接按输出采样点计算（与 JS 的 tmp[t+LPF_GD] 数学等价）：
//   y[g] = DECIM × Σ_k h[r + k·DECIM]·bb[(g−r)/DECIM − k]，g = 输出采样 + LPF_GD
static void synth_passband(const float *bb_re, const float *bb_im, int32_t n_bb, float *wave) {
    int32_t n_out = n_bb * OFDM_DECIM;
    for (int32_t s0 = 0; s0 < n_out; s0 += OFDM_GROSS_SYMBOL_LENGTH) {
        float phase = 0.0f; // 槽内局部相位（逐槽复位）
        for (int32_t t = 0; t < OFDM_GROSS_SYMBOL_LENGTH; t++) {
            int32_t g = s0 + t + LPF_GD;
            int32_t r = g % OFDM_DECIM;
            int32_t n0 = g / OFDM_DECIM;
            float ar = 0.0f, ai = 0.0f;
            for (int32_t k = 0; k * OFDM_DECIM + r < LPF_TAPS; k++) {
                int32_t idx = n0 - k;
                if (idx >= 0 && idx < n_bb) {
                    float w = LPF[r + k * OFDM_DECIM];
                    ar += w * bb_re[idx];
                    ai += w * bb_im[idx];
                }
            }
            ar *= OFDM_DECIM; ai *= OFDM_DECIM;
            int32_t k_idx = (int32_t)(phase * PHASE_TO_IDX) & (DDS_LUT_LEN - 1);
            wave[s0 + t] = ar * DDS_COS[k_idx] - ai * DDS_SIN[k_idx];
            phase += (float)MIX_STEP;
            if (phase >= (float)TWO_PI) phase -= (float)TWO_PI;
        }
        raised_cosine_window(wave + s0, OFDM_GROSS_SYMBOL_LENGTH, 0.01f);
    }
}

// JS 的 LCG 伪随机（位精确模拟 JS Number 语义：double 运算后 & 0x7fffffff）
// 训练/导频/SC 伪随机图案必须与 JS 版逐比特一致，保证空口互通。
static double lcg_next(uint32_t *seed) {
    double t = (double)(*seed) * 1103515245.0;
    t = t + 12345.0;
    double m = fmod(t, 2147483648.0); // 正数 & 0x7fffffff 等价于 mod 2^31
    *seed = (uint32_t)m;
    return m / 2147483647.0;
}

// 填充帧内槽调度（幂等）
static void ofdm_fill_schedule(void) {
    static int32_t ready = 0;
    if (ready) return;
    int32_t p = 0;
    FRAME_SCHEDULE[p++] = 'T'; FRAME_SCHEDULE[p++] = 'T';
    for (int32_t i = 0; i < OFDM_FRAME_DATA_SYMBOLS; i++) {
        FRAME_SCHEDULE[p++] = 'D';
        if ((i + 1) % 5 == 0 && i + 1 < OFDM_FRAME_DATA_SYMBOLS) FRAME_SCHEDULE[p++] = 'T';
    }
    ready = 1;
}

// ---------------- 码表生成与生命周期管理 ----------------

// 计算全部码表并写入指针（调用前须已通过 ofdm_tables_init 分配内存）
static void ofdm_tables_generate(void) {
    // 帧内槽调度
    ofdm_fill_schedule();

    // DDS 正弦/余弦 LUT
    for (int32_t n = 0; n < DDS_LUT_LEN; n++) {
        DDS_COS[n] = (float)cos(TWO_PI * n / DDS_LUT_LEN);
        DDS_SIN[n] = (float)sin(TWO_PI * n / DDS_LUT_LEN);
    }

    // 抽取/内插抗混叠低通（窗函数法FIR，Hamming，直流增益归一化）
    {
        double fc = (BANDWIDTH / 2.0 + (OFDM_SAMPLE_RATE / OFDM_DECIM) / 2.0) / 2.0 / OFDM_SAMPLE_RATE;
        double sum = 0.0;
        for (int32_t n = 0; n < LPF_TAPS; n++) {
            int32_t x = n - LPF_GD;
            double s = (x == 0) ? 2.0 * fc : sin(TWO_PI * fc * x) / (M_PI * x);
            double w = 0.54 - 0.46 * cos(TWO_PI * n / (LPF_TAPS - 1));
            LPF[n] = (float)(s * w);
            sum += LPF[n];
        }
        for (int32_t n = 0; n < LPF_TAPS; n++) LPF[n] /= (float)sum;
    }

    // 各子载波频率
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        CARRIER_FREQS_F[c] = (float)(OFDM_CARRIER_FREQ + (c - (OFDM_CARRIER_NUMBER - 1) / 2.0) * OFDM_BASE_FREQ);
    }

    // 训练符号频域图案（伪随机 QPSK，seed=54321）
    {
        uint32_t seed = 54321;
        for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
            TRAINING_I[c] = (lcg_next(&seed) < 0.5) ? -A1 : A1;
            TRAINING_Q[c] = (lcg_next(&seed) < 0.5) ? -A1 : A1;
        }
    }

    // 散布导频取值（伪随机 QPSK，seed=24680）
    {
        uint32_t seed = 24680;
        for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
            PILOT_I[c] = (lcg_next(&seed) < 0.5) ? -A1 : A1;
            PILOT_Q[c] = (lcg_next(&seed) < 0.5) ? -A1 : A1;
        }
    }

    // 训练符号：基带符号 + 时域模板（经完整发射链生成，取连续两槽中的第二槽以避开内插边缘瞬态）
    {
        baseband_symbol(TRAINING_I, TRAINING_Q, TRAINING_BB_RE, TRAINING_BB_IM);
        float *bb_re = s_gen_bb_re, *bb_im = s_gen_bb_im;
        float *pb = s_gen_pb;
        for (int32_t rep = 0; rep < 2; rep++) {
            for (int32_t i = OFDM_BB_FFT_LEN - OFDM_CP_BB; i < OFDM_BB_FFT_LEN; i++) {
                bb_re[rep * OFDM_SLOT_BB + (i - (OFDM_BB_FFT_LEN - OFDM_CP_BB))] = TRAINING_BB_RE[i];
                bb_im[rep * OFDM_SLOT_BB + (i - (OFDM_BB_FFT_LEN - OFDM_CP_BB))] = TRAINING_BB_IM[i];
            }
            for (int32_t i = 0; i < OFDM_BB_FFT_LEN; i++) {
                bb_re[rep * OFDM_SLOT_BB + OFDM_CP_BB + i] = TRAINING_BB_RE[i];
                bb_im[rep * OFDM_SLOT_BB + OFDM_CP_BB + i] = TRAINING_BB_IM[i];
            }
        }
        synth_passband(bb_re, bb_im, 2 * OFDM_SLOT_BB, pb);
        memcpy(TRAINING_SYMBOL_TIME, pb + OFDM_GROSS_SYMBOL_LENGTH, OFDM_GROSS_SYMBOL_LENGTH * sizeof(float));
        TRAINING_TPL_ENERGY = 0.0f;
        for (int32_t t = 0; t < OFDM_GROSS_SYMBOL_LENGTH; t++)
            TRAINING_TPL_ENERGY += TRAINING_SYMBOL_TIME[t] * TRAINING_SYMBOL_TIME[t];
    }

    // SC 前导（seed=12345）：基带 [B, B·e^{jδ}]，δ 补偿混频器半槽相位步进；通带模板峰值归一化到 0.9
    {
        uint32_t seed = 12345;
        float sc_i[OFDM_CARRIER_NUMBER], sc_q[OFDM_CARRIER_NUMBER];
        float sym_re[OFDM_BB_FFT_LEN], sym_im[OFDM_BB_FFT_LEN];
        for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
            sc_i[c] = (lcg_next(&seed) < 0.5 ? -1.0f : 1.0f) * A1;
            sc_q[c] = (lcg_next(&seed) < 0.5 ? -1.0f : 1.0f) * A1;
        }
        baseband_symbol(sc_i, sc_q, sym_re, sym_im);
        double dphi = -MIX_STEP * SC_HALF_LEN;
        float cr = (float)cos(dphi), ci = (float)sin(dphi);
        for (int32_t m = 0; m < SC_HALF_BB; m++) {
            SC_BB_RE[m] = sym_re[m];
            SC_BB_IM[m] = sym_im[m];
            SC_BB_RE[SC_HALF_BB + m] = sym_re[m] * cr - sym_im[m] * ci;
            SC_BB_IM[SC_HALF_BB + m] = sym_re[m] * ci + sym_im[m] * cr;
        }
        synth_passband(SC_BB_RE, SC_BB_IM, OFDM_SLOT_BB, SC_PREAMBLE);
        float peak = 0.0f;
        for (int32_t t = 0; t < OFDM_GROSS_SYMBOL_LENGTH; t++) {
            float a = fabsf(SC_PREAMBLE[t]);
            if (a > peak) peak = a;
        }
        SC_ENERGY = 0.0f;
        for (int32_t t = 0; t < OFDM_GROSS_SYMBOL_LENGTH; t++) {
            SC_PREAMBLE[t] *= (0.9f / peak);
            SC_ENERGY += SC_PREAMBLE[t] * SC_PREAMBLE[t];
        }
    }

    // GF(2^8)（本原多项式 0x11D）与 RS 生成多项式
    {
        uint32_t x = 1;
        for (int32_t i = 0; i < 255; i++) {
            GF_EXP[i] = (uint8_t)x;
            GF_LOG[x] = (uint8_t)i;
            x <<= 1;
            if (x & 0x100) x ^= 0x11D;
        }
        for (int32_t i = 255; i < 512; i++) GF_EXP[i] = GF_EXP[i - 255];

        uint8_t g[RS_NROOTS + 1] = {0};
        uint8_t ng[RS_NROOTS + 2];
        int32_t g_len = 1;
        g[0] = 1;
        for (int32_t i = 0; i < RS_NROOTS; i++) {
            uint8_t root = GF_EXP[RS_FCR + i];
            memset(ng, 0, sizeof(ng));
            for (int32_t j = 0; j < g_len; j++) {
                ng[j] ^= g[j];
                ng[j + 1] ^= (g[j] == 0 || root == 0) ? 0 : GF_EXP[GF_LOG[g[j]] + GF_LOG[root]];
            }
            g_len++;
            memcpy(g, ng, g_len);
        }
        memcpy(RS_GEN, g, RS_NROOTS + 1);
    }
}

// 初始化并生成全部码表（幂等）：内存由调用方注入（如 PSRAM），退出时 ofdm_tables_free 释放
int32_t ofdm_tables_init(ofdm_alloc_fn alloc) {
    if (s_tables_ready) return 0;
    if (!alloc) return -1;

    // 单块分配并切分
    size_t sz = 2 * DDS_LUT_LEN * sizeof(float)
              + LPF_TAPS * sizeof(float)
              + OFDM_CARRIER_NUMBER * sizeof(float) * 6        // CARRIER_FREQS/TRAINING_I/Q/PILOT_I/Q
              + 2 * OFDM_BB_FFT_LEN * sizeof(float)            // TRAINING_BB_RE/IM
              + 2 * OFDM_GROSS_SYMBOL_LENGTH * sizeof(float)   // TRAINING_SYMBOL_TIME/SC_PREAMBLE
              + 2 * OFDM_SLOT_BB * sizeof(float)               // SC_BB_RE/IM
              + (2 * OFDM_GROSS_SYMBOL_LENGTH + 4 * OFDM_SLOT_BB) * sizeof(float) // 生成暂存区（pb 2880 + bb_re/bb_im 各 360 float）
              + 512 + 256 + (RS_NROOTS + 1);                   // GF_EXP/GF_LOG/RS_GEN
    s_tables_block = alloc(sz);
    if (!s_tables_block) return -2;

    uint8_t *p = (uint8_t *)s_tables_block;
    #define TBL_TAKE(ptr, nfloat) do { ptr = (float *)p; p += (nfloat) * sizeof(float); } while (0)
    TBL_TAKE(DDS_COS, DDS_LUT_LEN);
    TBL_TAKE(DDS_SIN, DDS_LUT_LEN);
    TBL_TAKE(LPF, LPF_TAPS);
    TBL_TAKE(CARRIER_FREQS_F, OFDM_CARRIER_NUMBER);
    TBL_TAKE(TRAINING_I, OFDM_CARRIER_NUMBER);
    TBL_TAKE(TRAINING_Q, OFDM_CARRIER_NUMBER);
    TBL_TAKE(PILOT_I, OFDM_CARRIER_NUMBER);
    TBL_TAKE(PILOT_Q, OFDM_CARRIER_NUMBER);
    TBL_TAKE(TRAINING_BB_RE, OFDM_BB_FFT_LEN);
    TBL_TAKE(TRAINING_BB_IM, OFDM_BB_FFT_LEN);
    TBL_TAKE(TRAINING_SYMBOL_TIME, OFDM_GROSS_SYMBOL_LENGTH);
    TBL_TAKE(SC_PREAMBLE, OFDM_GROSS_SYMBOL_LENGTH);
    TBL_TAKE(SC_BB_RE, OFDM_SLOT_BB);
    TBL_TAKE(SC_BB_IM, OFDM_SLOT_BB);
    TBL_TAKE(s_gen_pb, 2 * OFDM_GROSS_SYMBOL_LENGTH);
    TBL_TAKE(s_gen_bb_re, 2 * OFDM_SLOT_BB);
    TBL_TAKE(s_gen_bb_im, 2 * OFDM_SLOT_BB);
    #undef TBL_TAKE
    GF_EXP = p; p += 512;
    GF_LOG = p; p += 256;
    RS_GEN = p;

    ofdm_tables_generate();
    s_tables_ready = 1;
    return 0;
}

// 释放码表内存（幂等）
void ofdm_tables_free(ofdm_free_fn dealloc) {
    if (s_tables_block && dealloc) dealloc(s_tables_block);
    s_tables_block = NULL;
    s_tables_ready = 0;
    DDS_COS = NULL; DDS_SIN = NULL; LPF = NULL; CARRIER_FREQS_F = NULL;
    TRAINING_I = NULL; TRAINING_Q = NULL; PILOT_I = NULL; PILOT_Q = NULL;
    TRAINING_BB_RE = NULL; TRAINING_BB_IM = NULL;
    TRAINING_SYMBOL_TIME = NULL; TRAINING_TPL_ENERGY = 0.0f;
    SC_BB_RE = NULL; SC_BB_IM = NULL; SC_PREAMBLE = NULL; SC_ENERGY = 0.0f;
    GF_EXP = NULL; GF_LOG = NULL; RS_GEN = NULL;
    s_gen_pb = NULL; s_gen_bb_re = NULL; s_gen_bb_im = NULL;
}

// ============================================================================
// GF(2^8) 与 RS(32,16)
// ============================================================================

static inline uint8_t gf_mul(uint8_t a, uint8_t b) {
    return (a == 0 || b == 0) ? 0 : GF_EXP[GF_LOG[a] + GF_LOG[b]];
}
static inline uint8_t gf_div(uint8_t a, uint8_t b) {
    return GF_EXP[(GF_LOG[a] - GF_LOG[b] + 255) % 255];
}
static inline uint8_t gf_pow(uint8_t a, int32_t n) {
    return (n == 0) ? 1 : GF_EXP[(GF_LOG[a] * n) % 255];
}

// msg[RS_K] → out[RS_N]（系统码：msg || 校验）
static void rs_encode(const uint8_t *msg, uint8_t *out) {
    uint8_t res[RS_N];
    memcpy(res, msg, RS_K);
    memset(res + RS_K, 0, RS_NROOTS);
    for (int32_t i = 0; i < RS_K; i++) {
        uint8_t coef = res[i];
        if (coef != 0) {
            for (int32_t j = 0; j <= RS_NROOTS; j++) res[i + j] ^= gf_mul(RS_GEN[j], coef);
        }
    }
    memcpy(out, msg, RS_K);
    memcpy(out + RS_K, res + RS_K, RS_NROOTS);
}

static void rs_syndromes(const uint8_t *cw, uint8_t *syn) {
    for (int32_t i = 0; i < RS_NROOTS; i++) {
        uint8_t a = GF_EXP[RS_FCR + i], s = 0;
        for (int32_t j = 0; j < RS_N; j++) s = gf_mul(s, a) ^ cw[j];
        syn[i] = s;
    }
}

// Berlekamp-Massey：返回连接多项式 C（长度 ≤ RS_NROOTS+1），返回长度
// n_syn：输入校验子个数（抹除译码时传入 Forney 校验子长度 RS_NROOTS-n_eras）
static int32_t rs_berlekamp_massey(const uint8_t *syn, int32_t n_syn, uint8_t *C) {
    uint8_t B[RS_NROOTS + 1] = {0}, T[RS_NROOTS + 1];
    int32_t L = 0, m = 1, C_len = 1, B_len = 1;
    uint8_t b = 1;
    memset(C, 0, RS_NROOTS + 1); C[0] = 1;
    B[0] = 1;
    for (int32_t n = 0; n < n_syn; n++) {
        uint8_t d = syn[n];
        for (int32_t i = 1; i <= L && i < C_len; i++) d ^= gf_mul(C[i], syn[n - i]);
        if (d == 0) {
            m++;
        }
        else if (2 * L <= n) {
            int32_t old_len = C_len;
            memcpy(T, C, old_len);
            uint8_t coef = gf_div(d, b);
            uint8_t nC[RS_NROOTS + 1] = {0};
            for (int32_t i = 0; i < old_len; i++) nC[i] = C[i];
            for (int32_t i = 0; i < B_len && i + m <= RS_NROOTS; i++) nC[i + m] ^= gf_mul(coef, B[i]);
            int32_t nC_len = (old_len > B_len + m) ? old_len : B_len + m;
            if (nC_len > RS_NROOTS + 1) nC_len = RS_NROOTS + 1;
            memcpy(C, nC, nC_len);
            C_len = nC_len;
            L = n + 1 - L;
            memcpy(B, T, old_len); // B ← 旧 C
            B_len = old_len;
            b = d;
            m = 1;
        }
        else {
            uint8_t coef = gf_div(d, b);
            int32_t nlen = B_len + m;
            if (nlen > RS_NROOTS + 1) nlen = RS_NROOTS + 1;
            for (int32_t i = C_len; i < nlen; i++) C[i] = 0; // 扩展区清零（JS push(0) 等价）
            for (int32_t i = 0; i < B_len && i + m <= RS_NROOTS; i++) C[i + m] ^= gf_mul(coef, B[i]);
            if (C_len < nlen) C_len = nlen;
            m++;
        }
    }
    return C_len;
}

// 译码一个 RS(32,16) 码字。返回纠正字节数；fail 置 1 表示译码失败（此时 out 仍为未纠正数据）。
static int32_t rs_decode(const uint8_t *cw, uint8_t *out, int32_t *fail) {
    uint8_t syn[RS_NROOTS];
    int32_t syn_zero = 1;
    *fail = 0;
    rs_syndromes(cw, syn);
    for (int32_t i = 0; i < RS_NROOTS; i++) if (syn[i] != 0) { syn_zero = 0; break; }
    if (syn_zero) { memcpy(out, cw, RS_K); return 0; }

    uint8_t lambda[RS_NROOTS + 1];
    int32_t lambda_len = rs_berlekamp_massey(syn, RS_NROOTS, lambda);

    // Chien 搜索错误位置
    int32_t positions[RS_NROOTS], n_pos = 0;
    for (int32_t i = 0; i < RS_N; i++) {
        uint8_t X = GF_EXP[(255 - i) % 255], y = 0;
        for (int32_t j = 0; j < lambda_len; j++) y ^= gf_mul(lambda[j], gf_pow(X, j));
        if (y == 0) positions[n_pos++] = RS_N - 1 - i;
    }
    if (n_pos != lambda_len - 1) {
        memcpy(out, cw, RS_K);
        *fail = 1;
        return 0;
    }

    // Forney：计算错误值并纠正
    uint8_t omega[RS_NROOTS] = {0};
    for (int32_t i = 0; i < RS_NROOTS; i++) {
        for (int32_t j = 0; j <= i && j < lambda_len; j++) omega[i] ^= gf_mul(lambda[j], syn[i - j]);
    }
    uint8_t buf[RS_N];
    memcpy(buf, cw, RS_N);
    for (int32_t p = 0; p < n_pos; p++) {
        int32_t pos = positions[p];
        uint8_t X = GF_EXP[(255 - (RS_N - 1 - pos)) % 255];
        uint8_t num = 0, den = 0;
        for (int32_t j = 0; j < RS_NROOTS; j++) num ^= gf_mul(omega[j], gf_pow(X, j));
        for (int32_t j = 1; j < lambda_len; j += 2) den ^= gf_mul(lambda[j], gf_pow(X, j - 1));
        buf[pos] ^= (den == 0) ? 0 : gf_div(num, den);
    }
    // 校验纠正结果
    uint8_t syn2[RS_NROOTS];
    rs_syndromes(buf, syn2);
    for (int32_t i = 0; i < RS_NROOTS; i++) {
        if (syn2[i] != 0) {
            memcpy(out, cw, RS_K);
            *fail = 1;
            return 0;
        }
    }
    memcpy(out, buf, RS_K);
    return n_pos;
}

// 含抹除的 RS 译码（errors-and-erasures）：抹除位置已知（低置信字节），未知错误位置未知。
// 纠错能力：2v + e ≤ RS_NROOTS（v=未知错误数，e=抹除数），已知位置不占未知错误搜索维度，
// 衰落信道中（错误集中于深衰落载波）等效纠错能力近似翻倍。
// 位置约定与 rs_decode 一致：pos ↔ X = α^{(255-(RS_N-1-pos)) mod 255}。
// 返回纠正字节数（含抹除）；fail 置 1 表示译码失败（out 为未纠正数据）。
static int32_t rs_decode_eras(const uint8_t *cw, uint8_t *out, int32_t *fail,
                              const int32_t *eras, int32_t n_eras) {
    if (n_eras <= 0) return rs_decode(cw, out, fail);
    *fail = 0;

    // 1) 校验子（无错捷径）
    uint8_t syn[RS_NROOTS];
    rs_syndromes(cw, syn);
    int32_t syn_zero = 1;
    for (int32_t i = 0; i < RS_NROOTS; i++) if (syn[i] != 0) { syn_zero = 0; break; }
    if (syn_zero) { memcpy(out, cw, RS_K); return 0; }

    // 2) 抹除定位多项式 Γ(x) = Π(1 - x/X_j)，根与本代码 Chien 约定一致（X_j = α^{(pos-31) mod 255}）
    uint8_t gamma[RS_NROOTS + 1];
    memset(gamma, 0, sizeof(gamma));
    gamma[0] = 1;
    int32_t gamma_len = 1;
    for (int32_t j = 0; j < n_eras; j++) {
        uint8_t Xi = GF_EXP[(RS_N - 1 - eras[j]) % 255]; // X_j^{-1} = α^{(31-pos) mod 255}
        for (int32_t k = gamma_len; k > 0; k--) gamma[k] ^= gf_mul(gamma[k - 1], Xi);
        gamma_len++;
    }

    // 3) Forney 校验子 = (S(x)·Γ(x)) mod x^RS_NROOTS，去掉前 n_eras 项
    uint8_t fsyn_full[RS_NROOTS] = {0};
    for (int32_t i = 0; i < RS_NROOTS; i++)
        for (int32_t j = 0; j <= i && j < gamma_len; j++)
            fsyn_full[i] ^= gf_mul(syn[i - j], gamma[j]);
    uint8_t fsyn[RS_NROOTS];
    int32_t fsyn_len = RS_NROOTS - n_eras;
    for (int32_t i = 0; i < fsyn_len; i++) fsyn[i] = fsyn_full[i + n_eras];

    // 4) BM 求未知错误定位多项式 Λ
    uint8_t lambda[RS_NROOTS + 1];
    int32_t lambda_len = rs_berlekamp_massey(fsyn, fsyn_len, lambda);

    // 5) 总 errata 定位多项式 = Λ·Γ
    uint8_t errata[RS_NROOTS + 1] = {0};
    for (int32_t i = 0; i < lambda_len; i++)
        for (int32_t j = 0; j < gamma_len; j++)
            errata[i + j] ^= gf_mul(lambda[i], gamma[j]);
    int32_t errata_len = lambda_len + gamma_len - 1;

    // 6) Chien 搜索全部 errata 位置
    int32_t positions[RS_NROOTS], n_pos = 0;
    for (int32_t i = 0; i < RS_N; i++) {
        uint8_t X = GF_EXP[(255 - i) % 255], y = 0;
        for (int32_t j = 0; j < errata_len; j++) y ^= gf_mul(errata[j], gf_pow(X, j));
        if (y == 0) positions[n_pos++] = RS_N - 1 - i;
    }
    if (n_pos != errata_len - 1) {
        memcpy(out, cw, RS_K);
        *fail = 1;
        return 0;
    }

    // 7) errata 求值多项式 Ω = (S(x)·errata(x)) mod x^RS_NROOTS，Forney 求 errata 值
    uint8_t omega[RS_NROOTS] = {0};
    for (int32_t i = 0; i < RS_NROOTS; i++) {
        for (int32_t j = 0; j <= i && j < errata_len; j++) omega[i] ^= gf_mul(errata[j], syn[i - j]);
    }
    uint8_t buf[RS_N];
    memcpy(buf, cw, RS_N);
    for (int32_t p = 0; p < n_pos; p++) {
        int32_t pos = positions[p];
        uint8_t X = GF_EXP[(255 - (RS_N - 1 - pos)) % 255];
        uint8_t num = 0, den = 0;
        for (int32_t j = 0; j < RS_NROOTS; j++) num ^= gf_mul(omega[j], gf_pow(X, j));
        for (int32_t j = 1; j < errata_len; j += 2) den ^= gf_mul(errata[j], gf_pow(X, j - 1));
        buf[pos] ^= (den == 0) ? 0 : gf_div(num, den);
    }
    // 校验纠正结果
    uint8_t syn2[RS_NROOTS];
    rs_syndromes(buf, syn2);
    for (int32_t i = 0; i < RS_NROOTS; i++) {
        if (syn2[i] != 0) {
            memcpy(out, cw, RS_K);
            *fail = 1;
            return 0;
        }
    }
    memcpy(out, buf, RS_K);
    return n_pos;
}

// ============================================================================
// 交织/解交织、加扰、QAM4 解映射
// ============================================================================

static void interleave(const uint8_t *data, uint8_t *out, int32_t rows, int32_t total) {
    int32_t cols = total / rows;
    for (int32_t c = 0; c < cols; c++)
        for (int32_t r = 0; r < rows; r++)
            out[c * rows + r] = data[r * cols + c];
}
static void deinterleave(const uint8_t *data, uint8_t *out, int32_t rows, int32_t total) {
    int32_t cols = total / rows;
    for (int32_t r = 0; r < rows; r++)
        for (int32_t c = 0; c < cols; c++)
            out[r * cols + c] = data[c * rows + r];
}
// float 版解交织（线字节置信度随数据同构解交织，供抹除选择）
static void deinterleave_f(const float *data, float *out, int32_t rows, int32_t total) {
    int32_t cols = total / rows;
    for (int32_t r = 0; r < rows; r++)
        for (int32_t c = 0; c < cols; c++)
            out[r * cols + c] = data[c * rows + r];
}

// 加扰/解扰（同一变换）：LFSR 初态 0xACE1，多项式 0xB400
static void scramble_stream(const uint8_t *data, uint8_t *out, int32_t len) {
    uint32_t lfsr = 0xACE1;
    for (int32_t i = 0; i < len; i++) {
        uint8_t b = 0;
        for (int32_t k = 0; k < 8; k++) {
            uint32_t bit = lfsr & 1;
            lfsr = (lfsr >> 1) ^ (bit ? 0xB400 : 0);
            b = (b >> 1) | (uint8_t)(bit << 7);
        }
        out[i] = data[i] ^ b;
    }
}

// QAM4 解映射：input_i/q（DATA_CARRIERS 点）→ out（BYTES_PER_SYMBOL 字节）
static void qam4_decoding(const float *input_i, const float *input_q, uint8_t *out) {
    uint8_t cur = 0;
    for (int32_t t = 0; t < OFDM_DATA_CARRIERS; t++) {
        uint8_t s = QAM4_L[(input_i[t] >= 0) ? 1 : 0][(input_q[t] >= 0) ? 1 : 0];
        cur |= (uint8_t)(s << ((3 - (t % 4)) * 2));
        if (t % 4 == 3) { out[t / 4] = cur; cur = 0; }
    }
}

// ============================================================================
// 发射机
// ============================================================================

struct OFDM_TX {
    ofdm_alloc_fn alloc;
    ofdm_free_fn dealloc;
    uint8_t *tx_bytes;     // n_packets × PKT_WIRE_LEN（RS编码+交织+加扰后的线字节）
    uint32_t n_packets;
    uint32_t next_packet;  // 下一个待渲染的 packet（帧）
    float *bb_re, *bb_im;  // 一帧基带槽流（FRAME_SLOTS × SLOT_BB = 3960 点）
    float sym_re[OFDM_BB_FFT_LEN], sym_im[OFDM_BB_FFT_LEN]; // 符号构建暂存
};

// 发射：文本 → 定长 packet 序列（每 packet 一帧），创建时完成编码，渲染时逐帧调制
OFDM_TX *ofdm_tx_create(ofdm_alloc_fn alloc, ofdm_free_fn dealloc,
                        const uint8_t *payload_utf8, uint32_t payload_len) {
    if (!s_tables_ready) return NULL; // 须先 ofdm_tables_init（寻呼机入口已初始化）
    if (!alloc || !dealloc) return NULL;
    if (!payload_utf8 && payload_len > 0) return NULL;

    OFDM_TX *tx = (OFDM_TX *)alloc(sizeof(OFDM_TX));
    if (!tx) return NULL;
    memset(tx, 0, sizeof(OFDM_TX));
    tx->alloc = alloc; tx->dealloc = dealloc;

    tx->n_packets = (payload_len + OFDM_PKT_PAYLOAD_MAX - 1) / OFDM_PKT_PAYLOAD_MAX;
    if (tx->n_packets == 0) tx->n_packets = 1;

    tx->tx_bytes = (uint8_t *)alloc(tx->n_packets * OFDM_PKT_WIRE_LEN);
    tx->bb_re = (float *)alloc(OFDM_FRAME_SLOTS * OFDM_SLOT_BB * sizeof(float));
    tx->bb_im = (float *)alloc(OFDM_FRAME_SLOTS * OFDM_SLOT_BB * sizeof(float));
    if (!tx->tx_bytes || !tx->bb_re || !tx->bb_im) {
        ofdm_tx_destroy(tx);
        return NULL;
    }

    // 逐 packet：组块 → RS 编码 → 交织 → 加扰 → 线字节
    for (uint32_t i = 0; i < tx->n_packets; i++) {
        uint32_t chunk_len = payload_len - i * OFDM_PKT_PAYLOAD_MAX;
        if (chunk_len > OFDM_PKT_PAYLOAD_MAX) chunk_len = OFDM_PKT_PAYLOAD_MAX;
        uint8_t block[OFDM_PKT_UNCODED_LEN];
        uint8_t coded[OFDM_PKT_WIRE_LEN];
        uint8_t ilv[OFDM_PKT_WIRE_LEN];
        memset(block, 0, sizeof(block));
        memcpy(block, PKT_MAGIC, 6);
        block[6] = (uint8_t)chunk_len;
        block[7] = (uint8_t)(i % 256);
        if (chunk_len > 0) memcpy(block + OFDM_PKT_HEADER_LEN, payload_utf8 + i * OFDM_PKT_PAYLOAD_MAX, chunk_len);
        for (int32_t b = 0; b < OFDM_PKT_RS_BLOCKS; b++)
            rs_encode(block + b * RS_K, coded + b * RS_N);
        interleave(coded, ilv, OFDM_FRAME_DATA_SYMBOLS, OFDM_PKT_WIRE_LEN);
        scramble_stream(ilv, tx->tx_bytes + i * OFDM_PKT_WIRE_LEN, OFDM_PKT_WIRE_LEN);
    }
    tx->next_packet = 0;
    return tx;
}

void ofdm_tx_destroy(OFDM_TX *tx) {
    if (!tx) return;
    if (tx->tx_bytes) tx->dealloc(tx->tx_bytes);
    if (tx->bb_re) tx->dealloc(tx->bb_re);
    if (tx->bb_im) tx->dealloc(tx->bb_im);
    tx->dealloc(tx);
}

void ofdm_tx_rewind(OFDM_TX *tx) {
    if (tx) tx->next_packet = 0;
}

uint32_t ofdm_tx_frame_count(const OFDM_TX *tx) {
    return tx ? tx->n_packets : 0;
}

// 附加一基带槽（CP + 符号）到 bb 流
static inline int32_t append_slot(float *bb_re, float *bb_im, int32_t p,
                                  const float *sym_re, const float *sym_im) {
    for (int32_t i = OFDM_BB_FFT_LEN - OFDM_CP_BB; i < OFDM_BB_FFT_LEN; i++) {
        bb_re[p] = sym_re[i]; bb_im[p] = sym_im[i]; p++;
    }
    for (int32_t i = 0; i < OFDM_BB_FFT_LEN; i++) {
        bb_re[p] = sym_re[i]; bb_im[p] = sym_im[i]; p++;
    }
    return p;
}

// 数据符号的基带复符号：56个数据子载波承载QAM4数据，8个散布导频点承载已知导频
static void build_data_symbol_bb(OFDM_TX *tx, const uint8_t *bytes14, int32_t s) {
    float si[OFDM_CARRIER_NUMBER], sq[OFDM_CARRIER_NUMBER];
    float pts_i[OFDM_DATA_CARRIERS], pts_q[OFDM_DATA_CARRIERS];
    int32_t np = 0;
    for (int32_t i = 0; i < OFDM_BYTES_PER_SYMBOL; i++) {
        uint8_t b = bytes14[i];
        pts_i[np] = QAM4_M[(b & 192) >> 6][0]; pts_q[np] = QAM4_M[(b & 192) >> 6][1]; np++;
        pts_i[np] = QAM4_M[(b & 48) >> 4][0];  pts_q[np] = QAM4_M[(b & 48) >> 4][1];  np++;
        pts_i[np] = QAM4_M[(b & 12) >> 2][0];  pts_q[np] = QAM4_M[(b & 12) >> 2][1];  np++;
        pts_i[np] = QAM4_M[b & 3][0];          pts_q[np] = QAM4_M[b & 3][1];          np++;
    }
    int32_t po = (PILOT_SHIFT * s) % PILOT_SPACING;
    int32_t di = 0;
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        if (c % PILOT_SPACING == po) { si[c] = PILOT_I[c]; sq[c] = PILOT_Q[c]; }
        else { si[c] = pts_i[di]; sq[c] = pts_q[di]; di++; }
    }
    baseband_symbol(si, sq, tx->sym_re, tx->sym_im);
}

// 渲染下一个物理帧（SC前导 + 训练×2 + 数据/中间训练调度），逐帧双区峰值归一化到 0.9
int32_t ofdm_tx_render_frame(OFDM_TX *tx, float *out) {
    if (!tx || !out) return -1;
    if (tx->next_packet >= tx->n_packets) return 1;

    const uint8_t *wire = tx->tx_bytes + tx->next_packet * OFDM_PKT_WIRE_LEN;
    float *bb_re = tx->bb_re, *bb_im = tx->bb_im;
    int32_t p = 0;

    // SC 粗同步前导（无 CP 的一基带槽）
    memcpy(bb_re, SC_BB_RE, OFDM_SLOT_BB * sizeof(float));
    memcpy(bb_im, SC_BB_IM, OFDM_SLOT_BB * sizeof(float));
    p += OFDM_SLOT_BB;

    // 训练A、训练B
    p = append_slot(bb_re, bb_im, p, TRAINING_BB_RE, TRAINING_BB_IM);
    p = append_slot(bb_re, bb_im, p, TRAINING_BB_RE, TRAINING_BB_IM);

    // 数据/中间训练调度
    for (int32_t s = 0; s < OFDM_FRAME_DATA_SYMBOLS; s++) {
        build_data_symbol_bb(tx, wire + s * OFDM_BYTES_PER_SYMBOL, s);
        p = append_slot(bb_re, bb_im, p, tx->sym_re, tx->sym_im);
        if ((s + 1) % 5 == 0 && s + 1 < OFDM_FRAME_DATA_SYMBOLS)
            p = append_slot(bb_re, bb_im, p, TRAINING_BB_RE, TRAINING_BB_IM);
    }

    // 基带 → 通带
    synth_passband(bb_re, bb_im, p, out);

    // 逐帧峰值归一化：SC 槽与帧体分别归一到 0.9
    for (int32_t region = 0; region < 2; region++) {
        int32_t n0 = (region == 0) ? 0 : OFDM_GROSS_SYMBOL_LENGTH;
        int32_t n1 = (region == 0) ? OFDM_GROSS_SYMBOL_LENGTH : OFDM_FRAME_LENGTH;
        float peak = 0.0f;
        for (int32_t n = n0; n < n1; n++) {
            float a = fabsf(out[n]);
            if (a > peak) peak = a;
        }
        if (peak > 0.0f) {
            float sc = 0.9f / peak;
            for (int32_t n = n0; n < n1; n++) out[n] *= sc;
        }
    }

    tx->next_packet++;
    return 0;
}

// ============================================================================
// 接收机（流式状态机："sync" 搜索SC前导 ⇄ "frame" 帧内逐槽接收）
// ============================================================================

// ---------------- 环形缓冲（定容、O(1)写入/丢弃、稳态零堆分配） ----------------
typedef struct {
    float *buf;
    uint32_t mask;   // cap - 1（容量为 2 的幂）
    uint32_t start;
    uint32_t len;
} Ring;

static void ring_clear(Ring *r) { r->start = 0; r->len = 0; }

// 满时覆盖最旧采样
static void ring_write(Ring *r, const float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (r->len <= r->mask) { r->buf[(r->start + r->len) & r->mask] = x[i]; r->len++; }
        else { r->buf[r->start] = x[i]; r->start = (r->start + 1) & r->mask; }
    }
}
static void ring_drop(Ring *r, uint32_t n) {
    if (n > r->len) n = r->len;
    r->start = (r->start + n) & r->mask;
    r->len -= n;
}
static void ring_read_to(const Ring *r, float *out, uint32_t off, uint32_t n) {
    uint32_t j = r->start + off;
    for (uint32_t i = 0; i < n; i++) out[i] = r->buf[(j + i) & r->mask];
}

// ---------------- 接收机状态 ----------------

typedef enum { RX_STATE_SYNC = 0, RX_STATE_FRAME = 1 } RX_State;

struct OFDM_RX {
    ofdm_alloc_fn alloc;
    ofdm_free_fn dealloc;
    ofdm_rx_text_cb text_cb;
    ofdm_rx_log_cb log_cb;
    void *user;

    Ring det_ring;           // SC检测历史
    Ring buf_ring;           // 帧接收缓冲
    float *det_buf;          // DET_RING_CAP
    float *buf_buf;          // BUF_RING_CAP

    RX_State state;
    uint32_t stall_count;

    // SC 检测器
    uint64_t feed_abs;       // 累计喂入采样数（绝对时钟）
    uint64_t searched_upto;
    float best_metric;
    uint64_t best_offset;

    // 帧内状态
    int32_t frame_slot;      // 已消费到的槽号（0=SC前导，1..=FRAME_SCHEDULE）
    int32_t frame_data_idx;  // 帧内数据符号索引（决定散布导频位置）
    uint8_t pkt_buf[OFDM_PKT_WIRE_LEN];
    float pkt_conf[OFDM_PKT_WIRE_LEN]; // 各线字节置信度（抹除选择依据）
    int32_t pkt_len;
    int32_t last_seq;

    // 信道估计与频偏补偿
    int32_t chan_valid;
    float chan_hi[OFDM_CARRIER_NUMBER], chan_hq[OFDM_CARRIER_NUMBER];
    float trA_i[OFDM_CARRIER_NUMBER], trA_q[OFDM_CARRIER_NUMBER]; // 训练A原始IQ（与B配对估计CFO/SFO）
    int32_t trA_valid;
    float cfo_a, sfo_b;      // 相位速率 φ(槽) = a + b·f（b 为门控后的启用值）
    float fit_res;           // 训练A/B相位拟合残差（rad），真伪同步判据
    float sfo_b_raw;         // SFO 斜率原始拟合值（未门控；供训练A相位对齐平均用）
    float sfo_b_ema;         // SFO 斜率的跨帧 EMA 平滑值
    int32_t sfo_enable_streak; // SFO 超门限连续帧数（迟滞确认）
    int32_t cfo_ref_slot;    // CFO/SFO补偿参考面所在槽号（逐训练槽推进）
    int32_t tau_ref;         // 信道估计的定时参考（最近训练槽细同步位置）
    int32_t trainA_fbd;      // 训练A的细同步位置（拟合扣除定时差用）
    int32_t train_miss;      // 训练符号连续失锁计数（容忍1次）

    // 符号处理工作区
    float sym_blk[SYM_BLK_LEN];
    float mix_re[SYM_BLK_LEN], mix_im[SYM_BLK_LEN];
    float bb_re[OFDM_BB_FFT_LEN], bb_im[OFDM_BB_FFT_LEN];
    float iq_i[OFDM_CARRIER_NUMBER], iq_q[OFDM_CARRIER_NUMBER];

    // 遥测/统计
    uint32_t frames_ok, frames_bad, rs_corrected, rs_fail_blocks;
    float cfo_hz, sfo_ppm;

    // 可视化钩子（最近一个数据符号）
    float viz_wave[OFDM_SYMBOL_LENGTH];
    float viz_iq_i[OFDM_CARRIER_NUMBER], viz_iq_q[OFDM_CARRIER_NUMBER];
    uint32_t viz_seq;

    // 前置高通（HPF）状态与分片暂存
    float hp_x1, hp_x2, hp_y1, hp_y2;
    float dc_scratch[OFDM_GROSS_SYMBOL_LENGTH];
};

static void rx_log(OFDM_RX *rx, const char *msg) {
    if (rx->log_cb) rx->log_cb(msg, rx->user);
}

// ---------------- OFDM 解调：IQ混频 → 多相抽取 → 复FFT ----------------
// blk: 全速率实采样块（符号有用部分起点位于块内 LPF_GD 处，左右各 GD 余量供滤波器瞬态）
// 混频相位与发射端同为"槽内局部相位"（符号起点 = 槽内第 CP_LENGTH 采样），输出 oi/oq。
static void demodulate_ofdm_symbol(OFDM_RX *rx, const float *blk, float *oi, float *oq) {
    // 1) IQ 混频到零中频（×e^{-jωn}）：DDS 初相对齐符号起点（= 槽内第 CP_LENGTH 采样）
    float phase = (float)fmod(MIX_STEP * (OFDM_CP_LENGTH - LPF_GD), TWO_PI);
    for (int32_t n = 0; n < SYM_BLK_LEN; n++) {
        int32_t k = (int32_t)(phase * PHASE_TO_IDX) & (DDS_LUT_LEN - 1);
        rx->mix_re[n] = blk[n] * DDS_COS[k];
        rx->mix_im[n] = -blk[n] * DDS_SIN[k];
        phase += (float)MIX_STEP;
        if (phase >= (float)TWO_PI) phase -= (float)TWO_PI;
    }
    // 2) 多相抽取 ×DECIM（含群延迟对齐：bb[m] 对应块内时刻 LPF_GD + m·DECIM）
    for (int32_t m = 0; m < OFDM_BB_FFT_LEN; m++) {
        int32_t c0 = 2 * LPF_GD + m * OFDM_DECIM;
        float ar = 0.0f, ai = 0.0f;
        for (int32_t j = 0; j < LPF_TAPS; j++) {
            int32_t x = c0 - j;
            ar += LPF[j] * rx->mix_re[x];
            ai += LPF[j] * rx->mix_im[x];
        }
        rx->bb_re[m] = ar; rx->bb_im[m] = ai;
    }
    // 3) 复 FFT，读载波 bin（×2 补偿实混频的半幅度）
    fft_radix2(rx->bb_re, rx->bb_im, 0);
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        int32_t k = bb_bin(c);
        oi[c] = 2.0f * rx->bb_re[k];
        oq[c] = 2.0f * rx->bb_im[k];
    }
}

// ---- 信道估计与频偏补偿 ----
// 由训练符号IQ估计各子载波信道响应 H = rx/tx
static void channel_from_training(const float *tr_i, const float *tr_q, float *hi, float *hq) {
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        float ti = TRAINING_I[c], tq = TRAINING_Q[c];
        float denom = ti * ti + tq * tq;
        hi[c] = (tr_i[c] * ti + tr_q[c] * tq) / denom;
        hq[c] = (tr_q[c] * ti - tr_i[c] * tq) / denom;
    }
}

// 训练A/B逐载波相位差 → 拟合相位速率 φ(槽) = a + b·f（a≈CFO，b≈SFO）
// 关键：相位差中含两段训练细同步的整数定时差 Δfbd 引起的线性斜坡 2πf·Δfbd/Fs，
// 与 SFO 不可区分，必须先扣除（dtau 参数），否则定时抖动(±1采样≈±1111ppm)被误判为SFO。
// 斜率项 b 跨帧 EMA 平滑（定时抖动零均值、真实SFO恒定相干积累），高门限迟滞启用。
static void fit_cfo(OFDM_RX *rx, const float *h1i, const float *h1q, const float *h2i, const float *h2q, int32_t dtau) {
    double sw = 0, swf = 0, swp = 0, swff = 0, swfp = 0;
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        double dpr = (double)h2i[c] * h1i[c] + (double)h2q[c] * h1q[c]; // H2·conj(H1)
        double dpi = (double)h2q[c] * h1i[c] - (double)h2i[c] * h1q[c];
        double phi = atan2(dpi, dpr) + TWO_PI * CARRIER_FREQS_F[c] * dtau / OFDM_SAMPLE_RATE; // 扣除定时斜坡
        double w = sqrt(dpr * dpr + dpi * dpi); // |H1||H2| 加权，深衰落载波降权
        double f = CARRIER_FREQS_F[c];
        sw += w; swf += w * f; swp += w * phi; swff += w * f * f; swfp += w * f * phi;
    }
    if (sw < 1e-12) return;
    double det = sw * swff - swf * swf;
    double a, b;
    if (fabs(det) < 1e-12) { a = swp / sw; b = 0; }
    else {
        a = (swp * swff - swf * swfp) / det;
        b = (sw * swfp - swf * swp) / det;
    }
    // 相位拟合残差（rad）：真前导训练对 H1/H2 相位差为低阶斜坡（残差小），
    // 帧内伪锁定（训练位置落在数据符号上）相位差随机（残差≈π/√3）
    {
        double res = 0, wsum = 0;
        for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
            double dpr = (double)h2i[c] * h1i[c] + (double)h2q[c] * h1q[c];
            double dpi = (double)h2q[c] * h1i[c] - (double)h2i[c] * h1q[c];
            double phi = atan2(dpi, dpr) + TWO_PI * CARRIER_FREQS_F[c] * dtau / OFDM_SAMPLE_RATE;
            double d = phi - (a + b * CARRIER_FREQS_F[c]);
            d = atan2(sin(d), cos(d)); // 回绕到 (-π,π]
            double w = sqrt(dpr * dpr + dpi * dpi);
            res += w * d * d; wsum += w;
        }
        rx->fit_res = (wsum > 1e-12) ? (float)sqrt(res / wsum) : 9.9f;
    }
    // 拟合质量门控（仅作用于 CFO 公共项）：残差大（含噪/伪拟合）时，宁可不施加显式
    // CFO 补偿——错误的显式去旋转会把逐槽相位斜坡注入后续全部数据符号，比不补偿更糟；
    // 散布导频逐符号公共相位跟踪在小频偏下本就能稳健收尾。
    rx->cfo_a = (rx->fit_res > 0.5f) ? 0.0f : (float)a;
    rx->sfo_b_raw = (float)b; // 原始拟合值（供训练A相位对齐；拟合可信时才有意义）
    // SFO 斜率项：跨帧 EMA 平滑（定时抖动零均值、真实 SFO 相干积累，个别帧的
    // 含噪估计在平滑中相互抵消），高门限（500ppm）+连续2帧确认的迟滞启用——
    // ≤300ppm 时被动跟踪已足够稳健；门限仅作为病态大频偏的安全阀。
    rx->sfo_b_ema = (1.0f - SFO_EMA_ALPHA) * rx->sfo_b_ema + SFO_EMA_ALPHA * (float)b;
    float ppm = (float)(rx->sfo_b_ema / (TWO_PI * T_SLOT) * 1e6);
    if (fabsf(ppm) >= SFO_ENABLE_PPM) {
        if (rx->sfo_enable_streak < 3) rx->sfo_enable_streak++;
    }
    else {
        rx->sfo_enable_streak = 0;
    }
    float ppm_used = 0.0f;
    if (rx->sfo_enable_streak >= SFO_ENABLE_STREAK) {
        ppm_used = ppm;
        if (ppm_used > SFO_CLAMP_PPM) ppm_used = SFO_CLAMP_PPM;
        if (ppm_used < -SFO_CLAMP_PPM) ppm_used = -SFO_CLAMP_PPM;
    }
    rx->sfo_b = (float)(ppm_used * (TWO_PI * T_SLOT) / 1e6);
    rx->cfo_hz = (float)(a / (TWO_PI * T_SLOT));
    rx->sfo_ppm = ppm; // 遥测报告平滑值（实际施加的补偿为门控后 ppm_used）
    char msg[128];
    snprintf(msg, sizeof(msg), "CFO/SFO 估计：CFO=%.3fHz，SFO=%.1fppm（平滑后 %.1f，%s）相位残差=%.2f",
             (double)rx->cfo_hz, b / (TWO_PI * T_SLOT) * 1e6, (double)ppm,
             (ppm_used == 0.0f) ? "被动跟踪" : "启用补偿", (double)rx->fit_res);
    rx_log(rx, msg);
}

// 对IQ施加显式CFO/SFO补偿：各子载波乘以 e^{-j(a+b·f)·slots}
static void derotate_iq(OFDM_RX *rx, float *iq_i, float *iq_q, int32_t slots) {
    if (slots == 0 || (rx->cfo_a == 0.0f && rx->sfo_b == 0.0f)) return;
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        float ph = (rx->cfo_a + rx->sfo_b * CARRIER_FREQS_F[c]) * slots;
        float cr = cosf(ph), ci = sinf(ph);
        float ii = iq_i[c], qq = iq_q[c];
        iq_i[c] = ii * cr + qq * ci;
        iq_q[c] = qq * cr - ii * ci;
    }
}

// 散布导频：估计本符号残余公共相位并去旋转，同时在导频点跟踪更新信道
static void pilot_track(OFDM_RX *rx, float *iq_i, float *iq_q, int32_t s) {
    int32_t po = (PILOT_SHIFT * s) % PILOT_SPACING;
    float zr = 0.0f, zi = 0.0f;
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        if (c % PILOT_SPACING != po) continue;
        // 期望接收值 = H·P
        float er = rx->chan_hi[c] * PILOT_I[c] - rx->chan_hq[c] * PILOT_Q[c];
        float ei = rx->chan_hi[c] * PILOT_Q[c] + rx->chan_hq[c] * PILOT_I[c];
        zr += iq_i[c] * er + iq_q[c] * ei; // Σ rx·conj(期望)
        zi += iq_q[c] * er - iq_i[c] * ei;
    }
    float theta = atan2f(zi, zr);
    if (fabsf(theta) > 1e-9f) {
        float cr = cosf(theta), ci = sinf(theta);
        for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
            float ii = iq_i[c], qq = iq_q[c];
            iq_i[c] = ii * cr + qq * ci;
            iq_q[c] = qq * cr - ii * ci;
        }
    }
    // 导频点信道跟踪（相位已对齐，α=0.25 指数融合）
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        if (c % PILOT_SPACING != po) continue;
        float pr = PILOT_I[c], pq = PILOT_Q[c], denom = pr * pr + pq * pq;
        float hr = (iq_i[c] * pr + iq_q[c] * pq) / denom;
        float hq = (iq_q[c] * pr - iq_i[c] * pq) / denom;
        rx->chan_hi[c] = (1.0f - PILOT_TRACK_ALPHA) * rx->chan_hi[c] + PILOT_TRACK_ALPHA * hr;
        rx->chan_hq[c] = (1.0f - PILOT_TRACK_ALPHA) * rx->chan_hq[c] + PILOT_TRACK_ALPHA * hq;
    }
}

// 单抽头频域均衡
static void equalize(OFDM_RX *rx, float *iq_i, float *iq_q) {
    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
        float hi = rx->chan_hi[c], hq = rx->chan_hq[c];
        float denom = hi * hi + hq * hq;
        if (denom < 1e-6f) denom = 1e-6f;
        float ii = iq_i[c], qq = iq_q[c];
        iq_i[c] = (ii * hi + qq * hq) / denom;
        iq_q[c] = (qq * hi - ii * hq) / denom;
    }
}

// ---------------- 帧尾解包：一个物理帧 = 一个 packet ----------------
// RS 抹除重试策略（永不劣化）：先走原 errors-only 译码（满血 8 错误能力）；
// 失败时再把置信度最低的若干线字节标为已知位置抹除重试（2v+e≤16，深衰落字节
// 不占未知错误搜索维度，衰落信道下等效纠错能力近似翻倍）。
#define RS_ERASE_MAX (12) // 抹除重试的最大抹除数（保留 ≥2 个未知错误纠正能力）
static void decode_frame(OFDM_RX *rx, const uint8_t *pkt_raw, const float *conf_raw) {
    uint8_t pkt[OFDM_PKT_WIRE_LEN];
    uint8_t rx_coded[OFDM_PKT_WIRE_LEN];
    float conf[OFDM_PKT_WIRE_LEN];
    uint8_t block[OFDM_PKT_UNCODED_LEN];
    uint8_t blk_fail[OFDM_PKT_RS_BLOCKS]; // 失败块记录（净荷中以'?'占位显示）
    int32_t corrected = 0, fail_blocks = 0, erased = 0;
    memset(blk_fail, 0, sizeof(blk_fail));

    scramble_stream(pkt_raw, pkt, OFDM_PKT_WIRE_LEN);
    deinterleave(pkt, rx_coded, OFDM_FRAME_DATA_SYMBOLS, OFDM_PKT_WIRE_LEN);
    deinterleave_f(conf_raw, conf, OFDM_FRAME_DATA_SYMBOLS, OFDM_PKT_WIRE_LEN);
    for (int32_t b = 0; b < OFDM_PKT_RS_BLOCKS; b++) {
        int32_t fail = 0;
        int32_t c = rs_decode(rx_coded + b * RS_N, block + b * RS_K, &fail);
        if (fail) {
            // errors-only 失败：按置信度升序选 RS_ERASE_MAX 个抹除重试
            //（32 元选择排序开销可忽略；索引-置信度对插入排序）
            int32_t idx[RS_N];
            for (int32_t i = 0; i < RS_N; i++) idx[i] = i;
            for (int32_t i = 1; i < RS_N; i++) {
                int32_t t = idx[i], j = i - 1;
                while (j >= 0 && conf[b * RS_N + idx[j]] > conf[b * RS_N + t]) { idx[j + 1] = idx[j]; j--; }
                idx[j + 1] = t;
            }
            int32_t eras[RS_ERASE_MAX];
            for (int32_t k = 0; k < RS_ERASE_MAX; k++) eras[k] = idx[k];
            c = rs_decode_eras(rx_coded + b * RS_N, block + b * RS_K, &fail, eras, RS_ERASE_MAX);
            if (!fail) erased += RS_ERASE_MAX; // 仅统计成功解救的抹除数
        }
        corrected += c;
        if (fail) { fail_blocks++; blk_fail[b] = 1; }
    }
    // 失败块内容不可信（rs_decode 失败时拷贝的是原始损坏字节，可能含 NUL 与非法
    // UTF-8，会把显示链路整体截断）：统一以 '?' 占位——净荷保持可显示 ASCII，
    // 且保留“哪一段缺了”的位置信息（块 0 含 magic/len/seq，已经过 syndrome 校验）。
    for (int32_t b = 0; b < OFDM_PKT_RS_BLOCKS; b++)
        if (blk_fail[b]) memset(block + b * RS_K, '?', RS_K);

    int32_t magic_ok = (memcmp(block, PKT_MAGIC, 6) == 0);
    if (!magic_ok) {
        rx->frames_bad++;
        rx->rs_fail_blocks += OFDM_PKT_RS_BLOCKS;
        char msg[96];
        snprintf(msg, sizeof(msg),
                 "坏帧（magic 不符，丢弃）头部=[%02x %02x %02x %02x %02x %02x] RS失败块=%d",
                 block[0], block[1], block[2], block[3], block[4], block[5], (int)fail_blocks);
        rx_log(rx, msg);
        return;
    }

    rx->frames_ok++;
    rx->rs_corrected += (uint32_t)corrected;
    rx->rs_fail_blocks += (uint32_t)fail_blocks;

    uint8_t seq = block[7];
    if (rx->last_seq >= 0 && seq != 0 && seq != (uint8_t)((rx->last_seq + 1) % 256)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "帧序号不连续：%d → %d（丢帧）", (int)rx->last_seq, (int)seq);
        rx_log(rx, msg);
    }
    rx->last_seq = seq;

    uint32_t len = block[6];
    if (len > OFDM_PKT_PAYLOAD_MAX) len = OFDM_PKT_PAYLOAD_MAX;
    {
        char msg[96];
        snprintf(msg, sizeof(msg), "帧 #%d 解出 %d 字节（RS纠%d，抹除%d，失败块%d）",
                 (int)seq, (int)len, (int)corrected, (int)erased, (int)fail_blocks);
        rx_log(rx, msg);
    }
    if (rx->text_cb) rx->text_cb(block + OFDM_PKT_HEADER_LEN, len, rx->user);
}

// ---------------- 创建/复位/销毁 ----------------

OFDM_RX *ofdm_rx_create(ofdm_alloc_fn alloc, ofdm_free_fn dealloc,
                        ofdm_rx_text_cb text_cb, ofdm_rx_log_cb log_cb, void *user) {
    if (!s_tables_ready) return NULL; // 须先 ofdm_tables_init（寻呼机入口已初始化）
    if (!alloc || !dealloc) return NULL;

    OFDM_RX *rx = (OFDM_RX *)alloc(sizeof(OFDM_RX));
    if (!rx) return NULL;
    memset(rx, 0, sizeof(OFDM_RX));
    rx->alloc = alloc; rx->dealloc = dealloc;
    rx->text_cb = text_cb; rx->log_cb = log_cb; rx->user = user;

    rx->det_buf = (float *)alloc(DET_RING_CAP * sizeof(float));
    rx->buf_buf = (float *)alloc(BUF_RING_CAP * sizeof(float));
    if (!rx->det_buf || !rx->buf_buf) {
        ofdm_rx_destroy(rx);
        return NULL;
    }
    rx->det_ring.buf = rx->det_buf; rx->det_ring.mask = DET_RING_CAP - 1;
    rx->buf_ring.buf = rx->buf_buf; rx->buf_ring.mask = BUF_RING_CAP - 1;
    ofdm_rx_reset(rx);
    return rx;
}

void ofdm_rx_reset(OFDM_RX *rx) {
    if (!rx) return;
    rx->state = RX_STATE_SYNC;
    ring_clear(&rx->det_ring);
    ring_clear(&rx->buf_ring);
    rx->stall_count = 0;
    rx->feed_abs = 0;
    rx->searched_upto = 0;
    rx->best_metric = -1.0f;
    rx->best_offset = 0;
    rx->frame_slot = 0;
    rx->frame_data_idx = 0;
    rx->pkt_len = 0;
    rx->last_seq = -1;
    rx->chan_valid = 0;
    rx->trA_valid = 0;
    rx->cfo_a = 0.0f; rx->sfo_b = 0.0f;
    rx->fit_res = 0.0f;
    rx->sfo_b_ema = 0.0f;
    rx->sfo_enable_streak = 0;
    rx->cfo_ref_slot = 2;
    rx->tau_ref = 0;
    rx->trainA_fbd = 0;
    rx->train_miss = 0;
    rx->frames_ok = 0; rx->frames_bad = 0;
    rx->rs_corrected = 0; rx->rs_fail_blocks = 0;
    rx->cfo_hz = 0.0f; rx->sfo_ppm = 0.0f;
    rx->viz_seq = 0;
    rx->hp_x1 = 0.0f; rx->hp_x2 = 0.0f; rx->hp_y1 = 0.0f; rx->hp_y2 = 0.0f;
}

void ofdm_rx_destroy(OFDM_RX *rx) {
    if (!rx) return;
    if (rx->det_buf) rx->dealloc(rx->det_buf);
    if (rx->buf_buf) rx->dealloc(rx->buf_buf);
    rx->dealloc(rx);
}

void ofdm_rx_get_stat(OFDM_RX *rx, OFDM_RX_Stat *stat) {
    if (!rx || !stat) return;
    stat->locked = (rx->state == RX_STATE_FRAME) ? 1 : 0;
    stat->frames_ok = rx->frames_ok;
    stat->frames_bad = rx->frames_bad;
    stat->rs_corrected = rx->rs_corrected;
    stat->rs_fail_blocks = rx->rs_fail_blocks;
    stat->cfo_hz = rx->cfo_hz;
    stat->sfo_ppm = rx->sfo_ppm;
    stat->sc_metric = (rx->state == RX_STATE_SYNC) ? rx->best_metric : -1.0f;
    stat->buf_len = rx->buf_ring.len;
    stat->det_len = rx->det_ring.len;
    stat->feed_abs = rx->feed_abs;
}

int32_t ofdm_rx_get_viz(OFDM_RX *rx, const float **wave,
                        const float **iq_i, const float **iq_q, uint32_t *viz_seq) {
    if (!rx || rx->viz_seq == 0) return -1;
    if (wave) *wave = rx->viz_wave;
    if (iq_i) *iq_i = rx->viz_iq_i;
    if (iq_q) *iq_q = rx->viz_iq_q;
    if (viz_seq) *viz_seq = rx->viz_seq;
    return 0;
}

// ---------------- 主喂入入口 ----------------
// 一帧采样块（任意长度）；全程环形缓冲 O(1) 写入/丢弃，稳态零堆分配。
// 背压策略：环形缓冲定容，写入满时覆盖最旧采样；UI 繁忙导致的丢样会使
// 训练符号失锁而自动回到同步搜索重捕获，不会崩溃或错位解调。
// 注意：单次写入超过 SC 检测历史环容量会把尚未搜索的采样挤出，
// 故统一由 ofdm_rx_feed 分片（≤1 槽）驱动本函数，调用方无需关心块长。
static void ofdm_rx_feed_slice(OFDM_RX *rx, const float *frame, uint32_t frame_len) {
    if (!rx || !frame || frame_len == 0) return;

    rx->feed_abs += frame_len;
    ring_write(&rx->det_ring, frame, frame_len);
    ring_write(&rx->buf_ring, frame, frame_len);

    // ---- 流式粗同步：SC前导检测（增量滑动自相关，每采样仅约12次乘加） ----
    if (rx->state == RX_STATE_SYNC) {
        const float *rb = rx->det_ring.buf;
        uint32_t rm = rx->det_ring.mask, rs = rx->det_ring.start;
        uint64_t base = rx->feed_abs - rx->det_ring.len;
        uint64_t d_begin = (rx->searched_upto > base) ? rx->searched_upto : base;
        if (rx->feed_abs >= 2 * SC_HALF_LEN) {
            uint64_t d_max = rx->feed_abs - 2 * SC_HALF_LEN;
            if (d_max >= d_begin) {
                uint32_t i0 = rs + (uint32_t)(d_begin - base);
                float P = 0.0f, R = 0.0f, E1 = 0.0f;
                for (int32_t n = 0; n < SC_HALF_LEN; n++) {
                    float x1 = rb[(i0 + n) & rm], x2 = rb[(i0 + n + SC_HALF_LEN) & rm];
                    P += x1 * x2; R += x2 * x2; E1 += x1 * x1;
                }
                for (uint64_t d = d_begin; d <= d_max; d += 4) {
                    float metric = (P * P) / (E1 * R + 1e-12f);
                    if (metric > rx->best_metric) { rx->best_metric = metric; rx->best_offset = d; }
                    uint64_t remain = d_max - d;
                    int32_t steps = (remain < 4) ? (int32_t)remain : 4;
                    uint32_t j0 = rs + (uint32_t)(d - base);
                    for (int32_t s = 0; s < steps; s++) {
                        uint32_t j = j0 + s;
                        float x0 = rb[j & rm], x1 = rb[(j + SC_HALF_LEN) & rm], x2 = rb[(j + 2 * SC_HALF_LEN) & rm];
                        P += -x0 * x1 + x1 * x2;
                        R += -x1 * x1 + x2 * x2;
                        E1 += -x0 * x0 + x1 * x1;
                    }
                }
                rx->searched_upto = d_max + 1;
            }
        }
        if (rx->best_metric >= SC_DETECT_THRESHOLD &&
            rx->searched_upto > rx->best_offset + SC_HALF_LEN) {
            uint64_t base2 = rx->feed_abs - rx->det_ring.len;
            uint64_t t_offset = rx->best_offset;
            float validate_metric = -1.0f;
            uint64_t d0 = (rx->best_offset > 360 && rx->best_offset - 360 > base2) ? rx->best_offset - 360 : base2;
            uint64_t d1 = rx->best_offset + 360;
            uint64_t d1_max = rx->feed_abs - 2 * SC_HALF_LEN;
            if (d1 > d1_max) d1 = d1_max;
            for (uint64_t d = d0; d <= d1; d++) {
                uint32_t j0 = rs + (uint32_t)(d - base2);
                float corr = 0.0f, e = 0.0f;
                for (int32_t t = 0; t < 2 * SC_HALF_LEN; t++) {
                    float x = rb[(j0 + t) & rm];
                    corr += x * SC_PREAMBLE[t];
                    e += x * x;
                }
                float m = (corr * corr) / (e * SC_ENERGY + 1e-12f);
                if (m > validate_metric) { validate_metric = m; t_offset = d; }
            }
            if (validate_metric < SC_VALIDATE_THRESHOLD) {
                // 自相关触发但模板验证未过：近似命中（采样率偏差/失真）排查的关键证据，
                // 仅记录接近阈值的近似命中，避免刷屏
                if (validate_metric >= 0.15f) {
                    char msg[96];
                    snprintf(msg, sizeof(msg), "SC 候选验证未过（自相关 %.3f，模板 %.3f @%llu）",
                             (double)rx->best_metric, (double)validate_metric,
                             (unsigned long long)t_offset);
                    rx_log(rx, msg);
                }
                rx->best_metric = -1.0f;
            }
            else {
                uint64_t avail_base = rx->feed_abs - rx->buf_ring.len;
                uint64_t target = t_offset + OFDM_GROSS_SYMBOL_LENGTH - FINE_SEARCH_LEN;
                if (target > avail_base) ring_drop(&rx->buf_ring, (uint32_t)(target - avail_base));
                rx->state = RX_STATE_FRAME;
                rx->frame_slot = 1;
                rx->frame_data_idx = 0;
                rx->pkt_len = 0;
                rx->chan_valid = 0;
                rx->trA_valid = 0;
                rx->cfo_a = 0.0f; rx->sfo_b = 0.0f;
                rx->cfo_ref_slot = 2;
                rx->train_miss = 0;
                rx->stall_count = 0;
                char msg[96];
                snprintf(msg, sizeof(msg), "SC 粗同步 @%llu（精化度量 %.3f）",
                         (unsigned long long)t_offset, (double)validate_metric);
                rx_log(rx, msg);
                rx->best_metric = -1.0f;
            }
        }
        // 注：环形缓冲定容后自动覆盖最旧采样，无需手动裁剪
    }

    // ---- 帧内逐槽消费 ----
    if (rx->state != RX_STATE_FRAME) return;
    while (rx->state == RX_STATE_FRAME &&
           rx->buf_ring.len >= OFDM_GROSS_SYMBOL_LENGTH + 2 * FINE_SEARCH_LEN + LPF_GD) {
        rx->stall_count = 0;
        char type = FRAME_SCHEDULE[rx->frame_slot - 1];
        if (type == 'T') {
            // 训练槽：细同步 + 信道估计（帧头A/B另用于CFO/SFO估计）
            const float *ab = rx->buf_ring.buf;
            uint32_t am = rx->buf_ring.mask;
            uint32_t as0 = rx->buf_ring.start;
            float fbm = -1.0f;
            int32_t fbd = FINE_SEARCH_LEN;
            for (int32_t d = 0; d <= 2 * FINE_SEARCH_LEN; d++) {
                uint32_t j0 = as0 + d;
                float corr = 0.0f, energy = 0.0f;
                for (int32_t t = 0; t < OFDM_GROSS_SYMBOL_LENGTH; t++) {
                    float x = ab[(j0 + t) & am];
                    corr += x * TRAINING_SYMBOL_TIME[t];
                    energy += x * x;
                }
                float m = (corr * corr) / (energy * TRAINING_TPL_ENERGY + 1e-12f);
                if (m > fbm) { fbm = m; fbd = d; }
            }
            if (fbm < TRAINING_LOCK_THRESHOLD) {
                // 细同步失锁一律回退标称定时（±CP 内的定时偏差由循环前缀与频域均衡吸收）。
                // 强多径下与干净模板的相关峰分裂，固中止逻辑不可取；误同步候选由
                // 训练A/B信道一致性检查在槽2快速否决。
                char msg[96];
                snprintf(msg, sizeof(msg), "训练符号相关度偏低（%.3f），按标称定时继续", (double)fbm);
                rx_log(rx, msg);
                fbd = FINE_SEARCH_LEN;
            }
            else {
                rx->train_miss = 0;
            }
            // 取符号块（含滤波器余量）→ 混频+抽取+FFT
            ring_read_to(&rx->buf_ring, rx->sym_blk, (uint32_t)(fbd + OFDM_CP_LENGTH - LPF_GD), SYM_BLK_LEN);
            ring_drop(&rx->buf_ring, (uint32_t)(fbd + OFDM_GROSS_SYMBOL_LENGTH - FINE_SEARCH_LEN));
            float tr_i[OFDM_CARRIER_NUMBER], tr_q[OFDM_CARRIER_NUMBER];
            demodulate_ofdm_symbol(rx, rx->sym_blk, tr_i, tr_q);
            if (rx->frame_slot == 1) {
                memcpy(rx->trA_i, tr_i, sizeof(tr_i));
                memcpy(rx->trA_q, tr_q, sizeof(tr_q));
                rx->trA_valid = 1; // 暂存训练A，等待B配对
                rx->trainA_fbd = fbd;
            }
            else if (rx->frame_slot == 2) {
                float h1i[OFDM_CARRIER_NUMBER], h1q[OFDM_CARRIER_NUMBER];
                channel_from_training(rx->trA_i, rx->trA_q, h1i, h1q);
                channel_from_training(tr_i, tr_q, rx->chan_hi, rx->chan_hq); // 信道参考面：槽2、定时 fbd
                fit_cfo(rx, h1i, h1q, rx->chan_hi, rx->chan_hq, fbd - rx->trainA_fbd); // 显式CFO/SFO估计（扣除定时差）
                // 真伪同步判定：真前导训练对相位差为低阶斜坡（残差小），
                // 帧内伪锁定相位差随机（残差大）。在槽2快速否决伪锁定，
                // 避免白跑整帧而错过下一个真前导。阈值 0.7 rad 兼顾低信噪比真帧。
                if (rx->fit_res > 0.7f) {
                    char msg[96];
                    snprintf(msg, sizeof(msg), "训练A/B相位拟合残差 %.2f rad 过大，判定伪同步，回退重新搜索", (double)rx->fit_res);
                    rx_log(rx, msg);
                    rx->state = RX_STATE_SYNC;
                    rx->trA_valid = 0;
                    break;
                }
                // 信道估计降噪：拟合可信时，把训练A的信道估计用原始拟合相位速率
                // 对齐到槽2参考面后与B平均（不可用门控值：门控未启用时对齐不足会涂抹信道）
                if (rx->fit_res <= 0.5f) {
                    for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
                        float ph = rx->cfo_a + rx->sfo_b_raw * CARRIER_FREQS_F[c]; // 1 槽
                        float cr = cosf(ph), ci = sinf(ph);
                        float ii = h1i[c], qq = h1q[c];
                        float ai = ii * cr + qq * ci;  // H1 × e^{-jφ}（与去旋转同向）
                        float aq = qq * cr - ii * ci;
                        rx->chan_hi[c] = 0.5f * (ai + rx->chan_hi[c]);
                        rx->chan_hq[c] = 0.5f * (aq + rx->chan_hq[c]);
                    }
                }
                rx->chan_valid = 1;
                rx->cfo_ref_slot = 2;
                rx->tau_ref = fbd;
                rx->trA_valid = 0;
            }
            else {
                // 帧中训练：先把信道参考面推进到本槽（漂移项 + 定时差旋转），再融合
                float hi[OFDM_CARRIER_NUMBER], hq[OFDM_CARRIER_NUMBER];
                channel_from_training(tr_i, tr_q, hi, hq);
                int32_t dt = rx->frame_slot - rx->cfo_ref_slot;
                for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
                    float ph = (rx->cfo_a + rx->sfo_b * CARRIER_FREQS_F[c]) * dt;
                    float tq = (float)(TWO_PI * CARRIER_FREQS_F[c] * (fbd - rx->tau_ref) / OFDM_SAMPLE_RATE);
                    float cr = cosf(ph - tq), ci = sinf(ph - tq);
                    float chi = rx->chan_hi[c], chq = rx->chan_hq[c];
                    rx->chan_hi[c] = chi * cr - chq * ci;
                    rx->chan_hq[c] = chi * ci + chq * cr;
                    rx->chan_hi[c] = 0.5f * rx->chan_hi[c] + 0.5f * hi[c];
                    rx->chan_hq[c] = 0.5f * rx->chan_hq[c] + 0.5f * hq[c];
                }
                rx->cfo_ref_slot = rx->frame_slot;
                rx->tau_ref = fbd;
            }
        }
        else {
            // 数据槽：显式CFO/SFO补偿 → 散布导频相位/信道跟踪 → 均衡 → 解映射
            ring_read_to(&rx->buf_ring, rx->sym_blk,
                         (uint32_t)(FINE_SEARCH_LEN + OFDM_CP_LENGTH - LPF_GD), SYM_BLK_LEN);
            ring_drop(&rx->buf_ring, OFDM_GROSS_SYMBOL_LENGTH);
            demodulate_ofdm_symbol(rx, rx->sym_blk, rx->iq_i, rx->iq_q);
            derotate_iq(rx, rx->iq_i, rx->iq_q, rx->frame_slot - rx->cfo_ref_slot);
            if (rx->chan_valid) {
                pilot_track(rx, rx->iq_i, rx->iq_q, rx->frame_data_idx);
                equalize(rx, rx->iq_i, rx->iq_q);
            }
            // 可视化钩子：暂存本符号时域块与IQ（供频谱/星座图等后续可视化使用）
            memcpy(rx->viz_wave, rx->sym_blk + LPF_GD, OFDM_SYMBOL_LENGTH * sizeof(float));
            memcpy(rx->viz_iq_i, rx->iq_i, sizeof(rx->iq_i));
            memcpy(rx->viz_iq_q, rx->iq_q, sizeof(rx->iq_q));
            rx->viz_seq++;
            // 提取数据子载波（剔除本符号的散布导频点），同时记录逐载波置信度
            //（判决边际 × 信道增益 ≈ 信噪比正比量，供帧尾 RS 抹除选择）
            int32_t po = (PILOT_SHIFT * rx->frame_data_idx) % PILOT_SPACING;
            float di[OFDM_DATA_CARRIERS], dq[OFDM_DATA_CARRIERS], dc[OFDM_DATA_CARRIERS];
            int32_t n = 0;
            for (int32_t c = 0; c < OFDM_CARRIER_NUMBER; c++) {
                if (c % PILOT_SPACING != po) {
                    di[n] = rx->iq_i[c]; dq[n] = rx->iq_q[c];
                    if (rx->chan_valid) {
                        float m = (fabsf(rx->iq_i[c]) < fabsf(rx->iq_q[c])) ? fabsf(rx->iq_i[c]) : fabsf(rx->iq_q[c]);
                        dc[n] = m * sqrtf(rx->chan_hi[c] * rx->chan_hi[c] + rx->chan_hq[c] * rx->chan_hq[c]);
                    }
                    else dc[n] = 1.0f; // 无有效信道：置信度齐一（不触发抹除）
                    n++;
                }
            }
            uint8_t frame_bytes[OFDM_BYTES_PER_SYMBOL];
            qam4_decoding(di, dq, frame_bytes);
            for (int32_t i = 0; i < OFDM_BYTES_PER_SYMBOL; i++) {
                if (rx->pkt_len < OFDM_PKT_WIRE_LEN) {
                    // 字节置信度 = 组成它的 4 个 QAM4 符号置信度的最小值
                    float cf = dc[i * 4];
                    for (int32_t k = 1; k < 4; k++) if (dc[i * 4 + k] < cf) cf = dc[i * 4 + k];
                    rx->pkt_conf[rx->pkt_len] = cf;
                    rx->pkt_buf[rx->pkt_len++] = frame_bytes[i];
                }
            }
            rx->frame_data_idx++;
        }
        rx->frame_slot++;
        if (rx->frame_slot > FRAME_SCHEDULE_LEN) {
            // 帧尾：解包，随后回到同步搜索，等待下一帧（定期重启同步/冷启动切入）
            if (rx->pkt_len == OFDM_PKT_WIRE_LEN) decode_frame(rx, rx->pkt_buf, rx->pkt_conf);
            else rx_log(rx, "帧长度异常，丢弃");
            rx->pkt_len = 0;
            rx->state = RX_STATE_SYNC;
        }
    }
    // 停滞看门狗：帧接收中途信号长时间中断则回同步搜索
    if (rx->state == RX_STATE_FRAME) {
        rx->stall_count++;
        if (rx->stall_count > STALL_WATCHDOG) {
            rx_log(rx, "帧接收停滞，回到同步搜索");
            rx->state = RX_STATE_SYNC;
            rx->stall_count = 0;
        }
    }
}

// 主喂入入口（公开 API）：任意块长，内部按 ≤1 槽（GROSS_SYMBOL_LENGTH）分片驱动。
// 分片的原因：SC 检测历史环（DET_RING_CAP=16384）容量有限，若单次写入过大，
// 尚未搜索的 SC 前导会被同一次写入驱逐出历史环，导致永远无法粗同步。
//
// 前置高通（二阶 Butterworth HPF，fc≈1500Hz @48kHz）：
// - 抑制 PDM 硅麦/ADC 原始输出的显著 DC 与低频分量（实测设备安静时 <2kHz
//   存在强 1/f 低频轰鸣：房间声学+电气耦合）。DC/低频对归一化相关检测是致命的：
//   同时抬高 SC 自相关的 P/E1/R 三项（虚至 ~0.9），又抬高模板验证的能量项；
//   弱信号时还稀释检测灵敏度。
// - 频率响应：2500Hz 信号带缘处仅 -0.5dB（频带已上移至 2500~5500Hz），
//   1000Hz -8dB，500Hz -19dB，100Hz -39dB。
// 系数（RBJ biquad，K=tan(π·1500/48000)）：
#define OFDM_HPF_B0 (0.870322f)
#define OFDM_HPF_B1 (-1.740644f)
#define OFDM_HPF_B2 (0.870322f)
#define OFDM_HPF_A1 (-1.723415f)
#define OFDM_HPF_A2 (0.757609f)
void ofdm_rx_feed(OFDM_RX *rx, const float *frame, uint32_t frame_len) {
    if (!rx || !frame || frame_len == 0) return;
    while (frame_len > 0) {
        uint32_t n = (frame_len > OFDM_GROSS_SYMBOL_LENGTH) ? OFDM_GROSS_SYMBOL_LENGTH : frame_len;
        float x1 = rx->hp_x1, x2 = rx->hp_x2, y1 = rx->hp_y1, y2 = rx->hp_y2;
        for (uint32_t i = 0; i < n; i++) {
            float x = frame[i];
            float y = OFDM_HPF_B0 * x + OFDM_HPF_B1 * x1 + OFDM_HPF_B2 * x2
                    - OFDM_HPF_A1 * y1 - OFDM_HPF_A2 * y2;
            x2 = x1; x1 = x; y2 = y1; y1 = y;
            rx->dc_scratch[i] = y;
        }
        rx->hp_x1 = x1; rx->hp_x2 = x2; rx->hp_y1 = y1; rx->hp_y2 = y2;
        ofdm_rx_feed_slice(rx, rx->dc_scratch, n);
        frame += n;
        frame_len -= n;
    }
}
