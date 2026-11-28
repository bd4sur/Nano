////////////////////////////////////////////////////////////////////////////
//  Native Library : Bird —— 设计说明见 am_native_Bird.h
//
//  分层（与 (native Music)/(native Audio) 同构：重活在 C，界面在 Scheme）：
//
//      Scheme App ──(Bird.select/play/pump/voices/…)
//                        │
//                        ├── 合成内核：Mindlin–Laje 鸣管 ODE + 标定表反解 β
//                        │             + 包络/颤音/调幅 + 声道带通 + 双声门/谐波
//                        ├── 乐句调度：按**采样计数**推进音节与休止（不依赖 UI 帧）
//                        └── hal_audio_out（块式喂入；无设备则静音走带）
//
//  本文件没有一行界面代码，也没有平台分支：桌面 / 浏览器 / 掌机行为一致。
//
//  算法出处：https://github.com/emwstudio/esp32-perch（ESP32 上的鸟叫合成器，
//  移植自 sha5b/M5Stack-Lyrebird 的 AudioWorklet 合成器）。**只参照算法**：
//  标定表由 infer/tools/gen_bird_calib.py 自测，鸟种参数是本节末尾的手工编写数据。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "am_native_Bird.h"

// 宿主本地库数量上限守卫（与 am_native_Audio.c 同一约定：忘加 -DAM_NATIVE_MAX_LIBS=32
// 时在**编译期**炸掉，而不是运行期静默「未找到 native 函数」）。
#if !defined(AM_NATIVE_MAX_LIBS) || (AM_NATIVE_MAX_LIBS < 20)
#error "需要 -DAM_NATIVE_MAX_LIBS=32（宿主本地库数量上限，见 am_native_Audio.c 文件头；本库是第 20 个）"
#endif

#include "bird_calib.h"

#include "hal_audio_out.h"
#include "platform.h"

// ===========================================================================
// 参数
// ===========================================================================

#define BIRD_SR            (22050)   // 合成采样率（hal_audio_out 负责重采样到设备速率）
#define BIRD_CTRL          (32)      // 控制率块长（音高轮廓 / 声道调谐 / 包络重锚）
#define BIRD_CTRL_MASK     (BIRD_CTRL - 1)
#define BIRD_MAX_VOICES    (8)       // 同时发声的声部数（4 只鸟 × 双声门共用池）
#define BIRD_MAX_PERF      (4)       // 表演者（"个体"）上限
#define BIRD_BLOCK         (512)     // 每次喂给音频设备的块长（23.2ms）
#define BIRD_MAX_CONTOUR   (8)       // 单个音节的最大轮廓点数
#define BIRD_MAX_SYLL      (6)       // 单个鸟种的最大音节数
#define BIRD_MAX_SONG      (3)       // 单个鸟种的最大曲目数
#define BIRD_MAX_SYL_SONG  (6)       // 单首曲目的最大音节数

// 固定步长比：每个声部按 gamma 取整"每采样积分步数"，使 γ·dt ≈ BIRD_REF_STEP。
// 标定表就是在这个步长比下测的（见 bird_calib.h 与 tools/gen_bird_calib.py），
// 因此同一张表对所有音高都成立——这是与参考实现的主要工程差异。
#define BIRD_REF_STEP      (0.05f)
#define BIRD_OS_MIN        (1)
#define BIRD_OS_MAX        (96)

#define BIRD_GAMMA_MIN     (1200.0f)
#define BIRD_GAMMA_MAX     (150000.0f)
// phi = f0/gamma 在 beta=1 时的取值（本仓标定表实测 0.17031，文献 0.17061）。
// 只用于"由目标音高猜 gamma"：gamma 决定的是**时间尺度**，取值只要让中位音高
// 落在表的中段即可（beta 会在每个采样点上被反解回目标音高）。
#define BIRD_PHI_AT_BETA1  (0.17031f)

#define BIRD_DETUNE_TWO    (1.0040348f)   // +7 音分：第二声门
#define BIRD_LOG_DETUNE    (0.0040267f)
#define BIRD_DEFAULT_ATK   (0.008f)       // 起振时间默认值（秒）
#define BIRD_RELEASE_S     (0.006f)       // 收尾淡出（秒）
#define BIRD_DECAY_DIV     (2.5f)         // 衰减时常数 = 音节时长 / 此值
#define BIRD_HOLD_FRAC     (0.75f)        // 保持型音节：衰减从 75% 处开始
#define BIRD_SOURCE_GAIN   (0.35f)
#define BIRD_ODE_X0        (1.0e-4f)

// 混合软限幅：tanh(drive·x)·trim（小信号增益 = drive·trim，天花板 = trim）
//   drive 里含一个 BIRD_OUT_GAIN：单声部的声源被 amp 归一化到 ~0.35，四声部叠加后
//   才到"响"的量级，故统一抬 2.6 倍（约 +8dB）再进软限幅——这样独唱与齐鸣都在
//   舒适区，重叠时由 tanh 柔和压住，而不是靠数字钳位削出方波。
#define BIRD_MIX_DRIVE     (1.28f)
#define BIRD_MIX_TRIM      (0.95f)
#define BIRD_OUT_GAIN      (2.6f)

// 音色类别（纯音 / 簧音 / 沙哑）：{增益, 带通 Q}
static const float BIRD_TIMBRE_GAIN[3] = { 1.0f, 1.25f, 1.75f };
static const float BIRD_TIMBRE_Q[3]    = { 9.0f, 5.0f,  3.0f };

// 音量域 0~128（与音乐盒同域），映射到 HAL 的 0~255
#define BIRD_VOL_MAX       (128)
#define BIRD_VOL_DEFAULT   (96)

// ===========================================================================
// 鸟种数据（手工编写；音高单位为 Hz，时间为秒）
//
//  每个音节的参数含义（与参考实现的 SyllableData 一一对应）：
//    dur      时长（秒）                  level   气流强度 0~1（→ 气压 α，决定音色亮度）
//    timbre   0 纯音 / 1 簧音 / 2 沙哑     hold    1=保持后衰减（0=起振后即衰减）
//    two      1=双声门（第二条 +7 音分的失谐声带，声音更"厚"）
//    attack   起振时间（秒；0=用默认 8ms）
//    harmonic 倍频支路增益（0=关；黄鹂/乌鸫这类"有泛音"的鸣叫靠它）
//    am/amDepth  调幅（Hz / 深度）：燕子的"咝咝"、喜鹊的"咔咔"、麻雀的"啾"
//    vibRate/vibDepth  颤音（Hz / 相对音高的对数深度）
//    contour  音高轮廓（时间比例 0~1 → Hz）
// ===========================================================================

typedef struct { float t; float hz; } BirdContourPoint;

typedef struct {
    float dur;
    float level;
    uint8_t timbre;
    uint8_t hold;
    uint8_t two;
    float attack;
    float harmonic;
    float am;
    float amDepth;
    float vibRate;
    float vibDepth;
    uint8_t nContour;
    const BirdContourPoint *contour;
} BirdSyllable;

typedef struct {
    uint8_t n;
    const uint8_t *syl;
    const float *gaps;
} BirdSong;

typedef struct {
    const char *name;      // 中文名（UTF-8）
    const char *latin;     // 拉丁学名
    const char *mnemonic;  // 拟声词
    uint8_t nSyll;
    const BirdSyllable *syll;
    uint8_t nSongs;
    const BirdSong *songs;
} BirdSpecies;

// ---- 1. 大山雀（Parus major）"叽叽—嘿"：两音一组的金属哨音 ------------------
static const BirdContourPoint ct_tit_hi[]  = { { 0.0f, 4000.0f }, { 0.35f, 4750.0f }, { 1.0f, 4300.0f } };
static const BirdContourPoint ct_tit_lo[]  = { { 0.0f, 3650.0f }, { 0.50f, 3350.0f }, { 1.0f, 3050.0f } };
static const BirdContourPoint ct_tit_tr[]  = { { 0.0f, 4450.0f }, { 0.60f, 4200.0f }, { 1.0f, 4050.0f } };
static const BirdSyllable syl_tit[] = {
    { 0.10f, 0.55f, 1, 0, 1, 0.006f, 0.00f,  0.0f, 0.00f, 0.0f, 0.000f, 3, ct_tit_hi },
    { 0.13f, 0.60f, 1, 0, 1, 0.006f, 0.00f,  0.0f, 0.00f, 0.0f, 0.000f, 3, ct_tit_lo },
    { 0.18f, 0.50f, 1, 0, 0, 0.010f, 0.00f, 22.0f, 0.45f, 0.0f, 0.000f, 3, ct_tit_tr },
};
static const uint8_t song_tit_0[] = { 0, 1 };
static const float   gap_tit_0[]  = { 0.07f };
static const uint8_t song_tit_1[] = { 2, 0, 1 };
static const float   gap_tit_1[]  = { 0.06f, 0.05f };
static const BirdSong songs_tit[] = {
    { 2, song_tit_0, gap_tit_0 },
    { 3, song_tit_1, gap_tit_1 },
};

// ---- 2. 乌鸫（Turdus merula）"笛—笛—啾"：圆润的笛音，泛音丰富 --------------
static const BirdContourPoint ct_blk_a[] = { { 0.0f, 1200.0f }, { 0.30f, 2100.0f }, { 0.75f, 2600.0f }, { 1.0f, 1900.0f } };
static const BirdContourPoint ct_blk_b[] = { { 0.0f, 2400.0f }, { 0.50f, 1800.0f }, { 1.0f, 1500.0f } };
static const BirdContourPoint ct_blk_c[] = { { 0.0f, 3000.0f }, { 0.55f, 2600.0f }, { 1.0f, 2250.0f } };
static const BirdSyllable syl_blk[] = {
    { 0.28f, 0.45f, 0, 1, 1, 0.030f, 0.55f,  0.0f, 0.00f,  0.0f, 0.000f, 4, ct_blk_a },
    { 0.22f, 0.42f, 0, 1, 0, 0.025f, 0.50f,  0.0f, 0.00f,  0.0f, 0.000f, 3, ct_blk_b },
    { 0.12f, 0.50f, 1, 0, 0, 0.008f, 0.10f,  0.0f, 0.00f, 12.0f, 0.020f, 3, ct_blk_c },
};
static const uint8_t song_blk_0[] = { 0, 1 };
static const float   gap_blk_0[]  = { 0.16f };
static const uint8_t song_blk_1[] = { 0, 0, 2 };
static const float   gap_blk_1[]  = { 0.20f, 0.12f };
static const BirdSong songs_blk[] = {
    { 2, song_blk_0, gap_blk_0 },
    { 3, song_blk_1, gap_blk_1 },
};

// ---- 3. 四声杜鹃（Cuculus micropterus）"快快割麦"：低沉的四声 --------------
static const BirdContourPoint ct_cuk_a[] = { { 0.0f, 760.0f }, { 0.40f, 720.0f }, { 1.0f, 690.0f } };
static const BirdContourPoint ct_cuk_b[] = { { 0.0f, 640.0f }, { 0.50f, 615.0f }, { 1.0f, 590.0f } };
static const BirdSyllable syl_cuk[] = {
    { 0.16f, 0.50f, 0, 1, 0, 0.020f, 0.30f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_cuk_a },
    { 0.24f, 0.45f, 0, 1, 0, 0.020f, 0.30f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_cuk_b },
};
static const uint8_t song_cuk_0[] = { 0, 0, 1 };
static const float   gap_cuk_0[]  = { 0.26f, 0.10f };
static const uint8_t song_cuk_1[] = { 0, 1 };
static const float   gap_cuk_1[]  = { 0.09f };
static const BirdSong songs_cuk[] = {
    { 3, song_cuk_0, gap_cuk_0 },
    { 2, song_cuk_1, gap_cuk_1 },
};

// ---- 4. 珠颈斑鸠（Spilopelia chinensis）"咕—咕—咕"：三个低沉的长音 --------
static const BirdContourPoint ct_dov_a[] = { { 0.0f, 490.0f }, { 0.50f, 445.0f }, { 1.0f, 430.0f } };
static const BirdContourPoint ct_dov_b[] = { { 0.0f, 425.0f }, { 0.50f, 405.0f }, { 1.0f, 395.0f } };
static const BirdSyllable syl_dov[] = {
    { 0.22f, 0.40f, 0, 1, 0, 0.035f, 0.25f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_dov_a },
    { 0.30f, 0.42f, 0, 1, 0, 0.040f, 0.30f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_dov_b },
};
static const uint8_t song_dov_0[] = { 0, 0, 1 };
static const float   gap_dov_0[]  = { 0.16f, 0.14f };
static const uint8_t song_dov_1[] = { 0, 1 };
static const float   gap_dov_1[]  = { 0.22f };
static const BirdSong songs_dov[] = {
    { 3, song_dov_0, gap_dov_0 },
    { 2, song_dov_1, gap_dov_1 },
};

// ---- 5. 麻雀（Passer montanus）"啾啾"：短促、略带沙哑的一串 -----------------
static const BirdContourPoint ct_spw_a[] = { { 0.0f, 3600.0f }, { 0.70f, 4100.0f }, { 1.0f, 4250.0f } };
static const BirdContourPoint ct_spw_b[] = { { 0.0f, 4400.0f }, { 0.60f, 4000.0f }, { 1.0f, 3750.0f } };
static const BirdContourPoint ct_spw_c[] = { { 0.0f, 3800.0f }, { 1.0f, 3550.0f } };
static const BirdSyllable syl_spw[] = {
    { 0.07f, 0.60f, 2, 0, 1, 0.004f, 0.00f,  0.0f, 0.00f, 0.0f, 0.000f, 3, ct_spw_a },
    { 0.06f, 0.55f, 2, 0, 0, 0.004f, 0.00f,  0.0f, 0.00f, 0.0f, 0.000f, 3, ct_spw_b },
    { 0.10f, 0.50f, 1, 0, 0, 0.006f, 0.00f, 30.0f, 0.30f, 0.0f, 0.000f, 2, ct_spw_c },
};
static const uint8_t song_spw_0[] = { 0, 1 };
static const float   gap_spw_0[]  = { 0.05f };
static const uint8_t song_spw_1[] = { 0, 0, 1, 2 };
static const float   gap_spw_1[]  = { 0.04f, 0.04f, 0.06f };
static const BirdSong songs_spw[] = {
    { 2, song_spw_0, gap_spw_0 },
    { 4, song_spw_1, gap_spw_1 },
};

// ---- 6. 家燕（Hirundo rustica）"咝—啾啾"：快速调幅的沙沙声 -----------------
static const BirdContourPoint ct_swl_a[] = { { 0.0f, 4800.0f }, { 0.40f, 5600.0f }, { 1.0f, 4250.0f } };
static const BirdContourPoint ct_swl_b[] = { { 0.0f, 3000.0f }, { 0.50f, 3500.0f }, { 1.0f, 3850.0f } };
static const BirdSyllable syl_swl[] = {
    { 0.26f, 0.35f, 2, 0, 0, 0.012f, 0.00f, 26.0f, 0.55f, 0.0f, 0.000f, 3, ct_swl_a },
    { 0.08f, 0.45f, 1, 0, 0, 0.005f, 0.00f,  0.0f, 0.00f, 0.0f, 0.000f, 3, ct_swl_b },
};
static const uint8_t song_swl_0[] = { 0, 1 };
static const float   gap_swl_0[]  = { 0.08f };
static const uint8_t song_swl_1[] = { 1, 1, 0 };
static const float   gap_swl_1[]  = { 0.07f, 0.07f };
static const BirdSong songs_swl[] = {
    { 2, song_swl_0, gap_swl_0 },
    { 3, song_swl_1, gap_swl_1 },
};

// ---- 7. 白头鹎（Pycnonotus sinensis）"滴哩—滴哩"：明亮的小调 --------------
static const BirdContourPoint ct_bul_a[] = { { 0.0f, 2600.0f }, { 0.50f, 3400.0f }, { 1.0f, 2900.0f } };
static const BirdContourPoint ct_bul_b[] = { { 0.0f, 3200.0f }, { 0.55f, 2700.0f }, { 1.0f, 2400.0f } };
static const BirdContourPoint ct_bul_c[] = { { 0.0f, 2800.0f }, { 0.60f, 3150.0f }, { 1.0f, 2600.0f } };
static const BirdSyllable syl_bul[] = {
    { 0.11f, 0.45f, 0, 0, 0, 0.008f, 0.35f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_bul_a },
    { 0.09f, 0.42f, 0, 0, 0, 0.008f, 0.30f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_bul_b },
    { 0.16f, 0.40f, 1, 0, 0, 0.012f, 0.15f, 0.0f, 0.00f, 9.0f, 0.015f, 3, ct_bul_c },
};
static const uint8_t song_bul_0[] = { 0, 1, 0, 1 };
static const float   gap_bul_0[]  = { 0.07f, 0.08f, 0.07f };
static const uint8_t song_bul_1[] = { 2, 0, 2 };
static const float   gap_bul_1[]  = { 0.13f, 0.13f };
static const BirdSong songs_bul[] = {
    { 4, song_bul_0, gap_bul_0 },
    { 3, song_bul_1, gap_bul_1 },
};

// ---- 8. 喜鹊（Pica pica）"咔咔咔"：干硬的快速调幅 --------------
static const BirdContourPoint ct_mag_a[] = { { 0.0f, 1750.0f }, { 1.0f, 1500.0f } };
static const BirdContourPoint ct_mag_b[] = { { 0.0f, 2050.0f }, { 1.0f, 1600.0f } };
static const BirdSyllable syl_mag[] = {
    { 0.055f, 0.75f, 2, 0, 0, 0.003f, 0.00f, 18.0f, 0.65f, 0.0f, 0.000f, 2, ct_mag_a },
    { 0.070f, 0.70f, 2, 0, 0, 0.003f, 0.00f, 16.0f, 0.60f, 0.0f, 0.000f, 2, ct_mag_b },
};
static const uint8_t song_mag_0[] = { 0, 0, 0, 0 };
static const float   gap_mag_0[]  = { 0.045f, 0.045f, 0.045f };
static const uint8_t song_mag_1[] = { 1, 1, 1 };
static const float   gap_mag_1[]  = { 0.060f, 0.060f };
static const BirdSong songs_mag[] = {
    { 4, song_mag_0, gap_mag_0 },
    { 3, song_mag_1, gap_mag_1 },
};

// ---- 9. 黄鹂（Oriolus chinensis）"嘀—嘀—嘀"：圆笛音，泛音最足 ---------------
static const BirdContourPoint ct_ori_a[] = { { 0.0f, 2200.0f }, { 0.40f, 3000.0f }, { 1.0f, 2600.0f } };
static const BirdContourPoint ct_ori_b[] = { { 0.0f, 3400.0f }, { 0.50f, 3050.0f }, { 1.0f, 2800.0f } };
static const BirdContourPoint ct_ori_c[] = { { 0.0f, 2500.0f }, { 0.50f, 3200.0f }, { 1.0f, 2100.0f } };
static const BirdSyllable syl_ori[] = {
    { 0.20f, 0.40f, 0, 1, 1, 0.020f, 0.60f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_ori_a },
    { 0.14f, 0.38f, 0, 1, 0, 0.015f, 0.55f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_ori_b },
    { 0.34f, 0.36f, 0, 1, 0, 0.030f, 0.50f, 0.0f, 0.00f, 5.0f, 0.010f, 3, ct_ori_c },
};
static const uint8_t song_ori_0[] = { 0, 1 };
static const float   gap_ori_0[]  = { 0.12f };
static const uint8_t song_ori_1[] = { 0, 1, 2 };
static const float   gap_ori_1[]  = { 0.10f, 0.10f };
static const BirdSong songs_ori[] = {
    { 2, song_ori_0, gap_ori_0 },
    { 3, song_ori_1, gap_ori_1 },
};

// ---- 10. 画眉（Garrulax canorus）"呜—啾哩哩"：婉转带颤音 -------------------
static const BirdContourPoint ct_hwa_a[] = { { 0.0f, 1400.0f }, { 0.50f, 2600.0f }, { 1.0f, 2200.0f } };
static const BirdContourPoint ct_hwa_b[] = { { 0.0f, 3000.0f }, { 1.0f, 2700.0f } };
static const BirdContourPoint ct_hwa_c[] = { { 0.0f, 1800.0f }, { 0.50f, 2200.0f }, { 1.0f, 2450.0f } };
static const BirdSyllable syl_hwa[] = {
    { 0.24f, 0.50f, 1, 1, 0, 0.020f, 0.40f, 0.0f, 0.00f, 8.0f, 0.020f, 3, ct_hwa_a },
    { 0.10f, 0.45f, 1, 0, 0, 0.008f, 0.10f, 20.0f, 0.30f, 0.0f, 0.000f, 2, ct_hwa_b },
    { 0.20f, 0.45f, 1, 1, 1, 0.018f, 0.20f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_hwa_c },
};
static const uint8_t song_hwa_0[] = { 0, 1, 1 };
static const float   gap_hwa_0[]  = { 0.12f, 0.08f };
static const uint8_t song_hwa_1[] = { 2, 2, 2 };
static const float   gap_hwa_1[]  = { 0.10f, 0.10f };
static const BirdSong songs_hwa[] = {
    { 3, song_hwa_0, gap_hwa_0 },
    { 3, song_hwa_1, gap_hwa_1 },
};

// ---- 11. 夜莺（Luscinia megarhynchos）：长颤音与快速的"啾" ----------------
static const BirdContourPoint ct_nig_a[] = { { 0.0f, 1600.0f }, { 0.50f, 2800.0f }, { 1.0f, 2000.0f } };
static const BirdContourPoint ct_nig_b[] = { { 0.0f, 3600.0f }, { 1.0f, 3200.0f } };
static const BirdContourPoint ct_nig_c[] = { { 0.0f, 2600.0f }, { 1.0f, 3600.0f } };
static const BirdSyllable syl_nig[] = {
    { 0.30f, 0.40f, 1, 1, 0, 0.025f, 0.45f, 0.0f, 0.00f, 6.0f, 0.020f, 3, ct_nig_a },
    { 0.16f, 0.50f, 2, 0, 0, 0.008f, 0.00f, 28.0f, 0.50f, 0.0f, 0.000f, 2, ct_nig_b },
    { 0.10f, 0.42f, 1, 0, 0, 0.006f, 0.10f, 0.0f, 0.00f, 0.0f, 0.000f, 2, ct_nig_c },
};
static const uint8_t song_nig_0[] = { 0, 2, 1 };
static const float   gap_nig_0[]  = { 0.10f, 0.06f };
static const uint8_t song_nig_1[] = { 1, 1, 1, 1 };
static const float   gap_nig_1[]  = { 0.07f, 0.07f, 0.07f };
static const BirdSong songs_nig[] = {
    { 3, song_nig_0, gap_nig_0 },
    { 4, song_nig_1, gap_nig_1 },
};

// ---- 12. 戴胜（Upupa epops）"扑—扑—扑"：低沉、软起振的三连音 --------------
static const BirdContourPoint ct_hoo_a[] = { { 0.0f, 700.0f }, { 0.35f, 600.0f }, { 1.0f, 540.0f } };
static const BirdSyllable syl_hoo[] = {
    { 0.18f, 0.40f, 0, 1, 1, 0.030f, 0.20f, 0.0f, 0.00f, 0.0f, 0.000f, 3, ct_hoo_a },
};
static const uint8_t song_hoo_0[] = { 0, 0, 0 };
static const float   gap_hoo_0[]  = { 0.22f, 0.22f };
static const uint8_t song_hoo_1[] = { 0, 0 };
static const float   gap_hoo_1[]  = { 0.30f };
static const BirdSong songs_hoo[] = {
    { 3, song_hoo_0, gap_hoo_0 },
    { 2, song_hoo_1, gap_hoo_1 },
};

#define BIRD_SPECIES(nm, lt, mn, sy, so) { nm, lt, mn, \
    (uint8_t)(sizeof(sy) / sizeof((sy)[0])), (sy), \
    (uint8_t)(sizeof(so) / sizeof((so)[0])), (so) }

static const BirdSpecies s_species[] = {
    BIRD_SPECIES("大山雀",   "Parus major",             "叽叽—嘿",   syl_tit, songs_tit),
    BIRD_SPECIES("乌鸫",     "Turdus merula",           "笛—笛—啾",  syl_blk, songs_blk),
    BIRD_SPECIES("四声杜鹃", "Cuculus micropterus",     "快快割麦",  syl_cuk, songs_cuk),
    BIRD_SPECIES("珠颈斑鸠", "Spilopelia chinensis",    "咕—咕—咕",  syl_dov, songs_dov),
    BIRD_SPECIES("麻雀",     "Passer montanus",         "啾啾啾",    syl_spw, songs_spw),
    BIRD_SPECIES("家燕",     "Hirundo rustica",         "咝—啾啾",   syl_swl, songs_swl),
    BIRD_SPECIES("白头鹎",   "Pycnonotus sinensis",     "滴哩—滴哩", syl_bul, songs_bul),
    BIRD_SPECIES("喜鹊",     "Pica pica",               "咔咔咔",    syl_mag, songs_mag),
    BIRD_SPECIES("黄鹂",     "Oriolus chinensis",       "嘀—嘀—嘀",  syl_ori, songs_ori),
    BIRD_SPECIES("画眉",     "Garrulax canorus",        "呜—啾哩哩", syl_hwa, songs_hwa),
    BIRD_SPECIES("夜莺",     "Luscinia megarhynchos",   "啾—噜噜噜", syl_nig, songs_nig),
    BIRD_SPECIES("戴胜",     "Upupa epops",             "扑—扑—扑",  syl_hoo, songs_hoo),
};

#define BIRD_SPECIES_N ((int32_t)(sizeof(s_species) / sizeof(s_species[0])))

// ===========================================================================
// 个体（voicing）与表演者
//
//  参考实现里"四只鸟"的差异（individual.ts）落到四个参数：音高倍数、时长倍数、
//  响度倍数、声像标签。这里再加"初始相位"与"乐句周期倍数"，于是同一首曲子可以
//  变成独唱 / 一唱一和 / 四只鸟轮唱。
// ===========================================================================

typedef struct {
    int32_t on;
    float pitchMul;     // 2^(音分/1200)
    float durScale;
    float levelScale;
    float gain;
    float phase;        // 0~1：相对乐句周期的初始相位
    float periodScale;  // 乐句周期 = 乐句时长 × 此值（>1 才有休止）
} BirdPart;

// 三种模式下的个体表（下标 = 模式号）
static const BirdPart s_parts[3][BIRD_MAX_PERF] = {
    {   // 0 = 独唱：一只鸟，乐句之间留一段寂静
        { 1, 1.000f, 1.000f, 1.00f, 1.00f, 0.00f, 1.60f },
        { 0, 1.000f, 1.000f, 1.00f, 1.00f, 0.00f, 1.60f },
        { 0, 1.000f, 1.000f, 1.00f, 1.00f, 0.00f, 1.60f },
        { 0, 1.000f, 1.000f, 1.00f, 1.00f, 0.00f, 1.60f },
    },
    {   // 1 = 应答：两只鸟一唱一和（第二只稍高、稍快、稍轻）
        { 1, 1.000f, 1.000f, 1.00f, 1.00f, 0.00f, 1.90f },
        { 1, 1.035f, 1.060f, 0.88f, 0.85f, 0.50f, 1.90f },
        { 0, 1.000f, 1.000f, 1.00f, 1.00f, 0.00f, 1.90f },
        { 0, 1.000f, 1.000f, 1.00f, 1.00f, 0.00f, 1.90f },
    },
    {   // 2 = 齐鸣：四只鸟错开四分之一周期轮唱
        { 1, 0.970f, 0.940f, 0.92f, 0.90f, 0.00f, 1.45f },
        { 1, 1.040f, 1.070f, 0.85f, 0.80f, 0.25f, 1.45f },
        { 1, 0.990f, 1.020f, 0.90f, 0.85f, 0.50f, 1.45f },
        { 1, 1.020f, 0.970f, 0.80f, 0.75f, 0.75f, 1.45f },
    },
};

typedef struct {
    int32_t on;
    BirdPart part;
    int32_t species;
    int32_t song;
    int32_t cursor;   // 下一个音节在曲目中的下标（0 = 乐句开头）
    int32_t timer;    // 距下一个事件的采样数（<=0 立即触发）
    uint8_t tag;
} BirdPerformer;

// ===========================================================================
// 合成内核状态（进程级单例）
// ===========================================================================

typedef struct {
    float b0, b1, b2, a1, a2;
    float x1, x2, y1, y2;
} BirdBiquad;

typedef struct {
    int32_t active;
    const BirdSyllable *syl;
    // 个体参数（起音时快照）
    float dur;
    float level;
    float harmonic;
    float amDepth;
    float gain;
    uint8_t tag;
    // 轮廓（log Hz，已乘音高倍数）
    int32_t nContour;
    float contT[BIRD_MAX_CONTOUR];
    float contLog[BIRD_MAX_CONTOUR];
    int32_t seg;
    // 时序
    int32_t pos;          // 自起音以来的采样数
    int32_t durSamples;
    int32_t envN;
    // 音高（每个采样点按乘性斜坡推进）
    float f0, logF0, f0Ratio, logF0Inc;
    int32_t pitchExact;
    float pitchVibOff, pitchVibInc;
    float gamma, invGamma;
    // 包络（增量求值）
    int32_t envPhase;
    int32_t envHold;
    int32_t envAtkN, envKneeN, envRelStart, envRelN;
    float envAtkInc, envAtkS, envKneeS, envInvTau, envDecMul, envRelInc, env;
    // 调幅 / 颤音（单位圆递推）
    int32_t hasAm, hasVib;
    float amC, amS, amRotC, amRotS, amOmega;
    float vibC, vibS, vibRotC, vibRotS, vibDepth;
    // 振荡器与滤波
    int32_t oversample;
    float dt;
    float x[2], y[2];
    BirdBiquad tract[2];
    BirdBiquad tractHarm[2];
    float srcHpX[2], srcHpY[2];
    float dcX, dcY, lpY;
} BirdVoice;

static BirdVoice   s_voices[BIRD_MAX_VOICES];
static BirdPerformer s_perf[BIRD_MAX_PERF];

static int32_t s_inited       = 0;
static int32_t s_species_sel  = 0;                 // 当前鸟种
static int32_t s_mode         = 0;                 // 0 独唱 / 1 应答 / 2 齐鸣
static int32_t s_playing      = 0;
static int32_t s_paused       = 0;
static int32_t s_audio_on     = 0;                 // audio_out 是否已 init
static int32_t s_silent       = 0;                 // 静音走带
static int32_t s_volume       = BIRD_VOL_DEFAULT;  // 0~128
static int32_t s_pitch_pct    = 0;                 // 音高微调 -50~50
static int32_t s_tempo_pct    = 0;                 // 速度微调 -50~50
static float   s_level        = 0.0f;              // 衰减峰值（0~1）
static uint64_t s_clock_ms    = 0;                 // 静音走带的虚拟时钟
static uint64_t s_feed_ms     = 0;                 // 上次成功投块的时刻（设备停滞看门狗）
static uint64_t s_retry_ms    = 0;                 // 上次尝试重开音频设备的时刻
static uint32_t s_rng         = 0x9e3779b9u;       // xorshift32（chirp 的随机微差）
static am_runtime_t *s_rt     = NULL;              // 时钟来源（宿主注入，可虚拟化）

// 一阶极点系数（由采样率决定，init 时算一次）
static float s_dcCoeff = 0.0f;
static float s_lpCoeff = 0.0f;

// alpha 网格的位置参数（从标定表推出，避免与生成脚本各写一份常量）
static float s_alpha_min  = 0.0f;
static float s_alpha_step = 1.0f;

// 块缓冲（喂给 HUD 的 int16 与合成用的 float 各一份，静态分配）
static int16_t s_out_buf[BIRD_BLOCK];
static float   s_fbuf[BIRD_BLOCK];

// ===========================================================================
// 操作数栈辅助（契约 N1/N6：弹完参数后压 1 值 + step 1 次）
// ===========================================================================

static bool bird_pop_number(am_process_t *proc, double *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (am_value_is_float(v)) { *out = (double)am_value_to_float(v); return true; }
    if (am_value_is_int(v))   { *out = (double)am_value_to_int(v);   return true; }
    if (am_value_is_uint(v))  { *out = (double)am_value_to_uint(v);  return true; }
    return false;
}

static bool bird_pop_int(am_process_t *proc, int32_t *out) {
    double f = 0.0;
    if (!bird_pop_number(proc, &f)) return false;
    if (isnan(f)) return false;
    if (f > 2147483647.0)  f = 2147483647.0;
    if (f < -2147483648.0) f = -2147483648.0;
    *out = (int32_t)f;
    return true;
}

static int32_t bird_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t bird_push_bool(am_process_t *proc, bool b) {
    return bird_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

static int32_t bird_push_null(am_process_t *proc) {
    return bird_push(proc, AM_VALUE_NULL);
}

static int32_t bird_push_int(am_process_t *proc, int32_t n) {
    return bird_push(proc, am_make_value_of_int((am_int_t)n));
}

static int32_t bird_push_number(am_process_t *proc, double x) {
    return bird_push(proc, am_make_value_of_float((am_float_t)x));
}

// 由 UTF-8 缓冲区构造 Scheme 字符串并压栈（失败压 #null）——模式同 am_native_Music.c
static int32_t bird_push_utf8(am_process_t *proc, const char *utf8) {
    size_t len;
    wchar_t *wbuf;
    am_handle_t hd;
    uint32_t wlen;

    if (utf8 == NULL) return bird_push_null(proc);
    len = strlen(utf8);
    wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (len + 1) * sizeof(wchar_t));
    if (!wbuf) return bird_push_null(proc);
    wlen = _mbstowcs(wbuf, utf8, (uint32_t)len);
    hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
    am_free(proc->heap_alloc, wbuf);
    if (hd == AM_HANDLE_NULL) return bird_push_null(proc);
    return bird_push(proc, am_make_value_of_handle(hd));
}

// 压入一个 n 元数值列表（分配失败按软失败压 #null）
static int32_t bird_push_list(am_process_t *proc, const double *vals, int32_t n) {
    am_list_t *lst = am_list_create(proc->heap_alloc, (size_t)n, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    am_handle_t lhd;
    if (lst == NULL) return bird_push_null(proc);
    for (int32_t i = 0; i < n; i++) {
        am_list_t *grown = am_list_push(proc->heap_alloc, lst,
                                        am_make_value_of_float((am_float_t)vals[i]));
        if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return bird_push_null(proc); }
        lst = grown;
    }
    lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return bird_push_null(proc); }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return bird_push_null(proc);
    }
    if (am_process_push_operand(proc, am_make_value_of_handle(lhd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// ===========================================================================
// 标定表查询（O(1)~O(log n)，无分配）
// ===========================================================================

// 在一行 alpha 上由 phi 反解 beta（phi 随 beta 单调递增；跳过低气压的非振荡区）
static inline float bird_row_beta(int32_t ai, float phi) {
    const float *ph = bird_cal_phi[ai];
    int32_t lo = 0;
    int32_t hi = BIRD_CAL_BETA_N - 1;
    while (lo < hi && ph[lo] <= 0.0f) lo++;      // 非振荡区（phi 恒 0）
    if (lo >= hi) return bird_cal_beta[hi];
    if (phi <= ph[lo]) return bird_cal_beta[lo];
    if (phi >= ph[hi]) return bird_cal_beta[hi];
    while (hi - lo > 1) {
        int32_t mid = (lo + hi) >> 1;
        if (ph[mid] <= phi) lo = mid; else hi = mid;
    }
    {
        float span = ph[hi] - ph[lo];
        float f = (span > 1e-9f) ? (phi - ph[lo]) / span : 0.0f;
        if (f < 0.0f) f = 0.0f;
        else if (f > 1.0f) f = 1.0f;
        return bird_cal_beta[lo] + (bird_cal_beta[hi] - bird_cal_beta[lo]) * f;
    }
}

// alpha 网格定位（返回行下标与行内比例）
static inline int32_t bird_alpha_pos(float alpha, float *frac) {
    float fa = (alpha - s_alpha_min) / (s_alpha_step > 1e-6f ? s_alpha_step : 1.0f);
    int32_t ai;
    if (fa < 0.0f) fa = 0.0f;
    if (fa > (float)(BIRD_CAL_ALPHA_N - 1)) fa = (float)(BIRD_CAL_ALPHA_N - 1);
    ai = (int32_t)fa;
    if (ai > BIRD_CAL_ALPHA_N - 2) ai = BIRD_CAL_ALPHA_N - 2;
    *frac = fa - (float)ai;
    return ai;
}

// 由 (alpha, phi) 反解 beta：相邻两行各反解一次，再按 alpha 线性插值
static inline float bird_beta_for(float alpha, float phi) {
    float af;
    int32_t ai = bird_alpha_pos(alpha, &af);
    float b0 = bird_row_beta(ai, phi);
    if (af <= 1e-4f) return b0;
    {
        float b1 = bird_row_beta(ai + 1, phi);
        return b0 + (b1 - b0) * af;
    }
}

// 由 (alpha, beta) 查无量纲峰峰振幅（双线性；退化时用解析兜底式）
static inline float bird_amp_for(float alpha, float beta) {
    float af, bf, a0, a1, a;
    int32_t ai = bird_alpha_pos(alpha, &af);
    int32_t lo = 0;
    int32_t hi = BIRD_CAL_BETA_N - 1;

    if (beta <= bird_cal_beta[0]) {
        lo = 0; hi = 1;
    }
    else if (beta >= bird_cal_beta[hi]) {
        lo = hi - 1;
    }
    else {
        while (hi - lo > 1) {
            int32_t mid = (lo + hi) >> 1;
            if (bird_cal_beta[mid] <= beta) lo = mid; else hi = mid;
        }
    }
    {
        float span = bird_cal_beta[hi] - bird_cal_beta[lo];
        bf = (span > 1e-9f) ? (beta - bird_cal_beta[lo]) / span : 0.0f;
        if (bf < 0.0f) bf = 0.0f;
        else if (bf > 1.0f) bf = 1.0f;
    }
    a0 = bird_cal_amp[ai][lo] + (bird_cal_amp[ai][hi] - bird_cal_amp[ai][lo]) * bf;
    a1 = bird_cal_amp[ai + 1][lo] + (bird_cal_amp[ai + 1][hi] - bird_cal_amp[ai + 1][lo]) * bf;
    a = a0 + (a1 - a0) * af;
    // 非振荡区（表里 amp≈0）会除出噪声，改用参考实现的解析兜底式
    return (a > 0.05f) ? a : (0.5f + 3.4f * alpha + 0.62f * beta);
}

// ===========================================================================
// 振荡器 / 滤波基元
// ===========================================================================

// tanh 的 7/6 阶 Padé 逼近（[-3,3] 上误差 ~1e-5，外侧饱和）
static inline float bird_tanh(float x) {
    float x2, num, den;
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    x2 = x * x;
    num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + 28.0f * x2));
    return num / den;
}

static inline void bird_biquad_reset(BirdBiquad *bq) {
    bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0.0f;
}

// 二阶带通（RBJ 常数峰值增益），用于"按 f0 跟踪的声道"
static inline void bird_biquad_bandpass(BirdBiquad *bq, float freq, float q) {
    float f = freq;
    float w0, alpha, a0;
    if (f < 40.0f) f = 40.0f;
    if (f > (float)BIRD_SR * 0.45f) f = (float)BIRD_SR * 0.45f;
    if (q < 0.3f) q = 0.3f;
    w0 = 2.0f * (float)M_PI * f / (float)BIRD_SR;
    alpha = sinf(w0) / (2.0f * q);
    a0 = 1.0f + alpha;
    bq->b0 = alpha / a0;
    bq->b1 = 0.0f;
    bq->b2 = -alpha / a0;
    bq->a1 = -2.0f * cosf(w0) / a0;
    bq->a2 = (1.0f - alpha) / a0;
}

static inline float bird_biquad_run(BirdBiquad *bq, float x) {
    float y = bq->b0 * x + bq->b1 * bq->x1 + bq->b2 * bq->x2
            - bq->a1 * bq->y1 - bq->a2 * bq->y2;
    bq->x2 = bq->x1;
    bq->x1 = x;
    bq->y2 = bq->y1;
    bq->y1 = y;
    return y;
}

// ===========================================================================
// 声部：起音
// ===========================================================================

// gamma = 时间尺度。取值让"中位音高"落在 beta≈1（表的中段）；音高倍数同时缩放它。
static float bird_gamma_for(const BirdSyllable *syl, float pitchMul) {
    float logSum = 0.0f;
    float ref;
    int32_t i;
    if (syl->nContour == 0) return 24000.0f;
    for (i = 0; i < syl->nContour; i++) {
        float hz = syl->contour[i].hz;
        logSum += logf(hz > 80.0f ? hz : 80.0f);
    }
    ref = expf(logSum / (float)syl->nContour) / BIRD_PHI_AT_BETA1;
    ref *= pitchMul;
    if (ref < BIRD_GAMMA_MIN) ref = BIRD_GAMMA_MIN;
    if (ref > BIRD_GAMMA_MAX) ref = BIRD_GAMMA_MAX;
    return ref;
}

// 轮廓 u∈[0,1] 处的 log(Hz)（轮廓以 log 存储，控制率块每块只付一次插值）
static inline float bird_contour_log(BirdVoice *v, float u) {
    if (v->nContour == 0) return 6.0866f;                 // log(440)
    if (u <= v->contT[0]) return v->contLog[0];
    if (u >= v->contT[v->nContour - 1]) return v->contLog[v->nContour - 1];
    while (v->seg < v->nContour - 2 && u > v->contT[v->seg + 1]) v->seg++;
    while (v->seg > 0 && u < v->contT[v->seg]) v->seg--;
    {
        float t0 = v->contT[v->seg];
        float t1 = v->contT[v->seg + 1];
        float span = t1 - t0;
        float frac = (span > 1e-6f) ? (u - t0) / span : 0.0f;
        return v->contLog[v->seg] + (v->contLog[v->seg + 1] - v->contLog[v->seg]) * frac;
    }
}

// 分配一个空闲声部并起音。返回 1 成功 / 0 声部耗尽。
static int32_t bird_note(const BirdSpecies *sp, int32_t sylIdx,
                         float pitchMul, float durScale, float levelScale,
                         float gain, uint8_t tag) {
    BirdVoice *v = NULL;
    const BirdSyllable *syl;
    int32_t i, n;
    float logPitch, envDur, atk, tau, vibRate, amRate;

    if (sp == NULL || sylIdx < 0 || sylIdx >= (int32_t)sp->nSyll) return 0;
    syl = &sp->syll[sylIdx];

    for (i = 0; i < BIRD_MAX_VOICES; i++) {
        if (!s_voices[i].active) { v = &s_voices[i]; break; }
    }
    if (v == NULL) return 0;                       // 声部耗尽：这一声丢掉（不抢别人的音）

    memset(v, 0, sizeof(*v));
    v->syl = syl;
    v->tag = tag;
    v->dur = syl->dur * durScale;
    if (v->dur < 0.02f) v->dur = 0.02f;
    v->level = syl->level * levelScale;
    if (v->level > 1.0f) v->level = 1.0f;
    v->harmonic = syl->harmonic;
    v->amDepth = syl->amDepth;
    v->gain = gain;

    // 轮廓：时间点直接拷贝，频率取 log 并叠加个体音高偏移
    logPitch = logf(pitchMul > 0.01f ? pitchMul : 1.0f);
    n = syl->nContour;
    if (n > BIRD_MAX_CONTOUR) n = BIRD_MAX_CONTOUR;
    v->nContour = n;
    for (i = 0; i < n; i++) {
        float hz = syl->contour[i].hz;
        v->contT[i] = syl->contour[i].t;
        v->contLog[i] = logf(hz > 1.0f ? hz : 1.0f) + logPitch;
    }
    v->seg = 0;

    // gamma：纯时间尺度，取整步数让 γ·dt 恒定（标定表才成立）
    v->gamma = bird_gamma_for(syl, pitchMul);
    {
        float os = v->gamma / ((float)BIRD_SR * BIRD_REF_STEP);
        int32_t n_os = (int32_t)(os + 0.5f);
        if (n_os < BIRD_OS_MIN) n_os = BIRD_OS_MIN;
        if (n_os > BIRD_OS_MAX) n_os = BIRD_OS_MAX;
        v->oversample = n_os;
        v->dt = 1.0f / ((float)BIRD_SR * (float)n_os);
    }
    v->invGamma = 1.0f / v->gamma;

    // 时序与包络
    v->pos = 0;
    v->durSamples = (int32_t)ceilf((v->dur + 0.02f) * (float)BIRD_SR);
    v->envN = (int32_t)lroundf(v->dur * (float)BIRD_SR);
    if (v->envN < 2) v->envN = 2;
    envDur = (float)v->envN / (float)BIRD_SR;
    atk = (syl->attack > 0.0f) ? syl->attack : BIRD_DEFAULT_ATK;
    if (atk < 1.0f / (float)BIRD_SR) atk = 1.0f / (float)BIRD_SR;
    if (atk > 0.9f * envDur) atk = 0.9f * envDur;
    v->envAtkS = atk;
    v->envAtkN = (int32_t)ceilf(atk * (float)BIRD_SR);
    if (v->envAtkN < 1) v->envAtkN = 1;
    v->envAtkInc = 1.0f / (atk * (float)BIRD_SR);
    v->envHold = (syl->hold != 0) ? 1 : 0;
    v->envKneeS = BIRD_HOLD_FRAC * envDur;
    v->envKneeN = (int32_t)floorf(v->envKneeS * (float)BIRD_SR) + 1;
    tau = envDur / BIRD_DECAY_DIV;
    if (tau < 1e-4f) tau = 1e-4f;
    v->envInvTau = 1.0f / tau;
    v->envDecMul = expf(-v->envInvTau / (float)BIRD_SR);
    {
        int32_t rel = (int32_t)lroundf(BIRD_RELEASE_S * (float)BIRD_SR);
        v->envRelN = (rel > 1 && rel < v->envN) ? rel : 0;
        v->envRelStart = v->envN - v->envRelN;
        v->envRelInc = (v->envRelN > 1) ? 1.0f / (float)(v->envRelN - 1) : 0.0f;
    }
    v->envPhase = 0;
    v->env = 0.0f;

    // 调幅 / 颤音：单位圆递推（每块重锚，长音不会累积相位误差）
    amRate = syl->am;
    v->hasAm = (amRate > 0.0f && syl->amDepth > 0.0f) ? 1 : 0;
    if (v->hasAm) {
        float w = 2.0f * (float)M_PI * amRate / (float)BIRD_SR;
        v->amOmega = w;
        v->amRotC = cosf(w);
        v->amRotS = sinf(w);
    }
    v->amC = 1.0f;
    v->amS = 0.0f;

    vibRate = syl->vibRate;
    v->hasVib = (vibRate > 0.0f && syl->vibDepth > 0.0f) ? 1 : 0;
    v->vibDepth = syl->vibDepth;
    if (v->hasVib) {
        float w = 2.0f * (float)M_PI * vibRate * (float)BIRD_CTRL / (float)BIRD_SR;
        v->vibRotC = cosf(w);
        v->vibRotS = sinf(w);
    }
    v->vibC = 1.0f;
    v->vibS = 0.0f;

    // 振荡器与滤波器初值
    v->x[0] = v->x[1] = BIRD_ODE_X0;
    v->y[0] = v->y[1] = 0.0f;
    v->srcHpX[0] = v->srcHpX[1] = 0.0f;
    v->srcHpY[0] = v->srcHpY[1] = 0.0f;
    v->dcX = v->dcY = v->lpY = 0.0f;
    bird_biquad_reset(&v->tract[0]);
    bird_biquad_reset(&v->tract[1]);
    bird_biquad_reset(&v->tractHarm[0]);
    bird_biquad_reset(&v->tractHarm[1]);

    v->logF0 = bird_contour_log(v, 0.0f);
    v->f0 = expf(v->logF0);
    v->f0Ratio = 1.0f;
    v->logF0Inc = 0.0f;
    v->pitchExact = 0;
    v->pitchVibOff = 0.0f;
    v->pitchVibInc = 0.0f;

    v->active = 1;
    return 1;
}

// 一个声部一个采样的振荡器：oversample 步欧拉，返回高通后的声源
static inline float bird_step_osc(BirdVoice *v, int32_t idx, float alpha, float beta, float amp) {
    float g = v->gamma;
    float g2 = g * g;
    float x = v->x[idx];
    float y = v->y[idx];
    float dt = v->dt;
    float aG2 = alpha * g2;
    float bG2 = beta * g2;
    float source, hp;
    int32_t s;

    for (s = 0; s < v->oversample; s++) {
        float x2 = x * x;
        float dy = -aG2 - bG2 * x - g2 * x2 * x - g * x2 * y + g2 * x2 - g * x * y;
        x += y * dt;
        y += dy * dt;
    }
    if (!(fabsf(x) < 1e6f && fabsf(y) < 1e12f)) {   // 数值爆炸：复位，避免拖死整块
        x = BIRD_ODE_X0;
        y = 0.0f;
    }
    v->x[idx] = x;
    v->y[idx] = y;

    // 除以峰峰振幅：响度跟着包络走，而不是跟着音高走
    source = (y / g) * BIRD_SOURCE_GAIN / amp;
    // 声道之前的 180Hz 一阶高通（去掉 ODE 的直流分量）
    hp = source - v->srcHpX[idx] + s_dcCoeff * v->srcHpY[idx];
    v->srcHpX[idx] = source;
    v->srcHpY[idx] = hp;
    return hp;
}

// 把 n 个采样合成到 out（累加；调用方负责清零与限幅）
static void bird_synth_add(float *out, int32_t n) {
    int32_t vi;

    for (vi = 0; vi < BIRD_MAX_VOICES; vi++) {
        BirdVoice *v = &s_voices[vi];
        const BirdSyllable *syl;
        int32_t timbre, i = 0;
        float tractQ, timbreGain, harmGain, invDur;
        int32_t twoVoices, harm;

        if (!v->active) continue;
        syl = v->syl;
        timbre = (syl->timbre < 3) ? syl->timbre : 0;
        tractQ = BIRD_TIMBRE_Q[timbre];
        timbreGain = BIRD_TIMBRE_GAIN[timbre];
        twoVoices = (syl->two != 0) ? 1 : 0;
        harm = (v->harmonic > 0.0f) ? 1 : 0;
        harmGain = v->harmonic * 0.5f;
        invDur = 1.0f / v->dur;

        while (i < n) {
            int32_t idx = v->pos;
            int32_t chunk, k;

            if (idx >= v->durSamples) { v->active = 0; break; }

            // ---- 控制率：音高轮廓 + 声道调谐 + 包络重锚 ----
            if ((idx & BIRD_CTRL_MASK) == 0) {
                float uNow = (float)idx / (float)BIRD_SR * invDur;
                float uEnd = (float)(idx + BIRD_CTRL) / (float)BIRD_SR * invDur;
                int32_t segBefore = v->seg;
                float logNow, logEnd, vibNow = 0.0f, vibEnd = 0.0f;
                int32_t controlN;

                if (uNow > 1.0f) uNow = 1.0f;
                if (uEnd > 1.0f) uEnd = 1.0f;
                logNow = bird_contour_log(v, uNow);
                logEnd = bird_contour_log(v, uEnd);
                v->pitchExact = (v->seg != segBefore) ? 1 : 0;   // 块内跨过拐点 → 逐点精确
                if (v->hasVib) {
                    // 颤音在对数域里加（f0 与 logF0 同步，beta 反解不会跳）
                    vibNow = log1pf(v->vibDepth * v->vibS);
                    logNow += vibNow;
                    {
                        float nc = v->vibC * v->vibRotC - v->vibS * v->vibRotS;
                        float ns = v->vibS * v->vibRotC + v->vibC * v->vibRotS;
                        v->vibC = nc;
                        v->vibS = ns;
                    }
                    vibEnd = log1pf(v->vibDepth * v->vibS);
                    logEnd += vibEnd;
                }
                v->logF0 = logNow;
                v->f0 = expf(logNow);
                controlN = v->envN - idx;
                if (controlN > BIRD_CTRL) controlN = BIRD_CTRL;
                if (controlN < 1) controlN = 1;
                v->logF0Inc = (logEnd - logNow) / (float)controlN;
                v->f0Ratio = expf(v->logF0Inc);
                v->pitchVibOff = vibNow;
                v->pitchVibInc = (vibEnd - vibNow) / (float)controlN;

                bird_biquad_bandpass(&v->tract[0], v->f0, tractQ);
                if (harm) bird_biquad_bandpass(&v->tractHarm[0], v->f0 * 2.0f, tractQ);
                if (twoVoices) {
                    bird_biquad_bandpass(&v->tract[1], v->f0 * BIRD_DETUNE_TWO, tractQ);
                    if (harm)
                        bird_biquad_bandpass(&v->tractHarm[1], v->f0 * 2.0f * BIRD_DETUNE_TWO, tractQ);
                }
            }

            // 跑到本控制块 / 本次缓冲 / 本音节的边界
            chunk = BIRD_CTRL - (int32_t)(idx & BIRD_CTRL_MASK);
            if (chunk > n - i) chunk = n - i;
            if (idx + chunk > v->durSamples) chunk = v->durSamples - idx;

            for (k = 0; k < chunk; k++, i++, idx++) {
                float env, psEnv, pressure, alpha, beta, amp, source, sample;

                // ---- 包络（增量求值，衰减用闭环重锚）----
                if (idx >= v->envN) {
                    env = 0.0f;
                    v->env = 0.0f;
                }
                else {
                    float t = (float)idx / (float)BIRD_SR;
                    if (t < v->envAtkS) {
                        v->env = t / v->envAtkS;
                    }
                    else if (v->envHold && idx < v->envKneeN) {
                        v->env = 1.0f;
                    }
                    else if (v->envPhase) {
                        if ((idx & BIRD_CTRL_MASK) == 0) {
                            float ref = v->envHold ? v->envKneeS : v->envAtkS;
                            v->env = (t > ref) ? expf(-(t - ref) * v->envInvTau) : 1.0f;
                        }
                        else {
                            v->env *= v->envDecMul;
                        }
                    }
                    else {
                        float ref = v->envHold ? v->envKneeS : v->envAtkS;
                        v->envPhase = 1;
                        v->env = (t > ref) ? expf(-(t - ref) * v->envInvTau) : 1.0f;
                    }
                    env = v->env;
                    if (v->envRelN && idx >= v->envRelStart) {
                        float r = 1.0f - (float)(idx - v->envRelStart) * v->envRelInc;
                        if (r < 0.0f) r = 0.0f;
                        env *= r;
                    }
                }

                // 调幅只调制**气压**（不调制输出增益，与参考实现一致）
                psEnv = env;
                if (v->hasAm) {
                    if ((idx & BIRD_CTRL_MASK) == 0) {
                        float phase = v->amOmega * (float)idx;
                        v->amC = cosf(phase);
                        v->amS = sinf(phase);
                    }
                    psEnv *= 1.0f - v->amDepth * (0.5f - 0.5f * v->amC);
                    {
                        float nc = v->amC * v->amRotC - v->amS * v->amRotS;
                        float ns = v->amS * v->amRotC + v->amC * v->amRotS;
                        v->amC = nc;
                        v->amS = ns;
                    }
                }
                pressure = v->level * psEnv * timbreGain;
                if (pressure > 1.4f) pressure = 1.4f;
                else if (pressure < 0.0f) pressure = 0.0f;
                alpha = 0.05f + 0.4f * pressure;

                beta = bird_beta_for(alpha, v->f0 * v->invGamma);
                amp = bird_amp_for(alpha, beta);
                source = bird_step_osc(v, 0, alpha, beta, amp);
                sample = bird_biquad_run(&v->tract[0], source);
                if (harm) sample += harmGain * bird_biquad_run(&v->tractHarm[0], source);

                if (twoVoices) {
                    float phiB = v->f0 * BIRD_DETUNE_TWO * v->invGamma;
                    float betaB = bird_beta_for(alpha, phiB);
                    float ampB = bird_amp_for(alpha, betaB);
                    float sourceB = bird_step_osc(v, 1, alpha, betaB, ampB);
                    float filteredB = bird_biquad_run(&v->tract[1], sourceB);
                    if (harm) filteredB += harmGain * bird_biquad_run(&v->tractHarm[1], sourceB);
                    sample = 0.5f * sample + 0.5f * filteredB;
                }

                // 音高斜坡推进（拐点块内逐点精确，其余块用乘性斜坡）
                if (v->pitchExact) {
                    float uNext = (float)(idx + 1) / (float)BIRD_SR * invDur;
                    float advance = (float)((idx & BIRD_CTRL_MASK) + 1);
                    if (uNext > 1.0f) uNext = 1.0f;
                    v->logF0 = bird_contour_log(v, uNext) + v->pitchVibOff + v->pitchVibInc * advance;
                    v->f0 = expf(v->logF0);
                }
                else {
                    v->f0 *= v->f0Ratio;
                    v->logF0 += v->logF0Inc;
                }

                // 输出直流阻断 + 12kHz 气管滚降
                {
                    float hp = sample - v->dcX + s_dcCoeff * v->dcY;
                    v->dcX = sample;
                    v->dcY = hp;
                    v->lpY += s_lpCoeff * (hp - v->lpY);
                }
                // 包络同时作为输出线性增益（参考实现 §8.1 的做法）
                out[i] += v->lpY * v->gain * env;
            }
            v->pos = idx;
        }
    }
}

// ===========================================================================
// 乐句调度（按采样计数推进，与 UI 帧率解耦）
// ===========================================================================

static inline uint32_t bird_rand(void) {
    uint32_t x = s_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s_rng = x;
    return x;
}

// 乐句时长（采样）：各音节时长之和 + 音节之间的间隔
static int32_t bird_phrase_samples(const BirdSpecies *sp, int32_t song, float durScale) {
    const BirdSong *so = &sp->songs[song];
    float sum = 0.0f;
    int32_t i;
    for (i = 0; i < (int32_t)so->n; i++) {
        sum += sp->syll[so->syl[i]].dur * durScale;
        if (i + 1 < (int32_t)so->n) sum += so->gaps[i];
    }
    return (int32_t)(sum * (float)BIRD_SR);
}

// 乐句之间的休止（采样）：周期 - 乐句时长，至少 120ms
static int32_t bird_rest_samples(const BirdPerformer *p) {
    const BirdSpecies *sp = &s_species[p->species];
    float scale = (float)(100 + s_tempo_pct) / 100.0f;
    int32_t phrase = bird_phrase_samples(sp, p->song, p->part.durScale * scale);
    int32_t period = (int32_t)((float)phrase * p->part.periodScale);
    int32_t rest = period - phrase;
    int32_t minRest = (int32_t)(0.12f * (float)BIRD_SR);
    return (rest > minRest) ? rest : minRest;
}

// 触发表演者的下一个事件（音节或下一遍乐句）
static void bird_perf_fire(BirdPerformer *p) {
    const BirdSpecies *sp = &s_species[p->species];
    const BirdSong *so = &sp->songs[p->song];
    float scale = (float)(100 + s_tempo_pct) / 100.0f;
    float durScale = p->part.durScale * scale;
    float pitchMul = p->part.pitchMul * (float)(100 + s_pitch_pct) / 100.0f;
    uint8_t si;
    int32_t durS, gapS = 0;

    if (so->n == 0) { p->timer = (int32_t)(0.5f * (float)BIRD_SR); return; }
    si = so->syl[p->cursor];
    bird_note(sp, (int32_t)si, pitchMul, durScale, p->part.levelScale, p->part.gain, p->tag);

    durS = (int32_t)(sp->syll[si].dur * durScale * (float)BIRD_SR);
    if (p->cursor + 1 < (int32_t)so->n) {
        gapS = (int32_t)(so->gaps[p->cursor] / scale * (float)BIRD_SR);
        p->cursor++;
        p->timer = durS + gapS;
    }
    else {
        // 乐句结束：换下一首曲目，隔一段休止再唱
        p->cursor = 0;
        if (sp->nSongs > 1) p->song = (p->song + 1) % (int32_t)sp->nSongs;
        p->timer = durS + bird_rest_samples(p);
    }
    if (p->timer < 1) p->timer = 1;
}

// 按当前模式/鸟种装载表演者（play、select、set_mode 时调用）
static void bird_arm(void) {
    int32_t i;
    for (i = 0; i < BIRD_MAX_PERF; i++) {
        BirdPerformer *p = &s_perf[i];
        const BirdPart *part = &s_parts[s_mode][i];
        p->part = *part;
        p->on = part->on;
        p->species = s_species_sel;
        p->song = 0;
        p->cursor = 0;
        p->tag = (uint8_t)i;
        if (!p->on) { p->timer = 0; continue; }
        {
            const BirdSpecies *sp = &s_species[p->species];
            float scale = (float)(100 + s_tempo_pct) / 100.0f;
            int32_t phrase = bird_phrase_samples(sp, 0, p->part.durScale * scale);
            int32_t period = (int32_t)((float)phrase * p->part.periodScale);
            p->timer = (int32_t)(p->part.phase * (float)period);
            if (p->timer < 0) p->timer = 0;
        }
    }
}

static void bird_panic(void) {
    int32_t i;
    for (i = 0; i < BIRD_MAX_VOICES; i++) s_voices[i].active = 0;
    s_level = 0.0f;
}

// 把最多 n 个采样合成为 int16（含软限幅与音量）
static void bird_render_block(int16_t *dst, int32_t n) {
    int32_t done = 0;
    int32_t i;
    float peak = 0.0f;
    float vol = (float)s_volume / (float)BIRD_VOL_MAX;

    if (n > BIRD_BLOCK) n = BIRD_BLOCK;
    memset(s_fbuf, 0, (size_t)n * sizeof(float));

    // 以"下一个表演者事件"为界把块切开：音节起点因此**采样精确**，
    // 不受 UI 帧率或块长量化影响（鸟叫的节奏感全靠这个）。
    while (done < n) {
        int32_t chunk = n - done;
        int32_t fired = 0;

        for (i = 0; i < BIRD_MAX_PERF; i++) {
            BirdPerformer *p = &s_perf[i];
            if (!p->on) continue;
            if (p->timer <= 0) { fired = 1; break; }
            if (p->timer < chunk) chunk = p->timer;
        }
        if (fired) {
            for (i = 0; i < BIRD_MAX_PERF; i++) {
                BirdPerformer *p = &s_perf[i];
                if (p->on && p->timer <= 0) bird_perf_fire(p);
            }
            continue;
        }
        if (chunk > 0) {
            bird_synth_add(&s_fbuf[done], chunk);
            done += chunk;
            for (i = 0; i < BIRD_MAX_PERF; i++) {
                BirdPerformer *p = &s_perf[i];
                if (p->on) p->timer -= chunk;
            }
        }
    }

    // 多声部叠加后软限幅，再按音量转 int16
    for (i = 0; i < n; i++) {
        float x = bird_tanh(s_fbuf[i] * (BIRD_MIX_DRIVE * BIRD_OUT_GAIN)) * BIRD_MIX_TRIM * vol;
        float ax = fabsf(x);
        if (ax > peak) peak = ax;
        if (x > 1.0f) x = 1.0f;
        else if (x < -1.0f) x = -1.0f;
        dst[i] = (int16_t)lrintf(x * 32767.0f);
    }
    // 峰值电平（衰减峰值，供电平表/星图用）
    s_level = (s_level * 0.85f > peak) ? (s_level * 0.85f) : peak;
}

// ===========================================================================
// 音频设备（含"静音走带"）
// ===========================================================================

static uint64_t bird_now_ms(void) {
    return (s_rt != NULL) ? (uint64_t)am_runtime_now_ms(s_rt) : 0;
}

// HAL 的音量域是 0~255，本库对外是 0~128（同音乐盒）：128 → 满量程
static uint8_t bird_hal_volume(void) {
    int32_t v = s_volume * 255 / BIRD_VOL_MAX;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (uint8_t)v;
}

static void bird_audio_open(void) {
    if (s_audio_on) return;
    if (audio_out_init((uint32_t)BIRD_SR, bird_hal_volume()) == 0) {
        s_audio_on = 1;
        s_silent = 0;
    }
    else {
        // 无音频设备（打桩平台 / 浏览器自动播放被挂起 / node 回归台）：
        // 转入静音走带——照常合成、按墙钟丢弃，界面照样有内容。
        s_audio_on = 0;
        s_silent = 1;
        s_clock_ms = bird_now_ms();
    }
}

static void bird_audio_close(void) {
    if (s_audio_on) {
        audio_out_stop();
        audio_out_close();
        s_audio_on = 0;
    }
    s_silent = 0;
}

// 合成若干块送给设备。返回本次送入设备的采样数。
//
// 三种状态：
//   · 设备正常     —— 有空槽就渲染一块并投进去（背压由 HAL 给，本函数不猜速率）；
//   · 静音走带     —— 没有设备（打桩平台/无 Web Audio）时按**墙钟**补足应播出的采样
//                    并丢弃，于是乐句、星图、电平表照常推进；
//   · 设备停滞     —— 设备在但 >500ms 一直不给空槽（浏览器把 AudioContext 挂起了、
//                    或设备被别的进程占住）→ 转静音走带，别让画面跟着冻住。
//   静音走带每 2 秒回头再试一次设备（用户点一下画面解锁音频后会自动接回来）。
static int32_t bird_pump(int32_t max_blocks) {
    int32_t total = 0;
    int32_t blocks = 0;
    uint64_t now;

    if (!s_playing || s_paused) return 0;
    if (!s_audio_on && !s_silent) bird_audio_open();
    now = bird_now_ms();

    if (s_silent && (now - s_retry_ms) >= 2000u) {
        s_retry_ms = now;
        if (audio_out_init((uint32_t)BIRD_SR, bird_hal_volume()) == 0) {
            s_audio_on = 1;
            s_silent = 0;
            s_feed_ms = now;
        }
    }

    if (s_silent) {
        int64_t due = (int64_t)((now - s_clock_ms) * (uint64_t)BIRD_SR / 1000u);
        if (due > (int64_t)BIRD_BLOCK * 8) due = (int64_t)BIRD_BLOCK * 8;   // 落后就重基
        if (due < BIRD_BLOCK) return 0;
        s_clock_ms = now;
        while (total < (int32_t)due) {
            int32_t n = BIRD_BLOCK;
            if (total + n > (int32_t)due) n = (int32_t)due - total;
            bird_render_block(s_out_buf, n);
            total += n;
        }
        return total;
    }

    while (blocks < max_blocks && audio_out_queue_free()) {
        bird_render_block(s_out_buf, BIRD_BLOCK);
        if (audio_out_enqueue(s_out_buf, (uint32_t)BIRD_BLOCK) != 0) {
            // 投放失败：转静音走带，画面继续（不让 App 卡死在一个坏设备上）
            s_audio_on = 0;
            s_silent = 1;
            s_clock_ms = now;
            s_retry_ms = now;
            break;
        }
        total += BIRD_BLOCK;
        blocks++;
        s_feed_ms = now;
    }
    if (total == 0) {
        if (s_feed_ms == 0) s_feed_ms = now;
        if (now - s_feed_ms > 500u) {          // 设备停滞看门狗
            s_silent = 1;
            s_clock_ms = now;
            s_retry_ms = now;
        }
    }
    return total;
}

// ===========================================================================
// 初始化
// ===========================================================================

static void bird_init_once(void) {
    if (s_inited) return;
    s_inited = 1;
    s_dcCoeff = expf(-2.0f * (float)M_PI * 180.0f / (float)BIRD_SR);
    s_lpCoeff = 1.0f - expf(-2.0f * (float)M_PI * 12000.0f / (float)BIRD_SR);
    s_alpha_min = bird_cal_alpha[0];
    s_alpha_step = (bird_cal_alpha[BIRD_CAL_ALPHA_N - 1] - bird_cal_alpha[0])
                 / (float)(BIRD_CAL_ALPHA_N - 1);
    bird_panic();
    bird_arm();
}

// ===========================================================================
// Native 函数实现
// ===========================================================================

int32_t am_native_Bird_count(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    return bird_push_int(proc, BIRD_SPECIES_N);
}

int32_t am_native_Bird_name(am_runtime_t *rt, am_process_t *proc) {
    int32_t i = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &i)) return -1;
    if (i < 0 || i >= BIRD_SPECIES_N) return bird_push_null(proc);
    return bird_push_utf8(proc, s_species[i].name);
}

int32_t am_native_Bird_latin(am_runtime_t *rt, am_process_t *proc) {
    int32_t i = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &i)) return -1;
    if (i < 0 || i >= BIRD_SPECIES_N) return bird_push_null(proc);
    return bird_push_utf8(proc, s_species[i].latin);
}

int32_t am_native_Bird_mnemonic(am_runtime_t *rt, am_process_t *proc) {
    int32_t i = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &i)) return -1;
    if (i < 0 || i >= BIRD_SPECIES_N) return bird_push_null(proc);
    return bird_push_utf8(proc, s_species[i].mnemonic);
}

// (Bird.voicing i) : (音节数 曲目数 最低音 最高音)；音域由轮廓端点算出
int32_t am_native_Bird_voicing(am_runtime_t *rt, am_process_t *proc) {
    double vals[4];
    int32_t i = 0, j, k;
    float lo = 1e9f, hi = 0.0f;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &i)) return -1;
    if (i < 0 || i >= BIRD_SPECIES_N) return bird_push_null(proc);
    for (j = 0; j < (int32_t)s_species[i].nSyll; j++) {
        const BirdSyllable *syl = &s_species[i].syll[j];
        for (k = 0; k < (int32_t)syl->nContour; k++) {
            float hz = syl->contour[k].hz;
            if (hz < lo) lo = hz;
            if (hz > hi) hi = hz;
        }
    }
    if (hi <= 0.0f) { lo = 0.0f; hi = 0.0f; }
    vals[0] = (double)s_species[i].nSyll;
    vals[1] = (double)s_species[i].nSongs;
    vals[2] = (double)lo;
    vals[3] = (double)hi;
    return bird_push_list(proc, vals, 4);
}

int32_t am_native_Bird_select(am_runtime_t *rt, am_process_t *proc) {
    int32_t i = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &i)) return -1;
    if (i < 0 || i >= BIRD_SPECIES_N) return bird_push_int(proc, s_species_sel);
    if (i != s_species_sel) {
        s_species_sel = i;
        bird_panic();
        bird_arm();
        if (s_playing && s_silent) s_clock_ms = bird_now_ms();
    }
    return bird_push_int(proc, s_species_sel);
}

int32_t am_native_Bird_selected(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    return bird_push_int(proc, s_species_sel);
}

int32_t am_native_Bird_mode(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    return bird_push_int(proc, s_mode);
}

int32_t am_native_Bird_set_mode(am_runtime_t *rt, am_process_t *proc) {
    int32_t m = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &m)) return -1;
    if (m < 0) m = 0;
    if (m > 2) m = 2;
    if (m != s_mode) {
        s_mode = m;
        bird_panic();
        bird_arm();
        if (s_playing && s_silent) s_clock_ms = bird_now_ms();
    }
    return bird_push_int(proc, s_mode);
}

int32_t am_native_Bird_play(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    s_rt = rt;
    if (s_playing) {
        s_paused = 0;
        return bird_push_bool(proc, true);
    }
    s_playing = 1;
    s_paused = 0;
    bird_panic();
    bird_arm();
    s_clock_ms = bird_now_ms();
    bird_audio_open();
    printf("[bird] play species=%d mode=%d audio=%s\n", (int)s_species_sel, (int)s_mode,
           s_silent ? "silent" : "on");
    return bird_push_bool(proc, true);
}

int32_t am_native_Bird_stop(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    s_playing = 0;
    s_paused = 0;
    bird_panic();
    bird_audio_close();
    return bird_push_bool(proc, true);
}

int32_t am_native_Bird_is_playing(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    return bird_push_bool(proc, (s_playing && !s_paused));
}

int32_t am_native_Bird_pause(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    if (s_playing) {
        s_paused = 1;
        bird_panic();
    }
    return bird_push_bool(proc, true);
}

int32_t am_native_Bird_resume(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    s_rt = rt;
    if (s_playing && s_paused) {
        s_paused = 0;
        s_clock_ms = bird_now_ms();
    }
    return bird_push_bool(proc, s_playing && !s_paused);
}

int32_t am_native_Bird_pump(am_runtime_t *rt, am_process_t *proc) {
    int32_t n;
    (void)rt;
    bird_init_once();
    s_rt = rt;
    n = bird_pump(6);
    return bird_push_int(proc, n);
}

// (Bird.chirp) —— 触屏点按试听：立刻叫一声（独立于乐句调度，可连点）
int32_t am_native_Bird_chirp(am_runtime_t *rt, am_process_t *proc) {
    const BirdSpecies *sp;
    float jitter;
    int32_t syl;
    (void)rt;
    bird_init_once();
    s_rt = rt;
    sp = &s_species[s_species_sel];
    if (sp->nSyll == 0) return bird_push_bool(proc, false);
    if (!s_audio_on && !s_silent) bird_audio_open();

    // 随机微差（±1.5% 音高、±8% 时长）：同一只鸟也不是复读机
    jitter = 1.0f + ((float)(bird_rand() % 300) - 150.0f) / 10000.0f;
    syl = (int32_t)(bird_rand() % (uint32_t)sp->nSyll);
    return bird_push_bool(proc, bird_note(sp, syl, jitter,
                                         1.0f + ((float)(bird_rand() % 160) - 80.0f) / 1000.0f,
                                         1.0f, 1.0f, 0) != 0);
}

int32_t am_native_Bird_voices(am_runtime_t *rt, am_process_t *proc) {
    int32_t i, n = 0;
    (void)rt;
    bird_init_once();
    for (i = 0; i < BIRD_MAX_VOICES; i++) if (s_voices[i].active) n++;
    return bird_push_int(proc, n);
}

int32_t am_native_Bird_vf0(am_runtime_t *rt, am_process_t *proc) {
    int32_t i = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &i)) return -1;
    if (i < 0 || i >= BIRD_MAX_VOICES || !s_voices[i].active) return bird_push_int(proc, 0);
    return bird_push_number(proc, (double)s_voices[i].f0);
}

int32_t am_native_Bird_venv(am_runtime_t *rt, am_process_t *proc) {
    int32_t i = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &i)) return -1;
    if (i < 0 || i >= BIRD_MAX_VOICES || !s_voices[i].active) return bird_push_int(proc, 0);
    return bird_push_int(proc, (int32_t)(s_voices[i].env * 100.0f));
}

int32_t am_native_Bird_vtag(am_runtime_t *rt, am_process_t *proc) {
    int32_t i = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &i)) return -1;
    if (i < 0 || i >= BIRD_MAX_VOICES || !s_voices[i].active) return bird_push_int(proc, 0);
    return bird_push_int(proc, (int32_t)s_voices[i].tag);
}

int32_t am_native_Bird_level(am_runtime_t *rt, am_process_t *proc) {
    int32_t lv;
    (void)rt;
    bird_init_once();
    lv = (int32_t)(s_level * 255.0f);
    if (lv < 0) lv = 0;
    if (lv > 255) lv = 255;
    return bird_push_int(proc, lv);
}

int32_t am_native_Bird_pitch(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    return bird_push_int(proc, s_pitch_pct);
}

int32_t am_native_Bird_set_pitch(am_runtime_t *rt, am_process_t *proc) {
    int32_t pct = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &pct)) return -1;
    if (pct < -50) pct = -50;
    if (pct > 50) pct = 50;
    s_pitch_pct = pct;
    return bird_push_int(proc, s_pitch_pct);
}

int32_t am_native_Bird_tempo(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    return bird_push_int(proc, s_tempo_pct);
}

int32_t am_native_Bird_set_tempo(am_runtime_t *rt, am_process_t *proc) {
    int32_t pct = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &pct)) return -1;
    if (pct < -50) pct = -50;
    if (pct > 50) pct = 50;
    s_tempo_pct = pct;
    return bird_push_int(proc, s_tempo_pct);
}

int32_t am_native_Bird_volume(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    return bird_push_int(proc, s_volume);
}

int32_t am_native_Bird_set_volume(am_runtime_t *rt, am_process_t *proc) {
    int32_t v = 0;
    (void)rt;
    bird_init_once();
    if (!bird_pop_int(proc, &v)) return -1;
    if (v < 0) v = 0;
    if (v > BIRD_VOL_MAX) v = BIRD_VOL_MAX;
    s_volume = v;
    if (s_audio_on) audio_out_set_volume(bird_hal_volume());
    return bird_push_int(proc, s_volume);
}

int32_t am_native_Bird_rate(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    bird_init_once();
    return bird_push_int(proc, BIRD_SR);
}

// ===========================================================================
// 库注册表
// ===========================================================================

static const am_native_func_entry_t am_native_Bird_funcs[] = {
    { L"count",      am_native_Bird_count },
    { L"name",       am_native_Bird_name },
    { L"latin",      am_native_Bird_latin },
    { L"mnemonic",   am_native_Bird_mnemonic },
    { L"voicing",    am_native_Bird_voicing },
    { L"select",     am_native_Bird_select },
    { L"selected",   am_native_Bird_selected },
    { L"mode",       am_native_Bird_mode },
    { L"set_mode",   am_native_Bird_set_mode },
    { L"play",       am_native_Bird_play },
    { L"stop",       am_native_Bird_stop },
    { L"is_playing", am_native_Bird_is_playing },
    { L"pause",      am_native_Bird_pause },
    { L"resume",     am_native_Bird_resume },
    { L"pump",       am_native_Bird_pump },
    { L"chirp",      am_native_Bird_chirp },
    { L"voices",     am_native_Bird_voices },
    { L"vf0",        am_native_Bird_vf0 },
    { L"venv",       am_native_Bird_venv },
    { L"vtag",       am_native_Bird_vtag },
    { L"level",      am_native_Bird_level },
    { L"pitch",      am_native_Bird_pitch },
    { L"set_pitch",  am_native_Bird_set_pitch },
    { L"tempo",      am_native_Bird_tempo },
    { L"set_tempo",  am_native_Bird_set_tempo },
    { L"volume",     am_native_Bird_volume },
    { L"set_volume", am_native_Bird_set_volume },
    { L"rate",       am_native_Bird_rate },
};

const am_native_lib_entry_t am_native_Bird_lib = {
    L"Bird",
    am_native_Bird_funcs,
    sizeof(am_native_Bird_funcs) / sizeof(am_native_Bird_funcs[0])
};
