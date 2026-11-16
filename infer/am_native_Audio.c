////////////////////////////////////////////////////////////////////////////
//  Native Library : Audio —— 设计说明见 am_native_Audio.h
//
//  分层（与 C 版频谱仪 ui_spectrogram.c 的信号链同构，但**独立实现**，
//  不改动既有模块）：
//
//      Scheme App ──(Audio.open/read/bin/sample/rms/...)
//                        │
//                        ├── hal_audio_in（mic_init / mic_read / mic_close）
//                        ├── Hann 加窗 + nano_fft（与 C 版同一份旋转因子实现）
//                        └── 对数幅度映射 → 每频点 0~255 强度（标量）
//
//  本文件只做「采样 → 频域强度标量」，一行界面代码也没有；麦克风经 HAL 访问，
//  因此桌面 / WASM / 掌机三处行为一致，无平台分支。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "am_native_Audio.h"

// ---------------------------------------------------------------------------
// 编译期守卫：宿主本地库数量上限。
//   上游 vendor/Animac 的 AM_NATIVE_MAX_LIBS 默认 16，本项目注册的前 16 个库
//   （System/Math/String/Table/Gfx/Time/Diag/Touch/Imu/Key/Fs/Ui/App/Chat/Raster/Music）
//   恰好占满，本库是第 17 个。超出上限时 am_runtime_register_native_lib 会**静默
//   失败**（返回 -1、无日志），表现为运行时「callnative: 未找到native函数
//   Audio.open」——极难定位。上游该宏有 #ifndef 保护，属"宿主可覆盖"的构建选项，
//   因此各构建入口（infer/Makefile、mp135.mk、luckfox.mk、build_wasm_gui.sh、
//   platformio.ini、android CMakeLists、tools/e2e 回归台）统一加
//   -DAM_NATIVE_MAX_LIBS=32（给后续新增库留余量，免得每加一个库都要动 8 个构建入口）；
//   忘记加的构建在这里**编译失败**而不是运行时静默失效。
// ---------------------------------------------------------------------------
#if !defined(AM_NATIVE_MAX_LIBS) || (AM_NATIVE_MAX_LIBS < 17)
#error "需要 -DAM_NATIVE_MAX_LIBS=32（宿主本地库数量上限，见 am_native_Audio.c 文件头与 VENDOR.md）"
#endif

#include "hal_audio_in.h"
#include "hal_audio_out.h"   // audio_out_get_master_volume：mic_init 的 restore_volume 来源
#include "hal_ram.h"         // platform_calloc / free（分析缓冲，与 C 版频谱仪同源）
#include "nano_fft.h"

// ===========================================================================
// 参数（与 infer/ui_spectrogram.c 逐项对齐）
// ===========================================================================

// FFT 点数（一帧采样数；@44.1kHz ≈ 23ms）。须为 2 的幂。
#define AUDIO_FFT_SIZE      (1024)
// 可绘制频点数：跳过直流分量（bin 0），与 C 版 SP_BINS 的取法一致（bin 1 起）
#define AUDIO_MAX_BINS      (AUDIO_FFT_SIZE / 2 - 1)
// 对数幅度 → 映射域的系数（C 版 SP_LOG_SCALE）
#define AUDIO_LOG_SCALE     (80.0f)
// 默认采样率（与 C 版 SP_SAMPLE_RATE 一致；(Audio.open) 可覆盖）
#define AUDIO_DEFAULT_RATE  (44100)

// 时域电平表映射范围：−60 dBFS…0 dBFS → 0…255
#define AUDIO_RMS_FLOOR_DB  (60.0f)
// 基准电平 / 动态范围的宽松钳位（C 版频谱仪的 0~240 / 32~512 属 App 侧策略）
#define AUDIO_BASE_MAX      (1000.0f)
#define AUDIO_RANGE_MIN     (1.0f)
#define AUDIO_RANGE_MAX     (4096.0f)

// ===========================================================================
// 引擎状态（进程级单例，与 (native Music) 同一形态）
// ===========================================================================

static int32_t  s_open = 0;         // 1 = 麦克风已打开
static int32_t  s_probe = -1;       // 可用性探测缓存：-1 未探测 / 0 不可用 / 1 可用
static int32_t  s_have = 0;         // 1 = 最近一次 read 产出了有效频率数据
static int32_t  s_win_ready = 0;    // 1 = 窗函数与旋转因子表已构建
static uint32_t s_rate = AUDIO_DEFAULT_RATE;

// 映射参数（基准电平 / 动态范围）：跨 open/close 保留，App 的设置不会因重试丢失
static float s_base  = 240.0f;      // C 版 SP_LEVEL_BASE_DEFAULT
static float s_range = 300.0f;      // C 版 SP_LEVEL_RANGE_DEFAULT

// 工作区（open 时按需分配，close 时释放；约 18.5KB）
static float   *s_window = NULL;    // Hann 窗
static float   *s_tw_re  = NULL;    // 旋转因子（全表，基 2 迭代按步长取用）
static float   *s_tw_im  = NULL;
static float   *s_re     = NULL;    // FFT 实部（同时是加窗后的时域帧）
static float   *s_im     = NULL;
static int16_t *s_mic    = NULL;    // 最新一帧原始采样（(Audio.sample k) 的来源）

// 最新一列的分析结果
static uint8_t s_bins[AUDIO_MAX_BINS];
static uint8_t s_rms = 0;

// ===========================================================================
// 操作数栈辅助（契约 N1/N6：弹完参数后压 1 值 + step 1 次）
// ===========================================================================

static bool audio_pop_number(am_process_t *proc, double *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;

    if (am_value_is_float(v)) { *out = (double)am_value_to_float(v); return true; }
    if (am_value_is_int(v))   { *out = (double)am_value_to_int(v);   return true; }
    if (am_value_is_uint(v))  { *out = (double)am_value_to_uint(v);  return true; }
    return false;
}

// 弹一个整数索引（浮点按四舍五入取整，越界钳位后由调用方判定）
static bool audio_pop_int(am_process_t *proc, int32_t *out) {
    double f = 0.0;
    if (!audio_pop_number(proc, &f)) return false;
    if (isnan(f)) return false;
    if (f > 2147483647.0)  f = 2147483647.0;
    if (f < -2147483648.0) f = -2147483648.0;
    *out = (int32_t)f;
    return true;
}

static int32_t audio_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t audio_push_bool(am_process_t *proc, bool b) {
    return audio_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

static int32_t audio_push_number(am_process_t *proc, double x) {
    return audio_push(proc, am_make_value_of_float((am_float_t)x));
}

static int32_t audio_push_int(am_process_t *proc, int32_t n) {
    return audio_push(proc, am_make_value_of_int((am_int_t)n));
}

// 压入一个 n 元数值列表（分配失败按软失败压 #null）
static int32_t audio_push_number_list(am_process_t *proc, const double *vals, int32_t n) {
    am_list_t *lst = am_list_create(proc->heap_alloc, (size_t)n, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    am_handle_t lhd;
    if (lst == NULL) return audio_push(proc, AM_VALUE_NULL);
    for (int32_t i = 0; i < n; i++) {
        am_list_t *grown = am_list_push(proc->heap_alloc, lst, am_make_value_of_float((am_float_t)vals[i]));
        if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return audio_push(proc, AM_VALUE_NULL); }
        lst = grown;
    }
    lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return audio_push(proc, AM_VALUE_NULL); }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return audio_push(proc, AM_VALUE_NULL);
    }
    if (am_process_push_operand(proc, am_make_value_of_handle(lhd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// ===========================================================================
// 分析流水线
// ===========================================================================

// 分配工作区并构建 Hann 窗与旋转因子表（幂等；失败返回 -1）
static int32_t audio_alloc(void) {
    if (s_window == NULL) s_window = (float *)platform_calloc(AUDIO_FFT_SIZE, sizeof(float));
    if (s_tw_re == NULL)  s_tw_re  = (float *)platform_calloc(AUDIO_FFT_SIZE / 2, sizeof(float));
    if (s_tw_im == NULL)  s_tw_im  = (float *)platform_calloc(AUDIO_FFT_SIZE / 2, sizeof(float));
    if (s_re == NULL)     s_re     = (float *)platform_calloc(AUDIO_FFT_SIZE, sizeof(float));
    if (s_im == NULL)     s_im     = (float *)platform_calloc(AUDIO_FFT_SIZE, sizeof(float));
    if (s_mic == NULL)    s_mic    = (int16_t *)platform_calloc(AUDIO_FFT_SIZE, sizeof(int16_t));
    if (s_window == NULL || s_tw_re == NULL || s_tw_im == NULL ||
        s_re == NULL || s_im == NULL || s_mic == NULL) {
        return -1;
    }
    if (!s_win_ready) {
        // Hann 窗（周期型）：w[n] = 0.5 · (1 − cos(2πn/N))，与 C 版逐字一致
        for (int32_t i = 0; i < AUDIO_FFT_SIZE; i++) {
            s_window[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * (float)i / (float)AUDIO_FFT_SIZE));
        }
        nano_fft_twiddle(s_tw_re, s_tw_im, AUDIO_FFT_SIZE);
        s_win_ready = 1;
    }
    return 0;
}

// 释放工作区（窗与旋转因子表也一并释放：重开时重建，代价可忽略）
static void audio_free(void) {
    if (s_window) { free(s_window); s_window = NULL; }
    if (s_tw_re)  { free(s_tw_re);  s_tw_re  = NULL; }
    if (s_tw_im)  { free(s_tw_im);  s_tw_im  = NULL; }
    if (s_re)     { free(s_re);     s_re     = NULL; }
    if (s_im)     { free(s_im);     s_im     = NULL; }
    if (s_mic)    { free(s_mic);    s_mic    = NULL; }
    s_win_ready = 0;
    s_have = 0;
    s_rms = 0;
    for (int32_t i = 0; i < AUDIO_MAX_BINS; i++) s_bins[i] = 0;
}

// 时域电平（RMS）→ 0~255（−60dBFS…0dBFS 线性映射；满量程附近即饱和）。
// 与时域帧一起算：即使 App 用"软 FFT"（自己在 Scheme 里做变换），电平表也照旧可用。
static void audio_update_rms(void) {
    float acc = 0.0f;
    for (int32_t i = 0; i < AUDIO_FFT_SIZE; i++) {
        float s = (float)s_mic[i];
        acc += s * s;
    }
    {
        float rms = sqrtf(acc / (float)AUDIO_FFT_SIZE);
        float db = 20.0f * log10f((rms + 1.0f) / 32768.0f);
        float lv = (db + AUDIO_RMS_FLOOR_DB) * (255.0f / AUDIO_RMS_FLOOR_DB);
        if (lv < 0.0f) lv = 0.0f;
        if (lv > 255.0f) lv = 255.0f;
        s_rms = (uint8_t)lv;
    }
}

// 对 s_mic 里的一帧做「加窗 + FFT + 对数幅度映射」（(Audio.read) 的后半段）
static void audio_analyze(void) {
    for (int32_t i = 0; i < AUDIO_FFT_SIZE; i++) {
        s_re[i] = (float)s_mic[i] * s_window[i];
        s_im[i] = 0.0f;
    }

    nano_fft_execute(s_re, s_im, s_tw_re, s_tw_im, AUDIO_FFT_SIZE);

    // v = (log10(mag+1)·80 − 基准) · 255 / 动态范围（与 C 版同一映射）
    for (int32_t x = 0; x < AUDIO_MAX_BINS; x++) {
        float re = s_re[x + 1];
        float im = s_im[x + 1];
        float mag = sqrtf(re * re + im * im);
        float v_raw = log10f(mag + 1.0f) * AUDIO_LOG_SCALE;
        float v = (v_raw - s_base) * 255.0f / s_range;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        s_bins[x] = (uint8_t)v;
    }
}

// ===========================================================================
// Native 函数实现
// ===========================================================================

int32_t am_native_Audio_is_available(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (s_open) return audio_push_bool(proc, true);
    if (s_probe < 0) {
        // 从未探测：打开一次再关闭（不建分析缓冲；(Audio.open) 会自己再开一次）
        uint8_t vol = audio_out_get_master_volume();
        if (mic_init(AUDIO_DEFAULT_RATE, vol) == 0) {
            s_probe = 1;
            mic_close();
        }
        else {
            s_probe = 0;
        }
    }
    return audio_push_bool(proc, s_probe == 1);
}

int32_t am_native_Audio_open(am_runtime_t *rt, am_process_t *proc) {
    double rate_v = 0.0;
    int32_t rate = AUDIO_DEFAULT_RATE;
    uint8_t vol;
    (void)rt;

    if (!audio_pop_number(proc, &rate_v)) return -1;
    if (isnan(rate_v)) return -1;
    if (rate_v >= 8000.0 && rate_v <= 192000.0) rate = (int32_t)(rate_v + 0.5);
    else if (rate_v > 0.0) rate = (rate_v < 8000.0) ? 8000 : 192000;

    if (audio_alloc() != 0) {
        // 内存不足：软失败（App 按"无麦克风"降级），不留半初始化状态
        audio_free();
        return audio_push_bool(proc, false);
    }

    if (s_open) {
        mic_close();
        s_open = 0;
    }

    // restore_volume 只在"麦克风与扬声器共用 I2S"的宿主上有意义（ESP32），
    // 由调用方注入主音量（hal_audio_out 的语义约定，见 hal_audio_out.h）。
    vol = audio_out_get_master_volume();
    if (mic_init((uint32_t)rate, vol) != 0) {
        s_probe = 0;
        s_open = 0;
        return audio_push_bool(proc, false);
    }

    s_rate = (uint32_t)rate;
    s_open = 1;
    s_probe = 1;
    s_have = 0;
    s_rms = 0;
    return audio_push_bool(proc, true);
}

int32_t am_native_Audio_close(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (s_open) {
        mic_close();
        s_open = 0;
    }
    audio_free();
    return audio_push_bool(proc, true);
}

int32_t am_native_Audio_info(am_runtime_t *rt, am_process_t *proc) {
    double vals[3];
    (void)rt;
    vals[0] = (double)s_rate;
    vals[1] = (double)AUDIO_FFT_SIZE;
    vals[2] = (double)AUDIO_MAX_BINS;
    return audio_push_number_list(proc, vals, 3);
}

// 取一帧原始采样到 s_mic（并更新时域电平）。返回 1=取到一帧 / 0=数据不足 / -1=硬错误。
// 不做任何频域处理——"软 FFT"模式下 App 自己在 Scheme 里做变换，这里只负责交付采样。
static int32_t audio_capture_frame(void) {
    int32_t n;
    if (!s_open || s_mic == NULL) return 0;

    // 每次只要一帧（不足一帧时 WASM HAL 立即返回 0、ALSA 阻塞至就绪或超时）。
    // "追上音频"由 App 侧的每拍多次调用解决（见头文件的取数语义说明）。
    n = mic_read(s_mic, AUDIO_FFT_SIZE);
    if (n < 0) {
        // 硬错误（权限被拒 / 采集中断）：置为未打开，让 App 走"无麦克风"画面
        s_open = 0;
        return -1;
    }
    if (n < AUDIO_FFT_SIZE) return 0;

    audio_update_rms();
    s_have = 1;
    return 1;
}

int32_t am_native_Audio_capture(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return audio_push_number(proc, (double)audio_capture_frame());
}

int32_t am_native_Audio_read(am_runtime_t *rt, am_process_t *proc) {
    int32_t rc;
    (void)rt;

    rc = audio_capture_frame();
    if (rc != 1) return audio_push_number(proc, (double)rc);

    audio_analyze();   // 加窗 + FFT + 对数映射（"硬 FFT"）
    return audio_push_number(proc, 1.0);
}

int32_t am_native_Audio_bin(am_runtime_t *rt, am_process_t *proc) {
    int32_t idx = 0;
    (void)rt;
    if (!audio_pop_int(proc, &idx)) return -1;
    if (!s_have || idx < 0 || idx >= AUDIO_MAX_BINS) return audio_push_int(proc, 0);
    return audio_push_int(proc, (int32_t)s_bins[idx]);
}

int32_t am_native_Audio_sample(am_runtime_t *rt, am_process_t *proc) {
    int32_t idx = 0;
    (void)rt;
    if (!audio_pop_int(proc, &idx)) return -1;
    if (!s_have || s_mic == NULL || idx < 0 || idx >= AUDIO_FFT_SIZE) return audio_push_int(proc, 0);
    return audio_push_int(proc, (int32_t)s_mic[idx]);
}

int32_t am_native_Audio_rms(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!s_have) return audio_push_int(proc, 0);
    return audio_push_int(proc, (int32_t)s_rms);
}

int32_t am_native_Audio_level(am_runtime_t *rt, am_process_t *proc) {
    double vals[2];
    (void)rt;
    vals[0] = (double)s_base;
    vals[1] = (double)s_range;
    return audio_push_number_list(proc, vals, 2);
}

int32_t am_native_Audio_set_level(am_runtime_t *rt, am_process_t *proc) {
    double range_v = 0.0;
    double base_v = 0.0;
    double vals[2];
    (void)rt;

    if (!audio_pop_number(proc, &range_v)) return -1;   // 逆序弹参：先动态范围，后基准
    if (!audio_pop_number(proc, &base_v)) return -1;
    if (isnan(base_v) || isnan(range_v)) return -1;

    if (base_v < 0.0) base_v = 0.0;
    if (base_v > AUDIO_BASE_MAX) base_v = AUDIO_BASE_MAX;
    if (range_v < AUDIO_RANGE_MIN) range_v = AUDIO_RANGE_MIN;
    if (range_v > AUDIO_RANGE_MAX) range_v = AUDIO_RANGE_MAX;

    s_base = (float)base_v;
    s_range = (float)range_v;

    vals[0] = base_v;
    vals[1] = range_v;
    return audio_push_number_list(proc, vals, 2);
}

// ===========================================================================
// 库注册表
// ===========================================================================

static const am_native_func_entry_t am_native_Audio_funcs[] = {
    { L"is_available", am_native_Audio_is_available },
    { L"open",         am_native_Audio_open },
    { L"close",        am_native_Audio_close },
    { L"info",         am_native_Audio_info },
    { L"read",         am_native_Audio_read },
    { L"capture",      am_native_Audio_capture },
    { L"bin",          am_native_Audio_bin },
    { L"sample",       am_native_Audio_sample },
    { L"rms",          am_native_Audio_rms },
    { L"level",        am_native_Audio_level },
    { L"set_level",    am_native_Audio_set_level },
};

const am_native_lib_entry_t am_native_Audio_lib = {
    L"Audio",
    am_native_Audio_funcs,
    sizeof(am_native_Audio_funcs) / sizeof(am_native_Audio_funcs[0])
};
