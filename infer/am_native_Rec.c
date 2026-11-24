////////////////////////////////////////////////////////////////////////////
//  Native Library : Rec —— 设计说明见 am_native_Rec.h
//
//  分层（整条录音链路，一行界面代码也没有）：
//
//      Scheme App ──(Rec.start / pump / stop / stats / level / wave)
//                        │
//                        ├── hal_audio_in（mic_init / mic_read / mic_close）
//                        ├── vendor/aqua_mp3（Aqua MP3 编码器，320kbps CBR，单声道）
//                        └── hal_fs（platform_mkdir / fopen 流式追加 / 落盘钩子）
//
//  取数—编码—落盘的节奏全由 App 驱动（(Rec.pump n) 一次一小块），本库不建线程、
//  不注册定时器：wasm32 是单线程，任何"后台编码"都只会表现为界面卡顿。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>

#include "am_native_Rec.h"

// ---------------------------------------------------------------------------
// 编译期守卫：宿主本地库数量上限。
//   本项目此前已注册 17 个库（System/Math/String/Table/Gfx/Time/Diag/Touch/Imu/
//   Key/Fs/Ui/App/Chat/Raster/Music/Audio），本库是第 18 个。各构建入口统一加
//   -DAM_NATIVE_MAX_LIBS=32（见 am_native_Audio.c 文件头）。超出上限时
//   am_runtime_register_native_lib 会**静默失败**（表现为运行时「callnative:
//   未找到native函数 Rec.start」），故在此编译期拦下。
// ---------------------------------------------------------------------------
#if !defined(AM_NATIVE_MAX_LIBS) || (AM_NATIVE_MAX_LIBS < 18)
#error "需要 -DAM_NATIVE_MAX_LIBS=32（宿主本地库数量上限，见 am_native_Audio.c 文件头）"
#endif

// Aqua MP3 编码器（vendor/aqua_mp3/）。用**相对 infer/ 的路径**包含：各构建入口
// （Makefile / mp135.mk / luckfox.mk / build_wasm_gui.sh / platformio 的 src_dir=. /
// android CMakeLists）都以 infer/ 为编译工作目录、以 infer/ 为引号包含的搜索起点，
// 因此无需为它再加 -I（它自己内部的 aqua_common.h/aqua_tables.h 按自身目录解析）。
#include "vendor/aqua_mp3/aqua_encoder.h"

#include "hal_audio_in.h"
#include "hal_audio_out.h"   // audio_out_get_master_volume：mic_init 的 restore_volume 来源
#include "hal_fs.h"          // platform_mkdir / platform_is_directory
#include "hal_ram.h"         // platform_malloc / free
#include "platform.h"        // PLATFORM_ROOT_DIR
#include "utils.h"           // _mbstowcs（locale 无关）

// ===========================================================================
// 目标格式（需求给定，不做成参数；见头文件的说明）
// ===========================================================================

#define REC_RATE          (44100)                       // MPEG-1 允许的采样率
#define REC_CHANNELS      (1)                           // 单声道
#define REC_KBPS          (320)                         // CBR；MPEG-1 的最高码率
#define REC_FRAME         (1152)                        // 一帧两个 granule 的采样数
#define REC_FRAME_MS      (26.122448979591837)          // 1152/44100·1000
#define REC_MAX_DIR       "music"                       // 落点目录（音乐盒可直接播放）
#define REC_PATH_MAX      (256)
#define REC_PUMP_MAX      (8)                           // 单次 pump 的帧数上限

// 落盘钩子（弱符号：只有链入 hal_fs_persist_wasm.c 的 WASM 目标才有实现）
#if defined(__GNUC__)
extern void am_native_fs_on_write(const char *path, const uint8_t *data, size_t size) __attribute__((weak));
#endif

// ===========================================================================
// 引擎状态（进程级单例，与 (native Audio) / (native Music) 同一形态）
// ===========================================================================

static int32_t  s_state = 0;        // 1 = 录音中
static AquaEncoder *s_enc = NULL;   // 编码器上下文（含跨帧状态，约 9KB；见 rec_alloc_workspace）
static FILE    *s_fp = NULL;        // 输出文件（流式追加；收尾时 fclose）
static char     s_rel[REC_PATH_MAX];// 资源根相对路径（"music/rec-….mp3"）
static char     s_abs[REC_PATH_MAX];// 宿主绝对路径
static int32_t  s_frames = 0;       // 已编成的帧数
static int32_t  s_bytes = 0;        // 已写出的 MP3 字节数

// 跨帧 PCM：编码器要求"上一帧 + 当前帧"连续传入（offset 指向当前帧开头）。
// 这几块（27KB+2KB）**按需从 HAL 堆分配**而不是放 .bss：ESP32 上 platform_malloc 落在
// PSRAM，静态 .bss 只会吃紧内 RAM（与 am_native_Audio 的工作区同一取舍）。
static double   *s_prev = NULL;
static double   *s_cur = NULL;
static double   *s_win = NULL;

// 最近一帧原始采样（电平表/波形用）与电平
static int16_t  *s_pcm = NULL;
static int32_t  s_level = 0;

// ===========================================================================
// 操作数栈辅助（契约 N1/N6：弹完参数后压 1 值 + step 1 次）
// ===========================================================================

static bool rec_pop_number(am_process_t *proc, double *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;

    if (am_value_is_float(v)) { *out = (double)am_value_to_float(v); return true; }
    if (am_value_is_int(v))   { *out = (double)am_value_to_int(v);   return true; }
    if (am_value_is_uint(v))  { *out = (double)am_value_to_uint(v);  return true; }
    return false;
}

static bool rec_pop_int(am_process_t *proc, int32_t *out) {
    double f = 0.0;
    if (!rec_pop_number(proc, &f)) return false;
    if (isnan(f)) return false;
    if (f > 2147483647.0)  f = 2147483647.0;
    if (f < -2147483648.0) f = -2147483648.0;
    *out = (int32_t)f;
    return true;
}

static int32_t rec_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t rec_push_int(am_process_t *proc, int32_t n) {
    return rec_push(proc, am_make_value_of_int((am_int_t)n));
}

static int32_t rec_push_null(am_process_t *proc) {
    return rec_push(proc, AM_VALUE_NULL);
}

// 压入一个 n 元数值列表（分配失败按软失败压 #null）
static int32_t rec_push_number_list(am_process_t *proc, const double *vals, int32_t n) {
    am_list_t *lst = am_list_create(proc->heap_alloc, (size_t)n, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    am_handle_t lhd;
    if (lst == NULL) return rec_push_null(proc);
    for (int32_t i = 0; i < n; i++) {
        am_list_t *grown = am_list_push(proc->heap_alloc, lst, am_make_value_of_float((am_float_t)vals[i]));
        if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return rec_push_null(proc); }
        lst = grown;
    }
    lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return rec_push_null(proc); }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return rec_push_null(proc);
    }
    if (am_process_push_operand(proc, am_make_value_of_handle(lhd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// 压入 (毫秒 帧数 字节数 路径|#null)：前 3 项数值 + 末项字符串或 #null。
//（(Rec.stop) 专用：收尾结果既有数值也有路径，普通的 number_list 表达不了。）
static int32_t rec_push_stop_list(am_process_t *proc, const double *vals, const char *path) {
    am_list_t *lst = am_list_create(proc->heap_alloc, 4, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    am_handle_t lhd;
    am_handle_t shd = AM_HANDLE_NULL;
    am_value_t lastv = AM_VALUE_NULL;

    if (lst == NULL) return rec_push_null(proc);
    if (path != NULL) {
        size_t blen = strlen(path);
        wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (blen + 1) * sizeof(wchar_t));
        if (wbuf) {
            uint32_t wlen = _mbstowcs(wbuf, path, (uint32_t)blen);
            shd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
            am_free(proc->heap_alloc, wbuf);
        }
        if (shd != AM_HANDLE_NULL) lastv = am_make_value_of_handle(shd);
    }
    for (int32_t i = 0; i < 3; i++) {
        am_list_t *grown = am_list_push(proc->heap_alloc, lst,
                                        am_make_value_of_float((am_float_t)vals[i]));
        if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return rec_push_null(proc); }
        lst = grown;
    }
    {
        am_list_t *grown = am_list_push(proc->heap_alloc, lst, lastv);
        if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return rec_push_null(proc); }
        lst = grown;
    }
    lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return rec_push_null(proc); }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return rec_push_null(proc);
    }
    return rec_push(proc, am_make_value_of_handle(lhd));
}

// 由 UTF-8 缓冲区构造 Scheme 字符串并压栈（失败压 #null）
static int32_t rec_push_string(am_process_t *proc, const char *utf8) {
    size_t len = strlen(utf8);
    wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (len + 1) * sizeof(wchar_t));
    am_handle_t hd = AM_HANDLE_NULL;
    if (wbuf) {
        uint32_t wlen = _mbstowcs(wbuf, utf8, (uint32_t)len);
        hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
        am_free(proc->heap_alloc, wbuf);
    }
    if (hd == AM_HANDLE_NULL) return rec_push_null(proc);
    return rec_push(proc, am_make_value_of_handle(hd));
}

// ===========================================================================
// 工作区（按需分配/释放；失败按软失败处理，不留半初始化状态）
// ===========================================================================

static void rec_free_workspace(void) {
    if (s_prev) { free(s_prev); s_prev = NULL; }
    if (s_cur)  { free(s_cur);  s_cur  = NULL; }
    if (s_win)  { free(s_win);  s_win  = NULL; }
    if (s_pcm)  { free(s_pcm);  s_pcm  = NULL; }
    if (s_enc)  { free(s_enc);  s_enc  = NULL; }
}

static bool rec_alloc_workspace(void) {
    if (s_enc == NULL)  s_enc  = (AquaEncoder *)platform_malloc(sizeof(AquaEncoder));
    if (s_prev == NULL) s_prev = (double *)platform_calloc(REC_FRAME, sizeof(double));
    if (s_cur == NULL)  s_cur  = (double *)platform_calloc(REC_FRAME, sizeof(double));
    if (s_win == NULL)  s_win  = (double *)platform_calloc(2 * REC_FRAME, sizeof(double));
    if (s_pcm == NULL)  s_pcm  = (int16_t *)platform_calloc(REC_FRAME, sizeof(int16_t));
    if (s_enc == NULL || s_prev == NULL || s_cur == NULL || s_win == NULL || s_pcm == NULL) {
        rec_free_workspace();
        return false;
    }
    memset(s_prev, 0, REC_FRAME * sizeof(double));
    memset(s_cur, 0, REC_FRAME * sizeof(double));
    memset(s_pcm, 0, REC_FRAME * sizeof(int16_t));
    return true;
}

// ===========================================================================
// 文件命名与落盘
// ===========================================================================

// path 是否已存在（fopen 探测：不占 platform_file_open 的全局单句柄）
static bool rec_file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return false;
    fclose(f);
    return true;
}

// 确保 music/ 目录存在（已存在视为成功）
static bool rec_ensure_dir(void) {
    char dir[REC_PATH_MAX];
    if (strlen(PLATFORM_ROOT_DIR) + 1 + strlen(REC_MAX_DIR) + 1 > sizeof dir) return false;
    snprintf(dir, sizeof dir, "%s/%s", PLATFORM_ROOT_DIR, REC_MAX_DIR);
    if (platform_is_directory(dir) != 0) return true;
    return platform_mkdir(dir) == 0;
}

// 生成本次录音的文件名：music/rec-YYYYMMDD-HHMMSS.mp3（本地时间；重名追加 -N）
static bool rec_make_path(char *rel, size_t relcap, char *abs, size_t abscap) {
    time_t t = time(NULL);
    struct tm *lt = localtime(&t);
    char stamp[32];
    int32_t n = 0;

    if (lt == NULL) return false;
    snprintf(stamp, sizeof stamp, "%04d%02d%02d-%02d%02d%02d",
             lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
             lt->tm_hour, lt->tm_min, lt->tm_sec);

    for (n = 0; n < 100; n++) {
        if (n == 0)
            snprintf(rel, relcap, "%s/rec-%s.mp3", REC_MAX_DIR, stamp);
        else
            snprintf(rel, relcap, "%s/rec-%s-%d.mp3", REC_MAX_DIR, stamp, n);
        if (strlen(PLATFORM_ROOT_DIR) + 1 + strlen(rel) + 1 > abscap) return false;
        snprintf(abs, abscap, "%s/%s", PLATFORM_ROOT_DIR, rel);
        if (!rec_file_exists(abs)) return true;
    }
    return false;
}

// 把整份录音交回"落盘钩子"（WASM：写进 IndexedDB，刷新页面后仍由 loader 挂载回来）。
// 只有钩子存在时才读回——其它平台零开销，也不占瞬时内存。
static void rec_persist(const char *abs) {
#if defined(__GNUC__)
    FILE *f;
    long sz;
    uint8_t *buf;

    if (!am_native_fs_on_write) return;
    f = fopen(abs, "rb");
    if (f == NULL) return;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return; }
    sz = ftell(f);
    if (sz <= 0 || sz > (64L * 1024L * 1024L)) { fclose(f); return; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return; }
    buf = (uint8_t *)platform_malloc((size_t)sz);
    if (buf == NULL) { fclose(f); return; }
    if (fread(buf, 1, (size_t)sz, f) == (size_t)sz) {
        am_native_fs_on_write(abs, buf, (size_t)sz);
    }
    free(buf);
    fclose(f);
#else
    (void)abs;
#endif
}

// ===========================================================================
// 收尾（(Rec.stop) 与硬错误共用）
// ===========================================================================

// 关麦 + 关文件 + 清状态，并把最终帧数/字节数经 out 回填（调用方先存好副本也行）。
// 一帧都没编成时删掉空文件（不留 0 字节残骸）；编成过则交给落盘钩子。
static void rec_finish(int32_t *out_frames, int32_t *out_bytes) {
    int32_t frames = s_frames;
    int32_t bytes = s_bytes;

    if (s_fp != NULL) {
        fclose(s_fp);          // 库缓冲在 fclose 时冲刷到文件
        s_fp = NULL;
    }
    if (s_state) {
        mic_close();
        s_state = 0;
    }
    if (frames > 0) {
        rec_persist(s_abs);
    }
    else if (s_abs[0] != '\0') {
        remove(s_abs);         // 没录到任何东西：不留 0 字节的残骸
    }
    s_rel[0] = '\0';          // 本次录音的路径到此失效（调用方已先取走副本）
    s_abs[0] = '\0';
    s_frames = 0;
    s_bytes = 0;
    s_level = 0;
    rec_free_workspace();      // 编码器上下文与 PCM 缓冲随录音一起释放

    if (out_frames) *out_frames = frames;
    if (out_bytes) *out_bytes = bytes;
}

// ===========================================================================
// 编码流水线
// ===========================================================================

// 时域电平（RMS）→ 0~255（与 am_native_Audio.c 同一映射：−60dBFS…0dBFS）
static void rec_update_level(void) {
    float acc = 0.0f;
    for (int32_t i = 0; i < REC_FRAME; i++) {
        float s = (float)s_pcm[i];
        acc += s * s;
    }
    {
        float rms = sqrtf(acc / (float)REC_FRAME);
        float db = 20.0f * log10f((rms + 1.0f) / 32768.0f);
        float lv = (db + 60.0f) * (255.0f / 60.0f);
        if (lv < 0.0f) lv = 0.0f;
        if (lv > 255.0f) lv = 255.0f;
        s_level = (int32_t)lv;
    }
}

// 读一帧 → 编码 → 追加写文件。返回 1 = 编成一帧 / 0 = 数据不足（不消费）/ -1 = 硬错误。
static int32_t rec_encode_frame(void) {
    int32_t n;
    int32_t nbytes;
    AquaBitsWriter bw;
    const double *pcm[2];

    if (!s_state || s_fp == NULL || s_enc == NULL || s_pcm == NULL) return -1;

    // 只取一帧：WASM 的 mic_read 在数据不足时立即返回 0 且**不消费**任何采样
    //（见 hal_audio_in_wasm.c），ALSA 则阻塞至就绪或 100ms 超时。
    n = mic_read(s_pcm, REC_FRAME);
    if (n < 0) return -1;
    if (n < REC_FRAME) return 0;

    rec_update_level();

    // int16 → [-1,1] 浮点 PCM，与 Aqua CLI（aqua_wav.c）的换算一致
    for (int32_t i = 0; i < REC_FRAME; i++) {
        s_cur[i] = (double)s_pcm[i] / 32768.0;
    }
    memcpy(s_win, s_prev, REC_FRAME * sizeof(double));
    memcpy(s_win + REC_FRAME, s_cur, REC_FRAME * sizeof(double));
    pcm[0] = s_win;
    pcm[1] = s_win;            // 单声道：编码器只看 pcm[0]，这里给同一个指针即可

    aqua_bw_init(&bw);
    if (!aqua_encoder_encode_frame(s_enc, pcm, 2 * REC_FRAME, REC_FRAME, &bw)) {
        aqua_bw_free(&bw);
        return -1;
    }
    nbytes = aqua_bw_nbytes(&bw);
    if (nbytes > 0 && fwrite(aqua_bw_packed(&bw), 1, (size_t)nbytes, s_fp) != (size_t)nbytes) {
        aqua_bw_free(&bw);
        return -1;
    }
    aqua_bw_free(&bw);

    s_bytes += nbytes;
    s_frames += 1;

    // 每 8 帧（≈209ms）刷一次库缓冲：录音期间"已经落盘"的窗口不超过 200ms，
    // 即使进程被强杀（关闭页面）也只丢最后不足 0.2 秒。逐帧 fflush 对 SD 卡
    // 是 38 次/秒的小写，不值；8 帧一批足够。
    if ((s_frames & 7) == 0 && fflush(s_fp) != 0) return -1;

    // 当前帧成为下一帧的"上一帧"（编码器要求逐帧连续，见 aqua_encoder.h）
    memcpy(s_prev, s_cur, REC_FRAME * sizeof(double));
    return 1;
}

// ===========================================================================
// Native 函数实现
// ===========================================================================

int32_t am_native_Rec_info(am_runtime_t *rt, am_process_t *proc) {
    double vals[4];
    (void)rt;
    vals[0] = (double)REC_RATE;
    vals[1] = (double)REC_CHANNELS;
    vals[2] = (double)REC_KBPS;
    vals[3] = (double)REC_FRAME;
    return rec_push_number_list(proc, vals, 4);
}

int32_t am_native_Rec_start(am_runtime_t *rt, am_process_t *proc) {
    uint8_t vol;
    (void)rt;

    if (s_state) return rec_push_null(proc);          // 已在录音：软失败

    if (!rec_ensure_dir()) return rec_push_null(proc);
    if (!rec_make_path(s_rel, sizeof s_rel, s_abs, sizeof s_abs)) return rec_push_null(proc);
    if (!rec_alloc_workspace()) {
        s_rel[0] = '\0';
        s_abs[0] = '\0';
        return rec_push_null(proc);                   // 内存不足：App 显示降级画面
    }

    vol = audio_out_get_master_volume();
    if (mic_init((uint32_t)REC_RATE, vol) != 0) {
        s_rel[0] = '\0';
        s_abs[0] = '\0';
        return rec_push_null(proc);                   // 没有麦克风：App 显示降级画面
    }

    s_fp = fopen(s_abs, "wb");
    if (s_fp == NULL) {
        mic_close();
        rec_free_workspace();
        s_rel[0] = '\0';
        s_abs[0] = '\0';
        return rec_push_null(proc);
    }

    if (!aqua_encoder_init(s_enc, REC_CHANNELS, SAMPLE_RATE_44100, BIT_RATE_320K) ||
        !aqua_encoder_reset(s_enc, REC_CHANNELS, SAMPLE_RATE_44100, BIT_RATE_320K)) {
        fclose(s_fp);
        s_fp = NULL;
        mic_close();
        remove(s_abs);
        rec_free_workspace();
        s_rel[0] = '\0';
        s_abs[0] = '\0';
        return rec_push_null(proc);
    }

    memset(s_prev, 0, REC_FRAME * sizeof(double));    // 首帧前补 1152 个零（与 CLI 一致）
    memset(s_cur, 0, REC_FRAME * sizeof(double));
    memset(s_pcm, 0, REC_FRAME * sizeof(int16_t));
    s_frames = 0;
    s_bytes = 0;
    s_level = 0;
    s_state = 1;

    return rec_push_string(proc, s_rel);
}

int32_t am_native_Rec_pump(am_runtime_t *rt, am_process_t *proc) {
    int32_t want = 1;
    int32_t done = 0;
    (void)rt;

    if (!rec_pop_int(proc, &want)) return -1;
    if (want < 1) want = 1;
    if (want > REC_PUMP_MAX) want = REC_PUMP_MAX;

    if (!s_state) return rec_push_int(proc, 0);

    while (done < want) {
        int32_t rc = rec_encode_frame();
        if (rc == 1) {
            done += 1;
            continue;
        }
        if (rc < 0) {
            // 硬错误（权限被拒/采集中断/写盘失败）：自行收尾，报 -1
            rec_finish(NULL, NULL);
            return rec_push_int(proc, -1);
        }
        break;                                        // 数据不足：本拍到此为止
    }
    return rec_push_int(proc, done);
}

int32_t am_native_Rec_stop(am_runtime_t *rt, am_process_t *proc) {
    double vals[3];
    int32_t frames = 0;
    int32_t bytes = 0;
    bool have_path = (s_abs[0] != '\0');
    char path[REC_PATH_MAX];
    (void)rt;

    path[0] = '\0';
    if (have_path) {
        strncpy(path, s_rel, sizeof path - 1);
        path[sizeof path - 1] = '\0';
    }
    rec_finish(&frames, &bytes);

    vals[0] = (double)frames * REC_FRAME_MS;
    vals[1] = (double)frames;
    vals[2] = (double)bytes;
    return rec_push_stop_list(proc, vals, (frames > 0 && path[0] != '\0') ? path : NULL);
}

int32_t am_native_Rec_stats(am_runtime_t *rt, am_process_t *proc) {
    double vals[4];
    (void)rt;
    vals[0] = (double)(s_state ? 1 : 0);
    vals[1] = (double)s_frames * REC_FRAME_MS;
    vals[2] = (double)s_frames;
    vals[3] = (double)s_bytes;
    return rec_push_number_list(proc, vals, 4);
}

int32_t am_native_Rec_level(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return rec_push_int(proc, s_level);
}

int32_t am_native_Rec_wave(am_runtime_t *rt, am_process_t *proc) {
    int32_t idx = 0;
    (void)rt;
    if (!rec_pop_int(proc, &idx)) return -1;
    if (s_pcm == NULL || idx < 0 || idx >= REC_FRAME) return rec_push_int(proc, 0);
    return rec_push_int(proc, (int32_t)s_pcm[idx]);
}

// ===========================================================================
// 库注册表
// ===========================================================================

static const am_native_func_entry_t am_native_Rec_funcs[] = {
    { L"info",  am_native_Rec_info },
    { L"start", am_native_Rec_start },
    { L"pump",  am_native_Rec_pump },
    { L"stop",  am_native_Rec_stop },
    { L"stats", am_native_Rec_stats },
    { L"level", am_native_Rec_level },
    { L"wave",  am_native_Rec_wave },
};

const am_native_lib_entry_t am_native_Rec_lib = {
    L"Rec",
    am_native_Rec_funcs,
    sizeof(am_native_Rec_funcs) / sizeof(am_native_Rec_funcs[0])
};
