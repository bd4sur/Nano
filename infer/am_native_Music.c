////////////////////////////////////////////////////////////////////////////
//  Native Library : Music —— 设计说明见 am_native_Music.h
//
//  分层（与 C 版音乐盒 ui_musicbox.c 的播放器段落同构，但**独立实现**，
//  不改动既有模块）：
//
//      Scheme App  ──(Music.pump/open/seek/...)
//                        │
//                        ├── 解码器（WAV chunk 解析 / minimp3 流式）
//                        │      统一输出 int16 单声道 PCM
//                        ├── 乒乓双缓冲 + 背压（audio_out_queue_free → enqueue）
//                        └── hal_fs 随机访问（平台无关：MEMFS / SD / 真实文件系统）
//
//  本文件只做「资源路径解析 → 打开 → 解码 → 喂 HAL → 报告标量状态」，
//  文件系统与扬声器全部经 HAL（hal_fs / hal_audio_out），因此桌面 / WASM /
//  掌机三处行为一致，无平台分支（唯一的平台差异是 WASM 的 p4 资源挂载请求，
//  由 __EMSCRIPTEN__ 下的 EM_JS 完成，其余平台为空操作）。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "am_native_Music.h"

#include "am_native_Fs.h" // 路径解析复用 Fs 本地库的规则（相对资源根 / 绝对 / 拒绝 "..")

#include "hal_audio_out.h"
#include "hal_fs.h"
#include "hal_ram.h"
#include "platform.h"

#include "vendor/minimp3.h"

// ---------------------------------------------------------------------------
// minimp3 实现单元（ui_musicbox_mp3.c）导出的外部 scratch 注入接口。
// 该工作区约 16KB，放栈上会撑爆渲染任务栈（ESP32 12KB），故由实现单元统一注入。
// 依赖：链接时须包含 ui_musicbox_mp3.c（GUI 目标与各掌机目标均已包含）。
// ---------------------------------------------------------------------------
extern void     minimp3_set_scratch(void *scratch);
extern uint32_t minimp3_scratch_size(void);

// ---------------------------------------------------------------------------
// WASM-GUI：请求页面把 manifest 的 p4 层（music/ 音频，~90MB）挂载进 MEMFS。
// 页面未提供该函数时（node 回归台 / 其它平台）为空操作，引擎按"空目录"降级。
// ---------------------------------------------------------------------------
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
EM_JS(void, music_request_tier_js, (void), {
    if (typeof nanoRequestMusicTier === "function") {
        try { nanoRequestMusicTier(); } catch (e) { /* 忽略 */ }
    }
});
#else
#define music_request_tier_js() ((void)0)
#endif

// ===========================================================================
// 参数与状态
// ===========================================================================

// 每块单声道 PCM 采样数（@44.1kHz ≈ 93ms；双槽队列 ≈ 0.19s 余量）
#define MUSIC_PCM_BLOCK (4096)
// 文件字节暂存容量（WAV 读块 / MP3 输入流共用；须 ≥ PCM_BLOCK × 4 字节）
#define MUSIC_SCRATCH_CAP (16384)
// 解码前最小输入水位（minimp3 需"完整一帧 + 下一帧头"做交叉验证，见 ui_musicbox.c 注释）
#define MUSIC_MP3_MIN_BUF (2560)
// 音量域（与 C 版音乐盒一致：0~128）
#define MUSIC_VOL_MAX (128)
// 每拍最多连续投放的块数（队列前瞻 0.3s / 一块 93ms → 3 块足够补满）
#define MUSIC_PUMP_MAX_BLOCKS (4)

#define MUSIC_TYPE_WAV (1)
#define MUSIC_TYPE_MP3 (2)

// (Music.pump) 的返回值
#define MUSIC_PUMP_OK    (0)
#define MUSIC_PUMP_EOF   (1)
#define MUSIC_PUMP_IDLE  (2)
#define MUSIC_PUMP_ERROR (-1)

typedef struct {
    int32_t  type;        // MUSIC_TYPE_WAV / MUSIC_TYPE_MP3
    int32_t  playing;     // 1=播放中（喂数据）/ 0=暂停
    int32_t  eof;         // 已读到文件尾（再次 pump 直接返回 EOF）
    int32_t  silent;      // 1=无音频设备（静音走带：解码后丢弃，按墙钟推进）
    uint32_t sample_rate;
    uint32_t file_size;
    uint64_t clock_ms;    // 静音走带的下一块应播放时刻（ms）

    // WAV
    uint32_t wav_data_start;
    uint32_t wav_data_total;
    uint32_t wav_data_left;
    uint16_t wav_channels;
    uint16_t wav_bits;

    // MP3
    uint32_t mp3_consumed;     // 已消费的**音频区**字节数（不含 ID3v2 标签，见 mp3_skip_id3）
    uint32_t mp3_data_start;   // 音频区起点（跳过 ID3v2 标签后的文件偏移）
    mp3dec_t mp3d;
    uint32_t mp3_in_start;
    uint32_t mp3_in_len;
    int32_t  mp3_file_eof;
    uint32_t mp3_frame_len;
    uint32_t mp3_frame_pos;
    int32_t  mp3_channels;
    int32_t  mp3_bitrate_kbps;
    int16_t  mp3_frame[MINIMP3_MAX_SAMPLES_PER_FRAME];
} Music_Decoder;

static Music_Decoder *s_dec = NULL;                            // 当前曲目（未打开为 NULL）
static int16_t *s_pcm[AUDIO_OUT_QUEUE_DEPTH] = {NULL, NULL};   // 乒乓双缓冲（enqueue 引用须常驻）
static int32_t  s_fill = 0;                                    // 下一个待填充缓冲序号
static uint8_t *s_scratch = NULL;                              // 文件字节暂存
static void    *s_mp3_scratch = NULL;                          // minimp3 工作区
static uint8_t  s_volume = 64;                                 // 播放音量（0~128）
static int32_t  s_volume_init = 0;                             // 是否已从主音量取过初值
static int32_t  s_audio_on = 0;                                // audio_out 是否已 init（对账 close）

// 时钟来源：**解释器运行时的 now_ms**（宿主注入），而不是平台墙钟。
//   · 与 (System.timestamp) 同一时基，App 侧算出来的时间与引擎内部推进一致；
//   · 回归台可以注入虚拟时钟，于是"静音走带"的进度推进也是可复现的；
//   · ESP32/桌面/浏览器三处都由各自宿主的 vtable 提供，本库不直接依赖 hal_os。
static am_runtime_t *s_rt = NULL;

static uint64_t music_now_ms(void) {
    return (s_rt != NULL) ? (uint64_t)am_runtime_now_ms(s_rt) : 0;
}

// ===========================================================================
// 小端读取
// ===========================================================================
static uint16_t rd16le(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32le(const uint8_t *p) {
    return (uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24));
}

static int32_t ext_is(const char *path, const char *ext) {
    size_t len = strlen(path);
    size_t el  = strlen(ext);
    if (len < el) return 0;
    return strcasecmp(path + len - el, ext) == 0;
}

// ===========================================================================
// 操作数栈辅助（契约 N1/N6：弹完参数后压 1 值 + step 1 次）
// ===========================================================================

static bool music_pop_number(am_process_t *proc, double *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;

    if (am_value_is_float(v)) { *out = (double)am_value_to_float(v); return true; }
    if (am_value_is_int(v))   { *out = (double)am_value_to_int(v);   return true; }
    if (am_value_is_uint(v))  { *out = (double)am_value_to_uint(v);  return true; }
    return false;
}

// 从操作数栈弹出一个字符串对象（模式同 am_native_Fs.c）
static bool music_pop_wstring(am_process_t *proc, am_wstring_t **out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return false;

    *out = (am_wstring_t *)obj;
    return true;
}

static int32_t music_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t music_push_bool(am_process_t *proc, bool b) {
    return music_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

static int32_t music_push_null(am_process_t *proc) {
    return music_push(proc, AM_VALUE_NULL);
}

static int32_t music_push_number(am_process_t *proc, double x) {
    return music_push(proc, am_make_value_of_float((am_float_t)x));
}

// 由 UTF-8 缓冲区构造 Scheme 字符串把柄（失败返回 AM_HANDLE_NULL）
static am_handle_t music_make_string_handle(am_process_t *proc, const char *utf8, size_t len) {
    wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (len + 1) * sizeof(wchar_t));
    am_handle_t hd;
    if (!wbuf) return AM_HANDLE_NULL;
    uint32_t wlen = _mbstowcs(wbuf, utf8, (uint32_t)len);
    hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
    am_free(proc->heap_alloc, wbuf);
    return hd;
}

// 压入一个 Scheme 字符串（分配失败按软失败压 #null）
static int32_t music_push_string(am_process_t *proc, const char *utf8, size_t len) {
    am_handle_t hd = music_make_string_handle(proc, utf8, len);
    if (hd == AM_HANDLE_NULL) return music_push_null(proc);
    return music_push(proc, am_make_value_of_handle(hd));
}

// 压入一个 n 元数值列表（分配失败压 #null）
static int32_t music_push_number_list(am_process_t *proc, const double *vals, int32_t n) {
    am_list_t *lst = am_list_create(proc->heap_alloc, (size_t)n, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    am_handle_t lhd;
    if (lst == NULL) return music_push_null(proc);
    for (int32_t i = 0; i < n; i++) {
        am_list_t *grown = am_list_push(proc->heap_alloc, lst, am_make_value_of_float((am_float_t)vals[i]));
        if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); return music_push_null(proc); }
        lst = grown;
    }
    lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return music_push_null(proc); }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return music_push_null(proc);
    }
    if (am_process_push_operand(proc, am_make_value_of_handle(lhd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// ===========================================================================
// WAV 解码器（RIFF chunk 遍历；PCM 8/16bit、单/双声道）
// ===========================================================================

static int32_t wav_open(Music_Decoder *d) {
    uint8_t hdr[12];
    uint32_t pos = 12;
    int32_t got_fmt = 0;

    if (platform_file_read(hdr, 12) != 12) return -1;
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) return -1;

    while (pos + 8 <= d->file_size) {
        uint8_t ch[8];
        uint32_t csize;

        if (platform_file_seek(pos) != 0) return -1;
        if (platform_file_read(ch, 8) != 8) return -1;
        pos += 8;
        csize = rd32le(ch + 4);

        if (memcmp(ch, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (csize < 16 || platform_file_read(fmt, 16) != 16) return -1;
            if (rd16le(fmt) != 1) return -1; // 仅 PCM
            d->wav_channels = rd16le(fmt + 2);
            d->sample_rate  = rd32le(fmt + 4);
            d->wav_bits     = rd16le(fmt + 14);
            if ((d->wav_channels != 1 && d->wav_channels != 2) ||
                (d->wav_bits != 8 && d->wav_bits != 16) ||
                d->sample_rate < 4000 || d->sample_rate > 96000) return -1;
            got_fmt = 1;
        }
        else if (memcmp(ch, "data", 4) == 0) {
            if (!got_fmt) return -1;
            d->wav_data_start = pos;
            d->wav_data_total = csize;
            d->wav_data_left  = csize;
            platform_file_seek(pos);
            return 0;
        }
        pos += csize + (csize & 1); // RIFF chunk 偶数字节对齐
    }
    return -1;
}

// 解码出 max_samples 个单声道 int16，返回采样数（0=EOF）
static uint32_t wav_read(Music_Decoder *d, int16_t *pcm, uint32_t max_samples) {
    uint32_t bpf = (uint32_t)d->wav_channels * (uint32_t)(d->wav_bits / 8);
    uint32_t want = max_samples * bpf;
    uint32_t frames;
    int32_t n;

    if (bpf == 0) return 0;
    if (want > MUSIC_SCRATCH_CAP) want = MUSIC_SCRATCH_CAP;
    if (want > d->wav_data_left) want = d->wav_data_left;
    want -= want % bpf;
    if (want == 0) return 0;

    n = platform_file_read(s_scratch, want);
    if (n <= 0) return 0;
    d->wav_data_left -= (uint32_t)n;
    frames = (uint32_t)n / bpf;

    if (d->wav_bits == 16) {
        const int16_t *s = (const int16_t *)s_scratch;
        if (d->wav_channels == 2) {
            for (uint32_t i = 0; i < frames; i++)
                pcm[i] = (int16_t)(((int32_t)s[2 * i] + (int32_t)s[2 * i + 1]) / 2);
        }
        else {
            memcpy(pcm, s, frames * sizeof(int16_t));
        }
    }
    else { // 8bit 无符号 PCM：偏置 128
        const uint8_t *s = s_scratch;
        if (d->wav_channels == 2) {
            for (uint32_t i = 0; i < frames; i++)
                pcm[i] = (int16_t)(((int32_t)s[2 * i] + (int32_t)s[2 * i + 1] - 256) << 7);
        }
        else {
            for (uint32_t i = 0; i < frames; i++)
                pcm[i] = (int16_t)(((int32_t)s[i] - 128) << 8);
        }
    }
    return frames;
}

// ===========================================================================
// MP3 解码器（minimp3 流式：16KB 输入流 + 逐帧解码）
// ===========================================================================

// 把剩余数据移到 scratch 头部并从文件补满；返回补入字节数（0=补不进）
static int32_t mp3_fill_input(Music_Decoder *d) {
    int32_t n;
    if (d->mp3_in_start > 0) {
        memmove(s_scratch, s_scratch + d->mp3_in_start, d->mp3_in_len);
        d->mp3_in_start = 0;
    }
    if (d->mp3_file_eof) return 0;
    n = platform_file_read(s_scratch + d->mp3_in_len, MUSIC_SCRATCH_CAP - d->mp3_in_len);
    if (n <= 0) { d->mp3_file_eof = 1; return 0; }
    d->mp3_in_len += (uint32_t)n;
    return n;
}

// 跳过 ID3v2 标签（"ID3" + 4 字节 synchsafe 长度 + 10 字节头）。
// 为什么值得做：① 标签（常见是专辑封面，动辄几百 KB）会被计入 mp3_consumed 与
// 打开时的帧头扫描，导致进度条一开始就跳到 tag/文件大小 的位置；② C 版音乐盒
// 的 64 次 × 16KB 扫描上限使 >1MB 的标签直接打不开，跳过之后没有这个限制。
static void mp3_skip_id3(Music_Decoder *d) {
    uint8_t hdr[10];

    d->mp3_consumed = 0;
    d->mp3_data_start = 0;
    if (platform_file_read(hdr, 10) == 10 && memcmp(hdr, "ID3", 3) == 0) {
        uint32_t sz = ((uint32_t)(hdr[6] & 0x7F) << 21) | ((uint32_t)(hdr[7] & 0x7F) << 14)
                    | ((uint32_t)(hdr[8] & 0x7F) << 7)  |  (uint32_t)(hdr[9] & 0x7F);
        if (sz + 10 < d->file_size) d->mp3_data_start = sz + 10;
    }
    platform_file_seek(d->mp3_data_start);
}

// 打开并预解首帧（确定采样率/声道/码率；首帧留在 mp3_frame 中供 read 消费）
static int32_t mp3_open(Music_Decoder *d) {
    mp3dec_init(&d->mp3d);
    d->mp3_in_start = 0;
    d->mp3_in_len = 0;
    d->mp3_file_eof = 0;
    d->mp3_frame_len = 0;
    d->mp3_frame_pos = 0;
    mp3_skip_id3(d);   // 定位到第一个音频帧（顺带把标签排除在进度统计之外）

    for (int32_t tries = 0; tries < 64; tries++) { // 限次扫描，防坏文件死循环
        mp3dec_frame_info_t info;
        int samples;

        if (d->mp3_in_len < MUSIC_SCRATCH_CAP && !d->mp3_file_eof) mp3_fill_input(d);
        if (d->mp3_in_len == 0) return -1;

        samples = mp3dec_decode_frame(&d->mp3d, s_scratch + d->mp3_in_start,
                                     (int)d->mp3_in_len, d->mp3_frame, &info);
        d->mp3_in_start += info.frame_bytes;
        d->mp3_in_len   -= info.frame_bytes;
        d->mp3_consumed += info.frame_bytes;
        if (samples > 0) {
            d->mp3_frame_len = (uint32_t)samples;
            d->mp3_channels  = info.channels;
            d->sample_rate   = (uint32_t)info.hz;
            d->mp3_bitrate_kbps = info.bitrate_kbps;
            return 0;
        }
        if (info.frame_bytes == 0 && d->mp3_file_eof) return -1;
    }
    return -1;
}

// 解码出 max_samples 个单声道 int16，返回采样数（0=EOF）
static uint32_t mp3_read(Music_Decoder *d, int16_t *pcm, uint32_t max_samples) {
    uint32_t out = 0;

    while (out < max_samples) {
        if (d->mp3_frame_pos < d->mp3_frame_len) {
            uint32_t avail = d->mp3_frame_len - d->mp3_frame_pos;
            uint32_t take = max_samples - out;
            const int16_t *f = d->mp3_frame;

            if (take > avail) take = avail;
            if (d->mp3_channels == 2) {
                for (uint32_t i = 0; i < take; i++) {
                    uint32_t k = (d->mp3_frame_pos + i) * 2;
                    pcm[out + i] = (int16_t)(((int32_t)f[k] + (int32_t)f[k + 1]) / 2);
                }
            }
            else {
                memcpy(pcm + out, f + d->mp3_frame_pos, take * sizeof(int16_t));
            }
            d->mp3_frame_pos += take;
            out += take;
            continue;
        }
        // 补足输入水位后再解码（避免 decode_frame 因"完整帧+下一帧头"不可见而清零状态，
        // 那会造成输入缓冲边界处的周期性破音；文件尾部不足时按现状解码，无害）
        while (!d->mp3_file_eof && d->mp3_in_len < MUSIC_MP3_MIN_BUF) {
            if (mp3_fill_input(d) <= 0) break;
        }
        {
            mp3dec_frame_info_t info;
            int samples = mp3dec_decode_frame(&d->mp3d, s_scratch + d->mp3_in_start,
                                             (int)d->mp3_in_len, d->mp3_frame, &info);
            d->mp3_in_start += info.frame_bytes;
            d->mp3_in_len   -= info.frame_bytes;
            d->mp3_consumed += info.frame_bytes;

            if (samples > 0) {
                d->mp3_frame_len = (uint32_t)samples;
                d->mp3_frame_pos = 0;
                d->mp3_channels  = info.channels;
                // 码率只用于信息显示与总时长估算：保留首帧码率（VBR 逐帧变化）
                if (d->mp3_bitrate_kbps == 0) d->mp3_bitrate_kbps = info.bitrate_kbps;
                continue;
            }
            if (info.frame_bytes == 0) {
                int32_t got;
                if (d->mp3_file_eof) break;
                got = mp3_fill_input(d);
                if (got <= 0 && d->mp3_in_len == 0) break;
                // 缓冲已满仍找不到帧头（坏数据）：丢弃 1 字节防死循环
                if (d->mp3_in_len == MUSIC_SCRATCH_CAP) {
                    d->mp3_in_start++;
                    d->mp3_in_len--;
                    d->mp3_consumed++;
                }
            }
            // frame_bytes > 0 但 samples == 0：跳过非音频数据（ID3 等），继续
        }
    }
    return out;
}

// ===========================================================================
// 引擎：时长 / 进度 / 喂数据 / 定位
// ===========================================================================

// MP3 的"音频区"字节数（文件尾部可能还有 ID3v1/APE 标签，这里不做精细剔除：
// 尾部标签只影响总时长的千分位量级，而 seek 的尾部余量 0.5% 足以吸收它）
static uint32_t music_mp3_audio_size(void) {
    if (!s_dec) return 0;
    if (s_dec->file_size <= s_dec->mp3_data_start) return 0;
    return s_dec->file_size - s_dec->mp3_data_start;
}

// 每秒消耗的字节数（WAV = 采样率×声道×位深/8；MP3 用首帧码率估算）
static uint32_t music_bytes_per_sec(void) {
    if (!s_dec) return 0;
    if (s_dec->type == MUSIC_TYPE_WAV) {
        return s_dec->sample_rate * (uint32_t)s_dec->wav_channels * (uint32_t)(s_dec->wav_bits / 8);
    }
    return (uint32_t)(s_dec->mp3_bitrate_kbps > 0 ? s_dec->mp3_bitrate_kbps * 125u : 0u);
}

static uint32_t music_elapsed_ms(void) {
    uint32_t bps;
    if (!s_dec) return 0;
    bps = music_bytes_per_sec();
    if (bps == 0) return 0;
    if (s_dec->type == MUSIC_TYPE_WAV) {
        uint32_t used = (s_dec->wav_data_total >= s_dec->wav_data_left)
                      ? (s_dec->wav_data_total - s_dec->wav_data_left) : 0;
        return (uint32_t)((uint64_t)used * 1000u / bps);
    }
    return (uint32_t)((uint64_t)s_dec->mp3_consumed * 1000u / bps);
}

static uint32_t music_duration_ms(void) {
    uint32_t bps;
    uint32_t bytes;
    if (!s_dec) return 0;
    bps = music_bytes_per_sec();
    if (bps == 0) return 0;
    bytes = (s_dec->type == MUSIC_TYPE_WAV) ? s_dec->wav_data_total : music_mp3_audio_size();
    return (uint32_t)((uint64_t)bytes * 1000u / bps);
}

static double music_progress(void) {
    if (!s_dec) return 0.0;
    if (s_dec->type == MUSIC_TYPE_WAV) {
        if (s_dec->wav_data_total == 0) return 0.0;
        return 1.0 - (double)s_dec->wav_data_left / (double)s_dec->wav_data_total;
    }
    {
        uint32_t audio = music_mp3_audio_size();
        if (audio == 0) return 0.0;
        return (double)s_dec->mp3_consumed / (double)audio;
    }
}

static void music_stop_decoder(void) {
    if (s_dec != NULL) {
        platform_file_close();
        free(s_dec);
        s_dec = NULL;
    }
}

// 释放播放态全部资源（解码器 / 缓冲 / audio_out；音量与"已初始化"标志保留）
static void music_player_cleanup(void) {
    if (s_audio_on) {
        audio_out_stop();
        audio_out_close();
        s_audio_on = 0;
    }
    music_stop_decoder();
    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++) {
        if (s_pcm[i] != NULL) { free(s_pcm[i]); s_pcm[i] = NULL; }
    }
    if (s_scratch != NULL) { free(s_scratch); s_scratch = NULL; }
    if (s_mp3_scratch != NULL) {
        minimp3_set_scratch(NULL);
        free(s_mp3_scratch);
        s_mp3_scratch = NULL;
    }
    s_fill = 0;
}

// 分配播放缓冲并注入 minimp3 scratch。返回 true 成功。
static bool music_alloc_buffers(void) {
    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++) {
        if (s_pcm[i] == NULL)
            s_pcm[i] = (int16_t *)platform_calloc(MUSIC_PCM_BLOCK, sizeof(int16_t));
    }
    if (s_scratch == NULL) s_scratch = (uint8_t *)platform_calloc(MUSIC_SCRATCH_CAP, 1);
    if (s_mp3_scratch == NULL) s_mp3_scratch = platform_calloc(1, minimp3_scratch_size());

    if (s_pcm[0] == NULL || s_pcm[1] == NULL || s_scratch == NULL || s_mp3_scratch == NULL) {
        music_player_cleanup();
        return false;
    }
    minimp3_set_scratch(s_mp3_scratch);
    return true;
}

// 静音走带的下一块节奏：把已解码块的时长累加到虚拟时钟
static void music_advance_clock(uint32_t samples) {
    uint64_t now = music_now_ms();
    uint32_t ms;
    if (s_dec->sample_rate == 0) return;
    ms = (uint32_t)((uint64_t)samples * 1000u / s_dec->sample_rate);
    if (ms == 0) ms = 1;
    if (s_dec->clock_ms < now) s_dec->clock_ms = now; // 落后就重基，不追帧
    s_dec->clock_ms += ms;
}

// 解码一块并投入扬声器队列（无音频设备时丢弃，按墙钟推进）。
// 返回 1=已推进 / 0=EOF（无更多数据）/ -1=投放失败
static int32_t music_pump_block(void) {
    int16_t *pcm = s_pcm[s_fill];
    uint32_t n = (s_dec->type == MUSIC_TYPE_WAV)
               ? wav_read(s_dec, pcm, MUSIC_PCM_BLOCK)
               : mp3_read(s_dec, pcm, MUSIC_PCM_BLOCK);
    if (n == 0) return 0;
    if (s_dec->silent) {
        music_advance_clock(n);
        return 1;
    }
    if (audio_out_enqueue(pcm, n) == 0) {
        s_fill ^= 1;
        return 1;
    }
    return -1;
}

// 定位到比例 ratio（0.0~1.0）
static int32_t music_seek_ratio(double ratio) {
    if (!s_dec) return -1;
    if (ratio != ratio) ratio = 0.0; // NaN
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 0.995) ratio = 0.995; // 留尾部余量，避免 seek 到 EOF 立刻结束

    if (!s_dec->silent) audio_out_stop();
    s_dec->eof = 0;

    if (s_dec->type == MUSIC_TYPE_WAV) {
        uint32_t bpf = (uint32_t)s_dec->wav_channels * (uint32_t)(s_dec->wav_bits / 8);
        uint32_t delta;
        if (bpf == 0) return -1;
        delta = (uint32_t)((double)s_dec->wav_data_total * ratio);
        delta -= delta % bpf; // 对齐到采样帧边界
        if (delta > s_dec->wav_data_total) delta = s_dec->wav_data_total;
        if (platform_file_seek(s_dec->wav_data_start + delta) != 0) return -1;
        s_dec->wav_data_left = s_dec->wav_data_total - delta;
    }
    else {
        // 比例按**音频区**（跳过 ID3v2 标签）计算：否则小比例会落进标签内，
        // 解码器要一路扫过剩余标签才找到帧头，进度条会凭空跳一大截。
        uint32_t audio = music_mp3_audio_size();
        uint32_t off = (uint32_t)((double)audio * ratio);
        if (off > audio) off = audio;
        if (platform_file_seek(s_dec->mp3_data_start + off) != 0) return -1;
        mp3dec_init(&s_dec->mp3d);
        s_dec->mp3_in_start = 0;
        s_dec->mp3_in_len = 0;
        s_dec->mp3_file_eof = 0;
        s_dec->mp3_consumed = off;
        s_dec->mp3_frame_len = 0;
        s_dec->mp3_frame_pos = 0;
    }
    s_dec->clock_ms = music_now_ms();
    s_fill = 0;
    return 0;
}

// ===========================================================================
// 打开 / 关闭曲目
// ===========================================================================

// 打开 path 并从当前位置起播放（含预填播放队列）。返回 true 成功。
static bool music_start_track(const char *path) {
    Music_Decoder *d;
    int32_t rc;

    music_player_cleanup();
    if (!music_alloc_buffers()) return false;

    d = (Music_Decoder *)platform_calloc(1, sizeof(Music_Decoder));
    if (d == NULL) { music_player_cleanup(); return false; }
    s_dec = d;

    if (platform_file_open(path) != 0) { music_player_cleanup(); return false; }
    d->file_size = platform_file_size();

    if (ext_is(path, ".wav")) {
        d->type = MUSIC_TYPE_WAV;
        rc = wav_open(d);
    }
    else {
        d->type = MUSIC_TYPE_MP3;
        rc = mp3_open(d);
    }
    if (rc != 0 || d->sample_rate == 0) { music_player_cleanup(); return false; }

    // 首次播放取主音量作初值（clamp 到音乐盒域 0~128），之后由 App 掌管
    if (!s_volume_init) {
        uint8_t master = audio_out_get_master_volume();
        s_volume = (master > MUSIC_VOL_MAX) ? (uint8_t)MUSIC_VOL_MAX : master;
        s_volume_init = 1;
    }

    // 采样率随曲目更新：上面的 music_player_cleanup() 已经 close 过（恢复进入前音量），
    // 这里只做 init（重新保存当前音量并设定新采样率），与 C 版音乐盒的 close→init 序列等价。
    // init 失败（无音频设备 / 打桩平台）则转入静音走带。
    if (audio_out_init(d->sample_rate, s_volume) == 0) {
        s_audio_on = 1;
        d->silent = 0;
    }
    else {
        d->silent = 1;
    }
    d->playing = 1;
    d->eof = 0;
    d->clock_ms = music_now_ms();
    s_fill = 0;

    // 预填播放队列的全部空槽（一块在播、一块待播），立即开始出声
    if (!d->silent) {
        music_pump_block();
        if (audio_out_queue_free()) music_pump_block();
    }
    printf("[music] open %s (%lu Hz, %ldch, %s)\n", path, (unsigned long)d->sample_rate,
           (long)(d->type == MUSIC_TYPE_WAV ? d->wav_channels : d->mp3_channels),
           d->silent ? "silent" : "audio");
    return true;
}

// ===========================================================================
// 目录枚举
// ===========================================================================

// music/ 目录下的 .wav/.mp3 文件名（platform_malloc 分配，调用方逐项 free + free 数组）
static char **music_list_names(int32_t *out_n) {
    const char *dir = PLATFORM_ROOT_DIR "/music";
    int32_t total = list_files(dir, NULL);
    char **names;
    char **out;
    int32_t n = 0;

    *out_n = 0;
    if (total <= 0) return NULL;

    names = (char **)platform_calloc((size_t)total, sizeof(char *));
    if (names == NULL) return NULL;
    if (list_files(dir, names) < 0) { free(names); return NULL; }

    out = (char **)platform_calloc((size_t)total, sizeof(char *));
    if (out == NULL) {
        for (int32_t i = 0; i < total; i++) free(names[i]);
        free(names);
        return NULL;
    }

    for (int32_t i = 0; i < total; i++) {
        if (names[i] == NULL) continue;
        if (!ext_is(names[i], ".wav") && !ext_is(names[i], ".mp3")) { free(names[i]); continue; }
        out[n++] = names[i]; // 接管所有权
    }
    free(names);

    // 按名字升序（插入排序：目录条目通常只有几十个；UTF-8 字节序即码点序）
    for (int32_t i = 1; i < n; i++) {
        char *key = out[i];
        int32_t j = i - 1;
        while (j >= 0 && strcmp(out[j], key) > 0) { out[j + 1] = out[j]; j--; }
        out[j + 1] = key;
    }
    *out_n = n;
    return out;
}

// ===========================================================================
// Native 函数实现
// ===========================================================================

int32_t am_native_Music_list(am_runtime_t *rt, am_process_t *proc) {
    int32_t n = 0;
    char **names;
    am_list_t *lst;
    am_handle_t lhd;
    (void)rt;

    names = music_list_names(&n);
    if (n == 0) {
        // 空目录：WASM 上顺手请求页面把 p4 音频层挂进 MEMFS（其它平台空操作）。
        // App 侧应据此提示"正在加载音乐资源"并按几百毫秒节奏重试 (Music.list)。
        music_request_tier_js();
    }

    lst = am_list_create(proc->heap_alloc, n > 0 ? (size_t)n : 1, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (lst == NULL) {
        for (int32_t i = 0; i < n; i++) free(names[i]);
        free(names);
        return music_push_null(proc);
    }
    for (int32_t i = 0; i < n; i++) {
        wchar_t *wbuf = (wchar_t *)am_malloc(proc->heap_alloc, (strlen(names[i]) + 1) * sizeof(wchar_t));
        am_handle_t hd = AM_HANDLE_NULL;
        am_list_t *grown;
        if (wbuf) {
            uint32_t wlen = _mbstowcs(wbuf, names[i], (uint32_t)strlen(names[i]));
            hd = am_process_make_wstring_handle(proc, wbuf, (size_t)wlen);
            am_free(proc->heap_alloc, wbuf);
        }
        free(names[i]);
        if (hd == AM_HANDLE_NULL) continue; // 单条分配失败：跳过该名字（不毁整张列表）
        grown = am_list_push(proc->heap_alloc, lst, am_make_value_of_handle(hd));
        if (grown == NULL) { am_list_destroy(proc->heap_alloc, lst); free(names); return music_push_null(proc); }
        lst = grown;
    }
    free(names);

    lhd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (lhd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return music_push_null(proc); }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, lhd);
        return music_push_null(proc);
    }
    return music_push(proc, am_make_value_of_handle(lhd));
}

int32_t am_native_Music_open(am_runtime_t *rt, am_process_t *proc) {
    am_wstring_t *ws = NULL;
    char path[256];
    bool ok = false;

    s_rt = rt;   // 记录运行时（引擎内部的时间基准取自 am_runtime_now_ms）
    if (!music_pop_wstring(proc, &ws)) return -1;
    if (am_native_fs_resolve(ws, path)) ok = music_start_track(path);
    return music_push_bool(proc, ok);
}

int32_t am_native_Music_close(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    music_player_cleanup();
    return music_push_bool(proc, true);
}

int32_t am_native_Music_pump(am_runtime_t *rt, am_process_t *proc) {
    s_rt = rt;
    if (s_dec == NULL) return music_push_number(proc, MUSIC_PUMP_IDLE);
    if (!s_dec->playing) return music_push_number(proc, MUSIC_PUMP_IDLE);
    if (s_dec->eof) return music_push_number(proc, MUSIC_PUMP_EOF);

    if (s_dec->silent) {
        // 静音走带：按墙钟节奏推进（一块可能跨多拍，故用 while 追帧）
        uint64_t now = music_now_ms();
        for (int32_t i = 0; i < MUSIC_PUMP_MAX_BLOCKS; i++) {
            int32_t rc;
            if (now < s_dec->clock_ms) return music_push_number(proc, MUSIC_PUMP_OK);
            rc = music_pump_block();
            if (rc == 0) { s_dec->eof = 1; return music_push_number(proc, MUSIC_PUMP_EOF); }
            if (rc < 0) return music_push_number(proc, MUSIC_PUMP_ERROR);
        }
        return music_push_number(proc, MUSIC_PUMP_OK);
    }

    for (int32_t i = 0; i < MUSIC_PUMP_MAX_BLOCKS; i++) {
        int32_t rc;
        if (!audio_out_queue_free()) return music_push_number(proc, MUSIC_PUMP_OK);
        rc = music_pump_block();
        if (rc == 0) { s_dec->eof = 1; return music_push_number(proc, MUSIC_PUMP_EOF); }
        if (rc < 0) return music_push_number(proc, MUSIC_PUMP_ERROR);
    }
    return music_push_number(proc, MUSIC_PUMP_OK);
}

int32_t am_native_Music_play(am_runtime_t *rt, am_process_t *proc) {
    s_rt = rt;
    if (s_dec != NULL) {
        if (!s_dec->playing) s_dec->clock_ms = music_now_ms(); // 恢复：重置静音走带基准
        s_dec->playing = 1;
        if (s_dec->eof) s_dec->eof = 0; // 允许从尾端重新开始（下次 pump 会报 EOF 或 0 采样）
    }
    return music_push_bool(proc, s_dec != NULL && s_dec->playing);
}

int32_t am_native_Music_pause(am_runtime_t *rt, am_process_t *proc) {
    s_rt = rt;
    if (s_dec != NULL) {
        s_dec->playing = 0;
        if (!s_dec->silent) audio_out_stop(); // 清空已排队的块（暂停 = 立即静音）
        else s_dec->clock_ms = music_now_ms();
    }
    return music_push_bool(proc, false);
}

int32_t am_native_Music_toggle(am_runtime_t *rt, am_process_t *proc) {
    s_rt = rt;
    if (s_dec == NULL) return music_push_bool(proc, false);
    if (s_dec->playing) {
        s_dec->playing = 0;
        if (!s_dec->silent) audio_out_stop();
        else s_dec->clock_ms = music_now_ms();
    }
    else {
        s_dec->playing = 1;
        s_dec->clock_ms = music_now_ms();
        s_dec->eof = 0;
    }
    return music_push_bool(proc, s_dec->playing != 0);
}

int32_t am_native_Music_is_playing(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return music_push_bool(proc, s_dec != NULL && s_dec->playing != 0);
}

int32_t am_native_Music_seek(am_runtime_t *rt, am_process_t *proc) {
    double ratio = 0.0;
    bool ok = false;

    s_rt = rt;
    if (!music_pop_number(proc, &ratio)) return -1;
    if (music_seek_ratio(ratio) == 0) {
        if (s_dec != NULL && s_dec->playing && !s_dec->silent) {
            music_pump_block();
            if (audio_out_queue_free()) music_pump_block();
        }
        ok = true;
    }
    return music_push_bool(proc, ok);
}

int32_t am_native_Music_progress(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return music_push_number(proc, music_progress());
}

int32_t am_native_Music_elapsed_ms(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return music_push_number(proc, (double)music_elapsed_ms());
}

int32_t am_native_Music_duration_ms(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return music_push_number(proc, (double)music_duration_ms());
}

int32_t am_native_Music_info(am_runtime_t *rt, am_process_t *proc) {
    double vals[4];
    (void)rt;

    if (s_dec == NULL) return music_push_null(proc);
    vals[0] = (double)s_dec->sample_rate;
    vals[1] = (double)((s_dec->type == MUSIC_TYPE_WAV) ? s_dec->wav_channels : s_dec->mp3_channels);
    vals[2] = (double)((s_dec->type == MUSIC_TYPE_WAV) ? s_dec->wav_bits : 0);
    vals[3] = (double)((s_dec->type == MUSIC_TYPE_MP3) ? s_dec->mp3_bitrate_kbps : 0);
    return music_push_number_list(proc, vals, 4);
}

int32_t am_native_Music_format(am_runtime_t *rt, am_process_t *proc) {
    const char *s;
    (void)rt;

    if (s_dec == NULL) return music_push_null(proc);
    s = (s_dec->type == MUSIC_TYPE_WAV) ? "WAV" : "MP3";
    return music_push_string(proc, s, strlen(s));
}

int32_t am_native_Music_volume(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return music_push_number(proc, (double)s_volume);
}

int32_t am_native_Music_set_volume(am_runtime_t *rt, am_process_t *proc) {
    double x = 0.0;
    (void)rt;

    if (!music_pop_number(proc, &x)) return -1;
    if (x != x) x = 0.0; // NaN
    if (x < 0.0) x = 0.0;
    if (x > (double)MUSIC_VOL_MAX) x = (double)MUSIC_VOL_MAX;
    s_volume = (uint8_t)x;
    s_volume_init = 1; // App 已接管音量，后续打开曲目不再从主音量取初值
    if (s_audio_on) audio_out_set_volume(s_volume);
    return music_push_number(proc, (double)s_volume);
}

static const am_native_func_entry_t am_native_Music_funcs[] = {
    { L"list",       am_native_Music_list },
    { L"open",       am_native_Music_open },
    { L"close",      am_native_Music_close },
    { L"pump",       am_native_Music_pump },
    { L"play",       am_native_Music_play },
    { L"pause",      am_native_Music_pause },
    { L"toggle",     am_native_Music_toggle },
    { L"is_playing", am_native_Music_is_playing },
    { L"seek",       am_native_Music_seek },
    { L"progress",   am_native_Music_progress },
    { L"elapsed_ms", am_native_Music_elapsed_ms },
    { L"duration_ms", am_native_Music_duration_ms },
    { L"info",       am_native_Music_info },
    { L"format",     am_native_Music_format },
    { L"volume",     am_native_Music_volume },
    { L"set_volume", am_native_Music_set_volume },
};

const am_native_lib_entry_t am_native_Music_lib = {
    L"Music",
    am_native_Music_funcs,
    sizeof(am_native_Music_funcs) / sizeof(am_native_Music_funcs[0])
};
