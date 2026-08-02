// ============================================================================
// 音乐盒 UI 模块 / BD4SUR 2026-08
// SD 卡根目录 WAV/MP3 文件浏览与播放：
//   - WAV：自解析 RIFF chunk（PCM 8/16bit、单/双声道），不假设 44 字节固定头；
//   - MP3：vendor/minimp3.h（实现单元 ui_musicbox_mp3.c），流式逐帧解码；
//   - 统一输出 int16 单声道 PCM，经 audio_out 流式 HAL 播放
//     （乒乓双缓冲 + 背压喂入，范式同 ui_ofdm.c TX 路径）。
// 暂停 = audio_out_stop + 解码器状态保持（文件偏移/MP3 输入流缓冲均存活），
// 恢复 = 直接续喂，无需重新 seek/解码。
// 内存纪律：进入音乐盒时申请（文件列表）、进入播放态时申请（解码器/缓冲），
// 退出时严格成对释放；本模块状态全部为文件级 static，不动 Global_State。
// ============================================================================

#include "ui_musicbox.h"

#include <string.h>
#include <strings.h>
#include <stdio.h>

#include "audio_out.h"
#include "input_device.h"
#include "ui_color.h"
#include "vendor/minimp3.h"

// minimp3 scratch 工作区注入（实现见 ui_musicbox_mp3.c；~16KB 放栈上会撑爆渲染任务栈）
extern void     minimp3_set_scratch(void *scratch);
extern uint32_t minimp3_scratch_size(void);

// 每块单声道 PCM 采样数（@44.1kHz ≈ 93ms；双槽队列 ≈ 0.19s 余量，UI 刷帧不饿音频）
#define MUSICBOX_PCM_BLOCK      (4096)
// 文件字节暂存容量（WAV 读块 / MP3 输入流缓冲共用一块，任一时刻仅一种解码器活动；
// 须 ≥ MUSICBOX_PCM_BLOCK × 4 字节（WAV 16bit 立体声一块的原始字节数））
#define MUSICBOX_SCRATCH_CAP    (16384)

#define MUSICBOX_VOLUME_DEFAULT (64)   // 静态初值（进入音乐盒时被全局主音量覆盖，见 ui_musicbox_menu_init）
#define MUSICBOX_VOLUME_MAX     (128)  // 音乐内容动态范围大，上限放宽到 128
#define MUSICBOX_VOLUME_STEP    (16)
#define MUSICBOX_UI_REFRESH_MS  (500)  // 进度条限频刷新间隔

// 解码前最小输入缓冲量（字节）：MP3 单帧最大 2304（含 free format，标准码率上限 1440）
// + 下一帧头 4 字节 + 余量。minimp3 的 mp3dec_decode_frame 要求缓冲中可见
// 「完整一帧 + 下一帧头 4 字节」做交叉验证，否则会把整个解码器状态 memset 清零
// （位蓄水池/IMDCT/合成滤波历史丢失），缓冲边界后一帧重建有误——表现为周期性轻微破音
// （周期 ≈ 16KB/码率，320kbps≈0.41s、256k≈0.5s、192k≈0.68s、128k≈1s）。
// 因此解码前必须补足到此水位，永远不把半截帧交给 decode_frame（EOF 尾部除外）。
#define MUSICBOX_MP3_MIN_BUF    (2560)

// ---------------- 文件列表（仿 ui_ebook.c 两遍枚举模式） ----------------
static char    **s_list_mb = NULL;  // 完整路径（多字节，含前导'/'）
static wchar_t **s_list_w  = NULL;  // 显示名（宽字符）
static const wchar_t **s_items = NULL; // 菜单项借用指针表
static int32_t   s_list_count = 0;

// ---------------- 播放器静态状态 ----------------
#define MUSICBOX_DEC_WAV (1)
#define MUSICBOX_DEC_MP3 (2)

typedef struct {
    int32_t  type;
    uint32_t sample_rate;
    uint32_t file_size;
    // WAV
    uint32_t wav_data_total;   // data chunk 总字节数
    uint32_t wav_data_left;    // data chunk 剩余字节
    uint16_t wav_channels;
    uint16_t wav_bits;
    // MP3
    mp3dec_t mp3d;
    uint32_t mp3_in_start;     // scratch 内有效数据起点
    uint32_t mp3_in_len;       // 有效数据长度
    int32_t  mp3_file_eof;     // 文件已读到尾
    uint32_t mp3_consumed;     // 已消费文件字节数（进度用）
    int16_t  mp3_frame[MINIMP3_MAX_SAMPLES_PER_FRAME]; // 当前帧（交错，含声道）
    uint32_t mp3_frame_len;    // 当前帧每声道采样数
    uint32_t mp3_frame_pos;    // 当前帧已消费（每声道）
    int32_t  mp3_channels;
} Musicbox_Decoder;

static Musicbox_Decoder *s_dec = NULL;
static int16_t *s_pcm[AUDIO_OUT_QUEUE_DEPTH] = {NULL, NULL}; // 乒乓双缓冲（playRaw 引用，须常驻至播完）
static int32_t  s_fill_buf = 0;      // 下一个待填充/投放的缓冲序号（0/1 轮换）
static uint8_t *s_scratch = NULL;    // 文件字节暂存（WAV/MP3 共用，PSRAM）
static void    *s_mp3_scratch = NULL;// minimp3 工作区（PSRAM，按需注入）
static int32_t  s_track_idx = 0;
static int32_t  s_playing = 0;       // 1=播放中 0=已暂停
static int32_t  s_audio_on = 0;      // audio_out_init 是否已调用（退出时对账 close）
static int32_t  s_open_failed = 0;   // 打开/解码失败（显示提示，任一键回菜单）
static uint8_t  s_volume = MUSICBOX_VOLUME_DEFAULT;
static uint64_t s_last_ui_ts = 0;    // 上一次进度条刷新时间戳

// ============================================================================
// 小端读取工具
// ============================================================================
static uint16_t rd16le(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32le(const uint8_t *p) { return (uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24)); }

static int32_t musicbox_ext_is_wav(const char *path) {
    size_t len = strlen(path);
    return (len >= 4 && strcasecmp(path + len - 4, ".wav") == 0);
}
static int32_t musicbox_ext_is_mp3(const char *path) {
    size_t len = strlen(path);
    return (len >= 4 && strcasecmp(path + len - 4, ".mp3") == 0);
}

// ============================================================================
// 文件列表（菜单态）
// ============================================================================
static void musicbox_free_list(void) {
    if (s_list_mb != NULL) {
        for (int32_t i = 0; i < s_list_count; i++) {
            if (s_list_mb[i] != NULL) free(s_list_mb[i]);
        }
        free(s_list_mb);
        s_list_mb = NULL;
    }
    if (s_list_w != NULL) {
        // s_list_count 为 0 时可能存在占位提示项（下标 0），统一按"已分配项"释放：
        // 占位项也计入 s_list_count（见 ui_musicbox_menu_init），此处无需特判
        for (int32_t i = 0; i < s_list_count; i++) {
            if (s_list_w[i] != NULL) free(s_list_w[i]);
        }
        free(s_list_w);
        s_list_w = NULL;
    }
    if (s_items != NULL) {
        free((void *)s_items);
        s_items = NULL;
    }
    s_list_count = 0;
}

void ui_musicbox_menu_init(Key_Event *key_event, Global_State *global_state) {
    musicbox_free_list();

    // 进入音乐盒：以全局主音量为初始音量（音乐盒内部 ←/→ 调节不回写全局设置）
    s_volume = (uint8_t)global_state->volume;

    int32_t total = list_files("/", NULL);
    int32_t cap = (total > 0) ? total : 1; // 至少留 1 项放占位提示
    char **names = (char **)platform_calloc((size_t)cap, sizeof(char *));
    s_list_mb = (char **)platform_calloc((size_t)cap, sizeof(char *));
    s_list_w  = (wchar_t **)platform_calloc((size_t)cap, sizeof(wchar_t *));
    s_items   = (const wchar_t **)platform_calloc((size_t)cap, sizeof(wchar_t *));
    if (names != NULL && s_list_mb != NULL && s_list_w != NULL && s_items != NULL
        && total > 0 && list_files("/", names) >= 0) {
        for (int32_t i = 0; i < total; i++) {
            if (names[i] == NULL) continue;
            // 规范为带前导'/'的完整路径
            char path[160];
            if (names[i][0] != '/') snprintf(path, sizeof(path), "/%s", names[i]);
            else                    strncpy(path, names[i], sizeof(path) - 1);
            path[sizeof(path) - 1] = '\0';
            free(names[i]);
            // 仅保留 WAV/MP3 文件（非目录）
            if (platform_is_directory(path)) continue;
            if (!musicbox_ext_is_wav(path) && !musicbox_ext_is_mp3(path)) continue;
            size_t plen = strlen(path);
            s_list_mb[s_list_count] = (char *)platform_malloc(plen + 1);
            s_list_w[s_list_count]  = (wchar_t *)platform_calloc(plen + 1, sizeof(wchar_t));
            if (s_list_mb[s_list_count] == NULL || s_list_w[s_list_count] == NULL) {
                if (s_list_mb[s_list_count] != NULL) free(s_list_mb[s_list_count]);
                if (s_list_w[s_list_count]  != NULL) free(s_list_w[s_list_count]);
                continue;
            }
            memcpy(s_list_mb[s_list_count], path, plen + 1);
            // 显示名：去掉前导'/'，UTF-8 转宽字符
            _mbstowcs(s_list_w[s_list_count], path + 1, (uint32_t)plen);
            s_list_count++;
        }
    }
    if (names != NULL) free(names);

    // 按路径字符串升序（插入排序，UTF-8 字节序即码点序）
    for (int32_t i = 1; i < s_list_count; i++) {
        char *key_mb = s_list_mb[i];
        wchar_t *key_w = s_list_w[i];
        int32_t j = i - 1;
        while (j >= 0 && strcmp(s_list_mb[j], key_mb) > 0) {
            s_list_mb[j + 1] = s_list_mb[j];
            s_list_w[j + 1] = s_list_w[j];
            j--;
        }
        s_list_mb[j + 1] = key_mb;
        s_list_w[j + 1] = key_w;
    }

    if (s_list_count == 0) {
        // 占位提示项（选中无操作）：s_list_mb[0] 保持 NULL，free_list 对 NULL 安全
        if (s_list_w != NULL) {
            s_list_w[0] = (wchar_t *)platform_calloc(32, sizeof(wchar_t));
            if (s_list_w[0] != NULL) {
                wcscpy(s_list_w[0], L"（未找到 WAV/MP3 文件）");
                s_list_count = 1; // 占位项计入计数，保证菜单可显示、free_list 可释放
            }
        }
    }
    for (int32_t i = 0; i < s_list_count; i++) {
        s_items[i] = s_list_w[i];
    }

    global_state->w_menu_main->title = L"音乐盒";
    global_state->w_menu_main->items = s_items;
    global_state->w_menu_main->item_num = s_list_count;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
}

int32_t ui_musicbox_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    (void)ke; (void)gs;
    // 占位提示项（s_list_mb[0] 为 NULL）或空列表：选中无操作
    if (s_list_count <= 0 || ms->current_item_index >= s_list_count
        || s_list_mb[ms->current_item_index] == NULL) {
        return STATE_MUSICBOX_MENU;
    }
    s_track_idx = ms->current_item_index;
    return STATE_MUSICBOX_PLAYING;
}

void ui_musicbox_menu_on_exit(void) {
    musicbox_free_list();
}

// ============================================================================
// WAV 解码器（RIFF chunk 遍历解析，不假设 44 字节固定头；PCM 8/16bit，单/双声道）
// ============================================================================
static int32_t wav_open(Musicbox_Decoder *d) {
    uint8_t hdr[12];
    if (platform_file_read(hdr, 12) != 12) return -1;
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) return -1;
    uint32_t pos = 12;
    int32_t got_fmt = 0;
    while (pos + 8 <= d->file_size) {
        uint8_t ch[8];
        if (platform_file_seek(pos) != 0) return -1;
        if (platform_file_read(ch, 8) != 8) return -1;
        pos += 8;
        uint32_t csize = rd32le(ch + 4);
        if (memcmp(ch, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (csize < 16 || platform_file_read(fmt, 16) != 16) return -1;
            if (rd16le(fmt) != 1) return -1; // 仅支持 PCM（非 PCM 如 ADPCM/float 不支持）
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
            d->wav_data_total = csize;
            d->wav_data_left  = csize;
            platform_file_seek(pos); // 定位于数据起点
            return 0;
        }
        pos += csize + (csize & 1); // RIFF chunk 按偶数字节对齐
    }
    return -1;
}

// 解码出 max_samples 个单声道 int16，返回采样数，0=EOF
static uint32_t wav_read(Musicbox_Decoder *d, int16_t *pcm, uint32_t max_samples) {
    uint32_t bpf = (uint32_t)d->wav_channels * (d->wav_bits / 8); // 每采样帧字节数
    uint32_t want = max_samples * bpf;
    if (want > MUSICBOX_SCRATCH_CAP) want = MUSICBOX_SCRATCH_CAP;
    if (want > d->wav_data_left) want = d->wav_data_left;
    want -= want % bpf;
    if (want == 0) return 0;
    int32_t n = platform_file_read(s_scratch, want);
    if (n <= 0) return 0;
    d->wav_data_left -= (uint32_t)n;
    uint32_t frames = (uint32_t)n / bpf;
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

// ============================================================================
// MP3 解码器（minimp3 流式：16KB 输入流缓冲 + 逐帧解码）
// ============================================================================

// 把剩余数据移到 scratch 头部并从文件补满；返回补入字节数（0=无法补入）
static int32_t mp3_fill_input(Musicbox_Decoder *d) {
    if (d->mp3_in_start > 0) {
        memmove(s_scratch, s_scratch + d->mp3_in_start, d->mp3_in_len);
        d->mp3_in_start = 0;
    }
    if (d->mp3_file_eof) return 0;
    int32_t n = platform_file_read(s_scratch + d->mp3_in_len, MUSICBOX_SCRATCH_CAP - d->mp3_in_len);
    if (n <= 0) { d->mp3_file_eof = 1; return 0; }
    d->mp3_in_len += (uint32_t)n;
    return n;
}

// 打开并预解首帧以确定采样率/声道数（首帧留在 mp3_frame 中供后续 read 消费）
static int32_t mp3_open(Musicbox_Decoder *d) {
    mp3dec_init(&d->mp3d);
    d->mp3_in_start = 0;
    d->mp3_in_len = 0;
    d->mp3_file_eof = 0;
    d->mp3_consumed = 0;
    d->mp3_frame_len = 0;
    d->mp3_frame_pos = 0;
    for (int32_t tries = 0; tries < 64; tries++) { // 限次扫描，防坏文件死循环
        if (d->mp3_in_len < MUSICBOX_SCRATCH_CAP && !d->mp3_file_eof) mp3_fill_input(d);
        if (d->mp3_in_len == 0) return -1;
        mp3dec_frame_info_t info;
        int samples = mp3dec_decode_frame(&d->mp3d, s_scratch + d->mp3_in_start,
                                          (int)d->mp3_in_len, d->mp3_frame, &info);
        d->mp3_in_start += info.frame_bytes;
        d->mp3_in_len   -= info.frame_bytes;
        d->mp3_consumed += info.frame_bytes;
        if (samples > 0) {
            d->mp3_frame_len = (uint32_t)samples;
            d->mp3_channels  = info.channels;
            d->sample_rate   = (uint32_t)info.hz;
            return 0;
        }
        if (info.frame_bytes == 0 && d->mp3_file_eof) return -1;
    }
    return -1;
}

// 解码出 max_samples 个单声道 int16，返回采样数，0=EOF
static uint32_t mp3_read(Musicbox_Decoder *d, int16_t *pcm, uint32_t max_samples) {
    uint32_t out = 0;
    while (out < max_samples) {
        if (d->mp3_frame_pos < d->mp3_frame_len) {
            // 从当前帧取数（立体声混音为单声道）
            uint32_t avail = d->mp3_frame_len - d->mp3_frame_pos;
            uint32_t take = max_samples - out;
            if (take > avail) take = avail;
            const int16_t *f = d->mp3_frame;
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
        // 解码下一帧。先补足输入缓冲到 MUSICBOX_MP3_MIN_BUF（见宏注释）：
        // 避免 decode_frame 因「完整帧+下一帧头」不可见而 memset 清零解码器状态，
        // 消除输入缓冲边界处的周期性破音（文件尾部不足时按现状解码，解码结束在即，无害）
        while (!d->mp3_file_eof && d->mp3_in_len < MUSICBOX_MP3_MIN_BUF) {
            if (mp3_fill_input(d) <= 0) break;
        }
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
            continue;
        }
        if (info.frame_bytes == 0) {
            // 数据不足/未找到帧头：补流；文件已尽则结束
            if (d->mp3_file_eof) break;
            int32_t got = mp3_fill_input(d);
            if (got <= 0 && d->mp3_in_len == 0) break;
            // 缓冲已满仍找不到帧头（坏数据）：丢弃 1 字节防死循环
            if (d->mp3_in_len == MUSICBOX_SCRATCH_CAP) {
                d->mp3_in_start++;
                d->mp3_in_len--;
                d->mp3_consumed++;
            }
        }
        // frame_bytes > 0 但 samples == 0：跳过非音频数据（ID3 标签等），继续
    }
    return out;
}

// ============================================================================
// 播放器（播放态）
// ============================================================================

static float musicbox_progress(void) {
    if (!s_dec) return 0.0f;
    if (s_dec->type == MUSICBOX_DEC_WAV) {
        return (s_dec->wav_data_total > 0)
             ? 1.0f - (float)s_dec->wav_data_left / (float)s_dec->wav_data_total : 0.0f;
    }
    return (s_dec->file_size > 0)
         ? (float)s_dec->mp3_consumed / (float)s_dec->file_size : 0.0f;
}

// 解码一块并投入扬声器队列。返回 1=已投放，0=EOF（无更多数据），-1=投放失败。
// 调用前提：audio_out_queue_free() 为真（队列有空槽，对应缓冲已播完可覆写）。
static int32_t musicbox_fill_and_play(void) {
    int16_t *pcm = s_pcm[s_fill_buf];
    uint32_t n = (s_dec->type == MUSICBOX_DEC_WAV)
               ? wav_read(s_dec, pcm, MUSICBOX_PCM_BLOCK)
               : mp3_read(s_dec, pcm, MUSICBOX_PCM_BLOCK);
    if (n == 0) return 0;
    if (audio_out_enqueue(pcm, n) == 0) {
        s_fill_buf ^= 1; // 乒乓轮换
        return 1;
    }
    return -1;
}

static void musicbox_stop_decoder(void) {
    if (s_dec != NULL) {
        platform_file_close();
        free(s_dec);
        s_dec = NULL;
    }
}

// 打开第 idx 首曲目并启动播放（含预填队列）。返回 0 成功。
static int32_t musicbox_start_track(int32_t idx) {
    musicbox_stop_decoder();
    if (idx < 0 || idx >= s_list_count || s_list_mb[idx] == NULL) return -1;
    s_dec = (Musicbox_Decoder *)platform_calloc(1, sizeof(Musicbox_Decoder));
    if (s_dec == NULL) return -1;
    if (platform_file_open(s_list_mb[idx]) != 0) { musicbox_stop_decoder(); return -1; }
    s_dec->file_size = platform_file_size();
    int32_t rc;
    if (musicbox_ext_is_wav(s_list_mb[idx])) {
        s_dec->type = MUSICBOX_DEC_WAV;
        rc = wav_open(s_dec);
    }
    else {
        s_dec->type = MUSICBOX_DEC_MP3;
        rc = mp3_open(s_dec);
    }
    if (rc != 0 || s_dec->sample_rate == 0) { musicbox_stop_decoder(); return -1; }

    // 采样率随曲目更新：close→init 序列使 audio_out_close 保存的"原音量"
    // 始终是进入音乐盒前的主音量（init 时会重新保存当前音量，close 已先恢复之）
    if (s_audio_on) { audio_out_close(); s_audio_on = 0; }
    audio_out_init(s_dec->sample_rate, s_volume);
    s_audio_on = 1;

    s_track_idx = idx;
    s_fill_buf = 0;
    s_playing = 1;
    s_last_ui_ts = 0;

    // 预填队列全部空槽（一块在播、一块待播），立即开始播放
    musicbox_fill_and_play();
    if (audio_out_queue_free()) musicbox_fill_and_play();
    printf("musicbox: play [%d/%d] %s (%lu Hz)\n", (int)(idx + 1), (int)s_list_count,
           s_list_mb[idx], (unsigned long)s_dec->sample_rate);
    return 0;
}

// 停止播放并释放播放态全部资源（解码器/缓冲/audio_out；文件列表保留供菜单使用）
static void musicbox_player_cleanup(void) {
    if (s_audio_on) {
        audio_out_stop();
        audio_out_close();
        s_audio_on = 0;
    }
    musicbox_stop_decoder();
    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++) {
        if (s_pcm[i] != NULL) { free(s_pcm[i]); s_pcm[i] = NULL; }
    }
    if (s_scratch != NULL) { free(s_scratch); s_scratch = NULL; }
    if (s_mp3_scratch != NULL) {
        minimp3_set_scratch(NULL);
        free(s_mp3_scratch);
        s_mp3_scratch = NULL;
    }
    s_playing = 0;
}

// ---------------- 播放界面绘制 ----------------

// 播放进度/音量条区域（进度条重绘时只擦除该区域，避免整屏重绘打断音频喂入节奏）
#define MUSICBOX_BAR_X      (20)
#define MUSICBOX_BAR_W      (280)
#define MUSICBOX_PROG_Y     (150)
#define MUSICBOX_VOL_Y      (185)
#define MUSICBOX_BAR_H      (10)

static void musicbox_theme_colors(Global_State *gs, uint8_t *fg, uint8_t *dim) {
    if (gs->ui_color_style == UI_COLOR_LIGHT) { fg[0] = 33; fg[1] = 33; fg[2] = 33; dim[0] = 200; dim[1] = 200; dim[2] = 200; }
    else                                      { fg[0] = 220; fg[1] = 220; fg[2] = 220; dim[0] = 90; dim[1] = 90; dim[2] = 90; }
}

static void musicbox_draw_bar(Global_State *gs, int32_t y, float ratio, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t fg[3], dim[3];
    musicbox_theme_colors(gs, fg, dim);
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;
    // 轨道（含 1px 留白边框效果：先画底色全宽，再画前景比例宽）
    gfx_draw_rectangle(gs->gfx, MUSICBOX_BAR_X, y, MUSICBOX_BAR_W, MUSICBOX_BAR_H, dim[0], dim[1], dim[2], 1);
    int32_t w = (int32_t)(MUSICBOX_BAR_W * ratio);
    if (w > 0) {
        gfx_draw_rectangle(gs->gfx, MUSICBOX_BAR_X, y, (uint32_t)w, MUSICBOX_BAR_H, r, g, b, 1);
    }
}

// 重绘进度/音量区域（擦除背景后重画两条bar与百分比文本；调用方负责 gfx_refresh）
static void musicbox_draw_bars(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    uint8_t fg[3], dim[3];
    musicbox_theme_colors(global_state, fg, dim);
    int32_t dark = (global_state->ui_color_style == UI_COLOR_DARK);
    // 擦除条形区背景
    gfx_draw_rectangle(global_state->gfx, 0, MUSICBOX_PROG_Y - 18, global_state->gfx->width,
                       MUSICBOX_VOL_Y + MUSICBOX_BAR_H - (MUSICBOX_PROG_Y - 18), dark ? 0 : 255, dark ? 0 : 255, dark ? 0 : 255, 1);
    // 进度
    float prog = musicbox_progress();
    wchar_t line[48];
    swprintf(line, sizeof(line) / sizeof(wchar_t), L"进度 %d%%", (int)(prog * 100.0f));
    gfx_font_draw_text(global_state->gfx, GFX_FONT_ALPHA_12, line, MUSICBOX_BAR_X, MUSICBOX_PROG_Y - 16, fg[0], fg[1], fg[2], 1);
    musicbox_draw_bar(global_state, MUSICBOX_PROG_Y, prog, 0, 170, 0);
    // 音量
    swprintf(line, sizeof(line) / sizeof(wchar_t), L"音量 %d", (int)s_volume);
    gfx_font_draw_text(global_state->gfx, GFX_FONT_ALPHA_12, line, MUSICBOX_BAR_X, MUSICBOX_VOL_Y - 16, fg[0], fg[1], fg[2], 1);
    musicbox_draw_bar(global_state, MUSICBOX_VOL_Y, (float)s_volume / MUSICBOX_VOLUME_MAX, 30, 100, 220);
}

// 全量绘制播放界面（进入/切歌/暂停状态切换时）
static void musicbox_draw_player(Key_Event *key_event, Global_State *global_state) {
    if (global_state->ui_color_style == UI_COLOR_LIGHT) gfx_fill_white(global_state->gfx);
    else                                                gfx_soft_clear(global_state->gfx);

    ui_draw_header(key_event, global_state, (wchar_t *)L"音乐盒", 1);
    ui_draw_footer(key_event, global_state, (wchar_t *)L"D:暂停 ←→:音量 4/6:切曲 A:返回", 1);

    uint8_t fg[3], dim[3];
    musicbox_theme_colors(global_state, fg, dim);

    // 曲目名（序号 x/N + 文件名），过长截断
    wchar_t title[64];
    if (s_open_failed || s_dec == NULL) {
        wcscpy(title, L"无法播放该文件");
    }
    else {
        wchar_t name[40];
        const wchar_t *full = (s_track_idx < s_list_count) ? s_list_w[s_track_idx] : L"";
        uint32_t nl = wcslen(full);
        if (nl > 28) { // 保留末尾（扩展名附近更有区分度）
            name[0] = L'…';
            wcsncpy(name + 1, full + (nl - 27), 27);
            name[28] = L'\0';
        }
        else {
            wcscpy(name, full);
        }
        swprintf(title, sizeof(title) / sizeof(wchar_t), L"%d/%d %ls",
                 (int)(s_track_idx + 1), (int)s_list_count, name);
    }
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_16, title,
                                global_state->gfx->width / 2, 55, fg[0], fg[1], fg[2], 1);

    // 状态行：播放/暂停 + 格式与采样率
    wchar_t status[48];
    if (s_dec != NULL) {
        swprintf(status, sizeof(status) / sizeof(wchar_t), L"%ls | %ls %luHz",
                 s_playing ? L"播放中" : L"已暂停",
                 (s_dec->type == MUSICBOX_DEC_WAV) ? L"WAV" : L"MP3",
                 (unsigned long)s_dec->sample_rate);
    }
    else {
        wcscpy(status, L"按任意键返回");
    }
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, status,
                                global_state->gfx->width / 2, 90, dim[0], dim[1], dim[2], 1);

    musicbox_draw_bars(key_event, global_state);
    gfx_refresh(global_state->gfx);
}

// 播放态首次获焦：分配资源并启动当前曲目
void ui_musicbox_playing_on_enter(Key_Event *key_event, Global_State *global_state) {
    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++)
        s_pcm[i] = (int16_t *)platform_calloc(MUSICBOX_PCM_BLOCK, sizeof(int16_t));
    s_scratch = (uint8_t *)platform_calloc(MUSICBOX_SCRATCH_CAP, 1);
    s_mp3_scratch = platform_calloc(1, minimp3_scratch_size());
    if (s_pcm[0] == NULL || s_pcm[1] == NULL || s_scratch == NULL || s_mp3_scratch == NULL) {
        printf("musicbox: alloc FAILED (oom)\n");
        musicbox_player_cleanup();
        s_open_failed = 1;
        musicbox_draw_player(key_event, global_state);
        return;
    }
    minimp3_set_scratch(s_mp3_scratch);

    s_open_failed = 0;
    if (musicbox_start_track(s_track_idx) != 0) {
        s_open_failed = 1;
    }
    musicbox_draw_player(key_event, global_state);
}

int32_t ui_musicbox_playing_event(Key_Event *key_event, Global_State *global_state) {
    // 打开失败/内存不足：显示提示，任一键返回文件列表
    if (s_open_failed || s_dec == NULL) {
        if (key_event->key_edge < 0) {
            musicbox_player_cleanup();
            return STATE_MUSICBOX_MENU;
        }
        return STATE_MUSICBOX_PLAYING;
    }

    // A：停止并返回文件列表
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_esc) {
        musicbox_player_cleanup();
        return STATE_MUSICBOX_MENU;
    }
    // D：播放/暂停（暂停=stop 清队列+解码器状态保持；恢复=续喂，断点无缝继续）
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_enter) {
        s_playing = !s_playing;
        if (!s_playing) audio_out_stop();
        musicbox_draw_player(key_event, global_state);
        return STATE_MUSICBOX_PLAYING;
    }
    // ←/→：音量减/加
    if (key_event->key_edge < 0 &&
        (key_event->key_code == NANO_KEY_left || key_event->key_code == NANO_KEY_right)) {
        int32_t v = (int32_t)s_volume + ((key_event->key_code == NANO_KEY_right) ? MUSICBOX_VOLUME_STEP : -MUSICBOX_VOLUME_STEP);
        if (v < 0) v = 0;
        if (v > MUSICBOX_VOLUME_MAX) v = MUSICBOX_VOLUME_MAX;
        s_volume = (uint8_t)v;
        audio_out_set_volume(s_volume);
        musicbox_draw_bars(key_event, global_state);
        gfx_refresh(global_state->gfx);
        return STATE_MUSICBOX_PLAYING;
    }
    // 4/6：上一曲/下一曲（列表循环；失败则尝试后续曲目，全部失败则提示）
    if (key_event->key_edge < 0 &&
        (key_event->key_code == NANO_KEY_4 || key_event->key_code == NANO_KEY_6)) {
        int32_t step = (key_event->key_code == NANO_KEY_6) ? 1 : -1;
        audio_out_stop();
        int32_t idx = s_track_idx;
        int32_t ok = 0;
        for (int32_t tries = 0; tries < s_list_count; tries++) {
            idx = (idx + step + s_list_count) % s_list_count;
            if (musicbox_start_track(idx) == 0) { ok = 1; break; }
        }
        s_open_failed = !ok;
        musicbox_draw_player(key_event, global_state);
        return STATE_MUSICBOX_PLAYING;
    }

    if (!s_playing) return STATE_MUSICBOX_PLAYING;

    // 背压安全的喂入：队列有空槽才填充
    if (audio_out_queue_free()) {
        int32_t rc = musicbox_fill_and_play();
        if (rc == 0) {
            // EOF：自动播放下一曲（循环列表）
            audio_out_stop();
            int32_t idx = s_track_idx;
            int32_t ok = 0;
            for (int32_t tries = 0; tries < s_list_count; tries++) {
                idx = (idx + 1) % s_list_count;
                if (musicbox_start_track(idx) == 0) { ok = 1; break; }
            }
            s_open_failed = !ok;
            musicbox_draw_player(key_event, global_state);
            return STATE_MUSICBOX_PLAYING;
        }
    }

    // 限频刷新进度条
    if (global_state->timestamp - s_last_ui_ts >= MUSICBOX_UI_REFRESH_MS) {
        s_last_ui_ts = global_state->timestamp;
        musicbox_draw_bars(key_event, global_state);
        gfx_refresh(global_state->gfx);
    }
    return STATE_MUSICBOX_PLAYING;
}
