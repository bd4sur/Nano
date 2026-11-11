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

#include "hal_audio_out.h"
#include "hal_key.h"
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
#define MUSICBOX_UI_REFRESH_MS  (250)  // 进度条限频刷新间隔（触屏进度条需比 0.5s 更跟手）
#define MUSICBOX_VOL_REPEAT_MS  (120)  // 触屏音量键按住后的连发间隔
#define MUSICBOX_VOL_HOLD_MS    (350)  // 触屏音量键按住的首次延迟（之后转入连发）

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
    uint32_t wav_data_start;   // data chunk 在文件中的绝对偏移（定位/拖动进度用）
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
    int32_t  mp3_bitrate_kbps; // 首帧码率（信息显示与总时长估算用）
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

// ---------------- 触屏控件状态（STATE_MUSICBOX_PLAYING） ----------------
// 控件命中用按下点、动作结算用松手沿（项目触屏范式，见 ui_app.c get_input_event 注释）。
// 拖动进度条期间只更新预览位置，松手沿才真正 seek（避免拖动中反复重开文件）。
static int32_t s_ui_pressed = -1;    // 当前按住的控件（高亮用；-1=无）
static int32_t s_seek_active = 0;    // 本次触摸序列起始于进度条
static float   s_seek_preview = -1.0f; // 拖动中的预览比例（<0=不在拖动）
static int32_t s_vol_hold = 0;       // 音量键按住方向：-1=减 +1=加 0=无
static uint64_t s_vol_next_ts = 0;   // 下一次音量连发的时刻

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

    // 进入音乐盒：以全局主音量为初始音量（音乐盒内部 ←/→ 调节不回写全局设置）。
    // 主音量域 0~255 比音乐盒内部 0~MUSICBOX_VOLUME_MAX 宽，进入时钳位，
    // 避免音量条比例越界、以及 255 的过大增益把音乐内容推到削顶。
    s_volume = (global_state->volume > MUSICBOX_VOLUME_MAX)
             ? (uint8_t)MUSICBOX_VOLUME_MAX : (uint8_t)global_state->volume;

    int32_t total = list_files(PLATFORM_ROOT_DIR "/music", NULL);
    int32_t cap = (total > 0) ? total : 1; // 至少留 1 项放占位提示
    char **names = (char **)platform_calloc((size_t)cap, sizeof(char *));
    s_list_mb = (char **)platform_calloc((size_t)cap, sizeof(char *));
    s_list_w  = (wchar_t **)platform_calloc((size_t)cap, sizeof(wchar_t *));
    s_items   = (const wchar_t **)platform_calloc((size_t)cap, sizeof(wchar_t *));
    if (names != NULL && s_list_mb != NULL && s_list_w != NULL && s_items != NULL
        && total > 0 && list_files(PLATFORM_ROOT_DIR "/music", names) >= 0) {
        for (int32_t i = 0; i < total; i++) {
            if (names[i] == NULL) continue;
            // 规范为带前缀的完整路径
            char path[160];
            snprintf(path, sizeof(path), PLATFORM_ROOT_DIR "/music/%s", names[i]);
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
            // 显示名：只取最后一段文件名，UTF-8 转宽字符
            const char *disp = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
            _mbstowcs(s_list_w[s_list_count], disp, (uint32_t)plen);
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
            d->wav_data_start = pos;
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
            d->mp3_bitrate_kbps = info.bitrate_kbps;
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
            // 码率只用于信息显示与总时长估算：保留首帧码率（VBR 逐帧码率变化，
            // 跟随更新会让总时长/进度显示跳动）；仅首帧码率为 0 时补一次
            if (d->mp3_bitrate_kbps == 0) d->mp3_bitrate_kbps = info.bitrate_kbps;
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

// 前置声明（musicbox_seek_ratio 定位后立即重新预填播放队列）
static int32_t musicbox_fill_and_play(void);

// 每秒消耗的字节数（WAV = 采样率 × 声道 × 位深/8；MP3 用首帧码率估算）
static uint32_t musicbox_bytes_per_sec(void) {
    if (!s_dec) return 0;
    if (s_dec->type == MUSICBOX_DEC_WAV) {
        return s_dec->sample_rate * (uint32_t)s_dec->wav_channels * (uint32_t)(s_dec->wav_bits / 8);
    }
    return (uint32_t)(s_dec->mp3_bitrate_kbps > 0 ? s_dec->mp3_bitrate_kbps * 125u : 0u); // kbps→B/s
}

// 已播放时长（ms）
static uint32_t musicbox_elapsed_ms(void) {
    if (!s_dec) return 0;
    uint32_t bps = musicbox_bytes_per_sec();
    if (bps == 0) return 0;
    if (s_dec->type == MUSICBOX_DEC_WAV) {
        uint32_t used = (s_dec->wav_data_total >= s_dec->wav_data_left)
                      ? s_dec->wav_data_total - s_dec->wav_data_left : 0;
        return (uint32_t)((uint64_t)used * 1000u / bps);
    }
    return (uint32_t)((uint64_t)s_dec->mp3_consumed * 1000u / bps);
}

// 总时长（ms）：WAV 精确；MP3 按首帧码率对文件大小估算（CBR 准确，VBR 仅近似）
static uint32_t musicbox_duration_ms(void) {
    if (!s_dec) return 0;
    uint32_t bps = musicbox_bytes_per_sec();
    if (bps == 0) return 0;
    uint32_t bytes = (s_dec->type == MUSICBOX_DEC_WAV) ? s_dec->wav_data_total : s_dec->file_size;
    return (uint32_t)((uint64_t)bytes * 1000u / bps);
}

// 播放进度（0.0~1.0），按字节/时长一致地定义（拖动进度条与自动刷新共用）
static float musicbox_progress(void) {
    if (!s_dec) return 0.0f;
    if (s_dec->type == MUSICBOX_DEC_WAV) {
        return (s_dec->wav_data_total > 0)
             ? 1.0f - (float)s_dec->wav_data_left / (float)s_dec->wav_data_total : 0.0f;
    }
    return (s_dec->file_size > 0)
         ? (float)s_dec->mp3_consumed / (float)s_dec->file_size : 0.0f;
}

// 定位到比例 ratio（0.0~1.0）：WAV 直接 seek data 区，MP3 定位到近似字节偏移后
// 重置解码器、由 mp3_read 重新同步帧头。松手沿调用，调用前队列已由 audio_out_stop 清空。
static int32_t musicbox_seek_ratio(float ratio) {
    if (!s_dec) return -1;
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 0.995f) ratio = 0.995f; // 留出尾部余量，避免 seek 到 EOF 立刻跳曲

    audio_out_stop();

    if (s_dec->type == MUSICBOX_DEC_WAV) {
        uint32_t bpf = (uint32_t)s_dec->wav_channels * (uint32_t)(s_dec->wav_bits / 8);
        if (bpf == 0) return -1;
        uint32_t delta = (uint32_t)((float)s_dec->wav_data_total * ratio);
        delta -= delta % bpf; // 对齐到采样帧边界
        if (delta > s_dec->wav_data_total) delta = s_dec->wav_data_total;
        if (platform_file_seek(s_dec->wav_data_start + delta) != 0) return -1;
        s_dec->wav_data_left = s_dec->wav_data_total - delta;
    }
    else {
        uint32_t off = (uint32_t)((float)s_dec->file_size * ratio);
        if (off > s_dec->file_size) off = s_dec->file_size;
        if (platform_file_seek(off) != 0) return -1;
        mp3dec_init(&s_dec->mp3d);
        s_dec->mp3_in_start = 0;
        s_dec->mp3_in_len = 0;
        s_dec->mp3_file_eof = 0;
        s_dec->mp3_consumed = off;
        s_dec->mp3_frame_len = 0;
        s_dec->mp3_frame_pos = 0;
    }

    s_fill_buf = 0;
    if (s_playing) {
        musicbox_fill_and_play();
        if (audio_out_queue_free()) musicbox_fill_and_play();
    }
    return 0;
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
    // 切曲/自动续播时丢弃上一曲遗留的触屏状态（进度拖动、按住高亮、音量连发）
    s_ui_pressed = -1;
    s_seek_active = 0;
    s_seek_preview = -1.0f;
    s_vol_hold = 0;
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
    s_ui_pressed = -1;
    s_seek_active = 0;
    s_seek_preview = -1.0f;
    s_vol_hold = 0;
}

// ---------------- 播放界面：显式触屏控件 ----------------
//
// 本界面不再依赖十六宫格触屏软按键（STATE_MUSICBOX_PLAYING 已加入
// ui_app_state_is_menu 抑制表，输入层不会为本状态的触摸生成任何软按键）。
// 触屏由本模块直接消费：命中判定用按下点（touch_down_x/y），动作结算用松手沿
// （TOUCH_EDGE_UP），松手点须仍落在同一控件内，否则视为拖出取消——与
// ui_linglong_ctrl.c / ui_water.c 的显式触屏控件同范式。
//
// 布局只由画布宽高推出，绘制（musicbox_draw_*）与命中（musicbox_ui_hit）
// 共用 Musicbox_Ui，保证“看得见的一定点得到”。整屏按区域局部重绘：
//   - 进度面板（内嵌圆角面板，不透明底）→ 播放中每 250ms 刷新 / 拖动时逐帧；
//   - 播放键与上/下一首 → 只在按下/松开/切歌时重绘；
//   - 底部音量排 → 只在音量变化时重绘；
//   - 卡片信息（文件名/格式/状态）→ 只在切歌/暂停态变化时重绘。
// 避免整屏重绘打断音频喂入节奏。
//
// 配色：深色（默认）与浅色（UI_COLOR_LIGHT）两套，控件全部为圆角 + 抗锯齿边缘。

#define MB_UI_NONE   (-1)
#define MB_UI_EXIT   (0)
#define MB_UI_SEEK   (1)
#define MB_UI_PREV   (2)
#define MB_UI_PLAY   (3)
#define MB_UI_NEXT   (4)
#define MB_UI_VOL_DN (5)
#define MB_UI_VOL_UP (6)

typedef struct {
    uint8_t bg_top[3];
    uint8_t bg_bot[3];
    uint8_t card[3];
    uint8_t card_edge[3];
    uint8_t surface[3];
    uint8_t surface_press[3];
    uint8_t accent[3];
    uint8_t accent_press[3];
    uint8_t text[3];
    uint8_t dim[3];
    uint8_t track[3];
    uint8_t danger[3];
    uint8_t shadow_alpha;
} Musicbox_Palette;

typedef struct {
    int32_t w;
    int32_t h;
    // 信息卡片
    int32_t card_x, card_y, card_w, card_h, card_r;
    int32_t counter_x, counter_y;
    int32_t filename_cy;
    int32_t format_cy;
    // 内嵌进度面板
    int32_t prog_x, prog_y, prog_w, prog_h, prog_r;
    int32_t time_y;
    int32_t track_x, track_y, track_w, track_h;
    int32_t thumb_r;
    // 顶栏退出按钮
    int32_t exit_x, exit_y, exit_w, exit_h, exit_r;
    // 播放控制
    int32_t play_cx, play_cy, play_r;
    int32_t prev_cx, prev_cy, prev_r;
    int32_t next_cx, next_cy, next_r;
    // 底部音量排
    int32_t vol_minus_x, vol_minus_y;
    int32_t vol_plus_x;
    int32_t vol_btn_w, vol_btn_h, vol_btn_r;
    int32_t vol_text_y;
    int32_t vol_meter_x, vol_meter_y, vol_meter_w, vol_meter_h;
    // 局部重绘区域
    int32_t topbar_bottom;
    int32_t transport_y0, transport_y1;
    int32_t volume_y0, volume_y1;
} Musicbox_Ui;

static void musicbox_palette(Global_State *gs, Musicbox_Palette *p) {
    static const Musicbox_Palette dark = {
        .bg_top      = {13, 16, 22},
        .bg_bot      = {24, 30, 40},
        .card        = {27, 33, 44},
        .card_edge   = {44, 53, 68},
        .surface     = {38, 47, 61},
        .surface_press = {62, 78, 100},
        .accent      = {56, 189, 248},
        .accent_press = {14, 140, 200},
        .text        = {237, 242, 249},
        .dim         = {140, 152, 168},
        .track       = {52, 63, 79},
        .danger      = {248, 113, 113},
        .shadow_alpha = 70,
    };
    static const Musicbox_Palette light = {
        .bg_top      = {246, 248, 252},
        .bg_bot      = {228, 235, 245},
        .card        = {255, 255, 255},
        .card_edge   = {218, 226, 238},
        .surface     = {237, 242, 249},
        .surface_press = {209, 223, 241},
        .accent      = {37, 99, 235},
        .accent_press = {23, 74, 184},
        .text        = {24, 32, 46},
        .dim         = {104, 118, 136},
        .track       = {223, 230, 240},
        .danger      = {220, 38, 38},
        .shadow_alpha = 30,
    };
    *p = (gs->ui_color_style == UI_COLOR_LIGHT) ? light : dark;
}

// 布局：320x240 设计基准，尺寸随画布宽高收缩/贴边（音乐盒只在 320x240 类设备可达，
// 这里仍按画布尺寸推导，避免写死屏幕常量导致小屏越界）。
static void musicbox_ui_layout(Nano_GFX *gfx, Musicbox_Ui *u) {
    int32_t W = (int32_t)gfx->width;
    int32_t H = (int32_t)gfx->height;
    memset(u, 0, sizeof(*u));
    u->w = W;
    u->h = H;

    // 顶栏
    u->topbar_bottom = 34;

    // 信息卡片
    u->card_x = 12;
    u->card_y = 36;
    u->card_w = W - 24;
    if (u->card_w < 96) { u->card_x = 4; u->card_w = W - 8; }
    u->card_h = 104;
    if (u->card_y + u->card_h > H - 100) u->card_h = H - 100 - u->card_y;
    if (u->card_h < 70) u->card_h = 70;
    u->card_r = 14;
    u->counter_x = u->card_x + 14;
    u->counter_y = u->card_y + 8;
    u->filename_cy = u->card_y + 35;
    u->format_cy = u->card_y + 53;

    // 内嵌进度面板
    u->prog_x = u->card_x + 10;
    u->prog_w = u->card_w - 20;
    u->prog_h = 36;
    u->prog_y = u->card_y + u->card_h - u->prog_h - 6;
    u->prog_r = 10;
    u->time_y = u->prog_y + 2;
    u->track_x = u->prog_x + 12;
    u->track_w = u->prog_w - 24;
    if (u->track_w < 20) u->track_w = 20;
    u->track_y = u->prog_y + 22;
    u->track_h = 5;
    u->thumb_r = 6;

    // 顶栏退出按钮
    u->exit_w = 58;
    u->exit_h = 24;
    u->exit_r = 12;
    u->exit_x = W - 16 - u->exit_w;
    u->exit_y = 6;

    // 播放控制（圆形按钮）
    u->play_r = 24;
    u->play_cy = u->card_y + u->card_h + 28;
    u->play_cx = W / 2;
    u->prev_r = 17;
    u->prev_cx = u->play_cx - 72;
    u->prev_cy = u->play_cy;
    u->next_r = 17;
    u->next_cx = u->play_cx + 72;
    u->next_cy = u->play_cy;

    // 底部音量排
    u->vol_btn_w = 34;
    u->vol_btn_h = 30;
    u->vol_btn_r = 10;
    u->vol_minus_x = 14;
    u->vol_minus_y = H - 36;
    u->vol_plus_x = W - 14 - u->vol_btn_w;
    u->vol_text_y = u->vol_minus_y - 2;
    u->vol_meter_x = u->vol_minus_x + u->vol_btn_w + 10;
    u->vol_meter_w = u->vol_plus_x - 10 - u->vol_meter_x;
    if (u->vol_meter_w < 40) u->vol_meter_w = 40;
    u->vol_meter_y = u->vol_minus_y + 20;
    u->vol_meter_h = 4;

    // 局部重绘区域
    u->transport_y0 = u->card_y + u->card_h + 1;
    u->transport_y1 = u->play_cy + u->play_r + 4;
    u->volume_y0 = u->transport_y1;
    u->volume_y1 = H;
}

// 抗锯齿圆：radius = 半边长，复用圆角矩形（8x8 超采样角部），比 gfx_draw_circle_fill
// 的逐点判定边缘更平滑（播放/上一首/下一首/进度拇指都用它）。
static void musicbox_fill_circle(Nano_GFX *gfx, int32_t cx, int32_t cy, int32_t r,
                                 uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    if (r <= 0) return;
    gfx_draw_rounded_rectangle(gfx, cx - r, cy - r, r * 2, r * 2, r, r, r, r,
                               red, green, blue, mode);
}

// 背景渐变：按 8px 条带填充，条带按全局 y 对齐（局部重绘与整屏绘制的同一 y 处颜色一致）
static void musicbox_bg_color(const Musicbox_Palette *p, int32_t y, int32_t h,
                              uint8_t *r, uint8_t *g, uint8_t *b) {
    int32_t t = (h > 1) ? y * 255 / (h - 1) : 0;
    if (t < 0) t = 0;
    if (t > 255) t = 255;
    *r = (uint8_t)((int32_t)p->bg_top[0] + ((int32_t)p->bg_bot[0] - (int32_t)p->bg_top[0]) * t / 255);
    *g = (uint8_t)((int32_t)p->bg_top[1] + ((int32_t)p->bg_bot[1] - (int32_t)p->bg_top[1]) * t / 255);
    *b = (uint8_t)((int32_t)p->bg_top[2] + ((int32_t)p->bg_bot[2] - (int32_t)p->bg_top[2]) * t / 255);
}

static void musicbox_fill_bg(Nano_GFX *gfx, const Musicbox_Palette *p,
                             int32_t x, int32_t y, int32_t w, int32_t h) {
    int32_t band0 = y - (y % 8);
    for (int32_t by = band0; by < y + h; by += 8) {
        int32_t yy = (by < y) ? y : by;
        int32_t ye = by + 8;
        if (ye > y + h) ye = y + h;
        if (ye <= yy) continue;
        uint8_t r, g, b;
        musicbox_bg_color(p, by + 4, (int32_t)gfx->height, &r, &g, &b);
        gfx_draw_rectangle(gfx, (uint32_t)x, (uint32_t)yy, (uint32_t)w,
                           (uint32_t)(ye - yy), r, g, b, 1);
    }
}

static void musicbox_draw_text_right(Nano_GFX *gfx, uint32_t font, const wchar_t *s,
                                     int32_t right_x, int32_t y_top,
                                     uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    int32_t tw = gfx_font_measure_text(font, (wchar_t *)s);
    gfx_font_draw_text(gfx, font, (wchar_t *)s, right_x - tw, y_top, red, green, blue, mode);
}

// 超宽文件名截断为“前缀…”后居中（保留开头比保留扩展名更利于辨认曲名）
static void musicbox_draw_text_ellipsis_centered(Nano_GFX *gfx, uint32_t font, const wchar_t *s,
                                                 int32_t cx, int32_t cy, int32_t max_w,
                                                 uint8_t red, uint8_t green, uint8_t blue, uint8_t mode) {
    wchar_t buf[96];
    int32_t total = gfx_font_measure_text(font, (wchar_t *)s);
    if (total <= max_w) {
        gfx_font_draw_text_centered(gfx, font, (wchar_t *)s, cx, cy, red, green, blue, mode);
        return;
    }
    int32_t ell_w = gfx_font_char_advance(font, (uint32_t)L'…');
    int32_t w = 0;
    uint32_t n = 0;
    while (s[n] != L'\0' && n + 2 < sizeof(buf) / sizeof(buf[0])) {
        int32_t adv = gfx_font_char_advance(font, (uint32_t)s[n]);
        if (w + adv + ell_w > max_w) break;
        buf[n] = s[n];
        w += adv;
        n++;
    }
    buf[n++] = L'…';
    buf[n] = L'\0';
    gfx_font_draw_text_centered(gfx, font, buf, cx, cy, red, green, blue, mode);
}

static void musicbox_fmt_time(uint32_t ms, wchar_t *out, size_t n) {
    uint32_t sec = ms / 1000u;
    swprintf(out, n, L"%lu:%02lu", (unsigned long)(sec / 60u), (unsigned long)(sec % 60u));
}

// 采样率显示为 44.1/48.0/8.0 kHz（0.1kHz 四舍五入，避免嵌入式浮点 printf）
static void musicbox_fmt_rate(uint32_t hz, wchar_t *out, size_t n) {
    uint32_t khz10 = (hz + 50u) / 100u;
    swprintf(out, n, L"%lu.%lu kHz", (unsigned long)(khz10 / 10u), (unsigned long)(khz10 % 10u));
}

// 第二行信息：文件格式参数
static void musicbox_format_info(wchar_t *out, size_t n) {
    if (s_open_failed || s_dec == NULL) {
        wcsncpy(out, L"文件格式不支持或已损坏", n);
        out[n - 1] = L'\0';
        return;
    }
    wchar_t rate[24];
    musicbox_fmt_rate(s_dec->sample_rate, rate, sizeof(rate) / sizeof(rate[0]));
    int32_t channels = (s_dec->type == MUSICBOX_DEC_WAV) ? (int32_t)s_dec->wav_channels : s_dec->mp3_channels;
    const wchar_t *ch = (channels == 1) ? L"单声道" : L"立体声";
    if (s_dec->type == MUSICBOX_DEC_WAV) {
        swprintf(out, n, L"WAV  ·  %ls  ·  %d bit  ·  %ls",
                 rate, (int)s_dec->wav_bits, ch);
    }
    else if (s_dec->mp3_bitrate_kbps > 0) {
        swprintf(out, n, L"MP3  ·  %ls  ·  %d kbps  ·  %ls",
                 rate, (int)s_dec->mp3_bitrate_kbps, ch);
    }
    else {
        swprintf(out, n, L"MP3  ·  %ls  ·  %ls", rate, ch);
    }
}

// ---------------- 矢量图标（不依赖字库里的 ▶/⏸ 等码点） ----------------

static void musicbox_draw_play_icon(Nano_GFX *gfx, int32_t cx, int32_t cy, int32_t r,
                                    int32_t playing, uint8_t red, uint8_t green, uint8_t blue) {
    if (playing) {
        // 暂停：两根圆角竖条
        int32_t bh = r;
        int32_t bw = r / 4;
        if (bw < 3) bw = 3;
        int32_t gap = r / 3;
        gfx_draw_rounded_rectangle(gfx, cx - gap / 2 - bw, cy - bh / 2, bw, bh,
                                   2, 2, 2, 2, red, green, blue, 1);
        gfx_draw_rounded_rectangle(gfx, cx + gap / 2, cy - bh / 2, bw, bh,
                                   2, 2, 2, 2, red, green, blue, 1);
    }
    else {
        // 播放：向右的三角形（视觉重心略右移）
        int32_t h = r * 4 / 5;
        int32_t w = r * 3 / 4;
        gfx_draw_triangle(gfx, (uint32_t)(cx - w / 3), (uint32_t)(cy - h / 2),
                          (uint32_t)(cx - w / 3), (uint32_t)(cy + h / 2),
                          (uint32_t)(cx + w * 2 / 3), (uint32_t)cy,
                          0, red, green, blue, 1);
    }
}

static void musicbox_draw_skip_icon(Nano_GFX *gfx, int32_t cx, int32_t cy, int32_t r,
                                    int32_t dir, uint8_t red, uint8_t green, uint8_t blue) {
    int32_t h = r * 4 / 5;
    if (h < 6) h = 6;
    int32_t bar_w = (r >= 20) ? 3 : 2;
    int32_t bar_x = (dir < 0) ? (cx - r * 9 / 16) : (cx + r * 7 / 16);
    gfx_draw_rounded_rectangle(gfx, bar_x, cy - h / 2, bar_w, h,
                               1, 1, 1, 1, red, green, blue, 1);
    if (dir < 0) {
        gfx_draw_triangle(gfx, (uint32_t)(cx + r * 7 / 16), (uint32_t)(cy - h / 2),
                          (uint32_t)(cx + r * 7 / 16), (uint32_t)(cy + h / 2),
                          (uint32_t)(cx - r * 5 / 16), (uint32_t)cy,
                          0, red, green, blue, 1);
    }
    else {
        gfx_draw_triangle(gfx, (uint32_t)(cx - r * 7 / 16), (uint32_t)(cy - h / 2),
                          (uint32_t)(cx - r * 7 / 16), (uint32_t)(cy + h / 2),
                          (uint32_t)(cx + r * 5 / 16), (uint32_t)cy,
                          0, red, green, blue, 1);
    }
}

static void musicbox_draw_vol_icon(Nano_GFX *gfx, int32_t cx, int32_t cy, int32_t plus,
                                   uint8_t red, uint8_t green, uint8_t blue) {
    int32_t len = 12;
    int32_t th = 2;
    gfx_draw_rounded_rectangle(gfx, cx - len / 2, cy - th / 2, len, th,
                               1, 1, 1, 1, red, green, blue, 1);
    if (plus) {
        gfx_draw_rounded_rectangle(gfx, cx - th / 2, cy - len / 2, th, len,
                                   1, 1, 1, 1, red, green, blue, 1);
    }
}

// ---------------- 各区域绘制 ----------------

static void musicbox_draw_status(Nano_GFX *gfx, const Musicbox_Ui *u, const Musicbox_Palette *p,
                                 int32_t clear_first) {
    const wchar_t *st;
    const uint8_t *dot;
    const uint8_t *st_color;
    if (s_open_failed || s_dec == NULL) {
        st = L"错误";
        dot = p->danger;
        st_color = p->danger;
    }
    else if (s_playing) {
        st = L"播放中";
        dot = p->accent;
        st_color = p->dim;
    }
    else {
        st = L"已暂停";
        dot = p->dim;
        st_color = p->dim;
    }
    int32_t tw = gfx_font_measure_text(GFX_FONT_ALPHA_12, (wchar_t *)st);
    int32_t right = u->card_x + u->card_w - 14;
    int32_t tx = right - tw;
    if (clear_first) {
        int32_t cx0 = tx - 16;
        if (cx0 < u->card_x + 1) cx0 = u->card_x + 1;
        gfx_draw_rectangle(gfx, (uint32_t)cx0, (uint32_t)(u->counter_y - 2),
                           (uint32_t)(right - cx0 + 2), 19,
                           p->card[0], p->card[1], p->card[2], 1);
    }
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, (wchar_t *)st, tx, u->counter_y,
                       st_color[0], st_color[1], st_color[2], 1);
    musicbox_fill_circle(gfx, tx - 10, u->counter_y + 7, 3,
                         dot[0], dot[1], dot[2], 1);
}

static void musicbox_draw_topbar(Nano_GFX *gfx, const Musicbox_Ui *u, const Musicbox_Palette *p) {
    // 标题左侧的强调色竖条
    gfx_draw_rounded_rectangle(gfx, 12, 13, 3, 12, 1, 1, 1, 1,
                               p->accent[0], p->accent[1], p->accent[2], 1);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, (wchar_t *)L"音乐盒", 20, 10,
                       p->text[0], p->text[1], p->text[2], 1);

    // 退出按钮（按下时填充强调色）
    int32_t pressed = (s_ui_pressed == MB_UI_EXIT);
    const uint8_t *bg = pressed ? p->accent : p->surface;
    uint8_t fg[3];
    if (pressed) { fg[0] = 255; fg[1] = 255; fg[2] = 255; }
    else         { fg[0] = p->text[0]; fg[1] = p->text[1]; fg[2] = p->text[2]; }
    gfx_draw_rounded_rectangle(gfx, u->exit_x, u->exit_y, u->exit_w, u->exit_h,
                               u->exit_r, u->exit_r, u->exit_r, u->exit_r,
                               bg[0], bg[1], bg[2], 1);
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)L"退出",
                                u->exit_x + u->exit_w / 2, u->exit_y + u->exit_h / 2,
                                fg[0], fg[1], fg[2], 1);
}

static void musicbox_draw_card(Nano_GFX *gfx, const Musicbox_Ui *u, const Musicbox_Palette *p) {
    // 边框 + 内缩 1px 的卡片底
    gfx_draw_rounded_rectangle(gfx, u->card_x, u->card_y, u->card_w, u->card_h,
                               u->card_r, u->card_r, u->card_r, u->card_r,
                               p->card_edge[0], p->card_edge[1], p->card_edge[2], 1);
    gfx_draw_rounded_rectangle(gfx, u->card_x + 1, u->card_y + 1, u->card_w - 2, u->card_h - 2,
                               u->card_r - 1, u->card_r - 1, u->card_r - 1, u->card_r - 1,
                               p->card[0], p->card[1], p->card[2], 1);

    // 序号：当前曲目用强调色，总数用弱化色
    wchar_t idx[16];
    wchar_t total[24];
    if (s_track_idx >= 0 && s_track_idx < s_list_count) {
        swprintf(idx, sizeof(idx) / sizeof(idx[0]), L"%d", (int)(s_track_idx + 1));
    }
    else {
        wcscpy(idx, L"-");
    }
    swprintf(total, sizeof(total) / sizeof(total[0]), L" / %d", (int)s_list_count);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, idx, u->counter_x, u->counter_y,
                       p->accent[0], p->accent[1], p->accent[2], 1);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, total,
                       u->counter_x + gfx_font_measure_text(GFX_FONT_ALPHA_12, idx), u->counter_y,
                       p->dim[0], p->dim[1], p->dim[2], 1);

    musicbox_draw_status(gfx, u, p, 0);

    // 文件名
    const wchar_t *name = L"";
    if (s_open_failed || s_dec == NULL) {
        name = L"无法播放该文件";
    }
    else if (s_track_idx >= 0 && s_track_idx < s_list_count && s_list_w != NULL && s_list_w[s_track_idx] != NULL) {
        name = s_list_w[s_track_idx];
    }
    musicbox_draw_text_ellipsis_centered(gfx, GFX_FONT_ALPHA_16, name, u->w / 2, u->filename_cy,
                                         u->card_w - 28, p->text[0], p->text[1], p->text[2], 1);

    // 格式参数
    wchar_t info[96];
    musicbox_format_info(info, sizeof(info) / sizeof(info[0]));
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, info, u->w / 2, u->format_cy,
                                p->dim[0], p->dim[1], p->dim[2], 1);
}

// 进度面板：不透明底，整块重绘（拖动/定时刷新都只动这一块）。
// 先用卡片色清底再画圆角面板：圆角抗锯齿像素每次都以卡片底色为背景重新混合，
// 避免高频局部重绘使面板四角逐次叠加变深/变糊。
static void musicbox_draw_progress(Nano_GFX *gfx, const Musicbox_Ui *u, const Musicbox_Palette *p) {
    gfx_draw_rectangle(gfx, (uint32_t)u->prog_x, (uint32_t)u->prog_y,
                       (uint32_t)u->prog_w, (uint32_t)u->prog_h,
                       p->card[0], p->card[1], p->card[2], 1);
    gfx_draw_rounded_rectangle(gfx, u->prog_x, u->prog_y, u->prog_w, u->prog_h,
                               u->prog_r, u->prog_r, u->prog_r, u->prog_r,
                               p->surface[0], p->surface[1], p->surface[2], 1);

    float ratio = (s_seek_preview >= 0.0f) ? s_seek_preview : musicbox_progress();
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;

    uint32_t dur_ms = musicbox_duration_ms();
    uint32_t el_ms = (s_seek_preview >= 0.0f && dur_ms > 0)
                   ? (uint32_t)((float)dur_ms * ratio) : musicbox_elapsed_ms();
    if (dur_ms > 0 && el_ms > dur_ms) el_ms = dur_ms;

    wchar_t t_el[16];
    wchar_t t_dur[16];
    musicbox_fmt_time(el_ms, t_el, sizeof(t_el) / sizeof(t_el[0]));
    musicbox_fmt_time(dur_ms, t_dur, sizeof(t_dur) / sizeof(t_dur[0]));

    int32_t text_left = u->prog_x + 8;
    int32_t text_right = u->prog_x + u->prog_w - 8;
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, t_el, text_left, u->time_y,
                       p->text[0], p->text[1], p->text[2], 1);
    musicbox_draw_text_right(gfx, GFX_FONT_ALPHA_12, t_dur, text_right, u->time_y,
                             p->dim[0], p->dim[1], p->dim[2], 1);

    // 轨道 + 已播放比例
    int32_t rh = u->track_h / 2;
    if (rh < 1) rh = 1;
    gfx_draw_rounded_rectangle(gfx, u->track_x, u->track_y, u->track_w, u->track_h,
                               rh, rh, rh, rh, p->track[0], p->track[1], p->track[2], 1);
    int32_t fill_w = (int32_t)((float)u->track_w * ratio + 0.5f);
    if (fill_w > u->track_w) fill_w = u->track_w;
    if (fill_w > 0) {
        gfx_draw_rounded_rectangle(gfx, u->track_x, u->track_y, fill_w, u->track_h,
                                   rh, rh, rh, rh, p->accent[0], p->accent[1], p->accent[2], 1);
    }

    // 拇指：强调色光晕 + 强调色环 + 白色内芯
    int32_t thumb_cx = u->track_x + fill_w;
    int32_t thumb_cy = u->track_y + u->track_h / 2;
    musicbox_fill_circle(gfx, thumb_cx, thumb_cy, u->thumb_r + 3,
                         p->accent[0], p->accent[1], p->accent[2], 40);
    musicbox_fill_circle(gfx, thumb_cx, thumb_cy, u->thumb_r,
                         p->accent[0], p->accent[1], p->accent[2], 1);
    musicbox_fill_circle(gfx, thumb_cx, thumb_cy, u->thumb_r - 2, 250, 250, 252, 1);
}

static void musicbox_draw_transport(Nano_GFX *gfx, const Musicbox_Ui *u, const Musicbox_Palette *p) {
    int32_t play_ok = !(s_open_failed || s_dec == NULL);
    int32_t playing = (play_ok && s_playing);

    // 上/下一首
    const uint8_t *prev_bg = (s_ui_pressed == MB_UI_PREV) ? p->surface_press : p->surface;
    musicbox_fill_circle(gfx, u->prev_cx, u->prev_cy, u->prev_r,
                         prev_bg[0], prev_bg[1], prev_bg[2], 1);
    musicbox_draw_skip_icon(gfx, u->prev_cx, u->prev_cy, u->prev_r, -1,
                            p->text[0], p->text[1], p->text[2]);

    const uint8_t *next_bg = (s_ui_pressed == MB_UI_NEXT) ? p->surface_press : p->surface;
    musicbox_fill_circle(gfx, u->next_cx, u->next_cy, u->next_r,
                         next_bg[0], next_bg[1], next_bg[2], 1);
    musicbox_draw_skip_icon(gfx, u->next_cx, u->next_cy, u->next_r, 1,
                            p->text[0], p->text[1], p->text[2]);

    // 播放/暂停主按钮
    if (play_ok) {
        const uint8_t *ac = (s_ui_pressed == MB_UI_PLAY) ? p->accent_press : p->accent;
        // 轻微投影，让按钮从渐变背景上“浮”起来
        musicbox_fill_circle(gfx, u->play_cx, u->play_cy + 2, u->play_r + 2,
                             0, 0, 0, p->shadow_alpha);
        musicbox_fill_circle(gfx, u->play_cx, u->play_cy, u->play_r,
                             ac[0], ac[1], ac[2], 1);
        musicbox_draw_play_icon(gfx, u->play_cx, u->play_cy, u->play_r, playing,
                                255, 255, 255);
    }
    else {
        musicbox_fill_circle(gfx, u->play_cx, u->play_cy, u->play_r,
                             p->surface[0], p->surface[1], p->surface[2], 1);
        musicbox_draw_play_icon(gfx, u->play_cx, u->play_cy, u->play_r, 0,
                                p->dim[0], p->dim[1], p->dim[2]);
    }
}

static void musicbox_draw_volume(Nano_GFX *gfx, const Musicbox_Ui *u, const Musicbox_Palette *p) {
    // 左侧名称 + 右侧百分比
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, (wchar_t *)L"音量", u->vol_meter_x, u->vol_text_y,
                       p->dim[0], p->dim[1], p->dim[2], 1);
    wchar_t val[16];
    int32_t pct = (int)s_volume * 100 / MUSICBOX_VOLUME_MAX;
    swprintf(val, sizeof(val) / sizeof(val[0]), L"%d%%", pct);
    musicbox_draw_text_right(gfx, GFX_FONT_ALPHA_12, val, u->vol_meter_x + u->vol_meter_w,
                             u->vol_text_y, p->text[0], p->text[1], p->text[2], 1);

    // 音量条
    int32_t rh = u->vol_meter_h / 2;
    if (rh < 1) rh = 1;
    gfx_draw_rounded_rectangle(gfx, u->vol_meter_x, u->vol_meter_y, u->vol_meter_w, u->vol_meter_h,
                               rh, rh, rh, rh, p->track[0], p->track[1], p->track[2], 1);
    int32_t fill_w = u->vol_meter_w * (int32_t)s_volume / MUSICBOX_VOLUME_MAX;
    if (fill_w > 0) {
        gfx_draw_rounded_rectangle(gfx, u->vol_meter_x, u->vol_meter_y, fill_w, u->vol_meter_h,
                                   rh, rh, rh, rh, p->accent[0], p->accent[1], p->accent[2], 1);
    }

    // 减 / 加按钮
    int32_t can_dn = (s_volume > 0);
    int32_t can_up = (s_volume < MUSICBOX_VOLUME_MAX);
    const uint8_t *minus_bg = (s_ui_pressed == MB_UI_VOL_DN) ? p->surface_press : p->surface;
    const uint8_t *plus_bg  = (s_ui_pressed == MB_UI_VOL_UP) ? p->surface_press : p->surface;
    gfx_draw_rounded_rectangle(gfx, u->vol_minus_x, u->vol_minus_y, u->vol_btn_w, u->vol_btn_h,
                               u->vol_btn_r, u->vol_btn_r, u->vol_btn_r, u->vol_btn_r,
                               minus_bg[0], minus_bg[1], minus_bg[2], 1);
    gfx_draw_rounded_rectangle(gfx, u->vol_plus_x, u->vol_minus_y, u->vol_btn_w, u->vol_btn_h,
                               u->vol_btn_r, u->vol_btn_r, u->vol_btn_r, u->vol_btn_r,
                               plus_bg[0], plus_bg[1], plus_bg[2], 1);
    const uint8_t *minus_fg = can_dn ? p->text : p->dim;
    const uint8_t *plus_fg  = can_up ? p->text : p->dim;
    musicbox_draw_vol_icon(gfx, u->vol_minus_x + u->vol_btn_w / 2, u->vol_minus_y + u->vol_btn_h / 2,
                           0, minus_fg[0], minus_fg[1], minus_fg[2]);
    musicbox_draw_vol_icon(gfx, u->vol_plus_x + u->vol_btn_w / 2, u->vol_minus_y + u->vol_btn_h / 2,
                           1, plus_fg[0], plus_fg[1], plus_fg[2]);
}

// 全量绘制（进入播放态 / 切歌 / 暂停态切换 / 打开失败）
static void musicbox_draw_all(Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    Musicbox_Ui u;
    Musicbox_Palette p;
    musicbox_ui_layout(gfx, &u);
    musicbox_palette(global_state, &p);

    musicbox_fill_bg(gfx, &p, 0, 0, u.w, u.h);
    musicbox_draw_topbar(gfx, &u, &p);
    musicbox_draw_card(gfx, &u, &p);
    musicbox_draw_progress(gfx, &u, &p);
    musicbox_draw_transport(gfx, &u, &p);
    musicbox_draw_volume(gfx, &u, &p);
    gfx_refresh(gfx);
}

// ---------------- 局部重绘 ----------------

static void musicbox_redraw_topbar(Global_State *gs) {
    Musicbox_Ui u;
    Musicbox_Palette p;
    musicbox_ui_layout(gs->gfx, &u);
    musicbox_palette(gs, &p);
    musicbox_fill_bg(gs->gfx, &p, 0, 0, u.w, u.topbar_bottom);
    musicbox_draw_topbar(gs->gfx, &u, &p);
    gfx_refresh(gs->gfx);
}

static void musicbox_redraw_transport(Global_State *gs) {
    Musicbox_Ui u;
    Musicbox_Palette p;
    musicbox_ui_layout(gs->gfx, &u);
    musicbox_palette(gs, &p);
    musicbox_fill_bg(gs->gfx, &p, 0, u.transport_y0, u.w, u.transport_y1 - u.transport_y0);
    musicbox_draw_transport(gs->gfx, &u, &p);
    gfx_refresh(gs->gfx);
}

static void musicbox_redraw_volume(Global_State *gs) {
    Musicbox_Ui u;
    Musicbox_Palette p;
    musicbox_ui_layout(gs->gfx, &u);
    musicbox_palette(gs, &p);
    musicbox_fill_bg(gs->gfx, &p, 0, u.volume_y0, u.w, u.volume_y1 - u.volume_y0);
    musicbox_draw_volume(gs->gfx, &u, &p);
    gfx_refresh(gs->gfx);
}

static void musicbox_redraw_progress(Global_State *gs) {
    Musicbox_Ui u;
    Musicbox_Palette p;
    musicbox_ui_layout(gs->gfx, &u);
    musicbox_palette(gs, &p);
    musicbox_draw_progress(gs->gfx, &u, &p);
    gfx_refresh(gs->gfx);
}

static void musicbox_redraw_status(Global_State *gs) {
    Musicbox_Ui u;
    Musicbox_Palette p;
    musicbox_ui_layout(gs->gfx, &u);
    musicbox_palette(gs, &p);
    musicbox_draw_status(gs->gfx, &u, &p, 1);
    gfx_refresh(gs->gfx);
}

static void musicbox_redraw_control(Global_State *gs, int32_t id) {
    switch (id) {
        case MB_UI_EXIT:   musicbox_redraw_topbar(gs); break;
        case MB_UI_SEEK:   musicbox_redraw_progress(gs); break;
        case MB_UI_PREV:
        case MB_UI_PLAY:
        case MB_UI_NEXT:   musicbox_redraw_transport(gs); break;
        case MB_UI_VOL_DN:
        case MB_UI_VOL_UP: musicbox_redraw_volume(gs); break;
        default: break;
    }
}

// ---------------- 命中判定（与绘制共用 Musicbox_Ui） ----------------

static int32_t musicbox_rect_hit(int32_t x, int32_t y, int32_t rx, int32_t ry, int32_t rw, int32_t rh) {
    return (x >= rx && x < rx + rw && y >= ry && y < ry + rh);
}

static int32_t musicbox_circle_hit(int32_t x, int32_t y, int32_t cx, int32_t cy, int32_t r) {
    int32_t dx = x - cx;
    int32_t dy = y - cy;
    return (dx * dx + dy * dy) <= r * r;
}

static int32_t musicbox_ui_hit(const Musicbox_Ui *u, int32_t x, int32_t y) {
    // 顶栏退出
    if (musicbox_rect_hit(x, y, u->exit_x - 2, u->exit_y - 2, u->exit_w + 4, u->exit_h + 4)) {
        return MB_UI_EXIT;
    }
    // 进度面板（整块可点可拖，比细轨道好按）
    if (musicbox_rect_hit(x, y, u->prog_x - 6, u->prog_y - 4, u->prog_w + 12, u->prog_h + 8)) {
        return MB_UI_SEEK;
    }
    // 播放控制
    if (musicbox_circle_hit(x, y, u->prev_cx, u->prev_cy, u->prev_r + 6)) return MB_UI_PREV;
    if (musicbox_circle_hit(x, y, u->play_cx, u->play_cy, u->play_r + 5)) return MB_UI_PLAY;
    if (musicbox_circle_hit(x, y, u->next_cx, u->next_cy, u->next_r + 6)) return MB_UI_NEXT;
    // 音量
    if (musicbox_rect_hit(x, y, u->vol_minus_x - 2, u->vol_minus_y - 2, u->vol_btn_w + 4, u->vol_btn_h + 4)) {
        return MB_UI_VOL_DN;
    }
    if (musicbox_rect_hit(x, y, u->vol_plus_x - 2, u->vol_minus_y - 2, u->vol_btn_w + 4, u->vol_btn_h + 4)) {
        return MB_UI_VOL_UP;
    }
    return MB_UI_NONE;
}

// 进度条像素 x → 比例（0.0~1.0）
static float musicbox_ratio_from_x(const Musicbox_Ui *u, int32_t x) {
    int32_t px = x;
    if (px < u->track_x) px = u->track_x;
    if (px > u->track_x + u->track_w) px = u->track_x + u->track_w;
    if (u->track_w <= 0) return 0.0f;
    return (float)(px - u->track_x) / (float)u->track_w;
}

// ---------------- 状态操作（触屏与实体键共用） ----------------

static void musicbox_change_volume(int32_t dir) {
    int32_t v = (int32_t)s_volume + dir * MUSICBOX_VOLUME_STEP;
    if (v < 0) v = 0;
    if (v > MUSICBOX_VOLUME_MAX) v = MUSICBOX_VOLUME_MAX;
    if ((uint8_t)v == s_volume) return;
    s_volume = (uint8_t)v;
    audio_out_set_volume(s_volume);
}

static void musicbox_toggle_play(void) {
    if (s_open_failed || s_dec == NULL) return;
    s_playing = !s_playing;
    if (!s_playing) audio_out_stop();
    s_last_ui_ts = 0; // 恢复播放后立即刷新一次进度
}

// 上/下一曲（列表循环；连续失败的曲目跳过，全部失败则进入“无法播放”提示页）
static int32_t musicbox_switch_track(int32_t step) {
    if (s_list_count <= 0) {
        s_open_failed = 1;
        return -1;
    }
    audio_out_stop();
    int32_t idx = s_track_idx;
    for (int32_t tries = 0; tries < s_list_count; tries++) {
        idx = (idx + step + s_list_count) % s_list_count;
        if (musicbox_start_track(idx) == 0) {
            s_open_failed = 0;
            return 0;
        }
    }
    s_open_failed = 1;
    return -1;
}

// 返回文件列表前同步高亮：让列表停在正在播放的曲目上（用户从列表深处点入某曲后，
// 返回时仍能看到它；列表控件会随状态重绘自然展示）。
static void musicbox_sync_menu_selection(Global_State *gs) {
    Widget_Menu_State *ms = gs->w_menu_main;
    if (ms == NULL || s_list_count <= 0) return;
    if (s_track_idx < 0 || s_track_idx >= s_list_count) return;
    ms->current_item_index = s_track_idx;
    int32_t per = ms->items_per_page;
    if (per <= 0) per = 1;
    if (ms->first_item_intex > s_track_idx) ms->first_item_intex = s_track_idx;
    if (ms->first_item_intex + per <= s_track_idx) ms->first_item_intex = s_track_idx - per + 1;
    if (ms->first_item_intex < 0) ms->first_item_intex = 0;
    // 吸附回整行，并清掉可能残留的惯性/回弹动画（避免返回列表后被继续滚动走）
    ms->scroll_sub_offset = 0;
    ms->fling_velocity = 0.0f;
    ms->bounce_offset_px = 0.0f;
    ms->bounce_velocity = 0.0f;
    ms->touch_active = 0;
}

// 触屏输入：DOWN 锁存/高亮，UP 结算；进度条拖动与音量连发在中间帧处理
static int32_t musicbox_handle_touch(Key_Event *key_event, Global_State *global_state) {
    Musicbox_Ui u;
    musicbox_ui_layout(global_state->gfx, &u);
    uint64_t now = global_state->timestamp;

    // ---- 按下沿 ----
    if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
        int32_t hit = musicbox_ui_hit(&u, key_event->touch_down_x, key_event->touch_down_y);
        s_ui_pressed = hit;

        if (hit == MB_UI_SEEK) {
            s_seek_active = 1;
            s_seek_preview = musicbox_ratio_from_x(&u, key_event->touch_down_x);
        }
        else if (hit == MB_UI_VOL_DN || hit == MB_UI_VOL_UP) {
            int32_t dir = (hit == MB_UI_VOL_UP) ? 1 : -1;
            s_vol_hold = dir;
            s_vol_next_ts = now + MUSICBOX_VOL_HOLD_MS;
            musicbox_change_volume(dir);
        }
        if (hit != MB_UI_NONE) musicbox_redraw_control(global_state, hit);
    }

    // ---- 进度条拖动：只更新预览，不 seek ----
    if (s_seek_active && key_event->is_touching) {
        float r = musicbox_ratio_from_x(&u, key_event->touch_x);
        if (r != s_seek_preview) {
            s_seek_preview = r;
            musicbox_redraw_progress(global_state);
        }
    }
    // UP 边沿异常丢失时的电平兜底：手指已离开则按最后一次预览位置结算
    if (s_seek_active && !key_event->is_touching && !(key_event->touch_edge & TOUCH_EDGE_UP)) {
        float r = (s_seek_preview >= 0.0f) ? s_seek_preview
                                           : musicbox_ratio_from_x(&u, key_event->touch_x);
        s_seek_active = 0;
        s_seek_preview = -1.0f;
        s_ui_pressed = MB_UI_NONE;
        if (!(s_open_failed || s_dec == NULL)) musicbox_seek_ratio(r);
        musicbox_redraw_progress(global_state);
    }

    // ---- 音量键按住连发（滑出按钮即停止） ----
    if (s_vol_hold != 0) {
        int32_t want = (s_vol_hold > 0) ? MB_UI_VOL_UP : MB_UI_VOL_DN;
        if (!key_event->is_touching) {
            s_vol_hold = 0;
        }
        else {
            int32_t hit = musicbox_ui_hit(&u, key_event->touch_x, key_event->touch_y);
            if (hit != want) {
                s_vol_hold = 0;
            }
            else if ((int64_t)now >= (int64_t)s_vol_next_ts) {
                s_vol_next_ts = now + MUSICBOX_VOL_REPEAT_MS;
                musicbox_change_volume(s_vol_hold);
                musicbox_redraw_volume(global_state);
            }
        }
    }

    // ---- 松开沿：结算动作 ----
    if (key_event->touch_edge & TOUCH_EDGE_UP) {
        int32_t down_hit = s_ui_pressed;
        int32_t up_hit = musicbox_ui_hit(&u, key_event->touch_x, key_event->touch_y);
        int32_t action = (down_hit != MB_UI_NONE && down_hit == up_hit) ? down_hit : MB_UI_NONE;
        s_ui_pressed = MB_UI_NONE;
        s_vol_hold = 0;

        // 进度条：松手沿才真正 seek（拖动中不反复重开文件）
        if (s_seek_active) {
            s_seek_active = 0;
            float r = musicbox_ratio_from_x(&u, key_event->touch_x);
            s_seek_preview = -1.0f;
            if (!(s_open_failed || s_dec == NULL)) musicbox_seek_ratio(r);
            musicbox_redraw_progress(global_state);
        }

        switch (action) {
            case MB_UI_EXIT:
                musicbox_sync_menu_selection(global_state);
                musicbox_player_cleanup();
                return STATE_MUSICBOX_MENU;
            case MB_UI_PLAY:
                musicbox_toggle_play();
                musicbox_redraw_transport(global_state);
                musicbox_redraw_status(global_state);
                break;
            case MB_UI_PREV:
                musicbox_switch_track(-1);
                musicbox_draw_all(global_state);
                break;
            case MB_UI_NEXT:
                musicbox_switch_track(1);
                musicbox_draw_all(global_state);
                break;
            case MB_UI_VOL_DN:
            case MB_UI_VOL_UP:
                // 音量已在按下沿/连发中生效，这里只清除高亮
                musicbox_redraw_volume(global_state);
                break;
            case MB_UI_SEEK:
                break; // 已在上方结算
            default:
                // 未命中，或按下/松开不在同一控件（拖出取消）：清除残留高亮
                if (down_hit != MB_UI_NONE) musicbox_redraw_control(global_state, down_hit);
                if (up_hit != MB_UI_NONE && up_hit != down_hit) {
                    musicbox_redraw_control(global_state, up_hit);
                }
                break;
        }
    }
    return STATE_MUSICBOX_PLAYING;
}

// 实体键盘兜底：保留原有快捷键；触屏派生的软按键（is_soft_key != 0）一律不响应，
// 配合 ui_app_state_is_menu 抑制表实现“十六宫格软按键彻底退场”。
static int32_t musicbox_handle_keys(Key_Event *key_event, Global_State *global_state) {
    if (key_event->is_soft_key != 0) return STATE_MUSICBOX_PLAYING;
    if (key_event->key_edge >= 0) return STATE_MUSICBOX_PLAYING;

    // 打开失败/内存不足：保持旧行为——任一实体键返回文件列表
    if (s_open_failed || s_dec == NULL) {
        musicbox_sync_menu_selection(global_state);
        musicbox_player_cleanup();
        return STATE_MUSICBOX_MENU;
    }

    switch (key_event->key_code) {
        case NANO_KEY_esc:
            musicbox_sync_menu_selection(global_state);
            musicbox_player_cleanup();
            return STATE_MUSICBOX_MENU;
        case NANO_KEY_enter:
            musicbox_toggle_play();
            musicbox_redraw_transport(global_state);
            musicbox_redraw_status(global_state);
            break;
        case NANO_KEY_left:
            musicbox_change_volume(-1);
            musicbox_redraw_volume(global_state);
            break;
        case NANO_KEY_right:
            musicbox_change_volume(1);
            musicbox_redraw_volume(global_state);
            break;
        case NANO_KEY_4:
            musicbox_switch_track(-1);
            musicbox_draw_all(global_state);
            break;
        case NANO_KEY_6:
            musicbox_switch_track(1);
            musicbox_draw_all(global_state);
            break;
        default:
            break;
    }
    return STATE_MUSICBOX_PLAYING;
}

// 播放态首次获焦：分配资源并启动当前曲目
void ui_musicbox_playing_on_enter(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    s_ui_pressed = MB_UI_NONE;
    s_seek_active = 0;
    s_seek_preview = -1.0f;
    s_vol_hold = 0;

    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++)
        s_pcm[i] = (int16_t *)platform_calloc(MUSICBOX_PCM_BLOCK, sizeof(int16_t));
    s_scratch = (uint8_t *)platform_calloc(MUSICBOX_SCRATCH_CAP, 1);
    s_mp3_scratch = platform_calloc(1, minimp3_scratch_size());
    if (s_pcm[0] == NULL || s_pcm[1] == NULL || s_scratch == NULL || s_mp3_scratch == NULL) {
        printf("musicbox: alloc FAILED (oom)\n");
        musicbox_player_cleanup();
        s_open_failed = 1;
        musicbox_draw_all(global_state);
        return;
    }
    minimp3_set_scratch(s_mp3_scratch);

    s_open_failed = 0;
    if (musicbox_start_track(s_track_idx) != 0) {
        s_open_failed = 1;
    }
    musicbox_draw_all(global_state);
}

int32_t ui_musicbox_playing_event(Key_Event *key_event, Global_State *global_state) {
    // 输入优先：触屏控件与实体键都可能改变播放状态/切歌，甚至返回文件列表
    int32_t next_state = musicbox_handle_touch(key_event, global_state);
    if (next_state != STATE_MUSICBOX_PLAYING) return next_state;

    next_state = musicbox_handle_keys(key_event, global_state);
    if (next_state != STATE_MUSICBOX_PLAYING) return next_state;

    if (s_open_failed || s_dec == NULL) return STATE_MUSICBOX_PLAYING;

    // 暂停：只保持解码器/文件偏移，不再向扬声器喂数据（原实现漏了这道门，
    // 暂停后 audio_out_stop 清空队列，紧接着下面的喂入又立刻把队列填满，
    // 表现为界面已显示“已暂停”但声音继续播放）。
    if (!s_playing) return STATE_MUSICBOX_PLAYING;

    // 背压安全的喂入：队列有空槽才填充
    if (audio_out_queue_free()) {
        int32_t rc = musicbox_fill_and_play();
        if (rc == 0) {
            // EOF：自动播放下一曲（循环列表）
            audio_out_stop();
            musicbox_switch_track(1);
            musicbox_draw_all(global_state);
            return STATE_MUSICBOX_PLAYING;
        }
    }

    // 限频刷新进度面板（拖动中由触屏分支逐帧刷新，避免两处互相覆盖）
    if (s_playing && !s_seek_active
        && global_state->timestamp - s_last_ui_ts >= MUSICBOX_UI_REFRESH_MS) {
        s_last_ui_ts = global_state->timestamp;
        musicbox_redraw_progress(global_state);
    }
    return STATE_MUSICBOX_PLAYING;
}
