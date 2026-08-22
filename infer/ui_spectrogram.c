#include <math.h>

#include "ui_spectrogram.h"
#include "hal_audio_in.h"
#include "nano_fft.h"
#include "hal_key.h"

// ===============================================================================
// 音频频谱仪实现
// ===============================================================================

#define SP_FFT_SIZE   (1024)  // FFT点数（一帧音频采样数，@44.1kHz 约23ms）
#define SP_BINS       (294)   // 显示频点数（每频点1px宽；320-左侧时间轴区26=294）
#define SP_ROW_H      (2)     // 每帧声谱行高度（px），即每次上滚的行数
#define SP_PI         (3.14159265358979f)
#define SP_LOG_SCALE  (80.0f) // 对数幅度→0..255 的映射系数（log10(mag+1)*80）
#define SP_SAMPLE_RATE (44100) // 采样率（Hz），与 mic_m5core2.cpp 一致
#define SP_BIN_HZ     ((float)SP_SAMPLE_RATE / (float)SP_FFT_SIZE) // 每频点频率宽度(Hz)
#define SP_PLOT_X0    (26)    // 左侧时间轴区宽度（px）
#define SP_AXIS_TOP_H (14)    // 顶部频率轴区高度（px）

// 基准电平与动态范围调节（映射域 0..255：v_raw = log10(mag+1)*80）
//   v = (v_raw - 基准电平) * 255 / 动态范围，低于基准的信号映射为黑
//   ←/→：降低/升高基准电平；4/6：减小/增大动态范围（默认值与原行为等价）
#define SP_LEVEL_BASE_DEFAULT  (240.0f)
#define SP_LEVEL_BASE_STEP     (8.0f)
#define SP_LEVEL_BASE_MAX      (240.0f)
#define SP_LEVEL_RANGE_DEFAULT (300.0f)
#define SP_LEVEL_RANGE_STEP    (16.0f)
#define SP_LEVEL_RANGE_MIN     (32.0f)
#define SP_LEVEL_RANGE_MAX     (512.0f)
#define SP_ADJUST_OVERLAY_MS   (1500)  // 调节后数值叠加显示时长（ms）

// 工作区指针（进入频谱仪时分配于PSRAM、退出时释放；避免约18.5KB常驻内部DRAM，
// 否则挤占DMA域连续块，导致启动时帧缓冲分配失败）
static float    *s_window = NULL;     // Hann窗
static float    *s_twiddle_re = NULL; // 旋转因子（全表，按步长取用）
static float    *s_twiddle_im = NULL;
static uint16_t *s_palette = NULL;    // 热力调色板（RGB565）
static float    *s_fft_re = NULL;
static float    *s_fft_im = NULL;
static int16_t  *s_mic_buf = NULL;
static uint8_t  s_inited = 0;
static uint64_t s_last_ts = 0;        // 上一帧时间戳（用于实测帧率）
static float    s_frame_interval_ema = 23.2f; // 帧间隔指数滑动平均（ms），初值=1024/44100s
static float    s_level_base = SP_LEVEL_BASE_DEFAULT;   // 基准电平（映射域下限）
static float    s_level_range = SP_LEVEL_RANGE_DEFAULT; // 动态范围（映射域宽度）
static uint64_t s_adjust_ts = 0;      // 最近一次调节时间戳（叠加显示用）

// 分配全部工作区（可重入：已分配则复用）
static int32_t sp_alloc_buffers() {
    if (!s_window)     s_window     = (float *)platform_calloc(SP_FFT_SIZE, sizeof(float));
    if (!s_twiddle_re) s_twiddle_re = (float *)platform_calloc(SP_FFT_SIZE / 2, sizeof(float));
    if (!s_twiddle_im) s_twiddle_im = (float *)platform_calloc(SP_FFT_SIZE / 2, sizeof(float));
    if (!s_palette)    s_palette    = (uint16_t *)platform_calloc(256, sizeof(uint16_t));
    if (!s_fft_re)     s_fft_re     = (float *)platform_calloc(SP_FFT_SIZE, sizeof(float));
    if (!s_fft_im)     s_fft_im     = (float *)platform_calloc(SP_FFT_SIZE, sizeof(float));
    if (!s_mic_buf)    s_mic_buf    = (int16_t *)platform_calloc(SP_FFT_SIZE, sizeof(int16_t));
    if (!s_window || !s_twiddle_re || !s_twiddle_im || !s_palette || !s_fft_re || !s_fft_im || !s_mic_buf) {
        return -1;
    }
    return 0;
}

// 释放全部工作区
static void sp_free_buffers() {
    if (s_window)     { free(s_window);     s_window = NULL; }
    if (s_twiddle_re) { free(s_twiddle_re); s_twiddle_re = NULL; }
    if (s_twiddle_im) { free(s_twiddle_im); s_twiddle_im = NULL; }
    if (s_palette)    { free(s_palette);    s_palette = NULL; }
    if (s_fft_re)     { free(s_fft_re);     s_fft_re = NULL; }
    if (s_fft_im)     { free(s_fft_im);     s_fft_im = NULL; }
    if (s_mic_buf)    { free(s_mic_buf);    s_mic_buf = NULL; }
}

// 绘制坐标轴：顶部频率轴（kHz刻度）与左侧时间轴（按实测帧率换算每秒像素数）
static void sp_draw_axes(Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    uint32_t font_id = GFX_FONT_BITMAP_12;

    // 更新帧间隔EMA，计算每秒对应的像素数（时间轴标定依据）
    uint64_t now = global_state->timestamp;
    if (s_last_ts) {
        float dt = (float)(now - s_last_ts);
        if (dt > 0.0f && dt < 1000.0f) {
            s_frame_interval_ema = s_frame_interval_ema * 0.9f + dt * 0.1f;
        }
    }
    s_last_ts = now;
    float px_per_sec = (float)SP_ROW_H * (1000.0f / s_frame_interval_ema);

    // ---- 顶部频率轴 ----
    gfx_draw_rectangle(gfx, 0, 0, gfx->width, SP_AXIS_TOP_H, 0, 0, 0, 1);
    gfx_font_draw_text(gfx, font_id, L"0", SP_PLOT_X0 - 2, 2, 150, 150, 150, 1);
    for (int32_t fk = 2; fk <= 12; fk += 2) {
        int32_t px = SP_PLOT_X0 + (int32_t)((fk * 1000.0f / SP_BIN_HZ) + 0.5f) - 1;
        wchar_t label[8];
        swprintf(label, 8, L"%dk", (int)fk);
        gfx_draw_line(gfx, px, 0, px, 3, 120, 120, 120, 1); // 刻度线
        gfx_font_draw_text(gfx, font_id, label, px - 6, 2, 150, 150, 150, 1);
    }

    // ---- 左侧时间轴 ----
    gfx_draw_rectangle(gfx, 0, SP_AXIS_TOP_H, SP_PLOT_X0, gfx->height - SP_AXIS_TOP_H, 0, 0, 0, 1);
    for (int32_t k = 1; k <= 3; k++) {
        int32_t y = (int32_t)(gfx->height - 1 - k * px_per_sec + 0.5f);
        if (y < SP_AXIS_TOP_H + 8) break;
        wchar_t label[8];
        swprintf(label, 8, L"-%ds", (int)k);
        gfx_draw_line(gfx, SP_PLOT_X0 - 4, y, SP_PLOT_X0 - 1, y, 120, 120, 120, 1); // 刻度线
        gfx_font_draw_text(gfx, font_id, label, 1, y - 6, 150, 150, 150, 1);
    }
    gfx_font_draw_text(gfx, font_id, L"0s", 1, gfx->height - 13, 150, 150, 150, 1); // 底部=当前时刻
}

int32_t ui_spectrogram_init(Key_Event *key_event, Global_State *global_state) {
    // 分配工作区（PSRAM）
    if (sp_alloc_buffers() != 0) {
        printf("[Spectrogram] 工作区分配失败（内存不足）\n");
        sp_free_buffers();
        return -1;
    }

    // Hann窗（周期型）：w[n] = 0.5 * (1 - cos(2πn/N))
    for (int32_t i = 0; i < SP_FFT_SIZE; i++) {
        s_window[i] = 0.5f * (1.0f - cosf(2.0f * SP_PI * i / SP_FFT_SIZE));
    }
    // 旋转因子全表
    nano_fft_twiddle(s_twiddle_re, s_twiddle_im, SP_FFT_SIZE);
    // 热力调色板：黑→蓝→青→黄→红（参考 M5StickC-Plus FactoryTest 的声谱图配色）
    for (int32_t i = 0; i < 256; i++) {
        uint8_t r, g, b;
        if      (i < 64)  { r = 0;             g = 0;             b = (uint8_t)(i * 4); }
        else if (i < 128) { r = 0;             g = (i - 64) * 4;  b = 255; }
        else if (i < 192) { r = (i - 128) * 4; g = 255;           b = (uint8_t)(255 - (i - 128) * 4); }
        else              { r = 255;           g = (uint8_t)(255 - (i - 192) * 4); b = 0; }
        s_palette[i] = ((uint16_t)(r & 0xF8) << 8) | ((uint16_t)(g & 0xFC) << 3) | ((uint16_t)b >> 3);
    }

    s_inited = 1;

    // 每次进入复位为默认基准电平/动态范围
    s_level_base = SP_LEVEL_BASE_DEFAULT;
    s_level_range = SP_LEVEL_RANGE_DEFAULT;
    s_adjust_ts = 0;

    // 清屏，从全黑开始滚动
    gfx_soft_clear(global_state->gfx);
    gfx_refresh(global_state->gfx);

    return mic_init(SP_SAMPLE_RATE, (uint8_t)global_state->volume);
}

int32_t ui_spectrogram_render_frame(Key_Event *key_event, Global_State *global_state) {
    if (!s_inited) return -1;

    Nano_GFX *gfx = global_state->gfx;

    // 按键调节（下降沿）：←/→ 降低/升高基准电平，4/6 减小/增大动态范围
    if (key_event->key_edge < 0) {
        if (key_event->key_code == NANO_KEY_left) {
            s_level_base -= SP_LEVEL_BASE_STEP;
            if (s_level_base < 0.0f) s_level_base = 0.0f;
            s_adjust_ts = global_state->timestamp;
        }
        else if (key_event->key_code == NANO_KEY_right) {
            s_level_base += SP_LEVEL_BASE_STEP;
            if (s_level_base > SP_LEVEL_BASE_MAX) s_level_base = SP_LEVEL_BASE_MAX;
            s_adjust_ts = global_state->timestamp;
        }
        else if (key_event->key_code == NANO_KEY_4) {
            s_level_range -= SP_LEVEL_RANGE_STEP;
            if (s_level_range < SP_LEVEL_RANGE_MIN) s_level_range = SP_LEVEL_RANGE_MIN;
            s_adjust_ts = global_state->timestamp;
        }
        else if (key_event->key_code == NANO_KEY_6) {
            s_level_range += SP_LEVEL_RANGE_STEP;
            if (s_level_range > SP_LEVEL_RANGE_MAX) s_level_range = SP_LEVEL_RANGE_MAX;
            s_adjust_ts = global_state->timestamp;
        }
    }

    // 读取一帧音频（阻塞至数据就绪）
    int32_t n = mic_read(s_mic_buf, SP_FFT_SIZE);
    if (n < SP_FFT_SIZE) {
        return 0; // 本帧数据不足，跳过（下一主循环继续）
    }

    // 加 Hann 窗并填入虚部（实数输入）
    for (int32_t i = 0; i < SP_FFT_SIZE; i++) {
        s_fft_re[i] = (float)s_mic_buf[i] * s_window[i];
        s_fft_im[i] = 0.0f;
    }

    nano_fft_execute(s_fft_re, s_fft_im, s_twiddle_re, s_twiddle_im, SP_FFT_SIZE);

    // 帧缓冲整体上滚，底部腾出 SP_ROW_H 行
    gfx_scroll_up(gfx, SP_ROW_H);

    // 将前 SP_BINS 个频点（跳过直流分量，从 bin 1 开始）按对数幅度绘制为新行
    // 映射：v = (log10(mag+1)*80 - 基准电平) * 255 / 动态范围（←/→ 调基准，4/6 调范围）
    for (int32_t x = 0; x < SP_BINS; x++) {
        float re = s_fft_re[x + 1];
        float im = s_fft_im[x + 1];
        float mag = sqrtf(re * re + im * im);
        float v_raw = log10f(mag + 1.0f) * SP_LOG_SCALE;
        int32_t v = (int32_t)((v_raw - s_level_base) * 255.0f / s_level_range);
        if (v > 255) v = 255;
        if (v < 0)   v = 0;
        gfx_fill_rect_rgb565(gfx, SP_PLOT_X0 + x, gfx->height - SP_ROW_H, 1, SP_ROW_H, s_palette[v]);
    }

    // 绘制坐标轴（覆盖在滚动后的内容上，每帧重绘）
    sp_draw_axes(global_state);

    // 调节后限时叠加显示当前基准电平/动态范围（位于绘图区，随声谱图自然上滚消失）
    if (s_adjust_ts && global_state->timestamp - s_adjust_ts < SP_ADJUST_OVERLAY_MS) {
        wchar_t info[32];
        swprintf(info, 32, L"基准 %d  范围 %d", (int)s_level_base, (int)s_level_range);
        gfx_draw_rectangle(gfx, SP_PLOT_X0 + 4, SP_AXIS_TOP_H + 4, 140, 16, 0, 0, 0, 1);
        gfx_font_draw_text(gfx, GFX_FONT_BITMAP_12, info, SP_PLOT_X0 + 8, SP_AXIS_TOP_H + 6, 255, 255, 128, 1);
    }

    gfx_refresh(gfx);

    return 0;
}

int32_t ui_spectrogram_deinit(Key_Event *key_event, Global_State *global_state) {
    s_inited = 0;
    int32_t ret = mic_close();
    sp_free_buffers();
    return ret;
}
