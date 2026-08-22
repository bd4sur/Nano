#include <math.h>

#include "ui_pedometer.h"
#include "ui_color.h"
#include "hal_key.h"
#include "hal_imu.h"
#include "nano_fft.h"

// ===============================================================================
// 计步器实现（时域峰值计数为主，频域周期性校验）
// ===============================================================================

#define PED_FS              (100)   // 采样率 Hz
#define PED_SAMPLE_MS       (1000 / PED_FS)
#define PED_RING            (512)   // 滤波后信号环形缓冲（5.12s）
#define PED_STATS_WIN       (200)   // 阈值统计窗口（2s）
#define PED_FFT_N           (256)   // 频域校验FFT点数（2.56s）
#define PED_VALID_MS        (1000)  // 频域校验周期（1s）

// 时域候选点检测参数（宽条件：仅自适应阈值+最小间隔，假峰由频域交叉验证兜底）
#define PED_MIN_INTERVAL_MS (250)   // 候选点最小间隔（上限240步/分）
#define PED_STD_K           (0.3f)  // 时域自适应阈值 = 均值 + K × 标准差
#define PED_TH_MIN          (0.015f) // 时域阈值下限（g）

// 频域分析参数
#define PED_FREQ_BIN_MAX    (10)    // 频谱分析bin上限（0.39~3.9Hz）
#define PED_SFM_MAX         (0.3f)  // 频谱平坦度阈值：低于此值判定为节奏明确的周期运动
#define PED_RATE_TOL_REL    (0.3f)  // 候选速率与主频的相对容差

// 显示参数
#define PED_WAVE_N          (160)   // 波形显示样本数
#define PED_WAVE_Y0         (100)   // 波形区顶部y
#define PED_WAVE_H          (80)    // 波形区高度
#define PED_REDRAW_MS       (66)    // 重绘节流（约15fps）

// 工作区（PSRAM，进入时分配、退出时释放）
static float *s_ring = NULL;        // 滤波后信号环形缓冲
static float *s_fft_re = NULL, *s_fft_im = NULL;
static float *s_tw_re = NULL, *s_tw_im = NULL;
static float *s_hann = NULL;

static int32_t  s_ring_pos = 0;     // 环形缓冲写指针
static uint32_t s_sample_total = 0; // 累计样本数

// 滤波器状态
static float s_dc_x1 = 0.0f, s_dc_y1 = 0.0f; // DC-blocker（高通）
static float s_lp_y1 = 0.0f;                 // EMA低通

// 采样门控
static uint64_t s_next_sample_ts = 0;

// 峰值检测状态
static float    s_peak_thresh = PED_TH_MIN;
static uint32_t s_last_step_ms = 0;
static uint32_t s_pending_steps = 0;  // 待校验窗口内的步数增量
static uint32_t s_total_steps = 0;    // 已确认总步数
static uint32_t s_step_intervals[6];  // 最近步间隔（ms）
static int32_t  s_int_num = 0, s_int_pos = 0;
static int16_t  s_peak_marks[PED_WAVE_N / 8]; // 波形区峰值标记（相对样本序号）

// 频域校验状态
static uint64_t s_next_valid_ts = 0;
static float    s_dom_freq = 0.0f;     // 主频 Hz
static float    s_sfm = 1.0f;          // 频谱平坦度（越小频谱越尖锐、节奏越明确）
static int32_t  s_walk_confirmed = 0;  // 1=确认 0=未确认
static float    s_cadence_hz = 0.0f;   // 时域步频

static uint64_t s_next_draw_ts = 0;

// -------------------------------------------------------------------------------

// 采集一个样本并完成滤波、峰值检测
static void ped_process_sample(Global_State *global_state, uint32_t now_ms) {
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    if (imu_read_angle(&ax, &ay, &az) != 0) return; // imu_read_angle 实际返回三轴加速度(g)

    // 合成幅值
    float x = sqrtf(ax * ax + ay * ay + az * az);

    // DC-blocker 高通去重力/直流：y[n] = x[n] - x[n-1] + R*y[n-1], R=0.95
    float y = x - s_dc_x1 + 0.95f * s_dc_y1;
    s_dc_x1 = x;
    s_dc_y1 = y;

    // EMA 低通平滑（~0.2s）
    s_lp_y1 += 0.15f * (y - s_lp_y1);
    float sig = s_lp_y1;

    // 写入环形缓冲
    s_ring[s_ring_pos] = sig;
    s_ring_pos = (s_ring_pos + 1) % PED_RING;
    s_sample_total++;

    // 每10个样本更新一次阈值统计（滑动2s窗 均值+0.5×标准差）
    if (s_sample_total % 10 == 0) {
        int32_t n = (s_sample_total < PED_STATS_WIN) ? s_sample_total : PED_STATS_WIN;
        float mean = 0.0f, var = 0.0f;
        for (int32_t i = 0; i < n; i++) {
            float v = s_ring[(s_ring_pos - n + i + PED_RING) % PED_RING];
            mean += v;
        }
        mean /= n;
        for (int32_t i = 0; i < n; i++) {
            float v = s_ring[(s_ring_pos - n + i + PED_RING) % PED_RING];
            var += (v - mean) * (v - mean);
        }
        float std = sqrtf(var / n);
        s_peak_thresh = mean + PED_STD_K * std;
        if (s_peak_thresh < PED_TH_MIN) s_peak_thresh = PED_TH_MIN;
    }

    // 时域候选点：超过自适应阈值 且 满足最小间隔（仅此两条，宽条件）
    if (sig > s_peak_thresh && (now_ms - s_last_step_ms) >= PED_MIN_INTERVAL_MS) {
        // 记录步间隔（用于时域步频估计）
        if (s_int_num < 6) s_int_num++;
        s_step_intervals[s_int_pos] = now_ms - s_last_step_ms;
        s_int_pos = (s_int_pos + 1) % 6;
        s_last_step_ms = now_ms;
        s_pending_steps++;
        // 记录波形峰值标记（相对样本序号）
        for (int32_t i = PED_WAVE_N / 8 - 1; i > 0; i--) s_peak_marks[i] = s_peak_marks[i - 1];
        s_peak_marks[0] = 0;
    }

    // 时域步频：最近若干步间隔的均值
    if (s_int_num >= 2) {
        uint32_t sum = 0;
        for (int32_t i = 0; i < s_int_num; i++) sum += s_step_intervals[i];
        s_cadence_hz = 1000.0f / ((float)sum / s_int_num);
    }
}

// 频域交叉验证（每1s）：频谱平坦度判规律/抖动 + 候选速率与主频比对
static void ped_freq_validate(Global_State *global_state) {
    // 拷贝最近 PED_FFT_N 个样本（最旧→最新），去均值、加Hann窗
    float mean = 0.0f;
    for (int32_t i = 0; i < PED_FFT_N; i++) {
        mean += s_ring[(s_ring_pos - PED_FFT_N + i + PED_RING) % PED_RING];
    }
    mean /= PED_FFT_N;
    for (int32_t i = 0; i < PED_FFT_N; i++) {
        s_fft_re[i] = (s_ring[(s_ring_pos - PED_FFT_N + i + PED_RING) % PED_RING] - mean) * s_hann[i];
        s_fft_im[i] = 0.0f;
    }

    nano_fft_execute(s_fft_re, s_fft_im, s_tw_re, s_tw_im, PED_FFT_N);

    // 频谱平坦度（SFM = 几何均值/算术均值，0..1；越小频谱越尖锐）与主频
    float log_sum = 0.0f, lin_sum = 0.0f, peak = 0.0f;
    int32_t peak_bin = 0;
    for (int32_t b = 1; b <= PED_FREQ_BIN_MAX; b++) {
        float p = s_fft_re[b] * s_fft_re[b] + s_fft_im[b] * s_fft_im[b] + 1e-12f;
        log_sum += logf(p);
        lin_sum += p;
        if (p > peak) { peak = p; peak_bin = b; }
    }
    s_sfm = expf(log_sum / PED_FREQ_BIN_MAX) / (lin_sum / PED_FREQ_BIN_MAX);
    s_dom_freq = peak_bin * ((float)PED_FS / PED_FFT_N);

    // 候选速率（本窗口候选点数/窗口时长）
    float cand_rate = (float)s_pending_steps / (PED_VALID_MS / 1000.0f);

    // 交叉验证：频谱尖锐（节奏明确的周期运动）且候选速率与主频大致一致（相对容差）
    float tol = (PED_RATE_TOL_REL * cand_rate > 0.3f) ? (PED_RATE_TOL_REL * cand_rate) : 0.3f;
    if (s_sfm < PED_SFM_MAX && cand_rate > 0.0f && fabsf(cand_rate - s_dom_freq) <= tol) {
        s_total_steps += s_pending_steps; // 确认该窗口的候选点
        s_walk_confirmed = 1;
    }
    else {
        s_walk_confirmed = 0;             // 判定为无规律抖动，丢弃候选点
    }
    s_pending_steps = 0;
}

// 绘制界面
static void ped_draw(Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    uint8_t dark = (global_state->ui_color_style == UI_COLOR_DARK);
    uint8_t txt_R = dark ? 220 : 20, txt_G = dark ? 220 : 20, txt_B = dark ? 220 : 20;

    gfx_soft_clear(gfx);

    ui_draw_header(NULL, global_state, L"计步器", 1);

    // 总步数（大字）
    wchar_t buf[48];
    swprintf(buf, 48, L"%u", s_total_steps);
    uint32_t big_font = GFX_FONT_ALPHA_16;
    gfx_font_draw_text_centered(gfx, big_font, buf, gfx->width / 2, 44, 0, 128, 255, 1);
    gfx_font_draw_text_centered(gfx, global_state->ui_font, L"步", gfx->width / 2 + 60, 50, txt_R, txt_G, txt_B, 1);

    // 状态行：步频 / 主频 / 峰均比 / 状态
    wchar_t *status = s_walk_confirmed ? L"行走" :
        ((global_state->timestamp - s_last_step_ms > 2000) ? L"静止" : L"未确认");
    uint8_t st_R = s_walk_confirmed ? 0 : 255, st_G = s_walk_confirmed ? 200 : 160,
            st_B = s_walk_confirmed ? 64 : 0;
    if (!s_walk_confirmed && global_state->timestamp - s_last_step_ms > 2000) {
        st_R = 140; st_G = 140; st_B = 140;
    }
    swprintf(buf, 48, L"步频 %d步/分  主频 %.2fHz(SFM %.2f)", (int32_t)(s_cadence_hz * 60.0f), s_dom_freq, s_sfm);
    gfx_font_draw_text(gfx, global_state->ui_font, buf, 4, 70, txt_R, txt_G, txt_B, 1);
    gfx_font_draw_text(gfx, global_state->ui_font, status, gfx->width - 40, 70, st_R, st_G, st_B, 1);

    // ---- 波形区 ----
    int32_t wy0 = PED_WAVE_Y0, wy1 = PED_WAVE_Y0 + PED_WAVE_H, wym = (wy0 + wy1) / 2;
    gfx_draw_rectangle(gfx, 0, wy0, gfx->width, PED_WAVE_H, dark ? 10 : 250, dark ? 10 : 250, dark ? 12 : 250, 1);
    gfx_draw_line(gfx, 0, wym, gfx->width - 1, wym, 60, 60, 60, 1); // 零线

    // 阈值线
    int32_t ty = wym - (int32_t)(s_peak_thresh * (PED_WAVE_H / 2) / 0.3f);
    if (ty < wy0) ty = wy0;
    gfx_draw_line(gfx, 0, ty, gfx->width - 1, ty, 80, 40, 40, 1);

    // 波形（最近 PED_WAVE_N 个样本，每样本2px；幅值按 0.3g→半高 缩放）
    float scale = (PED_WAVE_H / 2) / 0.3f;
    for (int32_t i = 0; i < PED_WAVE_N - 1; i++) {
        int32_t idx0 = (s_ring_pos - PED_WAVE_N + i + PED_RING) % PED_RING;
        int32_t idx1 = (s_ring_pos - PED_WAVE_N + i + 1 + PED_RING) % PED_RING;
        int32_t ya = wym - (int32_t)(s_ring[idx0] * scale);
        int32_t yb = wym - (int32_t)(s_ring[idx1] * scale);
        if (ya < wy0) ya = wy0; if (ya > wy1) ya = wy1;
        if (yb < wy0) yb = wy0; if (yb > wy1) yb = wy1;
        gfx_draw_line(gfx, i * 2, ya, i * 2 + 2, yb, 0, 200, 255, 1);
    }
    // 峰值标记
    for (int32_t i = 0; i < PED_WAVE_N / 8; i++) {
        if (s_peak_marks[i] < 0) break;
        int32_t rel = s_peak_marks[i];
        if (rel >= PED_WAVE_N) continue;
        int32_t px = (PED_WAVE_N - 1 - rel) * 2;
        gfx_draw_line(gfx, px, wy0 + 4, px, wy0 + 12, 255, 64, 0, 1);
    }
    for (int32_t i = 0; i < PED_WAVE_N / 8; i++) s_peak_marks[i]++;

    // 页脚
    ui_draw_footer(NULL, global_state, L"A-返回  D-清零", 1);

    gfx_refresh(gfx);
}

// -------------------------------------------------------------------------------

int32_t ui_pedometer_init(Key_Event *key_event, Global_State *global_state) {
    if (!s_ring)     s_ring     = (float *)platform_calloc(PED_RING, sizeof(float));
    if (!s_fft_re)   s_fft_re   = (float *)platform_calloc(PED_FFT_N, sizeof(float));
    if (!s_fft_im)   s_fft_im   = (float *)platform_calloc(PED_FFT_N, sizeof(float));
    if (!s_tw_re)    s_tw_re    = (float *)platform_calloc(PED_FFT_N / 2, sizeof(float));
    if (!s_tw_im)    s_tw_im    = (float *)platform_calloc(PED_FFT_N / 2, sizeof(float));
    if (!s_hann)     s_hann     = (float *)platform_calloc(PED_FFT_N, sizeof(float));
    if (!s_ring || !s_fft_re || !s_fft_im || !s_tw_re || !s_tw_im || !s_hann) {
        printf("[Pedometer] 工作区分配失败（内存不足）\n");
        return -1;
    }

    nano_fft_twiddle(s_tw_re, s_tw_im, PED_FFT_N);
    for (int32_t i = 0; i < PED_FFT_N; i++) {
        s_hann[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * i / PED_FFT_N));
    }

    // 复位全部状态
    s_ring_pos = 0; s_sample_total = 0;
    s_dc_x1 = s_dc_y1 = 0.0f; s_lp_y1 = 0.0f;
    s_peak_thresh = PED_TH_MIN;
    s_last_step_ms = 0; s_pending_steps = 0; s_total_steps = 0;
    s_int_num = 0; s_int_pos = 0;
    s_dom_freq = 0.0f; s_sfm = 1.0f; s_walk_confirmed = 0; s_cadence_hz = 0.0f;
    for (int32_t i = 0; i < PED_WAVE_N / 8; i++) s_peak_marks[i] = -1;
    s_next_sample_ts = global_state->timestamp;
    s_next_valid_ts = global_state->timestamp + PED_VALID_MS;
    s_next_draw_ts = 0;

    return 0;
}

int32_t ui_pedometer_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 按100Hz门控采样（一帧可能采0~3个样本）
    while (global_state->timestamp >= s_next_sample_ts) {
        ped_process_sample(global_state, (uint32_t)s_next_sample_ts);
        s_next_sample_ts += PED_SAMPLE_MS;
    }

    // 每1s频域校验
    if (global_state->timestamp >= s_next_valid_ts && s_sample_total >= PED_FFT_N) {
        ped_freq_validate(global_state);
        s_next_valid_ts += PED_VALID_MS;
    }

    // D键清零
    if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
        s_total_steps = 0; s_pending_steps = 0;
    }

    // 重绘节流
    if (global_state->timestamp >= s_next_draw_ts) {
        s_next_draw_ts = global_state->timestamp + PED_REDRAW_MS;
        ped_draw(global_state);
    }

    return 0;
}

int32_t ui_pedometer_deinit(Key_Event *key_event, Global_State *global_state) {
    if (s_ring)   { free(s_ring);   s_ring = NULL; }
    if (s_fft_re) { free(s_fft_re); s_fft_re = NULL; }
    if (s_fft_im) { free(s_fft_im); s_fft_im = NULL; }
    if (s_tw_re)  { free(s_tw_re);  s_tw_re = NULL; }
    if (s_tw_im)  { free(s_tw_im);  s_tw_im = NULL; }
    if (s_hann)   { free(s_hann);   s_hann = NULL; }
    return 0;
}
