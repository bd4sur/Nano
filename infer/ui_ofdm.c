// ============================================================================
// 寻呼机（OFDM 声波数传）UI 模块 / BD4SUR 2026-07
// 发射：w_input_main 输入文本 → D 提交 → ofdm_modem 调制 → 扬声器单通道队列无缝循环播放
// 接收：硅麦 48kHz 采集 → ofdm_modem 流式解调 → w_textarea_main 滚动显示
// 同一时刻仅工作于一种模式；进入/退出状态时严格成对申请/释放资源。
// ============================================================================

#include "ui_ofdm.h"

#include <string.h>
#include <stdio.h>
#include <math.h>

#include "ofdm_modem.h"
#include "hal_audio_out.h"
#include "hal_audio_in.h"
#include "hal_key.h"
#include "ui_color.h"

// 发射音量：跟随全局主音量（global_state->volume，系统设置中调节）。
// 注意：NS4168 功放带防破音（NCN/AGC）功能，过高音量会触发增益压缩，
// 对高 PAPR 的 OFDM 信号造成幅度失真；实测建议保持中低音量（≤64）。
#define OFDM_RX_CHUNK       (960)  // 每次采集/喂入的采样数（20ms @48kHz）
#define OFDM_RX_BYTES_CAP   (12288)// 接收文本字节累积上限（≈4096个中文字）
#define OFDM_TXT_REFRESH_MS (200)  // 接收文本限频刷新间隔
#define OFDM_STA_REFRESH_MS (200)  // 状态栏限频刷新间隔

// ---------------- ofdm_modem 内存注入（PSRAM） ----------------
static void *ofdm_alloc_psram(size_t size) { return platform_calloc(1, size); }
static void ofdm_free_psram(void *ptr) { free(ptr); }

// ---------------- 发射侧静态状态 ----------------
static OFDM_TX *s_tx = NULL;
static int16_t *s_pcm[AUDIO_OUT_QUEUE_DEPTH] = {NULL, NULL}; // 乒乓双缓冲（playRaw 引用，须常驻至播完）
static int32_t s_fill_buf = 0;     // 下一个待填充/投放的缓冲序号（0/1 轮换）
static float *s_frame_f32 = NULL;  // 帧渲染缓冲
static int32_t s_audio_on = 0;     // audio_out_init 是否已调用
static uint32_t s_tx_frames_played = 0;
static uint64_t s_tx_last_sta_ts = 0;
static wchar_t s_tx_info[512];     // 发射信息文本（静态，textarea_set 会复制）

// ---------------- 接收侧静态状态 ----------------
static OFDM_RX *s_rx = NULL;
static uint8_t *s_rx_bytes = NULL;     // 净荷字节累积（PSRAM）
static uint32_t s_rx_bytes_len = 0;
static wchar_t *s_rx_wcbuf = NULL;     // 转换暂存（PSRAM，RX_BYTES_CAP+1 宽字符）
static int32_t s_rx_new_text = 0;
static int32_t s_mic_ok = 0;
static uint64_t s_rx_last_txt_ts = 0;
static uint64_t s_rx_last_sta_ts = 0;
static int16_t *s_mic_i16 = NULL;        // 采集缓冲（PSRAM，OFDM_RX_CHUNK）
static float *s_mic_f32 = NULL;          // 转换缓冲（PSRAM，OFDM_RX_CHUNK）
static wchar_t s_rx_status[128];       // 状态栏文本
static int32_t s_rx_follow = 1;        // 自动滚动到底部跟随模式（左右键手动滚动时暂停）
static volatile int32_t s_mic_err_cnt = 0; // mic_read 错误计数（采集任务写，渲染任务读）
static float s_mic_rms_ema = 0.0f;     // 接收电平 RMS（int16 域，指数滑动平均）
static float s_mic_mean_ema = 0.0f;    // 直流分量（int16 域均值，指数滑动平均）
static int32_t s_mic_peak = 0;         // 最近块峰值（int16 域）
static Global_State *s_rx_gs = NULL;   // 供回调取 UI 色彩风格

// ---------------- RX 采集解耦（修复空口接收丢采样） ----------------
// 根因（2026-08-01 定位）：此前 RX 在渲染任务里 mic_read 与 UI 绘制/串口诊断串行，
// I2S DMA 缓冲仅 4096 采样（≈85ms@48k）。每次 SC 同步事件触发的 printf 群（≈45ms）
// + 文本重绘（30~100ms）+ lagscan 诊断计算（30~80ms）叠加即超 85ms → DMA 溢出丢样；
// 丢样恰落在同步事件之后（训练A/B之间），帧帧死于槽2（训练B相关≈0.01、相位残差≈1.7、
// SFO 乱值），宿主机 slip 仿真精确复现该签名（ofdm/test_slip.c）。
// 修复：专用高优先级采集任务（Core1，prio 3）持续把 I2S 采样搬进 PSRAM 大环形缓冲
//（65536 采样 ≈ 1.37s），渲染任务每次迭代从环中取数喂 modem；UI 卡顿由环吸收，
// 硬件层不再丢样。SPSC 无锁环：仅生产者写 s_ring_w、仅消费者写 s_ring_r。
#define OFDM_RX_RING_CAP    (65536)  // int16 采样，2 的幂
static int16_t *s_rx_ring = NULL;        // 采集环形缓冲（PSRAM）
static volatile uint32_t s_ring_w = 0;   // 单调写序号（仅采集任务写）
static volatile uint32_t s_ring_r = 0;   // 单调读序号（仅渲染任务写）
static volatile uint32_t s_ring_ov = 0;  // 环溢出丢样计数（诊断用，正常应恒 0）
static int16_t *s_mic_task_buf = NULL;   // 采集任务暂存（PSRAM，OFDM_RX_CHUNK）
static platform_task_handle_t s_mic_task = NULL;
static volatile int32_t s_mic_task_stop = 0;

static void ui_ofdm_mic_task(void *arg) {
    (void)arg;
    while (!s_mic_task_stop) {
        int32_t n = mic_read(s_mic_task_buf, OFDM_RX_CHUNK); // 阻塞至数据就绪（≤100ms）
        if (n <= 0) { s_mic_err_cnt++; continue; }
        uint32_t w = s_ring_w, r = s_ring_r;
        if (OFDM_RX_RING_CAP - (w - r) < (uint32_t)n) {
            s_ring_ov += (uint32_t)n; // 环满（UI 停滞 >1.3s）：弃新块保旧数据，计数告警
            continue;
        }
        for (int32_t i = 0; i < n; i++) s_rx_ring[(w + i) & (OFDM_RX_RING_CAP - 1)] = s_mic_task_buf[i];
        platform_memory_barrier(); // 数据先于写序号发布（跨核 SPSC）
        s_ring_w = w + (uint32_t)n;
    }
    s_mic_task = NULL;
    platform_task_delete_self();
}

// ---------------- RX 串口诊断（排障用，printf → UART0，串口监视器 115200） ----------------
#define OFDM_DBG_HIST_LEN (65536)
static float *s_dbg_hist = NULL;       // 最近接收采样历史环形缓冲（PSRAM）
static uint32_t s_dbg_hist_pos = 0;    // 下一写入位置
static uint64_t s_rx_last_dbg_ts = 0;  // 上一次串口诊断输出时间戳
static uint32_t s_rx_feed_total = 0;   // 累计喂入采样数
static uint32_t s_rx_dbg_tick = 0;     // 诊断输出节拍计数

// 滞后扫描：测量接收信号中 SC 前导 [A,A] 结构的实际滞后（标称 720@48kHz），
// 可直接反映麦克风真实采样率与标称值的偏差（如实际 44.1kHz 时峰值滞后≈661），
// 同时验证信号中是否真的存在该周期结构（无峰=内容不符/失真）。
static void ui_ofdm_rx_lag_scan(void) {
    if (!s_dbg_hist || s_rx_feed_total < OFDM_DBG_HIST_LEN) return;
    float best_m[3] = {0, 0, 0};
    int32_t best_l[3] = {0, 0, 0};
    for (int32_t lag = 600; lag <= 840; lag += 4) {
        // 仅扫描最新 8192 采样（控制计算量），覆盖一个完整前导周期
        for (int32_t off = 0; off + 2 * lag <= 8192; off += 64) {
            uint32_t base = (s_dbg_hist_pos + OFDM_DBG_HIST_LEN - 2 * lag - off) % OFDM_DBG_HIST_LEN;
            float P = 0.0f, E1 = 0.0f, R = 0.0f;
            for (int32_t i = 0; i < lag; i++) {
                float x1 = s_dbg_hist[(base + i) % OFDM_DBG_HIST_LEN];
                float x2 = s_dbg_hist[(base + lag + i) % OFDM_DBG_HIST_LEN];
                P += x1 * x2; E1 += x1 * x1; R += x2 * x2;
            }
            float m = (P * P) / (E1 * R + 1e-12f);
            if (m > best_m[0]) {
                best_m[2] = best_m[1]; best_l[2] = best_l[1];
                best_m[1] = best_m[0]; best_l[1] = best_l[0];
                best_m[0] = m; best_l[0] = lag;
            }
            else if (m > best_m[1]) {
                best_m[2] = best_m[1]; best_l[2] = best_l[1];
                best_m[1] = m; best_l[1] = lag;
            }
            else if (m > best_m[2]) {
                best_m[2] = m; best_l[2] = lag;
            }
        }
    }
    printf("ofdm-rx lagscan: lag1=%d(m=%.3f) lag2=%d(m=%.3f) lag3=%d(m=%.3f) [标称720@48k, 661≈44.1k]\n",
           (int)best_l[0], (double)best_m[0], (int)best_l[1], (double)best_m[1],
           (int)best_l[2], (double)best_m[2]);
}

// 频谱快照：对最近 2048 个采样（42.7ms）做 Goertzel，打印 0~5.5kHz 共 24 个频点的幅度(dB)。
// 用于区分：低频轰鸣、OFDM 信号带（2500~5500Hz）是否覆盖平坦、载波位置偏移、带外 EMI 音调。
static void ui_ofdm_rx_spectrum_scan(void) {
    static const float FREQS[] = {50, 100, 150, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750,
                                  2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4500, 5000, 5500};
#define SPEC_N (2048)
    if (!s_dbg_hist || s_rx_feed_total < SPEC_N) return;
    char line[512];
    int p = 0;
    p += snprintf(line + p, (int)sizeof(line) - p, "ofdm-rx spectrum(dB):");
    for (uint32_t fi = 0; fi < sizeof(FREQS) / sizeof(FREQS[0]); fi++) {
        double w = 6.283185307179586 * FREQS[fi] / 48000.0;
        double cw = 2.0 * cos(w);
        double s0 = 0, s1 = 0, s2 = 0;
        for (int32_t i = 0; i < SPEC_N; i++) {
            double x = s_dbg_hist[(s_dbg_hist_pos + OFDM_DBG_HIST_LEN - SPEC_N + i) % OFDM_DBG_HIST_LEN];
            s0 = x + cw * s1 - s2;
            s2 = s1; s1 = s0;
        }
        double power = s1 * s1 + s2 * s2 - cw * s1 * s2;
        double mag = sqrt(power) / (SPEC_N / 2);
        double db = 20.0 * log10(mag + 1e-9);
        p += snprintf(line + p, (int)sizeof(line) - p, " %.0f:%.0f", (double)FREQS[fi], db);
        if (p >= (int)sizeof(line) - 16) break;
    }
    printf("%s\n", line);
}

// ============================================================================
// 模式菜单
// ============================================================================

void ui_ofdm_menu_init(Key_Event *key_event, Global_State *global_state) {
    // 条目字符串借用字面量的静态存储，控件不复制
    static const wchar_t *ofdm_menu_items[] = {
        L"发射（文本 → 声波）",
        L"接收（声波 → 文本）",
        L"软件环路自测（自发自收）",
    };
    // 进入寻呼机模块：初始化 modem 码表到 PSRAM（退出时由 ui_ofdm_menu_on_exit 释放）
    if (ofdm_tables_init(ofdm_alloc_psram) != 0) {
        printf("ofdm: tables init FAILED (oom)\n");
    }
    global_state->w_menu_main->title = L"寻呼机（OFDM声波数传）";
    global_state->w_menu_main->items = ofdm_menu_items;
    global_state->w_menu_main->item_num = 3;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
}

void ui_ofdm_menu_on_exit(void) {
    ofdm_tables_free(ofdm_free_psram);
}

// 发射/环路自测输入编辑器的默认文本（火腿呼叫 + 测试文）
static const wchar_t OFDM_DEFAULT_TX_TEXT[] =
    L"CQ CQ DE BD4SUR 声学OFDM数传测试\n\n"
    L"本系统利用声波传输文本数据。\n"
    L"发射机对消息作RS编码、交织和加扰，调制为OFDM符号，上变频到4kHz发射。\n"
    L"接收机将音频信号解调为OFDM符号，经过解扰、解交织和RS解码，得到字节序列。\n"
    L"系统能够适应比较复杂的声学信道。\n\n";

// 以默认文本填充输入控件（init 已清空；此后用户可编辑，提交返回后保留修改）
static void ui_ofdm_input_fill_default(Key_Event *ke, Global_State *gs) {
    Widget_Textarea_State *ta = &gs->w_input_main->textarea;
    wcsncpy(ta->text, OFDM_DEFAULT_TX_TEXT, UI_STR_BUF_MAX_LENGTH - 1);
    ta->text[UI_STR_BUF_MAX_LENGTH - 1] = L'\0';
    ta->length = wcslen(ta->text);
    ta->is_modified = 1;
    ui_widget_input_refresh(ke, gs, gs->w_input_main);
}

int32_t ui_ofdm_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    switch (ms->current_item_index) {
        case 0:
            // 进入发射文本输入（参照鹦鹉笼：选模型时初始化输入控件，提交后返回可继续编辑）
            ui_widget_input_init(ke, gs, gs->w_input_main, (wchar_t *)L"寻呼机·发射");
            ui_ofdm_input_fill_default(ke, gs);
            return STATE_OFDM_TX;
        case 1:
            return STATE_OFDM_RX;
        case 2:
            ui_widget_input_init(ke, gs, gs->w_input_main, (wchar_t *)L"寻呼机·环路自测");
            ui_ofdm_input_fill_default(ke, gs);
            return STATE_OFDM_LOOP;
        default:
            return STATE_OFDM_MENU;
    }
}

// ============================================================================
// STATE_OFDM_TX（文本输入）
// ============================================================================

void ui_ofdm_tx_on_enter(Key_Event *key_event, Global_State *global_state) {
    // 从菜单进入时文本已由 ui_widget_input_init 清空；从 TXING 返回时保留原文
    ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
}

int32_t ui_ofdm_tx_event(Key_Event *key_event, Global_State *global_state) {
    int32_t next = ui_widget_input_event_handler(
        key_event, global_state, global_state->w_input_main,
        STATE_OFDM_MENU, STATE_OFDM_TX, STATE_OFDM_TXING);
    // 空文本不进入发射态（停留在输入态）
    if (next == STATE_OFDM_TXING && wcslen(global_state->w_input_main->textarea.text) == 0) {
        return STATE_OFDM_TX;
    }
    return next;
}

// ============================================================================
// STATE_OFDM_TXING（发射中）
// ============================================================================

// 渲染一帧并投入扬声器队列（帧尽自动回绕，实现循环播放）。
// 调用前提：audio_out_queue_free() 为真（队列有空槽，对应缓冲已播完可覆写）。
static void ui_ofdm_tx_fill_and_play(void) {
    if (ofdm_tx_render_frame(s_tx, s_frame_f32) != 0) {
        ofdm_tx_rewind(s_tx);
        if (ofdm_tx_render_frame(s_tx, s_frame_f32) != 0) return;
    }
    int16_t *pcm = s_pcm[s_fill_buf];
    for (int32_t i = 0; i < OFDM_FRAME_LENGTH; i++) {
        float v = s_frame_f32[i] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        else if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }
    if (audio_out_enqueue(pcm, OFDM_FRAME_LENGTH) == 0) {
        s_tx_frames_played++;
        s_fill_buf ^= 1; // 乒乓轮换
        printf("ofdm-tx: frame %lu enqueued\n", (unsigned long)s_tx_frames_played);
    }
    else {
        printf("ofdm-tx: enqueue failed\n");
    }
}

static void ui_ofdm_txing_cleanup(void) {
    if (s_audio_on) {
        audio_out_stop();
        audio_out_close();
        s_audio_on = 0;
    }
    if (s_tx) { ofdm_tx_destroy(s_tx); s_tx = NULL; }
    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++) {
        if (s_pcm[i]) { free(s_pcm[i]); s_pcm[i] = NULL; }
    }
    if (s_frame_f32) { free(s_frame_f32); s_frame_f32 = NULL; }
}

void ui_ofdm_txing_on_enter(Key_Event *key_event, Global_State *global_state) {
    // 文本（wchar_t）→ UTF-8 字节流
    wchar_t *text = global_state->w_input_main->textarea.text;
    uint32_t wlen = wcslen(text);
    uint32_t cap = wlen * 4 + 1;
    char *utf8 = (char *)platform_calloc(cap, 1);
    uint32_t blen = _wcstombs(utf8, text, cap);

    s_tx = ofdm_tx_create(ofdm_alloc_psram, ofdm_free_psram, (const uint8_t *)utf8, blen);
    free(utf8);

    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++)
        s_pcm[i] = (int16_t *)platform_calloc(OFDM_FRAME_LENGTH, sizeof(int16_t));
    s_frame_f32 = (float *)platform_calloc(OFDM_FRAME_LENGTH, sizeof(float));

    if (!s_tx || !s_pcm[0] || !s_pcm[1] || !s_frame_f32) {
        // 内存不足：清理并提示，由 event 函数兜底返回输入态
        ui_ofdm_txing_cleanup();
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main,
                               (wchar_t *)L"内存不足，无法启动发射", 0, 0);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        return;
    }

    // 钳位到64，防止幅度过大引发失真
    audio_out_init(OFDM_SAMPLE_RATE, (uint8_t)((global_state->volume > 64) ? 64 : global_state->volume));
    s_audio_on = 1;
    s_tx_frames_played = 0;
    s_tx_last_sta_ts = 0;
    s_fill_buf = 0;

    // 预填队列全部空槽（一块在播、一块待播），立即开始无缝循环播放
    ui_ofdm_tx_fill_and_play();
    ui_ofdm_tx_fill_and_play();

    // 绘制状态页（先画页眉页脚，textarea_draw 内部再全量刷帧）
    uint32_t n_frames = ofdm_tx_frame_count(s_tx);
    ui_draw_header(key_event, global_state, (wchar_t *)L"寻呼机·发射中", 1);
    ui_draw_footer(key_event, global_state, (wchar_t *)L"D/A：停止发射", 1);
    // 状态页参数全部取自 modem 宏（改参数时此处自动跟随，无需手改字符串）；
    // 颜色闭合标签按当前色彩风格选择（深色主题→白字、浅色主题→黑字），
    // 标签实参对应格式串中的第一个 %ls（其余文本必须全部留在格式串内才会被展开）
    // 颜色闭合标签按当前色彩风格选择（深色主题→白字、浅色主题→黑字）。
    // 必须用本函数入参 global_state：s_rx_gs 是接收态回调专用指针，仅在进入过
    // 接收态后才赋值；若开机后直接进入发射态，s_rx_gs 为 NULL，dark 恒为 0，
    // 闭合标签恒为 [#000000]——深色主题下黑字压黑底不可见（本 bug 的根因）。
    int32_t dark = (global_state->ui_color_style == UI_COLOR_DARK);
    swprintf(s_tx_info, sizeof(s_tx_info) / sizeof(wchar_t),
             L"[#00aa00]正在发射…%ls\n"
             L"净荷 %d 字节 → %d 个物理帧\n"
             L"物理层：子载波%d/基波%.3fHz/中心%dHz/带宽%.0fHz\n"
             L"帧结构：%d槽/%dms (SC前导+%d训练+%d数据)\n"
             L"信道编码：RS(%d,%d)×%d + 交织 + 加扰\n",
             (dark ? L"[#ffffff]" : L"[#000000]"),
             (int)blen, (int)n_frames,
             OFDM_CARRIER_NUMBER, (double)OFDM_BASE_FREQ, OFDM_CARRIER_FREQ,
             (double)(OFDM_CARRIER_NUMBER * OFDM_BASE_FREQ),
             OFDM_FRAME_SLOTS, (int32_t)((double)OFDM_FRAME_LENGTH / OFDM_SAMPLE_RATE * 1000),
             OFDM_FRAME_SLOTS - 1 - OFDM_FRAME_DATA_SYMBOLS, OFDM_FRAME_DATA_SYMBOLS,
             OFDM_PKT_WIRE_LEN / OFDM_PKT_RS_BLOCKS, OFDM_PKT_UNCODED_LEN / OFDM_PKT_RS_BLOCKS,
             OFDM_PKT_RS_BLOCKS);
    ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, s_tx_info, 0, 0);
    ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
}

int32_t ui_ofdm_txing_event(Key_Event *key_event, Global_State *global_state) {
    // 手动停止发射：短按 D 或 A
    if (key_event->key_edge < 0 &&
        (key_event->key_code == NANO_KEY_enter || key_event->key_code == NANO_KEY_esc)) {
        ui_ofdm_txing_cleanup();
        return STATE_OFDM_TX;
    }

    if (!s_tx) return STATE_OFDM_TX; // 异常兜底（如 on_enter 内存不足）

    // 背压安全的喂入：队列有空槽才填充（单通道双槽 ≈ 1.3s 音频余量，段尾无缝切换；
    // UI 刷帧不会造成饥饿；即使偶发间隙，接收端按帧独立重同步，协议天然容忍）
    if (audio_out_queue_free()) ui_ofdm_tx_fill_and_play();

    // 限频更新状态栏
    if (global_state->timestamp - s_tx_last_sta_ts >= OFDM_STA_REFRESH_MS) {
        s_tx_last_sta_ts = global_state->timestamp;
        swprintf(s_rx_status, sizeof(s_rx_status) / sizeof(wchar_t),
                 L"已发射 %d 帧（循环中） | D/A：停止", (int)s_tx_frames_played);
        ui_draw_footer(key_event, global_state, s_rx_status, 1);
        gfx_refresh(global_state->gfx);
    }
    return STATE_OFDM_TXING;
}

// ============================================================================
// STATE_OFDM_RX（接收中）
// ============================================================================

// 字节追加到接收显示缓冲（溢出丢弃最旧一半）
static void ui_ofdm_rx_append_bytes(const uint8_t *bytes, uint32_t len) {
    if (!s_rx_bytes || len == 0) return;
    if (s_rx_bytes_len + len > OFDM_RX_BYTES_CAP) {
        // 丢弃最旧一半（可能在多字节字符中间截断，显示为'?'，后续包到达后自愈）
        uint32_t keep = OFDM_RX_BYTES_CAP / 2;
        memmove(s_rx_bytes, s_rx_bytes + s_rx_bytes_len - keep, keep);
        s_rx_bytes_len = keep;
    }
    if (len > OFDM_RX_BYTES_CAP - s_rx_bytes_len) len = OFDM_RX_BYTES_CAP - s_rx_bytes_len;
    memcpy(s_rx_bytes + s_rx_bytes_len, bytes, len);
    s_rx_bytes_len += len;
    s_rx_new_text = 1;
}

// ---------------- Unicode 文本输出缓冲区 ----------------
// 传输文本是连续 UTF-8 字节流，104 字节 packet 界会强行截断多字节字符。
// 架构：每个 packet 到达后，pend 残字节 + 本包拼接，逐字符“尽力而为”解码——
// 只有经**完整 Unicode 校验**（continuation/overlong/代理区/超范围）的字符才进入
// 输出缓冲；数据损坏处插入 '?' 占位并跳过 1 字节重同步（绝不静默中止）；
// 本包尾部不完整字符（≤3 字节）**推迟**到下一包再解码，不输出任何暂态内容。
// 因此输出缓冲中只存在：正确解码的字符 + 数据损坏占位 '?'。
static uint8_t s_utf8_pend[4];       // 未完成的多字节序列暂存（推迟解码）
static uint32_t s_utf8_pend_len = 0;
static char s_utf8_line[144];        // Unicode 文本输出缓冲（行缓冲，满/换行即输出）
static uint32_t s_utf8_line_len = 0;

static void ui_ofdm_serial_flush(void) {
    if (s_utf8_line_len == 0) return;
    fwrite("ofdm-rx text: ", 1, 14, stdout);
    fwrite(s_utf8_line, 1, s_utf8_line_len, stdout);
    if (s_utf8_line[s_utf8_line_len - 1] != '\n') fwrite("\n", 1, 1, stdout);
    s_utf8_line_len = 0;
}
static void ui_ofdm_serial_put(const uint8_t *seq, uint32_t n) {
    if (s_utf8_line_len + n > sizeof(s_utf8_line) - 1) ui_ofdm_serial_flush();
    memcpy(s_utf8_line + s_utf8_line_len, seq, n);
    s_utf8_line_len += n;
    if (seq[n - 1] == '\n') ui_ofdm_serial_flush();
}

// Unicode 解码字符统一出口：双写——屏幕接收文本缓冲（s_rx_bytes，300ms 限频重绘）
// 与串口行缓冲。经此出口的字符均已通过完整 Unicode 校验（或为数据损坏占位 '?'），
// 屏幕与串口看到的解码内容完全一致。
static void ui_ofdm_uni_put(const uint8_t *seq, uint32_t n) {
    ui_ofdm_rx_append_bytes(seq, n); // 内部已置 s_rx_new_text
    ui_ofdm_serial_put(seq, n);
}

// 完整 UTF-8/Unicode 校验（1~4 字节）：continuation、overlong、代理区、超范围
static int32_t ui_ofdm_utf8_valid(const uint8_t *s, uint32_t n) {
    uint32_t cp;
    switch (n) {
        case 1: return (s[0] < 0x80);
        case 2:
            if ((s[1] & 0xC0) != 0x80) return 0;
            cp = ((uint32_t)(s[0] & 0x1F) << 6) | (s[1] & 0x3F);
            return (cp >= 0x80);
        case 3:
            if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return 0;
            cp = ((uint32_t)(s[0] & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
            return (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF));
        case 4:
            if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) return 0;
            cp = ((uint32_t)(s[0] & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
                 ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
            return (cp >= 0x10000 && cp <= 0x10FFFF);
    }
    return 0;
}

static void ui_ofdm_utf8_stream_out(const uint8_t *bytes, uint32_t len) {
    uint8_t buf[4 + OFDM_PKT_PAYLOAD_MAX];
    memcpy(buf, s_utf8_pend, s_utf8_pend_len);
    memcpy(buf + s_utf8_pend_len, bytes, len);
    uint32_t total = s_utf8_pend_len + len, p = 0;
    while (p < total) {
        uint8_t b = buf[p];
        uint32_t need;
        if ((b & 0x80) == 0) need = 1;
        else if ((b & 0xE0) == 0xC0) need = 2;
        else if ((b & 0xF0) == 0xE0) need = 3;
        else if ((b & 0xF8) == 0xF0) need = 4;
        else need = 0; // 非法前导字节
        if (need == 0) { ui_ofdm_uni_put((const uint8_t *)"?", 1); p++; continue; }
        if (p + need > total) break; // 尾部不完整：推迟到下一包解码，不输出
        if (!ui_ofdm_utf8_valid(buf + p, need)) { // 数据损坏：'?' 占位并跳 1 字节重同步
            ui_ofdm_uni_put((const uint8_t *)"?", 1); p++; continue;
        }
        ui_ofdm_uni_put(buf + p, need);
        p += need;
    }
    s_utf8_pend_len = total - p; // ≤3 字节
    memcpy(s_utf8_pend, buf + p, s_utf8_pend_len);
}

// 解出 packet 净荷回调：UTF-8 拼接解码 → 双写屏幕缓冲与串口
static void ui_ofdm_rx_on_text(const uint8_t *bytes, uint32_t len, void *user) {
    (void)user;
    ui_ofdm_utf8_stream_out(bytes, len);
}

// 接收机事件日志回调：以灰色标签行注入文本流（稀疏事件：锁定/CFO/帧解出/坏帧/失锁）
// 颜色标签会被文本控件解析（与 LLM 对话同色机制）； UTF-8 中文直接透传。
static void ui_ofdm_rx_on_log(const char *msg, void *user) {
    (void)user;
    if (!msg) return;
    printf("ofdm-rx log: %s\n", msg); // 串口同步输出（排障用）
    // int32_t dark = (s_rx_gs && s_rx_gs->ui_color_style == UI_COLOR_DARK);
    // ui_ofdm_rx_append_bytes((const uint8_t *)"\n[#888888]\xC2\xB7", 12); // "\n[#888888]·"
    // ui_ofdm_rx_append_bytes((const uint8_t *)msg, strlen(msg));
    // ui_ofdm_rx_append_bytes((const uint8_t *)(dark ? "[#ffffff]\n" : "[#000000]\n"), 10);
}

static void ui_ofdm_rx_cleanup(void) {
    // 退出：尾存不完整序列属暂态内容，不进入输出缓冲（丢弃），仅冲刷行缓冲
    s_utf8_pend_len = 0;
    ui_ofdm_serial_flush();
    // 先停采集任务（其内部持有 s_mic_task_buf/s_rx_ring 并调用 mic_read），再关 I2S
    if (s_mic_task) {
        s_mic_task_stop = 1;
        for (int32_t i = 0; i < 50 && s_mic_task; i++) platform_task_delay_ms(10); // 等其自删（≤500ms）
        if (s_mic_task) { platform_task_delete(s_mic_task); s_mic_task = NULL; } // 超时强删（兜底）
    }
    mic_close(); // 释放 I2S 并恢复扬声器（幂等）
    if (s_rx) { ofdm_rx_destroy(s_rx); s_rx = NULL; }
    if (s_rx_bytes) { free(s_rx_bytes); s_rx_bytes = NULL; }
    if (s_rx_wcbuf) { free(s_rx_wcbuf); s_rx_wcbuf = NULL; }
    if (s_mic_i16) { free(s_mic_i16); s_mic_i16 = NULL; }
    if (s_mic_f32) { free(s_mic_f32); s_mic_f32 = NULL; }
    if (s_rx_ring) { free(s_rx_ring); s_rx_ring = NULL; }
    if (s_mic_task_buf) { free(s_mic_task_buf); s_mic_task_buf = NULL; }
    if (s_dbg_hist) { free(s_dbg_hist); s_dbg_hist = NULL; }
    s_dbg_hist_pos = 0;
    s_rx_feed_total = 0;
    s_rx_bytes_len = 0;
    s_rx_new_text = 0;
}

void ui_ofdm_rx_on_enter(Key_Event *key_event, Global_State *global_state) {
    s_rx_bytes = (uint8_t *)platform_calloc(OFDM_RX_BYTES_CAP, 1);
    s_rx_wcbuf = (wchar_t *)platform_calloc(OFDM_RX_BYTES_CAP + 1, sizeof(wchar_t));
    s_mic_i16 = (int16_t *)platform_calloc(OFDM_RX_CHUNK, sizeof(int16_t));
    s_mic_f32 = (float *)platform_calloc(OFDM_RX_CHUNK, sizeof(float));
    s_dbg_hist = (float *)platform_calloc(OFDM_DBG_HIST_LEN, sizeof(float));
    s_rx_ring = (int16_t *)platform_calloc(OFDM_RX_RING_CAP, sizeof(int16_t));
    s_mic_task_buf = (int16_t *)platform_calloc(OFDM_RX_CHUNK, sizeof(int16_t));
    s_ring_w = 0; s_ring_r = 0; s_ring_ov = 0;
    s_rx_bytes_len = 0;
    s_rx_new_text = 0;
    s_rx_last_txt_ts = 0;
    s_rx_last_sta_ts = 0;
    s_rx_follow = 1;
    s_utf8_pend_len = 0;
    s_utf8_line_len = 0;
    s_mic_err_cnt = 0;
    s_mic_rms_ema = 0.0f;
    s_mic_peak = 0;
    s_rx_gs = global_state;

    s_rx = ofdm_rx_create(ofdm_alloc_psram, ofdm_free_psram, ui_ofdm_rx_on_text, ui_ofdm_rx_on_log, NULL);

    ui_draw_header(key_event, global_state, (wchar_t *)L"寻呼机·接收", 1);
    ui_draw_footer(key_event, global_state, (wchar_t *)L"←→滚行 D:回到底部 A:退出", 1);
    ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, (wchar_t *)L"", 0, 1);
    ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

    if (!s_rx || !s_rx_bytes || !s_rx_wcbuf || !s_mic_i16 || !s_mic_f32 || !s_dbg_hist ||
        !s_rx_ring || !s_mic_task_buf) {
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main,
                               (wchar_t *)L"内存不足，无法启动接收", 0, 0);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        printf("ofdm-rx: OOM, rx aborted\n");
        ui_ofdm_rx_cleanup();
        return;
    }

    // 接管麦克风（48kHz；会关闭扬声器，退出时 mic_close 恢复）
    s_mic_ok = (mic_init(OFDM_SAMPLE_RATE, (uint8_t)global_state->volume) == 0) ? 1 : 0;
    if (!s_mic_ok) {
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main,
                               (wchar_t *)L"麦克风初始化失败", 0, 0);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        printf("ofdm-rx: mic_init(48000) FAILED\n");
    }
    else {
        // 丢弃唤醒期采样（SPM1423HM4H-B 时钟起振后 10ms 才达规格，取 2 块 ≈40ms 裕量）
        mic_read(s_mic_i16, OFDM_RX_CHUNK);
        mic_read(s_mic_i16, OFDM_RX_CHUNK);
        // 启动采集任务（Core1 高优先级，与渲染任务解耦；此后渲染任务仅从环取数）
        s_mic_task_stop = 0;
        if (platform_task_create(ui_ofdm_mic_task, "ofdm_mic", 3072,
                                 NULL, 3, 1, &s_mic_task) != 0) {
            s_mic_task = NULL;
            s_mic_ok = 0;
            printf("ofdm-rx: mic task create FAILED\n");
        }
        printf("ofdm-rx: mic_init(48000) ok, rx ready\n");
        // 启动日志（同时验证屏幕日志显示路径）
        ui_ofdm_rx_on_log("接收已启动（48kHz），等待信号…", NULL);
    }
}

int32_t ui_ofdm_rx_event(Key_Event *key_event, Global_State *global_state) {
    // 异常兜底（如 on_enter 内存不足）：直接回模式菜单
    if (!s_rx) return STATE_OFDM_MENU;

    // 短按 A 退出（关闭麦克风并恢复扬声器）
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_esc) {
        ui_ofdm_rx_cleanup();
        return STATE_OFDM_MENU;
    }

    Widget_Textarea_State *ts = global_state->w_textarea_main;

    // 左右键：手动滚行（暂停自动跟随）；短按 D：回到底部并恢复自动跟随
    if (key_event->key_edge < 0 &&
        (key_event->key_code == NANO_KEY_left || key_event->key_code == NANO_KEY_right)) {
        s_rx_follow = 0;
        ui_widget_textarea_event_handler(key_event, global_state, ts, STATE_OFDM_RX, STATE_OFDM_RX);
    }
    else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
        s_rx_follow = 1;
        ui_widget_textarea_set(key_event, global_state, ts, ts->text, -1, 1);
        ui_widget_textarea_draw(key_event, global_state, ts);
    }
    // 短按 C：串口转储最近 4096 采样（CSV，int16 域），供宿主机离线分析
    else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_ctrl) {
        if (s_dbg_hist) {
            printf("DUMP_BEGIN %lu\n", (unsigned long)OFDM_DBG_HIST_LEN);
            for (uint32_t i = 0; i < OFDM_DBG_HIST_LEN; i++) {
                float v = s_dbg_hist[(s_dbg_hist_pos + i) % OFDM_DBG_HIST_LEN] * 32768.0f;
                printf("%d%c", (int)v, (i % 8 == 7) ? '\n' : ',');
            }
            printf("DUMP_END\n");
        }
    }

    // 从采集环取一块（≤960）→ int16 转 float → 流式喂入解调。
    // 采集由专用任务实时完成（见 ui_ofdm_mic_task），此处仅消费环中已有数据，不阻塞；
    // 环中暂无数据时直接跳过，事件循环以全速空转（vTaskDelay(0) 让出），UI 保持响应。
    if (s_mic_ok && s_rx && s_rx_ring) {
        uint32_t r = s_ring_r, w = s_ring_w;
        platform_memory_barrier(); // 写序号先于数据读取（跨核 SPSC）
        uint32_t avail = w - r;
        if (avail > OFDM_RX_RING_CAP) { s_ring_r = w; avail = 0; } // 序号异常兜底（不应发生）
        if (avail > 0) {
            int32_t n = (avail > OFDM_RX_CHUNK) ? OFDM_RX_CHUNK : (int32_t)avail; // 余量留环内，不丢
            double sum = 0.0, sumsq = 0.0;
            int32_t peak = 0;
            for (int32_t i = 0; i < n; i++) {
                int32_t v = s_rx_ring[(r + i) & (OFDM_RX_RING_CAP - 1)];
                sum += v;
                sumsq += (double)v * v;
                int32_t a = (v < 0) ? -v : v;
                if (a > peak) peak = a;
                s_mic_f32[i] = (float)v / 32768.0f;
            }
            s_ring_r = r + (uint32_t)n;
            double rms = sqrt(sumsq / n);
            s_mic_rms_ema = (s_mic_rms_ema == 0.0f) ? (float)rms : 0.7f * s_mic_rms_ema + 0.3f * (float)rms;
            s_mic_mean_ema = 0.7f * s_mic_mean_ema + 0.3f * (float)(sum / n);
            s_mic_peak = peak;
            ofdm_rx_feed(s_rx, s_mic_f32, (uint32_t)n);
            // 诊断：写入采样历史环形缓冲
            s_rx_feed_total += (uint32_t)n;
            for (int32_t i = 0; i < n; i++) {
                s_dbg_hist[s_dbg_hist_pos] = s_mic_f32[i];
                s_dbg_hist_pos = (s_dbg_hist_pos + 1) % OFDM_DBG_HIST_LEN;
            }
        }
    }

    // 串口诊断输出（1.5s 周期：状态行；每 2 拍：滞后扫描；每 4 拍：频谱快照）
    if (global_state->timestamp - s_rx_last_dbg_ts >= 1500) {
        s_rx_last_dbg_ts = global_state->timestamp;
        s_rx_dbg_tick++;
        OFDM_RX_Stat st;
        ofdm_rx_get_stat(s_rx, &st);
        float rms_db = (s_mic_rms_ema > 1.0f) ? 20.0f * log10f(s_mic_rms_ema / 32768.0f) : -99.0f;
        printf("ofdm-rx: %s sc_metric=%.3f rms=%.1fdB dc=%.0f pk=%d err=%d ov=%lu fed=%lu ok=%lu bad=%lu\n",
               st.locked ? "frame" : "sync", (double)st.sc_metric, (double)rms_db,
               (double)s_mic_mean_ema, (int)s_mic_peak, (int)s_mic_err_cnt, (unsigned long)s_ring_ov,
               (unsigned long)s_rx_feed_total,
               (unsigned long)st.frames_ok, (unsigned long)st.frames_bad);
        if (s_rx_dbg_tick % 2 == 0) ui_ofdm_rx_lag_scan();
        if (s_rx_dbg_tick % 4 == 0) ui_ofdm_rx_spectrum_scan();
    }

    // 限频刷新接收文本（跟随模式：滚到最底部）
    int32_t text_redrawn = 0;
    if (s_rx_new_text && global_state->timestamp - s_rx_last_txt_ts >= OFDM_TXT_REFRESH_MS) {
        s_rx_last_txt_ts = global_state->timestamp;
        s_rx_new_text = 0;
        text_redrawn = 1;
        _mbstowcs(s_rx_wcbuf, (const char *)s_rx_bytes, s_rx_bytes_len);
        // 文本超过控件上限时显示尾部（最新内容）
        uint32_t wlen = wcslen(s_rx_wcbuf);
        wchar_t *view = s_rx_wcbuf;
        if (wlen > UI_STR_BUF_MAX_LENGTH - 64) view = s_rx_wcbuf + (wlen - (UI_STR_BUF_MAX_LENGTH - 64));
        int32_t line = -1; // 跟随模式：自动滚动到最底部
        if (!s_rx_follow) {
            line = ts->current_line; // 手动滚动模式：保持视口
            int32_t max_line = ts->line_num - ts->view_lines;
            if (line > max_line) line = (max_line > 0) ? max_line : 0;
            if (line < 0) line = 0;
        }
        ui_widget_textarea_set(key_event, global_state, ts, view, line, 1);
        ui_widget_textarea_draw(key_event, global_state, ts);
    }

    // 限频更新状态栏（锁定状态 / CFO / SFO / 帧统计 / 麦克风电平诊断）
    // 注：与文本重绘错开（单次迭代至多一次全量刷帧，避免麦克风 DMA 消费间隙过长）
    if (!text_redrawn && global_state->timestamp - s_rx_last_sta_ts >= OFDM_STA_REFRESH_MS) {
        s_rx_last_sta_ts = global_state->timestamp;
        OFDM_RX_Stat st;
        ofdm_rx_get_stat(s_rx, &st);
        float rms_db = (s_mic_rms_ema > 1.0f) ? 20.0f * log10f(s_mic_rms_ema / 32768.0f) : -99.0f;
        if (st.locked) {
            swprintf(s_rx_status, sizeof(s_rx_status) / sizeof(wchar_t),
                     L"已锁定 | 解码帧%d/%d",
                     (int)st.frames_ok, (int)(st.frames_ok + st.frames_bad));
            // swprintf(s_rx_status, sizeof(s_rx_status) / sizeof(wchar_t),
            //          L"已锁定 %.0fdB | CFO%.1f SFO%.0f | 帧%d/%d",
            //          (double)rms_db, (double)st.cfo_hz, (double)st.sfo_ppm,
            //          (int)st.frames_ok, (int)st.frames_bad);
        }
        else {
            swprintf(s_rx_status, sizeof(s_rx_status) / sizeof(wchar_t),
                     L"搜索前导…  %.0fdB",
                     (double)rms_db);
            // swprintf(s_rx_status, sizeof(s_rx_status) / sizeof(wchar_t),
            //          L"搜索前导 %.0fdB pk%d err%d | 帧%d/%d",
            //          (double)rms_db, (int)s_mic_peak, (int)s_mic_err_cnt,
            //          (int)st.frames_ok, (int)st.frames_bad);
        }
        ui_draw_footer(key_event, global_state, s_rx_status, 1);
        gfx_refresh(global_state->gfx);
    }
    return STATE_OFDM_RX;
}

// ============================================================================
// STATE_OFDM_LOOP / STATE_OFDM_LOOPING（软件环路自测：自发自收，不出声）
// 发射机逐帧渲染 → 直接喂给本机接收机 → 显示解调文本与环回校验结果。
// 处理按帧分摊到多个事件循环迭代，UI 保持响应，可中途按 A 中止。
// ============================================================================

#define OFDM_LOOP_PAD (2400) // 前导静默采样数（模拟冷启动捕获）
#define OFDM_LOOP_LOG_LINES (10) // 屏幕调试日志行数（环形）
#define OFDM_LOOP_LOG_COLS  (64) // 每行宽字符数

static OFDM_TX *s_loop_tx = NULL;
static OFDM_RX *s_loop_rx = NULL;
static float *s_loop_frame = NULL;       // 帧渲染缓冲（兼作静默块）
static uint8_t *s_loop_payload = NULL;   // 发送净荷副本（环回校验用）
static uint32_t s_loop_payload_len = 0;
static uint8_t *s_loop_cap = NULL;       // 接收净荷累积（PSRAM）
static uint32_t s_loop_cap_len = 0;
static wchar_t *s_loop_view = NULL;      // 结果显示缓冲（PSRAM，UI_STR_BUF_MAX_LENGTH）
static wchar_t *s_loop_log = NULL;       // 调试日志环形行缓冲（PSRAM，LINES×COLS）
static int32_t s_loop_log_head = 0;      // 下一写入行
static int32_t s_loop_log_count = 0;     // 已记录行数（≤LINES）
static uint32_t s_loop_total = 0;        // 总帧数
static uint32_t s_loop_idx = 0;          // 已渲染帧数
static int32_t s_loop_stage = 0;         // 0=前导静默 1=逐帧环回
static int32_t s_loop_done = 0;

// 调试日志（来自 ofdm_rx 的 log_cb）：UTF-8 → wchar 追加一行到环形行缓冲
static void ui_ofdm_loop_log_line(const char *msg) {
    if (!s_loop_log || !msg) return;
    wchar_t *line = s_loop_log + s_loop_log_head * OFDM_LOOP_LOG_COLS;
    uint32_t blen = strlen(msg);
    if (blen > OFDM_LOOP_LOG_COLS - 1) blen = OFDM_LOOP_LOG_COLS - 1; // 超长截断（可能显示'?'）
    _mbstowcs(line, msg, blen);
    s_loop_log_head = (s_loop_log_head + 1) % OFDM_LOOP_LOG_LINES;
    if (s_loop_log_count < OFDM_LOOP_LOG_LINES) s_loop_log_count++;
}
static void ui_ofdm_loop_on_log(const char *msg, void *user) {
    (void)user;
    printf("ofdm-loop log: %s\n", msg);
    ui_ofdm_loop_log_line(msg);
}

// 接收净荷回调：追加到环回累积缓冲（溢出丢弃最旧一半）
static void ui_ofdm_loop_on_text(const uint8_t *bytes, uint32_t len, void *user) {
    (void)user;
    printf("ofdm-loop: packet 解出 %lu 字节\n", (unsigned long)len);
    if (!s_loop_cap || len == 0) return;
    if (s_loop_cap_len + len > OFDM_RX_BYTES_CAP) {
        uint32_t keep = OFDM_RX_BYTES_CAP / 2;
        memmove(s_loop_cap, s_loop_cap + s_loop_cap_len - keep, keep);
        s_loop_cap_len = keep;
    }
    if (len > OFDM_RX_BYTES_CAP - s_loop_cap_len) len = OFDM_RX_BYTES_CAP - s_loop_cap_len;
    memcpy(s_loop_cap + s_loop_cap_len, bytes, len);
    s_loop_cap_len += len;
}

static void ui_ofdm_looping_cleanup(void) {
    if (s_loop_tx) { ofdm_tx_destroy(s_loop_tx); s_loop_tx = NULL; }
    if (s_loop_rx) { ofdm_rx_destroy(s_loop_rx); s_loop_rx = NULL; }
    if (s_loop_frame) { free(s_loop_frame); s_loop_frame = NULL; }
    if (s_loop_payload) { free(s_loop_payload); s_loop_payload = NULL; }
    if (s_loop_cap) { free(s_loop_cap); s_loop_cap = NULL; }
    if (s_loop_view) { free(s_loop_view); s_loop_view = NULL; }
    if (s_loop_log) { free(s_loop_log); s_loop_log = NULL; }
    s_loop_log_head = 0;
    s_loop_log_count = 0;
    s_loop_payload_len = 0;
    s_loop_cap_len = 0;
    s_loop_total = 0;
    s_loop_idx = 0;
    s_loop_stage = 0;
    s_loop_done = 0;
}

void ui_ofdm_loop_on_enter(Key_Event *key_event, Global_State *global_state) {
    // 从菜单进入时文本已由 ui_widget_input_init 清空；从环回返回时保留原文
    ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
}

int32_t ui_ofdm_loop_event(Key_Event *key_event, Global_State *global_state) {
    int32_t next = ui_widget_input_event_handler(
        key_event, global_state, global_state->w_input_main,
        STATE_OFDM_MENU, STATE_OFDM_LOOP, STATE_OFDM_LOOPING);
    // 空文本不进入环回态（停留在输入态）
    if (next == STATE_OFDM_LOOPING && wcslen(global_state->w_input_main->textarea.text) == 0) {
        return STATE_OFDM_LOOP;
    }
    return next;
}

void ui_ofdm_looping_on_enter(Key_Event *key_event, Global_State *global_state) {
    // 文本（wchar_t）→ UTF-8 字节流（保留副本供环回校验）
    wchar_t *text = global_state->w_input_main->textarea.text;
    uint32_t wlen = wcslen(text);
    uint32_t cap = wlen * 4 + 1;
    s_loop_payload = (uint8_t *)platform_calloc(cap, 1);
    s_loop_payload_len = _wcstombs((char *)s_loop_payload, text, cap);

    s_loop_tx = ofdm_tx_create(ofdm_alloc_psram, ofdm_free_psram, s_loop_payload, s_loop_payload_len);
    s_loop_rx = ofdm_rx_create(ofdm_alloc_psram, ofdm_free_psram, ui_ofdm_loop_on_text, ui_ofdm_loop_on_log, NULL);
    s_loop_frame = (float *)platform_calloc(OFDM_FRAME_LENGTH, sizeof(float));
    s_loop_cap = (uint8_t *)platform_calloc(OFDM_RX_BYTES_CAP, 1);
    s_loop_view = (wchar_t *)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
    s_loop_log = (wchar_t *)platform_calloc(OFDM_LOOP_LOG_LINES * OFDM_LOOP_LOG_COLS, sizeof(wchar_t));
    s_loop_log_head = 0;
    s_loop_log_count = 0;
    s_loop_cap_len = 0;
    s_loop_total = ofdm_tx_frame_count(s_loop_tx);
    s_loop_idx = 0;
    s_loop_stage = 0;
    s_loop_done = 0;

    printf("ofdm-loop: 开始环回，净荷 %lu 字节 → %lu 帧（帧长 %d 采样）\n",
           (unsigned long)s_loop_payload_len, (unsigned long)s_loop_total, (int)OFDM_FRAME_LENGTH);
    if (!s_loop_tx || !s_loop_rx || !s_loop_frame || !s_loop_cap || !s_loop_view || !s_loop_log) {
        printf("ofdm-loop: 内存分配失败 tx=%p rx=%p frame=%p cap=%p view=%p log=%p\n",
               s_loop_tx, s_loop_rx, s_loop_frame, s_loop_cap, s_loop_view, s_loop_log);
        ui_ofdm_looping_cleanup();
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main,
                               (wchar_t *)L"内存不足，无法启动环路自测", 0, 0);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        return;
    }

    ui_draw_header(key_event, global_state, (wchar_t *)L"寻呼机·环路自测", 1);
    ui_draw_footer(key_event, global_state, (wchar_t *)L"环回处理中…", 1);
    swprintf(s_loop_view, UI_STR_BUF_MAX_LENGTH,
             L"发送 %d 字节 → %d 个物理帧\n\n正在环回处理（不出声）……",
             (int)s_loop_payload_len, (int)s_loop_total);
    ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, s_loop_view, 0, 0);
    ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
}

// 环回完成：汇总统计与校验结果，显示调试日志与解调文本
static void ui_ofdm_looping_show_result(Key_Event *key_event, Global_State *global_state) {
    OFDM_RX_Stat st;
    ofdm_rx_get_stat(s_loop_rx, &st);
    int32_t match = (s_loop_cap_len == s_loop_payload_len &&
                     memcmp(s_loop_cap, s_loop_payload, s_loop_payload_len) == 0) ? 1 : 0;

    // 串口详细校验报告（重点关注字节级错位：缓冲区问题的典型特征）
    printf("ofdm-loop: ==== 环回完成 ====\n");
    printf("ofdm-loop: 统计 ok=%lu bad=%lu RS纠错=%lu RS失败=%lu\n",
           (unsigned long)st.frames_ok, (unsigned long)st.frames_bad,
           (unsigned long)st.rs_corrected, (unsigned long)st.rs_fail_blocks);
    printf("ofdm-loop: 净荷校验：发出 %lu 字节，收到 %lu 字节，%s\n",
           (unsigned long)s_loop_payload_len, (unsigned long)s_loop_cap_len,
           match ? "逐字节一致" : "不一致");
    if (!match) {
        uint32_t n = (s_loop_cap_len < s_loop_payload_len) ? s_loop_cap_len : s_loop_payload_len;
        uint32_t diff = 0;
        while (diff < n && s_loop_cap[diff] == s_loop_payload[diff]) diff++;
        if (diff >= n && s_loop_cap_len != s_loop_payload_len) {
            printf("ofdm-loop: 前 %lu 字节一致，长度不符（发出 %lu / 收到 %lu）\n",
                   (unsigned long)diff, (unsigned long)s_loop_payload_len, (unsigned long)s_loop_cap_len);
        }
        else {
            printf("ofdm-loop: 首个差异 @%lu：发出=%02x 收到=%02x\n",
                   (unsigned long)diff, s_loop_payload[diff], s_loop_cap[diff]);
            int32_t s0 = (int32_t)diff - 8; if (s0 < 0) s0 = 0;
            int32_t s1 = (int32_t)diff + 8;
            printf("ofdm-loop: 发出[%d..]:", (int)s0);
            for (int32_t i = s0; i < s1 && i < (int32_t)s_loop_payload_len; i++) printf(" %02x", s_loop_payload[i]);
            printf("\nofdm-loop: 收到[%d..]:", (int)s0);
            for (int32_t i = s0; i < s1 && i < (int32_t)s_loop_cap_len; i++) printf(" %02x", s_loop_cap[i]);
            printf("\n");
        }
    }

    s_loop_view[0] = L'\0';
    swprintf(s_loop_view, UI_STR_BUF_MAX_LENGTH,
             L"发送 %d 字节 → %d 个物理帧\n"
             L"接收：成功 %d 帧 / 坏帧 %d / RS纠错 %d 字节\n"
             L"环回校验：%ls\n"
             L"----------------\n",
             (int)s_loop_payload_len, (int)s_loop_total,
             (int)st.frames_ok, (int)st.frames_bad, (int)st.rs_corrected,
             match ? L"一致 ✓（解调可行）" : L"不一致 ✗");
    // 追加调试日志尾部（最近 6 行，环形行缓冲按时间顺序回放）
    if (s_loop_log_count > 0) {
        wcscat(s_loop_view, L"日志：\n");
        int32_t show = (s_loop_log_count < 6) ? s_loop_log_count : 6;
        int32_t first = (s_loop_log_head - show + OFDM_LOOP_LOG_LINES) % OFDM_LOOP_LOG_LINES;
        for (int32_t i = 0; i < show; i++) {
            wchar_t *line = s_loop_log + ((first + i) % OFDM_LOOP_LOG_LINES) * OFDM_LOOP_LOG_COLS;
            if (wcslen(s_loop_view) + wcslen(line) + 3 < UI_STR_BUF_MAX_LENGTH - 1) {
                wcscat(s_loop_view, L"·");
                wcscat(s_loop_view, line);
                wcscat(s_loop_view, L"\n");
            }
        }
    }
    else {
        wcscat(s_loop_view, L"日志：（无，未捕获到任何接收机事件）\n");
    }
    wcscat(s_loop_view, L"----------------\n");
    // 追加解调出的文本（受显示缓冲上限截断）
    uint32_t cur = wcslen(s_loop_view);
    uint32_t remain = (UI_STR_BUF_MAX_LENGTH - 1 > cur) ? (UI_STR_BUF_MAX_LENGTH - 1 - cur) : 0;
    if (remain > 0 && s_loop_cap_len > 0) {
        uint32_t blen = (s_loop_cap_len < remain) ? s_loop_cap_len : remain;
        _mbstowcs(s_loop_view + cur, (const char *)s_loop_cap, blen);
    }
    ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, s_loop_view, 0, 1);
    ui_draw_footer(key_event, global_state, (wchar_t *)L"D: 再次环回  A: 返回编辑", 1);
    ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
}

int32_t ui_ofdm_looping_event(Key_Event *key_event, Global_State *global_state) {
    // 异常兜底（如 on_enter 内存不足）：直接回输入态
    if (!s_loop_tx || !s_loop_rx) return STATE_OFDM_LOOP;

    // 短按 A：中止/返回编辑（保留输入文本）
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_esc) {
        ui_ofdm_looping_cleanup();
        return STATE_OFDM_LOOP;
    }

    // 左右键：滚行翻阅（结果页含统计/日志/解调文本，内容较长；与 RX 态同一模式）
    if (key_event->key_edge < 0 &&
        (key_event->key_code == NANO_KEY_left || key_event->key_code == NANO_KEY_right)) {
        ui_widget_textarea_event_handler(key_event, global_state,
                                         global_state->w_textarea_main,
                                         STATE_OFDM_LOOPING, STATE_OFDM_LOOPING);
        return STATE_OFDM_LOOPING;
    }

    // 完成后：短按 D 以同一文本再次环回
    if (s_loop_done) {
        if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_enter) {
            ui_ofdm_looping_cleanup();
            ui_ofdm_looping_on_enter(key_event, global_state);
        }
        return STATE_OFDM_LOOPING;
    }

    if (s_loop_stage == 0) {
        // 先喂一段静默（模拟冷启动同步捕获）
        memset(s_loop_frame, 0, OFDM_LOOP_PAD * sizeof(float));
        ofdm_rx_feed(s_loop_rx, s_loop_frame, OFDM_LOOP_PAD);
        printf("ofdm-loop: 已喂前导静默 %d 采样\n", (int)OFDM_LOOP_PAD);
        s_loop_stage = 1;
        return STATE_OFDM_LOOPING;
    }

    // 每轮迭代环回一帧（渲染 → 直接喂接收机，不经过扬声器/麦克风）
    if (s_loop_idx < s_loop_total) {
        int32_t rr = ofdm_tx_render_frame(s_loop_tx, s_loop_frame);
        if (rr == 0) {
            ofdm_rx_feed(s_loop_rx, s_loop_frame, OFDM_FRAME_LENGTH);
        }
        else {
            printf("ofdm-loop: 帧 %lu 渲染返回 %d（异常）\n", (unsigned long)s_loop_idx, (int)rr);
        }
        s_loop_idx++;
        // 缓冲区遥测：帧接收环/SC检测历史环占用、锁定状态、SC度量
        OFDM_RX_Stat st;
        ofdm_rx_get_stat(s_loop_rx, &st);
        printf("ofdm-loop: 帧 %lu/%lu 已喂入 | locked=%d buf=%lu/65536 det=%lu/16384 fed=%llu sc=%.3f ok=%lu bad=%lu\n",
               (unsigned long)s_loop_idx, (unsigned long)s_loop_total,
               (int)st.locked, (unsigned long)st.buf_len, (unsigned long)st.det_len,
               (unsigned long long)st.feed_abs, (double)st.sc_metric,
               (unsigned long)st.frames_ok, (unsigned long)st.frames_bad);
        swprintf(s_rx_status, sizeof(s_rx_status) / sizeof(wchar_t),
                 L"环回处理中 %d/%d 帧", (int)s_loop_idx, (int)s_loop_total);
        ui_draw_footer(key_event, global_state, s_rx_status, 1);
        gfx_refresh(global_state->gfx);
    }
    // 末帧喂完后再补一段尾部静默：把最后一个槽冲出缓冲（接收机消费槽需 1520 采样余量，
    // 否则末帧无法完成解包——真实空口中后续音频/噪声天然提供该余量）
    if (s_loop_idx >= s_loop_total && s_loop_stage == 1) {
        memset(s_loop_frame, 0, OFDM_LOOP_PAD * sizeof(float));
        ofdm_rx_feed(s_loop_rx, s_loop_frame, OFDM_LOOP_PAD);
        printf("ofdm-loop: 已喂尾部静默 %d 采样（冲刷末帧）\n", (int)OFDM_LOOP_PAD);
        s_loop_stage = 2;
        return STATE_OFDM_LOOPING;
    }
    if (s_loop_idx >= s_loop_total && s_loop_stage >= 2) {
        s_loop_done = 1;
        ui_ofdm_looping_show_result(key_event, global_state);
    }
    return STATE_OFDM_LOOPING;
}
