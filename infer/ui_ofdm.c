// ============================================================================
// 寻呼机（OFDM 声波数传）UI 模块 / BD4SUR 2026-07，界面重构 2026-11
// ----------------------------------------------------------------------------
// 本文件包含两部分：
//   1) 信号处理与音频收发的宿主胶水（采集任务/环形缓冲/流式解调/循环播放），
//      这部分逻辑与重构前一致——算法核在 ofdm_modem.c，本文件不触碰；
//   2) 寻呼机的全部 GUI：模式卡菜单、发射状态页、接收状态页、环路自测页。
//      由 ofdm_palette() 统一出亮/暗两套配色，全部界面元素自绘（圆角卡片、
//      状态点、电平条、进度条、星座图、帧波形、参数网格），不依赖菜单控件的样式。
//
// 数据流：
//   发射：输入文本 → UTF-8 → ofdm_tx（分帧/RS/交织/加扰）→ 乒乓双缓冲 → 扬声器
//   接收：麦克风 48kHz → SPSC 大环（采集任务）→ ofdm_rx 流式解调 → 滚动文本
//   环路：发射机逐帧渲染 → 直接喂本机接收机（不出声）→ 文本比对校验
// ============================================================================

#include "ui_ofdm.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
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
#define OFDM_STA_REFRESH_MS (120)  // 状态区限频刷新间隔（状态卡/电平/星座/波形）

// ---------------- ofdm_modem 内存注入（PSRAM） ----------------
static void *ofdm_alloc_psram(size_t size) { return platform_calloc(1, size); }
static void ofdm_free_psram(void *ptr) { free(ptr); }

// ============================================================================
// 一、配色与绘图基元（本模块自绘界面的公共层）
// ============================================================================

typedef struct { uint8_t r, g, b; } OFDM_RGB;

typedef struct {
    OFDM_RGB bg;          // 屏幕底色
    OFDM_RGB panel;       // 卡片/面板填充
    OFDM_RGB edge;        // 卡片描边
    OFDM_RGB hairline;    // 分隔线/刻度
    OFDM_RGB text;        // 主文字
    OFDM_RGB dim;         // 次要文字
    OFDM_RGB accent;      // 主题强调色
    OFDM_RGB accent_soft; // 强调色的浅底（选中卡片/徽标底）
    OFDM_RGB surface;     // 文本区底色（与文本控件自身的清屏色一致）
    OFDM_RGB track;       // 电平条/进度条槽
    OFDM_RGB ok;          // 正常/锁定
    OFDM_RGB warn;        // 中等/搜索中
    OFDM_RGB err;         // 异常/发射指示灯
    OFDM_RGB on_accent;   // 强调色块上的文字
} OFDM_Palette;

static OFDM_RGB ofdm_rgb(uint8_t r, uint8_t g, uint8_t b) {
    OFDM_RGB c; c.r = r; c.g = g; c.b = b; return c;
}

// 亮/暗两套配色。亮色为浅灰蓝底＋白卡片＋品牌蓝强调；暗色为近黑底＋深卡片＋亮蓝强调。
static void ofdm_palette(Global_State *gs, OFDM_Palette *p) {
    if (gs->ui_color_style == UI_COLOR_DARK) {
        p->bg          = ofdm_rgb(10, 11, 13);
        p->panel       = ofdm_rgb(23, 25, 29);
        p->edge        = ofdm_rgb(42, 46, 54);
        p->hairline    = ofdm_rgb(34, 37, 44);
        p->text        = ofdm_rgb(235, 238, 243);
        p->dim         = ofdm_rgb(138, 148, 162);
        p->accent      = ofdm_rgb(84, 148, 255);
        p->accent_soft = ofdm_rgb(20, 34, 60);
        p->surface     = ofdm_rgb(6, 6, 6);      // 与文本控件暗色清屏色一致
        p->track       = ofdm_rgb(38, 42, 50);
        p->ok          = ofdm_rgb(62, 200, 130);
        p->warn        = ofdm_rgb(240, 180, 70);
        p->err         = ofdm_rgb(245, 100, 100);
        p->on_accent   = ofdm_rgb(255, 255, 255);
    }
    else {
        p->bg          = ofdm_rgb(243, 245, 249);
        p->panel       = ofdm_rgb(255, 255, 255);
        p->edge        = ofdm_rgb(219, 225, 234);
        p->hairline    = ofdm_rgb(230, 234, 241);
        p->text        = ofdm_rgb(28, 33, 41);
        p->dim         = ofdm_rgb(116, 127, 143);
        p->accent      = ofdm_rgb(17, 85, 238);
        p->accent_soft = ofdm_rgb(231, 240, 255);
        p->surface     = ofdm_rgb(255, 255, 255); // 与文本控件亮色清屏色一致
        p->track       = ofdm_rgb(230, 234, 241);
        p->ok          = ofdm_rgb(16, 150, 80);
        p->warn        = ofdm_rgb(200, 122, 20);
        p->err         = ofdm_rgb(212, 58, 58);
        p->on_accent   = ofdm_rgb(255, 255, 255);
    }
}

// 本模块统一使用的字体：标题/正文用 12px 抗锯齿（行高15），状态大字用 16px（行高18）。
// 刻意不跟随 global_state->ui_font：布局按固定行高设计，换字号会溢出。
#define OFDM_F_TITLE (GFX_FONT_ALPHA_12) // 页眉/卡片标题/说明（行高 15）
#define OFDM_F_BIG   (GFX_FONT_ALPHA_16) // 状态大字（行高 18）

// 布局常量（像素）
#define OFDM_HDR_H   (26) // 自绘页眉高度
#define OFDM_FTR_H   (18) // 自绘页脚高度

static void ofdm_fill(Global_State *gs, int32_t x, int32_t y, int32_t w, int32_t h, OFDM_RGB c) {
    if (w <= 0 || h <= 0) return;
    gfx_draw_rectangle(gs->gfx, (uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h, c.r, c.g, c.b, 1);
}

static void ofdm_round(Global_State *gs, int32_t x, int32_t y, int32_t w, int32_t h, int32_t radius, OFDM_RGB c) {
    if (w <= 0 || h <= 0) return;
    gfx_draw_rounded_rectangle(gs->gfx, x, y, w, h, radius, radius, radius, radius, c.r, c.g, c.b, 1);
}

// 圆角卡片：外圈描边色 + 内缩 1px 的填充色（等效 1px 描边）
static void ofdm_card(Global_State *gs, int32_t x, int32_t y, int32_t w, int32_t h,
                      int32_t radius, OFDM_RGB fill, OFDM_RGB edge) {
    ofdm_round(gs, x, y, w, h, radius, edge);
    ofdm_round(gs, x + 1, y + 1, w - 2, h - 2, radius - 1, fill);
}

static void ofdm_line(Global_State *gs, int32_t x0, int32_t y0, int32_t x1, int32_t y1, OFDM_RGB c) {
    gfx_draw_line(gs->gfx, (uint32_t)x0, (uint32_t)y0, (uint32_t)x1, (uint32_t)y1, c.r, c.g, c.b, 1);
}

static int32_t ofdm_text_w(uint32_t font, wchar_t *s) {
    return (s == NULL) ? 0 : gfx_font_measure_text(font, s);
}

static void ofdm_text(Global_State *gs, uint32_t font, wchar_t *s, int32_t x, int32_t y_top, OFDM_RGB c) {
    if (s == NULL || s[0] == L'\0') return;
    gfx_font_draw_text(gs->gfx, font, s, x, y_top, c.r, c.g, c.b, 1);
}

static void ofdm_text_right(Global_State *gs, uint32_t font, wchar_t *s, int32_t x_right, int32_t y_top, OFDM_RGB c) {
    if (s == NULL || s[0] == L'\0') return;
    ofdm_text(gs, font, s, x_right - ofdm_text_w(font, s), y_top, c);
}

static void ofdm_text_center(Global_State *gs, uint32_t font, wchar_t *s, int32_t cx, int32_t cy, OFDM_RGB c) {
    if (s == NULL || s[0] == L'\0') return;
    gfx_font_draw_text_centered(gs->gfx, font, s, cx, cy, c.r, c.g, c.b, 1);
}

// 状态圆点（可选外圈呼吸环）
static void ofdm_dot(Global_State *gs, int32_t cx, int32_t cy, int32_t r, OFDM_RGB c, int32_t ring) {
    if (ring) gfx_draw_circle(gs->gfx, (uint32_t)cx, (uint32_t)cy, (uint32_t)(r + 2), c.r, c.g, c.b, 1);
    gfx_draw_circle_fill(gs->gfx, (uint32_t)cx, (uint32_t)cy, (uint32_t)r, c.r, c.g, c.b, 1);
}

// 水平进度/电平条：槽 + 圆角填充
static void ofdm_bar(Global_State *gs, int32_t x, int32_t y, int32_t w, int32_t h,
                     float frac, OFDM_RGB track, OFDM_RGB fill) {
    if (w <= 0 || h <= 0) return;
    ofdm_round(gs, x, y, w, h, h / 2, track);
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;
    int32_t fw = (int32_t)(frac * (float)w + 0.5f);
    if (fw <= 0) return;
    if (fw < h) ofdm_fill(gs, x, y, fw, h, fill); // 极短进度：圆角会吃掉整段，退化为直条
    else ofdm_round(gs, x, y, fw, h, h / 2, fill);
}

// 页眉：底色带 + 左强调条 + 标题 + 右上角“返回”（触屏热点：页眉右 1/4）+ 底部分隔线
static void ofdm_chrome_header(Global_State *gs, const OFDM_Palette *p, wchar_t *title) {
    int32_t W = (int32_t)gs->gfx->width;
    ofdm_fill(gs, 0, 0, W, OFDM_HDR_H, p->panel);
    ofdm_round(gs, 10, 7, 4, 12, 2, p->accent);
    ofdm_text(gs, OFDM_F_TITLE, title, 20, (OFDM_HDR_H - 15) / 2, p->text);
    ofdm_text_right(gs, OFDM_F_TITLE, (wchar_t *)L"返回", W - 10, (OFDM_HDR_H - 15) / 2, p->dim);
    ofdm_fill(gs, 0, OFDM_HDR_H - 1, W, 1, p->hairline);
}

// 页眉“返回”热区命中（页眉带右 1/4，与菜单控件/输入控件约定一致）
static int32_t ofdm_back_hit(Global_State *gs, int32_t x, int32_t y) {
    return (y >= 0 && y < OFDM_HDR_H && x >= (int32_t)gs->gfx->width * 3 / 4);
}

// 简易点按判定器：按下锚定，位移超阈值即视为拖动，松手未拖动则报告点按坐标
typedef struct {
    int32_t active;
    int32_t dragging;
    int32_t x;
    int32_t y;
} OFDM_Tap;

static int32_t ofdm_tap_feed(OFDM_Tap *t, Key_Event *ke, int32_t *x, int32_t *y) {
    int32_t tapped = 0;
    if ((ke->touch_edge & TOUCH_EDGE_DOWN) && !t->active) {
        t->active = 1;
        t->dragging = 0;
        t->x = ke->touch_down_x;
        t->y = ke->touch_down_y;
    }
    else if (ke->is_touching && t->active) {
        int32_t dx = ke->touch_x - t->x;
        int32_t dy = ke->touch_y - t->y;
        if (dx > 10 || dx < -10 || dy > 10 || dy < -10) t->dragging = 1;
    }
    else if ((ke->touch_edge & TOUCH_EDGE_UP) && t->active) {
        t->active = 0;
        if (!t->dragging) {
            *x = t->x;
            *y = t->y;
            tapped = 1;
        }
    }
    return tapped;
}

// 页脚：底部提示带（左提示 + 右操作）
static void ofdm_chrome_footer(Global_State *gs, const OFDM_Palette *p, wchar_t *left, wchar_t *right) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t H = (int32_t)gs->gfx->height;
    int32_t y = H - OFDM_FTR_H;
    ofdm_fill(gs, 0, y, W, OFDM_FTR_H, p->panel);
    ofdm_fill(gs, 0, y, W, 1, p->hairline);
    ofdm_text(gs, OFDM_F_TITLE, left, 10, y + 2, p->dim);
    if (right) ofdm_text_right(gs, OFDM_F_TITLE, right, W - 10, y + 2, p->accent);
}

// 键值对（参数网格单元）：标签为次要色、数值为主色
static void ofdm_kv(Global_State *gs, const OFDM_Palette *p, int32_t x, int32_t y,
                    int32_t label_w, wchar_t *label, wchar_t *value) {
    ofdm_text(gs, OFDM_F_TITLE, label, x, y, p->dim);
    ofdm_text(gs, OFDM_F_TITLE, value, x + label_w, y, p->text);
}

// 文本区（接收内容/自测结果）的几何：卡片 + 标题行（22px）+ 嵌入文本框。
// 卡片下沿锚定在 y_bottom，高度按正文行高取整行，避免底部出现半行截断的空隙。
#define OFDM_TEXT_CARD_RADIUS (10)

static void ofdm_text_card_setup(Global_State *gs, Widget_Textarea_State *ts,
                                 int32_t y_top_limit, int32_t y_bottom) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t line_h = gfx_font_line_height(gs->ui_font);
    int32_t avail = (y_bottom - 6) - (y_top_limit + 24);
    int32_t lines = avail / line_h;
    if (lines < 1) lines = 1;
    int32_t ta_h = lines * line_h;
    int32_t card_h = 24 + ta_h + 6;
    ts->x = 16;
    ts->y = (y_bottom - card_h) + 24;
    ts->width = W - 32;
    ts->height = ta_h;
    ts->is_bare = 1;
    ts->is_show_scroll_bar = 1;
    ts->is_modified = 1;
    ts->scroll_sub_offset = 0;
}

static void ofdm_text_card_draw(Global_State *gs, const OFDM_Palette *p,
                                Widget_Textarea_State *ts, wchar_t *title, wchar_t *right) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t cx = 10;
    int32_t cw = W - 20;
    int32_t cy = ts->y - 24;
    int32_t ch = 24 + ts->height + 6;
    // 卡片（文本区底色与文本控件清屏色一致，视觉上浑然一体）
    ofdm_card(gs, cx, cy, cw, ch, OFDM_TEXT_CARD_RADIUS, p->surface, p->edge);
    ofdm_text(gs, OFDM_F_TITLE, title, cx + 10, cy + 3, p->dim);
    if (right) ofdm_text_right(gs, OFDM_F_TITLE, right, cx + cw - 10, cy + 3, p->dim);
    ofdm_fill(gs, cx + 1, cy + 21, cw - 2, 1, p->hairline);
}

// 退出嵌入模式时复位文本框为标准全屏几何（避免影响其它界面的文本控件）
static void ofdm_text_card_restore(Global_State *gs, Widget_Textarea_State *ts) {
    ts->is_bare = 0;
    ui_widget_textarea_reset_geometry(NULL, gs, ts);
    ts->is_show_scroll_bar = 1;
}

// 文本卡的空内容占位提示
static void ofdm_text_card_placeholder(Global_State *gs, const OFDM_Palette *p,
                                       Widget_Textarea_State *ts, wchar_t *line1, wchar_t *line2) {
    (void)p;
    OFDM_Palette pal;
    ofdm_palette(gs, &pal);
    ofdm_text_center(gs, OFDM_F_BIG, line1, ts->x + ts->width / 2, ts->y + ts->height / 2 - 10, pal.dim);
    ofdm_text_center(gs, OFDM_F_TITLE, line2, ts->x + ts->width / 2, ts->y + ts->height / 2 + 12, pal.dim);
}

// 通用错误页（内存不足/硬件不可用等）：页眉 + 居中提示 + 页脚
static void ofdm_error_page(Key_Event *ke, Global_State *gs, wchar_t *title, wchar_t *headline, wchar_t *hint) {
    (void)ke;
    OFDM_Palette p;
    ofdm_palette(gs, &p);
    int32_t W = (int32_t)gs->gfx->width;
    int32_t H = (int32_t)gs->gfx->height;
    ofdm_fill(gs, 0, 0, W, H, p.bg);
    ofdm_chrome_header(gs, &p, title);
    ofdm_card(gs, 10, OFDM_HDR_H + 30, W - 20, 92, 12, p.panel, p.edge);
    ofdm_dot(gs, 30, OFDM_HDR_H + 56, 5, p.err, 1);
    ofdm_text(gs, OFDM_F_BIG, headline, 46, OFDM_HDR_H + 46, p.err);
    ofdm_text(gs, OFDM_F_TITLE, hint, 22, OFDM_HDR_H + 82, p.dim);
    ofdm_chrome_footer(gs, &p, (wchar_t *)L"寻呼机", (wchar_t *)L"A 返回");
    gfx_refresh(gs->gfx);
}

// ============================================================================
// 二、状态变量
// ============================================================================

// ---------------- 发射侧静态状态 ----------------
static OFDM_TX *s_tx = NULL;
static int16_t *s_pcm[AUDIO_OUT_QUEUE_DEPTH] = {NULL, NULL}; // 乒乓双缓冲（playRaw 引用，须常驻至播完）
static int32_t s_fill_buf = 0;     // 下一个待填充/投放的缓冲序号（0/1 轮换）
static float *s_frame_f32 = NULL;  // 帧渲染缓冲
static int32_t s_audio_on = 0;     // audio_out_init 是否已调用
static int32_t s_tx_audio_ok = 0;  // 音频输出是否可用（失败时仅演示界面）
static uint32_t s_tx_frames_played = 0;
static uint32_t s_tx_total_frames = 0;  // 本轮净荷的物理帧总数
static uint32_t s_tx_payload_bytes = 0; // 净荷字节数
static uint64_t s_tx_start_ts = 0;      // 本轮发射起始时间戳（周期进度/播放头用）
static uint64_t s_tx_last_draw_ts = 0;
static OFDM_Tap s_tx_tap;          // 页眉“返回/停止”点按判定

// 帧波形可视化（每渲染一帧刷新一次；仅显示用，不参与信号处理）
#define OFDM_TX_BARS (70)
static float s_tx_bars[OFDM_TX_BARS];

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
static int32_t s_rx_follow = 1;        // 自动滚动到底部跟随模式（左右键手动滚动时暂停）
static volatile int32_t s_mic_err_cnt = 0; // mic_read 错误计数（采集任务写，渲染任务读）
static float s_mic_rms_ema = 0.0f;     // 接收电平 RMS（int16 域，指数滑动平均）
static float s_mic_mean_ema = 0.0f;    // 直流分量（int16 域均值，指数滑动平均）
static int32_t s_mic_peak = 0;         // 最近块峰值（int16 域）
static Global_State *s_rx_gs = NULL;   // 供回调取 UI 色彩风格
static int32_t s_rx_ui_init = 0;       // 接收界面（文本框嵌入几何）是否已建立
static OFDM_Tap s_rx_tap;              // 页眉“返回”点按判定

// 接收会话（接收页与消息页共用同一套采集/解调通路，二者互斥）
#define OFDM_RX_SINK_SCREEN (0)
#define OFDM_RX_SINK_MSG    (1)
static int32_t s_rx_sink = OFDM_RX_SINK_SCREEN;

// 前向声明：消息页的接收字节入口（定义见第八节）
static void ofdm_msg_on_rx_bytes(const uint8_t *bytes, uint32_t len);

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

#if defined(NANO_WASM_GUI)
// ============================================================================
// 【WASM 专属】单线程平台的麦克风采集（不改动其它平台的任何既有实现）
// ----------------------------------------------------------------------------
// 浏览器 WASM 没有 pthread，platform_task_create 恒失败 → 上面那条采集任务建不出来。
// 而 hal_audio_in_wasm 的 mic_read 是**非阻塞**的（数据不足一帧返回 0 且不消费），因此
// 可以把「读一块 → 入 SPSC 环」直接放进渲染循环，效果与采集任务等价。
// 注意：其它平台的 mic_read 是阻塞式（ESP32 I2S 等 100ms / ALSA、tinyalsa 等数据至
// 超时），在渲染循环里轮询会卡住 UI——所以这段只在 NANO_WASM_GUI 下编译。
// ============================================================================
static int32_t s_mic_poll = 0; // 1=采集由渲染循环自行完成（无采集任务）

static void ui_ofdm_mic_poll(void) {
    if (!s_mic_task_buf || !s_rx_ring) return;
    int32_t n = mic_read(s_mic_task_buf, OFDM_RX_CHUNK);
    if (n <= 0) return; // 0 = 未攒够一帧（未消费，下次再来）；<0 = 硬错误
    uint32_t w = s_ring_w, r = s_ring_r;
    if (OFDM_RX_RING_CAP - (w - r) < (uint32_t)n) {
        s_ring_ov += (uint32_t)n; // 环满（UI 停滞 >1.3s）：弃新块保旧数据，计数告警
        return;
    }
    for (int32_t i = 0; i < n; i++) s_rx_ring[(w + i) & (OFDM_RX_RING_CAP - 1)] = s_mic_task_buf[i];
    s_ring_w = w + (uint32_t)n; // 单线程：生产/消费同在渲染循环，无需内存屏障
}
#endif // NANO_WASM_GUI

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

// 消息收发（第八节）的存储生命周期：模式菜单进入/退出时调用
static void ofdm_msg_store_init(void);
static void ofdm_msg_store_free(void);

// ============================================================================
// 三、模式菜单（四张功能卡，2×2 宫格）
// ============================================================================

#define OFDM_HUB_ITEMS (4)

static int32_t s_hub_sel = 0;
static OFDM_Tap s_hub_tap;

// 发射/环路自测输入编辑器的默认文本（火腿呼叫 + 测试文）
static const wchar_t OFDM_DEFAULT_TX_TEXT[] =
    L"CQ CQ DE BD4SUR 声学OFDM数传测试\n\n"
    L"本系统利用声波传输文本数据。\n"
    L"发射机对消息作RS编码、交织和加扰，调制为OFDM符号，上变频到4kHz发射。\n"
    L"接收机将音频信号解调为OFDM符号，经过解扰、解交织和RS解码，得到字节序列。\n"
    L"系统能够适应比较复杂的声学信道。\n\n";

static wchar_t *ofdm_hub_code(int32_t idx) {
    switch (idx) {
        case 0: return (wchar_t *)L"TX";
        case 1: return (wchar_t *)L"RX";
        case 2: return (wchar_t *)L"LP";
        default: return (wchar_t *)L"MS";
    }
}

static wchar_t *ofdm_hub_title(int32_t idx) {
    switch (idx) {
        case 0: return (wchar_t *)L"发送消息";
        case 1: return (wchar_t *)L"接收消息";
        case 2: return (wchar_t *)L"环路自测";
        default: return (wchar_t *)L"消息收发";
    }
}

static wchar_t *ofdm_hub_desc(int32_t idx) {
    switch (idx) {
        case 0: return (wchar_t *)L"文本 → 声波";
        case 1: return (wchar_t *)L"声波 → 文本";
        case 2: return (wchar_t *)L"本机自发自收";
        default: return (wchar_t *)L"收发一屏搞定";
    }
}

// 每张卡片的强调色与其浅底（亮/暗各一套）：TX 蓝 / RX 绿 / LP 紫 / MS 琥珀
static void ofdm_hub_accent(Global_State *gs, int32_t idx, OFDM_RGB *accent, OFDM_RGB *soft) {
    int32_t dark = (gs->ui_color_style == UI_COLOR_DARK);
    switch (idx) {
        case 0:
            *accent = dark ? ofdm_rgb(96, 156, 255) : ofdm_rgb(17, 85, 238);
            *soft   = dark ? ofdm_rgb(19, 33, 60)   : ofdm_rgb(231, 240, 255);
            break;
        case 1:
            *accent = dark ? ofdm_rgb(62, 200, 130) : ofdm_rgb(15, 145, 78);
            *soft   = dark ? ofdm_rgb(17, 42, 32)   : ofdm_rgb(228, 247, 236);
            break;
        case 2:
            *accent = dark ? ofdm_rgb(178, 142, 255) : ofdm_rgb(124, 58, 237);
            *soft   = dark ? ofdm_rgb(36, 29, 58)    : ofdm_rgb(243, 238, 255);
            break;
        default:
            *accent = dark ? ofdm_rgb(245, 185, 70) : ofdm_rgb(180, 106, 10);
            *soft   = dark ? ofdm_rgb(48, 36, 14)   : ofdm_rgb(255, 246, 226);
            break;
    }
}

// 2×2 宫格：每卡 146×88，间距 8，内容区自页眉下 4px 起
static void ofdm_hub_card_rect(Global_State *gs, int32_t idx, int32_t *x, int32_t *y, int32_t *w, int32_t *h) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t gap = 8;
    *w = (W - 20 - gap) / 2;
    *h = 88;
    *x = 10 + (idx % 2) * (*w + gap);
    *y = OFDM_HDR_H + 4 + (idx / 2) * (*h + gap);
}

static void ofdm_hub_draw(Key_Event *ke, Global_State *gs) {
    (void)ke;
    OFDM_Palette p;
    ofdm_palette(gs, &p);
    int32_t W = (int32_t)gs->gfx->width;
    int32_t H = (int32_t)gs->gfx->height;

    ofdm_fill(gs, 0, 0, W, H, p.bg);
    ofdm_chrome_header(gs, &p, (wchar_t *)L"寻呼机");

    for (int32_t i = 0; i < OFDM_HUB_ITEMS; i++) {
        int32_t x, y, w, h;
        ofdm_hub_card_rect(gs, i, &x, &y, &w, &h);
        int32_t selected = (i == s_hub_sel);
        OFDM_RGB accent, soft;
        ofdm_hub_accent(gs, i, &accent, &soft);

        ofdm_card(gs, x, y, w, h, 12, selected ? soft : p.panel, selected ? accent : p.edge);
        if (selected) ofdm_round(gs, x + 5, y + 12, 3, h - 24, 2, accent);

        // 徽标：功能缩写（TX/RX/LP/MS）
        ofdm_round(gs, x + 10, y + 10, 30, 30, 9, soft);
        ofdm_text_center(gs, OFDM_F_TITLE, ofdm_hub_code(i), x + 25, y + 25, accent);

        // 标题 + 一行说明
        ofdm_text(gs, OFDM_F_TITLE, ofdm_hub_title(i), x + 48, y + 8, selected ? accent : p.text);
        ofdm_text(gs, OFDM_F_TITLE, ofdm_hub_desc(i), x + 48, y + 26, p.dim);

        // 右下角快捷键序号（1~4），选中时反白
        wchar_t key[2] = {(wchar_t)(L'1' + i), L'\0'};
        ofdm_round(gs, x + w - 24, y + h - 22, 16, 16, 8, selected ? accent : p.track);
        ofdm_text_center(gs, OFDM_F_TITLE, key, x + w - 16, y + h - 14, selected ? p.on_accent : p.dim);
    }

    ofdm_chrome_footer(gs, &p, (wchar_t *)L"1–4 直选 · 点按进入", (wchar_t *)L"D 进入 · A 返回");
    gfx_refresh(gs->gfx);
}

// 以默认文本填充输入控件（init 已清空；此后用户可编辑，提交返回后保留修改）
static void ofdm_input_fill_default_text(Key_Event *ke, Global_State *gs) {
    Widget_Textarea_State *ta = &gs->w_input_main->textarea;
    wcsncpy(ta->text, OFDM_DEFAULT_TX_TEXT, UI_STR_BUF_MAX_LENGTH - 1);
    ta->text[UI_STR_BUF_MAX_LENGTH - 1] = L'\0';
    ta->length = wcslen(ta->text);
    ta->is_modified = 1;
    ui_widget_input_refresh(ke, gs, gs->w_input_main);
}

// 卡片激活：进入对应功能
static int32_t ofdm_hub_activate(Key_Event *ke, Global_State *gs, int32_t idx) {
    switch (idx) {
        case 0:
            // 进入发射文本输入（参照鹦鹉笼：选模型时初始化输入控件，提交后返回可继续编辑）
            ui_widget_input_init(ke, gs, gs->w_input_main, (wchar_t *)L"发报（Ctrl+A 返回）");
            ofdm_input_fill_default_text(ke, gs);
            return STATE_OFDM_TX;
        case 1:
            return STATE_OFDM_RX;
        case 2:
            ui_widget_input_init(ke, gs, gs->w_input_main, (wchar_t *)L"环路自测（Ctrl+A 返回）");
            ofdm_input_fill_default_text(ke, gs);
            return STATE_OFDM_LOOP;
        default:
            return STATE_OFDM_MSG;
    }
}

void ui_ofdm_menu_init(Key_Event *key_event, Global_State *global_state) {
    // 进入寻呼机模块：初始化 modem 码表与消息记录到 PSRAM（退出时由 ui_ofdm_menu_on_exit 释放）
    if (ofdm_tables_init(ofdm_alloc_psram) != 0) {
        printf("ofdm: tables init FAILED (oom)\n");
    }
    ofdm_msg_store_init();
    s_hub_sel = 0;
    memset(&s_hub_tap, 0, sizeof(s_hub_tap));
    ofdm_hub_draw(key_event, global_state);
}

void ui_ofdm_menu_on_exit(void) {
    ofdm_msg_store_free();
    ofdm_tables_free(ofdm_free_psram);
}

int32_t ui_ofdm_menu_event(Key_Event *key_event, Global_State *global_state) {
    Key_Event *ke = key_event;

    // 触屏：点击卡片进入功能；点击页眉右 1/4（“返回”）回到主菜单
    {
        int32_t tx = 0, ty = 0;
        if (ofdm_tap_feed(&s_hub_tap, ke, &tx, &ty)) {
            if (ofdm_back_hit(global_state, tx, ty)) {
                return STATE_MAIN_MENU; // 码表由 ui_app.c 在回主菜单时统一释放
            }
            for (int32_t i = 0; i < OFDM_HUB_ITEMS; i++) {
                int32_t x, y, w, h;
                ofdm_hub_card_rect(global_state, i, &x, &y, &w, &h);
                if (tx >= x && tx < x + w && ty >= y && ty < y + h) {
                    s_hub_sel = i;
                    ofdm_hub_draw(ke, global_state); // 先给出选中反馈，再进入功能
                    return ofdm_hub_activate(ke, global_state, i);
                }
            }
        }
    }

    // 16 键虚拟键盘的键码不作为键码采纳（本状态全交互走触屏控件；
    // 硬按键仅作兼容性兜底）
    if (ke->key_code != NANO_KEY_IDLE && ke->is_soft_key && !ke->is_softkbd) {
        return STATE_OFDM_MENU;
    }

    if (ke->key_edge == -1 || ke->key_edge == -2) {
        // D 执行当前卡片；A 返回主菜单
        if (ke->key_code == NANO_KEY_enter) {
            return ofdm_hub_activate(ke, global_state, s_hub_sel);
        }
        if (ke->key_code == NANO_KEY_esc) {
            return STATE_MAIN_MENU; // 码表由 ui_app.c 在回主菜单时统一释放
        }
        // 2×2 宫格导航：左右换列、上下换行（硬按键兜底）
        else if (ke->key_code == NANO_KEY_left || ke->key_code == NANO_KEY_right) {
            s_hub_sel ^= 1;
            ofdm_hub_draw(ke, global_state);
        }
        else if (ke->key_code == NANO_KEY_up || ke->key_code == NANO_KEY_down) {
            s_hub_sel ^= 2;
            ofdm_hub_draw(ke, global_state);
        }
        // 数字键直选：1~4 对应四张卡片（与卡片右下角序号一致）
        else if (ke->key_code >= NANO_KEY_1 && ke->key_code < NANO_KEY_1 + OFDM_HUB_ITEMS) {
            int32_t idx = (int32_t)(ke->key_code - NANO_KEY_1);
            s_hub_sel = idx;
            ofdm_hub_draw(ke, global_state);
            return ofdm_hub_activate(ke, global_state, idx);
        }
    }
    return STATE_OFDM_MENU;
}

// ============================================================================
// 四、STATE_OFDM_TX（文本输入）
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
// 五、STATE_OFDM_TXING（发射中）
// ============================================================================

// 计算当前帧的幅度包络（70 根柱子）。调用前提：s_frame_f32 已渲染好一帧。
// 仅用于界面可视化，按 6 点抽样取块内峰值，代价可忽略。
static void ofdm_tx_wave_update(void) {
    if (!s_frame_f32) return;
    const int32_t stride = OFDM_FRAME_LENGTH / OFDM_TX_BARS;
    float peak_all = 0.0f;
    for (int32_t b = 0; b < OFDM_TX_BARS; b++) {
        int32_t s0 = b * stride;
        int32_t s1 = s0 + stride;
        float peak = 0.0f;
        for (int32_t i = s0; i < s1; i += 6) {
            float v = fabsf(s_frame_f32[i]);
            if (v > peak) peak = v;
        }
        s_tx_bars[b] = peak;
        if (peak > peak_all) peak_all = peak;
    }
    if (peak_all > 1e-6f) {
        for (int32_t b = 0; b < OFDM_TX_BARS; b++) s_tx_bars[b] /= peak_all;
    }
}

// 释放发射机与音频输出（幂等）
static void ofdm_tx_stop(void) {
    if (s_audio_on) {
        audio_out_stop();
        audio_out_close();
        s_audio_on = 0;
    }
    s_tx_audio_ok = 0;
    if (s_tx) { ofdm_tx_destroy(s_tx); s_tx = NULL; }
    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++) {
        if (s_pcm[i]) { free(s_pcm[i]); s_pcm[i] = NULL; }
    }
    if (s_frame_f32) { free(s_frame_f32); s_frame_f32 = NULL; }
}

// 建立发射机：切分/编码净荷 → 分配帧缓冲 → 打开扬声器。返回 0=成功，<0=内存不足
static int32_t ofdm_tx_start(const uint8_t *payload, uint32_t len, Global_State *gs) {
    if (s_tx || s_audio_on) ofdm_tx_stop();
    s_tx = ofdm_tx_create(ofdm_alloc_psram, ofdm_free_psram, payload, len);
    s_tx_payload_bytes = len;
    s_tx_total_frames = ofdm_tx_frame_count(s_tx);
    for (int32_t i = 0; i < AUDIO_OUT_QUEUE_DEPTH; i++)
        s_pcm[i] = (int16_t *)platform_calloc(OFDM_FRAME_LENGTH, sizeof(int16_t));
    s_frame_f32 = (float *)platform_calloc(OFDM_FRAME_LENGTH, sizeof(float));
    memset(s_tx_bars, 0, sizeof(s_tx_bars));
    if (!s_tx || !s_pcm[0] || !s_pcm[1] || !s_frame_f32) {
        ofdm_tx_stop();
        return -1;
    }
    // 钳位到64，防止幅度过大引发失真
    s_tx_audio_ok = (audio_out_init(OFDM_SAMPLE_RATE,
                                    (uint8_t)((gs->volume > 64) ? 64 : gs->volume)) == 0) ? 1 : 0;
    s_audio_on = 1; // 无论成功与否都记录“已尝试初始化”，stop 时统一 close（HAL 幂等）
    s_tx_frames_played = 0;
    s_fill_buf = 0;
    s_tx_start_ts = gs->timestamp;
    s_tx_last_draw_ts = 0;
    memset(&s_tx_tap, 0, sizeof(s_tx_tap));
    if (!s_tx_audio_ok) printf("ofdm-tx: audio_out_init FAILED, UI-only demo mode\n");
    return 0;
}

// 渲染并投放一帧（不做回绕）。返回 0=已投放；1=本遍帧已尽；<0=发射机未就绪
// 调用前提：audio_out_queue_free() 为真（队列有空槽，对应缓冲已播完可覆写）。
static int32_t ofdm_tx_render_push(void) {
    if (!s_tx || !s_frame_f32) return -1;
    if (ofdm_tx_render_frame(s_tx, s_frame_f32) != 0) return 1;
    ofdm_tx_wave_update();
    int16_t *pcm = s_pcm[s_fill_buf];
    for (int32_t i = 0; i < OFDM_FRAME_LENGTH; i++) {
        float v = s_frame_f32[i] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        else if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }
    if (s_tx_audio_ok && audio_out_enqueue(pcm, OFDM_FRAME_LENGTH) == 0) {
        s_tx_frames_played++;
        s_fill_buf ^= 1; // 乒乓轮换
        printf("ofdm-tx: frame %lu enqueued\n", (unsigned long)s_tx_frames_played);
    }
    else if (s_tx_audio_ok) {
        printf("ofdm-tx: enqueue failed\n");
    }
    else {
        // 音频设备不可用：仍推进计数，使界面（周期进度/波形）保持可演示
        s_tx_frames_played++;
        s_fill_buf ^= 1;
    }
    return 0;
}

// 循环播放用：帧尽自动回绕
static void ui_ofdm_tx_fill_and_play(void) {
    if (ofdm_tx_render_push() == 1) {
        ofdm_tx_rewind(s_tx);
        ofdm_tx_render_push();
    }
}

// 发射状态页：状态卡（指示灯/帧号/周期进度）+ 实时帧波形 + 参数网格
static void ofdm_txing_draw(Global_State *gs) {
    OFDM_Palette p;
    ofdm_palette(gs, &p);
    int32_t W = (int32_t)gs->gfx->width;
    int32_t H = (int32_t)gs->gfx->height;
    int32_t cw = W - 20;
    int32_t x0 = 10;

    ofdm_fill(gs, 0, 0, W, H, p.bg);
    ofdm_chrome_header(gs, &p, (wchar_t *)L"发射中");

    // ---------------- 状态卡 ----------------
    int32_t cy = OFDM_HDR_H + 6; // 32
    int32_t ch = 64;             // 32..96
    ofdm_card(gs, x0, cy, cw, ch, 12, p.panel, p.edge);

    int32_t pulse = (int32_t)((gs->timestamp / 450) % 2); // 呼吸指示灯（约 0.45s 一拍）
    ofdm_dot(gs, x0 + 16, cy + 18, pulse ? 5 : 4, s_tx_audio_ok ? p.err : p.warn, pulse ? 0 : 1);
    ofdm_text(gs, OFDM_F_BIG, (wchar_t *)L"正在发射", x0 + 30, cy + 7, p.text);

    // 右侧：本轮第几帧 / 共几帧（时间驱动，播放头与进度连续）
    wchar_t buf[128];
    uint32_t total = (s_tx_total_frames > 0) ? s_tx_total_frames : 1;
    double cycle_ms = (double)total * (double)OFDM_FRAME_LENGTH / (double)OFDM_SAMPLE_RATE * 1000.0;
    double frac = 0.0;
    int32_t cur_frame = 1;
    if (cycle_ms > 1.0 && gs->timestamp > s_tx_start_ts) {
        double el = (double)(gs->timestamp - s_tx_start_ts);
        double cyc = fmod(el, cycle_ms);
        frac = cyc / cycle_ms;
        cur_frame = (int32_t)(frac * (double)total) + 1;
        if (cur_frame > (int32_t)total) cur_frame = (int32_t)total;
    }
    if (s_tx_audio_ok) {
        swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%d / %d 帧", (int)cur_frame, (int)total);
    }
    else {
        swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"音频不可用");
    }
    ofdm_text_right(gs, OFDM_F_TITLE, buf, x0 + cw - 14, cy + 12,
                    s_tx_audio_ok ? p.accent : p.err);

    swprintf(buf, sizeof(buf) / sizeof(wchar_t),
             L"已发 %d 帧 · 一轮 %d 帧 · 每轮 %.1f 秒",
             (int)s_tx_frames_played, (int)total, cycle_ms / 1000.0);
    ofdm_text(gs, OFDM_F_TITLE, buf, x0 + 30, cy + 29, p.dim);

    // 周期进度条
    ofdm_bar(gs, x0 + 14, cy + 48, cw - 28, 8, (float)frac, p.track, p.accent);

    // ---------------- 帧波形卡 ----------------
    int32_t wy = cy + ch + 6;  // 102
    int32_t wh = 66;           // 102..168
    ofdm_card(gs, x0, wy, cw, wh, 12, p.panel, p.edge);
    ofdm_text(gs, OFDM_F_TITLE, (wchar_t *)L"帧波形", x0 + 12, wy + 5, p.dim);
    ofdm_text_right(gs, OFDM_F_TITLE, (wchar_t *)L"2.5 – 5.5 kHz", x0 + cw - 12, wy + 5, p.dim);

    int32_t bx0 = x0 + 12;
    int32_t bx1 = x0 + cw - 12;
    int32_t bbase = wy + wh - 10;     // 基线
    int32_t bmaxh = 30;               // 最大柱高
    ofdm_fill(gs, bx0, bbase + 1, bx1 - bx0, 1, p.hairline); // 基线
    int32_t span = bx1 - bx0;
    for (int32_t b = 0; b < OFDM_TX_BARS; b++) {
        int32_t bx = bx0 + (int32_t)((float)b / (float)OFDM_TX_BARS * (float)span);
        int32_t bh = (int32_t)(s_tx_bars[b] * (float)bmaxh + 0.5f);
        if (bh < 1) bh = 1;
        ofdm_fill(gs, bx, bbase - bh, 3, bh, p.accent);
    }
    // 播放头（本帧内的相对位置）
    {
        double in_frame = 0.0;
        if (cycle_ms > 1.0 && total > 0) in_frame = frac * (double)total - (double)(cur_frame - 1);
        int32_t px = bx0 + (int32_t)((float)((cur_frame - 1) + in_frame) / (float)total * (float)span);
        if (px > bx1 - 1) px = bx1 - 1;
        ofdm_line(gs, px, wy + 22, px, bbase + 2, p.text);
    }

    // ---------------- 参数网格 ----------------
    int32_t gy = wy + wh + 6; // 174
    swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%d B", (int)s_tx_payload_bytes);
    ofdm_kv(gs, &p, 16, gy, 40, (wchar_t *)L"净荷", buf);
    swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%d", (int)s_tx_total_frames);
    ofdm_kv(gs, &p, 168, gy, 44, (wchar_t *)L"物理帧", buf);

    swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%d ms",
             (int)((double)OFDM_FRAME_LENGTH / (double)OFDM_SAMPLE_RATE * 1000.0));
    ofdm_kv(gs, &p, 16, gy + 15, 40, (wchar_t *)L"帧长", buf);
    swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%d Hz", (int)(OFDM_CARRIER_NUMBER * OFDM_BASE_FREQ));
    ofdm_kv(gs, &p, 168, gy + 15, 44, (wchar_t *)L"带宽", buf);

    swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"RS(%d,%d)×%d", 32, 16, OFDM_PKT_RS_BLOCKS);
    ofdm_kv(gs, &p, 16, gy + 30, 40, (wchar_t *)L"编码", buf);
    swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%d Hz", OFDM_CARRIER_FREQ);
    ofdm_kv(gs, &p, 168, gy + 30, 44, (wchar_t *)L"载波", buf);

    ofdm_chrome_footer(gs, &p,
                       s_tx_audio_ok ? (wchar_t *)L"请保持扬声器开启" : (wchar_t *)L"音频设备不可用 · 仅界面演示",
                       (wchar_t *)L"D / A 停止");
    gfx_refresh(gs->gfx);
}

void ui_ofdm_txing_on_enter(Key_Event *key_event, Global_State *global_state) {
    // 文本（wchar_t）→ UTF-8 字节流
    wchar_t *text = global_state->w_input_main->textarea.text;
    uint32_t wlen = wcslen(text);
    uint32_t cap = wlen * 4 + 1;
    char *utf8 = (char *)platform_calloc(cap, 1);
    uint32_t blen = _wcstombs(utf8, text, cap);
    int32_t rc = ofdm_tx_start((const uint8_t *)utf8, blen, global_state);
    free(utf8);

    if (rc != 0) {
        // 内存不足：清理并提示，由 event 函数兜底返回输入态
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main,
                               (wchar_t *)L"内存不足，无法启动发射", 0, 0);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        return;
    }

    // 预填队列全部空槽（一块在播、一块待播），立即开始无缝循环播放
    ui_ofdm_tx_fill_and_play();
    ui_ofdm_tx_fill_and_play();

    ofdm_txing_draw(global_state);
}

int32_t ui_ofdm_txing_event(Key_Event *key_event, Global_State *global_state) {
    // 手动停止发射：短按 D 或 A；触屏点击页眉右 1/4（“返回”）同效
    int32_t stop = 0;
    if (key_event->key_edge < 0 &&
        (key_event->key_code == NANO_KEY_enter || key_event->key_code == NANO_KEY_esc)) {
        stop = 1;
    }
    else {
        int32_t tx = 0, ty = 0;
        if (ofdm_tap_feed(&s_tx_tap, key_event, &tx, &ty) && ofdm_back_hit(global_state, tx, ty)) stop = 1;
    }
    if (stop) {
        ofdm_tx_stop();
        return STATE_OFDM_TX;
    }
    // 16 键虚拟键盘的键码不作为键码采纳（本状态全交互走触屏控件）
    if (key_event->key_code != NANO_KEY_IDLE && key_event->is_soft_key && !key_event->is_softkbd) {
        return STATE_OFDM_TXING;
    }

    if (!s_tx) return STATE_OFDM_TX; // 异常兜底（如 on_enter 内存不足）

    // 背压安全的喂入：队列有空槽才填充（单通道双槽 ≈ 1.3s 音频余量，段尾无缝切换；
    // UI 刷帧不会造成饥饿；即使偶发间隙，接收端按帧独立重同步，协议天然容忍）
    if (s_tx_audio_ok && audio_out_queue_free()) ui_ofdm_tx_fill_and_play();
    else if (!s_tx_audio_ok) {
        // 音频不可用：按帧长节拍推进，界面仍可完整演示
        uint32_t total = (s_tx_total_frames > 0) ? s_tx_total_frames : 1;
        double cycle_ms = (double)total * (double)OFDM_FRAME_LENGTH / (double)OFDM_SAMPLE_RATE * 1000.0;
        uint32_t want = (uint32_t)((double)(global_state->timestamp - s_tx_start_ts) /
                                   (cycle_ms / (double)total));
        if (want > s_tx_frames_played) {
            if (want - s_tx_frames_played > 4) s_tx_frames_played = want - 1; // 长暂停后只补最近一帧
            while (s_tx_frames_played < want) ui_ofdm_tx_fill_and_play();
        }
    }

    // 限频刷新整页（波形/进度连续感；tty 每帧全屏推送，10Hz 足够顺滑）
    if (global_state->timestamp - s_tx_last_draw_ts >= OFDM_STA_REFRESH_MS) {
        s_tx_last_draw_ts = global_state->timestamp;
        ofdm_txing_draw(global_state);
    }
    return STATE_OFDM_TXING;
}

// ============================================================================
// 六、STATE_OFDM_RX（接收中）
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

// Unicode 解码字符统一出口：双写——屏幕接收文本缓冲（限频重绘）与串口行缓冲。
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

// 解出 packet 净荷回调：接收页 → UTF-8 拼接解码进屏幕缓冲；消息页 → 进消息记录
static void ui_ofdm_rx_on_text(const uint8_t *bytes, uint32_t len, void *user) {
    (void)user;
    if (s_rx_sink == OFDM_RX_SINK_MSG) {
        ofdm_msg_on_rx_bytes(bytes, len);
        return;
    }
    ui_ofdm_utf8_stream_out(bytes, len);
}

// 接收机事件日志回调：串口输出（屏幕内容以解调文本为主，日志不混入正文）
static void ui_ofdm_rx_on_log(const char *msg, void *user) {
    (void)user;
    if (!msg) return;
    printf("ofdm-rx log: %s\n", msg);
}

// 结束接收会话（幂等）：停采集任务 → 关麦恢复扬声器 → 释放缓冲
static void ofdm_rx_stop(void) {
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
    s_mic_ok = 0;
#if defined(NANO_WASM_GUI)
    s_mic_poll = 0;
#endif
    s_mic_task_stop = 0;
}

// 启动接收会话：分配采集/解调缓冲 → 接管麦克风（会关闭扬声器）。
// sink 决定解出的净荷去向（接收页文本缓冲 / 消息页记录）。返回 0=成功，<0=内存不足
static int32_t ofdm_rx_start(Global_State *gs, int32_t sink) {
    s_rx_sink = sink;
    s_mic_i16 = (int16_t *)platform_calloc(OFDM_RX_CHUNK, sizeof(int16_t));
    s_mic_f32 = (float *)platform_calloc(OFDM_RX_CHUNK, sizeof(float));
    s_dbg_hist = (float *)platform_calloc(OFDM_DBG_HIST_LEN, sizeof(float));
    s_rx_ring = (int16_t *)platform_calloc(OFDM_RX_RING_CAP, sizeof(int16_t));
    s_mic_task_buf = (int16_t *)platform_calloc(OFDM_RX_CHUNK, sizeof(int16_t));
    if (sink == OFDM_RX_SINK_SCREEN) {
        s_rx_bytes = (uint8_t *)platform_calloc(OFDM_RX_BYTES_CAP, 1);
        s_rx_wcbuf = (wchar_t *)platform_calloc(OFDM_RX_BYTES_CAP + 1, sizeof(wchar_t));
    }
    s_ring_w = 0; s_ring_r = 0; s_ring_ov = 0;
    s_rx_bytes_len = 0;
    s_rx_new_text = 0;
    s_utf8_pend_len = 0;
    s_utf8_line_len = 0;
    s_mic_err_cnt = 0;
    s_mic_rms_ema = 0.0f;
    s_mic_mean_ema = 0.0f;
    s_mic_peak = 0;
    s_mic_ok = 0;
    s_rx_last_dbg_ts = gs->timestamp;

    s_rx = ofdm_rx_create(ofdm_alloc_psram, ofdm_free_psram, ui_ofdm_rx_on_text, ui_ofdm_rx_on_log, NULL);

    if (!s_rx || !s_mic_i16 || !s_mic_f32 || !s_dbg_hist || !s_rx_ring || !s_mic_task_buf ||
        (sink == OFDM_RX_SINK_SCREEN && (!s_rx_bytes || !s_rx_wcbuf))) {
        printf("ofdm-rx: OOM, rx aborted\n");
        ofdm_rx_stop();
        return -1;
    }

    // 接管麦克风（48kHz；会关闭扬声器，退出时 mic_close 恢复）
    s_mic_ok = (mic_init(OFDM_SAMPLE_RATE, (uint8_t)gs->volume) == 0) ? 1 : 0;
    if (!s_mic_ok) {
        printf("ofdm-rx: mic_init(48000) FAILED\n");
        return 0;
    }
    // 丢弃唤醒期采样（SPM1423HM4H-B 时钟起振后 10ms 才达规格，取 2 块 ≈40ms 裕量）
    mic_read(s_mic_i16, OFDM_RX_CHUNK);
    mic_read(s_mic_i16, OFDM_RX_CHUNK);
    // 启动采集任务（Core1 高优先级，与渲染任务解耦；此后渲染任务仅从环取数）
    s_mic_task_stop = 0;
    if (platform_task_create(ui_ofdm_mic_task, "ofdm_mic", 3072, NULL, 3, 1, &s_mic_task) != 0) {
        s_mic_task = NULL;
        s_mic_ok = 0;
        printf("ofdm-rx: mic task create FAILED\n");
    }
#if defined(NANO_WASM_GUI)
    // 【WASM 专属】上面的任务创建必然失败（单线程无 pthread）：撤销 s_mic_ok=0，
    // 改为在渲染循环内轮询采集（见 ui_ofdm_mic_poll 与 ofdm_rx_poll）。
    if (!s_mic_task) {
        s_mic_ok = 1;
        s_mic_poll = 1;
        printf("ofdm-rx: (WASM) no RTOS task, polling mic in render loop\n");
    }
#endif
    printf("ofdm-rx: mic_init(48000) ok, rx ready\n");
    printf("ofdm-rx log: 接收已启动（48kHz），等待信号…\n");
    return 0;
}

// 接收会话每帧驱动：补采（WASM）→ 环取数 → 喂解调 → 电平/诊断统计 → 串口诊断
// 不涉及任何界面绘制，接收页与消息页各自调用后自行重绘。
static void ofdm_rx_poll(Global_State *gs) {
    if (!s_rx) return;
#if defined(NANO_WASM_GUI)
    // 【WASM 专属】先补采一块进环，再走下面完全未改动的消费/解调逻辑
    if (s_mic_poll && s_mic_ok) ui_ofdm_mic_poll();
#endif
    // 从采集环取一块（≤960）→ int16 转 float → 流式喂入解调。
    // 采集由专用任务实时完成（见 ui_ofdm_mic_task），此处仅消费环中已有数据，不阻塞；
    // 环中暂无数据时直接跳过，事件循环以全速空转（vTaskDelay(0) 让出），UI 保持响应。
    if (s_mic_ok && s_rx_ring) {
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
    if (gs->timestamp - s_rx_last_dbg_ts >= 1500) {
        s_rx_last_dbg_ts = gs->timestamp;
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
}

// 接收页退出：结束会话并复位嵌入文本框为标准几何（不影响其它界面的同一单例控件）
static void ui_ofdm_rx_stop_session(void) {
    ofdm_rx_stop();
    if (s_rx_ui_init && s_rx_gs && s_rx_gs->w_textarea_main) {
        ofdm_text_card_restore(s_rx_gs, s_rx_gs->w_textarea_main);
    }
    s_rx_ui_init = 0;
    s_rx_gs = NULL;
}

// 接收状态页：页眉 + 状态卡（锁定/电平/星座/帧统计）+ 滚动文本区 + 页脚
static void ofdm_rx_draw(Key_Event *ke, Global_State *gs) {
    OFDM_Palette p;
    ofdm_palette(gs, &p);
    int32_t W = (int32_t)gs->gfx->width;
    int32_t H = (int32_t)gs->gfx->height;
    int32_t x0 = 10, cw = W - 20;

    OFDM_RX_Stat st;
    memset(&st, 0, sizeof(st));
    if (s_rx) ofdm_rx_get_stat(s_rx, &st);

    ofdm_fill(gs, 0, 0, W, H, p.bg);
    ofdm_chrome_header(gs, &p, (wchar_t *)L"接收");

    // ---------------- 状态卡 ----------------
    int32_t cy = OFDM_HDR_H + 6; // 32
    int32_t ch = 64;             // 32..96
    ofdm_card(gs, x0, cy, cw, ch, 12, p.panel, p.edge);

    // 星座图（右上）：均衡后 IQ 散点，锁定时可见 4-QAM 四点簇
    int32_t box = 48;
    int32_t bx = x0 + cw - box - 10;
    int32_t by = cy + 8;
    ofdm_round(gs, bx, by, box, box, 6, p.bg);
    ofdm_line(gs, bx + box / 2, by + 3, bx + box / 2, by + box - 4, p.hairline);
    ofdm_line(gs, bx + 3, by + box / 2, bx + box - 4, by + box / 2, p.hairline);

    const float *wave = NULL, *viz_i = NULL, *viz_q = NULL;
    uint32_t viz_seq = 0;
    if (s_rx && ofdm_rx_get_viz(s_rx, &wave, &viz_i, &viz_q, &viz_seq) == 0 &&
        viz_i && viz_q) {
        const float scale = (float)(box / 2 - 4) / 1.35f;
        int32_t hx = bx + box / 2, hy = by + box / 2;
        for (int32_t i = 0; i < OFDM_CARRIER_NUMBER; i++) {
            float ii = viz_i[i], qq = viz_q[i];
            if (ii > 1.35f) ii = 1.35f; else if (ii < -1.35f) ii = -1.35f;
            if (qq > 1.35f) qq = 1.35f; else if (qq < -1.35f) qq = -1.35f;
            int32_t dx = hx + (int32_t)(ii * scale);
            int32_t dy = hy - (int32_t)(qq * scale);
            ofdm_fill(gs, dx, dy, 2, 2, st.locked ? p.ok : p.dim);
        }
    }

    // 状态行：指示灯 + 大字状态 + 右侧电平 dB
    float rms_db = (s_mic_rms_ema > 1.0f) ? 20.0f * log10f(s_mic_rms_ema / 32768.0f) : -99.0f;
    OFDM_RGB state_color = p.warn;
    wchar_t *state_text = (wchar_t *)L"搜索信号";
    if (!s_mic_ok) {
        state_color = p.err;
        state_text = (wchar_t *)L"麦克风不可用";
    }
    else if (st.locked) {
        state_color = p.ok;
        state_text = (wchar_t *)L"已锁定";
    }
    int32_t pulse = (int32_t)((gs->timestamp / 450) % 2);
    ofdm_dot(gs, x0 + 16, cy + 18, (st.locked || !s_mic_ok) ? 4 : (pulse ? 5 : 4), state_color,
             (!st.locked && s_mic_ok && !pulse) ? 1 : 0);
    ofdm_text(gs, OFDM_F_BIG, state_text, x0 + 30, cy + 7, state_color);

    // 右侧：电平 dB
    wchar_t buf[160];
    if (rms_db <= -90.0f) swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"— dB");
    else swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%.0f dB", (double)rms_db);
    ofdm_text_right(gs, OFDM_F_TITLE, buf, bx - 10, cy + 9, (rms_db > -45.0f) ? state_color : p.dim);

    // 副行：帧统计
    swprintf(buf, sizeof(buf) / sizeof(wchar_t),
             L"帧 %d · 坏帧 %d · RS 纠错 %d",
             (int)st.frames_ok, (int)st.frames_bad, (int)st.rs_corrected);
    ofdm_text(gs, OFDM_F_TITLE, buf, x0 + 30, cy + 29, p.dim);

    // 电平条（-60dBFS ~ 0dBFS）；颜色随电平分档
    float level = (rms_db + 60.0f) / 60.0f;
    if (level < 0.0f) level = 0.0f;
    if (level > 1.0f) level = 1.0f;
    OFDM_RGB level_color;
    if (!s_mic_ok) level_color = p.err;
    else if (st.locked) level_color = p.ok;                                  // 已锁定：信号可用
    else level_color = (rms_db > -45.0f) ? p.warn : p.dim;                    // 搜索中：按电平提示
    int32_t mw = (bx - 10) - (x0 + 14);
    ofdm_bar(gs, x0 + 14, cy + 48, mw, 8, level, p.track, level_color);
    // 刻度：-45 / -30 / -15 dBFS（-60 与 0 dBFS 为条两端）
    for (int32_t t = 1; t < 4; t++) {
        int32_t tx = x0 + 14 + mw * t / 4;
        ofdm_line(gs, tx, cy + 58, tx, cy + 60, p.hairline);
    }

    // ---------------- 接收内容 ----------------
    swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%d 字节", (int)s_rx_bytes_len);
    ofdm_text_card_draw(gs, &p, gs->w_textarea_main, (wchar_t *)L"接收内容", buf);
    if (wcslen(gs->w_textarea_main->text) > 0) {
        ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);
        // 排版已完成：清掉 modified 标记，后续状态帧（120ms）不再重排全文，
        // 只重绘已有排版结果（文本变化时由 ui_widget_textarea_set 重新置位）
        gs->w_textarea_main->is_modified = 0;
    }
    else if (!s_mic_ok) {
        ofdm_text_card_placeholder(gs, &p, gs->w_textarea_main,
                                   (wchar_t *)L"麦克风初始化失败", (wchar_t *)L"检查音频设备后重试");
    }
    else {
        ofdm_text_card_placeholder(gs, &p, gs->w_textarea_main,
                                   (wchar_t *)L"等待信号…", (wchar_t *)L"将设备靠近声源，保持安静");
    }

    // ---------------- 页脚 ----------------
    if (s_rx_follow) {
        ofdm_chrome_footer(gs, &p, (wchar_t *)L"←→ 手动滚动 · D 回到底部", (wchar_t *)L"A 退出");
    }
    else {
        ofdm_chrome_footer(gs, &p, (wchar_t *)L"手动滚动中 · D 回到底部", (wchar_t *)L"A 退出");
    }
    gfx_refresh(gs->gfx);
}

void ui_ofdm_rx_on_enter(Key_Event *key_event, Global_State *global_state) {
    s_rx_last_txt_ts = 0;
    s_rx_last_sta_ts = 0;
    s_rx_follow = 1;
    s_mic_err_cnt = 0;
    s_mic_rms_ema = 0.0f;
    s_mic_mean_ema = 0.0f;
    s_mic_peak = 0;
    s_rx_gs = global_state;
    memset(&s_rx_tap, 0, sizeof(s_rx_tap));

    // 文本区改为嵌入状态页下方的卡片内（裸模式：不触碰页眉、不自行推帧）
    ofdm_text_card_setup(global_state, global_state->w_textarea_main,
                         102, (int32_t)global_state->gfx->height - OFDM_FTR_H - 12);
    s_rx_ui_init = 1;
    ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, (wchar_t *)L"", -1, 1);

    if (ofdm_rx_start(global_state, OFDM_RX_SINK_SCREEN) != 0) {
        ui_ofdm_rx_stop_session();
        ofdm_error_page(key_event, global_state, (wchar_t *)L"接收",
                        (wchar_t *)L"内存不足", (wchar_t *)L"无法启动接收机，按 A 返回");
        return;
    }
    ofdm_rx_draw(key_event, global_state);
}

int32_t ui_ofdm_rx_event(Key_Event *key_event, Global_State *global_state) {
    // 异常兜底（如 on_enter 内存不足）：直接回模式菜单
    if (!s_rx) return STATE_OFDM_MENU;

    // 短按 A 退出（关闭麦克风并恢复扬声器）；触屏点击页眉右 1/4（“返回”）同效
    int32_t quit = 0;
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_esc) quit = 1;
    else {
        int32_t tx = 0, ty = 0;
        if (ofdm_tap_feed(&s_rx_tap, key_event, &tx, &ty) && ofdm_back_hit(global_state, tx, ty)) quit = 1;
    }
    if (quit) {
        ui_ofdm_rx_stop_session();
        return STATE_OFDM_MENU;
    }
    // 16 键虚拟键盘的键码不作为键码采纳（本状态全交互走触屏控件）
    if (key_event->key_code != NANO_KEY_IDLE && key_event->is_soft_key && !key_event->is_softkbd) {
        return STATE_OFDM_RX;
    }

    Widget_Textarea_State *ts = global_state->w_textarea_main;

    // 左右键：手动滚行（暂停自动跟随）；短按 D：回到底部并恢复自动跟随
    if (key_event->key_edge < 0 &&
        (key_event->key_code == NANO_KEY_left || key_event->key_code == NANO_KEY_right)) {
        s_rx_follow = 0;
        ui_widget_textarea_event_handler(key_event, global_state, ts, STATE_OFDM_RX, STATE_OFDM_RX);
        ofdm_rx_draw(key_event, global_state);
    }
    else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
        s_rx_follow = 1;
        ui_widget_textarea_set(key_event, global_state, ts, ts->text, -1, 1);
        ofdm_rx_draw(key_event, global_state);
    }
    // 短按 C：串口转储最近采样（CSV，int16 域），供宿主机离线分析
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

    // 触屏手势：文本区内拖动/惯性滚动（裸模式下只绘制文本区，最终整屏重推一次）
    if (key_event->touch_edge || key_event->is_touching) {
        if (ui_widget_textarea_touch_handler(key_event, global_state, ts) == 1) {
            s_rx_follow = 0;
            ofdm_rx_draw(key_event, global_state);
        }
    }

    // 接收通路（补采/喂解调/电平/串口诊断）：与消息页共用同一实现
    ofdm_rx_poll(global_state);

    // 文本更新（限频）：跟随模式滚到最底部；手动滚动模式保持视口
    int32_t need_draw = 0;
    if (s_rx_new_text && global_state->timestamp - s_rx_last_txt_ts >= OFDM_TXT_REFRESH_MS) {
        s_rx_last_txt_ts = global_state->timestamp;
        s_rx_new_text = 0;
        need_draw = 1;
        _mbstowcs(s_rx_wcbuf, (const char *)s_rx_bytes, s_rx_bytes_len);
        // 文本超过控件上限时显示尾部（最新内容）
        uint32_t wlen = wcslen(s_rx_wcbuf);
        wchar_t *view = s_rx_wcbuf;
        if (wlen > UI_STR_BUF_MAX_LENGTH - 64) view = s_rx_wcbuf + (wlen - (UI_STR_BUF_MAX_LENGTH - 64));
        int32_t line = -1; // 跟随模式：自动滚动到最底部
        if (!s_rx_follow) {
            line = ts->current_line; // 手动滚动模式：保持视口
            int32_t max_line = ts->line_num - (ts->height / gfx_font_line_height(global_state->ui_font));
            if (line > max_line) line = (max_line > 0) ? max_line : 0;
            if (line < 0) line = 0;
        }
        ui_widget_textarea_set(key_event, global_state, ts, view, line, 1);
    }

    // 状态卡（锁定/电平/星座/统计）限频刷新；文本更新帧顺带整屏重绘（单帧内至多一次）
    if (need_draw || global_state->timestamp - s_rx_last_sta_ts >= OFDM_STA_REFRESH_MS) {
        s_rx_last_sta_ts = global_state->timestamp;
        ofdm_rx_draw(key_event, global_state);
    }
    return STATE_OFDM_RX;
}

// ============================================================================
// 七、STATE_OFDM_MSG / STATE_OFDM_MSG_INPUT（消息收发：一屏完成收发）
// ----------------------------------------------------------------------------
// 版式（自上而下）：顶栏 → 消息记录（对话式，含身份标签/正文/间隔）→ 接收/发射
// 两个按钮 → 状态栏 → 屏幕最底部的发射进度条。
//   接收：按下「接收」接管麦克风，状态栏显示锁定/电平/帧数；记录区实时流式追加
//         收到的文字，尚未收到时显示暗色「正在接收…」占位；
//         同一条报文被对端重复发送（本机发射固定发 3 遍）时按“载荷段完全一致”
//         去重，只在记录里留一条。
//   发射：按下「发射」打开文本输入控件；提交后报文进入记录（暗色=发射中），
//         连发 3 遍，状态栏显示“第 n/3 遍 · 帧 a/b”，最底部进度条同步；
//         发完转亮色。
// 交互：全部走触屏控件（按钮/记录拖动滚动/页眉返回），硬按键仅作兼容性兜底
//       （1=接收 2=发射 D=发射/停止 A=返回）；触屏不派生任何键码。
// ============================================================================

#define OFDM_MSG_SLOTS      (6)      // 记录条数上限（超出丢最旧）
#define OFDM_MSG_CHAR_CAP   (768)    // 每条记录字符上限
#define OFDM_MSG_RAWCAP     (OFDM_RX_BYTES_CAP) // 一次接收会话的原始字节上限
#define OFDM_MSG_LOGLINES   (1024)   // 折行缓存容量
#define OFDM_MSG_LINE_ADV   (14)     // 记录区行距（px）
#define OFDM_MSG_PAD_X      (10)     // 记录区左右留白
#define OFDM_MSG_SB_W       (4)      // 记录区滚动条宽度
#define OFDM_MSG_LOG_TOP    (OFDM_HDR_H + 4)      // 记录区上沿
#define OFDM_MSG_BTN_H      (28)                  // 按钮高度
#define OFDM_MSG_BTN_Y      (184)                 // 按钮上沿
#define OFDM_MSG_STATUS_H   (18)                  // 状态栏高度
#define OFDM_MSG_STATUS_Y   (218)                 // 状态栏上沿
#define OFDM_MSG_PROG_H     (4)                   // 发射进度条高度（屏幕最底部）
#define OFDM_MSG_PROG_Y     (236)                 // 进度条上沿
#define OFDM_MSG_LOG_BOT    (OFDM_MSG_BTN_Y - 6)  // 记录区下沿
#define OFDM_MSG_TX_PASSES  (3)                   // 发射遍数（需求：发射三遍）
#define OFDM_MSG_RX_END_MS  (2000)                // 静默该时长视为一条消息收完

// 记录方向与状态
#define OFDM_MSG_RX   (0)
#define OFDM_MSG_TX   (1)
#define OFDM_MSG_DONE (0) // 已完成（正文亮色）
#define OFDM_MSG_LIVE (1) // 进行中（正文暗色）

typedef struct {
    int32_t dir;        // OFDM_MSG_RX / OFDM_MSG_TX
    int32_t state;      // OFDM_MSG_DONE / OFDM_MSG_LIVE
    uint32_t len;       // 字符数
    wchar_t *text;      // 槽位常驻缓冲（OFDM_MSG_CHAR_CAP）
} OFDM_Msg;

typedef struct {
    int32_t msg;        // 逻辑记录下标（0=最旧）
    int32_t off;        // 正文字符偏移
    int32_t len;        // 本行字符数（标签行/空行为 0）
    int32_t kind;       // 0=身份标签 1=正文 2=空行
} OFDM_LogLine;

static OFDM_Msg s_msgs[OFDM_MSG_SLOTS];     // 环形记录表
static int32_t s_msg_head = 0;              // 最旧记录所在槽位
static int32_t s_msg_count = 0;
static OFDM_LogLine *s_loglines = NULL;     // 折行缓存（PSRAM）
static int32_t s_logline_num = 0;
static int32_t s_log_dirty = 1;             // 需要重建折行缓存
static int32_t s_msg_scroll = 0;            // 记录区滚动量（px，0=贴最新）
static int32_t s_msg_auto = 1;              // 自动跟随最新
static OFDM_Tap s_msg_tap;                  // 点按判定（按钮/页眉返回）
static int32_t s_msg_drag = 0;              // 记录区拖动中
static int32_t s_msg_drag_y0 = 0;
static int32_t s_msg_drag_scroll0 = 0;
static uint64_t s_msg_last_draw_ts = 0;
static Global_State *s_msg_gs = NULL;

// ---- 接收会话（消息页） ----
static int32_t s_msg_rx_on = 0;
static uint8_t *s_msg_raw = NULL;      // 本会话累积的原始字节（已按下述规则去重）
static uint32_t s_msg_raw_len = 0;
static uint32_t s_msg_seg_start = 0;   // 当前载荷段在 raw 中的起点
static uint32_t s_msg_prev_start = 0;  // 上一保留段（用于“重复段”判定）
static uint32_t s_msg_prev_len = 0;
static int32_t s_msg_cur = -1;         // 当前接收记录槽位（-1=尚未收到内容）
static int32_t s_msg_rx_dirty = 0;     // raw 有更新，需要重建记录文本
static uint64_t s_msg_last_pkt_ts = 0;

// ---- 发射会话 ----
static int32_t s_msg_tx_on = 0;
static uint32_t s_msg_tx_pass = 0;     // 已完成的遍数
static uint32_t s_msg_tx_frame = 0;    // 当前遍内已投放帧数
static int32_t s_msg_tx_drain = 0;     // 末帧已入队，等待播放结束
static uint64_t s_msg_tx_last_enq_ts = 0;
static uint64_t s_msg_tx_start_ts = 0;
static int32_t s_msg_tx_entry = -1;    // 发射记录槽位

// ---- 输入页提交的待发射净荷（UTF-8） ----
static char *s_msg_payload = NULL;
static uint32_t s_msg_payload_len = 0;
static int32_t s_msg_payload_ready = 0;

// 会话收尾（定义见本节末）
static void ofdm_msg_stop_rx(void);
static void ofdm_msg_stop_tx(void);

static OFDM_Msg *ofdm_msg_slot(int32_t i) {
    return &s_msgs[(s_msg_head + i) % OFDM_MSG_SLOTS];
}

static void ofdm_msg_store_init(void) {
    int32_t i;
    for (i = 0; i < OFDM_MSG_SLOTS; i++) {
        if (!s_msgs[i].text) {
            s_msgs[i].text = (wchar_t *)platform_calloc(OFDM_MSG_CHAR_CAP, sizeof(wchar_t));
            if (!s_msgs[i].text) break;
            s_msgs[i].len = 0;
        }
    }
    if (!s_loglines) s_loglines = (OFDM_LogLine *)platform_calloc(OFDM_MSG_LOGLINES, sizeof(OFDM_LogLine));
    if (!s_msg_raw) s_msg_raw = (uint8_t *)platform_calloc(OFDM_MSG_RAWCAP, 1);
    s_log_dirty = 1;
}

static void ofdm_msg_store_free(void) {
    int32_t i;
    if (s_msg_tx_on) ofdm_msg_stop_tx();
    if (s_msg_rx_on) ofdm_msg_stop_rx();
    for (i = 0; i < OFDM_MSG_SLOTS; i++) {
        if (s_msgs[i].text) { free(s_msgs[i].text); s_msgs[i].text = NULL; }
        s_msgs[i].len = 0;
    }
    if (s_loglines) { free(s_loglines); s_loglines = NULL; }
    if (s_msg_raw) { free(s_msg_raw); s_msg_raw = NULL; }
    if (s_msg_payload) { free(s_msg_payload); s_msg_payload = NULL; }
    s_msg_payload_len = 0;
    s_msg_payload_ready = 0;
    s_msg_head = 0;
    s_msg_count = 0;
    s_msg_cur = -1;
    s_msg_tx_entry = -1;
    s_logline_num = 0;
    s_log_dirty = 1;
    s_msg_rx_on = 0;
    s_msg_tx_on = 0;
}

// 新建一条记录（环形，满则丢最旧）；返回槽位，失败 -1
static int32_t ofdm_msg_new(int32_t dir, int32_t state) {
    int32_t i, slot;
    for (i = 0; i < OFDM_MSG_SLOTS; i++) if (!s_msgs[i].text) return -1; // 未初始化
    if (s_msg_count >= OFDM_MSG_SLOTS) {
        slot = s_msg_head;
        if (slot == s_msg_cur) s_msg_cur = -1;
        if (slot == s_msg_tx_entry) s_msg_tx_entry = -1;
        s_msg_head = (s_msg_head + 1) % OFDM_MSG_SLOTS;
        s_msg_count--;
    }
    slot = (s_msg_head + s_msg_count) % OFDM_MSG_SLOTS;
    s_msgs[slot].dir = dir;
    s_msgs[slot].state = state;
    s_msgs[slot].len = 0;
    s_msgs[slot].text[0] = L'\0';
    s_msg_count++;
    s_log_dirty = 1;
    return slot;
}

// UTF-8 → wchar（尽力而为：非法/不完整序列跳过，尾部残序列留待下一包）
static uint32_t ofdm_msg_utf8_to_wcs(wchar_t *dst, uint32_t cap, const uint8_t *src, uint32_t n) {
    uint32_t p = 0, o = 0;
    while (p < n && o + 1 < cap) {
        uint8_t b = src[p];
        uint32_t need, cp, k;
        if ((b & 0x80) == 0) { need = 1; cp = b; }
        else if ((b & 0xE0) == 0xC0) { need = 2; cp = b & 0x1F; }
        else if ((b & 0xF0) == 0xE0) { need = 3; cp = b & 0x0F; }
        else if ((b & 0xF8) == 0xF0) { need = 4; cp = b & 0x07; }
        else { p++; continue; }                    // 非法前导字节：跳 1 字节重同步
        if (p + need > n) break;                   // 尾部不完整：等下一包
        int32_t ok = 1;
        for (k = 1; k < need; k++) {
            if ((src[p + k] & 0xC0) != 0x80) { ok = 0; break; }
            cp = (cp << 6) | (src[p + k] & 0x3F);
        }
        if (!ok) { p++; continue; }                // 损坏字节：跳 1 字节重同步
        p += need;
        if (cp >= 0xD800 && cp <= 0xDFFF) continue; // 代理区非法
        if (cp > 0x10FFFF) continue;
        if (cp > 0xFFFF && sizeof(wchar_t) < 4) { dst[o++] = L'?'; continue; }
        dst[o++] = (wchar_t)cp;
    }
    dst[o] = L'\0';
    return o;
}

// 用原始字节重建某条记录的正文
static void ofdm_msg_set_text_from_raw(int32_t slot, const uint8_t *raw, uint32_t n) {
    if (slot < 0 || slot >= OFDM_MSG_SLOTS || !s_msgs[slot].text) return;
    s_msgs[slot].len = ofdm_msg_utf8_to_wcs(s_msgs[slot].text, OFDM_MSG_CHAR_CAP, raw, n);
    s_log_dirty = 1;
}

// ---------------- 折行缓存 ----------------

static void ofdm_msg_log_rebuild(Global_State *gs) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t wrap = W - 2 * OFDM_MSG_PAD_X - OFDM_MSG_SB_W - 6;
    int32_t i;
    s_logline_num = 0;
    if (!s_loglines) { s_log_dirty = 0; return; }
    for (i = 0; i < s_msg_count; i++) {
        OFDM_Msg *m = ofdm_msg_slot(i);
        int32_t p = 0;
        if (i > 0 && s_logline_num < OFDM_MSG_LOGLINES) {
            s_loglines[s_logline_num].msg = -1;
            s_loglines[s_logline_num].kind = 2; // 记录间隔空行
            s_logline_num++;
        }
        // 身份标签行
        if (s_logline_num < OFDM_MSG_LOGLINES) {
            OFDM_LogLine *ln = &s_loglines[s_logline_num++];
            ln->msg = i; ln->off = 0; ln->len = 0; ln->kind = 0;
        }
        // 正文按像素宽度折行（尊重硬换行）
        while (p < (int32_t)m->len && s_logline_num < OFDM_MSG_LOGLINES) {
            int32_t st = p, wpx = 0;
            while (p < (int32_t)m->len) {
                wchar_t ch = m->text[p];
                if (ch == L'\n') { p++; break; }
                int32_t cw = gfx_font_char_advance(OFDM_F_TITLE, (uint32_t)ch);
                if (wpx + cw > wrap) break;
                wpx += cw;
                p++;
            }
            OFDM_LogLine *ln = &s_loglines[s_logline_num++];
            ln->msg = i; ln->off = st; ln->len = p - st; ln->kind = 1;
            if (p == st) p++; // 防死循环（单字超宽）
        }
    }
    s_log_dirty = 0;
}

// ---------------- 绘制 ----------------

static void ofdm_msg_buttons_rect(Global_State *gs, int32_t idx, int32_t *x, int32_t *y, int32_t *w, int32_t *h) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t gap = 8;
    *w = (W - 20 - gap) / 2;
    *h = OFDM_MSG_BTN_H;
    *x = 10 + idx * (*w + gap);
    *y = OFDM_MSG_BTN_Y;
}

static void ofdm_msg_buttons_draw(Global_State *gs, const OFDM_Palette *p) {
    int32_t i;
    for (i = 0; i < 2; i++) {
        int32_t x, y, w, h;
        ofdm_msg_buttons_rect(gs, i, &x, &y, &w, &h);
        if (i == 0) {
            // 接收 / 停止接收
            if (s_msg_rx_on) {
                ofdm_round(gs, x, y, w, h, h / 2, p->ok);
                ofdm_text_center(gs, OFDM_F_BIG, (wchar_t *)L"停止接收", x + w / 2, y + h / 2, p->on_accent);
            }
            else {
                ofdm_card(gs, x, y, w, h, h / 2, p->panel, p->edge);
                ofdm_dot(gs, x + 18, y + h / 2, 3, p->ok, 0);
                ofdm_text_center(gs, OFDM_F_BIG, (wchar_t *)L"接收", x + w / 2 + 6, y + h / 2, p->ok);
            }
        }
        else {
            // 发射 / 发射中（进行中不可再点）
            if (s_msg_tx_on) {
                ofdm_card(gs, x, y, w, h, h / 2, p->track, p->edge);
                ofdm_text_center(gs, OFDM_F_BIG, (wchar_t *)L"发射中…", x + w / 2, y + h / 2, p->dim);
            }
            else {
                ofdm_round(gs, x, y, w, h, h / 2, p->accent);
                ofdm_text_center(gs, OFDM_F_BIG, (wchar_t *)L"发射", x + w / 2, y + h / 2, p->on_accent);
            }
        }
    }
}

// 发射总进度（0..1）：音频可用时按“遍×帧”精确计数，无音频时按时间折算
static float ofdm_msg_tx_progress(Global_State *gs) {
    if (!s_msg_tx_on) return 0.0f;
    uint32_t n = (s_tx_total_frames > 0) ? s_tx_total_frames : 1;
    if (!s_tx_audio_ok) {
        double cycle_ms = (double)n * (double)OFDM_FRAME_LENGTH / (double)OFDM_SAMPLE_RATE * 1000.0;
        double el = (double)(gs->timestamp - s_msg_tx_start_ts);
        double f = el / (cycle_ms * (double)OFDM_MSG_TX_PASSES);
        if (f < 0.0) f = 0.0;
        if (f > 1.0) f = 1.0;
        return (float)f;
    }
    {
        uint32_t done = s_msg_tx_pass * n + s_msg_tx_frame;
        float f = (float)done / (float)(n * OFDM_MSG_TX_PASSES);
        if (f < 0.0f) f = 0.0f;
        if (f > 1.0f) f = 1.0f;
        return f;
    }
}

static void ofdm_msg_status_draw(Global_State *gs, const OFDM_Palette *p) {
    int32_t W = (int32_t)gs->gfx->width;
    wchar_t left[160];
    wchar_t right[64];
    OFDM_RGB lc = p->dim;
    left[0] = L'\0';
    right[0] = L'\0';

    ofdm_fill(gs, 0, OFDM_MSG_STATUS_Y, W, OFDM_MSG_STATUS_H, p->panel);
    ofdm_fill(gs, 0, OFDM_MSG_STATUS_Y, W, 1, p->hairline);

    if (s_msg_tx_on) {
        uint32_t n = (s_tx_total_frames > 0) ? s_tx_total_frames : 1;
        int32_t pass = (int32_t)s_msg_tx_pass + 1;
        if (pass > OFDM_MSG_TX_PASSES) pass = OFDM_MSG_TX_PASSES;
        swprintf(left, sizeof(left) / sizeof(wchar_t), L"发射中 · 第 %d/%d 遍",
                 pass, OFDM_MSG_TX_PASSES);
        swprintf(right, sizeof(right) / sizeof(wchar_t), L"帧 %d/%d",
                 (int)s_msg_tx_frame, (int)n);
        lc = p->accent;
    }
    else if (s_msg_rx_on) {
        OFDM_RX_Stat st;
        memset(&st, 0, sizeof(st));
        if (s_rx) ofdm_rx_get_stat(s_rx, &st);
        if (!s_mic_ok) {
            swprintf(left, sizeof(left) / sizeof(wchar_t), L"接收失败 · 麦克风不可用");
            lc = p->err;
        }
        else if (st.locked) {
            swprintf(left, sizeof(left) / sizeof(wchar_t), L"接收中 · 已锁定 · 帧 %d",
                     (int)st.frames_ok);
            lc = p->ok;
        }
        else {
            swprintf(left, sizeof(left) / sizeof(wchar_t), L"接收中 · 搜索前导…");
            lc = p->warn;
        }
        {
            float rms_db = (s_mic_rms_ema > 1.0f) ? 20.0f * log10f(s_mic_rms_ema / 32768.0f) : -99.0f;
            if (rms_db > -90.0f) swprintf(right, sizeof(right) / sizeof(wchar_t), L"%.0f dB", (double)rms_db);
        }
    }
    else {
        swprintf(left, sizeof(left) / sizeof(wchar_t), L"就绪 · 点「接收」或「发射」");
        lc = p->dim;
    }
    ofdm_text(gs, OFDM_F_TITLE, left, 10, OFDM_MSG_STATUS_Y + 2, lc);
    if (right[0]) ofdm_text_right(gs, OFDM_F_TITLE, right, W - 10, OFDM_MSG_STATUS_Y + 2, p->dim);

    // 最底部：发射进度条
    ofdm_fill(gs, 0, OFDM_MSG_PROG_Y, W, OFDM_MSG_PROG_H, p->track);
    if (s_msg_tx_on) {
        int32_t fw = (int32_t)(ofdm_msg_tx_progress(gs) * (float)W + 0.5f);
        if (fw > 0) ofdm_fill(gs, 0, OFDM_MSG_PROG_Y, fw, OFDM_MSG_PROG_H, p->accent);
    }
}

static void ofdm_msg_log_draw(Global_State *gs, const OFDM_Palette *p) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t view_h = OFDM_MSG_LOG_BOT - OFDM_MSG_LOG_TOP;
    int32_t placeholder = (s_msg_rx_on && s_msg_cur < 0) ? 3 : 0; // 空行+标签+正文
    int32_t total_lines, total_px, max_scroll, scroll, y0, i;

    if (s_log_dirty) ofdm_msg_log_rebuild(gs);
    total_lines = s_logline_num + placeholder;
    total_px = total_lines * OFDM_MSG_LINE_ADV;
    max_scroll = (total_px > view_h) ? (total_px - view_h) : 0;
    if (s_msg_scroll > max_scroll) s_msg_scroll = max_scroll;
    if (s_msg_scroll < 0) s_msg_scroll = 0;
    if (s_msg_auto) s_msg_scroll = 0;
    scroll = s_msg_scroll;
    y0 = OFDM_MSG_LOG_BOT - total_px + scroll;

    // 空记录：居中提示（避免大片留白）
    if (s_msg_count == 0 && placeholder == 0 && !s_msg_tx_on) {
        int32_t cy = (OFDM_MSG_LOG_TOP + OFDM_MSG_LOG_BOT) / 2;
        ofdm_text_center(gs, OFDM_F_BIG, (wchar_t *)L"暂无消息记录", W / 2, cy - 10, p->dim);
        ofdm_text_center(gs, OFDM_F_TITLE, (wchar_t *)L"点下方「接收」或「发射」开始", W / 2, cy + 12, p->dim);
    }

    gfx_set_clip(gs->gfx, 0, OFDM_MSG_LOG_TOP, W, view_h);

    for (i = 0; i < s_logline_num; i++) {
        OFDM_LogLine *ln = &s_loglines[i];
        int32_t y = y0 + i * OFDM_MSG_LINE_ADV;
        if (y + OFDM_MSG_LINE_ADV <= OFDM_MSG_LOG_TOP) continue;
        if (y >= OFDM_MSG_LOG_BOT) break;
        if (ln->kind == 2) continue;
        if (ln->msg < 0 || ln->msg >= s_msg_count) continue;
        {
            OFDM_Msg *m = ofdm_msg_slot(ln->msg);
            if (ln->kind == 0) {
                // 身份标签 + 右侧进行中提示
                OFDM_RGB lc = (m->dir == OFDM_MSG_TX) ? p->accent : p->ok;
                ofdm_text(gs, OFDM_F_TITLE, (wchar_t *)((m->dir == OFDM_MSG_TX) ? L"发出" : L"收到"),
                          OFDM_MSG_PAD_X, y, lc);
                if (m->state == OFDM_MSG_LIVE) {
                    ofdm_text_right(gs, OFDM_F_TITLE,
                                    (wchar_t *)((m->dir == OFDM_MSG_TX) ? L"发射中 · 3 遍" : L"接收中"),
                                    W - OFDM_MSG_SB_W - 8, y, p->dim);
                }
            }
            else {
                // 正文：进行中为暗色，完成为主文字色
                wchar_t line[128];
                int32_t n = ln->len;
                OFDM_RGB tc = (m->state == OFDM_MSG_LIVE) ? p->dim : p->text;
                if (n > 127) n = 127;
                if (n > 0) memcpy(line, m->text + ln->off, (size_t)n * sizeof(wchar_t));
                line[n] = L'\0';
                if (n == 0) { line[0] = L'…'; line[1] = L'\0'; } // 空正文占位
                ofdm_text(gs, OFDM_F_TITLE, line, OFDM_MSG_PAD_X, y, tc);
            }
        }
    }

    // 尚未收到任何内容时的安抚性占位
    if (placeholder) {
        int32_t base = s_logline_num;
        int32_t y_label = y0 + (base + 1) * OFDM_MSG_LINE_ADV;
        int32_t y_body = y0 + (base + 2) * OFDM_MSG_LINE_ADV;
        wchar_t dots[24];
        int32_t d = (int32_t)((gs->timestamp / 380) % 4);
        swprintf(dots, sizeof(dots) / sizeof(wchar_t), L"正在接收%s", d == 0 ? L"" : (d == 1 ? L"." : (d == 2 ? L".." : L"...")));
        if (y_label + OFDM_MSG_LINE_ADV > OFDM_MSG_LOG_TOP && y_label < OFDM_MSG_LOG_BOT) {
            ofdm_text(gs, OFDM_F_TITLE, (wchar_t *)L"收到", OFDM_MSG_PAD_X, y_label, p->ok);
        }
        if (y_body + OFDM_MSG_LINE_ADV > OFDM_MSG_LOG_TOP && y_body < OFDM_MSG_LOG_BOT) {
            ofdm_text(gs, OFDM_F_TITLE, dots, OFDM_MSG_PAD_X, y_body, p->dim);
        }
    }

    gfx_reset_clip(gs->gfx);

    // 记录区滚动条（右缘，贴底=最新）
    if (total_px > view_h) {
        int32_t track_h = view_h;
        int32_t th = track_h * view_h / total_px;
        int32_t frac;
        if (th < 10) th = 10;
        if (th > track_h) th = track_h;
        frac = (max_scroll > 0) ? (scroll * (track_h - th) / max_scroll) : 0;
        ofdm_fill(gs, W - OFDM_MSG_SB_W, OFDM_MSG_LOG_TOP, OFDM_MSG_SB_W, track_h, p->track);
        ofdm_fill(gs, W - OFDM_MSG_SB_W, OFDM_MSG_LOG_TOP + (track_h - th) - frac,
                  OFDM_MSG_SB_W, th, p->dim);
    }
}

static void ofdm_msg_render(Global_State *gs) {
    OFDM_Palette p;
    ofdm_palette(gs, &p);
    int32_t W = (int32_t)gs->gfx->width;
    int32_t H = (int32_t)gs->gfx->height;
    ofdm_fill(gs, 0, 0, W, H, p.bg);
    ofdm_chrome_header(gs, &p, (wchar_t *)L"消息收发");
    ofdm_msg_log_draw(gs, &p);
    ofdm_msg_buttons_draw(gs, &p);
    ofdm_msg_status_draw(gs, &p);
    gfx_refresh(gs->gfx);
}

// ---------------- 接收会话 ----------------

static void ofdm_msg_stop_rx(void) {
    if (!s_msg_rx_on) return;
    ofdm_rx_stop();
    if (s_msg_cur >= 0 && s_msg_cur < OFDM_MSG_SLOTS) s_msgs[s_msg_cur].state = OFDM_MSG_DONE;
    s_msg_cur = -1;
    s_msg_raw_len = 0;
    s_msg_seg_start = 0;
    s_msg_prev_start = 0;
    s_msg_prev_len = 0;
    s_msg_rx_dirty = 0;
    s_msg_rx_on = 0;
    s_log_dirty = 1;
    s_msg_last_draw_ts = 0;
}

static void ofdm_msg_start_rx(Global_State *gs) {
    if (s_msg_rx_on) return;
    if (s_msg_tx_on) ofdm_msg_stop_tx();
    if (!s_msg_raw) s_msg_raw = (uint8_t *)platform_calloc(OFDM_MSG_RAWCAP, 1);
    if (!s_msg_raw) return;
    s_msg_raw_len = 0;
    s_msg_seg_start = 0;
    s_msg_prev_start = 0;
    s_msg_prev_len = 0;
    s_msg_cur = -1;
    s_msg_rx_dirty = 0;
    s_msg_last_pkt_ts = get_timestamp_in_ms();
    if (ofdm_rx_start(gs, OFDM_RX_SINK_MSG) != 0) {
        printf("ofdm-msg: rx start FAILED (oom)\n");
        return;
    }
    s_msg_rx_on = 1;
    s_msg_auto = 1;
    s_msg_scroll = 0;
    s_log_dirty = 1;
    s_msg_last_draw_ts = 0;
}

// 解出 packet 净荷（消息页）：累积原始字节 → 段级去重 → 待重建记录文本
static void ofdm_msg_on_rx_bytes(const uint8_t *bytes, uint32_t len) {
    if (!s_msg_rx_on || !s_msg_raw || !bytes || len == 0) return;
    if (s_msg_raw_len + len > OFDM_MSG_RAWCAP) len = OFDM_MSG_RAWCAP - s_msg_raw_len;
    if (len == 0) return;
    memcpy(s_msg_raw + s_msg_raw_len, bytes, len);
    s_msg_raw_len += len;
    s_msg_last_pkt_ts = get_timestamp_in_ms();

    // 载荷段边界：短包（< 每帧净荷上限）表示一条净荷结束。
    // 对端把同一条报文连发多遍时，相邻两段逐字节相同 → 丢弃后一段（记录里只留一条）。
    if (len < OFDM_PKT_PAYLOAD_MAX) {
        uint32_t seg_len = s_msg_raw_len - s_msg_seg_start;
        if (seg_len > 0 && seg_len == s_msg_prev_len &&
            memcmp(s_msg_raw + s_msg_prev_start, s_msg_raw + s_msg_seg_start, seg_len) == 0) {
            s_msg_raw_len = s_msg_seg_start; // 丢弃重复段
        }
        else {
            s_msg_prev_start = s_msg_seg_start;
            s_msg_prev_len = seg_len;
        }
        s_msg_seg_start = s_msg_raw_len;
    }
    s_msg_rx_dirty = 1;
}

// 把累积字节刷新到当前记录（流式逐字追加）
static void ofdm_msg_rx_refresh(void) {
    if (!s_msg_rx_dirty) return;
    s_msg_rx_dirty = 0;
    if (s_msg_raw_len == 0) return;
    if (s_msg_cur < 0) {
        s_msg_cur = ofdm_msg_new(OFDM_MSG_RX, OFDM_MSG_DONE);
        if (s_msg_cur < 0) return;
    }
    ofdm_msg_set_text_from_raw(s_msg_cur, s_msg_raw, s_msg_raw_len);
    if (s_msg_auto) s_msg_scroll = 0; // 仅在跟随模式下贴底；用户上翻阅读时不动视口
}

// ---------------- 发射会话 ----------------

static void ofdm_msg_finish_tx(void) {
    ofdm_tx_stop();
    s_msg_tx_on = 0;
    s_msg_tx_drain = 0;
    s_msg_tx_pass = OFDM_MSG_TX_PASSES;
    if (s_msg_tx_entry >= 0 && s_msg_tx_entry < OFDM_MSG_SLOTS) {
        s_msgs[s_msg_tx_entry].state = OFDM_MSG_DONE; // 发射完成 → 正文转亮
        s_msg_tx_entry = -1;
    }
    s_log_dirty = 1;
    s_msg_last_draw_ts = 0;
}

static void ofdm_msg_stop_tx(void) {
    if (!s_msg_tx_on) return;
    ofdm_msg_finish_tx();
}

static void ofdm_msg_start_tx(Global_State *gs, const uint8_t *payload, uint32_t len) {
    if (s_msg_tx_on) ofdm_msg_stop_tx();
    if (s_msg_rx_on) ofdm_msg_stop_rx();
    if (ofdm_tx_start(payload, len, gs) != 0) {
        printf("ofdm-msg: tx start FAILED (oom)\n");
        if (s_msg_tx_entry >= 0 && s_msg_tx_entry < OFDM_MSG_SLOTS) {
            s_msgs[s_msg_tx_entry].state = OFDM_MSG_DONE;
            s_msg_tx_entry = -1;
        }
        return;
    }
    s_msg_tx_pass = 0;
    s_msg_tx_frame = 0;
    s_msg_tx_drain = 0;
    s_msg_tx_start_ts = gs->timestamp;
    s_msg_tx_last_enq_ts = 0;
    s_msg_tx_on = 1;
    s_msg_auto = 1;
    s_msg_scroll = 0;
    s_msg_last_draw_ts = 0;
}

static void ofdm_msg_tx_tick(Global_State *gs) {
    uint32_t n;
    double frame_ms;
    if (!s_msg_tx_on) return;
    n = (s_tx_total_frames > 0) ? s_tx_total_frames : 1;
    frame_ms = (double)OFDM_FRAME_LENGTH / (double)OFDM_SAMPLE_RATE * 1000.0;

    if (!s_tx_audio_ok) {
        // 音频设备不可用：按时间推进（界面仍可完整演示），到点收尾
        double cycle_ms = frame_ms * (double)n;
        if ((double)(gs->timestamp - s_msg_tx_start_ts) >= cycle_ms * (double)OFDM_MSG_TX_PASSES) {
            ofdm_msg_finish_tx();
        }
        return;
    }
    if (s_msg_tx_drain) {
        // 末帧已入队：等队列播空（再留一点余量）后收尾
        if (audio_out_queue_free() &&
            (double)(gs->timestamp - s_msg_tx_last_enq_ts) >= frame_ms + 60.0) {
            ofdm_msg_finish_tx();
        }
        return;
    }
    while (audio_out_queue_free()) {
        int32_t r = ofdm_tx_render_push();
        if (r < 0) { ofdm_msg_finish_tx(); return; }
        if (r == 1) {
            s_msg_tx_pass++;
            s_msg_tx_frame = 0;
            if (s_msg_tx_pass >= OFDM_MSG_TX_PASSES) { s_msg_tx_drain = 1; break; }
            ofdm_tx_rewind(s_tx);
            continue;
        }
        s_msg_tx_frame++;
        s_msg_tx_last_enq_ts = gs->timestamp;
    }
}

// ---------------- 入口/事件 ----------------

// 输入页提交的报文（wchar_t → UTF-8，供发射机切分）
static void ofdm_msg_payload_set(wchar_t *text) {
    uint32_t wlen = wcslen(text);
    uint32_t cap = wlen * 4 + 1;
    if (s_msg_payload) { free(s_msg_payload); s_msg_payload = NULL; }
    s_msg_payload = (char *)platform_calloc(cap, 1);
    if (!s_msg_payload) { s_msg_payload_len = 0; s_msg_payload_ready = 0; return; }
    s_msg_payload_len = _wcstombs(s_msg_payload, text, cap);
    s_msg_payload_ready = 1;
}

void ui_ofdm_msg_on_enter(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    s_msg_gs = global_state;
    memset(&s_msg_tap, 0, sizeof(s_msg_tap));
    s_msg_drag = 0;

    // 从发报输入页返回且带报文：记录一条“发出”并开始连发 3 遍
    if (s_msg_payload_ready) {
        s_msg_payload_ready = 0;
        int32_t slot = ofdm_msg_new(OFDM_MSG_TX, OFDM_MSG_LIVE);
        if (slot >= 0) {
            ofdm_msg_set_text_from_raw(slot, (const uint8_t *)s_msg_payload, s_msg_payload_len);
            s_msg_tx_entry = slot;
        }
        ofdm_msg_start_tx(global_state, (const uint8_t *)s_msg_payload, s_msg_payload_len);
        if (s_msg_payload) { free(s_msg_payload); s_msg_payload = NULL; }
        s_msg_payload_len = 0;
    }
    s_log_dirty = 1;
    s_msg_last_draw_ts = 0;
    ofdm_msg_render(global_state);
}

// 按钮动作：0=接收/停止接收，1=发射
static int32_t ofdm_msg_button_action(Key_Event *ke, Global_State *gs, int32_t idx) {
    if (idx == 0) {
        if (s_msg_rx_on) ofdm_msg_stop_rx();
        else ofdm_msg_start_rx(gs);
    }
    else {
        if (s_msg_tx_on) {
            ofdm_msg_stop_tx(); // 再次点按=中止发射
            return STATE_OFDM_MSG;
        }
        // 发报前先停接收（收/发共用音频外设，且编辑期间无人消费采集环）
        if (s_msg_rx_on) ofdm_msg_stop_rx();
        // 打开文本输入控件（编辑能力/输入法沿用宿主控件）
        ui_widget_input_init(ke, gs, gs->w_input_main, (wchar_t *)L"发报（Ctrl+A 返回）");
        ofdm_input_fill_default_text(ke, gs);
        return STATE_OFDM_MSG_INPUT;
    }
    return STATE_OFDM_MSG;
}

int32_t ui_ofdm_msg_event(Key_Event *key_event, Global_State *global_state) {
    Key_Event *ke = key_event;
    int32_t tx = 0, ty = 0;
    int32_t tapped = ofdm_tap_feed(&s_msg_tap, ke, &tx, &ty);
    // 硬按键仅作兼容兜底；触屏不派生任何键码，全部交互走本页自绘控件
    int32_t hard = !(ke->key_code != NANO_KEY_IDLE && ke->is_soft_key && !ke->is_softkbd);

    // A / 页眉“返回”：结束会话并回模式菜单
    if (hard && ke->key_edge < 0 && ke->key_code == NANO_KEY_esc) {
        ofdm_msg_stop_rx();
        ofdm_msg_stop_tx();
        return STATE_OFDM_MENU;
    }
    if (tapped && ofdm_back_hit(global_state, tx, ty)) {
        ofdm_msg_stop_rx();
        ofdm_msg_stop_tx();
        return STATE_OFDM_MENU;
    }

    // 点按两个按钮
    if (tapped) {
        for (int32_t i = 0; i < 2; i++) {
            int32_t bx, by, bw, bh;
            ofdm_msg_buttons_rect(global_state, i, &bx, &by, &bw, &bh);
            if (tx >= bx && tx < bx + bw && ty >= by && ty < by + bh) {
                int32_t next = ofdm_msg_button_action(ke, global_state, i);
                if (next != STATE_OFDM_MSG) return next; // 进输入页：由输入控件自行绘制
                s_log_dirty = 1;
                break;
            }
        }
    }

    // 记录区拖动滚动（触屏；硬按键 ←→ 兜底）
    if (ke->touch_edge & TOUCH_EDGE_DOWN) {
        if (ke->touch_down_y >= OFDM_MSG_LOG_TOP && ke->touch_down_y < OFDM_MSG_LOG_BOT) {
            s_msg_drag = 1;
            s_msg_drag_y0 = ke->touch_down_y;
            s_msg_drag_scroll0 = s_msg_auto ? 0 : s_msg_scroll;
            s_msg_auto = 0;
        }
    }
    else if (ke->is_touching && s_msg_drag) {
        s_msg_scroll = s_msg_drag_scroll0 + (ke->touch_y - s_msg_drag_y0);
        if (s_msg_scroll < 0) s_msg_scroll = 0;
    }
    else if ((ke->touch_edge & TOUCH_EDGE_UP) && s_msg_drag) {
        s_msg_drag = 0;
        if (s_msg_scroll <= 2) { s_msg_scroll = 0; s_msg_auto = 1; }
    }
    if (hard && (ke->key_edge == -1 || ke->key_edge == -2)) {
        if (ke->key_code == NANO_KEY_1) {
            int32_t next = ofdm_msg_button_action(ke, global_state, 0);
            if (next != STATE_OFDM_MSG) return next;
            s_log_dirty = 1;
        }
        else if (ke->key_code == NANO_KEY_2 || ke->key_code == NANO_KEY_enter) {
            int32_t next = ofdm_msg_button_action(ke, global_state, 1);
            if (next != STATE_OFDM_MSG) return next;
            s_log_dirty = 1;
        }
        else if (ke->key_code == NANO_KEY_left || ke->key_code == NANO_KEY_right) {
            s_msg_auto = 0;
            s_msg_scroll += (ke->key_code == NANO_KEY_left) ? OFDM_MSG_LINE_ADV * 3 : -OFDM_MSG_LINE_ADV * 3;
            if (s_msg_scroll < 0) { s_msg_scroll = 0; s_msg_auto = 1; }
        }
    }

    // 会话推进（接收/发射都是本页内的后台活动）
    if (s_msg_rx_on) {
        ofdm_rx_poll(global_state);
        ofdm_msg_rx_refresh();
        if (s_msg_cur >= 0 &&
            get_timestamp_in_ms() - s_msg_last_pkt_ts >= OFDM_MSG_RX_END_MS) {
            // 一条消息收完：转亮、清空累积，准备收下一条
            s_msgs[s_msg_cur].state = OFDM_MSG_DONE;
            s_msg_cur = -1;
            s_msg_raw_len = 0;
            s_msg_seg_start = 0;
            s_msg_prev_start = 0;
            s_msg_prev_len = 0;
            s_log_dirty = 1;
        }
    }
    if (s_msg_tx_on) ofdm_msg_tx_tick(global_state);

    // 限频整页重绘（文本/状态变化时立即重绘）
    if (s_log_dirty || global_state->timestamp - s_msg_last_draw_ts >= OFDM_STA_REFRESH_MS) {
        s_msg_last_draw_ts = global_state->timestamp;
        ofdm_msg_render(global_state);
    }
    return STATE_OFDM_MSG;
}

// ---------------- 发报输入页（宿主输入控件） ----------------

void ui_ofdm_msg_input_on_enter(Key_Event *key_event, Global_State *global_state) {
    ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
}

int32_t ui_ofdm_msg_input_event(Key_Event *key_event, Global_State *global_state) {
    int32_t next = ui_widget_input_event_handler(
        key_event, global_state, global_state->w_input_main,
        STATE_OFDM_MSG, STATE_OFDM_MSG_INPUT, STATE_OFDM_MSG);
    if (next == STATE_OFDM_MSG) {
        // 仅 D（提交）携带报文返回；A / 空报文一律视为取消
        if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_enter) {
            wchar_t *text = global_state->w_input_main->textarea.text;
            if (wcslen(text) == 0) return STATE_OFDM_MSG_INPUT;
            ofdm_msg_payload_set(text);
        }
    }
    return next;
}

// ============================================================================
// 八、STATE_OFDM_LOOP / STATE_OFDM_LOOPING（软件环路自测：自发自收，不出声）
// 发射机逐帧渲染 → 直接喂给本机接收机 → 显示解调文本与环回校验结果。
// 处理按帧分摊到多个事件循环迭代，UI 保持响应，可中途按 A 中止。
// ============================================================================

#define OFDM_LOOP_PAD (2400) // 前导静默采样数（模拟冷启动捕获）
#define OFDM_LOOP_LOG_LINES (10) // 调试日志行数（环形）
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
static int32_t s_loop_stage = 0;         // 0=前导静默 1=逐帧环回 2=尾部静默冲刷
static int32_t s_loop_done = 0;
static int32_t s_loop_match = 0;         // 1=净荷逐字节一致
static uint64_t s_loop_last_draw_ts = 0;
static int32_t s_loop_ui_init = 0;
static Global_State *s_loop_gs = NULL;
static OFDM_Tap s_loop_tap;              // 页眉“返回”点按判定

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
    s_loop_match = 0;
    if (s_loop_ui_init && s_loop_gs && s_loop_gs->w_textarea_main) {
        ofdm_text_card_restore(s_loop_gs, s_loop_gs->w_textarea_main);
    }
    s_loop_ui_init = 0;
    s_loop_gs = NULL;
}

// 环路自测状态页：进度卡（状态/帧进度/统计）+ 结果文本卡
static void ofdm_loop_draw(Key_Event *ke, Global_State *gs) {
    OFDM_Palette p;
    ofdm_palette(gs, &p);
    int32_t W = (int32_t)gs->gfx->width;
    int32_t H = (int32_t)gs->gfx->height;
    int32_t x0 = 10, cw = W - 20;

    ofdm_fill(gs, 0, 0, W, H, p.bg);
    ofdm_chrome_header(gs, &p, (wchar_t *)L"环路自测");

    OFDM_RX_Stat st;
    memset(&st, 0, sizeof(st));
    if (s_loop_rx) ofdm_rx_get_stat(s_loop_rx, &st);

    // ---------------- 进度卡 ----------------
    int32_t cy = OFDM_HDR_H + 6; // 32
    int32_t ch = 72;             // 32..104
    ofdm_card(gs, x0, cy, cw, ch, 12, p.panel, p.edge);

    OFDM_RGB state_color = p.accent;
    wchar_t *state_text = (wchar_t *)L"环回处理中";
    if (s_loop_done) {
        state_color = s_loop_match ? p.ok : p.err;
        state_text = s_loop_match ? (wchar_t *)L"校验通过" : (wchar_t *)L"校验失败";
    }
    int32_t pulse = (int32_t)((gs->timestamp / 450) % 2);
    ofdm_dot(gs, x0 + 16, cy + 18, s_loop_done ? 4 : (pulse ? 5 : 4), state_color,
             (!s_loop_done && !pulse) ? 1 : 0);
    ofdm_text(gs, OFDM_F_BIG, state_text, x0 + 30, cy + 7, state_color);

    wchar_t buf[160];
    swprintf(buf, sizeof(buf) / sizeof(wchar_t), L"%d / %d 帧",
             (int)s_loop_idx, (int)s_loop_total);
    ofdm_text_right(gs, OFDM_F_TITLE, buf, x0 + cw - 14, cy + 10, p.dim);

    float frac = (s_loop_total > 0) ? (float)s_loop_idx / (float)s_loop_total : 0.0f;
    if (s_loop_done) frac = 1.0f;
    ofdm_bar(gs, x0 + 14, cy + 38, cw - 28, 10, frac, p.track, state_color);

    swprintf(buf, sizeof(buf) / sizeof(wchar_t),
             L"已解出 %d 帧 · 坏帧 %d · RS 纠错 %d",
             (int)st.frames_ok, (int)st.frames_bad, (int)st.rs_corrected);
    ofdm_text(gs, OFDM_F_TITLE, buf, x0 + 14, cy + 52, p.dim);

    // ---------------- 结果/日志 ----------------
    ofdm_text_card_draw(gs, &p, gs->w_textarea_main, (wchar_t *)L"结果与日志", (wchar_t *)L"不出声自测");
    ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);
    gs->w_textarea_main->is_modified = 0; // 同上：避免每个状态帧重排全文

    if (s_loop_done) {
        ofdm_chrome_footer(gs, &p, (wchar_t *)L"←→ 滚动查看", (wchar_t *)L"D 重测 · A 返回");
    }
    else {
        ofdm_chrome_footer(gs, &p, (wchar_t *)L"←→ 滚动查看", (wchar_t *)L"A 中止");
    }
    gfx_refresh(gs->gfx);
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
    s_loop_match = 0;
    s_loop_last_draw_ts = 0;
    s_loop_gs = global_state;
    memset(&s_loop_tap, 0, sizeof(s_loop_tap));

    printf("ofdm-loop: 开始环回，净荷 %lu 字节 → %lu 帧（帧长 %d 采样）\n",
           (unsigned long)s_loop_payload_len, (unsigned long)s_loop_total, (int)OFDM_FRAME_LENGTH);
    if (!s_loop_tx || !s_loop_rx || !s_loop_frame || !s_loop_cap || !s_loop_view || !s_loop_log) {
        printf("ofdm-loop: 内存分配失败 tx=%p rx=%p frame=%p cap=%p view=%p log=%p\n",
               s_loop_tx, s_loop_rx, s_loop_frame, s_loop_cap, s_loop_view, s_loop_log);
        ui_ofdm_looping_cleanup();
        ofdm_error_page(key_event, global_state, (wchar_t *)L"环路自测",
                        (wchar_t *)L"内存不足", (wchar_t *)L"无法启动环回，按 A 返回");
        return;
    }

    // 文本区嵌入进度卡下方的卡片内
    ofdm_text_card_setup(global_state, global_state->w_textarea_main,
                         106, (int32_t)global_state->gfx->height - OFDM_FTR_H - 10);
    s_loop_ui_init = 1;
    swprintf(s_loop_view, UI_STR_BUF_MAX_LENGTH,
             L"发送 %d 字节 → %d 个物理帧\n\n正在环回处理（不出声）……\n"
             L"发射机逐帧渲染后直接喂给本机接收机，\n用于验证调制/解调链路是否可行。\n",
             (int)s_loop_payload_len, (int)s_loop_total);
    ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, s_loop_view, 0, 1);
    ofdm_loop_draw(key_event, global_state);
}

// 环回完成：汇总统计与校验结果，显示调试日志与解调文本
static void ui_ofdm_looping_show_result(Key_Event *key_event, Global_State *global_state) {
    OFDM_RX_Stat st;
    ofdm_rx_get_stat(s_loop_rx, &st);
    int32_t match = (s_loop_cap_len == s_loop_payload_len &&
                     memcmp(s_loop_cap, s_loop_payload, s_loop_payload_len) == 0) ? 1 : 0;
    s_loop_match = match;

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
             L"环回校验：%ls\n"
             L"发送 %d 字节 → %d 个物理帧\n"
             L"接收：成功 %d 帧 / 坏帧 %d / RS 纠错 %d 字节\n\n",
             match ? L"逐字节一致（解调可行）" : L"内容不一致",
             (int)s_loop_payload_len, (int)s_loop_total,
             (int)st.frames_ok, (int)st.frames_bad, (int)st.rs_corrected);
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
    wcscat(s_loop_view, L"\n解调文本：\n");
    // 追加解调出的文本（受显示缓冲上限截断）
    uint32_t cur = wcslen(s_loop_view);
    uint32_t remain = (UI_STR_BUF_MAX_LENGTH - 1 > cur) ? (UI_STR_BUF_MAX_LENGTH - 1 - cur) : 0;
    if (remain > 0 && s_loop_cap_len > 0) {
        uint32_t blen = (s_loop_cap_len < remain) ? s_loop_cap_len : remain;
        _mbstowcs(s_loop_view + cur, (const char *)s_loop_cap, blen);
    }
    ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, s_loop_view, 0, 1);
    ofdm_loop_draw(key_event, global_state);
}

int32_t ui_ofdm_looping_event(Key_Event *key_event, Global_State *global_state) {
    // 异常兜底（如 on_enter 内存不足）：直接回输入态
    if (!s_loop_tx || !s_loop_rx) return STATE_OFDM_LOOP;

    // 短按 A：中止/返回编辑（保留输入文本）；触屏点击页眉右 1/4（“返回”）同效
    int32_t abort = 0;
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_esc) abort = 1;
    else {
        int32_t tx = 0, ty = 0;
        if (ofdm_tap_feed(&s_loop_tap, key_event, &tx, &ty) && ofdm_back_hit(global_state, tx, ty)) abort = 1;
    }
    if (abort) {
        ui_ofdm_looping_cleanup();
        return STATE_OFDM_LOOP;
    }
    // 16 键虚拟键盘的键码不作为键码采纳（本状态全交互走触屏控件）
    if (key_event->key_code != NANO_KEY_IDLE && key_event->is_soft_key && !key_event->is_softkbd) {
        return STATE_OFDM_LOOPING;
    }

    // 左右键：滚行翻阅（结果页含统计/日志/解调文本，内容较长）
    if (key_event->key_edge < 0 &&
        (key_event->key_code == NANO_KEY_left || key_event->key_code == NANO_KEY_right)) {
        ui_widget_textarea_event_handler(key_event, global_state,
                                         global_state->w_textarea_main,
                                         STATE_OFDM_LOOPING, STATE_OFDM_LOOPING);
        ofdm_loop_draw(key_event, global_state);
        return STATE_OFDM_LOOPING;
    }

    // 触屏：结果页文本区拖动/惯性滚动
    if (key_event->touch_edge || key_event->is_touching) {
        if (ui_widget_textarea_touch_handler(key_event, global_state, global_state->w_textarea_main) == 1) {
            ofdm_loop_draw(key_event, global_state);
        }
    }

    // 完成后：短按 D（或点页脚右半「重测」）以同一文本再次环回
    if (s_loop_done) {
        int32_t retry = 0;
        if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_enter) retry = 1;
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            int32_t W = (int32_t)global_state->gfx->width;
            int32_t H = (int32_t)global_state->gfx->height;
            if (key_event->touch_down_y >= H - OFDM_FTR_H && key_event->touch_down_y < H &&
                key_event->touch_down_x >= W / 2) {
                retry = 1;
            }
        }
        if (retry) {
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
        return STATE_OFDM_LOOPING;
    }

    // 限频刷新进度卡（帧处理极快，避免每帧全屏推送拖慢环回）
    if (global_state->timestamp - s_loop_last_draw_ts >= OFDM_STA_REFRESH_MS) {
        s_loop_last_draw_ts = global_state->timestamp;
        ofdm_loop_draw(key_event, global_state);
    }
    return STATE_OFDM_LOOPING;
}
