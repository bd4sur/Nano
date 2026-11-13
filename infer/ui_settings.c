// ui_settings.c —— 系统设置（设置菜单 + 日期/时间/经纬度编辑器）
//
// 设计要点见 ui_settings.h。本文件只依赖图形层与全局状态，不直接访问触屏 HAL：
// 触屏样本由事件层统一采样后经 Key_Event 传入（与项目既有触屏化范式一致）。
//
// 配色隐喻：
//   开 = 绿；关 = 红；强调色 = #66CCFF（亮色主题下小字号文字用深一档的青色保证对比度）。
// 交互：
//   设置菜单：分组卡片列表，上下拖动滚动；点按行/开关/分段控件/步进器。
//   编辑器：大字预览 + 年/月/日（时/分/时区、度/分/秒）三列步进器；
//           点按箭头单步、按住连续调整；方向键切换字段、上下键调整、Enter 保存、Esc 取消。

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "platform.h"
#include "graphics.h"
#include "hal_key.h"
#include "hal_audio_out.h"

#include "ui.h"
#include "ui_app.h"
#include "ui_color.h"
#include "ui_settings.h"

// ===============================================================================
// 主题与通用绘图
// ===============================================================================

typedef struct {
    uint8_t bg_top[3];
    uint8_t bg_bot[3];
    uint8_t card[3];
    uint8_t card_edge[3];
    uint8_t surface[3];
    uint8_t surface_press[3];
    uint8_t text[3];
    uint8_t dim[3];
    uint8_t accent[3];      // 强调色 #66CCFF（填充/控件）
    uint8_t accent_ink[3];  // 强调色文字（亮色主题下更深）
    uint8_t accent_soft[3]; // 强调色浅底
    uint8_t on_accent[3];   // 强调色底上的文字
    uint8_t on[3];          // 开 = 绿
    uint8_t on_soft[3];
    uint8_t off[3];         // 关 = 红
    uint8_t off_soft[3];
    uint8_t track[3];
    uint8_t knob[3];
} UiSettingsPalette;

static void settings_palette(Global_State *gs, UiSettingsPalette *p) {
    static const UiSettingsPalette dark = {
        .bg_top        = {13, 16, 22},
        .bg_bot        = {22, 28, 38},
        .card          = {27, 33, 44},
        .card_edge     = {42, 52, 68},
        .surface       = {37, 46, 60},
        .surface_press = {58, 74, 96},
        .text          = {237, 242, 249},
        .dim           = {138, 150, 168},
        .accent        = {102, 204, 255},
        .accent_ink    = {102, 204, 255},
        .accent_soft   = {26, 52, 70},
        .on_accent     = {10, 20, 32},
        .on            = {34, 197, 94},
        .on_soft       = {18, 62, 42},
        .off           = {239, 68, 68},
        .off_soft      = {74, 32, 36},
        .track         = {52, 63, 79},
        .knob          = {248, 250, 252},
    };
    static const UiSettingsPalette light = {
        .bg_top        = {246, 248, 252},
        .bg_bot        = {228, 235, 245},
        .card          = {255, 255, 255},
        .card_edge     = {218, 226, 238},
        .surface       = {238, 243, 250},
        .surface_press = {212, 226, 242},
        .text          = {24, 32, 46},
        .dim           = {104, 118, 136},
        .accent        = {102, 204, 255},
        .accent_ink    = {12, 132, 186},
        .accent_soft   = {222, 242, 253},
        .on_accent     = {8, 24, 36},
        .on            = {22, 163, 74},
        .on_soft       = {219, 244, 228},
        .off           = {220, 38, 38},
        .off_soft      = {253, 228, 228},
        .track         = {223, 230, 240},
        .knob          = {255, 255, 255},
    };
    *p = (gs->ui_color_style == UI_COLOR_LIGHT) ? light : dark;
}

static inline int32_t settings_abs(int32_t v) { return (v < 0) ? -v : v; }

static inline int32_t settings_clamp(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// 四舍五入（避免依赖浮点 printf/lround 的差异）
static inline int32_t settings_round(float v) {
    return (int32_t)(v + ((v >= 0.0f) ? 0.5f : -0.5f));
}

static void settings_round_rect(Nano_GFX *gfx, int32_t x, int32_t y, int32_t w, int32_t h,
                                int32_t r, const uint8_t c[3], uint8_t mode) {
    if (w <= 0 || h <= 0) return;
    int32_t rm = (w < h) ? (w / 2) : (h / 2);
    if (r > rm) r = rm;
    if (r < 0) r = 0;
    gfx_draw_rounded_rectangle(gfx, x, y, w, h, r, r, r, r, c[0], c[1], c[2], mode);
}

static void settings_circle(Nano_GFX *gfx, int32_t cx, int32_t cy, int32_t r,
                            const uint8_t c[3], uint8_t mode) {
    if (r <= 0) return;
    settings_round_rect(gfx, cx - r, cy - r, r * 2, r * 2, r, c, mode);
}

static void settings_text(Nano_GFX *gfx, uint32_t font, const wchar_t *s,
                          int32_t x, int32_t y_top, const uint8_t c[3], uint8_t mode) {
    gfx_font_draw_text(gfx, font, (wchar_t *)s, x, y_top, c[0], c[1], c[2], mode);
}

static void settings_text_centered(Nano_GFX *gfx, uint32_t font, const wchar_t *s,
                                   int32_t cx, int32_t cy, const uint8_t c[3], uint8_t mode) {
    gfx_font_draw_text_centered(gfx, font, (wchar_t *)s, cx, cy, c[0], c[1], c[2], mode);
}

static void settings_text_right(Nano_GFX *gfx, uint32_t font, const wchar_t *s,
                                int32_t right_x, int32_t y_top, const uint8_t c[3], uint8_t mode) {
    int32_t tw = gfx_font_measure_text(font, (wchar_t *)s);
    gfx_font_draw_text(gfx, font, (wchar_t *)s, right_x - tw, y_top, c[0], c[1], c[2], mode);
}

static void settings_bg(Nano_GFX *gfx, const UiSettingsPalette *p) {
    int32_t H = (int32_t)gfx->height;
    for (int32_t y = 0; y < H; y += 8) {
        int32_t ye = y + 8;
        if (ye > H) ye = H;
        int32_t t = (H > 1) ? (y * 255 / (H - 1)) : 0;
        uint8_t r = (uint8_t)(p->bg_top[0] + ((int32_t)p->bg_bot[0] - (int32_t)p->bg_top[0]) * t / 255);
        uint8_t g = (uint8_t)(p->bg_top[1] + ((int32_t)p->bg_bot[1] - (int32_t)p->bg_top[1]) * t / 255);
        uint8_t b = (uint8_t)(p->bg_top[2] + ((int32_t)p->bg_bot[2] - (int32_t)p->bg_top[2]) * t / 255);
        gfx_draw_rectangle(gfx, 0, (uint32_t)y, (uint32_t)gfx->width, (uint32_t)(ye - y), r, g, b, 1);
    }
}

// 2px 粗线段：gfx_draw_line_anti_aliasing 内部使用加色混合（不认 mode），
// 在亮色主题的浅底上会把青色加成近白色，故此处用两条普通线段实现厚度。
static void settings_thick_line(Nano_GFX *gfx, int32_t x1, int32_t y1, int32_t x2, int32_t y2,
                                const uint8_t c[3]) {
    gfx_draw_line(gfx, (uint32_t)x1, (uint32_t)y1, (uint32_t)x2, (uint32_t)y2, c[0], c[1], c[2], 1);
    gfx_draw_line(gfx, (uint32_t)x1, (uint32_t)(y1 + 1), (uint32_t)x2, (uint32_t)(y2 + 1),
                  c[0], c[1], c[2], 1);
}

// 右向 chevron（列表行尾的「进入」提示，矢量绘制避免依赖字库码点）
static void settings_chevron_right(Nano_GFX *gfx, int32_t cx, int32_t cy, const uint8_t c[3]) {
    settings_thick_line(gfx, cx - 2, cy - 5, cx + 3, cy, c);
    settings_thick_line(gfx, cx + 3, cy, cx - 2, cy + 5, c);
}

static void settings_chevron_up(Nano_GFX *gfx, int32_t cx, int32_t cy, const uint8_t c[3]) {
    settings_thick_line(gfx, cx - 6, cy + 3, cx, cy - 3, c);
    settings_thick_line(gfx, cx, cy - 3, cx + 6, cy + 3, c);
}

static void settings_chevron_down(Nano_GFX *gfx, int32_t cx, int32_t cy, const uint8_t c[3]) {
    settings_thick_line(gfx, cx - 6, cy - 3, cx, cy + 3, c);
    settings_thick_line(gfx, cx, cy + 3, cx + 6, cy - 3, c);
}

// 矢量加/减号（步进器按钮）
static void settings_minus(Nano_GFX *gfx, int32_t cx, int32_t cy, const uint8_t c[3]) {
    settings_round_rect(gfx, cx - 6, cy - 1, 12, 2, 1, c, 1);
}

static void settings_plus(Nano_GFX *gfx, int32_t cx, int32_t cy, const uint8_t c[3]) {
    settings_round_rect(gfx, cx - 6, cy - 1, 12, 2, 1, c, 1);
    settings_round_rect(gfx, cx - 1, cy - 6, 2, 12, 1, c, 1);
}

// 开关：开=绿、关=红；白色圆钮
static void settings_switch(Nano_GFX *gfx, int32_t x, int32_t y, int32_t on,
                            const UiSettingsPalette *p) {
    const int32_t w = 42, h = 22, r = 11;
    const uint8_t *bg = on ? p->on : p->off;
    settings_round_rect(gfx, x, y, w, h, r, bg, 1);
    int32_t knob_cx = on ? (x + w - 11) : (x + 11);
    settings_circle(gfx, knob_cx, y + h / 2, 8, p->knob, 1);
}

// ===============================================================================
// 设置菜单：布局
// ===============================================================================

#define ST_HEADER_H   (32)
#define ST_CARD_X     (10)
#define ST_CARD_R     (12)
#define ST_TITLE_H    (22)
#define ST_SECTION_GAP (10)

enum {
    ROW_DATE = 0,
    ROW_TIME,
    ROW_LON,
    ROW_LAT,
    ROW_BRIGHT,
    ROW_THEME,
    ROW_CLOCK,
    ROW_VOLUME,
    ROW_KEYFB,
    ROW_IMU,
    ROW_LLM,
    ROW_TTS,
    ROW_ASR,
    ROW_AUTOOFF,
    ROW_COUNT
};

enum {
    SEC_TIME = 0,
    SEC_DISPLAY,
    SEC_SOUND,
    SEC_SYSTEM,
    SEC_COUNT
};

typedef struct {
    int32_t section;
    int32_t id;
    const wchar_t *label;
} SettingsRowDef;

static const SettingsRowDef SETTINGS_ROWS[ROW_COUNT] = {
    {SEC_TIME,    ROW_DATE,     L"日期"},
    {SEC_TIME,    ROW_TIME,     L"时间与时区"},
    {SEC_TIME,    ROW_LON,      L"经度"},
    {SEC_TIME,    ROW_LAT,      L"纬度"},
    {SEC_DISPLAY, ROW_BRIGHT,   L"屏幕亮度"},
    {SEC_DISPLAY, ROW_THEME,    L"颜色风格"},
    {SEC_DISPLAY, ROW_CLOCK,    L"时钟样式"},
    {SEC_SOUND,   ROW_VOLUME,   L"音量"},
    {SEC_SOUND,   ROW_KEYFB,    L"按键提示"},
    {SEC_SYSTEM,  ROW_IMU,      L"姿态传感器"},
    {SEC_SYSTEM,  ROW_LLM,      L"LLM 演示"},
    {SEC_SYSTEM,  ROW_TTS,      L"TTS 设置"},
    {SEC_SYSTEM,  ROW_ASR,      L"ASR 设置"},
    {SEC_SYSTEM,  ROW_AUTOOFF,  L"自动关机"},
};

static const wchar_t *SETTINGS_SECTION_TITLES[SEC_COUNT] = {
    L"时间与位置",
    L"显示",
    L"声音与输入",
    L"系统与智能",
};

typedef struct {
    int32_t w, h;
    int32_t content_y0, content_y1, content_h;
    int32_t row_y[ROW_COUNT];
    int32_t row_h[ROW_COUNT];
    int32_t section_title_y[SEC_COUNT];
    int32_t section_card_y[SEC_COUNT];
    int32_t section_card_h[SEC_COUNT];
    int32_t total_h;
    int32_t max_scroll;
    int32_t scroll;
} SettingsMenuLayout;

static int32_t settings_row_height(int32_t id) {
    return (id == ROW_BRIGHT || id == ROW_VOLUME) ? 44 : 34;
}

// 设置菜单当前滚动位置（内容区局部坐标的负偏移）
static int32_t s_menu_scroll = 0;

static int32_t settings_card_width(Global_State *gs) {
    int32_t w = (int32_t)gs->gfx->width - ST_CARD_X * 2;
    if (w < 40) w = 40;
    return w;
}

static void settings_menu_layout(Global_State *gs, SettingsMenuLayout *L) {
    memset(L, 0, sizeof(*L));
    L->w = (int32_t)gs->gfx->width;
    L->h = (int32_t)gs->gfx->height;
    L->content_y0 = ST_HEADER_H + 2;
    L->content_y1 = L->h - 2;
    if (L->content_y1 - L->content_y0 < 20) L->content_y1 = L->content_y0 + 20;
    L->content_h = L->content_y1 - L->content_y0;

    int32_t y = 4; // 内容区局部坐标
    for (int32_t s = 0; s < SEC_COUNT; s++) {
        L->section_title_y[s] = y;
        y += ST_TITLE_H;
        L->section_card_y[s] = y;
        int32_t card_start = y;
        for (int32_t i = 0; i < ROW_COUNT; i++) {
            if (SETTINGS_ROWS[i].section != s) continue;
            int32_t id = SETTINGS_ROWS[i].id;
            L->row_y[id] = y;
            L->row_h[id] = settings_row_height(id);
            y += L->row_h[id];
        }
        L->section_card_h[s] = y - card_start;
        y += ST_SECTION_GAP;
    }
    L->total_h = y;

    L->max_scroll = L->total_h - L->content_h + 4;
    if (L->max_scroll < 0) L->max_scroll = 0;

    s_menu_scroll = settings_clamp(s_menu_scroll, 0, L->max_scroll);
    L->scroll = s_menu_scroll;
}

static int32_t settings_row_screen_y(const SettingsMenuLayout *L, int32_t id) {
    return L->content_y0 + L->row_y[id] - L->scroll;
}

// ===============================================================================
// 设置菜单：行控件几何与绘制
// ===============================================================================

typedef enum {
    ST_HIT_NONE = 0,
    ST_HIT_BACK,
    ST_HIT_ROW,      // 整行（进入编辑器 / 开关 / 状态）
    ST_HIT_MINUS,
    ST_HIT_PLUS,
    ST_HIT_CHIP,
} SettingsHitKind;

typedef struct {
    int32_t kind;
    int32_t row;     // ROW_*
    int32_t chip;    // 分段控件序号
} SettingsHit;

#define ST_STEP_BTN_W (26)
#define ST_STEP_VAL_W (46)
#define ST_STEP_GAP   (6)
#define ST_STEP_H     (22)

static void settings_stepper_geom(Global_State *gs, int32_t *minus_x, int32_t *val_x,
                                  int32_t *plus_x, int32_t *btn_y, int32_t row_sy, int32_t row_h) {
    int32_t right = (int32_t)gs->gfx->width - 24;
    int32_t total = ST_STEP_BTN_W + ST_STEP_GAP + ST_STEP_VAL_W + ST_STEP_GAP + ST_STEP_BTN_W;
    int32_t x0 = right - total;
    int32_t by = row_sy + ((row_h >= 44) ? 4 : 6);
    *minus_x = x0;
    *val_x = x0 + ST_STEP_BTN_W + ST_STEP_GAP;
    *plus_x = *val_x + ST_STEP_VAL_W + ST_STEP_GAP;
    *btn_y = by;
}

static int32_t settings_chip_count(int32_t id) {
    switch (id) {
        case ROW_THEME: return 2;
        case ROW_CLOCK: return 2;
        case ROW_KEYFB: return 4;
        case ROW_TTS:   return 3;
        case ROW_ASR:   return 2;
        default:        return 0;
    }
}

static void settings_chip_text(int32_t id, int32_t idx, const wchar_t **out) {
    static const wchar_t *THEME[] = {L"亮色", L"暗色"};
    static const wchar_t *CLOCK[] = {L"小", L"大"};
    static const wchar_t *KEYFB[] = {L"无", L"灯光", L"蜂鸣", L"灯光+蜂鸣"};
    static const wchar_t *TTS[]   = {L"关闭", L"实时", L"统一"};
    static const wchar_t *ASR[]   = {L"编辑后提交", L"立刻提交"};
    *out = L"";
    switch (id) {
        case ROW_THEME: if (idx >= 0 && idx < 2) *out = THEME[idx]; break;
        case ROW_CLOCK: if (idx >= 0 && idx < 2) *out = CLOCK[idx]; break;
        case ROW_KEYFB: if (idx >= 0 && idx < 4) *out = KEYFB[idx]; break;
        case ROW_TTS:   if (idx >= 0 && idx < 3) *out = TTS[idx]; break;
        case ROW_ASR:   if (idx >= 0 && idx < 2) *out = ASR[idx]; break;
        default: break;
    }
}

static int32_t settings_chip_selected(Global_State *gs, int32_t id) {
    switch (id) {
        case ROW_THEME: return (gs->ui_color_style == UI_COLOR_LIGHT) ? 0 : 1;
        case ROW_CLOCK: return (gs->splash_clock_style == SPLASH_CLOCK_SMALL) ? 0 : 1;
        case ROW_KEYFB: return gs->key_feedback_mode & 3;
        case ROW_TTS:   return settings_clamp(gs->tts_req_mode, 0, 2);
        case ROW_ASR:   return (gs->is_auto_submit_after_asr != 0) ? 1 : 0;
        default:        return 0;
    }
}

// 分段控件：右对齐，返回总宽度；out_x/out_w 可为 NULL
static int32_t settings_chip_geom(Global_State *gs, int32_t id, int32_t *out_x, int32_t *out_w) {
    int32_t n = settings_chip_count(id);
    int32_t widths[8];
    int32_t total = 0;
    const int32_t gap = 4;
    for (int32_t i = 0; i < n; i++) {
        const wchar_t *t = NULL;
        settings_chip_text(id, i, &t);
        int32_t tw = gfx_font_measure_text(GFX_FONT_ALPHA_12, (wchar_t *)t);
        int32_t w = tw + 16;
        if (w < 30) w = 30;
        widths[i] = w;
        total += w;
        if (i > 0) total += gap;
    }
    int32_t right = (int32_t)gs->gfx->width - 24;
    int32_t x = right - total;
    for (int32_t i = 0; i < n; i++) {
        out_x[i] = x;
        out_w[i] = widths[i];
        x += widths[i] + gap;
    }
    return total;
}

static void settings_format_tz(int32_t tzmin, wchar_t *out, size_t n) {
    int32_t v = settings_abs(tzmin);
    swprintf(out, n, L"%lc%02d:%02d", (tzmin < 0) ? L'-' : L'+', v / 60, v % 60);
}

static void settings_menu_row_value(Global_State *gs, int32_t id, wchar_t *out, size_t n) {
    if (id == ROW_DATE) {
        swprintf(out, n, L"%04d-%02d-%02d", (int)gs->year, (int)gs->month, (int)gs->day);
    }
    else if (id == ROW_TIME) {
        wchar_t tz[16];
        settings_format_tz(settings_round(gs->timezone * 60.0f), tz, sizeof(tz) / sizeof(tz[0]));
        swprintf(out, n, L"%02d:%02d  UTC%ls", (int)gs->hour, (int)gs->minute, tz);
    }
    else if (id == ROW_LON || id == ROW_LAT) {
        float v = (id == ROW_LON) ? gs->longitude : gs->latitude;
        int32_t total = settings_round((v < 0.0f ? -v : v) * 3600.0f);
        int32_t deg = total / 3600;
        int32_t rem = total % 3600;
        const wchar_t *hemi;
        if (id == ROW_LON) hemi = (v < 0.0f) ? L"W" : L"E";
        else               hemi = (v < 0.0f) ? L"S" : L"N";
        if (id == ROW_LON) {
            swprintf(out, n, L"%ls %03d\u00B0%02d\u2032%02d\u2033", hemi, deg, rem / 60, rem % 60);
        }
        else {
            swprintf(out, n, L"%ls %02d\u00B0%02d\u2032%02d\u2033", hemi, deg, rem / 60, rem % 60);
        }
    }
    else if (id == ROW_BRIGHT) {
        swprintf(out, n, L"%d%%", (int)((gs->brightness * 100) / 255));
    }
    else if (id == ROW_VOLUME) {
        swprintf(out, n, L"%d%%", (int)((gs->volume * 100) / 255));
    }
    else if (id == ROW_AUTOOFF) {
        if (gs->auto_shutdown_minutes <= 0) wcscpy(out, L"关闭");
        else swprintf(out, n, L"%d 分钟", (int)gs->auto_shutdown_minutes);
    }
    else if (id == ROW_LLM) {
        wcscpy(out, (gs->llm_enable_observation != 0) ? L"开" : L"关");
    }
    else {
        out[0] = L'\0';
    }
}

static void settings_draw_header(Global_State *gs, const UiSettingsPalette *p,
                                 const wchar_t *title, const wchar_t *action,
                                 int32_t action_kind, int32_t center_title) {
    Nano_GFX *gfx = gs->gfx;
    int32_t W = (int32_t)gfx->width;

    if (center_title) {
        // 编辑器：标题居中，左右两侧分别留给「取消」「保存」
        settings_text_centered(gfx, GFX_FONT_ALPHA_16, title, W / 2, 16, p->text, 1);
    }
    else {
        // 菜单：标题左侧强调色竖条
        settings_round_rect(gfx, 14, 11, 3, 14, 1, p->accent, 1);
        settings_text(gfx, GFX_FONT_ALPHA_16, title, 24, 7, p->text, 1);
    }

    // 右侧动作胶囊（设置=返回；编辑器=取消/保存，由调用方用 action_kind 区分配色）
    int32_t bw = 58, bh = 24, bx = W - 10 - bw, by = 4;
    const uint8_t *bg = p->surface;
    const uint8_t *fg = p->text;
    if (action_kind == 1) { bg = p->accent; fg = p->on_accent; }
    else if (action_kind == 2) { bg = p->off_soft; fg = p->off; }
    settings_round_rect(gfx, bx, by, bw, bh, 12, bg, 1);
    settings_text_centered(gfx, GFX_FONT_ALPHA_12, action, bx + bw / 2, by + bh / 2, fg, 1);

    // 页眉分隔线
    gfx_draw_rectangle(gfx, 0, ST_HEADER_H - 1, (uint32_t)W, 1,
                       p->card_edge[0], p->card_edge[1], p->card_edge[2], 1);
}

static void settings_draw_chip(Nano_GFX *gfx, int32_t x, int32_t y, int32_t w, int32_t h,
                               const wchar_t *text, int32_t selected, int32_t pressed,
                               const UiSettingsPalette *p) {
    const uint8_t *bg;
    const uint8_t *fg;
    if (selected) {
        bg = pressed ? p->accent_ink : p->accent;
        fg = p->on_accent;
    }
    else {
        bg = pressed ? p->surface_press : p->surface;
        fg = p->dim;
    }
    settings_round_rect(gfx, x, y, w, h, h / 2, bg, 1);
    settings_text_centered(gfx, GFX_FONT_ALPHA_12, text, x + w / 2, y + h / 2, fg, 1);
}

// ===============================================================================
// 设置菜单：私有交互状态
// ===============================================================================

static SettingsHit s_menu_press;        // 当前按下目标
static int32_t s_menu_touch_active = 0;
static int32_t s_menu_touch_down_x = 0;
static int32_t s_menu_touch_down_y = 0;
static int32_t s_menu_dragging = 0;
static int32_t s_menu_scroll_at_down = 0;
static int32_t s_menu_repeat_started = 0;
static uint64_t s_menu_repeat_next_ms = 0;

static void settings_menu_hit_test(Global_State *gs, const SettingsMenuLayout *L,
                                   int32_t x, int32_t y, SettingsHit *out) {
    memset(out, 0, sizeof(*out));
    out->kind = ST_HIT_NONE;
    out->row = -1;
    out->chip = -1;

    int32_t W = (int32_t)gs->gfx->width;
    // 页眉返回
    if (y < ST_HEADER_H && x >= W - 68 && x < W - 10) {
        out->kind = ST_HIT_BACK;
        return;
    }
    if (y < L->content_y0 || y >= L->content_y1) return;

    for (int32_t i = 0; i < ROW_COUNT; i++) {
        int32_t id = SETTINGS_ROWS[i].id;
        int32_t sy = settings_row_screen_y(L, id);
        int32_t sh = L->row_h[id];
        if (y < sy || y >= sy + sh) continue;

        out->row = id;
        if (id == ROW_BRIGHT || id == ROW_VOLUME || id == ROW_AUTOOFF) {
            int32_t minus_x, val_x, plus_x, btn_y;
            settings_stepper_geom(gs, &minus_x, &val_x, &plus_x, &btn_y, sy, sh);
            (void)val_x;
            if (y >= btn_y && y < btn_y + ST_STEP_H) {
                if (x >= minus_x && x < minus_x + ST_STEP_BTN_W) {
                    out->kind = ST_HIT_MINUS;
                    return;
                }
                if (x >= plus_x && x < plus_x + ST_STEP_BTN_W) {
                    out->kind = ST_HIT_PLUS;
                    return;
                }
            }
            out->kind = ST_HIT_NONE;
            return;
        }
        if (settings_chip_count(id) > 0) {
            int32_t cx[8], cw[8];
            int32_t n = settings_chip_count(id);
            settings_chip_geom(gs, id, cx, cw);
            int32_t chip_y = sy + (sh - 22) / 2;
            if (y >= chip_y && y < chip_y + 22) {
                for (int32_t c = 0; c < n; c++) {
                    if (x >= cx[c] && x < cx[c] + cw[c]) {
                        out->kind = ST_HIT_CHIP;
                        out->chip = c;
                        return;
                    }
                }
            }
            out->kind = ST_HIT_NONE;
            return;
        }
        if (id == ROW_IMU) {
            out->kind = ST_HIT_NONE;
            return;
        }
        // 日期/时间/经纬度 / 开关：整行可点
        out->kind = ST_HIT_ROW;
        return;
    }
}

// ===============================================================================
// 设置菜单：动作
// ===============================================================================

static void settings_apply_chip(Global_State *gs, int32_t id, int32_t idx) {
    switch (id) {
        case ROW_THEME:
            gs->ui_color_style = (idx == 0) ? UI_COLOR_LIGHT : UI_COLOR_DARK;
            break;
        case ROW_CLOCK:
            gs->splash_clock_style = (idx == 0) ? SPLASH_CLOCK_SMALL : SPLASH_CLOCK_BIG;
            break;
        case ROW_KEYFB:
            gs->key_feedback_mode = idx & 3;
            break;
        case ROW_TTS:
            gs->tts_req_mode = settings_clamp(idx, 0, 2);
            break;
        case ROW_ASR:
            gs->is_auto_submit_after_asr = (idx == 0) ? 0 : 1;
            break;
        default:
            break;
    }
}

static void settings_menu_cycle_key(Global_State *gs, int32_t key_code) {
    switch (key_code) {
        case NANO_KEY_9:
            gs->is_auto_submit_after_asr = (gs->is_auto_submit_after_asr == 0) ? 1 : 0;
            break;
        case NANO_KEY_8:
            gs->tts_req_mode = (settings_clamp(gs->tts_req_mode, 0, 2) + 1) % 3;
            break;
        case NANO_KEY_7:
            gs->llm_enable_observation = (gs->llm_enable_observation == 0) ? 1 : 0;
            break;
        case NANO_KEY_0:
            gs->ui_color_style = (gs->ui_color_style == UI_COLOR_LIGHT) ? UI_COLOR_DARK : UI_COLOR_LIGHT;
            break;
        case NANO_KEY_left:
            gs->splash_clock_style = (gs->splash_clock_style == SPLASH_CLOCK_SMALL)
                ? SPLASH_CLOCK_BIG : SPLASH_CLOCK_SMALL;
            break;
        case NANO_KEY_right:
            gs->key_feedback_mode = (gs->key_feedback_mode + 1) & 3;
            break;
        case NANO_KEY_ctrl: {
            static const int32_t OPTIONS[] = {0, 1, 2, 3, 5, 10, 20, 30, 60};
            const int32_t n = (int32_t)(sizeof(OPTIONS) / sizeof(OPTIONS[0]));
            int32_t idx = 0;
            for (int32_t i = 0; i < n; i++) {
                if (OPTIONS[i] == gs->auto_shutdown_minutes) { idx = i; break; }
            }
            idx = (idx + 1) % n;
            gs->auto_shutdown_minutes = OPTIONS[idx];
            gs->auto_shutdown_deadline = (OPTIONS[idx] > 0)
                ? gs->timestamp + (uint64_t)OPTIONS[idx] * 60000ULL : 0;
            break;
        }
        default:
            break;
    }
}

static void settings_menu_adjust_stepper(Global_State *gs, int32_t id, int32_t dir) {
    if (id == ROW_BRIGHT) {
        int32_t v = gs->brightness + dir * 16;
        if (v < 16) v = 16;
        if (v > 255) v = 255;
        gs->brightness = v;
        gfx_set_brightness(gs->gfx, gs->brightness);
    }
    else if (id == ROW_VOLUME) {
        int32_t v = gs->volume + dir * 16;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        gs->volume = v;
        audio_out_set_master_volume((uint8_t)gs->volume);
    }
    else if (id == ROW_AUTOOFF) {
        static const int32_t OPTIONS[] = {0, 1, 2, 3, 5, 10, 20, 30, 60};
        const int32_t n = (int32_t)(sizeof(OPTIONS) / sizeof(OPTIONS[0]));
        int32_t idx = 0;
        for (int32_t i = 0; i < n; i++) {
            if (OPTIONS[i] == gs->auto_shutdown_minutes) { idx = i; break; }
        }
        idx = (idx + dir + n) % n;
        gs->auto_shutdown_minutes = OPTIONS[idx];
        gs->auto_shutdown_deadline = (OPTIONS[idx] > 0)
            ? gs->timestamp + (uint64_t)OPTIONS[idx] * 60000ULL : 0;
    }
}

// 返回 1 表示状态已切换（调用方不要再绘制菜单）
static int32_t settings_menu_activate(Global_State *gs, const SettingsHit *hit) {
    switch (hit->kind) {
        case ST_HIT_BACK:
            gs->STATE = STATE_SPLASH_SCREEN;
            return 1;
        case ST_HIT_CHIP:
            settings_apply_chip(gs, hit->row, hit->chip);
            return 0;
        case ST_HIT_MINUS:
            settings_menu_adjust_stepper(gs, hit->row, -1);
            return 0;
        case ST_HIT_PLUS:
            settings_menu_adjust_stepper(gs, hit->row, +1);
            return 0;
        case ST_HIT_ROW:
            switch (hit->row) {
                case ROW_DATE:
                    ui_settings_editor_enter(gs, UI_SETTINGS_EDIT_DATE);
                    return 1;
                case ROW_TIME:
                    ui_settings_editor_enter(gs, UI_SETTINGS_EDIT_TIME);
                    return 1;
                case ROW_LON:
                    ui_settings_editor_enter(gs, UI_SETTINGS_EDIT_LONGITUDE);
                    return 1;
                case ROW_LAT:
                    ui_settings_editor_enter(gs, UI_SETTINGS_EDIT_LATITUDE);
                    return 1;
                case ROW_LLM:
                    gs->llm_enable_observation = (gs->llm_enable_observation == 0) ? 1 : 0;
                    return 0;
                default:
                    return 0;
            }
        default:
            return 0;
    }
}

// ===============================================================================
// 设置菜单：绘制
// ===============================================================================

static void settings_draw_stepper(Global_State *gs, const UiSettingsPalette *p,
                                  int32_t id, int32_t sy, int32_t sh,
                                  const SettingsHit *press) {
    Nano_GFX *gfx = gs->gfx;
    int32_t minus_x, val_x, plus_x, btn_y;
    settings_stepper_geom(gs, &minus_x, &val_x, &plus_x, &btn_y, sy, sh);

    int32_t press_minus = (press != NULL && press->kind == ST_HIT_MINUS && press->row == id);
    int32_t press_plus = (press != NULL && press->kind == ST_HIT_PLUS && press->row == id);

    const uint8_t *mbg = press_minus ? p->accent : p->surface;
    const uint8_t *pfg = press_minus ? p->on_accent : p->accent_ink;
    settings_round_rect(gfx, minus_x, btn_y, ST_STEP_BTN_W, ST_STEP_H, 8, mbg, 1);
    settings_minus(gfx, minus_x + ST_STEP_BTN_W / 2, btn_y + ST_STEP_H / 2, pfg);

    const uint8_t *pbg = press_plus ? p->accent : p->surface;
    const uint8_t *pfg2 = press_plus ? p->on_accent : p->accent_ink;
    settings_round_rect(gfx, plus_x, btn_y, ST_STEP_BTN_W, ST_STEP_H, 8, pbg, 1);
    settings_plus(gfx, plus_x + ST_STEP_BTN_W / 2, btn_y + ST_STEP_H / 2, pfg2);

    wchar_t value[32];
    settings_menu_row_value(gs, id, value, sizeof(value) / sizeof(value[0]));
    settings_text_centered(gfx, GFX_FONT_ALPHA_12, value,
                           val_x + ST_STEP_VAL_W / 2, btn_y + ST_STEP_H / 2, p->text, 1);

    // 亮度/音量：底部进度条（纯展示，触控用步进按钮）
    if (id == ROW_BRIGHT || id == ROW_VOLUME) {
        int32_t meter_x = ST_CARD_X + 14;
        int32_t meter_w = (int32_t)gfx->width - 24 - meter_x;
        int32_t meter_y = sy + 30;
        if (meter_w < 20) meter_w = 20;
        settings_round_rect(gfx, meter_x, meter_y, meter_w, 6, 3, p->track, 1);
        int32_t v = (id == ROW_BRIGHT) ? gs->brightness : gs->volume;
        int32_t fill_w = meter_w * settings_clamp(v, 0, 255) / 255;
        if (fill_w > 0) {
            settings_round_rect(gfx, meter_x, meter_y, fill_w, 6, 3, p->accent, 1);
        }
        int32_t knob_x = meter_x + settings_clamp(fill_w, 4, meter_w - 4);
        settings_circle(gfx, knob_x, meter_y + 3, 5, p->knob, 1);
        settings_circle(gfx, knob_x, meter_y + 3, 3, p->accent, 1);
    }
}

static void settings_draw_row(Global_State *gs, const UiSettingsPalette *p,
                              const SettingsMenuLayout *L, int32_t id,
                              const SettingsHit *press) {
    Nano_GFX *gfx = gs->gfx;
    int32_t sy = settings_row_screen_y(L, id);
    int32_t sh = L->row_h[id];
    int32_t card_w = settings_card_width(gs);
    int32_t card_x = ST_CARD_X;

    // 只绘制与内容区相交的行
    if (sy + sh <= L->content_y0 || sy >= L->content_y1) return;

    const SettingsRowDef *def = NULL;
    for (int32_t i = 0; i < ROW_COUNT; i++) {
        if (SETTINGS_ROWS[i].id == id) { def = &SETTINGS_ROWS[i]; break; }
    }
    if (def == NULL) return;

    // 整行按下高亮（开关/进入类）
    if (press != NULL && press->kind == ST_HIT_ROW && press->row == id) {
        settings_round_rect(gfx, card_x + 2, sy + 1, card_w - 4, sh - 2, 8, p->surface_press, 1);
    }

    int32_t label_h = gfx_font_line_height(GFX_FONT_ALPHA_12);
    int32_t label_y = sy + (sh - label_h) / 2;
    settings_text(gfx, GFX_FONT_ALPHA_12, def->label, card_x + 14, label_y, p->text, 1);

    if (id == ROW_DATE || id == ROW_TIME || id == ROW_LON || id == ROW_LAT) {
        wchar_t value[48];
        settings_menu_row_value(gs, id, value, sizeof(value) / sizeof(value[0]));
        int32_t right = (int32_t)gfx->width - 24;
        settings_text_right(gfx, GFX_FONT_ALPHA_12, value, right - 16, label_y, p->text, 1);
        settings_chevron_right(gfx, right - 4, sy + sh / 2, p->dim);
    }
    else if (id == ROW_BRIGHT || id == ROW_VOLUME || id == ROW_AUTOOFF) {
        settings_draw_stepper(gs, p, id, sy, sh, press);
    }
    else if (settings_chip_count(id) > 0) {
        int32_t n = settings_chip_count(id);
        int32_t cx[8], cw[8];
        settings_chip_geom(gs, id, cx, cw);
        int32_t chip_y = sy + (sh - 22) / 2;
        int32_t sel = settings_chip_selected(gs, id);
        for (int32_t c = 0; c < n; c++) {
            const wchar_t *t = NULL;
            settings_chip_text(id, c, &t);
            int32_t pressed = (press != NULL && press->kind == ST_HIT_CHIP &&
                               press->row == id && press->chip == c);
            settings_draw_chip(gfx, cx[c], chip_y, cw[c], 22, t, c == sel, pressed, p);
        }
    }
    else if (id == ROW_IMU) {
        // 状态指示（不做可切换开关：姿态传感器由平台能力决定）
        const wchar_t *txt;
        const uint8_t *bg;
        const uint8_t *fg;
#ifdef IMU_ENABLED
        txt = L"已启用"; bg = p->on_soft; fg = p->on;
#else
        txt = L"不可用"; bg = p->off_soft; fg = p->off;
#endif
        int32_t pw = 68, ph = 22;
        int32_t px = (int32_t)gfx->width - 24 - pw;
        int32_t py = sy + (sh - ph) / 2;
        settings_round_rect(gfx, px, py, pw, ph, ph / 2, bg, 1);
        settings_text_centered(gfx, GFX_FONT_ALPHA_12, txt, px + pw / 2, py + ph / 2, fg, 1);
    }
    else if (id == ROW_LLM) {
        int32_t on = (gs->llm_enable_observation != 0);
        int32_t sx = (int32_t)gfx->width - 24 - 42;
        int32_t syy = sy + (sh - 22) / 2;
        settings_switch(gfx, sx, syy, on, p);
        settings_text_right(gfx, GFX_FONT_ALPHA_12, on ? L"开" : L"关", sx - 10,
                            sy + (sh - label_h) / 2, on ? p->on : p->off, 1);
    }
}

static void settings_draw_scrollbar(Global_State *gs, const UiSettingsPalette *p,
                                    const SettingsMenuLayout *L) {
    if (L->max_scroll <= 0) return;
    Nano_GFX *gfx = gs->gfx;
    int32_t x = (int32_t)gfx->width - 7;
    int32_t y = L->content_y0 + 3;
    int32_t h = L->content_h - 6;
    if (h < 20) return;
    settings_round_rect(gfx, x, y, 3, h, 1, p->track, 1);
    int32_t thumb_h = h * L->content_h / L->total_h;
    if (thumb_h < 20) thumb_h = 20;
    if (thumb_h > h) thumb_h = h;
    int32_t travel = h - thumb_h;
    int32_t thumb_y = y + (L->max_scroll > 0 ? travel * L->scroll / L->max_scroll : 0);
    settings_round_rect(gfx, x, thumb_y, 3, thumb_h, 1, p->accent, 150);
}

void ui_settings_menu_draw(Global_State *gs) {
    Nano_GFX *gfx = gs->gfx;
    UiSettingsPalette p;
    settings_palette(gs, &p);
    SettingsMenuLayout L;
    settings_menu_layout(gs, &L);

    settings_bg(gfx, &p);

    // 内容裁剪区：行/卡片/文字都限制在页眉与屏底之间
    gfx_set_clip(gfx, 0, L.content_y0, (int32_t)gfx->width, L.content_h);
    int32_t card_w = settings_card_width(gs);

    for (int32_t s = 0; s < SEC_COUNT; s++) {
        // 整组都在视口外（含「标题已露出但一行都没有」的情况）：跳过，
        // 避免列表底部出现一张只有上边框的空卡片。
        int32_t any_visible = 0;
        for (int32_t i = 0; i < ROW_COUNT; i++) {
            if (SETTINGS_ROWS[i].section != s) continue;
            int32_t id = SETTINGS_ROWS[i].id;
            int32_t sy = settings_row_screen_y(&L, id);
            if (sy + L.row_h[id] > L.content_y0 && sy < L.content_y1) {
                any_visible = 1;
                break;
            }
        }
        if (!any_visible) continue;

        // 分组标题
        int32_t ty = L.content_y0 + L.section_title_y[s] - L.scroll;
        if (ty + ST_TITLE_H > L.content_y0 && ty < L.content_y1) {
            settings_text(gfx, GFX_FONT_ALPHA_12, SETTINGS_SECTION_TITLES[s],
                          ST_CARD_X + 8, ty + 4, p.dim, 1);
        }

        // 分组卡片
        int32_t cy = L.content_y0 + L.section_card_y[s] - L.scroll;
        int32_t ch = L.section_card_h[s];
        if (cy + ch > L.content_y0 && cy < L.content_y1) {
            settings_round_rect(gfx, ST_CARD_X, cy, card_w, ch, ST_CARD_R, p.card_edge, 1);
            settings_round_rect(gfx, ST_CARD_X + 1, cy + 1, card_w - 2, ch - 2,
                                ST_CARD_R - 1, p.card, 1);
        }

        // 行间分隔线 + 行内容
        int32_t last_id = -1;
        for (int32_t i = 0; i < ROW_COUNT; i++) {
            if (SETTINGS_ROWS[i].section != s) continue;
            last_id = SETTINGS_ROWS[i].id;
        }
        for (int32_t i = 0; i < ROW_COUNT; i++) {
            if (SETTINGS_ROWS[i].section != s) continue;
            int32_t id = SETTINGS_ROWS[i].id;
            int32_t sy = settings_row_screen_y(&L, id);
            int32_t sh = L.row_h[id];
            if (id != last_id) {
                int32_t sep_y = sy + sh - 1;
                if (sep_y > L.content_y0 && sep_y < L.content_y1) {
                    gfx_draw_rectangle(gfx, (uint32_t)(ST_CARD_X + 14), (uint32_t)sep_y,
                                       (uint32_t)(card_w - 28), 1,
                                       p.card_edge[0], p.card_edge[1], p.card_edge[2], 1);
                }
            }
            settings_draw_row(gs, &p, &L, id, &s_menu_press);
        }
    }

    gfx_reset_clip(gfx);

    // 页眉（裁剪区外，最后绘制，避免被列表内容覆盖）
    settings_draw_header(gs, &p, L"系统设置", L"返回", 0, 0);
    settings_draw_scrollbar(gs, &p, &L);

    gfx_refresh(gfx);
}

// ===============================================================================
// 设置菜单：事件
// ===============================================================================

void ui_settings_menu_enter(Global_State *gs, int32_t reset) {
    if (reset) s_menu_scroll = 0;
    memset(&s_menu_press, 0, sizeof(s_menu_press));
    s_menu_press.row = -1;
    s_menu_press.chip = -1;
    s_menu_touch_active = 0;
    s_menu_dragging = 0;
    s_menu_repeat_started = 0;
    (void)gs;
}

void ui_settings_menu_event(Key_Event *ke, Global_State *gs) {
    // 实体键盘（下降沿 -1 短按 / -2 长按重复）
    if (ke->key_code != NANO_KEY_IDLE && (ke->key_edge == -1 || ke->key_edge == -2)) {
        int32_t key = ke->key_code;
        settings_menu_cycle_key(gs, key);

        if (key == NANO_KEY_esc) {
            gs->STATE = STATE_SPLASH_SCREEN;
            return;
        }
        if (key == NANO_KEY_1) {
            ui_settings_editor_enter(gs, UI_SETTINGS_EDIT_DATE);
            return;
        }
        if (key == NANO_KEY_2) {
            ui_settings_editor_enter(gs, UI_SETTINGS_EDIT_TIME);
            return;
        }
        if (key == NANO_KEY_4) {
            ui_settings_editor_enter(gs, UI_SETTINGS_EDIT_LONGITUDE);
            return;
        }
        if (key == NANO_KEY_5) {
            ui_settings_editor_enter(gs, UI_SETTINGS_EDIT_LATITUDE);
            return;
        }
        if (key == NANO_KEY_3) {
            settings_menu_adjust_stepper(gs, ROW_BRIGHT, +1);
        }
        else if (key == NANO_KEY_6) {
            settings_menu_adjust_stepper(gs, ROW_VOLUME, +1);
        }
        else if (key == NANO_KEY_up) {
            s_menu_scroll = settings_clamp(s_menu_scroll - 30, 0, 100000);
        }
        else if (key == NANO_KEY_down) {
            s_menu_scroll += 30;
        }
        else {
            // 数字/0/Ctrl/左右键的循环切换已在 settings_menu_cycle_key 中完成
        }
        ui_settings_menu_draw(gs);
        return;
    }

    // 触屏：按下沿
    if (ke->touch_edge & TOUCH_EDGE_DOWN) {
        s_menu_touch_active = 1;
        s_menu_touch_down_x = ke->touch_down_x;
        s_menu_touch_down_y = ke->touch_down_y;
        s_menu_dragging = 0;
        s_menu_repeat_started = 0;
        s_menu_scroll_at_down = s_menu_scroll;

        SettingsMenuLayout L;
        settings_menu_layout(gs, &L);
        settings_menu_hit_test(gs, &L, ke->touch_down_x, ke->touch_down_y, &s_menu_press);
        if (s_menu_press.kind == ST_HIT_MINUS || s_menu_press.kind == ST_HIT_PLUS) {
            s_menu_repeat_next_ms = gs->timestamp + 420;
        }
        else {
            s_menu_repeat_next_ms = 0;
        }
        ui_settings_menu_draw(gs);
    }

    // 触屏：按住（拖动滚动 / 步进器连续调整）
    if (ke->is_touching && s_menu_touch_active) {
        int32_t dx = settings_abs(ke->touch_x - s_menu_touch_down_x);
        int32_t dy = settings_abs(ke->touch_y - s_menu_touch_down_y);
        if (!s_menu_dragging && (dy > 6 || dx > 8)) {
            s_menu_dragging = 1;
            memset(&s_menu_press, 0, sizeof(s_menu_press));
            s_menu_press.kind = ST_HIT_NONE;
            s_menu_press.row = -1;
            s_menu_press.chip = -1;
        }

        if (s_menu_dragging) {
            SettingsMenuLayout L;
            settings_menu_layout(gs, &L);
            int32_t new_scroll = s_menu_scroll_at_down - (ke->touch_y - s_menu_touch_down_y);
            new_scroll = settings_clamp(new_scroll, 0, L.max_scroll);
            if (new_scroll != s_menu_scroll) {
                s_menu_scroll = new_scroll;
                ui_settings_menu_draw(gs);
            }
        }
        else if (s_menu_press.kind == ST_HIT_MINUS || s_menu_press.kind == ST_HIT_PLUS) {
            if (s_menu_repeat_next_ms != 0 && gs->timestamp >= s_menu_repeat_next_ms) {
                settings_menu_adjust_stepper(gs, s_menu_press.row,
                                             s_menu_press.kind == ST_HIT_PLUS ? +1 : -1);
                s_menu_repeat_started = 1;
                s_menu_repeat_next_ms = gs->timestamp + 90;
                ui_settings_menu_draw(gs);
            }
        }
        return;
    }

    // 触屏：松开沿
    if (ke->touch_edge & TOUCH_EDGE_UP) {
        int32_t changed_state = 0;
        // 同一帧内到达的 DOWN+UP（滚轮合成手势、极快点按）没有中间电平帧可做
        // 拖动判定：这里用按下点与松开点的总位移补判，避免把滑动误认成点按。
        if (s_menu_touch_active && !s_menu_dragging) {
            int32_t dx = settings_abs(ke->touch_x - s_menu_touch_down_x);
            int32_t dy = settings_abs(ke->touch_y - s_menu_touch_down_y);
            if (dx > 8 || dy > 6) {
                s_menu_dragging = 1;
                SettingsMenuLayout L;
                settings_menu_layout(gs, &L);
                int32_t ns = settings_clamp(s_menu_scroll_at_down -
                                            (ke->touch_y - s_menu_touch_down_y),
                                            0, L.max_scroll);
                if (ns != s_menu_scroll) {
                    s_menu_scroll = ns;
                    ui_settings_menu_draw(gs);
                }
            }
        }
        if (s_menu_touch_active && !s_menu_dragging) {
            SettingsHit up_hit;
            SettingsMenuLayout L;
            settings_menu_layout(gs, &L);
            settings_menu_hit_test(gs, &L, ke->touch_x, ke->touch_y, &up_hit);

            int32_t same = (up_hit.kind == s_menu_press.kind && up_hit.row == s_menu_press.row &&
                            up_hit.chip == s_menu_press.chip);
            if (same && s_menu_press.kind != ST_HIT_NONE) {
                if (s_menu_press.kind == ST_HIT_MINUS || s_menu_press.kind == ST_HIT_PLUS) {
                    if (!s_menu_repeat_started) {
                        settings_menu_adjust_stepper(gs, s_menu_press.row,
                                                     s_menu_press.kind == ST_HIT_PLUS ? +1 : -1);
                    }
                }
                else {
                    changed_state = settings_menu_activate(gs, &s_menu_press);
                }
            }
        }

        memset(&s_menu_press, 0, sizeof(s_menu_press));
        s_menu_press.row = -1;
        s_menu_press.chip = -1;
        s_menu_touch_active = 0;
        s_menu_dragging = 0;
        s_menu_repeat_started = 0;

        if (!changed_state) ui_settings_menu_draw(gs);
        return;
    }

    // 电平丢失（异常路径）：清理按住状态
    if (!ke->is_touching && s_menu_touch_active && !(ke->touch_edge & TOUCH_EDGE_UP)) {
        s_menu_touch_active = 0;
        s_menu_dragging = 0;
        s_menu_repeat_started = 0;
        memset(&s_menu_press, 0, sizeof(s_menu_press));
        s_menu_press.row = -1;
        s_menu_press.chip = -1;
    }
}

// ===============================================================================
// 日期/时间/经纬度编辑器
// ===============================================================================

static int32_t s_edit_type = UI_SETTINGS_EDIT_DATE;
static int32_t s_edit_focus = 0;

// 编辑中的值（进入编辑器时从 Global_State 捕获；保存时写回）
static int32_t e_year, e_month, e_day;
static int32_t e_hour, e_minute;
static int32_t e_tzmin;
static int32_t e_arcsec;   // 纬度/经度的绝对秒数（0..max）
static int32_t e_negative; // 1=西经/南纬

// 进入编辑器时的快照（撤销）
static int32_t o_year, o_month, o_day, o_hour, o_minute, o_tzmin, o_arcsec, o_negative;

// 编辑器交互状态
typedef enum {
    ED_HIT_NONE = 0,
    ED_HIT_CANCEL,
    ED_HIT_SAVE,
    ED_HIT_UP,
    ED_HIT_DOWN,
    ED_HIT_VALUE,
    ED_HIT_QUICK0,
    ED_HIT_QUICK1,
    ED_HIT_QUICK2,
} EditorHitKind;

typedef struct {
    int32_t kind;
    int32_t col;
    int32_t idx;
} EditorHit;

static EditorHit s_edit_press;
static int32_t s_edit_touch_active = 0;
static int32_t s_edit_down_x = 0;
static int32_t s_edit_down_y = 0;
static int32_t s_edit_moved = 0;
static int32_t s_edit_repeat_started = 0;
static uint64_t s_edit_repeat_next_ms = 0;

static const wchar_t *settings_edit_title(int32_t type) {
    switch (type) {
        case UI_SETTINGS_EDIT_DATE: return L"日期";
        case UI_SETTINGS_EDIT_TIME: return L"时间与时区";
        case UI_SETTINGS_EDIT_LONGITUDE: return L"经度";
        default: return L"纬度";
    }
}

static int32_t settings_is_leap(int32_t y) {
    return ((y % 4 == 0) && (y % 100 != 0)) || (y % 400 == 0);
}

static int32_t settings_days_in_month(int32_t y, int32_t m) {
    static const int32_t DAYS[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (m < 1) m = 1;
    if (m > 12) m = 12;
    if (m == 2 && settings_is_leap(y)) return 29;
    return DAYS[m - 1];
}

static int32_t settings_arcsec_max(int32_t type) {
    return (type == UI_SETTINGS_EDIT_LONGITUDE ? 180 : 90) * 3600;
}

static void settings_edit_capture(Global_State *gs, int32_t type) {
    e_year = gs->year;
    e_month = gs->month;
    e_day = gs->day;
    e_hour = gs->hour;
    e_minute = gs->minute;
    e_tzmin = settings_round(gs->timezone * 60.0f);
    e_tzmin = settings_clamp(e_tzmin, -12 * 60, 14 * 60);

    e_negative = 0;
    e_arcsec = 0;
    if (type == UI_SETTINGS_EDIT_LONGITUDE || type == UI_SETTINGS_EDIT_LATITUDE) {
        float v = (type == UI_SETTINGS_EDIT_LONGITUDE) ? gs->longitude : gs->latitude;
        if (v < 0.0f) { e_negative = 1; v = -v; }
        e_arcsec = settings_round(v * 3600.0f);
        e_arcsec = settings_clamp(e_arcsec, 0, settings_arcsec_max(type));
    }

    o_year = e_year; o_month = e_month; o_day = e_day;
    o_hour = e_hour; o_minute = e_minute; o_tzmin = e_tzmin;
    o_arcsec = e_arcsec; o_negative = e_negative;
}

static void settings_edit_clamp_day(void) {
    if (e_month < 1) e_month = 1;
    if (e_month > 12) e_month = 12;
    int32_t dim = settings_days_in_month(e_year, e_month);
    if (e_day < 1) e_day = 1;
    if (e_day > dim) e_day = dim;
}

static void settings_edit_adjust(Global_State *gs, int32_t col, int32_t dir) {
    if (s_edit_type == UI_SETTINGS_EDIT_DATE) {
        if (col == 0) {
            e_year = settings_clamp(e_year + dir, 1970, 2099);
            settings_edit_clamp_day();
        }
        else if (col == 1) {
            e_month = ((e_month - 1 + dir) % 12 + 12) % 12 + 1;
            settings_edit_clamp_day();
        }
        else {
            int32_t dim = settings_days_in_month(e_year, e_month);
            if (dir > 0) { e_day = (e_day >= dim) ? 1 : e_day + 1; }
            else         { e_day = (e_day <= 1) ? dim : e_day - 1; }
        }
    }
    else if (s_edit_type == UI_SETTINGS_EDIT_TIME) {
        if (col == 0) {
            e_hour = ((e_hour + dir) % 24 + 24) % 24;
        }
        else if (col == 1) {
            // 分钟步进跨 0/59 时向小时进位/借位（成熟的步进器行为）
            int32_t next = e_minute + dir;
            if (next >= 60) {
                e_minute = 0;
                e_hour = (e_hour + 1) % 24;
            }
            else if (next < 0) {
                e_minute = 59;
                e_hour = (e_hour + 23) % 24;
            }
            else {
                e_minute = next;
            }
        }
        else {
            e_tzmin = settings_clamp(e_tzmin + dir * 15, -12 * 60, 14 * 60);
        }
    }
    else {
        int32_t maxsec = settings_arcsec_max(s_edit_type);
        int32_t step = (col == 0) ? 3600 : ((col == 1) ? 60 : 1);
        e_arcsec = settings_clamp(e_arcsec + dir * step, 0, maxsec);
    }
    (void)gs;
}

static void settings_edit_quick_now(Global_State *gs) {
    if (s_edit_type == UI_SETTINGS_EDIT_DATE) {
        e_year = gs->year;
        e_month = gs->month;
        e_day = gs->day;
        settings_edit_clamp_day();
    }
    else if (s_edit_type == UI_SETTINGS_EDIT_TIME) {
        e_hour = gs->hour;
        e_minute = gs->minute;
    }
}

static void settings_edit_quick_reset(void) {
    e_year = o_year; e_month = o_month; e_day = o_day;
    e_hour = o_hour; e_minute = o_minute; e_tzmin = o_tzmin;
    e_arcsec = o_arcsec; e_negative = o_negative;
}

// 把「宿主本地墙钟时间」写入系统时钟：
//   mktime 按宿主时区把 struct tm 解释为本地时刻，gmtime 再转成 set_sys_time 需要的
//   UTC 字段。这样无论 gs->timezone（天文计算用）与宿主时区是否一致，保存后界面
//   显示的年月日时分秒都与编辑器里看到的一致；改变时区也不会把墙钟拨快/拨慢。
//   mktime 失败（极老/极新日期，或嵌入式 libc 的 TZ 支持缺失）时退回按 gs->timezone
//   做纯算术换算，保持与原实现相同的兜底行为。
static void settings_commit_local_time(int32_t y, int32_t mo, int32_t d,
                                       int32_t h, int32_t mi, int32_t s,
                                       float tz_fallback) {
    struct tm tmv;
    memset(&tmv, 0, sizeof(tmv));
    tmv.tm_year = y - 1900;
    tmv.tm_mon = mo - 1;
    tmv.tm_mday = d;
    tmv.tm_hour = h;
    tmv.tm_min = mi;
    tmv.tm_sec = s;
    tmv.tm_isdst = -1;

    time_t target = mktime(&tmv);
    if (target != (time_t)-1) {
        struct tm *utc = gmtime(&target);
        if (utc != NULL) {
            int32_t uy = utc->tm_year + 1900, um = utc->tm_mon + 1, ud = utc->tm_mday;
            int32_t uh = utc->tm_hour, umin = utc->tm_min, usec = utc->tm_sec;
            set_sys_time(uy, um, ud, uh, umin, usec);
            return;
        }
    }

    int32_t uy = 0, um = 0, ud = 0, uh = 0, umin = 0, usec = 0;
    local_time_to_utc(y, mo, d, h, mi, s, tz_fallback, &uy, &um, &ud, &uh, &umin, &usec);
    set_sys_time(uy, um, ud, uh, umin, usec);
}

static void settings_edit_save(Global_State *gs) {
    if (s_edit_type == UI_SETTINGS_EDIT_DATE) {
        // 只改日期：时分秒沿用当前时刻，避免编辑期间把时钟倒退
        settings_edit_clamp_day();
        settings_commit_local_time(e_year, e_month, e_day,
                                   gs->hour, gs->minute, gs->second, gs->timezone);
        // set_sys_time 改变了「时间戳 → 本地时间」的映射；本帧的 global_state->timestamp
        // 仍是输入事件的宿主时刻，若不在此时同步，下一帧 main_event_handler 会先用旧
        // 时间覆盖 gs 的年月日，导致菜单刚绘制出来仍是旧日期（要等下一次重绘才更新）。
        gs->timestamp = get_timestamp_in_ms();
        gs->year = e_year; gs->month = e_month; gs->day = e_day;
    }
    else if (s_edit_type == UI_SETTINGS_EDIT_TIME) {
        float tz = (float)e_tzmin / 60.0f;
        settings_commit_local_time(gs->year, gs->month, gs->day,
                                   e_hour, e_minute, 0, tz);
        gs->timestamp = get_timestamp_in_ms();
        // 墙钟时间保持编辑器中的值；timezone 单独保存，供天文计算使用。
        gs->hour = e_hour; gs->minute = e_minute; gs->second = 0;
        gs->timezone = tz;
    }
    else {
        float v = (float)e_arcsec / 3600.0f;
        if (e_negative) v = -v;
        if (s_edit_type == UI_SETTINGS_EDIT_LONGITUDE) gs->longitude = v;
        else gs->latitude = v;
    }
    gs->STATE = STATE_SETTING_MENU;
}

static void settings_editor_layout(Global_State *gs, int32_t *col_x, int32_t *col_w,
                                   int32_t *preview_y, int32_t *label_y, int32_t *up_y,
                                   int32_t *val_y, int32_t *down_y, int32_t *quick_y,
                                   int32_t *hint_y) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t H = (int32_t)gs->gfx->height;
    int32_t margin = 12;
    int32_t gap = 7;
    int32_t cw = (W - margin * 2 - gap * 2) / 3;
    for (int32_t i = 0; i < 3; i++) {
        col_x[i] = margin + i * (cw + gap);
        col_w[i] = cw;
    }
    *preview_y = 36;
    *quick_y = H - 44;
    *hint_y = H - 18;
    *down_y = *quick_y - 30;
    *val_y = *down_y - 42;
    *up_y = *val_y - 30;
    *label_y = *up_y - 16;
}

static void settings_edit_preview_text(Global_State *gs, wchar_t *out, size_t n) {
    (void)gs;
    if (s_edit_type == UI_SETTINGS_EDIT_DATE) {
        swprintf(out, n, L"%04d-%02d-%02d", e_year, e_month, e_day);
    }
    else if (s_edit_type == UI_SETTINGS_EDIT_TIME) {
        wchar_t tz[16];
        settings_format_tz(e_tzmin, tz, sizeof(tz) / sizeof(tz[0]));
        swprintf(out, n, L"%02d:%02d   UTC%ls", e_hour, e_minute, tz);
    }
    else {
        int32_t deg = e_arcsec / 3600;
        int32_t rem = e_arcsec % 3600;
        const wchar_t *hemi;
        if (s_edit_type == UI_SETTINGS_EDIT_LONGITUDE) hemi = e_negative ? L"W" : L"E";
        else hemi = e_negative ? L"S" : L"N";
        if (s_edit_type == UI_SETTINGS_EDIT_LONGITUDE) {
            swprintf(out, n, L"%ls %03d\u00B0%02d\u2032%02d\u2033", hemi, deg, rem / 60, rem % 60);
        }
        else {
            swprintf(out, n, L"%ls %02d\u00B0%02d\u2032%02d\u2033", hemi, deg, rem / 60, rem % 60);
        }
    }
}

static void settings_edit_col_text(int32_t col, wchar_t *out, size_t n) {
    if (s_edit_type == UI_SETTINGS_EDIT_DATE) {
        if (col == 0) swprintf(out, n, L"%04d", e_year);
        else if (col == 1) swprintf(out, n, L"%02d", e_month);
        else swprintf(out, n, L"%02d", e_day);
    }
    else if (s_edit_type == UI_SETTINGS_EDIT_TIME) {
        if (col == 0) swprintf(out, n, L"%02d", e_hour);
        else if (col == 1) swprintf(out, n, L"%02d", e_minute);
        else settings_format_tz(e_tzmin, out, n);
    }
    else {
        int32_t deg = e_arcsec / 3600;
        int32_t rem = e_arcsec % 3600;
        if (col == 0) {
            swprintf(out, n, (s_edit_type == UI_SETTINGS_EDIT_LONGITUDE) ? L"%03d" : L"%02d", deg);
        }
        else if (col == 1) swprintf(out, n, L"%02d", rem / 60);
        else swprintf(out, n, L"%02d", rem % 60);
    }
}

static void settings_edit_col_label(int32_t col, const wchar_t **out) {
    if (s_edit_type == UI_SETTINGS_EDIT_DATE) {
        *out = (col == 0) ? L"年" : ((col == 1) ? L"月" : L"日");
    }
    else if (s_edit_type == UI_SETTINGS_EDIT_TIME) {
        *out = (col == 0) ? L"时" : ((col == 1) ? L"分" : L"时区");
    }
    else {
        *out = (col == 0) ? L"度" : ((col == 1) ? L"分" : L"秒");
    }
}

static void settings_edit_quick_text(int32_t idx, const wchar_t **out, int32_t *selected) {
    *selected = 0;
    if (s_edit_type == UI_SETTINGS_EDIT_DATE || s_edit_type == UI_SETTINGS_EDIT_TIME) {
        *out = (idx == 0) ? L"此刻" : L"撤销";
    }
    else {
        if (idx == 0) {
            *out = (s_edit_type == UI_SETTINGS_EDIT_LONGITUDE) ? L"东经" : L"北纬";
            *selected = (e_negative == 0);
        }
        else if (idx == 1) {
            *out = (s_edit_type == UI_SETTINGS_EDIT_LONGITUDE) ? L"西经" : L"南纬";
            *selected = (e_negative != 0);
        }
        else {
            *out = L"撤销";
        }
    }
}

static int32_t settings_edit_quick_count(void) {
    return (s_edit_type == UI_SETTINGS_EDIT_DATE || s_edit_type == UI_SETTINGS_EDIT_TIME) ? 2 : 3;
}

static void settings_edit_quick_geom(Global_State *gs, int32_t idx, int32_t *x, int32_t *w) {
    int32_t W = (int32_t)gs->gfx->width;
    int32_t n = settings_edit_quick_count();
    if (n == 2) {
        int32_t gap = 12;
        int32_t cw = (W - 24 - gap) / 2;
        *x = 12 + idx * (cw + gap);
        *w = cw;
    }
    else {
        int32_t gap = 10;
        int32_t cw = (W - 24 - gap * 2) / 3;
        *x = 12 + idx * (cw + gap);
        *w = cw;
    }
}

void ui_settings_editor_enter(Global_State *gs, int32_t edit_type) {
    s_edit_type = edit_type;
    s_edit_focus = 0;
    settings_edit_capture(gs, edit_type);
    memset(&s_edit_press, 0, sizeof(s_edit_press));
    s_edit_press.kind = ED_HIT_NONE;
    s_edit_press.col = -1;
    s_edit_press.idx = -1;
    s_edit_touch_active = 0;
    s_edit_moved = 0;
    s_edit_repeat_started = 0;
    gs->STATE = STATE_SETTING_INPUT;
}

void ui_settings_editor_draw(Global_State *gs) {
    Nano_GFX *gfx = gs->gfx;
    UiSettingsPalette p;
    settings_palette(gs, &p);

    int32_t col_x[3], col_w[3];
    int32_t preview_y, label_y, up_y, val_y, down_y, quick_y, hint_y;
    settings_editor_layout(gs, col_x, col_w, &preview_y, &label_y, &up_y,
                           &val_y, &down_y, &quick_y, &hint_y);

    int32_t W = (int32_t)gfx->width;

    settings_bg(gfx, &p);

    // 页眉：取消 / 标题 / 保存
    settings_draw_header(gs, &p, settings_edit_title(s_edit_type), L"保存", 1, 1);
    // 左侧“取消”胶囊（覆盖在页眉左侧）
    settings_round_rect(gfx, 10, 4, 58, 24, 12, p.off_soft, 1);
    settings_text_centered(gfx, GFX_FONT_ALPHA_12, L"取消", 10 + 29, 4 + 12, p.off, 1);

    // 预览卡片
    int32_t card_x = 12, card_w = W - 24;
    settings_round_rect(gfx, card_x, preview_y, card_w, 40, 12, p.card_edge, 1);
    settings_round_rect(gfx, card_x + 1, preview_y + 1, card_w - 2, 38, 11, p.card, 1);
    settings_round_rect(gfx, card_x + 12, preview_y + 11, 3, 18, 1, p.accent, 1);
    wchar_t preview[64];
    settings_edit_preview_text(gs, preview, sizeof(preview) / sizeof(preview[0]));
    settings_text_centered(gfx, GFX_FONT_ALPHA_16, preview, W / 2, preview_y + 20, p.text, 1);

    // 三列步进器
    for (int32_t c = 0; c < 3; c++) {
        int32_t x = col_x[c];
        int32_t w = col_w[c];

        const wchar_t *lab;
        settings_edit_col_label(c, &lab);
        settings_text_centered(gfx, GFX_FONT_ALPHA_12, lab, x + w / 2, label_y + 7, p.dim, 1);

        int32_t pressed_up = (s_edit_press.kind == ED_HIT_UP && s_edit_press.col == c);
        int32_t pressed_down = (s_edit_press.kind == ED_HIT_DOWN && s_edit_press.col == c);
        settings_round_rect(gfx, x, up_y, w, 24, 8,
                            pressed_up ? p.accent : p.surface, 1);
        settings_chevron_up(gfx, x + w / 2, up_y + 12, pressed_up ? p.on_accent : p.accent_ink);
        settings_round_rect(gfx, x, down_y, w, 24, 8,
                            pressed_down ? p.accent : p.surface, 1);
        settings_chevron_down(gfx, x + w / 2, down_y + 12, pressed_down ? p.on_accent : p.accent_ink);

        // 值框：聚焦列用强调色描边
        int32_t focused = (s_edit_focus == c);
        const uint8_t *edge = focused ? p.accent : p.card_edge;
        settings_round_rect(gfx, x, val_y, w, 38, 10, edge, 1);
        settings_round_rect(gfx, x + 1, val_y + 1, w - 2, 36, 9,
                            (s_edit_press.kind == ED_HIT_VALUE && s_edit_press.col == c)
                                ? p.surface_press : p.card, 1);
        wchar_t val[32];
        settings_edit_col_text(c, val, sizeof(val) / sizeof(val[0]));
        settings_text_centered(gfx, GFX_FONT_ALPHA_16, val, x + w / 2, val_y + 19, p.text, 1);
    }

    // 快捷操作行
    int32_t qn = settings_edit_quick_count();
    for (int32_t i = 0; i < qn; i++) {
        int32_t qx, qw;
        settings_edit_quick_geom(gs, i, &qx, &qw);
        const wchar_t *t = NULL;
        int32_t selected = 0;
        settings_edit_quick_text(i, &t, &selected);
        int32_t pressed = (s_edit_press.kind == (ED_HIT_QUICK0 + i));
        if (s_edit_type == UI_SETTINGS_EDIT_DATE || s_edit_type == UI_SETTINGS_EDIT_TIME) {
            // 「此刻」=主操作（强调浅底），「撤销」=中性
            if (i == 0) {
                settings_round_rect(gfx, qx, quick_y, qw, 24, 12,
                                    pressed ? p.accent : p.accent_soft, 1);
                settings_text_centered(gfx, GFX_FONT_ALPHA_12, t, qx + qw / 2, quick_y + 12,
                                       pressed ? p.on_accent : p.accent_ink, 1);
            }
            else {
                settings_round_rect(gfx, qx, quick_y, qw, 24, 12,
                                    pressed ? p.surface_press : p.surface, 1);
                settings_text_centered(gfx, GFX_FONT_ALPHA_12, t, qx + qw / 2, quick_y + 12,
                                       p.dim, 1);
            }
        }
        else {
            if (selected) {
                settings_round_rect(gfx, qx, quick_y, qw, 24, 12,
                                    pressed ? p.accent_ink : p.accent, 1);
                settings_text_centered(gfx, GFX_FONT_ALPHA_12, t, qx + qw / 2, quick_y + 12,
                                       p.on_accent, 1);
            }
            else if (i == 2) {
                settings_round_rect(gfx, qx, quick_y, qw, 24, 12,
                                    pressed ? p.surface_press : p.surface, 1);
                settings_text_centered(gfx, GFX_FONT_ALPHA_12, t, qx + qw / 2, quick_y + 12,
                                       p.dim, 1);
            }
            else {
                settings_round_rect(gfx, qx, quick_y, qw, 24, 12,
                                    pressed ? p.surface_press : p.surface, 1);
                settings_text_centered(gfx, GFX_FONT_ALPHA_12, t, qx + qw / 2, quick_y + 12,
                                       p.text, 1);
            }
        }
    }

    // 底部提示
    const wchar_t *hint = L"长按箭头可连续调整";
    if (s_edit_type == UI_SETTINGS_EDIT_TIME) hint = L"时区每次调整 15 分钟";
    settings_text_centered(gfx, GFX_FONT_ALPHA_12, hint, W / 2, hint_y + 7, p.dim, 1);

    gfx_refresh(gfx);
}

static void settings_edit_hit_test(Global_State *gs, int32_t x, int32_t y, EditorHit *out) {
    memset(out, 0, sizeof(*out));
    out->kind = ED_HIT_NONE;
    out->col = -1;
    out->idx = -1;

    int32_t col_x[3], col_w[3];
    int32_t preview_y, label_y, up_y, val_y, down_y, quick_y, hint_y;
    settings_editor_layout(gs, col_x, col_w, &preview_y, &label_y, &up_y,
                           &val_y, &down_y, &quick_y, &hint_y);
    (void)preview_y; (void)label_y; (void)hint_y;

    int32_t W = (int32_t)gs->gfx->width;
    if (y < 32) {
        if (x >= 10 && x < 68) { out->kind = ED_HIT_CANCEL; return; }
        if (x >= W - 68 && x < W - 10) { out->kind = ED_HIT_SAVE; return; }
    }
    for (int32_t c = 0; c < 3; c++) {
        if (x >= col_x[c] && x < col_x[c] + col_w[c]) {
            if (y >= up_y && y < up_y + 24) { out->kind = ED_HIT_UP; out->col = c; return; }
            if (y >= down_y && y < down_y + 24) { out->kind = ED_HIT_DOWN; out->col = c; return; }
            if (y >= val_y && y < val_y + 38) { out->kind = ED_HIT_VALUE; out->col = c; return; }
        }
    }
    int32_t qn = settings_edit_quick_count();
    for (int32_t i = 0; i < qn; i++) {
        int32_t qx, qw;
        settings_edit_quick_geom(gs, i, &qx, &qw);
        if (y >= quick_y && y < quick_y + 24 && x >= qx && x < qx + qw) {
            out->kind = ED_HIT_QUICK0 + i;
            out->idx = i;
            return;
        }
    }
}

void ui_settings_editor_event(Key_Event *ke, Global_State *gs) {
    // 实体键盘
    if (ke->key_code != NANO_KEY_IDLE && (ke->key_edge == -1 || ke->key_edge == -2)) {
        int32_t key = ke->key_code;
        if (key == NANO_KEY_esc) {
            gs->STATE = STATE_SETTING_MENU;
            return;
        }
        if (key == NANO_KEY_enter) {
            settings_edit_save(gs);
            return;
        }
        if (key == NANO_KEY_left) {
            s_edit_focus = (s_edit_focus + 2) % 3;
            ui_settings_editor_draw(gs);
            return;
        }
        if (key == NANO_KEY_right) {
            s_edit_focus = (s_edit_focus + 1) % 3;
            ui_settings_editor_draw(gs);
            return;
        }
        if (key == NANO_KEY_up) {
            settings_edit_adjust(gs, s_edit_focus, +1);
            ui_settings_editor_draw(gs);
            return;
        }
        if (key == NANO_KEY_down) {
            settings_edit_adjust(gs, s_edit_focus, -1);
            ui_settings_editor_draw(gs);
            return;
        }
        if (key == NANO_KEY_ctrl && (s_edit_type == UI_SETTINGS_EDIT_LONGITUDE ||
                                     s_edit_type == UI_SETTINGS_EDIT_LATITUDE)) {
            e_negative = e_negative ? 0 : 1;
            ui_settings_editor_draw(gs);
            return;
        }
        return;
    }

    // 触屏按下
    if (ke->touch_edge & TOUCH_EDGE_DOWN) {
        s_edit_touch_active = 1;
        s_edit_down_x = ke->touch_down_x;
        s_edit_down_y = ke->touch_down_y;
        s_edit_moved = 0;
        s_edit_repeat_started = 0;
        settings_edit_hit_test(gs, ke->touch_down_x, ke->touch_down_y, &s_edit_press);
        if (s_edit_press.kind == ED_HIT_UP || s_edit_press.kind == ED_HIT_DOWN) {
            s_edit_focus = s_edit_press.col;
            s_edit_repeat_next_ms = gs->timestamp + 420;
        }
        else {
            s_edit_repeat_next_ms = 0;
        }
        ui_settings_editor_draw(gs);
    }

    // 按住：连续调整
    if (ke->is_touching && s_edit_touch_active) {
        if (!s_edit_moved &&
            (settings_abs(ke->touch_x - s_edit_down_x) > 8 ||
             settings_abs(ke->touch_y - s_edit_down_y) > 6)) {
            s_edit_moved = 1;
            memset(&s_edit_press, 0, sizeof(s_edit_press));
            s_edit_press.kind = ED_HIT_NONE;
            s_edit_press.col = -1;
            s_edit_press.idx = -1;
        }
        if (!s_edit_moved &&
            (s_edit_press.kind == ED_HIT_UP || s_edit_press.kind == ED_HIT_DOWN) &&
            s_edit_repeat_next_ms != 0 && gs->timestamp >= s_edit_repeat_next_ms) {
            settings_edit_adjust(gs, s_edit_press.col,
                                 (s_edit_press.kind == ED_HIT_UP) ? +1 : -1);
            s_edit_repeat_started = 1;
            s_edit_repeat_next_ms = gs->timestamp + 90;
            ui_settings_editor_draw(gs);
        }
        return;
    }

    // 松开
    if (ke->touch_edge & TOUCH_EDGE_UP) {
        int32_t changed_state = 0;
        // 同一帧 DOWN+UP 的补判（见设置菜单同名注释）：位移超过阈值即视为拖动，
        // 编辑器不滚动，只需取消本次点按。
        if (s_edit_touch_active && !s_edit_moved) {
            if (settings_abs(ke->touch_x - s_edit_down_x) > 8 ||
                settings_abs(ke->touch_y - s_edit_down_y) > 6) {
                s_edit_moved = 1;
            }
        }
        if (s_edit_touch_active && !s_edit_moved) {
            EditorHit up_hit;
            settings_edit_hit_test(gs, ke->touch_x, ke->touch_y, &up_hit);
            int32_t same = (up_hit.kind == s_edit_press.kind && up_hit.col == s_edit_press.col);
            if (same && s_edit_press.kind != ED_HIT_NONE) {
                switch (s_edit_press.kind) {
                    case ED_HIT_CANCEL:
                        gs->STATE = STATE_SETTING_MENU;
                        changed_state = 1;
                        break;
                    case ED_HIT_SAVE:
                        settings_edit_save(gs);
                        changed_state = 1;
                        break;
                    case ED_HIT_UP:
                    case ED_HIT_DOWN:
                        if (!s_edit_repeat_started) {
                            settings_edit_adjust(gs, s_edit_press.col,
                                                 (s_edit_press.kind == ED_HIT_UP) ? +1 : -1);
                        }
                        break;
                    case ED_HIT_VALUE:
                        s_edit_focus = s_edit_press.col;
                        break;
                    case ED_HIT_QUICK0:
                    case ED_HIT_QUICK1:
                    case ED_HIT_QUICK2: {
                        int32_t idx = s_edit_press.kind - ED_HIT_QUICK0;
                        if (s_edit_type == UI_SETTINGS_EDIT_DATE || s_edit_type == UI_SETTINGS_EDIT_TIME) {
                            if (idx == 0) settings_edit_quick_now(gs);
                            else settings_edit_quick_reset();
                        }
                        else {
                            if (idx == 0) e_negative = 0;
                            else if (idx == 1) e_negative = 1;
                            else settings_edit_quick_reset();
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        memset(&s_edit_press, 0, sizeof(s_edit_press));
        s_edit_press.kind = ED_HIT_NONE;
        s_edit_press.col = -1;
        s_edit_press.idx = -1;
        s_edit_touch_active = 0;
        s_edit_moved = 0;
        s_edit_repeat_started = 0;

        if (!changed_state) ui_settings_editor_draw(gs);
        return;
    }

    if (!ke->is_touching && s_edit_touch_active && !(ke->touch_edge & TOUCH_EDGE_UP)) {
        s_edit_touch_active = 0;
        s_edit_moved = 0;
        s_edit_repeat_started = 0;
        memset(&s_edit_press, 0, sizeof(s_edit_press));
        s_edit_press.kind = ED_HIT_NONE;
        s_edit_press.col = -1;
        s_edit_press.idx = -1;
    }
}
