// ui_flip.c —— 番茄表 / 电子沙漏（FLIP 流体模拟的 GUI 与交互层）
//
// 本模块从 ui_app.c 剥离而来（2026-11-28），只保留「状态页」相关的一切：布局、配色、
// 控件绘制、触屏/按键事件、计时与报警状态机；流体求解与渲染仍由 flip.c 提供
// （flip_init / render_flip / flip_set_color_style，纯计算与像素写入，与本模块解耦）。
//
// 分层：
//   flip.c        FLIP 求解器 + 网格/粒子渲染（与平台无关，不含任何 UI 概念）
//   ui_flip.c     本文件：320x240 设计基准的界面（顶栏 / 舞台 / 左右仪表 / 底部控制条 /
//                 配色弹层）、触屏热点命中、计时与振动报警、自由流体/沙漏两模式
//   ui_app.c      状态机：STATE_FLIP 分支只调 ui_flip_init / render_frame / event_handler
//
// 输入契约（全屏宫格隐式软键退场后）：
//   · 触屏：所有交互都是自绘控件，直接用 touch_edge 边沿 + touch_down_x/y 命中判定，
//     松手沿结算（范式见 ui_water.c / ui_musicbox.c）；
//   · 实体键盘仅作兜底（is_soft_key == 0）：ESC 退出、←/→ 画风、1 节流、C 节流度、D 复位、2 模式。
// ===============================================================================

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "ui_flip.h"

#include "graphics.h"
#include "hal_key.h"
#include "hal_misc.h"     // set_vibration：沙漏流完后的振动报警
#include "platform.h"
#include "ui_color.h"

#include "flip.h"         // FLIP 求解器 + 渲染（flip_init / render_flip / flip_set_color_style）

#ifdef IMU_ENABLED
    #include "hal_imu.h"  // imu_read_angle：重力方向驱动流体（番茄表随姿态倾斜）
#endif

// ===============================================================================
// 番茄表（电子沙漏 / FLIP 流体）
// ===============================================================================
//
// 2026-11 GUI 重做（触屏化）：
//   · 全部交互都是屏幕自绘控件（触屏不派生任何键码）；实体键盘（is_soft_key==0）
//     保留一份兜底：ESC 退出、←/→ 画风、1 节流、C 节流度、D 复位、2 模式。
//   · 布局（320x240 设计基准，按画布宽高等比缩放）：
//       顶栏 30px   状态灯 + 标题「电子沙漏」+ 状态胶囊 + 模式胶囊 + 数码计时 + 「退出」
//       中央舞台    4:3 流体视口（与屏幕同宽高比 ⇒ 物理初始化参数 k 不变）+ 深色边框
//       左仪表      进度竖槽 + 百分比 + 余粒数（自由模式：粒子数 + 矩形槽）
//       右仪表      节流度竖槽（可拖动/点按）+ 百分比 + 当前值（自由模式：模式信息）
//       底部控制条  复位 / 画风（粒子·网格·叠加）/ 节流开关 / 配色 / 流量读数
//   · 自由流体模式：顶栏「自由/沙漏」胶囊切换（键盘 2 兜底）。自由模式用同一 FLIP
//     求解器，只把固体边界换成矩形外壁（flip_init 的 hourglass_enabled=0）、粒子初始
//     沉降在槽底，并**关闭全部节流机制**；没有流完判定/报警，计时退化为秒表。
//   · 色彩风格：底部「配色」胶囊打开模态选择器，5 档——黄（暖沙，默认）/ 蓝 / 绿 /
//     青 / 彩（原彩色：蓝粒子 + 彩虹密度网格）。风格同时驱动流体颜色（flip.c 的
//     flip_set_color_style，纯渲染）与 UI 强调色（仪表/胶囊/按钮/完成态描边）。
//   · 沙漏模式的物理与计时逻辑一行未改：dt、重力源与换算、节流自适应公式、翻转判定、
//     报警状态机、粒子流量统计全部沿用；render_flip 只是把流体画进舞台视口（center/
//     width/height 参数），视口与屏幕同宽高比，因此网格分辨率、沙漏几何、粒子数不变。
// ===============================================================================

// ---- 触屏控件标识 ----------------------------------------------------------
#define FLIP_UI_NONE       (-1)
#define FLIP_UI_EXIT       (0)
#define FLIP_UI_RESET      (1)
#define FLIP_UI_VIEW       (2)
#define FLIP_UI_THROTTLE   (3)
#define FLIP_UI_SLIDER     (4)
#define FLIP_UI_PALETTE    (5)
#define FLIP_UI_MODE       (6)

// ---- 色彩风格（与 flip.c 的 flip_set_color_style 一一对应）------------------
#define FLIP_COLOR_STYLES  (5)   // 0黄 1蓝 2绿 3青 4彩（原彩色）
static int32_t s_ui_flip_color_style = 0;
static int32_t s_ui_flip_picker_open = 0;
static int32_t s_ui_flip_picker_pressed = FLIP_UI_NONE;

// 自由流体模式：0=沙漏（沙漏实体 + 节流），1=自由流体（矩形舞台、无沙漏壁、无节流）。
// 切换时重新初始化流体（两种模式的固体边界不同），计时器重新开始。
static int32_t s_ui_flip_free_mode = 0;

// ---- 画风（与 s_ui_flip_setting_count 同一套取值）--------------------------
#define FLIP_VIEW_PARTICLE (0)   // 只看粒子
#define FLIP_VIEW_GRID     (1)   // 只看网格（密度着色）
#define FLIP_VIEW_BOTH     (2)   // 粒子 + 网格叠加

static int32_t s_ui_flip_setting_count = FLIP_VIEW_PARTICLE;
static int32_t s_ui_flip_show_particles = 1;
static int32_t s_ui_flip_show_grid = 0;
static int32_t s_ui_flip_is_throttle = 0;
static int32_t s_ui_flip_throttle = 0;
static int32_t s_ui_flip_init_throttle = 50;

static int32_t s_ui_flip_last_upper_count = -1; // 用于计算粒子流量（-1 = 尚未建立基线）
static uint64_t s_ui_flip_last_upper_count_timestamp = 0; // 用于计算粒子流量
static uint64_t s_ui_fanqie_start_timestamp = 0;
static uint64_t s_ui_fanqie_stop_timestamp = 0;
static int32_t s_ui_fanqie_is_running = 0;

static uint64_t s_ui_fanqie_alarm_count = 0;
static int32_t  s_ui_fanqie_alarm_phase = 0;      // 非阻塞报警状态机相位：0=空闲 1=鸣震段 2=静默段
static uint64_t s_ui_fanqie_alarm_phase_ts = 0;   // 当前相位起始时间戳

// 触屏控件交互状态（按下点锁存 + 松手沿结算，项目触屏范式）
static int32_t s_ui_flip_ui_pressed = FLIP_UI_NONE;
static int32_t s_ui_flip_ui_pressed_view = FLIP_VIEW_PARTICLE;
static int32_t s_ui_flip_ui_slider_active = 0;

// ---- 配色 ------------------------------------------------------------------
// 深/浅两套主题跟随全局 ui_color_style；沙漏舞台本身恒为深色（流体是自发光画面，
// 浅色主题下就是一块嵌进面板的深色窗口），只切换外部 chrome 的配色。
// 强调色（accent / accent_dim / accent_off / on_accent）不写死，由当前色彩风格派生：
// 深色主题取明亮色、浅色主题取饱和深色，保证两种主题下的对比度。
typedef struct {
    uint8_t bg_top[3];
    uint8_t bg_bot[3];
    uint8_t card_edge[3];     // 舞台外框
    uint8_t panel[3];         // 舞台面板 / 沙漏实体（同色 ⇒ 圆角接缝不可见）
    uint8_t surface[3];       // 控件底色
    uint8_t surface_press[3];
    uint8_t surface_edge[3];
    uint8_t accent[3];        // 以下四项由色彩风格派生（见 flip_ui_palette）
    uint8_t accent_dim[3];
    uint8_t accent_off[3];    // 关闭态下的节流仪表（弱化但仍可读）
    uint8_t on_accent[3];
    uint8_t text[3];
    uint8_t dim[3];
    uint8_t timer[3];
    uint8_t track[3];
    uint8_t wall_edge[3];
    uint8_t fluid_top[3];
    uint8_t fluid_bot[3];
    uint8_t separator_alpha;
} Flip_Ui_Palette;

static const uint8_t flip_accent_dark[FLIP_COLOR_STYLES][3] = {
    {255, 181, 71},   // 黄（暖沙）
    {86, 156, 255},   // 蓝
    {86, 214, 132},   // 绿
    {86, 214, 214},   // 青
    {150, 172, 196},  // 彩（流体为原彩色，UI 用低饱和浅蓝灰强调）
};
static const uint8_t flip_accent_light[FLIP_COLOR_STYLES][3] = {
    {232, 138, 31},
    {37, 99, 235},
    {22, 163, 74},
    {8, 145, 178},
    {100, 116, 139},  // 彩：浅蓝灰在亮底上太弱，改用同色相的低饱和中灰蓝
};
static const wchar_t *const flip_style_names[FLIP_COLOR_STYLES] = {
    L"黄", L"蓝", L"绿", L"青", L"彩"
};

static const uint8_t *flip_style_accent(int32_t style, int32_t is_light) {
    if (style < 0 || style >= FLIP_COLOR_STYLES) style = 0;
    return is_light ? flip_accent_light[style] : flip_accent_dark[style];
}

// 强调色上的文字色：按亮度选深/浅，亮底配深字、暗底配白字
static void flip_style_on_accent(const uint8_t *a, uint8_t out[3]) {
    int32_t lum = ((int32_t)a[0] * 299 + (int32_t)a[1] * 587 + (int32_t)a[2] * 114) / 1000;
    if (lum > 140) { out[0] = 28; out[1] = 19; out[2] = 5; }
    else { out[0] = 255; out[1] = 255; out[2] = 255; }
}

static void flip_ui_palette(Global_State *global_state, Flip_Ui_Palette *p) {
    static const Flip_Ui_Palette dark = {
        .bg_top        = {8, 11, 18},
        .bg_bot        = {17, 23, 35},
        .card_edge     = {60, 74, 100},
        .panel         = {25, 32, 46},
        .surface       = {28, 36, 50},
        .surface_press = {48, 62, 84},
        .surface_edge  = {54, 68, 92},
        .text          = {236, 242, 250},
        .dim           = {138, 152, 172},
        .timer         = {250, 246, 238},
        .track         = {38, 48, 64},
        .wall_edge     = {84, 106, 142},
        .fluid_top     = {8, 12, 19},
        .fluid_bot     = {3, 5, 9},
        .separator_alpha = 70,
    };
    static const Flip_Ui_Palette light = {
        .bg_top        = {247, 249, 253},
        .bg_bot        = {228, 235, 245},
        .card_edge     = {196, 207, 224},
        .panel         = {25, 32, 46},
        .surface       = {255, 255, 255},
        .surface_press = {224, 234, 247},
        .surface_edge  = {206, 217, 232},
        .text          = {28, 36, 50},
        .dim           = {106, 121, 140},
        .timer         = {28, 36, 50},
        .track         = {219, 227, 239},
        .wall_edge     = {84, 106, 142},
        .fluid_top     = {8, 12, 19},
        .fluid_bot     = {3, 5, 9},
        .separator_alpha = 36,
    };
    int32_t is_light = (global_state->ui_color_style == UI_COLOR_LIGHT);
    *p = is_light ? light : dark;

    const uint8_t *a = flip_style_accent(s_ui_flip_color_style, is_light);
    const uint8_t *mixc = is_light ? p->surface : p->panel;
    int32_t dim_k = is_light ? 82 : 68;   // accent_dim = accent * dim_k%
    int32_t off_w = is_light ? 52 : 38;   // accent_off = accent:底色 = off_w : (100-off_w)
    for (int32_t i = 0; i < 3; i++) {
        p->accent[i] = a[i];
        p->accent_dim[i] = (uint8_t)((int32_t)a[i] * dim_k / 100);
        p->accent_off[i] = (uint8_t)(((int32_t)a[i] * off_w
                                     + (int32_t)mixc[i] * (100 - off_w)) / 100);
    }
    flip_style_on_accent(a, p->on_accent);
}

// ---- 布局几何（绘制与命中判定共用同一份）-----------------------------------
typedef struct {
    int32_t w;
    int32_t h;
    int32_t header_h;
    int32_t title_x, title_y;
    int32_t chip_x, chip_y, chip_w, chip_h;
    int32_t mode_x, mode_y, mode_w, mode_h, mode_r;
    int32_t timer_y, timer_right;
    int32_t seg_len, seg_thick, seg_gap;
    int32_t exit_x, exit_y, exit_w, exit_h, exit_r;
    // 舞台
    int32_t card_x, card_y, card_w, card_h, card_r;
    int32_t fluid_x, fluid_y, fluid_w, fluid_h;
    // 左右仪表
    int32_t rail_y, rail_h;
    int32_t lrail_x, lrail_w, lrail_cx;
    int32_t rrail_x, rrail_w, rrail_cx;
    int32_t label_cy;
    int32_t gauge_w, gauge_h, gauge_y;
    int32_t lgauge_x, rgauge_x;
    int32_t gauge_value_cy;
    int32_t gauge_caption_cy;
    int32_t knob_r;
    // 底部控制条
    int32_t dock_y, dock_h;
    int32_t btn_y, btn_h, btn_r;
    int32_t reset_x, reset_w;
    int32_t view_x, view_w, view_cell_w;
    int32_t thr_x, thr_w;
    int32_t pal_x, pal_w;
    int32_t stat_cx, stat_y1, stat_y2;
    // 配色选择弹层（模态）
    int32_t pick_x, pick_y, pick_w, pick_h, pick_r;
    int32_t pick_title_x, pick_title_y;
    int32_t sw_x, sw_y, sw_w, sw_h, sw_gap, sw_r;
} Flip_Ui;

// ---- 布局 ------------------------------------------------------------------
static void flip_ui_layout(Nano_GFX *gfx, Flip_Ui *u) {
    memset(u, 0, sizeof(*u));
    u->w = (int32_t)gfx->width;
    u->h = (int32_t)gfx->height;

    float sx = (float)u->w / 320.0f;
    float sy = (float)u->h / 240.0f;
    float rs = (sx < sy) ? sx : sy;
#define FX(v) ((int32_t)lroundf((float)(v) * sx))
#define FY(v) ((int32_t)lroundf((float)(v) * sy))
#define FR(v) ((int32_t)lroundf((float)(v) * rs))

    // 顶栏
    u->header_h = FY(30);
    u->title_x = FX(8);
    u->title_y = FY(5);
    u->chip_h = FY(17);
    u->chip_x = FX(78);      // 标题「电子沙漏」= 4 个 16px 全角字（8..72）
    u->chip_y = FY(6);
    u->chip_w = FX(54);
    u->mode_w = FX(33);
    u->mode_h = FY(17);
    u->mode_x = u->chip_x + u->chip_w + FX(2);
    u->mode_y = u->chip_y;
    u->mode_r = u->mode_h / 2;
    u->timer_y = FY(4);
    u->seg_len = FR(8);
    if (u->seg_len < 4) u->seg_len = 4;
    u->seg_thick = FR(2);
    if (u->seg_thick < 1) u->seg_thick = 1;
    u->seg_gap = FR(2);
    if (u->seg_gap < 1) u->seg_gap = 1;
    u->exit_w = FX(46);
    u->exit_h = FY(20);
    u->exit_r = u->exit_h / 2;
    u->exit_x = u->w - FX(8) - u->exit_w;
    u->exit_y = FY(5);
    u->timer_right = u->exit_x - FX(8);

    // 中央舞台：流体视口 224x168（4:3），四周 4px 面板边框
    u->fluid_w = FX(224);
    u->fluid_h = FY(168);
    u->card_r = FR(10);
    u->card_w = u->fluid_w + 2 * FX(4);
    u->card_h = u->fluid_h + 2 * FY(4);
    u->card_x = (u->w - u->card_w) / 2;
    u->card_y = u->header_h + FY(2);
    u->fluid_x = u->card_x + FX(4);
    u->fluid_y = u->card_y + FY(4);

    // 左右仪表
    u->rail_y = u->card_y;
    u->rail_h = u->card_h;
    u->lrail_x = 0;
    u->lrail_w = u->card_x;
    u->lrail_cx = u->lrail_x + u->lrail_w / 2;
    u->rrail_x = u->card_x + u->card_w;
    u->rrail_w = u->w - u->rrail_x;
    u->rrail_cx = u->rrail_x + u->rrail_w / 2;
    u->label_cy = u->rail_y + FY(10);
    u->gauge_w = FX(10);
    if (u->gauge_w < 6) u->gauge_w = 6;
    u->gauge_h = FY(88);
    u->gauge_y = u->rail_y + FY(22);
    u->lgauge_x = u->lrail_cx - u->gauge_w / 2;
    u->rgauge_x = u->rrail_cx - u->gauge_w / 2;
    u->gauge_value_cy = u->rail_y + FY(118);
    u->gauge_caption_cy = u->rail_y + FY(138);
    u->knob_r = FR(5);
    if (u->knob_r < 3) u->knob_r = 3;

    // 底部控制条
    u->dock_y = u->card_y + u->card_h;
    u->dock_h = u->h - u->dock_y;
    u->btn_h = FY(26);
    if (u->btn_h > u->dock_h - FY(4)) u->btn_h = u->dock_h - FY(4);
    if (u->btn_h < 12) u->btn_h = 12;
    u->btn_y = u->dock_y + (u->dock_h - u->btn_h) / 2;
    u->btn_r = FR(8);
    u->reset_x = FX(6);
    u->reset_w = FX(50);
    u->view_x = FX(60);
    u->view_w = FX(90);
    u->view_cell_w = u->view_w / 3;
    u->thr_x = FX(154);
    u->thr_w = FX(50);
    u->pal_x = FX(208);
    u->pal_w = FX(46);
    u->stat_cx = FX(286);
    u->stat_y1 = u->dock_y + FY(9);
    u->stat_y2 = u->dock_y + FY(23);

    // 配色选择弹层：屏幕居中的圆角卡片（标题 + 5 个大色块）
    u->pick_w = FX(300);
    u->pick_h = FY(66);
    u->pick_x = (u->w - u->pick_w) / 2;
    u->pick_y = (u->h - u->pick_h) / 2;
    u->pick_r = FR(10);
    u->pick_title_x = u->pick_x + FX(10);
    u->pick_title_y = u->pick_y + FY(6);
    u->sw_gap = FX(4);
    u->sw_w = (u->pick_w - FX(18) - 4 * u->sw_gap) / FLIP_COLOR_STYLES;
    u->sw_h = FY(34);
    u->sw_r = FR(8);
    u->sw_x = u->pick_x + (u->pick_w - (FLIP_COLOR_STYLES * u->sw_w
                                        + (FLIP_COLOR_STYLES - 1) * u->sw_gap)) / 2;
    u->sw_y = u->pick_y + FY(26);

#undef FX
#undef FY
#undef FR
}

// ---- 绘图小工具 ------------------------------------------------------------
static void flip_ui_round(Nano_GFX *g, int32_t x, int32_t y, int32_t w, int32_t h, int32_t r,
                          const uint8_t c[3], uint8_t mode) {
    if (w <= 0 || h <= 0) return;
    int32_t rm = (w < h) ? w : h;
    rm /= 2;
    if (r < 0) r = 0;
    if (r > rm) r = rm;
    gfx_draw_rounded_rectangle(g, x, y, w, h, r, r, r, r, c[0], c[1], c[2], mode);
}

static void flip_ui_circle(Nano_GFX *g, int32_t cx, int32_t cy, int32_t r,
                           const uint8_t c[3], uint8_t mode) {
    if (r <= 0) return;
    flip_ui_round(g, cx - r, cy - r, r * 2, r * 2, r, c, mode);
}

static void flip_ui_text(Nano_GFX *g, uint32_t font, const wchar_t *s,
                         int32_t x, int32_t y_top, const uint8_t c[3], uint8_t mode) {
    gfx_font_draw_text(g, font, (wchar_t *)s, x, y_top, c[0], c[1], c[2], mode);
}

static void flip_ui_text_centered(Nano_GFX *g, uint32_t font, const wchar_t *s,
                                  int32_t cx, int32_t cy, const uint8_t c[3], uint8_t mode) {
    gfx_font_draw_text_centered(g, font, (wchar_t *)s, cx, cy, c[0], c[1], c[2], mode);
}

// 垂直渐变背景（8px 条带，条带按全局 y 对齐：与局部重绘语义一致）
static void flip_ui_bg(Nano_GFX *g, const Flip_Ui_Palette *p) {
    int32_t H = (int32_t)g->height;
    for (int32_t y = 0; y < H; y += 8) {
        int32_t ye = y + 8;
        if (ye > H) ye = H;
        int32_t t = (H > 1) ? (y * 255 / (H - 1)) : 0;
        uint8_t r = (uint8_t)(p->bg_top[0] + ((int32_t)p->bg_bot[0] - (int32_t)p->bg_top[0]) * t / 255);
        uint8_t gg = (uint8_t)(p->bg_top[1] + ((int32_t)p->bg_bot[1] - (int32_t)p->bg_top[1]) * t / 255);
        uint8_t b = (uint8_t)(p->bg_top[2] + ((int32_t)p->bg_bot[2] - (int32_t)p->bg_top[2]) * t / 255);
        gfx_draw_rectangle(g, 0, y, (uint32_t)g->width, (uint32_t)(ye - y), r, gg, b, 1);
    }
}

// 按钮：state 0=常态 1=按下 2=选中 3=关闭态（弱化）
static void flip_ui_button(Nano_GFX *g, const Flip_Ui_Palette *p,
                           int32_t x, int32_t y, int32_t w, int32_t h, int32_t r,
                           const wchar_t *label, int32_t state) {
    if (w <= 0 || h <= 0) return;
    if (state == 2) {
        flip_ui_round(g, x, y, w, h, r, p->accent, 1);
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, label, x + w / 2, y + h / 2, p->on_accent, 1);
        return;
    }
    if (state == 1) {
        flip_ui_round(g, x, y, w, h, r, p->surface_press, 1);
        flip_ui_round(g, x, y, w, h, r, p->accent, 90);
    }
    else {
        flip_ui_round(g, x, y, w, h, r, p->surface, 1);
    }
    // 1px 描边（外圈描边色 + 内圈底色，圆角一致）
    flip_ui_round(g, x, y, w, h, r, p->surface_edge, 1);
    flip_ui_round(g, x + 1, y + 1, w - 2, h - 2, r - 1,
                  (state == 1) ? p->surface_press : p->surface, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, label, x + w / 2, y + h / 2,
                          (state == 3) ? p->dim : p->text, 1);
}

// ---- 顶栏 ------------------------------------------------------------------
static void flip_ui_draw_header(Key_Event *key_event, Global_State *global_state,
                                const Flip_Ui *u, const Flip_Ui_Palette *p,
                                uint64_t elapsed_ms, int32_t running,
                                int32_t pressed) {
    Nano_GFX *g = global_state->gfx;

    // 标题
    flip_ui_text(g, GFX_FONT_ALPHA_16, L"电子沙漏", u->title_x, u->title_y, p->text, 1);

    // 状态胶囊
    int32_t chip_cy = u->chip_y + u->chip_h / 2;
    if (running) {
        flip_ui_round(g, u->chip_x, u->chip_y, u->chip_w, u->chip_h, u->chip_h / 2, p->surface, 1);
        flip_ui_round(g, u->chip_x, u->chip_y, u->chip_w, u->chip_h, u->chip_h / 2, p->surface_edge, 1);
        flip_ui_round(g, u->chip_x + 1, u->chip_y + 1, u->chip_w - 2, u->chip_h - 2,
                      u->chip_h / 2 - 1, p->surface, 1);
        flip_ui_circle(g, u->chip_x + u->chip_h / 2, chip_cy, u->chip_h / 6, p->accent, 1);
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"运行中",
                              u->chip_x + u->chip_w / 2 + u->chip_h / 4, chip_cy, p->dim, 1);
    }
    else {
        flip_ui_round(g, u->chip_x, u->chip_y, u->chip_w, u->chip_h, u->chip_h / 2, p->accent, 1);
        flip_ui_circle(g, u->chip_x + u->chip_h / 2, chip_cy, u->chip_h / 6, p->on_accent, 1);
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"已完成",
                              u->chip_x + u->chip_w / 2 + u->chip_h / 4, chip_cy, p->on_accent, 1);
    }

    // 模式切换胶囊：沙漏 ⇄ 自由流体（文字是“将要切换到的模式”）
    int32_t mode_pressed = (pressed == FLIP_UI_MODE);
    flip_ui_round(g, u->mode_x, u->mode_y, u->mode_w, u->mode_h, u->mode_r,
                  mode_pressed ? p->surface_press : p->surface, 1);
    flip_ui_round(g, u->mode_x, u->mode_y, u->mode_w, u->mode_h, u->mode_r,
                  mode_pressed ? p->accent : p->surface_edge, 1);
    flip_ui_round(g, u->mode_x + 1, u->mode_y + 1, u->mode_w - 2, u->mode_h - 2,
                  u->mode_r - 1, mode_pressed ? p->surface_press : p->surface, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, s_ui_flip_free_mode ? L"沙漏" : L"自由",
                          u->mode_x + u->mode_w / 2, u->mode_y + u->mode_h / 2, p->accent, 1);

    // 数码计时：MM:SS + 百分秒（右对齐到退出按钮左侧）
    int32_t total_sec = (int32_t)(elapsed_ms / 1000u);
    int32_t mm = total_sec / 60;
    int32_t ss = total_sec % 60;
    int32_t cs = (int32_t)(elapsed_ms % 1000u) / 10;
    wchar_t tstr[16];
    wchar_t cstr[8];
    swprintf(tstr, sizeof(tstr) / sizeof(tstr[0]), L"%02d:%02d", mm, ss);
    swprintf(cstr, sizeof(cstr) / sizeof(cstr[0]), L".%02d", cs);

    int32_t tw = 0, th = 0;
    ui_measure_7seg_string(tstr, (float)u->seg_len, (float)u->seg_thick, (float)u->seg_gap, &tw, &th);
    int32_t cw = gfx_font_measure_text(GFX_FONT_ALPHA_12, cstr);
    int32_t tx = u->timer_right - (tw + 2 * u->seg_gap + cw);
    if (tx < u->mode_x + u->mode_w + 2) tx = u->mode_x + u->mode_w + 2;
    const uint8_t *tcol = running ? p->timer : p->accent;
    ui_draw_7seg_string(key_event, global_state, tx, u->timer_y, tstr,
                        tcol[0], tcol[1], tcol[2],
                        (float)u->seg_len, (float)u->seg_thick, (float)u->seg_gap, 0,
                        &tw, &th);
    int32_t cy_top = u->timer_y + th - 13;
    flip_ui_text(g, GFX_FONT_ALPHA_12, cstr, tx + tw + 2 * u->seg_gap, cy_top, p->dim, 1);

    // 退出按钮
    int32_t exit_state = (pressed == FLIP_UI_EXIT) ? 1 : 0;
    flip_ui_round(g, u->exit_x, u->exit_y, u->exit_w, u->exit_h, u->exit_r,
                  (exit_state == 1) ? p->surface_press : p->surface, 1);
    flip_ui_round(g, u->exit_x, u->exit_y, u->exit_w, u->exit_h, u->exit_r,
                  (exit_state == 1) ? p->accent : p->surface_edge, 1);
    flip_ui_round(g, u->exit_x + 1, u->exit_y + 1, u->exit_w - 2, u->exit_h - 2,
                  u->exit_r - 1, (exit_state == 1) ? p->surface_press : p->surface, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"退出",
                          u->exit_x + u->exit_w / 2, u->exit_y + u->exit_h / 2,
                          (exit_state == 1) ? p->accent : p->text, 1);

    // 顶栏分隔线
    gfx_draw_rectangle(g, 0, u->header_h - 1, (uint32_t)u->w, 1,
                       0, 0, 0, p->separator_alpha);
}

// ---- 舞台（拟物化：金属边框 + 上下压板 + 玻璃腔体 + 跟随重力的沙池底光）-------
// 舞台在深浅两套主题下都保持深色（自发光玻璃窗）；金属/玻璃色是固定值，只有外框
// 完成态描边、瓶颈光、沙池底光跟随色彩风格。
static const uint8_t k_stage_bevel[3]    = {78, 94, 120};   // 边框倒角高光
static const uint8_t k_stage_case[3]     = {34, 43, 58};    // 壳体
static const uint8_t k_stage_plate_hi[3] = {70, 86, 112};   // 压板亮面
static const uint8_t k_stage_plate_lo[3] = {16, 21, 30};    // 压板暗面

// 平滑径向光晕（逐像素 alpha + 平方衰减）：比一圈圈同心圆更细腻，也不会有环带。
// 只在粒子绘制之前铺底（瓶颈光 / 沙池底光），面积约 1 万像素，代价可控。
static void flip_ui_radial_glow(Nano_GFX *g, int32_t cx, int32_t cy, int32_t r,
                                const uint8_t c[3], int32_t alpha_max) {
    if (r <= 1 || alpha_max <= 0) return;
    int32_t x0 = cx - r, x1 = cx + r;
    int32_t y0 = cy - r, y1 = cy + r;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= (int32_t)g->width) x1 = (int32_t)g->width - 1;
    if (y1 >= (int32_t)g->height) y1 = (int32_t)g->height - 1;
    int32_t r2 = r * r;
    for (int32_t y = y0; y <= y1; y++) {
        int32_t dy = y - cy;
        for (int32_t x = x0; x <= x1; x++) {
            int32_t dx = x - cx;
            int32_t d2 = dx * dx + dy * dy;
            if (d2 >= r2) continue;
            float t = 1.0f - sqrtf((float)d2) / (float)r;
            int32_t a = (int32_t)(alpha_max * t * t);
            if (a > 0) gfx_blend_pixel(g, (uint32_t)x, (uint32_t)y, c[0], c[1], c[2], (uint8_t)a);
        }
    }
}

static void flip_ui_draw_stage_frame(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                     int32_t finished, int32_t sand_at_bottom,
                                     int32_t free_mode) {
    // 外框：1px 深边（完成态染强调色）+ 1px 倒角高光 + 壳体
    flip_ui_round(g, u->card_x - 1, u->card_y - 1, u->card_w + 2, u->card_h + 2,
                  u->card_r + 1, finished ? p->accent_dim : p->card_edge, 1);
    flip_ui_round(g, u->card_x, u->card_y, u->card_w, u->card_h, u->card_r, k_stage_bevel, 1);
    flip_ui_round(g, u->card_x + 1, u->card_y + 1, u->card_w - 2, u->card_h - 2,
                  u->card_r - 1, k_stage_case, 1);

    // 上下压板：让沙漏看起来被夹在两条金属板之间（都落在 4px 边框内，不遮流体）
    int32_t plate_x = u->card_x + u->card_r / 2 + 1;
    int32_t plate_w = u->card_w - 2 * (u->card_r / 2 + 1);
    flip_ui_round(g, plate_x, u->card_y + 1, plate_w, 2, 1, k_stage_plate_hi, 1);
    gfx_draw_rectangle(g, plate_x, u->card_y + 3, (uint32_t)plate_w, 1,
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
    gfx_draw_rectangle(g, plate_x, u->card_y + u->card_h - 3, (uint32_t)plate_w, 1,
                       k_stage_plate_hi[0], k_stage_plate_hi[1], k_stage_plate_hi[2], 1);
    flip_ui_round(g, plate_x, u->card_y + u->card_h - 2, plate_w, 2, 1, k_stage_plate_lo, 1);

    // 侧边框 1px 倒角：左亮右暗（光从左上来），两端留出圆角区
    gfx_draw_rectangle(g, u->card_x + 2, u->card_y + u->card_r + 2, 1,
                       (uint32_t)(u->card_h - 2 * (u->card_r + 2)), 52, 64, 84, 1);
    gfx_draw_rectangle(g, u->card_x + u->card_w - 3, u->card_y + u->card_r + 2, 1,
                       (uint32_t)(u->card_h - 2 * (u->card_r + 2)), 12, 16, 24, 1);

    // 流体视口底色：上暗下更暗的竖向渐变（4 段）
    int32_t fh = u->fluid_h;
    for (int32_t i = 0; i < 4; i++) {
        int32_t y0 = u->fluid_y + fh * i / 4;
        int32_t y1 = u->fluid_y + fh * (i + 1) / 4;
        int32_t t = i * 85;
        uint8_t r = (uint8_t)(p->fluid_top[0] + ((int32_t)p->fluid_bot[0] - (int32_t)p->fluid_top[0]) * t / 255);
        uint8_t gg = (uint8_t)(p->fluid_top[1] + ((int32_t)p->fluid_bot[1] - (int32_t)p->fluid_top[1]) * t / 255);
        uint8_t b = (uint8_t)(p->fluid_top[2] + ((int32_t)p->fluid_bot[2] - (int32_t)p->fluid_top[2]) * t / 255);
        gfx_draw_rectangle(g, u->fluid_x, y0, (uint32_t)u->fluid_w, (uint32_t)(y1 - y0), r, gg, b, 1);
    }

    // 瓶颈暖光：只透在空腔里（沙漏实体绘制后会盖掉外围）；自由模式没有瓶颈
    int32_t gcx = u->fluid_x + u->fluid_w / 2;
    int32_t gcy = u->fluid_y + u->fluid_h / 2;
    int32_t rg = (u->fluid_w < u->fluid_h ? u->fluid_w : u->fluid_h) / 5;
    if (!free_mode) {
        flip_ui_radial_glow(g, gcx, gcy, rg, p->accent, 26);
    }

    // 沙池底光：跟随重力落在沙粒堆积的一侧（画在粒子之前 ⇒ 沙粒像浸在一团暖光里）
    int32_t pool_y = u->fluid_y + (sand_at_bottom ? (u->fluid_h * 74 / 100)
                                                   : (u->fluid_h * 26 / 100));
    int32_t pr = (u->fluid_w < u->fluid_h ? u->fluid_w : u->fluid_h) / 4;
    flip_ui_radial_glow(g, gcx, pool_y, pr, p->accent, 18);
}

// 沙漏实体：竖向 6 段环境光渐变（上亮下暗），内沿为 1.8px 玻璃辉光 + 1px 亮边（加色）。
static void flip_ui_draw_walls(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p) {
    static const uint8_t k_wall_hi[3]  = {32, 41, 56};
    static const uint8_t k_wall_lo[3]  = {19, 25, 37};
    static const uint8_t k_rim[3]      = {104, 132, 170};
    static const uint8_t k_rim_glow[3] = {34, 46, 66};
    const int32_t BANDS = 6;

    int32_t fx = u->fluid_x;
    int32_t fy = u->fluid_y;
    int32_t fw = u->fluid_w;
    int32_t fh = u->fluid_h;
    int32_t fxe = fx + fw;
    int32_t fye = fy + fh;

    // 归一化：沿用旧版沙漏边界的比例（颈口 0.469~0.556，上下半区 0.467/0.5），
    // 并把实体延伸到视口四角，保证圆角卡片内没有漏底的空隙。
    float xl = (float)fx + 0.46875f * (float)fw;
    float xr = (float)fx + 0.55625f * (float)fw;
    float yn0 = (float)fy + 0.46667f * (float)fh;
    float yn1 = (float)fy + 0.5f * (float)fh;
    int32_t ix = (int32_t)lroundf(xl);
    int32_t iy0 = (int32_t)lroundf(yn0);
    int32_t iy1 = (int32_t)lroundf(yn1);
    int32_t ix2 = (int32_t)lroundf(xr);

    // 沙漏实体的内缘 x 随 y 变化：上半段沿斜边、颈部竖直、下半段沿斜边。
    // 用 6 条水平带拼出竖向渐变（每带一个梯形 = 两个三角形），避免上下两色的接缝。
    #define FLIP_WALL_EDGE(y, is_right) \
        ((y) <= yn0 ? ((is_right) ? (float)fxe + ((float)(y) - (float)fy) / (yn0 - (float)fy) * (xr - (float)fxe) \
                                  : (float)fx  + ((float)(y) - (float)fy) / (yn0 - (float)fy) * (xl - (float)fx)) \
        : ((y) <= yn1 ? ((is_right) ? xr : xl) \
                      : ((is_right) ? xr + ((float)(y) - yn1) / ((float)fye - yn1) * ((float)fxe - xr) \
                                    : xl + ((float)(y) - yn1) / ((float)fye - yn1) * ((float)fx - xl))))
    for (int32_t b = 0; b < BANDS; b++) {
        int32_t y0 = fy + fh * b / BANDS;
        int32_t y1 = fy + fh * (b + 1) / BANDS;
        float t = ((float)b + 0.5f) / (float)BANDS;
        uint8_t cr = (uint8_t)lroundf((float)k_wall_hi[0] + ((float)k_wall_lo[0] - (float)k_wall_hi[0]) * t);
        uint8_t cg = (uint8_t)lroundf((float)k_wall_hi[1] + ((float)k_wall_lo[1] - (float)k_wall_hi[1]) * t);
        uint8_t cb = (uint8_t)lroundf((float)k_wall_hi[2] + ((float)k_wall_lo[2] - (float)k_wall_hi[2]) * t);
        int32_t lx0 = (int32_t)lroundf(FLIP_WALL_EDGE(y0, 0));
        int32_t lx1 = (int32_t)lroundf(FLIP_WALL_EDGE(y1, 0));
        int32_t rx0 = (int32_t)lroundf(FLIP_WALL_EDGE(y0, 1));
        int32_t rx1 = (int32_t)lroundf(FLIP_WALL_EDGE(y1, 1));
        gfx_draw_triangle(g, fx, y0, lx0, y0, lx1, y1, 1, cr, cg, cb, 1);
        gfx_draw_triangle(g, fx, y0, lx1, y1, fx, y1, 1, cr, cg, cb, 1);
        gfx_draw_triangle(g, fxe, y0, rx0, y0, rx1, y1, 1, cr, cg, cb, 1);
        gfx_draw_triangle(g, fxe, y0, rx1, y1, fxe, y1, 1, cr, cg, cb, 1);
    }
    #undef FLIP_WALL_EDGE

    // 玻璃内沿：1.8px 低亮度辉光 + 1px 亮边（AA 线取**加色**模式 3：底色 + 覆盖率×颜色，
    // 天然像自发光玻璃；mode 语义见 graphics.h，2026-11-24 起 AA 线才真的认 mode）
    #define FLIP_RIM(x1, y1, x2, y2) do { \
        gfx_draw_line_anti_aliasing(g, (x1), (y1), (x2), (y2), 1.8f, \
            k_rim_glow[0], k_rim_glow[1], k_rim_glow[2], 3); \
        gfx_draw_line_anti_aliasing(g, (x1), (y1), (x2), (y2), 1.0f, \
            k_rim[0], k_rim[1], k_rim[2], 3); \
    } while (0)
    FLIP_RIM((float)fx, (float)fy, xl, yn0);
    FLIP_RIM(xl, yn0, xl, yn1);
    FLIP_RIM(xl, yn1, (float)fx, (float)fye);
    FLIP_RIM((float)fxe, (float)fy, xr, yn0);
    FLIP_RIM(xr, yn0, xr, yn1);
    FLIP_RIM(xr, yn1, (float)fxe, (float)fye);
    #undef FLIP_RIM

    // 颈口一线微光（提示这里是通道）
    gfx_draw_rectangle(g, ix, iy1, (uint32_t)(ix2 - ix), 1,
                       p->accent_dim[0], p->accent_dim[1], p->accent_dim[2], 110);

    // 压板内侧：顶部亮边 + 阴影（遮住粒子越界溢出的 1 像素）；底部阴影 + 暗边
    gfx_draw_rectangle(g, fx, fy, (uint32_t)fw, 1,
                       k_stage_plate_hi[0], k_stage_plate_hi[1], k_stage_plate_hi[2], 1);
    gfx_draw_rectangle(g, fx, fy + 1, (uint32_t)fw, 1, 0, 0, 0, 90);
    gfx_draw_rectangle(g, fx, fye - 2, (uint32_t)fw, 1, 0, 0, 0, 90);
    gfx_draw_rectangle(g, fx, fye - 1, (uint32_t)fw, 1,
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
}

// 自由流体：矩形槽的四壁（1px 暗边 + 顶部亮边/底部阴影，遮住粒子溢出的边缘）。
static void flip_ui_draw_tank_edges(Nano_GFX *g, const Flip_Ui *u) {
    int32_t fx = u->fluid_x;
    int32_t fy = u->fluid_y;
    int32_t fw = u->fluid_w;
    int32_t fh = u->fluid_h;
    int32_t fxe = fx + fw;
    int32_t fye = fy + fh;
    // 左右/底部槽壁
    gfx_draw_rectangle(g, fx, fy + 1, 1, (uint32_t)(fh - 2),
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
    gfx_draw_rectangle(g, fxe - 1, fy + 1, 1, (uint32_t)(fh - 2),
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
    gfx_draw_rectangle(g, fx, fye - 1, (uint32_t)fw, 1,
                       k_stage_plate_lo[0], k_stage_plate_lo[1], k_stage_plate_lo[2], 1);
    // 顶部：亮边 + 阴影；底部：阴影 + 暗边
    gfx_draw_rectangle(g, fx, fy, (uint32_t)fw, 1,
                       k_stage_plate_hi[0], k_stage_plate_hi[1], k_stage_plate_hi[2], 1);
    gfx_draw_rectangle(g, fx, fy + 1, (uint32_t)fw, 1, 0, 0, 0, 90);
    gfx_draw_rectangle(g, fx, fye - 2, (uint32_t)fw, 1, 0, 0, 0, 90);
}

// 前玻璃：斜向反光 + 上下内阴影，把粒子与沙漏实体统一到同一块玻璃后面。
// 注：gfx_draw_triangle 的 mode>=4 在扫描线路径里按“加色”处理，故这里用低加色值
// 叠 3 层形成柔和的斜向高光；矩形则走逐像素 alpha，用于压暗的内阴影。
static void flip_ui_draw_glass(Nano_GFX *g, const Flip_Ui *u, int32_t free_mode) {
    int32_t fx = u->fluid_x;
    int32_t fy = u->fluid_y;
    int32_t fw = u->fluid_w;
    int32_t fh = u->fluid_h;
    int32_t fxe = fx + fw;
    int32_t fye = fy + fh;

    // 斜向反光：三层由宽到窄、由暗到亮（低加色，避免压过粒子）
    gfx_draw_triangle(g, fx + fw * 4 / 100, fy, fx + fw * 40 / 100, fy,
                      fx + fw * 60 / 100, fye, 1, 2, 3, 4, 3);
    gfx_draw_triangle(g, fx + fw * 4 / 100, fy, fx + fw * 60 / 100, fye,
                      fx + fw * 26 / 100, fye, 1, 2, 3, 4, 3);
    gfx_draw_triangle(g, fx + fw * 19 / 100, fy, fx + fw * 30 / 100, fy,
                      fx + fw * 51 / 100, fye, 1, 3, 3, 5, 3);
    gfx_draw_triangle(g, fx + fw * 19 / 100, fy, fx + fw * 51 / 100, fye,
                      fx + fw * 40 / 100, fye, 1, 3, 3, 5, 3);
    gfx_draw_triangle(g, fx + fw * 24 / 100, fy, fx + fw * 27 / 100, fy,
                      fx + fw * 48 / 100, fye, 1, 3, 4, 6, 3);
    gfx_draw_triangle(g, fx + fw * 24 / 100, fy, fx + fw * 48 / 100, fye,
                      fx + fw * 45 / 100, fye, 1, 3, 4, 6, 3);

    // 上下内阴影（逐像素 alpha）：玻璃凹陷的深度
    for (int32_t i = 1; i <= 4; i++) {
        uint8_t a = (uint8_t)(28 - (i - 1) * 7);
        gfx_draw_rectangle(g, fx, fy + i, (uint32_t)fw, 1, 0, 0, 0, a);
        gfx_draw_rectangle(g, fx, fye - 1 - i, (uint32_t)fw, 1, 0, 0, 0, a);
    }

    // 自由模式的矩形槽：左右内壁也补一道内阴影，强调“玻璃缸”四壁
    if (free_mode) {
        for (int32_t i = 1; i <= 3; i++) {
            uint8_t a = (uint8_t)(22 - (i - 1) * 6);
            gfx_draw_rectangle(g, fx + i, fy + 4, 1, (uint32_t)(fh - 8), 0, 0, 0, a);
            gfx_draw_rectangle(g, fxe - 1 - i, fy + 4, 1, (uint32_t)(fh - 8), 0, 0, 0, a);
        }
    }
}

// ---- 左右仪表 --------------------------------------------------------------
static void flip_ui_draw_progress_rail(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                       float progress, int32_t remaining, int32_t grow_up) {
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"进度", u->lrail_cx, u->label_cy, p->dim, 1);

    // 竖槽
    flip_ui_round(g, u->lgauge_x, u->gauge_y, u->gauge_w, u->gauge_h, u->gauge_w / 2, p->track, 1);
    int32_t fill_h = (int32_t)lroundf(progress * (float)u->gauge_h);
    if (fill_h > 0) {
        // 起点与增长方向跟随重力：重力朝屏幕下方（沙漏正放）→ 从底部长起；
        // 重力朝屏幕上方（倒置）→ 从顶部长下来。与沙粒实际的落向一致。
        int32_t y = grow_up ? (u->gauge_y + u->gauge_h - fill_h) : u->gauge_y;
        flip_ui_round(g, u->lgauge_x, y, u->gauge_w, fill_h, u->gauge_w / 2, p->accent, 1);
    }
    // 槽内刻度（4 格，1px，很淡）
    for (int32_t i = 1; i < 4; i++) {
        int32_t y = u->gauge_y + u->gauge_h * i / 4;
        gfx_draw_rectangle(g, u->lgauge_x + 1, y, (uint32_t)(u->gauge_w - 2), 1,
                           p->bg_top[0], p->bg_top[1], p->bg_top[2], 90);
    }

    wchar_t val[8];
    swprintf(val, sizeof(val) / sizeof(val[0]), L"%d%%", (int32_t)floorf(progress * 100.0f + 0.5f));
    flip_ui_text_centered(g, GFX_FONT_ALPHA_16, val, u->lrail_cx, u->gauge_value_cy, p->text, 1);

    wchar_t rem[16];
    swprintf(rem, sizeof(rem) / sizeof(rem[0]), L"余%d", remaining);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, rem, u->lrail_cx, u->gauge_caption_cy, p->dim, 1);
}

static void flip_ui_slider_knob_y(const Flip_Ui *u, int32_t value, int32_t *out_y) {
    float t = ((float)value - 10.0f) / 90.0f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    *out_y = u->gauge_y + u->gauge_h - (int32_t)lroundf(t * (float)u->gauge_h);
}

static void flip_ui_draw_throttle_rail(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                       int32_t value, int32_t effective, int32_t enabled,
                                       int32_t pressed) {
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"节流度", u->rrail_cx, u->label_cy, p->dim, 1);

    // 竖槽
    flip_ui_round(g, u->rgauge_x, u->gauge_y, u->gauge_w, u->gauge_h, u->gauge_w / 2, p->track, 1);

    int32_t knob_y = 0;
    flip_ui_slider_knob_y(u, value, &knob_y);
    int32_t fill_y = knob_y;
    if (fill_y < u->gauge_y) fill_y = u->gauge_y;
    int32_t fill_h = u->gauge_y + u->gauge_h - fill_y;
    if (fill_h > 0) {
        flip_ui_round(g, u->rgauge_x, fill_y, u->gauge_w, fill_h, u->gauge_w / 2,
                      enabled ? p->accent : p->accent_off, 1);
    }

    // 当前自适应节流值：槽内一道细标线（关闭时不画）
    if (enabled) {
        int32_t eff_y = 0;
        flip_ui_slider_knob_y(u, effective, &eff_y);
        gfx_draw_rectangle(g, u->rgauge_x - 2, eff_y, (uint32_t)(u->gauge_w + 4), 1,
                           p->text[0], p->text[1], p->text[2], 120);
    }

    // 推钮：横向小胶囊（比槽宽，像推子）；外圈用底色描一圈，和同色填充的轨道分开
    int32_t knob_w = u->gauge_w * 2;
    int32_t knob_h = u->knob_r * 2;
    int32_t kx = u->rrail_cx - knob_w / 2;
    int32_t ky = knob_y - knob_h / 2;
    flip_ui_round(g, kx - 1, ky - 1, knob_w + 2, knob_h + 2, knob_h / 2 + 1, p->bg_top, 1);
    flip_ui_round(g, kx, ky, knob_w, knob_h, knob_h / 2,
                  (pressed == FLIP_UI_SLIDER) ? (enabled ? p->accent : p->accent_dim)
                                              : (enabled ? p->accent : p->accent_off), 1);
    // 推钮中点一道细亮线（握持感）
    gfx_draw_rectangle(g, kx + knob_w / 2 - 3, ky + knob_h / 2, 6, 1,
                       p->on_accent[0], p->on_accent[1], p->on_accent[2],
                       enabled ? 90 : 40);

    wchar_t val[8];
    swprintf(val, sizeof(val) / sizeof(val[0]), L"%d%%", value);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_16, val, u->rrail_cx, u->gauge_value_cy,
                          enabled ? p->text : p->dim, 1);

    wchar_t cur[16];
    // 当前自适应值只写数字（不写 %）：轨道只有 44px 宽，「当前100%」会被屏幕边缘裁掉
    if (enabled) swprintf(cur, sizeof(cur) / sizeof(cur[0]), L"当前%d", effective);
    else wcsncpy(cur, L"已关闭", sizeof(cur) / sizeof(cur[0]));
    cur[sizeof(cur) / sizeof(cur[0]) - 1] = L'\0';
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, cur, u->rrail_cx, u->gauge_caption_cy,
                          enabled ? p->accent : p->dim, 1);
}

// 自由流体模式的仪表：左=粒子总数/矩形槽，右=模式/自由/无节流。
static void flip_ui_draw_free_rails(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                    int32_t total_count) {
    // 没有竖槽/滑杆，三行文字在仪表列里垂直居中，和两侧留白平衡
    int32_t cy = u->rail_y + u->rail_h / 2;
    int32_t y_label = cy - 28;
    int32_t y_value = cy;
    int32_t y_caption = cy + 24;

    // 左：粒子数
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"粒子", u->lrail_cx, y_label, p->dim, 1);
    wchar_t cnt[16];
    swprintf(cnt, sizeof(cnt) / sizeof(cnt[0]), L"%d", total_count);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_16, cnt, u->lrail_cx, y_value, p->text, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"矩形槽", u->lrail_cx, y_caption, p->dim, 1);

    // 右：模式
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"模式", u->rrail_cx, y_label, p->dim, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_16, L"自由", u->rrail_cx, y_value, p->accent, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"无节流", u->rrail_cx, y_caption, p->dim, 1);
}

// ---- 「配色」胶囊（底部控制条；打开模态色板）--------------------------------
static void flip_ui_draw_palette_pill(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                      int32_t pressed) {
    int32_t x = u->pal_x, y = u->btn_y, w = u->pal_w, h = u->btn_h;
    int32_t st = (pressed == FLIP_UI_PALETTE) ? 1 : 0;
    // 与普通按钮同一描边风格：外圈描边色 + 内圈底色
    flip_ui_round(g, x, y, w, h, u->btn_r, p->surface_edge, 1);
    flip_ui_round(g, x + 1, y + 1, w - 2, h - 2, u->btn_r - 1,
                  st ? p->surface_press : p->surface, 1);
    if (st) {
        flip_ui_round(g, x + 1, y + 1, w - 2, h - 2, u->btn_r - 1, p->accent, 70);
    }
    // 当前强调色圆点 + 「配色」
    int32_t group = 6 + 3 + 24;
    int32_t gx = x + (w - group) / 2;
    int32_t cy = y + h / 2;
    flip_ui_circle(g, gx + 3, cy, 3, p->accent, 1);
    flip_ui_text(g, GFX_FONT_ALPHA_12, L"配色", gx + 9, cy - 7, p->text, 1);
}

// ---- 底部控制条 ------------------------------------------------------------
static void flip_ui_draw_dock(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                              int32_t pressed, int32_t pressed_view,
                              float flow_per_sec, int32_t free_mode) {
    // 复位
    flip_ui_button(g, p, u->reset_x, u->btn_y, u->reset_w, u->btn_h, u->btn_r, L"复位",
                   (pressed == FLIP_UI_RESET) ? 1 : 0);

    // 画风：三分段控件
    int32_t gx = u->view_x;
    int32_t gy = u->btn_y;
    int32_t gw = u->view_w;
    int32_t gh = u->btn_h;
    flip_ui_round(g, gx, gy, gw, gh, u->btn_r, p->surface, 1);
    flip_ui_round(g, gx, gy, gw, gh, u->btn_r, p->surface_edge, 1);
    flip_ui_round(g, gx + 1, gy + 1, gw - 2, gh - 2, u->btn_r - 1, p->surface, 1);
    static const wchar_t *view_labels[3] = { L"粒子", L"网格", L"叠加" };
    for (int32_t i = 0; i < 3; i++) {
        int32_t cx = gx + u->view_cell_w * i;
        int32_t cw = u->view_cell_w;
        if (i == 2) cw = gx + gw - cx;
        int32_t selected = (s_ui_flip_setting_count == i);
        if (selected) {
            flip_ui_round(g, cx + 2, gy + 2, cw - 4, gh - 4, u->btn_r - 2, p->accent, 1);
        }
        else if (pressed == FLIP_UI_VIEW && pressed_view == i) {
            flip_ui_round(g, cx + 2, gy + 2, cw - 4, gh - 4, u->btn_r - 2, p->surface_press, 1);
        }
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, view_labels[i], cx + cw / 2, gy + gh / 2,
                              selected ? p->on_accent : p->dim, 1);
    }

    // 节流开关（自由流体模式下节流机制整体关闭：画成不可用指示，不响应点按）
    if (free_mode) {
        flip_ui_round(g, u->thr_x, u->btn_y, u->thr_w, u->btn_h, u->btn_r, p->surface_edge, 1);
        flip_ui_round(g, u->thr_x + 1, u->btn_y + 1, u->thr_w - 2, u->btn_h - 2,
                      u->btn_r - 1, p->surface, 1);
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"节流 -",
                              u->thr_x + u->thr_w / 2, u->btn_y + u->btn_h / 2, p->dim, 1);
    }
    else {
        wchar_t thr_label[16];
        if (s_ui_flip_is_throttle) wcsncpy(thr_label, L"节流 开", 16);
        else wcsncpy(thr_label, L"节流 关", 16);
        thr_label[15] = L'\0';
        int32_t thr_state;
        if (s_ui_flip_is_throttle) {
            thr_state = 2;
            if (pressed == FLIP_UI_THROTTLE) {
                // 选中态按下：用稍暗的强调色反馈
                flip_ui_round(g, u->thr_x, u->btn_y, u->thr_w, u->btn_h, u->btn_r, p->accent_dim, 1);
                flip_ui_text_centered(g, GFX_FONT_ALPHA_12, thr_label,
                                      u->thr_x + u->thr_w / 2, u->btn_y + u->btn_h / 2, p->on_accent, 1);
                thr_state = -1;
            }
        }
        else {
            thr_state = (pressed == FLIP_UI_THROTTLE) ? 1 : 3;
        }
        if (thr_state >= 0) {
            flip_ui_button(g, p, u->thr_x, u->btn_y, u->thr_w, u->btn_h, u->btn_r, thr_label, thr_state);
        }
    }

    // 配色
    flip_ui_draw_palette_pill(g, u, p, pressed);

    // 流量读数（为 0 时弱化，避免流完后仍是一个抢眼的 0/s）
    wchar_t flow_str[20];
    swprintf(flow_str, sizeof(flow_str) / sizeof(flow_str[0]), L"%.0f/s", flow_per_sec);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, L"流量", u->stat_cx, u->stat_y1, p->dim, 1);
    flip_ui_text_centered(g, GFX_FONT_ALPHA_12, flow_str, u->stat_cx, u->stat_y2,
                          (flow_per_sec < 0.5f) ? p->dim : p->accent, 1);
}

// ---- 配色选择弹层（模态：半透明遮罩 + 居中卡片 + 5 个大色块）-----------------
static void flip_ui_draw_picker(Nano_GFX *g, const Flip_Ui *u, const Flip_Ui_Palette *p,
                                int32_t is_light, int32_t pressed_idx) {
    // 遮罩：压暗整屏，突出模态
    gfx_draw_rectangle(g, 0, 0, (uint32_t)u->w, (uint32_t)u->h, 0, 0, 0, 110);

    // 卡片
    flip_ui_round(g, u->pick_x - 1, u->pick_y - 1, u->pick_w + 2, u->pick_h + 2,
                  u->pick_r + 1, p->card_edge, 1);
    flip_ui_round(g, u->pick_x, u->pick_y, u->pick_w, u->pick_h, u->pick_r, p->surface, 1);
    flip_ui_text(g, GFX_FONT_ALPHA_12, L"色彩风格", u->pick_title_x, u->pick_title_y, p->text, 1);

    for (int32_t i = 0; i < FLIP_COLOR_STYLES; i++) {
        int32_t sx = u->sw_x + i * (u->sw_w + u->sw_gap);
        const uint8_t *a = flip_style_accent(i, is_light);
        uint8_t on[3];
        flip_style_on_accent(a, on);
        // 选中：2px 描边环（先画大一圈的环，再画色块，留下 2px 边）
        if (i == s_ui_flip_color_style) {
            flip_ui_round(g, sx - 2, u->sw_y - 2, u->sw_w + 4, u->sw_h + 4,
                          u->sw_r + 2, p->text, 1);
        }
        flip_ui_round(g, sx, u->sw_y, u->sw_w, u->sw_h, u->sw_r, a, 1);
        if (i == pressed_idx) {
            // 按下：叠一层文字色（亮底变暗、暗底变亮）
            flip_ui_round(g, sx, u->sw_y, u->sw_w, u->sw_h, u->sw_r, on, 70);
        }
        flip_ui_text_centered(g, GFX_FONT_ALPHA_12, flip_style_names[i],
                              sx + u->sw_w / 2, u->sw_y + u->sw_h / 2, on, 1);
    }
}

// ---- 画风切换 --------------------------------------------------------------
static void flip_ui_apply_view(int32_t mode) {
    if (mode < 0) mode = 0;
    if (mode > 2) mode = 2;
    s_ui_flip_setting_count = mode;
    if (mode == FLIP_VIEW_PARTICLE) {
        s_ui_flip_show_grid = 0;
        s_ui_flip_show_particles = 1;
    }
    else if (mode == FLIP_VIEW_GRID) {
        s_ui_flip_show_grid = 1;
        s_ui_flip_show_particles = 0;
    }
    else {
        s_ui_flip_show_grid = 1;
        s_ui_flip_show_particles = 1;
    }
}

// ============================================================================
// 公开接口
// ============================================================================

void ui_flip_init(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    s_ui_fanqie_start_timestamp = global_state->timestamp;
    s_ui_fanqie_stop_timestamp = 0;
    s_ui_fanqie_is_running = 1;
    s_ui_fanqie_alarm_count = 0;
    s_ui_fanqie_alarm_phase = 0;
    s_ui_flip_last_upper_count = -1;
    s_ui_flip_last_upper_count_timestamp = global_state->timestamp;
    s_ui_flip_ui_pressed = FLIP_UI_NONE;
    s_ui_flip_ui_slider_active = 0;
    s_ui_flip_picker_open = 0;
    s_ui_flip_picker_pressed = FLIP_UI_NONE;
    float k = (float)(global_state->gfx->width) / (float)(global_state->gfx->height);
    // 沙漏模式：带沙漏固体边界；自由模式：只有矩形外壁
    flip_init(k, 1.0f, FLIP_RESOLUTION, global_state->timestamp,
              s_ui_flip_free_mode ? 0 : 1);
}

void ui_flip_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 帧率节流：模拟按固定步长 dt 推进，帧率即物理流速。设备原生帧率受渲染耗时
    // 限制（ESP32/树莓派 SPI 推屏、nano_tty 终端 ANSI 渲染，均低于 30fps），表观
    // 流速正常；但高性能平台（nano_web 于 x86，主循环可达数千 FPS）若不节流，
    // 流体将以百倍于设计的速度演化（且远端只能采到极少中间步，中间过程全部丢失）。
    // 此处限制模拟/渲染最高 30fps（未到帧间隔直接返回）：原生帧率低于 30fps 的
    // 平台上本节流不触发，行为与既有版本完全一致。
    static uint64_t s_flip_last_frame_ts = 0;
    if (global_state->timestamp - s_flip_last_frame_ts < 33) {
        return;
    }
    s_flip_last_frame_ts = global_state->timestamp;

    int32_t free_mode = s_ui_flip_free_mode;

    Flip_Ui u;
    Flip_Ui_Palette pal;
    flip_ui_layout(global_state->gfx, &u);
    flip_ui_palette(global_state, &pal);

    // 获取重力方向（IMU每4帧读取一次并缓存：重力为慢变量，无需每帧I2C读取）
    static float gravity_x = 0.0f;
    static float gravity_y = -9.8f;

#ifdef IMU_ENABLED

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    static int imu_frame_div = 0;
    if ((imu_frame_div++ & 3) == 0) {
        float gx = 0.0f;
        float gy = 0.0f;
        float gz = 0.0f;
        imu_read_angle(&gx, &gy, &gz);
        gravity_x = -9.8f * gx;
        gravity_y = -9.8f * gy;
    }
#else
    // 调试打印已移除（旧版每帧 printf 会把浏览器控制台刷满，也拖慢 WASM 主循环）
    imu_read_angle(&(global_state->pitch), &(global_state->roll), &(global_state->yaw));
    gravity_x = -9.8f * sinf(global_state->roll / 180.0f * M_PI);
    gravity_y = -9.8f * cosf(global_state->roll / 180.0f * M_PI);
#endif

#endif

    float k = (float)(global_state->gfx->width) / (float)(global_state->gfx->height);

    int32_t upper_count = 0;
    int32_t lower_count = 0;

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    float dt = free_mode ? (1.6f / 60.0f)
                         : ((s_ui_flip_is_throttle) ? (0.8f / 60.0f) : (1.6f / 60.0f));
#else
    float dt = free_mode ? (1.6f / 60.0f)
                         : ((s_ui_flip_is_throttle) ? (0.6f / 60.0f) : (1.6f / 60.0f));
#endif

    // ---- 背景 + 舞台外框（用上一帧的运行态决定描边色，避免多算一次流体）----
    flip_ui_bg(global_state->gfx, &pal);
    flip_ui_draw_stage_frame(global_state->gfx, &u, &pal, !s_ui_fanqie_is_running,
                             gravity_y <= 0, free_mode);

    // ---- 流体：只画进舞台视口（视口与屏幕同宽高比 ⇒ 物理参数 k / 网格不变）----
    flip_set_color_style(s_ui_flip_color_style);   // 纯渲染配色（不影响物理）
    render_flip(
        global_state->gfx, u.fluid_x, u.fluid_y, u.fluid_w, u.fluid_h,
        k, 1.0f, FLIP_RESOLUTION,     /* pool_width, pool_height, resolution */
        gravity_x, gravity_y,    /* gravity_x, gravity_y */
        dt,    /* dt */
        0.8f,            /* flip_ratio */
        20, 2,           /* num_pressure_iters, num_particle_iters */
        1.0f,            /* over_relaxation */
        1, 1,            /* compensate_drift, separate_particles */
        s_ui_flip_show_particles, s_ui_flip_show_grid,
        (free_mode || !s_ui_flip_is_throttle) ? 0 : s_ui_flip_throttle,
        &upper_count, &lower_count
    );

    // 沙漏实体 / 矩形槽四壁盖在粒子之上（自由模式没有沙漏壁与节流）
    if (free_mode) {
        flip_ui_draw_tank_edges(global_state->gfx, &u);
    }
    else {
        flip_ui_draw_walls(global_state->gfx, &u, &pal);
    }
    // 前玻璃：斜向反光 + 内阴影，让粒子与实体看起来在同一块玻璃后面
    flip_ui_draw_glass(global_state->gfx, &u, free_mode);

    // ---- 沙漏重置判定（逻辑与旧版一致）；自由流体模式没有“流完”，计时器只做秒表 ----
    if (free_mode) {
        s_ui_fanqie_stop_timestamp = 0;
        s_ui_fanqie_is_running = 1;
    }
    else if ((lower_count < 1 && gravity_y < 0) || (upper_count < 1 && gravity_y > 0)) {
        s_ui_fanqie_start_timestamp = global_state->timestamp;
        s_ui_fanqie_stop_timestamp = 0;
        s_ui_fanqie_is_running = 1;
        s_ui_fanqie_alarm_count = 0;
    }
    else if ((upper_count < 1 && gravity_y < 0) || (lower_count < 1 && gravity_y > 0)) {
        if (!s_ui_fanqie_stop_timestamp) {
            s_ui_fanqie_stop_timestamp = global_state->timestamp;
        }
        s_ui_fanqie_is_running = 0;
    }

    // ---- 计时 ----
    uint64_t current_timestamp = global_state->timestamp;
    if (!s_ui_fanqie_is_running) {
        current_timestamp = s_ui_fanqie_stop_timestamp;
    }
    if (current_timestamp < s_ui_fanqie_start_timestamp) {
        current_timestamp = s_ui_fanqie_start_timestamp;
    }
    uint64_t elapsed_ms = current_timestamp - s_ui_fanqie_start_timestamp;

    // ---- 每1000ms统计一次粒子流量 ----
    // 首拍只建立基线（last=-1 哨兵）：否则「进入/复位后的第一帧」会把满池粒子数除以
    // 几毫秒的 dt，显示出一个几十万/秒的假流量尖峰。
    static float particle_flow_per_sec = 0.0f;
    if (s_ui_flip_last_upper_count < 0) {
        particle_flow_per_sec = 0.0f;
        s_ui_flip_last_upper_count = upper_count;
        s_ui_flip_last_upper_count_timestamp = global_state->timestamp;
    }
    else if (global_state->timestamp - s_ui_flip_last_upper_count_timestamp >= 1000) {
        uint64_t dt_ms = global_state->timestamp - s_ui_flip_last_upper_count_timestamp;
        if (dt_ms > 0) {
            particle_flow_per_sec = (float)(upper_count - s_ui_flip_last_upper_count) / (float)dt_ms * 1000;
        }
        s_ui_flip_last_upper_count = upper_count;
        s_ui_flip_last_upper_count_timestamp = global_state->timestamp;
    }

    // ---- 沙漏进度：相对重力方向，已落下的沙量占比 ----
    int32_t total_count = upper_count + lower_count;
    float hourglass_progress = 0.0f;
    if (total_count > 0) {
        hourglass_progress = (float)((gravity_y <= 0) ? lower_count : upper_count) / (float)total_count;
    }
    int32_t remaining_count = (gravity_y <= 0) ? upper_count : lower_count;

    // 根据沙漏进度调整节流度，避免来自上方的压力过小时，出现几乎不往下流的问题
    // （自由模式关闭全部节流机制）
    if (free_mode) {
        s_ui_flip_throttle = 0;
    }
    else {
        s_ui_flip_throttle = (int32_t)roundf((1.0f - (float)s_ui_flip_init_throttle) * hourglass_progress * hourglass_progress
                                             + (float)s_ui_flip_init_throttle);
    }

    // ---- 到时提醒（非阻塞状态机：跨帧推进震动节奏，不阻塞渲染任务）----
    if (!free_mode && !s_ui_fanqie_is_running && s_ui_flip_is_throttle && s_ui_fanqie_alarm_count < 6) {
        // 门槛：仅真实计时（运行时长≥3s）结束后才报警，杜绝“进入即误判流完”的伪报警
        if (s_ui_fanqie_stop_timestamp >= s_ui_fanqie_start_timestamp &&
            s_ui_fanqie_stop_timestamp - s_ui_fanqie_start_timestamp >= 3000) {
            uint64_t now = global_state->timestamp;
            if (s_ui_fanqie_alarm_phase == 0) {           // 开始一段鸣震
                set_vibration(222);
                s_ui_fanqie_alarm_phase_ts = now;
                s_ui_fanqie_alarm_phase = 1;
            }
            else if (s_ui_fanqie_alarm_phase == 1 && now - s_ui_fanqie_alarm_phase_ts >= 600) {
                set_vibration(0);                          // 鸣震 600ms 后进入静默段
                s_ui_fanqie_alarm_phase_ts = now;
                s_ui_fanqie_alarm_phase = 2;
            }
            else if (s_ui_fanqie_alarm_phase == 2 && now - s_ui_fanqie_alarm_phase_ts >= 600) {
                s_ui_fanqie_alarm_count++;                 // 静默 600ms 后完成一个周期
                s_ui_fanqie_alarm_phase = 0;
            }
        }
    }
    else if (s_ui_fanqie_alarm_phase != 0) {
        // 报警被中断（重新计时/关闭节流/报警完成）：确保马达关闭、状态机复位
        set_vibration(0);
        s_ui_fanqie_alarm_phase = 0;
    }

    // ---- chrome：顶栏 / 左右仪表 / 底部控制条 ----
    flip_ui_draw_header(key_event, global_state, &u, &pal, elapsed_ms,
                        s_ui_fanqie_is_running, s_ui_flip_ui_pressed);
    if (free_mode) {
        flip_ui_draw_free_rails(global_state->gfx, &u, &pal, total_count);
    }
    else {
        // 进度槽的起点/方向跟随重力（gravity_y < 0 = 重力朝屏幕下方 = 正放）
        flip_ui_draw_progress_rail(global_state->gfx, &u, &pal, hourglass_progress, remaining_count,
                                   gravity_y <= 0);
        flip_ui_draw_throttle_rail(global_state->gfx, &u, &pal, s_ui_flip_init_throttle,
                                   s_ui_flip_throttle, s_ui_flip_is_throttle, s_ui_flip_ui_pressed);
    }
    flip_ui_draw_dock(global_state->gfx, &u, &pal, s_ui_flip_ui_pressed,
                      s_ui_flip_ui_pressed_view, fabsf(particle_flow_per_sec), free_mode);

    // 配色选择弹层（模态）：画在全部 chrome 之上
    if (s_ui_flip_picker_open) {
        flip_ui_draw_picker(global_state->gfx, &u, &pal,
                            global_state->ui_color_style == UI_COLOR_LIGHT,
                            s_ui_flip_picker_pressed);
    }

    gfx_refresh(global_state->gfx);
}

// ---- 触屏命中判定（画风分段控件返回整组，另用 view_index 区分格子）---------
static int32_t flip_ui_hit(const Flip_Ui *u, int32_t x, int32_t y) {
    if (x >= u->exit_x && x < u->exit_x + u->exit_w &&
        y >= u->exit_y && y < u->exit_y + u->exit_h) {
        return FLIP_UI_EXIT;
    }
    // 顶栏模式切换胶囊（沙漏 ⇄ 自由流体）
    if (x >= u->mode_x && x < u->mode_x + u->mode_w &&
        y >= u->mode_y && y < u->mode_y + u->mode_h) {
        return FLIP_UI_MODE;
    }
    if (y >= u->btn_y && y < u->btn_y + u->btn_h) {
        if (x >= u->reset_x && x < u->reset_x + u->reset_w) return FLIP_UI_RESET;
        if (x >= u->view_x && x < u->view_x + u->view_w) return FLIP_UI_VIEW;
        // 自由流体模式关闭节流机制：节流开关不响应
        if (!s_ui_flip_free_mode && x >= u->thr_x && x < u->thr_x + u->thr_w) return FLIP_UI_THROTTLE;
        if (x >= u->pal_x && x < u->pal_x + u->pal_w) return FLIP_UI_PALETTE;
    }
    // 右仪表：轨道上下各放宽一个推钮半径，便于点按/拖动；标签与读数区不响应
    // （自由模式右栏是模式信息，没有节流滑杆）
    if (!s_ui_flip_free_mode &&
        x >= u->rrail_x && x < u->rrail_x + u->rrail_w &&
        y >= u->gauge_y - u->knob_r && y <= u->gauge_y + u->gauge_h + u->knob_r) {
        return FLIP_UI_SLIDER;
    }
    return FLIP_UI_NONE;
}

static int32_t flip_ui_view_index(const Flip_Ui *u, int32_t x) {
    if (x < u->view_x) return 0;
    int32_t rel = x - u->view_x;
    if (rel >= u->view_w) return 2;
    int32_t idx = rel / (u->view_cell_w > 0 ? u->view_cell_w : 1);
    if (idx < 0) idx = 0;
    if (idx > 2) idx = 2;
    return idx;
}

// 配色色块命中（返回 0..4，未命中 -1）；与弹层绘制共用同一份几何
static int32_t flip_ui_swatch_hit(const Flip_Ui *u, int32_t x, int32_t y) {
    if (y < u->sw_y || y >= u->sw_y + u->sw_h) {
        return FLIP_UI_NONE;
    }
    for (int32_t i = 0; i < FLIP_COLOR_STYLES; i++) {
        int32_t sx = u->sw_x + i * (u->sw_w + u->sw_gap);
        if (x >= sx && x < sx + u->sw_w) return i;
    }
    return FLIP_UI_NONE;
}

static int32_t flip_ui_in_rect(int32_t x, int32_t y, int32_t w, int32_t h,
                               int32_t px, int32_t py) {
    return (px >= x && px < x + w && py >= y && py < y + h);
}

static void flip_ui_slider_set(const Flip_Ui *u, int32_t y) {
    float t = (float)(u->gauge_y + u->gauge_h - y) / (float)(u->gauge_h > 0 ? u->gauge_h : 1);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    int32_t v = 10 + (int32_t)lroundf(t * 90.0f);
    if (v < 10) v = 10;
    if (v > 100) v = 100;
    s_ui_flip_init_throttle = v;
}

void ui_flip_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 渲染帧里可能已经切走（退出按钮），此时不再处理输入
    if (global_state->STATE != STATE_FLIP) {
        return;
    }

    Flip_Ui u;
    flip_ui_layout(global_state->gfx, &u);

    // ---- 配色选择弹层打开时：模态，先消费触屏/ESC，普通控件一律不响应 ----
    if (s_ui_flip_picker_open) {
        if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
            s_ui_flip_picker_pressed = flip_ui_swatch_hit(&u, key_event->touch_down_x, key_event->touch_down_y);
        }
        if (key_event->touch_edge & TOUCH_EDGE_UP) {
            int32_t idx = s_ui_flip_picker_pressed;
            int32_t rel = flip_ui_swatch_hit(&u, key_event->touch_x, key_event->touch_y);
            s_ui_flip_picker_pressed = FLIP_UI_NONE;
            if (idx >= 0 && idx == rel) {
                s_ui_flip_color_style = idx;     // 选中即生效并关闭
                s_ui_flip_picker_open = 0;
            }
            else if (!flip_ui_in_rect(u.pick_x, u.pick_y, u.pick_w, u.pick_h,
                                      key_event->touch_x, key_event->touch_y)) {
                s_ui_flip_picker_open = 0;       // 点卡片外：关闭，不改设置
            }
        }
        if (key_event->is_soft_key == 0 &&
            (key_event->key_edge == -1 || key_event->key_edge == -2) &&
            key_event->key_code == NANO_KEY_esc) {
            s_ui_flip_picker_open = 0;           // ESC 先关弹层，不退出应用
        }
        return;
    }

    // ---- 触屏：按下点锁存控件，松手沿结算（同 ui_water.c / ui_musicbox.c 范式）----
    if (key_event->touch_edge & TOUCH_EDGE_DOWN) {
        int32_t hit = flip_ui_hit(&u, key_event->touch_down_x, key_event->touch_down_y);
        s_ui_flip_ui_pressed = hit;
        s_ui_flip_ui_slider_active = 0;
        if (hit == FLIP_UI_VIEW) {
            s_ui_flip_ui_pressed_view = flip_ui_view_index(&u, key_event->touch_down_x);
        }
        else if (hit == FLIP_UI_SLIDER) {
            s_ui_flip_ui_slider_active = 1;
            flip_ui_slider_set(&u, key_event->touch_down_y);
        }
    }
    if (s_ui_flip_ui_slider_active && key_event->is_touching) {
        flip_ui_slider_set(&u, key_event->touch_y);
    }
    if (key_event->touch_edge & TOUCH_EDGE_UP) {
        int32_t hit = s_ui_flip_ui_pressed;
        if (hit == FLIP_UI_SLIDER) {
            s_ui_flip_ui_slider_active = 0;
        }
        else if (hit != FLIP_UI_NONE) {
            int32_t rel_hit = flip_ui_hit(&u, key_event->touch_x, key_event->touch_y);
            int32_t same = (rel_hit == hit);
            if (hit == FLIP_UI_VIEW) {
                same = same && (flip_ui_view_index(&u, key_event->touch_x) == s_ui_flip_ui_pressed_view);
            }
            if (same) {
                if (hit == FLIP_UI_EXIT) {
                    s_ui_flip_ui_pressed = FLIP_UI_NONE;
                    global_state->STATE = STATE_MAIN_MENU;
                    return;
                }
                else if (hit == FLIP_UI_RESET) {
                    ui_flip_init(key_event, global_state);
                }
                else if (hit == FLIP_UI_VIEW) {
                    flip_ui_apply_view(s_ui_flip_ui_pressed_view);
                }
                else if (hit == FLIP_UI_THROTTLE) {
                    s_ui_flip_is_throttle = s_ui_flip_is_throttle ? 0 : 1;
                }
                else if (hit == FLIP_UI_PALETTE) {
                    s_ui_flip_picker_open = 1;
                    s_ui_flip_picker_pressed = FLIP_UI_NONE;
                }
                else if (hit == FLIP_UI_MODE) {
                    s_ui_flip_free_mode = s_ui_flip_free_mode ? 0 : 1;
                    ui_flip_init(key_event, global_state);   // 固体边界变了，重初始化流体
                }
            }
        }
        s_ui_flip_ui_pressed = FLIP_UI_NONE;
        s_ui_flip_ui_slider_active = 0;
    }

    // ---- 实体键盘兜底（触屏派生的虚拟键盘键码一律不响应：is_soft_key == 0）----
    if (key_event->is_soft_key != 0) {
        return;
    }
    int32_t edge = (key_event->key_edge == -1 || key_event->key_edge == -2);
    if (edge && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_MAIN_MENU;
    }
    else if (edge && key_event->key_code == NANO_KEY_left) {
        // 与旧版一致：循环切换画风（粒子 → 网格 → 叠加）
        flip_ui_apply_view((s_ui_flip_setting_count + 1) % 3);
    }
    else if (edge && key_event->key_code == NANO_KEY_right) {
        flip_ui_apply_view((s_ui_flip_setting_count + 2) % 3);
    }
    else if (edge && key_event->key_code == NANO_KEY_1) {
        if (!s_ui_flip_free_mode) s_ui_flip_is_throttle = s_ui_flip_is_throttle ? 0 : 1;
    }
    else if (edge && key_event->key_code == NANO_KEY_2) {
        s_ui_flip_free_mode = s_ui_flip_free_mode ? 0 : 1;
        ui_flip_init(key_event, global_state);
    }
    else if (edge && key_event->key_code == NANO_KEY_ctrl) {
        s_ui_flip_init_throttle += 10;
        if (s_ui_flip_init_throttle > 100) {
            s_ui_flip_init_throttle = 10;
        }
    }
    else if (edge && key_event->key_code == NANO_KEY_enter) {
        ui_flip_init(key_event, global_state);
    }
}