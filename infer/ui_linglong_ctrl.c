#include <stdint.h>

#include "platform.h"
#include "graphics.h"
#include "ui.h" // TOUCH_EDGE_DOWN / TOUCH_EDGE_UP
#include "ui_cloud.h"   // 云量档位名称（档位语义的唯一真源）
#include "celestial.h"  // 云类 / 云精度 档位表

#include "ui_linglong_ctrl.h"

// ===============================================================================
// 玲珑天象仪主界面显式触屏控件
//
// 布局（320x240 逻辑屏；位置按 SCREEN_WIDTH/HEIGHT 推导，按钮尺寸取适合掌机的固定触控尺寸）：
//   顶部靠右：控制 退出（2 个按钮）
//   “控制”同显同隐的控件：
//     底部四分之一区域（让出最底部状态栏）：<< 实时 运行/暂停 >>
//     画面中央：3x3 半透明视角九宫格（对应十六宫格软键 1~9）
//     九宫格左侧竖排：云量 / 云类 / 云精度（与九宫格三行齐平，不改动已有按钮位置）
//     九宫格右上角空位（“右转”右侧）：设置
//
// 绘制与命中判定共用下方几何常量，保证“看得见的一定点得到”。
// ===============================================================================

// ---- 顶部按钮栏 -----------------------------------------------------------
#define LL_TOP_COUNT      (2)
#define LL_TOP_BTN_W      (52)
#define LL_TOP_BTN_H      (22)
#define LL_TOP_GAP        (4)
#define LL_TOP_MARGIN     (4)
#define LL_TOP_Y          (4)
#define LL_TOP_TOTAL_W    (LL_TOP_COUNT * LL_TOP_BTN_W + (LL_TOP_COUNT - 1) * LL_TOP_GAP)
#define LL_TOP_X0         (SCREEN_WIDTH - LL_TOP_MARGIN - LL_TOP_TOTAL_W)

// ---- 时间控制按钮（底部四分之一区域，底部让出状态栏）-----------------------
#define LL_TIME_COUNT     (4)
#define LL_TIME_BTN_H     (34)
#define LL_TIME_GAP       (4)
#define LL_TIME_MARGIN    (8)
#define LL_TIME_BTN_W     ((SCREEN_WIDTH - 2 * LL_TIME_MARGIN - (LL_TIME_COUNT - 1) * LL_TIME_GAP) / LL_TIME_COUNT)
#define LL_TIME_TOTAL_W   (LL_TIME_COUNT * LL_TIME_BTN_W + (LL_TIME_COUNT - 1) * LL_TIME_GAP)
#define LL_TIME_BOTTOM    (14) // 最底部状态栏高度（与 ui_app_linglong_render_frame 的版本/时间行一致）
#define LL_TIME_Y         (SCREEN_HEIGHT - LL_TIME_BOTTOM - 4 - LL_TIME_BTN_H)
#define LL_TIME_BG_Y      (LL_TIME_Y - 4)
#define LL_TIME_BG_H      (LL_TIME_BTN_H + 8)

// ---- 视角九宫格（画面中央，半透明）----------------------------------------
#define LL_VIEW_COLS      (3)
#define LL_VIEW_ROWS      (3)
#define LL_VIEW_BTN_W     (56)
#define LL_VIEW_BTN_H     (40)
#define LL_VIEW_GAP       (6)
#define LL_VIEW_TOTAL_W   (LL_VIEW_COLS * LL_VIEW_BTN_W + (LL_VIEW_COLS - 1) * LL_VIEW_GAP)
#define LL_VIEW_TOTAL_H   (LL_VIEW_ROWS * LL_VIEW_BTN_H + (LL_VIEW_ROWS - 1) * LL_VIEW_GAP)
#define LL_VIEW_X0        ((SCREEN_WIDTH - LL_VIEW_TOTAL_W) / 2)
// 纵向居中；若会压到时间面板，则上移到时间面板上方（320x240 下为 46，正好在画面中央偏上）
#define LL_VIEW_Y0_CENTER ((SCREEN_HEIGHT - LL_VIEW_TOTAL_H) / 2)
#define LL_VIEW_Y0_ABOVE_TIME (LL_TIME_BG_Y - 6 - LL_VIEW_TOTAL_H)
#define LL_VIEW_Y0        ((LL_VIEW_Y0_CENTER < LL_VIEW_Y0_ABOVE_TIME) ? LL_VIEW_Y0_CENTER : LL_VIEW_Y0_ABOVE_TIME)

// ---- 体积云参数按钮（九宫格左侧竖排，与九宫格三行齐平；不改动已有按钮位置）------
#define LL_CLOUD_COUNT    (3)
#define LL_CLOUD_BTN_W    (LL_VIEW_BTN_W)
#define LL_CLOUD_BTN_H    (LL_VIEW_BTN_H)
#define LL_CLOUD_GAP_X    (6)
#define LL_CLOUD_X0       (LL_VIEW_X0 - LL_CLOUD_GAP_X - LL_CLOUD_BTN_W)
#define LL_CLOUD_ROW_H    (LL_VIEW_BTN_H + LL_VIEW_GAP)
#define LL_CLOUD_TOTAL_H  (LL_CLOUD_COUNT * LL_CLOUD_ROW_H - LL_VIEW_GAP)
// 第 i 枚按钮的 y（直接沿用九宫格行距，视觉上三行对齐）
#define LL_CLOUD_Y(i)     (LL_VIEW_Y0 + (i) * LL_CLOUD_ROW_H)

// ---- “设置”按钮（控制面板右上角空位，即“右转”右侧；与面板同显同隐）------
#define LL_SETTINGS_BTN_X (LL_VIEW_X0 + LL_VIEW_TOTAL_W + LL_VIEW_GAP)
#define LL_SETTINGS_BTN_Y (LL_VIEW_Y0)
#define LL_SETTINGS_BTN_W (LL_VIEW_BTN_W)
#define LL_SETTINGS_BTN_H (LL_VIEW_BTN_H)

// ---- 设置界面（5x4 显式按钮；标题在第一行按钮上方）--------------------------
#define LL_SET_COLS        (UI_LL_SETTING_COLS)
#define LL_SET_ROWS        (UI_LL_SETTING_ROWS)
// 与控制按钮完全一致的尺寸/间距：56x40、间隙 6；5 列正好横跨 x=8..312，
// 4 行从控制面板首行 y=46 开始，末行到 y=224（底部状态栏 y>=227）。
#define LL_SET_BTN_W       (56)
#define LL_SET_BTN_H       (40)
#define LL_SET_GAP         (6)
#define LL_SET_X0          ((SCREEN_WIDTH - (LL_SET_COLS * LL_SET_BTN_W + (LL_SET_COLS - 1) * LL_SET_GAP)) / 2)
#define LL_SET_Y0          (LL_VIEW_Y0)
#define LL_SET_TITLE_X     (LL_SET_X0)
#define LL_SET_TITLE_Y     (LL_SET_Y0 - 18) // 标题行在第一行按钮上方，紧接顶部按钮下方

// ---- 配色 -----------------------------------------------------------------
#define LL_COLOR_BG_R       (18)
#define LL_COLOR_BG_G       (20)
#define LL_COLOR_BG_B       (26)
#define LL_COLOR_ACTIVE_R   (28)
#define LL_COLOR_ACTIVE_G   (78)
#define LL_COLOR_ACTIVE_B   (150)
#define LL_COLOR_PRESS_R    (64)
#define LL_COLOR_PRESS_G    (120)
#define LL_COLOR_PRESS_B    (200)
#define LL_COLOR_TEXT_R     (240)
#define LL_COLOR_TEXT_G     (240)
#define LL_COLOR_TEXT_B     (240)
#define LL_COLOR_RADIUS     (5)

// 顶部按钮 / 时间按钮 / 视角按钮的填充透明度（视角为“半透明”）
#define LL_ALPHA_TOP        (210)
#define LL_ALPHA_TIME       (210)
#define LL_ALPHA_VIEW       (100)
// 云参数按钮：两行文字（名称+现值）信息密度高，半透明 100 在正午亮天空上会洗掉
// 文字对比度，故单独用更实的底色（仍能透出背景）
#define LL_ALPHA_CLOUD      (168)
#define LL_ALPHA_CLOUD_DIM  (110)
#define LL_COLOR_DIM_R      (168)
#define LL_COLOR_DIM_G      (176)
#define LL_COLOR_DIM_B      (190)

static int32_t s_controls_visible = 0; // 时间按钮与视角九宫格同显同隐
static int32_t s_pressed = UI_LL_CTRL_NONE; // 当前按住的控件动作（用于高亮）
static uint8_t s_dirty = 0;                // 面板显隐/按住高亮变化，供调用方决定是否重绘

// 拖动视角状态：控制面板隐藏时，按下点不在顶部按钮上即启动拖动
static int32_t s_drag_active = 0;
static int32_t s_drag_last_x = 0;
static int32_t s_drag_last_y = 0;

// 设置界面按住高亮（-1=无；0..19=单元格索引）
static int32_t s_setting_pressed = -1;

// 顶部按钮从左到右的动作
static const int32_t S_TOP_ACTION[LL_TOP_COUNT] = {
    UI_LL_CTRL_CONTROLS, UI_LL_CTRL_EXIT,
};
static const wchar_t *S_TOP_LABEL[LL_TOP_COUNT] = {
    L"控制", L"退出",
};

// 时间按钮从左到右的动作
static const int32_t S_TIME_ACTION[LL_TIME_COUNT] = {
    UI_LL_CTRL_REWIND, UI_LL_CTRL_REALTIME, UI_LL_CTRL_RUNPAUSE, UI_LL_CTRL_FORWARD,
};
static const wchar_t *S_TIME_LABEL[LL_TIME_COUNT] = {
    L"<<", L"实时", L"运行/暂停", L">>",
};

// 视角九宫格：与十六宫格软键 1~9 一一对应
static const wchar_t *S_VIEW_LABEL[LL_VIEW_ROWS][LL_VIEW_COLS] = {
    {L"左转", L"低头", L"右转"},
    {L"左倾", L"归中", L"右倾"},
    {L"拉远", L"抬头", L"推近"},
};

// 体积云参数按钮（从上到下）：动作与名称（现值另外从玲珑仪配置读取并显示在下行）
static const int32_t S_CLOUD_ACTION[LL_CLOUD_COUNT] = {
    UI_LL_CTRL_CLOUD_COVERAGE, UI_LL_CTRL_CLOUD_LAYER, UI_LL_CTRL_CLOUD_PRECISION,
};
static const wchar_t *S_CLOUD_LABEL[LL_CLOUD_COUNT] = {
    L"云量", L"云类", L"云精度",
};

static int32_t ll_in_rect(int32_t x, int32_t y, int32_t rx, int32_t ry, int32_t rw, int32_t rh) {
    return (x >= rx && x < rx + rw && y >= ry && y < ry + rh);
}

// 命中判定：返回对应动作，未命中返回 NONE
static int32_t ll_hit_test(int32_t x, int32_t y) {
    // 顶部按钮
    if (ll_in_rect(x, y, LL_TOP_X0, LL_TOP_Y, LL_TOP_TOTAL_W, LL_TOP_BTN_H)) {
        for (int32_t i = 0; i < LL_TOP_COUNT; i++) {
            int32_t bx = LL_TOP_X0 + i * (LL_TOP_BTN_W + LL_TOP_GAP);
            if (ll_in_rect(x, y, bx, LL_TOP_Y, LL_TOP_BTN_W, LL_TOP_BTN_H)) {
                return S_TOP_ACTION[i];
            }
        }
        return UI_LL_CTRL_NONE;
    }

    // “设置”按钮（控制面板右上角空位，仅面板可见时）
    if (s_controls_visible && ll_in_rect(x, y, LL_SETTINGS_BTN_X, LL_SETTINGS_BTN_Y, LL_SETTINGS_BTN_W, LL_SETTINGS_BTN_H)) {
        return UI_LL_CTRL_SETTINGS;
    }

    // 时间控制按钮（仅“控制”面板可见时）
    if (s_controls_visible && ll_in_rect(x, y, LL_TIME_MARGIN, LL_TIME_Y, LL_TIME_TOTAL_W, LL_TIME_BTN_H)) {
        for (int32_t i = 0; i < LL_TIME_COUNT; i++) {
            int32_t bx = LL_TIME_MARGIN + i * (LL_TIME_BTN_W + LL_TIME_GAP);
            if (ll_in_rect(x, y, bx, LL_TIME_Y, LL_TIME_BTN_W, LL_TIME_BTN_H)) {
                return S_TIME_ACTION[i];
            }
        }
        return UI_LL_CTRL_NONE;
    }

    // 视角九宫格（仅“控制”面板可见时）
    if (s_controls_visible && ll_in_rect(x, y, LL_VIEW_X0, LL_VIEW_Y0, LL_VIEW_TOTAL_W, LL_VIEW_TOTAL_H)) {
        for (int32_t r = 0; r < LL_VIEW_ROWS; r++) {
            for (int32_t c = 0; c < LL_VIEW_COLS; c++) {
                int32_t bx = LL_VIEW_X0 + c * (LL_VIEW_BTN_W + LL_VIEW_GAP);
                int32_t by = LL_VIEW_Y0 + r * (LL_VIEW_BTN_H + LL_VIEW_GAP);
                if (ll_in_rect(x, y, bx, by, LL_VIEW_BTN_W, LL_VIEW_BTN_H)) {
                    return UI_LL_CTRL_VIEW_1 + r * LL_VIEW_COLS + c;
                }
            }
        }
        return UI_LL_CTRL_NONE;
    }

    // 体积云参数按钮（九宫格左侧竖排；仅“控制”面板可见时）
    if (s_controls_visible && ll_in_rect(x, y, LL_CLOUD_X0, LL_VIEW_Y0, LL_CLOUD_BTN_W, LL_CLOUD_TOTAL_H)) {
        for (int32_t i = 0; i < LL_CLOUD_COUNT; i++) {
            if (ll_in_rect(x, y, LL_CLOUD_X0, LL_CLOUD_Y(i), LL_CLOUD_BTN_W, LL_CLOUD_BTN_H)) {
                return S_CLOUD_ACTION[i];
            }
        }
        return UI_LL_CTRL_NONE;
    }

    return UI_LL_CTRL_NONE;
}

void ui_linglong_ctrl_init(void) {
    ui_linglong_ctrl_reset();
}

void ui_linglong_ctrl_reset(void) {
    s_controls_visible = 0;
    s_pressed = UI_LL_CTRL_NONE;
    s_dirty = 0;
    s_drag_active = 0;
    s_drag_last_x = 0;
    s_drag_last_y = 0;
}

void ui_linglong_ctrl_settings_reset(void) {
    s_setting_pressed = -1;
}

uint8_t ui_linglong_ctrl_take_dirty(void) {
    uint8_t d = s_dirty;
    s_dirty = 0;
    return d;
}

int32_t ui_linglong_ctrl_poll_drag(int32_t touch_edge, int32_t is_touching,
                                   int32_t x, int32_t y,
                                   int32_t down_x, int32_t down_y,
                                   int32_t *out_dx, int32_t *out_dy) {
    if (out_dx) *out_dx = 0;
    if (out_dy) *out_dy = 0;

    // 控制面板可见时触屏归按钮/面板，不拖动视角
    if (s_controls_visible) {
        s_drag_active = 0;
        return 0;
    }

    int32_t down = (touch_edge & TOUCH_EDGE_DOWN) != 0;
    int32_t up   = (touch_edge & TOUCH_EDGE_UP) != 0;

    if (down) {
        // 顶部按钮上的按下属于点击，不启动拖动（面板隐藏时顶部按钮仍常驻）
        if (ll_hit_test(down_x, down_y) == UI_LL_CTRL_NONE) {
            s_drag_active = 1;
            s_drag_last_x = down_x;
            s_drag_last_y = down_y;
        }
        else {
            s_drag_active = 0;
        }
        // 亚帧点按（DOWN+UP 同帧）：不产生拖动位移
        if (up) {
            s_drag_active = 0;
            return 0;
        }
    }

    if (!s_drag_active) return 0;

    // 松开沿：用最后有效触点计算最后一段位移，然后结束拖动
    if (!is_touching && !up) {
        s_drag_active = 0;
        return 0;
    }

    int32_t dx = x - s_drag_last_x;
    int32_t dy = y - s_drag_last_y;
    s_drag_last_x = x;
    s_drag_last_y = y;
    if (up) {
        s_drag_active = 0;
    }
    if (dx == 0 && dy == 0) return 0;

    if (out_dx) *out_dx = dx;
    if (out_dy) *out_dy = dy;
    return 1;
}

int32_t ui_linglong_ctrl_handle_touch(int32_t touch_edge, int32_t x, int32_t y,
                                      int32_t down_x, int32_t down_y) {
    int32_t action = UI_LL_CTRL_NONE;

    // 按下沿：只锁存高亮，不触发动作（状态切换一律在松手沿，遵循项目触屏范式）
    if (touch_edge & TOUCH_EDGE_DOWN) {
        int32_t pressed = ll_hit_test(down_x, down_y);
        if (pressed != s_pressed) {
            s_pressed = pressed;
            s_dirty = 1;
        }
    }

    // 松开沿：按下点与松开点必须命中同一个按钮才算一次点击（拖动离手不触发）
    if (touch_edge & TOUCH_EDGE_UP) {
        int32_t down_action = ll_hit_test(down_x, down_y);
        int32_t up_action = ll_hit_test(x, y);
        if (down_action != UI_LL_CTRL_NONE && down_action == s_pressed && up_action == down_action) {
            action = down_action;
        }
        if (s_pressed != UI_LL_CTRL_NONE) {
            s_pressed = UI_LL_CTRL_NONE;
            s_dirty = 1;
        }

        // 顶部“控制”：两组控件同显同隐
        if (action == UI_LL_CTRL_CONTROLS) {
            s_controls_visible = !s_controls_visible;
            s_dirty = 1;
        }
    }

    if (action != UI_LL_CTRL_NONE) {
        s_dirty = 1;
    }

    return action;
}

// 设置界面命中：返回单元格索引，未命中 -1
static int32_t ll_setting_hit_test(int32_t x, int32_t y) {
    for (int32_t r = 0; r < LL_SET_ROWS; r++) {
        for (int32_t c = 0; c < LL_SET_COLS; c++) {
            int32_t bx = LL_SET_X0 + c * (LL_SET_BTN_W + LL_SET_GAP);
            int32_t by = LL_SET_Y0 + r * (LL_SET_BTN_H + LL_SET_GAP);
            if (ll_in_rect(x, y, bx, by, LL_SET_BTN_W, LL_SET_BTN_H)) {
                return r * LL_SET_COLS + c;
            }
        }
    }
    return -1;
}

int32_t ui_linglong_ctrl_settings_handle_touch(int32_t touch_edge, int32_t x, int32_t y,
                                               int32_t down_x, int32_t down_y) {
    int32_t cell = -1;

    if (touch_edge & TOUCH_EDGE_DOWN) {
        s_setting_pressed = ll_setting_hit_test(down_x, down_y);
    }

    if (touch_edge & TOUCH_EDGE_UP) {
        int32_t down_cell = ll_setting_hit_test(down_x, down_y);
        int32_t up_cell = ll_setting_hit_test(x, y);
        if (down_cell >= 0 && down_cell == s_setting_pressed && up_cell == down_cell) {
            cell = down_cell;
        }
        s_setting_pressed = -1;
    }

    return cell;
}

// 单个按钮：底色 + 居中文字（active=功能激活，pressed=当前按住）
static void ll_draw_button(Nano_GFX *gfx, int32_t x, int32_t y, int32_t w, int32_t h,
                           const wchar_t *label, int32_t active, int32_t pressed, uint8_t alpha) {
    uint8_t r = LL_COLOR_BG_R, g = LL_COLOR_BG_G, b = LL_COLOR_BG_B;
    if (active) {
        r = LL_COLOR_ACTIVE_R;
        g = LL_COLOR_ACTIVE_G;
        b = LL_COLOR_ACTIVE_B;
    }
    if (pressed) {
        r = LL_COLOR_PRESS_R;
        g = LL_COLOR_PRESS_G;
        b = LL_COLOR_PRESS_B;
    }

    gfx_draw_rounded_rectangle(gfx, x, y, w, h,
        LL_COLOR_RADIUS, LL_COLOR_RADIUS, LL_COLOR_RADIUS, LL_COLOR_RADIUS,
        r, g, b, alpha);
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)label,
        x + w / 2, y + h / 2,
        LL_COLOR_TEXT_R, LL_COLOR_TEXT_G, LL_COLOR_TEXT_B, pressed ? 255 : 235);
}

static void ll_draw_top(Nano_GFX *gfx) {
    for (int32_t i = 0; i < LL_TOP_COUNT; i++) {
        int32_t x = LL_TOP_X0 + i * (LL_TOP_BTN_W + LL_TOP_GAP);
        int32_t active = (S_TOP_ACTION[i] == UI_LL_CTRL_CONTROLS && s_controls_visible);
        int32_t pressed = (s_pressed == S_TOP_ACTION[i]);
        ll_draw_button(gfx, x, LL_TOP_Y, LL_TOP_BTN_W, LL_TOP_BTN_H,
            S_TOP_LABEL[i], active, pressed, LL_ALPHA_TOP);
    }
}

static void ll_draw_time(Nano_GFX *gfx, int32_t running_state, int32_t speed) {
    if (!s_controls_visible) return;

    // 底衬：让按钮在下四分之一区域（地景/天空）上保持可读，同时仍能透出背景
    gfx_draw_rounded_rectangle(gfx, 4, LL_TIME_BG_Y, SCREEN_WIDTH - 8, LL_TIME_BG_H,
        8, 8, 8, 8, 0, 0, 0, 70);

    for (int32_t i = 0; i < LL_TIME_COUNT; i++) {
        int32_t x = LL_TIME_MARGIN + i * (LL_TIME_BTN_W + LL_TIME_GAP);
        int32_t active = 0;
        if (S_TIME_ACTION[i] == UI_LL_CTRL_REALTIME) {
            active = (running_state == 2);
        }
        else if (S_TIME_ACTION[i] == UI_LL_CTRL_RUNPAUSE) {
            active = (running_state != 0);
        }
        else if (S_TIME_ACTION[i] == UI_LL_CTRL_REWIND) {
            active = (running_state == 1 && speed < 0);
        }
        else if (S_TIME_ACTION[i] == UI_LL_CTRL_FORWARD) {
            active = (running_state == 1 && speed > 0);
        }
        int32_t pressed = (s_pressed == S_TIME_ACTION[i]);
        ll_draw_button(gfx, x, LL_TIME_Y, LL_TIME_BTN_W, LL_TIME_BTN_H,
            S_TIME_LABEL[i], active, pressed, LL_ALPHA_TIME);
    }
}

static void ll_draw_view(Nano_GFX *gfx) {
    if (!s_controls_visible) return;

    for (int32_t r = 0; r < LL_VIEW_ROWS; r++) {
        for (int32_t c = 0; c < LL_VIEW_COLS; c++) {
            int32_t x = LL_VIEW_X0 + c * (LL_VIEW_BTN_W + LL_VIEW_GAP);
            int32_t y = LL_VIEW_Y0 + r * (LL_VIEW_BTN_H + LL_VIEW_GAP);
            int32_t action = UI_LL_CTRL_VIEW_1 + r * LL_VIEW_COLS + c;
            int32_t pressed = (s_pressed == action);
            ll_draw_button(gfx, x, y, LL_VIEW_BTN_W, LL_VIEW_BTN_H,
                S_VIEW_LABEL[r][c], 0, pressed, pressed ? 180 : LL_ALPHA_VIEW);
        }
    }
}

// 两行按钮：上行名称，下行现值（体积云三个参数按钮用，现值直接可见）
// enabled=0 时（当前天空模型不吃体积云参数）整枚画暗，避免“点了没反应”的困惑；
// 点按仍然生效（参数被记住，切到体积云档位即体现）。
static void ll_draw_option_button(Nano_GFX *gfx, int32_t x, int32_t y, int32_t w, int32_t h,
                                 const wchar_t *label, const wchar_t *value,
                                 int32_t pressed, int32_t enabled) {
    uint8_t r = LL_COLOR_BG_R, g = LL_COLOR_BG_G, b = LL_COLOR_BG_B;
    if (pressed) {
        r = LL_COLOR_PRESS_R;
        g = LL_COLOR_PRESS_G;
        b = LL_COLOR_PRESS_B;
    }
    uint8_t alpha = enabled ? LL_ALPHA_CLOUD : LL_ALPHA_CLOUD_DIM;
    gfx_draw_rounded_rectangle(gfx, x, y, w, h,
        LL_COLOR_RADIUS, LL_COLOR_RADIUS, LL_COLOR_RADIUS, LL_COLOR_RADIUS,
        r, g, b, pressed ? 205 : alpha);
    // 名称行（白/暗灰）与现值行（淡青/暗青，突出可变量）
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)label,
        x + w / 2, y + h / 2 - 9,
        enabled ? LL_COLOR_TEXT_R : LL_COLOR_DIM_R,
        enabled ? LL_COLOR_TEXT_G : LL_COLOR_DIM_G,
        enabled ? LL_COLOR_TEXT_B : LL_COLOR_DIM_B,
        enabled ? 235 : 200);
    gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)value,
        x + w / 2, y + h / 2 + 9,
        enabled ? 200 : LL_COLOR_DIM_R,
        enabled ? 232 : LL_COLOR_DIM_G,
        enabled ? 255 : LL_COLOR_DIM_B,
        enabled ? 245 : 200);
}

static void ll_draw_cloud(Nano_GFX *gfx, const UiLinglongCtrlState *st) {
    if (!s_controls_visible) return;

    // 云类：按掩码在档位表里找回显名（表由 celestial.c 维护，UI 与控制逻辑共用）
    int32_t layer_num = 0;
    const Linglong_Cloud_Option *layers = linglong_cloud_layer_options(&layer_num);
    const wchar_t *layer_name = layers[0].name;
    for (int32_t i = 0; i < layer_num; i++) {
        if (layers[i].value == st->cloud_layer_mask) { layer_name = layers[i].name; break; }
    }
    // 云精度：档位序号直接索引
    int32_t prec_num = 0;
    const Linglong_Cloud_Option *precs = linglong_cloud_precision_options(&prec_num);
    int32_t prec = st->cloud_precision_level;
    if (prec < 0 || prec >= prec_num) prec = 0;

    const wchar_t *values[LL_CLOUD_COUNT];
    values[0] = ui_cloud_coverage_level_name(st->cloud_coverage_level);
    values[1] = layer_name;
    values[2] = precs[prec].name;

    for (int32_t i = 0; i < LL_CLOUD_COUNT; i++) {
        int32_t pressed = (s_pressed == S_CLOUD_ACTION[i]);
        ll_draw_option_button(gfx, LL_CLOUD_X0, LL_CLOUD_Y(i), LL_CLOUD_BTN_W, LL_CLOUD_BTN_H,
            S_CLOUD_LABEL[i], values[i], pressed, st->cloud_enabled);
    }
}

// 控制面板右上角的“设置”按钮（与视角按钮同一半透明样式）
static void ll_draw_settings_button(Nano_GFX *gfx) {
    if (!s_controls_visible) return;
    int32_t pressed = (s_pressed == UI_LL_CTRL_SETTINGS);
    ll_draw_button(gfx, LL_SETTINGS_BTN_X, LL_SETTINGS_BTN_Y, LL_SETTINGS_BTN_W, LL_SETTINGS_BTN_H,
        L"设置", 0, pressed, pressed ? 180 : LL_ALPHA_VIEW);
}

// 设置界面的单个按钮：圆角半透明底 + 标签/现值两行；返回按钮用特殊配色与单行大字。
// 调用方已把背景清成深色，故这里用略实的底色保证文字可读。
static void ll_draw_setting_button(Nano_GFX *gfx, int32_t x, int32_t y, int32_t w, int32_t h,
                                   const UiLinglongSettingCell *cell, int32_t pressed) {
    // 与云参数按钮同一半透明底色（两行文字需要比单行视角按钮更实一点）
    uint8_t r = LL_COLOR_BG_R, g = LL_COLOR_BG_G, b = LL_COLOR_BG_B;
    uint8_t alpha = LL_ALPHA_CLOUD;

    if (cell->is_special) {
        // “返回”特殊样式：红色系，略实以保证亮天空上仍明显
        r = 150; g = 60; b = 50;
        alpha = 230;
    }
    else if (cell->label == NULL) {
        // 无功能备用按钮：更透、无文字，只保留 5x4 网格的占位感
        alpha = 100;
    }

    if (pressed) {
        if (cell->is_special) { r = 200; g = 100; b = 80; }
        else { r = LL_COLOR_PRESS_R; g = LL_COLOR_PRESS_G; b = LL_COLOR_PRESS_B; }
        alpha = 210;
    }

    gfx_draw_rounded_rectangle(gfx, x, y, w, h,
        LL_COLOR_RADIUS, LL_COLOR_RADIUS, LL_COLOR_RADIUS, LL_COLOR_RADIUS,
        r, g, b, alpha);

    if (cell->label == NULL) return;

    if (cell->is_special) {
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_16, (wchar_t *)cell->label,
            x + w / 2, y + h / 2,
            LL_COLOR_TEXT_R, LL_COLOR_TEXT_G, LL_COLOR_TEXT_B, 255);
    }
    else {
        gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)cell->label,
            x + w / 2, y + h / 2 - 8,
            LL_COLOR_TEXT_R, LL_COLOR_TEXT_G, LL_COLOR_TEXT_B, 255);
        if (cell->value != NULL) {
            gfx_font_draw_text_centered(gfx, GFX_FONT_ALPHA_12, (wchar_t *)cell->value,
                x + w / 2, y + h / 2 + 8,
                cell->value_r, cell->value_g, cell->value_b, 255);
        }
    }
}

void ui_linglong_ctrl_draw_settings(Nano_GFX *gfx, const UiLinglongSettingCell *cells) {
    // 标题：左上角、第一行按钮上方；黑色阴影保证亮天空上仍可读
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, (wchar_t *)L"玲珑天象仪设置",
        LL_SET_TITLE_X + 1, LL_SET_TITLE_Y + 1, 0, 0, 0, 220);
    gfx_font_draw_text(gfx, GFX_FONT_ALPHA_12, (wchar_t *)L"玲珑天象仪设置",
        LL_SET_TITLE_X, LL_SET_TITLE_Y, 240, 240, 240, 255);

    for (int32_t r = 0; r < LL_SET_ROWS; r++) {
        for (int32_t c = 0; c < LL_SET_COLS; c++) {
            int32_t idx = r * LL_SET_COLS + c;
            int32_t x = LL_SET_X0 + c * (LL_SET_BTN_W + LL_SET_GAP);
            int32_t y = LL_SET_Y0 + r * (LL_SET_BTN_H + LL_SET_GAP);
            ll_draw_setting_button(gfx, x, y, LL_SET_BTN_W, LL_SET_BTN_H,
                &cells[idx], s_setting_pressed == idx);
        }
    }
}

void ui_linglong_ctrl_draw(Nano_GFX *gfx, const UiLinglongCtrlState *st) {
    ll_draw_top(gfx);
    ll_draw_time(gfx, st->running_state, st->speed);
    ll_draw_view(gfx);
    ll_draw_cloud(gfx, st);
    ll_draw_settings_button(gfx);
}
