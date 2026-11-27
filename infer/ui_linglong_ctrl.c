#include <stdint.h>
#include <string.h>

#include "platform.h"
#include "graphics.h"
#include "ui.h" // TOUCH_EDGE_DOWN / TOUCH_EDGE_UP
#include "ui_button.h"

#include "ui_linglong_ctrl.h"

// ===============================================================================
// 玲珑天象仪主界面显式触屏控件
//
// 布局（320x240 逻辑屏；位置按 SCREEN_WIDTH/HEIGHT 推导，按钮尺寸取适合掌机的固定触控尺寸）：
//   顶部靠右：控制 退出（2 个按钮）
//   “控制”同显同隐的控件：
//     底部四分之一区域（让出最底部状态栏）：<< 实时 运行/暂停 >>
//     画面中央：3x3 半透明视角九宫格（对应十六键数字键 1~9）
//     九宫格左侧竖排：云量 / 云类 / 云精度（与九宫格三行齐平，不改动已有按钮位置）
//     九宫格右上角空位（“右转”右侧）：设置
//
// 绘制与命中判定共用下方几何常量，保证“看得见的一定点得到”。
// ===============================================================================

// ---- 顶部按钮栏 -----------------------------------------------------------
// 控制 / 漫游 / 退出：漫游开启后底部行换成 6 键移动 pad（长按连续移动）
#define LL_TOP_COUNT      (3)
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
#define LL_TIME_BOTTOM    (14) // 最底部状态栏高度（与 ui_linglong_render_frame 的版本/时间行一致）
#define LL_TIME_Y         (SCREEN_HEIGHT - LL_TIME_BOTTOM - 4 - LL_TIME_BTN_H)
#define LL_TIME_BG_Y      (LL_TIME_Y - 4)
#define LL_TIME_BG_H      (LL_TIME_BTN_H + 8)

// ---- 漫游移动 pad（开启漫游后替换底部时间按钮行）---------------------------
#define LL_ROAM_COUNT     (6)
#define LL_ROAM_BTN_W     ((SCREEN_WIDTH - 2 * LL_TIME_MARGIN - (LL_ROAM_COUNT - 1) * LL_TIME_GAP) / LL_ROAM_COUNT)
#define LL_ROAM_TOTAL_W   (LL_ROAM_COUNT * LL_ROAM_BTN_W + (LL_ROAM_COUNT - 1) * LL_TIME_GAP)

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

// ---- 控制面板左右两列设置按钮（与九宫格三行齐平，不改动已有按钮位置）------
// 左列（原云量/云类/云精度位置）：投影算法、赤道坐标、地平坐标
// 右列：设置（顶格）、大气散射、地景（设置下方两格）
#define LL_OPT_COUNT      (3)
#define LL_OPT_BTN_W      (LL_VIEW_BTN_W)
#define LL_OPT_BTN_H      (LL_VIEW_BTN_H)
#define LL_OPT_GAP_X      (6)
#define LL_OPT_LEFT_X0    (LL_VIEW_X0 - LL_OPT_GAP_X - LL_OPT_BTN_W)
#define LL_OPT_RIGHT_X0   (LL_SETTINGS_BTN_X)
#define LL_OPT_ROW_H      (LL_VIEW_BTN_H + LL_VIEW_GAP)
#define LL_OPT_Y(i)       (LL_VIEW_Y0 + (i) * LL_OPT_ROW_H)

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
// 两行文字按钮（控制面板左右设置列 + 设置界面）：名称+现值信息密度高，
// 半透明 100 在正午亮天空上会洗掉文字对比度，故用更实的底色（仍能透出背景）
#define LL_ALPHA_OPTION      (168)
#define LL_ALPHA_OPTION_DIM  (110)
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
    UI_LL_CTRL_CONTROLS, UI_LL_CTRL_ROAM, UI_LL_CTRL_EXIT,
};
static const wchar_t *S_TOP_LABEL[LL_TOP_COUNT] = {
    L"控制", L"漫游", L"退出",
};

// 漫游移动 pad：动作、标签与方向位
static const int32_t S_ROAM_ACTION[LL_ROAM_COUNT] = {
    UI_LL_CTRL_MOVE_FWD, UI_LL_CTRL_MOVE_BACK, UI_LL_CTRL_MOVE_LEFT,
    UI_LL_CTRL_MOVE_RIGHT, UI_LL_CTRL_MOVE_UP, UI_LL_CTRL_MOVE_DOWN,
};
static const wchar_t *S_ROAM_LABEL[LL_ROAM_COUNT] = {
    L"前进", L"后退", L"左移", L"右移", L"上升", L"下降",
};
static const int32_t S_ROAM_BIT[LL_ROAM_COUNT] = {
    UI_LL_ROAM_FWD, UI_LL_ROAM_BACK, UI_LL_ROAM_LEFT,
    UI_LL_ROAM_RIGHT, UI_LL_ROAM_UP, UI_LL_ROAM_DOWN,
};

// 漫游状态（绘制/命中/按住轮询共用）：由 ui_linglong_ctrl_draw 的 state 刷新
static int32_t s_roam_enable = 0;
static int32_t s_roam_speed_level = 1;

// 时间按钮从左到右的动作
static const int32_t S_TIME_ACTION[LL_TIME_COUNT] = {
    UI_LL_CTRL_REWIND, UI_LL_CTRL_REALTIME, UI_LL_CTRL_RUNPAUSE, UI_LL_CTRL_FORWARD,
};
static const wchar_t *S_TIME_LABEL[LL_TIME_COUNT] = {
    L"<<", L"实时", L"运行/暂停", L">>",
};

// 视角九宫格：与十六键数字键 1~9 一一对应（正中“归中”不参与按住重复）
static const wchar_t *S_VIEW_LABEL[LL_VIEW_ROWS][LL_VIEW_COLS] = {
    {L"左转", L"低头", L"右转"},
    {L"左倾", L"归中", L"右倾"},
    {L"拉远", L"抬头", L"推近"},
};

// 控制面板左列（原云参数位置）：动作与名称
static const int32_t S_LEFT_ACTION[LL_OPT_COUNT] = {
    UI_LL_CTRL_PROJECTION, UI_LL_CTRL_EQUATORIAL, UI_LL_CTRL_HORIZONTAL,
};
static const wchar_t *S_LEFT_LABEL[LL_OPT_COUNT] = {
    L"投影算法", L"赤道坐标", L"地平坐标",
};

// 控制面板右列（设置按钮下方两格）：动作与名称
static const int32_t S_RIGHT_ACTION[2] = {
    UI_LL_CTRL_SKY_MODEL, UI_LL_CTRL_LANDSCAPE,
};
static const wchar_t *S_RIGHT_LABEL[2] = {
    L"大气散射", L"地景",
};

// 各设置项的现值名称（索引越界时统一回退到 0）
static const wchar_t *S_PROJECTION_NAME[2] = {L"鱼眼", L"透视"};
static const wchar_t *S_OFF_ON_NAME[2] = {L"关", L"开"};
static const wchar_t *S_HORIZONTAL_NAME[3] = {L"关", L"方位角", L"坐标圈"};
// 大气散射现值名（按下标与 Linglong_Config.sky_model 对应）。「简化模型」「一次散射」
// 两档已从选项里屏蔽（见 ui_linglong.c 的 ui_linglong_next_sky_model），循环不会落到它们，
// 这里保留名称只为“手改配置时也能显示出当前值”。
static const wchar_t *S_SKY_MODEL_NAME[6] = {L"关", L"简化模型", L"一次散射", L"二次散射", L"体积云", L"海洋天空"};
static const wchar_t *S_LANDSCAPE_NAME[4] = {L"关", L"草原", L"卫星照片", L"实时海洋"};

static int32_t ll_in_rect(int32_t x, int32_t y, int32_t rx, int32_t ry, int32_t rw, int32_t rh) {
    return (x >= rx && x < rx + rw && y >= ry && y < ry + rh);
}

// 通用按钮命中（定义在下方；ll_hit_test 先用）
static int32_t ll_rect_hit(int32_t x, int32_t y, int32_t w, int32_t h,
                           int32_t px, int32_t py);

// 命中判定：返回对应动作，未命中返回 NONE
static int32_t ll_hit_test(int32_t x, int32_t y) {
    // 顶部按钮
    if (ll_in_rect(x, y, LL_TOP_X0, LL_TOP_Y, LL_TOP_TOTAL_W, LL_TOP_BTN_H)) {
        for (int32_t i = 0; i < LL_TOP_COUNT; i++) {
            int32_t bx = LL_TOP_X0 + i * (LL_TOP_BTN_W + LL_TOP_GAP);
            if (ll_rect_hit(bx, LL_TOP_Y, LL_TOP_BTN_W, LL_TOP_BTN_H, x, y)) {
                return S_TOP_ACTION[i];
            }
        }
        return UI_LL_CTRL_NONE;
    }

    // “设置”按钮（控制面板右上角空位，仅面板可见时）
    if (s_controls_visible && ll_rect_hit(LL_SETTINGS_BTN_X, LL_SETTINGS_BTN_Y, LL_SETTINGS_BTN_W, LL_SETTINGS_BTN_H, x, y)) {
        return UI_LL_CTRL_SETTINGS;
    }

    // 漫游移动 pad（仅“控制”面板可见、且漫游开启时；位置与时间按钮行相同）
    if (s_controls_visible && s_roam_enable &&
        ll_in_rect(x, y, LL_TIME_MARGIN, LL_TIME_Y, LL_ROAM_TOTAL_W, LL_TIME_BTN_H)) {
        for (int32_t i = 0; i < LL_ROAM_COUNT; i++) {
            int32_t bx = LL_TIME_MARGIN + i * (LL_ROAM_BTN_W + LL_TIME_GAP);
            if (ll_rect_hit(bx, LL_TIME_Y, LL_ROAM_BTN_W, LL_TIME_BTN_H, x, y)) {
                return S_ROAM_ACTION[i];
            }
        }
        return UI_LL_CTRL_NONE;
    }

    // 时间控制按钮（仅“控制”面板可见时）
    if (s_controls_visible && !s_roam_enable &&
        ll_in_rect(x, y, LL_TIME_MARGIN, LL_TIME_Y, LL_TIME_TOTAL_W, LL_TIME_BTN_H)) {
        for (int32_t i = 0; i < LL_TIME_COUNT; i++) {
            int32_t bx = LL_TIME_MARGIN + i * (LL_TIME_BTN_W + LL_TIME_GAP);
            if (ll_rect_hit(bx, LL_TIME_Y, LL_TIME_BTN_W, LL_TIME_BTN_H, x, y)) {
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
                if (ll_rect_hit(bx, by, LL_VIEW_BTN_W, LL_VIEW_BTN_H, x, y)) {
                    return UI_LL_CTRL_VIEW_1 + r * LL_VIEW_COLS + c;
                }
            }
        }
        return UI_LL_CTRL_NONE;
    }

    // 控制面板左列设置按钮（原云量/云类/云精度位置）
    if (s_controls_visible && ll_in_rect(x, y, LL_OPT_LEFT_X0, LL_VIEW_Y0,
            LL_OPT_BTN_W, LL_OPT_COUNT * LL_OPT_ROW_H - LL_VIEW_GAP)) {
        for (int32_t i = 0; i < LL_OPT_COUNT; i++) {
            if (ll_rect_hit(LL_OPT_LEFT_X0, LL_OPT_Y(i), LL_OPT_BTN_W, LL_OPT_BTN_H, x, y)) {
                return S_LEFT_ACTION[i];
            }
        }
        return UI_LL_CTRL_NONE;
    }

    // 设置按钮下方两格：大气散射、地景
    if (s_controls_visible && ll_in_rect(x, y, LL_OPT_RIGHT_X0, LL_VIEW_Y0 + LL_OPT_ROW_H,
            LL_OPT_BTN_W, 2 * LL_OPT_ROW_H - LL_VIEW_GAP)) {
        for (int32_t i = 0; i < 2; i++) {
            if (ll_rect_hit(LL_OPT_RIGHT_X0, LL_OPT_Y(i + 1), LL_OPT_BTN_W, LL_OPT_BTN_H, x, y)) {
                return S_RIGHT_ACTION[i];
            }
        }
        return UI_LL_CTRL_NONE;
    }

    return UI_LL_CTRL_NONE;
}

// 视角方向按钮的按住查询（与 ll_hit_test 共用同一套几何，但只认可重复的 8 枚按钮）。
// 供调用方每帧轮询，实现“按住反复触发”。
int32_t ui_linglong_ctrl_view_held(int32_t is_touching, int32_t x, int32_t y) {
    if (!s_controls_visible || !is_touching) return UI_LL_CTRL_NONE;
    for (int32_t r = 0; r < LL_VIEW_ROWS; r++) {
        for (int32_t c = 0; c < LL_VIEW_COLS; c++) {
            int32_t action = UI_LL_CTRL_VIEW_1 + r * LL_VIEW_COLS + c;
            if (action == UI_LL_CTRL_VIEW_CENTER) continue; // 归中：单击生效，不重复
            int32_t bx = LL_VIEW_X0 + c * (LL_VIEW_BTN_W + LL_VIEW_GAP);
            int32_t by = LL_VIEW_Y0 + r * (LL_VIEW_BTN_H + LL_VIEW_GAP);
            if (ll_rect_hit(bx, by, LL_VIEW_BTN_W, LL_VIEW_BTN_H, x, y)) {
                return action;
            }
        }
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

void ui_linglong_ctrl_show_controls(void) {
    if (!s_controls_visible) {
        s_controls_visible = 1;
        s_dirty = 1;
    }
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
            if (ll_rect_hit(bx, by, LL_SET_BTN_W, LL_SET_BTN_H, x, y)) {
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

static int32_t ll_rect_hit(int32_t x, int32_t y, int32_t w, int32_t h,
                           int32_t px, int32_t py) {
    Ui_Button b;
    ui_button_init(&b, 0, x, y, w, h);
    b.style = UI_BTN_STYLE_ROUNDED;
    b.radius = LL_COLOR_RADIUS;
    return ui_button_hit(&b, px, py);
}

// 简单单行按钮的文本内容：常态 235、按下 255，保留原来的文字亮度差。
static void ll_simple_content(Nano_GFX *gfx, const Ui_Button *b,
                              int32_t base_cx, int32_t base_cy,
                              Ui_Button_Visual vis, void *user) {
    (void)base_cx;
    (void)base_cy;
    (void)user;
    if (b->label == NULL) return;
    int32_t cy = ui_button_label_cy(b, vis);
    uint8_t mode = (vis == UI_BTN_VIS_PRESSED || vis == UI_BTN_VIS_SELECTED_PRESSED) ? 255 : 235;
    gfx_font_draw_text_centered(gfx, b->font_id, (wchar_t *)b->label,
        b->x + b->w / 2, cy,
        LL_COLOR_TEXT_R, LL_COLOR_TEXT_G, LL_COLOR_TEXT_B, mode);
}

static void ll_simple_theme(Ui_Button_Theme *t, int32_t active, uint8_t alpha) {
    memset(t, 0, sizeof(*t));
    Ui_RGB base = active ? (Ui_RGB){LL_COLOR_ACTIVE_R, LL_COLOR_ACTIVE_G, LL_COLOR_ACTIVE_B}
                         : (Ui_RGB){LL_COLOR_BG_R, LL_COLOR_BG_G, LL_COLOR_BG_B};
    Ui_RGB press = {LL_COLOR_PRESS_R, LL_COLOR_PRESS_G, LL_COLOR_PRESS_B};
    Ui_RGB text = {LL_COLOR_TEXT_R, LL_COLOR_TEXT_G, LL_COLOR_TEXT_B};
    for (int32_t i = 0; i < UI_BTN_VIS_COUNT; i++) {
        t->bg[i] = base;
        t->fg[i] = text;
        t->bg_alpha[i] = alpha;
        t->border_alpha[i] = 0;
    }
    t->bg[UI_BTN_VIS_PRESSED] = press;
    t->bg[UI_BTN_VIS_SELECTED] = base;
    t->bg[UI_BTN_VIS_SELECTED_PRESSED] = press;
}

// 两行/特殊设置按钮的文本内容。颜色、字高、行偏移由调用方填入。
typedef struct {
    uint8_t lr, lg, lb;
    uint8_t vr, vg, vb;
    uint8_t label_alpha;
    uint8_t value_alpha;
    int16_t label_dy;
    int16_t value_dy;
    uint8_t has_value;
} LL_Cell_Content;

static void ll_cell_content(Nano_GFX *gfx, const Ui_Button *b,
                            int32_t base_cx, int32_t base_cy,
                            Ui_Button_Visual vis, void *user) {
    (void)base_cx;
    (void)base_cy;
    if (b->label == NULL) return;
    const LL_Cell_Content *c = (const LL_Cell_Content *)user;
    int32_t cy = ui_button_label_cy(b, vis);
    gfx_font_draw_text_centered(gfx, b->font_id, (wchar_t *)b->label,
        b->x + b->w / 2, cy + c->label_dy,
        c->lr, c->lg, c->lb, c->label_alpha);
    if (c->has_value && b->value != NULL) {
        gfx_font_draw_text_centered(gfx, b->value_font_id, (wchar_t *)b->value,
            b->x + b->w / 2, cy + c->value_dy,
            c->vr, c->vg, c->vb, c->value_alpha);
    }
}

static void ll_cell_theme(Ui_Button_Theme *t, Ui_RGB normal, Ui_RGB pressed,
                          uint8_t alpha_normal, uint8_t alpha_pressed) {
    memset(t, 0, sizeof(*t));
    Ui_RGB text = {LL_COLOR_TEXT_R, LL_COLOR_TEXT_G, LL_COLOR_TEXT_B};
    for (int32_t i = 0; i < UI_BTN_VIS_COUNT; i++) {
        t->bg[i] = normal;
        t->fg[i] = text;
        t->bg_alpha[i] = alpha_normal;
        t->border_alpha[i] = 0;
    }
    t->bg[UI_BTN_VIS_PRESSED] = pressed;
    t->bg_alpha[UI_BTN_VIS_PRESSED] = alpha_pressed;
    t->bg[UI_BTN_VIS_SELECTED] = normal;
    t->bg[UI_BTN_VIS_SELECTED_PRESSED] = pressed;
    t->bg_alpha[UI_BTN_VIS_SELECTED_PRESSED] = alpha_pressed;
}

// 单个按钮：底色 + 居中文字（active=功能激活，pressed=当前按住）
static void ll_draw_button(Nano_GFX *gfx, int32_t x, int32_t y, int32_t w, int32_t h,
                           const wchar_t *label, int32_t active, int32_t pressed, uint8_t alpha) {
    Ui_Button b;
    ui_button_init(&b, 0, x, y, w, h);
    b.style = UI_BTN_STYLE_ROUNDED;
    b.radius = LL_COLOR_RADIUS;
    b.flags = UI_BTN_FLAG_DRAW_BG;
    b.font_id = GFX_FONT_ALPHA_12;
    b.content_draw = ll_simple_content;
    ui_button_set_text(&b, label, NULL);

    Ui_Button_Theme t;
    ll_simple_theme(&t, active, alpha);

    Ui_Button_Visual vis;
    if (pressed) vis = active ? UI_BTN_VIS_SELECTED_PRESSED : UI_BTN_VIS_PRESSED;
    else         vis = active ? UI_BTN_VIS_SELECTED : UI_BTN_VIS_NORMAL;
    ui_button_draw(gfx, &b, &t, vis);
}

static void ll_draw_top(Nano_GFX *gfx) {
    for (int32_t i = 0; i < LL_TOP_COUNT; i++) {
        int32_t x = LL_TOP_X0 + i * (LL_TOP_BTN_W + LL_TOP_GAP);
        int32_t active = (S_TOP_ACTION[i] == UI_LL_CTRL_CONTROLS && s_controls_visible)
                      || (S_TOP_ACTION[i] == UI_LL_CTRL_ROAM && s_roam_enable);
        int32_t pressed = (s_pressed == S_TOP_ACTION[i]);
        ll_draw_button(gfx, x, LL_TOP_Y, LL_TOP_BTN_W, LL_TOP_BTN_H,
            S_TOP_LABEL[i], active, pressed, LL_ALPHA_TOP);
    }
}

static void ll_draw_roam(Nano_GFX *gfx, int32_t speed_level) {
    if (!s_controls_visible || !s_roam_enable) return;

    gfx_draw_rounded_rectangle(gfx, 4, LL_TIME_BG_Y, SCREEN_WIDTH - 8, LL_TIME_BG_H,
        8, 8, 8, 8, 0, 0, 0, 70);

    for (int32_t i = 0; i < LL_ROAM_COUNT; i++) {
        int32_t x = LL_TIME_MARGIN + i * (LL_ROAM_BTN_W + LL_TIME_GAP);
        int32_t pressed = (s_pressed == S_ROAM_ACTION[i]);
        // 上升/下降用不同底色区分（水平四键为主，垂直两键为辅）
        int32_t active = (i >= 4) ? 1 : 0;
        ll_draw_button(gfx, x, LL_TIME_Y, LL_ROAM_BTN_W, LL_TIME_BTN_H,
            S_ROAM_LABEL[i], active, pressed, LL_ALPHA_TIME);
    }
    (void)speed_level;
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
    Ui_Button b;
    ui_button_init(&b, 0, x, y, w, h);
    b.style = UI_BTN_STYLE_ROUNDED;
    b.radius = LL_COLOR_RADIUS;
    b.flags = UI_BTN_FLAG_DRAW_BG;
    b.font_id = GFX_FONT_ALPHA_12;
    b.value_font_id = GFX_FONT_ALPHA_12;
    b.content_draw = ll_cell_content;
    ui_button_set_text(&b, label, value);

    LL_Cell_Content content;
    content.lr = enabled ? LL_COLOR_TEXT_R : LL_COLOR_DIM_R;
    content.lg = enabled ? LL_COLOR_TEXT_G : LL_COLOR_DIM_G;
    content.lb = enabled ? LL_COLOR_TEXT_B : LL_COLOR_DIM_B;
    content.vr = enabled ? 200 : LL_COLOR_DIM_R;
    content.vg = enabled ? 232 : LL_COLOR_DIM_G;
    content.vb = enabled ? 255 : LL_COLOR_DIM_B;
    content.label_alpha = enabled ? 235 : 200;
    content.value_alpha = enabled ? 245 : 200;
    content.label_dy = -9;
    content.value_dy = 9;
    content.has_value = 1;
    b.content_user = &content;

    Ui_Button_Theme t;
    Ui_RGB normal = {LL_COLOR_BG_R, LL_COLOR_BG_G, LL_COLOR_BG_B};
    Ui_RGB press = {LL_COLOR_PRESS_R, LL_COLOR_PRESS_G, LL_COLOR_PRESS_B};
    ll_cell_theme(&t, normal, press,
                  enabled ? LL_ALPHA_OPTION : LL_ALPHA_OPTION_DIM, 205);
    ui_button_draw(gfx, &b, &t, pressed ? UI_BTN_VIS_PRESSED : UI_BTN_VIS_NORMAL);
}

static const wchar_t *ll_option_name(const wchar_t *const *table, int32_t count, int32_t index) {
    if (index < 0 || index >= count) index = 0;
    return table[index];
}

// 控制面板左右两列设置按钮：两行文字，上行名称、下行现值
static void ll_draw_control_options(Nano_GFX *gfx, const UiLinglongCtrlState *st) {
    if (!s_controls_visible) return;

    const wchar_t *left_values[LL_OPT_COUNT];
    left_values[0] = ll_option_name(S_PROJECTION_NAME, 2, st->projection);
    left_values[1] = ll_option_name(S_OFF_ON_NAME, 2, st->equatorial_coord);
    left_values[2] = ll_option_name(S_HORIZONTAL_NAME, 3, st->horizontal_coord);
    for (int32_t i = 0; i < LL_OPT_COUNT; i++) {
        int32_t pressed = (s_pressed == S_LEFT_ACTION[i]);
        ll_draw_option_button(gfx, LL_OPT_LEFT_X0, LL_OPT_Y(i), LL_OPT_BTN_W, LL_OPT_BTN_H,
            S_LEFT_LABEL[i], left_values[i], pressed, 1);
    }

    const wchar_t *right_values[2];
    right_values[0] = ll_option_name(S_SKY_MODEL_NAME, 6, st->sky_model);
    right_values[1] = ll_option_name(S_LANDSCAPE_NAME, 4, st->landscape_index);
    for (int32_t i = 0; i < 2; i++) {
        int32_t pressed = (s_pressed == S_RIGHT_ACTION[i]);
        ll_draw_option_button(gfx, LL_OPT_RIGHT_X0, LL_OPT_Y(i + 1), LL_OPT_BTN_W, LL_OPT_BTN_H,
            S_RIGHT_LABEL[i], right_values[i], pressed, 1);
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
// 设置界面叠加在天空上，用略实的底色保证文字可读。
static void ll_draw_setting_button(Nano_GFX *gfx, int32_t x, int32_t y, int32_t w, int32_t h,
                                   const UiLinglongSettingCell *cell, int32_t pressed) {
    // 与两行控制按钮同一半透明底色（两行文字需要比单行视角按钮更实一点）
    Ui_RGB normal = {LL_COLOR_BG_R, LL_COLOR_BG_G, LL_COLOR_BG_B};
    Ui_RGB press = {LL_COLOR_PRESS_R, LL_COLOR_PRESS_G, LL_COLOR_PRESS_B};
    uint8_t alpha_normal = LL_ALPHA_OPTION;
    uint8_t alpha_pressed = 210;

    if (cell->is_special) {
        // “返回”特殊样式：红色系，略实以保证亮天空上仍明显
        normal = (Ui_RGB){150, 60, 50};
        press = (Ui_RGB){200, 100, 80};
        alpha_normal = 230;
    }
    else if (cell->label == NULL) {
        // 无功能备用按钮：更透、无文字，只保留 5x4 网格的占位感
        alpha_normal = 100;
    }

    Ui_Button b;
    ui_button_init(&b, 0, x, y, w, h);
    b.style = UI_BTN_STYLE_ROUNDED;
    b.radius = LL_COLOR_RADIUS;
    b.flags = UI_BTN_FLAG_DRAW_BG;
    b.font_id = cell->is_special ? GFX_FONT_ALPHA_16 : GFX_FONT_ALPHA_12;
    b.value_font_id = GFX_FONT_ALPHA_12;
    b.content_draw = ll_cell_content;
    ui_button_set_text(&b, cell->label, cell->value);

    LL_Cell_Content content;
    content.lr = LL_COLOR_TEXT_R;
    content.lg = LL_COLOR_TEXT_G;
    content.lb = LL_COLOR_TEXT_B;
    content.vr = cell->value_r;
    content.vg = cell->value_g;
    content.vb = cell->value_b;
    content.label_alpha = 255;
    content.value_alpha = 255;
    content.label_dy = cell->is_special ? 0 : -8;
    content.value_dy = 8;
    content.has_value = (!cell->is_special && cell->value != NULL) ? 1 : 0;
    b.content_user = &content;

    Ui_Button_Theme t;
    ll_cell_theme(&t, normal, press, alpha_normal, alpha_pressed);
    ui_button_draw(gfx, &b, &t, pressed ? UI_BTN_VIS_PRESSED : UI_BTN_VIS_NORMAL);
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

int32_t ui_linglong_ctrl_roam_held(int32_t is_touching, int32_t x, int32_t y) {
    if (!s_controls_visible || !s_roam_enable || !is_touching) return 0;
    int32_t mask = 0;
    for (int32_t i = 0; i < LL_ROAM_COUNT; i++) {
        int32_t bx = LL_TIME_MARGIN + i * (LL_ROAM_BTN_W + LL_TIME_GAP);
        if (ll_rect_hit(bx, LL_TIME_Y, LL_ROAM_BTN_W, LL_TIME_BTN_H, x, y)) {
            mask |= S_ROAM_BIT[i];
        }
    }
    return mask;
}

void ui_linglong_ctrl_draw(Nano_GFX *gfx, const UiLinglongCtrlState *st) {
    s_roam_enable = st->roam_enable;
    s_roam_speed_level = st->roam_speed_level;
    ll_draw_top(gfx);
    if (s_roam_enable) ll_draw_roam(gfx, st->roam_speed_level);
    else ll_draw_time(gfx, st->running_state, st->speed);
    ll_draw_view(gfx);
    ll_draw_control_options(gfx, st);
    ll_draw_settings_button(gfx);
}
