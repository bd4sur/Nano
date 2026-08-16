// ui_calendar.c - 日历（每周从周一开始，每月一页）
//
// 进入时定位到系统当前年月，今天以高亮色块标出；静止画面不做重复推帧。
//   - 左右键：切换上月/下月；上下键：切换上一年/下一年；
//   - 0 键：回到系统当前年月；A(ESC) 或 D(回车)：返回小游戏菜单。

#include "ui_calendar.h"
#include "ui_color.h"
#include "ui_almanac.h"
#include "input_device.h"
#include "touch.h"

// 支持显示的年份范围（公历）。
#define CALENDAR_MIN_YEAR (1900)
#define CALENDAR_MAX_YEAR (2100)

// 当前显示的年月（进入时初始化为系统当前年月）
static int32_t s_cal_year = 0;
static int32_t s_cal_month = 0; // 1..12

// 重绘控制：静态画面不重复推帧；切换年月或跨天时置位
static int32_t s_cal_dirty = 1;
static int32_t s_cal_last_day = 0;

// 黄历模态框状态：点击日历上的日期数字打开，点击任意处/任意键关闭
static int32_t s_almanac_active = 0;          // 1=模态框打开（此时黄历模块持有结果）
static int32_t s_almanac_dirty  = 1;          // 模态框需要重绘
static int32_t s_almanac_touch_prev = 0;      // 触屏按下沿检测（上一帧是否按住）
static int32_t s_almanac_open_timestamp = 0;  // 打开时刻，用于吞掉同一手势的残留按键
static int32_t s_almanac_close_timestamp = 0; // 关闭时刻，同理吞掉关闭手势的残留按键
// 模态框内容固定：打开时计算并绘制一次，不随年月切换失效


// 是否闰年（公历）
static int32_t ui_calendar_is_leap_year(int32_t year) {
    return ((year % 4 == 0) && (year % 100 != 0)) || (year % 400 == 0);
}

// 某月天数
static int32_t ui_calendar_days_in_month(int32_t year, int32_t month) {
    static const int32_t days[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && ui_calendar_is_leap_year(year)) {
        return 29;
    }
    return days[month - 1];
}

// 某年某月 1 日是星期几（周一=0 .. 周日=6）
// 采用 Tomohiko Sakamoto 算法（结果 0=周日），再转换为周一起始。
static int32_t ui_calendar_first_weekday(int32_t year, int32_t month) {
    static const int32_t t[12] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    int32_t y = year - (month < 3 ? 1 : 0);
    int32_t dow = (y + y / 4 - y / 100 + y / 400 + t[month - 1] + 1) % 7; // 0=周日
    return (dow + 6) % 7; // 周一=0
}

// 触屏命中测试：把触点映射到当前显示的日期数字格子。
// 几何与 ui_calendar_render_frame 的网格布局保持一致（仅当显示本月时有效）。
// 命中返回日期数字（1..当月天数），未命中返回 0。
static int32_t ui_calendar_touch_hit(int32_t touch_x, int32_t touch_y, Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    int32_t line_height = gfx_font_line_height(global_state->ui_font);
    int32_t header_height = line_height + 1;
    int32_t footer_height = line_height + 1;
    int32_t top = header_height;
    int32_t bottom = gfx->height - footer_height;
    int32_t day_grid_top = top + 2 + line_height + 4;
    int32_t row_height = (bottom - day_grid_top) / 6;

    if (touch_x < 0 || touch_x >= (int32_t)gfx->width || touch_y < day_grid_top || touch_y >= bottom) {
        return 0;
    }
    int32_t col = touch_x * 7 / (int32_t)gfx->width;
    int32_t row = (touch_y - day_grid_top) / row_height;
    if (row < 0 || row > 5) return 0;

    int32_t idx = row * 7 + col;
    int32_t first_wd = ui_calendar_first_weekday(s_cal_year, s_cal_month);
    int32_t dim = ui_calendar_days_in_month(s_cal_year, s_cal_month);
    int32_t day = idx - first_wd + 1;
    if (day < 1 || day > dim) return 0;
    return day;
}


void ui_calendar_init(Key_Event *key_event, Global_State *global_state) {
    (void)key_event;
    s_cal_year = global_state->year;
    s_cal_month = global_state->month;
    if (s_cal_year < CALENDAR_MIN_YEAR) s_cal_year = CALENDAR_MIN_YEAR;
    if (s_cal_year > CALENDAR_MAX_YEAR) s_cal_year = CALENDAR_MAX_YEAR;
    if (s_cal_month < 1 || s_cal_month > 12) s_cal_month = 1;
    s_cal_dirty = 1;
    s_cal_last_day = 0;
    // 防御：若上次退出时模态框仍打开（正常流程不会发生），收尾释放 PSRAM
    if (s_almanac_active) {
        ui_almanac_close();
        s_almanac_active = 0;
        s_almanac_dirty = 1;
        s_almanac_touch_prev = 0;
    }
}


int32_t ui_calendar_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 黄历模态框激活：绘制模态框（打开/关闭时置位 dirty），静态画面不重复推帧
    if (s_almanac_active) {
        if (s_almanac_dirty) {
            ui_almanac_draw(global_state->gfx);
            gfx_refresh(global_state->gfx);
            s_almanac_dirty = 0;
        }
        return 0;
    }

    // 静态画面不重复推帧；仅进入、切换年月、以及显示本月时跨天（移动“今天”高亮）才重绘
    if (!s_cal_dirty) {
        int32_t is_current_month = (s_cal_year == global_state->year && s_cal_month == global_state->month);
        if (!(is_current_month && s_cal_last_day != global_state->day)) {
            return 0;
        }
    }

    Nano_GFX *gfx = global_state->gfx;
    uint32_t font_id = global_state->ui_font;
    int32_t line_height = gfx_font_line_height(font_id);
    int32_t header_height = line_height + 1;
    int32_t footer_height = line_height + 1;

    uint8_t bg_R, bg_G, bg_B;
    uint8_t text_R, text_G, text_B;
    uint8_t week_end_R, week_end_G, week_end_B;   // 周末（六、日）文字
    uint8_t today_bg_R, today_bg_G, today_bg_B;   // “今天”高亮色块
    uint8_t line_R, line_G, line_B;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        bg_R = 255; bg_G = 255; bg_B = 255;
        text_R = 0; text_G = 0; text_B = 0;
        week_end_R = 226; week_end_G = 54; week_end_B = 54;
        today_bg_R = 25; today_bg_G = 118; today_bg_B = 210;
        line_R = 176; line_G = 176; line_B = 176;
    }
    else {
        bg_R = 6; bg_G = 6; bg_B = 6;
        text_R = 255; text_G = 255; text_B = 255;
        week_end_R = 255; week_end_G = 80; week_end_B = 80;
        today_bg_R = 66; today_bg_G = 133; today_bg_B = 244;
        line_R = 66; line_G = 66; line_B = 66;
    }

    // 页眉标题：居中显示 “YYYY年M月”
    wchar_t title[24];
    swprintf(title, 24, L"%d年%d月", s_cal_year, s_cal_month);

    // 页眉/页脚先入帧缓冲，日历网格最后统一推帧
    ui_draw_header(key_event, global_state, title, 1);
    ui_draw_footer_softkeys(key_event, global_state, L"上月", L"今天", L"下月", L"返回");

    // 正文背景（页眉与页脚之间）
    int32_t top = header_height;
    int32_t bottom = gfx->height - footer_height;
    gfx_draw_rectangle(gfx, 0, top, gfx->width, bottom - top, bg_R, bg_G, bg_B, 1);

    // 星期表头（周一~周日），周末（六、日）红色
    static const wchar_t week_names[7][2] = {L"一", L"二", L"三", L"四", L"五", L"六", L"日"};
    int32_t weekday_row_center = top + 2 + line_height / 2;
    int32_t day_grid_top = top + 2 + line_height + 4;
    for (int32_t col = 0; col < 7; col++) {
        int32_t cx = (col * gfx->width + gfx->width / 2) / 7;
        int32_t is_weekend = (col >= 5);
        gfx_font_draw_text_centered(gfx, font_id, (wchar_t *)week_names[col], cx, weekday_row_center,
            is_weekend ? week_end_R : text_R,
            is_weekend ? week_end_G : text_G,
            is_weekend ? week_end_B : text_B, 1);
    }

    // 星期表头下方分隔线
    gfx_draw_line(gfx, 0, day_grid_top - 3, gfx->width - 1, day_grid_top - 3, line_R, line_G, line_B, 1);

    // 日期网格（固定 6 行，未排满的格子留空，切换月份时版式稳定）
    int32_t row_height = (bottom - day_grid_top) / 6;
    int32_t first_wd = ui_calendar_first_weekday(s_cal_year, s_cal_month); // 周一=0
    int32_t dim = ui_calendar_days_in_month(s_cal_year, s_cal_month);
    int32_t is_current_month = (s_cal_year == global_state->year && s_cal_month == global_state->month);

    wchar_t day_buf[4];
    for (int32_t day = 1; day <= dim; day++) {
        int32_t idx = first_wd + day - 1;
        int32_t col = idx % 7;
        int32_t row = idx / 7;
        if (row >= 6) break; // 理论上月最多占满 6 行，防御性保护
        int32_t cx = (col * gfx->width + gfx->width / 2) / 7;
        int32_t cy = day_grid_top + row * row_height + row_height / 2;
        int32_t is_weekend = (col >= 5);
        int32_t is_today = (is_current_month && day == global_state->day);

        swprintf(day_buf, 4, L"%d", day);
        if (is_today) {
            // “今天”：色块 + 白字高亮
            gfx_draw_rectangle(gfx, cx - 12, cy - row_height / 2 + 1, 25, row_height - 2,
                today_bg_R, today_bg_G, today_bg_B, 1);
            gfx_font_draw_text_centered(gfx, font_id, day_buf, cx, cy + 1, 255, 255, 255, 1);
        }
        else {
            gfx_font_draw_text_centered(gfx, font_id, day_buf, cx, cy,
                is_weekend ? week_end_R : text_R,
                is_weekend ? week_end_G : text_G,
                is_weekend ? week_end_B : text_B, 1);
        }
    }

    gfx_refresh(gfx);
    s_cal_dirty = 0;
    s_cal_last_day = global_state->day;

    return 0;
}


int32_t ui_calendar_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 触屏轮询（独立于 16 宫格键；按下沿判定见软键盘同款范式）：
    // 日期数字的点击无法经 16 宫格键区分（整个屏幕都被映射为 4x4 键），必须直读触屏。
    int32_t touch_x = 0, touch_y = 0, touch_pressed = 0;
    int32_t touch_edge = 0;
    if (touch_read(&touch_x, &touch_y, &touch_pressed) == 0) {
        touch_edge = (touch_pressed && !s_almanac_touch_prev);
    }

    // 黄历模态框激活：任意触屏按下沿、或（打开 300ms 后到达的）按键下降沿关闭。
    // 300ms 门限用于吞掉“打开该模态框的同一手势”经 16 宫格产生的残留按键事件。
    if (s_almanac_active) {
        int32_t key_edge = key_event->key_edge;
        int32_t since_open = global_state->timestamp - s_almanac_open_timestamp;
        if (touch_edge || (key_edge < 0 && since_open >= 300)) {
            ui_almanac_close();
            s_almanac_active = 0;
            s_almanac_close_timestamp = global_state->timestamp; // 吞掉关闭手势的残留按键
            s_cal_dirty = 1;   // 关闭后重绘日历页
        }
        s_almanac_touch_prev = touch_pressed;
        return 0;
    }

    // 关闭模态框的手势经 16 宫格产生的残留按键（在关闭后 ~150ms 内到达）直接吞掉，
    // 防止误触发切月/返回等（与打开侧 300ms 门限同一手势归并）
    if (key_event->key_edge < 0 &&
        (int32_t)(global_state->timestamp - s_almanac_close_timestamp) < 150) {
        return 0;
    }

    // 触屏按下沿命中日期数字 → 打开该日的黄历模态框；
    // 命中时吞掉本次按键事件（16 宫格键不再处理，防止误触发切月/返回等）。
    if (touch_edge) {
        int32_t day = ui_calendar_touch_hit(touch_x, touch_y, global_state);
        if (day != 0) {
            // 计算黄历（成功=内容；失败=错误态提示，如 1900/2100 边界外）
            ui_almanac_open(s_cal_year, s_cal_month, day,
                            global_state->hour, global_state->minute);
            s_almanac_active = 1;
            s_almanac_dirty = 1;
            s_almanac_open_timestamp = global_state->timestamp;
            s_cal_dirty = 1;   // 打开模态框后停止日历重绘
            s_almanac_touch_prev = touch_pressed;
            return 0;
        }
    }
    s_almanac_touch_prev = touch_pressed;

    // UI 事件仅认下降沿
    if (key_event->key_edge != -1 && key_event->key_edge != -2) {
        return 0;
    }

    switch (key_event->key_code) {
        case NANO_KEY_left: // 上月
            s_cal_month--;
            if (s_cal_month < 1) {
                s_cal_month = 12;
                s_cal_year--;
            }
            if (s_cal_year < CALENDAR_MIN_YEAR) s_cal_year = CALENDAR_MIN_YEAR;
            s_cal_dirty = 1;
            break;
        case NANO_KEY_right: // 下月
            s_cal_month++;
            if (s_cal_month > 12) {
                s_cal_month = 1;
                s_cal_year++;
            }
            if (s_cal_year > CALENDAR_MAX_YEAR) s_cal_year = CALENDAR_MAX_YEAR;
            s_cal_dirty = 1;
            break;
        case NANO_KEY_up: // 上一年
            if (s_cal_year > CALENDAR_MIN_YEAR) {
                s_cal_year--;
                s_cal_dirty = 1;
            }
            break;
        case NANO_KEY_down: // 下一年
            if (s_cal_year < CALENDAR_MAX_YEAR) {
                s_cal_year++;
                s_cal_dirty = 1;
            }
            break;
        case NANO_KEY_0: // 回到当前年月
            if (s_cal_year != global_state->year || s_cal_month != global_state->month) {
                s_cal_year = global_state->year;
                s_cal_month = global_state->month;
                if (s_cal_year < CALENDAR_MIN_YEAR) s_cal_year = CALENDAR_MIN_YEAR;
                if (s_cal_year > CALENDAR_MAX_YEAR) s_cal_year = CALENDAR_MAX_YEAR;
                s_cal_dirty = 1;
            }
            break;
        case NANO_KEY_esc:     // A 键返回小游戏菜单
        case NANO_KEY_enter:   // D（回车）返回小游戏菜单
            global_state->STATE = STATE_GAME_MENU;
            break;
        default:
            break;
    }
    return 0;
}
