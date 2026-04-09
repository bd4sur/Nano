#include <stdio.h>
#include <time.h>

#include "graphics.h"
#include "keyboard_hal.h"
#include "ui.h"

#include "platform.h"

#ifdef BADAPPLE_ENABLED
    #include "badapple.h"
#endif

#include "ephemeris.h"
#include "celestial.h"
#include "nongli.h"

#include "ui_app.h"


// ===============================================================================
// 获取按键事件
// ===============================================================================

void get_key_event(Key_Event *key_event, Global_State *global_state) {
    uint8_t key = keyboard_hal_read_key();
    // 边沿
    if (key_event->key_mask != 1 && (key != key_event->prev_key)) {
        // 按下瞬间（上升沿）
        if (key != KEYCODE_NUM_IDLE) {
            key_event->key_code = key;
            key_event->key_edge = 1;
        }
        // 松开瞬间（下降沿）
        else {
            key_event->key_code = key_event->prev_key;
            // 短按（或者通过长按触发重复动作状态后反复触发）
            if (key_event->key_repeat == 1 ||
                ((global_state->timestamp - key_event->key_timer) >= 0 &&
                    (global_state->timestamp - key_event->key_timer) < LONG_PRESS_THRESHOLD)) {
                key_event->key_edge = -1;
            }
            // 长按
            else if ((global_state->timestamp - key_event->key_timer) >= LONG_PRESS_THRESHOLD) {
                key_event->key_edge = -2;
                key_event->key_repeat = 1;
            }
        }
        key_event->key_timer = global_state->timestamp;
    }
    // 按住或松开
    else {
        // 按住
        if (key != KEYCODE_NUM_IDLE) {
            key_event->key_code = key;
            key_event->key_edge = 0;
            // key_event->key_timer++;
            // 若重复动作标记key_repeat在一次长按后点亮，则继续按住可以反复触发短按
            if (key_event->key_repeat == 1) {
                key_event->key_edge = -2;
                key_event->key_mask = 1; // 软复位置1，即强制恢复为无按键状态，以便下一次轮询检测到下降沿（尽管物理上有键按下），触发长按事件
                key = KEYCODE_NUM_IDLE; // 便于后面设置prev_key为KEYCODE_NUM_IDLE（无键按下）
                key_event->key_repeat = 1;
            }
            // 如果没有点亮动作标记key_repeat，则达到长按阈值后触发长按事件
            else if ((global_state->timestamp - key_event->key_timer) >= LONG_PRESS_THRESHOLD) {
                key_event->key_edge = -2;
                key_event->key_mask = 1; // 软复位置1，即强制恢复为无按键状态，以便下一次轮询检测到下降沿（尽管物理上有键按下），触发长按事件
                key = KEYCODE_NUM_IDLE; // 便于后面设置prev_key为KEYCODE_NUM_IDLE（无键按下）
            }
        }
        // 松开
        else {
            key_event->key_code = KEYCODE_NUM_IDLE;
            key_event->key_edge = 0;
            key_event->key_timer = global_state->timestamp;
            key_event->key_mask = 0;
            key_event->key_repeat = 0;
        }
    }
    key_event->prev_key = key;
}


// ===============================================================================
// 开机欢迎画面
// ===============================================================================

// 绘制点阵的版权信息
//   点阵数据通过bd4sur.com/html/am32.html取得
static void ui_draw_copyright_notice(Key_Event *key_event, Global_State *global_state, uint32_t x_offset, uint32_t y_offset) {
    uint32_t callsign[5]  = {3876120944, 2493057352, 3836266864, 2495359312, 3876177480}; // BD4SUR, width=29
    uint32_t year_2025[5] = {1662631936, 2493841408, 613010944, 1150296064, 4080910336};  // 2025-, width=23
    uint32_t year_2026[5] = {1662566400, 2493841408, 613007360, 1150361600, 4080844800};  // 2026, width=19
    uint32_t copy_mark[7] = {2013265920, 2214592512, 3019898880, 2751463424, 3019898880, 2214592512, 2013265920}; // (c), width=6

    uint32_t x = x_offset;

    for (uint32_t i = 0; i < 7; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            gfx_draw_point(global_state->gfx, (x + j), (y_offset + i), 255, 255, 255, (copy_mark[i] >> (32 - 1 - j)) & 0x1);
        }
    }
    x += (6 + 4);
    for (uint32_t i = 0; i < 5; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            gfx_draw_point(global_state->gfx, (x + j), (y_offset + 1 + i), 255, 255, 255, (year_2025[i] >> (32 - 1 - j)) & 0x1);
        }
    }
    x += (23 + 1);
    for (uint32_t i = 0; i < 5; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            gfx_draw_point(global_state->gfx, (x + j), (y_offset + 1 + i), 255, 255, 255, (year_2026[i] >> (32 - 1 - j)) & 0x1);
        }
    }
    x += (19 + 4);
    for (uint32_t i = 0; i < 5; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            gfx_draw_point(global_state->gfx, (x + j), (y_offset + 1 + i), 255, 255, 255, (callsign[i] >> (32 - 1 - j)) & 0x1);
        }
    }
}


void ui_app_splash_render_frame(Key_Event *key_event, Global_State *global_state) {
    gfx_fill_white(global_state->gfx);

    gfx_draw_image(global_state->gfx, "/home/bd4sur/ai/Nano/infer/web/nano.png", 235, 56, 85, 170, 1);

#if CONFIG_IDF_TARGET_ESP32S3
    gfx_draw_textline_centered(global_state->gfx, L"Project Nano", global_state->gfx->width / 2, 8, 255, 255, 255, 0);
    gfx_draw_textline_centered(global_state->gfx, L"电子鹦鹉@ESP32S3", global_state->gfx->width, 28, 255, 255, 255, 1);
    ui_draw_copyright_notice(key_event, global_state, 20, 53);
#else
    time_t rawtime;
    struct tm *timeinfo;
    char datetime_string_buffer[33];
    wchar_t datetime_wcs_buffer[33];
    wchar_t nongli_wcs_buffer[33];

    time(&rawtime); // 获取当前时间戳
    timeinfo = localtime(&rawtime); // 转换为本地时间
    strftime(datetime_string_buffer, sizeof(datetime_string_buffer), "%Y-%m-%d %H:%M:%S", timeinfo); // 格式化输出
    _mbstowcs(datetime_wcs_buffer, datetime_string_buffer, 33);

    LunarDate *nongli = lunar_calculate(timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, 8.0);
    _mbstowcs(nongli_wcs_buffer, nongli->full_display, 33);


    ui_draw_header(key_event, global_state, L"Project Nano", 1);

    // gfx_draw_textline_centered(global_state->gfx, L"电 子 鹦 鹉", global_state->gfx->width / 2, 28, 0, 255, 255, 1);
    gfx_draw_textline_centered(global_state->gfx, datetime_wcs_buffer, global_state->gfx->width / 2, 43, 0, 0, 0, 1);
    gfx_draw_textline_centered(global_state->gfx, nongli_wcs_buffer, global_state->gfx->width / 2, 58, 255, 180, 52, 1);
    if (global_state->gfx->width > 128) {
        ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);
        // gfx_draw_textline_centered(global_state->gfx, L"(c) 2025-2026 BD4SUR", global_state->gfx->width / 2, global_state->gfx->height - 3 - FONT_HEIGHT/2, 128, 128, 128, 1);
    }
    else {
        ui_draw_copyright_notice(key_event, global_state, 20, 53);
    }
#endif

    // gfx_draw_line(global_state->gfx, 0, 0, (global_state->gfx->width - 1), 0, 255, 255, 255, 1);
    // gfx_draw_line(global_state->gfx, 0, 15, (global_state->gfx->width - 1), 15, 255, 255, 255, 1);
    // gfx_draw_line(global_state->gfx, 0, 0, 0, (global_state->gfx->height - 1), 255, 255, 255, 1);
    // gfx_draw_line(global_state->gfx, (global_state->gfx->width - 1), 0, (global_state->gfx->width - 1), (global_state->gfx->height - 1), 255, 255, 255, 1);
    // gfx_draw_line(global_state->gfx, 0, (global_state->gfx->height - 1), (global_state->gfx->width - 1), (global_state->gfx->height - 1), 255, 255, 255, 1);

    ui_draw_7seg_time_string(key_event, global_state, 88, 180, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, 0);

#ifdef ASR_ENABLED
    // 检查ASR服务状态，如果ASR服务未启动，则在屏幕左上角画一个闪烁的点，表示ASR服务启动中
    if (global_state->is_asr_server_up < 1) {
        uint8_t v = (uint8_t)((global_state->timer >> 2) & 0x1);
        gfx_draw_line(global_state->gfx, 4, 6, 7, 6, 255, 255, 255, v);
        gfx_draw_line(global_state->gfx, 4, 7, 7, 7, 255, 255, 255, v);
        gfx_draw_line(global_state->gfx, 4, 8, 7, 8, 255, 255, 255, v);
        gfx_draw_line(global_state->gfx, 4, 9, 7, 9, 255, 255, 255, v);
    }
#endif

#ifdef UPS_ENABLED
    // 绘制电池电量
    uint32_t icon_x = global_state->gfx->width - 17;
    uint32_t icon_y = 3;
    gfx_draw_line(global_state->gfx, (icon_x+1),  (icon_y),   (icon_x+14), (icon_y),   255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+14), (icon_y),   (icon_x+14), (icon_y+7), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+1),  (icon_y+7), (icon_x+14), (icon_y+7), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+1),  (icon_y),   (icon_x+1),  (icon_y+7), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x),    (icon_y+2), (icon_x),    (icon_y+5), 255, 255, 255, 1);

    int32_t soc_bar_length = (int32_t)(10.0f * ((float)global_state->ups_soc / 100.0f));
    soc_bar_length = (soc_bar_length > 9) ? 9 : soc_bar_length;
    gfx_draw_line(global_state->gfx, (icon_x+12) - soc_bar_length, (icon_y+2), (icon_x+12), (icon_y+2), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+12) - soc_bar_length, (icon_y+3), (icon_x+12), (icon_y+3), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+12) - soc_bar_length, (icon_y+4), (icon_x+12), (icon_y+4), 255, 255, 255, 1);
    gfx_draw_line(global_state->gfx, (icon_x+12) - soc_bar_length, (icon_y+5), (icon_x+12), (icon_y+5), 255, 255, 255, 1);
#endif

    ui_app_linglong_draw_lite(key_event, global_state, 96, 88,
        timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec,
        119.0, 32.0, 8.0);

    gfx_refresh(global_state->gfx);
}


// ===============================================================================
// Bad Apple
// ===============================================================================

void ui_app_badapple_render_frame(Key_Event *key_event, Global_State *global_state) {
#ifdef BADAPPLE_ENABLED
    uint32_t center_x = global_state->gfx->width / 2;
    uint32_t center_y = global_state->gfx->height / 2;
    if (global_state->timestamp - global_state->ba_begin_timestamp >= 100 * global_state->ba_frame_count) {
        gfx_soft_clear(global_state->gfx);
        for (uint32_t row = 0; row < 64; row++) {
            uint32_t page_0 = bad_apple_10fps_64x64[global_state->ba_frame_count * 128 + row * 2];
            uint32_t page_1 = bad_apple_10fps_64x64[global_state->ba_frame_count * 128 + row * 2 + 1];
            for (uint32_t col = 0; col < 32; col++) {
                gfx_draw_point(global_state->gfx, col + (center_x - 32), row + (center_y - 32), 255, 255, 255, (page_0 >> (32 - 1 - col)) & 0x1);
            }
            for (uint32_t col = 32; col < 64; col++) {
                gfx_draw_point(global_state->gfx, col + (center_x - 32), row + (center_y - 32), 255, 255, 255, (page_1 >> (32 - 1 - (col-32))) & 0x1);
            }
        }
        gfx_refresh(global_state->gfx);
        global_state->ba_frame_count++;

        if (global_state->ba_frame_count > 2193) {
            global_state->ba_frame_count = 0;
            global_state->ba_begin_timestamp = global_state->timestamp;
        }
    }
#endif
}



// ===============================================================================
// Game of Life
// ===============================================================================

#define GOL_WIDTH (128)
#define GOL_HEIGHT (64)
static uint8_t gol_field[2][GOL_WIDTH][GOL_HEIGHT];
static uint8_t gol_field_page;
static uint64_t gol_refresh_timestamp;

void ui_app_gol_init(Key_Event *key_event, Global_State *global_state) {
    gol_field_page = 0;
    gol_refresh_timestamp = global_state->timestamp;
    uint64_t ts = global_state->timestamp;
    for (uint32_t x = 0; x < GOL_WIDTH; x++) {
        for (uint32_t y = 0; y < GOL_HEIGHT; y++) {
            uint8_t s = random_u32(&ts) % 2;
            gol_field[0][x][y] = s;
            gol_field[1][x][y] = s;
        }
    }
}

void ui_app_gol_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 节流：不大于50fps
    if (global_state->timestamp - gol_refresh_timestamp < 20) {
        return;
    }
    gol_refresh_timestamp = global_state->timestamp;
    gfx_soft_clear(global_state->gfx);
    for (uint32_t x = 0; x < GOL_WIDTH; x++) {
        for (uint32_t y = 0; y < GOL_HEIGHT; y++) {
            // 获取某个格子的8邻域
            uint32_t count = 0;
            uint32_t x_a = (x == 0) ? (GOL_WIDTH-1) : (x-1);
            uint32_t x_b = (x == (GOL_WIDTH-1)) ? 0 : (x+1);
            uint32_t y_a = (y == 0) ? (GOL_HEIGHT-1) : (y-1);
            uint32_t y_b = (y == (GOL_HEIGHT-1)) ? 0 : (y+1);
            uint8_t n1 = gol_field[gol_field_page][x_a][y_a]; count += (n1 > 0) ? 1 : 0;
            uint8_t n2 = gol_field[gol_field_page][ x ][y_a]; count += (n2 > 0) ? 1 : 0;
            uint8_t n3 = gol_field[gol_field_page][x_b][y_a]; count += (n3 > 0) ? 1 : 0;
            uint8_t n4 = gol_field[gol_field_page][x_a][ y ]; count += (n4 > 0) ? 1 : 0;
            uint8_t n5 = gol_field[gol_field_page][ x ][ y ]; // self
            uint8_t n6 = gol_field[gol_field_page][x_b][ y ]; count += (n6 > 0) ? 1 : 0;
            uint8_t n7 = gol_field[gol_field_page][x_a][y_b]; count += (n7 > 0) ? 1 : 0;
            uint8_t n8 = gol_field[gol_field_page][ x ][y_b]; count += (n8 > 0) ? 1 : 0;
            uint8_t n9 = gol_field[gol_field_page][x_b][y_b]; count += (n9 > 0) ? 1 : 0;

            uint8_t new_state = 0;
            if (n5 == 0) {
                new_state = (count == 3) ? 1 : 0;
            }
            else {
                new_state = (count == 2 || count == 3) ? 1 : 0;
            }

            gol_field[1-gol_field_page][x][y] = new_state;

            gfx_draw_point(global_state->gfx, x, y, 255, 255, 255, new_state);
        }
    }
    gfx_refresh(global_state->gfx);
    gol_field_page = 1 - gol_field_page;
}



// ===============================================================================
// 玲珑天象仪
// ===============================================================================

static uint64_t linglong_refresh_timestamp = 0;
static uint64_t linglong_first_call_timestamp = 0;
static uint32_t linglong_last_day = 0;
static uint32_t linglong_sunrise_time[2] = {0, 0}; // hour, minute
static uint32_t linglong_sunset_time[2] = {0, 0}; // hour, minute

static int32_t linglong_timemachine_running_state = 2; // 0-停止；1-时光机运行；2-实时
static int32_t linglong_timemachine_speed = 0; // 时光机速度，正数为未来，负数为过去，单位秒
static uint64_t linglong_timemachine_start_timestamp = 0;

void ui_app_linglong_draw_full(Key_Event *key_event, Global_State *global_state) {

    Linglong_Config *llcfg = global_state->linglong_cfg;

    // 节流：不大于50fps
    // if (global_state->timestamp - linglong_refresh_timestamp < 20) {
    //     return;
    // }
    // linglong_refresh_timestamp = global_state->timestamp;

    gfx_soft_clear(global_state->gfx);

    time_t ts = (time_t)global_state->timestamp / 1000;
    if (linglong_timemachine_running_state == 1) {
        linglong_timemachine_start_timestamp += (linglong_timemachine_speed * 1000);
        ts = (time_t)linglong_timemachine_start_timestamp / 1000;
        struct tm *timeinfo = localtime(&ts); // 转换为本地时间

        llcfg->second = timeinfo->tm_sec;
        llcfg->minute = timeinfo->tm_min;
        llcfg->hour = timeinfo->tm_hour;
        llcfg->day = timeinfo->tm_mday;
        llcfg->month = timeinfo->tm_mon + 1;
        llcfg->year = timeinfo->tm_year + 1900;
    }
    else if (linglong_timemachine_running_state == 2) {
        ts = (time_t)global_state->timestamp / 1000;
        struct tm *timeinfo = localtime(&ts);
        llcfg->second = timeinfo->tm_sec;
        llcfg->minute = timeinfo->tm_min;
        llcfg->hour = timeinfo->tm_hour;
        llcfg->day = timeinfo->tm_mday;
        llcfg->month = timeinfo->tm_mon + 1;
        llcfg->year = timeinfo->tm_year + 1900;
    }


    if (llcfg->enable_imu) {
        llcfg->view_alt  = global_state->pitch;
        llcfg->view_azi  = global_state->yaw + 180.0f;
        llcfg->view_roll = global_state->roll;
    }


    render_sky(global_state->gfx->frame_buffer_rgb888, global_state->gfx->width, global_state->gfx->height,
        MIN(global_state->gfx->width, global_state->gfx->height) / 2, global_state->gfx->width / 2, global_state->gfx->height / 2,
        llcfg->view_alt, llcfg->view_azi, llcfg->view_roll, llcfg->view_f,
        // 2026, 3, 24, 18, 10, 0, 8.0, 119.0, 31.0,
        llcfg->year, llcfg->month, llcfg->day, llcfg->hour, llcfg->minute, llcfg->second, llcfg->timezone, llcfg->longitude, llcfg->latitude,
        llcfg->downsampling_factor,
        llcfg->enable_opt_sym,
        llcfg->enable_opt_lut,
        llcfg->enable_opt_bilinear,
        llcfg->projection,
        llcfg->sky_model,
        llcfg->landscape_index,
        llcfg->enable_equatorial_coord,
        llcfg->enable_horizontal_coord,
        llcfg->enable_star_burst,
        llcfg->enable_star_name,
        llcfg->enable_planet,
        llcfg->enable_ecliptic_circle,
        llcfg->enable_att_indicator
    );

    dithering_fast(global_state->gfx->frame_buffer_rgb888, global_state->gfx->width, global_state->gfx->height);

    gfx_draw_textline(global_state->gfx, L"玲珑天象仪 V2603 (c) BD4SUR", 1, 226, 128, 128, 128, 3);

    wchar_t timestr[30];
    swprintf(timestr, 30, L"%04d-%02d-%02d %02d:%02d:%02d", llcfg->year, llcfg->month, llcfg->day, llcfg->hour, llcfg->minute, llcfg->second);
    gfx_draw_textline(global_state->gfx, timestr, 200, global_state->gfx->height - 14, 255, 255, 255, 1);

    gfx_refresh(global_state->gfx);
}

void ui_app_linglong_draw_lite(
    Key_Event *key_event, Global_State *global_state,
    int32_t x, int32_t y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double longitude, double latitude, double timezone
) {
    Linglong_Config *llcfg = global_state->linglong_cfg;

    gfx_draw_rectangle(global_state->gfx, x, y, 128, 64, 255, 255, 255, 1);

    gfx_draw_circle(global_state->gfx, x+64, y+32, 30,         222, 222, 222, 1);
    gfx_draw_circle(global_state->gfx, x+64, y+32, 20,         222, 222, 222, 1);
    gfx_draw_circle(global_state->gfx, x+64, y+32, 10,         222, 222, 222, 1);
    gfx_draw_line(global_state->gfx,   x+32, y+32, x+96, y+32, 222, 222, 222, 1);
    gfx_draw_line(global_state->gfx,   x+64, y+0, x+64, y+64,  222, 222, 222, 1);

    gfx_draw_rectangle(global_state->gfx, x+62-1, y+0,    5+2, 5+1, 255, 255, 255, 1); // N背景
    gfx_draw_rectangle(global_state->gfx, x+63-1, y+59-1, 3+2, 5+1, 255, 255, 255, 1); // S背景
    gfx_draw_rectangle(global_state->gfx, x+32-1, y+30-1, 5+2, 5+2, 255, 255, 255, 1); // W背景
    gfx_draw_rectangle(global_state->gfx, x+93-1, y+30-1, 3+2, 5+2, 255, 255, 255, 1); // E背景

    // 方位文字和周围的边框
    gfx_draw_textline_mini(global_state->gfx, L"N", x+62, y+0,  255, 0, 0, 1); gfx_draw_point(global_state->gfx, x+61, y+2,  255, 255, 255, 1); gfx_draw_point(global_state->gfx, x+67, y+2,  255, 255, 255, 1);  gfx_draw_point(global_state->gfx, x+64, y+5, 255, 255, 255, 1);
    gfx_draw_textline_mini(global_state->gfx, L"S", x+63, y+59, 255, 0, 0, 1); gfx_draw_point(global_state->gfx, x+62, y+62, 255, 255, 255, 1); gfx_draw_point(global_state->gfx, x+66, y+62, 255, 255, 255, 1); gfx_draw_point(global_state->gfx, x+64, y+58, 255, 255, 255, 1);
    gfx_draw_textline_mini(global_state->gfx, L"W", x+32, y+30, 255, 0, 0, 1); gfx_draw_point(global_state->gfx, x+34, y+29, 255, 255, 255, 1); gfx_draw_point(global_state->gfx, x+37, y+32, 255, 255, 255, 1); gfx_draw_point(global_state->gfx, x+34, y+35, 255, 255, 255, 1);
    gfx_draw_textline_mini(global_state->gfx, L"E", x+93, y+30, 255, 0, 0, 1); gfx_draw_point(global_state->gfx, x+92, y+32, 255, 255, 255, 1); gfx_draw_point(global_state->gfx, x+96, y+32, 255, 255, 255, 1); gfx_draw_point(global_state->gfx, x+94, y+29, 255, 255, 255, 1); gfx_draw_point(global_state->gfx, x+94, y+35, 255, 255, 255, 1);

    gfx_draw_line(global_state->gfx, x+0, y+43, x+30, y+43, 222, 222, 222, 1);

    wchar_t timestr[30];
    swprintf(timestr, 30, L"%04d-%02d-%02d\n%02d:%02d:%02d", year, month, day, hour, minute, second);
    gfx_draw_textline_mini(global_state->gfx, timestr, x+0, y+0, 0, 0, 255, 1);

    double altitude_moon = 0.0;
    double azimuth_moon = 0.0;

    where_is_the_moon(year, month, day, hour, minute, second, timezone, longitude, latitude, &azimuth_moon, &altitude_moon);
    double i_deg = moon_phase(year, month, day, hour, minute, second, timezone);
    double moon_k = (1.0 + cos(i_deg / 180.0 * M_PI)) / 2.0;

    wchar_t coordstr_moon[30];
    swprintf(coordstr_moon, 30, L"MOON\nP:%d%%\nA:%.1f\nE:%.1f", (int32_t)(moon_k * 100.0), azimuth_moon, altitude_moon);
    gfx_draw_textline_mini(global_state->gfx, coordstr_moon, x+0, y+18, 0, 0, 0, 1);

    double x_moon = 64 + (90.0 - altitude_moon) * 32.0 / 90.0 * sin(azimuth_moon / 180.0 * M_PI);
    double y_moon = 32 - (90.0 - altitude_moon) * 32.0 / 90.0 * cos(azimuth_moon / 180.0 * M_PI);

    if (x_moon >= 32 && x_moon <= 96 && y_moon >= 0 && y_moon <= 64) {
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon - 1, 255, 0, 255, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon - 0, 255, 0, 255, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon + 1, 255, 0, 255, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon - 1, 255, 0, 255, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon - 0, 255, 0, 255, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon + 1, 255, 0, 255, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon - 1, 255, 0, 255, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon - 0, 255, 0, 255, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon + 1, 255, 0, 255, 1);
    }

    double altitude_sun = 0.0;
    double azimuth_sun = 0.0;

    where_is_the_sun(year, month, day, hour, minute, second, +8.0, longitude, latitude, &azimuth_sun, &altitude_sun);

    wchar_t coordstr_sun[30];
    swprintf(coordstr_sun, 30, L"SUN\nA:%.1f\nE:%.1f", azimuth_sun, altitude_sun);
    gfx_draw_textline_mini(global_state->gfx, coordstr_sun, x+0, y+46, 0, 0, 0, 1);

    double x_sun = 64 + (90.0 - altitude_sun) * 32.0 / 90.0 * sin(azimuth_sun / 180.0 * M_PI);
    double y_sun = 32 - (90.0 - altitude_sun) * 32.0 / 90.0 * cos(azimuth_sun / 180.0 * M_PI);

    if (x_sun >= 32 && x_sun <= 96 && y_sun >= 0 && y_sun <= 64) {
        gfx_draw_circle(global_state->gfx, x+(int)x_sun, y+(int)y_sun, 2, 255, 0, 0, 1);
    }


    // 二分搜索日出日落时间
    if (linglong_first_call_timestamp == 0 || linglong_last_day != day) { // 只在首次调用和当天日期变化时计算
        int32_t sunrise_min = find_sunrise(year, month, day, timezone, longitude, latitude);
        if (sunrise_min != -1) {
            linglong_sunrise_time[0] = sunrise_min / 60;
            linglong_sunrise_time[1] = sunrise_min % 60;
        }
        int32_t sunset_min = find_sunset(year, month, day, timezone, longitude, latitude);
        if (sunset_min != -1) {
            linglong_sunset_time[0] = sunset_min / 60;
            linglong_sunset_time[1] = sunset_min % 60;
        }
    }
    wchar_t risefall_time[60];
    swprintf(risefall_time, 60, L"R:%02d:%02d\nS:%02d:%02d", linglong_sunrise_time[0], linglong_sunrise_time[1], linglong_sunset_time[0], linglong_sunset_time[1]);
    gfx_draw_textline_mini(global_state->gfx, risefall_time, x+98, y+0, 0, 0, 0, 1);

    gfx_draw_textline_mini(global_state->gfx, L"    BD4SUR\n2011-09-29", x+86, y+53, 0, 0, 0, 1);

    gfx_refresh(global_state->gfx);
}


void ui_app_linglong_render_frame(Key_Event *key_event, Global_State *global_state) {
    ui_app_linglong_draw_full(key_event, global_state);
}


void ui_app_linglong_toggle_timemachine(Key_Event *key_event, Global_State *global_state) {
    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = 1;
    }
    else {
        linglong_timemachine_running_state = 0;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

void ui_app_linglong_set_timemachine_speed(Key_Event *key_event, Global_State *global_state, int32_t speed) {
    linglong_timemachine_speed = speed;
    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = 1;
    }
    else {
        linglong_timemachine_running_state = 0;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

void ui_app_linglong_set_realtime(Key_Event *key_event, Global_State *global_state) {
    Linglong_Config *llcfg = global_state->linglong_cfg;
    time_t ts = (time_t)global_state->timestamp / 1000;
    struct tm *timeinfo = localtime(&ts); // 转换为本地时间
    llcfg->second = timeinfo->tm_sec;
    llcfg->minute = timeinfo->tm_min;
    llcfg->hour = timeinfo->tm_hour;
    llcfg->day = timeinfo->tm_mday;
    llcfg->month = timeinfo->tm_mon + 1;
    llcfg->year = timeinfo->tm_year + 1900;

    if (linglong_timemachine_running_state == 0) {
        linglong_timemachine_running_state = 2;
    }
    else {
        linglong_timemachine_running_state = 0;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}
