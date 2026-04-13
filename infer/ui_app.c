#include <stdio.h>
#include <time.h>

#include "graphics.h"
#include "keyboard_hal.h"
#include "ui.h"

#include "platform.h"

#include "infer.h"

#ifdef IMU_ENABLED
    #include "imu.h"
#endif

#ifdef UPS_ENABLED
    #include "ups.h"
#endif

#ifdef ASR_ENABLED
    #include "asr.h"
#endif

#ifdef TTS_ENABLED
    #include "tts.h"
#endif

#ifdef BADAPPLE_ENABLED
    #include "badapple.h"
#endif

#include "flip.h"

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
// 全局GUI+gfx初始化
// ===============================================================================

void ui_init(Key_Event *key_event, Global_State *global_state) {

    global_state->w_textarea_main = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));
    global_state->w_textarea_asr = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));
    global_state->w_textarea_prefill = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));

    global_state->w_input_main = (Widget_Input_State*)calloc(1, sizeof(Widget_Input_State));

    global_state->w_menu_main = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    global_state->w_menu_model = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    global_state->w_menu_setting = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    global_state->w_menu_asr_setting = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    global_state->w_menu_tts_setting = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    global_state->w_menu_linglong_setting = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));


    ///////////////////////////////////////
    // gfx初始化

    global_state->gfx = (Nano_GFX*)calloc(1, sizeof(Nano_GFX));
    gfx_init(global_state->gfx, SCREEN_WIDTH, SCREEN_HEIGHT, GFX_COLOR_MODE_RGB888);

    global_state->STATE = STATE_SPLASH_SCREEN;
    global_state->PREV_STATE = STATE_DEFAULT;
    global_state->is_ctrl_enabled = 0;
    global_state->llm_status = LLM_STOPPED_NORMALLY;
    global_state->llm_model_name = NULL;
    global_state->llm_is_thinking_model = 0;
    global_state->llm_model_path = NULL;
    global_state->llm_lora_path = NULL;
    global_state->llm_repetition_penalty = 1.05f;
    global_state->llm_temperature = 1.0f;
    global_state->llm_top_p = 0.5f;
    global_state->llm_top_k = 0;
    global_state->llm_max_seq_len = 512;
    global_state->is_thinking_enabled = 1;
    global_state->llm_output_of_last_session = (wchar_t*)calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
    global_state->tps_of_last_session = 0.0f;
    global_state->token_num_of_last_session = 0;
#ifdef ASR_ENABLED
    global_state->asr_output_buffer = (wchar_t*)calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
    wcscpy(global_state->asr_output_buffer, L"请说话...");
#else
    global_state->asr_output_buffer = NULL;
#endif
    global_state->is_auto_submit_after_asr = 1; // ASR结束后立刻提交识别内容到LLM
    global_state->is_asr_server_up = 0;
    global_state->is_recording = 0;
    global_state->asr_start_timestamp = 0;
    global_state->pitch = 0.0f;
    global_state->roll = 0.0f;
    global_state->yaw = 0.0f;
    global_state->imu_temperature = 0.0f;
    global_state->is_full_refresh = 1;
    global_state->llm_refresh_max_fps = 10;
    global_state->llm_refresh_timestamp = 0;
    global_state->ba_frame_count = 0;
    global_state->ba_begin_timestamp = 0;
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

            if (new_state) {
                gfx_draw_point(global_state->gfx, x, y, 0, 255, 255, 1);
            }
        }
    }
    gfx_refresh(global_state->gfx);
    gol_field_page = 1 - gol_field_page;
}


// ===============================================================================
// FLIP流体模拟
// ===============================================================================

void ui_app_flip_init(Key_Event *key_event, Global_State *global_state) {
    flip_init(1.33, 1);
}

void ui_app_flip_render_frame(Key_Event *key_event, Global_State *global_state) {
    int show_particles = 1;
    int show_grid      = 0;

    gfx_soft_clear(global_state->gfx);

    // 获取重力方向
    float gravity_x = 0.0f;
    float gravity_y = -9.8f;
#ifdef IMU_ENABLED

    int ret = -1;
    int32_t imu_count = 3000;
    do {
        // 以下代码适配树莓派盒子（NanoPod）：IMU的PCB平面与树莓派PCB平行，IMU顶部指向树莓派TypeC口方向，IMU串口指向树莓派的PCIe方向
        float q0 = 0.0f;
        float q1 = 0.0f;
        float q2 = 0.0f;
        float q3 = 0.0f;
        ret = imu_read_quaternion(&q0, &q1, &q2, &q3);
        quaternion_to_euler(q0, q1, q2, q3, &(global_state->roll), &(global_state->pitch), &(global_state->yaw));
        global_state->pitch -= 90.0f;
        global_state->yaw = fmod(-global_state->yaw, 360.0);
        if (global_state->yaw < 0) global_state->yaw += 360.0;

        // 以下代码适配 NANO_POD_PLUS_CUBIE_A7Z （2026-03-02制作的单板原型）
        // ret = imu_read_angle(&(global_state->roll), &(global_state->pitch), &(global_state->yaw));
        // global_state->pitch = -global_state->pitch;
        // global_state->yaw = -global_state->yaw;

        imu_count--;
        if (imu_count <= 0) {
            printf("IMU读取超时，重置\n");
            imu_reset();
            break;
        }
    } while(ret != 0);
    printf("俯仰=%-10.2f    滚转=%-10.2f    航向=%-10.2f\n", global_state->pitch, global_state->roll, global_state->yaw);

    // gravity_x = -9.8f * sinf(global_state->roll / 180.0f * M_PI) * cosf(global_state->pitch / 180.0f * M_PI);
    // gravity_y = -9.8f * cosf(global_state->roll / 180.0f * M_PI) * cosf(global_state->pitch / 180.0f * M_PI);

    gravity_x = -9.8f * sinf(global_state->roll / 180.0f * M_PI);
    gravity_y = -9.8f * cosf(global_state->roll / 180.0f * M_PI);

    printf("gx=%-10.2f    gy=%-10.2f\n", gravity_x, gravity_y);

#endif

    render_flip(global_state->gfx, 0, 0, 320, 240,
                1.33, 1,            /* pool_width, pool_height */
                gravity_x, gravity_y,    /* gravity_x, gravity_y */
                1.5f / 60.0f,    /* dt */
                0.9f,            /* flip_ratio */
                50, 2,           /* num_pressure_iters, num_particle_iters */
                1.9f,            /* over_relaxation */
                1, 1,            /* compensate_drift, separate_particles */
                show_particles, show_grid);
    gfx_refresh(global_state->gfx);
}


// ===============================================================================
// 玲珑天象仪
// ===============================================================================

static uint64_t linglong_first_call_timestamp = 0;
static uint32_t linglong_last_day = 0;
static uint32_t linglong_sunrise_time[2] = {0, 0}; // hour, minute
static uint32_t linglong_sunset_time[2] = {0, 0}; // hour, minute

static int32_t linglong_timemachine_running_state = 2; // 0-停止；1-时光机运行；2-实时
static int32_t linglong_timemachine_speed = 0; // 时光机速度，正数为未来，负数为过去，单位秒
static uint64_t linglong_timemachine_start_timestamp = 0;


void refresh_text_of_linglong_setting_menu(Key_Event *key_event, Global_State *global_state) {
    wchar_t buf[MAX_MENU_ITEM_LEN];

    wcscpy(global_state->w_menu_linglong_setting->title, L"玲珑仪设置");
    wcscpy(global_state->w_menu_linglong_setting->items[0], L"时间和位置");


    swprintf(buf, MAX_MENU_ITEM_LEN, L"陀螺仪：%ls",
        (global_state->linglong_cfg->enable_imu == 0) ? L"关" : L"开");
    wcscpy(global_state->w_menu_linglong_setting->items[1], buf);


    swprintf(buf, MAX_MENU_ITEM_LEN, L"投影算法：%ls",
        (global_state->linglong_cfg->projection == 0) ? L"鱼眼投影" : L"透视投影");
    wcscpy(global_state->w_menu_linglong_setting->items[2], buf);


    swprintf(buf, MAX_MENU_ITEM_LEN, L"姿态指示：%ls",
        (global_state->linglong_cfg->enable_att_indicator == 0) ? L"关" : L"开");
    wcscpy(global_state->w_menu_linglong_setting->items[3], buf);


    switch (global_state->linglong_cfg->landscape_index) {
        case 0: swprintf(buf, MAX_MENU_ITEM_LEN, L"地景：%ls", L"关闭"); break;
        case 1: swprintf(buf, MAX_MENU_ITEM_LEN, L"地景：%ls", L"卫星照片"); break;
        case 2: swprintf(buf, MAX_MENU_ITEM_LEN, L"地景：%ls", L"地面风景"); break;
        default: swprintf(buf, MAX_MENU_ITEM_LEN, L"地景：%ls", L"关闭"); break;
    }
    wcscpy(global_state->w_menu_linglong_setting->items[4], buf);


    switch (global_state->linglong_cfg->sky_model) {
        case 0: swprintf(buf, MAX_MENU_ITEM_LEN, L"大气散射模型：%ls", L"关闭"); break;
        case 1: swprintf(buf, MAX_MENU_ITEM_LEN, L"大气散射模型：%ls", L"简化模型"); break;
        case 2: swprintf(buf, MAX_MENU_ITEM_LEN, L"大气散射模型：%ls", L"一次散射"); break;
        case 3: swprintf(buf, MAX_MENU_ITEM_LEN, L"大气散射模型：%ls", L"二次散射"); break;
        default: swprintf(buf, MAX_MENU_ITEM_LEN, L"大气散射模型：%ls", L"关闭"); break;
    }
    wcscpy(global_state->w_menu_linglong_setting->items[5], buf);


    swprintf(buf, MAX_MENU_ITEM_LEN, L"赤道坐标：%ls",
        (global_state->linglong_cfg->enable_equatorial_coord == 0) ? L"关闭" : L"开启");
    wcscpy(global_state->w_menu_linglong_setting->items[6], buf);


    switch (global_state->linglong_cfg->enable_horizontal_coord) {
        case 0: swprintf(buf, MAX_MENU_ITEM_LEN, L"地平坐标：%ls", L"关闭"); break;
        case 1: swprintf(buf, MAX_MENU_ITEM_LEN, L"地平坐标：%ls", L"方位角"); break;
        case 2: swprintf(buf, MAX_MENU_ITEM_LEN, L"地平坐标：%ls", L"坐标圈"); break;
        default: swprintf(buf, MAX_MENU_ITEM_LEN, L"地平坐标：%ls", L"关闭"); break;
    }
    wcscpy(global_state->w_menu_linglong_setting->items[7], buf);


    switch (global_state->linglong_cfg->enable_star_name) {
        case 0: swprintf(buf, MAX_MENU_ITEM_LEN, L"天体名称：%ls", L"关闭"); break;
        case 1: swprintf(buf, MAX_MENU_ITEM_LEN, L"天体名称：%ls", L"除行星外的全部天体"); break;
        case 2: swprintf(buf, MAX_MENU_ITEM_LEN, L"天体名称：%ls", L"仅行星"); break;
        case 3: swprintf(buf, MAX_MENU_ITEM_LEN, L"天体名称：%ls", L"全部显示"); break;
        default: swprintf(buf, MAX_MENU_ITEM_LEN, L"天体名称：%ls", L"关闭"); break;
    }
    wcscpy(global_state->w_menu_linglong_setting->items[8], buf);


    swprintf(buf, MAX_MENU_ITEM_LEN, L"黄道：%ls",
        (global_state->linglong_cfg->enable_ecliptic_circle == 0) ? L"关闭" : L"开启");
    wcscpy(global_state->w_menu_linglong_setting->items[9], buf);


    swprintf(buf, MAX_MENU_ITEM_LEN, L"星芒：%ls",
        (global_state->linglong_cfg->enable_star_burst == 0) ? L"关闭" : L"开启");
    wcscpy(global_state->w_menu_linglong_setting->items[10], buf);


    wcscpy(global_state->w_menu_linglong_setting->items[11], L"校准陀螺仪");


    global_state->w_menu_linglong_setting->item_num = 12;
}

void init_linglong_setting_menu(Key_Event *key_event, Global_State *global_state) {
    refresh_text_of_linglong_setting_menu(key_event, global_state);
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_linglong_setting);
}


int32_t linglong_setting_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    int32_t item_index = ms->current_item_index;
    // 0.时间和位置
    if (item_index == 0) {
        return STATE_LINGLONG_TIMELOC;
    }
    // 1.陀螺仪
    else if (item_index == 1) {
        if (gs->linglong_cfg->enable_imu == 0) {
            gs->linglong_cfg->enable_imu = 1;
        }
        else {
            gs->linglong_cfg->enable_imu = 0;
        }
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 2.投影算法
    else if (item_index == 2) {
        if (gs->linglong_cfg->projection == 0) {
            gs->linglong_cfg->projection = 1;
        }
        else {
            gs->linglong_cfg->projection = 0;
        }
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 3.姿态指示
    else if (item_index == 3) {
        gs->linglong_cfg->enable_att_indicator++;
        gs->linglong_cfg->enable_att_indicator = gs->linglong_cfg->enable_att_indicator % 2;
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 4.地景
    else if (item_index == 4) {
        gs->linglong_cfg->landscape_index++;
        gs->linglong_cfg->landscape_index = gs->linglong_cfg->landscape_index % 3;
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 5.大气散射模型
    else if (item_index == 5) {
        gs->linglong_cfg->sky_model++;
        gs->linglong_cfg->sky_model = gs->linglong_cfg->sky_model % 4;
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 6.赤道坐标
    else if (item_index == 6) {
        gs->linglong_cfg->enable_equatorial_coord ++;
        gs->linglong_cfg->enable_equatorial_coord = gs->linglong_cfg->enable_equatorial_coord % 2;
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 7.地平坐标
    else if (item_index == 7) {
        gs->linglong_cfg->enable_horizontal_coord++;
        gs->linglong_cfg->enable_horizontal_coord = gs->linglong_cfg->enable_horizontal_coord % 3;
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 8.天体名称
    else if (item_index == 8) {
        gs->linglong_cfg->enable_star_name++;
        gs->linglong_cfg->enable_star_name = gs->linglong_cfg->enable_star_name % 4;
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 9.黄道
    else if (item_index == 9) {
        gs->linglong_cfg->enable_ecliptic_circle++;
        gs->linglong_cfg->enable_ecliptic_circle = gs->linglong_cfg->enable_ecliptic_circle % 2;
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 10.星芒
    else if (item_index == 10) {
        gs->linglong_cfg->enable_star_burst++;
        gs->linglong_cfg->enable_star_burst = gs->linglong_cfg->enable_star_burst % 2;
        refresh_text_of_linglong_setting_menu(ke, gs);
        ui_widget_menu_refresh(ke, gs, ms);
        return STATE_LINGLONG_SETTING;
    }
    // 11.校准陀螺仪
    else if (item_index == 11) {
#ifdef IMU_ENABLED
        ui_widget_textarea_set(ke, gs, gs->w_textarea_main, L" \n \n    正在校准IMU...", 0, 0);
        ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);
        imu_calib();
        sleep_in_ms(500);
        ui_widget_textarea_set(ke, gs, gs->w_textarea_main, L" \n \n    校准完成", 0, 0);
        ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);
#endif
        return STATE_LINGLONG;
    }
    // TODO
    else {
        return STATE_LINGLONG;
    }
}


void ui_app_linglong_draw_full(Key_Event *key_event, Global_State *global_state) {

    Linglong_Config *llcfg = global_state->linglong_cfg;

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

}

void ui_app_linglong_draw_lite(
    Key_Event *key_event, Global_State *global_state,
    int32_t x, int32_t y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double longitude, double latitude, double timezone
) {
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

}


void ui_app_linglong_render_frame(Key_Event *key_event, Global_State *global_state) {
    ui_app_linglong_draw_full(key_event, global_state);

    if (global_state->is_ctrl_enabled) {
        gfx_draw_rectangle(global_state->gfx, 2, 2, 4, 4, 255, 255, 255, 1);
        gfx_draw_rectangle(global_state->gfx, 10, 10, global_state->gfx->width - 20, global_state->gfx->height - 20, 128, 128, 128, 3);
        gfx_draw_textline_centered(global_state->gfx, L"=== 功能选择 ===", global_state->gfx->width/2, 12+6, 0, 255, 255, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+1 投影算法",          12, (12+1)*2, 0x0, 0x0, 0x0, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+2 赤道坐标圈",        12, (12+1)*3, 0x0, 0x0, 0x0, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+3 地平坐标圈和方位角", 12, (12+1)*4, 0x0, 0x0, 0x0, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+4 黄道",              12, (12+1)*5, 0x0, 0x0, 0x0, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+5 天体名称标签",       12, (12+1)*6, 0x0, 0x0, 0x0, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+6 姿态指示",           12, (12+1)*7, 0x0, 0x0, 0x0, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+7 大气散射模型",       12, (12+1)*8, 0x0, 0x0, 0x0, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+8 地景",              12, (12+1)*9, 0x0, 0x0, 0x0, 1);
        gfx_draw_textline(global_state->gfx, L"Ctrl+9 校准IMU",           12, (12+1)*10, 0x0, 0x0, 0x0, 1);
    }

    gfx_refresh(global_state->gfx);
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

void ui_app_linglong_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 获取机器姿态（欧拉角）
#ifdef IMU_ENABLED
    if (global_state->linglong_cfg->enable_imu) {
        int ret = -1;
        int32_t imu_count = 3000;
        do {
            // 以下代码适配树莓派盒子（NanoPod）：IMU的PCB平面与树莓派PCB平行，IMU顶部指向树莓派TypeC口方向，IMU串口指向树莓派的PCIe方向
            float q0 = 0.0f;
            float q1 = 0.0f;
            float q2 = 0.0f;
            float q3 = 0.0f;
            ret = imu_read_quaternion(&q0, &q1, &q2, &q3);
            quaternion_to_euler(q0, q1, q2, q3, &(global_state->roll), &(global_state->pitch), &(global_state->yaw));
            global_state->pitch -= 90.0f;
            global_state->yaw = fmod(-global_state->yaw, 360.0);
            if (global_state->yaw < 0) global_state->yaw += 360.0;

            // 以下代码适配 NANO_POD_PLUS_CUBIE_A7Z （2026-03-02制作的单板原型）
            // ret = imu_read_angle(&(global_state->roll), &(global_state->pitch), &(global_state->yaw));
            // global_state->pitch = -global_state->pitch;
            // global_state->yaw = -global_state->yaw;

            imu_count--;
            if (imu_count <= 0) {
                printf("IMU读取超时，重置\n");
                imu_reset();
                break;
            }
        } while(ret != 0);
        printf("俯仰=%-10.2f    滚转=%-10.2f    航向=%-10.2f\n", global_state->pitch, global_state->roll, global_state->yaw);
    }
#endif

    // 按1键向左偏航（yaw--），或者Ctrl时切换投影算法
    if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_1) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_azi -= 5.0f;
            if (global_state->linglong_cfg->view_azi <= 0.0f) {
                global_state->linglong_cfg->view_azi = 360.0f;
            }
        }
        else {
            global_state->is_ctrl_enabled = 0;
            if (global_state->linglong_cfg->projection == 0) {
                global_state->linglong_cfg->projection = 1;
            }
            else {
                global_state->linglong_cfg->projection = 0;
            }
        }
    }
    // 按2键推杆低头（pitch--），或者Ctrl时切换赤道坐标圈
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_2) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_alt -= 5.0f;
            if (global_state->linglong_cfg->view_alt <= -90.0f) {
                global_state->linglong_cfg->view_alt = -90.0f;
            }
        }
        else {
            global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_equatorial_coord ++;
            global_state->linglong_cfg->enable_equatorial_coord = global_state->linglong_cfg->enable_equatorial_coord % 2;
        }
    }
    // 按3键向右偏航（yaw++），或者Ctrl时切换地平坐标
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_3) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_azi += 5.0f;
            if (global_state->linglong_cfg->view_azi >= 360.0f) {
                global_state->linglong_cfg->view_azi = 0.0f;
            }
        }
        else {
            global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_horizontal_coord++;
            global_state->linglong_cfg->enable_horizontal_coord = global_state->linglong_cfg->enable_horizontal_coord % 3;
        }
    }
    // 按4键向左坡度（roll--），或者Ctrl时切换黄道
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_4) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_roll -= 5.0f;
            if (global_state->linglong_cfg->view_roll <= -90.0f) {
                global_state->linglong_cfg->view_roll = -90.0f;
            }
        }
        else {
            global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_ecliptic_circle++;
            global_state->linglong_cfg->enable_ecliptic_circle = global_state->linglong_cfg->enable_ecliptic_circle % 2;
        }
    }
    // 按5键归中，或切换陀螺仪状态，或者Ctrl时切换天体名称
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_5) {
        if (global_state->is_ctrl_enabled == 0) {
            // 如果IMU已关闭，则开启
            if (global_state->linglong_cfg->enable_imu == 0) {
                global_state->linglong_cfg->enable_imu = 1;
            }
            // 如果IMU已开启，则关闭并归中
            else {
                global_state->linglong_cfg->enable_imu = 0;
                global_state->linglong_cfg->view_alt = 90.0f;
                global_state->linglong_cfg->view_azi = 180.0f;
                global_state->linglong_cfg->view_f = 1.0f;
            }
        }
        else {
            global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_star_name++;
            global_state->linglong_cfg->enable_star_name = global_state->linglong_cfg->enable_star_name % 4;
        }
    }
    // 按6键向右坡度（roll++），或者Ctrl时切换姿态指示
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_6) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_roll += 5.0f;
            if (global_state->linglong_cfg->view_roll >= 90.0f) {
                global_state->linglong_cfg->view_roll = 90.0f;
            }
        }
        else {
            global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_att_indicator++;
            global_state->linglong_cfg->enable_att_indicator = global_state->linglong_cfg->enable_att_indicator % 2;
        }
    }
    // 按7键拉远，或者Ctrl时切换大气散射模型
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_7) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->view_f -= 0.1f;
            if (global_state->linglong_cfg->view_f <= 0.1f) global_state->linglong_cfg->view_f = 0.1f;
        }
        else {
            global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->sky_model++;
            global_state->linglong_cfg->sky_model = global_state->linglong_cfg->sky_model % 4;
        }
    }
    // 按8键拉杆抬头（pitch++），或者Ctrl时切换地景
    if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_8) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_alt += 5.0f;
            if (global_state->linglong_cfg->view_alt >= 90.0f) {
                global_state->linglong_cfg->view_alt = 90.0f;
            }
        }
        else {
            global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->landscape_index++;
            global_state->linglong_cfg->landscape_index = global_state->linglong_cfg->landscape_index % 3;
        }
    }
    // 按9键推近，或者Ctrl时校准陀螺仪
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_9) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->view_f += 0.1f;
            if (global_state->linglong_cfg->view_f >= 5.0f) global_state->linglong_cfg->view_f = 5.0f;
        }
        else {
            global_state->is_ctrl_enabled = 0;
#ifdef IMU_ENABLED
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n    正在校准IMU...", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
            imu_calib();
            sleep_in_ms(500);
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n    校准完成", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
#endif
        }
    }
    // 按A键返回主菜单
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
        global_state->STATE = STATE_MAIN_MENU;
    }
    // 按B键打开玲珑仪设置菜单
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_B) {
        init_linglong_setting_menu(key_event, global_state);
        global_state->STATE = STATE_LINGLONG_SETTING;
    }
    // 按C键切换Ctrl
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_C) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->is_ctrl_enabled = 1;
        }
        else {
            global_state->is_ctrl_enabled = 0;
        }
    }
    // 按*键时光机向前（过去）（反复按运行/暂停）
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_STAR) {
        ui_app_linglong_set_timemachine_speed(key_event, global_state, -120);
    }
    // 短按0键回到实时（反复按运行/暂停）
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_0) {
        ui_app_linglong_set_realtime(key_event, global_state);
    }
    // 按#键时光机向后（未来）（反复按运行/暂停）
    else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_HASH) {
        ui_app_linglong_set_timemachine_speed(key_event, global_state, 120);
    }
}

