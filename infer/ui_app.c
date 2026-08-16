#include <stdio.h>
#include <time.h>

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    #include "esp_heap_caps.h"
    #include "display_hal.h"
    #include "model/nano_psycho_230k_q80.h"
#endif

#include "graphics.h"
#include "input_device.h"
#include "ui.h"
#include "ui_icon.h"
#include "ui_softkbd.h"
#include "ui_pinyin_ime.h"

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

#include "ui_genetic.h"

#include "ui_tsp.h"

#include "ui_cloud.h"
#include "ephemeris.h"
#include "celestial.h"
#include "nongli.h"

#include "ui_color.h"
#include "ui_app.h"

#include "ui_animac.h"

#include "ui_spectrogram.h"

#include "ui_pedometer.h"

#include "ui_goldminer.h"

#include "ui_particlelife.h"

#include "ui_ripple.h"

#include "ui_water.h"

#include "ui_tetris.h"

#include "ui_ebook.h"

#include "ui_ofdm.h"

#include "ui_musicbox.h"

#include "ui_dict.h"

#include "ui_calendar.h"

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    #define WALLPAPER_PATH ("/wp.png")
#else
    #define WALLPAPER_PATH ("/home/bd4sur/wp.jpg")
#endif

// 全局变量（TODO 临时，后续要全部移到全局状态上下文中）

static uint64_t last_splash_timestamp = 0;

static int32_t s_album_count = 1;
static int32_t s_album_index = 0;
static char **s_album_path_list = NULL;
static int32_t s_album_is_autoplay = 0;
static uint64_t s_album_refresh_timestamp = 0;

// 指向图像缓冲区的指针
static uint8_t *s_image_file_buffer = NULL;
static size_t s_image_file_size = 0;
static char s_image_filename_buffer[128]; // 缓存图像文件名，用于确定是否要重新读取、重新解码
// 壁纸图像解码后的 RGB888 像素缓冲区（避免每次渲染都重新解码）
static uint8_t *s_image_rgb888_buffer = NULL;
static uint32_t s_image_width = 0;
static uint32_t s_image_height = 0;
static uint8_t s_image_decode_ready = 0;



static uint32_t s_animac_console_text_len = 0;
static uint32_t s_animac_prev_ui_font = 0; // 进入 STATE_ANIMAC_* 之前的 ui_font，退出时恢复





// 带容量上限的安全追加（剩余空间不足时直接截断，防堆溢出）
static void ui_app_wcscat_bounded(wchar_t *dst, const wchar_t *src, uint32_t cap) {
    size_t cur = wcslen(dst);
    if (cur < cap - 1) {
        wcsncat(dst, src, cap - 1 - cur);
    }
}

// ===============================================================================
// UI框架：获取按键事件
// ===============================================================================

// 滑动手势识别（Core1轮询侧）状态
static int32_t s_swipe_active = 0;   // 正在跟踪一次触摸序列
static int32_t s_swipe_start_y = 0;  // 序列起点y
static int32_t s_swipe_min_y = 0;    // 序列中的最小y（最高点）
static int32_t s_swipe_max_y = 0;    // 序列中的最大y（最低点）

// 滑动手势（均为松手确认，跨越一半以上屏幕高度即>120px）：
//   键盘隐藏时，从下到上滑动 = 呼出键盘；键盘可见时，从上到下滑动 = 关闭键盘。
// 有效手势方向上位移超过 UI_SWIPE_MASK_PX 即预备吞掉本次触摸序列的按键事件
// （key_mask软复位），防止滑动穿越网格键/软键盘键时产生杂散输入。
#define UI_SWIPE_CONFIRM_PX (SCREEN_HEIGHT / 2)
#define UI_SWIPE_MASK_PX    (20)


// 当前状态是否寄宿文本输入控件（w_input_main）：上滑/下滑手势（呼出/收起软键盘）仅在这些状态识别，
// 避免误吞水波、OFDM接收等其他状态的触屏拖动；新增输入状态只改这一处
static int32_t ui_app_state_hosts_input_widget(int32_t state) {
    return (state == STATE_LLM_INPUT || state == STATE_ANIMAC_CONSOLE ||
            state == STATE_ANIMAC_RUNNING || state == STATE_OFDM_TX);
}

void get_key_event(Key_Event *key_event, Global_State *global_state) {
    // 触屏电平锁存（本函数在 Core1 每 1-2ms 轮询一次）：为九键按键提示遮罩提供可靠的
    // 触发时间戳，避免渲染循环（Core0）帧率不均的低频采样漏掉短按（见 ui.c 遮罩实现）
    {
        int32_t touch_pressed = 0;
        touch_read(NULL, NULL, &touch_pressed);
        if (touch_pressed) {
            global_state->last_touch_timestamp = global_state->timestamp;
        }
    }

    // 滑动手势识别（仅寄宿文本输入控件的状态）：上滑呼出键盘（隐藏态）/下滑关闭键盘（可见态）
    if (ui_app_state_hosts_input_widget(global_state->STATE)) {
        int32_t touch_x = 0, touch_y = 0, touch_pressed = 0;
        touch_read(&touch_x, &touch_y, &touch_pressed);
        if (touch_pressed) {
            if (!s_swipe_active) {
                s_swipe_active = 1;
                s_swipe_start_y = touch_y;
                s_swipe_min_y = touch_y;
                s_swipe_max_y = touch_y;
            }
            else {
                if (touch_y < s_swipe_min_y) s_swipe_min_y = touch_y;
                if (touch_y > s_swipe_max_y) s_swipe_max_y = touch_y;
                // 仅在与当前键盘显隐对应的手势方向上，位移超过预备阈值即吞掉本次序列的按键边沿事件
                if ((!ui_softkbd_is_visible() && s_swipe_start_y - s_swipe_min_y > UI_SWIPE_MASK_PX) ||
                    ( ui_softkbd_is_visible() && s_swipe_max_y - s_swipe_start_y > UI_SWIPE_MASK_PX)) {
                    key_event->key_mask = 1;
                }
            }
        }
        else if (s_swipe_active) {
            // 松手确认：隐藏时上滑跨越半屏 = 呼出；可见时下滑跨越半屏 = 关闭
            if ((!ui_softkbd_is_visible() && s_swipe_start_y - s_swipe_min_y > UI_SWIPE_CONFIRM_PX) ||
                ( ui_softkbd_is_visible() && s_swipe_max_y - s_swipe_start_y > UI_SWIPE_CONFIRM_PX)) {
                key_event->key_mask = 1;
                ui_softkbd_request_toggle();
            }
            s_swipe_active = 0;
        }
    }
    else {
        s_swipe_active = 0;
    }

    uint8_t key = input_device_read_key();
    uint8_t key_is_softkbd = 0;

    // 触屏软键盘：可见时，键盘区域内的触摸由软键盘接管——吞掉触屏4x4网格键映射，
    // 改为注入软键盘键码（无边沿时为NANO_KEY_IDLE，仍走下方的边沿检测机制）。
    // 同时禁用键盘区域以外的十六键网格（仅保留右上角Esc键），防止误触。
    if (ui_softkbd_is_visible()) {
        uint8_t softkbd_key = ui_softkbd_poll();
        if (ui_softkbd_touch_claimed()) {
            key = softkbd_key;
            key_is_softkbd = (softkbd_key != NANO_KEY_IDLE);
        }
        else if (key != NANO_KEY_esc) {
            key = NANO_KEY_IDLE; // 软键盘激活时禁用十六键网格（Esc除外）
        }
    }

    // 边沿
    if (key_event->key_mask != 1 && (key != key_event->prev_key)) {
        // 按下瞬间（上升沿）
        if (key != NANO_KEY_IDLE) {
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
        if (key != NANO_KEY_IDLE) {
            key_event->key_code = key;
            key_event->key_edge = 0;
            // key_event->key_timer++;
            // 若重复动作标记key_repeat在一次长按后点亮，则继续按住可以反复触发短按
            if (key_event->key_repeat == 1) {
                key_event->key_edge = -2;
                key_event->key_mask = 1; // 软复位置1，即强制恢复为无按键状态，以便下一次轮询检测到下降沿（尽管物理上有键按下），触发长按事件
                key = NANO_KEY_IDLE; // 便于后面设置prev_key为KEY_IDLE（无键按下）
                key_event->key_repeat = 1;
            }
            // 如果没有点亮动作标记key_repeat，则达到长按阈值后触发长按事件
            else if ((global_state->timestamp - key_event->key_timer) >= LONG_PRESS_THRESHOLD) {
                key_event->key_edge = -2;
                key_event->key_mask = 1; // 软复位置1，即强制恢复为无按键状态，以便下一次轮询检测到下降沿（尽管物理上有键按下），触发长按事件
                key = NANO_KEY_IDLE; // 便于后面设置prev_key为KEY_IDLE（无键按下）
            }
        }
        // 松开
        else {
            key_event->key_code = NANO_KEY_IDLE;
            key_event->key_edge = 0;
            key_event->key_timer = global_state->timestamp;
            key_event->key_mask = 0;
            key_event->key_repeat = 0;
        }
    }
    if (key != NANO_KEY_IDLE) {
        key_event->is_softkbd = key_is_softkbd; // 记录按键来源，下降沿事件沿用它
    }
    key_event->prev_key = key;
}


// ===============================================================================
// UI框架：全局GUI+gfx初始化
// ===============================================================================

// 注：软键盘显隐切换已下沉为文本输入控件固有功能（ui.c 的 ui_widget_input_toggle_softkbd，
// 供 Ctrl+0 组合键与上滑/下滑手势在任何输入状态使用）。

void ui_init(Key_Event *key_event, Global_State *global_state) {

    global_state->w_textarea_main = (Widget_Textarea_State*)platform_calloc(1, sizeof(Widget_Textarea_State));

    global_state->w_input_main = (Widget_Input_State*)platform_calloc(1, sizeof(Widget_Input_State));

    global_state->w_menu_main = (Widget_Menu_State*)platform_calloc(1, sizeof(Widget_Menu_State));


    global_state->STATE = STATE_SPLASH_SCREEN;
    global_state->PREV_STATE = STATE_DEFAULT;

    global_state->ui_color_style = UI_COLOR_DARK;

    // 全局默认背光（与 display_hal_init 的 setBrightness 一致；设置菜单按此值显示/调节）
    global_state->brightness = 204;

    // 全局主音量（影响按键音、寻呼机发射音量、音乐盒初始音量）
    global_state->volume = 16;
    // 自动关机（默认关）
    global_state->auto_shutdown_minutes = 0;
    global_state->auto_shutdown_deadline = 0;
    global_state->ime_hint_timeout_s = 3; // 按键提示遮罩默认显示3秒

    global_state->timestamp_last = 0;
    global_state->last_touch_timestamp = 0;

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
    global_state->llm_output_of_last_session = (wchar_t*)platform_calloc(UI_STR_BUF_MAX_LENGTH * 2, sizeof(wchar_t));
    global_state->tps_of_last_session = 0.0f;
    global_state->token_num_of_last_session = 0;
    global_state->llm_enable_observation = 0;
#ifdef ASR_ENABLED
    global_state->asr_output_buffer = (wchar_t*)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
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

    global_state->ui_font = GFX_FONT_ALPHA_16;
}


void ui_draw_image(Key_Event *key_event, Global_State *global_state, const char *img_path, int32_t is_reload) {

    int32_t is_new_image_file = (strcmp(img_path, s_image_filename_buffer) != 0);

    // 如果显式指定reload，或者图像文件名跟上次调用不同，则清除缓冲区，重新读文件并解码
    if (is_reload || is_new_image_file) {
        printf("reload/refresh image: %s\n", img_path);
        strcpy(s_image_filename_buffer, img_path);
        s_image_decode_ready = 0;
        if (s_image_file_buffer != NULL) {
            free(s_image_file_buffer);
            s_image_file_buffer = NULL;
        }
        if (s_image_rgb888_buffer != NULL) {
            free(s_image_rgb888_buffer);
            s_image_rgb888_buffer = NULL;
        }
    }

    // 首次加载：从SD卡读取图像文件到文件缓冲区
    if (s_image_file_buffer == NULL) {
        int32_t ret = platform_read_file_to_buffer(img_path, &s_image_file_buffer, &s_image_file_size);
        // printf("platform_read_file_to_buffer %d\n", ret);
    }

    // 首次解码：将图像文件解码到 RGB888 像素缓冲区（避免每次渲染都重新解码）
    if (s_image_file_buffer != NULL && s_image_file_size > 0 && !s_image_decode_ready) {
        if (s_image_rgb888_buffer == NULL) {
            s_image_rgb888_buffer = (uint8_t *)platform_malloc(SCREEN_WIDTH * SCREEN_HEIGHT * 3);
        }
        if (s_image_rgb888_buffer != NULL) {
            int32_t ret = gfx_decode_image_buffer(
                s_image_file_buffer, s_image_file_size,
                SCREEN_WIDTH, SCREEN_HEIGHT,
                s_image_rgb888_buffer,
                &s_image_width, &s_image_height
            );
            if (ret == 0) {
                s_image_decode_ready = 1;
                printf("gfx_decode_image_buffer ok, %dx%d\n", s_image_width, s_image_height);
            } else {
                printf("gfx_decode_image_buffer failed\n");
            }
        }
    }

    // 优先使用已解码的 RGB888 缓冲区绘制壁纸
    if (s_image_decode_ready && s_image_rgb888_buffer != NULL) {
        gfx_draw_rgb888_buffer(
            global_state->gfx, s_image_rgb888_buffer,
            s_image_width, s_image_height,
            0, 0
        );
    }
    // 若解码尚未完成或失败，回退到原始方式（带实时解码）
    else if (s_image_file_buffer != NULL && s_image_file_size > 0) {
        gfx_draw_image_buffer(global_state->gfx, s_image_file_buffer, s_image_file_size, 0, 0, 0, 0);
        printf("gfx_draw_image_buffer\n");
    }
}


// ===============================================================================
// 核心业务：电子鹦鹉
// ===============================================================================

typedef struct {
    wchar_t *model_name;
    int32_t is_thinking_model;
    char *model_path;
    char *lora_path;
    float repetition_penalty;
    float temperature;
    float top_p;
    uint32_t top_k;
    uint32_t max_seq_len;
} Model_Config;

#define MODEL_CONFIG_ENTRY(name, is_think, m_path, l_path, rep_pen, temp, top_p_val, top_k_val, max_seq) \
    { \
        .model_name = (name), \
        .is_thinking_model = (is_think), \
        .model_path = (m_path), \
        .lora_path = (l_path), \
        .repetition_penalty = (rep_pen), \
        .temperature = (temp), \
        .top_p = (top_p_val), \
        .top_k = (top_k_val), \
        .max_seq_len = (max_seq) \
    }


static const Model_Config preset_model_configs[] = {
    MODEL_CONFIG_ENTRY(L"Nano-168M", 0, MODEL_ROOT_DIR "/nano-168m-q80.bin", NULL, 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Nano-56M", 0, MODEL_ROOT_DIR "/nano-56m-q80.bin", NULL, 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Nano-56M-Neko", 0, MODEL_ROOT_DIR "/nano-56m-base-q80.bin", MODEL_ROOT_DIR "/nano-56m-lora-neko.bin", 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Qwen3-0.6B", 1, MODEL_ROOT_DIR "/qwen3-0b6-q80.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-1.7B", 1, MODEL_ROOT_DIR "/qwen3-1b7-q80.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-4B-Inst", 0, MODEL_ROOT_DIR "/qwen3-4b-instruct-2507-q80.bin", NULL, 1.0f, 0.7f, 0.8f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-4B-Think", 1, MODEL_ROOT_DIR "/qwen3-4b-thinking-2507-q80.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Nano-168M-Q4KS", 0, MODEL_ROOT_DIR "/nano-168m-q4ks.bin", NULL, 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Nano-56M-Q4KS", 0, MODEL_ROOT_DIR "/nano-56m-q4ks.bin", NULL, 1.05f, 1.0f, 0.5f, 0, 512),
    MODEL_CONFIG_ENTRY(L"Qwen3-0.6B-Q4KS", 1, MODEL_ROOT_DIR "/qwen3-0b6-q4ks.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-1.7B-Q4KS", 1, MODEL_ROOT_DIR "/qwen3-1b7-q4ks.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-4B-Inst-Q4KS", 0, MODEL_ROOT_DIR "/qwen3-4b-instruct-2507-q4ks.bin", NULL, 1.0f, 0.7f, 0.8f, 20, 32768),
    MODEL_CONFIG_ENTRY(L"Qwen3-4B-Think-Q4KS", 1, MODEL_ROOT_DIR "/qwen3-4b-thinking-2507-q4ks.bin", NULL, 1.0f, 0.6f, 0.95f, 20, 32768)
};

// Qwen3思考模式和非思考模式的参数不同：分别是temperature和top-p
static const float qwen3_infer_args_thinking[2] = {0.6f, 0.95f};
static const float qwen3_infer_args_no_thinking[2] = {0.7f, 0.8f};


int32_t on_llm_prefilling(Key_Event *key_event, Global_State *global_state) {
    Nano_Session *session = global_state->llm_session;

    if (session->t_0 == 0) {
        session->t_0 = global_state->timestamp;
    }
    else {
        session->tps = (session->pos - 1) / (float)(global_state->timestamp - session->t_0) * 1000;
    }

    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        wcscpy(global_state->llm_output_of_last_session, L"");
        global_state->tps_of_last_session = session->tps;
        global_state->token_num_of_last_session = session->pos;
        return LLM_STOPPED_IN_PREFILLING;
    }

    // PREFILL_LED_ON

    // 屏幕刷新节流
    if (global_state->timestamp - global_state->llm_refresh_timestamp > (1000 / global_state->llm_refresh_max_fps)) {
        // 临时关闭draw_textarea的gfx_refresh，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
        global_state->is_full_refresh = 0;

        // 清屏
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            gfx_fill_white(global_state->gfx);
        }
        else if (global_state->ui_color_style == UI_COLOR_DARK) {
            gfx_soft_clear(global_state->gfx);
        }

        // 显示界面标题
        wchar_t prefill_title_str[50];
        swprintf(prefill_title_str, 50, L"%ls Reading...", global_state->llm_model_name);
        ui_draw_header(key_event, global_state, prefill_title_str, 1);

        // 显示已经处理的输入prompt（复用主文本框控件）
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, session->output_text, -1, 1);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

        // 进度条
        uint8_t progress_R = 102, progress_G = 204, progress_B = 255;
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            progress_R = 102; progress_G = 204; progress_B = 255;
        }
        else if (global_state->ui_color_style == UI_COLOR_DARK) {
            progress_R = 102; progress_G = 204; progress_B = 255;
        }
        // 进度条位于页脚（底部状态栏）上缘，页脚高度跟随当前字体行高（行高 + 1px 边距）
        uint32_t pg_bottom_y = global_state->gfx->height - (gfx_font_line_height(global_state->ui_font) + 1);
        uint32_t pgpos_x = MIN(global_state->gfx->width - 1, session->pos * global_state->gfx->width / (session->num_prompt_tokens - 1));
        gfx_draw_line(global_state->gfx, 1, (pg_bottom_y - 1), pgpos_x, (pg_bottom_y - 1), progress_R, progress_G, progress_B, 1);
        gfx_draw_line(global_state->gfx, 1, (pg_bottom_y - 2), pgpos_x, (pg_bottom_y - 2), progress_R, progress_G, progress_B, 1);

        // 进度百分比
        wchar_t progress_str[30];
        swprintf(progress_str, 30, L"%d/%d", session->pos, session->num_prompt_tokens);
        ui_draw_footer(key_event, global_state, progress_str, 1);

        gfx_refresh(global_state->gfx);

        // 重新开启整帧绘制，注意这个标记是所有函数共享的全局标记。
        global_state->is_full_refresh = 1;

        global_state->llm_refresh_timestamp = global_state->timestamp;
    }

#ifdef TTS_ENABLED
    reset_tts_split_status();
#endif

    // PREFILL_LED_OFF
    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_llm_decoding(Key_Event *key_event, Global_State *global_state) {
    Nano_Session *session = global_state->llm_session;

    if (session->t_0 == 0) {
        session->t_0 = global_state->timestamp;
    }
    else {
        session->tps = (session->pos - 1) / (float)(global_state->timestamp - session->t_0) * 1000;
    }

    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        if (session->output_text) {
            wcscpy(global_state->llm_output_of_last_session, session->output_text);
        }
        global_state->tps_of_last_session = session->tps;
        global_state->token_num_of_last_session = session->pos;
        return LLM_STOPPED_IN_DECODING;
    }

    // DECODE_LED_ON

    // 屏幕刷新节流
    if (global_state->timestamp - global_state->llm_refresh_timestamp > (1000 / global_state->llm_refresh_max_fps)) {
        // 标题
        wchar_t title_str[50];
        swprintf(title_str, 50, L"%ls Decoding...", global_state->llm_model_name);
        ui_draw_header(key_event, global_state, title_str, 1);

        // 底部
        wchar_t tps_str[50];
        swprintf(tps_str, 50, L"%ls | %d/%d | %.1f词元/秒", global_state->llm_model_name, session->pos, global_state->llm_max_seq_len, session->tps);
        ui_draw_footer(key_event, global_state, tps_str, 1);

        // 刷新输出文本
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, session->output_text, -1, 1);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        global_state->llm_refresh_timestamp = global_state->timestamp;
    }

    // DECODE_LED_OFF

#ifdef TTS_ENABLED
    if (global_state->tts_req_mode > 0) {
        send_tts_request(session->output_text, 0);
    }
#endif

    return LLM_RUNNING_IN_DECODING;
}

int32_t on_llm_finished(Key_Event *key_event, Global_State *global_state) {
    Nano_Session *session = global_state->llm_session;

    session->t_1 = global_state->timestamp;
    session->tps = (session->pos - 1) / (float)(session->t_1 - session->t_0) * 1000;

    if (session->output_text) {
        // 缓冲区容量为 UI_STR_BUF_MAX_LENGTH * 2 个 wchar_t（含结尾 L'\0'），截断拷贝防堆溢出
        wcsncpy(global_state->llm_output_of_last_session, session->output_text, UI_STR_BUF_MAX_LENGTH * 2 - 1);
        global_state->llm_output_of_last_session[UI_STR_BUF_MAX_LENGTH * 2 - 1] = L'\0';
    }

    // 将本轮对话写入日志
    // write_chat_log(LOG_FILE_PATH, global_state->timestamp, session->prompt, global_state->llm_output_of_last_session);

#ifdef TTS_ENABLED
    if (global_state->tts_req_mode > 0) {
        send_tts_request(session->output_text, 1);
    }
    reset_tts_split_status();
#endif

    global_state->tps_of_last_session = session->tps;
    global_state->token_num_of_last_session = session->pos;

    return LLM_STOPPED_NORMALLY;
}


void init_model_menu(Key_Event *key_event, Global_State *global_state) {
    size_t model_count = sizeof(preset_model_configs) / sizeof(preset_model_configs[0]);
    // 条目字符串借用 preset_model_configs 的静态存储，控件不复制
    static const wchar_t *model_menu_items[sizeof(preset_model_configs) / sizeof(preset_model_configs[0])];
    for (size_t i = 0; i < model_count; i++) {
        model_menu_items[i] = preset_model_configs[i].model_name;
    }
    global_state->w_menu_main->title = L"选择语言模型";
    global_state->w_menu_main->items = model_menu_items;
    global_state->w_menu_main->item_num = (int32_t)model_count;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
}


void init_game_menu(Key_Event *key_event, Global_State *global_state) {
    // 条目字符串借用字面量的静态存储，控件不复制
    static const wchar_t *game_menu_items[] = {
        L"黄金矿工",
        L"Bad Apple!",
        L"康威生命游戏",
        L"遗传算法求解旅行商问题",
        L"遗传算法拟合图像",
        L"俄罗斯方块",
        L"粒子生命",
        L"水波",
        L"水池",
        L"体积云",
        L"日历",
    };
    global_state->w_menu_main->title = L"小游戏";
    global_state->w_menu_main->items = game_menu_items;
    global_state->w_menu_main->item_num = 11;
    ui_widget_menu_init(key_event, global_state, global_state->w_menu_main);
}

int32_t game_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    switch (ms->current_item_index) {
        case 0: return STATE_GOLDMINER;
        case 1: return STATE_BADAPPLE;
        case 2: return STATE_GAMEOFLIFE;
        case 3: return STATE_GENETIC_TSP;
        case 4: return STATE_GENETIC;
        case 5: return STATE_TETRIS;
        case 6: return STATE_PARTICLELIFE;
        case 7: return STATE_RIPPLE;
        case 8: return STATE_WATER;
        case 9: return STATE_CLOUD;
        case 10: return STATE_CALENDAR;
        default: return STATE_MAIN_MENU;
    }
}


void ui_app_llm_model_diagram_draw(Key_Event *key_event, Global_State *global_state, int32_t x0, int32_t y0, int32_t total_layers, Nano_Observation obs) {
    Nano_GFX *gfx = global_state->gfx;
    int32_t layer = obs.layer;
    int32_t phase = obs.phase;

    // 色彩
    uint8_t bg_R = 0x00, bg_G = 0x00, bg_B = 0x00;
    uint8_t line_R = 0x99, line_G = 0x99, line_B = 0x99;
    uint8_t block_R = 0x33, block_G = 0x33, block_B = 0x33;
    uint8_t block_active_R = 0x00, block_active_G = 0xff, block_active_B = 0x00;
    uint8_t text_R = 0xcc, text_G = 0xcc, text_B = 0xcc;
    uint8_t text_active_R = 0xff, text_active_G = 0xff, text_active_B = 0xff;

    // 绘制连线
    gfx_draw_line(gfx, x0+55, y0+14, x0+55, y0+50, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+50, x0+95, y0+50, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+79, x0+95, y0+79, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+50, x0+15, y0+79, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+95, y0+50, x0+95, y0+79, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+55, y0+79, x0+55, y0+133, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+35, y0+133, x0+95, y0+133, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+35, y0+133, x0+35, y0+160, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+95, y0+133, x0+95, y0+203, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+160, x0+55, y0+160, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+203, x0+15, y0+160, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+55, y0+160, x0+55, y0+227, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+15, y0+203, x0+95, y0+203, line_R, line_G, line_B, 1);

    gfx_draw_line(gfx, x0+36, y0+103, x0+55, y0+103, line_R, line_G, line_B, 1); // Res Branch
    gfx_draw_line(gfx, x0+53, y0+103, x0+57, y0+103, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+55, y0+101, x0+55, y0+105, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+54, y0+103, x0+56, y0+103, line_R+10, line_G+10, line_B+10, 1);
    gfx_draw_line(gfx, x0+55, y0+102, x0+55, y0+104, line_R+10, line_G+10, line_B+10, 1);
    gfx_draw_textline_centered(gfx, L"X1", x0+36-6, y0+103, line_R, line_G, line_B, 1);
    gfx_draw_line(gfx, x0+36, y0+20, x0+55, y0+20, line_R, line_G, line_B, 1); // Res Branch
    gfx_draw_line(gfx, x0+53, y0+20, x0+57, y0+20, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+55, y0+20-2, x0+55, y0+20+2, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+54, y0+20, x0+56, y0+20, line_R+10, line_G+10, line_B+10, 1);
    gfx_draw_line(gfx, x0+55, y0+20-1, x0+55, y0+20+1, line_R+10, line_G+10, line_B+10, 1);
    gfx_draw_textline_centered(gfx, L"X2", x0+36-6, y0+20, line_R, line_G, line_B, 1);


    // 绘制方框和文字

    uint8_t bR = block_R, bG = block_G, bB = block_B;
    uint8_t tR = text_R, tG = text_G, tB = text_B;

    // NANO_LLM_PHASE_W2
    if (phase == NANO_LLM_PHASE_W2) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+40, y0+25, 30, 14, bR, bG, bB, 1); // W2
    gfx_draw_textline_centered(gfx, L"W2", x0+40+15, y0+25+7, tR, tG, tB, 1);
    gfx_draw_circle_fill(gfx, x0+55, y0+50, 6, bR, bG, bB, 1); // FFN Hadamard
    gfx_draw_line(gfx, x0+49, y0+44, x0+49+12, y0+44+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+49+12, y0+44, x0+49, y0+44+12, bg_R, bg_G, bg_B, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_W1W3
    if (phase == NANO_LLM_PHASE_W1W3) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+0, y0+61, 30, 14, bR, bG, bB, 1); // W1
    gfx_draw_textline_centered(gfx, L"W1", x0+0+15, y0+61+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+80, y0+43, 30, 14, bR, bG, bB, 1); // SiLU
    gfx_draw_textline_centered(gfx, L"SiLU", x0+80+15, y0+43+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+80, y0+61, 30, 14, bR, bG, bB, 1); // W3
    gfx_draw_textline_centered(gfx, L"W3", x0+80+15, y0+61+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_FFN_NORM
    if (phase == NANO_LLM_PHASE_FFN_NORM) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+40, y0+83, 30, 14, bR, bG, bB, 1); // FFN Norm
    gfx_draw_textline_centered(gfx, L"Norm", x0+40+15, y0+83+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_O
    if (phase == NANO_LLM_PHASE_O) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+40, y0+109, 30, 14, bR, bG, bB, 1); // O
    gfx_draw_textline_centered(gfx, L"O", x0+40+15, y0+109+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_MHA
    if (phase == NANO_LLM_PHASE_MHA) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+29, y0+138, 12, 12, bR, bG, bB, 1); // Mask
    gfx_draw_line(gfx, x0+29, y0+138, x0+29+12, y0+138+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_circle_fill(gfx, x0+55, y0+133, 6, bR, bG, bB, 1); // A*V
    gfx_draw_line(gfx, x0+49, y0+127, x0+49+12, y0+127+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+49+12, y0+127, x0+49, y0+127+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_circle_fill(gfx, x0+35, y0+160, 6, bR, bG, bB, 1); // Q*K
    gfx_draw_line(gfx, x0+29, y0+154, x0+29+12, y0+154+12, bg_R, bg_G, bg_B, 1);
    gfx_draw_line(gfx, x0+29+12, y0+154, x0+29, y0+154+12, bg_R, bg_G, bg_B, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_QK_ROPE
    if (phase == NANO_LLM_PHASE_QK_ROPE) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+0, y0+168, 30, 14, bR, bG, bB, 1); // RoPE Q
    gfx_draw_textline_centered(gfx, L"RoPE", x0+0+15, y0+168+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+40, y0+168, 30, 14, bR, bG, bB, 1); // RoPE K
    gfx_draw_textline_centered(gfx, L"RoPE", x0+40+15, y0+168+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_QKV
    if (phase == NANO_LLM_PHASE_QKV) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+0, y0+185, 30, 14, bR, bG, bB, 1); // Q
    gfx_draw_textline_centered(gfx, L"Q", x0+0+15, y0+185+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+40, y0+185, 30, 14, bR, bG, bB, 1); // K
    gfx_draw_textline_centered(gfx, L"K", x0+40+15, y0+185+7, tR, tG, tB, 1);
    gfx_draw_rectangle(gfx, x0+80, y0+185, 30, 14, bR, bG, bB, 1); // V
    gfx_draw_textline_centered(gfx, L"V", x0+80+15, y0+185+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;

    // NANO_LLM_PHASE_ATTN_NORM
    if (phase == NANO_LLM_PHASE_ATTN_NORM) { bR = block_active_R; bG = block_active_G; bB = block_active_B; tR = text_active_R; tG = text_active_G; tB = text_active_B;}
    gfx_draw_rectangle(gfx, x0+40, y0+207, 30, 14, bR, bG, bB, 1); // Attn Norm
    gfx_draw_textline_centered(gfx, L"Norm", x0+40+15, y0+207+7, tR, tG, tB, 1);
    bR = block_R; bG = block_G; bB = block_B;
    tR = text_R; tG = text_G; tB = text_B;


    // 绘制模型各层
    int32_t H = gfx->height - 14 - 14;
    int32_t layer_h = floorf(H / ((total_layers+2) * 2.0f));
    int32_t delta_y = (int32_t)floorf((float)(H - layer_h) / (float)((total_layers+2) - 1));
    int32_t y_pos = y0 + 14;
    for (int32_t ll = total_layers+1; ll >= 0; ll--) { // 包含Embd和Cls额外两层
        if ((layer == -1 && ll == 0) || (ll == layer + 1)) {
            bR = 0x00; bG = 0xff; bB = 0xff;
        }
        else {
            bR = block_R; bG = block_G; bB = block_B;
        }
        gfx_draw_rectangle(gfx, x0+120, y_pos, 16, layer_h, bR, bG, bB, 1);

        y_pos += delta_y;
    }

    // 显示top6
    static uint32_t tokens[6];
    Nano_Context *ctx = global_state->llm_ctx;
    if (obs.token_0) {
        tokens[0] = obs.token_0;
        tokens[1] = obs.token_1;
        tokens[2] = obs.token_2;
        tokens[3] = obs.token_3;
        tokens[4] = obs.token_4;
        tokens[5] = obs.token_5;
    }
    for (int32_t i = 0; i < 6; i++) {
        wchar_t *top_token_text = NULL;
        if (ctx->llm->arch == LLM_ARCH_NANO) {
            top_token_text = decode_nano(ctx->tokenizer, tokens + i, 1);
        }
        else if (ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3) {
            top_token_text = decode_bpe(ctx->tokenizer, tokens + i, 1);
        }
        else {
            return;
        }
        gfx_draw_textline(gfx, top_token_text, x0+ 140, y0+140+i*13, 255, 255, 0, 1);
        free(top_token_text);
    }

}


void llm_observation(Nano_Observation obs, void *env) {
    Global_State *global_state = (Global_State*)env;
    if (!global_state->llm_enable_observation) return;

    Nano_GFX *gfx = global_state->gfx;
    int32_t total_layers = global_state->llm_ctx->llm->config.n_layer;

    gfx_draw_rectangle(gfx, 0, 14, gfx->width/2, gfx->height-14-14, 0, 0, 0, 1);
    ui_app_llm_model_diagram_draw(NULL, global_state, 0, 0, total_layers, obs);
    gfx_refresh(gfx);
}

int32_t model_menu_item_action(Key_Event *ke, Global_State *gs, Widget_Menu_State *ms) {
    int32_t item_index = ms->current_item_index;

    if (gs->llm_ctx) {
        llm_context_free(gs->llm_ctx);
        gs->llm_ctx = NULL; // 释放后置NULL，避免悬垂指针被二次释放
    }

    int32_t model_count = (int32_t)(sizeof(preset_model_configs) / sizeof(preset_model_configs[0]));

    if (item_index < model_count) {
        Model_Config mc = preset_model_configs[item_index];
        gs->llm_model_name = mc.model_name;
        gs->llm_is_thinking_model = mc.is_thinking_model;
        gs->llm_model_path = mc.model_path;
        gs->llm_lora_path = mc.lora_path;
        gs->llm_repetition_penalty = mc.repetition_penalty;
        gs->llm_temperature = mc.temperature;
        gs->llm_top_p = mc.top_p;
        gs->llm_top_k = mc.top_k;
        gs->llm_max_seq_len = mc.max_seq_len;
    }
    else {
        return STATE_MAIN_MENU;
    }

    wchar_t llm_loading_prompt[88];
    swprintf(llm_loading_prompt, 88, L" 正在加载语言模型\n %ls\n 请稍等...", gs->llm_model_name);

    ui_widget_textarea_set(ke, gs, gs->w_textarea_main, llm_loading_prompt, 0, 0);
    ui_widget_textarea_draw(ke, gs, gs->w_textarea_main);

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    gs->llm_ctx = llm_context_init_from_buffer(
        (uint8_t *)psycho_230k_1214_q80,
        gs->llm_max_seq_len,
        1.0,
        1.0,
        0.8,
        20,
        gs->timestamp);
#else
    gs->llm_ctx = llm_context_init(
        gs->llm_model_path,
        gs->llm_lora_path,
        gs->llm_max_seq_len,
        gs->llm_repetition_penalty,
        gs->llm_temperature,
        gs->llm_top_p,
        gs->llm_top_k,
        gs->timestamp);
#endif


    gs->llm_ctx->observation = llm_observation;
    gs->llm_ctx->observation_env = gs; // 模拟闭包：将观测函数的词法环境指向UI全局上下文，这样就可以在观测回调中使用UI的API

    if (gs->llm_enable_observation) {
        gs->w_textarea_main->x = 160;
        gs->w_textarea_main->width = gs->gfx->width - 160;
    }

    // 进入电子鹦鹉
    ui_widget_input_init(ke, gs, gs->w_input_main, gs->llm_model_name);
    return STATE_LLM_INPUT;
}









// ===============================================================================
// 主菜单
// ===============================================================================


static void ui_app_main_menu_grid16_refresh_button(
    Key_Event *key_event, Global_State *global_state,
    int32_t col, int32_t row, wchar_t *text,
    const char *icon_path,
    uint8_t cell_bg_R, uint8_t cell_bg_G, uint8_t cell_bg_B, uint8_t cell_bg_mode,
    uint8_t cell_text_R, uint8_t cell_text_G, uint8_t cell_text_B, uint8_t cell_text_mode
) {
    int32_t bx = (col == 0) ? 1 : 0;
    int32_t by = (row == 0) ? 1 : 0;
    // 网格布局：上下留白与顶/底栏高度一致（当前 UI 字体行高 + 1）
    const int32_t bar_height = 0; // gfx_font_line_height(global_state->ui_font) + 1;
    UI_Grid_Layout grid = ui_grid_layout_make(global_state->gfx->width, global_state->gfx->height, bar_height, bar_height, 0, 0, 4, 4);
    int32_t cell_cx = ui_grid_cell_center_x(&grid,col);
    int32_t cell_cy = ui_grid_cell_center_y(&grid,row);
    gfx_draw_rectangle(global_state->gfx, ui_grid_cell_x0(&grid,col)+bx, ui_grid_cell_y0(&grid,row)+by, ui_grid_cell_width(&grid)-1-bx, ui_grid_cell_height(&grid)-1-by, cell_bg_R, cell_bg_G, cell_bg_B, cell_bg_mode);
    // 图标（36x36，带PSRAM缓存）：中心点位于格子中心偏上8px
    ui_icon_draw_centered(global_state->gfx, icon_path, cell_cx, cell_cy - 8);
    // 文字：中心点位于图标下沿下方8px。后绘制文字，使其覆盖在图标上面
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, text, cell_cx, cell_cy - 8 + 18 + 8, cell_text_R, cell_text_G, cell_text_B, cell_text_mode);
}

void ui_widget_grid16_draw(Key_Event *key_event, Global_State *global_state) {
    wchar_t cell_text[4][4][2][10] = {
        { {L"[1]", L"番茄表",}, {L"[2]", L"鹦鹉笼",}, {L"[3]", L"玲珑仪",}, {L"[A]", L"返回",}, },
        { {L"[4]", L"电子书",}, {L"[5]", L"音乐盒",}, {L"[6]", L"时光集",}, {L"[B]", L"设置",}, },
        { {L"[7]", L"小游戏",}, {L"[8]", L"频谱仪",}, {L"[9]", L"寻呼机",}, {L"[C]", L"自述",}, },
        // { {L"[*]", L"计步器",}, {L"[0]", L"手电筒",}, {L"[#]", L"控制台",}, {L"[D]", L"关机",}, }, // 原入口（手电筒已征用为电子词典）
        { {L"[*]", L"计步器",}, {L"[0]", L"词典",}, {L"[#]", L"控制台",}, {L"[D]", L"关机",}, },
    };

    // 清屏
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    uint8_t cell_bg_R = 0, cell_bg_G = 0, cell_bg_B = 0;
    uint8_t cell_text_R = 0, cell_text_G = 0, cell_text_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        cell_bg_R = 0xff;
        cell_bg_G = 0xff;
        cell_bg_B = 0xff;
        cell_text_R = 0x66;
        cell_text_G = 0x66;
        cell_text_B = 0x66;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        cell_bg_R = 30;
        cell_bg_G = 30;
        cell_bg_B = 32;
        cell_text_R = 0xcc;
        cell_text_G = 0xcc;
        cell_text_B = 0xcc;
    }

    // 所有格子的图标路径（暂定全部为 animac.png，后续逐格替换）。尺寸均为36x36。
    // 图标经 ui_icon_draw_centered 绘制：首次读取解码后常驻缓存于PSRAM，之后直接从缓存绘制
    const char *cell_icon_path[4][4] = {
        {"/icon/fanqie.png", "/icon/nano.png", "/icon/linglong.png", "/icon/home.png"},
        {"/icon/ebook.png", "/icon/default.png", "/icon/album.png", "/icon/settings.png"},
        {"/icon/game.png", "/icon/default.png", "/icon/ptt.png", "/icon/default.png"},
        {"/icon/default.png", "/icon/default.png", "/icon/animac.png", "/icon/poweroff.png"},
    };

    for (int32_t row = 0; row < 4; row++) {
        for (int32_t col = 0; col < 4; col++) {
            ui_app_main_menu_grid16_refresh_button(key_event, global_state,
                col, row, cell_text[row][col][1],
                cell_icon_path[row][col],
                cell_bg_R, cell_bg_G, cell_bg_B, 1,
                cell_text_R, cell_text_G, cell_text_B, 1);

        }
    }

    // ui_draw_header(key_event, global_state, L"Nano-Pod", 1);
    // ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);
}

void ui_widget_grid16_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 番茄表
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        global_state->STATE = STATE_FLIP;
    }
    // 鹦鹉笼
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_2) {
        init_model_menu(key_event, global_state);
        global_state->STATE = STATE_MODEL_MENU;
    }
    // 玲珑仪
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_3) {
        global_state->STATE = STATE_LINGLONG;
    }
    // 电子书
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_4) {
        ui_ebook_menu_init(key_event, global_state);
        global_state->STATE = STATE_EBOOK;
    }
    // 音乐盒
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_5) {
        ui_musicbox_menu_init(key_event, global_state);
        global_state->STATE = STATE_MUSICBOX_MENU;
    }
    // 时光集
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_6) {
        global_state->STATE = STATE_ALBUM;
    }
    // 小游戏
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_7) {
        init_game_menu(key_event, global_state);
        global_state->STATE = STATE_GAME_MENU;
    }
    // 频谱仪
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_8) {
        global_state->STATE = STATE_SPECTROGRAM;
    }
    // 寻呼机（OFDM 声波数传）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_9) {
        // TODO
        ui_ofdm_menu_init(key_event, global_state);
        global_state->STATE = STATE_OFDM_MENU;
    }
    // 电子词典（原“手电筒”入口及其颜色风格切换功能已取消，注释保留备查；
    // 颜色风格切换请使用“系统设置”中的正式按钮）
    // else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_0) {
    //     if (global_state->ui_color_style == UI_COLOR_LIGHT) {
    //         global_state->ui_color_style = UI_COLOR_DARK;
    //     }
    //     else {
    //         global_state->ui_color_style = UI_COLOR_LIGHT;
    //     }
    // }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_0) {
        // 索引就绪（缺失/失效则带进度构建）后进入查询状态；失败则停留在主菜单（错误画面已显示）
        if (ui_dict_enter(key_event, global_state) == 0) {
            global_state->STATE = STATE_DICT_QUERY;
        }
        else {
            sleep_in_ms(1500); // 错误画面停留片刻
            ui_widget_grid16_draw(key_event, global_state);
            gfx_refresh(global_state->gfx);
        }
    }
    // 返回
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_SPLASH_SCREEN;
    }
    // 设置
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_shift) {
        global_state->STATE = STATE_SETTING_MENU;
    }
    // 自述
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_ctrl) {
        global_state->STATE = STATE_README;
    }
    // 关机
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_enter) {
        global_state->STATE = STATE_SHUTDOWN;
    }
    // 计步器
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
        global_state->STATE = STATE_PEDOMETER;
    }
    // 控制台
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
        global_state->STATE = STATE_ANIMAC_INIT;
    }
    else {
        return;
    }
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

    // 清屏
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    // 绘制壁纸
    ui_draw_image(key_event, global_state, WALLPAPER_PATH, 0);

    // Header
    // ui_draw_header(key_event, global_state, L"Project Nano", 1);

    // 时间
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime); // 获取当前时间戳
    timeinfo = localtime(&rawtime); // 转换为本地时间

    int32_t year = timeinfo->tm_year + 1900;
    int32_t month = timeinfo->tm_mon + 1;
    int32_t day = timeinfo->tm_mday;
    int32_t hour = timeinfo->tm_hour;
    int32_t minute = timeinfo->tm_min;
    int32_t second = timeinfo->tm_sec;
    double timezone = 8.0;
    double longitude = 119.0;
    double latitude = 32.0;

    wchar_t datetime_wcs_buffer[33];
    wchar_t nongli_wcs_buffer[33];

    uint8_t time_red = 0, time_green = 0, time_blue = 0;
    uint8_t nongli_red = 0, nongli_green = 0, nongli_blue = 0;
    uint8_t sevenseg_red = 0, sevenseg_green = 0, sevenseg_blue = 0;
    uint32_t sevenseg_shadow = 1;

    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        time_red = 0; time_green = 0; time_blue = 0;
        nongli_red = 0xff; nongli_green = 0xfb; nongli_blue = 0;
        sevenseg_red = 255; sevenseg_green = 255; sevenseg_blue = 255;
        sevenseg_shadow = 1;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        time_red = 255; time_green = 255; time_blue = 255;
        nongli_red = 0xff; nongli_green = 0xfb; nongli_blue = 0;
        sevenseg_red = 255; sevenseg_green = 255; sevenseg_blue = 255;
        sevenseg_shadow = 1;
    }

    const wchar_t *weekdays[] = {L"日", L"一", L"二", L"三", L"四", L"五", L"六"};
    swprintf(datetime_wcs_buffer, 33, L"%04d年%02d月%02d日 星期%ls", year, month, day, weekdays[timeinfo->tm_wday]);
    // 背景（阴影）
    gfx_font_draw_text_centered(global_state->gfx, global_state->ui_font, datetime_wcs_buffer, global_state->gfx->width / 2 + 1, 16 + 1, 0x66, 0x66, 0x66, 1);
    // 前景
    gfx_font_draw_text_centered(global_state->gfx, global_state->ui_font, datetime_wcs_buffer, global_state->gfx->width / 2, 16, time_red, time_green, time_blue, 1);

    // 农历日期
    LunarDate *nongli = lunar_calculate(year, month, day, hour, minute, second, timezone);
    _mbstowcs(nongli_wcs_buffer, nongli->full_display, 33);
    // 背景（阴影）
    gfx_font_draw_text_centered(global_state->gfx, global_state->ui_font, nongli_wcs_buffer, global_state->gfx->width / 2 + 1, 190 + 1, 0x66, 0x66, 0x66, 1);
    // 前景
    gfx_font_draw_text_centered(global_state->gfx, global_state->ui_font, nongli_wcs_buffer, global_state->gfx->width / 2, 190, nongli_red, nongli_green, nongli_blue, 1);

    // 七段码时钟
    wchar_t time7seg_str[10];
    swprintf(time7seg_str, 10, L"%02d:%02d:%02d", hour, minute, second);
    int32_t s7seg_width = 0.0f;
    int32_t s7seg_height = 0.0f;
    ui_draw_7seg_string_centered(key_event, global_state,
        global_state->gfx->width / 2 + 6, 55,
        time7seg_str, sevenseg_red, sevenseg_green, sevenseg_blue, 12.0f, 4.0f, 8.0f, sevenseg_shadow,
        &s7seg_width, &s7seg_height);


    // 玲珑仪（青春版）
    // ui_app_linglong_draw_lite(key_event, global_state, (global_state->gfx->width - 128) / 2, 100,
    //     year, month, day, hour, minute, second, longitude, latitude, timezone);


    // Footer
    // if (global_state->gfx->width > 128) {
    //     ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);
    // }
    // else {
    //     ui_draw_copyright_notice(key_event, global_state, 20, 53);
    // }



    // 时间戳
    // wchar_t ts_text[100];
    // swprintf(ts_text, 100, L"Timestamp: %llu | Ticks: %d", global_state->timestamp, global_state->timer);
    // gfx_draw_textline_centered(global_state->gfx, ts_text, global_state->gfx->width/2, global_state->gfx->height-13*3-6, time_red, time_green, time_blue, 1);



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

    // 显示电量信息文字（若启用自动关机，在电量信息尾部追加关机倒计时 " | mm:ss"；
    // splash 每 100ms 重绘，倒计时随之逐秒刷新）
    wchar_t battery_info_buf[112];
    wchar_t auto_shutdown_buf[24] = L"";
    if (global_state->auto_shutdown_deadline != 0) {
        uint64_t remain_ms = (global_state->timestamp < global_state->auto_shutdown_deadline)
                           ? (global_state->auto_shutdown_deadline - global_state->timestamp) : 0;
        uint32_t remain_s = (uint32_t)((remain_ms + 999) / 1000); // 向上取整，避免剩余不足1秒时显示 00:00
        swprintf(auto_shutdown_buf, 24, L" | %02d:%02d", (int)(remain_s / 60), (int)(remain_s % 60));
    }
    swprintf(battery_info_buf, 112, L"电量:%d%% | %dmV | %dmA%ls%ls", global_state->ups_soc, global_state->ups_voltage, global_state->ups_current, (global_state->ups_is_charging ? L"  |  正在充电" : L""), auto_shutdown_buf);
    // 背景（阴影）
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, battery_info_buf, global_state->gfx->width/2 + 1, global_state->gfx->height-13*1-6 + 1, 0x66, 0x66, 0x66, 1);
    // 前景
    gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, battery_info_buf, global_state->gfx->width/2, global_state->gfx->height-13*1-6, time_red, time_green, time_blue, 1);

#endif


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

    gfx_refresh(global_state->gfx);
}


// ===============================================================================
// Bad Apple
// ===============================================================================

void ui_app_badapple_render_frame(Key_Event *key_event, Global_State *global_state) {
#ifdef BADAPPLE_ENABLED
    wchar_t ba_str[20];
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

        swprintf(ba_str, 20, L"%04d / 2193", global_state->ba_frame_count);
        gfx_draw_textline_centered(global_state->gfx, ba_str, global_state->gfx->width/2, global_state->gfx->height - 8, 255, 255, 255, 1);

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

static uint8_t *s_ui_app_gol_field_0 = NULL;
static uint8_t *s_ui_app_gol_field_1 = NULL;
static uint8_t s_ui_app_gol_field_page = 0;
static uint64_t s_ui_app_gol_refresh_timestamp = 0;
static uint32_t s_ui_app_gol_step_count = 0;
static int32_t s_gol_width = 0;
static int32_t s_gol_height = 0;

static inline uint8_t ui_app_gol_get_cell(uint8_t *field, int32_t w, int32_t h, int32_t x, int32_t y) {
    int32_t byte_index = (y * w + x) / 8;
    int32_t bit_rem = (y * w + x) % 8;
    return ((field[byte_index] & ((uint8_t)0x80u >> bit_rem)) != 0);
}

static inline void ui_app_gol_set_cell(uint8_t *field, int32_t w, int32_t h, int32_t x, int32_t y, uint8_t value) {
    int32_t byte_index = (y * w + x) / 8;
    int32_t bit_rem = (y * w + x) % 8;
    uint8_t oldv = field[byte_index];
    field[byte_index] = (oldv & ~((uint8_t)(0x80u >> bit_rem))) | ((uint8_t)(!!value << (7 - bit_rem)));
}

void ui_app_gol_init(Key_Event *key_event, Global_State *global_state, int32_t gol_width, int32_t gol_height) {
    s_gol_width = gol_width;
    s_gol_height = gol_height;

    s_ui_app_gol_step_count = 0;
    s_ui_app_gol_field_page = 0;
    s_ui_app_gol_refresh_timestamp = global_state->timestamp;
    uint64_t ts = global_state->timestamp;

    // 重新进入/刷新时先释放旧场，避免覆盖式分配造成泄漏（初次为NULL，free(NULL)安全）
    if (s_ui_app_gol_field_0) free(s_ui_app_gol_field_0);
    if (s_ui_app_gol_field_1) free(s_ui_app_gol_field_1);
    s_ui_app_gol_field_0 = (uint8_t*)platform_calloc(gol_width * gol_height / 8, sizeof(uint8_t));
    s_ui_app_gol_field_1 = (uint8_t*)platform_calloc(gol_width * gol_height / 8, sizeof(uint8_t));

    for (uint32_t x = 0; x < gol_width; x++) {
        for (uint32_t y = 0; y < gol_height; y++) {
            uint8_t s = random_u32(&ts) % 2;
            ui_app_gol_set_cell(s_ui_app_gol_field_0, gol_width, gol_height, x, y, s);
            ui_app_gol_set_cell(s_ui_app_gol_field_1, gol_width, gol_height, x, y, s);
        }
    }
}

void ui_app_gol_render_frame(Key_Event *key_event, Global_State *global_state) {
    // 节流：不大于50fps
    // if (global_state->timestamp - s_ui_app_gol_refresh_timestamp < 20) {
    //     return;
    // }
    // s_ui_app_gol_refresh_timestamp = global_state->timestamp;
    gfx_soft_clear(global_state->gfx);

    uint32_t total_count = 0;

    uint8_t *field     = (s_ui_app_gol_field_page) ? s_ui_app_gol_field_0 : s_ui_app_gol_field_1;
    uint8_t *field_new = (s_ui_app_gol_field_page) ? s_ui_app_gol_field_1 : s_ui_app_gol_field_0;
    for (uint32_t x = 0; x < s_gol_width; x++) {
        for (uint32_t y = 0; y < s_gol_height; y++) {
            // 获取某个格子的8邻域
            uint32_t count = 0;
            uint32_t x_a = (x == 0) ? (s_gol_width-1) : (x-1);
            uint32_t x_b = (x == (s_gol_width-1)) ? 0 : (x+1);
            uint32_t y_a = (y == 0) ? (s_gol_height-1) : (y-1);
            uint32_t y_b = (y == (s_gol_height-1)) ? 0 : (y+1);
            uint8_t n1 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_a, y_a); count += (n1 != 0);
            uint8_t n2 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height,  x , y_a); count += (n2 != 0);
            uint8_t n3 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_b, y_a); count += (n3 != 0);
            uint8_t n4 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_a,  y ); count += (n4 != 0);
            uint8_t n5 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height,  x ,  y ); // self
            uint8_t n6 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_b,  y ); count += (n6 != 0);
            uint8_t n7 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_a, y_b); count += (n7 != 0);
            uint8_t n8 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height,  x , y_b); count += (n8 != 0);
            uint8_t n9 = ui_app_gol_get_cell(field, s_gol_width, s_gol_height, x_b, y_b); count += (n9 != 0);

            uint8_t new_state = 0;
            if (n5 == 0) {
                new_state = (count == 3) ? 1 : 0;
            }
            else {
                new_state = (count == 2 || count == 3) ? 1 : 0;
            }

            ui_app_gol_set_cell(field_new, s_gol_width, s_gol_height, x, y, new_state);

            if (new_state) {
                gfx_draw_point(global_state->gfx, x, y, 0, 255, 255, 1);
                total_count++;
            }
        }
    }

    wchar_t text[100];
    swprintf(text, 100, L"康威生命游戏 | 迭代:%u | 存活:%u | 密度:%.2f%%", s_ui_app_gol_step_count, total_count, (float)total_count / (float)(s_gol_width * s_gol_height) * 100);
    gfx_draw_rectangle(global_state->gfx, 0, 0, global_state->gfx->width, 12, 39, 39, 39, 3);
    gfx_draw_textline(global_state->gfx, text, 0, 0, 255, 255, 255, 1);

    gfx_refresh(global_state->gfx);
    s_ui_app_gol_field_page = 1 - s_ui_app_gol_field_page;
    s_ui_app_gol_step_count++;
}


// ===============================================================================
// FLIP流体模拟
// ===============================================================================

static uint64_t s_ui_flip_first_load_timestamp = 0;
static int32_t s_ui_flip_setting_count = 0;
static int32_t s_ui_flip_show_particles = 1;
static int32_t s_ui_flip_show_grid = 1;
static int32_t s_ui_flip_is_throttle = 0;
static int32_t s_ui_flip_throttle = 0;
static int32_t s_ui_flip_init_throttle = 50;

static int32_t s_ui_flip_last_upper_count = 0; // 用于计算粒子流量
static uint64_t s_ui_flip_last_upper_count_timestamp = 0; // 用于计算粒子流量
static uint64_t s_ui_fanqie_start_timestamp = 0;
static uint64_t s_ui_fanqie_stop_timestamp = 0;
static int32_t s_ui_fanqie_is_running = 0;

static uint64_t s_ui_fanqie_alarm_start_timestamp = 0;
static uint64_t s_ui_fanqie_alarm_duration = 0;
static uint64_t s_ui_fanqie_alarm_count = 0;
static int32_t  s_ui_fanqie_alarm_phase = 0;      // 非阻塞报警状态机相位：0=空闲 1=鸣震段 2=静默段
static uint64_t s_ui_fanqie_alarm_phase_ts = 0;   // 当前相位起始时间戳

void ui_app_flip_init(Key_Event *key_event, Global_State *global_state) {
    s_ui_fanqie_start_timestamp = global_state->timestamp;
    s_ui_fanqie_stop_timestamp = 0;
    s_ui_fanqie_is_running = 1;
    s_ui_fanqie_alarm_count = 0;
    s_ui_fanqie_alarm_phase = 0;
    float k = (float)(global_state->gfx->width) / (float)(global_state->gfx->height);
    flip_init(k, 1.0f, FLIP_RESOLUTION, global_state->timestamp, 1);
}

void ui_app_flip_render_frame(Key_Event *key_event, Global_State *global_state) {

    static uint64_t frame_count = 0;
    static uint64_t last_time = 0;
    static int fps = 0;

    frame_count++;
    uint64_t now = global_state->timestamp;
    if (now - last_time >= 1000) {
        fps = (int)frame_count;
        frame_count = 0;
        last_time = now;
    }

    if (!s_ui_flip_first_load_timestamp) {
        s_ui_flip_first_load_timestamp = global_state->timestamp;
    }

    gfx_soft_clear(global_state->gfx);

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
    imu_read_angle(&(global_state->pitch), &(global_state->roll), &(global_state->yaw));
    printf("俯仰=%-10.2f    滚转=%-10.2f    航向=%-10.2f\n", global_state->pitch, global_state->roll, global_state->yaw);
    gravity_x = -9.8f * sinf(global_state->roll / 180.0f * M_PI);
    gravity_y = -9.8f * cosf(global_state->roll / 180.0f * M_PI);
#endif

#endif

    float k = (float)(global_state->gfx->width) / (float)(global_state->gfx->height);

    int32_t upper_count = 0;
    int32_t lower_count = 0;

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    float dt = (s_ui_flip_is_throttle) ? (0.8f / 60.0f) : (1.6f / 60.0f);
#else
    float dt = (s_ui_flip_is_throttle) ? (0.6f / 60.0f) : (1.6f / 60.0f);
#endif

    render_flip(
        global_state->gfx, 0, 0, global_state->gfx->width, global_state->gfx->height,
        k, 1.0f, FLIP_RESOLUTION,     /* pool_width, pool_height, resolution */
        gravity_x, gravity_y,    /* gravity_x, gravity_y */
        dt,    /* dt */
        0.8f,            /* flip_ratio */
        20, 2,           /* num_pressure_iters, num_particle_iters */
        1.0f,            /* over_relaxation */
        1, 1,            /* compensate_drift, separate_particles */
        s_ui_flip_show_particles, s_ui_flip_show_grid,
        (s_ui_flip_is_throttle) ? s_ui_flip_throttle : 0,
        &upper_count, &lower_count
    );

    // 绘制沙漏边界线 NOTE 硬编码
    gfx_draw_triangle(global_state->gfx, 0, 3, 150, 112, 0, 233, 0, 30, 31, 32, 1);
    gfx_draw_triangle(global_state->gfx, 150, 112, 0, 233, 150, 120, 0, 30, 31, 32, 1);

    gfx_draw_triangle(global_state->gfx, 319, 3, 178, 112, 178, 120, 0, 30, 31, 32, 1);
    gfx_draw_triangle(global_state->gfx, 319, 3, 178, 120, 319, 227, 0, 30, 31, 32, 1);

    // gfx_draw_line_anti_aliasing(global_state->gfx, 0, 3, 150, 112, 3, 0x00, 0x01, 0x02, 1);
    // gfx_draw_line_anti_aliasing(global_state->gfx, 319, 3, 178, 112, 3, 0x00, 0x01, 0x02, 1);

    // gfx_draw_line_anti_aliasing(global_state->gfx, 150, 112, 150, 120, 3, 0x00, 0x01, 0x02, 1);
    // gfx_draw_line_anti_aliasing(global_state->gfx, 178, 112, 178, 120, 3, 0x00, 0x01, 0x02, 1);

    // gfx_draw_line_anti_aliasing(global_state->gfx, 150, 120, 0, 233, 3, 0x00, 0x01, 0x02, 1);
    // gfx_draw_line_anti_aliasing(global_state->gfx, 178, 120, 319, 227, 3, 0x00, 0x01, 0x02, 1);

    // 进入沙漏画面若干秒内显示提示文字
    if (global_state->timestamp - s_ui_flip_first_load_timestamp < 10000) {
        gfx_draw_textline(global_state->gfx, L"节流", 0, 0, 59, 59, 59, 1);
        gfx_draw_textline(global_state->gfx, L"退出", 320-24-2, 0, 59, 59, 59, 1);
        gfx_draw_textline(global_state->gfx, L"画风", 0, 240-12, 59, 59, 59, 1);
        gfx_draw_textline(global_state->gfx, L"复位", 320-24-2, 240-12, 59, 59, 59, 1);
        gfx_draw_textline(global_state->gfx, L"调整节流度", 320-12*5-2, 120+36, 59, 59, 59, 1);
    }


    // 判断沙漏重置事件
    if ((lower_count < 1 && gravity_y < 0) || (upper_count < 1 && gravity_y > 0)) {
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

    // 计时
    uint64_t current_timestamp = global_state->timestamp;
    if (!s_ui_fanqie_is_running) {
        current_timestamp = s_ui_fanqie_stop_timestamp;
    }
    gfx_draw_textline(global_state->gfx, L"计时", 10, 102 - 20, 128, 128, 128, 1);
    wchar_t time7seg_str[10];
    wchar_t ms_str[5];
    int32_t countdown = (int32_t)((current_timestamp - s_ui_fanqie_start_timestamp) / 1000);
    int32_t ms = (int32_t)((current_timestamp - s_ui_fanqie_start_timestamp) % 1000) / 10;
    swprintf(time7seg_str, 10, L"%02d:%02d", countdown / 60, countdown % 60);
    swprintf(ms_str, 5, L".%02d", ms);
    int32_t s7seg_width = 0.0f;
    int32_t s7seg_height = 0.0f;
    ui_draw_7seg_string(key_event, global_state,
        10, 102,
        time7seg_str, 255, 255, 255, 10.0f, 3.0f, 7.0f, 0, &s7seg_width, &s7seg_height);
    gfx_draw_textline(global_state->gfx, ms_str, 8 + s7seg_width, 102 + s7seg_height/2 - 6 + 4, 255, 255, 255, 1);

    // FPS
    wchar_t fps_buf[20];
    swprintf(fps_buf, 20, L"FPS=%d", fps);
    gfx_draw_textline(global_state->gfx, fps_buf, 10, 102 + s7seg_height + 9, 128, 128, 128, 1);

    // 每1000ms统计一次粒子流量
    static float particle_flow_per_sec = 0.0f;
    if (s_ui_flip_last_upper_count == 0 || global_state->timestamp - s_ui_flip_last_upper_count_timestamp >= 1000) {
        particle_flow_per_sec = (float)(upper_count - s_ui_flip_last_upper_count) / (float)(global_state->timestamp - s_ui_flip_last_upper_count_timestamp) * 1000;
        s_ui_flip_last_upper_count = upper_count;
        s_ui_flip_last_upper_count_timestamp = global_state->timestamp;
    }
    wchar_t flow_str[30];
    swprintf(flow_str, 30, L"流量 %.1f /s", fabs(particle_flow_per_sec));

    // 根据重力方向计算沙漏进度
    float hourglass_progress = (float)((gravity_y <= 0) ? lower_count : upper_count) / (float)(upper_count + lower_count);
    wchar_t count_str[30];
    swprintf(count_str, 30, L"%03d/%03d", upper_count, lower_count);
    wchar_t percent_str[10];
    swprintf(percent_str, 10, L"%d", (int32_t)floor(hourglass_progress * 100.0f));
    wchar_t percent_decimal_str[10];
    swprintf(percent_decimal_str, 10, L".%d%%", (int32_t)floor(hourglass_progress * 100.0f * 10.0f) % 10);

    // 第一次绘制是获取长宽，清除后再重新绘制
    ui_draw_7seg_string(key_event, global_state,
        210, 102,
        percent_str, 255, 255, 255, 10.0f, 3.0f, 7.0f, 0, &s7seg_width, &s7seg_height);
    gfx_draw_rectangle(global_state->gfx, 210, 102, 320-210, s7seg_height, 30, 31, 32, 1);
    ui_draw_7seg_string(key_event, global_state,
        320-10-6*3-s7seg_width, 102,
        percent_str, 255, 255, 255, 10.0f, 3.0f, 7.0f, 0, &s7seg_width, &s7seg_height);
    gfx_draw_textline(global_state->gfx, percent_decimal_str, 320-10-6*3, 102 + s7seg_height/2 - 6 + 4, 255, 255, 255, 1);
    gfx_draw_textline(global_state->gfx, count_str, 190, 102 + s7seg_height/2 - 6 + 4, 64, 64, 64, 1);
    gfx_draw_textline(global_state->gfx, flow_str, 230, 102 - 20, 128, 128, 128, 1);


    wchar_t throttle_str[10];
    swprintf(throttle_str, 10, L"节流度 %d%%", (s_ui_flip_is_throttle) ? s_ui_flip_throttle : 0);
    gfx_draw_textline(global_state->gfx, throttle_str, 250, 102 + s7seg_height + 9, 128, 128, 128, 1);

    // 根据沙漏进度调整节流度，避免来自上方的压力过小时，出现几乎不往下流的问题
    s_ui_flip_throttle = roundf((1.0f - (float)s_ui_flip_init_throttle) * hourglass_progress * hourglass_progress + (float)s_ui_flip_init_throttle);

    // 到时提醒（非阻塞状态机：跨帧推进震动节奏，不阻塞渲染任务）。
    // 历史教训（2026-08-01 排障）：旧实现每帧 sleep_in_ms(600)×2 阻塞 1.2s，
    // CoreS3 上 IMU 轴向差异致计时状态误判“流完”，按节流键即误触发本报警，
    // 造成约 7.2s（6 周期×1.2s）的 1fps 卡顿；故改为非阻塞并加运行时长门槛。
    if (!s_ui_fanqie_is_running && s_ui_flip_is_throttle && s_ui_fanqie_alarm_count < 6) {
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

    gfx_refresh(global_state->gfx);
}


void ui_app_flip_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 按*键切换显示方式
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
        if (s_ui_flip_setting_count == 0) {
            s_ui_flip_show_grid = 0;
            s_ui_flip_show_particles = 1;
        }
        else if (s_ui_flip_setting_count == 1) {
            s_ui_flip_show_grid = 1;
            s_ui_flip_show_particles = 0;
        }
        else if (s_ui_flip_setting_count == 2) {
            s_ui_flip_show_grid = 1;
            s_ui_flip_show_particles = 1;
        }
        s_ui_flip_setting_count++;
        s_ui_flip_setting_count = s_ui_flip_setting_count % 3;
    }
    // 按1键切换漏斗阻尼
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        if (s_ui_flip_is_throttle) {
            s_ui_flip_is_throttle = 0;
        }
        else {
            s_ui_flip_is_throttle = 1;
        }
    }
    // 按A键返回主菜单
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_MAIN_MENU;
    }
    // 按C键切换节流度
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_ctrl) {
        s_ui_flip_init_throttle += 10;
        if (s_ui_flip_init_throttle > 100) {
            s_ui_flip_init_throttle = 10;
        }
    }
    // 按D键复位
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_enter) {
        ui_app_flip_init(key_event, global_state);
    }
}


// ===============================================================================
// 玲珑天象仪
// ===============================================================================

static uint64_t linglong_first_call_timestamp = 0;
static uint32_t linglong_last_day = 0;
static uint32_t linglong_sunrise_time[2] = {0, 0}; // hour, minute
static uint32_t linglong_sunset_time[2] = {0, 0}; // hour, minute

static int32_t linglong_refreshed = 0; // 记录暂停状态下是否已刷新
static int32_t linglong_timemachine_running_state = 2; // 0-停止；1-时光机运行；2-实时
static int32_t linglong_timemachine_speed = 0; // 时光机速度，正数为未来，负数为过去，单位秒
static uint64_t linglong_timemachine_start_timestamp = 0;

static int32_t linglong_state = 0; // 玲珑仪UI状态

#define LL_STATE_SKY (0)
#define LL_STATE_SETTING (1)
#define LL_STATE_SETTING_CALLBACK (2)


void ui_app_linglong_init(Key_Event *key_event, Global_State *global_state) {
    linglong_refreshed = 0;
    linglong_state = LL_STATE_SKY;
}

void ui_app_linglong_setting_draw(Key_Event *key_event, Global_State *global_state) {
    uint8_t txt_color[4][4][3] = {
        {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},},
        {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},},
        {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},},
        {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},},
    };
    wchar_t cell_text[4][4][2][10] = {
        { {L"投影算法", L"鱼眼",}, {L"赤道坐标", L"关",}, {L"地平坐标", L"方位角",}, {L"退出玲珑仪", L"",}, },
        { {L"黄道", L"关",}, {L"天体名称", L"关",}, {L"姿态指示", L"关",}, {L"校准IMU", L"",}, },
        { {L"大气散射", L"二次散射",}, {L"地景", L"草原全景",}, {L"平滑滤波", L"开",}, {L"返回", L"",}, },
        { {L"时间", L"",}, {L"跟踪太阳", L"关",}, {L"位置", L"",}, {L"", L"",}, },
    };

    // (0,0)投影算法
    if (global_state->linglong_cfg->projection == 0) {
        wcscpy(cell_text[0][0][1], L"鱼眼");
        txt_color[0][0][0] = 0;
        txt_color[0][0][1] = 255;
        txt_color[0][0][2] = 255;
    }
    else {
        wcscpy(cell_text[0][0][1], L"透视");
        txt_color[0][0][0] = 0;
        txt_color[0][0][1] = 255;
        txt_color[0][0][2] = 255;
    }

    // (0,1)赤道坐标
    if (global_state->linglong_cfg->enable_equatorial_coord == 0) {
        wcscpy(cell_text[0][1][1], L"关");
        txt_color[0][1][0] = 222;
        txt_color[0][1][1] = 0;
        txt_color[0][1][2] = 0;
    }
    else {
        wcscpy(cell_text[0][1][1], L"开");
        txt_color[0][1][0] = 0;
        txt_color[0][1][1] = 255;
        txt_color[0][1][2] = 0;
    }

    // (0,2)地平坐标
    switch (global_state->linglong_cfg->enable_horizontal_coord) {
        case 0:
            wcscpy(cell_text[0][2][1], L"关");
            txt_color[0][2][0] = 222;
            txt_color[0][2][1] = 0;
            txt_color[0][2][2] = 0;
            break;
        case 1:
            wcscpy(cell_text[0][2][1], L"方位角");
            txt_color[0][2][0] = 0;
            txt_color[0][2][1] = 255;
            txt_color[0][2][2] = 255;
            break;
        case 2:
            wcscpy(cell_text[0][2][1], L"坐标圈");
            txt_color[0][2][0] = 0;
            txt_color[0][2][1] = 255;
            txt_color[0][2][2] = 255;
            break;
        default: break;
    }

    // (1,0)黄道
    if (global_state->linglong_cfg->enable_ecliptic_circle == 0) {
        wcscpy(cell_text[1][0][1], L"关");
        txt_color[1][0][0] = 222;
        txt_color[1][0][1] = 0;
        txt_color[1][0][2] = 0;
    }
    else {
        wcscpy(cell_text[1][0][1], L"开");
        txt_color[1][0][0] = 0;
        txt_color[1][0][1] = 255;
        txt_color[1][0][2] = 0;
    }

    // (1,1)地平坐标
    switch (global_state->linglong_cfg->enable_star_name) {
        case 0:
            wcscpy(cell_text[1][1][1], L"关");
            txt_color[1][1][0] = 222;
            txt_color[1][1][1] = 0;
            txt_color[1][1][2] = 0;
            break;
        case 1:
            wcscpy(cell_text[1][1][1], L"仅恒星");
            txt_color[1][1][0] = 0;
            txt_color[1][1][1] = 255;
            txt_color[1][1][2] = 255;
            break;
        case 2:
            wcscpy(cell_text[1][1][1], L"仅行星");
            txt_color[1][1][0] = 0;
            txt_color[1][1][1] = 255;
            txt_color[1][1][2] = 255;
            break;
        case 3:
            wcscpy(cell_text[1][1][1], L"全部显示");
            txt_color[1][1][0] = 0;
            txt_color[1][1][1] = 255;
            txt_color[1][1][2] = 255;
            break;
        default: break;
    }

    // (1,2)姿态指示
    if (global_state->linglong_cfg->enable_att_indicator == 0) {
        wcscpy(cell_text[1][2][1], L"关");
        txt_color[1][2][0] = 222;
        txt_color[1][2][1] = 0;
        txt_color[1][2][2] = 0;
    }
    else {
        wcscpy(cell_text[1][2][1], L"开");
        txt_color[1][2][0] = 0;
        txt_color[1][2][1] = 255;
        txt_color[1][2][2] = 0;
    }

    // (2,0)大气散射
    switch (global_state->linglong_cfg->sky_model) {
        case 0:
            wcscpy(cell_text[2][0][1], L"关");
            txt_color[2][0][0] = 222;
            txt_color[2][0][1] = 0;
            txt_color[2][0][2] = 0;
            break;
        case 1:
            wcscpy(cell_text[2][0][1], L"简化模型");
            txt_color[2][0][0] = 0;
            txt_color[2][0][1] = 255;
            txt_color[2][0][2] = 255;
            break;
        case 2:
            wcscpy(cell_text[2][0][1], L"一次散射");
            txt_color[2][0][0] = 0;
            txt_color[2][0][1] = 255;
            txt_color[2][0][2] = 255;
            break;
        case 3:
            wcscpy(cell_text[2][0][1], L"二次散射");
            txt_color[2][0][0] = 0;
            txt_color[2][0][1] = 255;
            txt_color[2][0][2] = 255;
            break;
        case 4:
            wcscpy(cell_text[2][0][1], L"体积云");
            txt_color[2][0][0] = 255;
            txt_color[2][0][1] = 170;
            txt_color[2][0][2] = 0;
            break;
        default: break;
    }

    // (2,1)地景
    switch (global_state->linglong_cfg->landscape_index) {
        case 0:
            wcscpy(cell_text[2][1][1], L"关");
            txt_color[2][1][0] = 222;
            txt_color[2][1][1] = 0;
            txt_color[2][1][2] = 0;
            break;
        case 1:
            wcscpy(cell_text[2][1][1], L"草原");
            txt_color[2][1][0] = 0;
            txt_color[2][1][1] = 255;
            txt_color[2][1][2] = 255;
            break;
        case 2:
            wcscpy(cell_text[2][1][1], L"卫星照片");
            txt_color[2][1][0] = 0;
            txt_color[2][1][1] = 255;
            txt_color[2][1][2] = 255;
            break;
        default: break;
    }

    // (2,2)平滑滤波
    if (global_state->linglong_cfg->enable_opt_bilinear == 0) {
        wcscpy(cell_text[2][2][1], L"关");
        txt_color[2][2][0] = 222;
        txt_color[2][2][1] = 0;
        txt_color[2][2][2] = 0;
    }
    else {
        wcscpy(cell_text[2][2][1], L"开");
        txt_color[2][2][0] = 0;
        txt_color[2][2][1] = 255;
        txt_color[2][2][2] = 0;
    }

    // (3,1)跟踪太阳
    if (global_state->linglong_cfg->enable_tracking_sun == 0) {
        wcscpy(cell_text[3][1][1], L"关");
        txt_color[3][1][0] = 222;
        txt_color[3][1][1] = 0;
        txt_color[3][1][2] = 0;
    }
    else {
        wcscpy(cell_text[3][1][1], L"开");
        txt_color[3][1][0] = 0;
        txt_color[3][1][1] = 255;
        txt_color[3][1][2] = 0;
    }


    gfx_soft_clear(global_state->llgfx);

    // 网格布局：上下留白与顶/底栏高度一致（当前 UI 字体行高 + 1）
    const int32_t bar_height = gfx_font_line_height(global_state->ui_font) + 1;
    UI_Grid_Layout grid = ui_grid_layout_make(global_state->llgfx->width, global_state->llgfx->height, bar_height, bar_height, 0, 0, 4, 4);

    for (int32_t row = 0; row < 4; row++) {
        for (int32_t col = 0; col < 4; col++) {
            int32_t bx = (col == 0) ? 1 : 0;
            int32_t by = (row == 0) ? 1 : 0;
            gfx_draw_rectangle(global_state->llgfx, ui_grid_cell_x0(&grid,col)+bx, ui_grid_cell_y0(&grid,row)+by, ui_grid_cell_width(&grid)-1-bx, ui_grid_cell_height(&grid)-1-by, 37, 38, 41, 1);
            gfx_draw_textline_centered(global_state->llgfx, cell_text[row][col][0], ui_grid_cell_center_x(&grid,col), ui_grid_cell_center_y(&grid,row)-8, 255, 255, 255, 1);
            gfx_draw_textline_centered(global_state->llgfx, cell_text[row][col][1], ui_grid_cell_center_x(&grid,col), ui_grid_cell_center_y(&grid,row)+10, txt_color[row][col][0], txt_color[row][col][1], txt_color[row][col][2], 1);
        }
    }

    gfx_draw_textline_centered(global_state->llgfx, L"玲珑天象仪设置", global_state->llgfx->width/2, grid.padding_top/2, 222, 222, 222, 1);
}





void ui_app_linglong_draw_full(Key_Event *key_event, Global_State *global_state) {

    Linglong_Config *llcfg = global_state->linglong_cfg;

    // FPS统计
    static uint64_t fps_last_timestamp = 0;
    static uint32_t fps_frame_count = 0;
    static float fps_display_value = 0.0f;

    fps_frame_count++;
    if (fps_last_timestamp == 0) {
        fps_last_timestamp = global_state->timestamp;
    }
    else if (global_state->timestamp - fps_last_timestamp >= 1000) {
        fps_display_value = fps_frame_count * 1000.0f / (float)(global_state->timestamp - fps_last_timestamp);
        fps_frame_count = 0;
        fps_last_timestamp = global_state->timestamp;
    }

    time_t ts = (time_t)(global_state->timestamp / 1000);

    if (linglong_timemachine_running_state == 0) {
        if (linglong_refreshed && (!(llcfg->enable_imu))) {
            return;
        }
    }
    else if (linglong_timemachine_running_state == 1) {
        linglong_timemachine_start_timestamp += (linglong_timemachine_speed * 1000);
        ts = (time_t)(linglong_timemachine_start_timestamp / 1000);
        struct tm *timeinfo = localtime(&ts); // 转换为本地时间

        llcfg->second = timeinfo->tm_sec;
        llcfg->minute = timeinfo->tm_min;
        llcfg->hour = timeinfo->tm_hour;
        llcfg->day = timeinfo->tm_mday;
        llcfg->month = timeinfo->tm_mon + 1;
        llcfg->year = timeinfo->tm_year + 1900;
    }
    else if (linglong_timemachine_running_state == 2) {
        ts = (time_t)(global_state->timestamp / 1000);
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

    gfx_soft_clear(global_state->llgfx);

    render_sky(global_state->llgfx,
        MIN(global_state->llgfx->width, global_state->llgfx->height) / 2, global_state->llgfx->width / 2, global_state->llgfx->height / 2,
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
        llcfg->enable_att_indicator,
        llcfg->enable_tracking_sun,
        llcfg->cloud_coverage_level,
        llcfg->cloud_layer_mask,
        llcfg->cloud_brightness
    );

    gfx_dithering(global_state->llgfx);
    // gfx_gamma(global_state->llgfx, 1.3f);

    // 显示FPS
    wchar_t fps_str[16];
    swprintf(fps_str, 16, L"FPS=%.1f", fps_display_value);
    gfx_draw_textline(global_state->llgfx, fps_str, 1, 0, 0, 255, 0, 1);

    linglong_refreshed = 1;
}

void ui_app_linglong_draw_lite(
    Key_Event *key_event, Global_State *global_state,
    int32_t x, int32_t y,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second,
    double longitude, double latitude, double timezone
) {
    uint8_t BG_R = 255, BG_G = 255, BG_B = 255;
    uint8_t COORD_R = 222, COORD_G = 222, COORD_B = 222;
    uint8_t NSWE_R = 255, NSWE_G = 0, NSWE_B = 0;
    uint8_t DATETIME_R = 0, DATETIME_G = 0, DATETIME_B = 255;
    uint8_t TEXT_R = 0, TEXT_G = 0, TEXT_B = 0;
    uint8_t SUN_R = 255, SUN_G = 0, SUN_B = 0;
    uint8_t MOON_R = 255, MOON_G = 0, MOON_B = 255;

    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        // 同初始化
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        BG_R = 0, BG_G = 0, BG_B = 0;
        COORD_R = 66, COORD_G = 66, COORD_B = 66;
        NSWE_R = 255, NSWE_G = 0, NSWE_B = 0;
        DATETIME_R = 0, DATETIME_G = 255, DATETIME_B = 255;
        TEXT_R = 255, TEXT_G = 255, TEXT_B = 255;
        SUN_R = 255, SUN_G = 255, SUN_B = 0;
        MOON_R = 222, MOON_G = 222, MOON_B = 0;
    }

    gfx_draw_rectangle(global_state->gfx, x, y, 128, 64, BG_R, BG_G, BG_B, 1);

    gfx_draw_circle(global_state->gfx, x+64, y+32, 30,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_circle(global_state->gfx, x+64, y+32, 20,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_circle(global_state->gfx, x+64, y+32, 10,         COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_line(global_state->gfx,   x+32, y+32, x+96, y+32, COORD_R, COORD_G, COORD_B, 1);
    gfx_draw_line(global_state->gfx,   x+64, y+0, x+64, y+64,  COORD_R, COORD_G, COORD_B, 1);

    gfx_draw_rectangle(global_state->gfx, x+62-1, y+0,    5+2, 5+1, BG_R, BG_G, BG_B, 1); // N背景
    gfx_draw_rectangle(global_state->gfx, x+63-1, y+59-1, 3+2, 5+1, BG_R, BG_G, BG_B, 1); // S背景
    gfx_draw_rectangle(global_state->gfx, x+32-1, y+30-1, 5+2, 5+2, BG_R, BG_G, BG_B, 1); // W背景
    gfx_draw_rectangle(global_state->gfx, x+93-1, y+30-1, 3+2, 5+2, BG_R, BG_G, BG_B, 1); // E背景

    // 方位文字和周围的边框
    gfx_draw_textline_mini(global_state->gfx, L"N", x+62, y+0,  NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+61, y+2,  BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+67, y+2,  BG_R, BG_G, BG_B, 1);  gfx_draw_point(global_state->gfx, x+64, y+5, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"S", x+63, y+59, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+62, y+62, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+66, y+62, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+64, y+58, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"W", x+32, y+30, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+34, y+29, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+37, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+34, y+35, BG_R, BG_G, BG_B, 1);
    gfx_draw_textline_mini(global_state->gfx, L"E", x+93, y+30, NSWE_R, NSWE_G, NSWE_B, 1); gfx_draw_point(global_state->gfx, x+92, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+96, y+32, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+94, y+29, BG_R, BG_G, BG_B, 1); gfx_draw_point(global_state->gfx, x+94, y+35, BG_R, BG_G, BG_B, 1);

    gfx_draw_line(global_state->gfx, x+0, y+43, x+30, y+43, COORD_R, COORD_G, COORD_B, 1);

    wchar_t timestr[30];
    swprintf(timestr, 30, L"%04d-%02d-%02d\n%02d:%02d:%02d", year, month, day, hour, minute, second);
    gfx_draw_textline_mini(global_state->gfx, timestr, x+0, y+0, DATETIME_R, DATETIME_G, DATETIME_B, 1);

    double altitude_moon = 0.0;
    double azimuth_moon = 0.0;

    where_is_the_moon(year, month, day, hour, minute, second, timezone, longitude, latitude, &azimuth_moon, &altitude_moon);
    double i_deg = moon_phase(year, month, day, hour, minute, second, timezone);
    double moon_k = (1.0 + cos(i_deg / 180.0 * M_PI)) / 2.0;

    wchar_t coordstr_moon[30];
    swprintf(coordstr_moon, 30, L"MOON\nP:%d%%\nA:%.1f\nE:%.1f", (int32_t)(moon_k * 100.0), azimuth_moon, altitude_moon);
    gfx_draw_textline_mini(global_state->gfx, coordstr_moon, x+0, y+18, TEXT_R, TEXT_G, TEXT_B, 1);

    double x_moon = 64 + (90.0 - altitude_moon) * 32.0 / 90.0 * sin(azimuth_moon / 180.0 * M_PI);
    double y_moon = 32 - (90.0 - altitude_moon) * 32.0 / 90.0 * cos(azimuth_moon / 180.0 * M_PI);

    if (x_moon >= 32 && x_moon <= 96 && y_moon >= 0 && y_moon <= 64) {
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 1, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon - 0, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon - 1, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon - 0, MOON_R, MOON_G, MOON_B, 1);
        gfx_draw_point(global_state->gfx, x + (int)x_moon + 1, y + (int)y_moon + 1, MOON_R, MOON_G, MOON_B, 1);
    }

    double altitude_sun = 0.0;
    double azimuth_sun = 0.0;

    where_is_the_sun(year, month, day, hour, minute, second, +8.0, longitude, latitude, &azimuth_sun, &altitude_sun);

    wchar_t coordstr_sun[30];
    swprintf(coordstr_sun, 30, L"SUN\nA:%.1f\nE:%.1f", azimuth_sun, altitude_sun);
    gfx_draw_textline_mini(global_state->gfx, coordstr_sun, x+0, y+46, TEXT_R, TEXT_G, TEXT_B, 1);

    double x_sun = 64 + (90.0 - altitude_sun) * 32.0 / 90.0 * sin(azimuth_sun / 180.0 * M_PI);
    double y_sun = 32 - (90.0 - altitude_sun) * 32.0 / 90.0 * cos(azimuth_sun / 180.0 * M_PI);

    if (x_sun >= 32 && x_sun <= 96 && y_sun >= 0 && y_sun <= 64) {
        gfx_draw_circle(global_state->gfx, x+(int)x_sun, y+(int)y_sun, 2, SUN_R, SUN_G, SUN_B, 1);
    }


    // 二分搜索日出日落时间
    if (linglong_first_call_timestamp == 0 || linglong_last_day != day) { // 只在首次调用和当天日期变化时计算
        linglong_first_call_timestamp = global_state->timestamp;
        linglong_last_day = day;

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
    gfx_draw_textline_mini(global_state->gfx, risefall_time, x+98, y+0, TEXT_R, TEXT_G, TEXT_B, 1);

    gfx_draw_textline_mini(global_state->gfx, L"    BD4SUR\n 2011-2026", x+86, y+53, TEXT_R, TEXT_G, TEXT_B, 1);
}

void ui_app_linglong_splash(Key_Event *key_event, Global_State *global_state) {
    Nano_GFX *gfx = global_state->gfx;
    gfx_soft_clear(gfx);
    gfx_draw_textline_centered(gfx, L"玲珑天象仪 V" NANO_VERSION, gfx->width/2, gfx->height/2 - 14 * 6, 0, 255, 255, 1);
    gfx_draw_textline_centered(gfx, L"Der bestirnte Himmel ueber mir.", gfx->width/2, gfx->height/2 - 14 * 5, 222, 222, 230, 1);
    gfx_draw_textline_centered(gfx, L"(c) 2011-2026 BD4SUR", gfx->width/2, gfx->height/2 - 14 * 4, 96, 96, 96, 1);
    gfx_draw_textline_centered(gfx, L"正在渲染首帧...请稍等", gfx->width/2, gfx->height/2 - 14 * 1, 255, 255, 255, 1);
    gfx_draw_textline_centered(gfx, L"1左转   2推杆   3右转   A退出", gfx->width/2, gfx->height/2 + 14 * 3, 96, 96, 96, 1);
    gfx_draw_textline_centered(gfx, L"4左倾   5归中   6右倾   B    ", gfx->width/2, gfx->height/2 + 14 * 4, 96, 96, 96, 1);
    gfx_draw_textline_centered(gfx, L"7拉远   8拉杆   9推进   C设置", gfx->width/2, gfx->height/2 + 14 * 5, 96, 96, 96, 1);
    gfx_draw_textline_centered(gfx, L"*快退   0实时   #快进   D    ", gfx->width/2, gfx->height/2 + 14 * 6, 96, 96, 96, 1);

    gfx_refresh(gfx);
}

void ui_app_linglong_render_frame(Key_Event *key_event, Global_State *global_state) {
    // ui_app_linglong_draw_full(key_event, global_state);

    if (linglong_state == LL_STATE_SETTING || linglong_state == LL_STATE_SETTING_CALLBACK) {
        ui_app_linglong_setting_draw(key_event, global_state);
    }
    else {
        ui_app_linglong_draw_full(key_event, global_state);
    }

    gfx_draw_textline(global_state->llgfx, L"玲珑天象仪 V" NANO_VERSION, 1, global_state->llgfx->height - 13, 255, 255, 255, 200);

    wchar_t timestr[30];
    swprintf(timestr, 30, L"%ls %04d-%02d-%02d %02d:%02d:%02d",
        (linglong_timemachine_running_state == 0) ? L"  " :
            (((linglong_timemachine_running_state == 1) && (linglong_timemachine_speed > 0)) ? L">>" :
            (((linglong_timemachine_running_state == 1) && (linglong_timemachine_speed < 0)) ? L"<<" : L" >")),
        global_state->linglong_cfg->year, global_state->linglong_cfg->month, global_state->linglong_cfg->day, global_state->linglong_cfg->hour, global_state->linglong_cfg->minute, global_state->linglong_cfg->second);
    gfx_draw_textline(global_state->llgfx, timestr, global_state->llgfx->width - 134, global_state->llgfx->height - 13, 255, 255, 255, 1);

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    convert_rgb888_to_rgb565_double(global_state->gfx, global_state->llgfx->frame_buffer_rgb888, global_state->llgfx->width, global_state->llgfx->height);
    gfx_refresh(global_state->gfx);
#else
    // convert_rgb888_to_rgb565_double(global_state->gfx, global_state->llgfx->frame_buffer_rgb888, global_state->llgfx->width, global_state->llgfx->height);
    gfx_refresh(global_state->llgfx);
#endif

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
    switch (linglong_timemachine_running_state) {
        case 0: linglong_timemachine_running_state = 1; break;
        case 1: linglong_timemachine_running_state = 0; break;
        case 2: linglong_timemachine_running_state = 1; break;
        default: linglong_timemachine_running_state = 0; break;
    }
    if (linglong_timemachine_start_timestamp == 0) {
        linglong_timemachine_start_timestamp = global_state->timestamp;
    }
}

void ui_app_linglong_set_realtime(Key_Event *key_event, Global_State *global_state) {
    Linglong_Config *llcfg = global_state->linglong_cfg;
    time_t ts = (time_t)(global_state->timestamp / 1000);
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

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)

#else
    if (global_state->linglong_cfg->enable_imu) {
        imu_read_angle(&(global_state->pitch), &(global_state->roll), &(global_state->yaw));
        printf("俯仰=%-10.2f    滚转=%-10.2f    航向=%-10.2f\n", global_state->pitch, global_state->roll, global_state->yaw);
    }
#endif

#endif

    int32_t is_setting_refresh = 0;

    // 按任意键都重置玲珑仪刷新状态，以便响应按键活动
    if (key_event->key_edge < 0 && key_event->key_code != NANO_KEY_IDLE) {
        linglong_refreshed = 0;
    }

    // 按1键向左偏航（yaw--），或者Ctrl时切换投影算法
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_1) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_azi -= 5.0f;
            if (global_state->linglong_cfg->view_azi <= 0.0f) {
                global_state->linglong_cfg->view_azi = 360.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            if (global_state->linglong_cfg->projection == 0) {
                global_state->linglong_cfg->projection = 1;
            }
            else {
                global_state->linglong_cfg->projection = 0;
            }
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按2键推杆低头（pitch--），或者Ctrl时切换赤道坐标圈
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_2) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_alt -= 5.0f;
            if (global_state->linglong_cfg->view_alt <= -90.0f) {
                global_state->linglong_cfg->view_alt = -90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_equatorial_coord ++;
            global_state->linglong_cfg->enable_equatorial_coord = global_state->linglong_cfg->enable_equatorial_coord % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按3键向右偏航（yaw++），或者Ctrl时切换地平坐标
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_3) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_azi += 5.0f;
            if (global_state->linglong_cfg->view_azi >= 360.0f) {
                global_state->linglong_cfg->view_azi = 0.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_horizontal_coord++;
            global_state->linglong_cfg->enable_horizontal_coord = global_state->linglong_cfg->enable_horizontal_coord % 3;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按4键向左坡度（roll--），或者Ctrl时切换黄道
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_4) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_roll -= 5.0f;
            if (global_state->linglong_cfg->view_roll <= -90.0f) {
                global_state->linglong_cfg->view_roll = -90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_ecliptic_circle++;
            global_state->linglong_cfg->enable_ecliptic_circle = global_state->linglong_cfg->enable_ecliptic_circle % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按5键归中，或切换陀螺仪状态，或者Ctrl时切换天体名称
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_5) {
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
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_star_name++;
            global_state->linglong_cfg->enable_star_name = global_state->linglong_cfg->enable_star_name % 4;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按6键向右坡度（roll++），或者Ctrl时切换姿态指示
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_6) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_roll += 5.0f;
            if (global_state->linglong_cfg->view_roll >= 90.0f) {
                global_state->linglong_cfg->view_roll = 90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_att_indicator++;
            global_state->linglong_cfg->enable_att_indicator = global_state->linglong_cfg->enable_att_indicator % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按7键拉远，或者Ctrl时切换大气散射模型
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_7) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->view_f -= 0.1f;
            if (global_state->linglong_cfg->view_f <= 0.1f) global_state->linglong_cfg->view_f = 0.1f;
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->sky_model++;
            global_state->linglong_cfg->sky_model = global_state->linglong_cfg->sky_model % 5;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按8键拉杆抬头（pitch++），或者Ctrl时切换地景
    if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_8) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->enable_imu = 0; // 手动控制，关闭IMU
            global_state->linglong_cfg->view_alt += 5.0f;
            if (global_state->linglong_cfg->view_alt >= 90.0f) {
                global_state->linglong_cfg->view_alt = 90.0f;
            }
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->landscape_index++;
            global_state->linglong_cfg->landscape_index = global_state->linglong_cfg->landscape_index % 3;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按9键推近，或者Ctrl时切换平滑滤波
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_9) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->linglong_cfg->view_f += 0.1f;
            if (global_state->linglong_cfg->view_f >= 5.0f) global_state->linglong_cfg->view_f = 5.0f;
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_opt_bilinear++;
            global_state->linglong_cfg->enable_opt_bilinear = global_state->linglong_cfg->enable_opt_bilinear % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按A键返回主菜单
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->is_ctrl_enabled = 0;
        global_state->STATE = STATE_MAIN_MENU;
    }
    // 按B键+Ctrl校准IMU
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_shift) {
        if (global_state->is_ctrl_enabled == 0) {
            // TODO
        }
        else {
            // global_state->is_ctrl_enabled = 0;
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
    // 按C键切换Ctrl
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_ctrl) {
        if (global_state->is_ctrl_enabled == 0) {
            global_state->is_ctrl_enabled = 1;
            linglong_state = LL_STATE_SETTING;
        }
        else {
            global_state->is_ctrl_enabled = 0;
            linglong_state = LL_STATE_SKY;
        }
    }
    // 按*键时光机向前（过去）（反复按运行/暂停）
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_left) {
        ui_app_linglong_set_timemachine_speed(key_event, global_state, -120);
    }
    // 按0键回到实时（反复按运行/暂停），或者Ctrl时切换跟踪太阳
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_0) {
        if (global_state->is_ctrl_enabled == 0) {
            ui_app_linglong_set_realtime(key_event, global_state);
        }
        else {
            // global_state->is_ctrl_enabled = 0;
            global_state->linglong_cfg->enable_tracking_sun++;
            global_state->linglong_cfg->enable_tracking_sun = global_state->linglong_cfg->enable_tracking_sun % 2;
            linglong_state = LL_STATE_SETTING_CALLBACK;
        }
    }
    // 按#键时光机向后（未来）（反复按运行/暂停）
    else if (key_event->key_edge < 0 && key_event->key_code == NANO_KEY_right) {
        ui_app_linglong_set_timemachine_speed(key_event, global_state, 120);
    }
}




// ===============================================================================
// 设置菜单
// ===============================================================================

static int32_t year_edit = 0;
static int32_t month_edit = 0;
static int32_t day_edit = 0;
static int32_t hour_edit = 0;
static int32_t minute_edit = 0;
static float timezone_edit = 0;
static float longitude_edit = 0;
static float latitude_edit = 0;
static int32_t cursor_pos = 0;
static int32_t value_type = 0;
static wchar_t value_text[32] = L"00000000000";

// 将各类值转成可编辑的字符串
static void ui_app_setting_value_to_string(
    Key_Event *key_event, Global_State *global_state, wchar_t *value_text, int32_t value_type,
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, float timezone, float longitude, float latitude
) {
    // 日期 yyyy-mm-dd
    if (value_type == 0) {
        value_text[0] = (wchar_t)get_digit(year, 3);
        value_text[1] = (wchar_t)get_digit(year, 2);
        value_text[2] = (wchar_t)get_digit(year, 1);
        value_text[3] = (wchar_t)get_digit(year, 0);
        value_text[4] = L'-';
        value_text[5] = (wchar_t)get_digit(month, 1);
        value_text[6] = (wchar_t)get_digit(month, 0);
        value_text[7] = L'-';
        value_text[8] = (wchar_t)get_digit(day, 1);
        value_text[9] = (wchar_t)get_digit(day, 0);
        value_text[10] = 0;
    }
    // 时间时区 hh:mmsaabb
    else if (value_type == 1) {
        value_text[0] = (wchar_t)get_digit(hour, 1);
        value_text[1] = (wchar_t)get_digit(hour, 0);
        value_text[2] = L':';
        value_text[3] = (wchar_t)get_digit(minute, 1);
        value_text[4] = (wchar_t)get_digit(minute, 0);
        value_text[5] = (wchar_t)get_timezone_digit(timezone, 0);
        value_text[6] = (wchar_t)get_timezone_digit(timezone, 1);
        value_text[7] = (wchar_t)get_timezone_digit(timezone, 2);
        value_text[8] = (wchar_t)get_timezone_digit(timezone, 3);
        value_text[9] = (wchar_t)get_timezone_digit(timezone, 4);
        value_text[10] = 0;
    }
    // 经度 sddd_mm'ss"
    else if (value_type == 2) {
        value_text[0] = (wchar_t)get_lon_lat_digit(longitude, 0);
        value_text[1] = (wchar_t)get_lon_lat_digit(longitude, 1);
        value_text[2] = (wchar_t)get_lon_lat_digit(longitude, 2);
        value_text[3] = (wchar_t)get_lon_lat_digit(longitude, 3);
        value_text[4] = L' ';
        value_text[5] = (wchar_t)get_lon_lat_digit(longitude, 4);
        value_text[6] = (wchar_t)get_lon_lat_digit(longitude, 5);
        value_text[7] = L'\'';
        value_text[8] = (wchar_t)get_lon_lat_digit(longitude, 6);
        value_text[9] = (wchar_t)get_lon_lat_digit(longitude, 7);
        value_text[10] = L'"';
        value_text[11] = 0;
    }
    // 纬度 sdd_mm'ss"
    else if (value_type == 3) {
        value_text[0] = (wchar_t)get_lon_lat_digit(latitude, 0);
        value_text[1] = (wchar_t)get_lon_lat_digit(latitude, 2);
        value_text[2] = (wchar_t)get_lon_lat_digit(latitude, 3);
        value_text[3] = L' ';
        value_text[4] = (wchar_t)get_lon_lat_digit(latitude, 4);
        value_text[5] = (wchar_t)get_lon_lat_digit(latitude, 5);
        value_text[6] = L'\'';
        value_text[7] = (wchar_t)get_lon_lat_digit(latitude, 6);
        value_text[8] = (wchar_t)get_lon_lat_digit(latitude, 7);
        value_text[9] = L'"';
        value_text[10] = 0;
    }
}

static void ui_app_setting_grid16_refresh_button(
    Key_Event *key_event, Global_State *global_state, int32_t is_single_line,
    int32_t col, int32_t row, wchar_t *text0, wchar_t *text1,
    uint8_t cell_bg_R, uint8_t cell_bg_G, uint8_t cell_bg_B, uint8_t cell_bg_mode,
    uint8_t cell_text0_R, uint8_t cell_text0_G, uint8_t cell_text0_B, uint8_t cell_text0_mode,
    uint8_t cell_text1_R, uint8_t cell_text1_G, uint8_t cell_text1_B, uint8_t cell_text1_mode
) {
    int32_t bx = (col == 0) ? 1 : 0;
    int32_t by = (row == 0) ? 1 : 0;
    // 网格布局：上下留白与顶/底栏高度一致（当前 UI 字体行高 + 1）
    const int32_t bar_height = gfx_font_line_height(global_state->ui_font) + 1;
    UI_Grid_Layout grid = ui_grid_layout_make(global_state->gfx->width, global_state->gfx->height, bar_height, bar_height, 0, 0, 4, 4);
    gfx_draw_rectangle(global_state->gfx, ui_grid_cell_x0(&grid,col)+bx, ui_grid_cell_y0(&grid,row)+by, ui_grid_cell_width(&grid)-1-bx, ui_grid_cell_height(&grid)-1-by, cell_bg_R, cell_bg_G, cell_bg_B, cell_bg_mode);
    if (is_single_line) {
        gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_16, text0, ui_grid_cell_center_x(&grid,col), ui_grid_cell_center_y(&grid,row), cell_text0_R, cell_text0_G, cell_text0_B, cell_text0_mode);
    }
    else {
        gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, text0, ui_grid_cell_center_x(&grid,col), ui_grid_cell_center_y(&grid,row)-8, cell_text0_R, cell_text0_G, cell_text0_B, cell_text0_mode);
        gfx_font_draw_text_centered(global_state->gfx, GFX_FONT_ALPHA_12, text1, ui_grid_cell_center_x(&grid,col), ui_grid_cell_center_y(&grid,row)+10, cell_text1_R, cell_text1_G, cell_text1_B, cell_text1_mode);
    }
}

void ui_app_setting_grid16_draw(Key_Event *key_event, Global_State *global_state) {

    // 清屏
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    uint8_t cell_bg_R = 0, cell_bg_G = 0, cell_bg_B = 0;
    uint8_t cell_text0_R = 0, cell_text0_G = 0, cell_text0_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        cell_bg_R = 233;
        cell_bg_G = 239;
        cell_bg_B = 255;
        cell_text0_R = 0;
        cell_text0_G = 0;
        cell_text0_B = 0;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        cell_bg_R = 40;
        cell_bg_G = 40;
        cell_bg_B = 42;
        cell_text0_R = 255;
        cell_text0_G = 255;
        cell_text0_B = 255;
    }

    wchar_t date_str[32];
    ui_app_setting_value_to_string(
        key_event, global_state, date_str, 0,
        global_state->year, global_state->month, global_state->day,
        global_state->hour, global_state->minute, global_state->timezone,
        global_state->longitude, global_state->latitude);

    wchar_t time_str[32];
    ui_app_setting_value_to_string(
        key_event, global_state, time_str, 1,
        global_state->year, global_state->month, global_state->day,
        global_state->hour, global_state->minute, global_state->timezone,
        global_state->longitude, global_state->latitude);

    wchar_t longitude_str[32];
    ui_app_setting_value_to_string(
        key_event, global_state, longitude_str, 2,
        global_state->year, global_state->month, global_state->day,
        global_state->hour, global_state->minute, global_state->timezone,
        global_state->longitude, global_state->latitude);

    wchar_t latitude_str[32];
    ui_app_setting_value_to_string(
        key_event, global_state, latitude_str, 3,
        global_state->year, global_state->month, global_state->day,
        global_state->hour, global_state->minute, global_state->timezone,
        global_state->longitude, global_state->latitude);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        0, 0, L"日期", date_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        1, 0, L"时间", time_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);

    wchar_t brightness_str[32];
    swprintf(brightness_str, 32, L"%d%%", (global_state->brightness * 100) / 255);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        2, 0, L"屏幕亮度", brightness_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        3, 0, L"返回", NULL, cell_bg_R+10, cell_bg_G+10, cell_bg_B+10, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        0, 1, L"经度", longitude_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        1, 1, L"纬度", latitude_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);

    wchar_t volume_str[32];
    swprintf(volume_str, 32, L"%d%%", (global_state->volume * 100) / 255);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        2, 1, L"音量", volume_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        3, 1, L"IMU", L"开", cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0x00, 1);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        0, 2, L"LLM演示",
        (global_state->llm_enable_observation == 0) ? L"关闭" : L"开启",
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        1, 2, L"TTS设置",
        (global_state->tts_req_mode == 0) ? L"关闭" : ((global_state->tts_req_mode == 1) ? L"实时转换" : L"统一转换"),
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        2, 2, L"ASR设置",
        (global_state->is_auto_submit_after_asr == 0) ? L"编辑后提交" : L"立刻提交",
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    wchar_t auto_shutdown_str[32];
    if (global_state->auto_shutdown_minutes <= 0) {
        wcscpy(auto_shutdown_str, L"关");
    }
    else {
        swprintf(auto_shutdown_str, 32, L"%d分钟", (int)global_state->auto_shutdown_minutes);
    }

    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        3, 2, L"自动关机", auto_shutdown_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0x00, 0x00, 1);

    wchar_t ime_hint_str[32];
    if (global_state->ime_hint_timeout_s <= 0) {
        wcscpy(ime_hint_str, L"关闭");
    }
    else {
        swprintf(ime_hint_str, 32, L"%d秒", (int)global_state->ime_hint_timeout_s);
    }
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        0, 3, L"按键提示", ime_hint_str, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 0,
        1, 3, L"颜色风格", (global_state->ui_color_style == UI_COLOR_LIGHT) ? L"亮" : L"暗",
        cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0x00, 0xff, 0xff, 1);

    ui_draw_header(key_event, global_state, L"系统设置", 1);
    ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);
}

static inline void ui_app_setting_capture_value(Key_Event *key_event, Global_State *global_state) {
    year_edit = global_state->year;
    month_edit = global_state->month;
    day_edit = global_state->day;
    hour_edit = global_state->hour;
    minute_edit = global_state->minute;
    timezone_edit = global_state->timezone;
    longitude_edit = global_state->longitude;
    latitude_edit = global_state->latitude;
}

void ui_app_setting_grid16_event_handler(Key_Event *key_event, Global_State *global_state) {
    // 日期
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        value_type = 0;
        ui_app_setting_capture_value(key_event, global_state);
        ui_app_setting_value_to_string(
            key_event, global_state, value_text, value_type,
            year_edit, month_edit, day_edit, hour_edit, minute_edit, timezone_edit, longitude_edit, latitude_edit);
        global_state->STATE = STATE_SETTING_INPUT;
    }
    // 时间和时区
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_2) {
        value_type = 1;
        ui_app_setting_capture_value(key_event, global_state);
        ui_app_setting_value_to_string(
            key_event, global_state, value_text, value_type,
            year_edit, month_edit, day_edit, hour_edit, minute_edit, timezone_edit, longitude_edit, latitude_edit);
        global_state->STATE = STATE_SETTING_INPUT;
    }
    // 屏幕亮度
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_3) {
        global_state->brightness += 32;
        if (global_state->brightness == 256) {
            global_state->brightness = 255; // 最高亮度挡位
        }
        else if (global_state->brightness > 256) {
            global_state->brightness = 32; // 不允许亮度为0，回绕到最低挡位
        }
        gfx_set_brightness(global_state->gfx, global_state->brightness);
    }
    // 经度
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_4) {
        value_type = 2;
        ui_app_setting_capture_value(key_event, global_state);
        ui_app_setting_value_to_string(
            key_event, global_state, value_text, value_type,
            year_edit, month_edit, day_edit, hour_edit, minute_edit, timezone_edit, longitude_edit, latitude_edit);
        global_state->STATE = STATE_SETTING_INPUT;
    }
    // 纬度
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_5) {
        value_type = 3;
        ui_app_setting_capture_value(key_event, global_state);
        ui_app_setting_value_to_string(
            key_event, global_state, value_text, value_type,
            year_edit, month_edit, day_edit, hour_edit, minute_edit, timezone_edit, longitude_edit, latitude_edit);
        global_state->STATE = STATE_SETTING_INPUT;
    }
    // 音量（全局主音量：+16 步进；=256 钳位 255；>256 回绕 0 静音；影响按键音/寻呼机发射/音乐盒初始音量）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_6) {
        global_state->volume += 16;
        if (global_state->volume == 256) {
            global_state->volume = 255; // 最高音量挡位
        }
        else if (global_state->volume > 256) {
            global_state->volume = 0; // 回绕到静音
        }
        platform_set_master_volume((uint8_t)global_state->volume);
    }
    // LLM演示设置
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_7) {
        global_state->llm_enable_observation += 1;
        global_state->llm_enable_observation = global_state->llm_enable_observation % 2;
    }
    // TTS设置
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_8) {
        global_state->tts_req_mode += 1;
        global_state->tts_req_mode = global_state->tts_req_mode % 3;
    }
    // ASR设置
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_9) {
        global_state->is_auto_submit_after_asr += 1;
        global_state->is_auto_submit_after_asr = global_state->is_auto_submit_after_asr % 2;
    }
    // 颜色风格（全局UI色彩风格：亮/暗切换，默认暗；重绘由下方统一刷新块完成，
    // 不得在此绘制主菜单宫格——此前误调用 ui_widget_grid16_draw 导致点击后闪一下主菜单）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_0) {
        if (global_state->ui_color_style == UI_COLOR_LIGHT) {
            global_state->ui_color_style = UI_COLOR_DARK;
        }
        else {
            global_state->ui_color_style = UI_COLOR_LIGHT;
        }
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_SPLASH_SCREEN;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_shift) {
        // TODO
    }
    // 自动关机（关→1→2→3→5→10→20→30→60 分钟循环；设置即从当前时刻重新倒计时）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_ctrl) {
        static const int32_t AUTO_SHUTDOWN_OPTIONS[] = {0, 1, 2, 3, 5, 10, 20, 30, 60};
        int32_t idx = 0;
        for (int32_t i = 0; i < (int32_t)(sizeof(AUTO_SHUTDOWN_OPTIONS) / sizeof(AUTO_SHUTDOWN_OPTIONS[0])); i++) {
            if (AUTO_SHUTDOWN_OPTIONS[i] == global_state->auto_shutdown_minutes) { idx = i; break; }
        }
        idx = (idx + 1) % (int32_t)(sizeof(AUTO_SHUTDOWN_OPTIONS) / sizeof(AUTO_SHUTDOWN_OPTIONS[0]));
        global_state->auto_shutdown_minutes = AUTO_SHUTDOWN_OPTIONS[idx];
        global_state->auto_shutdown_deadline = (AUTO_SHUTDOWN_OPTIONS[idx] > 0)
            ? global_state->timestamp + (uint64_t)AUTO_SHUTDOWN_OPTIONS[idx] * 60000ULL : 0;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_enter) {
        // TODO
    }
    // 按键提示（文本输入控件按键提示遮罩显示时长：关闭(0秒)→3秒→6秒 循环）
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
        static const int32_t IME_HINT_TIMEOUT_OPTIONS[] = {0, 3, 6};
        int32_t idx = 0;
        for (int32_t i = 0; i < (int32_t)(sizeof(IME_HINT_TIMEOUT_OPTIONS) / sizeof(IME_HINT_TIMEOUT_OPTIONS[0])); i++) {
            if (IME_HINT_TIMEOUT_OPTIONS[i] == global_state->ime_hint_timeout_s) { idx = i; break; }
        }
        idx = (idx + 1) % (int32_t)(sizeof(IME_HINT_TIMEOUT_OPTIONS) / sizeof(IME_HINT_TIMEOUT_OPTIONS[0]));
        global_state->ime_hint_timeout_s = IME_HINT_TIMEOUT_OPTIONS[idx];
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
        // TODO
    }
    else {
        return;
    }

    // 有键按下则刷新
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code != NANO_KEY_IDLE) {
        ui_app_setting_grid16_draw(key_event, global_state);
        gfx_refresh(global_state->gfx);
    }
}

// 日期/时间/经度/纬度设置
// value_type: 0-日期 1-时间时区 2-经度 3-纬度
// cursor_pos: 光标相对于值字符串第一个字符的位置（不检测连字符等非值字符，位置由调用者处理），例如:
//   value_str   12:34+0800
//   cursor_pos  0123456789
void ui_app_setting_value_input_draw(Key_Event *key_event, Global_State *global_state, int32_t value_type, wchar_t *value_text, int32_t cursor_pos) {
    // 清屏
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        gfx_fill_white(global_state->gfx);
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        gfx_soft_clear(global_state->gfx);
    }

    uint8_t cell_bg_R = 0, cell_bg_G = 0, cell_bg_B = 0;
    uint8_t cell_text0_R = 0, cell_text0_G = 0, cell_text0_B = 0;
    if (global_state->ui_color_style == UI_COLOR_LIGHT) {
        cell_bg_R = 233;
        cell_bg_G = 239;
        cell_bg_B = 255;
        cell_text0_R = 0;
        cell_text0_G = 0;
        cell_text0_B = 0;
    }
    else if (global_state->ui_color_style == UI_COLOR_DARK) {
        cell_bg_R = 40;
        cell_bg_G = 40;
        cell_bg_B = 42;
        cell_text0_R = 255;
        cell_text0_G = 255;
        cell_text0_B = 255;
    }

    // 绘制顶栏（前缀）
    switch (value_type) {
        case 0: {
            ui_draw_header(key_event, global_state, L"", 0);
            gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, L"设置时间：", 0, 1, 255, 255, 255, 1);
            break;
        }
        case 1: {
            ui_draw_header(key_event, global_state, L"", 0);
            gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, L"设置日期：", 0, 1, 255, 255, 255, 1);
            break;
        }
        case 2: {
            ui_draw_header(key_event, global_state, L"", 0);
            gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, L"设置经度：", 0, 1, 255, 255, 255, 1);
            break;
        }
        case 3: {
            ui_draw_header(key_event, global_state, L"", 0);
            gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, L"设置纬度：", 0, 1, 255, 255, 255, 1);
            break;
        }
        default: return;
    }

    // 绘制设置值和光标
    int32_t x0 = 12 * 5; // 与顶栏前缀的长度有关
    int32_t x_cur = x0 + cursor_pos * 6;
    gfx_font_draw_text(global_state->gfx, GFX_FONT_BITMAP_12, value_text, x0, 1, 0x00, 0xff, 0xff, 1);
    gfx_draw_rectangle(global_state->gfx, x_cur, 12, 5, 2, 0x00, 0xff, 0xff, 1);

    // 绘制底栏
    ui_draw_footer(key_event, global_state, L"按数字键输入 光标自动右移", 1);

    // 绘制按键
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        0, 0, L"1", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        1, 0, L"2", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        2, 0, L"3", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        3, 0, L"取消", NULL, cell_bg_R+10, cell_bg_G, cell_bg_B, 1, 0xff, 0x00, 0x00, 1, 0x00, 0x00, 0x00, 1);

    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        0, 1, L"4", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        1, 1, L"5", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        2, 1, L"6", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    switch (value_type) {
        case 0: break;
        case 1:
            if (cursor_pos == 5) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 1, L"东(+)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        case 2:
            if (cursor_pos == 0) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 1, L"东经(+)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        case 3:
            if (cursor_pos == 0) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 1, L"北纬(+)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        default: return;
    }


    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        0, 2, L"7", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        1, 2, L"8", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        2, 2, L"9", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    switch (value_type) {
        case 0: break;
        case 1:
            if (cursor_pos == 5) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 2, L"西(-)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        case 2:
            if (cursor_pos == 0) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 2, L"西经(-)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        case 3:
            if (cursor_pos == 0) {
                ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
                    3, 2, L"南纬(-)", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
            }
            break;
        default: return;
    }

    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        0, 3, L"←", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        1, 3, L"0", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        2, 3, L"→", NULL, cell_bg_R, cell_bg_G, cell_bg_B, 1, cell_text0_R, cell_text0_G, cell_text0_B, 1, 0xff, 0xff, 0xff, 1);
    ui_app_setting_grid16_refresh_button(key_event, global_state, 1,
        3, 3, L"确认", NULL, cell_bg_R, cell_bg_G+10, cell_bg_B, 1, 0x00, 0xff, 0x00, 1, 0x00, 0x00, 0x00, 1);
}


// 计算下个光标位置
static int32_t ui_app_setting_next_pos(Key_Event *key_event, Global_State *global_state, int32_t value_type, int32_t current_pos) {
    // yyyy-mm-dd
    // 0123456789    
    if (value_type == 0) {
        switch (current_pos) {
            case 0: return 1; break;
            case 1: return 2; break;
            case 2: return 3; break;
            case 3: return 5; break;
            case 4: return 5; break;
            case 5: return 6; break;
            case 6: return 8; break;
            case 7: return 8; break;
            case 8: return 9; break;
            case 9: return 0; break;
            default: return 0; break;
        }
    }
    // hh:mmsaabb
    // 0123456789
    else if (value_type == 1) {
        switch (current_pos) {
            case 0: return 1; break;
            case 1: return 3; break;
            case 2: return 3; break;
            case 3: return 4; break;
            case 4: return 5; break;
            case 5: return 6; break;
            case 6: return 7; break;
            case 7: return 8; break;
            case 8: return 9; break;
            case 9: return 0; break;
            default: return 0; break;
        }
    }
    // 经度
    // sddd_mm'ss"
    // 0123456789A
    else if (value_type == 2) {
        switch (current_pos) {
            case 0: return 1; break;
            case 1: return 2; break;
            case 2: return 3; break;
            case 3: return 5; break;
            case 4: return 5; break;
            case 5: return 6; break;
            case 6: return 8; break;
            case 7: return 8; break;
            case 8: return 9; break;
            case 9: return 0; break;
            default: return 0; break;
        }
    }
    // 纬度
    // sdd_mm'ss"
    // 0123456789
    else if (value_type == 3) {
        switch (current_pos) {
            case 0: return 1; break;
            case 1: return 2; break;
            case 2: return 4; break;
            case 3: return 4; break;
            case 4: return 5; break;
            case 5: return 7; break;
            case 6: return 7; break;
            case 7: return 8; break;
            case 8: return 0; break;
            default: return 0; break;
        }
    }
    else return 0;
}

// 计算上个光标位置
static int32_t ui_app_setting_prev_pos(Key_Event *key_event, Global_State *global_state, int32_t value_type, int32_t current_pos) {
    // yyyy-mm-dd
    // 0123456789    
    if (value_type == 0) {
        switch (current_pos) {
            case 0: return 9; break;
            case 1: return 0; break;
            case 2: return 1; break;
            case 3: return 2; break;
            case 4: return 3; break;
            case 5: return 3; break;
            case 6: return 5; break;
            case 7: return 6; break;
            case 8: return 6; break;
            case 9: return 8; break;
            default: return 9; break;
        }
    }
    // hh:mmsaabb
    // 0123456789
    else if (value_type == 1) {
        switch (current_pos) {
            case 0: return 9; break;
            case 1: return 0; break;
            case 2: return 1; break;
            case 3: return 1; break;
            case 4: return 3; break;
            case 5: return 4; break;
            case 6: return 5; break;
            case 7: return 6; break;
            case 8: return 7; break;
            case 9: return 8; break;
            default: return 9; break;
        }
    }
    // 经度
    // sddd_mm'ss"
    // 0123456789A
    else if (value_type == 2) {
        switch (current_pos) {
            case 0: return 9; break;
            case 1: return 0; break;
            case 2: return 1; break;
            case 3: return 2; break;
            case 4: return 3; break;
            case 5: return 3; break;
            case 6: return 5; break;
            case 7: return 6; break;
            case 8: return 6; break;
            case 9: return 8; break;
            default: return 9; break;
        }
    }
    // 纬度
    // sdd_mm'ss"
    // 0123456789
    else if (value_type == 3) {
        switch (current_pos) {
            case 0: return 8; break;
            case 1: return 0; break;
            case 2: return 1; break;
            case 3: return 2; break;
            case 4: return 2; break;
            case 5: return 4; break;
            case 6: return 5; break;
            case 7: return 5; break;
            case 8: return 7; break;
            default: return 8; break;
        }
    }
    else return 0;
}

void ui_app_setting_value_input_event_handler(Key_Event *key_event, Global_State *global_state, int32_t value_type) {
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
        value_text[cursor_pos] = L'1';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_2) {
        value_text[cursor_pos] = L'2';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_3) {
        value_text[cursor_pos] = L'3';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_4) {
        value_text[cursor_pos] = L'4';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_5) {
        value_text[cursor_pos] = L'5';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_6) {
        value_text[cursor_pos] = L'6';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_7) {
        value_text[cursor_pos] = L'7';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_8) {
        value_text[cursor_pos] = L'8';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_9) {
        value_text[cursor_pos] = L'9';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_0) {
        value_text[cursor_pos] = L'0';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
        global_state->STATE = STATE_SETTING_MENU;
        return;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_shift) {
        value_text[cursor_pos] = L'+';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_ctrl) {
        value_text[cursor_pos] = L'-';
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_enter) {
        // 日期 yyyy-mm-dd
        if (value_type == 0) {
            global_state->year =(value_text[0] - L'0') * 1000 + 
                                (value_text[1] - L'0') * 100 + 
                                (value_text[2] - L'0') * 10 + 
                                (value_text[3] - L'0') * 1;
            global_state->month=(value_text[5] - L'0') * 10 + 
                                (value_text[6] - L'0') * 1;
            global_state->day = (value_text[8] - L'0') * 10 + 
                                (value_text[9] - L'0') * 1;

            int32_t utc_year = 0, utc_month = 0, utc_day = 0, utc_hour = 0, utc_minute = 0, utc_second = 0;
            local_time_to_utc(
                global_state->year, global_state->month, global_state->day,
                global_state->hour, global_state->minute, global_state->second,
                global_state->timezone,
                &utc_year, &utc_month, &utc_day, &utc_hour, &utc_minute, &utc_second
            );
            set_sys_time(utc_year, utc_month, utc_day, utc_hour, utc_minute, 0);
        }
        // 时间时区 hh:mmsaabb
        else if (value_type == 1) {
            global_state->hour =    (value_text[0] - L'0') * 10 + 
                                    (value_text[1] - L'0') * 1;
            global_state->minute =  (value_text[3] - L'0') * 10 + 
                                    (value_text[4] - L'0') * 1;
            float tz_sign =         (value_text[5] == '+') ? 1.0f : (-1.0f);
            float tz_hour =         (value_text[6] - '0') * 10.0f + 
                                    (value_text[7] - '0') * 1.0f;
            float tz_min =          (value_text[8] - '0') * 10.0f + 
                                    (value_text[9] - '0') * 1.0f;
            global_state->timezone = tz_sign * (tz_hour + tz_min / 60.0f);

            int32_t utc_year = 0, utc_month = 0, utc_day = 0, utc_hour = 0, utc_minute = 0, utc_second = 0;
            local_time_to_utc(
                global_state->year, global_state->month, global_state->day,
                global_state->hour, global_state->minute, global_state->second,
                global_state->timezone,
                &utc_year, &utc_month, &utc_day, &utc_hour, &utc_minute, &utc_second
            );
            set_sys_time(utc_year, utc_month, utc_day, utc_hour, utc_minute, 0);
        }
        // 经度 sddd_mm'ss"
        else if (value_type == 2) {
            float lon_sign= (value_text[0] == '+') ? 1.0f : (-1.0f);
            float lon_hour= (value_text[1] - '0') * 100 + 
                            (value_text[2] - '0') * 10 + 
                            (value_text[3] - '0') * 1;
            float lon_min = (value_text[5] - '0') * 10 + 
                            (value_text[6] - '0') * 1;
            float lon_sec = (value_text[8] - '0') * 10 + 
                            (value_text[9] - '0') * 1;
            global_state->longitude = lon_sign * (lon_hour + lon_min / 60.0f + lon_sec / 3600.0f);
        }
        // 纬度 sdd_mm'ss"
        else if (value_type == 3) {
            float lat_sign= (value_text[0] == '+') ? 1.0f : (-1.0f);
            float lat_hour= (value_text[1] - '0') * 10 + 
                            (value_text[2] - '0') * 1;
            float lat_min = (value_text[4] - '0') * 10 + 
                            (value_text[5] - '0') * 1;
            float lat_sec = (value_text[7] - '0') * 10 + 
                            (value_text[8] - '0') * 1;
            global_state->latitude = lat_sign * (lat_hour + lat_min / 60.0f + lat_sec / 3600.0f);
        }
        global_state->STATE = STATE_SETTING_MENU;
        return;
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_left) {
        cursor_pos = ui_app_setting_prev_pos(key_event, global_state, value_type, cursor_pos);
    }
    else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_right) {
        cursor_pos = ui_app_setting_next_pos(key_event, global_state, value_type, cursor_pos);
    }

    // 有键按下则刷新
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code != NANO_KEY_IDLE) {
        ui_app_setting_value_input_draw(key_event, global_state, value_type, value_text, cursor_pos);
        gfx_refresh(global_state->gfx);
    }
}












// ===============================================================================
// UI主体框架
// ===============================================================================

int32_t main_init(Key_Event *key_event, Global_State *global_state) {

    key_event->key_code = NANO_KEY_IDLE; // 大于等于16为没有任何按键，0-15为按键
    key_event->key_edge = 0;   // 0：松开  1：上升沿  -1：下降沿(短按结束)  -2：下降沿(长按结束)
    key_event->key_timer = 0;  // 按下计时器
    key_event->key_mask = 0;   // 长按超时后，键盘软复位标记。此时虽然物理上依然按键，只要软复位标记为1，则认为是无按键，无论是边沿还是按住都不触发。直到物理按键松开后，软复位标记清0。
    key_event->key_repeat = 0; // 触发一次长按后，只要不松手，该标记置1，直到物理按键松开后置0。若该标记为1，则在按住时触发连续重复动作。


    ///////////////////////////////////////
    // gfx初始化

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    
#else
    global_state->gfx = (Nano_GFX*)platform_calloc(1, sizeof(Nano_GFX));
    global_state->gfx->is_double_buffer = 0;
    gfx_init(global_state->gfx, SCREEN_WIDTH, SCREEN_HEIGHT, GFX_COLOR_MODE_RGB888);
#endif


    ui_init(key_event, global_state);

    ui_widget_textarea_init(key_event, global_state, global_state->w_textarea_main, UI_STR_BUF_MAX_LENGTH);




    ///////////////////////////////////////
    // UPS传感器初始化
#ifdef UPS_ENABLED
    ups_init();
#endif

    ///////////////////////////////////////
    // IMU初始化
#ifdef IMU_ENABLED
    imu_init();
    imu_calib();
#endif

    ///////////////////////////////////////
    // 输入设备初始化

    input_device_init();
    ui_softkbd_init(); // 触屏软键盘（内部初始化触屏HAL）
    key_event->prev_key = NANO_KEY_IDLE;

    ///////////////////////////////////////
    // 初始化玲珑天象仪

    global_state->linglong_cfg = (Linglong_Config *)platform_calloc(1, sizeof(Linglong_Config));
    linglong_init(global_state->linglong_cfg);

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    global_state->llgfx = (Nano_GFX*)platform_calloc(1, sizeof(Nano_GFX));
    global_state->llgfx->is_double_buffer = 0;
    gfx_init(global_state->llgfx, SCREEN_WIDTH, SCREEN_HEIGHT, GFX_COLOR_MODE_RGB888);
#else
    global_state->llgfx = global_state->gfx;
#endif


    ///////////////////////////////////////
    // 初始化文件系统

    fs_init();


    global_state->timezone = 8.0f;
    global_state->longitude = 119.0f;
    global_state->latitude = 32.0f;

    return 0;
}




int32_t main_event_handler(Key_Event *key_event, Global_State *global_state) {

    // 将时间戳转为本地日期时间
    time_t ts = (time_t)(global_state->timestamp / 1000);
    struct tm *timeinfo = localtime(&ts);
    global_state->year = timeinfo->tm_year + 1900;
    global_state->month = timeinfo->tm_mon + 1;
    global_state->day = timeinfo->tm_mday;
    global_state->hour = timeinfo->tm_hour;
    global_state->minute = timeinfo->tm_min;
    global_state->second = timeinfo->tm_sec;
    global_state->millisecond = global_state->timestamp % 1000;


    // 主状态机
    switch(global_state->STATE) {

    /////////////////////////////////////////////
    // 初始状态：欢迎屏幕。按任意键进入主菜单
    /////////////////////////////////////////////

    case STATE_SPLASH_SCREEN:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {

        }
        global_state->PREV_STATE = global_state->STATE;

        if (global_state->timestamp - last_splash_timestamp >= 100) {
            ui_app_splash_render_frame(key_event, global_state);
            last_splash_timestamp = global_state->timestamp;
        }

        // 按下任何键，不论长短按，进入主菜单
        if (key_event->key_edge < 0 && key_event->key_code != NANO_KEY_IDLE) {
            global_state->STATE = STATE_MAIN_MENU;
        }

        break;

    /////////////////////////////////////////////
    // 主菜单。
    /////////////////////////////////////////////

    case STATE_MAIN_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_widget_grid16_draw(key_event, global_state);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_widget_grid16_event_handler(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // 电子书：文件列表菜单
    /////////////////////////////////////////////

    case STATE_EBOOK:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 统一为“页眉页脚先入帧缓冲、菜单绘制最后统一刷屏”，避免两阶段断续感
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_draw_footer_softkeys(key_event, global_state, L"↑", L"", L"↓", L"选择");
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, ui_ebook_menu_item_action, STATE_MAIN_MENU, STATE_EBOOK);

        break;

    /////////////////////////////////////////////
    // 电子书：阅读
    /////////////////////////////////////////////

    case STATE_EBOOK_READING:

        // 首次获得焦点：渲染当前页
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ebook_reading_render(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 翻页/跳页/返回等按键处理（发生变化的按键内部自行触发重绘）
        ui_ebook_reading_event_handler(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // 文字编辑器状态
    /////////////////////////////////////////////

    case STATE_LLM_INPUT:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
        }
        global_state->PREV_STATE = global_state->STATE;

#ifdef ASR_ENABLED
        // 长按D键：开始PTT
        if (key_event->key_edge == -2 && key_event->key_code == NANO_KEY_enter) {
            global_state->STATE = STATE_ASR_RUNNING;
            break;
        }
#endif

        global_state->STATE = ui_widget_input_event_handler(key_event, global_state, global_state->w_input_main, STATE_MODEL_MENU, STATE_LLM_INPUT, STATE_LLM_ON_INFER);

        break;

    /////////////////////////////////////////////
    // 选择语言模型状态
    /////////////////////////////////////////////

    case STATE_MODEL_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_draw_footer_softkeys(key_event, global_state, L"↑", L"", L"↓", L"选择");
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, model_menu_item_action, STATE_MAIN_MENU, STATE_MODEL_MENU);

        // 退出模型菜单回到主菜单时，释放LLM上下文（KV cache/分词器/采样器等常驻PSRAM）；
        // 再次进入时会经“选模型”流程（model_menu_item_action）重新建立
        if (global_state->STATE == STATE_MAIN_MENU && global_state->llm_ctx != NULL) {
            llm_context_free(global_state->llm_ctx);
            global_state->llm_ctx = NULL;
        }

        break;


    /////////////////////////////////////////////
    // 小游戏菜单
    /////////////////////////////////////////////

    case STATE_GAME_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 统一为“页眉页脚先入帧缓冲、菜单绘制最后统一刷屏”，避免两阶段断续感
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_draw_footer_softkeys(key_event, global_state, L"↑", L"", L"↓", L"选择");
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, game_menu_item_action, STATE_MAIN_MENU, STATE_GAME_MENU);

        break;


    /////////////////////////////////////////////
    // 日历
    /////////////////////////////////////////////

    case STATE_CALENDAR:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_calendar_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_calendar_render_frame(key_event, global_state);
        ui_calendar_event_handler(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 设置菜单
    /////////////////////////////////////////////

    case STATE_SETTING_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_setting_grid16_draw(key_event, global_state);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_setting_grid16_event_handler(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 语言推理进行中（异步，每个iter结束后会将控制权交还事件循环，而非自行阻塞到最后一个token）
    //   实际上就是将generate_sync的while循环打开，将其置于大的事件循环。
    /////////////////////////////////////////////

    case STATE_LLM_ON_INFER:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            wchar_t *prompt = (wchar_t*)platform_calloc(global_state->llm_max_seq_len + 1, sizeof(wchar_t));

            // 如果输入为空，则随机选用一个预置prompt
            if (wcslen(global_state->w_input_main->textarea.text) == 0) {
                set_random_prompt(global_state->w_input_main->textarea.text, global_state->timestamp);
                global_state->w_input_main->textarea.length = wcslen(global_state->w_input_main->textarea.text);
            }

            // 根据模型类型应用prompt模板（NOTE 注意：prompt模板会占用max_seq_len长度）
            if (global_state->llm_ctx->llm->arch == LLM_ARCH_NANO) {
                wcscat(prompt, L"<|instruct_mark|>");
                wcscat(prompt, global_state->w_input_main->textarea.text);
                wcscat(prompt, L"<|response_mark|>");
            }
            else if (global_state->llm_ctx->llm->arch == LLM_ARCH_QWEN2 || global_state->llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
                wcscpy(prompt, global_state->w_input_main->textarea.text);
                // Qwen思考模型：涉及主动添加/no_think标记和生成参数调整
                if (global_state->llm_is_thinking_model != 0) {
                    if (global_state->is_thinking_enabled == 0) {
                        wcscat(prompt, L" /no_think");
                        // TODO 采样参数应该是session的参数，而不是ctx的参数
                        global_state->llm_ctx->sampler->temperature = qwen3_infer_args_no_thinking[0];
                        global_state->llm_ctx->sampler->top_p = qwen3_infer_args_no_thinking[1];
                    }
                    else {
                        global_state->llm_ctx->sampler->temperature = qwen3_infer_args_thinking[0];
                        global_state->llm_ctx->sampler->top_p = qwen3_infer_args_thinking[1];
                    }
                }
                // Qwen非思考模型：无论如何都不加/no_think标记；统一将思考标记打开，避免次元编码器输出多余的<think></think>占位词元
                else {
                    global_state->is_thinking_enabled = 1;
                }
            }
            else {
                global_state->STATE = STATE_SPLASH_SCREEN;
                break;
            }

            // 初始化对话session
            global_state->llm_session = llm_session_init(global_state->llm_ctx, prompt, global_state->llm_max_seq_len, global_state->is_thinking_enabled);
            // session内部已复制prompt（infer.c llm_session_init），此处释放原缓冲，避免每轮泄漏
            free(prompt);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 事件循环主体：即同步版本的while(1)的循环体

        global_state->llm_status = llm_session_step(global_state->llm_ctx, global_state->llm_session);

        if (global_state->llm_status == LLM_RUNNING_IN_PREFILLING) {
            global_state->llm_status = on_llm_prefilling(key_event, global_state);
            // 外部被动中止
            if (global_state->llm_status == LLM_STOPPED_IN_PREFILLING) {
                llm_session_free(global_state->llm_session);
                global_state->STATE = STATE_LLM_AFTER_INFER;
            }
            else {
                global_state->STATE = STATE_LLM_ON_INFER;
            }
        }
        else if (global_state->llm_status == LLM_RUNNING_IN_DECODING) {
            global_state->llm_status = on_llm_decoding(key_event, global_state);
            // 外部被动中止
            if (global_state->llm_status == LLM_STOPPED_IN_DECODING) {
#ifdef TTS_ENABLED
                if (global_state->tts_req_mode > 0) {
                    stop_tts();
                }
#endif
                llm_session_free(global_state->llm_session);
                global_state->STATE = STATE_LLM_AFTER_INFER;
            }
            else {
                global_state->STATE = STATE_LLM_ON_INFER;
            }
        }
        else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {
            global_state->llm_status = on_llm_finished(key_event, global_state);
            llm_session_free(global_state->llm_session);
            global_state->STATE = STATE_LLM_AFTER_INFER;
        }
        else {
            global_state->llm_status = on_llm_finished(key_event, global_state);
            llm_session_free(global_state->llm_session);
            global_state->STATE = STATE_LLM_AFTER_INFER;
        }

        break;


    /////////////////////////////////////////////
    // 推理结束（自然结束或中断），显示推理结果
    /////////////////////////////////////////////

    case STATE_LLM_AFTER_INFER:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 标题
            ui_draw_header(key_event, global_state, global_state->llm_model_name, 1);

            // 底部
            wchar_t tps_str[50];
            swprintf(tps_str, 50, L"%ls | 已生成%d词元 | %.1f词元/秒", global_state->llm_model_name, global_state->token_num_of_last_session, global_state->llm_max_seq_len, global_state->tps_of_last_session);
            ui_draw_footer(key_event, global_state, tps_str, 1);

            // 计算提示语+生成内容的行数
            wchar_t *prompt_and_output = (wchar_t *)platform_calloc(UI_STR_BUF_MAX_LENGTH * 2, sizeof(wchar_t));
            ui_app_wcscat_bounded(prompt_and_output, L"[#1155ee]Homo:", UI_STR_BUF_MAX_LENGTH * 2);
            if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#000000]\n", UI_STR_BUF_MAX_LENGTH * 2);
            }
            else if (global_state->ui_color_style == UI_COLOR_DARK) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]\n", UI_STR_BUF_MAX_LENGTH * 2);
            }
            ui_app_wcscat_bounded(prompt_and_output, global_state->w_input_main->textarea.text, UI_STR_BUF_MAX_LENGTH * 2);
            ui_app_wcscat_bounded(prompt_and_output, L"\n--------------------\n[#65bb00]Nano:", UI_STR_BUF_MAX_LENGTH * 2);
            if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#000000]\n", UI_STR_BUF_MAX_LENGTH * 2);
            }
            else if (global_state->ui_color_style == UI_COLOR_DARK) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]\n", UI_STR_BUF_MAX_LENGTH * 2);
            }
            ui_app_wcscat_bounded(prompt_and_output, global_state->llm_output_of_last_session, UI_STR_BUF_MAX_LENGTH * 2);
            // 推理中止
            if (global_state->llm_status == LLM_STOPPED_IN_PREFILLING || global_state->llm_status == LLM_STOPPED_IN_DECODING) {
                ui_app_wcscat_bounded(prompt_and_output, L"\n\n[#ff0000][Nano:推理中止]", UI_STR_BUF_MAX_LENGTH * 2);
                if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                    ui_app_wcscat_bounded(prompt_and_output, L"[#000000]", UI_STR_BUF_MAX_LENGTH * 2);
                }
                else if (global_state->ui_color_style == UI_COLOR_DARK) {
                    ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]", UI_STR_BUF_MAX_LENGTH * 2);
                }
            }
            // 推理自然结束
            else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {

            }
            // 推理异常结束
            else {
                ui_app_wcscat_bounded(prompt_and_output, L"\n\n[#ff0000][Nano:推理异常结束]", UI_STR_BUF_MAX_LENGTH * 2);
                if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                    ui_app_wcscat_bounded(prompt_and_output, L"[#000000]", UI_STR_BUF_MAX_LENGTH * 2);
                }
                else if (global_state->ui_color_style == UI_COLOR_DARK) {
                    ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]", UI_STR_BUF_MAX_LENGTH * 2);
                }
            }
            wchar_t tps_wcstr[50];
            swprintf(tps_wcstr, 50, L"\n\n[#ffc840][%d/%d|%.1fTPS]", global_state->token_num_of_last_session, global_state->llm_max_seq_len, global_state->tps_of_last_session);
            ui_app_wcscat_bounded(prompt_and_output, tps_wcstr, UI_STR_BUF_MAX_LENGTH * 2);
            if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#000000]", UI_STR_BUF_MAX_LENGTH * 2);
            }
            else if (global_state->ui_color_style == UI_COLOR_DARK) {
                ui_app_wcscat_bounded(prompt_and_output, L"[#ffffff]", UI_STR_BUF_MAX_LENGTH * 2);
            }

            // 缓冲区与 prompt_and_output 同容量（UI_STR_BUF_MAX_LENGTH * 2），并截断双保险
            wcsncpy(global_state->llm_output_of_last_session, prompt_and_output, UI_STR_BUF_MAX_LENGTH * 2 - 1);
            global_state->llm_output_of_last_session[UI_STR_BUF_MAX_LENGTH * 2 - 1] = L'\0';

            free(prompt_and_output);

            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, global_state->llm_output_of_last_session, -1, 1);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 短按D键：重新推理。推理完成后，并不清除输入缓冲区，因此再次按D键会重新推理。
        if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            global_state->STATE = STATE_LLM_ON_INFER;
        }
        else {
            // 短按A键：停止TTS
            if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
#ifdef TTS_ENABLED
                if (global_state->tts_req_mode > 0) {
                    stop_tts();
                }
#endif
            }
            global_state->STATE = ui_widget_textarea_event_handler(key_event, global_state, global_state->w_textarea_main, STATE_LLM_INPUT, STATE_LLM_AFTER_INFER);
        }

        break;

    /////////////////////////////////////////////
    // ASR实时识别进行中（响应ASR客户端回报的ASR文本内容）
    /////////////////////////////////////////////

    case STATE_ASR_RUNNING:
#ifdef ASR_ENABLED
        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 设置PTT状态为按下（>0）
            if (set_ptt_status(66) < 0) break;

            // 打开ASR管道
            if (open_asr_fifo() < 0) break;


            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L"请说话...", 0, 0);

            global_state->is_recording = 1;
            global_state->asr_start_timestamp = global_state->timestamp;
        }
        global_state->PREV_STATE = global_state->STATE;

        // 实时显示ASR结果
        if (global_state->is_recording == 1) {
            int32_t len = read_asr_fifo(global_state->asr_output_buffer);
            (void)len;

            // 临时关闭draw_textarea的整帧绘制，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
            global_state->is_full_refresh = 0;
            gfx_soft_clear(global_state->gfx);

            // 显示ASR结果
            // if (len > 0) {
                ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, global_state->asr_output_buffer, -1, 1);
                ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
            // }

            // 绘制录音持续时间（位于底栏位置，底栏高度跟随当前字体行高）
            wchar_t rec_duration[50];
            swprintf(rec_duration, 50, L" %ds ", (uint32_t)((global_state->timestamp - global_state->asr_start_timestamp) / 1000));
            gfx_draw_textline(global_state->gfx, rec_duration, 0, global_state->gfx->height - (gfx_font_line_height(global_state->ui_font) + 1), 255, 255, 255, 0);

            gfx_refresh(global_state->gfx);

            // 重新开启整帧绘制，注意这个标记是所有函数共享的全局标记。
            global_state->is_full_refresh = 1;

        }

        // 松开按钮，停止PTT
        if (global_state->is_recording > 0 && key_event->key_edge == 0 && key_event->key_code == NANO_KEY_IDLE) {

            global_state->is_recording = 0;
            global_state->asr_start_timestamp = 0;

            close_asr_fifo();

            // // 设置PTT状态为松开（==0）
            if (set_ptt_status(0) < 0) break;
            close_ptt_fifo();

            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n      识别完成", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

            sleep_in_ms(500);

            wcscpy(global_state->w_input_main->textarea.text, global_state->asr_output_buffer);
            global_state->w_input_main->textarea.length = wcslen(global_state->asr_output_buffer);

            wcscpy(global_state->asr_output_buffer, L"请说话...");

            // ASR后立刻提交到LLM？
            if (global_state->is_auto_submit_after_asr) {
                global_state->STATE = STATE_LLM_ON_INFER;
            }
            else {
                global_state->w_input_main->current_page = 0;
                global_state->STATE = STATE_LLM_INPUT;
            }

        }

        // 短按A键：清屏，清除输入缓冲区，回到初始状态
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_esc) {
            // 刷新文本输入框
            ui_widget_input_init(key_event, global_state, global_state->w_input_main, global_state->llm_model_name);
            global_state->STATE = STATE_LLM_INPUT;
        }
#endif
        break;

    /////////////////////////////////////////////
    // 本机自述
    /////////////////////////////////////////////

    case STATE_README: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_draw_header(key_event, global_state, L"本机自述", 1);
            ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);

            wchar_t readme[1024];
            wchar_t color_reset_tag[10];
            if (global_state->ui_color_style == UI_COLOR_LIGHT) {
                wcscpy(color_reset_tag, L"[#000000]");
            }
            else if (global_state->ui_color_style == UI_COLOR_DARK) {
                wcscpy(color_reset_tag, L"[#ffffff]");
            }
            swprintf(readme, 1024, L"[#66ccff]Nano-Pod%ls v" NANO_VERSION "\n电子核桃EDC | M5Core2(ESP32)\n(c) 2025-2026 BD4SUR\n\n番茄表：基于FLIP算法实现的流体仿真沙漏，可根据IMU测量到的重力方向定向流动，并具备可调节的瓶颈节流机制，因此可以当作番茄表使用。\n\n鹦鹉笼：与电子鹦鹉（端侧语言模型）对话，可以观测推理状态。在树莓派等算力丰富的硬件平台上，还具备语音输入、语音合成能力，并支持Qwen等更大规模的语言模型。\n\n玲珑仪：是一款天文计算和天空仿真程序，能够根据星历算法、时间地点，计算日月等重要天体的实时位置，同时渲染逼真的天空视觉效果。由于计算量大，在MCU上性能不佳，渲染一帧时间以秒计，但能够实时呈现。\n\n灵机引擎：自研Scheme语言解释器，可通过在电子核桃上直接编写程序控制电子核桃的行为。该功能同时提供了文字终端和软键盘，能够在不依赖任何外部硬件的条件下，独立实现程序代码的输入、执行和结果反馈。\n\ngithub.com/bd4sur/Nano", color_reset_tag);

            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, readme, 0, 1);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }

        global_state->STATE = ui_widget_textarea_event_handler(key_event, global_state, global_state->w_textarea_main, STATE_MAIN_MENU, STATE_README);

        break;
    }

    /////////////////////////////////////////////
    // 玲珑天象仪：计算太阳和月亮位置
    /////////////////////////////////////////////

    case STATE_LINGLONG:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_linglong_init(key_event, global_state);
            ui_app_linglong_splash(key_event, global_state);
            sleep_in_ms(1000);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_linglong_event_handler(key_event, global_state);
        ui_app_linglong_render_frame(key_event, global_state);

        break;

    /////////////////////////////////////////////
    // Bad Apple! 动画
    /////////////////////////////////////////////

    case STATE_BADAPPLE:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            global_state->ba_begin_timestamp = global_state->timestamp;
            global_state->ba_frame_count = 0;
        }
        global_state->PREV_STATE = global_state->STATE;

#ifdef BADAPPLE_ENABLED
        ui_app_badapple_render_frame(key_event, global_state);
#else
        ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L"未启用 Bad Apple ～", 0, 0);
        ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
#endif

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }

        break;


    /////////////////////////////////////////////
    // FLIP流体模拟
    /////////////////////////////////////////////

    case STATE_FLIP:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            s_ui_flip_first_load_timestamp = 0;
            ui_app_flip_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_flip_render_frame(key_event, global_state);
        ui_app_flip_event_handler(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 黄金矿工
    /////////////////////////////////////////////

    case STATE_GOLDMINER: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_goldminer_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回 / D或2放钩）；返回主菜单则跳过一帧渲染
        ui_goldminer_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_GOLDMINER) break;

        // 逻辑更新 + 渲染一帧
        ui_goldminer_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 粒子生命
    /////////////////////////////////////////////

    case STATE_PARTICLELIFE: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_particlelife_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回 / D或2重置）；返回则释放粒子数组并跳过一帧渲染
        ui_particlelife_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_PARTICLELIFE) {
            ui_particlelife_on_exit();
            break;
        }

        // 逻辑更新 + 渲染一帧
        ui_particlelife_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 水波
    /////////////////////////////////////////////

    case STATE_RIPPLE: {

        // 首次获得焦点：初始化（从SD卡读取/wp.png并解码为纹理）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ripple_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回）；返回则释放纹理与波场缓冲并跳过一帧渲染
        ui_ripple_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_RIPPLE) {
            ui_ripple_on_exit();
            break;
        }

        // 触摸激发 + 逻辑更新 + 渲染一帧
        ui_ripple_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 水池（WebGL Water 移植）
    /////////////////////////////////////////////

    case STATE_WATER: {

        // 首次获得焦点：初始化（分配高度/速度场、法线场、视野映射与输出缓冲）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_water_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回 / D或2雨滴）；返回则释放全部内存并跳过一帧渲染
        ui_water_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_WATER) {
            ui_water_on_exit();
            break;
        }

        // 模拟 + 渲染一帧（触摸落水在 render_frame 内完成）
        ui_water_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 俄罗斯方块
    /////////////////////////////////////////////

    case STATE_TETRIS: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_tetris_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（A返回 / 移动旋转落底）；返回菜单则跳过一帧渲染
        ui_tetris_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_TETRIS) break;

        // 重力更新 + 渲染一帧
        ui_tetris_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 体积云与天空仿真
    /////////////////////////////////////////////

    case STATE_CLOUD: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_cloud_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按键处理（方向键视角 / 回车太阳 / 返回主菜单）；返回主菜单则跳过一帧渲染
        ui_cloud_event_handler(key_event, global_state);
        if (global_state->STATE != STATE_CLOUD) break;

        // 渲染一帧
        ui_cloud_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 计步器（时域峰值计数 + 频域周期性校验）
    /////////////////////////////////////////////

    case STATE_PEDOMETER: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_pedometer_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            ui_pedometer_deinit(key_event, global_state);
            global_state->STATE = STATE_MAIN_MENU;
            break;
        }

        // 采样/分析/渲染一帧
        ui_pedometer_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 音频频谱仪：STFT声谱图（从下往上滚动）
    /////////////////////////////////////////////

    case STATE_SPECTROGRAM: {

        // 首次获得焦点：初始化（构建STFT工作区并接管麦克风）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_spectrogram_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 按A键返回主菜单（关闭麦克风并恢复扬声器）
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            ui_spectrogram_deinit(key_event, global_state);
            global_state->STATE = STATE_MAIN_MENU;
            break;
        }

        // 采集一帧音频并渲染一帧声谱图
        ui_spectrogram_render_frame(key_event, global_state);

        break;
    }


    /////////////////////////////////////////////
    // 寻呼机（OFDM 声波数传）：模式菜单（发射/接收，同一时刻仅一种模式）
    /////////////////////////////////////////////

    case STATE_OFDM_MENU:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // 统一为“页眉页脚先入帧缓冲、菜单绘制最后统一刷屏”，避免两阶段断续感
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_draw_footer_softkeys(key_event, global_state, L"↑", L"", L"↓", L"选择");
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, ui_ofdm_menu_item_action, STATE_MAIN_MENU, STATE_OFDM_MENU);

        // 退出寻呼机模块（回主菜单）：释放 modem 码表（PSRAM）
        if (global_state->STATE == STATE_MAIN_MENU) {
            ui_ofdm_menu_on_exit();
        }

        break;


    /////////////////////////////////////////////
    // 寻呼机：发射文本输入（复用全局单例输入控件，D 提交进入发射态）
    /////////////////////////////////////////////

    case STATE_OFDM_TX:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_tx_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_tx_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：发射中（OFDM 音频经扬声器多通道循环播放，直到手动停止）
    /////////////////////////////////////////////

    case STATE_OFDM_TXING:

        // 首次获得焦点：调制并启动循环播放
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_txing_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_txing_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：接收中（硅麦采集 → OFDM 解调 → 文本滚动显示）
    /////////////////////////////////////////////

    case STATE_OFDM_RX:

        // 首次获得焦点：接管麦克风并创建接收机
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_rx_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_rx_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：软件环路自测文本输入（复用全局单例输入控件，D 提交进入环回态）
    /////////////////////////////////////////////

    case STATE_OFDM_LOOP:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_loop_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_loop_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 寻呼机：软件环路自测中（发射机逐帧渲染 → 直接喂本机接收机，不出声）
    /////////////////////////////////////////////

    case STATE_OFDM_LOOPING:

        // 首次获得焦点：创建收发机并启动环回
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_ofdm_looping_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_ofdm_looping_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 音乐盒：文件列表（SD 卡根目录 WAV/MP3，复用全局单例菜单控件）
    /////////////////////////////////////////////

    case STATE_MUSICBOX_MENU:

        // 首次获得焦点：初始化（从主菜单进入时列表已由 ui_musicbox_menu_init 构建；
        // 从播放态返回时列表保留，仅重绘）
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_draw_header(key_event, global_state, (wchar_t *)global_state->w_menu_main->title, 1);
            ui_draw_footer_softkeys(key_event, global_state, L"↑", L"", L"↓", L"选择");
            ui_widget_menu_refresh(key_event, global_state, global_state->w_menu_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_widget_menu_event_handler(key_event, global_state, global_state->w_menu_main, ui_musicbox_menu_item_action, STATE_MAIN_MENU, STATE_MUSICBOX_MENU);

        // 退出音乐盒模块（回主菜单）：释放文件列表（PSRAM）
        if (global_state->STATE == STATE_MAIN_MENU) {
            ui_musicbox_menu_on_exit();
        }

        break;


    /////////////////////////////////////////////
    // 音乐盒：播放中（解码 → 扬声器流式播放；D 暂停/恢复，←→ 音量，4/6 切曲，A 返回）
    /////////////////////////////////////////////

    case STATE_MUSICBOX_PLAYING:

        // 首次获得焦点：分配资源、打开解码器并启动播放
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_musicbox_playing_on_enter(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_musicbox_playing_event(key_event, global_state);

        break;


    /////////////////////////////////////////////
    // 元胞自动机：Conway的生命游戏
    /////////////////////////////////////////////

    case STATE_GAMEOFLIFE:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_gol_init(key_event, global_state, global_state->gfx->width, global_state->gfx->height);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_gol_render_frame(key_event, global_state);

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }
        // 按D键刷新
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            ui_app_gol_init(key_event, global_state, global_state->gfx->width, global_state->gfx->height);
        }

        break;


    /////////////////////////////////////////////
    // 演化算法
    /////////////////////////////////////////////

    case STATE_GENETIC:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_genetic_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_genetic_refresh(key_event, global_state, 10);

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }
        // 按D键刷新
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            ui_app_genetic_init(key_event, global_state);
        }

        break;


    /////////////////////////////////////////////
    // 演化算法+TSP
    /////////////////////////////////////////////

    case STATE_GENETIC_TSP:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_tsp_init(key_event, global_state);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_tsp_refresh(key_event, global_state);

        // 按A键返回主菜单
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }
        // 按D键刷新
        else if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            ui_app_tsp_init(key_event, global_state);
        }

        break;

    /////////////////////////////////////////////
    // 电子词典：前缀查询（候选菜单 + 固定软键盘；统一 GFX_FONT_ALPHA_12）
    /////////////////////////////////////////////

    case STATE_DICT_QUERY:

        // 注意：必须先更新 PREV_STATE 再调用事件处理——若处理中发生状态迁移（如退出回主菜单），
        // 依赖“PREV_STATE != STATE 才重绘”的目标状态才能触发首帧重绘
        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_dict_query_event(key_event, global_state,
            STATE_MAIN_MENU, STATE_DICT_QUERY, STATE_DICT_DETAIL);

        break;

    /////////////////////////////////////////////
    // 电子词典：词条详情（字体 GFX_FONT_ALPHA_16）
    /////////////////////////////////////////////

    case STATE_DICT_DETAIL:

        global_state->PREV_STATE = global_state->STATE;

        global_state->STATE = ui_dict_detail_event(key_event, global_state,
            STATE_DICT_QUERY, STATE_DICT_DETAIL);

        break;


    /////////////////////////////////////////////
    // 关机确认
    /////////////////////////////////////////////

    case STATE_SHUTDOWN:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_draw_header(key_event, global_state, L"安全关机", 1);
            ui_draw_footer(key_event, global_state, L"(c) 2025-2026 BD4SUR", 1);
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L"确定关机？\n\n·长按D键: 关机\n·短按A键: 返回", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 长按D键确认关机
        if (key_event->key_edge == -2 && key_event->key_code == NANO_KEY_enter) {
            ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L" \n \n    正在安全关机...", 0, 0);
            ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

            if (graceful_shutdown() >= 0) {
                // exit(0);
            }
            // 关机失败，返回主菜单
            else {
                ui_widget_textarea_set(key_event, global_state, global_state->w_textarea_main, L"安全关机失败", 0, 0);
                ui_widget_textarea_draw(key_event, global_state, global_state->w_textarea_main);

                sleep_in_ms(1000);

                global_state->STATE = STATE_MAIN_MENU;
            }
        }

        // 长短按A键取消关机，返回主菜单
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
        }

        break;


    /////////////////////////////////////////////
    // 设置：虚拟键盘输入数值
    /////////////////////////////////////////////

    case STATE_SETTING_INPUT:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_app_setting_value_input_draw(key_event, global_state, value_type, value_text, cursor_pos);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        ui_app_setting_value_input_event_handler(key_event, global_state, value_type);

        break;


    /////////////////////////////////////////////
    // 时光集：相册
    /////////////////////////////////////////////

    case STATE_ALBUM:

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {

            s_album_index = 0; // 进入相册时重置浏览索引，防止越出本次文件数范围

            gfx_soft_clear(global_state->gfx);
            gfx_draw_textline_centered(global_state->gfx, L"枚举图片文件", 160, 10, 0x66, 0xcc, 0xff, 1);

            const char *path = "/image";

            // 获取数量
            int32_t count = list_files(path, NULL);
            if (count < 0) {
                printf("打开目录失败\n");
                break;
            }
            if (count == 0) {
                printf("目录为空\n");
                break;
            }

            // 分配指针数组
            s_album_path_list = (char **)platform_malloc(count * sizeof(char *));
            if (!s_album_path_list) {
                printf("内存不足\n");
                break;
            }

            // 获取文件名
            int32_t actual = list_files(path, s_album_path_list);
            if (actual < 0) {
                printf("读取失败\n");
                free(s_album_path_list);
                break;
            }
            s_album_count = actual;

            // 对文件名列表进行升序排序
            sort_strings(s_album_path_list, actual, 0);

            // 拼接成完整路径
            for (int32_t i = 0; i < actual; i++) {
                const char *filename = s_album_path_list[i];
                if (filename == NULL) continue;

                size_t path_len = strlen(path);
                size_t name_len = strlen(filename);
                int need_sep = (path_len == 0 || path[path_len - 1] == '/') ? 0 : 1;

                char *full_path = (char *)platform_malloc(path_len + need_sep + name_len + 1);
                if (full_path == NULL) {
                    printf("拼接路径内存不足\n");
                    continue;
                }

                snprintf(full_path, path_len + need_sep + name_len + 1,
                         "%s%s%s", path, need_sep ? "/" : "", filename);

                free(s_album_path_list[i]);
                s_album_path_list[i] = full_path;
            }

            // 显示文件列表
            printf("目录 %s 中有 %ld 个文件:\n", path, actual);
            for (int32_t i = 0; i < actual; i++) {
                wchar_t namew[128];
                _mbstowcs(namew, s_album_path_list[i], 128);
                gfx_draw_textline_centered(global_state->gfx, namew, 160, 10 + (i+1) * 17, 0xff, 0xff, 0xff, 1);
                printf("  [%ld] %s\n", i, s_album_path_list[i]);
            }


            gfx_refresh(global_state->gfx);
            sleep_in_ms(3000);









            gfx_draw_busy(global_state->gfx);
            gfx_refresh(global_state->gfx);
            ui_draw_image(key_event, global_state, s_album_path_list[s_album_index % s_album_count], 1);
            gfx_refresh(global_state->gfx);
        }
        global_state->PREV_STATE = global_state->STATE;

        // 长短按1键：自动放映
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_1) {
            s_album_is_autoplay++;
            s_album_is_autoplay = s_album_is_autoplay % 2;
            s_album_refresh_timestamp = global_state->timestamp;

            ui_draw_image(key_event, global_state, s_album_path_list[s_album_index], 1);
            // if (s_album_is_autoplay) {
            //     gfx_draw_textline(global_state->gfx, L"★", 0, 0, 0x00, 0xff, 0x00, 1);
            // }
            gfx_refresh(global_state->gfx);
        }
        // 长短按2/4/5/7/8/*/0键，切换上一张图片
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && (
            (key_event->key_code == NANO_KEY_2) ||
            (key_event->key_code == NANO_KEY_4) || (key_event->key_code == NANO_KEY_5) ||
            (key_event->key_code == NANO_KEY_7) || (key_event->key_code == NANO_KEY_8) ||
            (key_event->key_code == NANO_KEY_left) || (key_event->key_code == NANO_KEY_0)
        )) {
            ui_draw_image(key_event, global_state, s_album_path_list[s_album_index], 1);
            gfx_refresh(global_state->gfx);
            s_album_index--;
            if (s_album_index < 0) s_album_index = s_album_count - 1;
            s_album_index = s_album_index % s_album_count;
        }
        // 长短按A键返回主菜单
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == NANO_KEY_esc) {
            global_state->STATE = STATE_MAIN_MENU;
            // 释放内存
            for (int32_t i = 0; i < s_album_count; i++) {
                printf("Freeing %s\n", s_album_path_list[i]);
                free(s_album_path_list[i]);     // 释放每个文件名
            }
            free(s_album_path_list);            // 释放指针数组
            s_album_path_list = NULL;
            // 释放图像文件缓冲区与RGB888像素缓冲区（225KB+），避免常驻PSRAM造成堆碎片
            if (s_image_file_buffer != NULL) {
                free(s_image_file_buffer);
                s_image_file_buffer = NULL;
            }
            if (s_image_rgb888_buffer != NULL) {
                free(s_image_rgb888_buffer);
                s_image_rgb888_buffer = NULL;
            }
            s_image_file_size = 0;
            s_image_decode_ready = 0;
            s_image_filename_buffer[0] = '\0'; // 使下次进入相册时重新读文件解码
            printf("Free done.\n");
        }
        // 长短按3/6/B/9/C/#/D键，切换下一张图片
        else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && (
            (key_event->key_code == NANO_KEY_3) ||
            (key_event->key_code == NANO_KEY_6) || (key_event->key_code == NANO_KEY_shift) ||
            (key_event->key_code == NANO_KEY_9) || (key_event->key_code == NANO_KEY_ctrl) ||
            (key_event->key_code == NANO_KEY_right) || (key_event->key_code == NANO_KEY_enter)
        )) {
            ui_draw_image(key_event, global_state, s_album_path_list[s_album_index], 1);
            gfx_refresh(global_state->gfx);
            s_album_index++;
            s_album_index = s_album_index % s_album_count;
        }

        // 自动放映
        if (s_album_is_autoplay) {
            // 每6000ms切换
            if (global_state->timestamp - s_album_refresh_timestamp >= 6000) {
                ui_draw_image(key_event, global_state, s_album_path_list[s_album_index], 1);
                // gfx_draw_textline(global_state->gfx, L"★", 0, 0, 0x00, 0xff, 0x00, 1);
                gfx_refresh(global_state->gfx);
                s_album_index++;
                s_album_index = s_album_index % s_album_count;
                s_album_refresh_timestamp = global_state->timestamp;
            }
            else {
                // int32_t aa = (float)(global_state->timestamp - s_album_refresh_timestamp) / 6000.0f * 360;
                // gfx_draw_circle_fill(global_state->gfx, 6, 6, 6, 0xff, 0xff, 0xff, 1);
                // gfx_draw_sector(global_state->gfx, 6, 6, 6, 0, aa, 0x66, 0xcc, 0xff, 1);

                int32_t w = (float)(global_state->timestamp - s_album_refresh_timestamp) / 6000.0f * global_state->gfx->width;
                // gfx_draw_rectangle(global_state->gfx, 0, 239, global_state->gfx->width, 1, 0x00, 0x00, 0x00, 1);
                gfx_draw_rectangle(global_state->gfx, 0, 239, w, 1, 0x00, 0xaa, 0xff, 1);
                gfx_refresh(global_state->gfx);
            }
        }

        break;


    /////////////////////////////////////////////
    // Animac终端：初始化
    /////////////////////////////////////////////

    case STATE_ANIMAC_INIT: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            // ANIMAC终端临时将文字编辑控件字体改为 GFX_FONT_ALPHA_12，退出时恢复
            s_animac_prev_ui_font = global_state->ui_font;
            global_state->ui_font = GFX_FONT_ALPHA_12;
            ui_widget_input_init(key_event, global_state, global_state->w_input_main, L"电子核桃控制台");
            // 提示符
            wcscat(global_state->w_input_main->textarea.text, UI_ANIMAC_STARTUP_MESSAGE);
            wcscat(global_state->w_input_main->textarea.text, L"> ");
            // 刷新input控件，使光标到最后
            ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
        }
        global_state->PREV_STATE = global_state->STATE;

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32) || defined(ESP_PLATFORM)
    // NOTE 临时调试代码（后续删除）：初始化REPL前打印当前内存使用情况
    printf("[Animac] DRAM Free: %u | Largest: %u | PSRAM Free: %u | DMA Free: %u/%u\n",
        heap_caps_get_free_size(MALLOC_CAP_8BIT),
        heap_caps_get_largest_free_block(MALLOC_CAP_8BIT),
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        heap_caps_get_free_size(MALLOC_CAP_DMA),
        heap_caps_get_largest_free_block(MALLOC_CAP_DMA));
#endif

        ui_animac_init(key_event, global_state);

        global_state->STATE = STATE_ANIMAC_CONSOLE;

        break;
    }


    /////////////////////////////////////////////
    // Animac终端：终端等待输入
    /////////////////////////////////////////////

    case STATE_ANIMAC_CONSOLE: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
            s_animac_console_text_len = wcslen(global_state->w_input_main->textarea.text);
            // .load 指令载入的内容注入到提示符之后的输入区
            if (ui_animac_apply_pending_input(global_state, &s_animac_console_text_len)) {
                ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
            }
        }
        global_state->PREV_STATE = global_state->STATE;

        // 编辑器模式：驱动解释器事件循环（定时器等异步任务），并将其输出行搬入控制台
        ui_animac_idle_pump(key_event, global_state, &s_animac_console_text_len);

        // 注：软键盘切换请求消费与 Ctrl+0 绑定均已下沉到文本输入控件（ui.c），此处不再拦截

        // Ctrl+V（触屏软键盘）：恢复上次提交的输入内容到输入区
        if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->is_softkbd == 1
            && global_state->is_ctrl_enabled == 1
            && (key_event->key_code == NANO_KEY_v || key_event->key_code == NANO_KEY_V)) {
            global_state->is_ctrl_enabled = 0;
            ui_animac_restore_last_input(key_event, global_state, s_animac_console_text_len);
            break;
        }

        // Animac控制台交互：Enter换行、Ctrl+Enter提交执行（与通用输入框的语义相反，
        // 故在此拦截Enter键，不交给 ui_widget_input_event_handler 的原生Enter分支）
        if (key_event->key_edge == -1 && key_event->key_code == NANO_KEY_enter) {
            if (global_state->is_ctrl_enabled == 1) {
                // Ctrl+Enter：提交执行
                global_state->is_ctrl_enabled = 0;
                global_state->w_input_main->state = 0;
                global_state->STATE = STATE_ANIMAC_RUNNING;
            }
            else {
                // Enter：插入换行
                insert_char(global_state->w_input_main, L'\n');
                ui_draw_input_buffer(key_event, global_state, global_state->w_input_main);
            }
        }
        else {
            global_state->STATE = ui_widget_input_event_handler(key_event, global_state, global_state->w_input_main, STATE_MAIN_MENU, STATE_ANIMAC_CONSOLE, STATE_ANIMAC_RUNNING);
        }

        // 离开控制台时销毁解释器上下文释放内存（约2MB PSRAM）
        // （软键盘收起与布局恢复已由文本输入控件的退出路径固有处理，见 ui.c ui_widget_input_on_leave）
        if (global_state->STATE != STATE_ANIMAC_CONSOLE && global_state->STATE != STATE_ANIMAC_RUNNING) {
            ui_pinyin_ime_reset();
            ui_animac_close(key_event, global_state);
            global_state->ui_font = s_animac_prev_ui_font; // 恢复进入 STATE_ANIMAC_* 之前的字体
        }

        break;
    }


    /////////////////////////////////////////////
    // Animac终端：执行中
    /////////////////////////////////////////////

    case STATE_ANIMAC_RUNNING: {

        // 首次获得焦点：初始化
        if (global_state->PREV_STATE != global_state->STATE) {
            ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);
        }
        global_state->PREV_STATE = global_state->STATE;

        wchar_t *console_text = global_state->w_input_main->textarea.text;
        // 控制台输入缓冲区（一次性分配于PSRAM，避免占用任务栈与紧张的内部RAM；
        // 控制台单线程运行无重入；长度与控制台文本缓冲上限一致）
        static wchar_t *new_input = NULL;
        if (!new_input) new_input = (wchar_t*)platform_calloc(UI_STR_BUF_MAX_LENGTH, sizeof(wchar_t));
        wcscpy(new_input, &console_text[s_animac_console_text_len]);

        // 记录本次提交的输入，供 Ctrl+V 恢复
        ui_animac_save_last_input(new_input);

        ui_animac_exec(key_event, global_state, new_input, console_text, &s_animac_console_text_len);

        ui_animac_console_append(console_text, L"> ");
        ui_widget_input_refresh(key_event, global_state, global_state->w_input_main);

        global_state->STATE = STATE_ANIMAC_CONSOLE;

        break;
    }














    default:
        break;
    }

    return 0;
}



int32_t main_periodic_task(Key_Event *key_event, Global_State *global_state) {
    // 定期检查系统状态
    if (global_state->timer % 600 == 0) {
#ifdef ASR_ENABLED
        // ASR服务状态
        global_state->is_asr_server_up = check_asr_server_status();
#endif
#ifdef UPS_ENABLED
        global_state->ups_is_charging = read_ups_is_charging();
        global_state->ups_voltage = read_ups_voltage();
        global_state->ups_current = read_ups_current();
        global_state->ups_soc = read_ups_soc();
#endif
    }
    // 逻辑时间戳
    global_state->timer = (global_state->timer == 2147483647) ? 0 : (global_state->timer + 1);

    // 自动关机：倒计时到期即优雅关机（关机失败则取消本次设置，避免反复触发）
    if (global_state->auto_shutdown_deadline != 0 &&
        global_state->timestamp >= global_state->auto_shutdown_deadline) {
        graceful_shutdown();
        global_state->auto_shutdown_deadline = 0;
        global_state->auto_shutdown_minutes = 0;
    }

    return 0;
}


int32_t main_deinit(Key_Event *key_event, Global_State *global_state) {
    llm_context_free(global_state->llm_ctx);

    gfx_close(global_state->gfx);

    free(global_state->llm_output_of_last_session);

#ifdef ASR_ENABLED
    free(global_state->asr_output_buffer);
#endif

    free(global_state->w_textarea_main);

    free(global_state->w_input_main);

    free(global_state->w_menu_main);

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}