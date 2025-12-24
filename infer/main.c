

#include <locale.h>


#include "graphics.h"
#include "ui.h"
#include "ups.h"
#include "keyboard_hal.h"
#include "infer.h"
#include "prompt.h"

#include "platform.h"

#ifdef ASR_ENABLED
    #include "asr.h"
#endif
#ifdef TTS_ENABLED
    #include "tts.h"
#endif

#define PREFILL_LED_ON  system("echo \"1\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define PREFILL_LED_OFF system("echo \"0\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define DECODE_LED_ON   system("echo \"1\" > /sys/devices/platform/leds/leds/blue:status/brightness");
#define DECODE_LED_OFF  system("echo \"0\" > /sys/devices/platform/leds/leds/blue:status/brightness");


static char *LOG_FILE_PATH = "chat.jsonl";

static char *MODEL_PATH_1 = MODEL_ROOT_DIR "/nano_168m_625000_sft_947000_q80.bin";
static char *MODEL_PATH_2 = MODEL_ROOT_DIR "/nano_56m_99000_sft_v2_200000_q80.bin";
static char *MODEL_PATH_3 = MODEL_ROOT_DIR "/1-基础模型-99000_q80.bin";
static char *LORA_PATH_3  = MODEL_ROOT_DIR "/2-插件-猫娘.bin";
static char *MODEL_PATH_4 = MODEL_ROOT_DIR "/qwen3-0b6-q80.bin";
static char *MODEL_PATH_5 = MODEL_ROOT_DIR "/qwen3-1b7-q80.bin";
static char *MODEL_PATH_6 = MODEL_ROOT_DIR "/qwen3-4b-instruct-2507-q80.bin";

static uint32_t g_tokens_count = 0;
static float g_tps_of_last_session = 0.0f;
static wchar_t g_llm_output_of_last_session[OUTPUT_BUFFER_LENGTH] = L"";
static wchar_t g_asr_output[OUTPUT_BUFFER_LENGTH] = L"请说话...";


// 全局设置
int32_t g_config_auto_submit_after_asr = 1; // ASR结束后立刻提交识别内容到LLM
int32_t g_config_tts_mode = 0; // TTS工作模式：0-关闭   1-实时   2-全部生成后统一TTS


///////////////////////////////////////
// 全局GUI组件对象

Global_State           *global_state  = {0};
Key_Event              *key_event = {0};
Widget_Textarea_State  *widget_textarea_state = {0};
Widget_Textarea_State  *asr_textarea_state = {0};
Widget_Textarea_State  *prefilling_textarea_state = {0};
Widget_Input_State     *widget_input_state = {0};
Widget_Menu_State      *main_menu_state = {0};
Widget_Menu_State      *model_menu_state = {0};
Widget_Menu_State      *setting_menu_state = {0};

// 全局状态标志
int32_t STATE = -1;
int32_t PREV_STATE = -1;


int32_t on_llm_prefilling(Key_Event *key_event, Global_State *global_state) {
    Nano_Session *session = global_state->llm_session;

    if (session->t_0 == 0) {
        session->t_0 = global_state->timestamp;
    }
    else {
        session->tps = (session->pos - 1) / (float)(global_state->timestamp - session->t_0) * 1000;
    }

    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
        wcscpy(g_llm_output_of_last_session, L"");
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_PREFILLING;
    }

    // PREFILL_LED_ON

    // 屏幕刷新节流
    if (global_state->timestamp - global_state->llm_refresh_timestamp > (1000 / global_state->llm_refresh_max_fps)) {

        prefilling_textarea_state->x = 0;
        prefilling_textarea_state->y = 0;
        prefilling_textarea_state->width = 128;
        prefilling_textarea_state->height = 24;

        wcscpy(prefilling_textarea_state->text, L"Pre-filling...");
        prefilling_textarea_state->current_line = 0;
        prefilling_textarea_state->is_show_scroll_bar = 0;
    
        // 临时关闭draw_textarea的整帧绘制，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
        global_state->is_full_refresh = 0;

        fb_soft_clear();

        draw_textarea(key_event, global_state, prefilling_textarea_state);

        fb_draw_line(0, 60, 128, 60, 1);
        fb_draw_line(0, 63, 128, 63, 1);
        fb_draw_line(127, 60, 127, 63, 1);
        fb_draw_line(0, 61, session->pos * 128 / (session->num_prompt_tokens - 2), 61, 1);
        fb_draw_line(0, 62, session->pos * 128 / (session->num_prompt_tokens - 2), 62, 1);

        gfx_refresh();

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
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
        wcscpy(g_llm_output_of_last_session, session->output_text);
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_DECODING;
    }

    // DECODE_LED_ON

    // 屏幕刷新节流
    if (global_state->timestamp - global_state->llm_refresh_timestamp > (1000 / global_state->llm_refresh_max_fps)) {
        wcscpy(widget_textarea_state->text, session->output_text);
        widget_textarea_state->current_line = -1;
        widget_textarea_state->is_show_scroll_bar = 1;
        draw_textarea(key_event, global_state, widget_textarea_state);
        global_state->llm_refresh_timestamp = global_state->timestamp;
    }

    // DECODE_LED_OFF

#ifdef TTS_ENABLED
    if (g_config_tts_mode > 0) {
        send_tts_request(session->output_text, 0);
    }
#endif

    return LLM_RUNNING_IN_DECODING;
}

int32_t on_llm_finished(Key_Event *key_event, Global_State *global_state) {
    Nano_Session *session = global_state->llm_session;

    session->t_1 = global_state->timestamp;
    session->tps = (session->pos - 1) / (float)(session->t_1 - session->t_0) * 1000;

    wcscpy(g_llm_output_of_last_session, session->output_text);

    // 将本轮对话写入日志
    write_chat_log(LOG_FILE_PATH, global_state->timestamp, session->prompt, g_llm_output_of_last_session);

#ifdef TTS_ENABLED
    if (g_config_tts_mode > 0) {
        send_tts_request(session->output_text, 1);
    }
    reset_tts_split_status();
#endif

    g_tps_of_last_session = session->tps;
    g_tokens_count = session->pos;

    return LLM_STOPPED_NORMALLY;
}


///////////////////////////////////////
// 全局组件操作过程

void init_main_menu() {
    wcscpy(main_menu_state->title, L"Nano-Pod V2512");
    wcscpy(main_menu_state->items[0], L"电子鹦鹉");
    wcscpy(main_menu_state->items[1], L"电子书");
    wcscpy(main_menu_state->items[2], L"设置");
    wcscpy(main_menu_state->items[3], L"安全关机");
    wcscpy(main_menu_state->items[4], L"本机自述");
    main_menu_state->item_num = 5;
    init_menu(key_event, global_state, main_menu_state);
}

void init_model_menu() {
    wcscpy(model_menu_state->title, L"Select LLM");
    wcscpy(model_menu_state->items[0], L"Nano-168M-QA");
    wcscpy(model_menu_state->items[1], L"Nano-56M-QA");
    wcscpy(model_menu_state->items[2], L"Nano-56M-Neko");
    wcscpy(model_menu_state->items[3], L"Qwen3-0.6B");
    wcscpy(model_menu_state->items[4], L"Qwen3-1.7B");
    wcscpy(model_menu_state->items[5], L"Qwen3-4B-Inst-2507");
    model_menu_state->item_num = 6;
    init_menu(key_event, global_state, model_menu_state);
}

void refresh_model_menu() {
    draw_menu(key_event, global_state, model_menu_state);
}

void init_setting_menu() {
    wcscpy(setting_menu_state->title, L"设置");
    wcscpy(setting_menu_state->items[0], L"语言模型生成参数");
    wcscpy(setting_menu_state->items[1], L"语音合成(TTS)设置");
    wcscpy(setting_menu_state->items[2], L"语音识别(ASR)设置");
    setting_menu_state->item_num = 3;
    init_menu(key_event, global_state, setting_menu_state);
}


///////////////////////////////////////
// 菜单条目动作回调

// 主菜单各条目的动作
int32_t main_menu_item_action(int32_t item_index) {
    // 0.电子鹦鹉
    if (item_index == 0) {
        init_model_menu();
        return 4;
    }

    // 1.电子书
    else if (item_index == 1) {
        return -3;
    }

    // 2.设置
    else if (item_index == 2) {
        init_setting_menu();
        return 5;
    }

    // 3.安全关机
    else if (item_index == 3) {
        return 31;
    }

    // 4.本机自述
    else if (item_index == 4) {
        return 26;
    }
    return -2;
}

int32_t model_menu_item_action(int32_t item_index) {
    if (global_state->llm_ctx) {
        llm_context_free(global_state->llm_ctx);
    }

    if (item_index == 0) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Nano-168M-QA\n 请稍等...");
        global_state->llm_model_path = MODEL_PATH_1;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.05f;
        global_state->llm_temperature = 1.0f;
        global_state->llm_top_p = 0.5f;
        global_state->llm_top_k = 0;
        global_state->llm_max_seq_len = 512;
    }
    else if (item_index == 1) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Nano-56M-QA\n 请稍等...");
        global_state->llm_model_path = MODEL_PATH_2;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.05f;
        global_state->llm_temperature = 1.0f;
        global_state->llm_top_p = 0.5f;
        global_state->llm_top_k = 0;
        global_state->llm_max_seq_len = 512;
    }
    else if (item_index == 2) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Nano-56M-Neko\n 请稍等...");
        global_state->llm_model_path = MODEL_PATH_3;
        global_state->llm_lora_path = LORA_PATH_3;
        global_state->llm_repetition_penalty = 1.05f;
        global_state->llm_temperature = 1.0f;
        global_state->llm_top_p = 0.5f;
        global_state->llm_top_k = 0;
        global_state->llm_max_seq_len = 512;
    }
    else if (item_index == 3) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-0.6B\n 请稍等...");
        global_state->llm_model_path = MODEL_PATH_4;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.0f;
        global_state->llm_temperature = 0.6f;
        global_state->llm_top_p = 0.95f;
        global_state->llm_top_k = 20;
        global_state->llm_max_seq_len = 32768;
    }
    else if (item_index == 4) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-1.7B\n 请稍等...");
        global_state->llm_model_path = MODEL_PATH_5;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.0f;
        global_state->llm_temperature = 0.6f;
        global_state->llm_top_p = 0.95f;
        global_state->llm_top_k = 20;
        global_state->llm_max_seq_len = 32768;
    }
    else if (item_index == 5) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-4B-Inst-2507\n 请稍等...");
        global_state->llm_model_path = MODEL_PATH_6;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.0f;
        global_state->llm_temperature = 0.7f;
        global_state->llm_top_p = 0.8f;
        global_state->llm_top_k = 20;
        global_state->llm_max_seq_len = 32768;
    }

    widget_textarea_state->current_line = 0;
    widget_textarea_state->is_show_scroll_bar = 0;
    draw_textarea(key_event, global_state, widget_textarea_state);

    global_state->llm_ctx = llm_context_init(
        global_state->llm_model_path,
        global_state->llm_lora_path,
        global_state->llm_max_seq_len,
        global_state->llm_repetition_penalty,
        global_state->llm_temperature,
        global_state->llm_top_p,
        global_state->llm_top_k,
        global_state->timestamp);

    wcscpy(widget_textarea_state->text, L"加载完成~");
    widget_textarea_state->current_line = 0;
    widget_textarea_state->is_show_scroll_bar = 0;
    draw_textarea(key_event, global_state, widget_textarea_state);

    sleep_in_ms(500);

    // 以下两条路选一个：

    // 1、直接进入电子鹦鹉
    init_input(key_event, global_state, widget_input_state);
    return 0;

    // 2、或者回到主菜单
    // refresh_menu(key_event, global_state, main_menu_state);
    // return -2;
}

int32_t setting_menu_item_action(int32_t item_index) {
    // 语言模型生成参数设置
    if (item_index == 0) {
        wcscpy(widget_textarea_state->text, L"暂未实现");
        widget_textarea_state->current_line = 0;
        widget_textarea_state->is_show_scroll_bar = 0;
        draw_textarea(key_event, global_state, widget_textarea_state);

        sleep_in_ms(500);

        refresh_menu(key_event, global_state, setting_menu_state);
        return 5;
    }
    // TTS设置
    else if (item_index == 1) {
        return 32;
    }
    // ASR设置
    else if (item_index == 2) {
        return 33;
    }
    else {
        return 5;
    }
}



int main() {
    

    if(!setlocale(LC_CTYPE, "")) return -1;

    (void)g_asr_output; // 抑制编译器报变量未使用问题

    ///////////////////////////////////////
    // 初始化GUI状态

    global_state = (Global_State*)calloc(1, sizeof(Global_State));
    key_event = (Key_Event*)calloc(1, sizeof(Key_Event));
    widget_textarea_state = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));
    asr_textarea_state = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));
    prefilling_textarea_state = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));
    widget_input_state = (Widget_Input_State*)calloc(1, sizeof(Widget_Input_State));
    main_menu_state = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    model_menu_state = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    setting_menu_state = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));

    global_state->llm_status = LLM_STOPPED_NORMALLY;
    global_state->llm_model_path = NULL;
    global_state->llm_lora_path = NULL;
    global_state->llm_repetition_penalty = 1.05f;
    global_state->llm_temperature = 1.0f;
    global_state->llm_top_p = 0.5f;
    global_state->llm_top_k = 0;
    global_state->llm_max_seq_len = 512;
    global_state->is_asr_server_up = 0;
    global_state->is_recording = 0;
    global_state->asr_start_timestamp = 0;
    global_state->is_full_refresh = 1;
    global_state->llm_refresh_max_fps = 20;
    global_state->llm_refresh_timestamp = 0;

    widget_textarea_state->x = 0;
    widget_textarea_state->y = 0;
    widget_textarea_state->width = 128;
    widget_textarea_state->height = 64;
    widget_textarea_state->line_num = 0;
    widget_textarea_state->current_line = 0;

    key_event->key_code = KEYCODE_NUM_IDLE; // 大于等于16为没有任何按键，0-15为按键
    key_event->key_edge = 0;   // 0：松开  1：上升沿  -1：下降沿(短按结束)  -2：下降沿(长按结束)
    key_event->key_timer = 0;  // 按下计时器
    key_event->key_mask = 0;   // 长按超时后，键盘软复位标记。此时虽然物理上依然按键，只要软复位标记为1，则认为是无按键，无论是边沿还是按住都不触发。直到物理按键松开后，软复位标记清0。
    key_event->key_repeat = 0; // 触发一次长按后，只要不松手，该标记置1，直到物理按键松开后置0。若该标记为1，则在按住时触发连续重复动作。

    // 空按键状态：用于定时器事件
    Key_Event *void_key_event = (Key_Event*)calloc(1, sizeof(Key_Event));
    void_key_event->key_code = KEYCODE_NUM_IDLE;
    void_key_event->key_edge = 0;
    void_key_event->key_timer = 0;
    void_key_event->key_mask = 0;
    void_key_event->key_repeat = 0;

    ///////////////////////////////////////
    // UPS传感器初始化
#ifdef UPS_ENABLED
    ups_init();
#endif

    ///////////////////////////////////////
    // OLED 初始化

    gfx_init();

    show_splash_screen(key_event, global_state);

    ///////////////////////////////////////
    // 矩阵按键初始化与读取

    keyboard_hal_init();
    key_event->prev_key = KEYCODE_NUM_IDLE;

    ///////////////////////////////////////
    // 主循环

    while (1) {

        // 物理时间戳
        global_state->timestamp = get_timestamp_in_ms();

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




        switch(STATE) {

        /////////////////////////////////////////////
        // 初始状态：欢迎屏幕。按任意键进入主菜单
        /////////////////////////////////////////////

        case -1:

            // 节流
            if (global_state->timer % 10 == 0) {
                show_splash_screen(key_event, global_state);
            }

            // 按下任何键，不论长短按，进入主菜单
            if (key_event->key_edge < 0 && key_event->key_code != KEYCODE_NUM_IDLE) {
                init_main_menu();
                STATE = -2;
            }

            break;

        /////////////////////////////////////////////
        // 主菜单。
        /////////////////////////////////////////////

        case -2:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_menu(key_event, global_state, main_menu_state);
            }
            PREV_STATE = STATE;

            STATE = menu_event_handler(key_event, global_state, main_menu_state, main_menu_item_action, -1, -2);

            break;

        /////////////////////////////////////////////
        // 文本显示状态
        /////////////////////////////////////////////

        case -3:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                wchar_t* content = read_file_to_wchar(LOG_FILE_PATH);
                if (content) {
                    wcscpy(widget_textarea_state->text, content);
                    free(content);
                }
                else {
                    wcscpy(widget_textarea_state->text, L"文件不存在...");
                }
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 1;
                draw_textarea(key_event, global_state, widget_textarea_state);
            }
            PREV_STATE = STATE;

            STATE = textarea_event_handler(key_event, global_state, widget_textarea_state, -2, -3);

            break;

        /////////////////////////////////////////////
        // 文字编辑器状态
        /////////////////////////////////////////////

        case 0:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_input(key_event, global_state, widget_input_state);
            }
            PREV_STATE = STATE;

            // 长+短按A键：删除一个字符；如果输入缓冲区为空，则回到主菜单
            if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
                if (widget_input_state->state == 0 && widget_input_state->length <= 0) {
                    init_input(key_event, global_state, widget_input_state);
                    STATE = -2;
                }
            }
#ifdef ASR_ENABLED
            // 按下C键：开始PTT
            else if (key_event->key_edge > 0 && key_event->key_code == KEYCODE_NUM_C) {
                STATE = 21;
            }
#endif
            // 短按D键：提交
            else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_D) {
                if (widget_input_state->state == 0) {
                    STATE = 8;
                }
            }

            draw_input(key_event, global_state, widget_input_state);

            break;

        /////////////////////////////////////////////
        // 选择语言模型状态
        /////////////////////////////////////////////

        case 4:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_menu(key_event, global_state, model_menu_state);
            }
            PREV_STATE = STATE;

            STATE = menu_event_handler(key_event, global_state, model_menu_state, model_menu_item_action, -2, 4);

            break;


        /////////////////////////////////////////////
        // 设置菜单
        /////////////////////////////////////////////

        case 5:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_menu(key_event, global_state, setting_menu_state);
            }
            PREV_STATE = STATE;

            STATE = menu_event_handler(key_event, global_state, setting_menu_state, setting_menu_item_action, -2, 5);

            break;


        /////////////////////////////////////////////
        // 语言推理进行中（异步，每个iter结束后会将控制权交还事件循环，而非自行阻塞到最后一个token）
        //   实际上就是将generate_sync的while循环打开，将其置于大的事件循环。
        /////////////////////////////////////////////

        case 8:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                wchar_t prompt[MAX_PROMPT_BUFFER_LENGTH] = L"";

                // 如果输入为空，则随机选用一个预置prompt
                if (wcslen(widget_input_state->text) == 0) {
                    set_random_prompt(widget_input_state->text, global_state->timer);
                    widget_input_state->length = wcslen(widget_input_state->text);
                }

                // 根据模型类型应用prompt模板
                if (global_state->llm_ctx->llm->arch == LLM_ARCH_NANO) {
                    wcscat(prompt, L"<|instruct_mark|>");
                    wcscat(prompt, widget_input_state->text);
                    wcscat(prompt, L"<|response_mark|>");
                }
                else if (global_state->llm_ctx->llm->arch == LLM_ARCH_QWEN2 || global_state->llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
                    wcscpy(prompt, widget_input_state->text);
                    // wcscat(prompt, L" /no_think");
                }
                else {
                    STATE = -1;
                    break;
                }

                // 初始化对话session
                global_state->llm_session = llm_session_init(global_state->llm_ctx, prompt, global_state->llm_max_seq_len);
            }
            PREV_STATE = STATE;

            // 事件循环主体：即同步版本的while(1)的循环体

            global_state->llm_status = llm_session_step(global_state->llm_ctx, global_state->llm_session);

            if (global_state->llm_status == LLM_RUNNING_IN_PREFILLING) {
                global_state->llm_status = on_llm_prefilling(key_event, global_state);
                // 外部被动中止
                if (global_state->llm_status == LLM_STOPPED_IN_PREFILLING) {
                    llm_session_free(global_state->llm_session);
                    STATE = 10;
                }
                else {
                    STATE = 8;
                }
            }
            else if (global_state->llm_status == LLM_RUNNING_IN_DECODING) {
                global_state->llm_status = on_llm_decoding(key_event, global_state);
                // 外部被动中止
                if (global_state->llm_status == LLM_STOPPED_IN_DECODING) {
#ifdef TTS_ENABLED
                    if (g_config_tts_mode > 0) {
                        stop_tts();
                    }
#endif
                    llm_session_free(global_state->llm_session);
                    STATE = 10;
                }
                else {
                    STATE = 8;
                }
            }
            else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {
                global_state->llm_status = on_llm_finished(key_event, global_state);
                llm_session_free(global_state->llm_session);
                STATE = 10;
            }
            else {
                global_state->llm_status = on_llm_finished(key_event, global_state);
                llm_session_free(global_state->llm_session);
                STATE = 10;
            }

            break;


        /////////////////////////////////////////////
        // 推理结束（自然结束或中断），显示推理结果
        /////////////////////////////////////////////

        case 10:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                // 计算提示语+生成内容的行数
                wchar_t *prompt_and_output = (wchar_t *)calloc(OUTPUT_BUFFER_LENGTH * 2, sizeof(wchar_t));
                wcscat(prompt_and_output, L"Homo:\n");
                wcscat(prompt_and_output, widget_input_state->text);
                wcscat(prompt_and_output, L"\n--------------------\nNano:\n");
                wcscat(prompt_and_output, g_llm_output_of_last_session);
                // 推理中止
                if (global_state->llm_status == LLM_STOPPED_IN_PREFILLING || global_state->llm_status == LLM_STOPPED_IN_DECODING) {
                    wcscat(prompt_and_output, L"\n\n[Nano:推理中止]");
                }
                // 推理自然结束
                else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {

                }
                // 推理异常结束
                else {
                    wcscat(prompt_and_output, L"\n\n[Nano:推理异常结束]");
                }
                wchar_t tps_wcstr[50];
                swprintf(tps_wcstr, 50, L"\n\n[%d|%.1fTPS]", g_tokens_count, g_tps_of_last_session);
                wcscat(prompt_and_output, tps_wcstr);

                wcscpy(g_llm_output_of_last_session, prompt_and_output);

                free(prompt_and_output);

                wcscpy(widget_textarea_state->text, g_llm_output_of_last_session);
                widget_textarea_state->current_line = -1;
                widget_textarea_state->is_show_scroll_bar = 1;
                draw_textarea(key_event, global_state, widget_textarea_state);
            }
            PREV_STATE = STATE;

            // 短按D键：重新推理。推理完成后，并不清除输入缓冲区，因此再次按D键会重新推理。
            if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_D) {
                STATE = 8;
            }
            else {
                // 短按A键：停止TTS
                if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_A) {
#ifdef TTS_ENABLED
                    if (g_config_tts_mode > 0) {
                        stop_tts();
                    }
#endif
                }
                STATE = textarea_event_handler(key_event, global_state, widget_textarea_state, 0, 10);
            }

            break;

        /////////////////////////////////////////////
        // ASR实时识别进行中（响应ASR客户端回报的ASR文本内容）
        /////////////////////////////////////////////

        case 21:
#ifdef ASR_ENABLED
            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                // 设置PTT状态为按下（>0）
                if (set_ptt_status(66) < 0) break;

                // 打开ASR管道
                if (open_asr_fifo() < 0) break;


                asr_textarea_state->x = 0;
                asr_textarea_state->y = 0;
                asr_textarea_state->width = 128;
                asr_textarea_state->height = 51; // NOTE 详见结构体定义处的说明

                wcscpy(asr_textarea_state->text, L"请说话...");

                global_state->is_recording = 1;
                global_state->asr_start_timestamp = global_state->timestamp;
            }
            PREV_STATE = STATE;

            // 实时显示ASR结果
            if (global_state->is_recording == 1) {
                int32_t len = read_asr_fifo(g_asr_output);
                (void)len;

                // 临时关闭draw_textarea的整帧绘制，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
                global_state->is_full_refresh = 0;
                fb_soft_clear();

                // 显示ASR结果
                // if (len > 0) {
                    wcscpy(asr_textarea_state->text, g_asr_output);
                    asr_textarea_state->current_line = -1;
                    asr_textarea_state->is_show_scroll_bar = 1;
                    draw_textarea(key_event, global_state, asr_textarea_state);
                // }

                // 绘制录音持续时间
                wchar_t rec_duration[50];
                swprintf(rec_duration, 50, L" %ds ", (uint32_t)((global_state->timestamp - global_state->asr_start_timestamp) / 1000));
                render_line(rec_duration, 0, 52, 0);

                gfx_refresh();

                // 重新开启整帧绘制，注意这个标记是所有函数共享的全局标记。
                global_state->is_full_refresh = 1;

            }

            // 松开按钮，停止PTT
            if (global_state->is_recording > 0 && key_event->key_edge == 0 && key_event->key_code == KEYCODE_NUM_IDLE) {

                global_state->is_recording = 0;
                global_state->asr_start_timestamp = 0;

                close_asr_fifo();

                // // 设置PTT状态为松开（==0）
                if (set_ptt_status(0) < 0) break;
                close_ptt_fifo();

                wcscpy(asr_textarea_state->text, L" \n \n      识别完成");
                asr_textarea_state->current_line = 0;
                asr_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, asr_textarea_state);

                sleep_in_ms(500);

                wcscpy(widget_input_state->text, g_asr_output);
                widget_input_state->length = wcslen(g_asr_output);

                wcscpy(g_asr_output, L"请说话...");

                // ASR后立刻提交到LLM？
                if (g_config_auto_submit_after_asr) {
                    STATE = 8;
                }
                else {
                    widget_input_state->current_page = 0;
                    STATE = 0;
                }

            }

            // 短按A键：清屏，清除输入缓冲区，回到初始状态
            else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_A) {
                // 刷新文本输入框
                init_input(key_event, global_state, widget_input_state);
                STATE = 0;
            }
#endif
            break;

        /////////////////////////////////////////////
        // 本机自述
        /////////////////////////////////////////////

        case 26: {

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {

            }
            PREV_STATE = STATE;

            wchar_t readme_buf[128] = L"Nano-Pod v2512\n电子鹦鹉·端上大模型\n(c) 2025 BD4SUR\n\n";
            wchar_t status_buf[30];
            // 节流
            if (global_state->timer % 200 == 0) {
#ifdef UPS_ENABLED
                swprintf(status_buf, 30, L"UPS:%04dmV/%d%% ", global_state->ups_voltage, global_state->ups_soc);
#else
                wcscpy(status_buf, L"github.com/bd4sur");
#endif
                wcscat(readme_buf, status_buf);
                wcscpy(widget_textarea_state->text, readme_buf);
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);
            }

            // 按A键返回主菜单
            if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
                STATE = -2;
            }

            break;
        }

        /////////////////////////////////////////////
        // 关机确认
        /////////////////////////////////////////////

        case 31:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                wcscpy(widget_textarea_state->text, L"确定关机？\n\n·长按D键: 关机\n·短按A键: 返回");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);
            }
            PREV_STATE = STATE;

            // 长按D键确认关机
            if (key_event->key_edge == -2 && key_event->key_code == KEYCODE_NUM_D) {
                wcscpy(widget_textarea_state->text, L" \n \n    正在安全关机...");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                if (graceful_shutdown() >= 0) {
                    // exit(0);
                }
                // 关机失败，返回主菜单
                else {
                    wcscpy(widget_textarea_state->text, L"安全关机失败");
                    widget_textarea_state->current_line = 0;
                    widget_textarea_state->is_show_scroll_bar = 0;
                    draw_textarea(key_event, global_state, widget_textarea_state);

                    sleep_in_ms(1000);

                    STATE = -2;
                }
            }

            // 长短按A键取消关机，返回主菜单
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
                STATE = -2;
            }

            break;


        /////////////////////////////////////////////
        // TTS设置
        /////////////////////////////////////////////

        case 32:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                wcscpy(widget_textarea_state->text, L"语音合成(TTS)设置\n\n·0:关闭\n·1:实时请求合成\n·2:生成结束后合成");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);
            }
            PREV_STATE = STATE;

            // 选项0
            if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_0) {
                g_config_tts_mode = 0;

                wcscpy(widget_textarea_state->text, L"TTS已关闭。");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                sleep_in_ms(500);

                STATE = 5;
            }

            // 选项1
            else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_1) {
                g_config_tts_mode = 1;

                wcscpy(widget_textarea_state->text, L"TTS设置为实时请求。");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                sleep_in_ms(500);

                STATE = 5;
            }

            // 选项2
            else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_2) {
                g_config_tts_mode = 2;

                wcscpy(widget_textarea_state->text, L"TTS设置为生成结束后统一请求合成。");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                sleep_in_ms(500);

                STATE = 5;
            }

            // 长短按A键，返回设置菜单
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
                STATE = 5;
            }

            break;


        /////////////////////////////////////////////
        // ASR设置
        /////////////////////////////////////////////

        case 33:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                wcscpy(widget_textarea_state->text, L"语音识别(TTS)设置\n识别结果立刻提交？\n·0:先编辑再提交\n·1:立刻提交");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);
            }
            PREV_STATE = STATE;

            // 选项0
            if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_0) {
                g_config_auto_submit_after_asr = 0;

                wcscpy(widget_textarea_state->text, L"ASR自动提交已关闭");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                sleep_in_ms(500);

                STATE = 5;
            }

            // 选项1
            else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_1) {
                g_config_auto_submit_after_asr = 1;

                wcscpy(widget_textarea_state->text, L"ASR自动提交已开启");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                sleep_in_ms(500);

                STATE = 5;
            }

            // 长短按A键，返回设置菜单
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
                STATE = 5;
            }

            break;















        default:
            break;
        }



        // 定期检查系统状态
        if (global_state->timer % 600 == 0) {
#ifdef ASR_ENABLED
            // ASR服务状态
            global_state->is_asr_server_up = check_asr_server_status();
#endif
#ifdef UPS_ENABLED
            // UPS电压和电量
            global_state->ups_voltage = read_ups_voltage();
            global_state->ups_soc = read_ups_soc();
#endif
        }

        global_state->timer = (global_state->timer == 2147483647) ? 0 : (global_state->timer + 1);
    }

    llm_context_free(global_state->llm_ctx);

    free(global_state);
    free(key_event);
    free(widget_textarea_state);
    free(widget_input_state);
    free(main_menu_state);
    free(model_menu_state);
    free(prefilling_textarea_state);

    free(void_key_event);

    gfx_close();

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}
