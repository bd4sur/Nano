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

#define STATE_DEFAULT         (-100)
#define STATE_SPLASH_SCREEN   (-1)
#define STATE_MAIN_MENU       (-2)
#define STATE_EBOOK           (-3)
#define STATE_LLM_INPUT       (0)
#define STATE_MODEL_MENU      (4)
#define STATE_SETTING_MENU    (5)
#define STATE_LLM_ON_INFER    (8)
#define STATE_LLM_AFTER_INFER (10)
#define STATE_ASR_RUNNING     (21)
#define STATE_README          (26)
#define STATE_SHUTDOWN        (31)
#define STATE_TTS_SETTING     (32)
#define STATE_ASR_SETTING     (33)

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
static wchar_t g_llm_output_of_last_session[INPUT_BUFFER_LENGTH] = L"";
static wchar_t g_asr_output[INPUT_BUFFER_LENGTH] = L"请说话...";




///////////////////////////////////////
// 全局GUI组件对象

Global_State           *global_state  = {0};
Key_Event              *key_event = {0};

Widget_Textarea_State  *w_textarea_main = {0};
Widget_Textarea_State  *w_textarea_asr = {0};
Widget_Textarea_State  *w_textarea_prefill = {0};

Widget_Input_State     *w_input_main = {0};

Widget_Menu_State      *w_menu_main = {0};
Widget_Menu_State      *w_menu_model = {0};
Widget_Menu_State      *w_menu_setting = {0};
Widget_Menu_State      *w_menu_asr_setting = {0};
Widget_Menu_State      *w_menu_tts_setting = {0};


// 全局状态标志
int32_t STATE = STATE_SPLASH_SCREEN;
int32_t PREV_STATE = STATE_DEFAULT;


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

        w_textarea_prefill->x = 0;
        w_textarea_prefill->y = 0;
        w_textarea_prefill->width = 128;
        w_textarea_prefill->height = 24;

        set_textarea(key_event, global_state, w_textarea_prefill, L"Pre-filling...", 0, 0);
    
        // 临时关闭draw_textarea的整帧绘制，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
        global_state->is_full_refresh = 0;

        fb_soft_clear();

        draw_textarea(key_event, global_state, w_textarea_prefill);

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
        set_textarea(key_event, global_state, w_textarea_main, session->output_text, -1, 1);
        draw_textarea(key_event, global_state, w_textarea_main);
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

    wcscpy(g_llm_output_of_last_session, session->output_text);

    // 将本轮对话写入日志
    write_chat_log(LOG_FILE_PATH, global_state->timestamp, session->prompt, g_llm_output_of_last_session);

#ifdef TTS_ENABLED
    if (global_state->tts_req_mode > 0) {
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
    wcscpy(w_menu_main->title, L"Nano-Pod");
    wcscpy(w_menu_main->items[0], L"电子鹦鹉");
    wcscpy(w_menu_main->items[1], L"电子书");
    wcscpy(w_menu_main->items[2], L"设置");
    wcscpy(w_menu_main->items[3], L"安全关机");
    wcscpy(w_menu_main->items[4], L"本机自述");
    w_menu_main->item_num = 5;
    init_menu(key_event, global_state, w_menu_main);
}

void init_model_menu() {
    wcscpy(w_menu_model->title, L"Select LLM");
    wcscpy(w_menu_model->items[0], L"Nano-168M-QA");
    wcscpy(w_menu_model->items[1], L"Nano-56M-QA");
    wcscpy(w_menu_model->items[2], L"Nano-56M-Neko");
    wcscpy(w_menu_model->items[3], L"Qwen3-0.6B");
    wcscpy(w_menu_model->items[4], L"Qwen3-1.7B");
    wcscpy(w_menu_model->items[5], L"Qwen3-4B-Inst-2507");
    w_menu_model->item_num = 6;
    init_menu(key_event, global_state, w_menu_model);
}

void init_setting_menu() {
    wcscpy(w_menu_setting->title, L"设置");
    wcscpy(w_menu_setting->items[0], L"语言模型生成参数");
    wcscpy(w_menu_setting->items[1], L"语音合成(TTS)设置");
    wcscpy(w_menu_setting->items[2], L"语音识别(ASR)设置");
    w_menu_setting->item_num = 3;
    init_menu(key_event, global_state, w_menu_setting);
}

void init_asr_setting_menu() {
    wcscpy(w_menu_asr_setting->title, L"ASR自动提交设置");
    wcscpy(w_menu_asr_setting->items[0], L"0.先编辑再提交");
    wcscpy(w_menu_asr_setting->items[1], L"1.立刻提交");
    w_menu_asr_setting->item_num = 2;
    init_menu(key_event, global_state, w_menu_asr_setting);
}

void init_tts_setting_menu() {
    wcscpy(w_menu_tts_setting->title, L"TTS设置");
    wcscpy(w_menu_tts_setting->items[0], L"0.关闭");
    wcscpy(w_menu_tts_setting->items[1], L"1.实时TTS");
    wcscpy(w_menu_tts_setting->items[2], L"2.完成后统一TTS");
    w_menu_tts_setting->item_num = 3;
    init_menu(key_event, global_state, w_menu_tts_setting);
}



///////////////////////////////////////
// 菜单条目动作回调

// 主菜单各条目的动作
int32_t main_menu_item_action(int32_t item_index) {
    // 0.电子鹦鹉
    if (item_index == 0) {
        init_model_menu();
        return STATE_MODEL_MENU;
    }

    // 1.电子书
    else if (item_index == 1) {
        return STATE_EBOOK;
    }

    // 2.设置
    else if (item_index == 2) {
        init_setting_menu();
        return STATE_SETTING_MENU;
    }

    // 3.安全关机
    else if (item_index == 3) {
        return STATE_SHUTDOWN;
    }

    // 4.本机自述
    else if (item_index == 4) {
        return STATE_README;
    }
    return STATE_MAIN_MENU;
}

int32_t model_menu_item_action(int32_t item_index) {
    if (global_state->llm_ctx) {
        llm_context_free(global_state->llm_ctx);
    }

    wchar_t *llm_model_info = NULL;

    if (item_index == 0) {
        llm_model_info = L" 正在加载语言模型\n Nano-168M-QA\n 请稍等...";
        global_state->llm_model_path = MODEL_PATH_1;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.05f;
        global_state->llm_temperature = 1.0f;
        global_state->llm_top_p = 0.5f;
        global_state->llm_top_k = 0;
        global_state->llm_max_seq_len = 512;
    }
    else if (item_index == 1) {
        llm_model_info = L" 正在加载语言模型\n Nano-56M-QA\n 请稍等...";
        global_state->llm_model_path = MODEL_PATH_2;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.05f;
        global_state->llm_temperature = 1.0f;
        global_state->llm_top_p = 0.5f;
        global_state->llm_top_k = 0;
        global_state->llm_max_seq_len = 512;
    }
    else if (item_index == 2) {
        llm_model_info = L" 正在加载语言模型\n Nano-56M-Neko\n 请稍等...";
        global_state->llm_model_path = MODEL_PATH_3;
        global_state->llm_lora_path = LORA_PATH_3;
        global_state->llm_repetition_penalty = 1.05f;
        global_state->llm_temperature = 1.0f;
        global_state->llm_top_p = 0.5f;
        global_state->llm_top_k = 0;
        global_state->llm_max_seq_len = 512;
    }
    else if (item_index == 3) {
        llm_model_info = L" 正在加载语言模型\n Qwen3-0.6B\n 请稍等...";
        global_state->llm_model_path = MODEL_PATH_4;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.0f;
        global_state->llm_temperature = 0.6f;
        global_state->llm_top_p = 0.95f;
        global_state->llm_top_k = 20;
        global_state->llm_max_seq_len = 32768;
    }
    else if (item_index == 4) {
        llm_model_info = L" 正在加载语言模型\n Qwen3-1.7B\n 请稍等...";
        global_state->llm_model_path = MODEL_PATH_5;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.0f;
        global_state->llm_temperature = 0.6f;
        global_state->llm_top_p = 0.95f;
        global_state->llm_top_k = 20;
        global_state->llm_max_seq_len = 32768;
    }
    else if (item_index == 5) {
        llm_model_info = L" 正在加载语言模型\n Qwen3-4B-Inst-2507\n 请稍等...";
        global_state->llm_model_path = MODEL_PATH_6;
        global_state->llm_lora_path = NULL;
        global_state->llm_repetition_penalty = 1.0f;
        global_state->llm_temperature = 0.7f;
        global_state->llm_top_p = 0.8f;
        global_state->llm_top_k = 20;
        global_state->llm_max_seq_len = 32768;
    }

    set_textarea(key_event, global_state, w_textarea_main, llm_model_info, 0, 0);
    draw_textarea(key_event, global_state, w_textarea_main);

    global_state->llm_ctx = llm_context_init(
        global_state->llm_model_path,
        global_state->llm_lora_path,
        global_state->llm_max_seq_len,
        global_state->llm_repetition_penalty,
        global_state->llm_temperature,
        global_state->llm_top_p,
        global_state->llm_top_k,
        global_state->timestamp);

    set_textarea(key_event, global_state, w_textarea_main, L"加载完成~", 0, 0);
    draw_textarea(key_event, global_state, w_textarea_main);

    sleep_in_ms(500);

    // 以下两条路选一个：

    // 1、直接进入电子鹦鹉
    init_input(key_event, global_state, w_input_main);
    return STATE_LLM_INPUT;

    // 2、或者回到主菜单
    // refresh_menu(key_event, global_state, w_menu_main);
    // return STATE_MAIN_MENU;
}

int32_t setting_menu_item_action(int32_t item_index) {
    // 语言模型生成参数设置
    if (item_index == 0) {
        set_textarea(key_event, global_state, w_textarea_main, L"暂未实现", 0, 0);
        draw_textarea(key_event, global_state, w_textarea_main);

        sleep_in_ms(500);

        refresh_menu(key_event, global_state, w_menu_setting);
        return STATE_SETTING_MENU;
    }
    // TTS设置
    else if (item_index == 1) {
        init_tts_setting_menu();
        return STATE_TTS_SETTING;
    }
    // ASR设置
    else if (item_index == 2) {
        init_asr_setting_menu();
        return STATE_ASR_SETTING;
    }
    else {
        return STATE_SETTING_MENU;
    }
}


int32_t asr_setting_menu_item_action(int32_t item_index) {
    // 0.先编辑再提交
    if (item_index == 0) {
        global_state->is_auto_submit_after_asr = 0;

        set_textarea(key_event, global_state, w_textarea_main, L"ASR自动提交已关闭", 0, 0);
        draw_textarea(key_event, global_state, w_textarea_main);

        sleep_in_ms(500);

        refresh_menu(key_event, global_state, w_menu_asr_setting);
        return STATE_SETTING_MENU;
    }
    // 1.立刻提交
    else if (item_index == 1) {
        global_state->is_auto_submit_after_asr = 1;

        set_textarea(key_event, global_state, w_textarea_main, L"ASR自动提交已开启", 0, 0);
        draw_textarea(key_event, global_state, w_textarea_main);

        sleep_in_ms(500);

        refresh_menu(key_event, global_state, w_menu_asr_setting);
        return STATE_SETTING_MENU;
    }
    else {
        return STATE_ASR_SETTING;
    }
}


int32_t tts_setting_menu_item_action(int32_t item_index) {
    // 0.关闭
    if (item_index == 0) {
        global_state->tts_req_mode = 0;

        set_textarea(key_event, global_state, w_textarea_main, L"TTS已关闭。", 0, 0);
        draw_textarea(key_event, global_state, w_textarea_main);

        sleep_in_ms(500);

        refresh_menu(key_event, global_state, w_menu_tts_setting);
        return STATE_SETTING_MENU;
    }
    // 1.实时TTS
    else if (item_index == 1) {
        global_state->tts_req_mode = 1;

        set_textarea(key_event, global_state, w_textarea_main, L"TTS设置为实时请求。", 0, 0);
        draw_textarea(key_event, global_state, w_textarea_main);

        sleep_in_ms(500);

        refresh_menu(key_event, global_state, w_menu_tts_setting);
        return STATE_SETTING_MENU;
    }
    // 2.完成后统一TTS
    else if (item_index == 2) {
        global_state->tts_req_mode = 2;

        set_textarea(key_event, global_state, w_textarea_main, L"TTS设置为全部生成后统一请求。", 0, 0);
        draw_textarea(key_event, global_state, w_textarea_main);

        sleep_in_ms(500);

        refresh_menu(key_event, global_state, w_menu_tts_setting);
        return STATE_SETTING_MENU;
    }
    else {
        return STATE_TTS_SETTING;
    }
}






int main() {

    if(!setlocale(LC_CTYPE, "")) return -1;

    (void)g_asr_output; // 抑制编译器报变量未使用问题

    ///////////////////////////////////////
    // 初始化GUI状态

    global_state = (Global_State*)calloc(1, sizeof(Global_State));
    key_event = (Key_Event*)calloc(1, sizeof(Key_Event));

    w_textarea_main = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));
    w_textarea_asr = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));
    w_textarea_prefill = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));

    w_input_main = (Widget_Input_State*)calloc(1, sizeof(Widget_Input_State));

    w_menu_main = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    w_menu_model = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    w_menu_setting = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    w_menu_asr_setting = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));
    w_menu_tts_setting = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));

    global_state->is_ctrl_enabled = 0;
    global_state->llm_status = LLM_STOPPED_NORMALLY;
    global_state->llm_model_path = NULL;
    global_state->llm_lora_path = NULL;
    global_state->llm_repetition_penalty = 1.05f;
    global_state->llm_temperature = 1.0f;
    global_state->llm_top_p = 0.5f;
    global_state->llm_top_k = 0;
    global_state->llm_max_seq_len = 512;
    global_state->is_thinking_enabled = 1;
    global_state->is_auto_submit_after_asr = 1; // ASR结束后立刻提交识别内容到LLM
    global_state->is_asr_server_up = 0;
    global_state->is_recording = 0;
    global_state->asr_start_timestamp = 0;
    global_state->is_full_refresh = 1;
    global_state->llm_refresh_max_fps = 10;
    global_state->llm_refresh_timestamp = 0;

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

    init_textarea(key_event, global_state, w_textarea_main, INPUT_BUFFER_LENGTH);
    init_textarea(key_event, global_state, w_textarea_asr, INPUT_BUFFER_LENGTH);
    init_textarea(key_event, global_state, w_textarea_prefill, INPUT_BUFFER_LENGTH);

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

        case STATE_SPLASH_SCREEN:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {

            }
            PREV_STATE = STATE;

            // 节流
            if (global_state->timer % 10 == 0) {
                show_splash_screen(key_event, global_state);
            }

            // 按下任何键，不论长短按，进入主菜单
            if (key_event->key_edge < 0 && key_event->key_code != KEYCODE_NUM_IDLE) {
                STATE = STATE_MAIN_MENU;
            }

            break;

        /////////////////////////////////////////////
        // 主菜单。
        /////////////////////////////////////////////

        case STATE_MAIN_MENU:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                init_main_menu();
            }
            PREV_STATE = STATE;

            STATE = menu_event_handler(key_event, global_state, w_menu_main, main_menu_item_action, STATE_SPLASH_SCREEN, STATE_MAIN_MENU);

            break;

        /////////////////////////////////////////////
        // 文本显示状态
        /////////////////////////////////////////////

        case STATE_EBOOK:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
#ifdef TTS_ENABLED
                reset_tts_split_status();
#endif
                wchar_t* content = read_file_to_wchar(LOG_FILE_PATH);
                if (content) {
                    set_textarea(key_event, global_state, w_textarea_main, content, 0, 1);
                    free(content);
                }
                else {
                    set_textarea(key_event, global_state, w_textarea_main, L"文件不存在...", 0, 1);
                }
                draw_textarea(key_event, global_state, w_textarea_main);
            }
            PREV_STATE = STATE;

#ifdef TTS_ENABLED
            // TODO 应逐句发送请求，不要一次性请求

            // 短按A键：停止TTS
            if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_A) {
                stop_tts();
            }
            // 短按D键：请求TTS
            else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_D) {
                for (int32_t i = 0; i < w_textarea_main->length; i++) {
                    send_tts_request(w_textarea_main->text + i, 0);
                }
            }
#endif

            STATE = textarea_event_handler(key_event, global_state, w_textarea_main, STATE_MAIN_MENU, STATE_EBOOK);

            break;

        /////////////////////////////////////////////
        // 文字编辑器状态
        /////////////////////////////////////////////

        case STATE_LLM_INPUT:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_input(key_event, global_state, w_input_main);
            }
            PREV_STATE = STATE;

#ifdef ASR_ENABLED
            // 长按D键：开始PTT
            if (key_event->key_edge == -2 && key_event->key_code == KEYCODE_NUM_D) {
                STATE = STATE_ASR_RUNNING;
                break;
            }
#endif

            STATE = input_event_handler(key_event, global_state, w_input_main, STATE_MODEL_MENU, STATE_LLM_INPUT, STATE_LLM_ON_INFER);

            break;

        /////////////////////////////////////////////
        // 选择语言模型状态
        /////////////////////////////////////////////

        case STATE_MODEL_MENU:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_menu(key_event, global_state, w_menu_model);
            }
            PREV_STATE = STATE;

            STATE = menu_event_handler(key_event, global_state, w_menu_model, model_menu_item_action, STATE_MAIN_MENU, STATE_MODEL_MENU);

            break;


        /////////////////////////////////////////////
        // 设置菜单
        /////////////////////////////////////////////

        case STATE_SETTING_MENU:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_menu(key_event, global_state, w_menu_setting);
            }
            PREV_STATE = STATE;

            STATE = menu_event_handler(key_event, global_state, w_menu_setting, setting_menu_item_action, STATE_MAIN_MENU, STATE_SETTING_MENU);

            break;


        /////////////////////////////////////////////
        // 语言推理进行中（异步，每个iter结束后会将控制权交还事件循环，而非自行阻塞到最后一个token）
        //   实际上就是将generate_sync的while循环打开，将其置于大的事件循环。
        /////////////////////////////////////////////

        case STATE_LLM_ON_INFER:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                wchar_t *prompt = (wchar_t*)calloc(global_state->llm_max_seq_len + 1, sizeof(wchar_t));

                // 如果输入为空，则随机选用一个预置prompt
                if (wcslen(w_input_main->textarea.text) == 0) {
                    set_random_prompt(w_input_main->textarea.text, global_state->timer);
                    w_input_main->textarea.length = wcslen(w_input_main->textarea.text);
                }

                // 根据模型类型应用prompt模板（NOTE 注意：prompt模板会占用max_seq_len长度）
                if (global_state->llm_ctx->llm->arch == LLM_ARCH_NANO) {
                    wcscat(prompt, L"<|instruct_mark|>");
                    wcscat(prompt, w_input_main->textarea.text);
                    wcscat(prompt, L"<|response_mark|>");
                }
                else if (global_state->llm_ctx->llm->arch == LLM_ARCH_QWEN2 || global_state->llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
                    wcscpy(prompt, w_input_main->textarea.text);
                    // wcscat(prompt, L" /no_think");
                }
                else {
                    STATE = STATE_SPLASH_SCREEN;
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
                    STATE = STATE_LLM_AFTER_INFER;
                }
                else {
                    STATE = STATE_LLM_ON_INFER;
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
                    STATE = STATE_LLM_AFTER_INFER;
                }
                else {
                    STATE = STATE_LLM_ON_INFER;
                }
            }
            else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {
                global_state->llm_status = on_llm_finished(key_event, global_state);
                llm_session_free(global_state->llm_session);
                STATE = STATE_LLM_AFTER_INFER;
            }
            else {
                global_state->llm_status = on_llm_finished(key_event, global_state);
                llm_session_free(global_state->llm_session);
                STATE = STATE_LLM_AFTER_INFER;
            }

            break;


        /////////////////////////////////////////////
        // 推理结束（自然结束或中断），显示推理结果
        /////////////////////////////////////////////

        case STATE_LLM_AFTER_INFER:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                // 计算提示语+生成内容的行数
                wchar_t *prompt_and_output = (wchar_t *)calloc(INPUT_BUFFER_LENGTH * 2, sizeof(wchar_t));
                wcscat(prompt_and_output, L"Homo:\n");
                wcscat(prompt_and_output, w_input_main->textarea.text);
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

                set_textarea(key_event, global_state, w_textarea_main, g_llm_output_of_last_session, -1, 1);
                draw_textarea(key_event, global_state, w_textarea_main);
            }
            PREV_STATE = STATE;

            // 短按D键：重新推理。推理完成后，并不清除输入缓冲区，因此再次按D键会重新推理。
            if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_D) {
                STATE = STATE_LLM_ON_INFER;
            }
            else {
                // 短按A键：停止TTS
                if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_A) {
#ifdef TTS_ENABLED
                    if (global_state->tts_req_mode > 0) {
                        stop_tts();
                    }
#endif
                }
                STATE = textarea_event_handler(key_event, global_state, w_textarea_main, STATE_LLM_INPUT, STATE_LLM_AFTER_INFER);
            }

            break;

        /////////////////////////////////////////////
        // ASR实时识别进行中（响应ASR客户端回报的ASR文本内容）
        /////////////////////////////////////////////

        case STATE_ASR_RUNNING:
#ifdef ASR_ENABLED
            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                // 设置PTT状态为按下（>0）
                if (set_ptt_status(66) < 0) break;

                // 打开ASR管道
                if (open_asr_fifo() < 0) break;


                w_textarea_asr->x = 0;
                w_textarea_asr->y = 0;
                w_textarea_asr->width = 128;
                w_textarea_asr->height = 51; // NOTE 详见结构体定义处的说明

                set_textarea(key_event, global_state, w_textarea_asr, L"请说话...", 0, 0);

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
                    set_textarea(key_event, global_state, w_textarea_asr, g_asr_output, -1, 1);
                    draw_textarea(key_event, global_state, w_textarea_asr);
                // }

                // 绘制录音持续时间
                wchar_t rec_duration[50];
                swprintf(rec_duration, 50, L" %ds ", (uint32_t)((global_state->timestamp - global_state->asr_start_timestamp) / 1000));
                fb_draw_textline(rec_duration, 0, 52, 0);

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

                set_textarea(key_event, global_state, w_textarea_asr, L" \n \n      识别完成", 0, 0);
                draw_textarea(key_event, global_state, w_textarea_asr);

                sleep_in_ms(500);

                wcscpy(w_input_main->textarea.text, g_asr_output);
                w_input_main->textarea.length = wcslen(g_asr_output);

                wcscpy(g_asr_output, L"请说话...");

                // ASR后立刻提交到LLM？
                if (global_state->is_auto_submit_after_asr) {
                    STATE = STATE_LLM_ON_INFER;
                }
                else {
                    w_input_main->current_page = 0;
                    STATE = STATE_LLM_INPUT;
                }

            }

            // 短按A键：清屏，清除输入缓冲区，回到初始状态
            else if (key_event->key_edge == -1 && key_event->key_code == KEYCODE_NUM_A) {
                // 刷新文本输入框
                init_input(key_event, global_state, w_input_main);
                STATE = STATE_LLM_INPUT;
            }
#endif
            break;

        /////////////////////////////////////////////
        // 本机自述
        /////////////////////////////////////////////

        case STATE_README: {

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {

            }
            PREV_STATE = STATE;

            wchar_t readme_buf[128] = L"Nano-Pod v" NANO_VERSION "\n电子鹦鹉·端上大模型\n(c) 2025-2026 BD4SUR\n\n";
            wchar_t status_buf[30];
            // 节流
            if (global_state->timer % 200 == 0) {
#ifdef UPS_ENABLED
                swprintf(status_buf, 30, L"UPS:%04dmV/%d%% ", global_state->ups_voltage, global_state->ups_soc);
#else
                wcscpy(status_buf, L"github.com/bd4sur");
#endif
                wcscat(readme_buf, status_buf);

                set_textarea(key_event, global_state, w_textarea_main, readme_buf, 0, 0);
                draw_textarea(key_event, global_state, w_textarea_main);
            }

            // 按A键返回主菜单
            if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
                STATE = STATE_MAIN_MENU;
            }

            break;
        }

        /////////////////////////////////////////////
        // 关机确认
        /////////////////////////////////////////////

        case STATE_SHUTDOWN:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                set_textarea(key_event, global_state, w_textarea_main, L"确定关机？\n\n·长按D键: 关机\n·短按A键: 返回", 0, 0);
                draw_textarea(key_event, global_state, w_textarea_main);
            }
            PREV_STATE = STATE;

            // 长按D键确认关机
            if (key_event->key_edge == -2 && key_event->key_code == KEYCODE_NUM_D) {
                set_textarea(key_event, global_state, w_textarea_main, L" \n \n    正在安全关机...", 0, 0);
                draw_textarea(key_event, global_state, w_textarea_main);

                if (graceful_shutdown() >= 0) {
                    // exit(0);
                }
                // 关机失败，返回主菜单
                else {
                    set_textarea(key_event, global_state, w_textarea_main, L"安全关机失败", 0, 0);
                    draw_textarea(key_event, global_state, w_textarea_main);

                    sleep_in_ms(1000);

                    STATE = STATE_MAIN_MENU;
                }
            }

            // 长短按A键取消关机，返回主菜单
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == KEYCODE_NUM_A) {
                STATE = STATE_MAIN_MENU;
            }

            break;


        /////////////////////////////////////////////
        // TTS设置
        /////////////////////////////////////////////

        case STATE_TTS_SETTING:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_menu(key_event, global_state, w_menu_tts_setting);
            }
            PREV_STATE = STATE;

            STATE = menu_event_handler(key_event, global_state, w_menu_tts_setting, tts_setting_menu_item_action, STATE_SETTING_MENU, STATE_TTS_SETTING);

            break;


        /////////////////////////////////////////////
        // ASR设置
        /////////////////////////////////////////////

        case STATE_ASR_SETTING:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_menu(key_event, global_state, w_menu_asr_setting);
            }
            PREV_STATE = STATE;

            STATE = menu_event_handler(key_event, global_state, w_menu_asr_setting, asr_setting_menu_item_action, STATE_SETTING_MENU, STATE_ASR_SETTING);

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

    free(w_textarea_main);
    free(w_textarea_asr);
    free(w_textarea_prefill);

    free(w_input_main);

    free(w_menu_main);
    free(w_menu_model);
    free(w_menu_setting);
    free(w_menu_asr_setting);
    free(w_menu_tts_setting);

    free(void_key_event);

    gfx_close();

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}
