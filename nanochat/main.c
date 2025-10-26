#include <time.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "avltree.h"
#include "pinyin.h"
#include "oled.h"
#include "ui.h"
#include "keyboard.h"
#include "infer.h"

// 推理引擎实例（单例模式）
static Nano_Context *g_llm_ctx;

char bin_path[16384];

static char *MODEL_PATH_1 = "/nano_168m_625000_sft_947000_q80.bin";
static char *MODEL_PATH_4 = "/qwen3-0b6-q80.bin";
static char *MODEL_PATH_5 = "/qwen3-1b7-q80.bin";
static char *MODEL_PATH_6 = "/qwen3-4b-instruct-2507-q80.bin";

static float g_tps_of_last_session = 0.0f;
static wchar_t g_llm_output_of_last_session[OUTPUT_BUFFER_LENGTH];


// 全局设置
int32_t g_config_auto_submit_after_asr = 1; // ASR结束后立刻提交识别内容到LLM
int32_t g_config_tts_mode = 1; // TTS工作模式：0-关闭   1-实时   2-全部生成后统一TTS

char *g_model_path = NULL;
char *g_lora_path = NULL;
float g_repetition_penalty = 1.05f;
float g_temperature = 1.0f;
float g_top_p = 0.5f;
unsigned int g_top_k = 0;
unsigned long long g_random_seed = 0;
uint32_t g_max_seq_len = 512;


///////////////////////////////////////
// 全局GUI组件对象

Global_State           *global_state;
Key_Event              *key_event;
Widget_Textarea_State  *widget_textarea_state;
Widget_Textarea_State  *asr_textarea_state;
Widget_Textarea_State  *prefilling_textarea_state;
Widget_Input_State     *widget_input_state;
Widget_Menu_State      *main_menu_state;
Widget_Menu_State      *model_menu_state;
Widget_Menu_State      *setting_menu_state;

// 全局状态标志
int32_t STATE = -1;
int32_t PREV_STATE = -1;



// 优雅关机
int32_t graceful_shutdown() {
    // 同步所有文件系统数据
    sync();
    // 等待同步完成
    sleep(2);
    // 执行关机
    if (system("sudo shutdown -h now") == -1) {
        perror("关机失败");
        return -1;
    }
    return 0;
}


// 获取可执行文件所在目录
int get_executable_dir(char *dir_path) {
    char exe_path[16384];

    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        return -1;
    }
    exe_path[len] = '\0';

    char *last_slash = strrchr(exe_path, '/');
    if (last_slash == NULL) {
        return -1;
    }
    *last_slash = '\0';

    strcpy(dir_path, exe_path);
    return 0;
}


int32_t on_llm_prefilling(Key_Event *key_event, Global_State *global_state, Nano_Session *session) {
    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
        wcscpy(g_llm_output_of_last_session, L"");
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_PREFILLING;
    }

    prefilling_textarea_state->x = 0;
    prefilling_textarea_state->y = 0;
    prefilling_textarea_state->width = 128;
    prefilling_textarea_state->height = 24;

    wcscpy(prefilling_textarea_state->text, L"Pre-filling...");
    prefilling_textarea_state->current_line = 0;
    prefilling_textarea_state->is_show_scroll_bar = 0;

    // 临时关闭draw_textarea的整帧绘制，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
    global_state->is_full_refresh = 0;

    OLED_SoftClear();

    draw_textarea(key_event, global_state, prefilling_textarea_state);

    OLED_DrawLine(0, 60, 128, 60, 1);
    OLED_DrawLine(0, 63, 128, 63, 1);
    OLED_DrawLine(127, 60, 127, 63, 1);
    OLED_DrawLine(0, 61, session->pos * 128 / (session->num_prompt_tokens - 2), 61, 1);
    OLED_DrawLine(0, 62, session->pos * 128 / (session->num_prompt_tokens - 2), 62, 1);

    OLED_Refresh();

    // 重新开启整帧绘制，注意这个标记是所有函数共享的全局标记。
    global_state->is_full_refresh = 1;

    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_llm_decoding(Key_Event *key_event, Global_State *global_state, Nano_Session *session) {
    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
        wcscpy(g_llm_output_of_last_session, session->output_text);
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_DECODING;
    }

    wcscpy(widget_textarea_state->text, session->output_text);
    widget_textarea_state->current_line = -1;
    widget_textarea_state->is_show_scroll_bar = 1;
    draw_textarea(key_event, global_state, widget_textarea_state);

    free(session->output_text);
    return LLM_RUNNING_IN_DECODING;
}

int32_t on_llm_finished(Nano_Session *session) {
    wcscpy(g_llm_output_of_last_session, session->output_text);

    g_tps_of_last_session = session->tps;
    printf("TPS = %f\n", session->tps);
    return LLM_STOPPED_NORMALLY;
}


///////////////////////////////////////
// 全局组件操作过程

void init_main_menu() {
    wcscpy(main_menu_state->title, L"NanoChat V2510");
    wcscpy(main_menu_state->items[0], L"电子鹦鹉");
    wcscpy(main_menu_state->items[1], L"安全关机");
    wcscpy(main_menu_state->items[2], L"本机自述");
    main_menu_state->item_num = 3;
    init_menu(key_event, global_state, main_menu_state);
}

void init_model_menu() {
    wcscpy(model_menu_state->title, L"Select LLM");
    wcscpy(model_menu_state->items[0], L"Qwen3-0.6B");
    wcscpy(model_menu_state->items[1], L"Qwen3-1.7B");
    wcscpy(model_menu_state->items[2], L"Qwen3-4B-Inst-2507");
    wcscpy(model_menu_state->items[3], L"Nano-168M-QA");
    model_menu_state->item_num = 4;
    init_menu(key_event, global_state, model_menu_state);
}

void refresh_model_menu() {
    draw_menu(key_event, global_state, model_menu_state);
}



///////////////////////////////////////
// 事件处理回调
//   NOTE 现阶段，回调函数里面引用的都是全局变量。后续可以container或者ctx之类的参数形式传进去。

// 通用的菜单事件处理
int32_t menu_event_handler(
    Key_Event *ke, Global_State *gs, Widget_Menu_State *ms,
    int32_t (*menu_item_action)(int32_t), int32_t prev_focus, int32_t current_focus
) {
    // 短按0-9数字键：直接选中屏幕上显示的那页的相对第几项
    if (ke->key_edge == -1 && (ke->key_code >= 0 && ke->key_code <= 9)) {
        if (ke->key_code < ms->item_num) {
            ms->current_item_intex = ms->first_item_intex + (uint32_t)(ke->key_code) - 1;
        }
        return menu_item_action(ms->current_item_intex);
    }
    // 短按A键：返回上一个焦点状态
    else if (ke->key_edge == -1 && ke->key_code == 10) {
        return prev_focus;
    }
    // 短按D键：执行菜单项对应的功能
    else if (ke->key_edge == -1 && ke->key_code == 13) {
        return menu_item_action(ms->current_item_intex);
    }
    // 长+短按*键：光标向上移动
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == 14) {
        if (ms->first_item_intex == 0 && ms->current_item_intex == 0) {
            ms->first_item_intex = ms->item_num - ms->items_per_page;
            ms->current_item_intex = ms->item_num - 1;
        }
        else if (ms->current_item_intex == ms->first_item_intex) {
            ms->first_item_intex--;
            ms->current_item_intex--;
        }
        else {
            ms->current_item_intex--;
        }

        draw_menu(ke, gs, ms);

        return current_focus;
    }
    // 长+短按#键：光标向下移动
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == 15) {
        if (ms->first_item_intex == ms->item_num - ms->items_per_page && ms->current_item_intex == ms->item_num - 1) {
            ms->first_item_intex = 0;
            ms->current_item_intex = 0;
        }
        else if (ms->current_item_intex == ms->first_item_intex + ms->items_per_page - 1) {
            ms->first_item_intex++;
            ms->current_item_intex++;
        }
        else {
            ms->current_item_intex++;
        }

        draw_menu(ke, gs, ms);

        return current_focus;
    }

    return current_focus;
}


// 通用的文本框卷行事件处理
int32_t textarea_event_handler(
    Key_Event *ke, Global_State *gs, Widget_Textarea_State *ts,
    int32_t prev_focus, int32_t current_focus
) {
    // 短按A键：回到上一个焦点
    if (ke->key_edge == -1 && ke->key_code == 10) {
        return prev_focus;
    }

    // 长+短按*键：推理结果向上翻一行。如果翻到顶，则回到最后一行。
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == 14) {
        if (ts->current_line <= 0) { // 卷到顶
            ts->current_line = ts->line_num - 5;
        }
        else {
            ts->current_line--;
        }

        draw_textarea(ke, gs, ts);

        return current_focus;
    }

    // 长+短按#键：推理结果向下翻一行。如果翻到底，则回到第一行。
    else if ((ke->key_edge == -1 || ke->key_edge == -2) && ke->key_code == 15) {
        if (ts->current_line >= (ts->line_num - 5)) { // 卷到底
            ts->current_line = 0;
        }
        else {
            ts->current_line++;
        }

        draw_textarea(ke, gs, ts);

        return current_focus;
    }

    return current_focus;
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
    // 1.安全关机
    else if (item_index == 1) {
        return 31;
    }
    // 2.本机自述
    else if (item_index == 2) {
        return 26;
    }
    return -2;
}

int32_t model_menu_item_action(int32_t item_index) {
    if (g_llm_ctx) {
        llm_context_free(g_llm_ctx);
    }

    char model_path[16384];

    if (item_index == 0) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-0.6B\n 请稍等...");
        strcpy(model_path, bin_path);
        strcat(model_path, "/model");
        strcat(model_path, MODEL_PATH_4);
        g_model_path = model_path;
        g_lora_path = NULL;
        g_repetition_penalty = 1.0f;
        g_temperature = 0.6f;
        g_top_p = 0.95f;
        g_top_k = 20;
        g_max_seq_len = 32768;
    }
    else if (item_index == 1) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-1.7B\n 请稍等...");
        strcpy(model_path, bin_path);
        strcat(model_path, "/model");
        strcat(model_path, MODEL_PATH_5);
        g_model_path = model_path;
        g_lora_path = NULL;
        g_repetition_penalty = 1.0f;
        g_temperature = 0.6f;
        g_top_p = 0.95f;
        g_top_k = 20;
        g_max_seq_len = 32768;
    }
    else if (item_index == 2) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-4B-Inst-2507\n 请稍等...");
        strcpy(model_path, bin_path);
        strcat(model_path, "/model");
        strcat(model_path, MODEL_PATH_6);
        g_model_path = model_path;
        g_lora_path = NULL;
        g_repetition_penalty = 1.0f;
        g_temperature = 0.7f;
        g_top_p = 0.8f;
        g_top_k = 20;
        g_max_seq_len = 32768;
    }
    else if (item_index == 3) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Nano-168M-QA\n 请稍等...");
        strcpy(model_path, bin_path);
        strcat(model_path, "/model");
        strcat(model_path, MODEL_PATH_1);
        g_model_path = model_path;
        g_lora_path = NULL;
        g_repetition_penalty = 1.05f;
        g_temperature = 1.0f;
        g_top_p = 0.5f;
        g_top_k = 0;
        g_max_seq_len = 512;
    }

    widget_textarea_state->current_line = 0;
    widget_textarea_state->is_show_scroll_bar = 0;
    draw_textarea(key_event, global_state, widget_textarea_state);

    g_random_seed = (unsigned int)time(NULL);
    g_llm_ctx = llm_context_init(g_model_path, g_lora_path, g_max_seq_len, g_repetition_penalty, g_temperature, g_top_p, g_top_k, g_random_seed);

    wcscpy(widget_textarea_state->text, L"加载完成~");
    widget_textarea_state->current_line = 0;
    widget_textarea_state->is_show_scroll_bar = 0;
    draw_textarea(key_event, global_state, widget_textarea_state);

    usleep(500*1000);

    // 以下两条路选一个：

    // 1、直接进入电子鹦鹉
    init_input(key_event, global_state, widget_input_state);
    return 0;

    // 2、或者回到主菜单
    // refresh_menu(key_event, global_state, main_menu_state);
    // return -2;
}




int main() {

    // 获取可执行文件的路径
    get_executable_dir(bin_path);

    if(!setlocale(LC_CTYPE, "")) return -1;

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
    global_state->is_recording = 0;
    global_state->asr_start_timestamp = 0;
    global_state->is_full_refresh = 1;

    widget_textarea_state->x = 0;
    widget_textarea_state->y = 0;
    widget_textarea_state->width = 128;
    widget_textarea_state->height = 64;
    widget_textarea_state->line_num = 0;
    widget_textarea_state->current_line = 0;

    key_event->key_code = 16;  // 大于等于16为没有任何按键，0-15为按键
    key_event->key_edge = 0;   // 0：松开  1：上升沿  -1：下降沿(短按结束)  -2：下降沿(长按结束)
    key_event->key_timer = 0;  // 按下计时器
    key_event->key_mask = 0;   // 长按超时后，键盘软复位标记。此时虽然物理上依然按键，只要软复位标记为1，则认为是无按键，无论是边沿还是按住都不触发。直到物理按键松开后，软复位标记清0。
    key_event->key_repeat = 0; // 触发一次长按后，只要不松手，该标记置1，直到物理按键松开后置0。若该标记为1，则在按住时触发连续重复动作。

    // 空按键状态：用于定时器事件
    Key_Event *void_key_event = (Key_Event*)calloc(1, sizeof(Key_Event));
    void_key_event->key_code = 16;
    void_key_event->key_edge = 0;
    void_key_event->key_timer = 0;
    void_key_event->key_mask = 0;
    void_key_event->key_repeat = 0;

    ///////////////////////////////////////
    // OLED 初始化

    OLED_Init();
    OLED_Clear();

    show_splash_screen(key_event, global_state);

    ///////////////////////////////////////
    // 矩阵按键初始化与读取

    if(keyboard_init() < 0) return -1;
    key_event->prev_key = 16;


    while (1) {
        char key = keyboard_read_key();
        // 边沿
        if (key_event->key_mask != 1 && (key != key_event->prev_key)) {
            // 按下瞬间（上升沿）
            if (key != 16) {
                key_event->key_code = key;
                key_event->key_edge = 1;
                key_event->key_timer = 0;
            }
            // 松开瞬间（下降沿）
            else {
                key_event->key_code = key_event->prev_key;
                // 短按（或者通过长按触发重复动作状态后反复触发）
                if (key_event->key_repeat == 1 || (key_event->key_timer >= 0 && key_event->key_timer < LONG_PRESS_THRESHOLD)) {
                    key_event->key_edge = -1;
                    key_event->key_timer = 0;
                }
                // 长按
                else if (key_event->key_timer >= LONG_PRESS_THRESHOLD) {
                    key_event->key_edge = -2;
                    key_event->key_timer = 0;
                    key_event->key_repeat = 1;
                }
            }
        }
        // 按住或松开
        else {
            // 按住
            if (key != 16) {
                key_event->key_code = key;
                key_event->key_edge = 0;
                key_event->key_timer++;
                // 若重复动作标记key_repeat在一次长按后点亮，则继续按住可以反复触发短按
                if (key_event->key_repeat == 1) {
                    key_event->key_edge = -2;
                    key_event->key_mask = 1; // 软复位置1，即强制恢复为无按键状态，以便下一次轮询检测到下降沿（尽管物理上有键按下），触发长按事件
                    key = 16; // 便于后面设置prev_key为16（无键按下）
                    key_event->key_repeat = 1;
                }
                // 如果没有点亮动作标记key_repeat，则达到长按阈值后触发长按事件
                else if (key_event->key_timer >= LONG_PRESS_THRESHOLD) {
                    // printf("按住超时触发长按：%d，计时=%d，key_mask=%d\n", (int)key, key_event->key_timer, (int)key_event->key_mask);
                    key_event->key_edge = -2;
                    key_event->key_mask = 1; // 软复位置1，即强制恢复为无按键状态，以便下一次轮询检测到下降沿（尽管物理上有键按下），触发长按事件
                    key = 16; // 便于后面设置prev_key为16（无键按下）
                }
            }
            // 松开
            else {
                key_event->key_code = 16;
                key_event->key_edge = 0;
                key_event->key_timer = 0;
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
            if (key_event->key_edge < 0 && key_event->key_code < 16) {
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
        // 文字编辑器状态
        /////////////////////////////////////////////

        case 0:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                refresh_input(key_event, global_state, widget_input_state);
            }
            PREV_STATE = STATE;

            // 长+短按A键：删除一个字符；如果输入缓冲区为空，则回到主菜单
            if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
                if (widget_input_state->state == 0 && widget_input_state->length <= 0) {
                    init_input(key_event, global_state, widget_input_state);
                    STATE = -2;
                }
            }

            // 短按D键：提交
            else if (key_event->key_edge == -1 && key_event->key_code == 13) {
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
        // 语言推理进行中（异步，每个iter结束后会将控制权交还事件循环，而非自行阻塞到最后一个token）
        //   实际上就是将generate_sync的while循环打开，将其置于大的事件循环。
        /////////////////////////////////////////////

        case 8:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                wchar_t *prompt = NULL;
                // 首先根据模型类型应用prompt模板
                if (g_llm_ctx->llm->arch == LLM_ARCH_NANO) {
                    prompt = apply_chat_template(NULL, NULL, widget_input_state->text);
                }
                else if (g_llm_ctx->llm->arch == LLM_ARCH_QWEN2 || g_llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
                    prompt = widget_input_state->text;
                }
                else {
                    fprintf(stderr, "Error: unknown model arch.\n");
                    exit(EXIT_FAILURE);
                }

                // 初始化对话session
                global_state->llm_session = llm_session_init(g_llm_ctx, prompt, g_max_seq_len);
            }
            PREV_STATE = STATE;

            // 事件循环主体：即同步版本的while(1)的循环体

            global_state->llm_status = llm_session_step(g_llm_ctx, global_state->llm_session);
            global_state->llm_session->tps = (global_state->llm_session->pos - 1) / (double)(time_in_ms() - global_state->llm_session->t_0) * 1000;
            if (global_state->llm_status == LLM_RUNNING_IN_PREFILLING) {
                global_state->llm_status = on_llm_prefilling(key_event, global_state, global_state->llm_session);
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
                global_state->llm_status = on_llm_decoding(key_event, global_state, global_state->llm_session);
                // 外部被动中止
                if (global_state->llm_status == LLM_STOPPED_IN_DECODING) {
                    llm_session_free(global_state->llm_session);
                    STATE = 10;
                }
                else {
                    STATE = 8;
                }
            }
            else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {
                global_state->llm_session->t_1 = time_in_ms();
                global_state->llm_session->tps = (global_state->llm_session->pos - 1) / (double)(global_state->llm_session->t_1 - global_state->llm_session->t_0) * 1000;
                global_state->llm_status = on_llm_finished(global_state->llm_session);
                llm_session_free(global_state->llm_session);
                STATE = 10;
            }
            else {
                global_state->llm_status = LLM_STOPPED_WITH_ERROR;
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
                wchar_t prompt_and_output[OUTPUT_BUFFER_LENGTH] = L"Homo:\n";
                wcscat(prompt_and_output, widget_input_state->text);
                wcscat(prompt_and_output, L"\n--------------------\nNano:\n");
                wcscat(prompt_and_output, g_llm_output_of_last_session);
                if (global_state->llm_status == LLM_STOPPED_IN_PREFILLING || global_state->llm_status == LLM_STOPPED_IN_DECODING) {
                    printf("推理中止。\n");
                    wcscat(prompt_and_output, L"\n\n[Nano:推理中止]");
                }
                else if (global_state->llm_status == LLM_STOPPED_NORMALLY) {
                    printf("推理自然结束。\n");
                }
                else {
                    printf("推理异常结束。\n");
                    wcscat(prompt_and_output, L"\n\n[Nano:推理异常结束]");
                }
                wchar_t tps_wcstr[50];
                swprintf(tps_wcstr, 50, L"\n\n[平均速度%.1f词元/秒]", g_tps_of_last_session);
                wcscat(prompt_and_output, tps_wcstr);

                wcscpy(g_llm_output_of_last_session, prompt_and_output);

                wcscpy(widget_textarea_state->text, g_llm_output_of_last_session);
                widget_textarea_state->current_line = -1;
                widget_textarea_state->is_show_scroll_bar = 1;
                draw_textarea(key_event, global_state, widget_textarea_state);
            }
            PREV_STATE = STATE;

            // 短按D键：重新推理。推理完成后，并不清除输入缓冲区，因此再次按D键会重新推理。
            if (key_event->key_edge == -1 && key_event->key_code == 13) {
                STATE = 8;
            }
            else {
                STATE = textarea_event_handler(key_event, global_state, widget_textarea_state, 0, 10);
            }

            break;


        /////////////////////////////////////////////
        // 本机自述
        /////////////////////////////////////////////

        case 26:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {

            }
            PREV_STATE = STATE;

            wchar_t readme_buf[128];
            // 节流
            if (global_state->timer % 200 == 0) {
                swprintf(readme_buf, 128, L"Project Nano v2510\n电子鹦鹉·端上大模型\n(c) 2025 BD4SUR\n\nhttps://bd4sur.com");
                wcscpy(widget_textarea_state->text, readme_buf);
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);
            }

            // 按A键返回主菜单
            if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
                STATE = -2;
            }

            break;


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
            if (key_event->key_edge == -2 && key_event->key_code == 13) {
                wcscpy(widget_textarea_state->text, L" \n \n    正在安全关机...");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                if (graceful_shutdown() >= 0) {
                    exit(0);
                }
                // 关机失败，返回主菜单
                else {
                    wcscpy(widget_textarea_state->text, L"安全关机失败");
                    widget_textarea_state->current_line = 0;
                    widget_textarea_state->is_show_scroll_bar = 0;
                    draw_textarea(key_event, global_state, widget_textarea_state);

                    usleep(1000*1000);

                    STATE = -2;
                }
            }

            // 长短按A键取消关机，返回主菜单
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
                STATE = -2;
            }

            break;


        default:
            break;
        }


        global_state->timer = (global_state->timer == 2147483647) ? 0 : (global_state->timer + 1);
    }

    llm_context_free(g_llm_ctx);

    free(global_state);
    free(key_event);
    free(widget_textarea_state);
    free(widget_input_state);
    free(main_menu_state);
    free(model_menu_state);
    free(prefilling_textarea_state);

    free(void_key_event);

    OLED_Close();

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}
