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

#define MODEL_ROOT_DIR "/home/bd4sur/ai/_model/Nano"
// #define MODEL_ROOT_DIR "/emmc/_model"

#define PREFILL_LED_ON  system("echo \"1\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define PREFILL_LED_OFF system("echo \"0\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define DECODE_LED_ON   system("echo \"1\" > /sys/devices/platform/leds/leds/blue:status/brightness");
#define DECODE_LED_OFF  system("echo \"0\" > /sys/devices/platform/leds/leds/blue:status/brightness");

// 推理引擎实例（单例模式）
static Nano_Context *g_llm_ctx;

static char *MODEL_PATH_1 = MODEL_ROOT_DIR "/nano_168m_625000_sft_947000_q80.bin";
static char *MODEL_PATH_2 = MODEL_ROOT_DIR "/nano_56m_99000_sft_v2_200000_q80.bin";
static char *MODEL_PATH_3 = MODEL_ROOT_DIR "/1-基础模型-99000_q80.bin";
static char *LORA_PATH_3  = MODEL_ROOT_DIR "/2-插件-猫娘.bin";
static char *MODEL_PATH_4 = MODEL_ROOT_DIR "/qwen3-0b6-q80.bin";
static char *MODEL_PATH_5 = MODEL_ROOT_DIR "/qwen3-1b7-q80.bin";
static char *MODEL_PATH_6 = MODEL_ROOT_DIR "/qwen3-4b-instruct-2507-q80.bin";

static float g_tps_of_last_session = 0.0f;
static wchar_t g_llm_output_of_last_session[OUTPUT_BUFFER_LENGTH];
static wchar_t g_asr_output[OUTPUT_BUFFER_LENGTH] = L"请说话...";

static wchar_t g_anniversory[OUTPUT_BUFFER_LENGTH] = L"我在博客中，一直回避谈我自己。原因一方面固然是隐私安全考虑，而更重要的原因是，在博客中谈我自己，相当于直面“我是谁”这个终极问题，而我难以回答这个问题，甚至在求索的过程中，只会看到自己的空虚和肤浅。\n\n诸君应该知道，佛经中经常出现“如是我闻”这四个字，意思是“我听说事情是这样的…”。于是我转而回答“我知道什么”，试图迂回说明“什么是我”“什么属于我”，而非径直回答“我是什么”。\n\n一方面，我将个人博客转型为业余电台网站，以电台为载体，来间接呈现它的OP也就是我自己的所见所闻、所思所想。这样的好处是，业余电台是一个比“我”简单得多的系统，介绍“我的电台”，比介绍“我”更容易。电台是一个具象的抓手，可以允许我免于直接回答“我是谁”这个困难的问题。\n\n另一方面，我尽力将我的精神世界区分为“事实”和“观点”两部分，将事实放在“博客”栏目，将观点放在“灵感”栏目。尽管实践中难以明确区分二者，但我依然认为，将思维的依据和思维的结果解耦开来，通过罗列“什么是我”“什么属于我”来渐进式地刻画出我的精神世界的面貌，有助于以超脱的视角来观测我自己，有助于我接近“我是谁”这个问题的答案。\n\n还有一种策略。既然“我是谁”这个问题难以回答，不妨退而求其次，试图回答退化的问题：“我想成为什么样的人”。这个问题实际上包含三个方面，分别是我“想”、我“能”和我“得”。这问题表面上看起来是反思自我，实际上却有很强烈的“外部性”，涉及人作为社会人的价值的评判。\n\n具体而言，为了深刻反思自我，就必须以人为镜，对标他人。想要对标他人，就要了解他人。了解他人，除了了解抽象的他人，还应该了解具体的他人。求解“他是谁”这个问题，似乎比求解“我是谁”这个问题简单一点。既然谈的是博客，那么阅读某人的博客，实际上就是阅读一个“具体的人”。\n\n有人认为，当今网友思维极端化，“二极管思维”盛行，擅长扣帽子、贴标签。但这责任，依我看，也要归咎于许多人并不懂得如何呈现“具体”的自己。许多人活得太抽象，不仅在认识他人的时候太抽象，认识自己的时候也太抽象。人与人之间，都习惯于通过标签和简单归纳来互相认识，这难免产生“二极管思维”。我尽力避免成为这样的人，因此我希望回答好“我是谁”这个问题，呈现一个“具体”的自己。\n\n然而，活得“具体”是很难的。我有个点子，那就是为了观察某人的“专业性”，可以要求他在十秒内说出一句包含很多专业术语的话。一方面，认识具体的人，难免要花不少的时间去与对方交流、相处，也包括阅读他的文章。另一方面，为了让自己活得具体，就要输入足量的具体的事实，输出足量的具体的观点。这也就是说，人要活得“具体”，首先要活得“丰富”。泡利还是谁说过，所谓专家，就是把他所在领域中所有能犯的错误都犯过一遍的人。有了足量的具体细节，才“有资格”发展出自己的“高观点”，从“真懂”到“真信”，实现“我有什么”到“我是什么”的飞跃。\n\n这实际上就是人的认识规律，而且是认识规律的很小但很重要的一方面。这提醒我，要“把手弄脏”，先谈问题，再谈主义。这既是认识他人和世界的方法，也是认识自我的途径。\n\n取乎上得乎中，取乎中得乎下。对标什么人，想成为什么人，能成为什么人，必须要成为什么人。这是人生观的大问题，不可不察。\n";


// 全局设置
int32_t g_config_auto_submit_after_asr = 1; // ASR结束后立刻提交识别内容到LLM

char *g_model_path = NULL;
char *g_lora_path = NULL;
float g_repetition_penalty = 1.05f;
float g_temperature = 1.0f;
float g_top_p = 0.5f;
unsigned int g_top_k = 0;
unsigned long long g_random_seed = 0;
uint32_t g_max_seq_len = 512;


// 传递PTT状态的命名管道
int g_ptt_fifo_fd;
// 传递ASR识别结果的命名管道
int g_asr_fifo_fd;


#define ASR_FIFO_PATH "/tmp/asr_fifo"
#define ASR_BUFFER_SIZE (65536)

#define PTT_FIFO_PATH "/tmp/ptt_fifo"



// 优雅关机
int32_t graceful_shutdown() {
    // 同步所有文件系统数据
    sync();
    // 等待同步完成
    sleep(2);
    // 执行关机
    if (system("sudo poweroff") == -1) {
        perror("关机失败");
        return -1;
    }
    return 0;
}




// 穷人的ASR服务状态检测：通过读取ASR服务的日志前64kB中是否出现“init finished”来判断
#define ASR_SERVER_LOG_PATH "/home/bd4sur/ai/_model/FunASR/log.txt"
int32_t check_asr_server_status() {
    char asr_log_buffer[65536];
    FILE *file = fopen(ASR_SERVER_LOG_PATH, "r");
    if (file == NULL) {
        return -1;
    }
    // 读取最多max_chars个字符
    size_t chars_read = fread(asr_log_buffer, sizeof(char), 65536 - 1, file);
    // 添加字符串结束符
    asr_log_buffer[chars_read] = '\0';
    fclose(file);
    // 查找日志中的模式
    char pattern[] = "asr model init finished. listen on port";
    // 使用strstr查找子字符串
    if (strstr(asr_log_buffer, pattern) != NULL) {
        return 1;
    }
    else {
        return 0;
    }
}





// 以只读方式打开ASR命名管道（非阻塞）
int32_t open_asr_fifo() {
    g_asr_fifo_fd = open(ASR_FIFO_PATH, O_RDONLY | O_NONBLOCK);
    if (g_asr_fifo_fd == -1) {
        perror("打开管道失败");
        return -1;
    }
    printf("管道打开成功，开始读取数据...\n");
    return 0;
}

// 读取ASR管道内容
int32_t read_asr_fifo(wchar_t *asr_text) {
    char asr_buffer[ASR_BUFFER_SIZE];
    memset(asr_buffer, 0, ASR_BUFFER_SIZE);

    ssize_t asr_bytes_read = read(g_asr_fifo_fd, asr_buffer, ASR_BUFFER_SIZE - 1);

    if (asr_bytes_read > 0) {
        asr_buffer[asr_bytes_read] = '\0';
        printf("读取到数据: %s\n", asr_buffer); fflush(stdout);
        mbstowcs(asr_text, asr_buffer, ASR_BUFFER_SIZE);
    }
    else if (asr_bytes_read == 0) {
        // 管道写端关闭，重新打开
        printf("管道写端关闭，重新打开管道...\n");fflush(stdout);
        close(g_asr_fifo_fd);
        g_asr_fifo_fd = open(ASR_FIFO_PATH, O_RDONLY | O_NONBLOCK);
        if (g_asr_fifo_fd == -1) {
            // perror("重新打开管道失败");
            return -1;
        }
    }
    else {
        if (errno != EINTR) {
            // perror("读取管道失败");
        }
        return -1;
    }
    return (int32_t)asr_bytes_read;
}

// 向PTT状态FIFO中写PTT状态
int32_t set_ptt_status(uint8_t status) {
    if (mkfifo(PTT_FIFO_PATH, 0666) == -1 && errno != EEXIST) {
        perror("mkfifo failed");
        return -1;
    }
    // 以非阻塞写模式打开FIFO
    g_ptt_fifo_fd = open(PTT_FIFO_PATH, O_WRONLY | O_NONBLOCK);
    if (g_ptt_fifo_fd == -1) {
        perror("open fifo for writing failed");
        return -1;
    }
    // 尝试写入一个字节
    uint8_t data = status;
    ssize_t result = write(g_ptt_fifo_fd, &data, 1);
    if (result == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // FIFO缓冲区满，丢弃数据（不处理）
            // printf("FIFO full, data dropped\n");
        } else {
            perror("write failed");
            return -1;
        }
    } else {
        // 成功写入
        // printf("Wrote byte: %d\n", (unsigned char)data);
    }
    return 0;
}






int32_t on_prefilling(Nano_Session *session) {
    // 按住A键中止推理
    char key = keyboard_read_key();
    if (key == 10) {
        wcscpy(g_llm_output_of_last_session, L"");
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_PREFILLING;
    }
    // PREFILL_LED_ON
    render_text(L"Pre-filling...", 0);
    OLED_DrawLine(0, 60, 128, 60, 1);
    OLED_DrawLine(0, 63, 128, 63, 1);
    OLED_DrawLine(127, 60, 127, 63, 1);
    OLED_DrawLine(0, 61, session->pos * 128 / (session->num_prompt_tokens - 2), 61, 1);
    OLED_DrawLine(0, 62, session->pos * 128 / (session->num_prompt_tokens - 2), 62, 1);
    OLED_Refresh();
    // PREFILL_LED_OFF
    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_decoding(Nano_Session *session) {
    // 按住A键中止推理
    char key = keyboard_read_key();
    if (key == 10) {
        wcscpy(g_llm_output_of_last_session, session->output_text);
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_DECODING;
    }
    // DECODE_LED_ON
    OLED_SoftClear();
    int32_t line_num = render_text(session->output_text, -1);
    render_scroll_bar(line_num, line_num - 5);
    OLED_Refresh();
    // DECODE_LED_OFF

    free(session->output_text);
    return LLM_RUNNING_IN_DECODING;
}

int32_t on_finished(Nano_Session *session) {
    wcscpy(g_llm_output_of_last_session, session->output_text);

    g_tps_of_last_session = session->tps;
    printf("TPS = %f\n", session->tps);
    return LLM_STOPPED_NORMALLY;
}


int main() {

    if(!setlocale(LC_CTYPE, "")) return -1;

    ///////////////////////////////////////
    // 初始化各类状态

    Global_State           *global_state = (Global_State*)calloc(1, sizeof(Global_State));
    Key_Event              *key_event = (Key_Event*)calloc(1, sizeof(Key_Event));
    Widget_Textarea_State  *widget_textarea_state = (Widget_Textarea_State*)calloc(1, sizeof(Widget_Textarea_State));
    Widget_Input_State     *widget_input_state = (Widget_Input_State*)calloc(1, sizeof(Widget_Input_State));
    Widget_Menu_State      *main_menu_state = (Widget_Menu_State*)calloc(1, sizeof(Widget_Menu_State));


    global_state->is_recording = 0;
    global_state->asr_start_timestamp = 0;

    widget_input_state->state = 0;
    widget_input_state->ime_mode_flag = 0;
    widget_input_state->pinyin_keys = 0;
    widget_input_state->candidates = NULL;
    widget_input_state->candidate_num = 0;
    widget_input_state->candidate_pages = NULL;
    widget_input_state->candidate_page_num = 0;
    widget_input_state->current_page = 0;
    widget_input_state->input_buffer = (uint32_t *)calloc(INPUT_BUFFER_LENGTH, sizeof(uint32_t));
    widget_input_state->input_counter = 0;
    widget_input_state->cursor_pos = 0;

    widget_input_state->alphabet_countdown = -1;
    widget_input_state->alphabet_current_key = 255;
    widget_input_state->alphabet_index = 0;

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

    // 全局状态标志
    int32_t STATE = -1;


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
                printf("key_timer = %d\n", key_event->key_timer);
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
STATE_M1:// 初始状态：欢迎屏幕。按任意键进入主菜单
        /////////////////////////////////////////////

        case -1:

            show_splash_screen(key_event, global_state);

            // 按下任何键，不论长短按，进入主菜单
            if (key_event->key_edge < 0 && key_event->key_code < 16) {
                show_main_menu(key_event, global_state, main_menu_state);
                STATE = -2;
            }

            break;

        /////////////////////////////////////////////
STATE_M2:// 主菜单。
        /////////////////////////////////////////////

        case -2:

            // 短按1键
            if (key_event->key_edge == -1 && key_event->key_code == 1) {
                // 文本卷到顶，渲染
                widget_textarea_state->text = g_anniversory;
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 1;
                draw_textarea(key_event, global_state, widget_textarea_state);
                STATE = -3;
            }

            // 短按2键：进入文本输入就绪状态
            else if (key_event->key_edge == -1 && key_event->key_code == 2) {

                // LLM Init

                if (!g_llm_ctx) {
                    widget_textarea_state->text = L" 正在加载语言模型\n Nano-168M-QA\n 请稍等...";
                    widget_textarea_state->current_line = 0;
                    widget_textarea_state->is_show_scroll_bar = 0;
                    draw_textarea(key_event, global_state, widget_textarea_state);

                    g_model_path = MODEL_PATH_1;
                    g_lora_path = NULL;
                    g_repetition_penalty = 1.05f;
                    g_temperature = 1.0f;
                    g_top_p = 0.5f;
                    g_top_k = 0;
                    g_random_seed = (unsigned int)time(NULL);
                    g_max_seq_len = 512;
                    g_llm_ctx = llm_context_init(g_model_path, g_lora_path, g_max_seq_len, g_repetition_penalty, g_temperature, g_top_p, g_top_k, g_random_seed);

                    widget_textarea_state->text = L"加载完成~";
                    widget_textarea_state->current_line = 0;
                    widget_textarea_state->is_show_scroll_bar = 0;
                    draw_textarea(key_event, global_state, widget_textarea_state);
                    usleep(1000*1000);
                }

                // 刷新文本输入框
                init_input(key_event, global_state, widget_input_state);
                STATE = 0;
            }

            // 短按3键：选择语言模型
            else if (key_event->key_edge == -1 && key_event->key_code == 3) {
                widget_textarea_state->text = L"选择语言模型：\n1. Nano-168M-QA\n2. Nano-56M-QA\n3. Nano-56M-Neko\n4. Qwen3-0.6B";
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                STATE = 4;
            }

            // 短按5键：安全关机
            else if (key_event->key_edge == -1 && key_event->key_code == 5) {
                widget_textarea_state->text = L"正在安全关机...";
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                if (graceful_shutdown() >= 0) {
                    exit(0);
                }
                else {
                    widget_textarea_state->text = L"安全关机失败";
                    widget_textarea_state->current_line = 0;
                    widget_textarea_state->is_show_scroll_bar = 0;
                    draw_textarea(key_event, global_state, widget_textarea_state);

                    usleep(1000*1000);
                }
                show_main_menu(key_event, global_state, main_menu_state);
                STATE = -2;
            }

            // 短按A键：回到splash
            else if (key_event->key_edge == -1 && key_event->key_code == 10) {
                key_event->key_code = 16; // 取消按键状态
                STATE = -1;
                goto STATE_M1;
            }

            break;

        /////////////////////////////////////////////
STATE_M3:// 文本显示状态
        /////////////////////////////////////////////

        case -3:

            // 短按A键：回到主菜单
            if (key_event->key_edge == -1 && key_event->key_code == 10) {
                show_main_menu(key_event, global_state, main_menu_state);
                STATE = -2;
            }

            // 长+短按*键：推理结果向上翻一行。如果翻到顶，则回到最后一行。
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
                if (widget_textarea_state->current_line <= 0) { // 卷到顶
                    widget_textarea_state->current_line = widget_textarea_state->line_num - 5;
                }
                else {
                    widget_textarea_state->current_line--;
                }

                widget_textarea_state->text = g_anniversory;
                widget_textarea_state->is_show_scroll_bar = 1;
                draw_textarea(key_event, global_state, widget_textarea_state);

                STATE = -3;
            }

            // 长+短按#键：推理结果向下翻一行。如果翻到底，则回到第一行。
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
                if (widget_textarea_state->current_line >= (widget_textarea_state->line_num - 5)) { // 卷到底
                    widget_textarea_state->current_line = 0;
                }
                else {
                    widget_textarea_state->current_line++;
                }

                widget_textarea_state->text = g_anniversory;
                widget_textarea_state->is_show_scroll_bar = 1;
                draw_textarea(key_event, global_state, widget_textarea_state);

                STATE = -3;
            }

            break;

        /////////////////////////////////////////////
STATE_0:// 文字编辑器状态
        /////////////////////////////////////////////

        case 0:

            // 长+短按A键：删除一个字符；如果输入缓冲区为空，则回到主菜单
            if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
                if (widget_input_state->state == 0 && widget_input_state->input_counter <= 0) {
                    // widget_input_state->input_buffer = refresh_input_buffer(widget_input_state->input_buffer, &(widget_input_state->input_counter));
                    init_input(key_event, global_state, widget_input_state);
                    show_main_menu(key_event, global_state, main_menu_state);
                    STATE = -2;
                }
            }

            // 按下C键：开始PTT
            else if (key_event->key_edge > 0 && key_event->key_code == 12) {

                // 设置PTT状态为按下（>0）
                if (set_ptt_status(66) < 0) break;

                // 打开ASR管道
                if (open_asr_fifo() < 0) break;

                widget_textarea_state->text = L" \n \n     请说话...";
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                global_state->is_recording = 1;
                global_state->asr_start_timestamp = time(NULL);

                STATE = 21;
                goto STATE_21;
            }

            // 短按D键：提交
            else if (key_event->key_edge == -1 && key_event->key_code == 13) {
                // render_input_buffer(widget_input_state->input_buffer, widget_input_state->ime_mode_flag, -1);
                if (widget_input_state->state == 0) {
                    STATE = 10;
                    goto STATE_10;
                }
            }

            draw_input(key_event, global_state, widget_input_state);

            break;

        /////////////////////////////////////////////
STATE_4:// 选择语言模型状态
        /////////////////////////////////////////////

        case 4:

            // 短按1键
            if (key_event->key_edge == -1 && (key_event->key_code >= 1 && key_event->key_code <= 6)) {
                if (g_llm_ctx) {
                    llm_context_free(g_llm_ctx);
                }

                if (key_event->key_code == 1) {
                    widget_textarea_state->text = L" 正在加载语言模型\n Nano-168M-QA\n 请稍等...";
                    g_model_path = MODEL_PATH_1;
                    g_lora_path = NULL;
                    g_repetition_penalty = 1.05f;
                    g_temperature = 1.0f;
                    g_top_p = 0.5f;
                    g_top_k = 0;
                    g_max_seq_len = 512;
                }
                else if (key_event->key_code == 2) {
                    widget_textarea_state->text = L" 正在加载语言模型\n Nano-56M-QA\n 请稍等...";
                    g_model_path = MODEL_PATH_2;
                    g_lora_path = NULL;
                    g_repetition_penalty = 1.05f;
                    g_temperature = 1.0f;
                    g_top_p = 0.5f;
                    g_top_k = 0;
                    g_max_seq_len = 512;
                }
                else if (key_event->key_code == 3) {
                    widget_textarea_state->text = L" 正在加载语言模型\n Nano-56M-Neko\n 请稍等...";
                    g_model_path = MODEL_PATH_3;
                    g_lora_path = LORA_PATH_3;
                    g_repetition_penalty = 1.05f;
                    g_temperature = 1.0f;
                    g_top_p = 0.5f;
                    g_top_k = 0;
                    g_max_seq_len = 512;
                }
                else if (key_event->key_code == 4) {
                    widget_textarea_state->text = L" 正在加载语言模型\n Qwen3-0.6B\n 请稍等...";
                    g_model_path = MODEL_PATH_4;
                    g_lora_path = NULL;
                    g_repetition_penalty = 1.0f;
                    g_temperature = 0.6f;
                    g_top_p = 0.95f;
                    g_top_k = 20;
                    g_max_seq_len = 32768;
                }
                else if (key_event->key_code == 5) {
                    widget_textarea_state->text = L" 正在加载语言模型\n Qwen3-1.7B\n 请稍等...";
                    g_model_path = MODEL_PATH_5;
                    g_lora_path = NULL;
                    g_repetition_penalty = 1.0f;
                    g_temperature = 0.6f;
                    g_top_p = 0.95f;
                    g_top_k = 20;
                    g_max_seq_len = 32768;
                }
                else if (key_event->key_code == 6) {
                    widget_textarea_state->text = L" 正在加载语言模型\n Qwen3-4B-Inst-2507\n 请稍等...";
                    g_model_path = MODEL_PATH_6;
                    g_lora_path = NULL;
                    g_repetition_penalty = 1.0f;
                    g_temperature = 0.7f;
                    g_top_p = 0.8f;
                    g_top_k = 20;
                    g_max_seq_len = 32768;
                }

                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                g_random_seed = (unsigned int)time(NULL);
                g_llm_ctx = llm_context_init(g_model_path, g_lora_path, g_max_seq_len, g_repetition_penalty, g_temperature, g_top_p, g_top_k, g_random_seed);

                widget_textarea_state->text = L"加载完成~";
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                usleep(500*1000);

                show_main_menu(key_event, global_state, main_menu_state);
                STATE = -2;
            }

            // 短按A键：取消操作，回到主菜单
            else if (key_event->key_edge == -1 && key_event->key_code == 10) {
                show_main_menu(key_event, global_state, main_menu_state);
                STATE = -2;
            }

            break;

        /////////////////////////////////////////////
STATE_10: // 提交候选字到LLM，开始推理
        /////////////////////////////////////////////

        case 10:

            // 短按D键：开始推理。推理完成后，并不清除输入缓冲区，因此再次按D键会重新推理。
            if (key_event->key_edge == -1 && key_event->key_code == 13) {
                OLED_SoftClear();

                wchar_t *prompt;
                if (g_llm_ctx->llm->arch == LLM_ARCH_NANO) {
                    prompt = apply_chat_template(NULL, NULL, widget_input_state->input_buffer);
                }
                else if (g_llm_ctx->llm->arch == LLM_ARCH_QWEN2 || g_llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
                    prompt = widget_input_state->input_buffer;
                }
                else {
                    fprintf(stderr, "Error: unknown model arch.\n");
                    exit(EXIT_FAILURE);
                }

                int32_t flag = generate_sync(g_llm_ctx, prompt, g_max_seq_len, on_prefilling, on_decoding, on_finished);

                if (flag == LLM_STOPPED_IN_PREFILLING || flag == LLM_STOPPED_IN_DECODING) {
                    printf("推理中止。\n");

                    // 按键延时：等待（on_decoding回调中检测到的）按键松开，防止误触发
                    usleep(500 * 1000);

                    // 计算提示语+生成内容的行数
                    wchar_t prompt_and_output[OUTPUT_BUFFER_LENGTH] = L"Homo:\n";
                    wcscat(prompt_and_output, widget_input_state->input_buffer);
                    wcscat(prompt_and_output, L"\n--------------------\nNano:\n");
                    wcscat(prompt_and_output, g_llm_output_of_last_session);
                    wchar_t tps_wcstr[50];
                    swprintf(tps_wcstr, 50, L"\n\n[Nano:推理中止]\n\n[平均速度%.1f词元/秒]", g_tps_of_last_session);
                    wcscat(prompt_and_output, tps_wcstr);

                    wcscpy(g_llm_output_of_last_session, prompt_and_output);

                    widget_textarea_state->text = g_llm_output_of_last_session;
                    int32_t line_num = get_view_lines(widget_textarea_state->text);
                    widget_textarea_state->current_line = (line_num >= 5) ? line_num - 5 : 0;
                    widget_textarea_state->is_show_scroll_bar = 1;
                    draw_textarea(key_event, global_state, widget_textarea_state);

                    STATE = 10;
                }
                else if (flag == LLM_STOPPED_NORMALLY) {
                    printf("推理自然结束。\n");

                    // 计算提示语+生成内容的行数，绘制文本和滚动条
                    OLED_SoftClear();

                    wchar_t prompt_and_output[OUTPUT_BUFFER_LENGTH] = L"Homo:\n";
                    wcscat(prompt_and_output, widget_input_state->input_buffer);
                    wcscat(prompt_and_output, L"\n--------------------\nNano:\n");
                    wcscat(prompt_and_output, g_llm_output_of_last_session);
                    wchar_t tps_wcstr[50];
                    swprintf(tps_wcstr, 50, L"\n\n[平均速度%.1f词元/秒]", g_tps_of_last_session);
                    wcscat(prompt_and_output, tps_wcstr);

                    wcscpy(g_llm_output_of_last_session, prompt_and_output);

                    widget_textarea_state->text = g_llm_output_of_last_session;
                    int32_t line_num = get_view_lines(widget_textarea_state->text);
                    widget_textarea_state->current_line = (line_num >= 5) ? line_num - 5 : 0;
                    widget_textarea_state->is_show_scroll_bar = 1;
                    draw_textarea(key_event, global_state, widget_textarea_state);

                    STATE = 10;
                }
                else {
                    printf("推理过程异常结束。\n");

                    // 计算提示语+生成内容的行数，绘制文本和滚动条
                    OLED_SoftClear();

                    wchar_t prompt_and_output[OUTPUT_BUFFER_LENGTH] = L"Homo:\n";
                    wcscat(prompt_and_output, widget_input_state->input_buffer);
                    wcscat(prompt_and_output, L"\n--------------------\nNano:\n");
                    wcscat(prompt_and_output, g_llm_output_of_last_session);
                    wchar_t tps_wcstr[50];
                    swprintf(tps_wcstr, 50, L"\n\n推理过程异常结束！\n\n[平均速度%.1f词元/秒]", g_tps_of_last_session);
                    wcscat(prompt_and_output, tps_wcstr);

                    wcscpy(g_llm_output_of_last_session, prompt_and_output);

                    widget_textarea_state->text = g_llm_output_of_last_session;
                    int32_t line_num = get_view_lines(widget_textarea_state->text);
                    widget_textarea_state->current_line = (line_num >= 5) ? line_num - 5 : 0;
                    widget_textarea_state->is_show_scroll_bar = 1;
                    draw_textarea(key_event, global_state, widget_textarea_state);

                    STATE = 10;
                }
            }

            // 短按A键：清屏，清除输入缓冲区，回到初始状态
            else if (key_event->key_edge == -1 && key_event->key_code == 10) {
                // 刷新文本输入框
                init_input(key_event, global_state, widget_input_state);
                STATE = 0;
            }

            // 长+短按*键：推理结果向上翻一行。如果翻到顶，则回到最后一行。
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 14) {
                if (widget_textarea_state->current_line <= 0) { // 卷到顶
                    widget_textarea_state->current_line = widget_textarea_state->line_num - 5;
                }
                else {
                    widget_textarea_state->current_line--;
                }

                widget_textarea_state->text = g_llm_output_of_last_session;
                widget_textarea_state->is_show_scroll_bar = 1;
                draw_textarea(key_event, global_state, widget_textarea_state);

                STATE = 10;
            }

            // 长+短按#键：推理结果向下翻一行。如果翻到底，则回到第一行。
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 15) {
                if (widget_textarea_state->current_line >= (widget_textarea_state->line_num - 5)) { // 卷到底
                    widget_textarea_state->current_line = 0;
                }
                else {
                    widget_textarea_state->current_line++;
                }

                widget_textarea_state->text = g_llm_output_of_last_session;
                widget_textarea_state->is_show_scroll_bar = 1;
                draw_textarea(key_event, global_state, widget_textarea_state);

                STATE = 10;
            }

            break;


        /////////////////////////////////////////////
STATE_21: // ASR实时识别进行中（响应ASR客户端回报的ASR文本内容）
        /////////////////////////////////////////////

        case 21:

            // 实时显示ASR结果
            if (global_state->is_recording == 1) {
                int32_t len = read_asr_fifo(g_asr_output);
                // if (len > 0) {
                    // 显示录音持续时间
                    wchar_t asr_text_with_duration[ASR_BUFFER_SIZE] = L"";
                    wcscat(asr_text_with_duration, g_asr_output);
                    wchar_t rec_duration[50];
                    swprintf(rec_duration, 50, L"\n[ %d s ]", (int32_t)(time(NULL) - global_state->asr_start_timestamp));
                    wcscat(asr_text_with_duration, rec_duration);

                    widget_textarea_state->text = asr_text_with_duration;
                    widget_textarea_state->current_line = -1;
                    widget_textarea_state->is_show_scroll_bar = 1;
                    draw_textarea(key_event, global_state, widget_textarea_state);
                // }
            }

            // 松开按钮，停止PTT
            if (global_state->is_recording > 0 && key_event->key_edge == 0 && key_event->key_code == 16) {
                printf("松开PTT\n");
                global_state->is_recording = 0;
                global_state->asr_start_timestamp = 0;

                close(g_asr_fifo_fd);

                // // 设置PTT状态为松开（==0）
                if (set_ptt_status(0) < 0) break;
                close(g_ptt_fifo_fd);

                widget_textarea_state->text = L" \n \n     识别完成";
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                usleep(500*1000);

                wcscpy(widget_input_state->input_buffer, g_asr_output);
                widget_input_state->input_counter = wcslen(g_asr_output);
                wcscpy(g_asr_output, L"请说话...");

                // ASR后立刻提交到LLM？
                if (g_config_auto_submit_after_asr) {
                    printf("立刻提交LLM：%ls\n", widget_input_state->input_buffer);
                    // 软触发D键
                    key_event->key_edge = -1;
                    key_event->key_code = 13;
                    STATE = 10;
                    goto STATE_10;
                }
                else {
                    // 回到文字输widget_input_state->current_page = 0;
                    STATE = 0;
                }

            }

            // 短按A键：清屏，清除输入缓冲区，回到初始状态
            else if (key_event->key_edge == -1 && key_event->key_code == 10) {
                // 刷新文本输入框
                init_input(key_event, global_state, widget_input_state);
                STATE = 0;
            }

            break;

        default:
            break;
        }


        // 绘制焦点组件
        // if (STATE == 0) {
        //     draw_input(key_event, global_state, widget_input_state);
        // }










        /////////////////////////////////////////////
        // 英文字母输入模式的循环切换
        /////////////////////////////////////////////
        // draw_input(void_key_event, global_state, widget_input_state);

        // 定期检查ASR服务状态
        if (global_state->timer % 100 == 0) {
            global_state->is_asr_server_up = check_asr_server_status();
            printf("ASR Service = %d\n", global_state->is_asr_server_up);
        }

        global_state->timer = (global_state->timer == 2147483647) ? 0 : (global_state->timer + 1);
    }

    llm_context_free(g_llm_ctx);

    free(global_state);
    free(key_event);
    free(widget_textarea_state);
    free(widget_input_state);
    free(main_menu_state);

    free(void_key_event);

    OLED_Close();

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}
