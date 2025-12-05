#include <time.h>
#include <signal.h>
#include <locale.h>
#include <sys/stat.h>

#include "pinyin.h"
#include "oled.h"
#include "ui.h"
#include "ups.h"
#include "keyboard.h"
#include "infer.h"
#include "prompt.h"

#include "platform.h"

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
int32_t g_config_tts_mode = 0; // TTS工作模式：0-关闭   1-实时   2-全部生成后统一TTS

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
// 传递给TTS的文字内容的命名管道
int g_tts_fifo_fd;

#define PTT_FIFO_PATH "/tmp/ptt_fifo"

#define ASR_FIFO_PATH "/tmp/asr_fifo"
#define ASR_BUFFER_SIZE (65536)

#define TTS_FIFO_PATH "/tmp/tts_fifo"
#define TTS_BUFFER_SIZE (4096)


// TTS分句用全局变量
int32_t g_tts_split_from = 0; // 切句子的起始位置


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





// return time in milliseconds, for benchmarking the model speed
long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}




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




// 穷人的ASR服务状态检测：通过读取ASR服务的日志前64kB中是否出现“init finished”来判断
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
        // printf("读取到数据: %s\n", asr_buffer); fflush(stdout);
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

// 向TTS输入FIFO中写文本内容
int32_t write_tts_fifo(char *text_bytes, int32_t len) {
    // 如果没有fifo，创建fifo
    if (mkfifo(TTS_FIFO_PATH, 0666) == -1 && errno != EEXIST) {
        perror("tts fifo mkfifo failed");
        return -1;
    }
    // 以非阻塞写模式打开FIFO
    g_tts_fifo_fd = open(TTS_FIFO_PATH, O_WRONLY | O_NONBLOCK);
    if (g_tts_fifo_fd == -1) {
        perror("open tts fifo for writing failed");
        return -1;
    }

    ssize_t result = write(g_tts_fifo_fd, text_bytes, len);
    if (result == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // FIFO缓冲区满，丢弃数据（不处理）
            // printf("FIFO full, data dropped\n");
        } else {
            perror("write failed");
            close(g_tts_fifo_fd);
            return -1;
        }
    } else {
        // 成功写入
        // printf("Wrote byte: %d\n", (unsigned char)data);
    }
    close(g_tts_fifo_fd);
    return 0;
}


int32_t send_tts_request(wchar_t *text, int32_t is_finished) {
    // TTS切句子
    wchar_t tts_chunk[TTS_BUFFER_SIZE];
    memset(tts_chunk, 0, TTS_BUFFER_SIZE * sizeof(wchar_t));
    int32_t wlen = wcslen(text);
    if (is_finished) {
        wcsncpy(tts_chunk, text + g_tts_split_from, wlen - g_tts_split_from);
        g_tts_split_from = 0;
    }
    else {
        for (int32_t i = g_tts_split_from; i < wlen; i++) {
            if (text[i] == L'，' ||
                text[i] == L'。' ||
                text[i] == L'\n' ||
                text[i] == L'：' ||
                text[i] == L'；' ||
                text[i] == L'？' ||
                text[i] == L'！') {
                if (i - g_tts_split_from > 6) {
                    wcsncpy(tts_chunk, text + g_tts_split_from, i + 1 - g_tts_split_from);
                    g_tts_split_from = i + 1;
                    break;
                }
            }
        }
    }

    // 非阻塞写FIFO
    char text_bytes[TTS_BUFFER_SIZE];
    memset(text_bytes, 0, TTS_BUFFER_SIZE);
    size_t len = wcstombs(text_bytes, tts_chunk, TTS_BUFFER_SIZE);
    if (len <= 0) {
        return -1;
    }
    printf("Write TTS FIFO: %s (%ld)\n", text_bytes, len);

    return write_tts_fifo(text_bytes, len);
}


int32_t stop_tts() {
    return write_tts_fifo("_TTS_STOP_", 11);
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






int32_t on_llm_prefilling(Key_Event *key_event, Global_State *global_state, Nano_Session *session) {
    if (session->t_0 == 0) {
        session->t_0 = time_in_ms();
    }
    else {
        session->tps = (session->pos - 1) / (double)(time_in_ms() - session->t_0) * 1000;
    }

    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
        wcscpy(g_llm_output_of_last_session, L"");
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_PREFILLING;
    }

    // PREFILL_LED_ON

    prefilling_textarea_state->x = 0;
    prefilling_textarea_state->y = 0;
    prefilling_textarea_state->width = 128;
    prefilling_textarea_state->height = 24;

    wcscpy(prefilling_textarea_state->text, L"Pre-filling...");
    prefilling_textarea_state->current_line = 0;
    prefilling_textarea_state->is_show_scroll_bar = 0;

    // 每隔refresh_ratio个token刷新一次屏幕
    if (global_state->timer % global_state->refresh_ratio == 0) {
    
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

    }

    g_tts_split_from = 0;

    // PREFILL_LED_OFF
    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_llm_decoding(Key_Event *key_event, Global_State *global_state, Nano_Session *session) {
    if (session->t_0 == 0) {
        session->t_0 = time_in_ms();
    }
    else {
        session->tps = (session->pos - 1) / (double)(time_in_ms() - session->t_0) * 1000;
    }

    // 长/短按A键中止推理
    if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
        wcscpy(g_llm_output_of_last_session, session->output_text);
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_DECODING;
    }

    // DECODE_LED_ON

    wcscpy(widget_textarea_state->text, session->output_text);
    widget_textarea_state->current_line = -1;
    widget_textarea_state->is_show_scroll_bar = 1;

    // 每隔refresh_ratio个token刷新一次屏幕
    if (global_state->timer % global_state->refresh_ratio == 0) {
        draw_textarea(key_event, global_state, widget_textarea_state);
    }

    // DECODE_LED_OFF

#ifdef TTS_ENABLED
    if (g_config_tts_mode > 0) {
        send_tts_request(session->output_text, 0);
    }
#endif

    free(session->output_text);
    return LLM_RUNNING_IN_DECODING;
}

int32_t on_llm_finished(Nano_Session *session) {
    session->t_1 = time_in_ms();
    session->tps = (session->pos - 1) / (double)(session->t_1 - session->t_0) * 1000;

    wcscpy(g_llm_output_of_last_session, session->output_text);

#ifdef TTS_ENABLED
    if (g_config_tts_mode > 0) {
        send_tts_request(session->output_text, 1);
    }
#endif

    g_tts_split_from = 0;

    g_tps_of_last_session = session->tps;
    printf("TPS = %f\n", session->tps);
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
    if (g_llm_ctx) {
        llm_context_free(g_llm_ctx);
    }

    if (item_index == 0) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Nano-168M-QA\n 请稍等...");
        g_model_path = MODEL_PATH_1;
        g_lora_path = NULL;
        g_repetition_penalty = 1.05f;
        g_temperature = 1.0f;
        g_top_p = 0.5f;
        g_top_k = 0;
        g_max_seq_len = 512;
        global_state->refresh_ratio = 8;
    }
    else if (item_index == 1) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Nano-56M-QA\n 请稍等...");
        g_model_path = MODEL_PATH_2;
        g_lora_path = NULL;
        g_repetition_penalty = 1.05f;
        g_temperature = 1.0f;
        g_top_p = 0.5f;
        g_top_k = 0;
        g_max_seq_len = 512;
        global_state->refresh_ratio = 8;
    }
    else if (item_index == 2) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Nano-56M-Neko\n 请稍等...");
        g_model_path = MODEL_PATH_3;
        g_lora_path = LORA_PATH_3;
        g_repetition_penalty = 1.05f;
        g_temperature = 1.0f;
        g_top_p = 0.5f;
        g_top_k = 0;
        g_max_seq_len = 512;
        global_state->refresh_ratio = 8;
    }
    else if (item_index == 3) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-0.6B\n 请稍等...");
        g_model_path = MODEL_PATH_4;
        g_lora_path = NULL;
        g_repetition_penalty = 1.0f;
        g_temperature = 0.6f;
        g_top_p = 0.95f;
        g_top_k = 20;
        g_max_seq_len = 32768;
        global_state->refresh_ratio = 4;
    }
    else if (item_index == 4) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-1.7B\n 请稍等...");
        g_model_path = MODEL_PATH_5;
        g_lora_path = NULL;
        g_repetition_penalty = 1.0f;
        g_temperature = 0.6f;
        g_top_p = 0.95f;
        g_top_k = 20;
        g_max_seq_len = 32768;
        global_state->refresh_ratio = 1;
    }
    else if (item_index == 5) {
        wcscpy(widget_textarea_state->text, L" 正在加载语言模型\n Qwen3-4B-Inst-2507\n 请稍等...");
        g_model_path = MODEL_PATH_6;
        g_lora_path = NULL;
        g_repetition_penalty = 1.0f;
        g_temperature = 0.7f;
        g_top_p = 0.8f;
        g_top_k = 20;
        g_max_seq_len = 32768;
        global_state->refresh_ratio = 1;
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

int32_t setting_menu_item_action(int32_t item_index) {
    // 语言模型生成参数设置
    if (item_index == 0) {
        wcscpy(widget_textarea_state->text, L"暂未实现");
        widget_textarea_state->current_line = 0;
        widget_textarea_state->is_show_scroll_bar = 0;
        draw_textarea(key_event, global_state, widget_textarea_state);

        usleep(500*1000);

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
    global_state->refresh_ratio = 2; // 默认每2个token刷新一次屏幕

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
    // UPS传感器初始化
#ifdef UPS_ENABLED
    ups_init();
#endif

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
        // 文本显示状态
        /////////////////////////////////////////////

        case -3:

            // 首次获得焦点：初始化
            if (PREV_STATE != STATE) {
                wcscpy(widget_textarea_state->text, g_anniversory);
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
            if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
                if (widget_input_state->state == 0 && widget_input_state->length <= 0) {
                    init_input(key_event, global_state, widget_input_state);
                    STATE = -2;
                }
            }
#ifdef ASR_ENABLED
            // 按下C键：开始PTT
            else if (key_event->key_edge > 0 && key_event->key_code == 12) {
                STATE = 21;
            }
#endif
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
                    wcscpy(widget_input_state->text, get_random_prompt());
                    widget_input_state->length = wcslen(widget_input_state->text);
                }

                // 根据模型类型应用prompt模板
                if (g_llm_ctx->llm->arch == LLM_ARCH_NANO) {
                    wcscat(prompt, L"<|instruct_mark|>");
                    wcscat(prompt, widget_input_state->text);
                    wcscat(prompt, L"<|response_mark|>");
                }
                else if (g_llm_ctx->llm->arch == LLM_ARCH_QWEN2 || g_llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
                    wcscpy(prompt, widget_input_state->text);
                    // wcscat(prompt, L" /no_think");
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
                // 短按A键：停止TTS
                if (key_event->key_edge == -1 && key_event->key_code == 10) {
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
                global_state->asr_start_timestamp = time(NULL);
            }
            PREV_STATE = STATE;

            // 实时显示ASR结果
            if (global_state->is_recording == 1) {
                int32_t len = read_asr_fifo(g_asr_output);

                // 临时关闭draw_textarea的整帧绘制，以便在textarea上绘制进度条之后再统一写入屏幕，否则反复的clear会导致进度条闪烁。
                global_state->is_full_refresh = 0;
                OLED_SoftClear();

                // 显示ASR结果
                // if (len > 0) {
                    wcscpy(asr_textarea_state->text, g_asr_output);
                    asr_textarea_state->current_line = -1;
                    asr_textarea_state->is_show_scroll_bar = 1;
                    draw_textarea(key_event, global_state, asr_textarea_state);
                // }

                // 绘制录音持续时间
                wchar_t rec_duration[50];
                swprintf(rec_duration, 50, L" %ds ", (int32_t)(time(NULL) - global_state->asr_start_timestamp));
                render_line(rec_duration, 0, 52, 0);

                OLED_Refresh();

                // 重新开启整帧绘制，注意这个标记是所有函数共享的全局标记。
                global_state->is_full_refresh = 1;

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

                wcscpy(asr_textarea_state->text, L" \n \n      识别完成");
                asr_textarea_state->current_line = 0;
                asr_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, asr_textarea_state);

                usleep(500*1000);

                wcscpy(widget_input_state->text, g_asr_output);
                widget_input_state->length = wcslen(g_asr_output);

                wcscpy(g_asr_output, L"请说话...");

                // ASR后立刻提交到LLM？
                if (g_config_auto_submit_after_asr) {
                    printf("立刻提交LLM：%ls\n", widget_input_state->text);
                    STATE = 8;
                }
                else {
                    widget_input_state->current_page = 0;
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
                swprintf(readme_buf, 128, L"Nano-Pod v2512\n电子鹦鹉·端上大模型\n(c) 2025 BD4SUR\n\nUPS:%04dmV/%d%% ", global_state->ups_voltage, global_state->ups_soc);
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
            if (key_event->key_edge == -1 && key_event->key_code == 0) {
                g_config_tts_mode = 0;

                wcscpy(widget_textarea_state->text, L"TTS已关闭。");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                usleep(500*1000);

                STATE = 5;
            }

            // 选项1
            else if (key_event->key_edge == -1 && key_event->key_code == 1) {
                g_config_tts_mode = 1;

                wcscpy(widget_textarea_state->text, L"TTS设置为实时请求。");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                usleep(500*1000);

                STATE = 5;
            }

            // 选项2
            else if (key_event->key_edge == -1 && key_event->key_code == 2) {
                g_config_tts_mode = 2;

                wcscpy(widget_textarea_state->text, L"TTS设置为生成结束后统一请求合成。");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                usleep(500*1000);

                STATE = 5;
            }

            // 长短按A键，返回设置菜单
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
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
            if (key_event->key_edge == -1 && key_event->key_code == 0) {
                g_config_auto_submit_after_asr = 0;

                wcscpy(widget_textarea_state->text, L"ASR自动提交已关闭");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                usleep(500*1000);

                STATE = 5;
            }

            // 选项1
            else if (key_event->key_edge == -1 && key_event->key_code == 1) {
                g_config_auto_submit_after_asr = 1;

                wcscpy(widget_textarea_state->text, L"ASR自动提交已开启");
                widget_textarea_state->current_line = 0;
                widget_textarea_state->is_show_scroll_bar = 0;
                draw_textarea(key_event, global_state, widget_textarea_state);

                usleep(500*1000);

                STATE = 5;
            }

            // 长短按A键，返回设置菜单
            else if ((key_event->key_edge == -1 || key_event->key_edge == -2) && key_event->key_code == 10) {
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
            // printf("%d mV | %d%%\n", global_state->ups_voltage, global_state->ups_soc);
#endif
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
