#include <signal.h>

#include "avltree.h"
#include "pinyin.h"
#include "oled.h"
#include "ui.h"
#include "keyboard.h"
#include "infer.h"


#define ALPHABET_COUNTDOWN_MAX (30)
#define LONG_PRESS_THRESHOLD (360)

#define PREFILL_LED_ON  system("echo \"1\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define PREFILL_LED_OFF system("echo \"0\" > /sys/devices/platform/leds/leds/green:status/brightness");
#define DECODE_LED_ON   system("echo \"1\" > /sys/devices/platform/leds/leds/blue:status/brightness");
#define DECODE_LED_OFF  system("echo \"0\" > /sys/devices/platform/leds/leds/blue:status/brightness");

// 推理引擎实例（单例模式）
static Nano_Context *g_llm_ctx;

static char *MODEL_PATH_1 = "/emmc/_model/nano_168m_625000_sft_947000_q80.bin";
static char *MODEL_PATH_2 = "/emmc/_model/nano_56m_99000_sft_v2_200000_q80.bin";
static char *MODEL_PATH_3 = "/emmc/_model/1-基础模型-99000_q80.bin";
static char *LORA_PATH_3  = "/emmc/_model/2-插件-猫娘.bin";
static char *MODEL_PATH_4 = "/emmc/_model/qwen3-0b6-q80.bin";
static char *MODEL_PATH_5 = "/emmc/_model/qwen3-1b7-q80.bin";

static float g_tps_of_last_session = 0.0f;
static wchar_t g_output_of_last_session[OUTPUT_BUFFER_LENGTH];

static wchar_t g_anniversory[OUTPUT_BUFFER_LENGTH] = L"我在博客中，一直回避谈我自己。原因一方面固然是隐私安全考虑，而更重要的原因是，在博客中谈我自己，相当于直面“我是谁”这个终极问题，而我难以回答这个问题，甚至在求索的过程中，只会看到自己的空虚和肤浅。\n\n诸君应该知道，佛经中经常出现“如是我闻”这四个字，意思是“我听说事情是这样的…”。于是我转而回答“我知道什么”，试图迂回说明“什么是我”“什么属于我”，而非径直回答“我是什么”。\n\n一方面，我将个人博客转型为业余电台网站，以电台为载体，来间接呈现它的OP也就是我自己的所见所闻、所思所想。这样的好处是，业余电台是一个比“我”简单得多的系统，介绍“我的电台”，比介绍“我”更容易。电台是一个具象的抓手，可以允许我免于直接回答“我是谁”这个困难的问题。\n\n另一方面，我尽力将我的精神世界区分为“事实”和“观点”两部分，将事实放在“博客”栏目，将观点放在“灵感”栏目。尽管实践中难以明确区分二者，但我依然认为，将思维的依据和思维的结果解耦开来，通过罗列“什么是我”“什么属于我”来渐进式地刻画出我的精神世界的面貌，有助于以超脱的视角来观测我自己，有助于我接近“我是谁”这个问题的答案。\n\n还有一种策略。既然“我是谁”这个问题难以回答，不妨退而求其次，试图回答退化的问题：“我想成为什么样的人”。这个问题实际上包含三个方面，分别是我“想”、我“能”和我“得”。这问题表面上看起来是反思自我，实际上却有很强烈的“外部性”，涉及人作为社会人的价值的评判。\n\n具体而言，为了深刻反思自我，就必须以人为镜，对标他人。想要对标他人，就要了解他人。了解他人，除了了解抽象的他人，还应该了解具体的他人。求解“他是谁”这个问题，似乎比求解“我是谁”这个问题简单一点。既然谈的是博客，那么阅读某人的博客，实际上就是阅读一个“具体的人”。\n\n有人认为，当今网友思维极端化，“二极管思维”盛行，擅长扣帽子、贴标签。但这责任，依我看，也要归咎于许多人并不懂得如何呈现“具体”的自己。许多人活得太抽象，不仅在认识他人的时候太抽象，认识自己的时候也太抽象。人与人之间，都习惯于通过标签和简单归纳来互相认识，这难免产生“二极管思维”。我尽力避免成为这样的人，因此我希望回答好“我是谁”这个问题，呈现一个“具体”的自己。\n\n然而，活得“具体”是很难的。我有个点子，那就是为了观察某人的“专业性”，可以要求他在十秒内说出一句包含很多专业术语的话。一方面，认识具体的人，难免要花不少的时间去与对方交流、相处，也包括阅读他的文章。另一方面，为了让自己活得具体，就要输入足量的具体的事实，输出足量的具体的观点。这也就是说，人要活得“具体”，首先要活得“丰富”。泡利还是谁说过，所谓专家，就是把他所在领域中所有能犯的错误都犯过一遍的人。有了足量的具体细节，才“有资格”发展出自己的“高观点”，从“真懂”到“真信”，实现“我有什么”到“我是什么”的飞跃。\n\n这实际上就是人的认识规律，而且是认识规律的很小但很重要的一方面。这提醒我，要“把手弄脏”，先谈问题，再谈主义。这既是认识他人和世界的方法，也是认识自我的途径。\n\n取乎上得乎中，取乎中得乎下。对标什么人，想成为什么人，能成为什么人，必须要成为什么人。这是人生观的大问题，不可不察。\n";

pid_t record_pid = 0;

#define AUDIO_FILE_NAME "/tmp/nano_audio.wav"

// 启动录音进程
void start_recording() {
    record_pid = fork();
    if(record_pid == 0) {
        char *argv[] = {"arecord", "-f", "dat", "-t", "wav", AUDIO_FILE_NAME, NULL};
        execv("/usr/bin/arecord", argv);
        exit(1);
    }
}

// 停止录音
void stop_recording() {
    if(record_pid > 0) {
        kill(record_pid, SIGTERM);  // 终止录音进程
        record_pid = 0;
    }
}

// 播放录音
void play_recording() {
    char command[1024];
    snprintf(command, sizeof(command), "aplay %s", AUDIO_FILE_NAME);
    system(command);
}

int32_t on_prefilling(Nano_Session *session) {
    // 按住A键中止推理
    char key = keyboard_read_key();
    if (key == 10) {
        wcscpy(g_output_of_last_session, L"");
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
        wcscpy(g_output_of_last_session, session->output_text);
        g_tps_of_last_session = session->tps;
        return LLM_STOPPED_IN_DECODING;
    }
    // DECODE_LED_ON
    OLED_SoftClear();
    int32_t line_num = render_text(session->output_text, 0);
    render_scroll_bar(line_num, line_num - 5);
    OLED_Refresh();
    // DECODE_LED_OFF

    free(session->output_text);
    return LLM_RUNNING_IN_DECODING;
}

int32_t on_finished(Nano_Session *session) {
    wcscpy(g_output_of_last_session, session->output_text);

    g_tps_of_last_session = session->tps;
    printf("TPS = %f\n", session->tps);
    return LLM_STOPPED_NORMALLY;
}


int main() {
    if(!setlocale(LC_CTYPE, "")) return -1;

    ///////////////////////////////////////
    // OLED 初始化

    OLED_Init();
    OLED_Clear();

    ////////////////////////////////////////////////
    // LLM Init

    OLED_SoftClear();
    render_text(L" 正在加载语言模型\n Nano-168M-QA\n 请稍等...", 0);
    OLED_Refresh();
    usleep(500*1000);

    float repetition_penalty = 1.05f;
    float temperature = 1.0f;
    float top_p = 0.5f;
    unsigned int top_k = 0;
    unsigned long long random_seed = (unsigned int)time(NULL);
    uint32_t max_seq_len = 512;

    g_llm_ctx = llm_context_init(MODEL_PATH_1, NULL, max_seq_len, repetition_penalty, temperature, top_p, top_k, random_seed);

    show_splash_screen();

    ///////////////////////////////////////
    // 矩阵按键初始化与读取

    if(keyboard_init() < 0) return -1;
    char prev_key = 16;

    // 全局状态标志
    int32_t STATE = -1;

    // 汉英数输入模式标志
    uint32_t ime_mode_flag = 0; // 0汉字 1英文 2数字

    // 符号列表
    wchar_t symbols[55] = L"，。、？！：；“”‘’（）《》…―～・【】 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    // 按键对应的字母列表
    wchar_t alphabet[10][32] = {L"", L" .,;:?!-/+_=&\"*", L"abcABC", L"defDEF", L"ghiGHI", L"jklJKL", L"mnoMNO", L"pqrsPRQS", L"tuvTUV", L"wxyzWXYZ"};

    // 单字拼音键码暂存
    uint32_t pinyin_keys = 0;

    // 候选字翻页相关
    uint32_t *candidates = NULL;
    uint32_t candidate_num = 0;
    uint32_t **candidate_pages = NULL;
    uint32_t candidate_page_num = 0;
    uint32_t current_page = 0;

    // 全局文字输入缓冲
    uint32_t *input_buffer = (uint32_t *)calloc(INPUT_BUFFER_LENGTH, sizeof(uint32_t)); // 文字输入缓冲区
    int32_t input_counter = 0;

    // 推理结果翻页相关
    int32_t output_line_num = 0;
    int32_t output_shift = 0;

    // 英文字母输入模式的倒计时
    int32_t alphabet_countdown = -1; // 从ALPHABET_COUNTDOWN_MAX开始，每轮主循环后倒数减1，减到0时清除进度条，减到-1意味着英文字母输入状态结束
    char alphabet_current_key = 255;
    uint32_t alphabet_index = 0;

    // 录音状态
    int32_t is_recording = 0;

    // 按键状态
    uint8_t  key_code = 16; // 大于等于16为没有任何按键，0-15为按键
    int8_t   key_edge = 0;  // 0：松开  1：上升沿  -1：下降沿(短按结束)  -2：下降沿(长按结束)
    uint32_t key_timer = 0; // 按下计时器
    uint8_t  key_mask = 0;  // 长按超时后，键盘软复位标记。此时虽然物理上依然按键，只要软复位标记为1，则认为是无按键，无论是边沿还是按住都不触发。直到物理按键松开后，软复位标记清0。

    while (1) {
        char key = keyboard_read_key();
        // 边沿
        if (key_mask != 1 && (key != prev_key)) {
            // 按下瞬间（上升沿）
            if (key != 16) {
                key_code = key;
                key_edge = 1;
                key_timer = 0;
            }
            // 松开瞬间（下降沿）
            else {
                key_code = prev_key;
                // 短按
                if (key_timer >= 0 && key_timer < LONG_PRESS_THRESHOLD) {
                    // printf("下降 短按：%d，计时=%d，key_mask=%d\n", (int)key, key_timer, (int)key_mask);
                    key_edge = -1;
                    key_timer = 0;
                }
                // 长按
                else if (key_timer >= LONG_PRESS_THRESHOLD) {
                    // printf("下降 长按：%d，计时=%d，key_mask=%d\n", (int)key, key_timer, (int)key_mask);
                    key_edge = -2;
                    key_timer = 0;
                }
            }
        }
        // 按住或松开
        else {
            // 按住（按住可以反复触发长按）
            if (key != 16) {
                key_code = key;
                key_edge = 0;
                key_timer++;
                if (key_timer >= LONG_PRESS_THRESHOLD) {
                    // printf("按住超时触发长按：%d，计时=%d，key_mask=%d\n", (int)key, key_timer, (int)key_mask);
                    key_edge = -2;
                    key_timer = 0;
                    key_mask = 1; // 软复位置1，即强制恢复为无按键状态，尽管物理上有键按下
                    key = 16; // 便于后面设置prev_key为16（无键按下）
                }
            }
            // 松开
            else {
                key_code = 16;
                key_edge = 0;
                key_timer = 0;
                key_mask = 0;
            }
        }
        prev_key = key;




        switch(STATE) {

        /////////////////////////////////////////////
STATE_M1:// 初始状态：欢迎屏幕。按任意键进入主菜单
        /////////////////////////////////////////////

        case -1:

            show_splash_screen();

            // 按下任何键，不论长短按，进入主菜单
            if (key_edge < 0 && key_code < 16) {
                show_main_menu();
                STATE = -2;
            }

            break;

        /////////////////////////////////////////////
STATE_M2:// 主菜单。
        /////////////////////////////////////////////

        case -2:

            // 短按1键
            if (key_edge == -1 && key_code == 1) {
                // 先计算文本行数，不实际渲染
                output_line_num = render_text(g_anniversory, 0);

                // 文本卷到顶，渲染
                OLED_SoftClear();
                output_shift = (output_line_num - 5);
                render_text(g_anniversory, output_shift);
                render_scroll_bar(output_line_num, output_line_num - output_shift - 5);
                OLED_Refresh();
                STATE = -3;
            }

            // 短按2键：进入文本输入就绪状态
            else if (key_edge == -1 && key_code == 2) {
                input_buffer = refresh_input_buffer(input_buffer, &input_counter);
                render_input_buffer(input_buffer, ime_mode_flag, 1);
                STATE = 0;
            }

            // 短按A键：回到splash
            else if (key_edge == -1 && key_code == 10) {
                key_code = 16; // 取消按键状态
                STATE = -1;
                goto STATE_M1;
            }

            break;

        /////////////////////////////////////////////
STATE_M3:// 文本显示状态
        /////////////////////////////////////////////

        case -3:

            // 短按A键：回到主菜单
            if (key_edge == -1 && key_code == 10) {
                show_main_menu();
                STATE = -2;
            }

            // 长+短按*键：推理结果向上翻一行。如果翻到顶，则回到最后一行。
            else if ((key_edge == -1 || key_edge == -2) && key_code == 14) {
                if (output_shift == (output_line_num - 5)) { // 卷到顶的卷动量
                    output_shift = 0;
                }
                else {
                    output_shift++;
                }

                OLED_SoftClear();
                render_text(g_anniversory, output_shift);
                render_scroll_bar(output_line_num, output_line_num - output_shift - 5);
                OLED_Refresh();

                STATE = -3;
            }

            // 长+短按#键：推理结果向下翻一行。如果翻到底，则回到第一行。
            else if ((key_edge == -1 || key_edge == -2) && key_code == 15) {
                if (output_shift == 0) {
                    output_shift = (output_line_num - 5); // 卷到顶的卷动量
                }
                else {
                    output_shift--;
                }

                OLED_SoftClear();
                render_text(g_anniversory, output_shift);
                render_scroll_bar(output_line_num, output_line_num - output_shift - 5);
                OLED_Refresh();

                STATE = -3;
            }

            break;

        /////////////////////////////////////////////
STATE_0:// 就绪状态：等待输入拼音/字母/数字，或者将文字输入缓冲区的内容提交给大模型
        /////////////////////////////////////////////

        case 0:

            // 长按0：输入符号
            if (key_edge == -2 && key_code == 0) {
                candidates = (uint32_t *)calloc(54, sizeof(uint32_t));
                for (int i = 0; i < 54; i++) candidates[i] = (uint32_t)symbols[i];
                candidate_pages = candidate_paging(candidates, 54, 10, &candidate_page_num);
                render_symbol_input(candidate_pages, current_page, candidate_page_num);

                current_page = 0;
                STATE = 3;
            }

            // 短按0：数字输入模式下是直接输入0，其余模式无动作
            else if (key_edge == -1 && key_code == 0) {
                if (ime_mode_flag == IME_MODE_NUMBER) {
                    input_buffer[input_counter++] = L'0';
                    render_input_buffer(input_buffer, ime_mode_flag, 1);
                    STATE = 0;
                }
            }

            // 短按1-9：输入拼音/字母/数字，根据输入模式标志，转向不同的状态
            else if (key_edge == -1 && (key_code >= 1 && key_code <= 9)) {
                if (ime_mode_flag == IME_MODE_HANZI) {
                    if (key_code >= 2 && key_code <= 9) { // 仅响应按键2-9；1无动作
                        STATE = 1;
                        goto STATE_1;
                    }
                }
                else if (ime_mode_flag == IME_MODE_NUMBER) {
                    input_buffer[input_counter++] = L'0' + key_code;
                    render_input_buffer(input_buffer, ime_mode_flag, 1);
                    STATE = 0;
                }
                else if (ime_mode_flag == IME_MODE_ALPHABET) {
                    // 如果按键按下时，不是字母切换状态，则开始循环切换，并开始倒计时。
                    if (alphabet_countdown == -1) {
                        alphabet_countdown = ALPHABET_COUNTDOWN_MAX;
                        alphabet_current_key = key_code;
                        alphabet_index = 0;
                    }
                    // 如果按键按下时，倒计时尚未结束，则切换到下一个字母。
                    else if (alphabet_countdown > 0) {
                        alphabet_countdown = ALPHABET_COUNTDOWN_MAX;
                        alphabet_current_key = key_code;
                        alphabet_index = (alphabet_index + 1) % wcslen(alphabet[(int)key_code]);
                    }

                    // 在屏幕上循环显示当前选中的字母
                    wchar_t letter[2];
                    uint32_t x_pos = 1;
                    for (int i = 0; i < wcslen(alphabet[(int)key_code]); i++) {
                        letter[0] = alphabet[(int)key_code][i]; letter[1] = 0;
                        render_line(letter, x_pos, 50, (i != alphabet_index));
                        x_pos += 8;
                    }

                    STATE = 0;
                }
            }

            // 长+短按A键：删除一个字符；如果输入缓冲区为空，则回到主菜单
            else if ((key_edge == -1 || key_edge == -2) && key_code == 10) {
                if (input_counter >= 1) {
                    input_buffer[--input_counter] = 0;
                    render_input_buffer(input_buffer, ime_mode_flag, 1);
                    STATE = 0;
                }
                else {
                    input_buffer = refresh_input_buffer(input_buffer, &input_counter);
                    show_main_menu();
                    STATE = -2;
                }
            }

            // 短按B键：转到设置
            else if (key_edge == -1 && key_code == 11) {
                OLED_SoftClear();
                render_text(L"选择语言模型：\n1. Nano-168M-QA\n2. Nano-56M-QA\n3. Nano-56M-Neko\n4. Qwen3-0.6B", 0);
                OLED_Refresh();

                STATE = 4;
            }

            // 长+短按C键：依次切换汉-英-数输入模式
            else if ((key_edge == -1 || key_edge == -2) && key_code == 12) {
                ime_mode_flag = (ime_mode_flag + 1) % 3;
                render_input_buffer(input_buffer, ime_mode_flag, 1);
                STATE = 0;
            }

            // 短按D键：提交
            else if (key_edge == -1 && key_code == 13) {
                render_input_buffer(input_buffer, ime_mode_flag, 1);

                STATE = 10;
                goto STATE_10;
            }

            // 长+短按*：光标向左移动
            else if ((key_edge == -1 || key_edge == -2) && key_code == 14) {
                
            }

            // 按下瞬间*：开始录音
            else if (key_edge > 0 && key_code == 14) {
                OLED_SoftClear();
                render_text(L" \n \n     正在录音...", 0);
                OLED_Refresh();
                is_recording = 1;
                start_recording();

                STATE = 20;
                goto STATE_20;
            }

            // 长+短按#键：（关于）光标向右移动
            else if ((key_edge == -1 || key_edge == -2) && key_code == 15) {
                OLED_SoftClear();
                render_text(L"Project MARGA!\nV2025.5\n电子鹦鹉笼\n\n(c) 2025 BD4SUR", 0);
                OLED_Refresh();

                STATE = 5;
            }

            break;

        /////////////////////////////////////////////
STATE_1:// 拼音输入状态
        /////////////////////////////////////////////

        case 1:

            // 短按D键：开始选字
            if (key_edge == -1 && key_code == 13) {
                if (candidate_pages) {
                    render_pinyin_input(candidate_pages, pinyin_keys, current_page, candidate_page_num, 1);
                    STATE = 2;
                }
            }

            // 短按A键：取消输入拼音，清除已输入的所有按键，回到初始状态
            else if (key_edge == -1 && key_code == 10) {
                render_input_buffer(input_buffer, ime_mode_flag, 1);

                current_page = 0;
                pinyin_keys = 0;
                STATE = 0;
            }

            // 短按2-9键：继续输入拼音
            else if (key_edge == -1 && (key_code >= 2 && key_code <= 9)) {
                pinyin_keys *= 10;
                pinyin_keys += (uint32_t)key_code;

                if (candidates) { free(candidates); candidates = NULL; }
                free_candidate_pages(candidate_pages, candidate_page_num); candidate_pages = NULL;

                candidates = candidate_hanzi_list(pinyin_keys, &candidate_num);

                if (candidates) { // 如果当前键码有对应的候选字
                    // 候选字列表分页
                    candidate_pages = candidate_paging(candidates, candidate_num, 10, &candidate_page_num);
                    render_pinyin_input(candidate_pages, pinyin_keys, current_page, candidate_page_num, 0);
                }
                else {
                    render_pinyin_input(NULL, pinyin_keys, 0, 0, 0);
                }

                STATE = 1;
            }

            break;

        /////////////////////////////////////////////
STATE_2:// 候选字选择状态
        /////////////////////////////////////////////

        case 2:

            // 短按0-9键：从候选字列表中选定一个字，选定后转到初始状态
            if (key_edge == -1 && (key_code >= 0 && key_code <= 9)) {
                uint32_t index = (key_code == 0) ? 9 : (key_code - 1); // 按键0对应9
                // 将选中的字加入输入缓冲区
                uint32_t ch = candidate_pages[current_page][index];
                if (ch) {
                    input_buffer[input_counter++] = ch;
                }
                else {
                    printf("选定了列表之外的字，忽略。\n");
                }

                render_input_buffer(input_buffer, ime_mode_flag, 1);

                free(candidates); candidates = NULL;
                free_candidate_pages(candidate_pages, candidate_page_num); candidate_pages = NULL;
                current_page = 0;

                pinyin_keys = 0;
                STATE = 0;
            }

            // 长+短按*键：候选字翻页到上一页
            else if ((key_edge == -1 || key_edge == -2) && key_code == 14) {
                if(current_page > 0) {
                    current_page--;
                    render_pinyin_input(candidate_pages, pinyin_keys, current_page, candidate_page_num, 1);
                }

                STATE = 2;
            }

            // 长+短按#键：候选字翻页到下一页
            else if ((key_edge == -1 || key_edge == -2) && key_code == 15) {
                if(current_page < candidate_page_num - 1) {
                    current_page++;
                    render_pinyin_input(candidate_pages, pinyin_keys, current_page, candidate_page_num, 1);
                }

                STATE = 2;
            }

            // 短按A键：取消选择，回到初始状态
            else if (key_edge == -1 && key_code == 10) {
                render_input_buffer(input_buffer, ime_mode_flag, 1);

                current_page = 0;
                pinyin_keys = 0;
                STATE = 0;
            }

            break;

        /////////////////////////////////////////////
STATE_3:// 符号选择状态
        /////////////////////////////////////////////

        case 3:

            // 短按0-9键：从符号列表中选定一个符号，选定后转到初始状态
            if (key_edge == -1 && (key_code >= 0 && key_code <= 9)) {
                uint32_t index = (key_code == 0) ? 9 : (key_code - 1); // 按键0对应9
                // 将选中的符号加入输入缓冲区
                uint32_t ch = candidate_pages[current_page][index];
                if (ch) {
                    input_buffer[input_counter++] = ch;
                }
                else {
                    printf("选定了列表之外的符号，忽略。\n");
                }

                render_input_buffer(input_buffer, ime_mode_flag, 1);

                free(candidates); candidates = NULL;
                free_candidate_pages(candidate_pages, candidate_page_num); candidate_pages = NULL;
                current_page = 0;

                pinyin_keys = 0;
                STATE = 0;
            }

            // 长+短按*键：候选字翻页到上一页
            else if ((key_edge == -1 || key_edge == -2) && key_code == 14) {
                if(current_page > 0) {
                    current_page--;
                    render_symbol_input(candidate_pages, current_page, candidate_page_num);
                }

                STATE = 3;
            }

            // 长+短按#键：候选字翻页到下一页
            else if ((key_edge == -1 || key_edge == -2) && key_code == 15) {
                if(current_page < candidate_page_num - 1) {
                    current_page++;
                    render_symbol_input(candidate_pages, current_page, candidate_page_num);
                }

                STATE = 3;
            }

            // 短按A键：取消选择，回到初始状态
            else if (key_edge == -1 && key_code == 10) {
                render_input_buffer(input_buffer, ime_mode_flag, 1);

                current_page = 0;
                pinyin_keys = 0;
                STATE = 0;
            }

            break;

        /////////////////////////////////////////////
STATE_4:// 选择语言模型状态
        /////////////////////////////////////////////

        case 4:

            // 短按1键
            if (key_edge == -1 && key_code == 1) {
                llm_context_free(g_llm_ctx);
                OLED_SoftClear(); render_text(L" 正在加载语言模型\n Nano-168M-QA\n 请稍等...", 0); OLED_Refresh();
                max_seq_len = 512;
                g_llm_ctx = llm_context_init(MODEL_PATH_1, NULL, max_seq_len, repetition_penalty, temperature, top_p, top_k, random_seed);
                OLED_SoftClear(); render_text(L"加载完成~", 0); OLED_Refresh();
                usleep(1000*1000);
                render_input_buffer(input_buffer, ime_mode_flag, 1);
                current_page = 0;
                STATE = 0;
            }

            // 短按2键
            else if (key_edge == -1 && key_code == 2) {
                llm_context_free(g_llm_ctx);
                OLED_SoftClear(); render_text(L" 正在加载语言模型\n Nano-56M-QA\n 请稍等...", 0); OLED_Refresh();
                max_seq_len = 512;
                g_llm_ctx = llm_context_init(MODEL_PATH_2, NULL, max_seq_len, repetition_penalty, temperature, top_p, top_k, random_seed);
                OLED_SoftClear(); render_text(L"加载完成~", 0); OLED_Refresh();
                usleep(1000*1000);
                render_input_buffer(input_buffer, ime_mode_flag, 1);
                current_page = 0;
                STATE = 0;
            }

            // 短按3键
            else if (key_edge == -1 && key_code == 3) {
                llm_context_free(g_llm_ctx);
                OLED_SoftClear(); render_text(L" 正在加载语言模型\n Nano-56M-Neko\n 请稍等...", 0); OLED_Refresh();
                max_seq_len = 512;
                g_llm_ctx = llm_context_init(MODEL_PATH_3, LORA_PATH_3, max_seq_len, repetition_penalty, temperature, top_p, top_k, random_seed);
                OLED_SoftClear(); render_text(L"加载完成~", 0); OLED_Refresh();
                usleep(1000*1000);
                render_input_buffer(input_buffer, ime_mode_flag, 1);
                current_page = 0;
                STATE = 0;
            }

            // 短按4键
            else if (key_edge == -1 && key_code == 4) {
                llm_context_free(g_llm_ctx);
                OLED_SoftClear(); render_text(L" 正在加载语言模型\n Qwen3-0.6B\n 请稍等...", 0); OLED_Refresh();
                max_seq_len = 32768;
                g_llm_ctx = llm_context_init(MODEL_PATH_4, NULL, max_seq_len, 1.0, 0.6, 0.95, 20, random_seed);
                OLED_SoftClear(); render_text(L"加载完成~", 0); OLED_Refresh();
                usleep(1000*1000);
                render_input_buffer(input_buffer, ime_mode_flag, 1);
                current_page = 0;
                STATE = 0;
            }

            // 短按5键
            else if (key_edge == -1 && key_code == 5) {
                llm_context_free(g_llm_ctx);
                OLED_SoftClear(); render_text(L" 正在加载语言模型\n Qwen3-1.7B\n 请稍等...", 0); OLED_Refresh();
                max_seq_len = 32768;
                g_llm_ctx = llm_context_init(MODEL_PATH_5, NULL, max_seq_len, 1.0, 0.6, 0.95, 20, random_seed);
                OLED_SoftClear(); render_text(L"加载完成~", 0); OLED_Refresh();
                usleep(1000*1000);
                render_input_buffer(input_buffer, ime_mode_flag, 1);
                current_page = 0;
                STATE = 0;
            }

            // 短按A键：取消操作，回到初始状态
            else if (key_edge == -1 && key_code == 10) {
                OLED_SoftClear();
                render_text(L"操作已取消", 0);
                OLED_Refresh();

                usleep(1000*1000);

                render_input_buffer(input_buffer, ime_mode_flag, 1);

                current_page = 0;
                STATE = 0;
            }

            break;

        /////////////////////////////////////////////
STATE_5:// 显示帮助和关于信息状态
        /////////////////////////////////////////////

        case 5:

            // 短按A键：回到初始状态
            if (key_edge == -1 && key_code == 10) {
                render_input_buffer(input_buffer, ime_mode_flag, 1);

                current_page = 0;
                pinyin_keys = 0;
                STATE = 0;
            }

            break;

        /////////////////////////////////////////////
STATE_10: // 提交候选字到LLM，开始推理
        /////////////////////////////////////////////

        case 10:

            // 短按D键：从STATE_0跳转过来，响应D键，开始推理。推理完成后，并不清除输入缓冲区，因此再次按D键会重新推理。
            if (key_edge == -1 && key_code == 13) {
                OLED_SoftClear();

                wchar_t *prompt;
                if (g_llm_ctx->llm->arch == LLM_ARCH_NANO) {
                    prompt = apply_chat_template(NULL, NULL, input_buffer);
                }
                else if (g_llm_ctx->llm->arch == LLM_ARCH_QWEN2 || g_llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
                    prompt = input_buffer;
                }
                else {
                    fprintf(stderr, "Error: unknown model arch.\n");
                    exit(EXIT_FAILURE);
                }

                int32_t flag = generate_sync(g_llm_ctx, prompt, max_seq_len, on_prefilling, on_decoding, on_finished);

                if (flag == LLM_STOPPED_IN_PREFILLING || flag == LLM_STOPPED_IN_DECODING) {
                    printf("推理中止。\n");

                    // 按键延时：等待（on_decoding回调中检测到的）按键松开，防止误触发
                    usleep(500 * 1000);

                    // 计算提示语+生成内容的行数，绘制文本和滚动条
                    OLED_SoftClear();

                    wchar_t prompt_and_output[OUTPUT_BUFFER_LENGTH] = L"Homo:\n";
                    wcscat(prompt_and_output, input_buffer);
                    wcscat(prompt_and_output, L"\n--------------------\nNano:\n");
                    wcscat(prompt_and_output, g_output_of_last_session);
                    wchar_t tps_wcstr[50];
                    swprintf(tps_wcstr, 50, L"\n\n[Nano:推理中止]\n\n[平均速度%.1f词元/秒]", g_tps_of_last_session);
                    wcscat(prompt_and_output, tps_wcstr);

                    wcscpy(g_output_of_last_session, prompt_and_output);
                    output_line_num = render_text(g_output_of_last_session, 0);
                    render_scroll_bar(output_line_num, output_line_num - 5);
                    OLED_Refresh();

                    // OLED_SoftClear();
                    // render_text(L"推理中止 QAQ\n\n\n\n按[取消]键返回。", 0);
                    // OLED_Refresh();
                    // usleep(1000 * 1000);

                    STATE = 10;
                }
                else if (flag == LLM_STOPPED_NORMALLY) {
                    printf("推理自然结束。\n");

                    // 计算提示语+生成内容的行数，绘制文本和滚动条
                    OLED_SoftClear();

                    wchar_t prompt_and_output[OUTPUT_BUFFER_LENGTH] = L"Homo:\n";
                    wcscat(prompt_and_output, input_buffer);
                    wcscat(prompt_and_output, L"\n--------------------\nNano:\n");
                    wcscat(prompt_and_output, g_output_of_last_session);
                    wchar_t tps_wcstr[50];
                    swprintf(tps_wcstr, 50, L"\n\n[平均速度%.1f词元/秒]", g_tps_of_last_session);
                    wcscat(prompt_and_output, tps_wcstr);

                    wcscpy(g_output_of_last_session, prompt_and_output);
                    output_line_num = render_text(g_output_of_last_session, 0);
                    render_scroll_bar(output_line_num, output_line_num - 5);
                    OLED_Refresh();

                    STATE = 10;
                }
                else {
                    printf("推理过程异常结束。\n");

                    // 计算提示语+生成内容的行数，绘制文本和滚动条
                    OLED_SoftClear();

                    wchar_t prompt_and_output[OUTPUT_BUFFER_LENGTH] = L"Homo:\n";
                    wcscat(prompt_and_output, input_buffer);
                    wcscat(prompt_and_output, L"\n--------------------\nNano:\n");
                    wcscat(prompt_and_output, g_output_of_last_session);
                    wchar_t tps_wcstr[50];
                    swprintf(tps_wcstr, 50, L"\n\n推理过程异常结束！\n\n[平均速度%.1f词元/秒]", g_tps_of_last_session);
                    wcscat(prompt_and_output, tps_wcstr);

                    wcscpy(g_output_of_last_session, prompt_and_output);
                    output_line_num = render_text(g_output_of_last_session, 0);
                    render_scroll_bar(output_line_num, output_line_num - 5);
                    OLED_Refresh();

                    // OLED_SoftClear();
                    // render_text(L"推理过程异常退出。\n\n\n\n按[取消]键返回。", 0);
                    // OLED_Refresh();
                    // usleep(1000 * 1000);

                    STATE = 10;
                }
            }

            // 短按A键：清屏，清除输入缓冲区，回到初始状态
            else if (key_edge == -1 && key_code == 10) {

                input_buffer = refresh_input_buffer(input_buffer, &input_counter);
                render_input_buffer(input_buffer, ime_mode_flag, 1);

                current_page = 0;
                STATE = 0;
            }

            // 长+短按*键：推理结果向上翻一行。如果翻到顶，则回到最后一行。
            else if ((key_edge == -1 || key_edge == -2) && key_code == 14) {
                if (output_shift == (output_line_num - 5)) { // 卷到顶的卷动量
                    output_shift = 0;
                }
                else {
                    output_shift++;
                }

                OLED_SoftClear();
                render_text(g_output_of_last_session, output_shift);
                render_scroll_bar(output_line_num, output_line_num - output_shift - 5);
                OLED_Refresh();

                STATE = 10;
            }

            // 长+短按#键：推理结果向下翻一行。如果翻到底，则回到第一行。
            else if ((key_edge == -1 || key_edge == -2) && key_code == 15) {
                if (output_shift == 0) {
                    output_shift = (output_line_num - 5); // 卷到顶的卷动量
                }
                else {
                    output_shift--;
                }

                OLED_SoftClear();
                render_text(g_output_of_last_session, output_shift);
                render_scroll_bar(output_line_num, output_line_num - output_shift - 5);
                OLED_Refresh();

                STATE = 10;
            }

            break;

        /////////////////////////////////////////////
STATE_20: // 录音进行中
        /////////////////////////////////////////////

        case 20:

            // 松开按钮，停止录音并播放
            if (is_recording > 0 && key_edge == 0 && key_code == 16) {
                is_recording = 0;

                OLED_SoftClear();
                render_text(L" \n \n     正在播放...", 0);
                OLED_Refresh();

                stop_recording();
                play_recording();

                STATE = 0;
                // 软触发A键
                key_edge = -1;
                key_code = 10;
                goto STATE_0;
            }

            break;


        default:
            break;
        }

        /////////////////////////////////////////////
        // 英文字母输入模式的循环切换
        /////////////////////////////////////////////

        if (STATE == 0 && ime_mode_flag == IME_MODE_ALPHABET) {
            // 倒计时进行中，绘制进度条
            if (alphabet_countdown > 0) {
                alphabet_countdown--;
                uint8_t x_pos = (uint8_t)(alphabet_countdown * 128 / ALPHABET_COUNTDOWN_MAX);
                OLED_DrawLine(0, 63, x_pos, 63, 1);
                OLED_DrawLine(x_pos + 1, 63, 127, 63, 0);
                OLED_Refresh();
                STATE = 0;
            }
            // 倒计时结束，提交当前选中的字母，清除进度条
            else if (alphabet_countdown == 0) {
                // 清除进度条
                alphabet_countdown--;
                OLED_DrawLine(0, 63, 127, 63, 0);
                OLED_Refresh();

                // 将当前选中的字母加入输入缓冲区
                uint32_t ch = alphabet[(int)alphabet_current_key][alphabet_index];
                if (ch) {
                    input_buffer[input_counter++] = ch;
                }
                else {
                    printf("选定了列表之外的字母，忽略。\n");
                }

                render_input_buffer(input_buffer, ime_mode_flag, 1);

                alphabet_current_key = 255;
                alphabet_index = 0;
                STATE = 0;
            }
        }


    }

    llm_context_free(g_llm_ctx);

    OLED_Close();

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}
