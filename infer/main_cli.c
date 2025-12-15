#include <time.h>
#include <wctype.h>
#include <locale.h>

#include "infer.h"
#include "prompt.h"

#define OUTPUT_BUFFER_LENGTH (32768)

// 推理引擎实例（单例模式）
static Nano_Context *g_llm_ctx;

static char *MODEL_PATH = "/home/bd4sur/ai/_model/Nano/qwen3-0b6-q80.bin";

// 是否是第一次decoding：用于判断何时清除Pre-filling进度内容
int32_t g_is_first_decoding = 1;


wchar_t *drop_thinking(wchar_t *input) {
    if (input == NULL) {
        return NULL;
    }
    wchar_t *current = input;

    // 第一步：处理<think>标签及其内容
    while (*current != L'\0') {
        wchar_t *start = wcsstr(current, L"<think>");
        if (start == NULL) break;
        wchar_t *end_tag = wcsstr(start + 7, L"</think>");
        if (end_tag != NULL) {
            wchar_t *after_end = end_tag + 8;
            wmemmove(start, after_end, wcslen(after_end) + 1);
            current = start;
        }
        else {
            *start = L'\0';
            break;
        }
    }

    // 第二步：删除所有出现的"'<|im_end|>"
    current = input;
    while (*current != L'\0') {
        wchar_t *found = wcsstr(current, L"<|im_end|>");
        if (found == NULL) break;

        wchar_t *after_tag = found + 10;
        wmemmove(found, after_tag, wcslen(after_tag) + 1);
        current = found;
    }

    // 第三步：删除所有空白字符
    wchar_t *dest = input;
    wchar_t *src = input;
    while (*src != L'\0') {
        if (!iswspace(*src)) {
            *dest = *src;
            dest++;
        }
        src++;
    }
    *dest = L'\0';
    return input;
}

#define INITIAL_CAPACITY 8
#define LINE_BUFFER_SIZE 1024

/**
 * 读取多行输入，直到 EOF（Ctrl+D on Linux/macOS, Ctrl+Z + Enter on Windows）
 * 返回：char** 字符串数组，每行包含换行符（与Python readlines一致）
 *        *line_count 为实际行数
 * 调用者需使用 freelines() 释放内存
 */
char** readlines(int *line_count) {
    if (!line_count) return NULL;

    clearerr(stdin);

    *line_count = 0;
    int capacity = INITIAL_CAPACITY;
    char** lines = malloc(capacity * sizeof(char*));
    if (!lines) return NULL;

    char buffer[LINE_BUFFER_SIZE];
    while (fgets(buffer, sizeof(buffer), stdin) != NULL) {
        // 检查是否需要扩容
        if (*line_count >= capacity) {
            capacity *= 2;
            char** new_lines = realloc(lines, capacity * sizeof(char*));
            if (!new_lines) {
                // 释放已分配内存
                for (int i = 0; i < *line_count; i++) {
                    free(lines[i]);
                }
                free(lines);
                return NULL;
            }
            lines = new_lines;
        }

        // 计算实际长度（包含\n）
        size_t len = strlen(buffer);

        // 分配内存并复制（保留换行符）
        lines[*line_count] = malloc((len + 1) * sizeof(char));
        if (!lines[*line_count]) {
            // 释放之前分配的内存
            for (int i = 0; i < *line_count; i++) {
                free(lines[i]);
            }
            free(lines);
            return NULL;
        }
        strcpy(lines[*line_count], buffer);
        (*line_count)++;
    }

    // 缩减到实际大小（可选优化）
    if (*line_count > 0) {
        char** final_lines = realloc(lines, (*line_count) * sizeof(char*));
        if (final_lines) {
            lines = final_lines;
        }
        // 若realloc失败，仍可使用原指针（只是多占了点内存）
    } else {
        free(lines);
        lines = NULL;
    }

    return lines;
}

/**
 * 释放 readlines() 返回的内存
 */
void freelines(char** lines, int line_count) {
    if (!lines) return;
    for (int i = 0; i < line_count; i++) {
        free(lines[i]);
    }
    free(lines);
}

int32_t on_prefilling(Nano_Session *session) {
    if (session->t_0 == 0) {
        session->t_0 = get_timestamp_in_ms();
    }
    else {
        session->tps = (session->pos - 1) / (double)(get_timestamp_in_ms() - session->t_0) * 1000;
    }

    printf("\033[1A\033[2K\r");
    printf("\x1b[36;1mPre-filling: %.1f%%\x1b[0m\n", ((float)(session->pos + 1) / (float)session->num_prompt_tokens * 100.0f));
    fflush(stdout);
    g_is_first_decoding = 1;
    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_decoding(Nano_Session *session) {
    if (g_is_first_decoding) {
        g_is_first_decoding = 0;
        printf("\033[1A\033[2K\r");
        printf("\n\x1b[34;1mNano:\x1b[0m ");
    }

    if (session->t_0 == 0) {
        session->t_0 = get_timestamp_in_ms();
    }
    else {
        session->tps = (session->pos - 1) / (double)(get_timestamp_in_ms() - session->t_0) * 1000;
    }

    // NOTE Qwen模型有时会输出奇怪的token，也就是把unicode字符从中间切开的不完整token。因此Qwen模型仍然需要直接从vocab中解码出这样的裸字符串并输出。
    if (g_llm_ctx->llm->arch == LLM_ARCH_NANO) {
        uint32_t new_token[1];
        new_token[0] = session->next_token;
        printf("%ls", decode_nano(g_llm_ctx->tokenizer, new_token, 1));
    }
    else if (g_llm_ctx->llm->arch == LLM_ARCH_QWEN2 || g_llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
        printf("%s", g_llm_ctx->tokenizer->vocab[session->next_token]);
    }
    else {
        printf("Error: unknown LLM arch.\n");
        return LLM_STOPPED_WITH_ERROR;
    }

    // printf("\033[1A\033[2K\r");
    // printf("Decoding: %d\n", session->pos);

    fflush(stdout);

    return LLM_RUNNING_IN_DECODING;
}

int32_t on_finished(Nano_Session *session) {
    // printf("[%ls]\n", drop_thinking(session->output_text));
    session->t_1 = get_timestamp_in_ms();
    session->tps = (session->pos - 1) / (double)(session->t_1 - session->t_0) * 1000;

    printf("\nTPS = %f\n\n", session->tps);
    return LLM_STOPPED_NORMALLY;
}

wchar_t *get_current_time_wstring() {
    time_t raw_time = time(NULL);
    struct tm timeinfo;
    localtime_r(&raw_time, &timeinfo);
    wchar_t *wbuf = malloc(80 * sizeof(wchar_t));
    if (wcsftime(wbuf, 80, L"%H%M", &timeinfo) == 0) {
        free(wbuf);
        return NULL;
    }
    return wbuf;
}

int main() {
    if(!setlocale(LC_CTYPE, "")) return -1;

    printf("Nano Language Model Inference Engine CLI\n\n");

    unsigned long long random_seed = get_timestamp_in_ms();
    uint32_t max_seq_len = 2048;

    printf("Using model: %s\n", MODEL_PATH);

    g_llm_ctx = llm_context_init(MODEL_PATH, NULL, max_seq_len, 1.0, 0.7, 0.8, 20, random_seed);

    printf("  block_size = %d\n", g_llm_ctx->llm->config.block_size);
    printf("  vocab_size = %d\n", g_llm_ctx->llm->config.vocab_size);
    printf("  n_layer = %d\n", g_llm_ctx->llm->config.n_layer);
    printf("  n_embd = %d\n", g_llm_ctx->llm->config.n_embd);
    printf("  n_head = %d\n", g_llm_ctx->llm->config.n_head);
    printf("  n_kv_head = %d\n", g_llm_ctx->llm->config.n_kv_head);
    printf("  n_hidden = %d\n", g_llm_ctx->llm->config.n_hidden);
    printf("  is_shared_classifier = %d\n", g_llm_ctx->llm->config.is_shared_classifier);
    printf("  head_dim = %d\n", g_llm_ctx->llm->config.head_dim);
    printf("  llm->arch = %d\n", g_llm_ctx->llm->arch);
    printf("  llm->quant_type = %d\n", g_llm_ctx->llm->quant_type);
    printf("  llm->group_size = %d\n", g_llm_ctx->llm->group_size);

    printf("\n请输入问题，过程中可按Enter换行；输入完成请按两次Ctrl+D提交。\n\n");

    while (1) {
        wchar_t input_text[MAX_PROMPT_BUFFER_LENGTH] = L"";
        wchar_t prompt[MAX_PROMPT_BUFFER_LENGTH] = L"";

        printf("\x1b[32;1mHomo:\x1b[0m ");

        int line_count = 0;
        char** lines = readlines(&line_count);

        for (int i = 0; i < line_count; i++) {
            wchar_t wcline[MAX_PROMPT_BUFFER_LENGTH];
            mbstowcs(wcline, lines[i], MAX_PROMPT_BUFFER_LENGTH);
            wcscat(input_text, wcline);
        }

        // 如果输入为空，则随机选用一个默认prompt
        if (wcslen(input_text) == 0) {
            set_random_prompt(input_text, get_timestamp_in_ms());
            printf("%ls\n", input_text);
        }

        // 根据模型类型应用prompt模板
        if (g_llm_ctx->llm->arch == LLM_ARCH_NANO) {
            wcscat(prompt, L"<|instruct_mark|>");
            wcscat(prompt, input_text);
            wcscat(prompt, L"<|response_mark|>");
        }
        else if (g_llm_ctx->llm->arch == LLM_ARCH_QWEN2 || g_llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
            wcscpy(prompt, input_text);
            // wcscat(prompt, L" /no_think");
        }
        else {
            fprintf(stderr, "Error: unknown model arch.\n");
            exit(EXIT_FAILURE);
        }

        // wchar_t *time_str = get_current_time_wstring();
        // swprintf(prompt, 2048, L"You must output instruction according to following rules. Do not output any other extra content. Current number: %ls. Rules: Check whether current number is between 1530 and 1540, inclusive. If yes, output 'alarm_on'. Otherwise, output 'alarm_off'.", time_str);
        printf("\n\n");
        generate_sync(g_llm_ctx, prompt, max_seq_len, on_prefilling, on_decoding, on_finished);

        freelines(lines, line_count);
    }

    llm_context_free(g_llm_ctx);

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}
