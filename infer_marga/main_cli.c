#include <wctype.h>

#include "infer.h"

#define OUTPUT_BUFFER_LENGTH (512)

// 推理引擎实例（单例模式）
static Nano_Context *g_llm_ctx;

static char *MODEL_PATH_1 = "/home/bd4sur/ai/_model/qwen3-1b7-q80.bin";

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

int32_t on_prefilling(Nano_Session *session) {
    printf("\033[1A\033[2K\r");
    printf("Pre-filling: %d%%\n", (int)((float)(session->pos + 1) / (float)session->num_prompt_tokens * 100));
    fflush(stdout);
    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_decoding(Nano_Session *session) {
    uint32_t new_token[1];
    new_token[0] = session->next_token;
    printf("%ls", decode(g_llm_ctx, new_token, 1));

    // printf("%s", g_llm_ctx->tokenizer->vocab[session->next_token]);
    // printf("%ls\n", session->output_text);

    // printf("\033[1A\033[2K\r");
    // printf("Decoding: %d\n", session->pos);

    fflush(stdout);
    free(session->output_text);
    return LLM_RUNNING_IN_DECODING;
}

int32_t on_finished(Nano_Session *session) {
    printf("[%ls]\n", drop_thinking(session->output_text));
    printf("TPS = %f\n", session->tps);
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

    unsigned long long random_seed = (unsigned int)time(NULL);
    uint32_t max_seq_len = 32768;

    g_llm_ctx = llm_context_init(MODEL_PATH_1, NULL, max_seq_len, 1.0, 0.7, 0.8, 20, random_seed);

    printf("block_size = %d\n", g_llm_ctx->llm->config.block_size);
    printf("vocab_size = %d\n", g_llm_ctx->llm->config.vocab_size);
    printf("n_layer = %d\n", g_llm_ctx->llm->config.n_layer);
    printf("n_embd = %d\n", g_llm_ctx->llm->config.n_embd);
    printf("n_head = %d\n", g_llm_ctx->llm->config.n_head);
    printf("n_kv_head = %d\n", g_llm_ctx->llm->config.n_kv_head);
    printf("n_hidden = %d\n", g_llm_ctx->llm->config.n_hidden);
    printf("is_shared_classifier = %d\n", g_llm_ctx->llm->config.is_shared_classifier);
    printf("head_dim = %d\n", g_llm_ctx->llm->config.head_dim);
    printf("llm->arch = %d\n", g_llm_ctx->llm->arch);

    while (1) {
        wchar_t prompt[2048];
        wchar_t *time_str = get_current_time_wstring();

        swprintf(prompt, 2048, L"You must output instruction according to following rules. Do not output any other extra content. Current number: %ls. Rules: Check whether current number is between 1530 and 1540, inclusive. If yes, output 'alarm_on'. Otherwise, output 'alarm_off'.", time_str);
        printf("\n\n");
        generate_sync(g_llm_ctx, prompt, max_seq_len, on_prefilling, on_decoding, on_finished);
    }

    llm_context_free(g_llm_ctx);

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}
