#include "infer.h"

#define OUTPUT_BUFFER_LENGTH (512)

// 推理引擎实例（单例模式）
static Nano_Context *g_llm_ctx;

static char *MODEL_PATH_1 = "/emmc/_model/qwen3-0b6.bin";


int32_t on_prefilling(Nano_Session *session) {
    // printf("Pre-filling...\n");
    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_decoding(Nano_Session *session) {
    printf("%s", g_llm_ctx->tokenizer->vocab[session->next_token]);

    // printf("%ls\n", session->output_text);
    fflush(stdout);
    free(session->output_text);
    return LLM_RUNNING_IN_DECODING;
}

int32_t on_finished(Nano_Session *session) {
    printf("\nTPS = %f\n", session->tps);
    return LLM_STOPPED_NORMALLY;
}


int main() {
    if(!setlocale(LC_CTYPE, "")) return -1;

    unsigned long long random_seed = (unsigned int)time(NULL);
    uint32_t max_seq_len = 32768;

    g_llm_ctx = llm_context_init(MODEL_PATH_1, NULL, 1.0, 0.7, 0.8, 20, random_seed);

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

    wchar_t *prompt = L"-9.9和-9.11哪个大？";
    // wchar_t *prompt = apply_chat_template(NULL, NULL, L"西红柿炒鸡蛋怎么做？");

    printf("%ls\n", prompt);

    generate_sync(g_llm_ctx, prompt, max_seq_len, on_prefilling, on_decoding, on_finished);

    llm_context_free(g_llm_ctx);

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}
