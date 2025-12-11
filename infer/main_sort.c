#include <wctype.h>
#include <locale.h>

#include "infer.h"
#include "prompt.h"

#define OUTPUT_BUFFER_LENGTH (32768)

// 推理引擎实例（单例模式）
static Nano_Context *g_llm_ctx;

static char *MODEL_PATH = "/home/bd4sur/ai/_model/Nano/sort6.bin";

int main() {
    if(!setlocale(LC_CTYPE, "")) return -1;

    printf("Nano Language Model Inference Engine CLI\n\n");

    unsigned long long random_seed = (unsigned int)time(NULL);
    uint32_t max_seq_len = 6;

    printf("Using model: %s\n", MODEL_PATH);

    g_llm_ctx = llm_context_init(MODEL_PATH, NULL, max_seq_len, 0, 0, 0, 1, random_seed);

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

    wchar_t input[10] = L"114514";
    wchar_t sorted[10];

    seq2seq(g_llm_ctx, input, sorted, max_seq_len);

    printf("Sorted: %ls\n", sorted);

    llm_context_free(g_llm_ctx);

    return 0;
}
