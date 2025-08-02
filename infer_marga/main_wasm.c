#include "bpe.h"
#include "infer.h"

static Nano_Context *g_llm_ctx;

int init_nano(char *buffer, uint32_t max_seq_len, uint32_t random_seed) {
    if(!setlocale(LC_CTYPE, "")) return -1;
    g_llm_ctx = (Nano_Context*)calloc(1, sizeof(Nano_Context));
    g_llm_ctx->max_seq_len = max_seq_len;
    g_llm_ctx->random_seed = random_seed;
    g_llm_ctx->llm = (LLM *)calloc(1, (sizeof(LLM)));
    g_llm_ctx->tokenizer = (Tokenizer *)calloc(1, (sizeof(Tokenizer)));
    g_llm_ctx->lora = NULL;
    load_llm_from_buffer(g_llm_ctx->llm, g_llm_ctx->tokenizer, buffer, g_llm_ctx->max_seq_len);
    return 0;
}

// 仅当random_seed不为0时更新random_seed
int set_sampler(float repetition_penalty, float temperature, float top_p, int top_k, uint32_t random_seed) {
    if(random_seed != 0) g_llm_ctx->random_seed = random_seed;
    g_llm_ctx->sampler = build_sampler(g_llm_ctx->llm->config.vocab_size, repetition_penalty, temperature, top_p, top_k, g_llm_ctx->random_seed);
    return 0;
}

uint32_t generate_next_token_external(uint32_t *ids, uint32_t pos, int is_prefilling) {
    return generate_next_token(g_llm_ctx, ids, pos, is_prefilling);
}

uint32_t *encode_external(wchar_t *text, uint32_t *n_tokens_ptr) {
    // return encode(g_llm_ctx->tokenizer, text, n_tokens_ptr);
    if (g_llm_ctx->llm->arch == LLM_ARCH_NANO) {
        return encode(g_llm_ctx->tokenizer, text, n_tokens_ptr);
    }
    else if (g_llm_ctx->llm->arch == LLM_ARCH_QWEN2 || g_llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
        return apply_qwen_chat_template(g_llm_ctx->tokenizer, text, n_tokens_ptr, 1);
    }
    else return NULL;
}

wchar_t *decode_external(uint32_t *ids, uint32_t len) {
    // return decode(g_llm_ctx->tokenizer, ids, len);
    if (g_llm_ctx->llm->arch == LLM_ARCH_NANO) {
        return decode(g_llm_ctx->tokenizer, ids, len);
    }
    else if (g_llm_ctx->llm->arch == LLM_ARCH_QWEN2 || g_llm_ctx->llm->arch == LLM_ARCH_QWEN3) {
        return decode_bpe(g_llm_ctx->tokenizer, ids, len);
    }
    else return NULL;
}

int load_lora_external(char *lora_buffer) {
    if(NULL != g_llm_ctx->lora) return -1;
    g_llm_ctx->lora = load_lora_from_buffer(g_llm_ctx->llm, lora_buffer);
    return 0;
}

int unload_lora_external() {
    if(NULL != g_llm_ctx->lora) {
        free(g_llm_ctx->lora);
        g_llm_ctx->lora = NULL;
    }
    return 0;
}

int close_nano() {
    free_llm(g_llm_ctx->llm, g_llm_ctx->tokenizer);
    free_sampler(g_llm_ctx->sampler);

    char *LORA_PATH  = NULL;
    if(NULL != LORA_PATH) free_lora(g_llm_ctx->llm, g_llm_ctx->lora);

    return 0;
}

