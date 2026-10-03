//
// ref_greedy.c - 基于原 CPU 引擎（infer.c）的贪心解码对照器
//
//   用于与 CUDA 引擎做逐 token 正确性对照：固定 temperature=0（argmax）、
//   repetition_penalty=1.0，对给定 prompt 生成 max_gen 个 token，
//   每行打印一个 token id（首行为 prompt token 数）。
//
//   用法: ./ref_greedy <model_path> <prompt> <max_gen>
//

#include <locale.h>

#include "infer.h"

#define REF_MAX_INPUT (4096)

static void noop_observation(Nano_Observation obs, void *env) {
    (void)obs; (void)env;
}

int main(int argc, char **argv) {
    if (!setlocale(LC_CTYPE, "")) return -1;

    const char *model_path = (argc > 1) ? argv[1] : "/home/bd4sur/ai/_model/Nano/qwen3-0b6-q80.bin";
    const char *prompt_str = (argc > 2) ? argv[2] : "请你介绍一下你自己。";
    uint32_t max_gen       = (argc > 3) ? (uint32_t)atoi(argv[3]) : 64;
    uint32_t max_seq_len   = 2048;

    Nano_Context *ctx = llm_context_init((char *)model_path, NULL, max_seq_len, 1.0f, 0.0f, 1.0f, 0, 42);
    ctx->observation = noop_observation;

    wchar_t wprompt[REF_MAX_INPUT];
    mbstowcs(wprompt, prompt_str, REF_MAX_INPUT);

    uint32_t num_prompt_tokens = 0;
    uint32_t *prompt_tokens = apply_qwen_chat_template(ctx->tokenizer, wprompt, &num_prompt_tokens, 1);

    uint32_t *ids = (uint32_t *)calloc(max_seq_len + 1, sizeof(uint32_t));
    memcpy(ids, prompt_tokens, num_prompt_tokens * sizeof(uint32_t));

    printf("%u\n", num_prompt_tokens);
    fflush(stdout);

    uint32_t total = num_prompt_tokens - 1 + max_gen;
    for (uint32_t pos = 0; pos < total; pos++) {
        int is_prefilling = (pos < num_prompt_tokens - 1) ? 1 : 0;
        uint32_t next = generate_next_token(ctx, ids, pos, is_prefilling);
        if (is_prefilling) continue;
        ids[pos + 1] = next;
        printf("%u\n", next);
        fflush(stdout);
        if (next == 151643 || next == 151645) break;
    }

    free(ids);
    free(prompt_tokens);
    llm_context_free(ctx);
    return 0;
}
