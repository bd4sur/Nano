#ifndef __NANO_INFER_H__
#define __NANO_INFER_H__

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <locale.h>
#include <wchar.h>

// 是否使用mmap？
// #define NANO_USE_MMAP

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #if defined NANO_USE_MMAP
        #include <sys/mman.h>
    #endif
#endif

#define uint32_t unsigned int

#define STATUS_PREFILLING (11)
#define STATUS_DECODING   (12)
#define STATUS_STOPPED    (21)
#define MAX_TOKEN_LENGTH  (17) // NOTE 虽然可以扫描词表得到该值，但是考虑到性能，设置为固定值（对于16384词表而言，至少17）

#define VOCAB_SIZE        (16384) // Trie树字符数。为效率考虑（避免动态内存分配），固定为16384。
#define INITIAL_POOL_SIZE (16384) // 初始内存池大小。

// ===============================================================================
// 数据结构定义
// ===============================================================================

typedef struct {
    uint32_t block_size;
    uint32_t vocab_size;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_kv_head;
    uint32_t n_hidden;
    uint32_t is_shared_classifier;
} LLM_Config;

typedef struct {
    float* token_embedding;    // (vocab_size, n_embd)
    float* rms_norm_attn;      // (layer, n_embd)

    float* wq; // (layer, n_embd, n_heads * head_size)
    float* wk; // (layer, n_embd, n_kv_heads * head_size)
    float* wv; // (layer, n_embd, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, n_embd)

    float* rms_norm_ffn; // (layer, n_embd)

    float* w1; // (layer, n_hidden, n_embd)
    float* w2; // (layer, n_embd, n_hidden)
    float* w3; // (layer, n_hidden, n_embd)

    float* rms_norm_final; // (n_embd,)

    float* token_classifier;
    float* freq_cis_real;
    float* freq_cis_imag;
} LLM_Param;

typedef struct {
    float *x;       // activation at current time stamp (n_embd,)
    float *xb;      // same, but inside a residual branch (n_embd,)
    float *xb2;     // an additional buffer just for convenience (n_embd,)
    float *hb;      // buffer for hidden dimension in the ffn (n_hidden,)
    float *hb2;     // buffer for hidden dimension in the ffn (n_hidden,)
    float *q;       // query (n_embd,)
    float *k;       // key (kv_dim,)
    float *v;       // value (kv_dim,)
    float *k_cache; // (layer, block_size, kv_dim)
    float *v_cache; // (layer, block_size, kv_dim)
    float *att;     // buffer for scores/attention values (n_heads, block_size)
    float *logits;  // output logits

    float *q0;      // query  LoRA branch (lora_cfg.lora_rank,)
    float *k0;      // key    LoRA branch (lora_cfg.lora_rank,)
    float *v0;      // value  LoRA branch (lora_cfg.lora_rank,)
    float *o0;      // output LoRA branch (lora_cfg.lora_rank,)
    float *q1;      // query  LoRA branch (dim,)
    float *k1;      // key    LoRA branch (kv_dim,)
    float *v1;      // value  LoRA branch (kv_dim,)
    float *o1;      // output LoRA branch (kv_dim,)
} FwdBuffer;

typedef struct {
    LLM_Config config;
    LLM_Param params;
    FwdBuffer state;
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;            // file descriptor for memory mapping
    float *data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} LLM;

typedef struct {
    uint32_t lora_rank;
    uint32_t lora_alpha;
    uint32_t n_layer;      // 用于校验
    uint32_t n_embd;       // 用于校验
    uint32_t n_head;       // 用于校验
    uint32_t n_kv_head;    // 用于校验
    uint32_t n_hidden;     // 用于校验
    uint32_t lora_config;  // 预留：用于控制LoRA用到哪些层
} LoRA_Config;

typedef struct {
    float *wq_lora_a;
    float *wq_lora_b;
    float *wk_lora_a;
    float *wk_lora_b;
    float *wv_lora_a;
    float *wv_lora_b;
    float *wo_lora_a;
    float *wo_lora_b;
} LoRA_Param;

typedef struct {
    LoRA_Config config;
    LoRA_Param  params;
    float *data;
} LoRA;

typedef struct {
    uint32_t vocab_size;
    wchar_t *unicode_charset;
    wchar_t **token_list;
    struct Trie *vocab_trie;
    struct Map *unicode_to_id_map;
    struct Map *token_to_id_map;
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float repetition_penalty;
    float temperature;
    float top_p;
    uint32_t top_k;
    unsigned long long rng_state;
} Sampler;

typedef struct {
    LLM *llm;
    LoRA *lora;
    Tokenizer *tokenizer;
    Sampler *sampler;
    int random_seed;
} Nano_Context;



void load_llm(LLM *llm, Tokenizer *tk, char *model_path);
Sampler *build_sampler(int vocab_size, float repetition_penalty, float temperature, float top_p, unsigned int top_k, unsigned long long rng_seed);
LoRA *load_lora(LLM *llm, char *lora_path);
unsigned int *encode(Tokenizer *t, wchar_t *text, unsigned int *n_tokens_ptr);
wchar_t *decode(Tokenizer *t, unsigned int *ids, unsigned int len);
wchar_t *apply_template_to_str(char *str, unsigned int max_seq_len);
unsigned int generate_next_token(Nano_Context ctx, unsigned int *output_ids, unsigned int pos, int is_prefilling);
void generate(Nano_Context ctx, wchar_t *prompt, unsigned int max_seq_len, unsigned int (*on_running)(wchar_t *, unsigned int), unsigned int (*on_finished)(float, unsigned int));
void free_lora(LLM *llm, LoRA *lora);
void free_llm(LLM *llm, Tokenizer *tk);
void free_sampler(Sampler *sampler);















#endif