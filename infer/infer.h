//
// Nano Language Model - Inference Engine written in C
//
//   BD4SUR 2024-10 2024-05
//
//   Forked from: https://github.com/karpathy/llama2.c
//


#ifndef __NANO_INFER_H__
#define __NANO_INFER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <wchar.h>

#include <fcntl.h>

#include "utils.h"
#include "tokenizer.h"
#include "tensor.h"

#include "platform.h"


#if defined _WIN32
    #include "win.h"
#elif defined __unix__ || defined __unix || (defined(__APPLE__) && defined(__MACH__))
    // 是否使用mmap？
    #define NANO_USE_MMAP
    #include <unistd.h>
    #if defined NANO_USE_MMAP
        #include <sys/mman.h>
    #endif
    #ifdef MATMUL_PTHREAD
        #include "matmul_pthread.h"
    #endif
#endif


#define LLM_ARCH_NANO  (0)
#define LLM_ARCH_QWEN2 (2)
#define LLM_ARCH_QWEN3 (3)

#define LLM_RUNNING_IN_PREFILLING (11)
#define LLM_RUNNING_IN_DECODING   (12)
#define LLM_STOPPED_WITH_ERROR    (-1)
#define LLM_STOPPED_NORMALLY      (20)
#define LLM_STOPPED_IN_PREFILLING (21)
#define LLM_STOPPED_IN_DECODING   (22)

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
    uint32_t head_dim; // 对于Nano和Qwen2为0，head_dim等于n_embd/n_head，q、k、v向量的维数固定为n_embd。对于Qwen3为固定值，该值决定了q、k、v向量的维数。
} LLM_Config;

typedef struct {
    Typed_Tensor *q_tokens;  // (vocab_size, n_embd) quantized
    float *token_embedding; // (vocab_size, n_embd)

    float *rms_norm_attn;   // (layer, n_embd)
    float *rms_norm_ffn;    // (layer, n_embd)
    float *rms_norm_final;  // (n_embd,)

    Typed_Tensor *wq;    // (layer, n_embd, n_heads * head_dim)
    Typed_Tensor *wk;    // (layer, n_embd, n_kv_heads * head_dim)
    Typed_Tensor *wv;    // (layer, n_embd, n_kv_heads * head_dim)
    Typed_Tensor *wo;    // (layer, n_heads * head_dim, n_embd)

    // Qwen2 only
    float *bq;              // (layer, n_heads * head_dim)
    float *bk;              // (layer, n_kv_heads * head_dim)
    float *bv;              // (layer, n_kv_heads * head_dim)

    // Qwen3 only
    float *q_norm;          // (layer, head_size)
    float *k_norm;          // (layer, head_size)

    Typed_Tensor *w1;    // (layer, n_hidden, n_embd)
    Typed_Tensor *w2;    // (layer, n_embd, n_hidden)
    Typed_Tensor *w3;    // (layer, n_hidden, n_embd)

    float *freq_cis_real;
    float *freq_cis_imag;

    Typed_Tensor *token_classifier; // (vocab_size, n_embd)
} LLM_Param;

typedef struct {
    float *xbuf;    // 中间激活值的统一内存
    QTYPE *qvbuf;   // 量化值统一内存
    float *qsbuf;   // 量化缩放因子统一内存
    float *kvcache; // KV缓存统一内存
    // 以下指针实际上都是指向上面的内存池（的某个偏移位置）
    float *x;       // activation at current time stamp (n_embd,)
    float *xb;      // same, but inside a residual branch (n_embd,)
    float *xba;     // output of attention block (q_dim,) q_dim = (n_embd if model == Nano||Qwen2 else (head_dim * n_head))
    float *xb2;     // an additional buffer just for convenience (n_embd,)
    float *hb;      // buffer for hidden dimension in the ffn (n_hidden,)
    float *hb2;     // buffer for hidden dimension in the ffn (n_hidden,)
    Typed_Tensor xq;   // quantized x/xb (n_embd,)
    Typed_Tensor xbaq; // quantized xba (q_dim,)
    Typed_Tensor hq;   // quantized hb (n_hidden,)
    float *q;       // query (q_dim,) q_dim = (n_embd if model == Nano||Qwen2 else (head_dim * n_head))
    float *k;       // key (kv_dim,)
    float *v;       // value (kv_dim,)
    float *k_cache; // (layer, block_size, kv_dim)
    float *v_cache; // (layer, block_size, kv_dim)
    float *att;     // buffer for scores/attention values (n_heads, block_size)
    float *logits;  // output logits

    // LoRA激活值暂不做池化
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
    // 语言模型架构类别
    uint32_t arch;
    // 量化参数
    uint32_t quant_type;
    uint32_t group_size;
    // 与mmap相关的
    int fd;            // file descriptor for memory mapping
    uint8_t *buffer;       // memory mapped data pointer
    size_t file_size; // size of the checkpoint file in bytes
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
    uint64_t rng_state;
} Sampler;

typedef struct {
    LLM *llm;
    LoRA *lora;
    Tokenizer *tokenizer;
    Sampler *sampler;
    uint32_t max_seq_len; // NOTE 每个context的最大序列长度可设置，不一定等于模型的block_size。这样可以按需控制KV缓存的大小。
    uint64_t random_seed;
} Nano_Context;

typedef struct {
    wchar_t *prompt;
    uint32_t num_prompt_tokens;
    uint32_t max_seq_len;
    uint32_t *output_ids;
    uint32_t output_count;
    wchar_t *output_text;
    uint32_t next_token;
    uint32_t pos;
    int32_t is_prefilling;
    uint64_t t_0;
    uint64_t t_1;
    float tps;
} Nano_Session;


void load_llm_from_buffer(LLM *llm, Tokenizer *tk, uint8_t *buffer, uint32_t max_seq_len);
void load_llm(LLM *llm, Tokenizer *tk, char *model_path, uint32_t max_seq_len);
Sampler *build_sampler(int vocab_size, float repetition_penalty, float temperature, float top_p, uint32_t top_k, uint64_t rng_seed);
LoRA *load_lora_from_buffer(LLM *llm, uint8_t *buffer);
LoRA *load_lora(LLM *llm, char *lora_path);

Nano_Context *llm_context_init_from_buffer(uint8_t *buffer, uint32_t max_seq_len, float repetition_penalty, float temperature, float top_p, uint32_t top_k, uint64_t random_seed);
Nano_Context *llm_context_init(char *model_path, char *lora_path, uint32_t max_seq_len, float repetition_penalty, float temperature, float top_p, uint32_t top_k, uint64_t random_seed);
void llm_context_free(Nano_Context *ctx);

uint32_t generate_next_token(Nano_Context *ctx, uint32_t *output_ids, uint32_t pos, int is_prefilling);

Nano_Session *llm_session_init(Nano_Context *ctx, wchar_t *prompt, uint32_t max_seq_len);
int32_t llm_session_step(Nano_Context *ctx, Nano_Session *session);
void llm_session_free(Nano_Session *session);

int32_t generate_sync(
    Nano_Context *ctx,
    wchar_t *prompt,
    uint32_t max_seq_len,
    int32_t (*on_prefilling)(Nano_Session*),
    int32_t (*on_decoding)(Nano_Session*),
    int32_t (*on_finished)(Nano_Session*)
);

void seq2seq(Nano_Context *ctx, wchar_t *input_list, wchar_t *output_list, uint32_t max_seq_len);

void free_lora(LLM *llm, LoRA *lora);
void free_llm(LLM *llm, Tokenizer *tk);
void free_sampler(Sampler *sampler);

#ifdef __cplusplus
}
#endif

#endif
