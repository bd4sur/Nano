#ifndef __NANO_TOKENIZER_H__
#define __NANO_TOKENIZER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

#define MAX_TOKEN_LENGTH  (17) // NOTE 虽然可以扫描词表得到该值，但是考虑到性能，设置为固定值（对于16384词表而言，至少17）

// ===============================================================================
// 公共数据结构
// ===============================================================================

// 仅用于LLM_ARCH_QWEN2/3
typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    // 共享
    uint32_t vocab_size;
    // 仅LLM_ARCH_NANO
    wchar_t *unicode_charset;
    wchar_t **token_list;
    struct Trie *vocab_trie;
    struct Map *unicode_to_id_map;
    struct Map *token_to_id_map;
    // 仅LLM_ARCH_QWEN2/3
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

// ===============================================================================
// BPE tokenizer
// ===============================================================================

void build_bpe_tokenizer(Tokenizer* t, uint8_t* tokenizer_buffer, int vocab_size);
void free_bpe_tokenizer(Tokenizer* t);

char* decode_bpe_pair(Tokenizer* t, uint32_t prev_token, uint32_t token);
wchar_t *decode_bpe(Tokenizer *t, uint32_t *ids, uint32_t len);
void encode_bpe(Tokenizer* t, char *text, uint32_t *tokens, uint32_t *n_tokens);

uint32_t *apply_qwen_chat_template(Tokenizer *t, wchar_t *user_prompt_wchar, uint32_t *prompt_length, int32_t enable_thinking);

// ===============================================================================
// 朴素分词器和词元编解码（用于自研Nano模型）
// ===============================================================================


uint32_t *string_to_ids(struct Map *unicode_to_id_map, wchar_t *utext);

uint32_t *encode_nano(Tokenizer *t, wchar_t *text, uint32_t *n_tokens_ptr);
wchar_t *decode_nano(Tokenizer *t, uint32_t *ids, uint32_t len);

void free_tokenizer(Tokenizer *tk);

#ifdef __cplusplus
}
#endif

#endif
