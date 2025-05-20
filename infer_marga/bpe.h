#ifndef __NANO_BPE_TOKENIZER__
#define __NANO_BPE_TOKENIZER__

#include "infer.h"

void build_bpe_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_bpe_tokenizer(Tokenizer* t);

char* decode_bpe_pair(Tokenizer* t, uint32_t prev_token, uint32_t token);
wchar_t *decode_bpe(Tokenizer *t, uint32_t *ids, uint32_t len);
void encode_bpe(Tokenizer* t, char *text, uint32_t *tokens, uint32_t *n_tokens);

uint32_t *apply_qwen_chat_template(Tokenizer *t, wchar_t *user_prompt_wchar, uint32_t *prompt_length, int32_t enable_thinking);

#endif
