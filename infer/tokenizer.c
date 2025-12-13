#include "tokenizer.h"

#include <locale.h>
#include <wchar.h>

// ===============================================================================
// BPE tokenizer
// ===============================================================================

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_bpe_tokenizer(Tokenizer* t, uint8_t* buffer, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    uint32_t offset = 0;

    // uint32_t tokenizer_field_byte_length = *((uint32_t*)(buffer + offset));
    offset += sizeof(uint32_t);

    t->max_token_length = *((uint32_t*)(buffer + offset));
    offset += sizeof(uint32_t);

    for (uint32_t i = 0; i < vocab_size; i++) {
        t->vocab_scores[i] = *((float*)(buffer + offset));
        offset += sizeof(float);

        uint32_t len = *((uint32_t*)(buffer + offset));
        offset += sizeof(uint32_t);

        t->vocab[i] = (char *)malloc(len + 1);
        for (uint32_t j = 0; j < len; j++) {
            t->vocab[i][j] = *((char*)(buffer + offset));
            offset += sizeof(char);
        }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
}

void free_bpe_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode_bpe_pair(Tokenizer* t, uint32_t prev_token, uint32_t token) {
    char *piece = t->vocab[token];
    /*
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    */
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

wchar_t *decode_bpe(Tokenizer *t, uint32_t *ids, uint32_t len) {
    wchar_t *output_wc = (wchar_t *)calloc(len * t->max_token_length + 1, sizeof(wchar_t));
    char    *output_ac = (char *   )calloc(len * t->max_token_length + 1, sizeof(char));
    output_ac[0] = 0;
    for (uint32_t i = 0; i < len; i++) {
        strcat(output_ac, t->vocab[ids[i]]);
    }
    _mbstowcs(output_wc, output_ac, len * t->max_token_length);
    free(output_ac);
    return output_wc;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode_bpe(Tokenizer* t, char *text, uint32_t *tokens, uint32_t *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    // if (text[0] != '\0') {
    //     int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    //     tokens[(*n_tokens)++] = dummy_prefix;
    // }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            // sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);

            const char *str1 = t->vocab[tokens[i]];
            const char *str2 = t->vocab[tokens[i+1]];
            char *ptr = str_buffer;
            while (*str1 != '\0') *ptr++ = *str1++;
            while (*str2 != '\0') *ptr++ = *str2++;
            *ptr = '\0';

            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}

uint32_t *apply_qwen_chat_template(Tokenizer *t, wchar_t *user_prompt_wchar, uint32_t *prompt_length, int32_t enable_thinking) {
    if (wcslen(user_prompt_wchar) <= 0) {
        printf("Warning: empty prompt string. Use default prompt '请你自我介绍。'.\n");
        user_prompt_wchar = L"请你自我介绍。";
    }

    char user_prompt[MAX_PROMPT_BUFFER_LENGTH];
    _wcstombs(user_prompt, user_prompt_wchar, MAX_PROMPT_BUFFER_LENGTH);
    uint32_t num_user_prompt_tokens = 0;
    uint32_t *user_prompt_tokens = (uint32_t*)calloc(strlen(user_prompt), sizeof(uint32_t));
    encode_bpe(t, user_prompt, user_prompt_tokens, &num_user_prompt_tokens);

    uint32_t *prompt_tokens = (uint32_t*)calloc(num_user_prompt_tokens + 16, sizeof(uint32_t));

    prompt_tokens[0] = 151644; // <|im_start|>
    prompt_tokens[1] = 872;    // user
    prompt_tokens[2] = 198;    // \n
    for(uint32_t i = 0; i < num_user_prompt_tokens; i++) *(prompt_tokens + 3 + i) = user_prompt_tokens[i];
    prompt_tokens[3 + num_user_prompt_tokens] = 151645; // <|im_end|>
    prompt_tokens[4 + num_user_prompt_tokens] = 198;    // \n
    prompt_tokens[5 + num_user_prompt_tokens] = 151644; // <|im_start|>
    prompt_tokens[6 + num_user_prompt_tokens] = 77091;  // assistant
    prompt_tokens[7 + num_user_prompt_tokens] = 198;    // \n

    if (enable_thinking) {
        prompt_tokens[8 + num_user_prompt_tokens] = 0;
        *prompt_length = 8 + num_user_prompt_tokens;
    }
    else {
        prompt_tokens[8 + num_user_prompt_tokens] = 151667; // <think>
        prompt_tokens[9 + num_user_prompt_tokens] = 198;    // \n
        prompt_tokens[10 + num_user_prompt_tokens] = 198;    // \n
        prompt_tokens[11 + num_user_prompt_tokens] = 151668; // </think>
        prompt_tokens[12 + num_user_prompt_tokens] = 198;    // \n
        prompt_tokens[13 + num_user_prompt_tokens] = 198;    // \n
        prompt_tokens[14 + num_user_prompt_tokens] = 0;
        *prompt_length = 14 + num_user_prompt_tokens;
    }

    free(user_prompt_tokens);
    return prompt_tokens;
}


// ===============================================================================
// 朴素分词器和词元编解码（用于自研Nano模型）
// ===============================================================================

void free_tokenizer(Tokenizer *tk) {
    for(uint32_t i = 0; i < tk->vocab_size; i++) {
        if(NULL != tk->token_list[i])
            free(tk->token_list[i]);
    }
    free(tk->unicode_charset);
    free_trie(tk->vocab_trie);
    free_map(tk->token_to_id_map);
    free_map(tk->unicode_to_id_map);
}

uint32_t *string_to_ids(struct Map *unicode_to_id_map, wchar_t *utext) {
    uint32_t len = wcslen(utext);
    uint32_t *ids = calloc(len, sizeof(uint32_t));
    for(uint32_t i = 0; i < wcslen(utext); i++) {
        ids[i] = map_get(unicode_to_id_map, utext[i]);
    }
    return ids;
}

wchar_t *decode_nano(Tokenizer *t, uint32_t *ids, uint32_t len) {
    wchar_t *out = (wchar_t *)calloc(len * MAX_TOKEN_LENGTH + 1, sizeof(wchar_t));
    uint32_t count = 0;
    for(uint32_t i = 0; i < len; i++) {
        wchar_t *utoken = t->token_list[ids[i]];
        for(uint32_t j = 0; j < wcslen(utoken); j++) {
            out[count] = utoken[j];
            count++;
        }
    }
    out[count] = 0;
    return out;
}

uint32_t *encode_nano(Tokenizer *t, wchar_t *text, uint32_t *n_tokens_ptr) {
    uint32_t *input_ids = string_to_ids(t->unicode_to_id_map, text);
    uint32_t *output_ids = (uint32_t *)calloc(wcslen(text), sizeof(uint32_t *));
    uint32_t token_count = tokenize(t->vocab_trie, output_ids, input_ids, wcslen(text), MAX_TOKEN_LENGTH);
    free(input_ids);
    *n_tokens_ptr = token_count;
    return output_ids;
}

