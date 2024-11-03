#ifndef __NANO_INFER_TOKENIZER_H__
#define __NANO_INFER_TOKENIZER_H__

#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <wchar.h>

#define MAX_TOKEN_LENGTH (17) // TODO 扫描词表得到该值
#define uint32_t unsigned int

// ===============================================================================
// 散列表
// ===============================================================================

struct Entry {
    uint32_t key;
    uint32_t value;
    struct Entry *next;
};

struct Map {
    uint32_t bucket_num;
    struct Entry **buckets;
};

struct Map *new_map(uint32_t bucket_num) {
    struct Map *pm = (struct Map *)calloc(1, sizeof(struct Map));
    pm->bucket_num = bucket_num;
    pm->buckets = (struct Entry **)calloc(bucket_num, sizeof(struct Entry *));
    for(uint32_t i = 0; i < bucket_num; i++) {
        pm->buckets[i] = NULL;
    }
    return pm;
}

struct Entry *new_entry(uint32_t key, uint32_t value) {
    struct Entry *e = (struct Entry *)calloc(1, sizeof(struct Entry));
    e->key = key;
    e->value = value;
    e->next = NULL;
    return e;
}

uint32_t map_hash(uint32_t key, uint32_t bucket_num) {
    // TODO 有巨大优化空间：大部分汉字是0x4e00~0xa000
    return key % bucket_num;
}

uint32_t map_set(struct Map *m, uint32_t key, uint32_t value) {
    uint32_t hashcode = map_hash(key, m->bucket_num);
    struct Entry *e = m->buckets[hashcode];
    if(NULL == e) {
        m->buckets[hashcode] = new_entry(key, value);
    }
    else {
        while(NULL != e->next) {
            e = e->next;
        }
        e->next = new_entry(key, value);
    }
    return hashcode;
}

uint32_t map_get(struct Map *m, uint32_t key) {
    uint32_t hashcode = map_hash(key, m->bucket_num);
    struct Entry *e = m->buckets[hashcode];
    if(NULL == e) {
        return 0; // 等同于<|padding|>
    }
    else {
        do {
            if(e->key == key) {
                return e->value;
            }
            e = e->next;
        } while(NULL != e);
    }
    return 0;
}

// ===============================================================================
// Trie树
// ===============================================================================

struct Trie {
    struct Trie **children;
    uint32_t vocab_size;
    uint32_t token_id;
    uint32_t is_end_of_token;
};

struct Trie *new_trie(uint32_t vocab_size, uint32_t is_end_of_token) {
    struct Trie *pnode = (struct Trie *)calloc(1, sizeof(struct Trie));
    if(NULL == pnode) {
        return NULL;
    }
    else {
        pnode->children = (struct Trie **)calloc(vocab_size, sizeof(struct Trie*));
        pnode->vocab_size = vocab_size;
        pnode->token_id = 0;
        pnode->is_end_of_token = is_end_of_token;
        return pnode;
    }
}

int add_token(struct Trie *trie_node, uint32_t *token, uint32_t token_len, uint32_t token_id) {
    struct Trie *current_node = trie_node;
    for(uint32_t i = 0; i < token_len; i++) {
        uint32_t cid = token[i];
        uint32_t is_eot = (i == token_len - 1) ? 1 : 0;
        struct Trie *next_node = current_node->children[cid];
        if(NULL == next_node) {
            next_node = new_trie(trie_node->vocab_size, is_eot);
            if(NULL == next_node) {
                return -1;
            }
            else {
                current_node->children[cid] = next_node;
                current_node = next_node;
            }
        }
        else {
            current_node = next_node;
        }
    }
    current_node->is_end_of_token = 1;
    current_node->token_id = token_id;
    return 0;
}

int match_token(struct Trie *trie_node, uint32_t *pattern, uint32_t pattern_len, uint32_t *token_id) {
    struct Trie *current_node = trie_node;
    for(uint32_t i = 0; i < pattern_len; i++) {
        uint32_t cid = pattern[i];
        struct Trie *next_node = current_node->children[cid];
        if(NULL == next_node) {
            return -1;
        }
        current_node = next_node;
        if(i == pattern_len - 1) {
            if(current_node->is_end_of_token == 1) {
                *token_id = current_node->token_id;
                return 0;
            }
            else {
                return -1;
            }
        }
    }
}

// ===============================================================================
// 分词器和词元编解码器
// ===============================================================================

typedef struct {
    uint32_t vocab_size;
    wchar_t *unicode_charset;
    wchar_t **token_list;
    struct Trie *vocab_trie;
    struct Map *unicode_to_id_map;
    struct Map *token_to_id_map;
} Tokenizer;

uint32_t tokenize(struct Trie *vocab_trie, uint32_t *output_token_ids, const uint32_t *input_char_ids, uint32_t input_length, uint32_t max_token_length) {
    uint32_t token_count = 0;
    uint32_t pos = 0;
    while(pos < input_length) {
        uint32_t available_max_token_length = (input_length - pos < max_token_length) ? (input_length - pos) : max_token_length;
        for(uint32_t n = available_max_token_length; n > 0; n--) {
            uint32_t *prefix = (uint32_t*)calloc(n, sizeof(uint32_t));
            uint32_t tid = 0;
            for(uint32_t i = 0; i < n; i++) {
                prefix[i] = input_char_ids[pos + i];
            }
            if(n == 1 || match_token(vocab_trie, prefix, n, &tid) == 0) {
                output_token_ids[token_count] = (n == 1) ? prefix[0] : tid;
                token_count++;
                pos += n;
                break;
            }
        }
    }
    return token_count;
}

uint32_t *string_to_ids(struct Map *unicode_to_id_map, wchar_t *utext) {
    uint32_t len = wcslen(utext);
    uint32_t *ids = calloc(len, sizeof(uint32_t));
    for(uint32_t i = 0; i < wcslen(utext); i++) {
        ids[i] = map_get(unicode_to_id_map, utext[i]);
    }
    return ids;
}

wchar_t *decode(Tokenizer *t, uint32_t *ids, uint32_t len) {
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

uint32_t *encode(Tokenizer *t, wchar_t *text, uint32_t *n_tokens) {
    uint32_t *input_ids = string_to_ids(t->unicode_to_id_map, text);
    uint32_t *optput_ids = (uint32_t *)calloc(wcslen(text), sizeof(uint32_t *));
    uint32_t token_count = tokenize(t->vocab_trie, optput_ids, input_ids, wcslen(text), MAX_TOKEN_LENGTH);
    *n_tokens = token_count;
    return optput_ids;
}

#endif
