#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "am_native_LLM.h"


// ===============================================================================
// 内部辅助：数值与容器操作（参考 native_Math.c / native_String.c 的套路）
// ===============================================================================

// 将数值 TPV 统一转换为浮点数
static am_float_t am_llm_value_to_float(am_value_t v) {
    if (am_value_is_float(v)) return am_value_to_float(v);
    if (am_value_is_int(v)) return (am_float_t)am_value_to_int(v);
    if (am_value_is_uint(v)) return (am_float_t)am_value_to_uint(v);
    return 0.0;
}

// 从操作数栈中弹出一个数值，统一转换为 float。
static bool am_llm_pop_number(am_process_t *proc, am_float_t *out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (am_value_is_float(v)) { *out = am_value_to_float(v); return true; }
    if (am_value_is_int(v)) { *out = (am_float_t)am_value_to_int(v); return true; }
    if (am_value_is_uint(v)) { *out = (am_float_t)am_value_to_uint(v); return true; }
    return false;
}

// 从操作数栈中弹出一个字符串对象。
static bool am_llm_pop_wstring(am_process_t *proc, am_wstring_t **out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return false;

    *out = (am_wstring_t *)obj;
    return true;
}

// 从操作数栈中弹出一个列表对象（不复制，仅返回对象指针）。
static bool am_llm_pop_list(am_process_t *proc, am_list_t **out) {
    am_value_t v = am_process_pop_operand(proc);
    if (v == (am_value_t)UINTPTR_MAX) return false;
    if (!am_value_is_handle(v)) return false;

    am_handle_t hd = am_value_to_handle(v);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return false;

    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return false;

    *out = (am_list_t *)obj;
    return true;
}

// 读取列表中指定下标的元素为浮点数（越界返回 0.0）。
static am_float_t am_llm_list_get_float(am_list_t *lst, size_t index) {
    if (!lst || index >= lst->length) return 0.0;
    return am_llm_value_to_float(lst->children[index]);
}

// 将 float 数组打包成 Scheme 列表对象并返回其 handle。
static am_handle_t am_llm_make_float_list_handle(am_process_t *proc, const float *arr, size_t len) {
    if (len == 0) {
        am_list_t *lst = am_list_create(proc->heap_alloc, 4, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
        if (!lst) return AM_HANDLE_NULL;
        am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
        if (hd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return AM_HANDLE_NULL; }
        if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                        am_make_value_of_ptr((am_object_t *)lst)) != 0) {
            am_list_destroy(proc->heap_alloc, lst);
            am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
            return AM_HANDLE_NULL;
        }
        return hd;
    }

    am_list_t *lst = am_list_create(proc->heap_alloc, len, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!lst) return AM_HANDLE_NULL;
    lst->length = len;
    for (size_t i = 0; i < len; i++) {
        lst->children[i] = am_make_value_of_float((am_float_t)arr[i]);
    }

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, lst);
        return AM_HANDLE_NULL;
    }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return AM_HANDLE_NULL;
    }
    return hd;
}

// 将 uint32_t 数组打包成 Scheme 列表对象并返回其 handle。
static am_handle_t am_llm_make_uint_list_handle(am_process_t *proc, const uint32_t *arr, size_t len) {
    if (len == 0) {
        am_list_t *lst = am_list_create(proc->heap_alloc, 4, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
        if (!lst) return AM_HANDLE_NULL;
        am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
        if (hd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return AM_HANDLE_NULL; }
        if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                        am_make_value_of_ptr((am_object_t *)lst)) != 0) {
            am_list_destroy(proc->heap_alloc, lst);
            am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
            return AM_HANDLE_NULL;
        }
        return hd;
    }

    am_list_t *lst = am_list_create(proc->heap_alloc, len, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!lst) return AM_HANDLE_NULL;
    lst->length = len;
    for (size_t i = 0; i < len; i++) {
        lst->children[i] = am_make_value_of_uint((am_uint_t)arr[i]);
    }

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, lst);
        return AM_HANDLE_NULL;
    }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return AM_HANDLE_NULL;
    }
    return hd;
}

// 从 wchar_t 缓冲区创建字符串对象并压回操作数栈。
static int32_t am_llm_push_wstring_buf(am_process_t *proc, const wchar_t *buf, size_t len) {
    am_handle_t hd = am_process_make_wstring_handle(proc, buf, len);
    if (hd == AM_HANDLE_NULL) return -1;

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// ===============================================================================
// Base64 解码（从头实现，不依赖外部库）
// ===============================================================================

static int am_llm_base64_char_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static uint8_t *am_llm_base64_decode(const char *in, size_t *out_len) {
    if (!in || !out_len) return NULL;
    size_t in_len = strlen(in);

    // 统计有效 Base64 字符数（含 '=' 填充）
    size_t valid_count = 0;
    for (size_t i = 0; i < in_len; i++) {
        char c = in[i];
        if (c == '=' || am_llm_base64_char_value(c) >= 0) {
            valid_count++;
        }
    }
    if (valid_count == 0 || (valid_count & 3) != 0) return NULL;

    size_t max_out = (valid_count / 4) * 3;
    uint8_t *out = (uint8_t *)malloc(max_out);
    if (!out) return NULL;

    size_t j = 0;
    int buffer[4];
    int buf_len = 0;
    size_t padding = 0;

    for (size_t i = 0; i < in_len; i++) {
        char c = in[i];
        if (c == '=') {
            buffer[buf_len++] = 0;
            padding++;
        }
        else {
            int v = am_llm_base64_char_value(c);
            if (v < 0) continue; // 跳过空白等非法字符
            buffer[buf_len++] = v;
        }

        if (buf_len == 4) {
            out[j++] = (uint8_t)((buffer[0] << 2) | (buffer[1] >> 4));
            if (padding < 2) out[j++] = (uint8_t)(((buffer[1] & 0x0F) << 4) | (buffer[2] >> 2));
            if (padding < 1) out[j++] = (uint8_t)(((buffer[2] & 0x03) << 6) | buffer[3]);
            buf_len = 0;
        }
    }

    *out_len = j;
    return out;
}


// ===============================================================================
// Nano 分词器所需的基础数据结构（参考 llm/utils.c、llm/tokenizer.c）
// ===============================================================================

#define AM_LLM_VOCAB_SIZE     (16384)
#define AM_LLM_MAX_TOKEN_LEN  (17)
#define AM_LLM_ARCH_NANO      (0)

// ---------------------------------------------------------------
// HashMap<uint32_t, uint32_t>
// ---------------------------------------------------------------
typedef struct am_llm_map_entry_t {
    uint32_t key;
    uint32_t value;
    struct am_llm_map_entry_t *next;
} am_llm_map_entry_t;

typedef struct am_llm_map_t {
    uint32_t bucket_num;
    am_llm_map_entry_t **buckets;
} am_llm_map_t;

static am_llm_map_t *am_llm_new_map(uint32_t bucket_num) {
    am_llm_map_t *m = (am_llm_map_t *)calloc(1, sizeof(am_llm_map_t));
    m->bucket_num = bucket_num;
    m->buckets = (am_llm_map_entry_t **)calloc(bucket_num, sizeof(am_llm_map_entry_t *));
    return m;
}

static void am_llm_free_map(am_llm_map_t *m) {
    if (!m) return;
    for (uint32_t i = 0; i < m->bucket_num; i++) {
        am_llm_map_entry_t *e = m->buckets[i];
        while (e) {
            am_llm_map_entry_t *next = e->next;
            free(e);
            e = next;
        }
    }
    free(m->buckets);
    free(m);
}

static uint32_t am_llm_map_hash(uint32_t key, uint32_t bucket_num) {
    return key % bucket_num;
}

static void am_llm_map_set(am_llm_map_t *m, uint32_t key, uint32_t value) {
    uint32_t h = am_llm_map_hash(key, m->bucket_num);
    am_llm_map_entry_t *e = m->buckets[h];
    if (!e) {
        m->buckets[h] = (am_llm_map_entry_t *)calloc(1, sizeof(am_llm_map_entry_t));
        m->buckets[h]->key = key;
        m->buckets[h]->value = value;
        return;
    }
    while (e->next) e = e->next;
    e->next = (am_llm_map_entry_t *)calloc(1, sizeof(am_llm_map_entry_t));
    e->next->key = key;
    e->next->value = value;
}

static uint32_t am_llm_map_get(am_llm_map_t *m, uint32_t key) {
    if (!m) return 0;
    uint32_t h = am_llm_map_hash(key, m->bucket_num);
    am_llm_map_entry_t *e = m->buckets[h];
    while (e) {
        if (e->key == key) return e->value;
        e = e->next;
    }
    return 0;
}


// ---------------------------------------------------------------
// Radix Tree / Trie
// ---------------------------------------------------------------
typedef struct am_llm_radix_node_t {
    uint32_t token_id;
    uint8_t  is_end_of_token;
    uint32_t prefix_len;
    uint32_t *prefix;
    struct am_llm_radix_node_t **children;
    uint32_t num_children;
    uint32_t child_capacity;
} am_llm_radix_node_t;

typedef struct am_llm_trie_t {
    am_llm_radix_node_t *root;
} am_llm_trie_t;

static am_llm_radix_node_t *am_llm_new_radix_node(void) {
    return (am_llm_radix_node_t *)calloc(1, sizeof(am_llm_radix_node_t));
}

static void am_llm_free_radix_node(am_llm_radix_node_t *node) {
    if (!node) return;
    if (node->prefix) free(node->prefix);
    for (uint32_t i = 0; i < node->num_children; i++) {
        am_llm_free_radix_node(node->children[i]);
    }
    free(node->children);
    free(node);
}

static am_llm_trie_t *am_llm_new_trie(uint32_t vocab_size, uint8_t is_end_of_token) {
    (void)vocab_size;
    (void)is_end_of_token;
    am_llm_trie_t *trie = (am_llm_trie_t *)malloc(sizeof(am_llm_trie_t));
    trie->root = am_llm_new_radix_node();
    return trie;
}

static void am_llm_free_trie(am_llm_trie_t *trie) {
    if (!trie) return;
    am_llm_free_radix_node(trie->root);
    free(trie);
}

static am_llm_radix_node_t *am_llm_find_child(am_llm_radix_node_t *parent, uint32_t key) {
    for (uint32_t i = 0; i < parent->num_children; i++) {
        if (parent->children[i]->prefix_len > 0 && parent->children[i]->prefix[0] == key) {
            return parent->children[i];
        }
    }
    return NULL;
}

static void am_llm_add_child(am_llm_radix_node_t *parent, am_llm_radix_node_t *child) {
    if (parent->num_children >= parent->child_capacity) {
        uint32_t new_cap = parent->child_capacity == 0 ? 4 : parent->child_capacity * 2;
        am_llm_radix_node_t **new_children = (am_llm_radix_node_t **)realloc(
            parent->children, new_cap * sizeof(am_llm_radix_node_t *));
        if (!new_children) return;
        parent->children = new_children;
        parent->child_capacity = new_cap;
    }
    parent->children[parent->num_children++] = child;
}

static int am_llm_add_token(am_llm_trie_t *trie, uint32_t *token, uint32_t token_len, uint32_t token_id) {
    if (!trie || !token || token_len == 0) return -1;
    for (uint32_t i = 0; i < token_len; i++) {
        if (token[i] >= AM_LLM_VOCAB_SIZE) return -1;
    }

    am_llm_radix_node_t *node = trie->root;
    uint32_t pos = 0;
    while (pos < token_len) {
        am_llm_radix_node_t *child = am_llm_find_child(node, token[pos]);
        if (!child) {
            am_llm_radix_node_t *new_node = am_llm_new_radix_node();
            new_node->prefix_len = token_len - pos;
            new_node->prefix = (uint32_t *)malloc(new_node->prefix_len * sizeof(uint32_t));
            memcpy(new_node->prefix, token + pos, new_node->prefix_len * sizeof(uint32_t));
            new_node->is_end_of_token = 1;
            new_node->token_id = token_id;
            am_llm_add_child(node, new_node);
            return 0;
        }

        uint32_t common_len = 0;
        uint32_t min_len = (token_len - pos < child->prefix_len) ? (token_len - pos) : child->prefix_len;
        while (common_len < min_len && token[pos + common_len] == child->prefix[common_len]) {
            common_len++;
        }

        if (common_len == child->prefix_len && common_len == token_len - pos) {
            if (child->is_end_of_token) return -1;
            child->is_end_of_token = 1;
            child->token_id = token_id;
            return 0;
        }
        else if (common_len < child->prefix_len) {
            am_llm_radix_node_t *split = am_llm_new_radix_node();
            split->prefix_len = common_len;
            split->prefix = (uint32_t *)malloc(common_len * sizeof(uint32_t));
            memcpy(split->prefix, child->prefix, common_len * sizeof(uint32_t));

            uint32_t remaining = child->prefix_len - common_len;
            uint32_t *new_prefix = (uint32_t *)malloc(remaining * sizeof(uint32_t));
            memcpy(new_prefix, child->prefix + common_len, remaining * sizeof(uint32_t));
            free(child->prefix);
            child->prefix = new_prefix;
            child->prefix_len = remaining;

            am_llm_add_child(split, child);
            for (uint32_t i = 0; i < node->num_children; i++) {
                if (node->children[i] == child) {
                    node->children[i] = split;
                    break;
                }
            }

            if (common_len == token_len - pos) {
                split->is_end_of_token = 1;
                split->token_id = token_id;
            }
            else {
                am_llm_radix_node_t *new_node = am_llm_new_radix_node();
                new_node->prefix_len = token_len - pos - common_len;
                new_node->prefix = (uint32_t *)malloc(new_node->prefix_len * sizeof(uint32_t));
                memcpy(new_node->prefix, token + pos + common_len, new_node->prefix_len * sizeof(uint32_t));
                new_node->is_end_of_token = 1;
                new_node->token_id = token_id;
                am_llm_add_child(split, new_node);
            }
            return 0;
        }
        else {
            pos += common_len;
            node = child;
        }
    }
    return -1;
}

static int am_llm_match_token(am_llm_trie_t *trie, uint32_t *pattern, uint32_t pattern_len, uint32_t *token_id) {
    if (!trie || !pattern || pattern_len == 0) return -1;
    for (uint32_t i = 0; i < pattern_len; i++) {
        if (pattern[i] >= AM_LLM_VOCAB_SIZE) return -1;
    }

    am_llm_radix_node_t *node = trie->root;
    uint32_t pos = 0;
    while (pos < pattern_len) {
        am_llm_radix_node_t *child = am_llm_find_child(node, pattern[pos]);
        if (!child) return -1;
        uint32_t remaining = pattern_len - pos;
        if (remaining < child->prefix_len) return -1;
        for (uint32_t i = 0; i < child->prefix_len; i++) {
            if (pattern[pos + i] != child->prefix[i]) return -1;
        }
        pos += child->prefix_len;
        node = child;
    }
    if (node->is_end_of_token) {
        if (token_id) *token_id = node->token_id;
        return 0;
    }
    return -1;
}

static uint32_t am_llm_tokenize(am_llm_trie_t *vocab_trie, uint32_t *output_token_ids,
                                const uint32_t *input_char_ids, uint32_t input_length,
                                uint32_t max_token_length) {
    uint32_t token_count = 0;
    uint32_t pos = 0;
    while (pos < input_length) {
        uint32_t available_max = (input_length - pos < max_token_length) ? (input_length - pos) : max_token_length;
        uint32_t n;
        for (n = available_max; n > 0; n--) {
            uint32_t *prefix = (uint32_t *)calloc(n, sizeof(uint32_t));
            for (uint32_t i = 0; i < n; i++) prefix[i] = input_char_ids[pos + i];
            uint32_t tid = 0;
            if (n == 1 || am_llm_match_token(vocab_trie, prefix, n, &tid) == 0) {
                output_token_ids[token_count] = (n == 1) ? prefix[0] : tid;
                token_count++;
                pos += n;
                free(prefix);
                break;
            }
            free(prefix);
        }
        if (n == 0) {
            // 防御性处理：理论上不会发生
            output_token_ids[token_count++] = input_char_ids[pos++];
        }
    }
    return token_count;
}


// ---------------------------------------------------------------
// Tokenizer 封装
// ---------------------------------------------------------------
typedef struct am_llm_tokenizer_t {
    uint32_t vocab_size;
    wchar_t *unicode_charset;
    wchar_t **token_list;
    am_llm_trie_t *vocab_trie;
    am_llm_map_t *unicode_to_id_map;
} am_llm_tokenizer_t;

static uint32_t *am_llm_string_to_ids(am_llm_map_t *unicode_to_id_map, const wchar_t *utext) {
    uint32_t len = (uint32_t)wcslen(utext);
    uint32_t *ids = (uint32_t *)calloc(len, sizeof(uint32_t));
    for (uint32_t i = 0; i < len; i++) {
        ids[i] = am_llm_map_get(unicode_to_id_map, (uint32_t)utext[i]);
    }
    return ids;
}

static wchar_t *am_llm_decode_nano(am_llm_tokenizer_t *t, uint32_t *ids, uint32_t len) {
    wchar_t *out = (wchar_t *)calloc((size_t)len * AM_LLM_MAX_TOKEN_LEN + 1, sizeof(wchar_t));
    uint32_t count = 0;
    for (uint32_t i = 0; i < len; i++) {
        wchar_t *utoken = t->token_list[ids[i]];
        if (!utoken) continue;
        uint32_t token_len = (uint32_t)wcslen(utoken);
        for (uint32_t j = 0; j < token_len; j++) {
            out[count++] = utoken[j];
        }
    }
    out[count] = 0;
    return out;
}

static uint32_t *am_llm_encode_nano(am_llm_tokenizer_t *t, const wchar_t *text, uint32_t *n_tokens_ptr) {
    uint32_t *input_ids = am_llm_string_to_ids(t->unicode_to_id_map, text);
    uint32_t text_len = (uint32_t)wcslen(text);
    uint32_t *output_ids = (uint32_t *)calloc(text_len, sizeof(uint32_t));
    uint32_t token_count = am_llm_tokenize(t->vocab_trie, output_ids, input_ids, text_len, AM_LLM_MAX_TOKEN_LEN);
    free(input_ids);
    *n_tokens_ptr = token_count;
    return output_ids;
}


// ===============================================================================
// 模型结构与全局状态
// ===============================================================================

typedef struct am_llm_config_t {
    uint32_t block_size;
    uint32_t vocab_size;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_kv_head;
    uint32_t n_hidden;
    uint32_t is_shared_classifier;
    uint32_t head_dim;
} am_llm_config_t;

typedef struct am_llm_weights_t {
    float *rms_norm_attn;    size_t rms_norm_attn_len;
    float *rms_norm_ffn;     size_t rms_norm_ffn_len;
    float *rms_norm_final;   size_t rms_norm_final_len;
    float *token_embedding;  size_t token_embedding_len;
    float *wq;               size_t wq_len;
    float *wk;               size_t wk_len;
    float *wv;               size_t wv_len;
    float *wo;               size_t wo_len;
    float *w1;               size_t w1_len;
    float *w2;               size_t w2_len;
    float *w3;               size_t w3_len;
    float *freq_cis_real;    size_t freq_cis_real_len;
    float *freq_cis_imag;    size_t freq_cis_imag_len;
    float *token_classifier; size_t token_classifier_len;
} am_llm_weights_t;

typedef struct am_llm_state_t {
    int loaded;
    uint8_t *buffer;
    size_t buffer_size;
    am_llm_config_t config;
    am_llm_weights_t weights;
    am_llm_tokenizer_t tokenizer;
} am_llm_state_t;

static am_llm_state_t g_llm = { 0 };

static void am_llm_unload(void) {
    if (g_llm.tokenizer.token_list) {
        for (uint32_t i = 0; i < g_llm.tokenizer.vocab_size; i++) {
            free(g_llm.tokenizer.token_list[i]);
        }
        free(g_llm.tokenizer.token_list);
    }
    free(g_llm.tokenizer.unicode_charset);
    if (g_llm.tokenizer.unicode_to_id_map) {
        am_llm_free_map(g_llm.tokenizer.unicode_to_id_map);
    }
    if (g_llm.tokenizer.vocab_trie) {
        am_llm_free_trie(g_llm.tokenizer.vocab_trie);
    }
    free(g_llm.buffer);
    memset(&g_llm, 0, sizeof(g_llm));
}

static int32_t am_llm_parse_model(uint8_t *buffer, size_t size) {
    if (!buffer || size < 256) return -1;

    uint32_t *header = (uint32_t *)buffer;
    size_t off = 0;
    uint32_t magic0 = header[off++];
    uint32_t magic1 = header[off++];
    (void)magic0;
    (void)magic1;
    uint32_t major = header[off++]; (void)major;
    uint32_t minor = header[off++]; (void)minor;
    uint32_t model_type = header[off++];
    uint32_t config_length = header[off++]; (void)config_length;

    g_llm.config.block_size = header[off++];
    g_llm.config.vocab_size = header[off++];
    g_llm.config.n_layer = header[off++];
    g_llm.config.n_embd = header[off++];
    g_llm.config.n_head = header[off++];
    g_llm.config.n_kv_head = header[off++];
    g_llm.config.n_hidden = header[off++];
    g_llm.config.is_shared_classifier = header[off++];
    g_llm.config.head_dim = header[off++];

    uint32_t quant_type = header[off++];
    uint32_t group_size = header[off++]; (void)group_size;

    // 当前仅支持 Nano 架构 + F32 权重
    if (model_type != AM_LLM_ARCH_NANO) return -1;
    if (quant_type != 0) return -1;

    // -----------------------------------------------------------
    // 解析词表（位于 256 字节头之后）
    // -----------------------------------------------------------
    uint32_t *tokenizer_ptr = (uint32_t *)(buffer + 256);
    uint32_t tokenizer_field_bytes = tokenizer_ptr[0];
    if (size < 256 + tokenizer_field_bytes) return -1;

    uint32_t *vocab_ptr = tokenizer_ptr + 1;
    g_llm.tokenizer.vocab_size = *vocab_ptr++;

    g_llm.tokenizer.token_list = (wchar_t **)calloc(g_llm.tokenizer.vocab_size, sizeof(wchar_t *));
    g_llm.tokenizer.unicode_charset = (wchar_t *)calloc(g_llm.tokenizer.vocab_size, sizeof(wchar_t));
    g_llm.tokenizer.unicode_to_id_map = am_llm_new_map(g_llm.tokenizer.vocab_size * 2 + 1);

    uint32_t byte_count = 0;
    uint32_t char_count = 0;
    while (byte_count < tokenizer_field_bytes - 8) {
        uint32_t token_header = *vocab_ptr++; byte_count += sizeof(uint32_t);
        uint32_t token_id = *vocab_ptr++; byte_count += sizeof(uint32_t);
        uint32_t token_length = token_header & 0xFF;

        if (token_id >= g_llm.tokenizer.vocab_size) return -1;

        wchar_t *token = (wchar_t *)calloc(token_length + 1, sizeof(wchar_t));
        if (token_length == 1) {
            g_llm.tokenizer.unicode_charset[char_count] = (wchar_t)*vocab_ptr;
            am_llm_map_set(g_llm.tokenizer.unicode_to_id_map, *vocab_ptr, token_id);
            char_count++;
        }
        for (uint32_t i = 0; i < token_length; i++) {
            token[i] = (wchar_t)*vocab_ptr;
            vocab_ptr++;
            byte_count += sizeof(uint32_t);
        }
        token[token_length] = 0;
        g_llm.tokenizer.token_list[token_id] = token;
    }

    // 构建 Trie 树（用于 tokenize）
    g_llm.tokenizer.vocab_trie = am_llm_new_trie(g_llm.tokenizer.vocab_size, 0);
    for (uint32_t i = 0; i < g_llm.tokenizer.vocab_size; i++) {
        wchar_t *utoken = g_llm.tokenizer.token_list[i];
        if (!utoken) continue;
        size_t len = wcslen(utoken);
        if (len > 1) {
            uint32_t *ids = am_llm_string_to_ids(g_llm.tokenizer.unicode_to_id_map, utoken);
            am_llm_add_token(g_llm.tokenizer.vocab_trie, ids, (uint32_t)len, i);
            free(ids);
        }
    }

    // -----------------------------------------------------------
    // 映射权重参数（F32，直接指向模型缓冲区）
    // -----------------------------------------------------------
    float *params = (float *)(buffer + 256 + tokenizer_field_bytes);

    uint32_t n_layer = g_llm.config.n_layer;
    uint32_t n_embd = g_llm.config.n_embd;
    uint32_t n_head = g_llm.config.n_head;
    uint32_t n_kv_head = g_llm.config.n_kv_head;
    uint32_t n_hidden = g_llm.config.n_hidden;
    uint32_t vocab_size = g_llm.config.vocab_size;
    uint32_t block_size = g_llm.config.block_size;
    uint32_t head_dim = g_llm.config.head_dim;
    if (head_dim == 0) head_dim = n_embd / n_head;
    uint32_t kv_dim = n_kv_head * head_dim;

    g_llm.weights.rms_norm_attn = params;
    params += (size_t)n_layer * n_embd;
    g_llm.weights.rms_norm_ffn = params;
    params += (size_t)n_layer * n_embd;
    g_llm.weights.rms_norm_final = params;
    params += n_embd;

    g_llm.weights.token_embedding = params;
    params += (size_t)vocab_size * n_embd;

    g_llm.weights.wq = params;
    params += (size_t)n_layer * (n_head * head_dim) * n_embd;
    g_llm.weights.wk = params;
    params += (size_t)n_layer * (n_kv_head * head_dim) * n_embd;
    g_llm.weights.wv = params;
    params += (size_t)n_layer * (n_kv_head * head_dim) * n_embd;
    g_llm.weights.wo = params;
    params += (size_t)n_layer * n_embd * (n_head * head_dim);

    g_llm.weights.w1 = params;
    params += (size_t)n_layer * n_hidden * n_embd;
    g_llm.weights.w2 = params;
    params += (size_t)n_layer * n_embd * n_hidden;
    g_llm.weights.w3 = params;
    params += (size_t)n_layer * n_hidden * n_embd;

    g_llm.weights.freq_cis_real = params;
    params += (size_t)block_size * head_dim / 2;
    g_llm.weights.freq_cis_imag = params;
    params += (size_t)block_size * head_dim / 2;

    g_llm.weights.token_classifier = g_llm.weights.token_embedding;

    // 记录长度，便于 get_param 时创建列表
    g_llm.weights.rms_norm_attn_len = (size_t)n_layer * n_embd;
    g_llm.weights.rms_norm_ffn_len = (size_t)n_layer * n_embd;
    g_llm.weights.rms_norm_final_len = n_embd;
    g_llm.weights.token_embedding_len = (size_t)vocab_size * n_embd;
    g_llm.weights.token_classifier_len = (size_t)vocab_size * n_embd;
    g_llm.weights.wq_len = (size_t)n_layer * n_embd * n_embd;
    g_llm.weights.wk_len = (size_t)n_layer * n_embd * kv_dim;
    g_llm.weights.wv_len = (size_t)n_layer * n_embd * kv_dim;
    g_llm.weights.wo_len = (size_t)n_layer * n_embd * n_embd;
    g_llm.weights.w1_len = (size_t)n_layer * n_hidden * n_embd;
    g_llm.weights.w2_len = (size_t)n_layer * n_embd * n_hidden;
    g_llm.weights.w3_len = (size_t)n_layer * n_hidden * n_embd;
    g_llm.weights.freq_cis_real_len = (size_t)block_size * head_dim / 2;
    g_llm.weights.freq_cis_imag_len = (size_t)block_size * head_dim / 2;

    (void)params; // 后续还有数据但不使用
    return 0;
}


// ===============================================================================
// Native 函数实现
// ===============================================================================

// (LLM.init modelFileBase64:String) : undefined
int32_t am_native_LLM_init(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_wstring_t *ws = NULL;
    if (!am_llm_pop_wstring(proc, &ws)) return -1;

    // 将宽字符串内容转换为多字节 char（base64 仅含 ASCII）
    char *base64_str = NULL;
    if (ws && ws->length > 0) {
        base64_str = (char *)malloc(ws->length + 1);
        if (!base64_str) return -1;
        for (size_t i = 0; i < ws->length; i++) {
            base64_str[i] = (char)am_value_to_wchar(ws->content[i]);
        }
        base64_str[ws->length] = '\0';
    }
    else {
        base64_str = strdup("");
    }

    size_t decoded_size = 0;
    uint8_t *decoded = am_llm_base64_decode(base64_str, &decoded_size);
    free(base64_str);
    if (!decoded) return -1;

    am_llm_unload();
    g_llm.buffer = decoded;
    g_llm.buffer_size = decoded_size;

    if (am_llm_parse_model(g_llm.buffer, g_llm.buffer_size) != 0) {
        am_llm_unload();
        return -1;
    }
    g_llm.loaded = 1;

    // 语句型 native（单值纪律 N2）：不产出有意义结果，压 1 个 #undefined 再返回
    if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// (LLM.get_config) : List
int32_t am_native_LLM_get_config(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!g_llm.loaded) return -1;

    am_list_t *lst = am_list_create(proc->heap_alloc, 9, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!lst) return -1;
    lst->length = 9;
    lst->children[0] = am_make_value_of_uint((am_uint_t)g_llm.config.block_size);
    lst->children[1] = am_make_value_of_uint((am_uint_t)g_llm.config.vocab_size);
    lst->children[2] = am_make_value_of_uint((am_uint_t)g_llm.config.n_layer);
    lst->children[3] = am_make_value_of_uint((am_uint_t)g_llm.config.n_embd);
    lst->children[4] = am_make_value_of_uint((am_uint_t)g_llm.config.n_head);
    lst->children[5] = am_make_value_of_uint((am_uint_t)g_llm.config.n_kv_head);
    lst->children[6] = am_make_value_of_uint((am_uint_t)g_llm.config.n_hidden);
    lst->children[7] = am_make_value_of_uint((am_uint_t)g_llm.config.is_shared_classifier);
    lst->children[8] = am_make_value_of_uint((am_uint_t)g_llm.config.head_dim);

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return -1; }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return -1;
    }

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// (LLM.get_param) : List
int32_t am_native_LLM_get_param(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!g_llm.loaded) return -1;

    am_handle_t handles[14];
    handles[0] = am_llm_make_float_list_handle(proc, g_llm.weights.rms_norm_attn, g_llm.weights.rms_norm_attn_len);
    handles[1] = am_llm_make_float_list_handle(proc, g_llm.weights.rms_norm_ffn, g_llm.weights.rms_norm_ffn_len);
    handles[2] = am_llm_make_float_list_handle(proc, g_llm.weights.rms_norm_final, g_llm.weights.rms_norm_final_len);
    handles[3] = am_llm_make_float_list_handle(proc, g_llm.weights.token_embedding, g_llm.weights.token_embedding_len);
    handles[4] = am_llm_make_float_list_handle(proc, g_llm.weights.wq, g_llm.weights.wq_len);
    handles[5] = am_llm_make_float_list_handle(proc, g_llm.weights.wk, g_llm.weights.wk_len);
    handles[6] = am_llm_make_float_list_handle(proc, g_llm.weights.wv, g_llm.weights.wv_len);
    handles[7] = am_llm_make_float_list_handle(proc, g_llm.weights.wo, g_llm.weights.wo_len);
    handles[8] = am_llm_make_float_list_handle(proc, g_llm.weights.w1, g_llm.weights.w1_len);
    handles[9] = am_llm_make_float_list_handle(proc, g_llm.weights.w2, g_llm.weights.w2_len);
    handles[10] = am_llm_make_float_list_handle(proc, g_llm.weights.w3, g_llm.weights.w3_len);
    handles[11] = am_llm_make_float_list_handle(proc, g_llm.weights.freq_cis_real, g_llm.weights.freq_cis_real_len);
    handles[12] = am_llm_make_float_list_handle(proc, g_llm.weights.freq_cis_imag, g_llm.weights.freq_cis_imag_len);
    handles[13] = am_llm_make_float_list_handle(proc, g_llm.weights.token_classifier, g_llm.weights.token_classifier_len);

    for (int i = 0; i < 14; i++) {
        if (handles[i] == AM_HANDLE_NULL) return -1;
    }

    am_list_t *lst = am_list_create(proc->heap_alloc, 14, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!lst) return -1;
    lst->length = 14;
    for (int i = 0; i < 14; i++) {
        lst->children[i] = am_make_value_of_handle(handles[i]);
    }

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) { am_list_destroy(proc->heap_alloc, lst); return -1; }
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                    am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(proc->heap_alloc, lst);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return -1;
    }

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// (LLM.encode text:String) : List<Number>
int32_t am_native_LLM_encode(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!g_llm.loaded) return -1;

    am_wstring_t *ws = NULL;
    if (!am_llm_pop_wstring(proc, &ws)) return -1;
    if (!ws) return -1;

    // 将 wstring 内容复制到 wchar_t 缓冲区
    wchar_t *text = (wchar_t *)malloc((ws->length + 1) * sizeof(wchar_t));
    if (!text) return -1;
    for (size_t i = 0; i < ws->length; i++) {
        text[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    }
    text[ws->length] = 0;

    uint32_t n_tokens = 0;
    uint32_t *ids = am_llm_encode_nano(&g_llm.tokenizer, text, &n_tokens);
    free(text);

    am_handle_t hd = am_llm_make_uint_list_handle(proc, ids, n_tokens);
    free(ids);
    if (hd == AM_HANDLE_NULL) return -1;

    if (am_process_push_operand(proc, am_make_value_of_handle(hd)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

// (LLM.decode token_id:Number) : String
int32_t am_native_LLM_decode(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!g_llm.loaded) return -1;

    am_float_t token_f;
    if (!am_llm_pop_number(proc, &token_f)) return -1;
    uint32_t token_id = (uint32_t)token_f;

    uint32_t ids[1] = { token_id };
    wchar_t *text = am_llm_decode_nano(&g_llm.tokenizer, ids, 1);

    int32_t ret = am_llm_push_wstring_buf(proc, text, wcslen(text));
    free(text);
    return ret;
}

// (LLM.matmul xout x w xout_offset w_offset n d) : undefined
// 语句型 native（单值纪律 N2）：副作用写回 xout 列表，不产出有意义结果，压 1 个 #undefined 再返回
int32_t am_native_LLM_matmul(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    am_float_t d_f, n_f, w_offset_f, xout_offset_f;
    am_list_t *xout = NULL, *x = NULL, *w = NULL;

    // 参数退栈顺序与参数列表顺序相反
    if (!am_llm_pop_number(proc, &d_f)) return -1;
    if (!am_llm_pop_number(proc, &n_f)) return -1;
    if (!am_llm_pop_number(proc, &w_offset_f)) return -1;
    if (!am_llm_pop_number(proc, &xout_offset_f)) return -1;
    if (!am_llm_pop_list(proc, &w)) return -1;
    if (!am_llm_pop_list(proc, &x)) return -1;
    if (!am_llm_pop_list(proc, &xout)) return -1;

    int d = (int)d_f;
    int n = (int)n_f;
    int w_offset = (int)w_offset_f;
    int xout_offset = (int)xout_offset_f;

    if (d > 0 && n > 0) {
        for (int i = 0; i < d; i++) {
            am_float_t val = 0.0;
            for (int j = 0; j < n; j++) {
                am_float_t wv = am_llm_list_get_float(w, (size_t)(w_offset + i * n + j));
                am_float_t xv = am_llm_list_get_float(x, (size_t)j);
                val += wv * xv;
            }
            if ((size_t)(xout_offset + i) < xout->length) {
                xout->children[xout_offset + i] = am_make_value_of_float(val);
            }
        }
    }

    if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
    am_process_step(proc);
    return 0;
}


// ===============================================================================
// Native 库注册表
// ===============================================================================

static const am_native_func_entry_t am_native_LLM_funcs[] = {
    { L"init",       am_native_LLM_init },
    { L"get_config", am_native_LLM_get_config },
    { L"get_param",  am_native_LLM_get_param },
    { L"encode",     am_native_LLM_encode },
    { L"decode",     am_native_LLM_decode },
    { L"matmul",     am_native_LLM_matmul },
};

const am_native_lib_entry_t am_native_LLM_lib = {
    L"LLM",
    am_native_LLM_funcs,
    sizeof(am_native_LLM_funcs) / sizeof(am_native_LLM_funcs[0])
};
