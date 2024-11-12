#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <locale.h>
#include <wchar.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#define uint32_t unsigned int

#define STATUS_PREFILLING (11)
#define STATUS_DECODING   (12)
#define STATUS_STOPPED    (21)
#define MAX_TOKEN_LENGTH  (17) // NOTE 虽然可以扫描词表得到该值，但是考虑到性能，设置为固定值（对于16384词表而言，至少17）


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
    float* data;       // memory mapped data pointer
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
} Nano_Context;

// ===============================================================================
// 词元编解码、分词器（基于Trie树）
// ===============================================================================

// 散列表

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

void free_map(struct Map *pm) {
    for(uint32_t i = 0; i < pm->bucket_num; i++) {
        struct Entry *current_entry = pm->buckets[i];
        while(NULL != current_entry) {
            struct Entry *next_entry = current_entry->next;
            free(current_entry);
            current_entry = next_entry;
        }
    }
    free(pm);
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

// Trie树

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

void free_trie(struct Trie *t) {
    if(NULL == t) {
        return;
    }
    else {
        for(uint32_t i = 0; i < t->vocab_size; i++) {
            free_trie(t->children[i]);
        }
        free(t);
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
    return -1;
}

// 分词器和词元编解码

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

uint32_t tokenize(struct Trie *vocab_trie, uint32_t *output_token_ids, const uint32_t *input_char_ids, uint32_t input_length, uint32_t max_token_length) {
    uint32_t token_count = 0;
    uint32_t pos = 0;
    // uint32_t prefix[MAX_TOKEN_LENGTH];
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
            free(prefix);
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

uint32_t *encode(Tokenizer *t, wchar_t *text, uint32_t *n_tokens_ptr) {
    uint32_t *input_ids = string_to_ids(t->unicode_to_id_map, text);
    uint32_t *optput_ids = (uint32_t *)calloc(wcslen(text), sizeof(uint32_t *));
    uint32_t token_count = tokenize(t->vocab_trie, optput_ids, input_ids, wcslen(text), MAX_TOKEN_LENGTH);
    free(input_ids);
    *n_tokens_ptr = token_count;
    return optput_ids;
}


// ===============================================================================
// 模型文件解析·内存管理
// ===============================================================================

void malloc_fwd_buffer(FwdBuffer *s, LLM_Config *llm_cfg) {
    uint32_t kv_dim = (llm_cfg->n_embd * llm_cfg->n_kv_head) / llm_cfg->n_head;
    s->x       = calloc(llm_cfg->n_embd, sizeof(float));
    s->xb      = calloc(llm_cfg->n_embd, sizeof(float));
    s->xb2     = calloc(llm_cfg->n_embd, sizeof(float));
    s->hb      = calloc(llm_cfg->n_hidden, sizeof(float));
    s->hb2     = calloc(llm_cfg->n_hidden, sizeof(float));
    s->q       = calloc(llm_cfg->n_embd, sizeof(float));
    s->k_cache = calloc(llm_cfg->n_layer * llm_cfg->block_size * kv_dim, sizeof(float));
    s->v_cache = calloc(llm_cfg->n_layer * llm_cfg->block_size * kv_dim, sizeof(float));
    s->att     = calloc(llm_cfg->n_head * llm_cfg->block_size, sizeof(float));
    s->logits  = calloc(llm_cfg->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
        !s->k_cache || !s->v_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}


void malloc_fwd_buffer_with_lora(FwdBuffer *s, LLM_Config *llm_cfg, LoRA_Config *lora_cfg) {
    uint32_t kv_dim = (llm_cfg->n_embd * llm_cfg->n_kv_head) / llm_cfg->n_head;
    s->q0 = calloc(lora_cfg->lora_rank, sizeof(float));
    s->k0 = calloc(lora_cfg->lora_rank, sizeof(float));
    s->v0 = calloc(lora_cfg->lora_rank, sizeof(float));
    s->o0 = calloc(lora_cfg->lora_rank, sizeof(float));
    s->q1 = calloc(llm_cfg->n_embd, sizeof(float));
    s->k1 = calloc(kv_dim, sizeof(float));
    s->v1 = calloc(kv_dim, sizeof(float));
    s->o1 = calloc(llm_cfg->n_embd, sizeof(float));
    // ensure all mallocs went fine
    if (!s->q0 || !s->k0 || !s->v0 || !s->o0 ||
        !s->q1 || !s->k1 || !s->v1 || !s->o1 ) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}


void free_fwd_buffer(FwdBuffer* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->k_cache);
    free(s->v_cache);
    // free(s->q0); free(s->k0); free(s->v0); free(s->o0);
    // free(s->q1); free(s->k1); free(s->v1); free(s->o1);
}

void memory_map_params(LLM_Param *w, LLM_Config* p, float* ptr, int shared_weights) {
    int head_size = p->n_embd / p->n_head;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layer = p->n_layer;

    w->token_embedding = ptr;ptr += p->vocab_size * p->n_embd;
    w->rms_norm_attn = ptr;  ptr += n_layer * p->n_embd;
    w->wq = ptr;             ptr += n_layer * p->n_embd * (p->n_head * head_size);
    w->wk = ptr;             ptr += n_layer * p->n_embd * (p->n_kv_head * head_size);
    w->wv = ptr;             ptr += n_layer * p->n_embd * (p->n_kv_head * head_size);
    w->wo = ptr;             ptr += n_layer * (p->n_head * head_size) * p->n_embd;
    w->rms_norm_ffn = ptr;   ptr += n_layer * p->n_embd;
    w->w1 = ptr;             ptr += n_layer * p->n_embd * p->n_hidden;
    w->w2 = ptr;             ptr += n_layer * p->n_hidden * p->n_embd;
    w->w3 = ptr;             ptr += n_layer * p->n_embd * p->n_hidden;
    w->rms_norm_final = ptr; ptr += p->n_embd;
    w->freq_cis_real = ptr;  ptr += p->block_size * head_size / 2;
    w->freq_cis_imag = ptr;  ptr += p->block_size * head_size / 2;
    w->token_classifier = shared_weights ? w->token_embedding : ptr;
}


void read_model_file_mmap(char* model_path, LLM *llm, Tokenizer *tk) {

    LLM_Config *config = &(llm->config);
    LLM_Param *params = &(llm->params);
    int *fd = &(llm->fd);
    float** data = &(llm->data);
    ssize_t* file_size = &(llm->file_size);

    FILE *file = fopen(model_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", model_path); exit(EXIT_FAILURE); }

    // 读取文件头
    const uint32_t header_byte_length = 256;
    const uint32_t header_uint_length = header_byte_length / sizeof(uint32_t);
    uint32_t *header = (uint32_t *)calloc(header_uint_length, sizeof(uint32_t));

    if(fread(header, sizeof(uint32_t), header_uint_length, file) != header_uint_length) { exit(EXIT_FAILURE); }

    uint32_t offset = 0;

    uint32_t magic_number_0 = header[offset]; offset++;
    uint32_t magic_number_1 = header[offset]; offset++;

    uint32_t major_version  = header[offset]; offset++;
    uint32_t minor_version  = header[offset]; offset++;

    uint32_t model_type     = header[offset]; offset++;
    uint32_t config_length  = header[offset]; offset++;

    config->block_size      = header[offset]; offset++;
    config->vocab_size      = header[offset]; offset++;
    config->n_layer         = header[offset]; offset++;
    config->n_embd          = header[offset]; offset++;
    config->n_head          = header[offset]; offset++;
    config->n_kv_head       = header[offset]; offset++;
    config->n_hidden        = header[offset]; offset++;
    config->is_shared_classifier = header[offset]; offset++;

    // 读取参数部分的长度字段
    // unsigned long long param_num = 0;
    // if(fread(&param_num, sizeof(unsigned long long), 1, file) != 1) { exit(EXIT_FAILURE); }

    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    // memory map the Transformer parameters into the data pointer
    *fd = open(model_path, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    // 解析词表，同时构建trie树和hashmap

    uint32_t *tokenzier_ptr = (uint32_t *)(*data) + (header_byte_length / sizeof(uint32_t));
    uint32_t tokenizer_field_bytes = *tokenzier_ptr;
    uint32_t *vocab_ptr = tokenzier_ptr + 1;

    uint32_t byte_count = 0;
    uint32_t char_count = 0;

    tk->vocab_size = *vocab_ptr; vocab_ptr++;
    printf("Vocab size = %d\n", tk->vocab_size);

    tk->token_list        = (wchar_t **)calloc(tk->vocab_size, sizeof(wchar_t *));
    tk->unicode_charset   = (wchar_t  *)calloc(tk->vocab_size, sizeof(wchar_t));
    tk->unicode_to_id_map = new_map(tk->vocab_size);
    tk->token_to_id_map   = new_map(tk->vocab_size);
    tk->vocab_trie        = new_trie(tk->vocab_size, 0);

    while(byte_count < tokenizer_field_bytes - 8) { // 不含tokenizer_field_bytes和vocab_size字段的8个字节
        uint32_t token_header = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);
        uint32_t token_id     = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);

        // NOTE Little endian 小端序！如果按照uint32解析，顺序是 MSB(reserved_1 reserved_0 is_special token_length)LSB
        uint32_t reserved_1   = (token_header & 0xff000000) >> 24;
        uint32_t reserved_0   = (token_header & 0x00ff0000) >> 16;
        uint32_t is_special   = (token_header & 0x0000ff00) >> 8;
        uint32_t token_length = (token_header & 0x000000ff);

        wchar_t *token = (wchar_t *)calloc(token_length+1, sizeof(wchar_t));
        // 如果是单个字符，则加入unicode_charset
        if(token_length == 1) {
            tk->unicode_charset[char_count] = *vocab_ptr;
            map_set(tk->unicode_to_id_map, *vocab_ptr, token_id);
            char_count++;
        }
        for(uint32_t i = 0; i < token_length; i++) {
            token[i] = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);
        }
        token[token_length] = 0;
        tk->token_list[token_id] = token;
    }

    // 构建trie树
    for(uint32_t i = 0; i < tk->vocab_size; i++) {
        wchar_t *utoken = tk->token_list[i];
        uint32_t len = wcslen(utoken);
        if(len > 1) {
            uint32_t *ids = string_to_ids(tk->unicode_to_id_map, utoken);
            add_token(tk->vocab_trie, ids, len, i);
        }
    }


    // 解析模型参数

    float* param_ptr = (float*)(tokenzier_ptr + tokenizer_field_bytes/sizeof(uint32_t));
    memory_map_params(params, config, param_ptr, config->is_shared_classifier);

}






void parse_model_file(char* buffer, LLM *llm, Tokenizer *tk) {

    LLM_Config *config = &(llm->config);
    LLM_Param *params = &(llm->params);

    // 读取文件头
    const uint32_t header_byte_length = 256;
    const uint32_t header_uint_length = header_byte_length / sizeof(uint32_t);
    uint32_t *header = (uint32_t *)buffer;

    uint32_t offset = 0;

    uint32_t magic_number_0 = header[offset]; offset++;
    uint32_t magic_number_1 = header[offset]; offset++;

    uint32_t major_version  = header[offset]; offset++;
    uint32_t minor_version  = header[offset]; offset++;

    uint32_t model_type     = header[offset]; offset++;
    uint32_t config_length  = header[offset]; offset++;

    config->block_size      = header[offset]; offset++;
    config->vocab_size      = header[offset]; offset++;
    config->n_layer         = header[offset]; offset++;
    config->n_embd          = header[offset]; offset++;
    config->n_head          = header[offset]; offset++;
    config->n_kv_head       = header[offset]; offset++;
    config->n_hidden        = header[offset]; offset++;
    config->is_shared_classifier = header[offset]; offset++;

    // 解析词表，同时构建trie树和hashmap

    uint32_t *tokenzier_ptr = (uint32_t *)(buffer) + (header_byte_length / sizeof(uint32_t));
    uint32_t tokenizer_field_bytes = *tokenzier_ptr;
    uint32_t *vocab_ptr = tokenzier_ptr + 1;

    uint32_t byte_count = 0;
    uint32_t char_count = 0;

    tk->vocab_size = *vocab_ptr; vocab_ptr++;
    printf("Vocab size = %d\n", tk->vocab_size);

    tk->token_list        = (wchar_t **)calloc(tk->vocab_size, sizeof(wchar_t *));
    tk->unicode_charset   = (wchar_t  *)calloc(tk->vocab_size, sizeof(wchar_t));
    tk->unicode_to_id_map = new_map(tk->vocab_size);
    tk->token_to_id_map   = new_map(tk->vocab_size);
    tk->vocab_trie        = new_trie(tk->vocab_size, 0);

    while(byte_count < tokenizer_field_bytes - 8) { // 不含tokenizer_field_bytes和vocab_size字段的8个字节
        uint32_t token_header = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);
        uint32_t token_id     = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);

        // NOTE Little endian 小端序！如果按照uint32解析，顺序是 MSB(reserved_1 reserved_0 is_special token_length)LSB
        uint32_t reserved_1   = (token_header & 0xff000000) >> 24;
        uint32_t reserved_0   = (token_header & 0x00ff0000) >> 16;
        uint32_t is_special   = (token_header & 0x0000ff00) >> 8;
        uint32_t token_length = (token_header & 0x000000ff);

        wchar_t *token = (wchar_t *)calloc(token_length+1, sizeof(wchar_t));
        // 如果是单个字符，则加入unicode_charset
        if(token_length == 1) {
            tk->unicode_charset[char_count] = *vocab_ptr;
            map_set(tk->unicode_to_id_map, *vocab_ptr, token_id);
            char_count++;
        }
        for(uint32_t i = 0; i < token_length; i++) {
            token[i] = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);
        }
        token[token_length] = 0;
        tk->token_list[token_id] = token;
    }

    // 构建trie树
    for(uint32_t i = 0; i < tk->vocab_size; i++) {
        wchar_t *utoken = tk->token_list[i];
        uint32_t len = wcslen(utoken);
        if(len > 1) {
            uint32_t *ids = string_to_ids(tk->unicode_to_id_map, utoken);
            add_token(tk->vocab_trie, ids, len, i);
        }
    }


    // 解析模型参数

    float* param_ptr = (float*)(tokenzier_ptr + tokenizer_field_bytes/sizeof(uint32_t));
    memory_map_params(params, config, param_ptr, config->is_shared_classifier);
    printf("Finished\n");
}






void read_lora_file(char* lora_path, LoRA *lora, LLM *llm) {
    LoRA_Config *lora_cfg    = &(lora->config);
    LoRA_Param  *lora_params = &(lora->params);
    LLM_Config  *llm_cfg     = &(llm->config);

    FILE *file = fopen(lora_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open LoRA module file %s\n", lora_path); exit(EXIT_FAILURE); }

    // 读取文件头
    const uint32_t header_byte_length = 256;
    const uint32_t header_uint_length = header_byte_length / sizeof(uint32_t);
    uint32_t *header = (uint32_t *)calloc(header_uint_length, sizeof(uint32_t));

    if(fread(header, sizeof(uint32_t), header_uint_length, file) != header_uint_length) { exit(EXIT_FAILURE); }

    uint32_t offset = 0;

    uint32_t magic_number_0 = header[offset]; offset++;
    uint32_t magic_number_1 = header[offset]; offset++;

    uint32_t major_version  = header[offset]; offset++;
    uint32_t minor_version  = header[offset]; offset++;

    uint32_t model_type     = header[offset]; offset++;
    uint32_t config_length  = header[offset]; offset++;

    lora_cfg->lora_rank     = header[offset]; offset++;
    lora_cfg->lora_alpha    = header[offset]; offset++;
    lora_cfg->n_layer       = header[offset]; offset++;
    lora_cfg->n_embd        = header[offset]; offset++;
    lora_cfg->n_head        = header[offset]; offset++;
    lora_cfg->n_kv_head     = header[offset]; offset++;
    lora_cfg->n_hidden      = header[offset]; offset++;
    lora_cfg->lora_config   = header[offset]; offset++;

    // 校验LoRA模块与基座模型是否匹配
    if (llm_cfg->n_layer != lora_cfg->n_layer ||
        llm_cfg->n_embd != lora_cfg->n_embd ||
        llm_cfg->n_head != lora_cfg->n_head ||
        llm_cfg->n_kv_head != lora_cfg->n_kv_head ||
        llm_cfg->n_hidden != lora_cfg->n_hidden) {
        fprintf(stderr, "Error: LoRA module does not fit the base model.");
        exit(EXIT_FAILURE);
    }

    // 读取参数部分的长度字段
    // unsigned long long param_num = 0;
    // if(fread(&param_num, sizeof(unsigned long long), 1, file) != 1) { exit(EXIT_FAILURE); }

    // 读取LoRA模块参数到内存

    uint32_t head_dim = llm_cfg->n_embd / llm_cfg->n_head;
    uint32_t kv_dim = head_dim * llm_cfg->n_kv_head;

    uint32_t wq_lora_a_len = llm_cfg->n_layer * lora_cfg->lora_rank * llm_cfg->n_embd;
    uint32_t wq_lora_b_len = llm_cfg->n_layer * llm_cfg->n_embd * lora_cfg->lora_rank;
    uint32_t wk_lora_a_len = llm_cfg->n_layer * lora_cfg->lora_rank * llm_cfg->n_embd;
    uint32_t wk_lora_b_len = llm_cfg->n_layer * kv_dim * lora_cfg->lora_rank;
    uint32_t wv_lora_a_len = llm_cfg->n_layer * lora_cfg->lora_rank * llm_cfg->n_embd;
    uint32_t wv_lora_b_len = llm_cfg->n_layer * kv_dim * lora_cfg->lora_rank;
    uint32_t wo_lora_a_len = llm_cfg->n_layer * lora_cfg->lora_rank * llm_cfg->n_embd;
    uint32_t wo_lora_b_len = llm_cfg->n_layer * llm_cfg->n_embd * lora_cfg->lora_rank;

    lora_params->wq_lora_a = (float *)calloc(wq_lora_a_len, sizeof(float));
    lora_params->wq_lora_b = (float *)calloc(wq_lora_b_len, sizeof(float));
    lora_params->wk_lora_a = (float *)calloc(wk_lora_a_len, sizeof(float));
    lora_params->wk_lora_b = (float *)calloc(wk_lora_b_len, sizeof(float));
    lora_params->wv_lora_a = (float *)calloc(wv_lora_a_len, sizeof(float));
    lora_params->wv_lora_b = (float *)calloc(wv_lora_b_len, sizeof(float));
    lora_params->wo_lora_a = (float *)calloc(wo_lora_a_len, sizeof(float));
    lora_params->wo_lora_b = (float *)calloc(wo_lora_b_len, sizeof(float));

    if(fread(lora_params->wq_lora_a, sizeof(float), wq_lora_a_len, file) != wq_lora_a_len) { exit(EXIT_FAILURE); }
    if(fread(lora_params->wq_lora_b, sizeof(float), wq_lora_b_len, file) != wq_lora_b_len) { exit(EXIT_FAILURE); }
    if(fread(lora_params->wk_lora_a, sizeof(float), wk_lora_a_len, file) != wk_lora_a_len) { exit(EXIT_FAILURE); }
    if(fread(lora_params->wk_lora_b, sizeof(float), wk_lora_b_len, file) != wk_lora_b_len) { exit(EXIT_FAILURE); }
    if(fread(lora_params->wv_lora_a, sizeof(float), wv_lora_a_len, file) != wv_lora_a_len) { exit(EXIT_FAILURE); }
    if(fread(lora_params->wv_lora_b, sizeof(float), wv_lora_b_len, file) != wv_lora_b_len) { exit(EXIT_FAILURE); }
    if(fread(lora_params->wo_lora_a, sizeof(float), wo_lora_a_len, file) != wo_lora_a_len) { exit(EXIT_FAILURE); }
    if(fread(lora_params->wo_lora_b, sizeof(float), wo_lora_b_len, file) != wo_lora_b_len) { exit(EXIT_FAILURE); }

    fclose(file);
}

void load_llm(LLM *llm, Tokenizer *tk, char *model_path) {

#ifdef NANO_USE_MMAP
    read_model_file_mmap(model_path, llm, tk);
#else
    FILE *file = fopen(model_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open LoRA module file %s\n", model_path); exit(EXIT_FAILURE); }

    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long long unsigned int file_size = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc(file_size + 1);
    if (!buffer)  { fprintf(stderr, "malloc failed.\n"); exit(EXIT_FAILURE); }

    if(fread(buffer, sizeof(char), file_size, file) != file_size) { exit(EXIT_FAILURE); }

    parse_model_file(buffer, llm, tk);

    fclose(file);
#endif

    malloc_fwd_buffer(&llm->state, &llm->config);
}

void load_llm_from_buffer(LLM *llm, Tokenizer *tk, char *buffer) {
    parse_model_file(buffer, llm, tk);
    malloc_fwd_buffer(&llm->state, &llm->config);
}

void free_llm(LLM* t, Tokenizer *tk) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }

    free_tokenizer(tk);
    free_fwd_buffer(&t->state);
}

LoRA *load_lora(LLM *llm, char *lora_path) {
    LoRA *p_lora = (LoRA *)calloc(1, sizeof(LoRA));
    read_lora_file(lora_path, p_lora, llm);
    malloc_fwd_buffer_with_lora(&llm->state, &llm->config, &p_lora->config);
    return p_lora;
}

void free_lora(LLM *llm, LoRA *lora) {
    free(&(lora->config));
    free(lora->params.wq_lora_a);
    free(lora->params.wq_lora_b);
    free(lora->params.wk_lora_a);
    free(lora->params.wk_lora_b);
    free(lora->params.wv_lora_a);
    free(lora->params.wv_lora_b);
    free(lora->params.wo_lora_a);
    free(lora->params.wo_lora_b);

    free(llm->state.q0); free(llm->state.k0); free(llm->state.v0); free(llm->state.o0);
    free(llm->state.q1); free(llm->state.k1); free(llm->state.v1); free(llm->state.o1);
}


// ===============================================================================
// 基础算子
//   所有算子都是C风格的：函数本身不返回值，通过参数引用的buffer来传递计算结果。
// ===============================================================================

void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void scale(float *a, float k, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] * k;
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}



float* llm_forward(uint32_t token, uint32_t pos, LLM* llm, LoRA *lora) {

    // a few convenience variables
    LLM_Config *cfg = &llm->config;
    LLM_Param *w = &llm->params;
    FwdBuffer *s = &llm->state;

    int use_lora = (NULL == lora) ? 0 : 1;
    LoRA_Param *a;
    uint32_t lora_rank;
    uint32_t lora_alpha;
    if(use_lora == 1) {
        a = &(lora->params);
        lora_rank = lora->config.lora_rank;
        lora_alpha = lora->config.lora_alpha;
    }

    float *x = s->x;
    int n_embd = cfg->n_embd;
    int kv_dim = (cfg->n_embd * cfg->n_kv_head) / cfg->n_head;
    int kv_mul = cfg->n_head / cfg->n_kv_head; // integer multiplier of the kv sharing in multiquery
    int n_hidden =  cfg->n_hidden;
    int head_dim = n_embd / cfg->n_head;

    // copy the token embedding into x
    float* content_row = w->token_embedding + token * n_embd;
    memcpy(x, content_row, n_embd*sizeof(*x));

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float *freq_cis_real_row = w->freq_cis_real + pos * head_dim / 2;
    float *freq_cis_imag_row = w->freq_cis_imag + pos * head_dim / 2;

    // forward all the layers
    for(unsigned long long l = 0; l < cfg->n_layer; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_norm_attn + l*n_embd, n_embd);

        // key and value point to the kv cache
        int loff = l * cfg->block_size * kv_dim; // kv cache layer offset for convenience
        s->k = s->k_cache + loff + pos * kv_dim;
        s->v = s->v_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*n_embd*n_embd, n_embd, n_embd);
        matmul(s->k, s->xb, w->wk + l*n_embd*kv_dim, n_embd, kv_dim);
        matmul(s->v, s->xb, w->wv + l*n_embd*kv_dim, n_embd, kv_dim);

        if(use_lora == 1) {
            matmul(s->q0, s->xb, a->wq_lora_a + l * lora_rank * n_embd, n_embd, lora_rank);
            matmul(s->k0, s->xb, a->wk_lora_a + l * lora_rank * n_embd, n_embd, lora_rank);
            matmul(s->v0, s->xb, a->wv_lora_a + l * lora_rank * n_embd, n_embd, lora_rank);

            matmul(s->q1, s->q0, a->wq_lora_b + l * n_embd * lora_rank, lora_rank, n_embd);
            matmul(s->k1, s->k0, a->wk_lora_b + l * kv_dim * lora_rank, lora_rank, kv_dim);
            matmul(s->v1, s->v0, a->wv_lora_b + l * kv_dim * lora_rank, lora_rank, kv_dim);

            scale(s->q1, ((float)lora_alpha / (float)lora_rank), n_embd);
            scale(s->k1, ((float)lora_alpha / (float)lora_rank), kv_dim);
            scale(s->v1, ((float)lora_alpha / (float)lora_rank), kv_dim);

            accum(s->q, s->q1, n_embd);
            accum(s->k, s->k1, kv_dim);
            accum(s->v, s->v1, kv_dim);
        }

        // RoPE旋转位置编码实现方式1：使用模型提供的旋转系数
        for (int h = 0; h < cfg->n_head; h++) {
            float *q = s->q + h * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                float q0 = q[i];
                float q1 = q[i + 1];
                float fcr = freq_cis_real_row[i / 2];
                float fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
            }
        }
        for (int m = 0; m < cfg->n_kv_head; m++) {
            float *k = s->k + m * head_dim;
            for (int i = 0; i < head_dim; i += 2) {
                float k0 = k[i];
                float k1 = k[i + 1];
                float fcr = freq_cis_real_row[i / 2];
                float fci = freq_cis_imag_row[i / 2];
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // RoPE旋转位置编码实现方式2：直接计算旋转系数
        /*
        for (int i = 0; i < n_embd; i+=2) {
            int head_dim = i % head_dim;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_dim);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
        */

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < cfg->n_head; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_dim;
            // attention scores for this head
            float* att = s->att + h * cfg->block_size;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->k_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_dim);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_dim;
            memset(xb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->v_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_dim; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*n_embd*n_embd, n_embd, n_embd);

        // 计算output的低秩分解分支，并将其累加到原来的输出上
        if(use_lora == 1) {
            matmul(s->o0, s->xb, a->wo_lora_a + l * lora_rank * n_embd, n_embd, lora_rank);
            matmul(s->o1, s->o0, a->wo_lora_b + l * n_embd * lora_rank, lora_rank, n_embd);
            scale(s->o1, ((float)lora_alpha / (float)lora_rank), n_embd);
            accum(s->xb2, s->o1, n_embd);
        }

        // residual connection back into x
        for (int i = 0; i < n_embd; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_norm_ffn + l*n_embd, n_embd);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*n_embd*n_hidden, n_embd, n_hidden);
        matmul(s->hb2, s->xb, w->w3 + l*n_embd*n_hidden, n_embd, n_hidden);

        // SwiGLU non-linearity
        for (int i = 0; i < n_hidden; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*n_embd*n_hidden, n_hidden, n_embd);

        // residual connection
        for (int i = 0; i < n_embd; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_norm_final, n_embd);

    // classifier into logits
    matmul(s->logits, x, w->token_classifier, cfg->n_embd, cfg->vocab_size);
    return s->logits;
}


// ===============================================================================
// 采样策略
// ===============================================================================

// 贪心采样：返回概率最大的下标
int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// 概率采样（香草味的）
// sample index from probabilities (they must sum to 1!)
// coin is a random number in [0, 1), usually from random_f32()
int sample_multinomial(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

// Top-P采样（核采样）：只在累积概率达到p的概率最高的若干个词元中采样
int sample_top_p(float* probabilities, int n, float top_p, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - top_p) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds top_p
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > top_p) {
            last_idx = i;
            break; // we've exceeded top_p by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

Sampler *build_sampler(int vocab_size, float repetition_penalty, float temperature, float top_p, uint32_t top_k, unsigned long long rng_seed) {
    Sampler *sampler = (Sampler *)calloc(1, sizeof(Sampler));
    sampler->vocab_size = vocab_size;
    sampler->repetition_penalty = repetition_penalty;
    sampler->temperature = temperature;
    sampler->top_p = top_p;
    sampler->top_k = top_k;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
    return sampler;
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}


// ===============================================================================
// 工具函数
// ===============================================================================

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

static uint32_t last_output_length = 0;
uint32_t typewriter(wchar_t *output_text, uint32_t status) {
    if(status == STATUS_DECODING) {
        // printf("%ls", output_text);
        uint32_t output_length = wcslen(output_text);
        for(uint32_t i = last_output_length; i < output_length; i++) {
            printf("%lc", output_text[i]);
        }
        fflush(stdout);
        last_output_length = output_length;
    }
    return 0;
}

uint32_t report(float tps, uint32_t status) {
    printf("\nTPS = %f\n", tps);
    return 0;
}


// ===============================================================================
// 自回归文本生成
// ===============================================================================

uint32_t generate_next_token(
    Nano_Context ctx,
    uint32_t *output_ids,
    uint32_t pos,
    int is_prefilling
) {

    LLM *llm = ctx.llm;
    LoRA *lora = ctx.lora;
    Tokenizer *tokenizer = ctx.tokenizer;
    Sampler *sampler = ctx.sampler;

    uint32_t next_token = output_ids[pos];

    float* logits = llm_forward(next_token, pos, llm, lora);

    // Pre-fill: if we are still processing the input prompt, force the next prompt token
    if (is_prefilling == 1) {
        next_token = output_ids[pos + 1];
        return next_token;
    }
    // Auto-regressive Decode
    else {
        // 复读惩罚：对过往出现过的词元施加惩罚，词元出现得越多，概率越低: ref arxiv:1909.05858
        uint32_t *tokenset = (uint32_t *)calloc(sampler->vocab_size, sizeof(uint32_t));
        for(uint32_t i = 0; i < pos; i++) {
            tokenset[output_ids[i]] = 1;  // 1表示output_ids中出现了这个token
        }
        for(uint32_t id = 0; id < sampler->vocab_size; id++) {
            if(tokenset[id] == 1) {
                logits[id] /= sampler->repetition_penalty;
            }
        }
        free(tokenset);

        // 温度采样：当温度设为0时，退化为贪心采样
        if (sampler->temperature == 0.0f) {
            next_token = sample_argmax(logits, sampler->vocab_size);
        }
        else {
            for(uint32_t q = 0; q < sampler->vocab_size; q++) {
                logits[q] /= sampler->temperature;
            }

            softmax(logits, sampler->vocab_size);

            // flip a (float) coin (this is our source of entropy for sampling)
            float coin = random_f32(&sampler->rng_state);

            if (sampler->top_p > 0 || sampler->top_p < 1) {
                next_token = sample_top_p(logits, sampler->vocab_size, sampler->top_p, sampler->probindex, coin);
            }
            else {
                next_token = sample_multinomial(logits, sampler->vocab_size, coin);
            }
        }
    }

    return next_token;
}


void generate(
    Nano_Context ctx,

    wchar_t *prompt,

    uint32_t max_seq_len,

    uint32_t (*on_running)(wchar_t*, uint32_t),
    uint32_t (*on_finished)(float, uint32_t)
) {

    Tokenizer *tokenizer = ctx.tokenizer;
    Sampler *sampler = ctx.sampler;

    wchar_t *empty_prompt = L"";
    if (prompt == NULL) { prompt = empty_prompt; }

    long t_0 = 0;
    long t_1 = 0;

    uint32_t *output_ids = (uint32_t *)calloc(max_seq_len + 1, sizeof(uint32_t));
    uint32_t output_count = 0;

    uint32_t num_prompt_tokens = 0;
    uint32_t *prompt_tokens = encode(tokenizer, prompt, &num_prompt_tokens);

    for(int i = 0; i < num_prompt_tokens; i++) {
        output_ids[i] = prompt_tokens[i];
    }

    uint32_t next_token = prompt_tokens[0]; // kick off with the first token in the prompt
    uint32_t pos = 0;     // position in the sequence

    while (pos < max_seq_len) {

        int is_prefilling = (pos < num_prompt_tokens - 1) ? 1 : 0;

        next_token = generate_next_token(ctx, output_ids, pos, is_prefilling);

        if(is_prefilling == 0) {
            output_ids[num_prompt_tokens + output_count++] = next_token;
            wchar_t *output_text = decode(tokenizer, output_ids + num_prompt_tokens, output_count);
            on_running(output_text, STATUS_DECODING);
            free(output_text);
        }
        else if(is_prefilling == 0) {
            on_running(NULL, STATUS_PREFILLING);
        }

        pos++;

        if(next_token == 0 || next_token == 3) break;

        if (t_0 == 0) { t_0 = time_in_ms(); }
    }

    t_1 = time_in_ms();
    float tps = (pos-1) / (double)(t_1 - t_0) * 1000;
    on_finished(tps, STATUS_STOPPED);

    printf("\n");
    free(output_ids);
    free(prompt_tokens);
}


void show_usage() {
    fprintf(stderr, "NanoLM - Inference Engine\n");
    fprintf(stderr, "  BD4SUR 2024-11\n");
    fprintf(stderr, "  forked from github.com/karpathy/llama2.c\n\n");
    fprintf(stderr, "Usage:   run <model_path> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"我是复读机，你是什么？\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -l <string> path to LoRA module file, default null\n");
    fprintf(stderr, "  -r <float>  repetition penalty in (0,inf], default 1.11\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in (0,1) default 0.5\n");
    fprintf(stderr, "  -k <int>    k value in top-k sampling in [0, vocab_size) default 0 (no use)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 512. 0 = max_seq_len\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -i <string> prompt (in chat/instruct mode)\n");
    fprintf(stderr, "  -g <string> prompt (in text generation mode)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if(!setlocale(LC_CTYPE, "")) {
        fprintf(stderr, "Can't set the specified locale! Check LANG, LC_CTYPE, LC_ALL.\n");
        return -1;
    }

    char *MODEL_PATH = "/home/bd4sur/ai/Nano/checkpoint/1-基础模型-99000.bin";
    char *LORA_PATH  = NULL; // "/home/bd4sur/ai/Nano/checkpoint/2-插件-猫娘.bin";

    float rep_pnty = 1.11;
    float temperature = 1.1;
    float top_p = 0.5;
    int   top_k = 0;
    int   max_seq_len = 512;
    int   random_seed = (unsigned int)time(NULL);

    wchar_t *prompt = L"<|instruct_mark|>你是Nano，是<|BD4SUR|>开发的大模型，是一只电子鹦鹉<|response_mark|>";

    if(argc >= 2) { MODEL_PATH = argv[1]; } else { show_usage(); }
    for(int i = 2; i < argc; i += 2) {
        // do some basic validation
        if (i + 1 >= argc) { show_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { show_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { show_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if      (argv[i][1] == 'l') { LORA_PATH = argv[i + 1]; }
        else if (argv[i][1] == 'r') { rep_pnty = atof(argv[i + 1]); }
        else if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { top_p = atof(argv[i + 1]); }
        else if (argv[i][1] == 'k') { top_k = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { max_seq_len = atoi(argv[i + 1]); }
        else if (argv[i][1] == 's') { random_seed = atoi(argv[i + 1]); }

        else if (argv[i][1] == 'i') {
            wchar_t *winput = (wchar_t *)calloc(max_seq_len, sizeof(wchar_t));
            uint32_t plen = mbstowcs(winput, argv[i + 1], max_seq_len);
            wchar_t *prompt_template = L"<|instruct_mark|><|response_mark|>";
            wchar_t *wprompt = (wchar_t *)calloc(plen + 35, sizeof(wchar_t));
            for(uint32_t i = 0; i < plen + 35; i++) {
                if(i >= 0 && i < 17) {
                    wprompt[i] = prompt_template[i];
                }
                else if(i >= 17 && i < plen+17) {
                    wprompt[i] = winput[i-17];
                }
                else if(i >= plen+17) {
                    wprompt[i] = prompt_template[i - plen];
                }
            }
            free(winput);
            prompt = wprompt;
        }
        else if (argv[i][1] == 'g') {
            prompt = (wchar_t *)argv[i + 1];
        }

        else { show_usage(); }
    }
    printf("Prompt > %ls\n", prompt);


    Nano_Context ctx;

    ctx.llm = (LLM *)calloc(1, (sizeof(LLM)));
    ctx.lora = NULL; // (LoRA *)calloc(1, (sizeof(LoRA *)));
    ctx.tokenizer = (Tokenizer *)calloc(1, (sizeof(Tokenizer)));

    load_llm(ctx.llm, ctx.tokenizer, MODEL_PATH);

    ctx.sampler = build_sampler(ctx.llm->config.vocab_size, rep_pnty, temperature, top_p, top_k, random_seed);

    if(NULL != LORA_PATH) ctx.lora = load_lora(ctx.llm, LORA_PATH);

    last_output_length = 0;
    generate(ctx, prompt, max_seq_len, typewriter, report);

    if(NULL != LORA_PATH) free_lora(ctx.llm, ctx.lora);

    free(prompt);
    free_llm(ctx.llm, ctx.tokenizer);
    free_sampler(ctx.sampler);

    return 0;
}


//////////////////////////////////////////////////////////////////////////////


static Nano_Context NANO_CTX;

LLM *test_wasm() {
    return NANO_CTX.llm;
}

int init_nano(char *buffer) {
    if(!setlocale(LC_CTYPE, "")) {
        fprintf(stderr, "Can't set the specified locale! Check LANG, LC_CTYPE, LC_ALL.\n");
        return -1;
    }

    float rep_pnty = 1.11;
    float temperature = 1.1;
    float top_p = 0.5;
    int   top_k = 0;
    int   max_seq_len = 512;
    int   random_seed = 0; // (unsigned int)time(NULL);

    char *LORA_PATH  = NULL;

    NANO_CTX.llm = (LLM *)calloc(1, (sizeof(LLM)));
    NANO_CTX.lora = NULL; // (LoRA *)calloc(1, (sizeof(LoRA *)));
    NANO_CTX.tokenizer = (Tokenizer *)calloc(1, (sizeof(Tokenizer)));

    printf("wasm\n");
    load_llm_from_buffer(NANO_CTX.llm, NANO_CTX.tokenizer, buffer);


    // NANO_CTX.sampler = build_sampler(NANO_CTX.llm->config.vocab_size, rep_pnty, temperature, top_p, top_k, random_seed);

    // if(NULL != LORA_PATH) NANO_CTX.lora = load_lora(NANO_CTX.llm, LORA_PATH);

    // generate(NANO_CTX, prompt, max_seq_len, typewriter, report);

    return 0;
}

uint32_t generate_next_token_external(uint32_t *ids, uint32_t pos, int is_prefilling) {
    return generate_next_token(NANO_CTX, ids, pos, is_prefilling);
}

uint32_t *encode_external(wchar_t *text, uint32_t *n_tokens_ptr) {
    return encode(NANO_CTX.tokenizer, text, n_tokens_ptr);
}

wchar_t *decode_external(uint32_t *ids, uint32_t len) {
    return decode(NANO_CTX.tokenizer, ids, len);
}

int close_nano() {
    free_llm(NANO_CTX.llm, NANO_CTX.tokenizer);
    free_sampler(NANO_CTX.sampler);

    char *LORA_PATH  = NULL;
    if(NULL != LORA_PATH) free_lora(NANO_CTX.llm, NANO_CTX.lora);

    return 0;
}

