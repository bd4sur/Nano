#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <locale.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include "tokenizer.h"

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
} ModelConfig;

typedef struct {
    float* token_embedding;    // (vocab_size, n_embd)
    float* rms_norm_attn; // (layer, n_embd) rmsnorm weights

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
} ModelParam;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (n_embd,)
    float *xb; // same, but inside a residual branch (n_embd,)
    float *xb2; // an additional buffer just for convenience (n_embd,)
    float *hb; // buffer for hidden dimension in the ffn (n_hidden,)
    float *hb2; // buffer for hidden dimension in the ffn (n_hidden,)
    float *q; // query (n_embd,)
    float *k; // key (kv_dim,)
    float *v; // value (kv_dim,)
    float *k_cache; // (layer, block_size, kv_dim)
    float *v_cache; // (layer, block_size, kv_dim)
    float *att; // buffer for scores/attention values (n_heads, block_size)
    float *logits; // output logits
} FwdBuffer;

typedef struct {
    ModelConfig config;
    ModelParam weights;
    FwdBuffer state;
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} LLM;


// ===============================================================================
// 词元编解码、分词器（基于Trie树）
// ===============================================================================





// ===============================================================================
// 模型文件解析·内存管理
// ===============================================================================

void malloc_run_state(FwdBuffer* s, ModelConfig* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->n_embd * p->n_kv_head) / p->n_head;
    s->x = calloc(p->n_embd, sizeof(float));
    s->xb = calloc(p->n_embd, sizeof(float));
    s->xb2 = calloc(p->n_embd, sizeof(float));
    s->hb = calloc(p->n_hidden, sizeof(float));
    s->hb2 = calloc(p->n_hidden, sizeof(float));
    s->q = calloc(p->n_embd, sizeof(float));
    s->k_cache = calloc(p->n_layer * p->block_size * kv_dim, sizeof(float));
    s->v_cache = calloc(p->n_layer * p->block_size * kv_dim, sizeof(float));
    s->att = calloc(p->n_head * p->block_size, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k_cache || !s->v_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}


void free_run_state(FwdBuffer* s) {
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
}


void memory_map_weights(ModelParam *w, ModelConfig* p, float* ptr, int shared_weights) {
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


void read_checkpoint(char* checkpoint, ModelConfig* config, ModelParam* weights,
                     int* fd, float** data, ssize_t* file_size, Tokenizer *tk) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

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
    unsigned long long param_num = 0;
    if(fread(&param_num, sizeof(unsigned long long), 1, file) != 1) { exit(EXIT_FAILURE); }

    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    // 解析模型参数

    float* weights_ptr = *data + 2 + (header_byte_length / sizeof(float)); // NOTE "2"指的是param_count（uint64）占了两个uint32的位置
    memory_map_weights(weights, config, weights_ptr, config->is_shared_classifier);

    // 解析词表，同时构建trie树和hashmap

    uint32_t *tokenzier_ptr = (uint32_t *)(weights_ptr + param_num);
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

    while(byte_count < tokenizer_field_bytes - 4) { // 不含vocab_size字段的4个字节
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
}

void build_transformer(LLM *t, Tokenizer *tk, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size, tk);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(LLM* t, Tokenizer *tk) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the FwdBuffer buffers
    free_run_state(&t->state);
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



float* forward(LLM* llm, int token, int pos) {

    // a few convenience variables
    ModelConfig* p = &llm->config;
    ModelParam* w = &llm->weights;
    FwdBuffer* s = &llm->state;
    float *x = s->x;
    int n_embd = p->n_embd;
    int kv_dim = (p->n_embd * p->n_kv_head) / p->n_head;
    int kv_mul = p->n_head / p->n_kv_head; // integer multiplier of the kv sharing in multiquery
    int n_hidden =  p->n_hidden;
    int head_size = n_embd / p->n_head;

    // copy the token embedding into x
    float* content_row = w->token_embedding + token * n_embd;
    memcpy(x, content_row, n_embd*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layer; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_norm_attn + l*n_embd, n_embd);

        // key and value point to the kv cache
        int loff = l * p->block_size * kv_dim; // kv cache layer offset for convenience
        s->k = s->k_cache + loff + pos * kv_dim;
        s->v = s->v_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*n_embd*n_embd, n_embd, n_embd);
        matmul(s->k, s->xb, w->wk + l*n_embd*kv_dim, n_embd, kv_dim);
        matmul(s->v, s->xb, w->wv + l*n_embd*kv_dim, n_embd, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < n_embd; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
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

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_head; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->block_size;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->k_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->v_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*n_embd*n_embd, n_embd, n_embd);

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
    matmul(s->logits, x, w->token_classifier, p->n_embd, p->vocab_size);
    return s->logits;
}


// ===============================================================================
// 采样策略
// ===============================================================================


typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

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

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
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

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
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

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
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

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}


// ===============================================================================
// 文本生成
// ===============================================================================

void generate(LLM *llm, Tokenizer *tokenizer, Sampler *sampler, wchar_t *prompt, int steps) {
    wchar_t *empty_prompt = L"";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    uint32_t num_prompt_tokens = 0;
    uint32_t *prompt_tokens = encode(tokenizer, prompt, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    uint32_t next;        // will store the next token in the sequence
    uint32_t token = prompt_tokens[0]; // kick off with the first token in the prompt
    uint32_t pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the model to get logits for the next token
        float* logits = forward(llm, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        uint32_t output_id[1];
        output_id[0] = next;
        wchar_t *out = decode(tokenizer, output_id, 1);
        printf("%ls", out);

        token = next;

    }
    printf("\n");
    free(prompt_tokens);
}


int main(int argc, char **argv) {
    if(!setlocale(LC_CTYPE, "")) {
        fprintf(stderr, "Can't set the specified locale! Check LANG, LC_CTYPE, LC_ALL.\n");
        return -1;
    }

    LLM llm;
    Tokenizer tokenizer;
    Sampler sampler;
    build_transformer(&llm, &tokenizer, "/home/bd4sur/ai/Nano/test.bin");
    build_sampler(&sampler, llm.config.vocab_size, 1.1, 0.9, (unsigned int)time(NULL));

    wchar_t *prompt = L"Nano是BD4SUR开发的大模型，是一只电子鹦鹉";
    uint32_t token_count = 0;

    generate(&llm, &tokenizer, &sampler, prompt, 100);

    free_sampler(&sampler);
    free_transformer(&llm, &tokenizer);
    return 0;
}
