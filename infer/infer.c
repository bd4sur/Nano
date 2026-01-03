//
// Nano Language Model - Inference Engine written in C
//
//   BD4SUR 2024-10 2024-05
//
//   Forked from: https://github.com/karpathy/llama2.c
//

#include "infer.h"

// ===============================================================================
// 模型文件解析·内存管理
// ===============================================================================

void malloc_fwd_buffer(LLM *llm, uint32_t max_seq_len) {
    FwdBuffer *s = &(llm->state);
    LLM_Config *llm_cfg = &(llm->config);

    uint32_t q_dim = llm_cfg->n_embd;
    uint32_t kv_dim = (llm_cfg->n_embd * llm_cfg->n_kv_head) / llm_cfg->n_head;

    if (llm->arch == LLM_ARCH_NANO || llm->arch == LLM_ARCH_QWEN2) {
        q_dim = llm_cfg->n_embd;
        kv_dim = (llm_cfg->n_embd * llm_cfg->n_kv_head) / llm_cfg->n_head;
    }
    else if (llm->arch == LLM_ARCH_QWEN3) {
        q_dim = llm_cfg->head_dim * llm_cfg->n_head;
        kv_dim = llm_cfg->head_dim * llm_cfg->n_kv_head;
    }

    uint32_t xbuf_length = llm_cfg->n_embd + llm_cfg->n_embd + q_dim + llm_cfg->n_embd + llm_cfg->n_hidden + llm_cfg->n_hidden
                         + q_dim + llm_cfg->n_head * max_seq_len + llm_cfg->vocab_size;        
    s->xbuf = (float *)calloc_dev(xbuf_length, sizeof(float));
    float *xbuf = s->xbuf;

    s->x       = xbuf;  xbuf += llm_cfg->n_embd;
    s->xb      = xbuf;  xbuf += llm_cfg->n_embd;
    s->xba     = xbuf;  xbuf += q_dim;
    s->xb2     = xbuf;  xbuf += llm_cfg->n_embd;
    s->hb      = xbuf;  xbuf += llm_cfg->n_hidden;
    s->hb2     = xbuf;  xbuf += llm_cfg->n_hidden;
    s->q       = xbuf;  xbuf += q_dim;
    s->att     = xbuf;  xbuf += llm_cfg->n_head * max_seq_len;
    s->logits  = xbuf;  xbuf += llm_cfg->vocab_size;

    uint32_t kvcache_length = llm_cfg->n_layer * max_seq_len * kv_dim * 2;        
    s->kvcache = (float *)calloc_dev(kvcache_length, sizeof(float));
    float *kvcache = s->kvcache;

    s->k_cache = kvcache;  kvcache += llm_cfg->n_layer * max_seq_len * kv_dim;
    s->v_cache = kvcache;  kvcache += llm_cfg->n_layer * max_seq_len * kv_dim;

    if (llm->quant_type == QUANT_TYPE_Q80) {
        uint32_t gs = llm->group_size;

        uint32_t qvbuf_length = llm_cfg->n_embd + q_dim + llm_cfg->n_hidden;
        uint32_t qsbuf_length = llm_cfg->n_embd / gs + q_dim / gs + llm_cfg->n_hidden / gs;
        s->qvbuf = (QTYPE *)calloc_dev(qvbuf_length, sizeof(QTYPE));
        s->qsbuf = (float *)calloc_dev(qsbuf_length, sizeof(float));
        QTYPE *qvbuf = s->qvbuf;
        float *qsbuf = s->qsbuf;

        Q80_Tensor xq_tmp   = (Q80_Tensor) { .q = qvbuf, .s = qsbuf };  qvbuf += llm_cfg->n_embd; qsbuf += llm_cfg->n_embd / gs;
        Q80_Tensor xbaq_tmp = (Q80_Tensor) { .q = qvbuf, .s = qsbuf };  qvbuf += q_dim; qsbuf += q_dim / gs;
        Q80_Tensor hq_tmp   = (Q80_Tensor) { .q = qvbuf, .s = qsbuf };  qvbuf += llm_cfg->n_hidden; qsbuf += llm_cfg->n_hidden / gs;
        
        s->xq               = (Typed_Tensor) { .tensor_q80 = xq_tmp };
        s->xbaq             = (Typed_Tensor) { .tensor_q80 = xbaq_tmp };
        s->hq               = (Typed_Tensor) { .tensor_q80 = hq_tmp };
    }


    // ensure all mallocs went fine
    if (!s->xbuf || !s->kvcache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}


void free_fwd_buffer(LLM* llm) {
    FwdBuffer* s = &llm->state;

    free(s->xbuf);
    free(s->kvcache);

    if (llm->quant_type == QUANT_TYPE_Q80) {
        free(s->qvbuf);
        free(s->qsbuf);
    }
}

void memory_map_params(LLM *llm, void* ptr) {
    LLM_Param *w = &(llm->params);
    LLM_Config *cfg = &(llm->config);

    int head_size = cfg->n_embd / cfg->n_head;

    if (llm->arch == LLM_ARCH_NANO || llm->arch == LLM_ARCH_QWEN2) {
        head_size = cfg->n_embd / cfg->n_head;
    }
    else if (llm->arch == LLM_ARCH_QWEN3) {
        head_size = cfg->head_dim;
    }

    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    uint64_t n_layer = cfg->n_layer;

    float* fptr = (float*) ptr;

    w->rms_norm_attn  = fptr;   fptr += n_layer * cfg->n_embd;
    w->rms_norm_ffn   = fptr;   fptr += n_layer * cfg->n_embd;
    w->rms_norm_final = fptr;   fptr += cfg->n_embd;

    if (llm->quant_type == QUANT_TYPE_Q80) {
        ptr = (void*)fptr;

        w->q_tokens = parse_quantized_tensors(&ptr, 1, cfg->vocab_size * cfg->n_embd, llm->group_size);
        w->token_embedding = (float *)calloc_dev(cfg->vocab_size * cfg->n_embd, sizeof(float));
        dequantize(&w->q_tokens->tensor_q80, w->token_embedding, cfg->vocab_size * cfg->n_embd, llm->group_size);

        w->wq = parse_quantized_tensors(&ptr, n_layer, cfg->n_embd * (cfg->n_head * head_size), llm->group_size);
        w->wk = parse_quantized_tensors(&ptr, n_layer, cfg->n_embd * (cfg->n_kv_head * head_size), llm->group_size);
        w->wv = parse_quantized_tensors(&ptr, n_layer, cfg->n_embd * (cfg->n_kv_head * head_size), llm->group_size);
        w->wo = parse_quantized_tensors(&ptr, n_layer, (cfg->n_head * head_size) * cfg->n_embd, llm->group_size);

        w->w1 = parse_quantized_tensors(&ptr, n_layer, cfg->n_embd * cfg->n_hidden, llm->group_size);
        w->w2 = parse_quantized_tensors(&ptr, n_layer, cfg->n_hidden * cfg->n_embd, llm->group_size);
        w->w3 = parse_quantized_tensors(&ptr, n_layer, cfg->n_embd * cfg->n_hidden, llm->group_size);

        fptr = (float*)ptr;
    }
    else if (llm->quant_type == QUANT_TYPE_F32) {
        w->token_embedding = fptr; fptr += cfg->vocab_size * cfg->n_embd;

        w->wq = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->wq->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * (cfg->n_head * head_size);
        w->wk = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->wk->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * (cfg->n_kv_head * head_size);
        w->wv = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->wv->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * (cfg->n_kv_head * head_size);
        w->wo = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->wo->tensor_f32 = fptr;  fptr += n_layer * (cfg->n_head * head_size) * cfg->n_embd;

        w->w1 = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->w1->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * cfg->n_hidden;
        w->w2 = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->w2->tensor_f32 = fptr;  fptr += n_layer * cfg->n_hidden * cfg->n_embd;
        w->w3 = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->w3->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * cfg->n_hidden;
    }

    if (llm->arch == LLM_ARCH_QWEN2) {
        w->bq = fptr;         fptr += n_layer * (cfg->n_head * head_size);
        w->bk = fptr;         fptr += n_layer * (cfg->n_kv_head * head_size);
        w->bv = fptr;         fptr += n_layer * (cfg->n_kv_head * head_size);
    }
    else if (llm->arch == LLM_ARCH_QWEN3) {
        w->q_norm = fptr;     fptr += n_layer * head_size;
        w->k_norm = fptr;     fptr += n_layer * head_size;
    }

    if (llm->arch == LLM_ARCH_NANO || llm->arch == LLM_ARCH_QWEN2) {
        w->freq_cis_real = fptr;  fptr += cfg->block_size * head_size / 2;
        w->freq_cis_imag = fptr;  fptr += cfg->block_size * head_size / 2;
    }
    else if (llm->arch == LLM_ARCH_QWEN3) {
        w->freq_cis_real = calloc_dev(cfg->block_size * head_size / 2, sizeof(float));
        w->freq_cis_imag = calloc_dev(cfg->block_size * head_size / 2, sizeof(float));

        for (uint32_t pos = 0; pos < cfg->block_size; pos++) {
            for (uint32_t i = 0; i < head_size / 2; i++) {
                float freq = 1.0f / powf(1000000.0f, (float)(i * 2) / (float)head_size); // QWEN3_ROPE_THETA = 1000000.0
                float fcr = cosf(pos * freq);
                float fci = sinf(pos * freq);
                w->freq_cis_real[pos * head_size / 2 + i] = fcr;
                w->freq_cis_imag[pos * head_size / 2 + i] = fci;
            }
        }
        fptr += cfg->block_size * head_size / 2;
        fptr += cfg->block_size * head_size / 2;
    }

    if (llm->quant_type == QUANT_TYPE_Q80) {
        ptr = (void*)fptr;
        w->token_classifier = cfg->is_shared_classifier ? w->q_tokens : parse_quantized_tensors(&ptr, 1, cfg->n_embd * cfg->vocab_size, llm->group_size);
    }
    else if (llm->quant_type == QUANT_TYPE_F32) {
        w->token_classifier = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));
        w->token_classifier->tensor_f32 = cfg->is_shared_classifier ? w->token_embedding : ptr;
    }
}


void parse_model_file(uint8_t* buffer, LLM *llm, Tokenizer *tk) {

    LLM_Config *config = &(llm->config);

    // 读取文件头
    const uint32_t header_byte_length = 256;

    uint32_t *header = (uint32_t *)buffer;

    uint32_t offset = 0;

    uint32_t magic_number_0 = header[offset]; offset++; (void)magic_number_0;
    uint32_t magic_number_1 = header[offset]; offset++; (void)magic_number_1;

    uint32_t major_version  = header[offset]; offset++; (void)major_version;
    uint32_t minor_version  = header[offset]; offset++; (void)minor_version;

    uint32_t model_type     = header[offset]; offset++;
    uint32_t config_length  = header[offset]; offset++; (void)config_length;

    config->block_size      = header[offset]; offset++;
    config->vocab_size      = header[offset]; offset++;
    config->n_layer         = header[offset]; offset++;
    config->n_embd          = header[offset]; offset++;
    config->n_head          = header[offset]; offset++;
    config->n_kv_head       = header[offset]; offset++;
    config->n_hidden        = header[offset]; offset++;
    config->is_shared_classifier = header[offset]; offset++;
    config->head_dim        = header[offset]; offset++;

    uint32_t quant_type     = header[offset]; offset++;
    uint32_t group_size     = header[offset]; offset++;

    llm->arch = model_type;

    llm->quant_type = (quant_type == 0) ? QUANT_TYPE_F32 : QUANT_TYPE_Q80; // TODO 量化类型enum待统一
    llm->group_size = group_size;

    // 解析词表，同时构建trie树和hashmap

    uint32_t *tokenzier_ptr = (uint32_t *)(buffer) + (header_byte_length / sizeof(uint32_t));
    uint32_t tokenizer_field_bytes = *tokenzier_ptr;

    if (llm->arch == LLM_ARCH_NANO) {
        uint32_t *vocab_ptr = tokenzier_ptr + 1;

        uint32_t byte_count = 0;
        uint32_t char_count = 0;

        tk->vocab_size = *vocab_ptr; vocab_ptr++;

        tk->token_list        = (wchar_t **)calloc_dev(tk->vocab_size, sizeof(wchar_t *));
        tk->unicode_charset   = (wchar_t  *)calloc_dev(tk->vocab_size, sizeof(wchar_t));
        tk->unicode_to_id_map = new_map(tk->vocab_size);
        tk->token_to_id_map   = new_map(tk->vocab_size);
        tk->vocab_trie        = new_trie(tk->vocab_size, 0);

        while(byte_count < tokenizer_field_bytes - 8) { // 不含tokenizer_field_bytes和vocab_size字段的8个字节
            uint32_t token_header = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);
            uint32_t token_id     = *vocab_ptr; vocab_ptr++; byte_count += sizeof(uint32_t);

            // NOTE Little endian 小端序！如果按照uint32解析，顺序是 MSB(reserved_1 reserved_0 is_special token_length)LSB
            uint32_t reserved_1   = (token_header & 0xff000000) >> 24; (void)reserved_1;
            uint32_t reserved_0   = (token_header & 0x00ff0000) >> 16; (void)reserved_0;
            uint32_t is_special   = (token_header & 0x0000ff00) >> 8;  (void)is_special;
            uint32_t token_length = (token_header & 0x000000ff);

            wchar_t *token = (wchar_t *)calloc_dev(token_length+1, sizeof(wchar_t));
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
    else if (llm->arch == LLM_ARCH_QWEN2 || llm->arch == LLM_ARCH_QWEN3) {
        build_bpe_tokenizer(tk, (uint8_t*)tokenzier_ptr, 151669);
    }

    // 解析模型参数

    void *param_ptr = (float*)((uint8_t*)tokenzier_ptr + tokenizer_field_bytes);
    memory_map_params(llm, param_ptr);
}


void load_llm_from_buffer(LLM *llm, Tokenizer *tk, uint8_t *buffer, uint32_t max_seq_len) {
    parse_model_file(buffer, llm, tk);
    malloc_fwd_buffer(llm, max_seq_len);
}


void load_llm(LLM *llm, Tokenizer *tk, char *model_path, uint32_t max_seq_len) {
    // 获得文件长度
    FILE *_file = fopen(model_path, "rb");
    if (!_file) { fprintf(stderr, "Couldn't open file %s\n", model_path); exit(EXIT_FAILURE); }
    fseek(_file, 0, SEEK_END);
    llm->file_size = ftell(_file);
    fclose(_file);

#ifdef NANO_USE_MMAP

    // 将文件mmap到虚拟内存
    int fd = open(model_path, O_RDONLY);
    if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    uint8_t *buffer = mmap(NULL, llm->file_size, PROT_READ, (MAP_PRIVATE | MAP_POPULATE), fd, 0);
    if (buffer == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    llm->fd = fd;
    llm->buffer = buffer;

    load_llm_from_buffer(llm, tk, buffer, max_seq_len);

#else

    FILE *file = fopen(model_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open model file %s\n", model_path); exit(EXIT_FAILURE); }
    uint8_t *buffer = (uint8_t *)calloc(llm->file_size + 1, sizeof(uint8_t));
    if (!buffer)  { fprintf(stderr, "malloc failed.\n"); exit(EXIT_FAILURE); }
    if(fread(buffer, sizeof(uint8_t), llm->file_size, file) != llm->file_size) { exit(EXIT_FAILURE); }
    llm->buffer = buffer;

    load_llm_from_buffer(llm, tk, buffer, max_seq_len);

    fclose(file);
#endif
}


void free_llm(LLM* llm, Tokenizer *tk) {
#ifdef NANO_USE_MMAP
    // close the memory mapping
    if (llm->buffer != MAP_FAILED) { munmap(llm->buffer, llm->file_size); }
    if (llm->fd != -1) { close(llm->fd); }
#else
    free(llm->buffer);
#endif

    if (llm->quant_type == QUANT_TYPE_Q80) {
        free(llm->params.token_embedding);
    }
    free(llm->params.q_tokens);
    if (llm->config.is_shared_classifier == 0) {
        free(llm->params.token_classifier);
    }
    free(llm->params.wq);
    free(llm->params.wk);
    free(llm->params.wv);
    free(llm->params.wo);
    free(llm->params.w1);
    free(llm->params.w2);
    free(llm->params.w3);

    if (llm->arch == LLM_ARCH_NANO) {
        free_tokenizer(tk);
    }
    else if (llm->arch == LLM_ARCH_QWEN2 || llm->arch == LLM_ARCH_QWEN3) {
        free_bpe_tokenizer(tk);
        if (llm->arch == LLM_ARCH_QWEN3) {
            free(llm->params.freq_cis_real);
            free(llm->params.freq_cis_imag);
        }
    }

    free_fwd_buffer(llm);

    free(llm);
}



// ===============================================================================
// LoRA插件文件解析·LoRA相关内存管理
// ===============================================================================


void malloc_fwd_buffer_with_lora(LLM *llm, LoRA_Config *lora_cfg) {
    FwdBuffer *s = &(llm->state);
    LLM_Config *llm_cfg = &(llm->config);
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


void parse_lora_file(uint8_t* buffer, LoRA *lora, LLM *llm) {
    LoRA_Config *lora_cfg    = &(lora->config);
    LoRA_Param  *lora_params = &(lora->params);
    LLM_Config  *llm_cfg     = &(llm->config);

    // 读取文件头
    const uint32_t header_byte_length = 256;
    // const uint32_t header_uint_length = header_byte_length / sizeof(uint32_t);
    uint32_t *header = (uint32_t *)buffer;

    uint32_t offset = 0;

    uint32_t magic_number_0 = header[offset]; offset++; (void)magic_number_0;
    uint32_t magic_number_1 = header[offset]; offset++; (void)magic_number_1;

    uint32_t major_version  = header[offset]; offset++; (void)major_version;
    uint32_t minor_version  = header[offset]; offset++; (void)minor_version;

    uint32_t model_type     = header[offset]; offset++; (void)model_type;
    uint32_t config_length  = header[offset]; offset++; (void)config_length;

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
        fprintf(stderr, "Error: LoRA module does not fit the base model.\n");
        exit(EXIT_FAILURE);
    }

    // 解析LoRA模块参数

    float *lora_param_ptr = (float*)(buffer + header_byte_length);

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

    lora_params->wq_lora_a = lora_param_ptr; lora_param_ptr += wq_lora_a_len;
    lora_params->wq_lora_b = lora_param_ptr; lora_param_ptr += wq_lora_b_len;
    lora_params->wk_lora_a = lora_param_ptr; lora_param_ptr += wk_lora_a_len;
    lora_params->wk_lora_b = lora_param_ptr; lora_param_ptr += wk_lora_b_len;
    lora_params->wv_lora_a = lora_param_ptr; lora_param_ptr += wv_lora_a_len;
    lora_params->wv_lora_b = lora_param_ptr; lora_param_ptr += wv_lora_b_len;
    lora_params->wo_lora_a = lora_param_ptr; lora_param_ptr += wo_lora_a_len;
    lora_params->wo_lora_b = lora_param_ptr; lora_param_ptr += wo_lora_b_len;
}

LoRA *load_lora(LLM *llm, char *lora_path) {
    FILE *file = fopen(lora_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open LoRA module file %s\n", lora_path); exit(EXIT_FAILURE); }

    // 获取文件大小
    fseek(file, 0, SEEK_END);
    uint64_t file_size = ftell(file);
    rewind(file);

    uint8_t *lora_buffer = (uint8_t *)calloc(file_size + 1, sizeof(uint8_t));
    if (!lora_buffer)  { fprintf(stderr, "malloc failed.\n"); exit(EXIT_FAILURE); }

    if(fread(lora_buffer, sizeof(uint8_t), file_size, file) != file_size) { exit(EXIT_FAILURE); }
    LoRA *p_lora = (LoRA *)calloc(1, sizeof(LoRA));

    p_lora->data = (float*)lora_buffer;

    parse_lora_file(lora_buffer, p_lora, llm);

    malloc_fwd_buffer_with_lora(llm, &p_lora->config);

    return p_lora;
}

LoRA *load_lora_from_buffer(LLM *llm, uint8_t *buffer) {
    LoRA *p_lora = (LoRA *)calloc(1, sizeof(LoRA));
    p_lora->data = (float*)buffer;
    parse_lora_file(buffer, p_lora, llm);
    malloc_fwd_buffer_with_lora(llm, &p_lora->config);
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
// 推理引擎单例（现在暂且叫context）管理
// ===============================================================================

Nano_Context *llm_context_init_from_buffer(uint8_t *buffer, uint32_t max_seq_len, float repetition_penalty, float temperature, float top_p, uint32_t top_k, uint64_t random_seed) {
    Nano_Context *ctx = (Nano_Context*)calloc(1, sizeof(Nano_Context));
    ctx->max_seq_len = max_seq_len;
    ctx->random_seed = random_seed;
    ctx->llm = (LLM *)calloc(1, (sizeof(LLM)));
    ctx->tokenizer = (Tokenizer *)calloc(1, (sizeof(Tokenizer)));
    ctx->lora = NULL;
    load_llm_from_buffer(ctx->llm, ctx->tokenizer, buffer, ctx->max_seq_len);
    ctx->sampler = build_sampler(ctx->llm->config.vocab_size, repetition_penalty, temperature, top_p, top_k, ctx->random_seed);
    return ctx;
}

Nano_Context *llm_context_init(char *model_path, char *lora_path, uint32_t max_seq_len, float repetition_penalty, float temperature, float top_p, uint32_t top_k, uint64_t random_seed) {
    Nano_Context *ctx = (Nano_Context*)calloc(1, sizeof(Nano_Context));
    ctx->max_seq_len = max_seq_len;
    ctx->random_seed = random_seed;
    ctx->llm = (LLM *)calloc(1, (sizeof(LLM)));
    ctx->tokenizer = (Tokenizer *)calloc(1, (sizeof(Tokenizer)));
    load_llm(ctx->llm, ctx->tokenizer, model_path, max_seq_len);
    ctx->sampler = build_sampler(ctx->llm->config.vocab_size, repetition_penalty, temperature, top_p, top_k, ctx->random_seed);
    ctx->lora = (lora_path) ? load_lora(ctx->llm, lora_path) : NULL;
    return ctx;
}

void llm_context_free(Nano_Context *ctx) {
    free_llm(ctx->llm, ctx->tokenizer);
    free_sampler(ctx->sampler);
    free(ctx);
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

// W (d,n) @ x (n,) -> xout (d,)
void matmul(float* xout, float* x, float* w, int n, int d) {
#ifdef MATMUL_PTHREAD
    matmul_pthread(xout, x, w, n, d);
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
#endif
}

// W (d,n) @ x (n,) -> xout (d,)
void matmul_quant(float* xout, Typed_Tensor *x, Typed_Tensor *w, int n, int d, uint32_t group_size) {
#ifdef MATMUL_PTHREAD
    matmul_quant_pthread(xout, x->tensor_q80, w->tensor_q80, n, d, group_size);
#else
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of group_size
        int j;
        for (j = 0; j <= n - group_size; j += group_size) {
            for (int k = 0; k < group_size; k++) {
                ival += ((int32_t) x->tensor_q80.q[j + k]) * ((int32_t) w->tensor_q80.q[in + j + k]);
            }
            val += ((float) ival) * w->tensor_q80.s[(in + j) / group_size] * x->tensor_q80.s[j / group_size];
            ival = 0;
        }

        xout[i] = val;
    }
#endif
}

void rope(float *head, uint32_t head_dim, uint32_t pos, float *freq_cis_real_row, float *freq_cis_imag_row) {
    for (uint32_t i = 0; i < head_dim; i += 2) {
        float v0 = head[i];
        float v1 = head[i + 1];
        float fcr = freq_cis_real_row[i / 2];
        float fci = freq_cis_imag_row[i / 2];
        head[  i  ] = v0 * fcr - v1 * fci;
        head[i + 1] = v0 * fci + v1 * fcr;
    }
}

void rope_qwen3(float *head, uint32_t head_dim, uint32_t pos, float *freq_cis_real_row, float *freq_cis_imag_row) {
    for (uint32_t i = 0; i < head_dim / 2; i++) {
        // float freq = 1.0f / powf(1000000.0f, (float)(i * 2) / (float)head_dim); // QWEN3_ROPE_THETA = 1000000.0
        // float fcr = cosf(pos * freq);
        // float fci = sinf(pos * freq);

        float fcr = freq_cis_real_row[i];
        float fci = freq_cis_imag_row[i];

        float v0 = head[i];
        float v1 = head[i + head_dim / 2];
        head[       i        ] = v0 * fcr - v1 * fci;
        head[i + head_dim / 2] = v1 * fcr + v0 * fci;
    }
}


// 前向传播一个Transformer块
// 函数的输入输出都是x，输入的x是上一层传入的值，输出即为本层输出。复用同一块内存（原地更新），波动传递中间激活值。
// NOTE is_causal参数的说明详见llm_forward函数
// NOTE 本函数每次执行都会重新计算一次模型的结构参数，但是开销很小。比起逻辑上的清晰和后续流水线并行改造，这点代价是值得的。
void transformer_block_forward(float *x, uint32_t layer, uint32_t pos, uint32_t max_seq_len, uint32_t is_causal, LLM* llm, LoRA *lora) {

    LLM_Config *cfg = &llm->config;
    LLM_Param *w = &llm->params;
    FwdBuffer *s = &llm->state;

    // 模型结构参数

    uint32_t n_embd = cfg->n_embd;
    uint32_t kv_mul = cfg->n_head / cfg->n_kv_head; // integer multiplier of the kv sharing in multiquery
    uint32_t n_hidden =  cfg->n_hidden;

    uint32_t head_dim = 0, q_dim = 0, kv_dim = 0;

    if (llm->arch == LLM_ARCH_NANO || llm->arch == LLM_ARCH_QWEN2) {
        head_dim = n_embd / cfg->n_head;
        q_dim = cfg->n_embd;
        kv_dim = (cfg->n_embd * cfg->n_kv_head) / cfg->n_head;
    }
    else if (llm->arch == LLM_ARCH_QWEN3) {
        head_dim = cfg->head_dim;
        q_dim = head_dim * cfg->n_head;
        kv_dim = head_dim * cfg->n_kv_head;
    }

    // 设定LoRA相关参数

    int32_t use_lora = (NULL == lora) ? 0 : 1;
    LoRA_Param *a = NULL;
    uint32_t lora_rank = 0;
    uint32_t lora_alpha = 0;;
    if(use_lora == 1) {
        a = &(lora->params);
        lora_rank = lora->config.lora_rank;
        lora_alpha = lora->config.lora_alpha;
    }

    // 第pos位置上的RoPE系数
    float *freq_cis_real_row = w->freq_cis_real + pos * head_dim / 2;
    float *freq_cis_imag_row = w->freq_cis_imag + pos * head_dim / 2;
    
    // attention rmsnorm
    rmsnorm(s->xb, x, w->rms_norm_attn + layer * n_embd, n_embd);

    // key and value point to the kv cache
    // NOTE KV缓存的长度由max_seq_len参数指定，不一定等于模型本身的block_size。这样做的目的是为了按需控制KV缓存的大小。
    int layer_offset = layer * max_seq_len * kv_dim;
    s->k = s->k_cache + layer_offset + pos * kv_dim;
    s->v = s->v_cache + layer_offset + pos * kv_dim;

    // qkv matmuls for this position
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->q, s->xb, w->wq->tensor_f32 + layer*q_dim*n_embd,  n_embd, q_dim);
        matmul(s->k, s->xb, w->wk->tensor_f32 + layer*kv_dim*n_embd, n_embd, kv_dim);
        matmul(s->v, s->xb, w->wv->tensor_f32 + layer*kv_dim*n_embd, n_embd, kv_dim);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, s->xb, n_embd, llm->group_size);
        matmul_quant(s->q, &s->xq, w->wq + layer, n_embd, q_dim  , llm->group_size);
        matmul_quant(s->k, &s->xq, w->wk + layer, n_embd, kv_dim , llm->group_size);
        matmul_quant(s->v, &s->xq, w->wv + layer, n_embd, kv_dim , llm->group_size);
    }

    if (llm->arch == LLM_ARCH_QWEN2) {
        // TODO Qwen2 bq bk bv
    }

    if(llm->arch == LLM_ARCH_NANO && use_lora == 1) {
        matmul(s->q0, s->xb, a->wq_lora_a + layer * lora_rank * n_embd, n_embd, lora_rank);
        matmul(s->k0, s->xb, a->wk_lora_a + layer * lora_rank * n_embd, n_embd, lora_rank);
        matmul(s->v0, s->xb, a->wv_lora_a + layer * lora_rank * n_embd, n_embd, lora_rank);

        matmul(s->q1, s->q0, a->wq_lora_b + layer * n_embd * lora_rank, lora_rank, n_embd);
        matmul(s->k1, s->k0, a->wk_lora_b + layer * kv_dim * lora_rank, lora_rank, kv_dim);
        matmul(s->v1, s->v0, a->wv_lora_b + layer * kv_dim * lora_rank, lora_rank, kv_dim);

        scale(s->q1, ((float)lora_alpha / (float)lora_rank), n_embd);
        scale(s->k1, ((float)lora_alpha / (float)lora_rank), kv_dim);
        scale(s->v1, ((float)lora_alpha / (float)lora_rank), kv_dim);

        accum(s->q, s->q1, n_embd);
        accum(s->k, s->k1, kv_dim);
        accum(s->v, s->v1, kv_dim);
    }

    // RoPE位置编码
    if (llm->arch == LLM_ARCH_NANO || llm->arch == LLM_ARCH_QWEN2) {
        for (uint32_t h = 0; h < cfg->n_head; h++) {
            float *q = s->q + h * head_dim;
            rope(q, head_dim, pos, freq_cis_real_row, freq_cis_imag_row);
        }
        for (uint32_t m = 0; m < cfg->n_kv_head; m++) {
            float *k = s->k + m * head_dim;
            rope(k, head_dim, pos, freq_cis_real_row, freq_cis_imag_row);
        }
    }
    else if (llm->arch == LLM_ARCH_QWEN3) {
        for (uint32_t h = 0; h < cfg->n_head; h++) {
            float *q = s->q + h * head_dim;
            rmsnorm(q, q, w->q_norm + layer * head_dim, head_dim);
            rope_qwen3(q, head_dim, pos, freq_cis_real_row, freq_cis_imag_row);
        }
        for (uint32_t h = 0; h < cfg->n_kv_head; h++) {
            float *k = s->k + h * head_dim;
            rmsnorm(k, k, w->k_norm + layer * head_dim, head_dim);
            rope_qwen3(k, head_dim, pos, freq_cis_real_row, freq_cis_imag_row);
        }
    }

    // multihead attention. iterate over all heads
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < cfg->n_head; h++) {
        // get the query vector for this head
        float* q = s->q + h * head_dim;
        // attention scores for this head
        float* att = s->att + h * max_seq_len; // NOTE max_seq_len不一定等于模型本身的block_size。这样做的目的是为了节约内存。
        // iterate over all timesteps, including the current one
        uint32_t attn_range = (is_causal) ? (pos + 1) : max_seq_len; // NOTE 用于兼容因果自注意力和全局自注意力
        for (int t = 0; t < attn_range; t++) {
            // get the key vector for this head and at this timestep
            float* k = s->k_cache + layer_offset + t * kv_dim + (h / kv_mul) * head_dim;
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
        softmax(att, attn_range);

        // weighted sum of the values, store back into xba
        float* xba = s->xba + h * head_dim;
        memset(xba, 0, head_dim * sizeof(float));
        for (int t = 0; t < attn_range; t++) {
            // get the value vector for this head and at this timestep
            float* v = s->v_cache + layer_offset + t * kv_dim + (h / kv_mul) * head_dim;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xba
            for (int i = 0; i < head_dim; i++) {
                xba[i] += a * v[i];
            }
        }
    }

    // final matmul to get the output of the attention
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->xb2, s->xba, w->wo->tensor_f32 + layer*n_embd*q_dim, q_dim, n_embd);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xbaq.tensor_q80, s->xba, q_dim, llm->group_size);
        matmul_quant(s->xb2, &s->xbaq, w->wo + layer, q_dim, n_embd, llm->group_size);
    }

    // 计算output的低秩分解分支，并将其累加到原来的输出上
    if(use_lora == 1) {
        matmul(s->o0, s->xba, a->wo_lora_a + layer * lora_rank * q_dim,  q_dim,    lora_rank);
        matmul(s->o1, s->o0,  a->wo_lora_b + layer * n_embd * lora_rank, lora_rank, n_embd);
        scale(s->o1, ((float)lora_alpha / (float)lora_rank), n_embd);
        accum(s->xb2, s->o1, n_embd);
    }

    // residual connection back into x
    for (int i = 0; i < n_embd; i++) {
        x[i] += s->xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_norm_ffn + layer*n_embd, n_embd);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->hb, s->xb, w->w1->tensor_f32 + layer*n_hidden*n_embd, n_embd, n_hidden);
        matmul(s->hb2, s->xb, w->w3->tensor_f32 + layer*n_hidden*n_embd, n_embd, n_hidden);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, s->xb, n_embd, llm->group_size);
        matmul_quant(s->hb, &s->xq, w->w1 + layer, n_embd, n_hidden , llm->group_size);
        matmul_quant(s->hb2, &s->xq, w->w3 + layer, n_embd, n_hidden, llm->group_size);
    }

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
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->xb, s->hb, w->w2->tensor_f32 + layer*n_embd*n_hidden, n_hidden, n_embd);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->hq.tensor_q80, s->hb, n_hidden, llm->group_size);
        matmul_quant(s->xb, &s->hq, w->w2 + layer, n_hidden, n_embd, llm->group_size);
    }

    // residual connection
    for (int i = 0; i < n_embd; i++) {
        x[i] += s->xb[i];
    }
}

// NOTE 该函数实现了在1个token+过往KVCache上的完整的因果自注意力前向推理。对于一般的自回归因果语言模型推理，is_causal=1。
//      参数is_causal=0启用全局自注意力，仅用于（且必须搭配用于）seq2seq函数。因该函数并没有实现完整的seq2seq全局自注意力前向推理。
//      seq2seq函数主要是整活用途，有大量冗余计算，效率极低。
float* llm_forward(uint32_t token, uint32_t pos, uint32_t max_seq_len, uint32_t is_causal, LLM* llm, LoRA *lora) {

    LLM_Config *cfg = &llm->config;
    LLM_Param *w = &llm->params;
    FwdBuffer *s = &llm->state;

    // 模型结构参数
    uint32_t n_embd = cfg->n_embd;

    // Transformer层间传递的中间激活值
    float *x = s->x;

    // 模型入口：将输入token的one-hot转为嵌入向量，等价于直接查token_embedding表
    float* content_row = w->token_embedding + token * n_embd;
    memcpy(x, content_row, n_embd*sizeof(*x));

    // 前向传播，遍历各层的Transformer块，以x为层间传递的中间值（原地更新）
    for(uint64_t l = 0; l < cfg->n_layer; l++) {
        transformer_block_forward(x, l, pos, max_seq_len, is_causal, llm, lora);
    }

    // 最后一层RMSNorm
    rmsnorm(x, x, w->rms_norm_final, n_embd);

    // 模型出口：将最终的嵌入向量解码成logits
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->logits, x, w->token_classifier->tensor_f32, cfg->n_embd, cfg->vocab_size);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, x, cfg->n_embd, llm->group_size);
        matmul_quant(s->logits, &s->xq, w->token_classifier, cfg->n_embd, cfg->vocab_size, llm->group_size);
    }

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

Sampler *build_sampler(int vocab_size, float repetition_penalty, float temperature, float top_p, uint32_t top_k, uint64_t rng_seed) {
    Sampler *sampler = (Sampler *)calloc(1, sizeof(Sampler));
    sampler->vocab_size = vocab_size;
    sampler->repetition_penalty = repetition_penalty;
    sampler->temperature = temperature;
    sampler->top_p = top_p;
    sampler->top_k = top_k;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *)calloc(sampler->vocab_size, sizeof(ProbIndex));
    return sampler;
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}



// ===============================================================================
// 自回归文本生成
// ===============================================================================

uint32_t generate_next_token(Nano_Context *ctx, uint32_t *output_ids, uint32_t pos, int is_prefilling) {

    LLM *llm = ctx->llm;
    LoRA *lora = ctx->lora;
    Sampler *sampler = ctx->sampler;

    uint32_t next_token = output_ids[pos];

    float* logits = llm_forward(next_token, pos, ctx->max_seq_len, 1, llm, lora);

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


Nano_Session *llm_session_init(Nano_Context *ctx, wchar_t *prompt, unsigned int max_seq_len) {
    Nano_Session *session = (Nano_Session *)calloc_dev(1, sizeof(Nano_Session));
    Tokenizer *tokenizer = ctx->tokenizer;

    session->prompt = (wchar_t *)calloc_dev(max_seq_len + 1, sizeof(wchar_t));
    if (prompt == NULL) {
        wcscpy(session->prompt, L"");
    }
    else {
        wcscpy(session->prompt, prompt);
    }

    session->t_0 = 0;
    session->t_1 = 0;
    session->tps = 0.0f;

    session->max_seq_len = max_seq_len;

    session->output_ids = (uint32_t *)calloc_dev(max_seq_len + 1, sizeof(uint32_t));
    session->output_count = 0;

    session->num_prompt_tokens = 0;
    uint32_t *prompt_tokens;
    if (ctx->llm->arch == LLM_ARCH_NANO) {
        prompt_tokens = encode_nano(tokenizer, session->prompt, &(session->num_prompt_tokens));
    }
    else if (ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3) {
        prompt_tokens = apply_qwen_chat_template(tokenizer, session->prompt, &(session->num_prompt_tokens), 1);
    }
    else {
        printf("Error: unknown LLM arch.\n");
        return NULL;
    }

    for(int i = 0; i < session->num_prompt_tokens; i++) {
        session->output_ids[i] = prompt_tokens[i];
    }

    session->next_token = prompt_tokens[0]; // kick off with the first token in the prompt
    session->pos = 0;     // position in the sequence

    session->is_prefilling = 0;

    session->output_text = NULL;

    free(prompt_tokens);

    return session;
}


int32_t llm_session_step(Nano_Context *ctx, Nano_Session *session) {
    if (session->pos < session->max_seq_len) {

        session->is_prefilling = (session->pos < session->num_prompt_tokens - 1) ? 1 : 0;

        session->next_token = generate_next_token(ctx, session->output_ids, session->pos, session->is_prefilling);

        if (session->is_prefilling == 0) {
            session->output_ids[session->num_prompt_tokens + (session->output_count)++] = session->next_token;
            if (ctx->llm->arch == LLM_ARCH_NANO) {
                session->output_text = decode_nano(ctx->tokenizer, session->output_ids + session->num_prompt_tokens, session->output_count);
            }
            else if (ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3) {
                session->output_text = decode_bpe(ctx->tokenizer, session->output_ids + session->num_prompt_tokens, session->output_count);
            }
            else {
                printf("Error: unknown LLM arch.\n");
                return LLM_STOPPED_WITH_ERROR;
            }
        }

        session->pos++;

        if (ctx->llm->arch == LLM_ARCH_NANO && (session->next_token == 0 || session->next_token == 3)) {
            return LLM_STOPPED_NORMALLY; // 遇到结束符号，主动结束
        }
        else if (
            (ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3) && 
            (session->is_prefilling == 0) &&
            (session->next_token == 151643 || session->next_token == 151645)) {
            return LLM_STOPPED_NORMALLY; // 遇到结束符号，主动结束
        }
        else {
            return (session->is_prefilling == 1) ? LLM_RUNNING_IN_PREFILLING : LLM_RUNNING_IN_DECODING;
        }
    }
    else {
        return LLM_STOPPED_WITH_ERROR;
    }
}


void llm_session_free(Nano_Session *session) {
    if(session->prompt)      free(session->prompt);
    if(session->output_ids)  free(session->output_ids);
    if(session->output_text) free(session->output_text);
    if(session)              free(session);
}




int32_t generate_sync(
    Nano_Context *ctx,
    wchar_t *prompt,
    uint32_t max_seq_len,
    int32_t (*on_prefilling)(Nano_Session*),
    int32_t (*on_decoding)(Nano_Session*),
    int32_t (*on_finished)(Nano_Session*)
) {
    Nano_Session *session = llm_session_init(ctx, prompt, max_seq_len);
    int32_t status = 0;
    while (1) {
        status = llm_session_step(ctx, session);
        if (status == LLM_RUNNING_IN_PREFILLING) {
            int32_t callback_flag = on_prefilling(session);
            // 外部被动中止
            if (callback_flag == LLM_STOPPED_IN_PREFILLING) {
                status = callback_flag;
                break;
            }
        }
        else if (status == LLM_RUNNING_IN_DECODING) {
            int32_t callback_flag = on_decoding(session);
            // 外部被动中止
            if (callback_flag == LLM_STOPPED_IN_DECODING) {
                status = callback_flag;
                break;
            }
        }
        else if (status == LLM_STOPPED_NORMALLY) {
            status = on_finished(session);
            break;
        }
        else {
            on_finished(session);
            status = LLM_STOPPED_WITH_ERROR;
            break;
        }
    }
    llm_session_free(session);
    return status;
}


// 序列生成：用于推理诸如排序、回文数等Nano架构的序列生成模型
void seq2seq(Nano_Context *ctx, wchar_t *input_list, wchar_t *output_list, uint32_t max_seq_len) {
    uint32_t num_prompt_tokens = 0;
    uint32_t *input_ids = encode_nano(ctx->tokenizer, input_list, &num_prompt_tokens);
    uint32_t *output_ids = (uint32_t *)calloc_dev(max_seq_len, sizeof(uint32_t));

    LLM *llm = ctx->llm;
    Sampler *sampler = ctx->sampler;

    float *output_logits = (float*)calloc_dev(max_seq_len * sampler->vocab_size, sizeof(float));

    // 阶段1：预填充KVCache。
    //   内层循环对输入序列的每一个pos进行前向计算，填充每一层的第pos位置上的KVCache。
    //   一次遍历完整个序列后，只有第一层的KVCache完成了对整个序列的全局自注意力，但下游各层并没有。
    //   下游各层的KVCache依赖于上游各层完成全局自注意力（输出的中间值）。因此这个过程需要执行L次。
    for (uint32_t i = 0; i < llm->config.n_layer; i++) {
        for (uint32_t pos = 0; pos < max_seq_len; pos++) {
            (void)llm_forward(input_ids[pos], pos, max_seq_len, 0, llm, NULL);

        }
    }

    // 阶段2：KVCache全部填充后，再对每个pos执行完整的forward，获得每个pos上的logits
    for (uint32_t pos = 0; pos < max_seq_len; pos++) {
        float *logits = llm_forward(input_ids[pos], pos, max_seq_len, 0, llm, NULL);
        memcpy(output_logits + pos * sampler->vocab_size, logits, sampler->vocab_size *sizeof(float));
    }

    // 阶段3：对每个pos上的logits进行单独的采样
    for (uint32_t pos = 0; pos < max_seq_len; pos++) {
        float *logits = output_logits + pos * sampler->vocab_size;
        output_ids[pos] = sample_argmax(logits, sampler->vocab_size);
    }

    free(input_ids);
    free(output_logits);

    wcscpy(output_list, decode_nano(ctx->tokenizer, output_ids, max_seq_len));
}

