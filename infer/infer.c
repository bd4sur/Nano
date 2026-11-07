//
// Nano Language Model - Inference Engine written in C
//
//   BD4SUR 2024-10 2024-05
//
//   Forked from: https://github.com/karpathy/llama2.c
//

#include "infer.h"

// ===============================================================================
// Qwen3.5 辅助函数
// ===============================================================================

static uint32_t popcount32(uint32_t x) {
    uint32_t n = 0;
    while (x) { x &= (x - 1); n++; }
    return n;
}

static uint32_t qwen35_n_full_layers(const LLM_Config *cfg) {
    return popcount32(cfg->layer_type_mask);
}

static int qwen35_is_full_layer(const LLM_Config *cfg, uint32_t layer) {
    return ((cfg->layer_type_mask >> layer) & 1u) ? 1 : 0;
}

// 返回 layer 之前（不含 layer）同类层的数量，即该层在 compact 数组中的下标
static uint32_t qwen35_compact_index(const LLM_Config *cfg, uint32_t layer, int full) {
    uint32_t idx = 0;
    for (uint32_t i = 0; i < layer; i++) {
        if (qwen35_is_full_layer(cfg, i) == full) idx++;
    }
    return idx;
}

// ===============================================================================
// 模型文件解析·内存管理
// ===============================================================================

void malloc_fwd_buffer(LLM *llm, uint32_t max_seq_len) {
    FwdBuffer *s = &(llm->state);
    LLM_Config *llm_cfg = &(llm->config);

    uint32_t q_dim = llm_cfg->n_embd;
    uint32_t kv_dim = (llm_cfg->n_embd * llm_cfg->n_kv_head) / llm_cfg->n_head;
    uint32_t value_dim = 0;
    uint32_t key_dim = 0;
    uint32_t conv_dim = 0;
    uint32_t n_full = 0;
    uint32_t n_linear = 0;

    if (llm->arch == LLM_ARCH_NANO || llm->arch == LLM_ARCH_QWEN2) {
        q_dim = llm_cfg->n_embd;
        kv_dim = (llm_cfg->n_embd * llm_cfg->n_kv_head) / llm_cfg->n_head;
    }
    else if (llm->arch == LLM_ARCH_QWEN3) {
        q_dim = llm_cfg->head_dim * llm_cfg->n_head;
        kv_dim = llm_cfg->head_dim * llm_cfg->n_kv_head;
    }
    else if (llm->arch == LLM_ARCH_QWEN35) {
        n_full = qwen35_n_full_layers(llm_cfg);
        n_linear = llm_cfg->n_layer - n_full;
        q_dim = llm_cfg->head_dim * llm_cfg->n_head;
        kv_dim = llm_cfg->head_dim * llm_cfg->n_kv_head;
        key_dim = llm_cfg->linear_n_k_head * llm_cfg->linear_k_head_dim;
        value_dim = llm_cfg->linear_n_v_head * llm_cfg->linear_v_head_dim;
        conv_dim = 2 * key_dim + value_dim;
    }

    uint32_t xba_dim = (value_dim > q_dim) ? value_dim : q_dim;

    uint32_t xbuf_length = llm_cfg->n_embd + llm_cfg->n_embd + xba_dim + llm_cfg->n_embd + llm_cfg->n_hidden + llm_cfg->n_hidden
                         + q_dim + llm_cfg->n_head * max_seq_len + llm_cfg->vocab_size;
    s->xbuf = (float *)platform_calloc(xbuf_length, sizeof(float));
    float *xbuf = s->xbuf;

    s->x       = xbuf;  xbuf += llm_cfg->n_embd;
    s->xb      = xbuf;  xbuf += llm_cfg->n_embd;
    s->xba     = xbuf;  xbuf += xba_dim;
    s->xb2     = xbuf;  xbuf += llm_cfg->n_embd;
    s->hb      = xbuf;  xbuf += llm_cfg->n_hidden;
    s->hb2     = xbuf;  xbuf += llm_cfg->n_hidden;
    s->q       = xbuf;  xbuf += q_dim;
    s->att     = xbuf;  xbuf += llm_cfg->n_head * max_seq_len;
    s->logits  = xbuf;  xbuf += llm_cfg->vocab_size;

    // Qwen3.5 只有 full_attention 层需要 KV cache
    uint32_t n_kv_layers = (llm->arch == LLM_ARCH_QWEN35) ? n_full : llm_cfg->n_layer;
    uint32_t kvcache_length = n_kv_layers * max_seq_len * kv_dim * 2;
    s->kvcache = (float *)platform_calloc(kvcache_length, sizeof(float));
    float *kvcache = s->kvcache;

    s->k_cache = kvcache;  kvcache += n_kv_layers * max_seq_len * kv_dim;
    s->v_cache = kvcache;  kvcache += n_kv_layers * max_seq_len * kv_dim;

    if (llm->quant_type == QUANT_TYPE_Q80) {
        uint32_t gs = llm->group_size;

        uint32_t qvbuf_length = llm_cfg->n_embd + xba_dim + llm_cfg->n_hidden + ((llm->arch == LLM_ARCH_QWEN35) ? value_dim : 0);
        uint32_t qsbuf_length = llm_cfg->n_embd / gs + xba_dim / gs + llm_cfg->n_hidden / gs
                              + ((llm->arch == LLM_ARCH_QWEN35) ? value_dim / gs : 0);
        s->qvbuf = (QTYPE *)platform_calloc(qvbuf_length, sizeof(QTYPE));
        s->qsbuf = (float *)platform_calloc(qsbuf_length, sizeof(float));
        QTYPE *qvbuf = s->qvbuf;
        float *qsbuf = s->qsbuf;

        Q80_Tensor xq_tmp   = (Q80_Tensor) { .q = qvbuf, .s = qsbuf };  qvbuf += llm_cfg->n_embd; qsbuf += llm_cfg->n_embd / gs;
        Q80_Tensor xbaq_tmp = (Q80_Tensor) { .q = qvbuf, .s = qsbuf };  qvbuf += xba_dim; qsbuf += xba_dim / gs;
        Q80_Tensor hq_tmp   = (Q80_Tensor) { .q = qvbuf, .s = qsbuf };  qvbuf += llm_cfg->n_hidden; qsbuf += llm_cfg->n_hidden / gs;
        
        s->xq               = (Typed_Tensor) { .tensor_q80 = xq_tmp };
        s->xbaq             = (Typed_Tensor) { .tensor_q80 = xbaq_tmp };
        s->hq               = (Typed_Tensor) { .tensor_q80 = hq_tmp };

        if (llm->arch == LLM_ARCH_QWEN35) {
            Q80_Tensor vq_tmp = (Q80_Tensor) { .q = qvbuf, .s = qsbuf };  qvbuf += value_dim; qsbuf += value_dim / gs;
            s->lin_vq = (Typed_Tensor) { .tensor_q80 = vq_tmp };
        }
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        Q4k_Tensor *xq_q4k   = make_q4k_tensor(1, (uint32_t[]){llm_cfg->n_embd});
        Q4k_Tensor *xbaq_q4k = make_q4k_tensor(1, (uint32_t[]){xba_dim});
        Q4k_Tensor *hq_q4k   = make_q4k_tensor(1, (uint32_t[]){llm_cfg->n_hidden});
        s->xq   = (Typed_Tensor) { .tensor_q4k = xq_q4k };
        s->xbaq = (Typed_Tensor) { .tensor_q4k = xbaq_q4k };
        s->hq   = (Typed_Tensor) { .tensor_q4k = hq_q4k };
        if (llm->arch == LLM_ARCH_QWEN35) {
            Q4k_Tensor *vq_q4k = make_q4k_tensor(1, (uint32_t[]){value_dim});
            s->lin_vq = (Typed_Tensor) { .tensor_q4k = vq_q4k };
        }
    }

    // Qwen3.5 线性注意力：scratch + conv 状态 + recurrent 状态，统一分配
    if (llm->arch == LLM_ARCH_QWEN35) {
        uint32_t n_v = llm_cfg->linear_n_v_head;
        uint32_t k_dim = llm_cfg->linear_k_head_dim;
        uint32_t v_dim = llm_cfg->linear_v_head_dim;
        uint32_t k_hist = (llm_cfg->conv_kernel_dim > 1) ? (llm_cfg->conv_kernel_dim - 1) : 0;

        uint32_t scratch_len = 2 * q_dim      // qg
                             + conv_dim       // lin_conv
                             + value_dim      // lin_z
                             + 2 * n_v        // lin_b, lin_a
                             + conv_dim       // lin_conv_out
                             + value_dim      // lin_out
                             + 2 * n_v * v_dim; // kv_mem, delta
        uint32_t conv_state_len = n_linear * conv_dim * k_hist;
        uint32_t rec_state_len = n_linear * n_v * k_dim * v_dim;

        s->lin_buf = (float *)platform_calloc((uint64_t)scratch_len + conv_state_len + rec_state_len, sizeof(float));
        float *lb = s->lin_buf;

        s->qg          = lb;  lb += 2 * q_dim;
        s->lin_conv    = lb;  lb += conv_dim;
        s->lin_z       = lb;  lb += value_dim;
        s->lin_b       = lb;  lb += n_v;
        s->lin_a       = lb;  lb += n_v;
        s->lin_conv_out= lb;  lb += conv_dim;
        s->lin_out     = lb;  lb += value_dim;
        s->lin_kv_mem  = lb;  lb += n_v * v_dim;  // 每个 v head 一份，便于 OpenMP 并行
        s->lin_delta   = lb;  lb += n_v * v_dim;
        s->lin_conv_state = lb;  lb += conv_state_len;
        s->lin_recurrent  = lb;  lb += rec_state_len;
    }

    // ensure all mallocs went fine
    if (!s->xbuf || !s->kvcache || (llm->arch == LLM_ARCH_QWEN35 && !s->lin_buf)) {
        fprintf(stderr, "mem alloc failed!\n");
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
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        free(s->xq.tensor_q4k);
        free(s->xbaq.tensor_q4k);
        free(s->hq.tensor_q4k);
        if (llm->arch == LLM_ARCH_QWEN35) free(s->lin_vq.tensor_q4k);
    }

    if (llm->arch == LLM_ARCH_QWEN35) {
        free(s->lin_buf);
    }
}


static Typed_Tensor *qwen35_unpack_q4k_group(void **ptr) {
    uint64_t tlen = 0;
    Q4k_Tensor *t = unpack_q4k_tensor(*ptr, &tlen);
    Typed_Tensor *tt = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
    tt->tensor_q4k = t;
    *ptr = (uint8_t *)(*ptr) + tlen;
    return tt;
}

// Qwen3.5 的参数文件布局（与 tools/export_qwen35.py 一一对应）
static void memory_map_params_qwen35(LLM *llm, void *ptr) {
    LLM_Param *w = &(llm->params);
    LLM_Config *cfg = &(llm->config);

    uint32_t n_layer = cfg->n_layer;
    uint32_t n_embd = cfg->n_embd;
    uint32_t head_dim = cfg->head_dim;
    uint32_t q_dim = cfg->n_head * head_dim;
    uint32_t kv_dim = cfg->n_kv_head * head_dim;
    uint32_t n_full = qwen35_n_full_layers(cfg);
    uint32_t n_linear = n_layer - n_full;
    uint32_t key_dim = cfg->linear_n_k_head * cfg->linear_k_head_dim;
    uint32_t value_dim = cfg->linear_n_v_head * cfg->linear_v_head_dim;
    uint32_t conv_dim = 2 * key_dim + value_dim;
    uint32_t n_v = cfg->linear_n_v_head;
    uint32_t v_dim = cfg->linear_v_head_dim;

    float *fptr = (float *)ptr;

    // 所有层的输入/后注意力 RMSNorm，以及最终 RMSNorm
    w->rms_norm_attn  = fptr;  fptr += n_layer * n_embd;
    w->rms_norm_ffn   = fptr;  fptr += n_layer * n_embd;
    w->rms_norm_final = fptr;  fptr += n_embd;

    // 线性注意力的非量化参数（紧凑排列）
    w->conv1d  = fptr;  fptr += n_linear * conv_dim * cfg->conv_kernel_dim;
    w->A_log   = fptr;  fptr += n_linear * n_v;
    w->dt_bias = fptr;  fptr += n_linear * n_v;
    w->lin_norm= fptr;  fptr += n_linear * v_dim;

    w->in_proj_b = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
    w->in_proj_b->tensor_f32 = fptr;  fptr += n_linear * n_v * n_embd;
    w->in_proj_a = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
    w->in_proj_a->tensor_f32 = fptr;  fptr += n_linear * n_v * n_embd;

    // 全注意力层的 q_norm / k_norm（紧凑排列）
    w->q_norm = fptr;  fptr += n_full * head_dim;
    w->k_norm = fptr;  fptr += n_full * head_dim;

    // RoPE 表在 C 端按 rope_theta 生成，避免把 262144 长度的表写进模型文件
    uint32_t rope_half = cfg->rope_dim / 2;
    w->freq_cis_real = (float *)platform_calloc(cfg->block_size * rope_half, sizeof(float));
    w->freq_cis_imag = (float *)platform_calloc(cfg->block_size * rope_half, sizeof(float));
    for (uint32_t pos = 0; pos < cfg->block_size; pos++) {
        for (uint32_t i = 0; i < rope_half; i++) {
            float freq = 1.0f / powf(cfg->rope_theta, (float)(i * 2) / (float)cfg->rope_dim);
            w->freq_cis_real[pos * rope_half + i] = cosf(pos * freq);
            w->freq_cis_imag[pos * rope_half + i] = sinf(pos * freq);
        }
    }

    // 主参数（量化或 FP32）
    if (llm->quant_type == QUANT_TYPE_F32) {
        w->token_embedding = fptr;  fptr += (uint64_t)cfg->vocab_size * n_embd;

        w->w1 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->w1->tensor_f32 = fptr;  fptr += (uint64_t)n_layer * cfg->n_hidden * n_embd;
        w->w2 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->w2->tensor_f32 = fptr;  fptr += (uint64_t)n_layer * n_embd * cfg->n_hidden;
        w->w3 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->w3->tensor_f32 = fptr;  fptr += (uint64_t)n_layer * cfg->n_hidden * n_embd;

        w->wq = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->wq->tensor_f32 = fptr;  fptr += (uint64_t)n_full * (2 * q_dim) * n_embd;
        w->wk = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->wk->tensor_f32 = fptr;  fptr += (uint64_t)n_full * kv_dim * n_embd;
        w->wv = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->wv->tensor_f32 = fptr;  fptr += (uint64_t)n_full * kv_dim * n_embd;
        w->wo = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->wo->tensor_f32 = fptr;  fptr += (uint64_t)n_full * n_embd * q_dim;

        w->wqkv = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->wqkv->tensor_f32 = fptr;  fptr += (uint64_t)n_linear * conv_dim * n_embd;
        w->wz = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->wz->tensor_f32 = fptr;  fptr += (uint64_t)n_linear * value_dim * n_embd;
        w->wlo = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->wlo->tensor_f32 = fptr;  fptr += (uint64_t)n_linear * n_embd * value_dim;

        w->token_classifier = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
        w->token_classifier->tensor_f32 = cfg->is_shared_classifier ? w->token_embedding : fptr;
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        uint32_t gs = llm->group_size;
        void *p = (void *)fptr;

        w->q_tokens = parse_quantized_tensors(&p, 1, cfg->vocab_size * n_embd, gs);
        w->token_embedding = (float *)platform_calloc((uint64_t)cfg->vocab_size * n_embd, sizeof(float));
        dequantize(&w->q_tokens->tensor_q80, w->token_embedding, cfg->vocab_size * n_embd, gs);

        w->w1 = parse_quantized_tensors(&p, n_layer, cfg->n_hidden * n_embd, gs);
        w->w2 = parse_quantized_tensors(&p, n_layer, n_embd * cfg->n_hidden, gs);
        w->w3 = parse_quantized_tensors(&p, n_layer, cfg->n_hidden * n_embd, gs);

        w->wq = parse_quantized_tensors(&p, n_full, (2 * q_dim) * n_embd, gs);
        w->wk = parse_quantized_tensors(&p, n_full, kv_dim * n_embd, gs);
        w->wv = parse_quantized_tensors(&p, n_full, kv_dim * n_embd, gs);
        w->wo = parse_quantized_tensors(&p, n_full, n_embd * q_dim, gs);

        w->wqkv = parse_quantized_tensors(&p, n_linear, conv_dim * n_embd, gs);
        w->wz   = parse_quantized_tensors(&p, n_linear, value_dim * n_embd, gs);
        w->wlo  = parse_quantized_tensors(&p, n_linear, n_embd * value_dim, gs);

        w->token_classifier = cfg->is_shared_classifier ? w->q_tokens
                             : parse_quantized_tensors(&p, 1, n_embd * cfg->vocab_size, gs);
        fptr = (float *)p;
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        void *p = (void *)fptr;
        uint32_t ndim = 0;

        w->q_tokens = qwen35_unpack_q4k_group(&p);
        w->token_embedding = (float *)platform_calloc((uint64_t)cfg->vocab_size * n_embd, sizeof(float));
        uint32_t edim[2] = {cfg->vocab_size, n_embd};
        dequantize_tensor_q4k(w->q_tokens->tensor_q4k, w->token_embedding, &ndim, edim);

        w->w1 = qwen35_unpack_q4k_group(&p);
        w->w2 = qwen35_unpack_q4k_group(&p);
        w->w3 = qwen35_unpack_q4k_group(&p);

        w->wq = qwen35_unpack_q4k_group(&p);
        w->wk = qwen35_unpack_q4k_group(&p);
        w->wv = qwen35_unpack_q4k_group(&p);
        w->wo = qwen35_unpack_q4k_group(&p);

        w->wqkv = qwen35_unpack_q4k_group(&p);
        w->wz   = qwen35_unpack_q4k_group(&p);
        w->wlo  = qwen35_unpack_q4k_group(&p);

        w->token_classifier = cfg->is_shared_classifier ? w->q_tokens : qwen35_unpack_q4k_group(&p);
        fptr = (float *)p;
    }

    (void)fptr;
}

void memory_map_params(LLM *llm, void* ptr) {
    if (llm->arch == LLM_ARCH_QWEN35) {
        memory_map_params_qwen35(llm, ptr);
        return;
    }

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
        w->token_embedding = (float *)platform_calloc(cfg->vocab_size * cfg->n_embd, sizeof(float));
        dequantize(&w->q_tokens->tensor_q80, w->token_embedding, cfg->vocab_size * cfg->n_embd, llm->group_size);

        w->wq = parse_quantized_tensors(&ptr, n_layer, (cfg->n_head * head_size) * cfg->n_embd, llm->group_size);
        w->wk = parse_quantized_tensors(&ptr, n_layer, (cfg->n_kv_head * head_size) * cfg->n_embd, llm->group_size);
        w->wv = parse_quantized_tensors(&ptr, n_layer, (cfg->n_kv_head * head_size) * cfg->n_embd, llm->group_size);
        w->wo = parse_quantized_tensors(&ptr, n_layer, cfg->n_embd * (cfg->n_head * head_size), llm->group_size);

        w->w1 = parse_quantized_tensors(&ptr, n_layer, cfg->n_hidden * cfg->n_embd, llm->group_size);
        w->w2 = parse_quantized_tensors(&ptr, n_layer, cfg->n_embd * cfg->n_hidden, llm->group_size);
        w->w3 = parse_quantized_tensors(&ptr, n_layer, cfg->n_hidden * cfg->n_embd, llm->group_size);

        fptr = (float*)ptr;
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        ptr = (void*)fptr;

        uint64_t tlen = 0;

        Q4k_Tensor *q_tokens_q4k = unpack_q4k_tensor(ptr, &tlen);
        w->q_tokens = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor)); w->q_tokens->tensor_q4k = q_tokens_q4k;  ptr += tlen;
        w->token_embedding = (float *)platform_calloc(cfg->vocab_size * cfg->n_embd, sizeof(float));
        uint32_t tedim = 2;
        dequantize_tensor_q4k(q_tokens_q4k, w->token_embedding, &tedim, (uint32_t[]){cfg->vocab_size, cfg->n_embd});

        Q4k_Tensor *wq_q4k = unpack_q4k_tensor(ptr, &tlen); w->wq = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor)); w->wq->tensor_q4k = wq_q4k; ptr += tlen;
        Q4k_Tensor *wk_q4k = unpack_q4k_tensor(ptr, &tlen); w->wk = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor)); w->wk->tensor_q4k = wk_q4k; ptr += tlen;
        Q4k_Tensor *wv_q4k = unpack_q4k_tensor(ptr, &tlen); w->wv = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor)); w->wv->tensor_q4k = wv_q4k; ptr += tlen;
        Q4k_Tensor *wo_q4k = unpack_q4k_tensor(ptr, &tlen); w->wo = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor)); w->wo->tensor_q4k = wo_q4k; ptr += tlen;

        Q4k_Tensor *w1_q4k = unpack_q4k_tensor(ptr, &tlen); w->w1 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor)); w->w1->tensor_q4k = w1_q4k; ptr += tlen;
        Q4k_Tensor *w2_q4k = unpack_q4k_tensor(ptr, &tlen); w->w2 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor)); w->w2->tensor_q4k = w2_q4k; ptr += tlen;
        Q4k_Tensor *w3_q4k = unpack_q4k_tensor(ptr, &tlen); w->w3 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor)); w->w3->tensor_q4k = w3_q4k; ptr += tlen;

        fptr = (float*)ptr;
    }
    else if (llm->quant_type == QUANT_TYPE_F32) {
        w->token_embedding = fptr; fptr += cfg->vocab_size * cfg->n_embd;

        w->wq = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));  w->wq->tensor_f32 = fptr;  fptr += n_layer * (cfg->n_head * head_size) * cfg->n_embd;
        w->wk = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));  w->wk->tensor_f32 = fptr;  fptr += n_layer * (cfg->n_kv_head * head_size) * cfg->n_embd;
        w->wv = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));  w->wv->tensor_f32 = fptr;  fptr += n_layer * (cfg->n_kv_head * head_size) * cfg->n_embd;
        w->wo = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));  w->wo->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * (cfg->n_head * head_size);

        w->w1 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));  w->w1->tensor_f32 = fptr;  fptr += n_layer * cfg->n_hidden * cfg->n_embd;
        w->w2 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));  w->w2->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * cfg->n_hidden;
        w->w3 = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));  w->w3->tensor_f32 = fptr;  fptr += n_layer * cfg->n_hidden * cfg->n_embd;
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
        w->freq_cis_real = platform_calloc(cfg->block_size * head_size / 2, sizeof(float));
        w->freq_cis_imag = platform_calloc(cfg->block_size * head_size / 2, sizeof(float));

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
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        w->token_classifier = w->q_tokens;
    }
    else if (llm->quant_type == QUANT_TYPE_F32) {
        w->token_classifier = (Typed_Tensor *)platform_calloc(1, sizeof(Typed_Tensor));
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

    if (model_type == LLM_ARCH_QWEN35) {
        config->layer_type_mask   = header[offset]; offset++;
        config->conv_kernel_dim   = header[offset]; offset++;
        config->linear_n_k_head   = header[offset]; offset++;
        config->linear_n_v_head   = header[offset]; offset++;
        config->linear_k_head_dim = header[offset]; offset++;
        config->linear_v_head_dim = header[offset]; offset++;
        config->rope_dim          = header[offset]; offset++;
        uint32_t theta_bits       = header[offset]; offset++;
        memcpy(&config->rope_theta, &theta_bits, sizeof(float));
    }

    llm->arch = model_type;

    llm->quant_type = (quant_type == QUANT_TYPE_F32 || quant_type == QUANT_TYPE_Q80 || quant_type == QUANT_TYPE_Q4K) ? quant_type : QUANT_TYPE_Q80;
    llm->group_size = group_size;

    // 解析词表，同时构建trie树和hashmap

    uint32_t *tokenzier_ptr = (uint32_t *)(buffer) + (header_byte_length / sizeof(uint32_t));
    uint32_t tokenizer_field_bytes = *tokenzier_ptr;

    if (llm->arch == LLM_ARCH_NANO) {
        uint32_t *vocab_ptr = tokenzier_ptr + 1;

        uint32_t byte_count = 0;
        uint32_t char_count = 0;

        tk->vocab_size = *vocab_ptr; vocab_ptr++;

        tk->token_list        = (wchar_t **)platform_calloc(tk->vocab_size, sizeof(wchar_t *));
        tk->unicode_charset   = (wchar_t  *)platform_calloc(tk->vocab_size, sizeof(wchar_t));
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

            wchar_t *token = (wchar_t *)platform_calloc(token_length+1, sizeof(wchar_t));
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
                free(ids); // add_token为拷贝语义（memcpy），ids由本函数释放
            }
        }
    }
    else if (llm->arch == LLM_ARCH_QWEN35) {
        // Qwen3.5 的导出工具会为全部 vocab_size 个条目写入词表（含尾部空槽位）
        build_bpe_tokenizer(tk, (uint8_t*)tokenzier_ptr, config->vocab_size);
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
    uint8_t *buffer = (uint8_t *)platform_calloc(llm->file_size + 1, sizeof(uint8_t));
    if (!buffer)  { fprintf(stderr, "mem alloc failed.\n"); exit(EXIT_FAILURE); }
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

    if (llm->quant_type == QUANT_TYPE_Q80 || llm->quant_type == QUANT_TYPE_Q4K) {
        free(llm->params.token_embedding);
    }
    free(llm->params.q_tokens);
    // 共享分类器时 token_classifier 与 q_tokens 指向同一对象，不能重复释放
    if (llm->params.token_classifier && llm->params.token_classifier != llm->params.q_tokens) {
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
    else if (llm->arch == LLM_ARCH_QWEN2 || llm->arch == LLM_ARCH_QWEN3 || llm->arch == LLM_ARCH_QWEN35) {
        free_bpe_tokenizer(tk);
        if (llm->arch == LLM_ARCH_QWEN3) {
            free(llm->params.freq_cis_real);
            free(llm->params.freq_cis_imag);
        }
        else if (llm->arch == LLM_ARCH_QWEN35) {
            free(llm->params.freq_cis_real);
            free(llm->params.freq_cis_imag);
            free(llm->params.in_proj_b);
            free(llm->params.in_proj_a);
            free(llm->params.wqkv);
            free(llm->params.wz);
            free(llm->params.wlo);
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
    s->q0 = platform_calloc(lora_cfg->lora_rank, sizeof(float));
    s->k0 = platform_calloc(lora_cfg->lora_rank, sizeof(float));
    s->v0 = platform_calloc(lora_cfg->lora_rank, sizeof(float));
    s->o0 = platform_calloc(lora_cfg->lora_rank, sizeof(float));
    s->q1 = platform_calloc(llm_cfg->n_embd, sizeof(float));
    s->k1 = platform_calloc(kv_dim, sizeof(float));
    s->v1 = platform_calloc(kv_dim, sizeof(float));
    s->o1 = platform_calloc(llm_cfg->n_embd, sizeof(float));
    // ensure all mallocs went fine
    if (!s->q0 || !s->k0 || !s->v0 || !s->o0 ||
        !s->q1 || !s->k1 || !s->v1 || !s->o1 ) {
        fprintf(stderr, "mem alloc failed!\n");
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

    uint8_t *lora_buffer = (uint8_t *)platform_calloc(file_size + 1, sizeof(uint8_t));
    if (!lora_buffer)  { fprintf(stderr, "mem alloc failed.\n"); exit(EXIT_FAILURE); }

    if(fread(lora_buffer, sizeof(uint8_t), file_size, file) != file_size) { exit(EXIT_FAILURE); }
    LoRA *p_lora = (LoRA *)platform_calloc(1, sizeof(LoRA));

    p_lora->data = (float*)lora_buffer;

    parse_lora_file(lora_buffer, p_lora, llm);

    malloc_fwd_buffer_with_lora(llm, &p_lora->config);

    return p_lora;
}

LoRA *load_lora_from_buffer(LLM *llm, uint8_t *buffer) {
    LoRA *p_lora = (LoRA *)platform_calloc(1, sizeof(LoRA));
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
    Nano_Context *ctx = (Nano_Context*)platform_calloc(1, sizeof(Nano_Context));
    ctx->max_seq_len = max_seq_len;
    ctx->random_seed = random_seed;
    ctx->llm = (LLM *)platform_calloc(1, (sizeof(LLM)));
    ctx->tokenizer = (Tokenizer *)platform_calloc(1, (sizeof(Tokenizer)));
    ctx->lora = NULL;
    load_llm_from_buffer(ctx->llm, ctx->tokenizer, buffer, ctx->max_seq_len);
    ctx->sampler = build_sampler(ctx->llm->config.vocab_size, repetition_penalty, temperature, top_p, top_k, ctx->random_seed);
    return ctx;
}

Nano_Context *llm_context_init(char *model_path, char *lora_path, uint32_t max_seq_len, float repetition_penalty, float temperature, float top_p, uint32_t top_k, uint64_t random_seed) {
    Nano_Context *ctx = (Nano_Context*)platform_calloc(1, sizeof(Nano_Context));
    ctx->max_seq_len = max_seq_len;
    ctx->random_seed = random_seed;
    ctx->llm = (LLM *)platform_calloc(1, (sizeof(LLM)));
    ctx->tokenizer = (Tokenizer *)platform_calloc(1, (sizeof(Tokenizer)));
    load_llm(ctx->llm, ctx->tokenizer, model_path, max_seq_len);
    ctx->sampler = build_sampler(ctx->llm->config.vocab_size, repetition_penalty, temperature, top_p, top_k, ctx->random_seed);
    ctx->lora = (lora_path) ? load_lora(ctx->llm, lora_path) : NULL;
    return ctx;
}

void llm_context_free(Nano_Context *ctx) {
    free_llm(ctx->llm, ctx->tokenizer);
    free(ctx->tokenizer); // free_llm内部只调用free_tokenizer释放成员，Tokenizer结构体本体在此释放（infer.c:556/568分配）
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

void rmsnorm_eps(float* o, float* x, float* weight, int size, float eps) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += eps;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    rmsnorm_eps(o, x, weight, size, 1e-5f);
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



// ===============================================================================
// Qwen3.5 前向传播
// ===============================================================================

#define QWEN35_RMS_EPS (1e-6f)


static float qwen35_silu(float x) {
    return x / (1.0f + expf(-x));
}

static float qwen35_softplus(float x) {
    // log(1 + exp(x))，x 较大时直接返回 x 以避免溢出
    return (x > 20.0f) ? x : log1pf(expf(x));
}

static void qwen35_l2norm_inplace(float *x, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss + 1e-6f);
    for (uint32_t i = 0; i < n; i++) x[i] *= inv;
}

// Gated DeltaNet 线性注意力（单 token 递推形式）
static void qwen35_linear_attention(
    Nano_Context *ctx, float *xb, float *xb2,
    uint32_t layer, uint32_t lin_idx, uint32_t pos, LLM *llm)
{
    LLM_Config *cfg = &llm->config;
    LLM_Param *w = &llm->params;
    FwdBuffer *s = &llm->state;

    uint32_t n_embd = cfg->n_embd;
    uint32_t key_dim = cfg->linear_n_k_head * cfg->linear_k_head_dim;
    uint32_t value_dim = cfg->linear_n_v_head * cfg->linear_v_head_dim;
    uint32_t conv_dim = 2 * key_dim + value_dim;
    uint32_t k_dim = cfg->linear_k_head_dim;
    uint32_t v_dim = cfg->linear_v_head_dim;
    uint32_t n_v = cfg->linear_n_v_head;
    uint32_t n_k = cfg->linear_n_k_head;
    uint32_t ratio = n_v / n_k;
    uint32_t gs = llm->group_size;

    // 1) QKV / gate / beta / decay 投影
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_QKV}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->lin_conv, xb, w->wqkv->tensor_f32 + (uint64_t)lin_idx * conv_dim * n_embd, n_embd, conv_dim);
        matmul(s->lin_z,    xb, w->wz->tensor_f32   + (uint64_t)lin_idx * value_dim * n_embd, n_embd, value_dim);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, xb, n_embd, gs);
        matmul_quant(s->lin_conv, &s->xq, w->wqkv + lin_idx, n_embd, conv_dim, gs);
        matmul_quant(s->lin_z,    &s->xq, w->wz   + lin_idx, n_embd, value_dim, gs);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(xb, 1, (uint32_t[]){n_embd}, s->xq.tensor_q4k);
        matmul_q4k(s->lin_conv, s->xq.tensor_q4k, w->wqkv->tensor_q4k, lin_idx);
        matmul_q4k(s->lin_z,    s->xq.tensor_q4k, w->wz->tensor_q4k,   lin_idx);
    }

    matmul(s->lin_a, xb, w->in_proj_a->tensor_f32 + (uint64_t)lin_idx * n_v * n_embd, n_embd, n_v);
    matmul(s->lin_b, xb, w->in_proj_b->tensor_f32 + (uint64_t)lin_idx * n_v * n_embd, n_embd, n_v);

    // 2) depthwise causal conv1d + SiLU
    uint32_t K = cfg->conv_kernel_dim;
    uint32_t hist_len = (K > 1) ? (K - 1) : 0;
    float *hist = s->lin_conv_state + (uint64_t)lin_idx * conv_dim * hist_len;

    // conv 计算需要 K 个输入：state 保存前 K-1 个，当前 token 直接从 lin_conv 取。
    // 结果先写到 lin_conv_out，再更新 state、最后覆盖 lin_conv，避免丢失当前 token 的原始值。
    float *conv_out = s->lin_conv_out;
    float *cw = w->conv1d + (uint64_t)lin_idx * conv_dim * K;
    for (uint32_t c = 0; c < conv_dim; c++) {
        float val = cw[c * K + (K - 1)] * s->lin_conv[c];
        for (uint32_t k = 0; k + 1 < K; k++) {
            val += cw[c * K + k] * hist[k * conv_dim + c];
        }
        conv_out[c] = qwen35_silu(val);
    }

    if (hist_len > 0) {
        if (hist_len > 1) {
            memmove(hist, hist + conv_dim, (hist_len - 1) * conv_dim * sizeof(float));
        }
        memcpy(hist + (hist_len - 1) * conv_dim, s->lin_conv, conv_dim * sizeof(float));
    }
    memcpy(s->lin_conv, conv_out, conv_dim * sizeof(float));


    float *q = s->lin_conv;
    float *k = s->lin_conv + key_dim;
    float *v = s->lin_conv + 2 * key_dim;

    // beta / decay：直接用 lin_b/lin_a 原地替换
    for (uint32_t hv = 0; hv < n_v; hv++) {
        float bval = s->lin_b[hv];
        float aval = s->lin_a[hv];
        s->lin_b[hv] = 1.0f / (1.0f + expf(-bval));
        s->lin_a[hv] = -expf(w->A_log[lin_idx * n_v + hv]) * qwen35_softplus(aval + w->dt_bias[lin_idx * n_v + hv]);
    }

    // 3) 先对每个 key head 做 l2norm（若 n_v > n_k，多个 v head 共享同一 key head）
    for (uint32_t kh = 0; kh < n_k; kh++) {
        qwen35_l2norm_inplace(q + kh * k_dim, k_dim);
        qwen35_l2norm_inplace(k + kh * k_dim, k_dim);
    }

    // 4) 逐 v head 递推更新 recurrent state，并计算输出
    float *rec_base = s->lin_recurrent + (uint64_t)lin_idx * n_v * k_dim * v_dim;

    int hv_i;
    #pragma omp parallel for private(hv_i)
    for (hv_i = 0; hv_i < (int)n_v; hv_i++) {
        uint32_t hv = (uint32_t)hv_i;
        uint32_t kh = hv / ratio;

        float *qh = q + kh * k_dim;
        float *khp = k + kh * k_dim;
        float *vh = v + hv * v_dim;
        float *S = rec_base + (uint64_t)hv * k_dim * v_dim;
        float *kv_mem = s->lin_kv_mem + hv * v_dim;
        float *delta = s->lin_delta + hv * v_dim;
        float *out = s->lin_out + hv * v_dim;

        float decay = expf(s->lin_a[hv]);
        for (uint32_t i = 0; i < k_dim * v_dim; i++) S[i] *= decay;

        memset(kv_mem, 0, v_dim * sizeof(float));
        for (uint32_t i = 0; i < k_dim; i++) {
            float ki = khp[i];
            float *row = S + (uint64_t)i * v_dim;
            for (uint32_t j = 0; j < v_dim; j++) kv_mem[j] += ki * row[j];
        }

        float bval = s->lin_b[hv];
        for (uint32_t j = 0; j < v_dim; j++) delta[j] = (vh[j] - kv_mem[j]) * bval;

        for (uint32_t i = 0; i < k_dim; i++) {
            float ki = khp[i];
            float *row = S + (uint64_t)i * v_dim;
            for (uint32_t j = 0; j < v_dim; j++) row[j] += ki * delta[j];
        }

        float qscale = 1.0f / sqrtf((float)k_dim);
        for (uint32_t j = 0; j < v_dim; j++) out[j] = 0.0f;
        for (uint32_t i = 0; i < k_dim; i++) {
            float qi = qh[i] * qscale;
            float *row = S + (uint64_t)i * v_dim;
            for (uint32_t j = 0; j < v_dim; j++) out[j] += qi * row[j];
        }

        // 5) RMSNormGated：norm -> weight -> silu(gate)
        float ss = 0.0f;
        for (uint32_t j = 0; j < v_dim; j++) ss += out[j] * out[j];
        float scale = 1.0f / sqrtf(ss / (float)v_dim + QWEN35_RMS_EPS);
        float *zn = s->lin_z + hv * v_dim;
        float *ln = w->lin_norm + (uint64_t)lin_idx * v_dim;
        for (uint32_t j = 0; j < v_dim; j++) {
            out[j] = ln[j] * (out[j] * scale) * qwen35_silu(zn[j]);
        }
    }


    // 6) out_proj
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_O}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(xb2, s->lin_out, w->wlo->tensor_f32 + (uint64_t)lin_idx * n_embd * value_dim, value_dim, n_embd);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->lin_vq.tensor_q80, s->lin_out, value_dim, gs);
        matmul_quant(xb2, &s->lin_vq, w->wlo + lin_idx, value_dim, n_embd, gs);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(s->lin_out, 1, (uint32_t[]){value_dim}, s->lin_vq.tensor_q4k);
        matmul_q4k(xb2, s->lin_vq.tensor_q4k, w->wlo->tensor_q4k, lin_idx);
    }

}

// 全注意力层（部分旋转 RoPE + 输出 gate）
static void qwen35_full_attention(
    Nano_Context *ctx, float *xb, float *xb2,
    uint32_t layer, uint32_t full_idx, uint32_t pos,
    uint32_t max_seq_len, uint32_t is_causal, LLM *llm)
{
    LLM_Config *cfg = &llm->config;
    LLM_Param *w = &llm->params;
    FwdBuffer *s = &llm->state;

    uint32_t n_embd = cfg->n_embd;
    uint32_t head_dim = cfg->head_dim;
    uint32_t q_dim = cfg->n_head * head_dim;
    uint32_t kv_dim = cfg->n_kv_head * head_dim;
    uint32_t kv_mul = cfg->n_head / cfg->n_kv_head;
    uint32_t gs = llm->group_size;

#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_QKV}, ctx->observation_env);
#endif

    // q_proj 输出 2*q_dim：每个 head 依次是 [query(head_dim), gate(head_dim)]
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->qg, xb, w->wq->tensor_f32 + (uint64_t)full_idx * (2 * q_dim) * n_embd, n_embd, 2 * q_dim);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, xb, n_embd, gs);
        matmul_quant(s->qg, &s->xq, w->wq + full_idx, n_embd, 2 * q_dim, gs);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(xb, 1, (uint32_t[]){n_embd}, s->xq.tensor_q4k);
        matmul_q4k(s->qg, s->xq.tensor_q4k, w->wq->tensor_q4k, full_idx);
    }


    uint32_t layer_offset = full_idx * max_seq_len * kv_dim;
    s->k = s->k_cache + layer_offset + pos * kv_dim;
    s->v = s->v_cache + layer_offset + pos * kv_dim;

    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->k, xb, w->wk->tensor_f32 + (uint64_t)full_idx * kv_dim * n_embd, n_embd, kv_dim);
        matmul(s->v, xb, w->wv->tensor_f32 + (uint64_t)full_idx * kv_dim * n_embd, n_embd, kv_dim);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, xb, n_embd, gs);
        matmul_quant(s->k, &s->xq, w->wk + full_idx, n_embd, kv_dim, gs);
        matmul_quant(s->v, &s->xq, w->wv + full_idx, n_embd, kv_dim, gs);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(xb, 1, (uint32_t[]){n_embd}, s->xq.tensor_q4k);
        matmul_q4k(s->k, s->xq.tensor_q4k, w->wk->tensor_q4k, full_idx);
        matmul_q4k(s->v, s->xq.tensor_q4k, w->wv->tensor_q4k, full_idx);
    }

    // q/k norm + 部分旋转 RoPE
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_QK_ROPE}, ctx->observation_env);
#endif
    uint32_t rope_half = cfg->rope_dim / 2;
    float *freq_r = w->freq_cis_real + (uint64_t)pos * rope_half;
    float *freq_i = w->freq_cis_imag + (uint64_t)pos * rope_half;

    for (uint32_t h = 0; h < cfg->n_head; h++) {
        float *q = s->q + h * head_dim;
        memcpy(q, s->qg + h * 2 * head_dim, head_dim * sizeof(float));
        rmsnorm_eps(q, q, w->q_norm + (uint64_t)full_idx * head_dim, head_dim, QWEN35_RMS_EPS);
        rope_qwen3(q, cfg->rope_dim, pos, freq_r, freq_i);
    }
    for (uint32_t h = 0; h < cfg->n_kv_head; h++) {
        float *k = s->k + h * head_dim;
        rmsnorm_eps(k, k, w->k_norm + (uint64_t)full_idx * head_dim, head_dim, QWEN35_RMS_EPS);
        rope_qwen3(k, cfg->rope_dim, pos, freq_r, freq_i);
    }


    // attention
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_MHA}, ctx->observation_env);
#endif
    int h_i;
    #pragma omp parallel for private(h_i)
    for (h_i = 0; h_i < (int)cfg->n_head; h_i++) {
        uint32_t h = (uint32_t)h_i;
        float *q = s->q + h * head_dim;
        float *att = s->att + (uint64_t)h * max_seq_len;
        uint32_t attn_range = is_causal ? (pos + 1) : max_seq_len;

        for (uint32_t t = 0; t < attn_range; t++) {
            float *k = s->k_cache + layer_offset + (uint64_t)t * kv_dim + (h / kv_mul) * head_dim;
            float score = 0.0f;
            for (uint32_t i = 0; i < head_dim; i++) score += q[i] * k[i];
            att[t] = score / sqrtf((float)head_dim);
        }
        softmax(att, attn_range);

        float *xba = s->xba + h * head_dim;
        memset(xba, 0, head_dim * sizeof(float));
        for (uint32_t t = 0; t < attn_range; t++) {
            float *v = s->v_cache + layer_offset + (uint64_t)t * kv_dim + (h / kv_mul) * head_dim;
            float a = att[t];
            for (uint32_t i = 0; i < head_dim; i++) xba[i] += a * v[i];
        }
    }


    // 输出 gate：sigmoid(gate)
    for (uint32_t h = 0; h < cfg->n_head; h++) {
        float *xba = s->xba + h * head_dim;
        float *gate = s->qg + h * 2 * head_dim + head_dim;
        for (uint32_t i = 0; i < head_dim; i++) {
            xba[i] *= 1.0f / (1.0f + expf(-gate[i]));
        }
    }

#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_O}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(xb2, s->xba, w->wo->tensor_f32 + (uint64_t)full_idx * n_embd * q_dim, q_dim, n_embd);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xbaq.tensor_q80, s->xba, q_dim, gs);
        matmul_quant(xb2, &s->xbaq, w->wo + full_idx, q_dim, n_embd, gs);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(s->xba, 1, (uint32_t[]){q_dim}, s->xbaq.tensor_q4k);
        matmul_q4k(xb2, s->xbaq.tensor_q4k, w->wo->tensor_q4k, full_idx);
    }
}

// Qwen3.5 的 FFN（与 Nano/Qwen3 的 SwiGLU 相同）
static void qwen35_ffn(Nano_Context *ctx, float *x, uint32_t layer, LLM *llm) {
    LLM_Config *cfg = &llm->config;
    LLM_Param *w = &llm->params;
    FwdBuffer *s = &llm->state;
    uint32_t n_embd = cfg->n_embd;
    uint32_t n_hidden = cfg->n_hidden;
    uint32_t gs = llm->group_size;

#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_FFN_NORM}, ctx->observation_env);
#endif
    rmsnorm_eps(s->xb, x, w->rms_norm_ffn + (uint64_t)layer * n_embd, n_embd, QWEN35_RMS_EPS);

#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_W1W3}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->hb,  s->xb, w->w1->tensor_f32 + (uint64_t)layer * n_hidden * n_embd, n_embd, n_hidden);
        matmul(s->hb2, s->xb, w->w3->tensor_f32 + (uint64_t)layer * n_hidden * n_embd, n_embd, n_hidden);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, s->xb, n_embd, gs);
        matmul_quant(s->hb,  &s->xq, w->w1 + layer, n_embd, n_hidden, gs);
        matmul_quant(s->hb2, &s->xq, w->w3 + layer, n_embd, n_hidden, gs);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(s->xb, 1, (uint32_t[]){n_embd}, s->xq.tensor_q4k);
        matmul_q4k(s->hb,  s->xq.tensor_q4k, w->w1->tensor_q4k, layer);
        matmul_q4k(s->hb2, s->xq.tensor_q4k, w->w3->tensor_q4k, layer);
    }

    for (uint32_t i = 0; i < n_hidden; i++) {
        float val = s->hb[i];
        val *= 1.0f / (1.0f + expf(-val));
        val *= s->hb2[i];
        s->hb[i] = val;
    }

#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_W2}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->xb, s->hb, w->w2->tensor_f32 + (uint64_t)layer * n_embd * n_hidden, n_hidden, n_embd);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->hq.tensor_q80, s->hb, n_hidden, gs);
        matmul_quant(s->xb, &s->hq, w->w2 + layer, n_hidden, n_embd, gs);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(s->hb, 1, (uint32_t[]){n_hidden}, s->hq.tensor_q4k);
        matmul_q4k(s->xb, s->hq.tensor_q4k, w->w2->tensor_q4k, layer);
    }


    for (uint32_t i = 0; i < n_embd; i++) x[i] += s->xb[i];
}

static void transformer_block_forward_qwen35(
    Nano_Context *ctx, float *x, uint32_t layer, uint32_t pos,
    uint32_t max_seq_len, uint32_t is_causal, LLM *llm, LoRA *lora)
{
    (void)lora; // Qwen3.5 暂不支持 LoRA

    LLM_Config *cfg = &llm->config;
    LLM_Param *w = &llm->params;
    FwdBuffer *s = &llm->state;
    uint32_t n_embd = cfg->n_embd;

#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_ATTN_NORM}, ctx->observation_env);
#endif
    rmsnorm_eps(s->xb, x, w->rms_norm_attn + (uint64_t)layer * n_embd, n_embd, QWEN35_RMS_EPS);

    int is_full = qwen35_is_full_layer(cfg, layer);
    uint32_t compact_idx = qwen35_compact_index(cfg, layer, is_full);

    if (is_full) {
        qwen35_full_attention(ctx, s->xb, s->xb2, layer, compact_idx, pos, max_seq_len, is_causal, llm);
    }
    else {
        qwen35_linear_attention(ctx, s->xb, s->xb2, layer, compact_idx, pos, llm);
    }

    for (uint32_t i = 0; i < n_embd; i++) x[i] += s->xb2[i];

    qwen35_ffn(ctx, x, layer, llm);
}

// 前向传播一个Transformer块
// 函数的输入输出都是x，输入的x是上一层传入的值，输出即为本层输出。复用同一块内存（原地更新），波动传递中间激活值。
// NOTE is_causal参数的说明详见llm_forward函数
// NOTE 本函数每次执行都会重新计算一次模型的结构参数，但是开销很小。比起逻辑上的清晰和后续流水线并行改造，这点代价是值得的。
void transformer_block_forward(Nano_Context *ctx, float *x, uint32_t layer, uint32_t pos, uint32_t max_seq_len, uint32_t is_causal, LLM* llm, LoRA *lora) {

    if (llm->arch == LLM_ARCH_QWEN35) {
        transformer_block_forward_qwen35(ctx, x, layer, pos, max_seq_len, is_causal, llm, lora);
        return;
    }

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
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_ATTN_NORM}, ctx->observation_env);
#endif
    rmsnorm(s->xb, x, w->rms_norm_attn + layer * n_embd, n_embd);

    // key and value point to the kv cache
    // NOTE KV缓存的长度由max_seq_len参数指定，不一定等于模型本身的block_size。这样做的目的是为了按需控制KV缓存的大小。
    int layer_offset = layer * max_seq_len * kv_dim;
    s->k = s->k_cache + layer_offset + pos * kv_dim;
    s->v = s->v_cache + layer_offset + pos * kv_dim;

    // qkv matmuls for this position
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_QKV}, ctx->observation_env);
#endif
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
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(s->xb, 1, (uint32_t[]){n_embd}, s->xq.tensor_q4k);
        matmul_q4k(s->q, s->xq.tensor_q4k, w->wq->tensor_q4k, layer);
        matmul_q4k(s->k, s->xq.tensor_q4k, w->wk->tensor_q4k, layer);
        matmul_q4k(s->v, s->xq.tensor_q4k, w->wv->tensor_q4k, layer);
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
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_QK_ROPE}, ctx->observation_env);
#endif
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
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_MHA}, ctx->observation_env);
#endif
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
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_O}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->xb2, s->xba, w->wo->tensor_f32 + layer*n_embd*q_dim, q_dim, n_embd);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xbaq.tensor_q80, s->xba, q_dim, llm->group_size);
        matmul_quant(s->xb2, &s->xbaq, w->wo + layer, q_dim, n_embd, llm->group_size);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(s->xba, 1, (uint32_t[]){q_dim}, s->xbaq.tensor_q4k);
        matmul_q4k(s->xb2, s->xbaq.tensor_q4k, w->wo->tensor_q4k, layer);
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
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_FFN_NORM}, ctx->observation_env);
#endif
    rmsnorm(s->xb, x, w->rms_norm_ffn + layer*n_embd, n_embd);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_W1W3}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->hb, s->xb, w->w1->tensor_f32 + layer*n_hidden*n_embd, n_embd, n_hidden);
        matmul(s->hb2, s->xb, w->w3->tensor_f32 + layer*n_hidden*n_embd, n_embd, n_hidden);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, s->xb, n_embd, llm->group_size);
        matmul_quant(s->hb, &s->xq, w->w1 + layer, n_embd, n_hidden , llm->group_size);
        matmul_quant(s->hb2, &s->xq, w->w3 + layer, n_embd, n_hidden, llm->group_size);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(s->xb, 1, (uint32_t[]){n_embd}, s->xq.tensor_q4k);
        matmul_q4k(s->hb,  s->xq.tensor_q4k, w->w1->tensor_q4k, layer);
        matmul_q4k(s->hb2, s->xq.tensor_q4k, w->w3->tensor_q4k, layer);
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
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = layer, .phase = NANO_LLM_PHASE_W2}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->xb, s->hb, w->w2->tensor_f32 + layer*n_embd*n_hidden, n_hidden, n_embd);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->hq.tensor_q80, s->hb, n_hidden, llm->group_size);
        matmul_quant(s->xb, &s->hq, w->w2 + layer, n_hidden, n_embd, llm->group_size);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(s->hb, 1, (uint32_t[]){n_hidden}, s->hq.tensor_q4k);
        matmul_q4k(s->xb, s->hq.tensor_q4k, w->w2->tensor_q4k, layer);
    }

    // residual connection
    for (int i = 0; i < n_embd; i++) {
        x[i] += s->xb[i];
    }
}

// NOTE 该函数实现了在1个token+过往KVCache上的完整的因果自注意力前向推理。对于一般的自回归因果语言模型推理，is_causal=1。
//      参数is_causal=0启用全局自注意力，仅用于（且必须搭配用于）seq2seq函数。因该函数并没有实现完整的seq2seq全局自注意力前向推理。
//      seq2seq函数主要是整活用途，有大量冗余计算，效率极低。
float* llm_forward(Nano_Context *ctx, uint32_t token, uint32_t pos, uint32_t max_seq_len, uint32_t is_causal, LLM* llm, LoRA *lora) {

    LLM_Config *cfg = &llm->config;
    LLM_Param *w = &llm->params;
    FwdBuffer *s = &llm->state;

    // 模型结构参数
    uint32_t n_embd = cfg->n_embd;

    // Transformer层间传递的中间激活值
    float *x = s->x;

    // Qwen3.5 的线性注意力使用跨 token 的 recurrent 状态：每个新序列开始时清零
    if (llm->arch == LLM_ARCH_QWEN35 && pos == 0) {
        uint32_t n_full = qwen35_n_full_layers(cfg);
        uint32_t n_linear = cfg->n_layer - n_full;
        uint32_t key_dim = cfg->linear_n_k_head * cfg->linear_k_head_dim;
        uint32_t value_dim = cfg->linear_n_v_head * cfg->linear_v_head_dim;
        uint32_t conv_dim = 2 * key_dim + value_dim;
        uint32_t k_hist = (cfg->conv_kernel_dim > 1) ? (cfg->conv_kernel_dim - 1) : 0;
        memset(s->lin_conv_state, 0, (uint64_t)n_linear * conv_dim * k_hist * sizeof(float));
        memset(s->lin_recurrent, 0, (uint64_t)n_linear * cfg->linear_n_v_head * cfg->linear_k_head_dim * cfg->linear_v_head_dim * sizeof(float));
    }

    // 模型入口：将输入token的one-hot转为嵌入向量，等价于直接查token_embedding表
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = -1, .phase = NANO_LLM_PHASE_EMBEDDING}, ctx->observation_env);
#endif
    float* content_row = w->token_embedding + token * n_embd;
    memcpy(x, content_row, n_embd*sizeof(*x));

    // 前向传播，遍历各层的Transformer块，以x为层间传递的中间值（原地更新）
    for(uint64_t l = 0; l < cfg->n_layer; l++) {
        transformer_block_forward(ctx, x, l, pos, max_seq_len, is_causal, llm, lora);
    }

    // 最后一层RMSNorm
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = cfg->n_layer, .phase = NANO_LLM_PHASE_FINAL_NORM}, ctx->observation_env);
#endif
    rmsnorm(x, x, w->rms_norm_final, n_embd);

    // 模型出口：将最终的嵌入向量解码成logits
#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){.layer = cfg->n_layer, .phase = NANO_LLM_PHASE_CLASSIFY}, ctx->observation_env);
#endif
    if (llm->quant_type == QUANT_TYPE_F32) {
        matmul(s->logits, x, w->token_classifier->tensor_f32, cfg->n_embd, cfg->vocab_size);
    }
    else if (llm->quant_type == QUANT_TYPE_Q80) {
        quantize(&s->xq.tensor_q80, x, cfg->n_embd, llm->group_size);
        matmul_quant(s->logits, &s->xq, w->token_classifier, cfg->n_embd, cfg->vocab_size, llm->group_size);
    }
    else if (llm->quant_type == QUANT_TYPE_Q4K) {
        quantize_tensor_q4k_in_situ(x, 1, (uint32_t[]){cfg->n_embd}, s->xq.tensor_q4k);
        matmul_q4k(s->logits, s->xq.tensor_q4k, w->token_classifier->tensor_q4k, 0);
    }

    return s->logits;
}


// ===============================================================================
// 采样策略
// ===============================================================================

// 贪心采样：返回概率最大的下标
int sample_argmax(Nano_Context *ctx, float* probabilities, int n) {
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
int sample_multinomial(Nano_Context *ctx, float* probabilities, int n, float coin) {
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
int sample_top_p(Nano_Context *ctx, float* probabilities, int n, float top_p, ProbIndex* probindex, float coin) {
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

#if ENABLE_NANO_OBSERVATION
    ctx->observation((Nano_Observation){
        .layer = -1,
        .phase = NANO_LLM_PHASE_SAMPLE,
        .token_0 = (n0 > 0) ? probindex[0].index : 0,
        .token_1 = (n0 > 1) ? probindex[1].index : 0,
        .token_2 = (n0 > 2) ? probindex[2].index : 0,
        .token_3 = (n0 > 3) ? probindex[3].index : 0,
        .token_4 = (n0 > 4) ? probindex[4].index : 0,
        .token_5 = (n0 > 5) ? probindex[5].index : 0,
    }, ctx->observation_env);
#endif

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
    Sampler *sampler = (Sampler *)platform_calloc(1, sizeof(Sampler));
    sampler->vocab_size = vocab_size;
    sampler->repetition_penalty = repetition_penalty;
    sampler->temperature = temperature;
    sampler->top_p = top_p;
    sampler->top_k = top_k;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *)platform_calloc(sampler->vocab_size, sizeof(ProbIndex));
    return sampler;
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
    free(sampler); // 释放采样器结构体本体（build_sampler中platform_calloc分配）
}



// ===============================================================================
// 自回归文本生成
// ===============================================================================

uint32_t generate_next_token(Nano_Context *ctx, uint32_t *output_ids, uint32_t pos, int is_prefilling) {

    LLM *llm = ctx->llm;
    LoRA *lora = ctx->lora;
    Sampler *sampler = ctx->sampler;

    uint32_t next_token = output_ids[pos];

    float* logits = llm_forward(ctx, next_token, pos, ctx->max_seq_len, 1, llm, lora);

    // Pre-fill: if we are still processing the input prompt, force the next prompt token
    if (is_prefilling == 1) {
        next_token = output_ids[pos + 1];
        return next_token;
    }
    // Auto-regressive Decode
    else {
#if ENABLE_NANO_OBSERVATION
        ctx->observation((Nano_Observation){.layer = -1, .phase = NANO_LLM_PHASE_SAMPLE}, ctx->observation_env);
#endif
        // 复读惩罚：对过往出现过的词元施加惩罚，词元出现得越多，概率越低: ref arxiv:1909.05858
        uint32_t *tokenset = (uint32_t *)platform_calloc(sampler->vocab_size, sizeof(uint32_t));
        if (tokenset) { // 分配失败（内存紧张）时跳过复读惩罚，避免空指针写入
            for(uint32_t i = 0; i < pos; i++) {
                tokenset[output_ids[i]] = 1;  // 1表示output_ids中出现了这个token
            }
            for(uint32_t id = 0; id < sampler->vocab_size; id++) {
                if(tokenset[id] == 1) {
                    logits[id] /= sampler->repetition_penalty;
                }
            }
            free(tokenset);
        }

        // 温度采样：当温度设为0时，退化为贪心采样
        if (sampler->temperature == 0.0f) {
            next_token = sample_argmax(ctx, logits, sampler->vocab_size);
        }
        else {
            for(uint32_t q = 0; q < sampler->vocab_size; q++) {
                logits[q] /= sampler->temperature;
            }

            softmax(logits, sampler->vocab_size);

            // flip a (float) coin (this is our source of entropy for sampling)
            float coin = random_f32(&sampler->rng_state);

            if (sampler->top_p > 0 || sampler->top_p < 1) {
                next_token = sample_top_p(ctx, logits, sampler->vocab_size, sampler->top_p, sampler->probindex, coin);
            }
            else {
                next_token = sample_multinomial(ctx, logits, sampler->vocab_size, coin);
            }
        }
    }

    return next_token;
}


Nano_Session *llm_session_init(Nano_Context *ctx, wchar_t *prompt, uint32_t max_seq_len, int32_t is_thinking_enabled) {
    Nano_Session *session = (Nano_Session *)platform_calloc(1, sizeof(Nano_Session));
    Tokenizer *tokenizer = ctx->tokenizer;

    session->prompt = (wchar_t *)platform_calloc(max_seq_len + 1, sizeof(wchar_t));
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

    session->output_ids = (uint32_t *)platform_calloc(max_seq_len + 1, sizeof(uint32_t));
    session->output_count = 0;

    session->num_prompt_tokens = 0;
    uint32_t *prompt_tokens;
    if (ctx->llm->arch == LLM_ARCH_NANO) {
        prompt_tokens = encode_nano(tokenizer, session->prompt, &(session->num_prompt_tokens));
    }
    else if (ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3) {
        prompt_tokens = apply_qwen_chat_template(tokenizer, session->prompt, &(session->num_prompt_tokens), is_thinking_enabled);
    }
    else if (ctx->llm->arch == LLM_ARCH_QWEN35) {
        prompt_tokens = apply_qwen35_chat_template(tokenizer, session->prompt, &(session->num_prompt_tokens), is_thinking_enabled);
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
        // 释放上一步的输出文本：output_text 每生成一步都会重新分配，
        // 若不及时释放，将按词元数平方级泄漏（约34*N^2字节）
        // 注意：必须在确认本步会重新分配（pos < max_seq_len）后才释放，
        // 否则到达长度上限停止时 output_text 为 NULL，后续 on_llm_finished 会读空指针
        if (session->output_text) {
            free(session->output_text);
            session->output_text = NULL;
        }

        session->is_prefilling = (session->pos < session->num_prompt_tokens - 1) ? 1 : 0;

        session->next_token = generate_next_token(ctx, session->output_ids, session->pos, session->is_prefilling);

        // Pre-filling
        if (session->is_prefilling == 1) {
            if (ctx->llm->arch == LLM_ARCH_NANO) {
                session->output_text = decode_nano(ctx->tokenizer, session->output_ids, session->pos);
            }
            else if (ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3 || ctx->llm->arch == LLM_ARCH_QWEN35) {
                session->output_text = decode_bpe(ctx->tokenizer, session->output_ids, session->pos);
            }
            else {
                printf("Error: unknown LLM arch.\n");
                return LLM_STOPPED_WITH_ERROR;
            }
        }
        // Decoding
        else {
            session->output_ids[session->num_prompt_tokens + (session->output_count)++] = session->next_token;
            if (ctx->llm->arch == LLM_ARCH_NANO) {
                session->output_text = decode_nano(ctx->tokenizer, session->output_ids + session->num_prompt_tokens, session->output_count);
            }
            else if (ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3 || ctx->llm->arch == LLM_ARCH_QWEN35) {
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
            (session->is_prefilling == 0) &&
            (
                ((ctx->llm->arch == LLM_ARCH_QWEN2 || ctx->llm->arch == LLM_ARCH_QWEN3) &&
                 (session->next_token == 151643 || session->next_token == 151645)) ||
                (ctx->llm->arch == LLM_ARCH_QWEN35 &&
                 (session->next_token == 248044 || session->next_token == 248046))
            )) {
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
    Nano_Session *session = llm_session_init(ctx, prompt, max_seq_len, 1);
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
    uint32_t *output_ids = (uint32_t *)platform_calloc(max_seq_len, sizeof(uint32_t));

    LLM *llm = ctx->llm;
    Sampler *sampler = ctx->sampler;

    float *output_logits = (float*)platform_calloc(max_seq_len * sampler->vocab_size, sizeof(float));

    // 阶段1：预填充KVCache。
    //   内层循环对输入序列的每一个pos进行前向计算，填充每一层的第pos位置上的KVCache。
    //   一次遍历完整个序列后，只有第一层的KVCache完成了对整个序列的全局自注意力，但下游各层并没有。
    //   下游各层的KVCache依赖于上游各层完成全局自注意力（输出的中间值）。因此这个过程需要执行L次。
    for (uint32_t i = 0; i < llm->config.n_layer; i++) {
        for (uint32_t pos = 0; pos < max_seq_len; pos++) {
            (void)llm_forward(ctx, input_ids[pos], pos, max_seq_len, 0, llm, NULL);

        }
    }

    // 阶段2：KVCache全部填充后，再对每个pos执行完整的forward，获得每个pos上的logits
    for (uint32_t pos = 0; pos < max_seq_len; pos++) {
        float *logits = llm_forward(ctx, input_ids[pos], pos, max_seq_len, 0, llm, NULL);
        memcpy(output_logits + pos * sampler->vocab_size, logits, sampler->vocab_size *sizeof(float));
    }

    // 阶段3：对每个pos上的logits进行单独的采样
    for (uint32_t pos = 0; pos < max_seq_len; pos++) {
        float *logits = output_logits + pos * sampler->vocab_size;
        output_ids[pos] = sample_argmax(ctx, logits, sampler->vocab_size);
    }

    free(input_ids);
    free(output_logits);

    wcscpy(output_list, decode_nano(ctx->tokenizer, output_ids, max_seq_len));
}

