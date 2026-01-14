//
// Nano Language Model - Inference Engine written in C
//
//   BD4SUR 2026-01
//
//   从FP32模型文件生成Q4K量化模型文件
//
// Usage:
//   1) 修改 INPUT_FILE_PATH 和 OUTPUT_FILE_NAME 常量为所需的输入输出文件路径。
//   2) 将本文件移动到 infer 目录下。
//   3) 编译并运行本文件。
//      cc -O3 -march=native -ffast-math -Wall -fopenmp export_q4k.c platform_linux.c utils.c prompt.c tokenizer.c tensor.c infer.c -o export_q4k -lm -fopenmp
//      ./export_q4k
//


#include "infer.h"



#define OUTPUT_FILE_NAME "qwen3-0b6-q4k.bin"
#define INPUT_FILE_PATH "/mnt/d/Desktop/LLM/Nano/nano-168m.bin"




void pack_q4k_model_file(LLM *llm, uint8_t *buffer, uint64_t header_byte_length, uint64_t tokenizer_field_bytes) {

    uint8_t *header_ptr = buffer;
    uint8_t *tokenzier_ptr = buffer + header_byte_length;
    uint8_t *ptr = tokenzier_ptr + tokenizer_field_bytes;

    
    // 输出缓冲区
    uint8_t *output_buffer = (uint8_t*)calloc(3999999999, sizeof(uint8_t)); // TODO 动态分配更合适的大小
    uint64_t offset = 0;

    // header
    memcpy(output_buffer + offset, header_ptr, sizeof(uint8_t) * header_byte_length);
    offset += sizeof(uint8_t) * header_byte_length;

    // 修改quant_type字段
    uint32_t qt = QUANT_TYPE_Q4K;
    memcpy(output_buffer + 15 * sizeof(uint32_t), &qt, sizeof(uint32_t));

    // tokenizer
    memcpy(output_buffer + offset, tokenzier_ptr, sizeof(uint8_t) * tokenizer_field_bytes);
    offset += sizeof(uint8_t) * tokenizer_field_bytes;

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


    // 直接复制前面无需量化的部分

    // rms_norm_attn
    memcpy(output_buffer + offset, ptr, sizeof(float) * (n_layer * cfg->n_embd));
    ptr    += sizeof(float) * (n_layer * cfg->n_embd);
    offset += sizeof(float) * (n_layer * cfg->n_embd);

    // rms_norm_ffn
    memcpy(output_buffer + offset, ptr, sizeof(float) * (n_layer * cfg->n_embd));
    ptr    += sizeof(float) * (n_layer * cfg->n_embd);
    offset += sizeof(float) * (n_layer * cfg->n_embd);

    // rms_norm_final
    memcpy(output_buffer + offset, ptr, sizeof(float) * (cfg->n_embd));
    ptr    += sizeof(float) * (cfg->n_embd);
    offset += sizeof(float) * (cfg->n_embd);

    assert(llm->quant_type == QUANT_TYPE_F32);

    // 加载要量化的FP32参数

    float* fptr = (float*) ptr;

    w->token_embedding = fptr; fptr += cfg->vocab_size * cfg->n_embd;

    w->wq = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->wq->tensor_f32 = fptr;  fptr += n_layer * (cfg->n_head * head_size) * cfg->n_embd;
    w->wk = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->wk->tensor_f32 = fptr;  fptr += n_layer * (cfg->n_kv_head * head_size) * cfg->n_embd;
    w->wv = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->wv->tensor_f32 = fptr;  fptr += n_layer * (cfg->n_kv_head * head_size) * cfg->n_embd;
    w->wo = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->wo->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * (cfg->n_head * head_size);

    w->w1 = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->w1->tensor_f32 = fptr;  fptr += n_layer * cfg->n_hidden * cfg->n_embd;
    w->w2 = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->w2->tensor_f32 = fptr;  fptr += n_layer * cfg->n_embd * cfg->n_hidden;
    w->w3 = (Typed_Tensor *)calloc_dev(1, sizeof(Typed_Tensor));  w->w3->tensor_f32 = fptr;  fptr += n_layer * cfg->n_hidden * cfg->n_embd;

    // 将fp32转成Q4k
    // TODO 逐层

    Q4k_Tensor *q_tokens_q4k = quantize_tensor_q4k(w->token_embedding, 2, (uint32_t[]){cfg->vocab_size, cfg->n_embd});

    Q4k_Tensor *wq_q4k = quantize_tensor_q4k(w->wq->tensor_f32, 3, (uint32_t[]){n_layer, cfg->n_head * head_size, cfg->n_embd});
    Q4k_Tensor *wk_q4k = quantize_tensor_q4k(w->wk->tensor_f32, 3, (uint32_t[]){n_layer, cfg->n_kv_head * head_size, cfg->n_embd});
    Q4k_Tensor *wv_q4k = quantize_tensor_q4k(w->wv->tensor_f32, 3, (uint32_t[]){n_layer, cfg->n_kv_head * head_size, cfg->n_embd});
    Q4k_Tensor *wo_q4k = quantize_tensor_q4k(w->wo->tensor_f32, 3, (uint32_t[]){n_layer, cfg->n_embd, cfg->n_head * head_size});

    Q4k_Tensor *w1_q4k = quantize_tensor_q4k(w->w1->tensor_f32, 3, (uint32_t[]){n_layer, cfg->n_hidden, cfg->n_embd});
    Q4k_Tensor *w2_q4k = quantize_tensor_q4k(w->w2->tensor_f32, 3, (uint32_t[]){n_layer, cfg->n_embd, cfg->n_hidden});
    Q4k_Tensor *w3_q4k = quantize_tensor_q4k(w->w3->tensor_f32, 3, (uint32_t[]){n_layer, cfg->n_hidden, cfg->n_embd});

    // 将Q4k写入输出缓冲区

    uint64_t q4k_tensor_bytes = 0;

    uint8_t *q_tokens_q4k_tensor_stream = pack_q4k_tensor(q_tokens_q4k);
    q4k_tensor_bytes = bytes_num_of_q4k_tensor(q_tokens_q4k);
    printf("q_tokens_q4k_tensor_bytes: %ld\n", (uint64_t)q4k_tensor_bytes);
    memcpy(output_buffer + offset, q_tokens_q4k_tensor_stream, q4k_tensor_bytes);
    offset += q4k_tensor_bytes;
    free(q_tokens_q4k_tensor_stream);

    uint8_t *wq_q4k_tensor_stream = pack_q4k_tensor(wq_q4k);
    q4k_tensor_bytes = bytes_num_of_q4k_tensor(wq_q4k);
    memcpy(output_buffer + offset, wq_q4k_tensor_stream, q4k_tensor_bytes);
    offset += q4k_tensor_bytes;
    free(wq_q4k_tensor_stream);

    uint8_t *wk_q4k_tensor_stream = pack_q4k_tensor(wk_q4k);
    q4k_tensor_bytes = bytes_num_of_q4k_tensor(wk_q4k);
    memcpy(output_buffer + offset, wk_q4k_tensor_stream, q4k_tensor_bytes);
    offset += q4k_tensor_bytes;
    free(wk_q4k_tensor_stream);

    uint8_t *wv_q4k_tensor_stream = pack_q4k_tensor(wv_q4k);
    q4k_tensor_bytes = bytes_num_of_q4k_tensor(wv_q4k);
    memcpy(output_buffer + offset, wv_q4k_tensor_stream, q4k_tensor_bytes);
    offset += q4k_tensor_bytes;
    free(wv_q4k_tensor_stream);

    uint8_t *wo_q4k_tensor_stream = pack_q4k_tensor(wo_q4k);
    q4k_tensor_bytes = bytes_num_of_q4k_tensor(wo_q4k);
    memcpy(output_buffer + offset, wo_q4k_tensor_stream, q4k_tensor_bytes);
    offset += q4k_tensor_bytes;
    free(wo_q4k_tensor_stream);

    uint8_t *w1_q4k_tensor_stream = pack_q4k_tensor(w1_q4k);
    q4k_tensor_bytes = bytes_num_of_q4k_tensor(w1_q4k);
    memcpy(output_buffer + offset, w1_q4k_tensor_stream, q4k_tensor_bytes);
    offset += q4k_tensor_bytes;
    free(w1_q4k_tensor_stream);

    uint8_t *w2_q4k_tensor_stream = pack_q4k_tensor(w2_q4k);
    q4k_tensor_bytes = bytes_num_of_q4k_tensor(w2_q4k);
    memcpy(output_buffer + offset, w2_q4k_tensor_stream, q4k_tensor_bytes);
    offset += q4k_tensor_bytes;
    free(w2_q4k_tensor_stream);

    uint8_t *w3_q4k_tensor_stream = pack_q4k_tensor(w3_q4k);
    q4k_tensor_bytes = bytes_num_of_q4k_tensor(w3_q4k);
    memcpy(output_buffer + offset, w3_q4k_tensor_stream, q4k_tensor_bytes);
    offset += q4k_tensor_bytes;
    free(w3_q4k_tensor_stream);

    // 继续复制后面无需量化的部分

    ptr = (uint8_t*) fptr;

    if (llm->arch == LLM_ARCH_QWEN2) {
        // w->bq = fptr;         fptr += n_layer * (cfg->n_head * head_size);
        // w->bk = fptr;         fptr += n_layer * (cfg->n_kv_head * head_size);
        // w->bv = fptr;         fptr += n_layer * (cfg->n_kv_head * head_size);
    }
    else if (llm->arch == LLM_ARCH_QWEN3) {
        // w->q_norm = fptr;     fptr += n_layer * head_size;
        // w->k_norm = fptr;     fptr += n_layer * head_size;

        // q_norm
        memcpy(output_buffer + offset, ptr, sizeof(float) * (n_layer * head_size));
        ptr    += sizeof(float) * (n_layer * head_size);
        offset += sizeof(float) * (n_layer * head_size);

        // k_norm
        memcpy(output_buffer + offset, ptr, sizeof(float) * (n_layer * head_size));
        ptr    += sizeof(float) * (n_layer * head_size);
        offset += sizeof(float) * (n_layer * head_size);
    }

    if (llm->arch == LLM_ARCH_NANO || llm->arch == LLM_ARCH_QWEN2) {
        // w->freq_cis_real = fptr;  fptr += cfg->block_size * head_size / 2;
        // w->freq_cis_imag = fptr;  fptr += cfg->block_size * head_size / 2;

        // freq_cis_real
        memcpy(output_buffer + offset, ptr, sizeof(float) * (cfg->block_size * head_size / 2));
        ptr    += sizeof(float) * (cfg->block_size * head_size / 2);
        offset += sizeof(float) * (cfg->block_size * head_size / 2);

        // freq_cis_imag
        memcpy(output_buffer + offset, ptr, sizeof(float) * (cfg->block_size * head_size / 2));
        ptr    += sizeof(float) * (cfg->block_size * head_size / 2);
        offset += sizeof(float) * (cfg->block_size * head_size / 2);
    }

    printf("字节数：%d\n", (int)offset);

    // 写入文件

    FILE *fp = fopen(OUTPUT_FILE_NAME, "wb");
    if (!fp) {
        perror("fopen");
        return;
    }

    size_t written = fwrite(output_buffer, sizeof(uint8_t), offset, fp);
    if (written != offset) {
        perror("fwrite");
        fclose(fp);
        return;
    }

    fclose(fp);

}





void parse_model_file_for_quant(uint8_t* buffer, LLM *llm, Tokenizer *tk) {

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

    // void *param_ptr = (float*)((uint8_t*)tokenzier_ptr + tokenizer_field_bytes);
    pack_q4k_model_file(llm, buffer, header_byte_length, tokenizer_field_bytes);

}








void load_llm_for_quant(LLM *llm, Tokenizer *tk, char *model_path, uint32_t max_seq_len) {
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

    parse_model_file_for_quant(buffer, llm, tk);

#else

    FILE *file = fopen(model_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open model file %s\n", model_path); exit(EXIT_FAILURE); }
    uint8_t *buffer = (uint8_t *)calloc(llm->file_size + 1, sizeof(uint8_t));
    if (!buffer)  { fprintf(stderr, "malloc failed.\n"); exit(EXIT_FAILURE); }
    if(fread(buffer, sizeof(uint8_t), llm->file_size, file) != llm->file_size) { exit(EXIT_FAILURE); }
    llm->buffer = buffer;

    parse_model_file_for_quant(buffer, llm, tk);

    fclose(file);
#endif
}




Nano_Context *llm_context_init_for_quant(char *model_path, char *lora_path, uint32_t max_seq_len, float repetition_penalty, float temperature, float top_p, uint32_t top_k, uint64_t random_seed) {
    Nano_Context *ctx = (Nano_Context*)calloc(1, sizeof(Nano_Context));
    ctx->max_seq_len = max_seq_len;
    ctx->random_seed = random_seed;
    ctx->llm = (LLM *)calloc(1, (sizeof(LLM)));
    ctx->tokenizer = (Tokenizer *)calloc(1, (sizeof(Tokenizer)));
    load_llm_for_quant(ctx->llm, ctx->tokenizer, model_path, max_seq_len);
    ctx->sampler = build_sampler(ctx->llm->config.vocab_size, repetition_penalty, temperature, top_p, top_k, ctx->random_seed);
    ctx->lora = (lora_path) ? load_lora(ctx->llm, lora_path) : NULL;
    return ctx;
}



int main(void) {
/*
    uint64_t seed = 39;

    const uint32_t d = 8;
    const uint32_t n = 768;

    float *w_f32 = (float*)calloc(d*n, sizeof(float));
    float *x_f32 = (float*)calloc(n, sizeof(float));
    float *y_f32 = (float*)calloc(d, sizeof(float));

    for (uint32_t i = 0; i < d*n; i++) w_f32[i] = random_f32(&seed) * ((random_f32(&seed) > 0.5) ? (+1) : (-1));
    for (uint32_t i = 0; i <  n ; i++) x_f32[i] = random_f32(&seed) * ((random_f32(&seed) > 0.5) ? (+1) : (-1));
    for (uint32_t i = 0; i <  d ; i++) y_f32[i] = 0.0f;

    // FP32原值乘
    matmul(y_f32, x_f32, w_f32, n, d);
    printf("原值乘 = ");
    for (uint32_t i = 0; i < d; i++) {
        printf("%.3f  ", y_f32[i]);
    }
    printf("\n");

    // 量化后直接乘
    uint32_t w_shape[2] = {d, n};
    Q4k_Tensor *w = quantize_tensor_q4k(w_f32, 2, w_shape);
    uint32_t x_shape[1] = {n};
    Q4k_Tensor *x = quantize_tensor_q4k(x_f32, 1, x_shape);

    uint8_t *w_stream = pack_q4k_tensor(w);
    uint8_t *x_stream = pack_q4k_tensor(x);

    uint64_t frame_length = 0;
    Q4k_Tensor *w_up = unpack_q4k_tensor(w_stream, &frame_length);
    Q4k_Tensor *x_up = unpack_q4k_tensor(x_stream, &frame_length);

    matmul_q4k(y_f32, x_up, w_up, 0);
    printf("量化乘 = ");
    for (uint32_t i = 0; i < d; i++) {
        printf("%.3f  ", y_f32[i]);
    }
    printf("\n");

    // 量化后再反量化回FP32再乘
    uint32_t ndim = 2;
    uint32_t shape_tmp[2];
    float *ww_f32 = (float*)calloc(d*n, sizeof(float));
    float *xx_f32 = (float*)calloc(n,   sizeof(float));
    dequantize_tensor_q4k(w_up, ww_f32, &ndim, shape_tmp);
    dequantize_tensor_q4k(x_up, xx_f32, &ndim, shape_tmp);
    matmul(y_f32, xx_f32, ww_f32, n, d);
    printf("反量乘 = ");
    for (uint32_t i = 0; i < d; i++) {
        printf("%.3f  ", y_f32[i]);
    }
    printf("\n");

*/
    // 打开bin模型
    Nano_Context *g_llm_ctx = llm_context_init_for_quant(INPUT_FILE_PATH, NULL, 256, 1.0, 0.7, 0.8, 20, 39);


    // free(all);

    return 0;
}