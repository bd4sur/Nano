#include <libwebsockets.h>
#include <string.h>
#include <wchar.h>
#include <locale.h>
#include <pthread.h>
#include "nano_infer.h"

struct session_data {
    wchar_t *input;
    uint32_t pos;
    uint32_t max_seq_len;
    uint32_t *output_ids;
    uint32_t output_count;
    uint32_t num_prompt_tokens;
    uint32_t *prompt_tokens;
    uint32_t next_token;
    char *output_buffer;
};

static Nano_Context ctx;

static uint32_t max_seq_len;

void load_model(char *model_path, float repetition_penalty, float temperature, float top_p, unsigned int top_k, unsigned long long rng_seed) {
    ctx.random_seed = rng_seed;
    ctx.llm = (LLM *)calloc(1, (sizeof(LLM)));
    ctx.lora = NULL; // (LoRA *)calloc(1, (sizeof(LoRA *)));
    ctx.tokenizer = (Tokenizer *)calloc(1, (sizeof(Tokenizer)));
    load_llm(ctx.llm, ctx.tokenizer, model_path);
    ctx.sampler = build_sampler(ctx.llm->config.vocab_size, repetition_penalty, temperature, top_p, top_k, ctx.random_seed);
}

void unload_model() {
    free_llm(ctx.llm, ctx.tokenizer);
    free_sampler(ctx.sampler);
}

static int callback_chat(struct lws *wsi, enum lws_callback_reasons reason,
                         void *user, void *in, size_t len)
{
    struct session_data *data = (struct session_data *)user;
    
    switch (reason) {
    case LWS_CALLBACK_ESTABLISHED:
        data->input = NULL;
        data->pos = 0;
        break;

    case LWS_CALLBACK_RECEIVE:
        char *utf8_input = (char*)in;
        size_t max_wlen = len + 1;
        wchar_t *w_input = malloc(max_wlen * sizeof(wchar_t));
        mbstowcs(w_input, utf8_input, max_wlen);

        uint32_t prompt_char_length = \
            ((char)(w_input[0]) - '0') * 10000 + \
            ((char)(w_input[1]) - '0') * 1000 + \
            ((char)(w_input[2]) - '0') * 100 + \
            ((char)(w_input[3]) - '0') * 10 + \
            ((char)(w_input[4]) - '0');

        data->input = (wchar_t *)calloc(prompt_char_length + 1, sizeof(wchar_t));
        for(int i = 0; i < prompt_char_length; i++) {
            data->input[i] = w_input[i+6];
        }
        data->input[prompt_char_length] = 0;
        free(w_input);

        data->max_seq_len = max_seq_len;
        data->output_ids = (uint32_t *)calloc(data->max_seq_len + 1, sizeof(uint32_t));
        data->output_count = 0;
    
        data->num_prompt_tokens = 0;
        data->output_buffer = (char *)calloc(data->max_seq_len * 4, sizeof(char));

        data->prompt_tokens = encode(ctx.tokenizer, data->input, &(data->num_prompt_tokens));
        for(int i = 0; i < data->num_prompt_tokens; i++) {
            data->output_ids[i] = data->prompt_tokens[i];
        }
    
        data->next_token = data->prompt_tokens[0];
        data->pos = 0;

        lws_callback_on_writable(wsi);

        break;

    case LWS_CALLBACK_SERVER_WRITEABLE:
        if (data->pos < data->max_seq_len) {

            int is_prefilling = (data->pos < data->num_prompt_tokens - 1) ? 1 : 0;

            data->next_token = generate_next_token(ctx, data->output_ids, data->pos, is_prefilling);

            if (is_prefilling == 0) {
                data->output_ids[data->num_prompt_tokens + (data->output_count)++] = data->next_token;
                wchar_t *output_text = decode(ctx.tokenizer, data->output_ids + data->num_prompt_tokens, data->output_count);

                size_t msg_len = wcstombs(data->output_buffer, output_text, data->max_seq_len * 4 * sizeof(char));
                free(output_text);

                unsigned char *pkt = malloc(LWS_PRE + msg_len);
                memcpy(pkt + LWS_PRE, data->output_buffer, msg_len);
                lws_write(wsi, pkt + LWS_PRE, msg_len, LWS_WRITE_TEXT);
                free(pkt);
            }
            data->pos++;

            if(data->next_token == 0 || data->next_token == 3) break;

            lws_callback_on_writable(wsi);
        }
        break;

    case LWS_CALLBACK_CLOSED:
        if(data->input) free(data->input);
        if(data->output_ids) free(data->output_ids);
        if(data->prompt_tokens) free(data->prompt_tokens);
        if(data->output_buffer) free(data->output_buffer);
        break;

    default:
        break;
    }
    return 0;
}

static struct lws_protocols protocols[] = {
    {"chat", callback_chat, sizeof(struct session_data), 0},
    {NULL, NULL, 0, 0}
};

void show_usage() {
    fprintf(stderr, "NanoLM Inference Engine - WebSocket Server for OpenWrt LuCI\n");
    fprintf(stderr, "  (c) BD4SUR 2024-11 2025-04\n");
    fprintf(stderr, "Usage:   nano_infer_ws_server <model_path> [options]\n");
    fprintf(stderr, "Example: nano_infer_ws_server model.bin -n 512 -P 8080\n");
    fprintf(stderr, "Options:\n");
    // fprintf(stderr, "  -l <string> path to LoRA module file, default null\n");
    fprintf(stderr, "  -r <float>  repetition penalty in (0,inf], default 1.11\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in (0,1) default 0.5\n");
    fprintf(stderr, "  -k <int>    k value in top-k sampling in [0, vocab_size) default 0 (no use)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 512. 0 = max_seq_len\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -P <int>    port number of WebSocket service\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {

    char *model_path = "/home/bd4sur/ai/nano_168m_625000_sft_947000.bin";

    unsigned int port = 8080;

    float repetition_penalty = 1.11;
    float temperature = 1.0;
    float top_p = 0.5;
    unsigned int top_k = 0;
    unsigned long long random_seed = (unsigned int)time(NULL);

    if(argc >= 2) { model_path = argv[1]; } else { show_usage(); }
    for(int i = 2; i < argc; i += 2) {
        // do some basic validation
        if (i + 1 >= argc) { show_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { show_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { show_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if      (argv[i][1] == 'r') { repetition_penalty = atof(argv[i + 1]); }
        else if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { top_p = atof(argv[i + 1]); }
        else if (argv[i][1] == 'k') { top_k = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { max_seq_len = atoi(argv[i + 1]); }
        else if (argv[i][1] == 's') { random_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'P') { port = atoi(argv[i + 1]); }

        else { show_usage(); }
    }

    load_model(model_path, repetition_penalty, temperature, top_p, top_k, random_seed);

    if(!setlocale(LC_CTYPE, "")) return -1;

    struct lws_context_creation_info info;
    memset(&info, 0, sizeof(info));
    
    info.port = port;
    info.protocols = protocols;
    info.gid = -1;
    info.uid = -1;

    struct lws_context *context = lws_create_context(&info);
    if (!context) return -1;
    
    while (1) {
        lws_service(context, 50);
    }
    
    lws_context_destroy(context);

    unload_model();

    return 0;
}