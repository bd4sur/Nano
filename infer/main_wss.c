#include <libwebsockets.h>
#include <string.h>
#include <wchar.h>
#include <locale.h>
#include <pthread.h>
#include "infer.h"

static Nano_Context *g_llm_ctx;

static Nano_Session *g_llm_session;


int32_t on_prefilling(Nano_Session *session) {
    // printf("Pre-filling...\n");
    return LLM_RUNNING_IN_PREFILLING;
}

int32_t on_decoding(Nano_Session *session) {
    return LLM_RUNNING_IN_DECODING;
}

int32_t on_finished(Nano_Session *session) {
    printf("\nTPS = %f\n", session->tps);
    return LLM_STOPPED_NORMALLY;
}

static int callback_chat(struct lws *wsi, enum lws_callback_reasons reason,
                         void *user, void *in, size_t len)
{

    switch (reason) {
    case LWS_CALLBACK_ESTABLISHED:
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

        wchar_t *prompt = (wchar_t *)calloc(prompt_char_length + 1, sizeof(wchar_t));
        for(int i = 0; i < prompt_char_length; i++) {
            prompt[i] = w_input[i+6];
        }
        prompt[prompt_char_length] = 0;
        free(w_input);

        printf("Received prompt: %ls\n", prompt);
        g_llm_session = llm_session_init(g_llm_ctx, prompt, 32768, 1);

        lws_callback_on_writable(wsi);

        break;

    case LWS_CALLBACK_SERVER_WRITEABLE:

        int32_t status = llm_session_step(g_llm_ctx, g_llm_session);
        g_llm_session->tps = (g_llm_session->pos - 1) / (double)(get_timestamp_in_ms() - g_llm_session->t_0) * 1000;
        if (status == LLM_RUNNING_IN_PREFILLING) {
            int32_t callback_flag = on_prefilling(g_llm_session);
            // 外部被动中止
            if (callback_flag == LLM_STOPPED_IN_PREFILLING) {
                status = callback_flag;
            }
            else {
                lws_callback_on_writable(wsi);
            }
        }
        else if (status == LLM_RUNNING_IN_DECODING) {
            int32_t callback_flag = on_decoding(g_llm_session);

            printf("%s", g_llm_ctx->tokenizer->vocab[g_llm_session->next_token]);
            fflush(stdout);

            char *output_buffer = calloc(g_llm_session->max_seq_len * 4, sizeof(char));
            size_t msg_len = wcstombs(output_buffer, g_llm_session->output_text, g_llm_session->max_seq_len * 4 * sizeof(char));
            free(g_llm_session->output_text);

            unsigned char *pkt = malloc(LWS_PRE + msg_len);
            memcpy(pkt + LWS_PRE, output_buffer, msg_len);
            lws_write(wsi, pkt + LWS_PRE, msg_len, LWS_WRITE_TEXT);
            free(pkt);

            // 外部被动中止
            if (callback_flag == LLM_STOPPED_IN_DECODING) {
                status = callback_flag;
            }
            else {
                lws_callback_on_writable(wsi);
            }
        }
        else if (status == LLM_STOPPED_NORMALLY) {
            g_llm_session->t_1 = get_timestamp_in_ms();
            g_llm_session->tps = (g_llm_session->pos - 1) / (double)(g_llm_session->t_1 - g_llm_session->t_0) * 1000;
            status = on_finished(g_llm_session);
        }
        else {
            status = LLM_STOPPED_WITH_ERROR;
        }

        break;

    case LWS_CALLBACK_CLOSED:
        llm_session_free(g_llm_session);
        break;

    default:
        break;
    }
    return 0;
}

static struct lws_protocols protocols[] = {
    {"chat", callback_chat, sizeof(Nano_Session), 0},
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
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -P <int>    port number of WebSocket service\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {

    char *model_path = "/home/bd4sur/ai/nano_168m_625000_sft_947000.bin";

    unsigned int port = 8080;

    float repetition_penalty = 1.0f;
    float temperature = 0.6f;
    float top_p = 0.95f;
    unsigned int top_k = 20;
    uint64_t random_seed = get_timestamp_in_ms();

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
        else if (argv[i][1] == 's') { random_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'P') { port = atoi(argv[i + 1]); }

        else { show_usage(); }
    }

    g_llm_ctx = llm_context_init(model_path, NULL, 32768, repetition_penalty, temperature, top_p, top_k, random_seed);

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

    llm_context_free(g_llm_ctx);

#ifdef MATMUL_PTHREAD
    matmul_pthread_cleanup();
#endif

    return 0;
}