#include "nano_infer.h"

void show_usage() {
    fprintf(stderr, "NanoLM - Inference Engine\n");
    fprintf(stderr, "  BD4SUR 2024-11\n");
    fprintf(stderr, "  forked from github.com/karpathy/llama2.c\n\n");
    fprintf(stderr, "Usage:   nano_infer_shell <model_path> [options]\n");
    fprintf(stderr, "Example: nano_infer_shell model.bin -n 256 -i \"我是复读机，你是什么？\"\n");
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


int main(int argc, char **argv) {
    if(!setlocale(LC_CTYPE, "")) {
        fprintf(stderr, "Can't set the specified locale! Check LANG, LC_CTYPE, LC_ALL.\n");
        return -1;
    }

    char *MODEL_PATH = NULL;
    char *LORA_PATH  = NULL;

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
            prompt = apply_template_to_str(argv[i + 1], max_seq_len);
        }
        else if (argv[i][1] == 'g') {
            prompt = (wchar_t *)argv[i + 1];
        }

        else { show_usage(); }
    }
    // printf("Prompt > %ls\n", prompt);


    Nano_Context ctx;

    ctx.random_seed = random_seed;
    ctx.llm = (LLM *)calloc(1, (sizeof(LLM)));
    ctx.lora = NULL; // (LoRA *)calloc(1, (sizeof(LoRA *)));
    ctx.tokenizer = (Tokenizer *)calloc(1, (sizeof(Tokenizer)));

    load_llm(ctx.llm, ctx.tokenizer, MODEL_PATH);

    ctx.sampler = build_sampler(ctx.llm->config.vocab_size, rep_pnty, temperature, top_p, top_k, ctx.random_seed);

    if(NULL != LORA_PATH) ctx.lora = load_lora(ctx.llm, LORA_PATH);

    printf("\x1b[32;1mNano:\x1b[0m ");
    last_output_length = 0;
    generate(ctx, prompt, max_seq_len, typewriter, report);
/*
    while(1) {
        printf("\x1b[32;1mHomo:\x1b[0m ");
        char prompt_str[1024] = "";
        char line_buffer[1024] = "";
        while (fgets(line_buffer, sizeof(line_buffer), stdin) != NULL) {
            if (line_buffer[0] == '\n') {
                break;
            }
            strcat(prompt_str, line_buffer);
        }
        prompt_str[strlen(prompt_str)-1] = 0; // 去掉最后一个换行符
        // scanf("%s", prompt_str);

        prompt = apply_template_to_str(prompt_str, max_seq_len);
        // printf("Prompt = %ls\n", prompt);

        printf("\x1b[34;1mNano:\x1b[0m ");
        last_output_length = 0;
        generate(ctx, prompt, max_seq_len, typewriter, report);

        prompt_str[0] = 0;
    }
*/
    if(NULL != LORA_PATH) free_lora(ctx.llm, ctx.lora);

    free(prompt);
    free_llm(ctx.llm, ctx.tokenizer);
    free_sampler(ctx.sampler);

    return 0;
}
