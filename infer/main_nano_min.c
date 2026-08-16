//
// main_nano_min.c - nano_min 引擎的最小化终端交互验证程序
//
//   参照 main_cli.c，保持最简：单行输入 -> 套提示词模板 -> 流式输出。
//   按模型架构自动分派：
//     - NANO：宽字符输入，模板 <|instruct_mark|>...<|response_mark|>，结束符 0/3；
//     - QWEN2/3：UTF-8 字节输入，ChatML 模板（同 tokenizer.c apply_qwen_chat_template），
//       结束符 151643/151645。
//
//   构建（WSL2，在 infer 目录下）：
//     gcc -DNANO_CLI -O2 -Wall -o bin/nano_min main_nano_min.c nano_min.c platform_linux.c utils.c -lm -pthread
//
//   运行：
//     ./bin/nano_min [model_path] [work_file_path] [max_seq_len] [temperature] [top_p]
//   环境变量 NM_PRINT_IDS=1 时打印生成的 token id（用于与原引擎逐 token 对比验证）。
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <locale.h>

#include "platform.h"
#include "nano_min.h"

#define NM_MAX_INPUT (2048)

static char *MODEL_PATH = "/home/bd4sur/ai/_model/Nano/nano-168m-q80.bin";
static char *WORK_PATH  = "./nano_min_work.tmp";

// 读取进程当前 RSS 高水位（KiB），用于验证 RAM 占用（仅 Linux 验证用）
static long read_vmrss_hwm_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) { sscanf(line + 6, "%ld", &kb); break; }
    }
    fclose(f);
    return kb;
}

int main(int argc, char **argv) {
    if (!setlocale(LC_CTYPE, "")) return -1;

    if (argc > 1) MODEL_PATH = argv[1];
    if (argc > 2) WORK_PATH  = argv[2];
    uint32_t max_seq_len = (argc > 3) ? (uint32_t)atoi(argv[3]) : 512;
    float temperature    = (argc > 4) ? (float)atof(argv[4]) : 0.7f;
    float top_p          = (argc > 5) ? (float)atof(argv[5]) : 0.8f;
    int print_ids = (getenv("NM_PRINT_IDS") != NULL);
    uint32_t max_gen = (getenv("NM_MAX_GEN") != NULL) ? (uint32_t)atoi(getenv("NM_MAX_GEN")) : 0; // 0=不限

    printf("Nano Minimal-Memory Inference Engine (filesystem-as-memory)\n\n");
    printf("Using model: %s\n", MODEL_PATH);

    fs_init(); // 初始化文件系统（ESP32 上挂载 SD 卡；Linux 为空操作）

    NM_Engine *e = nm_open(MODEL_PATH, WORK_PATH, max_seq_len);
    nm_set_sampler(e, 1.0f, temperature, top_p, get_timestamp_in_ms());
    nm_print_info(e);

    uint32_t arch = nm_get_arch(e);

    printf("\n单行输入问题并回车；直接回车使用默认问题；Ctrl+D 退出。\n");

    uint32_t *ids = (uint32_t *)malloc((max_seq_len + 1) * sizeof(uint32_t));

    while (1) {
        char line[NM_MAX_INPUT];

        printf("\n\x1b[32;1mHomo:\x1b[0m ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;

        // 去掉行尾换行符
        size_t ll = strlen(line);
        while (ll > 0 && (line[ll - 1] == '\n' || line[ll - 1] == '\r')) line[--ll] = 0;

        uint32_t n_prompt = 0;

        if (arch == NM_ARCH_NANO) {
            wchar_t wline[NM_MAX_INPUT];
            wchar_t prompt[NM_MAX_INPUT * 2];
            mbstowcs(wline, line, NM_MAX_INPUT - 1);
            if (wcslen(wline) == 0) wcscpy(wline, L"请你自我介绍。");
            // Nano 架构提示词模板（同 main_cli.c）
            swprintf(prompt, sizeof(prompt) / sizeof(wchar_t), L"<|instruct_mark|>%ls<|response_mark|>", wline);
            n_prompt = nm_encode(e, prompt, ids, max_seq_len);
        }
        else {
            if (ll == 0) strcpy(line, "请你自我介绍。");
            // ChatML 模板（同 tokenizer.c apply_qwen_chat_template，enable_thinking=1）
            uint32_t n_user = 0;
            uint32_t user_ids[NM_MAX_INPUT];
            n_user = nm_encode_bpe(e, line, user_ids, NM_MAX_INPUT);
            ids[n_prompt++] = 151644; // <|im_start|>
            ids[n_prompt++] = 872;    // user
            ids[n_prompt++] = 198;    // \n
            for (uint32_t i = 0; i < n_user && n_prompt + 8 < max_seq_len; i++) ids[n_prompt++] = user_ids[i];
            ids[n_prompt++] = 151645; // <|im_end|>
            ids[n_prompt++] = 198;    // \n
            ids[n_prompt++] = 151644; // <|im_start|>
            ids[n_prompt++] = 77091;  // assistant
            ids[n_prompt++] = 198;    // \n
        }

        if (n_prompt == 0) { printf("(无法分词)\n"); continue; }
        if (print_ids) {
            printf("prompt_ids:");
            for (uint32_t i = 0; i < n_prompt; i++) printf(" %u", ids[i]);
            printf("\n");
        }

        printf("\n\x1b[34;1mNano:\x1b[0m ");
        fflush(stdout);

        uint64_t t0 = get_timestamp_in_ms();

        // Pre-filling：逐 token 前向，填充 KV-Cache
        uint32_t pos = 0;
        uint32_t token = ids[0];
        for (pos = 0; pos + 1 < n_prompt; pos++) {
            nm_forward(e, token, pos);
            token = ids[pos + 1];
        }

        // Decoding：采样并流式输出，遇结束符停止
        uint32_t n_gen = 0;
        while (pos < max_seq_len) {
            if (max_gen > 0 && n_gen >= max_gen) break;
            nm_forward(e, token, pos);
            if (print_ids) nm_debug_top2(e);
            uint32_t next = nm_sample(e, ids, pos + 1);
            pos++;
            if (nm_is_eos(e, next)) break;
            ids[pos] = next;
            token = next;
            n_gen++;
            if (print_ids) { printf("[%u]", next); continue; }
            if (arch == NM_ARCH_NANO) {
                printf("%ls", nm_token_str(e, next));
            }
            else {
                char tok[300];
                printf("%s", nm_bpe_token_str(e, next, tok, sizeof(tok)));
            }
            fflush(stdout);
        }

        double secs = (double)(get_timestamp_in_ms() - t0) / 1000.0;
        printf("\n\n[%u tokens, %.1f s, %.2f tok/s | 引擎动态RAM %.1f KB | 进程RSS高水位 %ld KB]\n",
               n_gen, secs, secs > 0 ? n_gen / secs : 0.0,
               (double)nm_ram_bytes(e) / 1024.0, read_vmrss_hwm_kb());
    }

    free(ids);
    nm_close(e);
    printf("\nBye.\n");
    return 0;
}
