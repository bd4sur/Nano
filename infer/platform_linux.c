#include "platform.h"

#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <locale.h>
#include <sys/stat.h>
#include <errno.h>

void sleep_in_ms(uint32_t ms) {
    usleep(ms * 1000);
}

uint64_t get_timestamp_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// 优雅关机
int32_t graceful_shutdown() {
    // 同步所有文件系统数据
    sync();
    // 等待同步完成
    sleep(2);
    // 执行关机
    if (system("sudo shutdown -h now") == -1) {
        perror("关机失败");
        return -1;
    }
    return 0;
}


// 辅助函数：在 wchar_t 层面对字符串进行 JSON 转义（\n, ", \）
wchar_t* escape_wchar_string(const wchar_t* wstr) {
    if (!wstr) return NULL;

    size_t len = wcslen(wstr);
    // 最坏情况：每个字符扩展为 2 个宽字符（如 L'\\' → L'\\'L'\\'）
    size_t max_escaped_len = len * 2 + 1;
    wchar_t* escaped = (wchar_t*)calloc(max_escaped_len, sizeof(wchar_t));
    if (!escaped) return NULL;

    wchar_t* out = escaped;
    const wchar_t* p = wstr;

    while (*p) {
        if (*p == L'\n') {
            *out++ = L'\\';
            *out++ = L'n';
        } else if (*p == L'"') {
            *out++ = L'\\';
            *out++ = L'"';
        } else if (*p == L'\\') {
            *out++ = L'\\';
            *out++ = L'\\';
        } else {
            *out++ = *p;
        }
        p++;
    }
    *out = L'\0';
    return escaped;
}

// 将转义后的 wchar_t* 用 wcstombs 转为 char*（依赖当前 locale）
char* wchar_to_utf8(const wchar_t* wstr) {
    if (!wstr) return NULL;

    size_t len = wcslen(wstr);
    size_t buf_size = len * MB_CUR_MAX + 1;
    char* mbstr = (char*)calloc(buf_size, sizeof(char));
    if (!mbstr) return NULL;

    size_t result = _wcstombs(mbstr, wstr, buf_size);
    if (result == (size_t)-1) {
        free(mbstr);
        return NULL;
    }

    return mbstr;
}

// 将 prompt 和 response 转义、转换、写入 log.jsonl
int32_t write_chat_log(char *filepath, uint64_t timestamp, wchar_t* prompt, wchar_t* response) {

    // 第一步：在 wchar_t 层面转义
    wchar_t* escaped_prompt_w = escape_wchar_string(prompt);
    wchar_t* escaped_response_w = escape_wchar_string(response);

    if (!escaped_prompt_w || !escaped_response_w) {
        free(escaped_prompt_w);
        free(escaped_response_w);
        return -1;
    }

    // 第二步：用 wcstombs 转为 char*（UTF-8）
    char* escaped_prompt = wchar_to_utf8(escaped_prompt_w);
    char* escaped_response = wchar_to_utf8(escaped_response_w);

    free(escaped_prompt_w);
    free(escaped_response_w);

    if (!escaped_prompt || !escaped_response) {
        free(escaped_prompt);
        free(escaped_response);
        return -1;
    }

    // 第三步：写入 JSONL 文件
    FILE* fp = fopen(filepath, "a");
    if (!fp) {
        free(escaped_prompt);
        free(escaped_response);
        return -1;
    }

    fprintf(fp, "{\"timestamp\": %ld, \"prompt\": \"%s\", \"response\": \"%s\"}\n",
            timestamp, escaped_prompt, escaped_response);

    fclose(fp);
    free(escaped_prompt);
    free(escaped_response);
    return 0;
}

/**
 * 读取文件内容（UTF-8），并转换为 wchar_t* 字符串
 *
 * @param filename 文件名
 * @return 成功时返回动态分配的 wchar_t*（以 L'\0' 结尾），失败返回 NULL。
 *         调用者需用 free() 释放返回值。
 */
wchar_t* read_file_to_wchar(char* filename) {
    if (!filename) return NULL;

    // 2. 打开文件（当前工作目录）
    FILE* fp = fopen(filename, "rb"); // 用二进制模式避免换行转换
    if (!fp) {
        return NULL;
    }

    // 3. 获取文件大小（可选，用于高效分配）
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    size_t size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }

    // 4. 读取全部内容到 char 缓冲区（+1 保证可加 '\0'）
    char* buffer = (char*)calloc(size + 1, sizeof(char));
    if (!buffer) {
        fclose(fp);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, size, fp);
    fclose(fp);

    if ((size_t)bytes_read != size) {
        free(buffer);
        return NULL;
    }
    buffer[size] = '\0'; // 确保以 null 结尾（UTF-8 是 null-safe 的）

    // 5. 计算所需 wchar_t 数量
    size_t wlen = size;

    // 6. 分配 wchar_t 缓冲区
    wchar_t* wstr = (wchar_t*)calloc((wlen + 1), sizeof(wchar_t));
    if (!wstr) {
        free(buffer);
        return NULL;
    }

    // 7. 执行实际转换
    (void)_mbstowcs(wstr, buffer, wlen + 1);
    free(buffer);

    return wstr; // 调用者负责 free()
}
