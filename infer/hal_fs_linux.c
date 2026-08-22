// hal_fs Linux 实现（PC / 树莓派等普通 Linux 系统）：直接使用宿主原生文件系统。
//
// 路径语义说明：业务路径已统一经 PLATFORM_ROOT_DIR 前缀拼接（见 platform.h），
// 本实现将拼接后的路径按原样传递给宿主文件系统。电子书/音乐盒等以 "/" 为根
// 枚举文件的应用，实际枚举的是 PLATFORM_ROOT_DIR 下的内容。

#include "platform.h"
#include "hal_fs.h"
#include "hal_ram.h" // list_files / platform_read_file_to_buffer 内部经 platform_malloc 分配缓冲

#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <dirent.h>
#include <sys/stat.h>

// Linux 使用宿主原生文件系统，无需初始化
int32_t fs_init() {
    return 0;
}

int32_t platform_read_file_to_buffer(const char *filepath, uint8_t **buffer, size_t *size) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }

    long file_size = ftell(fp);
    if (file_size < 0) {
        fclose(fp);
        return -1;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }

    *buffer = (uint8_t *)platform_malloc(file_size);
    if (*buffer == NULL) {
        fclose(fp);
        return -1;
    }

    size_t bytes_read = fread(*buffer, 1, file_size, fp);
    fclose(fp);

    if (bytes_read != (size_t)file_size) {
        free(*buffer);
        *buffer = NULL;
        return -1;
    }

    *size = file_size;
    return 0;
}

// 将缓冲写入文件（不存在则创建，存在则截断）；成功 0，失败 -1
int32_t platform_write_buffer_to_file(const char *filepath, const uint8_t *buffer, size_t size) {
    if (!filepath || (!buffer && size > 0)) return -1;

    FILE *fp = fopen(filepath, "wb");
    if (!fp) return -1;

    size_t written = (size > 0) ? fwrite(buffer, 1, size, fp) : 0;
    fclose(fp);

    return (written == size) ? 0 : -1;
}

// 判断路径是否为目录：1 是；0 否或 stat 失败
int32_t platform_is_directory(const char *path) {
    if (!path) return 0;
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode) ? 1 : 0;
}

int32_t platform_mkdir(const char *path) {
    if (platform_is_directory(path)) {
        return 0; // 已存在且为目录
    }
    return (mkdir(path, 0755) == 0) ? 0 : -1;
}

// ---------------- 随机访问文件读取（全局单句柄，与 ESP32 SD File 语义一致） ----------------

static FILE *s_platform_file = NULL;

int32_t platform_file_open(const char *filepath) {
    if (!filepath) return -1;
    platform_file_close(); // 同一时刻仅允许一个打开文件：先关闭旧句柄
    s_platform_file = fopen(filepath, "rb");
    return s_platform_file ? 0 : -1;
}

uint32_t platform_file_size(void) {
    if (!s_platform_file) return 0;
    long cur = ftell(s_platform_file);
    if (fseek(s_platform_file, 0, SEEK_END) != 0) return 0;
    long size = ftell(s_platform_file);
    if (cur >= 0) fseek(s_platform_file, cur, SEEK_SET); // 恢复原读写位置
    return (size > 0) ? (uint32_t)size : 0;
}

int32_t platform_file_seek(uint32_t offset) {
    if (!s_platform_file) return -1;
    return (fseek(s_platform_file, (long)offset, SEEK_SET) == 0) ? 0 : -1;
}

int32_t platform_file_read(uint8_t *buffer, size_t size) {
    if (!s_platform_file || !buffer) return -1;
    return (int32_t)fread(buffer, 1, size, s_platform_file); // 返回实际读取字节数
}

void platform_file_close(void) {
    if (s_platform_file) {
        fclose(s_platform_file);
        s_platform_file = NULL;
    }
}

// 枚举目录下的文件名（不含路径，不含 "." / ".."）。
//   dir       ：目录路径
//   filenames ：为 NULL 时仅返回条目数量；
//               非 NULL 时逐个 platform_malloc 填充文件名（调用方负责 free）。
// 返回值：>=0 条目数量；-1 目录打开失败；-2 内存分配失败（已填充项回滚释放）。
int32_t list_files(const char *dir, char **filenames) {
    if (!dir) return -1;

    DIR *dp = opendir(dir);
    if (!dp) return -1;

    int32_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (filenames) {
            size_t len = strlen(entry->d_name);
            char *name = (char *)platform_malloc(len + 1);
            if (!name) { // 分配失败：回滚已填充项
                for (int32_t i = 0; i < count; i++) {
                    free(filenames[i]);
                    filenames[i] = NULL;
                }
                closedir(dp);
                return -2;
            }
            memcpy(name, entry->d_name, len + 1);
            filenames[count] = name;
        }
        count++;
    }

    closedir(dp);
    return count;
}

// 辅助函数：在 wchar_t 层面对字符串进行 JSON 转义（\n, ", \）
static wchar_t* escape_wchar_string(const wchar_t* wstr) {
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
static char* wchar_to_utf8(const wchar_t* wstr) {
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
