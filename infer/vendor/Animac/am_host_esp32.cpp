#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <esp32-hal-psram.h>

#include <Arduino.h>
#include <SD.h>

#include "am_host.h"

// 短时睡眠（毫秒）
void am_sleep_in_ms(uint64_t ms) {
    delay(ms);
}

// 获取当前时间戳（毫秒）。优先使用 POSIX clock_gettime，失败则回退到 time()。
uint64_t am_current_timestamp_in_ms() {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
    }
    return (uint64_t)time(NULL) * 1000;
}

// 读取文件内容（UTF-8），并转换为 wchar_t* 字符串
// 成功时返回动态分配的 wchar_t*（以 L'\0' 结尾），失败返回 NULL。调用者需用 free() 释放返回值。
wchar_t* am_read_file_to_wchar(char* filename) {
    if (!filename) return NULL;

    // 从 SD 卡打开文件（二进制读取，与 main.cpp 中 .load 逻辑一致）
    File f = SD.open(filename, FILE_READ);
    if (!f) {
        return NULL;
    }

    size_t size = f.size();
    if (size == 0) {
        f.close();
        wchar_t* empty = (wchar_t*)calloc(1, sizeof(wchar_t));
        if (empty) *empty = L'\0';
        return empty;
    }

    // 读取全部内容到 char 缓冲区（+1 保证可加 '\0'）
    char* buffer = (char*)calloc(size + 1, sizeof(char));
    if (!buffer) {
        f.close();
        return NULL;
    }

    size_t bytes_read = 0;
    char chunk[256];
    while (bytes_read < size) {
        size_t to_read = sizeof(chunk);
        if (size - bytes_read < to_read) {
            to_read = size - bytes_read;
        }
        int n = f.read((uint8_t*)chunk, to_read);
        if (n <= 0) break;
        memcpy(buffer + bytes_read, chunk, n);
        bytes_read += n;
    }
    f.close();

    if (bytes_read != size) {
        free(buffer);
        return NULL;
    }
    buffer[size] = '\0'; // 确保以 null 结尾（UTF-8 是 null-safe 的）

    // 计算所需 wchar_t 数量
    size_t wlen = size;

    // 分配 wchar_t 缓冲区
    wchar_t* wstr = (wchar_t*)calloc((wlen + 1), sizeof(wchar_t));
    if (!wstr) {
        free(buffer);
        return NULL;
    }

    // 执行实际转换（length 为 buffer 中实际字节数，不含结尾额外 \0）
    (void)am_mbstowcs(wstr, buffer, size);
    free(buffer);

    return wstr; // 调用者负责 free()
}


// 链接器模块源码读取回调的宿主侧默认实现（文件系统版）。
// 将宽字符绝对路径转为 UTF-8 路径后读取文件内容，再用 alloc 分配并返回源码字符串。
wchar_t *am_host_read_source_from_file(am_allocator_t *alloc, const wchar_t *abs_path, void *user_data) {
    (void)user_data;
    if (!alloc || !abs_path) return NULL;

    // UTF-8 最坏情况每个码点 4 字节
    size_t buf_size = wcslen(abs_path) * 4 + 1;
    char *mb_path = (char *)am_malloc(alloc, buf_size);
    if (!mb_path) return NULL;
    am_wcstombs(mb_path, abs_path, (uint32_t)buf_size);

    wchar_t *raw = am_read_file_to_wchar(mb_path);
    am_free(alloc, mb_path);
    if (!raw) return NULL;

    // am_read_file_to_wchar 返回的缓冲区由 free() 管理，此处拷贝一份纳入 allocator 管理
    size_t len = wcslen(raw);
    wchar_t *code = (wchar_t *)am_malloc(alloc, (len + 1) * sizeof(wchar_t));
    if (!code) {
        free(raw);
        return NULL;
    }
    wcscpy(code, raw);
    free(raw);
    return code;
}


// 从 Linux 格式的文件路径中提取文件所在目录的绝对路径
// 即最后一个 '/' 之前的内容，不包含末尾的 '/'
// 返回值：动态分配的字符串，调用者需 free()；失败或路径不含 '/' 时返回 NULL
char* am_path_dirname(const char *path) {
    if (!path) return NULL;

    const char *last_slash = strrchr(path, '/');
    if (!last_slash) return NULL;

    size_t len = (size_t)(last_slash - path);
    char *dir = (char *)malloc(len + 1);
    if (!dir) return NULL;

    memcpy(dir, path, len);
    dir[len] = '\0';
    return dir;
}


// ===============================================================================
// 宿主内存分配
// ===============================================================================

void *am_host_calloc(size_t n, size_t sizeoftype) {
    return heap_caps_calloc((n), (sizeoftype), MALLOC_CAP_SPIRAM);
}

void *am_host_malloc(size_t nbytes) {
    return heap_caps_malloc((nbytes), MALLOC_CAP_SPIRAM);
}

void *am_host_realloc(void *ptr, size_t n) {
    return heap_caps_realloc((ptr), (n), MALLOC_CAP_SPIRAM);
}

void am_host_free(void *ptr) {
    return free(ptr);
}

// 宿主内存分配虚函数表的默认实例
const am_allocator_host_vtable_t am_host_default_vtable = {
    am_host_malloc,
    am_host_calloc,
    am_host_realloc,
    am_host_free
};


// ===============================================================================
// 字符编码
// ===============================================================================

// 将 UTF-32 码点（wchar_t）数组转换为 UTF-8 字符串
// 注意：此函数假设 wchar_t 为 32 位（即 UTF-32）
uint32_t am_wcstombs(char *dest, const wchar_t *src, uint32_t dest_size) {
    const wchar_t *p = src;
    char *q = dest;
    uint32_t dest_len = 0;
    char *end = dest + dest_size - 1; // 留 1 字节给 \0

    while (*p != L'\0' && q < end) {
        uint32_t cp = (uint32_t)*p++;
        if (cp <= 0x7F && q < end) {
            *q++ = (char)cp;
            dest_len++;
        } else if (cp <= 0x7FF && q + 1 < end) {
            *q++ = 0xC0 | (cp >> 6);
            *q++ = 0x80 | (cp & 0x3F);
            dest_len += 2;
        } else if (cp <= 0xFFFF && q + 2 < end) {
            *q++ = 0xE0 | (cp >> 12);
            *q++ = 0x80 | ((cp >> 6) & 0x3F);
            *q++ = 0x80 | (cp & 0x3F);
            dest_len += 3;
        } else if (cp <= 0x10FFFF && q + 3 < end) {
            *q++ = 0xF0 | (cp >> 18);
            *q++ = 0x80 | ((cp >> 12) & 0x3F);
            *q++ = 0x80 | ((cp >> 6) & 0x3F);
            *q++ = 0x80 | (cp & 0x3F);
            dest_len += 4;
        } else if (q < end) {
            *q++ = '?';
            dest_len++;
        }
    }
    *q = '\0';
    return dest_len;
}

// 将 UTF-8 字符串转换为 null-terminated 的 UTF-32 (wchar_t) 字符串
// 返回值：成功转换的 wchar_t 字符数量（不包括结尾的 L'\0'）
// 注意：dest 必须有至少 (length + 1) 个 wchar_t 的空间（最坏情况）
uint32_t am_mbstowcs(wchar_t *dest, const char *src, uint32_t length) {
    const uint8_t *p = (const uint8_t *)src;
    const uint8_t *end = p + length;
    wchar_t *out = dest;

    while (p < end) {
        uint8_t byte = *p;
        if (byte == '\0') break;
        p++;

        if ((byte & 0x80) == 0) {
            // 1-byte: ASCII
            if (dest) *out++ = (wchar_t)byte;
            else out++;
        } else if ((byte & 0xE0) == 0xC0) {
            // 2-byte
            if (p >= end) goto invalid;
            uint8_t b2 = *p++;
            if ((b2 & 0xC0) != 0x80) goto invalid;
            uint32_t cp = ((byte & 0x1F) << 6) | (b2 & 0x3F);
            if (cp < 0x80) goto invalid; // overlong
            if (dest) *out++ = (wchar_t)cp;
            else out++;
        } else if ((byte & 0xF0) == 0xE0) {
            // 3-byte
            if (p + 1 >= end) goto invalid;
            uint8_t b2 = *p++;
            uint8_t b3 = *p++;
            if ((b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) goto invalid;
            uint32_t cp = ((byte & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
            if (cp < 0x800) goto invalid;
            if (cp >= 0xD800 && cp <= 0xDFFF) goto invalid; // surrogate
            if (dest) *out++ = (wchar_t)cp;
            else out++;
        } else if ((byte & 0xF8) == 0xF0) {
            // 4-byte
            if (p + 2 >= end) goto invalid;
            uint8_t b2 = *p++;
            uint8_t b3 = *p++;
            uint8_t b4 = *p++;
            if ((b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 || (b4 & 0xC0) != 0x80) goto invalid;
            uint32_t cp = ((byte & 0x07) << 18) | ((b2 & 0x3F) << 12) |
                          ((b3 & 0x3F) << 6) | (b4 & 0x3F);
            if (cp < 0x10000) goto invalid;
            if (cp > 0x10FFFF) goto invalid;
            if (dest) *out++ = (wchar_t)cp;
            else out++;
        } else {
            goto invalid;
        }
    }

    // 正常结束：添加 null 终止符
    *out = (wchar_t)0;
    return (uint32_t)(out - dest);

invalid:
    // 遇到无效 UTF-8：用 '?' 替代并终止
    if (dest) *out++ = (wchar_t)'?';
    else out++;
    if (dest) *out = (wchar_t)0;  // 仍然保证 null-terminated
    return (uint32_t)(out - dest); // 返回包含 '?' 的字符数
}
