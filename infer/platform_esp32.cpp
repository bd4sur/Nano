#include "platform.h"

#include <sys/time.h>
#include <Arduino.h>
#include <esp32-hal-psram.h>
#include <Wire.h>

extern "C" {

void sleep_in_ms(uint32_t ms) {
    delay(ms);
}

uint64_t get_timestamp_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
}

// 优雅关机
int32_t graceful_shutdown() {
    return 0;
}

// 主函数：将 prompt 和 response 转义、转换、写入 log.jsonl
int32_t write_chat_log(char *filepath, uint64_t timestamp, wchar_t* prompt, wchar_t* response) {
    // Stub
    return 0;
}

// 读取文件内容（UTF-8），并转换为 wchar_t* 字符串
wchar_t* read_file_to_wchar(char* filename) {
    // Stub
    return NULL;
}


void *platform_calloc(size_t n, size_t sizeoftype) {
    return heap_caps_calloc((n), (sizeoftype), MALLOC_CAP_SPIRAM);
}

void *platform_calloc_internal(size_t n, size_t sizeoftype) {
    return heap_caps_calloc((n), (sizeoftype), MALLOC_CAP_DEFAULT);
}

void *platform_malloc(size_t nbytes) {
    return heap_caps_malloc((nbytes), MALLOC_CAP_SPIRAM);
}

void *platform_malloc_internal(size_t nbytes) {
    return heap_caps_malloc((nbytes), MALLOC_CAP_DEFAULT);
}

void *platform_realloc(void *ptr, size_t n) {
    return heap_caps_realloc((ptr), (n), MALLOC_CAP_SPIRAM);
}

void *platform_realloc_internal(void *ptr, size_t n) {
    return heap_caps_realloc((ptr), (n), MALLOC_CAP_DEFAULT);
}

}
