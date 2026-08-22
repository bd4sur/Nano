// hal_ram Linux 实现（PC / 树莓派等普通 Linux 系统）。
// Linux 上 malloc 直接由内核按需分配虚拟内存，无 ESP32 的 PSRAM/内部 RAM 之分，
// external 与 internal 变体均直接映射到 glibc 的 c/malloc/realloc。

#include "platform.h"
#include "hal_ram.h"

#include <stdlib.h>
#include <sys/sysinfo.h>

void *platform_calloc(size_t nmemb, size_t size) {
    return calloc(nmemb, size);
}

void *platform_calloc_internal(size_t nmemb, size_t size) {
    return calloc(nmemb, size);
}

void *platform_malloc(size_t n) {
    return malloc(n);
}

void *platform_malloc_internal(size_t n) {
    return malloc(n);
}

void *platform_realloc(void *ptr, size_t n) {
    return realloc(ptr, n);
}

void *platform_realloc_internal(void *ptr, size_t n) {
    return realloc(ptr, n);
}

// 堆状态查询：以 sysinfo 报告的可用物理内存作为近似值。注意：返回值不可为 0，
// ui_animac.h 会依据 largest_free_block 是否超过 512K+64K 决定是否允许
// 创建编辑器内存池。

#define PLATFORM_HEAP_FALLBACK_BYTES (512u * 1024u * 1024u) // sysinfo 失败时的兜底值

static uint32_t platform_sys_free_bytes(void) {
    struct sysinfo info;
    if (sysinfo(&info) != 0) return PLATFORM_HEAP_FALLBACK_BYTES;
    uint64_t free_bytes = (uint64_t)info.freeram * (uint64_t)info.mem_unit;
    if (free_bytes > UINT32_MAX) free_bytes = UINT32_MAX;
    return (uint32_t)free_bytes;
}

uint32_t platform_get_free_heap_size() {
    return platform_sys_free_bytes();
}

uint32_t platform_get_largest_free_block() {
    // 虚拟内存下最大连续可分配块约等于可用内存
    return platform_sys_free_bytes();
}

// Linux 不区分 PSRAM / 内部 RAM，_internal 变体与外部一致
uint32_t platform_get_free_heap_size_internal() {
    return platform_sys_free_bytes();
}

uint32_t platform_get_largest_free_block_internal() {
    return platform_sys_free_bytes();
}
