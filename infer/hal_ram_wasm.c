// hal_ram_wasm.c —— Nano-WASM-GUI 内存分配/堆查询（Emscripten）
//
// 分配直通 emscripten libc（dlmalloc）；堆查询返回固定大值：
// Emscripten 无 sysinfo，且 wasm 线性内存可经 ALLOW_MEMORY_GROWTH 按需增长，
// 以 512MB 上报（注意不可为 0：ui_animac.h 以 512K+64K 门槛判定可否创建编辑器）。

#include "platform.h"
#include "hal_ram.h"

#include <stdlib.h>

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

// 堆使用情况查询（固定上报 512MB；wasm 线性内存按需增长，无实际碎片化问题）
uint32_t platform_get_free_heap_size(void) {
    return 512u * 1024u * 1024u;
}

uint32_t platform_get_largest_free_block(void) {
    return 512u * 1024u * 1024u;
}

uint32_t platform_get_free_heap_size_internal(void) {
    return 512u * 1024u * 1024u;
}

uint32_t platform_get_largest_free_block_internal(void) {
    return 512u * 1024u * 1024u;
}
