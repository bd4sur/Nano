#include "platform.h"
#include "hal_ram.h"

#include <Arduino.h>
#include <esp32-hal-psram.h>

// ---------------- 内存分配（PSRAM 主堆 = external） ----------------

void *platform_calloc(size_t n, size_t sizeoftype) {
    return heap_caps_calloc((n), (sizeoftype), MALLOC_CAP_SPIRAM);
}

void *platform_malloc(size_t nbytes) {
    return heap_caps_malloc((nbytes), MALLOC_CAP_SPIRAM);
}

void *platform_realloc(void *ptr, size_t n) {
    return heap_caps_realloc((ptr), (n), MALLOC_CAP_SPIRAM);
}

// ---------------- 内部 RAM 堆 = internal ----------------

void *platform_calloc_internal(size_t n, size_t sizeoftype) {
    return heap_caps_calloc((n), (sizeoftype), MALLOC_CAP_DEFAULT);
}

void *platform_malloc_internal(size_t nbytes) {
    return heap_caps_malloc((nbytes), MALLOC_CAP_DEFAULT);
}

void *platform_realloc_internal(void *ptr, size_t n) {
    return heap_caps_realloc((ptr), (n), MALLOC_CAP_DEFAULT);
}

// ---------------- 堆状态查询 ----------------

uint32_t platform_get_free_heap_size() {
    return heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}

uint32_t platform_get_largest_free_block() {
    return heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
}

uint32_t platform_get_free_heap_size_internal() {
    return heap_caps_get_free_size(MALLOC_CAP_DEFAULT);
}

uint32_t platform_get_largest_free_block_internal() {
    return heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT);
}
