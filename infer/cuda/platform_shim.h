//
// platform_shim.h - 为 CUDA 引擎提供 platform.h 的最小替代
//
//   原 platform.h 面向多平台 HAL（RAM/FS/OS/显示/触摸...），CUDA 引擎只需要
//   内存分配与宽窄字符转换。这里用 glibc 直接实现，避免拖入整个 HAL。
//
//   注意：原工程中 platform_malloc/calloc 是 hal_ram_linux.c 里的外部函数，
//   这里用 static inline 宏级等价物替代，语义一致（calloc/malloc/realloc）。
//

#ifndef __NANO_PLATFORM_SHIM_H__
#define __NANO_PLATFORM_SHIM_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <wchar.h>

static inline void *platform_malloc(size_t nbytes) {
    return malloc(nbytes);
}
static inline void *platform_calloc(size_t n, size_t sizeoftype) {
    return calloc(n, sizeoftype);
}
static inline void *platform_realloc(void *ptr, size_t n) {
    return realloc(ptr, n);
}

// 原 utils.c 中的宽窄字符转换（libc 直通封装）
static inline uint32_t _wcstombs(char *dest, const wchar_t *src, uint32_t length) {
    return (uint32_t)wcstombs(dest, src, length);
}
static inline uint32_t _mbstowcs(wchar_t *dest, const char *src, uint32_t length) {
    return (uint32_t)mbstowcs(dest, src, length);
}

#endif
