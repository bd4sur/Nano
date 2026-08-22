#ifndef __NANO_HAL_RAM_H__
#define __NANO_HAL_RAM_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================================
// hal_ram：内存抽象层（硬件无关接口；ESP32 实现见 hal_ram_m5esp.cpp）
//   external：大容量主堆（ESP32 上为 PSRAM 堆，供大块内存分配）；
//   internal：内部 RAM 堆（小块/性能敏感/DMA 缓冲）
//   宿主机自测/其他平台可分别提供 calloc/malloc 打桩实现。
// ===============================================================================

// 根据设备类型选择不同的 m/calloc 实现（external = PSRAM 主堆）
void *platform_calloc(size_t n, size_t sizeoftype);
void *platform_malloc(size_t nbytes);
void *platform_realloc(void *ptr, size_t n);

// internal = 内部 RAM 堆
void *platform_calloc_internal(size_t n, size_t sizeoftype);
void *platform_malloc_internal(size_t nbytes);
void *platform_realloc_internal(void *ptr, size_t n);

// 内存使用情况查询（字节）
uint32_t platform_get_free_heap_size(void);             // 主堆当前空闲总量
uint32_t platform_get_largest_free_block(void);         // 主堆最大连续空闲块
uint32_t platform_get_free_heap_size_internal(void);    // 内部RAM堆当前空闲总量
uint32_t platform_get_largest_free_block_internal(void); // 内部RAM堆最大连续空闲块

#ifdef __cplusplus
}
#endif

#endif
