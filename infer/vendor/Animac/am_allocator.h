#ifndef __AM_ALLOCATOR_H__
#define __AM_ALLOCATOR_H__

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////
// 抽象内存分配器
// NOTE 注意allocator返回的指针必须是2字节对齐的！以确保其am_value_t的最低位恒0。
///////////////////////////////////////////

// 定义抽象内存管理的虚接口（虚函数表）
typedef struct am_allocator_vtable_t {
    void* (*malloc)(void *state, size_t size);
    void* (*calloc)(void *state, size_t size);
    void* (*realloc)(void *state, void *ptr, size_t size);
    void  (*free)(void *state, void *ptr);
    void  (*destroy)(void *state); // 销毁整个分配器
} am_allocator_vtable_t;

// 抽象内存管理器：其实现待定
typedef struct am_allocator_t {
    const am_allocator_vtable_t *vtable; // 指向具体的实现
    void *state; // TODO 具体策略的上下文（如FreeList的头指针，Arena的指针等）
} am_allocator_t;

// 抽象内存管理接口
static inline void* am_malloc(am_allocator_t *alloc, size_t size) {
    return alloc->vtable->malloc(alloc->state, size);
}
static inline void* am_calloc(am_allocator_t *alloc, size_t size) {
    return alloc->vtable->calloc(alloc->state, size);
}
static inline void* am_realloc(am_allocator_t *alloc, void *ptr, size_t size) {
    return alloc->vtable->realloc(alloc->state, ptr, size);
}
static inline void am_free(am_allocator_t *alloc, void *ptr) {
    return alloc->vtable->free(alloc->state, ptr);
}

// 示例：虚函数的具体实现
// void* am_malloc_impl(void *state, size_t size) { return malloc(size); }
// void am_free_impl(void *state, void *ptr) { free(ptr); }
// const am_allocator_vtable_t malloc_vtable = { am_malloc_impl, am_free_impl, NULL };





#ifndef AM_ALLOCATOR_PRINT_COMPACT_REPORT
#define AM_ALLOCATOR_PRINT_COMPACT_REPORT (0)
#endif

///////////////////////////////////////////
// 宿主内存分配虚函数表（依赖倒置）
// 说明：allocator 不直接依赖宿主系统的 malloc/calloc/realloc/free，
// 而是由宿主在调用 am_allocator_pool_create 时，通过本虚函数表注入具体实现。
// 四个成员均为必需能力，任一为 NULL 时 am_allocator_pool_create 失败。
///////////////////////////////////////////

typedef struct am_allocator_host_vtable_t {
    void *(*host_malloc)(size_t nbytes);
    void *(*host_calloc)(size_t n, size_t sizeoftype);
    void *(*host_realloc)(void *ptr, size_t n);
    void  (*host_free)(void *ptr);
} am_allocator_host_vtable_t;

///////////////////////////////////////////
// 共享内存池与双分配器管理
///////////////////////////////////////////

// 动态边界调整相关阈值与限制。
// 边界以占总池比例表示；heap 区最小/最大比例受以下宏约束。
#ifndef AM_POOL_MIN_HEAP_RATIO
#define AM_POOL_MIN_HEAP_RATIO (0.1)
#endif
#ifndef AM_POOL_MIN_VM_RATIO
#define AM_POOL_MIN_VM_RATIO (0.1)
#endif

#ifndef AM_POOL_VM_EXPAND_THRESHOLD
#define AM_POOL_VM_EXPAND_THRESHOLD (0.75)
#endif
#ifndef AM_POOL_HEAP_EXPAND_THRESHOLD
#define AM_POOL_HEAP_EXPAND_THRESHOLD (0.75)
#endif
#ifndef AM_POOL_VM_SLACK_THRESHOLD
#define AM_POOL_VM_SLACK_THRESHOLD (0.30)
#endif
#ifndef AM_POOL_HEAP_SLACK_THRESHOLD
#define AM_POOL_HEAP_SLACK_THRESHOLD (0.30)
#endif
#ifndef AM_POOL_BOUNDARY_ADJ_STEP
#define AM_POOL_BOUNDARY_ADJ_STEP (0.05)
#endif

// 不透明内存池类型
typedef struct am_allocator_pool_t am_allocator_pool_t;

// 创建/销毁统一内存池。成功返回池指针，失败返回 NULL。
// host_vtable 为宿主内存分配虚函数表，不允许为 NULL，且四个成员均不允许为 NULL；
// 池仅保存指针，不拷贝，宿主须保证 vtable 的生命周期不短于池。
am_allocator_pool_t *am_allocator_pool_create(size_t total_size, const am_allocator_host_vtable_t *host_vtable);
void am_allocator_pool_destroy(am_allocator_pool_t *pool);

// 获取池中 VM 工作区与堆区分配器。
am_allocator_t *am_allocator_pool_get_vm(am_allocator_pool_t *pool);
am_allocator_t *am_allocator_pool_get_heap(am_allocator_pool_t *pool);

// 重置 VM 工作区/堆区。重置会丢弃当前已分配内容，回到初始状态。
void am_allocator_pool_reset_vm(am_allocator_pool_t *pool);
void am_allocator_pool_reset_heap(am_allocator_pool_t *pool);

// 查询池大小与已使用字节数。
size_t am_allocator_pool_total_size(const am_allocator_pool_t *pool);
size_t am_allocator_pool_vm_used(const am_allocator_pool_t *pool);
size_t am_allocator_pool_heap_used(const am_allocator_pool_t *pool);
size_t am_allocator_pool_heap_capacity(const am_allocator_pool_t *pool);

// 经池的宿主内存分配虚函数表分配/释放临时内存（供 GC 等上层做暂存）。
// 仅支持内存池的堆区分配器（am_allocator_pool_get_heap 的返回值），其余返回 NULL。
void *am_allocator_host_malloc(am_allocator_t *alloc, size_t size);
void *am_allocator_host_realloc(am_allocator_t *alloc, void *ptr, size_t size);
void  am_allocator_host_free(am_allocator_t *alloc, void *ptr);

// 查询堆区分配器的使用量、最大空闲块与近期最大分配请求（供 GC 水位与碎片判断；后两个参数可传 NULL 跳过）。
// 仅支持内存池的堆区分配器（am_allocator_pool_get_heap 的返回值），其余返回 -1。
int32_t am_allocator_heap_usage(const am_allocator_t *alloc, size_t *used_bytes, size_t *capacity,
                                size_t *largest_free_block, size_t *largest_request);

// 读取并清除堆区分配失败标志：此前曾发生彻底分配失败（边界让渡重试后仍失败）
// 返回 1 并清除标志，否则返回 0；alloc 非堆区分配器返回 -1。
int32_t am_allocator_heap_take_oom_flag(am_allocator_t *alloc);

// 重定位回调：存活对象被搬移到 new_payload 后由压缩引擎回调，按地址升序逐次触发。
typedef void (*am_allocator_relocate_fn)(void *ctx, void *old_payload, void *new_payload);

// 对堆区执行标记-压缩引擎（纯物理操作，不依赖逻辑堆）：
// 遍历堆区物理块，将 payload 出现在 live_payloads 中的已用块搬移到堆区前端，
// 每搬移一个对象经 on_relocate 回调报告一次重定位（old/new payload 均按地址升序），
// 最后在尾部重建一个空闲块。live_payloads 必须是按指针升序且无重复的数组。
// 必须在 GC 安全点调用。成功返回 0，失败返回 -1。
int32_t am_allocator_heap_compact(am_allocator_t *heap_alloc,
                                  void *const *live_payloads, size_t live_count,
                                  am_allocator_relocate_fn on_relocate, void *ctx);

// 按占总池比例调整 VM/heap 边界。
// - ratio 为 heap 区所占比例；内部会被裁剪到 [AM_POOL_MIN_HEAP_RATIO, 1 - AM_POOL_MIN_VM_RATIO]。
// - 若新边界大于当前边界（heap 扩张），仅当 VM 工作区为空时才允许。
// - 若新边界小于当前边界（VM 扩张），要求当前已用 heap 对象能够放入新的 heap 容量中。
int32_t am_allocator_pool_adjust_boundary(am_allocator_pool_t *pool, double ratio);

// 根据 VM/heap 使用压力自动调整边界。通常在每个 GC 安全点之后调用。
int32_t am_allocator_pool_auto_adjust(am_allocator_pool_t *pool);

// 返回当前活动的内存池（单池场景下使用）。
am_allocator_pool_t *am_allocator_pool_current(void);

#ifdef __cplusplus
}
#endif

#endif