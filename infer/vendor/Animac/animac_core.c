/* =============================================================================
 * Animac（灵机）解释器 —— Amalgamation 单文件实现
 *
 * 本文件由 amalgamate.sh 自动生成，请勿手工编辑。
 * 内容来源：解释器核心全部实现文件（src/ 下与核心头文件同名的 .c），
 *           按依赖顺序合并；局部 #include 已剔除；
 *           跨文件重名的 static 符号已按 “<文件基名>__<原名>” 规则改名。
 *           不含 am_host.c / am_native_*.c / am_highlight.c / am_repl.c 等宿主相关实现。
 * 生成时间：2026-07-25 18:08:59 +0800
 * ============================================================================ */

#include "animac_core.h"

/* ===== begin: src/am_allocator.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>


#define AM_ALLOC_ALIGN      (sizeof(void *))
#define AM_ALIGN_UP(x, a)   (((x) + (a) - 1) & ~((a) - 1))
#define AM_BLOCK_USED_FLAG  ((size_t)1)

/* 用于压缩报告：当前活动的内存池。单池场景下使用。 */
static am_allocator_pool_t *g_current_pool = NULL;

/* =============================================================================
 * VM 工作区分配器：Segregated Free-List + 边界标签合并
 *
 * 工作区对象生命周期差异大（map/list 频繁扩容，临时缓冲区等），且代码中
 * 大量调用 am_free/am_realloc。若继续使用 bump pointer，废弃对象会钉死
 * 内存，导致 VM 空闲空间快速耗尽。改用分离空闲链表：
 *   - 小/中对象按预定义 size class 分桶，分配 O(1)；
 *   - 大对象使用单独的有序空闲链表；
 *   - 释放时按边界标签与相邻空闲块合并，再插回对应桶。
 * ============================================================================ */

/* 预定义 size classes：
 *   48..512  按 16 字节递增（减少 map/list 等常见小对象的内部碎片）
 *   1024..524288 按 2 的幂递增
 *   大于 524288 的块放入 large_free_head 链表 */
static const size_t am_vm_size_classes[] = {
    48, 64, 80, 96, 112, 128, 144, 160, 176, 192,
    208, 224, 240, 256, 272, 288, 304, 320, 336, 352,
    368, 384, 400, 416, 432, 448, 464, 480, 496, 512,
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288
};
#define AM_VM_N_CLASSES (sizeof(am_vm_size_classes) / sizeof(am_vm_size_classes[0]))
#define AM_VM_SMALL_MAX_CLASS (524288)

typedef struct am_vm_block_header_t {
    size_t size;       /* 总块大小（含头部），最低位为 1 表示已分配 */
    size_t prev_size;  /* 前一个块的总大小，首块为 0 */
    struct am_vm_block_header_t *next_free;
    struct am_vm_block_header_t *prev_free;
} am_vm_block_header_t;

#define AM_VM_HEADER_SIZE AM_ALIGN_UP(sizeof(am_vm_block_header_t), AM_ALLOC_ALIGN)
#define AM_VM_MIN_BLOCK_SIZE (AM_VM_HEADER_SIZE + AM_ALLOC_ALIGN)

typedef struct am_segregated_state_t {
    uint8_t *base;      /* VM 区起始地址 */
    uint8_t *end;       /* VM 区结束地址 */
    size_t used_bytes;  /* 已分配字节数 */
    am_vm_block_header_t *buckets[AM_VM_N_CLASSES];
    am_vm_block_header_t *large_free_head; /* 管理大于 AM_VM_SMALL_MAX_CLASS 的块 */
} am_segregated_state_t;

static inline size_t vm_block_real_size(const am_vm_block_header_t *b) {
    return b->size & ~AM_BLOCK_USED_FLAG;
}

static inline bool vm_block_is_used(const am_vm_block_header_t *b) {
    return (b->size & AM_BLOCK_USED_FLAG) != 0;
}

static inline void vm_block_set_size(am_vm_block_header_t *b, size_t sz, bool used) {
    b->size = (sz & ~AM_BLOCK_USED_FLAG) | (used ? AM_BLOCK_USED_FLAG : 0);
}

static inline uint8_t *vm_block_payload(const am_vm_block_header_t *b) {
    return (uint8_t *)b + AM_VM_HEADER_SIZE;
}

static inline am_vm_block_header_t *vm_block_from_payload(void *p) {
    return (am_vm_block_header_t *)((uint8_t *)p - AM_VM_HEADER_SIZE);
}

static inline am_vm_block_header_t *vm_block_next(const am_segregated_state_t *s,
                                                   const am_vm_block_header_t *b) {
    uint8_t *p = (uint8_t *)b + vm_block_real_size(b);
    if (p >= s->end) return NULL;
    return (am_vm_block_header_t *)p;
}

static inline am_vm_block_header_t *vm_block_prev(const am_segregated_state_t *s,
                                                   const am_vm_block_header_t *b) {
    size_t ps = b->prev_size & ~AM_BLOCK_USED_FLAG;
    (void)s;
    if (ps == 0) return NULL;
    return (am_vm_block_header_t *)((uint8_t *)b - ps);
}

/* 扫描 VM 区，返回第一个已用块相对于 VM 基址的偏移。
 * 若 VM 为空或全部空闲，返回 VM 区总容量。
 * 用于判断边界能否向 heap 方向移动：只要新边界不超过第一个已用块地址，
 * 低端空闲空间就可以安全地划归 heap。 */
static size_t vm_lowest_used_offset(const am_segregated_state_t *s) {
    if (!s || s->base >= s->end) return 0;
    if (s->used_bytes == 0) return (size_t)(s->end - s->base);

    uint8_t *p = s->base;
    while (p < s->end) {
        am_vm_block_header_t *b = (am_vm_block_header_t *)p;
        if (vm_block_is_used(b)) return (size_t)(p - s->base);
        p += vm_block_real_size(b);
    }
    return (size_t)(s->end - s->base);
}

/* 根据块大小返回 size class 索引；若超过所有桶则返回 SIZE_MAX */
static size_t vm_size_to_class_index(size_t size) {
    if (size <= 512) {
        if (size <= 48) return 0;
        return ((size + 15) / 16) - 3; /* class[0] = 48 = 3*16 */
    }
    if (size <= AM_VM_SMALL_MAX_CLASS) {
        size_t s = 1024;
        size_t idx = 30;
        while (s < size) {
            s <<= 1;
            idx++;
        }
        return idx;
    }
    return SIZE_MAX;
}

/* 将空闲块插入对应 bucket 或 large list */
static void vm_bucket_insert(am_segregated_state_t *s, am_vm_block_header_t *b) {
    size_t sz = vm_block_real_size(b);
    size_t idx = vm_size_to_class_index(sz);
    if (idx != SIZE_MAX) {
        b->prev_free = NULL;
        b->next_free = s->buckets[idx];
        if (s->buckets[idx]) s->buckets[idx]->prev_free = b;
        s->buckets[idx] = b;
    } else {
        b->prev_free = NULL;
        b->next_free = s->large_free_head;
        if (s->large_free_head) s->large_free_head->prev_free = b;
        s->large_free_head = b;
    }
}

/* 从 bucket 或 large list 中移除空闲块 */
static void vm_bucket_remove(am_segregated_state_t *s, am_vm_block_header_t *b) {
    size_t sz = vm_block_real_size(b);
    size_t idx = vm_size_to_class_index(sz);
    if (idx != SIZE_MAX) {
        if (b->prev_free) b->prev_free->next_free = b->next_free;
        else s->buckets[idx] = b->next_free;
        if (b->next_free) b->next_free->prev_free = b->prev_free;
    } else {
        if (b->prev_free) b->prev_free->next_free = b->next_free;
        else s->large_free_head = b->next_free;
        if (b->next_free) b->next_free->prev_free = b->prev_free;
    }
    b->prev_free = NULL;
    b->next_free = NULL;
}

/* 将释放/拆分出的块尝试与相邻块合并，再插回桶中 */
static void vm_coalesce_and_insert(am_segregated_state_t *s, am_vm_block_header_t *b) {
    size_t new_size = vm_block_real_size(b);
    am_vm_block_header_t *prev = vm_block_prev(s, b);
    am_vm_block_header_t *next = vm_block_next(s, b);

    if (prev && !vm_block_is_used(prev)) {
        vm_bucket_remove(s, prev);
        new_size += vm_block_real_size(prev);
        b = prev;
    }
    if (next && !vm_block_is_used(next)) {
        vm_bucket_remove(s, next);
        new_size += vm_block_real_size(next);
    }

    vm_block_set_size(b, new_size, false);
    am_vm_block_header_t *new_next = vm_block_next(s, b);
    if (new_next) new_next->prev_size = new_size;
    vm_bucket_insert(s, b);
}

/* 在 large list 中查找足够大的空闲块，优先返回地址最高（最靠近 VM 区顶部）的块，
 * 使已用块尽量向 VM 区顶部聚集，低端留出连续空闲空间供边界向 heap 方向移动。 */
static am_vm_block_header_t *vm_find_large_block(am_segregated_state_t *s, size_t min_size) {
    am_vm_block_header_t *best = NULL;
    am_vm_block_header_t *best_prev = NULL;
    am_vm_block_header_t *prev = NULL;
    for (am_vm_block_header_t *cur = s->large_free_head; cur; cur = cur->next_free) {
        if (vm_block_real_size(cur) >= min_size) {
            if (!best || cur > best) {
                best = cur;
                best_prev = prev;
            }
        }
        prev = cur;
    }
    if (!best) return NULL;

    if (best_prev) best_prev->next_free = best->next_free;
    else s->large_free_head = best->next_free;
    if (best->next_free) best->next_free->prev_free = best_prev;
    best->next_free = best->prev_free = NULL;
    return best;
}

/* 从指定 class 开始向上查找足够大的空闲块。
 * 在每个 size class 内部选择地址最高（最靠近 VM 区顶部）的块，
 * 配合“从高端拆分”策略，使已用块向 VM 区顶部聚集。 */
static am_vm_block_header_t *vm_find_free_block(am_segregated_state_t *s, size_t class_idx, size_t alloc_size) {
    for (size_t i = class_idx; i < AM_VM_N_CLASSES; i++) {
        am_vm_block_header_t *prev = NULL;
        am_vm_block_header_t *best = NULL;
        am_vm_block_header_t *best_prev = NULL;
        for (am_vm_block_header_t *cur = s->buckets[i]; cur; cur = cur->next_free) {
            size_t cur_sz = vm_block_real_size(cur);
            if (cur_sz >= alloc_size) {
                if (!best || cur > best) {
                    best = cur;
                    best_prev = prev;
                }
            }
            prev = cur;
        }
        if (best) {
            if (best_prev) best_prev->next_free = best->next_free;
            else s->buckets[i] = best->next_free;
            if (best->next_free) best->next_free->prev_free = best_prev;
            best->next_free = best->prev_free = NULL;
            return best;
        }
    }
    return vm_find_large_block(s, alloc_size);
}

static void *segregated_malloc(void *state, size_t size) {
    am_segregated_state_t *s = (am_segregated_state_t *)state;
    if (size == 0 || !s) return NULL;

    size_t needed = AM_ALIGN_UP(AM_VM_HEADER_SIZE + size, AM_ALLOC_ALIGN);
    if (needed < AM_VM_MIN_BLOCK_SIZE) needed = AM_VM_MIN_BLOCK_SIZE;

    size_t class_idx = vm_size_to_class_index(needed);
    size_t alloc_size = (class_idx != SIZE_MAX) ? am_vm_size_classes[class_idx] : needed;

    am_vm_block_header_t *b;
    if (class_idx != SIZE_MAX) {
        b = vm_find_free_block(s, class_idx, alloc_size);
    } else {
        b = vm_find_large_block(s, needed);
    }
    if (!b) {
        fprintf(stderr, "[allocator] VM segregated 分配失败: 请求 %zu bytes (含头部对齐后 %zu), 已用 %zu / %zu bytes\n",
                size, needed, s->used_bytes, (size_t)(s->end - s->base));
        return NULL;
    }

    size_t block_sz = vm_block_real_size(b);

    /* 拆分：从块的高端分配，低端保留为空闲。这样已用块尽量向 VM 区顶部聚集，
     * 低端形成连续空闲区，便于在需要时把边界向 heap 方向移动。 */
    if (block_sz >= alloc_size + AM_VM_MIN_BLOCK_SIZE) {
        size_t split_size = block_sz - alloc_size;
        am_vm_block_header_t *freeb = b;                       // 低端：保持空闲
        am_vm_block_header_t *usedb = (am_vm_block_header_t *)((uint8_t *)b + split_size); // 高端：分配出去
        vm_block_set_size(freeb, split_size, false);
        // freeb->prev_size 继承原 b 的 prev_size
        vm_block_set_size(usedb, alloc_size, true);
        usedb->prev_size = split_size;
        am_vm_block_header_t *next = vm_block_next(s, usedb);
        if (next) next->prev_size = alloc_size;
        vm_bucket_insert(s, freeb);
        b = usedb;
        block_sz = alloc_size;
    }

    vm_block_set_size(b, block_sz, true);
    s->used_bytes += block_sz;
    return vm_block_payload(b);
}

static void *segregated_calloc(void *state, size_t size) {
    void *p = segregated_malloc(state, size);
    if (p) memset(p, 0, size);
    return p;
}

static void segregated_free(void *state, void *ptr) {
    am_segregated_state_t *s = (am_segregated_state_t *)state;
    if (!ptr || !s) return;

    am_vm_block_header_t *b = vm_block_from_payload(ptr);
    if (!vm_block_is_used(b)) return; /* 重复释放 */

    s->used_bytes -= vm_block_real_size(b);
    vm_block_set_size(b, vm_block_real_size(b), false);
    vm_coalesce_and_insert(s, b);
}

static void *segregated_realloc(void *state, void *ptr, size_t size) {
    if (ptr == NULL) return segregated_malloc(state, size);
    if (size == 0) {
        segregated_free(state, ptr);
        return NULL;
    }

    am_vm_block_header_t *b = vm_block_from_payload(ptr);
    size_t old_payload = vm_block_real_size(b) - AM_VM_HEADER_SIZE;
    if (size <= old_payload) return ptr;

    void *new_ptr = segregated_malloc(state, size);
    if (new_ptr) {
        size_t copy = old_payload < size ? old_payload : size;
        memcpy(new_ptr, ptr, copy);
        segregated_free(state, ptr);
    }
    return new_ptr;
}

static void segregated_destroy(void *state) {
    (void)state;
}

static const am_allocator_vtable_t segregated_vtable = {
    segregated_malloc,
    segregated_calloc,
    segregated_realloc,
    segregated_free,
    segregated_destroy
};

/* =============================================================================
 * 堆区分配器：First-Fit Free-List + 边界标签合并
 * 这是受 GC 管理的用户堆区的底层物理内存分配器。
 * ============================================================================ */

typedef struct am_heap_block_header_t {
    size_t size;       /* 总块大小（含头部），最低位为 1 表示已分配 */
    size_t prev_size;  /* 前一个块的总大小，首块为 0 */
    struct am_heap_block_header_t *next_free;
    struct am_heap_block_header_t *prev_free;
    bool live;         /* 压缩阶段使用的临时标记 */
} am_heap_block_header_t;

#define AM_HEAP_HEADER_SIZE AM_ALIGN_UP(sizeof(am_heap_block_header_t), AM_ALLOC_ALIGN)
#define AM_BLOCK_MIN_SIZE   (AM_HEAP_HEADER_SIZE + AM_ALLOC_ALIGN)

typedef struct am_freelist_state_t {
    uint8_t *base;
    size_t capacity;
    am_heap_block_header_t *free_list_head;
    size_t used_bytes;
    am_allocator_pool_t *pool; /* 指回所属内存池，用于访问宿主内存分配虚函数表 */
    bool oom_flag;             /* 最近一次分配彻底失败（扩界重试后仍失败）时置位，供运行期观测 */
    size_t largest_request;    /* 近期最大分配请求（含块头对齐后），供碎片水位判断；压缩后清零 */
} am_freelist_state_t;

static inline size_t block_real_size(const am_heap_block_header_t *b) {
    return b->size & ~AM_BLOCK_USED_FLAG;
}

static inline bool block_is_used(const am_heap_block_header_t *b) {
    return (b->size & AM_BLOCK_USED_FLAG) != 0;
}

static inline void block_set_size(am_heap_block_header_t *b, size_t sz, bool used) {
    b->size = (sz & ~AM_BLOCK_USED_FLAG) | (used ? AM_BLOCK_USED_FLAG : 0);
}

static inline uint8_t *block_payload(const am_heap_block_header_t *b) {
    return (uint8_t *)b + AM_HEAP_HEADER_SIZE;
}

static inline am_heap_block_header_t *block_from_payload(void *p) {
    return (am_heap_block_header_t *)((uint8_t *)p - AM_HEAP_HEADER_SIZE);
}

static inline am_heap_block_header_t *block_next(const am_freelist_state_t *s,
                                                  const am_heap_block_header_t *b) {
    uint8_t *p = (uint8_t *)b + block_real_size(b);
    if (p >= s->base + s->capacity) return NULL;
    return (am_heap_block_header_t *)p;
}

static inline am_heap_block_header_t *block_prev(const am_freelist_state_t *s,
                                                  const am_heap_block_header_t *b) {
    size_t ps = b->prev_size & ~AM_BLOCK_USED_FLAG;
    (void)s;
    if (ps == 0) return NULL;
    return (am_heap_block_header_t *)((uint8_t *)b - ps);
}

static void freelist_insert(am_freelist_state_t *s, am_heap_block_header_t *b) {
    b->prev_free = NULL;
    b->next_free = s->free_list_head;
    if (s->free_list_head) {
        s->free_list_head->prev_free = b;
    }
    s->free_list_head = b;
}

static void freelist_remove(am_freelist_state_t *s, am_heap_block_header_t *b) {
    if (b->prev_free) {
        b->prev_free->next_free = b->next_free;
    } else {
        s->free_list_head = b->next_free;
    }
    if (b->next_free) {
        b->next_free->prev_free = b->prev_free;
    }
    b->prev_free = NULL;
    b->next_free = NULL;
}

static void freelist_coalesce(am_freelist_state_t *s, am_heap_block_header_t *b) {
    size_t new_size = block_real_size(b);
    am_heap_block_header_t *prev = block_prev(s, b);
    am_heap_block_header_t *next = block_next(s, b);

    if (prev && !block_is_used(prev)) {
        freelist_remove(s, prev);
        new_size += block_real_size(prev);
        b = prev;
    }
    if (next && !block_is_used(next)) {
        freelist_remove(s, next);
        new_size += block_real_size(next);
    }

    block_set_size(b, new_size, false);
    b->live = false;

    am_heap_block_header_t *new_next = block_next(s, b);
    if (new_next) {
        new_next->prev_size = new_size;
    }
    freelist_insert(s, b);
}

// 在空闲链表中 first-fit 查找并分配 needed 字节（含块头）。成功返回 payload 指针，失败返回 NULL。
static void *freelist_first_fit(am_freelist_state_t *s, size_t needed) {
    am_heap_block_header_t *cur = s->free_list_head;
    while (cur) {
        size_t cur_size = block_real_size(cur);
        if (cur_size >= needed) {
            freelist_remove(s, cur);
            size_t remainder = cur_size - needed;
            if (remainder >= AM_BLOCK_MIN_SIZE) {
                am_heap_block_header_t *split = (am_heap_block_header_t *)((uint8_t *)cur + needed);
                block_set_size(split, remainder, false);
                split->prev_size = needed;
                split->live = false;
                am_heap_block_header_t *split_next = block_next(s, split);
                if (split_next) split_next->prev_size = remainder;
                freelist_insert(s, split);
                block_set_size(cur, needed, true);
            } else {
                block_set_size(cur, cur_size, true);
            }
            cur->live = false;
            s->used_bytes += block_real_size(cur);
            return block_payload(cur);
        }
        cur = cur->next_free;
    }
    return NULL;
}

static void *freelist_malloc(void *state, size_t size) {
    am_freelist_state_t *s = (am_freelist_state_t *)state;
    if (size == 0 || !s) return NULL;

    size_t needed = AM_ALIGN_UP(AM_HEAP_HEADER_SIZE + size, AM_ALLOC_ALIGN);
    if (needed < AM_BLOCK_MIN_SIZE) needed = AM_BLOCK_MIN_SIZE;
    if (needed > s->largest_request) s->largest_request = needed;

    void *p = freelist_first_fit(s, needed);
    if (p) return p;

    // L0 兜底：heap 耗尽时，先尝试向 VM 区让渡边界（heap 扩张）再重试，最多 4 次。
    // 注意：分配器层级不允许在此触发 GC，只能“吃掉 VM 区富余”。
    //（以 heap_state.capacity 的变化判定扩界是否成功，避免依赖 pool 结构体定义次序。）
    for (int attempt = 0; attempt < 4; attempt++) {
        size_t old_capacity = s->capacity;
        if (am_allocator_pool_auto_adjust(s->pool) != 0) break;
        if (s->capacity == old_capacity) break;         // 无法进一步扩界
        p = freelist_first_fit(s, needed);
        if (p) return p;
    }

    s->oom_flag = true;
    fprintf(stderr, "[allocator] heap freelist 分配失败: 请求 %zu bytes (含头部对齐后 %zu), 堆已用 %zu / %zu bytes\n",
            size, needed, s->used_bytes, s->capacity);
    return NULL;
}

static void *freelist_calloc(void *state, size_t size) {
    void *p = freelist_malloc(state, size);
    if (p) memset(p, 0, size);
    return p;
}

static void freelist_free(void *state, void *ptr) {
    am_freelist_state_t *s = (am_freelist_state_t *)state;
    if (!ptr || !s) return;

    am_heap_block_header_t *b = block_from_payload(ptr);
    if (!block_is_used(b)) return; /* 重复释放 */

    s->used_bytes -= block_real_size(b);
    block_set_size(b, block_real_size(b), false);
    b->live = false;
    freelist_coalesce(s, b);
}

static void *freelist_realloc(void *state, void *ptr, size_t size) {
    if (ptr == NULL) return freelist_malloc(state, size);
    if (size == 0) {
        freelist_free(state, ptr);
        return NULL;
    }

    am_heap_block_header_t *b = block_from_payload(ptr);
    size_t old_payload = block_real_size(b) - AM_HEAP_HEADER_SIZE;
    if (size <= old_payload) return ptr;

    void *new_ptr = freelist_malloc(state, size);
    if (new_ptr) {
        size_t copy = old_payload < size ? old_payload : size;
        memcpy(new_ptr, ptr, copy);
        freelist_free(state, ptr);
    }
    return new_ptr;
}

static void freelist_destroy(void *state) {
    (void)state;
}

static const am_allocator_vtable_t freelist_vtable = {
    freelist_malloc,
    freelist_calloc,
    freelist_realloc,
    freelist_free,
    freelist_destroy
};

/* =============================================================================
 * 内存池：统一管理 VM 工作区与堆区
 * ============================================================================ */

struct am_allocator_pool_t {
    uint8_t *base;
    size_t total_size;
    size_t boundary;

    const am_allocator_host_vtable_t *host_vtable; /* 宿主内存分配虚函数表 */

    am_segregated_state_t vm_state;
    am_allocator_t vm_alloc;

    am_freelist_state_t heap_state;
    am_allocator_t heap_alloc;
};

static void pool_init_heap(am_allocator_pool_t *pool) {
    am_freelist_state_t *s = &pool->heap_state;
    s->base = pool->base;
    s->capacity = pool->boundary;
    s->used_bytes = 0;
    s->free_list_head = NULL;
    s->pool = pool;
    s->oom_flag = false;
    s->largest_request = 0;

    am_heap_block_header_t *b = (am_heap_block_header_t *)s->base;
    block_set_size(b, s->capacity, false);
    b->prev_size = 0;
    b->next_free = NULL;
    b->prev_free = NULL;
    b->live = false;
    s->free_list_head = b;
}

static void pool_init_vm(am_allocator_pool_t *pool) {
    am_segregated_state_t *s = &pool->vm_state;
    s->base = pool->base + pool->boundary;
    s->end = pool->base + pool->total_size;
    s->used_bytes = 0;
    for (size_t i = 0; i < AM_VM_N_CLASSES; i++) {
        s->buckets[i] = NULL;
    }
    s->large_free_head = NULL;

    size_t cap = (size_t)(s->end - s->base);
    if (cap >= AM_VM_MIN_BLOCK_SIZE) {
        am_vm_block_header_t *b = (am_vm_block_header_t *)s->base;
        vm_block_set_size(b, cap, false);
        b->prev_size = 0;
        vm_bucket_insert(s, b);
    }
}

am_allocator_pool_t *am_allocator_pool_create(size_t total_size, const am_allocator_host_vtable_t *host_vtable) {
    // 宿主内存分配是内存池的必需能力，缺失则创建失败
    if (!host_vtable || !host_vtable->host_malloc || !host_vtable->host_calloc ||
        !host_vtable->host_realloc || !host_vtable->host_free) {
        return NULL;
    }

    am_allocator_pool_t *pool = (am_allocator_pool_t *)host_vtable->host_malloc(sizeof(am_allocator_pool_t));
    if (!pool) {
        fprintf(stderr, "[allocator] 内存池控制块分配失败: sizeof=%zu\n", sizeof(am_allocator_pool_t));
        return NULL;
    }

    pool->base = (uint8_t *)host_vtable->host_malloc(total_size);
    if (!pool->base) {
        fprintf(stderr, "[allocator] 内存池底层内存分配失败: 请求 %zu bytes\n", total_size);
        host_vtable->host_free(pool);
        return NULL;
    }
    pool->host_vtable = host_vtable;
    pool->total_size = total_size;
    pool->boundary = (total_size / 2) & ~(AM_ALLOC_ALIGN - 1);

    pool_init_heap(pool);
    pool_init_vm(pool);

    g_current_pool = pool;

    pool->heap_alloc.vtable = &freelist_vtable;
    pool->heap_alloc.state = &pool->heap_state;

    pool->vm_alloc.vtable = &segregated_vtable;
    pool->vm_alloc.state = &pool->vm_state;

    return pool;
}

void am_allocator_pool_destroy(am_allocator_pool_t *pool) {
    if (!pool) return;
    if (g_current_pool == pool) {
        g_current_pool = NULL;
    }
    if (pool->base) {
        pool->host_vtable->host_free(pool->base);
        pool->base = NULL;
    }
    pool->host_vtable->host_free(pool);
}

am_allocator_t *am_allocator_pool_get_vm(am_allocator_pool_t *pool) {
    if (!pool) return NULL;
    return &pool->vm_alloc;
}

am_allocator_t *am_allocator_pool_get_heap(am_allocator_pool_t *pool) {
    if (!pool) return NULL;
    return &pool->heap_alloc;
}

void am_allocator_pool_reset_vm(am_allocator_pool_t *pool) {
    if (!pool) return;
    pool_init_vm(pool);
}

void am_allocator_pool_reset_heap(am_allocator_pool_t *pool) {
    if (!pool) return;
    pool_init_heap(pool);
}

size_t am_allocator_pool_total_size(const am_allocator_pool_t *pool) {
    return pool ? pool->total_size : 0;
}

size_t am_allocator_pool_vm_used(const am_allocator_pool_t *pool) {
    if (!pool) return 0;
    return pool->vm_state.used_bytes;
}

size_t am_allocator_pool_heap_used(const am_allocator_pool_t *pool) {
    if (!pool) return 0;
    return pool->heap_state.used_bytes;
}

size_t am_allocator_pool_heap_capacity(const am_allocator_pool_t *pool) {
    if (!pool) return 0;
    return pool->boundary;
}

am_allocator_pool_t *am_allocator_pool_current(void) {
    return g_current_pool;
}

/* 计算 pos 之前紧邻的已分配/空闲块大小（从堆首线性扫描），用于重建边界后的空闲块 prev_size。 */
static size_t pool_prev_block_size(const am_freelist_state_t *s, const uint8_t *pos) {
    uint8_t *p = s->base;
    size_t last_size = 0;
    while (p < pos) {
        am_heap_block_header_t *b = (am_heap_block_header_t *)p;
        last_size = block_real_size(b);
        if (p + last_size >= pos) break;
        p += last_size;
    }
    return last_size;
}

/* 返回 heap 区顶部（紧贴 capacity）的连续空闲块大小。若顶部不是空闲块则返回 0。 */
static size_t heap_top_free_size(const am_freelist_state_t *s) {
    am_heap_block_header_t *top_free = NULL;
    uint8_t *p = s->base;
    while (p < s->base + s->capacity) {
        am_heap_block_header_t *b = (am_heap_block_header_t *)p;
        if (!block_is_used(b)) top_free = b;
        p += block_real_size(b);
    }
    if (!top_free) return 0;
    size_t free_start = (size_t)((uint8_t *)top_free - s->base);
    size_t free_size = block_real_size(top_free);
    if (free_start + free_size != s->capacity) return 0;
    return free_size;
}

/* 在不移动 heap 对象的前提下，按新的 boundary 重新初始化 heap 空闲链表。
 * - 收缩时（new_boundary <= capacity）假设 heap 已压缩：已用块集中在底部，
 *   空闲块从 used_bytes 开始延伸到新边界。
 * - 扩张时（new_boundary > capacity）要求 heap 顶部存在连续空闲块，
 *   并将其延伸到新边界。 */
static int32_t pool_reinit_heap_at(am_allocator_pool_t *pool, size_t new_boundary) {
    am_freelist_state_t *s = &pool->heap_state;
    if (new_boundary < s->used_bytes) return -1;

    if (new_boundary > s->capacity) {
        // heap 扩张：延伸顶部空闲块
        size_t top_free = heap_top_free_size(s);
        if (top_free == 0) return -1;
        size_t free_start = s->capacity - top_free;
        am_heap_block_header_t *freeb = (am_heap_block_header_t *)(s->base + free_start);
        s->capacity = new_boundary;
        block_set_size(freeb, new_boundary - free_start, false);
        return 0;
    }

    size_t free_size = new_boundary - s->used_bytes;
    if (free_size != 0 && free_size < AM_BLOCK_MIN_SIZE) return -1;

    s->capacity = new_boundary;
    if (free_size == 0) {
        s->free_list_head = NULL;
        return 0;
    }

    am_heap_block_header_t *freeb = (am_heap_block_header_t *)(pool->base + s->used_bytes);
    block_set_size(freeb, free_size, false);
    freeb->prev_size = (s->used_bytes > 0) ? pool_prev_block_size(s, (uint8_t *)freeb) : 0;
    freeb->next_free = NULL;
    freeb->prev_free = NULL;
    freeb->live = false;
    s->free_list_head = freeb;
    return 0;
}

/* 按新的 boundary 重新初始化 VM segregated 分配器。
 * - 若 VM 为空（used_bytes == 0），允许边界向任意方向移动。
 * - 若边界左移（VM 扩张），保留现有 VM 对象，把新增区域 [new_base, old_base)
 *   作为空闲块加入。
 * - 若边界右移（heap 扩张），调用前必须保证 [old_base, new_base) 内没有已用块，
 *   即低端空闲块可以全部划归 heap。 */
static void pool_reinit_vm_at(am_allocator_pool_t *pool, size_t new_boundary) {
    am_segregated_state_t *s = &pool->vm_state;
    uint8_t *new_base = pool->base + new_boundary;

    if (s->used_bytes == 0) {
        // VM 为空：直接按新区域重新开始
        s->base = new_base;
        s->end = pool->base + pool->total_size;
        s->used_bytes = 0;
        for (size_t i = 0; i < AM_VM_N_CLASSES; i++) {
            s->buckets[i] = NULL;
        }
        s->large_free_head = NULL;

        size_t cap = (size_t)(s->end - s->base);
        if (cap >= AM_VM_MIN_BLOCK_SIZE) {
            am_vm_block_header_t *b = (am_vm_block_header_t *)s->base;
            vm_block_set_size(b, cap, false);
            b->prev_size = 0;
            vm_bucket_insert(s, b);
        }
        return;
    }

    uint8_t *old_base = s->base;
    s->base = new_base;
    s->end = pool->base + pool->total_size;

    if (new_base <= old_base) {
        // VM 扩张（边界左移）：新增低端空间
        size_t added = (size_t)(old_base - new_base);
        if (added >= AM_VM_MIN_BLOCK_SIZE) {
            am_vm_block_header_t *b = (am_vm_block_header_t *)new_base;
            vm_block_set_size(b, added, false);
            b->prev_size = 0;
            // 与紧邻的已有块连接
            am_vm_block_header_t *next = vm_block_next(s, b);
            if (next) next->prev_size = added;
            vm_coalesce_and_insert(s, b);
        }
    } else {
        // VM 收缩（边界右移，heap 扩张）：把 [old_base, new_base) 移出 VM，
        // 保留 [new_base, vm_used_low) 作为新的低端空闲块。
        size_t removed = (size_t)(new_base - old_base);
        am_vm_block_header_t *first = (am_vm_block_header_t *)old_base;
        size_t first_sz = vm_block_real_size(first);
        if (removed > first_sz) removed = first_sz; // 防御性截断，理论上不会发生
        vm_bucket_remove(s, first);

        size_t remaining = first_sz - removed;
        if (remaining >= AM_VM_MIN_BLOCK_SIZE) {
            am_vm_block_header_t *freeb = (am_vm_block_header_t *)new_base;
            vm_block_set_size(freeb, remaining, false);
            freeb->prev_size = 0;
            am_vm_block_header_t *next = vm_block_next(s, freeb);
            if (next) next->prev_size = remaining;
            vm_bucket_insert(s, freeb);
        } else {
            // remaining 为 0：第一个已用块直接位于 VM 区首
            am_vm_block_header_t *next = vm_block_next(s, first);
            if (next) next->prev_size = 0;
        }
    }
}

int32_t am_allocator_pool_adjust_boundary(am_allocator_pool_t *pool, double ratio) {
    if (!pool) return -1;

    double min_ratio = AM_POOL_MIN_HEAP_RATIO;
    double max_ratio = 1.0 - AM_POOL_MIN_VM_RATIO;
    if (ratio < min_ratio) ratio = min_ratio;
    if (ratio > max_ratio) ratio = max_ratio;

    size_t align_mask = ~(AM_ALLOC_ALIGN - 1);
    size_t min_boundary = ((size_t)(pool->total_size * min_ratio)) & align_mask;
    size_t max_boundary = ((size_t)(pool->total_size * max_ratio)) & align_mask;
    size_t new_boundary = ((size_t)(pool->total_size * ratio)) & align_mask;

    if (new_boundary < min_boundary) new_boundary = min_boundary;
    if (new_boundary > max_boundary) new_boundary = max_boundary;
    if (new_boundary == pool->boundary) return 0;

    if (new_boundary > pool->boundary) {
        // heap 扩张：要求 VM 区低端有足够连续空闲空间
        size_t vm_free_bottom = vm_lowest_used_offset(&pool->vm_state);
        size_t vm_used_low = pool->boundary + vm_free_bottom;
        size_t max_new_boundary = vm_used_low;
        if (max_new_boundary > max_boundary) max_new_boundary = max_boundary;
        if (new_boundary > max_new_boundary) new_boundary = max_new_boundary;
        // 若剩余 VM 空闲空间太小无法构成有效空闲块，则全部让给 heap
        if (new_boundary < vm_used_low) {
            size_t remaining = vm_used_low - new_boundary;
            if (remaining > 0 && remaining < AM_VM_MIN_BLOCK_SIZE) {
                new_boundary = vm_used_low;
            }
        }
        if (new_boundary == pool->boundary) return 0;
        if (pool_reinit_heap_at(pool, new_boundary) != 0) return -1;
        pool->boundary = new_boundary;
        pool_reinit_vm_at(pool, new_boundary);
        return 0;
    } else {
        // VM 扩张：要求当前已用 heap 对象能放入新的 heap 容量
        if (pool->heap_state.used_bytes > new_boundary) return -1;
        if (pool_reinit_heap_at(pool, new_boundary) != 0) return -1;
        pool->boundary = new_boundary;
        pool_reinit_vm_at(pool, new_boundary);
        return 0;
    }
}

static void compact_print_boundary_adjust_report(const am_allocator_pool_t *pool,
                                                 size_t old_boundary,
                                                 size_t old_heap_used,
                                                 size_t old_vm_used,
                                                 const char *direction);

int32_t am_allocator_pool_auto_adjust(am_allocator_pool_t *pool) {
    if (!pool) return -1;

    size_t total = pool->total_size;
    size_t heap_cap = pool->boundary;
    size_t vm_cap = total - pool->boundary;
    if (heap_cap == 0 || vm_cap == 0) return -1;

    size_t heap_used = pool->heap_state.used_bytes;
    size_t vm_used = pool->vm_state.used_bytes;

    double heap_ratio = (double)heap_used / (double)heap_cap;
    double vm_ratio = (double)vm_used / (double)vm_cap;
    double current_ratio = (double)heap_cap / (double)total;

    size_t old_boundary = pool->boundary;
    size_t old_heap_used = heap_used;
    size_t old_vm_used = vm_used;

    // VM 压力大且 heap 有富余：把边界让给 VM（减小 heap 比例）
    if (vm_ratio > AM_POOL_VM_EXPAND_THRESHOLD && heap_ratio < AM_POOL_HEAP_SLACK_THRESHOLD) {
        double target = current_ratio - AM_POOL_BOUNDARY_ADJ_STEP;
        int32_t ret = am_allocator_pool_adjust_boundary(pool, target);
        if (ret == 0 && pool->boundary != old_boundary) {
            compact_print_boundary_adjust_report(pool, old_boundary, old_heap_used, old_vm_used,
                                                 "VM 扩张（heap 比例减小）");
        }
        return ret;
    }

    // heap 压力大且 VM 有富余：把边界让给 heap（增大 heap 比例）
    if (heap_ratio > AM_POOL_HEAP_EXPAND_THRESHOLD &&
        vm_ratio < AM_POOL_VM_SLACK_THRESHOLD) {
        double target = current_ratio + AM_POOL_BOUNDARY_ADJ_STEP;
        int32_t ret = am_allocator_pool_adjust_boundary(pool, target);
        if (ret == 0 && pool->boundary != old_boundary) {
            compact_print_boundary_adjust_report(pool, old_boundary, old_heap_used, old_vm_used,
                                                 "heap 扩张（heap 比例增大）");
        }
        return ret;
    }

    return 0;
}

/* =============================================================================
 * 宿主临时内存分配（供 GC 等上层做暂存）
 * 仅支持内存池的堆区分配器（freelist），其余分配器返回 NULL。
 * ============================================================================ */

void *am_allocator_host_malloc(am_allocator_t *alloc, size_t size) {
    if (!alloc || !alloc->state || alloc->vtable != &freelist_vtable) return NULL;
    am_freelist_state_t *s = (am_freelist_state_t *)alloc->state;
    if (!s->pool || !s->pool->host_vtable) return NULL;
    return s->pool->host_vtable->host_malloc(size);
}

void *am_allocator_host_realloc(am_allocator_t *alloc, void *ptr, size_t size) {
    if (!alloc || !alloc->state || alloc->vtable != &freelist_vtable) return NULL;
    am_freelist_state_t *s = (am_freelist_state_t *)alloc->state;
    if (!s->pool || !s->pool->host_vtable) return NULL;
    return s->pool->host_vtable->host_realloc(ptr, size);
}

void am_allocator_host_free(am_allocator_t *alloc, void *ptr) {
    if (!alloc || !alloc->state || alloc->vtable != &freelist_vtable) return;
    am_freelist_state_t *s = (am_freelist_state_t *)alloc->state;
    if (!s->pool || !s->pool->host_vtable) return;
    s->pool->host_vtable->host_free(ptr);
}

// 遍历空闲链表，返回最大空闲块的真实大小（无空闲块时为 0）。O(空闲块数)。
static size_t freelist_largest_free_block(const am_freelist_state_t *s) {
    size_t largest = 0;
    for (am_heap_block_header_t *cur = s->free_list_head; cur; cur = cur->next_free) {
        size_t cur_size = block_real_size(cur);
        if (cur_size > largest) largest = cur_size;
    }
    return largest;
}

int32_t am_allocator_heap_usage(const am_allocator_t *alloc, size_t *used_bytes, size_t *capacity,
                                size_t *largest_free_block, size_t *largest_request) {
    if (!alloc || !alloc->state || alloc->vtable != &freelist_vtable) return -1;
    const am_freelist_state_t *s = (const am_freelist_state_t *)alloc->state;
    if (used_bytes) *used_bytes = s->used_bytes;
    if (capacity) *capacity = s->capacity;
    if (largest_free_block) *largest_free_block = freelist_largest_free_block(s);
    if (largest_request) *largest_request = s->largest_request;
    return 0;
}

int32_t am_allocator_heap_take_oom_flag(am_allocator_t *alloc) {
    if (!alloc || !alloc->state || alloc->vtable != &freelist_vtable) return -1;
    am_freelist_state_t *s = (am_freelist_state_t *)alloc->state;
    if (s->oom_flag) {
        s->oom_flag = false;
        return 1;
    }
    return 0;
}

/* =============================================================================
 * 堆区压缩引擎（纯物理操作）：在 GC 安全点搬移存活对象，经回调报告重定位
 * ============================================================================ */

/* payload 指针比较（升序），供二分查找存活对象数组 */
static int cmp_payload_ptr(const void *a, const void *b) {
    void *const *pa = (void *const *)a;
    void *const *pb = (void *const *)b;
    if ((uintptr_t)*pa < (uintptr_t)*pb) return -1;
    if ((uintptr_t)*pa > (uintptr_t)*pb) return 1;
    return 0;
}

/* 压缩报告用到的空闲块快照 */
typedef struct compact_free_info_t {
    uint8_t *start;
    size_t size;
} compact_free_info_t;

/* 遍历堆区，打印所有空闲块的位置和大小（要求块头部有效） */
static void compact_print_free_blocks(const am_freelist_state_t *s, const char *label) {
    fprintf(stderr, "%s\n", label);
    uint8_t *p = s->base;
    int n = 0;
    while (p < s->base + s->capacity) {
        am_heap_block_header_t *b = (am_heap_block_header_t *)p;
        size_t sz = block_real_size(b);
        if (!block_is_used(b)) {
            fprintf(stderr, "  起始=%p 结束=%p 大小=%zu\n",
                    (void *)p, (void *)(p + sz), sz);
            n++;
        }
        p += sz;
    }
    if (n == 0) {
        fprintf(stderr, "  (无空闲块)\n");
    }
}

/* 打印 VM 工作区信息 */
static void compact_print_vm_section(const am_allocator_pool_t *pool) {
    size_t vm_cap = (size_t)(pool->vm_state.end - pool->vm_state.base);
    size_t vm_used = pool->vm_state.used_bytes;
    fprintf(stderr, "---------- VM 工作区 ----------\n");
    fprintf(stderr, "  起始地址=%p\n", (void *)pool->vm_state.base);
    fprintf(stderr, "  结束地址=%p\n", (void *)pool->vm_state.end);
    fprintf(stderr, "  容量=%zu bytes\n", vm_cap);
    fprintf(stderr, "  已用=%zu bytes\n", vm_used);
    fprintf(stderr, "  空闲=%zu bytes\n", vm_cap - vm_used);
    if (vm_cap > 0) {
        fprintf(stderr, "  使用率=%.2f%%\n", 100.0 * (double)vm_used / (double)vm_cap);
    }
}

/* 打印边界位置信息 */
static void compact_print_boundary_section(const am_allocator_pool_t *pool) {
    uint8_t *boundary_addr = pool->base + pool->boundary;
    double heap_ratio = (double)pool->boundary / (double)pool->total_size;
    fprintf(stderr, "---------- 边界 ----------\n");
    fprintf(stderr, "  边界地址=%p\n", (void *)boundary_addr);
    fprintf(stderr, "  heap 占比=%.2f%%\n", heap_ratio * 100.0);
    fprintf(stderr, "  VM 占比=%.2f%%\n", (1.0 - heap_ratio) * 100.0);
}

/* 打印用户堆区压缩信息 */
static void compact_print_heap_section(const am_freelist_state_t *s,
                                       size_t used_before,
                                       const compact_free_info_t *before_free,
                                       size_t before_free_count,
                                       size_t live_count) {
    fprintf(stderr, "---------- 用户堆区 ----------\n");
    fprintf(stderr, "  起始地址=%p\n", (void *)s->base);
    fprintf(stderr, "  结束地址=%p\n", (void *)(s->base + s->capacity));
    fprintf(stderr, "  容量=%zu bytes\n", s->capacity);
    fprintf(stderr, "  压缩前: 已用=%zu 空闲=%zu\n",
            used_before, s->capacity - used_before);
    fprintf(stderr, "  压缩前空闲块: %zu个\n", before_free_count);
    (void)before_free;
    // if (before_free_count == 0) {
    //     fprintf(stderr, "    (无空闲块)\n");
    // } else {
    //     for (size_t i = 0; i < before_free_count; i++) {
    //         fprintf(stderr, "    起始=%p 结束=%p 大小=%zu\n",
    //                 (void *)before_free[i].start,
    //                 (void *)(before_free[i].start + before_free[i].size),
    //                 before_free[i].size);
    //     }
    // }

    fprintf(stderr, "  压缩后: 已用=%zu 空闲=%zu\n",
            s->used_bytes, s->capacity - s->used_bytes);
    compact_print_free_blocks(s, "  压缩后空闲块:");
    fprintf(stderr, "  存活对象: %zu 个, 共 %zu bytes\n", live_count, s->used_bytes);
    fprintf(stderr, "  使用率: %.2f%%\n", 100.0 * (double)s->used_bytes / (double)s->capacity);
}

/* 打印一次完整的压缩报告。调用时压缩已完成，before_* 参数记录压缩前状态。 */
static void compact_print_report(const am_freelist_state_t *s,
                                 size_t used_before,
                                 const compact_free_info_t *before_free,
                                 size_t before_free_count,
                                 size_t live_count) {
#if !AM_ALLOCATOR_PRINT_COMPACT_REPORT
    (void)s;
    (void)used_before;
    (void)before_free;
    (void)before_free_count;
    (void)live_count;
    return;
#endif

    fprintf(stderr, "\n========== 内存池压缩报告 ==========\n");
    if (g_current_pool) {
        compact_print_vm_section(g_current_pool);
        compact_print_boundary_section(g_current_pool);
        compact_print_heap_section(s, used_before, before_free, before_free_count, live_count);
    } else {
        fprintf(stderr, "内存池信息: (未知)\n");
    }
    fprintf(stderr, "====================================\n\n");
}

/* 打印边界调整报告。调用时调整已完成。 */
static void compact_print_boundary_adjust_report(const am_allocator_pool_t *pool,
                                                 size_t old_boundary,
                                                 size_t old_heap_used,
                                                 size_t old_vm_used,
                                                 const char *direction) {
#if !AM_ALLOCATOR_PRINT_COMPACT_REPORT
    (void)pool;
    (void)old_boundary;
    (void)old_heap_used;
    (void)old_vm_used;
    (void)direction;
    return;
#endif

    if (!pool) return;

    fprintf(stderr, "\n========== 内存池边界调整报告 ==========\n");

    fprintf(stderr, "---------- 调整前 ----------\n");
    size_t old_vm_cap = pool->total_size - old_boundary;
    fprintf(stderr, "  VM 工作区: 起始=%p 容量=%zu 已用=%zu\n",
            (void *)(pool->base + old_boundary), old_vm_cap, old_vm_used);
    fprintf(stderr, "  边界地址=%p (heap 占比 %.2f%%)\n",
            (void *)(pool->base + old_boundary),
            100.0 * (double)old_boundary / (double)pool->total_size);
    fprintf(stderr, "  用户堆区: 起始=%p 容量=%zu 已用=%zu\n",
            (void *)pool->base, old_boundary, old_heap_used);

    fprintf(stderr, "---------- 调整后 ----------\n");
    size_t vm_cap = (size_t)(pool->vm_state.end - pool->vm_state.base);
    size_t vm_used = pool->vm_state.used_bytes;
    fprintf(stderr, "  VM 工作区: 起始=%p 容量=%zu 已用=%zu\n",
            (void *)pool->vm_state.base, vm_cap, vm_used);
    fprintf(stderr, "  边界地址=%p (heap 占比 %.2f%%)\n",
            (void *)(pool->base + pool->boundary),
            100.0 * (double)pool->boundary / (double)pool->total_size);
    fprintf(stderr, "  用户堆区: 起始=%p 容量=%zu 已用=%zu\n",
            (void *)pool->heap_state.base, pool->boundary, pool->heap_state.used_bytes);

    fprintf(stderr, "  调整方向: %s\n", direction ? direction : "无");
    fprintf(stderr, "========================================\n\n");
}

/* 标记-压缩引擎：遍历堆区物理块，将 payload 出现在 live_payloads 中的已用块
 * 搬移到堆区前端，每搬移一个对象经 on_relocate 回调报告一次重定位，
 * 最后在尾部重建一个空闲块。不感知逻辑堆（heap/handle），由上层负责回写指针。
 * 必须在 GC 安全点调用。 */
int32_t am_allocator_heap_compact(am_allocator_t *heap_alloc,
                                  void *const *live_payloads, size_t live_count,
                                  am_allocator_relocate_fn on_relocate, void *ctx) {
    if (!heap_alloc || !heap_alloc->state) return -1;
    if (live_count > 0 && !live_payloads) return -1;
    am_freelist_state_t *s = (am_freelist_state_t *)heap_alloc->state;

#if AM_ALLOCATOR_PRINT_COMPACT_REPORT
    /* 记录压缩前的堆区统计与空闲块分布，用于最后输出报告 */
    const am_allocator_host_vtable_t *hv = s->pool->host_vtable;
    size_t used_before = s->used_bytes;
    compact_free_info_t *before_free = NULL;
    size_t before_free_count = 0;
    size_t before_free_cap = 0;
    {
        uint8_t *p = s->base;
        while (p < s->base + s->capacity) {
            am_heap_block_header_t *b = (am_heap_block_header_t *)p;
            size_t sz = block_real_size(b);
            if (!block_is_used(b)) {
                if (before_free_count >= before_free_cap) {
                    before_free_cap = before_free_cap ? before_free_cap * 2 : 16;
                    compact_free_info_t *tmp = (compact_free_info_t *)hv->host_realloc(
                        before_free, before_free_cap * sizeof(compact_free_info_t));
                    if (!tmp) {
                        fprintf(stderr, "[allocator] 压缩失败: before_free realloc 失败 (%zu bytes)\n",
                                before_free_cap * sizeof(compact_free_info_t));
                        hv->host_free(before_free);
                        return -1;
                    }
                    before_free = tmp;
                }
                before_free[before_free_count].start = p;
                before_free[before_free_count].size = sz;
                before_free_count++;
            }
            p += sz;
        }
    }
#endif

    /* 按地址升序遍历物理块：存活块搬移到堆区前端，其余空间回收。
     * 升序搬移保证 dest 始终不超过源地址，memmove 安全。 */
    uint8_t *dest = s->base;
    size_t prev_size = 0;
    size_t live_moved = 0;
    uint8_t *p = s->base;
    while (p < s->base + s->capacity) {
        am_heap_block_header_t *b = (am_heap_block_header_t *)p;
        size_t sz = block_real_size(b);
        bool live = false;
        if (block_is_used(b) && live_count > 0) {
            void *payload = block_payload(b);
            live = bsearch(&payload, live_payloads, live_count, sizeof(void *), cmp_payload_ptr) != NULL;
        }
        if (live) {
            void *old_payload = block_payload(b);
            if (p != dest) {
                memmove(dest, b, sz);
            }
            am_heap_block_header_t *newb = (am_heap_block_header_t *)dest;
            block_set_size(newb, sz, true);
            newb->prev_size = prev_size;
            newb->live = false;
            if (on_relocate) {
                on_relocate(ctx, old_payload, block_payload(newb));
            }
            dest += sz;
            prev_size = sz;
            live_moved++;
        }
        p += sz;
    }

    /* 尾部重建空闲块 */
    size_t free_size = (s->base + s->capacity) - dest;
    if (free_size > 0) {
        am_heap_block_header_t *freeb = (am_heap_block_header_t *)dest;
        block_set_size(freeb, free_size, false);
        freeb->prev_size = prev_size;
        freeb->next_free = NULL;
        freeb->prev_free = NULL;
        freeb->live = false;
        s->free_list_head = freeb;
    } else {
        s->free_list_head = NULL;
    }
    s->used_bytes = (size_t)(dest - s->base);
    s->largest_request = 0;   // 压缩后空闲空间已重新连续，清零近期最大请求记录

#if AM_ALLOCATOR_PRINT_COMPACT_REPORT
    compact_print_report(s, used_before, before_free, before_free_count, live_moved);
    hv->host_free(before_free);
#endif

    return 0;
}
/* ===== end:   src/am_allocator.c ===== */

/* ===== begin: src/am_object.c ===== */
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


#define AM_OBJECT_STATIC_MASK    ((uint32_t)0x00000001u)
#define AM_OBJECT_KEEPALIVE_MASK ((uint32_t)0x00000002u)
#define AM_OBJECT_ALIVE_MASK     ((uint32_t)0x80000000u)

int32_t am_object_check_static(am_object_t *obj) {
    if (obj == NULL) {
        return -1;
    }
    return (obj->header & AM_OBJECT_STATIC_MASK) ? 0 : -1;
}

int32_t am_object_set_static(am_object_t *obj, int32_t is_static) {
    if (obj == NULL) {
        return -1;
    }
    if (is_static == 0) {
        obj->header |= AM_OBJECT_STATIC_MASK;
        return 0;
    }
    if (is_static == -1) {
        obj->header &= ~AM_OBJECT_STATIC_MASK;
        return 0;
    }
    return -1;
}

int32_t am_object_check_keepalive(am_object_t *obj) {
    if (obj == NULL) {
        return -1;
    }
    return (obj->header & AM_OBJECT_KEEPALIVE_MASK) ? 0 : -1;
}

int32_t am_object_set_keepalive(am_object_t *obj, int32_t is_keepalive) {
    if (obj == NULL) {
        return -1;
    }
    if (is_keepalive == 0) {
        obj->header |= AM_OBJECT_KEEPALIVE_MASK;
        return 0;
    }
    if (is_keepalive == -1) {
        obj->header &= ~AM_OBJECT_KEEPALIVE_MASK;
        return 0;
    }
    return -1;
}

int32_t am_object_check_alive(am_object_t *obj) {
    if (obj == NULL) {
        return -1;
    }
    return (obj->gcmark & AM_OBJECT_ALIVE_MASK) ? 0 : -1;
}

int32_t am_object_set_alive(am_object_t *obj, int32_t is_alive) {
    if (obj == NULL) {
        return -1;
    }
    if (is_alive == 0) {
        obj->gcmark |= AM_OBJECT_ALIVE_MASK;
        return 0;
    }
    if (is_alive == -1) {
        obj->gcmark &= ~AM_OBJECT_ALIVE_MASK;
        return 0;
    }
    return -1;
}
/* ===== end:   src/am_object.c ===== */

/* ===== begin: src/am_map.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>



// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 将容量向上取整为不小于它的最小 2 的幂
static size_t am_map_round_up_capacity(size_t capacity) {
    size_t cap = 1;
    while (cap < capacity) cap <<= 1;
    return cap;
}

// 查找 key 所在槽位。
// 找到返回 0；未找到返回 -1。
// 无论是否找到，*out_insert_idx 都会返回可插入位置（首个墓碑或空槽）。
static int32_t am_map_find_slot(const am_map_t *m, am_value_t key, size_t *out_insert_idx) {
    size_t idx = am_value_hash(key) & m->mask;
    size_t insert_idx = UINT32_MAX;

    while (1) {
        am_value_t k = m->slots[idx].key;
        if (k == AM_MAP_KEY_EMPTY) {
            if (insert_idx == UINT32_MAX) insert_idx = idx;
            *out_insert_idx = insert_idx;
            return -1;
        }
        if (k == AM_MAP_KEY_TOMBSTONE) {
            if (insert_idx == UINT32_MAX) insert_idx = idx;
        } else if (am_value_equal(k, key)) {
            *out_insert_idx = idx;
            return 0;
        }
        idx = (idx + 1) & m->mask;
    }
}

// 原地重哈希：清除墓碑，不改变容量
static int32_t am_map_rehash(am_allocator_t *alloc, am_map_t *map) {
    am_map_t *m = (am_map_t *)map;
    size_t cap = m->capacity;
    size_t entries_size = cap * sizeof(am_map_entry_t);

    am_map_entry_t *old_slots = (am_map_entry_t *)am_malloc(alloc, entries_size);
    if (!old_slots) return -1;
    memcpy(old_slots, m->slots, entries_size);

    for (size_t i = 0; i < cap; i++) {
        m->slots[i].key = AM_MAP_KEY_EMPTY;
        m->slots[i].value = AM_VALUE_NULL;
    }
    m->length = 0;
    m->tombstones = 0;

    for (size_t i = 0; i < cap; i++) {
        if (old_slots[i].key != AM_MAP_KEY_EMPTY && old_slots[i].key != AM_MAP_KEY_TOMBSTONE) {
            size_t insert_idx;
            am_map_find_slot(m, old_slots[i].key, &insert_idx);
            m->slots[insert_idx].key = old_slots[i].key;
            m->slots[insert_idx].value = old_slots[i].value;
            m->length++;
        }
    }

    am_free(alloc, old_slots);
    return 0;
}

// 扩容并重哈希到新容量（new_capacity 会被向上取整为 2 的幂）。
// 返回新的 map 对象指针；失败返回 NULL。原 map 对象会被释放，调用者必须使用返回的新指针。
static am_map_t *am_map_resize(am_allocator_t *alloc, am_map_t *map, size_t new_capacity) {
    am_map_t *m = (am_map_t *)map;
    size_t cap = am_map_round_up_capacity(new_capacity);
    if (cap <= m->capacity) {
        // 容量未增加：仅做重哈希清理墓碑
        if (am_map_rehash(alloc, map) != 0) return NULL;
        return map;
    }

    size_t old_capacity = m->capacity;
    size_t old_entries_size = old_capacity * sizeof(am_map_entry_t);

    am_map_entry_t *old_slots = NULL;
    if (old_entries_size > 0) {
        old_slots = (am_map_entry_t *)am_malloc(alloc, old_entries_size);
        if (!old_slots) return NULL;
        memcpy(old_slots, m->slots, old_entries_size);
    }

    size_t new_total_size = sizeof(am_map_t) + cap * sizeof(am_map_entry_t);
    am_map_t *new_m = (am_map_t *)am_malloc(alloc, new_total_size);
    if (!new_m) {
        am_free(alloc, old_slots);
        return NULL;
    }

    new_m->base = m->base;
    new_m->capacity = cap;
    new_m->mask = cap - 1;
    new_m->length = 0;
    new_m->tombstones = 0;

    for (size_t i = 0; i < cap; i++) {
        new_m->slots[i].key = AM_MAP_KEY_EMPTY;
        new_m->slots[i].value = AM_VALUE_NULL;
    }

    for (size_t i = 0; i < old_capacity; i++) {
        if (old_slots[i].key != AM_MAP_KEY_EMPTY && old_slots[i].key != AM_MAP_KEY_TOMBSTONE) {
            size_t insert_idx;
            am_map_find_slot(new_m, old_slots[i].key, &insert_idx);
            new_m->slots[insert_idx].key = old_slots[i].key;
            new_m->slots[insert_idx].value = old_slots[i].value;
            new_m->length++;
        }
    }

    am_free(alloc, old_slots);
    am_free(alloc, m);
    return (am_map_t *)new_m;
}

// ===============================================================================
// 构造函数
// ===============================================================================

// 以初始容量新建哈希表。capacity 会被向上取整为不小于它的最小 2 的幂。
// 所有 key 初始化为 AM_MAP_KEY_EMPTY，value 初始化为 AM_VALUE_NULL。
am_map_t *am_map_create(am_allocator_t *alloc, size_t capacity) {
    size_t cap = am_map_round_up_capacity(capacity);
    if (cap < 8) cap = 8;

    size_t total_size = sizeof(am_map_t) + cap * sizeof(am_map_entry_t);
    am_map_t *map = (am_map_t *)am_malloc(alloc, total_size);
    if (!map) return NULL;

    memset(map, 0, total_size);

    map->base.type = AM_OBJECT_TYPE_MAP;
    map->capacity = cap;
    map->mask = cap - 1;
    map->length = 0;
    map->tombstones = 0;

    for (size_t i = 0; i < cap; i++) {
        map->slots[i].key = AM_MAP_KEY_EMPTY;
        map->slots[i].value = AM_VALUE_NULL;
    }

    return (am_map_t *)map;
}

// ===============================================================================
// 析构与清理
// ===============================================================================

// 清空哈希表：对所有有效 entry，若 value 是指针则先释放，再将 key 置为 EMPTY、value 置为 NULL
int32_t am_map_clear(am_allocator_t *alloc, am_map_t *map) {
    am_map_t *m = (am_map_t *)map;
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->slots[i].key != AM_MAP_KEY_EMPTY && m->slots[i].key != AM_MAP_KEY_TOMBSTONE) {
            if (am_value_is_ptr(m->slots[i].value)) {
                am_free(alloc, am_value_to_ptr(m->slots[i].value));
            }
        }
        m->slots[i].key = AM_MAP_KEY_EMPTY;
        m->slots[i].value = AM_VALUE_NULL;
    }
    m->length = 0;
    m->tombstones = 0;
    return 0;
}

// 彻底销毁哈希表对象
int32_t am_map_destroy(am_allocator_t *alloc, am_map_t *map) {
    am_map_clear(alloc, map);
    am_free(alloc, map);
    return 0;
}

// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝：创建并返回一个与原 map 内容完全一致的新 map 对象。
// 所有 key/value 按位拷贝（与闭包 Copy 语义一致，不递归拷贝指针指向的对象）。
am_map_t *am_map_copy(am_allocator_t *alloc, am_map_t *map) {
    am_map_t *m = (am_map_t *)map;
    am_map_t *copy = am_map_create(alloc, m->capacity);
    if (!copy) return NULL;

    copy->length = m->length;
    copy->tombstones = m->tombstones;

    for (uint32_t i = 0; i < m->capacity; i++) {
        copy->slots[i].key = m->slots[i].key;
        copy->slots[i].value = m->slots[i].value;
    }

    return copy;
}

// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_map_size(am_allocator_t *alloc, am_map_t *obj) {
    (void)alloc;
    if (!obj) return SIZE_MAX;

    if (obj->capacity > (SIZE_MAX - sizeof(am_map_t)) / sizeof(am_map_entry_t)) {
        return SIZE_MAX;
    }
    return sizeof(am_map_t) + obj->capacity * sizeof(am_map_entry_t);
}

// ===============================================================================
// 对象二进制转储 TODO
// ===============================================================================

// 功能说明：将散列表对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，丢弃墓碑和空闲槽位。
// 磁盘格式（平台无关固定宽度，小端；详见 include/object.h）：
//   [16B] 对象基类头（type=AM_OBJECT_TYPE_MAP）
//   [uvarint] length（有效键值对数量；capacity/墓碑/空槽均不落盘）
//   [length * (dvalue key, dvalue value)] 有效表项
size_t am_map_dump(am_allocator_t *alloc, am_map_t *map, uint8_t *buffer, size_t offset) {
    (void)alloc;
    am_map_t *m = (am_map_t *)map;

    if (!m) return SIZE_MAX;

    size_t pos = offset;
    if (buffer != NULL && offset != SIZE_MAX) {
        am_disk_write_base(buffer, pos, &m->base);
    }
    pos += AM_DISK_BASE_SIZE;
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)m->length);

    for (size_t i = 0; i < m->capacity; i++) {
        if (m->slots[i].key != AM_MAP_KEY_EMPTY && m->slots[i].key != AM_MAP_KEY_TOMBSTONE) {
            pos += am_disk_write_value(buffer, pos, m->slots[i].key);
            pos += am_disk_write_value(buffer, pos, m->slots[i].value);
        }
    }

    return pos - offset;
}


// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的散列表对象，构造散列表对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_map_t对象的指针，失败则返回NULL。
am_map_t *am_map_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset) {
    if (!alloc || !buffer) return NULL;

    size_t pos = offset;
    am_object_t base;
    am_disk_read_base(buffer, pos, &base);
    pos += AM_DISK_BASE_SIZE;
    if (base.type != AM_OBJECT_TYPE_MAP) return NULL;

    uint64_t length = 0;
    size_t n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &length))) return NULL;
    pos += n;
    if (length > (uint64_t)(SIZE_MAX / sizeof(am_map_entry_t))) return NULL;

    // 重新构造一个功能完整的散列表（capacity 取不小于 length 的 2 的幂，留有空槽）。
    am_map_t *map = am_map_create(alloc, (size_t)length);
    if (!map) return NULL;
    map->base = base;

    for (size_t i = 0; i < (size_t)length; i++) {
        am_value_t key = 0, value = 0;
        if (!(n = am_disk_read_value(buffer, pos, &key))) goto fail;
        pos += n;
        if (!(n = am_disk_read_value(buffer, pos, &value))) goto fail;
        pos += n;
        if (key == AM_MAP_KEY_EMPTY || key == AM_MAP_KEY_TOMBSTONE) goto fail;

        am_map_t *new_map = am_map_set(alloc, map, key, value);
        if (!new_map) goto fail;
        map = new_map;
    }

    return map;

fail:
    // 清空所有 value，避免 am_map_destroy 误释放尚未加载的指针对象
    for (size_t i = 0; i < map->capacity; i++) {
        map->slots[i].value = AM_VALUE_NULL;
    }
    am_map_destroy(alloc, map);
    return NULL;
}


// ===============================================================================
// 基本操作
// ===============================================================================

// 查找：返回对应的 value；若不存在返回 AM_VALUE_NULL
am_value_t am_map_get(am_allocator_t *alloc, am_map_t *map, am_value_t key) {
    (void)alloc;
    am_map_t *m = (am_map_t *)map;
    if (m->length == 0) return AM_VALUE_NULL;

    size_t idx;
    if (am_map_find_slot(m, key, &idx) < 0) return AM_VALUE_NULL;
    return m->slots[idx].value;
}

// 存在性检查：存在返回 0，不存在返回 -1
int32_t am_map_contains(am_allocator_t *alloc, am_map_t *map, am_value_t key) {
    (void)alloc;
    am_map_t *m = (am_map_t *)map;
    if (m->length == 0) return -1;

    size_t idx;
    return am_map_find_slot(m, key, &idx) < 0 ? -1 : 0;
}

// 不扩容地插入或修改（stable 版本）。
// 仅做插入/替换，绝不分配或释放 map 对象本身，因此 map 指针保持稳定。
// 若 map 已满且 key 不存在，返回 -1；成功返回 0。
// 替换已存在的 key 时，会释放旧的指针 value。
int32_t am_map_set_stable(am_allocator_t *alloc, am_map_t *map, am_value_t key, am_value_t value) {
    if (key == AM_MAP_KEY_EMPTY || key == AM_MAP_KEY_TOMBSTONE) return -1;

    am_map_t *m = (am_map_t *)map;

    // 表已完全填满（无空槽、无墓碑）：只能替换已有 key。
    if (m->length == m->capacity) {
        size_t idx = am_value_hash(key) & m->mask;
        for (size_t i = 0; i < m->capacity; i++) {
            am_value_t k = m->slots[idx].key;
            if (am_value_equal(k, key)) {
                if (am_value_is_ptr(m->slots[idx].value)) {
                    am_free(alloc, am_value_to_ptr(m->slots[idx].value));
                }
                m->slots[idx].value = value;
                return 0;
            }
            idx = (idx + 1) & m->mask;
        }
        return -1; // 表满且 key 不存在
    }

    // 存在空槽或墓碑，find_slot 必然终止
    size_t idx;
    int32_t found = am_map_find_slot(m, key, &idx);
    if (found >= 0) {
        if (am_value_is_ptr(m->slots[idx].value)) {
            am_free(alloc, am_value_to_ptr(m->slots[idx].value));
        }
        m->slots[idx].value = value;
    } else {
        if (m->slots[idx].key == AM_MAP_KEY_TOMBSTONE) {
            m->tombstones--;
        }
        m->slots[idx].key = key;
        m->slots[idx].value = value;
        m->length++;
    }
    return 0;
}

// 插入或修改。
// 插入新键值对；若 key 已存在则替换 value，并释放旧的指针 value。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的 map 对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有 map 指针。
am_map_t *am_map_set(am_allocator_t *alloc, am_map_t *map, am_value_t key, am_value_t value) {
    if (key == AM_MAP_KEY_EMPTY || key == AM_MAP_KEY_TOMBSTONE) return NULL;

    am_map_t *m = (am_map_t *)map;

    // 负载因子超过 75% 时扩容
    if ((m->length + m->tombstones + 1) * 4 > m->capacity * 3) {
        am_map_t *new_map = am_map_resize(alloc, map, m->capacity * 2);
        if (!new_map) return NULL;
        map = new_map;
        m = (am_map_t *)map;
    }

    size_t idx;
    int32_t found = am_map_find_slot(m, key, &idx);
    if (found >= 0) {
        if (am_value_is_ptr(m->slots[idx].value)) {
            am_free(alloc, am_value_to_ptr(m->slots[idx].value));
        }
        m->slots[idx].value = value;
    } else {
        if (m->slots[idx].key == AM_MAP_KEY_TOMBSTONE) {
            m->tombstones--;
        }
        m->slots[idx].key = key;
        m->slots[idx].value = value;
        m->length++;
    }
    return map;
}

// 删除指定 key。若存在且 value 为指针则释放。
// 删除成功返回 0；key 不存在返回 -1。
int32_t am_map_delete(am_allocator_t *alloc, am_map_t *map, am_value_t key) {
    am_map_t *m = (am_map_t *)map;
    if (m->length == 0) return -1;

    size_t idx;
    if (am_map_find_slot(m, key, &idx) < 0) return -1;

    if (am_value_is_ptr(m->slots[idx].value)) {
        am_free(alloc, am_value_to_ptr(m->slots[idx].value));
    }
    m->slots[idx].key = AM_MAP_KEY_TOMBSTONE;
    m->slots[idx].value = AM_VALUE_NULL;
    m->length--;
    m->tombstones++;

    // 墓碑过多时原地重哈希（失败则 map 未被修改，继续保留墓碑）
    if (m->tombstones * 2 > m->capacity) {
        if (am_map_rehash(alloc, map) != 0) {
            // 内存不足，重哈希失败；删除操作本身已完成
        }
    }
    return 0;
}

// 当前有效键值对数量
size_t am_map_length(am_allocator_t *alloc, am_map_t *map) {
    (void)alloc;
    return ((am_map_t *)map)->length;
}

// 物理槽位数
size_t am_map_capacity(am_allocator_t *alloc, am_map_t *map) {
    (void)alloc;
    return ((am_map_t *)map)->capacity;
}

// ===============================================================================
// 遍历与键列表
// ===============================================================================

// 遍历所有有效键值对，调用回调 cb
void am_map_iter(am_allocator_t *alloc, am_map_t *map, am_map_iter_callback_t cb, void *user_data) {
    (void)alloc;
    if (!cb) return;
    am_map_t *m = (am_map_t *)map;
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->slots[i].key != AM_MAP_KEY_EMPTY && m->slots[i].key != AM_MAP_KEY_TOMBSTONE) {
            cb(m->slots[i].key, m->slots[i].value, user_data);
        }
    }
}

// 获取所有 key 的副本列表，使用 allocator 分配。
// 调用者负责使用 am_free(alloc, ...) 释放返回的指针；size 为 0 时返回 NULL。
am_value_t *am_map_keys(am_allocator_t *alloc, am_map_t *map) {
    if (!alloc || !map) return NULL;
    am_map_t *m = (am_map_t *)map;
    if (m->length == 0) return NULL;

    am_value_t *keys = (am_value_t *)am_malloc(alloc, m->length * sizeof(am_value_t));
    if (!keys) return NULL;

    size_t count = 0;
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->slots[i].key != AM_MAP_KEY_EMPTY && m->slots[i].key != AM_MAP_KEY_TOMBSTONE) {
            keys[count++] = m->slots[i].key;
        }
    }
    return keys;
}
/* ===== end:   src/am_map.c ===== */

/* ===== begin: src/am_list.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>



// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 将列表扩容到新容量，返回新的列表对象指针；失败返回 NULL。
// 原列表对象会被释放，调用者必须使用返回的新指针。
static am_list_t *am_list_resize(am_allocator_t *alloc, am_list_t *lst, size_t new_capacity) {
    if (new_capacity < lst->length) new_capacity = lst->length;

    size_t total_size = sizeof(am_list_t) + new_capacity * sizeof(am_value_t);
    am_list_t *new_lst = (am_list_t *)am_malloc(alloc, total_size);
    if (!new_lst) return NULL;

    new_lst->base = lst->base;
    new_lst->capacity = new_capacity;
    new_lst->length = lst->length;
    new_lst->type = lst->type;
    new_lst->parent = lst->parent;

    if (lst->length > 0) {
        memcpy(new_lst->children, lst->children, lst->length * sizeof(am_value_t));
    }

    am_free(alloc, lst);
    return new_lst;
}


// 若空间不足则扩容。绝大多数情况下不会触发实际分配。
// 返回原指针或新指针；失败返回 NULL。
static am_list_t *am_list_grow_if_needed(am_allocator_t *alloc, am_list_t *lst) {
    if (lst->length < lst->capacity) return lst;

    size_t new_capacity = lst->capacity * 2;
    if (new_capacity < 4) new_capacity = 4;
    return am_list_resize(alloc, lst, new_capacity);
}


// ===============================================================================
// 构造函数
// ===============================================================================

am_list_t *am_list_create(am_allocator_t *alloc, size_t capacity, int32_t type, am_handle_t parent) {
    if (capacity < 4) capacity = 4;

    size_t total_size = sizeof(am_list_t) + capacity * sizeof(am_value_t);
    am_list_t *lst = (am_list_t *)am_calloc(alloc, total_size);
    if (!lst) return NULL;

    lst->base.type = AM_OBJECT_TYPE_LIST;
    lst->capacity = capacity;
    lst->length = 0;
    lst->type = type;
    lst->parent = parent;

    return lst;
}


// ===============================================================================
// 析构
// ===============================================================================

int32_t am_list_destroy(am_allocator_t *alloc, am_list_t *lst) {
    if (!lst) return 0;
    am_free(alloc, lst);
    return 0;
}


// ===============================================================================
// 拷贝
// ===============================================================================

am_list_t *am_list_copy(am_allocator_t *alloc, am_list_t *lst) {
    if (!lst) return NULL;

    am_list_t *copy = am_list_create(alloc, lst->capacity, lst->type, lst->parent);
    if (!copy) return NULL;

    copy->base = lst->base;
    copy->length = lst->length;
    if (lst->length > 0) {
        memcpy(copy->children, lst->children, lst->length * sizeof(am_value_t));
    }
    return copy;
}


// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_list_size(am_allocator_t *alloc, am_list_t *obj) {
    (void)alloc;
    if (!obj) return SIZE_MAX;

    if (obj->capacity > (SIZE_MAX - sizeof(am_list_t)) / sizeof(am_value_t)) {
        return SIZE_MAX;
    }
    return sizeof(am_list_t) + obj->capacity * sizeof(am_value_t);
}


// ===============================================================================
// 遍历
// ===============================================================================

void am_list_iter(am_allocator_t *alloc, am_list_t *lst, am_list_iter_callback_t cb, void *user_data) {
    (void)alloc;
    if (!lst || !cb) return;
    for (size_t i = 0; i < lst->length; i++) {
        cb(i, lst->children[i], user_data);
    }
}


// ===============================================================================
// 对象二进制转储
// ===============================================================================

// 功能说明：将列表对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分。
// 磁盘格式（平台无关固定宽度，小端；详见 include/object.h）：
//   [16B] 对象基类头（type=AM_OBJECT_TYPE_LIST）
//   [uvarint] length（capacity 压缩为与 length 一致，不落盘）
//   [uvarint] List 子类型（AM_LIST_TYPE_*）
//   [uvarint] parent 把柄
//   [length * dvalue] children（每个元素为变长编码的 TPV）
size_t am_list_dump(am_allocator_t *alloc, am_list_t *lst, uint8_t *buffer, size_t offset) {
    (void)alloc;
    if (!lst) return SIZE_MAX;

    size_t pos = offset;
    if (buffer != NULL && offset != SIZE_MAX) {
        am_disk_write_base(buffer, pos, &lst->base);
    }
    pos += AM_DISK_BASE_SIZE;

    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)lst->length);
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)(uint32_t)lst->type);
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)lst->parent);

    for (size_t i = 0; i < lst->length; i++) {
        pos += am_disk_write_value(buffer, pos, lst->children[i]);
    }

    return pos - offset;
}


// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的列表对象，构造列表对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_list_t对象的指针，失败则返回NULL。
am_list_t *am_list_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset) {
    if (!alloc || !buffer) return NULL;

    size_t pos = offset;
    am_object_t base;
    am_disk_read_base(buffer, pos, &base);
    pos += AM_DISK_BASE_SIZE;
    if (base.type != AM_OBJECT_TYPE_LIST) return NULL;

    uint64_t length = 0, type = 0, parent = 0;
    size_t n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &length))) return NULL;
    pos += n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &type))) return NULL;
    pos += n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &parent))) return NULL;
    pos += n;

    // 长度与本宿主字长适配性检查
    if (length > (uint64_t)((SIZE_MAX - sizeof(am_list_t)) / sizeof(am_value_t))) return NULL;
    if (parent > (uint64_t)AM_HANDLE_NULL) return NULL;

    am_list_t *lst = am_list_create(alloc, (size_t)length, (int32_t)type, (am_handle_t)parent);
    if (!lst) return NULL;
    lst->base = base;
    lst->length = (size_t)length;

    for (size_t i = 0; i < lst->length; i++) {
        am_value_t v = 0;
        if (!(n = am_disk_read_value(buffer, pos, &v))) {
            am_free(alloc, lst);
            return NULL;
        }
        pos += n;
        lst->children[i] = v;
    }

    return lst;
}


// ===============================================================================
// 基本操作
// ===============================================================================

am_value_t am_list_get(am_allocator_t *alloc, am_list_t *lst, size_t index) {
    (void)alloc;
    if (!lst || index >= lst->length) return AM_VALUE_UNDEFINED;
    return lst->children[index];
}


int32_t am_list_set(am_allocator_t *alloc, am_list_t *lst, size_t index, am_value_t item) {
    (void)alloc;
    if (!lst || index >= lst->length) return -1;
    lst->children[index] = item;
    return 0;
}


am_list_t *am_list_push(am_allocator_t *alloc, am_list_t *lst, am_value_t item) {
    if (!lst) return NULL;

    lst = am_list_grow_if_needed(alloc, lst);
    if (!lst) return NULL;

    lst->children[lst->length++] = item;
    return lst;
}


am_value_t am_list_pop(am_allocator_t *alloc, am_list_t *lst) {
    (void)alloc;
    if (!lst || lst->length == 0) return AM_VALUE_UNDEFINED;
    return lst->children[--lst->length];
}


am_value_t am_list_shift(am_allocator_t *alloc, am_list_t *lst) {
    (void)alloc;
    if (!lst || lst->length == 0) return AM_VALUE_UNDEFINED;

    am_value_t first = lst->children[0];
    for (size_t i = 1; i < lst->length; i++) {
        lst->children[i - 1] = lst->children[i];
    }
    lst->length--;
    return first;
}


size_t am_list_find(am_allocator_t *alloc, am_list_t *lst, am_value_t item, size_t from_index) {
    (void)alloc;
    if (!lst || from_index >= lst->length) return SIZE_MAX;
    for (size_t i = from_index; i < lst->length; i++) {
        if (am_value_equal(lst->children[i], item)) return i;
    }
    return SIZE_MAX;
}


// ===============================================================================
// Lambda 表相关函数
// ===============================================================================

// Lambda表结构：children[0]='lambda, children[1]=n_param(uint), children[2..2+n)=params, children[2+n..)=bodies

static inline am_uint_t lambda_param_count(am_list_t *lambda) {
    if (!lambda || lambda->length < 2) return 0;
    am_value_t n = lambda->children[1];
    if (!am_value_is_uint(n)) return 0;
    return am_value_to_uint(n);
}


static inline void lambda_set_param_count(am_list_t *lambda, am_uint_t n) {
    if (lambda && lambda->length >= 2) {
        lambda->children[1] = am_make_value_of_uint(n);
    }
}


am_list_t *am_list_lambda_add_parameter(am_allocator_t *alloc, am_list_t *lst, am_value_t param) {
    if (!lst) return NULL;
    if (!am_value_is_varid(param)) {
        // 允许 '...' 作为形式参数出现，用于 syntax-rules 宏模板中的可变参数列表。
        if (!(am_value_is_symbol(param) && am_value_to_symbol(param) == am_value_to_symbol(AM_VALUE_KW_dot3))) {
            return NULL;
        }
    }
    if (lst->type != AM_LIST_TYPE_LAMBDA) return NULL;

    am_uint_t n_param = lambda_param_count(lst);
    size_t insert_pos = 2 + n_param;

    lst = am_list_grow_if_needed(alloc, lst);
    if (!lst) return NULL;

    // 将原有 bodies 后移一位
    for (size_t i = lst->length; i > insert_pos; i--) {
        lst->children[i] = lst->children[i - 1];
    }
    lst->children[insert_pos] = param;
    lst->length++;
    lambda_set_param_count(lst, n_param + 1);

    return lst;
}


am_list_t *am_list_lambda_add_body(am_allocator_t *alloc, am_list_t *lst, am_value_t body) {
    if (!lst || lst->type != AM_LIST_TYPE_LAMBDA) return NULL;
    return am_list_push(alloc, lst, body);
}


size_t am_list_lambda_get_body_number(am_allocator_t *alloc, am_list_t *lst) {
    (void)alloc;
    if (!lst || lst->type != AM_LIST_TYPE_LAMBDA) return 0;
    am_uint_t n_param = lambda_param_count(lst);
    if (lst->length < 2 + n_param) return 0;
    return lst->length - 2 - n_param;
}


am_value_t *am_list_lambda_get_bodies(am_allocator_t *alloc, am_list_t *lst, size_t *n_body) {
    (void)alloc;
    if (!lst || lst->type != AM_LIST_TYPE_LAMBDA) {
        if (n_body) *n_body = 0;
        return NULL;
    }

    am_uint_t n_param = lambda_param_count(lst);
    size_t body_start = 2 + n_param;
    size_t body_count = (lst->length > body_start) ? (lst->length - body_start) : 0;

    if (n_body) *n_body = body_count;
    if (body_count == 0) return NULL;

    am_value_t *bodies = (am_value_t *)malloc(body_count * sizeof(am_value_t));
    if (!bodies) {
        if (n_body) *n_body = 0;
        return NULL;
    }

    memcpy(bodies, &lst->children[body_start], body_count * sizeof(am_value_t));
    return bodies;
}


am_list_t *am_list_lambda_set_bodies(am_allocator_t *alloc, am_list_t *lst, am_value_t *bodies, size_t *n_body) {
    if (!lst || lst->type != AM_LIST_TYPE_LAMBDA || !bodies || !n_body) return NULL;

    am_uint_t n_param = lambda_param_count(lst);
    size_t body_count = *n_body;
    size_t new_length = 2 + n_param + body_count;

    // 若容量不足则扩容
    if (new_length > lst->capacity) {
        size_t new_capacity = lst->capacity * 2;
        while (new_capacity < new_length) new_capacity *= 2;
        lst = am_list_resize(alloc, lst, new_capacity);
        if (!lst) return NULL;
    }

    // 覆盖 bodies
    for (size_t i = 0; i < body_count; i++) {
        lst->children[2 + n_param + i] = bodies[i];
    }
    lst->length = new_length;

    return lst;
}
/* ===== end:   src/am_list.c ===== */

/* ===== begin: src/am_wstring.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>



// 创建并初始化一个字符串对象。字符串对象是不可变的。
// 注意：am_wstring_t.content是am_value_t数组，每个元素是一个am_wchar_t。
am_wstring_t *am_wstring_create(am_allocator_t *alloc, wchar_t *str, size_t length) {
    if (!alloc || !str) return NULL;

    am_wstring_t *ws = (am_wstring_t *)am_malloc(alloc, sizeof(am_wstring_t) + length * sizeof(am_value_t));
    if (!ws) return NULL;

    ws->base.header = 0;
    ws->base.hash = 0;
    ws->base.gcmark = 0;
    ws->base.type = AM_OBJECT_TYPE_WSTRING;
    ws->length = length;
    for (size_t i = 0; i < length; i++) {
        ws->content[i] = am_make_value_of_wchar((am_wchar_t)str[i]);
    }
    return ws;
}


// 销毁对象。obj 为 NULL 时视为成功。成功返回 0，失败返回 -1。
int32_t am_wstring_destroy(am_allocator_t *alloc, am_wstring_t *obj) {
    if (!obj) return 0;
    if (!alloc) return -1;
    am_free(alloc, obj);
    return 0;
}


// 功能说明：拷贝wstring对象。成功则返回新副本对象的指针，失败则返回NULL。
am_wstring_t *am_wstring_copy(am_allocator_t *alloc, am_wstring_t *obj) {
    if (!alloc || !obj) return NULL;

    size_t total_size = sizeof(am_wstring_t) + obj->length * sizeof(am_value_t);
    am_wstring_t *copy = (am_wstring_t *)am_malloc(alloc, total_size);
    if (!copy) return NULL;

    copy->base = obj->base;
    copy->length = obj->length;
    if (obj->length > 0) {
        memcpy(copy->content, obj->content, obj->length * sizeof(am_value_t));
    }
    return copy;
}


// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_wstring_size(am_allocator_t *alloc, am_wstring_t *obj) {
    (void)alloc;
    if (!obj) return SIZE_MAX;

    if (obj->length > (SIZE_MAX - sizeof(am_wstring_t)) / sizeof(am_value_t)) {
        return SIZE_MAX;
    }
    return sizeof(am_wstring_t) + obj->length * sizeof(am_value_t);
}


// 磁盘格式（平台无关固定宽度，小端；详见 include/object.h）：
//   [16B] 对象基类头（type=AM_OBJECT_TYPE_WSTRING）
//   [uvarint] length（字符个数）
//   [length * uvarint] 字符内容（每个字符以其 Unicode 码点存储，省去逐字符类型标签）

// 功能说明：将字符串对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
size_t am_wstring_dump(am_allocator_t *alloc, am_wstring_t *obj, uint8_t *buffer, size_t offset) {
    (void)alloc;
    if (!obj) return SIZE_MAX;

    size_t pos = offset;
    if (buffer != NULL && offset != SIZE_MAX) {
        am_disk_write_base(buffer, pos, &obj->base);
    }
    pos += AM_DISK_BASE_SIZE;
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)obj->length);

    for (size_t i = 0; i < obj->length; i++) {
        am_value_t ch = obj->content[i];
        if (!am_value_is_wchar(ch)) return SIZE_MAX;
        pos += am_disk_write_uvarint(buffer, pos, (uint64_t)am_value_to_wchar(ch));
    }

    return pos - offset;
}


// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的字符串对象，构造字符串对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_wstring_t对象的指针，失败则返回NULL。
am_wstring_t *am_wstring_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset) {
    if (!alloc || !buffer) return NULL;

    size_t pos = offset;
    am_object_t base;
    am_disk_read_base(buffer, pos, &base);
    pos += AM_DISK_BASE_SIZE;
    if (base.type != AM_OBJECT_TYPE_WSTRING) return NULL;

    uint64_t length = 0;
    size_t n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &length))) return NULL;
    pos += n;
    if (length > (uint64_t)((SIZE_MAX - sizeof(am_wstring_t)) / sizeof(am_value_t))) return NULL;

    am_wstring_t *ws = (am_wstring_t *)am_malloc(alloc, sizeof(am_wstring_t) + (size_t)length * sizeof(am_value_t));
    if (!ws) return NULL;

    ws->base = base;
    ws->length = (size_t)length;
    for (size_t i = 0; i < ws->length; i++) {
        uint64_t cp = 0;
        if (!(n = am_disk_read_uvarint(buffer, pos, &cp))) {
            am_free(alloc, ws);
            return NULL;
        }
        pos += n;
        if (cp > (uint64_t)0x10FFFF) { // Unicode 码点上界
            am_free(alloc, ws);
            return NULL;
        }
        ws->content[i] = am_make_value_of_wchar((am_wchar_t)cp);
    }
    return ws;
}


// ===============================================================================
// 多值字符串索引表 am_strindex_t 实现
// ===============================================================================

// FNV-1a 32-bit 参数
#define AM_FNV1A_OFFSET_BASIS ((uint32_t)0x811c9dc5u)
#define AM_FNV1A_PRIME        ((uint32_t)0x01000193u)

// 计算 wchar_t 字符串的 FNV-1a 32-bit 哈希值
uint32_t am_strindex_hash_string(const wchar_t *str) {
    uint32_t hash = AM_FNV1A_OFFSET_BASIS;
    while (*str != L'\0') {
        hash ^= (uint32_t)(*str);
        hash *= AM_FNV1A_PRIME;
        str++;
    }
    return hash;
}

// 内部静态别名，保持现有代码风格
static uint32_t am_strindex_hash(const wchar_t *str) {
    return am_strindex_hash_string(str);
}

// 将容量向上取整为不小于它的最小 2 的幂
static size_t am_strindex_round_up_capacity(size_t capacity) {
    size_t cap = 1;
    while (cap < capacity) cap <<= 1;
    return cap;
}

// 查找可插入槽位：从 hash 对应位置开始，返回第一个 EMPTY 或 TOMBSTONE 槽的索引
static size_t am_strindex_find_insert_slot(const am_strindex_t *si, uint32_t hash) {
    size_t idx = (size_t)hash & si->mask;
    while (si->slots[idx].hash != AM_STRINDEX_KEY_EMPTY &&
           si->slots[idx].hash != AM_STRINDEX_KEY_TOMBSTONE) {
        idx = (idx + 1) & si->mask;
    }
    return idx;
}

// 收集指定 hash 对应的所有 value。
// values 为 NULL 或 n_values 为 0 时仅计数；否则最多写入 n_values 个。
// 返回实际匹配数量。
static size_t am_strindex_collect_values(const am_strindex_t *si, uint32_t hash,
                                         am_value_t *values, size_t n_values) {
    size_t idx = (size_t)hash & si->mask;
    size_t count = 0;
    while (si->slots[idx].hash != AM_STRINDEX_KEY_EMPTY) {
        if (si->slots[idx].hash == hash) {
            if (values != NULL && count < n_values) {
                values[count] = si->slots[idx].value;
            }
            count++;
        }
        idx = (idx + 1) & si->mask;
    }
    return count;
}

// 原地重哈希：清除墓碑，不改变容量
static int32_t am_strindex_rehash(am_allocator_t *alloc, am_strindex_t *si) {
    size_t cap = si->capacity;
    size_t entries_size = cap * sizeof(am_strindex_entry_t);

    am_strindex_entry_t *old_slots = (am_strindex_entry_t *)am_malloc(alloc, entries_size);
    if (!old_slots) return -1;
    memcpy(old_slots, si->slots, entries_size);

    for (size_t i = 0; i < cap; i++) {
        si->slots[i].hash = AM_STRINDEX_KEY_EMPTY;
        si->slots[i].value = AM_VALUE_NULL;
    }
    si->length = 0;
    si->tombstones = 0;

    for (size_t i = 0; i < cap; i++) {
        if (old_slots[i].hash != AM_STRINDEX_KEY_EMPTY &&
            old_slots[i].hash != AM_STRINDEX_KEY_TOMBSTONE) {
            size_t insert_idx = am_strindex_find_insert_slot(si, old_slots[i].hash);
            si->slots[insert_idx].hash = old_slots[i].hash;
            si->slots[insert_idx].value = old_slots[i].value;
            si->length++;
        }
    }

    am_free(alloc, old_slots);
    return 0;
}

// 扩容并重哈希到新容量（new_capacity 会被向上取整为 2 的幂）。
// 返回新的 strindex 对象指针；失败返回 NULL。原对象会被释放，调用者必须使用返回的新指针。
static am_strindex_t *am_strindex_resize(am_allocator_t *alloc, am_strindex_t *si, size_t new_capacity) {
    size_t cap = am_strindex_round_up_capacity(new_capacity);
    if (cap < 8) cap = 8;

    if (cap <= si->capacity) {
        if (am_strindex_rehash(alloc, si) != 0) return NULL;
        return si;
    }

    size_t old_capacity = si->capacity;
    size_t old_entries_size = old_capacity * sizeof(am_strindex_entry_t);

    am_strindex_entry_t *old_slots = NULL;
    if (old_entries_size > 0) {
        old_slots = (am_strindex_entry_t *)am_malloc(alloc, old_entries_size);
        if (!old_slots) return NULL;
        memcpy(old_slots, si->slots, old_entries_size);
    }

    size_t new_total_size = sizeof(am_strindex_t) + cap * sizeof(am_strindex_entry_t);
    am_strindex_t *new_si = (am_strindex_t *)am_malloc(alloc, new_total_size);
    if (!new_si) {
        am_free(alloc, old_slots);
        return NULL;
    }

    new_si->base = si->base;
    new_si->capacity = cap;
    new_si->mask = cap - 1;
    new_si->length = 0;
    new_si->tombstones = 0;

    for (size_t i = 0; i < cap; i++) {
        new_si->slots[i].hash = AM_STRINDEX_KEY_EMPTY;
        new_si->slots[i].value = AM_VALUE_NULL;
    }

    for (size_t i = 0; i < old_capacity; i++) {
        if (old_slots[i].hash != AM_STRINDEX_KEY_EMPTY &&
            old_slots[i].hash != AM_STRINDEX_KEY_TOMBSTONE) {
            size_t insert_idx = am_strindex_find_insert_slot(new_si, old_slots[i].hash);
            new_si->slots[insert_idx].hash = old_slots[i].hash;
            new_si->slots[insert_idx].value = old_slots[i].value;
            new_si->length++;
        }
    }

    am_free(alloc, old_slots);
    am_free(alloc, si);
    return new_si;
}

// ===============================================================================
// 构造函数
// ===============================================================================

// 以初始容量新建多值哈希表。capacity 会被向上取整为不小于它的最小 2 的幂。
// 所有 key 初始化为 AM_STRINDEX_KEY_EMPTY，value 初始化为 AM_VALUE_NULL。
am_strindex_t *am_strindex_create(am_allocator_t *alloc, size_t capacity) {
    if (!alloc) return NULL;

    size_t cap = am_strindex_round_up_capacity(capacity);
    if (cap < 8) cap = 8;

    size_t total_size = sizeof(am_strindex_t) + cap * sizeof(am_strindex_entry_t);
    am_strindex_t *si = (am_strindex_t *)am_malloc(alloc, total_size);
    if (!si) return NULL;

    memset(si, 0, total_size);

    si->base.type = AM_OBJECT_TYPE_STRINDEX;
    si->capacity = cap;
    si->mask = cap - 1;
    si->length = 0;
    si->tombstones = 0;

    for (size_t i = 0; i < cap; i++) {
        si->slots[i].hash = AM_STRINDEX_KEY_EMPTY;
        si->slots[i].value = AM_VALUE_NULL;
    }

    return si;
}

// ===============================================================================
// 析构与清理
// ===============================================================================

// 彻底销毁
int32_t am_strindex_destroy(am_allocator_t *alloc, am_strindex_t *obj) {
    if (!obj) return 0;
    if (!alloc) return -1;
    am_free(alloc, obj);
    return 0;
}

// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝：创建并返回一个与原 strindex 内容完全一致的新对象。所有 key/value 按位拷贝。
am_strindex_t *am_strindex_copy(am_allocator_t *alloc, am_strindex_t *obj) {
    if (!alloc || !obj) return NULL;

    am_strindex_t *copy = am_strindex_create(alloc, obj->capacity);
    if (!copy) return NULL;

    copy->length = obj->length;
    copy->tombstones = obj->tombstones;

    for (size_t i = 0; i < obj->capacity; i++) {
        copy->slots[i].hash = obj->slots[i].hash;
        copy->slots[i].value = obj->slots[i].value;
    }

    return copy;
}

// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_strindex_size(am_allocator_t *alloc, am_strindex_t *obj) {
    (void)alloc;
    if (!obj) return SIZE_MAX;

    if (obj->capacity > (SIZE_MAX - sizeof(am_strindex_t)) / sizeof(am_strindex_entry_t)) {
        return SIZE_MAX;
    }
    return sizeof(am_strindex_t) + obj->capacity * sizeof(am_strindex_entry_t);
}

// ===============================================================================
// 对象二进制转储
// ===============================================================================

// 功能说明：将表对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，丢弃墓碑和空闲槽位。
// 磁盘格式（平台无关固定宽度，小端；详见 include/object.h）：
//   [16B] 对象基类头（type=AM_OBJECT_TYPE_STRINDEX）
//   [uvarint] length（有效表项数量；capacity/墓碑/空槽均不落盘）
//   [length * (u32 hash, dvalue value)] 有效表项
size_t am_strindex_dump(am_allocator_t *alloc, am_strindex_t *obj, uint8_t *buffer, size_t offset) {
    (void)alloc;
    if (!obj) return SIZE_MAX;

    size_t pos = offset;
    if (buffer != NULL && offset != SIZE_MAX) {
        am_disk_write_base(buffer, pos, &obj->base);
    }
    pos += AM_DISK_BASE_SIZE;
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)obj->length);

    for (size_t i = 0; i < obj->capacity; i++) {
        if (obj->slots[i].hash != AM_STRINDEX_KEY_EMPTY &&
            obj->slots[i].hash != AM_STRINDEX_KEY_TOMBSTONE) {
            if (buffer != NULL && offset != SIZE_MAX) {
                am_disk_write_u32(buffer, pos, obj->slots[i].hash);
            }
            pos += 4;
            pos += am_disk_write_value(buffer, pos, obj->slots[i].value);
        }
    }

    return pos - offset;
}

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的对象，构造对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_strindex_t对象的指针，失败则返回NULL。
am_strindex_t *am_strindex_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset) {
    if (!alloc || !buffer) return NULL;

    size_t pos = offset;
    am_object_t base;
    am_disk_read_base(buffer, pos, &base);
    pos += AM_DISK_BASE_SIZE;
    if (base.type != AM_OBJECT_TYPE_STRINDEX) return NULL;

    uint64_t length = 0;
    size_t n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &length))) return NULL;
    pos += n;
    if (length > (uint64_t)(SIZE_MAX / sizeof(am_strindex_entry_t))) return NULL;

    // dump 中 capacity 与 length 一致，创建功能表时使用稍大的容量，确保有空槽。
    am_strindex_t *si = am_strindex_create(alloc, (size_t)length);
    if (!si) return NULL;
    si->base = base;

    // 直接从 dump 的 hash/value 重建，不重新计算字符串 hash。
    for (size_t i = 0; i < (size_t)length; i++) {
        uint32_t hash = am_disk_read_u32(buffer, pos);
        pos += 4;
        am_value_t value = 0;
        if (!(n = am_disk_read_value(buffer, pos, &value))) {
            am_strindex_destroy(alloc, si);
            return NULL;
        }
        pos += n;

        if (hash == AM_STRINDEX_KEY_EMPTY || hash == AM_STRINDEX_KEY_TOMBSTONE) {
            am_strindex_destroy(alloc, si);
            return NULL;
        }

        if ((si->length + si->tombstones + 1) * 4 > si->capacity * 3) {
            am_strindex_t *new_si = am_strindex_resize(alloc, si, si->capacity * 2);
            if (!new_si) {
                am_strindex_destroy(alloc, si);
                return NULL;
            }
            si = new_si;
        }

        size_t insert_idx = am_strindex_find_insert_slot(si, hash);
        if (si->slots[insert_idx].hash == AM_STRINDEX_KEY_TOMBSTONE) {
            si->tombstones--;
        }
        si->slots[insert_idx].hash = hash;
        si->slots[insert_idx].value = value;
        si->length++;
    }

    return si;
}

// ===============================================================================
// 基本操作
// ===============================================================================

// 查找：输入一个wchar_t字符串，计算其uint32_t哈希值，得到所有对应的value的列表（values由调用者管理）。
// values 为 NULL 或 n_values 为 0 时，仅返回匹配条目的数量，不写入 values。
// 返回值为实际匹配条目数量；若不存在则返回 0；若出错则返回 SIZE_MAX。
size_t am_strindex_get_all(am_allocator_t *alloc, am_strindex_t *obj, wchar_t *str,
                           am_value_t *values, size_t n_values) {
    if (!alloc || !obj || !str) return SIZE_MAX;

    uint32_t hash = am_strindex_hash(str);
    return am_strindex_collect_values(obj, hash, values, n_values);
}

// 插入新键值对。对输入的字符串计算hash，插入(key=hash,handle)时，直接根据hash找到对应的桶，如果被占用，则往后寻找第一个空桶插入。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有指针。
am_strindex_t *am_strindex_set(am_allocator_t *alloc, am_strindex_t *obj, wchar_t *str, am_value_t value) {
    if (!alloc || !obj || !str) return NULL;

    uint32_t hash = am_strindex_hash(str);

    // 负载因子超过 75% 时扩容
    if ((obj->length + obj->tombstones + 1) * 4 > obj->capacity * 3) {
        am_strindex_t *new_si = am_strindex_resize(alloc, obj, obj->capacity * 2);
        if (!new_si) return NULL;
        obj = new_si;
    }

    size_t idx = am_strindex_find_insert_slot(obj, hash);
    if (obj->slots[idx].hash == AM_STRINDEX_KEY_TOMBSTONE) {
        obj->tombstones--;
    }
    obj->slots[idx].hash = hash;
    obj->slots[idx].value = value;
    obj->length++;

    return obj;
}

// 按已知 hash 直接插入 (hash, value)，不重新计算字符串 hash。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有指针。
am_strindex_t *am_strindex_set_raw(am_allocator_t *alloc, am_strindex_t *obj, uint32_t hash, am_value_t value) {
    if (!alloc || !obj) return NULL;
    if (hash == AM_STRINDEX_KEY_EMPTY || hash == AM_STRINDEX_KEY_TOMBSTONE) return NULL;

    // 负载因子超过 75% 时扩容
    if ((obj->length + obj->tombstones + 1) * 4 > obj->capacity * 3) {
        am_strindex_t *new_si = am_strindex_resize(alloc, obj, obj->capacity * 2);
        if (!new_si) return NULL;
        obj = new_si;
    }

    size_t idx = am_strindex_find_insert_slot(obj, hash);
    if (obj->slots[idx].hash == AM_STRINDEX_KEY_TOMBSTONE) {
        obj->tombstones--;
    }
    obj->slots[idx].hash = hash;
    obj->slots[idx].value = value;
    obj->length++;

    return obj;
}

// 删除指定 value（handle）所在的条目。按 value 的位模式精确匹配；删除成功返回 0；未找到返回 -1。
int32_t am_strindex_delete(am_allocator_t *alloc, am_strindex_t *obj, am_value_t value) {
    if (!alloc || !obj) return -1;

    for (size_t i = 0; i < obj->capacity; i++) {
        if (obj->slots[i].hash != AM_STRINDEX_KEY_EMPTY &&
            obj->slots[i].hash != AM_STRINDEX_KEY_TOMBSTONE &&
            obj->slots[i].value == value) {
            obj->slots[i].hash = AM_STRINDEX_KEY_TOMBSTONE;
            obj->slots[i].value = AM_VALUE_NULL;
            obj->length--;
            obj->tombstones++;

            // 墓碑过多时原地重哈希
            if (obj->tombstones * 2 > obj->capacity) {
                if (am_strindex_rehash(alloc, obj) != 0) {
                    // 内存不足，重哈希失败；删除操作本身已完成
                }
            }
            return 0;
        }
    }
    return -1;
}

// 当前有效键值对数量
size_t am_strindex_length(am_allocator_t *alloc, am_strindex_t *obj) {
    (void)alloc;
    if (!obj) return SIZE_MAX;
    return obj->length;
}

// 物理槽位数
size_t am_strindex_capacity(am_allocator_t *alloc, am_strindex_t *obj) {
    (void)alloc;
    if (!obj) return SIZE_MAX;
    return obj->capacity;
}
/* ===== end:   src/am_wstring.c ===== */

/* ===== begin: src/am_vocab.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>



// ===============================================================================
// 内部辅助函数
// ===============================================================================

static am_vocab_t *am_vocab_resize(am_allocator_t *alloc, am_vocab_t *vocab, size_t new_capacity) {
    if (new_capacity < vocab->length) new_capacity = vocab->length;

    size_t total_size = sizeof(am_vocab_t) + new_capacity * sizeof(wchar_t *);
    am_vocab_t *new_vocab = (am_vocab_t *)am_malloc(alloc, total_size);
    if (!new_vocab) return NULL;

    new_vocab->base = vocab->base;
    new_vocab->capacity = new_capacity;
    new_vocab->length = vocab->length;

    if (vocab->length > 0) {
        memcpy(new_vocab->words, vocab->words, vocab->length * sizeof(wchar_t *));
    }

    am_free(alloc, vocab);
    return new_vocab;
}


static am_vocab_t *am_vocab_grow_if_needed(am_allocator_t *alloc, am_vocab_t *vocab) {
    if (vocab->length < vocab->capacity) return vocab;

    size_t new_capacity = vocab->capacity * 2;
    if (new_capacity < 8) new_capacity = 8;
    return am_vocab_resize(alloc, vocab, new_capacity);
}


// ===============================================================================
// 构造函数
// ===============================================================================

am_vocab_t *am_vocab_create(am_allocator_t *alloc, size_t capacity) {
    if (capacity < 4) capacity = 4;

    size_t total_size = sizeof(am_vocab_t) + capacity * sizeof(wchar_t *);
    am_vocab_t *vocab = (am_vocab_t *)am_calloc(alloc, total_size);
    if (!vocab) return NULL;

    vocab->base.type = AM_OBJECT_TYPE_VOCAB;
    vocab->capacity = capacity;
    vocab->length = 0;

    return vocab;
}


// ===============================================================================
// 析构
// ===============================================================================

int32_t am_vocab_destroy(am_allocator_t *alloc, am_vocab_t *vocab) {
    if (!vocab) return 0;
    for (size_t i = 0; i < vocab->length; i++) {
        if (vocab->words[i]) am_free(alloc, vocab->words[i]);
    }
    am_free(alloc, vocab);
    return 0;
}


// ===============================================================================
// 拷贝
// ===============================================================================

am_vocab_t *am_vocab_copy(am_allocator_t *alloc, am_vocab_t *vocab) {
    if (!vocab) return NULL;

    am_vocab_t *copy = am_vocab_create(alloc, vocab->capacity);
    if (!copy) return NULL;

    copy->base = vocab->base;
    copy->length = vocab->length;

    for (size_t i = 0; i < vocab->length; i++) {
        size_t len = wcslen(vocab->words[i]);
        copy->words[i] = (wchar_t *)am_malloc(alloc, (len + 1) * sizeof(wchar_t));
        if (!copy->words[i]) {
            am_vocab_destroy(alloc, copy);
            return NULL;
        }
        wcscpy(copy->words[i], vocab->words[i]);
    }

    return copy;
}


// ===============================================================================
// 对象二进制转储
// ===============================================================================

// 功能说明：将词典对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       将words所指向的wchar_t*宽字符串依次展平拼接，各字符串之间以L'\0'为间隔符，最后一个字符串以L'\0'结束。
//       压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分。
// 磁盘格式（平台无关固定宽度，小端；详见 include/object.h）：
//   [16B] 对象基类头（type=AM_OBJECT_TYPE_VOCAB）
//   [uvarint] length（词条数；capacity 压缩为与 length 一致，不落盘）
//   [length * (uvarint 码点数, 码点0..n-1 各一个 uvarint)] 词条内容
//   说明：词条以 Unicode 码点序列存储，不使用平台相关的 wchar_t 宽度，
//         也不存储运行时指针（原实现会将 words[i] 绝对地址落盘，跨地址空间失效）。
size_t am_vocab_dump(am_allocator_t *alloc, am_vocab_t *vocab, uint8_t *buffer, size_t offset) {
    (void)alloc;
    if (!vocab) return SIZE_MAX;

    size_t pos = offset;
    if (buffer != NULL && offset != SIZE_MAX) {
        am_disk_write_base(buffer, pos, &vocab->base);
    }
    pos += AM_DISK_BASE_SIZE;
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)vocab->length);

    for (size_t i = 0; i < vocab->length; i++) {
        size_t len = wcslen(vocab->words[i]);
        pos += am_disk_write_uvarint(buffer, pos, (uint64_t)len);
        for (size_t j = 0; j < len; j++) {
            pos += am_disk_write_uvarint(buffer, pos, (uint64_t)(uint32_t)vocab->words[i][j]);
        }
    }

    return pos - offset;
}


// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的词典对象，构造词典对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_vocab_t对象的指针，失败则返回NULL。
am_vocab_t *am_vocab_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset) {
    if (!alloc || !buffer) return NULL;

    size_t pos = offset;
    am_object_t base;
    am_disk_read_base(buffer, pos, &base);
    pos += AM_DISK_BASE_SIZE;
    if (base.type != AM_OBJECT_TYPE_VOCAB) return NULL;

    uint64_t length = 0;
    size_t n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &length))) return NULL;
    pos += n;
    if (length > (uint64_t)((SIZE_MAX - sizeof(am_vocab_t)) / sizeof(wchar_t *))) return NULL;

    // 本宿主 wchar_t 可表示的码点上界（16 位 wchar_t 平台不支持代理项）
    const uint64_t cp_max = (sizeof(wchar_t) >= 4) ? (uint64_t)0x10FFFF : (uint64_t)0xFFFF;

    am_vocab_t *vocab = am_vocab_create(alloc, (size_t)length);
    if (!vocab) return NULL;
    vocab->base = base;

    for (size_t i = 0; i < (size_t)length; i++) {
        uint64_t len = 0;
        if (!(n = am_disk_read_uvarint(buffer, pos, &len))) goto fail;
        pos += n;
        if (len > (uint64_t)(SIZE_MAX / sizeof(wchar_t)) - 1) goto fail;

        wchar_t *word = (wchar_t *)am_malloc(alloc, ((size_t)len + 1) * sizeof(wchar_t));
        if (!word) goto fail;

        int ok = 1;
        for (size_t j = 0; j < (size_t)len; j++) {
            uint64_t cp = 0;
            if (!(n = am_disk_read_uvarint(buffer, pos, &cp)) || cp > cp_max) {
                ok = 0;
                break;
            }
            pos += n;
            word[j] = (wchar_t)cp;
        }
        if (!ok) {
            am_free(alloc, word);
            goto fail;
        }
        word[(size_t)len] = L'\0';
        vocab->words[i] = word;
        vocab->length++;
    }

    return vocab;

fail:
    am_vocab_destroy(alloc, vocab);
    return NULL;
}


// ===============================================================================
// 基本操作
// ===============================================================================

size_t am_vocab_find(am_allocator_t *alloc, am_vocab_t *vocab, wchar_t *word) {
    (void)alloc;
    if (!vocab || !word) return SIZE_MAX;
    for (size_t i = 0; i < vocab->length; i++) {
        if (vocab->words[i] && wcscmp(vocab->words[i], word) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}


am_vocab_t *am_vocab_insert(am_allocator_t *alloc, am_vocab_t *vocab, wchar_t *word, size_t *out_index) {
    if (!vocab || !word) return NULL;
    if (out_index) *out_index = SIZE_MAX;

    size_t existing = am_vocab_find(alloc, vocab, word);
    if (existing != SIZE_MAX) {
        if (out_index) *out_index = existing;
        return vocab;
    }

    /* 先复制待插入的字符串：如果这里分配失败，不会破坏原 vocab。
     * 注意：必须在扩容前完成，因为 am_vocab_grow_if_needed 会释放旧 vocab。 */
    size_t len = wcslen(word);
    wchar_t *word_copy = (wchar_t *)am_malloc(alloc, (len + 1) * sizeof(wchar_t));
    if (!word_copy) return NULL;
    wcscpy(word_copy, word);

    am_vocab_t *new_vocab = am_vocab_grow_if_needed(alloc, vocab);
    if (!new_vocab) {
        am_free(alloc, word_copy);
        return NULL;
    }

    new_vocab->words[new_vocab->length] = word_copy;
    if (out_index) *out_index = new_vocab->length;
    new_vocab->length++;
    return new_vocab;
}


wchar_t *am_vocab_get(am_allocator_t *alloc, am_vocab_t *vocab, size_t *index) {
    (void)alloc;
    if (!vocab || !index || *index >= vocab->length) return NULL;
    return vocab->words[*index];
}
/* ===== end:   src/am_vocab.c ===== */

/* ===== begin: src/am_heap.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


// 全局把柄计数器，保证同一进程内不同 AST 堆的把柄不冲突。
// 后续若需要严格的进程隔离，可将此计数器移入 am_heap_t 并通过模块 ID 哈希生成前缀。
static am_handle_t g_heap_handle_counter = 1;

// ===============================================================================
// 内部辅助函数：在 map 中查找 key 所在槽位（与 src/map.c 一致）
// ===============================================================================

/* 安全销毁 deep_dump 临时堆表：先把所有 value 置空，
 * 避免 am_map_destroy 把偏移量或原堆指针误当对象释放。 */
static void am_heap_temp_table_destroy(am_allocator_t *alloc, am_map_t *table) {
    if (!table) return;
    for (size_t i = 0; i < table->capacity; i++) {
        table->slots[i].value = AM_VALUE_NULL;
    }
    am_map_destroy(alloc, table);
}

static int32_t am_heap_find_slot(const am_map_t *m, am_value_t key, size_t *out_insert_idx) {
    size_t idx = am_value_hash(key) & m->mask;
    size_t insert_idx = UINT32_MAX;

    while (1) {
        am_value_t k = m->slots[idx].key;
        if (k == AM_MAP_KEY_EMPTY) {
            if (insert_idx == UINT32_MAX) insert_idx = idx;
            *out_insert_idx = insert_idx;
            return -1;
        }
        if (k == AM_MAP_KEY_TOMBSTONE) {
            if (insert_idx == UINT32_MAX) insert_idx = idx;
        } else if (am_value_equal(k, key)) {
            *out_insert_idx = idx;
            return 0;
        }
        idx = (idx + 1) & m->mask;
    }
}


// ===============================================================================
// 构造函数
// ===============================================================================

am_heap_t *am_heap_create(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, size_t capacity) {
    (void)obj_alloc;
    if (capacity < 16) capacity = 16;

    am_heap_t *heap = (am_heap_t *)am_malloc(container_alloc, sizeof(am_heap_t));
    if (!heap) return NULL;

    heap->capacity = capacity;
    heap->table = am_map_create(container_alloc, capacity);
    heap->metadata = am_map_create(container_alloc, capacity);
    heap->handle_counter = g_heap_handle_counter;

    if (!heap->table || !heap->metadata) {
        if (heap->table) am_map_destroy(container_alloc, heap->table);
        if (heap->metadata) am_map_destroy(container_alloc, heap->metadata);
        am_free(container_alloc, heap);
        return NULL;
    }

    return heap;
}


// ===============================================================================
// 析构
// ===============================================================================

int32_t am_heap_destroy(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap) {
    if (!heap) return 0;

    if (heap->table) {
        // 释放所有指针对象，并清空槽位 value，避免 am_map_delete 重复释放
        size_t count = am_map_length(container_alloc, heap->table);
        am_value_t *keys = am_map_keys(container_alloc, heap->table);
        for (size_t i = 0; i < count; i++) {
            am_value_t v = am_map_get(container_alloc, heap->table, keys[i]);
            if (am_value_is_ptr(v)) {
                am_free(obj_alloc, am_value_to_ptr(v));
                // 将槽位 value 置空，避免 map_delete 再用 container_alloc 释放对象
                size_t idx;
                if (am_heap_find_slot(heap->table, keys[i], &idx) >= 0) {
                    heap->table->slots[idx].value = AM_VALUE_NULL;
                }
            }
            am_map_delete(container_alloc, heap->table, keys[i]);
        }
        am_free(container_alloc, keys);
        am_map_destroy(container_alloc, heap->table);
    }

    if (heap->metadata) {
        am_map_destroy(container_alloc, heap->metadata);
    }

    am_free(container_alloc, heap);
    return 0;
}


// ===============================================================================
// 拷贝
// ===============================================================================

am_heap_t *am_heap_copy(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap) {
    (void)obj_alloc;
    if (!heap) return NULL;

    am_heap_t *copy = (am_heap_t *)am_malloc(container_alloc, sizeof(am_heap_t));
    if (!copy) return NULL;

    copy->handle_counter = heap->handle_counter;
    copy->table = heap->table ? am_map_copy(container_alloc, heap->table) : NULL;
    copy->metadata = heap->metadata ? am_map_copy(container_alloc, heap->metadata) : NULL;
    copy->capacity = copy->table ? copy->table->capacity : heap->capacity;

    if ((heap->table && !copy->table) || (heap->metadata && !copy->metadata)) {
        // copy 与源堆共享指针对象，失败时只释放 map 容器本身，不得释放对象
        if (copy->table) {
            for (size_t i = 0; i < copy->table->capacity; i++) {
                copy->table->slots[i].value = AM_VALUE_NULL;
            }
            am_map_destroy(container_alloc, copy->table);
        }
        if (copy->metadata) {
            for (size_t i = 0; i < copy->metadata->capacity; i++) {
                copy->metadata->slots[i].value = AM_VALUE_NULL;
            }
            am_map_destroy(container_alloc, copy->metadata);
        }
        am_free(container_alloc, copy);
        return NULL;
    }

    return copy;
}


// ===============================================================================
// 遍历
// ===============================================================================

typedef struct {
    am_heap_iter_callback_t cb;
    void *user_data;
} am_heap_iter_wrapper_t;

static void am_heap_iter_map_cb(am_value_t key, am_value_t value, void *user_data) {
    am_heap_iter_wrapper_t *ctx = (am_heap_iter_wrapper_t *)user_data;
    am_handle_t handle = am_value_to_handle(key);
    ctx->cb(handle, value, ctx->user_data);
}


void am_heap_iter(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_heap_iter_callback_t cb, void *user_data) {
    (void)obj_alloc;
    if (!heap || !heap->table || !cb) return;
    am_heap_iter_wrapper_t ctx = { cb, user_data };
    am_map_iter(container_alloc, heap->table, am_heap_iter_map_cb, &ctx);
}


// ===============================================================================
// 对象二进制转储
// ===============================================================================

// 辅助：对heap的entry按handle升序排序（用于deep_dump时保证顺序稳定）。
typedef struct {
    am_value_t key;
    am_value_t value;
    am_map_entry_t *slot;
} am_heap_entry_t;

static int am_heap_entry_compare(const void *a, const void *b) {
    const am_heap_entry_t *ea = (const am_heap_entry_t *)a;
    const am_heap_entry_t *eb = (const am_heap_entry_t *)b;
    am_handle_t ha = am_value_to_handle(ea->key);
    am_handle_t hb = am_value_to_handle(eb->key);
    if (ha < hb) return -1;
    if (ha > hb) return 1;
    return 0;
}

// 功能说明：将am_heap_t对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩底层map对象，将table和metadata的capacity压缩到跟length一致，删除多余分配的空闲部分。
// 磁盘格式（平台无关固定宽度，小端；详见 include/object.h）：
//   [uvarint] handle_counter
//   [uvarint] table_dump_size
//   [table_dump_size bytes] table 的 map 转储
//   [uvarint] metadata_dump_size（0 表示无 metadata）
//   [metadata_dump_size bytes] metadata 的 map 转储
//   说明：table/metadata 指针不再以原生指针宽度落盘，改为自描述的顺序布局；
//         capacity 可由 table 重建，不落盘。
size_t am_heap_dump(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, uint8_t *buffer, size_t offset) {
    (void)obj_alloc;
    if (!heap) return SIZE_MAX;

    size_t table_dump_size = heap->table ? am_map_dump(container_alloc, heap->table, NULL, 0) : 0;
    size_t metadata_dump_size = heap->metadata ? am_map_dump(container_alloc, heap->metadata, NULL, 0) : 0;

    if ((heap->table && table_dump_size == SIZE_MAX) ||
        (heap->metadata && metadata_dump_size == SIZE_MAX)) {
        return SIZE_MAX;
    }

    size_t pos = offset;
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)heap->handle_counter);
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)table_dump_size);

    if (heap->table) {
        size_t written = am_map_dump(container_alloc, heap->table, buffer, pos);
        if (written != table_dump_size) return SIZE_MAX;
        pos += written;
    }

    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)metadata_dump_size);
    if (heap->metadata) {
        size_t written = am_map_dump(container_alloc, heap->metadata, buffer, pos);
        if (written != metadata_dump_size) return SIZE_MAX;
        pos += written;
    }

    return pos - offset;
}


// 功能说明：am_heap_dump的逆操作。从二进制字节序列buffer[offset]开始，读取转储的heap对象，构造heap并返回其指针。
// 实现说明：成功则返回加载后am_heap_t对象的指针，失败则返回NULL。
am_heap_t *am_heap_load(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, uint8_t *buffer, size_t offset) {
    (void)obj_alloc;
    if (!container_alloc || !buffer) return NULL;

    size_t pos = offset;
    uint64_t handle_counter = 0, table_size = 0, metadata_size = 0;
    size_t n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &handle_counter))) return NULL;
    pos += n;
    if (handle_counter > (uint64_t)AM_HANDLE_NULL) return NULL;
    if (!(n = am_disk_read_uvarint(buffer, pos, &table_size))) return NULL;
    pos += n;

    am_heap_t *heap = (am_heap_t *)am_malloc(container_alloc, sizeof(am_heap_t));
    if (!heap) return NULL;

    heap->handle_counter = (am_handle_t)handle_counter;

    heap->table = am_map_load(container_alloc, buffer, pos);
    if (!heap->table) {
        am_free(container_alloc, heap);
        return NULL;
    }
    heap->capacity = heap->table->capacity;
    pos += (size_t)table_size;

    if (!(n = am_disk_read_uvarint(buffer, pos, &metadata_size))) {
        am_map_destroy(container_alloc, heap->table);
        am_free(container_alloc, heap);
        return NULL;
    }
    pos += n;

    if (metadata_size != 0) {
        heap->metadata = am_map_load(container_alloc, buffer, pos);
        if (!heap->metadata) {
            am_map_destroy(container_alloc, heap->table);
            am_free(container_alloc, heap);
            return NULL;
        }
    } else {
        heap->metadata = NULL;
    }

    return heap;
}


// 功能说明：深度转储整个heap及其指向的对象
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       仅处理value为ptr且指向AM_OBJECT_TYPE_LIST或AM_OBJECT_TYPE_WSTRING类型对象的情况。
//       词法作用域对象（AM_OBJECT_TYPE_SCOPE）仅用于编译期，不参与持久化转储。
size_t am_heap_deep_dump(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, uint8_t *buffer, size_t offset) {
    if (!container_alloc || !heap || !heap->table) return SIZE_MAX;

    // 收集需要转储的有效条目，跳过编译期作用域对象
    size_t capacity = heap->table->capacity;
    size_t count = 0;
    am_heap_entry_t *entries = NULL;
    for (size_t i = 0; i < capacity; i++) {
        am_value_t k = heap->table->slots[i].key;
        if (k == AM_MAP_KEY_EMPTY || k == AM_MAP_KEY_TOMBSTONE) continue;

        am_value_t v = heap->table->slots[i].value;
        if (!am_value_is_ptr(v)) continue;

        am_object_t *obj = am_value_to_ptr(v);
        if (obj->type == AM_OBJECT_TYPE_SCOPE) continue;

        am_heap_entry_t *new_entries = (am_heap_entry_t *)am_realloc(container_alloc, entries, (count + 1) * sizeof(am_heap_entry_t));
        if (!new_entries) {
            am_free(container_alloc, entries);
            return SIZE_MAX;
        }
        entries = new_entries;

        entries[count].key = k;
        entries[count].value = v;
        entries[count].slot = &heap->table->slots[i];
        count++;
    }

    qsort(entries, count, sizeof(am_heap_entry_t), am_heap_entry_compare);

    // 构造临时heap，仅包含需要转储的条目，避免修改原始heap
    am_heap_t temp_heap;
    temp_heap.capacity = heap->capacity;
    temp_heap.handle_counter = heap->handle_counter;
    temp_heap.table = am_map_create(container_alloc, count > 0 ? count : 1);
    if (!temp_heap.table) {
        am_free(container_alloc, entries);
        return SIZE_MAX;
    }
    temp_heap.metadata = NULL;

    for (size_t i = 0; i < count; i++) {
        am_map_t *m = am_map_set(container_alloc, temp_heap.table, entries[i].key, entries[i].value);
        if (!m) {
            am_heap_temp_table_destroy(container_alloc, temp_heap.table);
            am_free(container_alloc, entries);
            return SIZE_MAX;
        }
        temp_heap.table = m;
    }
    temp_heap.capacity = temp_heap.table->capacity;

    // 深度转储磁盘格式（平台无关固定宽度，小端）：
    //   [u32] total_size（区域总字节数，含本头部）
    //   [u32] heap_size（heap map 转储字节数）
    //   [heap_size bytes] heap 转储（table 中的 value 为对象相对区域起点的偏移量，以 PTR TPV 编码）
    //   [对象转储...] 按 handle 升序排列；每个对象起点与区域起点的距离保持偶数（必要时填充1字节），
    //                 以维持 PTR TPV 最低位为 0 的不变量。

    // 先计算每个对象的转储字节数（与偏移量无关）
    size_t *obj_sizes = (size_t *)am_malloc(container_alloc, (count > 0 ? count : 1) * sizeof(size_t));
    if (!obj_sizes) {
        am_heap_temp_table_destroy(container_alloc, temp_heap.table);
        am_free(container_alloc, entries);
        return SIZE_MAX;
    }
    for (size_t i = 0; i < count; i++) {
        am_object_t *obj = am_value_to_ptr(entries[i].value);
        switch (obj->type) {
            case AM_OBJECT_TYPE_LIST:
                obj_sizes[i] = am_list_dump(obj_alloc, (am_list_t *)obj, NULL, 0);
                break;
            case AM_OBJECT_TYPE_WSTRING:
                obj_sizes[i] = am_wstring_dump(obj_alloc, (am_wstring_t *)obj, NULL, 0);
                break;
            default:
                obj_sizes[i] = SIZE_MAX;
                break;
        }
        if (obj_sizes[i] == SIZE_MAX) {
            am_free(container_alloc, obj_sizes);
            am_heap_temp_table_destroy(container_alloc, temp_heap.table);
            am_free(container_alloc, entries);
            return SIZE_MAX;
        }
    }

    // 计算heap对象本身的dump大小
    // 注意（变长编码特有的不动点问题）：对象偏移量的取值依赖于 heap dump 的字节数，
    // 而 heap dump 的字节数又取决于 table 中偏移量变长编码的长度。
    // 此处先以原始指针（其编码长度是偏移量编码的上界）估算，再迭代至不动点；
    // 由于偏移量随 heap_map_size 减小而单调不增，迭代必然收敛。
    size_t heap_map_size = am_heap_dump(container_alloc, obj_alloc, &temp_heap, NULL, 0);
    if (heap_map_size == SIZE_MAX) {
        am_free(container_alloc, obj_sizes);
        am_heap_temp_table_destroy(container_alloc, temp_heap.table);
        am_free(container_alloc, entries);
        return SIZE_MAX;
    }

    size_t final_total = 0;
    for (;;) {
        size_t obj_offset = offset + 8 + heap_map_size;
        for (size_t i = 0; i < count; i++) {
            // 对象偏移量保持偶数（PTR TPV 标签位要求）
            if ((obj_offset - offset) & 1) obj_offset++;

            // 对象偏移量以deep_dump区域起点为基准，便于整体自描述与重定位
            am_value_t offset_value = am_make_value_of_ptr((am_object_t *)(uintptr_t)(obj_offset - offset));
            size_t idx;
            if (am_heap_find_slot(temp_heap.table, entries[i].key, &idx) >= 0) {
                temp_heap.table->slots[idx].value = offset_value;
            }
            obj_offset += obj_sizes[i];
        }

        size_t new_size = am_heap_dump(container_alloc, obj_alloc, &temp_heap, NULL, 0);
        if (new_size == SIZE_MAX) {
            am_free(container_alloc, obj_sizes);
            am_heap_temp_table_destroy(container_alloc, temp_heap.table);
            am_free(container_alloc, entries);
            return SIZE_MAX;
        }
        if (new_size == heap_map_size) {
            final_total = obj_offset - offset;
            break;
        }
        heap_map_size = new_size;
    }

    if (buffer != NULL && offset != SIZE_MAX) {
        // 将临时heap对象dump到buffer[offset+8]
        size_t written = am_heap_dump(container_alloc, obj_alloc, &temp_heap, buffer, offset + 8);
        if (written != heap_map_size) {
            am_free(container_alloc, obj_sizes);
            am_heap_temp_table_destroy(container_alloc, temp_heap.table);
            am_free(container_alloc, entries);
            return SIZE_MAX;
        }

        // 依次写入各对象
        size_t obj_offset = offset + 8 + heap_map_size;
        for (size_t i = 0; i < count; i++) {
            am_object_t *obj = am_value_to_ptr(entries[i].value);

            if ((obj_offset - offset) & 1) {
                buffer[obj_offset] = 0;
                obj_offset++;
            }

            size_t obj_size;
            if (obj->type == AM_OBJECT_TYPE_LIST) {
                obj_size = am_list_dump(obj_alloc, (am_list_t *)obj, buffer, obj_offset);
            } else {
                obj_size = am_wstring_dump(obj_alloc, (am_wstring_t *)obj, buffer, obj_offset);
            }
            if (obj_size != obj_sizes[i]) {
                am_free(container_alloc, obj_sizes);
                am_heap_temp_table_destroy(container_alloc, temp_heap.table);
                am_free(container_alloc, entries);
                return SIZE_MAX;
            }
            obj_offset += obj_size;
        }

        // 写入总字节长度和heap dump长度
        am_disk_write_u32(buffer, offset, (uint32_t)(obj_offset - offset));
        am_disk_write_u32(buffer, offset + 4, (uint32_t)heap_map_size);
    }

    am_free(container_alloc, obj_sizes);
    am_heap_temp_table_destroy(container_alloc, temp_heap.table);
    am_free(container_alloc, entries);

    return final_total;
}


// 功能说明：am_heap_deep_dump的逆操作。从二进制字节序列buffer[offset]开始，读取转储的heap及其指向的对象，构造heap并返回其指针。
// 实现说明：成功则返回加载后am_heap_t对象的指针，失败则返回NULL。
// 注意：仅处理value为ptr且指向AM_OBJECT_TYPE_LIST或AM_OBJECT_TYPE_WSTRING类型对象的情况。
am_heap_t *am_heap_deep_load(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, uint8_t *buffer, size_t offset) {
    if (!container_alloc || !obj_alloc || !buffer) return NULL;

    uint32_t total_size = am_disk_read_u32(buffer, offset);
    uint32_t heap_size = am_disk_read_u32(buffer, offset + 4);
    (void)total_size;
    (void)heap_size;

    am_heap_t *heap = am_heap_load(container_alloc, obj_alloc, buffer, offset + 8);
    if (!heap) return NULL;

    size_t count = am_map_length(container_alloc, heap->table);
    am_object_t **loaded = NULL;
    if (count > 0) {
        loaded = (am_object_t **)am_calloc(container_alloc, count * sizeof(am_object_t *));
        if (!loaded) {
            am_heap_destroy(container_alloc, obj_alloc, heap);
            return NULL;
        }
    }

    size_t idx = 0;
    for (size_t i = 0; i < heap->table->capacity; i++) {
        am_value_t k = heap->table->slots[i].key;
        if (k == AM_MAP_KEY_EMPTY || k == AM_MAP_KEY_TOMBSTONE) continue;

        am_value_t v = heap->table->slots[i].value;
        if (!am_value_is_ptr(v)) continue;

        size_t obj_rel_offset = (size_t)am_value_to_ptr(v);
        // 按字节读取对象类型（对象在转储区中仅保证偶数对齐，不能直接强制转换）
        int32_t obj_type = (int32_t)am_disk_read_u32(buffer, offset + obj_rel_offset + 12);
        am_object_t *obj = NULL;

        if (obj_type == AM_OBJECT_TYPE_LIST) {
            obj = (am_object_t *)am_list_load(obj_alloc, buffer, offset + obj_rel_offset);
        } else if (obj_type == AM_OBJECT_TYPE_WSTRING) {
            obj = (am_object_t *)am_wstring_load(obj_alloc, buffer, offset + obj_rel_offset);
        } else {
            // 不支持的类型：清理已加载对象
            for (size_t j = 0; j < idx; j++) {
                am_free(obj_alloc, loaded[j]);
            }
            am_free(container_alloc, loaded);
            // 将table中的偏移量值清空，避免am_heap_destroy误释放
            for (size_t j = 0; j < heap->table->capacity; j++) {
                heap->table->slots[j].value = AM_VALUE_NULL;
            }
            am_heap_destroy(container_alloc, obj_alloc, heap);
            return NULL;
        }

        if (!obj) {
            for (size_t j = 0; j < idx; j++) {
                am_free(obj_alloc, loaded[j]);
            }
            am_free(container_alloc, loaded);
            for (size_t j = 0; j < heap->table->capacity; j++) {
                heap->table->slots[j].value = AM_VALUE_NULL;
            }
            am_heap_destroy(container_alloc, obj_alloc, heap);
            return NULL;
        }

        heap->table->slots[i].value = am_make_value_of_ptr(obj);
        loaded[idx++] = obj;
    }

    am_free(container_alloc, loaded);
    return heap;
}


// ===============================================================================
// 把柄操作
// ===============================================================================

int32_t am_heap_has_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle) {
    (void)obj_alloc;
    if (!heap || !heap->table) return -1;
    return am_map_contains(container_alloc, heap->table, am_make_value_of_handle(handle));
}


am_handle_t am_heap_alloc_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap) {
    (void)obj_alloc;
    if (!heap || !heap->table) return AM_HANDLE_NULL;

    am_handle_t handle = heap->handle_counter++;
    g_heap_handle_counter = heap->handle_counter;
    am_value_t handle_val = am_make_value_of_handle(handle);

    am_map_t *new_table = am_map_set(container_alloc, heap->table, handle_val, AM_VALUE_NULL);
    if (!new_table) return AM_HANDLE_NULL;
    heap->table = new_table;
    heap->capacity = new_table->capacity;

    return handle;
}


int32_t am_heap_free_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle) {
    if (!heap || !heap->table) return -1;

    am_value_t handle_val = am_make_value_of_handle(handle);

    size_t idx;
    if (am_heap_find_slot(heap->table, handle_val, &idx) < 0) return -1;

    am_value_t old_val = heap->table->slots[idx].value;
    am_value_t old_ptr = AM_VALUE_NULL;
    if (am_value_is_ptr(old_val)) {
        old_ptr = old_val;
        // 清空槽位 value，避免 am_map_delete 用 container_alloc 释放对象
        heap->table->slots[idx].value = AM_VALUE_NULL;
    }

    int32_t ret = am_map_delete(container_alloc, heap->table, handle_val);
    if (ret < 0 && am_value_is_ptr(old_ptr)) {
        // 删除失败（极少发生），恢复旧值，不释放对象
        size_t idx2;
        if (am_heap_find_slot(heap->table, handle_val, &idx2) >= 0) {
            heap->table->slots[idx2].value = old_ptr;
        }
        return -1;
    }

    if (am_value_is_ptr(old_ptr)) {
        am_free(obj_alloc, am_value_to_ptr(old_ptr));
    }
    return ret;
}


int32_t am_heap_set_metadata(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle, am_uint_t property) {
    (void)container_alloc;
    (void)obj_alloc;
    (void)heap;
    (void)handle;
    (void)property;
    return 0;
}


am_uint_t am_heap_get_metadata(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle) {
    (void)container_alloc;
    (void)obj_alloc;
    (void)heap;
    (void)handle;
    return 0;
}


// ===============================================================================
// 值操作
// ===============================================================================

am_value_t am_heap_get(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle) {
    (void)obj_alloc;
    if (!heap || !heap->table) return AM_VALUE_UNDEFINED;
    return am_map_get(container_alloc, heap->table, am_make_value_of_handle(handle));
}


int32_t am_heap_set(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle, am_value_t value) {
    if (!heap || !heap->table) return -1;

    am_value_t handle_val = am_make_value_of_handle(handle);
    // 把柄必须遵循先申请后使用的原则，不允许直接创建把柄
    if (am_map_contains(container_alloc, heap->table, handle_val) < 0) {
        return -1;
    }

    // 取出旧指针值并清空槽位，避免 map 接口用 container_alloc 释放对象
    //（堆中对象由 obj_alloc 分配，所有权语义与 container_alloc 不同，
    //  因此必须先取出旧指针，再交给 am_map_set_stable 替换，最后由 obj_alloc 释放）。
    am_value_t old_ptr = AM_VALUE_NULL;
    size_t idx;
    if (am_heap_find_slot(heap->table, handle_val, &idx) >= 0) {
        am_value_t old = heap->table->slots[idx].value;
        if (am_value_is_ptr(old)) {
            old_ptr = old;
            heap->table->slots[idx].value = AM_VALUE_NULL;
        }
    }

    // 把柄已存在，仅做 value 替换，无需扩容；使用 am_map_set_stable 保证指针稳定。
    if (am_map_set_stable(container_alloc, heap->table, handle_val, value) != 0) {
        // 设置失败，恢复旧指针值，不释放对象
        if (am_value_is_ptr(old_ptr)) {
            size_t idx2;
            if (am_heap_find_slot(heap->table, handle_val, &idx2) >= 0) {
                heap->table->slots[idx2].value = old_ptr;
            }
        }
        return -1;
    }

    if (am_value_is_ptr(old_ptr)) {
        am_free(obj_alloc, am_value_to_ptr(old_ptr));
    }
    return 0;
}
/* ===== end:   src/am_heap.c ===== */

/* ===== begin: src/am_closure.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>



// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 在闭包中线性查找指定 varid 与 type 的绑定。
// 返回指向匹配 binding 的指针；未找到返回 NULL。
static am_binding_t *am_closure_find(am_obj_closure_t *closure, am_varid_t varid, int32_t type) {
    for (size_t i = 0; i < closure->length; i++) {
        if (closure->bindings[i].varid == varid && closure->bindings[i].type == type) {
            return &closure->bindings[i];
        }
    }
    return NULL;
}

// 仅按 varid 查找（不区分绑定类型），用于脏标记查询。
// 返回指向匹配 binding 的指针；未找到返回 NULL。
static am_binding_t *am_closure_find_by_varid(am_obj_closure_t *closure, am_varid_t varid) {
    for (size_t i = 0; i < closure->length; i++) {
        if (closure->bindings[i].varid == varid) {
            return &closure->bindings[i];
        }
    }
    return NULL;
}

// 将闭包扩容到新容量，返回新的闭包对象指针；失败返回 NULL。
// 原闭包对象会被释放，调用者必须使用返回的新指针。
static am_obj_closure_t *am_closure_resize(am_allocator_t *alloc, am_obj_closure_t *closure, size_t new_capacity) {
    size_t total_size = sizeof(am_obj_closure_t) + new_capacity * sizeof(am_binding_t);
    am_obj_closure_t *new_closure = (am_obj_closure_t *)am_malloc(alloc, total_size);
    if (!new_closure) return NULL;

    // 拷贝头部与已有 binding（拷贝长度按原 length，而非原 capacity）
    memcpy(new_closure, closure, sizeof(am_obj_closure_t));
    if (closure->length > 0) {
        memcpy(new_closure->bindings, closure->bindings, closure->length * sizeof(am_binding_t));
    }
    new_closure->capacity = new_capacity;

    am_free(alloc, closure);
    return new_closure;
}

// 若空间不足则扩容。绝大多数情况下不会触发实际分配。
// 返回原指针或新指针；失败返回 NULL。
static am_obj_closure_t *am_closure_grow_if_needed(am_allocator_t *alloc, am_obj_closure_t *closure) {
    if (closure->length < closure->capacity) return closure;

    size_t new_capacity = closure->capacity * 2;
    if (new_capacity < 16) new_capacity = 16;
    return am_closure_resize(alloc, closure, new_capacity);
}

// ===============================================================================
// 构造函数
// ===============================================================================

// 创建闭包。capacity 为 0 时默认使用 16。
am_obj_closure_t *am_closure_create(am_allocator_t *alloc, am_iaddr_t iaddr, am_handle_t parent, size_t capacity) {
    if (capacity == 0) capacity = 16;

    size_t total_size = sizeof(am_obj_closure_t) + capacity * sizeof(am_binding_t);
    am_obj_closure_t *closure = (am_obj_closure_t *)am_malloc(alloc, total_size);
    if (!closure) return NULL;

    memset(closure, 0, total_size);

    closure->base.type = AM_OBJECT_TYPE_CLOSURE;
    closure->iaddr = iaddr;
    closure->parent = parent;
    closure->length = 0;
    closure->capacity = capacity;

    return closure;
}

// ===============================================================================
// 析构
// ===============================================================================

// 销毁闭包对象。binding 中的 value 按引用处理，不由闭包释放。
int32_t am_closure_destroy(am_allocator_t *alloc, am_obj_closure_t *closure) {
    am_free(alloc, closure);
    return 0;
}

// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝（头部与所有 binding）。value 按位拷贝（与 TS Copy 语义一致，不递归释放对象）。
am_obj_closure_t *am_closure_copy(am_allocator_t *alloc, am_obj_closure_t *closure) {
    am_obj_closure_t *copy = am_closure_create(alloc, closure->iaddr, closure->parent, closure->capacity);
    if (!copy) return NULL;

    copy->length = closure->length;
    if (closure->length > 0) {
        memcpy(copy->bindings, closure->bindings, closure->length * sizeof(am_binding_t));
    }
    return copy;
}

// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_closure_size(am_allocator_t *alloc, am_obj_closure_t *obj) {
    (void)alloc;
    if (!obj) return SIZE_MAX;

    if (obj->capacity > (SIZE_MAX - sizeof(am_obj_closure_t)) / sizeof(am_binding_t)) {
        return SIZE_MAX;
    }
    return sizeof(am_obj_closure_t) + obj->capacity * sizeof(am_binding_t);
}


// ===============================================================================
// 对象二进制转储 TODO
// ===============================================================================

// 功能说明：将闭包对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分。
// 磁盘格式（平台无关固定宽度，小端；详见 include/object.h）：
//   [16B] 对象基类头（type=AM_OBJECT_TYPE_CLOSURE）
//   [uvarint] iaddr
//   [uvarint] parent 把柄
//   [uvarint] length（capacity 压缩为与 length 一致，不落盘）
//   [length * (uvarint varid, u8 type, u8 dirty_flag, dvalue value)] 绑定表项
size_t am_closure_dump(am_allocator_t *alloc, am_obj_closure_t *closure, uint8_t *buffer, size_t offset) {
    (void)alloc;
    if (!closure) return SIZE_MAX;

    size_t pos = offset;
    if (buffer != NULL && offset != SIZE_MAX) {
        am_disk_write_base(buffer, pos, &closure->base);
    }
    pos += AM_DISK_BASE_SIZE;
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)closure->iaddr);
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)closure->parent);
    pos += am_disk_write_uvarint(buffer, pos, (uint64_t)closure->length);

    for (size_t i = 0; i < closure->length; i++) {
        am_binding_t *b = &closure->bindings[i];
        pos += am_disk_write_uvarint(buffer, pos, (uint64_t)b->varid);
        if (buffer != NULL && offset != SIZE_MAX) {
            buffer[pos] = (uint8_t)b->type;
            buffer[pos + 1] = (uint8_t)b->dirty_flag;
        }
        pos += 2;
        pos += am_disk_write_value(buffer, pos, b->value);
    }

    return pos - offset;
}


// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的闭包对象，构造闭包对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_obj_closure_t对象的指针，失败则返回NULL。
am_obj_closure_t *am_closure_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset) {
    if (!alloc || !buffer) return NULL;

    size_t pos = offset;
    am_object_t base;
    am_disk_read_base(buffer, pos, &base);
    pos += AM_DISK_BASE_SIZE;
    if (base.type != AM_OBJECT_TYPE_CLOSURE) return NULL;

    uint64_t iaddr = 0, parent = 0, length = 0;
    size_t n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &iaddr))) return NULL;
    pos += n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &parent))) return NULL;
    pos += n;
    if (!(n = am_disk_read_uvarint(buffer, pos, &length))) return NULL;
    pos += n;

    if (iaddr > (uint64_t)AM_HANDLE_NULL) return NULL; // 与 am_iaddr_t/am_handle_t 同宽检查
    if (parent > (uint64_t)AM_HANDLE_NULL) return NULL;
    if (length > (uint64_t)((SIZE_MAX - sizeof(am_obj_closure_t)) / sizeof(am_binding_t))) return NULL;

    am_obj_closure_t *closure = am_closure_create(alloc, (am_iaddr_t)iaddr, (am_handle_t)parent, (size_t)length);
    if (!closure) return NULL;
    closure->base = base;
    closure->length = (size_t)length;

    for (size_t i = 0; i < closure->length; i++) {
        uint64_t varid = 0;
        if (!(n = am_disk_read_uvarint(buffer, pos, &varid))) goto fail;
        pos += n;
        if (varid > (uint64_t)AM_HANDLE_NULL) goto fail;

        int32_t type = (int32_t)buffer[pos];
        int32_t dirty_flag = (int32_t)buffer[pos + 1];
        pos += 2;

        am_value_t value = 0;
        if (!(n = am_disk_read_value(buffer, pos, &value))) goto fail;
        pos += n;

        closure->bindings[i].varid = (am_varid_t)varid;
        closure->bindings[i].type = type;
        closure->bindings[i].dirty_flag = dirty_flag;
        closure->bindings[i].value = value;
    }

    return closure;

fail:
    am_free(alloc, closure);
    return NULL;
}


// ===============================================================================
// 约束变量操作
// ===============================================================================

// 初始化约束变量（不加脏标记）。若已存在则更新 value 并清除脏标记。
// 如涉及扩容，返回新闭包对象指针；否则返回原指针。失败返回 NULL。
am_obj_closure_t *am_closure_init_bound_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable, am_value_t value) {
    am_binding_t *binding = am_closure_find(closure, variable, AM_BINDING_BOUND);
    if (binding) {
        binding->value = value;
        binding->dirty_flag = 0;
        // 同一 varid 的自由变量绑定也清除脏标记，保持该变量脏标记一致
        am_binding_t *free_binding = am_closure_find(closure, variable, AM_BINDING_FREE);
        if (free_binding) {
            free_binding->dirty_flag = 0;
        }
        return closure;
    }

    closure = am_closure_grow_if_needed(alloc, closure);
    if (!closure) return NULL;

    closure->bindings[closure->length].varid = variable;
    closure->bindings[closure->length].type = AM_BINDING_BOUND;
    closure->bindings[closure->length].dirty_flag = 0;
    closure->bindings[closure->length].value = value;
    closure->length++;
    // 若已存在同 varid 的自由变量绑定，也将其脏标记清除
    am_binding_t *free_binding = am_closure_find(closure, variable, AM_BINDING_FREE);
    if (free_binding) {
        free_binding->dirty_flag = 0;
    }
    return closure;
}

// 设置约束变量（加脏标记，仅用于 set 指令）。若不存在则插入。
// 返回新指针或原指针；失败返回 NULL。
am_obj_closure_t *am_closure_set_bound_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable, am_value_t value) {
    am_binding_t *binding = am_closure_find(closure, variable, AM_BINDING_BOUND);
    if (binding) {
        binding->value = value;
        binding->dirty_flag = 1;
    }
    else {
        closure = am_closure_grow_if_needed(alloc, closure);
        if (!closure) return NULL;

        closure->bindings[closure->length].varid = variable;
        closure->bindings[closure->length].type = AM_BINDING_BOUND;
        closure->bindings[closure->length].dirty_flag = 1;
        closure->bindings[closure->length].value = value;
        closure->length++;
    }

    // 同一 varid 的自由变量绑定也置脏，保持变量级脏标记一致
    am_binding_t *free_binding = am_closure_find(closure, variable, AM_BINDING_FREE);
    if (free_binding) {
        free_binding->dirty_flag = 1;
        free_binding->value = value;
    }

    return closure;
}

// 获取约束变量。未找到返回 AM_VALUE_UNDEFINED。
am_value_t am_closure_get_bound_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable) {
    (void)alloc;
    am_binding_t *binding = am_closure_find(closure, variable, AM_BINDING_BOUND);
    return binding ? binding->value : AM_VALUE_UNDEFINED;
}

// ===============================================================================
// 自由变量操作
// ===============================================================================

// 初始化自由变量（不加脏标记）。若已存在则更新 value 并清除脏标记。
am_obj_closure_t *am_closure_init_free_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable, am_value_t value) {
    am_binding_t *binding = am_closure_find(closure, variable, AM_BINDING_FREE);
    if (binding) {
        binding->value = value;
        binding->dirty_flag = 0;
        am_binding_t *bound_binding = am_closure_find(closure, variable, AM_BINDING_BOUND);
        if (bound_binding) {
            bound_binding->dirty_flag = 0;
        }
        return closure;
    }

    closure = am_closure_grow_if_needed(alloc, closure);
    if (!closure) return NULL;

    closure->bindings[closure->length].varid = variable;
    closure->bindings[closure->length].type = AM_BINDING_FREE;
    closure->bindings[closure->length].dirty_flag = 0;
    closure->bindings[closure->length].value = value;
    closure->length++;
    am_binding_t *bound_binding = am_closure_find(closure, variable, AM_BINDING_BOUND);
    if (bound_binding) {
        bound_binding->dirty_flag = 0;
    }
    return closure;
}

// 设置自由变量（加脏标记，仅用于 set 指令）。若不存在则插入。
am_obj_closure_t *am_closure_set_free_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable, am_value_t value) {
    am_binding_t *binding = am_closure_find(closure, variable, AM_BINDING_FREE);
    if (binding) {
        binding->value = value;
        binding->dirty_flag = 1;
    }
    else {
        closure = am_closure_grow_if_needed(alloc, closure);
        if (!closure) return NULL;

        closure->bindings[closure->length].varid = variable;
        closure->bindings[closure->length].type = AM_BINDING_FREE;
        closure->bindings[closure->length].dirty_flag = 1;
        closure->bindings[closure->length].value = value;
        closure->length++;
    }

    // 同一 varid 的约束变量绑定也置脏，保持变量级脏标记一致
    am_binding_t *bound_binding = am_closure_find(closure, variable, AM_BINDING_BOUND);
    if (bound_binding) {
        bound_binding->dirty_flag = 1;
        bound_binding->value = value;
    }

    return closure;
}

// 获取自由变量。未找到返回 AM_VALUE_UNDEFINED。
am_value_t am_closure_get_free_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable) {
    (void)alloc;
    am_binding_t *binding = am_closure_find(closure, variable, AM_BINDING_FREE);
    return binding ? binding->value : AM_VALUE_UNDEFINED;
}

// ===============================================================================
// 查询
// ===============================================================================

// 判断变量是否为脏。为脏返回 0，未找到或不为脏返回 -1。
int32_t am_closure_is_dirty_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable) {
    (void)alloc;
    am_binding_t *binding = am_closure_find_by_varid(closure, variable);
    return (binding && binding->dirty_flag != 0) ? 0 : -1;
}

// 是否存在约束变量绑定。存在返回 0，不存在返回 -1。
int32_t am_closure_has_bound_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable) {
    (void)alloc;
    return am_closure_find(closure, variable, AM_BINDING_BOUND) ? 0 : -1;
}

// 是否存在自由变量绑定。存在返回 0，不存在返回 -1。
int32_t am_closure_has_free_var(am_allocator_t *alloc, am_obj_closure_t *closure, am_varid_t variable) {
    (void)alloc;
    return am_closure_find(closure, variable, AM_BINDING_FREE) ? 0 : -1;
}
/* ===== end:   src/am_closure.c ===== */

/* ===== begin: src/am_continuation.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>



// ===============================================================================
// 构造函数
// ===============================================================================

// 创建续体对象。成功返回指针，失败返回NULL。
// opstack 与 fstack 按值拷贝到柔性数组 stacks 中；传入的数组指针在拷贝后不再被引用。
am_continuation_t *am_continuation_create(
    am_allocator_t *alloc, am_iaddr_t cont_return_target, am_handle_t current_closure_handle,
    am_value_t *opstack, size_t opstack_length, am_value_t *fstack, size_t fstack_length,
    am_handle_t dynamic_wind_stack_handle) {
    if (!alloc) return NULL;

    size_t total_length = opstack_length + fstack_length;
    size_t total_size = sizeof(am_continuation_t) + total_length * sizeof(am_value_t);

    am_continuation_t *cont = (am_continuation_t *)am_malloc(alloc, total_size);
    if (!cont) return NULL;

    memset(cont, 0, total_size);

    cont->base.type = AM_OBJECT_TYPE_CONTINUATION;
    cont->length = total_length;
    cont->fstack_offset = opstack_length;
    cont->cont_return_target = cont_return_target;
    cont->current_closure_handle = current_closure_handle;
    cont->dynamic_wind_stack_handle = dynamic_wind_stack_handle;
    cont->dynamic_wind_after_stack_handle = AM_HANDLE_NULL;
    cont->current_dynamic_wind_entry_handle = AM_HANDLE_NULL;
    cont->current_dynamic_wind_thunk_handle = AM_HANDLE_NULL;

    if (opstack_length > 0) {
        memcpy(&cont->stacks[0], opstack, opstack_length * sizeof(am_value_t));
    }
    if (fstack_length > 0) {
        memcpy(&cont->stacks[opstack_length], fstack, fstack_length * sizeof(am_value_t));
    }

    return cont;
}


// ===============================================================================
// 析构
// ===============================================================================

// 销毁续体对象。stacks 中的 value 按引用处理，不由续体释放。
int32_t am_continuation_destroy(am_allocator_t *alloc, am_continuation_t *obj) {
    if (!alloc || !obj) return -1;
    am_free(alloc, obj);
    return 0;
}


// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝（头部与整个 stacks 数组）。value 按位拷贝（与 TS Copy 语义一致，不递归释放对象）。
am_continuation_t *am_continuation_copy(am_allocator_t *alloc, am_continuation_t *obj) {
    if (!alloc || !obj) return NULL;

    size_t opstack_length = obj->fstack_offset;
    size_t fstack_length = obj->length - obj->fstack_offset;

    am_continuation_t *copy = am_continuation_create(
        alloc, obj->cont_return_target, obj->current_closure_handle,
        &obj->stacks[0], opstack_length,
        &obj->stacks[obj->fstack_offset], fstack_length,
        obj->dynamic_wind_stack_handle);

    if (!copy) return NULL;

    // 拷贝基类元数据（type 已在 create 中设置）
    copy->base.header = obj->base.header;
    copy->base.hash = obj->base.hash;
    copy->base.gcmark = obj->base.gcmark;

    return copy;
}


// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_continuation_size(am_allocator_t *alloc, am_continuation_t *obj) {
    (void)alloc;
    if (!obj) return SIZE_MAX;

    if (obj->length > (SIZE_MAX - sizeof(am_continuation_t)) / sizeof(am_value_t)) {
        return SIZE_MAX;
    }
    return sizeof(am_continuation_t) + obj->length * sizeof(am_value_t);
}


// ===============================================================================
// 查询
// ===============================================================================

// 获取 opstack 数组的副本。成功返回新数组指针（通过 alloc 分配，由调用者负责释放），失败返回NULL。
am_value_t *am_continuation_get_opstack(am_allocator_t *alloc, am_continuation_t *obj, size_t *length) {
    if (!alloc || !obj || !length) return NULL;

    size_t opstack_length = obj->fstack_offset;
    *length = opstack_length;

    // 即使长度为 0 也分配至少一个元素大小，避免 malloc(0) 行为不确定导致误判失败
    size_t alloc_size = opstack_length > 0 ? opstack_length * sizeof(am_value_t) : sizeof(am_value_t);
    am_value_t *result = (am_value_t *)am_malloc(alloc, alloc_size);
    if (!result) return NULL;

    if (opstack_length > 0) {
        memcpy(result, &obj->stacks[0], opstack_length * sizeof(am_value_t));
    }

    return result;
}

// 获取 fstack 数组的副本。成功返回新数组指针（通过 alloc 分配，由调用者负责释放），失败返回NULL。
am_value_t *am_continuation_get_fstack(am_allocator_t *alloc, am_continuation_t *obj, size_t *length) {
    if (!alloc || !obj || !length) return NULL;

    size_t fstack_length = obj->length - obj->fstack_offset;
    *length = fstack_length;

    // 即使长度为 0 也分配至少一个元素大小，避免 malloc(0) 行为不确定导致误判失败
    size_t alloc_size = fstack_length > 0 ? fstack_length * sizeof(am_value_t) : sizeof(am_value_t);
    am_value_t *result = (am_value_t *)am_malloc(alloc, alloc_size);
    if (!result) return NULL;

    if (fstack_length > 0) {
        memcpy(result, &obj->stacks[obj->fstack_offset], fstack_length * sizeof(am_value_t));
    }

    return result;
}
/* ===== end:   src/am_continuation.c ===== */

/* ===== begin: src/am_scope.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>



// ===============================================================================
// 内部辅助函数
// ===============================================================================

static am_scope_t *am_scope_resize(am_allocator_t *alloc, am_scope_t *scope, size_t new_capacity) {
    if (new_capacity < scope->length) new_capacity = scope->length;

    size_t total_size = sizeof(am_scope_t) + new_capacity * sizeof(am_scope_binding_t);
    am_scope_t *new_scope = (am_scope_t *)am_malloc(alloc, total_size);
    if (!new_scope) return NULL;

    new_scope->base = scope->base;
    new_scope->parent_scope_handle = scope->parent_scope_handle;
    new_scope->parent_lambda_handle = scope->parent_lambda_handle;
    new_scope->current_lambda_handle = scope->current_lambda_handle;
    new_scope->capacity = new_capacity;
    new_scope->length = scope->length;

    if (scope->length > 0) {
        memcpy(new_scope->bindings, scope->bindings, scope->length * sizeof(am_scope_binding_t));
    }

    am_free(alloc, scope);
    return new_scope;
}


static am_scope_t *am_scope_grow_if_needed(am_allocator_t *alloc, am_scope_t *scope) {
    if (scope->length < scope->capacity) return scope;

    size_t new_capacity = scope->capacity * 2;
    if (new_capacity < 16) new_capacity = 16;
    return am_scope_resize(alloc, scope, new_capacity);
}


// ===============================================================================
// 构造函数
// ===============================================================================

am_scope_t *am_scope_create(am_allocator_t *alloc, am_handle_t parent_scope_handle, am_handle_t parent_lambda_handle, am_handle_t current_lambda_handle, size_t capacity) {
    if (capacity == 0) capacity = 16;

    size_t total_size = sizeof(am_scope_t) + capacity * sizeof(am_scope_binding_t);
    am_scope_t *scope = (am_scope_t *)am_calloc(alloc, total_size);
    if (!scope) return NULL;

    scope->base.type = AM_OBJECT_TYPE_SCOPE;
    scope->parent_scope_handle = parent_scope_handle;
    scope->parent_lambda_handle = parent_lambda_handle;
    scope->current_lambda_handle = current_lambda_handle;
    scope->capacity = capacity;
    scope->length = 0;

    return scope;
}


// ===============================================================================
// 析构
// ===============================================================================

int32_t am_scope_destroy(am_allocator_t *alloc, am_scope_t *scope) {
    if (!scope) return 0;
    am_free(alloc, scope);
    return 0;
}


// ===============================================================================
// 拷贝
// ===============================================================================

am_scope_t *am_scope_copy(am_allocator_t *alloc, am_scope_t *scope) {
    if (!scope) return NULL;

    am_scope_t *copy = am_scope_create(alloc, scope->parent_scope_handle, scope->parent_lambda_handle, scope->current_lambda_handle, scope->capacity);
    if (!copy) return NULL;

    copy->base = scope->base;
    copy->length = scope->length;
    if (scope->length > 0) {
        memcpy(copy->bindings, scope->bindings, scope->length * sizeof(am_scope_binding_t));
    }
    return copy;
}


// ===============================================================================
// 对象二进制转储 TODO
// ===============================================================================

uint8_t *am_scope_dump(am_allocator_t *alloc, am_scope_t *scope, size_t *size) {
    (void)alloc;
    (void)scope;
    if (size) *size = 0;
    return NULL;
}


// ===============================================================================
// 查询与新增
// ===============================================================================

int32_t am_scope_has_var(am_allocator_t *alloc, am_scope_t *scope, am_varid_t variable) {
    (void)alloc;
    if (!scope) return -1;
    for (size_t i = 0; i < scope->length; i++) {
        if (scope->bindings[i].varid == variable) return 0;
    }
    return -1;
}


am_scope_t *am_scope_add_var(am_allocator_t *alloc, am_scope_t *scope, am_varid_t variable, am_value_t value) {
    if (!scope) return NULL;
    if (am_scope_has_var(alloc, scope, variable) >= 0) return NULL;

    scope = am_scope_grow_if_needed(alloc, scope);
    if (!scope) return NULL;

    scope->bindings[scope->length].varid = variable;
    scope->bindings[scope->length].value = value;
    scope->length++;

    return scope;
}
/* ===== end:   src/am_scope.c ===== */

/* ===== begin: src/am_debug.c ===== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <wchar.h>



#define AST_PRINT_INDENT 2


static void debug_ast_print_indent(FILE *out, int depth) {
    for (int i = 0; i < depth * AST_PRINT_INDENT; i++) {
        fputwc(L' ', out);
    }
}


static int debug_ast_is_handle_visited(am_handle_t *visited, size_t count, am_handle_t handle) {
    for (size_t i = 0; i < count; i++) {
        if (visited[i] == handle) return 1;
    }
    return 0;
}


static void debug_ast_print_value(am_ast_t *ast, am_value_t value, FILE *out,
                                  am_handle_t **visited, size_t *visited_count, size_t *visited_capacity, int depth);


static void debug_ast_print_ensure_visited_capacity(am_handle_t **visited, size_t *visited_capacity, size_t visited_count) {
    if (visited_count < *visited_capacity) return;
    size_t new_capacity = *visited_capacity ? *visited_capacity * 2 : 256;
    if (new_capacity < visited_count + 1) new_capacity = visited_count + 1;
    am_handle_t *new_visited = (am_handle_t *)realloc(*visited, new_capacity * sizeof(am_handle_t));
    if (!new_visited) {
        fprintf(stderr, "debug_ast_print: failed to allocate visited buffer\n");
        return;
    }
    *visited = new_visited;
    *visited_capacity = new_capacity;
}


static void debug_ast_print_node(am_ast_t *ast, am_handle_t handle, FILE *out,
                                 am_handle_t **visited, size_t *visited_count, size_t *visited_capacity, int depth) {
    am_value_t node_val = am_ast_get_node(ast, handle);
    if (!am_value_is_ptr(node_val)) {
        fwprintf(out, L"<H:%zu> (invalid)", (size_t)handle);
        return;
    }

    am_object_t *obj = am_value_to_ptr(node_val);

    if (obj->type == AM_OBJECT_TYPE_WSTRING) {
        am_wstring_t *ws = (am_wstring_t *)obj;
        fputwc(L'"', out);
        for (size_t i = 0; i < ws->length; i++) {
            wchar_t c = (wchar_t)am_value_to_wchar(ws->content[i]);
            if (c == L'"') fputwc(L'\\', out);
            fwprintf(out, L"%lc", c);
        }
        fputwc(L'"', out);
        return;
    }

    if (obj->type != AM_OBJECT_TYPE_LIST) {
        fwprintf(out, L"<H:%zu> (unknown type %d)", (size_t)handle, obj->type);
        return;
    }

    am_list_t *lst = (am_list_t *)obj;

    if (debug_ast_is_handle_visited(*visited, *visited_count, handle)) {
        fwprintf(out, L"<H:%zu>", (size_t)handle);
        return;
    }
    debug_ast_print_ensure_visited_capacity(visited, visited_capacity, *visited_count);
    (*visited)[(*visited_count)++] = handle;

    fwprintf(out, L"<H:%zu> {\n", (size_t)handle);
    debug_ast_print_indent(out, depth + 1);
    fwprintf(out, L"type: ");
    switch (lst->type) {
        case AM_LIST_TYPE_LAMBDA:      fwprintf(out, L"\"LAMBDA\"\n"); break;
        case AM_LIST_TYPE_APPLICATION: fwprintf(out, L"\"APPLICATION\"\n"); break;
        case AM_LIST_TYPE_QUOTE:       fwprintf(out, L"\"QUOTE\"\n"); break;
        case AM_LIST_TYPE_QUASIQUOTE:  fwprintf(out, L"\"QUASIQUOTE\"\n"); break;
        case AM_LIST_TYPE_UNQUOTE:     fwprintf(out, L"\"UNQUOTE\"\n"); break;
        default:                       fwprintf(out, L"\"UNKNOWN(%d)\"\n", lst->type); break;
    }

    debug_ast_print_indent(out, depth + 1);
    if (lst->parent == AM_HANDLE_NULL) {
        fwprintf(out, L"parent: null\n");
    }
    else {
        fwprintf(out, L"parent: <H:%zu>\n", (size_t)lst->parent);
    }

    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        am_uint_t n_param = 0;
        if (lst->length >= 2) {
            am_value_t np = am_list_get(ast->alloc, lst, 1);
            if (am_value_is_uint(np)) n_param = am_value_to_uint(np);
        }

        debug_ast_print_indent(out, depth + 1);
        fwprintf(out, L"parameters: [\n");
        for (am_uint_t i = 0; i < n_param; i++) {
            debug_ast_print_indent(out, depth + 2);
            am_value_t param = am_list_get(ast->alloc, lst, 2 + i);
            debug_ast_print_value(ast, param, out, visited, visited_count, visited_capacity, depth + 2);
            if (i + 1 < n_param) fputwc(L',', out);
            fputwc(L'\n', out);
        }
        debug_ast_print_indent(out, depth + 1);
        fwprintf(out, L"]\n");

        debug_ast_print_indent(out, depth + 1);
        fwprintf(out, L"bodies: [\n");
        size_t body_start = 2 + n_param;
        for (size_t i = body_start; i < lst->length; i++) {
            debug_ast_print_indent(out, depth + 2);
            am_value_t body = am_list_get(ast->alloc, lst, i);
            debug_ast_print_value(ast, body, out, visited, visited_count, visited_capacity, depth + 2);
            if (i + 1 < lst->length) fputwc(L',', out);
            fputwc(L'\n', out);
        }
        debug_ast_print_indent(out, depth + 1);
        fwprintf(out, L"]\n");
    }
    else {
        debug_ast_print_indent(out, depth + 1);
        fwprintf(out, L"children: [\n");
        for (size_t i = 0; i < lst->length; i++) {
            debug_ast_print_indent(out, depth + 2);
            am_value_t child = am_list_get(ast->alloc, lst, i);
            debug_ast_print_value(ast, child, out, visited, visited_count, visited_capacity, depth + 2);
            if (i + 1 < lst->length) fputwc(L',', out);
            fputwc(L'\n', out);
        }
        debug_ast_print_indent(out, depth + 1);
        fwprintf(out, L"]\n");
    }

    debug_ast_print_indent(out, depth);
    fwprintf(out, L"}");
}


static void debug_ast_print_value(am_ast_t *ast, am_value_t value, FILE *out,
                                  am_handle_t **visited, size_t *visited_count, size_t *visited_capacity, int depth) {
    if (am_value_is_handle(value)) {
        am_handle_t h = am_value_to_handle(value);
        debug_ast_print_node(ast, h, out, visited, visited_count, visited_capacity, depth);
    }
    else if (am_value_is_varid(value)) {
        am_varid_t varid = am_value_to_varid(value);
        size_t idx = (size_t)varid;
        wchar_t *name = am_vocab_get(ast->alloc, ast->var_vocab, &idx);
        if (name) {
            fwprintf(out, L"\"%ls\" (varid=%zu)", name, (size_t)varid);
        }
        else {
            fwprintf(out, L"<varid=%zu>", (size_t)varid);
        }
    }
    else if (am_value_is_symbol(value)) {
        am_symbol_t sym = am_value_to_symbol(value);
        size_t idx = (size_t)sym;
        wchar_t *name = am_vocab_get(ast->alloc, ast->symbol_vocab, &idx);
        if (name) {
            fwprintf(out, L"\"%ls\" (symbol=%zu)", name, (size_t)sym);
        }
        else {
            fwprintf(out, L"<symbol=%zu>", (size_t)sym);
        }
    }
    else if (am_value_is_uint(value)) {
        fwprintf(out, L"%llu", (unsigned long long)am_value_to_uint(value));
    }
    else if (am_value_is_int(value)) {
        fwprintf(out, L"%lld", (long long)am_value_to_int(value));
    }
    else if (am_value_is_float(value)) {
        fwprintf(out, L"%g", (double)am_value_to_float(value));
    }
    else if (am_value_is_boolean(value)) {
        fwprintf(out, L"%ls", am_value_to_boolean(value) ? L"#t" : L"#f");
    }
    else if (am_value_is_null(value)) {
        fwprintf(out, L"#null");
    }
    else if (am_value_is_undefined(value)) {
        fwprintf(out, L"#undefined");
    }
    else {
        fwprintf(out, L"<value=%llu>", (unsigned long long)value);
    }
}


static void debug_ast_print_vocab(FILE *out, am_ast_t *ast, am_vocab_t *vocab) {
    fputwc(L'[', out);
    for (size_t i = 0; i < vocab->length; i++) {
        size_t idx = i;
        wchar_t *word = am_vocab_get(ast->alloc, vocab, &idx);
        if (i > 0) fwprintf(out, L", ");
        if (word) {
            fputwc(L'"', out);
            fwprintf(out, L"%ls", word);
            fputwc(L'"', out);
        }
        else {
            fwprintf(out, L"null");
        }
    }
    fputwc(L']', out);
}


static void debug_ast_print_handle_list(FILE *out, am_ast_t *ast, am_list_t *lst) {
    (void)ast;
    fputwc(L'[', out);
    for (size_t i = 0; i < lst->length; i++) {
        if (i > 0) fwprintf(out, L", ");
        fwprintf(out, L"<H:%zu>", (size_t)am_value_to_handle(am_list_get(ast->alloc, lst, i)));
    }
    fputwc(L']', out);
}


static void debug_ast_print_value_inline(am_ast_t *ast, am_value_t value, FILE *out);


static void debug_ast_print_value_list(FILE *out, am_ast_t *ast, am_list_t *lst) {
    fputwc(L'[', out);
    for (size_t i = 0; i < lst->length; i++) {
        if (i > 0) fwprintf(out, L", ");
        debug_ast_print_value_inline(ast, am_list_get(ast->alloc, lst, i), out);
    }
    fputwc(L']', out);
}


static void debug_ast_print_map_varid_to_handle(FILE *out, am_ast_t *ast, am_map_t *map) {
    fputwc(L'{', out);
    size_t count = am_map_length(ast->alloc, map);
    am_value_t *keys = am_map_keys(ast->alloc, map);
    for (size_t i = 0; i < count; i++) {
        if (i > 0) fwprintf(out, L", ");
        am_varid_t varid = am_value_to_varid(keys[i]);
        size_t idx = (size_t)varid;
        wchar_t *name = am_vocab_get(ast->alloc, ast->var_vocab, &idx);
        if (name) fwprintf(out, L"\"%ls\": ", name);
        else fwprintf(out, L"<varid=%zu>: ", (size_t)varid);

        am_value_t v = am_map_get(ast->alloc, map, keys[i]);
        if (am_value_is_handle(v)) {
            fwprintf(out, L"<H:%zu>", (size_t)am_value_to_handle(v));
        }
        else {
            fwprintf(out, L"null");
        }
    }
    if (keys) am_free(ast->alloc, keys);
    fputwc(L'}', out);
}


static const wchar_t *debug_var_type_name(am_uint_t type) {
    switch (type) {
        case AM_VAR_TYPE_OLD:          return L"OLD";
        case AM_VAR_TYPE_NEW:          return L"NEW";
        case AM_VAR_TYPE_BUILTIN:      return L"BUILTIN";
        case AM_VAR_TYPE_EXT_REF:      return L"EXT_REF";
        case AM_VAR_TYPE_IMPORT_REF:   return L"IMPORT_REF";
        case AM_VAR_TYPE_NATIVE_REF:   return L"NATIVE_REF";
        case AM_VAR_TYPE_IMPORT_ALIAS: return L"IMPORT_ALIAS";
        case AM_VAR_TYPE_NATIVE_ID:    return L"NATIVE_ID";
        default:                       return L"UNKNOWN";
    }
}


static void debug_ast_print_var_type(FILE *out, am_ast_t *ast, am_list_t *lst) {
    (void)ast;
    fputwc(L'[', out);
    for (size_t i = 0; i < lst->length; i++) {
        if (i > 0) fwprintf(out, L", ");
        am_value_t v = am_list_get(ast->alloc, lst, i);
        if (am_value_is_uint(v)) {
            am_uint_t t = am_value_to_uint(v);
            fwprintf(out, L"%ls(%zu)", debug_var_type_name(t), (size_t)t);
        }
        else {
            fwprintf(out, L"?");
        }
    }
    fputwc(L']', out);
}


static void debug_ast_print_var_arn_mapping(FILE *out, am_ast_t *ast, am_map_t *map) {
    fputwc(L'{', out);
    size_t count = am_map_length(ast->alloc, map);
    am_value_t *keys = am_map_keys(ast->alloc, map);
    for (size_t i = 0; i < count; i++) {
        if (i > 0) fwprintf(out, L", ");
        am_varid_t new_varid = am_value_to_varid(keys[i]);
        size_t new_idx = (size_t)new_varid;
        wchar_t *new_name = am_vocab_get(ast->alloc, ast->var_vocab, &new_idx);
        if (new_name) fwprintf(out, L"\"%ls\": ", new_name);
        else fwprintf(out, L"<varid=%zu>: ", (size_t)new_varid);

        am_value_t v = am_map_get(ast->alloc, map, keys[i]);
        if (am_value_is_varid(v)) {
            am_varid_t old_varid = am_value_to_varid(v);
            size_t old_idx = (size_t)old_varid;
            wchar_t *old_name = am_vocab_get(ast->alloc, ast->var_vocab, &old_idx);
            if (old_name) fwprintf(out, L"\"%ls\"", old_name);
            else fwprintf(out, L"<varid=%zu>", (size_t)old_varid);
        }
        else {
            fwprintf(out, L"null");
        }
    }
    if (keys) am_free(ast->alloc, keys);
    fputwc(L'}', out);
}


static void debug_ast_print_value_inline(am_ast_t *ast, am_value_t value, FILE *out) {
    if (am_value_is_handle(value)) {
        fwprintf(out, L"<H:%zu>", (size_t)am_value_to_handle(value));
    }
    else if (am_value_is_varid(value)) {
        am_varid_t varid = am_value_to_varid(value);
        size_t idx = (size_t)varid;
        wchar_t *name = am_vocab_get(ast->alloc, ast->var_vocab, &idx);
        if (name) fwprintf(out, L"\"%ls\"", name);
        else fwprintf(out, L"<varid=%zu>", (size_t)varid);
    }
    else if (am_value_is_symbol(value)) {
        am_symbol_t sym = am_value_to_symbol(value);
        size_t idx = (size_t)sym;
        wchar_t *name = am_vocab_get(ast->alloc, ast->symbol_vocab, &idx);
        if (name) fwprintf(out, L"\"%ls\"", name);
        else fwprintf(out, L"<symbol=%zu>", (size_t)sym);
    }
    else if (am_value_is_uint(value)) {
        fwprintf(out, L"%llu", (unsigned long long)am_value_to_uint(value));
    }
    else if (am_value_is_int(value)) {
        fwprintf(out, L"%lld", (long long)am_value_to_int(value));
    }
    else if (am_value_is_float(value)) {
        fwprintf(out, L"%g", (double)am_value_to_float(value));
    }
    else if (am_value_is_boolean(value)) {
        fwprintf(out, L"%ls", am_value_to_boolean(value) ? L"#t" : L"#f");
    }
    else if (am_value_is_null(value)) {
        fwprintf(out, L"#null");
    }
    else if (am_value_is_undefined(value)) {
        fwprintf(out, L"#undefined");
    }
    else {
        fwprintf(out, L"<value=%llu>", (unsigned long long)value);
    }
}


void am_debug_ast_print_node_summary(FILE *out, am_ast_t *ast, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ast, handle);
    if (!am_value_is_ptr(node_val)) {
        fwprintf(out, L"<H:%zu>: (invalid)\n", (size_t)handle);
        return;
    }

    am_object_t *obj = am_value_to_ptr(node_val);
    if (obj->type == AM_OBJECT_TYPE_WSTRING) {
        am_wstring_t *ws = (am_wstring_t *)obj;
        fwprintf(out, L"<H:%zu>: WSTRING len=%zu \"", (size_t)handle, ws->length);
        for (size_t i = 0; i < ws->length; i++) {
            wchar_t c = (wchar_t)am_value_to_wchar(ws->content[i]);
            if (c == L'"') fputwc(L'\\', out);
            fwprintf(out, L"%lc", c);
        }
        fwprintf(out, L"\"\n");
        return;
    }

    if (obj->type != AM_OBJECT_TYPE_LIST) {
        fwprintf(out, L"<H:%zu>: (unknown type %d)\n", (size_t)handle, obj->type);
        return;
    }

    am_list_t *lst = (am_list_t *)obj;
    const wchar_t *type_name = L"UNKNOWN";
    switch (lst->type) {
        case AM_LIST_TYPE_LAMBDA:      type_name = L"LAMBDA"; break;
        case AM_LIST_TYPE_APPLICATION: type_name = L"APPLICATION"; break;
        case AM_LIST_TYPE_QUOTE:       type_name = L"QUOTE"; break;
        case AM_LIST_TYPE_QUASIQUOTE:  type_name = L"QUASIQUOTE"; break;
        case AM_LIST_TYPE_UNQUOTE:     type_name = L"UNQUOTE"; break;
    }

    fwprintf(out, L"<H:%zu>: %ls parent=", (size_t)handle, type_name);
    if (lst->parent == AM_HANDLE_NULL) {
        fwprintf(out, L"null");
    }
    else {
        fwprintf(out, L"<H:%zu>", (size_t)lst->parent);
    }

    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        am_uint_t n_param = 0;
        if (lst->length >= 2) {
            am_value_t np = am_list_get(ast->alloc, lst, 1);
            if (am_value_is_uint(np)) n_param = am_value_to_uint(np);
        }
        size_t n_body = (lst->length > 2 + n_param) ? (lst->length - 2 - n_param) : 0;
        fwprintf(out, L" params=%u bodies=%zu", (unsigned)n_param, n_body);
    }
    else {
        fwprintf(out, L" length=%zu", lst->length);
    }

    fwprintf(out, L" children=[");
    for (size_t i = 0; i < lst->length; i++) {
        if (i > 0) fwprintf(out, L", ");
        debug_ast_print_value_inline(ast, am_list_get(ast->alloc, lst, i), out);
    }
    fwprintf(out, L"]\n");
}


typedef struct {
    am_handle_t *handles;
    size_t count;
    size_t capacity;
} debug_ast_collect_handles_ctx_t;


static void debug_ast_collect_handles_cb(am_handle_t handle, am_value_t value, void *user_data) {
    (void)value;
    debug_ast_collect_handles_ctx_t *ctx = (debug_ast_collect_handles_ctx_t *)user_data;

    if (ctx->count >= ctx->capacity) {
        size_t new_capacity = ctx->capacity ? ctx->capacity * 2 : 16;
        am_handle_t *new_handles = (am_handle_t *)realloc(ctx->handles, new_capacity * sizeof(am_handle_t));
        if (!new_handles) return;
        ctx->handles = new_handles;
        ctx->capacity = new_capacity;
    }

    ctx->handles[ctx->count++] = handle;
}


static int debug_ast_compare_handles(const void *a, const void *b) {
    am_handle_t ha = *(const am_handle_t *)a;
    am_handle_t hb = *(const am_handle_t *)b;
    if (ha < hb) return -1;
    if (ha > hb) return 1;
    return 0;
}


static void debug_ast_print_nodes_map(FILE *out, am_ast_t *ast) {
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"nodes: {\n");

    debug_ast_collect_handles_ctx_t ctx = { NULL, 0, 0 };
    am_heap_iter(ast->alloc, ast->alloc, ast->nodes, debug_ast_collect_handles_cb, &ctx);

    if (ctx.handles) {
        qsort(ctx.handles, ctx.count, sizeof(am_handle_t), debug_ast_compare_handles);
        for (size_t i = 0; i < ctx.count; i++) {
            debug_ast_print_indent(out, 2);
            am_debug_ast_print_node_summary(out, ast, ctx.handles[i]);
        }
        free(ctx.handles);
    }

    debug_ast_print_indent(out, 1);
    fwprintf(out, L"}\n");
}


void am_debug_ast_print(FILE *out, am_ast_t *ast) {
    if (!ast) {
        fwprintf(out, L"null\n");
        return;
    }

    size_t visited_capacity = 256;
    am_handle_t *visited = (am_handle_t *)malloc(visited_capacity * sizeof(am_handle_t));
    if (!visited) {
        fwprintf(out, L"(failed to allocate visited buffer)\n");
        return;
    }
    size_t visited_count = 0;
    (void)visited_count;

    fwprintf(out, L"AST {\n");
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"absolute_path: \"%ls\"\n", ast->absolute_path ? ast->absolute_path : L"(null)");
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"module_id: \"%ls\"\n", ast->module_id ? ast->module_id : L"(null)");
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"top_lambda_handle: <H:%zu>\n", (size_t)ast->top_lambda_handle);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"token_count: %zu\n", ast->token_count);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"symbol_vocab: ");
    debug_ast_print_vocab(out, ast, ast->symbol_vocab);
    fputwc(L'\n', out);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"var_vocab: ");
    debug_ast_print_vocab(out, ast, ast->var_vocab);
    fputwc(L'\n', out);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"var_type: ");
    debug_ast_print_var_type(out, ast, ast->var_type);
    fputwc(L'\n', out);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"var_arn_mapping: ");
    debug_ast_print_var_arn_mapping(out, ast, ast->var_arn_mapping);
    fputwc(L'\n', out);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"lambda_handles: ");
    debug_ast_print_handle_list(out, ast, ast->lambda_handles);
    fputwc(L'\n', out);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"tailcall_handles: ");
    debug_ast_print_handle_list(out, ast, ast->tailcall_handles);
    fputwc(L'\n', out);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"var_top: ");
    debug_ast_print_value_list(out, ast, ast->var_top);
    fputwc(L'\n', out);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"dependencies: ");
    debug_ast_print_map_varid_to_handle(out, ast, ast->dependencies);
    fputwc(L'\n', out);
    debug_ast_print_indent(out, 1);
    fwprintf(out, L"natives: ");
    debug_ast_print_map_varid_to_handle(out, ast, ast->natives);
    fputwc(L'\n', out);

    debug_ast_print_nodes_map(out, ast);

    // debug_ast_print_indent(out, 1);
    // fwprintf(out, L"top_node: ");
    // am_handle_t top = am_ast_get_top_node_handle(ast);
    // if (top == AM_HANDLE_NULL) {
    //     fwprintf(out, L"null\n");
    // }
    // else {
    //     fputwc(L'\n', out);
    //     debug_ast_print_node(ast, top, out, &visited, &visited_count, &visited_capacity, 2);
    //     fputwc(L'\n', out);
    // }

    free(visited);
    fwprintf(out, L"}\n");
}


void am_debug_ast_print_to_stdout(am_ast_t *ast) {
    FILE *out = tmpfile();
    assert(out != NULL);

    am_debug_ast_print(out, ast);
    rewind(out);

    size_t cap = 4096;
    size_t len = 0;
    wchar_t *buf = (wchar_t *)malloc(cap * sizeof(wchar_t));
    assert(buf != NULL);

    wint_t c;
    while ((c = fgetwc(out)) != WEOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            wchar_t *new_buf = (wchar_t *)realloc(buf, cap * sizeof(wchar_t));
            if (!new_buf) {
                free(buf);
                fclose(out);
                return;
            }
            buf = new_buf;
        }
        buf[len++] = (wchar_t)c;
    }
    buf[len] = L'\0';
    fclose(out);

    printf("\n%ls\n", buf);
    free(buf);
}


// ===============================================================================
// 中间语言指令（IL Code）打印辅助
// ===============================================================================

const char *am_debug_opcode_name(uint32_t opcode) {
    switch (opcode) {
        case AM_VM_OP_nop:         return "nop";
        case AM_VM_OP_store:       return "store";
        case AM_VM_OP_load:        return "load";
        case AM_VM_OP_loadclosure: return "loadclosure";
        case AM_VM_OP_push:        return "push";
        case AM_VM_OP_pop:         return "pop";
        case AM_VM_OP_swap:        return "swap";
        case AM_VM_OP_set:         return "set";
        case AM_VM_OP_call:        return "call";
        case AM_VM_OP_callnative:  return "callnative";
        case AM_VM_OP_tailcall:    return "tailcall";
        case AM_VM_OP_return:      return "return";
        case AM_VM_OP_capturecc:   return "capturecc";
        case AM_VM_OP_iftrue:      return "iftrue";
        case AM_VM_OP_iffalse:     return "iffalse";
        case AM_VM_OP_goto:        return "goto";
        case AM_VM_OP_read:        return "read";
        case AM_VM_OP_write:       return "write";
        case AM_VM_OP_pause:       return "pause";
        case AM_VM_OP_halt:        return "halt";
        case AM_VM_OP_fork:        return "fork";
        case AM_VM_OP_display:     return "display";
        case AM_VM_OP_newline:     return "newline";
        case AM_VM_OP_add:         return "add";
        case AM_VM_OP_sub:         return "sub";
        case AM_VM_OP_mul:         return "mul";
        case AM_VM_OP_div:         return "div";
        case AM_VM_OP_mod:         return "mod";
        case AM_VM_OP_eq:          return "eq";
        case AM_VM_OP_eqv:         return "eqv";
        case AM_VM_OP_equal:       return "equal";
        case AM_VM_OP_ge:          return "ge";
        case AM_VM_OP_le:          return "le";
        case AM_VM_OP_gt:          return "gt";
        case AM_VM_OP_lt:          return "lt";
        case AM_VM_OP_not:         return "not";
        case AM_VM_OP_and:         return "and";
        case AM_VM_OP_or:          return "or";
        case AM_VM_OP_typeof:      return "typeof";
        case AM_VM_OP_car:         return "car";
        case AM_VM_OP_cdr:         return "cdr";
        case AM_VM_OP_cons:        return "cons";
        case AM_VM_OP_get_item:    return "get_item";
        case AM_VM_OP_set_item:    return "set_item";
        case AM_VM_OP_list_push:   return "list_push";
        case AM_VM_OP_list_pop:    return "list_pop";
        case AM_VM_OP_length:      return "length";
        case AM_VM_OP_concat:      return "concat";
        case AM_VM_OP_duplicate:   return "duplicate";
        case AM_VM_OP_evalcleanup: return "evalcleanup";
        case AM_VM_OP_dynamicwind:              return "dynamicwind";
        case AM_VM_OP_dynamicwind_after_before: return "dynamicwind_after_before";
        case AM_VM_OP_dynamicwind_before_after: return "dynamicwind_before_after";
        case AM_VM_OP_dynamicwind_done:         return "dynamicwind_done";
        case AM_VM_OP_wind:                     return "wind";
        default:                   return "?";
    }
}


void am_debug_print_operand(am_ast_t *ast, am_value_t op) {
    if (!ast) {
        printf("?");
        return;
    }

    if (am_value_is_varid(op)) {
        am_varid_t v = am_value_to_varid(op);
        wchar_t *name = am_vocab_get(ast->alloc, ast->var_vocab, &v);
        printf("%ls(%zu)", name ? name : L"?", (size_t)v);
    }
    else if (am_value_is_handle(op)) {
        printf("handle_%zu", am_value_to_handle(op));
    }
    else if (am_value_is_iaddr(op)) {
        printf("iaddr_%zu", am_value_to_iaddr(op));
    }
    else if (am_value_is_label(op)) {
        printf("label_%zu", am_value_to_label(op));
    }
    else if (am_value_is_symbol(op)) {
        am_symbol_t s = am_value_to_symbol(op);
        wchar_t *name = am_vocab_get(ast->alloc, ast->symbol_vocab, &s);
        printf("%ls", name ? name : L"?");
    }
    else if (am_value_is_uint(op)) {
        printf("%llu", (unsigned long long)am_value_to_uint(op));
    }
    else if (am_value_is_int(op)) {
        printf("%lld", (long long)am_value_to_int(op));
    }
    else if (am_value_is_float(op)) {
        printf("%g", (double)am_value_to_float(op));
    }
    else if (am_value_is_boolean(op)) {
        printf("%s", am_value_to_boolean(op) ? "#t" : "#f");
    }
    else if (am_value_is_null(op)) {
        printf("#null");
    }
    else if (am_value_is_undefined(op)) {
        printf("#undefined");
    }
    else {
        printf("?");
    }
}


void am_debug_print_ilcode(am_ast_t *ast, am_instruction_t *ilcode, am_iaddr_t icount) {
    if (!ilcode) return;

    for (am_iaddr_t i = 0; i < icount; i++) {
        printf("[%4zu] %-12s", (size_t)i, am_debug_opcode_name(ilcode[i].opcode));
        if (!am_value_is_undefined(ilcode[i].operand)) {
            printf(" ");
            am_debug_print_operand(ast, ilcode[i].operand);
        }
        printf("\n");
    }
}


void am_debug_print_ilcode_raw(am_instruction_t *ilcode, am_iaddr_t icount) {
    if (!ilcode) return;

    for (am_iaddr_t i = 0; i < icount; i++) {
        printf("[%4zu] %-12s operand=%016llx\n",
               (size_t)i,
               am_debug_opcode_name(ilcode[i].opcode),
               (unsigned long long)ilcode[i].operand);
    }
}
/* ===== end:   src/am_debug.c ===== */

/* ===== begin: src/am_lexer.c ===== */
#include <stdint.h>
#include <wchar.h>
#include <wctype.h>
#include <string.h>



// 关键字表
const wchar_t* AM_KEYWORDS[] = {
    L"lambda", L"define", L"set!", L"let", L"begin", L"return", L"...", L"_",
    L"if", L"and", L"or", L"cond", L"else", L"for", L"while", L"break", L"continue", L"case", L"do",
    L"quote", L"quasiquote", L"unquote",
    L"import", L"native",
    L"define-syntax", L"let-syntax", L"letrec-syntax", L"syntax-rules", NULL
};

// 关键字对应的保留symbol值，索引与AM_KEYWORDS一一对应
static const am_value_t AM_KEYWORD_SYMBOLS[] = {
    AM_VALUE_KW_lambda,
    AM_VALUE_KW_define,
    AM_VALUE_KW_set,
    AM_VALUE_KW_let,
    AM_VALUE_KW_begin,
    AM_VALUE_KW_return,
    AM_VALUE_KW_dot3,
    AM_VALUE_KW_underscore,
    AM_VALUE_KW_if,
    AM_VALUE_KW_and,
    AM_VALUE_KW_or,
    AM_VALUE_KW_cond,
    AM_VALUE_KW_else,
    AM_VALUE_KW_for,
    AM_VALUE_KW_while,
    AM_VALUE_KW_break,
    AM_VALUE_KW_continue,
    AM_VALUE_KW_case,
    AM_VALUE_KW_do,
    AM_VALUE_KW_quote,
    AM_VALUE_KW_quasiquote,
    AM_VALUE_KW_unquote,
    AM_VALUE_KW_import,
    AM_VALUE_KW_native,
    AM_VALUE_KW_define_syntax,
    AM_VALUE_KW_let_syntax,
    AM_VALUE_KW_letrec_syntax,
    AM_VALUE_KW_syntax_rules
};

// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 解析字符串字面量，返回长度；出错返回 -1
static int32_t parse_string(wchar_t *code, int32_t start, int32_t *end_pos) {
    if(code[start] != L'"') return -1;
    int32_t pos = start + 1;
    while(code[pos]) {
        if(code[pos] == L'"' && !is_escaped(code, pos)) {
            *end_pos = pos + 1;
            return pos - start + 1;
        }
        if(code[pos] == L'\n' || code[pos] == L'\r') return -1;
        pos++;
    }
    return -1;
}

// 更严格的数字字面值判断
// 是合法数字返回 1；不是数字返回 -1。
static int32_t is_number(wchar_t *code, int32_t start, int32_t len) {
    if(len == 0) return -1;
    wchar_t first = code[start];
    // 首字符必须是 +/- 或数字
    if(!(first == L'-' || first == L'+' || iswdigit(first))) return -1;

    int32_t has_digit = 0, has_dot = 0, has_exp = 0;
    for(int32_t i = 0; i < len; i++) {
        wchar_t c = code[start + i];
        if(iswdigit(c)) { has_digit = 1; }
        else if(c == L'.') {
            if(has_dot || has_exp) return -1; // 多个. 或 . 在指数后
            has_dot = 1;
        }
        else if(c == L'e' || c == L'E') {
            if(has_exp || !has_digit) return -1; // 多个e 或 e前无数字
            has_exp = 1;
            // e后可跟 +/-
            if(i+1 < len && (code[start+i+1] == L'+' || code[start+i+1] == L'-')) i++;
            has_digit = 0; // 指数部分需重新验证有数字
        }
        else if(c == L'+' || c == L'-') {
            // +/- 只能出现在开头或e/E之后
            if(i != 0 && code[start+i-1] != L'e' && code[start+i-1] != L'E') return -1;
        }
        else {
            return -1; // 非法字符
        }
    }
    return has_digit ? 1 : -1; // 必须至少有一个数字
}

// 判断是否为关键字。
// 是关键字返回在 AM_KEYWORDS 中的索引（非负）；不是关键字返回 -1。
static int32_t is_keyword(wchar_t *code, int32_t start, int32_t len) {
    if(len == 0) return -1;
    for(int32_t i = 0; AM_KEYWORDS[i]; i++) {
        if(wcsncmp(&code[start], AM_KEYWORDS[i], len) == 0 && AM_KEYWORDS[i][len] == L'\0') {
            return i;
        }
    }
    return -1;
}

// 解析 # 开头的特殊字面值
// 成功返回对应的 AM_TOKEN_TYPE_*（正数），并通过 len_out 输出长度；失败返回 -1。
static int32_t parse_hash_literal(wchar_t *code, int32_t start, int32_t *len_out) {
    if(code[start] != L'#') return -1;
    int32_t pos = start + 1;

    // #t / #f (布尔值)
    if(code[pos] == L't' || code[pos] == L'f') {
        wchar_t next = code[pos + 1];
        if(!is_delimiter(next) && !is_whitespace(next) && next != L'\0') {
            return -1; // #ta 不是合法布尔值
        }
        *len_out = 2;
        return AM_TOKEN_TYPE_BOOLEAN;
    }

    // #undefined
    if(wcsncmp(&code[pos], L"undefined", 9) == 0) {
        wchar_t next = code[pos + 9];
        if(!is_delimiter(next) && !is_whitespace(next) && next != L'\0') return -1;
        *len_out = 10; // # + undefined(9)
        return AM_TOKEN_TYPE_UNDEFINED;
    }

    // #null
    if(wcsncmp(&code[pos], L"null", 4) == 0) {
        wchar_t next = code[pos + 4];
        if(!is_delimiter(next) && !is_whitespace(next) && next != L'\0') return -1;
        *len_out = 5; // # + null(4)
        return AM_TOKEN_TYPE_NULL;
    }

    return -1; // 不识别的 # 字面值
}

static void update_pos(wchar_t c, int32_t *line, int32_t *column) {
    if(c == L'\n') { *line += 1; *column = 0; }
    else if(c == L'\r') {
        if(*column >= 0) { *line += 1; *column = 0; }
    }
    else if(c != L'\0') { (*column)++; }
}

// ===============================================================================
// 主Lexer函数
// ===============================================================================

int32_t am_lexer(wchar_t *code, am_token_t *tokens) {
    if(!code || !tokens) return -1;

    int32_t pos = 0, tok_cnt = 0;
    int32_t line = 1, col = 0;
    int32_t buf_start = -1, buf_line = -1, buf_col = -1;

#define EMIT(t_type, t_len, t_idx, t_id)    \
    do {                                    \
        am_token_t *t = &tokens[tok_cnt++]; \
        t->index = (t_idx);                 \
        t->length = (t_len);                \
        t->type = (t_type);                 \
        t->id = (t_id);                     \
        t->line = buf_line;                 \
        t->column = buf_col;                \
        buf_start = -1;                     \
    } while(0)

// 重写：类型判断逻辑匹配新Token定义
#define FLUSH()                                                 \
    do {                                                        \
        if(buf_start != -1) {                                   \
            int32_t len = pos - buf_start;                      \
            int32_t t_type = AM_TOKEN_TYPE_IDENTIFIER;          \
            size_t t_id = SIZE_MAX;                             \
            wchar_t first = code[buf_start];                    \
            /* #开头的特殊字面值 */                              \
            if(first == L'#') {                                 \
                int32_t hash_len;                               \
                int32_t hash_type = parse_hash_literal(code, buf_start, &hash_len); \
                if(hash_type >= 0 && hash_len == len) {         \
                    t_type = hash_type;                         \
                } else {                                        \
                    t_type = AM_TOKEN_TYPE_UNEXPECTED;          \
                }                                               \
            }                                                   \
            else if (first == L'\'') {                          \
                if (len > 1) t_type = AM_TOKEN_TYPE_SYMBOL;     \
                else t_type = AM_TOKEN_TYPE_QUOTE;              \
            }                                                   \
            /* 数字字面值 */                                     \
            else if(is_number(code, buf_start, len) >= 0) {     \
                t_type = AM_TOKEN_TYPE_NUMBER;                  \
            }                                                   \
            /* 关键字 */                                        \
            else {                                              \
                int32_t kw_idx = is_keyword(code, buf_start, len); \
                if (kw_idx >= 0) {                              \
                    t_type = AM_TOKEN_TYPE_KEYWORD;             \
                    t_id = am_value_to_symbol(AM_KEYWORD_SYMBOLS[kw_idx]); \
                }                                               \
            }                                                   \
            EMIT(t_type, len, buf_start, t_id);                 \
        }                                                       \
    } while(0)

    while(code[pos]) {
        wchar_t c = code[pos];

        /* 1. 注释: ; 到行尾 */
        if(c == L';' && !is_escaped(code, pos)) {
            FLUSH();
            while(code[pos] && code[pos] != L'\n' && code[pos] != L'\r')
                update_pos(code[pos++], &line, &col);
            if(code[pos]) update_pos(code[pos++], &line, &col);
            continue;
        }

        /* 2. 定界符处理 */
        if(is_delimiter(c) && !is_escaped(code, pos)) {
            FLUSH();

            // 2.1 字符串字面量
            if(c == L'"') {
                int32_t end_pos;
                int32_t len = parse_string(code, pos, &end_pos);
                if(len < 0) return -1;
                buf_line = line; buf_col = col;
                EMIT(AM_TOKEN_TYPE_STRING, len, pos, SIZE_MAX);
                while(pos < end_pos) update_pos(code[pos++], &line, &col);
                continue;
            }

            // 2.2 { 特殊转换: { -> ( + begin (虚拟token)
            if(c == L'{') {
                buf_line = line; buf_col = col;
                EMIT(AM_TOKEN_TYPE_LB, 1, pos, SIZE_MAX);

                am_token_t *t2 = &tokens[tok_cnt++];
                t2->index = SIZE_MAX; t2->length = 5; t2->type = AM_TOKEN_TYPE_KEYWORD;
                t2->id = am_value_to_symbol(AM_VALUE_KW_begin);
                t2->line = line; t2->column = col + 1;

                update_pos(c, &line, &col);
                pos++;
                continue;
            }

            // 2.4 普通定界符类型映射
            int32_t t_type;
            if(c == L'(' || c == L'[')
                t_type = AM_TOKEN_TYPE_LB;
            else if(c == L')' || c == L']' || c == L'}')
                t_type = AM_TOKEN_TYPE_RB;
            else if(c == L'`')
                t_type = AM_TOKEN_TYPE_QUASIQUOTE;
            else if(c == L',')
                t_type = AM_TOKEN_TYPE_UNQUOTE;
            else
                t_type = AM_TOKEN_TYPE_DELIMITER;

            buf_line = line; buf_col = col;
            EMIT(t_type, 1, pos, SIZE_MAX);
            update_pos(c, &line, &col);
            pos++;
            continue;
        }

        /* 3. 空白字符 */
        if(is_whitespace(c)) {
            FLUSH();
            update_pos(c, &line, &col);
            if(c == L'\r' && code[pos+1] == L'\n')
                update_pos(code[++pos], &line, &col);
            pos++;
            continue;
        }

        /* 4. 普通字符累积 */
        if(buf_start == -1) {
            buf_start = pos;
            buf_line = line;
            buf_col = col;
        }
        update_pos(c, &line, &col);
        pos++;
    }

    FLUSH();
    return tok_cnt;

#undef EMIT
#undef FLUSH
}

// 安全获取 token 文本（处理虚拟 token）
const wchar_t* token_text(am_token_t *tok, wchar_t *code) {
    static wchar_t buf[256];
    if(tok->index == SIZE_MAX) {
        // 虚拟 token
        if(tok->type == AM_TOKEN_TYPE_KEYWORD && tok->length == 5) return L"begin";
        return L"[virtual]";
    }
    // 真实 token
    int32_t len = tok->length < 255 ? tok->length : 255;
    wcsncpy(buf, &code[tok->index], len);
    buf[len] = L'\0';
    return buf;
}
/* ===== end:   src/am_lexer.c ===== */

/* ===== begin: src/am_ast.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>
#include <wctype.h>



// 全局内置变量
const wchar_t* AM_GLOBAL_BUILTIN_VAR[] = {
    L"+", L"-", L"*", L"/", L"mod", L"pow",
    L"not", L">", L"<", L">=", L"<=", L"==",
    L"eq?", L"eqv?", L"equal?", L"null?", L"undefined?", L"atom?", L"list?", L"number?", L"nan?", L"typeof",
    L"car", L"cdr", L"cons", L"get_item", L"set_item!", L"push", L"pop", L"length",
    L"display", L"newline", L"write", L"read", L"call/cc", L"fork", L"dynamic-wind", NULL
};



// 全局内置变量到 VM opcode 的映射表。
// 下标与 AM_GLOBAL_BUILTIN_VAR 一一对应；-1 表示该 builtin 没有对应 opcode。
// AM_GLOBAL_BUILTIN_VAR_NUM 定义在 ast.h
const int32_t AM_BUILTIN_OPCODE_MAP[AM_GLOBAL_BUILTIN_VAR_NUM] = {
    [0]  = AM_VM_OP_add,         // +
    [1]  = AM_VM_OP_sub,         // -
    [2]  = AM_VM_OP_mul,         // *
    [3]  = AM_VM_OP_div,         // /
    [4]  = AM_VM_OP_mod,         // mod
    [5]  = AM_VM_OP_pow,         // pow
    [6]  = AM_VM_OP_not,         // not
    [7]  = AM_VM_OP_gt,          // >
    [8]  = AM_VM_OP_lt,          // <
    [9]  = AM_VM_OP_ge,          // >=
    [10] = AM_VM_OP_le,          // <=
    [11] = AM_VM_OP_eqv,         // ==
    [12] = AM_VM_OP_eq,          // eq?
    [13] = AM_VM_OP_eqv,         // eqv?
    [14] = AM_VM_OP_equal,       // equal?
    [15] = AM_VM_OP_isnull,      // null?
    [16] = AM_VM_OP_isundef,     // undefined?
    [17] = AM_VM_OP_isatom,      // atom?
    [18] = AM_VM_OP_islist,      // list?
    [19] = AM_VM_OP_isnumber,    // number?
    [20] = AM_VM_OP_isnan,       // nan?
    [21] = AM_VM_OP_typeof,      // typeof
    [22] = AM_VM_OP_car,         // car
    [23] = AM_VM_OP_cdr,         // cdr
    [24] = AM_VM_OP_cons,        // cons
    [25] = AM_VM_OP_get_item,    // get_item
    [26] = AM_VM_OP_set_item,    // set_item!
    [27] = AM_VM_OP_list_push,   // push
    [28] = AM_VM_OP_list_pop,    // pop
    [29] = AM_VM_OP_length,      // length
    [30] = AM_VM_OP_display,     // display
    [31] = AM_VM_OP_newline,     // newline
    [32] = AM_VM_OP_write,       // write
    [33] = AM_VM_OP_read,        // read
    [34] = -1,                   // call/cc
    [35] = AM_VM_OP_fork,        // fork
    [36] = -1,                   // dynamic-wind
};



// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 将模块绝对路径转换为模块ID。
// 规则（对应TS的PathUtils.PathToModuleID）：将斜杠/反斜杠替换为点号，空格替换为下划线，去掉冒号，去掉.scm后缀。
// 返回新分配的 wchar_t*（使用ast分配器），失败返回NULL。
wchar_t *am_absolute_path_to_module_id(am_allocator_t *alloc, const wchar_t *absolute_path) {
    if (!absolute_path) return NULL;

    size_t len = wcslen(absolute_path);
    wchar_t *module_id = (wchar_t *)am_malloc(alloc, (len + 1) * sizeof(wchar_t));
    if (!module_id) return NULL;

    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        wchar_t c = absolute_path[i];
        if (i == 0 && c == L'/') {
            // 首字符为/时直接去掉
        }
        else if (c == L'/' || c == L'\\') {
            module_id[j++] = L'.';
        }
        else if (c == L' ') {
            module_id[j++] = L'_';
        }
        else if (c == L':') {
            // 去掉冒号
        }
        else {
            module_id[j++] = c;
        }
    }
    module_id[j] = L'\0';

    // 去掉末尾的 .scm（不区分大小写）
    size_t id_len = wcslen(module_id);
    if (id_len >= 4) {
        if (module_id[id_len - 4] == L'.' &&
            (module_id[id_len - 3] == L's' || module_id[id_len - 3] == L'S') &&
            (module_id[id_len - 2] == L'c' || module_id[id_len - 2] == L'C') &&
            (module_id[id_len - 1] == L'm' || module_id[id_len - 1] == L'M')) {
            module_id[id_len - 4] = L'\0';
        }
    }

    return module_id;
}


// 提取token对应的源码文本，返回新分配的以L'\0'结尾的宽字符串（使用系统malloc）。
// 调用者负责free。虚拟token（index == SIZE_MAX）返回NULL，但begin虚拟token返回L"begin"。
static wchar_t *am_token_text_dup(am_token_t *tok, wchar_t *code) {
    if (!tok) return NULL;

    // 虚拟token：仅begin返回对应文本，其他返回NULL
    if (tok->index == SIZE_MAX) {
        if (tok->type == AM_TOKEN_TYPE_KEYWORD && tok->length == 5) {
            wchar_t *buf = (wchar_t *)malloc(6 * sizeof(wchar_t));
            if (!buf) return NULL;
            wcscpy(buf, L"begin");
            return buf;
        }
        return NULL;
    }

    if (!code) return NULL;

    size_t len = tok->length;
    wchar_t *buf = (wchar_t *)malloc((len + 1) * sizeof(wchar_t));
    if (!buf) return NULL;

    wcsncpy(buf, &code[tok->index], len);
    buf[len] = L'\0';
    return buf;
}


// 根据token文本查找关键字在AM_KEYWORDS中的索引。
static size_t keyword_index(const wchar_t *text) {
    for (size_t i = 0; AM_KEYWORDS[i]; i++) {
        if (wcscmp(text, AM_KEYWORDS[i]) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}


// 辅助：从am_value_t解包出am_list_t*（不做类型检查，调用者应确保）
static am_list_t *value_to_list(am_value_t v) {
    return (am_list_t *)am_value_to_ptr(v);
}


// ===============================================================================
// 构造函数 / 析构函数 / 拷贝
// ===============================================================================

// 功能描述：创建AST对象。
am_ast_t *am_ast_create(am_allocator_t *alloc, wchar_t *code, wchar_t *absolute_path, am_token_t *tokens, size_t token_count) {
    if (!alloc) return NULL;

    am_ast_t *ast = (am_ast_t *)am_calloc(alloc, sizeof(am_ast_t));
    if (!ast) return NULL;

    ast->alloc = alloc;
    ast->code = code;
    ast->tokens = tokens;
    ast->token_count = token_count;
    ast->absolute_path = absolute_path;
    ast->module_id = am_absolute_path_to_module_id(alloc, absolute_path);
    if (!ast->module_id) {
        am_free(alloc, ast);
        return NULL;
    }

    ast->symbol_vocab = am_vocab_create(alloc, 64);
    ast->var_vocab = am_vocab_create(alloc, 64);
    ast->var_type = am_list_create(alloc, 64, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    ast->nodes = am_heap_create(alloc, alloc, 512);
    ast->node_token_mapping = am_map_create(alloc, 64);
    ast->strindex = am_strindex_create(alloc, 512);
    ast->scopes = am_map_create(alloc, 64);
    ast->var_arn_mapping = am_map_create(alloc, 64);
    ast->lambda_handles = am_list_create(alloc, 32, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    ast->tailcall_handles = am_list_create(alloc, 32, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    ast->var_top = am_list_create(alloc, 32, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    ast->dependencies = am_map_create(alloc, 16);
    ast->natives = am_map_create(alloc, 16);

    if (!ast->symbol_vocab || !ast->var_vocab || !ast->var_type || !ast->nodes ||
        !ast->node_token_mapping || !ast->strindex || !ast->scopes || !ast->var_arn_mapping ||
        !ast->lambda_handles || !ast->tailcall_handles || !ast->var_top ||
        !ast->dependencies || !ast->natives) {
        am_ast_destroy(ast);
        return NULL;
    }

    return ast;
}


// 功能描述：销毁AST对象。
int32_t am_ast_destroy(am_ast_t *ast) {
    if (!ast) return 0;

    am_allocator_t *alloc = ast->alloc;

    if (ast->tokens) am_free(alloc, ast->tokens);
    if (ast->module_id) am_free(alloc, ast->module_id);
    if (ast->symbol_vocab) am_vocab_destroy(alloc, ast->symbol_vocab);
    if (ast->var_vocab) am_vocab_destroy(alloc, ast->var_vocab);
    if (ast->var_type) am_list_destroy(alloc, ast->var_type);
    if (ast->nodes) am_heap_destroy(alloc, alloc, ast->nodes);
    if (ast->node_token_mapping) am_map_destroy(alloc, ast->node_token_mapping);
    if (ast->strindex) am_strindex_destroy(alloc, ast->strindex);
    if (ast->scopes) am_map_destroy(alloc, ast->scopes);
    if (ast->var_arn_mapping) am_map_destroy(alloc, ast->var_arn_mapping);
    if (ast->lambda_handles) am_list_destroy(alloc, ast->lambda_handles);
    if (ast->tailcall_handles) am_list_destroy(alloc, ast->tailcall_handles);
    if (ast->var_top) am_list_destroy(alloc, ast->var_top);
    if (ast->dependencies) am_map_destroy(alloc, ast->dependencies);
    if (ast->natives) am_map_destroy(alloc, ast->natives);

    am_free(alloc, ast);
    return 0;
}


// 功能描述：深拷贝AST对象。
am_ast_t *am_ast_copy(am_ast_t *ast) {
    if (!ast) return NULL;

    am_ast_t *copy = (am_ast_t *)am_calloc(ast->alloc, sizeof(am_ast_t));
    if (!copy) return NULL;

    copy->alloc = ast->alloc;
    copy->code = ast->code;
    copy->tokens = ast->tokens;
    copy->token_count = ast->token_count;
    copy->absolute_path = ast->absolute_path;
    copy->module_id = am_absolute_path_to_module_id(ast->alloc, ast->absolute_path);
    if (!copy->module_id) {
        am_free(ast->alloc, copy);
        return NULL;
    }

    copy->symbol_vocab = ast->symbol_vocab ? am_vocab_copy(ast->alloc, ast->symbol_vocab) : NULL;
    copy->var_vocab = ast->var_vocab ? am_vocab_copy(ast->alloc, ast->var_vocab) : NULL;
    copy->var_type = ast->var_type ? am_list_copy(ast->alloc, ast->var_type) : NULL;
    copy->nodes = ast->nodes ? am_heap_copy(ast->alloc, ast->alloc, ast->nodes) : NULL;
    copy->node_token_mapping = ast->node_token_mapping ? am_map_copy(ast->alloc, ast->node_token_mapping) : NULL;
    copy->strindex = ast->strindex ? am_strindex_copy(ast->alloc, ast->strindex) : NULL;
    copy->scopes = ast->scopes ? am_map_copy(ast->alloc, ast->scopes) : NULL;
    copy->var_arn_mapping = ast->var_arn_mapping ? am_map_copy(ast->alloc, ast->var_arn_mapping) : NULL;
    copy->lambda_handles = ast->lambda_handles ? am_list_copy(ast->alloc, ast->lambda_handles) : NULL;
    copy->tailcall_handles = ast->tailcall_handles ? am_list_copy(ast->alloc, ast->tailcall_handles) : NULL;
    copy->var_top = ast->var_top ? am_list_copy(ast->alloc, ast->var_top) : NULL;
    copy->dependencies = ast->dependencies ? am_map_copy(ast->alloc, ast->dependencies) : NULL;
    copy->natives = ast->natives ? am_map_copy(ast->alloc, ast->natives) : NULL;

    if (!copy->symbol_vocab || !copy->var_vocab || !copy->var_type || !copy->nodes ||
        !copy->node_token_mapping || !copy->strindex || !copy->scopes || !copy->var_arn_mapping ||
        !copy->lambda_handles || !copy->tailcall_handles || !copy->var_top ||
        !copy->dependencies || !copy->natives) {
        am_ast_destroy(copy);
        return NULL;
    }

    return copy;
}


// 功能描述：设置AST节点把柄对应的token索引。
int32_t am_ast_set_node_token_index(am_ast_t *ast, am_handle_t node_handle, size_t token_index) {
    if (!ast || !ast->node_token_mapping) return -1;
    am_map_t *map = am_map_set(ast->alloc, ast->node_token_mapping,
                                am_make_value_of_handle(node_handle),
                                am_make_value_of_uint((am_uint_t)token_index));
    if (!map) return -1;
    ast->node_token_mapping = map;
    return 0;
}


// 功能描述：获取AST节点把柄对应的token索引。
size_t am_ast_get_node_token_index(am_ast_t *ast, am_handle_t node_handle) {
    if (!ast || !ast->node_token_mapping) return SIZE_MAX;
    am_value_t v = am_map_get(ast->alloc, ast->node_token_mapping, am_make_value_of_handle(node_handle));
    if (am_value_is_uint(v)) {
        return (size_t)am_value_to_uint(v);
    }
    return SIZE_MAX;
}


// am_ast_merge 内部辅助：收集 importee->nodes 中所有把柄与值

typedef struct {
    am_handle_t old_handle;
    am_value_t  old_value;
} merge_node_entry_t;

typedef struct {
    merge_node_entry_t *entries;
    size_t              length;
    size_t              capacity;
} merge_node_collect_ctx_t;

static void merge_collect_node_cb(am_handle_t handle, am_value_t value, void *user_data) {
    merge_node_collect_ctx_t *ctx = (merge_node_collect_ctx_t *)user_data;

    // 词法作用域对象仅在编译期使用，不参与模块合并
    if (am_value_is_ptr(value)) {
        am_object_t *obj = am_value_to_ptr(value);
        if (obj->type == AM_OBJECT_TYPE_SCOPE) return;
    }

    if (ctx->length >= ctx->capacity) {
        size_t new_cap = ctx->capacity ? ctx->capacity * 2 : 16;
        merge_node_entry_t *new_entries = (merge_node_entry_t *)realloc(ctx->entries,
                                                                         new_cap * sizeof(merge_node_entry_t));
        if (!new_entries) return;
        ctx->entries = new_entries;
        ctx->capacity = new_cap;
    }
    ctx->entries[ctx->length].old_handle = handle;
    ctx->entries[ctx->length].old_value = value;
    ctx->length++;
}

static am_wstring_t *merge_copy_wstring(am_allocator_t *alloc, am_wstring_t *ws) {
    if (!ws) return NULL;
    size_t total_size = sizeof(am_wstring_t) + ws->length * sizeof(am_value_t);
    am_wstring_t *copy = (am_wstring_t *)am_malloc(alloc, total_size);
    if (!copy) return NULL;
    copy->base = ws->base;
    copy->length = ws->length;
    if (ws->length > 0) {
        memcpy(copy->content, ws->content, ws->length * sizeof(am_value_t));
    }
    return copy;
}


// 从 handle 列表中移除指定的 handle（按值过滤，in-place 缩短 length）。
static am_list_t *merge_remove_handle_from_list(am_allocator_t *alloc, am_list_t *lst, am_handle_t h) {
    if (!lst) return NULL;
    size_t write = 0;
    for (size_t i = 0; i < lst->length; i++) {
        am_value_t v = am_list_get(alloc, lst, i);
        if (am_value_is_handle(v) && am_value_to_handle(v) == h) continue;
        lst->children[write++] = v;
    }
    lst->length = write;
    return lst;
}


// 功能描述：将importee融合进importer，也就是importer吃掉importee。
// 实现说明：成功返回0；失败返回-1。
int32_t am_ast_merge(am_ast_t *importer, am_ast_t *importee, int32_t order) {
    if (!importer || !importee || !importer->alloc || !importee->alloc) return -1;

    // =============================================================================
    // 第1步：修改importee的元数据，将其映射/合并到importer
    // =============================================================================

    // 1.1 symbol 映射
    size_t symbol_count = importee->symbol_vocab ? importee->symbol_vocab->length : 0;
    am_map_t *symbol_merge_mapping = am_map_create(importer->alloc,
                                                    symbol_count > 0 ? symbol_count : 8);
    if (!symbol_merge_mapping) return -1;

    for (size_t i = 0; i < symbol_count; i++) {
        wchar_t *word = am_vocab_get(importee->alloc, importee->symbol_vocab, &i);
        if (!word) return -1;
        size_t new_idx;
        importer->symbol_vocab = am_vocab_insert(importer->alloc, importer->symbol_vocab, word, &new_idx);
        if (!importer->symbol_vocab || new_idx == SIZE_MAX) return -1;

        am_map_t *m = am_map_set(importer->alloc, symbol_merge_mapping,
                                  am_make_value_of_symbol((am_symbol_t)i),
                                  am_make_value_of_symbol((am_symbol_t)new_idx));
        if (!m) return -1;
        symbol_merge_mapping = m;
    }

    // 1.2 variable 映射及相关元数据
    size_t var_count = importee->var_vocab ? importee->var_vocab->length : 0;
    am_map_t *varid_merge_mapping = am_map_create(importer->alloc,
                                                   var_count > 0 ? var_count : 8);
    if (!varid_merge_mapping) return -1;

    for (size_t i = 0; i < var_count; i++) {
        wchar_t *word = am_vocab_get(importee->alloc, importee->var_vocab, &i);
        if (!word) return -1;
        size_t new_varid;
        importer->var_vocab = am_vocab_insert(importer->alloc, importer->var_vocab, word, &new_varid);
        if (!importer->var_vocab || new_varid == SIZE_MAX) return -1;

        am_map_t *m = am_map_set(importer->alloc, varid_merge_mapping,
                                  am_make_value_of_varid((am_varid_t)i),
                                  am_make_value_of_varid((am_varid_t)new_varid));
        if (!m) return -1;
        varid_merge_mapping = m;

        am_value_t vtype = am_list_get(importee->alloc, importee->var_type, i);
        if (new_varid >= importer->var_type->length) {
            am_list_t *vt = am_list_push(importer->alloc, importer->var_type, vtype);
            if (!vt) return -1;
            importer->var_type = vt;
        } else {
            if (am_list_set(importer->alloc, importer->var_type, new_varid, vtype) != 0) return -1;
        }
    }

    // var_top 迁移：将 importee 的顶级变量 varid 映射后追加到 importer，去重
    if (importee->var_top) {
        for (size_t i = 0; i < importee->var_top->length; i++) {
            am_value_t vv = am_list_get(importee->alloc, importee->var_top, i);
            if (!am_value_is_varid(vv)) continue;
            am_value_t mapped = am_map_get(importer->alloc, varid_merge_mapping, vv);
            if (!am_value_is_varid(mapped)) continue;

            // 检查 importer->var_top 中是否已存在相同 varid
            int duplicate = 0;
            for (size_t j = 0; j < importer->var_top->length; j++) {
                if (am_value_equal(am_list_get(importer->alloc, importer->var_top, j), mapped)) {
                    duplicate = 1;
                    break;
                }
            }
            if (duplicate) continue;

            am_list_t *lst = am_list_push(importer->alloc, importer->var_top, mapped);
            if (!lst) return -1;
            importer->var_top = lst;
        }
    }

    // 预先收集 importee->nodes 中所有节点，便于多遍扫描
    merge_node_collect_ctx_t node_ctx = { NULL, 0, 0 };
    if (importee->nodes) {
        am_heap_iter(importee->alloc, importee->alloc, importee->nodes, merge_collect_node_cb, &node_ctx);
    }
    size_t n_nodes = node_ctx.length;

    // 1.3 handle 映射
    am_map_t *handle_merge_mapping = am_map_create(importer->alloc,
                                                    n_nodes > 0 ? n_nodes : 8);
    if (!handle_merge_mapping) {
        free(node_ctx.entries);
        return -1;
    }

    for (size_t i = 0; i < n_nodes; i++) {
        am_handle_t new_handle = am_heap_alloc_handle(importer->alloc, importer->alloc, importer->nodes);
        if (new_handle == AM_HANDLE_NULL) {
            free(node_ctx.entries);
            return -1;
        }
        am_map_t *m = am_map_set(importer->alloc, handle_merge_mapping,
                                  am_make_value_of_handle(node_ctx.entries[i].old_handle),
                                  am_make_value_of_handle(new_handle));
        if (!m) {
            free(node_ctx.entries);
            return -1;
        }
        handle_merge_mapping = m;
    }

    // 迁移 lambda_handles
    if (importee->lambda_handles) {
        for (size_t i = 0; i < importee->lambda_handles->length; i++) {
            am_value_t old_hv = am_list_get(importee->alloc, importee->lambda_handles, i);
            am_value_t new_hv = am_map_get(importer->alloc, handle_merge_mapping, old_hv);
            if (!am_value_is_handle(new_hv)) continue;
            am_list_t *lst = am_list_push(importer->alloc, importer->lambda_handles, new_hv);
            if (!lst) {
                free(node_ctx.entries);
                return -1;
            }
            importer->lambda_handles = lst;
        }
    }

    // 迁移 tailcall_handles
    if (importee->tailcall_handles) {
        for (size_t i = 0; i < importee->tailcall_handles->length; i++) {
            am_value_t old_hv = am_list_get(importee->alloc, importee->tailcall_handles, i);
            am_value_t new_hv = am_map_get(importer->alloc, handle_merge_mapping, old_hv);
            if (!am_value_is_handle(new_hv)) continue;
            am_list_t *lst = am_list_push(importer->alloc, importer->tailcall_handles, new_hv);
            if (!lst) {
                free(node_ctx.entries);
                return -1;
            }
            importer->tailcall_handles = lst;
        }
    }

    // 迁移 dependencies
    if (importee->dependencies) {
        size_t dep_count = am_map_length(importee->alloc, importee->dependencies);
        am_value_t *dep_keys = am_map_keys(importee->alloc, importee->dependencies);
        for (size_t i = 0; i < dep_count; i++) {
            am_value_t old_varid_val = dep_keys[i];
            am_value_t old_h_val = am_map_get(importee->alloc, importee->dependencies, old_varid_val);
            am_value_t new_varid_val = am_map_get(importer->alloc, varid_merge_mapping, old_varid_val);
            am_value_t new_h_val = am_map_get(importer->alloc, handle_merge_mapping, old_h_val);
            if (am_value_is_varid(new_varid_val) && am_value_is_handle(new_h_val)) {
                am_map_t *m = am_map_set(importer->alloc, importer->dependencies,
                                          new_varid_val, new_h_val);
                if (!m) {
                    am_free(importee->alloc, dep_keys);
                    free(node_ctx.entries);
                    return -1;
                }
                importer->dependencies = m;
            }
        }
        am_free(importee->alloc, dep_keys);
    }

    // 迁移 natives
    if (importee->natives) {
        size_t nat_count = am_map_length(importee->alloc, importee->natives);
        am_value_t *nat_keys = am_map_keys(importee->alloc, importee->natives);
        for (size_t i = 0; i < nat_count; i++) {
            am_value_t old_varid_val = nat_keys[i];
            am_value_t old_h_val = am_map_get(importee->alloc, importee->natives, old_varid_val);
            am_value_t new_varid_val = am_map_get(importer->alloc, varid_merge_mapping, old_varid_val);
            am_value_t new_h_val = am_map_get(importer->alloc, handle_merge_mapping, old_h_val);
            if (am_value_is_varid(new_varid_val) && am_value_is_handle(new_h_val)) {
                am_map_t *m = am_map_set(importer->alloc, importer->natives,
                                          new_varid_val, new_h_val);
                if (!m) {
                    am_free(importee->alloc, nat_keys);
                    free(node_ctx.entries);
                    return -1;
                }
                importer->natives = m;
            }
        }
        am_free(importee->alloc, nat_keys);
    }

    // 第三遍扫描：替换所有 list 节点 children 中的 symbol/varid/handle
    for (size_t i = 0; i < n_nodes; i++) {
        am_value_t val = node_ctx.entries[i].old_value;
        if (!am_value_is_ptr(val)) {
            free(node_ctx.entries);
            return -1;
        }
        am_object_t *obj = am_value_to_ptr(val);
        if (obj->type != AM_OBJECT_TYPE_LIST) continue;

        am_list_t *lst = (am_list_t *)obj;
        for (size_t j = 0; j < lst->length; j++) {
            am_value_t child = am_list_get(importee->alloc, lst, j);
            am_value_t new_child = child;
            int replaced = 0;
            if (am_value_is_symbol(child)) {
                new_child = am_map_get(importer->alloc, symbol_merge_mapping, child);
                if (am_value_is_symbol(new_child)) replaced = 1;
            } else if (am_value_is_varid(child)) {
                new_child = am_map_get(importer->alloc, varid_merge_mapping, child);
                if (am_value_is_varid(new_child)) replaced = 1;
            } else if (am_value_is_handle(child)) {
                new_child = am_map_get(importer->alloc, handle_merge_mapping, child);
                if (am_value_is_handle(new_child)) replaced = 1;
            }
            if (replaced && new_child != child) {
                if (am_list_set(importee->alloc, lst, j, new_child) != 0) {
                    free(node_ctx.entries);
                    return -1;
                }
            }
        }
    }

    // 在拷贝节点之前先确定 importer / importee 的顶层 lambda，
    // 避免 importee 顶层 application 也被拷贝到 importer 后产生查找歧义。
    am_handle_t importer_top_lambda = importer->top_lambda_handle;
    if (importer_top_lambda == AM_HANDLE_NULL ||
        am_heap_has_handle(importer->alloc, importer->alloc, importer->nodes, importer_top_lambda) != 0) {
        importer_top_lambda = am_ast_get_top_lambda_node_handle(importer);
    }
    am_handle_t importee_top_lambda = importee->top_lambda_handle;
    if (importee_top_lambda == AM_HANDLE_NULL ||
        am_heap_has_handle(importee->alloc, importee->alloc, importee->nodes, importee_top_lambda) != 0) {
        importee_top_lambda = am_ast_get_top_lambda_node_handle(importee);
    }
    if (importer_top_lambda == AM_HANDLE_NULL || importee_top_lambda == AM_HANDLE_NULL) {
        free(node_ctx.entries);
        return -1;
    }

    // =============================================================================
    // 第2步：将 importee 的所有 nodes 深拷贝到 importer->nodes 中
    // =============================================================================
    for (size_t i = 0; i < n_nodes; i++) {
        am_value_t old_h_val = am_make_value_of_handle(node_ctx.entries[i].old_handle);
        am_value_t new_h_val = am_map_get(importer->alloc, handle_merge_mapping, old_h_val);
        if (!am_value_is_handle(new_h_val)) continue;
        am_handle_t new_h = am_value_to_handle(new_h_val);

        am_value_t val = node_ctx.entries[i].old_value;
        if (!am_value_is_ptr(val)) {
            free(node_ctx.entries);
            return -1;
        }
        am_object_t *obj = am_value_to_ptr(val);
        am_value_t new_val;
        if (obj->type == AM_OBJECT_TYPE_LIST) {
            am_list_t *old_lst = (am_list_t *)obj;
            am_list_t *new_lst = am_list_copy(importer->alloc, old_lst);
            if (!new_lst) {
                free(node_ctx.entries);
                return -1;
            }
            if (new_lst->parent != AM_HANDLE_NULL) {
                am_value_t mapped_parent = am_map_get(importer->alloc, handle_merge_mapping,
                                                       am_make_value_of_handle(new_lst->parent));
                if (am_value_is_handle(mapped_parent)) {
                    new_lst->parent = am_value_to_handle(mapped_parent);
                }
            }
            new_val = am_make_value_of_ptr((am_object_t *)new_lst);
        } else if (obj->type == AM_OBJECT_TYPE_WSTRING) {
            am_wstring_t *new_ws = merge_copy_wstring(importer->alloc, (am_wstring_t *)obj);
            if (!new_ws) {
                free(node_ctx.entries);
                return -1;
            }
            new_val = am_make_value_of_ptr((am_object_t *)new_ws);
        } else if (obj->type == AM_OBJECT_TYPE_SCOPE) {
            // scope 对象是编译期词法作用域，不参与模块合并
            continue;
        } else {
            free(node_ctx.entries);
            return -1;
        }

        if (am_heap_set(importer->alloc, importer->alloc, importer->nodes, new_h, new_val) != 0) {
            free(node_ctx.entries);
            return -1;
        }
    }

    // =============================================================================
    // 第2.5步：合并 strindex（机械合并：把 importee 的 hash/handle 映射后插入 importer）
    // =============================================================================
    if (importee->strindex && importee->strindex->length > 0) {
        for (size_t i = 0; i < importee->strindex->capacity; i++) {
            uint32_t hash = importee->strindex->slots[i].hash;
            if (hash == AM_STRINDEX_KEY_EMPTY || hash == AM_STRINDEX_KEY_TOMBSTONE) continue;

            am_value_t old_h_val = importee->strindex->slots[i].value;
            if (!am_value_is_handle(old_h_val)) continue;

            am_value_t new_h_val = am_map_get(importer->alloc, handle_merge_mapping, old_h_val);
            if (!am_value_is_handle(new_h_val)) continue;

            am_strindex_t *new_si = am_strindex_set_raw(importer->alloc, importer->strindex,
                                                         hash, new_h_val);
            if (!new_si) {
                free(node_ctx.entries);
                return -1;
            }
            importer->strindex = new_si;
        }
    }

    // =============================================================================
    // 第3步：将 importee 的顶级节点嫁接到 importer 的顶层作用域
    // =============================================================================

    // importee 顶层 lambda 的函数体
    am_value_t importee_top_lambda_val = am_heap_get(importee->alloc, importee->alloc, importee->nodes,
                                                       importee_top_lambda);
    if (!am_value_is_ptr(importee_top_lambda_val)) {
        free(node_ctx.entries);
        return -1;
    }
    am_list_t *importee_top_lambda_lst = (am_list_t *)am_value_to_ptr(importee_top_lambda_val);
    am_handle_t importee_top_app = importee_top_lambda_lst->parent;
    size_t n_importee_bodies = 0;
    am_value_t *importee_bodies = am_list_lambda_get_bodies(importee->alloc,
                                                              importee_top_lambda_lst,
                                                              &n_importee_bodies);

    // importee_bodies 中的 handle 已在第1步第三遍扫描中被替换为 importer->nodes 中的新 handle，
    // 因此可以直接使用，无需再次查表映射。
    if (n_importee_bodies > 0 && !importee_bodies) {
        free(node_ctx.entries);
        return -1;
    }

    // 将嫁接到 importer 的顶级节点的 parent 修正为 importer 顶层 lambda
    for (size_t i = 0; i < n_importee_bodies; i++) {
        am_value_t body = importee_bodies[i];
        if (am_value_is_handle(body)) {
            am_handle_t body_h = am_value_to_handle(body);
            am_value_t body_val = am_heap_get(importer->alloc, importer->alloc, importer->nodes, body_h);
            if (am_value_is_ptr(body_val)) {
                am_object_t *body_obj = am_value_to_ptr(body_val);
                if (body_obj->type == AM_OBJECT_TYPE_LIST) {
                    ((am_list_t *)body_obj)->parent = importer_top_lambda;
                }
            }
        }
    }

    // importer 现有的顶层函数体
    am_value_t importer_top_lambda_val = am_heap_get(importer->alloc, importer->alloc, importer->nodes,
                                                       importer_top_lambda);
    if (!am_value_is_ptr(importer_top_lambda_val)) {
        free(importee_bodies);
        free(node_ctx.entries);
        return -1;
    }
    am_list_t *importer_top_lambda_lst = (am_list_t *)am_value_to_ptr(importer_top_lambda_val);
    size_t n_importer_bodies = 0;
    am_value_t *importer_bodies = am_list_lambda_get_bodies(importer->alloc,
                                                              importer_top_lambda_lst,
                                                              &n_importer_bodies);

    size_t total_bodies = n_importee_bodies + n_importer_bodies;
    am_value_t *new_bodies = NULL;
    if (total_bodies > 0) {
        new_bodies = (am_value_t *)malloc(total_bodies * sizeof(am_value_t));
        if (!new_bodies) {
            free(importer_bodies);
            free(importee_bodies);
            free(node_ctx.entries);
            return -1;
        }
        if (order == 0) {
            if (n_importee_bodies > 0) {
                memcpy(new_bodies, importee_bodies,
                       n_importee_bodies * sizeof(am_value_t));
            }
            if (n_importer_bodies > 0) {
                memcpy(new_bodies + n_importee_bodies, importer_bodies,
                       n_importer_bodies * sizeof(am_value_t));
            }
        } else {
            if (n_importer_bodies > 0) {
                memcpy(new_bodies, importer_bodies,
                       n_importer_bodies * sizeof(am_value_t));
            }
            if (n_importee_bodies > 0) {
                memcpy(new_bodies + n_importer_bodies, importee_bodies,
                       n_importee_bodies * sizeof(am_value_t));
            }
        }
    }

    int32_t set_result = 0;
    if (total_bodies > 0) {
        am_list_t *new_lambda = am_list_lambda_set_bodies(importer->alloc, importer_top_lambda_lst,
                                                          new_bodies, &total_bodies);
        if (!new_lambda) {
            set_result = -1;
        } else if (new_lambda != importer_top_lambda_lst) {
            if (am_heap_set(importer->alloc, importer->alloc, importer->nodes, importer_top_lambda,
                            am_make_value_of_ptr((am_object_t *)new_lambda)) != 0) {
                /* 扩容成功但 heap 更新失败：新 lambda 已不再被任何 handle 引用，
                 * 必须释放，否则会造成泄漏。 */
                am_list_destroy(importer->alloc, new_lambda);
                set_result = -1;
            }
        }
    }

    // 清理 importee 遗留的、已不可达的顶层 application 与顶层 lambda 节点
    if (importee_top_app != AM_HANDLE_NULL) {
        am_value_t dead_app_val = am_map_get(importer->alloc, handle_merge_mapping,
                                              am_make_value_of_handle(importee_top_app));
        am_value_t dead_lambda_val = am_map_get(importer->alloc, handle_merge_mapping,
                                                 am_make_value_of_handle(importee_top_lambda));
        if (am_value_is_handle(dead_app_val) && am_value_is_handle(dead_lambda_val)) {
            am_handle_t dead_app_h = am_value_to_handle(dead_app_val);
            am_handle_t dead_lambda_h = am_value_to_handle(dead_lambda_val);

            // 从 lambda_handles / tailcall_handles 中移除对这些死节点的引用
            importer->lambda_handles = merge_remove_handle_from_list(importer->alloc,
                                                                      importer->lambda_handles,
                                                                      dead_lambda_h);
            importer->tailcall_handles = merge_remove_handle_from_list(importer->alloc,
                                                                        importer->tailcall_handles,
                                                                        dead_app_h);

            // 从 importer->nodes 中释放死节点
            am_heap_free_handle(importer->alloc, importer->alloc, importer->nodes, dead_app_h);
            am_heap_free_handle(importer->alloc, importer->alloc, importer->nodes, dead_lambda_h);
        }
    }

    free(new_bodies);
    free(importer_bodies);
    free(importee_bodies);
    free(node_ctx.entries);

    am_map_destroy(importer->alloc, symbol_merge_mapping);
    am_map_destroy(importer->alloc, varid_merge_mapping);
    am_map_destroy(importer->alloc, handle_merge_mapping);

    if (set_result != 0) return -1;
    return 0;
}


// ===============================================================================
// 词汇表构建
// ===============================================================================

// 功能描述：遍历tokens，使用其中的KEYWORD和SYMBOL构建ast->symbol_vocab。
size_t am_build_symbol_vocabulary(am_ast_t *ast) {
    if (!ast || !ast->symbol_vocab || !ast->tokens) return 0;

    // 预置所有关键字到symbol_vocab的前端条目，索引与AM_VALUE_KW_*常量一致
    for (size_t i = 0; AM_KEYWORDS[i]; i++) {
        size_t idx;
        ast->symbol_vocab = am_vocab_insert(ast->alloc, ast->symbol_vocab, (wchar_t *)AM_KEYWORDS[i], &idx);
        if (!ast->symbol_vocab || idx == SIZE_MAX) return 0;
    }

    for (size_t i = 0; i < ast->token_count; i++) {
        am_token_t *t = &ast->tokens[i];
        // if (t->index == SIZE_MAX) continue; // 跳过虚拟token

        if (t->type == AM_TOKEN_TYPE_KEYWORD) {
            wchar_t *text = am_token_text_dup(t, ast->code);
            if (!text) return 0;
            size_t kw_idx = keyword_index(text);
            free(text);
            if (kw_idx != SIZE_MAX) {
                t->id = kw_idx;
            }
        }
        else if (t->type == AM_TOKEN_TYPE_SYMBOL) {
            wchar_t *text = am_token_text_dup(t, ast->code);
            if (!text) return 0;
            size_t idx;
            ast->symbol_vocab = am_vocab_insert(ast->alloc, ast->symbol_vocab, text, &idx);
            free(text);
            if (!ast->symbol_vocab || idx == SIZE_MAX) return 0;
            t->id = idx;
        }
    }

    return ast->symbol_vocab->length;
}


// 功能描述：遍历tokens，使用其中的IDENTIFIER构建ast->var_vocab。
size_t am_build_variable_vocabulary(am_ast_t *ast) {
    if (!ast || !ast->var_vocab || !ast->var_type || !ast->tokens) return 0;

    for (size_t i = 0; i < ast->token_count; i++) {
        am_token_t *t = &ast->tokens[i];
        if (t->index == SIZE_MAX) continue; // 跳过虚拟token

        if (t->type == AM_TOKEN_TYPE_IDENTIFIER) {
            wchar_t *text = am_token_text_dup(t, ast->code);
            if (!text) return 0;
            size_t old_len = ast->var_vocab->length;
            size_t idx;
            ast->var_vocab = am_vocab_insert(ast->alloc, ast->var_vocab, text, &idx);
            free(text);
            if (!ast->var_vocab || idx == SIZE_MAX) return 0;
            // 新变量加入时，同步在 var_type 中追加默认类型
            if (idx == old_len) {
                am_list_t *vt = am_list_push(ast->alloc, ast->var_type,
                                              am_make_value_of_uint(AM_VAR_TYPE_OLD));
                if (!vt) return 0;
                ast->var_type = vt;
            }
            t->id = idx;
        }
    }

    return ast->var_vocab->length;
}


// ===============================================================================
// EXT/NATIVE/IMPORT 引用判断
// ===============================================================================

// 功能描述：判断某个变量在形式上是否是“前缀.后缀”的格式（EXT_REF）。
int32_t am_ast_check_ext_ref(am_ast_t *ast, am_varid_t v) {
    if (!ast || !ast->var_vocab) return -1;

    wchar_t *var_str = am_vocab_get(ast->alloc, ast->var_vocab, &v);
    if (!var_str) return -1;

    // 必须有且仅有一个点号，且不在开头和末尾
    wchar_t *first_dot = wcschr(var_str, L'.');
    if (!first_dot || first_dot == var_str || first_dot[1] == L'\0') return -1;
    if (wcschr(first_dot + 1, L'.')) return -1;

    return 0;
}


// 功能描述：判断某个变量是否是 AM_VAR_TYPE_NATIVE_REF。
int32_t am_ast_check_native_ref(am_ast_t *ast, am_varid_t v) {
    if (!ast || !ast->var_vocab || !ast->natives) return -1;

    wchar_t *var_str = am_vocab_get(ast->alloc, ast->var_vocab, &v);
    if (!var_str) return -1;

    // 提取点号分隔的第1部分
    size_t len = wcslen(var_str);
    wchar_t *prefix = (wchar_t *)am_malloc(ast->alloc, (len + 1) * sizeof(wchar_t));
    if (!prefix) return -1;

    size_t i = 0;
    while (i < len && var_str[i] != L'.') {
        prefix[i] = var_str[i];
        i++;
    }
    prefix[i] = L'\0';

    size_t native_varid = am_vocab_find(ast->alloc, ast->var_vocab, prefix);
    am_free(ast->alloc, prefix);
    if (native_varid == SIZE_MAX) return -1;

    return am_map_contains(ast->alloc, ast->natives, am_make_value_of_varid(native_varid));
}


// 功能描述：判断某个变量是否是 AM_VAR_TYPE_IMPORT_REF。
int32_t am_ast_check_import_ref(am_ast_t *ast, am_varid_t v) {
    if (!ast || !ast->var_vocab || !ast->dependencies) return -1;

    wchar_t *var_str = am_vocab_get(ast->alloc, ast->var_vocab, &v);
    if (!var_str) return -1;

    // 提取最后一个点号分隔的第1部分作为 alias
    wchar_t *last_dot = wcsrchr(var_str, L'.');
    if (!last_dot || last_dot == var_str) return -1;

    size_t prefix_len = (size_t)(last_dot - var_str);
    wchar_t *prefix = (wchar_t *)am_malloc(ast->alloc, (prefix_len + 1) * sizeof(wchar_t));
    if (!prefix) return -1;
    wcsncpy(prefix, var_str, prefix_len);
    prefix[prefix_len] = L'\0';

    size_t alias_varid = am_vocab_find(ast->alloc, ast->var_vocab, prefix);
    am_free(ast->alloc, prefix);
    if (alias_varid == SIZE_MAX) return -1;

    return am_map_contains(ast->alloc, ast->dependencies, am_make_value_of_varid(alias_varid));
}


// ===============================================================================
// 节点访问
// ===============================================================================

// 功能描述：根据把柄从AST->nodes堆中获取相应的am_value_t。
am_value_t am_ast_get_node(am_ast_t *ast, am_handle_t handle) {
    if (!ast || !ast->nodes) return AM_VALUE_UNDEFINED;
    return am_heap_get(ast->alloc, ast->alloc, ast->nodes, handle);
}


// ===============================================================================
// 节点创建
// ===============================================================================

// 功能描述：创建lambda对象，返回其在AST->nodes堆中的把柄。
am_handle_t am_ast_make_lambda_node(am_ast_t *ast, am_handle_t parent) {
    if (!ast || !ast->nodes) return AM_HANDLE_NULL;

    am_handle_t handle = am_heap_alloc_handle(ast->alloc, ast->alloc, ast->nodes);
    if (handle == AM_HANDLE_NULL) return AM_HANDLE_NULL;

    am_list_t *lambda = am_list_create(ast->alloc, 32, AM_LIST_TYPE_LAMBDA, parent);
    if (!lambda) {
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        return AM_HANDLE_NULL;
    }

    // Lambda表结构：children[0] = 'lambda, children[1] = 参数数量(uint)
    lambda = am_list_push(ast->alloc, lambda, AM_VALUE_KW_lambda);
    if (!lambda) {
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        return AM_HANDLE_NULL;
    }
    lambda = am_list_push(ast->alloc, lambda, am_make_value_of_uint(0));
    if (!lambda) {
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        return AM_HANDLE_NULL;
    }

    if (am_heap_set(ast->alloc, ast->alloc, ast->nodes, handle, am_make_value_of_ptr((am_object_t *)lambda)) != 0) {
        am_list_destroy(ast->alloc, lambda);
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        return AM_HANDLE_NULL;
    }

    am_list_t *lst = am_list_push(ast->alloc, ast->lambda_handles, am_make_value_of_handle(handle));
    if (!lst) {
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        return AM_HANDLE_NULL;
    }
    ast->lambda_handles = lst;

    return handle;
}


// 功能描述：创建SList对象，返回其在AST->nodes堆中的把柄。
am_handle_t am_ast_make_slist_node(am_ast_t *ast, am_handle_t parent, int32_t type) {
    if (!ast || !ast->nodes) return AM_HANDLE_NULL;
    if (type != AM_LIST_TYPE_APPLICATION && type != AM_LIST_TYPE_QUOTE &&
        type != AM_LIST_TYPE_QUASIQUOTE && type != AM_LIST_TYPE_UNQUOTE) {
        return AM_HANDLE_NULL;
    }

    am_handle_t handle = am_heap_alloc_handle(ast->alloc, ast->alloc, ast->nodes);
    if (handle == AM_HANDLE_NULL) return AM_HANDLE_NULL;

    am_list_t *lst = am_list_create(ast->alloc, 32, type, parent);
    if (!lst) {
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        return AM_HANDLE_NULL;
    }

    if (am_heap_set(ast->alloc, ast->alloc, ast->nodes, handle, am_make_value_of_ptr((am_object_t *)lst)) != 0) {
        am_list_destroy(ast->alloc, lst);
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        return AM_HANDLE_NULL;
    }

    return handle;
}


// 功能描述：创建WString对象，返回其在AST->nodes堆中的把柄。
// 实现说明：基于全局字符串驻留索引 strindex 实现同值复用。先查索引，若存在相同内容
//         的字符串则复用其 handle；否则新建对象并登记到 strindex。

// 将词法层面截取的字符串字面量内容做转义还原。
// 支持 \" \\ \n \t \r；未知转义序列保留反斜杠与原字符。
static size_t ast_unescape_string(wchar_t *dst, const wchar_t *src, size_t len) {
    size_t i = 0, j = 0;
    while (i < len) {
        if (src[i] == L'\\' && i + 1 < len) {
            switch (src[i + 1]) {
                case L'"': dst[j++] = L'"'; i += 2; continue;
                case L'\\': dst[j++] = L'\\'; i += 2; continue;
                case L'n': dst[j++] = L'\n'; i += 2; continue;
                case L't': dst[j++] = L'\t'; i += 2; continue;
                case L'r': dst[j++] = L'\r'; i += 2; continue;
                default: break;
            }
        }
        dst[j++] = src[i++];
    }
    dst[j] = L'\0';
    return j;
}

am_handle_t am_ast_make_wstring_node(am_ast_t *ast, am_token_t *str_token) {
    if (!ast || !ast->nodes || !ast->strindex || !str_token) return AM_HANDLE_NULL;

    // 从token指示的位置截取字符串（去掉两侧引号）
    size_t len = str_token->length;
    if (len >= 2) len -= 2;
    wchar_t *text = (wchar_t *)malloc((len + 1) * sizeof(wchar_t));
    if (!text) return AM_HANDLE_NULL;
    wcsncpy(text, &ast->code[str_token->index + 1], len);
    text[len] = L'\0';

    // 还原转义序列
    size_t unescaped_len = ast_unescape_string(text, text, len);
    len = unescaped_len;

    uint32_t hash = am_strindex_hash_string(text);

    // 在 strindex 中查找候选 handle
    size_t n_candidates = am_strindex_get_all(ast->alloc, ast->strindex, text, NULL, 0);
    if (n_candidates > 0) {
        am_value_t *candidates = (am_value_t *)malloc(n_candidates * sizeof(am_value_t));
        if (!candidates) {
            free(text);
            return AM_HANDLE_NULL;
        }
        size_t got = am_strindex_get_all(ast->alloc, ast->strindex, text, candidates, n_candidates);

        for (size_t i = 0; i < got; i++) {
            am_handle_t cand_h = am_value_to_handle(candidates[i]);
            am_value_t cand_val = am_heap_get(ast->alloc, ast->alloc, ast->nodes, cand_h);
            if (!am_value_is_ptr(cand_val)) continue;
            am_object_t *obj = am_value_to_ptr(cand_val);
            if (obj->type != AM_OBJECT_TYPE_WSTRING) continue;
            am_wstring_t *ws = (am_wstring_t *)obj;

            // 先比长度，再比内容
            if (ws->length != len) continue;
            bool match = true;
            for (size_t j = 0; j < len; j++) {
                am_wchar_t wc = am_value_to_wchar(ws->content[j]);
                if (wc != (am_wchar_t)text[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                free(candidates);
                free(text);
                return cand_h;
            }
        }
        free(candidates);
    }

    // 不存在可复用的字符串，新建对象
    am_handle_t handle = am_heap_alloc_handle(ast->alloc, ast->alloc, ast->nodes);
    if (handle == AM_HANDLE_NULL) {
        free(text);
        return AM_HANDLE_NULL;
    }

    am_wstring_t *ws = am_wstring_create(ast->alloc, text, len);
    if (!ws) {
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        free(text);
        return AM_HANDLE_NULL;
    }

    // 缓存 hash 到对象头，便于后续快速判等
    ws->base.hash = hash;

    if (am_heap_set(ast->alloc, ast->alloc, ast->nodes, handle, am_make_value_of_ptr((am_object_t *)ws)) != 0) {
        // 注：am_wstring_t 的 content 是柔性数组，am_free 即可释放整个对象
        am_free(ast->alloc, ws);
        am_heap_free_handle(ast->alloc, ast->alloc, ast->nodes, handle);
        free(text);
        return AM_HANDLE_NULL;
    }

    // 登记到 strindex。注意 strindex_set 可能扩容并改变指针。
    am_strindex_t *new_si = am_strindex_set(ast->alloc, ast->strindex, text, am_make_value_of_handle(handle));
    if (new_si) {
        ast->strindex = new_si;
    }
    // 即使 strindex 登记失败，字符串对象已经创建并绑定到 heap，仍返回 handle。

    free(text);
    return handle;
}


// ===============================================================================
// 顶级节点操作
// ===============================================================================

typedef struct {
    am_handle_t found_handle;
} am_top_node_search_t;

static void am_ast_top_node_iter(am_handle_t handle, am_value_t value, void *user_data) {
    am_top_node_search_t *ctx = (am_top_node_search_t *)user_data;
    if (ctx->found_handle != AM_HANDLE_NULL) return;
    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->parent == AM_TOP_NODE_HANDLE) {
        ctx->found_handle = handle;
    }
}


// 功能描述：查找最顶级Application的handle。
am_handle_t am_ast_get_top_node_handle(am_ast_t *ast) {
    if (!ast || !ast->nodes) return AM_HANDLE_NULL;

    am_top_node_search_t ctx = { AM_HANDLE_NULL };
    am_heap_iter(ast->alloc, ast->alloc, ast->nodes, am_ast_top_node_iter, &ctx);
    return ctx.found_handle;
}


// 功能描述：查找顶级Lambda（全局作用域）节点的handle。
am_handle_t am_ast_get_top_lambda_node_handle(am_ast_t *ast) {
    am_handle_t top_app = am_ast_get_top_node_handle(ast);
    if (top_app == AM_HANDLE_NULL) return AM_HANDLE_NULL;

    am_value_t app_val = am_heap_get(ast->alloc, ast->alloc, ast->nodes, top_app);
    if (!am_value_is_ptr(app_val)) return AM_HANDLE_NULL;

    am_list_t *app = (am_list_t *)am_value_to_ptr(app_val);
    if (app->length == 0) return AM_HANDLE_NULL;

    am_value_t first = am_list_get(ast->alloc, app, 0);
    if (!am_value_is_handle(first)) return AM_HANDLE_NULL;

    return am_value_to_handle(first);
}


// 功能描述：获取位于全局作用域的node列表（函数体列表）。
am_value_t *am_ast_get_global_nodes(am_ast_t *ast) {
    if (!ast || !ast->nodes) return NULL;

    am_handle_t top_lambda = am_ast_get_top_lambda_node_handle(ast);
    if (top_lambda == AM_HANDLE_NULL) return NULL;

    am_value_t lambda_val = am_heap_get(ast->alloc, ast->alloc, ast->nodes, top_lambda);
    if (!am_value_is_ptr(lambda_val)) return NULL;

    am_list_t *lambda = (am_list_t *)am_value_to_ptr(lambda_val);
    size_t n_body = 0;
    return am_list_lambda_get_bodies(ast->alloc, lambda, &n_body);
}


// 功能描述：设置全局作用域（顶层lambda）的node列表（函数体列表）。
int32_t am_ast_set_global_nodes(am_ast_t *ast, am_value_t *bodies, size_t n_body) {
    if (!ast || !ast->nodes || !bodies) return -1;

    am_handle_t top_lambda = am_ast_get_top_lambda_node_handle(ast);
    if (top_lambda == AM_HANDLE_NULL) return -1;

    am_value_t lambda_val = am_heap_get(ast->alloc, ast->alloc, ast->nodes, top_lambda);
    if (!am_value_is_ptr(lambda_val)) return -1;

    am_list_t *lambda = (am_list_t *)am_value_to_ptr(lambda_val);

    am_list_t *new_lambda = am_list_lambda_set_bodies(ast->alloc, lambda, bodies, &n_body);
    if (!new_lambda) return -1;

    // 如果lambda对象指针发生变化，更新heap中的绑定
    if (new_lambda != lambda) {
        if (am_heap_set(ast->alloc, ast->alloc, ast->nodes, top_lambda, am_make_value_of_ptr((am_object_t *)new_lambda)) != 0) {
            am_list_destroy(ast->alloc, new_lambda);
            return -1;
        }
    }

    return 0;
}


// ===============================================================================
// 作用域上溯查找
// ===============================================================================

// 功能描述：从某个节点开始，向上上溯查找某个varid归属的lambda节点把柄。
am_handle_t am_ast_find_var_lambda_handle(am_ast_t *ast, am_varid_t varid, am_handle_t from_node_handle) {
    if (!ast || !ast->nodes) return AM_HANDLE_NULL;

    am_handle_t current = from_node_handle;
    while (current != AM_TOP_NODE_HANDLE) {
        am_value_t node_val = am_heap_get(ast->alloc, ast->alloc, ast->nodes, current);
        if (!am_value_is_ptr(node_val)) return AM_HANDLE_NULL;

        am_object_t *obj = am_value_to_ptr(node_val);
        if (obj->type == AM_OBJECT_TYPE_LIST) {
            am_list_t *lst = (am_list_t *)obj;
            if (lst->type == AM_LIST_TYPE_LAMBDA) {
                // 获取该lambda对应的scope
                am_value_t scope_handle_val = am_map_get(ast->alloc, ast->scopes, am_make_value_of_handle(current));
                if (am_value_is_handle(scope_handle_val)) {
                    am_handle_t scope_handle = am_value_to_handle(scope_handle_val);
                    am_value_t scope_val = am_heap_get(ast->alloc, ast->alloc, ast->nodes, scope_handle);
                    if (am_value_is_ptr(scope_val)) {
                        am_scope_t *scope = (am_scope_t *)am_value_to_ptr(scope_val);
                        if (am_scope_has_var(ast->alloc, scope, varid) >= 0) {
                            return current;
                        }
                    }
                }
            }
            current = lst->parent;
        }
        else {
            // 非list节点无法继续上溯
            return AM_HANDLE_NULL;
        }
    }

    return AM_HANDLE_NULL;
}


// ===============================================================================
// AST 节点转字符串
// ===============================================================================

// 动态宽字符串缓冲区
typedef struct {
    am_allocator_t *alloc;
    wchar_t        *buf;
    size_t          len;
    size_t          cap;
} am_ast_strbuf_t;


// 初始化字符串缓冲区。成功返回 0，失败返回 -1。
static int32_t am_ast_strbuf_init(am_allocator_t *alloc, am_ast_strbuf_t *sb, size_t initial_cap) {
    if (!alloc || !sb || initial_cap == 0) return -1;
    sb->alloc = alloc;
    sb->buf = (wchar_t *)am_malloc(alloc, initial_cap * sizeof(wchar_t));
    if (!sb->buf) return -1;
    sb->buf[0] = L'\0';
    sb->len = 0;
    sb->cap = initial_cap;
    return 0;
}


// 确保缓冲区容量至少为 needed（含结尾 L'\0'）。成功返回 0，失败返回 -1。
static int32_t am_ast_strbuf_ensure(am_ast_strbuf_t *sb, size_t needed) {
    if (!sb || !sb->buf) return -1;
    if (needed <= sb->cap) return 0;

    size_t new_cap = sb->cap;
    while (new_cap < needed) {
        new_cap *= 2;
    }

    wchar_t *new_buf = (wchar_t *)am_malloc(sb->alloc, new_cap * sizeof(wchar_t));
    if (!new_buf) return -1;

    memcpy(new_buf, sb->buf, (sb->len + 1) * sizeof(wchar_t));
    am_free(sb->alloc, sb->buf);
    sb->buf = new_buf;
    sb->cap = new_cap;
    return 0;
}


// 追加一个宽字符。成功返回 0，失败返回 -1。
static int32_t am_ast_strbuf_append_char(am_ast_strbuf_t *sb, wchar_t c) {
    if (!sb) return -1;
    if (am_ast_strbuf_ensure(sb, sb->len + 2) != 0) return -1;
    sb->buf[sb->len++] = c;
    sb->buf[sb->len] = L'\0';
    return 0;
}


// 追加一个宽字符串。成功返回 0，失败返回 -1。
static int32_t am_ast_strbuf_append_string(am_ast_strbuf_t *sb, const wchar_t *s) {
    if (!sb || !s) return -1;
    size_t slen = wcslen(s);
    if (am_ast_strbuf_ensure(sb, sb->len + slen + 1) != 0) return -1;
    memcpy(&sb->buf[sb->len], s, slen * sizeof(wchar_t));
    sb->len += slen;
    sb->buf[sb->len] = L'\0';
    return 0;
}


// 前向声明
static int32_t am_ast_append_value_to_strbuf(am_ast_strbuf_t *sb, am_ast_t *ast, am_value_t value);


// 将 lambda 节点追加到缓冲区。成功返回 0，失败返回 -1。
static int32_t am_ast_append_lambda_to_strbuf(am_ast_strbuf_t *sb, am_ast_t *ast, am_list_t *lambda) {
    if (!sb || !ast || !lambda) return -1;

    if (am_ast_strbuf_append_string(sb, L"(lambda (") != 0) return -1;

    size_t n_param = 0;
    if (lambda->length >= 2) {
        am_value_t n_param_val = am_list_get(ast->alloc, lambda, 1);
        if (am_value_is_uint(n_param_val)) {
            n_param = (size_t)am_value_to_uint(n_param_val);
        }
    }

    // 形参
    for (size_t i = 0; i < n_param; i++) {
        if (i > 0) {
            if (am_ast_strbuf_append_char(sb, L' ') != 0) return -1;
        }
        am_value_t param = am_list_get(ast->alloc, lambda, 2 + i);
        if (am_ast_append_value_to_strbuf(sb, ast, param) != 0) return -1;
    }
    if (am_ast_strbuf_append_char(sb, L')') != 0) return -1;

    // 函数体
    size_t n_body = am_list_lambda_get_body_number(ast->alloc, lambda);
    for (size_t i = 0; i < n_body; i++) {
        if (am_ast_strbuf_append_char(sb, L' ') != 0) return -1;
        am_value_t body = am_list_get(ast->alloc, lambda, 2 + n_param + i);
        if (am_ast_append_value_to_strbuf(sb, ast, body) != 0) return -1;
    }

    if (am_ast_strbuf_append_char(sb, L')') != 0) return -1;
    return 0;
}


// 将 application / quote / quasiquote / unquote 列表追加到缓冲区。成功返回 0，失败返回 -1。
static int32_t am_ast_append_list_to_strbuf(am_ast_strbuf_t *sb, am_ast_t *ast, am_list_t *lst) {
    if (!sb || !ast || !lst) return -1;

    const wchar_t *prefix = L"(";
    if (lst->type == AM_LIST_TYPE_QUOTE)       prefix = L"'(";
    else if (lst->type == AM_LIST_TYPE_QUASIQUOTE) prefix = L"`(";
    else if (lst->type == AM_LIST_TYPE_UNQUOTE)    prefix = L",(";

    if (am_ast_strbuf_append_string(sb, prefix) != 0) return -1;

    for (size_t i = 0; i < lst->length; i++) {
        if (i > 0) {
            if (am_ast_strbuf_append_char(sb, L' ') != 0) return -1;
        }
        am_value_t child = am_list_get(ast->alloc, lst, i);
        if (am_ast_append_value_to_strbuf(sb, ast, child) != 0) return -1;
    }

    if (am_ast_strbuf_append_char(sb, L')') != 0) return -1;
    return 0;
}


// 将任意 AST 值追加到缓冲区。成功返回 0，失败返回 -1。
static int32_t am_ast_append_value_to_strbuf(am_ast_strbuf_t *sb, am_ast_t *ast, am_value_t value) {
    if (!sb || !ast) return -1;

    if (am_value_is_handle(value)) {
        // 子节点以 handle 立即数形式引用，需到 heap 中查找
        am_handle_t h = am_value_to_handle(value);
        if (h == AM_HANDLE_NULL) {
            return am_ast_strbuf_append_string(sb, L"#<null-handle>");
        }
        am_value_t node_val = am_ast_get_node(ast, h);
        return am_ast_append_value_to_strbuf(sb, ast, node_val);
    }
    else if (am_value_is_ptr(value)) {
        am_object_t *obj = am_value_to_ptr(value);
        if (obj->type == AM_OBJECT_TYPE_LIST) {
            am_list_t *lst = (am_list_t *)obj;
            if (lst->type == AM_LIST_TYPE_LAMBDA) {
                return am_ast_append_lambda_to_strbuf(sb, ast, lst);
            }
            return am_ast_append_list_to_strbuf(sb, ast, lst);
        }
        else if (obj->type == AM_OBJECT_TYPE_WSTRING) {
            am_wstring_t *ws = (am_wstring_t *)obj;
            for (size_t i = 0; i < ws->length; i++) {
                am_value_t cv = ws->content[i];
                if (!am_value_is_wchar(cv)) continue;
                if (am_ast_strbuf_append_char(sb, (wchar_t)am_value_to_wchar(cv)) != 0) return -1;
            }
            return 0;
        }
        return am_ast_strbuf_append_string(sb, L"#<object>");
    }
    else if (am_value_is_varid(value)) {
        am_varid_t varid = am_value_to_varid(value);
        wchar_t *text = am_vocab_get(ast->alloc, ast->var_vocab, &varid);
        if (!text) return am_ast_strbuf_append_string(sb, L"#<var>");
        return am_ast_strbuf_append_string(sb, text);
    }
    else if (am_value_is_symbol(value)) {
        am_symbol_t sym = am_value_to_symbol(value);
        wchar_t *text = am_vocab_get(ast->alloc, ast->symbol_vocab, &sym);
        if (!text) return am_ast_strbuf_append_string(sb, L"#<sym>");
        // symbol 在词汇表中可能以单引号开头（如被 quote 的标识符），输出时去掉前导单引号
        // while (*text == L'\'') text++;
        return am_ast_strbuf_append_string(sb, text);
    }
    else if (am_value_is_uint(value)) {
        wchar_t tmp[64];
        swprintf(tmp, 64, L"%llu", (unsigned long long)am_value_to_uint(value));
        return am_ast_strbuf_append_string(sb, tmp);
    }
    else if (am_value_is_int(value)) {
        wchar_t tmp[64];
        swprintf(tmp, 64, L"%lld", (long long)am_value_to_int(value));
        return am_ast_strbuf_append_string(sb, tmp);
    }
    else if (am_value_is_float(value)) {
        wchar_t tmp[128];
        swprintf(tmp, 128, L"%g", (double)am_value_to_float(value));
        return am_ast_strbuf_append_string(sb, tmp);
    }
    else if (am_value_is_boolean(value)) {
        return am_ast_strbuf_append_string(sb, am_value_to_boolean(value) ? L"#t" : L"#f");
    }
    else if (am_value_is_null(value)) {
        return am_ast_strbuf_append_string(sb, L"#null");
    }
    else if (am_value_is_undefined(value)) {
        return am_ast_strbuf_append_string(sb, L"#undefined");
    }

    return am_ast_strbuf_append_string(sb, L"#<value>");
}


// 功能描述：将AST中的某个节点转成Scheme代码字符串（对应TS的AST.NodeToString）。
// 实现说明：返回使用 alloc 分配器分配的以 L'\0' 结尾的宽字符串，失败返回 NULL。
//         若 length 不为 NULL，则将字符串的逻辑长度（字符数）写入 *length。
wchar_t *am_ast_node_to_string(am_allocator_t *alloc, am_ast_t *ast, am_handle_t node_handle, size_t *length) {
    if (!alloc || !ast) return NULL;

    am_value_t value = am_ast_get_node(ast, node_handle);

    am_ast_strbuf_t sb;
    if (am_ast_strbuf_init(alloc, &sb, 256) != 0) return NULL;

    if (am_ast_append_value_to_strbuf(&sb, ast, value) != 0) {
        am_free(alloc, sb.buf);
        return NULL;
    }

    if (length) *length = sb.len;
    return sb.buf;
}




// 功能描述：从某个节点开始，向上上溯查找最近的lambda节点的把柄。
am_handle_t am_ast_find_nearest_lambda_handle(am_ast_t *ast, am_handle_t from_node_handle) {
    if (!ast || !ast->nodes) return AM_HANDLE_NULL;

    am_handle_t current = from_node_handle;
    while (current != AM_TOP_NODE_HANDLE) {
        am_value_t node_val = am_heap_get(ast->alloc, ast->alloc, ast->nodes, current);
        if (!am_value_is_ptr(node_val)) return AM_HANDLE_NULL;

        am_object_t *obj = am_value_to_ptr(node_val);
        if (obj->type == AM_OBJECT_TYPE_LIST) {
            am_list_t *lst = (am_list_t *)obj;
            if (lst->type == AM_LIST_TYPE_LAMBDA) {
                return current;
            }
            current = lst->parent;
        }
        else {
            return AM_HANDLE_NULL;
        }
    }

    return AM_HANDLE_NULL;
}


// ===============================================================================
// 变量换名
// ===============================================================================

// 功能描述：生成模块（AST）内唯一的变量名。
am_varid_t am_ast_make_unique_variable(am_ast_t *ast, am_varid_t varid, am_handle_t lambda_handle) {
    if (!ast || !ast->var_vocab || !ast->var_type) return SIZE_MAX;

    wchar_t *var_str = am_vocab_get(ast->alloc, ast->var_vocab, &varid);
    if (!var_str) return SIZE_MAX;

    // 生成新变量名：module_id.lambda_handle.var_string
    // 估算所需空间：module_id + 分隔点1 + handle最大20位 + 分隔点1 + var_str + 结尾\0
    size_t module_id_len = wcslen(ast->module_id);
    size_t var_len = wcslen(var_str);
    size_t buf_size = module_id_len + 1 + 20 + 1 + var_len + 1;

    wchar_t *new_name = (wchar_t *)am_malloc(ast->alloc, buf_size * sizeof(wchar_t));
    if (!new_name) return SIZE_MAX;

    int n = swprintf(new_name, buf_size, L"%ls.%zu.%ls", ast->module_id, lambda_handle, var_str);
    if (n <= 0 || (size_t)n >= buf_size) {
        am_free(ast->alloc, new_name);
        return SIZE_MAX;
    }

    size_t old_len = ast->var_vocab->length;
    size_t new_varid;
    ast->var_vocab = am_vocab_insert(ast->alloc, ast->var_vocab, new_name, &new_varid);
    am_free(ast->alloc, new_name);

    if (!ast->var_vocab || new_varid == SIZE_MAX) return SIZE_MAX;
    // 新变量加入时，同步在 var_type 中追加默认类型
    if (new_varid == old_len) {
        am_list_t *vt = am_list_push(ast->alloc, ast->var_type,
                                      am_make_value_of_uint(AM_VAR_TYPE_NEW));
        if (!vt) return SIZE_MAX;
        ast->var_type = vt;
    }
    return (am_varid_t)new_varid;
}


// 功能描述：为 import 别名生成模块级唯一变量名。
am_varid_t am_ast_make_unique_module_alias(am_ast_t *ast, am_varid_t alias_varid) {
    if (!ast || !ast->var_vocab || !ast->var_type) return SIZE_MAX;

    wchar_t *alias_str = am_vocab_get(ast->alloc, ast->var_vocab, &alias_varid);
    if (!alias_str) return SIZE_MAX;

    // 生成新变量名：module_id.alias
    size_t module_id_len = wcslen(ast->module_id);
    size_t alias_len = wcslen(alias_str);
    size_t buf_size = module_id_len + 1 + alias_len + 1;

    wchar_t *new_name = (wchar_t *)am_malloc(ast->alloc, buf_size * sizeof(wchar_t));
    if (!new_name) return SIZE_MAX;

    int n = swprintf(new_name, buf_size, L"%ls.%ls", ast->module_id, alias_str);
    if (n <= 0 || (size_t)n >= buf_size) {
        am_free(ast->alloc, new_name);
        return SIZE_MAX;
    }

    size_t old_len = ast->var_vocab->length;
    size_t new_varid;
    ast->var_vocab = am_vocab_insert(ast->alloc, ast->var_vocab, new_name, &new_varid);
    am_free(ast->alloc, new_name);

    if (!ast->var_vocab || new_varid == SIZE_MAX) return SIZE_MAX;

    // 设置 var_type 为 AM_VAR_TYPE_IMPORT_ALIAS
    if (new_varid == old_len) {
        am_list_t *vt = am_list_push(ast->alloc, ast->var_type,
                                      am_make_value_of_uint(AM_VAR_TYPE_IMPORT_ALIAS));
        if (!vt) return SIZE_MAX;
        ast->var_type = vt;
    }
    else {
        if (am_list_set(ast->alloc, ast->var_type, new_varid,
                        am_make_value_of_uint(AM_VAR_TYPE_IMPORT_ALIAS)) != 0) {
            return SIZE_MAX;
        }
    }

    return (am_varid_t)new_varid;
}


// 功能描述：为 import 外部引用生成模块级唯一变量名。
am_varid_t am_ast_make_unique_import_ref(am_ast_t *ast, am_varid_t import_ref_varid) {
    if (!ast || !ast->var_vocab || !ast->var_type) return SIZE_MAX;

    wchar_t *ref_str = am_vocab_get(ast->alloc, ast->var_vocab, &import_ref_varid);
    if (!ref_str) return SIZE_MAX;

    // 生成新变量名：module_id.import_ref
    size_t module_id_len = wcslen(ast->module_id);
    size_t ref_len = wcslen(ref_str);
    size_t buf_size = module_id_len + 1 + ref_len + 1;

    wchar_t *new_name = (wchar_t *)am_malloc(ast->alloc, buf_size * sizeof(wchar_t));
    if (!new_name) return SIZE_MAX;

    int n = swprintf(new_name, buf_size, L"%ls.%ls", ast->module_id, ref_str);
    if (n <= 0 || (size_t)n >= buf_size) {
        am_free(ast->alloc, new_name);
        return SIZE_MAX;
    }

    size_t old_len = ast->var_vocab->length;
    size_t new_varid;
    ast->var_vocab = am_vocab_insert(ast->alloc, ast->var_vocab, new_name, &new_varid);
    am_free(ast->alloc, new_name);

    if (!ast->var_vocab || new_varid == SIZE_MAX) return SIZE_MAX;

    // 设置 var_type 为 AM_VAR_TYPE_IMPORT_REF
    if (new_varid == old_len) {
        am_list_t *vt = am_list_push(ast->alloc, ast->var_type,
                                      am_make_value_of_uint(AM_VAR_TYPE_IMPORT_REF));
        if (!vt) return SIZE_MAX;
        ast->var_type = vt;
    }
    else {
        if (am_list_set(ast->alloc, ast->var_type, new_varid,
                        am_make_value_of_uint(AM_VAR_TYPE_IMPORT_REF)) != 0) {
            return SIZE_MAX;
        }
    }

    return (am_varid_t)new_varid;
}


// ===============================================================================
// AST 基本操作辅助接口
// ===============================================================================

// 功能描述：向 tailcall_handles 中添加一个尾调用节点把柄。
int32_t am_ast_add_tailcall(am_ast_t *ast, am_handle_t handle) {
    if (!ast || !ast->tailcall_handles) return -1;
    am_list_t *lst = am_list_push(ast->alloc, ast->tailcall_handles, am_make_value_of_handle(handle));
    if (!lst) return -1;
    ast->tailcall_handles = lst;
    return 0;
}


// 功能描述：向 var_top 中添加一个顶级变量 varid。
int32_t am_ast_add_var_top(am_ast_t *ast, am_varid_t varid) {
    if (!ast || !ast->var_top) return -1;
    am_list_t *lst = am_list_push(ast->alloc, ast->var_top, am_make_value_of_varid(varid));
    if (!lst) return -1;
    ast->var_top = lst;
    return 0;
}


// 功能描述：设置依赖模块记录。
int32_t am_ast_set_dependency(am_ast_t *ast, am_varid_t alias_varid, am_handle_t path_handle) {
    if (!ast || !ast->dependencies) return -1;
    am_map_t *map = am_map_set(ast->alloc, ast->dependencies,
                                am_make_value_of_varid(alias_varid),
                                am_make_value_of_handle(path_handle));
    if (!map) return -1;
    ast->dependencies = map;
    return 0;
}


// 功能描述：设置本地库记录。
int32_t am_ast_set_native(am_ast_t *ast, am_varid_t native_varid, am_handle_t handle) {
    if (!ast || !ast->natives) return -1;
    am_map_t *map = am_map_set(ast->alloc, ast->natives,
                                am_make_value_of_varid(native_varid),
                                am_make_value_of_handle(handle));
    if (!map) return -1;
    ast->natives = map;
    return 0;
}


// 功能描述：为lambda节点设置对应的词法作用域把柄。
int32_t am_ast_set_scope(am_ast_t *ast, am_handle_t lambda_handle, am_handle_t scope_handle) {
    if (!ast || !ast->scopes) return -1;
    am_map_t *map = am_map_set(ast->alloc, ast->scopes,
                                am_make_value_of_handle(lambda_handle),
                                am_make_value_of_handle(scope_handle));
    if (!map) return -1;
    ast->scopes = map;
    return 0;
}


// 功能描述：获取lambda节点对应的词法作用域把柄。
am_handle_t am_ast_get_scope(am_ast_t *ast, am_handle_t lambda_handle) {
    if (!ast || !ast->scopes) return AM_HANDLE_NULL;
    am_value_t v = am_map_get(ast->alloc, ast->scopes, am_make_value_of_handle(lambda_handle));
    if (am_value_is_handle(v)) {
        return am_value_to_handle(v);
    }
    return AM_HANDLE_NULL;
}
/* ===== end:   src/am_ast.c ===== */

/* ===== begin: src/am_parser.c ===== */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>



#define PARSER_LOG(x) ((void)x); // printf(x);


// ===============================================================================
// 解析器状态
// ===============================================================================

#define AM_PARSER_STATE_NONE         (0)
#define AM_PARSER_STATE_PARAMETER    (1)
#define AM_PARSER_STATE_QUOTE        (2)
#define AM_PARSER_STATE_UNQUOTE      (3)
#define AM_PARSER_STATE_QUASIQUOTE   (4)

#define AM_PARSER_SPECIAL_APP_NONE   (0)
#define AM_PARSER_SPECIAL_APP_IMPORT (1)
#define AM_PARSER_SPECIAL_APP_NATIVE (2)


typedef struct parser_ctx_t {
    am_ast_t *ast;
    am_token_t *tokens;
    size_t token_count;

    am_value_t *node_stack;
    size_t node_stack_capacity;
    size_t node_stack_length;

    int *state_stack;
    size_t state_stack_capacity;
    size_t state_stack_length;

    am_handle_t *lambda_stack;
    size_t lambda_stack_capacity;
    size_t lambda_stack_length;

    int *special_app_stack;
    size_t special_app_stack_capacity;
    size_t special_app_stack_length;

    int32_t is_keep_free; // 为 1 时，将未定义变量视为全局自由变量

    int error;
    wchar_t error_msg[256];
} parser_ctx_t;


// ===============================================================================
// 前向声明
// ===============================================================================

static size_t am_parser__parse_term(parser_ctx_t *ctx, size_t index);
static size_t parse_slist(parser_ctx_t *ctx, size_t index);
static size_t parse_slist_seq(parser_ctx_t *ctx, size_t index);
static size_t parse_lambda(parser_ctx_t *ctx, size_t index);
static size_t parse_arg_list(parser_ctx_t *ctx, size_t index);
static size_t parse_arg_list_seq(parser_ctx_t *ctx, size_t index);
static size_t parse_arg_identifier(parser_ctx_t *ctx, size_t index);
static size_t parse_body(parser_ctx_t *ctx, size_t index);
static size_t parse_body_tail(parser_ctx_t *ctx, size_t index);
static size_t parse_body_term(parser_ctx_t *ctx, size_t index);
static size_t parse_quote(parser_ctx_t *ctx, size_t index);
static size_t parse_unquote(parser_ctx_t *ctx, size_t index);
static size_t parse_quasiquote(parser_ctx_t *ctx, size_t index);
static size_t parse_quote_term(parser_ctx_t *ctx, size_t index);
static size_t parse_unquote_term(parser_ctx_t *ctx, size_t index);
static size_t parse_quasiquote_term(parser_ctx_t *ctx, size_t index);
static size_t parse_identifier(parser_ctx_t *ctx, size_t index);


// ===============================================================================
// 内部辅助函数
// ===============================================================================

static void parser_set_error(parser_ctx_t *ctx, const wchar_t *msg) {
    if (!ctx || ctx->error) return;
    ctx->error = 1;
    wcsncpy(ctx->error_msg, msg, 255);
    ctx->error_msg[255] = L'\0';
}


static am_token_t *token_at(parser_ctx_t *ctx, size_t index) {
    if (!ctx || index >= ctx->token_count) return NULL;
    return &ctx->tokens[index];
}


static int state_stack_top(parser_ctx_t *ctx) {
    if (!ctx || ctx->state_stack_length == 0) return AM_PARSER_STATE_NONE;
    return ctx->state_stack[ctx->state_stack_length - 1];
}


static void state_stack_push(parser_ctx_t *ctx, int state) {
    if (!ctx) return;
    if (ctx->state_stack_length >= ctx->state_stack_capacity) {
        size_t new_cap = ctx->state_stack_capacity ? ctx->state_stack_capacity * 2 : 16;
        int *new_stack = (int *)realloc(ctx->state_stack, new_cap * sizeof(int));
        if (!new_stack) {
            parser_set_error(ctx, L"state stack out of memory");
            return;
        }
        ctx->state_stack = new_stack;
        ctx->state_stack_capacity = new_cap;
    }
    ctx->state_stack[ctx->state_stack_length++] = state;
}


static void state_stack_pop(parser_ctx_t *ctx) {
    if (!ctx || ctx->state_stack_length == 0) return;
    ctx->state_stack_length--;
}


static int special_app_stack_top(parser_ctx_t *ctx) {
    if (!ctx || ctx->special_app_stack_length == 0) return AM_PARSER_SPECIAL_APP_NONE;
    return ctx->special_app_stack[ctx->special_app_stack_length - 1];
}


static void special_app_stack_push(parser_ctx_t *ctx, int state) {
    if (!ctx) return;
    if (ctx->special_app_stack_length >= ctx->special_app_stack_capacity) {
        size_t new_cap = ctx->special_app_stack_capacity ? ctx->special_app_stack_capacity * 2 : 16;
        int *new_stack = (int *)realloc(ctx->special_app_stack, new_cap * sizeof(int));
        if (!new_stack) {
            parser_set_error(ctx, L"special app stack out of memory");
            return;
        }
        ctx->special_app_stack = new_stack;
        ctx->special_app_stack_capacity = new_cap;
    }
    ctx->special_app_stack[ctx->special_app_stack_length++] = state;
}


static void special_app_stack_pop(parser_ctx_t *ctx) {
    if (!ctx || ctx->special_app_stack_length == 0) return;
    ctx->special_app_stack_length--;
}


static void lambda_stack_push(parser_ctx_t *ctx, am_handle_t lambda_handle) {
    if (!ctx) return;
    if (ctx->lambda_stack_length >= ctx->lambda_stack_capacity) {
        size_t new_cap = ctx->lambda_stack_capacity ? ctx->lambda_stack_capacity * 2 : 16;
        am_handle_t *new_stack = (am_handle_t *)realloc(ctx->lambda_stack, new_cap * sizeof(am_handle_t));
        if (!new_stack) {
            parser_set_error(ctx, L"lambda stack out of memory");
            return;
        }
        ctx->lambda_stack = new_stack;
        ctx->lambda_stack_capacity = new_cap;
    }
    ctx->lambda_stack[ctx->lambda_stack_length++] = lambda_handle;
}


static void lambda_stack_pop(parser_ctx_t *ctx) {
    if (!ctx || ctx->lambda_stack_length == 0) return;
    ctx->lambda_stack_length--;
}


static am_value_t node_stack_top(parser_ctx_t *ctx) {
    if (!ctx || ctx->node_stack_length == 0) return AM_VALUE_UNDEFINED;
    return ctx->node_stack[ctx->node_stack_length - 1];
}


static void node_stack_push(parser_ctx_t *ctx, am_value_t value) {
    if (!ctx) return;
    if (ctx->node_stack_length >= ctx->node_stack_capacity) {
        size_t new_cap = ctx->node_stack_capacity ? ctx->node_stack_capacity * 2 : 16;
        am_value_t *new_stack = (am_value_t *)realloc(ctx->node_stack, new_cap * sizeof(am_value_t));
        if (!new_stack) {
            parser_set_error(ctx, L"node stack out of memory");
            return;
        }
        ctx->node_stack = new_stack;
        ctx->node_stack_capacity = new_cap;
    }
    ctx->node_stack[ctx->node_stack_length++] = value;
}


static am_value_t node_stack_pop(parser_ctx_t *ctx) {
    if (!ctx || ctx->node_stack_length == 0) return AM_VALUE_UNDEFINED;
    return ctx->node_stack[--ctx->node_stack_length];
}


static int detect_special_app(parser_ctx_t *ctx) {
    if (!ctx || ctx->node_stack_length == 0) return AM_PARSER_SPECIAL_APP_NONE;

    am_value_t top = node_stack_top(ctx);
    if (!am_value_is_handle(top)) return AM_PARSER_SPECIAL_APP_NONE;

    am_handle_t list_handle = am_value_to_handle(top);
    if (list_handle == AM_TOP_NODE_HANDLE) return AM_PARSER_SPECIAL_APP_NONE;

    am_value_t list_val = am_ast_get_node(ctx->ast, list_handle);
    if (!am_value_is_ptr(list_val)) return AM_PARSER_SPECIAL_APP_NONE;

    am_list_t *lst = (am_list_t *)am_value_to_ptr(list_val);
    if (lst->length != 1) return AM_PARSER_SPECIAL_APP_NONE;

    am_value_t first = am_list_get(ctx->ast->alloc, lst, 0);
    if (!am_value_is_symbol(first)) return AM_PARSER_SPECIAL_APP_NONE;

    am_symbol_t sym = am_value_to_symbol(first);
    if (sym == am_value_to_symbol(AM_VALUE_KW_import)) return AM_PARSER_SPECIAL_APP_IMPORT;
    if (sym == am_value_to_symbol(AM_VALUE_KW_native)) return AM_PARSER_SPECIAL_APP_NATIVE;

    return AM_PARSER_SPECIAL_APP_NONE;
}


static int is_identifier_token(am_token_t *tok) {
    if (!tok) return 0;
    switch (tok->type) {
        case AM_TOKEN_TYPE_NUMBER:
        case AM_TOKEN_TYPE_STRING:
        case AM_TOKEN_TYPE_SYMBOL:
        case AM_TOKEN_TYPE_IDENTIFIER:
        case AM_TOKEN_TYPE_KEYWORD:
        case AM_TOKEN_TYPE_BOOLEAN:
        case AM_TOKEN_TYPE_NULL:
        case AM_TOKEN_TYPE_UNDEFINED:
            return 1;
        default:
            return 0;
    }
}


static int is_term_start_token(am_token_t *tok) {
    if (!tok) return 0;
    return tok->type == AM_TOKEN_TYPE_LB ||
           tok->type == AM_TOKEN_TYPE_QUOTE ||
           tok->type == AM_TOKEN_TYPE_UNQUOTE ||
           tok->type == AM_TOKEN_TYPE_QUASIQUOTE ||
           is_identifier_token(tok);
}


static wchar_t *token_text_dup(am_token_t *tok, wchar_t *code) {
    if (!tok || !code) return NULL;
    wchar_t *s = (wchar_t *)malloc((tok->length + 1) * sizeof(wchar_t));
    if (!s) return NULL;
    wcsncpy(s, code + tok->index, tok->length);
    s[tok->length] = L'\0';
    return s;
}


static am_value_t parse_number_token(am_token_t *tok, wchar_t *code) {
    if (!tok || !code) return AM_VALUE_UNDEFINED;

    wchar_t *text = token_text_dup(tok, code);
    if (!text) return AM_VALUE_UNDEFINED;

    am_value_t result = AM_VALUE_UNDEFINED;
    int has_dot = 0;
    int has_e = 0;
    size_t len = wcslen(text);

    for (size_t i = 0; i < len; i++) {
        if (text[i] == L'.') has_dot = 1;
        else if (text[i] == L'e' || text[i] == L'E') has_e = 1;
    }

    if (has_dot || has_e) {
        double d = wcstod(text, NULL);
        result = am_make_value_of_float((am_float_t)d);
    }
    else {
        long long ll = wcstoll(text, NULL, 10);
        if (ll >= 0) {
            result = am_make_value_of_uint((am_uint_t)ll);
        }
        else {
            result = am_make_value_of_int((am_int_t)ll);
        }
    }

    free(text);
    return result;
}


static int32_t append_child_to_top(parser_ctx_t *ctx) {
    if (!ctx || ctx->error) return -1;
    if (ctx->node_stack_length < 2) {
        parser_set_error(ctx, L"node stack underflow");
        return -1;
    }

    am_value_t child = node_stack_pop(ctx);
    if (ctx->error) return -1;

    am_value_t parent_val = node_stack_top(ctx);
    if (!am_value_is_handle(parent_val)) {
        parser_set_error(ctx, L"parent is not a handle");
        return -1;
    }

    am_handle_t parent_handle = am_value_to_handle(parent_val);
    am_value_t node_val = am_ast_get_node(ctx->ast, parent_handle);
    if (!am_value_is_ptr(node_val)) {
        parser_set_error(ctx, L"parent node not found");
        return -1;
    }

    am_object_t *obj = am_value_to_ptr(node_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) {
        parser_set_error(ctx, L"parent is not a list");
        return -1;
    }

    am_list_t *lst = (am_list_t *)obj;
    am_list_t *new_lst = am_list_push(ctx->ast->alloc, lst, child);
    if (!new_lst) {
        parser_set_error(ctx, L"failed to append child");
        return -1;
    }

    if (new_lst != lst) {
        if (am_heap_set(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, parent_handle,
                        am_make_value_of_ptr((am_object_t *)new_lst)) != 0) {
            am_list_destroy(ctx->ast->alloc, new_lst);
            parser_set_error(ctx, L"failed to update parent node");
            return -1;
        }
    }

    return 0;
}


static int32_t add_parameter_to_top_lambda(parser_ctx_t *ctx, am_value_t param) {
    if (!ctx || ctx->error) return -1;

    am_value_t top = node_stack_top(ctx);
    if (!am_value_is_handle(top)) {
        parser_set_error(ctx, L"lambda stack corrupted");
        return -1;
    }

    am_handle_t lambda_handle = am_value_to_handle(top);
    am_value_t node_val = am_ast_get_node(ctx->ast, lambda_handle);
    if (!am_value_is_ptr(node_val)) {
        parser_set_error(ctx, L"lambda node not found");
        return -1;
    }

    am_list_t *lst = (am_list_t *)am_value_to_ptr(node_val);
    am_list_t *new_lst = am_list_lambda_add_parameter(ctx->ast->alloc, lst, param);
    if (!new_lst) {
        parser_set_error(ctx, L"failed to add parameter");
        return -1;
    }

    if (new_lst != lst) {
        if (am_heap_set(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, lambda_handle,
                        am_make_value_of_ptr((am_object_t *)new_lst)) != 0) {
            am_list_destroy(ctx->ast->alloc, new_lst);
            parser_set_error(ctx, L"failed to update lambda node");
            return -1;
        }
    }

    return 0;
}


static int32_t add_body_to_top_lambda(parser_ctx_t *ctx, am_value_t body) {
    if (!ctx || ctx->error) return -1;

    am_value_t top = node_stack_top(ctx);
    if (!am_value_is_handle(top)) {
        parser_set_error(ctx, L"lambda stack corrupted");
        return -1;
    }

    am_handle_t lambda_handle = am_value_to_handle(top);
    am_value_t node_val = am_ast_get_node(ctx->ast, lambda_handle);
    if (!am_value_is_ptr(node_val)) {
        parser_set_error(ctx, L"lambda node not found");
        return -1;
    }

    am_list_t *lst = (am_list_t *)am_value_to_ptr(node_val);
    am_list_t *new_lst = am_list_lambda_add_body(ctx->ast->alloc, lst, body);
    if (!new_lst) {
        parser_set_error(ctx, L"failed to add body");
        return -1;
    }

    if (new_lst != lst) {
        if (am_heap_set(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, lambda_handle,
                        am_make_value_of_ptr((am_object_t *)new_lst)) != 0) {
            am_list_destroy(ctx->ast->alloc, new_lst);
            parser_set_error(ctx, L"failed to update lambda node");
            return -1;
        }
    }

    return 0;
}


static am_varid_t ensure_varid(parser_ctx_t *ctx, wchar_t *word) {
    if (!ctx || !ctx->ast || !ctx->ast->var_vocab || !ctx->ast->var_type) return SIZE_MAX;
    size_t idx = am_vocab_find(ctx->ast->alloc, ctx->ast->var_vocab, word);
    if (idx != SIZE_MAX) return (am_varid_t)idx;
    size_t old_len = ctx->ast->var_vocab->length;
    ctx->ast->var_vocab = am_vocab_insert(ctx->ast->alloc, ctx->ast->var_vocab, word, &idx);
    if (!ctx->ast->var_vocab || idx == SIZE_MAX) return SIZE_MAX;
    // 新变量加入时，同步在 var_type 中追加默认类型
    if (idx == old_len) {
        am_list_t *vt = am_list_push(ctx->ast->alloc, ctx->ast->var_type,
                                      am_make_value_of_uint(AM_VAR_TYPE_OLD));
        if (!vt) return SIZE_MAX;
        ctx->ast->var_type = vt;
    }
    return (am_varid_t)idx;
}


static am_symbol_t ensure_symbol(parser_ctx_t *ctx, wchar_t *word) {
    if (!ctx || !ctx->ast || !ctx->ast->symbol_vocab) return SIZE_MAX;
    size_t idx = am_vocab_find(ctx->ast->alloc, ctx->ast->symbol_vocab, word);
    if (idx != SIZE_MAX) return (am_symbol_t)idx;
    ctx->ast->symbol_vocab = am_vocab_insert(ctx->ast->alloc, ctx->ast->symbol_vocab, word, &idx);
    if (!ctx->ast->symbol_vocab || idx == SIZE_MAX) return SIZE_MAX;
    return (am_symbol_t)idx;
}


static int is_global_builtin_variable(const wchar_t *text) {
    if (!text) return 0;
    for (size_t i = 0; i < AM_GLOBAL_BUILTIN_VAR_NUM; i++) {
        if (AM_GLOBAL_BUILTIN_VAR[i] && wcscmp(text, AM_GLOBAL_BUILTIN_VAR[i]) == 0) {
            return 1;
        }
    }
    return 0;
}


// ===============================================================================
// 递归下降分析
// ===============================================================================

static size_t am_parser__parse_term(parser_ctx_t *ctx, size_t index) { PARSER_LOG("Term\n");
    am_token_t *tok = token_at(ctx, index);
    am_token_t *next = token_at(ctx, index + 1);
    int state = state_stack_top(ctx);

    if (ctx->error) return index;
    if (!tok) {
        parser_set_error(ctx, L"unexpected end of input in term");
        return index;
    }

    // (lambda ...) 且不在 quote/quasiquote 状态
    if (state != AM_PARSER_STATE_QUOTE && state != AM_PARSER_STATE_QUASIQUOTE &&
        tok->type == AM_TOKEN_TYPE_LB && next && next->type == AM_TOKEN_TYPE_KEYWORD &&
        next->id == am_value_to_symbol(AM_VALUE_KW_lambda)) {
        return parse_lambda(ctx, index);
    }
    // (quote ...)
    else if (tok->type == AM_TOKEN_TYPE_LB && next && next->type == AM_TOKEN_TYPE_KEYWORD &&
             next->id == am_value_to_symbol(AM_VALUE_KW_quote)) {
        size_t next_index = parse_quote(ctx, index + 1);
        am_token_t *after = token_at(ctx, next_index);
        if (!after || after->type != AM_TOKEN_TYPE_RB) {
            parser_set_error(ctx, L"quote 右侧括号未闭合");
            return index;
        }
        return next_index + 1;
    }
    // (unquote ...)
    else if (tok->type == AM_TOKEN_TYPE_LB && next && next->type == AM_TOKEN_TYPE_KEYWORD &&
             next->id == am_value_to_symbol(AM_VALUE_KW_unquote)) {
        size_t next_index = parse_unquote(ctx, index + 1);
        am_token_t *after = token_at(ctx, next_index);
        if (!after || after->type != AM_TOKEN_TYPE_RB) {
            parser_set_error(ctx, L"unquote 右侧括号未闭合");
            return index;
        }
        return next_index + 1;
    }
    // (quasiquote ...)
    else if (tok->type == AM_TOKEN_TYPE_LB && next && next->type == AM_TOKEN_TYPE_KEYWORD &&
             next->id == am_value_to_symbol(AM_VALUE_KW_quasiquote)) {
        size_t next_index = parse_quasiquote(ctx, index + 1);
        am_token_t *after = token_at(ctx, next_index);
        if (!after || after->type != AM_TOKEN_TYPE_RB) {
            parser_set_error(ctx, L"quasiquote 右侧括号未闭合");
            return index;
        }
        return next_index + 1;
    }
    // '...
    else if (tok->type == AM_TOKEN_TYPE_QUOTE) {
        return parse_quote(ctx, index);
    }
    // ,...
    else if (tok->type == AM_TOKEN_TYPE_UNQUOTE) {
        return parse_unquote(ctx, index);
    }
    // `...
    else if (tok->type == AM_TOKEN_TYPE_QUASIQUOTE) {
        return parse_quasiquote(ctx, index);
    }
    // ( ... )
    else if (tok->type == AM_TOKEN_TYPE_LB) {
        return parse_slist(ctx, index);
    }
    // identifier
    else if (is_identifier_token(tok)) {
        return parse_identifier(ctx, index);
    }
    else {
        parser_set_error(ctx, L"unexpected token in term");
        return index;
    }
}


static size_t parse_slist(parser_ctx_t *ctx, size_t index) { PARSER_LOG("SList\n");
    am_token_t *tok = token_at(ctx, index);
    if (!tok || tok->type != AM_TOKEN_TYPE_LB) {
        parser_set_error(ctx, L"expected '(' for slist");
        return index;
    }

    int state = state_stack_top(ctx);
    int32_t list_type = AM_LIST_TYPE_APPLICATION;
    if (state == AM_PARSER_STATE_QUOTE) list_type = AM_LIST_TYPE_QUOTE;
    else if (state == AM_PARSER_STATE_QUASIQUOTE) list_type = AM_LIST_TYPE_QUASIQUOTE;
    else if (state == AM_PARSER_STATE_UNQUOTE) list_type = AM_LIST_TYPE_UNQUOTE;

    am_handle_t parent_handle = AM_HANDLE_NULL;
    am_value_t top = node_stack_top(ctx);
    if (!ctx->error && am_value_is_handle(top) && top != am_make_value_of_handle(AM_TOP_NODE_HANDLE)) {
        parent_handle = am_value_to_handle(top);
    }

    am_handle_t list_handle = am_ast_make_slist_node(ctx->ast, parent_handle, list_type);
    if (list_handle == AM_HANDLE_NULL) {
        parser_set_error(ctx, L"failed to create slist node");
        return index;
    }

    node_stack_push(ctx, am_make_value_of_handle(list_handle));
    am_ast_set_node_token_index(ctx->ast, list_handle, index);

    size_t next_index = parse_slist_seq(ctx, index + 1);
    if (ctx->error) return index;

    am_token_t *after = token_at(ctx, next_index);
    if (!after || after->type != AM_TOKEN_TYPE_RB) {
        parser_set_error(ctx, L"slist 右侧括号未闭合");
        return index;
    }
    return next_index + 1;
}


static size_t parse_slist_seq(parser_ctx_t *ctx, size_t index) { PARSER_LOG("SListSeq\n");
    am_token_t *tok = token_at(ctx, index);
    if (ctx->error) return index;
    if (!tok) {
        parser_set_error(ctx, L"unexpected end of input in slist seq");
        return index;
    }

    if (is_term_start_token(tok)) {
        size_t next_index = am_parser__parse_term(ctx, index);
        if (ctx->error) return index;

        if (append_child_to_top(ctx) < 0) return index;

        // 如果刚解析完 (import ...) 或 (native ...) 的第一个元素，
        // 则对第二个元素推送特殊 application 状态，解析完成后立即弹出
        int special_app = detect_special_app(ctx);
        if (special_app != AM_PARSER_SPECIAL_APP_NONE) {
            special_app_stack_push(ctx, special_app);
            next_index = am_parser__parse_term(ctx, next_index);
            if (ctx->error) return index;
            if (append_child_to_top(ctx) < 0) {
                special_app_stack_pop(ctx);
                return index;
            }
            special_app_stack_pop(ctx);
        }

        return parse_slist_seq(ctx, next_index);
    }
    else {
        return index;
    }
}


static size_t parse_lambda(parser_ctx_t *ctx, size_t index) { PARSER_LOG("Lambda\n");
    am_token_t *tok = token_at(ctx, index);
    if (!tok || tok->type != AM_TOKEN_TYPE_LB) {
        parser_set_error(ctx, L"expected '(' for lambda");
        return index;
    }

    am_handle_t parent_handle = AM_HANDLE_NULL;
    am_value_t top = node_stack_top(ctx);
    if (!ctx->error && am_value_is_handle(top) && top != am_make_value_of_handle(AM_TOP_NODE_HANDLE)) {
        parent_handle = am_value_to_handle(top);
    }

    am_handle_t lambda_handle = am_ast_make_lambda_node(ctx->ast, parent_handle);
    if (lambda_handle == AM_HANDLE_NULL) {
        parser_set_error(ctx, L"failed to create lambda node");
        return index;
    }

    node_stack_push(ctx, am_make_value_of_handle(lambda_handle));
    am_ast_set_node_token_index(ctx->ast, lambda_handle, index);
    lambda_stack_push(ctx, lambda_handle);

    size_t result = index; // 出错时默认返回原索引

    size_t next_index = parse_arg_list(ctx, index + 2); // 跳过 '(' 和 'lambda'
    if (ctx->error) goto lambda_done;

    next_index = parse_body(ctx, next_index);
    if (ctx->error) goto lambda_done;

    am_token_t *after = token_at(ctx, next_index);
    if (!after || after->type != AM_TOKEN_TYPE_RB) {
        parser_set_error(ctx, L"lambda 右侧括号未闭合");
        goto lambda_done;
    }
    result = next_index + 1;

lambda_done:
    lambda_stack_pop(ctx);
    return result;
}


static size_t parse_arg_list(parser_ctx_t *ctx, size_t index) { PARSER_LOG("ArgList\n");
    am_token_t *tok = token_at(ctx, index);
    if (!tok || tok->type != AM_TOKEN_TYPE_LB) {
        parser_set_error(ctx, L"expected '(' for arglist");
        return index;
    }

    state_stack_push(ctx, AM_PARSER_STATE_PARAMETER);
    size_t next_index = parse_arg_list_seq(ctx, index + 1);
    state_stack_pop(ctx);
    if (ctx->error) return index;

    am_token_t *after = token_at(ctx, next_index);
    if (!after || after->type != AM_TOKEN_TYPE_RB) {
        parser_set_error(ctx, L"arglist 右侧括号未闭合");
        return index;
    }
    return next_index + 1;
}


static size_t parse_arg_list_seq(parser_ctx_t *ctx, size_t index) { PARSER_LOG("ArgListSeq\n");
    am_token_t *tok = token_at(ctx, index);
    if (ctx->error) return index;
    if (!tok) {
        parser_set_error(ctx, L"unexpected end of input in arglist seq");
        return index;
    }

    if (is_identifier_token(tok)) {
        size_t next_index = parse_arg_identifier(ctx, index);
        if (ctx->error) return index;

        am_value_t param = node_stack_pop(ctx);
        if (ctx->error) return index;
        if (!am_value_is_varid(param)) {
            // 允许 '...' 出现在 lambda 参数列表中，用于 syntax-rules 宏模板。
            if (!(am_value_is_symbol(param) && am_value_to_symbol(param) == am_value_to_symbol(AM_VALUE_KW_dot3))) {
                parser_set_error(ctx, L"lambda parameter must be variable");
                return index;
            }
        }

        if (add_parameter_to_top_lambda(ctx, param) < 0) return index;

        return parse_arg_list_seq(ctx, next_index);
    }
    else {
        return index;
    }
}


static size_t parse_arg_identifier(parser_ctx_t *ctx, size_t index) { PARSER_LOG("ArgId\n");
    return parse_identifier(ctx, index);
}


static size_t parse_body(parser_ctx_t *ctx, size_t index) { PARSER_LOG("Body\n");
    size_t next_index = parse_body_term(ctx, index);
    if (ctx->error) return index;

    am_value_t body = node_stack_pop(ctx);
    if (ctx->error) return index;

    if (add_body_to_top_lambda(ctx, body) < 0) return index;

    return parse_body_tail(ctx, next_index);
}


static size_t parse_body_tail(parser_ctx_t *ctx, size_t index) { PARSER_LOG("BodyTail\n");
    am_token_t *tok = token_at(ctx, index);
    if (ctx->error) return index;
    if (!tok) {
        parser_set_error(ctx, L"unexpected end of input in body tail");
        return index;
    }

    if (is_term_start_token(tok)) {
        size_t next_index = parse_body_term(ctx, index);
        if (ctx->error) return index;

        am_value_t body = node_stack_pop(ctx);
        if (ctx->error) return index;

        if (add_body_to_top_lambda(ctx, body) < 0) return index;

        return parse_body_tail(ctx, next_index);
    }
    else {
        return index;
    }
}


static size_t parse_body_term(parser_ctx_t *ctx, size_t index) { PARSER_LOG("BodyTerm\n");
    return am_parser__parse_term(ctx, index);
}


static size_t parse_quote(parser_ctx_t *ctx, size_t index) {
    am_token_t *tok = token_at(ctx, index);
    if (!tok) {
        parser_set_error(ctx, L"unexpected end of input in quote");
        return index;
    }

    size_t start = index;
    if (tok->type == AM_TOKEN_TYPE_QUOTE) {
        start = index + 1;
    }
    else if (tok->type == AM_TOKEN_TYPE_KEYWORD && tok->id == am_value_to_symbol(AM_VALUE_KW_quote)) {
        start = index + 1;
    }
    else if (tok->type == AM_TOKEN_TYPE_LB) {
        start = index + 2; // 跳过 ( quote
    }
    else {
        parser_set_error(ctx, L"expected quote");
        return index;
    }

    state_stack_push(ctx, AM_PARSER_STATE_QUOTE);
    size_t next_index = parse_quote_term(ctx, start);
    state_stack_pop(ctx);
    return next_index;
}


static size_t parse_unquote(parser_ctx_t *ctx, size_t index) {
    am_token_t *tok = token_at(ctx, index);
    if (!tok) {
        parser_set_error(ctx, L"unexpected end of input in unquote");
        return index;
    }

    size_t start = index;
    if (tok->type == AM_TOKEN_TYPE_UNQUOTE) {
        start = index + 1;
    }
    else if (tok->type == AM_TOKEN_TYPE_KEYWORD && tok->id == am_value_to_symbol(AM_VALUE_KW_unquote)) {
        start = index + 1;
    }
    else if (tok->type == AM_TOKEN_TYPE_LB) {
        start = index + 2; // 跳过 ( unquote
    }
    else {
        parser_set_error(ctx, L"expected unquote");
        return index;
    }

    state_stack_push(ctx, AM_PARSER_STATE_UNQUOTE);
    size_t next_index = parse_unquote_term(ctx, start);
    state_stack_pop(ctx);
    return next_index;
}


static size_t parse_quasiquote(parser_ctx_t *ctx, size_t index) {
    am_token_t *tok = token_at(ctx, index);
    if (!tok) {
        parser_set_error(ctx, L"unexpected end of input in quasiquote");
        return index;
    }

    size_t start = index;
    if (tok->type == AM_TOKEN_TYPE_QUASIQUOTE) {
        start = index + 1;
    }
    else if (tok->type == AM_TOKEN_TYPE_KEYWORD && tok->id == am_value_to_symbol(AM_VALUE_KW_quasiquote)) {
        start = index + 1;
    }
    else if (tok->type == AM_TOKEN_TYPE_LB) {
        start = index + 2; // 跳过 ( quasiquote
    }
    else {
        parser_set_error(ctx, L"expected quasiquote");
        return index;
    }

    state_stack_push(ctx, AM_PARSER_STATE_QUASIQUOTE);
    size_t next_index = parse_quasiquote_term(ctx, start);
    state_stack_pop(ctx);
    return next_index;
}


static size_t parse_quote_term(parser_ctx_t *ctx, size_t index) {
    return am_parser__parse_term(ctx, index);
}


static size_t parse_unquote_term(parser_ctx_t *ctx, size_t index) {
    return am_parser__parse_term(ctx, index);
}


static size_t parse_quasiquote_term(parser_ctx_t *ctx, size_t index) {
    return am_parser__parse_term(ctx, index);
}


// ===============================================================================
// Identifier 解析
// ===============================================================================

static size_t parse_identifier(parser_ctx_t *ctx, size_t index) { PARSER_LOG("Identifier\n");
    am_token_t *tok = token_at(ctx, index);
    if (!tok) {
        parser_set_error(ctx, L"unexpected end of input in identifier");
        return index;
    }

    int state = state_stack_top(ctx);
    am_value_t value = AM_VALUE_UNDEFINED;

    switch (tok->type) {
        case AM_TOKEN_TYPE_NUMBER: {
            value = parse_number_token(tok, ctx->ast->code);
            if (value == AM_VALUE_UNDEFINED) {
                parser_set_error(ctx, L"invalid number token");
                return index;
            }
            break;
        }

        case AM_TOKEN_TYPE_STRING: {
            am_handle_t str_handle = am_ast_make_wstring_node(ctx->ast, tok);
            if (str_handle == AM_HANDLE_NULL) {
                parser_set_error(ctx, L"failed to create string node");
                return index;
            }
            am_ast_set_node_token_index(ctx->ast, str_handle, index);
            value = am_make_value_of_handle(str_handle);
            break;
        }

        case AM_TOKEN_TYPE_SYMBOL: {
            if (state == AM_PARSER_STATE_UNQUOTE) {
                // 解除引用：去掉前导单引号，作为变量
                wchar_t *text = token_text_dup(tok, ctx->ast->code);
                if (!text) {
                    parser_set_error(ctx, L"out of memory");
                    return index;
                }
                wchar_t *var_text = text;
                while (*var_text == L'\'') var_text++;
                am_varid_t varid = ensure_varid(ctx, var_text);
                free(text);
                if (varid == SIZE_MAX) {
                    parser_set_error(ctx, L"failed to create varid");
                    return index;
                }
                value = am_make_value_of_varid(varid);
            }
            else {
                value = am_make_value_of_symbol((am_symbol_t)tok->id);
            }
            break;
        }

        case AM_TOKEN_TYPE_IDENTIFIER:
        case AM_TOKEN_TYPE_KEYWORD: {
            if (state == AM_PARSER_STATE_QUOTE || state == AM_PARSER_STATE_QUASIQUOTE) {
                // 被 quote 的标识符/关键字变成 symbol，前加单引号
                wchar_t *text = token_text_dup(tok, ctx->ast->code);
                if (!text) {
                    parser_set_error(ctx, L"out of memory");
                    return index;
                }
                size_t len = wcslen(text);
                wchar_t *sym_text = (wchar_t *)malloc((len + 2) * sizeof(wchar_t));
                if (!sym_text) {
                    free(text);
                    parser_set_error(ctx, L"out of memory");
                    return index;
                }
                sym_text[0] = L'\'';
                wcscpy(sym_text + 1, text);
                free(text);

                am_symbol_t sym_id = ensure_symbol(ctx, sym_text);
                free(sym_text);
                if (sym_id == SIZE_MAX) {
                    parser_set_error(ctx, L"failed to create symbol");
                    return index;
                }
                value = am_make_value_of_symbol(sym_id);
            }
            // else if (state == AM_PARSER_STATE_UNQUOTE) {
            //     // 作为变量处理
            //     wchar_t *text = token_text_dup(tok, ctx->ast->code);
            //     if (!text) {
            //         parser_set_error(ctx, L"out of memory");
            //         return index;
            //     }
            //     am_varid_t varid = ensure_varid(ctx, text);
            //     free(text);
            //     if (varid == SIZE_MAX) {
            //         parser_set_error(ctx, L"failed to create varid");
            //         return index;
            //     }
            //     value = am_make_value_of_varid(varid);
            // }
            else { // 含state == AM_PARSER_STATE_UNQUOTE
                // 普通状态：关键字作为 symbol，变量作为 varid
                if (tok->type == AM_TOKEN_TYPE_KEYWORD) {
                    value = am_make_value_of_symbol((am_symbol_t)tok->id);
                }
                else {
                    wchar_t *var_text = token_text_dup(tok, ctx->ast->code);
                    if (!var_text) {
                        parser_set_error(ctx, L"out of memory");
                        return index;
                    }

                    // 按需将实际变量注册到 var_vocab；quote/quasiquote 中的标识符
                    // 在前面已被转换为 symbol，不会走到这里，因此不会污染 var_vocab。
                    am_varid_t varid = ensure_varid(ctx, var_text);
                    if (varid == SIZE_MAX) {
                        free(var_text);
                        parser_set_error(ctx, L"failed to create varid");
                        return index;
                    }
                    tok->id = (size_t)varid;

                    // (import ...) / (native ...) 的第二个元素保持原形，不参与 Alpha-renaming
                    int special_app = special_app_stack_top(ctx);
                    if (special_app != AM_PARSER_SPECIAL_APP_NONE) {
                        if (am_list_set(ctx->ast->alloc, ctx->ast->var_type, (size_t)varid,
                                        am_make_value_of_uint(AM_VAR_TYPE_OLD)) != 0) {
                            free(var_text);
                            parser_set_error(ctx, L"failed to set var_type");
                            return index;
                        }
                        value = am_make_value_of_varid(varid);
                        free(var_text);
                    }
                    // EXT_REF 格式（前缀.后缀）保持原形，不参与 Alpha-renaming
                    else if (am_ast_check_ext_ref(ctx->ast, varid) == 0) {
                        if (am_list_set(ctx->ast->alloc, ctx->ast->var_type, (size_t)varid,
                                        am_make_value_of_uint(AM_VAR_TYPE_EXT_REF)) != 0) {
                            free(var_text);
                            parser_set_error(ctx, L"failed to set var_type");
                            return index;
                        }
                        value = am_make_value_of_varid(varid);
                        free(var_text);
                    }
                    // 全局内置变量不做 Alpha-renaming
                    else if (is_global_builtin_variable(var_text)) {
                        value = am_make_value_of_varid(varid);
                        free(var_text);
                    }
                    else {
                        // 普通变量：解析阶段保持原 varid，var_type 保持 OLD，
                        // Alpha-renaming 在后续独立的 ARN 阶段完成
                        value = am_make_value_of_varid(varid);
                        free(var_text);
                    }
                }
            }
            break;
        }

        case AM_TOKEN_TYPE_BOOLEAN: {
            wchar_t *text = token_text_dup(tok, ctx->ast->code);
            if (!text) {
                parser_set_error(ctx, L"out of memory");
                return index;
            }
            value = (wcscmp(text, L"#t") == 0) ? AM_VALUE_TRUE : AM_VALUE_FALSE;
            free(text);
            break;
        }

        case AM_TOKEN_TYPE_NULL: {
            value = AM_VALUE_NULL;
            break;
        }

        case AM_TOKEN_TYPE_UNDEFINED: {
            value = AM_VALUE_UNDEFINED;
            break;
        }

        default: {
            parser_set_error(ctx, L"illegal identifier token");
            return index;
        }
    }

    node_stack_push(ctx, value);
    return index + 1;
}


// ===============================================================================
// 预处理指令解析
// ===============================================================================

typedef struct {
    parser_ctx_t *ctx;
} preprocess_iter_data_t;

static void preprocess_iter_cb(am_handle_t handle, am_value_t value, void *user_data) {
    (void)handle;
    preprocess_iter_data_t *data = (preprocess_iter_data_t *)user_data;
    parser_ctx_t *ctx = data->ctx;
    am_ast_t *ast = ctx->ast;

    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->type != AM_LIST_TYPE_APPLICATION || lst->length == 0) return;

    am_value_t first = am_list_get(ast->alloc, lst, 0);
    if (!am_value_is_symbol(first)) return;

    am_symbol_t first_sym = am_value_to_symbol(first);

    // (import <Alias> <Path>)
    if (first_sym == am_value_to_symbol(AM_VALUE_KW_import) && lst->length == 3) {
        am_value_t alias_val = am_list_get(ast->alloc, lst, 1);
        am_value_t path_handle_val = am_list_get(ast->alloc, lst, 2);

        if (!am_value_is_varid(alias_val) || !am_value_is_handle(path_handle_val)) {
            parser_set_error(ctx, L"invalid import syntax");
            return;
        }

        am_varid_t alias_varid = am_value_to_varid(alias_val);
        am_handle_t path_handle = am_value_to_handle(path_handle_val);

        am_value_t path_node_val = am_ast_get_node(ast, path_handle);
        if (!am_value_is_ptr(path_node_val) ||
            ((am_object_t *)am_value_to_ptr(path_node_val))->type != AM_OBJECT_TYPE_WSTRING) {
            parser_set_error(ctx, L"import path must be string");
            return;
        }

        if (am_ast_set_dependency(ast, alias_varid, path_handle) < 0) {
            parser_set_error(ctx, L"failed to set dependency");
            return;
        }
    }
    // (native <NativeLibName>)
    else if (first_sym == am_value_to_symbol(AM_VALUE_KW_native) && lst->length == 2) {
        am_value_t name_val = am_list_get(ast->alloc, lst, 1);
        if (!am_value_is_varid(name_val)) {
            parser_set_error(ctx, L"invalid native syntax");
            return;
        }
        am_varid_t native_varid = am_value_to_varid(name_val);
        if (am_list_set(ast->alloc, ast->var_type, (size_t)native_varid,
                        am_make_value_of_uint(AM_VAR_TYPE_NATIVE_ID)) != 0) {
            parser_set_error(ctx, L"failed to set native var type");
            return;
        }
        if (am_ast_set_native(ast, native_varid, AM_HANDLE_NULL) < 0) {
            parser_set_error(ctx, L"failed to set native");
            return;
        }
    }
}


static void preprocess_analysis(parser_ctx_t *ctx) {
    if (!ctx || !ctx->ast) return;
    preprocess_iter_data_t data = { ctx };
    am_heap_iter(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, preprocess_iter_cb, &data);
}


// ===============================================================================
// Alpha-renaming（变量换名）
// ===============================================================================

// 第一阶段：递归下降解析完成后，所有变量保持原名（OLD）。
// 第二阶段 ARN 再分两趟扫描：
//   Pass 1: 扫描整棵 AST，构建词法作用域 scope 的树状嵌套关系，并在 scope 上挂载 old 变量；
//   Pass 2: 根据 scope 嵌套关系，执行 ARN，并在 ast->var_arn_mapping 中登记新旧 varid 映射。


typedef struct {
    parser_ctx_t *ctx;
    am_handle_t *handles;
    size_t count;
    size_t capacity;
} arn_lambda_collect_t;


typedef struct {
    am_varid_t varid;
    am_handle_t lambda_handle;
} arn_define_entry_t;


typedef struct {
    parser_ctx_t *ctx;
    arn_define_entry_t *entries;
    size_t count;
    size_t capacity;
} arn_define_collect_t;


typedef struct {
    parser_ctx_t *ctx;
} arn_rename_iter_t;


static void arn_free_lambda_collect(arn_lambda_collect_t *data) {
    if (!data) return;
    if (data->handles) {
        free(data->handles);
        data->handles = NULL;
    }
    data->count = 0;
    data->capacity = 0;
}


static void arn_free_define_collect(arn_define_collect_t *data) {
    if (!data) return;
    if (data->entries) {
        free(data->entries);
        data->entries = NULL;
    }
    data->count = 0;
    data->capacity = 0;
}


// Pass 1-1: 收集所有 lambda 节点把柄
static void arn_collect_lambda_cb(am_handle_t handle, am_value_t value, void *user_data) {
    arn_lambda_collect_t *data = (arn_lambda_collect_t *)user_data;
    parser_ctx_t *ctx = data->ctx;

    if (ctx->error) return;
    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->type != AM_LIST_TYPE_LAMBDA) return;

    if (data->count >= data->capacity) {
        size_t new_cap = data->capacity ? data->capacity * 2 : 16;
        am_handle_t *new_arr = (am_handle_t *)realloc(data->handles, new_cap * sizeof(am_handle_t));
        if (!new_arr) {
            parser_set_error(ctx, L"out of memory collecting lambda handles");
            return;
        }
        data->handles = new_arr;
        data->capacity = new_cap;
    }
    data->handles[data->count++] = handle;
}


// Pass 1-2: 为每个 lambda 节点创建对应的 scope
static int32_t arn_create_scopes(parser_ctx_t *ctx, arn_lambda_collect_t *lambdas) {
    am_ast_t *ast = ctx->ast;

    for (size_t i = 0; i < lambdas->count; i++) {
        am_handle_t lambda_handle = lambdas->handles[i];
        am_value_t node_val = am_ast_get_node(ast, lambda_handle);
        if (!am_value_is_ptr(node_val)) return -1;

        am_list_t *lst = (am_list_t *)am_value_to_ptr(node_val);
        if (lst->type != AM_LIST_TYPE_LAMBDA) return -1;

        am_handle_t parent_lambda = AM_HANDLE_NULL;
        am_handle_t parent_scope = AM_HANDLE_NULL;
        if (lst->parent != AM_TOP_NODE_HANDLE) {
            parent_lambda = am_ast_find_nearest_lambda_handle(ast, lst->parent);
            if (parent_lambda != AM_HANDLE_NULL) {
                parent_scope = am_ast_get_scope(ast, parent_lambda);
            }
        }

        am_scope_t *scope = am_scope_create(ast->alloc, parent_scope, parent_lambda, lambda_handle, 16);
        if (!scope) return -1;

        am_handle_t scope_handle = am_heap_alloc_handle(ast->alloc, ast->alloc, ast->nodes);
        if (scope_handle == AM_HANDLE_NULL) {
            am_scope_destroy(ast->alloc, scope);
            return -1;
        }

        if (am_heap_set(ast->alloc, ast->alloc, ast->nodes, scope_handle,
                        am_make_value_of_ptr((am_object_t *)scope)) != 0) {
            am_scope_destroy(ast->alloc, scope);
            return -1;
        }

        if (am_ast_set_scope(ast, lambda_handle, scope_handle) != 0) {
            return -1;
        }
    }

    return 0;
}


// Pass 1-3: 将 lambda 的 parameters 注册到对应 scope
static int32_t arn_add_lambda_params_to_scope(parser_ctx_t *ctx, am_handle_t lambda_handle) {
    am_ast_t *ast = ctx->ast;
    am_value_t node_val = am_ast_get_node(ast, lambda_handle);
    if (!am_value_is_ptr(node_val)) return -1;

    am_list_t *lst = (am_list_t *)am_value_to_ptr(node_val);
    if (lst->type != AM_LIST_TYPE_LAMBDA) return -1;

    am_handle_t scope_handle = am_ast_get_scope(ast, lambda_handle);
    if (scope_handle == AM_HANDLE_NULL) return -1;

    am_value_t scope_val = am_ast_get_node(ast, scope_handle);
    if (!am_value_is_ptr(scope_val)) return -1;

    am_scope_t *scope = (am_scope_t *)am_value_to_ptr(scope_val);

    size_t n_param = 0;
    if (lst->length >= 2) {
        am_value_t n_param_val = am_list_get(ast->alloc, lst, 1);
        if (am_value_is_uint(n_param_val)) n_param = (size_t)am_value_to_uint(n_param_val);
    }

    for (size_t i = 0; i < n_param; i++) {
        am_value_t param = am_list_get(ast->alloc, lst, 2 + i);
        if (!am_value_is_varid(param)) continue;

        am_scope_t *new_scope = am_scope_add_var(ast->alloc, scope, am_value_to_varid(param), AM_VALUE_NULL);
        if (!new_scope) return -1;

        if (new_scope != scope) {
            if (am_heap_set(ast->alloc, ast->alloc, ast->nodes, scope_handle,
                            am_make_value_of_ptr((am_object_t *)new_scope)) != 0) {
                return -1;
            }
            scope = new_scope;
        }
    }

    return 0;
}


// Pass 1-4: 收集所有 (define var ...) 定义的变量
static void arn_collect_defines_cb(am_handle_t handle, am_value_t value, void *user_data) {
    (void)handle;
    arn_define_collect_t *data = (arn_define_collect_t *)user_data;
    parser_ctx_t *ctx = data->ctx;
    am_ast_t *ast = ctx->ast;

    if (ctx->error) return;
    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->type != AM_LIST_TYPE_APPLICATION || lst->length < 2) return;

    am_value_t first = am_list_get(ast->alloc, lst, 0);
    if (!am_value_is_symbol(first)) return;
    if (am_value_to_symbol(first) != am_value_to_symbol(AM_VALUE_KW_define)) return;

    am_value_t second = am_list_get(ast->alloc, lst, 1);
    if (!am_value_is_varid(second)) return;

    am_handle_t parent_handle = (lst->parent != AM_TOP_NODE_HANDLE) ? lst->parent : AM_HANDLE_NULL;
    am_handle_t lambda_handle = AM_HANDLE_NULL;
    if (parent_handle != AM_HANDLE_NULL) {
        lambda_handle = am_ast_find_nearest_lambda_handle(ast, parent_handle);
    }
    if (lambda_handle == AM_HANDLE_NULL) {
        parser_set_error(ctx, L"define outside lambda scope");
        return;
    }

    if (data->count >= data->capacity) {
        size_t new_cap = data->capacity ? data->capacity * 2 : 16;
        arn_define_entry_t *new_arr = (arn_define_entry_t *)realloc(data->entries,
                                                                     new_cap * sizeof(arn_define_entry_t));
        if (!new_arr) {
            parser_set_error(ctx, L"out of memory collecting defines");
            return;
        }
        data->entries = new_arr;
        data->capacity = new_cap;
    }
    data->entries[data->count].varid = am_value_to_varid(second);
    data->entries[data->count].lambda_handle = lambda_handle;
    data->count++;
}


// Pass 1-5: 将 define 的变量注册到所在 lambda 的 scope
static int32_t arn_add_defines_to_scope(parser_ctx_t *ctx, arn_define_collect_t *defines) {
    am_ast_t *ast = ctx->ast;

    for (size_t i = 0; i < defines->count; i++) {
        am_handle_t scope_handle = am_ast_get_scope(ast, defines->entries[i].lambda_handle);
        if (scope_handle == AM_HANDLE_NULL) return -1;

        am_value_t scope_val = am_ast_get_node(ast, scope_handle);
        if (!am_value_is_ptr(scope_val)) return -1;

        am_scope_t *scope = (am_scope_t *)am_value_to_ptr(scope_val);

        am_scope_t *new_scope = am_scope_add_var(ast->alloc, scope,
                                                  defines->entries[i].varid, AM_VALUE_NULL);
        if (!new_scope) return -1;

        if (new_scope != scope) {
            if (am_heap_set(ast->alloc, ast->alloc, ast->nodes, scope_handle,
                            am_make_value_of_ptr((am_object_t *)new_scope)) != 0) {
                return -1;
            }
            scope = new_scope;
        }
    }

    return 0;
}


// Pass 2: 判断某个 varid 是否属于不参与 ARN 的特殊类型
static int arn_should_skip_var_type(am_ast_t *ast, am_varid_t varid) {
    am_value_t type_val = am_list_get(ast->alloc, ast->var_type, (size_t)varid);
    if (!am_value_is_uint(type_val)) return 1;

    am_uint_t t = am_value_to_uint(type_val);
    return (t == AM_VAR_TYPE_IMPORT_ALIAS ||
            t == AM_VAR_TYPE_NATIVE_ID ||
            t == AM_VAR_TYPE_IMPORT_REF ||
            t == AM_VAR_TYPE_NATIVE_REF ||
            t == AM_VAR_TYPE_EXT_REF);
}


// Pass 2: 将 list 中指定位置的变量替换为 ARN 后的新 varid
static void arn_rename_varid(parser_ctx_t *ctx, am_handle_t node_handle, am_list_t *lst,
                              size_t index, int is_parameter) {
    am_ast_t *ast = ctx->ast;
    am_value_t child = am_list_get(ast->alloc, lst, index);
    if (!am_value_is_varid(child)) return;

    am_varid_t old_varid = am_value_to_varid(child);

    if (arn_should_skip_var_type(ast, old_varid)) return;

    am_handle_t lambda_handle = AM_HANDLE_NULL;
    if (is_parameter) {
        lambda_handle = node_handle;
    } else {
        lambda_handle = am_ast_find_var_lambda_handle(ast, old_varid, node_handle);
    }

    if (lambda_handle == AM_HANDLE_NULL) {
        // 未定义变量：全局内置变量保持原 varid；其余保持原 varid（宽松语义，兼容自由变量）
        wchar_t *var_str = am_vocab_get(ast->alloc, ast->var_vocab, &old_varid);
        if (var_str && is_global_builtin_variable(var_str)) {
            return;
        }
        // 若启用 is_keep_free，将未定义变量标记为全局自由变量
        if (ctx->is_keep_free) {
            if (am_list_set(ast->alloc, ast->var_type, (size_t)old_varid,
                            am_make_value_of_uint(AM_VAR_TYPE_GLOBAL_FREE)) != 0) {
                parser_set_error(ctx, L"failed to set global free var type");
            }
        }
        return;
    }

    am_varid_t new_varid = am_ast_make_unique_variable(ast, old_varid, lambda_handle);
    if (new_varid == SIZE_MAX) {
        parser_set_error(ctx, L"failed to create unique variable");
        return;
    }

    if (am_list_set(ast->alloc, lst, index, am_make_value_of_varid(new_varid)) != 0) {
        parser_set_error(ctx, L"failed to set renamed varid");
        return;
    }

    am_map_t *map = am_map_set(ast->alloc, ast->var_arn_mapping,
                                am_make_value_of_varid(new_varid),
                                am_make_value_of_varid(old_varid));
    if (!map) {
        parser_set_error(ctx, L"failed to set var_arn_mapping");
        return;
    }
    ast->var_arn_mapping = map;
}


// Pass 2: 处理 lambda 节点（parameters 与 body 中出现的变量）
static void arn_rename_lambda(parser_ctx_t *ctx, am_handle_t handle, am_list_t *lst) {
    size_t n_param = 0;
    if (lst->length >= 2) {
        am_value_t n_param_val = am_list_get(ctx->ast->alloc, lst, 1);
        if (am_value_is_uint(n_param_val)) n_param = (size_t)am_value_to_uint(n_param_val);
    }

    for (size_t i = 0; i < n_param; i++) {
        arn_rename_varid(ctx, handle, lst, 2 + i, 1);
        if (ctx->error) return;
    }

    for (size_t i = 2 + n_param; i < lst->length; i++) {
        am_value_t child = am_list_get(ctx->ast->alloc, lst, i);
        if (am_value_is_varid(child)) {
            arn_rename_varid(ctx, handle, lst, i, 0);
            if (ctx->error) return;
        }
    }
}


// Pass 2: 处理 application / unquote / quasiquote 节点
static void arn_rename_application(parser_ctx_t *ctx, am_handle_t handle, am_list_t *lst) {
    am_ast_t *ast = ctx->ast;
    if (lst->length == 0) return;

    am_value_t first = am_list_get(ast->alloc, lst, 0);
    if (am_value_is_symbol(first) &&
        (am_value_to_symbol(first) == am_value_to_symbol(AM_VALUE_KW_import) ||
         am_value_to_symbol(first) == am_value_to_symbol(AM_VALUE_KW_native))) {
        return;
    }

    for (size_t i = 0; i < lst->length; i++) {
        am_value_t child = am_list_get(ast->alloc, lst, i);
        if (am_value_is_varid(child)) {
            arn_rename_varid(ctx, handle, lst, i, 0);
            if (ctx->error) return;
        }
    }
}


// Pass 2: 节点遍历回调
static void arn_rename_iter_cb(am_handle_t handle, am_value_t value, void *user_data) {
    arn_rename_iter_t *data = (arn_rename_iter_t *)user_data;
    parser_ctx_t *ctx = data->ctx;

    if (ctx->error) return;
    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        arn_rename_lambda(ctx, handle, lst);
    }
    else if (lst->type == AM_LIST_TYPE_APPLICATION ||
             lst->type == AM_LIST_TYPE_QUASIQUOTE ||
             lst->type == AM_LIST_TYPE_UNQUOTE) {
        arn_rename_application(ctx, handle, lst);
    }
}


static void alpha_rename_analysis(parser_ctx_t *ctx) {
    if (!ctx || !ctx->ast) return;
    am_ast_t *ast = ctx->ast;

    arn_lambda_collect_t lambdas = { ctx, NULL, 0, 0 };
    arn_define_collect_t defines = { ctx, NULL, 0, 0 };

    // Pass 1-1: 收集所有 lambda handle
    am_heap_iter(ast->alloc, ast->alloc, ast->nodes, arn_collect_lambda_cb, &lambdas);
    if (ctx->error) goto arn_cleanup;

    // Pass 1-2: 为每个 lambda 创建 scope
    if (arn_create_scopes(ctx, &lambdas) != 0) {
        if (!ctx->error) parser_set_error(ctx, L"failed to create scopes");
        goto arn_cleanup;
    }

    // Pass 1-3: 注册 lambda parameters
    for (size_t i = 0; i < lambdas.count; i++) {
        if (arn_add_lambda_params_to_scope(ctx, lambdas.handles[i]) != 0) {
            if (!ctx->error) parser_set_error(ctx, L"failed to add lambda params to scope");
            goto arn_cleanup;
        }
    }

    // Pass 1-4: 收集 define
    am_heap_iter(ast->alloc, ast->alloc, ast->nodes, arn_collect_defines_cb, &defines);
    if (ctx->error) goto arn_cleanup;

    // Pass 1-5: 注册 define 变量
    if (arn_add_defines_to_scope(ctx, &defines) != 0) {
        if (!ctx->error) parser_set_error(ctx, L"failed to add defines to scope");
        goto arn_cleanup;
    }

    // Pass 2: 执行变量换名
    arn_rename_iter_t rename_data = { ctx };
    am_heap_iter(ast->alloc, ast->alloc, ast->nodes, arn_rename_iter_cb, &rename_data);

arn_cleanup:
    arn_free_lambda_collect(&lambdas);
    arn_free_define_collect(&defines);
}


// ===============================================================================
// 清理 ARN 阶段遗留的 scope 对象
// ===============================================================================

typedef struct {
    parser_ctx_t *ctx;
    am_handle_t *scope_handles;
    size_t count;
    size_t capacity;
} arn_scope_cleanup_data_t;


static void arn_collect_scope_handles_cb(am_handle_t handle, am_value_t value, void *user_data) {
    arn_scope_cleanup_data_t *data = (arn_scope_cleanup_data_t *)user_data;
    parser_ctx_t *ctx = data->ctx;

    if (ctx->error) return;
    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_SCOPE) return;

    if (data->count >= data->capacity) {
        size_t new_cap = data->capacity ? data->capacity * 2 : 16;
        am_handle_t *new_arr = (am_handle_t *)realloc(data->scope_handles, new_cap * sizeof(am_handle_t));
        if (!new_arr) {
            parser_set_error(ctx, L"out of memory collecting scope handles");
            return;
        }
        data->scope_handles = new_arr;
        data->capacity = new_cap;
    }
    data->scope_handles[data->count++] = handle;
}


static void cleanup_scope_objects(parser_ctx_t *ctx) {
    if (!ctx || !ctx->ast) return;

    arn_scope_cleanup_data_t data = { ctx, NULL, 0, 0 };
    am_heap_iter(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, arn_collect_scope_handles_cb, &data);

    if (!ctx->error) {
        for (size_t i = 0; i < data.count; i++) {
            am_heap_free_handle(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, data.scope_handles[i]);
        }
    }
    free(data.scope_handles);
}


// ===============================================================================
// 引用模块别名（alias）和外部引用（ext_ref）更名
// ===============================================================================

typedef struct {
    parser_ctx_t *ctx;
    am_varid_t *old_aliases;
    size_t old_aliases_capacity;
    size_t old_aliases_length;
} alias_rename_iter_data_t;

static int alias_rename_record_old_alias(alias_rename_iter_data_t *data, am_varid_t varid) {
    if (!data) return 0;
    if (data->old_aliases_length >= data->old_aliases_capacity) {
        size_t new_cap = data->old_aliases_capacity ? data->old_aliases_capacity * 2 : 8;
        am_varid_t *new_arr = (am_varid_t *)realloc(data->old_aliases, new_cap * sizeof(am_varid_t));
        if (!new_arr) return 0;
        data->old_aliases = new_arr;
        data->old_aliases_capacity = new_cap;
    }
    data->old_aliases[data->old_aliases_length++] = varid;
    return 1;
}

static void alias_rename_free_old_aliases(alias_rename_iter_data_t *data) {
    if (!data) return;
    free(data->old_aliases);
    data->old_aliases = NULL;
    data->old_aliases_capacity = 0;
    data->old_aliases_length = 0;
}

static void alias_rename_iter_cb(am_handle_t handle, am_value_t value, void *user_data) {
    (void)handle;
    alias_rename_iter_data_t *data = (alias_rename_iter_data_t *)user_data;
    parser_ctx_t *ctx = data->ctx;
    am_ast_t *ast = ctx->ast;

    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;

    am_list_t *lst = (am_list_t *)obj;

    // (import <Alias> <Path>)
    if (lst->type == AM_LIST_TYPE_APPLICATION && lst->length == 3) {
        am_value_t first = am_list_get(ast->alloc, lst, 0);
        if (am_value_is_symbol(first) &&
            am_value_to_symbol(first) == am_value_to_symbol(AM_VALUE_KW_import)) {
            am_value_t alias_val = am_list_get(ast->alloc, lst, 1);
            am_value_t path_handle_val = am_list_get(ast->alloc, lst, 2);

            if (!am_value_is_varid(alias_val) || !am_value_is_handle(path_handle_val)) {
                parser_set_error(ctx, L"invalid import syntax in alias rename");
                return;
            }

            am_varid_t old_alias_varid = am_value_to_varid(alias_val);
            am_varid_t new_alias_varid = am_ast_make_unique_module_alias(ast, old_alias_varid);
            if (new_alias_varid == SIZE_MAX) {
                parser_set_error(ctx, L"failed to create unique module alias");
                return;
            }

            if (am_list_set(ast->alloc, lst, 1, am_make_value_of_varid(new_alias_varid)) != 0) {
                parser_set_error(ctx, L"failed to set renamed import alias");
                return;
            }

            if (am_ast_set_dependency(ast, new_alias_varid, am_value_to_handle(path_handle_val)) < 0) {
                parser_set_error(ctx, L"failed to set dependency for renamed alias");
                return;
            }

            if (!alias_rename_record_old_alias(data, old_alias_varid)) {
                parser_set_error(ctx, L"out of memory recording old alias");
                return;
            }

            return; // import 节点本身不处理 children 中的 ext_ref
        }
    }

    // 其他节点：遍历 children，处理 AM_VAR_TYPE_EXT_REF 的 varid（IMPORT_REF 更名，NATIVE_REF 设置类型）
    for (size_t i = 0; i < lst->length; i++) {
        am_value_t child = am_list_get(ast->alloc, lst, i);
        if (!am_value_is_varid(child)) continue;

        am_varid_t varid = am_value_to_varid(child);
        am_value_t type_val = am_list_get(ast->alloc, ast->var_type, (size_t)varid);
        if (!am_value_is_uint(type_val)) continue;

        if (am_value_to_uint(type_val) != AM_VAR_TYPE_EXT_REF) continue;

        if (am_ast_check_native_ref(ast, varid) == 0) {
            if (am_list_set(ast->alloc, ast->var_type, (size_t)varid,
                            am_make_value_of_uint(AM_VAR_TYPE_NATIVE_REF)) != 0) {
                parser_set_error(ctx, L"failed to set native ref var type");
                return;
            }
            continue;
        }

        if (am_ast_check_import_ref(ast, varid) != 0) continue;

        am_varid_t new_varid = am_ast_make_unique_import_ref(ast, varid);
        if (new_varid == SIZE_MAX) {
            parser_set_error(ctx, L"failed to create unique import ref");
            return;
        }

        if (am_list_set(ast->alloc, lst, i, am_make_value_of_varid(new_varid)) != 0) {
            parser_set_error(ctx, L"failed to set renamed import ref");
            return;
        }
    }
}


static void alias_rename_analysis(parser_ctx_t *ctx) {
    if (!ctx || !ctx->ast) return;
    alias_rename_iter_data_t data = { ctx, NULL, 0, 0 };
    am_heap_iter(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, alias_rename_iter_cb, &data);

    if (!ctx->error) {
        for (size_t i = 0; i < data.old_aliases_length; i++) {
            am_map_delete(ctx->ast->alloc, ctx->ast->dependencies,
                          am_make_value_of_varid(data.old_aliases[i]));
        }
    }

    alias_rename_free_old_aliases(&data);
}


// ===============================================================================
// 设置顶层 lambda 与顶级变量列表
// ===============================================================================

static void populate_top_lambda_and_var_top(parser_ctx_t *ctx) {
    if (!ctx || !ctx->ast) return;
    am_ast_t *ast = ctx->ast;

    // 清空 var_top，避免与 macro_rebuild_var_top 重复累积
    if (ast->var_top) {
        ast->var_top->length = 0;
    }

    ast->top_lambda_handle = am_ast_get_top_lambda_node_handle(ast);
    if (ast->top_lambda_handle == AM_HANDLE_NULL) {
        parser_set_error(ctx, L"failed to get top lambda handle");
        return;
    }

    am_value_t *bodies = am_ast_get_global_nodes(ast);
    if (!bodies) return;

    am_value_t lambda_val = am_ast_get_node(ast, ast->top_lambda_handle);
    if (!am_value_is_ptr(lambda_val)) {
        free(bodies);
        parser_set_error(ctx, L"top lambda node not found");
        return;
    }

    am_list_t *lambda = (am_list_t *)am_value_to_ptr(lambda_val);
    size_t n_body = am_list_lambda_get_body_number(ast->alloc, lambda);

    for (size_t i = 0; i < n_body; i++) {
        am_value_t body = bodies[i];
        if (!am_value_is_handle(body)) continue;

        am_value_t node_val = am_ast_get_node(ast, am_value_to_handle(body));
        if (!am_value_is_ptr(node_val)) continue;

        am_object_t *obj = am_value_to_ptr(node_val);
        if (obj->type != AM_OBJECT_TYPE_LIST) continue;

        am_list_t *lst = (am_list_t *)obj;
        if (lst->type != AM_LIST_TYPE_APPLICATION || lst->length < 2) continue;

        am_value_t first = am_list_get(ast->alloc, lst, 0);
        if (!am_value_is_symbol(first) ||
            am_value_to_symbol(first) != am_value_to_symbol(AM_VALUE_KW_define)) {
            continue;
        }

        am_value_t second = am_list_get(ast->alloc, lst, 1);
        if (!am_value_is_varid(second)) continue;

        if (am_ast_add_var_top(ast, am_value_to_varid(second)) < 0) {
            parser_set_error(ctx, L"failed to add top variable");
            break;
        }
    }

    free(bodies);
}


// ===============================================================================
// 尾位置分析
// ===============================================================================

static int32_t tail_call_analysis_item(am_ast_t *ast, am_value_t item, int is_tail);


static int32_t tail_call_analysis_application(am_ast_t *ast, am_handle_t handle, am_list_t *lst, int is_tail) {
    if (!ast || !lst) return -1;

    if (lst->length == 0) {
        if (is_tail) {
            if (am_ast_add_tailcall(ast, handle) != 0) return -1;
        }
        return 0;
    }

    am_value_t first = am_list_get(ast->alloc, lst, 0);

    // if 特殊构造: (if cond then else)
    if (am_value_is_symbol(first) &&
        am_value_to_symbol(first) == am_value_to_symbol(AM_VALUE_KW_if)) {
        if (lst->length > 1) {
            if (tail_call_analysis_item(ast, am_list_get(ast->alloc, lst, 1), 0) != 0) return -1;
        }
        if (lst->length > 2) {
            if (tail_call_analysis_item(ast, am_list_get(ast->alloc, lst, 2), is_tail) != 0) return -1;
        }
        if (lst->length > 3) {
            if (tail_call_analysis_item(ast, am_list_get(ast->alloc, lst, 3), is_tail) != 0) return -1;
        }
    }
    // cond 特殊构造: (cond (c1 e1) (c2 e2) ...)
    else if (am_value_is_symbol(first) &&
             am_value_to_symbol(first) == am_value_to_symbol(AM_VALUE_KW_cond)) {
        for (size_t i = 1; i < lst->length; i++) {
            am_value_t clause = am_list_get(ast->alloc, lst, i);
            if (!am_value_is_handle(clause)) continue;

            am_value_t clause_val = am_ast_get_node(ast, am_value_to_handle(clause));
            if (!am_value_is_ptr(clause_val)) continue;

            am_object_t *clause_obj = am_value_to_ptr(clause_val);
            if (clause_obj->type != AM_OBJECT_TYPE_LIST) continue;

            am_list_t *clause_lst = (am_list_t *)clause_obj;
            if (clause_lst->type != AM_LIST_TYPE_APPLICATION) continue;

            if (clause_lst->length > 0) {
                if (tail_call_analysis_item(ast, am_list_get(ast->alloc, clause_lst, 0), 0) != 0) return -1;
            }
            if (clause_lst->length > 1) {
                if (tail_call_analysis_item(ast, am_list_get(ast->alloc, clause_lst, 1), is_tail) != 0) return -1;
            }
        }
    }
    // 其他构造，含 begin、and、or
    else {
        int is_seq_form = am_value_is_symbol(first) &&
            (am_value_to_symbol(first) == am_value_to_symbol(AM_VALUE_KW_begin) ||
             am_value_to_symbol(first) == am_value_to_symbol(AM_VALUE_KW_and) ||
             am_value_to_symbol(first) == am_value_to_symbol(AM_VALUE_KW_or));

        for (size_t i = 0; i < lst->length; i++) {
            int child_is_tail = 0;
            if (is_seq_form && i == lst->length - 1) {
                child_is_tail = is_tail;
            }
            if (tail_call_analysis_item(ast, am_list_get(ast->alloc, lst, i), child_is_tail) != 0) return -1;
        }
    }

    if (is_tail) {
        if (am_ast_add_tailcall(ast, handle) != 0) return -1;
    }
    return 0;
}


static int32_t tail_call_analysis_item(am_ast_t *ast, am_value_t item, int is_tail) {
    if (!ast) return -1;
    if (!am_value_is_handle(item)) return 0;

    am_handle_t handle = am_value_to_handle(item);
    am_value_t node_val = am_ast_get_node(ast, handle);
    if (!am_value_is_ptr(node_val)) return 0;

    am_object_t *obj = am_value_to_ptr(node_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return 0;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->type == AM_LIST_TYPE_APPLICATION) {
        return tail_call_analysis_application(ast, handle, lst, is_tail);
    }
    else if (lst->type == AM_LIST_TYPE_LAMBDA) {
        size_t n_body = 0;
        am_value_t *bodies = am_list_lambda_get_bodies(ast->alloc, lst, &n_body);
        if (!bodies) return 0;

        int32_t result = 0;
        for (size_t i = 0; i < n_body; i++) {
            if (tail_call_analysis_item(ast, bodies[i], (i == n_body - 1) ? 1 : 0) != 0) {
                result = -1;
                break;
            }
        }
        free(bodies);
        return result;
    }
    // QUOTE / QUASIQUOTE / UNQUOTE 不进入深层分析（被引用的代码不参与执行）
    return 0;
}


// 对 AST 执行整体的尾位置分析。
// 调用前会清空 ast->tailcall_handles，因此可安全地重复调用。
int32_t am_parser_tail_call_analysis(am_ast_t *ast) {
    if (!ast) return -1;
    if (ast->tailcall_handles) {
        ast->tailcall_handles->length = 0;
    }

    am_handle_t top_app = am_ast_get_top_node_handle(ast);
    if (top_app == AM_HANDLE_NULL) return -1;

    return tail_call_analysis_item(ast, am_make_value_of_handle(top_app), 1);
}




// ===============================================================================
// 解析器入口
// ===============================================================================

am_ast_t *am_parse(am_allocator_t *alloc, wchar_t *code, wchar_t *absolute_path, int32_t is_keep_free) {
    if (!alloc || !code || !absolute_path) return NULL;

    // 词法分析
    size_t max_tokens = wcslen(code) + 1;
    am_token_t *tokens = (am_token_t *)am_calloc(alloc, max_tokens * sizeof(am_token_t));
    if (!tokens) return NULL;

    int32_t count = am_lexer(code, tokens);
    if (count < 0) {
        am_free(alloc, tokens);
        return NULL;
    }

    // 创建 AST
    am_ast_t *ast = am_ast_create(alloc, code, absolute_path, tokens, (size_t)count);
    if (!ast) {
        am_free(alloc, tokens);
        return NULL;
    }

    // 构建词汇表：symbol 预置所有关键字；变量在 parse_identifier 中按需注册，
    // 避免把 quote/quasiquote 内部的标识符错误加入 var_vocab。
    am_build_symbol_vocabulary(ast);

    // 初始化解析器上下文
    parser_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.ast = ast;
    ctx.tokens = tokens;
    ctx.token_count = (size_t)count;
    ctx.is_keep_free = is_keep_free;

    // 初始栈：顶部为 TOP_NODE_HANDLE（已是编码后的 am_value_t）
    node_stack_push(&ctx, am_make_value_of_handle(AM_TOP_NODE_HANDLE));

    // 递归下降语法分析
    size_t final_index = am_parser__parse_term(&ctx, 0);

    if (ctx.error || final_index != ctx.token_count) {
        if (!ctx.error) {
            parser_set_error(&ctx, L"trailing tokens after parse");
        }
        fprintf(stderr, "[Parser Error] %ls\n", ctx.error_msg);
        free(ctx.node_stack);
        free(ctx.state_stack);
        free(ctx.lambda_stack);
        free(ctx.special_app_stack);
        am_ast_destroy(ast);
        am_free(alloc, tokens);
        return NULL;
    }

    // 预处理指令解析
    preprocess_analysis(&ctx);
    if (ctx.error) {
        fprintf(stderr, "[Parser Error] %ls\n", ctx.error_msg);
        free(ctx.node_stack);
        free(ctx.state_stack);
        free(ctx.lambda_stack);
        free(ctx.special_app_stack);
        am_ast_destroy(ast);
        am_free(alloc, tokens);
        return NULL;
    }

    // 引用模块别名（alias）和外部引用（ext_ref）更名
    alias_rename_analysis(&ctx);
    if (ctx.error) {
        fprintf(stderr, "[Parser Error] %ls\n", ctx.error_msg);
        free(ctx.node_stack);
        free(ctx.state_stack);
        free(ctx.lambda_stack);
        free(ctx.special_app_stack);
        am_ast_destroy(ast);
        am_free(alloc, tokens);
        return NULL;
    }

    // Alpha-renaming（两阶段变量换名）
    alpha_rename_analysis(&ctx);
    if (ctx.error) {
        fprintf(stderr, "[Parser Error] %ls\n", ctx.error_msg);
        free(ctx.node_stack);
        free(ctx.state_stack);
        free(ctx.lambda_stack);
        free(ctx.special_app_stack);
        am_ast_destroy(ast);
        am_free(alloc, tokens);
        return NULL;
    }

    // syntax-rules 卫生宏展开
    if (am_macro_expand(ast) != 0) {
        free(ctx.node_stack);
        free(ctx.state_stack);
        free(ctx.lambda_stack);
        free(ctx.special_app_stack);
        am_ast_destroy(ast);
        am_free(alloc, tokens);
        return NULL;
    }

    // 清理 ARN 阶段遗留的 scope 对象
    cleanup_scope_objects(&ctx);
    if (ctx.error) {
        fprintf(stderr, "[Parser Error] %ls\n", ctx.error_msg);
        free(ctx.node_stack);
        free(ctx.state_stack);
        free(ctx.lambda_stack);
        free(ctx.special_app_stack);
        am_ast_destroy(ast);
        am_free(alloc, tokens);
        return NULL;
    }

    // 设置顶层 lambda 与顶级变量列表
    populate_top_lambda_and_var_top(&ctx);
    if (ctx.error) {
        fprintf(stderr, "[Parser Error] %ls\n", ctx.error_msg);
        free(ctx.node_stack);
        free(ctx.state_stack);
        free(ctx.lambda_stack);
        free(ctx.special_app_stack);
        am_ast_destroy(ast);
        am_free(alloc, tokens);
        return NULL;
    }

    // 尾位置分析：eval 等不经 am_link 直接编译的路径依赖此结果；
    // 经 am_link 的主流程会在模块融合后清空并重新分析
    if (am_parser_tail_call_analysis(ast) != 0) {
        fprintf(stderr, "[Parser Error] tail call analysis failed\n");
        free(ctx.node_stack);
        free(ctx.state_stack);
        free(ctx.lambda_stack);
        free(ctx.special_app_stack);
        am_ast_destroy(ast);
        am_free(alloc, tokens);
        return NULL;
    }

    free(ctx.node_stack);
    free(ctx.state_stack);
    free(ctx.lambda_stack);
    free(ctx.special_app_stack);
    return ast;
}
/* ===== end:   src/am_parser.c ===== */

/* ===== begin: src/am_macro.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>



// ===============================================================================
// 宏描述符与宏环境
// ===============================================================================

typedef struct am_macro_clause_t {
    am_value_t  pattern;
    am_value_t  template;
    am_varid_t *pvars;
    size_t      pvar_count;
} am_macro_clause_t;


typedef struct am_macro_t {
    am_varid_t          name;
    am_ast_t           *ast;
    am_value_t          literals; // handle to list of literal varids, or AM_VALUE_NULL
    am_macro_clause_t  *clauses;
    size_t              clause_count;
    size_t              expansion_counter;
} am_macro_t;


typedef struct am_macro_env_frame_t {
    am_map_t *bindings; // varid -> am_macro_t*
    struct am_macro_env_frame_t *parent;
    am_ast_t *ast;      // 拥有 bindings map 的 AST，用于销毁 map
} am_macro_env_frame_t;


typedef struct am_macro_expand_ctx_t {
    am_ast_t                *ast;
    am_macro_env_frame_t    *env;
    size_t                   expansion_id;
    am_map_t                *fresh_map; // template-bound varid -> fresh varid
    am_map_t                *subst;     // pattern varid -> matched value (or list handle for ellipsis)
    am_handle_t              parent;
    int                      error;
    int                      changed;   // 是否实际发生过宏展开或 define-syntax 消除
    wchar_t                  error_msg[256];
    am_macro_t             **allocated_macros;
    size_t                   allocated_macro_count;
    size_t                   allocated_macro_capacity;
} am_macro_expand_ctx_t;


// ===============================================================================
// 基本工具函数
// ===============================================================================

static void macro_set_error(am_macro_expand_ctx_t *ctx, const wchar_t *msg) {
    if (!ctx || ctx->error) return;
    ctx->error = 1;
    wcsncpy(ctx->error_msg, msg, 255);
    ctx->error_msg[255] = L'\0';
}


static am_list_t *macro_list_from_handle(am_ast_t *ast, am_handle_t h) {
    if (!ast || h == AM_HANDLE_NULL) return NULL;
    am_value_t v = am_ast_get_node(ast, h);
    if (!am_value_is_ptr(v)) return NULL;
    return (am_list_t *)am_value_to_ptr(v);
}


static int macro_update_list_handle(am_ast_t *ast, am_handle_t h, am_list_t *old_lst, am_list_t *new_lst) {
    if (!ast || h == AM_HANDLE_NULL || !new_lst) return -1;
    if (new_lst == old_lst) return 0;
    return am_heap_set(ast->alloc, ast->alloc, ast->nodes, h,
                       am_make_value_of_ptr((am_object_t *)new_lst));
}


static int macro_list_push(am_ast_t *ast, am_handle_t h, am_value_t item, am_list_t **out_lst) {
    am_list_t *old_lst = macro_list_from_handle(ast, h);
    if (!old_lst) return -1;
    am_list_t *lst = am_list_push(ast->alloc, old_lst, item);
    if (!lst) return -1;
    if (macro_update_list_handle(ast, h, old_lst, lst) != 0) {
        am_list_destroy(ast->alloc, lst);
        return -1;
    }
    if (out_lst) *out_lst = lst;
    return 0;
}


static int macro_lambda_add_param(am_ast_t *ast, am_handle_t h, am_varid_t param) {
    am_list_t *old_lst = macro_list_from_handle(ast, h);
    if (!old_lst) return -1;
    am_list_t *lst = am_list_lambda_add_parameter(ast->alloc, old_lst, am_make_value_of_varid(param));
    if (!lst) return -1;
    if (macro_update_list_handle(ast, h, old_lst, lst) != 0) {
        am_list_destroy(ast->alloc, lst);
        return -1;
    }
    return 0;
}


static int macro_lambda_add_body(am_ast_t *ast, am_handle_t h, am_value_t body, am_list_t **out_lst) {
    am_list_t *old_lst = macro_list_from_handle(ast, h);
    if (!old_lst) return -1;
    am_list_t *lst = am_list_lambda_add_body(ast->alloc, old_lst, body);
    if (!lst) return -1;
    if (macro_update_list_handle(ast, h, old_lst, lst) != 0) {
        am_list_destroy(ast->alloc, lst);
        return -1;
    }
    if (out_lst) *out_lst = lst;
    return 0;
}


static int macro_is_symbol_value(am_value_t v, am_value_t kw) {
    return am_value_is_symbol(v) && am_value_to_symbol(v) == am_value_to_symbol(kw);
}


static am_list_t *macro_as_list(am_ast_t *ast, am_value_t v);


// 判断 v 是否是 ellipsis 标记。
// 支持以下形式：
//   - AM_VALUE_KW_dot3（application / unquote 中的关键字 ...）
//   - symbol "'..."（quote/quasiquote 中被 parser 转换后的 ...）
//   - (unquote ...) 包装
// in_quote 为真时，不把 "'..." 当成 ellipsis，以保留 (quote ...) 中的字面 ...。
static int macro_is_ellipsis_marker(am_ast_t *ast, am_value_t v, int in_quote) {
    if (macro_is_symbol_value(v, AM_VALUE_KW_dot3)) return 1;

    if (!in_quote && am_value_is_symbol(v)) {
        am_symbol_t sym = am_value_to_symbol(v);
        wchar_t *name = am_vocab_get(ast->alloc, ast->symbol_vocab, &sym);
        if (name && wcscmp(name, L"'...") == 0) return 1;
    }

    if (am_value_is_handle(v)) {
        am_list_t *lst = macro_as_list(ast, v);
        if (lst && lst->type == AM_LIST_TYPE_UNQUOTE && lst->length == 1) {
            return macro_is_ellipsis_marker(ast, am_list_get(ast->alloc, lst, 0), 0);
        }
    }

    return 0;
}


static int macro_is_varid_in_list(am_ast_t *ast, am_value_t v, am_value_t list_handle) {
    if (!am_value_is_varid(v)) return 0;
    if (am_value_is_null(list_handle)) return 0;
    if (!am_value_is_handle(list_handle)) return 0;
    am_list_t *lst = macro_list_from_handle(ast, am_value_to_handle(list_handle));
    if (!lst) return 0;
    for (size_t i = 0; i < lst->length; i++) {
        if (am_value_equal(v, am_list_get(ast->alloc, lst, i))) return 1;
    }
    return 0;
}


// ===============================================================================
// 宏展开产生的新 lambda 作用域注册
// ===============================================================================

static int macro_register_lambda_scope(am_macro_expand_ctx_t *ctx, am_handle_t lambda_h, am_handle_t parent_h) {
    if (lambda_h == AM_HANDLE_NULL) return -1;
    am_handle_t parent_lambda = AM_HANDLE_NULL;
    am_handle_t parent_scope_h = AM_HANDLE_NULL;
    if (parent_h != AM_HANDLE_NULL && parent_h != AM_TOP_NODE_HANDLE) {
        parent_lambda = am_ast_find_nearest_lambda_handle(ctx->ast, parent_h);
        if (parent_lambda != AM_HANDLE_NULL) {
            am_value_t v = am_map_get(ctx->ast->alloc, ctx->ast->scopes,
                                      am_make_value_of_handle(parent_lambda));
            if (am_value_is_handle(v)) parent_scope_h = am_value_to_handle(v);
        }
    }

    am_scope_t *scope = am_scope_create(ctx->ast->alloc, parent_scope_h, parent_lambda, lambda_h, 16);
    if (!scope) return -1;

    am_handle_t scope_h = am_heap_alloc_handle(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes);
    if (scope_h == AM_HANDLE_NULL) {
        am_scope_destroy(ctx->ast->alloc, scope);
        return -1;
    }
    if (am_heap_set(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, scope_h,
                    am_make_value_of_ptr((am_object_t *)scope)) != 0) {
        am_scope_destroy(ctx->ast->alloc, scope);
        return -1;
    }

    am_map_t *m = am_map_set(ctx->ast->alloc, ctx->ast->scopes,
                              am_make_value_of_handle(lambda_h),
                              am_make_value_of_handle(scope_h));
    if (!m) return -1;
    ctx->ast->scopes = m;
    return 0;
}


static int macro_lambda_scope_add_var(am_macro_expand_ctx_t *ctx, am_handle_t lambda_h, am_varid_t var) {
    am_value_t scope_h_val = am_map_get(ctx->ast->alloc, ctx->ast->scopes,
                                        am_make_value_of_handle(lambda_h));
    if (!am_value_is_handle(scope_h_val)) return -1;
    am_handle_t scope_h = am_value_to_handle(scope_h_val);
    am_value_t scope_val = am_heap_get(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, scope_h);
    if (!am_value_is_ptr(scope_val)) return -1;
    am_scope_t *scope = (am_scope_t *)am_value_to_ptr(scope_val);
    am_scope_t *new_scope = am_scope_add_var(ctx->ast->alloc, scope, var, AM_VALUE_UNDEFINED);
    if (!new_scope) return -1;
    if (new_scope != scope) {
        if (am_heap_set(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, scope_h,
                        am_make_value_of_ptr((am_object_t *)new_scope)) != 0) {
            return -1;
        }
    }
    return 0;
}


// ===============================================================================
// 宏环境帧
// ===============================================================================

static am_macro_env_frame_t *macro_env_frame_create(am_ast_t *ast) {
    am_macro_env_frame_t *frame = (am_macro_env_frame_t *)malloc(sizeof(am_macro_env_frame_t));
    if (!frame) return NULL;
    frame->bindings = am_map_create(ast->alloc, 16);
    frame->parent = NULL;
    frame->ast = ast;
    if (!frame->bindings) {
        free(frame);
        return NULL;
    }
    return frame;
}


static void macro_env_frame_destroy(am_macro_env_frame_t *frame) {
    if (!frame) return;
    // bindings 中的 value（am_macro_t*）由 macro_expand 上下文统一释放，
    // 这里先将所有有效槽位的 value 置空，避免 am_map_destroy 误释放它们。
    if (frame->bindings && frame->ast) {
        am_map_t *m = frame->bindings;
        for (size_t i = 0; i < m->capacity; i++) {
            if (m->slots[i].key != AM_MAP_KEY_EMPTY && m->slots[i].key != AM_MAP_KEY_TOMBSTONE) {
                m->slots[i].value = AM_VALUE_NULL;
            }
        }
        am_map_destroy(frame->ast->alloc, frame->bindings);
    }
    free(frame);
}


static am_macro_t *macro_env_lookup(am_macro_env_frame_t *frame, am_varid_t name) {
    for (am_macro_env_frame_t *f = frame; f; f = f->parent) {
        am_value_t v = am_map_get(NULL, f->bindings, am_make_value_of_varid(name));
        if (am_value_is_ptr(v)) {
            return (am_macro_t *)am_value_to_ptr(v);
        }
    }
    return NULL;
}


static int macro_env_define(am_ast_t *ast, am_macro_env_frame_t *frame,
                            am_varid_t name, am_macro_t *macro) {
    am_map_t *m = am_map_set(ast->alloc, frame->bindings,
                              am_make_value_of_varid(name),
                              am_make_value_of_ptr((am_object_t *)macro));
    if (!m) return -1;
    frame->bindings = m;
    return 0;
}


// ===============================================================================
// 模式变量收集
// ===============================================================================

typedef struct {
    am_ast_t *ast;
    am_value_t *pvars;
    size_t count;
    size_t capacity;
    am_value_t literals;
} macro_pvar_collect_t;


static int macro_pvar_collect_add(macro_pvar_collect_t *collect, am_value_t v) {
    if (!collect || !am_value_is_varid(v)) return 0;
    // 避免重复
    for (size_t i = 0; i < collect->count; i++) {
        if (am_value_equal(collect->pvars[i], v)) return 0;
    }
    if (collect->count >= collect->capacity) {
        size_t new_cap = collect->capacity ? collect->capacity * 2 : 8;
        am_value_t *new_arr = (am_value_t *)realloc(collect->pvars, new_cap * sizeof(am_value_t));
        if (!new_arr) return -1;
        collect->pvars = new_arr;
        collect->capacity = new_cap;
    }
    collect->pvars[collect->count++] = v;
    return 0;
}


static int macro_collect_pattern_vars_recursive(macro_pvar_collect_t *collect, am_value_t pattern);


static int macro_collect_pattern_vars_list(macro_pvar_collect_t *collect, am_list_t *lst) {
    for (size_t i = 0; i < lst->length; i++) {
        am_value_t child = am_list_get(collect->ast->alloc, lst, i);
        // 跳过 ellipsis 标记本身
        if (macro_is_symbol_value(child, AM_VALUE_KW_dot3)) continue;
        // 如果当前 child 后面跟着 ...，则该 child 是 ellipsis 模式，仍递归收集其内部模式变量
        if (macro_collect_pattern_vars_recursive(collect, child) != 0) return -1;
    }
    return 0;
}


static int macro_collect_pattern_vars_recursive(macro_pvar_collect_t *collect, am_value_t pattern) {
    if (!collect) return 0;

    if (am_value_is_varid(pattern)) {
        // 排除 literals 与 _
        if (macro_is_symbol_value(pattern, AM_VALUE_KW_underscore)) return 0;
        if (macro_is_varid_in_list(collect->ast, pattern, collect->literals)) return 0;
        return macro_pvar_collect_add(collect, pattern);
    }

    if (am_value_is_handle(pattern)) {
        am_list_t *lst = macro_list_from_handle(collect->ast, am_value_to_handle(pattern));
        if (!lst) return 0;
        return macro_collect_pattern_vars_list(collect, lst);
    }

    return 0;
}


// ===============================================================================
// syntax-rules 解析
// ===============================================================================

static void macro_free_macro(am_macro_t *macro) {
    if (!macro) return;
    if (macro->clauses) {
        for (size_t i = 0; i < macro->clause_count; i++) {
            free(macro->clauses[i].pvars);
        }
        free(macro->clauses);
    }
    free(macro);
}


// 如果一个标识符的名字与某个模式变量相同，则把它规范化为该模式变量的 varid。
// 这是必要的，因为 ARN 可能给模板中的同名标识符分配了与模式变量不同的 varid
//（例如模板标识符恰好与外层变量同名）。
static const wchar_t *macro_var_basename(const wchar_t *name) {
    const wchar_t *p = wcsrchr(name, L'.');
    return p ? p + 1 : name;
}


static am_value_t macro_canonicalize_varid(am_ast_t *ast, am_varid_t varid,
                                            am_varid_t *pvars, wchar_t **pvar_names, size_t pvar_count) {
    wchar_t *name = am_vocab_get(ast->alloc, ast->var_vocab, &varid);
    if (!name) return am_make_value_of_varid(varid);
    const wchar_t *base = macro_var_basename(name);
    for (size_t i = 0; i < pvar_count; i++) {
        if (pvar_names[i] && wcscmp(base, macro_var_basename(pvar_names[i])) == 0) {
            return am_make_value_of_varid(pvars[i]);
        }
    }
    return am_make_value_of_varid(varid);
}


// 递归规范化模板中的模式变量标识符。quote 内部保持原样。
static int macro_canonicalize_template_vars(am_ast_t *ast, am_value_t value,
                                             am_varid_t *pvars, wchar_t **pvar_names, size_t pvar_count) {
    if (am_value_is_varid(value)) {
        return 0; // 由调用处直接替换
    }
    if (!am_value_is_handle(value)) return 0;
    am_handle_t h = am_value_to_handle(value);
    am_value_t node_val = am_ast_get_node(ast, h);
    if (!am_value_is_ptr(node_val)) return 0;
    am_object_t *obj = am_value_to_ptr(node_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return 0;
    am_list_t *lst = (am_list_t *)obj;
    if (lst->type == AM_LIST_TYPE_QUOTE) return 0;
    for (size_t i = 0; i < lst->length; i++) {
        am_value_t child = am_list_get(ast->alloc, lst, i);
        if (am_value_is_varid(child)) {
            lst->children[i] = macro_canonicalize_varid(ast, am_value_to_varid(child),
                                                        pvars, pvar_names, pvar_count);
        } else if (am_value_is_handle(child)) {
            if (macro_canonicalize_template_vars(ast, child, pvars, pvar_names, pvar_count) != 0) return -1;
        }
    }
    return 0;
}


static am_macro_t *macro_parse_syntax_rules(am_ast_t *ast, am_varid_t name, am_handle_t sr_handle) {
    am_list_t *sr_lst = macro_list_from_handle(ast, sr_handle);
    if (!sr_lst || sr_lst->type != AM_LIST_TYPE_APPLICATION || sr_lst->length < 3) {
        return NULL;
    }

    am_value_t first = am_list_get(ast->alloc, sr_lst, 0);
    if (!macro_is_symbol_value(first, AM_VALUE_KW_syntax_rules)) {
        return NULL;
    }

    am_value_t literals = am_list_get(ast->alloc, sr_lst, 1);
    if (!am_value_is_handle(literals) && !am_value_is_null(literals)) {
        return NULL;
    }
    if (am_value_is_handle(literals)) {
        am_list_t *lit_lst = macro_list_from_handle(ast, am_value_to_handle(literals));
        if (!lit_lst) return NULL;
        // literals 列表中的每个元素必须是 varid 或 symbol（关键字）
        for (size_t i = 0; i < lit_lst->length; i++) {
            am_value_t lit = am_list_get(ast->alloc, lit_lst, i);
            if (!am_value_is_varid(lit) && !am_value_is_symbol(lit)) {
                return NULL;
            }
        }
    }

    size_t clause_count = sr_lst->length - 2;
    if (clause_count == 0) return NULL;

    am_macro_t *macro = (am_macro_t *)calloc(1, sizeof(am_macro_t));
    if (!macro) return NULL;
    macro->name = name;
    macro->ast = ast;
    macro->literals = literals;
    macro->expansion_counter = 0;
    macro->clause_count = clause_count;
    macro->clauses = (am_macro_clause_t *)calloc(clause_count, sizeof(am_macro_clause_t));
    if (!macro->clauses) {
        free(macro);
        return NULL;
    }

    for (size_t i = 0; i < clause_count; i++) {
        am_value_t clause_val = am_list_get(ast->alloc, sr_lst, i + 2);
        if (!am_value_is_handle(clause_val)) {
            macro_free_macro(macro);
            return NULL;
        }
        am_list_t *clause_lst = macro_list_from_handle(ast, am_value_to_handle(clause_val));
        if (!clause_lst || clause_lst->type != AM_LIST_TYPE_APPLICATION || clause_lst->length != 2) {
            macro_free_macro(macro);
            return NULL;
        }

        am_macro_clause_t *clause = &macro->clauses[i];
        clause->pattern = am_list_get(ast->alloc, clause_lst, 0);
        clause->template = am_list_get(ast->alloc, clause_lst, 1);

        macro_pvar_collect_t collect = { ast, NULL, 0, 0, literals };
        if (macro_collect_pattern_vars_recursive(&collect, clause->pattern) != 0) {
            free(collect.pvars);
            macro_free_macro(macro);
            return NULL;
        }
        clause->pvar_count = collect.count;
        clause->pvars = (am_varid_t *)malloc(collect.count * sizeof(am_varid_t));
        if (!clause->pvars) {
            free(collect.pvars);
            macro_free_macro(macro);
            return NULL;
        }
        for (size_t j = 0; j < collect.count; j++) {
            clause->pvars[j] = am_value_to_varid(collect.pvars[j]);
        }
        free(collect.pvars);

        // 规范化模板中的模式变量：把与模式变量同名的标识符统一成模式变量的 varid。
        if (clause->pvar_count > 0) {
            wchar_t **pvar_names = (wchar_t **)malloc(clause->pvar_count * sizeof(wchar_t *));
            if (!pvar_names) {
                macro_free_macro(macro);
                return NULL;
            }
            for (size_t j = 0; j < clause->pvar_count; j++) {
                pvar_names[j] = am_vocab_get(ast->alloc, ast->var_vocab, &clause->pvars[j]);
            }
            if (am_value_is_varid(clause->template)) {
                clause->template = macro_canonicalize_varid(ast, am_value_to_varid(clause->template),
                                                            clause->pvars, pvar_names, clause->pvar_count);
            } else if (am_value_is_handle(clause->template)) {
                if (macro_canonicalize_template_vars(ast, clause->template, clause->pvars,
                                                     pvar_names, clause->pvar_count) != 0) {
                    free(pvar_names);
                    macro_free_macro(macro);
                    return NULL;
                }
            }
            free(pvar_names);
        }
    }

    return macro;
}


// ===============================================================================
// 模式匹配
// ===============================================================================

static int macro_is_pattern_var(am_macro_t *macro, am_macro_clause_t *clause, am_varid_t varid) {
    (void)macro;
    for (size_t i = 0; i < clause->pvar_count; i++) {
        if (clause->pvars[i] == varid) return 1;
    }
    return 0;
}


static int macro_is_literal(am_macro_t *macro, am_varid_t varid) {
    return macro_is_varid_in_list(macro->ast, am_make_value_of_varid(varid), macro->literals);
}


static am_map_t *macro_subst_clone(am_ast_t *ast, am_map_t *subst) {
    if (!subst) return am_map_create(ast->alloc, 16);
    return am_map_copy(ast->alloc, subst);
}


static am_handle_t macro_ellipsis_list_get_or_create(am_macro_expand_ctx_t *ctx, am_varid_t pvar) {
    am_value_t v = am_map_get(NULL, ctx->subst, am_make_value_of_varid(pvar));
    if (am_value_is_handle(v)) return am_value_to_handle(v);
    am_handle_t parent = ctx->parent;
    if (parent == AM_HANDLE_NULL || parent == AM_TOP_NODE_HANDLE) parent = 0;
    am_handle_t h = am_ast_make_slist_node(ctx->ast, parent, AM_LIST_TYPE_APPLICATION);
    if (h == AM_HANDLE_NULL) return AM_HANDLE_NULL;
    am_map_t *m = am_map_set(ctx->ast->alloc, ctx->subst,
                              am_make_value_of_varid(pvar), am_make_value_of_handle(h));
    if (!m) return AM_HANDLE_NULL;
    ctx->subst = m;
    return h;
}


static int macro_ellipsis_list_append(am_macro_expand_ctx_t *ctx, am_varid_t pvar, am_value_t value) {
    am_handle_t h = macro_ellipsis_list_get_or_create(ctx, pvar);
    if (h == AM_HANDLE_NULL) return -1;
    return macro_list_push(ctx->ast, h, value, NULL);
}


// 判断 value 是否是列表（任意 list 类型），返回 list 指针；不是则返回 NULL
static am_list_t *macro_as_list(am_ast_t *ast, am_value_t v) {
    if (!am_value_is_handle(v)) return NULL;
    am_list_t *lst = macro_list_from_handle(ast, am_value_to_handle(v));
    if (!lst) return NULL;
    if (lst->type != AM_LIST_TYPE_APPLICATION &&
        lst->type != AM_LIST_TYPE_LAMBDA &&
        lst->type != AM_LIST_TYPE_QUOTE &&
        lst->type != AM_LIST_TYPE_QUASIQUOTE &&
        lst->type != AM_LIST_TYPE_UNQUOTE) {
        return NULL;
    }
    return lst;
}


static int macro_match_value(am_macro_expand_ctx_t *ctx, am_macro_t *macro,
                              am_macro_clause_t *clause, am_value_t pattern, am_value_t input);


static int macro_collect_pvars_in_value(am_macro_t *macro, am_macro_clause_t *clause,
                                         am_value_t value, am_value_t **out_pvars, size_t *out_count);


static int macro_match_list(am_macro_expand_ctx_t *ctx, am_macro_t *macro,
                             am_macro_clause_t *clause, am_list_t *pat_lst, am_list_t *in_lst) {
    // 检测 ellipsis 位置：只允许列表顶层出现一个 ellipsis
    int ellipsis_pos = -1;
    for (size_t i = 0; i < pat_lst->length; i++) {
        am_value_t child = am_list_get(ctx->ast->alloc, pat_lst, i);
        if (macro_is_symbol_value(child, AM_VALUE_KW_dot3)) {
            if (ellipsis_pos >= 0) {
                macro_set_error(ctx, L"multiple ellipses in macro pattern");
                return -1;
            }
            if (i == 0) {
                macro_set_error(ctx, L"ellipsis at beginning of macro pattern");
                return -1;
            }
            ellipsis_pos = (int)(i - 1);
        }
    }

    if (ellipsis_pos < 0) {
        // 无 ellipsis，逐元素匹配
        if (pat_lst->length != in_lst->length) return -1;
        for (size_t i = 0; i < pat_lst->length; i++) {
            am_value_t p = am_list_get(ctx->ast->alloc, pat_lst, i);
            am_value_t in = am_list_get(ctx->ast->alloc, in_lst, i);
            if (macro_match_value(ctx, macro, clause, p, in) != 0) return -1;
        }
        return 0;
    }

    // 有 ellipsis
    size_t prefix_len = (size_t)ellipsis_pos;
    size_t suffix_len = pat_lst->length - prefix_len - 2;
    size_t input_len = in_lst->length;
    if (input_len < prefix_len + suffix_len) return -1;
    size_t k = input_len - prefix_len - suffix_len;

    // 匹配前缀
    for (size_t i = 0; i < prefix_len; i++) {
        am_value_t p = am_list_get(ctx->ast->alloc, pat_lst, i);
        am_value_t in = am_list_get(ctx->ast->alloc, in_lst, i);
        if (macro_match_value(ctx, macro, clause, p, in) != 0) return -1;
    }

    // 匹配 ellipsis 区域
    am_value_t ellip_pattern = am_list_get(ctx->ast->alloc, pat_lst, prefix_len);

    // 预先将 ellipsis 中的模式变量绑定到空列表，以处理匹配 0 次的情况
    am_value_t *ellip_pvars = NULL;
    size_t ellip_pvar_count = 0;
    if (macro_collect_pvars_in_value(macro, clause, ellip_pattern, &ellip_pvars, &ellip_pvar_count) != 0) {
        macro_set_error(ctx, L"out of memory collecting ellipsis pattern variables");
        return -1;
    }
    for (size_t j = 0; j < ellip_pvar_count; j++) {
        if (macro_ellipsis_list_get_or_create(ctx, am_value_to_varid(ellip_pvars[j])) == AM_HANDLE_NULL) {
            free(ellip_pvars);
            return -1;
        }
    }
    free(ellip_pvars);

    for (size_t i = 0; i < k; i++) {
        am_value_t in = am_list_get(ctx->ast->alloc, in_lst, prefix_len + i);
        // 为每次匹配创建独立的 subst，避免跨迭代污染
        am_macro_expand_ctx_t sub_ctx = *ctx;
        sub_ctx.subst = macro_subst_clone(ctx->ast, NULL);
        if (!sub_ctx.subst) {
            macro_set_error(ctx, L"out of memory in macro ellipsis match");
            return -1;
        }
        int ok = macro_match_value(&sub_ctx, macro, clause, ellip_pattern, in);
        if (ok == 0) {
            // 收集本次匹配产生的绑定到 ellipsis 列表
            size_t key_count = am_map_length(NULL, sub_ctx.subst);
            am_value_t *keys = am_map_keys(ctx->ast->alloc, sub_ctx.subst);
            for (size_t j = 0; j < key_count; j++) {
                am_value_t val = am_map_get(NULL, sub_ctx.subst, keys[j]);
                if (macro_ellipsis_list_append(ctx, am_value_to_varid(keys[j]), val) != 0) {
                    am_free(ctx->ast->alloc, keys);
                    am_map_destroy(ctx->ast->alloc, sub_ctx.subst);
                    return -1;
                }
            }
            am_free(ctx->ast->alloc, keys);
        }
        am_map_destroy(ctx->ast->alloc, sub_ctx.subst);
        if (ok != 0) return -1;
    }

    // 匹配后缀
    for (size_t i = 0; i < suffix_len; i++) {
        am_value_t p = am_list_get(ctx->ast->alloc, pat_lst, prefix_len + 2 + i);
        am_value_t in = am_list_get(ctx->ast->alloc, in_lst, prefix_len + k + i);
        if (macro_match_value(ctx, macro, clause, p, in) != 0) return -1;
    }

    return 0;
}


static int macro_match_value(am_macro_expand_ctx_t *ctx, am_macro_t *macro,
                              am_macro_clause_t *clause, am_value_t pattern, am_value_t input) {
    // _ 通配符
    if (macro_is_symbol_value(pattern, AM_VALUE_KW_underscore)) {
        return 0;
    }

    // 模式变量
    if (am_value_is_varid(pattern) && macro_is_pattern_var(macro, clause, am_value_to_varid(pattern))) {
        // am_map 用 AM_VALUE_NULL 作为“不存在”的哨兵，需用 contains 判断是否存在。
        if (am_map_contains(NULL, ctx->subst, pattern) == 0) {
            // 已绑定，要求相等
            am_value_t existing = am_map_get(NULL, ctx->subst, pattern);
            return am_value_equal(existing, input) ? 0 : -1;
        }
        // 新绑定
        am_map_t *m = am_map_set(ctx->ast->alloc, ctx->subst, pattern, input);
        if (!m) {
            macro_set_error(ctx, L"out of memory in macro match");
            return -1;
        }
        ctx->subst = m;
        return 0;
    }

    // 普通 varid：必须是 literal
    if (am_value_is_varid(pattern)) {
        am_varid_t pvid = am_value_to_varid(pattern);
        if (!macro_is_literal(macro, pvid)) {
            macro_set_error(ctx, L"unbound identifier in macro pattern");
            return -1;
        }
        return am_value_equal(pattern, input) ? 0 : -1;
    }

    // 其它立即数（symbol、number、boolean 等）按位比较
    if (!am_value_is_handle(pattern)) {
        return am_value_equal(pattern, input) ? 0 : -1;
    }

    // pattern 是 handle，input 也必须是同类型列表
    am_list_t *pat_lst = macro_as_list(ctx->ast, pattern);
    am_list_t *in_lst = macro_as_list(ctx->ast, input);
    if (!pat_lst || !in_lst) return -1;
    return macro_match_list(ctx, macro, clause, pat_lst, in_lst);
}


// ===============================================================================
// fresh varid 生成
// ===============================================================================

static am_varid_t macro_make_fresh_varid(am_ast_t *ast, am_varid_t base, size_t expansion_id) {
    if (!ast || !ast->var_vocab || !ast->var_type) return SIZE_MAX;

    wchar_t *base_str = am_vocab_get(ast->alloc, ast->var_vocab, &base);
    if (!base_str) return SIZE_MAX;

    size_t module_id_len = wcslen(ast->module_id);
    size_t base_len = wcslen(base_str);
    size_t buf_size = module_id_len + 3 + 20 + 1 + base_len + 1;

    wchar_t *new_name = (wchar_t *)am_malloc(ast->alloc, buf_size * sizeof(wchar_t));
    if (!new_name) return SIZE_MAX;

    int n = swprintf(new_name, buf_size, L"%ls.M%zu.%ls", ast->module_id, expansion_id, base_str);
    if (n <= 0 || (size_t)n >= buf_size) {
        am_free(ast->alloc, new_name);
        return SIZE_MAX;
    }

    size_t old_len = ast->var_vocab->length;
    size_t new_varid;
    ast->var_vocab = am_vocab_insert(ast->alloc, ast->var_vocab, new_name, &new_varid);
    am_free(ast->alloc, new_name);

    if (!ast->var_vocab || new_varid == SIZE_MAX) return SIZE_MAX;
    if (new_varid == old_len) {
        am_list_t *vt = am_list_push(ast->alloc, ast->var_type,
                                      am_make_value_of_uint(AM_VAR_TYPE_NEW));
        if (!vt) return SIZE_MAX;
        ast->var_type = vt;
    }
    return (am_varid_t)new_varid;
}


// ===============================================================================
// 模板绑定收集（lambda 形参与 define 左值）
// ===============================================================================

static int macro_collect_template_bindings_recursive(am_macro_t *macro, am_macro_clause_t *clause,
                                                      am_value_t template, am_map_t **bindings_out);


static int macro_collect_template_bindings_list(am_macro_t *macro, am_macro_clause_t *clause,
                                                 am_list_t *lst, am_map_t **bindings_out) {
    if (!lst) return 0;

    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        am_uint_t n_param = 0;
        if (lst->length >= 2) {
            am_value_t n_val = am_list_get(macro->ast->alloc, lst, 1);
            if (am_value_is_uint(n_val)) n_param = am_value_to_uint(n_val);
        }
        for (size_t i = 0; i < (size_t)n_param; i++) {
            am_value_t p = am_list_get(macro->ast->alloc, lst, 2 + i);
            if (am_value_is_varid(p) && !macro_is_pattern_var(macro, clause, am_value_to_varid(p))) {
                am_map_t *m = am_map_set(macro->ast->alloc, *bindings_out, p, AM_VALUE_TRUE);
                if (!m) return -1;
                *bindings_out = m;
            }
        }
        size_t n_body = am_list_lambda_get_body_number(macro->ast->alloc, lst);
        am_value_t *bodies = am_list_lambda_get_bodies(macro->ast->alloc, lst, &n_body);
        if (bodies) {
            for (size_t i = 0; i < n_body; i++) {
                if (macro_collect_template_bindings_recursive(macro, clause, bodies[i], bindings_out) != 0) {
                    free(bodies);
                    return -1;
                }
            }
            free(bodies);
        }
        return 0;
    }

    // (define var ...) 形式
    if (lst->type == AM_LIST_TYPE_APPLICATION && lst->length >= 2) {
        am_value_t first = am_list_get(macro->ast->alloc, lst, 0);
        if (macro_is_symbol_value(first, AM_VALUE_KW_define)) {
            am_value_t second = am_list_get(macro->ast->alloc, lst, 1);
            if (am_value_is_varid(second) && !macro_is_pattern_var(macro, clause, am_value_to_varid(second))) {
                am_map_t *m = am_map_set(macro->ast->alloc, *bindings_out, second, AM_VALUE_TRUE);
                if (!m) return -1;
                *bindings_out = m;
            }
        }
    }

    for (size_t i = 0; i < lst->length; i++) {
        am_value_t child = am_list_get(macro->ast->alloc, lst, i);
        if (macro_collect_template_bindings_recursive(macro, clause, child, bindings_out) != 0) return -1;
    }
    return 0;
}


static int macro_collect_template_bindings_recursive(am_macro_t *macro, am_macro_clause_t *clause,
                                                      am_value_t template, am_map_t **bindings_out) {
    if (!bindings_out) return 0;
    if (am_value_is_handle(template)) {
        am_list_t *lst = macro_as_list(macro->ast, template);
        if (lst) {
            return macro_collect_template_bindings_list(macro, clause, lst, bindings_out);
        }
    }
    return 0;
}


// ===============================================================================
// 深拷贝（用于替换模式变量时拷贝使用处子树）
// ===============================================================================

static am_value_t macro_deep_copy_list(am_macro_expand_ctx_t *ctx, am_list_t *lst, am_handle_t parent);


static am_value_t macro_deep_copy_value(am_macro_expand_ctx_t *ctx, am_value_t value, am_handle_t parent) {
    if (ctx->error) return AM_VALUE_UNDEFINED;
    if (am_value_is_handle(value)) {
        am_list_t *lst = macro_as_list(ctx->ast, value);
        if (lst) {
            return macro_deep_copy_list(ctx, lst, parent);
        }
        // WString
        am_value_t node_val = am_ast_get_node(ctx->ast, am_value_to_handle(value));
        if (am_value_is_ptr(node_val)) {
            am_object_t *obj = am_value_to_ptr(node_val);
            if (obj->type == AM_OBJECT_TYPE_WSTRING) {
                am_wstring_t *ws = am_wstring_copy(ctx->ast->alloc, (am_wstring_t *)obj);
                if (!ws) { macro_set_error(ctx, L"out of memory copying string"); return AM_VALUE_UNDEFINED; }
                am_handle_t h = am_heap_alloc_handle(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes);
                if (h == AM_HANDLE_NULL) { macro_set_error(ctx, L"out of memory allocating handle"); return AM_VALUE_UNDEFINED; }
                if (am_heap_set(ctx->ast->alloc, ctx->ast->alloc, ctx->ast->nodes, h,
                                am_make_value_of_ptr((am_object_t *)ws)) != 0) {
                    macro_set_error(ctx, L"failed to set heap handle");
                    return AM_VALUE_UNDEFINED;
                }
                return am_make_value_of_handle(h);
            }
        }
    }
    return value;
}


static am_value_t macro_deep_copy_list(am_macro_expand_ctx_t *ctx, am_list_t *lst, am_handle_t parent) {
    if (ctx->error) return AM_VALUE_UNDEFINED;

    am_handle_t new_h;
    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        new_h = am_ast_make_lambda_node(ctx->ast, parent);
    } else {
        new_h = am_ast_make_slist_node(ctx->ast, parent, lst->type);
    }
    if (new_h == AM_HANDLE_NULL) {
        macro_set_error(ctx, L"out of memory creating node");
        return AM_VALUE_UNDEFINED;
    }

    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        if (macro_register_lambda_scope(ctx, new_h, parent) != 0) {
            macro_set_error(ctx, L"failed to register copied lambda scope");
            return AM_VALUE_UNDEFINED;
        }
        am_uint_t n_param = 0;
        if (lst->length >= 2) {
            am_value_t n_val = am_list_get(ctx->ast->alloc, lst, 1);
            if (am_value_is_uint(n_val)) n_param = am_value_to_uint(n_val);
        }
        for (size_t i = 0; i < (size_t)n_param; i++) {
            am_value_t p = am_list_get(ctx->ast->alloc, lst, 2 + i);
            if (macro_lambda_add_param(ctx->ast, new_h, am_value_to_varid(p)) != 0) {
                macro_set_error(ctx, L"failed to copy lambda param");
                return AM_VALUE_UNDEFINED;
            }
            if (macro_lambda_scope_add_var(ctx, new_h, am_value_to_varid(p)) != 0) {
                macro_set_error(ctx, L"failed to add copied lambda param to scope");
                return AM_VALUE_UNDEFINED;
            }
        }
        size_t n_body = am_list_lambda_get_body_number(ctx->ast->alloc, lst);
        am_value_t *bodies = am_list_lambda_get_bodies(ctx->ast->alloc, lst, &n_body);
        if (bodies) {
            for (size_t i = 0; i < n_body; i++) {
                am_value_t copied = macro_deep_copy_value(ctx, bodies[i], new_h);
                if (ctx->error) { free(bodies); return AM_VALUE_UNDEFINED; }
                if (macro_lambda_add_body(ctx->ast, new_h, copied, NULL) != 0) {
                    macro_set_error(ctx, L"failed to copy lambda body");
                    free(bodies);
                    return AM_VALUE_UNDEFINED;
                }
            }
            free(bodies);
        }
    } else {
        for (size_t i = 0; i < lst->length; i++) {
            am_value_t child = am_list_get(ctx->ast->alloc, lst, i);
            am_value_t copied = macro_deep_copy_value(ctx, child, new_h);
            if (ctx->error) return AM_VALUE_UNDEFINED;
            if (macro_list_push(ctx->ast, new_h, copied, NULL) != 0) {
                macro_set_error(ctx, L"failed to copy list child");
                return AM_VALUE_UNDEFINED;
            }
        }
    }

    return am_make_value_of_handle(new_h);
}


// ===============================================================================
// 模板实例化
// ===============================================================================

static am_value_t macro_instantiate(am_macro_expand_ctx_t *ctx, am_macro_t *macro,
                                     am_macro_clause_t *clause, am_value_t template,
                                     am_map_t *template_bindings, am_handle_t parent);


static int macro_collect_pvars_in_value(am_macro_t *macro, am_macro_clause_t *clause,
                                         am_value_t value, am_value_t **out_pvars, size_t *out_count);


static int macro_collect_pvars_in_list(am_macro_t *macro, am_macro_clause_t *clause,
                                        am_list_t *lst, am_value_t **out_pvars, size_t *out_count) {
    for (size_t i = 0; i < lst->length; i++) {
        am_value_t child = am_list_get(macro->ast->alloc, lst, i);
        if (macro_collect_pvars_in_value(macro, clause, child, out_pvars, out_count) != 0) return -1;
    }
    return 0;
}


static int macro_collect_pvars_in_value(am_macro_t *macro, am_macro_clause_t *clause,
                                         am_value_t value, am_value_t **out_pvars, size_t *out_count) {
    if (am_value_is_varid(value) && macro_is_pattern_var(macro, clause, am_value_to_varid(value))) {
        // 去重
        for (size_t i = 0; i < *out_count; i++) {
            if (am_value_equal((*out_pvars)[i], value)) return 0;
        }
        am_value_t *new_arr = (am_value_t *)realloc(*out_pvars, (*out_count + 1) * sizeof(am_value_t));
        if (!new_arr) return -1;
        *out_pvars = new_arr;
        (*out_pvars)[(*out_count)++] = value;
        return 0;
    }
    if (am_value_is_handle(value)) {
        am_list_t *lst = macro_as_list(macro->ast, value);
        if (lst) {
            return macro_collect_pvars_in_list(macro, clause, lst, out_pvars, out_count);
        }
    }
    return 0;
}


// 计算 ellipsis 模板应重复的次数，同时收集其中的模式变量。
// 成功返回重复次数；失败返回 SIZE_MAX，并释放 *out_pvars（如果已分配）。
static size_t macro_ellipsis_length(am_macro_expand_ctx_t *ctx, am_macro_t *macro,
                                     am_macro_clause_t *clause, am_value_t ellip_template,
                                     am_value_t **out_pvars, size_t *out_pvar_count) {
    if (macro_collect_pvars_in_value(macro, clause, ellip_template, out_pvars, out_pvar_count) != 0) {
        return SIZE_MAX;
    }
    if (*out_pvar_count == 0) {
        free(*out_pvars);
        *out_pvars = NULL;
        return SIZE_MAX;
    }

    size_t n = 0;
    int first = 1;
    for (size_t j = 0; j < *out_pvar_count; j++) {
        am_value_t list_val = am_map_get(NULL, ctx->subst, (*out_pvars)[j]);
        if (!am_value_is_handle(list_val)) {
            free(*out_pvars);
            *out_pvars = NULL;
            return SIZE_MAX;
        }
        am_list_t *ellip_list = macro_list_from_handle(ctx->ast, am_value_to_handle(list_val));
        if (!ellip_list) {
            free(*out_pvars);
            *out_pvars = NULL;
            return SIZE_MAX;
        }
        if (first) {
            n = ellip_list->length;
            first = 0;
        } else if (ellip_list->length != n) {
            free(*out_pvars);
            *out_pvars = NULL;
            return SIZE_MAX;
        }
    }
    return n;
}


// 为第 j 次 ellipsis 迭代创建临时 subst：拷贝当前 subst 并覆盖 ellipsis 模式变量。
// 成功返回 0；失败返回 -1。
static int macro_ellipsis_iter_setup(am_macro_expand_ctx_t *ctx, am_macro_expand_ctx_t *iter_ctx,
                                      am_value_t *pvars, size_t pvar_count, size_t j) {
    iter_ctx->subst = am_map_copy(ctx->ast->alloc, ctx->subst);
    if (!iter_ctx->subst) return -1;
    for (size_t k = 0; k < pvar_count; k++) {
        am_value_t list_val = am_map_get(NULL, ctx->subst, pvars[k]);
        am_list_t *ellip_list = macro_list_from_handle(ctx->ast, am_value_to_handle(list_val));
        am_value_t elem = am_list_get(ctx->ast->alloc, ellip_list, j);
        am_map_t *m = am_map_set(ctx->ast->alloc, iter_ctx->subst, pvars[k], elem);
        if (!m) {
            am_map_destroy(ctx->ast->alloc, iter_ctx->subst);
            iter_ctx->subst = NULL;
            return -1;
        }
        iter_ctx->subst = m;
    }
    return 0;
}


// 实例化 lambda 模板，支持参数和函数体中的 ellipsis 展开。
static am_value_t macro_instantiate_lambda(am_macro_expand_ctx_t *ctx, am_macro_t *macro,
                                            am_macro_clause_t *clause, am_list_t *lst,
                                            am_map_t *template_bindings, am_handle_t parent) {
    if (ctx->error) return AM_VALUE_UNDEFINED;

    am_handle_t new_h = am_ast_make_lambda_node(ctx->ast, parent);
    if (new_h == AM_HANDLE_NULL) {
        macro_set_error(ctx, L"out of memory instantiating lambda");
        return AM_VALUE_UNDEFINED;
    }

    if (macro_register_lambda_scope(ctx, new_h, parent) != 0) {
        macro_set_error(ctx, L"failed to register lambda scope");
        return AM_VALUE_UNDEFINED;
    }

    // 参数处理
    am_uint_t n_param = 0;
    if (lst->length >= 2) {
        am_value_t n_val = am_list_get(ctx->ast->alloc, lst, 1);
        if (am_value_is_uint(n_val)) n_param = am_value_to_uint(n_val);
    }

    size_t param_fixed_end = (size_t)n_param;
    int param_has_ellipsis = 0;
    if (n_param >= 2) {
        am_value_t last_param = am_list_get(ctx->ast->alloc, lst, 2 + n_param - 1);
        if (macro_is_ellipsis_marker(ctx->ast, last_param, 0)) {
            param_has_ellipsis = 1;
            param_fixed_end = n_param - 2;
        }
    }

    for (size_t i = 0; i < param_fixed_end; i++) {
        am_value_t p = am_list_get(ctx->ast->alloc, lst, 2 + i);
        am_value_t inst_p = macro_instantiate(ctx, macro, clause, p, template_bindings, new_h);
        if (ctx->error) return AM_VALUE_UNDEFINED;
        if (!am_value_is_varid(inst_p)) {
            macro_set_error(ctx, L"lambda parameter must be variable");
            return AM_VALUE_UNDEFINED;
        }
        if (macro_lambda_add_param(ctx->ast, new_h, am_value_to_varid(inst_p)) != 0) {
            macro_set_error(ctx, L"failed to add instantiated lambda param");
            return AM_VALUE_UNDEFINED;
        }
        if (macro_lambda_scope_add_var(ctx, new_h, am_value_to_varid(inst_p)) != 0) {
            macro_set_error(ctx, L"failed to add lambda param to scope");
            return AM_VALUE_UNDEFINED;
        }
    }

    if (param_has_ellipsis) {
        am_value_t ellip_template = am_list_get(ctx->ast->alloc, lst, 2 + param_fixed_end);
        am_value_t *pvars = NULL;
        size_t pvar_count = 0;
        size_t n = macro_ellipsis_length(ctx, macro, clause, ellip_template, &pvars, &pvar_count);
        if (n == SIZE_MAX) {
            macro_set_error(ctx, L"invalid ellipsis in lambda parameter list");
            return AM_VALUE_UNDEFINED;
        }

        for (size_t j = 0; j < n; j++) {
            am_macro_expand_ctx_t iter_ctx = *ctx;
            if (macro_ellipsis_iter_setup(ctx, &iter_ctx, pvars, pvar_count, j) != 0) {
                free(pvars);
                macro_set_error(ctx, L"out of memory in lambda parameter ellipsis");
                return AM_VALUE_UNDEFINED;
            }
            am_value_t inst_p = macro_instantiate(&iter_ctx, macro, clause, ellip_template,
                                                 template_bindings, new_h);
            am_map_destroy(ctx->ast->alloc, iter_ctx.subst);
            if (iter_ctx.error) {
                ctx->error = 1;
                wcsncpy(ctx->error_msg, iter_ctx.error_msg, 256);
                free(pvars);
                return AM_VALUE_UNDEFINED;
            }
            if (!am_value_is_varid(inst_p)) {
                macro_set_error(ctx, L"lambda ellipsis parameter must be variable");
                free(pvars);
                return AM_VALUE_UNDEFINED;
            }
            if (macro_lambda_add_param(ctx->ast, new_h, am_value_to_varid(inst_p)) != 0) {
                macro_set_error(ctx, L"failed to add lambda ellipsis param");
                free(pvars);
                return AM_VALUE_UNDEFINED;
            }
            if (macro_lambda_scope_add_var(ctx, new_h, am_value_to_varid(inst_p)) != 0) {
                macro_set_error(ctx, L"failed to add lambda ellipsis param to scope");
                free(pvars);
                return AM_VALUE_UNDEFINED;
            }
        }
        free(pvars);
    }

    // 函数体处理
    size_t n_body = 0;
    am_value_t *bodies = am_list_lambda_get_bodies(ctx->ast->alloc, lst, &n_body);

    size_t body_fixed_end = n_body;
    int body_has_ellipsis = 0;
    if (n_body >= 2) {
        am_value_t last_body = bodies[n_body - 1];
        if (macro_is_ellipsis_marker(ctx->ast, last_body, 0)) {
            body_has_ellipsis = 1;
            body_fixed_end = n_body - 2;
        }
    }

    for (size_t i = 0; i < body_fixed_end; i++) {
        am_value_t inst = macro_instantiate(ctx, macro, clause, bodies[i], template_bindings, new_h);
        if (ctx->error) {
            free(bodies);
            return AM_VALUE_UNDEFINED;
        }
        if (macro_lambda_add_body(ctx->ast, new_h, inst, NULL) != 0) {
            macro_set_error(ctx, L"failed to add instantiated lambda body");
            free(bodies);
            return AM_VALUE_UNDEFINED;
        }
    }

    if (body_has_ellipsis) {
        am_value_t ellip_template = bodies[body_fixed_end];
        am_value_t *pvars = NULL;
        size_t pvar_count = 0;
        size_t n = macro_ellipsis_length(ctx, macro, clause, ellip_template, &pvars, &pvar_count);
        if (n == SIZE_MAX) {
            free(bodies);
            macro_set_error(ctx, L"invalid ellipsis in lambda body");
            return AM_VALUE_UNDEFINED;
        }

        for (size_t j = 0; j < n; j++) {
            am_macro_expand_ctx_t iter_ctx = *ctx;
            if (macro_ellipsis_iter_setup(ctx, &iter_ctx, pvars, pvar_count, j) != 0) {
                free(pvars);
                free(bodies);
                macro_set_error(ctx, L"out of memory in lambda body ellipsis");
                return AM_VALUE_UNDEFINED;
            }
            am_value_t inst = macro_instantiate(&iter_ctx, macro, clause, ellip_template,
                                                 template_bindings, new_h);
            am_map_destroy(ctx->ast->alloc, iter_ctx.subst);
            if (iter_ctx.error) {
                ctx->error = 1;
                wcsncpy(ctx->error_msg, iter_ctx.error_msg, 256);
                free(pvars);
                free(bodies);
                return AM_VALUE_UNDEFINED;
            }
            if (macro_lambda_add_body(ctx->ast, new_h, inst, NULL) != 0) {
                macro_set_error(ctx, L"failed to add lambda ellipsis body");
                free(pvars);
                free(bodies);
                return AM_VALUE_UNDEFINED;
            }
        }
        free(pvars);
    }

    free(bodies);
    return am_make_value_of_handle(new_h);
}


static am_value_t macro_instantiate_list(am_macro_expand_ctx_t *ctx, am_macro_t *macro,
                                          am_macro_clause_t *clause, am_list_t *lst,
                                          am_map_t *template_bindings, am_handle_t parent) {
    if (ctx->error) return AM_VALUE_UNDEFINED;

    am_handle_t new_h;
    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        return macro_instantiate_lambda(ctx, macro, clause, lst, template_bindings, parent);
    } else {
        new_h = am_ast_make_slist_node(ctx->ast, parent, lst->type);
    }
    if (new_h == AM_HANDLE_NULL) {
        macro_set_error(ctx, L"out of memory instantiating list");
        return AM_VALUE_UNDEFINED;
    }

    // 普通列表：扫描子元素，处理 ellipsis
    int in_quote = (lst->type == AM_LIST_TYPE_QUOTE);
    for (size_t i = 0; i < lst->length; ) {
        am_value_t child = am_list_get(ctx->ast->alloc, lst, i);

        // ellipsis 模板：T ...
        if (i + 1 < lst->length &&
            macro_is_ellipsis_marker(ctx->ast, am_list_get(ctx->ast->alloc, lst, i + 1), in_quote)) {
            am_value_t ellip_template = child;

            am_value_t *pvars = NULL;
            size_t pvar_count = 0;
            if (macro_collect_pvars_in_value(macro, clause, ellip_template, &pvars, &pvar_count) != 0) {
                macro_set_error(ctx, L"out of memory collecting ellipsis pattern variables");
                return AM_VALUE_UNDEFINED;
            }
            if (pvar_count == 0) {
                macro_set_error(ctx, L"ellipsis template contains no pattern variables");
                free(pvars);
                return AM_VALUE_UNDEFINED;
            }

            // 所有模式变量的 ellipsis 列表长度必须相同
            size_t n = 0;
            int first = 1;
            for (size_t j = 0; j < pvar_count; j++) {
                am_value_t list_val = am_map_get(NULL, ctx->subst, pvars[j]);
                if (!am_value_is_handle(list_val)) {
                    macro_set_error(ctx, L"ellipsis pattern variable not bound to list");
                    free(pvars);
                    return AM_VALUE_UNDEFINED;
                }
                am_list_t *ellip_list = macro_list_from_handle(ctx->ast, am_value_to_handle(list_val));
                if (!ellip_list) {
                    macro_set_error(ctx, L"ellipsis binding is not a list");
                    free(pvars);
                    return AM_VALUE_UNDEFINED;
                }
                if (first) {
                    n = ellip_list->length;
                    first = 0;
                } else if (ellip_list->length != n) {
                    macro_set_error(ctx, L"ellipsis pattern variables have inconsistent lengths");
                    free(pvars);
                    return AM_VALUE_UNDEFINED;
                }
            }

            for (size_t j = 0; j < n; j++) {
                // 构造临时 subst：拷贝当前 subst 并覆盖 ellipsis 模式变量
                am_macro_expand_ctx_t iter_ctx = *ctx;
                iter_ctx.subst = am_map_copy(ctx->ast->alloc, ctx->subst);
                if (!iter_ctx.subst) {
                    macro_set_error(ctx, L"out of memory in ellipsis instantiation");
                    free(pvars);
                    return AM_VALUE_UNDEFINED;
                }
                int subst_ok = 1;
                for (size_t k = 0; k < pvar_count; k++) {
                    am_value_t list_val = am_map_get(NULL, ctx->subst, pvars[k]);
                    am_list_t *ellip_list = macro_list_from_handle(ctx->ast, am_value_to_handle(list_val));
                    am_value_t elem = am_list_get(ctx->ast->alloc, ellip_list, j);
                    am_map_t *m = am_map_set(ctx->ast->alloc, iter_ctx.subst, pvars[k], elem);
                    if (!m) { subst_ok = 0; break; }
                    iter_ctx.subst = m;
                }
                if (!subst_ok) {
                    am_map_destroy(ctx->ast->alloc, iter_ctx.subst);
                    macro_set_error(ctx, L"out of memory in ellipsis instantiation");
                    free(pvars);
                    return AM_VALUE_UNDEFINED;
                }
                am_value_t inst = macro_instantiate(&iter_ctx, macro, clause, ellip_template,
                                                     template_bindings, new_h);
                am_map_destroy(ctx->ast->alloc, iter_ctx.subst);
                if (iter_ctx.error) {
                    ctx->error = 1;
                    wcsncpy(ctx->error_msg, iter_ctx.error_msg, 256);
                    free(pvars);
                    return AM_VALUE_UNDEFINED;
                }
                if (macro_list_push(ctx->ast, new_h, inst, NULL) != 0) {
                    macro_set_error(ctx, L"failed to push ellipsis instantiation");
                    free(pvars);
                    return AM_VALUE_UNDEFINED;
                }
            }

            free(pvars);
            i += 2;
        } else {
            am_value_t inst = macro_instantiate(ctx, macro, clause, child, template_bindings, new_h);
            if (ctx->error) return AM_VALUE_UNDEFINED;
            if (macro_list_push(ctx->ast, new_h, inst, NULL) != 0) {
                macro_set_error(ctx, L"failed to push instantiated child");
                return AM_VALUE_UNDEFINED;
            }
            i += 1;
        }
    }

    return am_make_value_of_handle(new_h);
}


static am_value_t macro_instantiate(am_macro_expand_ctx_t *ctx, am_macro_t *macro,
                                     am_macro_clause_t *clause, am_value_t template,
                                     am_map_t *template_bindings, am_handle_t parent) {
    if (ctx->error) return AM_VALUE_UNDEFINED;

    // 模式变量：替换为使用处子树
    if (am_value_is_varid(template)) {
        am_varid_t vid = am_value_to_varid(template);
        if (macro_is_pattern_var(macro, clause, vid)) {
            // 注意：am_map 用 AM_VALUE_NULL 作为“不存在”的哨兵，
            // 因此不能通过返回值是否为 null 判断绑定是否存在。
            if (am_map_contains(NULL, ctx->subst, template) != 0) {
                macro_set_error(ctx, L"unbound pattern variable in template");
                return AM_VALUE_UNDEFINED;
            }
            am_value_t subst_val = am_map_get(NULL, ctx->subst, template);
            return macro_deep_copy_value(ctx, subst_val, parent);
        }

        // 模板内绑定：freshen
        if (am_map_contains(NULL, template_bindings, template) == 0) {
            am_value_t fresh_val = am_map_get(NULL, ctx->fresh_map, template);
            if (am_value_is_varid(fresh_val)) return fresh_val;
            am_varid_t fresh = macro_make_fresh_varid(ctx->ast, vid, ctx->expansion_id);
            if (fresh == SIZE_MAX) {
                macro_set_error(ctx, L"failed to make fresh variable");
                return AM_VALUE_UNDEFINED;
            }
            fresh_val = am_make_value_of_varid(fresh);
            am_map_t *m = am_map_set(ctx->ast->alloc, ctx->fresh_map, template, fresh_val);
            if (!m) {
                macro_set_error(ctx, L"out of memory in fresh map");
                return AM_VALUE_UNDEFINED;
            }
            ctx->fresh_map = m;
            return fresh_val;
        }

        // 自由标识符：保持 ARN 结果
        return template;
    }

    // 非 handle 立即数直接返回
    if (!am_value_is_handle(template)) {
        return template;
    }

    // handle：列表或字符串
    am_list_t *lst = macro_as_list(ctx->ast, template);
    if (!lst) {
        return macro_deep_copy_value(ctx, template, parent);
    }

    return macro_instantiate_list(ctx, macro, clause, lst, template_bindings, parent);
}


// ===============================================================================
// AST 展开
// ===============================================================================

static int macro_is_define_syntax(am_ast_t *ast, am_list_t *lst) {
    if (!lst || lst->type != AM_LIST_TYPE_APPLICATION || lst->length != 3) return 0;
    am_value_t first = am_list_get(ast->alloc, lst, 0);
    if (!macro_is_symbol_value(first, AM_VALUE_KW_define_syntax)) return 0;
    am_value_t second = am_list_get(ast->alloc, lst, 1);
    return am_value_is_varid(second);
}


static int macro_track_allocated_macro(am_macro_expand_ctx_t *ctx, am_macro_t *macro) {
    if (ctx->allocated_macro_count >= ctx->allocated_macro_capacity) {
        size_t new_cap = ctx->allocated_macro_capacity ? ctx->allocated_macro_capacity * 2 : 16;
        am_macro_t **new_arr = (am_macro_t **)realloc(ctx->allocated_macros, new_cap * sizeof(am_macro_t *));
        if (!new_arr) return -1;
        ctx->allocated_macros = new_arr;
        ctx->allocated_macro_capacity = new_cap;
    }
    ctx->allocated_macros[ctx->allocated_macro_count++] = macro;
    return 0;
}


static am_value_t macro_expand_value(am_macro_expand_ctx_t *ctx, am_value_t value,
                                      am_macro_env_frame_t *env, am_handle_t parent);


static int macro_expand_body_sequence(am_macro_expand_ctx_t *ctx, am_value_t *bodies, size_t n_body,
                                       am_macro_env_frame_t *env, am_handle_t parent,
                                       am_value_t **out_bodies, size_t *out_n_body) {
    // 第一趟：收集 define-syntax
    am_macro_env_frame_t *new_frame = macro_env_frame_create(ctx->ast);
    if (!new_frame) {
        macro_set_error(ctx, L"out of memory creating macro env frame");
        return -1;
    }
    new_frame->parent = env;

    for (size_t i = 0; i < n_body; i++) {
        am_value_t body = bodies[i];
        if (!am_value_is_handle(body)) continue;
        am_list_t *body_lst = macro_as_list(ctx->ast, body);
        if (!body_lst) continue;
        if (macro_is_define_syntax(ctx->ast, body_lst)) {
            am_value_t name_val = am_list_get(ctx->ast->alloc, body_lst, 1);
            am_value_t transformer_val = am_list_get(ctx->ast->alloc, body_lst, 2);
            if (!am_value_is_handle(transformer_val)) {
                macro_set_error(ctx, L"invalid define-syntax transformer");
                macro_env_frame_destroy(new_frame);
                return -1;
            }
            am_macro_t *macro = macro_parse_syntax_rules(ctx->ast, am_value_to_varid(name_val),
                                                          am_value_to_handle(transformer_val));
            if (!macro) {
                macro_set_error(ctx, L"failed to parse syntax-rules");
                macro_env_frame_destroy(new_frame);
                return -1;
            }
            if (macro_track_allocated_macro(ctx, macro) != 0) {
                macro_free_macro(macro);
                macro_env_frame_destroy(new_frame);
                macro_set_error(ctx, L"out of memory tracking macro");
                return -1;
            }
            if (macro_env_define(ctx->ast, new_frame, am_value_to_varid(name_val), macro) != 0) {
                macro_env_frame_destroy(new_frame);
                macro_set_error(ctx, L"out of memory defining macro");
                return -1;
            }
        }
    }

    // 第二趟：展开非 define-syntax 的 body
    am_value_t *new_bodies = (am_value_t *)malloc(n_body * sizeof(am_value_t));
    if (!new_bodies) {
        macro_env_frame_destroy(new_frame);
        macro_set_error(ctx, L"out of memory expanding body sequence");
        return -1;
    }
    size_t count = 0;
    for (size_t i = 0; i < n_body; i++) {
        am_value_t body = bodies[i];
        if (am_value_is_handle(body)) {
            am_list_t *body_lst = macro_as_list(ctx->ast, body);
            if (body_lst && macro_is_define_syntax(ctx->ast, body_lst)) {
                ctx->changed = 1;
                continue;
            }
        }
        am_value_t expanded = macro_expand_value(ctx, body, new_frame, parent);
        if (ctx->error) {
            free(new_bodies);
            macro_env_frame_destroy(new_frame);
            return -1;
        }
        new_bodies[count++] = expanded;
    }

    macro_env_frame_destroy(new_frame);
    *out_bodies = new_bodies;
    *out_n_body = count;
    return 0;
}


static am_value_t macro_expand_lambda(am_macro_expand_ctx_t *ctx, am_handle_t old_h, am_list_t *old_lst,
                                       am_macro_env_frame_t *env, am_handle_t parent) {
    size_t n_body = 0;
    am_value_t *bodies = am_list_lambda_get_bodies(ctx->ast->alloc, old_lst, &n_body);
    am_value_t *new_bodies = NULL;
    size_t new_n_body = 0;
    if (bodies) {
        if (macro_expand_body_sequence(ctx, bodies, n_body, env, old_h, &new_bodies, &new_n_body) != 0) {
            free(bodies);
            return AM_VALUE_UNDEFINED;
        }
    }

    // 如果 body 没有变化（数量与内容均相同），直接返回原 lambda，避免制造冗余节点
    int bodies_changed = 0;
    if (new_n_body != n_body) {
        bodies_changed = 1;
    } else if (bodies) {
        for (size_t i = 0; i < n_body; i++) {
            if (!am_value_equal(bodies[i], new_bodies[i])) {
                bodies_changed = 1;
                break;
            }
        }
    }

    if (!bodies_changed) {
        free(bodies);
        free(new_bodies);
        return am_make_value_of_handle(old_h);
    }

    am_handle_t new_h = am_ast_make_lambda_node(ctx->ast, parent);
    if (new_h == AM_HANDLE_NULL) {
        free(bodies);
        free(new_bodies);
        macro_set_error(ctx, L"out of memory expanding lambda");
        return AM_VALUE_UNDEFINED;
    }

    if (macro_register_lambda_scope(ctx, new_h, parent) != 0) {
        free(bodies);
        free(new_bodies);
        macro_set_error(ctx, L"failed to register expanded lambda scope");
        return AM_VALUE_UNDEFINED;
    }

    am_uint_t n_param = 0;
    if (old_lst->length >= 2) {
        am_value_t n_val = am_list_get(ctx->ast->alloc, old_lst, 1);
        if (am_value_is_uint(n_val)) n_param = am_value_to_uint(n_val);
    }
    for (size_t i = 0; i < (size_t)n_param; i++) {
        am_value_t p = am_list_get(ctx->ast->alloc, old_lst, 2 + i);
        if (macro_lambda_add_param(ctx->ast, new_h, am_value_to_varid(p)) != 0) {
            free(bodies);
            free(new_bodies);
            macro_set_error(ctx, L"failed to copy lambda param");
            return AM_VALUE_UNDEFINED;
        }
        if (macro_lambda_scope_add_var(ctx, new_h, am_value_to_varid(p)) != 0) {
            free(bodies);
            free(new_bodies);
            macro_set_error(ctx, L"failed to add expanded lambda param to scope");
            return AM_VALUE_UNDEFINED;
        }
    }

    if (new_bodies) {
        for (size_t i = 0; i < new_n_body; i++) {
            if (macro_lambda_add_body(ctx->ast, new_h, new_bodies[i], NULL) != 0) {
                free(bodies);
                free(new_bodies);
                macro_set_error(ctx, L"failed to add expanded lambda body");
                return AM_VALUE_UNDEFINED;
            }
        }
    }

    free(bodies);
    free(new_bodies);
    return am_make_value_of_handle(new_h);
}


static am_value_t macro_expand_slist(am_macro_expand_ctx_t *ctx, am_handle_t old_h, am_list_t *old_lst,
                                      am_macro_env_frame_t *env, am_handle_t parent) {
    am_value_t *expanded_children = (am_value_t *)malloc(old_lst->length * sizeof(am_value_t));
    if (!expanded_children) {
        macro_set_error(ctx, L"out of memory expanding slist");
        return AM_VALUE_UNDEFINED;
    }

    int any_changed = 0;
    for (size_t i = 0; i < old_lst->length; i++) {
        am_value_t child = am_list_get(ctx->ast->alloc, old_lst, i);
        am_value_t expanded = macro_expand_value(ctx, child, env, old_h);
        if (ctx->error) {
            free(expanded_children);
            return AM_VALUE_UNDEFINED;
        }
        expanded_children[i] = expanded;
        if (!any_changed && !am_value_equal(child, expanded)) {
            any_changed = 1;
        }
    }

    if (!any_changed) {
        free(expanded_children);
        return am_make_value_of_handle(old_h);
    }

    am_handle_t new_h = am_ast_make_slist_node(ctx->ast, parent, old_lst->type);
    if (new_h == AM_HANDLE_NULL) {
        free(expanded_children);
        macro_set_error(ctx, L"out of memory expanding slist");
        return AM_VALUE_UNDEFINED;
    }

    for (size_t i = 0; i < old_lst->length; i++) {
        if (macro_list_push(ctx->ast, new_h, expanded_children[i], NULL) != 0) {
            free(expanded_children);
            macro_set_error(ctx, L"failed to push expanded child");
            return AM_VALUE_UNDEFINED;
        }
    }
    free(expanded_children);

    return am_make_value_of_handle(new_h);
}


static am_value_t macro_expand_macro_use(am_macro_expand_ctx_t *ctx, am_handle_t use_h,
                                          am_macro_t *macro, am_macro_env_frame_t *env,
                                          am_handle_t parent) {
    ctx->parent = parent;
    am_value_t input = am_make_value_of_handle(use_h);

    for (size_t ci = 0; ci < macro->clause_count; ci++) {
        am_macro_clause_t *clause = &macro->clauses[ci];

        am_map_t *subst = am_map_create(ctx->ast->alloc, 16);
        if (!subst) {
            macro_set_error(ctx, L"out of memory creating macro subst");
            return AM_VALUE_UNDEFINED;
        }
        ctx->subst = subst;

        if (macro_match_value(ctx, macro, clause, clause->pattern, input) == 0) {
            ctx->changed = 1;
            // 收集模板内绑定
            am_map_t *template_bindings = am_map_create(ctx->ast->alloc, 16);
            if (!template_bindings) {
                am_map_destroy(ctx->ast->alloc, subst);
                ctx->subst = NULL;
                macro_set_error(ctx, L"out of memory creating template bindings");
                return AM_VALUE_UNDEFINED;
            }
            if (macro_collect_template_bindings_recursive(macro, clause, clause->template, &template_bindings) != 0) {
                am_map_destroy(ctx->ast->alloc, subst);
                am_map_destroy(ctx->ast->alloc, template_bindings);
                ctx->subst = NULL;
                macro_set_error(ctx, L"failed to collect template bindings");
                return AM_VALUE_UNDEFINED;
            }

            // 创建 fresh map
            am_map_t *fresh_map = am_map_create(ctx->ast->alloc, 16);
            if (!fresh_map) {
                am_map_destroy(ctx->ast->alloc, subst);
                am_map_destroy(ctx->ast->alloc, template_bindings);
                ctx->subst = NULL;
                macro_set_error(ctx, L"out of memory creating fresh map");
                return AM_VALUE_UNDEFINED;
            }
            ctx->fresh_map = fresh_map;
            macro->expansion_counter++;
            ctx->expansion_id = macro->expansion_counter;

            am_value_t inst = macro_instantiate(ctx, macro, clause, clause->template,
                                                 template_bindings, parent);

            am_map_destroy(ctx->ast->alloc, subst);
            am_map_destroy(ctx->ast->alloc, fresh_map);
            am_map_destroy(ctx->ast->alloc, template_bindings);
            ctx->subst = NULL;
            ctx->fresh_map = NULL;

            if (ctx->error) return AM_VALUE_UNDEFINED;

            // 递归展开实例化结果中的嵌套宏
            am_value_t expanded = macro_expand_value(ctx, inst, env, parent);
            return expanded;
        }

        am_map_destroy(ctx->ast->alloc, subst);
        ctx->subst = NULL;
    }

    macro_set_error(ctx, L"macro use did not match any clause");
    return AM_VALUE_UNDEFINED;
}


static am_value_t macro_expand_let_syntax(am_macro_expand_ctx_t *ctx, am_handle_t h, am_list_t *lst,
                                           am_macro_env_frame_t *env, am_handle_t parent, int isrec) {
    (void)isrec;
    (void)h;
    ctx->changed = 1;
    if (lst->length < 2) {
        macro_set_error(ctx, L"invalid let-syntax form");
        return AM_VALUE_UNDEFINED;
    }

    am_value_t bindings_val = am_list_get(ctx->ast->alloc, lst, 1);
    if (!am_value_is_handle(bindings_val)) {
        macro_set_error(ctx, L"invalid let-syntax bindings");
        return AM_VALUE_UNDEFINED;
    }
    am_list_t *bindings_lst = macro_list_from_handle(ctx->ast, am_value_to_handle(bindings_val));
    if (!bindings_lst || bindings_lst->type != AM_LIST_TYPE_APPLICATION) {
        macro_set_error(ctx, L"invalid let-syntax bindings list");
        return AM_VALUE_UNDEFINED;
    }

    am_macro_env_frame_t *new_frame = macro_env_frame_create(ctx->ast);
    if (!new_frame) {
        macro_set_error(ctx, L"out of memory creating let-syntax env");
        return AM_VALUE_UNDEFINED;
    }
    new_frame->parent = env;

    for (size_t i = 0; i < bindings_lst->length; i++) {
        am_value_t binding_val = am_list_get(ctx->ast->alloc, bindings_lst, i);
        if (!am_value_is_handle(binding_val)) {
            macro_set_error(ctx, L"invalid let-syntax binding");
            macro_env_frame_destroy(new_frame);
            return AM_VALUE_UNDEFINED;
        }
        am_list_t *binding_lst = macro_list_from_handle(ctx->ast, am_value_to_handle(binding_val));
        if (!binding_lst || binding_lst->type != AM_LIST_TYPE_APPLICATION || binding_lst->length != 2) {
            macro_set_error(ctx, L"invalid let-syntax binding form");
            macro_env_frame_destroy(new_frame);
            return AM_VALUE_UNDEFINED;
        }
        am_value_t name_val = am_list_get(ctx->ast->alloc, binding_lst, 0);
        am_value_t transformer_val = am_list_get(ctx->ast->alloc, binding_lst, 1);
        if (!am_value_is_varid(name_val) || !am_value_is_handle(transformer_val)) {
            macro_set_error(ctx, L"invalid let-syntax binding content");
            macro_env_frame_destroy(new_frame);
            return AM_VALUE_UNDEFINED;
        }
        am_macro_t *macro = macro_parse_syntax_rules(ctx->ast, am_value_to_varid(name_val),
                                                      am_value_to_handle(transformer_val));
        if (!macro) {
            macro_set_error(ctx, L"failed to parse let-syntax syntax-rules");
            macro_env_frame_destroy(new_frame);
            return AM_VALUE_UNDEFINED;
        }
        if (macro_track_allocated_macro(ctx, macro) != 0) {
            macro_free_macro(macro);
            macro_env_frame_destroy(new_frame);
            macro_set_error(ctx, L"out of memory tracking let-syntax macro");
            return AM_VALUE_UNDEFINED;
        }
        if (macro_env_define(ctx->ast, new_frame, am_value_to_varid(name_val), macro) != 0) {
            macro_env_frame_destroy(new_frame);
            macro_set_error(ctx, L"out of memory defining let-syntax macro");
            return AM_VALUE_UNDEFINED;
        }
    }

    size_t n_body = lst->length - 2;
    am_value_t *expanded_bodies = (am_value_t *)malloc(n_body * sizeof(am_value_t));
    if (!expanded_bodies) {
        macro_env_frame_destroy(new_frame);
        macro_set_error(ctx, L"out of memory expanding let-syntax bodies");
        return AM_VALUE_UNDEFINED;
    }
    size_t count = 0;
    for (size_t i = 2; i < lst->length; i++) {
        am_value_t body = am_list_get(ctx->ast->alloc, lst, i);
        am_value_t expanded = macro_expand_value(ctx, body, new_frame, parent);
        if (ctx->error) {
            free(expanded_bodies);
            macro_env_frame_destroy(new_frame);
            return AM_VALUE_UNDEFINED;
        }
        expanded_bodies[count++] = expanded;
    }
    macro_env_frame_destroy(new_frame);

    if (count == 1) {
        am_value_t result = expanded_bodies[0];
        free(expanded_bodies);
        return result;
    }

    // 多 body 时包装为 begin
    am_handle_t begin_h = am_ast_make_slist_node(ctx->ast, parent, AM_LIST_TYPE_APPLICATION);
    if (begin_h == AM_HANDLE_NULL) {
        free(expanded_bodies);
        macro_set_error(ctx, L"out of memory creating begin wrapper");
        return AM_VALUE_UNDEFINED;
    }
    if (macro_list_push(ctx->ast, begin_h, AM_VALUE_KW_begin, NULL) != 0) {
        free(expanded_bodies);
        macro_set_error(ctx, L"failed to push begin keyword");
        return AM_VALUE_UNDEFINED;
    }
    for (size_t i = 0; i < count; i++) {
        if (macro_list_push(ctx->ast, begin_h, expanded_bodies[i], NULL) != 0) {
            free(expanded_bodies);
            macro_set_error(ctx, L"failed to push begin body");
            return AM_VALUE_UNDEFINED;
        }
    }
    free(expanded_bodies);
    return am_make_value_of_handle(begin_h);
}


static am_value_t macro_expand_value(am_macro_expand_ctx_t *ctx, am_value_t value,
                                      am_macro_env_frame_t *env, am_handle_t parent) {
    if (ctx->error) return AM_VALUE_UNDEFINED;
    if (!am_value_is_handle(value)) return value;

    am_handle_t h = am_value_to_handle(value);
    am_list_t *lst = macro_as_list(ctx->ast, value);
    if (!lst) {
        // WString 或其他对象，无需展开
        return value;
    }

    // 宏使用
    if (lst->length > 0) {
        am_value_t first = am_list_get(ctx->ast->alloc, lst, 0);
        if (am_value_is_varid(first)) {
            am_macro_t *macro = macro_env_lookup(env, am_value_to_varid(first));
            if (macro) {
                return macro_expand_macro_use(ctx, h, macro, env, parent);
            }
        }
    }

    // let-syntax / letrec-syntax
    if (lst->length >= 2) {
        am_value_t first = am_list_get(ctx->ast->alloc, lst, 0);
        if (macro_is_symbol_value(first, AM_VALUE_KW_let_syntax)) {
            return macro_expand_let_syntax(ctx, h, lst, env, parent, 0);
        }
        if (macro_is_symbol_value(first, AM_VALUE_KW_letrec_syntax)) {
            return macro_expand_let_syntax(ctx, h, lst, env, parent, 1);
        }
    }

    // quote / quasiquote / unquote 内部不展开宏，避免用户 symbol 与关键字冲突
    if (lst->type == AM_LIST_TYPE_QUOTE ||
        lst->type == AM_LIST_TYPE_QUASIQUOTE ||
        lst->type == AM_LIST_TYPE_UNQUOTE) {
        return value;
    }

    // 普通列表：深拷贝并递归展开子元素
    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        return macro_expand_lambda(ctx, h, lst, env, parent);
    }
    return macro_expand_slist(ctx, h, lst, env, parent);
}


// ===============================================================================
// 展开后元数据刷新
// ===============================================================================

// 从顶层节点做可达性遍历，递归收集所有 lambda 节点把柄。
// 宏展开后 nodes 堆中仍残留被替换掉的旧节点（死节点），因此不能遍历全堆，
// 否则死 lambda 会被重新编译进 IL，并引用已被消除的变量（如宏关键字）。
static void macro_collect_lambda_walk(am_ast_t *ast, am_value_t item) {
    if (!am_value_is_handle(item)) return;
    am_handle_t handle = am_value_to_handle(item);
    am_value_t node_val = am_ast_get_node(ast, handle);
    if (!am_value_is_ptr(node_val)) return;
    am_object_t *obj = am_value_to_ptr(node_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;
    am_list_t *lst = (am_list_t *)obj;

    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        // 同一节点可能被多处引用，避免重复登记
        if (am_list_find(ast->alloc, ast->lambda_handles, item, 0) == SIZE_MAX) {
            am_list_t *new_lst = am_list_push(ast->alloc, ast->lambda_handles, item);
            if (new_lst) {
                ast->lambda_handles = new_lst;
            }
        }
    }
    for (size_t i = 0; i < lst->length; i++) {
        macro_collect_lambda_walk(ast, am_list_get(ast->alloc, lst, i));
    }
}


static void macro_rebuild_lambda_handles(am_ast_t *ast) {
    if (!ast || !ast->lambda_handles) return;
    ast->lambda_handles->length = 0;
    am_handle_t top = am_ast_get_top_node_handle(ast);
    if (top == AM_HANDLE_NULL) return;
    macro_collect_lambda_walk(ast, am_make_value_of_handle(top));
}


static void macro_rebuild_var_top(am_ast_t *ast) {
    if (!ast || !ast->var_top) return;
    ast->var_top->length = 0;

    am_handle_t top_lambda = am_ast_get_top_lambda_node_handle(ast);
    if (top_lambda == AM_HANDLE_NULL) return;

    am_value_t lambda_val = am_ast_get_node(ast, top_lambda);
    if (!am_value_is_ptr(lambda_val)) return;
    am_list_t *lambda = (am_list_t *)am_value_to_ptr(lambda_val);
    size_t n_body = am_list_lambda_get_body_number(ast->alloc, lambda);
    am_value_t *bodies = am_list_lambda_get_bodies(ast->alloc, lambda, &n_body);
    if (!bodies) return;

    for (size_t i = 0; i < n_body; i++) {
        am_value_t body = bodies[i];
        if (!am_value_is_handle(body)) continue;
        am_value_t node_val = am_ast_get_node(ast, am_value_to_handle(body));
        if (!am_value_is_ptr(node_val)) continue;
        am_object_t *obj = am_value_to_ptr(node_val);
        if (obj->type != AM_OBJECT_TYPE_LIST) continue;
        am_list_t *lst = (am_list_t *)obj;
        if (lst->type != AM_LIST_TYPE_APPLICATION || lst->length < 2) continue;
        am_value_t first = am_list_get(ast->alloc, lst, 0);
        if (!macro_is_symbol_value(first, AM_VALUE_KW_define)) continue;
        am_value_t second = am_list_get(ast->alloc, lst, 1);
        if (!am_value_is_varid(second)) continue;
        am_ast_add_var_top(ast, am_value_to_varid(second));
    }

    free(bodies);
}


// ===============================================================================
// 入口函数
// ===============================================================================

static int macro_is_any_macro_keyword(am_value_t v) {
    return macro_is_symbol_value(v, AM_VALUE_KW_define_syntax) ||
           macro_is_symbol_value(v, AM_VALUE_KW_let_syntax) ||
           macro_is_symbol_value(v, AM_VALUE_KW_letrec_syntax) ||
           macro_is_symbol_value(v, AM_VALUE_KW_syntax_rules);
}


typedef struct {
    am_ast_t *ast;
    int       found;
} macro_fast_path_scan_t;


static void macro_fast_path_scan_cb(am_handle_t handle, am_value_t value, void *user_data) {
    macro_fast_path_scan_t *data = (macro_fast_path_scan_t *)user_data;
    if (data->found) return;
    (void)handle;

    if (!am_value_is_ptr(value)) return;
    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->length == 0) return;

    am_value_t first = am_list_get(data->ast->alloc, lst, 0);
    if (macro_is_any_macro_keyword(first)) {
        data->found = 1;
    }
}


int32_t am_macro_expand(am_ast_t *ast) {
    if (!ast) return -1;

    // 快速路径：扫描 AST 堆中是否出现任何宏关键字。
    // 若整个 AST 都不含 define-syntax / let-syntax / letrec-syntax / syntax-rules，
    // 则无需进行递归宏展开，直接返回成功。
    macro_fast_path_scan_t scan = { ast, 0 };
    am_heap_iter(ast->alloc, ast->alloc, ast->nodes, macro_fast_path_scan_cb, &scan);
    if (!scan.found) return 0;

    am_macro_expand_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.ast = ast;

    am_handle_t top_lambda = am_ast_get_top_lambda_node_handle(ast);
    if (top_lambda == AM_HANDLE_NULL) {
        macro_set_error(&ctx, L"failed to get top lambda handle");
        fprintf(stderr, "[Macro Error] %ls\n", ctx.error_msg);
        return -1;
    }

    am_value_t lambda_val = am_ast_get_node(ast, top_lambda);
    if (!am_value_is_ptr(lambda_val)) {
        macro_set_error(&ctx, L"top lambda node is not a list");
        fprintf(stderr, "[Macro Error] %ls\n", ctx.error_msg);
        return -1;
    }
    am_list_t *lambda = (am_list_t *)am_value_to_ptr(lambda_val);

    size_t n_body = am_list_lambda_get_body_number(ast->alloc, lambda);
    am_value_t *bodies = am_list_lambda_get_bodies(ast->alloc, lambda, &n_body);
    if (!bodies) return 0;

    am_value_t *new_bodies = NULL;
    size_t new_n_body = 0;
    int ok = macro_expand_body_sequence(&ctx, bodies, n_body, NULL, top_lambda,
                                         &new_bodies, &new_n_body);
    free(bodies);

    if (ok != 0 || ctx.error) {
        free(new_bodies);
        for (size_t i = 0; i < ctx.allocated_macro_count; i++) {
            macro_free_macro(ctx.allocated_macros[i]);
        }
        free(ctx.allocated_macros);
        if (ctx.error) {
            fprintf(stderr, "[Macro Error] %ls\n", ctx.error_msg);
        }
        return -1;
    }

    // 若实际发生过宏展开或 define-syntax 消除，才替换顶层节点并刷新元数据。
    // 无宏时直接复用原 AST，避免制造冗余节点与重复元数据。
    if (ctx.changed) {
        if (am_ast_set_global_nodes(ast, new_bodies, new_n_body) != 0) {
            free(new_bodies);
            for (size_t i = 0; i < ctx.allocated_macro_count; i++) {
                macro_free_macro(ctx.allocated_macros[i]);
            }
            free(ctx.allocated_macros);
            macro_set_error(&ctx, L"failed to set global nodes");
            fprintf(stderr, "[Macro Error] %ls\n", ctx.error_msg);
            return -1;
        }
        free(new_bodies);

        // 刷新元数据（tailcall_handles 由 am_parse / am_link 的尾位置分析统一重建）
        macro_rebuild_lambda_handles(ast);
        macro_rebuild_var_top(ast);
    } else {
        free(new_bodies);
    }

    // 释放宏描述符
    for (size_t i = 0; i < ctx.allocated_macro_count; i++) {
        macro_free_macro(ctx.allocated_macros[i]);
    }
    free(ctx.allocated_macros);

    return 0;
}
/* ===== end:   src/am_macro.c ===== */

/* ===== begin: src/am_linker.c ===== */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>



// 链接器处理的模块数上限
#define AM_LINKER_MAX_MODULES (32)


// 链接器上下文
struct am_linker_ctx_t {
    am_allocator_t *alloc;           // AST 使用的分配器
    am_ast_t *main_ast;              // 引用根模块，由调用者管理生命周期
    am_vocab_t *all_module_path;     // mod_index -> module_path
    am_ast_t **ALLAST;               // mod_index -> ast
    wchar_t **codes;                 // mod_index -> code（由 linker 读取，需释放）
    wchar_t **paths;                 // mod_index -> absolute_path（由 linker 分配，需释放）
    size_t (*DAG)[2];                // 邻接关系列表 importer_index -> importee_index
    size_t edge_num;                 // 当前边数
    size_t module_counter;           // 当前模块数
    wchar_t *base_dir;               // 基准工作目录
    am_linker_read_source_fn read_source;  // 模块源码读取回调（由调用方注入）
    void *read_source_user_data;     // 透传给 read_source 的上下文指针
};


// 宽字符串复制（使用 allocator 在内存池上分配）
static wchar_t *linker_wcsdup(am_allocator_t *alloc, const wchar_t *s) {
    if (!s) return NULL;
    size_t len = wcslen(s);
    wchar_t *dup = (wchar_t *)am_malloc(alloc, (len + 1) * sizeof(wchar_t));
    if (!dup) return NULL;
    wcscpy(dup, s);
    return dup;
}


// 解析 import 路径为绝对路径。
// 绝对路径（以 '/' 开头）直接返回；相对路径与 base_dir 拼接。
// 返回 allocator 分配的字符串；失败返回 NULL。
static wchar_t *linker_resolve_path(am_allocator_t *alloc, const wchar_t *base_dir, const wchar_t *path) {
    if (!path) return NULL;
    if (path[0] == L'/') {
        return linker_wcsdup(alloc, path);
    }

    size_t base_len = base_dir ? wcslen(base_dir) : 0;
    size_t path_len = wcslen(path);

    if (base_len == 0) {
        return linker_wcsdup(alloc, path);
    }

    size_t total = base_len + 1 + path_len + 1;
    wchar_t *result = (wchar_t *)am_malloc(alloc, total * sizeof(wchar_t));
    if (!result) return NULL;

    int n = swprintf(result, total, L"%ls/%ls", base_dir, path);
    if (n <= 0 || (size_t)n >= total) {
        am_free(alloc, result);
        return NULL;
    }
    return result;
}


// 从 WString 节点中提取 import 路径。
// 返回 allocator 分配的字符串；失败返回 NULL。
static wchar_t *linker_extract_path_from_wstring(am_allocator_t *alloc, am_wstring_t *ws) {
    if (!ws || ws->length <= 0) return NULL;

    size_t len = ws->length;
    wchar_t *buf = (wchar_t *)am_malloc(alloc, (len + 1) * sizeof(wchar_t));
    if (!buf) return NULL;

    for (size_t i = 0; i < len; i++) {
        buf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
    }
    buf[len] = L'\0';
    return buf;
}


// 前向声明
static int32_t import_analysis(am_linker_ctx_t *ctx, wchar_t *importee_path, size_t importer_index);


typedef struct {
    am_linker_ctx_t *ctx;
    am_ast_t *current_ast;
    size_t current_module_index;
    int32_t error;
} dep_iter_t;


static void import_analysis_dep_iter_cb(am_value_t key, am_value_t value, void *user_data) {
    (void)key;
    dep_iter_t *it = (dep_iter_t *)user_data;
    if (it->error) return;
    if (!am_value_is_handle(value)) return;

    am_value_t path_node_val = am_ast_get_node(it->current_ast, am_value_to_handle(value));
    if (!am_value_is_ptr(path_node_val)) return;

    am_object_t *obj = am_value_to_ptr(path_node_val);
    if (obj->type != AM_OBJECT_TYPE_WSTRING) return;

    wchar_t *dep_path = linker_extract_path_from_wstring(it->ctx->alloc, (am_wstring_t *)obj);
    if (!dep_path) {
        it->error = -1;
        return;
    }

    wchar_t *abs_dep_path = linker_resolve_path(it->ctx->alloc, it->ctx->base_dir, dep_path);
    am_free(it->ctx->alloc, dep_path);
    if (!abs_dep_path) {
        it->error = -1;
        return;
    }

    // printf("Importee absolute path = %ls\n", abs_dep_path);

    int32_t res = import_analysis(it->ctx, abs_dep_path, it->current_module_index);
    am_free(it->ctx->alloc, abs_dep_path);
    if (res < 0) {
        it->error = -1;
    }
}


// 创建链接器上下文。成功返回指针；失败返回 NULL。
static am_linker_ctx_t *linker_ctx_create(am_allocator_t *alloc, am_ast_t *main_ast, wchar_t *base_dir,
                                          am_linker_read_source_fn read_source, void *user_data) {
    if (!alloc || !main_ast || !read_source) return NULL;

    am_linker_ctx_t *ctx = (am_linker_ctx_t *)am_malloc(alloc, sizeof(am_linker_ctx_t));
    if (!ctx) return NULL;

    ctx->alloc = alloc;
    ctx->main_ast = main_ast;
    ctx->base_dir = base_dir;
    ctx->read_source = read_source;
    ctx->read_source_user_data = user_data;
    ctx->edge_num = 0;
    ctx->module_counter = 0;

    ctx->all_module_path = am_vocab_create(alloc, AM_LINKER_MAX_MODULES);
    ctx->ALLAST = (am_ast_t **)am_calloc(alloc, AM_LINKER_MAX_MODULES * sizeof(am_ast_t *));
    ctx->codes = (wchar_t **)am_calloc(alloc, AM_LINKER_MAX_MODULES * sizeof(wchar_t *));
    ctx->paths = (wchar_t **)am_calloc(alloc, AM_LINKER_MAX_MODULES * sizeof(wchar_t *));
    ctx->DAG = (size_t (*)[2])am_calloc(alloc, AM_LINKER_MAX_MODULES * AM_LINKER_MAX_MODULES * sizeof(size_t [2]));

    if (!ctx->all_module_path || !ctx->ALLAST || !ctx->codes || !ctx->paths || !ctx->DAG) {
        am_free(alloc, ctx->ALLAST);
        am_free(alloc, ctx->codes);
        am_free(alloc, ctx->paths);
        am_free(alloc, ctx->DAG);
        if (ctx->all_module_path) am_vocab_destroy(alloc, ctx->all_module_path);
        am_free(alloc, ctx);
        return NULL;
    }

    return ctx;
}


// 销毁链接器上下文。
// 注意：main_ast 及其 code/absolute_path 由调用者管理，此处不释放。
static void linker_ctx_destroy(am_linker_ctx_t *ctx) {
    if (!ctx) return;

    if (ctx->ALLAST && ctx->codes && ctx->paths) {
        for (size_t i = 0; i < ctx->module_counter; i++) {
            if (ctx->ALLAST[i] && ctx->ALLAST[i] != ctx->main_ast) {
                am_ast_destroy(ctx->ALLAST[i]);
            }
            if (ctx->codes[i]) am_free(ctx->alloc, ctx->codes[i]);
            if (ctx->paths[i]) am_free(ctx->alloc, ctx->paths[i]);
        }
    }

    if (ctx->all_module_path) am_vocab_destroy(ctx->alloc, ctx->all_module_path);
    am_free(ctx->alloc, ctx->ALLAST);
    am_free(ctx->alloc, ctx->codes);
    am_free(ctx->alloc, ctx->paths);
    am_free(ctx->alloc, ctx->DAG);
    am_free(ctx->alloc, ctx);
}


// 递归解析依赖模块。
// importee_path 为当前要解析模块的绝对路径；importer_index 为引用它的模块索引，SIZE_MAX 表示无引用者（根模块）。
static int32_t import_analysis(am_linker_ctx_t *ctx, wchar_t *importee_path, size_t importer_index) {
    if (!ctx || !importee_path) return -1;

    size_t current_module_index = am_vocab_find(ctx->alloc, ctx->all_module_path, importee_path);

    if (current_module_index == SIZE_MAX) {
        if (ctx->module_counter >= AM_LINKER_MAX_MODULES) return -1;
        current_module_index = ctx->module_counter;

        size_t inserted;
        ctx->all_module_path = am_vocab_insert(ctx->alloc, ctx->all_module_path, importee_path, &inserted);
        if (!ctx->all_module_path || inserted == SIZE_MAX || inserted != current_module_index) return -1;

        int is_main = (ctx->main_ast->absolute_path != NULL) &&
                      (wcscmp(importee_path, ctx->main_ast->absolute_path) == 0);

        am_ast_t *current_ast = NULL;

        if (is_main) {
            // 引用根模块直接使用调用者传入的 AST
            current_ast = ctx->main_ast;
            ctx->ALLAST[current_module_index] = current_ast;
            ctx->codes[current_module_index] = NULL;
            ctx->paths[current_module_index] = NULL;
        }
        else {
            wchar_t *path_copy = linker_wcsdup(ctx->alloc, importee_path);
            if (!path_copy) return -1;

            // 通过调用方注入的回调获取模块源码（缓冲区由 ctx->alloc 分配）
            wchar_t *raw_code = ctx->read_source(ctx->alloc, importee_path, ctx->read_source_user_data);
            if (!raw_code) {
                am_free(ctx->alloc, path_copy);
                return -1;
            }

            // 模块源码需包装为 ((lambda () <file_content> )) 形式
            const wchar_t *prefix = L"((lambda () ";
            const wchar_t *suffix = L" ))";
            size_t raw_len = wcslen(raw_code);
            size_t prefix_len = wcslen(prefix);
            size_t suffix_len = wcslen(suffix);
            size_t code_len = prefix_len + raw_len + suffix_len;

            wchar_t *code = (wchar_t *)am_malloc(ctx->alloc, (code_len + 1) * sizeof(wchar_t));
            if (!code) {
                am_free(ctx->alloc, raw_code);
                am_free(ctx->alloc, path_copy);
                return -1;
            }
            size_t pos = 0;
            for (size_t i = 0; i < prefix_len; i++) code[pos++] = prefix[i];
            for (size_t i = 0; i < raw_len; i++) code[pos++] = raw_code[i];
            for (size_t i = 0; i < suffix_len; i++) code[pos++] = suffix[i];
            code[pos] = L'\0';
            am_free(ctx->alloc, raw_code);

            current_ast = am_parse(ctx->alloc, code, path_copy, 0);
            if (!current_ast) {
                am_free(ctx->alloc, code);
                am_free(ctx->alloc, path_copy);
                return -1;
            }

            ctx->ALLAST[current_module_index] = current_ast;
            ctx->codes[current_module_index] = code;
            ctx->paths[current_module_index] = path_copy;
        }

        ctx->module_counter++;

        // 递归处理当前模块的依赖
        dep_iter_t dep_it = { ctx, current_ast, current_module_index, 0 };
        am_map_iter(current_ast->alloc, current_ast->dependencies, import_analysis_dep_iter_cb, &dep_it);
        if (dep_it.error != 0) {
            return -1;
        }
    }

    if (importer_index != SIZE_MAX) {
        if (ctx->edge_num >= AM_LINKER_MAX_MODULES * AM_LINKER_MAX_MODULES) return -1;
        ctx->DAG[ctx->edge_num][0] = importer_index;
        ctx->DAG[ctx->edge_num][1] = current_module_index;
        ctx->edge_num++;
    }

    return 0;
}


// ===============================================================================
// 拓扑排序
// ===============================================================================

// 向邻接表中追加一个邻接节点。成功返回0，失败返回-1。
static int32_t topo_sort_push_adj(am_allocator_t *alloc, size_t **adj, size_t *len, size_t *cap, size_t node) {
    if (*len >= *cap) {
        size_t new_cap = *cap ? *cap * 2 : 4;
        size_t *new_adj = (size_t *)am_realloc(alloc, *adj, new_cap * sizeof(size_t));
        if (!new_adj) return -1;
        *adj = new_adj;
        *cap = new_cap;
    }
    (*adj)[(*len)++] = node;
    return 0;
}


// 功能描述：对DAG进行拓扑排序。
// 参数说明：DAG[i] = {出节点索引, 入节点索引}，表示一条从出节点指向入节点的有向边。
//          edge_num 为边的数量。
// 返回值：  成功返回排序后的节点索引数组（由调用者使用 alloc 释放），长度等于节点总数（最大索引+1）。
//          失败（如检测到环或内存分配失败）返回 (size_t *)SIZE_MAX。
// 算法：    Kahn算法（基于入度的BFS拓扑排序）。
static size_t *linker_topo_sort(am_allocator_t *alloc, size_t DAG[][2], size_t edge_num) {
    if (!DAG && edge_num > 0) return (size_t *)SIZE_MAX;

    // 计算节点总数：取所有边中最大索引 + 1
    size_t node_count = 0;
    for (size_t i = 0; i < edge_num; i++) {
        if (DAG[i][0] >= node_count) node_count = DAG[i][0] + 1;
        if (DAG[i][1] >= node_count) node_count = DAG[i][1] + 1;
    }

    size_t *in_degree = (size_t *)am_calloc(alloc, node_count * sizeof(size_t));
    size_t *adj_len = (size_t *)am_calloc(alloc, node_count * sizeof(size_t));
    size_t *adj_cap = (size_t *)am_calloc(alloc, node_count * sizeof(size_t));
    size_t **adj = (size_t **)am_calloc(alloc, node_count * sizeof(size_t *));

    if ((!in_degree || !adj_len || !adj_cap || !adj) && node_count > 0) {
        am_free(alloc, in_degree);
        am_free(alloc, adj_len);
        am_free(alloc, adj_cap);
        am_free(alloc, adj);
        return (size_t *)SIZE_MAX;
    }

    // 构建邻接表和入度数组
    for (size_t i = 0; i < edge_num; i++) {
        size_t out = DAG[i][0];
        size_t in = DAG[i][1];
        in_degree[in]++;
        if (topo_sort_push_adj(alloc, &adj[out], &adj_len[out], &adj_cap[out], in) < 0) {
            for (size_t j = 0; j < node_count; j++) am_free(alloc, adj[j]);
            am_free(alloc, in_degree);
            am_free(alloc, adj_len);
            am_free(alloc, adj_cap);
            am_free(alloc, adj);
            return (size_t *)SIZE_MAX;
        }
    }

    size_t *result = (size_t *)am_malloc(alloc, node_count * sizeof(size_t));
    size_t *queue = (size_t *)am_malloc(alloc, node_count * sizeof(size_t));

    if ((!result || !queue) && node_count > 0) {
        am_free(alloc, result);
        am_free(alloc, queue);
        for (size_t j = 0; j < node_count; j++) am_free(alloc, adj[j]);
        am_free(alloc, in_degree);
        am_free(alloc, adj_len);
        am_free(alloc, adj_cap);
        am_free(alloc, adj);
        return (size_t *)SIZE_MAX;
    }

    // Kahn算法：将入度为0的节点入队
    size_t front = 0, rear = 0;
    for (size_t i = 0; i < node_count; i++) {
        if (in_degree[i] == 0) {
            queue[rear++] = i;
        }
    }

    // 依次取出节点，并将其邻接节点入度减1
    size_t result_idx = 0;
    while (front < rear) {
        size_t node = queue[front++];
        result[result_idx++] = node;

        for (size_t i = 0; i < adj_len[node]; i++) {
            size_t neighbor = adj[node][i];
            if (--in_degree[neighbor] == 0) {
                queue[rear++] = neighbor;
            }
        }
    }

    for (size_t j = 0; j < node_count; j++) am_free(alloc, adj[j]);
    am_free(alloc, in_degree);
    am_free(alloc, adj_len);
    am_free(alloc, adj_cap);
    am_free(alloc, adj);
    am_free(alloc, queue);

    // 若结果节点数不足，说明图中存在环
    if (result_idx != node_count) {
        am_free(alloc, result);
        return (size_t *)SIZE_MAX;
    }

    return result;
}


// ===============================================================================
// 外部引用解析
// ===============================================================================

typedef struct {
    am_ast_t *ast;
    wchar_t *base_dir;
    int32_t   error;
} import_ref_resolution_ctx_t;


static void import_ref_resolution_iter_cb(am_handle_t handle, am_value_t value, void *user_data) {
    (void)handle;

    import_ref_resolution_ctx_t *ctx = (import_ref_resolution_ctx_t *)user_data;
    am_ast_t *ast = ctx->ast;

    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return;

    am_list_t *lst = (am_list_t *)obj;

    for (size_t i = 0; i < lst->length; i++) {
        am_value_t child = am_list_get(ast->alloc, lst, i);
        if (!am_value_is_varid(child)) continue;

        am_varid_t varid = am_value_to_varid(child);
        if ((size_t)varid >= ast->var_type->length) {
            ctx->error = -1;
            return;
        }

        am_value_t type_val = am_list_get(ast->alloc, ast->var_type, (size_t)varid);
        if (!am_value_is_uint(type_val)) continue;
        if (am_value_to_uint(type_val) != AM_VAR_TYPE_IMPORT_REF) continue;

        // 从最后一个点号分割 prefix 与 suffix
        wchar_t *var_str = am_vocab_get(ast->alloc, ast->var_vocab, &varid);
        if (!var_str) {
            ctx->error = -1;
            return;
        }

        wchar_t *last_dot = wcsrchr(var_str, L'.');
        if (!last_dot || last_dot == var_str) {
            ctx->error = -1;
            return;
        }

        size_t prefix_len = (size_t)(last_dot - var_str);
        wchar_t *prefix = (wchar_t *)am_malloc(ast->alloc, (prefix_len + 1) * sizeof(wchar_t));
        if (!prefix) {
            ctx->error = -1;
            return;
        }
        wcsncpy(prefix, var_str, prefix_len);
        prefix[prefix_len] = L'\0';

        const wchar_t *suffix = last_dot + 1;

        // 查询 dependencies，找到 alias 对应的 importee 路径
        size_t alias_varid = am_vocab_find(ast->alloc, ast->var_vocab, prefix);
        am_free(ast->alloc, prefix);
        if (alias_varid == SIZE_MAX) {
            ctx->error = -1;
            return;
        }

        am_value_t path_h_val = am_map_get(ast->alloc, ast->dependencies,
                                           am_make_value_of_varid((am_varid_t)alias_varid));
        if (!am_value_is_handle(path_h_val)) {
            ctx->error = -1;
            return;
        }

        am_value_t path_node_val = am_ast_get_node(ast, am_value_to_handle(path_h_val));
        if (!am_value_is_ptr(path_node_val)) {
            ctx->error = -1;
            return;
        }

        am_object_t *path_obj = am_value_to_ptr(path_node_val);
        if (path_obj->type != AM_OBJECT_TYPE_WSTRING) {
            ctx->error = -1;
            return;
        }

        wchar_t *importee_path = linker_extract_path_from_wstring(ast->alloc, (am_wstring_t *)path_obj);
        if (!importee_path) {
            ctx->error = -1;
            return;
        }

        // 转换为绝对路径
        wchar_t *abs_importee_path = linker_resolve_path(ast->alloc, ctx->base_dir, importee_path);
        if (!abs_importee_path) {
            ctx->error = -1;
            am_free(ast->alloc, importee_path);
            return;
        }

        wchar_t *importee_id = am_absolute_path_to_module_id(ast->alloc, abs_importee_path);
        am_free(ast->alloc, importee_path);
        am_free(ast->alloc, abs_importee_path);
        if (!importee_id) {
            ctx->error = -1;
            return;
        }

        size_t id_len = wcslen(importee_id);
        size_t suffix_len = wcslen(suffix);
        am_varid_t resolved_varid = SIZE_MAX;
        size_t match_count = 0;

        // 在 var_top 中匹配 importee_id.<lambda_handle>.suffix
        for (size_t k = 0; k < ast->var_top->length; k++) {
            am_value_t top_val = am_list_get(ast->alloc, ast->var_top, k);
            if (!am_value_is_varid(top_val)) continue;

            am_varid_t top_varid = am_value_to_varid(top_val);
            wchar_t *top_name = am_vocab_get(ast->alloc, ast->var_vocab, &top_varid);
            if (!top_name) continue;

            size_t top_len = wcslen(top_name);
            if (top_len > id_len + 1 + suffix_len + 1 &&
                wcsncmp(top_name, importee_id, id_len) == 0 &&
                top_name[id_len] == L'.' &&
                wcscmp(top_name + top_len - suffix_len, suffix) == 0 &&
                top_name[top_len - suffix_len - 1] == L'.' &&
                (top_len - suffix_len - 1) > id_len) {
                resolved_varid = top_varid;
                match_count++;
            }
        }

        am_free(ast->alloc, importee_id);

        if (match_count != 1) {
            ctx->error = -1;
            return;
        }

        if (am_list_set(ast->alloc, lst, i,
                        am_make_value_of_varid(resolved_varid)) != 0) {
            ctx->error = -1;
            return;
        }
    }
}


// 对合并后的AST执行外部引用解析，也就是将AST中所有的 var_type=AM_VAR_TYPE_IMPORT_REF 类型的变量，
// 替换为 dependencies 对应模块中的变量全限定名。
// 成功返回 0，失败返回 -1。
int32_t am_linker_import_ref_resolution(am_ast_t *merged_ast, wchar_t *base_dir) {

    if (!merged_ast || !merged_ast->alloc || !merged_ast->nodes ||
        !merged_ast->var_vocab || !merged_ast->var_type || !merged_ast->var_top ||
        !merged_ast->dependencies) {
        return -1;
    }

    import_ref_resolution_ctx_t iter_ctx = { merged_ast, base_dir, 0 };
    am_heap_iter(merged_ast->alloc, merged_ast->alloc, merged_ast->nodes,
                 import_ref_resolution_iter_cb, &iter_ctx);

    return iter_ctx.error;
}


// ===============================================================================
// 静态对象标记
// ===============================================================================

typedef struct {
    int32_t error;
} mark_static_ctx_t;


static void mark_node_static_cb(am_handle_t handle, am_value_t value, void *user_data) {
    (void)handle;
    mark_static_ctx_t *ms = (mark_static_ctx_t *)user_data;
    if (ms->error) return;
    if (!am_value_is_ptr(value)) return;

    am_object_t *obj = am_value_to_ptr(value);
    if (am_object_set_static(obj, 0) != 0) {
        ms->error = -1;
    }
}


// 将 AST->nodes 中的所有对象标记为 static。
// 链接阶段的所有对象均为静态对象（永生对象）。
// 成功返回 0，失败返回 -1。
static int32_t linker_mark_all_nodes_static(am_ast_t *ast) {
    if (!ast || !ast->alloc || !ast->nodes) return -1;
    mark_static_ctx_t ms = { 0 };
    am_heap_iter(ast->alloc, ast->alloc, ast->nodes, mark_node_static_cb, &ms);
    return ms.error;
}


// ===============================================================================
// 链接器入口
// ===============================================================================

// 功能描述：链接器入口。从 main_ast 出发，递归解析所有依赖模块，按拓扑顺序合并成一个大 AST。
// 参数说明：main_ast 为引用根模块的 AST；base_dir 为基准工作目录（用于解析相对路径 import）；
//          read_source 为模块源码读取回调（不可为 NULL）；user_data 透传给 read_source。
// 返回值：  成功返回链接后的 AST（即基于 main_ast 修改后的 AST）；失败返回 NULL。
am_ast_t *am_link(am_ast_t *main_ast, wchar_t *base_dir,
                  am_linker_read_source_fn read_source, void *user_data) {
    if (!main_ast || !main_ast->alloc || !main_ast->absolute_path || !read_source) return NULL;

    am_linker_ctx_t *ctx = linker_ctx_create(main_ast->alloc, main_ast, base_dir,
                                             read_source, user_data);
    if (!ctx) return NULL;

    // 递归解析所有依赖模块
    if (import_analysis(ctx, main_ast->absolute_path, SIZE_MAX) != 0) {
        linker_ctx_destroy(ctx);
        return NULL;
    }

    if (ctx->module_counter == 0 || ctx->ALLAST[0] != main_ast) {
        linker_ctx_destroy(ctx);
        return NULL;
    }

    am_ast_t *global_ast = NULL;

    // 只有一个模块时无需拓扑排序与合并
    if (ctx->module_counter == 1) {
        global_ast = main_ast;
    }
    else {
        // 对 DAG 做拓扑排序，同时检查是否成环
        size_t *sorted = linker_topo_sort(ctx->alloc, ctx->DAG, ctx->edge_num);
        if (sorted == (size_t *)SIZE_MAX) {
            linker_ctx_destroy(ctx);
            return NULL;
        }
        // 以排序后的第一个模块为全局 importer，逐个吃掉 importee
        global_ast = ctx->ALLAST[sorted[0]];
        for (size_t i = 1; i < ctx->module_counter; i++) {
            size_t importee_index = sorted[i];
            if (am_ast_merge(global_ast, ctx->ALLAST[importee_index], 0) != 0) {
                am_free(ctx->alloc, sorted);
                linker_ctx_destroy(ctx);
                return NULL;
            }
        }
        am_free(ctx->alloc, sorted);
    }

    // 模块合并会改变 AST 结构，需要重新进行整体的尾位置分析
    if (am_parser_tail_call_analysis(global_ast) != 0) {
        linker_ctx_destroy(ctx);
        return NULL;
    }

    // 对合并后的AST执行外部引用解析
    if (am_linker_import_ref_resolution(global_ast, base_dir) != 0) {
        linker_ctx_destroy(ctx);
        return NULL;
    }

    // AST 解析得到的所有对象都是静态（永生）对象
    if (linker_mark_all_nodes_static(global_ast) != 0) {
        linker_ctx_destroy(ctx);
        return NULL;
    }

    linker_ctx_destroy(ctx);
    return global_ast;
}
/* ===== end:   src/am_linker.c ===== */

/* ===== begin: src/am_compiler.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>



// ===============================================================================
// 内部辅助函数
// ===============================================================================

// AST节点类型分类（用于编译器内部的节点类型判断）
#define AM_COMPILER_NODE_KIND_UNKNOWN   (-1)
#define AM_COMPILER_NODE_KIND_LAMBDA    (0)
#define AM_COMPILER_NODE_KIND_APPLICATION (1)
#define AM_COMPILER_NODE_KIND_QUOTE     (2)
#define AM_COMPILER_NODE_KIND_QUASIQUOTE (3)
#define AM_COMPILER_NODE_KIND_UNQUOTE   (4)
#define AM_COMPILER_NODE_KIND_STRING    (5)

// 编译器视角的值的类型分类
#define AM_COMPILER_VALUE_TYPE_HANDLE   (0)
#define AM_COMPILER_VALUE_TYPE_VARID    (1)
#define AM_COMPILER_VALUE_TYPE_SYMBOL   (2)
#define AM_COMPILER_VALUE_TYPE_NUMBER   (3)
#define AM_COMPILER_VALUE_TYPE_BOOLEAN  (4)
#define AM_COMPILER_VALUE_TYPE_NULL     (5)
#define AM_COMPILER_VALUE_TYPE_UNDEFINED (6)
#define AM_COMPILER_VALUE_TYPE_WCHAR    (8)
#define AM_COMPILER_VALUE_TYPE_OTHER    (9)


static int32_t compiler_value_type(am_value_t v) {
    if (am_value_is_handle(v))    return AM_COMPILER_VALUE_TYPE_HANDLE;
    if (am_value_is_varid(v))     return AM_COMPILER_VALUE_TYPE_VARID;
    if (am_value_is_symbol(v))    return AM_COMPILER_VALUE_TYPE_SYMBOL;
    if (am_value_is_number(v))    return AM_COMPILER_VALUE_TYPE_NUMBER;
    if (am_value_is_boolean(v))   return AM_COMPILER_VALUE_TYPE_BOOLEAN;
    if (am_value_is_null(v))      return AM_COMPILER_VALUE_TYPE_NULL;
    if (am_value_is_undefined(v)) return AM_COMPILER_VALUE_TYPE_UNDEFINED;
    if (am_value_is_wchar(v))     return AM_COMPILER_VALUE_TYPE_WCHAR;
    return AM_COMPILER_VALUE_TYPE_OTHER;
}


static int32_t compiler_node_kind(am_compiler_ctx_t *ctx, am_handle_t h) {
    am_value_t v = am_ast_get_node(ctx->ast, h);
    if (!am_value_is_ptr(v)) return AM_COMPILER_NODE_KIND_UNKNOWN;

    am_object_t *obj = am_value_to_ptr(v);
    if (obj->type == AM_OBJECT_TYPE_WSTRING) {
        return AM_COMPILER_NODE_KIND_STRING;
    }
    if (obj->type == AM_OBJECT_TYPE_LIST) {
        am_list_t *lst = (am_list_t *)obj;
        switch (lst->type) {
            case AM_LIST_TYPE_LAMBDA:      return AM_COMPILER_NODE_KIND_LAMBDA;
            case AM_LIST_TYPE_APPLICATION: return AM_COMPILER_NODE_KIND_APPLICATION;
            case AM_LIST_TYPE_QUOTE:       return AM_COMPILER_NODE_KIND_QUOTE;
            case AM_LIST_TYPE_QUASIQUOTE:  return AM_COMPILER_NODE_KIND_QUASIQUOTE;
            case AM_LIST_TYPE_UNQUOTE:     return AM_COMPILER_NODE_KIND_UNQUOTE;
            default: break;
        }
    }
    return AM_COMPILER_NODE_KIND_UNKNOWN;
}


static am_uint_t compiler_lambda_param_count(am_list_t *lambda) {
    if (!lambda || lambda->length < 2) return 0;
    am_value_t n = lambda->children[1];
    if (!am_value_is_uint(n)) return 0;
    return am_value_to_uint(n);
}


static int32_t compiler_is_tailcall(am_compiler_ctx_t *ctx, am_handle_t handle) {
    if (!ctx || !ctx->ast || !ctx->ast->tailcall_handles) return -1;
    size_t idx = am_list_find(ctx->ast->alloc, ctx->ast->tailcall_handles,
                               am_make_value_of_handle(handle), 0);
    return (idx != SIZE_MAX) ? 0 : -1;
}


static int32_t compiler_is_native_ref(am_compiler_ctx_t *ctx, am_value_t v) {
    if (!am_value_is_varid(v)) return -1;
    return am_ast_check_native_ref(ctx->ast, am_value_to_varid(v));
}


static int32_t compiler_builtin_opcode_for_varid(am_compiler_ctx_t *ctx, am_varid_t varid) {
    if (!ctx || !ctx->ast || !ctx->ast->var_vocab) return -1;
    wchar_t *name = am_vocab_get(ctx->ast->alloc, ctx->ast->var_vocab, &varid);
    if (!name) return -1;

    // 通过 AM_GLOBAL_BUILTIN_VAR 查找 builtin 下标，再通过 AM_BUILTIN_OPCODE_MAP 取得 opcode。
    // 这样 compiler 与 parser 对 builtin 的认知保持一致。
    for (size_t i = 0; i < AM_GLOBAL_BUILTIN_VAR_NUM; i++) {
        if (wcscmp(name, AM_GLOBAL_BUILTIN_VAR[i]) == 0) {
            return AM_BUILTIN_OPCODE_MAP[i];
        }
    }
    return -1;
}


static int32_t compiler_is_break_continue(am_value_t v, int *is_break) {
    if (!am_value_is_symbol(v)) return -1;
    am_symbol_t sym = am_value_to_symbol(v);
    if (sym == am_value_to_symbol(AM_VALUE_KW_break)) {
        if (is_break) *is_break = 1;
        return 0;
    }
    if (sym == am_value_to_symbol(AM_VALUE_KW_continue)) {
        if (is_break) *is_break = 0;
        return 0;
    }
    return -1;
}


static int32_t compiler_varid_name_equals(am_compiler_ctx_t *ctx, am_varid_t varid, const wchar_t *name) {
    if (!ctx || !ctx->ast || !ctx->ast->var_vocab) return -1;
    wchar_t *vname = am_vocab_get(ctx->ast->alloc, ctx->ast->var_vocab, &varid);
    if (!vname) return -1;
    return wcscmp(vname, name) == 0 ? 0 : -1;
}


// ===============================================================================
// while 标签栈操作
// ===============================================================================

static int32_t while_tag_stack_push(am_compiler_ctx_t *ctx, am_value_t cond_tag, am_value_t end_tag) {
    if (!ctx || !ctx->while_tag_stack) return -1;
    am_list_t *lst = am_list_push(ctx->ast->alloc, ctx->while_tag_stack, cond_tag);
    if (!lst) return -1;
    ctx->while_tag_stack = lst;
    lst = am_list_push(ctx->ast->alloc, ctx->while_tag_stack, end_tag);
    if (!lst) return -1;
    ctx->while_tag_stack = lst;
    return 0;
}


static int32_t while_tag_stack_top(am_compiler_ctx_t *ctx, am_value_t *cond_tag, am_value_t *end_tag) {
    if (!ctx || !ctx->while_tag_stack || ctx->while_tag_stack->length < 2) return -1;
    size_t len = ctx->while_tag_stack->length;
    if (cond_tag) *cond_tag = am_list_get(ctx->ast->alloc, ctx->while_tag_stack, len - 2);
    if (end_tag)  *end_tag  = am_list_get(ctx->ast->alloc, ctx->while_tag_stack, len - 1);
    return 0;
}


static int32_t while_tag_stack_pop(am_compiler_ctx_t *ctx) {
    if (!ctx || !ctx->while_tag_stack || ctx->while_tag_stack->length < 2) return -1;
    ctx->while_tag_stack->length -= 2;
    return 0;
}


// ===============================================================================
// 工具函数：指令添加、标签构造/定位/解析、临时变量
// ===============================================================================

// 功能说明：向am_compiler_ctx_t的ilcode中，增加一个am_instruction_t，并更新icount。
// 实现说明：成功返回0；失败返回-1
static int32_t emit_instruction(am_compiler_ctx_t *ctx, uint32_t opcode, am_value_t operand) {
    if (!ctx) return -1;

    if (ctx->icount >= ctx->ilcode_capacity) {
        am_iaddr_t new_cap = ctx->ilcode_capacity ? ctx->ilcode_capacity * 2 : 64;
        am_instruction_t *new_ilcode = (am_instruction_t *)am_realloc(
            ctx->ast->alloc, ctx->ilcode, new_cap * sizeof(am_instruction_t));
        if (!new_ilcode) return -1;
        ctx->ilcode = new_ilcode;
        ctx->ilcode_capacity = new_cap;
    }

    ctx->ilcode[ctx->icount].opcode = opcode;
    ctx->ilcode[ctx->icount].operand = operand;
    ctx->icount++;
    return 0;
}


// 功能说明：标签构造——根据给定的索引TPV（index_value），构造标签（am_value_t）。
// 实现说明：基于任意TPV（一般是handle、varid，称为“索引”TPV）构造一个新的标签TPV（AM_VALUE_TYPE_LABEL）。如果相同索引TPV的标签已存在，则获取已构造的标签TPV，以便后面加入指令的operand。由于编译过程中存在先使用后出现的情况，因此对于同一索引的标签，第一次调用本函数，是从无到有地创建标签，后续调用则是返回已创建的同一标签。只要用于构造标签的索引TPV相等，则构造出来的标签就是同一个标签，这种判定原则与symbol类似。成功返回标签TPV，失败返回AM_VALUE_NULL。
static am_value_t am_compiler_make_label(am_compiler_ctx_t *ctx, am_value_t index_value) {
    if (!ctx || !ctx->value_label_mapping) return AM_VALUE_NULL;

    am_value_t existing = am_map_get(ctx->ast->alloc, ctx->value_label_mapping, index_value);
    if (am_value_is_label(existing)) {
        return existing;
    }

    am_label_t new_label_id = ctx->label_counter++;
    am_value_t label = am_make_value_of_label(new_label_id);

    am_map_t *new_map = am_map_set(ctx->ast->alloc, ctx->value_label_mapping, index_value, label);
    if (!new_map) return AM_VALUE_NULL;
    ctx->value_label_mapping = new_map;
    return label;
}


// 功能说明：标签定位——为标签指定iaddr。
// 实现说明：标签的功能是指代指令序列中的位置。定位指的是将某个标签TPV与已知的iaddr（过去和当前的iaddr，不可能预知未来的iaddr）进行绑定，将标签->iaddr的映射关系，登记到label_iaddr_mapping中。编译过程中，标签的构造和定位，未必是同时发生的，但必须遵守先构造后定位的原则。成功返回0，失败返回-1。
static int32_t am_compiler_locate_label(am_compiler_ctx_t *ctx, am_value_t index_value, am_iaddr_t iaddr) {
    if (!ctx || !ctx->value_label_mapping || !ctx->label_iaddr_mapping) return -1;

    am_value_t label = am_map_get(ctx->ast->alloc, ctx->value_label_mapping, index_value);
    if (!am_value_is_label(label)) return -1;

    am_map_t *new_map = am_map_set(ctx->ast->alloc, ctx->label_iaddr_mapping,
                                    label, am_make_value_of_iaddr(iaddr));
    if (!new_map) return -1;
    ctx->label_iaddr_mapping = new_map;
    return 0;
}


// 功能说明：标签解析——通过标签TPV，获取对应的iaddr。
// 实现说明：在AST全部编译完成后，编译器收集到全部的label及其与iaddr的映射关系，此时即可通过label_iaddr_mapping，将所有的label解析并成绝对的iaddr。成功返回iaddr，失败返回SIZE_MAX。
static am_iaddr_t am_compiler_parse_label_to_iaddr(am_compiler_ctx_t *ctx, am_value_t label) {
    if (!ctx || !ctx->label_iaddr_mapping || !am_value_is_label(label)) return SIZE_MAX;

    am_value_t iaddr_val = am_map_get(ctx->ast->alloc, ctx->label_iaddr_mapping, label);
    if (!am_value_is_iaddr(iaddr_val)) return SIZE_MAX;
    return am_value_to_iaddr(iaddr_val);
}


// 功能说明：构造一个临时变量，加入AST，返回其varid；或者查询符合给定条件的临时变量的varid。
// 设计说明：编译过程中，某些结构需要引入临时变量，本函数即用于这类过程。
// 实现说明：成功返回varid，失败返回SIZE_MAX
static am_varid_t am_compiler_make_temp_varid(am_compiler_ctx_t *ctx, wchar_t *name, am_value_t label, size_t id) {
    if (!ctx || !ctx->ast || !name) return SIZE_MAX;

    wchar_t buf[256];
    int n = swprintf(buf, 256, L"%ls_%zx_%zx", name, (size_t)label, id);
    if (n <= 0 || (size_t)n >= 256) return SIZE_MAX;

    size_t existing = am_vocab_find(ctx->ast->alloc, ctx->ast->var_vocab, buf);
    if (existing != SIZE_MAX) {
        return (am_varid_t)existing;
    }

    size_t old_len = ctx->ast->var_vocab->length;
    size_t new_varid;
    ctx->ast->var_vocab = am_vocab_insert(ctx->ast->alloc, ctx->ast->var_vocab, buf, &new_varid);
    if (!ctx->ast->var_vocab || new_varid == SIZE_MAX) return SIZE_MAX;

    if (new_varid >= old_len) {
        am_list_t *vt = am_list_push(ctx->ast->alloc, ctx->ast->var_type,
                                      am_make_value_of_uint(AM_VAR_TYPE_ILTEMP));
        if (!vt) return SIZE_MAX;
        ctx->ast->var_type = vt;
    }
    else {
        am_list_set(ctx->ast->alloc, ctx->ast->var_type, new_varid,
                    am_make_value_of_uint(AM_VAR_TYPE_ILTEMP));
    }

    return (am_varid_t)new_varid;
}


// ===============================================================================
// 前向声明
// ===============================================================================

static int32_t compile_value(am_compiler_ctx_t *ctx, am_value_t v);
static int32_t compile_application(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_complex_application(am_compiler_ctx_t *ctx, am_handle_t handle, int32_t is_tail);
static int32_t compile_lambda(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_callcc(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_dynamicwind(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_begin(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_define(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_set(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_cond(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_if(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_while(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_and(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_or(am_compiler_ctx_t *ctx, am_handle_t handle);
static int32_t compile_quasiquote(am_compiler_ctx_t *ctx, am_handle_t handle);


// ===============================================================================
// 值编译
// ===============================================================================

// 编译条件表达式（predicate）：handle application需要求值，其余直接push或load
static int32_t compile_predicate(am_compiler_ctx_t *ctx, am_value_t v) {
    if (!ctx) return -1;

    int32_t vt = compiler_value_type(v);
    if (vt == AM_COMPILER_VALUE_TYPE_HANDLE) {
        int32_t kind = compiler_node_kind(ctx, am_value_to_handle(v));
        if (kind == AM_COMPILER_NODE_KIND_APPLICATION) {
            return compile_application(ctx, am_value_to_handle(v));
        }
        return emit_instruction(ctx, AM_VM_OP_push, v);
    }

    if (vt == AM_COMPILER_VALUE_TYPE_VARID) {
        if (compiler_is_native_ref(ctx, v) == 0) {
            return emit_instruction(ctx, AM_VM_OP_push, v);
        }
        return emit_instruction(ctx, AM_VM_OP_load, v);
    }

    if (vt == AM_COMPILER_VALUE_TYPE_SYMBOL) {
        int is_break;
        if (compiler_is_break_continue(v, &is_break) == 0) return -1; // predicate中不允许break/continue
        return emit_instruction(ctx, AM_VM_OP_push, v);
    }

    if (vt == AM_COMPILER_VALUE_TYPE_NUMBER ||
        vt == AM_COMPILER_VALUE_TYPE_BOOLEAN ||
        vt == AM_COMPILER_VALUE_TYPE_NULL ||
        vt == AM_COMPILER_VALUE_TYPE_UNDEFINED ||
        vt == AM_COMPILER_VALUE_TYPE_WCHAR) {
        return emit_instruction(ctx, AM_VM_OP_push, v);
    }

    return -1;
}


// 编译一般的值：根据类型生成push/load/loadclosure等指令
static int32_t compile_value(am_compiler_ctx_t *ctx, am_value_t v) {
    if (!ctx) return -1;

    int32_t vt = compiler_value_type(v);
    if (vt == AM_COMPILER_VALUE_TYPE_HANDLE) {
        am_handle_t h = am_value_to_handle(v);
        int32_t kind = compiler_node_kind(ctx, h);
        switch (kind) {
            case AM_COMPILER_NODE_KIND_LAMBDA:
                return emit_instruction(ctx, AM_VM_OP_loadclosure, am_compiler_make_label(ctx, v));
            case AM_COMPILER_NODE_KIND_QUOTE:
            case AM_COMPILER_NODE_KIND_STRING:
                return emit_instruction(ctx, AM_VM_OP_push, v);
            case AM_COMPILER_NODE_KIND_QUASIQUOTE:
                return compile_quasiquote(ctx, h);
            case AM_COMPILER_NODE_KIND_APPLICATION:
            case AM_COMPILER_NODE_KIND_UNQUOTE:
                return compile_application(ctx, h);
            default:
                return -1;
        }
    }

    if (vt == AM_COMPILER_VALUE_TYPE_SYMBOL) {
        int is_break;
        if (compiler_is_break_continue(v, &is_break) == 0) {
            am_value_t cond_tag, end_tag;
            if (while_tag_stack_top(ctx, &cond_tag, &end_tag) != 0) return -1;
            return emit_instruction(ctx, AM_VM_OP_goto, is_break ? end_tag : cond_tag);
        }
        return emit_instruction(ctx, AM_VM_OP_push, v);
    }

    if (vt == AM_COMPILER_VALUE_TYPE_VARID) {
        if (compiler_is_native_ref(ctx, v) == 0) {
            return emit_instruction(ctx, AM_VM_OP_push, v);
        }
        am_varid_t varid = am_value_to_varid(v);
        if (compiler_builtin_opcode_for_varid(ctx, varid) >= 0) {
            return emit_instruction(ctx, AM_VM_OP_push, v);
        }
        return emit_instruction(ctx, AM_VM_OP_load, v);
    }

    if (vt == AM_COMPILER_VALUE_TYPE_NUMBER ||
        vt == AM_COMPILER_VALUE_TYPE_BOOLEAN ||
        vt == AM_COMPILER_VALUE_TYPE_NULL ||
        vt == AM_COMPILER_VALUE_TYPE_UNDEFINED ||
        vt == AM_COMPILER_VALUE_TYPE_WCHAR) {
        return emit_instruction(ctx, AM_VM_OP_push, v);
    }

    return -1;
}


// ===============================================================================
// Application 编译
// ===============================================================================

static int32_t compile_complex_application(am_compiler_ctx_t *ctx, am_handle_t handle, int32_t is_tail) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length == 0) return 0;

    size_t n = node->length;
    size_t uid = ctx->unique_id_counter;
    ctx->unique_id_counter += 2;

    am_value_t apply_begin_idx  = am_make_value_of_uint(uid);
    am_value_t temp_lambda_idx  = am_make_value_of_uint(uid + 1);

    am_value_t apply_begin_label  = am_compiler_make_label(ctx, apply_begin_idx);
    am_value_t temp_lambda_label  = am_compiler_make_label(ctx, temp_lambda_idx);

    // goto apply_begin
    if (emit_instruction(ctx, AM_VM_OP_goto, apply_begin_label) != 0) return -1;

    // 临时lambda标签
    if (am_compiler_locate_label(ctx, temp_lambda_idx, ctx->icount) != 0) return -1;

    // 按逆序存储形式参数
    for (size_t i = n; i-- > 0;) {
        am_varid_t p = am_compiler_make_temp_varid(ctx, L"TEMP_LAMBDA_PARAM", temp_lambda_label, i);
        if (p == SIZE_MAX) return -1;
        if (emit_instruction(ctx, AM_VM_OP_store, am_make_value_of_varid(p)) != 0) return -1;
    }

    // 加载参数1..n-1
    for (size_t i = 1; i < n; i++) {
        am_varid_t p = am_compiler_make_temp_varid(ctx, L"TEMP_LAMBDA_PARAM", temp_lambda_label, i);
        if (p == SIZE_MAX) return -1;
        if (emit_instruction(ctx, AM_VM_OP_load, am_make_value_of_varid(p)) != 0) return -1;
    }

    // 尾调用参数0（被调用函数）
    am_varid_t p0 = am_compiler_make_temp_varid(ctx, L"TEMP_LAMBDA_PARAM", temp_lambda_label, 0);
    if (p0 == SIZE_MAX) return -1;
    if (emit_instruction(ctx, AM_VM_OP_tailcall, am_make_value_of_varid(p0)) != 0) return -1;

    // return
    if (emit_instruction(ctx, AM_VM_OP_return, AM_VALUE_UNDEFINED) != 0) return -1;

    // apply_begin标签
    if (am_compiler_locate_label(ctx, apply_begin_idx, ctx->icount) != 0) return -1;

    // 编译实参
    for (size_t i = 0; i < n; i++) {
        if (compile_value(ctx, am_list_get(ctx->ast->alloc, node, i)) != 0) return -1;
    }

    // 调用临时lambda
    uint32_t call_opcode = (is_tail == 0) ? AM_VM_OP_tailcall : AM_VM_OP_call;
    return emit_instruction(ctx, call_opcode, temp_lambda_label);
}


static int32_t compile_application(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length == 0) return 0;

    am_value_t first = am_list_get(ctx->ast->alloc, node, 0);

    // 尾调用判断
    int32_t is_tail = compiler_is_tailcall(ctx, handle);

    // 特殊形式
    if (am_value_is_symbol(first)) {
        am_symbol_t sym = am_value_to_symbol(first);
        if (sym == am_value_to_symbol(AM_VALUE_KW_import))  return 0;
        if (sym == am_value_to_symbol(AM_VALUE_KW_native))  return 0;
        if (sym == am_value_to_symbol(AM_VALUE_KW_define_syntax))  return 0;
        if (sym == am_value_to_symbol(AM_VALUE_KW_let_syntax))     return 0;
        if (sym == am_value_to_symbol(AM_VALUE_KW_letrec_syntax))  return 0;
        if (sym == am_value_to_symbol(AM_VALUE_KW_syntax_rules))   return 0;
        if (sym == am_value_to_symbol(AM_VALUE_KW_begin))   return compile_begin(ctx, handle);
        if (sym == am_value_to_symbol(AM_VALUE_KW_define))  return compile_define(ctx, handle);
        if (sym == am_value_to_symbol(AM_VALUE_KW_set))     return compile_set(ctx, handle);
        if (sym == am_value_to_symbol(AM_VALUE_KW_cond))    return compile_cond(ctx, handle);
        if (sym == am_value_to_symbol(AM_VALUE_KW_if))      return compile_if(ctx, handle);
        if (sym == am_value_to_symbol(AM_VALUE_KW_while))   return compile_while(ctx, handle);
        if (sym == am_value_to_symbol(AM_VALUE_KW_and))     return compile_and(ctx, handle);
        if (sym == am_value_to_symbol(AM_VALUE_KW_or))      return compile_or(ctx, handle);
    }

    // 首项是待求值的Application，需要进行η变换。
    // 说明：不能先编译参数、再把首项求值结果存入全局/闭包级临时变量，然后 call 该变量。
    // 因为 call/cc 捕获的续体会保存当时的闭包/栈状态，若函数值放在可变的临时变量里，
    // 续体恢复后会读到错误的函数值；而η变换将函数与实参作为临时lambda的参数（局部绑定），
    // 每次调用都生成新的闭包，续体捕获的是正确的参数绑定，从而保证 yinyang 等用例正确。
    if (am_value_is_handle(first) &&
        compiler_node_kind(ctx, am_value_to_handle(first)) == AM_COMPILER_NODE_KIND_APPLICATION) {
        return compile_complex_application(ctx, handle, is_tail);
    }

    // 首项是合法的可调用项，包括变量、Native、Builtin、Lambda
    if (am_value_is_handle(first) || am_value_is_varid(first) || am_value_is_symbol(first)) {
        // call/cc 与 fork 是全局内置变量形式的特殊形式
        if (am_value_is_varid(first)) {
            am_varid_t first_varid = am_value_to_varid(first);
            // 特殊Builtin：call/cc
            if (compiler_varid_name_equals(ctx, first_varid, L"call/cc") == 0) {
                return compile_callcc(ctx, handle);
            }
            // 特殊Builtin：dynamic-wind
            if (compiler_varid_name_equals(ctx, first_varid, L"dynamic-wind") == 0) {
                return compile_dynamicwind(ctx, handle);
            }
        }

        // 处理参数列表
        for (size_t i = 1; i < node->length; i++) {
            if (compile_value(ctx, am_list_get(ctx->ast->alloc, node, i)) != 0) return -1;
        }

        // 一般Builtin：对应特定VM指令
        if (am_value_is_varid(first)) {
            int32_t opcode = compiler_builtin_opcode_for_varid(ctx, am_value_to_varid(first));
            if (opcode >= 0) {
                return emit_instruction(ctx, (uint32_t)opcode, AM_VALUE_UNDEFINED);
            }
        }

        uint32_t call_opcode = (is_tail == 0) ? AM_VM_OP_tailcall : AM_VM_OP_call;

        if (am_value_is_handle(first) &&
            compiler_node_kind(ctx, am_value_to_handle(first)) == AM_COMPILER_NODE_KIND_LAMBDA) {
            return emit_instruction(ctx, call_opcode, am_compiler_make_label(ctx, first));
        }
        else if (am_value_is_varid(first)) {
            return emit_instruction(ctx, call_opcode, first);
        }

        return -1;
    }

    return -1;
}


// ===============================================================================
// Lambda 编译
// ===============================================================================

// TODO 处理pop问题
static int32_t compile_lambda(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);

    // 定位lambda开始标签
    if (am_compiler_locate_label(ctx, am_make_value_of_handle(handle), ctx->icount) != 0) return -1;

    // 按参数列表逆序，插入store指令
    am_uint_t n_param = compiler_lambda_param_count(node);
    for (am_int_t i = (am_int_t)n_param - 1; i >= 0; i--) {
        am_value_t param = am_list_get(ctx->ast->alloc, node, (size_t)(2 + i));
        if (emit_instruction(ctx, AM_VM_OP_store, param) != 0) return -1;
    }

    // 逐个编译函数体
    for (size_t i = 2 + n_param; i < node->length; i++) {
        if (compile_value(ctx, node->children[i]) != 0) return -1;
        // 除最后一个子表达式外，其余表达式的结果都pop掉
        // if (i < node->length - 1) {
        //     if (emit_instruction(ctx, AM_VM_OP_pop, AM_VALUE_UNDEFINED) != 0) return -1;
        // }
    }

    return emit_instruction(ctx, AM_VM_OP_return, AM_VALUE_UNDEFINED);
}


// ===============================================================================
// 特殊形式编译
// ===============================================================================

static int32_t compile_callcc(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length < 2) return -1;

    am_value_t thunk = am_list_get(ctx->ast->alloc, node, 1);

    // 用于标识此call/cc的唯一标签
    am_value_t thunk_label = am_compiler_make_label(ctx, thunk);
    am_varid_t cont_varid = am_compiler_make_temp_varid(ctx, L"CC", thunk_label,
                                                         ctx->unique_id_counter++);
    if (cont_varid == SIZE_MAX) return -1;
    am_value_t cont_idx = am_make_value_of_varid(cont_varid);

    // capturecc cont_varid
    if (emit_instruction(ctx, AM_VM_OP_capturecc, cont_idx) != 0) return -1;
    // load cont_varid
    if (emit_instruction(ctx, AM_VM_OP_load, cont_idx) != 0) return -1;

    // 调用thunk
    if (am_value_is_handle(thunk) && compiler_node_kind(ctx, am_value_to_handle(thunk)) == AM_COMPILER_NODE_KIND_LAMBDA) {
        if (emit_instruction(ctx, AM_VM_OP_call, am_compiler_make_label(ctx, thunk)) != 0) return -1;
    }
    else if (am_value_is_varid(thunk)) {
        if (emit_instruction(ctx, AM_VM_OP_call, thunk) != 0) return -1;
    }
    else {
        return -1;
    }

    // 续体返回点标签
    am_value_t cont_label = am_compiler_make_label(ctx, cont_idx);
    (void)cont_label;
    return am_compiler_locate_label(ctx, cont_idx, ctx->icount);
}


static int32_t compile_dynamicwind(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length != 4) return -1;

    am_value_t before = am_list_get(ctx->ast->alloc, node, 1);
    am_value_t thunk  = am_list_get(ctx->ast->alloc, node, 2);
    am_value_t after  = am_list_get(ctx->ast->alloc, node, 3);

    if (compile_value(ctx, before) != 0) return -1;
    if (compile_value(ctx, thunk) != 0) return -1;
    if (compile_value(ctx, after) != 0) return -1;

    if (emit_instruction(ctx, AM_VM_OP_dynamicwind, AM_VALUE_UNDEFINED) != 0) return -1;
    if (emit_instruction(ctx, AM_VM_OP_dynamicwind_after_before, AM_VALUE_UNDEFINED) != 0) return -1;
    if (emit_instruction(ctx, AM_VM_OP_dynamicwind_before_after, AM_VALUE_UNDEFINED) != 0) return -1;
    if (emit_instruction(ctx, AM_VM_OP_dynamicwind_done, AM_VALUE_UNDEFINED) != 0) return -1;
    return 0;
}


static int32_t compile_define(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length < 3) return -1;

    am_value_t left = am_list_get(ctx->ast->alloc, node, 1);
    am_value_t right = am_list_get(ctx->ast->alloc, node, 2);

    if (!am_value_is_varid(left)) return -1;

    // 编译右值：lambda节点直接push其标签，其他按普通值编译
    if (am_value_is_handle(right) && compiler_node_kind(ctx, am_value_to_handle(right)) == AM_COMPILER_NODE_KIND_LAMBDA) {
        if (emit_instruction(ctx, AM_VM_OP_push, am_compiler_make_label(ctx, right)) != 0) return -1;
    }
    else {
        if (compile_value(ctx, right) != 0) return -1;
    }

    return emit_instruction(ctx, AM_VM_OP_store, left);
}


static int32_t compile_set(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length < 3) return -1;

    am_value_t left = am_list_get(ctx->ast->alloc, node, 1);
    am_value_t right = am_list_get(ctx->ast->alloc, node, 2);

    if (!am_value_is_varid(left)) return -1;

    // 编译右值：lambda节点生成闭包实例，其他按普通值编译
    if (am_value_is_handle(right) && compiler_node_kind(ctx, am_value_to_handle(right)) == AM_COMPILER_NODE_KIND_LAMBDA) {
        if (emit_instruction(ctx, AM_VM_OP_loadclosure, am_compiler_make_label(ctx, right)) != 0) return -1;
    }
    else {
        if (compile_value(ctx, right) != 0) return -1;
    }

    return emit_instruction(ctx, AM_VM_OP_set, left);
}


// 编译begin节点：依次求值并保留最后一个表达式的结果
// TODO 处理pop问题
static int32_t compile_begin(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length <= 1) return 0;

    for (size_t i = 1; i < node->length; i++) {
        if (compile_value(ctx, am_list_get(ctx->ast->alloc, node, i)) != 0) return -1;
        // 除最后一个子表达式外，其余表达式的结果都pop掉
        // if (i < node->length - 1) {
        //     if (emit_instruction(ctx, AM_VM_OP_pop, AM_VALUE_UNDEFINED) != 0) return -1;
        // }
    }
    return 0;
}


static int32_t compile_cond(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    size_t n = node->length;
    if (n < 2) return -1;

    // COND_END 标签：使用临时变量作为索引，确保同一 cond 的结束标签唯一
    am_varid_t end_lbl_varid = am_compiler_make_temp_varid(
        ctx, L"COND_END", am_make_value_of_handle(handle), 0);
    if (end_lbl_varid == SIZE_MAX) return -1;
    am_value_t end_lbl_idx = am_make_value_of_varid(end_lbl_varid);
    am_value_t end_lbl = am_compiler_make_label(ctx, end_lbl_idx);

    for (size_t i = 1; i < n; i++) {
        // 插入分支开始标签（第一个分支的标签实际上不会被引用，但为统一逻辑仍定位）
        am_varid_t branch_lbl_varid = am_compiler_make_temp_varid(
            ctx, L"COND_BRANCH", am_make_value_of_handle(handle), i);
        if (branch_lbl_varid == SIZE_MAX) return -1;
        am_value_t branch_lbl_idx = am_make_value_of_varid(branch_lbl_varid);
        am_value_t branch_lbl = am_compiler_make_label(ctx, branch_lbl_idx);
        (void)branch_lbl;
        if (am_compiler_locate_label(ctx, branch_lbl_idx, ctx->icount) != 0) return -1;

        am_value_t clause_handle = am_list_get(ctx->ast->alloc, node, i);
        if (!am_value_is_handle(clause_handle)) return -1;
        am_value_t clause_val = am_ast_get_node(ctx->ast, am_value_to_handle(clause_handle));
        if (!am_value_is_ptr(clause_val)) return -1;
        am_list_t *clause = (am_list_t *)am_value_to_ptr(clause_val);
        if (clause->length < 2) return -1;

        am_value_t predicate = am_list_get(ctx->ast->alloc, clause, 0);
        am_value_t branch = am_list_get(ctx->ast->alloc, clause, 1);

        int32_t is_else = am_value_is_symbol(predicate) &&
                          am_value_to_symbol(predicate) == am_value_to_symbol(AM_VALUE_KW_else);

        if (!is_else) {
            if (compile_predicate(ctx, predicate) != 0) return -1;
            if (i == n - 1) {
                if (emit_instruction(ctx, AM_VM_OP_iffalse, end_lbl) != 0) return -1;
            }
            else {
                am_varid_t next_branch_lbl_varid = am_compiler_make_temp_varid(
                    ctx, L"COND_BRANCH", am_make_value_of_handle(handle), i + 1);
                if (next_branch_lbl_varid == SIZE_MAX) return -1;
                am_value_t next_branch_lbl_idx = am_make_value_of_varid(next_branch_lbl_varid);
                am_value_t next_branch_lbl = am_compiler_make_label(ctx, next_branch_lbl_idx);
                if (emit_instruction(ctx, AM_VM_OP_iffalse, next_branch_lbl) != 0) return -1;
            }
        }

        if (compile_value(ctx, branch) != 0) return -1;

        if (is_else || i == n - 1) {
            return am_compiler_locate_label(ctx, end_lbl_idx, ctx->icount);
        }
        else {
            if (emit_instruction(ctx, AM_VM_OP_goto, end_lbl) != 0) return -1;
        }
    }

    return 0;
}


static int32_t compile_if(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length < 3) return -1; // 至少需要predicate和true分支

    am_value_t predicate = am_list_get(ctx->ast->alloc, node, 1);
    am_value_t true_branch = am_list_get(ctx->ast->alloc, node, 2);

    size_t uid = ctx->unique_id_counter;
    ctx->unique_id_counter += 2;
    am_value_t true_label_idx = am_make_value_of_uint(uid);
    am_value_t true_label = am_compiler_make_label(ctx, true_label_idx);
    am_value_t end_label_idx = am_make_value_of_uint(uid + 1);
    am_value_t end_label = am_compiler_make_label(ctx, end_label_idx);

    if (compile_predicate(ctx, predicate) != 0) return -1;

    if (node->length > 3) {
        am_value_t false_branch = am_list_get(ctx->ast->alloc, node, 3);
        if (emit_instruction(ctx, AM_VM_OP_iftrue, true_label) != 0) return -1;
        if (compile_value(ctx, false_branch) != 0) return -1;
        if (emit_instruction(ctx, AM_VM_OP_goto, end_label) != 0) return -1;
        if (am_compiler_locate_label(ctx, true_label_idx, ctx->icount) != 0) return -1;
        if (compile_value(ctx, true_branch) != 0) return -1;
    }
    else {
        if (emit_instruction(ctx, AM_VM_OP_iffalse, end_label) != 0) return -1;
        if (compile_value(ctx, true_branch) != 0) return -1;
    }

    return am_compiler_locate_label(ctx, end_label_idx, ctx->icount);
}


static int32_t compile_while(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    if (node->length < 3) return -1;

    size_t uid = ctx->unique_id_counter;
    ctx->unique_id_counter += 2;
    am_value_t cond_label_idx = am_make_value_of_uint(uid);
    am_value_t cond_label = am_compiler_make_label(ctx, cond_label_idx);
    am_value_t end_label_idx = am_make_value_of_uint(uid + 1);
    am_value_t end_label = am_compiler_make_label(ctx, end_label_idx);

    if (while_tag_stack_push(ctx, cond_label, end_label) != 0) return -1;

    if (am_compiler_locate_label(ctx, cond_label_idx, ctx->icount) != 0) return -1;
    if (compile_predicate(ctx, am_list_get(ctx->ast->alloc, node, 1)) != 0) return -1;
    if (emit_instruction(ctx, AM_VM_OP_iffalse, end_label) != 0) return -1;
    for (size_t i = 2; i < node->length; i++) {
        if (compile_value(ctx, am_list_get(ctx->ast->alloc, node, i)) != 0) return -1;
    }
    if (emit_instruction(ctx, AM_VM_OP_goto, cond_label) != 0) return -1;
    if (am_compiler_locate_label(ctx, end_label_idx, ctx->icount) != 0) return -1;

    return while_tag_stack_pop(ctx);
}


static int32_t compile_and(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    size_t n = node->length;

    size_t uid = ctx->unique_id_counter;
    ctx->unique_id_counter += 2;
    am_value_t end_label_idx = am_make_value_of_uint(uid);
    am_value_t end_label = am_compiler_make_label(ctx, end_label_idx);
    am_value_t false_label_idx = am_make_value_of_uint(uid + 1);
    am_value_t false_label = am_compiler_make_label(ctx, false_label_idx);

    for (size_t i = 1; i < n; i++) {
        if (compile_value(ctx, am_list_get(ctx->ast->alloc, node, i)) != 0) return -1;
        if (emit_instruction(ctx, AM_VM_OP_iffalse, false_label) != 0) return -1;
    }

    if (emit_instruction(ctx, AM_VM_OP_push, AM_VALUE_TRUE) != 0) return -1;
    if (emit_instruction(ctx, AM_VM_OP_goto, end_label) != 0) return -1;
    if (am_compiler_locate_label(ctx, false_label_idx, ctx->icount) != 0) return -1;
    if (emit_instruction(ctx, AM_VM_OP_push, AM_VALUE_FALSE) != 0) return -1;
    return am_compiler_locate_label(ctx, end_label_idx, ctx->icount);
}


static int32_t compile_or(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);
    size_t n = node->length;

    size_t uid = ctx->unique_id_counter;
    ctx->unique_id_counter += 2;
    am_value_t end_label_idx = am_make_value_of_uint(uid);
    am_value_t end_label = am_compiler_make_label(ctx, end_label_idx);
    am_value_t true_label_idx = am_make_value_of_uint(uid + 1);
    am_value_t true_label = am_compiler_make_label(ctx, true_label_idx);

    for (size_t i = 1; i < n; i++) {
        if (compile_value(ctx, am_list_get(ctx->ast->alloc, node, i)) != 0) return -1;
        if (emit_instruction(ctx, AM_VM_OP_iftrue, true_label) != 0) return -1;
    }

    if (emit_instruction(ctx, AM_VM_OP_push, AM_VALUE_FALSE) != 0) return -1;
    if (emit_instruction(ctx, AM_VM_OP_goto, end_label) != 0) return -1;
    if (am_compiler_locate_label(ctx, true_label_idx, ctx->icount) != 0) return -1;
    if (emit_instruction(ctx, AM_VM_OP_push, AM_VALUE_TRUE) != 0) return -1;
    return am_compiler_locate_label(ctx, end_label_idx, ctx->icount);
}


static int32_t compile_quasiquote(am_compiler_ctx_t *ctx, am_handle_t handle) {
    am_value_t node_val = am_ast_get_node(ctx->ast, handle);
    if (!am_value_is_ptr(node_val)) return -1;
    am_list_t *node = (am_list_t *)am_value_to_ptr(node_val);

    for (size_t i = 0; i < node->length; i++) {
        am_value_t child = am_list_get(ctx->ast->alloc, node, i);
        int32_t vt = compiler_value_type(child);

        if (vt == AM_COMPILER_VALUE_TYPE_HANDLE) {
            int32_t kind = compiler_node_kind(ctx, am_value_to_handle(child));
            if (kind == AM_COMPILER_NODE_KIND_APPLICATION ||
                kind == AM_COMPILER_NODE_KIND_UNQUOTE) {
                if (compile_application(ctx, am_value_to_handle(child)) != 0) return -1;
            }
            else if (kind == AM_COMPILER_NODE_KIND_QUASIQUOTE) {
                if (compile_quasiquote(ctx, am_value_to_handle(child)) != 0) return -1;
            }
            else {
                if (emit_instruction(ctx, AM_VM_OP_push, child) != 0) return -1;
            }
        }
        else if (vt == AM_COMPILER_VALUE_TYPE_SYMBOL) {
            int is_break;
            if (compiler_is_break_continue(child, &is_break) == 0) return -1;
            if (emit_instruction(ctx, AM_VM_OP_push, child) != 0) return -1;
        }
        else if (vt == AM_COMPILER_VALUE_TYPE_VARID) {
            if (compiler_is_native_ref(ctx, child) == 0) {
                if (emit_instruction(ctx, AM_VM_OP_push, child) != 0) return -1;
            }
            else {
                if (emit_instruction(ctx, AM_VM_OP_load, child) != 0) return -1;
            }
        }
        else if (vt == AM_COMPILER_VALUE_TYPE_NUMBER ||
                 vt == AM_COMPILER_VALUE_TYPE_BOOLEAN ||
                 vt == AM_COMPILER_VALUE_TYPE_NULL ||
                 vt == AM_COMPILER_VALUE_TYPE_UNDEFINED ||
                 vt == AM_COMPILER_VALUE_TYPE_WCHAR) {
            if (emit_instruction(ctx, AM_VM_OP_push, child) != 0) return -1;
        }
        else {
            return -1;
        }
    }

    if (emit_instruction(ctx, AM_VM_OP_push, am_make_value_of_uint((am_uint_t)node->length)) != 0) return -1;
    return emit_instruction(ctx, AM_VM_OP_concat, AM_VALUE_UNDEFINED);
}


// ===============================================================================
// opstack 最大深度的静态分析
// ===============================================================================

// 分析上下文：记录一个函数入口点（iaddr）及其初始栈深度
typedef struct compiler_depth_entry_t {
    am_iaddr_t iaddr;
    size_t init_depth;
} compiler_depth_entry_t;


// 单条指令对操作数栈的净影响（保守估计）。
// 返回值：正数表示压栈，负数表示出栈，0 表示不变。
static int32_t compiler_stack_effect(am_compiler_ctx_t *ctx, am_iaddr_t iaddr) {
    if (!ctx || iaddr >= ctx->icount) return 0;

    uint32_t op = ctx->ilcode[iaddr].opcode;
    switch (op) {
        case AM_VM_OP_nop:         return 0;
        case AM_VM_OP_store:       return -1;
        case AM_VM_OP_load:        return 1;
        case AM_VM_OP_loadclosure: return 1;
        case AM_VM_OP_push:        return 1;
        case AM_VM_OP_pop:         return -1;
        case AM_VM_OP_swap:        return 0;
        case AM_VM_OP_set:         return -1;
        case AM_VM_OP_call:        return 0;
        case AM_VM_OP_callnative:  return 0;
        case AM_VM_OP_tailcall:    return 0;
        case AM_VM_OP_return:      return 0;
        case AM_VM_OP_capturecc:   return 0;
        case AM_VM_OP_iftrue:      return -1;
        case AM_VM_OP_iffalse:     return -1;
        case AM_VM_OP_goto:        return 0;
        case AM_VM_OP_read:        return 1;
        case AM_VM_OP_write:       return -2;
        case AM_VM_OP_pause:       return 0;
        case AM_VM_OP_halt:        return 0;
        case AM_VM_OP_fork:        return 0;
        case AM_VM_OP_display:     return 0;
        case AM_VM_OP_newline:     return 0;
        case AM_VM_OP_add:
        case AM_VM_OP_sub:
        case AM_VM_OP_mul:
        case AM_VM_OP_div:
        case AM_VM_OP_mod:
        case AM_VM_OP_pow:
        case AM_VM_OP_eq:
        case AM_VM_OP_eqv:
        case AM_VM_OP_equal:
        case AM_VM_OP_ge:
        case AM_VM_OP_le:
        case AM_VM_OP_gt:
        case AM_VM_OP_lt:
        case AM_VM_OP_and:
        case AM_VM_OP_or:
        case AM_VM_OP_cons:
        case AM_VM_OP_get_item:
        case AM_VM_OP_list_push:
            return -1;
        case AM_VM_OP_not:
        case AM_VM_OP_isnull:
        case AM_VM_OP_isundef:
        case AM_VM_OP_isatom:
        case AM_VM_OP_islist:
        case AM_VM_OP_isnumber:
        case AM_VM_OP_isnan:
        case AM_VM_OP_typeof:
        case AM_VM_OP_car:
        case AM_VM_OP_cdr:
        case AM_VM_OP_list_pop:
        case AM_VM_OP_length:
        case AM_VM_OP_duplicate:
            return 0;
        case AM_VM_OP_set_item:
            return -2;
        case AM_VM_OP_concat:
            return -1;
        case AM_VM_OP_dynamicwind:
            return -2;
        case AM_VM_OP_dynamicwind_after_before:
            return 1;
        case AM_VM_OP_dynamicwind_before_after:
            return 0;
        case AM_VM_OP_dynamicwind_done:
            return -1;
        case AM_VM_OP_wind:
            return 0;
        default:
            return 0;
    }
}


static int32_t compiler_depth_add_entry(am_allocator_t *alloc, compiler_depth_entry_t **entries,
                                         size_t *count, size_t *capacity,
                                         am_iaddr_t iaddr, size_t init_depth) {
    if (iaddr == SIZE_MAX) return 0;

    // 去重
    for (size_t i = 0; i < *count; i++) {
        if ((*entries)[i].iaddr == iaddr) return 0;
    }

    if (*count >= *capacity) {
        size_t new_cap = *capacity ? *capacity * 2 : 16;
        compiler_depth_entry_t *new_entries = (compiler_depth_entry_t *)am_realloc(
            alloc, *entries, new_cap * sizeof(compiler_depth_entry_t));
        if (!new_entries) return -1;
        *entries = new_entries;
        *capacity = new_cap;
    }

    (*entries)[*count].iaddr = iaddr;
    (*entries)[*count].init_depth = init_depth;
    (*count)++;
    return 0;
}


typedef struct {
    am_iaddr_t iaddr;
    size_t depth;
} compiler_depth_frame_t;


// 使用显式栈的迭代 DFS，避免循环体净压栈导致 C 调用栈溢出。
static void compiler_depth_search(am_compiler_ctx_t *ctx, am_iaddr_t entry, size_t init_depth,
                                   size_t *best_depth, size_t *global_max) {
    if (!ctx || entry >= ctx->icount) return;

    compiler_depth_frame_t *stack = (compiler_depth_frame_t *)am_malloc(
        ctx->ast->alloc, ctx->icount * 4 * sizeof(compiler_depth_frame_t));
    if (!stack) return;

    size_t stack_capacity = ctx->icount * 4;
    size_t stack_top = 0;
    stack[stack_top].iaddr = entry;
    stack[stack_top].depth = init_depth;
    stack_top++;

    while (stack_top > 0) {
        compiler_depth_frame_t frame = stack[--stack_top];
        am_iaddr_t iaddr = frame.iaddr;
        size_t depth = frame.depth;

        if (iaddr >= ctx->icount) continue;
        if (best_depth[iaddr] != SIZE_MAX && depth <= best_depth[iaddr]) continue;
        // 防止循环体净压栈导致无限展开：深度超过指令数时停止跟随该路径
        if (depth > ctx->icount + 16) continue;

        best_depth[iaddr] = depth;
        if (depth > *global_max) *global_max = depth;

        uint32_t op = ctx->ilcode[iaddr].opcode;
        int32_t effect = compiler_stack_effect(ctx, iaddr);
        size_t next_depth;
        if (effect >= 0) {
            next_depth = depth + (size_t)effect;
        }
        else {
            size_t abs_effect = (size_t)(-effect);
            next_depth = (depth >= abs_effect) ? depth - abs_effect : 0;
        }

        // 辅助宏：将后继状态压栈
        #define DEPTH_PUSH(addr, d) do { \
            if (stack_top < stack_capacity) { \
                stack[stack_top].iaddr = (addr); \
                stack[stack_top].depth = (d); \
                stack_top++; \
            } \
        } while (0)

        switch (op) {
            case AM_VM_OP_goto: {
                am_iaddr_t target = am_compiler_parse_label_to_iaddr(ctx, ctx->ilcode[iaddr].operand);
                if (target != SIZE_MAX) DEPTH_PUSH(target, next_depth);
                break;
            }
            case AM_VM_OP_iftrue:
            case AM_VM_OP_iffalse: {
                am_iaddr_t target = am_compiler_parse_label_to_iaddr(ctx, ctx->ilcode[iaddr].operand);
                if (target != SIZE_MAX) DEPTH_PUSH(target, next_depth);
                DEPTH_PUSH(iaddr + 1, next_depth);
                break;
            }
            case AM_VM_OP_call: {
                // 不进入被调用函数；假设被调用函数净栈效果为 0，继续在调用点之后执行
                DEPTH_PUSH(iaddr + 1, next_depth);
                break;
            }
            case AM_VM_OP_tailcall: {
                // 尾调用不返回，停止当前路径
                break;
            }
            case AM_VM_OP_return:
            case AM_VM_OP_halt: {
                break;
            }
            default: {
                DEPTH_PUSH(iaddr + 1, next_depth);
                break;
            }
        }

        #undef DEPTH_PUSH
    }

    am_free(ctx->ast->alloc, stack);
}


size_t am_compiler_opstack_depth_analysis(am_compiler_ctx_t *ctx) {
    if (!ctx || !ctx->ilcode || ctx->icount == 0) return SIZE_MAX;

    size_t global_max = 0;
    compiler_depth_entry_t *entries = NULL;
    size_t entry_count = 0;
    size_t entry_capacity = 0;

    // 程序入口
    if (compiler_depth_add_entry(ctx->ast->alloc, &entries, &entry_count, &entry_capacity, 0, 0) != 0) {
        am_free(ctx->ast->alloc, entries);
        return SIZE_MAX;
    }

    // 真实 lambda 入口
    if (ctx->ast && ctx->ast->lambda_handles) {
        for (size_t i = 0; i < ctx->ast->lambda_handles->length; i++) {
            am_value_t h = am_list_get(ctx->ast->alloc, ctx->ast->lambda_handles, i);
            if (!am_value_is_handle(h)) continue;

            am_value_t label = am_compiler_make_label(ctx, h);
            if (!am_value_is_label(label)) continue;

            am_iaddr_t iaddr = am_compiler_parse_label_to_iaddr(ctx, label);
            if (iaddr == SIZE_MAX) continue;

            am_value_t node_val = am_ast_get_node(ctx->ast, am_value_to_handle(h));
            am_uint_t n_param = 0;
            if (am_value_is_ptr(node_val)) {
                am_list_t *lambda = (am_list_t *)am_value_to_ptr(node_val);
                n_param = compiler_lambda_param_count(lambda);
            }

            if (compiler_depth_add_entry(ctx->ast->alloc, &entries, &entry_count, &entry_capacity,
                                          iaddr, (size_t)n_param) != 0) {
                am_free(ctx->ast->alloc, entries);
                return SIZE_MAX;
            }
        }
    }

    // 临时 lambda 入口（η 变换等编译器生成的临时 lambda）
    for (am_iaddr_t i = 0; i < ctx->icount; i++) {
        uint32_t op = ctx->ilcode[i].opcode;
        if (op != AM_VM_OP_call && op != AM_VM_OP_tailcall) continue;

        am_value_t operand = ctx->ilcode[i].operand;
        if (!am_value_is_label(operand)) continue;

        am_iaddr_t target = am_compiler_parse_label_to_iaddr(ctx, operand);
        if (target == SIZE_MAX || target >= ctx->icount) continue;
        if (ctx->ilcode[target].opcode != AM_VM_OP_store) continue;

        size_t n_param = 0;
        for (am_iaddr_t j = target; j < ctx->icount && ctx->ilcode[j].opcode == AM_VM_OP_store; j++) {
            n_param++;
        }

        if (compiler_depth_add_entry(ctx->ast->alloc, &entries, &entry_count, &entry_capacity,
                                      target, n_param) != 0) {
            am_free(ctx->ast->alloc, entries);
            return SIZE_MAX;
        }
    }

    size_t *best_depth = (size_t *)am_malloc(ctx->ast->alloc, ctx->icount * sizeof(size_t));
    if (!best_depth) {
        am_free(ctx->ast->alloc, entries);
        return SIZE_MAX;
    }

    for (size_t i = 0; i < entry_count; i++) {
        for (am_iaddr_t j = 0; j < ctx->icount; j++) {
            best_depth[j] = SIZE_MAX;
        }
        compiler_depth_search(ctx, entries[i].iaddr, entries[i].init_depth, best_depth, &global_max);
    }

    am_free(ctx->ast->alloc, best_depth);
    am_free(ctx->ast->alloc, entries);

    return global_max > 0 ? global_max : 1;
}


// ===============================================================================
// 编译器入口与标签解析
// ===============================================================================

int32_t am_compile_all(am_compiler_ctx_t *ctx) {
    if (!ctx || !ctx->ast) return -1;

    // 程序入口：调用顶级lambda
    am_handle_t top_lambda = ctx->ast->top_lambda_handle;
    if (top_lambda == AM_HANDLE_NULL) {
        top_lambda = am_ast_get_top_lambda_node_handle(ctx->ast);
    }
    if (top_lambda == AM_HANDLE_NULL) return -1;

    if (emit_instruction(ctx, AM_VM_OP_call,
                        am_compiler_make_label(ctx, am_make_value_of_handle(top_lambda))) != 0) {
        return -1;
    }
    // ret 为 0 时程序结束使用 halt；否则跳转到返回目标
    if (ctx->ret > 0) {
        if (emit_instruction(ctx, AM_VM_OP_goto, am_make_value_of_iaddr(ctx->ret)) != 0) return -1;
    }
    else {
        if (emit_instruction(ctx, AM_VM_OP_halt, AM_VALUE_UNDEFINED) != 0) return -1;
    }

    // 顺序编译所有lambda节点
    if (!ctx->ast->lambda_handles) return -1;

    // 预创建所有 lambda 标签，避免 lambda_handles 顺序导致内层 lambda 标签未创建
    for (size_t i = 0; i < ctx->ast->lambda_handles->length; i++) {
        am_value_t h = am_list_get(ctx->ast->alloc, ctx->ast->lambda_handles, i);
        if (!am_value_is_handle(h)) continue;
        if (am_value_is_null(am_compiler_make_label(ctx, h))) return -1;
    }

    for (size_t i = 0; i < ctx->ast->lambda_handles->length; i++) {
        am_value_t h = am_list_get(ctx->ast->alloc, ctx->ast->lambda_handles, i);
        if (!am_value_is_handle(h)) continue;
        if (compile_lambda(ctx, am_value_to_handle(h)) != 0) return -1;
    }

    return 0;
}


int32_t am_compiler_label_resolution(am_compiler_ctx_t *ctx, am_iaddr_t offset) {
    if (!ctx || !ctx->ilcode) return -1;

    for (am_iaddr_t i = 0; i < ctx->icount; i++) {
        if (am_value_is_label(ctx->ilcode[i].operand)) {
            am_iaddr_t addr = am_compiler_parse_label_to_iaddr(ctx, ctx->ilcode[i].operand);
            if (addr == SIZE_MAX) return -1;
            ctx->ilcode[i].operand = am_make_value_of_iaddr(addr + offset);
        }
    }
    return 0;
}


am_module_t *am_compile(am_ast_t *ast, am_iaddr_t offset, am_iaddr_t ret) {
    if (!ast || !ast->alloc) return NULL;

    am_compiler_ctx_t *ctx = am_compiler_ctx_create(ast);
    if (!ctx) return NULL;

    ctx->offset = offset;
    ctx->ret = ret;

    if (am_compile_all(ctx) != 0) {
        am_compiler_ctx_destroy(ctx);
        return NULL;
    }

    size_t opstack_depth = am_compiler_opstack_depth_analysis(ctx);
    if (opstack_depth == SIZE_MAX) {
        am_compiler_ctx_destroy(ctx);
        return NULL;
    }

    if (am_compiler_label_resolution(ctx, offset) != 0) {
        am_compiler_ctx_destroy(ctx);
        return NULL;
    }

    am_module_t *mod = (am_module_t *)am_calloc(ast->alloc, sizeof(am_module_t));
    if (!mod) {
        am_compiler_ctx_destroy(ctx);
        return NULL;
    }

    mod->base.type = AM_OBJECT_TYPE_MODULE;
    mod->opstack_depth = opstack_depth;
    mod->ast = ast;
    mod->ilcode = ctx->ilcode;
    mod->ilcode_length = ctx->icount;

    // ilcode所有权转移给module，避免ctx销毁时释放ilcode
    ctx->ilcode = NULL;
    ctx->ilcode_capacity = 0;

    am_compiler_ctx_destroy(ctx);
    return mod;
}


// ===============================================================================
// 上下文创建与销毁
// ===============================================================================

am_compiler_ctx_t *am_compiler_ctx_create(am_ast_t *ast) {
    if (!ast || !ast->alloc) return NULL;

    am_compiler_ctx_t *ctx = (am_compiler_ctx_t *)am_calloc(ast->alloc, sizeof(am_compiler_ctx_t));
    if (!ctx) return NULL;

    ctx->ast = ast;
    ctx->icount = 0;
    ctx->ilcode_capacity = 0;
    ctx->ilcode = NULL;
    ctx->label_counter = 0;
    ctx->unique_id_counter = 0;

    ctx->value_label_mapping = am_map_create(ast->alloc, 64);
    ctx->label_iaddr_mapping = am_map_create(ast->alloc, 64);
    ctx->while_tag_stack = am_list_create(ast->alloc, 16, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    ctx->offset = 0;
    ctx->ret = 0;

    if (!ctx->value_label_mapping || !ctx->label_iaddr_mapping || !ctx->while_tag_stack) {
        am_compiler_ctx_destroy(ctx);
        return NULL;
    }

    return ctx;
}


void am_compiler_ctx_destroy(am_compiler_ctx_t *ctx) {
    if (!ctx) return;

    am_allocator_t *alloc = ctx->ast ? ctx->ast->alloc : NULL;
    if (!alloc) return;

    if (ctx->ilcode) am_free(alloc, ctx->ilcode);
    if (ctx->value_label_mapping) am_map_destroy(alloc, ctx->value_label_mapping);
    if (ctx->label_iaddr_mapping) am_map_destroy(alloc, ctx->label_iaddr_mapping);
    if (ctx->while_tag_stack) am_list_destroy(alloc, ctx->while_tag_stack);
    am_free(alloc, ctx);
}
/* ===== end:   src/am_compiler.c ===== */

/* ===== begin: src/am_module.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


#define MODULE_MAGIC     "BD4SURAM"
#define MODULE_VERSION   ((uint32_t)202607u)

// flags 位定义：目前格式固定为小端序（bit0=0），其余位保留
#define MODULE_FLAGS_LITTLE_ENDIAN ((uint32_t)0u)

// 模块磁盘格式（平台无关固定宽度，小端；详见 include/object.h）。
// 头部为以下定宽字段的顺序拼接（所有多字节整数小端），总长 104 字节：
//   [8]  magic "BD4SURAM"
//   [u32] version
//   [u32] flags（bit0=0：小端）
//   [u32] total_size（模块转储总字节数）
//   [i32] base_type / [u32] base_hash / [u32] base_gcmark
//   [u64] header（保留元数据）
//   [u32] opstack_depth
//   [u32] ilcode_length（指令条数）
//   [u32] ilcode_offset
//   [u32] nodes_heap_offset
//   [u32] var_vocab_offset / symbol_vocab_offset / var_type_offset
//   [u32] natives_offset / dependencies_offset / scopes_offset
//   [u32] var_arn_mapping_offset / node_token_mapping_offset
//   [u32] lambda_handles_offset / tailcall_handles_offset / var_top_offset
//   [u32] strindex_offset
// 各区段在头部之后紧密排列（无对齐填充），偏移量相对于模块转储起点，0 表示该区段不存在。
// ilcode 区段：每条指令为 [u8 opcode, dvalue operand]。

typedef struct {
    uint32_t total_size;

    int32_t  base_type;
    uint32_t base_hash;
    uint32_t base_gcmark;
    uint64_t header;

    uint32_t opstack_depth;
    uint32_t ilcode_length;

    uint32_t ilcode_offset;
    uint32_t nodes_heap_offset;

    uint32_t var_vocab_offset;
    uint32_t symbol_vocab_offset;
    uint32_t var_type_offset;
    uint32_t natives_offset;
    uint32_t dependencies_offset;
    uint32_t scopes_offset;
    uint32_t var_arn_mapping_offset;
    uint32_t node_token_mapping_offset;
    uint32_t lambda_handles_offset;
    uint32_t tailcall_handles_offset;
    uint32_t var_top_offset;
    uint32_t strindex_offset;
} module_header_t;

#define MODULE_HEADER_DISK_SIZE (104)

// 将模块头写入 buffer（字段逐个小端写入，与宿主字节序/填充无关）
static void module_header_write(uint8_t *buffer, size_t offset, const module_header_t *hdr) {
    size_t pos = offset;
    memcpy(buffer + pos, MODULE_MAGIC, 8);            pos += 8;
    am_disk_write_u32(buffer, pos, MODULE_VERSION);   pos += 4;
    am_disk_write_u32(buffer, pos, MODULE_FLAGS_LITTLE_ENDIAN); pos += 4;
    am_disk_write_u32(buffer, pos, hdr->total_size);  pos += 4;
    am_disk_write_u32(buffer, pos, (uint32_t)hdr->base_type);   pos += 4;
    am_disk_write_u32(buffer, pos, hdr->base_hash);   pos += 4;
    am_disk_write_u32(buffer, pos, hdr->base_gcmark); pos += 4;
    am_disk_write_u64(buffer, pos, hdr->header);      pos += 8;
    am_disk_write_u32(buffer, pos, hdr->opstack_depth);  pos += 4;
    am_disk_write_u32(buffer, pos, hdr->ilcode_length);  pos += 4;
    am_disk_write_u32(buffer, pos, hdr->ilcode_offset);  pos += 4;
    am_disk_write_u32(buffer, pos, hdr->nodes_heap_offset); pos += 4;
    am_disk_write_u32(buffer, pos, hdr->var_vocab_offset);  pos += 4;
    am_disk_write_u32(buffer, pos, hdr->symbol_vocab_offset); pos += 4;
    am_disk_write_u32(buffer, pos, hdr->var_type_offset);   pos += 4;
    am_disk_write_u32(buffer, pos, hdr->natives_offset);    pos += 4;
    am_disk_write_u32(buffer, pos, hdr->dependencies_offset); pos += 4;
    am_disk_write_u32(buffer, pos, hdr->scopes_offset);     pos += 4;
    am_disk_write_u32(buffer, pos, hdr->var_arn_mapping_offset); pos += 4;
    am_disk_write_u32(buffer, pos, hdr->node_token_mapping_offset); pos += 4;
    am_disk_write_u32(buffer, pos, hdr->lambda_handles_offset);  pos += 4;
    am_disk_write_u32(buffer, pos, hdr->tailcall_handles_offset); pos += 4;
    am_disk_write_u32(buffer, pos, hdr->var_top_offset);    pos += 4;
    am_disk_write_u32(buffer, pos, hdr->strindex_offset);   pos += 4;
}

// 从 buffer 读取模块头。成功返回 0，失败（magic/version/flags 不匹配）返回 -1。
static int32_t module_header_read(const uint8_t *buffer, size_t offset, module_header_t *hdr) {
    size_t pos = offset;
    if (memcmp(buffer + pos, MODULE_MAGIC, 8) != 0) {
        fprintf(stderr, "[module_load] bad magic\n");
        return -1;
    }
    pos += 8;
    uint32_t version = am_disk_read_u32(buffer, pos); pos += 4;
    if (version != MODULE_VERSION) {
        fprintf(stderr, "[module_load] unsupported version %u\n", version);
        return -1;
    }
    uint32_t flags = am_disk_read_u32(buffer, pos); pos += 4;
    if (flags != MODULE_FLAGS_LITTLE_ENDIAN) {
        fprintf(stderr, "[module_load] unsupported flags %u\n", flags);
        return -1;
    }

    hdr->total_size    = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->base_type     = (int32_t)am_disk_read_u32(buffer, pos); pos += 4;
    hdr->base_hash     = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->base_gcmark   = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->header        = am_disk_read_u64(buffer, pos); pos += 8;
    hdr->opstack_depth = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->ilcode_length = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->ilcode_offset = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->nodes_heap_offset = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->var_vocab_offset  = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->symbol_vocab_offset = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->var_type_offset   = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->natives_offset    = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->dependencies_offset = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->scopes_offset     = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->var_arn_mapping_offset = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->node_token_mapping_offset = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->lambda_handles_offset  = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->tailcall_handles_offset = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->var_top_offset    = am_disk_read_u32(buffer, pos); pos += 4;
    hdr->strindex_offset   = am_disk_read_u32(buffer, pos); pos += 4;
    return 0;
}

// 计算 ilcode 区段的磁盘字节数（每条指令：u8 opcode + dvalue operand）
static size_t module_ilcode_disk_size(am_module_t *mod) {
    size_t size = 0;
    for (am_iaddr_t i = 0; i < mod->ilcode_length; i++) {
        size += 1 + am_disk_value_size(mod->ilcode[i].operand);
    }
    return size;
}

// 转储 ilcode 区段。返回写入字节数。buffer 为 NULL 时仅计算字节数。
static size_t module_ilcode_dump(am_module_t *mod, uint8_t *buffer, size_t offset) {
    size_t pos = offset;
    for (am_iaddr_t i = 0; i < mod->ilcode_length; i++) {
        if (buffer) buffer[pos] = (uint8_t)mod->ilcode[i].opcode;
        pos += 1;
        pos += am_disk_write_value(buffer, pos, mod->ilcode[i].operand);
    }
    return pos - offset;
}

// 加载 ilcode 区段。成功返回 0，失败返回 -1。
static int32_t module_ilcode_load(am_module_t *mod, const uint8_t *buffer, size_t offset) {
    size_t pos = offset;
    for (am_iaddr_t i = 0; i < mod->ilcode_length; i++) {
        mod->ilcode[i].opcode = (uint32_t)buffer[pos];
        pos += 1;
        am_value_t operand = 0;
        size_t n = am_disk_read_value(buffer, pos, &operand);
        if (!n) return -1;
        pos += n;
        mod->ilcode[i].operand = operand;
    }
    return 0;
}

static void module_free_ast(am_allocator_t *container_alloc,
                            am_allocator_t *obj_alloc,
                            am_ast_t *ast,
                            int parts) {
    if (!ast) return;

    /* parts 用于区分哪些子对象已经加载成功；
     * 0 表示全部尝试释放，1 表示只释放已经加载的节点堆。 */
    if (parts == 0) {
        if (ast->var_vocab)        am_vocab_destroy(obj_alloc, ast->var_vocab);
        if (ast->symbol_vocab)     am_vocab_destroy(obj_alloc, ast->symbol_vocab);
        if (ast->var_type)         am_list_destroy(obj_alloc, ast->var_type);
        if (ast->natives)          am_map_destroy(obj_alloc, ast->natives);
        if (ast->dependencies)     am_map_destroy(obj_alloc, ast->dependencies);
        if (ast->scopes)           am_map_destroy(obj_alloc, ast->scopes);
        if (ast->var_arn_mapping)  am_map_destroy(obj_alloc, ast->var_arn_mapping);
        if (ast->node_token_mapping) am_map_destroy(obj_alloc, ast->node_token_mapping);
        if (ast->lambda_handles)   am_list_destroy(obj_alloc, ast->lambda_handles);
        if (ast->tailcall_handles) am_list_destroy(obj_alloc, ast->tailcall_handles);
        if (ast->var_top)          am_list_destroy(obj_alloc, ast->var_top);
        if (ast->strindex)         am_strindex_destroy(obj_alloc, ast->strindex);
    }

    if (ast->nodes) {
        am_heap_destroy(container_alloc, obj_alloc, ast->nodes);
    }

    am_free(container_alloc, ast);
}

size_t am_module_dump(am_allocator_t *container_alloc,
                      am_allocator_t *obj_alloc,
                      am_module_t *mod,
                      uint8_t *buffer,
                      size_t offset) {
    (void)container_alloc;
    (void)obj_alloc;

    if (!mod || !mod->ast || !mod->ilcode) {
        fprintf(stderr, "[module_dump] invalid module\n");
        return SIZE_MAX;
    }

    am_ast_t *ast = mod->ast;

    if (mod->ilcode_length > UINT32_MAX || mod->opstack_depth > UINT32_MAX) {
        fprintf(stderr, "[module_dump] module too large\n");
        return SIZE_MAX;
    }

    module_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.base_type = mod->base.type;
    hdr.base_hash = mod->base.hash;
    hdr.base_gcmark = mod->base.gcmark;
    hdr.header = mod->header;
    hdr.opstack_depth = (uint32_t)mod->opstack_depth;
    hdr.ilcode_length = (uint32_t)mod->ilcode_length;

    /* 各区段在头部之后紧密排列（无对齐填充，加载端全部按字节解码） */
    size_t off = offset + MODULE_HEADER_DISK_SIZE;

    /* IL code */
    hdr.ilcode_offset = (uint32_t)(off - offset);
    size_t il_size = module_ilcode_disk_size(mod);
    off += il_size;

    /* AST nodes heap (deep dump) */
    hdr.nodes_heap_offset = (uint32_t)(off - offset);
    size_t nodes_size = am_heap_deep_dump(ast->alloc, ast->alloc, ast->nodes, NULL, 0);
    if (nodes_size == SIZE_MAX) {
        fprintf(stderr, "[module_dump] failed to compute nodes heap size\n");
        return SIZE_MAX;
    }
    off += nodes_size;

    /* symbol / variable vocabularies */
    if (ast->var_vocab) {
        hdr.var_vocab_offset = (uint32_t)(off - offset);
        size_t sz = am_vocab_dump(ast->alloc, ast->var_vocab, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }
    if (ast->symbol_vocab) {
        hdr.symbol_vocab_offset = (uint32_t)(off - offset);
        size_t sz = am_vocab_dump(ast->alloc, ast->symbol_vocab, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }

    /* var_type list */
    if (ast->var_type) {
        hdr.var_type_offset = (uint32_t)(off - offset);
        size_t sz = am_list_dump(ast->alloc, ast->var_type, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }

    /* maps */
    if (ast->natives) {
        hdr.natives_offset = (uint32_t)(off - offset);
        size_t sz = am_map_dump(ast->alloc, ast->natives, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }
    if (ast->dependencies) {
        hdr.dependencies_offset = (uint32_t)(off - offset);
        size_t sz = am_map_dump(ast->alloc, ast->dependencies, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }
    if (ast->scopes) {
        hdr.scopes_offset = (uint32_t)(off - offset);
        size_t sz = am_map_dump(ast->alloc, ast->scopes, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }
    if (ast->var_arn_mapping) {
        hdr.var_arn_mapping_offset = (uint32_t)(off - offset);
        size_t sz = am_map_dump(ast->alloc, ast->var_arn_mapping, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }
    if (ast->node_token_mapping) {
        hdr.node_token_mapping_offset = (uint32_t)(off - offset);
        size_t sz = am_map_dump(ast->alloc, ast->node_token_mapping, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }

    /* lists */
    if (ast->lambda_handles) {
        hdr.lambda_handles_offset = (uint32_t)(off - offset);
        size_t sz = am_list_dump(ast->alloc, ast->lambda_handles, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }
    if (ast->tailcall_handles) {
        hdr.tailcall_handles_offset = (uint32_t)(off - offset);
        size_t sz = am_list_dump(ast->alloc, ast->tailcall_handles, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }
    if (ast->var_top) {
        hdr.var_top_offset = (uint32_t)(off - offset);
        size_t sz = am_list_dump(ast->alloc, ast->var_top, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }

    /* strindex */
    if (ast->strindex) {
        hdr.strindex_offset = (uint32_t)(off - offset);
        size_t sz = am_strindex_dump(ast->alloc, ast->strindex, NULL, 0);
        if (sz == SIZE_MAX) return SIZE_MAX;
        off += sz;
    }

    if (off - offset > UINT32_MAX) {
        fprintf(stderr, "[module_dump] module dump exceeds 4GiB\n");
        return SIZE_MAX;
    }
    hdr.total_size = (uint32_t)(off - offset);

    if (buffer != NULL && offset != SIZE_MAX) {
        module_header_write(buffer, offset, &hdr);

        size_t il_written = module_ilcode_dump(mod, buffer, offset + hdr.ilcode_offset);
        if (il_written != il_size) {
            fprintf(stderr, "[module_dump] ilcode dump size mismatch\n");
            return SIZE_MAX;
        }

        size_t written = am_heap_deep_dump(ast->alloc, ast->alloc, ast->nodes,
                                           buffer, offset + hdr.nodes_heap_offset);
        if (written != nodes_size) {
            fprintf(stderr, "[module_dump] nodes heap dump size mismatch\n");
            return SIZE_MAX;
        }

        if (hdr.var_vocab_offset) {
            am_vocab_dump(ast->alloc, ast->var_vocab,
                          buffer, offset + hdr.var_vocab_offset);
        }
        if (hdr.symbol_vocab_offset) {
            am_vocab_dump(ast->alloc, ast->symbol_vocab,
                          buffer, offset + hdr.symbol_vocab_offset);
        }
        if (hdr.var_type_offset) {
            am_list_dump(ast->alloc, ast->var_type,
                         buffer, offset + hdr.var_type_offset);
        }
        if (hdr.natives_offset) {
            am_map_dump(ast->alloc, ast->natives,
                        buffer, offset + hdr.natives_offset);
        }
        if (hdr.dependencies_offset) {
            am_map_dump(ast->alloc, ast->dependencies,
                        buffer, offset + hdr.dependencies_offset);
        }
        if (hdr.scopes_offset) {
            am_map_dump(ast->alloc, ast->scopes,
                        buffer, offset + hdr.scopes_offset);
        }
        if (hdr.var_arn_mapping_offset) {
            am_map_dump(ast->alloc, ast->var_arn_mapping,
                        buffer, offset + hdr.var_arn_mapping_offset);
        }
        if (hdr.node_token_mapping_offset) {
            am_map_dump(ast->alloc, ast->node_token_mapping,
                        buffer, offset + hdr.node_token_mapping_offset);
        }
        if (hdr.lambda_handles_offset) {
            am_list_dump(ast->alloc, ast->lambda_handles,
                         buffer, offset + hdr.lambda_handles_offset);
        }
        if (hdr.tailcall_handles_offset) {
            am_list_dump(ast->alloc, ast->tailcall_handles,
                         buffer, offset + hdr.tailcall_handles_offset);
        }
        if (hdr.var_top_offset) {
            am_list_dump(ast->alloc, ast->var_top,
                         buffer, offset + hdr.var_top_offset);
        }
        if (hdr.strindex_offset) {
            am_strindex_dump(ast->alloc, ast->strindex,
                             buffer, offset + hdr.strindex_offset);
        }
    }

    return (size_t)hdr.total_size;
}

am_module_t *am_module_load(am_allocator_t *container_alloc,
                            am_allocator_t *obj_alloc,
                            uint8_t *buffer,
                            size_t offset) {
    if (!container_alloc || !obj_alloc || !buffer) {
        fprintf(stderr, "[module_load] invalid arguments\n");
        return NULL;
    }

    module_header_t hdr_buf;
    module_header_t *hdr = &hdr_buf;
    if (module_header_read(buffer, offset, hdr) != 0) {
        return NULL;
    }

    am_module_t *mod = (am_module_t *)am_malloc(container_alloc, sizeof(am_module_t));
    if (!mod) {
        fprintf(stderr, "[module_load] failed to allocate module\n");
        return NULL;
    }

    mod->base.type = hdr->base_type;
    mod->base.hash = hdr->base_hash;
    mod->base.gcmark = hdr->base_gcmark;
    mod->header = hdr->header;
    mod->opstack_depth = hdr->opstack_depth;
    mod->ilcode_length = hdr->ilcode_length;

    if ((uint64_t)hdr->ilcode_length * (uint64_t)sizeof(am_instruction_t) > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "[module_load] ilcode too large\n");
        am_free(container_alloc, mod);
        return NULL;
    }

    mod->ilcode = (am_instruction_t *)am_malloc(container_alloc,
                                                (size_t)mod->ilcode_length * sizeof(am_instruction_t));
    if (!mod->ilcode) {
        fprintf(stderr, "[module_load] failed to allocate ilcode\n");
        am_free(container_alloc, mod);
        return NULL;
    }
    if (module_ilcode_load(mod, buffer, offset + hdr->ilcode_offset) != 0) {
        fprintf(stderr, "[module_load] failed to decode ilcode\n");
        am_free(container_alloc, mod->ilcode);
        am_free(container_alloc, mod);
        return NULL;
    }

    am_ast_t *ast = (am_ast_t *)am_malloc(container_alloc, sizeof(am_ast_t));
    if (!ast) {
        fprintf(stderr, "[module_load] failed to allocate ast\n");
        am_free(container_alloc, mod->ilcode);
        am_free(container_alloc, mod);
        return NULL;
    }
    memset(ast, 0, sizeof(am_ast_t));
    ast->alloc = obj_alloc;

    mod->ast = ast;

    if (hdr->nodes_heap_offset) {
        ast->nodes = am_heap_deep_load(container_alloc, obj_alloc,
                                       buffer, offset + hdr->nodes_heap_offset);
        if (!ast->nodes) {
            fprintf(stderr, "[module_load] failed to load nodes heap\n");
            goto fail;
        }
    }

    if (hdr->var_vocab_offset) {
        ast->var_vocab = am_vocab_load(obj_alloc, buffer,
                                       offset + hdr->var_vocab_offset);
        if (!ast->var_vocab) goto fail;
    }
    if (hdr->symbol_vocab_offset) {
        ast->symbol_vocab = am_vocab_load(obj_alloc, buffer,
                                          offset + hdr->symbol_vocab_offset);
        if (!ast->symbol_vocab) goto fail;
    }
    if (hdr->var_type_offset) {
        ast->var_type = am_list_load(obj_alloc, buffer,
                                     offset + hdr->var_type_offset);
        if (!ast->var_type) goto fail;
    }

    if (hdr->natives_offset) {
        ast->natives = am_map_load(obj_alloc, buffer,
                                   offset + hdr->natives_offset);
        if (!ast->natives) goto fail;
    }
    if (hdr->dependencies_offset) {
        ast->dependencies = am_map_load(obj_alloc, buffer,
                                        offset + hdr->dependencies_offset);
        if (!ast->dependencies) goto fail;
    }
    if (hdr->scopes_offset) {
        ast->scopes = am_map_load(obj_alloc, buffer,
                                  offset + hdr->scopes_offset);
        if (!ast->scopes) goto fail;
    }
    if (hdr->var_arn_mapping_offset) {
        ast->var_arn_mapping = am_map_load(obj_alloc, buffer,
                                           offset + hdr->var_arn_mapping_offset);
        if (!ast->var_arn_mapping) goto fail;
    }
    if (hdr->node_token_mapping_offset) {
        ast->node_token_mapping = am_map_load(obj_alloc, buffer,
                                              offset + hdr->node_token_mapping_offset);
        if (!ast->node_token_mapping) goto fail;
    }

    if (hdr->lambda_handles_offset) {
        ast->lambda_handles = am_list_load(obj_alloc, buffer,
                                           offset + hdr->lambda_handles_offset);
        if (!ast->lambda_handles) goto fail;
    }
    if (hdr->tailcall_handles_offset) {
        ast->tailcall_handles = am_list_load(obj_alloc, buffer,
                                             offset + hdr->tailcall_handles_offset);
        if (!ast->tailcall_handles) goto fail;
    }
    if (hdr->var_top_offset) {
        ast->var_top = am_list_load(obj_alloc, buffer,
                                    offset + hdr->var_top_offset);
        if (!ast->var_top) goto fail;
    }

    if (hdr->strindex_offset) {
        ast->strindex = am_strindex_load(obj_alloc, buffer,
                                         offset + hdr->strindex_offset);
        if (!ast->strindex) goto fail;
    }

    return mod;

fail:
    fprintf(stderr, "[module_load] failed to load AST sub-object\n");
    module_free_ast(container_alloc, obj_alloc, ast, 0);
    am_free(container_alloc, mod->ilcode);
    am_free(container_alloc, mod);
    return NULL;
}

// =============================================================
// PackBits 压缩/解压
// =============================================================

size_t am_packbits_compress(uint8_t *src, size_t src_len, uint8_t *dst) {
    if (!src) return SIZE_MAX;

    size_t i = 0;
    size_t out_pos = 0;

    while (i < src_len) {
        // 探测从当前位置开始的重复字节游程
        size_t run_end = i + 1;
        while (run_end < src_len &&
               src[run_end] == src[i] &&
               run_end - i < 128) {
            run_end++;
        }
        size_t run_len = run_end - i;

        // 重复 3 次及以上才编码为游程，否则并入字面量
        if (run_len >= 3) {
            if (dst) dst[out_pos] = (uint8_t)(257 - run_len);
            out_pos++;
            if (dst) dst[out_pos] = src[i];
            out_pos++;
            i = run_end;
        } else {
            // 编码字面量游程
            size_t lit_start = i;
            while (i < src_len) {
                // 遇到 3 个及以上重复字节时结束字面量
                if (i + 2 < src_len &&
                    src[i] == src[i + 1] &&
                    src[i] == src[i + 2]) {
                    break;
                }
                i++;
                if (i - lit_start >= 128) break;
            }
            size_t lit_len = i - lit_start;
            if (dst) dst[out_pos] = (uint8_t)(lit_len - 1);
            out_pos++;
            if (dst) memcpy(dst + out_pos, src + lit_start, lit_len);
            out_pos += lit_len;
        }
    }

    return out_pos;
}

size_t am_packbits_decompress(uint8_t *src, size_t src_len, uint8_t *dst) {
    if (!src) return SIZE_MAX;

    size_t i = 0;
    size_t out_pos = 0;

    while (i < src_len) {
        int8_t ctrl = (int8_t)src[i++];

        if (ctrl >= 0) {
            // 0..127：复制接下来的 ctrl+1 个字节
            size_t count = (size_t)ctrl + 1;
            if (i + count > src_len) return SIZE_MAX;
            if (dst) memcpy(dst + out_pos, src + i, count);
            out_pos += count;
            i += count;
        } else if (ctrl != -128) {
            // -127..-1：将下一个字节重复 -ctrl+1 次
            size_t count = (size_t)(-ctrl + 1);
            if (i >= src_len) return SIZE_MAX;
            if (dst) memset(dst + out_pos, src[i], count);
            out_pos += count;
            i++;
        }
        // ctrl == -128 为无操作
    }

    return out_pos;
}
/* ===== end:   src/am_module.c ===== */

/* ===== begin: src/am_js2scm.c ===== */
/* JS -> Scheme (non-standard) mechanical translator.
 * Migrated from jstoscm.c into the Animac project as a reusable library.
 *
 * 宽字符版本：内部全程使用 wchar_t 处理，不再通过 UTF-8 中转。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <stdarg.h>
#include <wchar.h>
#include <wctype.h>
#include <setjmp.h>


static jmp_buf g_err_jmp;

// JS 翻译器最近一次词法/语法错误消息（UTF-32），供上层（REPL）取用；translate() 每次进入时清空
static wchar_t g_am_js_last_error[256] = {0};

const wchar_t *am_js_last_error(void) {
    return g_am_js_last_error;
}

// 将 UTF-32 字符串按 UTF-8 编码输出到文件流。
// 说明：newlib（ESP32）的宽字符文件流输出 vfwprintf/fwprintf 不可用，宽字符串会被按窄字节流吐出；
// 且 am_js2scm 属于解释器核心（amalgamation），不依赖宿主侧的 am_host 转换函数，故在此手写编码。
static void js_err_fputws(const wchar_t *ws, FILE *fp) {
    char buf[1024];
    size_t n = 0;
    for (const wchar_t *p = ws; *p && n + 4 < sizeof(buf); p++) {
        uint32_t cp = (uint32_t)*p;
        if (cp <= 0x7F) {
            buf[n++] = (char)cp;
        } else if (cp <= 0x7FF) {
            buf[n++] = (char)(0xC0 | (cp >> 6));
            buf[n++] = (char)(0x80 | (cp & 0x3F));
        } else if (cp <= 0xFFFF) {
            buf[n++] = (char)(0xE0 | (cp >> 12));
            buf[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            buf[n++] = (char)(0x80 | (cp & 0x3F));
        } else {
            buf[n++] = (char)(0xF0 | (cp >> 18));
            buf[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
            buf[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            buf[n++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    buf[n] = '\0';
    fputs(buf, fp);
}

static void *xrealloc(void *p, size_t sz) {
    void *q = realloc(p, sz);
    if (!q) {
        fprintf(stderr, "js2scm: out of memory\n");
        longjmp(g_err_jmp, 1);
    }
    return q;
}

/* ======================== String builder ======================== */

typedef struct {
    wchar_t *buf;
    size_t len;
    size_t cap;
} SB;

static void sb_init(SB *sb) {
    sb->buf = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static void sb_free(SB *sb) {
    free(sb->buf);
    sb->buf = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static void sb_grow(SB *sb, size_t need) {
    if (sb->len + need + 1 <= sb->cap) return;
    size_t newcap = sb->cap ? sb->cap * 2 : 256;
    while (newcap < sb->len + need + 1) newcap *= 2;
    sb->buf = (wchar_t *)xrealloc(sb->buf, newcap * sizeof(wchar_t));
    sb->cap = newcap;
}

static void sb_append(SB *sb, const wchar_t *s) {
    if (!s) return;
    size_t n = wcslen(s);
    sb_grow(sb, n);
    memcpy(sb->buf + sb->len, s, n * sizeof(wchar_t));
    sb->len += n;
    sb->buf[sb->len] = L'\0';
}

static void sb_appendf(SB *sb, const wchar_t *fmt, ...) {
    va_list ap;
    wchar_t tmp[1024];
    va_start(ap, fmt);
    int n = vswprintf(tmp, sizeof(tmp) / sizeof(tmp[0]), fmt, ap);
    va_end(ap);
    if (n > 0) sb_append(sb, tmp);
}

static wchar_t *xstrdup(const wchar_t *s) {
    if (!s) return NULL;
    size_t n = wcslen(s) + 1;
    wchar_t *p = (wchar_t *)malloc(n * sizeof(wchar_t));
    if (p) memcpy(p, s, n * sizeof(wchar_t));
    return p;
}

/* ======================== Lexer ======================== */

typedef enum {
    T_EOF,
    T_NUM, T_STR, T_ID,
    T_IF, T_ELSE, T_WHILE, T_VAR, T_FUNCTION, T_RETURN,
    T_CONTINUE, T_BREAK,
    T_TRUE, T_FALSE, T_NULL, T_UNDEFINED,
    T_LPAREN, T_RPAREN, T_LBRACKET, T_RBRACKET, T_LBRACE, T_RBRACE,
    T_SEMI, T_COMMA, T_COLON, T_QUESTION,
    T_NOT, T_PLUS, T_MINUS, T_MUL, T_DIV, T_MOD, T_ASSIGN, T_POW,
    T_LT, T_GT, T_EQ, T_LE, T_GE, T_OR, T_AND, T_INC, T_DEC, T_ARROW
} TokType;

typedef struct {
    TokType type;
    wchar_t *value;
    int line;
    int col;
} Token;

typedef struct {
    Token *data;
    int n;
    int cap;
} TokList;

static void tl_init(TokList *tl) {
    tl->data = NULL;
    tl->n = 0;
    tl->cap = 0;
}

static void tl_add(TokList *tl, TokType type, const wchar_t *value, int line, int col) {
    if (tl->n + 1 > tl->cap) {
        tl->cap = tl->cap ? tl->cap * 2 : 64;
        tl->data = (Token *)xrealloc(tl->data, tl->cap * sizeof(Token));
    }
    Token *t = &tl->data[tl->n++];
    t->type = type;
    t->value = value ? xstrdup(value) : NULL;
    t->line = line;
    t->col = col;
}

static void lex_error(int line, int col, const wchar_t *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    wchar_t msg[224];
    vswprintf(msg, sizeof(msg) / sizeof(msg[0]), fmt, ap);
    va_end(ap);
    // 记录完整错误消息（含位置），供 REPL 取用并显示到用户界面
    swprintf(g_am_js_last_error, sizeof(g_am_js_last_error) / sizeof(g_am_js_last_error[0]),
             L"词法错误 @ %d:%d: ", line, col);
    wcsncat(g_am_js_last_error, msg,
            sizeof(g_am_js_last_error) / sizeof(g_am_js_last_error[0]) - wcslen(g_am_js_last_error) - 1);
    fprintf(stderr, "词法错误 @ %d:%d: ", line, col);
    js_err_fputws(msg, stderr);
    fprintf(stderr, "\n");
    longjmp(g_err_jmp, 1);
}

static Token *tokenize(const wchar_t *src, int *out_count) {
    TokList tl;
    tl_init(&tl);
    int i = 0;
    int len = (int)wcslen(src);
    int line = 1, col = 1;
    int parenDepth = 0, bracketDepth = 0, braceDepth = 0;

#define PEEK(off) ((i + (off) < len) ? src[i + (off)] : L'\0')

    while (i < len) {
        wchar_t ch = PEEK(0);

        if (ch == L' ' || ch == L'\t' || ch == L'\r') {
            i++; col++;
            continue;
        }

        if (ch == L'\n') {
            i++; line++; col = 1;
            if (parenDepth == 0 && bracketDepth == 0) {
                if (tl.n == 0 || tl.data[tl.n - 1].type != T_SEMI)
                    tl_add(&tl, T_SEMI, NULL, line, col);
            }
            continue;
        }

        /* comments */
        if (ch == L'/' && PEEK(1) == L'/') {
            i += 2; col += 2;
            while (i < len && PEEK(0) != L'\n') { i++; col++; }
            continue;
        }
        if (ch == L'/' && PEEK(1) == L'*') {
            i += 2; col += 2;
            while (i < len) {
                if (PEEK(0) == L'*' && PEEK(1) == L'/') {
                    i += 2; col += 2;
                    break;
                }
                if (PEEK(0) == L'\n') { i++; line++; col = 1; }
                else { i++; col++; }
            }
            continue;
        }

        /* number */
        if (iswdigit((wint_t)ch) || (ch == L'.' && iswdigit((wint_t)PEEK(1)))) {
            int startLine = line, startCol = col;
            SB num; sb_init(&num);
            int hasDigits = 0;
            while (i < len && iswdigit((wint_t)PEEK(0))) {
                sb_appendf(&num, L"%lc", (wint_t)PEEK(0));
                i++; col++; hasDigits = 1;
            }
            if (PEEK(0) == L'.') {
                sb_appendf(&num, L"%lc", (wint_t)PEEK(0));
                i++; col++;
                while (i < len && iswdigit((wint_t)PEEK(0))) {
                    sb_appendf(&num, L"%lc", (wint_t)PEEK(0));
                    i++; col++; hasDigits = 1;
                }
            }
            if (PEEK(0) == L'e' || PEEK(0) == L'E') {
                wchar_t next = PEEK(1);
                if (iswdigit((wint_t)next) ||
                    ((next == L'+' || next == L'-') && iswdigit((wint_t)PEEK(2)))) {
                    sb_appendf(&num, L"%lc", (wint_t)PEEK(0));
                    i++; col++;
                    if (next == L'+' || next == L'-') {
                        sb_appendf(&num, L"%lc", (wint_t)PEEK(0));
                        i++; col++;
                    }
                    while (i < len && iswdigit((wint_t)PEEK(0))) {
                        sb_appendf(&num, L"%lc", (wint_t)PEEK(0));
                        i++; col++; hasDigits = 1;
                    }
                }
            }
            if (!hasDigits) lex_error(line, col, L"非法数字 \"%ls\"", num.buf);
            tl_add(&tl, T_NUM, num.buf, startLine, startCol);
            sb_free(&num);
            continue;
        }

        /* string */
        if (ch == L'"') {
            int startLine = line, startCol = col;
            i++; col++;
            SB str; sb_init(&str);
            while (i < len && PEEK(0) != L'"') {
                if (PEEK(0) == L'\\') {
                    sb_appendf(&str, L"%lc", (wint_t)PEEK(0));
                    i++; col++;
                    if (i < len) {
                        sb_appendf(&str, L"%lc", (wint_t)PEEK(0));
                        if (PEEK(0) == L'\n') { line++; col = 1; }
                        else { col++; }
                        i++;
                    }
                    continue;
                }
                sb_appendf(&str, L"%lc", (wint_t)PEEK(0));
                if (PEEK(0) == L'\n') { line++; col = 1; }
                else { col++; }
                i++;
            }
            if (PEEK(0) != L'"') lex_error(startLine, startCol, L"未终止的字符串");
            i++; col++;
            tl_add(&tl, T_STR, str.buf, startLine, startCol);
            sb_free(&str);
            continue;
        }

        /* identifier / keyword */
        /* 按 JS 规范接受非 ASCII 字符（如中文标识符）；newlib 的 iswalpha/iswalnum
           在 C locale 下仅覆盖 ASCII，故显式放宽为 ch >= 0x80 */
        if (iswalpha((wint_t)ch) || ch == L'_' || ch == L'$' || ch >= 0x80) {
            int startLine = line, startCol = col;
            SB id; sb_init(&id);
            while (i < len && (iswalnum((wint_t)PEEK(0)) || PEEK(0) == L'_' || PEEK(0) == L'$' || PEEK(0) == L'.' || PEEK(0) >= 0x80)) {
                sb_appendf(&id, L"%lc", (wint_t)PEEK(0));
                i++; col++;
            }
            TokType type = T_ID;
            if (wcscmp(id.buf, L"if") == 0) type = T_IF;
            else if (wcscmp(id.buf, L"else") == 0) type = T_ELSE;
            else if (wcscmp(id.buf, L"while") == 0) type = T_WHILE;
            else if (wcscmp(id.buf, L"var") == 0) type = T_VAR;
            else if (wcscmp(id.buf, L"function") == 0) type = T_FUNCTION;
            else if (wcscmp(id.buf, L"return") == 0) type = T_RETURN;
            else if (wcscmp(id.buf, L"continue") == 0) type = T_CONTINUE;
            else if (wcscmp(id.buf, L"break") == 0) type = T_BREAK;
            else if (wcscmp(id.buf, L"true") == 0) type = T_TRUE;
            else if (wcscmp(id.buf, L"false") == 0) type = T_FALSE;
            else if (wcscmp(id.buf, L"null") == 0) type = T_NULL;
            else if (wcscmp(id.buf, L"undefined") == 0) type = T_UNDEFINED;
            tl_add(&tl, type, id.buf, startLine, startCol);
            sb_free(&id);
            continue;
        }

        /* two-char operators */
        if (ch == L'=' && PEEK(1) == L'=') { i += 2; col += 2; tl_add(&tl, T_EQ, NULL, line, col - 2); continue; }
        if (ch == L'=' && PEEK(1) == L'>') { i += 2; col += 2; tl_add(&tl, T_ARROW, NULL, line, col - 2); continue; }
        if (ch == L'<' && PEEK(1) == L'=') { i += 2; col += 2; tl_add(&tl, T_LE, NULL, line, col - 2); continue; }
        if (ch == L'>' && PEEK(1) == L'=') { i += 2; col += 2; tl_add(&tl, T_GE, NULL, line, col - 2); continue; }
        if (ch == L'|' && PEEK(1) == L'|') { i += 2; col += 2; tl_add(&tl, T_OR, NULL, line, col - 2); continue; }
        if (ch == L'&' && PEEK(1) == L'&') { i += 2; col += 2; tl_add(&tl, T_AND, NULL, line, col - 2); continue; }
        if (ch == L'+' && PEEK(1) == L'+') { i += 2; col += 2; tl_add(&tl, T_INC, NULL, line, col - 2); continue; }
        if (ch == L'-' && PEEK(1) == L'-') { i += 2; col += 2; tl_add(&tl, T_DEC, NULL, line, col - 2); continue; }

        /* single-char punctuation / operators */
        TokType stype = T_EOF;
        switch (ch) {
            case L'(': stype = T_LPAREN;   parenDepth++;   break;
            case L')': stype = T_RPAREN;   parenDepth--; if (parenDepth < 0) parenDepth = 0; break;
            case L'[': stype = T_LBRACKET; bracketDepth++; break;
            case L']': stype = T_RBRACKET; bracketDepth--; if (bracketDepth < 0) bracketDepth = 0; break;
            case L'{': stype = T_LBRACE;   braceDepth++;   break;
            case L'}': stype = T_RBRACE;   braceDepth--; if (braceDepth < 0) braceDepth = 0; break;
            case L';': stype = T_SEMI;     break;
            case L',': stype = T_COMMA;    break;
            case L':': stype = T_COLON;    break;
            case L'?': stype = T_QUESTION; break;
            case L'!': stype = T_NOT;      break;
            case L'+': stype = T_PLUS;     break;
            case L'-': stype = T_MINUS;    break;
            case L'*': stype = T_MUL;      break;
            case L'/': stype = T_DIV;      break;
            case L'%': stype = T_MOD;      break;
            case L'=': stype = T_ASSIGN;   break;
            case L'^': stype = T_POW;      break;
            case L'<': stype = T_LT;       break;
            case L'>': stype = T_GT;       break;
        }
        if (stype != T_EOF) {
            tl_add(&tl, stype, NULL, line, col);
            i++; col++;
            continue;
        }

        lex_error(line, col, L"非法字符 \"%lc\"", (wint_t)ch);
    }

    tl_add(&tl, T_EOF, NULL, line, col);
    *out_count = tl.n;
    return tl.data;

#undef PEEK
}

/* ======================== Parser ======================== */

typedef enum {
    N_PROGRAM, N_BLOCK,
    N_ID, N_NUMBER, N_STRING, N_BOOL, N_NULL, N_UNDEFINED,
    N_LIST, N_CALL, N_INDEX,
    N_UNARY, N_BINARY, N_TERNARY, N_ASSIGN,
    N_VARDEF, N_FUNCDEF, N_LAMBDA,
    N_IF, N_WHILE, N_RETURN,
    N_CONTINUE, N_BREAK,
    N_PREINC, N_POSTINC, N_PREDEC, N_POSTDEC
} NodeType;

typedef struct Node Node;
struct Node {
    NodeType type;
    wchar_t *value;              /* id/number/string/bool/op/varname/funcname */
    Node **items;
    int n_items;
    int cap_items;
    Node *left;
    Node *right;
    Node *cond;
    Node *trueBranch;
    Node *falseBranch;
    Node *expr;
    Node *obj;
    Node *idx;
    wchar_t **params;
    int n_params;
    int cap_params;
};

static Node *new_node(NodeType type) {
    Node *n = (Node *)calloc(1, sizeof(Node));
    n->type = type;
    return n;
}

typedef struct {
    Node **data;
    int n;
    int cap;
} NodeList;

static void nl_init(NodeList *l) {
    l->data = NULL;
    l->n = 0;
    l->cap = 0;
}

static void nl_add(NodeList *l, Node *n) {
    if (l->n + 1 > l->cap) {
        l->cap = l->cap ? l->cap * 2 : 4;
        l->data = (Node **)xrealloc(l->data, l->cap * sizeof(Node *));
    }
    l->data[l->n++] = n;
}

typedef struct {
    wchar_t **data;
    int n;
    int cap;
} ParamList;

static void pl_init(ParamList *p) {
    p->data = NULL;
    p->n = 0;
    p->cap = 0;
}

static void pl_add(ParamList *p, const wchar_t *s) {
    if (p->n + 1 > p->cap) {
        p->cap = p->cap ? p->cap * 2 : 4;
        p->data = (wchar_t **)xrealloc(p->data, p->cap * sizeof(wchar_t *));
    }
    p->data[p->n++] = xstrdup(s);
}

static Token *tokens = NULL;
static int pos = 0;

static Token *peek(void) { return &tokens[pos]; }
static int at_end(void) { return peek()->type == T_EOF; }
static Token *previous(void) { return &tokens[pos - 1]; }
static Token *advance_tok(void) { if (!at_end()) pos++; return previous(); }

static int match(TokType t) {
    if (peek()->type == t) { pos++; return 1; }
    return 0;
}

static int match_any(int n, ...) {
    va_list ap;
    va_start(ap, n);
    for (int i = 0; i < n; i++) {
        TokType t = va_arg(ap, TokType);
        if (peek()->type == t) { pos++; va_end(ap); return t; }
    }
    va_end(ap);
    return T_EOF;
}

static Token *expect(TokType t) {
    if (peek()->type != t) {
        Token *tok = peek();
        swprintf(g_am_js_last_error, sizeof(g_am_js_last_error) / sizeof(g_am_js_last_error[0]),
                 L"语法错误 @ %d:%d: 意外的记号", tok->line, tok->col);
        fprintf(stderr, "语法错误 @ %d:%d: 期望 \"%d\"，得到 \"%d\"\n",
                tok->line, tok->col, t, tok->type);
        longjmp(g_err_jmp, 1);
    }
    return advance_tok();
}

static void parse_error(const wchar_t *fmt, ...) {
    Token *tok = peek();
    va_list ap;
    va_start(ap, fmt);
    wchar_t msg[224];
    vswprintf(msg, sizeof(msg) / sizeof(msg[0]), fmt, ap);
    va_end(ap);
    // 记录完整错误消息（含位置），供 REPL 取用并显示到用户界面
    swprintf(g_am_js_last_error, sizeof(g_am_js_last_error) / sizeof(g_am_js_last_error[0]),
             L"语法错误 @ %d:%d: ", tok->line, tok->col);
    wcsncat(g_am_js_last_error, msg,
            sizeof(g_am_js_last_error) / sizeof(g_am_js_last_error[0]) - wcslen(g_am_js_last_error) - 1);
    fprintf(stderr, "语法错误 @ %d:%d: ", tok->line, tok->col);
    js_err_fputws(msg, stderr);
    fprintf(stderr, "\n");
    longjmp(g_err_jmp, 1);
}

/* Forward declarations */
static Node *parse_expr(void);
static Node *am_js2scm__parse_term(void);
static Node *parse_unary(void);
static Node *parse_postfix(void);
static Node *parse_primary(void);
static Node *parse_paren_or_lambda(void);
static Node *parse_block(void);

static NodeList parse_expr_seq(TokType stop);
static ParamList parse_param_seq(void);
static NodeList parse_term_seq(TokType stop);

static Node *parse_ternary_from(Node *left);
static Node *parse_or_from(Node *left);
static Node *parse_and_from(Node *left);
static Node *parse_cmp_from(Node *left);
static Node *parse_add_from(Node *left);
static Node *parse_mul_from(Node *left);
static Node *parse_exp_from(Node *left);

static Node *parse_and(void);
static Node *parse_cmp(void);
static Node *parse_add(void);
static Node *parse_mul(void);
static Node *parse_exp(void);

static Node *new_binary(const wchar_t *op, Node *l, Node *r) {
    Node *n = new_node(N_BINARY);
    n->value = xstrdup(op);
    n->left = l;
    n->right = r;
    return n;
}

static Node *parse_program(void) {
    Node *n = new_node(N_PROGRAM);
    NodeList body = parse_expr_seq(T_EOF);
    expect(T_EOF);
    n->items = body.data;
    n->n_items = body.n;
    n->cap_items = body.cap;
    return n;
}

static NodeList parse_expr_seq(TokType stop) {
    NodeList list;
    nl_init(&list);
    while (peek()->type != stop && peek()->type != T_EOF) {
        if (match(T_SEMI)) continue;
        Node *e = parse_expr();
        if (e) nl_add(&list, e);
        while (match(T_SEMI)) {}
    }
    return list;
}

static Node *parse_block(void) {
    expect(T_LBRACE);
    NodeList body = parse_expr_seq(T_RBRACE);
    expect(T_RBRACE);
    Node *n = new_node(N_BLOCK);
    n->items = body.data;
    n->n_items = body.n;
    n->cap_items = body.cap;
    return n;
}

static Node *parse_function(void) {
    advance_tok(); /* function */
    Token *name = expect(T_ID);
    expect(T_LPAREN);
    ParamList params = parse_param_seq();
    expect(T_RPAREN);
    expect(T_LBRACE);
    NodeList body = parse_expr_seq(T_RBRACE);
    expect(T_RBRACE);
    Node *n = new_node(N_FUNCDEF);
    n->value = xstrdup(name->value);
    n->params = params.data;
    n->n_params = params.n;
    n->cap_params = params.cap;
    n->items = body.data;
    n->n_items = body.n;
    n->cap_items = body.cap;
    return n;
}

static Node *parse_if(void) {
    advance_tok(); /* if */
    expect(T_LPAREN);
    Node *cond = am_js2scm__parse_term();
    expect(T_RPAREN);
    Node *thenBranch = parse_block();
    Node *elseBranch = NULL;
    while (match(T_SEMI)) {}
    if (match(T_ELSE)) {
        elseBranch = (peek()->type == T_IF) ? parse_if() : parse_block();
    }
    Node *n = new_node(N_IF);
    n->cond = cond;
    n->trueBranch = thenBranch;
    n->falseBranch = elseBranch;
    return n;
}

static Node *parse_while(void) {
    advance_tok(); /* while */
    expect(T_LPAREN);
    Node *cond = am_js2scm__parse_term();
    expect(T_RPAREN);
    Node *body = parse_block();
    Node *n = new_node(N_WHILE);
    n->cond = cond;
    n->expr = body;
    return n;
}

static Node *parse_vardef(void) {
    advance_tok(); /* var */
    Token *name = expect(T_ID);
    Node *value = new_node(N_UNDEFINED);
    if (match(T_ASSIGN)) value = am_js2scm__parse_term();
    Node *n = new_node(N_VARDEF);
    n->value = xstrdup(name->value);
    n->right = value;
    return n;
}

static Node *parse_return(void) {
    advance_tok(); /* return */
    Node *n = new_node(N_RETURN);
    TokType t = peek()->type;
    if (t == T_SEMI || t == T_RBRACE || t == T_EOF) return n;
    n->expr = am_js2scm__parse_term();
    return n;
}

static ParamList parse_param_seq(void) {
    ParamList p;
    pl_init(&p);
    if (peek()->type == T_ID) {
        pl_add(&p, advance_tok()->value);
        while (match(T_COMMA)) {
            pl_add(&p, expect(T_ID)->value);
        }
    }
    return p;
}

static NodeList parse_term_seq(TokType stop) {
    NodeList list;
    nl_init(&list);
    if (peek()->type != stop) {
        while (1) {
            nl_add(&list, am_js2scm__parse_term());
            if (match(T_COMMA)) continue;
            break;
        }
    }
    expect(stop);
    return list;
}

static Node *parse_expr(void) {
    switch (peek()->type) {
        case T_LBRACE:    return parse_block();
        case T_FUNCTION:  return parse_function();
        case T_IF:        return parse_if();
        case T_WHILE:     return parse_while();
        case T_VAR:       return parse_vardef();
        case T_RETURN:    return parse_return();
        case T_CONTINUE:  advance_tok(); return new_node(N_CONTINUE);
        case T_BREAK:     advance_tok(); return new_node(N_BREAK);
        default:          return am_js2scm__parse_term();
    }
}

static Node *am_js2scm__parse_term(void) {
    TokType t = peek()->type;
    if (t == T_NOT || t == T_MINUS || t == T_INC || t == T_DEC) {
        Node *u = parse_unary();
        return parse_ternary_from(u);
    }
    Node *left = parse_postfix();
    if (match(T_ASSIGN)) {
        Node *right = am_js2scm__parse_term();
        Node *n = new_node(N_ASSIGN);
        n->left = left;
        n->right = right;
        return n;
    }
    return parse_ternary_from(left);
}

static Node *parse_ternary_from(Node *left) {
    Node *cond = parse_or_from(left);
    if (match(T_QUESTION)) {
        Node *trueBranch = am_js2scm__parse_term();
        expect(T_COLON);
        Node *falseBranch = am_js2scm__parse_term();
        Node *n = new_node(N_TERNARY);
        n->cond = cond;
        n->trueBranch = trueBranch;
        n->falseBranch = falseBranch;
        return n;
    }
    return cond;
}

static Node *parse_or_from(Node *left) {
    Node *node = parse_and_from(left);
    while (match(T_OR)) {
        node = new_binary(L"or", node, parse_and());
    }
    return node;
}
static Node *parse_and_from(Node *left) {
    Node *node = parse_cmp_from(left);
    while (match(T_AND)) {
        node = new_binary(L"and", node, parse_cmp());
    }
    return node;
}
static const wchar_t *cmp_op(TokType t) {
    switch (t) {
        case T_GT: return L">";
        case T_LT: return L"<";
        case T_EQ: return L"==";
        case T_LE: return L"<=";
        case T_GE: return L">=";
        default: return L"?";
    }
}
static Node *parse_cmp_from(Node *left) {
    Node *node = parse_add_from(left);
    TokType op;
    while ((op = match_any(5, T_GT, T_LT, T_EQ, T_LE, T_GE)) != T_EOF) {
        node = new_binary(cmp_op(op), node, parse_add());
    }
    return node;
}
static Node *parse_add_from(Node *left) {
    Node *node = parse_mul_from(left);
    TokType op;
    while ((op = match_any(2, T_PLUS, T_MINUS)) != T_EOF) {
        const wchar_t *s = (op == T_PLUS) ? L"+" : L"-";
        node = new_binary(s, node, parse_mul());
    }
    return node;
}
static Node *parse_mul_from(Node *left) {
    Node *node = parse_exp_from(left);
    TokType op;
    while ((op = match_any(3, T_MUL, T_DIV, T_MOD)) != T_EOF) {
        const wchar_t *s;
        if (op == T_MOD) s = L"mod";
        else if (op == T_MUL) s = L"*";
        else s = L"/";
        node = new_binary(s, node, parse_exp());
    }
    return node;
}
static Node *parse_exp_from(Node *left) {
    if (match(T_POW)) {
        return new_binary(L"pow", left, parse_exp());
    }
    return left;
}

static Node *parse_and(void) { return parse_and_from(parse_cmp()); }
static Node *parse_cmp(void) { return parse_cmp_from(parse_add()); }
static Node *parse_add(void) { return parse_add_from(parse_mul()); }
static Node *parse_mul(void) { return parse_mul_from(parse_exp()); }
static Node *parse_exp(void) { return parse_exp_from(parse_unary()); }

static Node *parse_unary(void) {
    if (match(T_NOT)) {
        Node *n = new_node(N_UNARY);
        n->value = xstrdup(L"not");
        n->expr = parse_unary();
        return n;
    }
    if (match(T_MINUS)) {
        Node *e = parse_unary();
        if (e->type == N_NUMBER && e->value[0] != L'-') {
            size_t len = wcslen(e->value);
            wchar_t *v = (wchar_t *)malloc((len + 2) * sizeof(wchar_t));
            v[0] = L'-';
            wcscpy(v + 1, e->value);
            free(e->value);
            e->value = v;
            return e;
        }
        Node *n = new_node(N_UNARY);
        n->value = xstrdup(L"-");
        n->expr = e;
        return n;
    }
    if (match(T_INC)) {
        Node *n = new_node(N_PREINC);
        n->expr = parse_unary();
        return n;
    }
    if (match(T_DEC)) {
        Node *n = new_node(N_PREDEC);
        n->expr = parse_unary();
        return n;
    }
    return parse_postfix();
}

static Node *parse_postfix(void) {
    Node *node = parse_primary();
    while (1) {
        if (match(T_LPAREN)) {
            Node *call = new_node(N_CALL);
            call->expr = node;
            NodeList args = parse_term_seq(T_RPAREN);
            call->items = args.data;
            call->n_items = args.n;
            call->cap_items = args.cap;
            node = call;
        } else if (match(T_LBRACKET)) {
            Node *idx = new_node(N_INDEX);
            idx->obj = node;
            idx->idx = am_js2scm__parse_term();
            expect(T_RBRACKET);
            node = idx;
        } else if (match(T_INC)) {
            Node *n = new_node(N_POSTINC);
            n->expr = node;
            node = n;
        } else if (match(T_DEC)) {
            Node *n = new_node(N_POSTDEC);
            n->expr = node;
            node = n;
        } else {
            break;
        }
    }
    return node;
}

static Node *parse_primary(void) {
    Token *tok = peek();
    switch (tok->type) {
        case T_ID: {
            advance_tok();
            Node *n = new_node(N_ID);
            n->value = xstrdup(tok->value);
            return n;
        }
        case T_NUM: {
            advance_tok();
            Node *n = new_node(N_NUMBER);
            n->value = xstrdup(tok->value);
            return n;
        }
        case T_STR: {
            advance_tok();
            Node *n = new_node(N_STRING);
            n->value = xstrdup(tok->value);
            return n;
        }
        case T_TRUE:  advance_tok(); { Node *n = new_node(N_BOOL); n->value = xstrdup(L"true"); return n; }
        case T_FALSE: advance_tok(); { Node *n = new_node(N_BOOL); n->value = xstrdup(L"false"); return n; }
        case T_NULL:      advance_tok(); return new_node(N_NULL);
        case T_UNDEFINED: advance_tok(); return new_node(N_UNDEFINED);
        case T_LBRACKET: {
            advance_tok();
            NodeList items = parse_term_seq(T_RBRACKET);
            Node *n = new_node(N_LIST);
            n->items = items.data;
            n->n_items = items.n;
            n->cap_items = items.cap;
            return n;
        }
        case T_LPAREN:
            return parse_paren_or_lambda();
        default:
            parse_error(L"意外的 \"%ls\"", tok->value ? tok->value : L"");
            return NULL;
    }
}

static Node *parse_paren_or_lambda(void) {
    int saved = pos;
    expect(T_LPAREN);
    ParamList params = parse_param_seq();
    if (peek()->type == T_RPAREN && tokens[pos + 1].type == T_ARROW) {
        advance_tok(); /* ) */
        advance_tok(); /* => */
        expect(T_LBRACE);
        NodeList body = parse_expr_seq(T_RBRACE);
        expect(T_RBRACE);
        Node *lam = new_node(N_LAMBDA);
        lam->params = params.data;
        lam->n_params = params.n;
        lam->cap_params = params.cap;
        lam->items = body.data;
        lam->n_items = body.n;
        lam->cap_items = body.cap;
        return lam;
    }
    /* not a lambda: backtrack and parse parenthesised expression */
    pos = saved;
    expect(T_LPAREN);
    Node *expr = am_js2scm__parse_term();
    expect(T_RPAREN);
    return expr;
}

/* ======================== Emitter helpers ======================== */

static void emit_node(Node *node, SB *sb);

typedef struct {
    wchar_t **strs;
    int n;
} StrList;

static StrList emit_children(Node **items, int n_items) {
    StrList sl;
    sl.strs = (wchar_t **)malloc((n_items > 0 ? n_items : 1) * sizeof(wchar_t *));
    sl.n = 0;
    for (int i = 0; i < n_items; i++) {
        SB t; sb_init(&t);
        emit_node(items[i], &t);
        if (t.len > 0) {
            sl.strs[sl.n++] = t.buf;
        } else {
            sb_free(&t);
        }
    }
    return sl;
}

static void free_strlist(StrList *sl) {
    for (int i = 0; i < sl->n; i++) free(sl->strs[i]);
    free(sl->strs);
    sl->strs = NULL;
    sl->n = 0;
}

static void emit_params(Node *n, SB *sb) {
    for (int i = 0; i < n->n_params; i++) {
        if (i) sb_append(sb, L" ");
        sb_append(sb, n->params[i]);
    }
}

static void emit_incdec(Node *n, SB *sb, const wchar_t *op) {
    sb_append(sb, L"(set! ");
    emit_node(n->expr, sb);
    sb_append(sb, L" (");
    sb_append(sb, op);
    sb_append(sb, L" ");
    emit_node(n->expr, sb);
    sb_append(sb, L" 1))");
}

/* ======================== Emitter ======================== */

static void emit_node_qq(Node *node, SB *sb);

static void emit_node(Node *node, SB *sb) {
    if (!node) return;
    switch (node->type) {
        case N_PROGRAM: {
            StrList terms = emit_children(node->items, node->n_items);
            sb_append(sb, L"((lambda ()");
            if (terms.n > 0) {
                sb_append(sb, L"\n  ");
                for (int i = 0; i < terms.n; i++) {
                    sb_append(sb, terms.strs[i]);
                    if (i + 1 < terms.n) sb_append(sb, L"\n  ");
                }
            }
            sb_append(sb, L"))\n");
            free_strlist(&terms);
            break;
        }
        case N_BLOCK: {
            StrList parts = emit_children(node->items, node->n_items);
            if (parts.n == 0) {
                sb_append(sb, L"{}");
            } else {
                sb_append(sb, L"{");
                for (int i = 0; i < parts.n; i++) {
                    sb_append(sb, parts.strs[i]);
                    if (i + 1 < parts.n) sb_append(sb, L" ");
                }
                sb_append(sb, L"}");
            }
            free_strlist(&parts);
            break;
        }
        case N_ID:       sb_append(sb, node->value); break;
        case N_NUMBER:   sb_append(sb, node->value); break;
        case N_STRING:   sb_append(sb, L"\""); sb_append(sb, node->value); sb_append(sb, L"\""); break;
        case N_BOOL:     sb_append(sb, wcscmp(node->value, L"true") == 0 ? L"#t" : L"#f"); break;
        case N_NULL:     sb_append(sb, L"#null"); break;
        case N_UNDEFINED:sb_append(sb, L"#undefined"); break;
        case N_LIST: {
            if (node->n_items == 0) {
                sb_append(sb, L"`()");
            } else {
                sb_append(sb, L"`(");
                for (int i = 0; i < node->n_items; i++) {
                    if (i) sb_append(sb, L" ");
                    emit_node_qq(node->items[i], sb);
                }
                sb_append(sb, L")");
            }
            break;
        }
        case N_CALL: {
            sb_append(sb, L"(");
            emit_node(node->expr, sb);
            for (int i = 0; i < node->n_items; i++) {
                sb_append(sb, L" ");
                emit_node(node->items[i], sb);
            }
            sb_append(sb, L")");
            break;
        }
        case N_INDEX: {
            sb_append(sb, L"(get_item ");
            emit_node(node->obj, sb);
            sb_append(sb, L" ");
            emit_node(node->idx, sb);
            sb_append(sb, L")");
            break;
        }
        case N_UNARY: {
            sb_append(sb, L"(");
            sb_append(sb, node->value);
            sb_append(sb, L" ");
            emit_node(node->expr, sb);
            sb_append(sb, L")");
            break;
        }
        case N_BINARY: {
            sb_append(sb, L"(");
            sb_append(sb, node->value);
            sb_append(sb, L" ");
            emit_node(node->left, sb);
            sb_append(sb, L" ");
            emit_node(node->right, sb);
            sb_append(sb, L")");
            break;
        }
        case N_TERNARY: {
            sb_append(sb, L"(if ");
            emit_node(node->cond, sb);
            sb_append(sb, L" ");
            emit_node(node->trueBranch, sb);
            sb_append(sb, L" ");
            emit_node(node->falseBranch, sb);
            sb_append(sb, L")");
            break;
        }
        case N_ASSIGN: {
            Node *rhs = node->right;
            if (node->left->type == N_INDEX) {
                sb_append(sb, L"(set_item! ");
                emit_node(node->left->obj, sb);
                sb_append(sb, L" ");
                emit_node(node->left->idx, sb);
                sb_append(sb, L" ");
                emit_node(rhs, sb);
                sb_append(sb, L")");
            } else {
                sb_append(sb, L"(set! ");
                emit_node(node->left, sb);
                sb_append(sb, L" ");
                emit_node(rhs, sb);
                sb_append(sb, L")");
            }
            break;
        }
        case N_VARDEF: {
            sb_append(sb, L"(define ");
            sb_append(sb, node->value);
            sb_append(sb, L" ");
            emit_node(node->right, sb);
            sb_append(sb, L")");
            break;
        }
        case N_FUNCDEF: {
            sb_append(sb, L"(define ");
            sb_append(sb, node->value);
            sb_append(sb, L" (lambda (");
            emit_params(node, sb);
            sb_append(sb, L")");
            StrList body = emit_children(node->items, node->n_items);
            if (body.n > 0) {
                sb_append(sb, L" ");
                for (int i = 0; i < body.n; i++) {
                    sb_append(sb, body.strs[i]);
                    if (i + 1 < body.n) sb_append(sb, L" ");
                }
            }
            sb_append(sb, L"))");
            free_strlist(&body);
            break;
        }
        case N_LAMBDA: {
            sb_append(sb, L"(lambda (");
            emit_params(node, sb);
            sb_append(sb, L")");
            StrList body = emit_children(node->items, node->n_items);
            if (body.n > 0) {
                sb_append(sb, L" ");
                for (int i = 0; i < body.n; i++) {
                    sb_append(sb, body.strs[i]);
                    if (i + 1 < body.n) sb_append(sb, L" ");
                }
            }
            sb_append(sb, L")");
            free_strlist(&body);
            break;
        }
        case N_IF: {
            sb_append(sb, L"(if ");
            emit_node(node->cond, sb);
            sb_append(sb, L" ");
            emit_node(node->trueBranch, sb);
            if (node->falseBranch) {
                sb_append(sb, L" ");
                emit_node(node->falseBranch, sb);
            }
            sb_append(sb, L")");
            break;
        }
        case N_WHILE: {
            sb_append(sb, L"(while ");
            emit_node(node->cond, sb);
            sb_append(sb, L" ");
            emit_node(node->expr, sb);
            sb_append(sb, L")");
            break;
        }
        case N_RETURN: {
            if (node->expr) emit_node(node->expr, sb);
            break;
        }
        case N_CONTINUE: sb_append(sb, L"continue"); break;
        case N_BREAK:    sb_append(sb, L"break"); break;
        case N_PREINC:
        case N_POSTINC:  emit_incdec(node, sb, L"+"); break;
        case N_PREDEC:
        case N_POSTDEC:  emit_incdec(node, sb, L"-"); break;
    }
}

/* 在 quasiquote 上下文中输出节点。
 * JS 列表转换为 Scheme quasiquote 列表：字面量保持原样，
 * 变量标识符前加 "," 进行去引用。 */
static void emit_node_qq(Node *node, SB *sb) {
    if (!node) return;
    switch (node->type) {
        case N_ID:
            sb_append(sb, L",");
            sb_append(sb, node->value);
            break;
        case N_NUMBER:
            sb_append(sb, node->value);
            break;
        case N_STRING:
            sb_append(sb, L"\"");
            sb_append(sb, node->value);
            sb_append(sb, L"\"");
            break;
        case N_BOOL:
            sb_append(sb, wcscmp(node->value, L"true") == 0 ? L"#t" : L"#f");
            break;
        case N_NULL:
            sb_append(sb, L"#null");
            break;
        case N_UNDEFINED:
            sb_append(sb, L"#undefined");
            break;
        case N_LIST: {
            if (node->n_items == 0) {
                sb_append(sb, L"`()");
            } else {
                sb_append(sb, L"`(");
                for (int i = 0; i < node->n_items; i++) {
                    if (i) sb_append(sb, L" ");
                    emit_node_qq(node->items[i], sb);
                }
                sb_append(sb, L")");
            }
            break;
        }
        default:
            sb_append(sb, L",");
            emit_node(node, sb);
            break;
    }
}

/* ======================== Cleanup helpers ======================== */

static void free_node(Node *n) {
    if (!n) return;
    free(n->value);
    for (int i = 0; i < n->n_items; i++) free_node(n->items[i]);
    free(n->items);
    for (int i = 0; i < n->n_params; i++) free(n->params[i]);
    free(n->params);
    free_node(n->left);
    free_node(n->right);
    free_node(n->cond);
    free_node(n->trueBranch);
    free_node(n->falseBranch);
    free_node(n->expr);
    free_node(n->obj);
    free_node(n->idx);
    free(n);
}

static void free_tokens(Token *toks, int count) {
    if (!toks) return;
    for (int i = 0; i < count; i++) free(toks[i].value);
    free(toks);
}

/* ======================== Wide-char translate ======================== */

static wchar_t *translate(const wchar_t *js_source) {
    if (!js_source) return NULL;
    g_am_js_last_error[0] = L'\0'; // 清空上一次的错误消息
    int ntok = 0;
    Node *ast = NULL;
    if (setjmp(g_err_jmp)) {
        if (tokens) {
            free_tokens(tokens, ntok);
            tokens = NULL;
            pos = 0;
        }
        if (ast) {
            free_node(ast);
            ast = NULL;
        }
        return NULL;
    }
    tokens = tokenize(js_source, &ntok);
    pos = 0;
    ast = parse_program();
    SB out;
    sb_init(&out);
    emit_node(ast, &out);
    free_node(ast);
    free_tokens(tokens, ntok);
    tokens = NULL;
    pos = 0;
    return out.buf;
}

/* 将 JS 代码字符串翻译成 Scheme 代码字符串。
 * 输入/输出均为宽字符字符串；返回的指针由调用者负责 free。 */
wchar_t *am_js_to_scheme(const wchar_t *js_source) {
    if (!js_source) return NULL;
    return translate(js_source);
}
/* ===== end:   src/am_js2scm.c ===== */

/* ===== begin: src/am_process.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>



// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 将proc->heap中所有对象标记为静态对象（通常用于从模块加载的初始AST数据）
static void set_all_heap_objects_static(am_handle_t handle, am_value_t value, void *user_data) {
    (void)handle;
    (void)user_data;
    if (am_value_is_ptr(value)) {
        am_object_t *obj = am_value_to_ptr(value);
        if (obj != NULL) {
            am_object_set_static(obj, 0);
        }
    }
}




// ===============================================================================
// dynamic-wind 内部辅助函数
// ===============================================================================

// 从 dynamic-wind 条目中读取 before/after/mark/saved
static inline am_handle_t am_process__dynamic_wind_entry_before(am_list_t *entry) {
    if (!entry || entry->length < 4) return AM_HANDLE_NULL;
    return am_value_to_handle(entry->children[0]);
}

static inline am_handle_t am_process__dynamic_wind_entry_after(am_list_t *entry) {
    if (!entry || entry->length < 4) return AM_HANDLE_NULL;
    return am_value_to_handle(entry->children[1]);
}

static inline am_uint_t dynamic_wind_entry_mark(am_list_t *entry) {
    if (!entry || entry->length < 4) return 0;
    return am_value_to_uint(entry->children[2]);
}

static inline am_value_t am_process__dynamic_wind_entry_saved(am_list_t *entry) {
    if (!entry || entry->length < 4) return AM_VALUE_UNDEFINED;
    return entry->children[3];
}

static inline void am_process__dynamic_wind_entry_set_saved(am_list_t *entry, am_value_t v) {
    if (!entry || entry->length < 4) return;
    entry->children[3] = v;
}

// 根据 handle 获取条目对象指针
static am_list_t *am_process__dynamic_wind_get_entry(am_process_t *proc, am_handle_t entry_hd) {
    if (!proc || !proc->heap || entry_hd == AM_HANDLE_NULL) return NULL;
    am_value_t v = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, entry_hd);
    if (!am_value_is_ptr(v)) return NULL;
    am_object_t *obj = am_value_to_ptr(v);
    if (!obj || obj->type != AM_OBJECT_TYPE_LIST) return NULL;
    return (am_list_t *)obj;
}

// 计算两个 dynamic-wind 栈（list of entry handles）的最长公共前缀长度（按 mark 比较）
static size_t dynamic_wind_common_prefix(am_process_t *proc, am_list_t *target) {
    if (!proc) return 0;
    am_list_t *current = proc->dynamic_wind_stack;
    size_t cur_len = current ? current->length : 0;
    size_t tgt_len = target ? target->length : 0;
    size_t min_len = cur_len < tgt_len ? cur_len : tgt_len;
    size_t prefix = 0;
    for (size_t i = 0; i < min_len; i++) {
        am_handle_t cur_hd = am_value_to_handle(current->children[i]);
        am_handle_t tgt_hd = am_value_to_handle(target->children[i]);
        am_list_t *cur_entry = am_process__dynamic_wind_get_entry(proc, cur_hd);
        am_list_t *tgt_entry = am_process__dynamic_wind_get_entry(proc, tgt_hd);
        if (!cur_entry || !tgt_entry) break;
        if (dynamic_wind_entry_mark(cur_entry) != dynamic_wind_entry_mark(tgt_entry)) break;
        prefix++;
    }
    return prefix;
}


// ===============================================================================
// 字符串驻留
// ===============================================================================

// 功能说明：根据 wchar_t 缓冲区和长度创建/复用字符串堆对象，并返回其 handle。
// 实现说明：当 len <= AM_PROCESS_STRINDEX_MAX_LEN 时，会先查询 proc->strindex；
//         若已存在内容相同的字符串则复用其 handle，否则新建并登记。
//         超过阈值的字符串直接新建，不参与驻留。
//         失败返回 AM_HANDLE_NULL。
am_handle_t am_process_make_wstring_handle(am_process_t *proc, const wchar_t *str, size_t len) {
    if (!proc || !proc->heap || !proc->heap_alloc || !str) return AM_HANDLE_NULL;

    // 构造以 L'\0' 结尾的临时缓冲区，供 hash 计算和 strindex 查询使用
    wchar_t *tmp = (wchar_t *)am_malloc(proc->vm_alloc, (len + 1) * sizeof(wchar_t));
    if (!tmp) return AM_HANDLE_NULL;
    if (len > 0) {
        memcpy(tmp, str, len * sizeof(wchar_t));
    }
    tmp[len] = L'\0';

    uint32_t hash = am_strindex_hash_string(tmp);
    am_handle_t result = AM_HANDLE_NULL;

    if (proc->strindex && len <= AM_PROCESS_STRINDEX_MAX_LEN) {
        size_t n_candidates = am_strindex_get_all(proc->vm_alloc, proc->strindex, tmp, NULL, 0);
        if (n_candidates != SIZE_MAX && n_candidates > 0) {
            am_value_t *candidates = (am_value_t *)am_malloc(proc->vm_alloc,
                                                              n_candidates * sizeof(am_value_t));
            if (candidates) {
                size_t got = am_strindex_get_all(proc->vm_alloc, proc->strindex, tmp,
                                                 candidates, n_candidates);
                for (size_t i = 0; i < got; i++) {
                    am_handle_t cand_h = am_value_to_handle(candidates[i]);
                    am_value_t cand_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, cand_h);
                    if (!am_value_is_ptr(cand_val)) continue;
                    am_object_t *obj = am_value_to_ptr(cand_val);
                    if (obj->type != AM_OBJECT_TYPE_WSTRING) continue;
                    am_wstring_t *ws = (am_wstring_t *)obj;
                    if (ws->length != len) continue;

                    bool match = true;
                    for (size_t j = 0; j < len; j++) {
                        am_wchar_t wc = am_value_to_wchar(ws->content[j]);
                        if (wc != (am_wchar_t)tmp[j]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        result = cand_h;
                        break;
                    }
                }
                am_free(proc->vm_alloc, candidates);
            }
        }
    }

    if (result == AM_HANDLE_NULL) {
        am_wstring_t *ws = am_wstring_create(proc->heap_alloc, tmp, len);
        if (!ws) {
            am_free(proc->vm_alloc, tmp);
            return AM_HANDLE_NULL;
        }
        ws->base.hash = hash;

        am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
        if (hd == AM_HANDLE_NULL) {
            am_wstring_destroy(proc->heap_alloc, ws);
            am_free(proc->vm_alloc, tmp);
            return AM_HANDLE_NULL;
        }

        if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd,
                        am_make_value_of_ptr((am_object_t *)ws)) != 0) {
            am_wstring_destroy(proc->heap_alloc, ws);
            am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
            am_free(proc->vm_alloc, tmp);
            return AM_HANDLE_NULL;
        }

        result = hd;

        // 短字符串登记到 strindex
        if (proc->strindex && len <= AM_PROCESS_STRINDEX_MAX_LEN) {
            am_strindex_t *new_si = am_strindex_set(proc->vm_alloc, proc->strindex, tmp,
                                                     am_make_value_of_handle(hd));
            if (new_si) {
                proc->strindex = new_si;
            }
        }
    }

    am_free(proc->vm_alloc, tmp);
    return result;
}


// ===============================================================================
// 生命周期
// ===============================================================================

// 功能说明：从模块构造并初始化一个新的进程数据结构
// 实现说明：成功返回新进程对象指针；失败返回NULL
am_process_t *am_process_load_from_module(am_allocator_t *vm_alloc, am_allocator_t *heap_alloc, am_module_t *mod) {
    if (!vm_alloc || !heap_alloc || !mod || !mod->ast || !mod->ilcode) {
        return NULL;
    }

    am_process_t *proc = (am_process_t *)am_calloc(vm_alloc, sizeof(am_process_t));
    if (!proc) return NULL;

    proc->base.type = AM_OBJECT_TYPE_BASE;
    proc->vm_alloc = vm_alloc;
    proc->heap_alloc = heap_alloc;
    proc->pid = 0;
    proc->parent_pid = 0;
    proc->state = AM_PROCESS_STATE_READY;
    proc->PC = 0;
    proc->current_closure_handle = AM_HANDLE_NULL;
    proc->host_context = NULL;

    // 复制中间语言代码到进程
    proc->ilcode_length = mod->ilcode_length;
    proc->ilcode = (am_instruction_t *)am_malloc(vm_alloc, (mod->ilcode_length + 1) * sizeof(am_instruction_t));
    if (!proc->ilcode) {
        am_free(vm_alloc, proc);
        return NULL;
    }
    memcpy(proc->ilcode, mod->ilcode, mod->ilcode_length * sizeof(am_instruction_t));

    // 在 ilcode 末尾追加 wind 跳板指令
    proc->wind_trampoline_iaddr = proc->ilcode_length;
    proc->ilcode[proc->ilcode_length].opcode = AM_VM_OP_wind;
    proc->ilcode[proc->ilcode_length].operand = AM_VALUE_UNDEFINED;
    proc->ilcode_length += 1;

    // 将mod->ast->nodes深拷贝到proc->heap
    // 先通过deep_dump计算大小并序列化，再用deep_load到进程堆
    size_t dump_size = am_heap_deep_dump(mod->ast->alloc, mod->ast->alloc, mod->ast->nodes, NULL, 0);
    if (dump_size == SIZE_MAX) {
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }

    uint8_t *buffer = (uint8_t *)am_malloc(vm_alloc, dump_size);
    if (!buffer) {
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }
    memset(buffer, 0, dump_size);

    size_t written = am_heap_deep_dump(mod->ast->alloc, mod->ast->alloc, mod->ast->nodes, buffer, 0);
    if (written != dump_size) {
        am_free(vm_alloc, buffer);
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }

    proc->heap = am_heap_deep_load(vm_alloc, heap_alloc, buffer, 0);
    am_free(vm_alloc, buffer);
    if (!proc->heap) {
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }

    // 将拷贝进来的AST节点全部标记为静态对象，避免被GC回收
    am_heap_iter(vm_alloc, heap_alloc, proc->heap, set_all_heap_objects_static, NULL);

    // 拷贝 strindex（用于运行时字符串驻留）
    proc->strindex = am_strindex_copy(proc->vm_alloc, mod->ast->strindex);

    // 拷贝符号表
    proc->var_vocab = am_vocab_copy(proc->vm_alloc, mod->ast->var_vocab);
    proc->symbol_vocab = am_vocab_copy(proc->vm_alloc, mod->ast->symbol_vocab);
    proc->var_type = am_list_copy(proc->vm_alloc, mod->ast->var_type);
    proc->natives = am_map_copy(proc->vm_alloc, mod->ast->natives);
    proc->var_top = am_list_copy(proc->vm_alloc, mod->ast->var_top);
    proc->var_arn_mapping = am_map_copy(proc->vm_alloc, mod->ast->var_arn_mapping);
    if (!proc->strindex || !proc->var_vocab || !proc->symbol_vocab || !proc->var_type || !proc->natives || !proc->var_top || !proc->var_arn_mapping) {
        if (proc->strindex) am_strindex_destroy(proc->vm_alloc, proc->strindex);
        if (proc->var_vocab) am_vocab_destroy(proc->vm_alloc, proc->var_vocab);
        if (proc->symbol_vocab) am_vocab_destroy(proc->vm_alloc, proc->symbol_vocab);
        if (proc->var_type) am_list_destroy(proc->vm_alloc, proc->var_type);
        if (proc->natives) am_map_destroy(proc->vm_alloc, proc->natives);
        if (proc->var_top) am_list_destroy(proc->vm_alloc, proc->var_top);
        if (proc->var_arn_mapping) am_map_destroy(proc->vm_alloc, proc->var_arn_mapping);
        am_heap_destroy(vm_alloc, heap_alloc, proc->heap);
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }

    // 分配操作数栈
    proc->opstack_capacity = (size_t)(mod->opstack_depth > 0 ? mod->opstack_depth : 256);
    proc->opstack = (am_value_t *)am_calloc(vm_alloc, proc->opstack_capacity * sizeof(am_value_t));
    if (!proc->opstack) {
        am_heap_destroy(vm_alloc, heap_alloc, proc->heap);
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }
    proc->opstack_top = proc->opstack;

    // 分配函数调用栈
    proc->fstack_capacity = 2048;
    proc->fstack = (am_value_t *)am_calloc(vm_alloc, proc->fstack_capacity * sizeof(am_value_t));
    if (!proc->fstack) {
        am_free(vm_alloc, proc->opstack);
        am_heap_destroy(vm_alloc, heap_alloc, proc->heap);
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }
    proc->fstack_top = proc->fstack;

    // 初始化 dynamic-wind 状态
    proc->dynamic_wind_stack = am_list_create(vm_alloc, 8, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!proc->dynamic_wind_stack) {
        am_free(vm_alloc, proc->fstack);
        am_free(vm_alloc, proc->opstack);
        am_heap_destroy(vm_alloc, heap_alloc, proc->heap);
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }
    proc->dynamic_wind_after_stack = am_list_create(vm_alloc, 8, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!proc->dynamic_wind_after_stack) {
        am_list_destroy(vm_alloc, proc->dynamic_wind_stack);
        am_free(vm_alloc, proc->fstack);
        am_free(vm_alloc, proc->opstack);
        am_heap_destroy(vm_alloc, heap_alloc, proc->heap);
        am_free(vm_alloc, proc->ilcode);
        am_free(vm_alloc, proc);
        return NULL;
    }
    proc->dynamic_wind_mark_counter = 1;
    proc->current_dynamic_wind_entry = AM_HANDLE_NULL;
    proc->current_dynamic_wind_thunk = AM_HANDLE_NULL;

    // 初始化 wind 跳板状态
    proc->wind_state = 0;
    proc->pending_cont_handle = AM_HANDLE_NULL;
    proc->pending_cont_value = AM_VALUE_UNDEFINED;
    proc->pending_after_entries = NULL;
    proc->pending_after_count = 0;
    proc->pending_before_entries = NULL;
    proc->pending_before_count = 0;

    return proc;
}


// 功能说明：销毁进程数据结构，释放其占用的全部资源
// 实现说明：成功返回0，失败返回-1
int32_t am_process_destroy(am_process_t *proc) {
    if (!proc) return 0;

    if (proc->ilcode) {
        am_free(proc->vm_alloc, proc->ilcode);
        proc->ilcode = NULL;
    }
    if (proc->opstack) {
        am_free(proc->vm_alloc, proc->opstack);
        proc->opstack = NULL;
        proc->opstack_top = NULL;
    }
    if (proc->fstack) {
        am_free(proc->vm_alloc, proc->fstack);
        proc->fstack = NULL;
        proc->fstack_top = NULL;
    }
    if (proc->var_type) {
        am_list_destroy(proc->vm_alloc, proc->var_type);
        proc->var_type = NULL;
    }
    if (proc->natives) {
        am_map_destroy(proc->vm_alloc, proc->natives);
        proc->natives = NULL;
    }
    if (proc->var_top) {
        am_list_destroy(proc->vm_alloc, proc->var_top);
        proc->var_top = NULL;
    }
    if (proc->var_arn_mapping) {
        am_map_destroy(proc->vm_alloc, proc->var_arn_mapping);
        proc->var_arn_mapping = NULL;
    }
    if (proc->strindex) {
        am_strindex_destroy(proc->vm_alloc, proc->strindex);
        proc->strindex = NULL;
    }
    if (proc->dynamic_wind_stack) {
        am_list_destroy(proc->vm_alloc, proc->dynamic_wind_stack);
        proc->dynamic_wind_stack = NULL;
    }
    if (proc->dynamic_wind_after_stack) {
        am_list_destroy(proc->vm_alloc, proc->dynamic_wind_after_stack);
        proc->dynamic_wind_after_stack = NULL;
    }
    if (proc->pending_after_entries) {
        am_free(proc->vm_alloc, proc->pending_after_entries);
        proc->pending_after_entries = NULL;
    }
    if (proc->pending_before_entries) {
        am_free(proc->vm_alloc, proc->pending_before_entries);
        proc->pending_before_entries = NULL;
    }
    if (proc->heap) {
        am_heap_destroy(proc->vm_alloc, proc->heap_alloc, proc->heap);
        proc->heap = NULL;
    }

    am_free(proc->vm_alloc, proc);
    return 0;
}


// ===============================================================================
// 操作数栈操作
// ===============================================================================

// 功能说明：向操作数栈中压入值。成功返回0，失败返回-1
int32_t am_process_push_operand(am_process_t *proc, am_value_t v) {
    if (!proc || !proc->opstack || !proc->opstack_top) return -1;
    size_t used = (size_t)(proc->opstack_top - proc->opstack);
    // if (used >= proc->opstack_capacity) {
    //     fprintf(stderr, "[Process] am_process_push_operand OPSTACK容量不足\n");
    //     return -1;
    // }
    // 注：以下是opstack深度估计不准或者无估计时的权宜之计
    if (used >= proc->opstack_capacity) {
        size_t new_capacity = proc->opstack_capacity * 2;
        if (new_capacity < 16) new_capacity = 16;
        am_value_t *new_opstack = (am_value_t *)am_realloc(proc->vm_alloc, proc->opstack,
                                                             new_capacity * sizeof(am_value_t));
        if (!new_opstack) return -1;
        proc->opstack_top = new_opstack + used;
        proc->opstack = new_opstack;
        proc->opstack_capacity = new_capacity;
    }
    *proc->opstack_top++ = v;
    return 0;
}


// 功能说明：从操作数栈中弹出一个值。成功返回弹出值，失败返回UINTPTR_MAX
am_value_t am_process_pop_operand(am_process_t *proc) {
    if (!proc || !proc->opstack || !proc->opstack_top) return (am_value_t)UINTPTR_MAX;
    if (proc->opstack_top <= proc->opstack) return (am_value_t)UINTPTR_MAX;
    return *--proc->opstack_top;
}


// 功能说明：根据栈顶指针计算opstack中有多少个am_value_t。成功返回长度值，失败返回SIZE_MAX
size_t am_process_length_of_opstack(am_process_t *proc) {
    if (!proc || !proc->opstack || !proc->opstack_top) return SIZE_MAX;
    return (size_t)(proc->opstack_top - proc->opstack);
}


// ===============================================================================
// 函数调用栈操作
// ===============================================================================

// 功能说明：向fstack中压入栈帧（两个值）。成功返回0，失败返回-1
int32_t am_process_push_stack_frame(am_process_t *proc, am_value_t closure_handle_value, am_value_t return_target_iaddr_value) {
    if (!proc || !proc->fstack || !proc->fstack_top) return -1;
    if (!am_value_is_handle(closure_handle_value)) return -1;
    if (!am_value_is_iaddr(return_target_iaddr_value)) return -1;

    size_t used = (size_t)(proc->fstack_top - proc->fstack);
    if (used + 2 > proc->fstack_capacity) return -1;

    *proc->fstack_top++ = closure_handle_value;
    *proc->fstack_top++ = return_target_iaddr_value;
    return 0;
}


// 功能说明：从fstack中弹出栈帧的两个值，通过两个指针传出。成功返回0，失败返回-1
int32_t am_process_pop_stack_frame(am_process_t *proc, am_value_t *closure_handle_value, am_value_t *return_target_iaddr_value) {
    if (!proc || !proc->fstack || !proc->fstack_top || !closure_handle_value || !return_target_iaddr_value) return -1;
    if (proc->fstack_top - proc->fstack < 2) return -1;

    *return_target_iaddr_value = *--proc->fstack_top;
    *closure_handle_value = *--proc->fstack_top;
    return 0;
}


// 功能说明：根据栈顶指针计算fstack中有多少个am_value_t（因为是成对push/pop，所以正常情况下必为偶数）。成功返回长度值，失败返回SIZE_MAX
size_t am_process_length_of_fstack(am_process_t *proc) {
    if (!proc || !proc->fstack || !proc->fstack_top) return SIZE_MAX;
    return (size_t)(proc->fstack_top - proc->fstack);
}


// ===============================================================================
// 闭包操作
// ===============================================================================

// 功能说明：新建闭包并返回其handle。成功返回handle，失败返回AM_HANDLE_NULL
am_handle_t am_process_make_closure(am_process_t *proc, am_iaddr_t iaddr, am_handle_t parent) {
    if (!proc || !proc->heap || !proc->heap_alloc) return AM_HANDLE_NULL;

    // 首先在proc->heap中申请一个新的handle
    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) return AM_HANDLE_NULL;

    // 新建闭包对象
    am_obj_closure_t *closure_obj = am_closure_create(proc->heap_alloc, iaddr, parent, 16);
    if (!closure_obj) {
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return AM_HANDLE_NULL;
    }

    // 将闭包对象的指针绑定到hd上
    am_value_t closure_value = am_make_value_of_ptr((am_object_t *)closure_obj);
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd, closure_value) != 0) {
        am_closure_destroy(proc->heap_alloc, closure_obj);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return AM_HANDLE_NULL;
    }

    return hd;
}


// 功能说明：根据闭包handle获取闭包对象。成功返回指针，失败返回NULL
am_obj_closure_t *am_process_get_closure(am_process_t *proc, am_handle_t hd) {
    if (!proc || !proc->heap) return NULL;

    am_value_t v = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(v)) return NULL;

    am_object_t *obj = am_value_to_ptr(v);
    if (!obj || obj->type != AM_OBJECT_TYPE_CLOSURE) return NULL;

    return (am_obj_closure_t *)obj;
}


// 功能说明：获取进程的当前闭包对象。成功返回指针，失败返回NULL
am_obj_closure_t *am_process_get_current_closure(am_process_t *proc) {
    if (!proc) return NULL;
    return am_process_get_closure(proc, proc->current_closure_handle);
}


// 功能说明：变量解引用。成功返回TPV，失败返回UINTPTR_MAX
// 设计说明：先在当前闭包查找约束变量；若不存在，则沿闭包链查找约束变量定义位置，
//          根据脏标记决定使用定义位置的约束变量值，还是使用当前闭包中的自由变量值。
am_value_t am_process_dereference(am_process_t *proc, am_varid_t varid) {
    if (!proc) return (am_value_t)UINTPTR_MAX;

    am_obj_closure_t *current_closure_obj = am_process_get_current_closure(proc);
    if (!current_closure_obj) return (am_value_t)UINTPTR_MAX;

    // 查找当前闭包的约束变量
    if (am_closure_has_bound_var(proc->heap_alloc, current_closure_obj, varid) == 0) {
        return am_closure_get_bound_var(proc->heap_alloc, current_closure_obj, varid);
    }

    // 查找当前闭包的自由变量（如果存在）对应的词法定义环境
    am_handle_t closure_handle = proc->current_closure_handle;
    while (closure_handle != AM_HANDLE_NULL) {
        am_obj_closure_t *closure_obj = am_process_get_closure(proc, closure_handle);
        if (!closure_obj) break;

        if (am_closure_has_bound_var(proc->heap_alloc, closure_obj, varid) == 0) {
            // 找到约束变量定义位置
            if (am_closure_is_dirty_var(proc->heap_alloc, closure_obj, varid) == 0) {
                // 脏标记为真：使用约束变量定义位置的新值
                return am_closure_get_bound_var(proc->heap_alloc, closure_obj, varid);
            }
            else {
                // 脏标记为假：使用当前闭包中的自由变量值
                return am_closure_get_free_var(proc->heap_alloc, current_closure_obj, varid);
            }
        }

        closure_handle = closure_obj->parent;
    }

    // 未找到变量定义
    return (am_value_t)UINTPTR_MAX;
}


// ===============================================================================
// 程序流程控制
// ===============================================================================

// 功能说明：获取当前指令，并取出opcode和operand。成功返回0，失败返回-1
int32_t am_process_current_instruction(am_process_t *proc, uint32_t *opcode, am_value_t *operand) {
    if (!proc || !opcode || !operand) return -1;
    if (!proc->ilcode || proc->PC >= proc->ilcode_length) return -1;

    *opcode = proc->ilcode[proc->PC].opcode;
    *operand = proc->ilcode[proc->PC].operand;
    return 0;
}


// 功能说明：前进一步（PC加1）
void am_process_step(am_process_t *proc) {
    if (!proc) return;
    proc->PC++;
}


// 功能说明：无条件跳转（PC置数iaddr）
void am_process_goto(am_process_t *proc, am_iaddr_t iaddr) {
    if (!proc) return;
    proc->PC = iaddr;
}


// 功能说明：设置进程状态
void am_process_set_state(am_process_t *proc, int32_t s) {
    if (!proc) return;
    proc->state = s;
}


// ===============================================================================
// 计算续体（continuation）的捕获和恢复
// ===============================================================================

// 功能说明：捕获当前续体，保存为堆对象，并返回其handle。成功返回handle，失败返回AM_HANDLE_NULL
am_handle_t am_process_capture_continuation(am_process_t *proc, am_iaddr_t cont_return_target_iaddr) {
    if (!proc || !proc->heap || !proc->heap_alloc) return AM_HANDLE_NULL;

    size_t opstack_length = am_process_length_of_opstack(proc);
    size_t fstack_length = am_process_length_of_fstack(proc);
    if (opstack_length == SIZE_MAX || fstack_length == SIZE_MAX) return AM_HANDLE_NULL;

    // 深拷贝当前 dynamic_wind_stack 到堆中，作为续体快照
    am_handle_t dw_snapshot_handle = AM_HANDLE_NULL;
    if (proc->dynamic_wind_stack) {
        am_list_t *dw_snapshot = am_list_copy(proc->heap_alloc, proc->dynamic_wind_stack);
        if (!dw_snapshot) return AM_HANDLE_NULL;
        dw_snapshot_handle = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
        if (dw_snapshot_handle == AM_HANDLE_NULL) {
            am_list_destroy(proc->heap_alloc, dw_snapshot);
            return AM_HANDLE_NULL;
        }
        am_value_t snapshot_value = am_make_value_of_ptr((am_object_t *)dw_snapshot);
        if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_snapshot_handle, snapshot_value) != 0) {
            am_list_destroy(proc->heap_alloc, dw_snapshot);
            am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_snapshot_handle);
            return AM_HANDLE_NULL;
        }
    }

    // 深拷贝当前 dynamic_wind_after_stack 到堆中（after 内捕获续体时需要）
    am_handle_t dw_after_snapshot_handle = AM_HANDLE_NULL;
    if (proc->dynamic_wind_after_stack && proc->dynamic_wind_after_stack->length > 0) {
        am_list_t *dw_after_snapshot = am_list_copy(proc->heap_alloc, proc->dynamic_wind_after_stack);
        if (!dw_after_snapshot) {
            if (dw_snapshot_handle != AM_HANDLE_NULL) {
                am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_snapshot_handle);
            }
            return AM_HANDLE_NULL;
        }
        dw_after_snapshot_handle = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
        if (dw_after_snapshot_handle == AM_HANDLE_NULL) {
            am_list_destroy(proc->heap_alloc, dw_after_snapshot);
            if (dw_snapshot_handle != AM_HANDLE_NULL) {
                am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_snapshot_handle);
            }
            return AM_HANDLE_NULL;
        }
        am_value_t snapshot_value = am_make_value_of_ptr((am_object_t *)dw_after_snapshot);
        if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_after_snapshot_handle, snapshot_value) != 0) {
            am_list_destroy(proc->heap_alloc, dw_after_snapshot);
            am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_after_snapshot_handle);
            if (dw_snapshot_handle != AM_HANDLE_NULL) {
                am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_snapshot_handle);
            }
            return AM_HANDLE_NULL;
        }
    }

    // 创建续体对象，深拷贝当前opstack和fstack
    am_continuation_t *cont = am_continuation_create(
        proc->heap_alloc,
        cont_return_target_iaddr,
        proc->current_closure_handle,
        proc->opstack, opstack_length,
        proc->fstack, fstack_length,
        dw_snapshot_handle
    );
    if (!cont) {
        if (dw_snapshot_handle != AM_HANDLE_NULL) {
            am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_snapshot_handle);
        }
        if (dw_after_snapshot_handle != AM_HANDLE_NULL) {
            am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, dw_after_snapshot_handle);
        }
        return AM_HANDLE_NULL;
    }

    // 保存 dynamic-wind 的 transient 状态，使得在 before/after 内捕获的续体也能正确恢复
    cont->current_dynamic_wind_entry_handle = proc->current_dynamic_wind_entry;
    cont->current_dynamic_wind_thunk_handle = proc->current_dynamic_wind_thunk;
    cont->dynamic_wind_after_stack_handle = dw_after_snapshot_handle;

    // 在堆中分配handle并绑定续体对象
    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) {
        am_continuation_destroy(proc->heap_alloc, cont);
        return AM_HANDLE_NULL;
    }

    am_value_t cont_value = am_make_value_of_ptr((am_object_t *)cont);
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd, cont_value) != 0) {
        am_continuation_destroy(proc->heap_alloc, cont);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return AM_HANDLE_NULL;
    }

    return hd;
}


// 功能说明：直接恢复续体快照（opstack/fstack/closure），不执行 wind 调整。成功返回 cont_return_target，失败返回 SIZE_MAX
am_iaddr_t am_process_restore_continuation_snapshot(am_process_t *proc, am_handle_t hd) {
    if (!proc || !proc->heap) return SIZE_MAX;

    am_value_t v = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(v)) return SIZE_MAX;

    am_object_t *obj = am_value_to_ptr(v);
    if (!obj || obj->type != AM_OBJECT_TYPE_CONTINUATION) return SIZE_MAX;

    am_continuation_t *cont = (am_continuation_t *)obj;

    // 获取续体中保存的opstack和fstack副本
    size_t cont_opstack_length = 0;
    size_t cont_fstack_length = 0;
    am_value_t *cont_opstack = am_continuation_get_opstack(proc->vm_alloc, cont, &cont_opstack_length);
    am_value_t *cont_fstack = am_continuation_get_fstack(proc->vm_alloc, cont, &cont_fstack_length);

    if (!cont_opstack || !cont_fstack) {
        if (cont_opstack) am_free(proc->vm_alloc, cont_opstack);
        if (cont_fstack) am_free(proc->vm_alloc, cont_fstack);
        return SIZE_MAX;
    }

    // 检查容量是否足够
    if (cont_opstack_length > proc->opstack_capacity || cont_fstack_length > proc->fstack_capacity) {
        am_free(proc->vm_alloc, cont_opstack);
        am_free(proc->vm_alloc, cont_fstack);
        return SIZE_MAX;
    }

    // 恢复运行时状态
    if (cont_opstack_length > 0) {
        memcpy(proc->opstack, cont_opstack, cont_opstack_length * sizeof(am_value_t));
    }
    proc->opstack_top = proc->opstack + cont_opstack_length;

    if (cont_fstack_length > 0) {
        memcpy(proc->fstack, cont_fstack, cont_fstack_length * sizeof(am_value_t));
    }
    proc->fstack_top = proc->fstack + cont_fstack_length;

    proc->current_closure_handle = cont->current_closure_handle;
    proc->current_dynamic_wind_entry = cont->current_dynamic_wind_entry_handle;
    proc->current_dynamic_wind_thunk = cont->current_dynamic_wind_thunk_handle;

    // 恢复 dynamic_wind_after_stack（after 内捕获续体时可能需要）
    if (proc->dynamic_wind_after_stack) {
        am_list_destroy(proc->vm_alloc, proc->dynamic_wind_after_stack);
        proc->dynamic_wind_after_stack = NULL;
    }
    if (cont->dynamic_wind_after_stack_handle != AM_HANDLE_NULL) {
        am_value_t after_stack_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap,
                                                  cont->dynamic_wind_after_stack_handle);
        if (am_value_is_ptr(after_stack_val)) {
            am_list_t *after_stack_obj = (am_list_t *)am_value_to_ptr(after_stack_val);
            am_list_t *restored = am_list_copy(proc->vm_alloc, after_stack_obj);
            if (!restored) {
                am_free(proc->vm_alloc, cont_opstack);
                am_free(proc->vm_alloc, cont_fstack);
                return SIZE_MAX;
            }
            proc->dynamic_wind_after_stack = restored;
        }
    }
    if (!proc->dynamic_wind_after_stack) {
        proc->dynamic_wind_after_stack = am_list_create(proc->vm_alloc, 8, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
        if (!proc->dynamic_wind_after_stack) {
            am_free(proc->vm_alloc, cont_opstack);
            am_free(proc->vm_alloc, cont_fstack);
            return SIZE_MAX;
        }
    }

    am_free(proc->vm_alloc, cont_opstack);
    am_free(proc->vm_alloc, cont_fstack);

    return cont->cont_return_target;
}


// 功能说明：恢复指定的计算续体到当前进程。成功返回其返回目标位置的iaddr，失败返回SIZE_MAX
// 实现说明：传入的 value 为调用续体时传入的值；若需要 wind 调整，则 value 暂存于 proc，待跳板恢复时压栈。
am_iaddr_t am_process_load_continuation(am_process_t *proc, am_handle_t hd, am_value_t value) {
    if (!proc || !proc->heap) return SIZE_MAX;

    am_value_t v = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(v)) return SIZE_MAX;

    am_object_t *obj = am_value_to_ptr(v);
    if (!obj || obj->type != AM_OBJECT_TYPE_CONTINUATION) return SIZE_MAX;

    am_continuation_t *cont = (am_continuation_t *)obj;

    // 获取续体捕获时的 dynamic-wind 栈快照
    am_list_t *target_dw_stack = NULL;
    if (cont->dynamic_wind_stack_handle != AM_HANDLE_NULL) {
        am_value_t dw_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, cont->dynamic_wind_stack_handle);
        if (am_value_is_ptr(dw_val)) {
            am_object_t *dw_obj = am_value_to_ptr(dw_val);
            if (dw_obj && dw_obj->type == AM_OBJECT_TYPE_LIST) {
                target_dw_stack = (am_list_t *)dw_obj;
            }
        }
    }

    size_t prefix = dynamic_wind_common_prefix(proc, target_dw_stack);
    size_t current_len = proc->dynamic_wind_stack ? proc->dynamic_wind_stack->length : 0;
    size_t target_len = target_dw_stack ? target_dw_stack->length : 0;

    // 如果当前栈与目标栈完全一致，直接恢复续体
    if (prefix == current_len && prefix == target_len) {
        am_iaddr_t cont_target = am_process_restore_continuation_snapshot(proc, hd);
        if (cont_target == SIZE_MAX) return SIZE_MAX;
        if (am_process_push_operand(proc, value) != 0) return SIZE_MAX;
        return cont_target;
    }

    // 需要 wind 调整：计算 afters（当前栈中多出部分，从内到外）和 befores（目标栈中多出部分，从外到内）
    size_t after_count = current_len - prefix;
    size_t before_count = target_len - prefix;

    am_handle_t *after_entries = NULL;
    am_handle_t *before_entries = NULL;
    if (after_count > 0) {
        after_entries = (am_handle_t *)am_malloc(proc->vm_alloc, after_count * sizeof(am_handle_t));
        if (!after_entries) return SIZE_MAX;
        for (size_t i = 0; i < after_count; i++) {
            size_t idx = current_len - 1 - i;
            after_entries[i] = am_value_to_handle(proc->dynamic_wind_stack->children[idx]);
        }
    }
    if (before_count > 0) {
        before_entries = (am_handle_t *)am_malloc(proc->vm_alloc, before_count * sizeof(am_handle_t));
        if (!before_entries) {
            if (after_entries) am_free(proc->vm_alloc, after_entries);
            return SIZE_MAX;
        }
        for (size_t i = 0; i < before_count; i++) {
            size_t idx = prefix + i;
            before_entries[i] = am_value_to_handle(target_dw_stack->children[idx]);
        }
    }

    // 释放旧的 pending 数组（如果存在）
    if (proc->pending_after_entries) {
        am_free(proc->vm_alloc, proc->pending_after_entries);
    }
    if (proc->pending_before_entries) {
        am_free(proc->vm_alloc, proc->pending_before_entries);
    }

    proc->pending_cont_handle = hd;
    proc->pending_cont_value = value;
    proc->pending_after_entries = after_entries;
    proc->pending_after_count = after_count;
    proc->pending_before_entries = before_entries;
    proc->pending_before_count = before_count;
    proc->wind_state = 1;

    return proc->wind_trampoline_iaddr;
}


// ===============================================================================
// 列表字符串化辅助结构
// ===============================================================================

typedef struct {
    am_allocator_t *alloc;
    wchar_t *buf;
    size_t len;
    size_t cap;
} am_process_strbuf_t;


static int32_t am_process_strbuf_init(am_allocator_t *alloc, am_process_strbuf_t *sb, size_t initial_cap) {
    if (!alloc || !sb || initial_cap == 0) return -1;
    sb->alloc = alloc;
    sb->buf = (wchar_t *)am_malloc(alloc, initial_cap * sizeof(wchar_t));
    if (!sb->buf) return -1;
    sb->buf[0] = L'\0';
    sb->len = 0;
    sb->cap = initial_cap;
    return 0;
}


static int32_t am_process_strbuf_ensure(am_process_strbuf_t *sb, size_t needed) {
    if (!sb || !sb->buf) return -1;
    if (needed <= sb->cap) return 0;

    size_t new_cap = sb->cap;
    while (new_cap < needed) {
        new_cap *= 2;
    }

    wchar_t *new_buf = (wchar_t *)am_malloc(sb->alloc, new_cap * sizeof(wchar_t));
    if (!new_buf) return -1;

    memcpy(new_buf, sb->buf, (sb->len + 1) * sizeof(wchar_t));
    am_free(sb->alloc, sb->buf);
    sb->buf = new_buf;
    sb->cap = new_cap;
    return 0;
}


static int32_t am_process_strbuf_append_char(am_process_strbuf_t *sb, wchar_t c) {
    if (!sb) return -1;
    if (am_process_strbuf_ensure(sb, sb->len + 2) != 0) return -1;
    sb->buf[sb->len++] = c;
    sb->buf[sb->len] = L'\0';
    return 0;
}


static int32_t am_process_strbuf_append_string(am_process_strbuf_t *sb, const wchar_t *s) {
    if (!sb || !s) return -1;
    size_t slen = wcslen(s);
    if (am_process_strbuf_ensure(sb, sb->len + slen + 1) != 0) return -1;
    memcpy(&sb->buf[sb->len], s, slen * sizeof(wchar_t));
    sb->len += slen;
    sb->buf[sb->len] = L'\0';
    return 0;
}


static int32_t am_process_append_value_to_strbuf(am_process_strbuf_t *sb, am_process_t *proc, am_value_t value, bool in_quote);


static int32_t am_process_append_lambda_to_strbuf(am_process_strbuf_t *sb, am_process_t *proc, am_list_t *lambda, bool in_quote) {
    if (!sb || !proc || !lambda) return -1;

    if (am_process_strbuf_append_string(sb, L"(lambda (") != 0) return -1;

    size_t n_param = 0;
    if (lambda->length >= 2) {
        am_value_t n_param_val = am_list_get(proc->vm_alloc, lambda, 1);
        if (am_value_is_uint(n_param_val)) {
            n_param = (size_t)am_value_to_uint(n_param_val);
        }
    }

    for (size_t i = 0; i < n_param; i++) {
        if (i > 0) {
            if (am_process_strbuf_append_char(sb, L' ') != 0) return -1;
        }
        am_value_t param = am_list_get(proc->vm_alloc, lambda, 2 + i);
        if (am_process_append_value_to_strbuf(sb, proc, param, in_quote) != 0) return -1;
    }
    if (am_process_strbuf_append_char(sb, L')') != 0) return -1;

    size_t n_body = am_list_lambda_get_body_number(proc->vm_alloc, lambda);
    for (size_t i = 0; i < n_body; i++) {
        if (am_process_strbuf_append_char(sb, L' ') != 0) return -1;
        am_value_t body = am_list_get(proc->vm_alloc, lambda, 2 + n_param + i);
        if (am_process_append_value_to_strbuf(sb, proc, body, in_quote) != 0) return -1;
    }

    if (am_process_strbuf_append_char(sb, L')') != 0) return -1;
    return 0;
}


static int32_t am_process_append_list_to_strbuf(am_process_strbuf_t *sb, am_process_t *proc, am_list_t *lst, bool in_quote) {
    if (!sb || !proc || !lst) return -1;

    const wchar_t *prefix = L"(";
    if (lst->length == 0) {
        // 空列表无论位于何处都显示前导单引号
        prefix = L"'(";
    }
    else if (lst->type == AM_LIST_TYPE_QUOTE) {
        // quote 列表（无论最外层还是嵌套内层）不显示前导单引号
        prefix = L"(";
    }
    else if (lst->type == AM_LIST_TYPE_QUASIQUOTE) prefix = L"`(";
    else if (lst->type == AM_LIST_TYPE_UNQUOTE)    prefix = L",(";

    if (am_process_strbuf_append_string(sb, prefix) != 0) return -1;

    bool child_in_quote = in_quote || (lst->type == AM_LIST_TYPE_QUOTE) || (lst->type == AM_LIST_TYPE_QUASIQUOTE);

    for (size_t i = 0; i < lst->length; i++) {
        if (i > 0) {
            if (am_process_strbuf_append_char(sb, L' ') != 0) return -1;
        }
        am_value_t child = am_list_get(proc->vm_alloc, lst, i);
        if (am_process_append_value_to_strbuf(sb, proc, child, child_in_quote) != 0) return -1;
    }

    if (am_process_strbuf_append_char(sb, L')') != 0) return -1;
    return 0;
}


static int32_t am_process_append_value_to_strbuf(am_process_strbuf_t *sb, am_process_t *proc, am_value_t value, bool in_quote) {
    if (!sb || !proc) return -1;

    if (am_value_is_handle(value)) {
        am_handle_t h = am_value_to_handle(value);
        if (h == AM_HANDLE_NULL) {
            return am_process_strbuf_append_string(sb, L"#<null-handle>");
        }
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, h);
        if (!am_value_is_ptr(obj_val)) {
            return am_process_strbuf_append_string(sb, L"#<handle>");
        }
        am_object_t *obj = am_value_to_ptr(obj_val);
        if (obj->type == AM_OBJECT_TYPE_LIST) {
            am_list_t *lst = (am_list_t *)obj;
            if (lst->type == AM_LIST_TYPE_LAMBDA) {
                return am_process_append_lambda_to_strbuf(sb, proc, lst, in_quote);
            }
            return am_process_append_list_to_strbuf(sb, proc, lst, in_quote);
        }
        else if (obj->type == AM_OBJECT_TYPE_WSTRING) {
            am_wstring_t *ws = (am_wstring_t *)obj;
            for (size_t i = 0; i < ws->length; i++) {
                am_value_t cv = ws->content[i];
                if (!am_value_is_wchar(cv)) continue;
                if (am_process_strbuf_append_char(sb, (wchar_t)am_value_to_wchar(cv)) != 0) return -1;
            }
            return 0;
        }
        return am_process_strbuf_append_string(sb, L"#<object>");
    }
    else if (am_value_is_varid(value)) {
        am_varid_t varid = am_value_to_varid(value);
        wchar_t *text = am_vocab_get(proc->vm_alloc, proc->var_vocab, &varid);
        if (!text) return am_process_strbuf_append_string(sb, L"#<var>");
        return am_process_strbuf_append_string(sb, text);
    }
    else if (am_value_is_symbol(value)) {
        am_symbol_t sym = am_value_to_symbol(value);
        wchar_t *text = am_vocab_get(proc->vm_alloc, proc->symbol_vocab, &sym);
        if (!text) return am_process_strbuf_append_string(sb, L"#<sym>");
        if (*text == L'\'') {
            // symbol 字面量（词汇表中已带前导单引号）
            // if (in_quote) {
                // 在 quote 列表内部：去掉前导单引号
                while (*text == L'\'') text++;
            // }
            return am_process_strbuf_append_string(sb, text);
        }
        // 关键字等不带前导单引号的 symbol：原样输出
        return am_process_strbuf_append_string(sb, text);
    }
    else if (am_value_is_uint(value)) {
        wchar_t tmp[64];
        swprintf(tmp, 64, L"%llu", (unsigned long long)am_value_to_uint(value));
        return am_process_strbuf_append_string(sb, tmp);
    }
    else if (am_value_is_int(value)) {
        wchar_t tmp[64];
        swprintf(tmp, 64, L"%lld", (long long)am_value_to_int(value));
        return am_process_strbuf_append_string(sb, tmp);
    }
    else if (am_value_is_float(value)) {
        wchar_t tmp[128];
        swprintf(tmp, 128, L"%g", (double)am_value_to_float(value));
        return am_process_strbuf_append_string(sb, tmp);
    }
    else if (am_value_is_boolean(value)) {
        return am_process_strbuf_append_string(sb, am_value_to_boolean(value) ? L"#t" : L"#f");
    }
    else if (am_value_is_null(value)) {
        return am_process_strbuf_append_string(sb, L"#null");
    }
    else if (am_value_is_undefined(value)) {
        return am_process_strbuf_append_string(sb, L"#undefined");
    }

    return am_process_strbuf_append_string(sb, L"#<value>");
}


// 功能说明：将进程堆中的列表对象转换为可显示宽字符串。成功返回新分配的 wchar_t*，失败返回 NULL。
// 实现说明：从 proc->heap 中取得对象，从 proc->var_vocab / proc->symbol_vocab 中解析变量名和符号名。
//          symbol 的处理规则：不在 quote 列表内时带前导单引号；在 quote 列表内时不带前导单引号。
wchar_t *am_process_list_to_string(am_process_t *proc, am_handle_t hd, size_t *length) {
    if (!proc || !proc->heap || hd == AM_HANDLE_NULL) return NULL;

    am_value_t value = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(value)) return NULL;

    am_object_t *obj = am_value_to_ptr(value);
    if (obj->type != AM_OBJECT_TYPE_LIST) return NULL;

    am_process_strbuf_t sb;
    if (am_process_strbuf_init(proc->vm_alloc, &sb, 256) != 0) return NULL;

    am_list_t *lst = (am_list_t *)obj;
    bool in_quote = (lst->type == AM_LIST_TYPE_QUOTE);
    if (lst->type == AM_LIST_TYPE_LAMBDA) {
        if (am_process_append_lambda_to_strbuf(&sb, proc, lst, in_quote) != 0) {
            am_free(proc->vm_alloc, sb.buf);
            return NULL;
        }
    }
    else {
        if (am_process_append_list_to_strbuf(&sb, proc, lst, in_quote) != 0) {
            am_free(proc->vm_alloc, sb.buf);
            return NULL;
        }
    }

    if (length) *length = sb.len;
    return sb.buf;
}


/* ===== end:   src/am_process.c ===== */

/* ===== begin: src/am_gc.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>



// ===============================================================================
// 标记-清除：GC 根收集
// ===============================================================================

// 判断一个am_value_t是否为指向堆对象的把柄（handle）
static inline bool is_handle_value(am_value_t v) {
    return am_value_is_handle(v);
}


// GC根收集辅助函数：分析一组运行时环境（当前闭包、opstack、fstack）中的GC根
static int32_t gc_root_helper(
    am_process_t *proc, am_list_t **gcroots,
    am_handle_t current_closure_handle,
    am_value_t *opstack, size_t opstack_length,
    am_value_t *fstack, size_t fstack_length
) {
    if (!proc || !gcroots || !*gcroots) return -1;

    // 加入当前闭包handle
    am_list_t *lst = am_list_push(proc->vm_alloc, *gcroots, am_make_value_of_handle(current_closure_handle));
    if (!lst) return -1;
    *gcroots = lst;

    // 加入当前闭包内的变量绑定（约束变量和自由变量）
    am_obj_closure_t *current_closure_obj = am_process_get_closure(proc, current_closure_handle);
    if (current_closure_obj) {
        for (size_t i = 0; i < current_closure_obj->length; i++) {
            am_value_t value = current_closure_obj->bindings[i].value;
            if (is_handle_value(value)) {
                lst = am_list_push(proc->vm_alloc, *gcroots, value);
                if (!lst) return -1;
                *gcroots = lst;
            }
        }
    }

    // 加入操作数栈内的把柄
    for (size_t i = 0; i < opstack_length; i++) {
        am_value_t v = opstack[i];
        if (is_handle_value(v)) {
            lst = am_list_push(proc->vm_alloc, *gcroots, v);
            if (!lst) return -1;
            *gcroots = lst;
        }
    }

    // 加入函数调用栈中每个栈帧对应的闭包把柄，以及这些闭包内的变量绑定
    // fstack成对存储：closure_handle_value, return_target_iaddr_value
    for (size_t i = 0; i + 1 < fstack_length; i += 2) {
        am_value_t closure_handle_value = fstack[i];
        am_value_t return_target_iaddr_value = fstack[i + 1];
        (void)return_target_iaddr_value;

        if (!am_value_is_handle(closure_handle_value)) {
            fprintf(stderr, "[gc_root_helper] 预期闭包handle，实际非handle\n");
            return -1;
        }

        am_handle_t closure_handle = am_value_to_handle(closure_handle_value);
        if (closure_handle == AM_HANDLE_NULL) continue;
        am_obj_closure_t *closure_obj = am_process_get_closure(proc, closure_handle);
        if (!closure_obj) {
            fprintf(stderr, "[gc_root_helper] 无法获取闭包对象 %zu\n", closure_handle);
            continue;
        }
        if (closure_obj->base.type != AM_OBJECT_TYPE_CLOSURE) {
            fprintf(stderr, "[gc_root_helper] 预期闭包，实际非闭包\n");
            return -1;
        }

        // 将栈帧的闭包handle加入GC根
        lst = am_list_push(proc->vm_alloc, *gcroots, closure_handle_value);
        if (!lst) return -1;
        *gcroots = lst;

        // 将该闭包内的变量绑定中的handle加入GC根
        for (size_t j = 0; j < closure_obj->length; j++) {
            am_value_t value = closure_obj->bindings[j].value;
            if (is_handle_value(value)) {
                lst = am_list_push(proc->vm_alloc, *gcroots, value);
                if (!lst) return -1;
                *gcroots = lst;
            }
        }
    }

    return 0;
}


// 功能说明：从当前进程和续体环境中收集GC根。成功返回0，失败返回-1
// 设计说明：可达性分析的根（GC根）有：当前闭包本身、当前闭包和函数调用栈对应闭包内的变量绑定、操作数栈内的把柄、函数调用栈内所有栈帧对应的闭包把柄、所有continuation中保留的上面的各项
// 实现说明：gcroots是收集到的GC根的TPV的列表，由外部分配和释放。
static int32_t gc_root(am_process_t *proc, am_list_t **gcroots) {
    if (!proc || !gcroots || !*gcroots || !proc->heap) return -1;

    // 分析当前进程中的GC根
    size_t opstack_length = am_process_length_of_opstack(proc);
    size_t fstack_length = am_process_length_of_fstack(proc);
    if (opstack_length == SIZE_MAX || fstack_length == SIZE_MAX) return -1;

    if (gc_root_helper(proc, gcroots, proc->current_closure_handle,
                       proc->opstack, opstack_length,
                       proc->fstack, fstack_length) != 0) {
        return -1;
    }

    // 将 strindex 中所有有效 handle 加入 GC 根，防止驻留字符串被回收后产生悬空引用
    if (proc->strindex) {
        for (size_t i = 0; i < proc->strindex->capacity; i++) {
            uint32_t hash = proc->strindex->slots[i].hash;
            if (hash == AM_STRINDEX_KEY_EMPTY || hash == AM_STRINDEX_KEY_TOMBSTONE) continue;

            am_value_t h_val = proc->strindex->slots[i].value;
            if (!am_value_is_handle(h_val)) continue;

            am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots, h_val);
            if (!new_roots) return -1;
            *gcroots = new_roots;
        }
    }

    // 将当前 dynamic-wind 栈中的 entry handle 加入 GC 根
    if (proc->dynamic_wind_stack) {
        for (size_t i = 0; i < proc->dynamic_wind_stack->length; i++) {
            am_value_t entry_val = proc->dynamic_wind_stack->children[i];
            if (am_value_is_handle(entry_val)) {
                am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots, entry_val);
                if (!new_roots) return -1;
                *gcroots = new_roots;
            }
        }
    }

    // 将正在执行 after 的 dynamic-wind 条目 handle 加入 GC 根
    if (proc->dynamic_wind_after_stack) {
        for (size_t i = 0; i < proc->dynamic_wind_after_stack->length; i++) {
            am_value_t entry_val = proc->dynamic_wind_after_stack->children[i];
            if (am_value_is_handle(entry_val)) {
                am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots, entry_val);
                if (!new_roots) return -1;
                *gcroots = new_roots;
            }
        }
    }

    // 将 wind 跳板暂存的 continuation 把柄/值和待执行条目加入 GC 根
    if (proc->pending_cont_handle != AM_HANDLE_NULL) {
        am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                             am_make_value_of_handle(proc->pending_cont_handle));
        if (!new_roots) return -1;
        *gcroots = new_roots;
    }
    if (am_value_is_handle(proc->pending_cont_value)) {
        am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots, proc->pending_cont_value);
        if (!new_roots) return -1;
        *gcroots = new_roots;
    }
    if (proc->current_dynamic_wind_entry != AM_HANDLE_NULL) {
        am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                             am_make_value_of_handle(proc->current_dynamic_wind_entry));
        if (!new_roots) return -1;
        *gcroots = new_roots;
    }
    if (proc->current_dynamic_wind_thunk != AM_HANDLE_NULL) {
        am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                             am_make_value_of_handle(proc->current_dynamic_wind_thunk));
        if (!new_roots) return -1;
        *gcroots = new_roots;
    }
    if (proc->pending_after_entries) {
        for (size_t i = 0; i < proc->pending_after_count; i++) {
            am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                                 am_make_value_of_handle(proc->pending_after_entries[i]));
            if (!new_roots) return -1;
            *gcroots = new_roots;
        }
    }
    if (proc->pending_before_entries) {
        for (size_t i = 0; i < proc->pending_before_count; i++) {
            am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                                 am_make_value_of_handle(proc->pending_before_entries[i]));
            if (!new_roots) return -1;
            *gcroots = new_roots;
        }
    }

    // 分析所有已保存的续体环境中的GC根
    // 遍历堆中所有对象，找到continuation对象
    size_t heap_count = am_map_length(proc->heap_alloc, proc->heap->table);
    am_value_t *keys = am_map_keys(proc->vm_alloc, proc->heap->table);
    if (!keys && heap_count > 0) return -1;

    int32_t ret = 0;
    for (size_t i = 0; i < heap_count; i++) {
        am_handle_t hd = am_value_to_handle(keys[i]);
        am_value_t value = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (!am_value_is_ptr(value)) continue;

        am_object_t *obj = am_value_to_ptr(value);
        if (!obj || obj->type != AM_OBJECT_TYPE_CONTINUATION) continue;

        am_continuation_t *cont = (am_continuation_t *)obj;

        // 将续体保存的 dynamic-wind 相关 handle 加入 GC 根
        if (cont->dynamic_wind_stack_handle != AM_HANDLE_NULL) {
            am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                                 am_make_value_of_handle(cont->dynamic_wind_stack_handle));
            if (!new_roots) {
                am_free(proc->vm_alloc, keys);
                return -1;
            }
            *gcroots = new_roots;
        }
        if (cont->dynamic_wind_after_stack_handle != AM_HANDLE_NULL) {
            am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                                 am_make_value_of_handle(cont->dynamic_wind_after_stack_handle));
            if (!new_roots) {
                am_free(proc->vm_alloc, keys);
                return -1;
            }
            *gcroots = new_roots;
        }
        if (cont->current_dynamic_wind_entry_handle != AM_HANDLE_NULL) {
            am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                                 am_make_value_of_handle(cont->current_dynamic_wind_entry_handle));
            if (!new_roots) {
                am_free(proc->vm_alloc, keys);
                return -1;
            }
            *gcroots = new_roots;
        }
        if (cont->current_dynamic_wind_thunk_handle != AM_HANDLE_NULL) {
            am_list_t *new_roots = am_list_push(proc->vm_alloc, *gcroots,
                                                 am_make_value_of_handle(cont->current_dynamic_wind_thunk_handle));
            if (!new_roots) {
                am_free(proc->vm_alloc, keys);
                return -1;
            }
            *gcroots = new_roots;
        }

        // 将续体内部环境加入GC根
        size_t cont_opstack_length = 0;
        size_t cont_fstack_length = 0;
        am_value_t *cont_opstack = am_continuation_get_opstack(proc->vm_alloc, cont, &cont_opstack_length);
        am_value_t *cont_fstack = am_continuation_get_fstack(proc->vm_alloc, cont, &cont_fstack_length);

        if (!cont_opstack || !cont_fstack) {
            if (cont_opstack) am_free(proc->vm_alloc, cont_opstack);
            if (cont_fstack) am_free(proc->vm_alloc, cont_fstack);
            ret = -1;
            break;
        }

        if (gc_root_helper(proc, gcroots, cont->current_closure_handle,
                           cont_opstack, cont_opstack_length,
                           cont_fstack, cont_fstack_length) != 0) {
            am_free(proc->vm_alloc, cont_opstack);
            am_free(proc->vm_alloc, cont_fstack);
            ret = -1;
            break;
        }

        am_free(proc->vm_alloc, cont_opstack);
        am_free(proc->vm_alloc, cont_fstack);
    }

    am_free(proc->vm_alloc, keys);
    return ret;
}


// ===============================================================================
// 标记-清除：递归标记与清除
// ===============================================================================

// 功能说明：从GC根开始，递归标记存活对象。成功返回0，失败返回-1（或更小的负数）
static int32_t gc_mark(am_process_t *proc, am_value_t v) {
    if (!proc || !proc->heap) return -1;

    int32_t ret = 0;

    // 仅处理handle类型的值
    if (!am_value_is_handle(v)) return 0;

    am_handle_t hd = am_value_to_handle(v);
    if (hd == AM_HANDLE_NULL) return 0;

    // handle必须存在于当前进程的堆中
    if (am_heap_has_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd) != 0) return 0;

    am_value_t obj_value = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_value)) return -1;

    am_object_t *obj = am_value_to_ptr(obj_value);
    if (!obj) return -1;

    // 已经标记过，避免循环引用导致无限递归
    if (am_object_check_alive(obj) == 0) return 0;

    // 根据对象类型进行标记和递归
    int32_t obj_type = obj->type;

    if (obj_type == AM_OBJECT_TYPE_LIST) {
        // 标记当前list对象存活
        am_object_set_alive(obj, 0);

        am_list_t *lst = (am_list_t *)obj;
        for (size_t i = 0; i < lst->length; i++) {
            ret += gc_mark(proc, lst->children[i]);
        }
    }
    else if (obj_type == AM_OBJECT_TYPE_WSTRING) {
        am_object_set_alive(obj, 0);
    }
    else if (obj_type == AM_OBJECT_TYPE_MAP) {
        am_object_set_alive(obj, 0);
        am_map_t *m = (am_map_t *)obj;
        for (size_t i = 0; i < m->capacity; i++) {
            am_value_t k = m->slots[i].key;
            if (k == AM_MAP_KEY_EMPTY || k == AM_MAP_KEY_TOMBSTONE) continue;
            if (am_value_is_handle(k)) ret += gc_mark(proc, k);
            am_value_t v = m->slots[i].value;
            if (am_value_is_handle(v)) ret += gc_mark(proc, v);
        }
    }
    else if (obj_type == AM_OBJECT_TYPE_CLOSURE) {
        am_object_set_alive(obj, 0);

        am_obj_closure_t *closure = (am_obj_closure_t *)obj;
        // 递归标记亲闭包
        ret += gc_mark(proc, am_make_value_of_handle(closure->parent));

        // 递归标记变量绑定中的handle
        for (size_t i = 0; i < closure->length; i++) {
            am_value_t value = closure->bindings[i].value;
            if (am_value_is_handle(value)) {
                ret += gc_mark(proc, value);
            }
        }
    }
    else if (obj_type == AM_OBJECT_TYPE_CONTINUATION) {
        // 续体对象本身标记为存活；其stacks中的handle已通过gc_root_helper加入GC根，
        // 因此无需在此递归展开，避免重复遍历。
        am_object_set_alive(obj, 0);
    }

    return ret;
}


// 功能说明：基于存活标记结果，删除所有未被标记存活的非静态对象和对应的handle。成功返回0，失败返回-1
static int32_t gc_sweep(am_process_t *proc) {
    if (!proc || !proc->heap || !proc->heap->table) return -1;

    size_t gcount = 0;
    size_t count = 0;

    size_t heap_count = am_map_length(proc->heap_alloc, proc->heap->table);
    am_value_t *keys = am_map_keys(proc->vm_alloc, proc->heap->table);
    if (!keys && heap_count > 0) return -1;

    for (size_t i = 0; i < heap_count; i++) {
        am_handle_t hd = am_value_to_handle(keys[i]);
        am_value_t value = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (!am_value_is_ptr(value)) continue;

        am_object_t *obj = am_value_to_ptr(value);
        if (!obj) continue;

        count++;

        // 静态对象永不清理
        if (am_object_check_static(obj) == 0) continue;

        // keepalive 对象（如异步回调闭包）应跳过清理
        if (am_object_check_keepalive(obj) == 0) {
            am_object_set_alive(obj, -1);
            continue;
        }

        int32_t obj_type = obj->type;
        if (obj_type == AM_OBJECT_TYPE_LIST ||
            obj_type == AM_OBJECT_TYPE_MAP ||
            obj_type == AM_OBJECT_TYPE_WSTRING ||
            obj_type == AM_OBJECT_TYPE_CLOSURE ||
            obj_type == AM_OBJECT_TYPE_CONTINUATION) {

            if (am_object_check_alive(obj) != 0) {
                // 未被标记为存活：删除handle，同时穿透释放其映射的obj
                am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
                gcount++;
            }
            else {
                // 对于存活对象，将其alive标识清空为否，以便下次gc重新标记
                am_object_set_alive(obj, -1);
            }
        }
    }

    am_free(proc->vm_alloc, keys);

    // printf("[GC] 已清理 %zu / %zu 个对象\n", gcount, count);

    // TODO 暂不实现allocator管理的底层物理内存的整理

    return 0;
}


// 功能说明：对进程执行全量的标记-清除GC。成功返回0，失败返回-1
int32_t am_gc_process(am_process_t *proc) {
    if (!proc || !proc->heap || !proc->heap_alloc || !proc->vm_alloc) return -1;

    // 收集GC根对象 TODO 初始容量可调
    am_list_t *gcroots = am_list_create(proc->vm_alloc, 2048, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!gcroots) return -1;

    int32_t ret = 0;

    if (gc_root(proc, &gcroots) != 0) {
        ret = -1;
        goto cleanup;
    }

    // 将堆中所有 keepalive 对象也加入 GC 根，确保异步回调闭包及其引用的
    // 父闭包链、捕获变量等不会被 GC 回收。
    size_t heap_count = am_map_length(proc->heap_alloc, proc->heap->table);
    am_value_t *keys = am_map_keys(proc->vm_alloc, proc->heap->table);
    if (!keys && heap_count > 0) {
        ret = -1;
        goto cleanup;
    }
    for (size_t i = 0; i < heap_count; i++) {
        am_handle_t hd = am_value_to_handle(keys[i]);
        am_value_t value = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (!am_value_is_ptr(value)) continue;
        am_object_t *obj = am_value_to_ptr(value);
        if (obj && am_object_check_keepalive(obj) == 0) {
            am_list_t *new_roots = am_list_push(proc->vm_alloc, gcroots, am_make_value_of_handle(hd));
            if (new_roots) gcroots = new_roots;
        }
    }
    am_free(proc->vm_alloc, keys);

    // 从GC根对象开始递归标记存活对象
    for (size_t i = 0; i < gcroots->length; i++) {
        am_value_t v = am_list_get(proc->vm_alloc, gcroots, i);
        if (gc_mark(proc, v) != 0) {
            ret = -1;
        }
    }

    // 清除未被标记为存活的非静态对象及其handle
    if (gc_sweep(proc) != 0) {
        ret = -1;
    }

    proc->gc_count++;

cleanup:
    am_list_destroy(proc->vm_alloc, gcroots);
    return ret;
}


// ===============================================================================
// 标记-压缩：存活对象收集、引擎调用与 heap 表指针回写
// ===============================================================================

// payload 指针比较（升序），供排序与二分查找
static int gc_cmp_ptr(const void *a, const void *b) {
    void *const *pa = (void *const *)a;
    void *const *pb = (void *const *)b;
    if ((uintptr_t)*pa < (uintptr_t)*pb) return -1;
    if ((uintptr_t)*pa > (uintptr_t)*pb) return 1;
    return 0;
}

// 重定位记录条目与上下文（引擎回调时按 old_ptr 升序逐条追加）
typedef struct {
    void *old_ptr;
    void *new_ptr;
} gc_reloc_entry_t;

typedef struct {
    gc_reloc_entry_t *entries;
    size_t count;
    size_t capacity;
    bool failed;
} gc_reloc_ctx_t;

// 压缩引擎的重定位回调：记录一次 old_payload -> new_payload 的搬移
static void gc_on_relocate(void *ctx, void *old_payload, void *new_payload) {
    gc_reloc_ctx_t *rc = (gc_reloc_ctx_t *)ctx;
    if (!rc || rc->failed) return;
    if (rc->count >= rc->capacity) {
        // 搬移次数不会超过存活对象数，理论上不会发生
        rc->failed = true;
        return;
    }
    rc->entries[rc->count].old_ptr = old_payload;
    rc->entries[rc->count].new_ptr = new_payload;
    rc->count++;
}

// 在升序重定位表中查找 old_ptr，找到返回对应 new_ptr，未找到返回 NULL
static void *gc_reloc_lookup(const gc_reloc_ctx_t *rc, void *old_ptr) {
    size_t lo = 0, hi = rc->count;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if ((uintptr_t)rc->entries[mid].old_ptr < (uintptr_t)old_ptr) lo = mid + 1;
        else hi = mid;
    }
    if (lo < rc->count && rc->entries[lo].old_ptr == old_ptr) {
        return rc->entries[lo].new_ptr;
    }
    return NULL;
}

// 对多个进程堆一起执行全局标记-压缩：把所有 heap 中被 handle 引用的存活对象
// 搬到堆区前端，更新所有 heap 表中的指针。
int32_t am_gc_compact(am_allocator_t *heap_alloc, am_heap_t **heaps, size_t heap_count) {
    if (!heap_alloc || !heaps) return -1;

    /* 第一遍：收集所有 heap 表中指向堆对象的 payload 指针 */
    void **live = NULL;
    size_t live_count = 0;
    size_t live_cap = 0;
    for (size_t h = 0; h < heap_count; h++) {
        am_heap_t *heap = heaps[h];
        if (!heap || !heap->table) continue;
        size_t cap = heap->table->capacity;
        for (size_t i = 0; i < cap; i++) {
            am_value_t key = heap->table->slots[i].key;
            if (key == AM_MAP_KEY_EMPTY || key == AM_MAP_KEY_TOMBSTONE) continue;
            am_value_t v = heap->table->slots[i].value;
            if (!am_value_is_ptr(v)) continue;

            if (live_count >= live_cap) {
                live_cap = live_cap ? live_cap * 2 : 64;
                void **tmp = (void **)am_allocator_host_realloc(heap_alloc, live,
                                                                live_cap * sizeof(void *));
                if (!tmp) {
                    fprintf(stderr, "[gc] 压缩失败: live realloc 失败 (%zu bytes)\n",
                            live_cap * sizeof(void *));
                    am_allocator_host_free(heap_alloc, live);
                    return -1;
                }
                live = tmp;
            }
            live[live_count++] = am_value_to_ptr(v);
        }
    }

    /* 排序去重，得到压缩引擎要求的升序无重复存活对象数组 */
    if (live_count > 1) {
        qsort(live, live_count, sizeof(void *), gc_cmp_ptr);
    }
    size_t live_n = 0;
    for (size_t i = 0; i < live_count; i++) {
        if (live_n == 0 || live[i] != live[live_n - 1]) {
            live[live_n++] = live[i];
        }
    }

    /* 调用压缩引擎搬移存活对象；搬移次数不超过 live_n，故重定位表按 live_n 预分配 */
    gc_reloc_ctx_t rc = {NULL, 0, 0, false};
    if (live_n > 0) {
        rc.entries = (gc_reloc_entry_t *)am_allocator_host_malloc(heap_alloc,
                                                                  live_n * sizeof(gc_reloc_entry_t));
        if (!rc.entries) {
            fprintf(stderr, "[gc] 压缩失败: reloc malloc 失败 (%zu bytes)\n",
                    live_n * sizeof(gc_reloc_entry_t));
            am_allocator_host_free(heap_alloc, live);
            return -1;
        }
        rc.capacity = live_n;
    }

    int32_t ret = am_allocator_heap_compact(heap_alloc, live, live_n, gc_on_relocate, &rc);
    am_allocator_host_free(heap_alloc, live);
    if (ret != 0 || rc.failed) {
        am_allocator_host_free(heap_alloc, rc.entries);
        return -1;
    }

    /* 第二遍：回写所有 heap 表中仍指向旧地址的指针。
     * 统一在此回写（不区分主/次 slot），重定位表按 old_ptr 升序，二分查找。 */
    if (rc.count > 0) {
        for (size_t h = 0; h < heap_count; h++) {
            am_heap_t *heap = heaps[h];
            if (!heap || !heap->table) continue;
            size_t cap = heap->table->capacity;
            for (size_t i = 0; i < cap; i++) {
                am_value_t key = heap->table->slots[i].key;
                if (key == AM_MAP_KEY_EMPTY || key == AM_MAP_KEY_TOMBSTONE) continue;
                am_value_t v = heap->table->slots[i].value;
                if (!am_value_is_ptr(v)) continue;

                void *new_ptr = gc_reloc_lookup(&rc, am_value_to_ptr(v));
                if (new_ptr) {
                    heap->table->slots[i].value = am_make_value_of_ptr((am_object_t *)new_ptr);
                }
            }
        }
    }

    am_allocator_host_free(heap_alloc, rc.entries);
    return 0;
}


// ===============================================================================
// 编排：对进程池执行一轮完整 GC
// ===============================================================================

int32_t am_gc_collect(am_allocator_t *heap_alloc, am_process_t **process_pool,
                      size_t process_count, size_t gc_seq, int32_t force_compact) {
    if (!heap_alloc || !process_pool) return -1;

    /* 标记-清除：对所有现存进程执行 GC。
     * 仅 GC 成功的进程堆纳入压缩列表，避免压缩数组越界。 */
    am_heap_t **heaps = NULL;
    if (process_count > 0) {
        heaps = (am_heap_t **)am_allocator_host_malloc(heap_alloc, process_count * sizeof(am_heap_t *));
    }
    size_t heap_count = 0;
    for (size_t i = 0; i < process_count; i++) {
        am_process_t *proc = process_pool[i];
        if (!proc) continue;
        if (am_gc_process(proc) == 0 && proc->heap && heaps) {
            heaps[heap_count++] = proc->heap;
        }
    }

#if AM_HEAP_COMPACT_INTERVAL > 0
    /* 标记-压缩：在 GC 安全点一次性压缩所有进程的存活对象。
     * 所有进程共享同一个底层 heap_alloc，全局压缩避免互相覆盖。
     * force_compact 非 0 时无视 AM_HEAP_COMPACT_INTERVAL 当轮强制压缩。 */
    if ((force_compact || (gc_seq % AM_HEAP_COMPACT_INTERVAL) == 0) && heap_count > 0) {
        if (am_gc_compact(heap_alloc, heaps, heap_count) == 0) {
            am_allocator_pool_t *pool = am_allocator_pool_current();
            if (pool) {
                (void)am_allocator_pool_auto_adjust(pool);
            }
        }
    }
#else
    (void)gc_seq;
    (void)force_compact;
#endif

    if (heaps) am_allocator_host_free(heap_alloc, heaps);
    return 0;
}

int32_t am_gc_heap_watermark_level(am_allocator_t *heap_alloc) {
    if (!heap_alloc) return -1;

    // 第一趟：仅查询用量（不遍历空闲链表），低于碎片下限直接返回，避免每次检查都走链表
    size_t used = 0, capacity = 0;
    if (am_allocator_heap_usage(heap_alloc, &used, &capacity, NULL, NULL) != 0) return -1;
    if (capacity == 0) return -1;

    double ratio = (double)used / (double)capacity;
    if (ratio >= AM_GC_HEAP_CRITICAL_RATIO) return 2;
    if (ratio >= AM_GC_HEAP_HIGH_WATER_RATIO) return 1;
    if (ratio < AM_GC_HEAP_FRAG_FLOOR_RATIO) return 0;

    // 第二趟：碎片维度（需遍历空闲链表）。最大空闲块小于
    // max(容量 × AM_GC_HEAP_FRAG_MIN_BLOCK_RATIO, 近期最大分配请求) 时，
    // first-fit 随时可能失败，需要提前压缩整理（标记-清除+强制压缩）。
    size_t largest_free = 0, largest_request = 0;
    if (am_allocator_heap_usage(heap_alloc, NULL, NULL, &largest_free, &largest_request) != 0) return -1;
    double min_block = (double)capacity * AM_GC_HEAP_FRAG_MIN_BLOCK_RATIO;
    if ((double)largest_request > min_block) min_block = (double)largest_request;
    if ((double)largest_free < min_block) return 2;
    return 0;
}
/* ===== end:   src/am_gc.c ===== */

/* ===== begin: src/am_runtime.c ===== */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>
#include <math.h>
#include <stdbool.h>



// ===============================================================================
// 内部辅助函数
// ===============================================================================

// 将数值 TPV 统一转换为浮点数
am_float_t am_runtime_number_to_float(am_value_t v) {
    if (am_value_is_float(v)) return am_value_to_float(v);
    if (am_value_is_int(v)) return (am_float_t)am_value_to_int(v);
    if (am_value_is_uint(v)) return (am_float_t)am_value_to_uint(v);
    return 0.0;
}

// 将数值 TPV 统一（强制）转换为int
am_int_t am_runtime_number_to_int(am_value_t v) {
    if (am_value_is_float(v)) return (am_int_t)am_value_to_float(v);
    if (am_value_is_int(v)) return am_value_to_int(v);
    if (am_value_is_uint(v)) return (am_int_t)am_value_to_uint(v);
    return 0;
}

// 将数值 TPV 统一（强制）转换为uint
am_int_t am_runtime_number_to_uint(am_value_t v) {
    if (am_value_is_float(v)) return (am_uint_t)am_value_to_float(v);
    if (am_value_is_int(v)) return (am_uint_t)am_value_to_int(v);
    if (am_value_is_uint(v)) return am_value_to_uint(v);
    return 0;
}


// 释放 FIFO 中保存的 wstring 对象并销毁列表
static void destroy_fifo(am_runtime_t *rt, am_list_t *fifo) {
    if (!fifo) return;
    for (size_t i = 0; i < fifo->length; i++) {
        am_value_t v = am_list_get(rt->vm_alloc, fifo, i);
        if (am_value_is_ptr(v)) {
            am_object_t *obj = am_value_to_ptr(v);
            if (obj && obj->type == AM_OBJECT_TYPE_WSTRING) {
                am_wstring_destroy(rt->vm_alloc, (am_wstring_t *)obj);
            }
        }
    }
    am_list_destroy(rt->vm_alloc, fifo);
}


// ===============================================================================
// 队列 IPC 内部辅助函数
// ===============================================================================

// 根据 ID 在队列列表中线性查找队列。
static am_queue_t *runtime_find_queue(am_runtime_t *rt, size_t queue_id) {
    if (!rt || !rt->queue_list) return NULL;
    for (size_t i = 0; i < rt->queue_list->length; i++) {
        am_value_t v = am_list_get(rt->vm_alloc, rt->queue_list, i);
        if (!am_value_is_ptr(v)) continue;
        am_queue_t *q = (am_queue_t *)am_value_to_ptr(v);
        if (q && q->id == queue_id) return q;
    }
    return NULL;
}


// 分配并初始化一个等待者节点。
static am_queue_waiter_t *runtime_queue_waiter_create(am_runtime_t *rt, am_pid_t pid,
                                                       am_value_t value,
                                                       am_timestamp_t deadline_ms,
                                                       bool is_writer) {
    if (!rt) return NULL;
    am_queue_waiter_t *w = (am_queue_waiter_t *)am_malloc(rt->vm_alloc, sizeof(am_queue_waiter_t));
    if (!w) return NULL;
    w->pid = pid;
    w->value = value;
    w->deadline_ms = deadline_ms;
    w->is_writer = is_writer;
    w->next = NULL;
    return w;
}


// 唤醒指定进程：将结果压入操作数栈、步进 PC、置为 READY 并入队。
static void runtime_queue_wake_process(am_runtime_t *rt, am_pid_t pid, am_value_t result) {
    if (!rt || pid >= rt->process_poll_counter) return;
    am_process_t *proc = rt->process_pool[pid];
    if (!proc) return;

    if (am_process_push_operand(proc, result) != 0) return;
    am_process_step(proc);
    am_process_set_state(proc, AM_PROCESS_STATE_READY);

    am_list_t *new_queue = am_list_push(rt->vm_alloc, rt->process_queue,
                                        am_make_value_of_uint((am_uint_t)pid));
    if (new_queue) rt->process_queue = new_queue;
}


// 扫描所有队列，将已超时的等待者唤醒。
static void runtime_queue_check_waiters(am_runtime_t *rt) {
    if (!rt || !rt->queue_list) return;

    am_timestamp_t now = am_runtime_now_ms(rt);
    for (size_t i = 0; i < rt->queue_list->length; i++) {
        am_value_t qv = am_list_get(rt->vm_alloc, rt->queue_list, i);
        if (!am_value_is_ptr(qv)) continue;
        am_queue_t *q = (am_queue_t *)am_value_to_ptr(qv);
        if (!q) continue;

        am_queue_waiter_t **cur = &q->send_waiters;
        while (*cur) {
            am_queue_waiter_t *w = *cur;
            // 防御性检查：进程已被 kill，直接丢弃等待者而不唤醒
            am_process_t *wproc = am_runtime_get_process(rt, w->pid);
            if (wproc && wproc->state == AM_PROCESS_STATE_KILLED) {
                *cur = w->next;
                am_free(rt->vm_alloc, w);
                continue;
            }
            if (w->deadline_ms <= now) {
                *cur = w->next;
                runtime_queue_wake_process(rt, w->pid, AM_VALUE_FALSE);
                am_free(rt->vm_alloc, w);
            } else {
                cur = &w->next;
            }
        }

        cur = &q->recv_waiters;
        while (*cur) {
            am_queue_waiter_t *w = *cur;
            am_process_t *wproc = am_runtime_get_process(rt, w->pid);
            if (wproc && wproc->state == AM_PROCESS_STATE_KILLED) {
                *cur = w->next;
                am_free(rt->vm_alloc, w);
                continue;
            }
            if (w->deadline_ms <= now) {
                *cur = w->next;
                runtime_queue_wake_process(rt, w->pid, AM_VALUE_UNDEFINED);
                am_free(rt->vm_alloc, w);
            } else {
                cur = &w->next;
            }
        }
    }
}


// 判断当前是否还有阻塞等待者，并返回最近的超时时间。
static bool runtime_queue_has_waiters(am_runtime_t *rt, am_timestamp_t *nearest) {
    if (!rt || !rt->queue_list) return false;

    bool has = false;
    for (size_t i = 0; i < rt->queue_list->length; i++) {
        am_value_t qv = am_list_get(rt->vm_alloc, rt->queue_list, i);
        if (!am_value_is_ptr(qv)) continue;
        am_queue_t *q = (am_queue_t *)am_value_to_ptr(qv);
        if (!q) continue;

        for (am_queue_waiter_t *w = q->send_waiters; w; w = w->next) {
            if (nearest && (!has || w->deadline_ms < *nearest)) *nearest = w->deadline_ms;
            has = true;
        }
        for (am_queue_waiter_t *w = q->recv_waiters; w; w = w->next) {
            if (nearest && (!has || w->deadline_ms < *nearest)) *nearest = w->deadline_ms;
            has = true;
        }
    }
    return has;
}


am_queue_t *am_runtime_get_queue(am_runtime_t *rt, size_t queue_id) {
    return runtime_find_queue(rt, queue_id);
}


am_queue_t *am_runtime_queue_create(am_runtime_t *rt, size_t capacity) {
    if (!rt || capacity == 0) return NULL;

    am_queue_t *q = (am_queue_t *)am_malloc(rt->vm_alloc, sizeof(am_queue_t));
    if (!q) return NULL;

    q->id = rt->queue_next_id++;
    if (q->id == 0) q->id = rt->queue_next_id++; // 跳过 0
    q->capacity = capacity;
    q->items = am_list_create(rt->vm_alloc, capacity, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    q->send_waiters = NULL;
    q->recv_waiters = NULL;

    if (!q->items) {
        am_free(rt->vm_alloc, q);
        return NULL;
    }

    am_list_t *new_list = am_list_push(rt->vm_alloc, rt->queue_list,
                                       am_make_value_of_ptr((am_object_t *)q));
    if (!new_list) {
        am_list_destroy(rt->vm_alloc, q->items);
        am_free(rt->vm_alloc, q);
        return NULL;
    }
    rt->queue_list = new_list;
    return q;
}


int32_t am_runtime_queue_destroy(am_runtime_t *rt, am_queue_t *q) {
    if (!rt || !q) return 0;

    am_queue_waiter_t *w = q->send_waiters;
    while (w) {
        am_queue_waiter_t *next = w->next;
        am_free(rt->vm_alloc, w);
        w = next;
    }
    w = q->recv_waiters;
    while (w) {
        am_queue_waiter_t *next = w->next;
        am_free(rt->vm_alloc, w);
        w = next;
    }

    if (q->items) {
        am_list_destroy(rt->vm_alloc, q->items);
        q->items = NULL;
    }

    am_free(rt->vm_alloc, q);
    return 0;
}


int32_t am_runtime_queue_write(am_runtime_t *rt, am_queue_t *q, am_value_t value,
                               am_timestamp_t timeout_ms, am_process_t *proc) {
    if (!rt || !q || !proc) return -1;

    // 优先直接交给等待中的接收者
    if (q->recv_waiters) {
        am_queue_waiter_t *reader = q->recv_waiters;
        q->recv_waiters = reader->next;
        runtime_queue_wake_process(rt, reader->pid, value);
        am_free(rt->vm_alloc, reader);

        if (am_process_push_operand(proc, AM_VALUE_TRUE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    // 队列未满，直接入队
    if (q->items->length < q->capacity) {
        am_list_t *new_items = am_list_push(rt->vm_alloc, q->items, value);
        if (!new_items) return -1;
        q->items = new_items;

        if (am_process_push_operand(proc, AM_VALUE_TRUE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    // 队列已满：超时为 0 时立即失败
    if (timeout_ms == 0) {
        if (am_process_push_operand(proc, AM_VALUE_FALSE) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    // 阻塞当前发送者
    am_queue_waiter_t *w = runtime_queue_waiter_create(rt, proc->pid, value,
                                                         am_runtime_now_ms(rt) + timeout_ms, true);
    if (!w) return -1;
    w->next = q->send_waiters;
    q->send_waiters = w;

    am_process_set_state(proc, AM_PROCESS_STATE_BLOCKED);
    return 0;
}


int32_t am_runtime_queue_read(am_runtime_t *rt, am_queue_t *q, am_timestamp_t timeout_ms,
                              am_process_t *proc) {
    if (!rt || !q || !proc) return -1;

    // 队列非空，直接出队
    if (q->items->length > 0) {
        am_value_t v = am_list_shift(rt->vm_alloc, q->items);

        // 若有等待中的发送者，现在腾出了空间，允许一个发送者入队
        if (q->send_waiters) {
            am_queue_waiter_t *writer = q->send_waiters;
            q->send_waiters = writer->next;

            am_list_t *new_items = am_list_push(rt->vm_alloc, q->items, writer->value);
            if (new_items) q->items = new_items;
            // 即使 push 失败，也已腾出一个位置，发送者仍视为成功
            runtime_queue_wake_process(rt, writer->pid, AM_VALUE_TRUE);
            am_free(rt->vm_alloc, writer);
        }

        if (am_process_push_operand(proc, v) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    // 队列空：超时为 0 时立即失败
    if (timeout_ms == 0) {
        if (am_process_push_operand(proc, AM_VALUE_UNDEFINED) != 0) return -1;
        am_process_step(proc);
        return 0;
    }

    // 阻塞当前接收者
    am_queue_waiter_t *w = runtime_queue_waiter_create(rt, proc->pid, AM_VALUE_UNDEFINED,
                                                         am_runtime_now_ms(rt) + timeout_ms, false);
    if (!w) return -1;
    w->next = q->recv_waiters;
    q->recv_waiters = w;

    am_process_set_state(proc, AM_PROCESS_STATE_BLOCKED);
    return 0;
}


// 复制当前闭包的所有绑定到新闭包作为自由变量（用于 load/loadclosure/call）
// new_closure_hd 必须已绑定到一个新创建的 closure 对象。
static int32_t copy_current_closure_bindings_as_free_vars(am_process_t *proc, am_handle_t new_closure_hd) {
    am_obj_closure_t *current = am_process_get_current_closure(proc);
    am_obj_closure_t *new_closure = am_process_get_closure(proc, new_closure_hd);
    if (!new_closure) return -1;

    if (current) {
        for (size_t i = 0; i < current->length; i++) {
            am_binding_t *b = &current->bindings[i];
            new_closure = am_closure_init_free_var(proc->heap_alloc, new_closure, b->varid, b->value);
            if (!new_closure) return -1;
        }
        if (new_closure != am_process_get_closure(proc, new_closure_hd)) {
            am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, new_closure_hd,
                        am_make_value_of_ptr((am_object_t *)new_closure));
        }
    }
    return 0;
}


// 将值以人类可读形式输出到宽字符串缓冲区（简易实现）
static void value_to_wchar_buf(am_process_t *proc, am_value_t v, wchar_t *buf, size_t buf_size) {
    if (buf_size == 0) return;
    buf[0] = L'\0';

    if (am_value_is_handle(v)) {
        am_handle_t hd = am_value_to_handle(v);
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (am_value_is_ptr(obj_val)) {
            am_object_t *obj = am_value_to_ptr(obj_val);
            if (obj->type == AM_OBJECT_TYPE_WSTRING) {
                am_wstring_t *ws = (am_wstring_t *)obj;
                size_t n = ws->length;
                if (n >= buf_size) n = buf_size - 1;
                for (size_t i = 0; i < n; i++) {
                    buf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
                }
                buf[n] = L'\0';
                return;
            }
            else if (obj->type == AM_OBJECT_TYPE_LIST) {
                swprintf(buf, buf_size, L"#<list:%#x>", (uintptr_t)obj);
                return;
            }
            else if (obj->type == AM_OBJECT_TYPE_CLOSURE) {
                swprintf(buf, buf_size, L"#<closure:%#x>", (uintptr_t)obj);
                return;
            }
            else if (obj->type == AM_OBJECT_TYPE_CONTINUATION) {
                swprintf(buf, buf_size, L"#<continuation:%#x>", (uintptr_t)obj);
                return;
            }
        }
        swprintf(buf, buf_size, L"#<handle:%zu>", hd);
    }
    else if (am_value_is_float(v)) {
        swprintf(buf, buf_size, L"%g", (double)am_value_to_float(v));
    }
    else if (am_value_is_int(v)) {
        swprintf(buf, buf_size, L"%lld", (long long)am_value_to_int(v));
    }
    else if (am_value_is_uint(v)) {
        swprintf(buf, buf_size, L"%llu", (unsigned long long)am_value_to_uint(v));
    }
    else if (am_value_is_boolean(v)) {
        swprintf(buf, buf_size, L"%ls", am_value_to_boolean(v) ? L"#t" : L"#f");
    }
    else if (am_value_is_null(v)) {
        swprintf(buf, buf_size, L"#null");
    }
    else if (am_value_is_undefined(v)) {
        swprintf(buf, buf_size, L"#undefined");
    }
    else if (am_value_is_symbol(v)) {
        am_symbol_t sym = am_value_to_symbol(v);
        wchar_t *s = am_vocab_get(proc->vm_alloc, proc->symbol_vocab, &sym);
        if (!s) {
            swprintf(buf, buf_size, L"#<symbol>");
            return;
        }
        else {
            swprintf(buf, buf_size, L"%ls", s);
        }
    }
    else if (am_value_is_varid(v)) {
        am_varid_t vid = am_value_to_varid(v);
        wchar_t *s = am_vocab_get(proc->vm_alloc, proc->var_vocab, &vid);
        if (!s) {
            swprintf(buf, buf_size, L"#<varid:%zu>", vid);
            return;
        }
        else {
            swprintf(buf, buf_size, L"#<builtin:%ls>", s);
        }
    }
    // else {
    //     swprintf(buf, buf_size, L"#<value>");
    // }
}


// 递归比较两个值是否结构相等（用于 equal?）
static bool runtime_value_equal(am_process_t *proc, am_value_t a, am_value_t b) {
    if (a == b) return true;

    // 数字按数值比较
    if (am_value_is_number(a) && am_value_is_number(b)) {
        am_float_t fa = am_runtime_number_to_float(a);
        am_float_t fb = am_runtime_number_to_float(b);
        return fa == fb;
    }

    // 同为 handle 时按对象类型递归比较
    if (am_value_is_handle(a) && am_value_is_handle(b)) {
        am_handle_t ha = am_value_to_handle(a);
        am_handle_t hb = am_value_to_handle(b);
        am_value_t va = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, ha);
        am_value_t vb = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hb);
        if (!am_value_is_ptr(va) || !am_value_is_ptr(vb)) return false;

        am_object_t *oa = am_value_to_ptr(va);
        am_object_t *ob = am_value_to_ptr(vb);
        if (oa->type != ob->type) return false;

        if (oa->type == AM_OBJECT_TYPE_LIST) {
            am_list_t *la = (am_list_t *)oa;
            am_list_t *lb = (am_list_t *)ob;
            if (la->length != lb->length) return false;
            for (size_t i = 0; i < la->length; i++) {
                if (!runtime_value_equal(proc, la->children[i], lb->children[i])) return false;
            }
            return true;
        }

        if (oa->type == AM_OBJECT_TYPE_WSTRING) {
            am_wstring_t *wa = (am_wstring_t *)oa;
            am_wstring_t *wb = (am_wstring_t *)ob;
            if (wa->length != wb->length) return false;
            for (size_t i = 0; i < wa->length; i++) {
                if (wa->content[i] != wb->content[i]) return false;
            }
            return true;
        }
    }

    return false;
}


// ===============================================================================
// 第一类：基本存取指令
// ===============================================================================

static int32_t op_store(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!am_value_is_varid(operand)) return -1;

    am_varid_t varid = am_value_to_varid(operand);
    am_value_t value = am_process_pop_operand(proc);

    am_obj_closure_t *current = am_process_get_current_closure(proc);
    if (!current) return -1;

    am_obj_closure_t *new_current = am_closure_init_bound_var(proc->heap_alloc, current, varid, value);
    if (!new_current) return -1;
    if (new_current != current) {
        am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, proc->current_closure_handle,
                    am_make_value_of_ptr((am_object_t *)new_current));
    }

    am_process_step(proc);
    return 0;
}


static int32_t op_load(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!am_value_is_varid(operand)) return -1;

    am_varid_t varid = am_value_to_varid(operand);
    am_value_t value = am_process_dereference(proc, varid);
    if (value == (am_value_t)UINTPTR_MAX) {
        wchar_t *name = am_vocab_get(proc->vm_alloc, proc->var_vocab, &varid);
        wchar_t errmsg[256];
        swprintf(errmsg, 256, L"[Runtime] load: 变量 %ls 未定义\n", name ? name : L"?");
        am_runtime_error(rt, errmsg);
        return -1;
    }

    // 若值为 iaddr，说明是 lambda 标签解析后的地址，创建闭包
    if (am_value_is_iaddr(value)) {
        am_iaddr_t iaddr = am_value_to_iaddr(value);
        am_handle_t closure_hd = am_process_make_closure(proc, iaddr, proc->current_closure_handle);
        if (closure_hd == AM_HANDLE_NULL) return -1;
        if (copy_current_closure_bindings_as_free_vars(proc, closure_hd) != 0) return -1;
        am_process_push_operand(proc, am_make_value_of_handle(closure_hd));
    }
    else {
        am_process_push_operand(proc, value);
    }

    am_process_step(proc);
    return 0;
}


static int32_t op_loadclosure(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!am_value_is_iaddr(operand)) return -1;

    am_iaddr_t iaddr = am_value_to_iaddr(operand);
    am_handle_t closure_hd = am_process_make_closure(proc, iaddr, proc->current_closure_handle);
    if (closure_hd == AM_HANDLE_NULL) return -1;
    if (copy_current_closure_bindings_as_free_vars(proc, closure_hd) != 0) return -1;

    am_process_push_operand(proc, am_make_value_of_handle(closure_hd));
    am_process_step(proc);
    return 0;
}


static int32_t op_push(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (am_process_push_operand(proc, operand) != 0) return -1;
    am_process_step(proc);
    return 0;
}


static int32_t op_pop(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_process_pop_operand(proc);
    am_process_step(proc);
    return 0;
}


static int32_t op_swap(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t top1 = am_process_pop_operand(proc);
    am_value_t top2 = am_process_pop_operand(proc);
    if (top1 == (am_value_t)UINTPTR_MAX || top2 == (am_value_t)UINTPTR_MAX) return -1;
    am_process_push_operand(proc, top1);
    am_process_push_operand(proc, top2);
    am_process_step(proc);
    return 0;
}


static int32_t op_set(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!am_value_is_varid(operand)) return -1;

    am_varid_t varid = am_value_to_varid(operand);
    am_value_t right = am_process_pop_operand(proc);

    am_handle_t current_h = proc->current_closure_handle;
    am_obj_closure_t *current = am_process_get_current_closure(proc);
    if (!current) return -1;

    // 若当前闭包中存在该自由变量，则更新自由变量（带脏标记）
    if (am_closure_has_free_var(proc->heap_alloc, current, varid) == 0) {
        am_obj_closure_t *new_current = am_closure_set_free_var(proc->heap_alloc, current, varid, right);
        if (!new_current) return -1;
        if (new_current != current) {
            am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, current_h,
                        am_make_value_of_ptr((am_object_t *)new_current));
            current = new_current;
        }
    }

    // 沿闭包链上溯，找到约束变量定义位置并更新（带脏标记）
    am_handle_t h = current_h;
    while (h != AM_HANDLE_NULL) {
        am_obj_closure_t *closure = am_process_get_closure(proc, h);
        if (!closure) break;
        if (am_closure_has_bound_var(proc->heap_alloc, closure, varid) == 0) {
            am_obj_closure_t *new_closure = am_closure_set_bound_var(proc->heap_alloc, closure, varid, right);
            if (!new_closure) return -1;
            if (new_closure != closure) {
                am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, h,
                            am_make_value_of_ptr((am_object_t *)new_closure));
            }
            break;
        }
        h = closure->parent;
    }

    am_process_step(proc);
    return 0;
}


// ===============================================================================
// 第二类：分支跳转指令
// ===============================================================================

// 判断变量名是否对应 builtin 操作。
// 基于 AM_GLOBAL_BUILTIN_VAR 与 AM_BUILTIN_OPCODE_MAP：若变量名是 builtin 且映射opcode非 -1，则是 builtin。
static int32_t check_builtin_varid(am_process_t *proc, am_varid_t varid) {
    if (!proc || !proc->var_vocab) return -1;
    wchar_t *name = am_vocab_get(proc->vm_alloc, proc->var_vocab, &varid);
    if (!name) return -1;

    for (size_t i = 0; i < AM_GLOBAL_BUILTIN_VAR_NUM; i++) {
        if (wcscmp(name, AM_GLOBAL_BUILTIN_VAR[i]) == 0) {
            return AM_BUILTIN_OPCODE_MAP[i];
        }
    }
    return -1;
}

static int32_t op_callnative(am_runtime_t *rt, am_process_t *proc, am_value_t operand);

// 功能描述：检查call指令参数（已解析为varid）是否是本地宿主库的调用
// 是返回0，不是返回-1
int32_t am_runtime_check_native_ref(am_runtime_t *rt, am_process_t *proc, am_varid_t varid) {
    (void)rt;
    if (!proc || !proc->var_type) return -1;

    if ((size_t)varid >= proc->var_type->length) return -1;

    am_value_t type_val = am_list_get(proc->vm_alloc, proc->var_type, (size_t)varid);
    if (!am_value_is_uint(type_val)) return -1;

    return (am_value_to_uint(type_val) == (am_uint_t)AM_VAR_TYPE_NATIVE_REF) ? 0 : -1;
}


// ===============================================================================
// dynamic-wind 内部辅助函数
// ===============================================================================

// 创建 dynamic-wind 条目：[before_handle, after_handle, mark_value, saved_value, opstack_base]
// opstack_base 记录进入 dynamic-wind 时 opstack 的长度，用于判断 before/thunk/after 是否压入返回值。
// 条目本身为 AM_OBJECT_TYPE_LIST 对象，绑定到 proc->heap。成功返回 handle，失败返回 AM_HANDLE_NULL。
static am_handle_t dynamic_wind_create_entry(am_process_t *proc, am_handle_t before, am_handle_t after, am_uint_t mark, size_t opstack_base) {
    if (!proc || !proc->heap || !proc->heap_alloc) return AM_HANDLE_NULL;

    am_list_t *entry = am_list_create(proc->heap_alloc, 5, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!entry) return AM_HANDLE_NULL;

    entry->children[0] = am_make_value_of_handle(before);
    entry->children[1] = am_make_value_of_handle(after);
    entry->children[2] = am_make_value_of_uint(mark);
    entry->children[3] = AM_VALUE_UNDEFINED;
    entry->children[4] = am_make_value_of_uint((am_uint_t)opstack_base);
    entry->length = 5;

    am_handle_t hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (hd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, entry);
        return AM_HANDLE_NULL;
    }

    am_value_t entry_value = am_make_value_of_ptr((am_object_t *)entry);
    if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, hd, entry_value) != 0) {
        am_list_destroy(proc->heap_alloc, entry);
        am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        return AM_HANDLE_NULL;
    }

    return hd;
}

static am_list_t *am_runtime__dynamic_wind_get_entry(am_process_t *proc, am_handle_t entry_hd) {
    if (!proc || !proc->heap || entry_hd == AM_HANDLE_NULL) return NULL;
    am_value_t v = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, entry_hd);
    if (!am_value_is_ptr(v)) return NULL;
    am_object_t *obj = am_value_to_ptr(v);
    if (!obj || obj->type != AM_OBJECT_TYPE_LIST) return NULL;
    return (am_list_t *)obj;
}

static inline am_handle_t am_runtime__dynamic_wind_entry_before(am_list_t *entry) {
    if (!entry || entry->length < 4) return AM_HANDLE_NULL;
    return am_value_to_handle(entry->children[0]);
}

static inline am_handle_t am_runtime__dynamic_wind_entry_after(am_list_t *entry) {
    if (!entry || entry->length < 4) return AM_HANDLE_NULL;
    return am_value_to_handle(entry->children[1]);
}

static inline am_value_t am_runtime__dynamic_wind_entry_saved(am_list_t *entry) {
    if (!entry || entry->length < 4) return AM_VALUE_UNDEFINED;
    return entry->children[3];
}

static inline void am_runtime__dynamic_wind_entry_set_saved(am_list_t *entry, am_value_t v) {
    if (!entry || entry->length < 4) return;
    entry->children[3] = v;
}

static inline size_t dynamic_wind_entry_base(am_list_t *entry) {
    if (!entry || entry->length < 5) return 0;
    return (size_t)am_value_to_uint(entry->children[4]);
}

// 调用一个闭包 handle，返回地址为 return_iaddr
static int32_t dynamic_wind_call_closure(am_runtime_t *rt, am_process_t *proc, am_handle_t closure_handle, am_iaddr_t return_iaddr) {
    (void)rt;
    if (!proc) return -1;

    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, closure_handle);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (!obj || obj->type != AM_OBJECT_TYPE_CLOSURE) return -1;
    am_obj_closure_t *closure = (am_obj_closure_t *)obj;

    am_value_t closure_val = am_make_value_of_handle(proc->current_closure_handle);
    am_value_t ret_val = am_make_value_of_iaddr(return_iaddr);
    if (am_process_push_stack_frame(proc, closure_val, ret_val) != 0) return -1;

    am_process_set_current_closure(proc, closure_handle);
    am_process_goto(proc, closure->iaddr);
    return 0;
}

// 将 entry handle 压入 proc->dynamic_wind_stack
static int32_t dynamic_wind_stack_push(am_process_t *proc, am_handle_t entry_hd) {
    if (!proc || !proc->dynamic_wind_stack) return -1;
    am_list_t *lst = am_list_push(proc->vm_alloc, proc->dynamic_wind_stack, am_make_value_of_handle(entry_hd));
    if (!lst) return -1;
    proc->dynamic_wind_stack = lst;
    return 0;
}

// 从 proc->dynamic_wind_stack 弹出 entry handle
static am_handle_t dynamic_wind_stack_pop(am_process_t *proc) {
    if (!proc || !proc->dynamic_wind_stack || proc->dynamic_wind_stack->length == 0) return AM_HANDLE_NULL;
    return am_value_to_handle(am_list_pop(proc->vm_alloc, proc->dynamic_wind_stack));
}


// call / tailcall 的共享实现。return_target 为 SIZE_MAX 表示尾调用不压栈帧。
static int32_t op_call_async(am_runtime_t *rt, am_process_t *proc, am_value_t operand, am_iaddr_t return_target) {
    (void)rt;

    am_value_t target;
    if (am_value_is_varid(operand)) {
        am_varid_t varid = am_value_to_varid(operand);
        // 判断 native 调用
        if (am_runtime_check_native_ref(rt, proc, varid) == 0) {
            return op_callnative(rt, proc, operand);
        }
        else {
            target = am_process_dereference(proc, varid);
            if (target == (am_value_t)UINTPTR_MAX) {
                wchar_t *name = am_vocab_get(proc->vm_alloc, proc->var_vocab, &varid);
                wchar_t errmsg[256];
                swprintf(errmsg, 256, L"[Runtime] call: 变量 %ls 未定义\n", name ? name : L"?");
                am_runtime_error(rt, errmsg);
                return -1;
            }
        }
    }
    else {
        target = operand;
    }

    // iaddr：直接调用 lambda
    if (am_value_is_iaddr(target)) {
        am_iaddr_t iaddr = am_value_to_iaddr(target);

        if (return_target != SIZE_MAX) {
            am_value_t closure_val = am_make_value_of_handle(proc->current_closure_handle);
            am_value_t ret_val = am_make_value_of_iaddr(return_target);
            if (am_process_push_stack_frame(proc, closure_val, ret_val) != 0) return -1;
        }
        else {
            // 尾调用优化：若目标与当前闭包地址相同，复用当前闭包
            am_obj_closure_t *cur = am_process_get_current_closure(proc);
            if (cur && cur->iaddr == iaddr) {
                am_process_goto(proc, iaddr);
                return 0;
            }
        }

        am_handle_t closure_hd = am_process_make_closure(proc, iaddr, proc->current_closure_handle);
        if (closure_hd == AM_HANDLE_NULL) return -1;
        if (copy_current_closure_bindings_as_free_vars(proc, closure_hd) != 0) return -1;

        am_process_set_current_closure(proc, closure_hd);
        am_process_goto(proc, iaddr);
        return 0;
    }

    // handle：闭包或 continuation
    if (am_value_is_handle(target)) {
        am_handle_t hd = am_value_to_handle(target);
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (!am_value_is_ptr(obj_val)) return -1;
        am_object_t *obj = am_value_to_ptr(obj_val);

        if (obj->type == AM_OBJECT_TYPE_CLOSURE) {
            am_obj_closure_t *closure = (am_obj_closure_t *)obj;
            if (return_target != SIZE_MAX) {
                am_value_t closure_val = am_make_value_of_handle(proc->current_closure_handle);
                am_value_t ret_val = am_make_value_of_iaddr(return_target);
                if (am_process_push_stack_frame(proc, closure_val, ret_val) != 0) return -1;
            }
            am_process_set_current_closure(proc, hd);
            am_process_goto(proc, closure->iaddr);
            return 0;
        }
        else if (obj->type == AM_OBJECT_TYPE_CONTINUATION) {
            if (return_target != SIZE_MAX) {
                am_value_t closure_val = am_make_value_of_handle(proc->current_closure_handle);
                am_value_t ret_val = am_make_value_of_iaddr(return_target);
                if (am_process_push_stack_frame(proc, closure_val, ret_val) != 0) return -1;
            }
            am_value_t top = am_process_pop_operand(proc);
            am_iaddr_t cont_target = am_process_load_continuation(proc, hd, top);
            if (cont_target == SIZE_MAX) return -1;
            am_process_goto(proc, cont_target);
            return 0;
        }
        else {
            am_runtime_error(rt, L"[Runtime] call: 目标对象类型错误\n");
            return -1;
        }
    }

    if (am_value_is_varid(target)) {
        am_varid_t v = am_value_to_varid(target);
        // 判断builtin函数
        int32_t opcode = check_builtin_varid(proc, v);
        if (opcode >= 0) {
            return am_runtime_op_dispatch(rt, proc, (uint32_t)opcode, operand);
        }
        // 判断 native 调用
        else if (am_runtime_check_native_ref(rt, proc, v) == 0) {
            return op_callnative(rt, proc, target);
        }
    }

    am_runtime_error(rt, L"[Runtime] call: 错误的调用目标\n");

    return -1;
}


static int32_t op_call(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    return op_call_async(rt, proc, operand, proc->PC + 1);
}


static int32_t op_tailcall(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    return op_call_async(rt, proc, operand, SIZE_MAX);
}


static int32_t op_callnative(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!proc || !proc->var_vocab) return -1;
    if (!am_value_is_varid(operand)) return -1;

    am_varid_t varid = am_value_to_varid(operand);
    size_t idx = (size_t)varid;
    wchar_t *name = am_vocab_get(proc->vm_alloc, proc->var_vocab, &idx);
    if (!name) return -1;

    // 变量名应为 "LibID.funcName" 形式，且只能有一个点号
    wchar_t *dot = wcschr(name, L'.');
    if (!dot || dot == name || dot[1] == L'\0') {
        am_runtime_error(rt, L"[Runtime] callnative: 错误的native变量名\n");
        return -1;
    }
    if (wcschr(dot + 1, L'.')) {
        am_runtime_error(rt, L"[Runtime] callnative: native变量名包含多个点号\n");
        return -1;
    }

    size_t len = wcslen(name);
    wchar_t *buf = (wchar_t *)am_malloc(proc->vm_alloc, (len + 1) * sizeof(wchar_t));
    if (!buf) return -1;
    wcscpy(buf, name);

    wchar_t *prefix = buf;
    wchar_t *suffix = buf + (dot - name);
    *suffix = L'\0';
    suffix++;

    am_native_func_t func = am_native_find_func(prefix, suffix);
    if (!func) {
        wchar_t errmsg[256];
        swprintf(errmsg, 256, L"[Runtime] callnative: 未找到native函数 %ls.%ls\n", prefix, suffix);
        am_runtime_error(rt, errmsg);
        am_free(proc->vm_alloc, buf);
        return -1;
    }
    am_free(proc->vm_alloc, buf);

    return func(rt, proc);
}


static int32_t op_return(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t closure_val, ret_val;
    if (am_process_pop_stack_frame(proc, &closure_val, &ret_val) != 0) {
        am_runtime_error(rt, L"[Runtime] return: 函数调用栈为空\n");
        return -1;
    }

    am_process_set_current_closure(proc, am_value_to_handle(closure_val));
    am_process_goto(proc, am_value_to_iaddr(ret_val));
    return 0;
}


static int32_t op_capturecc(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!am_value_is_varid(operand)) return -1;

    am_varid_t varid = am_value_to_varid(operand);
    // 续体返回目标：capturecc 之后固定有 load 和 call 两条指令，返回点位于 PC+3
    am_iaddr_t ret_target = proc->PC + 3;

    am_handle_t cont_hd = am_process_capture_continuation(proc, ret_target);
    if (cont_hd == AM_HANDLE_NULL) return -1;

    am_obj_closure_t *current = am_process_get_current_closure(proc);
    if (!current) return -1;
    am_obj_closure_t *new_current = am_closure_init_bound_var(
        proc->heap_alloc, current, varid, am_make_value_of_handle(cont_hd));
    if (!new_current) return -1;
    if (new_current != current) {
        am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, proc->current_closure_handle,
                    am_make_value_of_ptr((am_object_t *)new_current));
    }

    am_process_step(proc);
    return 0;
}


// dynamic-wind 第一阶段：opstack 上有 [..., before, thunk, after]
// 创建条目，暂存 entry/thunk，调用 before
static int32_t op_dynamicwind(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt; (void)operand;
    if (!proc) return -1;

    am_value_t after_val = am_process_pop_operand(proc);
    am_value_t thunk_val = am_process_pop_operand(proc);
    am_value_t before_val = am_process_pop_operand(proc);

    if (!am_value_is_handle(before_val) || !am_value_is_handle(thunk_val) || !am_value_is_handle(after_val)) {
        am_runtime_error(rt, L"[Runtime] dynamic-wind: 参数必须是闭包\n");
        return -1;
    }

    am_handle_t before_hd = am_value_to_handle(before_val);
    am_handle_t thunk_hd = am_value_to_handle(thunk_val);
    am_handle_t after_hd = am_value_to_handle(after_val);

    if (!am_process_get_closure(proc, before_hd) || !am_process_get_closure(proc, thunk_hd) || !am_process_get_closure(proc, after_hd)) {
        am_runtime_error(rt, L"[Runtime] dynamic-wind: 参数必须是闭包\n");
        return -1;
    }

    size_t opstack_base = am_process_length_of_opstack(proc);
    am_uint_t mark = (am_uint_t)proc->dynamic_wind_mark_counter++;
    am_handle_t entry_hd = dynamic_wind_create_entry(proc, before_hd, after_hd, mark, opstack_base);
    if (entry_hd == AM_HANDLE_NULL) return -1;

    proc->current_dynamic_wind_entry = entry_hd;
    proc->current_dynamic_wind_thunk = thunk_hd;

    return dynamic_wind_call_closure(rt, proc, before_hd, proc->PC + 1);
}


// 辅助：将 opstack 恢复到指定长度（弹出多余部分），返回最后弹出的值（若无则返回 AM_VALUE_UNDEFINED）
static am_value_t dynamic_wind_trim_opstack_to_base(am_process_t *proc, size_t base) {
    size_t current_len = am_process_length_of_opstack(proc);
    am_value_t last = AM_VALUE_UNDEFINED;
    while (current_len > base) {
        last = am_process_pop_operand(proc);
        current_len--;
    }
    return last;
}


// dynamic-wind 第二阶段：before 已返回
// 丢弃 before 返回值（若有），将条目压入 dynamic_wind_stack，调用 thunk
static int32_t op_dynamicwind_after_before(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt; (void)operand;
    if (!proc) return -1;

    am_handle_t entry_hd = proc->current_dynamic_wind_entry;

    if (entry_hd == AM_HANDLE_NULL) {
        am_runtime_error(rt, L"[Runtime] dynamicwind_after_before: 无当前条目\n");
        return -1;
    }

    am_list_t *entry = am_runtime__dynamic_wind_get_entry(proc, entry_hd);
    if (!entry) return -1;
    size_t opstack_base = dynamic_wind_entry_base(entry);

    // 丢弃 before 的返回值（0 个或 1 个）
    dynamic_wind_trim_opstack_to_base(proc, opstack_base);

    proc->current_dynamic_wind_entry = AM_HANDLE_NULL;

    if (dynamic_wind_stack_push(proc, entry_hd) != 0) return -1;

    am_handle_t thunk_hd = proc->current_dynamic_wind_thunk;
    if (thunk_hd == AM_HANDLE_NULL) {
        am_runtime_error(rt, L"[Runtime] dynamicwind_after_before: 无当前 thunk\n");
        return -1;
    }

    return dynamic_wind_call_closure(rt, proc, thunk_hd, proc->PC + 1);
}


// dynamic-wind 第三阶段：thunk 已返回
// 保存返回值（若有），弹出 dynamic_wind_stack 条目，将条目 handle 压入 dynamic_wind_after_stack，调用 after
static int32_t op_dynamicwind_before_after(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt; (void)operand;
    if (!proc) return -1;

    am_handle_t entry_hd = dynamic_wind_stack_pop(proc);
    if (entry_hd == AM_HANDLE_NULL) {
        am_runtime_error(rt, L"[Runtime] dynamicwind_before_after: dynamic_wind_stack 为空\n");
        return -1;
    }

    am_list_t *entry = am_runtime__dynamic_wind_get_entry(proc, entry_hd);
    if (!entry) return -1;
    size_t opstack_base = dynamic_wind_entry_base(entry);

    // thunk 的返回值：若 thunk 压入了返回值，则弹出并保存；否则保存 AM_VALUE_UNDEFINED
    am_value_t thunk_result = dynamic_wind_trim_opstack_to_base(proc, opstack_base);
    am_runtime__dynamic_wind_entry_set_saved(entry, thunk_result);

    // 将条目 handle 压入 dynamic_wind_after_stack，供 dynamicwind_done 取回 saved_value
    am_list_t *after_stack = am_list_push(proc->vm_alloc, proc->dynamic_wind_after_stack,
                                          am_make_value_of_handle(entry_hd));
    if (!after_stack) return -1;
    proc->dynamic_wind_after_stack = after_stack;

    proc->current_dynamic_wind_thunk = AM_HANDLE_NULL;

    am_handle_t after_hd = am_runtime__dynamic_wind_entry_after(entry);
    if (after_hd == AM_HANDLE_NULL) return -1;

    return dynamic_wind_call_closure(rt, proc, after_hd, proc->PC + 1);
}


// dynamic-wind 第四阶段：after 已返回
// 丢弃 after 返回值（若有），从 dynamic_wind_after_stack 弹出条目，将 saved_value 压回 opstack，继续执行
static int32_t op_dynamicwind_done(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt; (void)operand;
    if (!proc) return -1;

    if (!proc->dynamic_wind_after_stack || proc->dynamic_wind_after_stack->length == 0) {
        am_runtime_error(rt, L"[Runtime] dynamicwind_done: dynamic_wind_after_stack 为空\n");
        return -1;
    }

    am_handle_t entry_hd = am_value_to_handle(
        am_list_pop(proc->vm_alloc, proc->dynamic_wind_after_stack));

    am_list_t *entry = am_runtime__dynamic_wind_get_entry(proc, entry_hd);
    if (!entry) return -1;
    size_t opstack_base = dynamic_wind_entry_base(entry);

    // 丢弃 after 的返回值（0 个或 1 个）
    dynamic_wind_trim_opstack_to_base(proc, opstack_base);

    am_value_t saved = am_runtime__dynamic_wind_entry_saved(entry);
    if (am_process_push_operand(proc, saved) != 0) return -1;

    am_process_step(proc);
    return 0;
}


// wind 跳板：continuation 恢复前执行 afters / befores
static int32_t op_wind(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt; (void)operand;
    if (!proc) return -1;

    switch (proc->wind_state) {
        case 1: { // 执行 afters（从内到外）
            if (proc->pending_after_count > 0) {
                am_handle_t entry_hd = proc->pending_after_entries[--proc->pending_after_count];
                // 验证条目在 dynamic_wind_stack 栈顶
                am_handle_t top_hd = proc->dynamic_wind_stack && proc->dynamic_wind_stack->length > 0
                    ? am_value_to_handle(proc->dynamic_wind_stack->children[proc->dynamic_wind_stack->length - 1])
                    : AM_HANDLE_NULL;
                if (top_hd != entry_hd) {
                    am_runtime_error(rt, L"[Runtime] op_wind: after 条目与栈顶不一致\n");
                    return -1;
                }
                dynamic_wind_stack_pop(proc);

                am_list_t *entry = am_runtime__dynamic_wind_get_entry(proc, entry_hd);
                if (!entry) return -1;
                am_handle_t after_hd = am_runtime__dynamic_wind_entry_after(entry);
                if (after_hd == AM_HANDLE_NULL) return -1;

                return dynamic_wind_call_closure(rt, proc, after_hd, proc->wind_trampoline_iaddr);
            }
            // afters 执行完毕，转入 befores
            proc->wind_state = 2;
            // 不推进 PC，下一条 tick 继续执行本指令的 state 2 分支
            return 0;
        }
        case 2: { // 执行 befores（从外到内）
            if (proc->pending_before_count > 0) {
                am_handle_t entry_hd = proc->pending_before_entries[0];
                // 将 pending_before_entries 前移
                for (size_t i = 1; i < proc->pending_before_count; i++) {
                    proc->pending_before_entries[i - 1] = proc->pending_before_entries[i];
                }
                proc->pending_before_count--;

                if (dynamic_wind_stack_push(proc, entry_hd) != 0) return -1;

                am_list_t *entry = am_runtime__dynamic_wind_get_entry(proc, entry_hd);
                if (!entry) return -1;
                am_handle_t before_hd = am_runtime__dynamic_wind_entry_before(entry);
                if (before_hd == AM_HANDLE_NULL) return -1;

                return dynamic_wind_call_closure(rt, proc, before_hd, proc->wind_trampoline_iaddr);
            }
            // befores 执行完毕，转入恢复续体
            proc->wind_state = 3;
            // 不推进 PC，下一条 tick 继续执行本指令的 state 3 分支
            return 0;
        }
        case 3: { // 真正恢复续体
            am_handle_t cont_hd = proc->pending_cont_handle;
            am_value_t value = proc->pending_cont_value;

            am_iaddr_t cont_target = am_process_restore_continuation_snapshot(proc, cont_hd);
            if (cont_target == SIZE_MAX) return -1;

            if (am_process_push_operand(proc, value) != 0) return -1;

            // 清空 wind 状态
            proc->wind_state = 0;
            proc->pending_cont_handle = AM_HANDLE_NULL;
            proc->pending_cont_value = AM_VALUE_UNDEFINED;
            if (proc->pending_after_entries) {
                am_free(proc->vm_alloc, proc->pending_after_entries);
                proc->pending_after_entries = NULL;
            }
            proc->pending_after_count = 0;
            if (proc->pending_before_entries) {
                am_free(proc->vm_alloc, proc->pending_before_entries);
                proc->pending_before_entries = NULL;
            }
            proc->pending_before_count = 0;

            am_process_goto(proc, cont_target);
            return 0;
        }
        default: {
            am_runtime_error(rt, L"[Runtime] op_wind: 非法 wind_state\n");
            return -1;
        }
    }
}


static int32_t op_iftrue(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!am_value_is_iaddr(operand)) return -1;

    am_value_t condition = am_process_pop_operand(proc);
    if (condition != AM_VALUE_FALSE) {
        am_process_goto(proc, am_value_to_iaddr(operand));
    }
    else {
        am_process_step(proc);
    }
    return 0;
}


static int32_t op_iffalse(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!am_value_is_iaddr(operand)) return -1;

    am_value_t condition = am_process_pop_operand(proc);
    if (condition == AM_VALUE_FALSE) {
        am_process_goto(proc, am_value_to_iaddr(operand));
    }
    else {
        am_process_step(proc);
    }
    return 0;
}


static int32_t op_goto(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!am_value_is_iaddr(operand)) return -1;
    am_process_goto(proc, am_value_to_iaddr(operand));
    return 0;
}


// ===============================================================================
// 第三类：列表操作指令
// ===============================================================================

static int32_t op_car(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t list_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(list_val)) return -1;

    am_handle_t list_hd = am_value_to_handle(list_val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return -1;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->length == 0) return -1;

    am_process_push_operand(proc, lst->children[0]);
    am_process_step(proc);
    return 0;
}


static int32_t op_cdr(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t list_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(list_val)) return -1;

    am_handle_t list_hd = am_value_to_handle(list_val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return -1;

    am_list_t *lst = (am_list_t *)obj;
    if (lst->length == 0) return -1;

    am_list_t *new_lst = am_list_create(proc->heap_alloc, lst->length, lst->type, list_hd);
    if (!new_lst) return -1;

    for (size_t i = 1; i < lst->length; i++) {
        new_lst = am_list_push(proc->heap_alloc, new_lst, lst->children[i]);
        if (!new_lst) return -1;
    }

    am_handle_t new_hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (new_hd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, new_lst);
        return -1;
    }
    am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, new_hd,
                am_make_value_of_ptr((am_object_t *)new_lst));
    am_value_t result = am_make_value_of_handle(new_hd);
    am_process_push_operand(proc, result);
    am_process_step(proc);
    return 0;
}


static int32_t op_cons(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t list_val = am_process_pop_operand(proc);
    am_value_t first = am_process_pop_operand(proc);
    if (!am_value_is_handle(list_val)) return -1;

    am_handle_t list_hd = am_value_to_handle(list_val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return -1;

    am_list_t *lst = (am_list_t *)obj;
    am_list_t *new_lst = am_list_create(proc->heap_alloc, lst->length + 1, lst->type, list_hd);
    if (!new_lst) return -1;

    new_lst = am_list_push(proc->heap_alloc, new_lst, first);
    if (!new_lst) return -1;
    for (size_t i = 0; i < lst->length; i++) {
        new_lst = am_list_push(proc->heap_alloc, new_lst, lst->children[i]);
        if (!new_lst) return -1;
    }

    am_handle_t new_hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (new_hd == AM_HANDLE_NULL) {
        am_list_destroy(proc->heap_alloc, new_lst);
        return -1;
    }
    am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, new_hd,
                am_make_value_of_ptr((am_object_t *)new_lst));
    am_process_push_operand(proc, am_make_value_of_handle(new_hd));
    am_process_step(proc);
    return 0;
}


static int32_t op_get_item(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t index_val = am_process_pop_operand(proc);
    am_value_t list_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(list_val) || !am_value_is_number(index_val)) return -1;

    am_handle_t list_hd = am_value_to_handle(list_val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return -1;

    am_list_t *lst = (am_list_t *)obj;
    int32_t idx_type = am_value_type(index_val);
    am_int_t idx = 0;
    if (idx_type == AM_VALUE_TYPE_FLOAT) {
        idx = (am_int_t)roundf(am_value_to_float(index_val));
    }
    else if (idx_type == AM_VALUE_TYPE_UINT) {
        idx = (am_int_t)(am_value_to_uint(index_val));
    }
    else if (idx_type == AM_VALUE_TYPE_INT) {
        idx = am_value_to_int(index_val);
    }
    else {
        am_process_push_operand(proc, AM_VALUE_UNDEFINED);
        am_process_step(proc);
        return 0;
    }

    if (idx < 0 || (size_t)idx >= lst->length) {
        am_process_push_operand(proc, AM_VALUE_UNDEFINED);
    }
    else {
        am_process_push_operand(proc, lst->children[idx]);
    }
    am_process_step(proc);
    return 0;
}


static int32_t op_set_item(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t value = am_process_pop_operand(proc);
    am_value_t index_val = am_process_pop_operand(proc);
    am_value_t list_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(list_val) || !am_value_is_number(index_val)) return -1;

    am_handle_t list_hd = am_value_to_handle(list_val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return -1;

    am_list_t *lst = (am_list_t *)obj;
    int32_t idx_type = am_value_type(index_val);
    am_int_t idx = 0;
    if (idx_type == AM_VALUE_TYPE_FLOAT) {
        idx = (am_int_t)roundf(am_value_to_float(index_val));
    }
    else if (idx_type == AM_VALUE_TYPE_UINT) {
        idx = (am_int_t)(am_value_to_uint(index_val));
    }
    else if (idx_type == AM_VALUE_TYPE_INT) {
        idx = am_value_to_int(index_val);
    }
    else {
        return -1;
    }
    if (idx < 0 || (size_t)idx >= lst->length) return -1;

    am_list_set(proc->heap_alloc, lst, (size_t)idx, value);
    am_process_step(proc);
    return 0;
}


static int32_t op_list_push(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t value = am_process_pop_operand(proc);
    am_value_t list_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(list_val)) return -1;

    am_handle_t list_hd = am_value_to_handle(list_val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return -1;

    am_list_t *lst = (am_list_t *)obj;
    am_list_t *new_lst = am_list_push(proc->heap_alloc, lst, value);
    if (!new_lst) return -1;
    if (new_lst != lst) {
        if (am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd,
                        am_make_value_of_ptr((am_object_t *)new_lst)) != 0) {
            am_list_destroy(proc->heap_alloc, new_lst);
            return -1;
        }
    }
    // am_process_push_operand(proc, am_make_value_of_uint((am_uint_t)new_lst->length));
    am_process_step(proc);
    return 0;
}


static int32_t op_list_pop(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t list_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(list_val)) return -1;

    am_handle_t list_hd = am_value_to_handle(list_val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return -1;

    am_list_t *lst = (am_list_t *)obj;
    am_value_t popped = am_list_pop(proc->heap_alloc, lst);
    am_process_push_operand(proc, popped);
    am_process_step(proc);
    return 0;
}


static int32_t op_length(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t list_val = am_process_pop_operand(proc);
    if (!am_value_is_handle(list_val)) return -1;

    am_handle_t list_hd = am_value_to_handle(list_val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, list_hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);
    if (obj->type != AM_OBJECT_TYPE_LIST) return -1;

    am_list_t *lst = (am_list_t *)obj;
    am_process_push_operand(proc, am_make_value_of_uint((am_uint_t)lst->length));
    am_process_step(proc);
    return 0;
}


static int32_t op_concat(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t count_val = am_process_pop_operand(proc);
    if (!am_value_is_number(count_val)) return -1;
    am_int_t count = am_value_is_int(count_val) ? am_value_to_int(count_val) : (am_int_t)am_value_to_uint(count_val);
    if (count < 0) return -1;

    am_value_t *children = (am_value_t *)am_malloc(proc->vm_alloc, (size_t)count * sizeof(am_value_t));
    if (!children && count > 0) return -1;

    for (am_int_t i = count - 1; i >= 0; i--) {
        children[i] = am_process_pop_operand(proc);
    }

    am_list_t *new_lst = am_list_create(proc->heap_alloc, (size_t)count, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!new_lst) {
        am_free(proc->vm_alloc, children);
        return -1;
    }

    for (am_int_t i = 0; i < count; i++) {
        new_lst = am_list_push(proc->heap_alloc, new_lst, children[i]);
        if (!new_lst) {
            am_free(proc->vm_alloc, children);
            return -1;
        }
    }

    am_handle_t new_hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
    if (new_hd == AM_HANDLE_NULL) {
        am_free(proc->vm_alloc, children);
        am_list_destroy(proc->heap_alloc, new_lst);
        return -1;
    }
    am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, new_hd,
                am_make_value_of_ptr((am_object_t *)new_lst));

    // 设置子列表的 parent 字段
    for (size_t i = 0; i < new_lst->length; i++) {
        am_value_t child = new_lst->children[i];
        if (am_value_is_handle(child)) {
            am_handle_t child_hd = am_value_to_handle(child);
            am_value_t child_obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, child_hd);
            if (am_value_is_ptr(child_obj_val)) {
                am_object_t *child_obj = am_value_to_ptr(child_obj_val);
                if (child_obj->type == AM_OBJECT_TYPE_LIST) {
                    ((am_list_t *)child_obj)->parent = new_hd;
                }
            }
        }
    }

    am_free(proc->vm_alloc, children);
    am_process_push_operand(proc, am_make_value_of_handle(new_hd));
    am_process_step(proc);
    return 0;
}


static int32_t op_duplicate(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;

    am_value_t val = am_process_pop_operand(proc);
    if (!am_value_is_handle(val)) return -1;

    am_handle_t hd = am_value_to_handle(val);
    am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
    if (!am_value_is_ptr(obj_val)) return -1;
    am_object_t *obj = am_value_to_ptr(obj_val);

    if (obj->type == AM_OBJECT_TYPE_LIST) {
        am_list_t *lst = (am_list_t *)obj;
        am_list_t *copy = am_list_copy(proc->heap_alloc, lst);
        if (!copy) return -1;
        am_handle_t new_hd = am_heap_alloc_handle(proc->vm_alloc, proc->heap_alloc, proc->heap);
        if (new_hd == AM_HANDLE_NULL) {
            am_list_destroy(proc->heap_alloc, copy);
            return -1;
        }
        am_heap_set(proc->vm_alloc, proc->heap_alloc, proc->heap, new_hd,
                    am_make_value_of_ptr((am_object_t *)copy));
        am_process_push_operand(proc, am_make_value_of_handle(new_hd));
    }
    else if (obj->type == AM_OBJECT_TYPE_WSTRING) {
        am_wstring_t *ws = (am_wstring_t *)obj;
        // 构造临时 wchar_t 缓冲区供驻留查询使用（wchar_t 与 am_wchar_t 同为 32 位 Unicode 码点）
        wchar_t *buf = (wchar_t *)am_malloc(proc->vm_alloc, ws->length * sizeof(wchar_t));
        if (!buf) return -1;
        for (size_t i = 0; i < ws->length; i++) {
            buf[i] = (wchar_t)am_value_to_wchar(ws->content[i]);
        }
        am_handle_t new_hd = am_process_make_wstring_handle(proc, buf, ws->length);
        am_free(proc->vm_alloc, buf);
        if (new_hd == AM_HANDLE_NULL) return -1;
        am_process_push_operand(proc, am_make_value_of_handle(new_hd));
    }
    else {
        // 其他类型暂不做深拷贝，直接返回原 handle
        am_process_push_operand(proc, val);
    }

    am_process_step(proc);
    return 0;
}


// ===============================================================================
// evalcleanup：System.eval 执行结束后的清理指令
// ===============================================================================

typedef struct {
    am_handle_t first_handle;
    am_handle_t last_handle;
} eval_unmark_ctx_t;


static void eval_unmark_static_cb(am_handle_t handle, am_value_t value, void *user_data) {
    eval_unmark_ctx_t *ctx = (eval_unmark_ctx_t *)user_data;
    if (handle < ctx->first_handle || handle > ctx->last_handle) return;
    if (!am_value_is_ptr(value)) return;
    am_object_t *obj = am_value_to_ptr(value);
    if (am_object_check_static(obj) == 0) {
        am_object_set_static(obj, -1);
    }
}


static void eval_shrink_var_type(am_process_t *proc, size_t old_len) {
    if (!proc || !proc->var_type || proc->var_type->length <= old_len) return;

    am_list_t *new_list = am_list_create(proc->vm_alloc,
                                         old_len > 0 ? old_len : 4,
                                         AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    if (!new_list) return;

    for (size_t i = 0; i < old_len; i++) {
        new_list = am_list_push(proc->vm_alloc, new_list,
                                am_list_get(proc->vm_alloc, proc->var_type, i));
        if (!new_list) return;
    }

    am_list_destroy(proc->vm_alloc, proc->var_type);
    proc->var_type = new_list;
}


static void eval_shrink_var_vocab(am_process_t *proc, size_t old_len) {
    if (!proc || !proc->var_vocab) return;

    // 释放 eval 引入的所有尾部 var_vocab 条目，保持 var_vocab / var_type 同步
    while (proc->var_vocab->length > old_len) {
        size_t idx = proc->var_vocab->length - 1;
        wchar_t *word = proc->var_vocab->words[idx];
        if (word) {
            am_free(proc->vm_alloc, word);
            proc->var_vocab->words[idx] = NULL;
        }
        proc->var_vocab->length--;
    }

    if (proc->var_vocab->length == old_len && proc->var_vocab->capacity > old_len * 2 + 8) {
        am_vocab_t *new_vocab = am_vocab_create(proc->vm_alloc,
                                                old_len > 0 ? old_len : 4);
        if (!new_vocab) return;

        new_vocab->length = old_len;
        for (size_t i = 0; i < old_len; i++) {
            new_vocab->words[i] = proc->var_vocab->words[i];
            proc->var_vocab->words[i] = NULL;
        }
        proc->var_vocab->length = 0;
        am_vocab_destroy(proc->vm_alloc, proc->var_vocab);
        proc->var_vocab = new_vocab;
    }
}


// 清理 eval 引入的 native 记录，避免 stale varid 残留在 proc->natives 中
static void eval_cleanup_natives(am_process_t *proc, size_t old_var_vocab_len) {
    if (!proc || !proc->natives) return;

    am_map_t *m = proc->natives;
    for (size_t i = 0; i < m->capacity; i++) {
        am_value_t key = m->slots[i].key;
        if (key == AM_MAP_KEY_EMPTY || key == AM_MAP_KEY_TOMBSTONE) continue;
        if (!am_value_is_varid(key)) continue;
        if ((size_t)am_value_to_varid(key) >= old_var_vocab_len) {
            m->slots[i].key = AM_MAP_KEY_TOMBSTONE;
            m->slots[i].value = AM_VALUE_NULL;
            if (m->length > 0) m->length--;
            m->tombstones++;
        }
    }
}


static void eval_cleanup_var_tables(am_process_t *proc, size_t old_var_vocab_len) {
    if (!proc) return;
    eval_shrink_var_type(proc, old_var_vocab_len);
    eval_shrink_var_vocab(proc, old_var_vocab_len);
    eval_cleanup_natives(proc, old_var_vocab_len);
}


static int32_t op_evalcleanup(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    if (!proc || !proc->heap) return -1;
    if (!am_value_is_handle(operand)) return -1;

    am_handle_t rec_h = am_value_to_handle(operand);
    am_value_t rec_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, rec_h);
    if (!am_value_is_ptr(rec_val)) return -1;
    am_list_t *rec = (am_list_t *)am_value_to_ptr(rec_val);
    if (!rec || rec->type != AM_LIST_TYPE_DEFAULT || rec->length < 6) return -1;

    am_value_t ret_val           = rec->children[0];
    am_value_t saved_len_val     = rec->children[1];
    am_value_t first_h_val       = rec->children[2];
    am_value_t last_h_val        = rec->children[3];
    am_value_t old_ilen_val      = rec->children[4];
    am_value_t old_var_len_val   = rec->children[5];

    if (!am_value_is_iaddr(ret_val) ||
        !am_value_is_uint(saved_len_val) ||
        !am_value_is_handle(first_h_val) ||
        !am_value_is_handle(last_h_val) ||
        !am_value_is_uint(old_ilen_val) ||
        !am_value_is_uint(old_var_len_val)) {
        return -1;
    }

    am_iaddr_t ret_iaddr      = am_value_to_iaddr(ret_val);
    size_t     saved_len      = (size_t)am_value_to_uint(saved_len_val);
    am_handle_t first_handle  = am_value_to_handle(first_h_val);
    am_handle_t last_handle   = am_value_to_handle(last_h_val);
    size_t     old_ilen       = (size_t)am_value_to_uint(old_ilen_val);
    size_t     old_var_len    = (size_t)am_value_to_uint(old_var_len_val);

    // 1. 将操作数栈恢复到 eval 调用前的高度
    size_t cur_len = am_process_length_of_opstack(proc);
    if (cur_len != SIZE_MAX) {
        while (cur_len > saved_len) {
            am_process_pop_operand(proc);
            cur_len--;
        }
    }

    // 2. 截断 eval 追加的 ilcode
    if (proc->ilcode && old_ilen < proc->ilcode_length) {
        am_instruction_t *shrunk = (am_instruction_t *)am_realloc(
            proc->vm_alloc, proc->ilcode, old_ilen * sizeof(am_instruction_t));
        if (shrunk) {
            proc->ilcode = shrunk;
        }
        proc->ilcode_length = old_ilen;
    }

    // 3. 清除 eval 引入的静态对象标记，使它们可以被后续 GC 回收
    if (first_handle <= last_handle) {
        eval_unmark_ctx_t ctx = { first_handle, last_handle };
        am_heap_iter(proc->vm_alloc, proc->heap_alloc, proc->heap,
                     eval_unmark_static_cb, &ctx);
    }

    // 4. 清理 eval 引入的 ILTEMP 临时变量（只在尾部且类型为 ILTEMP 时收缩）
    eval_cleanup_var_tables(proc, old_var_len);

    // 5. 释放清理记录本身
    am_object_set_keepalive((am_object_t *)rec, -1);
    am_heap_free_handle(proc->vm_alloc, proc->heap_alloc, proc->heap, rec_h);

    // 6. 跳回到 eval 调用点之后
    am_process_goto(proc, ret_iaddr);
    return 0;
}


// ===============================================================================
// 第四类：算术逻辑运算和谓词
// ===============================================================================

// 数值类型转换规则
// +  u  i  f
// u  u  i  f
// i  i  i  f
// f  f  f  f
static int32_t op_add(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;

    if (am_value_is_float(a) || am_value_is_float(b)) {
        am_float_t result = am_runtime_number_to_float(b) + am_runtime_number_to_float(a);
        am_process_push_operand(proc, am_make_value_of_float(result));
    }
    else if (am_value_is_int(a) || am_value_is_int(b)) {
        am_int_t result = am_runtime_number_to_int(b) + am_runtime_number_to_int(a);
        am_process_push_operand(proc, am_make_value_of_int(result));
    }
    else {
        am_uint_t result = am_runtime_number_to_uint(b) + am_runtime_number_to_uint(a);
        am_process_push_operand(proc, am_make_value_of_uint(result));
    }

    am_process_step(proc);
    return 0;
}


// 数值类型转换规则
// -  u  i  f
// u  i  i  f
// i  i  i  f
// f  f  f  f
static int32_t op_sub(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;

    if (am_value_is_float(a) || am_value_is_float(b)) {
        am_float_t result = am_runtime_number_to_float(b) - am_runtime_number_to_float(a);
        am_process_push_operand(proc, am_make_value_of_float(result));
    }
    else {
        am_int_t result = am_runtime_number_to_int(b) - am_runtime_number_to_int(a);
        am_process_push_operand(proc, am_make_value_of_int(result));
    }

    am_process_step(proc);
    return 0;
}


// 数值类型转换规则
// *  u  i  f
// u  u  i  f
// i  i  i  f
// f  f  f  f
static int32_t op_mul(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;

    if (am_value_is_float(a) || am_value_is_float(b)) {
        am_float_t result = am_runtime_number_to_float(b) * am_runtime_number_to_float(a);
        am_process_push_operand(proc, am_make_value_of_float(result));
    }
    else if (am_value_is_int(a) || am_value_is_int(b)) {
        am_int_t result = am_runtime_number_to_int(b) * am_runtime_number_to_int(a);
        am_process_push_operand(proc, am_make_value_of_int(result));
    }
    else {
        am_uint_t result = am_runtime_number_to_uint(b) * am_runtime_number_to_uint(a);
        am_process_push_operand(proc, am_make_value_of_uint(result));
    }

    am_process_step(proc);
    return 0;
}


// 数值类型转换规则
// /  u  i  f
// u  f  f  f
// i  f  f  f
// f  f  f  f
static int32_t op_div(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;
    am_float_t fa = am_runtime_number_to_float(a);
    if (fa == 0.0) {
        am_runtime_error(rt, L"[Runtime] 除零错误\n");
        return -1;
    }
    am_float_t result = am_runtime_number_to_float(b) / fa;
    am_process_push_operand(proc, am_make_value_of_float(result));
    am_process_step(proc);
    return 0;
}


static int32_t op_mod(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;
    am_float_t result = fmod(am_runtime_number_to_float(b), am_runtime_number_to_float(a));
    am_process_push_operand(proc, am_make_value_of_float(result));
    am_process_step(proc);
    return 0;
}


static int32_t op_pow(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;
    am_float_t result = pow(am_runtime_number_to_float(b), am_runtime_number_to_float(a));
    am_process_push_operand(proc, am_make_value_of_float(result));
    am_process_step(proc);
    return 0;
}


static int32_t op_eq(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    am_value_t result = (b == a) ? AM_VALUE_TRUE : AM_VALUE_FALSE;
    am_process_push_operand(proc, result);
    am_process_step(proc);
    return 0;
}


static int32_t op_eqv(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    bool equal;
    if (am_value_is_number(a) && am_value_is_number(b)) {
        equal = (am_runtime_number_to_float(a) == am_runtime_number_to_float(b));
    }
    else {
        equal = (a == b);
    }
    am_process_push_operand(proc, equal ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_equal(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    bool equal = runtime_value_equal(proc, b, a);
    am_process_push_operand(proc, equal ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_ge(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;
    bool result = am_runtime_number_to_float(b) >= am_runtime_number_to_float(a);
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_le(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;
    bool result = am_runtime_number_to_float(b) <= am_runtime_number_to_float(a);
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_gt(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;
    bool result = am_runtime_number_to_float(b) > am_runtime_number_to_float(a);
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_lt(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    if (!am_value_is_number(a) || !am_value_is_number(b)) return -1;
    bool result = am_runtime_number_to_float(b) < am_runtime_number_to_float(a);
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_not(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_process_push_operand(proc, (a == AM_VALUE_FALSE) ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_and(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    bool result = (a != AM_VALUE_FALSE) && (b != AM_VALUE_FALSE);
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_or(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t a = am_process_pop_operand(proc);
    am_value_t b = am_process_pop_operand(proc);
    bool result = (a != AM_VALUE_FALSE) || (b != AM_VALUE_FALSE);
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_isnull(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t v = am_process_pop_operand(proc);
    bool result = false;
    if (am_value_is_null(v)) {
        result = true;
    }
    else if (am_value_is_handle(v)) {
        am_handle_t hd = am_value_to_handle(v);
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (am_value_is_ptr(obj_val)) {
            am_object_t *obj = am_value_to_ptr(obj_val);
            if (obj->type == AM_OBJECT_TYPE_LIST && ((am_list_t *)obj)->length == 0) {
                result = true;
            }
        }
    }
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_isundef(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t v = am_process_pop_operand(proc);
    am_process_push_operand(proc, am_value_is_undefined(v) ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_isatom(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t v = am_process_pop_operand(proc);
    am_process_push_operand(proc, !am_value_is_handle(v) ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_islist(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t v = am_process_pop_operand(proc);
    bool result = false;
    if (am_value_is_handle(v)) {
        am_handle_t hd = am_value_to_handle(v);
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (am_value_is_ptr(obj_val)) {
            am_object_t *obj = am_value_to_ptr(obj_val);
            if (obj->type == AM_OBJECT_TYPE_LIST) {
                result = true;
            }
        }
    }
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_isnumber(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t v = am_process_pop_operand(proc);
    am_process_push_operand(proc, am_value_is_number(v) ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_isnan(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t v = am_process_pop_operand(proc);
    bool result = false;
    if (am_value_is_float(v)) {
        result = isnan(am_runtime_number_to_float(v));
    }
    am_process_push_operand(proc, result ? AM_VALUE_TRUE : AM_VALUE_FALSE);
    am_process_step(proc);
    return 0;
}


static int32_t op_typeof(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_value_t v = am_process_pop_operand(proc);
    const wchar_t *type_name = L"unknown";

    if (am_value_is_ptr(v)) {
        am_object_t *obj = am_value_to_ptr(v);
        switch (obj->type) {
            case AM_OBJECT_TYPE_LIST:         type_name = L"list"; break;
            case AM_OBJECT_TYPE_MAP:          type_name = L"map"; break;
            case AM_OBJECT_TYPE_WSTRING:      type_name = L"string"; break;
            case AM_OBJECT_TYPE_PORT:         type_name = L"port"; break;
            case AM_OBJECT_TYPE_CLOSURE:      type_name = L"closure"; break;
            case AM_OBJECT_TYPE_CONTINUATION: type_name = L"continuation"; break;
            case AM_OBJECT_TYPE_FRAME:        type_name = L"frame"; break;
            case AM_OBJECT_TYPE_ILCODE:       type_name = L"ilcode"; break;
            case AM_OBJECT_TYPE_BOX:          type_name = L"box"; break;
            case AM_OBJECT_TYPE_TOKEN:        type_name = L"token"; break;
            case AM_OBJECT_TYPE_SCOPE:        type_name = L"scope"; break;
            case AM_OBJECT_TYPE_VOCAB:        type_name = L"vocab"; break;
            default:                          type_name = L"object"; break;
        }
    }
    else if (am_value_is_handle(v))       type_name = L"handle";
    else if (am_value_is_varid(v))        type_name = L"varid";
    else if (am_value_is_symbol(v))       type_name = L"symbol";
    else if (am_value_is_iaddr(v))        type_name = L"iaddr";
    else if (am_value_is_label(v))        type_name = L"label";
    else if (am_value_is_boolean(v))      type_name = L"boolean";
    else if (am_value_is_null(v))         type_name = L"null";
    else if (am_value_is_undefined(v))    type_name = L"undefined";
    else if (am_value_is_uint(v) ||
             am_value_is_int(v) ||
             am_value_is_float(v))        type_name = L"number";
    else if (am_value_is_wchar(v))        type_name = L"wchar";

    size_t type_name_len = wcslen(type_name);
    am_handle_t hd = am_process_make_wstring_handle(proc, type_name, type_name_len);
    if (hd == AM_HANDLE_NULL) return -1;
    am_process_push_operand(proc, am_make_value_of_handle(hd));
    am_process_step(proc);
    return 0;
}


// ===============================================================================
// 第五类：其他指令
// ===============================================================================

static int32_t op_fork(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)proc;
    (void)operand;
    // NOTE 废弃fork
    am_runtime_error(rt, L"[Runtime] fork 指令已废弃\n");
    return -1;
}


static int32_t op_display_scalar(am_runtime_t *rt, am_process_t *proc, am_value_t content) {
    // 把大缓冲区放在单独的函数帧中，避免在显示列表（尤其是递归字符串化）的路径上占用栈空间
    wchar_t buf[1024];
    value_to_wchar_buf(proc, content, buf, 1024);
    am_runtime_output(rt, buf);
    am_process_step(proc);
    return 0;
}


static int32_t op_display(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)operand;
    am_value_t content = am_process_pop_operand(proc);

    // 列表对象使用专门的字符串化函数
    if (am_value_is_handle(content)) {
        am_handle_t hd = am_value_to_handle(content);
        am_value_t obj_val = am_heap_get(proc->vm_alloc, proc->heap_alloc, proc->heap, hd);
        if (am_value_is_ptr(obj_val)) {
            am_object_t *obj = am_value_to_ptr(obj_val);
            if (obj->type == AM_OBJECT_TYPE_LIST) {
                size_t len = 0;
                wchar_t *str = am_process_list_to_string(proc, hd, &len);
                if (str) {
                    am_runtime_output(rt, str);
                    am_free(proc->vm_alloc, str);
                    am_process_step(proc);
                    return 0;
                }
            }
        }
    }

    return op_display_scalar(rt, proc, content);
}


static int32_t op_newline(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)operand;
    am_runtime_output(rt, L"\n");
    am_process_step(proc);
    return 0;
}


static int32_t op_read(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    // NOTE 废弃该指令
    am_process_step(proc);
    return 0;
}


static int32_t op_write(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    // NOTE 废弃该指令
    am_process_step(proc);
    return 0;
}


static int32_t op_nop(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_process_step(proc);
    return 0;
}


static int32_t op_pause(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_process_set_state(proc, AM_PROCESS_STATE_SUSPENDED);
    return 0;
}


static int32_t op_halt(am_runtime_t *rt, am_process_t *proc, am_value_t operand) {
    (void)rt;
    (void)operand;
    am_process_set_state(proc, AM_PROCESS_STATE_STOPPED);
    return 0;
}


// ===============================================================================
// 异步定时器基础设施（类型定义）
// ===============================================================================

struct am_timer_t {
    size_t        id;          // 定时器编号（全局自增，从1开始）
    am_pid_t      pid;         // 关联进程ID
    am_handle_t   callback;    // 回调闭包把柄
    am_timestamp_t expire_ms;  // 到期时间戳（毫秒）
    bool          repeat;      // 是否周期触发
    am_timestamp_t interval_ms;// 周期触发间隔（毫秒）
    am_timer_t    *next;       // 链表下一个节点
};


// ===============================================================================
// 生命周期
// ===============================================================================

am_runtime_t *am_runtime_create(am_allocator_t *vm_alloc, am_allocator_t *heap_alloc, const wchar_t *base_dir,
                                const am_runtime_vtable_t *vtable) {
    if (!vm_alloc || !heap_alloc) return NULL;
    // 时间戳与睡眠是 VM 必需能力（定时器、队列超时、事件循环休眠均依赖），缺失则创建失败
    if (!vtable || !vtable->now_ms || !vtable->sleep_in_ms) return NULL;

    am_runtime_t *rt = (am_runtime_t *)am_calloc(vm_alloc, sizeof(am_runtime_t));
    if (!rt) return NULL;

    rt->vm_alloc = vm_alloc;
    rt->heap_alloc = heap_alloc;
    rt->vtable = vtable;

    if (base_dir) {
        size_t len = wcslen(base_dir);
        rt->working_dir = (wchar_t *)am_malloc(vm_alloc, (len + 1) * sizeof(wchar_t));
        if (rt->working_dir) {
            memcpy(rt->working_dir, base_dir, (len + 1) * sizeof(wchar_t));
        }
    }

    rt->process_pool_capacity = 16;
    rt->process_pool = (am_process_t **)am_calloc(vm_alloc, rt->process_pool_capacity * sizeof(am_process_t *));
    if (!rt->process_pool) {
        am_free(vm_alloc, rt);
        return NULL;
    }
    rt->process_poll_counter = 0;

    rt->process_queue = am_list_create(vm_alloc, 16, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    rt->input_fifo = am_list_create(vm_alloc, 16, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    rt->output_fifo = am_list_create(vm_alloc, 16, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    rt->error_fifo = am_list_create(vm_alloc, 16, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);
    rt->queue_list = am_list_create(vm_alloc, 8, AM_LIST_TYPE_DEFAULT, AM_HANDLE_NULL);

    if (!rt->process_queue || !rt->input_fifo || !rt->output_fifo || !rt->error_fifo || !rt->queue_list) {
        am_runtime_destroy(rt);
        return NULL;
    }

    rt->queue_next_id = 1;

    rt->tick_counter = 0;
    rt->gc_count = 0;
    rt->gc_periodic_counter = 0;

    rt->timeslice = 8192;

    rt->timer_list = NULL;
    rt->timer_next_id = 1;

    rt->host_context = NULL;

    return rt;
}


int32_t am_runtime_destroy(am_runtime_t *rt) {
    if (!rt) return 0;

    if (rt->process_pool) {
        for (size_t i = 0; i < rt->process_poll_counter; i++) {
            if (rt->process_pool[i]) {
                am_process_destroy(rt->process_pool[i]);
            }
        }
        am_free(rt->vm_alloc, rt->process_pool);
        rt->process_pool = NULL;
    }

    if (rt->process_queue) {
        am_list_destroy(rt->vm_alloc, rt->process_queue);
        rt->process_queue = NULL;
    }

    destroy_fifo(rt, rt->input_fifo);
    rt->input_fifo = NULL;
    destroy_fifo(rt, rt->output_fifo);
    rt->output_fifo = NULL;
    destroy_fifo(rt, rt->error_fifo);
    rt->error_fifo = NULL;

    if (rt->queue_list) {
        for (size_t i = 0; i < rt->queue_list->length; i++) {
            am_value_t qv = am_list_get(rt->vm_alloc, rt->queue_list, i);
            if (am_value_is_ptr(qv)) {
                am_queue_t *q = (am_queue_t *)am_value_to_ptr(qv);
                if (q) am_runtime_queue_destroy(rt, q);
            }
        }
        am_list_destroy(rt->vm_alloc, rt->queue_list);
        rt->queue_list = NULL;
    }

    if (rt->working_dir) {
        am_free(rt->vm_alloc, rt->working_dir);
        rt->working_dir = NULL;
    }

    am_timer_t *timer = rt->timer_list;
    while (timer) {
        am_timer_t *next = timer->next;
        am_free(rt->vm_alloc, timer);
        timer = next;
    }
    rt->timer_list = NULL;

    am_free(rt->vm_alloc, rt);
    return 0;
}


// ===============================================================================
// 异步定时器基础设施（操作实现）
// ===============================================================================

// 获取当前时间戳（毫秒）。经由 vtable 分派到宿主实现。
am_timestamp_t am_runtime_now_ms(am_runtime_t *rt) {
    if (!rt || !rt->vtable || !rt->vtable->now_ms) return 0;
    return rt->vtable->now_ms(rt);
}


// 短时睡眠（毫秒）。经由 vtable 分派到宿主实现。
static void runtime_sleep_ms(am_runtime_t *rt, am_timestamp_t ms) {
    if (!rt || !rt->vtable || !rt->vtable->sleep_in_ms) return;
    rt->vtable->sleep_in_ms(rt, ms);
}


// 以异步方式调用一个闭包：压入栈帧并跳转到闭包入口，返回地址为 return_target。
int32_t am_runtime_call_async(am_runtime_t *rt, am_process_t *proc, am_handle_t callback,
                              am_iaddr_t return_target) {
    (void)rt;
    if (!proc) return -1;

    am_obj_closure_t *closure = am_process_get_closure(proc, callback);
    if (!closure) return -1;

    am_value_t current_closure_val = am_make_value_of_handle(proc->current_closure_handle);
    am_value_t return_target_val = am_make_value_of_iaddr(return_target);
    if (am_process_push_stack_frame(proc, current_closure_val, return_target_val) != 0) {
        return -1;
    }

    am_process_set_current_closure(proc, callback);
    am_process_goto(proc, closure->iaddr);
    return 0;
}


// 注册一个定时器。成功返回大于0的定时器编号，失败返回0。
size_t am_runtime_set_timer(am_runtime_t *rt, am_pid_t pid, am_handle_t callback,
                            am_timestamp_t delay_ms, bool repeat, am_timestamp_t interval_ms) {
    if (!rt) return 0;

    am_timer_t *timer = (am_timer_t *)am_malloc(rt->vm_alloc, sizeof(am_timer_t));
    if (!timer) return 0;

    if (rt->timer_next_id == 0) rt->timer_next_id = 1;
    timer->id = rt->timer_next_id++;
    timer->pid = pid;
    timer->callback = callback;
    timer->expire_ms = am_runtime_now_ms(rt) + delay_ms;
    timer->repeat = repeat;
    timer->interval_ms = interval_ms;
    timer->next = rt->timer_list;
    rt->timer_list = timer;

    return timer->id;
}


// 根据编号取消一个定时器。成功返回 true，未找到返回 false。
bool am_runtime_clear_timer(am_runtime_t *rt, size_t timer_id) {
    if (!rt || timer_id == 0) return false;

    am_timer_t **cur = &rt->timer_list;
    while (*cur) {
        if ((*cur)->id == timer_id) {
            am_timer_t *to_free = *cur;
            *cur = (*cur)->next;
            am_free(rt->vm_alloc, to_free);
            return true;
        }
        cur = &(*cur)->next;
    }
    return false;
}


// 检查是否存在至少一个关联进程未处于 BLOCKED 状态的定时器。
// 若 nearest 非 NULL，同时返回最近的未阻塞定时器到期时间。
static bool runtime_has_nonblocked_timer(am_runtime_t *rt, am_timestamp_t *nearest) {
    if (!rt || !rt->timer_list) return false;
    bool has = false;
    for (am_timer_t *t = rt->timer_list; t; t = t->next) {
        am_process_t *proc = am_runtime_get_process(rt, t->pid);
        if (proc && proc->state == AM_PROCESS_STATE_BLOCKED) continue;
        if (proc && proc->state == AM_PROCESS_STATE_KILLED) continue;
        if (nearest && (!has || t->expire_ms < *nearest)) *nearest = t->expire_ms;
        has = true;
    }
    return has;
}


// 触发所有已到期的定时器。
static void runtime_fire_expired_timers(am_runtime_t *rt) {
    if (!rt || !rt->timer_list) return;

    am_timestamp_t now = am_runtime_now_ms(rt);
    am_timer_t **cur = &rt->timer_list;
    while (*cur) {
        am_timer_t *timer = *cur;
        if (timer->expire_ms > now) {
            cur = &timer->next;
            continue;
        }

        am_process_t *proc = am_runtime_get_process(rt, timer->pid);
        if (proc) {
            // 进程正在执行阻塞式队列操作：用户定时器暂不触发，避免破坏队列操作状态
            if (proc->state == AM_PROCESS_STATE_BLOCKED) {
                cur = &timer->next;
                continue;
            }

            // 防御性检查：进程已被 kill，清理残留定时器
            if (proc->state == AM_PROCESS_STATE_KILLED) {
                am_timer_t *to_free = *cur;
                *cur = (*cur)->next;
                am_free(rt->vm_alloc, to_free);
                continue;
            }

            am_iaddr_t return_target;
            if (proc->state == AM_PROCESS_STATE_STOPPED) {
                // 进程已停止：回调结束后回到 halt 指令（地址1），并重新入队
                return_target = 1;
                am_value_t pid_val = am_make_value_of_uint((am_uint_t)timer->pid);
                am_list_t *new_queue = am_list_push(rt->vm_alloc, rt->process_queue, pid_val);
                if (new_queue) rt->process_queue = new_queue;
            }
            else {
                // 进程仍在运行：回调结束后回到当前 PC
                return_target = proc->PC;
            }

            am_runtime_call_async(rt, proc, timer->callback, return_target);
            proc->state = AM_PROCESS_STATE_RUNNING;
        }

        if (timer->repeat && proc) {
            // 周期定时器：更新下次到期时间并保留
            timer->expire_ms = now + timer->interval_ms;
            cur = &timer->next;
        }
        else {
            // 一次性定时器：移除并释放
            *cur = timer->next;
            am_free(rt->vm_alloc, timer);
        }
    }
}


// ===============================================================================
// 进程 kill 内部辅助函数
// ===============================================================================

// 释放进程内部资源，但保留 am_process_t 壳（pid/state 等）供外部查检。
static int32_t runtime_process_gut(am_process_t *proc) {
    if (!proc) return -1;

    if (proc->ilcode) {
        am_free(proc->vm_alloc, proc->ilcode);
        proc->ilcode = NULL;
    }
    proc->ilcode_length = 0;

    if (proc->opstack) {
        am_free(proc->vm_alloc, proc->opstack);
        proc->opstack = NULL;
        proc->opstack_top = NULL;
    }
    proc->opstack_capacity = 0;

    if (proc->fstack) {
        am_free(proc->vm_alloc, proc->fstack);
        proc->fstack = NULL;
        proc->fstack_top = NULL;
    }
    proc->fstack_capacity = 0;

    if (proc->var_type) {
        am_list_destroy(proc->vm_alloc, proc->var_type);
        proc->var_type = NULL;
    }
    if (proc->natives) {
        am_map_destroy(proc->vm_alloc, proc->natives);
        proc->natives = NULL;
    }
    if (proc->var_top) {
        am_list_destroy(proc->vm_alloc, proc->var_top);
        proc->var_top = NULL;
    }
    if (proc->var_arn_mapping) {
        am_map_destroy(proc->vm_alloc, proc->var_arn_mapping);
        proc->var_arn_mapping = NULL;
    }
    if (proc->strindex) {
        am_strindex_destroy(proc->vm_alloc, proc->strindex);
        proc->strindex = NULL;
    }
    if (proc->var_vocab) {
        am_vocab_destroy(proc->vm_alloc, proc->var_vocab);
        proc->var_vocab = NULL;
    }
    if (proc->symbol_vocab) {
        am_vocab_destroy(proc->vm_alloc, proc->symbol_vocab);
        proc->symbol_vocab = NULL;
    }
    if (proc->heap) {
        am_heap_destroy(proc->vm_alloc, proc->heap_alloc, proc->heap);
        proc->heap = NULL;
    }

    proc->PC = 0;
    proc->current_closure_handle = AM_HANDLE_NULL;
    proc->gc_count = 0;
    proc->pending_kill = false;

    return 0;
}


// 删除运行时定时器链表中所有属于指定 pid 的定时器。
static void runtime_kill_timers_for_pid(am_runtime_t *rt, am_pid_t pid) {
    if (!rt) return;
    am_timer_t **cur = &rt->timer_list;
    while (*cur) {
        if ((*cur)->pid == pid) {
            am_timer_t *to_free = *cur;
            *cur = (*cur)->next;
            am_free(rt->vm_alloc, to_free);
        } else {
            cur = &(*cur)->next;
        }
    }
}


// 删除所有 IPC 队列中属于指定 pid 的等待者节点。
static void runtime_kill_queue_waiters_for_pid(am_runtime_t *rt, am_pid_t pid) {
    if (!rt || !rt->queue_list) return;
    for (size_t i = 0; i < rt->queue_list->length; i++) {
        am_value_t qv = am_list_get(rt->vm_alloc, rt->queue_list, i);
        if (!am_value_is_ptr(qv)) continue;
        am_queue_t *q = (am_queue_t *)am_value_to_ptr(qv);
        if (!q) continue;

        am_queue_waiter_t **cur = &q->send_waiters;
        while (*cur) {
            if ((*cur)->pid == pid) {
                am_queue_waiter_t *w = *cur;
                *cur = w->next;
                am_free(rt->vm_alloc, w);
            } else {
                cur = &(*cur)->next;
            }
        }

        cur = &q->recv_waiters;
        while (*cur) {
            if ((*cur)->pid == pid) {
                am_queue_waiter_t *w = *cur;
                *cur = w->next;
                am_free(rt->vm_alloc, w);
            } else {
                cur = &(*cur)->next;
            }
        }
    }
}


// 从调度队列中移除指定 pid 的所有待调度条目。
static void runtime_process_queue_remove_pid(am_runtime_t *rt, am_pid_t pid) {
    if (!rt || !rt->process_queue) return;
    size_t write = 0;
    for (size_t i = 0; i < rt->process_queue->length; i++) {
        am_value_t v = am_list_get(rt->vm_alloc, rt->process_queue, i);
        if (am_value_is_uint(v) && (am_pid_t)am_value_to_uint(v) == pid) {
            continue;
        }
        if (write != i) {
            am_list_set(rt->vm_alloc, rt->process_queue, write, v);
        }
        write++;
    }
    rt->process_queue->length = write;
}


// ===============================================================================
// 模块与进程管理
// ===============================================================================

am_pid_t am_runtime_load_module(am_runtime_t *rt, am_module_t *mod) {
    if (!rt || !mod) return (am_pid_t)-1;

    am_process_t *proc = am_process_load_from_module(rt->vm_alloc, rt->heap_alloc, mod);
    if (!proc) return (am_pid_t)-1;

    am_pid_t pid = rt->process_poll_counter;
    proc->pid = pid;
    proc->parent_pid = 0;

    if (pid >= rt->process_pool_capacity) {
        size_t new_cap = rt->process_pool_capacity * 2;
        am_process_t **new_pool = (am_process_t **)am_realloc(
            rt->vm_alloc, rt->process_pool, new_cap * sizeof(am_process_t *));
        if (!new_pool) {
            am_process_destroy(proc);
            return (am_pid_t)-1;
        }
        rt->process_pool = new_pool;
        rt->process_pool_capacity = new_cap;
    }

    rt->process_pool[pid] = proc;
    rt->process_poll_counter++;

    am_value_t pid_val = am_make_value_of_uint((am_uint_t)pid);
    am_list_t *new_queue = am_list_push(rt->vm_alloc, rt->process_queue, pid_val);
    if (!new_queue) {
        rt->process_pool[pid] = NULL;
        rt->process_poll_counter--;
        am_process_destroy(proc);
        return (am_pid_t)-1;
    }
    rt->process_queue = new_queue;

    return pid;
}


am_process_t *am_runtime_get_process(am_runtime_t *rt, am_pid_t pid) {
    if (!rt || pid >= rt->process_poll_counter) return NULL;
    return rt->process_pool[pid];
}


int32_t am_runtime_kill_process(am_runtime_t *rt, am_pid_t pid) {
    if (!rt || pid >= rt->process_poll_counter) return -1;

    am_process_t *proc = rt->process_pool[pid];
    if (!proc || proc->state == AM_PROCESS_STATE_KILLED) return -1;

    int32_t old_state = proc->state;
    am_process_set_state(proc, AM_PROCESS_STATE_KILLED);

    // 立即清理异步任务与调度队列，避免被后续事件触发
    runtime_kill_timers_for_pid(rt, pid);
    runtime_kill_queue_waiters_for_pid(rt, pid);
    runtime_process_queue_remove_pid(rt, pid);

    if (old_state == AM_PROCESS_STATE_RUNNING) {
        // 目标进程正在执行本 native：延迟到调度器安全点再销毁资源
        proc->pending_kill = true;
    } else {
        runtime_process_gut(proc);
    }

    return 0;
}


void am_runtime_set_default_timeslice(am_runtime_t *rt, uint32_t ticks) {
    if (!rt) return;
    rt->timeslice = ticks;
}


am_process_t *am_rumtime_get_process_by_pid(am_runtime_t *rt, am_pid_t pid) {
    return am_runtime_get_process(rt, pid);
}


int32_t am_set_runtime_host_context(am_runtime_t *rt, void *ctx) {
    if (!rt) return -1;
    rt->host_context = ctx;
    return 0;
}


void *am_get_runtime_host_context(am_runtime_t *rt) {
    if (!rt) return NULL;
    return rt->host_context;
}


int32_t am_set_process_host_context(am_runtime_t *rt, am_process_t *proc, void *ctx) {
    (void)rt;
    if (!proc) return -1;
    proc->host_context = ctx;
    return 0;
}


void *am_get_process_host_context(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!proc) return NULL;
    return proc->host_context;
}


// ===============================================================================
// 调度器
// ===============================================================================


int32_t am_runtime_op_dispatch(am_runtime_t *rt, am_process_t *proc, uint32_t opcode, am_value_t operand) {
    switch (opcode) {
        case AM_VM_OP_nop:         return op_nop(rt, proc, operand);
        case AM_VM_OP_store:       return op_store(rt, proc, operand);
        case AM_VM_OP_load:        return op_load(rt, proc, operand);
        case AM_VM_OP_loadclosure: return op_loadclosure(rt, proc, operand);
        case AM_VM_OP_push:        return op_push(rt, proc, operand);
        case AM_VM_OP_pop:         return op_pop(rt, proc, operand);
        case AM_VM_OP_swap:        return op_swap(rt, proc, operand);
        case AM_VM_OP_set:         return op_set(rt, proc, operand);
        case AM_VM_OP_call:        return op_call(rt, proc, operand);
        case AM_VM_OP_callnative:  return op_callnative(rt, proc, operand);
        case AM_VM_OP_tailcall:    return op_tailcall(rt, proc, operand);
        case AM_VM_OP_return:      return op_return(rt, proc, operand);
        case AM_VM_OP_capturecc:   return op_capturecc(rt, proc, operand);
        case AM_VM_OP_iftrue:      return op_iftrue(rt, proc, operand);
        case AM_VM_OP_iffalse:     return op_iffalse(rt, proc, operand);
        case AM_VM_OP_goto:        return op_goto(rt, proc, operand);
        case AM_VM_OP_read:        return op_read(rt, proc, operand);
        case AM_VM_OP_write:       return op_write(rt, proc, operand);
        case AM_VM_OP_pause:       return op_pause(rt, proc, operand);
        case AM_VM_OP_halt:        return op_halt(rt, proc, operand);
        case AM_VM_OP_fork:        return op_fork(rt, proc, operand);
        case AM_VM_OP_display:     return op_display(rt, proc, operand);
        case AM_VM_OP_newline:     return op_newline(rt, proc, operand);
        case AM_VM_OP_add:         return op_add(rt, proc, operand);
        case AM_VM_OP_sub:         return op_sub(rt, proc, operand);
        case AM_VM_OP_mul:         return op_mul(rt, proc, operand);
        case AM_VM_OP_div:         return op_div(rt, proc, operand);
        case AM_VM_OP_mod:         return op_mod(rt, proc, operand);
        case AM_VM_OP_pow:         return op_pow(rt, proc, operand);
        case AM_VM_OP_eq:          return op_eq(rt, proc, operand);
        case AM_VM_OP_eqv:         return op_eqv(rt, proc, operand);
        case AM_VM_OP_equal:       return op_equal(rt, proc, operand);
        case AM_VM_OP_ge:          return op_ge(rt, proc, operand);
        case AM_VM_OP_le:          return op_le(rt, proc, operand);
        case AM_VM_OP_gt:          return op_gt(rt, proc, operand);
        case AM_VM_OP_lt:          return op_lt(rt, proc, operand);
        case AM_VM_OP_not:         return op_not(rt, proc, operand);
        case AM_VM_OP_and:         return op_and(rt, proc, operand);
        case AM_VM_OP_or:          return op_or(rt, proc, operand);
        case AM_VM_OP_isnull:      return op_isnull(rt, proc, operand);
        case AM_VM_OP_isundef:     return op_isundef(rt, proc, operand);
        case AM_VM_OP_isatom:      return op_isatom(rt, proc, operand);
        case AM_VM_OP_islist:      return op_islist(rt, proc, operand);
        case AM_VM_OP_isnumber:    return op_isnumber(rt, proc, operand);
        case AM_VM_OP_isnan:       return op_isnan(rt, proc, operand);
        case AM_VM_OP_typeof:      return op_typeof(rt, proc, operand);
        case AM_VM_OP_car:         return op_car(rt, proc, operand);
        case AM_VM_OP_cdr:         return op_cdr(rt, proc, operand);
        case AM_VM_OP_cons:        return op_cons(rt, proc, operand);
        case AM_VM_OP_get_item:    return op_get_item(rt, proc, operand);
        case AM_VM_OP_set_item:    return op_set_item(rt, proc, operand);
        case AM_VM_OP_list_push:   return op_list_push(rt, proc, operand);
        case AM_VM_OP_list_pop:    return op_list_pop(rt, proc, operand);
        case AM_VM_OP_length:      return op_length(rt, proc, operand);
        case AM_VM_OP_concat:      return op_concat(rt, proc, operand);
        case AM_VM_OP_duplicate:   return op_duplicate(rt, proc, operand);
        case AM_VM_OP_evalcleanup: return op_evalcleanup(rt, proc, operand);
        case AM_VM_OP_dynamicwind:              return op_dynamicwind(rt, proc, operand);
        case AM_VM_OP_dynamicwind_after_before: return op_dynamicwind_after_before(rt, proc, operand);
        case AM_VM_OP_dynamicwind_before_after: return op_dynamicwind_before_after(rt, proc, operand);
        case AM_VM_OP_dynamicwind_done:         return op_dynamicwind_done(rt, proc, operand);
        case AM_VM_OP_wind:                     return op_wind(rt, proc, operand);
        default: {
            wchar_t errmsg[256];
            swprintf(errmsg, 256, L"[Runtime] 未知指令: %u\n", opcode);
            am_runtime_error(rt, errmsg);
            return -1;
        }
    }
}


int32_t am_runtime_execute(am_runtime_t *rt, am_process_t *proc) {
    if (!rt || !proc) return -1;

    uint32_t opcode;
    am_value_t operand;
    if (am_process_current_instruction(proc, &opcode, &operand) != 0) {
        return -1;
    }

    // printf("Exec: PC=%zu | OpCode=%u | Oprand=%zu(varid=%zu)\n", proc->PC, opcode, operand, am_value_to_varid(operand));

    return am_runtime_op_dispatch(rt, proc, opcode, operand);
}


// 按堆水位在安全点触发 GC（L1 主策略）：
//   水位级别 1（高水位）→ 一轮标记-清除；级别 2（临界水位）→ 当轮强制压缩；
//   若堆区曾彻底分配失败（oom_flag，L0 扩界重试仍失败），也强制做一轮 GC 以挽救其余进程。
// 指令之间的tick内部、tick 末尾都是 GC 安全点（各进程的 GC 根处于一致状态）。
static void runtime_gc_watermark_check(am_runtime_t *rt) {
    if (!rt) return;
#if AM_ENABLE_GC
    int32_t level = am_gc_heap_watermark_level(rt->heap_alloc);
    if (am_allocator_heap_take_oom_flag(rt->heap_alloc) == 1 && level < 2) level = 2;
    if (level >= 1) {
        rt->gc_count++;
        (void)am_gc_collect(rt->heap_alloc, rt->process_pool, rt->process_poll_counter,
                            rt->gc_count, (level >= 2) ? 1 : 0);
    }
#else
    (void)rt;
#endif
}

int32_t am_runtime_tick(am_runtime_t *rt, uint32_t timeslice) {
    if (!rt || !rt->process_queue) return AM_VM_STATE_IDLE;
    if (rt->process_queue->length == 0) return AM_VM_STATE_IDLE;

    am_value_t pid_val = am_list_shift(rt->vm_alloc, rt->process_queue);
    if (!am_value_is_uint(pid_val)) return AM_VM_STATE_IDLE;

    am_pid_t pid = (am_pid_t)am_value_to_uint(pid_val);
    if (pid >= rt->process_poll_counter || !rt->process_pool[pid]) {
        return AM_VM_STATE_IDLE;
    }

    am_process_t *proc = rt->process_pool[pid];
    proc->state = AM_PROCESS_STATE_RUNNING;

    uint32_t since_check = 0;
    while (timeslice > 0 && proc->state == AM_PROCESS_STATE_RUNNING) {
        if (am_runtime_execute(rt, proc) != 0) {
            // 补救：若失败由堆分配失败（OOM）引起，立即做一轮 GC 以挽救其余进程
            runtime_gc_watermark_check(rt);
            proc->state = AM_PROCESS_STATE_STOPPED;
            if (rt->vtable->on_error) rt->vtable->on_error(rt);
            wchar_t errmsg[256];
            swprintf(errmsg, 256, L"[Runtime] 指令执行异常: PID=%zu PC=%zu\n", (size_t)pid, (size_t)proc->PC);
            am_runtime_error(rt, errmsg);
            break;
        }
        timeslice--;
        // 每 AM_GC_WATERMARK_CHECK_STRIDE 条指令检查一次堆水位，收窄失控分配的逃逸窗口
        if (++since_check >= AM_GC_WATERMARK_CHECK_STRIDE) {
            since_check = 0;
            runtime_gc_watermark_check(rt);
        }
    }

    // tick 末尾安全点：再检查一次堆水位
    runtime_gc_watermark_check(rt);

    if (proc->state == AM_PROCESS_STATE_RUNNING) {
        proc->state = AM_PROCESS_STATE_READY;
        am_list_t *new_queue = am_list_push(rt->vm_alloc, rt->process_queue, pid_val);
        if (!new_queue) {
            // 入队失败，停止进程以避免丢失
            proc->state = AM_PROCESS_STATE_STOPPED;
            return AM_VM_STATE_IDLE;
        }
        rt->process_queue = new_queue;
    }

    // 在 tick 结束的安全点完成延迟 kill
    if (proc->state == AM_PROCESS_STATE_KILLED && proc->pending_kill) {
        runtime_process_gut(proc);
    }

    rt->tick_counter++;
    if (rt->vtable->on_tick) rt->vtable->on_tick(rt);

    return (rt->process_queue->length > 0) ? AM_VM_STATE_RUNNING : AM_VM_STATE_IDLE;
}


/* 获取运行时内存统计快照。
 * 通过 allocator 提供的抽象查询接口获取数据，与 allocator 内部实现策略无关。 */
int32_t am_runtime_get_memory_stats(am_runtime_t *rt, am_runtime_memory_stats_t *out) {
    (void)rt;
    if (!out) return -1;

    am_allocator_pool_t *pool = am_allocator_pool_current();
    if (!pool) return -1;

    size_t total_size = am_allocator_pool_total_size(pool);
    size_t heap_cap   = am_allocator_pool_heap_capacity(pool);

    out->vm_capacity   = (total_size > heap_cap) ? (total_size - heap_cap) : 0;
    out->vm_used       = am_allocator_pool_vm_used(pool);
    out->heap_capacity = heap_cap;
    out->heap_used     = am_allocator_pool_heap_used(pool);
    return 0;
}


/* 打印运行时内存总体使用状况（VM 工作区 + 用户堆区）。
 * 通过 allocator 提供的抽象查询接口获取数据，与 allocator 内部实现策略无关。 */
void am_runtime_print_memory_stats(am_runtime_t *rt) {
    am_runtime_memory_stats_t stats;
    if (am_runtime_get_memory_stats(rt, &stats) != 0) {
        fprintf(stderr, "[MemoryStats] 当前内存池信息不可用\n");
        return;
    }

    size_t total_size = stats.vm_capacity + stats.heap_capacity;

    fprintf(stderr, "\n========== 运行时内存使用状况 ==========\n");
    fprintf(stderr, "  内存池总容量: %zu bytes (%.2f MB)\n",
            total_size, (double)total_size / (1024.0 * 1024.0));
    fprintf(stderr, "\n");
    fprintf(stderr, "  VM 工作区:\n");
    fprintf(stderr, "    容量=%zu bytes\n", stats.vm_capacity);
    fprintf(stderr, "    已用=%zu bytes\n", stats.vm_used);
    fprintf(stderr, "    空闲=%zu bytes\n", stats.vm_capacity - stats.vm_used);
    if (stats.vm_capacity > 0) {
        fprintf(stderr, "    使用率=%.2f%%\n",
                100.0 * (double)stats.vm_used / (double)stats.vm_capacity);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "  用户堆区:\n");
    fprintf(stderr, "    容量=%zu bytes\n", stats.heap_capacity);
    fprintf(stderr, "    已用=%zu bytes\n", stats.heap_used);
    fprintf(stderr, "    空闲=%zu bytes\n", stats.heap_capacity - stats.heap_used);
    if (stats.heap_capacity > 0) {
        fprintf(stderr, "    使用率=%.2f%%\n",
                100.0 * (double)stats.heap_used / (double)stats.heap_capacity);
    }
    fprintf(stderr, "========================================\n\n");
}


int32_t am_runtime_event_handler(am_runtime_t *rt) {
    if (!rt) return AM_VM_STATE_IDLE;

    // NOTE GC 触发策略为“堆水位为主、周期兜底为辅”（三级触发）：
    //   L0 分配失败时分配器内部向 VM 区让渡边界并重试（见 freelist_malloc）；
    //   L1 tick 内每 AM_GC_WATERMARK_CHECK_STRIDE 条指令及 tick 末尾按堆水位触发（见 am_runtime_tick），
    //     逃逸窗口为 STRIDE 条指令，不再受时间片长度影响；
    //   L2 每 AM_GC_PERIODIC_INTERVAL 轮事件循环执行一轮兜底 GC（见下）。
    int32_t vm_state = AM_VM_STATE_IDLE;
    for (int i = 0; i < AM_COMPUTATION_PHASE_LENGTH; i++) {
        vm_state = am_runtime_tick(rt, rt->timeslice);
        if (vm_state == AM_VM_STATE_IDLE) break;
    }

#if AM_ENABLE_GC && AM_GC_PERIODIC_INTERVAL > 0
    // 周期兜底：每 AM_GC_PERIODIC_INTERVAL 轮事件循环执行一轮 GC，
    // 保证分配缓慢但持续产生垃圾的程序最终也能回收，压缩与边界调整仍能周期发生。
    rt->gc_periodic_counter++;
    if ((rt->gc_periodic_counter % AM_GC_PERIODIC_INTERVAL) == 0) {
        rt->gc_count++;
        (void)am_gc_collect(rt->heap_alloc, rt->process_pool, rt->process_poll_counter, rt->gc_count, 0);
    }
#endif

    // 检查队列阻塞等待者：唤醒超时的发送者/接收者
    runtime_queue_check_waiters(rt);

    runtime_fire_expired_timers(rt);

    // 若触发定时器或队列唤醒后有进程入队，继续保持 RUNNING 状态
    if (rt->process_queue && rt->process_queue->length > 0) {
        vm_state = AM_VM_STATE_RUNNING;
    }
    // 即使暂无就绪进程，只要还有未到期定时器（且其关联进程未阻塞）
    // 或阻塞中的队列等待者，事件循环也应继续运转，等待它们到期或被唤醒。
    if (vm_state == AM_VM_STATE_IDLE &&
        (runtime_has_nonblocked_timer(rt, NULL) || runtime_queue_has_waiters(rt, NULL))) {
        vm_state = AM_VM_STATE_RUNNING;
    }

    if (rt->vtable->on_event) rt->vtable->on_event(rt);
    return vm_state;
}


void am_runtime_start(am_runtime_t *rt) {
    if (!rt) return;

    while (1) {
        int32_t vm_state = am_runtime_event_handler(rt);
        if (vm_state == AM_VM_STATE_IDLE) {
            if (rt->vtable->on_halt) rt->vtable->on_halt(rt);
            break;
        }

        // 若当前无就绪进程但仍有未到期定时器（关联进程未阻塞）
        // 或队列阻塞等待者，则睡眠到最近的到期时间。
        if (rt->process_queue && rt->process_queue->length == 0 &&
            (runtime_has_nonblocked_timer(rt, NULL) || runtime_queue_has_waiters(rt, NULL))) {
            am_timestamp_t now = am_runtime_now_ms(rt);
            am_timestamp_t next = 0;
            am_timestamp_t tnext;
            if (runtime_has_nonblocked_timer(rt, &tnext)) {
                next = tnext;
            }
            am_timestamp_t qnext;
            if (runtime_queue_has_waiters(rt, &qnext)) {
                if (next == 0 || qnext < next) next = qnext;
            }
            if (next > now) {
                runtime_sleep_ms(rt, next - now);
            }
        }
    }
}


void am_start(am_runtime_t *rt) {
    am_runtime_start(rt);
}


// ===============================================================================
// 控制台输入输出
// ===============================================================================

// 将宽字符串中的 "\\n"、"\\r"、"\\t"、"\\b"、"\\\\"、"\\\"" 等字符序列
// 替换为对应的 ASCII 控制字符。返回新分配的宽字符串，调用者负责释放。
static wchar_t *runtime_unescape_output_string(am_allocator_t *alloc, const wchar_t *str, size_t *out_len) {
    if (!alloc || !str) return NULL;

    size_t len = wcslen(str);
    wchar_t *result = (wchar_t *)am_malloc(alloc, (len + 1) * sizeof(wchar_t));
    if (!result) return NULL;

    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (str[i] == L'\\' && i + 1 < len) {
            bool replaced = true;
            wchar_t replacement = L'\\';
            switch (str[i + 1]) {
                case L'n': replacement = L'\n'; break;
                case L'r': replacement = L'\r'; break;
                case L't': replacement = L'\t'; break;
                case L'b': replacement = L'\b'; break;
                case L'\\': replacement = L'\\'; break;
                case L'"': replacement = L'"'; break;
                default: replaced = false; break;
            }
            if (replaced) {
                result[j++] = replacement;
                i++;
                continue;
            }
        }
        result[j++] = str[i];
    }
    result[j] = L'\0';
    if (out_len) *out_len = j;
    return result;
}


void am_runtime_output(am_runtime_t *rt, const wchar_t *str) {
    if (!rt || !str) return;

    size_t output_len = wcslen(str);
    wchar_t *unescaped = runtime_unescape_output_string(rt->vm_alloc, str, &output_len);
    const wchar_t *output_str = unescaped ? unescaped : str;

    if (rt->output_fifo) {
        for (size_t i = 0; i < output_len; i++) {
            am_value_t ch = am_make_value_of_wchar((am_wchar_t)output_str[i]);
            am_list_t *new_fifo = am_list_push(rt->vm_alloc, rt->output_fifo, ch);
            if (new_fifo) rt->output_fifo = new_fifo;
        }
    }

    if (unescaped) am_free(rt->vm_alloc, unescaped);
}


void am_runtime_error(am_runtime_t *rt, const wchar_t *str) {
    if (!rt || !str) return;

    if (rt->error_fifo) {
        size_t len = wcslen(str);
        for (size_t i = 0; i < len; i++) {
            am_value_t ch = am_make_value_of_wchar((am_wchar_t)str[i]);
            am_list_t *new_fifo = am_list_push(rt->vm_alloc, rt->error_fifo, ch);
            if (new_fifo) rt->error_fifo = new_fifo;
        }
    }

    if (rt->vtable->on_error) rt->vtable->on_error(rt);
}





// ===============================================================================
// 本地宿主函数机制（Native）
// ===============================================================================

static const am_native_lib_entry_t *g_native_libs[AM_NATIVE_MAX_LIBS];
static size_t g_native_lib_count = 0;


// 注册一个native库到全局native库表中。成功返回0，失败返回-1。
static int32_t am_native_register_lib(const am_native_lib_entry_t *lib) {
    if (!lib || !lib->name || !lib->funcs || lib->func_count == 0) return -1;
    if (g_native_lib_count >= AM_NATIVE_MAX_LIBS) return -1;
    g_native_libs[g_native_lib_count++] = lib;
    return 0;
}

// 向运行时注册一个native库。成功返回0，失败返回-1。
int32_t am_runtime_register_native_lib(am_runtime_t *rt, const am_native_lib_entry_t *lib) {
    if (!rt) return -1;
    return am_native_register_lib(lib);
}

// 运行时查表：根据库名和函数名定位native函数实现。
am_native_func_t am_native_find_func(const wchar_t *lib_name, const wchar_t *func_name) {
    if (!lib_name || !func_name) return NULL;

    for (size_t i = 0; i < g_native_lib_count; i++) {
        const am_native_lib_entry_t *lib = g_native_libs[i];
        if (!lib) continue;
        if (wcscmp(lib->name, lib_name) != 0) continue;

        for (size_t j = 0; j < lib->func_count; j++) {
            if (wcscmp(lib->funcs[j].name, func_name) == 0) {
                return lib->funcs[j].func;
            }
        }
    }
    return NULL;
}
/* ===== end:   src/am_runtime.c ===== */
