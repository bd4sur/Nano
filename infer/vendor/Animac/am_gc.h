#ifndef __AM_GC_H__
#define __AM_GC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "am_allocator.h"
#include "am_heap.h"
#include "am_process.h"


///////////////////////////////////////////
// 垃圾回收（GC）
// 说明：GC 是解释器的核心功能，统一由本模块实现：
//   - 分进程标记-清除（am_gc_process）：GC 根收集、递归标记、清除；
//   - 全局标记-压缩（am_gc_compact）：收集各进程逻辑堆中的存活对象，
//     调用 allocator 的纯物理压缩引擎搬移对象，再回写所有 heap 表指针；
//   - 编排（am_gc_collect）：对进程池执行一轮完整 GC。
// 层级：位于 process/heap 之上、runtime 之下，不依赖 runtime.h。
///////////////////////////////////////////

// GC 配置

#ifndef AM_ENABLE_GC
#define AM_ENABLE_GC (1)
#endif

// 每经历 AM_HEAP_COMPACT_INTERVAL 次 GC 后触发一次标记-压缩。
// 设为 0 表示不在 GC 时自动触发压缩（可手动调用 am_gc_compact）。
#ifndef AM_HEAP_COMPACT_INTERVAL
#define AM_HEAP_COMPACT_INTERVAL (1)
#endif

// GC 触发策略：堆水位 + 慢速周期兜底（见 am_runtime.c 的调用点）。
// 堆区已用比例达到 AM_GC_HEAP_HIGH_WATER_RATIO 时触发一轮标记-清除；
// 达到 AM_GC_HEAP_CRITICAL_RATIO 时当轮强制标记-压缩（无视 AM_HEAP_COMPACT_INTERVAL）。
#ifndef AM_GC_HEAP_HIGH_WATER_RATIO
#define AM_GC_HEAP_HIGH_WATER_RATIO (0.75)
#endif
#ifndef AM_GC_HEAP_CRITICAL_RATIO
#define AM_GC_HEAP_CRITICAL_RATIO (0.90)
#endif
// 碎片维度：堆用量达到 AM_GC_HEAP_FRAG_FLOOR_RATIO 且最大空闲块小于容量的
// AM_GC_HEAP_FRAG_MIN_BLOCK_RATIO 时，视为临界水位（first-fit 随时可能失败，需压缩整理）。
#ifndef AM_GC_HEAP_FRAG_FLOOR_RATIO
#define AM_GC_HEAP_FRAG_FLOOR_RATIO (0.30)
#endif
#ifndef AM_GC_HEAP_FRAG_MIN_BLOCK_RATIO
#define AM_GC_HEAP_FRAG_MIN_BLOCK_RATIO (0.03125)
#endif
// 事件循环每 AM_GC_PERIODIC_INTERVAL 轮执行一轮兜底 GC（保证分配缓慢但持续
// 产生垃圾的程序最终也能回收）。设为 0 表示禁用周期兜底（纯水位触发）。
#ifndef AM_GC_PERIODIC_INTERVAL
#define AM_GC_PERIODIC_INTERVAL (32)
#endif
// 进程执行的每个 tick 内，每 AM_GC_WATERMARK_CHECK_STRIDE 条指令检查一次堆水位，
// 将失控分配的逃逸窗口从整个时间片收窄到 STRIDE 条指令。
#ifndef AM_GC_WATERMARK_CHECK_STRIDE
#define AM_GC_WATERMARK_CHECK_STRIDE (256)
#endif


// 对单个进程执行全量的标记-清除 GC。成功返回 0，失败返回 -1。
int32_t am_gc_process(am_process_t *proc);

// 对多个进程堆一起执行全局标记-压缩：扫描所有 heap 表收集存活对象，
// 调用 am_allocator_heap_compact 引擎搬移对象，并回写所有 heap 表中的指针。
// 用于多进程共享同一个 heap_alloc 的场景。必须在 GC 安全点调用
//（所有相关进程已完成标记-清除）。成功返回 0，失败返回 -1。
int32_t am_gc_compact(am_allocator_t *heap_alloc, am_heap_t **heaps, size_t heap_count);

// 对进程池执行一轮完整 GC：逐进程标记-清除，随后按 gc_seq 与
// AM_HEAP_COMPACT_INTERVAL 决定是否执行全局标记-压缩与内存池边界自动调整。
// process_pool 为进程指针数组（允许含 NULL 槽位），process_count 为数组长度；
// gc_seq 为本轮 GC 的序号（通常由调用方维护的计数器提供）；
// force_compact 非 0 时无视 AM_HEAP_COMPACT_INTERVAL 当轮强制压缩。
// 仅 GC 成功的进程堆才会纳入压缩。成功返回 0，失败返回 -1。
int32_t am_gc_collect(am_allocator_t *heap_alloc, am_process_t **process_pool,
                      size_t process_count, size_t gc_seq, int32_t force_compact);

// 查询堆区水位级别（供运行期按水位触发 GC）：
//   0 = 正常（已用 < AM_GC_HEAP_HIGH_WATER_RATIO）；
//   1 = 高水位（应执行一轮标记-清除）；
//   2 = 临界水位（应执行一轮标记-清除并强制压缩）；
//  负值 = 查询失败（alloc 非池的堆区分配器等）。
int32_t am_gc_heap_watermark_level(am_allocator_t *heap_alloc);


#ifdef __cplusplus
}
#endif

#endif
