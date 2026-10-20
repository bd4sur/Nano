#ifndef __AM_HEAP_H__
#define __AM_HEAP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "am_object.h"
#include "am_map.h"
#include "am_allocator.h"

///////////////////////////////////////////
// 解释器基础设施：抽象堆
//   抽象堆的实质是 am_handle_t 到 am_value_t 的映射表，以及管理用元数据。
//   抽象堆的功能是 管理把柄（逻辑地址），将逻辑地址与物理地址解耦，使得无论物理地址怎么变化，把柄永远指向同一个对象。
//   抽象堆的归属是 每个进程拥有属于自己的堆实例，以此实现逻辑地址的进程隔离。
//   抽象堆的基础是 RT提供的内存分配器，堆上存储的value的指针指向的都是RT（宿主环境）提供的物理内存。
//
//   从内存管理角度看，am_heap_t 本身（结构体、table、metadata）属于“容器/元数据”，
//   由 container_alloc 管理；table 中存储的指针所指向的用户数据对象，由 obj_alloc 管理。
//   在编译阶段，container_alloc 和 obj_alloc 通常是同一个分配器；
//   在运行时阶段，container_alloc 对应 vm_alloc，obj_alloc 对应 heap_alloc。
///////////////////////////////////////////

// 堆数据结构
typedef struct am_heap_t {
    size_t   capacity; // 当前 table 的物理槽位数（随 table 扩容同步更新）
    am_map_t *table;
    am_map_t *metadata;
    am_handle_t handle_counter; // 简单的自增计数器
} am_heap_t;

// 遍历回调类型
typedef void (*am_heap_iter_callback_t)(am_handle_t handle, am_value_t value, void *user_data);


// 创建堆。container_alloc 管理堆结构本身及其 table/metadata；obj_alloc 管理堆中对象。
am_heap_t *am_heap_create(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, size_t capacity);

int32_t am_heap_destroy(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap);

am_heap_t *am_heap_copy(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap);

void am_heap_iter(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_heap_iter_callback_t cb, void *user_data);

// 功能说明：将am_heap_t对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩底层map对象，将table和metadata的capacity压缩到跟length一致，删除多余分配的空闲部分。
size_t am_heap_dump(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, uint8_t *buffer, size_t offset);

// 功能说明：深度转储整个heap及其指向的对象
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       仅处理value为ptr且指向AM_OBJECT_TYPE_LIST或AM_OBJECT_TYPE_WSTRING类型对象的情况。
size_t am_heap_deep_dump(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, uint8_t *buffer, size_t offset);

// 功能说明：am_heap_dump的逆操作。从二进制字节序列buffer[offset]开始，读取转储的heap对象，构造heap并返回其指针。
// 实现说明：成功则返回加载后am_heap_t对象的指针，失败则返回NULL。
am_heap_t *am_heap_load(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, uint8_t *buffer, size_t offset);

// 功能说明：am_heap_deep_dump的逆操作。从二进制字节序列buffer[offset]开始，读取转储的heap及其指向的对象，构造heap并返回其指针。
// 实现说明：成功则返回加载后am_heap_t对象的指针，失败则返回NULL。
// 注意：仅处理value为ptr且指向AM_OBJECT_TYPE_LIST或AM_OBJECT_TYPE_WSTRING类型对象的情况。
am_heap_t *am_heap_deep_load(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, uint8_t *buffer, size_t offset);



// 存在性检查：存在返回 0，不存在或 heap/table 为空返回 -1。
int32_t am_heap_has_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle);

am_handle_t am_heap_alloc_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap);

// 不仅删除entry，还要穿透free对应的堆对象（被GC调用）。
// 释放成功返回 0；handle 不存在或 heap/table 为空返回 -1。
int32_t am_heap_free_handle(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle);

int32_t am_heap_set_metadata(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle, am_uint_t property); // TODO

am_uint_t am_heap_get_metadata(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle); // TODO


am_value_t am_heap_get(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle);

int32_t am_heap_set(am_allocator_t *container_alloc, am_allocator_t *obj_alloc, am_heap_t *heap, am_handle_t handle, am_value_t value); // 不扩容，且set前检查把柄有效性





#ifdef __cplusplus
}
#endif

#endif
