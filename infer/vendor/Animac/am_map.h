#ifndef __AM_MAP_H__
#define __AM_MAP_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "am_object.h"
#include "am_allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

///////////////////////////////////////////
// 基础数据结构：通用散列表 am_map_t<am_value_t,am_value_t>
///////////////////////////////////////////

// 定义两个特殊key
#define AM_MAP_KEY_EMPTY AM_VALUE_NULL
#define AM_MAP_KEY_TOMBSTONE AM_VALUE_UNDEFINED

// 表项
typedef struct am_map_entry_t {
    am_value_t key;
    am_value_t value;
} am_map_entry_t;

// 散列表（开放寻址法）
// NOTE 说明：虽然am_map_t是基础设施，但定义上让它携带am_object_t头（base），赋予其解释器基础设施和语言对象的双重身份。
//           am_map_t作为解释器的底层基础设施使用时，由解释器本身和宿主环境管理；
//           而作为对象语言的数据对象使用时，它作为am_object_t的一个派生类，接受抽象堆和内存分配器的管理。
typedef struct am_map_t {
    am_object_t base;

    size_t length;     // 当前有效键值对数量
    size_t capacity;   // 物理槽位数 (必须是2的幂)
    size_t mask;       // capacity - 1，用于快速取模
    size_t tombstones; // 墓碑数量，用于触发重哈希
    am_map_entry_t slots[];  // 连续槽位区
} am_map_t;

// 遍历回调类型
typedef void (*am_map_iter_callback_t)(am_value_t key, am_value_t value, void *user_data);

// ===============================================================================
// 通用辅助函数（按位操作）
// ===============================================================================

// 计算 am_value_t 的哈希值（基于其底层位模式）
static inline uint32_t am_value_hash(am_value_t v) {
    uint32_t h = (uint32_t)v;
#if UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFu
    h ^= (uint32_t)(v >> 32);
#endif
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// am_value_t 相等性：按位比较
static inline bool am_value_equal(am_value_t a, am_value_t b) {
    return a == b;
}

// ===============================================================================
// 构造函数
// ===============================================================================

// 以初始容量新建哈希表。capacity 会被向上取整为不小于它的最小 2 的幂。
// 所有 key 初始化为 AM_MAP_KEY_EMPTY，value 初始化为 AM_VALUE_NULL。
am_map_t *am_map_create(am_allocator_t *alloc, size_t capacity);

// ===============================================================================
// 析构与清理
// ===============================================================================

// 清空哈希表：对所有有效 entry，若 value 是指针则先释放，再将 key 置为 EMPTY、value 置为 NULL
int32_t am_map_clear(am_allocator_t *alloc, am_map_t *map);

// 彻底销毁哈希表对象
int32_t am_map_destroy(am_allocator_t *alloc, am_map_t *map);

// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝：创建并返回一个与原 map 内容完全一致的新 map 对象。
// 所有 key/value 按位拷贝（与闭包 Copy 语义一致，不递归拷贝指针指向的对象）。
am_map_t *am_map_copy(am_allocator_t *alloc, am_map_t *map);

// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_map_size(am_allocator_t *alloc, am_map_t *obj);

// ===============================================================================
// 对象二进制转储 TODO
// ===============================================================================

// 功能说明：将散列表对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，丢弃墓碑和空闲槽位。
size_t am_map_dump(am_allocator_t *alloc, am_map_t *map, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的散列表对象，构造散列表对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_map_t对象的指针，失败则返回NULL。
am_map_t *am_map_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);

// ===============================================================================
// 基本操作
// ===============================================================================

// 查找：返回对应的 value；若不存在返回 AM_VALUE_NULL
am_value_t am_map_get(am_allocator_t *alloc, am_map_t *map, am_value_t key);

// 存在性检查：存在返回 0，不存在返回 -1
int32_t am_map_contains(am_allocator_t *alloc, am_map_t *map, am_value_t key);

// 不扩容地插入或修改（stable 版本）。
// 仅做插入/替换，绝不分配或释放 map 对象本身，因此 map 指针保持稳定。
// 若 map 已满且 key 不存在，返回 -1；成功返回 0。
// 替换已存在的 key 时，会释放旧的指针 value。
int32_t am_map_set_stable(am_allocator_t *alloc, am_map_t *map, am_value_t key, am_value_t value);

// 插入或修改。
// 插入新键值对；若 key 已存在则替换 value，并释放旧的指针 value。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的 map 对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有 map 指针。
am_map_t *am_map_set(am_allocator_t *alloc, am_map_t *map, am_value_t key, am_value_t value);

// 删除指定 key。若存在且 value 为指针则释放。
// 删除成功返回 0；key 不存在返回 -1。
int32_t am_map_delete(am_allocator_t *alloc, am_map_t *map, am_value_t key);

// 当前有效键值对数量
size_t am_map_length(am_allocator_t *alloc, am_map_t *map);

// 物理槽位数
size_t am_map_capacity(am_allocator_t *alloc, am_map_t *map);

// ===============================================================================
// 遍历与键列表
// ===============================================================================

// 遍历所有有效键值对，调用回调 cb
void am_map_iter(am_allocator_t *alloc, am_map_t *map, am_map_iter_callback_t cb, void *user_data);

// 获取所有 key 的副本列表，使用 allocator 分配。
// 调用者负责使用 am_free(alloc, ...) 释放返回的指针；size 为 0 时返回 NULL。
am_value_t *am_map_keys(am_allocator_t *alloc, am_map_t *map);

#ifdef __cplusplus
}
#endif

#endif
