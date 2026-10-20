#ifndef __AM_WSTRING_H__
#define __AM_WSTRING_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "am_object.h"
#include "am_allocator.h"


// 宽字符串（不可扩容）：同时作为基础数据结构和语言数据对象
// NOTE 说明：虽然是基础数据结构，但实质上可作为对象语言的数据对象。详见am_map_t的说明。
typedef struct am_wstring_t {
    am_object_t base;

    size_t     length;     // content字符个数（最后一个字符的下标+1）=content容量
    am_value_t content[];  // Array<am_value_t(am_wchar_t)> 柔性数组
} am_wstring_t;


// 创建并初始化一个字符串对象。字符串对象是不可变的。
// 注意：am_wstring_t.content是am_value_t数组，每个元素是一个am_wchar_t。
am_wstring_t *am_wstring_create(am_allocator_t *alloc, wchar_t *str, size_t length);

// 销毁对象。obj 为 NULL 时视为成功。成功返回 0，失败返回 -1。
int32_t am_wstring_destroy(am_allocator_t *alloc, am_wstring_t *obj);

// 功能说明：拷贝wstring对象。成功则返回新副本对象的指针，失败则返回NULL。
am_wstring_t *am_wstring_copy(am_allocator_t *alloc, am_wstring_t *obj);

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_wstring_size(am_allocator_t *alloc, am_wstring_t *obj);

// 功能说明：将字符串对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
size_t am_wstring_dump(am_allocator_t *alloc, am_wstring_t *obj, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的字符串对象，构造字符串对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_wstring_t对象的指针，失败则返回NULL。
am_wstring_t *am_wstring_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);








///////////////////////////////////////////
// 多值字符串索引表 am_strindex_t<hash_t, am_value_t>
// 用于全局字符串驻留：一个字符串 hash 可对应多个 candidate handle。
///////////////////////////////////////////

// 多值哈希表特殊 key
#define AM_STRINDEX_KEY_EMPTY     ((uint32_t)UINT32_MAX)
#define AM_STRINDEX_KEY_TOMBSTONE ((uint32_t)(UINT32_MAX - 1))

// 表项
typedef struct am_strindex_entry_t {
    uint32_t   hash;  // 字符串内容的 FNV-1a hash tag
    am_value_t value; // 对应的 handle（或其他 am_value_t）
} am_strindex_entry_t;

// 多值字符串索引表（开放寻址 + 线性探测）
// 作为解释器底层基础设施和语言对象的双重身份，与 am_map_t 一致。
typedef struct am_strindex_t {
    am_object_t base;

    size_t length;     // 当前有效键值对数量
    size_t capacity;   // 物理槽位数（必须是2的幂）
    size_t mask;       // capacity - 1
    size_t tombstones; // 墓碑数量
    am_strindex_entry_t slots[]; // 连续槽位区
} am_strindex_t;

// ===============================================================================
// 哈希函数
// ===============================================================================

// 计算 wchar_t 字符串的 FNV-1a 32-bit 哈希值
uint32_t am_strindex_hash_string(const wchar_t *str);

// ===============================================================================
// 构造函数
// ===============================================================================

// 以初始容量新建多值哈希表。capacity 会被向上取整为不小于它的最小 2 的幂。
// 所有 key 初始化为 AM_STRINDEX_KEY_EMPTY，value 初始化为 AM_VALUE_NULL。
am_strindex_t *am_strindex_create(am_allocator_t *alloc, size_t capacity);

// ===============================================================================
// 析构与清理
// ===============================================================================

// 彻底销毁
int32_t am_strindex_destroy(am_allocator_t *alloc, am_strindex_t *obj);

// ===============================================================================
// 拷贝
// ===============================================================================

// 深拷贝：创建并返回一个与原 strindex 内容完全一致的新对象。所有 key/value 按位拷贝。
am_strindex_t *am_strindex_copy(am_allocator_t *alloc, am_strindex_t *obj);

// ===============================================================================
// 对象大小
// ===============================================================================

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_strindex_size(am_allocator_t *alloc, am_strindex_t *obj);

// ===============================================================================
// 对象二进制转储
// ===============================================================================

// 功能说明：将表对象序列化成二进制序列，并转储到buffer[offset]
// 实现说明：offset是写入buffer的起点offset。成功则返回向buffer新增字节数，失败则返回SIZE_MAX。
// 注意：若buffer设为NULL，或者offset设为SIZE_MAX，则仅计算转储后的二进制序列的字节数，不实际写入buffer。
//       压缩对象，将capacity压缩到跟length一致，丢弃墓碑和空闲槽位。
size_t am_strindex_dump(am_allocator_t *alloc, am_strindex_t *obj, uint8_t *buffer, size_t offset);

// 功能说明：转储（dump）操作的逆操作。从二进制字节序列buffer[offset]开始，读取转储的对象，构造对象并返回其指针。
// 实现说明：offset是读取buffer的起点offset。成功则返回加载后am_strindex_t对象的指针，失败则返回NULL。
am_strindex_t *am_strindex_load(am_allocator_t *alloc, uint8_t *buffer, size_t offset);

// ===============================================================================
// 基本操作
// ===============================================================================

// 查找：输入一个wchar_t字符串，计算其uint32_t哈希值，得到所有对应的value的列表（values由调用者管理）。
// values 为 NULL 或 n_values 为 0 时，仅返回匹配条目的数量，不写入 values。
// 返回值为实际匹配条目数量；若不存在则返回 0；若出错则返回 SIZE_MAX。
size_t am_strindex_get_all(am_allocator_t *alloc, am_strindex_t *obj, wchar_t *str, am_value_t *values, size_t n_values);

// 插入新键值对。对输入的字符串计算hash，插入(key=hash,handle)时，直接根据hash找到对应的桶，如果被占用，则往后寻找第一个空桶插入。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有指针。
am_strindex_t *am_strindex_set(am_allocator_t *alloc, am_strindex_t *obj, wchar_t *str, am_value_t value);

// 按已知 hash 直接插入 (hash, value)，不重新计算字符串 hash。
// 当负载因子（含墓碑）超过 75% 时自动扩容。
// 返回新的对象指针；失败返回 NULL。调用者必须使用返回的指针替换原有指针。
am_strindex_t *am_strindex_set_raw(am_allocator_t *alloc, am_strindex_t *obj, uint32_t hash, am_value_t value);

// 删除指定 value（handle）所在的条目。按 value 的位模式精确匹配；删除成功返回 0；未找到返回 -1。
int32_t am_strindex_delete(am_allocator_t *alloc, am_strindex_t *obj, am_value_t value);

// 当前有效键值对数量
size_t am_strindex_length(am_allocator_t *alloc, am_strindex_t *obj);

// 物理槽位数
size_t am_strindex_capacity(am_allocator_t *alloc, am_strindex_t *obj);









#ifdef __cplusplus
}
#endif

#endif
