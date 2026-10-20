#ifndef __AM_NATIVE_TABLE_H__
#define __AM_NATIVE_TABLE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"


////////////////////////////////////////////////////////////////////////////
//  Native Library : Table
////////////////////////////////////////////////////////////////////////////

// 由 src/native_Table.c 定义
extern const am_native_lib_entry_t am_native_Table_lib;


// (Table.make) : Table
// 创建一个新的空哈希表对象，将其 handle 压入操作数栈。
int32_t am_native_Table_make(am_runtime_t *rt, am_process_t *proc);

// (Table.set tbl key value) : #undefined
// 根据 key 设置 value。key 可以是 number、symbol 或短字符串（长度不超过 AM_PROCESS_STRINDEX_MAX_LEN）。
// value 为 #undefined 时相当于删除该 key。
int32_t am_native_Table_set(am_runtime_t *rt, am_process_t *proc);

// (Table.get tbl key) : value | #undefined
// 根据 key 获取 value；key 不存在时返回 #undefined。
int32_t am_native_Table_get(am_runtime_t *rt, am_process_t *proc);

// (Table.keys tbl) : List
// 返回包含 tbl 中所有 key 的列表（顺序不定）。空表返回空列表。
int32_t am_native_Table_keys(am_runtime_t *rt, am_process_t *proc);

// (Table.contains tbl key) : Boolean
// 检查 tbl 中是否存在 key，存在返回 #t，否则返回 #f。
int32_t am_native_Table_contains(am_runtime_t *rt, am_process_t *proc);

// (Table.delete tbl key) : #undefined
// 删除 tbl 中指定的 key。
int32_t am_native_Table_delete(am_runtime_t *rt, am_process_t *proc);

// (Table.length tbl) : Number
// 返回 tbl 中有效 entry 的数量，以 am_float_t 形式压栈。
int32_t am_native_Table_length(am_runtime_t *rt, am_process_t *proc);



#ifdef __cplusplus
}
#endif

#endif
