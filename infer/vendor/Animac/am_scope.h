#ifndef __AM_SCOPE_H__
#define __AM_SCOPE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "am_object.h"
#include "am_allocator.h"


///////////////////////////////////////////
// 词法作用域对象（环境帧）
///////////////////////////////////////////

// 词法作用域中的变量绑定字段
typedef struct am_scope_binding_t {
    am_varid_t  varid;
    am_value_t  value;
} am_scope_binding_t;

// 环境帧（支持扩容）
typedef struct am_scope_t {
    am_object_t base;

    am_handle_t parent_scope_handle;
    am_handle_t parent_lambda_handle;
    am_handle_t current_lambda_handle;
    size_t capacity;
    size_t length;
    am_scope_binding_t bindings[]; // 柔性数组，按顺序逐个插入
} am_scope_t;

// 创建环境帧。capacity 为 0 时默认使用 16。
am_scope_t *am_scope_create(am_allocator_t *alloc, am_handle_t parent_scope_handle, am_handle_t parent_lambda_handle, am_handle_t current_lambda_handle, size_t capacity);

// 销毁环境帧。scope 为 NULL 时视为成功。成功返回 0，失败返回 -1。
int32_t am_scope_destroy(am_allocator_t *alloc, am_scope_t *scope);

// 深拷贝（头部与所有 binding）。value 按位拷贝（与 TS Copy 语义一致，不递归释放对象）。
am_scope_t *am_scope_copy(am_allocator_t *alloc, am_scope_t *scope);

// 将对象的二进制内存布局从alloc管理的内存中倒出来，返回一个系统malloc的二进制序列，以及序列长度
//   注意：压缩对象，将capacity压缩到跟length一致，删除多余分配的空闲部分
uint8_t *am_scope_dump(am_allocator_t *alloc, am_scope_t *scope, size_t *size);

// 查询是否存在变量绑定。存在返回 0，不存在或 scope 为 NULL 返回 -1。
int32_t am_scope_has_var(am_allocator_t *alloc, am_scope_t *scope, am_varid_t variable);

// 新增一个变量绑定。若已存在相同变量，则返回NULL表示失败。
// 如涉及扩容，返回新闭包对象指针；否则返回原指针。扩容失败返回NULL。
am_scope_t *am_scope_add_var(am_allocator_t *alloc, am_scope_t *scope, am_varid_t variable, am_value_t value);





#ifdef __cplusplus
}
#endif

#endif
