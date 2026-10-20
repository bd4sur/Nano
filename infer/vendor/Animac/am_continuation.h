#ifndef __AM_CONTINUATION_H__
#define __AM_CONTINUATION_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "am_object.h"
#include "am_allocator.h"

///////////////////////////////////////////
// 语言数据对象：计算续体
///////////////////////////////////////////

// 计算续体（continuation）数据结构，是am_object_t的子类。
// 设计说明：计算续体保存了进程在某一时刻的运行状态快照，包含续体返回iaddr、当前闭包handle、opstack和fstack四个字段。由于opstack和fstack都是朴素数组模拟的栈，且捕获续体后只读不写，故将其展平紧密排列存储到柔性数组stacks中。
// stacks的布局是：[0 ...opstack... (fstack_offset-1)  |  (fstack_offset) ...fstack... (length-1)]
// 即以fstack_offset为界，0<=index<fstack_offset属于opstack，fstack_offset<=index<length属于fstack。index较大的方向是栈顶。
typedef struct am_continuation_t {
    am_object_t base;

    size_t length; // 续体对象stacks字段的长度
    size_t fstack_offset; // stacks数组中，fstack区段起点（栈底）在stacks数组中的offset
    am_iaddr_t cont_return_target;
    am_handle_t current_closure_handle;
    am_handle_t dynamic_wind_stack_handle; // 捕获时刻 dynamic_wind_stack 的深拷贝快照的 handle
    am_handle_t dynamic_wind_after_stack_handle; // 捕获时刻 dynamic_wind_after_stack 的深拷贝快照的 handle
    am_handle_t current_dynamic_wind_entry_handle; // 捕获时刻 proc->current_dynamic_wind_entry
    am_handle_t current_dynamic_wind_thunk_handle; // 捕获时刻 proc->current_dynamic_wind_thunk
    am_value_t stacks[];
} am_continuation_t;


// 构造函数。成功返回指针，失败返回NULL。
am_continuation_t *am_continuation_create(
    am_allocator_t *alloc, am_iaddr_t cont_return_target, am_handle_t current_closure_handle,
    am_value_t *opstack, size_t opstack_length, am_value_t *fstack, size_t fstack_length,
    am_handle_t dynamic_wind_stack_handle);

// 析构函数。成功返回0，失败返回-1
int32_t am_continuation_destroy(am_allocator_t *alloc, am_continuation_t *obj);

// 拷贝
am_continuation_t *am_continuation_copy(am_allocator_t *alloc, am_continuation_t *obj);

// 功能说明：计算对象所占用的实际字节数（考虑结构体填充和对齐问题）
// 成功返回字节数，失败返回SIZE_MAX
size_t am_continuation_size(am_allocator_t *alloc, am_continuation_t *obj);

// 获取opstack数组，用于GC遍历和续体恢复。成功返回新数组指针（通过alloc分配，由调用者负责释放），失败返回NULL。
am_value_t *am_continuation_get_opstack(am_allocator_t *alloc, am_continuation_t *obj, size_t *length);

// 获取fstack数组，用于GC遍历和续体恢复。成功返回新数组指针（通过alloc分配，由调用者负责释放），失败返回NULL。
am_value_t *am_continuation_get_fstack(am_allocator_t *alloc, am_continuation_t *obj, size_t *length);


#ifdef __cplusplus
}
#endif

#endif
