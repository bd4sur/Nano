#ifndef __AM_PROCESS_H__
#define __AM_PROCESS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

#include "am_object.h"
#include "am_allocator.h"
#include "am_heap.h"
#include "am_closure.h"
#include "am_continuation.h"
#include "am_wstring.h"
#include "am_compiler.h"
#include "am_ast.h"
#include "am_list.h"
#include "am_module.h"


///////////////////////////////////////////
// 进程状态常量
///////////////////////////////////////////

#define AM_PROCESS_STATE_READY     (1)
#define AM_PROCESS_STATE_RUNNING   (2)
#define AM_PROCESS_STATE_SLEEPING  (3)
#define AM_PROCESS_STATE_SUSPENDED (4)
#define AM_PROCESS_STATE_STOPPED   (5)
#define AM_PROCESS_STATE_BLOCKED   (6)
#define AM_PROCESS_STATE_KILLED    (7)


///////////////////////////////////////////
// 进程ID
///////////////////////////////////////////

typedef size_t am_pid_t;


///////////////////////////////////////////
// 进程数据结构
// 说明：进程是Scheme解释器的核心运行时数据结构，包含执行所需的全部状态。
//       每个进程拥有独立的堆、操作数栈和函数调用栈，是虚拟机调度的基本单位。
///////////////////////////////////////////

typedef struct am_process_t {
    am_object_t base; // 基类头：am_process_t也视为对象语言的数据对象

    am_allocator_t *vm_alloc; // VM工作内存分配器
    am_allocator_t *heap_alloc; // 堆内存分配器

    am_pid_t pid;        // 进程ID
    am_pid_t parent_pid; // 亲进程ID
    int32_t state;       // 进程状态

    am_iaddr_t PC;     // 程序计数器：代表下一条指令的iaddr
    am_instruction_t *ilcode; // 中间语言代码
    am_iaddr_t ilcode_length; // 中间语言代码长度

    am_heap_t *heap;   // 进程私有堆（由堆内存专用allocator管理）

    am_strindex_t *strindex;  // 用于全局字符串驻留的多值哈希表，检查某个字符串（的哈希值）是否已存在于heap

    am_vocab_t *var_vocab;    // 变量词表
    am_vocab_t *symbol_vocab; // 符号词表
    am_list_t *var_type;      // 变量类型表（内容同AST，用于运行时判断变量类型，尤其是native_ref）
    am_map_t *natives;        // 本地库记录（内容同AST，用于判断模块使用了哪些本地库）
    am_list_t *var_top;       // 顶级变量varid列表（内容同AST）
    am_map_t *var_arn_mapping; // 变量ARN（Alpha-renaming）前后的映射（内容同AST）

    am_handle_t current_closure_handle; // 指向当前闭包的把柄

    bool pending_kill; // 延迟kill标记：在进程自己的native调用中触发kill时，由调度器安全点完成实际销毁

    size_t gc_count; // GC 触发次数，用于控制标记-压缩频率

    // 操作数栈（其容量为opstack_depth）
    am_value_t *opstack;
    am_value_t *opstack_top; // opstack栈顶指针
    size_t opstack_capacity; // opstack容量（am_value_t元素个数）

    // 函数调用栈（默认容量1000，TODO 后面改成可配置）
    // 注意，成对入栈出栈，栈帧结构为{am_value_t(handle) closure_handle; am_value_t(iaddr) return_target_iaddr; }
    am_value_t *fstack;
    am_value_t *fstack_top; // fstack栈顶指针，注意每次操作加减2个元素
    size_t fstack_capacity; // fstack容量（am_value_t元素个数）

    // dynamic-wind 相关状态
    am_list_t *dynamic_wind_stack;      // 当前 dynamic-wind 栈，元素为 entry handle
    am_list_t *dynamic_wind_after_stack; // 正在执行 after 的条目 handle 栈（与 dynamic_wind_stack 配合）
    size_t     dynamic_wind_mark_counter; // 自增唯一 mark
    am_handle_t current_dynamic_wind_entry; // 当前 dynamic-wind 条目 handle（在 before/thunk 之间暂存）
    am_handle_t current_dynamic_wind_thunk; // 当前 dynamic-wind 的 thunk handle（在 thunk 执行前暂存）

    // continuation 恢复时的 wind 跳板状态
    am_iaddr_t wind_trampoline_iaddr;   // 进程中预留的 wind 指令地址
    int32_t    wind_state;              // 0=空闲, 1=执行 afters, 2=执行 befores, 3=恢复续体
    am_handle_t pending_cont_handle;    // 待恢复的目标续体 handle
    am_value_t  pending_cont_value;     // 调用续体时传入的值
    am_handle_t *pending_after_entries; // 待执行的 after 条目 handle 数组
    size_t      pending_after_count;
    am_handle_t *pending_before_entries;// 待执行的 before 条目 handle 数组
    size_t      pending_before_count;

    void *host_context;     // 宿主提供的不透明上下文
} am_process_t;


///////////////////////////////////////////
// 字符串驻留相关
///////////////////////////////////////////

// 运行时字符串驻留长度阈值：仅对长度不超过该值的字符串启用同值复用
#ifndef AM_PROCESS_STRINDEX_MAX_LEN
#define AM_PROCESS_STRINDEX_MAX_LEN (32)
#endif

// 功能说明：根据 wchar_t 缓冲区和长度创建/复用字符串堆对象，并返回其 handle。
// 实现说明：当 len <= AM_PROCESS_STRINDEX_MAX_LEN 时，会先查询 proc->strindex；
//         若已存在内容相同的字符串则复用其 handle，否则新建并登记。
//         超过阈值的字符串直接新建，不参与驻留。
//         失败返回 AM_HANDLE_NULL。
am_handle_t am_process_make_wstring_handle(am_process_t *proc, const wchar_t *str, size_t len);


///////////////////////////////////////////
// 生命周期
///////////////////////////////////////////

// 功能说明：从模块构造并初始化一个新的进程数据结构
// 实现说明：成功返回新进程对象指针；失败返回NULL
am_process_t *am_process_load_from_module(am_allocator_t *vm_alloc, am_allocator_t *heap_alloc, am_module_t *mod);

// 功能说明：销毁进程数据结构，释放其占用的全部资源
// 实现说明：成功返回0，失败返回-1
int32_t am_process_destroy(am_process_t *proc);


///////////////////////////////////////////
// 操作数栈操作
///////////////////////////////////////////

// 功能说明：向操作数栈中压入值。成功返回0，失败返回-1
int32_t am_process_push_operand(am_process_t *proc, am_value_t v);

// 功能说明：从操作数栈中弹出一个值。成功返回弹出值，失败返回UINTPTR_MAX
am_value_t am_process_pop_operand(am_process_t *proc);

// 功能说明：根据栈顶指针计算opstack中有多少个am_value_t。成功返回长度值，失败返回SIZE_MAX
size_t am_process_length_of_opstack(am_process_t *proc);


///////////////////////////////////////////
// 函数调用栈操作
///////////////////////////////////////////

// 功能说明：向fstack中压入栈帧（两个值）。成功返回0，失败返回-1
int32_t am_process_push_stack_frame(am_process_t *proc, am_value_t closure_handle_value, am_value_t return_target_iaddr_value);

// 功能说明：从fstack中弹出栈帧的两个值，通过两个指针传出。成功返回0，失败返回-1
int32_t am_process_pop_stack_frame(am_process_t *proc, am_value_t *closure_handle_value, am_value_t *return_target_iaddr_value);

// 功能说明：根据栈顶指针计算fstack中有多少个am_value_t（因为是成对push/pop，所以正常情况下必为偶数）。成功返回长度值，失败返回SIZE_MAX
size_t am_process_length_of_fstack(am_process_t *proc);


///////////////////////////////////////////
// 闭包操作
///////////////////////////////////////////

// 功能说明：新建闭包并返回其handle。成功返回handle，失败返回AM_HANDLE_NULL
am_handle_t am_process_make_closure(am_process_t *proc, am_iaddr_t iaddr, am_handle_t parent);

// 功能说明：根据闭包handle获取闭包对象。成功返回指针，失败返回NULL
am_obj_closure_t *am_process_get_closure(am_process_t *proc, am_handle_t hd);

// 功能说明：获取进程的当前闭包对象。成功返回指针，失败返回NULL
am_obj_closure_t *am_process_get_current_closure(am_process_t *proc);

// 功能说明：设置进程的当前闭包handle字段。成功返回0，失败返回-1
static inline int32_t am_process_set_current_closure(am_process_t *proc, am_handle_t hd) {
    if (!proc) return -1;
    proc->current_closure_handle = hd;
    return 0;
}

// 功能说明：变量解引用。成功返回TPV，失败返回UINTPTR_MAX
am_value_t am_process_dereference(am_process_t *proc, am_varid_t varid);


// 功能说明：将进程堆中的列表对象转换为可显示宽字符串。成功返回新分配的 wchar_t*（由调用者释放），失败返回 NULL。
// 实现说明：从 proc->heap 中取得对象，从 proc->var_vocab / proc->symbol_vocab 中解析变量名和符号名。
//          symbol 的处理规则：不在 quote 列表内时带前导单引号；在 quote 列表内时不带前导单引号。
wchar_t *am_process_list_to_string(am_process_t *proc, am_handle_t hd, size_t *length);


///////////////////////////////////////////
// 程序流程控制
///////////////////////////////////////////

// 功能说明：获取当前指令，并取出opcode和operand。成功返回0，失败返回-1
int32_t am_process_current_instruction(am_process_t *proc, uint32_t *opcode, am_value_t *operand);

// 功能说明：前进一步（PC加1）
void am_process_step(am_process_t *proc);

// 功能说明：无条件跳转（PC置数iaddr）
void am_process_goto(am_process_t *proc, am_iaddr_t iaddr);

// 功能说明：设置进程状态
void am_process_set_state(am_process_t *proc, int32_t s);


///////////////////////////////////////////
// 计算续体（continuation）的捕获和恢复
///////////////////////////////////////////

// 功能说明：捕获当前续体，保存为堆对象，并返回其handle。成功返回handle，失败返回AM_HANDLE_NULL
am_handle_t am_process_capture_continuation(am_process_t *proc, am_iaddr_t cont_return_target_iaddr);

// 功能说明：恢复指定的计算续体到当前进程。成功返回其返回目标位置的iaddr，失败返回SIZE_MAX
// 实现说明：传入的 value 为调用续体时传入的值；若需要 wind 调整，则 value 暂存于 proc，待跳板恢复时压栈。
am_iaddr_t am_process_load_continuation(am_process_t *proc, am_handle_t hd, am_value_t value);

// 功能说明：直接恢复续体快照（opstack/fstack/closure），不执行 wind 调整。成功返回 cont_return_target，失败返回 SIZE_MAX
am_iaddr_t am_process_restore_continuation_snapshot(am_process_t *proc, am_handle_t hd);


#ifdef __cplusplus
}
#endif

#endif
