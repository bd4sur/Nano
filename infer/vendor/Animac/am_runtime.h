#ifndef __AM_RUNTIME_H__
#define __AM_RUNTIME_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

#include "am_object.h"
#include "am_allocator.h"
#include "am_process.h"
#include "am_module.h"
#include "am_list.h"

// 前向声明：回调函数指针需要引用运行时类型
typedef struct am_runtime_t am_runtime_t;


///////////////////////////////////////////
// 虚拟机状态
///////////////////////////////////////////

#define AM_VM_STATE_IDLE    (0)
#define AM_VM_STATE_RUNNING (1)


///////////////////////////////////////////
// 运行时事件处理器配置
///////////////////////////////////////////

#ifndef AM_COMPUTATION_PHASE_LENGTH
#define AM_COMPUTATION_PHASE_LENGTH (1)
#endif


///////////////////////////////////////////
// 异步定时器基础设施
///////////////////////////////////////////

// 时间戳类型：毫秒
typedef uint64_t am_timestamp_t;

// 定时器条目（不透明类型，具体定义见 runtime.c）
typedef struct am_timer_t am_timer_t;

// 注册一个定时器。delay_ms 为首次触发的延迟，repeat 表示是否周期触发，
// interval_ms 为周期触发的间隔。成功返回大于 0 的定时器编号，失败返回 0。
size_t am_runtime_set_timer(am_runtime_t *rt, am_pid_t pid, am_handle_t callback,
                            am_timestamp_t delay_ms, bool repeat, am_timestamp_t interval_ms);

// 根据编号取消一个定时器。成功返回 true，未找到返回 false。
bool am_runtime_clear_timer(am_runtime_t *rt, size_t timer_id);

// 获取当前时间戳（毫秒）。经由 vtable 分派到宿主实现。
am_timestamp_t am_runtime_now_ms(am_runtime_t *rt);

// 以异步方式调用一个闭包：压入栈帧并跳转到闭包入口，返回地址为 return_target。
// 用于定时器回调等场景。成功返回 0，失败返回 -1。
int32_t am_runtime_call_async(am_runtime_t *rt, am_process_t *proc, am_handle_t callback,
                              am_iaddr_t return_target);


///////////////////////////////////////////
// 宿主虚函数表（依赖倒置）
// 说明：runtime 不直接依赖宿主提供的输入输出回调、定时器与时间戳函数，
// 而是由宿主在调用 am_runtime_create 时，通过本虚函数表注入具体实现。
// on_tick/on_event/on_halt/on_error 允许为 NULL（不触发）；
// now_ms/sleep_in_ms 为必需能力，为 NULL 时 am_runtime_create 失败。
///////////////////////////////////////////

typedef struct am_runtime_vtable_t {
    void (*on_tick)(am_runtime_t *rt);    // 每个 Tick 结束后触发
    void (*on_event)(am_runtime_t *rt);   // 每个事件循环结束后触发
    void (*on_halt)(am_runtime_t *rt);    // 虚拟机进入 IDLE 后触发
    void (*on_error)(am_runtime_t *rt);   // 虚拟机捕获异常时触发
    void (*sleep_in_ms)(am_runtime_t *rt, am_timestamp_t ms); // 短时睡眠（毫秒）
    am_timestamp_t (*now_ms)(am_runtime_t *rt);               // 获取当前时间戳（毫秒）
} am_runtime_vtable_t;


///////////////////////////////////////////
// 运行时环境
// 说明：运行时是虚拟机调度的核心，管理进程池、进程队列、FIFO 和回调。
///////////////////////////////////////////

typedef struct am_runtime_t {
    am_allocator_t *vm_alloc;   // 运行时工作内存分配器
    am_allocator_t *heap_alloc; // 进程堆内存分配器

    wchar_t *working_dir;       // 基准工作目录

    size_t process_pool_capacity; // 进程池容量
    am_process_t **process_pool;  // 进程池：am_process_t* 动态数组
    size_t process_poll_counter;  // 进程计数器，也作为下一个 PID
    am_list_t *process_queue;     // 进程队列：List<am_value_t(uint:pid)>

    am_list_t *input_fifo;   // 输入 FIFO（存储 wchar 值）
    am_list_t *output_fifo;  // 输出 FIFO（存储 wchar 值）
    am_list_t *error_fifo;   // 错误 FIFO（存储 wchar 值）

    am_list_t *queue_list;   // 队列列表：List<am_queue_t*>
    size_t queue_next_id;    // 下一个队列编号，从 1 开始递增

    const am_runtime_vtable_t *vtable;  // 宿主虚函数表（由 am_runtime_create 注入，宿主拥有其生命周期）

    size_t tick_counter;     // Tick 计数器
    size_t gc_count;         // 全局 GC 周期计数器（作为 am_gc_collect 的 gc_seq）
    size_t gc_periodic_counter; // 事件循环轮计数器（周期兜底 GC 用）

    uint32_t timeslice;      // 默认时间片长度（单位：VM指令周期数）

    am_timer_t *timer_list;  // 定时器链表头
    size_t timer_next_id;    // 下一个定时器编号

    void *host_context;      // 宿主提供的全局不透明上下文
} am_runtime_t;


///////////////////////////////////////////
// 生命周期
///////////////////////////////////////////

// 创建运行时。成功返回运行时指针，失败返回 NULL。
// base_dir 为基准工作目录，允许为 NULL。
// vtable 为宿主虚函数表，不允许为 NULL，且其 now_ms/sleep_in_ms 成员不允许为 NULL；
// runtime 仅保存指针，不拷贝，宿主须保证 vtable 的生命周期不短于 runtime。
am_runtime_t *am_runtime_create(am_allocator_t *vm_alloc, am_allocator_t *heap_alloc, const wchar_t *base_dir,
                                const am_runtime_vtable_t *vtable);

// 销毁运行时，释放其占用的全部资源。成功返回 0，失败返回 -1。
int32_t am_runtime_destroy(am_runtime_t *rt);


///////////////////////////////////////////
// 入口函数（兼容参考用法）
///////////////////////////////////////////

// 将模块加载到运行时中，创建并启动一个进程。成功返回 PID，失败返回 -1。
am_pid_t am_runtime_load_module(am_runtime_t *rt, am_module_t *mod);

// 启动虚拟机主循环，直到所有进程执行结束进入 IDLE。
void am_runtime_start(am_runtime_t *rt);


///////////////////////////////////////////
// 队列 IPC 基础设施
///////////////////////////////////////////

typedef struct am_queue_waiter_t am_queue_waiter_t;
typedef struct am_queue_t am_queue_t;

// 队列阻塞等待者
struct am_queue_waiter_t {
    am_pid_t pid;                 // 阻塞的进程 ID
    am_value_t value;             // 发送等待者要写入的值（接收等待者忽略）
    am_timestamp_t deadline_ms;   // 超时绝对时间（毫秒）
    bool is_writer;               // true=发送等待者，false=接收等待者
    am_queue_waiter_t *next;      // 链表下一个节点
};

// 多生产者多消费者 FIFO 队列
struct am_queue_t {
    size_t id;                    // 队列编号
    size_t capacity;              // 最大容量
    am_list_t *items;             // 数据项列表（FIFO）
    am_queue_waiter_t *send_waiters; // 等待可写的发送者链表
    am_queue_waiter_t *recv_waiters; // 等待可读的接收者链表
};

// 根据 ID 查找队列。成功返回指针，失败返回 NULL。
am_queue_t *am_runtime_get_queue(am_runtime_t *rt, size_t queue_id);

// 创建一个容量为 capacity 的队列。成功返回队列指针，失败返回 NULL。
am_queue_t *am_runtime_queue_create(am_runtime_t *rt, size_t capacity);

// 销毁队列并释放其占用的全部资源。成功返回 0，失败返回 -1。
int32_t am_runtime_queue_destroy(am_runtime_t *rt, am_queue_t *q);

// 尝试/阻塞地向队列写入一个值。由 native_System.write 调用。
// 立即成功、立即失败或超时失败时都会直接修改 proc 的操作数栈并步进 PC；
// 进入阻塞时设置进程状态并返回 0，不步进 PC。
int32_t am_runtime_queue_write(am_runtime_t *rt, am_queue_t *q, am_value_t value,
                               am_timestamp_t timeout_ms, am_process_t *proc);

// 尝试/阻塞地从队列读取一个值。由 native_System.read 调用。
// 立即成功、立即失败或超时失败时都会直接修改 proc 的操作数栈并步进 PC；
// 进入阻塞时设置进程状态并返回 0，不步进 PC。
int32_t am_runtime_queue_read(am_runtime_t *rt, am_queue_t *q, am_timestamp_t timeout_ms,
                              am_process_t *proc);


///////////////////////////////////////////
// 模块与进程管理
///////////////////////////////////////////

// 将模块加载到运行时中。成功返回 PID，失败返回 (am_pid_t)-1。
am_pid_t am_runtime_load_module(am_runtime_t *rt, am_module_t *mod);

// 根据 PID 获取进程。成功返回进程指针，失败返回 NULL。
am_process_t *am_runtime_get_process(am_runtime_t *rt, am_pid_t pid);

// 彻底终止指定 PID 的进程：释放其堆、栈、AST 相关表及异步任务，但保留 am_process_t 壳。
// 允许在目标进程自己的 native 调用中同步调用，此时会标记为延迟销毁，由调度器安全点完成。
// 成功返回 0；pid 无效或进程已是 KILLED 返回 -1。
int32_t am_runtime_kill_process(am_runtime_t *rt, am_pid_t pid);

// 直接设置 rt->timeslice 字段（单位：VM指令周期数）
void am_runtime_set_default_timeslice(am_runtime_t *rt, uint32_t ticks);

// 根据 pid 返回对应的 process 对象。若失败，返回 NULL。
am_process_t *am_rumtime_get_process_by_pid(am_runtime_t *rt, am_pid_t pid);

// 设置/获取 VM 的全局宿主上下文（不透明数据）。设置成功返回 0，失败返回 -1。
int32_t am_set_runtime_host_context(am_runtime_t *rt, void *ctx);
void *am_get_runtime_host_context(am_runtime_t *rt);

// 设置/获取某进程的宿主上下文（不透明数据）。设置成功返回 0，失败返回 -1。
int32_t am_set_process_host_context(am_runtime_t *rt, am_process_t *proc, void *ctx);
void *am_get_process_host_context(am_runtime_t *rt, am_process_t *proc);


///////////////////////////////////////////
// 调度器
///////////////////////////////////////////

// 执行一次事件循环：执行若干 Tick，触发 GC 和事件回调。
// 返回 AM_VM_STATE_IDLE 或 AM_VM_STATE_RUNNING。
int32_t am_runtime_event_handler(am_runtime_t *rt);

// 执行一个时间片。返回 AM_VM_STATE_IDLE 或 AM_VM_STATE_RUNNING。
int32_t am_runtime_tick(am_runtime_t *rt, uint32_t timeslice);

// 根据opcode和operand分派具体的执行逻辑（指令译码）
int32_t am_runtime_op_dispatch(am_runtime_t *rt, am_process_t *proc, uint32_t opcode, am_value_t operand);

// 执行当前进程的一条指令。成功返回 0，失败返回 -1。
int32_t am_runtime_execute(am_runtime_t *rt, am_process_t *proc);

// 启动虚拟机主循环。
void am_runtime_start(am_runtime_t *rt);


///////////////////////////////////////////
// 内存统计
///////////////////////////////////////////

// 运行时内存统计快照（与 allocator 实现策略无关的抽象结构）。
typedef struct {
    size_t vm_capacity;   // VM 工作区容量（bytes）
    size_t vm_used;       // VM 工作区已用（bytes）
    size_t heap_capacity; // 用户堆区容量（bytes）
    size_t heap_used;     // 用户堆区已用（bytes）
} am_runtime_memory_stats_t;

// 获取运行时内存统计快照。成功返回 0，失败返回 -1。
int32_t am_runtime_get_memory_stats(am_runtime_t *rt, am_runtime_memory_stats_t *out);

// 打印运行时内存总体使用状况（VM 工作区 + 用户堆区）。
// 可在运行时任意时刻调用，接口与 allocator 实现策略无关。
void am_runtime_print_memory_stats(am_runtime_t *rt);


///////////////////////////////////////////
// 控制台输入输出
///////////////////////////////////////////

// 向 stdout 输出字符串，并记录到 output_fifo。
void am_runtime_output(am_runtime_t *rt, const wchar_t *str);

// 向 stderr 输出字符串，并记录到 error_fifo。
void am_runtime_error(am_runtime_t *rt, const wchar_t *str);


///////////////////////////////////////////
// 其他辅助函数
///////////////////////////////////////////

// 将数值 TPV 统一转换为浮点数
am_float_t am_runtime_number_to_float(am_value_t v);

// 将数值 TPV 统一（强制）转换为int
am_int_t am_runtime_number_to_int(am_value_t v);

// 将数值 TPV 统一（强制）转换为uint
am_int_t am_runtime_number_to_uint(am_value_t v);



///////////////////////////////////////////
// 本地宿主函数机制（Native）
///////////////////////////////////////////

// Native函数指针类型：与op_*指令函数签名一致
typedef int32_t (*am_native_func_t)(am_runtime_t *rt, am_process_t *proc);


// 函数表项：库内的单个函数
typedef struct am_native_func_entry_t {
    const wchar_t *name; // 函数名（suffix）
    am_native_func_t func;
} am_native_func_entry_t;


// 库表项：一个native库及其函数表
typedef struct am_native_lib_entry_t {
    const wchar_t *name;                  // 库名（prefix / native_id）
    const am_native_func_entry_t *funcs;  // 该库的函数表
    size_t func_count;                    // 函数表长度
} am_native_lib_entry_t;


#ifndef AM_NATIVE_MAX_LIBS
#define AM_NATIVE_MAX_LIBS (16)
#endif

// 向运行时注册一个native库。成功返回0，失败返回-1。
int32_t am_runtime_register_native_lib(am_runtime_t *rt, const am_native_lib_entry_t *lib);


// 运行时查表：根据库名和函数名查找对应的Native函数实现。
// 成功返回函数指针，失败返回NULL。
am_native_func_t am_native_find_func(const wchar_t *lib_name, const wchar_t *func_name);


#ifdef __cplusplus
}
#endif

#endif
