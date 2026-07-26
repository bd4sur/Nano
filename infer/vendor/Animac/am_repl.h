#ifndef __AM_REPL_H__
#define __AM_REPL_H__

#include <stddef.h>
#include <stdint.h>

#include "animac.h"

#ifdef __cplusplus
extern "C" {
#endif

// REPL 上下文状态码，供外层逻辑决定如何呈现。
typedef enum {
    AM_REPL_STATUS_CONTINUE = 0,  // 表达式不完整，需要继续输入
    AM_REPL_STATUS_OUTPUT,        // 产生正常输出
    AM_REPL_STATUS_ERROR,         // 发生错误
    AM_REPL_STATUS_EXIT,          // 请求退出
} am_repl_status_t;

// REPL 上下文。内部状态直接暴露给外层，便于外层根据状态决定呈现方式。
typedef struct repl_ctx {
    // 运行时基础设施
    size_t pool_size;  // 内存池总大小（字节），由 am_repl_ctx_create 指定
    am_allocator_pool_t *pool;
    am_allocator_t *vm_alloc;
    am_allocator_t *heap_alloc;
    wchar_t *base_dir;
    am_runtime_t *rt;
    am_pid_t pid;
    volatile int runtime_error;
    volatile int should_stop;  // 由 am_repl_ctx_interrupt() 置位，用于中断运行

    // 会话与当前累积输入
    wchar_t *session;
    size_t session_len;
    wchar_t *accum;
    size_t accum_len;
    int multiline;

    // 标准输出缓冲区（由上下文拥有，有效到下一次 feed 或 destroy）
    char *stdout_buf;
    size_t stdout_cap;
    size_t stdout_len;

    // 错误输出缓冲区（由上下文拥有，有效到下一次 feed 或 destroy）
    char *stderr_buf;
    size_t stderr_cap;
    size_t stderr_len;

    // 模式与提示符
    int js_mode;
    const char *prompt_main;
    const char *prompt_cont;

    // 供外层读取的内部状态
    am_repl_status_t status;
    const char *output;  // 指向 stdout_buf 或 NULL
    const char *error;   // 指向 stderr_buf 或 NULL
    int indent;          // 建议的续行缩进层级
} am_repl_ctx_t;

// 一次 feed 的返回结果。output/error 均指向 ctx 内部缓冲区，调用者无需释放。
typedef struct {
    am_repl_status_t status;
    const char *output;
    const char *error;
    int indent;
} am_repl_result_t;

// 建立 REPL 上下文。pool_size 指定内存池总大小（单位：字节）。
am_repl_ctx_t *am_repl_ctx_create(size_t pool_size);

// 销毁 REPL 上下文。
void am_repl_ctx_destroy(am_repl_ctx_t *ctx);

// 中断当前上下文中的运行（例如响应 SIGINT）。
void am_repl_ctx_interrupt(am_repl_ctx_t *ctx);

// 设置 JS 解释器模式。
void am_repl_ctx_set_js_mode(am_repl_ctx_t *ctx, int js_mode);

// 向 REPL 上下文传入一个字符串（通常是一行输入），返回结果并更新上下文内部状态。
am_repl_result_t am_repl_ctx_feed(am_repl_ctx_t *ctx, const char *input);

#ifdef __cplusplus
}
#endif

#endif // __AM_REPL_H__
