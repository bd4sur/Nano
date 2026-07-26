#ifndef __AM_NATIVE_SYSTEM_H__
#define __AM_NATIVE_SYSTEM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"


////////////////////////////////////////////////////////////////////////////
//  Native Library : System
////////////////////////////////////////////////////////////////////////////

// 由 src/native_System.c 定义
extern const am_native_lib_entry_t am_native_System_lib;


// (System.exit) : void
// 立即停止当前进程，将其状态置为 STOPPED。不影响已存在的异步任务。
int32_t am_native_System_exit(am_runtime_t *rt, am_process_t *proc);

// (System.kill pid:Number) : Boolean
// 彻底终止指定 PID 的进程：释放其堆、栈、AST 相关表及异步任务，但保留 am_process_t 壳。
int32_t am_native_System_kill(am_runtime_t *rt, am_process_t *proc);

// (System.exec code:String) : Number|-1
// 将 Scheme 源码编译成 module，原地替换当前 process 的全部内容并从 PC=0 开始执行。
// 失败时压栈 -1 并继续执行。
int32_t am_native_System_exec(am_runtime_t *rt, am_process_t *proc);

// (System.set_timeout time_ms:Number callback:(void->undefined)) : Number
int32_t am_native_System_set_timeout(am_runtime_t *rt, am_process_t *proc);

// (System.set_interval time_ms:Number callback:(void->undefined)) : Number
int32_t am_native_System_set_interval(am_runtime_t *rt, am_process_t *proc);

// (System.clear_timeout timer:Number) : void
int32_t am_native_System_clear_timeout(am_runtime_t *rt, am_process_t *proc);

// (System.clear_interval timer:Number) : void
int32_t am_native_System_clear_interval(am_runtime_t *rt, am_process_t *proc);

// (System.timestamp) : Number
int32_t am_native_System_timestamp(am_runtime_t *rt, am_process_t *proc);

// (System.memstat) : List
// 返回 '(vm_capacity vm_used heap_capacity heap_used)，单位 bytes。
int32_t am_native_System_memstat(am_runtime_t *rt, am_process_t *proc);

// (System.test x:any) : String
int32_t am_native_System_test(am_runtime_t *rt, am_process_t *proc);

// (System.fork) : Number
// 深度复制当前进程，创建子进程并加入调度队列。亲进程返回子进程 pid，子进程返回 0。
int32_t am_native_System_fork(am_runtime_t *rt, am_process_t *proc);

// (System.eval codestr:String) : void
// 运行时动态编译并执行 Scheme 代码字符串。仅捕获当前进程的顶级变量绑定。
int32_t am_native_System_eval(am_runtime_t *rt, am_process_t *proc);

// (System.make_queue len:Number) : Number|null
// 创建一个容量为 len 的队列，返回队列编号；失败返回 #null。
int32_t am_native_System_make_queue(am_runtime_t *rt, am_process_t *proc);

// (System.write qid:Number v:any timeout_ms:Number) : Boolean
// 向队列 qid 写入 v，timeout_ms 为超时时间。成功返回 #t，失败返回 #f。
int32_t am_native_System_write(am_runtime_t *rt, am_process_t *proc);

// (System.read qid:Number timeout_ms:Number) : any
// 从队列 qid 读取一个值，timeout_ms 为超时时间。成功返回值，失败返回 #undefined。
int32_t am_native_System_read(am_runtime_t *rt, am_process_t *proc);



#ifdef __cplusplus
}
#endif

#endif
