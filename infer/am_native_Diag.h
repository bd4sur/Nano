////////////////////////////////////////////////////////////////////////////
//  Native Library : Diag —— 灵机（Animac）自观测本地库（"解释器 X 光"）
//
//  用途：把虚拟机内部只读状态暴露给 Scheme，供 app/xray.scm 等"自观测"应用
//  画成仪表盘；也是排查性能问题时的现场仪器（栈深、PC、闭包槽位统计、GC 计数）。
//
//  设计原则：
//    1) **纯只读**：不修改运行时/进程的任何字段、不分配堆对象、不推进 PC 之外的
//       任何状态（每次调用只压一个返回值并 am_process_step），因此可以被任意
//       应用安全调用，不改变程序语义；
//    2) 返回值一律 `am_uint_t`（字符/整型载荷）而非浮点：32 位宿主的浮点只有
//       约 19 位有效位（见 am_native_Time.h 的说明），而这些计数器都能落在
//       27 位无符号范围内；
//    3) 时钟用 **单调实时钟**（clock_gettime(CLOCK_MONOTONIC)），与宿主的
//       `System.timestamp`（可被宿主做成虚拟时钟）无关 —— 这样在回归测试台的
//       虚拟时钟下也能量出真实的吞吐。
//
//  为什么需要"自观测"：本项目此前排查过闭包帧/槽位寻址类缺陷，最费力的一步
//  就是拿不到现场数据（操作数栈深、帧描述符命中率、退化闭包数）。把这些量做进
//  一个 App，比事后加 printf 再重编快得多。
////////////////////////////////////////////////////////////////////////////

#ifndef __AM_NATIVE_DIAG_H__
#define __AM_NATIVE_DIAG_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

// 由 infer/am_native_Diag.c 定义
extern const am_native_lib_entry_t am_native_Diag_lib;

// (Diag.ms) : Number —— 单调毫秒，自本库首次调用起算（不做绝对时间，避免 27 位溢出）
int32_t am_native_Diag_ms(am_runtime_t *rt, am_process_t *proc);

// ---- 运行时级计数器 --------------------------------------------------------
// (Diag.tick) : Number —— 宿主事件循环轮数（rt->tick_counter）
int32_t am_native_Diag_tick(am_runtime_t *rt, am_process_t *proc);
// (Diag.gcs) : Number —— 全局 GC 周期计数（rt->gc_count）
int32_t am_native_Diag_gcs(am_runtime_t *rt, am_process_t *proc);
// (Diag.timeslice) : Number —— 默认时间片（VM 指令周期数）
int32_t am_native_Diag_timeslice(am_runtime_t *rt, am_process_t *proc);
// (Diag.pids) : Number —— 已分配的进程号计数（≈真机上"曾经创建过多少进程"）
int32_t am_native_Diag_pids(am_runtime_t *rt, am_process_t *proc);

// ---- 当前进程的现场 --------------------------------------------------------
// (Diag.pc) : Number —— 程序计数器（下一条指令的 iaddr）
int32_t am_native_Diag_pc(am_runtime_t *rt, am_process_t *proc);
// (Diag.ilcode) : Number —— 指令区长度（iaddr 上限）
int32_t am_native_Diag_ilcode(am_runtime_t *rt, am_process_t *proc);
// (Diag.opstack) : Number —— 操作数栈当前深度（元素个数）
int32_t am_native_Diag_opstack(am_runtime_t *rt, am_process_t *proc);
// (Diag.fstack) : Number —— 调用栈当前深度（帧数）
int32_t am_native_Diag_fstack(am_runtime_t *rt, am_process_t *proc);
// (Diag.pgc) : Number —— 本进程的 GC 次数（proc->gc_count）
int32_t am_native_Diag_pgc(am_runtime_t *rt, am_process_t *proc);

// ---- 闭包帧统计（诊断"帧描述符 vs 退化复制"两条路径的占比）------------------
// (Diag.frames) : Number —— 按帧描述符创建的闭包数
int32_t am_native_Diag_frames(am_runtime_t *rt, am_process_t *proc);
// (Diag.legacy) : Number —— 退化为"复制全部绑定"的闭包数
int32_t am_native_Diag_legacy(am_runtime_t *rt, am_process_t *proc);
// (Diag.slots) : Number —— 描述符路径创建的槽位总数
int32_t am_native_Diag_slots(am_runtime_t *rt, am_process_t *proc);
// (Diag.freeslots) : Number —— 其中自由变量槽总数
int32_t am_native_Diag_freeslots(am_runtime_t *rt, am_process_t *proc);
// (Diag.fillgeneric) : Number —— 自由变量填充走"沿链通用解析"的次数
int32_t am_native_Diag_fillgeneric(am_runtime_t *rt, am_process_t *proc);
// (Diag.fastcl) : Boolean —— 闭包快路径是否可用（System.eval 触发的降级）
int32_t am_native_Diag_fastcl(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif // __AM_NATIVE_DIAG_H__