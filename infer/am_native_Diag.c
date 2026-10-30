////////////////////////////////////////////////////////////////////////////
//  Native Library : Diag —— 设计说明见 am_native_Diag.h
//
//  全部函数都是"取值 → 压栈 → am_process_step → return 0"的形态，
//  与 am_native_Time.c 一致（值型 native 的栈纪律）。
////////////////////////////////////////////////////////////////////////////

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "am_native_Diag.h"
#include "am_host.h"

// ---- 单调时钟 --------------------------------------------------------------
// 只用于"经过多久"，因此不需要绝对纪元；自首次调用起算，天然落在 27 位内
// （2^27 ms ≈ 37 小时，足够一次演示/一次性能采样的时长）。

static uint64_t s_boot_ms = 0;
static int      s_boot_ready = 0;

static uint64_t diag_now_ms(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)(ts.tv_nsec / 1000000);
    }
#endif
    // 退化：用宿主 HAL 的墙钟（可能被宿主做成虚拟时钟，精度/单调性不保证）
    return (uint64_t)am_current_timestamp_in_ms();
}

// ---- 值型 native 的收尾：压入无符号整数并推进 PC ---------------------------

static int32_t diag_push_uint(am_process_t *proc, uint64_t v) {
    if (am_process_push_operand(proc, am_make_value_of_uint((am_uint_t)v)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t diag_push_bool(am_process_t *proc, bool b) {
    if (am_process_push_operand(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE) != 0) return -1;
    am_process_step(proc);
    return 0;
}

int32_t am_native_Diag_ms(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    uint64_t now = diag_now_ms();
    if (!s_boot_ready) { s_boot_ms = now; s_boot_ready = 1; }
    return diag_push_uint(proc, now - s_boot_ms);
}

// ---- 运行时级 --------------------------------------------------------------

int32_t am_native_Diag_tick(am_runtime_t *rt, am_process_t *proc) {
    return diag_push_uint(proc, rt ? (uint64_t)rt->tick_counter : 0);
}

int32_t am_native_Diag_gcs(am_runtime_t *rt, am_process_t *proc) {
    return diag_push_uint(proc, rt ? (uint64_t)rt->gc_count : 0);
}

int32_t am_native_Diag_timeslice(am_runtime_t *rt, am_process_t *proc) {
    return diag_push_uint(proc, rt ? (uint64_t)rt->timeslice : 0);
}

int32_t am_native_Diag_pids(am_runtime_t *rt, am_process_t *proc) {
    return diag_push_uint(proc, rt ? (uint64_t)rt->process_poll_counter : 0);
}

// ---- 当前进程现场 ----------------------------------------------------------

int32_t am_native_Diag_pc(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_uint(proc, proc ? (uint64_t)proc->PC : 0);
}

int32_t am_native_Diag_ilcode(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_uint(proc, proc ? (uint64_t)proc->ilcode_length : 0);
}

int32_t am_native_Diag_opstack(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!proc || !proc->opstack || !proc->opstack_top) return diag_push_uint(proc, 0);
    return diag_push_uint(proc, (uint64_t)(proc->opstack_top - proc->opstack));
}

int32_t am_native_Diag_fstack(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    // 函数调用栈按 {closure_handle, return_target} 成对入栈，故元素数 / 2 = 帧数
    if (!proc || !proc->fstack || !proc->fstack_top) return diag_push_uint(proc, 0);
    return diag_push_uint(proc, (uint64_t)(proc->fstack_top - proc->fstack) / 2);
}

int32_t am_native_Diag_pgc(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_uint(proc, proc ? (uint64_t)proc->gc_count : 0);
}

// 本进程堆中的存活对象数（把柄表有效条目数 O(1)）。
// 为什么需要它：System.memstat 是**池级**统计（解释器池由全部进程共享，
// vm_used / heap_used 是整池口径），fork 之后各进程看到的数字几乎一样；
// 要区分"每个进程自己占了多少"，只能数它自己那把柄表里的活对象。
int32_t am_native_Diag_heap(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    if (!proc || !proc->heap || !proc->heap->table) return diag_push_uint(proc, 0);
    return diag_push_uint(proc, (uint64_t)proc->heap->table->length);
}

// ---- 闭包帧统计 ------------------------------------------------------------

int32_t am_native_Diag_frames(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_uint(proc, proc ? (uint64_t)proc->stat_frame_closures : 0);
}

int32_t am_native_Diag_legacy(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_uint(proc, proc ? (uint64_t)proc->stat_legacy_closures : 0);
}

int32_t am_native_Diag_slots(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_uint(proc, proc ? (uint64_t)proc->stat_slots_total : 0);
}

int32_t am_native_Diag_freeslots(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_uint(proc, proc ? (uint64_t)proc->stat_free_slots_total : 0);
}

int32_t am_native_Diag_fillgeneric(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_uint(proc, proc ? (uint64_t)proc->stat_fill_generic : 0);
}

int32_t am_native_Diag_fastcl(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return diag_push_bool(proc, proc ? proc->closure_fast : false);
}

// ---- 注册表 ----------------------------------------------------------------

static const am_native_func_entry_t am_native_Diag_funcs[] = {
    { L"ms",           am_native_Diag_ms },
    { L"tick",         am_native_Diag_tick },
    { L"gcs",          am_native_Diag_gcs },
    { L"timeslice",    am_native_Diag_timeslice },
    { L"pids",         am_native_Diag_pids },
    { L"pc",           am_native_Diag_pc },
    { L"ilcode",       am_native_Diag_ilcode },
    { L"opstack",      am_native_Diag_opstack },
    { L"fstack",       am_native_Diag_fstack },
    { L"pgc",          am_native_Diag_pgc },
    { L"heap",         am_native_Diag_heap },
    { L"frames",       am_native_Diag_frames },
    { L"legacy",       am_native_Diag_legacy },
    { L"slots",        am_native_Diag_slots },
    { L"freeslots",    am_native_Diag_freeslots },
    { L"fillgeneric",  am_native_Diag_fillgeneric },
    { L"fastcl",       am_native_Diag_fastcl },
};

const am_native_lib_entry_t am_native_Diag_lib = {
    L"Diag",
    am_native_Diag_funcs,
    sizeof(am_native_Diag_funcs) / sizeof(am_native_Diag_funcs[0])
};