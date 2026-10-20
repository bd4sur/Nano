////////////////////////////////////////////////////////////////////////////
//  Native Library : Time
//  设计说明见 am_native_Time.h。
//
//  返回值一律用「无符号整数」（值域 27 位，约 1.34e8）：
//    · utc_days：自 1970-01-01 的 UTC 日数（当前约 2e4）
//    · utc_sod ：当日 UTC 秒数（0..86399）
//  两者都远小于 2^27，因此在 32/64 位宿主上都是精确整数。
////////////////////////////////////////////////////////////////////////////

#include <stdbool.h>
#include <stddef.h>

#include "am_native_Time.h"
#include "am_host.h"

// 取当前 UTC 毫秒时间戳（宿主实现：clock_gettime(CLOCK_REALTIME) / time()）
static uint64_t time_now_ms(void) {
    return am_current_timestamp_in_ms();
}

// 数值收尾：压入无符号整数并推进 PC（值型函数）
static int32_t time_push_uint(am_process_t *proc, uint64_t x) {
    if (am_process_push_operand(proc, am_make_value_of_uint((am_uint_t)x)) != 0) return -1;
    am_process_step(proc);
    return 0;
}

int32_t am_native_Time_utc_days(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    uint64_t ms = time_now_ms();
    return time_push_uint(proc, ms / 86400000ULL);
}

int32_t am_native_Time_utc_sod(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    uint64_t ms = time_now_ms();
    return time_push_uint(proc, (ms % 86400000ULL) / 1000ULL);
}

static const am_native_func_entry_t am_native_Time_funcs[] = {
    { L"utc_days", am_native_Time_utc_days },
    { L"utc_sod",  am_native_Time_utc_sod },
};

const am_native_lib_entry_t am_native_Time_lib = {
    L"Time",
    am_native_Time_funcs,
    sizeof(am_native_Time_funcs) / sizeof(am_native_Time_funcs[0])
};
