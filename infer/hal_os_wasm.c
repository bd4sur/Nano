// hal_os_wasm.c —— Nano-WASM-GUI OS/任务抽象（Emscripten）
//
// 时间戳/延时走 emscripten 的 libc（clock_gettime/usleep 可用）；
// 任务抽象全部打桩（单线程构建，无 pthread）：platform_task_create 恒失败——
// 各调用点（ui_ofdm 采集任务等）均有失败处理，优雅降级。

#include "platform.h"
#include "hal_os.h"

#include <unistd.h>
#include <time.h>
#include <stdlib.h>

// ---------------- 延时 / 时间戳 ----------------

void sleep_in_ms(uint32_t ms) {
    usleep(ms * 1000); // emscripten 下为忙等（仅关机/错误画面停留等短路径使用）
}

// 浏览器内「设置系统时间」的等价实现：
//   浏览器不允许网页修改操作系统时钟，故用一个相对宿主时钟的偏移量模拟。
//   get_timestamp_in_ms() 返回「宿主时钟 + 偏移」，于是主事件循环用 localtime()
//   推导出的年月日时分秒、天文计算、自动关机倒计时都统一跟随用户在设置界面
//   编辑后的时间；刷新页面/重新加载后偏移归零（与浏览器无持久时钟权限一致）。
static int64_t s_time_offset_ms = 0;

// 供输入事件时间戳（页面传入的宿主 epoch ms）换算到应用时间轴：
// 设置时间后，事件时间与 get_timestamp_in_ms() 必须落在同一条时间轴上，
// 否则事件帧会用宿主时间覆盖 global_state->timestamp，导致界面瞬时跳回旧日期。
int64_t wasm_time_offset_ms(void) {
    return s_time_offset_ms;
}

static uint64_t wasm_host_time_ms(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return (uint64_t)time.tv_sec * 1000ULL + (uint64_t)time.tv_nsec / 1000000ULL;
}

uint64_t get_timestamp_in_ms() {
    int64_t t = (int64_t)wasm_host_time_ms() + s_time_offset_ms;
    if (t < 0) t = 0;
    return (uint64_t)t;
}

// 公历日期 → 1970-01-01 起的天数（Howard Hinnant days_from_civil）
static int64_t wasm_days_from_civil(int32_t y, uint32_t m, uint32_t d) {
    y -= (m <= 2) ? 1 : 0;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const uint32_t yoe = (uint32_t)(y - (int32_t)(era * 400));
    const int32_t mp = (int32_t)m + ((m > 2) ? -3 : 9);
    const uint32_t doy = (uint32_t)((153 * mp + 2) / 5 + (int32_t)d - 1);
    const uint32_t doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    return era * 146097 + (int64_t)doe - 719468;
}

// 优雅关机（浏览器中无意义：直接失败，调用方按「关机失败」处理并取消自动关机）
int32_t graceful_shutdown() {
    return -1;
}

// 设置系统时间：记录「目标时刻 − 宿主当前时刻」的偏移（参数为 UTC 年月日时分秒）
void set_sys_time(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second) {
    const int64_t days = wasm_days_from_civil(year, (uint32_t)month, (uint32_t)day);
    const int64_t target_ms = (days * 86400LL + (int64_t)hour * 3600LL +
                               (int64_t)minute * 60LL + (int64_t)second) * 1000LL;
    s_time_offset_ms = target_ms - (int64_t)wasm_host_time_ms();
}

// ---------------- 任务抽象（单线程打桩） ----------------

int32_t platform_task_create(platform_task_func_t func, const char *name,
                             uint32_t stack_bytes, void *arg, int32_t priority,
                             int32_t core, platform_task_handle_t *out_handle) {
    (void)func; (void)name; (void)stack_bytes; (void)arg; (void)priority; (void)core;
    if (out_handle) *out_handle = NULL;
    return -1; // 单线程构建：任务创建恒失败（调用方降级）
}

void platform_task_delete_self(void) {
    return;
}

void platform_task_delete(platform_task_handle_t handle) {
    (void)handle;
}

void platform_task_delay_ms(uint32_t ms) {
    usleep(ms * 1000);
}
