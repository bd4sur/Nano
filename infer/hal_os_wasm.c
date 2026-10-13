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

uint64_t get_timestamp_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return (uint64_t)time.tv_sec * 1000ULL + (uint64_t)time.tv_nsec / 1000000ULL;
}

// 优雅关机（浏览器中无意义：直接失败，调用方按「关机失败」处理并取消自动关机）
int32_t graceful_shutdown() {
    return -1;
}

// 设置系统时间（浏览器中无权限/无意义，空操作）
void set_sys_time(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second) {
    (void)year; (void)month; (void)day; (void)hour; (void)minute; (void)second;
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
