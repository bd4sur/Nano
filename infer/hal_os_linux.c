// hal_os Linux 实现（PC / 树莓派等普通 Linux 系统）。
//
// 依赖：
//   - pthread（任务抽象基于 pthread 实现，编译/链接需加 -pthread）
//   - 其余均为 glibc / POSIX 标准接口，无额外第三方依赖。

#define _GNU_SOURCE // pthread_setaffinity_np / CPU_ZERO / CPU_SET / timegm

#include "platform.h"
#include "hal_os.h"

#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h> // PTHREAD_STACK_MIN
#include <pthread.h>

#ifdef __linux__
#include <sched.h> // CPU_ZERO / CPU_SET / pthread_setaffinity_np
#endif

// ---------------- 延时 / 时间戳 ----------------

void sleep_in_ms(uint32_t ms) {
    usleep(ms * 1000);
}

uint64_t get_timestamp_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return (uint64_t)time.tv_sec * 1000ULL + (uint64_t)time.tv_nsec / 1000000ULL;
}

// 优雅关机
int32_t graceful_shutdown() {
    // 同步所有文件系统数据
    sync();
    // 等待同步完成
    sleep(2);
    // 执行关机
    if (system("poweroff") == -1) {
        perror("关机失败");
        return -1;
    }
    return 0;
}

// 设置系统时间（入参为 UTC；对齐 ESP32 M5.Rtc.setDateTime 语义）。
// 需要 root 或 CAP_SYS_TIME 权限，非特权用户调用时静默失败。
void set_sys_time(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second) {
    struct tm t;
    memset(&t, 0, sizeof(t));
    t.tm_year = year - 1900;
    t.tm_mon  = month - 1;
    t.tm_mday = day;
    t.tm_hour = hour;
    t.tm_min  = minute;
    t.tm_sec  = second;

    time_t sec = timegm(&t); // 按 UTC 解释
    if (sec == (time_t)-1) return;

    struct timespec ts;
    ts.tv_sec  = sec;
    ts.tv_nsec = 0;
    (void)clock_settime(CLOCK_REALTIME, &ts); // EPERM 时忽略
}

// ---------------- 任务抽象（pthread 实现，对应 ESP32 的 FreeRTOS） ----------------
//
// 说明：
// - 任务以 DETACHED 状态创建，入口函数返回即自动回收（业务代码约定入口
//   返回前调用 platform_task_delete_self，见 ui_ofdm.c）；
// - stack_bytes 语义与 xTaskCreate 一致（字节数），小于 PTHREAD_STACK_MIN
//   时提升到下限；
// - core >= 0 时尝试绑核（pthread_setaffinity_np），失败不视为错误；
// - priority 在 Linux 上无法在无特权情况下映射为实时调度优先级，故忽略；
// - 句柄直接承载 pthread_t 值（glibc 下为指针宽度整数）。

typedef struct {
    platform_task_func_t func;
    void *arg;
} Platform_Task_Bootstrap;

static void *platform_task_trampoline(void *p) {
    Platform_Task_Bootstrap *bootstrap = (Platform_Task_Bootstrap *)p;
    platform_task_func_t func = bootstrap->func;
    void *arg = bootstrap->arg;
    free(bootstrap);
    func(arg); // 业务约定：函数末尾调用 platform_task_delete_self()，不会返回
    return NULL;
}

int32_t platform_task_create(platform_task_func_t func, const char *name,
                             uint32_t stack_bytes, void *arg, int32_t priority,
                             int32_t core, platform_task_handle_t *out_handle) {
    (void)name;     // pthread 无需任务名
    (void)priority; // 无特权下无法设置实时优先级，忽略
    if (!func) return -1;

    Platform_Task_Bootstrap *bootstrap =
        (Platform_Task_Bootstrap *)calloc(1, sizeof(Platform_Task_Bootstrap));
    if (!bootstrap) return -1;
    bootstrap->func = func;
    bootstrap->arg  = arg;

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    if (stack_bytes >= (uint32_t)PTHREAD_STACK_MIN) {
        pthread_attr_setstacksize(&attr, (size_t)stack_bytes);
    }

    pthread_t tid;
    int err = pthread_create(&tid, &attr, platform_task_trampoline, bootstrap);
    pthread_attr_destroy(&attr);
    if (err != 0) {
        free(bootstrap);
        return -1;
    }

#ifdef __linux__
    if (core >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        (void)pthread_setaffinity_np(tid, sizeof(cpuset), &cpuset); // 失败忽略
    }
#else
    (void)core;
#endif

    if (out_handle) {
        *out_handle = (platform_task_handle_t)tid;
    }
    return 0;
}

// 任务入口返回前必须调用（不返回）
void platform_task_delete_self(void) {
    pthread_exit(NULL);
}

// 强制删除任务（清理兜底；对应 vTaskDelete）
void platform_task_delete(platform_task_handle_t handle) {
    if (!handle) return;
    pthread_cancel((pthread_t)handle);
}

void platform_task_delay_ms(uint32_t ms) {
    usleep(ms * 1000);
}
