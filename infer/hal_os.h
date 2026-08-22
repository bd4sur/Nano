#ifndef __NANO_HAL_OS_H__
#define __NANO_HAL_OS_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===============================================================================
// hal_os：系统/OS 抽象层（硬件无关接口；ESP32 实现见 hal_os_m5esp.cpp）
//   延时/时间戳/优雅关机/RTC 时间设置/全局主音量/任务创建管理；
//   任务抽象在 ESP32 上由 FreeRTOS 实现，其他平台可用 pthread 等实现。
// ===============================================================================

// 阻塞延时（毫秒）
void sleep_in_ms(uint32_t ms);

// 单调墙钟时间戳（毫秒，UTC 纪元起）
uint64_t get_timestamp_in_ms();

// 优雅关机：经电源管理芯片切断整机电源。成功时不返回；仍返回说明断电未生效，返回 -1
int32_t graceful_shutdown();

// 设置 RTC 时间
void set_sys_time(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second);

// ---------------- 任务抽象（ESP32 上由 FreeRTOS 实现；其他平台可用 pthread 等实现） ----------------
// 注意：句柄为不透明 void*，业务代码不得依赖其具体类型；
//       任务入口函数返回前必须调用 platform_task_delete_self() 自删（不返回）。

typedef void* platform_task_handle_t;
typedef void (*platform_task_func_t)(void *arg);

// 创建任务。stack_bytes 为栈字节数；priority 数值越大优先级越高；
// core >= 0 时绑定到指定核，core < 0 时不绑核。返回 0 成功，负数失败。
int32_t platform_task_create(platform_task_func_t func, const char *name,
                             uint32_t stack_bytes, void *arg, int32_t priority,
                             int32_t core, platform_task_handle_t *out_handle);
// 任务内自删除（不返回）
void platform_task_delete_self(void);
// 按句柄强制删除任务（用于清理兜底）
void platform_task_delete(platform_task_handle_t handle);
// 任务内延时（毫秒）
void platform_task_delay_ms(uint32_t ms);

// 全内存屏障（跨核 SPSC 无锁同步用：保证屏障前的写先于屏障后的写对外可见）
#if defined(__GNUC__)
    #define platform_memory_barrier() __sync_synchronize()
#else
    #define platform_memory_barrier()  // 其他平台按需实现
#endif

#ifdef __cplusplus
}
#endif

#endif
