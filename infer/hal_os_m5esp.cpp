#include "platform.h"
#include "hal_os.h"

#include <sys/time.h>
#include <Arduino.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <M5Unified.h>

// ---------------- 延时 / 时间戳 ----------------

void sleep_in_ms(uint32_t ms) {
    delay(ms);
}

uint64_t get_timestamp_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
}

// 优雅关机：通过 PMIC（Core2: AXP192 / CoreS3: AXP2101，M5.Power 已封装板型差异）切断整机全部电源。
// 成功时本函数不返回（设备已断电）；若仍然返回，说明断电未生效（如 USB 供电时），
// 返回 -1 由调用方提示关机失败。
int32_t graceful_shutdown() {
    sleep_in_ms(500); // 让“正在安全关机”提示在屏幕上停留片刻
    M5.Power.powerOff(); // 切断所有电源输出
    sleep_in_ms(1000); // 等待断电生效；仍在运行则判定失败
    return -1;
}

// 设置 RTC 时间
void set_sys_time(
    int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second
) {
    M5.Rtc.setDateTime( { { year, month, day }, { hour, minute, second } } );
}

// ---------------- 任务抽象（FreeRTOS 实现） ----------------

int32_t platform_task_create(platform_task_func_t func, const char *name,
                             uint32_t stack_bytes, void *arg, int32_t priority,
                             int32_t core, platform_task_handle_t *out_handle) {
    TaskHandle_t handle = NULL;
    BaseType_t ok;
    if (core >= 0) {
        ok = xTaskCreatePinnedToCore(func, name, stack_bytes, arg,
                                     (UBaseType_t)priority, &handle, (BaseType_t)core);
    }
    else {
        ok = xTaskCreate(func, name, stack_bytes, arg,
                         (UBaseType_t)priority, &handle);
    }
    if (ok != pdPASS) return -1;
    if (out_handle) *out_handle = (platform_task_handle_t)handle;
    return 0;
}

void platform_task_delete_self(void) {
    vTaskDelete(NULL);
}

void platform_task_delete(platform_task_handle_t handle) {
    if (handle) vTaskDelete((TaskHandle_t)handle);
}

void platform_task_delay_ms(uint32_t ms) {
    vTaskDelay(pdMS_TO_TICKS(ms));
}

// 振动马达已移至 hal_misc（hal_misc_m5esp.cpp）
