#include "platform.h"

#include <Arduino.h>
#include <esp32-hal-psram.h>
#include <Wire.h>

extern "C" {

void sleep_in_ms(uint32_t ms) {
    delay(ms);
}

// NOTE 返回的时间戳是32位的，存在2038问题！
uint32_t get_timestamp_in_ms() {
    (uint32_t)millis();
}

// 优雅关机
int32_t graceful_shutdown() {
    return 0;
}




}
