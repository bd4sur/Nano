#include "platform.h"

#include <Arduino.h>
#include <esp32-hal-psram.h>
#include <Wire.h>

extern "C" {

void sleep_in_ms(uint32_t ms) {
    delay(ms);
}

uint64_t get_timestamp_in_ms() {
    return (uint64_t)millis();
}

// 优雅关机
int32_t graceful_shutdown() {
    return 0;
}




}
