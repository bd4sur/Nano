#include <Arduino.h>
#include <esp32-hal-psram.h>
#include "M5Unified.h"

#include "ups.h"

int ups_init() {
    return 0;
}

int32_t read_ups_is_charging() {
    bool isCharging = M5.Power.isCharging();
    if (isCharging) {
        M5.Power.setLed(255);
        return 1;
    } else {
        M5.Power.setLed(0);
        return 0;
    }
}

// 电压(mV)
int32_t read_ups_voltage() {
    return M5.Power.getBatteryVoltage();
}

// 电流(mA)
int32_t read_ups_current() {
    return M5.Power.getBatteryCurrent();
}

// 电量
int32_t read_ups_soc() {
    return M5.Power.getBatteryLevel();
}
