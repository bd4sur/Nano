#include <Arduino.h>
#include <esp32-hal-psram.h>
#include "M5Unified.h"

#include "hal_power.h"

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

// 电源键（PMIC PEK）轮询。
// 不可直接调 M5.Power.getKeyState()：该寄存器为读清式，而 M5.config() 默认 pmic_button=true，
// M5.update()（Core1 每 ~1ms）内部以 4ms 节流（BTNPWR_MIN_UPDATE_MSEC）抢占读取同一寄存器，
// 旁路轮询几乎必然读到 0——2026-08 实测锁屏键大概率无响应、偶尔成功（恰抢在 M5.update 前时）。
// 故改经 M5.BtnPWR 判定（M5.update 内维护的按键状态机；wasClicked 为锁存标志，读取零 I2C 开销）。
// M5.BtnPWR 短按=clicked（对应 getKeyState()==2）；长按约 4 秒 PMIC 硬件强制断电，软件无法屏蔽。
int32_t power_key_poll(void) {
    if (M5.BtnPWR.wasClicked()) return 1; // 短按
    if (M5.BtnPWR.wasHold())    return 2; // 长按（仅上报，未消费）
    return 0;
}
