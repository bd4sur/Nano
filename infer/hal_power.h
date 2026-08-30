#ifndef __NANO_UPS_H__
#define __NANO_UPS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

int ups_init();

// 充电状态
int32_t read_ups_is_charging();
// 电压(mV)
int32_t read_ups_voltage();
// 电流(mV)
int32_t read_ups_current();
// 电池电量
int32_t read_ups_soc();

// 电源键（PMIC PEK）轮询：返回 0=无 / 1=短按 / 2=长按。
// 实现经 M5.BtnPWR 按键状态机（M5.update 内以 4ms 节流读取 PEK 寄存器维护）；
// 调用方无需节流（读取为锁存标志、零 I2C 开销），但必须在 M5.update 之后调用。
// 注意不可直接调 M5.Power.getKeyState()：读清式，会被 M5.update（默认 pmic_button=true）抢占。
// 长按约 4 秒 PMIC 会硬件强制断电（软件无法屏蔽），返回值 2 仅作上报。
int32_t power_key_poll(void);

#ifdef __cplusplus
}
#endif

#endif
