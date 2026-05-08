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

#ifdef __cplusplus
}
#endif

#endif
