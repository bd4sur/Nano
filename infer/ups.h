#ifndef __NANO_UPS_H__
#define __NANO_UPS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

int ups_init();
// 读取电压寄存器(mV)
int32_t read_ups_voltage();
// 读取电池容量寄存器
int32_t read_ups_soc();

#ifdef __cplusplus
}
#endif

#endif
