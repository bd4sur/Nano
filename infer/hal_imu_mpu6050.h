#ifndef __NANO_MPU6050_H__
#define __NANO_MPU6050_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

int mpu6050_init();

int mpu6050_read_angle(float *pitch, float *roll, float *yaw, float *temp);

int mpu6050_close();

#ifdef __cplusplus
}
#endif

#endif
