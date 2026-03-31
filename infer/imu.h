#ifndef __NANO_IMU_H__
#define __NANO_IMU_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "platform.h"

int imu_init();

int imu_reset();

int imu_calib();

int imu_read_angle(float *pitch, float *roll, float *yaw);

int imu_read_quaternion(float *q0, float *q1, float *q2, float *q3);

int imu_close();

#ifdef __cplusplus
}
#endif

#endif
