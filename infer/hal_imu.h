#ifndef __NANO_IMU_H__
#define __NANO_IMU_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

int imu_init();

int imu_reset();

int imu_calib();

int imu_read_angle(float *pitch, float *roll, float *yaw);

// 三轴加速度（含重力），单位 **g**（1g ≈ 9.8 m/s²），设备本体坐标系。
// 返回 0 成功、负值表示本平台不提供该通道（无 IMU、或该 HAL 只实现了欧拉角）。
//
// 为什么单独一个入口（2026-11，计步器 Scheme App 需要）：
//   imu_read_angle 的语义在各平台**不一致**——M5Core2/CoreS3 的 HAL 实际把加速度计的
//   x/y/z（g）写进 pitch/roll/yaw（见 hal_imu_m5esp.cpp），而 Linux/web/wasm 的 HAL
//   写的是真正的欧拉角（度）。计步器需要的是「加速度模值随步伐起伏」，只有前者能提供，
//   后者从角度里无法还原。故新增本入口，由能提供加速度的 HAL 实现，其余返回负值。
//
// 已实现：hal_imu_m5esp.cpp（M5Unified accel，g）、hal_imu_wasm.c（浏览器
// DeviceMotion.accelerationIncludingGravity，m/s²→g）、hal_imu_web_linux.c（网页上报的
// 同一向量）、Android（android/app/src/main/cpp/hal_imu_android.c，TYPE_ACCELEROMETER）。
// 串口 IMU（imu10/imu6）与纯桩平台返回 -1，调用方应软失败降级。
int imu_read_accel(float *ax, float *ay, float *az);

int imu_read_quaternion(float *q0, float *q1, float *q2, float *q3);

int imu_close();

#ifdef __cplusplus
}
#endif

#endif
