#include "platform.h"
#include "hal_imu.h"

// Linux IMU HAL打桩实现：本平台无IMU硬件，仅用于满足链接。
// 所有接口返回失败/空操作，上层应将其视为无IMU可用。

int imu_init() {
    return -1;  // 无IMU硬件
}

int imu_reset() {
    return -1;
}

int imu_calib() {
    return -1;
}

int imu_read_angle(float *pitch, float *roll, float *yaw) {
    (void)pitch;
    (void)roll;
    (void)yaw;
    return -1;
}

int imu_read_quaternion(float *q0, float *q1, float *q2, float *q3) {
    (void)q0;
    (void)q1;
    (void)q2;
    (void)q3;
    return -1;
}

int imu_close() {
    return 0;
}
