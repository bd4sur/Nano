#include "imu.h"
#include "platform.h"

#include <esp32-hal-psram.h>
#include <M5Unified.h>
#include <math.h>

// 配置参数（根据实际硬件和场景调整）
#define IMU_SAMPLE_RATE     100.0f      // 采样频率 Hz
#define IMU_DT              (1.0f / IMU_SAMPLE_RATE)  // 采样周期
#define ALPHA               0.98f       // 互补滤波系数（陀螺仪权重，0.98表示更信任陀螺仪）

// 全局静态变量保存角度状态（替代卡尔曼滤波的简易实现）
static float g_pitch = 0.0f;  // 俯仰角（绕Y轴）
static float g_roll  = 0.0f;  // 横滚角（绕X轴）
static float g_yaw   = 0.0f;  // 偏航角（绕Z轴）

// 陀螺仪零偏校准值（建议在初始化时校准）
static float g_gyro_offset_x = 0.0f;
static float g_gyro_offset_y = 0.0f;
static float g_gyro_offset_z = 0.0f;


int imu_init() {
    g_pitch = 0.0f;
    g_roll  = 0.0f;
    g_yaw   = 0.0f;
    return 0;
}

int imu_reset(){
    g_pitch = 0.0f;
    g_roll  = 0.0f;
    g_yaw   = 0.0f;
    return 0;
}

int imu_calib(){
    int samples = 100;

    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    
    for (int i = 0; i < samples; i++) {
        auto ImuData = M5.Imu.getImuData();
        sum_x += ImuData.gyro.x;
        sum_y += ImuData.gyro.y;
        sum_z += ImuData.gyro.z;
        delay(10);  // 100Hz采样间隔
    }
    
    g_gyro_offset_x = sum_x / samples;
    g_gyro_offset_y = sum_y / samples;
    g_gyro_offset_z = sum_z / samples;
    
    // 校准后重置角度
    g_pitch = 0.0f;
    g_roll  = 0.0f;
    g_yaw   = 0.0f;
    return 0;
}

int imu_read_angle(float *pitch, float *roll, float *yaw){
    // 1. 参数校验
    if (pitch == NULL || roll == NULL || yaw == NULL) {
        return -1;
    }

    auto imu_update = M5.Imu.update();

    if (imu_update) {

        // 2. 获取IMU原始数据（M5Stack特定API）
        auto ImuData = M5.Imu.getImuData();

        // 3. 提取加速度计数据（单位：g）
        *pitch = ImuData.accel.x;
        *roll = ImuData.accel.y;
        *yaw = ImuData.accel.z;

        // printf("x=%-10.2f    y=%-10.2f    z=%-10.2f\n", ImuData.accel.x, ImuData.accel.y, ImuData.accel.z);

        return 0;
    }
    else {
        return -1;
    }
}

int imu_read_quaternion(float *q0, float *q1, float *q2, float *q3){
    return 0;
}

int imu_close(){
    return 0;
}
