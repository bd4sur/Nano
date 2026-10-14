// hal_imu_wasm.c —— Nano-WASM-GUI 姿态（IMU）HAL（Emscripten）
//
// 姿态来源：页面 JS 的 DeviceMotionEvent.accelerationIncludingGravity（重力向量）
// 与 DeviceOrientationEvent[absolute].alpha（罗盘航向）+ screen.orientation.angle，
// 经 ccall 调用本文件导出的 wasm_gui_on_orientation 注入。
// 解算公式与 hal_imu_web_linux.c 同源（屏幕方向角旋转修正 R(θ)、atan2 坡度），
// 但 az（z 轴）与 alpha（旋转方向）按本形态实测浏览器约定做符号适配
//（2026-10-14 真机反馈：俯仰反向、方位反向；FLIP 只用 roll 不受影响）。

#include "platform.h"
#include "hal_imu.h"

#include <emscripten.h>

static int32_t s_has_acc = 0;
static float   s_ax = 0.0f, s_ay = 0.0f, s_az = 0.0f;
static float   s_yaw_deg = 0.0f;
static int32_t s_orient_deg = 0;
static int32_t s_has_yaw = 0;

#define NANO_WASM_ATT_LPF_ALPHA (0.18f) // 重力低通滤波系数

// 姿态事件入口（页面 JS ccall）：flags bit0=acc 有效 bit1=yaw 有效
EMSCRIPTEN_KEEPALIVE
void wasm_gui_on_orientation(int32_t flags, float ax, float ay, float az,
                             float yaw_deg, int32_t orient_deg) {
    if (flags & 0x01) {
        if (!s_has_acc) {
            s_ax = ax;
            s_ay = ay;
            s_az = az;
            s_has_acc = 1;
        }
        else {
            s_ax += NANO_WASM_ATT_LPF_ALPHA * (ax - s_ax);
            s_ay += NANO_WASM_ATT_LPF_ALPHA * (ay - s_ay);
            s_az += NANO_WASM_ATT_LPF_ALPHA * (az - s_az);
        }
    }
    if (flags & 0x02) {
        s_yaw_deg = yaw_deg;
        s_has_yaw = 1;
    }
    s_orient_deg = orient_deg;
}

int imu_init() {
    return 0;
}

int imu_reset() {
    s_ax = s_ay = s_az = 0.0f;
    s_yaw_deg = 0.0f;
    s_orient_deg = 0;
    s_has_acc = 0;
    s_has_yaw = 0;
    return 0;
}

int imu_calib() {
    return 0;
}

int imu_read_angle(float *pitch, float *roll, float *yaw) {
    if (pitch == NULL || roll == NULL || yaw == NULL) return -1;
    if (!s_has_acc) return -1; // 尚无姿态数据（无传感器/未授权/非安全上下文）

    float ax = s_ax, ay = s_ay, az = s_az;
    float yaw_d = s_yaw_deg;
    int32_t th = s_orient_deg;

    // 屏幕方向角修正（设备坐标 → 屏幕显示坐标）：标准旋转 R(θ)
    double rad = (double)th * M_PI / 180.0;
    float ax_s = (float)(ax * cos(rad) - ay * sin(rad));
    float ay_s = (float)(ax * sin(rad) + ay * cos(rad));

    // roll：屏幕平面内坡度（直立 0，右缘下沉为负）
    *roll = atan2f(ax_s, ay_s) * 180.0f / (float)M_PI;

    // pitch：屏幕法向仰角（平放朝天 +90，竖直朝前 0）
    // 适配（2026-10-14 真机反馈）：本 WASM 形态实测环境中 DeviceMotionEvent
    // .accelerationIncludingGravity 的 z 轴符号与 hal_imu_web_linux.c 验证环境相反
    //（屏幕朝上平放静止时 az≈-9.8），导致玲珑仪俯仰反向，此处取反适配。
    // FLIP（番茄表）只用 roll（由 ax/ay 决定），不受 az 符号影响——与真机观察一致。
    float n = sqrtf(ax * ax + ay * ay + az * az);
    if (n > 1.0e-3f) {
        float z = -az / n; // z 取反（见上方注释）
        if (z > 1.0f) z = 1.0f;
        if (z < -1.0f) z = -1.0f;
        *pitch = asinf(z) * 180.0f / (float)M_PI;
    }
    else {
        *pitch = 0.0f;
    }

    // yaw：屏幕上方指向的罗盘航向（0~360，北=0，顺时针）。
    // 适配（同日真机反馈：方位角反向）：W3C 规范中 alpha 是绕 z 轴的【逆时针】旋转角，
    // 与罗盘航向（顺时针，北=0）方向相反；验证环境（hal_imu_web_linux.c）的浏览器把
    // absolute alpha 直接实现为罗盘航向，而本环境按规范为逆时针——取镜像并保留屏幕
    // 方向角 θ 的顺时针修正（yaw = θ - alpha），真机验证方位随机身同向转动。
    *yaw = fmodf((float)th - yaw_d + 720.0f, 360.0f);

    return 0;
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
