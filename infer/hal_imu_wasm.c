// hal_imu_wasm.c —— Nano-WASM-GUI 姿态（IMU）HAL（Emscripten）
//
// 姿态来源：页面 JS 的 DeviceMotionEvent.accelerationIncludingGravity（重力向量）
// 与 DeviceOrientationEvent[absolute].alpha（罗盘航向）+ screen.orientation.angle，
// 经 ccall 调用本文件导出的 wasm_gui_on_orientation 注入。
// 角度语义与解算公式和 hal_imu_web_linux.c 完全一致（同一套屏幕方向角旋转修正、
// atan2 坡度/asin 仰角；浏览器安全上下文限制同样适用：localhost/HTTPS 页面可用，
// 普通局域网 HTTP 页面事件不触发，此时 imu_read_angle 返回失败、业务保持默认）。

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
    float n = sqrtf(ax * ax + ay * ay + az * az);
    if (n > 1.0e-3f) {
        float z = az / n;
        if (z > 1.0f) z = 1.0f;
        if (z < -1.0f) z = -1.0f;
        *pitch = asinf(z) * 180.0f / (float)M_PI;
    }
    else {
        *pitch = 0.0f;
    }

    // yaw：屏幕上方指向的罗盘航向
    *yaw = fmodf(yaw_d + (float)th + 360.0f, 360.0f);

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
