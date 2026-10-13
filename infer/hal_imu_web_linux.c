// hal_imu_web_linux.c —— Nano-Web 姿态（IMU）HAL
//
// 姿态来源：网页客户端浏览器的 DeviceMotionEvent.accelerationIncludingGravity
// （重力向量，精确稳定，无需校准）与 DeviceOrientationEvent.alpha（罗盘航向），
// 经 WebSocket 上报（nano_ws_server 客户端接收线程调用本文件 web_remote_on_orientation）。
//
// 角度语义对齐 Linux 实体 IMU 路径（hal_imu_imu10_linux.c 输出约定，供 ui_app.c
// 非 ESP32 分支消费）：
//   roll  —— 屏幕平面内的坡度（重力方向角）：屏幕水平持机为 0，右边缘下沉为负（度）。
//             番茄表（FLIP）按 gravity_x=-9.8·sin(roll°)、gravity_y=-9.8·cos(roll°) 换算重力。
//   pitch —— 屏幕法向（视线）仰角：屏幕竖直朝前为 0，平放朝天为 +90，朝下为 -90（度）。
//             玲珑仪以 view_alt=pitch 作视线仰角（enable_imu 时）。
//   yaw   —— 屏幕上方指向的罗盘航向（0~360，北=0；浏览器 alpha 经屏幕方向角修正）。
//
// 屏幕方向角 θ（screen.orientation.angle，横屏时 ±90°）修正为标准旋转 R(θ)：
//   ax_s = ax·cosθ - ay·sinθ； ay_s = ax·sinθ + ay·cosθ
// （θ=0 竖屏：恒等；θ=90 横屏（设备左缘朝下）：屏幕上=设备 +x_d、屏幕右=-y_d，
//   即 ax_s=-ay_d、ay_s=ax_d；θ=180/270 类推。
//   验证：任意持机方向下「屏幕右缘下沉」均得 roll<0，与竖屏右倾一致；
//   水平持机（roll=0）与倒置（roll=±180）各方向自洽。）

#include "platform.h"
#include "hal_imu.h"
#include "nano_ws_server.h" // web_remote_on_orientation 由本文件实现（ws 服务器接收线程回调）

#include <pthread.h>

// 姿态原始数据缓存（ws 接收线程写，imu_read_angle 读）
static pthread_mutex_t s_att_lock = PTHREAD_MUTEX_INITIALIZER;
static int32_t s_has_acc = 0;          // 1-收到过加速度数据（此前 imu_read_angle 返回失败，对齐无 IMU 语义）
static float   s_ax = 0.0f, s_ay = 0.0f, s_az = 0.0f; // 低通滤波后的重力向量（设备坐标系，m/s²）
static float   s_yaw_deg = 0.0f;       // 浏览器罗盘航向 alpha（度）
static int32_t s_orient_deg = 0;       // 屏幕方向角 θ（0/90/180/270）
static int32_t s_has_yaw = 0;          // 1-浏览器提供了有效航向

// 重力低通滤波系数（浏览器上报约 20Hz；对齐实体 IMU 的平滑效果，抑制抖动）
#define NANO_WEB_ATT_LPF_ALPHA (0.18f)

// 姿态事件入口（nano_ws_server 客户端接收线程调用）：
// acc 为设备坐标系重力向量（m/s²，可为 NULL 表示本帧无加速度数据）；
// yaw_deg 为浏览器 alpha（度）；orient_deg 为 screen.orientation.angle；flags 见协议。
void web_remote_on_orientation(const float *acc, float yaw_deg, int32_t orient_deg, uint8_t flags) {
    pthread_mutex_lock(&s_att_lock);
    if ((flags & 0x01) && acc != NULL) {
        if (!s_has_acc) { // 首次直接采用，避免从 0 缓慢爬升
            s_ax = acc[0];
            s_ay = acc[1];
            s_az = acc[2];
            s_has_acc = 1;
        }
        else {
            s_ax += NANO_WEB_ATT_LPF_ALPHA * (acc[0] - s_ax);
            s_ay += NANO_WEB_ATT_LPF_ALPHA * (acc[1] - s_ay);
            s_az += NANO_WEB_ATT_LPF_ALPHA * (acc[2] - s_az);
        }
    }
    if (flags & 0x02) {
        s_yaw_deg = yaw_deg;
        s_has_yaw = 1;
    }
    s_orient_deg = orient_deg;
    pthread_mutex_unlock(&s_att_lock);
}

int imu_init() {
    return 0; // 无硬件可初始化；客户端上报开始后即可读数
}

int imu_reset() {
    pthread_mutex_lock(&s_att_lock);
    s_ax = s_ay = s_az = 0.0f;
    s_yaw_deg = 0.0f;
    s_orient_deg = 0;
    s_has_acc = 0;
    s_has_yaw = 0;
    pthread_mutex_unlock(&s_att_lock);
    return 0;
}

int imu_calib() {
    return 0; // 重力向量/罗盘由浏览器给出，无需陀螺零偏校准
}

int imu_read_angle(float *pitch, float *roll, float *yaw) {
    if (pitch == NULL || roll == NULL || yaw == NULL) return -1;

    pthread_mutex_lock(&s_att_lock);
    if (!s_has_acc) { // 尚无姿态数据：返回失败（调用方保持旧值/默认重力，对齐 hal_imu_linux 桩语义）
        pthread_mutex_unlock(&s_att_lock);
        return -1;
    }
    float ax = s_ax, ay = s_ay, az = s_az;
    float yaw_d = s_yaw_deg;
    int32_t th = s_orient_deg;
    pthread_mutex_unlock(&s_att_lock);

    // 屏幕方向角修正（浏览器设备坐标 → 屏幕显示坐标）：标准旋转 R(θ)
    double rad = (double)th * M_PI / 180.0;
    float ax_s = (float)(ax * cos(rad) - ay * sin(rad));
    float ay_s = (float)(ax * sin(rad) + ay * cos(rad));

    // roll：屏幕平面内坡度（直立 0，右缘下沉为负）。静止时加速度计读数为重力反向，
    // 故 ax_s/ay_s 直接给出「重力反向」的屏幕分量，atan2 得坡度角。
    *roll = atan2f(ax_s, ay_s) * 180.0f / (float)M_PI;

    // pitch：屏幕法向仰角（法向不受横竖屏修正影响）
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

    // yaw：屏幕上方指向的罗盘航向（浏览器 alpha + 屏幕方向角修正，无罗盘数据时保持旧值/0）
    *yaw = fmodf(yaw_d + (float)th + 360.0f, 360.0f);

    return 0;
}

int imu_read_quaternion(float *q0, float *q1, float *q2, float *q3) {
    (void)q0;
    (void)q1;
    (void)q2;
    (void)q3;
    return -1; // 浏览器不提供四元数，调用方一律用 imu_read_angle（Linux 路径本就如此）
}

int imu_close() {
    return 0;
}
