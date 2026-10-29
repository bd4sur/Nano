////////////////////////////////////////////////////////////////////////////
//  Native Library : Imu —— 设计说明见 am_native_Imu.h
//
//  HAL 契约（hal_imu.h）：imu_read_angle(&pitch, &roll, &yaw) 返回 0 成功、
//  负值表示无 IMU/读取失败；角度单位为度。注意参数顺序是（pitch, roll, yaw）。
//
//  重力分量换算（屏幕像素坐标系：x 向右、y 向下为正，可直接用于 Gfx 坐标）：
//      gravity_x = -9.8 * sin(roll°)
//      gravity_y = +9.8 * cos(roll°)
//  与 C 层 ui_app.c 的 STATE_FLIP（gravity_y = -9.8*cos(roll)）只差 y 符号——
//  那边是 flip.c 的物理坐标系（y 向上），这里是 Gfx 的像素坐标系（y 向下），
//  物理含义相同（都是"竖直立起时物体向屏幕下方掉落"）。
////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#include "am_native_Imu.h"
#include "hal_imu.h"

#define IMU_GRAVITY_ACCEL (9.8f)

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

// 读一次姿态。返回 true 表示读到有效数据（*pitch/*roll/*yaw 度为单位的欧拉角）。
static bool imu_sample(float *pitch, float *roll, float *yaw) {
    float p = 0.0f, r = 0.0f, y = 0.0f;
    if (imu_read_angle(&p, &r, &y) != 0) return false;
    // 防御非有限值（部分 HAL 在数据未就绪时可能写入 NaN）
    if (!isfinite(p) || !isfinite(r) || !isfinite(y)) return false;
    if (pitch) *pitch = p;
    if (roll) *roll = r;
    if (yaw) *yaw = y;
    return true;
}

// 屏幕像素坐标系下的重力分量（见文件头：y 向下为正）。
// 返回 true 表示姿态有效；false 时给出缺省重力 (0, +9.8)（屏幕上竖直向下）。
static bool imu_gravity(float *gx, float *gy) {
    float roll = 0.0f;
    if (!imu_sample(NULL, &roll, NULL)) {
        if (gx) *gx = 0.0f;
        if (gy) *gy = IMU_GRAVITY_ACCEL;
        return false;
    }
    float rad = roll / 180.0f * (float)M_PI;
    if (gx) *gx = -IMU_GRAVITY_ACCEL * sinf(rad);
    if (gy) *gy = IMU_GRAVITY_ACCEL * cosf(rad);
    return true;
}

// ---------------------------------------------------------------------------
// 值型收尾（契约 N1 + N6）：压 1 个值 + step 1 次。
// ---------------------------------------------------------------------------
static int32_t imu_push(am_process_t *proc, am_value_t v) {
    if (am_process_push_operand(proc, v) != 0) return -1;
    am_process_step(proc);
    return 0;
}

static int32_t imu_push_bool(am_process_t *proc, bool b) {
    return imu_push(proc, b ? AM_VALUE_TRUE : AM_VALUE_FALSE);
}

// 角度：有效则压浮点，无效压 #null（软失败值，与 Math 库 NaN→#null 同惯例）
static int32_t imu_push_angle(am_process_t *proc, bool ok, float deg) {
    return imu_push(proc, ok ? am_make_value_of_float((am_float_t)deg) : AM_VALUE_NULL);
}

static int32_t imu_push_float(am_process_t *proc, float f) {
    return imu_push(proc, am_make_value_of_float((am_float_t)f));
}

// ===============================================================================
// Native 函数实现
// ===============================================================================

int32_t am_native_Imu_is_available(am_runtime_t *rt, am_process_t *proc) {
    (void)rt;
    return imu_push_bool(proc, imu_sample(NULL, NULL, NULL));
}

int32_t am_native_Imu_roll(am_runtime_t *rt, am_process_t *proc) {
    float roll = 0.0f;
    bool ok;
    (void)rt;
    ok = imu_sample(NULL, &roll, NULL);
    return imu_push_angle(proc, ok, roll);
}

int32_t am_native_Imu_pitch(am_runtime_t *rt, am_process_t *proc) {
    float pitch = 0.0f;
    bool ok;
    (void)rt;
    ok = imu_sample(&pitch, NULL, NULL);
    return imu_push_angle(proc, ok, pitch);
}

int32_t am_native_Imu_yaw(am_runtime_t *rt, am_process_t *proc) {
    float yaw = 0.0f;
    bool ok;
    (void)rt;
    ok = imu_sample(NULL, NULL, &yaw);
    return imu_push_angle(proc, ok, yaw);
}

int32_t am_native_Imu_gravity_x(am_runtime_t *rt, am_process_t *proc) {
    float gx = 0.0f;
    (void)rt;
    imu_gravity(&gx, NULL);
    return imu_push_float(proc, gx);
}

int32_t am_native_Imu_gravity_y(am_runtime_t *rt, am_process_t *proc) {
    float gy = 0.0f;
    (void)rt;
    imu_gravity(NULL, &gy);
    return imu_push_float(proc, gy);
}

// ===============================================================================
// 库注册表
// ===============================================================================

static const am_native_func_entry_t am_native_Imu_funcs[] = {
    { L"is_available", am_native_Imu_is_available },
    { L"roll",         am_native_Imu_roll },
    { L"pitch",        am_native_Imu_pitch },
    { L"yaw",          am_native_Imu_yaw },
    { L"gravity_x",    am_native_Imu_gravity_x },
    { L"gravity_y",    am_native_Imu_gravity_y },
};

const am_native_lib_entry_t am_native_Imu_lib = {
    L"Imu",
    am_native_Imu_funcs,
    sizeof(am_native_Imu_funcs) / sizeof(am_native_Imu_funcs[0])
};