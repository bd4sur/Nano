////////////////////////////////////////////////////////////////////////////
//  Native Library : Imu —— 灵机（Animac）姿态/重力本地库
//
//  为什么需要本库：
//    "灵机应用程序"（Scheme App）此前看不到机器姿态。项目内需要重力感的 App
//    （番茄表/流体等，见 ui_app.c 的 STATE_FLIP）都在 C 层用 IMU 的 roll 换算
//    重力分量；Scheme 侧要做到同样的事情（小球滚动、液体倾斜、水平仪……），
//    就必须有姿态通道。本库把 HAL 姿态（hal_imu.h 的 imu_read_angle）以只读方式
//    暴露给 Scheme，并直接给出与 C 层一致的【屏幕平面内重力分量】。
//
//  角度约定（HAL 的欧拉角，各平台一致，见 hal_imu_imu10_linux.c / hal_imu_wasm.c）：
//    · roll  ：屏幕平面内滚转角（度），水平 0，右缘下沉为负；
//    · pitch ：屏幕法向仰角（度），竖直朝前 0，平放朝天 +90；
//    · yaw   ：屏幕上方指向的罗盘航向（度，0~360，北=0，顺时针）。
//
//  重力约定：本库给出的是【屏幕像素坐标系（x 向右、y 向下为正）】下的重力分量，
//  可直接加进 Gfx 坐标里的速度：
//    · gravity_x = -9.8·sin(roll°)
//    · gravity_y = +9.8·cos(roll°)
//  直立（roll=0）时 (0, +9.8) —— 屏幕上"向下"，与直觉一致；右缘下沉（roll<0）时
//  gravity_x>0 —— 物体向屏幕右侧滑落。这与 C 层 FLIP（ui_app.c → flip.c）使用的
//  物理坐标系（y 向上为正，故 gravity_y=-9.8·cos(roll)）**y 分量符号相反**，原因只是
//  两套坐标系的 y 朝向不同；模块内 app/flip.scm 同样按"y 向下"使用重力，与此一致。
//  z 轴分量（进出屏幕方向）不提供：2D 绘制/物理用不到，且 HAL 只给欧拉角。
//
//  软失败约定（manual §1.1.1）：宿主无 IMU（imu_read_angle 返回负值，如 Linux 打桩
//  hal_imu_linux.c、未授权的浏览器姿态）时——
//    · (Imu.is_available) 返回 #f；
//    · (Imu.roll)/(Imu.pitch)/(Imu.yaw) 返回 #null（App 可用 number? 判定）；
//    · (Imu.gravity_x)/(Imu.gravity_y) 返回缺省重力 (0, +9.8)（屏幕上向下）——
//      让"无传感器的设备也能跑通物理"，与 app/flip.scm 里"重力缺省竖直向下"同语义。
//
//  Scheme 侧用法：源码中声明 (native Imu) 后以 Imu.函数名 调用。
//
//  栈纪律（上游本地库契约 N1/N6）：所有函数零参数、值型——压 1 个值并 step 一次。
////////////////////////////////////////////////////////////////////////////

#ifndef __AM_NATIVE_IMU_H__
#define __AM_NATIVE_IMU_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

// 由 infer/am_native_Imu.c 定义
extern const am_native_lib_entry_t am_native_Imu_lib;

// (Imu.is_available) : Boolean —— 宿主是否有可用姿态数据
int32_t am_native_Imu_is_available(am_runtime_t *rt, am_process_t *proc);

// (Imu.roll) : Number | #null —— 屏幕平面内滚转角（度）
int32_t am_native_Imu_roll(am_runtime_t *rt, am_process_t *proc);

// (Imu.pitch) : Number | #null —— 屏幕法向仰角（度）
int32_t am_native_Imu_pitch(am_runtime_t *rt, am_process_t *proc);

// (Imu.yaw) : Number | #null —— 罗盘航向（度，0~360）
int32_t am_native_Imu_yaw(am_runtime_t *rt, am_process_t *proc);

// (Imu.gravity_x) : Number —— 屏幕坐标系（y 向下）重力 x 分量，m/s²；无姿态数据时 0
int32_t am_native_Imu_gravity_x(am_runtime_t *rt, am_process_t *proc);

// (Imu.gravity_y) : Number —— 屏幕坐标系（y 向下）重力 y 分量，m/s²；无姿态数据时 +9.8
int32_t am_native_Imu_gravity_y(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif