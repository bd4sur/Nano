////////////////////////////////////////////////////////////////////////////
//  Native Library : Touch —— 灵机（Animac）触屏本地库
//
//  为什么需要本库：
//    GUI 层的触屏消费走 ui_app.c 的 get_input_event（电平 + 边沿 + 宫格软按键），
//    而"灵机应用程序"（Scheme App）需要【自己的一段触屏输入】——画笔、拖拽、
//    虚拟摇杆等。此前 Scheme 侧只能看 Gfx（画布）与 Time/Diag（时钟/自观测），
//    完全没有输入通道，交互式 App 无法编写。本库把 HAL 触屏（hal_touch.h）以
//    只读方式暴露给 Scheme，使 App 可以逐帧取触点，并在 Scheme 内自行解释手势。
//
//  设计要点：
//    1) 只读、旁路：不消费、不改变 HAL 状态，宿主 GUI 的输入管线完全不受影响
//       （与 ui_app.c 各读各的同一份电平，互不干扰）。
//    2) 电平模型：与 get_input_event 同一哲学——每次调用采样一次 HAL 电平，
//       边沿判定由调用方（或本库的 take_press/take_release 锁存）完成。
//       HAL 只给电平不给事件，因此**同一帧内的按下+松开（亚帧点按）可能被漏检**；
//       人手点按远长于一帧（~16ms），实际使用无影响。
//    3) 坐标缓存：hal_touch.h 约定"未按下时不保证坐标有效"，故本库缓存最后一次
//       按下期间的有效坐标，松开后 x/y 仍返回该值（调试与"最后落点"语义都靠它）。
//    4) 触屏不可用（无触屏硬件 / 端口打开失败）时：is_available 返回 #f，
//       is_pressed 恒 #f，x/y 恒 0，take_* 恒 #f——不报错，App 可据此降级。
//
//  Scheme 侧用法：源码中声明 (native Touch) 后以 Touch.函数名 调用。
//
//  栈纪律（上游本地库契约 N1/N2，见解释器文档 doc/todo/stack_bal.md §11）：
//  所有函数都是值型——弹完参数后向操作数栈压入恰好 1 个值（本库全部零参数，
//  故净效应恒为 +1），随后调用一次 am_process_step。
////////////////////////////////////////////////////////////////////////////

#ifndef __AM_NATIVE_TOUCH_H__
#define __AM_NATIVE_TOUCH_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

// 由 infer/am_native_Touch.c 定义
extern const am_native_lib_entry_t am_native_Touch_lib;

// (Touch.is_available) : Boolean —— 宿主是否提供可用的触屏（HAL 探测成功）
int32_t am_native_Touch_is_available(am_runtime_t *rt, am_process_t *proc);

// (Touch.is_pressed) : Boolean —— 当前是否正在触摸
int32_t am_native_Touch_is_pressed(am_runtime_t *rt, am_process_t *proc);

// (Touch.x) : Number —— 触点 x 像素坐标（未触摸时为最后一次有效值；从未触摸为 0）
int32_t am_native_Touch_x(am_runtime_t *rt, am_process_t *proc);

// (Touch.y) : Number —— 触点 y 像素坐标（同上）
int32_t am_native_Touch_y(am_runtime_t *rt, am_process_t *proc);

// (Touch.take_press) : Boolean —— 自上次调用以来是否出现过"按下"沿（读后清零）
int32_t am_native_Touch_take_press(am_runtime_t *rt, am_process_t *proc);

// (Touch.take_release) : Boolean —— 自上次调用以来是否出现过"松开"沿（读后清零）
int32_t am_native_Touch_take_release(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif