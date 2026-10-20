////////////////////////////////////////////////////////////////////////////
//  Native Library : Time —— 灵机（Animac）时间本地库
//
//  为什么需要本库：
//    1) vendored System.timestamp 以 am_uint_t 承载毫秒时间戳；32 位宿主（wasm32）
//       上无符号整数的逻辑载荷仅 27 位（约 1.34e8），纪元毫秒（约 1.8e12）会被
//       截断为「纪元毫秒 mod 2^27」，无法还原日期；
//    2) 32 位宿主的 am_float_t 是 IEEE-754 单精度，直接以浮点传毫秒/秒同样损失
//       秒级精度（1.8e12 在 float32 下分辨率约 131 秒）。
//  另注：vendored 虚拟机把 float 的位模式右移 5 位存放（腾出类型标签），32 位宿主
//  上有效尾数只剩约 19 位，2.4e6 量级（儒略日）的浮点粒度可达 8 天，故本库既不
//  返回毫秒也不返回儒略日，只返回两个小整数。
//  因此本库把时间拆成两个都在 float32 下精确可表示的量返回：
//    · utc_days：自 1970-01-01 起的 UTC 日数（当前约 2e4）
//    · utc_sod ：当日 UTC 秒数（0..86399）
//  调用方据此重建民用时并套用本地时区（见 app/ephemeris.scm）。
//  两个函数各自读取一次系统时钟；取用之间的跨秒抖动对秒级显示无影响。
////////////////////////////////////////////////////////////////////////////

#ifndef __AM_NATIVE_TIME_H__
#define __AM_NATIVE_TIME_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "animac.h"

// 由 infer/am_native_Time.c 定义
extern const am_native_lib_entry_t am_native_Time_lib;

// (Time.utc_days) : Number（无符号整数）—— 自 1970-01-01 起的 UTC 日数
int32_t am_native_Time_utc_days(am_runtime_t *rt, am_process_t *proc);

// (Time.utc_sod) : Number（无符号整数）—— 当日 UTC 秒数（0..86399）
int32_t am_native_Time_utc_sod(am_runtime_t *rt, am_process_t *proc);

#ifdef __cplusplus
}
#endif

#endif
