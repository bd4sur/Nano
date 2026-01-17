#ifndef __NANO_EPHEMERIS_H__
#define __NANO_EPHEMERIS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"
#include "ui.h"

// 给定时间地点，计算太阳地平坐标
void where_is_the_sun(
    int year, int month, int day, int hour, int minute, int second,
    double timezone_offset, // 时区偏移（小时），如北京时间 +8.0
    double longitude,       // 观测者经度 (东正)
    double latitude,        // 观测者纬度 (北正)
    double* azimuth,        // 输出：方位角（北=0°，东=90°）
    double* altitude        // 输出：高度角（度）
);

// 给定时间地点，计算月球地平坐标
void where_is_the_moon(
    int year, int month, int day, int hour, int minute, int second,
    double timezone_offset, // 时区偏移（小时），如北京时间 +8.0
    double longitude,       // 观测者经度 (东正)
    double latitude,        // 观测者纬度 (北正)
    double* azimuth,        // 输出：方位角（北=0°，东=90°）
    double* altitude        // 输出：高度角（度）
);

void draw_ephemeris_screen(Key_Event *key_event, Global_State *global_state);

#ifdef __cplusplus
}
#endif

#endif
