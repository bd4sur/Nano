#ifndef __LINGLONG_NONGLI_H__
#define __LINGLONG_NONGLI_H__

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 数据结构定义 ---- */

typedef struct {
    int year, month, day, hour, minute, second;
} DateTime;

typedef struct {
    double      jd;
    double      longitude;
    const char *name;
    int         month_index; /* 0=正月(雨水), …, 11=腊月(大寒) */
} MajorTerm;

/* 对应 JS 版本 calculate_lunar_date() 的返回对象 */
typedef struct {
    int  year;
    int  month;        /* 1–12 */
    int  day;
    int  is_leap;
    char year_ganzhi [16];
    char month_ganzhi[16];
    char day_ganzhi  [16];
    char zodiac      [16];
    char month_name  [32];
    char day_name    [16];
    char full_display[128];
} LunarDate;

/*
 * 计算指定公历日期对应的农历信息。
 *
 * 对应 JS 版本的 LunarCalendar.calculate()（即内部 calculate_lunar_date()）。
 *
 * 参数：
 *   year, month, day      - 公历日期
 *   hour, minute, second  - 本地时间
 *   timezone_offset       - 时区偏移（小时），东八区传 8.0
 *
 * 返回：
 *   指向内部静态 LunarDate 结构的指针；推算失败返回 NULL。
 *   返回值在下次调用前有效（非线程安全）。
 */
LunarDate *lunar_calculate(int year, int month, int day,
                            int hour, int minute, int second,
                            double timezone_offset);

#ifdef __cplusplus
}
#endif

#endif
