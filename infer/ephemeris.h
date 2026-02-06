#ifndef __NANO_EPHEMERIS_H__
#define __NANO_EPHEMERIS_H__

#ifdef __cplusplus
extern "C" {
#endif

// 计算儒略日（输入地方标准时间）
double julian_day(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second, double timezone_offset);

// 计算月球黄道坐标（黄经lambda, 黄纬beta, 地心月心距离delta）
void calculate_lunar_ecliptic_coordinates(double jd, double *lambda, double *beta, double *delta);

// 计算月球赤道坐标（RA, Dec）
void calculate_lunar_equatorial_coordinates(double jd, double *RA, double *Dec);

// 计算太阳黄道坐标（黄经lambda, 黄纬beta, 地心月心距离delta）
void calculate_solar_ecliptic_coordinates(double jd, double *lambda, double *beta, double *delta);

// 计算太阳赤道坐标（RA, Dec）
void calculate_solar_equatorial_coordinates(double jd, double *RA, double *Dec);

// 月球相位角
double moon_phase(int year, int month, int day, int hour, int minute, int second, double timezone_offset);

// 月球亮面方向角
double moon_bright_limb_pos_angle(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second, double timezone_offset);

// 赤道坐标 → 地平坐标
void equatorial_to_horizontal(
    double ra_deg,          // 赤经 (0~360°)
    double dec_deg,         // 赤纬 (-90~+90°)
    int year, int month, int day, int hour, int minute, int second,
    double timezone_offset, // 时区偏移（小时），如北京时间 +8.0
    double longitude,       // 观测者经度 (东正)
    double latitude,        // 观测者纬度 (北正)
    double* azimuth,        // 输出：方位角（北=0°，东=90°）
    double* altitude        // 输出：高度角（度）
);


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

// 二分查找日出时间
int32_t find_sunrise(int32_t year, int32_t month, int32_t day, double timezone, double longitude, double latitude);

// 二分查找日落时间
int32_t find_sunset(int32_t year, int32_t month, int32_t day, double timezone, double longitude, double latitude);


// 基于VSOP87c计算大行星地平坐标（不做章动、光行差等精密修正）
void where_is_the_planet(
    int year, int month, int day, int hour, int minute, int second,
    double timezone_offset, // 时区偏移（小时），如北京时间 +8.0
    double longitude,       // 观测者经度 (东正)
    double latitude,        // 观测者纬度 (北正)
    int32_t planet_index,   // 计算哪个行星：1-水 2-金 3-地球（直接返回） 4-火 5-木 6-土 7-天王 8-海王 其他无定义
    double* azimuth,        // 输出：方位角（北=0°，东=90°）
    double* altitude        // 输出：高度角（度）
);



#ifdef __cplusplus
}
#endif

#endif
