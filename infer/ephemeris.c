#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "ephemeris.h"
#include "ui.h"
#include "graphics.h"

// gcc ephemeris.c -o ephemeris

#define PI (3.1415926565358979323846)

static inline double to_deg(double rad) {
    return rad * (180.0 / PI);
}

static inline double to_rad(double deg) {
    return deg * (PI / 180.0);
}

// 辅助函数：归一化角度到 [0, 360)
// static inline double normalize_angle(double n) {
//     return 360.0 * (n / 360.0 - trunc(n / 360.0));
// }
static double normalize_angle(double angle) {
    angle = fmod(angle, 360.0);
    if (angle < 0) angle += 360.0;
    return angle;
}

// 判断格里历年份是否为闰年（仅适用于1583年及以后）
static inline int is_leap(int32_t year) {
    return (year % 4 == 0) && (year % 100 != 0 || year % 400 == 0);
}

// 计算从1583年1月1日到给定日期（y, m, d）之间的天数（1583-01-01 为第 0 天）
static int64_t days_since_1583(int32_t y, int32_t m, int32_t d) {
    static const int8_t days_in_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    // 题目保证输入为合法格里历日期，且 y >= 1583
    int64_t total_days = 0;

    // 累加从1583年到 y-1 年每一年的天数
    for (int32_t year = 1583; year < y; ++year) {
        total_days += 365 + (is_leap(year) ? 1 : 0);
    }

    // 累加当年1月到 m-1 月的天数
    for (int month = 1; month < m; ++month) {
        total_days += days_in_month[month - 1];
        if (month == 2 && is_leap(y)) {
            total_days += 1;
        }
    }

    // 加上当月天数（d 从1开始，所以加 d - 1）
    total_days += d - 1;

    return total_days;
}

int64_t days_prd(int32_t y1, int32_t m1, int32_t d1,
                 int32_t y2, int32_t m2, int32_t d2) {
    int64_t days1 = days_since_1583(y1, m1, d1);
    int64_t days2 = days_since_1583(y2, m2, d2);
    return days2 - days1;
}

// 计算儒略日（地方标准时间）
static double julian_day(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second, double timezone_offset) {
    double local_hours = hour + minute / 60.0 + second / 3600.0;
    double utc_hours = local_hours - timezone_offset;
    int32_t utc_day = day;
    int32_t utc_month = month;
    int32_t utc_year = year;

    // 处理跨日/月/年
    static const uint32_t days_in_month[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    while (utc_hours < 0) {
        utc_hours += 24.0;
        utc_day--;
        // 处理跨月和跨年
        if (utc_day < 1) {
            utc_month--;
            if (utc_month < 1) {
                utc_month = 12;
                utc_year--;
            }
            // 计算前一月的最后一天
            utc_day = days_in_month[utc_month];
            utc_day += (utc_month == 2 && is_leap(utc_year)) ? 1 : 0;
        }
    }
    while (utc_hours >= 24.0) {
        utc_hours -= 24.0;
        utc_day++;
        uint32_t last_day = days_in_month[utc_month];
        last_day += (utc_month == 2 && is_leap(utc_year)) ? 1 : 0;
        if (utc_day > last_day) {
            utc_day = 1;
            utc_month++;
            if (utc_month > 12) {
                utc_month = 1;
                utc_year++;
            }
        }
    }

    if (utc_month <= 2) {
        utc_year -= 1;
        utc_month += 12;
    }
    int32_t A = utc_year / 100;
    int32_t B = 2 - A + (A / 4);
    double jd = trunc(365.25 * (utc_year + 4716)) + trunc(30.6001 * (utc_month + 1)) + utc_day + B - 1524.5;
    jd += utc_hours / 24.0;
    return jd;
}

// 计算格林尼治恒星时（GMST，单位：度）
static double greenwich_sidereal_time(double jd) {
    double jd0 = floor(jd - 0.5) + 0.5; // 当日 0h UT 的 JD
    double T = (jd0 - 2451545.0) / 36525.0; // 儒略世纪
    double theta0 = 100.46061837 +
                    36000.770053608 * T +
                    0.000387933 * T * T -
                    T * T * T / 38710000.0;
    theta0 = normalize_angle(theta0); // GMST at 0h UT, in degrees

    double ut_hours = (jd - jd0) * 24.0;
    double gmst = theta0 + 360.98564724 * ut_hours / 24.0;
    return normalize_angle(gmst); // in degrees
}

// 将 0~360 度的赤经转换为 h, m, s
void ra_deg_to_hms(double ra_deg, int* h, int* m, double* s) {
    // 1. 归一化到 [0, 360)
    ra_deg = fmod(ra_deg, 360.0);
    if (ra_deg < 0) ra_deg += 360.0;
    // 2. 转换为小时（0 ~ 24）
    double hours = ra_deg / 15.0;  // 因为 360° / 24h = 15°/h
    // 3. 归一化到 [0, 24)
    hours = fmod(hours, 24.0);
    if (hours < 0) hours += 24.0;
    // 4. 分离整数小时
    *h = (int)floor(hours);
    double frac_h = hours - *h;
    // 5. 转换为分钟
    double minutes = frac_h * 60.0;
    *m = (int)floor(minutes);
    *s = (minutes - *m) * 60.0;
    // 6. 处理四舍五入导致的进位（可选但推荐）
    if (*s >= 60.0) {
        *s -= 60.0;
        (*m)++;
        if (*m >= 60) {
            *m -= 60;
            (*h)++;
            if (*h >= 24) *h -= 24;
        }
    }
}

// 示例：打印 hms 格式
void print_ra_hms(double ra_deg) {
    int h, m;
    double s;
    ra_deg_to_hms(ra_deg, &h, &m, &s);
    printf("%02dh %02dm %.3fs\n", h, m, s);
}


// 计算月球赤道坐标（RA, Dec）
//   输入时间是儒略日，因天体在天球上的位置与观察者位置无关
void calculate_lunar_equatorial_coordinates(double jd, double *RA, double *Dec) {
    double H0 = jd - 2451545.0; // 自 J2000.0 起算的儒略日数
    double H = H0 / 36525.0; // 自 J2000.0 起算的儒略世纪数

    // 月球平黄经
    double I = 218.3164591 + 481267.88134236 * H - 0.0013268 * H*H + H*H*H / 538841.0 - H*H*H*H / 65194000.0;
    double E = normalize_angle(I);
    double T = (E <= 270.0) ? (270.0 - E) : (630.0 - E);

    // 月球升交点黄经
    double J = 125.044555 - 1934.1361849 * H + 0.0020762 * H*H + H*H*H / 467410.0 - H*H*H*H / 60616000.0;
    double G = (J >= 0.0) ? (360.0 - J) : (360.0 - (360.0 * (1 - (-J / 360.0 - trunc(-J / 360.0)))));

    // 计算月球黄道坐标
    double R = 1;
    double r = 180.0 - 5.14; // 黄白交角
    r = to_rad(r); G = to_rad(G); T = to_rad(T);
    double M = R * cos(r) * cos(G) * sin(T) + R * sin(G) * cos(T);
    double N = R * cos(r) * sin(G) * sin(T) - R * cos(G) * cos(T);
    double O = R * sin(r) * sin(T);

    // 月球黄纬
    double F = to_deg(atan2(O, sqrt(M * M + N * N)));

    // 月球赤道坐标
    double theta = 23.0 + (26.0 / 60.0); // 黄赤交角
    F = to_rad(F); theta = to_rad(theta); E = to_rad(E);
    double X = -R * cos(F) * cos(theta) * sin(E) + R * sin(F) * sin(theta);
    double Y = -R * cos(F) * cos(E);
    double Z =  R * cos(F) * sin(theta) * sin(E) + R * sin(F) * cos(theta);

    // 月球RA
    double ra_deg = to_deg(atan2(Y, X));
    if (ra_deg > -90.0 && ra_deg <= 180.0) {
        *RA = 270.0 - ra_deg;
    }
    else if (ra_deg <= -90.0 && ra_deg > -180.0) {
        *RA = -90.0 - ra_deg;
    }

    // 月球Dec
    *Dec = to_deg(atan2(Z, sqrt(X * X + Y * Y)));
}


// 计算太阳赤道坐标（RA, Dec）
//   输入时间是儒略日，因天体在天球上的位置与观察者位置无关
void calculate_solar_equatorial_coordinates(double jd, double *RA, double *Dec) {
    double H0 = jd - 2451545.0; // 自 J2000.0 起算的儒略日数
    double H = H0 / 36525.0; // 自 J2000.0 起算的儒略世纪数

    // 太阳的近似黄经（极其粗糙的估算，但不是不能用）
    // double U1 = days_prd(year, 03, 21, year, month, day);
    // U1 = (U1 >= 0) ? U1 * 360.0 / 365.0 : (365.0 + U1) * 360.0 / 365.0;

    // 太阳平黄经
    double U1 = normalize_angle(280.46646 + 36000.76983 * H + 0.0003032 * H * H);
    // 平近点角 Mm（转为弧度）
    double Mm_deg = normalize_angle(357.52911 + 35999.05029 * H - 0.0001537 * H * H);
    double Mm = to_rad(Mm_deg);
    // 中心差C（度）
    double C = (1.914602 - 0.004817 * H - 0.000014 * H * H) * sin(Mm)
             + (0.019993 - 0.000101 * H) * sin(2 * Mm)
             + 0.000289 * sin(3 * Mm);
    // 真黄经（无其余修正）
    double U = normalize_angle(U1 + C);

    // 太阳的赤道坐标
    double theta = 23.0 + (26.0 / 60.0); // 黄赤交角
    double M = -cos(to_rad(theta)) * sin(to_rad(U));
    double N = -cos(to_rad(U));
    double O = sin(to_rad(theta)) * sin(to_rad(U));

    // 太阳的赤经和赤纬
    if (U >= 0.0 && U < 180.0) {
        *RA = 90.0 - to_deg(asin(-N / sqrt(M*M + N*N)));
    }
    else {
        *RA = 270.0 + to_deg(asin(-N / sqrt(M*M + N*N)));
    }
    *Dec = to_deg(atan2(O, sqrt(M*M + N*N)));
}

// 带符号月相：-1.0 ~ +1.0
// 假设输入 RA 和 Dec 单位为弧度
double moon_phase(int year, int month, int day, int hour, int minute, int second, double timezone_offset) {
    double sun_ra = 0.0;
    double sun_dec = 0.0;
    double moon_ra = 0.0;
    double moon_dec = 0.0;
    double jd = julian_day(year, month, day, hour, minute, second, timezone_offset);
    calculate_lunar_equatorial_coordinates(jd, &moon_ra, &moon_dec);
    calculate_solar_equatorial_coordinates(jd, &sun_ra, &sun_dec);

    sun_ra = to_rad(sun_ra);
    sun_dec = to_rad(sun_dec);
    moon_ra = to_rad(moon_ra);
    moon_dec = to_rad(moon_dec);

    // 1. 转换为单位向量
    double sun_x = cos(sun_dec) * cos(sun_ra);
    double sun_y = cos(sun_dec) * sin(sun_ra);
    double sun_z = sin(sun_dec);

    double moon_x = cos(moon_dec) * cos(moon_ra);
    double moon_y = cos(moon_dec) * sin(moon_ra);
    double moon_z = sin(moon_dec);

    // 2. 计算日月角距 psi（弧度）
    double dot = sun_x * moon_x + sun_y * moon_y + sun_z * moon_z;
    // 限制 dot 在 [-1, 1] 防止数值误差
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    double psi = acos(dot); // [0, π]

    // 3. 照明比例（0 ~ 1）
    double illumination = (1.0 + cos(psi)) / 2.0; // 注意：这是简化模型！

    // 4. 判断盈亏：比较赤经（需处理 0/2π 跳变）
    double ra_diff = fmod(moon_ra - sun_ra, 2.0 * PI);
    if (ra_diff < 0) ra_diff += 2.0 * PI;

    // 若月亮赤经领先太阳（0 < diff < π），则为盈（waxing，正）
    // 若落后（π < diff < 2π），则为亏（waning，负）
    double sign = (ra_diff < PI) ? 1.0 : -1.0;

    // 5. 带符号月相：-1.0 ~ +1.0
    double signed_phase = sign * illumination;

    return signed_phase;
}


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
) {
    // 计算儒略日（JD）
    double jd = julian_day(year, month, day, hour, minute, second, timezone_offset);
    // 格林尼治恒星时（GMST，度）
    double gmst = greenwich_sidereal_time(jd);

    // 计算地平坐标
    // double H = (LMT >= 0.0 && LMT < 12.0) ? (180.0 + LMT * (360.0/24.0)) : (LMT - 12.0) * (360.0/24.0); // 太阳本地时角
    // double L = (E + H <= 360.0) ? (E + H) : (E + H - 360.0); // 本地恒星时（春分点本地时角）
    double L = gmst + longitude; // 本地恒星时（春分点本地时角）
    double I = normalize_angle(L - ra_deg);
    double T = (I >= 0 && I < 180.0) ? (I + 90.0) : (I - 270.0); // 月球本地时角

    double R = 1.0;
    dec_deg = to_rad(dec_deg); T = to_rad(T); latitude = to_rad(latitude);
    double X =  R * cos(dec_deg) * sin(latitude) * sin(T) - R * sin(dec_deg) * cos(latitude);
    double Y = -R * cos(dec_deg) * cos(T);
    double Z =  R * cos(dec_deg) * cos(latitude) * sin(T) + R * sin(dec_deg) * sin(latitude);

    // Elevation (Altitude)
    *altitude = to_deg(atan2(Z, sqrt(X * X + Y * Y)));
    // Azimuth
    double az_deg = to_deg(atan2(Y, X));
    *azimuth = 180.0 + az_deg;
}

// 给定时间地点，计算太阳地平坐标
void where_is_the_sun(
    int year, int month, int day, int hour, int minute, int second,
    double timezone_offset, // 时区偏移（小时），如北京时间 +8.0
    double longitude,       // 观测者经度 (东正)
    double latitude,        // 观测者纬度 (北正)
    double* azimuth,        // 输出：方位角（北=0°，东=90°）
    double* altitude        // 输出：高度角（度）
) {
    double RA = 0.0;
    double Dec = 0.0;
    double jd = julian_day(year, month, day, hour, minute, second, timezone_offset);
    calculate_solar_equatorial_coordinates(jd, &RA, &Dec);

    // printf("太阳赤经: ");
    // print_ra_hms(RA);
    // printf("太阳赤纬: %.6f degrees\n", Dec);

    equatorial_to_horizontal(
        RA, Dec, year, month, day, hour, minute, second, timezone_offset,
        longitude, latitude,
        azimuth, altitude
    );
}

// 给定时间地点，计算月球地平坐标
void where_is_the_moon(
    int year, int month, int day, int hour, int minute, int second,
    double timezone_offset, // 时区偏移（小时），如北京时间 +8.0
    double longitude,       // 观测者经度 (东正)
    double latitude,        // 观测者纬度 (北正)
    double* azimuth,        // 输出：方位角（北=0°，东=90°）
    double* altitude        // 输出：高度角（度）
) {
    double RA = 0.0;
    double Dec = 0.0;
    double jd = julian_day(year, month, day, hour, minute, second, timezone_offset);
    calculate_lunar_equatorial_coordinates(jd, &RA, &Dec);

    // printf("月球赤经: ");
    // print_ra_hms(RA);
    // printf("月球赤纬: %.6f degrees\n", Dec);

    equatorial_to_horizontal(
        RA, Dec, year, month, day, hour, minute, second, timezone_offset,
        longitude, latitude,
        azimuth, altitude
    );
}


// 辅助函数：计算某分钟时刻的太阳高度角（单位：分钟从00:00起）
static double get_solar_altitude(int32_t year, int32_t month, int32_t day, int32_t total_minutes, double timezone, double longitude, double latitude) {
    int32_t h = total_minutes / 60;
    int32_t m = total_minutes % 60;
    // 注意：原函数 where_is_the_sun 的秒参数固定为0
    double azi, alt;
    where_is_the_sun(year, month, day, h, m, 0, timezone, longitude, latitude, &azi, &alt);
    return alt;
}

// 二分查找日出时间（升交点）：在 [start, end) 区间内找第一个 alt >= 0 的点，
// 且前一时刻 alt < 0。假设函数在此区间单调递增。
static int32_t find_sunrise(int32_t year, int32_t month, int32_t day, double timezone, double longitude, double latitude) {
    int32_t low = 0;               // 00:00
    int32_t high = 12 * 60;        // 12:00（不含）

    // 先检查边界：如果全天都在地平线下，无日出
    if (get_solar_altitude(year, month, day, high - 1, timezone, longitude, latitude) < 0.0) {
        return -1; // 无日出
    }
    if (get_solar_altitude(year, month, day, low, timezone, longitude, latitude) >= 0.0) {
        return low; // 极昼情况，日出在00:00或之前
    }

    while (low < high) {
        int32_t mid = (low + high) / 2;
        double alt_mid = get_solar_altitude(year, month, day, mid, timezone, longitude, latitude);
        double alt_prev = get_solar_altitude(year, month, day, mid - 1, timezone, longitude, latitude);
        if (alt_prev < 0.0 && alt_mid >= 0.0) {
            return mid; // 找到升交点
        }
        else if (alt_mid < 0.0) {
            low = mid + 1;
        }
        else {
            high = mid;
        }
    }
    return -1; // 理论上不应到达这里
}

// 二分查找日落时间（降交点）：在 [start, end) 区间内找第一个 alt < 0 的点，
// 且前一时刻 alt >= 0。假设函数在此区间单调递减。
static int32_t find_sunset(int32_t year, int32_t month, int32_t day, double timezone, double longitude, double latitude) {
    int32_t low = 12 * 60 + 1;     // 12:01
    int32_t high = 24 * 60;        // 23:59 + 1（即24:00，作为上界）

    // 边界检查：如果12:01时已低于地平线，则可能无日落（极夜）
    if (get_solar_altitude(year, month, day, low, timezone, longitude, latitude) < 0.0) {
        return -1;
    }
    // 如果23:59仍 >= 0，则极昼，无日落
    if (get_solar_altitude(year, month, day, high - 1, timezone, longitude, latitude) >= 0.0) {
        return -1;
    }

    while (low < high) {
        int32_t mid = (low + high) / 2;
        double alt_mid = get_solar_altitude(year, month, day, mid, timezone, longitude, latitude);
        double alt_prev = get_solar_altitude(year, month, day, mid - 1, timezone, longitude, latitude);
        if (alt_prev >= 0.0 && alt_mid < 0.0) {
            return mid; // 找到降交点
        }
        else if (alt_prev >= 0.0) {
            low = mid + 1;
        }
        else {
            high = mid;
        }
    }
    return -1;
}



static uint64_t ephemeris_refersh_timestamp = 0;
static uint64_t ephemeris_first_call_timestamp = 0;
static uint32_t last_day = 0;
static uint32_t sunrise_time[2] = {0, 0}; // hour, minute
static uint32_t sunset_time[2] = {0, 0}; // hour, minute
static int32_t is_speed_up = 0;
static uint64_t start_timestamp = 0;
void draw_ephemeris_screen(Key_Event *key_event, Global_State *global_state) {
    const double longitude = 119.0; // 东经
    const double latitude = 32.0;   // 北纬
    const double timezone = +8.0;   // 东八区

    // 节流：不大于50fps
    if (global_state->timestamp - ephemeris_refersh_timestamp < 20) {
        return;
    }
    ephemeris_refersh_timestamp = global_state->timestamp;

    fb_soft_clear();

    fb_draw_circle(64, 32, 30);
    fb_draw_circle(64, 32, 20);
    fb_draw_circle(64, 32, 10);
    fb_draw_line(32, 32, 96, 32, 1);
    fb_draw_line(64, 0, 64, 64, 1);

    // 方位文字和周围的边框
    fb_draw_textline_mini(L"N", 62, 0, 1);  fb_plot(61, 2, 0); fb_plot(67, 2, 0); fb_plot(64, 5, 0);
    fb_draw_textline_mini(L"S", 63, 59, 1); fb_plot(62, 62, 0); fb_plot(66, 62, 0); fb_plot(64, 58, 0);
    fb_draw_textline_mini(L"W", 32, 30, 1); fb_plot(34, 29, 0); fb_plot(37, 32, 0); fb_plot(34, 35, 0);
    fb_draw_textline_mini(L"E", 93, 30, 1); fb_plot(92, 32, 0); fb_plot(96, 32, 0); fb_plot(94, 29, 0); fb_plot(94, 35, 0);

    fb_draw_line(0, 43, 30, 43, 1);

    time_t ts = 0;
    if (is_speed_up) {
        start_timestamp += 600000;
        ts = (time_t)start_timestamp / 1000;
    }
    else {
        ts = (time_t)global_state->timestamp / 1000;
    }
    struct tm *timeinfo = localtime(&ts); // 转换为本地时间
    
    int32_t second = timeinfo->tm_sec;
    int32_t minute = timeinfo->tm_min;
    int32_t hour = timeinfo->tm_hour;
    int32_t day = timeinfo->tm_mday;
    int32_t month = timeinfo->tm_mon + 1;
    int32_t year = timeinfo->tm_year + 1900;


    wchar_t timestr[30];
    swprintf(timestr, 30, L"%04d-%02d-%02d\n%02d:%02d:%02d", year, month, day, hour, minute, second);
    fb_draw_textline_mini(timestr, 0, 0, 1);


    double altitude_moon = 0.0;
    double azimuth_moon = 0.0;

    where_is_the_moon(year, month, day, hour, minute, second, timezone, longitude, latitude, &azimuth_moon, &altitude_moon);
    double phase = moon_phase(year, month, day, hour, minute, second, timezone);

    wchar_t coordstr_moon[30];
    swprintf(coordstr_moon, 30, L"MOON\nP:%d%%\nA:%.1f\nE:%.1f", (int32_t)(phase * 100.0), azimuth_moon, altitude_moon);
    fb_draw_textline_mini(coordstr_moon, 0, 18, 1);

    double x_moon = 64 + (90.0 - altitude_moon) * 32.0 / 90.0 * sin(to_rad(azimuth_moon));
    double y_moon = 32 - (90.0 - altitude_moon) * 32.0 / 90.0 * cos(to_rad(azimuth_moon));

    fb_plot((int)x_moon - 1, (int)y_moon - 1, 1);
    fb_plot((int)x_moon - 1, (int)y_moon - 0, 1);
    fb_plot((int)x_moon - 1, (int)y_moon + 1, 1);
    fb_plot((int)x_moon - 0, (int)y_moon - 1, 1);
    fb_plot((int)x_moon - 0, (int)y_moon - 0, 1);
    fb_plot((int)x_moon - 0, (int)y_moon + 1, 1);
    fb_plot((int)x_moon + 1, (int)y_moon - 1, 1);
    fb_plot((int)x_moon + 1, (int)y_moon - 0, 1);
    fb_plot((int)x_moon + 1, (int)y_moon + 1, 1);



    double altitude_sun = 0.0;
    double azimuth_sun = 0.0;

    where_is_the_sun(year, month, day, hour, minute, second, +8.0, longitude, latitude, &azimuth_sun, &altitude_sun);

    wchar_t coordstr_sun[30];
    swprintf(coordstr_sun, 30, L"SUN\nA:%.1f\nE:%.1f", azimuth_sun, altitude_sun);
    fb_draw_textline_mini(coordstr_sun, 0, 46, 1);

    double x_sun = 64 + (90.0 - altitude_sun) * 32.0 / 90.0 * sin(to_rad(azimuth_sun));
    double y_sun = 32 - (90.0 - altitude_sun) * 32.0 / 90.0 * cos(to_rad(azimuth_sun));

    fb_draw_circle((int)x_sun, (int)y_sun, 2);


    // 二分搜索日出日落时间
    if (ephemeris_first_call_timestamp == 0 || last_day != day) { // 只在首次调用和当天日期变化时计算
        int32_t sunrise_min = find_sunrise(year, month, day, timezone, longitude, latitude);
        if (sunrise_min != -1) {
            sunrise_time[0] = sunrise_min / 60;
            sunrise_time[1] = sunrise_min % 60;
        }
        int32_t sunset_min = find_sunset(year, month, day, timezone, longitude, latitude);
        if (sunset_min != -1) {
            sunset_time[0] = sunset_min / 60;
            sunset_time[1] = sunset_min % 60;
        }
    }
    wchar_t risefall_time[60];
    swprintf(risefall_time, 60, L"R:%02d:%02d\nS:%02d:%02d", sunrise_time[0], sunrise_time[1], sunset_time[0], sunset_time[1]);
    fb_draw_textline_mini(risefall_time, 98, 0, 1);


    fb_draw_textline_mini(L"    BD4SUR\n2011-09-29", 86, 53, 1);

    gfx_refresh();

    if (ephemeris_first_call_timestamp == 0) {
        ephemeris_first_call_timestamp = global_state->timestamp;
    }
    last_day = day;
}

void ephemeris_toggle_speedup(Key_Event *key_event, Global_State *global_state) {
    is_speed_up = !is_speed_up;
    start_timestamp = global_state->timestamp;
}





void ephemeris(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second, double timezone, double longitude, double latitude) {

    double Elevation = 0.0;
    double Azimuth = 0.0;

    where_is_the_sun(year, month, day, hour, minute, second, timezone, longitude, latitude, &Azimuth, &Elevation);
    printf("太阳地平高度: %.6f degrees\n", Elevation);
    printf("太阳方位角: %.6f degrees\n", Azimuth);

    where_is_the_moon(year, month, day, hour, minute, second, timezone, longitude, latitude, &Azimuth, &Elevation);
    printf("月球地平高度: %.6f degrees\n", Elevation);
    printf("月球方位角: %.6f degrees\n", Azimuth);

}

int __main__(void) {
    ephemeris(2026, 1, 17, 22, 00, 00, +8.0, 119.0, 32.0);
    return 0;
}
