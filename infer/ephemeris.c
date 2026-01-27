/*
 * 星历计算
 *   计算太阳、月球的黄道和赤道坐标及其地平坐标、月相、日出日落时间，并根据这些数据推算农历日期。
 *   基于 J. Meeus 的 Astronomical Algorithms (Second Edition) 所述算法，以及自行推导算法实现
 *   (c) BD4SUR 2011-08 2011-09 2026-01
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "ephemeris.h"
#include "ui.h"
#include "graphics.h"

// gcc ephemeris.c -o ephemeris

#define PI (3.14159265358979323846)

// ===============================================================================
// 用于计算月球位置的参数 (AA Ch. 47)
// ===============================================================================

static const int32_t A_multiple_of_D[60] = {
    0, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 1, 0, 2, 0, 0, 4, 0, 4, 2, 2, 1, 1, 2, 2, 4, 2, 0, 2, 2,
    1, 2, 0, 0, 2, 2, 2, 4, 0, 3, 2, 4, 0, 2, 2, 2, 4, 0, 4, 1, 2, 0, 1, 3, 4, 2, 0, 1, 2, 2
};

static const int32_t A_multiple_of_M[60] = {
    0, 0, 0, 0, 1, 0, 0, -1, 0, -1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, -1, 0, 0, 0, 1, 0, -1,
    0, -2, 1, 2, -2, 0, 0, -1, 0, 0, 1, -1, 2, 2, 1, -1, 0, 0, -1, 0, 1, 0, 1, 0, 0, -1, 2, 1, 0, 0
};

static const int32_t A_multiple_of_M2[60] = {
    1, -1, 0, 2, 0, 0, -2, -1, 1, 0, -1, 0, 1, 0, 1, 1, -1, 3, -2, -1, 0, -1, 0, 1, 2, 0, -3, -2, -1, -2,
    1, 0, 2, 0, -1, 1, 0, -1, 2, -1, 1, -2, -1, -1, -2, 0, 1, 4, 0, -2, 0, 2, 1, -2, -3, 2, 1, -1, 3, -1
};

static const int32_t A_multiple_of_F[60] = {
    0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, -2, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, -2, 2, 0, 2, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, -2
};

static const int32_t A_coeff_sine[60] = {
    6288774, 1274027, 658314, 213618, -185116, -114332, 58793, 57066, 53322, 45758, -40923, -34720,
    -30383, 15327, -12528, 10980, 10675, 10034, 8548, -7888, -6766, -5163, 4987, 4036, 3994, 3861,
    3665, -2689, -2602, 2390, -2348, 2236, -2120, -2069, 2048, -1773, -1595, 1215, -1110, -892, -810,
    759, -713, -700, 691, 596, 549, 537, 520, -487, -399, -381, 351, -340, 330, 327, -323, 299, 294, 0
};

static const int32_t A_coeff_cosine[60] = {
    -20905355, -3699111, -2955968, -569925, 48888, -3149, 246158, -152138, -170733, -204586, -129620,
    108743, 104755, 10321, 0, 79661, -34782, -23210, -21636, 24208, 30824, -8379, -16675, -12831, -10445,
    -11650, 14403, -7003, 0, 10056, 6322, -9884, 5751, 0, -4950, 4130, 0, -3958, 0, 3258, 2616, -1897,
    -2117, 2354, 0, 0, -1423, -1117, -1571, -1739, 0, -4421, 0, 0, 0, 0, 1165, 0, 0, 8752
};


static const int32_t B_multiple_of_D[60] = {
    0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 4, 4,
    0, 4, 2, 2, 2, 2, 0, 2, 2, 2, 2, 4, 2, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 4, 4, 1, 4, 1, 4, 2
};

static const int32_t B_multiple_of_M[60] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, -1, -1, -1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 1, 0, -1, -2, 0, 1, 1, 1, 1, 1, 0, -1, 1, 0, -1, 0, 0, 0, -1, -2
};

static const int32_t B_multiple_of_M2[60] = {
    0, 1, 1, 0, -1, -1, 0, 2, 1, 2, 0, -2, 1, 0, -1, 0, -1, -1, -1, 0, 0, -1, 0, 1, 1, 0, 0, 3, 0, -1,
    1, -2, 0, 2, 1, -2, 3, 2, -3, -1, 0, 0, 1, 0, 1, 1, 0, 0, -2, -1, 1, -2, 2, -2, -1, 1, 1, -1, 0, 0
};

static const int32_t B_multiple_of_F[60] = {
    1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 3, 1, 1, 1, -1, -1, -1, 1, -1, 1,
    -3, 1, -3, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 3, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1
};

static const int32_t B_coeff_sine[60] = {
    5128122, 280602, 277693, 173237, 55413, 46271, 32573, 17198, 9266, 8822, 8216, 4324, 4200, -3359, 2463, 2211,
    2065, -1870, 1828, -1794, -1749, -1565, -1492, -1475, -1410, -1344, -1335, 1107, 1021, 833, 777, 671, 607, 596,
    491, -451, 439, 422, 421, -366, -351, 331, 315, 302, -283, -229, 223, 223, -220, -220, -185, 181, -177, 176, 166,
    -164, 132, -119, 115, 107
};








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

static int64_t days_prd(int32_t y1, int32_t m1, int32_t d1,
                 int32_t y2, int32_t m2, int32_t d2) {
    int64_t days1 = days_since_1583(y1, m1, d1);
    int64_t days2 = days_since_1583(y2, m2, d2);
    return days2 - days1;
}

// 计算儒略日（输入地方标准时间）
double julian_day(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t minute, int32_t second, double timezone_offset) {
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


// 计算月球黄道坐标（黄经lambda, 黄纬beta, 地心月心距离delta）
void calculate_lunar_ecliptic_coordinates(double jd, double *lambda, double *beta, double *delta) {
    double H0 = jd - 2451545.0; // 自 J2000.0 起算的儒略日数
    double H = H0 / 36525.0; // 自 J2000.0 起算的儒略世纪数
/*
    // 月球平黄经
    double I = 218.3164591 + 481267.88134236 * H - 0.0013268 * H*H + H*H*H / 538841.0 - H*H*H*H / 65194000.0;
    double lambda_moon = normalize_angle(I);
    double T = (lambda_moon <= 270.0) ? (270.0 - lambda_moon) : (630.0 - lambda_moon);

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
    double beta_moon = to_deg(atan2(O, sqrt(M * M + N * N)));
*/

    double H2 = H*H;
    double H3 = H2*H;
    double H4 = H3*H;

    // 月球平黄经
    double LL = normalize_angle(
        218.3164477 + 481267.88123421 * H - 0.0015786 * H2 + H3 / 538841.0 - H4 / 65194000.0);
    // 月日平均距角
    double D = normalize_angle(
        297.8501921 + 445267.1114034 * H - 0.0018819 * H2 + H3 / 545868.0 - H4 / 113065000.0);
    // 太阳平近点角
    double M = normalize_angle(
        357.5291092 + 35999.0502909 * H - 0.0001536 * H2 + H3 / 24490000.0);
    // 月球平近点角
    double M2 = normalize_angle(
        134.9633964 + 477198.8675055 * H + 0.0087414 * H2 + H3 / 69699.0 - H4 / 14712000.0);
    // 月球经度参数（月球到其升交点的平均角距离）
    double F = normalize_angle(
        93.2720950 + 483202.0175233 * H - 0.0036539 * H2 - H3 / 3526000.0 + H4 / 863310000.0);
    // 三个修正系数
    double A1 = normalize_angle(119.75 + 131.849 * H);
    double A2 = normalize_angle(53.09 + 479264.290 * H);
    double A3 = normalize_angle(313.45 + 481266.484 * H);
    // 与地球公转轨道离心率相关的修正项
    double E = 1 - 0.002516 * H - 0.0000074 * H2;
    double E2 = E*E;

    // 计算修正项Σl、Σr、Σb
    double S_l = 0.0;
    double S_r = 0.0;
    double S_b = 0.0;
    for (int32_t i = 0; i < 60; i++) {
        double sc = A_coeff_sine[i];
        double cc = A_coeff_cosine[i];
        double m_D  = A_multiple_of_D[i];
        double m_M  = A_multiple_of_M[i];
        double m_M2 = A_multiple_of_M2[i];
        double m_F  = A_multiple_of_F[i];
        double e = (m_M == 2 || m_M == -2) ? E2 : ((m_M == 1 || m_M == -1) ? E : 1);
        S_l += sc * e * sin(m_D * to_rad(D) + m_M * to_rad(M) + m_M2 * to_rad(M2) + m_F * to_rad(F));
        S_r += cc * e * sin(m_D * to_rad(D) + m_M * to_rad(M) + m_M2 * to_rad(M2) + m_F * to_rad(F));
    }
    for (int32_t i = 0; i < 60; i++) {
        double sc = B_coeff_sine[i];
        double m_D  = B_multiple_of_D[i];
        double m_M  = B_multiple_of_M[i];
        double m_M2 = B_multiple_of_M2[i];
        double m_F  = B_multiple_of_F[i];
        double e = (m_M == 2 || m_M == -2) ? E2 : ((m_M == 1 || m_M == -1) ? E : 1);
        S_b += sc * e * sin(m_D * to_rad(D) + m_M * to_rad(M) + m_M2 * to_rad(M2) + m_F * to_rad(F));
    }
    S_l += (3958.0 * sin(to_rad(A1)) + 1962.0 * sin(to_rad(LL - F)) + 318.0 * sin(to_rad(A2)));
    S_b += (-2235.0 * sin(to_rad(LL)) + 382.0 * sin(to_rad(A3)) + 175.0 * sin(to_rad(A1 - F)) + 
            175.0 * sin(to_rad(A1 + F)) + 127.0 * sin(to_rad(LL - M2)) - 115.0 * sin(to_rad(LL + M2)));

    *lambda = LL + S_l / 1000000.0;
    *beta = S_b / 1000000.0;
    *delta = 385000.56 + S_r / 1000.0;

}


// 计算月球赤道坐标（RA, Dec）
void calculate_lunar_equatorial_coordinates(double jd, double *RA, double *Dec) {

    double lambda_moon = 0.0;
    double beta_moon = 0.0;
    double R_moon = 0.0;

    // 计算黄道坐标
    calculate_lunar_ecliptic_coordinates(jd, &lambda_moon, &beta_moon, &R_moon);

    // 计算赤道坐标
    double eps = 23.0 + (26.0 / 60.0); // 黄赤交角
    beta_moon = to_rad(beta_moon); eps = to_rad(eps); lambda_moon = to_rad(lambda_moon);
    double X = -R_moon * cos(beta_moon) * cos(eps) * sin(lambda_moon) + R_moon * sin(beta_moon) * sin(eps);
    double Y = -R_moon * cos(beta_moon) * cos(lambda_moon);
    double Z =  R_moon * cos(beta_moon) * sin(eps) * sin(lambda_moon) + R_moon * sin(beta_moon) * cos(eps);

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


// 计算太阳黄道坐标（黄经lambda, 黄纬beta, 地心月心距离delta）
void calculate_solar_ecliptic_coordinates(double jd, double *lambda, double *beta, double *delta) {
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

    *lambda = U;
    *beta = 0.0;
    *delta = 1.0;
}

// 计算太阳赤道坐标（RA, Dec）
void calculate_solar_equatorial_coordinates(double jd, double *RA, double *Dec) {

    double lambda_sun = 0.0;
    double beta_sun = 0.0;
    double R_sun = 1.0;

    // 计算黄道坐标（仅用到黄经）
    calculate_solar_ecliptic_coordinates(jd, &lambda_sun, &beta_sun, &R_sun);
    (void)beta_sun;
    (void)R_sun;

    // 太阳的赤道坐标
    double eps = 23.0 + (26.0 / 60.0); // 黄赤交角
    double M = -cos(to_rad(eps)) * sin(to_rad(lambda_sun));
    double N = -cos(to_rad(lambda_sun));
    double O = sin(to_rad(eps)) * sin(to_rad(lambda_sun));

    // 太阳的赤经和赤纬
    if (lambda_sun >= 0.0 && lambda_sun < 180.0) {
        *RA = 90.0 - to_deg(asin(-N / sqrt(M*M + N*N)));
    }
    else {
        *RA = 270.0 + to_deg(asin(-N / sqrt(M*M + N*N)));
    }
    *Dec = to_deg(atan2(O, sqrt(M*M + N*N)));
}




// 带符号月相：-1.0 ~ +1.0
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

// 二分查找日出时间（升交点）
int32_t find_sunrise(int32_t year, int32_t month, int32_t day, double timezone, double longitude, double latitude) {
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

// 二分查找日落时间
int32_t find_sunset(int32_t year, int32_t month, int32_t day, double timezone, double longitude, double latitude) {
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

