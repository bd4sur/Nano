/*
 * 农历推算 - 基于天体位置计算的"第一原理"实现
 *
 * 依据 GB/T 33661-2017《农历的编算和表示》国家标准：
 * 1. 定朔：以朔日（日月黄经相等时刻）为月首
 * 2. 中气定月：以二十四节气中的中气确定月份
 * 3. 置闰：无中气之月为闰月
 *
 * 本实现不使用任何查找表，完全基于星历计算（朔日、中气时刻）
 * 直接调用 eph.c 中的天文计算函数
 *
 * 注意：本文件需以 UTF-8 编码保存，以支持中文字符串字面量。
 *
 * (c) BD4SUR 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ephemeris.h"
#include "nongli.h"

/* ============================================
 * 农历计算核心 - 基于星历的朔日和中气计算
 * ============================================ */

// 辅助函数：归一化角度到 [0, 360)
static double normalize_angle(double angle) {
    angle = fmod(angle, 360.0);
    if (angle < 0) angle += 360.0;
    return angle;
}

/* ---- 内部常量 ---- */

#define MAX_NEW_MOONS   40
#define MAX_MAJOR_TERMS 30

/* 十二中气对应的太阳黄经 */
static const double MAJOR_TERM_LONGITUDES[12] = {
    330,  /* 雨水 - 正月 */
    0,    /* 春分 - 二月 */
    30,   /* 谷雨 - 三月 */
    60,   /* 小满 - 四月 */
    90,   /* 夏至 - 五月 */
    120,  /* 大暑 - 六月 */
    150,  /* 处暑 - 七月 */
    180,  /* 秋分 - 八月 */
    210,  /* 霜降 - 九月 */
    240,  /* 小雪 - 十月 */
    270,  /* 冬至 - 十一月 */
    300   /* 大寒 - 腊月 */
};

static const char *TERM_NAMES[12] = {
    "雨水", "春分", "谷雨", "小满", "夏至", "大暑",
    "处暑", "秋分", "霜降", "小雪", "冬至", "大寒"
};

static const char *MONTH_NAMES[12] = {
    "正", "二", "三", "四", "五", "六",
    "七", "八", "九", "十", "冬", "腊"
};

static const char *CELESTIAL_STEMS[10] = {
    "甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"
};

static const char *TERRESTRIAL_BRANCHES[12] = {
    "子", "丑", "寅", "卯", "辰", "巳",
    "午", "未", "申", "酉", "戌", "亥"
};

static const char *ZODIAC_ANIMALS[12] = {
    "鼠", "牛", "虎", "兔", "龙", "蛇",
    "马", "羊", "猴", "鸡", "狗", "猪"
};

static const char *LUNAR_DAY_NAMES[30] = {
    "初一","初二","初三","初四","初五","初六","初七","初八","初九","初十",
    "十一","十二","十三","十四","十五","十六","十七","十八","十九","二十",
    "廿一","廿二","廿三","廿四","廿五","廿六","廿七","廿八","廿九","三十"
};

/* [修复 Bug 4] 五虎遁年起月法：甲/己年→丙(2)寅，乙/庚年→戊(4)寅，
 * 丙/辛年→庚(6)寅，丁/壬年→壬(8)寅，戊/癸年→甲(0)寅
 * 原 JS 代码为 [0,2,4,6,8,0,2,4,6,8]，差了两位，已修正如下 */
static const int MONTH_GAN_START[10] = {2, 4, 6, 8, 0, 2, 4, 6, 8, 0};


/* ---------------- 缓存机制 ---------------- */

/*
 * JS 版本使用闭包变量 cache 实现模块内缓存；
 * C 版本改用文件作用域静态结构体等效实现。
 */
static struct {
    int       year;                        /* 缓存所对应的公历年（lunar_year_start） */
    double    new_moons[MAX_NEW_MOONS];
    int       new_moons_count;
    MajorTerm major_terms[MAX_MAJOR_TERMS];
    int       major_terms_count;
    LunarDate last_lunar_date;
    int       has_last_lunar_date;
    char      last_check_day[32];          /* 格式："YYYY-M-D" */
} g_cache = { .year = -999999 };
/* 其余字段由 C 静态存储保证零初始化 */


/* ---- qsort 比较函数 ---- */

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static int cmp_major_term_jd(const void *a, const void *b) {
    const MajorTerm *ta = (const MajorTerm *)a;
    const MajorTerm *tb = (const MajorTerm *)b;
    return (ta->jd > tb->jd) - (ta->jd < tb->jd);
}


/* ---------------- 辅助函数 ---------------- */

static double julian_day_utc(int year, int month, int day, double hour) {
    return julian_day(year, month, day, (int)floor(hour), 0, 0, 0.0);
}

static DateTime jd_to_datetime(double jd) {
    double    jd1 = jd + 0.5;
    long long Z   = (long long)floor(jd1);
    double    F   = jd1 - (double)Z;

    long long A, alpha;
    if (Z < 2299161LL) {
        A = Z;
    } else {
        alpha = (long long)floor(((double)Z - 1867216.25) / 36524.25);
        A = Z + 1 + alpha - (long long)floor((double)alpha / 4.0);
    }

    long long B = A + 1524;
    long long C = (long long)floor(((double)B - 122.1) / 365.25);
    long long D = (long long)floor(365.25 * (double)C);
    long long E = (long long)floor(((double)B - (double)D) / 30.6001);

    DateTime dt;
    dt.day    = (int)((double)B - (double)D - floor(30.6001 * (double)E));
    dt.month  = (E < 14) ? (int)(E - 1) : (int)(E - 13);
    dt.year   = (dt.month > 2) ? (int)(C - 4716) : (int)(C - 4715);
    dt.hour   = (int)floor(F * 24.0);
    dt.minute = (int)floor((F * 24.0 - dt.hour) * 60.0);
    dt.second = (int)round(((F * 24.0 - dt.hour) * 60.0 - dt.minute) * 60.0);
    return dt;
}

/* [修复 Bug 2] 北京时间（CST, UTC+8）民用日编号
 * 同一北京时间自然日内返回相同整数，子夜（00:00 CST = 前一日 16:00 UTC）翻转
 * 公式：floor(JD_UT + 8/24 + 0.5)，即先转 CST，再按儒略日正午规则取整 */
static long long cst_day_number(double jd) {
    return (long long)floor(jd + 8.0 / 24.0 + 0.5);
}


/* ---------------- 星历函数（调用 eph.c）---------------- */

static double solar_longitude(double jd) {
    double lambda = 0.0;
    double beta = 0.0;
    double delta = 0.0;
    calculate_solar_ecliptic_coordinates(jd, &lambda, &beta, &delta);
    return normalize_angle(lambda);
}

static double lunar_longitude(double jd) {
    double lambda = 0.0;
    double beta = 0.0;
    double delta = 0.0;
    calculate_lunar_ecliptic_coordinates(jd, &lambda, &beta, &delta);
    return normalize_angle(lambda);
}

static double moon_sun_longitude_diff(double jd) {
    double sun_lon  = solar_longitude(jd);
    double moon_lon = lunar_longitude(jd);
    return normalize_angle(moon_lon - sun_lon);
}


/* ---------------- 迭代求解 ---------------- */

static double find_new_moon(double jd_guess) {
    double       jd       = jd_guess;
    const int    max_iter = 20;
    const double epsilon  = 1e-6;

    for (int i = 0; i < max_iter; i++) {
        double f = moon_sun_longitude_diff(jd);
        if (f > 180.0) f -= 360.0;

        double delta = 0.001;
        double f2    = moon_sun_longitude_diff(jd + delta / 24.0);
        if (f2 > 180.0) f2 -= 360.0;
        double fp = (f2 - f) / (delta / 24.0);

        if (fabs(fp) < 1e-10) break;

        double delta_jd = -f / fp;
        jd += delta_jd;
        if (fabs(delta_jd) < epsilon) break;
    }

    return jd;
}

static double find_major_term(double jd_guess, double target_longitude) {
    double       jd       = jd_guess;
    const int    max_iter = 20;
    const double epsilon  = 1e-6;

    for (int i = 0; i < max_iter; i++) {
        double f = normalize_angle(solar_longitude(jd) - target_longitude);
        if (f > 180.0) f -= 360.0;

        const double fp = 0.9856; /* 太阳黄经变化率 °/天 */
        double delta_jd = -f / fp;
        jd += delta_jd;
        if (fabs(delta_jd) < epsilon) break;
    }

    const double PERIOD = 365.25;
    while (jd - jd_guess >  PERIOD / 2.0) jd -= PERIOD;
    while (jd - jd_guess < -PERIOD / 2.0) jd += PERIOD;

    return jd;
}


/* ---------------- 农历推算 ---------------- */

static int find_new_moons_of_year(int year, double *out, int max_count) {
    const double SYNODIC_MONTH = 29.5306;

    double jd_start   = julian_day_utc(year,     1, 1, 0.0);
    double jd_end     = julian_day_utc(year + 1, 1, 1, 0.0);
    double current_jd = find_new_moon(jd_start - SYNODIC_MONTH);
    int    count      = 0;

    while (current_jd < jd_end + SYNODIC_MONTH) {
        if (current_jd >= jd_start - 1.0 && current_jd <= jd_end + 1.0) {
            if (count < max_count)
                out[count++] = current_jd;
        }
        current_jd = find_new_moon(current_jd + SYNODIC_MONTH * 0.8);
    }

    qsort(out, count, sizeof(double), cmp_double);
    return count;
}

/* [修复 Bug 1] 使用年初太阳黄经推算各中气的初始猜测时刻
 * 原代码用等间距偏移，导致大寒（index=11，约在年初1月）的猜测落在年末12月，
 * 牛顿迭代收敛到下一年1月的大寒，造成当年大寒丢失、腊月被误判为闰月。 */
static int find_major_terms_of_year(int year, MajorTerm *out, int max_count) {
    double jd_start = julian_day_utc(year,     1, 1, 0.0);
    double jd_end   = julian_day_utc(year + 1, 1, 1, 0.0);

    /* 计算年初（1月1日）时的太阳黄经，作为各中气初始猜测的基准 */
    double sun_lon_at_start = solar_longitude(jd_start);
    int    count = 0;

    for (int i = 0; i < 12; i++) {
        double target_lon = MAJOR_TERM_LONGITUDES[i];

        /* 从年初太阳黄经向前走多少度到达目标黄经（结果在 [0°, 360°)） */
        double diff = normalize_angle(target_lon - sun_lon_at_start);
        /* 换算为天数（太阳约 0.9856°/天），得到更准确的初始猜测 */
        double jd_guess = jd_start + diff / 0.9856;

        double jd_term = find_major_term(jd_guess, target_lon);

        if (jd_term >= jd_start - 30.0 && jd_term <= jd_end + 30.0) {
            if (count < max_count) {
                out[count].jd          = jd_term;
                out[count].longitude   = target_lon;
                out[count].name        = TERM_NAMES[i];
                out[count].month_index = i; /* 0=正月(雨水), …, 11=腊月(大寒) */
                count++;
            }
        }
    }

    qsort(out, count, sizeof(MajorTerm), cmp_major_term_jd);
    return count;
}


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
                            double timezone_offset)
{
    /* 检查缓存 */
    char current_day_key[32];
    snprintf(current_day_key, sizeof(current_day_key), "%d-%d-%d", year, month, day);
    if (strcmp(g_cache.last_check_day, current_day_key) == 0
            && g_cache.has_last_lunar_date) {
        return &g_cache.last_lunar_date;
    }
    strncpy(g_cache.last_check_day, current_day_key,
            sizeof(g_cache.last_check_day) - 1);
    g_cache.last_check_day[sizeof(g_cache.last_check_day) - 1] = '\0';

    /* 转换为 UT 儒略日 */
    double total_hour = hour + (minute + second / 60.0) / 60.0 - timezone_offset;
    double jd_now     = julian_day_utc(year, month, day, total_hour);

    /* [修复 Bug 2] 使用北京时间民用日编号，避免时刻精度带来的日期偏差
     * 农历月首（朔日）以北京时间所在的自然日为准，必须用 CST 日期比较，
     * 而非直接比较含时刻的儒略日数值。 */
    long long today_cst = cst_day_number(jd_now);

    /* 农历年大约从公历年初或前一年开始 */
    int lunar_year_start = (month < 2) ? year - 1 : year;

    /* 取缓存或重新计算朔日和中气 */
    double    new_moons[MAX_NEW_MOONS];
    int       new_moons_count;
    MajorTerm major_terms[MAX_MAJOR_TERMS];
    int       major_terms_count;

    if (g_cache.year == lunar_year_start
            && g_cache.new_moons_count   > 0
            && g_cache.major_terms_count > 0) {
        /* 命中缓存 */
        memcpy(new_moons,   g_cache.new_moons,
               g_cache.new_moons_count   * sizeof(double));
        new_moons_count   = g_cache.new_moons_count;
        memcpy(major_terms, g_cache.major_terms,
               g_cache.major_terms_count * sizeof(MajorTerm));
        major_terms_count = g_cache.major_terms_count;
    } else {
        double    nm1[MAX_NEW_MOONS],    nm2[MAX_NEW_MOONS];
        MajorTerm mt1[MAX_MAJOR_TERMS],  mt2[MAX_MAJOR_TERMS];

        int cnt_nm1 = find_new_moons_of_year(lunar_year_start,       nm1, MAX_NEW_MOONS);
        int cnt_mt1 = find_major_terms_of_year(lunar_year_start,     mt1, MAX_MAJOR_TERMS);

        /* 合并下一公历年的数据（覆盖农历年跨年的月份） */
        int cnt_nm2 = find_new_moons_of_year(lunar_year_start + 1,   nm2, MAX_NEW_MOONS);
        int cnt_mt2 = find_major_terms_of_year(lunar_year_start + 1, mt2, MAX_MAJOR_TERMS);

        /* 拼接并排序 new_moons */
        int avail_nm2   = (cnt_nm1 + cnt_nm2 > MAX_NEW_MOONS)
                          ? MAX_NEW_MOONS - cnt_nm1 : cnt_nm2;
        new_moons_count = cnt_nm1 + avail_nm2;
        memcpy(new_moons,           nm1, cnt_nm1   * sizeof(double));
        memcpy(new_moons + cnt_nm1, nm2, avail_nm2 * sizeof(double));
        qsort(new_moons, new_moons_count, sizeof(double), cmp_double);

        /* 拼接并排序 major_terms */
        int avail_mt2     = (cnt_mt1 + cnt_mt2 > MAX_MAJOR_TERMS)
                            ? MAX_MAJOR_TERMS - cnt_mt1 : cnt_mt2;
        major_terms_count = cnt_mt1 + avail_mt2;
        memcpy(major_terms,           mt1, cnt_mt1   * sizeof(MajorTerm));
        memcpy(major_terms + cnt_mt1, mt2, avail_mt2 * sizeof(MajorTerm));
        qsort(major_terms, major_terms_count, sizeof(MajorTerm), cmp_major_term_jd);

        /* 存入缓存 */
        g_cache.year = lunar_year_start;
        memcpy(g_cache.new_moons,   new_moons,
               new_moons_count   * sizeof(double));
        g_cache.new_moons_count = new_moons_count;
        memcpy(g_cache.major_terms, major_terms,
               major_terms_count * sizeof(MajorTerm));
        g_cache.major_terms_count = major_terms_count;
    }

    /* [修复 Bug 2] 用 CST 日编号查找所在朔望月，消除时刻偏差 */
    int    current_month_index = -1;
    double month_start_jd      = 0.0;
    double next_month_start_jd = 0.0;

    for (int i = 0; i < new_moons_count - 1; i++) {
        long long nm_cst      = cst_day_number(new_moons[i]);
        long long next_nm_cst = cst_day_number(new_moons[i + 1]);
        if (today_cst >= nm_cst && today_cst < next_nm_cst) {
            current_month_index = i;
            month_start_jd      = new_moons[i];
            next_month_start_jd = new_moons[i + 1];
            break;
        }
    }

    if (current_month_index < 0) return NULL;

    /* [修复 Bug 2] 农历日用 CST 日编号差值计算（初一=1） */
    int lunar_day = (int)(today_cst - cst_day_number(month_start_jd) + 1);

    /* [修复 Bug 2] 查找本朔望月内的中气，同样用 CST 日编号比较 */
    long long month_start_cst      = cst_day_number(month_start_jd);
    long long next_month_start_cst = cst_day_number(next_month_start_jd);

    MajorTerm terms_in_month[MAX_MAJOR_TERMS];
    int       terms_in_month_count = 0;
    for (int i = 0; i < major_terms_count; i++) {
        long long t_cst = cst_day_number(major_terms[i].jd);
        if (t_cst >= month_start_cst && t_cst < next_month_start_cst)
            terms_in_month[terms_in_month_count++] = major_terms[i];
    }

    /* 判断是否为闰月 */
    int        is_leap_month      = (terms_in_month_count == 0);
    int        lunar_month_num    = 0;
    const char *leap_month_marker = "";

    if (is_leap_month) {
        /* [修复 Bug 3] 闰月序号与其前面那个普通月相同（原代码多加了 1）
         * 例：大暑（六月）之后的无中气月 → 闰六月，而非闰七月 */
        int last_prev_idx = -1;
        for (int i = 0; i < major_terms_count; i++) {
            if (cst_day_number(major_terms[i].jd) < month_start_cst)
                last_prev_idx = i;
        }
        if (last_prev_idx >= 0)
            lunar_month_num = major_terms[last_prev_idx].month_index; /* 修正：不再 +1 */
        leap_month_marker = "闰";
    } else {
        lunar_month_num = terms_in_month[0].month_index;
    }

    /* 月份名称，如"正月"或"闰六月" */
    char lunar_month_name[32];
    snprintf(lunar_month_name, sizeof(lunar_month_name), "%s%s月",
             leap_month_marker, MONTH_NAMES[lunar_month_num]);

    /* 农历日名称 */
    int day_idx = lunar_day - 1;
    if (day_idx <  0) day_idx =  0;
    if (day_idx > 29) day_idx = 29;
    const char *lunar_day_name = LUNAR_DAY_NAMES[day_idx];

    /* [修复 Bug 5] 农历年判断：找出本公历年（lunar_year_start）的正月初一（含雨水的朔望月）
     * 原代码以冬至为年界，导致十月等月份在冬至前被误算为上一年。 */
    long long zhengyu_cst   = 0;
    int       found_zhengyu = 0;

    for (int i = 0; i < new_moons_count - 1 && !found_zhengyu; i++) {
        long long nm_cst      = cst_day_number(new_moons[i]);
        long long next_nm_cst = cst_day_number(new_moons[i + 1]);

        /* 含雨水（month_index == 0）即为正月 */
        for (int j = 0; j < major_terms_count; j++) {
            long long tc = cst_day_number(major_terms[j].jd);
            if (tc >= nm_cst && tc < next_nm_cst
                    && major_terms[j].month_index == 0) {
                /* 确认该正月初一落在公历 lunar_year_start 年（北京时间） */
                DateTime nm_dt = jd_to_datetime(new_moons[i] + 8.0 / 24.0);
                if (nm_dt.year == lunar_year_start) {
                    zhengyu_cst   = nm_cst;
                    found_zhengyu = 1;
                }
                break;
            }
        }
    }

    int lunar_year = lunar_year_start;
    if (found_zhengyu && today_cst < zhengyu_cst) {
        /* 当前日期在本公历年正月初一之前，属于上一农历年 */
        lunar_year = lunar_year_start - 1;
    }

    /* 干支纪年（1984年为甲子年） */
    int  year_gan = ((lunar_year - 1984) % 10 + 10) % 10;
    int  year_zhi = ((lunar_year - 1984) % 12 + 12) % 12;
    char year_ganzhi[16];
    snprintf(year_ganzhi, sizeof(year_ganzhi), "%s%s",
             CELESTIAL_STEMS[year_gan], TERRESTRIAL_BRANCHES[year_zhi]);
    const char *zodiac = ZODIAC_ANIMALS[year_zhi];

    /* [修复 Bug 4] 干支纪月 - 五虎遁年起月法，修正正月天干起始值
     * 甲/己年→丙(2)寅，乙/庚年→戊(4)寅，丙/辛年→庚(6)寅，
     * 丁/壬年→壬(8)寅，戊/癸年→甲(0)寅
     * 原代码为 [0,2,4,6,8,0,2,4,6,8]，差了两位，已修正如下 */
    int  month_gan = (MONTH_GAN_START[year_gan] + lunar_month_num) % 10;
    int  month_zhi = (lunar_month_num + 2) % 12; /* 正月=寅(2) */
    char month_ganzhi[16];
    snprintf(month_ganzhi, sizeof(month_ganzhi), "%s%s",
             CELESTIAL_STEMS[month_gan], TERRESTRIAL_BRANCHES[month_zhi]);

    /* 干支纪日（以1900年1月31日甲辰日为基准）
     * [修复 Bug 2] 同样使用 CST 日编号，与日期判断保持一致 */
    double    base_jd    = julian_day_utc(1900, 1, 31, 0.0);
    long long base_cst   = cst_day_number(base_jd);
    long long day_offset = today_cst - base_cst;
    int day_gan = (int)(((day_offset % 10) + 10) % 10);      /* 甲=0 */
    int day_zhi = (int)(((day_offset + 4) % 12 + 12) % 12);  /* 辰=4 */
    char day_ganzhi[16];
    snprintf(day_ganzhi, sizeof(day_ganzhi), "%s%s",
             CELESTIAL_STEMS[day_gan], TERRESTRIAL_BRANCHES[day_zhi]);

    /* 填写结果至缓存结构 */
    LunarDate *res = &g_cache.last_lunar_date;
    memset(res, 0, sizeof(LunarDate));
    res->year    = lunar_year;
    res->month   = lunar_month_num + 1; /* 1–12 */
    res->day     = lunar_day;
    res->is_leap = is_leap_month;

    snprintf(res->year_ganzhi, sizeof(res->year_ganzhi), "%s", year_ganzhi);
    snprintf(res->month_ganzhi, sizeof(res->month_ganzhi), "%s", month_ganzhi);
    snprintf(res->day_ganzhi, sizeof(res->day_ganzhi), "%s", day_ganzhi);
    snprintf(res->zodiac, sizeof(res->zodiac), "%s", zodiac);
    snprintf(res->month_name, sizeof(res->month_name), "%s", lunar_month_name);
    snprintf(res->day_name, sizeof(res->day_name), "%s", lunar_day_name);

    snprintf(res->full_display, sizeof(res->full_display),
             "%s%s年 %s%s", year_ganzhi, zodiac, lunar_month_name, lunar_day_name);

    g_cache.has_last_lunar_date = 1;
    return res;
}
